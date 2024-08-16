New uploads on arXiv(cs.CL)

### ScalingFilter: Assessing Data Quality through Inverse Utilization of Scaling Laws (https://arxiv.org/abs/2408.08310)
- **What's New**: 이 논문에서는 기존의 질 필터링 방법의 한계를 극복하기 위해 ScalingFilter라는 새로운 접근 방식을 제안합니다. 이 방법은 두 개의 언어 모델 간의 perplexity 차이를 기반으로 텍스트 품질을 평가하여 기준 데이터셋의 영향을 제거합니다.

- **Technical Details**: ScalingFilter는 모델의 크기와 데이터셋 품질 간의 관계를 활용하여 품질 요소(quality factor)를 정의합니다. 이는 데이터 샘플의 품질을 역으로 측정하는 방식으로, 두 개의 메타 모델(124M과 774M 파라미터)을 사용하여 raw 데이터셋 내 문서의 perplexity 차이를 평가합니다.

- **Performance Highlights**: ScalingFilter는 필터링된 데이터 품질을 개선하며, 이전의 질 필터링 방법에 비해 3.09%의 downstream accuracy 개선과 2.23 증가한 semantic diversity를 달성했습니다. 이로써 데이터 다양성과 성능 간의 최적 균형을 이룰 수 있음을 보여줍니다.



### The ShareLM Collection and Plugin: Contributing Human-Model Chats for the Benefit of the Community (https://arxiv.org/abs/2408.08291)
- **What's New**: ShareLM 컬렉션 소개: 모델과의 인간 간 대화 데이터 세트를 통합하고, 사용자가 자발적으로 대화를 기여할 수 있는 Chrome 플러그인을 발표합니다. 이 플러그인은 여러 플랫폼에서 대화를 수집할 수 있으며, 대화를 쉽고 투명하게 공유할 수 있도록 돕습니다.

- **Technical Details**: ShareLM 플러그인은 사용자가 모델과의 대화를 쉽게 기여하고 관리할 수 있도록 설계되었습니다. 이 플러그인은 대화의 내용을 수집하고, 사용자 피드백을 수집할 수 있는 기능을 제공합니다. 또한, 수집된 데이터는 REST API를 통해 서버에 업로드되며, 개인 식별 정보를 제거하는 익명화 기능이 포함되어 있습니다.

- **Performance Highlights**: 현재 ShareLM 컬렉션에는 2.3M 이상의 대화가 포함되어 있으며, 이는 4040개의 서로 다른 모델에서 수집된 것입니다. 플러그인을 통해 사용자는 자신의 대화를 평가하고, 삭제할 수 있는 권한을 가지며, 이는 커뮤니티에 기여할 수 있는 기회를 제공합니다.



### mhGPT: A Lightweight Generative Pre-Trained Transformer for Mental Health Text Analysis (https://arxiv.org/abs/2408.08261)
- **What's New**: 새로운 연구 결과로, mhGPT는 정신 건강 관련 데이터셋(주로 PubMed 아티클과 Reddit 게시물)을 기반으로 경량화된 트랜스포머 모델로서, 제한된 하드웨어 환경에서도 강력한 성능을 발휘합니다.

- **Technical Details**: mhGPT는 19억 8천만 개의 파라미터를 가지고 있으며, 5%의 데이터셋만 사용하여 훈련되었습니다. 이를 통해, 이전의 대규모 모델들이 필요로 하던 높은 컴퓨팅 자원 없이도 경쟁력 있는 성능을 발휘합니다. 주요 기여로는 정신 건강 관련 연구 데이터 통합, 커스텀 토크나이저 생성, 그리고 저자원 환경에 최적화된 소형 아키텍처가 있습니다.

- **Performance Highlights**: mhGPT는 MentaLLaMA 및 Gemma와 같은 최신 모델들을 초과하는 성능을 기록했으며, 특히 산후 우울증과 같은 특정 정신 건강 과제에 대해 보다 정확한 답변을 제공하는 것으로 나타났습니다.



### Covert Bias: The Severity of Social Views' Unalignment Towards Implicit and Explicit Opinion (https://arxiv.org/abs/2408.08212)
Comments:
          This work is under-review

- **What's New**: 최근 연구에서 편향(Bias) 식별을 위한 다양한 접근 방식이 논의되었지만, 직접적인 견해를 명시하지 않는 암묵적인 언어가 대형 언어 모델(LLM)의 편향 증폭(Bias Amplification)에는 어떤 영향을 미치는지에 대한 정보는 부족했습니다. 이 연구는 암묵적 및 명시적 사회 집단의 지식이 LLM 성능에 미치는 영향을 평가했습니다.

- **Technical Details**: 연구에서는 LLM이 상반된 관점을 갖는 암묵적 및 명시적 의견에 어떻게 응답하는지를 평가했습니다. 모델은 여성 및 종교 그룹에 대한 편향을 스트레스 테스트(Stress Testing)하여 심각도를 평가했습니다. 두 개의 다운스트림 작업인 혐오 발언 및 입장 탐지(Stance Detection)에서 모델 성능을 분석하였으며, LLM이 강한 동기 부여를 받아 보다 신중한 응답을 생성하는 경향을 보여주었습니다.

- **Performance Highlights**: 연구 결과, LLM은 명시적 의견을 선호하는 경향이 있으며, 편향이 맞춰진 모델은 불확실성 표현을 사용하는 더 신중한 응답을 제공합니다. 반면, 비편향 모델은 보다 직접적이고 신중하지 않은 응답을 생성하는 경향이 있어 사회적으로 민감한 주제에 대한 더 나은 신뢰성을 위해 불확실성 표기를 포함하는 것이 필요하다는 결론에 도달했습니다.



### DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search (https://arxiv.org/abs/2408.08152)
- **What's New**: DeepSeek-Prover-V1.5는 Lean 4에서 정리 증명을 위해 설계된 오픈소스 언어 모델로, DeepSeek-Prover-V1에 비해 훈련 및 추론 프로세스를 최적화하였습니다. 강화 학습 및 새로운 검색 알고리즘(RMaxTS)을 도입하여 증명 생성의 다양성을 확보했습니다.

- **Technical Details**: 모델은 DeepSeekMath-Base에서 미리 훈련되고, DeepSeek-Prover-V1에서 파생된 데이터셋을 사용하여 감독 학습을 통해 조정됩니다. Monte-Carlo tree search를 활용한 새로운 탐색 알고리즘 RMaxTS는 증명 경로의 다양성을 증대시키고, 빠른 정확성을 위해 저희는 truncate-and-resume 메커니즘을 사용합니다. 이 모델은 formal theorem proving의 기초와 자연어와의 정렬을 중시합니다.

- **Performance Highlights**: DeepSeek-Prover-V1.5는 miniF2F 벤치마크에서 63.5%의 통과율을 달성했으며, 높은 학교 수준에서의 새로운 최고 성능을 기록했습니다. ProofNet 기준에서도 뛰어난 성능을 보이며, 검증 세트에서 25.4%의 통과율을 기록했습니다.



### KOALA: Enhancing Speculative Decoding for LLM via Multi-Layer Draft Heads with Adversarial Learning (https://arxiv.org/abs/2408.08146)
- **What's New**: 이번 논문에서는 KOALA (K-layer Optimized Adversarial Learning Architecture)를 소개합니다. KOALA는 기존의 단일 레이어 드래프트 헤드를 다층 아키텍처로 변환하고, 전통적인 감독 학습(superevised training)에 적대적 학습(adversarial learning)을 통합하여, 다층 구조가 드래프트 헤드의 정확도를 크게 향상시키는 방법을 제안합니다.

- **Technical Details**: KOALA는 드래프트 헤드의 성능 격차를 해소하기 위해 다층 아키텍처를 도입하며, 이를 통해 드래프트 헤드가 목표 LLM (Large Language Models)의 기능을 보다 잘 반영할 수 있도록 개선합니다. 또한 기존 감독 학습 방식에 적대적 학습을 추가하여, 드래프트 헤드가 목표 LLM의 출력 분포와 일치하도록 보다 정교한 토큰 생성 과정을 캡처할 수 있도록 지원합니다. 이러한 접근 방식은 드래프트-그러면-검증(draft-then-verify) 사이클에서 생성되는 토큰 수를 증가시키고, 알고리즘 반복 횟수를 줄여서 효율성을 높입니다.

- **Performance Highlights**: KOALA는 다양한 작업에서 수행 평가를 통해 0.24x-0.41x의 지연 속도 개선 비율을 달성했습니다. 이는 기존 드래프트 헤드보다 10.57%-14.09% 빠른 결과입니다. 이를 통해 KOALA는 스펙ulative decoding의 효율성 및 정확성을 크게 향상시켰습니다.



### MIDAS: Multi-level Intent, Domain, And Slot Knowledge Distillation for Multi-turn NLU (https://arxiv.org/abs/2408.08144)
- **What's New**: 이 연구는 다중 턴 자연어 이해(NLU)를 향상시키기 위한 새로운 다중 수준, 다중 교사 지식 증류 모델인 MIDAS를 소개합니다. 기존의 NLU 모델은 주로 단일 턴 발화를 처리했으나, MIDAS는 복잡한 대화를 처리하기 위해 다중 수준의 대화 지식을 결합합니다.

- **Technical Details**: MIDAS는 세 가지 수준의 지식을 가진 교사 모델을 구축하여 학생 모델을 훈련합니다: 문장 수준의 의도 감지(Intent Detection), 단어 수준의 슬롯 채움(Slot Filling), 및 대화 수준의 도메인 분류(Domain Classification). 이러한 다중 교사 지식 증류는 각 턴의 발화를 통해 대화의 문맥을 이해하고 이전 정보를 유지하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, MIDAS 모델은 여러 NLU 데이터셋의 벤치마크에서 우수한 성능을 보여주며, LLM들과 비교해도 모든 의도 감지, 슬롯 채움 및 도메인 분류 작업에서 뛰어난 성과를 나타냅니다.



### AgentCourt: Simulating Court with Adversarial Evolvable Lawyer Agents (https://arxiv.org/abs/2408.08089)
- **What's New**: 이번 연구에서는 AgentCourt라는 시뮬레이션 시스템을 소개합니다. AgentCourt는 법정 절차를 전면적으로 시뮬레이션하며, 변호사 에이전트들이 사건을 논의하고 법적 기술을 강화하도록 돕습니다.

- **Technical Details**: AgentCourt는 여러 역할(법관, 원고 변호사, 피고 변호사 등)을 가진 자율 에이전트로 구성되어 있으며, 이는 Large Language Models (LLMs) 기반으로 작동합니다. 이 시스템은 대립적 진화 접근 방식을 통해 변호사 에이전트들이 경험을 쌓고 사례를 통해 지속적으로 학습토록 지원합니다.

- **Performance Highlights**: 연구 결과, AgentCourt에서 1,000건의 법적 사례를 처리한 후, 진화된 변호사 에이전트는 법적 작업 처리 능력이 일관되게 향상되었습니다. 법률 전문가 panel의 평가 결과, 이들은 응답성 및 전문성에서 주목할 만한 발전을 보였습니다.



### Extracting Sentence Embeddings from Pretrained Transformer Models (https://arxiv.org/abs/2408.08073)
- **What's New**: 이번 연구에서는 다양한 방법을 통해 BERT(Bidirectional Encoder Representations from Transformers) 모델의 문장 표현을 개선하는 방안을 제시합니다. 특히, 문장 레벨의 임베딩을 효과적으로 추출하는 기술을 평가하고, BERT 기반 모델의 성능을 향상시키기 위한 실험적인 방법들을 탐구합니다.

- **Technical Details**: 본 연구에서는 110백만 개의 파라미터를 가진 BERT의 숨겨진 표현을 여러 레이어와 토큰에서 추출하여 최적의 문장 표현을 얻기 위한 다양한 방법을 시도했습니다. 이를 위해 토큰 집계(token aggregation) 및 표현 후처리(post-processing) 기법들을 테스트했으며, 여러 짧은 텍스트 클러스터링 및 분류 작업에서 성능을 평가했습니다.

- **Performance Highlights**: 제안된 표현 추출 방법들은 모든 모델에서 STS(Semantic Textual Similarity) 및 클러스터링 작업의 성능을 개선시켰습니다. 특히, 정적 토큰 기반 모델의 성능이 매우 크게 향상되었고, 무작위 임베딩(random embeddings)의 STS 작업 성능이 BERT 기반 표현에 거의 도달했습니다.



### I-SHEEP: Self-Alignment of LLM from Scratch through an Iterative Self-Enhancement Paradigm (https://arxiv.org/abs/2408.08072)
- **What's New**: 이번 논문에서는 ВI-SHEEP, 즉 Iterative Self-Enhancement Paradigm을 소개합니다. 이 방식은 LLM(large language models)이 아무 것도 없이 스스로 지속적으로 훈련할 수 있도록 지원하는 인간 유사 학습 패러다임입니다.

- **Technical Details**: I-SHEEP는 최초 데이터(seed data)와 LLM의 강력한 이해 및 생성 능력을 활용하여 추가적인 지시-출력 쌍 데이터를 생성합니다. 그런 다음 모델이 자신의 학습 과정 모니터링 및 평가를 수행하게 하여 부정확한 인식을 필터링하고 올바른 인식만을 유지함으로써 자체 정렬(self-align)을 진행합니다. 이 과정이 반복되면서 모델은 내부 지식만으로 지속적이고 자율적으로 스스로 정렬합니다.

- **Performance Highlights**: I-SHEEP는 Qwen-1.5 72B 모델에서 Alpaca Eval에서 최대 78.2% 상대 개선, MT Bench에서 24.0% 개선, IFEval의 정확도에서 8.88%의 절대적 증가를 달성했습니다. 또한 코드 생성 과제에서 평균 24.77%, TrivialQA에서 12.04%, SQuAD에서 20.29%의 평균 개선을 이루었습니다.



### RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation (https://arxiv.org/abs/2408.08067)
Comments:
          Under Review

- **What's New**: 이 논문에서는 RAG 시스템을 평가하기 위한 새로운 프레임워크인 RAGChecker를 제안합니다. 이 프레임워크는 리트리버와 제너레이터 모듈 각각에 대한 상세한 진단 지표를 포함하여 RAG 시스템의 성능을 포괄적으로 평가할 수 있도록 설계되었습니다.

- **Technical Details**: RAGChecker는 클레임 수준의 함의 검사를 기반으로 하여 응답 및 정답과 관련된 클레임을 추출하고 이를 다른 텍스트와 비교하여 미세한 평가를 수행합니다. 이 방식은 RAG 시스템의 리트리버 및 제너레이터를 모두 평가할 수 있도록 해줍니다. RAGChecker의 지표에는 전체 성능 지표, 리트리버의 진단 지표, 제너레이터의 진단 지표가 포함됩니다.

- **Performance Highlights**: RAGChecker는 8개의 최신 RAG 시스템을 평가하였고, 그 결과는 RAG 아키텍처의 디자인 선택에서의 통찰력 있는 패턴과 상충점을 드러냈습니다. 메타 평가를 통해 RAGChecker는 기존의 평가 지표보다 인간의 판단과의 상관관계가 훨씬 더 우수함을 입증했습니다.



### Leveraging Web-Crawled Data for High-Quality Fine-Tuning (https://arxiv.org/abs/2408.08003)
- **What's New**: 이 논문에서는 고비용의 인간 주석 데이터나 GPT-4 생성 데이터를 사용하지 않고도 웹 크롤링 데이터(Web-Crawled Data)를 활용한 고품질 지도 학습(Supervised Fine-Tuning)의 가능성을 보여줍니다.

- **Technical Details**: 논문에서는 웹 크롤링 데이터를 고품질 데이터와 정렬하여 자동으로 쌍을 이루는 훈련 데이터셋을 생성합니다. 이를 통해 비정형 웹 데이터를 고품질로 변환하여 언어 모델을 훈련시키는 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, 모델 변환 데이터를 사용하여 훈련한 경우 중국 수학 문제에서 고품질 데이터만을 사용했을 때보다 평균 9.4% 향상된 성과를 보였습니다. 또한 7B 모델은 32B 이상의 여러 오픈소스 모델과 유명한 닫힌 소스 모델인 GPT-3.5를 초월하는 성능을 보여줍니다.



### FuseChat: Knowledge Fusion of Chat Models (https://arxiv.org/abs/2408.07990)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 오픈 소스 챗 LLM(large language models)들을 활용하여 FuseChat이라는 새로운 지식 융합 프레임워크를 제안합니다. FuseChat은 두 단계로 구성되어 있으며, 다양한 아키텍처와 규모의 챗 LLM들을 통합하여 더 강력한 LLM을 생성하려고 합니다.

- **Technical Details**: FuseChat 프레임워크는 첫 번째로는 페어와이즈(별도 모델 동기화 없이) 지식 융합을 통해 여러 개의 대상 LLM을 생성하고, 두 번째로는 신규 프레임워크인SCE를 통해 매개변수 공간 내에서 이 대상 LLM들을 통합합니다. SCE는 미세 조정 전후의 매개변수 업데이트의 크기에 따라 융합 계수를 결정합니다. 이를 통해 추가적인 교육 없이 미세한 그레인 수준에서 매개변수를 병합할 수 있게 합니다.

- **Performance Highlights**: FuseChat-7B는 두 개의 기준 벤치마크인 AlpacaEval 2.0 및 MT-Bench에서 실험하여 다양한 크기에서의 여러 LLM에게 우수한 성능을 보여주었으며, Mixtral-8x7B-Instruct 및 GPT-3.5-Turbo-1106와 비슷한 성능을 보였습니다. 또한, 이 모델은 공개된 코드 및 모델 웨이트를 통해 재현 가능성을 갖추고 있습니다.



### ArabLegalEval: A Multitask Benchmark for Assessing Arabic Legal Knowledge in Large Language Models (https://arxiv.org/abs/2408.07983)
- **What's New**: 이 논문에서는 아랍어 법률 지식을 평가하기 위한 멀티태스킹 벤치마크 데이터셋인 ArabLegalEval을 도입합니다. 이는 LLM(대형 언어 모델)의 법률 관련 문제 해결 능력을 분석하고 성능을 벤치마킹하는 것을 목표로 합니다.

- **Technical Details**: ArabLegalEval은 사우디 법률 문서와 합성 질문을 기반으로 여러 과제를 포함하며, LLM의 법률 추론 능력을 평가하는 데 중점을 두고 있습니다. 이 데이터셋은 법률 전문가와 상담을 통해 개발되었으며, 영어 법률 벤치마크(예: LegalBench)에서 번역된 과제도 포함되어 있습니다. 또한, Retrieval-Augmented Generation (RAG) 시스템을 통한 정보 검색 방법을 연구하고 있습니다.

- **Performance Highlights**: GPT-4와 Jais 모델을 포함한 다국어 및 아랍어 중심 LLM의 성능을 벤치마킹하여, 아랍어 법률 도메인에서의 LLM의 현재 상태를 밝혀내고자 합니다. 초기 평가 결과, GPT-4는 아랍어MMLU 모든 섹션에서 뛰어난 성능을 보였으며, 다른 모델들을 초월하는 성과를 보였습니다.



### Predicting Lung Cancer Patient Prognosis with Large Language Models (https://arxiv.org/abs/2408.07971)
- **What's New**: 이번 연구에서는 GPT-4o mini와 GPT-3.5를 활용하여 폐암 환자의 예후 예측 가능성을 평가했습니다. 전통적인 로지스틱 회귀 모델과 비교했을 때, LLMs는 데이터 사용 없이도 경쟁력 있는 예후 예측 성능을 달성했습니다.

- **Technical Details**: 연구에서는 2010-2018년 폐암 환자의 전산화 단층촬영(CT) 보고서 총 847개를 수집하고, 정보를 추출하기 위해 명명된 개체 인식(NER), 관계 추출 등의 작업을 정의했습니다. ChatGPT를 이용한 IE(정보 추출) 시스템은 문의 각 질문에 대한 답변을 작성하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, ChatGPT는 로지스틱 회귀 모델에 비해 폐암 예후 예측에서 유사하거나 우수한 성능을 보였습니다. 이러한 결과는 특히 환자 데이터가 제한적일 때 LLMs가 효과적인 도구가 될 수 있음을 시사합니다.



### GERestaurant: A German Dataset of Annotated Restaurant Reviews for Aspect-Based Sentiment Analysis (https://arxiv.org/abs/2408.07955)
Comments:
          Accepted in KONVENS 2024. Camera Ready submission

- **What's New**: GERestaurant는 3,078개의 독일어 음식점 리뷰 데이터셋으로, Aspect-Based Sentiment Analysis (ABSA)를 위해 수작업으로 주석이 달린 첫 번째 독일어 데이터셋입니다. 이 데이터셋은 명시적 및 암시적 측면을 포함하여 리뷰에서 표현된 감정과 관련된 모든 측면 용어와 그에 해당하는 감정 범주로 구성되어 있습니다.

- **Technical Details**: 데이터셋은 Tripadvisor에서 수집된 리뷰로 구성되며, Aspect Category Detection (ACD), Aspect Category Sentiment Analysis (ACSA), End-to-End ABSA (E2E-ABSA), Target Aspect Sentiment Detection (TASD) 등 4가지 ABSA 작업을 위한 기준 성과를 제공합니다. 리뷰는 2022년 10월 15일부터 2023년 10월 15일까지의 기간 중 위생 규정이 없는 시기에 수집되었습니다.

- **Performance Highlights**: 가장 최신의 transformer 기반 언어 모델을 이용한 미세 조정으로, 각 ABSA 작업에 대해 기초 성능을 제시하며, 독일어 리뷰 도메인에서 ABSA의 새로운 연구 방향을 탐색할 수 있는 기회를 제공합니다.



### MAG-SQL: Multi-Agent Generative Approach with Soft Schema Linking and Iterative Sub-SQL Refinement for Text-to-SQL (https://arxiv.org/abs/2408.07930)
Comments:
          22 pages, 14 figures

- **What's New**: 최근의 In-Context Learning (ICL) 기반 방법이 Text-to-SQL 작업에서 상당한 성공을 거둔 반면, 복잡한 데이터베이스 스키마와 어려운 질문을 가진 데이터셋인 BIRD에서 인간 성능과 여전히 큰 격차가 존재합니다. 이러한 문제를 해결하기 위해 MAG-SQL이라는 다중 에이전트 생성 방식을 제안합니다.

- **Technical Details**: MAG-SQL은 Soft Schema Linking과 반복적 Sub-SQL 정제를 포함한 다중 에이전트 생성 접근 방식을 채택합니다. 이 프레임워크에서는 엔티티 기반 방법을 사용해 데이터베이스의 컬럼을 선택하고, 복잡한 질문을 분해하기 위한 타겟-조건 분해(Targets-Conditions Decomposition) 방법을 도입합니다. Sub-SQL Generator와 Sub-SQL Refiner를 포함하는 반복 생성 모듈을 구축하여, 생성 과정 중 각 단계에 외부 감시를 도입합니다.

- **Performance Highlights**: BIRD 벤치마크에서 GPT-4를 사용하여 평가한 결과, MAG-SQL은 61.08%의 실행 정확도를 달성하였고, 이는 기본 GPT-4의 46.35% 및 MAC-SQL의 57.56%와 비교하여 우수한 성능을 보여줍니다. 이 접근법은 Spider 데이터셋에서도 유사한 향상을 보였습니다.



### Assessing Language Models' Worldview for Fiction Generation (https://arxiv.org/abs/2408.07904)
Comments:
          Short paper

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 픽션 생성 능력을 평가하며, 특히 세계관을 일관되게 유지하는 능력에 초점을 맞춥니다. 9개의 LLM을 통해 조사한 결과, 오직 2개 모델만이 일관된 세계관을 보이는 반면, 나머지 모델은 자기 모순적인 답변을 나타냄을 발견했습니다.

- **Technical Details**: 우리는 885개의 문장으로 구성된 데이터셋을 사용하여 LLM들이 픽션 생성에 필요한 진리와 허구를 구분할 수 있는지를 평가했습니다. 고정된 세계 상태를 유지할 수 있는지를 평가하는 일련의 질문(P0-P4)을 통해 모델들의 일관성과 강인성을 분석했습니다.

- **Performance Highlights**: 현재 LLM들은 픽션 작성을 위한 일관된 세계 상태를 유지하는 데 한계를 보이며, 특히 고정 관념이나 논란 있는 주제에 대해 일관性 없는 답변을 보여줍니다. Mistral-7B와 같은 여러 모델이 일관성 부족을 드러내며, 이는 픽션 작성에 있어 신뢰성을 떨어뜨립니다.



### Fine-tuning Large Language Models with Human-inspired Learning Strategies in Medical Question Answering (https://arxiv.org/abs/2408.07888)
- **What's New**: 이 논문은 이론적으로 학습 접근 방식인 커리큘럼 학습(curriculum learning)과 비커리큘럼 학습(non-curriculum-based learning)을 다양한 대규모 언어 모델(LLMs)에 적용하여 의료 질문 답변(medical question answering) 데이터 세트에서 평가한 새로운 연구입니다. 기존 연구보다 더 넓은 평가를 통해 인간의 학습 방식을 모방한 효과적인 데이터 정렬 전략을 탐색하였습니다.

- **Technical Details**: 논문에서는 4개의 LLM(모델)과 이들을 대상으로 한 5가지 인간 영감을 받은 학습 방식을 비교하여, 각 모델과 데이터 세트 조합에 따라 커리큘럼 학습의 효과가 다르다는 것을 입증했습니다. 특히, LLM이 정의한 질문 난이도가 인간이 정의한 난이도보다 나은 성과를 보인다는 점을 강조했습니다.

- **Performance Highlights**: 본 연구의 결과, LLM을 커리큘럼 학습을 통해 미세 조정할 때 최대 1.77%의 정확도 향상을 모델당, 그리고 1.81% 향상을 데이터 세트당 기록했습니다. 그러나 각 모델과 데이터 세트 조합에 따라 학습 전략의 효과가 크게 달라진다는 점에서 주의가 필요하다고 강조했습니다.



### Instruct Large Language Models to Generate Scientific Literature Survey Step by Step (https://arxiv.org/abs/2408.07884)
Comments:
          NLPCC 2024

- **What's New**: 이 논문에서는 과학 문헌 조사를 자동으로 생성하기 위한 새로운 접근 방식을 제안합니다. 특히, 대형 언어 모델(LLMs)을 활용하여 제목, 초록, 계층적 제목 및 주요 내용을 단계별로 생성할 수 있도록 돕는 지침형 프로프트를 설계하였습니다.

- **Technical Details**: 제안된 방법은 주어진 주제 및 참고 문서를 바탕으로 문헌 조사를 구조적으로 생성합니다. 전체 프로세스는 두 단계로 나뉘며, 첫 번째 단계에서는 LLM이 제목, 섹션 제목, 초록을 순차적으로 생성합니다. 두 번째 단계에서는 생선된 제목에 따라 내용 생성이 이루어지며, 이 과정에서 LLM의 API 비용을 줄일 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 시스템은 NLPCC 2024 문헌 조사 생성 평가에서 3위를 차지하였으며, 전체 점수는 61.11로 2위 팀과 0.03% 차이에 불과합니다. 이 방법은 소프트 제목 회수율이 95.84%로, 제출된 팀 중 두 번째로 높은 성능을 보였습니다. 각 문헌 조사 생성 비용은 단 0.1 RMB로 줄어들어 실용적인 가치를 높였습니다.



### Words Matter: Reducing Stigma in Online Conversations about Substance Use with Large Language Models (https://arxiv.org/abs/2408.07873)
- **What's New**: 이 연구는 물질 사용 장애(SUD)로 고통받는 개인들에 대한 사회적 낙인이 치료 접근에 미치는 영향을 분석하고, Reddit와 같은 소셜 미디어에서의 낙인을 줄이기 위한 기술적 접근법을 제안합니다. 이 연구는 낙인 언어를 공감적인 언어로 변환하는 모델을 개발하며, 전체 1.2백만 개의 게시물 중 3,207개에서 낙인 언어를 확인했습니다.

- **Technical Details**: 대형 언어 모델(LLM) 및 정보기반 스타일화 모델(Informed + Stylized LLMs)을 활용하여 낙인 언어를 공감적으로 변화시키는 프레임워크를 개발했습니다. 주요 분석 결과, 자극제와 대마초가 가장 자주 언급되었으며, 낙인은 대인 관계 및 도덕적 판단과 연관성이 높았습니다. 연구에서는 1,649개의 낙인 완화 문구 쌍을 생성했습니다.

- **Performance Highlights**: 인간 평가 결과, GPT-4를 사용한 Informed + Stylized 시스템이 원래의 톤과 관련성을 유지하면서도 효과적으로 낙인을 줄일 수 있음을 보여주었습니다. 자동 평가 또한 이 접근법이 원본 게시물의 스타일 및 심리언어학적 특성을 유지하면서 낙인을 줄이는 데 효과적임을 확인했습니다.



### Training Language Models on the Knowledge Graph: Insights on Hallucinations and Their Detectability (https://arxiv.org/abs/2408.07852)
Comments:
          Published at COLM 2024. 16 pages, 11 figures

- **What's New**: 본 연구는 Knowledge Graph (KG)을 기반으로 한 데이터셋을 활용하여 언어 모델(LM)의 환각(hallucination) 현상을 분석합니다. LM의 크기와 훈련 기간에 따른 환각의 발생 빈도 변화를 조사하여, 모델의 성능 향상에 필요한 비용을 정리합니다.

- **Technical Details**: 저자들은 3.15M에서 1.61B개의 비 embedding 파라미터를 가진 Transformer LM을 KG 데이터셋을 기반으로 훈련하였고, 각 LM의 출력에서 환각을 탐지하고 수정하는 방법을 연구했습니다. 환각을 정확히 정의하기 어려운 자연어 환경에서, KG는 LM의 훈련 시 정보에 대한 완전한 제어를 가능하게 하며, 주어-서술어-목적어 3항 데이터 구조를 사용했습니다.

- **Performance Highlights**: LM의 크기가 커지고 훈련기간이 길어질수록 환각 현상이 줄어드는 경향이 있지만, 훈련 데이터 환각률을 5% 이하로 낮추기 위해서는 훨씬 더 큰 모델과 계산 비용이 필요합니다. 반면 LM의 크기가 커질수록 환각 탐지기의 성능은 개선되지만, LM의 환각 탐지 가능성과는 역관계에 있다는 점이 발견되었습니다.



### SER Evals: In-domain and Out-of-domain Benchmarking for Speech Emotion Recognition (https://arxiv.org/abs/2408.07851)
Comments:
          Accepted at INTERSPEECH 2024

- **What's New**: 이번 논문은 언어 및 감정 표현의 다양성에 대한 모델의 일반화 가능성을 평가하기 위한 대규모 Speech Emotion Recognition (SER) 벤치마크를 제안합니다. 이 벤치마크는 다국어 데이터 세트를 포함하여, 적은 사용 빈도를 가진 데이터 세트에 중점을 두어 새로운 데이터에 대한 일반화를 평가합니다.

- **Technical Details**: 본 연구는 Whisper, CLAP과 같은 최신 Speech Representation 모델을 활용하여 cross-lingual SER 성능을 분석합니다. 또한, logit adjustment를 통해 다양한 클래스 분포를 고려하여 공정한 비교를 보장합니다. 평가 프로토콜을 통해 in-domain 및 out-of-domain 성능을 분석하여 모델의 적응력과 일반화 능력을 평가합니다.

- **Performance Highlights**: 놀랍게도, Whisper 모델은 자동 음성 인식(Automatic Speech Recognition) 용도로 설계되었음에도 불구하고 cross-lingual SER에서 전용 SSL 모델보다 뛰어난 성능을 보였습니다. 이러한 결과는 개발할 SER 모델이 더 강력하고 일반화 가능성이 있어야 함을 강조합니다.



### ONSEP: A Novel Online Neural-Symbolic Framework for Event Prediction Based on Large Language Mod (https://arxiv.org/abs/2408.07840)
Comments:
          16 pages, ACL 2024 Findings

- **What's New**: 본 논문에서는 온라인 신경-상징적 이벤트 예측(ONSEP) 프레임워크를 소개하며, 이는 동적 인과 규칙 마이닝(DCRM)과 이중 이력 증강 생성(DHAG)을 통합하여 더 나은 이벤트 예측을 가능하게 한다.

- **Technical Details**: DCRM은 실시간 데이터를 기반으로 동적으로 인과 규칙을 구성하여 새로운 인과 관계에 신속하게 적응할 수 있도록 하며, DHAG는 단기 및 장기 이력 컨텍스트를 통합하여 이벤트 예측을 풍부하게 만든다.

- **Performance Highlights**: ONSEP 프레임워크는 다양한 데이터 세트에서 Hit@k (k=1,3,10)의 성능 개선을 보여주며, 대규모 언어 모델(LLMs)을 위한 이벤트 예측을 확장하는 능력을 증명한다.



### Language Driven Slice Discovery and Error Rectification (https://arxiv.org/abs/2408.07832)
- **What's New**: 이 논문은 기존의 오류 슬라이스 발견 방법을 탈피하여, Large Language Model (LLM)의 추론 능력을 활용하여 복잡한 오류 패턴을 분석하고 테스트 가능한 가설을 생성하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법인 LADDER (Language Driven slice Discovery and Error Rectification)는 모델의 표현을 언어 정렬 피쳐 공간 (feature space)(예: CLIP)에 투영하여 원본 모델 피쳐 공간의 의미를 보존합니다. 이 과정에서 모델의 오류를 강조하는 문장을 정확하게 검색할 수 있습니다. 이후 LLM은 이러한 문장을 사용하여 오류 슬라이스를 발견하기 위한 가설을 생성하고, 마지막으로 가설을 통해 생성된 그룹 균형 데이터셋을 이용하여 분류 헤드를 미세 조정 (fine-tuning)함으로써 오류를 완화합니다.

- **Performance Highlights**: 모든 과정에서 속성 주석 (attribute annotation)이 필요하지 않으며, 다섯 개의 이미지 분류 데이터셋을 통해 방법이 검증되었습니다. 제공된 코드는 연구 결과의 재현성을 높이는 데 기여합니다.



### Can Large Language Models Understand Symbolic Graphics Programs? (https://arxiv.org/abs/2408.08313)
Comments:
          Technical Report v1 (44 pages, 23 figures, project page: this https URL)

- **What's New**: 이 연구는 LLMs(대형 언어 모델)가 상징적 그래픽 프로그램(symbolic graphics programs)을 이해하는 능력을 평가하는 새로운 기준을 설정하며, 이를 통해 LLM의 시각적 장면에 대한 추론 능력을 평가합니다.

- **Technical Details**: 연구팀은 SGP-Bench라는 새로운 벤치마크를 구축하며, 이는 데이터에 대한 최소한의 인간 노력을 요구하는 프로그램-그래픽 대응(program-graphics correspondence)에 기반합니다. 또한, Symbolic Instruction Tuning (SIT)를 통해 LLM의 지식 이해 능력을 향상시키는 방법을 제안합니다.

- **Performance Highlights**: 현재의 LLM은 벤치마크에서 다양한 성과를 보여주며, 특히 강력한 추론 능력을 가진 모델이 더 나은 성과를 나타냈습니다.



### Benchmarking the Capabilities of Large Language Models in Transportation System Engineering: Accuracy, Consistency, and Reasoning Behaviors (https://arxiv.org/abs/2408.08302)
- **What's New**: 이 논문에서는 최신의 대형 언어 모델(LLMs)인 GPT-4, GPT-4o, Claude 3.5 Sonnet, Claude 3 Opus, Gemini 1.5 Pro, Llama 3, Llama 3.1이 특정 대학 수준의 교통 공학 문제를 해결하는 능력을 탐구합니다.

- **Technical Details**: TransportBench라는 벤치마크 데이터셋을 소개합니다. 이 데이터셋에는 교통 시스템의 계획, 설계, 관리 및 제어와 관련된 다양한 주제의 교통 공학 문제 샘플이 포함되어 있습니다. 이 데이터셋은 인간 전문가가 다양한 상업용 및 오픈소스 LLM의 능력을 평가하기 위해 사용됩니다.

- **Performance Highlights**: 각 LLM의 독특한 강점과 제한 사항을 발견했습니다. 예를 들어, TransportBench 문제를 해결하는 데 있어 Claude 3.5 Sonnet의 인상적인 정확성과 예상치 못한 불일치 행동을 보여주었습니다.



### P/D-Serve: Serving Disaggregated Large Language Model at Sca (https://arxiv.org/abs/2408.08147)
- **What's New**: 이 논문은 분산 대형 언어 모델(LLMs)의 신뢰할 수 있는 성능을 확보하기 위한 새로운 P/D-Serve 시스템을 제안합니다. 이 시스템은 MLOps의 패러다임을 따르며, P/D 성능을 모델링하고 세 가지 주요 최적화를 통해 서비스의 효과성을 높입니다.

- **Technical Details**: P/D-Serve는 동적인 P/D 비율 조정, 요청 거부 시 대기 중인 prefill 인스턴스에 요청을 전달하는 on-demand forwarding 그리고 연속 버퍼를 사용하여 D2D KVCache 전송을 최적화하는 기능을 포함합니다. 이를 통해 다양한 워크로드에 따라 서비스 전반을 최적화할 수 있습니다.

- **Performance Highlights**: P/D-Serve는 Ascend와 MindSpore에서 구현되었으며, 상용 환경에서 80개월 이상 운영되었습니다. 전체 E2E 처리량(throughput)에서 60%, 첫 번째 토큰까지의 대기 시간(TTFT SLO)에서 42%, D2D 전송 시간에서 46% 개선을 달성했습니다. 특히, 집계된 LLM에 비해 처리량이 6.7배 증가했습니다.



### Text2BIM: Generating Building Models Using a Large Language Model-based Multi-Agent Framework (https://arxiv.org/abs/2408.08054)
- **What's New**: 론문에서는 Text2BIM이라는 LLM 기반의 멀티 에이전트 프레임워크를 제안하여 자연어 지침을 통해 3D 건물 모델을 생성할 수 있게 하였습니다. 기존의 복잡한 모델링 명령을 숙지할 필요 없이, 텍스트 입력을 바탕으로 BIM 모델을 보다 직관적으로 표현할 수 있는 방법입니다.

- **Technical Details**: Text2BIM 프레임워크는 여러 LLM 에이전트가 협력하여 사용자 입력을 수동 코드로 변환하며, BIM 저널링 툴의 API를 호출하여 수정 가능한 BIM 모델을 생성합니다. 또한, 규칙 기반 모델 체커를 도입하여 생성된 모델의 품질을 향상시키고, 다중 피드백 루프를 통해 반복적으로 모델을 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 사용자 입력에 명시된 추상 개념과 잘 정렬된 높은 품질의 구조적으로 합리적인 건물 모델을 효율적으로 생성할 수 있음을 보여주었습니다. 또한, 이 프레임워크를 Vectorworks라는 BIM 저자 툴에 통합한 인터랙티브 소프트웨어 프로토타입을 개발하여 새로운 모델링 가능성을 시연하였습니다.



### Enhancing Large Language Model-based Speech Recognition by Contextualization for Rare and Ambiguous Words (https://arxiv.org/abs/2408.08027)
Comments:
          13 pages, 1 figure, and 7 tables

- **What's New**: 본 논문에서는 키워드를 텍스트 프롬프트로 제공하여 맥락화를 할 수 있는 대용량 언어 모델(LLM) 기반의 자동 음성 인식(ASR) 시스템을 개발하였습니다. 우리 시스템은 디코더 전용 아키텍처를 채택하고 있으며, 일본어 및 영어 데이터셋을 기반으로 처음부터 훈련된 PLaMo-100B LLM을 사용하여 디코더로 활용합니다.

- **Technical Details**: 오디오 인코더로는 사전 훈련된 Whisper encoder를 사용하고, 오디오 인코더로부터 생성된 오디오 임베딩은 어댑터 레이어를 통해 텍스트 임베딩 공간으로 변환되며, 텍스트 프롬프트로부터 생성된 텍스트 임베딩과 결합되어 디코더의 입력을 형성합니다. 키워드 제공을 통해 LLM 기반 ASR 시스템의 성능을 개선하고, 희귀하고 모호한 단어에 대한 인식 성능을(Recognition Performance) 높일 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, 디코더에 키워드를 제공하는 것이 희귀하고 모호한 단어의 인식 성능을 상당히 향상시키는 것으로 나타났습니다. 이를 통해 모델 아키텍처를 변경하지 않고도 보다 정확한 음성 인식을 달성할 수 있습니다.



### Coupling without Communication and Drafter-Invariant Speculative Decoding (https://arxiv.org/abs/2408.07978)
Comments:
          16 pages

- **What's New**: 이번 논문에서는 통신 없이 두 분포에서 샘플을 생성하는 'communication-free coupling' 문제를 다룹니다. Alice와 Bob은 각각 두 개의 분포 P와 Q를 가지고, 최대한 높은 확률로 동일한 값을 얻는 방법을 탐구합니다.

- **Technical Details**: Alice와 Bob은 각각 분포 P에서 a를 샘플링하고, Q에서 b를 샘플링하며, a=b가 되는 확률을 최대화합니다. 이때, 통신이 불가능하지만 공용 난수를 이용하여 $Pr[a=b] \geq \frac{1 - D_{TV}(P,Q)}{1 + D_{TV}(P,Q)} \geq 1 - 2D_{TV}(P,Q)$ 를 달성할 수 있습니다. Weighted MinHash 알고리즘을 기반으로 한 간단한 프로토콜을 통해 이 경계를 얻을 수 있습니다.

- **Performance Highlights**: 이 연구에서는 Gumbel 샘플링 기반의 또 다른 간단한 프로토콜이 Weighted MinHash 접근 방식의 최악의 경우 보장을 유지하면서도 실제로 더 나은 성능을 나타낸다고 증명하였습니다. 또한, $O(log(n/\epsilon))$ 비트를 사용하여 $Pr[a=b] = 1 - D_{TV}(P,Q) - \epsilon$를 달성하는 방안도 제시하였습니다. 마지막으로, 통신 없는 coupling의 응용으로 최근의 autoregressive large language model 가속 방법인 speculative decoding에 대한 적용을 보여주었습니다.



### Polaris: Open-ended Interactive Robotic Manipulation via Syn2Real Visual Grounding and Large Language Models (https://arxiv.org/abs/2408.07975)
Comments:
          Accepted by IROS 2024. 8 pages, 5 figures. See this https URL

- **What's New**: 이 논문에서는 테이블탑(Tairo-top) 환경에서의 오픈 엔디드(interactor) 로봇 조작 과제를 다룹니다. 기존 대형 언어 모델(LLM)의 한계를 극복하기 위해 시각적 기초(visual grounding)를 강화한 상호작용 로봇 조작 프레임워크인 Polaris를 도입하였습니다.

- **Technical Details**: Polaris 프레임워크는 GPT-4와 함께 구조적 시각 모델을 사용하여 로봇의 환경을 이해하고, 목표 물체 위치를 정밀하게 추정하는 데 필요한 단계적 방식인 Synthtic-to-Real (Syn2Real) 포즈 추정 파이프라인을 포함합니다. 이 파이프라인은 렌더링된 합성 데이터를 이용해 훈련하고 이를 실제 조작 작업으로 전이합니다.

- **Performance Highlights**: 실제 로봇 실험을 통해 Polaris의 우수한 성능을 입증하였으며, 다양한 조작 작업에 대한 성공률이 높음을 보여주었습니다. 이는 장식된 테이블 이상의 다양한 상황으로 그 가능성을 확장할 수 있는 잠재력을 시사합니다.



### DM2RM: Dual-Mode Multimodal Ranking for Target Objects and Receptacles Based on Open-Vocabulary Instructions (https://arxiv.org/abs/2408.07910)
- **What's New**: 이 연구에서는 open-vocabulary (개방형 어휘) 명령어에 따라 일상적인 물체를 특정 가구로 전달할 수 있는 Domestic Service Robot (DSR)을 개발하는 것을 목표로 하고 있습니다. 기존의 방법들은 이미지 검색 환경에서 open-vocabulary 명령어를 활용한 이동 조작(task) 작업을 처리하는 경우가 드뭅니다.

- **Technical Details**: Dual-Mode Multimodal Ranking model (DM2RM)을 제안하며, 이 모델은 단일 모델을 기반으로 두 가지 모드인 목표 객체 및 수납 공간(receptacles)을 위한 이미지를 검색할 수 있게 합니다. Switching Phrase Encoder (SPE) 모듈과 Task Paraphraser (TP) 모듈을 도입하여 예측 대상을 기준으로 임베딩 공간을 전환하고, 명령어를 표준화된 형식으로 패러프레이즈합니다.

- **Performance Highlights**: DM2RM은 실제 건물 환경에서 수집된 데이터셋을 기반으로 평가되었으며, 이미지 검색 설정에서 기존 방법보다 우수한 성능을 보였습니다. DSR 플랫폼에서 fetch-and-carry 작업의 성공률이 82%에 달하며, 이는 zero-shot 전이 환경에서도 달성된 결과입니다.



### Enhancing Supply Chain Visibility with Knowledge Graphs and Large Language Models (https://arxiv.org/abs/2408.07705)
- **What's New**: 이 연구는 공급망 가시성을 향상시키기 위해 Knowledge Graphs (KGs)와 Large Language Models (LLMs)를 활용하는 새로운 프레임워크를 제안합니다. 이 방법은 이해관계자의 정보 공유에 의존하지 않으며, 다양한 공공 소스에서 정보를 자동으로 추출합니다.

- **Technical Details**: 프레임워크는 'zero-shot prompting' 방식을 사용하여 Named Entity Recognition (NER) 및 Relation Extraction (RE) 작업을 수행합니다. 이는 도메인별 대규모 데이터셋 없이도 가능하게 하여, 복잡한 공급망 관계를 적절하게 해석하고 추출합니다.

- **Performance Highlights**: 전기차 공급망을 대상으로 한 사례 연구를 통해, 이 프레임워크는 중요한 광물의 출처를 추적하여 가시성을 크게 향상시킵니다. 결과적으로 Tier-1 및 Tier-2 공급업체를 넘어서는 중요한 의존성과 대체 조달 옵션을 드러내며, 위험 관리와 전략적 계획에 기여합니다.



### MathBridge: A Large Corpus Dataset for Translating Spoken Mathematical Expressions into $LaTeX$ Formulas for Improved Readability (https://arxiv.org/abs/2408.07081)
Comments:
          9page, 6 figures

- **What's New**: 이 논문에서는 수학적 표현을 텍스트 형태로 이해하는 데 있어 발생하는 문제를 해결하기 위해 MathBridge라는 데이터셋을 소개합니다. 이 데이터셋은 약 2300만 개의 LaTeX 공식과 영어로 된 구술 표현이 쌍으로 이루어져 있으며, 이는 텍스트-투-LaTeX(text-to-LaTeX) 번역 연구의 기초를 구축합니다.

- **Technical Details**: MathBridge 데이터셋은 수학적 영어 표현을 LaTeX으로 변환하기 위한 것으로, 공립 대학교에 있는 오픈 소스 교재 및 arXiv에 업로드된 논문으로부터 수집된 데이터를 포함하고 있습니다. 논문에서는 영어 음성을 LaTeX 구문으로 변환하는 과정이 필요하며, 이를 위해 미리 훈련된 언어 모델(pretrained language model)을 활용합니다. 이 모델은 ASR(Automatic Speech Recognition)을 통해 수집된 수학적 표현을 LaTeX로 변환합니다.

- **Performance Highlights**: MathBridge를 이용한 평가에서, T5-large 모델의 sacreBLEU 점수가 4.77에서 46.8로 증가하여, MathBridge 데이터셋이 영어-LaTeX 번역을 위한 우수한 데이터셋임을 보여주었습니다. 또한, 논문에서는 기존 평가 지표가 LaTeX 텍스트 정렬 평가에는 적합하지 않다고 판단하고, 새로운 평가 메트릭의 필요성을 제시하였습니다.



New uploads on arXiv(cs.IR)

### DaRec: A Disentangled Alignment Framework for Large Language Model and Recommender System (https://arxiv.org/abs/2408.08231)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)과 협업 모델 간의 의미적 표현을 최적화하는 전통적인 방법과 차별화된 새로운 접근법을 제안합니다. 특히, LLMs와 협업 모델의 표현을 구별(disentangle)하여 특정 정보의 부정적 영향을 최소화하고 지식 전이를 더 효율적으로 수행하는 방법을 모색합니다.

- **Technical Details**: 연구에서는 LLM과 협업 모델의 표현을 특정(component)과 공유(shared) 구성요소로 분리하여, 상이한 정보와 의미를 보존할 수 있는 구조 정렬(alignment) 기법을 적용합니다. 이를 통해 적극적으로 관련 정보를 더할 수 있는 잠재 공간(latent space)의 활용성을 증대시킵니다.

- **Performance Highlights**: 다양한 기준 데이터셋을 통해 기존의 최첨단 알고리즘 대비 우수한 성능을 입증하였습니다. 연구에서 제안한 방법은 관찰된 결과를 통해 추천 시스템의 효율성과 정확성을 향상시키는 것으로 나타났습니다.



### Modeling Domain and Feedback Transitions for Cross-Domain Sequential Recommendation (https://arxiv.org/abs/2408.08209)
- **What's New**: 본 논문에서는 사용자의 도메인 및 피드백 전환을 모델링하기 위한 새로운 접근 방식인 Transition²를 제안합니다. 이는 기존의 교차 도메인 추천 시스템이 간과했던 피드백 전환 정보를 통합하여 개선된 사용자 만족도를 반영합니다.

- **Technical Details**: Transition²는 사용자 이력을 기반으로 한 전환 인식 그래프 인코더를 도입하고, 피드백 유형에 따라 엣지에 가중치를 부여하여 다양한 도메인 및 피드백 타입 간의 전환 정보를 추출합니다. 또한, 교차 전환 다중 머리 자기 주의 메커니즘을 사용하여 사용자 이력을 인코딩하고, 다양한 마스크를 활용하여 전환 정보의 구분을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, Transition²는 두 개의 공개 데이터 셋에서 기존의 순차 추천 및 교차 도메인 순차 추천 모델을 능가하는 성능을 보였습니다. 이는 다양한 도메인 간의 전환 및 피드백 전환의 중요성을 강조합니다.



### LLM4DSR: Leveraing Large Language Model for Denoising Sequential Recommendation (https://arxiv.org/abs/2408.08208)
- **What's New**: 이번 연구에서는 LLMs(대형 언어 모델)를 활용하여 시퀀셜 추천(Sequential Recommendation)에서 발생하는 노이즈를 제거하는 새로운 접근 방식인 LLM4DSR을 제안합니다. 이는 사용자 히스토리 상호작용 시퀀스의 노이즈를 효과적으로 탐지하고 대체할 수 있도록 LLMs를 조정합니다.

- **Technical Details**: LLM4DSR은 LLMs의 능력을 활용하기 위해 자기 지도 학습(Self-supervised learning) 방식을 통해 미세 조정(Fine-tuning)을 수행합니다. 이를 위해 시퀀스의 일부 항목을 랜덤 선택된 대체 항목으로 교체하는 지침 데이터셋을 구성하고, 이를 기반으로 노이즈 항목을 식별하고 적절한 대체 항목을 제안하도록 LLMs를 훈련합니다. 또한, 불확실성 추정 모듈을 도입하여 식별된 노이즈 항목의 신뢰성을 평가합니다.

- **Performance Highlights**: 폭넓은 실험을 통해 LLM4DSR이 기존의 방법들에 비해 우수한 성능을 발휘함을 입증했습니다. 다양한 추천 모델에 걸쳐 LLM4DSR의 적용 가능성과 성능 향상을 확인했으며, 특히 3개의 데이터셋과 3개의 추천 백본에서 기존 최첨단 노이즈 제거 방법들을 초월하는 성능을 보여주었습니다.



### From Clicks to Carbon: The Environmental Toll of Recommender Systems (https://arxiv.org/abs/2408.08203)
Comments:
          Accepted for presentation at the 18th ACM Conference on Recommender Systems in the Reproducibility Track

- **What's New**: 본 논문은 추천 시스템(recommender systems) 연구의 환경적 영향을 포괄적으로 분석하며, 2013년과 2023년에 발표된 논문들에서 에너지 소비 및 탄소 배출에 대한 비교를 수행합니다.

- **Technical Details**: 79개의 논문을 분석하여 전통적 알고리즘과 딥러닝 알고리즘을 사용하는 추천 시스템 실험 파이프라인을 재현하고, 하드웨어 에너지 미터를 통해 전기 소비를 측정한 후 CO2(이산화탄소) 등가물로 변환하여 탄소 발자국을 추정하였습니다. 2023년 연구 논문은 딥러닝 알고리즘을 사용할 경우 평균적으로 3,297kg의 CO2 등가물이 발생하며, 전통적인 알고리즘을 사용하는 논문에 비해 약 42배 더 많은 배출량을 보여줍니다.

- **Performance Highlights**: 딥러닝 기반의 추천 시스템 연구는 전통적인 방법에 비해 급격히 증가하는 탄소 배출을 야기하며, 특정 하드웨어 구성에 따라 탄소 발자국이 10배까지 차이나는 것으로 확인되었습니다. 연구 결과는 지속 가능한 연구 방법론 개발 및 연구 공동체 내에서 환경 의식을 제고하는 데 기여할 것으로 기대됩니다.



### Mamba Retriever: Utilizing Mamba for Effective and Efficient Dense Retrieva (https://arxiv.org/abs/2408.08066)
- **What's New**: 이 논문에서는 정보 검색 분야에서 Mamba Retriever라는 새로운 모델을 제안하여, 전통적인 Transformer 기반의 모델보다 효율성과 효과성을 모두 갖춘 인코더를 구현했습니다. 특히 긴 텍스트 검색에서의 성능 향상을 중점적으로 다루었습니다.

- **Technical Details**: Mamba Retriever는 bi-encoder 아키텍처를 사용하여 쿼리와 패시지를 임베딩 공간으로 인코딩해 계산합니다. Mamba 기반의 PLMs는 시퀀스 길이에 대해 선형적인 시간 복잡도를 지니며, 이를 통해 긴 텍스트 처리에서 효율성을 높입니다.

- **Performance Highlights**: MS MARCO 패시지 랭킹 데이터셋 및 LoCoV0 데이터셋에서의 실험 결과에 따르면, Mamba Retriever는 Transformer 기반의 모델보다 동등하거나 더 나은 성능을 보여주며, 긴 텍스트 검색에서도 빠른 추론 속도를 기록했습니다.



### AIE: Auction Information Enhanced Framework for CTR Prediction in Online Advertising (https://arxiv.org/abs/2408.07907)
- **What's New**: 본 연구에서는 Click-Through Rate (CTR) 예측의 성능 향상을 위해, 경매 정보(auction information)를 효과적으로 활용하는 방법을 제안합니다. 이를 위해 두 가지 모듈인 Adaptive Market-price Auxiliary Module (AM2) 및 Bid Calibration Module (BCM)을 도입하여 경매 신호를 기반으로 한 CTR 예측의 차별화된 접근 방식을 시도합니다.

- **Technical Details**: 제안하는 AIE 프레임워크는 경매 신호의 활용 부족 문제와 경매에서 발생하는 데이터 편향(auction bias)을 해결하는 데 중점을 둡니다. AM2는 시장 가격의 변동성을 포착하기 위해 동적 네트워크를 사용하는 보조 작업을 구축하며, BCM은 입찰(bid) 정보를 활용하여 데이터의 경매 편향을 완화합니다. 이 두 모듈은 경량화되어 있으며, 여러 모델에 독립적이고 인퍼런스 지연 속도에 친화적인 특성을 가지고 있습니다.

- **Performance Highlights**: 대규모 광고 플랫폼에서 시행된 한 달간의 A/B 테스트 결과, AIE는 기본 모델에 비해 eCPM을 5.76% 향상시키고 CTR을 2.44% 개선하는 성과를 보였습니다. 다양한 산업 데이터셋을 이용한 실험에서도 AIE의 효과성과 호환성이 입증되었습니다.



### SWaT: Statistical Modeling of Video Watch Time through User Behavior Analysis (https://arxiv.org/abs/2408.07759)
- **What's New**: 본 논문에서는 사용자 중심의 관점에서 (short) 비디오 시청 시간을 모델링하는 새로운 백박스 통계 프레임워크를 제안합니다. 이전 연구와 달리, 사용자의 행동 패턴을 직접 고려하여 비디오 시청 시간을 보다 효과적으로 예측할 수 있습니다.

- **Technical Details**: 모델링 기법으로는 사용자의 다양한 시청 행동을 명세하기 위해 도메인 지식을 활용한 통계적 모델을 구축하였습니다. 비디오 시청 시간의 비선형적 성질을 다루기 위해 'bucketization'을 도입하여 비디오 길이에 따라 시청 확률을 조정하였습니다. 이 접근법은 실제 산업 데이터셋 및 A/B 테스트에 효과적임을 입증했습니다.

- **Performance Highlights**: 우리는 두 개의 공개 데이터셋과 대규모 산업 데이터셋에서 실험을 수행하였으며, 모든 실험에서 제안된 모델이 강력한 기준선에 맞서 경쟁력을 나타냈습니다. 또한 인기 있는 짧은 비디오 플랫폼에서 온라인 A/B 테스트를 통해 비디오 시청 시간을 유의미하게 증가시킬 수 있었습니다.



### A Guide to Similarity Measures (https://arxiv.org/abs/2408.07706)
Comments:
          27 pages

- **What's New**: 이 논문은 다양한 데이터 과학 어플리케이션 도메인에서 널리 사용되는 유사도 측정 방식을 포괄적으로 설명하고 있습니다. 비전문가와 전문가 모두를 위한 유용한 가이드를 제공하며, 각 측정 방식의 정의와 성질을 이해할 수 있도록 구성되었습니다.

- **Technical Details**: 유사도 측정은 데이터 과학의 다양한 응용에서 중심적인 역할을 하며, 여기에는 machine learning, artificial intelligence, information retrieval 등이 포함됩니다. 특정 작업을 위해 적절한 유사도 또는 거리 측정을 선택하는 것이 중요합니다. 이 논문은 50개 이상의 유사도/거리 측정 방식을 제시하며, 그들의 기본 변형 또한 설명합니다. 이 측정 방식들은 주로 벡터 공간 내에서의 내적(inner product)을 기반으로 합니다.

- **Performance Highlights**: 이 논문에서는 Mahalanobis distance와 같은 유용한 거리 측정 방법도 다루며, 이 방법이 다변량 이상 탐지 및 분류에 매우 유용하다고 강조합니다. 유사도 및 거리 측정은 데이터 인스턴스 간의 유사성을 측정하는 기본적인 방법이 되며, 주로 유사한 객체일수록 높은 점수를 부여받는 방식으로 발전합니다.



### Enhancing Supply Chain Visibility with Knowledge Graphs and Large Language Models (https://arxiv.org/abs/2408.07705)
- **What's New**: 이 연구는 공급망 가시성을 향상시키기 위해 Knowledge Graphs (KGs)와 Large Language Models (LLMs)를 활용하는 새로운 프레임워크를 제안합니다. 이 방법은 이해관계자의 정보 공유에 의존하지 않으며, 다양한 공공 소스에서 정보를 자동으로 추출합니다.

- **Technical Details**: 프레임워크는 'zero-shot prompting' 방식을 사용하여 Named Entity Recognition (NER) 및 Relation Extraction (RE) 작업을 수행합니다. 이는 도메인별 대규모 데이터셋 없이도 가능하게 하여, 복잡한 공급망 관계를 적절하게 해석하고 추출합니다.

- **Performance Highlights**: 전기차 공급망을 대상으로 한 사례 연구를 통해, 이 프레임워크는 중요한 광물의 출처를 추적하여 가시성을 크게 향상시킵니다. 결과적으로 Tier-1 및 Tier-2 공급업체를 넘어서는 중요한 의존성과 대체 조달 옵션을 드러내며, 위험 관리와 전략적 계획에 기여합니다.



### Empathic Responding for Digital Interpersonal Emotion Regulation via Content Recommendation (https://arxiv.org/abs/2408.07704)
- **What's New**: 이 논문에서는 온라인 플랫폼에서 대인 간 감정 조절(Interpersonal Emotion Regulation, IER)을 향상시키기 위한 콘텐츠 추천 시스템을 제안합니다. 이 시스템은 사용자가 감정을 조절할 수 있도록 돕고, 특히 공감적 반응(empathic responding) 전략에 맞춘 미디어 콘텐츠를 추천합니다.

- **Technical Details**: 제안된 추천 시스템은 사용자 활동과 선호도를 기반으로 한 Contextual Multi-Armed Bandits (CMAB) 알고리즘을 사용하여, 37.5K개의 사용자 게시물과 상호작용 데이터를 통해 맞춤형 공감적 추천을 생성합니다. 이 연구는 혼합 방법 연구 설계를 통해 텍스트 기반 소셜 미디어 데이터 분석과 사용자 설문 조사를 포함합니다.

- **Performance Highlights**: 실험 결과, 제안된 추천 시스템에 의해 생성된 공감적 추천이 사용자들 사이에서 집중(distraction) 및 회피(avoidance)와 같은 기존 감정 조절 전략보다 선호되는 것으로 나타났습니다.



### KGV: Integrating Large Language Models with Knowledge Graphs for Cyber Threat Intelligence Credibility Assessmen (https://arxiv.org/abs/2408.08088)
- **What's New**: 본 논문은 Cyber Threat Intelligence (CTI) 품질 평가를 위한 새로운 방법론인 Knowledge Graph Verifier (KGV)를 제안합니다. 이 프레임워크는 지식 그래프(knowledge graphs)와 대형 언어 모델(large language models, LLMs)을 결합하여 OSCTI의 주요 주장(key claims)을 자동으로 추출하고 이를 사실 확인(fact-checking)하는 과정을 간소화합니다.

- **Technical Details**: KGV는 구문 이해와 사실 확인을 위해 지식 그래프를 논문 단락(paragraph)으로 구성하고, 노드로서 단락을 사용하며, 시멘틱 유사성(semantic similarity)을 엣지로 활용합니다. 이를 통해 모델의 시맨틱 이해 능력을 향상시키고 레이블링 요구사항을 단순화합니다. 또한, 이 연구는 이질적인 소스에서 발췌한 첫 번째 데이터를 공개하여 CTI 품질 평가의 데이터 부족 문제를 해결합니다.

- **Performance Highlights**: 경험적 결과에 따르면, KGV는 LLM의 정보 품질 평가 성능을 크게 향상시키며, 기존의 방법들과 비교할 때 데이터 주석(labeling) 요구를 대폭 줄이면서도 여전히 강력한 추론 능력을 유지합니다. KGV는 네트워크 위협 평가에서 XXX의 정확도를 달성할 수 있습니다.



### Extracting Sentence Embeddings from Pretrained Transformer Models (https://arxiv.org/abs/2408.08073)
- **What's New**: 이번 연구에서는 다양한 방법을 통해 BERT(Bidirectional Encoder Representations from Transformers) 모델의 문장 표현을 개선하는 방안을 제시합니다. 특히, 문장 레벨의 임베딩을 효과적으로 추출하는 기술을 평가하고, BERT 기반 모델의 성능을 향상시키기 위한 실험적인 방법들을 탐구합니다.

- **Technical Details**: 본 연구에서는 110백만 개의 파라미터를 가진 BERT의 숨겨진 표현을 여러 레이어와 토큰에서 추출하여 최적의 문장 표현을 얻기 위한 다양한 방법을 시도했습니다. 이를 위해 토큰 집계(token aggregation) 및 표현 후처리(post-processing) 기법들을 테스트했으며, 여러 짧은 텍스트 클러스터링 및 분류 작업에서 성능을 평가했습니다.

- **Performance Highlights**: 제안된 표현 추출 방법들은 모든 모델에서 STS(Semantic Textual Similarity) 및 클러스터링 작업의 성능을 개선시켰습니다. 특히, 정적 토큰 기반 모델의 성능이 매우 크게 향상되었고, 무작위 임베딩(random embeddings)의 STS 작업 성능이 BERT 기반 표현에 거의 도달했습니다.



### An Efficient Continuous Control Perspective for Reinforcement-Learning-based Sequential Recommendation (https://arxiv.org/abs/2408.08047)
- **What's New**: 이 논문에서는 사용자 선호를 동적으로 추론하고 오프라인에서 강화 학습 기반 추천 시스템을 최적화하기 위해 연속 정책을 처리할 수 있는 알고리즘 프레임워크를 설계합니다. 이를 통해 사용자 선호 공간의 낮은 차원에서 효과적인 제어를 가능합니다.

- **Technical Details**: 우리는 효율적인 연속 제어 프레임워크(Efficient Continuous Control framework, ECoC)를 제안합니다. 이 프레임워크는 정량 사용자 및 항목 공간에서 추출한 통합 행동 표현을 기반으로 정책 평가 및 개선 절차를 개발합니다. 전략적 탐색 및 방향 제어가 최종 추천 결정에 중요합니다.

- **Performance Highlights**: ECoC는 기존의 이산 기반 방법들과 비교하여 훈련 효율성을 극대화하고, 오프라인 데이터 캡처 및 장기 보상 획득에서 성능을 향상시킵니다. 실험 결과 ECoC는 사용자 행동을 모방하고 누적 장기 보상에서 우수한 결과를 보여줍니다.



### The Nah Bandit: Modeling User Non-compliance in Recommendation Systems (https://arxiv.org/abs/2408.07897)
Comments:
          12 pages, 8 figures, under review

- **What's New**: 이 논문에서는 사용자가 추천을 거부하고 자신의 선호 옵션을 선택하는 상황을 다루는 새로운 프레임워크인 Nah Bandit를 소개합니다. 기존의 추천 시스템들은 디지털 세계에서 효과를 보여왔으나, 물리적 세계에서의 효과적인 구현에서 어려움이 있었습니다. Nah Bandit 문제를 통해 추천 시스템의 비준수(non-compliance) 피드백을 활용하여 선호 학습을 가속화할 수 있는 방법을 제시합니다.

- **Technical Details**: Nah Bandit 문제는 사용자가 추천에 대해 'nah'라고 말하고 대신 선호하는 옵션을 선택할 수 있는 경우를 모델링합니다. 이 연구는 사용자 비준수 모델을 도입하여 추천의 앵커링 효과를 매개변수화하고, 사용자 추천 및 비추천 피드백을 통합하여 빠른 선호 학습을 위한 Expert with Clustering (EWC) 알고리즘을 제안합니다. EWC는 사용자의 비준수 모델을 사용하여 클러스터별 선호 매개변수를 결정하고, Hedge 알고리즘을 사용하여 가장 적합한 전문가를 선택합니다.

- **Performance Highlights**: EWC 알고리즘은 N명의 사용자, T회의 라운드, K개의 클러스터를 가진 추천 시나리오에서 O(N√(TlogK) + NT)의 후회 경계를 달성하여 LinUCB 알고리즘보다 우수한 이론적 성능을 보여줍니다. 실험 결과, EWC는 감독 학습 및 전통적인 컨텍스트 밴딧 접근 방식보다 더 좋은 성능을 발휘하였으며, 이는 추천 정확도를 향상시키는 데 기여합니다.



New uploads on arXiv(cs.CV)

### Towards Flexible Visual Relationship Segmentation (https://arxiv.org/abs/2408.08305)
- **What's New**: FleVRS라는 새로운 모델을 제안하여 인간-객체 상호작용(HOI) 감지, 장면 그래프 생성(SGG), 지칭 관계(RR) 작업을 통합적으로 처리합니다. 이 모델은 단일 프레임워크에서 다양한 관계를 유연하게 분류할 수 있도록 설계되었습니다.

- **Technical Details**: FleVRS는 텍스트와 이미지 모달리티 간의 시너지를 활용하여 관계를 지각하고, Vision-Language 모델로부터 텍스트적 특징을 사용하여 시각적 개념 이해를 향상시킵니다. 이 모델은 dynamic query-기반 Transformer 아키텍처를 사용하여 <subject, predicate, object>의 형태로 관계를 출력합니다.

- **Performance Highlights**: FleVRS는 다양한 데이터셋에서 HICO-DET에서 +1.9 mAP, VRD에서 +11.4 Acc, 보지 못한 HICO-DET에서 +4.7 mAP 등 기존 모델보다 뛰어난 성능을 보였습니다. 일반적인 VRS, promptable, open-vocabulary 작업에서 경쟁력 있는 성능을 나타냅니다.



### SLCA++: Unleash the Power of Sequential Fine-tuning for Continual Learning with Pre-training (https://arxiv.org/abs/2408.08295)
Comments:
          This paper is an extension of our ICCV 23 paper (arXiv:2303.05118)

- **What's New**: 본 논문은 연속 학습(Local Learning)에서의 사전 학습(Pre-training)을 활용한 새로운 접근 방식인 SLCA++를 제안합니다. SLCA++는 Sequential Fine-tuning(Seq FT)의 힘을 활용하는 강력한 기준 방법으로, 기존 방법의 한계를 극복하고 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: 이 연구에서는 SLCA++라는 프레임워크를 도입하여, 학습률을 선택적으로 줄이고, 분류 레이어(classification layer)를 정렬하는 후처리(post-hoc) 과정을 포함합니다. 또한, Symmetric Cross-Entropy (SCE) 손실 함수를 채택하여 효율성을 높이고, Hybrid Slow Learner(Hybrid-SL) 전략을 통해 파라미터 효율적인 Seq FT 구현을 지원합니다.

- **Performance Highlights**: SLCA++는 이미지 분류 벤치마크에서 다양한 연속 학습 시나리오를 통해 보여준 결과로, Split CIFAR-100, Split ImageNet-R, Split CUB-200, Split Cars-196에서 기존 SOTA(state-of-the-art) 방법보다 45% 이상의 성능 향상을 달성하며, 정량적인 평가를 통해 우수한 성과를 입증합니다.



### HeightLane: BEV Heightmap guided 3D Lane Detection (https://arxiv.org/abs/2408.08270)
Comments:
          10 pages, 6 figures, 5 tables

- **What's New**: HeightLane라는 새로운 방법론을 도입하여 단일 이미지로부터 깊이 맵을 예측하고, 다층 경사를 기반으로 한 앵커를 생성하여 노면을 정확히 모델링합니다.

- **Technical Details**: HeightLane은 단일 이미지로부터 깊이 맵을 예측하며, 변형 가능한 주의 기반의 공간적 특징 변환 프레임워크를 사용하여 2D 이미지 특징을 3D bird's eye view (BEV) 특징으로 효과적으로 변환합니다. 이 과정에서 예측된 높이 맵을 POSitional Encoding에 사용하여 공간적 정확성을 높입니다.

- **Performance Highlights**: OpenLane 검증 세트에서 HeightLane이 F-score 면에서 최첨단 성능을 달성하며, 복잡한 도로 환경에서 3D 차선 감지의 잠재력을 강조합니다.



### Snuffy: Efficient Whole Slide Image Classifier (https://arxiv.org/abs/2408.08258)
Comments:
          Accepted for ECCV 2024

- **What's New**: 이번 연구에서는 Whole Slide Image (WSI) 분류를 위한 새로운 MIL-pooling 방법인 Snuffy 아키텍처를 도입하여 제한된 프리트레이닝으로 인한 성능 저하를 완화하고, 지속적인 few-shot pre-training을 가능하게 합니다.

- **Technical Details**: Snuffy는 sparse transformer를 기반으로 한 새로운 MIL-pooling 방법을 제안하며, 이 sparsity 패턴은 병리학에 맞게 조정되어 있습니다. 이 연구는 Snuffy의 sparsity가 보편적 근사기(approximation) 역할을 하며, 레이어 수에 대한 확률적 경계를 제공합니다.

- **Performance Highlights**: Snuffy 아키텍처는 CAMELYON16 및 TCGA 폐암 데이터셋에서 뛰어난 WSI 및 패치(level) 정확도를 달성하며, AUC 0.987, ROI 감지에서 FROC 0.675의 새로운 SOTA를 기록했습니다.



### Computer Vision Model Compression Techniques for Embedded Systems: A Survey (https://arxiv.org/abs/2408.08250)
- **What's New**: 이 논문은 컴퓨터 비전(Computer Vision)에서 모델 압축(Model Compression) 기술을 다루며, 현대 모델들이 임베디드 시스템(Embedded Systems)에서 사용될 수 있도록 하는 방법에 대해 설명합니다.

- **Technical Details**: 최근 Vision Transformer (ViT)와 고급 Convolutional Neural Networks (CNNs)가 채택됨에 따라, 최신 아키텍처의 파라미터 수가 2012년 AlexNet의 6200만 개에서 2024년 AIM-7B의 70억 개로 증가했습니다. 이 논문은 주요 모델 압축 기술을 설명하고, 임베디드 장치에서 분석할 때 최선의 기법을 선택하는 방법에 대해 논의합니다.

- **Performance Highlights**: 논문에서는 다양한 임베디드 장치에서 모델 압축 기술을 비교하고, 구현 초기 단계의 문제를 극복하기 위해 연구자들에게 유용한 코드도 공유합니다. 또한, 모델 압축의 최신 동향을 제시하며, 압축 모델에 대한 사례 연구도 제공합니다.



### Comparative Evaluation of 3D Reconstruction Methods for Object Pose Estimation (https://arxiv.org/abs/2408.08234)
- **What's New**: 이번 논문에서는 3D 객체 재구성 하에 머신러닝 모델을 활용한 객체 자세 추정의 정확성을 평가하기 위한 새로운 벤치마크를 제안합니다. CAD 모델 없이도 높은 정확도를 보장하는 방법을 모색합니다.

- **Technical Details**: 우리는 기존의 CAD 기반 및 CAD-free 접근 방식을 포함한 다양한 최첨단 3D 재구성 방법을 평가합니다. 벤치마크는 YCB-V 데이터셋을 기반으로 하며, 로봇 팔을 통해 캡처한 이미지를 사용하여 객체 재구성을 수행합니다. 이 과정에서 재구성의 품질과 자세 추정의 정확성 간의 상관 관계를 분석합니다.

- **Performance Highlights**: (1) 3D 재구성 품질을 측정하는 일반적인 메트릭이 자세 추정의 정확성과는 반드시 일치하지 않음을 보여줍니다. (2) 고전적 비학습 기반 접근 방식이 현대의 학습 기반 재구성 기법과 유사한 성능을 보이며, 더 빠른 재구성 시간을 제공합니다. (3) CAD 모델과 유사한 성능을 얻을 수 있는 객체 유형도 있지만, 세밀한 디테일과 반사 표면을 가진 객체에서는 여전히 성능 차이가 존재합니다.



### The Dawn of KAN in Image-to-Image (I2I) Translation: Integrating Kolmogorov-Arnold Networks with GANs for Unpaired I2I Translation (https://arxiv.org/abs/2408.08216)
Comments:
          10 pages, 6 Figures, 1 Table

- **What's New**: 이 논문은 Kolmogorov-Arnold Network (KAN)를 사용하여 Generative AI의 이미지-이미지 변환 모델을 개선하는 방법을 소개합니다. KAN은 기존의 Multi-layer Perceptron (MLP) 대신 사용되며, 정보가 더 풍부한 저차원 벡터 표현을 생성할 수 있도록 돕습니다. 또한, KAN-CUT 모델을 제안하여 더 우수한 생성 품질을 달성합니다.

- **Technical Details**: 기존 Contrastive Unpaired Image-to-Image Translation (CUT) 모델의 두 개의 MLP를 KAN으로 효율적으로 대체하여 KAN-CUT 모델을 개발하였습니다. 이 모델은 Gated Linear Units (GLU)를 사용한 활성화 함수를 통합하여 KAN 레이어를 향상시켰습니다.

- **Performance Highlights**: 광범위한 실험 결과, KAN은 Generative AI 중 특히 이미지-이미지 변환에서 GAN과 함께 효과적으로 적용됨을 보여줍니다. KAN-CUT 모델은 목표 도메인에서 고품질 이미지를 생성하는 데 유리함을 입증하였습니다.



### WaterSplatting: Fast Underwater 3D Scene Reconstruction Using Gaussian Splatting (https://arxiv.org/abs/2408.08206)
Comments:
          Web: this https URL

- **What's New**: 본 연구에서는 수중 3D 장면 재구성을 위한 혁신적인 접근 방식을 제안합니다. 이 방법은 3D Gaussian Splatting (3DGS)와 볼류메트릭 렌더링을 융합하여 수중 데이터 처리를 효과적으로 수행합니다.

- **Technical Details**: 제안된 방법은 3DGS를 사용하여 명시적인 기하형상을 표현하고, 별도의 볼륨 필드를 사용하여 산란 매체를 포착합니다. 이는 장면 복원에 있어 산란 매체를 제거하는 것을 허용합니다. 또한, 새로운 손실 함수가 3DGS와 인간의 HDR 및 저조도 장면 인식을 정렬하는데 사용됩니다.

- **Performance Highlights**: SeaThru-NeRF 데이터셋에서의 평가 결과, 제안된 방법이 기존 NeRF 기반 방법보다 더 우수한 렌더링 품질을 달성했으며, 실시간 렌더링 성능을 제공하여 기존 방법의 효율성 한계를 해결했습니다.



### A Multi-task Adversarial Attack Against Face Authentication (https://arxiv.org/abs/2408.08205)
Comments:
          Accepted by ACM Transactions on Multimedia Computing, Communications, and Applications

- **What's New**: 이번 논문에서는 MTADV라는 다중 작업 적대적 공격(multi-task adversarial attack) 알고리즘을 제안하여, 다수의 사용자 또는 시스템에 적응할 수 있는 공격 방법을 소개합니다. 기존의 공격 방법들은 단일 작업에만 초점을 맞추고 있었으나, MTADV는 이를 넘어서 다양한 상황에서 효과적으로 적용될 수 있는 특성을 갖추고 있습니다.

- **Technical Details**: MTADV는 다양한 공격 시나리오를 다룰 수 있도록 설계되어 있으며, 침투 공격(morphing attack), 보편적 공격(universal attack), 전이 공격(transferable attack), 그리고 반격 공격(counterattack)과 같은 여러 공격 방법을 지원합니다. 이 알고리즘은 단일 사용자 시스템에 대한 효율성을 유지하면서도 여러 사용자에 대한 적응력을 제공합니다. MTADV는 흰 상자(white-box) 및 회색 상자(gray-box) 환경에서도 적용 가능하며, LFW, CelebA 및 CelebA-HQ와 같은 다양한 얼굴 데이터셋과 여러 딥러닝 모델(FaceNet, InsightFace, CurricularFace)에서도 효과적입니다.

- **Performance Highlights**: MTADV는 실제 상황에서 나타날 수 있는 다섯 가지 대표적인 공격 시나리오에 대한 효과성을 실험적으로 입증하였으며, 다양한 매개 변수와 공격 성공률, 효율성 및 이미지 품질 간의 상관 관계에 대한 철저한 분석을 수행하였습니다. 또한, 논문의 코드가 공개되어 있어 연구자들이 실험을 replicable할 수 있도록 지원합니다.



### Towards Practical Human Motion Prediction with LiDAR Point Clouds (https://arxiv.org/abs/2408.08202)
- **What's New**: 이번 연구에서는 LiDAR 기반의 첫 번째 3D 인간 동작 예측 방법인 'LiDAR-HMP'를 제안합니다. 이 방법은 기존의 인간 자세 추정 방식이 필요 없이, 원시 LiDAR 포인트 클라우드를 직접 입력받아 미래의 인체 자세를 예측합니다.

- **Technical Details**: LiDAR-HMP는 구조 인식 본체 특징 기술자(Structure-aware Body Feature Descriptor), 적응형 운동 잠재 매핑(Adaptive Motion Latent Mapping), 그리고 공간-시간 상관관계 정제(Spatial-Temporal Correlations Refinement)로 구성된 세 가지 모듈로 이루어져 있습니다. 이 모듈들은 원시 포인트 클라우드에서 중요한 정보를 추출하고, 예측의 정확성을 높입니다.

- **Performance Highlights**: 리서치 결과, LiDAR-HMP는 LiDARHuman26M과 LIPD 두 공공 데이터셋에서 기존 방법에 비해 평균 관절 위치 오류(MPJPE)가 큰 폭으로 감소되는 성과를 기록했습니다. 짧은 기간 예측에서 평균 17.42mm, 긴 기간 예측에서 평균 11.62mm의 정확성을 보여주며, 실제 상황에서도 실시간 동작 예측이 가능합니다.



### Heavy Labels Out! Dataset Distillation with Label Space Lightening (https://arxiv.org/abs/2408.08201)
- **What's New**: 본 연구에서는 데이터셋 증류(dataset distillation) 과정에서 발생하는 '무거운 레이블(heavy labels)' 문제를 해결하기 위한 새로운 프레임워크인 HeLlO(Heavy Labels Out)를 제안합니다. HeLlO는 기존의 방대한 soft label을 저장하는 대신, 이미지에서 레이블로 직접 생성할 수 있는 효율적인 이미지-레벨 프로젝트(predictor)를 구현합니다.

- **Technical Details**: HeLlO는 오픈 소스 기초 모델인 CLIP를 활용하여 low-rank 행렬로 구성된 경량화된 레이블 표현을 생성합니다. 이 과정에서 LoRA(Low-Rank Adaptation)와 유사한 전략을 채택하여 사전 훈련된 CLIP의 특성 공간을 목표 데이터셋의 특성 공간으로 변환하고, 라벨 카테고리의 텍스트 표현으로 프로젝트를 초기화합니다. 최적의 이미지 생성을 위해 이미지 최적화 방법도 제안됩니다.

- **Performance Highlights**: 실험 결과, HeLlO는 원본 소프트 레이블의 0.003%에 해당하는 저장 용량만으로도 현재의 주요 데이터셋 증류 방법들과 동등하거나 더 나은 성능을 보이는 것으로 나타났습니다. 이로서 대규모 데이터셋에서의 증류 효율성 문제가 해결될 것으로 기대됩니다.



### Beyond Full Label: Single-Point Prompt for Infrared Small Target Label Generation (https://arxiv.org/abs/2408.08191)
- **What's New**: 본 연구는 적외선 소형 타겟 라벨 생성을 위한 학습 기반의 단일 점 주석 패러다임인 IRSTLG를 최초로 시도하였습니다. 라벨 생성을 위해 타겟 검출보다 단 1개의 추가 점 프롬프트가 필요하다는 직관을 기반으로 합니다.

- **Technical Details**: 제안된 프레임워크 EDGSP는 다음과 같은 세 가지 주요 요소로 구성됩니다: 1) Target Energy Initialization (TEI) - 모든 타겟의 초기 패턴을 진화시키기 위해 각 점 프롬프트를 Gaussian으로 확장합니다. 2) Double Prompt Embedding (DPE) - 두 차례 인코딩하여 모델이 관심 영역에 집중하도록 합니다. 3) Bounding Box-based Matching (BBM) - 후보 타겟에 대해 바운딩 박스를 생성하고 좌표로 점 프롬프트와 일치시킵니다.

- **Performance Highlights**: 실험 결과, EDGSP를 장착한 세 가지 기준선의 생성된 의사 라벨이 SIRST, NUDT-SIRST, IRSTD-1k 데이터셋에서 100% 객체 레벨 탐지 확률(Pd)을 달성하고, 0% 허위 경고 비율(Fa)을 기록했습니다. 또한, 다운스트림 탐지 작업에서 우리의 중심 주석 의사 라벨이 전체 라벨을 초과하는 성능을 보이며, 조잡한 단일 점 주석에도 불구하고 전체 라벨링의 99.5% 성능을 달성했습니다.



### FancyVideo: Towards Dynamic and Consistent Video Generation via Cross-frame Textual Guidanc (https://arxiv.org/abs/2408.08189)
- **What's New**: 본 연구에서는 FancyVideo라는 새로운 텍스트-비디오(전환된 비디오) 생성 모델을 소개합니다. 이 모델은 기존의 텍스트 제어 메커니즘을 개선하여 특정 프레임에 대한 텍스트 가이드를 강화합니다. 이를 통해 더욱 동적이고 일관성 있는 비디오를 생성할 수 있습니다.

- **Technical Details**: FancyVideo는 세 가지 주요 구성 요소로 이루어진 Cross-frame Textual Guidance Module (CTGM)을 사용합니다: Temporal Information Injector (TII), Temporal Affinity Refiner (TAR), Temporal Feature Booster (TFB). 각각은 텍스트 조건에 시간 정보를 주입하고, 상관 행렬을 정제하며, 잠재 특징의 시간적 일관성을 높이는 역할을 합니다.

- **Performance Highlights**: FancyVideo는 EvalCrafter 벤치마크에서 최첨단(TOP) 성과를 달성했으며, UCF-101 및 MSR-VTT 데이터셋에서도 경쟁력 있는 성능을 보여주었습니다. 여러 실험을 통해 이 모델이 동적이고 일관성 있는 비디오 생성을 성공적으로 수행함을 입증하였습니다.



### Not Every Image is Worth a Thousand Words: Quantifying Originality in Stable Diffusion (https://arxiv.org/abs/2408.08184)
Comments:
          GenLaw ICML 2024

- **What's New**: 본 연구는 text-to-image (T2I) 생성 확산 모델에서의 독창성을 정량화하는 문제를 다루며, 저작권과 독창성에 초점을 맞추고 있다. T2I 모델이 혁신하고 일반화할 수 있는 능력을 평가하기 위해 제어된 실험을 수행하고, 모델이 훈련 데이터에 대한 친숙함을 바탕으로 이미지의 독창성을 측정하는 방법을 제안한다.

- **Technical Details**: 제안된 방법은 텍스처 인버전(textual inversion) 기법을 활용하여 모델이 재구성하는 데 필요한 토큰(token)의 수를 기반으로 이미지의 독창성을 측정한다. 연구에서는 사전 훈련된 안정적인 확산 모델을 사용하여 실험을 진행하고, 생성된 이미지의 독창성과 토큰 수 사이의 상관관계를 보여준다.

- **Performance Highlights**: 실험 결과, 안정적인 확산 모델이 다양한 프롬프트를 통해 알려지지 않은 요소를 효과적으로 재창조할 수 있음이 확인되었으며, 훈련 데이터의 다양성이 증가할수록 모델의 일반화 능력이 향상되었음을 보여준다. 새로운 개념의 재구성을 용이하게 하는 요소의 수와 생성된 이미지 간의 독창성 사이에 유의미한 상관관계가 나타났다.



### Your Turn: Real-World Turning Angle Estimation for Parkinson's Disease Severity Assessmen (https://arxiv.org/abs/2408.08182)
- **What's New**: 이 논문은 파킨슨병(Parkinson's Disease, PD) 환자의 실생활에서의 회전 각도를 자동으로 측정하는 심층 학습 기반의 접근 방식을 제안합니다. 이를 위해 비디오에서 3D 골격을 추출하고 엉덩이 및 무릎 관절의 회전을 계산합니다.

- **Technical Details**: 본 연구에서는 Fastpose 및 Strided Transformer와 같은 최신 인간 자세 추정 모델을 활용하여 24명의 피험자(12명의 PD 환자 및 12명의 건강한 대조군)의 1386개 회전 비디오 클립을 분석합니다. 데이터셋은 Turn-REMAP과 Turn-H3.6M으로 구성되어 있으며, 수동으로 수집한 환경에서의 회전 활동을 포함합니다.

- **Performance Highlights**: Turn-REMAP 데이터셋에서 회전 계산 정확도가 41.6%, 평균 절대 오차(Mean Absolute Error, MAE)가 34.7°, 가중 정밀도(Weighted Precision, WPrec) 68.3%를 달성했습니다. 이는 단일 단안 카메라 데이터를 활용한 최초의 연구로 실제 환경에서 PD 환자의 회전을 quantification할 수 있는 방법을 제공합니다.



### Towards flexible perception with visual memory (https://arxiv.org/abs/2408.08172)
- **What's New**: 본 논문에서는 신경망의 대표성을 손상시키지 않고도 유연하게 지식을 추가, 삭제할 수 있는 새로운 방법론을 제시합니다. 이는 기존의 고정된 신경망 아키텍처의 한계를 극복하기 위한 접근입니다.

- **Technical Details**: 제안된 시스템은 이미지 분류 작업을 이미지 유사성(Pre-trained embedding 사용)과 검색(Fast Nearest Neighbor Retrieval 기술 활용)으로 분해하여, 데이터의 추가 및 삭제가 가능한 시각적 기억(visual memory) 체계를 구성합니다. 데이터 추가는 개별 샘플에서 전체 클래스, 심지어 수십억 개의 데이터 규모까지 가능하며, 불필요한 데이터는 unlearning과 memory pruning을 통해 제거할 수 있습니다. 또한, 의사결정 메커니즘은 해석 가능하며, 이는 사용자 개입이 가능합니다.

- **Performance Highlights**: 이 시스템은 ImageNet-1K 및 iNaturalist 데이터셋에서의 성능 비교를 포함하여, 다양한 하이퍼파라미터 설정에 대한 분석 결과를 통해 효율성을 입증합니다. 메모리 프루닝 기술을 통해 잘못된 의사결정에 기여한 이미지를 효과적으로 제거하여 정확도를 향상시키는 등의 성과를 거두었습니다.



### Unsupervised Variational Translator for Bridging Image Restoration and High-Level Vision Tasks (https://arxiv.org/abs/2408.08149)
- **What's New**: 본 논문은 전통적인 이미지 복원 기술로 부터 기계 인식(mechanical perception)을 위한 고급 비전(high-level vision) 작업 성능을 향상시키는 데 중점을 둡니다. 기존의 지도 학습(supervised learning) 방식에서 벗어나 기존 네트워크를 재훈련하지 않고 연결할 수 있는 비지도 학습(unsupervised learning) 방법인 Variational Translator (VaT)를 제안합니다.

- **Technical Details**: VaT는 복원 출력(restoration output)과 고급 비전 입력(high-level vision input)의 결합 분포(joint distribution)를 모델링하며, вариational inference를 통해 최적화 목표를 콘텐츠 보존(content preservation)과 고급 비전 작업에 관련된 주변 우도(marginal likelihood) 극대화로 나누어 설정합니다. 이를 통해 라벨이 필요 없는 자가 학습(self-training) 패러다임을 활용하여 복원 이미지를 원래의 콘텐츠와 유사하게 유지하면서도 고급 비전 작업에서 뛰어난 성능을 발휘하도록 합니다.

- **Performance Highlights**: VaT는 안개 낀 환경에서 객체 탐지(task of object detection under foggy)에서 평균 정밀도(mean Average Precision, mAP) 기준으로 비지도 방법들에 비해 10% 향상된 성능을 보여주며, 일부 복잡한 현실 시나리오에서는 지도 방법들보다 약 4% 더 나은 성과를 기록합니다. 이 결과는 VaT의 효과적인 성능 향상을 입증합니다.



### CorrAdaptor: Adaptive Local Context Learning for Correspondence Pruning (https://arxiv.org/abs/2408.08134)
Comments:
          8 pages, 4 figures, accepted by ECAI

- **What's New**: 이번 논문에서는 이미지 매칭을 위한 새로운 모델인 CorrAdaptor를 제안합니다. CorrAdaptor는 복잡한 이미지 변형에 적응할 수 있는 이중 분기 구조를 가지고 있으며, 명시적 및 암시적 지역 그래프 학습을 통해 지역 맥락을 조정하는 기능을 갖추고 있습니다.

- **Technical Details**: CorrAdaptor의 구조는 명시적 분기와 암시적 분기로 나뉩니다. 명시적 분기는 KNN 기반 그래프를 사용하여 초기 이웃을 식별하고, 암시적 분기는 학습 가능한 행렬을 활용하여 이웃을 부드럽게 할당합니다. 또한, 모션 일관성(motion consistency)을 통합하는 모션 주입 모듈(motion injection module)을 설계하여 아웃라이어(outlier)의 영향을 억제하고 지역 맥락 학습을 정제합니다.

- **Performance Highlights**: CorrAdaptor는 여러 데이터셋에서 실험을 진행하며 최첨단 성능을 달성했습니다. 논문의 실험 결과는 CorrAdaptor가 질적 및 양적으로 우수한 결과를 보여주며, 이중 분기 구조와 모션 주입 모듈의 장점을 강조합니다.



### Category-Prompt Refined Feature Learning for Long-Tailed Multi-Label Image Classification (https://arxiv.org/abs/2408.08125)
Comments:
          Accepted by ACM MM 2024

- **What's New**: 이번 연구에서는 Long-Tailed Multi-Label 이미지 분류(LTMLC) 문제를 해결하기 위해 Category-Prompt Refined Feature Learning (CPRFL)이라는 새로운 접근 방식을 제안합니다. 이 방식은 CLIP의 텍스트 인코더를 활용하여 카테고리 간의 의미적 상관관계를 구축하고 각 카테고리에 대한 시각적 표현을 분리합니다.

- **Technical Details**: CPRFL은 pretrained된 CLIP의 임베딩에서 카테고리 프롬프트를 초기화하고, 시각적 특징과의 상호작용을 통해 카테고리별 시각적 표현을 분리합니다. 데이터의 시각-의미 도메인 편향을 완화하기 위해, 점진적인 Dual-Path Back-Propagation 메커니즘을 설계하여 프롬프트에 서서히 맥락 관련 시각 정보를 포함시킵니다. 또한, Asymmetric Loss를 최적화 목표로 사용하여 모든 클래스에서 음성 샘플을 억제합니다.

- **Performance Highlights**: COCO-LT 및 VOC-LT와 같은 두 개의 LTMLC 벤치마크에서 우리의 방법의 효능을 검증하였으며, 실험 결과는 기존의 최신 방법들보다 본 연구의 효과성과 우 superior성을 강조합니다.



### Unsupervised Part Discovery via Dual Representation Alignmen (https://arxiv.org/abs/2408.08108)
Comments:
          Accepted by TPAMI-2024

- **What's New**: 본 연구는 새로운 패러다임을 통해 레이블 없이 파트 전용 주의를 학습하고, 이러한 파트 표현을 사용하여 파트 발견 성능을 개선하는 것에 중점을 두고 있습니다. 이를 위해 다중 기하학적 변환이 적용된 쌍 이미지로부터 파트 표현을 추출하는 PartFormer 모듈을 제안하였습니다.

- **Technical Details**: 이 연구에서는 파트 표현 학습을 위한 기하학적 제약 조건과 의미적 제약 조건을 적용하여, 파트 표현이 관련 영역의 픽셀 표현과 높은 유사성을 가지고 무관한 영역과는 낮은 유사성을 가지도록 유도합니다. 또한, PartFormer는 학습 가능한 여러 개의 파트 임베딩으로 구성되어 있습니다.

- **Performance Highlights**: 다양한 데이터셋(CelebA, AFLW, CUB, DeepFashion, PartImageNet)에서 광범위한 실험이 진행되었으며, 제안된 방법이 기존의 최첨단 방법보다 경쟁력 있고 강건한 성능을 달성했다고 보고되었습니다.



### Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Infer Causal Links Between Siamese Images (https://arxiv.org/abs/2408.08105)
Comments:
          20 pages

- **What's New**: 새로운 다중모달 인과 추론 벤치마크인 MuCR을 제안하여 시각적 단서만으로 VLLMs가 인과 관계를 추론할 수 있는 능력을 평가합니다.

- **Technical Details**: MuCR는 시맨틱 원인-결과 관계를 내포한 시암 이미지(siamese images)를 생성하기 위한 프롬프트 기반 이미지 합성 방식을 도입하여 개발되었습니다. VLLMs의 인과 추론 능력을 종합적으로 평가하기 위해 이미지 수준, 구문 수준, 문장 수준에서 맞춤형 메트릭(metrics)을 개발했습니다.

- **Performance Highlights**: 최신 VLLMs는 다중 모달 인과 추론에서 기대 이하의 성능을 보였으며, 특히 배경 시각 정보에 대한 이해도가 낮았습니다. VLLMs는 다중 이미지 정보를 통해 인과 관계를 추론하는 데 큰 한계를 가지며, 현재의 SOTA 모델조차도 인간 수준의 성과에 미치지 못했습니다.



### When Video Coding Meets Multimodal Large Language Models: A Unified Paradigm for Video Coding (https://arxiv.org/abs/2408.08093)
- **What's New**: 본 연구에서는 Multimodal Large Language Models (MLLMs)를 활용한 Cross-Modality Video Coding (CMVC)라는 새로운 비디오 코딩 패러다임을 제안합니다. 이는 비디오 콘텐츠를 공간적 요소와 모션 요소로 분리한 후, MLLMs를 통해 압축된 표현을 생성합니다.

- **Technical Details**: CMVC는 비디오를 keyframe과 모션 요소로 나눈 후, 이를 각각 텍스트 형식으로 변환하는 과정을 포함합니다. 디코딩 과정에서는 Text-Text-to-Video (TT2V)와 Image-Text-to-Video (IT2V) 모드 활용하여 비디오 재구성을 최적화합니다. 또한, Low-Rank Adaption (LoRA) 기법을 활용하여 프레임 간의 부드러운 모션을 보장하는 프레임 인터폴레이션 모델을 제안합니다.

- **Performance Highlights**: 실험 결과, TT2V 모드는 효과적인 의미적 재구성을 나타내며, IT2V 모드는 경쟁력 있는 시각적 일관성을 보여줍니다. CMVC 파이프라인은 HEVC Class B, C, D, E, UVG 및 MCL-JCV 벤치마크에서 우수한 비디오 재구성을 달성하였습니다.



### OC3D: Weakly Supervised Outdoor 3D Object Detection with Only Coarse Click Annotation (https://arxiv.org/abs/2408.08092)
- **What's New**: 본 논문에서는 LiDAR 기반의 3D 객체 탐지에서 단순 클릭 주석을 통해 포괄적으로 성능을 향상시킬 수 있는 OC3D라는 혁신적인 약한 지도 학습 방법을 제안합니다. 이는 과거의 경계 상자 주석에 의존하지 않고, 3D 포인트 클라우드의 Bird’s Eye View에서 대충의 클릭만으로 학습이 가능합니다.

- **Technical Details**: OC3D는 2단계 전략을 활용하여 동적 및 정적 인스턴스에 대한 Click2Box 및 Click2Mask 모듈을 설계하고, Mask2Box 모듈을 통해 마스크 수준의 가짜 레이블을 경계 상자 레이블로 업데이트합니다. 이 과정에서 주기적인 클릭의 밀도 분석을 통해 빈약한 점군에서 움직이는 객체와 정적인 객체를 구분합니다.

- **Performance Highlights**: OC3D는 KITTI와 nuScenes 데이터셋에서 실험을 수행하였으며, 약한 감독 하에 뛰어난 성능을 달성했습니다. OC3D++를 통해 데이터셋에 단지 0.2%의 주석 비용만으로도 전통적인 완전 감독 방법과 유사한 성능을 발휘합니다.



### HAIR: Hypernetworks-based All-in-One Image Restoration (https://arxiv.org/abs/2408.08091)
Comments:
          13 pages, 4 figures, 6 tables

- **What's New**: HAIR라는 새로운 Hypernetworks 기반의 이미지 복원 방법을 제안합니다. 이 방법은 입력 이미지의 특성에 따라 동적으로 파라미터를 생성하여 다양한 손상 유형을 처리할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: HAIR는 Classifier (C1)와 Hyper Selecting Net (HSN) 두 개의 주요 구성 요소로 이루어져 있습니다. Classifier는 입력 이미지의 손상 정보를 포함하는 Global Information Vector (GIV)를 생성하는 단순 이미지 분류 네트워크입니다. HSN은 GIV를 입력받아 해당 모듈에 대한 파라미터를 출력하는 Fully-connected Neural Network로 기능합니다.

- **Performance Highlights**: HAIR를 Restormer라는 인기 있는 아키텍처에 통합함으로써 다양한 이미지 복원 작업에서 현재의 최첨단 방법들과 비교해 우수하거나 최소한 동등한 성능을 달성했습니다. HAIR는 기존의 모델 구조는 전혀 변경하지 않고도 성능을 크게 향상시킬 수 있습니다.



### ColorMamba: Towards High-quality NIR-to-RGB Spectral Translation with Mamba (https://arxiv.org/abs/2408.08087)
Comments:
          Code is available at this https URL

- **What's New**: 이 연구에서는 색상 변환 작업을 위해 Mamba 모델을 도입한 ColorMamba라는 새로운 백본을 제안합니다. ColorMamba는 NIR에서 RGB로의 스펙트럼 변환에 적합한 첫 번째 Mamba 기반 방법입니다.

- **Technical Details**: ColorMamba는 두 가지 주요 메커니즘인 로컬 컨볼루션 강화(local convolutional enhancement)와 에이전트 어텐션(agent attention)을 포함하여, Visual State Space Blocks (VSSBs)로 발전하였습니다. 이 블록은 글로벌 장기 종속성(global long-range dependencies)과 로컬 맥락(local context)을 모델링할 수 있도록 설계되었습니다. 또한, HSV 색상 예측 서브 네트워크를 통해 복구 프로세스에서 다중 스케일 가이드를 제공합니다.

- **Performance Highlights**: ColorMamba는 최신 방법과 비교하여 PSNR에서 1.02의 개선을 달성했습니다. 이는 ColorMamba가 스펙트럼 변환 작업에서 강력하고 유망한 기반 구조를 제공함을 나타냅니다.



### Single-image coherent reconstruction of objects and humans (https://arxiv.org/abs/2408.08086)
Comments:
          Accepted at AI for 3D Generation, CVPR Workshop

- **What's New**: 이 논문에서는 단일 이미지에서 상호작용하는 객체와 사람들의 전역적으로 일관된 3D 재구성을 얻기 위한 새로운 방법을 소개합니다. 특히, 사람이 여러 상호작용을 하는 복잡한 장면을 효과적으로 처리할 수 있는 최적화 프레임워크를 제안합니다.

- **Technical Details**: 제안된 방법은 단일 RGB 이미지 하나를 입력으로 받아 상호작용하는 인간과 객체를 공간적으로 일관된 방식으로 재구성합니다. 이 과정에서 충돌 손실(collision loss)과 깊이 순서 손실(depth ordering loss)을 사용하여 사람과 객체의 초기 포즈를 최적화합니다. 또한, 강한 가려짐이 있는 객체의 포즈 추정을 개선하는 새로운 6 자유도(6 DOF) 포즈 추정 방법이 도입되었습니다. 이를 통해 인페인팅(image inpainting)을 활용하여 가려진 객체의 분할 마스크를 정제합니다.

- **Performance Highlights**: 제안된 방법은 COCO-2017 데이터셋을 대상으로 한 광범위한 질적 및 양적 평가를 통해, 상호작용하는 사람과 객체가 많은 복잡한 이미지에서 물체 간의 충돌을 상당히 줄이고, 일관된 장면 재구성 결과를 보여 주었습니다.



### Treat Stillness with Movement: Remote Sensing Change Detection via Coarse-grained Temporal Foregrounds Mining (https://arxiv.org/abs/2408.08078)
Comments:
          In Peer Review

- **What's New**: 본 연구에서는 변경 감지 과제를 위해 기존의 bi-temporal 이미지를 기반으로 하는 프레임워크를 재검토하고, Coarse-grained Temporal Mining Augmented (CTMA) 프레임워크를 제안합니다. 이 프레임워크는 시간 정보를 효과적으로 활용하여 더 정확한 변경 예측을 가능하게 합니다.

- **Technical Details**: CTMA 프레임워크는 bi-temporal 이미지를 비디오로 변환하는 것으로 시작합니다. 그리고 temporal encoders를 사용하여 비디오에서 모션 특징을 추출하고, Coarse-grained Foregrounds Augmented Spatial Encoder 모듈을 통해 전 세계 및 지역 정보를 통합합니다. 또한, 모션 인식을 위한 전략과 마스크 증강 방식을 도입하여 최종 변경 예측을 향상시킵니다.

- **Performance Highlights**: 다양한 벤치마크 데이터 세트를 통해 실험한 결과, 제안된 CTMA 프레임워크는 기존 모델에 비해 변경 탐지 성능이 유의미하게 향상된 것으로 나타났습니다.



### MambaMIM: Pre-training Mamba with State Space Token-interpolation (https://arxiv.org/abs/2408.08070)
Comments:
          10 pages, 7 figures

- **What's New**: MambaMIM이라는 새로운 생성적 자기 지도 학습 방법을 소개하며, 선택적 구조 상태 공간 시퀀스 토큰 보간법(S6T)을 통한 pre-training 기법을 개발하였습니다.

- **Technical Details**: MambaMIM에서는 CNN과 Mamba의 하이브리드 모델을 사용하여, 3D CT 데이터셋에서 마스킹 일관성을 유지하며 인코더를 통해 학습합니다. 선택적 구조 상태 공간 시퀀스 토큰 보간법(S6T)을 적용하여 상태 공간 내의 구조 관계를 활용합니다.

- **Performance Highlights**: MambaMIM으로 사전 훈련된 CNN-Mamba 하이브리드 모델이 다른 최첨단 자기 지도 사전 훈련 방법과 아키텍처에 비해 우수한 성능을 나타냈습니다.



### Navigating Data Scarcity using Foundation Models: A Benchmark of Few-Shot and Zero-Shot Learning Approaches in Medical Imaging (https://arxiv.org/abs/2408.08058)
Comments:
          Accepted as an oral presentation in MICCAI 2024 2nd International Workshop on Foundation Models for General Medical AI

- **What's New**: 이 연구에서는 다양한 의료 이미징 도메인에서 사전 훈련된 모델의 Few-Shot Learning (FSL) 및 Zero-Shot Learning (ZSL) 성능을 비교한 첫 번째 대규모 연구입니다. 16개의 모델을 19개의 다양한 의료 이미징 데이터셋을 사용하여 평가하였습니다.

- **Technical Details**: MedIMeta 데이터셋(테스트용으로 구성된 19개의 데이터셋과 10개의 이미징 종류 포함)을 활용하여 FSL 및 ZSL 성능을 측정하였습니다. 연구는 다양한 사전 훈련된 모델(예: ResNet, Vision Transformer)들의 성능을 평가하며, Fine-Tuning과 Linear Probing 전략을 채택했습니다.

- **Performance Highlights**: BiomedCLIP 모델이 매우 적은 훈련 샘플의 경우 평균적으로 가장 좋은 성과를 내며, LAION-2B에서 사전 훈련된 대형 CLIP 모델은 다소 더 많은 훈련 샘플에서 성능이 뛰어났습니다. 또한, ImageNet에서 사전 훈련된 ResNet-18의 경우, 각 클래스당 5개 이상의 훈련 예제에서 비슷한 성능을 보여주었습니다.



### CamoTeacher: Dual-Rotation Consistency Learning for Semi-Supervised Camouflaged Object Detection (https://arxiv.org/abs/2408.08050)
Comments:
          Accepted to ECCV 2024

- **What's New**: CamoTeacher는 세미 슈퍼바이즈드 학습을 기반으로 한 새로운 카모플라주 객체 탐지 프레임워크로, Dual-Rotation Consistency Learning (DRCL)을 통해 псевдо 레이블 노이즈 문제를 효과적으로 해결합니다.

- **Technical Details**: DRCL은 Pixel-wise Consistency Learning (PCL)과 Instance-wise Consistency Learning (ICL) 두 가지 핵심 구성 요소를 사용하여 노이즈를 최소화합니다. PCL은 псевдо 레이블 내의 다양한 부분에 가중치를 할당하여 pixel-level 노이즈를 처리하고, ICL은 인스턴스 간의 일관성을 조정하여 instance-level 노이즈를 처리합니다.

- **Performance Highlights**: CamoTeacher는 CAMO, CHAMELEON, COD10K, NC4K 데이터셋에서 광범위한 실험을 통해 세미 슈퍼바이즈드 학습 방법들과 비교하여 최첨단 성능을 달성했으며, 기존의 풀 슈퍼바이즈드 학습 방법들과도 경쟁할 수 있는 성과를 보였습니다.



### An Advanced Deep Learning Based Three-Stream Hybrid Model for Dynamic Hand Gesture Recognition (https://arxiv.org/abs/2408.08035)
- **What's New**: 본 논문은 손 제스처 인식의 최신 기법으로, RGB 픽셀과 스켈레톤 기반 특징을 융합한 새로운 3스트림 하이브리드 모델을 제안합니다. 이 모델은 다양한 손 형태와 외부 환경 조건을 극복하기 위해 데이터셋을 강화하는 방법을 사용합니다.

- **Technical Details**: 3스트림 하이브리드 모델에서는 첫 번째 스트림에서 pre-trained Imagenet 모듈을 통해 초기 특징을 추출하고, GRU와 LSTM 모듈을 활용하여 이를 개선합니다. 두 번째 스트림에서는 pre-trained ReseNet 모듈을 통해 초기 특징을 추출하고 다양한 GRU 및 LSTM 조합으로 개선합니다. 세 번째 스트림에서는 media pipe를 사용하여 손 자세 키 포인트를 추출하고, 이를 stacked LSTM을 통해 계층적 특징으로 발전시킵니다. 마지막으로, 이 세 가지 특징을 결합하여 최종 출력 예측을 위한 분류 모듈을 적용합니다.

- **Performance Highlights**: 본 연구에서 개발한 모델은 98.35%의 평균 정확도를 기록하며, 기존의 최신 솔루션들과 비교하여 우수한 성능을 보입니다. 이는 높은 수준의 제스처 인식 능력을 입증하며, 실시간 환경에서도 효율적으로 작동할 수 있음을 나타냅니다.



### DIVE: Towards Descriptive and Diverse Visual Commonsense Generation (https://arxiv.org/abs/2408.08021)
Comments:
          19 pages, 10 figuers, EMNLP 2023 (main)

- **What's New**: 이 연구에서는 DIVE라는 새로운 프레임워크를 제안하여 시각적commonsense generation의 기술에서 서술성과 다양성을 향상시키려는 시도가 이루어졌습니다. DIVE는 기존의 모델들이 간과했던 서술적이고 다양한 추론 생성을 중점적으로 다룹니다.

- **Technical Details**: DIVE는 두 가지 주요 방법론, 즉 generic inference filtering과 contrastive retrieval learning을 포함합니다. Generic inference filtering은 특정 이미지의 의미적 농도를 활용하여 일반적인 추론을 제거하는 동시에 균형 잡힌 시각적 commonsense 그래프를 구성합니다. Contrastive retrieval learning은 이미지의 특정 세부 사항을 인식할 수 있도록 모델을 지원합니다.

- **Performance Highlights**: DIVE는 Visual Commonsense Generation(VCG) 데이터셋에서 기존의 최첨단 모델들보다 서술성과 다양성 점수에서 우수한 성능을 나타내며, 인간 수준의 서술성 및 다양성 점수를 달성했습니다. 또한, 생성된 추론의 신뢰성, 서술성 및 다양성에 대한 인간 평가에서는 DIVE가 인간의 평가와도 밀접하게 일치한다는 결과를 보였습니다.



### Adaptive Learning of Consistency and Inconsistency Information for Fake News Detection (https://arxiv.org/abs/2408.08013)
- **What's New**: 최근 소셜 미디어 플랫폼에서의 가짜 뉴스 문제를 해결하기 위해, MFF-Net이라는 새로운 적응형 멀티모달 특징 융합 네트워크를 제안합니다. MFF-Net은 뉴스 정보의 일관성과 불일치 정보를 모두 활용하여 가짜 뉴스를 탐지하는 데 중점을 둡니다.

- **Technical Details**: MFF-Net은 이미지와 텍스트에서 각각 의미 및 전역 특징을 추출하고, 다중 특징 융합 모듈을 통해 모달 간 일관성 정보를 학습합니다. 또한, 단일 모달 특징 필터링 전략을 통해 불일치 정보를 별도로 캡처하여 모달 정보가 쉽게 가려지는 문제를 해결합니다. 이 과정에서 SWIN Transformer(SWIN-T)와 BERT, CLIP과 같은 사전 훈련된 모델을 사용합니다.

- **Performance Highlights**: MFF-Net은 세 가지 공공 뉴스 데이터셋에서 최신 기술보다 우수한 성능을 보임을 실험을 통해 입증하며, 불일치 및 일관성 정보가 모두 탐지 성능에 긍정적인 기여를 함을 보여줍니다.



### MVInpainter: Learning Multi-View Consistent Inpainting to Bridge 2D and 3D Editing (https://arxiv.org/abs/2408.08000)
Comments:
          Project page: this https URL

- **What's New**: 이번 논문에서 소개하는 MVInpainter는 다중 뷰(Multi-view) 2D 인페인팅(inpainting) 작업으로 3D 편집을 재구성하는 혁신적인 모델로, 기존의 NVS 방식의 한계를 극복하고 다양한 실제 상황에서도 적용할 수 있도록 설계되었습니다.

- **Technical Details**: MVInpainter는 입력된 다중 뷰 이미지를 기준으로 부분 인페인팅을 수행하며, 이를 통해 새로운 뷰를 완전히 생성하는 대신 손쉽게 편집할 수 있는 장점을 가지고 있습니다. 이 모델은 카메라 포즈에 대한 의존성을 줄이고, 동작 컴포넌트와 결합된 모션 및 외관 가이드를 통해 크로스 뷰 일관성을 보장합니다. 또한, 슬롯 어텐션(slot attention) 메커니즘을 활용하여 고차원 옵티컬 플로우 피처를 집계하여 카메라 움직임을 제어합니다.

- **Performance Highlights**: 객체 중심과 전방향 데이터셋에서의 충분한 장면 수준 실험을 통해 MVInpainter의 효과가 검증되었으며, 다양한 작업에서 높은 성능을 발휘했습니다. 포함된 작업으로는 다중 뷰 객체 제거, 합성, 삽입 및 교체 등이 있습니다.



### Co-Fix3D: Enhancing 3D Object Detection with Collaborative Refinemen (https://arxiv.org/abs/2408.07999)
- **What's New**: 본 연구에서는 Co-Fix3D라는 새로운 접근 방식을 제안하여 BEV (Bird's Eye View) 표현을 위한 협력적 하이브리드 다단계 병렬 쿼리 생성 메커니즘을 도입했습니다. 이 방법은 Local-Global Feature Enhancement (LGE) 모듈을 통합하여 약한 양성 샘플을 보다 효과적으로 강조하도록 BEV 기능을 개선합니다.

- **Technical Details**: Co-Fix3D는 Discrete Wavelet Transform (DWT)을 활용하여 지역적 영역에서의 노이즈 감소와 특징 세분화를 정확하게 수행하며, 전역 BEV 특징을 포괄적으로 최적화하기 위해 주의(attention) 메커니즘을 포함합니다. 이 방식은 다단계 병렬 처리와 LGE 모듈을 통해 BEV 쿼리의 양을 증가시켜 약한 양성 샘플을 선택할 가능성을 높입니다.

- **Performance Highlights**: Co-Fix3D는 nuScenes 벤치마크에서 모든 기존 모델을 능가하면서 69.1% mAP 및 72.9% NDS를 기록하였고, LiDAR 기반 벤치마크에서 72.3% mAP 및 74.1% NDS를 달성했습니다. 이 연구는 테스트 시 증강(test-time augmentation) 또는 추가 데이터셋에 의존하지 않고도 이러한 성능을 이끌어냈습니다.



### Monte Carlo Path Tracing and Statistical Event Detection for Event Camera Simulation (https://arxiv.org/abs/2408.07996)
Comments:
          10 pages, 7 figures, Presented at ICCP 2024

- **What's New**: 이 논문은 물리적 기반의 Monte Carlo path tracing에 기반한 새로운 이벤트 카메라 시뮬레이션 시스템을 제안하며, 적응형 샘플링(Adaptive Sampling) 기법을 활용합니다. 기존 RGB 카메라를 모방하는 방식과 달리, 로그 조도(Logarithmic Luminance)를 수집하는 시스템을 개발하였습니다.

- **Technical Details**: 제안된 방법은 통계적 기법인 가설 테스트(Hypothesis Testing)를 사용하여 두 다른 시점에서의 로그 조도 차이가 정해진 이벤트 임계값보다 유의미하게 큰지 평가합니다. 중심극한정리(Central Limit Theorem)를 활용하여 로그 조도의 분포를 정규 분포(Normal Distribution)로 모델링하고, Student's t-test를 통해 이벤트 발생 여부를 검증합니다.

- **Performance Highlights**: 제안된 방법은 이벤트가 발생하지 않을 것으로 예측되는 픽셀에 대한 경로 샘플링을 적절히 제거함으로써 기존의 단순 접근 방식에 비해 약 10배 빠른 성능 향상을 보여줍니다. 이 연구는 이벤트 카메라의 물리적으로 정확한 동작을 시뮬레이션하는 최초의 접근 방식이라 할 수 있습니다.



### IIU: Independent Inference Units for Knowledge-based Visual Question Answering (https://arxiv.org/abs/2408.07989)
- **What's New**: 본 논문에서는 기능적으로 독립적인 유닛을 통해 intra-modal 정보를 분해하는 세밀한 다중 모달(dual modality) 추론을 위한 Independent Inference Units (IIU)를 제안합니다. 기존 모델의 한계를 극복하고 모델의 일반화 능력을 향상시키는 것을 목표로 합니다.

- **Technical Details**: IIU는 ATTENTION 기법을 통해 각 의미별로 특정한 intra-modal 단서를 처리합니다. 독립 추론 유닛이 서로 소통을 통해 보완적인 정보를 수집하며, 메모리 업데이트 모듈을 도입하여 추론 과정에서 중요한 의미를 유지하고 중복 정보를 줄입니다. 그래프 구조를 사용하여 질문과 이미지 간의 관련 정보를 시각적으로 나타냅니다.

- **Performance Highlights**: 실험 결과, 제안된 IIU 모델은 기존의 비사전학습된 다중 모달 추론 모델에 비해 3% 성능 향상을 보여 주목할만한 성과를 기록했습니다. 이 모델은 다양한 데이터셋에서도 효과적으로 정보의 분리를 수행하고 해석 가능한 추론 증거를 제공합니다.



### Exploring learning environments for label\-efficient cancer diagnosis (https://arxiv.org/abs/2408.07988)
Comments:
          Submitted to the journal

- **What's New**: 이 논문은 암 진단을 위한 새로운 접근 방식을 제안합니다. 전통적인 지도 학습(SL) 외에도 반지도 학습(Semi-SL) 및 자기 지도 학습(Self-SL) 방법을 이용해 신장암, 폐암 및 유방암을 예측하는 연구를 수행했습니다. 데이터 주석 작업의 부담을 줄이며, 라벨링된 데이터가 부족한 상황에서도 효과적인 암 예측이 가능함을 강조합니다.

- **Technical Details**: 연구에서는 Residual Network-50, Visual Geometry Group-16 및 EfficientNetB0과 같은 세 가지 사전 훈련된 심층 학습 모델을 사용하여, 지도 학습(SL), 반지도 학습(Semi-SL), 자기 지도 학습(Self-SL) 세 가지 학습 환경에서 성능을 비교했습니다. 각 학습 설정은 서로 다른 라벨이 붙여진 이미지 샘플로 구성된 훈련 세트를 기반으로 하며, TS1부터 TS7까지의 다양한 훈련 세트가 사용되었습니다.

- **Performance Highlights**: 반지도 학습(Semi-SL) 환경에서 도출된 결과는 지도 학습(SL) 환경에서 달성된 결과와 높은 일치를 보였습니다. 연구는 최소한의 주석 샘플 수와 낮은 컴퓨팅 비용으로, 반지도 학습이 주석 제한 시나리오에서 효과적인 대안이 될 수 있음을 보여줍니다.



### LLaVA-Surg: Towards Multimodal Surgical Assistant via Structured Surgical Video Learning (https://arxiv.org/abs/2408.07981)
- **What's New**: 이 연구는 Surg-QA라는 새로운 데이터셋을 생성하여, 102,000개의 외과 비디오-지침 쌍을 포함하는 최초의 대규모 외과 비디오 지침-튜닝 데이터셋을 소개합니다. 또한, 멀티모달 대화형 AI 모델인 LLaVA-Surg를 개발하여 외과 비디오에 대한 개방형 질문을 처리할 수 있는 능력을 갖추었습니다.

- **Technical Details**: Surg-QA 데이터셋은 2,201개의 외과 절차에서 44,000개 이상의 외과 비디오 클립에서 추출된 102K 개의 질문-답변 쌍으로 구성되어 있습니다. LLaVA-Surg 모델은 CLIP의 시각 인코더와 Llama를 통합하여 외과 비디오에 대한 지식을 이해하고 대화형으로 답변할 수 있도록 훈련되었습니다.

- **Performance Highlights**: LLaVA-Surg는 기존의 일반 도메인 모델들과 비교하여 외과 비디오 질문-답변 작업에서 뛰어난 성과를 보였으며, 특히 제로샷(Zero-shot) 상황에서 강력한 멀티모달 대화 능력을 입증하였습니다.



### FlashGS: Efficient 3D Gaussian Splatting for Large-scale and High-resolution Rendering (https://arxiv.org/abs/2408.07967)
- **What's New**: 새로운 논문에서 FlashGS를 소개합니다. FlashGS는 효율적인 3D Gaussian Splatting의 차별적 래스터화를 촉진하기 위해 설계된 오픈 소스 CUDA 파이썬 라이브러리입니다. 이 라이브러리는 알고리즘 및 커널 수준의 최적화를 통해 개발되었습니다.

- **Technical Details**: FlashGS는 렌더링 프로세스의 포괄적인 분석을 바탕으로 계산 효율성을 개선하기 위해 다양한 최적화 전략을 포함합니다. 여기에는 중복 제거, 효율적인 파이프라인 처리, 세밀한 제어 및 스케줄링 메커니즘, 그리고 메모리 접근 최적화 등이 포함됩니다. FlashGS는 소비자 GPU에서의 성능 병목현상을 분석하여 새로운 래스터화 워크플로를 제안하며, 고급 기하학적 및 대수적 단순화를 통해 계산 비용을 줄입니다.

- **Performance Highlights**: FlashGS는 모바일 소비자 GPU에서 평균 4배 가속과 함께 메모리 소비는 감소시키며, 고화질 이미지 품질을 유지합니다. 이는 FlashGS가 3D 렌더링 분야에서 매우 유용한 도구로 자리매김하게 합니다.



### Training Spatial-Frequency Visual Prompts and Probabilistic Clusters for Accurate Black-Box Transfer Learning (https://arxiv.org/abs/2408.07944)
Comments:
          ACM Multimedia 2024

- **What's New**: 이 논문에서는 블랙박스(pre-trained models) 상황에서 비전 인식 모델을 위한 새로운 파라미터 효율적(parameter-efficient) 전이 학습 프레임워크를 제안합니다. 이 프레임워크는 두 가지 새로운 훈련 기법을 통합하여 모델의 성능을 향상시킵니다.

- **Technical Details**: 이 연구는 입력 공간(input space)(이미지)를 목표 데이터 분포에 맞추기 위해 시각적 프롬프트(visual prompts)를 생성하고, 확률적 클러스터에 기반한 새로운 훈련 기법을 설계하여 출력 공간의 클래스 분리를 개선합니다.

- **Performance Highlights**: 실험 결과, 본 모델은 다양한 비주얼 인식 데이터셋에서 몇 가지 샷(few-shot) 전이 학습 설정에서 최신 기술 대비 뛰어난 성능을 보여주며, 훈련 및 추론 단계에서 계산 비용을 효과적으로 줄이는 것을 입증했습니다.



### Surgical SAM 2: Real-time Segment Anything in Surgical Video by Efficient Frame Pruning (https://arxiv.org/abs/2408.07931)
Comments:
          16 pages, 2 figures

- **What's New**: SurgSAM-2는 수술 비디오 세분화를 위해 SAM2을 최적화한 모델로, 효율적인 프레임 프루닝(Efficient Frame Pruning, EFP) 메커니즘을 도입하여 실시간 비디오 세분화 성능을 향상시킵니다.

- **Technical Details**: SurgSAM-2는 동적 메모리 관리 시스템을 통합하여 이전 프레임을 선택적으로 유지하고, 코사인 유사도 기반의 점수 매기기 메커니즘을 통해 가장 정보가 많은 프레임만 보존합니다. 이를 통해 컴퓨팅 비용을 줄이고 더 빠른 예측 속도를 달성했습니다.

- **Performance Highlights**: SurgSAM-2는 SAM2 대비 3배 더 높은 FPS를 기록하며, EndoVis17 및 EndoVis18 데이터셋에서 우수한 세분화 정확도를 유지합니다. 이러한 성장은 리소스가 제한된 환경에서도 실시간 수술 비디오 세분화의 가능성을 제시합니다.



### A Deep Features-Based Approach Using Modified ResNet50 and Gradient Boosting for Visual Sentiments Classification (https://arxiv.org/abs/2408.07922)
Comments:
          4 pages, 4 figures, 3 tables, IEEE International Conference on Multimedia Information Processing and Retrieval (MIPR) 2024

- **What's New**: 이번 연구는 단일 모달리티(모드)인 텍스트 기반 감정 분석(Sentiment Analysis, SA)에 중점을 두었던 이전 연구의 한계를 극복하고, Visual Sentiment Analysis (VSA)의 통합적 접근법을 제안합니다. 특히 감정이 포함된 이미지를 분석하는 데 있어 심층 학습(deep learning)과 기계 학습(machine learning) 알고리즘의 융합이 강조됩니다.

- **Technical Details**: 이 연구에서는 수정된 ResNet50을 활용하여 다중 클래스 분류를 위한 심층 특징(deep feature)을 추출하였습니다. 또한, 감정 콘텐츠가 포함된 이미지를 분류하기 위해 gradient boosting 알고리즘이 적용되었습니다. 이를 통해 복잡한 모달 관계를 이해하고 감정을 필요에 맞게 효율적으로 분류할 수 있습니다.

- **Performance Highlights**: CrowdFlower와 GAPED의 두 가지 벤치마크 데이터셋에서 평가한 결과, 제안된 방법이 최신 기술(state-of-the-art)과 비교하여 우수한 성능을 보였습니다. 이는 비주얼 정보와 감정 분석의 융합이 효과적이라는 것을 나타냅니다.



### Persistence Image from 3D Medical Image: Superpixel and Optimized Gaussian Coefficien (https://arxiv.org/abs/2408.07905)
- **What's New**: 이 논문은 기존의 2D 이미지 분석에 주목할 것이 아니라 3D 의료 영상에 대한 혁신적인 3D Topological Data Analysis (TDA) 접근 방식과 슈퍼픽셀(superpixels) 개념을 통합하여 3D 의료 이미지의 특징을 포인트 클라우드 데이터로 변환하는 방법을 제안합니다. 이는 TDA가 대량의 데이터에 대한 저비용의 계산을 가능하게 함을 보여줍니다.

- **Technical Details**: 이 연구는 지속적인 동형체(persistent homology)의 개념을 사용하여 3D 의료 이미지를 처리하고, 최적화된 가우시안 계수(Optimized Gaussian Coefficient)를 통해 3D 볼륨 데이터에서 전체적인 지속성 이미지(Persistence Images)를 효율적으로 생성하는 방법을 소개합니다. 이 과정에서, 픽셀 단위의 제약을 최적화한 가우시안 계수는 의료 이미지 처리의 정확도를 높이는 데 기여합니다.

- **Performance Highlights**: 제안된 3D TDA 방법은 MedMNist3D 데이터셋에서 기존의 전통적인 방법들과 비교했을 때 우수한 성능을 보여주며, 3D 지속적 동형체 기반의 토폴로지 분석을 통한 분류 작업에서 효과적인 모델링 능력을 보여주는 것을 강조합니다.



### Quantum-inspired Interpretable Deep Learning Architecture for Text Sentiment Analysis (https://arxiv.org/abs/2408.07891)
- **What's New**: 본 논문은 양자역학의 원리를 통합한 심층 학습 아키텍처를 제안하여 텍스트 감정 분석의 정확성과 해석 가능성을 향상시킵니다. 특히, 텍스트 표현과 양자역학 원리 간의 공통점을 분석하여 새로운 텍스트 임베딩 레이어를 개발했습니다.

- **Technical Details**: 제안된 모델은 양자 복소수의 원리를 사용하여 텍스트 밀도 행렬을 계산하고, LSTM 네트워크와 SAM을 기반으로 한 기능 추출 레이어를 설계합니다. 이를 통해 2D CNN을 이용한 특징 응축 및 차원 축소가 이루어집니다.

- **Performance Highlights**: 본 연구에서 제안한 QITSA 모델은 여러 텍스트 감정 분석 데이터셋에서 이전 모델들과 비교하여 정확도와 효율성에서 뛰어난 성능을 보여주었으며, 양자역학의 원리를 통합함으로써 어느 정도의 해석 가능성도 달성했습니다.



### MambaVT: Spatio-Temporal Contextual Modeling for robust RGB-T Tracking (https://arxiv.org/abs/2408.07889)
- **What's New**: 본 논문에서는 RGB-T 추적(RGB-T tracking) 분야에 Mamba 기반의 새로운 프레임워크인 MambaVT를 제안합니다. 이 프레임워크는 스페이셜-템포럴(Spatio-temporal) 맥락 모델링을 활용하여 단일 객체 추적의 성능을 향상시킵니다.

- **Technical Details**: MambaVT는 Transformer의 고전적인 쌍 이미지 매칭 방식의 한계를 극복하기 위해 Mamba의 선형 복잡성을 활용하여 긴 시퀀스 모델링 기능을 구현하고, 전체 비디오에서 다중 템플릿 정보를 통합하여 목표 상태를 예측합니다. 핵심 구성 요소로는 길게 있는 프레임을 통합하는 컴포넌트와 짧은 기간의 과거 궤적 프롬프트가 포함됩니다.

- **Performance Highlights**: MambaVT는 LasHeR 데이터셋에서 57.9%의 AUC 성능을 달성하며, 여러 RGB-T 추적 데이터셋에서도 최첨단 성능을 나타냈습니다. 낮은 계산 비용에도 불구하고 우수한 성능을 보이며, 향후 연구를 자극할 수 있는 강력한 기초 모델로 자리매김할 것으로 기대됩니다.



### To Impute or Not: Recommendations for Multibiometric Fusion (https://arxiv.org/abs/2408.07883)
Comments:
          Proc. of IEEE International Workshop on Information Forensics and Security (WIFS), (Nuremberg, Germany), December 2023

- **What's New**: 이 논문은 다양한 멀티모달 생체인식 점수 데이터셋에서 점수 대체 기법을 평가하고, 이러한 기법의 효과성에 영향을 미치는 요인을 조사합니다.

- **Technical Details**: 점수 대체 방법(Imputation techniques)으로는 Univariate와 Multivariate 접근 방식이 있으며, 특히 Multivariate Imputation by Chained Equations (MICE) 방법이 강조됩니다. 또한, Missing Completely at Random (MCAR), Missing at Random (MAR), Missing Not at Random (MNAR) 등의 결측 패턴을 정의하고, 결측 데이터 처리의 중요성을 논의합니다.

- **Performance Highlights**: 결과에 따르면, 결측 점수를 대체하는 것(Imputation)이 결측 점수를 무시하는 것보다 더 나은 성능을 보이며, 클래스 균형이 중요한 역할을 함을 보여줍니다. 특히, 다변량 접근 방식이 점수 간 상관관계가 있을 때 유리하게 작용하며, 단변량 접근 방식은 그럴 때가 덜한 경우에 유리한 것으로 나타났습니다.



### Continuous Perception Benchmark (https://arxiv.org/abs/2408.07867)
- **What's New**: 본 논문에서는 Continuous Perception Benchmark (CPB)라는 새로운 비디오 질문응답 작업을 제안합니다. 이 작업은 기존의 비디오 모델이 키 프레임이나 작은 청크에 국한되던 방식을 넘어, 전체 비디오를 지속적이고 통합적으로 처리하도록 요구합니다.

- **Technical Details**: 논문은 현재의 비디오 모델들이 비디오를 처리하는 두 가지 접근 방식을 분석합니다: 첫 번째는 입력 비디오에서 드문드문 샘플링한 프레임만 처리하는 것과, 두 번째는 비디오를 청크로 나누어 각 청크를 독립적으로 캡션 처리한 후, 이 정보를 종합하는 방식입니다. 이와 달리, CPB는 모델이 전체 비디오 스트림을 연속적으로 분석해야 하므로, 비디오 이해에 있어 더 포괄적이고 지속적인 이해가 필요합니다.

- **Performance Highlights**: 현재의 상용 또는 오픈소스 비디오 모델들이 CPB 작업에서 성능이 부족함을 보여주는 결과가 나타났습니다. 예를 들어, 가장 성능이 좋은 모델도 12%의 질문만을 올바르게 답변할 수 있었습니다. 이는 새로운 기술적 발전이 필요함을 시사합니다.



### Learned Single-Pass Multitasking Perceptual Graphics for Immersive Displays (https://arxiv.org/abs/2408.07836)
- **What's New**: 이번 연구에서는 텍스트 안내 기반의 경량 학습 멀티태스킹(perceptual multitasking) 그래픽 모델을 제안하여, RGB 입력 이미지를 처리하여 인지적으로 향상된 이미지를 출력합니다. 이 모델은 foveated rendering, dynamic range enhancement, image denoising, chromostereopsis와 같은 다양한 인지적 작업을 지원합니다.

- **Technical Details**: 본 모델은 멀티태스킹 U-Net의 병목 구간에서 RGB 이미지와 텍스트 프롬프트의 임베딩을 효율적으로 결합하는 새로운 학습 구성요소를 활용합니다. 이로 인해 단일 추론 단계에서 여러 인지적 작업의 다양한 조합을 수행할 수 있습니다.

- **Performance Highlights**: 제안된 모델은 고유의 단일 추론 단계에서 최신의 작업 특화 방법들과 동등한 품질을 달성하면서도, 더 빠른 추론 속도와 다양한 강도로 효과를 혼합할 수 있는 유연성을 제공합니다.



### Space-scale Exploration of the Poor Reliability of Deep Learning Models: the Case of the Remote Sensing of Rooftop Photovoltaic Systems (https://arxiv.org/abs/2408.07828)
Comments:
          24 pages, 13 figures, 5 tables, manuscript submitted to Environmental Data Science

- **What's New**: 이번 연구는 지붕에 설치된 태양광(PV) 패널에 대한 분류 정확도에 미치는 분포 변화의 영향을 포괄적으로 평가하였다. 새로운 방법론을 제안하여 설명 가능한 인공지능(XAI) 기법과 입력 이미지 및 모델의 결정의 분해를 통해 분포 변화가 딥러닝 모델에 미치는 영향을 이해하고, 데이터 증강 기법을 도입하여 딥러닝 분류기의 강인성을 개선하였다.

- **Technical Details**: 딥러닝 모델이 지붕 태양광 패널을 감지하는 데 민감한 분포 변화(distribution shifts) 문제를 해결하기 위해, BDAPPV111 데이터셋을 사용하여 실증적 기준을 구축하고, XAI 기법을 결합하였다. 이를 통해 딥러닝 모델의 강인성을 개선할 수 있는 간단한 해결책을 제안하였다.

- **Performance Highlights**: 제안된 접근 방식은 기존 방법들보다 뛰어난 성능을 보였으며, 연구 결과는 태양광 시스템을 지도로 매핑하고 전력망에 통합하는 데 도움이 될 수 있다. 코드와 모델 가중치는 GitHub 및 Zenodo에서 제공된다.



### SSRFlow: Semantic-aware Fusion with Spatial Temporal Re-embedding for Real-world Scene Flow (https://arxiv.org/abs/2408.07825)
Comments:
          19 pages,12 figures. arXiv admin note: substantial text overlap with arXiv:2403.07032

- **What's New**: 이번 논문에서는 동적 장면 인식을 위한 3D scene flow 추정의 새로운 접근법인 Dual Cross Attentive (DCA) 방법을 제안합니다. 이는 두 개의 연속적인 포인트 클라우드 간의 의미(context) 기반의 통합 및 정렬을 통해 성능을 개선합니다.

- **Technical Details**: 해당 방법은 Global Fusion Flow Embedding (GF)에 통합되어 전역적 상관관계를 기반으로 흐름 임베딩을 초기화합니다. Spatial Temporal Re-embedding (STR) 모듈은 변형된 객체의 비보정화를 해결하며, Domain Adaptive Losses (DA Losses)를 활용하여 합성 데이터와 실제 LiDAR 스캔 데이터 간의 도메인 차이를 해소합니다.

- **Performance Highlights**: 이 연구는 다양한 데이터셋에서 state-of-the-art (SOTA) 성능을 달성하였으며, 특히 실제 LiDAR 스캔 데이터에서 두드러진 성과를 보여주었습니다.



### Algebraic Representations for Faster Predictions in Convolutional Neural Networks (https://arxiv.org/abs/2408.07815)
Comments:
          Accepted for publication in the proceedings of the 27th International Workshop on Computer Algebra in Scientific Computing (CASC 2024)

- **What's New**: 이 논문에서는 심층 신경망에서 스킵 커넥션 (skip connections) 을 이용한 선형 CNNs(Convolutional Neural Networks) 모델을 단일 레이어 모델로 단순화할 수 있다는 점과, 비선형 모델의 교육 과정에서 스킵 커넥션을 점진적으로 제거하는 방법을 제시합니다.

- **Technical Details**: 스킵 커넥션은 NNs(Neural Networks)의 훈련 과정을 보다 용이하게 만들어주며, 이는 정보가 네트워크를 따라 이동할 때 손실되지 않도록 하고 그래디언트가 쉽게 역전파(backpropagate) 될 수 있게 해줍니다. 논문에서 제시하는 정리는 학습된 LCNs(Linear CNNs)에서의 예측 함수를 사전에 계산할 수 있게 해주며, 이를 통해 예측 시 단일 레이어 퍼셉트론(single-layer perceptron)의 자원만으로 강력한 성능을 갖는 모델을 생성할 수 있습니다.

- **Performance Highlights**: ResNet34의 선형화된 버전에서는 98%의 속도 향상을 관찰하였고, 비선형 CNN의 경우 스킵 커넥션을 제거하면서 22%에서 46%까지 예측 시간 단축을 기록하였습니다.



### Cropper: Vision-Language Model for Image Cropping through In-Context Learning (https://arxiv.org/abs/2408.07790)
- **What's New**: 이번 논문에서는 이미지 크롭핑(image cropping)을 개선하기 위해 새로운 프레임워크인 Cropper를 제안합니다. 이 프레임워크는 대규모 비전-언어 모델(Vision-Language Models, VLM)을 활용하여 자동으로 크롭 후보를 생성하고, 이 컨텍스트 학습을 사용해 예제 수를 줄이며 효율성을 높입니다.

- **Technical Details**: Cropper는 두 단계로 구성됩니다. 첫 번째 단계는 적합한 예제를 자동으로 검색하는 포맷 리트리벌(prompt retrieval) 메커니즘이며, 두 번째 단계는 예측된 크롭의 품질을 반복적으로 개선하는 피드백 기반의 반복 개선(iterative refinement) 전략입니다. 이로 인해 Cropper는 자유형(free-form), 주체 인식(subject-aware), 및 비율 인식(aspect ratio-aware) 크롭핑 작업에 모두 적용할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과, Cropper는 여러 벤치마크와 사용자 연구에서 기존의 최첨단(state-of-the-art) 방법보다 월등한 성능을 보여주었으며, 몇 가지 인-컨텍스트 예제와 별도의 훈련 없이도 탁월한 성과를 보였습니다. Cropper는 다양한 크롭핑 작업을 위한 통합 프레임워크를 제공하여 기존 방법들이 가지지 못한 이점을 제공합니다.



### NeuroPapyri: A Deep Attention Embedding Network for Handwritten Papyri Retrieva (https://arxiv.org/abs/2408.07785)
- **What's New**: 이 논문에서는 고대 그리스 파피루스 문서의 이미지를 분석하기 위해 설계된 새로운 딥러닝 모델인 NeuroPapyri를 소개합니다. 이 모델은 투명성과 해석 가능성 문제를 해결하기 위해 Attention mechanism(주목 메커니즘)을 통합하였습니다.

- **Technical Details**: NeuroPapyri 모델은 이미지에서 쓰여진 텍스트의 라인을 처리하도록 특별히 조정된 Convolutional Neural Networks (CNN)와 여러 개의 Attention heads로 구성된 multi-head attention layer를 통합하여 설계되었습니다. 각 Attention head는 손으로 쓴 이미지 내의 특정 특징에 집중합니다.

- **Performance Highlights**: NeuroPapyri는 문서 검색에서 효과성을 입증하였으며, 고대 문서 분석을 발전시킬 수 있는 잠재력을 보여주었습니다. 실험 결과는 모델의 성능을 평가하는 데 있어서 중요한 역할을 하였습니다.



### Can Large Language Models Understand Symbolic Graphics Programs? (https://arxiv.org/abs/2408.08313)
Comments:
          Technical Report v1 (44 pages, 23 figures, project page: this https URL)

- **What's New**: 이 연구는 LLMs(대형 언어 모델)가 상징적 그래픽 프로그램(symbolic graphics programs)을 이해하는 능력을 평가하는 새로운 기준을 설정하며, 이를 통해 LLM의 시각적 장면에 대한 추론 능력을 평가합니다.

- **Technical Details**: 연구팀은 SGP-Bench라는 새로운 벤치마크를 구축하며, 이는 데이터에 대한 최소한의 인간 노력을 요구하는 프로그램-그래픽 대응(program-graphics correspondence)에 기반합니다. 또한, Symbolic Instruction Tuning (SIT)를 통해 LLM의 지식 이해 능력을 향상시키는 방법을 제안합니다.

- **Performance Highlights**: 현재의 LLM은 벤치마크에서 다양한 성과를 보여주며, 특히 강력한 추론 능력을 가진 모델이 더 나은 성과를 나타냈습니다.



### Understanding the Local Geometry of Generative Model Manifolds (https://arxiv.org/abs/2408.08307)
Comments:
          Pre-print. 11 pages main, 8 pages app., 28 figures

- **What's New**: 본 논문은 사전 훈련된 생성 모델의 학습된 매니폴드의 국소 기하와 생성 성능 간의 관계를 탐구합니다. 특히, 국소 메트릭을 사용하여 모델이 생성하는 샘플의 품질을 평가하는 새로운 접근법을 제안합니다.

- **Technical Details**: 우리는 Continuous Piecewise-Linear (CPWL) 생성 모델의 세 가지 국소 기하적 특성 - 스케일링 (scaling) $
ψ$, 순위 (rank) $
ν$, 복잡성 (complexity) $
δ$ - 을 사용하여 생성 모델의 성능을 분석합니다. 이러한 특성은 주어진 잠재 변수에 대한 샘플의 생성 미적, 아티팩트 및 불확실성과 밀접하게 관련되어 있습니다.

- **Performance Highlights**: 우리의 실험 결과는 대규모 텍스트-이미지 라텐트 확산 모델에서 국소 기하적 특성과 생성 품질, 미적, 다양성 및 메모리 제어가 강한 상관관계를 가짐을 보여줍니다. 또한, 이러한 기하적 정보는 훈련 데이터에 의해 크게 영향을 받으며, 이를 통해 분포의 제어 및 아웃 오브 디스트리뷰션 탐지에 응용할 수 있는 가능성을 제시합니다.



### Rethinking Medical Anomaly Detection in Brain MRI: An Image Quality Assessment Perspectiv (https://arxiv.org/abs/2408.08228)
- **What's New**: 이 논문은 Brain MRI의 이상 탐지에 대한 새로운 접근 방식으로, 이미지 품질 평가(image quality assessment) 관점에서의 재구성 기반 방법을 제안합니다. 특히, 기존의 ℓ1 (L1) 손실 함수 외에 Structural Similarity Index Measure (SSIM) 손실 함수를 결합한 Fusion Quality Loss를 사용하여 더욱 효과적인 재구성 품질 평가를 달성합니다.

- **Technical Details**: 제안된 방법은 Fusion Quality Loss와 Average Intensity Ratio (AIR) 향상 전처리 전략을 결합하여 이상 탐지 성능을 향상시키는 데 초점을 맞추고 있습니다. 데이터 세트에 대한 전처리에서 AIR 비율을 강화하여 이상과 정상 영역 간의 구별을 더욱 명확히 하고, 이미지 품질 평가의 관점에서 이상 탐지를 수행합니다.

- **Performance Highlights**: 제안된 IQA 접근 방식은 BraTS21 (T2, FLAIR) 및 MSULB 데이터 세트에서 기존의 최첨단(state-of-the-art) 방법 대비 DICE 계수(Dice coefficient)가 10% 이상 개선되었으며, 이는 의료 이상 탐지의 이미지 품질 평가의 중요성을 강조합니다.



### Moving Healthcare AI-Support Systems for Visually Detectable Diseases onto Constrained Devices (https://arxiv.org/abs/2408.08215)
Comments:
          6 pages, 5 figures

- **What's New**: 본 연구는 제한된 디바이스에서 AI 도우미를 호스팅하여 헬스케어 지원을 제공하는 'TinyML'의 가능성을 탐구합니다. 이는 특히 인터넷 연결이 불안정한 원거리 지역에서 유용합니다.

- **Technical Details**: 이 파일럿 연구에서는 10,000개의 피부 병변 이미지로 모델을 훈련시켜, 가시적으로 감지 가능한 질병(Visually Detectable Diseases, VDDs)을 분류합니다. 이후 학습된 모델 가중치는 웹캠이 장착된 Raspberry Pi로 전이되어 인터넷 없이 피부 병변을 분류하는 데 사용됩니다.

- **Performance Highlights**: 개발된 프로토타입은 테스트 정확도 78%와 테스트 손실(Test Loss) 1.08을 달성하였습니다.



### Learned Multimodal Compression for Autonomous Driving (https://arxiv.org/abs/2408.08211)
Comments:
          6 pages, 5 figures, IEEE MMSP 2024

- **What's New**: 이 논문에서는 자율 주행을 위한 학습 기반 다중 모드 압축 시스템을 제안하였습니다. 특히 카메라와 LiDAR 모드를 대상으로 하여 3D 물체 감지에 초점을 맞추고, 기존의 단일 모드 압축 방식보다 더 나은 성능을 발휘하는 융합 모달리티의 공동 코딩 기법을 탐구하였습니다.

- **Technical Details**: 제안된 방법은 카메라와 LiDAR 데이터를 활용하여 불필요한 정보와 중복성을 제거하는 것을 목표로 합니다. 연구에서는 두 가지 주요 접근 방식을 사용: 첫째, 융합된 모드의 공동 코딩, 둘째, 한 모드를 먼저 코딩한 후 또 다른 모드를 조건적으로 코딩하는 방식입니다.

- **Performance Highlights**: nuScenes 데이터셋에서의 실험 결과, 융합 모달리티의 공동 코딩 방식이 다른 대안보다 더 높은 성능을 보여주었습니다. 이는 자율 주행 시스템의 데이터 처리 효율성을 크게 향상시킬 것으로 기대됩니다.



### Unlearnable Examples Detection via Iterative Filtering (https://arxiv.org/abs/2408.08143)
Comments:
          Accepted by ICANN 2024

- **What's New**: 본 논문은 Iterative Filtering (IF)이라는 새로운 접근법을 제안하여, 데이터 오염 공격 중 하나인 Unlearnable Examples (UEs)를 식별하고 필터링하는 문제를 해결합니다. 이 방법은 모델이 깨끗한 데이터와 UEs Mixed Dataset에서 훈련되었을 때 효율적으로 UEs를 감지할 수 있음을 보여줍니다.

- **Technical Details**: IF 방법은 모델이 UEs와 깨끗한 데이터에 빠르게 적응하는 것을 이용하여 두 샘플 간의 정확도 차이를 통해 UEs를 판단합니다. 추가 클래스와 반복적인 정교화 과정을 통해 정밀도를 향상시키고, 다양한 공격 유형 및 데이터셋에서의 성능을 검증하여 기존 방법보다 뛰어난 결과를 보입니다.

- **Performance Highlights**: 실험 결과, proposed IF 방법이 다양한 공격 및 오염 비율에 대해 최첨단 탐지 방식보다 우수한 성능을 보였으며, Half Total Error Rate (HTER)를 현저하게 감소시켰습니다. 이를 통해 UEs에 대한 방어 능력이 강화됨을 입증합니다.



### PI-Att: Topology Attention for Segmentation Networks through Adaptive Persistence Image Representation (https://arxiv.org/abs/2408.08038)
- **What's New**: 본 논문에서는 의학 이미지 분석에서 세분화 네트워크가 종종 필요로 하는 형태(topology)에 대한 이해를 기반으로 하는 새로운 topology-aware loss function인 PI-Att을 도입합니다.

- **Technical Details**: PI-Att loss는 ground truth와 예측 맵 간의 형태적 불일치를 최소화하도록 네트워크를 명시적으로 강제합니다. 지속 이미지(persistence image) 표현을 통해 각 맵의 형태를 정량화하며, 이는 세분화 네트워크 손실의 맥락에서 처음으로 이루어진 것입니다. 또한, 학습 성능에 따라 각 epoch의 끝에서 적응적으로 지속 이미지를 계산하는 새로운 메커니즘을 제안합니다.

- **Performance Highlights**: 제안된 PI-Att 손실의 효과는 컴퓨터 단층 촬영 이미지에서 대동맥(aorta) 및 대혈관(great vessel) 세분화에 대한 두 개의 서로 다른 데이터셋에서 입증되었습니다.



### Asteroid: Resource-Efficient Hybrid Pipeline Parallelism for Collaborative DNN Training on Heterogeneous Edge Devices (https://arxiv.org/abs/2408.08015)
Comments:
          Accepted by The 30th Annual International Conference on Mobile Computing and Networking (MobiCom'24)

- **What's New**: 최근의 연구는 분산된 엣지 장치들을 활용하여 On-device Deep Neural Network(DNN) 훈련 시스템인 Asteroid를 제안하며, 다양한 신뢰할 수 있는 엣지 장치 간의 자원을 효과적으로 결합하는 접근 방식을 채택했습니다. 이는 기존의 단일 장치 자원 관리 방식에서 벗어나, 여러 장치의 유휴 자원을 활용하여 훈련 효율성을 높이는 것을 목표로 합니다.

- **Technical Details**: Asteroid는 하이브리드 파이프라인 병렬 처리(hybrid pipeline parallelism)를 채택하여 분산 훈련을 조정하고, 자원 제약 하에서 처리량(maximizing throughput)을 극대화하기 위한 신중한 병렬 처리 계획(parallelism planning)을 수행합니다. 또한, 경량화된 파이프라인 재생(mechanism) 메커니즘을 도입하여 장치 수준의 동적 변동(device-level dynamics)에 대처할 수 있는 견고성을 보장합니다.

- **Performance Highlights**: Asteroid는 기존의 병렬 처리 방법에 비해 최대 12.2배 빠른 훈련 속도를 보여주며, 최신 하이브리드 병렬 처리 기술보다 2.1배 빠른 성능을 발휘합니다. 불규칙적인 장치 종료 및 실패에도 불구하고, 훈련 파이프라인을 기존 방법보다 14배 빠르게 복구할 수 있습니다.



### Analytical Uncertainty-Based Loss Weighting in Multi-Task Learning (https://arxiv.org/abs/2408.07985)
- **What's New**: 이 논문에서는 다중 작업 학습(Multi-task Learning, MTL)에서 개별 작업 손실을 균형 있게 조정하여 성능을 향상시키기 위한 새로운 방법인 Soft Optimal Uncertainty Weighting (UW-SO)를 제안합니다. 이는 Uncertainty Weighting (UW) 방법을 기반으로 하며, 분석적으로 최적의 불확실성 기반 가중치를 계산하여 소프트맥스(softmax) 함수로 정규화합니다.

- **Technical Details**: UW-SO는 손실 손실 기능을 통해 목적에 맞는 실시간 가중치를 학습합니다. UW의 한계를 극복하기 위해 UW의 불확실성 가중치를 최소화하여 각 작업의 손실의 역수로 설정하며, 이는 소프트맥스 함수의 온도 매개변수에 의해 정규화됩니다. 기존의 Scalarization 방법과 비교하여 UW-SO는 단일 하이퍼파라미터를 타겟으로 하여 더 낮은 단계에서 최적화됩니다.

- **Performance Highlights**: 광범위한 데이터셋과 아키텍처에 대한 실험을 통해 UW-SO는 여섯 가지 일반적인 가중치 방식보다 일관되게 우수한 성능을 보였습니다. 연구 결과, 대형 네트워크에서는 가중치 방법의 성능 차이가 줄어들고, 가중치 감소(weight decay)보다 학습률(learning rate) 조정이 더 중요함을 발견했습니다.



### Polaris: Open-ended Interactive Robotic Manipulation via Syn2Real Visual Grounding and Large Language Models (https://arxiv.org/abs/2408.07975)
Comments:
          Accepted by IROS 2024. 8 pages, 5 figures. See this https URL

- **What's New**: 이 논문에서는 테이블탑(Tairo-top) 환경에서의 오픈 엔디드(interactor) 로봇 조작 과제를 다룹니다. 기존 대형 언어 모델(LLM)의 한계를 극복하기 위해 시각적 기초(visual grounding)를 강화한 상호작용 로봇 조작 프레임워크인 Polaris를 도입하였습니다.

- **Technical Details**: Polaris 프레임워크는 GPT-4와 함께 구조적 시각 모델을 사용하여 로봇의 환경을 이해하고, 목표 물체 위치를 정밀하게 추정하는 데 필요한 단계적 방식인 Synthtic-to-Real (Syn2Real) 포즈 추정 파이프라인을 포함합니다. 이 파이프라인은 렌더링된 합성 데이터를 이용해 훈련하고 이를 실제 조작 작업으로 전이합니다.

- **Performance Highlights**: 실제 로봇 실험을 통해 Polaris의 우수한 성능을 입증하였으며, 다양한 조작 작업에 대한 성공률이 높음을 보여주었습니다. 이는 장식된 테이블 이상의 다양한 상황으로 그 가능성을 확장할 수 있는 잠재력을 시사합니다.



### Conditional Brownian Bridge Diffusion Model for VHR SAR to Optical Image Translation (https://arxiv.org/abs/2408.07947)
Comments:
          5 pages, 2 figures, 1 table

- **What's New**: 이 논문에서는 Brownian Bridge Diffusion Model (BBDM)을 기반으로 한 조건부 이미지 간 변환 접근 방식을 제안합니다. 이를 통해 저해상도 데이터를 사용하지 않고 고해상도 SAR 이미지를 광학 이미지로 변환하는 방법을 개선했습니다.

- **Technical Details**: 연구진은 0.5m 해상도의 MSAW 데이터를 활용하여 BBDM을 적용했습니다. BBDM은 확률적 특성을 가진 브라운 조정 과정을 바탕으로 두 도메인 간의 변환을 수학적으로 매핑합니다. 이 방법은 픽셀 공간에서 정보를 선형 보간하여 조건으로 사용하며, 이를 통해 SAR 이미지의 공간 정보를 보존하고 변환 과정을 효과적으로 안내합니다.

- **Performance Highlights**: 실험 결과 BBDM 방법이 Conditional Diffusion Model (CDM) 및 기존 GAN 기반 모델들을 여러 지표에서 초월하는 성과를 거두었음을 보여줍니다. 이로 인해 SAR와 광학 이미지 간의 변환 품질이 유의미하게 향상되었습니다.



### MobileMEF: Fast and Efficient Method for Multi-Exposure Fusion (https://arxiv.org/abs/2408.07932)
- **What's New**: 최근 카메라 디자인과 이미징 기술의 발전으로 스마트폰을 사용하여 고화질 이미지를 캡처할 수 있게 되었습니다. 그러나 디지털 카메라의 제한된 다이내믹 레인지 때문에 조명 불균형이 심한 환경에서 촬영된 사진 품질이 저하되는 문제가 있습니다. 이 문제를 해결하기 위해, 우리는 mobile 환경에 최적화된 encoder-decoder 기반의 새로운 multi-exposure fusion 방법인 MobileMEF를 제안합니다.

- **Technical Details**: MobileMEF는 효율적인 블록 구성으로 모바일 기기의 성능을 최적화한 encoder-decoder 구조를 기반으로 합니다. 모델은 YUV 색 공간으로 입력 이미지를 처리하며, ConvNeXt 기반의 convolutional 블록을 사용합니다. 또한, Single-Scale Fusion(SSF) 우회 모듈을 도입하여 입력의 융합 정보를 모델의 출력으로 전달합니다.

- **Performance Highlights**: MobileMEF는 중급 스마트폰에서 4K 해상도 이미지를 2초 이내에 처리할 수 있는 능력을 갖추고 있습니다. 전체 참조 품질 측정 및 계산 효율성(실행 시간 및 메모리 사용)에 있어서 기존 최첨단 기술보다 우수하며, 이는 하드웨어 제약이 있는 장치에서 실시간 적용이 가능하게 합니다.



### GOReloc: Graph-based Object-Level Relocalization for Visual SLAM (https://arxiv.org/abs/2408.07917)
Comments:
          8 pages, accepted by IEEE RAL

- **What's New**: 이 논문은 로봇 시스템의 객체 수준 재위치 설정(object-level relocalization)을 위한 새로운 방법을 제시합니다. 이 방법은 현재 프레임에서의 객체 감지 결과와 경량 객체 수준 맵에 있는 3D 객체들을 견고하게 연결하여 카메라 센서의 포즈를 결정하는 데 집중합니다.

- **Technical Details**: 제안된 GOReloc 시스템은 그래프 기반 접근 방식을 사용하여 RGB 이미지와 객체 수준 맵 간의 객체 연관성을 강력하게 형성합니다. 객체들은 그래프 노드로 표현되며, 각 노드는 고유한 의미론적(descriptor) 설명자를 사용해 객체 감지를 위한 가능성 있는 연관성을 식별합니다. RANSAC-inspired 전략을 통해 포즈 추정과 객체 연관성을 정제하는 방법을 적용합니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, GOReloc 방법이 기존 방법들에 비해 더 정확한 데이터 연관성을 달성하고 재위치 성공률을 크게 증가시킨 것으로 나타났습니다. 이 시스템은 실시간 처리도 가능하여 실용적인 애플리케이션에 적합합니다.



### DM2RM: Dual-Mode Multimodal Ranking for Target Objects and Receptacles Based on Open-Vocabulary Instructions (https://arxiv.org/abs/2408.07910)
- **What's New**: 이 연구에서는 open-vocabulary (개방형 어휘) 명령어에 따라 일상적인 물체를 특정 가구로 전달할 수 있는 Domestic Service Robot (DSR)을 개발하는 것을 목표로 하고 있습니다. 기존의 방법들은 이미지 검색 환경에서 open-vocabulary 명령어를 활용한 이동 조작(task) 작업을 처리하는 경우가 드뭅니다.

- **Technical Details**: Dual-Mode Multimodal Ranking model (DM2RM)을 제안하며, 이 모델은 단일 모델을 기반으로 두 가지 모드인 목표 객체 및 수납 공간(receptacles)을 위한 이미지를 검색할 수 있게 합니다. Switching Phrase Encoder (SPE) 모듈과 Task Paraphraser (TP) 모듈을 도입하여 예측 대상을 기준으로 임베딩 공간을 전환하고, 명령어를 표준화된 형식으로 패러프레이즈합니다.

- **Performance Highlights**: DM2RM은 실제 건물 환경에서 수집된 데이터셋을 기반으로 평가되었으며, 이미지 검색 설정에서 기존 방법보다 우수한 성능을 보였습니다. DSR 플랫폼에서 fetch-and-carry 작업의 성공률이 82%에 달하며, 이는 zero-shot 전이 환경에서도 달성된 결과입니다.



### Deep Joint Denoising and Detection for Enhanced Intracellular Particle Analysis (https://arxiv.org/abs/2408.07903)
Comments:
          11 pages, 4 figures, 4 tables

- **What's New**: DENODET (denoising-detection network)이라는 새로운 딥 뉴럴 네트워크를 제안하여 이미지의 노이즈 제거와 입자 검출을 동시에 수행합니다. 이 방법은 기존의 입자 검출 및 노이즈 제거 방법보다 높은 성능과 효율성을 제공합니다.

- **Technical Details**: DENODET는 U-Net 아키텍처를 기반으로 하여 여러 해상도의 입력 이미지를 활용하는 한 인코더-두 개의 디코더 구조를 채택합니다. 이 구조는 한 디코더는 이미지의 노이즈 제거를 담당하고, 다른 디코더는 입자 검출을 지향합니다. 인코더와 디코더 간의 스킵 연결을 통해 두 디코더간의 지식 교환을 명시적으로 허용합니다.

- **Performance Highlights**: 상태-of-the-art 입자 검출 방법들과 비교하여, DENODET는 입자 추적 챌린지 데이터셋 및 실제 형광 현미경 이미지 데이터에서 우수한 결과를 달성했습니다.



### A Novel Generative Artificial Intelligence Method for Interference Study on Multiplex Brightfield Immunohistochemistry Images (https://arxiv.org/abs/2408.07860)
- **What's New**: 이번 연구에서는 단일 슬라이드에서 여러 바이오마커(biomarker)를 동시에 분석할 수 있는 다중 밝은 필드 이미징(multiplex brightfield imaging)을 도입하여, 각각의 셀 내에서 바이오마커를 정확히 분석할 수 있도록 하는 새로운 사이클 생성적 적대 신경망(cycle-GAN) 접근 방식을 개발했습니다.

- **Technical Details**: 본 연구에서 사용된 두 가지 대표적인 바이오마커 세트는 cMET-PDL1-EGFR와 CD8-LAG3-PDL1로, 세 가지 바이오마커가 세포막에서 공존할 수 있는 모델입니다. 네트워크의 입력으로는 일반적으로 사용되는 RGB 이미지 대신, 광학 밀도(optical density) 도메인 이미지를 사용했습니다. 이는 합성 단일 중복(singleplex) 이미지의 흐림 현상을 감소시키는 데 도움을 주었습니다.

- **Performance Highlights**: 사이클-GAN 모델은 cMET-PDL1-EGFR 어세이(assay)에 대해 10,800개의 폐, 위 및 대장 이미지를, CD8-LAG3-PDL1 어세이에 대해 3,600개의 대장 이미지를 사용해 검증되었습니다. 시각적 및 정량적 평가 모두 제안된 방법이 수동 리뷰 결과에 비해 효과적이고 효율적임을 보여주었으며, 다양한 다중 어세이에 쉽게 적용할 수 있음을 확인했습니다.



### Language Driven Slice Discovery and Error Rectification (https://arxiv.org/abs/2408.07832)
- **What's New**: 이 논문은 기존의 오류 슬라이스 발견 방법을 탈피하여, Large Language Model (LLM)의 추론 능력을 활용하여 복잡한 오류 패턴을 분석하고 테스트 가능한 가설을 생성하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법인 LADDER (Language Driven slice Discovery and Error Rectification)는 모델의 표현을 언어 정렬 피쳐 공간 (feature space)(예: CLIP)에 투영하여 원본 모델 피쳐 공간의 의미를 보존합니다. 이 과정에서 모델의 오류를 강조하는 문장을 정확하게 검색할 수 있습니다. 이후 LLM은 이러한 문장을 사용하여 오류 슬라이스를 발견하기 위한 가설을 생성하고, 마지막으로 가설을 통해 생성된 그룹 균형 데이터셋을 이용하여 분류 헤드를 미세 조정 (fine-tuning)함으로써 오류를 완화합니다.

- **Performance Highlights**: 모든 과정에서 속성 주석 (attribute annotation)이 필요하지 않으며, 다섯 개의 이미지 분류 데이터셋을 통해 방법이 검증되었습니다. 제공된 코드는 연구 결과의 재현성을 높이는 데 기여합니다.



### Regularized Contrastive Partial Multi-view Outlier Detection (https://arxiv.org/abs/2408.07819)
Comments:
          Proceedings of the 32nd ACM International Conference on Multimedia

- **What's New**: 최근 다중 관점 이상 탐지 방법(Multi-view outlier detection, MVOD)이 크게 발전하였습니다. 본 연구에서는 RCPMOD라는 새로운 방법을 제안하여, 다중 관점 데이터에서 이상 값의 탐지를 개선하고자 합니다. 기존 방법의 한계를 극복하기 위해, 뷰 일관성 정보를 학습하고 이상 값을 일관성 정도에 따라 구별할 수 있는 정규화된 대조적 손실을 도입하였습니다.

- **Technical Details**: 제안된 RCPMOD 방법은 (1) 잠재 이상 메모리 뱅크를 가진 이상 탐지에 민감한 대조적 손실, (2) 지역 구조 상관관계를 포착하기 위한 이웃 정렬 대조적 손실, (3) 이상에 대한 과적합을 방지하기 위한 확산 정규화 손실을 포함합니다. 더욱이, Cross-view Relation Transfer 기술을 통해 이웃의 특성을 기반으로 누락된 뷰 샘플을 쉽게 보정할 수 있습니다.

- **Performance Highlights**: 실험 결과는 RCPMOD가 4개의 벤치마크 데이터셋에서 다양한 설정 하에 현재 최첨단 방법들보다 우수한 성능을 보인다는 것을 보여줍니다.



### An Efficient and Explanatory Image and Text Clustering System with Multimodal Autoencoder Architectur (https://arxiv.org/abs/2408.07791)
- **What's New**: 본 논문에서는 국제 뉴스 사건에 대한 서로 다른 문화적 접근법을 비교하는 새로운 맥락에서 Autoencoders와 LLM 해석기의 확장성을 보여줍니다. CRVAE (Convolutional-Recurrent Variational Autoencoder) 모델이 소개되며, 비디오 프레임의 CNN 인코딩과 관련 텍스트의 LSTM 인코딩을 병렬로 결합하여 멀티 모달 인코딩을 강화합니다.

- **Technical Details**: CRVAE 모델은 고해상도 비디오 프레임의 압축을 위한 Dense CVAE (CVAE with Dense Layers) 구조를 사용합니다. 이 모델은 각 비디오 클러스터에 대한 캡션을 생성하는 BLIP 모델과 협력하여 LLaMA 모델을 통해 최종 태그를 생성합니다. 시스템은 K-means (클러스터링 알고리즘) 최적 클러스터 수 선택을 제외하고 거의 자동화되어 있습니다.

- **Performance Highlights**: COVID-19와 동계 올림픽 두 가지 뉴스 주제에 이 시스템을 적용하여 세 개에서 다섯 개의 주제 클러스터로 요약했습니다. 각 주제는 LLM을 통해 생성된 10개의 문구로 설명되며, 이 작업은 비디오 당 30분 미만의 훈련 시간으로 비용 효율적입니다.



### Perspectives: Comparison of Deep Learning Segmentation Models on Biophysical and Biomedical Data (https://arxiv.org/abs/2408.07786)
- **What's New**: 본 연구는 생물물리학 분야의 소량 트레이닝 데이터세트를 고려하여 다양한 심층 학습 아키텍처의 성능을 비교합니다. 주로 사용되는 아키텍처인 Convolutional Neural Networks (CNNs), U-Nets, Vision Transformers (ViTs), 및 Vision State Space Models (VSSMs) 사이의 비교를 통해 각 모델이 최적의 성능을 발휘하는 조건을 제시합니다.

- **Technical Details**: 이 연구는 일반적으로 생물물리학 실험에서 사용 가능한 소규모 트레이닝 데이터 세트 크기를 고려하여 segmentation 작업에 초점을 맞추고 있습니다. 각 모델은 standard metrics인 정확도(accuracy) 및 특이도(specificity)로 평가되며, 파라미터 수 및 훈련 시간과 같은 실용적인 고려 사항도 강조합니다.

- **Performance Highlights**: 결과적으로, 연구에서는 어떤 단일 모델도 모든 시나리오에서 가장 뛰어난 성능을 보이지 않는다는 점을 밝히며, 생물물리학 프로젝트에 적합한 아키텍처 선택에 대한 기준을 제시합니다.



### A Guide to Similarity Measures (https://arxiv.org/abs/2408.07706)
Comments:
          27 pages

- **What's New**: 이 논문은 다양한 데이터 과학 어플리케이션 도메인에서 널리 사용되는 유사도 측정 방식을 포괄적으로 설명하고 있습니다. 비전문가와 전문가 모두를 위한 유용한 가이드를 제공하며, 각 측정 방식의 정의와 성질을 이해할 수 있도록 구성되었습니다.

- **Technical Details**: 유사도 측정은 데이터 과학의 다양한 응용에서 중심적인 역할을 하며, 여기에는 machine learning, artificial intelligence, information retrieval 등이 포함됩니다. 특정 작업을 위해 적절한 유사도 또는 거리 측정을 선택하는 것이 중요합니다. 이 논문은 50개 이상의 유사도/거리 측정 방식을 제시하며, 그들의 기본 변형 또한 설명합니다. 이 측정 방식들은 주로 벡터 공간 내에서의 내적(inner product)을 기반으로 합니다.

- **Performance Highlights**: 이 논문에서는 Mahalanobis distance와 같은 유용한 거리 측정 방법도 다루며, 이 방법이 다변량 이상 탐지 및 분류에 매우 유용하다고 강조합니다. 유사도 및 거리 측정은 데이터 인스턴스 간의 유사성을 측정하는 기본적인 방법이 되며, 주로 유사한 객체일수록 높은 점수를 부여받는 방식으로 발전합니다.



New uploads on arXiv(cs.AI)

### Benchmarking the Capabilities of Large Language Models in Transportation System Engineering: Accuracy, Consistency, and Reasoning Behaviors (https://arxiv.org/abs/2408.08302)
- **What's New**: 이 논문에서는 최신의 대형 언어 모델(LLMs)인 GPT-4, GPT-4o, Claude 3.5 Sonnet, Claude 3 Opus, Gemini 1.5 Pro, Llama 3, Llama 3.1이 특정 대학 수준의 교통 공학 문제를 해결하는 능력을 탐구합니다.

- **Technical Details**: TransportBench라는 벤치마크 데이터셋을 소개합니다. 이 데이터셋에는 교통 시스템의 계획, 설계, 관리 및 제어와 관련된 다양한 주제의 교통 공학 문제 샘플이 포함되어 있습니다. 이 데이터셋은 인간 전문가가 다양한 상업용 및 오픈소스 LLM의 능력을 평가하기 위해 사용됩니다.

- **Performance Highlights**: 각 LLM의 독특한 강점과 제한 사항을 발견했습니다. 예를 들어, TransportBench 문제를 해결하는 데 있어 Claude 3.5 Sonnet의 인상적인 정확성과 예상치 못한 불일치 행동을 보여주었습니다.



### Conformalized Answer Set Prediction for Knowledge Graph Embedding (https://arxiv.org/abs/2408.08248)
Comments:
          Under Review

- **What's New**: 본 논문에서는 Knowledge Graph Embeddings (KGE) 방법의 한계를 극복하기 위해 Conformal Prediction 이론을 활용하여 답변 집합을 생성하는 방법을 제안합니다. 이 방법은 답변에 대한 확률적 보장을 제공합니다.

- **Technical Details**: KGE는 엔티티와 술어를 벡터 형식으로 매핑하고, Conformal Prediction을 통해 답변 집합을 생성합니다. 이 과정에서는 주어진 쿼리에 맞는 답변이 기존의 학습 데이터와 얼마나 잘 일치하는지를 판단하며, 실제 답변을 포함하는 집합을 구성할 수 있습니다.

- **Performance Highlights**: 4개의 기준 데이터 세트와 6개의 대표적인 KGE 방법을 사용한 실험에서, 제안된 Conformal Prediction 기반의 답변 집합이 통계적 보장과 함께 제공되며, 쿼리의 난이도에 따라 크기가 조절되는 것을 확인했습니다.



### Explaining an Agent's Future Beliefs through Temporally Decomposing Future Reward Estimators (https://arxiv.org/abs/2408.08230)
Comments:
          7 pages + 3 pages of supplementary material. Published at ECAI 2024

- **What's New**: 이번 논문에서는 기존의 Q-value와 state-value 함수를 기반으로 한 사용자 행동 설명을 개선하는 새로운 방법인 Temporal Reward Decomposition (TRD)을 제안합니다. TRD는 에이전트가 기대하는 다음 N개의 보상을 예측함으로써 보다 세부적인 보상 정보를 제공합니다. 이를 통해 보상의 시간적 중요성을 평가하고 에이전트의 행동 선택에 대한 명확한 설명을 제공할 수 있습니다.

- **Technical Details**: Temporal Reward Decomposition (TRD)은 에이전트의 미래 보상 예측 네트워크의 출력을 N+1로 확장하여 각 행동 선택에 대한 예상 보상의 연속적인 값을 제공합니다. TRD는 Q-value와 state-value 함수의 등가성을 유지하면서 행동 선택 시 보상의 즉각성과 양을 보고할 수 있습니다. TRD는 DQN 에이전트를 위한 새로운 손실 함수와 함께 구현됩니다.

- **Performance Highlights**: DQN 에이전트가 Atari 환경에서 TRD를 통합하도록 효율적으로 재훈련되었으며, 성능에 미미한 영향을 주었습니다. TRD를 통해 에이전트의 행동 선택 과정을 설명할 수 있으며, 기대 보상 간의 차이를 시각화하여 중요한 결정 요인을 강조할 수 있습니다.



### Predictive Multiplicity of Knowledge Graph Embeddings in Link Prediction (https://arxiv.org/abs/2408.08226)
Comments:
          Under Review

- **What's New**: 이번 연구에서는 Knowledge Graph Embedding (KGE) 링크 예측의 예측 다중성(predictive multiplicity)를 정의하고, 이로 인해 발생하는 conflicting predictions 문제를 해결하기 위한 접근법을 제안합니다.

- **Technical Details**: 저자들은 KGE 모델들이 링크 예측에서 동일한 성능을 보여줄 수 있으나, 개별 쿼리에 대해 서로 모순된 예측을 하는 현상인 predictive multiplicity를 정량적으로 측정하는 새로운 평가 지표를 도입했습니다. 이를 통해 실험적으로 8%에서 39%의 쿼리에서 conflicting predictions가 나타남을 확인했습니다.

- **Performance Highlights**: 사회적 선택 이론(social choice theory)에서의 투표(voting) 방법을 통해 predictive multiplicity 문제를 해결하려는 접근은 효과적이며, 실험 결과에 따라 66%에서 78%까지의 conflicting predictions 감소가 관찰되었습니다.



### Winning Snake: Design Choices in Multi-Shot ASP (https://arxiv.org/abs/2408.08150)
Comments:
          17 pages, 3 figures, to appear in Theory and Practice of Logic Programming (TPLP), Proceedings of ICLP 2024

- **What's New**: 이 논문은 클링고(clingo)를 사용하여 아케이드 게임 스네이크(snake)를 해결하는 다양한 기술들을 보여줍니다. 이 게임은 해밀토니안 사이클(Hamiltonian Cycle) 문제를 해결함으로써 승리를 보장할 수 있는 흥미로운 예시를 제공합니다.

- **Technical Details**: 논문에서는 여러 개의 멀티샷(multi-shot) 기법을 통해 로직 프로그램을 재활용하는 방법을 설명합니다. 클링고의 멀티샷 솔루션을 활용하여, 사실(facts), 외부 원자(external atoms), 가정(assumptions) 등을 조작함으로써 결과적인 답변 집합(answer sets)에 영향을 미칠 수 있습니다. 각 접근 방식의 장단점을 분석합니다.

- **Performance Highlights**: 다섯 가지 다른 접근 방식의 성능을 실증적 평가를 통해 비교한 결과, 각 방법의 특징을 공유하며 멀티샷 애플리케이션 개발에 대한 통찰을 제공합니다. 이 논문에서 제공하는 소프트웨어는 흥미로운 이미지 출력을 포함하여 온라인에서 접근 가능합니다.



### Model-based Workflow for the Automated Generation of PDDL Descriptions (https://arxiv.org/abs/2408.08145)
- **What's New**: 이 논문은 통합 시스템 및 제품 모델에서 Planning Domain Definition Language (PDDL) 설명의 자동 생성을 위한 광범위한 워크플로우를 제안합니다. 이 워크플로우는 Model-Based Systems Engineering (MBSE)을 활용하여 시스템 및 제품 정보를 조직하고 관리하여 자동으로 PDDL 구문으로 변환합니다.

- **Technical Details**: 제안된 워크플로우는 시스템 모델 분석 및 준비, 시스템 모델 보강, 제품 모델 제공, PDDL 설명 생성의 네 가지 단계로 구성됩니다. 이 과정에서 시스템 구조 및 구성요소를 체계적으로 분석하고, PDDL 도메인에 필요한 동적 및 정적 요소를 정의하며, SysML을 통해 시스템 모델의 내용을 PDDL 형식으로 변환합니다.

- **Performance Highlights**: 이 워크플로우는 항공기 조립 사례에서 검증되었으며, 시스템 모델과 제품 모델의 변경이 PDDL 설명에 빠르게 반영될 수 있도록 하여 효율적이고 적응 가능한 계획 프로세스를 촉진합니다.



### Text2BIM: Generating Building Models Using a Large Language Model-based Multi-Agent Framework (https://arxiv.org/abs/2408.08054)
- **What's New**: 론문에서는 Text2BIM이라는 LLM 기반의 멀티 에이전트 프레임워크를 제안하여 자연어 지침을 통해 3D 건물 모델을 생성할 수 있게 하였습니다. 기존의 복잡한 모델링 명령을 숙지할 필요 없이, 텍스트 입력을 바탕으로 BIM 모델을 보다 직관적으로 표현할 수 있는 방법입니다.

- **Technical Details**: Text2BIM 프레임워크는 여러 LLM 에이전트가 협력하여 사용자 입력을 수동 코드로 변환하며, BIM 저널링 툴의 API를 호출하여 수정 가능한 BIM 모델을 생성합니다. 또한, 규칙 기반 모델 체커를 도입하여 생성된 모델의 품질을 향상시키고, 다중 피드백 루프를 통해 반복적으로 모델을 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 사용자 입력에 명시된 추상 개념과 잘 정렬된 높은 품질의 구조적으로 합리적인 건물 모델을 효율적으로 생성할 수 있음을 보여주었습니다. 또한, 이 프레임워크를 Vectorworks라는 BIM 저자 툴에 통합한 인터랙티브 소프트웨어 프로토타입을 개발하여 새로운 모델링 가능성을 시연하였습니다.



### Solving a Rubik's Cube Using its Local Graph Structur (https://arxiv.org/abs/2408.07945)
- **What's New**: 이번 연구에서는 Rubik's Cube의 해결을 위한 새로운 휴리스틱인 weighted convolutional distance를 제안합니다. 이 방법은 Graph Convolutional Networks (GCNs)를 활용하여 인접 노드의 정보를 더 깊이 활용하고, 최단 경로 탐색을 강화합니다.

- **Technical Details**: 이 연구는 A* search algorithm을 사용하여 Rubik's Cube의 섞인 상태를 해결합니다. weighted convolutional distance는 인접 노드와의 거리 정보를 활용해 상관 계수처럼 작용하는 가중치를 컨볼루션하여 최적의 해결 방안을 찾습니다. 이 접근 방식은 전체 구조를 저장하기 어려운 Rubik's Cube의 대규모 상태 공간에서 지역 구조를 고려하여 보다 효율적인 계산을 가능하게 합니다.

- **Performance Highlights**: deepCubeA와 같은 기존 방법들과 비교했을 때, 제안된 방법이 리소스가 제한된 환경에서도 더 효과적으로 Rubik's Cube의 최단 해결 경로를 찾는 것으로 나타났습니다.



### IReCa: Intrinsic Reward-enhanced Context-aware Reinforcement Learning for Human-AI Coordination (https://arxiv.org/abs/2408.07877)
- **What's New**: 제안된 Intrinsic Reward-enhanced Context-aware (IReCa) 강화 학습 알고리즘은 인간과 AI 간의 비대칭 행동 문제를 해결하기 위해 설계되었습니다. 이 알고리즘은 전통적인 외부 보상에 내재적 보상을 추가하여 희소한 보상을 더 효율적으로 획득할 수 있도록 합니다.

- **Technical Details**: IReCa 알고리즘은 세 가지 주요 특징을 가지고 있습니다: (i) 희소 보상을 탐색하도록 장려하는 내재적 보상, (ii) 해당 희소 상태-행동 쌍을 우선시하여 희소 보상 획득을 개선, (iii) 맥락 인지 가중치를 통해 탐색 및 활용 최적화.

- **Performance Highlights**: IReCa 알고리즘은 Overcooked 환경에서의 시뮬레이션을 통해 누적 보상을 약 20% 증가시키고, 수렴에 필요한 에포크 수를 약 67% 줄이는 성과를 보였습니다.



### CON-FOLD -- Explainable Machine Learning with Confidenc (https://arxiv.org/abs/2408.07854)
- **What's New**: 이 논문에서는 기존의 FOLD-RM 알고리즘을 확장한 CON-FOLD라는 새로운 기계 학습 분류 알고리즘을 소개합니다. CON-FOLD는 분류 작업을 위한 규칙에 대한 확률 기반의 신뢰도 점수를 할당하여 사용자가 예측의 신뢰도를 이해할 수 있도록 합니다.

- **Technical Details**: CON-FOLD는 FOLD-RM의 구조적 특징을 활용한 신뢰 기반 프루닝(pruning) 알고리즘을 사용하여 과적합(overfitting)을 방지하며, 사용자가 기존 지식을 고정된 배경 지식(fixed background knowledge) 또는 수정 가능한 초기 규칙 후보(modifiable initial rule candidates) 형태로 제공할 수 있게 합니다. 또한 우리는 새로운 메트릭인 Inverse Brier Score를 도입하여 생성된 신뢰도 점수의 정확성을 평가합니다.

- **Performance Highlights**: UCI 머신 러닝 리포지토리의 벤치마크 데이터셋을 사용하여 알고리즘의 성능을 입증하였으며, 호주 물리 올림피아드의 짧은 답변 질문에 대한 학생 응답의 평가와 같은 실제 사례에 이 확장을 적용하여 설명력을 강조하였습니다.



### On learning capacities of Sugeno integrals with systems of fuzzy relational equations (https://arxiv.org/abs/2408.07768)
- **What's New**: 이번 논문에서는 시스템의 모호한 관계 방정식에 기초하여 Sugeno 적분(Sugeno integral) 하부의 용량을 학습하는 방법을 소개합니다. 훈련 데이터에 두 개의 방정식 시스템을 결합하여 $	ext{max-min}$ 시스템과 $	ext{min-max}$ 시스템을 형성합니다.

- **Technical Details**: 우리는 Sanchez의 결과를 사용하여 이 두 방정식 시스템의 일관성을 검증하고 직접적으로 훈련 데이터를 나타내는 극단 용량(extremal capacities)을 도출할 수 있음을 보입니다. 방정식 시스템을 특정 기준의 부분 집합으로 축소함으로써, $q$-maxitive 및 $q$-minitive 용량을 도출하기 위한 충분 조건을 제공합니다.

- **Performance Highlights**: 일관성이 없는 경우에도, 최근의 결과를 활용하여 가장 높은 근사 $q$-maxitive 용량과 가장 낮은 근사 $q$-minitive 용량을 얻는 방법을 제시합니다.



### Re-Thinking Process Mining in the AI-Based Agents Era (https://arxiv.org/abs/2408.07720)
- **What's New**: 이 논문에서는 AI 기반 에이전트 워크플로(AgWf) 패러다임을 활용하여 프로세스 마이닝(PM)을 LLM에서 더욱 효과적으로 수행할 수 있도록 제안합니다. AgWf를 통해 복잡한 작업을 간단한 워크플로로 분해하고, 결정론적 도구와 LLM의 도메인 지식을 통합할 수 있습니다.

- **Technical Details**: PM은 이벤트 데이터에서 프로세스관련 통찰을 추출하는 데이터 과학의 한 분야입니다. 이 논문에서 제안하는 AgWf는 결정론적 도구(F)와 비결정론적 AI 기반 작업(T)을 결합하여 작업을 효과적으로 처리합니다. AgWf는 복잡한 과업을 관리 가능한 단위로 나누는 divide-et-impera 원칙에 기반하여 전체 결과의 품질을 높이는데 초점을 맞추고 있습니다.

- **Performance Highlights**: AgWf는 LLM의 성능을 극대화하고, 프로세스 마이닝 작업에서 비결정론적 특성을 통해 유연성과 효율성을 향상시킵니다. CrewAI 구현 프레임워크와 함께 다양한 PM 응용사례를 통해 AgWf의 효과적인 적용을 보여줍니다.



### An Introduction to Reinforcement Learning: Fundamental Concepts and Practical Applications (https://arxiv.org/abs/2408.07712)
- **What's New**: 이 논문은 강화 학습 (Reinforcement Learning, RL)의 핵심 개념과 방법론에 대한 포괄적인 개요를 제공합니다. 독자가 RL 알고리즘을 이해하고 구현할 수 있도록 돕기 위한 다양한 리소스도 소개되고 있습니다.

- **Technical Details**: 강화 학습의 주요 원칙으로는 상태 (State), 행동 (Actions), 정책 (Policies), 보상 (Rewards) 등이 있습니다. 이 논문에서는 Markov Decision Processes (MDPs)를 사용한 문제 해결 접근 방식을 설명하며, Epsilon-Greedy 방법을 통한 탐색 (exploration)과 활용 (exploitation) 간의 균형에 대해서도 논의합니다.

- **Performance Highlights**: 이 논문은 RL 알고리즘, 예를 들어 모델 기반 (model-based) 및 모델 없는 (model-free) 방법론을 포함하고 있습니다. 또한, RL의 다양한 방법과 최신 발전 이력을 통해 독자들이 RL의 실제적 문제 해결 능력을 이해할 수 있도록 돕고 있습니다.



### Can Large Language Models Understand Symbolic Graphics Programs? (https://arxiv.org/abs/2408.08313)
Comments:
          Technical Report v1 (44 pages, 23 figures, project page: this https URL)

- **What's New**: 이 연구는 LLMs(대형 언어 모델)가 상징적 그래픽 프로그램(symbolic graphics programs)을 이해하는 능력을 평가하는 새로운 기준을 설정하며, 이를 통해 LLM의 시각적 장면에 대한 추론 능력을 평가합니다.

- **Technical Details**: 연구팀은 SGP-Bench라는 새로운 벤치마크를 구축하며, 이는 데이터에 대한 최소한의 인간 노력을 요구하는 프로그램-그래픽 대응(program-graphics correspondence)에 기반합니다. 또한, Symbolic Instruction Tuning (SIT)를 통해 LLM의 지식 이해 능력을 향상시키는 방법을 제안합니다.

- **Performance Highlights**: 현재의 LLM은 벤치마크에서 다양한 성과를 보여주며, 특히 강력한 추론 능력을 가진 모델이 더 나은 성과를 나타냈습니다.



### HyperTaxel: Hyper-Resolution for Taxel-Based Tactile Signals Through Contrastive Learning (https://arxiv.org/abs/2408.08312)
Comments:
          Accepted by IROS 2024

- **What's New**: 이번 연구에서는 기존의 low-resolution taxel 기반 촉각 신호의 해상도를 높이기 위해 HyperTaxel이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 기하학적 정보를 활용하여 촉각 센서 데이터를 효과적으로 학습하고 변환하는 방법을 모색합니다.

- **Technical Details**: HyperTaxel 프레임워크는 두 단계로 구성됩니다. 첫 번째 단계에서는 contrastive learning을 이용하여 촉각 신호의 기하학적으로 유익한 표현을 학습합니다. 두 번째 단계에서는 multi-contact localization 전략을 활용하여 low-resolution taxel 신호를 고해상도 3D 표면으로 맵핑합니다. 이 과정에서 joint probability distributions를 활용하여 신호의 불확실성을 줄입니다.

- **Performance Highlights**: 실험 결과, 제안된 표현 방식이 두 가지 기준 모델에 비해 우수한 성능을 보임을 확인하였고, 6D 포즈 추정 및 표면 분류 등의 다양한 downstream task에서 성능 개선이 나타났습니다. 또한, HyperTaxel이 다양한 객체와 센서 구성에 일반화된다는 점에서 정량적 및 정성적 결과가 이를 뒷받침합니다.



### SLCA++: Unleash the Power of Sequential Fine-tuning for Continual Learning with Pre-training (https://arxiv.org/abs/2408.08295)
Comments:
          This paper is an extension of our ICCV 23 paper (arXiv:2303.05118)

- **What's New**: 본 논문은 연속 학습(Local Learning)에서의 사전 학습(Pre-training)을 활용한 새로운 접근 방식인 SLCA++를 제안합니다. SLCA++는 Sequential Fine-tuning(Seq FT)의 힘을 활용하는 강력한 기준 방법으로, 기존 방법의 한계를 극복하고 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: 이 연구에서는 SLCA++라는 프레임워크를 도입하여, 학습률을 선택적으로 줄이고, 분류 레이어(classification layer)를 정렬하는 후처리(post-hoc) 과정을 포함합니다. 또한, Symmetric Cross-Entropy (SCE) 손실 함수를 채택하여 효율성을 높이고, Hybrid Slow Learner(Hybrid-SL) 전략을 통해 파라미터 효율적인 Seq FT 구현을 지원합니다.

- **Performance Highlights**: SLCA++는 이미지 분류 벤치마크에서 다양한 연속 학습 시나리오를 통해 보여준 결과로, Split CIFAR-100, Split ImageNet-R, Split CUB-200, Split Cars-196에서 기존 SOTA(state-of-the-art) 방법보다 45% 이상의 성능 향상을 달성하며, 정량적인 평가를 통해 우수한 성과를 입증합니다.



### Autonomous Behavior Planning For Humanoid Loco-manipulation Through Grounded Language Mod (https://arxiv.org/abs/2408.08282)
Comments:
          Paper accepted by IROS 2024

- **What's New**: 인공지능 로봇이 비구조적인 환경에서 자율적으로 loco-manipulation (이동 및 조작) 작업을 수행할 수 있는 새로운 언어 모델 기반 프레임워크가 제안되었습니다. 이 프레임워크는 로봇이 주어진 텍스트 지침에 따라 행동을 계획하고 저수준의 실행을 수행할 수 있도록 지원합니다.

- **Technical Details**: 이 연구에서는 대형 언어 모델(LLM)의 의미 이해 능력을 활용하여 로봇의 행동을 계획하는 방식이 채택되었습니다. LLM이 생성한 작업 그래프를 기반으로 여러 행동 및 감지 행동의 라이브러리를 통해 로봇은 복잡한 작업을 수행하는 데 필요한 여러 단계를 계획합니다.

- **Performance Highlights**: CENTAURO 로봇을 사용하여 시뮬레이션 및 실제 환경에서 모빌 매니풀이 실험이 진행되었습니다. 다중 모드 센서 데이터를 통합하여 실패를 감지하고 수정하는 절차를 포함하여, 이 접근 방식이 자율 로봇 시스템의 강건성을 향상시키고 작업 성공률을 높이는 데 효과적임을 검증하였습니다.



### InVAErt networks for amortized inference and identifiability analysis of lumped parameter hemodynamic models (https://arxiv.org/abs/2408.08264)
- **What's New**: 본 연구에서는 stiff dynamical systems에 대한 enhanced digital twin 분석을 위한 새로운 neural network 프레임워크인 inVAErt networks를 제안합니다. 이 네트워크는 드물게 발생하는 데이터 문제와 연관된 ill-posed inverse problems를 해결하기 위해 고안되었습니다.

- **Technical Details**: inVAErt networks는 입력에 종속적인 latent space를 학습하여 bijectivity(단사함수 성질)를 복원하는 데 사용됩니다. 이 네트워크는 다양한 솔루션의 manifold를 얻는데 도움을 줍니다. 연구는 six-compartment lumped parameter hemodynamic model에 대한 모델 합성을 수행합니다.

- **Performance Highlights**: inVAErt networks는 synthetic 및 실제 데이터의 매개변수 추정을 통해 시스템의 구조적 비가역성을 해결하며, EHR 데이터셋에서 결측치가 있는 실세계 임상 데이터를 처리할 수 있는 능력을 보여줍니다.



### Snuffy: Efficient Whole Slide Image Classifier (https://arxiv.org/abs/2408.08258)
Comments:
          Accepted for ECCV 2024

- **What's New**: 이번 연구에서는 Whole Slide Image (WSI) 분류를 위한 새로운 MIL-pooling 방법인 Snuffy 아키텍처를 도입하여 제한된 프리트레이닝으로 인한 성능 저하를 완화하고, 지속적인 few-shot pre-training을 가능하게 합니다.

- **Technical Details**: Snuffy는 sparse transformer를 기반으로 한 새로운 MIL-pooling 방법을 제안하며, 이 sparsity 패턴은 병리학에 맞게 조정되어 있습니다. 이 연구는 Snuffy의 sparsity가 보편적 근사기(approximation) 역할을 하며, 레이어 수에 대한 확률적 경계를 제공합니다.

- **Performance Highlights**: Snuffy 아키텍처는 CAMELYON16 및 TCGA 폐암 데이터셋에서 뛰어난 WSI 및 패치(level) 정확도를 달성하며, AUC 0.987, ROI 감지에서 FROC 0.675의 새로운 SOTA를 기록했습니다.



### Derivative-Free Guidance in Continuous and Discrete Diffusion Models with Soft Value-Based Decoding (https://arxiv.org/abs/2408.08252)
Comments:
          The code is available at this https URL

- **What's New**: 본 논문에서는 Diffusion 모델을 사용하여 자연적인 설계 공간을 포착할 뿐만 아니라, 이러한 설계 공간의 자연스러움을 유지하면서 다운스트림 보상 함수(다운스트림 reward functions)를 최적화할 수 있는 새로운 방법을 제안합니다. 이 방법은 두 가지 알고리즘(SVDD-MC 및 SVDD-PM)을 도입하여 기존 모델의 미세 조정 없이 효과적인 결과를 도출할 수 있는 방법입니다.

- **Technical Details**: SVDD(Soft Value-based Decoding in Diffusion models)라는 새로운 기법은 정책(Policy)으로부터 여러 노이즈 상태를 얻고 각 시간 단계에서 가장 높은 가치 함수(value function)를 가지는 샘플을 선택합니다. 이 기법은 사전 훈련된 Diffusion 모델의 표준 추론 절차에 통합되어 있으며, 중간의 노이즈 상태가 미래의 높은 보상으로 이어질 수 있도록 예측합니다. 이 방법은 미세 조정이 필요 없고, 보상 피드백을 활용할 수 있어 다양한 과학 분야에 적용 가능합니다.

- **Performance Highlights**: 본 연구에서 제안한 방법은 이미지 생성, 분자 생성, 그리고 DNA/RNA 서열 생성 등 여러 분야에서 효과적으로 입증되었습니다. 기존의 비차별적인 피드백이나 특성들을 사용하여, 높은 효율성과 자연스러운 결과를 동시에 달성할 수 있는 가능성을 제시합니다.



### A Conflicts-free, Speed-lossless KAN-based Reinforcement Learning Decision System for Interactive Driving in Roundabouts (https://arxiv.org/abs/2408.08242)
Comments:
          15 pages, 12 figures, submitted to an IEEE journal

- **What's New**: 본 논문은 혼합 교통 상황에서 원형 교차로에서 자율주행차(AV)가 안전하고 효율적으로 운전할 수 있도록 하는 학습 기반 알고리즘을 제안합니다. 이 알고리즘은 복잡한 다차량 환경에서 안전하고 효율적인 주행 전략을 학습하기 위해 심층 Q-학습 네트워크(DQN)와 Kolmogorov-Arnold 네트워크(KAN)를 결합합니다.

- **Technical Details**: 제안된 K-DQN 시스템은 행동 검사기(action inspector)와 경로 계획기(route planner)를 통합하여 AV가 주변 환경과의 상호작용에서 발생할 수 있는 충돌을 피할 수 있도록 합니다. 모델 예측 제어(MPC)를 적용하여 주행 동작의 안정성과 정밀성을 보장하며, 시간에 따른 충돌 위험을 평가합니다.

- **Performance Highlights**: 제안된 K-DQN 알고리즘은 인상적인 성능을 보여주었으며, 낮은 충돌 수와 목적지까지의 경로 시간 단축을 달성했습니다. 보상 함수의 원활한 수렴과 다양한 교통 흐름에서의 낮은 변동성을 통해 안정적인 훈련 과정을 유지합니다.



### Evolving A* to Efficiently Solve the k Shortest-Path Problem (Extended Version) (https://arxiv.org/abs/2408.08227)
Comments:
          249 plots in 48 figures, and 81 tables. This is an extended version of the paper Linares López, Carlos and Herman, Ian. 2024. Evolving A* to Efficiently Solve the k Shortest-Path Problem. Proceedings of the European Conference on Artificial Intelligence (ECAI). To appear

- **What's New**: 본 논문은 그래프 G(V, E) 내에서 κ (k)개의 최단 경로를 찾는 새로운 검색 알고리즘인 BELA∗ (Bidirectional Edge Labeling A∗)를 소개합니다. 이 알고리즘은 기존의 A∗ 알고리즘을 기반으로 하여 자연스럽게 발전한 것으로 시간 복잡도가 O (|E| + |V|log{|V|}+k|V|)라는 점에서 기존 알고리즘들과 동일합니다.

- **Technical Details**: BELA∗ 알고리즘은 사이드트랙 엣지 (sidetrack edges)를 활용하여 경로를 두 개의 구성 요소로 분할하는 새로운 개념을 도입합니다. 이 알고리즘은 브루트포스(brute-force) 변형인 BELA0와 함께 이론적 특성과 시간 복잡도를 분석합니다. 실험을 통해 BELA0와 BELA∗가 mA∗ 및 K∗ 알고리즘보다 종종 1~2배 더 빠른 성능을 보임을 입증했습니다.

- **Performance Highlights**: 다양한 실험을 통해 BELA0 및 BELA∗가 mA∗와 K∗는 물론 그들의 브루트포스 변형들보다 폭넓은 문제에서 성능이 현저하게 향상됨을 보여줍니다. 성능 향상은 일부 경우 1~2배, 심지어 그 이상에 이르렀습니다.



### The Dawn of KAN in Image-to-Image (I2I) Translation: Integrating Kolmogorov-Arnold Networks with GANs for Unpaired I2I Translation (https://arxiv.org/abs/2408.08216)
Comments:
          10 pages, 6 Figures, 1 Table

- **What's New**: 이 논문은 Kolmogorov-Arnold Network (KAN)를 사용하여 Generative AI의 이미지-이미지 변환 모델을 개선하는 방법을 소개합니다. KAN은 기존의 Multi-layer Perceptron (MLP) 대신 사용되며, 정보가 더 풍부한 저차원 벡터 표현을 생성할 수 있도록 돕습니다. 또한, KAN-CUT 모델을 제안하여 더 우수한 생성 품질을 달성합니다.

- **Technical Details**: 기존 Contrastive Unpaired Image-to-Image Translation (CUT) 모델의 두 개의 MLP를 KAN으로 효율적으로 대체하여 KAN-CUT 모델을 개발하였습니다. 이 모델은 Gated Linear Units (GLU)를 사용한 활성화 함수를 통합하여 KAN 레이어를 향상시켰습니다.

- **Performance Highlights**: 광범위한 실험 결과, KAN은 Generative AI 중 특히 이미지-이미지 변환에서 GAN과 함께 효과적으로 적용됨을 보여줍니다. KAN-CUT 모델은 목표 도메인에서 고품질 이미지를 생성하는 데 유리함을 입증하였습니다.



### Moving Healthcare AI-Support Systems for Visually Detectable Diseases onto Constrained Devices (https://arxiv.org/abs/2408.08215)
Comments:
          6 pages, 5 figures

- **What's New**: 본 연구는 제한된 디바이스에서 AI 도우미를 호스팅하여 헬스케어 지원을 제공하는 'TinyML'의 가능성을 탐구합니다. 이는 특히 인터넷 연결이 불안정한 원거리 지역에서 유용합니다.

- **Technical Details**: 이 파일럿 연구에서는 10,000개의 피부 병변 이미지로 모델을 훈련시켜, 가시적으로 감지 가능한 질병(Visually Detectable Diseases, VDDs)을 분류합니다. 이후 학습된 모델 가중치는 웹캠이 장착된 Raspberry Pi로 전이되어 인터넷 없이 피부 병변을 분류하는 데 사용됩니다.

- **Performance Highlights**: 개발된 프로토타입은 테스트 정확도 78%와 테스트 손실(Test Loss) 1.08을 달성하였습니다.



### Federated Fairness Analytics: Quantifying Fairness in Federated Learning (https://arxiv.org/abs/2408.08214)
- **What's New**: 이번 연구에서는 Federated Learning (FL) 시스템의 공정성을 측정하기 위한 방법론인 Federated Fairness Analytics를 제안합니다. 이는 기존 문헌에서의 공정성 정의의 한계를 보완하며, FL에 내재된 공정성 문제를 다루기 위해 개발되었습니다.

- **Technical Details**: FL의 모든 훈련 라운드에서 k개의 클라이언트를 선택하여 각 클라이언트가 자신의 로컬 데이터셋을 사용하여 모델을 훈련합니다. 본 연구에서 제시하는 공정성 정의에는 통계적 이질성(statistical heterogeneity), 모델 이질성(model heterogeneity), 통신 이질성(communication heterogeneity), 장치 중심 이질성(device-centric heterogeneity)이 포함됩니다. 제안된 메트릭(metrics)을 통해 FL 시스템의 공정성 성능을 평가할 수 있습니다.

- **Performance Highlights**: 다양한 FL 접근법, 머신러닝 작업 및 데이터 설정을 다양하게 조정하여 실험하였으며, 통계적 이질성과 클라이언트 참여가 공정성에 미치는 영향을 발견했습니다. Ditto와 q-FedAvg와 같은 공정성 인식 접근법이 공정성과 성능 간의 균형을 약간 개선하는 것으로 나타났습니다. 이 연구 결과는 FL 실무자들이 시스템의 공정성에 대한 깊은 통찰력을 제공받을 수 있도록 돕습니다.



### LLM4DSR: Leveraing Large Language Model for Denoising Sequential Recommendation (https://arxiv.org/abs/2408.08208)
- **What's New**: 이번 연구에서는 LLMs(대형 언어 모델)를 활용하여 시퀀셜 추천(Sequential Recommendation)에서 발생하는 노이즈를 제거하는 새로운 접근 방식인 LLM4DSR을 제안합니다. 이는 사용자 히스토리 상호작용 시퀀스의 노이즈를 효과적으로 탐지하고 대체할 수 있도록 LLMs를 조정합니다.

- **Technical Details**: LLM4DSR은 LLMs의 능력을 활용하기 위해 자기 지도 학습(Self-supervised learning) 방식을 통해 미세 조정(Fine-tuning)을 수행합니다. 이를 위해 시퀀스의 일부 항목을 랜덤 선택된 대체 항목으로 교체하는 지침 데이터셋을 구성하고, 이를 기반으로 노이즈 항목을 식별하고 적절한 대체 항목을 제안하도록 LLMs를 훈련합니다. 또한, 불확실성 추정 모듈을 도입하여 식별된 노이즈 항목의 신뢰성을 평가합니다.

- **Performance Highlights**: 폭넓은 실험을 통해 LLM4DSR이 기존의 방법들에 비해 우수한 성능을 발휘함을 입증했습니다. 다양한 추천 모델에 걸쳐 LLM4DSR의 적용 가능성과 성능 향상을 확인했으며, 특히 3개의 데이터셋과 3개의 추천 백본에서 기존 최첨단 노이즈 제거 방법들을 초월하는 성능을 보여주었습니다.



### Scaling Up Natural Language Understanding for Multi-Robots Through the Lens of Hierarchy (https://arxiv.org/abs/2408.08188)
- **What's New**: 이번 연구에서는 인간의 지시를 기반으로 다중 로봇 계획을 촉진하기 위해 작업 계층(task hierarchy)을 활용하는 접근 방법을 제안합니다. 이는 기존의 길고 복잡한 계획에서의 과제를 해결하는 데 기여할 수 있습니다.

- **Technical Details**: 이 연구에서는 Large Language Models (LLMs)를 활용하여, 다수의 문장을 구조화된 언어인 Hierarchical Linear Temporal Logic (LTL)로 변환하는 두 단계의 접근 방식을 제안합니다. 첫 번째 단계에서는 지시사항을 Hierarchical Task Tree로 변환하여 작업 간의 논리적 및 시간적 관계를 포착합니다. 그 후, 특정 도메인에 맞춘 LLM의 세부 조정을 통해 각 과제의 하위 작업을 플랫 LTL 공식으로 변환하고 이를 집계하여 계층적 LTL 명세를 만듭니다.

- **Performance Highlights**: 이 프레임워크는 지시사항과 알고리즘 계획 간의 간극을 해소할 뿐만 아니라, 다중 로봇 작업 계획에서 LLM의 계층적 추론의 잠재력을 보여줍니다. 시뮬레이션과 실제 실험을 통해 우리의 방법이 기존 방법보다 더 복잡한 지시를 처리할 수 있으며, 다중 로봇 작업 배정 및 계획 생성에서 더 높은 성공률과 낮은 비용을 달성한다는 결과를 보여줍니다.



### Your Turn: Real-World Turning Angle Estimation for Parkinson's Disease Severity Assessmen (https://arxiv.org/abs/2408.08182)
- **What's New**: 이 논문은 파킨슨병(Parkinson's Disease, PD) 환자의 실생활에서의 회전 각도를 자동으로 측정하는 심층 학습 기반의 접근 방식을 제안합니다. 이를 위해 비디오에서 3D 골격을 추출하고 엉덩이 및 무릎 관절의 회전을 계산합니다.

- **Technical Details**: 본 연구에서는 Fastpose 및 Strided Transformer와 같은 최신 인간 자세 추정 모델을 활용하여 24명의 피험자(12명의 PD 환자 및 12명의 건강한 대조군)의 1386개 회전 비디오 클립을 분석합니다. 데이터셋은 Turn-REMAP과 Turn-H3.6M으로 구성되어 있으며, 수동으로 수집한 환경에서의 회전 활동을 포함합니다.

- **Performance Highlights**: Turn-REMAP 데이터셋에서 회전 계산 정확도가 41.6%, 평균 절대 오차(Mean Absolute Error, MAE)가 34.7°, 가중 정밀도(Weighted Precision, WPrec) 68.3%를 달성했습니다. 이는 단일 단안 카메라 데이터를 활용한 최초의 연구로 실제 환경에서 PD 환자의 회전을 quantification할 수 있는 방법을 제공합니다.



### Towards flexible perception with visual memory (https://arxiv.org/abs/2408.08172)
- **What's New**: 본 논문에서는 신경망의 대표성을 손상시키지 않고도 유연하게 지식을 추가, 삭제할 수 있는 새로운 방법론을 제시합니다. 이는 기존의 고정된 신경망 아키텍처의 한계를 극복하기 위한 접근입니다.

- **Technical Details**: 제안된 시스템은 이미지 분류 작업을 이미지 유사성(Pre-trained embedding 사용)과 검색(Fast Nearest Neighbor Retrieval 기술 활용)으로 분해하여, 데이터의 추가 및 삭제가 가능한 시각적 기억(visual memory) 체계를 구성합니다. 데이터 추가는 개별 샘플에서 전체 클래스, 심지어 수십억 개의 데이터 규모까지 가능하며, 불필요한 데이터는 unlearning과 memory pruning을 통해 제거할 수 있습니다. 또한, 의사결정 메커니즘은 해석 가능하며, 이는 사용자 개입이 가능합니다.

- **Performance Highlights**: 이 시스템은 ImageNet-1K 및 iNaturalist 데이터셋에서의 성능 비교를 포함하여, 다양한 하이퍼파라미터 설정에 대한 분석 결과를 통해 효율성을 입증합니다. 메모리 프루닝 기술을 통해 잘못된 의사결정에 기여한 이미지를 효과적으로 제거하여 정확도를 향상시키는 등의 성과를 거두었습니다.



### General-purpose Clothes Manipulation with Semantic Keypoints (https://arxiv.org/abs/2408.08160)
- **What's New**: 본 논문에서는 일반화 가능한 의류 조작에 관한 새로운 접근 방식을 제안합니다. 기존의 task-specific 조작 기술이 충분히 일반화되지 못하는 문제를 해결하기 위해 언어 지침을 사용하고, 대형 언어 모델(large language model)을 활용한 계층 학습 방법을 도입하였습니다.

- **Technical Details**: 논문에서 제안하는 계층 학습 방법은 세 가지 계층으로 나뉘며, 각각 계획(planning), 기초(grounding), 행동(action)으로 나눌 수 있습니다. 언어 지침을 기반으로 의류 조작 작업을 지정하고, semantic keypoints를 통해 의류의 기하학적 구조를 온전히 파악하여 조작 방식을 정의합니다. 키포인트 탐지는 masked auto-encoder를 통해 수행되어, 고차원 상태 공간을 효과적으로 다룹니다.

- **Performance Highlights**: 시뮬레이션 실험 결과, 제안된 방법은 기존 방법보다 성공률과 일반화 면에서 뛰어난 성과를 보여주었습니다. 즉, 로봇이 의류 조작 작업에 대해 언어 및 시각 개념을 전이 학습.apply (transfer learning) 할 수 있게 됩니다.



### DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search (https://arxiv.org/abs/2408.08152)
- **What's New**: DeepSeek-Prover-V1.5는 Lean 4에서 정리 증명을 위해 설계된 오픈소스 언어 모델로, DeepSeek-Prover-V1에 비해 훈련 및 추론 프로세스를 최적화하였습니다. 강화 학습 및 새로운 검색 알고리즘(RMaxTS)을 도입하여 증명 생성의 다양성을 확보했습니다.

- **Technical Details**: 모델은 DeepSeekMath-Base에서 미리 훈련되고, DeepSeek-Prover-V1에서 파생된 데이터셋을 사용하여 감독 학습을 통해 조정됩니다. Monte-Carlo tree search를 활용한 새로운 탐색 알고리즘 RMaxTS는 증명 경로의 다양성을 증대시키고, 빠른 정확성을 위해 저희는 truncate-and-resume 메커니즘을 사용합니다. 이 모델은 formal theorem proving의 기초와 자연어와의 정렬을 중시합니다.

- **Performance Highlights**: DeepSeek-Prover-V1.5는 miniF2F 벤치마크에서 63.5%의 통과율을 달성했으며, 높은 학교 수준에서의 새로운 최고 성능을 기록했습니다. ProofNet 기준에서도 뛰어난 성능을 보이며, 검증 세트에서 25.4%의 통과율을 기록했습니다.



### EXPLAIN, AGREE, LEARN: Scaling Learning for Neural Probabilistic Logic (https://arxiv.org/abs/2408.08133)
- **What's New**: 이 연구에서는 신경-기호(Neuro-Symbolic, NeSy) 인공지능 시스템에서 확률적 논리의 강건함과 신경망의 인식 및 학습 능력을 결합하려는 새로운 접근 방식을 제안합니다. 특히, 복잡한 시스템에서 학습을 확장하기 위해 샘플링 기반의 목표 최적화를 채택하였으며, 이를 통해 기존 NeSy 방법들보다 오류 경계가 견고한 학습 방법을 제공하여 더 큰 문제 크기에 적용할 수 있습니다.

- **Technical Details**: 제안하는 EXPLAIN, AGREE, LEARN (EXAL) 방법은 세 가지 단계로 구성됩니다. 첫째, EXPLAIN은 데이터에 대한 설명을 샘플링하여 신경망의 결과를 기반으로 학습 신호를 전달합니다. 둘째, AGREE는 각 설명의 중요성을 신경망의 예측에 따라 재조정합니다. 마지막으로, LEARN은 재조정된 설명을 이용하여 신경망의 학습을 수행합니다. 이 방법은 샘플 수를 선택하여 자원 사용을 조절할 수 있으며, 이러한 샘플의 다양성을 통해 오류를 감소시킬 수 있습니다.

- **Performance Highlights**: EXAL 방법은 MNIST 덧셈 및 Warcraft 경로 찾기 문제를 포함한 두 가지 주요 NeSy 문제에서 다른 최신 NeSy 방법들과 비교하여 실행 시간과 정확성 모두에서 뛰어난 성능을 보였습니다. 이 연구의 실험은 EXAL 방법이 좋은 성능을 달성하기 위해 세 단계가 모두 필요하다는 것을 입증합니다.



### Multimodal Causal Reasoning Benchmark: Challenging Vision Large Language Models to Infer Causal Links Between Siamese Images (https://arxiv.org/abs/2408.08105)
Comments:
          20 pages

- **What's New**: 새로운 다중모달 인과 추론 벤치마크인 MuCR을 제안하여 시각적 단서만으로 VLLMs가 인과 관계를 추론할 수 있는 능력을 평가합니다.

- **Technical Details**: MuCR는 시맨틱 원인-결과 관계를 내포한 시암 이미지(siamese images)를 생성하기 위한 프롬프트 기반 이미지 합성 방식을 도입하여 개발되었습니다. VLLMs의 인과 추론 능력을 종합적으로 평가하기 위해 이미지 수준, 구문 수준, 문장 수준에서 맞춤형 메트릭(metrics)을 개발했습니다.

- **Performance Highlights**: 최신 VLLMs는 다중 모달 인과 추론에서 기대 이하의 성능을 보였으며, 특히 배경 시각 정보에 대한 이해도가 낮았습니다. VLLMs는 다중 이미지 정보를 통해 인과 관계를 추론하는 데 큰 한계를 가지며, 현재의 SOTA 모델조차도 인간 수준의 성과에 미치지 못했습니다.



### OC3D: Weakly Supervised Outdoor 3D Object Detection with Only Coarse Click Annotation (https://arxiv.org/abs/2408.08092)
- **What's New**: 본 논문에서는 LiDAR 기반의 3D 객체 탐지에서 단순 클릭 주석을 통해 포괄적으로 성능을 향상시킬 수 있는 OC3D라는 혁신적인 약한 지도 학습 방법을 제안합니다. 이는 과거의 경계 상자 주석에 의존하지 않고, 3D 포인트 클라우드의 Bird’s Eye View에서 대충의 클릭만으로 학습이 가능합니다.

- **Technical Details**: OC3D는 2단계 전략을 활용하여 동적 및 정적 인스턴스에 대한 Click2Box 및 Click2Mask 모듈을 설계하고, Mask2Box 모듈을 통해 마스크 수준의 가짜 레이블을 경계 상자 레이블로 업데이트합니다. 이 과정에서 주기적인 클릭의 밀도 분석을 통해 빈약한 점군에서 움직이는 객체와 정적인 객체를 구분합니다.

- **Performance Highlights**: OC3D는 KITTI와 nuScenes 데이터셋에서 실험을 수행하였으며, 약한 감독 하에 뛰어난 성능을 달성했습니다. OC3D++를 통해 데이터셋에 단지 0.2%의 주석 비용만으로도 전통적인 완전 감독 방법과 유사한 성능을 발휘합니다.



### AgentCourt: Simulating Court with Adversarial Evolvable Lawyer Agents (https://arxiv.org/abs/2408.08089)
- **What's New**: 이번 연구에서는 AgentCourt라는 시뮬레이션 시스템을 소개합니다. AgentCourt는 법정 절차를 전면적으로 시뮬레이션하며, 변호사 에이전트들이 사건을 논의하고 법적 기술을 강화하도록 돕습니다.

- **Technical Details**: AgentCourt는 여러 역할(법관, 원고 변호사, 피고 변호사 등)을 가진 자율 에이전트로 구성되어 있으며, 이는 Large Language Models (LLMs) 기반으로 작동합니다. 이 시스템은 대립적 진화 접근 방식을 통해 변호사 에이전트들이 경험을 쌓고 사례를 통해 지속적으로 학습토록 지원합니다.

- **Performance Highlights**: 연구 결과, AgentCourt에서 1,000건의 법적 사례를 처리한 후, 진화된 변호사 에이전트는 법적 작업 처리 능력이 일관되게 향상되었습니다. 법률 전문가 panel의 평가 결과, 이들은 응답성 및 전문성에서 주목할 만한 발전을 보였습니다.



### An Efficient Replay for Class-Incremental Learning with Pre-trained Models (https://arxiv.org/abs/2408.08084)
- **What's New**: 본 연구에서는 Weight Balancing Replay (WBR)라는 새로운 클래스 증가 학습(class-incremental learning) 방법을 제안합니다. WBR은 각 클래스의 단일 샘플 유닛을 메모리에 유지하고 간단한 그래디언트 제약을 적용하여 잊어버림(catasrophic forgetting)을 극복하는 기술입니다.

- **Technical Details**: WBR은 잊어버림 모델과 정상 모델 간의 매개변수 차이를 신속하게 조정하여 학습 중 파라미터 변화에 기초하여 새로운 과제와 오래된 과제 간의 가중치를 균형있게 유지합니다. 학습 진행 중에 각 과제를 대표하는 단일 메모리 벡터만을 유지하며, 이 벡터는 과제 내 모든 샘플을 샘플링하여 생성됩니다.

- **Performance Highlights**: 실험 결과, WBR은 이전의 동시 학습 벤치마크에서 경쟁력 있는 성능을 낼 수 있음을 입증하였습니다. 특히, 대규모 모델에 대한 높은 훈련 비용을 줄일 수 있는 장점이 있으며, 복잡한 장치 환경에서도 효과적으로 활용될 수 있습니다.



### Confidence-weighted integration of human and machine judgments for superior decision-making (https://arxiv.org/abs/2408.08083)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)과 인간이 팀을 이루어 협력할 경우, 각 개별 구성원이 평균적으로 성능이 낮은 인간이다 할지라도 팀 전체의 성과를 향상시킬 수 있음을 보여줍니다. 이를 위해 Bayesian 접근법을 간소화하고 확장하여 신뢰도 가중치를 포함한 판단 통합을 가능하게 하는 로지스틱 회귀 프레임워크를 제안합니다.

- **Technical Details**: 우리는 BrainBench라는 벤치마크를 사용하여 LLM과 인간이 신경과학 연구 결과를 예측하는 능력을 평가했습니다. 100개의 테스트 사례에서 원본 초록과 수정된 초록의 선택을 통해 각 팀원의 신뢰도를 평가하며, 기계와 인간의 판단을 통합하여 최적의 성과를 낼 수 있는 조건인 신뢰도 보정(calibration)과 분류 다양성(classification diversity)을 충족합니다. 또한, LLM의 경우 PPL(Perplexity) 점수를 사용하여 선택합니다.

- **Performance Highlights**: 신경과학 예측 작업에서 LLM이 인간보다 우수하더라도, 인간과 LLM의 조합이 팀 성과를 일관되게 개선하는 것을 보여주었습니다. 이는 이전 연구들에서 확인된 인간-기계 팀의 보완성(complementarity)을 뒷받침합니다. 이 결과는 단순하고 효과적인 인간-기계 협업 전략을 통해 생산적인 협업으로 이어질 것이라 기대하고 있습니다.



### Treat Stillness with Movement: Remote Sensing Change Detection via Coarse-grained Temporal Foregrounds Mining (https://arxiv.org/abs/2408.08078)
Comments:
          In Peer Review

- **What's New**: 본 연구에서는 변경 감지 과제를 위해 기존의 bi-temporal 이미지를 기반으로 하는 프레임워크를 재검토하고, Coarse-grained Temporal Mining Augmented (CTMA) 프레임워크를 제안합니다. 이 프레임워크는 시간 정보를 효과적으로 활용하여 더 정확한 변경 예측을 가능하게 합니다.

- **Technical Details**: CTMA 프레임워크는 bi-temporal 이미지를 비디오로 변환하는 것으로 시작합니다. 그리고 temporal encoders를 사용하여 비디오에서 모션 특징을 추출하고, Coarse-grained Foregrounds Augmented Spatial Encoder 모듈을 통해 전 세계 및 지역 정보를 통합합니다. 또한, 모션 인식을 위한 전략과 마스크 증강 방식을 도입하여 최종 변경 예측을 향상시킵니다.

- **Performance Highlights**: 다양한 벤치마크 데이터 세트를 통해 실험한 결과, 제안된 CTMA 프레임워크는 기존 모델에 비해 변경 탐지 성능이 유의미하게 향상된 것으로 나타났습니다.



### A Survey on Integrated Sensing, Communication, and Computation (https://arxiv.org/abs/2408.08074)
- **What's New**: 6G의 발전은 기존의 데이터 중심 서비스에서 벗어나 모든 것을 연결하고 지능화하는 시대를 예고하고 있습니다. 이 성과를 이루기 위해 Sensing, Communication, Computation의 통합이 필요하며, 이를 통해 정보의 수집, 공유, 처리 및 의사결정을 매끄럽게 수행할 수 있습니다.

- **Technical Details**: 통합 감지, 통신 및 계산(Integrated Sensing, Communication, and Computation, ISCC) 접근 방식은 기존의 통합 기술들인 ICC(Integrated Communication and Computation), ISC(Integrated Sensing and Computation), ISAC(Integrated Sensing and Communication)의 한계를 보완하기 위한 새로운 기술을 개발함으로써 정보의 처리 성능을 향상시킵니다. ISCC는 복합 작업에서의 성능 저하를 방지하기 위해, 자원 경쟁을 고려한 알고리즘과 효율적인 네트워크 자원 관리 전략을 제안합니다.

- **Performance Highlights**: ISCC의 도입으로 인해 스마트 시티, 자율주행, 원격 의료와 같은 다양한 분야에서 정교하고 혁신적인 응용 프로그램과 서비스의 가능성이 열릴 것으로 기대됩니다. 이 통합 접근 방식은 미래의 6G 네트워크에서 지능형 서비스를 제공하는 데 필수적입니다.



### RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation (https://arxiv.org/abs/2408.08067)
Comments:
          Under Review

- **What's New**: 이 논문에서는 RAG 시스템을 평가하기 위한 새로운 프레임워크인 RAGChecker를 제안합니다. 이 프레임워크는 리트리버와 제너레이터 모듈 각각에 대한 상세한 진단 지표를 포함하여 RAG 시스템의 성능을 포괄적으로 평가할 수 있도록 설계되었습니다.

- **Technical Details**: RAGChecker는 클레임 수준의 함의 검사를 기반으로 하여 응답 및 정답과 관련된 클레임을 추출하고 이를 다른 텍스트와 비교하여 미세한 평가를 수행합니다. 이 방식은 RAG 시스템의 리트리버 및 제너레이터를 모두 평가할 수 있도록 해줍니다. RAGChecker의 지표에는 전체 성능 지표, 리트리버의 진단 지표, 제너레이터의 진단 지표가 포함됩니다.

- **Performance Highlights**: RAGChecker는 8개의 최신 RAG 시스템을 평가하였고, 그 결과는 RAG 아키텍처의 디자인 선택에서의 통찰력 있는 패턴과 상충점을 드러냈습니다. 메타 평가를 통해 RAGChecker는 기존의 평가 지표보다 인간의 판단과의 상관관계가 훨씬 더 우수함을 입증했습니다.



### SPEED: Scalable Preprocessing of EEG Data for Self-Supervised Learning (https://arxiv.org/abs/2408.08065)
Comments:
          To appear in proceedings of 2024 IEEE International workshop on Machine Learning for Signal Processing

- **What's New**: 본 논문은 Electroencephalography (EEG) 데이터를 효율적으로 처리하기 위한 새로운 Python 기반의 전처리 파이프라인인 SPEED를 소개합니다. 이 파이프라인은 자기 지도 학습(self-supervised learning, SSL)을 위해 최적화되어 있으며, 대규모 데이터를 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: SPEED 파이프라인은 EEG 데이터의 일관된 형식을 보장하기 위해 여러 전처리 구성 요소를 통합하고, 표준화된 채널 이름, 불필요한 채널 삭제, 60초 창(segment) 생성 등 간단하면서도 필수적인 단계를 포함하고 있습니다. 후속 처리 단계에서는 고유한 품질 평가 방법을 통해 데이터를 효율적으로 관리합니다.

- **Performance Highlights**: 실험 결과 SPEED를 사용한 경우 자기 지도 학습 모델의 대조 정확도(contrastive accuracy)가 향상되고, 다운스트림 태스크에서의 성능이 원시 데이터(raw data)로 훈련했을 때보다 증가함을 보여줍니다. 특히, Temple University Hospital EEG Corpus (TUEG) 데이터셋을 활용하여 이 연구가 수행되었습니다.



### Maximally Permissive Reward Machines (https://arxiv.org/abs/2408.08059)
Comments:
          Paper accepted for publication at the European Conference on Artificial Intelligence (ECAI) 2024

- **What's New**: 이번 논문에서는 'maximally permissive' 보상 기계(MPRM)를 새롭게 제안하여, 목표에 대한 부분 순서 계획들을 기반으로 보상 기계를 생성하는 접근 방식을 설명합니다. 이는 기존의 단일 계획에 기반한 보상 기계보다 더 높은 보상을 얻을 수 있다는 것을 이론적으로 증명합니다.

- **Technical Details**: 보상 기계는 Mealy 기계로 구성되며, 상태는 작업 내의 '단계'를, 전이는 환경 내에서의 높은 수준 사건의 관측을 나타냅니다. 이 논문에서 제안된 알고리즘은 계획 작업에 대한 부분 순서 계획의 집합을 계산하고, 이를 통해 MPRM을 합성하는 구성을 제공합니다. MPRM을 사용한 최적 정책의 기대 보상은 단일 부분 순서 계획에서 합성된 RM에서 배운 정책의 기대 보상과 같거나 높습니다.

- **Performance Highlights**: CraftWorld 환경에서 세 가지 작업을 통해 MPRM을 평가하였으며, 그 결과 MPRM을 사용한 에이전트가 단일 부분 순서 계획 또는 단일 순차 계획 기반의 RM보다 높은 보상을 획득함을 보여주었습니다.



### Navigating Data Scarcity using Foundation Models: A Benchmark of Few-Shot and Zero-Shot Learning Approaches in Medical Imaging (https://arxiv.org/abs/2408.08058)
Comments:
          Accepted as an oral presentation in MICCAI 2024 2nd International Workshop on Foundation Models for General Medical AI

- **What's New**: 이 연구에서는 다양한 의료 이미징 도메인에서 사전 훈련된 모델의 Few-Shot Learning (FSL) 및 Zero-Shot Learning (ZSL) 성능을 비교한 첫 번째 대규모 연구입니다. 16개의 모델을 19개의 다양한 의료 이미징 데이터셋을 사용하여 평가하였습니다.

- **Technical Details**: MedIMeta 데이터셋(테스트용으로 구성된 19개의 데이터셋과 10개의 이미징 종류 포함)을 활용하여 FSL 및 ZSL 성능을 측정하였습니다. 연구는 다양한 사전 훈련된 모델(예: ResNet, Vision Transformer)들의 성능을 평가하며, Fine-Tuning과 Linear Probing 전략을 채택했습니다.

- **Performance Highlights**: BiomedCLIP 모델이 매우 적은 훈련 샘플의 경우 평균적으로 가장 좋은 성과를 내며, LAION-2B에서 사전 훈련된 대형 CLIP 모델은 다소 더 많은 훈련 샘플에서 성능이 뛰어났습니다. 또한, ImageNet에서 사전 훈련된 ResNet-18의 경우, 각 클래스당 5개 이상의 훈련 예제에서 비슷한 성능을 보여주었습니다.



### COTODE: COntinuous Trajectory neural Ordinary Differential Equations for modelling event sequences (https://arxiv.org/abs/2408.08055)
- **What's New**: 본 논문은 Gaussian Process를 사용하여 사건 순서의 시간 발자국을 수집하는 새로운 접근 방법을 제시합니다. 일반적인 Neural ODE 모델과 결합하여 이벤트 간 발생하는 상태 변화를 연속적으로 모사합니다.

- **Technical Details**: 우리 연구에서는 이벤트를 별개의 현상으로 보지 않고, 대신 Gaussian Process에 의한 관찰로 파악합니다. 이를 통해 사건 간의 불확실성을 평가하고, 이를 토대로 부정적 피드백 메커니즘을 개발하였습니다. 이 방법은 Neural ODE 프레임워크 내에서 이벤트 임베딩의 GP 보간을 통해 연속적인 숨겨진 경로를 생성합니다.

- **Performance Highlights**: 우리는 20% AUROC 개선된 성능을 통해 COTODE 모델이 기존 RNN 및 ODE 기반 방법들보다 우수한 결과를 보여주는 것을 확인했습니다. 다양한 실험과 비교를 통해 우리의 접근 방법이 긴 시퀀스 데이터셋에서 다양한 적용 가능성을 입증했습니다.



### The Clever Hans Effect in Unsupervised Learning (https://arxiv.org/abs/2408.08041)
Comments:
          12 pages + supplement

- **What's New**: 본 연구는 자율 학습(Unsupervised Learning)에서 Clever Hans 효과가 광범위하게 발생하는 것을 처음으로 증명합니다. 이 현상은 모델이 올바른 예측을 하더라도 잘못된 이유에 기반한 예측이라는 것을 경고합니다.

- **Technical Details**: 자율 학습을 통해 Covid-19를 X-ray 스캔에서 탐지하는 대표적인 작업에서 Clever Hans ефfect을 조사했습니다. Explainable AI 기법인 BiLRP를 사용하여, 모델이 유사한 사례를 탐지하기 위한 적절한 전략이 아닌 Clever Hans 전략을 사용하고 있는지를 평가했습니다.

- **Performance Highlights**: 모델은 Covid-19 감염 여부를 구분하는 데 88.6%의 클래스 균형 정확도를 달성했지만, 데이터의 이질성으로 인해 GitHub 하위 그룹에서 40%의 허위 긍정률(FPR)을 보였습니다. 그러나 감독 모델에 비해 자율 학습 모델은 하위 그룹에서의 예측 일관성이 떨어져 실제 응용에서의 사용이 어렵다는 결과를 도출했습니다.



### Adaptive User Journeys in Pharma E-Commerce with Reinforcement Learning: Insights from SwipeRx (https://arxiv.org/abs/2408.08024)
Comments:
          Presented at the Third Workshop on End-to-End Customer Journey Optimization at KDD 2024 (KDD CJ Workshop '24), August 26, Barcelona, Spain

- **What's New**: 이번 논문에서는 개인화된 사용자 여정을 통해 의료 디지털 도구에서의 사용자 경험을 향상시키는 강화 학습(리인포스먼트 러닝) 플랫폼을 소개합니다. 특히, 동남아시아에서 약사들을 위한 가장 인기 있는 앱인 SwipeRx를 연구 사례로 하여 이 플랫폼의 효용성을 보여줍니다.

- **Technical Details**: 플랫폼은 소프트웨어 개발 키트(SDK)와 사용자 인터페이스(UI) 분석, 모델 관리, 행동 및 실험 기능으로 구성됩니다. 데이터 파이프라인을 통해 수집된 사용자 행동 데이터를 실시간 구매 이력 및 앱 내 상호작용을 기반으로 개인화된 제품 추천에 적용하며, 이는 약국의 장바구니 크기를 유의미하게 증가시키는 결과를 보여줍니다. 머신 러닝(ML) 엔진은 예측 모델링, 추천 시스템, 알고리즘 결정 서비스로 구성되어 있으며, 사용자 여정을 효과적으로 개선하기 위한 개인화된 개입을 제공합니다.

- **Performance Highlights**: 이 플랫폼을 통한 실험 결과는 개인화된 추천이 이루어진 후 약국의 장바구니에 유의미한 증가를 이끌어냈으며, 이는 의료 공급망 관리, 건강 근로자 역량 강화 및 임상 결정 지원에 긍정적인 영향을 미쳤습니다. SwipeRx는 80,000명 이상의 약사 전문 인력이 교육을 받았으며, 영업에서의 경쟁력을 유지하도록 지원하는 혁신적인 도구로 자리잡고 있습니다.



### Causal Discovery from Time-Series Data with Short-Term Invariance-Based Convolutional Neural Networks (https://arxiv.org/abs/2408.08023)
- **What's New**: 본 논문에서는 시계열 데이터로부터 인과 관계를 발견하기 위한 새로운 기법인 STIC(Short-Term Invariance-based Convolutional causal discovery approach)를 제안합니다. STIC는 컨볼루션 신경망(Convolutional Neural Networks)을 활용하여 짧은 시간 불변성을 통해 시계열 데이터의 인과성 발견을 효율적으로 개선합니다.

- **Technical Details**: STIC는 두 가지 인과성 커널을 사용하여 짧은 시간과 메커니즘 불변성을 추정하며, 시계열 데이터의 창(in-window) 인과 그래프를 표현합니다. 논문에서는 ADDITIVE NOISE 모델을 가정하여 컨볼루션과 시계열 데이터의 생성 원리 간의 동등성을 이론적으로 도출합니다. 또한, STIC는 관찰된 시계열 데이터에서 동시적(corresponding) 및 시간 지연(causal) 구조를 동적으로 캡처합니다.

- **Performance Highlights**: STIC는 합성(synthetic) 및 FMRI 벤치마크 데이터셋에서 기존 기법보다 월등한 성능을 보이며, 특히 제한된 관찰 시간 단계가 있는 경우에 상태-최고 성능을 달성했습니다. 실험 결과 STIC는 효율성과 정확성을 모두 갖춘 인과성 발견 성능을 보여주었습니다.



### DIVE: Towards Descriptive and Diverse Visual Commonsense Generation (https://arxiv.org/abs/2408.08021)
Comments:
          19 pages, 10 figuers, EMNLP 2023 (main)

- **What's New**: 이 연구에서는 DIVE라는 새로운 프레임워크를 제안하여 시각적commonsense generation의 기술에서 서술성과 다양성을 향상시키려는 시도가 이루어졌습니다. DIVE는 기존의 모델들이 간과했던 서술적이고 다양한 추론 생성을 중점적으로 다룹니다.

- **Technical Details**: DIVE는 두 가지 주요 방법론, 즉 generic inference filtering과 contrastive retrieval learning을 포함합니다. Generic inference filtering은 특정 이미지의 의미적 농도를 활용하여 일반적인 추론을 제거하는 동시에 균형 잡힌 시각적 commonsense 그래프를 구성합니다. Contrastive retrieval learning은 이미지의 특정 세부 사항을 인식할 수 있도록 모델을 지원합니다.

- **Performance Highlights**: DIVE는 Visual Commonsense Generation(VCG) 데이터셋에서 기존의 최첨단 모델들보다 서술성과 다양성 점수에서 우수한 성능을 나타내며, 인간 수준의 서술성 및 다양성 점수를 달성했습니다. 또한, 생성된 추론의 신뢰성, 서술성 및 다양성에 대한 인간 평가에서는 DIVE가 인간의 평가와도 밀접하게 일치한다는 결과를 보였습니다.



### Accelerating High-Fidelity Waveform Generation via Adversarial Flow Matching Optimization (https://arxiv.org/abs/2408.08019)
Comments:
          9 pages, 9 tables, 1 figure,

- **What's New**: 이 논문은 고충실도(high-fidelity) 및 고효율적인 파형 생성 모델인 PeriodWave-Turbo를 소개합니다. 최근 조건부 흐름 매칭 (Conditional Flow Matching, CFM) 생성을 활용하여 효과적으로 파형 생성을 수행할 수 있으며, 이는 단일 벡터 필드 추정 객관식을 통해 훈련됩니다. 하지만 기존 모델들은 생성 단계 수가 상대적으로 많고, 고주파 정보의 결여 문제가 있었는데, 이를 해결하기 위한 방식으로 고정 단계 생성기를 추가했습니다.

- **Technical Details**: PeriodWave-Turbo는 적대적 흐름 매칭 최적화(adversarial flow matching optimization)를 통해 훈련된 CFM 기반의 파형 생성기를 수정하여 고정 단계로 동작합니다. 이 과정에서 재구성 손실(reconstruction losses)과 적대적 피드백(adversarial feedback)을 이용하고, 최적의 성능을 뒷받침하기 위해 1,000 단계의 세밀한 조정이 필요합니다. 최종적으로는 16단계에서 단 2 또는 4단계로 추론 속도가 감소하였습니다.

- **Performance Highlights**: PeriodWave-Turbo는 LibriTTS 데이터 세트에서 4.454의 PESQ 점수를 기록하며, 이를 통해 주관적 품질 평가에서 획기적인 성능을 달성했습니다. 또한, 본 연구에서는 모델 사이즈를 29M에서 70M로 증가시켜 일반화 성능을 개선하였습니다.



### Asteroid: Resource-Efficient Hybrid Pipeline Parallelism for Collaborative DNN Training on Heterogeneous Edge Devices (https://arxiv.org/abs/2408.08015)
Comments:
          Accepted by The 30th Annual International Conference on Mobile Computing and Networking (MobiCom'24)

- **What's New**: 최근의 연구는 분산된 엣지 장치들을 활용하여 On-device Deep Neural Network(DNN) 훈련 시스템인 Asteroid를 제안하며, 다양한 신뢰할 수 있는 엣지 장치 간의 자원을 효과적으로 결합하는 접근 방식을 채택했습니다. 이는 기존의 단일 장치 자원 관리 방식에서 벗어나, 여러 장치의 유휴 자원을 활용하여 훈련 효율성을 높이는 것을 목표로 합니다.

- **Technical Details**: Asteroid는 하이브리드 파이프라인 병렬 처리(hybrid pipeline parallelism)를 채택하여 분산 훈련을 조정하고, 자원 제약 하에서 처리량(maximizing throughput)을 극대화하기 위한 신중한 병렬 처리 계획(parallelism planning)을 수행합니다. 또한, 경량화된 파이프라인 재생(mechanism) 메커니즘을 도입하여 장치 수준의 동적 변동(device-level dynamics)에 대처할 수 있는 견고성을 보장합니다.

- **Performance Highlights**: Asteroid는 기존의 병렬 처리 방법에 비해 최대 12.2배 빠른 훈련 속도를 보여주며, 최신 하이브리드 병렬 처리 기술보다 2.1배 빠른 성능을 발휘합니다. 불규칙적인 장치 종료 및 실패에도 불구하고, 훈련 파이프라인을 기존 방법보다 14배 빠르게 복구할 수 있습니다.



### IIU: Independent Inference Units for Knowledge-based Visual Question Answering (https://arxiv.org/abs/2408.07989)
- **What's New**: 본 논문에서는 기능적으로 독립적인 유닛을 통해 intra-modal 정보를 분해하는 세밀한 다중 모달(dual modality) 추론을 위한 Independent Inference Units (IIU)를 제안합니다. 기존 모델의 한계를 극복하고 모델의 일반화 능력을 향상시키는 것을 목표로 합니다.

- **Technical Details**: IIU는 ATTENTION 기법을 통해 각 의미별로 특정한 intra-modal 단서를 처리합니다. 독립 추론 유닛이 서로 소통을 통해 보완적인 정보를 수집하며, 메모리 업데이트 모듈을 도입하여 추론 과정에서 중요한 의미를 유지하고 중복 정보를 줄입니다. 그래프 구조를 사용하여 질문과 이미지 간의 관련 정보를 시각적으로 나타냅니다.

- **Performance Highlights**: 실험 결과, 제안된 IIU 모델은 기존의 비사전학습된 다중 모달 추론 모델에 비해 3% 성능 향상을 보여 주목할만한 성과를 기록했습니다. 이 모델은 다양한 데이터셋에서도 효과적으로 정보의 분리를 수행하고 해석 가능한 추론 증거를 제공합니다.



### Analytical Uncertainty-Based Loss Weighting in Multi-Task Learning (https://arxiv.org/abs/2408.07985)
- **What's New**: 이 논문에서는 다중 작업 학습(Multi-task Learning, MTL)에서 개별 작업 손실을 균형 있게 조정하여 성능을 향상시키기 위한 새로운 방법인 Soft Optimal Uncertainty Weighting (UW-SO)를 제안합니다. 이는 Uncertainty Weighting (UW) 방법을 기반으로 하며, 분석적으로 최적의 불확실성 기반 가중치를 계산하여 소프트맥스(softmax) 함수로 정규화합니다.

- **Technical Details**: UW-SO는 손실 손실 기능을 통해 목적에 맞는 실시간 가중치를 학습합니다. UW의 한계를 극복하기 위해 UW의 불확실성 가중치를 최소화하여 각 작업의 손실의 역수로 설정하며, 이는 소프트맥스 함수의 온도 매개변수에 의해 정규화됩니다. 기존의 Scalarization 방법과 비교하여 UW-SO는 단일 하이퍼파라미터를 타겟으로 하여 더 낮은 단계에서 최적화됩니다.

- **Performance Highlights**: 광범위한 데이터셋과 아키텍처에 대한 실험을 통해 UW-SO는 여섯 가지 일반적인 가중치 방식보다 일관되게 우수한 성능을 보였습니다. 연구 결과, 대형 네트워크에서는 가중치 방법의 성능 차이가 줄어들고, 가중치 감소(weight decay)보다 학습률(learning rate) 조정이 더 중요함을 발견했습니다.



### ArabLegalEval: A Multitask Benchmark for Assessing Arabic Legal Knowledge in Large Language Models (https://arxiv.org/abs/2408.07983)
- **What's New**: 이 논문에서는 아랍어 법률 지식을 평가하기 위한 멀티태스킹 벤치마크 데이터셋인 ArabLegalEval을 도입합니다. 이는 LLM(대형 언어 모델)의 법률 관련 문제 해결 능력을 분석하고 성능을 벤치마킹하는 것을 목표로 합니다.

- **Technical Details**: ArabLegalEval은 사우디 법률 문서와 합성 질문을 기반으로 여러 과제를 포함하며, LLM의 법률 추론 능력을 평가하는 데 중점을 두고 있습니다. 이 데이터셋은 법률 전문가와 상담을 통해 개발되었으며, 영어 법률 벤치마크(예: LegalBench)에서 번역된 과제도 포함되어 있습니다. 또한, Retrieval-Augmented Generation (RAG) 시스템을 통한 정보 검색 방법을 연구하고 있습니다.

- **Performance Highlights**: GPT-4와 Jais 모델을 포함한 다국어 및 아랍어 중심 LLM의 성능을 벤치마킹하여, 아랍어 법률 도메인에서의 LLM의 현재 상태를 밝혀내고자 합니다. 초기 평가 결과, GPT-4는 아랍어MMLU 모든 섹션에서 뛰어난 성능을 보였으며, 다른 모델들을 초월하는 성과를 보였습니다.



### Toward a Dialogue System Using a Large Language Model to Recognize User Emotions with a Camera (https://arxiv.org/abs/2408.07982)
Comments:
          4 pages, 5 figures, 1 table, The 1st InterAI: Interactive AI for Human-Centered Robotics workshop in conjuction with IEEE Ro-MAN 2024, Pasadona, LA, USA, Aug. 2024

- **What's New**: 이 연구는 LLM 기반의 AI 에이전트가 사용자와의 대화 중 얼굴 표정에서 감정을 인식하고, 이를 기반으로 적절한 반응을 생성할 수 있는지에 대한 새로운 방법을 탐구했습니다.

- **Technical Details**: 제안하는 시스템인 FacingBot (FBot)은 Python 라이브러리 FER을 활용하여 사용자의 얼굴 표정에서 감정 정보를 인식하고, 이를 JSON 포맷으로 gpt-3.5-Turbo에 추가하여 대화의 맥락을 제공하는 구조입니다.

- **Performance Highlights**: 실험 결과, FBot은 행복(Happy) 및 분노(Angry)와 같은 감정 상태에 따라 적절하게 반응할 수 있음을 확인했습니다. 예를 들어, 사용자가 웃는 얼굴로 "Hello."라고 말했을 때, FBot은 "I’m glad you are happy! How can I help you today?"라는 긍정적인 반응을 보였습니다. 반면, 분노한 얼굴로 "Hello."라고 말을 할 경우 더 배려하는 반응을 보였습니다.



### LLaVA-Surg: Towards Multimodal Surgical Assistant via Structured Surgical Video Learning (https://arxiv.org/abs/2408.07981)
- **What's New**: 이 연구는 Surg-QA라는 새로운 데이터셋을 생성하여, 102,000개의 외과 비디오-지침 쌍을 포함하는 최초의 대규모 외과 비디오 지침-튜닝 데이터셋을 소개합니다. 또한, 멀티모달 대화형 AI 모델인 LLaVA-Surg를 개발하여 외과 비디오에 대한 개방형 질문을 처리할 수 있는 능력을 갖추었습니다.

- **Technical Details**: Surg-QA 데이터셋은 2,201개의 외과 절차에서 44,000개 이상의 외과 비디오 클립에서 추출된 102K 개의 질문-답변 쌍으로 구성되어 있습니다. LLaVA-Surg 모델은 CLIP의 시각 인코더와 Llama를 통합하여 외과 비디오에 대한 지식을 이해하고 대화형으로 답변할 수 있도록 훈련되었습니다.

- **Performance Highlights**: LLaVA-Surg는 기존의 일반 도메인 모델들과 비교하여 외과 비디오 질문-답변 작업에서 뛰어난 성과를 보였으며, 특히 제로샷(Zero-shot) 상황에서 강력한 멀티모달 대화 능력을 입증하였습니다.



### Meta SAC-Lag: Towards Deployable Safe Reinforcement Learning via MetaGradient-based Hyperparameter Tuning (https://arxiv.org/abs/2408.07962)
Comments:
          Main text accepted to the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2024, 10 pages, 4 figures, 3 tables

- **What's New**: 본 논문에서는 안전한 강화 학습(Safe Reinforcement Learning, Safe RL)의 새로운 접근 방식인 Meta Soft Actor-Critic Lagrangian (Meta SAC-Lag)을 제안합니다. 이 모델은 Lagrangian 방법을 기반으로 하며, 메타-그래디언트 최적화를 통해 안전 관련 하이퍼파라미터를 자동으로 업데이트하여 최소한의 조정으로 안전 탐색과 임계값 조정을 달성하고자 합니다.

- **Technical Details**: Meta SAC-Lag는 기존의 Lagrangian 방법의 하이퍼파라미터 조정 문제를 해결하기 위해 메타-그래디언트 최적화(metagradient optimization)를 활용합니다. 이를 통해 에이전트는 안전 임계값을 빠르게 조정할 수 있으며, 다섯 개의 시뮬레이션 환경에서 Lagrangian 기준선과 비교하여 성능을 평가하였습니다. 실제 환경에서는 로봇 암을 사용하여 커피를 컵에 쏟는 작업을 수행하며, 적은 노력으로 임무를 수행할 수 있음을 보여줍니다.

- **Performance Highlights**: Meta SAC-Lag는 안전 및 보상 성능 면에서 더 나은 혹은 경쟁력 있는 결과를 달성하였으며, 하이퍼파라미터 조정 없이도 안정적인 학습을 이뤄냈습니다. 특히, 실제 환경에서의 'Pour Coffee' 테스트에서 작업 목표를 성공적으로 달성하며 시스템에 최소한의 노력이 가해졌습니다.



### RandomNet: Clustering Time Series Using Untrained Deep Neural Networks (https://arxiv.org/abs/2408.07956)
Comments:
          25 pages, 10 figures

- **What's New**: 이 논문에서는 훈련되지 않은 심층 신경망을 활용하여 시계열 데이터를 클러스터링하는 새로운 방법인 RandomNet을 제안합니다. RandomNet은 서로 다른 임의의 가중치 세트를 사용하여 다양한 시계열 표현을 추출하고, 이러한 표현에서 유래한 클러스터링 관계를 조합하여 최종 클러스터링 결과를 구축합니다.

- **Technical Details**: RandomNet은 특정 학습 규칙에 따라 매개변수를 조정하는 대신, 전혀 훈련되지 않은 상태의 신경망을 사용합니다. 이 과정에서 인스턴스 수에 비례하는 선형 시간 복잡도를 가지며, 모든 매개변수는 임의로 생성됩니다. 또한, 랜덤한 매개변수의 조합을 통해 의미 있는 클러스터링을 생성할 수 있습니다.

- **Performance Highlights**: RandomNet의 성능을 검증하기 위해 128개의 데이터셋에 대한 광범위한 실험을 진행한 결과, 기존 최첨단 방법들과 비교하여 Rand Index에서 최고의 성능을 나타냈습니다. 다양한 데이터 크기와 시퀀스 길이를 가진 데이터셋에서도 우수한 결과를 보였습니다.



### Conditional Brownian Bridge Diffusion Model for VHR SAR to Optical Image Translation (https://arxiv.org/abs/2408.07947)
Comments:
          5 pages, 2 figures, 1 table

- **What's New**: 이 논문에서는 Brownian Bridge Diffusion Model (BBDM)을 기반으로 한 조건부 이미지 간 변환 접근 방식을 제안합니다. 이를 통해 저해상도 데이터를 사용하지 않고 고해상도 SAR 이미지를 광학 이미지로 변환하는 방법을 개선했습니다.

- **Technical Details**: 연구진은 0.5m 해상도의 MSAW 데이터를 활용하여 BBDM을 적용했습니다. BBDM은 확률적 특성을 가진 브라운 조정 과정을 바탕으로 두 도메인 간의 변환을 수학적으로 매핑합니다. 이 방법은 픽셀 공간에서 정보를 선형 보간하여 조건으로 사용하며, 이를 통해 SAR 이미지의 공간 정보를 보존하고 변환 과정을 효과적으로 안내합니다.

- **Performance Highlights**: 실험 결과 BBDM 방법이 Conditional Diffusion Model (CDM) 및 기존 GAN 기반 모델들을 여러 지표에서 초월하는 성과를 거두었음을 보여줍니다. 이로 인해 SAR와 광학 이미지 간의 변환 품질이 유의미하게 향상되었습니다.



### Surgical SAM 2: Real-time Segment Anything in Surgical Video by Efficient Frame Pruning (https://arxiv.org/abs/2408.07931)
Comments:
          16 pages, 2 figures

- **What's New**: SurgSAM-2는 수술 비디오 세분화를 위해 SAM2을 최적화한 모델로, 효율적인 프레임 프루닝(Efficient Frame Pruning, EFP) 메커니즘을 도입하여 실시간 비디오 세분화 성능을 향상시킵니다.

- **Technical Details**: SurgSAM-2는 동적 메모리 관리 시스템을 통합하여 이전 프레임을 선택적으로 유지하고, 코사인 유사도 기반의 점수 매기기 메커니즘을 통해 가장 정보가 많은 프레임만 보존합니다. 이를 통해 컴퓨팅 비용을 줄이고 더 빠른 예측 속도를 달성했습니다.

- **Performance Highlights**: SurgSAM-2는 SAM2 대비 3배 더 높은 FPS를 기록하며, EndoVis17 및 EndoVis18 데이터셋에서 우수한 세분화 정확도를 유지합니다. 이러한 성장은 리소스가 제한된 환경에서도 실시간 수술 비디오 세분화의 가능성을 제시합니다.



### MAG-SQL: Multi-Agent Generative Approach with Soft Schema Linking and Iterative Sub-SQL Refinement for Text-to-SQL (https://arxiv.org/abs/2408.07930)
Comments:
          22 pages, 14 figures

- **What's New**: 최근의 In-Context Learning (ICL) 기반 방법이 Text-to-SQL 작업에서 상당한 성공을 거둔 반면, 복잡한 데이터베이스 스키마와 어려운 질문을 가진 데이터셋인 BIRD에서 인간 성능과 여전히 큰 격차가 존재합니다. 이러한 문제를 해결하기 위해 MAG-SQL이라는 다중 에이전트 생성 방식을 제안합니다.

- **Technical Details**: MAG-SQL은 Soft Schema Linking과 반복적 Sub-SQL 정제를 포함한 다중 에이전트 생성 접근 방식을 채택합니다. 이 프레임워크에서는 엔티티 기반 방법을 사용해 데이터베이스의 컬럼을 선택하고, 복잡한 질문을 분해하기 위한 타겟-조건 분해(Targets-Conditions Decomposition) 방법을 도입합니다. Sub-SQL Generator와 Sub-SQL Refiner를 포함하는 반복 생성 모듈을 구축하여, 생성 과정 중 각 단계에 외부 감시를 도입합니다.

- **Performance Highlights**: BIRD 벤치마크에서 GPT-4를 사용하여 평가한 결과, MAG-SQL은 61.08%의 실행 정확도를 달성하였고, 이는 기본 GPT-4의 46.35% 및 MAC-SQL의 57.56%와 비교하여 우수한 성능을 보여줍니다. 이 접근법은 Spider 데이터셋에서도 유사한 향상을 보였습니다.



### CEGRL-TKGR: A Causal Enhanced Graph Representation Learning Framework for Improving Temporal Knowledge Graph Extrapolation Reasoning (https://arxiv.org/abs/2408.07911)
- **What's New**: 본 연구는 인과적 개입(causal intervention) 이론을 그래프 표현 학습(graph representation learning) 프레임워크에 적용하여, 시계열 지식 그래프(temporal knowledge graph) 추론을 위한 혁신적인 새로운 기법을 제안하고 있습니다. 이 방법은 복잡한 사건 간의 원인-결과 관계를 명확히 하여 기존의 편향된 데이터 표현을 극복하고, 더 정확한 예측을 가능하게 합니다.

- **Technical Details**: 제안된 인과적 강화 그래프 표현 학습 프레임워크(CEGRL-TKGR)는 시계열 그래프(sequence of temporal graphs)에서 사건 간의 인과 관계를 모델링하기 위해 인과적 구조(causal structures)를 포함합니다. 이 프레임워크는 인과적 표현(causal representations)과 혼란적 표현(confounding representations)을 분리하고, 그래프 신경망(Graph Neural Networks)과 인과적 개입(causal intervention) 기법을 활용하여 더 견고한 예측을 위한 학습을 수행합니다.

- **Performance Highlights**: CEGRL-TKGR 모델은 여섯 개의 벤치마크 데이터셋에서 링크 예측(link prediction) 작업에 대해 최첨단 모델들보다 우수한 성능을 보였습니다. 이는 제안된 인과 구조와 개입이 모델 예측 성능에 실질적인 기여를 한다는 것을 입증합니다.



### KAN versus MLP on Irregular or Noisy Functions (https://arxiv.org/abs/2408.07906)
- **What's New**: 이 논문은 Kolmogorov-Arnold Networks (KAN)와 Multi-Layer Perceptron (MLP) 네트워크의 비정상적이거나 노이즈가 있는 함수에서의 성능을 비교합니다. 다양한 함수 유형을 분류하고 공정한 비교를 위해 매개변수와 훈련 샘플의 크기를 제어했습니다.

- **Technical Details**: KAN은 Kolmogorov-Arnold 표현 정리를 기반으로 한 새로운 신경망 아키텍처입니다. KAN은 학습 가능한 활성화 함수를 포함하여, 전통적인 MLP와 비교할 때 해석 가능성과 정확성을 높입니다. 논문에서는 정규 함수, 국소 비 미분 가능 점을 가진 연속 함수, 점프 불연속 함수, 특이점이 있는 함수, 일관된 진동을 가진 함수, 노이즈가 있는 함수를 여섯 가지 유형으로 구분하여 성능을 평가합니다.

- **Performance Highlights**: 실험 결과, KAN은 모든 함수 유형에서 항상 최고의 성능을 보이지 않았습니다. 일부 함수 유형에서는 MLP가 KAN보다 우수하거나 유사한 성능을 보였습니다. 노이즈가 추가된 함수적인 정보는 종종 노이즈로 가려지기 때문에, MLP와 KAN 모두 이러한 특성을 효과적으로 추출하는 데 어려움을 겪습니다.



### Assessing Language Models' Worldview for Fiction Generation (https://arxiv.org/abs/2408.07904)
Comments:
          Short paper

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 픽션 생성 능력을 평가하며, 특히 세계관을 일관되게 유지하는 능력에 초점을 맞춥니다. 9개의 LLM을 통해 조사한 결과, 오직 2개 모델만이 일관된 세계관을 보이는 반면, 나머지 모델은 자기 모순적인 답변을 나타냄을 발견했습니다.

- **Technical Details**: 우리는 885개의 문장으로 구성된 데이터셋을 사용하여 LLM들이 픽션 생성에 필요한 진리와 허구를 구분할 수 있는지를 평가했습니다. 고정된 세계 상태를 유지할 수 있는지를 평가하는 일련의 질문(P0-P4)을 통해 모델들의 일관성과 강인성을 분석했습니다.

- **Performance Highlights**: 현재 LLM들은 픽션 작성을 위한 일관된 세계 상태를 유지하는 데 한계를 보이며, 특히 고정 관념이나 논란 있는 주제에 대해 일관性 없는 답변을 보여줍니다. Mistral-7B와 같은 여러 모델이 일관성 부족을 드러내며, 이는 픽션 작성에 있어 신뢰성을 떨어뜨립니다.



### Quantum-inspired Interpretable Deep Learning Architecture for Text Sentiment Analysis (https://arxiv.org/abs/2408.07891)
- **What's New**: 본 논문은 양자역학의 원리를 통합한 심층 학습 아키텍처를 제안하여 텍스트 감정 분석의 정확성과 해석 가능성을 향상시킵니다. 특히, 텍스트 표현과 양자역학 원리 간의 공통점을 분석하여 새로운 텍스트 임베딩 레이어를 개발했습니다.

- **Technical Details**: 제안된 모델은 양자 복소수의 원리를 사용하여 텍스트 밀도 행렬을 계산하고, LSTM 네트워크와 SAM을 기반으로 한 기능 추출 레이어를 설계합니다. 이를 통해 2D CNN을 이용한 특징 응축 및 차원 축소가 이루어집니다.

- **Performance Highlights**: 본 연구에서 제안한 QITSA 모델은 여러 텍스트 감정 분석 데이터셋에서 이전 모델들과 비교하여 정확도와 효율성에서 뛰어난 성능을 보여주었으며, 양자역학의 원리를 통합함으로써 어느 정도의 해석 가능성도 달성했습니다.



### Training Language Models on the Knowledge Graph: Insights on Hallucinations and Their Detectability (https://arxiv.org/abs/2408.07852)
Comments:
          Published at COLM 2024. 16 pages, 11 figures

- **What's New**: 본 연구는 Knowledge Graph (KG)을 기반으로 한 데이터셋을 활용하여 언어 모델(LM)의 환각(hallucination) 현상을 분석합니다. LM의 크기와 훈련 기간에 따른 환각의 발생 빈도 변화를 조사하여, 모델의 성능 향상에 필요한 비용을 정리합니다.

- **Technical Details**: 저자들은 3.15M에서 1.61B개의 비 embedding 파라미터를 가진 Transformer LM을 KG 데이터셋을 기반으로 훈련하였고, 각 LM의 출력에서 환각을 탐지하고 수정하는 방법을 연구했습니다. 환각을 정확히 정의하기 어려운 자연어 환경에서, KG는 LM의 훈련 시 정보에 대한 완전한 제어를 가능하게 하며, 주어-서술어-목적어 3항 데이터 구조를 사용했습니다.

- **Performance Highlights**: LM의 크기가 커지고 훈련기간이 길어질수록 환각 현상이 줄어드는 경향이 있지만, 훈련 데이터 환각률을 5% 이하로 낮추기 위해서는 훨씬 더 큰 모델과 계산 비용이 필요합니다. 반면 LM의 크기가 커질수록 환각 탐지기의 성능은 개선되지만, LM의 환각 탐지 가능성과는 역관계에 있다는 점이 발견되었습니다.



### SER Evals: In-domain and Out-of-domain Benchmarking for Speech Emotion Recognition (https://arxiv.org/abs/2408.07851)
Comments:
          Accepted at INTERSPEECH 2024

- **What's New**: 이번 논문은 언어 및 감정 표현의 다양성에 대한 모델의 일반화 가능성을 평가하기 위한 대규모 Speech Emotion Recognition (SER) 벤치마크를 제안합니다. 이 벤치마크는 다국어 데이터 세트를 포함하여, 적은 사용 빈도를 가진 데이터 세트에 중점을 두어 새로운 데이터에 대한 일반화를 평가합니다.

- **Technical Details**: 본 연구는 Whisper, CLAP과 같은 최신 Speech Representation 모델을 활용하여 cross-lingual SER 성능을 분석합니다. 또한, logit adjustment를 통해 다양한 클래스 분포를 고려하여 공정한 비교를 보장합니다. 평가 프로토콜을 통해 in-domain 및 out-of-domain 성능을 분석하여 모델의 적응력과 일반화 능력을 평가합니다.

- **Performance Highlights**: 놀랍게도, Whisper 모델은 자동 음성 인식(Automatic Speech Recognition) 용도로 설계되었음에도 불구하고 cross-lingual SER에서 전용 SSL 모델보다 뛰어난 성능을 보였습니다. 이러한 결과는 개발할 SER 모델이 더 강력하고 일반화 가능성이 있어야 함을 강조합니다.



### A System for Automated Unit Test Generation Using Large Language Models and Assessment of Generated Test Suites (https://arxiv.org/abs/2408.07846)
- **What's New**: 본 논문에서는 real-life(실제 상황) 복잡성을 고려한 유닛 테스트 생성을 위한 새로운 접근 방식을 제안합니다. 이 접근 방식은 클래스 단위(class-level) 테스트 코드 생성을 중심으로 하며, 자동화된 테스트 생성 및 평가 프로세스를 통해 AgoneTest라는 시스템을 개발하였습니다.

- **Technical Details**: AgoneTest는 Java 프로젝트를 위한 테스트 스위트 생성을 자동화하는 시스템으로, Methods2Test 데이터셋을 활용하여 인공지능 모델(LLMs)로 생성된 테스트와 사람이 작성한 테스트를 비교하는 새로운 데이터셋을 구축하였습니다. 테스트 평가를 위해 JaCoCo, PITest, TsDetect와 같은 라이브러리를 통합하여 다양한 메트릭을 계산합니다.

- **Performance Highlights**: AgoneTest는 여러 LLM과의 비교 평가를 지원하며, 9,410개의 Github 리포지토리를 사용하여 테스트 생성의 폭넓은 적용 가능성을 증명합니다. 이 시스템은 자동화를 통해 테스트 품질에 대한 심도 있는 분석을 제공하며, 기존의 EvoSuite보다 높은 가독성과 품질 평가를 가능하게 합니다.



### Enhancing Equitable Access to AI in Housing and Homelessness System of Care through Federated Learning (https://arxiv.org/abs/2408.07845)
Comments:
          Accepted at the 2024 AAAI/ACM Conference on AI, Ethics, and Society (AIES)

- **What's New**: 이번 연구에서는 주택 및 노숙자 돌봄 시스템(HHSC) 내에서 연합 학습(Federated Learning, FL) 접근 방식을 도입하여, 서로 민감한 데이터를 공유하지 않고도 여러 기관이 협력하여 예측 모델을 학습할 수 있는 방법을 제시합니다.

- **Technical Details**: 연합 학습 접근 방식은 HHSC의 분리된 데이터 세트를 활용하여 ML(기계 학습) 도구에 대한 공정한 접근을 보장하며, 개인 식별 정보를 공유하지 않으면서도 데이터의 프라이버시를 유지합니다. 연구에서는 k-평균 클러스터 분석을 사용하여 고립된 데이터 세트 내에서 만의 라벨링을 수행하는 방법을 설명합니다.

- **Performance Highlights**: 알버타의 캘거리에서 실제 HHSC 데이터를 사용하여, FL 접근 방식이 데이터가 중앙 집중식으로 공유되었을 때의 이상적인 시나리오와 거의 유사한 성능을 보여주었으며, 특히 작은 기관에서 가장 큰 혜택을 받는 것으로 나타났습니다.



### SustainDC -- Benchmarking for Sustainable Data Center Contro (https://arxiv.org/abs/2408.07841)
Comments:
          Under review at Advances in Neural Information Processing Systems 2024 (NeurIPS 2024)

- **What's New**: 이 논문에서는 데이터 센터(DC)에서 멀티-에이전트 강화 학습(MARL) 알고리즘을 벤치마킹하기 위한 파이썬 환경 세트인 SustainDC를 소개합니다. SustainDC는 사용자 정의 데이터 센터 구성 및 작업을 지원하며, 여러 에이전트가 서로의 영향을 고려하면서 작업을 관리합니다.

- **Technical Details**: SustainDC는 워크로드 스케줄링(workload scheduling), 냉각 최적화(cooling optimization), 보조 배터리 관리(auxiliary battery management) 등과 같은 다양한 작업을 위한 사용자 정의 설정을 지원합니다. 또한 다양한 DC 디자인, 위치, 기상 조건, 전력망 탄소 강도(grid carbon intensity), 워크로드 요구 사항에 대한 MARL 알고리즘의 성능을 평가합니다.

- **Performance Highlights**: 본 연구는 MARL 알고리즘을 사용하여 데이터 센터 운영 개선에 대한 중요한 기회를 보여줍니다. AI의 사용 증가로 인해 데이터 센터의 중요성이 높아짐에 따라, SustainDC는 지속 가능한 컴퓨팅과 다양한 실제 문제 해결을 위한 고급 알고리즘 개발 및 벤치마킹을 위한 중요한 플랫폼을 제공합니다.



### ONSEP: A Novel Online Neural-Symbolic Framework for Event Prediction Based on Large Language Mod (https://arxiv.org/abs/2408.07840)
Comments:
          16 pages, ACL 2024 Findings

- **What's New**: 본 논문에서는 온라인 신경-상징적 이벤트 예측(ONSEP) 프레임워크를 소개하며, 이는 동적 인과 규칙 마이닝(DCRM)과 이중 이력 증강 생성(DHAG)을 통합하여 더 나은 이벤트 예측을 가능하게 한다.

- **Technical Details**: DCRM은 실시간 데이터를 기반으로 동적으로 인과 규칙을 구성하여 새로운 인과 관계에 신속하게 적응할 수 있도록 하며, DHAG는 단기 및 장기 이력 컨텍스트를 통합하여 이벤트 예측을 풍부하게 만든다.

- **Performance Highlights**: ONSEP 프레임워크는 다양한 데이터 세트에서 Hit@k (k=1,3,10)의 성능 개선을 보여주며, 대규모 언어 모델(LLMs)을 위한 이벤트 예측을 확장하는 능력을 증명한다.



### A Culturally-Aware Tool for Crowdworkers: Leveraging Chronemics to Support Diverse Work Styles (https://arxiv.org/abs/2408.07838)
Comments:
          32 pages, 9 figures, Computer Supported Cooperative Work (CSCW) 2024

- **What's New**: 이 논문은 문화적 다양성을 무시한 기존 크라우드소싱 시장 인터페이스의 문제점을 지적하며, 이를 해결하기 위해 문화 인식 기반의 도구인 'CultureFit'을 제안합니다.

- **Technical Details**: CultureFit은 '단일시간(monochronic)' 및 '다양한 시간(polychronic)' 작업 스타일을 반영하여 인터페이스를 동적으로 조정하는 도구입니다. 이 도구는 주로 크로노믹스(Chronemics) 이론을 기반으로 설계되었습니다.

- **Performance Highlights**: 필드 실험 결과, CultureFit을 사용한 다양성 배경을 가진 작업자들의 수익이 258% 증가했습니다. 이 연구는 디지털 노동에서 문화 인식을 고려한 최초의 노력 중 하나로, 향후 연구에 활용 가능한 200만 개 이상의 데이터 포인트를 공개할 예정입니다.



### SSRFlow: Semantic-aware Fusion with Spatial Temporal Re-embedding for Real-world Scene Flow (https://arxiv.org/abs/2408.07825)
Comments:
          19 pages,12 figures. arXiv admin note: substantial text overlap with arXiv:2403.07032

- **What's New**: 이번 논문에서는 동적 장면 인식을 위한 3D scene flow 추정의 새로운 접근법인 Dual Cross Attentive (DCA) 방법을 제안합니다. 이는 두 개의 연속적인 포인트 클라우드 간의 의미(context) 기반의 통합 및 정렬을 통해 성능을 개선합니다.

- **Technical Details**: 해당 방법은 Global Fusion Flow Embedding (GF)에 통합되어 전역적 상관관계를 기반으로 흐름 임베딩을 초기화합니다. Spatial Temporal Re-embedding (STR) 모듈은 변형된 객체의 비보정화를 해결하며, Domain Adaptive Losses (DA Losses)를 활용하여 합성 데이터와 실제 LiDAR 스캔 데이터 간의 도메인 차이를 해소합니다.

- **Performance Highlights**: 이 연구는 다양한 데이터셋에서 state-of-the-art (SOTA) 성능을 달성하였으며, 특히 실제 LiDAR 스캔 데이터에서 두드러진 성과를 보여주었습니다.



### Exploration of LLMs, EEG, and behavioral data to measure and support attention and sleep (https://arxiv.org/abs/2408.07822)
- **What's New**: 본 연구에서는 대형 언어 모델(LLMs)을 활용하여 주의력, 수면 단계, 수면 질을 감지하고 개선하는 방법을 탐구합니다. 특히, EEG 및 신체 활동 데이터를 기반으로 개인 맞춤형 수면 개선 제안과 안내된 이미지 스크립트를 생성하는 데 중점을 두었습니다.

- **Technical Details**: 이 연구에서는 LLMs를 사용하여 1) 주의 상태 감지, 2) 수면 개선 제안 생성을 수행하는 세 가지 실험을 진행하였습니다. 각 실험에 사용된 데이터세트로는 Mental Attention State, Sleep EDF expanded, Student Life가 포함됩니다. 특히, LLM의 학습 방식으로 제로샷(Zero-shot), 인컨텍스트(In-context), 파인튜닝(Fine-tuning) 방법을 비교하며, 전통적인 기계 학습 모델(XGBoost)과 성능을 비교했습니다.

- **Performance Highlights**: 연구 결과, 전통적인 기계 학습 모델이 주의력 감지 작업에서 LLM 기반 모델보다 더 높은 성능을 보여주었습니다. 그러나 파인튜닝된 GPT-3.5 모델이 가장 좋은 성과를 보였고, 특정 상황에서는 LLM이 사용자의 상태를 감지하는 데 한계가 있음을 나타냈습니다. 예를 들어, GPT-4 비전 모드는 사용자 상태를 거의 25-30%의 경우에 실패했습니다.



### An Efficient and Explanatory Image and Text Clustering System with Multimodal Autoencoder Architectur (https://arxiv.org/abs/2408.07791)
- **What's New**: 본 논문에서는 국제 뉴스 사건에 대한 서로 다른 문화적 접근법을 비교하는 새로운 맥락에서 Autoencoders와 LLM 해석기의 확장성을 보여줍니다. CRVAE (Convolutional-Recurrent Variational Autoencoder) 모델이 소개되며, 비디오 프레임의 CNN 인코딩과 관련 텍스트의 LSTM 인코딩을 병렬로 결합하여 멀티 모달 인코딩을 강화합니다.

- **Technical Details**: CRVAE 모델은 고해상도 비디오 프레임의 압축을 위한 Dense CVAE (CVAE with Dense Layers) 구조를 사용합니다. 이 모델은 각 비디오 클러스터에 대한 캡션을 생성하는 BLIP 모델과 협력하여 LLaMA 모델을 통해 최종 태그를 생성합니다. 시스템은 K-means (클러스터링 알고리즘) 최적 클러스터 수 선택을 제외하고 거의 자동화되어 있습니다.

- **Performance Highlights**: COVID-19와 동계 올림픽 두 가지 뉴스 주제에 이 시스템을 적용하여 세 개에서 다섯 개의 주제 클러스터로 요약했습니다. 각 주제는 LLM을 통해 생성된 10개의 문구로 설명되며, 이 작업은 비디오 당 30분 미만의 훈련 시간으로 비용 효율적입니다.



### Enhancing Model Interpretability with Local Attribution over Global Exploration (https://arxiv.org/abs/2408.07736)
Comments:
          Accepted by ACMMM 2024

- **What's New**: 이번 연구에서는 AI 모델의 해석 가능성에 대한 관심이 증가함에 따라 새로운 방법론인 Local Attribution (LA) 알고리즘을 제안하였습니다. LA 알고리즘은 targeted 및 untargeted exploration 단계를 포함하여 모델의 결정에 대한 더 정확한 설명을 제공합니다.

- **Technical Details**: LA 알고리즘은 중간 상태를 효과적으로 생성하기 위해 지역적 특성을 이용하여 attribution의 정확성을 높입니다. 또한 In-Distribution (ID)와 Out-Of-Distribution (OOD) 공간을 정의하고, 중간 상태의 중요성을 검토하여 각각의 attribution 결과의 신뢰성을 강조합니다.

- **Performance Highlights**: LA 방법은 최신 방법들과 비교하여 평균 38.21%의 attribution 효과 개선을 달성하였으며, 실험을 통해 각 구성 요소의 중요성을 검증하였습니다. LA 알고리즘의 코드는 공개되어 있어 다른 연구자들과의 협업을 촉진할 것입니다.



### Graph neural network surrogate for strategic transport planning (https://arxiv.org/abs/2408.07726)
- **What's New**: 이 논문은 복잡한 도시 환경에서의 대중교통 시스템 모델링을 위한 새로운 접근 방식을 제안합니다. 기존 Graph Convolution Network (GCN)와 최신 Graph Attention Network (GAT)의 비교 분석에 기반하여, GAT의 변형 모델인 GATv3를 소개하여 over-smoothing 문제를 해결하려 합니다.

- **Technical Details**: 이 연구는 GNN (Graph Neural Network) 아키텍처를 대체 모델로 활용하여 전략적 대중교통 계획을 위해 GCN과 GAT 모델을 비교합니다. GATv3는 성능 향상을 위한 새로운 변형이며, GCN과 GAT의 혼합 모델도 탐구됩니다. 이들은 복잡한 분류 및 회귀 작업을 처리하기 위해 설계되었습니다. 최종적으로, 합성 데이터 생성기를 통해 훈련 데이터의 양을 증가시켜 모델 성능을 개선하는 방법을 소개합니다.

- **Performance Highlights**: 실험 결과, GATv3가 분류 작업에서 우수한 성능을 나타내었으며, GCN은 추가적인 훈련 데이터를 통해 예상치 못한 우위를 보였습니다. 이러한 결과는 합성 데이터 생성기를 통해 모델의 과적합 없이 성능을 향상시킬 수 있는 가능성을 시사합니다.



### Operator Feature Neural Network for Symbolic Regression (https://arxiv.org/abs/2408.07719)
Comments:
          12 pages

- **What's New**: 본 논문에서는 수학 표현의 내재된 운영 논리를 고려하여 수치 손실( numeric loss) 대신, 연산자 특성을 기반으로 한 예측 방법인 OF-Net(Operator Feature Neural Network)을 제안합니다. 이는 기존의 기법들이 단순히 자연어의 문자로 연산자와 변수를 취급한 것과는 달리, 수학적 본질을 반영하려는 혁신적인 접근법입니다.

- **Technical Details**: OF-Net은 연산자 표현을 사용하여 수학적 표현을 학습하며, 심층 연산자 네트워크(Deep Operator Networks) 및 다양한 기능 인코더를 기반으로 합니다. 이 모델은 공공 데이터셋에서 평가되어 높은 R^2 점수와 뛰어난 회수율(recovery rate)을 기록했습니다. 연산자 특성을 이용하여 각 표현의 조합을 예측하며, 연산자 간의 함수적 상호작용을 분석합니다. 또한 깊이 있는 최적화 방안을 제안하고 있습니다.

- **Performance Highlights**: 모델의 성능 테스트 결과, OF-Net은 기존의 기법들에 비해 뛰어난 회복률과 높은 $R^2$ 점수를 보였습니다. 이러한 결과를 통해 OF-Net의 장단점을 분석하고, 향후 최적화 방안에 대한 논의도 포함되어 있습니다.



### Impact of Inaccurate Contamination Ratio on Robust Unsupervised Anomaly Detection (https://arxiv.org/abs/2408.07718)
Comments:
          This is an accepted extended abstract at Black in AI Workshop which will be co-located with NeurIPS 2024 in Canada

- **What's New**: 이 논문은 기존의 가정인 오염이 없는 훈련 데이터 세트에 대한 의존성을 재조명합니다. 연구자들은 다양한 데이터 세트에서 오염 비율의 정보가 부정확한 상황에서도 모델이 어떻게 교란을 견디는지를 조사했습니다.

- **Technical Details**: 탈지도적(anomaly detection) 오염 탐지 모델은 오염 비율(contamination ratio)에 크게 의존하며, 이는 데이터 셋 내의 이상 데이터의 비율을 나타냅니다. 주요 실험은 IF(Isolation Forest), LOF(Local Outlier Factor), OCSVM(One-Class Support Vector Machine)과 같은 얕은 모델을 기반으로 진행되었습니다.

- **Performance Highlights**: 연구 결과, 모델은 잘못된 오염 비율에 노출되더라도 성능이 저하되지 않으며, 오히려 성능이 개선되는 경우가 발견되었습니다. 이는 기존의 오염 비율에 대한 정확한 요구가 없을 수 있으며, 향후 이상 탐지 방법론의 발전에 중요한 단서를 제공합니다.



### Enhancing Supply Chain Visibility with Knowledge Graphs and Large Language Models (https://arxiv.org/abs/2408.07705)
- **What's New**: 이 연구는 공급망 가시성을 향상시키기 위해 Knowledge Graphs (KGs)와 Large Language Models (LLMs)를 활용하는 새로운 프레임워크를 제안합니다. 이 방법은 이해관계자의 정보 공유에 의존하지 않으며, 다양한 공공 소스에서 정보를 자동으로 추출합니다.

- **Technical Details**: 프레임워크는 'zero-shot prompting' 방식을 사용하여 Named Entity Recognition (NER) 및 Relation Extraction (RE) 작업을 수행합니다. 이는 도메인별 대규모 데이터셋 없이도 가능하게 하여, 복잡한 공급망 관계를 적절하게 해석하고 추출합니다.

- **Performance Highlights**: 전기차 공급망을 대상으로 한 사례 연구를 통해, 이 프레임워크는 중요한 광물의 출처를 추적하여 가시성을 크게 향상시킵니다. 결과적으로 Tier-1 및 Tier-2 공급업체를 넘어서는 중요한 의존성과 대체 조달 옵션을 드러내며, 위험 관리와 전략적 계획에 기여합니다.



### Empathic Responding for Digital Interpersonal Emotion Regulation via Content Recommendation (https://arxiv.org/abs/2408.07704)
- **What's New**: 이 논문에서는 온라인 플랫폼에서 대인 간 감정 조절(Interpersonal Emotion Regulation, IER)을 향상시키기 위한 콘텐츠 추천 시스템을 제안합니다. 이 시스템은 사용자가 감정을 조절할 수 있도록 돕고, 특히 공감적 반응(empathic responding) 전략에 맞춘 미디어 콘텐츠를 추천합니다.

- **Technical Details**: 제안된 추천 시스템은 사용자 활동과 선호도를 기반으로 한 Contextual Multi-Armed Bandits (CMAB) 알고리즘을 사용하여, 37.5K개의 사용자 게시물과 상호작용 데이터를 통해 맞춤형 공감적 추천을 생성합니다. 이 연구는 혼합 방법 연구 설계를 통해 텍스트 기반 소셜 미디어 데이터 분석과 사용자 설문 조사를 포함합니다.

- **Performance Highlights**: 실험 결과, 제안된 추천 시스템에 의해 생성된 공감적 추천이 사용자들 사이에서 집중(distraction) 및 회피(avoidance)와 같은 기존 감정 조절 전략보다 선호되는 것으로 나타났습니다.



### MathBridge: A Large Corpus Dataset for Translating Spoken Mathematical Expressions into $LaTeX$ Formulas for Improved Readability (https://arxiv.org/abs/2408.07081)
Comments:
          9page, 6 figures

- **What's New**: 이 논문에서는 수학적 표현을 텍스트 형태로 이해하는 데 있어 발생하는 문제를 해결하기 위해 MathBridge라는 데이터셋을 소개합니다. 이 데이터셋은 약 2300만 개의 LaTeX 공식과 영어로 된 구술 표현이 쌍으로 이루어져 있으며, 이는 텍스트-투-LaTeX(text-to-LaTeX) 번역 연구의 기초를 구축합니다.

- **Technical Details**: MathBridge 데이터셋은 수학적 영어 표현을 LaTeX으로 변환하기 위한 것으로, 공립 대학교에 있는 오픈 소스 교재 및 arXiv에 업로드된 논문으로부터 수집된 데이터를 포함하고 있습니다. 논문에서는 영어 음성을 LaTeX 구문으로 변환하는 과정이 필요하며, 이를 위해 미리 훈련된 언어 모델(pretrained language model)을 활용합니다. 이 모델은 ASR(Automatic Speech Recognition)을 통해 수집된 수학적 표현을 LaTeX로 변환합니다.

- **Performance Highlights**: MathBridge를 이용한 평가에서, T5-large 모델의 sacreBLEU 점수가 4.77에서 46.8로 증가하여, MathBridge 데이터셋이 영어-LaTeX 번역을 위한 우수한 데이터셋임을 보여주었습니다. 또한, 논문에서는 기존 평가 지표가 LaTeX 텍스트 정렬 평가에는 적합하지 않다고 판단하고, 새로운 평가 메트릭의 필요성을 제시하였습니다.



### ConfusedPilot: Confused Deputy Risks in RAG-based LLMs (https://arxiv.org/abs/2408.04870)
- **What's New**: 본 논문에서는 Retrieval Augmented Generation (RAG) 시스템의 보안 취약점인 ConfusedPilot를 소개합니다. RAG 시스템이 기업에서 널리 사용되는 가운데, 이 시스템의 보안적 측면이 불분명임을 강조하고 있습니다. 특히 Microsoft 365의 Copilot을 예로 들어, 잘못된 정보가 기업 운영에 미치는 영향을 연구하였습니다.

- **Technical Details**: ConfusedPilot는 LLM이 생성하는 응답의 무결성(integrity) 및 기밀성(confidentiality)을 침해하는 다양한 취약점을 포함합니다. 공격자는 악성 텍스트를 수정된 프롬프트에 삽입하거나, 캐싱 메커니즘을 이용해 비밀 데이터를 유출할 수 있습니다. 이러한 취약점을 통해 잘못된 정보가 기업 내에서 퍼지고, 이는 판매 및 제조와 같은 운영에 영향을 미칠 수 있습니다.

- **Performance Highlights**: 이 연구에서는 Copilot의 동작을 변경시키는 방법과 잘못된 정보를 숨기는 공격 방법을 제시했습니다. 또한, 이미 삭제된 '팬텀 문서'가 Copilot의 대응을 어떻게 여전히 변경할 수 있는지를 분석했습니다. 최종적으로, 보안 강화를 위한 다양한 완화 전략과 미래 RAG 시스템을 위한 디자인 가이드라인을 제안합니다.




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



### What Color Scheme is More Effective in Assisting Readers to Locate Information in a Color-Coded Article? (https://arxiv.org/abs/2408.06494)
- **What's New**: 이 논문은 대규모 언어 모델(LLM)이 문서 컬러 코딩(color-coding)을 자동화하는 데 사용될 수 있는 가능성을 탐구합니다. 이 연구는 LLM이 생성한 다양한 컬러 스킴(color scheme)이 정보 탐색에 미치는 영향을 조사했습니다. 특히, 아날로그가 아닌(non-analogous) 컬러 스킴과 노란색을 포함하는 컬러 스킴이 정보 탐색 성능을 향상시키는 것으로 나타났습니다.



### Natural Language Outlines for Code: Literate Programming in the LLM Era (https://arxiv.org/abs/2408.04820)
- **What's New**: 이 논문은 소프트웨어 개발 과정 전반에 걸쳐 개발자에게 AI 지원을 제공하기 위한 새로운 방식 및 인터페이스로서 자연어 개요(NL outline)를 사용하는 것을 제안합니다. 코드 함수의 NL 개요는 간결한 산문으로 작성된 여러 문장으로 구성되며, 코드를 분할하고 리터럴 프로그래밍 스타일로 주요 아이디어를 요약합니다. 중요한 것은, 최신 LLMs가 실제로 정확하고 고품질의 NL 개요를 생성할 수 있다는 점입니다. 또한 NL 개요는 코드와 NL 간의 양방향 동기화를 가능하게 하여 한쪽의 변경 사항이 다른 쪽에 자동으로 반영됩니다. NL 개요의 다양한 사용 사례를 논의합니다. NL 개요는 코드와 차이의 이해 및 탐색 속도를 높이고, 코드 유지 관리를 간소화하고, 코드 검색을 강화하고, 코드 생성을 안내하고, 그 외에도 여러 가지 역할을 수행할 수 있습니다. 그런 다음 NL 개요 생성을 위한 여러 LLM 프롬프팅 기법을 제안하고 비교하여 전문 개발자가 개요 품질을 평가하도록 합니다. 마지막으로 코드 검토 및 악성 코드 탐지라는 어려운 작업을 향한 NL 개요 적용에 대한 두 가지 사례 연구를 제시합니다.



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




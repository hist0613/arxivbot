New uploads on arXiv(cs.CL)

### Language Models as Causal Effect Generators (https://arxiv.org/abs/2411.08019)
- **What's New**: 이 논문에서는 큰 언어 모델(LLM)에 기반한 데이터 생성 프레임워크를 제시하며, 특정한 인과 구조를 제어할 수 있는 방법을 설명합니다. 이 방법은 언어 모델과 방향 비순환 그래프(DAG)를 결합하여 순차적으로 구동되는 구조적 인과 모델(SD-SCM)을 생성합니다.

- **Technical Details**: SD-SCM은 사용자 정의 구조와 LLM 정의 구조 방정식을 포함하는 인과 모델을 나타냅니다. SD-SCM을 사용하면 관찰적, 개입적, 반사적 분포에서 샘플링할 수 있는 방법을 제공합니다. 이 모델을 통해 개별 반사적 데이터를 자동으로 생성할 수 있으며, 이는 기존의 기능적 관계를 수동으로 명시하지 않아도 가능하게 합니다. 코드와 데이터셋은 GitHub에서 접근할 수 있습니다.

- **Performance Highlights**: SD-SCM을 활용하여 생성된 데이터셋에 대해 여러 평균처리효과(ATE), 조건부 평균처리효과(CATE), 개별처리효과(ITE)를 추정하는 방법을 테스트했습니다. 이 절차는 LLM이 잘못된 정보, 차별, 또는 기타 바람직하지 않은 행동을 감지하는 데도 사용할 수 있어 LLM 감사를 위한 기반이 될 수 있습니다.



### ExpressivityArena: Can LLMs Express Information Implicitly? (https://arxiv.org/abs/2411.08010)
Comments:
          8 pages, 22 figures

- **What's New**: 본 논문은 Large Language Models (LLMs)의 암시적 언어 신호 표현 능력을 평가하기 위한 Python 라이브러리인 ExpressivityArena를 제시합니다. LLM의 표현력을 평가할 수 있는 포괄적인 프레임워크를 제공하였으며, 창의적 및 논리적인 과제에 대한 실험을 통해 모델의 암시적 의사소통 능력을 탐색했습니다.

- **Technical Details**: ExpressivityArena 프레임워크를 사용하여 LLM의 표현력을 객관적으로 평가합니다. 다양한 LLM의 아웃풋을 평가하기 위한 자동 채점기를 설정하였으며, 이 채점기의 유효성을 인간 연구를 통해 검증했습니다. 실험을 통해 시가 생성, 코드 생성 등의 작업에서 모델의 표현력이 얼마나 다양한지를 살펴보았습니다.

- **Performance Highlights**: 모델은 감정 표현 시 시간이 지남에 따라 표현력이 줄어드는 경향을 보였으나, 직업 표현에서는 대화가 진행될수록 표현력이 증가하는 결과를 보여주었습니다. 이는 LLM이 표현 예정이다 다양한 경우에 따라 제한이 있음을 시사하며, 향후 발전 방향에 중요한 통찰을 제공합니다.



### Derivational Morphology Reveals Analogical Generalization in Large Language Models (https://arxiv.org/abs/2411.07990)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 언어적 일반화(linguistic generalization) 메커니즘을 분석했다. 특히, 규칙 기반(rule-based) 접근 방식과 유사성 기반(analogical) 접근 방식의 둘 다 설명할 수 있는 다양한 문법 현상 중 영어 형용사의 명사화(adjective nominalization)를 조사했다.

- **Technical Details**: 본 연구는 GPT-J 모델을 중심으로 하여, 규칙 기반 모델인 Minimal Generalization Learner (MGL)와 유사성 모델인 Generalized Context Model (GCM)을 비교 분석하였다. 이 과정에서는 4가지 형용사 클래스에 대해 -ity와 -ness의 선호도를 분석하며, 각 모델의 예측 결과를 검토하였다. 연구는 모델이 유형 빈도(type frequency)와 토큰 빈도(token frequency) 의 영향을 어떻게 반영하는지를 평가하였다.

- **Performance Highlights**: 연구 결과, GPT-J는 정규적인 명사화 패턴에 대해서는 MGL과 GCM 모두와 비슷한 예측을 수행한다. 하지만, 다양한 명사화 패턴을 가진 형용사에서는 GCM의 유사성 모델이 GPT-J의 행동을 더 잘 설명한다는 사실이 드러났다. 이는 GPT-J가 규칙이 아닌 유사한 예를 바탕으로 언어적 일반화를 이루고 있음을 시사한다.



### From General to Specific: Utilizing General Hallucation to Automatically Measure the Role Relationship Fidelity for Specific Role-Play Agents (https://arxiv.org/abs/2411.07965)
- **What's New**: 본 논문은 Role-Playing Agents (RPA)의 성능을 평가하기 위해 기존의 비효율적인 벤치마크에 대한 자동화된 새로운 패러다임을 제안합니다.

- **Technical Details**: 연구팀은 일반 지식 그래프에서 관계를 추출하고, ChatGPT를 활용하여 스탠스 탐지(stance detection)를 수행하며, 관계 환각(relationship hallucination)과 관련된 세 가지 메트릭을 정의합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안한 메트릭의 효과성과 안정성을 검증하였으며, RPA의 관계 환각과 사실성(factuality) 간의 상충(trade-off)에 대해서도 논의하였습니다.



### CryptoLLM: Unleashing the Power of Prompted LLMs for SmartQnA and Classification of Crypto Posts (https://arxiv.org/abs/2411.07917)
Comments:
          Accepted at FIRE 2024 (Track: Opinion Extraction and Question Answering from CryptoCurrency-Related Tweets and Reddit posts (CryptOQA))

- **What's New**: 이 연구는 암호화폐 관련 소셜 미디어 게시물을 분류하기 위한 강력한 모델을 개발하는 데 중점을 두고 있습니다. 특히 Reddit 및 Twitter에서의 사용자 생성 콘텐츠를 효율적으로 분석하여 의사 결정에 도움이 되는 정보를 제공하고자 합니다.

- **Technical Details**: 연구진은 Large Language Models (LLMs)인 GPT-4-Turbo를 활용하여 프롬프트 기반의 접근 방식을 사용해 소셜 미디어 게시물을 텍스트 분류 작업에 적용했습니다. 64-shot 기법을 통해 주어진 질문에 대한 답변의 관련성을 평가하는 모델을 구현하였습니다.

- **Performance Highlights**: 이 모델은 사회적 쓰레기(post)를 포함하여 객관적(objective), 긍정적(positive), 부정적(negative) 등 아홉 가지 범주로 분류하며, 관련된 답변을 찾아내는 능력을 개선하여 시장 통찰력 및 사용자 경험을 향상시킬 것으로 기대됩니다.



### Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus (https://arxiv.org/abs/2411.07892)
Comments:
          9 pages, 3 figures

- **What's New**: 이 연구에서는 2020년 5월과 6월에 공개 RSS 피드를 통해 사용 가능한 모든 영어 팟캐스트를 포함하는 110만 개 이상의 팟캐스트 전사를 포함한 대규모 데이터 세트인 SPoRC(Structured Podcast Research Corpus)를 소개합니다. 이 데이터 세트는 텍스트뿐만 아니라 37만 개 에피소드의 오디오 기능 및 화자 전환도 포함되어 있습니다.

- **Technical Details**: SPoRC 데이터 세트는 2020년 5월과 6월에 방송된 110만 개의 팟캐스트 에피소드에 대한 전사, 호스팅 플랫폼, 유추된 호스트 및 게스트 이름, 기타 메타데이터가 포함되어 있습니다. Whisper라는 자동 음성 인식 시스템을 사용하여 오디오 파일을 텍스트 형식으로 전사하였으며, openSMILE 도구를 통해 화자의 커뮤니케이션과 관련된 추가 정보를 추출하였습니다.

- **Performance Highlights**: 이 연구에서 수행한 초기 분석은 팟캐스트 생태계의 구조와 반응성을 연구하는 데 사용되었으며, 팟캐스트에서 논의되는 주제 분포 및 게스트 공동 출현에 의해 생성된 커뮤니티 네트워크 구조를 보여주었습니다. 또한 주요 미디어 사건의 영향과 각 매체의 다양한 반응을 분석하였습니다.



### Trustful LLMs: Customizing and Grounding Text Generation with Knowledge Bases and Dual Decoders (https://arxiv.org/abs/2411.07870)
- **What's New**: 이번 연구에서는 LLMs (Large Language Models)의 Hallucination (환각) 문제를 해결하기 위해 지식 트리플(knowledge triplets)을 활용하는 후처리 알고리즘과 RAG (Retrieval-Augmented Generation) 문맥을 융합한 이중 디코더(Dual-Decoder) 모델을 제안합니다.

- **Technical Details**: 제안된 후처리 알고리즘과 이중 디코더 모델은 지식 그래프(knowledge graph)로부터 추출한 지식 트리플을 기반으로 Hallucinations를 수정하며, RAG의 맥락을 통합하여 생성 과정을 안내합니다. 알고리즘은 생성된 텍스트의 진위를 검증하고, 출력 길이에 대한 제약을 완화하며, 관련된 맥락과 프롬프트에만 초점을 맞춥니다.

- **Performance Highlights**: 이 방법론은 Microsoft 제품 문의를 지원하는 실제 상용 애플리케이션 시나리오에 적용되며, 고객에 대한 서비스의 완성도와 정확성을 높이는 데 기여합니다. 또한, 다양한 LLM 버전(예: ChatGPT, LLama-3)에서 Fluency (유창성) 문제를 보완하여, 신뢰성을 높이고 사실성을 강화하는 것으로 나타났습니다.



### Verbosity $\neq$ Veracity: Demystify Verbosity Compensation Behavior of Large Language Models (https://arxiv.org/abs/2411.07858)
Comments:
          19 pages, 6 figures

- **What's New**: 본 논문에서는 'Verbosity Compensation' (VC)이라는 새로운 개념을 정의하고 분석하며, 이 현상이 LLM(대규모 언어 모델)에 미치는 영향을 논의합니다. VC는 필요한 것 이상으로 많은 단어를 사용하여 답변하는 현상으로, 이는 사용자 이해를 혼란스럽게 하고 모델의 효율성을 떨어뜨립니다.

- **Technical Details**: VC는 정보 손실 없이 압축 가능한 응답을 생성하는 행동으로 정의됩니다. 실험은 14개의 새로운 LLM을 대상으로 5개의 데이터 세트에서 수행되었으며, VC의 영향을 체계적으로 분석하였습니다. VC 빈도는 모든 모델과 데이터 세트에서 관찰되었으며, 특히 GPT-4는 50.40%의 VC 빈도를 보였습니다. VC가 성능에 미치는 영향을 평가하기 위해 여러 지표가 사용되었습니다.

- **Performance Highlights**: 실험 결과, Mistral 모델의 VC 빈도가 63.81%에서 16.16%로 감소하는 것을 확인하였습니다. 또한, verbose(장황한) 응답은 모든 데이터 세트에서 높은 불확실성을 보였으며, 이는 모델의 VC와 불확실성 간의 강한 연결을 나타냅니다.



### Tucano: Advancing Neural Text Generation for Portugues (https://arxiv.org/abs/2411.07854)
- **What's New**: 본 연구는 포르투갈어의 신경 텍스트 생성을 위한 새로운 자원을 소개하고 있습니다. GigaVerbo라는 대규모의 포르투갈어 데이터셋을 구축하여, 이를 활용해 Tucano라는 디코더-트랜스포머 모델을 훈련하였습니다.

- **Technical Details**: GigaVerbo는 2000억 개의 토큰으로 구성된 중복 제거된 포르투갈어 텍스트 코퍼스의 집합으로, 여기서 Tucano 모델을 훈련시켰습니다. 이 모델은 여러 포르투갈어 벤치마크에서 다른 포르투갈어 및 다국어 모델들과 동등하거나 우수한 성능을 보였습니다.

- **Performance Highlights**: Tucano 모델은 기존의 포르투갈어 NLP 커뮤니티에서 사용되는 벤치마크와의 성능 평가에서도 좋은 성과를 냈으며, 특히 기존 모델과의 성능 상관관계의 한계를 드러냈습니다.



### IAE: Irony-based Adversarial Examples for Sentiment Analysis Systems (https://arxiv.org/abs/2411.07850)
- **What's New**: 이 논문에서는 Irony 기반의 적대적 예제(Irony-based Adversarial Examples, IAE)를 제안합니다. 이는 본래의 문장을 아이러니한 문장으로 변환하여 적대적 텍스트를 생성하는 방법입니다. 기존의 아이러니 말뭉치에 의존하지 않기 때문에 다양한 자연어 처리(NLP) 작업에 적합한 도구로 자리 잡을 수 있습니다.

- **Technical Details**: IAE 방법은 평가 단어를 정확히 찾아 적절한 복합어로 대체하며, 아이러니한 요소를 포함하여 텍스트의 의미 일관성을 유지해야 한다는 점에서 전문적입니다. 텍스트의 직설적인 의미와는 반대되는 상황적 의미를 파악해야 하며, 세 가지 주요 도전 과제가 있습니다: 1) 평가 단어의 위치 찾기, 2) 적절한 복합어로 대체하기, 3) 필요할 때 적절한 아이러니 평가로 텍스트 확장하기.

- **Performance Highlights**: 우리는 여러 최첨단 딥러닝 모델이 감정 분석 작업에서 IAE 공격에 노출되었을 때 성능이 현격히 저하된다는 것을 보여주었습니다. 이는 현재 NLP 시스템이 아이러니를 통한 적대적 조작에 취약함을 강조합니다. 반면 인간은 텍스트에서 아이러니의 영향에 덜 민감함을 보여주었습니다.



### Ethical Concern Identification in NLP: A Corpus of ACL Anthology Ethics Statements (https://arxiv.org/abs/2411.07845)
- **What's New**: 본 논문은 LLM(대형 언어 모델) 연구자들이 어떤 윤리적 우려를 가지고 있는지를 조사하고, 그런 우려를 자동으로 식별하기 위한 EthiCon이라는 데이터셋을 구축하였습니다. 이 데이터셋은 1,580개의 윤리적 우려 진술문으로 구성되어 있으며, NLP(자연어 처리) 커뮤니티와 일반 대중 간의 우려를 비교하는 것도 목적입니다.

- **Technical Details**: EthiCon은 ACL(Association for Computational Linguistics) 자료에서 추출된 윤리적 진술문으로 이루어진 주석이 달린 코퍼스입니다. 저자는 4,691개 및 3,357개의 논문에서 윤리적 진술문을 스크래핑하여 1,580개의 진술을 수집하고, 이를 윤리적 우려의 5개 카테고리로 분류하였습니다. LLM 실험을 통해 자동화된 윤리적 우려식별 방법도 제시하였습니다.

- **Performance Highlights**: EthiCon 데이터셋을 활용하여 LLM 모델들이 윤리적 우려를 식별하는 데에 높은 정확도를 나타냈습니다. 주목할만한 점은 생성 작업에서 모델들이 인간 주석가보다 더 많은 우려를 식별했으나, 비논리적 생성이나 허위 정보는 발견되지 않았다는 것입니다. 이는 LLMs이 윤리적 우려 모니터링을 위한 도구로서 유용성 가능성을 시사합니다.



### Chain Association-based Attacking and Shielding Natural Language Processing Systems (https://arxiv.org/abs/2411.07843)
- **What's New**: 이번 연구에서는 자연어 처리(NLP) 시스템에 대한 새로운 공격 방법으로 체인 연관 기반의 적대적 공격을 제안합니다. 이 방법은 사람과 기계 간의 이해 격차를 활용하며, 특히 중국어 캐릭터에 대한 체인 연관 그래프를 생성하여 잠재적인 적대적 예시를 검색하는 데 사용됩니다.

- **Technical Details**: 체인 연관 그래프는 중국어 캐릭터 구축을 위한 연관 패러다임에 기반하여 생성되며, 최적의 적대적 예시를 검색하기 위해 이산 입자 군집 최적화(Discrete Particle Swarm Optimization) 알고리즘을 도입합니다. 이 연구에서는 체인 연관이 적대적 공격에 적용된 첫 번째 사례로, 적대적 훈련(Adversarial Training) 및 연관 그래프 기반 복구를 통해 시스템을 보호하는 두 가지 방법도 탐색합니다.

- **Performance Highlights**: 고급 NLP 모델과 애플리케이션이 체인 연관 기반 공격에 매우 취약함을 보여줍니다. 실험 결과, 사람들이 변형된 텍스트를 이해하는 능력이 뛰어난 반면, NLP 시스템은 이 공격에 의해 비정확한 결과를 초래할 수 있음을 확인했습니다.



### Query Optimization for Parametric Knowledge Refinement in Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2411.07820)
- **What's New**: 본 논문에서는 Extraction-Refinement-Retrieval-Read (ERRR) 프레임워크를 소개합니다. ERRR은 Retrieval-Augmented Generation (RAG) 시스템의 정보 검색 과정에서 발생하는 사전 검색 정보 격차를 해소하기 위해 설계된 새로운 접근 방식입니다. 이 프레임워크는 대형 언어 모델(LLM)의 특정 지식 요구사항을 충족하기 위한 쿼리 최적화를 포함합니다.

- **Technical Details**: ERRR 프레임워크는 LLM에서 파라메트릭 지식을 추출하는 것으로 시작하여, 이 지식을 바탕으로 쿼리를 세밀하게 조정하는 전문 쿼리 최적화기를 사용합니다. 이 과정에서 오직 가장 관련성 높은 정보만을 검색하여 정확한 응답을 생성하는 데 필요한 정보를 보장합니다. ERRR은 지식 증류를 통해 더 큰 모델에서 개선된 소형 조정 가능한 모델을 쿼리 최적화기로 사용합니다.

- **Performance Highlights**: 다양한 질문-응답(QA) 데이터셋에서 ERRR은 기존의 기준선을 지속적으로 초월하는 성능을 발휘했습니다. ERRR은 여러 가지 검색 시스템과 설정에서도 높은 적응성을 보이며 RAG 시스템의 유용성과 정확성을 개선하는 데 매우 효과적임을 입증했습니다.



### Likelihood as a Performance Gauge for Retrieval-Augmented Generation (https://arxiv.org/abs/2411.07773)
Comments:
          Under review at NAACL 2025. Code is available at this https URL

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LMs)에서의 retrieval-augmented generation(RAG) 과정 중 문서의 순서가 결과에 미치는 영향을 분석합니다. 특히, 질문의 likelihood가 모델 성능을 예측할 수 있는 지표가 될 수 있음을 보여주며, 이를 바탕으로 더 나은 성능을 위한 프롬프트 최적화 방법을 제안합니다.

- **Technical Details**: 본 연구는 NQ-Open과 ELI5 두 가지 질문-답변 데이터셋에서 다양한 최첨단 LMs(LLaMA-2, LLaMA-3, LLaMA-3.1, Mistral-v0.3, MPT)를 활용하여 질문의 likelihood와 답변 정확도 간의 상관관계를 조사하였습니다. 입력 프롬프트의 세 가지 구성 요소인 컨텍스트, 질문, 금지 답변(gold answer)의 log-likelihood를 분석하여, 높은 log-likelihood를 가진 질문에 대해 LMs가 더 나은 답변을 할 수 있음을 발견했습니다.

- **Performance Highlights**: 제안된 방법은 질문 likelihood를 기반으로 한 프롬프트 최적화를 통해 두 데이터셋에서 답변 정확도를 개선했습니다. 효율적인 계산 방식이 특징적이며, LM 응답을 생성하기 위해 여러 번 실행할 필요가 적어 계산 비용을 절감할 수 있습니다.



### Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows (https://arxiv.org/abs/2411.07763)
- **What's New**: Spider 2.0은 기업 데이터베이스의 실제 사용 사례에서 파생된 632개의 텍스트-투-SQL(workflow) 문제를 포함하는 새로운 평가 프레임워크입니다.

- **Technical Details**: Spider 2.0은 BigQuery와 Snowflake와 같은 클라우드 또는 로컬 데이터베이스 시스템에서 1,000개 이상의 컬럼을 포함할 수 있는 데이터베이스로 구성되며, 문제 해결에는 데이터베이스 메타데이터, 방언(dialect) 문서 및 프로젝트 코드베이스와의 상호 작용이 필요합니다.

- **Performance Highlights**: Spider 2.0에서 코드 에이전트 프레임워크는 17.0%의 작업을 성공적으로 해결하여, Spider 1.0의 91.2% 및 BIRD의 73.0%와 비교됩니다. 이는 실제 기업 사용에서 충분한 성능을 달성하기 위해 언어 모델의 상당한 개선이 필요함을 보여줍니다.



### Mitigating Bias in Queer Representation within Large Language Models: A Collaborative Agent Approach (https://arxiv.org/abs/2411.07656)
Comments:
          NeurIPS 2024 Queer in AI Workshop

- **What's New**: 본 논문은 대형 언어 모델(LLMs)의 성별 대명사 사용에서의 편향 문제를 다루며, 이러한 편향이 퀴어(queer) 개인의 대표성에 미치는 부정적 영향을 줄이기 위한 협력 에이전트 파이프라인을 소개합니다.

- **Technical Details**: 우리는 비대칭 편향 탐지 및 수정에 집중하는 다중 에이전트 프레임워크를 설계했습니다. Tango 데이터셋을 사용한 실험 평가 결과, 기존 모델에 비해 포괄적인 대명사 분류에서 32.6%의 향상을 보여주었습니다. 각 에이전트는 입력에 대한 판단의 일관성을 검토하고 최종 결정을 내리는 방식으로 작업합니다.

- **Performance Highlights**: 본 연구의 결과는 에이전트 기반 프레임워크가 AI 생성 콘텐츠의 공정성과 포괄성을 향상시킬 수 있는 잠재력을 강조하며, 특히 퀴어 개인에 대한 언어 모델의 대표성을 개선하는 데 효과적임을 나타냅니다.



### Annotating Constructions with UD: the experience of the Italian Constructicon (https://arxiv.org/abs/2411.07623)
- **What's New**: 이 논문은 이탈리아어 Constructicon과 UD 자원(Universal Dependencies)을 연결하려는 첫 번째 시도를 설명합니다.

- **Technical Details**: 이 논문은 Constructicography와 관련된 두 가지 상호 연결된 측면을 다룹니다. 첫 번째는 cxns(구문)의 사전이며, 두 번째는 텍스트에서 cxn의 실제 발생을 식별하고 주석을 달아야 합니다. cxns는 형태와 기능이 결합된 전통적인 쌍으로 정의되며, 형태론(morphology)에서 문법(syntax) 및 담화(discourse)에 이르기까지 다양한 복잡성과 추상성을 포함합니다.

- **Performance Highlights**: UD 인프라는 다양한 언어를 표현하는 데 사용되는 사실상의 표준을 제공하며, 이탈리아어 Constructicon은 cxns의 그래프(GCxns)와 주석이 달린 예제의 두 가지 관련 데이터 구조로 구성됩니다.



### Multimodal Clinical Reasoning through Knowledge-augmented Rationale Generation (https://arxiv.org/abs/2411.07611)
Comments:
          11 pages. 4 figures

- **What's New**: ClinRaGen은 질병 진단을 위한 멀티모달 (multimodal) 이론 생성에 최적화된 소형 언어 모델 (SLM)로, 도메인 지식을 통합한 주목 메커니즘을 포함하여 시간적 전자 건강 기록 (EHR) 데이터와 결합하며, 단계별 (stepwise) 이론 증류 전략을 통해 임상 이론을 생성할 수 있도록 설정되었습니다.

- **Technical Details**: ClinRaGen은 도메인 지식과 시간적 EHR 데이터를 통합하기 위해 독특한 지식 증강 주목 메커니즘을 사용합니다. 이 방법은 텍스트와 시간적 기반의 임상 이론을 모두 생성하고, 멀티모달 EHR 데이터를 효과적으로 해석하여 더 정확한 질병 진단을 지원합니다.

- **Performance Highlights**: MIMIC-III 및 MIMIC-IV 데이터세트에서 수행한 평가에 따르면, ClinRaGen은 멀티모달 EHR 데이터를 효과적으로 해석하고, LLM들과 유사한 정확도를 가진 임상 이론을 생성하는데 있어 뛰어난 성능을 발휘하였습니다. 이는 LLM과 SLM 간의 성능 격차를 좁히는 데 기여합니다.



### Problem-Oriented Segmentation and Retrieval: Case Study on Tutoring Conversations (https://arxiv.org/abs/2411.07598)
Comments:
          EMNLP 2024 Findings. Our code and dataset are open-sourced at this https URL

- **What's New**: 이번 연구에서는 Problem-Oriented Segmentation & Retrieval (POSR)이라는 새로운 연구 과제를 소개합니다. 이는 대화 내용을 세그먼트(segment)로 나누고 각 세그먼트를 적절한 참조 항목(reference item)에 연결하는 작업입니다. 특히 교육 분야에서 이 방법이 적용되며, LessonLink라는 실세계 튜터링 수업 데이터셋이 개발되었습니다.

- **Technical Details**: 이 연구에서는 3,500개의 세그먼트와 116개의 SAT 수학 문제로 구성된 24,300분 분량의 튜터링 수업을 포함하는 LessonLink 데이터셋을 제시합니다. 여러 POSR 접근 방식을 정의하고 평가하며, 세그멘테이션 방법(예: TextTiling), 정보 검색(IR) 방법(예: ColBERT) 및 대형 언어 모델(LLMs) 방법을 포함합니다. 새로운 세그멘테이션 및 검색 점수(SRS)를 도입하여 교실 내 대화 세그먼트와 참조 자료 간의 정확성을 측정합니다.

- **Performance Highlights**: POSR 방법은 독립적인 세그멘테이션 및 검색 파이프라인보다 최대 76% 향상된 성능을 보였으며, 전통적인 세그멘테이션 방법보다 최대 78% 더 높은 정확도를 기록했습니다. 또한, 튜터가 각 문제에 소요하는 시간에 따라 개념적 설명과 절차적 설명의 차이가 있다는 통찰을 제공하였으며, 이는 교육 현장에서의 언어 및 시간 관리 실천을 개선할 수 있는 기회를 제시합니다.



### Large Language Models as Neurolinguistic Subjects: Identifying Internal Representations for Form and Meaning (https://arxiv.org/abs/2411.07533)
- **What's New**: 본 연구는 대규모 언어 모델(Large Language Models, LLMs)의 언어적 이해도를 평가하기 위한 새로운 방법으로 신경언어학적(Neurolinguistic) 접근 방식을 제안합니다. 이는 기존의 심리언어학적(Psycholinguistic) 평가 방식과 비교하여 LLM의 내부 표현을 더 심층적으로 분석합니다.

- **Technical Details**: 신경언어학적 접근법으로는 최소 쌍(Minimal Pair) 및 진단 프로빙(Diagnostic Probing) 기법을 결합하여 모델의 다양한 층에서의 활성화 패턴을 분석합니다. 이를 통해 LLMs의 언어 형식과 의미를 어떻게 표현하는지 더 세분화된 방식으로 검토합니다. 또한, 두 개의 새로운 개념 최소 쌍 데이터 세트인 중국어(COMPS-ZH)와 독일어(COMPS-DE)를 도입했습니다.

- **Performance Highlights**: 연구 결과, LLM들은 의미에 비해 형태에서 더 높은 능력을 보이며, 의미 능력은 형태와 선형적으로 상관관계가 있음이 나타났습니다. 이는 LLM의 의미와 형태가 독립적이지 않으며, 이들을 지탱하는 개념적 표현이 형태에 기반한 통계적 상관관계에 의존할 수 있음을 시사합니다.



### Prompt-enhanced Network for Hateful Meme Classification (https://arxiv.org/abs/2411.07527)
Comments:
          Published in Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence Main Track. Pages 6397-6405

- **What's New**: Pen이라는 새로운 모델 프레임워크를 제안하여, 기존의 다중 양식(multi-modal) 증오밈(classification) 분류 방식의 한계를 극복합니다. 이 프레임워크는 프로프트(prompt) 학습 접근법을 바탕으로 하여 전체 정보(global information)를 활용하여 분류 정확도를 향상시킵니다.

- **Technical Details**: Pen 프레임워크는 프로프트를 통해 입력 PLM의 시퀀스를 처리하고, 이를 지역 분할(region segmentation)하여 글로벌 정보 특징을 추출합니다.다중 관점(multi-view perception) 모듈을 통해 실제 인스턴스와 demonstration의 글로벌 특징을 인식하여 증오 감정을 판단합니다. 또한, 프레임워크에 프롬프트 인식 대비 학습(prompt-aware contrastive learning)을 도입하여 샘플 특징 분포의 품질을 개선합니다.

- **Performance Highlights**: Pen은 두 개의 공개 데이터셋에서 다각적인 실험을 통해 효과성을 검증했으며, 기존 최첨단 모델 차원에서의 성능을 초월하는 성적을 기록했습니다. 수작업 프롬프트 방법에 비해 일반화(generalization) 및 분류 정확도가 뛰어난 것으로 나타났습니다.



### Fair Summarization: Bridging Quality and Diversity in Extractive Summaries (https://arxiv.org/abs/2411.07521)
Comments:
          Accepted at Algorithmic Fairness through the Lens of Metrics and Evaluation Workshop @ NeurIPS 2024

- **What's New**: 이 논문에서는 사용자 생성 콘텐츠의 다중 문서 요약에서 공정성을 보장하기 위한 두 가지 새로운 방법, FairExtract와 FairGPT를 소개합니다. 이 방법들은 서로 다른 사회적 집단을 공평하게 대표할 수 있는 방법을 제시하여 기존 요약 기법의 한계를 극복합니다.

- **Technical Details**: FairExtract는 클러스터링 기반의 접근 방식으로, 정교한 군집화를 통해 문서에서 중요한 정보를 추출하며, FairGPT는 GPT-3.5-turbo 모델을 활용하여 공정성 제약을 적용합니다. 논문에서는 다양한 요약 품질 지표(SUPERT, BLANC, SummaQA 등)와 공정성 지표(F)를 사용하여 이 두 방법을 평가하고, 그들의 성능을 기존 방법과 비교합니다.

- **Performance Highlights**: FairExtract와 FairGPT는 고품질의 요약을 유지하면서도 뛰어난 공정성을 달성하며, composite metrics(SUPERT+F, BLANC+F)를 통해 품질과 공정성을 동시에 평가합니다. 이 연구는 요약 작업에서 공정성의 중요성을 강조하고 공정성을 고려한 NLP 모델의 미래 연구를 위한 기준을 설정합니다.



### Rapid Response: Mitigating LLM Jailbreaks with a Few Examples (https://arxiv.org/abs/2411.07494)
- **What's New**: 본 논문은 Large Language Models (LLMs)의 남용 가능성을 줄이기 위해 새로운 접근 방식인 Jailbreak Rapid Response를 제안합니다. 기존의 공격에 대한 방어에 중점을 두는 대신, 몇 가지 공격을 관찰한 후 새로운 유형의 jailbreak 공격에 신속하게 대응하는 기술을 개발하는 것입니다.

- **Technical Details**: RapidResponseBench라는 새로운 벤치마크를 도입하여, 다양한 jailbreak 공격 전략에 대한 방어의 효과성을 평가합니다. 우리는 5개의 신속 대응 기법을 평가하며, jailbreak proliferation을 활용하여 관찰된 예제에서 유사한 추가 jailbreak을 자동으로 생성합니다. 우리의 방법은 입력 기반의 언어 모델에서 선택한 공격 범주별로 단 하나의 예제를 관찰한 후 공격 성공률을 크게 줄입니다.

- **Performance Highlights**: 최상의 방법은 공격 성공률(ASR)을 in-distribution 세트에서 240배 이상, out-of-distribution 세트에서 15배 이상 감소시켰습니다. 정책의 품질과 생성된 예제 수가 방어의 효과성에 중요한 역할을 한다는 점도 강조되었습니다.



### Controlled Evaluation of Syntactic Knowledge in Multilingual Language Models (https://arxiv.org/abs/2411.07474)
- **What's New**: 이번 연구에서는 인적 자원이 적은 세 가지 언어(Basque, Hindi, Swahili)에서의 언어 모델(LM)의 구문적 일반화 능력을 평가하기 위해 목표 지향적 구문 평가 테스트를 개발했습니다. 이는 고자원 언어에서 사용된 방법을 저자원 언어에 확장한 것으로, LM의 다양한 구조에 대한 이해를 높이는 데 기여합니다.

- **Technical Details**: 연구는 Basque 언어의 보조 동사 일치(auxiliary verb agreement), Hindi의 분리 능동(split ergativity), Swahili의 명사 클래스 일치(noun class agreement)와 같은 뚜렷한 형태 통사적 현상에 대해 세 가지 TSE 사례 연구를 기반으로 합니다. 각 언어에서의 구문적 현상은 LM의 예측 정확성에 맞추어 설계된 최소 쌍(minimal pairs)을 포함하여 평가됩니다.

- **Performance Highlights**: LM은 Basque 언어에서의 일치에 대해 대체로 좋은 성능을 보였으나, 간접 목적어가 포함된 문장에서는 오류가 발생하였습니다. Hindi에서는 숙련된 동사 형태 선택에 성공했지만, multilingual BERT는 문법적으로 올바르지 않음에도 불구하고 습관적(aspectual) 형태를 선호하는 편이었습니다. Swahili에서는 주어의 명사 클래스와 일치하는 기술적 과제가 특히 어려웠고, XGLM-4.5B 모델은 유사한 크기의 모델보다 성능이 떨어지는 경향을 보였습니다.



### IdentifyMe: A Challenging Long-Context Mention Resolution Benchmark (https://arxiv.org/abs/2411.07466)
Comments:
          9 pages, 5 figures

- **What's New**: 이번 연구에서는 LLMs(대규모 언어 모델)의 핵심 연합 해결(coreference resolution) 능력을 평가하기 위한 새로운 벤치마크인 IdentifyMe를 소개합니다. 이 벤치마크는 다국적 선택형 질문(MCQ) 형식으로 제공되어 LLM의 참조 이해도를 보다 효과적으로 측정할 수 있도록 합니다.

- **Technical Details**: IdentifyMe는 문서에서 언급된 객체를 식별하는 MCQ 기반 벤치마크로, LitBank와 FantasyCoref라는 두 가지 장문 코어페런스 데이터셋에서 파생된 언급을 사용합니다. 벤치마크는 특정 유형의 언급(대명사 및 명사)을 필터링하고 난이도 조정을 위한 휴리스틱(heuristics)을 적용하여 모델이 보다 복잡한 문제를 해결하도록 만듭니다.

- **Performance Highlights**: 가장 높은 점수를 기록한 모델인 GPT-4o는 81.9%의 정확도를 달성하였으며, 이는 현재 LLMs의 참조 능력이 상당히 우수하지만 여전히 개선의 여지가 있음을 보여주고 있습니다. 또한, 모델은 대명사 언급을 해결하는 데 더 큰 어려움을 겪었으며 이는 표면 정보가 제한적이기 때문입니다.



### DecoPrompt : Decoding Prompts Reduces Hallucinations when Large Language Models Meet False Premises (https://arxiv.org/abs/2411.07457)
- **What's New**: 본 논문은 LLM(대형 언어 모델)이 허위 전제(false premise)로 인한 환각(output hallucination)을 생성하는 문제를 해결하기 위한 새로운 프롬프트 프로세스인 DecoPrompt를 제안합니다. DecoPrompt는 사용자가 제공한 허위 전제 질문의 엔트로피(entropy) 기반 불확실성(uncertainty)을 고려하여 환각 생성을 감소시킵니다.

- **Technical Details**: DecoPrompt는 높은 로짓 기반 불확실성(logit-based uncertainty)을 가진 모델 출력의 필터링 또는 수정하는 기존의 환각 완화 방법들로부터 영감을 얻었습니다. 이 알고리즘은 LLM을 활용하여 허위 전제를 '디코드(decode)' 하면서도 실제로 환각 출력을 생성하지 않습니다. 이를 통해 다양한 LLM 크기와 데이터셋에서 실험을 진행해 DecoPrompt의 효과를 증명하였습니다.

- **Performance Highlights**: DecoPrompt는 Fictitious 및 Authorship mismatches 두 개의 데이터셋에서 환각을 효과적으로 감소시키며, 다양한 LLM에 대한 이식성(cross-model transferability)을 보입니다. 이는 LLM의 크기나 프라이빗 모델의 로짓 접근 불가와 같은 문제를 해결할 수 있는 강력한 프롬프트 기반 접근 방식입니다.



### Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection (https://arxiv.org/abs/2411.07446)
- **What's New**: 이 연구에서는 Exemplar-Guided Reflection with Memory 메커니즘(ERM)을 소개하여 대형 언어 모델(LLMs)의 프롬프트 최적화를 보다 효율적이고 정확하게 구현하는 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 생성된 Exemplars(예시)를 통해 피드백 생성 과정을 추가로 안내하는 Exemplars-guided reflection 메커니즘을 기반으로 합니다. 또한, 역사적 피드백 정보를 최대한 활용하기 위해 두 가지 종류의 메모리를 구축하여 보다 효과적인 Exemplars 검색을 지원합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 LIAR 데이터셋에서 F1 점수를 10.1 포인트 향상시키고, ProTeGi에서 최적화 단계 수를 절반으로 줄이는 등 이전의 최첨단 모델들을 초월하는 성과를 보여주었습니다.



### Untangling Hate Speech Definitions: A Semantic Componential Analysis Across Cultures and Domains (https://arxiv.org/abs/2411.07417)
- **What's New**: 본 논문은 문화적 요인의 영향을 받는 증오 발언(hate speech) 정의를 분석하기 위해 종합적 의미 구성 분석(Semantic Componential Analysis, SCA) 프레임워크를 제안합니다. 첫 번째로 다섯 개의 도메인(온라인 사전, 연구 논문, 위키백과 기사, 법률, 온라인 플랫폼)에서 유래된 증오 발언 정의의 데이터셋을 생성하고 이를 분석하였습니다.

- **Technical Details**: 분석 결과, 증오 발언 정의는 도메인에 따라 상이한 구성 요소를 가지며, 많은 도메인에서 서로의 정의를 차용하고 있음을 드러냈습니다. 저자는 3개의 오픈 소스 LLM을 활용하여 제안된 데이터셋을 기반으로 제로샷(zero-shot) 모델 실험을 수행했습니다. 그 결과, 정의의 복잡도에 따라 LLM의 반응이 민감하게 변화한다는 사실을 발견했습니다.

- **Performance Highlights**: SCA 프레임워크를 통해 문화적 맥락과 도메인별로 정의가 어떻게 다르게 작용하는지를 효율적으로 분석할 수 있음을 입증했습니다. 이는 LLM의 성능에 특정 정의가 미치는 영향을 탐구하는 중요한 시사점을 제공합니다.



### Using Generative AI and Multi-Agents to Provide Automatic Feedback (https://arxiv.org/abs/2411.07407)
- **What's New**: 이번 연구는 교육 환경에서의 자동화된 피드백 제공을 위한 Generative AI와 Multi-Agent 시스템의 활용을 조사하였습니다. 특히 학생들이 과학 평가에서 작성한 답변에 대한 피드백 품질을 향상시키기 위해 AutoFeedback이라는 이름의 Multi-Agent 시스템을 개발하였습니다.

- **Technical Details**: 이 연구에서는 피드백 생성을 위한 AI 에이전트와 이를 검증 및 수정하는 에이전트 두 개로 구성된 Multi-Agent 시스템을 개발했습니다. AutoFeedback 시스템은 240명의 학생이 작성한 답변 데이터셋을 기반으로 하며, 단일 에이전트 LLM과의 성능 비교를 통해 피드백의 과다 칭찬(over-praise)과 과도한 추론(over-inference) 문제를 현저히 감소시켰습니다.

- **Performance Highlights**: 연구 결과, AutoFeedback 시스템은 과다 칭찬과 과도한 추론 오류의 발생을 크게 줄였으며, 더욱 정확하고 교육적으로 유용한 피드백을 제공하는 것으로 나타났습니다. 이는 교육 현장에서 자동화된 피드백 생성을 위한 신뢰할 수 있는 솔루션을 제안하며, 개인화된 학습 지원의 가능성을 강조합니다.



### Controllable Context Sensitivity and the Knob Behind I (https://arxiv.org/abs/2411.07404)
- **What's New**: 이 논문에서는 언어 모델이 문맥(context)과 이전 지식(prior knowledge) 간의 의존도를 조절할 수 있는 방법을 탐구합니다. 이를 통해 언어 모델이 질문에 답변할 때 문맥에서 오는 정보와 내재된 지식 중 무엇을 더 신뢰해야 하는지를 조율하는 메커니즘을 조명합니다.

- **Technical Details**: 연구진은 controllable context sensitivity (CCS)라는 과제를 설정하여 모델이 특정 문맥(예: 파리(PARIS)라는 대도시가 잉글랜드에 있다고 가정하고)에서 문맥을 따를 것인지, 아니면 이전 지식을 따를 것인지를 구분하도록 합니다. 실험에서는 Llama-3.1, Mistral-v0.3, Gemma-2 모델을 사용하여 85-95%의 정확도로 과제를 해결했습니다. 이 과정에서 기계적 해석 가능성(mechanistic interpretability) 도구를 활용하여 모델의 레이어(layer) 중 문맥 민감도가 가장 높은 부분을 규명합니다.

- **Performance Highlights**: 연구의 결과 나타난 바에 따르면, 문맥과 이전 지식 간의 의사결정을 효과적으로 지원하는 단일차원(subspace) 메커니즘은 다양한 대형 언어 모델에서 공통적으로 나타났습니다. 이 발견은 보다 강력한 언어 모델을 개발하는 데 방향성을 제시하며, 컨트롤 가능한 컨텍스트와 이전 지식에 대한 의존도를 조절할 수 있는 가능성을 보여줍니다.



### Beyond Keywords: A Context-based Hybrid Approach to Mining Ethical Concern-related App Reviews (https://arxiv.org/abs/2411.07398)
- **What's New**: 이번 연구에서는 도메인 특화된 가설을 사용하는 혁신적인 방법을 제안하여 윤리적인 문제와 관련된 모바일 앱 리뷰를 효과적으로 추출하는 자연어 처리(NLP) 기반의 접근 방식을 탐구합니다.

- **Technical Details**: 연구는 43,647개의 정신 건강 관련 앱 리뷰를 활용하여 자연어 추론(NLI)과 디코더 전용 대형 언어 모델(LLM)을 통합하는 하이브리드 방법을 개발했습니다. 이 방법은 도메인 특화된 사생활 가설을 적용하여 윤리적 문제와 관련된 앱 리뷰를 분류하고 추출하는 데 중점을 두었습니다. DeBERTa-v3-base-mnli-fever-anli NLI 모델이 도메인 특화된 가설을 사용하여 최상의 성능을 보여주었으며, Llama3.1-8B-Instruct LLM이 앱 리뷰 분류에 가장 뛰어난 성능을 나타냈습니다.

- **Performance Highlights**: 제안된 방법은 기존의 키워드 기반 접근법으로는 식별할 수 없었던 1,008개의 새로운 사생활 관련 리뷰를 추가로 추출하는 데 성공하였습니다. 이는 NLI와 LLM을 결합한 접근 방식의 효과성을 입증하는 결과입니다.



### Toward Optimal Search and Retrieval for RAG (https://arxiv.org/abs/2411.07396)
Comments:
          Accepted to NeurIPS 2024 Workshop ATTRIB

- **What's New**: 이 연구는 Retrieval-augmented generation (RAG) 파이프라인에서 Retriever와 Reader의 성능 향상을 위한 최적화 방법을 모색합니다. 특히, Question Answering (QA) 작업에서의 성과 관계를 실험을 통해 여러 통찰을 제공합니다.

- **Technical Details**: RAG 파이프라인은 두 개의 별도 시스템으로 구성되어 있으며, 이는 검색을 통해 관련 문서를 식별하는 retriever와 LLM (Large Language Models)으로 구성된 reader입니다. 연구에서는 standard QA와 attributed QA의 성능 차이에 대한 분석을 수행하였으며, RAG 성능에 대한 검색 정확도와 기억 효율성을 최적화한 다양한 매개변수에 대한 영향을 검토했습니다.

- **Performance Highlights**: 검색 정확도를 낮추는 것이 RAG 성능에 미치는 영향은 미미하지만, 검색 속도와 메모리 효율성은 증가할 수 있다는 점이 발견되었습니다. 또한, 노이즈를 주입한 검색 결과는 성능 저하를 초래하며, 이전 보고서와는 달리 블랙홀 기준 이상의 성능 향상을 증명하는 설정은 발견되지 않았습니다.



### Isochrony-Controlled Speech-to-Text Translation: A study on translating from Sino-Tibetan to Indo-European Languages (https://arxiv.org/abs/2411.07387)
- **What's New**: 최근 몇 년 간 음성 번역(End-to-end speech translation, ST) 기술에 대한 많은 관심이 집중되고 있습니다. 본 논문에서는 ST 모델의 지속시간 정렬(duration alignment) 구성 요소를 개선하여 번역 길이를 제어하는 새로운 방법을 제안합니다. 이를 통해 음성과 정지(pause) 세그먼트의 지속시간을 고려한 번역 품질을 높이고자 합니다.

- **Technical Details**: 이 연구에서는 sequence-to-sequence ST 시스템을 사용하여 음성과 정지의 지속 시간을 예측하며, 변환 과정을 통해 번역 길이를 제어합니다. 디코더에 타이밍 정보를 제공하고, 남아 있는 음성 및 정지의 지속 시간을 추적하면서 번역을 생성하는 방식으로 진행됩니다. 또한, Sino-Tibetan 언어와 영어 간의 ST를 위해 CoVoST2 벤치마크에 대한 성능을 경쟁력 있게 유지하기 위한 전처리 데이터 수집을 제안합니다.

- **Performance Highlights**: Zh-En 테스트 세트에서 Isochrony-Controlled ST는 0.92의 음성 겹침(speech overlap)과 8.9의 BLEU 점수를 달성하였으며, 이는 기존 ST 기준 대비 1.4 BLEU만 감소한 결과입니다. 이로 인해 번역의 자연스러움과 품질을 개선하였습니다.



### BeeManc at the PLABA Track of TAC-2024: RoBERTa for task 1 and LLaMA3.1 and GPT-4o for task 2 (https://arxiv.org/abs/2411.07381)
Comments:
          ongoing work - system report

- **What's New**: 이 보고서는 BeeManc 팀의 PLABA(Plain Language Adaptation of Biomedical Abstracts) 2024 공유 과제 시스템 설명입니다. PLABA 2024의 두 가지 하위 작업에 대한 상세 내용을 담고 있습니다.

- **Technical Details**: 첫 번째 작업에서는 fine-tuned ReBERTa-Base 모델을 사용하여 생물 의학 초록에서 어려운 용어, 전문 용어, 약어를 식별 및 분류하고 F1 score를 보고하였습니다. 두 번째 작업에서는 Llama3.1-70B-Instruct와 GPT-4o를 활용하여 초록 적응 작업을 완료하고 BLEU, SARI, BERTScore, LENS 및 SALSA 점수를 보고하였습니다. 특히, PLABA-2024 평가에서 소형 모델인 fine-tuned RoBERTa-Base가 두 하위 작업에서 각각 3위, 2위를 기록하였으며, 두 작업의 평균 F1 점수에서는 1위를 차지했습니다.

- **Performance Highlights**: 우리는 PLABA 2024의 두 가지 하위 과제(Task 1A 및 1B)에서 각각 836개의 문장을 훈련 세트로 사용하고, 고급 자연어 처리 성능을 자랑하는 RoBERTa 모델을 적용하였습니다. 최종 성과로는 F1 점수에서 1위를 기록했으며, 우리의 fine-tuned 모델과 관련 자원은 URL에서 공유됩니다.



### Multi-head Span-based Detector for AI-generated Fragments in Scientific Papers (https://arxiv.org/abs/2411.07343)
- **What's New**: 본 논문에서는 AI가 생성한 과학 텍스트와 인간이 쓴 텍스트를 구別하기 위한 시스템을 설명합니다. DAGPap24 대회에서 제안된 이 시스템은 다중 작업 학습(multi-task learning) 아키텍처를 활용하여, 생성된 텍스트 조각을 효과적으로 식별하는 방법을 제공합니다.

- **Technical Details**: 우리의 접근법은 두 개의 클래스 분류자(classifier)를 사용하는 다중 작업(multi-task) 아키텍처를 기반으로 합니다. 각 토큰이 포함된 텍스트 시퀀스의 상태 벡터(state vector)를 얻기 위해 다양한 인코더 변형을 고려했습니다. 또한, 변환 인코더(transform-based encoder)로 입력하기 위해 조각을 토큰으로 분할하는 방법의 변형도 포함되었습니다. 이 시스템은 평균 매크로 F1-score를 기준으로 개발 세트에서 0.86에서 0.95로 9%의 품질 향상을 달성했습니다.

- **Performance Highlights**: 대회 데이터셋의 폐쇄형 테스트 부분에서 0.96의 스코어를 기록했습니다. 제안된 방법은 기존의 접근 방식보다 더 좋은 성능을 제공하였으며, 과학 문서의 AI 생성 조각을 효과적으로 탐지하는 데 있어 강력한 결과를 보여줍니다.



### SetLexSem Challenge: Using Set Operations to Evaluate the Lexical and Semantic Robustness of Language Models (https://arxiv.org/abs/2411.07336)
Comments:
          10 pages, 8 figures, NeurIPS 2024 Datasets and Benchmarks track

- **What's New**: 본 논문에서는 SetLexSem Challenge라는 합성 벤치마크를 제시하고, 이를 통해 대규모 언어 모델(Large Language Models, LLMs)의 집합 연산(set operations) 성능을 평가합니다.

- **Technical Details**: SetLexSem은 LLM의 지침 수용 능력의 강건성을 다양한 조건 하에서 평가하며, 집합 구성원(set members)의 성격과 구조에 초점을 맞춥니다. 통계적으로 몇 가지 집합 연산을 수행하는 LLM을 평가한 결과, 이들 모델은 작업과 피연산자의 변화를 결코 견디지 못함을 발견했습니다.

- **Performance Highlights**: 현재 LLM들은 SetLexSem 벤치마크가 평가하는 차원에서 전반적으로 높은 변동성을 보이며, 특히 '속임수' 집합의 쉬운 생성 도전에 대해 취약했습니다. 이는 향후 더 발전된 모델을 설계할 때 중요한 함의를 제공합니다.



### Richer Output for Richer Countries: Uncovering Geographical Disparities in Generated Stories and Travel Recommendations (https://arxiv.org/abs/2411.07320)
Comments:
          Submitted to ARR - October 2024

- **What's New**: 이번 연구는 대형 언어 모델이 지리적 지식에 대한 편향을 분석하며, 여행 추천과 지리 기반 이야기 생성의 두 가지 일반적인 시나리오를 탐구합니다. 특히, 저소득 국가에 대한 추천이 상대적으로 덜 독창적이며 빈곤과 슬픔의 감정을 더 많이 포함하고 있음을 발견했습니다.

- **Technical Details**: 연구는 ShareGPT 데이터를 기반으로 1.7%의 쿼리가 여행 추천, 1.5%가 이야기 생성에 관한 것임을 파악했습니다. 444개의 모델에서 300K 이상의 응답을 분석했으며, 이는 전 세계 150K 이상의 장소에 걸쳐 있습니다. 각 모델에 대해 평균 독창성과 감정 표현을 비교했습니다.

- **Performance Highlights**: 부유한 국가에 비해 저소득 국가에서 생성된 이야기의 65%가 더 많은 고난의 정서를 담고 있으며, 여행 추천은 평균적으로 40% 이상의 독창성 차이를 보였습니다. 이러한 결과는 현재 모델들이 서구 중심적 내용을 생성하고 있음을 나타내며, 다양한 인구 집단에 대한 서비스를 보장하기 위한 더 큰 노력이 필요함을 강조합니다.



### Target-driven Attack for Large Language Models (https://arxiv.org/abs/2411.07268)
Comments:
          12 pages, 7 figures. arXiv admin note: substantial text overlap with arXiv:2404.07234

- **What's New**: 본 논문에서는 기존의 블랙박스 공격 방법의 한계를 극복하기 위해 새로운 목표 기반 블랙박스 공격 기법을 제안했습니다. 이 기법은 클린 텍스트와 공격 텍스트 간의 KL divergence를 최대화하여 공격의 목표를 재정의합니다.

- **Technical Details**: 제안된 방법은 공격 목표에 따라 두 개의 볼록 최적화 문제로 거리 최대화 문제를 변환하며, 이는 프로젝트 경량 하강 알고리즘을 활용하여 공격 텍스트에 해당하는 벡터를 해결하는 데 사용됩니다. 또한, 필요한 두 가지 공격 전략인 토큰 조작(token manipulation)과 잘못된 정보 공격(misinformation attack)을 포함합니다.

- **Performance Highlights**: 다양한 대규모 언어 모델(LLM)과 데이터셋에 대한 실험 결과, 제안된 공격 방식은 기존 방법보다 효과적인 성과를 보였습니다. 이로 인해 모델의 보안과 강건성 향상에 기여할 수 있을 것으로 기대됩니다.



### Can adversarial attacks by large language models be attributed? (https://arxiv.org/abs/2411.08003)
Comments:
          7 pages, 1 figure

- **What's New**: 이 논문은 사이버 공격 및 허위 정보와 같은 적대적 환경에서의 대형 언어 모델(LLM) 출력의 귀속 문제를 다룹니다. 특히 포멀 언어 이론(formal language theory)을 사용하여 언어 식별(language identification) 문제를 조사합니다.

- **Technical Details**: 연구에서는 LLM의 출력을 포멀 언어로 모델링하고, 유한한 텍스트 샘플이 원래 모델을 유일하게 식별할 수 있는지 여부를 분석합니다. 결과적으로 특정 언어 클래스의 비식별성(non-identifiability) 때문에, 세밀한 모델 조정에서의 겹치는 출력에 관한 약간의 가정 하에서도, 출력의 귀속을 특정 LLM에 확실하게 할 수 없는 이론적 한계를 발견했습니다.

- **Performance Highlights**: 추가적으로, Transformer 아키텍처의 표현성 한계를 고려하더라도, 모델에 대한 직접적인 접근이나 포괄적인 모니터링이 있더라도 귀속 노력에 심각한 계산적 장애가 존재하는 것을 보여주었습니다. 이 연구 결과는 적대적 LLM 사용으로 인한 위험을 완화하기 위한 적극적인 조치의 필요성을 강조합니다.



### JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation (https://arxiv.org/abs/2411.07975)
- **What's New**: 이번 연구에서는 이미지 이해와 생성을 통합한 강력한 모델인 JanusFlow를 소개합니다. JanusFlow는 자가 회귀 언어 모델(autoregressive language models)과 정제된 흐름(rectified flow)을 결합하는 미니멀리스트 아키텍처를 통해 기존의 복잡한 모델 구조를 간소화했습니다.

- **Technical Details**: JanusFlow는 독립적인 이해 및 생성 인코더(encoders)를 유지하고, 통합 훈련 중에 이들의 표현을 정렬(aligning)하는 두 가지 주요 전략을 채택하여 성능을 최적화합니다. 이를 통해 이미지 생성 및 이해 작업 간의 간섭을 방지하고 의미적 일관성을 강화합니다.

- **Performance Highlights**: JanusFlow는 텍스트-이미지 생성 및 다중 모드 이해 벤치마크에서 기존의 통합 모델들을 능가하는 성능을 보입니다. 특히, MJHQ FID에서 9.51, GenEval에서 0.63, DPG-Bench에서 80.09%를 기록하며, LLaVA-v1.5 및 Qwen-VL-Chat 같은 전문 모델을 초월합니다. 또한, JanusFlow는 단 1.3B 파라미터로 이 성능을 달성합니다.



### Automatic Album Sequencing (https://arxiv.org/abs/2411.07772)
Comments:
          presented as a late breaking demo in the 25th International Society for Music Information Retrieval Conference; 3 pages in main text, 3 figures in main text; source code available at this https URL

- **What's New**: 앨범 시퀀싱(album sequencing) 과정에서 사용자 친화적인 웹 기반 도구가 도입되었습니다. 이 도구를 통해 비전문가도 쉽게 음악 트랙을 업로드하고, 한 번의 클릭으로 시퀀싱 기법을 실행하여 결과를 시각화할 수 있습니다.

- **Technical Details**: 이 연구에서는 Transformer를 기반으로 한 새로운 앨범 시퀀싱 방법을 소개합니다. 이는 이전 연구의 복잡한 파이프라인을 단순화하여 하나의 모델로 대체하였으며, 알고리즘은 FMA 데이터셋을 기반으로 하여 두 층의 완전 연결 신경망과 두 층의 인코더-디코더 Transformer 모델을 사용합니다.

- **Performance Highlights**: 새로운 방법은 무작위 베이스라인(random baseline)보다 뛰어난 성능을 보였지만, 이전의 내러티브 본질(narrative essence) 접근법에는 미치지 못했습니다. 이 연구의 모든 구현은 공개적으로 제공되며, 비전문가도 사용할 수 있는 사용자 인터페이스가 제공됩니다.



### Direct Preference Optimization Using Sparse Feature-Level Constraints (https://arxiv.org/abs/2411.07618)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 인간 선호와의 정렬을 효율적으로 달성하기 위한 Feature-level Constrained Preference Optimization(FPO)라는 새로운 방법을 제안합니다. FPO는 사전 훈련된 Sparse Autoencoders(SAEs)를 사용하여 훈련의 안정성을 보장하면서 정렬 프로세스를 단순화합니다.

- **Technical Details**: FPO는 base 모델 대신 feature-level 제약 조건을 도입하여 모델의 출력을 조정합니다. 이 방법은 Sparse Autoencoders(SAEs)를 통해 희소성을 강화하고, 이를 통해 메모리 사용량과 계산 복잡성을 효과적으로 줄입니다. FPO는 SimPO와 DPO의 개념을 조합하여 참조 모델 없이도 안정적인 성능을 유지합니다.

- **Performance Highlights**: FPO는 AlpacaEval-2와 Arena-Hard 벤치마크에서 5% 이상의 절대적인 승률 향상을 달성하였으며, TDPO에 비해 계산 비용을 17.6% 줄이는 성과를 보였습니다. 실험 결과는 FPO가 상태-of-더-아트 방법들에 비해 메모리 및 시간 복잡성에서 우수한 성능을 발휘함을 보여줍니다.



### Circuit Complexity Bounds for RoPE-based Transformer Architectur (https://arxiv.org/abs/2411.07602)
- **What's New**: 이번 연구는 Rotary Position Embedding (RoPE)을 사용하는 Transformer 아키텍처의 표현력을 제한하는 엄격한 회로 복잡도 경계(circuit complexity bounds)를 설정합니다. 이 연구를 통해 RoPE 기반 Transformer의 근본적인 한계를 이론적으로 밝혀냈습니다.

- **Technical Details**: RoPE (Rotation Position Embedding)는 절대 및 상대 위치 정보를 인코딩하여 Transformer의 성능을 향상시키는 기술입니다. 연구에서는 RoPE 기반 아키텍처의 각 구성 요소에 대한 회로 복잡도를 체계적으로 조사하였고, 이 모델들이 TC⁰ 회로들로 시뮬레이션될 수 있음을 증명했습니다. 또한, TC⁰ = NC¹이 아닌 경우 poly(n) 정밀도와 O(1) 레이어, 그리고 d ≤ O(n) 조건 하에서 RoPE 기반 Transformer가 산술 문제 또는 부울 포뮬라 값 문제를 해결할 수 없음을 보여주었습니다.

- **Performance Highlights**: RoPE 기반 Transformer는 일반 Transformer 모델에 비해 더 높은 일반화 능력을 나타내며, 긴 컨텍스트 정보 처리에서 우수한 성능을 발휘합니다. 최근의 실험적 결과들은 RoPE가 긴 문서 요약 및 지속적인 대화 등 긴 컨텍스트 작업에서 탁월한 능력을 발휘함을 보여주고 있습니다.



### Entropy Controllable Direct Preference Optimization (https://arxiv.org/abs/2411.07595)
- **What's New**: 본 연구에서는 Direct Preference Optimization (DPO) 방식의 수정인 H-DPO를 제안합니다. H-DPO는 정책의 엔트로피를 조정할 수 있어 분포의 선명도를 높이고, 효과적인 mode-seeking fitting을 가능하게 합니다.

- **Technical Details**: H-DPO는 기존 DPO의 손실 계산 수식을 단순히 수정하여 엔트로피 조정을 통해 성능을 향상시킬 수 있도록 설계되었습니다. 이 접근법에서는 손실 함수의 정규화 항을 변경하여 분포의 엔트로피 H(π)를 직접 제어하게 됩니다.

- **Performance Highlights**: 실험 결과, H-DPO는 다양한 작업에서 DPO보다 우수한 성능을 보여주었으며, 특히 수학적 과제의 pass@$k$ 평가에서 더욱 뛰어난 성과를 나타냈습니다.



### Contrastive Language Prompting to Ease False Positives in Medical Anomaly Detection (https://arxiv.org/abs/2411.07546)
Comments:
          4 pages, 3 figures, 2 tables

- **What's New**: 이 논문은 의료 영상에서의 이상 탐지를 위해 CLIP의 새로운 응용 방법인 Contrastive Language Prompting (CLAP)을 제안합니다. CLAP은 긍정적 및 부정적 텍스트 프롬프트를 활용하여 비정상 영역을 보다 정확하게 식별하고, 일반적으로 나타나는 오탐지를 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: CLAP 방법은 CLIP의 시각적 주의(attention) 메커니즘을 통해 비정상 영역에 대한 시각적 주의를 유도하고, 부정적 프롬프트를 사용해 정상 영역에 대한 주의를 약화시킵니다. 이를 통해 BMAD 데이터셋을 활용한 실험에서 이상 탐지 성능이 향상됨을 보여줍니다. U-Net을 이용한 무감독 이상 탐지(UAD) 방법도 제안되어, 정상 샘플을 기반으로 한 재구성을 통해 비정상 패턴을 처리합니다.

- **Performance Highlights**: BMAD 데이터셋을 사용한 실험을 통해 CLAP 방법이 비정상 영역에 대한 강한 주의 문제를 극복하고, 기존 방법들에 비해 UAD 성능을 향상시킴을 입증했습니다. 특히, 정상 샘플로 훈련된 U-Net이 비정상 패턴 재구성에서 어려움을 겪는다는 점을 이용하여, 더욱 정밀한 이상 진단을 가능하게 했습니다.



### SecEncoder: Logs are All You Need in Security (https://arxiv.org/abs/2411.07528)
- **What's New**: 이번 논문에서는 보안 로그(security logs)를 사용하여 사전 훈련된 SecEncoder라는 특화된 작은 언어 모델(small language model)을 소개합니다. 이는 일반적인 언어 모델들이 가지고 있는 도메인 특정 제한사항을 해결하고, 보안 로그에서 발견되는 고유한 언어와 패턴에 집중하기 위해 설계되었습니다.

- **Technical Details**: SecEncoder는 보안 로그 데이터 세트를 기반으로 사전 훈련된 인코더 전용 모델입니다. 이 모델은 다양한 보안 사건과 관련된 이벤트 및 활동을 포착하는 로그를 분석하고, 이상 탐지(anomaly detection), 로그 검색(log search), 사건 분류(incident classification)와 같은 작업에서 평가됩니다.

- **Performance Highlights**: SecEncoder는 BERTlarge, DeBERTa-v3-large 및 OpenAI의 Embedding(textembedding-ada-002) 모델보다 다양한 작업에서 우수한 성능을 보였습니다. 보안 로그에만 주로 사전 훈련되었음에도 불구하고, 사고 우선순위 설정과 위협 인텔리전스 문서 검색과 같은 로그 분석을 넘는 작업에서도 더 나은 성능을 나타냈습니다. 이는 보안 로그로의 도메인 특정 사전 훈련이 LMs의 성능을 상당히 향상시킬 수 있음을 시사합니다.



### SparrowVQE: Visual Question Explanation for Course Content Understanding (https://arxiv.org/abs/2411.07516)
- **What's New**: 이 논문에서는 Visual Question Answering (VQA) 시스템을 발전시키기 위한 Visual Question Explanation (VQE) 방법을 소개합니다. 기존 VQA의 단순한 대답에서 벗어나, 복잡한 시각적 내용과의 상호작용을 통해 상세한 설명을 제공하게 됩니다.

- **Technical Details**: MLVQE 데이터셋을 구성하였으며, 3억 파라미터를 가진 SparrowVQE라는 새로운 멀티모달 모델을 도입하였습니다. 모델 학습은 세 단계로 이루어지며, 멀티모달 사전 학습, 지침 튜닝, 도메인 세부 튜닝이 포함됩니다. 이 과정에서 SigLIP 모델을 활용하여 시각 정보와 문서(transcript)를 연결합니다.

- **Performance Highlights**: SparrowVQE 모델은 MLVQE 데이터셋에서 높은 성능을 보여주었으며, 다른 다섯 개의 VQA 벤치마크 데이터셋에서도 최첨단 성능을 초과하였습니다. 이는 교육 도메인에서의 VQA 효과성을 향상시키는 데 기여할 것으로 기대됩니다.



### BudgetMLAgent: A Cost-Effective LLM Multi-Agent system for Automating Machine Learning Tasks (https://arxiv.org/abs/2411.07464)
Comments:
          Presented at AIMLSystems '24

- **What's New**: 이번 연구에서는 저비용 및 무비용 모델을 활용하여 복잡한 머신 러닝 (ML) 작업을 해결하기 위한 멀티 에이전트 시스템을 제안합니다. 이전 시스템들이 고비용의 대형 모델에 의존했던 반면, 이 새로운 접근법은 비용 효율성을 강조합니다.

- **Technical Details**: 다양한 LLM 전문가의 조합을 이용한 Multi-Agent 시스템을 사용하며, 여기에는 프로파일링, 과거 관찰의 효율적인 검색, LLM cascade, 간헐적인 전문가 호출을 포함합니다. 실험은 MLAgentBench 벤치마크에서 수행됩니다.

- **Performance Highlights**: 본 시스템은 GPT-4 기반 단일 에이전트 시스템에 비해 평균 94.2%의 비용 절감과 32.95%의 성공률을 기록했습니다. 이는 단일 에이전트 GPT-4 시스템의 22.72%의 성공률에 비해 월등히 높은 수치입니다.



### The Surprising Effectiveness of Test-Time Training for Abstract Reasoning (https://arxiv.org/abs/2411.07279)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 모델의 추론 과정에서 입력 데이터에 기반한 손실을 사용하여 모델 파라미터를 임시로 업데이트하는 테스트 타임 트레이닝(test-time training, TTT)이 모델의 추론 능력을 향상시키는 효과를 탐구했습니다. 특히 Abstraction and Reasoning Corpus (ARC)를 기준으로 효율적인 TTT의 세 가지 주요 요소를 식별했습니다.

- **Technical Details**: TTT의 성공을 위해 세 가지 주 구성 요소가 필요합니다: (1) 유사한 작업에서의 초기 파인튜닝, (2) 보조 작업 형식 및 데이터 증강(augmentation), (3) 인스턴스별 훈련. 이 방법은 LMs가 기존의 Fine-tuned 모델보다 최대 6배 더 높은 정확도를 달성하도록 돕습니다. 8B 파라미터 언어 모델에 TTT를 적용했을 때, ARC의 공개 검증 세트에서 53%의 정확도를 달성했습니다.

- **Performance Highlights**: 우리의 연구 결과, TTT를 통해 모델이 네오-심볼릭 접근 방식의 성능에 버금가는 결과를 도출할 수 있음을 보여주었습니다. 최근의 프로그램 생성 접근 방식과 조합하여, ARC 검증 세트에서 61.9%의 최첨단(public validation SoTA) 정확도를 달성했습니다. 이는 평균 인간 성능에 해당하는 수치입니다.



### Multi-Document Financial Question Answering using LLMs (https://arxiv.org/abs/2411.07264)
- **What's New**: 이 논문에서는 다중 문서 재무 질문 응답을 위한 두 가지 새로운 방법인 RAG_SEM( Retrieval Augmented Generation with Semantic Tagging)과 KG_RAG (Knowledge Graph RAG)를 제안합니다. RAG_SEM은 의미 태깅(semantic tagging)을 이용해 인덱스를 쿼리하여 맥락(context)을 얻고, KG_RAG는 그래프 데이터베이스에서 지식 그래프 삼중(triples)을 검색하여 맥락을 제공합니다.

- **Technical Details**: RAG_SEM과 KG_RAG는 각각 의미 태깅과 지식 그래프를 활용하여 다중 문서 질문 응답 시스템을 향상시키는 방법입니다. RAG_SEM은 문서와 질문에 대해 의미 태깅을 수행한 후, 질문이 주어지면 적절한 파일을 선택하는 데 도움을 줍니다. KG_RAG는 문서로부터 지식 그래프 삼중을 생성하여, 정보 검색 과정에서 의미를 더합니다. 두 방법 모두 2021년부터 2023년까지의 Apple, Microsoft, Alphabet, NVIDIA, Amazon, Tesla의 18개의 10K 보고서 데이터로 검증되었습니다.

- **Performance Highlights**: KG_RAG는 9개의 평가 메트릭에서 RAG_SEM보다 4개 항목에서 우수한 성능을 보여주며, 일반적인 RAG 방식을 현저히 초과하는 결과를 보였습니다. 평가에 사용된 메트릭에는 성실성(faithfulness), 관련성(relevance), 정확성(correctness), 유사성(similarity), LLM 기반 종합 점수, rouge 점수 및 임베딩 유사성이 포함됩니다.



New uploads on arXiv(cs.IR)

### A Theoretical Analysis of Recommendation Loss Functions under Negative Sampling (https://arxiv.org/abs/2411.07770)
Comments:
          main paper 8 pages, 4 figures

- **What's New**: 본 논문에서는 추천 시스템(Recommender Systems, RSs)에서 일반적으로 사용되는 손실 함수(loss function)에 대한 비교 분석을 수행하였습니다. 특히, Binary Cross-Entropy (BCE), Categorical Cross-Entropy (CCE), Bayesian Personalized Ranking (BPR) 손실 함수의 특성을 두 가지 샘플링 설정에서 분석합니다.

- **Technical Details**: 우리는 다양한 네거티브 샘플링(negative sampling) 설정에서 BPR과 CCE가 하나의 네거티브 샘플을 사용할 때 동등하다는 것을 발견했습니다. 또한 모든 손실 함수는 동일한 전역 최소값(shared common global minimum)을 공유함을 보여주었습니다. 평가 메트릭으로는 Normalized Discounted Cumulative Gain (NDCG)와 Mean Reciprocal Rank (MRR)를 사용하였습니다. 각 손실 함수의 NDCG에 대한 확률적 하한(probabilistic lower bound)을 제시하였습니다.

- **Performance Highlights**: 다섯 개의 데이터셋과 네 가지 모델에서 실시한 실험 결과, BPR의 NDCG 경계(bound)가 BCE보다 약하다는 것을 밝혀내며, 이는 BPR이 추천 시스템 학습에서 항상 우수하다는 통상적인 가정에 도전하는 결과입니다. 이론적 발견을 실증적으로 뒷받침하는 실험 결과 또한 제시하였습니다.



### Advancing Sustainability via Recommender Systems: A Survey (https://arxiv.org/abs/2411.07658)
Comments:
          20pages, 10 figures. Working paper: this https URL

- **What's New**: 이 논문은 지속 가능한 추천 시스템(Sustainable Recommender Systems)이라는 중요 연구 분야를 심도 있게 분석하며, 이러한 시스템이 환경적 및 사회적 지속 가능성을 증진하기 위한 방법을 제시합니다. 기존 추천 시스템이 사용자 참여와 경제적 지표에 초점을 맞추고 있던 반면, 지속 가능한 추천 시스템은 사용자 선택을 에코-친화적이고 사회적으로 책임 있는 방식으로 변화시키는 데 초점을 맞추고 있습니다.

- **Technical Details**: 이 논문은 지속 가능한 추천 시스템의 구현과 알고리즘 최적화에 대한 체계적인 연구를 제공합니다. 세부 사항으로는 교통, 음식, 건물 관리 분야에서의 추천 시스템 구현이 포함되어 있으며, 정책 추천(Policy Recommendations), 경로 추천(Route Recommendations), 그리고 소비자 행동 분석 등 다양한 기술적 접근 방식을 사용하고 있습니다. 이 시스템은 사용자의 선호를 예측하고 환경 친화적인 제품 및 서비스를 제안하여 자원 보존(Resource Conservation), 에너지 소비 최적화(Energy Efficiency Optimization), 지속 가능한 소비 촉진(Promotion of Sustainable Consumption) 및 사회적 복지 향상(Enhancement of Social Well-being)의 목표를 달성하도록 설계되었습니다.

- **Performance Highlights**: 연구 결과, 지속 가능한 추천 시스템이 사용자에게 에너지를 절약하면서도 보다 지속 가능한 소비를 촉진한다는 것입니다. 실증 연구에 따르면, 지속 가능한 제품을 위한 추천 시스템은 에너지 사용을 감소시키고 온실가스 배출을 줄이는 데 기여하고 있습니다. 이 시스템은 다양한 분야에서 사용되며, 사용자 경험을 향상시키고 환경적 및 사회적 지속 가능성을 증진하는 데 기여할 수 있는 잠재력이 큽니다.



### Overhead-free User-side Recommender Systems (https://arxiv.org/abs/2411.07589)
Comments:
          arXiv admin note: text overlap with arXiv:2208.09864, arXiv:2403.15757

- **What's New**: 최근에 사용자 측 추천 시스템(user-side recommender systems)의 새로운 패러다임이 제안되었습니다. 이 시스템은 최종 사용자가 직접 구축하고 사용할 수 있습니다. 기존의 제공자 측 추천 시스템과는 대조적으로, 사용자는 불공정한 추천 시스템에 대해 스스로 해결책을 마련할 수 있습니다.

- **Technical Details**: 이 논문에서는 RecCycle이라는 오버헤드 없는 사용자 측 추천 시스템을 제안합니다. RecCycle의 주요 아이디어는 이전 추천 결과를 재활용(recycle)하여 사용자 측에서 추천을 생성하는 것입니다. 이를 통해 추가적인 통신 비용을 없애고, 기존의 추천 시스템과 결합하여 실시간 추천을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, RecCycle은 최신 사용자 측 추천 알고리즘과 동등한 성능을 보이며, 통신 비용을 크게 줄이는 효과를 확인했습니다. 실제 환경인 X(Twitter)에서 사용자들이 요구하는 기능을 갖춘 추천 시스템을 구현할 수 있음을 증명했습니다.



### Towards Automated Model Design on Recommender Systems (https://arxiv.org/abs/2411.07569)
Comments:
          Accepted in ACM Transactions on Recommender Systems. arXiv admin note: substantial text overlap with arXiv:2207.07187

- **What's New**: 딥러닝 모델의 인기가 높아짐에 따라 AI 기반 추천 시스템 개발을 위한 새로운 기회가 생겼습니다. 이에 따라, 모델 아키텍처와 하드웨어를 공동 최적화하는데 필요한 설계 자동화(Design automation)가 더욱 중요해졌습니다.

- **Technical Details**: 우리는 가중치 공유(weight sharing)를 활용하여 풍부한 솔루션 공간을 탐색하는 새로운 패러다임을 소개합니다. 이 패러다임은 다양한 연산자(operators)와 밀집 연결(dense connectivity), 치수 탐색(dimension search) 옵션을 포함하는 대규모 슈퍼넷(supernet)을 구축하여 추천 시스템의 데이터 다중 양상(data multi-modality)과 이질성(heterogeneity) 문제를 해결합니다. 또한, 다양한 Processing-In-Memory (PIM) 구성을 통해 하드웨어 효율적인 모델을 생성합니다.

- **Performance Highlights**: 개발한 모델은 세 가지 Click-Through Rates (CTR) 예측 벤치마크에서 뛰어난 성과를 보여주었으며, 수동으로 설계된 모델과 AutoML로 제작된 모델보다 우수한 성능을 기록했습니다. 공동 설계 측면에서는 2배 FLOPs 효율성, 1.8배 에너지 효율성, 그리고 1.5배 성능 개선을 달성했습니다.



### Feature Interaction Fusion Self-Distillation Network For CTR Prediction (https://arxiv.org/abs/2411.07508)
- **What's New**: FSDNet은 CTR 예측을 위한 새로운 프레임워크로, plug-and-play 아키텍처에 자가 증류(self-distillation) 모듈을 통합하여 정보 공유를 개선하며 성능을 향상시킵니다.

- **Technical Details**: FSDNet은 명시적(explicit) 그리고 암묵적(implicit) 특징 상호작용을 각 레이어에서 연결하여 정보 공유를 강화합니다. 자가 증류를 통해 가장 깊은 연결된 레이어를 teacher 모델로 사용하고, 이를 통해 더 얕은 레이어(학생 모델)의 학습을 유도합니다. 이러한 접근 방식은 전통적인 teacher-student 구조의 복잡성을 줄이고, 보다 효율적인 지식 전이를 가능하게 합니다.

- **Performance Highlights**: FSDNet은 4개의 벤치마크 데이터셋에서 실험을 수행하여 효과성과 일반화 능력을 검증하였습니다. 결과적으로, FSDNet은 기존 모델들보다 더 나은 성능을 보이며 자원 제약이 있는 환경에서도 적용 가능함을 입증했습니다.



### AdaS&S: a One-Shot Supernet Approach for Automatic Embedding Size Search in Deep Recommender System (https://arxiv.org/abs/2411.07504)
- **What's New**: 본 논문에서는 AdaS&S라는 새로운 프레임워크를 제안하여 기존의 Automatic Embedding size Search (AES) 방법의 여러 문제를 해결하고자 하였습니다. 이 방법은 다양한 후보 임베딩을 포함하는 슈퍼넷(supernet)을 구성하고, 이를 통해 안정적이고 효과적인 임베딩 크기를 추출할 수 있도록 합니다.

- **Technical Details**: AdaS&S 프레임워크는 두 단계로 구성됩니다: 첫 번째 단계에서는 파라미터 훈련과 임베딩 크기 검색을 분리하여 Adaptive Sampling 방법을 통해 잘 훈련된 슈퍼넷을 생성합니다. 두 번째 단계에서는 강화 학습(Reinforcement Learning) 기반의 검색 과정을 통해 모델 성능을 향상시키는 임베딩 크기를 도출하며, 자원 제약(resource constraint)에 맞추기 위해 자원 경쟁 패널티를 도입합니다.

- **Performance Highlights**: AdaS&S 방법은 공공 데이터셋에서 실험을 통해 AUC를 약 0.3% 개선하고, 모델 파라미터를 약 20% 절감하는 성과를 보였습니다. 또한, 검색 결과의 안정성이 다른 방법들에 비해 현저히 뛰어난 것으로 나타났습니다.



### Multi-Document Financial Question Answering using LLMs (https://arxiv.org/abs/2411.07264)
- **What's New**: 이 논문에서는 다중 문서 재무 질문 응답을 위한 두 가지 새로운 방법인 RAG_SEM( Retrieval Augmented Generation with Semantic Tagging)과 KG_RAG (Knowledge Graph RAG)를 제안합니다. RAG_SEM은 의미 태깅(semantic tagging)을 이용해 인덱스를 쿼리하여 맥락(context)을 얻고, KG_RAG는 그래프 데이터베이스에서 지식 그래프 삼중(triples)을 검색하여 맥락을 제공합니다.

- **Technical Details**: RAG_SEM과 KG_RAG는 각각 의미 태깅과 지식 그래프를 활용하여 다중 문서 질문 응답 시스템을 향상시키는 방법입니다. RAG_SEM은 문서와 질문에 대해 의미 태깅을 수행한 후, 질문이 주어지면 적절한 파일을 선택하는 데 도움을 줍니다. KG_RAG는 문서로부터 지식 그래프 삼중을 생성하여, 정보 검색 과정에서 의미를 더합니다. 두 방법 모두 2021년부터 2023년까지의 Apple, Microsoft, Alphabet, NVIDIA, Amazon, Tesla의 18개의 10K 보고서 데이터로 검증되었습니다.

- **Performance Highlights**: KG_RAG는 9개의 평가 메트릭에서 RAG_SEM보다 4개 항목에서 우수한 성능을 보여주며, 일반적인 RAG 방식을 현저히 초과하는 결과를 보였습니다. 평가에 사용된 메트릭에는 성실성(faithfulness), 관련성(relevance), 정확성(correctness), 유사성(similarity), LLM 기반 종합 점수, rouge 점수 및 임베딩 유사성이 포함됩니다.



### Query Optimization for Parametric Knowledge Refinement in Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2411.07820)
- **What's New**: 본 논문에서는 Extraction-Refinement-Retrieval-Read (ERRR) 프레임워크를 소개합니다. ERRR은 Retrieval-Augmented Generation (RAG) 시스템의 정보 검색 과정에서 발생하는 사전 검색 정보 격차를 해소하기 위해 설계된 새로운 접근 방식입니다. 이 프레임워크는 대형 언어 모델(LLM)의 특정 지식 요구사항을 충족하기 위한 쿼리 최적화를 포함합니다.

- **Technical Details**: ERRR 프레임워크는 LLM에서 파라메트릭 지식을 추출하는 것으로 시작하여, 이 지식을 바탕으로 쿼리를 세밀하게 조정하는 전문 쿼리 최적화기를 사용합니다. 이 과정에서 오직 가장 관련성 높은 정보만을 검색하여 정확한 응답을 생성하는 데 필요한 정보를 보장합니다. ERRR은 지식 증류를 통해 더 큰 모델에서 개선된 소형 조정 가능한 모델을 쿼리 최적화기로 사용합니다.

- **Performance Highlights**: 다양한 질문-응답(QA) 데이터셋에서 ERRR은 기존의 기준선을 지속적으로 초월하는 성능을 발휘했습니다. ERRR은 여러 가지 검색 시스템과 설정에서도 높은 적응성을 보이며 RAG 시스템의 유용성과 정확성을 개선하는 데 매우 효과적임을 입증했습니다.



### Unlocking Legal Knowledge with Multi-Layered Embedding-Based Retrieva (https://arxiv.org/abs/2411.07739)
Comments:
          27 pages, 10 figures

- **What's New**: 이 연구는 법률 지식의 복잡성을 포착하기 위한 다층적인 embedding 기반 검색 방법을 제안합니다. 개별 조항뿐만 아니라 그 구성요소(문단, 조항)와 구조적 집합(책, 제목, 장 등)에도 embedding을 생성하여 법률 정보의 미세함을 캡처하고 있습니다.

- **Technical Details**: 법률 문서의 고유한 계층 구조에 따라, 다양한 세분화의 레벨에서 embedding을 통해 작은 조각에서부터 전반적인 섹션까지 정보를 검색할 수 있게 합니다. 또한, embedding을 통해 법률 문서에 내재된 관계를 표현하며, 이 방법론은 Retrieval Augmented Generation (RAG) 시스템을 통해 사용자 쿼리에 정확하게 응답할 수 있도록 합니다.

- **Performance Highlights**: 이 방법론은 브라질의 법제와 헌법에 주로 초점을 맞추고 있지만, 원칙적으로 공통법 체제에서도 적용 가능하다고 주장합니다. 또한, 이 기법은 법률 분야를 넘어, 계층적인 텍스트로 인코딩된 정보를 조직하고 검색하는 데 유용한 통찰을 제공합니다.



### Enhancing Link Prediction with Fuzzy Graph Attention Networks and Dynamic Negative Sampling (https://arxiv.org/abs/2411.07482)
- **What's New**: 본 논문에서는 전통적인 Graph Neural Networks(GNNs)가 무작위 negative sampling에 의존하는 한계를 보완하기 위해 Fuzzy Graph Attention Networks(FGAT)를 제안합니다. 이 접근법은 fuzzy rough set을 통합하여 동적인 negative sampling과 향상된 노드 특징 집계를 가능하게 합니다.

- **Technical Details**: 본 연구에서는 Fuzzy Negative Sampling(FNS)을 통해 fuzzy 유사도 기반의 고품질 negative edges 선택하는 메커니즘을 도입합니다. FGAT 레이어는 fuzzy rough set 원리를 통합하여 강력하고 구분 가능한 노드 표현을 가능하게 합니다. 이를 통해 GNN의 전반적인 학습 효율성을 높이고 있습니다.

- **Performance Highlights**: 실험 결과, FGAT는 두 개의 연구 협력 네트워크에서 기존의 최첨단 방법들보다 우수한 링크 예측 정확도를 보여주었습니다. 특히, fuzzy rough set의 힘을 활용하여 효과적인 negative sampling과 노드 특징 학습을 구현함으로써 성능이 개선되었습니다.



### Music Discovery Dialogue Generation Using Human Intent Analysis and Large Language Models (https://arxiv.org/abs/2411.07439)
Comments:
          Accepted for publication at the 25th International Society for Music Information Retrieval Conference (ISMIR 2024)

- **What's New**: 본 논문에서는 대화형 음악 검색 시스템의 데이터 생성 프레임워크를 제안하여, 사용자의 음악 발견 대화를 위한 LP-MusicDialog라는 대규모 합성 대화 데이터셋을 생성했습니다. 이를 통해 기존 소규모 인간 대화 데이터셋보다 대화의 일관성 및 자연스러움 측면에서 경쟁력을 가지고 있음을 입증했습니다.

- **Technical Details**: 제안된 프레임워크는 사용자의 의도(intent), 시스템의 행동(action), 음악 속성(musical attributes)을 분석하여 LLM(large language model)을 활용하여 대화형 음악 검색 대화를 생성하는 방법론입니다. 이 과정은 대화 의도 분석, 속성 시퀀스 생성, LLM을 통한 발화 생성으로 이루어집니다. 또한, LP-MusicDialog 데이터셋은 288,000개 이상의 음악 대화와 319,000개 이상의 음악 아이템을 포함합니다.

- **Performance Highlights**: 합성된 LP-MusicDialog 데이터셋은 기존의 소규모 인간 대화 데이터셋과 비교하여 대화의 일관성, 아이템 적합성, 자연스러움에서 좋은 성능을 보였으며, 이 데이터셋을 활용하여 훈련된 대화형 음악 검색 모델이 유망한 결과를 나타냈습니다.



New uploads on arXiv(cs.CV)

### Material Transforms from Disentangled NeRF Representations (https://arxiv.org/abs/2411.08037)
- **What's New**: 이번 논문에서는 다양한 장면 간의 재질 변환을 효과적으로 전이할 수 있는 새로운 방법을 제안합니다. 이는 서로 다른 조건(예: 건조하던 상태와 젖은 상태)에서 관찰된 장면 쌍을 토대로 Bidirectional Reflectance Distribution Functions (BRDF)를 매핑하는 방식으로 작동합니다.

- **Technical Details**: 우리의 접근 방식은 분리된 Neural Radiance Field (NeRF) 표현을 기반으로 하며, BRDF 변환을 학습하여 비슷한 재질을 가진 새로운 장면에도 적용할 수 있도록 합니다. 이때, Multi-Layer Perceptron (MLP)을 사용하여 원본 장면 BRDF를 조건화하고 변환을 근사화합니다. 또한, 높은 반사율을 가진 물질의 분해에서 발생하는 한계를 극복하기 위해 새로운 빛 추정 기법을 도입합니다.

- **Performance Highlights**: 종합적으로, 우리는 합성 장면과 실제 객체에 대한 광범위한 실험을 수행하며, 이 방법이 젖은 상태, 페인팅, 코팅 등 다양한 변환을 학습할 수 있음을 입증했습니다. 두 가지 새로운 데이터셋(합성 데이터셋 및 다양한 재료 상태의 피규어 데이터셋)에서 신뢰성 높은 변환 성능을 보여주었습니다.



### Scaling Properties of Diffusion Models for Perceptual Tasks (https://arxiv.org/abs/2411.08034)
- **What's New**: 본 논문에서는 확산 모델( diffusion models)과 반복 계산(iterative computation)을 활용하여 생성 및 시각적 인식(visual perception) 작업에 대해 효과적으로 접근할 수 있는 새로운 패러다임을 제안합니다. 이전 연구들과 달리 깊이 추정(depth estimation), 광학 흐름(optical flow), 세분화(segmentation) 작업을 이미지-투-이미지 변환(image-to-image translation)으로 통합하였으며, 이러한 작업에 대한 훈련 및 테스트 시간 컴퓨팅(compute) 규모에 따른 변화도 분석하였습니다.

- **Technical Details**: 논문에서는 효율적인 확산 모델의 훈련 및 추론을 위해 다양한 기술을 제시합니다. 특히, 밀집 모델(dense models)과 전문가 혼합 모델(mixture-of-expert models)의 사전 훈련(pre-training)을 통해 확산 모델의 성능을 최적화합니다. 또한, 여러 테스트 시간 컴퓨팅 기술, 예를 들어 확산 단계(diffusion steps)를 증가시키거나, 테스트 시간 집계(test-time ensembling) 및 가변 소음 스케줄(noise variance schedules)을 통한 컴퓨팅 재구성을 실험합니다.

- **Performance Highlights**: 우리의 접근법을 통해, 기존의 최첨단 방법들에 비해 데이터와 컴퓨팅 자원을 현저히 절약하면서도 개선된 성능을 보였습니다. 다양한 벤치마크에서 최첨단 결과(state-of-the-art results)를 달성하였으며, 이를 통해 광범위한 시각적 인식 작업에 대한 일반화된 전문가 혼합 모델(generalist mixture-of-experts model)의 효율적인 훈련이 가능함을 보여줍니다.



### GaussianAnything: Interactive Point Cloud Latent Diffusion for 3D Generation (https://arxiv.org/abs/2411.08033)
Comments:
          project page: this https URL

- **What's New**: 본 연구는 기존의 3D 생성 방식에서의 입력 포맷, 잠재 공간 설계, 그리고 출력 표현의 문제점을 해결하기 위해 새로운 3D 생성 프레임워크인 GaussianAnything를 제안합니다. 이 프레임워크는 포인트 클라우드 구조의 잠재 공간을 사용하여 확장 가능하고 고품질의 3D 생성을 지원합니다.

- **Technical Details**: GaussianAnything는 Variational Autoencoder (VAE)를 기반으로 하며, 다중 시점 RGB-D(Depth) 이미지가 입력으로 사용됩니다. 이 프레임워크는 3D 모양 정보를 보존하는 독특한 잠재 공간 설계를 포함하고 있으며, 이로 인해 형상-질감(Shape-Texture) 분리를 개선합니다. 또한, Cascaded Latent Diffusion 모델을 도입하여 다양한 입력에 대해 조건부 3D 생성을 지원합니다.

- **Performance Highlights**: 실험 결과, GaussianAnything는 텍스트 및 이미지 조건부 3D 생성에서 기존 방법보다 뛰어난 성능을 보여주었으며, 여러 데이터셋에서 효과성을 입증하였습니다.



### Wavelet Latent Diffusion (Wala): Billion-Parameter 3D Generative Model with Compact Wavelet Encodings (https://arxiv.org/abs/2411.08017)
- **What's New**: 본 논문에서는 Wavelet Latent Diffusion (WaLa)라는 새로운 접근 방식을 소개하여 3D 형태를 wavelet 기반의 압축된 잠재 인코딩으로 인코딩합니다. 이를 통해 $256^3$의 signed distance field를 $12^3 	imes 4$의 잠재 그리드로 압축하여 2427배의 압축 비율을 달성했습니다.

- **Technical Details**: WaLa는 압축 과정에서 정보 손실 없이 wavelet 표현을 더욱 압축하여, diffusion 기반의 생성 모델을 효율적으로 확장할 수 있도록 합니다. 구체적으로는 convolution 기반의 VQ-VAE 모델을 사용하여 압축을 진행하며, 이는 약 10억 개의 매개변수를 포함하고 있습니다.

- **Performance Highlights**: WaLa는 고해상도 3D 생성에서 최첨단 성능을 보여주며, 다양한 입력 모달리티를 지원합니다. 모델의 생성 속도는 2~4초이며, 제어된 생성 또한 가능하여 복잡한 기하학, 신뢰할 수 있는 구조와 세밀한 토폴로지를 가진 3D 형태를 생성합니다.



### Artistic Neural Style Transfer Algorithms with Activation Smoothing (https://arxiv.org/abs/2411.08014)
Comments:
          8 pages,7 figures

- **What's New**: 본 논문에서는 Gatys et al.의 연구를 토대로 Convolutional Neural Networks (CNNs)를 활용한 예술적 스타일 이미지 생성능력을 재구현했습니다. 특히, Neural Style Transfer (NST) 기법의 이미지 기반, 빠른 NST, 그리고 임의 NST를 구현하고 있으며, ResNet을 활용한 활성화 평활화(activation smoothing)의 가능성을 탐구하고 있습니다.

- **Technical Details**: 본 논문에서는 이미지 기반 NST(Neural Style Transfer), 빠른 NST, 그리고 임의 NST를 재구현하고, ResNet 구조와 활성화 평활화를 결합하여 NST 성능을 향상시키는 방법을 제안합니다. 실험 결과는 평활화 변환(smoothing transformation)이 스타일화 결과의 품질 개선에 크게 기여함을 보여줍니다.

- **Performance Highlights**: 다양한 실험 결과를 통해 모델이 생성한 스타일화 이미지의 품질이 개선되었음을 확인하였으며, 활성화 평활화 접근 방식을 적용한 NST가 기존 방법들에 비해 뛰어난 성능을 보여주었습니다.



### JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation (https://arxiv.org/abs/2411.07975)
- **What's New**: 이번 연구에서는 이미지 이해와 생성을 통합한 강력한 모델인 JanusFlow를 소개합니다. JanusFlow는 자가 회귀 언어 모델(autoregressive language models)과 정제된 흐름(rectified flow)을 결합하는 미니멀리스트 아키텍처를 통해 기존의 복잡한 모델 구조를 간소화했습니다.

- **Technical Details**: JanusFlow는 독립적인 이해 및 생성 인코더(encoders)를 유지하고, 통합 훈련 중에 이들의 표현을 정렬(aligning)하는 두 가지 주요 전략을 채택하여 성능을 최적화합니다. 이를 통해 이미지 생성 및 이해 작업 간의 간섭을 방지하고 의미적 일관성을 강화합니다.

- **Performance Highlights**: JanusFlow는 텍스트-이미지 생성 및 다중 모드 이해 벤치마크에서 기존의 통합 모델들을 능가하는 성능을 보입니다. 특히, MJHQ FID에서 9.51, GenEval에서 0.63, DPG-Bench에서 80.09%를 기록하며, LLaVA-v1.5 및 Qwen-VL-Chat 같은 전문 모델을 초월합니다. 또한, JanusFlow는 단 1.3B 파라미터로 이 성능을 달성합니다.



### SimBase: A Simple Baseline for Temporal Video Grounding (https://arxiv.org/abs/2411.07945)
Comments:
          Technical report

- **What's New**: 이번 논문에서는 SimBase라는 간단하고 효과적인 temporal video grounding의 기본선을 제시합니다. 최근 temporal grounding의 발전에도 불구하고 네트워크 구조가 점점 복잡해지고 있는 상황에서, 본 논문은 단순한 접근법이 얼마나 효과적일 수 있는지를 탐구합니다.

- **Technical Details**: SimBase는 복잡한 temporal 구조 대신에 경량의 일차원 temporal convolutional layers를 활용합니다. 또한, cross-modal interaction에서는 복잡한 멀티모달 fusion 대신에 기본적인 element-wise product만을 사용합니다. 이러한 간단한 설계를 통해 SimBase는 두 개의 대규모 데이터셋에서 state-of-the-art 성능을 달성했습니다.

- **Performance Highlights**: SimBase는 Charades-STA 벤치마크에서 기존 최첨단 방법을 크게 능가하는 결과를 보여주었습니다. 이 연구는 SimBase가 간단한 듯하면서도 강력한 기본선으로서, temporal video grounding에 대한 새로운 아이디어를 자극하고 미래의 평가를 간소화하는 데 기여할 것으로 기대합니다.



### Learning Disentangled Representations for Perceptual Point Cloud Quality Assessment via Mutual Information Minimization (https://arxiv.org/abs/2411.07936)
- **What's New**: 본 논문에서는 No-Reference Point Cloud Quality Assessment (NR-PCQA) 분야에서 새로운 Disentangled Representation Learning 프레임워크인 DisPA를 제안합니다. 기존 NR-PCQA 모델들은 콘텐츠(내용)와 왜곡(distortion) 정보를 단일 네트워크에서 학습하여 품질 정보를 간과하고 있습니다.

- **Technical Details**: DisPA는 상호 정보(mutual information, MI)를 최소화하도록 정훈 된 이중 분기(disentanglement) 네트워크로, 콘텐츠 인식 및 왜곡 인식 인코더를 포함합니다. 이 프레임워크는 masked auto-encoding 전략을 활용하여 콘텐츠 인식 인코더를 사전 훈련하며, 왜곡을 강조하는 mini-patch 맵을 채택하여 왜곡 인식 인코더가 낮은 수준의 왜곡 패턴에 집중하도록 합니다. 또한, MI 추정기를 활용하여 MI의 상한을 추정하고 이를 최소화하여 명시적인 표현 분리(disentanglement)를 달성합니다.

- **Performance Highlights**: DisPA는 SJTU-PCQA, WPC 및 LS-PCQA와 같은 여러 데이터셋에서 기존의 최첨단 방법들을 능가하는 우수한 성능을 보였습니다. 실험 결과는 DisPA가 NR-PCQA 성능을 크게 향상시킬 수 있음을 입증합니다.



### Isometric Transformations for Image Augmentation in Mueller Matrix Polarimetry (https://arxiv.org/abs/2411.07918)
Comments:
          preprint

- **What's New**: 이 연구에서는 Mueller 행렬 이미지를 위한 물리적으로 일관된 데이터 증강 방법을 새롭게 소개합니다. 기존의 일반적인 방법들은 polarimetric 이미지의 고유한 특성을 보존하지 못하는 문제를 지적하며, 물리적 원칙을 기반으로 한 변형 방법의 필요성을 강조합니다.

- **Technical Details**: Mueller 행렬은 산란된 빛의 편광 상태를 표현하는 4x4 실수 값 전이 행렬로, 이를 통해 입력 및 출력 빛의 Stokes 벡터 간의 관계를 나타냅니다. 연구에서는 공간적 변화 및 편광 정보를 고려한 새로운 데이터 증강 기법을 제안하며, 이를 통해 모델의 일반화 및 성능을 개선합니다.

- **Performance Highlights**: 제안한 증강 방법은 semantic segmentation 작업에서 모델의 일반화와 성능을 크게 향상시키며, 특히 데이터가 제한된 polarimetric 이미지의 학습에 중요한 역할을 할 수 있습니다.



### TLDR: Traffic Light Detection using Fourier Domain Adaptation in Hostile WeatheR (https://arxiv.org/abs/2411.07901)
Comments:
          Under Review at IEEE Transactions of Artificial Intelligence. 10 Pages, 7 Figures

- **What's New**: 이 논문은 LISA와 S2TLD라는 두 개의 데이터셋을 결합하여 교통신호등 인식의 성능을 개선하는 새로운 접근 방식을 제안합니다. 또한, 교통신호등 검출의 클래스 불균형 문제를 해결하기 위해 합성 우천 및 안개 이미지로 이루어진 새로운 타겟 도메인을 생성합니다.

- **Technical Details**: 제안된 방법에서는 이미지를 처리하기 위해 Fourier Domain Adaptation (FDA)을 사용하며, 이는 Fast Fourier Transform (FFT)와 Inverse FFT (iFFT)를 통해 우천 및 안개와 같은 저조도 기후의 특징을 모델에 적용하는 기법입니다. Semi-Supervised Learning (SSL) 기법을 활용하여 레이블 없는 데이터를 더 효과적으로 사용할 수 있습니다.

- **Performance Highlights**: FTC를 통한 성능 향상이 주목할 만합니다. YOLOv8 모델은 Precision에서 5.1860%, Recall에서 14.8009%, mAP50에서 9.5074%, mAP50-95에서 19.5035%의 증가를 달성했습니다. 전체 모델에서의 평균 퍼센트 증가율은 Precision 7.6892%, Recall 19.9069%, mAP50 15.8506%, mAP50-95 23.8099%를 기록하며, 이는 FDA의 효과성을 잘 보여줍니다. 이 결과는 실제 환경에서도 신뢰성 있는 성능을 확보할 수 있는 가능성을 제시합니다.



### Joint multi-dimensional dynamic attention and transformer for general image restoration (https://arxiv.org/abs/2411.07893)
- **What's New**: 이 논문은 다차원 동적 주의(multi-dimensional dynamic attention)와 자가 주의(self-attention)를 U-Net 프레임워크 내에서 결합한 새로운 이미지 복원 아키텍처를 소개합니다. 이를 통해 변형된 이미지 복원 기술을 개선하고자 하며, 복잡한 손상도 효율적으로 처리할 수 있는 방법을 제공합니다.

- **Technical Details**: 제안된 MDDA-former는 인코더-디코더에는 CNN 기반 모듈을, 잠재 계층(latent layer)에는 트랜스포머 블록을 배치하여 U-Net 아키텍처의 다중 스케일 구조적 차이를 완전히 활용합니다. 또한, 다차원 동적 주의 블록(MDAB)을 설계하여 CNN 블록의 용량을 개선하고, 잠재 계층에 있는 효율적인 변환기 블록(ETB)을 통해 전역 정보 추출 능력을 향상시킵니다.

- **Performance Highlights**: 이 방법은 18개 벤치마크 데이터세트에서 다섯 가지 이미지 복원 작업을 수행하는 동안 대부분의 최신 방법보다 성능과 복잡성을 더 잘 조화롭게 조정하여 우수한 성능을 달성했습니다. 해당 방법은 고수준 시각 작업에서도 탁월한 성능을 보였습니다.



### INTRABENCH: Interactive Radiological Benchmark (https://arxiv.org/abs/2411.07885)
Comments:
          Undergoing Peer-Review

- **What's New**: IntRaBench는 3D 의료 영상에서의 인터랙티브 세분화 방법을 효과적으로 평가할 수 있는 새로운 벤치마크 프레임워크입니다. 이 프레임워크는 다양한 데이터셋과 세분화 모델을 포함하며, 임상에서의 실제 사용을 고려하여 개발되었습니다.

- **Technical Details**: IntRaBench는 2D 및 3D 인터랙티브 세분화 방법의 공정하고 재현 가능한 평가를 지원합니다. 특정한 프로트핑(prompting) 및 수정(refinement) 전략을 통해 2D 모델에서도 사용자 상호작용을 간소화하고, 대시(board)에서 인간의 노력을 최소화합니다. 이 벤치마크는 10개의 데이터셋과 7개의 모델을 포함하며, 모두 공개되어 있어 사용자가 쉽게 다운로드 및 전처리할 수 있습니다.

- **Performance Highlights**: IntRaBench는 최초로 2D와 3D 인터랙티브 세분화 방법 간의 공정한 비교를 가능하게 합니다. 연구자들은 이 프레임워크를 이용하여 새로운 방법을 평가하고, 지속적이고 투명한 세분화 모델 평가를 통해 3D 의료 영상 세분화 분야에서의 진전을 추적할 수 있습니다.



### CDXFormer: Boosting Remote Sensing Change Detection with Extended Long Short-Term Memory (https://arxiv.org/abs/2411.07863)
- **What's New**: 본 논문에서는 성능과 효율성을 균형 있게 고려한 새로운 접근 방식인 CDXFormer를 제안합니다. CDXFormer는 XLSTM 기반의 강력한 Feature Enhancer 레이어를 핵심 요소로 사용하여, 공간적 맥락(spatial context) 인식 및 변화 감지를 설명 가능하게 합니다.

- **Technical Details**: CDXFormer는 Scale-specific Feature Enhancer가 포함된 구조로, Cross-Temporal Global Perceptron (CTGP)와 Cross-Temporal Spatial Refiner (CTSR)로 구성됩니다. CTGP는 심층의 영역에서 의미적 차이를 강화하며, CTSR은 얕은 영역에서 세부 정보를 보강합니다. Cross-Scale Interactive Fusion (CSIF) 모듈이 결합되어 글로벌 변화 표현과 공간 응답을 상호작용하게 합니다.

- **Performance Highlights**: CDXFormer는 세 가지 기준 데이터셋에서 이전의 최첨단(SOTA) 접근법을 능가하며, 효율성과 정확성의 뛰어난 균형을 제공합니다.



### Towards Vision Mixture of Experts for Wildlife Monitoring on the Edg (https://arxiv.org/abs/2411.07834)
- **What's New**: 이 논문은 mobile vision transformers의 패치 수준에서의 조건부 계산을 처음으로 탐구하여, 단일 타워 다중 모드(edge) 모델에 적용 가능성을 보여줍니다.

- **Technical Details**: 제안된 모델은 MobileViT를 기반으로 하며, 패치 수준의 mixture of experts를 사용하여 더 적은 파라미터로도 좋은 성능을 보입니다. 초기 실험에서는 MobileViTV2-1.0 모델에 비해 $4X$ 적은 파라미터로 $1$%의 정확도 감소를 보여주었습니다.

- **Performance Highlights**: Cornell Sap Sucker Woods 60 데이터셋의 iNaturalist '21 테스트 데이터를 사용하여 조류 종 분류에서 장기적으로 높은 해상도의 생태 모니터링을 지원할 수 있는 가능성을 보여줍니다.



### Large-scale Remote Sensing Image Target Recognition and Automatic Annotation (https://arxiv.org/abs/2411.07802)
- **What's New**: 이번 논문에서는 LRSAA라는 이름의 대규모 원거리 센싱 이미지에서의 객체 인식 및 자동 라벨링 방법을 제안하고 있습니다. YOLOv11와 MobileNetV3-SSD 객체 탐지 알고리즘을 앙상블 학습(ensemble learning)을 통해 통합하여 모델 성능을 향상시키고 있으며, Poisson disk sampling 분할 기법과 EIOU 지표를 적용하여 훈련 및 추론 과정을 최적화합니다.

- **Technical Details**: 이 연구는 두 가지 고급 객체 탐지 모델인 YOLOv11과 MobileNetV3-SSD를 사용하여 훈련 데이터셋을 구축하고 초기 객체 탐지 모델을 획득합니다. 이후 Poisson disk sampling 기법을 도입하여 이미지 분할을 수행하고, 이로부터 검출된 결과를 원본 대규모 이미지에 매핑(map)하여, 합성 데이터(synthetic data)를 추가하여 재훈련합니다.

- **Performance Highlights**: LRSAA 모델은 XView 데이터셋을 사용하여 훈련 및 평가되었고, Tianjin의 원거리 센싱 이미지에 자동 주석(annotation)을 적용했습니다. 합성 데이터가 포함된 자동 주석 모델은 인식 능력을 크게 향상시켰음을 보여주었습니다. 결과적으로 이는 원거리 센싱 이미지 분석의 효율성과 정확성을 높이는 데 기여할 것입니다.



### Horticultural Temporal Fruit Monitoring via 3D Instance Segmentation and Re-Identification using Point Clouds (https://arxiv.org/abs/2411.07799)
Comments:
          Submitted to IEEE Robotics and Automation Letters

- **What's New**: 이 논문에서는 온실에서 수집된 포인트 클라우드를 기반으로 새로운 방법을 제안하여 과일의 시간적 모니터링을 수행합니다. 이는 3D 구조와 깊이 정보를 활용하여 과일을 분리하고 재식별하는 데 중점을 둡니다.

- **Technical Details**: 제안된 방법은 학습 기반(instance segmentation) 기법을 사용하여 포인트 클라우드에서 과일을 세분화하고, 3D 희소(convolutional neural network)를 통해 각 과일의 설명자를 추출합니다. 이후, 주의 기반(attention-based) 매칭 네트워크를 이용해 이전 데이터 수집에서 각 과일을 재식별합니다. 또한, 새로운 과일 인스턴스의 경우를 처리하기 위해 특정 설명자를 사용하여 확률 분포를 예측합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 다른 기존 기법보다 월등한 성능을 보였습니다. 이는 복잡한 시나리오에서도 정확한 과일 모니터링이 가능하게 하여, 농장 자동화에 기여할 수 있는 새로운 가능성을 제공합니다.



### Feature Fusion Transferability Aware Transformer for Unsupervised Domain Adaptation (https://arxiv.org/abs/2411.07794)
Comments:
          IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 이 논문에서는 비전 트랜스포머(Vision Transformer, ViT)를 이용한 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)을 개선하기 위해 새로운 특징 융합 전이 가능성 인식 트랜스포머(Feature Fusion Transferability Aware Transformer, FFTAT)를 제안합니다. 주요 혁신으로는 패치 전이 가능성 평가를 위한 패치 판별기와 임베딩을 융합하는 기법을 도입하여 모델의 일반화를 향상시킵니다.

- **Technical Details**: FFTAT의 두 가지 주요 구성 요소는 (1) 전이 가능성 그래프 기반 자기 주의 메커니즘(transferability graph-guided self-attention, TG-SA)과 (2) 잠재 공간에서 임베딩 정보를 융합하는 특징 융합(feature fusion) 기술입니다. TG-SA는 고전이 가능성 패치 간의 정보를 강조하고 저전이 가능성 패치 간의 정보를 억제함으로써 도메인 불변 특징을 중심으로 학습할 수 있게 합니다.

- **Performance Highlights**: FFTAT는 기존의 다양한 UDA 벤치마크에서 실험을 통해 UDA 성능을 크게 향상시키며 최신 성능(state-of-the-art, SOTA) 결과를 달성합니다.



### Novel View Synthesis with Pixel-Space Diffusion Models (https://arxiv.org/abs/2411.07765)
- **What's New**: 본 논문에서는 단일 입력 이미지를 기반으로 새로운 시점을 생성하는 Novel View Synthesis (NVS) 문제에 대한 접근방식을 다룹니다. 최신의 Diffusion Model 아키텍처를 사용하여 NVS를 위해 성능이 크게 향상되었으며, 이를 통해 더 간편하고 견고한 end-to-end 솔루션을 제공합니다.

- **Technical Details**: 이 연구에서는 Pixel Space에서 작동하는 Diffusion Model을 통해 기존의 최첨단 기법(SOTA)을 훨씬 뛰어넘는 성과를 거두었습니다. 모델은 Cross-Attention 레이어를 사용하여 기하학적 정보를 효과적으로 인코딩하며, VIVID라는 모델 이름으로 명명되었습니다. 특히, 단일 뷰 데이터셋을 활용하여 훈련할 수 있는 새로운 NVS 훈련 체계를 소개합니다.

- **Performance Highlights**: 이 모델은 RealEstate10K 데이터셋에서 state-of-the-art 성능을 달성했으며, FID(Fréchet Inception Distance)와 PSNR(Peak Signal-to-Noise Ratio) 지표에서 우수한 값을 기록했습니다. 또한, 새로운 데이터 증강 기법을 통해 미지의 장면에 대해 잘 일반화되는 성능을 보여줍니다.



### AdaSemiCD: An Adaptive Semi-Supervised Change Detection Method Based on Pseudo-Label Evaluation (https://arxiv.org/abs/2411.07758)
- **What's New**: 본 논문에서는 새로운 적응형 동적 반지도 학습 방법인 AdaSemiCD를 제안하여, Change Detection (CD) 작업에서 부족한 라벨 데이터와 풍부한 비라벨 데이터 활용을 극대화합니다. 이 방법은 클래스 불균형 문제를 해결하고, 신뢰성 높은 비슷한 예제 수렴을 위한 모듈을 포함합니다.

- **Technical Details**: AdaSemiCD는 두 가지 주요 모듈인 AdaFusion과 AdaEMA를 통합하여 작동합니다. AdaFusion은 불확실한 샘플을 동적으로 식별하고, 신뢰할 수 있는 콘텐츠로 대체하여 높은 품질의 pseudo-labels를 생성합니다. AdaEMA는 훈련 샘플의 품질에 따라 teacher 모델의 업데이트를 조정하여 훈련의 안정성을 높입니다.

- **Performance Highlights**: LEVIR-CD, WHU-CD, CDD 데이터셋에서 실험 결과 AdaSemiCD의 효과성과 범용성이 입증되었습니다. 이 프레임워크는 CD 성능을 현저히 개선하며, 기존 방법들에 비해 훈련 안정성을 향상시킵니다.



### Constraint Learning for Parametric Point Cloud (https://arxiv.org/abs/2411.07747)
- **What's New**: 기존의 CAD 도형 분석 방법들은 주로 기하학적 특징에 초점을 맞추었으나, 본 연구에서는 CAD 도형 내재된 제약 조건(Constraints)의 중요성을 강조하며 이를 기반으로 한 학습 방법을 제안함.

- **Technical Details**: 제안된 CstNet은 두 단계로 구성되어 있으며, Stage 1에서는 B-Rep 데이터 또는 포인트 클라우드에서 제약을 추출하고, Stage 2에서는 좌표 및 제약을 활용하여 CAD 도형의 이해도를 향상시킴. 또한, Parametric 20,000 Multi-modal Dataset을 구축하여 라벨이 지정된 B-Rep 데이터셋의 부족 문제를 해결.

- **Performance Highlights**: CstNet은 공개된 CAD 도형 데이터셋 및 제안된 데이터셋에서 최첨단 성과를 달성하였으며, MCB 데이터셋에서 2.98%의 전체 정확도 향상을 보였고, 360 Gallery 데이터셋에서는 6.74% 향상된 성능을 기록함.



### Efficient 3D Perception on Multi-Sweep Point Cloud with Gumbel Spatial Pruning (https://arxiv.org/abs/2411.07742)
- **What's New**: 이번 논문은 야외 환경에서의 포인트 클라우드(perception) 인식 문제를 다루고 있습니다. 특히, LiDAR 스위프(LiDAR sweeps)를 여러 번 축적하여 인식 정확도를 크게 향상시키는 방법을 제시합니다. 기존 기술들이 멀리 떨어진 또는 가려진 객체를 인식하는 데 한계가 있었음을 지적하며, 계산 비용의 증가를 해결하기 위한 Gumbel Spatial Pruning (GSP) 레이어를 도입했습니다.

- **Technical Details**: GSP 레이어는 포인트 클라우드에서 중복된 포인트를 동적으로 제거할 수 있는 간단하면서도 효과적인 방법입니다. 이 레이어는 기존 네트워크 아키텍처와 원활하게 통합될 수 있으며, FLOPS나 지연(latency)을 추가적으로 증가시키지 않으면서 LiDAR 스위프의 수를 10개에서 최대 40개로 늘릴 수 있습니다. GSP 레이어는 Gumbel Softmax를 활용하여 이진 분류기를 학습하고, 이를 통해 어떤 포인트를 제거할지를 결정합니다. 이러한 접근으로 불필요한 포인트를 제거하더라도 인식 성능은 크게 개선됩니다.

- **Performance Highlights**: nuScenes 데이터셋을 사용한 실험에서, GSP 레이어를 도입함으로써 3D 객체 탐지 및 BEV 맵 분할 작업에서 기존 TransL 및 LargeKernel3D와 같은 3D 인식 네트워크의 성능을 크게 개선했습니다. 안정적인 성능 향상을 가져오면서도 계산 비용의 증가 없이 4배 더 많은 스위프를 사용할 수 있게 되었습니다.



### 3D Focusing-and-Matching Network for Multi-Instance Point Cloud Registration (https://arxiv.org/abs/2411.07740)
Comments:
          Accepted to NeurIPS 2024

- **What's New**: 이 논문은 다중 인스턴스 포인트 클라우드 등록(multi-instance point cloud registration)을 위해 3D 포커싱 및 매칭 네트워크를 제안합니다. 기존의 방법들은 전역 대응(global correspondence)을 얻고 클러스터링하는 방식으로 작동하지만, 복잡한 장면에서 객체들이 클러스터링되고 가려져 있어 정확한 대응을 찾기 어려웠습니다. 본 연구는 객체 중심(object centers)을 먼저 집중적으로 찾고, 이후 CAD 모델과의 매칭을 학습하는 새로운 접근방식을 제시합니다.

- **Technical Details**: 제안된 방법은 크게 두 단계로 나뉘어 있습니다. 첫 번째 단계는 3D 다중 객체 포커싱 모듈(3D multi-object focusing module)로, 입력된 모델 포인트 클라우드와 장면 포인트 클라우드 간의 상관관계를 학습하여 잠재적 객체 중심을 회귀(regressing)하여 찾습니다. 두 번째 단계는 3D 이중 마스킹 인스턴스 매칭 모듈(3D dual masking instance matching module)으로, 모델 포인트 클라우드와 각 객체 중심의 지역 간 쌍별 대응을 예측합니다.

- **Performance Highlights**: Scan2CAD와 ROBI 데이터세트에서 본 방법이 다중 인스턴스 포인트 클라우드 등록 작업에서 새로운 최첨단 성능(state-of-the-art performance)을 달성했습니다. 특히 ROBI 데이터셋에서는 기존의 SOTA인 MIRETR보다 약 7%의 성능 향상을 보였습니다.



### No-Reference Point Cloud Quality Assessment via Graph Convolutional Network (https://arxiv.org/abs/2411.07728)
Comments:
          Accepted by IEEE Transactions on Multimedia

- **What's New**: 이번 논문은 그래프 컨볼루션 네트워크(GCN)를 이용한 새로운 비참조( 비참조) 포인트 클라우드 품질 평가 방법(GC-PCQA)을 제안합니다.

- **Technical Details**: GC-PCQA는 다중 뷰 프로젝션, 그래프 생성, GCN 기반 품질 예측의 세 가지 모듈로 구성됩니다. 포인트 클라우드에 대해 먼저 다중 뷰 프로젝션을 수행하여 수평 및 수직으로 투영된 이미지 세트를 얻습니다. 다음으로, 다양한 투영된 이미지 간의 공간 관계를 바탕으로 지각 일관성 그래프를 구축합니다. 마지막으로, GCN을 통해 구축된 그래프에 대한 추론을 진행하여 서로 다른 투영 이미지 간의 상호 의존성과 상호작용을 특성화하여 품질 예측을 진행합니다.

- **Performance Highlights**: 실험 결과에 따르면, GC-PCQA는 기존의 최첨단 품질 평가 메트릭보다 우수한 성능을 보여주며, 사람의 지각 점수를 더욱 정확하게 예측할 수 있습니다. 개발한 코드는 공개 연구용으로 제공될 예정입니다.



### ALOcc: Adaptive Lifting-based 3D Semantic Occupancy and Cost Volume-based Flow Prediction (https://arxiv.org/abs/2411.07725)
- **What's New**: 이 논문에서는 비전 기반 3D 의미 점유 예측(3D semantic occupancy prediction)과 흐름 예측(flow prediction)을 위한 새로운 개선을 제안합니다. 구체적으로, 깊이 노이즈 제거(depth denoising) 기술과 결합된 폐색 인식 적응형 리프팅 메커니즘을 도입하여 2D에서 3D로의 특징 변환의 견고성을 높이며, 또한 2D 및 3D 특징 간의 의미의 일관성을 강화하는 공유 의미 프로토타입을 이용한 방법론을 제공합니다.

- **Technical Details**: 연구에서는 occlusion-aware adaptive lifting 메커니즘과 함께 깊이 노이즈 제거 기법을 활용하여 2D에서 3D로의 변환을 개선합니다. 또한, 3D와 원래의 2D 모달리티 간의 의미 일관성을 강화하기 위해 공유 의미 프로토타입을 활용하였고, 긴 꼬리 문제(long-tail challenges)를 해결하기 위해 신뢰도(confidence)와 카테고리 기반 샘플링 전략을 제안합니다. 이와 함께 BEV(cost volume) 기반 예측 방법을 제안하여 흐름과 의미 특징을 연결합니다.

- **Performance Highlights**: ALOcc라는 새로운 순수 컨볼루션 아키텍처는 여러 벤치마크에서 최첨단 성능을 달성하며, 특히 Occ3D에서 카메라 가시 마스크가 없는 상태에서 RayIoU 기준으로 2.5%의 절대 향상을 보였습니다. 또한, CVPR24 점유 및 흐름 예측 경연 대회에서 2위에 올랐습니다.



### Emotion Classification of Children Expressions (https://arxiv.org/abs/2411.07708)
- **What's New**: 이 논문은 아동의 감정을 '행복'과 '슬픔'의 두 가지로 분류할 수 있는 얼굴 표정 인식 모델을 제안합니다. 기존의 시스템들이 성인의 얼굴을 기반으로 훈련되어 아동의 정서 표현을 정확히 인식하지 못하는 문제를 해결하기 위해, Squeeze-and-Excitation 블록 및 Convolutional Block Attention 모듈을 활용한 모델이 개발되었습니다.

- **Technical Details**: 제안된 모델은 Batch Normalisation, Dropout, SE Attention 메커니즘을 사용하여 어린이의 감정 분류에서 89%의 정확도를 기록했습니다. 또한 Stable Diffusion 이미지 합성을 통해 데이터 세트를 확장하고 다양화하여 보다 현실적이고 다양한 훈련 샘플을 생성했습니다.

- **Performance Highlights**: 이 연구에서 개발된 모델은 감정 인식의 정확성을 향상시키기 위한 방법을 제시하였으며, 정서 감지 시스템에서 아동을 위한 보다 구체적인 모델의 필요성을 강조하고 있습니다. 모델의 성능 분석에서 다양한 정규화 및 주의 유형에 따라 8개의 실험이 수행되었습니다.



### Enhancing Ultra High Resolution Remote Sensing Imagery Analysis with ImageRAG (https://arxiv.org/abs/2411.07688)
- **What's New**: 본 논문에서는 Ultra High Resolution (UHR) 원거리 감지 이미지 (Remote Sensing Imagery) 분석의 복잡성을 해결하기 위해 훈련이 필요 없는 프레임워크인 ImageRAG를 제안합니다. 이 프레임워크는 UHR 이미지의 긴 컨텍스트 선택(task)을 이미지 분석 과제로 전환하여, 가장 관련성이 높은 부분만 선택적으로 검색하고 집중할 수 있는 혁신적인 이미지 컨텍스트 검색 메커니즘을 설계했습니다.

- **Technical Details**: ImageRAG는 Retrieval-Augmented Generation (RAG) 기법을 바탕으로 설계되었으며, 두 가지 단계인 Retrieval 단계와 Generation 단계를 포함합니다. 이 프레임워크는 Fast path와 Slow path 두 가지 방법으로 작동하여 UHR 이미지에서 중요 정보를 효율적으로 관리할 수 있습니다.

- **Performance Highlights**: ImageRAG는 UHR 이미지를 분석하는 데 있어, 각 모델이 정밀한 세부 사항을 처리할 수 있도록 시각적 컨텍스트를 효과적으로 검색하고 강조합니다. 또한 이 프레임워크는 추가적인 훈련 없이도 적용 가능하여, UHR 원거리 감지 이미지를 보다 정확하고 효율적으로 다룰 수 있는 실제적 솔루션을 제공합니다.



### Fast Disentangled Slim Tensor Learning for Multi-view Clustering (https://arxiv.org/abs/2411.07685)
Comments:
          13 pages,6 figures, will be published to IEEE TMM

- **What's New**: 이 논문에서는 Tensor 기반의 다중 뷰 클러스터링(Multi-View Clustering) 방법 중 특히 제한된 부분을 극복하기 위해 새로운 접근 방법인 빠른 Disentangled Slim Tensor Learning(DSTL)을 제안합니다. DSTL은 높은 차수의 상관관계를 직접 탐색하고, latent semantic-unrelated 정보를 분리하여 성능을 향상시키는데 중점을 두고 있습니다.

- **Technical Details**: DSTL 방법은 행렬 분해(matrix factorization)에 기반하여 다양한 뷰의 latent 표현에서 고차원 상관관계를 탐색합니다. 이 방법은 Robust PCA에 영감을 받아 각 뷰의 표현을 의미와 관련 있는 부분과 의미와 무관한 부분으로 나누어 두 개의 슬림 텐서를 구성합니다. 또한, Consensus Alignment Indicator를 사용하여 semantic 관련 표현의 정렬을 수행하여 정보의 왜곡을 방지합니다.

- **Performance Highlights**: DSTL은 기존의 여러 최첨단 방법들보다 우수한 성능과 효율성을 보이며, 실제 데이터셋에서 더 나은 분류 및 클러스터링 결과를 제공합니다. 관련 코드도 공개되어 있어 연구자들이 쉽게 사용할 수 있습니다.



### Evaluating the Generation of Spatial Relations in Text and Image Generative Models (https://arxiv.org/abs/2411.07664)
- **What's New**: 이번 연구는 텍스트-이미지 변환(model) 모델과 대형 언어 모델(LLM)의 공간적 관계 이해도를 포괄적으로 평가하기 위해 LLM 출력을 이미지로 변환하는 접근 방식을 개발했습니다. 이 방식으로 두 모델 유형의 성능을 시각적으로 평가할 수 있게 되었습니다.

- **Technical Details**: 연구에서는 8개의 저명한 생성 모델(3개의 T2I 모델과 5개의 LLM)을 사용하여 10개의 일반적인 전치사에 대한 공간 관계 이해도를 평가했습니다. 또한 자동 평가 방법의 실용성도 분석했습니다. 기존의 T2I 모델이 주로 텍스트 데이터에 의해 훈련된 LLM보다 공간 관계 생성에서 더 정확하지 않은 결과를 보였습니다.

- **Performance Highlights**: 연구 결과, T2I 모델은 일반 이미지 생성 능력에도 불구하고 공간 관계 이해 및 생성에서는 미흡한 성능을 보였습니다. 특히 LLM이 T2I 모델보다 공간 관계 생성에서 더 높은 정확도를 보였다는 점이 놀랍습니다. 앞으로 모델이 공간적 관계 생성을 더 잘 할 수 있도록 개선할 필요성이 제기되고 있습니다.



### HMIL: Hierarchical Multi-Instance Learning for Fine-Grained Whole Slide Image Classification (https://arxiv.org/abs/2411.07660)
Comments:
          Under Review

- **What's New**: 이 논문에서는 계층 다중 인스턴스 학습(HMIL) 프레임워크를 소개하여 암 진단을 위한 세밀한 분류를 더욱 효과적으로 수행할 수 있는 방법을 제안합니다. 기존의 다중 인스턴스 학습 방식이 계층적 레이블 간의 상관관계를 간과했음을 지적하며, 이를 해결하기 위해 새로운 접근 방식을 취했습니다.

- **Technical Details**: HMIL은 두 개의 분기가 있는 구조로 구성되며, 하나는 대략적 분류를 위한 coarse branch이고, 다른 하나는 세밀한 분류를 위한 fine branch입니다. 이 프레임워크는 인스턴스 및 배갑 수준에서 계층 정렬을 통해 학습을 안내하며, 클래스별 주의(attention) 메커니즘과 감독 대조 학습(supervised contrastive learning)을 통합하여 모델의 분별력을 향상시킵니다.

- **Performance Highlights**: 대규모 자궁 경부암(CCC) 데이터셋과 두 개의 공개 조직학 데이터셋(BRACS, PANDA)에서 extensive 실험을 통해, HMIL이 기존의 기준 모델들과 비교하여 최첨단(class-wise) 및 전반적인 성능을 달성했음을 입증했습니다. 이는 레이블 계층을 모델에 통합하는 것의 중요성을 강조합니다.



### Understanding Audiovisual Deepfake Detection: Techniques, Challenges, Human Factors and Perceptual Insights (https://arxiv.org/abs/2411.07650)
- **What's New**: 이 연구는 복합적인 심층fake (deepfake)를 탐지하기 위해 음향(오디오)과 시각(비주얼) 모달리티를 함께 분석하는 접근 방식을 다룬 최초의 종합적인 조사입니다.

- **Technical Details**: 이 연구는 네 가지 유형의 deepfake를 정의하고 각각의 생성 기법 및 최신 Deep Learning (DL) 기술을 분석합니다. 특히, Generative Adversarial Networks (GAN)과 Variational Autoencoders (VAE)를 사용하여 음향과 시각의 결합된 디지털 조작 방법을 활용합니다.

- **Performance Highlights**: 음향 및 시각 모달리티를 활용한 기존 탐지 방법의 한계를 지적하고, 오디오 및 비주얼 deepfake 탐지에 대한 기존 연구의 격차를 해소하기 위한 연구 방향을 제시합니다. 또한, 이러한 방법들을 훈련시키기 위한 공개 데이터 세트에 대해서도 자세히 분석하였습니다.



### Maritime Search and Rescue Missions with Aerial Images: A Survey (https://arxiv.org/abs/2411.07649)
- **What's New**: 본 논문은 해상에서 인명을 탐지하기 위한 UAV(무인 항공기)를 사용하는 검색 및 구조(SAR) 작업에 대한 최초의 포괄적 리뷰를 제공합니다. 기존의 문헌들에 비해 현재의 방법들, 사용된 데이터 세트, 합성 데이터 생성 도구 및 기술을 상세히 검토합니다.

- **Technical Details**: 연구는 이미지 분류, 이미지 세분화, 객체 탐지 및 추적과 같은 다양한 기술 범주로 나뉩니다. 합성 데이터를 사용하는 것이 중요한데, 이는 다양한 훈련 데이터 세트를 생성하는 데 도움을 줍니다. UAV는 강력한 비전 시스템을 통합하여 실시간으로 신속하고 정확하게 사람을 탐지할 수 있습니다. 이 시스템은 AI(인공지능)를 통해 높은 처리 속도와 정밀도를 유지합니다.

- **Performance Highlights**: 본 논문의 주요 기여로는 해외 인명 구조 작업을 위한 현존하는 방법론, 실제 및 합성 데이터 세트에 대한 리뷰, 결과를 비교하여 주요 벤치마크의 성과를 분석합니다. 또한, 해양 SAR의 현재 연구 환경을 논의하며, 향후 방향성과 개방된 도전 과제를 제시합니다.



### xCG: Explainable Cell Graphs for Survival Prediction in Non-Small Cell Lung Cancer (https://arxiv.org/abs/2411.07643)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 11 pages

- **What's New**: 이번 논문에서는 그래프 신경망(Graph Neural Networks)을 활용하여 폐 선암 환자의 생존 예측을 위한 설명 가능한 세포 그래프(xCG) 접근법을 소개합니다. 이는 정밀 의학 데이터 기반의 결정에 기여할 수 있습니다.

- **Technical Details**: 저자들은 다중 조직 샘플 및 그래프를 처리할 수 있는 GNN 프레임워크를 제안하며, 세포 수준의 다양한 특징 영역(예: marker 표현과 임상 메타데이터)을 통합합니다. 이를 통해 생존 회귀(survival regression)와 분류(classification)가 가능합니다. 또한 grid 기반의 layer-wise importance propagation (LRP) 방법을 도입하였습니다.

- **Performance Highlights**: 제안된 xCG 방법은 416명의 폐 선암 환자에 대한 이미징 질량 세포 분석(IMC) 데이터로 검증되었으며, 암 병기와 모델 앙상블을 결합함으로써 리스크 추정의 품질을 개선하는 데 중요한 요소로 작용했습니다.



### Breaking the Low-Rank Dilemma of Linear Attention (https://arxiv.org/abs/2411.07635)
- **What's New**: 본 논문에서는 Softmax attention 메커니즘의 한계를 해결하고, linear attention의 성능 저하 문제를 극복하기 위해 Rank-Augmented Linear Attention (RALA)를 제안합니다. RALA는 성능을 유지하면서도 복잡성을 선형적으로 줄이는 것을 목표로 합니다.

- **Technical Details**: RALA는 두 가지 측면에서 rank 분석을 수행하여 저차원 문제를 해결합니다: KV buffer와 출력 특성에서의 분석을 통해, 정보의 다양성을 증가시키고, 정보의 구조를 재구성하여 출력을 최적화합니다. 이 방식을 통해 RALA는 복잡한 공간적 특징을 모델링할 수 있게 됩니다.

- **Performance Highlights**: RALA를 기반으로 구축된 Rank-Augmented Vision Linear Transformer (RAVLT)는 ImageNet-1k 데이터셋에서 26M 파라미터와 4.6G FLOPs로 84.4% Top-1 정확도를 달성하였습니다. 이는 기존의 linear attention 메커니즘을 초월하며, 다양한 비전 작업에서 뛰어난 성능을 보여줍니다.



### Leveraging Previous Steps: A Training-free Fast Solver for Flow Diffusion (https://arxiv.org/abs/2411.07627)
- **What's New**: 본 연구에서는 기존의 ordinary differential equation (ODE) solver인 Euler solver의 느린 생성 속도를 해결하기 위해 novel training-free flow-solver를 제안합니다. 이 flow-solver는 과거의 스텝을 활용하여 많은 function evaluations (NFE)을 줄이며, 고품질 생성 성능을 유지하는 것이 특징입니다.

- **Technical Details**: flow-solver는 Taylor expansion을 사용하여 ODE를 근사하고, 이전 스텝의 결과를 통해 고차 도함수의 근사를 수행합니다. 이를 통해 이전 스텝과 현재 스텝 간의 결과 간격을 계산하여 polynomial interpolation을 이용해 부정확성을 해결합니다. 따라서, 이전 스텝을 활용해 연속적 적분을 보다 정확하게 근사할 수 있습니다.

- **Performance Highlights**: CIFAR-10, CelebA-HQ, LSUN-Bedroom, LSUN-Church, ImageNet와 같은 다양한 데이터셋에서 flow-solver는 기존 solver들에 비해 FID-30K 지표를 13.79에서 6.75로, CIFAR-10에서는 46.64에서 19.49로 개선하는 성과를 보였습니다.



### Unraveling the Connections between Flow Matching and Diffusion Probabilistic Models in Training-free Conditional Generation (https://arxiv.org/abs/2411.07625)
- **What's New**: 이 연구에서는 훈련이 필요 없는 조건부 생성(Training-free conditional generation) 방법으로서, 흐름 일치(Flow Matching)와 확산 확률 모델(Diffusion Probabilistic Models) 간의 연결성을 도출하여 새로운 훈련 필요 없음 조건부 생성 방법인 FMPS(Flow-Match based Posterior Sampling)를 제안합니다.

- **Technical Details**: FMPS는 조건부 생성을 위해 DPMs의 점수 함수(Score Function)를 이용하여 훈련이 필요 없는 흐름 모델(Flow Models)에게 조건을 쉽게 통합할 수 있도록 합니다. 이 과정에서 FMPS는 ODE(Ordinary Differential Equation)를 DPM의 점수 함수에 맞춰 재정의합니다. 이를 통해 FMPS는 마진 분포에서 조건 분포로 변경할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: 실험 결과, FMPS는 선형 및 비선형 역문제(Linear/Non-linear Inverse Problems)와 텍스트-이미지 생성(Text-to-Image Generation)과 같은 다양한 조건부 생성 작업에서 기존 최첨단 방법들보다 더 높은 품질의 결과를 생성할 수 있음을 보여주었습니다.



### Mix from Failure: Confusion-Pairing Mixup for Long-Tailed Recognition (https://arxiv.org/abs/2411.07621)
- **What's New**: 이 논문에서는 Confusion-Pairing Mixup (CP-Mix)라는 새로운 접근 방식을 통해 장기(prevalent long-tailed) 이미지 인식을 위한 샘플 다양성을 개선하는 방법을 제안합니다. 기존의 방법들이 손실 함수 조정이나 분류기 학습의 분리 등을 통해 문제를 우회하는 것과는 달리, CP-Mix는 훈련 데이터셋을 실시간으로 증강하여 소수 클래스의 샘플 다양성을 높입니다.

- **Technical Details**: CP-Mix는 모델의 혼동 분포(confusion distribution)를 추정하여 혼동 쌍(confusion pairs)에서 샘플을 증강함으로써 데이터 부족(data deficiency) 문제를 해결합니다. 이 접근법은 각 클래스 쌍의 혼돈을 줄이고, 비대칭 데이터셋에서 기인하는 결정 경계의 편향을 처리하기 위해 새로운 mixup 공식을 사용합니다. 또한, CP-Mix는 기존의 Mixup 알고리즘보다 소수 클래스에 더 효과적입니다.

- **Performance Highlights**: 광범위한 실험을 통해 CP-Mix는 기존의 장기 이미지 인식 방법들을 초월하며, 분류기의 혼동을 성공적으로 완화함을 보여주었습니다. CP-Mix는 다양한 장기 데이터셋에서 기준 모델들에 비해 성능 향상을 달성합니다.



### Artificial Intelligence for Biomedical Video Generation (https://arxiv.org/abs/2411.07619)
- **What's New**: 이번 연구에서는 비디오 생성 기술의 최근 발전과 생물의학 분야에서의 응용 가능성을 다루고 있습니다. 특히, Sora와 같은 모델들이 비디오 합성 품질을 크게 향상시키는 중대한 발전을 이뤘습니다. 또한, 다양한 의료 개념 설명 및 질병 시뮬레이션에 대한 비디오 생성 기술의 잠재력을 강조하고 있습니다.

- **Technical Details**: 비디오 생성의 기존 주요 기술로는 Generative Adversarial Networks (GANs), Auto-Regressive (AR) 모델, 및 디퓨전 모델이 있습니다. 특히, AR + Diffusion 모델이 AR 모델과 디퓨전 모델의 장점을 결합한 차세대 비디오 생성 모델로 주목받고 있습니다. 이 논문은 생물의학 영상 데이터를 정리하고, 비디오 생성의 신규 아키텍처 및 교육, 평가 프로세스를 소개합니다.

- **Performance Highlights**: 비디오 생성 모델들은 현실적인 의료 시나리오를 시뮬레이션하는 데 큰 능력을 보여주고 있으나, 더욱 정교한 시뮬레이션을 위해 물리 원칙에 대한 이해, 평가 기준의 설정, 생성물의 제어 가능성과 설명 가능성을 고려해야 한다고 언급합니다. 기존 모델들이 생물의학적으로 유용한 비디오 생성을 지원할 수 있는 가능성을 제시합니다.



### Quantum Information-Empowered Graph Neural Network for Hyperspectral Change Detection (https://arxiv.org/abs/2411.07608)
Comments:
          This work has been accepted by IEEE Transactions on Geoscience and Remote Sensing (TGRS)

- **What's New**: 이번 연구에서는 변화 탐지(change detection, CD) 분야에 처음으로 양자 깊은 네트워크(quantum deep network, QUEEN)를 도입하였습니다. QUEEN은 기존의 그래프 신경망(graph neural networks, GNN) 및 합성곱 신경망(convolutional neural networks, CNN)과는 다르게 단위량자 정보를 활용하여 효과적인 고차원 특성을 추출합니다.

- **Technical Details**: 연구는 그래프 특징 학습(graph feature learning, GFL) 모듈과 양자 특징 학습(quantum feature learning, QFL) 모듈을 계층적으로 설계하여, GFL은 슈퍼픽셀(superpixel) 수준에서 bitemporal HSI의 그래프 구조를 활용하고, QFL은 픽셀 수준에서 양자 특징을 학습합니다. 마지막으로, 양자 분류기(quantum classifier)를 전통적인 완전 연결 분류기와 함께 사용하여 최종 분류 단계에서 최상의 성능을 발휘하도록 설계되었습니다.

- **Performance Highlights**: 제안된 QUEEN 기반 GNN(QUEEN-G)의 우수한 변화 탐지 성능이 실제 하이퍼스펙트럴 데이터셋(hyperspectral datasets)에서 실험적으로 검증될 예정입니다.



### Grounded Video Caption Generation (https://arxiv.org/abs/2411.07584)
- **What's New**: 본 연구에서는 새로운 작업, 데이터셋 및 모델인 GROunded Video Caption Generation (GROC)을 제안합니다. 이 작업은 비디오 내 객체의 설명과 잘 어울리도록 자막(Summary) 생성을 위한 방법을 통합했습니다.

- **Technical Details**: 연구에서는 자동 주석(annotation) 방법을 도입하여 기존의 빠른 동영상 자막 모델과 LLM (Large Language Model)을 결합하여 프레임 수준의 자막을 시간적으로 일관된 자막으로 요약하는 새로운 방법을 개발했습니다. 이를 통해 HowTo100M 데이터셋에서 자동 주석이 달린 HowToGround 데이터셋을 생성했습니다. 새로운 모델 VideoGround는 spatio-temporal 정보 처리를 위한 spatio-temporal adapters와 일관된 라벨을 가진 bounding box를 생성합니다.

- **Performance Highlights**: VideoGround 모델은 새로운 grounded video caption generation 작업에서 최고 성능을 기록했습니다. 본 모델은 대규모 데이터셋에서 핵심 기술 기여에 대한 폭넓은 실험을 수행하였고 이러한 기여의 중요성을 강조했습니다.



### Semantic segmentation on multi-resolution optical and microwave data using deep learning (https://arxiv.org/abs/2411.07581)
- **What's New**: 이 논문에서는 고해상도 인도 원거리 감지 위성 이미지를 사용하여 객체를 자동으로 식별하고 픽셀 단위로 분류하는 수정된 U-Net 모델 및 VGG-UNet 모델을 구현했습니다. 특히, Cartosat 2S 데이터셋을 사용하여 건물 형태 및 선박을 탐지하는 데 95% 이상의 정확도를 달성했습니다.

- **Technical Details**: 딥 러닝(deep learning) 모델을 기반으로 하는 수정된 U-Net 및 VGG-UNet 아키텍처가 활용되었습니다. Cartosat 2S(약 1m 공간 해상도) 데이터를 사용하여 이미지를 다중 클래스(multi-class)로 분류하기 위한 훈련이 수행되었습니다. 또한, RISAT-1의 마이크로웨이브 데이터(microwave data)를 입력으로 사용하여 선박과 나무를 탐지했습니다.

- **Performance Highlights**: 총 여섯 가지 문제를 딥 러닝 모델을 사용하여 시도하였고, IoU(Intersection over Union) 정확도는 복잡도에 따라 85%에서 98% 범위에 도달했습니다. 특정 실험에서 95% 이상의 정확도로 건물 형태와 선박을 탐지하였으며, 다중 레이블(multi-label) 분류의 경우 95% 이상의 정확도로 결과를 생성했습니다.



### Projecting Gaussian Ellipsoids While Avoiding Affine Projection Approximation (https://arxiv.org/abs/2411.07579)
- **What's New**: 이 논문에서는 3D Gaussian Splatting의 렌더링 품질 저하 문제를 해결하기 위해 엘립소이드 기반의 프로젝션 방법을 도입하였습니다. 이 방법은 Gaussian ellipsoid의 이미지 평면 투영을 개선하여 흐림 현상(blur)과 아티팩트(artifact)를 줄입니다.

- **Technical Details**: 제안된 엘립소이드 기반 프로젝션 방법은 David Eberly의 작업을 기반으로 하여 Gaussian ellipsoid의 방정식을 유도하고, 카메라의 원점에서 평면에 투영하는 과정을 통해 이루어집니다. 이 과정에서 카메라의 원점이 ellipsoid 내부에 있는 경우와 카메라 공간의 z=0 평면 아래에 위치한 경우를 고려한 전처리 필터링 전략도 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 3D Gaussian Splatting에 비해 렌더링 품질과 속도를 모두 향상시키는 것으로 나타났습니다. 특히, 다양한 벤치마크 데이터셋에서 확인된 결과는 기존 방법 보다 우수함을 입증하였습니다.



### Atmospheric turbulence restoration by diffeomorphic image registration and blind deconvolution (https://arxiv.org/abs/2411.07578)
- **What's New**: 이 논문에서는 대기 난류에 의해 변화된 이미지를 개선하기 위한 새로운 접근 방식을 제시합니다. 두 가지 새로운 알고리즘이 제안되며, 이는 블라인드 디콘볼루션(Blind Deconvolution) 블록, 엘라스틱 레지스트레이션(Elastic Registration) 블록 및 시간 필터 블록의 조합으로 구성됩니다.

- **Technical Details**: 우리는 이미지 왜곡 복원 분야의 연구를 다룬다. 제안된 기술은 이미지 왜곡을 모델링하기 위해 이미지 왜곡과 블러링을 결합합니다. 이후 블라인드 디콘볼루션과 이미지 레지스트레이션을 통해 이 과정을 역행하려고 시도합니다. 이 논문에서는 Frakes 외의 프레임워크를 사용하여 대기 난류 모델링을 제안하며, Chan 외의 블라인드 디콘볼루션 알고리즘 및 Younès와 Beg의 디피오모르피즘 접근법에 기초한 레지스트레이션을 사용합니다.

- **Performance Highlights**: NATO RTG40 그룹이 뉴 멕시코 사막에서 수집한 실제 이미지에 대해 알고리즘을 테스트한 결과, 제안된 방법이 대기 난류로 인해 왜곡된 이미지를 효과적으로 복원할 수 있다는 것을 보여주었습니다. 이 연구는 자동 목표 인식 알고리즘에 필요한 높은 성능 복원률을 달성하는 데 중점을 두고 있습니다.



### IR image databases generation under target intrinsic thermal variability constraints (https://arxiv.org/abs/2411.07577)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2411.06695

- **What's New**: 이 논문은 ATR(Automatic Target Recognition) 평가를 위한 적외선 이미지 데이터베이스 생성 문제를 다루고 있습니다. 저자는 이미지 품질 지표 제약하에 배경 위에 목표물과 장애물을 중첩시켜 사실적인 이미지를 생성하는 방법을 제안합니다.

- **Technical Details**: 제안하는 방법은 다양한 열적 변동성을 가지는 목표물 서명을 생성하기 위해 3D 모델에 실제 적외선 텍스처를 입히는 기술을 포함합니다. 이전 연구에서 제안된 하이브리드 장면 생성 원칙에 기반하여, 목표물 서명을 실제 배경에 중첩하는 방식으로 이미지 품질을 조절합니다. 사용되는 지표는 지역 대비(contrast), 감지 가능성(detectability), 신호 대비 잡음 비율(signal to clutter ratio) 등입니다.

- **Performance Highlights**: 제안된 방법은 목표물의 작동 상태에 따른 열적 변동성을 고려하여 다양한 환경에서 효과적으로 이미지를 생성할 수 있으며, 실제 차량의 극한 상태 이미지와 결합하여 높은 현실감을 제공하는 것으로 나타났습니다.



### G\'en\'eration de bases de donn\'ees images IR sous contraintes avec variabilit\'e thermique intrins\`eque des cibles (https://arxiv.org/abs/2411.07575)
Comments:
          in French language, GRETSI Symposium on Signal and Image Processing, Dijon, France, September 2009

- **What's New**: 본 논문에서는 차량 서명을 배경에 겹쳐서 적외선 영상에서 목표물을 시뮬레이션하는 방법을 제안합니다. 이 방법은 다양한 열적 구성의 목표 서명을 생성할 수 있어 ATR(Automatic Target Recognition) 알고리즘의 성능 평가를 위한 대규모 데이터셋 생성을 용이하게 합니다.

- **Technical Details**: 소스 이미지와 상황에 맞는 변수를 조합하여 3D 모델 위에 차량 서명을 매핑하는 방식으로 이미지를 생성합니다. 이 과정에서 지역적 대비(RSS), 감지 가능성(QD), 신호 대 잡음 비율(RSC) 등을 이용하여 이미지 품질을 조절합니다. 각 차량의 실제 이미지에서 열적 변동성을 반영한 서명을 실시간으로 생성하여 알고리즘의 학습 성능을 높입니다.

- **Performance Highlights**: 제안된 방법은 검증된 알고리즘에 비해 더 다양한 시나리오에서의 인식 성능을 향상시킵니다. 실제 차량 데이터를 기반으로 하여 만들어진 서명 덕분에, 알고리즘이 다양한 열 상태에서도 목표물 인식에 효과적임을 입증했습니다.



### Multi-task Feature Enhancement Network for No-Reference Image Quality Assessmen (https://arxiv.org/abs/2411.07556)
- **What's New**: 새로운 multi-task 기반의 No-Reference Image Quality Assessment (NR-IQA) 프레임워크를 소개하며, 고주파수 정보 추출, 품질 추정 및 왜곡 인식 네트워크로 구성되어 있습니다.

- **Technical Details**: 이 프레임워크는 이미지의 texture details와 왜곡 종류를 구분하기 위해 고주파수 추출 네트워크와 왜곡 인식 네트워크를 설계하고, Attention mechanism에 기반한 feature fusion 모듈을 통해 여러 작업의 특성을 효과적으로 통합합니다.

- **Performance Highlights**: 다섯 개의 표준 IQA 데이터베이스에서 우리의 방법이 높은 성능을 달성했고, 견고한 일반화 능력을 보여주었음을 실험 결과로 확인하였습니다.



### GaussianCut: Interactive segmentation via graph cut for 3D Gaussian Splatting (https://arxiv.org/abs/2411.07555)
- **What's New**: GaussianCut이라는 새로운 메소드를 소개합니다. 이 방법은 3D Gaussian으로 표현된 장면의 상호작용적 다중 보기(segmentation)를 가능하게 합니다. 사용자는 점 클릭, 대강의 낙서 또는 텍스트와 같은 직관적인 입력으로 분할할 객체를 선택할 수 있습니다.

- **Technical Details**: 본 방법은 3D Gaussian Splatting(3DGS)이라는 장면 표현을 사용하여 관심 객체의 추출을 간소화합니다. 장면은 그래프로 표현되고, graph-cut 알고리즘을 사용하여 에너지 함수를 최소화함으로써 Gaussian을 효과적으로 전경과 배경으로 분리합니다. 초기 거친 분할을 얻기 위해 2D 이미지/비디오 분할 모델을 활용하고, 이 후 그래프 구조를 통해 이를 정교하게 다듬습니다.

- **Performance Highlights**: GaussianCut은 다양한 장면 세트에서 적응성을 보여주며, 추가적인 분할 인지 훈련 없이도 3D 분할에 대해 최첨단 접근법과 경쟁력 있는 성능을 달성합니다.



### Contrastive Language Prompting to Ease False Positives in Medical Anomaly Detection (https://arxiv.org/abs/2411.07546)
Comments:
          4 pages, 3 figures, 2 tables

- **What's New**: 이 논문은 의료 영상에서의 이상 탐지를 위해 CLIP의 새로운 응용 방법인 Contrastive Language Prompting (CLAP)을 제안합니다. CLAP은 긍정적 및 부정적 텍스트 프롬프트를 활용하여 비정상 영역을 보다 정확하게 식별하고, 일반적으로 나타나는 오탐지를 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: CLAP 방법은 CLIP의 시각적 주의(attention) 메커니즘을 통해 비정상 영역에 대한 시각적 주의를 유도하고, 부정적 프롬프트를 사용해 정상 영역에 대한 주의를 약화시킵니다. 이를 통해 BMAD 데이터셋을 활용한 실험에서 이상 탐지 성능이 향상됨을 보여줍니다. U-Net을 이용한 무감독 이상 탐지(UAD) 방법도 제안되어, 정상 샘플을 기반으로 한 재구성을 통해 비정상 패턴을 처리합니다.

- **Performance Highlights**: BMAD 데이터셋을 사용한 실험을 통해 CLAP 방법이 비정상 영역에 대한 강한 주의 문제를 극복하고, 기존 방법들에 비해 UAD 성능을 향상시킴을 입증했습니다. 특히, 정상 샘플로 훈련된 U-Net이 비정상 패턴 재구성에서 어려움을 겪는다는 점을 이용하여, 더욱 정밀한 이상 진단을 가능하게 했습니다.



### Depthwise Separable Convolutions with Deep Residual Convolutions (https://arxiv.org/abs/2411.07544)
Comments:
          Course Project Report

- **What's New**: 본 연구에서는 Xception 아키텍처를 최적화하여 엣지 디바이스에 적합한 경량화된 모델을 제안합니다. 기존의 Xception 아키텍처는 높은 성능을 자랑하지만, 계산 비용이 크게 드는 단점이 있습니다. 이를 해결하기 위해 깊이 분리 가능 합성곱(depthwise separable convolutions)과 딥 잔차 연결(deep residual connections)을 결합하여 효율적인 모델을 개발했습니다.

- **Technical Details**: 본 논문에서 제안하는 최적화된 Xception 아키텍처는 깊이 분리 가능한 합성곱 층으로 구성되어 있으며, 일부 층에 딥 잔차를 추가하여 새로운 합성곱 신경망 층을 두 개 더 포함하고 있습니다. 이는 파라미터 수와 메모리 사용량, 계산 부담을 크게 줄입니다.

- **Performance Highlights**: CIFAR-10 객체 검출 데이터셋에서 평가를 수행한 결과, 제안된 아키텍처는 파라미터 크기가 작으며 훈련 시간도 짧으면서 기존의 Xception 아키텍처보다 더 우수한 성능을 보였습니다.



### HiCoM: Hierarchical Coherent Motion for Streamable Dynamic Scene with 3D Gaussian Splatting (https://arxiv.org/abs/2411.07541)
Comments:
          Accepted to NeurIPS 2024; Code is avaliable at this https URL

- **What's New**: 이번 연구에서는 HiCoM(금관계 일관성 모션) 프레임워크를 통해 다중 뷰 비디오 스트림에서 동적 장면의 온라인 재구성을 개선하는 방법을 제안합니다. HiCoM은 초기 3D Gaussian Splatting(3DGS) 표현을 학습하기 위한 perturbation smoothing 전략을 기반으로 하며, 효율적인 모션 캡처를 위한 계층적 일관성 모션 메커니즘을 포함하고 있습니다.

- **Technical Details**: HiCoM 프레임워크는 초기 3DGS 표현을 견고하게 만드는 데 필요한 과정으로 perturbation smoothing 전략을 사용합니다. 그 후, 3D Gaussians의 비균일 분포와 지역적 일관성을 활용하여 계층적 일관성 모션 메커니즘을 도입합니다. 이 방식은 장면을 여러 영역으로 나누어서 비어있지 않은 영역에만 구체적인 모션을 모델링하며, 시간에 따른 변화를 빠르고 정확하게 학습할 수 있게 합니다.

- **Performance Highlights**: 실험 결과, HiCoM 프레임워크는 기존의 최첨단 방법들과 비교하여 학습 효율성을 약 20% 개선하고 데이터 저장량을 85% 감소시켰습니다. 또한 병렬 학습 전략을 통해 여러 프레임을 동시에 학습하면서 훈련 시간을 <2 초로 줄이는 데 성공하였으며, 실세계 애플리케이션에서의 응답성을 크게 향상시켰습니다.



### SparrowVQE: Visual Question Explanation for Course Content Understanding (https://arxiv.org/abs/2411.07516)
- **What's New**: 이 논문에서는 Visual Question Answering (VQA) 시스템을 발전시키기 위한 Visual Question Explanation (VQE) 방법을 소개합니다. 기존 VQA의 단순한 대답에서 벗어나, 복잡한 시각적 내용과의 상호작용을 통해 상세한 설명을 제공하게 됩니다.

- **Technical Details**: MLVQE 데이터셋을 구성하였으며, 3억 파라미터를 가진 SparrowVQE라는 새로운 멀티모달 모델을 도입하였습니다. 모델 학습은 세 단계로 이루어지며, 멀티모달 사전 학습, 지침 튜닝, 도메인 세부 튜닝이 포함됩니다. 이 과정에서 SigLIP 모델을 활용하여 시각 정보와 문서(transcript)를 연결합니다.

- **Performance Highlights**: SparrowVQE 모델은 MLVQE 데이터셋에서 높은 성능을 보여주었으며, 다른 다섯 개의 VQA 벤치마크 데이터셋에서도 최첨단 성능을 초과하였습니다. 이는 교육 도메인에서의 VQA 효과성을 향상시키는 데 기여할 것으로 기대됩니다.



### GUS-IR: Gaussian Splatting with Unified Shading for Inverse Rendering (https://arxiv.org/abs/2411.07478)
Comments:
          15 pages, 11 figures

- **What's New**: 본 논문은 GUS-IR이라는 새로운 프레임워크를 통해 복잡한 장면에서의 inverse rendering 문제를 해결하기 위해 forward shading과 deferred shading 기술을 비교 분석하고, 두 가지 장점을 통합하는 통합 shading 솔루션을 제안합니다.

- **Technical Details**: GUS-IR은 3D Gaussian Splatting(3DGS) 기술을 활용하여 각 파티클의 가장 짧은 축을 normal로 사용하고, 깊이 관련 정규화를 통해 기하학적 표현을 개선합니다. forward shading과 deferred shading의 이점을 결합하여 복잡한 재료의 분해를 용이하게 하고, GS-IR에서 제공한 probe-based baking 스킴을 개선하여 간접 조명을 다룹니다.

- **Performance Highlights**: GUS-IR은 TensoIR Synthesis, Shiny Blender, Glossy Blender, Mip-NeRF 360, Ref-NeRF Real 데이터셋을 포함한 다양한 도전적인 장면에서 정밀한 intrinsic decomposition 및 기하학적 표현을 달성하여 기존 방법들보다 우수한 성능을 입증하였습니다.



### Semi-Truths: A Large-Scale Dataset of AI-Augmented Images for Evaluating Robustness of AI-Generated Image detectors (https://arxiv.org/abs/2411.07472)
Comments:
          Accepted at NeurIPS 2024 Track Datasets & Benchmarks Track

- **What's New**: 이 논문에서는 SEMI-TRUTHS라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 실제 이미지, AI-증강 이미지 및 다양한 왜곡(Perturbation) 기법을 포함하여 AI 생성 이미지 탐지기의 정확성과 신뢰성을 평가할 수 있는 보다 사실적인 리스트를 제공합니다.

- **Technical Details**: SEMI-TRUTHS에는 27,600개의 실제 이미지, 223,400개의 마스크, 1,472,700개의 AI-증강된 이미지가 포함되어 있습니다. 각 증강 이미지는 탐지기의 강건함을 표준화 및 목표 평가할 수 있도록 메타데이터와 함께 제공됩니다.

- **Performance Highlights**: 최신 탐지기들은 왜곡 종류와 정도, 데이터 분포, 증강 방법에 따라 다양한 민감도를 보이며, 이 논문은 이러한 특성의 새로운 통찰을 제공합니다.



### MSEG-VCUQ: Multimodal SEGmentation with Enhanced Vision Foundation Models, Convolutional Neural Networks, and Uncertainty Quantification for High-Speed Video Phase Detection Data (https://arxiv.org/abs/2411.07463)
Comments:
          Under Review in EAAI

- **What's New**: MSEG-VCUQ는 VideoSAM을 통해 고속 비디오 단계 감지(HSV PD) 분할을 위한 혁신적인 하이브리드 프레임워크를 제시합니다. 이 모델은 CNN과 transformer 기반 비전 모델을 결합하여 다중 모드 데이터에서의 분할 정확성을 향상시킵니다.

- **Technical Details**: VideoSAM은 U-Net(CNN)과 Segment Anything Model(SAM)을 통합하여 다양한 HSV PD 양식에서 고급 특징 추출 및 분할을 수행합니다. 이 프레임워크는 치수 기반 불확실성 정량화(UQ)도 포함하여 실험적 조건에 따른 신뢰할 수 있는 메트릭을 제공합니다.

- **Performance Highlights**: VideoSAM은 복잡한 위상 경계, 겹치는 기포 및 동적 액체-증기 상호작용이 있는 환경에서도 SAM 및 특정 모드 CNN 모델보다 뛰어난 분할 정확도를 보여줍니다. 이 모델은 다양한 데이터 세트에 효과적으로 적응하여 신뢰성이 높은 결과를 제공합니다.



### MureObjectStitch: Multi-reference Image Composition (https://arxiv.org/abs/2411.07462)
- **What's New**: 이번 연구에서는 generative image composition에서 foreground 객체와 background 이미지의 조화를 이루는 새로운 finetuning 전략을 제안합니다. 주목할 점은 하나 이상의 reference 이미지로 모델을 효과적으로 fine-tuning할 수 있는 multi-reference 전략을 도입한 것입니다.

- **Technical Details**: 제안된 방법은 기존 모델(ObjectStitch, ControlCom)을 기반으로 하는 multi-reference finetuning 전략으로, N개의 이미지에서 bounding box를 استخراج하고 foreground 객체와 조화롭게 결합된 composite 이미지를 생성하는 과정을 포함합니다. 이 구조는 cross-attention을 통해 다양한 reference 이미지의 feature를 활용하는 방식으로, detail preservation과 pose/viewpoint 조정의 균형을 잡았습니다.

- **Performance Highlights**: MureObjectStitch는 background와 foreground의 조화를 잘 이뤄내며, 다양한 anecdotal 예시에서 충분히 만족스러운 결과를 보여 줍니다. 실험 결과, prepetrained ObjectStitch와 비교했을 때 detail preservation이 크게 향상된 점이 눈에 띄며, 150 epochs의 finetuning을 통해 높은 품질의 composite 이미지를 생성할 수 있음을 확인했습니다.



### BLIP3-KALE: Knowledge Augmented Large-Scale Dense Captions (https://arxiv.org/abs/2411.07461)
- **What's New**: KALE 데이터셋은 2억 1천 8백만 개의 이미지-텍스트 쌍을 포함하며, 사실적으로 기반한 이미지 캡션 생성을 가능하게 하는 새로운 접근 방식을 제시합니다. KALE는 기존 CapsFusion보다 규모와 밀도가 뛰어나며, 강화된 지식 기반 캡션 생성을 통해 여러 멀티모달 모델의 훈련 성능을 개선합니다.

- **Technical Details**: KALE는 2단계 접근 방식을 사용합니다. 첫 번째 단계에서는 CogVLM-17B를 활용하여 1억 개의 밀집 캡션을 생성하고, Mistral 모델을 통해 이 캡션에 사실 정보를 추가합니다. 두 번째 단계에서는 특화된 VLM을 훈련하여 KALE 데이터셋을 확대하고, 최종적으로 2억 1천 8백만 개의 이미지-텍스트 쌍을 생성합니다.

- **Performance Highlights**: KALE로 훈련된 VLM은 TextVQA에서 59.92%, VQAv2에서 70.10%, ScienceQA에서 72.68%의 성능을 달성하며, 그 평균 성능은 51.96%에 이릅니다. 이는 기존의 데이터셋인 Datacomp-1B 및 CapsFusion보다 뛰어난 성능을 보여줍니다.



### Tracing the Roots: Leveraging Temporal Dynamics in Diffusion Trajectories for Origin Attribution (https://arxiv.org/abs/2411.07449)
- **What's New**: 이번 논문에서는 이미지 생성에서의 Diffusion 모델을 이용한 새로운 알고리즘을 제안하여, 특정 이미지가 모델의 훈련 데이터셋의 일부분인지, 모델에 의해 생성된 것인지, 또는 외부 데이터셋에서 온 것인지를 분류하는 방법을 연구합니다. 이를 통해 기존의 두 가지 이진 분류 문제를 통합하여 새로운 세 가지 분류 문제인 Origin Attribution (OA)을 제안합니다.

- **Technical Details**: Diffusion 모델은 데이터 분포와 정규 분포 간의 매핑을 정의하며, 이 과정에서 노이즈에서 시작하여 단계별로 이미지를 생성하는 반복적 알고리즘을 사용합니다. 연구에서는 고차 gradient feature를 활용하여 이미지 분류 성능을 개선하는 방법을 제시하며, Membership Inference Attack (MIA) 설정 하에서 기존 공격 방법과 비교하여 더 강력한 알고리즘을 설계했습니다.

- **Performance Highlights**: 실험을 통해, 높은 차수의 gradient feature를 사용하면 MIA, Model Attribution (MA), 그리고 OA 공격에서 성능 향상이 크게 이루어졌음을 입증했습니다. 특히, 우리 접근 방식은 기존 임계 기반 MIA 프로그램의 한계를 극복하고, 다양한 데이터 분포 변화에 보다 견고하게 대처할 수 있는 능력을 보여주었습니다.



### All-in-one Weather-degraded Image Restoration via Adaptive Degradation-aware Self-prompting Mod (https://arxiv.org/abs/2411.07445)
- **What's New**: 이 논문에서는 날씨에 의해 악화된 이미지를 복원하기 위한 새로운 접근 방식을 제안합니다. 본 연구의 핵심은 적응형 손상 인식 자기 프롬프트 모델(ADSM)을 개발하여 여러 날씨 조건에 효과적으로 적응할 수 있도록 하는 것입니다.

- **Technical Details**: ADSM은 CLIP(contrastive language-image pre-training model)을 활용하여 손상 유형, 손상 속성 및 이미지 캡션을 특성화하는 세 가지 타입의 잠재 프롬프트(latent prompt)를 생성하는 잠재 프롬프트 생성기(LPG)를 개발합니다. 또한, 획득한 손상 인식 프롬프트를 확산 모델(difussion model)의 시간 임베딩(time embedding)에 통합하여 손상 인식을 개선합니다. WNE-Net(wavelet-oriented noise estimating network)을 도입하여 역 샘플링(reverse sampling) 절차를 가속하고 주파수 인식의 한계를 해결합니다.

- **Performance Highlights**: 여덟 개의 공개 데이터셋을 대상으로 한 실험에서 ADSM은 특정 작업 및 올-인-원(all-in-one) 응용 프로그램 모두에서 효과적인 성능을 보여줍니다. ADSM은 복원된 이미지의 자연스러움을 향상시키면서 감독 기준 점수(supervised metric scores)를 개선했습니다.



### XPoint: A Self-Supervised Visual-State-Space based Architecture for Multispectral Image Registration (https://arxiv.org/abs/2411.07430)
Comments:
          13 pages, 11 figures, 1 table, Journal

- **What's New**: 이 논문에서는 XPoint라는 새로운 self-supervised 방법을 제안하여 서로 다른 스펙트럼 모달리티 간의 이미지 매칭에서 적응 훈련과 미세 조정이 가능하도록 하고 있습니다. 이는 라벨링된 데이터의 부족을 해결하려는 노력에서 중요한 진전을 보여줍니다.

- **Technical Details**: XPoint는 모듈화된 아키텍처를 기반으로 하며, aligned multispectral datasets로부터 pseudo-ground truth keypoints를 생성하는 base detector와 VMamba encoder, 특화된 세 가지 joint decoder heads로 구성되어 있습니다. 이 구조는 다양한 모달리티에 신속하게 적응할 수 있도록 설계되었습니다.

- **Performance Highlights**: XPoint는 Optical-Thermal 데이터를 기반으로 훈련되고, VIS-NIR, VIS-IR, VIS-LWIR, VIS-SAR 등 다양한 설정에서 미세 조정되어, 다섯 개의 서로 다른 multispectral 데이터셋에서 기존의 최첨단 방법들을 꾸준히 초월하거나 일치하는 성능을 보였습니다.



### Feature-Space Semantic Invariance: Enhanced OOD Detection for Open-Set Domain Generalization (https://arxiv.org/abs/2411.07392)
Comments:
          IEEE BigData 2024, Ph.D. Forum

- **What's New**: 이번 논문은 Open-set domain generalization의 복합적인 문제를 해결하기 위해 Feature-space Semantic Invariance (FSI)라는 새로운 프레임워크를 제안합니다. FSI는 다양한 도메인에서의 의미 일관성을 유지하여 unseen 도메인에서 OOD 인스턴스를 보다 정확하게 탐지할 수 있도록 합니다.

- **Technical Details**: FSI는 특성 공간에서의 의미 일관성을 강제하여 high-quality domain-invariant features를 학습하는 데 도움을 줍니다. 또한, ID 샘플을 기반으로 생성된 synthetic OOD 데이터를 활용함으로써 ID와 OOD 인스턴스 간의 결정 경계를 더욱 명확하게 설정하고 모델의 강건성을 향상시킵니다.

- **Performance Highlights**: 초기 실험 결과, ColoredMNIST 데이터셋에서 AUROC가 9.1%에서 18.9%까지 개선되었으며, ID 분류 정확도 또한 유의미하게 증가했습니다.



### Generalization of Brady-Yong Algorithm for Fast Hough Transform to Arbitrary Image Siz (https://arxiv.org/abs/2411.07351)
Comments:
          6 pages, 2 figures. Accepted to Symposium on Pattern Recognition and Applications 2024 (SPRA 2024)

- **What's New**: 이번 논문은 Hough 변환(Hough Transform, HT)을 계산하기 위한 새로운 알고리즘을 제안하여, 기존의 Brady-Yong 알고리즘을 일반화하여 임의 크기의 이미지에서도 사용할 수 있도록 하였습니다. 이로 인해 계산의 정확성이 크게 향상되었으며 이론적 분석과 실험 결과가 일치함을 보여주었습니다.

- **Technical Details**: 제안된 알고리즘은 Brady-Yong 알고리즘과 동일한 최적 계산 복잡도(Θ(n² log₂n))를 유지하며, 더욱 높은 정확도로 Hough 변환을 계산할 수 있도록 설계되었습니다. 알고리즘은 이미지의 임의 크기를 지원하며, 이로 인해 다양한 응용 프로그램에서 실시간으로 사용할 수 있는 가능성을 높였습니다.

- **Performance Highlights**: 실험 결과, 제안된 FHT2DT 알고리즘은 기존의 FHT2DS 알고리즘보다 현저히 높은 정확도를 기록하였으며, 이론 분석 결과와 실험 결과가 일치함을 확인하였습니다. 이는 Hough 변환의 계산 성능을 획기적으로 개선할 수 있는 기반이 될 것입니다.



### $SE(3)$ Equivariant Ray Embeddings for Implicit Multi-View Depth Estimation (https://arxiv.org/abs/2411.07326)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 본 논문에서는 3D 학습을 위한 효과적인 방안으로서, multi-view 환경에서 강력한 특성을 생성하는 데 필수적인 equivariance(동일성)를 통합한 depth 추정 기술을 탐구합니다. 특히 S⁢E⁢(3)𝑆𝐸3SE(3) 틀의 eqivariant 특성을 Perceiver IO 아키텍처에 통합하는 방법을 제안합니다.

- **Technical Details**: Spherical Harmonics를 사용하여 ray embedding의 positional encoding을 구현하고, Perceiver IO의 encoder 및 decoder를 통해 specialized equivariant architecture를 개발합니다. 이 모델은 stereo depth estimation의 범위에서 검증되었으며, 전통적인 기하학적 제약이나 데이터 증강을 사용하지 않고도 state-of-the-art 성능을 달성합니다.

- **Performance Highlights**: 논문에서 제안한 모델은 stereo depth estimation 작업에서 기존의 비-equivariant 모델을 크게 초월하며, 새로운 scene representation 스킴을 통해 효과적인 3D 구조 학습을 가능하게 합니다. 이를 통해 모델은 다양항 상황에서도 3D 깊이 추정에서 탁월한 성능을 발휘합니다.



### GPU-Accelerated Inverse Lithography Towards High Quality Curvy Mask Generation (https://arxiv.org/abs/2411.07311)
Comments:
          10 pages, 5 figures, Accepted by International Symposium on Physical Design (ISPD), 2025, Austin TX

- **What's New**: 본 논문에서는 GPU 가속 ILT 알고리즘을 소개하며, 이는 커브형(mask shapes) 최적화 문제를 해결하고 프로세스 윈도우(process window)와 형상 정밀도를 개선합니다. 기존 알고리즘의 한계를 분석하고, 향상된 최적화를 달성하는 방법을 제안합니다.

- **Technical Details**: 본 연구는 기존 ILT 알고리즘의 한계에 대한 철저한 분석을 수행하였고, 코너를 매끄럽게 만드는 목표로 ILT 해결기가 최적화할 수 있도록 하는 커브형 디자인 리타게팅(curvilinear design retargeting) 아이디어를 개발했습니다. 또한 최종 결과의 품질(QoR) 저하 없이 마스크의 곡률(curvature)과 형상을 제어할 수 있는 미분 가능한 형태 연산자(differentiable morphological operators)를 도입합니다.

- **Performance Highlights**: 실험 결과, 본 알고리즘은 실제와 합성 디자인의 다양한 레이어(layer)에서 뛰어난 성능을 발휘하였고, 기존의 주요 ILT 엔진들에 비해 품질(QoR)과 프로세스 윈도우 개선에 있어 유의미한 이점을 입증했습니다.



### ViTOC: Vision Transformer and Object-aware Captioner (https://arxiv.org/abs/2411.07265)
- **What's New**: 이 논문은 Vision Transformer와 Object-aware Captioner의 조합인 ViTOC(Vision Transformer and Object-aware Captioner)를 소개하여 이미지 캡셔닝에서 생성된 설명의 정확성과 다양성을 개선하고자 합니다. ViTOC는 Vision Transformer와 객체 탐지기를 기반으로 한 이중 경로 아키텍처를 채택하여 전역 시각적 특징과 지역 객체 정보를 효과적으로 융합합니다.

- **Technical Details**: ViTOC는 여러 이미지 인코더를 사용하여 시각적 특징의 표현을 향상시키고, 객체 탐지기 출력을 프롬프트로 통합하는 혁신적인 방법을 설계하였습니다. 최종적으로 ViTOC는 YOLO-Tiny 모델을 객체 탐지기로 사용하며, COCO 데이터셋을 통해 성능을 평가합니다.

- **Performance Highlights**: ViTOC는 모든 평가 지표에서 기준 모델보다 뛰어난 성능을 보이며, CIDEr와 SPICE에서 각각 71.26과 17.82의 성과를 달성하였습니다. 또한, CLIP 기반의 참조 없는 평가 방법을 제안하여 모델의 효과성을 추가적으로 검증하였습니다.



### LLMPhy: Complex Physical Reasoning Using Large Language Models and World Models (https://arxiv.org/abs/2411.08027)
- **What's New**: 본 논문에서는 로봇이 실제 세계에서 작업할 때 필요한 물리적 추론(physical reasoning) 능력을 향상시키기 위한 새로운 과제와 데이터셋인 TraySim을 제안합니다. 이 방법은 대형 언어 모델(LLMs)과 물리 엔진을 결합하여, 복잡한 물체의 동역학을 예측하는 것을 목표로 합니다.

- **Technical Details**: TraySim은 여러 물체가 놓인 쟁반에서 외부 충격에 대한 동역학을 예측하는 과제로 구성되어 있습니다. LLMPhy는 LLM의 프로그램 합성 능력을 활용하여 물리적 하이퍼파라미터를 추정하는 제로샷(Zero-shot) 최적화 프레임워크입니다. 이 프레임워크는 비분화가능한 시뮬레이터와 상호작용하여 물리적 특성을 상상하는 데 사용됩니다.

- **Performance Highlights**: LLMPhy는 TraySim 데이터셋에서 실험을 통해 기존의 블랙 박스 최적화 방법들에 비해 우수한 성능을 입증하였으며, 물리적 파라미터 추정의 정확도 또한 크게 향상되었습니다. 또한 LLMPhy의 성능은 OpenAI o1-preview와 같은 최신 LLM에서 더욱 두드러진 최적화 수렴 경향을 보여주었습니다.



### DINO-LG: A Task-Specific DINO Model for Coronary Calcium Scoring (https://arxiv.org/abs/2411.07976)
Comments:
          Developed by Center for Applied Artificial Intelligence (CAAI), University of Kentucky

- **What's New**: 이 논문에서는 기존의 딥러닝 기반 CAC 스코어링 시스템을 개선하기 위해 self-supervised learning (SSL) 기법을 사용하는 DINO 모델을 도입해, 주석 데이터가 부족한 문제를 해결하고 CAC 세분화 및 스코어링 성능을 향상시키고자 합니다. DINO-LG 모델을 통해 CT 이미지에서 석회화된 영역을 자동으로 구분합니다.

- **Technical Details**: 이 연구에서는 DINO(무주석 자기 증류) 기법을 적용하여 CT 스캔에서 CAC 영역을 분리하는 새로운 self-supervised 학습 체계를 제안합니다. DINO 모델은 석회화된 영역에 주목하여 주석된 데이터를 거의 필요로 하지 않으며, Label-Guided DINO(dino-LG) 기법을 통해 보다 정교한 특정 특성을 캡처합니다. 기본 U-NET 아키텍처를 사용하여 CAC 세분화 작업을 수행합니다.

- **Performance Highlights**: DINO-LG 모델은 석회화된 CT 슬라이스를 구분하는 작업에서 기존 DINO 모델보다 57% 향상된 분류 성능을 보여주었으며, CAC 세분화 성능을 약 10% 향상시켰고, CAC 스코어링 정확성을 획기적으로 증가시켰습니다.



### Commissioning An All-Sky Infrared Camera Array for Detection Of Airborne Objects (https://arxiv.org/abs/2411.07956)
- **What's New**: 본 논문은 미확인 비행 현상(Unidentified Aerial Phenomena, UAP)에 대한 과학적 데이터가 부족함을 해결하기 위한 갈릴레오 프로젝트(Galileo Project)의 다중 모드 지상 관측소에 관한 내용을 담고 있습니다. 새로운 극초음파 카메라 배열을 이용해 하늘을 지속적으로 모니터링하며 데이터 수집과 분석을 위한 시스템 성능 검사 및 검증이 이루어질 예정입니다.

- **Technical Details**: 이 프로젝트는 8개의 비 냉각식 장파 적외선 FLIR Boson 640 카메라를 포함하는 전천후 적외선 카메라 배열을 사용하고, ADS-B 데이터에서 비행기 위치를 활용한 새로운 외부 보정 방법을 통해 기기 보정을 할 계획입니다. 관측 기간 동안 50만 개의 비행체 궤적을 재구성하였으며, 다양한 기상 조건, 범위 및 비행체 크기를 고려하여 탐지 효율을 평가합니다.

- **Performance Highlights**: 5개월의 필드 운영을 통해 성능 기준을 설정하고, 약 16%의 궤적이 이상치로 표시되었습니다. 최종적으로, 검토된 144개의 궤적이 모호하게 남아 있으며, 이로 인해 95% 신뢰 수준에서 최대 18,271개의 이상치 수치가 도출됩니다. 향후 이상치 검색을 위한 우도 기반 접근 방식이 제시되었습니다.



### DuoLift-GAN:Reconstructing CT from Single-view and Biplanar X-Rays with Generative Adversarial Networks (https://arxiv.org/abs/2411.07941)
- **What's New**: 본 논문에서는 2D 이미지를 독립적으로 3D 표현으로 높이는 듀얼 브랜치 구조인 DuoLift Generative Adversarial Networks (DuoLift-GAN)을 제안합니다. 이 아키텍처는 시각적 사실성을 우선시하는 기존 모델들과는 달리 구조적 정확성을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: DuoLift-GAN은 2D 이미지를 접합하면서 3D 피쳐 맵으로 변환하기 위해 Masked Loss Function을 사용합니다. 이 모델은 X-ray 이미지를 이용하여 정밀한 3D 볼륨을 재구성하도록 설계되어 있으며, 이를 통해 공간적 관계와 일관성을 보존하며 더 정확한 3D 피쳐 맵을 생성합니다.

- **Performance Highlights**: DuoLift-GAN은 기존의 방법들에 비해 재구성 정확성을 크게 향상시켰으며, 시각적 사실성에서도 우수한 성능을 보입니다. 논문에서는 다양한 평가 지표를 통해 DuoLift-GAN의 효과를 입증하고, LIDC-IDRI 데이터셋을 사용하여 기존의 최신 기법들과 비교한 결과를 제공합니다.



### Automatic dataset shift identification to support root cause analysis of AI performance drif (https://arxiv.org/abs/2411.07940)
Comments:
          Code available at this https URL

- **What's New**: 이 논문은 임상 AI 모델의 성능에 영향을 미치는 데이터셋 이동(datasets shift)의 종류를 자동으로 식별할 수 있는 첫 번째 비지도 데이터셋 이동 식별 프레임워크를 제안합니다. 기존의 방법들은 이동의 존재를 감지할 수 있었으나, 어떤 종류의 이동이 발생했는지를 식별하는 것은 어려운 문제였습니다.

- **Technical Details**: 제안된 프레임워크는 두 단계로 나눠지며, 첫 번째 단계는 이동이 존재하는지 감지하는 '이동 감지(shift detection)' 모듈이고, 두 번째 단계는 발견된 이동의 유형을 특성화하는 '이동 식별(shift identification)' 모듈입니다. 이 프레임워크는 prevalence shift, covariate shift, mixed shifts를 구별합니다. 특히, self-supervised encoders 를 활용하여 미세한 covariate shift를 감지할 수 있습니다.

- **Performance Highlights**: 제안된 식별 프레임워크는 3가지 이미지 모달리티(흉부 방사선, 디지털 유방 촬영술, 망막 이미지)에서 5종류의 실제 데이터셋 이동에 대해 유망한 결과를 도출했습니다. 이 연구는 모델 성능과 관련된 데이터셋 이동의 올바른 대응 전략을 선택하기 위한 중요한 기초자료를 제공합니다.



### Rendering-Oriented 3D Point Cloud Attribute Compression using Sparse Tensor-based Transformer (https://arxiv.org/abs/2411.07899)
- **What's New**: 본 논문은 point cloud 데이터의 색상 속성을 위한 최초의 렌더링 지향 압축 프레임워크(RO-PCAC)를 제안합니다. 본 프레임워크는 압축 효율성과 렌더링된 다중 뷰 이미지의 품질 간의 최적의 균형을 목표로 합니다.

- **Technical Details**: RO-PCAC는 전통적인 입력의 재구성에서 벗어나 압축 모듈과 렌더링 모듈을 단일 딥러닝 프레임워크 내에서 통합합니다. 또한, SP-Trans라는 희소 텐서 기반의 변환기(transformer)를 사용하여 point cloud 내 지역적 특성을 분석하고 합성 기능을 향상시킵니다. 이 구조는 로컬 밀도에 따라 지역 이웃을 조정하며, 로컬 자기 주의 메커니즘을 활용합니다.

- **Performance Highlights**: RO-PCAC는 8i Voxelized Full Bodies(8iVFB)와 Owlii 동적 인간 메시(Owlii)와 같은 기준 데이터 세트에서 기존의 전통적, 학습 기반 및 혼합 접근법과 비교하여 최첨단 압축 성능을 획득했습니다.



### Diverse capability and scaling of diffusion and auto-regressive models when learning abstract rules (https://arxiv.org/abs/2411.07873)
Comments:
          12 pages, 5 figures. Accepted to NeurIPS2024 Workshop on System 2 Reasoning At Scale as long paper

- **What's New**: 본 논문에서는 현대 생성 모델이 유한 샘플로부터 기본 규칙을 학습하고 조건적 샘플링을 통해 추론할 수 있는지를 조사합니다. 이를 위해 Raven's Progressive Matrices 작업에서 영감을 받아 GenRAVEN 데이터세트를 설계하였습니다.

- **Technical Details**: GenRAVEN 데이터세트는 40개의 관계 규칙을 기반으로 하며, 각 샘플은 3행으로 구성됩니다. 생성 모델은 diffusion 모델(EDM, DiT, SiT)과 autoregressive 모델(GPT2, Mamba) 두 그룹으로 훈련되었습니다. 다양한 데이터 스케일에서의 성능을 비교하였으며, 각 모델의 샘플 생성 능력을 평가했습니다.

- **Performance Highlights**: diffusion 모델은 기본적인 생성에서 뛰어난 성능을 보여주었고, 새로운 샘플을 더 일관되게 생성할 수 있었습니다. 반면, autoregressive 모델은 누락된 패널을 규칙 일관성 있게 완성하는 데 강점을 보였으나, 기본 생성에서는 일관성이 떨어졌습니다. 데이터 규모에 따른 다양한 성능 변화를 관찰하였고, 앞으로의 연구 방향에 대한 통찰을 제공합니다.



### NL-SLAM for OC-VLN: Natural Language Grounded SLAM for Object-Centric VLN (https://arxiv.org/abs/2411.07848)
- **What's New**: 이번 연구에서는 물체 중심의 자연어 내비게이션 지침에 대한 새로운 데이터셋인 OC-VLN을 통해 랜드마크 기반 내비게이션을 평가하고 NL-SLAM(자연어 기반 SLAM)이라는 새로운 방법을 제안합니다.

- **Technical Details**: NL-SLAM은 자연어 지침을 로봇 관측 및 자세에 맞추는 방법으로, 사전 훈련된 비전 및 언어 기초 모델을 활용하여 작업 별 훈련이 필요 없습니다. 연구는 OC-VLN 데이터셋을 사용하여 물체 중심의 내비게이션 지침을 따르는 능력을 평가합니다.

- **Performance Highlights**: NL-SLAM은 해당 분야의 최신 기법 두 가지(정 OBJECT Goal Navigation, Vision Language Navigation)를 초월하는 성과를 보였으며, Boston Dynamics Spot 로봇에서 실제 내비게이션 지침을 성공적으로 수행하며 그 효과를 입증했습니다.



### Reliable-loc: Robust sequential LiDAR global localization in large-scale street scenes based on verifiable cues (https://arxiv.org/abs/2411.07815)
- **What's New**: 본 논문에서는 신뢰할 수 있는 순차적 글로벌 로컬라이제이션 방법인 Reliable-loc을 제안합니다. 이는 공간적 및 시간적 검증 가능한 단서를 활용하여 LiDAR 기반의 로컬라이제이션 문제를 해결합니다.

- **Technical Details**: Reliable-loc은 지역 특성에서 포함된 풍부한 정보를 활용하여 입자 가중치를 조정하는 몬테 카를로 로컬라이제이션(Monte Carlo Localization, MCL)을 개선합니다. 또한, 순차적인 포즈 불확실성에 의해 안내되는 로컬라이제이션 상태 모니터링 메커니즘을 포함하여, 시간적 검증 가능한 단서를 통해 로컬라이제이션 모드를 적응적으로 전환합니다.

- **Performance Highlights**: Reliable-loc은 대규모 복잡한 거리 장면에서 1.66m의 위치 정확도와 3.09도의 요aw 정확도를 달성하며, 20km 이상의 다양한 거리 장면을 포함한 데이터셋에서 높은 강건성과 효율성을 보여주었습니다.



### Interaction Asymmetry: A General Principle for Learning Composable Abstractions (https://arxiv.org/abs/2411.07784)
Comments:
          Preprint, under review

- **What's New**: 이 연구에서는 개념의 분리된 표현(Disentangled Representations) 학습을 위한 새로운 원리인 상호작용 비대칭(Interaction Asymmetry)을 제안했습니다. 이 원리는 동일한 개념의 부분들이 서로 다른 개념의 부분들보다 더 복잡한 상호작용을 갖는다고 설명합니다.

- **Technical Details**: 우리는 이 원리를 개념을 관측된 데이터로 변환하는 생성기(Generator)의 $(n+1)$차 미분에 대한 블록 대각 조건(Block Diagonality Conditions)을 통해 정 형화합니다. 여기서 서로 다른 '복잡성'의 차수는 서로 다른 $n$에 대응합니다.

- **Performance Highlights**: 합성 이미지 데이터셋에서는 제안된 모델이 더 명시적인 객체 중심 선행 조건을 사용하는 기존 모델과 비교해 유사한 객체 분리 성능을 달성할 수 있음을 입증하였습니다.



### SAV-SE: Scene-aware Audio-Visual Speech Enhancement with Selective State Space Mod (https://arxiv.org/abs/2411.07751)
- **What's New**: 본 논문에서는 새로운 오디오-비주얼 음성 향상(Speech Enhancement, SE) 작업인 SAV-SE를 제안합니다. 기존 연구가 주로 얼굴 및 입술 운동에 집중되었던 반면, 본 연구는 환경의 맥락적 시각 정보(visual contextual cues)를 활용하여 음성 향상 성능을 개선하는 방법을 제시합니다.

- **Technical Details**: VC-S$^2$E 방법은 Conformer 및 Mamba 모듈을 통합하여 두 가지의 상호 보완적인 강점을 활용합니다. Hybrid convolution-SSM 아키텍처를 기반으로 하는 ConMamba는 글로벌 인터랙션(long-range global interactions)과 세밀한 특징 패턴(localized fine-grained feature patterns)을 포착할 수 있습니다.

- **Performance Highlights**: MUSIC, AVSpeech 및 AudioSet 데이터셋에서 수행된 실험 결과, VC-S$^2$E가 기존의 경쟁 방식들에 비해 음성 품질과 음성의 이해도를 개선하는데 우수함을 입증했습니다.



### LapGSR: Laplacian Reconstructive Network for Guided Thermal Super-Resolution (https://arxiv.org/abs/2411.07750)
- **What's New**: 최근 연구에서, LapGSR라는 새로운 경량화된 다중 모달 생성 모델이 제안되었습니다. 이 모델은 RGB 이미지의 라플라시안 이미지 피라미드를 활용하여 열 화상 이미지를 고해상도로 변환하는 데 중점을 두고 있습니다. 이는 고가의 고해상도 이미지 센서를 사용하지 않고도 성능을 개선할 수 있는 방법을 제공합니다.

- **Technical Details**: LapGSR 모델은 RGB 이미지에서 중요한 엣지 정보를 추출하여 열 화상 이미지의 고해상도를 달성합니다. 이 접근법은 pixel 및 adversarial loss(적대적 손실)를 결합하여 모델의 효율성과 compactness(소형화)를 유지합니다. 추가로, 고해상도 RGB 이미지를 가이드를 사용하여 저해상도 열 이미지를 회복하는 Guided Thermal Super Resolution 기법에 중점을 둡니다.

- **Performance Highlights**: 이 모델은 ULB17-VT 및 VGTSR 데이터셋에서 다른 최신 모델들보다 90% 및 50% 더 적은 학습 가능한 매개변수로 일관된 성능을 보였습니다. LapGSR는 고해상도 이미지 복원에서 뛰어난 시각적 충실도와 정확성을 달성하며, 실제 배치에서도 효과적인 강점을 가지고 있습니다.



### EMPERROR: A Flexible Generative Perception Error Model for Probing Self-Driving Planners (https://arxiv.org/abs/2411.07719)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 새로운 트랜스포머 기반의 생성형 인식 오류 모델(PEM)인 Emperror를 제안합니다. 이를 통해 경험 기반의 데이터로부터 자율 주행 플래너를 스트레스 테스트할 수 있는 틀을 제공합니다.

- **Technical Details**: Emperror는 트랜스포머 아키텍처를 기반으로 구축되어, 현대의 객체 탐지기와의 불일치를 더 정밀하게 모방합니다. 이러한 PEM은 다양한 오류 패턴을 모델링하고, 자율 주행 플래너의 강인성을 시험할 수 있는 도전적인 샘플을 생성하는 역할을 합니다.

- **Performance Highlights**: Emperror를 사용하여 생성된 노이즈 입력으로는 플래너의 충돌률이 최대 85%까지 증가하여, 자율 주행 플래너의 평가에 있어 유용한 도구로 기능함을 보였습니다.



### AI enhanced diagnosis of Peyronies disease a novel approach using Computer Vision (https://arxiv.org/abs/2411.07684)
Comments:
          8 pages, 6 figures, 4 tables

- **What's New**: 본 연구는 페이로니병(Peyronie's Disease, PD) 진단을 위한 혁신적인 AI 기반 도구를 제시합니다. 이 도구는 전 세계 남성의 0.3%에서 13.1%에 영향을 미치는 이 질환의 진단을 개선하기 위해 개발되었습니다.

- **Technical Details**: 저희 방법은 이미지와 비디오에서 주요 포인트 탐지(key point detection)를 사용하여 음경의 굴곡 각도를 측정합니다. 고급 컴퓨터 비전 기술(advanced computer vision techniques)을 활용하여 해부학적 마크(anatomical landmarks)를 정확하게 확인합니다. 전통적인 진단법은 주관적(subjective)이고 침습적(invasive)인 방법을 포함해 환자에게 불편함과 정확도 저하를 초래할 수 있습니다.

- **Performance Highlights**: 본 모델은 PD와 정상 해부학적 변화(normal anatomical changes)를 96.7%의 민감도(sensitivity)와 100%의 특이도(specificity)로 구별합니다. 이러한 발전은 비뇨기학 진단에 상당한 개선을 가져오며, 의료 제공자와 환자를 위한 PD 평가의 효율성과 편리성을 크게 향상시킵니다.



### SegQC: a segmentation network-based framework for multi-metric segmentation quality control and segmentation error detection in volumetric medical images (https://arxiv.org/abs/2411.07601)
Comments:
          28 pages, 9 figures

- **What's New**: SegQC는 볼륨 의료 이미지에서 세그멘테이션(segmentation) 품질 추정 및 오류 감지를 위한 새로운 프레임워크입니다. 이 프레임워크는 의료 실습에서의 세그멘테이션 오류 검출 및 모델 개발을 용이하게 합니다.

- **Technical Details**: SegQC는 다음의 주요 요소들로 구성됩니다: 1. SegQC-Net - 스캔과 세그멘테이션 마스크를 입력으로 받아 각 복셀(voxel)에 대한 세그멘테이션 오류 확률을 출력하는 딥 네트워크(deep network); 2. 세그멘테이션 오류 확률을 바탕으로 계산된 세 가지 새로운 세그멘테이션 품질 메트릭(metrics) - 두 개의 오버랩(overlap) 메트릭과 구조 크기 메트릭(structure size metric); 3. 스캔 슬라이스에서 가능한 세그멘테이션 오류를 감지하는 새로운 방법.

- **Performance Highlights**: SegQC는 198개의 태아 MRI 스캔에서 태아 뇌, 태아 몸, 태반의 세 구조에 대해 시험되었습니다. SegQC는 Pearson correlation과 MAE(Mean Absolute Error) 측면에서 TTA 기반 품질 추정보다 더 우수한 성능을 보였습니다. 세그멘테이션 오류 감지 방법은 태아 몸과 태아 뇌의 경우 각각 0.77 및 0.48, 0.74 및 0.55의 재현율(recall)과 정밀도(precision) 비율을 달성하였습니다.



### Uncertainty-Aware Test-Time Adaptation for Inverse Consistent Diffeomorphic Lung Image Registration (https://arxiv.org/abs/2411.07567)
Comments:
          5 pages, 4 figures

- **What's New**: 본 연구에서는 불확실성을 고려한 테스트 시간 적응 프레임워크를 제안하여 폐 이미지의 역일관성 있는 차원화 방법을 개선하고자 하였습니다. 이 방법은 Monte Carlo (MC) dropout을 활용하여 공간적 불확실성(map) 지도를 생성하고 이를 통해 모델 성능을 향상시킵니다.

- **Technical Details**: 우리는 변형될 이미지 맞춤(matching)을 위해 변형적 이미지 등록(Deformable Image Registration, DIR) 프레임워크를 사용하였습니다. 고정 이미지와 이동 이미지를 정렬하는 최적의 변환을 찾는 문제로, 이는 미분 가능하고 역변환 가능성의 보장을 포함하는 LDDMM(Large Deformation Diffeomorphic Metric Mapping) 프레임워크를 바탕으로 하였습니다. 몬테카를로 dropout을 통해 불확실성 지도를 생성하고 이를 이용해 모델을 적응 및 최적화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 폐 경계에서 Dice 유사도 지수(DSC) 0.966을 달성하였으며, 이는 기존의 VoxelMorph(0.953) 및 TransMorph(0.953)보다 높은 수치입니다. 양방향 등록에서도 역등록 방향에 대한 일관된 개선이 관찰되었습니다.



### A Novel Automatic Real-time Motion Tracking Method for Magnetic Resonance Imaging-guided Radiotherapy: Leveraging the Enhanced Tracking-Learning-Detection Framework with Automatic Segmentation (https://arxiv.org/abs/2411.07503)
- **What's New**: 이번 연구에서는 MRI 유도 방사선 치료(MRIgRT)의 정확한 모션 트래킹(motion tracking)을 보장하기 위해 자동 실시간 트래킹 방법을 개선했습니다. ETLD(Enhanced Tracking-Learning-Detection) 프레임워크와 자동 분할(automatic segmentation)을 결합하여 ETLD+ICV(Improved Chan-Vese model)라는 새로운 방식을 구현하였습니다.

- **Technical Details**: ETLD+ICV 방법은 두 가지 주요 방법을 통합한 것입니다. TLD 프레임워크는 실시간 cine MRI에 적합하도록 업그레이드되었으며, 고급 이미지 전처리, 비참조 이미지 품질 평가, 향상된 메디안 흐름 추적기(median-flow tracker), 동적 검색 영역 조정이 가능한 정제된 탐지기를 포함합니다. ICV는 타겟 볼륨(target volume)을 정밀하게 커버하기 위해 사용하는 것으로, 트래킹 결과를 기반으로 분할된 영역을 프레임 별로 개선합니다.

- **Performance Highlights**: 106,000 프레임을 77개의 치료 분획(fraction)에 걸쳐 평가한 결과, 모든 대상에서 서브 밀리미터(less than 0.8mm)의 트래킹 오류와 99% 이상의 정밀도 및 98%의 재현율(recall)을 달성했습니다. ETLD+ICV는 모든 대상에서 82% 이상의 Dice global score를 기록하였으며, 이는 제안된 방법의 확장성과 정밀한 타겟 볼륨 커버리지를 잘 보여줍니다.



### LAUREL: Learned Augmented Residual Layer (https://arxiv.org/abs/2411.07501)
Comments:
          Accepted at the 2nd Efficient Systems for Foundation Models Workshop at the International Conference on Machine Learning (ICML) 2024

- **What's New**: 이 논문에서는 전통적인 잔여 연결(residual connection)의 일반화된 형태인 학습된 증강 잔여 레이어(Learned Augmented Residual Layer, LAuReL)를 소개합니다. LAuReL은 모델 품질과 메모리 사용량 모두에서 기존 방법을 초월하는 것을 목표로 합니다.

- **Technical Details**: LAuReL의 주요 아이디어는 잔여 연결을 다음과 같이 재구성하는 것입니다: α는 학습된 스칼라 매개변수이며, g(⋅)는 학습된 선형 함수입니다. 이 함수는 잔여 연결의 출력을 입력으로 사용하여 더 복잡한 정보 흐름을 형성합니다. LAuReL은 모델의 크기와 지연(latency) 측면에서 경량화된 방식으로 이러한 잔여 흐름을 학습할 수 있습니다.

- **Performance Highlights**: ResNet-50과 ImageNet 1K 작업에서 LAuReL은 추가 레이어를 추가했을 때의 성능 향상 중 60%를 달성하면서도 파라미터 수는 0.003%만 증가했습니다. 이는 LAuReL이 적은 파라미터로도 높은 성능을 보장한다는 것을 보여줍니다.



### Quantifying Knowledge Distillation Using Partial Information Decomposition (https://arxiv.org/abs/2411.07483)
Comments:
          Accepted at NeurIPS 2024 Machine Learning and Compression Workshop

- **What's New**: 본 논문은 Knowledge Distillation의 정보 이론적 한계를 규명하기 위한 새로운 메트릭을 도입했습니다. 이를 통해 교사 모델(teacher model)의 지식을 학생 모델(student model)과 특정 다운스트림 작업(downstream task)에 맞춰 정량화하는 방법을 제안합니다.

- **Technical Details**: 우리의 연구에서는 Partial Information Decomposition (PID)을 활용하여 교사가 제공할 수 있는 정보에 대한 새로운 양적 지표를 정의했습니다. 이 지표는 Task에 대한 교사만의 유일한 정보를 나타내며, 이로써 지식 증류 과정에서 필터를 통한 정보 정제 및 개선이 가능합니다.

- **Performance Highlights**: CIFAR10 데이터셋에 대한 실험을 통해, 새로운 Knowledge Distillation 프레임워크인 Redundant Information Distillation (RID)이 기존의 Variational Information Distillation (VID)보다 성능이 우수하다는 것을 입증했습니다.



### T2-Only Prostate Cancer Prediction by Meta-Learning from Bi-Parametric MR Imaging (https://arxiv.org/abs/2411.07416)
Comments:
          Code: this https URL

- **What's New**: 기존의 MR 이미징을 기반으로 한 전립선암 진단에서 사용되는 T2-weighted (T2w) 및 diffusion-weighted imaging (DWI) 시퀀스를 결합한 접근법 대신, 본 연구는 T2w 시퀀스만을 사용하여 머신러닝 (ML) 모델을 개선하는 가능성을 탐구합니다. 이는 DWI 시퀀스 없이도 ML 모델이 방사선 전문의 수준의 정확도로 전립선암을 탐지할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 방법은 T2w 이미지만을 이용하여 전립선암을 국소화하는 비수준 메타러닝 프레임워크입니다. 모델은 두 가지 주요 요소인 모달리티 변환기와 암 예측기를 최적화하여 작동하며, 여기서 모달리티 변환기는 DWI 시퀀스를 사용하여 T2w 이미지를 학습하고, 암 예측기는 T2w 이미지를 통해 암 병변을 예측합니다.

- **Performance Highlights**: 3000명 이상의 전립선암 환자를 대상으로 수행된 여러 데이터세트를 사용하여, T2w 만을 입력으로 사용하는 모델이 방사선 전문의가 식별한 전립선암을 국소화하는 데 있어서 다른 모델들에 비해 우수하거나 비교 가능한 성능을 보였습니다. 이 연구는 T2w만을 사용한 모델의 실제 환자 사례를 통해, 다양한 입력 시퀀스를 가진 모델이 실제 긍정 사례를 얼마나 잘 식별할 수 있는지를 처음으로 보여줍니다.



### Federated Learning Client Pruning for Noisy Labels (https://arxiv.org/abs/2411.07391)
- **What's New**: 이 논문은 Federated Learning(FL) 환경에서 노이즈가 있는 레이블 문제를 해결하기 위해 ClipFL(석방된 훈련 클라이언트 제외)이라는 새로운 접근 방식을 제시합니다.

- **Technical Details**: ClipFL은 세 가지 단계로 구성되어 있습니다: (1) 클라이언트의 성능을 기준으로 노이즈 클라이언트를 식별하고 Noise Candidacy Score(NCS)를 계산합니다. (2) NCS가 가장 높은 클라이언트의 일부를 제외합니다. (3) 남은 클라이언트에 대해 표준 FL을 수행하여 모델을 정제합니다.

- **Performance Highlights**: ClipFL은 다양한 데이터셋과 노이즈 수준에서 우수한 성능을 나타내어, 노이즈가 있는 클라이언트를 80% 이상의 정확도로 식별하고, 기존 FL 최적화 방법들보다 개선된 성능과 더 빠른 수렴을 보였습니다. 또한 통신 비용이 줄어드는 효과도 확인되었습니다.



### Exploring Variational Autoencoders for Medical Image Generation: A Comprehensive Study (https://arxiv.org/abs/2411.07348)
Comments:
          for associated mpeg file, see this https URL

- **What's New**: 본 논문은 의료 이미지 생성 분야에서 Variational Autoencoder (VAE)의 연구 동향을 종합적으로 리뷰하고 있습니다. 특히 VAE가 실제 데이터와 유사한 합성 이미지를 생성하는 능력에 집중하며, 데이터 증강(data augmentation)으로의 활용 가능성을 강조합니다.

- **Technical Details**: VAE는 작은 데이터셋이나 클래스 불균형이 있는 데이터셋에서 샘플을 추가하여 데이터셋을 개선하는 장점을 가지고 있습니다. 본 논문에서는 의료 이미지를 위한 VAE의 주요 구조와 방법론, 그리고 GANs와 같은 다른 생성 모델과의 비교를 다룹니다.

- **Performance Highlights**: 최근 의료 분야에서의 응용을 통해 VAE가 분할(segmentation) 및 분류(classification) 정확도를 개선할 수 있는 능력을 강조합니다.



### Multimodal Fusion Balancing Through Game-Theoretic Regularization (https://arxiv.org/abs/2411.07335)
Comments:
          21 pages, 6 figures, 4 tables, 1 algorithm

- **What's New**: 이 논문에서는 Multimodal Competition Regularizer (MCR)이라는 새로운 손실 함수를 소개하고, 이를 통해 멀티모달 훈련에서 경쟁으로 인한 부작용을 방지할 수 있도록 설계하였습니다. MCR은 서로 다른 모달리티 간의 의존성을 분해하여 학습의 효율성을 높이고자 합니다.

- **Technical Details**: MCR은 서로 경쟁하는 모달리티가 최종 결과에 미치는 영향을 최대화하는 게임 이론적 원칙을 도입하여 두 가지 상하한을 설정합니다. 이를 통해 학습 과정에서 각 모달리티의 기여도를 조정하고,조건부 MI(Mutual Information)의 추정에 대한 잠재 공간의 변환을 제안하여 계산 효율성을 크게 개선합니다.

- **Performance Highlights**: MCR은 기존의 훈련 전략들보다 뛰어난 성능을 발휘하며, 단순한 앙상블 기준을 넘어 멀티모달 학습을 일관되게 개선하는 첫 번째 방법으로, 합성 및 실제 데이터셋에서 모두 성능 향상을 입증합니다. MCR은 특히 AVE, UCF, CREMA-D, CMU-MOSI, CMU-MOSEI 등 다양한 데이터셋에서 우수한 성능을 기록하였습니다.



### Artificial Intelligence-Informed Handheld Breast Ultrasound for Screening: A Systematic Review of Diagnostic Test Accuracy (https://arxiv.org/abs/2411.07322)
- **What's New**: 유방암 조기검진을 위한 AI 기반의 휴대용 초음파(Handheld Breast Ultrasound, BUS)에 대한 연구가 진행되었습니다. 이 기술은 저비용으로 유방암을 탐지하고 분류하는 데 도움을 줄 수 있습니다.

- **Technical Details**: 이 리뷰는 PRISMA 및 SWiM 가이드라인에 따라 수행되었으며, 2016년 1월 1일부터 2023년 12월 12일까지의 문헌을 검색했습니다. 763개의 후보 연구 중에서 314개의 전체 텍스트가 검토되었고, 34개의 연구가 포함되었습니다. 포함된 연구는 이미지 프레임 선택(1개), 탐지(detection, 6개), 세분화(segmentation, 11개), 분류(classification, 16개)의 AI 작업 유형에 따라 그룹화되었습니다. 570만 개의 BUS 이미지가 AI 교육 및 검증에 사용되었습니다.

- **Performance Highlights**: 모든 연구에서 AI 기반 BUS가 높은 성능을 보였지만, AI 강화 BUS를 지지하는 증거는 전반적으로 안정성이 부족했습니다. 자원이 제한된 환경에서 스크리닝 접근성을 증가시키기 위한 AI 강화 BUS의 잠재력을 실현하기 위해서는 고품질 모델 검증이 중요합니다.



New uploads on arXiv(cs.AI)

### Learning with Less: Knowledge Distillation from Large Language Models via Unlabeled Data (https://arxiv.org/abs/2411.08028)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)을 활용하여 레이블이 없는 데이터를 대상으로 소형 모델을 훈련하는 효율적인 방법인 LLKD를 제안합니다. LLKD는 학습 과정 중 발생할 수 있는 최적화 문제와 노이즈 있는 pseudo-labels에 대한 해결책을 제공합니다.

- **Technical Details**: LLKD는 적응형 샘플 선택 방법으로, 교사 모델(LLM)의 레이블링에 대한 높은 신뢰도를 보이는 샘플과 학생 모델이 높은 불확실성을 보이는 샘플을 우선적으로 선택하여, 지식 증류(Knowledge Distillation) 과정의 효율을 높입니다. 이 방법은 각 훈련 단계에서 교사와 학생의 신뢰도 및 불확실성을 기반으로 하여 샘플을 선정합니다.

- **Performance Highlights**: 종합적인 실험 결과, LLKD는 여러 데이터 세트에서 모델 성능을 향상시키며, 데이터 효율성 측면에서도 유의미한 개선을 보였습니다.



### Leonardo vindicated: Pythagorean trees for minimal reconstruction of the natural branching structures (https://arxiv.org/abs/2411.08024)
Comments:
          22 pages, lots of hi res figures I had to reduce quality of, submitting as a requirement to the Theory of Computing Journal

- **What's New**: 최근 Pythagorean tree(피타고라스 나무) 구조의 다양한 변형을 조사하여 자연에서 관찰되는 나뭇가지 구조와 유사한 형상을 찾고 설명하는 연구가 진행되었습니다. 이 연구는 나무의 구조적 아름다움과 그 공학적 최적성을 동시에 탐구하고 있습니다.

- **Technical Details**: Pythagorean tree는 정삼각형의 변에 사각형을 두어 구성된 프랙탈 디자인으로, 나뭇가지의 지점에서 자연적인 나무의 분기 구조를 모방하는 알고리즘을 개발하였습니다. 이를 통해 다양한 혼합 파라미터를 조정하여 나무의 구조를 시뮬레이션하고 CNN(Convolutional Neural Networks)을 이용하여 나무의 진짜 이미지를 분류하는 과정에서 자연적인 나무 구조를 재현하는 데 성공하였습니다.

- **Performance Highlights**: 이 연구에서 생성된 프랙탈 트리는 CNN의 분류 정확도를 높이는 결과를 가져왔으며, 이는 Leonardo da Vinci의 분기 규칙과 황금비를 기반으로 한 나무의 구조적 원리를 뒷받침하는 것으로 나타났습니다. 이를 통해 인공적으로 생성된 트리 모델이 다양한 나무 종의 탐지를 위한 강력한 훈련 데이터로 사용될 수 있음을 주장합니다.



### Can adversarial attacks by large language models be attributed? (https://arxiv.org/abs/2411.08003)
Comments:
          7 pages, 1 figure

- **What's New**: 이 논문은 사이버 공격 및 허위 정보와 같은 적대적 환경에서의 대형 언어 모델(LLM) 출력의 귀속 문제를 다룹니다. 특히 포멀 언어 이론(formal language theory)을 사용하여 언어 식별(language identification) 문제를 조사합니다.

- **Technical Details**: 연구에서는 LLM의 출력을 포멀 언어로 모델링하고, 유한한 텍스트 샘플이 원래 모델을 유일하게 식별할 수 있는지 여부를 분석합니다. 결과적으로 특정 언어 클래스의 비식별성(non-identifiability) 때문에, 세밀한 모델 조정에서의 겹치는 출력에 관한 약간의 가정 하에서도, 출력의 귀속을 특정 LLM에 확실하게 할 수 없는 이론적 한계를 발견했습니다.

- **Performance Highlights**: 추가적으로, Transformer 아키텍처의 표현성 한계를 고려하더라도, 모델에 대한 직접적인 접근이나 포괄적인 모니터링이 있더라도 귀속 노력에 심각한 계산적 장애가 존재하는 것을 보여주었습니다. 이 연구 결과는 적대적 LLM 사용으로 인한 위험을 완화하기 위한 적극적인 조치의 필요성을 강조합니다.



### Gini Coefficient as a Unified Metric for Evaluating Many-versus-Many Similarity in Vector Spaces (https://arxiv.org/abs/2411.07983)
- **What's New**: 이번 논문에서는 Gini 계수를 벡터 공간에서 다대다(모든 벡터 간의 유사성) 유사도를 평가할 수 있는 통합 메트릭으로 사용 가능하다는 것을 보여주었습니다. 여러 이미지 데이터셋에 대한 분석 결과, Gini 계수가 높은 이미지들이 서로 더 유사하고 Gini 계수가 낮은 이미지들은 덜 유사하다는 것을 발견했습니다. 텍스트 벡터 임베딩에도 마찬가지로 적용되며, 다양한 데이터 유형에 대한 일관성을 강조합니다.

- **Technical Details**: Gini 계수는 일반적으로 소득이나 부의 불평등을 측정하는 경제학적 도구로 알려져 있으나, 본 논문에서는 이를 벡터 공간에서의 다대다 유사성을 평가하는 데 활용합니다. 벡터는 ℝ^d 차원의 실수 벡터로 표현되며, Gini 계수를 통해 각 벡터가 다른 벡터들과의 유사성을 측정할 수 있습니다. 이러한 방법은 전통적인 유사성 검색이 놓칠 수 있는 데이터를 평가하는 데 유용합니다.

- **Performance Highlights**: 기계 학습의 훈련 샘플을 선택할 때 Gini 계수가 높은 샘플을 선택하는 것이 중요하다는 사실을 발견했습니다. 테스트 데이터셋의 분포와 유사한 훈련 샘플을 선정하는 것이 데이터 다양성을 보장하는 것보다 모델 성능을 크게 향상시킵니다. 선정한 샘플이 Gini 계수가 높은 경우, 랜덤 샘플링보다 우수한 성능을 보였습니다.



### How To Discover Short, Shorter, and the Shortest Proofs of Unsatisfiability: A Branch-and-Bound Approach for Resolution Proof Length Minimization (https://arxiv.org/abs/2411.07955)
Comments:
          42 pages, 16 figures, 8 tables, submitted to Journal of Artificial Intelligence Research

- **What's New**: 이 논문은 각기 다른 단계의 절차를 그룹화하는 레이어 리스트 표현(layer list representation)을 사용하여 가장 짧은 resolution proof를 찾기 위한 새로운 branch-and-bound 알고리즘을 제안합니다. 이 접근법은 permutation의 대칭을 모두 파괴하며, 기존의 symmetry-breaking 기법들을 개선합니다.

- **Technical Details**: 우선, 'proof length lower bound', 'clause subsumption', 및 'dominance'를 기반으로 한 pruning 절차를 설계하였습니다. 새로운 알고리즘은 2002 SAT Competition의 unsatisfiable 인스턴스에서 기존 solver보다 증명을 30-60% 단축할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법은 기존의 SAT solver에 비해 두 배의 인스턴스를 해결할 수 있었으며, 최적화 시간이 대폭 감소했습니다. 또한, 이 알고리즘은 resolution proofs의 메모리 소비와 관련된 한계를 발견하였고, 이는 proofs가 10^6 단계를 초과할 때 지속적으로 작동하지 않는 경향을 보입니다.



### Towards Low-bit Communication for Tensor Parallel LLM Inferenc (https://arxiv.org/abs/2411.07942)
- **What's New**: 이 논문에서는 tensor parallelism의 커뮤니케이션 비용을 줄이기 위한 새로운 양자화(quantization) 방법을 제안합니다. 기존 tensor parallelism의 양자화 방법은 주로 높은 정밀도로 출력 기능을 유지하는 것을 목표로 했으나, 본 연구는 양자화를 통해 커뮤니케이션을 평균 16비트에서 4.2비트로 줄이고, 원래 모델의 성능을 거의 유지하는 것을 이룹니다.

- **Technical Details**: 제안된 방법은 각 디바이스 간의 tensor parallelized attention 및 feedforward 블록에서 저비트(low-bit) 출력을 효과적으로 통신할 수 있도록 설계되었습니다. 주요 아이디어는 아웃라이어(outlier) 기능을 정적으로 선택하여 BF16으로 유지하고, 나머지는 4비트로 양자화(quantization)하는 것입니다. 이 과정에서 원래 가중치(weights)에 교란을 주지 않는 특징이 있습니다.

- **Performance Highlights**: 이 방법은 Gemma 2 27B 모델의 경우 원래 성능의 약 98%, Llama 2 13B 모델의 경우 99.5%를 유지하며, 통신하는 정보의 양은 약 1/4로 줄어듭니다. 이러한 성능 유지는 재구성 오류를 최소화하며 고성능의 대규모 언어 모델을 위한 전략적인 접근법을 제공합니다.



### Automatic dataset shift identification to support root cause analysis of AI performance drif (https://arxiv.org/abs/2411.07940)
Comments:
          Code available at this https URL

- **What's New**: 이 논문은 임상 AI 모델의 성능에 영향을 미치는 데이터셋 이동(datasets shift)의 종류를 자동으로 식별할 수 있는 첫 번째 비지도 데이터셋 이동 식별 프레임워크를 제안합니다. 기존의 방법들은 이동의 존재를 감지할 수 있었으나, 어떤 종류의 이동이 발생했는지를 식별하는 것은 어려운 문제였습니다.

- **Technical Details**: 제안된 프레임워크는 두 단계로 나눠지며, 첫 번째 단계는 이동이 존재하는지 감지하는 '이동 감지(shift detection)' 모듈이고, 두 번째 단계는 발견된 이동의 유형을 특성화하는 '이동 식별(shift identification)' 모듈입니다. 이 프레임워크는 prevalence shift, covariate shift, mixed shifts를 구별합니다. 특히, self-supervised encoders 를 활용하여 미세한 covariate shift를 감지할 수 있습니다.

- **Performance Highlights**: 제안된 식별 프레임워크는 3가지 이미지 모달리티(흉부 방사선, 디지털 유방 촬영술, 망막 이미지)에서 5종류의 실제 데이터셋 이동에 대해 유망한 결과를 도출했습니다. 이 연구는 모델 성능과 관련된 데이터셋 이동의 올바른 대응 전략을 선택하기 위한 중요한 기초자료를 제공합니다.



### Leveraging Multimodal Models for Enhanced Neuroimaging Diagnostics in Alzheimer's Diseas (https://arxiv.org/abs/2411.07871)
Comments:
          The paper has been accepted by the conference: "2024 International Conference on Big Data (IEEE Big Data 2024)"

- **What's New**: 이 논문은 신경영상(neuroimaging) 분야에서 알츠하이머병(Alzheimer's Disease) 진단을 위한 합성 진단 보고서를 생성하여 기존 데이터세트의 부족한 부분을 보완하는 새로운 접근법을 제시합니다.

- **Technical Details**: 연구팀은 OASIS-4 데이터셋에서 663명의 환자 데이터를 활용하여 GPT-4o-mini를 통해 합성 진단 보고서를 생성했습니다. 이후 BiomedCLIP과 T5 모델을 사용하여 이미지를 기반으로 신경학적 보고서를 생성하였습니다. 이 방법론은 BLU-4 점수 0.1827, ROUGE-L 점수 0.3719, METEOR 점수 0.4163을 기록하여 임상적으로 중요한 진단 보고서 생성을 가능하게 하는 잠재력을 보여주었습니다.

- **Performance Highlights**: 제안된 방법은 기존 알고리즘 대비 신경영상 데이터에 대한 향상된 진단 보고서 생성 성능을 보였으며, 특히 텍스트 기반 진단 데이터가 부족한 알츠하이머병 진단의 질을 향상시킬 수 있는 가능성을 제시합니다.



### Federated Learning for Discrete Optimal Transport with Large Population under Incomplete Information (https://arxiv.org/abs/2411.07841)
- **What's New**: 이 논문은 대규모 이질적(target heterogeneous) 대상을 처리할 수 있는 새로운 이산 최적 수송(discrete optimal transport) 프레임워크를 제안합니다. 두 가지 시나리오, 즉 대상을 유형 분포(type distribution)가 알려진 경우와 알려지지 않은 경우를 고려합니다.

- **Technical Details**: 유형 분포가 알려진 경우, 각 타겟 노드에 리소스를 최적 분배하기 위한 완전 분산 알고리즘(fully distributed algorithm)을 제안하고, 알려지지 않은 경우에는 연합 학습(federated learning) 기반 접근 방식을 개발하여 개인 정보를 보호하면서 최적 수송 계획(optimal transport scheme)을 효율적으로 계산합니다.

- **Performance Highlights**: 제안된 학습 알고리즘의 성능을 평가하기 위해 여러 사례 연구(case studies)를 제공하고 있으며, 개인 정보 보호가 중요한 환경에서의 활용 가능성을 강조합니다. 또한, 대규모 이질적 노드들 간의 자원 할당 문제를 해결하기 위해 연합 학습 알고리즘을 효과적으로 적용한 사례들을 다룹니다.



### Community Research Earth Digital Intelligence Twin (CREDIT) (https://arxiv.org/abs/2411.07814)
- **What's New**: 최근 인공지능(AI)을 활용한 수치 기상 예측(NWP) 모델의 발전은 대기 모델링 분야를 크게 변화시켰습니다. 특히, CREDIT 프레임워크를 통해 AI NWP 모델의 효과성과 사용 편의성을 높이며, 높은 성능의 컴퓨팅 시스템에서의 모델 훈련 및 배포가 가능해졌습니다.

- **Technical Details**: CREDIT 프레임워크는 데이터 전처리(data preprocessing), 모델 훈련(model training), 평가(evaluation)를 포함하는 사용자 친화적인 파이프라인을 제공합니다. WXFormer이라는 새로운 비전 변환기 모델을 통해 대기 상태를 자가회귀적으로 예측할 수 있으며, 스펙트럴 정규화(spectral normalization)와 다단계 훈련(multi-step training) 등의 기술을 활용하여 일반적인 AI NWP 문제를 해결하고 있습니다.

- **Performance Highlights**: FUXI 아키텍처와 WXFormer 모두 6시간마다 수집된 ERA5 하이브리드 시그마-압력 수준에서 훈련된 결과, 10일 예측에서 IFS HRES를 일반적으로 초월하는 성능을 보였습니다. CREDIT의 모듈형 설계를 통해 다양한 모델, 데이터셋, 훈련 구성을 탐색할 수 있습니다.



### PatchCTG: Patch Cardiotocography Transformer for Antepartum Fetal Health Monitoring (https://arxiv.org/abs/2411.07796)
- **What's New**: 이 논문에서는 Antepartum Cardiotocography (CTG) 분석을 위한 최신 Transformer 기반 모델인 PatchCTG를 소개합니다. 기존의 CTG 분석 방법은 해석의 일관성이 부족하여 오류가 발생하기 쉬운데, PatchCTG는 패치 기반 토크나이제이션과 인스턴스 정규화, 채널 독립 처리 기술을 이용하여 이러한 한계를 극복하고자 합니다.

- **Technical Details**: PatchCTG는 CTG 데이터를 패치로 나누고, 각 패치마다 채널 독립적으로 처리하여 로컬 및 글로벌 시간 종속성(temporal dependencies)을 포착합니다. 또한 인스턴스 정규화를 통해 데이터의 분포 변화(distribution shifts)를 관리하고, FHR(심박수) 및 자궁 수축 패턴을 더 정확하게 모델링할 수 있도록 설계되었습니다. 이를 통해 PatchCTG는 임신 중 다양한 임상 요구에 적합한 분석을 수행할 수 있습니다.

- **Performance Highlights**: PatchCTG는 Oxford Maternity (OXMAT) 데이터 세트를 활용하여 검증되었습니다. 실험 결과, PatchCTG는 AUC 77%, 특이도(specificity) 88%, 민감도(sensitivity) 57%를 기록했으며, 특히 출산 직전 데이터를 활용한 미세 조정(fine-tuning)에서 민감도 52% 및 특이도 88%를 달성했습니다. 이러한 성과는 PatchCTG가 안정적이고 신뢰할 수 있는 임신 중 건강 상태 평가 도구로 사용될 가능성을 시사합니다.



### Unlocking Legal Knowledge with Multi-Layered Embedding-Based Retrieva (https://arxiv.org/abs/2411.07739)
Comments:
          27 pages, 10 figures

- **What's New**: 이 연구는 법률 지식의 복잡성을 포착하기 위한 다층적인 embedding 기반 검색 방법을 제안합니다. 개별 조항뿐만 아니라 그 구성요소(문단, 조항)와 구조적 집합(책, 제목, 장 등)에도 embedding을 생성하여 법률 정보의 미세함을 캡처하고 있습니다.

- **Technical Details**: 법률 문서의 고유한 계층 구조에 따라, 다양한 세분화의 레벨에서 embedding을 통해 작은 조각에서부터 전반적인 섹션까지 정보를 검색할 수 있게 합니다. 또한, embedding을 통해 법률 문서에 내재된 관계를 표현하며, 이 방법론은 Retrieval Augmented Generation (RAG) 시스템을 통해 사용자 쿼리에 정확하게 응답할 수 있도록 합니다.

- **Performance Highlights**: 이 방법론은 브라질의 법제와 헌법에 주로 초점을 맞추고 있지만, 원칙적으로 공통법 체제에서도 적용 가능하다고 주장합니다. 또한, 이 기법은 법률 분야를 넘어, 계층적인 텍스트로 인코딩된 정보를 조직하고 검색하는 데 유용한 통찰을 제공합니다.



### Is Cognition consistent with Perception? Assessing and Mitigating Multimodal Knowledge Conflicts in Document Understanding (https://arxiv.org/abs/2411.07722)
Comments:
          Preprint

- **What's New**: 최근 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)에 대한 연구가 급격히 발전하고 있으며 문서 이해(document understanding) 분야에서 인상적인 성과를 보여주고 있습니다. 이 논문은 MLLMs 내의 인지(cognition)와 지각(perception) 간의 갈등을 정의하고 이를 해결하기 위한 새로운 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 인지 및 지각 지식 충돌(Cognition and Perception (C&P) knowledge conflicts)이라는 개념을 도입하고, 현재의 MLLMs에서 이러한 충돌을 체계적으로 평가합니다. 새로운 기법인 Multimodal Knowledge Consistency Fine-tuning을 통해 이 문제를 해결하는 방법을 설명하며, 이 방법은 인지 일관성 인지 및 지각 내용 일치 작업을 포함한 세 가지 작업으로 구성됩니다.

- **Performance Highlights**: 다양한 실험 결과에 따르면, 제안된 Multimodal Knowledge Consistency Fine-tuning 기법은 모든 테스트된 MLLMs에서 C&P 일관성을 34% 이상 개선시키며, 대다수 시나리오에서 인지 및 지각 작업의 성능을 향상시키는 것으로 나타났습니다.



### Training Data for Large Language Mod (https://arxiv.org/abs/2411.07715)
Comments:
          in Chinese language

- **What's New**: 2022년 ChatGPT의 출시에 힘입어 대규모 언어 모델이 주목받고 있습니다. ChatGPT는 이전 모델들보다 매개변수 수와 사전 학습 코퍼스의 규모에서 우수하며, 고품질 인간 주석 데이터로 미세 조정하여 혁신적인 성능을 달성했습니다.

- **Technical Details**: 논문은 대규모 언어 모델의 사전 훈련(pretraining)과 미세 조정(fine-tuning) 데이터를 다루며, 데이터의 규모, 수집 방법, 종류와 특성, 처리 워크플로우 등의 측면을 설명합니다. 특히 Transformer 아키텍처와 Self-Attention 메커니즘을 통해 언어 모델의 훈련 효율성이 크게 증가하였음을 강조합니다.

- **Performance Highlights**: 대규모의 고품질 데이터셋의 구축과 최적화는 인공지능 분야의 핵심적인 초점으로 자리 잡았습니다. 과거 데이터의 구축은 지속적인 과정이며, 향후 데이터의 다양성과 품질, 해석 가능성에 대한 중요성이 더욱 강조될 것으로 예상됩니다.



### New Emerged Security and Privacy of Pre-trained Model: a Survey and Outlook (https://arxiv.org/abs/2411.07691)
- **What's New**: 이 논문은 사전 훈련된 모델의 보안 위험에 대한 체계적인 조사를 수행하고, 새로운 공격 및 방어 방법에 대한 세분화된 분류 체계를 제안합니다. 이는 기존 문헌에서 자주 나타나지 않았던 새로운 연구 기회를 강조합니다.

- **Technical Details**: 사전 훈련된 모델은 대규모 데이터셋을 활용하여 GPT, BERT와 같은 고급 구조를 가진 모델로서, 모델의 입력과 가중치 접근에 따라 공격 및 방어 방법을 No-Change, Input-Change, Model-Change로 분류합니다. 다양한 공격 및 방어 사례에 대한 특성과 그 차이를 분석합니다.

- **Performance Highlights**: 이 surveyed 결과에 따르면, 새로운 공격 기법들이 등장함에 따라 기존의 안전 문제를 더욱 심화시키며, 큰 모델이 보안 및 개인 정보 문제를 어떻게 더 악화시키는지에 대한 심층적인 논의와 함께 향후 연구 방향에 대해 제안합니다.



### World Models: The Safety Perspectiv (https://arxiv.org/abs/2411.07690)
Comments:
          8 pages, 3 figures, accepted at the International Workshop on Dependability Modeling and Design (WDMD) during the IEEE International Symposium on Software Reliability Engineering (ISSRE)

- **What's New**: 이번 논문은 AI 에이전트 시스템 구축을 위한 핵심적인 기초로 여겨지는 월드 모델(World Model, WM)의 최신 기술 동향과 안전성(trustworthiness) 및 안전(safety) 측면에서의 연구 필요성을 다루고 있습니다.

- **Technical Details**: 이 논문에서는 대규모 언어 모델(LLM)의 발전이 월드 모델에 미치는 영향을 분석하고, 현재의 안전 관련 문제들을 조사합니다. 연구에서는 RNN, LSTM, Transformer와 같은 다양한 신경망 아키텍처가 월드 모델 구현에 사용되고 있으며, 특히 Transformers의 사용이 발전을 이끌고 있음을 강조합니다.

- **Performance Highlights**: 저자들은 자율주행, 로봇공학 등 안전이 중요한 분야에서 월드 모델의 안전성 문제를 검토하고, 더 나아가 신뢰할 수 있는 월드 모델 개발을 위한 차세대 연구 방향을 제안합니다. 이 논문은 GM 기반의 AI 에이전트 시스템의 더 안전하고 신뢰할 수 있는 발전에 기여할 것으로 기대됩니다.



### Exploring Multi-Agent Reinforcement Learning for Unrelated Parallel Machine Scheduling (https://arxiv.org/abs/2411.07634)
Comments:
          11 pages, 5 figures, 4 tables, article submitted to a journal

- **What's New**: 이번 연구는 Multi-Agent Reinforcement Learning (MARL) 접근법을 통해 Unrelated Parallel Machine Scheduling Problem (UPMS)을 다루며, 기존 Single-Agent 알고리즘과 비교하여 MARL의 효능을 입증합니다. 특히, MASKABLE PPO 알고리즘은 Single-Agent 시나리오에서 우수한 성능을 보여주고, Multi-Agent 환경에서는 협동 학습의 도전 과제를 선보입니다.

- **Technical Details**: 본 논문은 세팅 시간과 자원 활용을 고려하여 Unrelated Parallel Machine Scheduling 문제를 정의하고 최적 작업 스케줄링을 위한 MARL 환경을 설정합니다. 다양한 딥 뉴럴 네트워크 정책을 적용하여 Single-Agent 및 Multi-Agent 접근법의 성과를 비교 분석합니다.

- **Performance Highlights**: 실험 결과, Single-Agent 알고리즘은 제한된 시나리오에서 적절한 성능을 보였으나, Multi-Agent 접근법은 협동 학습의 도전 과제를 드러내면서도 확장 가능한 능력을 지니고 있음을 확인했습니다. 이 연구는 알고리즘의 정교함과 스케일러블한 특성을 균형있게 고려하여 지능형 스케줄링 솔루션을 위한 MARL 기술의 적용에 대한 통찰력을 제공합니다.



### Direct Preference Optimization Using Sparse Feature-Level Constraints (https://arxiv.org/abs/2411.07618)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 인간 선호와의 정렬을 효율적으로 달성하기 위한 Feature-level Constrained Preference Optimization(FPO)라는 새로운 방법을 제안합니다. FPO는 사전 훈련된 Sparse Autoencoders(SAEs)를 사용하여 훈련의 안정성을 보장하면서 정렬 프로세스를 단순화합니다.

- **Technical Details**: FPO는 base 모델 대신 feature-level 제약 조건을 도입하여 모델의 출력을 조정합니다. 이 방법은 Sparse Autoencoders(SAEs)를 통해 희소성을 강화하고, 이를 통해 메모리 사용량과 계산 복잡성을 효과적으로 줄입니다. FPO는 SimPO와 DPO의 개념을 조합하여 참조 모델 없이도 안정적인 성능을 유지합니다.

- **Performance Highlights**: FPO는 AlpacaEval-2와 Arena-Hard 벤치마크에서 5% 이상의 절대적인 승률 향상을 달성하였으며, TDPO에 비해 계산 비용을 17.6% 줄이는 성과를 보였습니다. 실험 결과는 FPO가 상태-of-더-아트 방법들에 비해 메모리 및 시간 복잡성에서 우수한 성능을 발휘함을 보여줍니다.



### A Comprehensive Survey of AI-Driven Advancements and Techniques in Automated Program Repair and Code Generation (https://arxiv.org/abs/2411.07586)
Comments:
          A survey of recent developments in AI-assisted automated program repair

- **What's New**: 이 논문에서는 최근 발표된 27개의 연구 논문을 검토하고, 이들을 자동화된 프로그램 수정(Automated Program Repair, APR)과 코드 생성(code generation)으로 나누어 분석합니다. LLM(대규모 언어 모델)의 통합이 코드 관련 작업에 미치는 긍정적인 영향을 강조하며, 자동 디버깅의 정확성과 효율성을 높이는 혁신적인 방법들도 소개됩니다.

- **Technical Details**: 이 조사는 LLM을 활용한 버그 탐지 및 수정 기법, 코드 생성 기법 등을 포함합니다. 특히, 정적 분석(Static Analysis), 동적 분석(Dynamic Analysis) 기법과 함께, 코드 생성을 위한 맥락 인식(Context-Aware) 수정을 살펴봅니다. 각 논문은 사용하는 LLM의 종류와 목표 언어, 버그 수리 및 코드 생성 접근 방식 등을 기준으로 분류됩니다.

- **Performance Highlights**: LLM을 활용한 코드 관련 작업은 프로그래밍을 자동화하고 버그를 발견하는 데 있어 훨씬 더 높은 품질과 속도를 제공합니다. 결과적으로 LLM 기반 모델들은 코드 완료 및 요약 작업에서 높아진 효율성과 정확성을 보여주며, 이는 전통적인 방법들과 비교할 때 엄청난 성과로 평가받고 있습니다.



### Improving Grapheme-to-Phoneme Conversion through In-Context Knowledge Retrieval with Large Language Models (https://arxiv.org/abs/2411.07563)
Comments:
          accepted by ISCSLP 2024

- **What's New**: 본 연구는 Grapheme-to-Phoneme (G2P) 변환 과정에서의 모호성 문제를 해결하기 위해 GPT-4의 In-Context Knowledge Retrieval (ICKR) 기능을 활용한 최초의 G2P 시스템을 제안합니다. 기존의 연구는 데이터 증강이나 구조적 수정에 국한되었던 반면, 본 연구는 LLM의 맥락적 능력을 활용하는 새로운 접근 방법을 제시합니다.

- **Technical Details**: 제안된 시스템은 GPT-4를 활용하여 입력 문장에서 음소 시퀀스를 생성하는 One-Shot Prompting 접근 방식을 채택하고 있습니다. 이 과정에서 동음이의어의 의미나 품사를 결정하고, 맥락적으로 가장 관련성 높은 음소 발음을 검색하여 G2P 변환을 수행합니다. 이 시스템은 주어진 문장에서 단어의 의미를 이해하고 소리 변환을 지원하기 위해 구조화된 딕셔너리를 사용합니다.

- **Performance Highlights**: Librig2p 데이터셋에서 ICKR을 활용한 시스템은 음소 오류율(PER)이 4.9%로 감소하였으며, 동음이의어 정확도는 95.7%에 달합니다. 이는 기존의 비맥락적 방법들과 비교할 때, 각각 2.0% 및 3.5%의 절대적 성능 향상을 이룬 것입니다.



### LLM App Squatting and Cloning (https://arxiv.org/abs/2411.07518)
- **What's New**: 이번 연구는 대규모 언어 모델(LLM) 앱 스토어에서의 앱 스쿼팅(app squatting)과 클로닝(app cloning)에 대한 최초의 대규모 분석을 제공합니다. 연구팀은 LLMappCrazy라는 툴을 개발하여 14개의 스쿼팅 생성 기법을 사용하여 분석하였고, 5,000개 이상의 스쿼팅 앱을 발견했습니다.

- **Technical Details**: LLMappCrazy는 Levenshtein distance와 BERT 기반의 의미 분석을 활용하여 앱의 기능 유사성을 분석함으로써 클로닝을 탐지합니다. 이 툴을 사용하여, 상위 1000개 앱 이름의 변형을 생성했으며, 6개의 주요 플랫폼에서 3,509개의 스쿼팅 앱과 9,575개의 클로닝 사례를 확인했습니다.

- **Performance Highlights**: 연구 결과에서 발견된 18.7%의 스쿼팅 앱과 4.9%의 클로닝 앱이 피싱(phishing), 악성 소프트웨어 배포, 가짜 콘텐츠 유포, 공격적인 광고 삽입과 같은 악의적인 행동을 보였습니다. 이로 인해 LLM 기반 애플리케이션 사용자가 직면하는 보안 위험이 더욱 심각해졌습니다.



### An Attack Traffic Identification Method Based on Temporal Spectrum (https://arxiv.org/abs/2411.07510)
Comments:
          20 pages, 7 figures, 7 tables, 8 formulas

- **What's New**: 이 논문은 기존 네트워크 공격 탐지 및 식별 모델의 부족한 견고성, 불안정한 특징, 데이터 노이즈 간섭 문제를 해결하기 위한 새로운 방법론을 제안합니다. 시간 스펙트럼 기반의 공격 트래픽 탐지 및 식별 방법이 중심 내용입니다.

- **Technical Details**: 제안된 방법은 슬라이딩 윈도우(sliding window)를 사용하여 트래픽 데이터를 세분화하고, 이에 따른 특징 시퀀스(feature sequence)와 레이블 시퀀스(label sequence)를 구성합니다. 이후, 스펙트럴 레이블 생성 방법인 SSPE(Spectral Sequence Perceptual Encoding)와 COAP(Content-Oriented Attack Pattern)를 적용하여 레이블 시퀀스를 스펙트럴 레이블로 변환하고, 특징 시퀀스를 시간적 특징(temporal features)으로 변환합니다.

- **Performance Highlights**: 실험 결과, SSPE 또는 COAP 방법으로 훈련된 모델은 기존 방법에 비해 식별 정확도가 10% 향상되었으며, 특히 노이즈 환경에서도 높은 견고성을 보였습니다.



### Evaluating Detection Thresholds: The Impact of False Positives and Negatives on Super-Resolution Ultrasound Localization Microscopy (https://arxiv.org/abs/2411.07426)
- **What's New**: 본 연구는 마이크로버블 (microbubble, MB) 검출 시 False Positives (FPs)와 False Negatives (FNs)가 초음파 로컬라이제이션 마이크로스코프(ULM) 이미징 품질에 미치는 영향을 체계적으로 분석하였습니다. FPs와 FNs의 다양한 비율을 통해 Structural Similarity Index (SSIM)와 Peak Signal-to-Noise Ratio (PSNR) 지표를 평가하였습니다.

- **Technical Details**: 연구는 IEEE UltraSR Challenge의 Ground Truth (GT) 데이터셋을 활용하여, 두 가지 중심 주파수(2.841MHz 및 7.24MHz)에서 시뮬레이션을 진행했습니다. 데이터셋에 FPs와 FNs를 주입하여 SR 맵의 품질을 분석하였으며, Gaussian Kernel Density Estimate (KDE)를 사용하여 마이크로버블의 밀도를 평가했습니다.

- **Performance Highlights**: Dense 지역에서는 SSIM 지표가 FP 및 FN 비율이 증가할 때도 상대적으로 안정적으로 유지되어 0.6 이상의 값을 나타냈습니다. 반면, Sparse 지역에서는 SSIM과 PSNR 값이 FP 및 FN 비율 증가에 따라 급격히 하락했으며, 특히 높은 FN 비율로 인해 구조적 유사성이 크게 영향을 받는 것으로 나타났습니다.



### Data-Centric Learning Framework for Real-Time Detection of Aiming Beam in Fluorescence Lifetime Imaging Guided Surgery (https://arxiv.org/abs/2411.07395)
- **What's New**: 본 연구는 섬유 기반 형광 수명 이미징(FLIm)을 이용한 실시간 수술 안내를 개선하기 위한 데이터 중심 접근법을 소개합니다.

- **Technical Details**: 본 방법론의 핵심 요소는 정확한 조준 빔(aiming beam) 탐지로, 이는 FLIm 측정을 수술 부위의 조직에 매핑하는 데 필수적입니다. 복잡하고 가변적인 수술 환경, 특히 경구 로봇 수술(TORS)에서의 도전이 존재합니다. 이러한 문제를 해결하기 위해 데이터 중심 훈련 전략을 사용하는 인스턴스 분할(instance segmentation) 모델을 개발하였습니다.

- **Performance Highlights**: 모델은 40개의 생체 내 수술 비디오 데이터셋에서 평가되었으며, 85%의 중위 탐지률(median detection rate)을 기록했습니다. 환자에서 시행된 TORS 절차 중에도 비슷한 탐지률을 유지하며, 약 24 프레임 초(FPS)의 계산 효율성이 실시간 수술 안내를 위해 충분하다는 것을 입증했습니다.



### Data-Driven Analysis of AI in Medical Device Software in China: Deep Learning and General AI Trends Based on Regulatory Data (https://arxiv.org/abs/2411.07378)
- **What's New**: 본 연구는 중국의 AI 의료기기 소프트웨어(AIMD)에 대한 최초의 광범위한 데이터 기반 탐색을 제공하며, 자동화된 규제 데이터 분석의 가능성을 보여줍니다.

- **Technical Details**: 이 연구는 NMPA(National Medical Products Administration)의 규제 데이터베이스에서 AI가 탑재된 의료기기(AIMD)를 자동으로 추출하고 분석하는 데이터 기반 접근 방식을 활용합니다. 400만 개 이상의 항목을 평가하여 2,174개의 MDSW 등록을 확인했습니다. 이 중 43개는 AI 기능이 있는 의료 소프트웨어입니다.

- **Performance Highlights**: AI 의료기기의 주요 사용 전문 분야는 호흡기(20.5%), 안과/내분비학(12.8%), 정형외과(10.3%)로 나타났습니다. 이 접근 방식은 데이터 추출 속도를 크게 향상시키며 비교 및 대조의 능력을 증가시킵니다.



### Artificial Intelligence Ecosystem for Automating Self-Directed Teaching (https://arxiv.org/abs/2411.07300)
Comments:
          13 pages, 15 figures, 12 references and 1 table

- **What's New**: 이 연구에서는 인공지능(AI) 기반 교육 개념이 도입되었습니다. 이는 개인 맞춤형 강의 제공 및 자동화된 교수 지원을 통해 자율 학습(self-directed learning)을 최적화하는 것을 목표로 합니다.

- **Technical Details**: 시스템은 세밀하게 조정된 AI 모델을 활용하여 맞춤형 로드맵(customized roadmaps), 자동 프레젠테이션 생성(automated presentation generation), 복잡한 개념 시각화를 위한 3D 모델링(three-dimensional modeling) 등으로 구성된 적응형 학습 환경을 제공합니다. 실시간 가상 지원(real-time virtual assistance)을 통합하여 학습자의 즉각적인 교육 필요를 충족하며 자율 학습을 촉진합니다.

- **Performance Highlights**: 이 연구는 자율 학습의 심리적 장점을 탐구하고 AI 자동화가 개인화된 콘텐츠 제공과 상호작용 지원 메커니즘을 통해 교육 결과를 향상시킬 수 있음을 보여줍니다. 예비 결과는 이 접근 방식이 다양한 학습 스타일을 지원하고 자율적이며 독립적인 학습 방법론을 강조함으로써 학생의 참여도와 지식 유지력을 강화함을 시사합니다.



### The Surprising Effectiveness of Test-Time Training for Abstract Reasoning (https://arxiv.org/abs/2411.07279)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 모델의 추론 과정에서 입력 데이터에 기반한 손실을 사용하여 모델 파라미터를 임시로 업데이트하는 테스트 타임 트레이닝(test-time training, TTT)이 모델의 추론 능력을 향상시키는 효과를 탐구했습니다. 특히 Abstraction and Reasoning Corpus (ARC)를 기준으로 효율적인 TTT의 세 가지 주요 요소를 식별했습니다.

- **Technical Details**: TTT의 성공을 위해 세 가지 주 구성 요소가 필요합니다: (1) 유사한 작업에서의 초기 파인튜닝, (2) 보조 작업 형식 및 데이터 증강(augmentation), (3) 인스턴스별 훈련. 이 방법은 LMs가 기존의 Fine-tuned 모델보다 최대 6배 더 높은 정확도를 달성하도록 돕습니다. 8B 파라미터 언어 모델에 TTT를 적용했을 때, ARC의 공개 검증 세트에서 53%의 정확도를 달성했습니다.

- **Performance Highlights**: 우리의 연구 결과, TTT를 통해 모델이 네오-심볼릭 접근 방식의 성능에 버금가는 결과를 도출할 수 있음을 보여주었습니다. 최근의 프로그램 생성 접근 방식과 조합하여, ARC 검증 세트에서 61.9%의 최첨단(public validation SoTA) 정확도를 달성했습니다. 이는 평균 인간 성능에 해당하는 수치입니다.



### Scaling Properties of Diffusion Models for Perceptual Tasks (https://arxiv.org/abs/2411.08034)
- **What's New**: 본 논문에서는 확산 모델( diffusion models)과 반복 계산(iterative computation)을 활용하여 생성 및 시각적 인식(visual perception) 작업에 대해 효과적으로 접근할 수 있는 새로운 패러다임을 제안합니다. 이전 연구들과 달리 깊이 추정(depth estimation), 광학 흐름(optical flow), 세분화(segmentation) 작업을 이미지-투-이미지 변환(image-to-image translation)으로 통합하였으며, 이러한 작업에 대한 훈련 및 테스트 시간 컴퓨팅(compute) 규모에 따른 변화도 분석하였습니다.

- **Technical Details**: 논문에서는 효율적인 확산 모델의 훈련 및 추론을 위해 다양한 기술을 제시합니다. 특히, 밀집 모델(dense models)과 전문가 혼합 모델(mixture-of-expert models)의 사전 훈련(pre-training)을 통해 확산 모델의 성능을 최적화합니다. 또한, 여러 테스트 시간 컴퓨팅 기술, 예를 들어 확산 단계(diffusion steps)를 증가시키거나, 테스트 시간 집계(test-time ensembling) 및 가변 소음 스케줄(noise variance schedules)을 통한 컴퓨팅 재구성을 실험합니다.

- **Performance Highlights**: 우리의 접근법을 통해, 기존의 최첨단 방법들에 비해 데이터와 컴퓨팅 자원을 현저히 절약하면서도 개선된 성능을 보였습니다. 다양한 벤치마크에서 최첨단 결과(state-of-the-art results)를 달성하였으며, 이를 통해 광범위한 시각적 인식 작업에 대한 일반화된 전문가 혼합 모델(generalist mixture-of-experts model)의 효율적인 훈련이 가능함을 보여줍니다.



### GaussianAnything: Interactive Point Cloud Latent Diffusion for 3D Generation (https://arxiv.org/abs/2411.08033)
Comments:
          project page: this https URL

- **What's New**: 본 연구는 기존의 3D 생성 방식에서의 입력 포맷, 잠재 공간 설계, 그리고 출력 표현의 문제점을 해결하기 위해 새로운 3D 생성 프레임워크인 GaussianAnything를 제안합니다. 이 프레임워크는 포인트 클라우드 구조의 잠재 공간을 사용하여 확장 가능하고 고품질의 3D 생성을 지원합니다.

- **Technical Details**: GaussianAnything는 Variational Autoencoder (VAE)를 기반으로 하며, 다중 시점 RGB-D(Depth) 이미지가 입력으로 사용됩니다. 이 프레임워크는 3D 모양 정보를 보존하는 독특한 잠재 공간 설계를 포함하고 있으며, 이로 인해 형상-질감(Shape-Texture) 분리를 개선합니다. 또한, Cascaded Latent Diffusion 모델을 도입하여 다양한 입력에 대해 조건부 3D 생성을 지원합니다.

- **Performance Highlights**: 실험 결과, GaussianAnything는 텍스트 및 이미지 조건부 3D 생성에서 기존 방법보다 뛰어난 성능을 보여주었으며, 여러 데이터셋에서 효과성을 입증하였습니다.



### LLMPhy: Complex Physical Reasoning Using Large Language Models and World Models (https://arxiv.org/abs/2411.08027)
- **What's New**: 본 논문에서는 로봇이 실제 세계에서 작업할 때 필요한 물리적 추론(physical reasoning) 능력을 향상시키기 위한 새로운 과제와 데이터셋인 TraySim을 제안합니다. 이 방법은 대형 언어 모델(LLMs)과 물리 엔진을 결합하여, 복잡한 물체의 동역학을 예측하는 것을 목표로 합니다.

- **Technical Details**: TraySim은 여러 물체가 놓인 쟁반에서 외부 충격에 대한 동역학을 예측하는 과제로 구성되어 있습니다. LLMPhy는 LLM의 프로그램 합성 능력을 활용하여 물리적 하이퍼파라미터를 추정하는 제로샷(Zero-shot) 최적화 프레임워크입니다. 이 프레임워크는 비분화가능한 시뮬레이터와 상호작용하여 물리적 특성을 상상하는 데 사용됩니다.

- **Performance Highlights**: LLMPhy는 TraySim 데이터셋에서 실험을 통해 기존의 블랙 박스 최적화 방법들에 비해 우수한 성능을 입증하였으며, 물리적 파라미터 추정의 정확도 또한 크게 향상되었습니다. 또한 LLMPhy의 성능은 OpenAI o1-preview와 같은 최신 LLM에서 더욱 두드러진 최적화 수렴 경향을 보여주었습니다.



### Language Models as Causal Effect Generators (https://arxiv.org/abs/2411.08019)
- **What's New**: 이 논문에서는 큰 언어 모델(LLM)에 기반한 데이터 생성 프레임워크를 제시하며, 특정한 인과 구조를 제어할 수 있는 방법을 설명합니다. 이 방법은 언어 모델과 방향 비순환 그래프(DAG)를 결합하여 순차적으로 구동되는 구조적 인과 모델(SD-SCM)을 생성합니다.

- **Technical Details**: SD-SCM은 사용자 정의 구조와 LLM 정의 구조 방정식을 포함하는 인과 모델을 나타냅니다. SD-SCM을 사용하면 관찰적, 개입적, 반사적 분포에서 샘플링할 수 있는 방법을 제공합니다. 이 모델을 통해 개별 반사적 데이터를 자동으로 생성할 수 있으며, 이는 기존의 기능적 관계를 수동으로 명시하지 않아도 가능하게 합니다. 코드와 데이터셋은 GitHub에서 접근할 수 있습니다.

- **Performance Highlights**: SD-SCM을 활용하여 생성된 데이터셋에 대해 여러 평균처리효과(ATE), 조건부 평균처리효과(CATE), 개별처리효과(ITE)를 추정하는 방법을 테스트했습니다. 이 절차는 LLM이 잘못된 정보, 차별, 또는 기타 바람직하지 않은 행동을 감지하는 데도 사용할 수 있어 LLM 감사를 위한 기반이 될 수 있습니다.



### Wavelet Latent Diffusion (Wala): Billion-Parameter 3D Generative Model with Compact Wavelet Encodings (https://arxiv.org/abs/2411.08017)
- **What's New**: 본 논문에서는 Wavelet Latent Diffusion (WaLa)라는 새로운 접근 방식을 소개하여 3D 형태를 wavelet 기반의 압축된 잠재 인코딩으로 인코딩합니다. 이를 통해 $256^3$의 signed distance field를 $12^3 	imes 4$의 잠재 그리드로 압축하여 2427배의 압축 비율을 달성했습니다.

- **Technical Details**: WaLa는 압축 과정에서 정보 손실 없이 wavelet 표현을 더욱 압축하여, diffusion 기반의 생성 모델을 효율적으로 확장할 수 있도록 합니다. 구체적으로는 convolution 기반의 VQ-VAE 모델을 사용하여 압축을 진행하며, 이는 약 10억 개의 매개변수를 포함하고 있습니다.

- **Performance Highlights**: WaLa는 고해상도 3D 생성에서 최첨단 성능을 보여주며, 다양한 입력 모달리티를 지원합니다. 모델의 생성 속도는 2~4초이며, 제어된 생성 또한 가능하여 복잡한 기하학, 신뢰할 수 있는 구조와 세밀한 토폴로지를 가진 3D 형태를 생성합니다.



### Investigating the Effectiveness of Explainability Methods in Parkinson's Detection from Speech (https://arxiv.org/abs/2411.08013)
Comments:
          The first two authors contributed equally to this research: author order is alphabetical

- **What's New**: 이 연구는 파킨슨병(Parkinson's Disease, PD) 진단을 위한 음성 기초 모델의 해석 가능성을 높이기 위한 설명 가능성 방법을 체계적으로 평가합니다. 이는 PD에 특화된 음성 특징을 식별하고, 임상 의사결정에 있어 정확하고 해석 가능한 모델의 개발을 지원하는 것을 목표로 합니다.

- **Technical Details**: 연구 방법론은 (i) 주류 해석 가능성 기술을 사용하여 속성 및 주목도 맵(attribution and saliency maps)을 획득하고, (ii) 이러한 맵의 충실성을 정량적으로 평가하며, (iii) 보조 분류기로부터 PD 탐지를 위한 주목도 맵이 전하는 정보를 평가하는 과정을 포함합니다. Saliency, SmoothGrad, Integrated Gradients, Guided GradCAM 등의 다양한 해석 가능성 기법이 사용되었습니다.

- **Performance Highlights**: 결과적으로, 설명은 분류기의 예측과 일치는 하나, 도메인 전문가들에게 진정으로 유용한 통찰을 제공하지 못하는 경우가 많았습니다. 이는 기존의 해석 가능성 방법들이 실제적인 사용에 필요한 수준의 해석 가능성을 결여하고 있음을 강조합니다.



### ExpressivityArena: Can LLMs Express Information Implicitly? (https://arxiv.org/abs/2411.08010)
Comments:
          8 pages, 22 figures

- **What's New**: 본 논문은 Large Language Models (LLMs)의 암시적 언어 신호 표현 능력을 평가하기 위한 Python 라이브러리인 ExpressivityArena를 제시합니다. LLM의 표현력을 평가할 수 있는 포괄적인 프레임워크를 제공하였으며, 창의적 및 논리적인 과제에 대한 실험을 통해 모델의 암시적 의사소통 능력을 탐색했습니다.

- **Technical Details**: ExpressivityArena 프레임워크를 사용하여 LLM의 표현력을 객관적으로 평가합니다. 다양한 LLM의 아웃풋을 평가하기 위한 자동 채점기를 설정하였으며, 이 채점기의 유효성을 인간 연구를 통해 검증했습니다. 실험을 통해 시가 생성, 코드 생성 등의 작업에서 모델의 표현력이 얼마나 다양한지를 살펴보았습니다.

- **Performance Highlights**: 모델은 감정 표현 시 시간이 지남에 따라 표현력이 줄어드는 경향을 보였으나, 직업 표현에서는 대화가 진행될수록 표현력이 증가하는 결과를 보여주었습니다. 이는 LLM이 표현 예정이다 다양한 경우에 따라 제한이 있음을 시사하며, 향후 발전 방향에 중요한 통찰을 제공합니다.



### Derivational Morphology Reveals Analogical Generalization in Large Language Models (https://arxiv.org/abs/2411.07990)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 언어적 일반화(linguistic generalization) 메커니즘을 분석했다. 특히, 규칙 기반(rule-based) 접근 방식과 유사성 기반(analogical) 접근 방식의 둘 다 설명할 수 있는 다양한 문법 현상 중 영어 형용사의 명사화(adjective nominalization)를 조사했다.

- **Technical Details**: 본 연구는 GPT-J 모델을 중심으로 하여, 규칙 기반 모델인 Minimal Generalization Learner (MGL)와 유사성 모델인 Generalized Context Model (GCM)을 비교 분석하였다. 이 과정에서는 4가지 형용사 클래스에 대해 -ity와 -ness의 선호도를 분석하며, 각 모델의 예측 결과를 검토하였다. 연구는 모델이 유형 빈도(type frequency)와 토큰 빈도(token frequency) 의 영향을 어떻게 반영하는지를 평가하였다.

- **Performance Highlights**: 연구 결과, GPT-J는 정규적인 명사화 패턴에 대해서는 MGL과 GCM 모두와 비슷한 예측을 수행한다. 하지만, 다양한 명사화 패턴을 가진 형용사에서는 GCM의 유사성 모델이 GPT-J의 행동을 더 잘 설명한다는 사실이 드러났다. 이는 GPT-J가 규칙이 아닌 유사한 예를 바탕으로 언어적 일반화를 이루고 있음을 시사한다.



### Exact, Tractable Gauss-Newton Optimization in Deep Reversible Architectures Reveal Poor Generalization (https://arxiv.org/abs/2411.07979)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 본 논문에서는 딥 리버서블 아키텍처에서 정확한 Gauss-Newton (GN) 업데이트의 가능성을 처음으로 보여줍니다. 이 연구는 실제 데이터셋에서 GN 최적화 기법의 훈련 및 일반화 특성을 조사합니다.

- **Technical Details**: 딥 리버서블 네트워크에서 Jacobian의 일반화된 역을 분석적으로 유도하였으며, 이를 통해 GN 업데이트를 정확하고 신속하게 수행할 수 있는 방법을 제시합니다. 또한, 이 모델은 NTK(신경탄젠트핵)가 훈련 중 비특이적(non-singular)인 경우에 해당합니다.

- **Performance Highlights**: GN 최적화 기법은 훈련 손실에서 과적합을 초래하며, 미니 배치 설정에서 일반화 능력이 부족한 현상을 보였습니다. 또한, 연구 결과는 GN 업데이트가 '게으른' 레짐(lazy regime)으로 남아있어 모델의 표현력이 초기 상태와 유사하게 유지된다는 점을 강조합니다.



### DINO-LG: A Task-Specific DINO Model for Coronary Calcium Scoring (https://arxiv.org/abs/2411.07976)
Comments:
          Developed by Center for Applied Artificial Intelligence (CAAI), University of Kentucky

- **What's New**: 이 논문에서는 기존의 딥러닝 기반 CAC 스코어링 시스템을 개선하기 위해 self-supervised learning (SSL) 기법을 사용하는 DINO 모델을 도입해, 주석 데이터가 부족한 문제를 해결하고 CAC 세분화 및 스코어링 성능을 향상시키고자 합니다. DINO-LG 모델을 통해 CT 이미지에서 석회화된 영역을 자동으로 구분합니다.

- **Technical Details**: 이 연구에서는 DINO(무주석 자기 증류) 기법을 적용하여 CT 스캔에서 CAC 영역을 분리하는 새로운 self-supervised 학습 체계를 제안합니다. DINO 모델은 석회화된 영역에 주목하여 주석된 데이터를 거의 필요로 하지 않으며, Label-Guided DINO(dino-LG) 기법을 통해 보다 정교한 특정 특성을 캡처합니다. 기본 U-NET 아키텍처를 사용하여 CAC 세분화 작업을 수행합니다.

- **Performance Highlights**: DINO-LG 모델은 석회화된 CT 슬라이스를 구분하는 작업에서 기존 DINO 모델보다 57% 향상된 분류 성능을 보여주었으며, CAC 세분화 성능을 약 10% 향상시켰고, CAC 스코어링 정확성을 획기적으로 증가시켰습니다.



### JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation (https://arxiv.org/abs/2411.07975)
- **What's New**: 이번 연구에서는 이미지 이해와 생성을 통합한 강력한 모델인 JanusFlow를 소개합니다. JanusFlow는 자가 회귀 언어 모델(autoregressive language models)과 정제된 흐름(rectified flow)을 결합하는 미니멀리스트 아키텍처를 통해 기존의 복잡한 모델 구조를 간소화했습니다.

- **Technical Details**: JanusFlow는 독립적인 이해 및 생성 인코더(encoders)를 유지하고, 통합 훈련 중에 이들의 표현을 정렬(aligning)하는 두 가지 주요 전략을 채택하여 성능을 최적화합니다. 이를 통해 이미지 생성 및 이해 작업 간의 간섭을 방지하고 의미적 일관성을 강화합니다.

- **Performance Highlights**: JanusFlow는 텍스트-이미지 생성 및 다중 모드 이해 벤치마크에서 기존의 통합 모델들을 능가하는 성능을 보입니다. 특히, MJHQ FID에서 9.51, GenEval에서 0.63, DPG-Bench에서 80.09%를 기록하며, LLaVA-v1.5 및 Qwen-VL-Chat 같은 전문 모델을 초월합니다. 또한, JanusFlow는 단 1.3B 파라미터로 이 성능을 달성합니다.



### DuoLift-GAN:Reconstructing CT from Single-view and Biplanar X-Rays with Generative Adversarial Networks (https://arxiv.org/abs/2411.07941)
- **What's New**: 본 논문에서는 2D 이미지를 독립적으로 3D 표현으로 높이는 듀얼 브랜치 구조인 DuoLift Generative Adversarial Networks (DuoLift-GAN)을 제안합니다. 이 아키텍처는 시각적 사실성을 우선시하는 기존 모델들과는 달리 구조적 정확성을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: DuoLift-GAN은 2D 이미지를 접합하면서 3D 피쳐 맵으로 변환하기 위해 Masked Loss Function을 사용합니다. 이 모델은 X-ray 이미지를 이용하여 정밀한 3D 볼륨을 재구성하도록 설계되어 있으며, 이를 통해 공간적 관계와 일관성을 보존하며 더 정확한 3D 피쳐 맵을 생성합니다.

- **Performance Highlights**: DuoLift-GAN은 기존의 방법들에 비해 재구성 정확성을 크게 향상시켰으며, 시각적 사실성에서도 우수한 성능을 보입니다. 논문에서는 다양한 평가 지표를 통해 DuoLift-GAN의 효과를 입증하고, LIDC-IDRI 데이터셋을 사용하여 기존의 최신 기법들과 비교한 결과를 제공합니다.



### Doubly Mild Generalization for Offline Reinforcement Learning (https://arxiv.org/abs/2411.07934)
Comments:
          Accepted to NeurIPS 2024. arXiv admin note: substantial text overlap with arXiv:2410.19400

- **What's New**: 이 논문은 오프라인 강화 학습(Offline Reinforcement Learning)에서 약간의 일반화(mild generalization)를 통해 성능을 향상시킬 수 있는 가능성을 탐구합니다. 이를 위해 새로운 방법인 Doubly Mild Generalization (DMG)을 제안합니다.

- **Technical Details**: DMG는 (i) mild action generalization과 (ii) mild generalization propagation으로 구성됩니다. 첫 번째는 데이터셋의 근처에서 Q 값을 최대화하기 위해 행동을 선택하는 것을 의미합니다. 두 번째는 RL 학습 신호의 전파를 방해하지 않으면서 일반화 전파를 줄이는 방법입니다.

- **Performance Highlights**: DMG는 Gym-MuJoCo locomotion 작업과 AntMaze 작업에서 최첨단 성능을 달성하였으며, 오프라인에서 온라인 학습으로 매끄럽게 전환할 수 있는 유연성을 가지고 있습니다.



### INTRABENCH: Interactive Radiological Benchmark (https://arxiv.org/abs/2411.07885)
Comments:
          Undergoing Peer-Review

- **What's New**: IntRaBench는 3D 의료 영상에서의 인터랙티브 세분화 방법을 효과적으로 평가할 수 있는 새로운 벤치마크 프레임워크입니다. 이 프레임워크는 다양한 데이터셋과 세분화 모델을 포함하며, 임상에서의 실제 사용을 고려하여 개발되었습니다.

- **Technical Details**: IntRaBench는 2D 및 3D 인터랙티브 세분화 방법의 공정하고 재현 가능한 평가를 지원합니다. 특정한 프로트핑(prompting) 및 수정(refinement) 전략을 통해 2D 모델에서도 사용자 상호작용을 간소화하고, 대시(board)에서 인간의 노력을 최소화합니다. 이 벤치마크는 10개의 데이터셋과 7개의 모델을 포함하며, 모두 공개되어 있어 사용자가 쉽게 다운로드 및 전처리할 수 있습니다.

- **Performance Highlights**: IntRaBench는 최초로 2D와 3D 인터랙티브 세분화 방법 간의 공정한 비교를 가능하게 합니다. 연구자들은 이 프레임워크를 이용하여 새로운 방법을 평가하고, 지속적이고 투명한 세분화 모델 평가를 통해 3D 의료 영상 세분화 분야에서의 진전을 추적할 수 있습니다.



### Diverse capability and scaling of diffusion and auto-regressive models when learning abstract rules (https://arxiv.org/abs/2411.07873)
Comments:
          12 pages, 5 figures. Accepted to NeurIPS2024 Workshop on System 2 Reasoning At Scale as long paper

- **What's New**: 본 논문에서는 현대 생성 모델이 유한 샘플로부터 기본 규칙을 학습하고 조건적 샘플링을 통해 추론할 수 있는지를 조사합니다. 이를 위해 Raven's Progressive Matrices 작업에서 영감을 받아 GenRAVEN 데이터세트를 설계하였습니다.

- **Technical Details**: GenRAVEN 데이터세트는 40개의 관계 규칙을 기반으로 하며, 각 샘플은 3행으로 구성됩니다. 생성 모델은 diffusion 모델(EDM, DiT, SiT)과 autoregressive 모델(GPT2, Mamba) 두 그룹으로 훈련되었습니다. 다양한 데이터 스케일에서의 성능을 비교하였으며, 각 모델의 샘플 생성 능력을 평가했습니다.

- **Performance Highlights**: diffusion 모델은 기본적인 생성에서 뛰어난 성능을 보여주었고, 새로운 샘플을 더 일관되게 생성할 수 있었습니다. 반면, autoregressive 모델은 누락된 패널을 규칙 일관성 있게 완성하는 데 강점을 보였으나, 기본 생성에서는 일관성이 떨어졌습니다. 데이터 규모에 따른 다양한 성능 변화를 관찰하였고, 앞으로의 연구 방향에 대한 통찰을 제공합니다.



### Trustful LLMs: Customizing and Grounding Text Generation with Knowledge Bases and Dual Decoders (https://arxiv.org/abs/2411.07870)
- **What's New**: 이번 연구에서는 LLMs (Large Language Models)의 Hallucination (환각) 문제를 해결하기 위해 지식 트리플(knowledge triplets)을 활용하는 후처리 알고리즘과 RAG (Retrieval-Augmented Generation) 문맥을 융합한 이중 디코더(Dual-Decoder) 모델을 제안합니다.

- **Technical Details**: 제안된 후처리 알고리즘과 이중 디코더 모델은 지식 그래프(knowledge graph)로부터 추출한 지식 트리플을 기반으로 Hallucinations를 수정하며, RAG의 맥락을 통합하여 생성 과정을 안내합니다. 알고리즘은 생성된 텍스트의 진위를 검증하고, 출력 길이에 대한 제약을 완화하며, 관련된 맥락과 프롬프트에만 초점을 맞춥니다.

- **Performance Highlights**: 이 방법론은 Microsoft 제품 문의를 지원하는 실제 상용 애플리케이션 시나리오에 적용되며, 고객에 대한 서비스의 완성도와 정확성을 높이는 데 기여합니다. 또한, 다양한 LLM 버전(예: ChatGPT, LLama-3)에서 Fluency (유창성) 문제를 보완하여, 신뢰성을 높이고 사실성을 강화하는 것으로 나타났습니다.



### Tucano: Advancing Neural Text Generation for Portugues (https://arxiv.org/abs/2411.07854)
- **What's New**: 본 연구는 포르투갈어의 신경 텍스트 생성을 위한 새로운 자원을 소개하고 있습니다. GigaVerbo라는 대규모의 포르투갈어 데이터셋을 구축하여, 이를 활용해 Tucano라는 디코더-트랜스포머 모델을 훈련하였습니다.

- **Technical Details**: GigaVerbo는 2000억 개의 토큰으로 구성된 중복 제거된 포르투갈어 텍스트 코퍼스의 집합으로, 여기서 Tucano 모델을 훈련시켰습니다. 이 모델은 여러 포르투갈어 벤치마크에서 다른 포르투갈어 및 다국어 모델들과 동등하거나 우수한 성능을 보였습니다.

- **Performance Highlights**: Tucano 모델은 기존의 포르투갈어 NLP 커뮤니티에서 사용되는 벤치마크와의 성능 평가에서도 좋은 성과를 냈으며, 특히 기존 모델과의 성능 상관관계의 한계를 드러냈습니다.



### IAE: Irony-based Adversarial Examples for Sentiment Analysis Systems (https://arxiv.org/abs/2411.07850)
- **What's New**: 이 논문에서는 Irony 기반의 적대적 예제(Irony-based Adversarial Examples, IAE)를 제안합니다. 이는 본래의 문장을 아이러니한 문장으로 변환하여 적대적 텍스트를 생성하는 방법입니다. 기존의 아이러니 말뭉치에 의존하지 않기 때문에 다양한 자연어 처리(NLP) 작업에 적합한 도구로 자리 잡을 수 있습니다.

- **Technical Details**: IAE 방법은 평가 단어를 정확히 찾아 적절한 복합어로 대체하며, 아이러니한 요소를 포함하여 텍스트의 의미 일관성을 유지해야 한다는 점에서 전문적입니다. 텍스트의 직설적인 의미와는 반대되는 상황적 의미를 파악해야 하며, 세 가지 주요 도전 과제가 있습니다: 1) 평가 단어의 위치 찾기, 2) 적절한 복합어로 대체하기, 3) 필요할 때 적절한 아이러니 평가로 텍스트 확장하기.

- **Performance Highlights**: 우리는 여러 최첨단 딥러닝 모델이 감정 분석 작업에서 IAE 공격에 노출되었을 때 성능이 현격히 저하된다는 것을 보여주었습니다. 이는 현재 NLP 시스템이 아이러니를 통한 적대적 조작에 취약함을 강조합니다. 반면 인간은 텍스트에서 아이러니의 영향에 덜 민감함을 보여주었습니다.



### Ethical Concern Identification in NLP: A Corpus of ACL Anthology Ethics Statements (https://arxiv.org/abs/2411.07845)
- **What's New**: 본 논문은 LLM(대형 언어 모델) 연구자들이 어떤 윤리적 우려를 가지고 있는지를 조사하고, 그런 우려를 자동으로 식별하기 위한 EthiCon이라는 데이터셋을 구축하였습니다. 이 데이터셋은 1,580개의 윤리적 우려 진술문으로 구성되어 있으며, NLP(자연어 처리) 커뮤니티와 일반 대중 간의 우려를 비교하는 것도 목적입니다.

- **Technical Details**: EthiCon은 ACL(Association for Computational Linguistics) 자료에서 추출된 윤리적 진술문으로 이루어진 주석이 달린 코퍼스입니다. 저자는 4,691개 및 3,357개의 논문에서 윤리적 진술문을 스크래핑하여 1,580개의 진술을 수집하고, 이를 윤리적 우려의 5개 카테고리로 분류하였습니다. LLM 실험을 통해 자동화된 윤리적 우려식별 방법도 제시하였습니다.

- **Performance Highlights**: EthiCon 데이터셋을 활용하여 LLM 모델들이 윤리적 우려를 식별하는 데에 높은 정확도를 나타냈습니다. 주목할만한 점은 생성 작업에서 모델들이 인간 주석가보다 더 많은 우려를 식별했으나, 비논리적 생성이나 허위 정보는 발견되지 않았다는 것입니다. 이는 LLMs이 윤리적 우려 모니터링을 위한 도구로서 유용성 가능성을 시사합니다.



### Chain Association-based Attacking and Shielding Natural Language Processing Systems (https://arxiv.org/abs/2411.07843)
- **What's New**: 이번 연구에서는 자연어 처리(NLP) 시스템에 대한 새로운 공격 방법으로 체인 연관 기반의 적대적 공격을 제안합니다. 이 방법은 사람과 기계 간의 이해 격차를 활용하며, 특히 중국어 캐릭터에 대한 체인 연관 그래프를 생성하여 잠재적인 적대적 예시를 검색하는 데 사용됩니다.

- **Technical Details**: 체인 연관 그래프는 중국어 캐릭터 구축을 위한 연관 패러다임에 기반하여 생성되며, 최적의 적대적 예시를 검색하기 위해 이산 입자 군집 최적화(Discrete Particle Swarm Optimization) 알고리즘을 도입합니다. 이 연구에서는 체인 연관이 적대적 공격에 적용된 첫 번째 사례로, 적대적 훈련(Adversarial Training) 및 연관 그래프 기반 복구를 통해 시스템을 보호하는 두 가지 방법도 탐색합니다.

- **Performance Highlights**: 고급 NLP 모델과 애플리케이션이 체인 연관 기반 공격에 매우 취약함을 보여줍니다. 실험 결과, 사람들이 변형된 텍스트를 이해하는 능력이 뛰어난 반면, NLP 시스템은 이 공격에 의해 비정확한 결과를 초래할 수 있음을 확인했습니다.



### Efficient Federated Finetuning of Tiny Transformers with Resource-Constrained Devices (https://arxiv.org/abs/2411.07826)
- **What's New**: 이번 연구에서는 미리 훈련된 딥러닝 모델을 활용하여 자원이 제한된 크로스 디바이스 페더레이티드 러닝(Federated Learning, FL)을 위한 새로운 레이어 파인튜닝 기법을 제안하였습니다. LoRA 기법이 여전히 매개변수 효율적이지만 메모리와 FLOPs 면에서 비효율적임을 관찰하고, 그에 따라 더 나은 성능을 제공하는 방법을 개발했습니다.

- **Technical Details**: 제안된 기법은 크로스 디바이스 FL에서 미리 훈련된 신경망을 사용하며 주어진 자원 제약을 준수합니다. 레이어 파인튜닝 기법을 통해 메모리 사용을 최적화하고, LoRA와 비교하여 계산 비용을 감소시킵니다. 실험에서는 소형 모델을 사용하여 동질적 및 이질적 환경에서 성능을 평가하였으며, 기존의 FL 방법들보다 우수한 결과를 기록했습니다.

- **Performance Highlights**: 제안된 레이어 파인튜닝 기법은 LoRA 기반 기법 및 최신 FL 방법들과 비교하여 동질적 및 이질적 계산 및 메모리 제약을 처리하는 데 있어 월등한 성능을 나타냈습니다. 또한, 통신이 제한적인 환경에서도 높은 정확도를 달성하여 FL 교육 효율성을 크게 향상시켰습니다.



### InvisMark: Invisible and Robust Watermarking for AI-generated Image Provenanc (https://arxiv.org/abs/2411.07795)
- **What's New**: InvisMark라는 새로운 워터마킹 기술을 소개합니다. 이 기술은 고해상도 AI 생성 이미지에 맞춰 설계되었으며, 불감정적이며 강력한 워터마크를 삽입할 수 있습니다.

- **Technical Details**: InvisMark는 고급 신경망 아키텍처와 훈련 전략을 이용해 PSNR이 약 51, SSIM이 0.998에 이르는 성능을 자랑합니다. 256비트의 워터마크를 제공하며, 다양한 이미지 조작에서도 97% 이상의 비트 정확성을 유지합니다. 이는 UUID와 오류 수정 코드를 결합하여 강력한 디코딩 성공률을 달성합니다.

- **Performance Highlights**: InvisMark는 기존의 방법들보다 불감정성과 강건성이 우수하여 AI 생성 이미지 및 비 AI 생성 이미지 데이터셋 모두에서 최첨단 성능을 보입니다. 이를 통해 더 큰 페이로드를 지원하며, 실사 이미지 품질을 충족시키는 동시에 왜곡에 대한 강한 저항성을 발휘합니다.



### Feature Fusion Transferability Aware Transformer for Unsupervised Domain Adaptation (https://arxiv.org/abs/2411.07794)
Comments:
          IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025

- **What's New**: 이 논문에서는 비전 트랜스포머(Vision Transformer, ViT)를 이용한 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)을 개선하기 위해 새로운 특징 융합 전이 가능성 인식 트랜스포머(Feature Fusion Transferability Aware Transformer, FFTAT)를 제안합니다. 주요 혁신으로는 패치 전이 가능성 평가를 위한 패치 판별기와 임베딩을 융합하는 기법을 도입하여 모델의 일반화를 향상시킵니다.

- **Technical Details**: FFTAT의 두 가지 주요 구성 요소는 (1) 전이 가능성 그래프 기반 자기 주의 메커니즘(transferability graph-guided self-attention, TG-SA)과 (2) 잠재 공간에서 임베딩 정보를 융합하는 특징 융합(feature fusion) 기술입니다. TG-SA는 고전이 가능성 패치 간의 정보를 강조하고 저전이 가능성 패치 간의 정보를 억제함으로써 도메인 불변 특징을 중심으로 학습할 수 있게 합니다.

- **Performance Highlights**: FFTAT는 기존의 다양한 UDA 벤치마크에서 실험을 통해 UDA 성능을 크게 향상시키며 최신 성능(state-of-the-art, SOTA) 결과를 달성합니다.



### RedCode: Risky Code Execution and Generation Benchmark for Code Agents (https://arxiv.org/abs/2411.07781)
Comments:
          Accepted by NeurIPS 2024 Datasets and Benchmarks Track

- **What's New**: 코드 에이전트(Code Agents)의 안전성에 관한 새로운 벤치마크인 RedCode를 제안합니다. 이 벤치마크는 위험한 코드(execution and generation) 실행 및 생성을 평가하기 위한 것입니다.

- **Technical Details**: RedCode는 두 가지 주요 구성 요소로 구성되어 있습니다: (1) RedCode-Exec는 위험한 코드 실행을 유도할 수 있는 도전적인 프롬프트(prompts)를 제공하여 코드 에이전트들이 안전하지 않은 코드를 인식하고 처리하는 능력을 평가합니다. 이 시스템은 Python 및 Bash 태스크를 포함하여 총 4,050개의 위험한 테스트 케이스를 제공합니다. (2) RedCode-Gen은 함수 시그니처(function signatures) 및 문서 문자열(docstrings)을 입력으로 사용하여 코드 에이전트가 지시 사항에 따라 해로운 코드를 생성할지 평가하는 160개의 프롬프트를 제공합니다.

- **Performance Highlights**: RedCode-Exec 평가 결과, 에이전트는 운영 체제에서 위험한 작업을 실행하는 것을 거부할 가능성이 높지만, 기술적으로 결함이 있는 코드를 실행하는 것은 덜 거부하는 경향을 보였습니다. 자연어로 설명된 위험한 작업이 코드 형식보다 거부율이 낮았고, RedCode-Gen 평가 결과, 더 강력한 기본 모델(base models)과 전반적인 코딩 능력이 우수한 에이전트(예: GPT4)가 더 정교하고 효과적인 해로운 소프트웨어를 생성하는 경향이 있었습니다.



### Likelihood as a Performance Gauge for Retrieval-Augmented Generation (https://arxiv.org/abs/2411.07773)
Comments:
          Under review at NAACL 2025. Code is available at this https URL

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LMs)에서의 retrieval-augmented generation(RAG) 과정 중 문서의 순서가 결과에 미치는 영향을 분석합니다. 특히, 질문의 likelihood가 모델 성능을 예측할 수 있는 지표가 될 수 있음을 보여주며, 이를 바탕으로 더 나은 성능을 위한 프롬프트 최적화 방법을 제안합니다.

- **Technical Details**: 본 연구는 NQ-Open과 ELI5 두 가지 질문-답변 데이터셋에서 다양한 최첨단 LMs(LLaMA-2, LLaMA-3, LLaMA-3.1, Mistral-v0.3, MPT)를 활용하여 질문의 likelihood와 답변 정확도 간의 상관관계를 조사하였습니다. 입력 프롬프트의 세 가지 구성 요소인 컨텍스트, 질문, 금지 답변(gold answer)의 log-likelihood를 분석하여, 높은 log-likelihood를 가진 질문에 대해 LMs가 더 나은 답변을 할 수 있음을 발견했습니다.

- **Performance Highlights**: 제안된 방법은 질문 likelihood를 기반으로 한 프롬프트 최적화를 통해 두 데이터셋에서 답변 정확도를 개선했습니다. 효율적인 계산 방식이 특징적이며, LM 응답을 생성하기 위해 여러 번 실행할 필요가 적어 계산 비용을 절감할 수 있습니다.



### Automatic Album Sequencing (https://arxiv.org/abs/2411.07772)
Comments:
          presented as a late breaking demo in the 25th International Society for Music Information Retrieval Conference; 3 pages in main text, 3 figures in main text; source code available at this https URL

- **What's New**: 앨범 시퀀싱(album sequencing) 과정에서 사용자 친화적인 웹 기반 도구가 도입되었습니다. 이 도구를 통해 비전문가도 쉽게 음악 트랙을 업로드하고, 한 번의 클릭으로 시퀀싱 기법을 실행하여 결과를 시각화할 수 있습니다.

- **Technical Details**: 이 연구에서는 Transformer를 기반으로 한 새로운 앨범 시퀀싱 방법을 소개합니다. 이는 이전 연구의 복잡한 파이프라인을 단순화하여 하나의 모델로 대체하였으며, 알고리즘은 FMA 데이터셋을 기반으로 하여 두 층의 완전 연결 신경망과 두 층의 인코더-디코더 Transformer 모델을 사용합니다.

- **Performance Highlights**: 새로운 방법은 무작위 베이스라인(random baseline)보다 뛰어난 성능을 보였지만, 이전의 내러티브 본질(narrative essence) 접근법에는 미치지 못했습니다. 이 연구의 모든 구현은 공개적으로 제공되며, 비전문가도 사용할 수 있는 사용자 인터페이스가 제공됩니다.



### Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows (https://arxiv.org/abs/2411.07763)
- **What's New**: Spider 2.0은 기업 데이터베이스의 실제 사용 사례에서 파생된 632개의 텍스트-투-SQL(workflow) 문제를 포함하는 새로운 평가 프레임워크입니다.

- **Technical Details**: Spider 2.0은 BigQuery와 Snowflake와 같은 클라우드 또는 로컬 데이터베이스 시스템에서 1,000개 이상의 컬럼을 포함할 수 있는 데이터베이스로 구성되며, 문제 해결에는 데이터베이스 메타데이터, 방언(dialect) 문서 및 프로젝트 코드베이스와의 상호 작용이 필요합니다.

- **Performance Highlights**: Spider 2.0에서 코드 에이전트 프레임워크는 17.0%의 작업을 성공적으로 해결하여, Spider 1.0의 91.2% 및 BIRD의 73.0%와 비교됩니다. 이는 실제 기업 사용에서 충분한 성능을 달성하기 위해 언어 모델의 상당한 개선이 필요함을 보여줍니다.



### ASER: Activation Smoothing and Error Reconstruction for Large Language Model Quantization (https://arxiv.org/abs/2411.07762)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 양자화(Quantization)를 다루며, 특히 Post-Training Quantization 중의 오류 분포를 분석합니다. 새로운 알고리즘 ASER(Activation Smoothing and Error Reconstruction)가 소개되어, 양자화 오류를 보정하는 방법을 제시합니다.

- **Technical Details**: ASER는 (1) 오류 재구성(Error Reconstruction)과 (2) 활성화 부드럽게 하기(Activation Smoothing)를 포함합니다. 오류 재구성에서는 LoRA 스타일 행렬을 이용한 저계수 보정을, 활성화 부드럽게 하기는 이상치 분석을 통해 수행됩니다.

- **Performance Highlights**: ASER는 일반적인 LLM을 저비트로 양자화하는 데 효과적이며, W4A8 스킴에서 정확도를 유지할 수 있습니다. 실험 결과 ASER는 최첨단 양자화 알고리즘들과 비교해 경쟁력이 있음이 입증되었습니다.



### Navigation with QPHIL: Quantizing Planner for Hierarchical Implicit Q-Learning (https://arxiv.org/abs/2411.07760)
Comments:
          Under review. Code will be released upon acceptance

- **What's New**: 이번 연구에서는 Offline Reinforcement Learning (RL)의 새로운 접근 방식인 QPHIL을 제안합니다. QPHIL은 사전 기록된 시연을 활용하여 상태 공간의 이산 표현 및 시간적으로 일관된 표현을 학습합니다.

- **Technical Details**: QPHIL은 계층적 구조를 가진 목표 조건 강화 학습 방법입니다. 학습된 양자화기(quantizer)를 통해 공간을 분할하고, 이로 인해 하위 목표(subgoal) 계획이 단순화됩니다. 이 접근 방식은 복잡한 경로 추적(low-level path following)과 경로 계획(high-level path planning)을 분리하여 효율적인 환경 내 탐색이 가능하게 합니다.

- **Performance Highlights**: QPHIL은 복잡한 장거리 탐색 환경에서 최첨단 성능을 달성하였으며, 인간-영감을 받은 방식으로 하위 정책을 최종 목표로 유도합니다. 이 방법은 장기 탐색 벤치마크에서 뛰어난 결과를 보여주며, 오프라인 RL 설정에서의 계획 및 디지털화의 향후 조치를 위한 유망한 방향을 제시합니다.



### Optimizing Traffic Signal Control using High-Dimensional State Representation and Efficient Deep Reinforcement Learning (https://arxiv.org/abs/2411.07759)
Comments:
          Under Review

- **What's New**: 이 연구는 RL 기반의 교통 신호 제어(Traffic Signal Control, TSC)에서 고차원(state representation) 상태 표현이 성능 향상에 미치는 영향을 다룹니다. 기존 연구들은 고차원 표현이 TSC 성능에 도움이 되지 않는다고 주장했으나, 본 연구는 실험을 통해 평균 대기 시간을 최대 17.9% 줄일 수 있다는 것을 보여줍니다.

- **Technical Details**: 고차원 상태 표현은 차량과 인프라 간의 통신(vehicle-to-infrastructure, V2I)을 통해 얻을 수 있으며, 이는 TSC에 대한 채택을 장려합니다. 또한, 상태의 크기가 커짐에 따라 계산 효율성(computational efficiency)을 요구하며, 모델 절단(pruning)을 통한 모델 압축을 탐색합니다.

- **Performance Highlights**: 실험적으로 고차원 상태 표현을 활용함으로써, TSC의 평균 대기 시간을 최대 17.9% 개선하는 성과를 달성했습니다.



### SAV-SE: Scene-aware Audio-Visual Speech Enhancement with Selective State Space Mod (https://arxiv.org/abs/2411.07751)
- **What's New**: 본 논문에서는 새로운 오디오-비주얼 음성 향상(Speech Enhancement, SE) 작업인 SAV-SE를 제안합니다. 기존 연구가 주로 얼굴 및 입술 운동에 집중되었던 반면, 본 연구는 환경의 맥락적 시각 정보(visual contextual cues)를 활용하여 음성 향상 성능을 개선하는 방법을 제시합니다.

- **Technical Details**: VC-S$^2$E 방법은 Conformer 및 Mamba 모듈을 통합하여 두 가지의 상호 보완적인 강점을 활용합니다. Hybrid convolution-SSM 아키텍처를 기반으로 하는 ConMamba는 글로벌 인터랙션(long-range global interactions)과 세밀한 특징 패턴(localized fine-grained feature patterns)을 포착할 수 있습니다.

- **Performance Highlights**: MUSIC, AVSpeech 및 AudioSet 데이터셋에서 수행된 실험 결과, VC-S$^2$E가 기존의 경쟁 방식들에 비해 음성 품질과 음성의 이해도를 개선하는데 우수함을 입증했습니다.



### No-Reference Point Cloud Quality Assessment via Graph Convolutional Network (https://arxiv.org/abs/2411.07728)
Comments:
          Accepted by IEEE Transactions on Multimedia

- **What's New**: 이번 논문은 그래프 컨볼루션 네트워크(GCN)를 이용한 새로운 비참조( 비참조) 포인트 클라우드 품질 평가 방법(GC-PCQA)을 제안합니다.

- **Technical Details**: GC-PCQA는 다중 뷰 프로젝션, 그래프 생성, GCN 기반 품질 예측의 세 가지 모듈로 구성됩니다. 포인트 클라우드에 대해 먼저 다중 뷰 프로젝션을 수행하여 수평 및 수직으로 투영된 이미지 세트를 얻습니다. 다음으로, 다양한 투영된 이미지 간의 공간 관계를 바탕으로 지각 일관성 그래프를 구축합니다. 마지막으로, GCN을 통해 구축된 그래프에 대한 추론을 진행하여 서로 다른 투영 이미지 간의 상호 의존성과 상호작용을 특성화하여 품질 예측을 진행합니다.

- **Performance Highlights**: 실험 결과에 따르면, GC-PCQA는 기존의 최첨단 품질 평가 메트릭보다 우수한 성능을 보여주며, 사람의 지각 점수를 더욱 정확하게 예측할 수 있습니다. 개발한 코드는 공개 연구용으로 제공될 예정입니다.



### Enhancing Ultra High Resolution Remote Sensing Imagery Analysis with ImageRAG (https://arxiv.org/abs/2411.07688)
- **What's New**: 본 논문에서는 Ultra High Resolution (UHR) 원거리 감지 이미지 (Remote Sensing Imagery) 분석의 복잡성을 해결하기 위해 훈련이 필요 없는 프레임워크인 ImageRAG를 제안합니다. 이 프레임워크는 UHR 이미지의 긴 컨텍스트 선택(task)을 이미지 분석 과제로 전환하여, 가장 관련성이 높은 부분만 선택적으로 검색하고 집중할 수 있는 혁신적인 이미지 컨텍스트 검색 메커니즘을 설계했습니다.

- **Technical Details**: ImageRAG는 Retrieval-Augmented Generation (RAG) 기법을 바탕으로 설계되었으며, 두 가지 단계인 Retrieval 단계와 Generation 단계를 포함합니다. 이 프레임워크는 Fast path와 Slow path 두 가지 방법으로 작동하여 UHR 이미지에서 중요 정보를 효율적으로 관리할 수 있습니다.

- **Performance Highlights**: ImageRAG는 UHR 이미지를 분석하는 데 있어, 각 모델이 정밀한 세부 사항을 처리할 수 있도록 시각적 컨텍스트를 효과적으로 검색하고 강조합니다. 또한 이 프레임워크는 추가적인 훈련 없이도 적용 가능하여, UHR 원거리 감지 이미지를 보다 정확하고 효율적으로 다룰 수 있는 실제적 솔루션을 제공합니다.



### Data-Driven Graph Switching for Cyber-Resilient Control in Microgrids (https://arxiv.org/abs/2411.07686)
Comments:
          Accepted in IEEE Design Methodologies Conference (DMC) 2024

- **What's New**: 이 논문은 통신 네트워크의 중단 없이 마이크로그리드의 2차 제어 목표를 달성하기 위해 물리학 기반의 인공지능 신경망(Artificial Neural Network, ANN) 프레임워크를 제안합니다. 이 프레임워크는 통신 수준에서의 사이버 공격을 식별하고 신뢰할 수 없는 장치에서 받은 입력 데이터에 의존하지 않으면서 안정성을 추구하는 방법을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 입력 측정값을 분석하여 2차 제어 레이어의 비정상 동작을 유발하는지 여부를 점검합니다. 비정상이 감지되면 가능한 스패닝 트리(graph topology)를 반복적으로 확인하여 신뢰성 있게 2차 제어 목표를 수행할 수 있는 통신 네트워크 토폴로지를 찾습니다. 이 과정에서 False Data Injections 및 Man-in-the-Middle 공격에 대한 경우 연구를 포함하여 다양한 크기의 마이크로그리드에 대한 견고성의 검증이 이루어졌습니다.

- **Performance Highlights**: 성능 평가 결과, 저소음 환경에서 프레임워크가 효과적으로 작동함을 보여주었으나, 소음이 있는 환경에서 확장성을 시도할 경우 성능 저하가 발생할 수 있음을 발견했습니다.



### Fast Disentangled Slim Tensor Learning for Multi-view Clustering (https://arxiv.org/abs/2411.07685)
Comments:
          13 pages,6 figures, will be published to IEEE TMM

- **What's New**: 이 논문에서는 Tensor 기반의 다중 뷰 클러스터링(Multi-View Clustering) 방법 중 특히 제한된 부분을 극복하기 위해 새로운 접근 방법인 빠른 Disentangled Slim Tensor Learning(DSTL)을 제안합니다. DSTL은 높은 차수의 상관관계를 직접 탐색하고, latent semantic-unrelated 정보를 분리하여 성능을 향상시키는데 중점을 두고 있습니다.

- **Technical Details**: DSTL 방법은 행렬 분해(matrix factorization)에 기반하여 다양한 뷰의 latent 표현에서 고차원 상관관계를 탐색합니다. 이 방법은 Robust PCA에 영감을 받아 각 뷰의 표현을 의미와 관련 있는 부분과 의미와 무관한 부분으로 나누어 두 개의 슬림 텐서를 구성합니다. 또한, Consensus Alignment Indicator를 사용하여 semantic 관련 표현의 정렬을 수행하여 정보의 왜곡을 방지합니다.

- **Performance Highlights**: DSTL은 기존의 여러 최첨단 방법들보다 우수한 성능과 효율성을 보이며, 실제 데이터셋에서 더 나은 분류 및 클러스터링 결과를 제공합니다. 관련 코드도 공개되어 있어 연구자들이 쉽게 사용할 수 있습니다.



### AI enhanced diagnosis of Peyronies disease a novel approach using Computer Vision (https://arxiv.org/abs/2411.07684)
Comments:
          8 pages, 6 figures, 4 tables

- **What's New**: 본 연구는 페이로니병(Peyronie's Disease, PD) 진단을 위한 혁신적인 AI 기반 도구를 제시합니다. 이 도구는 전 세계 남성의 0.3%에서 13.1%에 영향을 미치는 이 질환의 진단을 개선하기 위해 개발되었습니다.

- **Technical Details**: 저희 방법은 이미지와 비디오에서 주요 포인트 탐지(key point detection)를 사용하여 음경의 굴곡 각도를 측정합니다. 고급 컴퓨터 비전 기술(advanced computer vision techniques)을 활용하여 해부학적 마크(anatomical landmarks)를 정확하게 확인합니다. 전통적인 진단법은 주관적(subjective)이고 침습적(invasive)인 방법을 포함해 환자에게 불편함과 정확도 저하를 초래할 수 있습니다.

- **Performance Highlights**: 본 모델은 PD와 정상 해부학적 변화(normal anatomical changes)를 96.7%의 민감도(sensitivity)와 100%의 특이도(specificity)로 구별합니다. 이러한 발전은 비뇨기학 진단에 상당한 개선을 가져오며, 의료 제공자와 환자를 위한 PD 평가의 효율성과 편리성을 크게 향상시킵니다.



### Spike Talk in Power Electronic Grids -- Leveraging Post Moore's Computing Laws (https://arxiv.org/abs/2411.07654)
Comments:
          The manuscript has been accepted for publication in the Proceedings of 2024 IEEE Design Methodologies for Power Electronics Conference (DMC2024)

- **What's New**: 본 논문에서는 'Spike Talk'라는 새로운 분산 제어 시스템을 제안하여 마이크로그리드(microgrid)에서의 전력 흐름을 기반으로 정보를 추론하고, 온라인 학습(online learning)을 통해 적응적이고 유연한 제어를 구현하는 방법을 제시합니다.

- **Technical Details**: Spike Talk는 생물학적으로 영감을 받은 스파이킹 신경망(spiking neural network, SNN)을 사용하여 통신 없이도 전력 및 정보를 효율적으로 전송합니다. 이 접근법은 스파이킹 뉴런의 시냅스 가소성(synaptic plasticity) 기능을 활용하여 온라인 학습을 가능하게 하며, 이는 Von Neumann Bottleneck 문제를 해결하기 위한 것입니다.

- **Performance Highlights**: 기초 사례 연구(preliminary case studies)를 통해 초기 결과가 제시되었으며, 향후 연구에서는 더 광범위한 검증이 이루어질 예정입니다. Spike Talk는 시스템의 회복력(resilience)과 운영 신뢰성을 향상시키는 데 기여할 것으로 예상됩니다.



### Understanding Audiovisual Deepfake Detection: Techniques, Challenges, Human Factors and Perceptual Insights (https://arxiv.org/abs/2411.07650)
- **What's New**: 이 연구는 복합적인 심층fake (deepfake)를 탐지하기 위해 음향(오디오)과 시각(비주얼) 모달리티를 함께 분석하는 접근 방식을 다룬 최초의 종합적인 조사입니다.

- **Technical Details**: 이 연구는 네 가지 유형의 deepfake를 정의하고 각각의 생성 기법 및 최신 Deep Learning (DL) 기술을 분석합니다. 특히, Generative Adversarial Networks (GAN)과 Variational Autoencoders (VAE)를 사용하여 음향과 시각의 결합된 디지털 조작 방법을 활용합니다.

- **Performance Highlights**: 음향 및 시각 모달리티를 활용한 기존 탐지 방법의 한계를 지적하고, 오디오 및 비주얼 deepfake 탐지에 대한 기존 연구의 격차를 해소하기 위한 연구 방향을 제시합니다. 또한, 이러한 방법들을 훈련시키기 위한 공개 데이터 세트에 대해서도 자세히 분석하였습니다.



### Multimodal Clinical Reasoning through Knowledge-augmented Rationale Generation (https://arxiv.org/abs/2411.07611)
Comments:
          11 pages. 4 figures

- **What's New**: ClinRaGen은 질병 진단을 위한 멀티모달 (multimodal) 이론 생성에 최적화된 소형 언어 모델 (SLM)로, 도메인 지식을 통합한 주목 메커니즘을 포함하여 시간적 전자 건강 기록 (EHR) 데이터와 결합하며, 단계별 (stepwise) 이론 증류 전략을 통해 임상 이론을 생성할 수 있도록 설정되었습니다.

- **Technical Details**: ClinRaGen은 도메인 지식과 시간적 EHR 데이터를 통합하기 위해 독특한 지식 증강 주목 메커니즘을 사용합니다. 이 방법은 텍스트와 시간적 기반의 임상 이론을 모두 생성하고, 멀티모달 EHR 데이터를 효과적으로 해석하여 더 정확한 질병 진단을 지원합니다.

- **Performance Highlights**: MIMIC-III 및 MIMIC-IV 데이터세트에서 수행한 평가에 따르면, ClinRaGen은 멀티모달 EHR 데이터를 효과적으로 해석하고, LLM들과 유사한 정확도를 가진 임상 이론을 생성하는데 있어 뛰어난 성능을 발휘하였습니다. 이는 LLM과 SLM 간의 성능 격차를 좁히는 데 기여합니다.



### Optimizing Service Function Chain Mapping in Network Function Virtualization through Simultaneous NF Decomposition and VNF Placemen (https://arxiv.org/abs/2411.07606)
- **What's New**: 본 논문에서는 Network Function (NF) 분해와 Virtual Network Function (VNF) 배치를 하나의 문제로 동시에 다룹니다. 이는 서비스 기능 체인( Service Function Chain, SFC)의 복잡한 서비스를 위한 새로운 접근법을 제안합니다.

- **Technical Details**: 연구에서는 비지배 정렬 유전 다목적 알고리즘(Non-Dominated Sorting Genetic Multi-Objective Algorithm, NSGAII)을 기반으로 한 다목적 분해 및 VNF 매핑 방법(Multi-Objective Decomposition and Mapping VNFs, MODMVNF)을 도입하여, 최적의 해법을 찾기 어려운 NP-hard 문제를 해결합니다.

- **Performance Highlights**: 제안된 방법은 ILP(정수 선형 프로그래밍) 수식을 통한 해결 결과 및 다목적 입자군 알고리즘(Particle Swarm Algorithm)으로 얻은 결과와 비교 시 비용 및 통신 지연 측면에서 효율성과 효과성을 보여줍니다.



### Circuit Complexity Bounds for RoPE-based Transformer Architectur (https://arxiv.org/abs/2411.07602)
- **What's New**: 이번 연구는 Rotary Position Embedding (RoPE)을 사용하는 Transformer 아키텍처의 표현력을 제한하는 엄격한 회로 복잡도 경계(circuit complexity bounds)를 설정합니다. 이 연구를 통해 RoPE 기반 Transformer의 근본적인 한계를 이론적으로 밝혀냈습니다.

- **Technical Details**: RoPE (Rotation Position Embedding)는 절대 및 상대 위치 정보를 인코딩하여 Transformer의 성능을 향상시키는 기술입니다. 연구에서는 RoPE 기반 아키텍처의 각 구성 요소에 대한 회로 복잡도를 체계적으로 조사하였고, 이 모델들이 TC⁰ 회로들로 시뮬레이션될 수 있음을 증명했습니다. 또한, TC⁰ = NC¹이 아닌 경우 poly(n) 정밀도와 O(1) 레이어, 그리고 d ≤ O(n) 조건 하에서 RoPE 기반 Transformer가 산술 문제 또는 부울 포뮬라 값 문제를 해결할 수 없음을 보여주었습니다.

- **Performance Highlights**: RoPE 기반 Transformer는 일반 Transformer 모델에 비해 더 높은 일반화 능력을 나타내며, 긴 컨텍스트 정보 처리에서 우수한 성능을 발휘합니다. 최근의 실험적 결과들은 RoPE가 긴 문서 요약 및 지속적인 대화 등 긴 컨텍스트 작업에서 탁월한 능력을 발휘함을 보여주고 있습니다.



### Problem-Oriented Segmentation and Retrieval: Case Study on Tutoring Conversations (https://arxiv.org/abs/2411.07598)
Comments:
          EMNLP 2024 Findings. Our code and dataset are open-sourced at this https URL

- **What's New**: 이번 연구에서는 Problem-Oriented Segmentation & Retrieval (POSR)이라는 새로운 연구 과제를 소개합니다. 이는 대화 내용을 세그먼트(segment)로 나누고 각 세그먼트를 적절한 참조 항목(reference item)에 연결하는 작업입니다. 특히 교육 분야에서 이 방법이 적용되며, LessonLink라는 실세계 튜터링 수업 데이터셋이 개발되었습니다.

- **Technical Details**: 이 연구에서는 3,500개의 세그먼트와 116개의 SAT 수학 문제로 구성된 24,300분 분량의 튜터링 수업을 포함하는 LessonLink 데이터셋을 제시합니다. 여러 POSR 접근 방식을 정의하고 평가하며, 세그멘테이션 방법(예: TextTiling), 정보 검색(IR) 방법(예: ColBERT) 및 대형 언어 모델(LLMs) 방법을 포함합니다. 새로운 세그멘테이션 및 검색 점수(SRS)를 도입하여 교실 내 대화 세그먼트와 참조 자료 간의 정확성을 측정합니다.

- **Performance Highlights**: POSR 방법은 독립적인 세그멘테이션 및 검색 파이프라인보다 최대 76% 향상된 성능을 보였으며, 전통적인 세그멘테이션 방법보다 최대 78% 더 높은 정확도를 기록했습니다. 또한, 튜터가 각 문제에 소요하는 시간에 따라 개념적 설명과 절차적 설명의 차이가 있다는 통찰을 제공하였으며, 이는 교육 현장에서의 언어 및 시간 관리 실천을 개선할 수 있는 기회를 제시합니다.



### Entropy Controllable Direct Preference Optimization (https://arxiv.org/abs/2411.07595)
- **What's New**: 본 연구에서는 Direct Preference Optimization (DPO) 방식의 수정인 H-DPO를 제안합니다. H-DPO는 정책의 엔트로피를 조정할 수 있어 분포의 선명도를 높이고, 효과적인 mode-seeking fitting을 가능하게 합니다.

- **Technical Details**: H-DPO는 기존 DPO의 손실 계산 수식을 단순히 수정하여 엔트로피 조정을 통해 성능을 향상시킬 수 있도록 설계되었습니다. 이 접근법에서는 손실 함수의 정규화 항을 변경하여 분포의 엔트로피 H(π)를 직접 제어하게 됩니다.

- **Performance Highlights**: 실험 결과, H-DPO는 다양한 작업에서 DPO보다 우수한 성능을 보여주었으며, 특히 수학적 과제의 pass@$k$ 평가에서 더욱 뛰어난 성과를 나타냈습니다.



### Overhead-free User-side Recommender Systems (https://arxiv.org/abs/2411.07589)
Comments:
          arXiv admin note: text overlap with arXiv:2208.09864, arXiv:2403.15757

- **What's New**: 최근에 사용자 측 추천 시스템(user-side recommender systems)의 새로운 패러다임이 제안되었습니다. 이 시스템은 최종 사용자가 직접 구축하고 사용할 수 있습니다. 기존의 제공자 측 추천 시스템과는 대조적으로, 사용자는 불공정한 추천 시스템에 대해 스스로 해결책을 마련할 수 있습니다.

- **Technical Details**: 이 논문에서는 RecCycle이라는 오버헤드 없는 사용자 측 추천 시스템을 제안합니다. RecCycle의 주요 아이디어는 이전 추천 결과를 재활용(recycle)하여 사용자 측에서 추천을 생성하는 것입니다. 이를 통해 추가적인 통신 비용을 없애고, 기존의 추천 시스템과 결합하여 실시간 추천을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, RecCycle은 최신 사용자 측 추천 알고리즘과 동등한 성능을 보이며, 통신 비용을 크게 줄이는 효과를 확인했습니다. 실제 환경인 X(Twitter)에서 사용자들이 요구하는 기능을 갖춘 추천 시스템을 구현할 수 있음을 증명했습니다.



### Reinforcement Learning Framework for Quantitative Trading (https://arxiv.org/abs/2411.07585)
Comments:
          8 pages, 9 figures, 3 tables, accepted at ICAIF 2024 FM4TS Workshop

- **What's New**: 이 연구는 금융 거래에서 강화 학습(RL) 에이전트의 효과 재설계를 목표로 합니다. 특히, 기존 문헌에서의 거래 전략 모델이 역사적 데이터에 대한 백테스트(back testing)에만 초점을 맞추었던 점을 보완하며, 다양한 금융 지표를 활용하여 긍정적 및 부정적 매수/매도 신호를 식별하는 능력을 향상시키고자 합니다.

- **Technical Details**: 이 논문은 기술적 지표(technical indicators)를 활용하여 RL 에이전트가 더 나은 거래 결정을 내릴 수 있도록 하는 방법론을 제시합니다. Markov Decision Process(MDP)를 기반으로 한 강화 학습의 원리를 통해, 시간에 따라 변동하는 금융 데이터를 고려한 결정적 요소를 분석합니다. 또한, 차원 축소(dimensionality reduction), 특징 선택(feature selection), CNN, RNN 및 LSTM과 같은 다양한 모델을 사용하여 데이터의 특성을 이해하는 방법을 설명합니다.

- **Performance Highlights**: 실험에서 사용된 20개의 기술적 지표를 통해 RL 에이전트가 다양한 시장 상황에 맞춰 적절히 대응할 수 있는 능력을 갖추도록 하는 방안을 제시합니다. 이 연구는 데이터 전처리(data pre-processing), 보상 함수(reward function), 정규화(normalization) 방법 등을 포함하여 RL 기반 거래 전략의 성능을 극대화할 수 있는 기초를 마련하였습니다.



### Disentangling Tabular Data towards Better One-Class Anomaly Detection (https://arxiv.org/abs/2411.07574)
- **What's New**: 이 논문은 Tabular anomaly detection(표 형태의 이상 탐지)에서의 one-class classification(단일 클래스 분류) 접근법을 발전시킵니다. 기존의 learnable mask 전략으로는 mask가 균일하게 생성될 위험이 있었는데, 이 문제를 해결하기 위해 두 개의 non-overlapping(비포괄적) 그리고 correlated(상호연관된) attribute subsets(속성 집합)인 CorrSets를 도입하여 이상을 효과적으로 탐지합니다.

- **Technical Details**: 이 연구에서 제안된 새로운 방법론인 Disent-AD는 two-head self-attention module을 활용하여 latent space(잠재 공간)에서 두 개의 CorrSets를 효과적으로 분리합니다. 이 과정에서 attention maps(어텐션 맵)을 통해 속성의 상관관계를 포착하고, reconstruction task(재구성 작업)를 통해 원본 데이터를 복원합니다. 이를 통해 model은 normal samples(정상 샘플) 간의 내부 상관관계를 잘 학습합니다.

- **Performance Highlights**: 20개의 표 데이터셋에서 진행된 실험 결과, 이 방법은 AUC-PR(Area Under the Curve - Precision-Recall)에서 평균 6.1% 및 AUC-ROC(Area Under the Curve - Receiver Operating Characteristic)에서 평균 2.1% 성능 향상을 보여주며, 최신 기법들을 크게 초월하는 성능을 입증했습니다.



### EUR/USD Exchange Rate Forecasting incorporating Text Mining Based on Pre-trained Language Models and Deep Learning Methods (https://arxiv.org/abs/2411.07560)
- **What's New**: 본 연구는 EUR/USD 환율 예측을 위한 혁신적인 접근방법을 소개하며, 이는 딥러닝(Deep Learning), 텍스트 분석(Textual Analysis), 입자군집 최적화(Particle Swarm Optimization, PSO)를 통합합니다. 제안된 PSO-LSTM 모델은 전통적인 계량경제학(econometrics) 및 머신러닝(machine learning) 모델에 비해 우수한 성능을 보임을 입증합니다.

- **Technical Details**: 이 연구는 RoBERTa-Large 모델을 활용한 감정 분석(Sentiment Analysis)과 LDA를 통한 주제 모델링(Topic Modeling) 등 고급 텍스트 마이닝 기법을 사용합니다. PSO-LSTM 모델은 텍스트 데이터의 각 카테고리가 예측 성능에 미치는 기여도를 검증하기 위한 검증 실험을 포함하여, 정량적 데이터와 질적 데이터를 통합하여 환율 예측의 정확성 및 강건성을 향상시킵니다.

- **Performance Highlights**: PSO-LSTM 모델은 SVM, SVR, ARIMA 및 GARCH와 같은 기준 모델보다 우수한 예측 성능을 보였습니다. 본 연구는 금융 분야에서 인공지능의 변혁적 잠재력을 강조하며, 실시간 예측 및 대체 데이터 소스의 통합에 대한 미래 연구를 위한 길을 마련합니다.



### Zer0-Jack: A Memory-efficient Gradient-based Jailbreaking Method for Black-box Multi-modal Large Language Models (https://arxiv.org/abs/2411.07559)
Comments:
          Accepted to Neurips SafeGenAi Workshop 2024

- **What's New**: 본 논문에서는 Zer0-Jack이라는 새로운 방법을 제안합니다. 이 방법은 Multi-modal Large Language Models (MLLMs)의 안전 시스템을 우회하기 위해 zeroth-order optimization을 활용하며, 기존의 white-box 접근 방식 없이도 작동합니다.

- **Technical Details**: Zer0-Jack은 patch coordinate descent를 사용하여 블랙박스 MLLMs를 직접 공격할 수 있는 악의적인 이미지 입력을 효율적으로 생성하며, 이에 따라 메모리 사용량을 줄입니다. 이 방법은 고차원 입력에서 높은 추정 오류 문제를 완화하여 특정 이미지 부분만 최적화합니다.

- **Performance Highlights**: Zer0-Jack은 다양한 모델에서 높은 공격 성공률을 달성하였으며, MiniGPT-4에 대해 95%의 성공률을 기록했습니다. 또한, 상업용 MLLMs인 GPT-4o를 직접 공격할 수 있는 가능성을 보여줍니다.



### Contrastive Language Prompting to Ease False Positives in Medical Anomaly Detection (https://arxiv.org/abs/2411.07546)
Comments:
          4 pages, 3 figures, 2 tables

- **What's New**: 이 논문은 의료 영상에서의 이상 탐지를 위해 CLIP의 새로운 응용 방법인 Contrastive Language Prompting (CLAP)을 제안합니다. CLAP은 긍정적 및 부정적 텍스트 프롬프트를 활용하여 비정상 영역을 보다 정확하게 식별하고, 일반적으로 나타나는 오탐지를 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: CLAP 방법은 CLIP의 시각적 주의(attention) 메커니즘을 통해 비정상 영역에 대한 시각적 주의를 유도하고, 부정적 프롬프트를 사용해 정상 영역에 대한 주의를 약화시킵니다. 이를 통해 BMAD 데이터셋을 활용한 실험에서 이상 탐지 성능이 향상됨을 보여줍니다. U-Net을 이용한 무감독 이상 탐지(UAD) 방법도 제안되어, 정상 샘플을 기반으로 한 재구성을 통해 비정상 패턴을 처리합니다.

- **Performance Highlights**: BMAD 데이터셋을 사용한 실험을 통해 CLAP 방법이 비정상 영역에 대한 강한 주의 문제를 극복하고, 기존 방법들에 비해 UAD 성능을 향상시킴을 입증했습니다. 특히, 정상 샘플로 훈련된 U-Net이 비정상 패턴 재구성에서 어려움을 겪는다는 점을 이용하여, 더욱 정밀한 이상 진단을 가능하게 했습니다.



### Model Stealing for Any Low-Rank Language Mod (https://arxiv.org/abs/2411.07536)
- **What's New**: 이 논문에서는 모델 스틸링(Model Stealing)의 이론적 기초를 탐구하며, Hidden Markov Models (HMMs) 및 저차원 언어 모델(low-rank language models)의 도용을 위한 효율적인 알고리즘을 제시합니다.

- **Technical Details**: 저자들은 조건부 쿼리 모델(conditional query model)을 활용하여 저차원 확률 분포를 학습하는 알고리즘을 개발했습니다. 이 알고리즘은 특정 시점에서의 조건부 분포를 나타내기 위해 바리센트릭 스파너(barycentric spanners)를 구성하고, 오류를 방지하기 위해 상대 엔트로피(relative entropy)를 포함하는 볼록 최적화(convex optimization) 문제를 반복적으로 해결합니다.

- **Performance Highlights**: 이 연구는 Kakade et al.의 이전 결과를 개선하여, 높은 '신뢰도'가 필요한 복잡한 조건을 제거하고, 저차원 출력 분포를 가진 모든 언어 모델을 도용할 수 있게 되었습니다. 이론적으로, ML 모델이 추론 시 더 복잡한 문제를 해결할 수 있도록 하는 것이 성능 향상에 기여할 수 있다는 흥미로운 예시로 작용합니다.



### Evaluating ChatGPT-3.5 Efficiency in Solving Coding Problems of Different Complexity Levels: An Empirical Analysis (https://arxiv.org/abs/2411.07529)
- **What's New**: 이 연구는 ChatGPT의 GPT-3.5-turbo 모델이 LeetCode에서 알고리즘 코딩 문제를 해결하는 성능을 평가합니다. 다양한 난이도(쉬운, 중간, 어려움)에서 ChatGPT의 문제 해결 능력, 프롬프트 엔지니어링(prompt engineering)의 효과, 프로그래밍 언어에 대한 성능 차이를 분석합니다.

- **Technical Details**: 연구는 Python 스크립트를 사용하여 자동화된 실험을 수행하고, ChatGPT에게 문제 해결을 지시하는 프롬프트를 생성하여 성과를 평가합니다. 무작위 샘플을 통해 Java에서 78%, C++에서 50%의 문제 해결율을 보였지만, Elixir, Erlang, Racket와 같은 덜 일반적인 언어에 대한 성과는 전무했습니다.

- **Performance Highlights**: ChatGPT는 쉬운 문제에서 92%, 중간 문제에서 79%, 어려운 문제에서 51%의 성공률을 기록했습니다. Chain of Thought Prompting을 통해 14-29%의 성능 향상이 있었고, GPT-4로 전환 시 33-58%의 개선효과를 보였습니다.



### SecEncoder: Logs are All You Need in Security (https://arxiv.org/abs/2411.07528)
- **What's New**: 이번 논문에서는 보안 로그(security logs)를 사용하여 사전 훈련된 SecEncoder라는 특화된 작은 언어 모델(small language model)을 소개합니다. 이는 일반적인 언어 모델들이 가지고 있는 도메인 특정 제한사항을 해결하고, 보안 로그에서 발견되는 고유한 언어와 패턴에 집중하기 위해 설계되었습니다.

- **Technical Details**: SecEncoder는 보안 로그 데이터 세트를 기반으로 사전 훈련된 인코더 전용 모델입니다. 이 모델은 다양한 보안 사건과 관련된 이벤트 및 활동을 포착하는 로그를 분석하고, 이상 탐지(anomaly detection), 로그 검색(log search), 사건 분류(incident classification)와 같은 작업에서 평가됩니다.

- **Performance Highlights**: SecEncoder는 BERTlarge, DeBERTa-v3-large 및 OpenAI의 Embedding(textembedding-ada-002) 모델보다 다양한 작업에서 우수한 성능을 보였습니다. 보안 로그에만 주로 사전 훈련되었음에도 불구하고, 사고 우선순위 설정과 위협 인텔리전스 문서 검색과 같은 로그 분석을 넘는 작업에서도 더 나은 성능을 나타냈습니다. 이는 보안 로그로의 도메인 특정 사전 훈련이 LMs의 성능을 상당히 향상시킬 수 있음을 시사합니다.



### Fair Summarization: Bridging Quality and Diversity in Extractive Summaries (https://arxiv.org/abs/2411.07521)
Comments:
          Accepted at Algorithmic Fairness through the Lens of Metrics and Evaluation Workshop @ NeurIPS 2024

- **What's New**: 이 논문에서는 사용자 생성 콘텐츠의 다중 문서 요약에서 공정성을 보장하기 위한 두 가지 새로운 방법, FairExtract와 FairGPT를 소개합니다. 이 방법들은 서로 다른 사회적 집단을 공평하게 대표할 수 있는 방법을 제시하여 기존 요약 기법의 한계를 극복합니다.

- **Technical Details**: FairExtract는 클러스터링 기반의 접근 방식으로, 정교한 군집화를 통해 문서에서 중요한 정보를 추출하며, FairGPT는 GPT-3.5-turbo 모델을 활용하여 공정성 제약을 적용합니다. 논문에서는 다양한 요약 품질 지표(SUPERT, BLANC, SummaQA 등)와 공정성 지표(F)를 사용하여 이 두 방법을 평가하고, 그들의 성능을 기존 방법과 비교합니다.

- **Performance Highlights**: FairExtract와 FairGPT는 고품질의 요약을 유지하면서도 뛰어난 공정성을 달성하며, composite metrics(SUPERT+F, BLANC+F)를 통해 품질과 공정성을 동시에 평가합니다. 이 연구는 요약 작업에서 공정성의 중요성을 강조하고 공정성을 고려한 NLP 모델의 미래 연구를 위한 기준을 설정합니다.



### TIPS: Threat Actor Informed Prioritization of Applications using SecEncoder (https://arxiv.org/abs/2411.07519)
- **What's New**: 이 논문은 보안 분야에 특화된 언어 모델인 SecEncoder를 사용하여 TIPS: Threat Actor Informed Prioritization을 소개합니다. 이는 위협 행위자(threat actor) 정보를 통합하여 양호한 애플리케이션의 탐지 및 우선순위 설정을 강화합니다.

- **Technical Details**: TIPS는 encoder와 decoder 언어 모델의 강점을 결합하여, 손상된 애플리케이션을 탐지하고 우선순위를 매깁니다. 위협 행위자 정보(threat actor intelligence)를 통합하여 탐지의 정확성과 관련성을 높입니다.

- **Performance Highlights**: 실제 애플리케이션 벤치마크 데이터세트에 대한 광범위한 실험을 통해 TIPS는 악성 애플리케이션 탐지에서 0.90의 F-1 점수를 달성하였습니다. 또한, TIPS는 보안 분석가의 조사를 87% 줄여 위협 대응 프로세스를 간소화하고 전반적인 보안 상태를 개선합니다.



### FM-TS: Flow Matching for Time Series Generation (https://arxiv.org/abs/2411.07506)
- **What's New**: FM-TS라는 새로운 Time Series generation 프레임워크가 소개되었습니다. 이 프레임워크는 Flow Matching에 기반하여 시간 경과에 따른 데이터의 생성 과정을 간소화하고, Conditional 및 Unconditional 설정 모두에서 최적화할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: FM-TS는 Rectified Flow Matching을 이용하여 시간 series 생성의 효율성과 성능을 높입니다. 이 모델은 반복적인 샘플링이나 복잡한 노이즈 일정이 필요하지 않으며, 단일 프레임 전달을 통해 샘플링을 수행합니다. 특히, 본 연구는 Rectified Flow Matching을 시간 series 생성에 처음 적용한 사례입니다.

- **Performance Highlights**: FM-TS는 다양한 데이터셋에 대해 기존 방법보다 우수한 성능을 보이며, Sines, Stocks, ETTh, MuJoCo, Energy, fMRI와 같은 데이터셋에서 차별적인 점수(discriminative score)를 달성했습니다. 예를 들어, Stock 데이터셋에서는 0.019, ETTh에서는 0.011의 점수를 기록하며 이전 최고의 성능인 0.067을 크게 초과하였습니다. 또한, Solar forecasting 및 MuJoCo imputation 과제에서도 성능이 크게 향상되었습니다.



### LAUREL: Learned Augmented Residual Layer (https://arxiv.org/abs/2411.07501)
Comments:
          Accepted at the 2nd Efficient Systems for Foundation Models Workshop at the International Conference on Machine Learning (ICML) 2024

- **What's New**: 이 논문에서는 전통적인 잔여 연결(residual connection)의 일반화된 형태인 학습된 증강 잔여 레이어(Learned Augmented Residual Layer, LAuReL)를 소개합니다. LAuReL은 모델 품질과 메모리 사용량 모두에서 기존 방법을 초월하는 것을 목표로 합니다.

- **Technical Details**: LAuReL의 주요 아이디어는 잔여 연결을 다음과 같이 재구성하는 것입니다: α는 학습된 스칼라 매개변수이며, g(⋅)는 학습된 선형 함수입니다. 이 함수는 잔여 연결의 출력을 입력으로 사용하여 더 복잡한 정보 흐름을 형성합니다. LAuReL은 모델의 크기와 지연(latency) 측면에서 경량화된 방식으로 이러한 잔여 흐름을 학습할 수 있습니다.

- **Performance Highlights**: ResNet-50과 ImageNet 1K 작업에서 LAuReL은 추가 레이어를 추가했을 때의 성능 향상 중 60%를 달성하면서도 파라미터 수는 0.003%만 증가했습니다. 이는 LAuReL이 적은 파라미터로도 높은 성능을 보장한다는 것을 보여줍니다.



### Enhancing Link Prediction with Fuzzy Graph Attention Networks and Dynamic Negative Sampling (https://arxiv.org/abs/2411.07482)
- **What's New**: 본 논문에서는 전통적인 Graph Neural Networks(GNNs)가 무작위 negative sampling에 의존하는 한계를 보완하기 위해 Fuzzy Graph Attention Networks(FGAT)를 제안합니다. 이 접근법은 fuzzy rough set을 통합하여 동적인 negative sampling과 향상된 노드 특징 집계를 가능하게 합니다.

- **Technical Details**: 본 연구에서는 Fuzzy Negative Sampling(FNS)을 통해 fuzzy 유사도 기반의 고품질 negative edges 선택하는 메커니즘을 도입합니다. FGAT 레이어는 fuzzy rough set 원리를 통합하여 강력하고 구분 가능한 노드 표현을 가능하게 합니다. 이를 통해 GNN의 전반적인 학습 효율성을 높이고 있습니다.

- **Performance Highlights**: 실험 결과, FGAT는 두 개의 연구 협력 네트워크에서 기존의 최첨단 방법들보다 우수한 링크 예측 정확도를 보여주었습니다. 특히, fuzzy rough set의 힘을 활용하여 효과적인 negative sampling과 노드 특징 학습을 구현함으로써 성능이 개선되었습니다.



### IdentifyMe: A Challenging Long-Context Mention Resolution Benchmark (https://arxiv.org/abs/2411.07466)
Comments:
          9 pages, 5 figures

- **What's New**: 이번 연구에서는 LLMs(대규모 언어 모델)의 핵심 연합 해결(coreference resolution) 능력을 평가하기 위한 새로운 벤치마크인 IdentifyMe를 소개합니다. 이 벤치마크는 다국적 선택형 질문(MCQ) 형식으로 제공되어 LLM의 참조 이해도를 보다 효과적으로 측정할 수 있도록 합니다.

- **Technical Details**: IdentifyMe는 문서에서 언급된 객체를 식별하는 MCQ 기반 벤치마크로, LitBank와 FantasyCoref라는 두 가지 장문 코어페런스 데이터셋에서 파생된 언급을 사용합니다. 벤치마크는 특정 유형의 언급(대명사 및 명사)을 필터링하고 난이도 조정을 위한 휴리스틱(heuristics)을 적용하여 모델이 보다 복잡한 문제를 해결하도록 만듭니다.

- **Performance Highlights**: 가장 높은 점수를 기록한 모델인 GPT-4o는 81.9%의 정확도를 달성하였으며, 이는 현재 LLMs의 참조 능력이 상당히 우수하지만 여전히 개선의 여지가 있음을 보여주고 있습니다. 또한, 모델은 대명사 언급을 해결하는 데 더 큰 어려움을 겪었으며 이는 표면 정보가 제한적이기 때문입니다.



### BudgetMLAgent: A Cost-Effective LLM Multi-Agent system for Automating Machine Learning Tasks (https://arxiv.org/abs/2411.07464)
Comments:
          Presented at AIMLSystems '24

- **What's New**: 이번 연구에서는 저비용 및 무비용 모델을 활용하여 복잡한 머신 러닝 (ML) 작업을 해결하기 위한 멀티 에이전트 시스템을 제안합니다. 이전 시스템들이 고비용의 대형 모델에 의존했던 반면, 이 새로운 접근법은 비용 효율성을 강조합니다.

- **Technical Details**: 다양한 LLM 전문가의 조합을 이용한 Multi-Agent 시스템을 사용하며, 여기에는 프로파일링, 과거 관찰의 효율적인 검색, LLM cascade, 간헐적인 전문가 호출을 포함합니다. 실험은 MLAgentBench 벤치마크에서 수행됩니다.

- **Performance Highlights**: 본 시스템은 GPT-4 기반 단일 에이전트 시스템에 비해 평균 94.2%의 비용 절감과 32.95%의 성공률을 기록했습니다. 이는 단일 에이전트 GPT-4 시스템의 22.72%의 성공률에 비해 월등히 높은 수치입니다.



### BLIP3-KALE: Knowledge Augmented Large-Scale Dense Captions (https://arxiv.org/abs/2411.07461)
- **What's New**: KALE 데이터셋은 2억 1천 8백만 개의 이미지-텍스트 쌍을 포함하며, 사실적으로 기반한 이미지 캡션 생성을 가능하게 하는 새로운 접근 방식을 제시합니다. KALE는 기존 CapsFusion보다 규모와 밀도가 뛰어나며, 강화된 지식 기반 캡션 생성을 통해 여러 멀티모달 모델의 훈련 성능을 개선합니다.

- **Technical Details**: KALE는 2단계 접근 방식을 사용합니다. 첫 번째 단계에서는 CogVLM-17B를 활용하여 1억 개의 밀집 캡션을 생성하고, Mistral 모델을 통해 이 캡션에 사실 정보를 추가합니다. 두 번째 단계에서는 특화된 VLM을 훈련하여 KALE 데이터셋을 확대하고, 최종적으로 2억 1천 8백만 개의 이미지-텍스트 쌍을 생성합니다.

- **Performance Highlights**: KALE로 훈련된 VLM은 TextVQA에서 59.92%, VQAv2에서 70.10%, ScienceQA에서 72.68%의 성능을 달성하며, 그 평균 성능은 51.96%에 이릅니다. 이는 기존의 데이터셋인 Datacomp-1B 및 CapsFusion보다 뛰어난 성능을 보여줍니다.



### Research on fault diagnosis of nuclear power first-second circuit based on hierarchical multi-granularity classification network (https://arxiv.org/abs/2411.07453)
- **What's New**: 이 논문에서는 핵발전소의 복잡한 전기기계 시스템에서 중요한 기계 부품 고장을 시뮬레이션 하기 위해 AP1000 풀스케일 시뮬레이터를 활용하였습니다. 이를 통해 핵발전소의 fault dataset(고장 데이터셋)을 구축하고, 효과적인 fault diagnosis(고장 진단) 모델을 제안하였습니다.

- **Technical Details**: 제안된 모델은 EfficientNet 대규모 모델을 기반으로 하여 hierarchical multi granularity classification(계층적 다중 세분화 분류) 고장 진단 모델을 구성하였습니다. 이는 서로 다른 회로와 시스템 구성 요소의 고장을 계층적으로 분류할 수 있는 능력을 갖추고 있습니다.

- **Performance Highlights**: 연구 결과, 제안된 고장 진단 모델은 핵발전소의 다양한 회로 및 시스템 구성 요소의 고장을 효과적으로 계층적으로 분류할 수 있음을 보여주었습니다. 그러나 시뮬레이터에서 얻어진 fault dataset은 매개변수 중복으로 인해 추가 정보를 유입시켜 모델의 진단 성능에 영향을 미칠 수 있습니다.



### Optimizing Data Delivery: Insights from User Preferences on Visuals, Tables, and Tex (https://arxiv.org/abs/2411.07451)
- **What's New**: 이 연구는 사용자가 특정 질문에 대해 차트(chart), 테이블(table), 또는 텍스트(text) 중 어떤 형식으로 결과를 보고자 하는지를 조사합니다. 이를 통해 특정 질문에 대해 사용자에게 가장 적합한 결과 형식을 제시하는 방법을 이해합니다.

- **Technical Details**: 사용자 연구(user study)를 수행하여 사용자의 선호 데이터를 수집하였으며, 다양한 사용자 특성이 데이터 출력에 미치는 영향을 분석했습니다. 본 연구는 LLMs(대형 언어 모델)의 사용을 통해 사용자 선호를 복제할 수 있는 정도에 대해서도 탐구합니다.

- **Performance Highlights**: 본 연구는 사용자 특성이 데이터 출력 선호도에 미치는 영향을 분석하고, 특정 데이터 질문에 대해 사용자가 선호하는 출력 형식을 정량적으로 파악하는 데 기여합니다. LLM을 활용하여 사용자 선호도를 예측하는 가능성을 제시하며, 사용자 맞춤형 LLM의 효과성도 강조합니다.



### The Effect of Scheduling and Preemption on the Efficiency of LLM Inference Serving (https://arxiv.org/abs/2411.07447)
- **What's New**: 이 논문에서는 INFERMAX라는 분석 프레임워크를 소개하여 다양한 스케줄러를 비교하고 최적 스케줄러를 찾는 문제를 제약 만족 문제(Constraint Satisfaction Problem, CSP)로 모델링합니다. 이 프레임워크는 인퍼런스 비용 모델을 사용하여 효율적인 스케줄링 기회를 탐구합니다.

- **Technical Details**: INFERMAX 프레임워크는 다양한 하드웨어와 모델 구성에서 스케줄러를 시뮬레이션할 수 있는 기능을 확장합니다. 이는 GPU를 실제로 실행할 필요 없이 비용 모델을 기반으로 비용 효율성을 제공합니다. 스케줄링 정책을 CSP의 형태로 제약으로 강제하고, 지연(latency) 및 처리량(throughput) 최적화를 목표로 합니다.

- **Performance Highlights**: 사전 요청을 중단(preempting)하는 것이 요청을 피하는 것보다 GPU 비용을 30%까지 줄일 수 있다는 결과를 도출했습니다. 이 연구는 스케줄링 최적화가 연관된 요청 길이와 중단의 상관관계를 밝혀내며, 총 200 GPU 시간을 소모한 분석 결과를 제공합니다.



### Input-Based Ensemble-Learning Method for Dynamic Memory Configuration of Serverless Computing Functions (https://arxiv.org/abs/2411.07444)
Comments:
          10 pages, 2 tables, 28 figures, accepted conference paper - UCC'24

- **What's New**: 논문은 함수 메모리 할당 과정에서 입력 특성을 고려하여 리소스를 최적화하는 MemFigLess라는 서버리스 솔루션을 제안합니다. 이 프레임워크는 오프라인 프로파일링을 통해 입력 기반 메모리 요구 사항을 예측하여 리소스 낭비를 줄이고 효율적인 비용 최적화 가격 구조를 제공합니다.

- **Technical Details**: MemFigLess는 입력 파라미터와 메트릭을 기반으로 하는 Multi-output Random Forest Regression 모델을 학습하여 입력 인식 최적화 구성을 유도합니다. 프레임워크는 입력 인수, 예상 실행 시간 및 비용 SLOs을 기반으로 메모리 할당 범위를 탐색하며, 성능 변화를 반영하기 위한 피드백 루프를 제공합니다.

- **Performance Highlights**: MemFigLess는 AWS Lambda 서비스에서 최신 기술과 비교했을 때 입력 인지 리소스 관계를 캡처하고, 최대 82%의 리소스를 절감하며 최대 87%의 실행 비용 절약을 증명했습니다.



### Automatically Detecting Online Deceptive Patterns in Real-tim (https://arxiv.org/abs/2411.07441)
- **What's New**: 이 논문에서는 사용자에게 실시간으로 디지털 인터페이스의 속임수 패턴을 감지하고 알리는 'AutoBot'이라는 자동화된 도구를 소개합니다. 이 도구는 웹사이트 스크린샷을 분석하고 기계 학습 기법을 활용하여 속임수 패턴을 식별합니다.

- **Technical Details**: AutoBot은 웹사이트의 시각적 외관을 분석하여 사용자 인터랙션 요소를 찾아내고, HTML 구조에 의존하지 않고 텍스트 특징을 추출하는 두 단계의 파이프라인을 사용합니다. 커스텀 언어 모델을 사용해 요소에 대한 맥락을 이해하여 속임수 패턴의 존재를 판단합니다.

- **Performance Highlights**: AutoBot은 크롬 브라우저의 경량 확장 프로그램으로 구현되었으며, 모든 분석을 로컬에서 수행하여 대기 시간을 최소화하고 사용자 개인정보를 보호합니다. 광범위한 평가를 통해, AutoBot이 사용자들이 안전하게 디지털 환경을 탐색할 수 있도록 능력을 향상시킨다는 것을 입증하였습니다.



### Predicting BWR Criticality with Data-Driven Machine Learning Mod (https://arxiv.org/abs/2411.07425)
- **What's New**: 이 논문은 수조(boiling water) 원자로에서 과도한 비판성을(data-driven deep learning model) 추정하는 혁신적인 방법을 제안합니다.

- **Technical Details**: 기존 방법과는 달리, 이 접근법은 머신러닝 기법을 활용하여 원자로 주기(cycle) 동안 필요한 연료량을 정확히 결정하는 데 도움을 줍니다. 이 연구에서는 원자로의 비판 성능(critical performance) 예측을 위한 데이터를 분석하여 훈련된 딥러닝(dedep learning) 모델을 사용합니다.

- **Performance Highlights**: 이 방법은 원자로가 주기의 끝까지 비판 상태를 유지하도록 필요한 연료량을 최적화함으로써, 조기 감속(coastdown) 및 부족한 연료로 인한 비용 손실을 방지할 수 있습니다.



### Controllable Context Sensitivity and the Knob Behind I (https://arxiv.org/abs/2411.07404)
- **What's New**: 이 논문에서는 언어 모델이 문맥(context)과 이전 지식(prior knowledge) 간의 의존도를 조절할 수 있는 방법을 탐구합니다. 이를 통해 언어 모델이 질문에 답변할 때 문맥에서 오는 정보와 내재된 지식 중 무엇을 더 신뢰해야 하는지를 조율하는 메커니즘을 조명합니다.

- **Technical Details**: 연구진은 controllable context sensitivity (CCS)라는 과제를 설정하여 모델이 특정 문맥(예: 파리(PARIS)라는 대도시가 잉글랜드에 있다고 가정하고)에서 문맥을 따를 것인지, 아니면 이전 지식을 따를 것인지를 구분하도록 합니다. 실험에서는 Llama-3.1, Mistral-v0.3, Gemma-2 모델을 사용하여 85-95%의 정확도로 과제를 해결했습니다. 이 과정에서 기계적 해석 가능성(mechanistic interpretability) 도구를 활용하여 모델의 레이어(layer) 중 문맥 민감도가 가장 높은 부분을 규명합니다.

- **Performance Highlights**: 연구의 결과 나타난 바에 따르면, 문맥과 이전 지식 간의 의사결정을 효과적으로 지원하는 단일차원(subspace) 메커니즘은 다양한 대형 언어 모델에서 공통적으로 나타났습니다. 이 발견은 보다 강력한 언어 모델을 개발하는 데 방향성을 제시하며, 컨트롤 가능한 컨텍스트와 이전 지식에 대한 의존도를 조절할 수 있는 가능성을 보여줍니다.



### Beyond Keywords: A Context-based Hybrid Approach to Mining Ethical Concern-related App Reviews (https://arxiv.org/abs/2411.07398)
- **What's New**: 이번 연구에서는 도메인 특화된 가설을 사용하는 혁신적인 방법을 제안하여 윤리적인 문제와 관련된 모바일 앱 리뷰를 효과적으로 추출하는 자연어 처리(NLP) 기반의 접근 방식을 탐구합니다.

- **Technical Details**: 연구는 43,647개의 정신 건강 관련 앱 리뷰를 활용하여 자연어 추론(NLI)과 디코더 전용 대형 언어 모델(LLM)을 통합하는 하이브리드 방법을 개발했습니다. 이 방법은 도메인 특화된 사생활 가설을 적용하여 윤리적 문제와 관련된 앱 리뷰를 분류하고 추출하는 데 중점을 두었습니다. DeBERTa-v3-base-mnli-fever-anli NLI 모델이 도메인 특화된 가설을 사용하여 최상의 성능을 보여주었으며, Llama3.1-8B-Instruct LLM이 앱 리뷰 분류에 가장 뛰어난 성능을 나타냈습니다.

- **Performance Highlights**: 제안된 방법은 기존의 키워드 기반 접근법으로는 식별할 수 없었던 1,008개의 새로운 사생활 관련 리뷰를 추가로 추출하는 데 성공하였습니다. 이는 NLI와 LLM을 결합한 접근 방식의 효과성을 입증하는 결과입니다.



### Feature-Space Semantic Invariance: Enhanced OOD Detection for Open-Set Domain Generalization (https://arxiv.org/abs/2411.07392)
Comments:
          IEEE BigData 2024, Ph.D. Forum

- **What's New**: 이번 논문은 Open-set domain generalization의 복합적인 문제를 해결하기 위해 Feature-space Semantic Invariance (FSI)라는 새로운 프레임워크를 제안합니다. FSI는 다양한 도메인에서의 의미 일관성을 유지하여 unseen 도메인에서 OOD 인스턴스를 보다 정확하게 탐지할 수 있도록 합니다.

- **Technical Details**: FSI는 특성 공간에서의 의미 일관성을 강제하여 high-quality domain-invariant features를 학습하는 데 도움을 줍니다. 또한, ID 샘플을 기반으로 생성된 synthetic OOD 데이터를 활용함으로써 ID와 OOD 인스턴스 간의 결정 경계를 더욱 명확하게 설정하고 모델의 강건성을 향상시킵니다.

- **Performance Highlights**: 초기 실험 결과, ColoredMNIST 데이터셋에서 AUROC가 9.1%에서 18.9%까지 개선되었으며, ID 분류 정확도 또한 유의미하게 증가했습니다.



### Federated Learning Client Pruning for Noisy Labels (https://arxiv.org/abs/2411.07391)
- **What's New**: 이 논문은 Federated Learning(FL) 환경에서 노이즈가 있는 레이블 문제를 해결하기 위해 ClipFL(석방된 훈련 클라이언트 제외)이라는 새로운 접근 방식을 제시합니다.

- **Technical Details**: ClipFL은 세 가지 단계로 구성되어 있습니다: (1) 클라이언트의 성능을 기준으로 노이즈 클라이언트를 식별하고 Noise Candidacy Score(NCS)를 계산합니다. (2) NCS가 가장 높은 클라이언트의 일부를 제외합니다. (3) 남은 클라이언트에 대해 표준 FL을 수행하여 모델을 정제합니다.

- **Performance Highlights**: ClipFL은 다양한 데이터셋과 노이즈 수준에서 우수한 성능을 나타내어, 노이즈가 있는 클라이언트를 80% 이상의 정확도로 식별하고, 기존 FL 최적화 방법들보다 개선된 성능과 더 빠른 수렴을 보였습니다. 또한 통신 비용이 줄어드는 효과도 확인되었습니다.



### Firing Rate Models as Associative Memory: Excitatory-Inhibitory Balance for Robust Retrieva (https://arxiv.org/abs/2411.07388)
Comments:
          20 pages, 7 figures

- **What's New**: 본 논문은 firing rate 모델을 활용한 연상 기억(associative memory) 네트워크에 대한 새로운 접근 방식과 설계 방법론을 제시합니다. 이 방법론은 안정적인 평형 상태에서 기억 패턴이 재조정된 형태로 나타나도록 보장하며, 생물학적으로 그럴듯하고 강력한 시스템을 구현하기 위한 유용한 통찰을 제공합니다.

- **Technical Details**: 저자들은 synaptic matrix의 설계 및 그에 대한 지역적(local) 및 전역적(global) 안정성을 분석합니다. 새로운 알고리즘을 통해 firing rate 모델 내에 기억을 평형점으로 인코딩하는 synaptic matrix를 설계할 수 있습니다. 제안된 방법론은 자극(excitation), 억제(inhibition), 항상성(homeostasis) 조절 등의 생물학적 해석이 가능합니다.

- **Performance Highlights**: 시뮬레이션 결과에 따르면, 부정적인 항상성 강도의 선택은 매개변수 공간에서 더 넓은 안정성 영역을 생성하는 것으로 나타났습니다. 이러한 결과는 문헌에서 발견된 일반적인 항상성 항의 부호 선택과 일치합니다.



### Ensemble Learning for Microbubble Localization in Super-Resolution Ultrasound (https://arxiv.org/abs/2411.07376)
- **What's New**: 본 연구에서는 Super-resolution ultrasound (SR-US) 이미지에서 microbubble (MB) 위치 추적의 정확성을 향상시키기 위해 ensemble learning 기법을 활용하는 방법을 탐구합니다. 이로써 검출 민감도를 높이고 허위 긍정 사례를 줄이는 데 초점을 맞춥니다.

- **Technical Details**: 연구에서는 Deformable Detection Transformer (Deformable DETR) 네트워크의 in vivo 및 시뮬레이션 결과에 대한 ensemble 기법의 효과를 평가하였습니다. 제안된 구조는 다양한 탐지기들의 강점을 활용하고, 이들의 결합된 성능을 통해 MB 탐지의 정밀도를 높이고 강건성을 개선합니다.

- **Performance Highlights**: 실험을 통해 제안된 ensemble 프레임워크가 MB 검출의 정확도 및 신뢰성을 크게 향상시켰음을 보여주었습니다. 특히, precision, recall, RMSE 점수가 개선된 결과를 확인하였습니다.



### Warmstarting for Scaling Language Models (https://arxiv.org/abs/2411.07340)
- **What's New**: 본 연구는 대형 모델을 훈련하기 위한 비용을 줄이기 위해 소형 모델에서 시작하여 대형 모델을 학습하는 방법인 warmstarting을 활용하여 최적의 하이퍼파라미터를 유지할 수 있는지를 탐구합니다.

- **Technical Details**: 연구에서는 \,\mu Transfer을 활용하여 최적 하이퍼파라미터의 제로샷 전이를 이론적으로 동기화된 방법으로 적용할 수 있는 간단한 작업을 탐색합니다. 또한, warmstarting을 통한 수렴 속도 향상과 안정적인 훈련 역학의 유지 요인에 대해 조사합니다.

- **Performance Highlights**: 작은 모델의 가중치를 줄이고 제로 패딩을 적용하며, \mu Transfer로부터 확장된 초기화로 큰 모델을 섞는 방법을 통해 효과적인 warmstarting을 달성했습니다.



### Multimodal Fusion Balancing Through Game-Theoretic Regularization (https://arxiv.org/abs/2411.07335)
Comments:
          21 pages, 6 figures, 4 tables, 1 algorithm

- **What's New**: 이 논문에서는 Multimodal Competition Regularizer (MCR)이라는 새로운 손실 함수를 소개하고, 이를 통해 멀티모달 훈련에서 경쟁으로 인한 부작용을 방지할 수 있도록 설계하였습니다. MCR은 서로 다른 모달리티 간의 의존성을 분해하여 학습의 효율성을 높이고자 합니다.

- **Technical Details**: MCR은 서로 경쟁하는 모달리티가 최종 결과에 미치는 영향을 최대화하는 게임 이론적 원칙을 도입하여 두 가지 상하한을 설정합니다. 이를 통해 학습 과정에서 각 모달리티의 기여도를 조정하고,조건부 MI(Mutual Information)의 추정에 대한 잠재 공간의 변환을 제안하여 계산 효율성을 크게 개선합니다.

- **Performance Highlights**: MCR은 기존의 훈련 전략들보다 뛰어난 성능을 발휘하며, 단순한 앙상블 기준을 넘어 멀티모달 학습을 일관되게 개선하는 첫 번째 방법으로, 합성 및 실제 데이터셋에서 모두 성능 향상을 입증합니다. MCR은 특히 AVE, UCF, CREMA-D, CMU-MOSI, CMU-MOSEI 등 다양한 데이터셋에서 우수한 성능을 기록하였습니다.



### Richer Output for Richer Countries: Uncovering Geographical Disparities in Generated Stories and Travel Recommendations (https://arxiv.org/abs/2411.07320)
Comments:
          Submitted to ARR - October 2024

- **What's New**: 이번 연구는 대형 언어 모델이 지리적 지식에 대한 편향을 분석하며, 여행 추천과 지리 기반 이야기 생성의 두 가지 일반적인 시나리오를 탐구합니다. 특히, 저소득 국가에 대한 추천이 상대적으로 덜 독창적이며 빈곤과 슬픔의 감정을 더 많이 포함하고 있음을 발견했습니다.

- **Technical Details**: 연구는 ShareGPT 데이터를 기반으로 1.7%의 쿼리가 여행 추천, 1.5%가 이야기 생성에 관한 것임을 파악했습니다. 444개의 모델에서 300K 이상의 응답을 분석했으며, 이는 전 세계 150K 이상의 장소에 걸쳐 있습니다. 각 모델에 대해 평균 독창성과 감정 표현을 비교했습니다.

- **Performance Highlights**: 부유한 국가에 비해 저소득 국가에서 생성된 이야기의 65%가 더 많은 고난의 정서를 담고 있으며, 여행 추천은 평균적으로 40% 이상의 독창성 차이를 보였습니다. 이러한 결과는 현재 모델들이 서구 중심적 내용을 생성하고 있음을 나타내며, 다양한 인구 집단에 대한 서비스를 보장하기 위한 더 큰 노력이 필요함을 강조합니다.



### Harnessing Smartphone Sensors for Enhanced Road Safety: A Comprehensive Dataset and Review (https://arxiv.org/abs/2411.07315)
Comments:
          29 pages, 14 Figures, journal paper, submitted into Scientific Data Journal

- **What's New**: 이번 연구는 스마트폰 센서를 이용한 종합적인 도로 상태 및 운전 패턴 데이터셋을 소개합니다. 이 데이터셋은 기존 데이터셋보다 다양한 센서를 포함하고 있어 도로 안전성을 높이는 데 기여할 수 있습니다.

- **Technical Details**: 제안된 데이터셋은 가속도계 (accelerometer), 자이로스코프 (gyroscope), 지자기계 (magnetometer), GPS, 중력 센서, 방향 센서 및 미보정 센서 등 다양한 센서로부터 발생하는 데이터를 포함합니다. 이 센서들은 가속도 (acceleration), 중력 (gravitational) 힘, 회전 속도 (rotation rate), 자기장 강도 (magnetic field strength), 차량 속도 (vehicle speed) 등 폭넓은 매개변수를 측정하여 도로 조건과 운전 행동에 대한 깊이 있는 이해를 제공합니다.

- **Performance Highlights**: 이 데이터셋은 도로 안전, 인프라 유지보수, 교통 관리 및 도시 계획 (urban planning)을 위한 기초 자료로 활용될 수 있으며, 커뮤니티에 제공하여 협업과 연구 개발을 촉진하는 것을 목표로 합니다.



### X-DFS: Explainable Artificial Intelligence Guided Design-for-Security Solution Space Exploration (https://arxiv.org/abs/2411.07308)
- **What's New**: 이 논문은 X-DFS(Explainable - Design For Security)를 제안하여 디지털 설계의 보안 문제에 대한 효율적인 대응 방법을 제공합니다. X-DFS는 설명 가능한 인공지능(AI)을 이용해 설계 보안 솔루션 탐색을 자동화하여 시간 소모를 줄이고, 사용자가 이해할 수 있는 결정 근거를 제공합니다.

- **Technical Details**: X-DFS는 휴리스틱 기반 탐색 방법을 사용하여 다수의 Design-for-Security (DFS) 후보 인스턴스를 결정합니다. 이렇게 생성된 후보들은 설명 가능한 AI 모델을 훈련시키는 데 사용되며, (1) 공격 취약성에 대한 효율적인 완화 규칙을 생성하고, (2) 이를 자동으로 설계에 적용합니다. 이 방법은 공격 모델에 대한 방어뿐만 아니라 이해 가능한 설계 변환 규칙을 생성하는데 기여합니다.

- **Performance Highlights**: X-DFS는 세 가지 주요 로직 잠금 공격(SAIL, SWEEP, OMLA)에 대해 방어 전략을 성공적으로 학습하였으며, 효과적인 보안 전략을 명확한 규칙으로 추출하여 다른 로직 잠금 프레임워크 또는 전문가들이 활용할 수 있도록 하였습니다. 이 프레임워크는 대규모 설계에 대해 높은 효율성과 유연성을 보여주었습니다.



### Multi-hop Upstream Preemptive Traffic Signal Control with Deep Reinforcement Learning (https://arxiv.org/abs/2411.07271)
Comments:
          5 tables, 12 figures. arXiv admin note: text overlap with arXiv:2409.00753

- **What's New**: 본 논문은 마르코프 체인 이론(Markov chain theory)을 기반으로 하는 새로운 개념인 다중 홉 업스트림 압력(multi-hop upstream pressure)을 소개하여 기존의 단기적 트래픽 신호 제어 방식의 한계를 극복하고자 한다. 이 접근 방식은 현재 위치에서 즉각적인 링크만 고려하는 대신, 더 넓은 범위의 트래픽 조건을 반영하여 신호 타이밍을 최적화할 수 있도록 한다.

- **Technical Details**: 다중 홉 업스트림 압력 개념은 주변 환경의 트래픽 조건을 종합적으로 고려하여 트래픽 신호 제어의 효과성을 높인다. 이를 위해 강화학습(deep reinforcement learning) 에이전트는 현재 대기열을 미리 정리하도록 안내되며, 시뮬레이션 결과는 이 새로운 지표가 네트워크의 전체 지연을 감소시키는 데 효과적임을 보여준다.

- **Performance Highlights**: 시뮬레이션(예: 토론토 시나리오) 결과, 다중 홉 업스트림 압력을 활용하는 컨트롤러는 넓은 범위의 선행 혼잡을 이해하여 트래픽 흐름을 우선시함으로써 전체 네트워크 지연을 상당히 줄이는 것으로 나타났다.



### Learning From Graph-Structured Data: Addressing Design Issues and Exploring Practical Applications in Graph Representation Learning (https://arxiv.org/abs/2411.07269)
Comments:
          arXiv admin note: text overlap with arXiv:2205.11691, arXiv:2304.14621

- **What's New**: 이번 논문에서는 그래프 표현 학습(Graph Representation Learning) 및 그래프 신경망(Graph Neural Networks, GNNs)의 최신 발전에 대한 포괄적인 리뷰를 제공합니다. 특히, 복잡한 노드 상호작용을 포착할 수 있는 고차 풀링(high-order pooling) 함수를 갖춘 GNN을 소개하며, 이는 노드 및 그래프 수준의 작업에서 GNN의 효능을 크게 향상시킵니다. 또한, GNN을 기반으로 한 분자 그래프 생성 모델도 제안합니다.

- **Technical Details**: GNN은 그래프 구조 데이터를 처리하도록 설계된 신경망으로, 여러 단계의 메시지 전파를 통해 이웃으로부터 정보를 집계하여 노드 표현을 반복적으로 업데이트합니다. 본 연구에서는 대칭 텐서 분해(symmetric tensor decomposition)를 기반으로 한 집계 함수를 설계하여 비선형 고차 곱셈 상호작용을 모델링하고, 이는 수치적으로 효율적인 방식으로 비가변(permutation-invariant) 멀티선형 함수들을 처리할 수 있습니다. 마지막으로, CP 레이어(CANDECOMP/PARAFAC decomposition)는 그래프의 전체 표현을 계산할 수 있게 해 주며, 이를 통해 비선형 고차 상호작용을 효과적으로 모델링 할 수 있습니다.

- **Performance Highlights**: 신뢰할 수 있는 방법들과의 철저한 실험 평가 및 비교를 통해, 제안된 모델들이 다양한 데이터셋을 사용하여 여러 실제 문제를 해결하는 데 있어 뛰어난 성능을 보임을 입증했습니다. 특히, GNN은 분자 그래프 생성 작업에 강력한 기반 모델로 활용되어, 화합물의 구조를 정확히 재현할 수 있음을 보여 주었습니다.



### Target-driven Attack for Large Language Models (https://arxiv.org/abs/2411.07268)
Comments:
          12 pages, 7 figures. arXiv admin note: substantial text overlap with arXiv:2404.07234

- **What's New**: 본 논문에서는 기존의 블랙박스 공격 방법의 한계를 극복하기 위해 새로운 목표 기반 블랙박스 공격 기법을 제안했습니다. 이 기법은 클린 텍스트와 공격 텍스트 간의 KL divergence를 최대화하여 공격의 목표를 재정의합니다.

- **Technical Details**: 제안된 방법은 공격 목표에 따라 두 개의 볼록 최적화 문제로 거리 최대화 문제를 변환하며, 이는 프로젝트 경량 하강 알고리즘을 활용하여 공격 텍스트에 해당하는 벡터를 해결하는 데 사용됩니다. 또한, 필요한 두 가지 공격 전략인 토큰 조작(token manipulation)과 잘못된 정보 공격(misinformation attack)을 포함합니다.

- **Performance Highlights**: 다양한 대규모 언어 모델(LLM)과 데이터셋에 대한 실험 결과, 제안된 공격 방식은 기존 방법보다 효과적인 성과를 보였습니다. 이로 인해 모델의 보안과 강건성 향상에 기여할 수 있을 것으로 기대됩니다.



### A Survey on Data Markets (https://arxiv.org/abs/2411.07267)
- **What's New**: 이 논문은 21세기 데이터 시장의 발전과 그 가치 극대화를 위한 다양한 측면을 포괄적으로 조사하고 있습니다. 특히, 데이터 제품화(data productization), 데이터 거래(data transaction), 가격 책정(data pricing), 수익 분배(revenue allocation) 및 개인정보 보호, 보안(security), 신뢰(trust) 문제 등을 다루고 있습니다.

- **Technical Details**: 데이터 시장(data market)은 데이터 구매자(data buyers)와 데이터 판매자(data sellers)가 직접 또는 중개(agent)를 통해 교류하여 데이터를 거래하는 모든 메커니즘을 의미합니다. 이 시장은 데이터의 가격(picing) 및 분배(distribution)와 같은 여러 기능이 상호작용하여 데이터의 가치를 최대한 활용하고 증대시키는 조정 메커니즘으로 작용합니다.

- **Performance Highlights**: 논문은 여러 국가 및 다양한 분야의 데이터 시장에 대한 정부 정책 및 산업 현황을 연구하고 있으며, 해결되지 않은 도전 과제를 식별하고 데이터 시장의 발전을 위한 미래 방향에 대한 논의를 포함합니다.



### High quality ECG dataset based on MIT-BIH recordings for improved heartbeats classification (https://arxiv.org/abs/2411.07252)
Comments:
          4 pages, 5 figures, 5 tables, presented during IEEE COINS 2023 Berlin. link to ieeexploere: this https URL

- **What's New**: 본 논문에서는 MIT-BIH 기록을 기반으로 하여 새로운 고품질 심박수 데이터세트를 생성하는 방법론을 제안합니다. 이 방법론은 이상치를 제거하고 10초 창(window)에서 평균 값을 계산하여 최적의 심박수 크기를 산출하는 방식으로, 연속된 심박수 혼합 문제를 피할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법론은 IQR 방법론을 사용하여 이상 심박수를 제거하며, 현재 10초 창에서 RR 시간 간격의 평균 값에 따라 적응적인 심박수 크기를 계산합니다. 이는 QRS 중심의 심박수를 생성하게 하여 연속적인 심박수 혼합 문제를 해결합니다. 연구팀은 또한 1-D ResNet 아키텍처 모델을 개발하여 새로운 데이터세트의 성능을 평가하였습니다.

- **Performance Highlights**: 개발된 1-D ResNet 모델은 99.24%의 정확도를 달성하였으며, 이는 기존 방법에 비해 5.7% 향상된 수치입니다. 데이터세트를 다운샘플링하여 모델의 실행 시간을 33% 단축하고 메모리 사용량도 3배 감소시켰습니다.



### Navigating AI in Social Work and Beyond: A Multidisciplinary Review (https://arxiv.org/abs/2411.07245)
Comments:
          30 pages

- **What's New**: 이번 리뷰는 사회복지 직종이 인공지능(AI)과 어떻게 상호작용하며 그 영향을 받는지에 대한 간단한 논평 작성으로 시작되었습니다. 그러나 AI의 심오한 영향력을 충분히 포착하기 위해 더 깊이 있는 탐구가 필요하다는 것이 빠르게 드러났습니다.

- **Technical Details**: 리뷰는 AI를 광범위한 사회적 및 학문적 논의에 배치하며, 2025년이 다가오고 있도록 기술 기업의 선두주자, 문화 아이콘, CEO, 정치인의 시각과 수학, 사회학, 철학, 경제학 등 다양한 분야의 AI 엔지니어 및 혁신가들의 기여를 아우르는 내용을 포함하고 있습니다.

- **Performance Highlights**: 또한, 리뷰는 AI의 현실 세계에 미치는 영향, 윤리적 도전 과제 및 사회복지에 대한 함의를 간략히 분석합니다. 특히, AI를 활용한 고급 개인화 시뮬레이션 훈련(Advanced Personalised Simulation Training, APST)을 통해 사회복지 교육에서 혁신적인 변화를 가져올 수 있는 비전을 제시합니다.



### A Tutorial on Teaching Data Analytics with Generative AI (https://arxiv.org/abs/2411.07244)
- **What's New**: 이 튜토리얼은 데이터 분석 수업에서 ChatGPT와 같은 대규모 언어 모델(LLMs)을 통합하는 도전 과제를 다룹니다. AI를 활용하여 수업 내외에서 여러 가지 새로운 교수 기법을 소개하고, 학생들이 서로 다른 맞춤형 GPT와 상호작용하여 분석의 다양한 부분을 학습하고 서로 가르치는 방법을 제안합니다.

- **Technical Details**: 학생들은 자연어로 데이터 변환을 표현하고 AI를 사용하여 해당 코드를 생성하는 'Programming in English' (PIE) 방법을 통해 데이터 작업을 보다 효과적으로 수행할 수 있습니다. 이러한 접근법은 AI와의 협업을 학습하는 데 중점을 두며, 각 학생이 AI에 대한 지식과 기술을 공유하는 방법을 제안합니다.  또한, AI 튜터링 세션과 같은 새로운 과제 설계를 통해 학생들의 참여와 학습을 증진시킬 수 있습니다.

- **Performance Highlights**: 강의 종료 후 설문조사에 따르면, 필수 과목의 63%와 MBA 학생들의 86%가 AI 통합 수업에 대해 7점 중 5점 이상으로 긍정적인 반응을 보였습니다. 이는 AI가 데이터 분석 교육에 새로운 패러다임을 열었다는 것을 의미합니다.



### Neuropsychology and Explainability of AI: A Distributional Approach to the Relationship Between Activation Similarity of Neural Categories in Synthetic Cognition (https://arxiv.org/abs/2411.07243)
- **What's New**: 이 논문은 인공 신경망의 설명 가능성에 대한 신경심리학적 접근법을 제안하며, 인간 인지 심리학의 개념을 바탕으로 인공 신경망의 행동을 해석하는 새로운 방법론을 개발하고자 합니다.

- **Technical Details**: 신경망의 동작을 설명하는 과정에서 인간의 인지 종류 중 카테고리화(categorization)와 유사성(similarity) 개념이 핵심 요소로 사용됩니다. 이 연구에서는 주어진 입력 벡터 공간 내에서 카테고리 세그먼트를 생성하는 방식과 노드 간의 상호작용을 분석하여 인공 신경망의 내부 인지 체계를 이해하려고 합니다.

- **Performance Highlights**: 이 연구의 결과는 인공지능 모델이 인간의 사고 방식과 일치하도록 조정할 수 있는 방법을 제시하며, 특히 언어 모델의 성능을 향상시키기 위해 설명 가능성에 대한 세밀한 분석이 필요함을 강조합니다. 또한, 카테고리 지식을 형성하고 적용하는 과정을 더 깊이 이해할 수 있는 기회를 제공합니다.



New uploads on arXiv(cs.LG)

### LLMPhy: Complex Physical Reasoning Using Large Language Models and World Models (https://arxiv.org/abs/2411.08027)
- **What's New**: 본 논문에서는 로봇이 실제 세계에서 작업할 때 필요한 물리적 추론(physical reasoning) 능력을 향상시키기 위한 새로운 과제와 데이터셋인 TraySim을 제안합니다. 이 방법은 대형 언어 모델(LLMs)과 물리 엔진을 결합하여, 복잡한 물체의 동역학을 예측하는 것을 목표로 합니다.

- **Technical Details**: TraySim은 여러 물체가 놓인 쟁반에서 외부 충격에 대한 동역학을 예측하는 과제로 구성되어 있습니다. LLMPhy는 LLM의 프로그램 합성 능력을 활용하여 물리적 하이퍼파라미터를 추정하는 제로샷(Zero-shot) 최적화 프레임워크입니다. 이 프레임워크는 비분화가능한 시뮬레이터와 상호작용하여 물리적 특성을 상상하는 데 사용됩니다.

- **Performance Highlights**: LLMPhy는 TraySim 데이터셋에서 실험을 통해 기존의 블랙 박스 최적화 방법들에 비해 우수한 성능을 입증하였으며, 물리적 파라미터 추정의 정확도 또한 크게 향상되었습니다. 또한 LLMPhy의 성능은 OpenAI o1-preview와 같은 최신 LLM에서 더욱 두드러진 최적화 수렴 경향을 보여주었습니다.



### Exact, Tractable Gauss-Newton Optimization in Deep Reversible Architectures Reveal Poor Generalization (https://arxiv.org/abs/2411.07979)
Comments:
          Accepted at NeurIPS 2024

- **What's New**: 본 논문에서는 딥 리버서블 아키텍처에서 정확한 Gauss-Newton (GN) 업데이트의 가능성을 처음으로 보여줍니다. 이 연구는 실제 데이터셋에서 GN 최적화 기법의 훈련 및 일반화 특성을 조사합니다.

- **Technical Details**: 딥 리버서블 네트워크에서 Jacobian의 일반화된 역을 분석적으로 유도하였으며, 이를 통해 GN 업데이트를 정확하고 신속하게 수행할 수 있는 방법을 제시합니다. 또한, 이 모델은 NTK(신경탄젠트핵)가 훈련 중 비특이적(non-singular)인 경우에 해당합니다.

- **Performance Highlights**: GN 최적화 기법은 훈련 손실에서 과적합을 초래하며, 미니 배치 설정에서 일반화 능력이 부족한 현상을 보였습니다. 또한, 연구 결과는 GN 업데이트가 '게으른' 레짐(lazy regime)으로 남아있어 모델의 표현력이 초기 상태와 유사하게 유지된다는 점을 강조합니다.



### Sleep Staging from Airflow Signals Using Fourier Approximations of Persistence Curves (https://arxiv.org/abs/2411.07964)
- **What's New**: 이번 연구에서는 비침습적인 방법으로 수면을 단계별로 분류하는 새로운 접근 방식을 제안합니다. 기존의 electroencephalogram(EEG) 신호 대신, 호흡 신호를 사용하여 수면 단계를 분류하는 자동화된 알고리즘을 개발하였습니다.

- **Technical Details**: 본 연구에서는 Fourier Approximations of Persistence Curves (FAPC)를 도입하여, 기존의 Hermite function Expansions of Persistence Curves (HEPC)보다 더 나은 성능을 기대할 수 있는 방법론을 제시합니다. 1155개의 소아 환자 수면 연구를 바탕으로 XGBoost 모델을 사용하여 성능을 분석하였습니다.

- **Performance Highlights**: FAPC 방법을 적용하여 기존 방법에 비해 4.9% 향상된 성능을 달성하였습니다. 연구 결과는 호흡 신호를 포함한 비침습적인 방법이 수면 스테이징에서 EEG 데이터 없이도 효과적임을 지원합니다.



### On the Convergence of Continual Federated Learning Using Incrementally Aggregated Gradients (https://arxiv.org/abs/2411.07959)
- **What's New**: 이번 연구에서는 Continual Federated Learning (CFL) 시스템의 효율성 향상을 위한 새로운 접근법인 C-FLAG (Continual Federated Learning with Aggregated Gradients)를 제안합니다. 이 방법은 메모리 기반의 리플레이(replay) 전략을 사용하여 글로벌 모델에서 발생할 수 있는 catastrophic forgetting 문제를 해결하고, 학습 속도가 
$O(1/
\sqrt{T})$로 수렴함을 보입니다.

- **Technical Details**: C-FLAG는 메모리 버퍼에서의 gradient 업데이트와 현재 데이터에서의 aggregated gradient를 통해 작동합니다. 이를 통해 모델은 과거 작업의 경험을 적절히 활용하면서 새로운 작업에 대한 이해를 지속적으로 확장합니다. 또 다른 특징은, catastrophic forgetting을 최소화하기 위한 최적화 하위 문제를 공식화하여, 각 작업 간의 원활한 학습을 보장하는 adaptive learning rates를 사용하는 것입니다.

- **Performance Highlights**: 실험 결과 C-FLAG는 task-incremental FL 설정에서 기존의 여러 최첨단 기법들에 비해 일관되게 더 높은 정확도와 forget률 감소 성과를 보였습니다. 또한, 데이터 이질성, 클라이언트 수, 리플레이 버퍼의 크기 및 유형에 대한 ablation 연구를 통해 C-FLAG의 강력한 성능을 입증하였습니다.



### Learning Memory Mechanisms for Decision Making through Demonstrations (https://arxiv.org/abs/2411.07954)
- **What's New**: 본 연구에서는 Partially Observable Markov Decision Processes (POMDPs)에서 의사결정 과정에 메모리를 통합하는 새로운 접근법으로, memory dependency pair (p,q)를 도입하여 과거 이벤트가 의사결정에 미치는 영향을 모델링합니다.

- **Technical Details**: memory dependency pairs는 특정 시간 p에서의 이벤트가 시간 q의 의사결정에 회상되어야 함을 나타냅니다. 이를 강화하기 위해 self-attention 메커니즘에 손실을 적용하여 Transformer 아키텍처에서 메모리 메커니즘을 통합합니다. AttentionTuner는 이러한 memory dependency pairs를 활용하여 기존 Transformer에 비해 성능을 개선합니다.

- **Performance Highlights**: AttentionTuner는 Memory Gym 및 Long-term Memory Benchmark (LTMB)에서 다양한 작업에서 뛰어난 성공률을 달성하였으며, 기존 Transformer 접근 방식에 비해 일반화 능력을 향상시켰습니다. 실험 결과, 최소 0.1%의 주석이 달린 데모로도 학습 성능이 향상될 수 있음을 보여줍니다.



### Doubly Mild Generalization for Offline Reinforcement Learning (https://arxiv.org/abs/2411.07934)
Comments:
          Accepted to NeurIPS 2024. arXiv admin note: substantial text overlap with arXiv:2410.19400

- **What's New**: 이 논문은 오프라인 강화 학습(Offline Reinforcement Learning)에서 약간의 일반화(mild generalization)를 통해 성능을 향상시킬 수 있는 가능성을 탐구합니다. 이를 위해 새로운 방법인 Doubly Mild Generalization (DMG)을 제안합니다.

- **Technical Details**: DMG는 (i) mild action generalization과 (ii) mild generalization propagation으로 구성됩니다. 첫 번째는 데이터셋의 근처에서 Q 값을 최대화하기 위해 행동을 선택하는 것을 의미합니다. 두 번째는 RL 학습 신호의 전파를 방해하지 않으면서 일반화 전파를 줄이는 방법입니다.

- **Performance Highlights**: DMG는 Gym-MuJoCo locomotion 작업과 AntMaze 작업에서 최첨단 성능을 달성하였으며, 오프라인에서 온라인 학습으로 매끄럽게 전환할 수 있는 유연성을 가지고 있습니다.



### A Stochastic Optimization Framework for Private and Fair Learning From Decentralized Data (https://arxiv.org/abs/2411.07889)
- **What's New**: 이 논문에서는 개인 정보 보호 및 공정성을 고려한 새로운 연합 학습(federated learning) 알고리즘을 개발하였습니다. 이 알고리즘은 다양한 인구 통계 그룹에 대해 공정한 결정을 내릴 수 있도록 설계되었습니다.

- **Technical Details**: 제안한 알고리즘은 inter-silo record-level differential privacy (ISRL-DP)를 만족하며, 이는 각 사일로에서 전송된 메시지가 모든 사일로의 데이터 프라이버시를 보장하도록 요구합니다. 또한, 손실 함수에 대한 부드러움 조건 하에 수렴성을 보장합니다.

- **Performance Highlights**: 실험 결과, 이 알고리즘은 기존의 최첨단 기술보다 공정성과 정확도의 균형(tradeoff)에서 현저한 개선을 이루었으며, 특정 정확도 수준에서 공정성 위반이 95% 감소했습니다. 또한, ISRL-DP 기준 아래에서 비선형 강한 볼록성이 요구되지 않는 최초의 수렴 보장을 제공합니다.



### Diverse capability and scaling of diffusion and auto-regressive models when learning abstract rules (https://arxiv.org/abs/2411.07873)
Comments:
          12 pages, 5 figures. Accepted to NeurIPS2024 Workshop on System 2 Reasoning At Scale as long paper

- **What's New**: 본 논문에서는 현대 생성 모델이 유한 샘플로부터 기본 규칙을 학습하고 조건적 샘플링을 통해 추론할 수 있는지를 조사합니다. 이를 위해 Raven's Progressive Matrices 작업에서 영감을 받아 GenRAVEN 데이터세트를 설계하였습니다.

- **Technical Details**: GenRAVEN 데이터세트는 40개의 관계 규칙을 기반으로 하며, 각 샘플은 3행으로 구성됩니다. 생성 모델은 diffusion 모델(EDM, DiT, SiT)과 autoregressive 모델(GPT2, Mamba) 두 그룹으로 훈련되었습니다. 다양한 데이터 스케일에서의 성능을 비교하였으며, 각 모델의 샘플 생성 능력을 평가했습니다.

- **Performance Highlights**: diffusion 모델은 기본적인 생성에서 뛰어난 성능을 보여주었고, 새로운 샘플을 더 일관되게 생성할 수 있었습니다. 반면, autoregressive 모델은 누락된 패널을 규칙 일관성 있게 완성하는 데 강점을 보였으나, 기본 생성에서는 일관성이 떨어졌습니다. 데이터 규모에 따른 다양한 성능 변화를 관찰하였고, 앞으로의 연구 방향에 대한 통찰을 제공합니다.



### Evidential time-to-event prediction model with well-calibrated uncertainty estimation (https://arxiv.org/abs/2411.07853)
- **What's New**: 본 논문에서는 시간-사건 예측(Task of Time-to-Event Prediction)을 위한 새로운 증거 기반 회귀 모델을 제안하고, 이 모델은 Gaussian Random Fuzzy Numbers (GRFNs)를 활용하여 사건 발생 시간을 정량화합니다. 또한, 기존의 분포 가정에 얽매이지 않고 데이터의 복잡성을 처리할 수 있는 유연성을 제공합니다.

- **Technical Details**: 제안된 모델은 데이터의 절단(censoring) 문제를 해결하기 위해 일반화된 음의 로그 우도 함수(Generalized Negative Log-Likelihood Function)를 최소화하여 적합을 수행합니다. 이 모델은 알레이터리(aleatory)와 에피스테믹(epistemic) 불확실성을 각각 정량화하여 보다 상세한 임상 의사결정 지침을 제공합니다.

- **Performance Highlights**: 실험 결과, 본 모델은 다양한 데이터 분포와 절단 시나리오에서 높은 정확성과 신뢰성을 보이며, 기존의 최첨단 방법들을 초월하는 성능을 발휘했습니다.



### FRUGAL: Memory-Efficient Optimization by Reducing State Overhead for Scalable Training (https://arxiv.org/abs/2411.07837)
- **What's New**: 이번 논문에서는 메모리 효율적인 최적화 프레임워크인 FRUGAL (Full-Rank Updates with Gradient splitting)을 소개합니다. FRUGAL은 low-dimensional update를 수행하기 위해 gradient splitting을 활용하며, 기존의 low-rank 알고리즘과 통합하여 높은 차원의 업데이트를 가능케 합니다.

- **Technical Details**: FRUGAL은 Adam과 같은 고급 최적화 알고리즘을 상태가 있는 subspace에 사용하고, SGD나 signSGD와 같은 상태가 없는 알고리즘을 보조적으로 사용하여 업데이트를 실행합니다. 이 프레임워크는 다양한 state-full 및 state-free 최적화 알고리즘을 지원하며, 이러한 구조가 이론적 수렴 보장을 제공합니다.

- **Performance Highlights**: FRUGAL은 다양한 메모리 예산 하에서 기존의 메모리 효율적인 알고리즘보다 우수한 성능을 보여주며, LLaMA와 RoBERTa 모델의 사전 훈련 및 파인 튜닝 작업에서 최첨단 결과를 달성했습니다. 특히, transformer 모델의 Logits layer는 Adam과 같은 고급 최적화를 필요로 하지만, 다른 구성 요소는 signSGD와 같은 단순한 방법으로도 충분한 성능을 유지할 수 있음을 보여줍니다.



### Dynamical-VAE-based Hindsight to Learn the Causal Dynamics of Factored-POMDPs (https://arxiv.org/abs/2411.07832)
- **What's New**: 이 논문은 Partially Observable Markov Decision Processes (POMDPs)에서 부분적인 관측치로부터 환경의 동적 특성을 학습하는 새로운 방법을 제시합니다. 특히, 미래 정보를 활용하여 인과적 동역학(causal dynamics)을 정확히 캡처하고 상태 표현(state representations)을 개선하는 것이 중요하다는 점에 중점을 둡니다.

- **Technical Details**: 제안된 Dynamical Variational Auto-Encoder (DVAE)는 POMDP 내 오프라인 경로(offline trajectories)로부터 인과적 마르코프 동역학을 학습하도록 설계되었습니다. 이 방법은 과거, 현재 및 다단계 미래 정보를 통합하는 확장된 힌트(hindsight) 프레임워크를 활용하여 factored-POMDP 설정 내에서 동작합니다.

- **Performance Highlights**: 경험적 결과는 이 접근 방식이 역사 기반 모델 및 전형적인 힌트 기반 모델보다 숨겨진 상태 전환을 지배하는 인과 그래프(causal graph)를 더 효과적으로 발견함을 보여줍니다.



### Suite-IN: Aggregating Motion Features from Apple Suite for Robust Inertial Navigation (https://arxiv.org/abs/2411.07828)
- **What's New**: 웨어러블 (wearable) 장치에서 수집된 IMU 데이터를 활용하여 보행자 위치 추정 (pedestrian localization) 성능 향상을 목표로 한 새로운 다중 장치 딥러닝 프레임워크 Suite-IN을 소개합니다.

- **Technical Details**: 이 프레임워크는 로컬 (local) 및 글로벌 (global) 움직임 정보를 결합하여 의미 있는 글로벌 움직임 특징을 추출하기 위해 대조 학습 (contrastive learning) 모듈을 적용합니다. Apple Suite (iPhone, Apple Watch, AirPods)를 사용하여 소비자 등급 IMU 데이터의 실용성을 제고합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기존 최첨단 (state-of-the-art) 접근 방식에 비해 보다 강력하고 정확한 보행자 위치 추정을 제공하며, 다양한 동작 패턴을 효과적으로 처리하였습니다.



### Efficient Federated Finetuning of Tiny Transformers with Resource-Constrained Devices (https://arxiv.org/abs/2411.07826)
- **What's New**: 이번 연구에서는 미리 훈련된 딥러닝 모델을 활용하여 자원이 제한된 크로스 디바이스 페더레이티드 러닝(Federated Learning, FL)을 위한 새로운 레이어 파인튜닝 기법을 제안하였습니다. LoRA 기법이 여전히 매개변수 효율적이지만 메모리와 FLOPs 면에서 비효율적임을 관찰하고, 그에 따라 더 나은 성능을 제공하는 방법을 개발했습니다.

- **Technical Details**: 제안된 기법은 크로스 디바이스 FL에서 미리 훈련된 신경망을 사용하며 주어진 자원 제약을 준수합니다. 레이어 파인튜닝 기법을 통해 메모리 사용을 최적화하고, LoRA와 비교하여 계산 비용을 감소시킵니다. 실험에서는 소형 모델을 사용하여 동질적 및 이질적 환경에서 성능을 평가하였으며, 기존의 FL 방법들보다 우수한 결과를 기록했습니다.

- **Performance Highlights**: 제안된 레이어 파인튜닝 기법은 LoRA 기반 기법 및 최신 FL 방법들과 비교하여 동질적 및 이질적 계산 및 메모리 제약을 처리하는 데 있어 월등한 성능을 나타냈습니다. 또한, 통신이 제한적인 환경에서도 높은 정확도를 달성하여 FL 교육 효율성을 크게 향상시켰습니다.



### Dual-Criterion Model Aggregation in Federated Learning: Balancing Data Quantity and Quality (https://arxiv.org/abs/2411.07816)
Comments:
          6 pages

- **What's New**: 이 연구는 기존의 평범한 평균 집계 알고리즘의 한계를 극복하기 위해 데이터의 양과 질을 모두 고려하는 새로운 이중 기준 가중 집계 알고리즘을 제안합니다.

- **Technical Details**: 이중 기준 가중 집계 알고리즘은 클라이언트 노드에서 훈련에 사용된 데이터의 양과 질을 정량화하고, 특정 데이터셋에서 여러 번의 로컬 모델 추론 정확도 평가를 진행하여 각 클라이언트의 데이터 질을 평가합니다. 이 두 가지 요소를 가중치로 사용하여 최적화된 집계 과정이 이루어집니다.

- **Performance Highlights**: 제안된 알고리즘은 일반 목적으로 사용되는 오픈 소스 데이터셋인 CIFAR-10과 시각적 장애물 회피 관련 데이터셋에서 기존의 여러 최첨단 집계 접근법보다 뛰어난 성능을 보였습니다.



### Federated Low-Rank Adaptation with Differential Privacy over Wireless Networks (https://arxiv.org/abs/2411.07806)
Comments:
          6 pages, 3 figures, submitted to IEEE ICC 2025

- **What's New**: 이 논문에서는 분산 엣지 디바이스에서 대규모 사전 훈련된 모델의 파인 튜닝 중 발생하는 계산 및 개인정보 보호 문제를 해결하기 위해, 차등 개인정보 보호(differential privacy, DP)를 결합한 분할 페더레이티드 파인 튜닝(split FedFT) 프레임워크를 제안합니다. 이를 통해 원시 데이터를 공유하지 않고도 협력적인 모델 훈련이 가능하며, 엣지 디바이스의 컴퓨팅 부담을 줄일 수 있습니다.

- **Technical Details**: 제안된 분할 FedFT 프레임워크는 LoRA(저순위 적응)와 페더레이티드 학습을 결합하여 파라미터 효율적인 파인 튜닝을 가능하게 합니다. 이 프레임워크는 엣지 디바이스와 중앙 서버 간에 모델의 구성 요소를 분산시켜, 엣지 디바이스에서 전체 모델을 배포할 필요성을 줄입니다. 특히, 무선 네트워크의 내재된 채널 노이즈를 활용하여 DP 보장을 제공하며, 인공적인 노이즈를 추가할 필요가 없습니다.

- **Performance Highlights**: 시뮬레이션 결과에 따르면, 제안된 프레임워크는 엄격한 개인정보 보호 기준 하에서도 기존 방법들에 비해 더 높은 정확도를 달성하였습니다. 특별히, 한 개의 저순위 행렬만 업데이트함으로써 노이즈 증폭 효과를 완화시켜 우수한 성능을 보여줍니다.



### Kernel-based retrieval models for hyperspectral image data optimized with Kernel Flows (https://arxiv.org/abs/2411.07800)
- **What's New**: 이 논문에서는 Kernel Flows (KF)를 활용하여 Kernel Principal Component Regression (K-PCR)을 최적화하는 새로운 접근 방식을 제안합니다. KF는 Kernel Partial Least-Squares (K-PLS) 회귀에 대한 이전 연구를 확장한 것으로, 두 개의 하이퍼스펙트럴 원격 탐지 데이터세트를 이용하여 성능을 평가합니다.

- **Technical Details**: Kernel Flows (KF) 방법은 교차 검증(cross-validation) 접근 방식을 통해 데이터 오버피팅(overfitting)을 줄이고, 초기 값에 관계없이 올바른 파라미터로 수렴하며, 지역 최소점(local minima) 대신 전역 최적값(global optimum value)을 달성하는 데 도움을 줍니다. 연구에서 제안된 KF 기반 K-PCR 방법은 두 가지 하이퍼스펙트럴 데이터인 토양 수분 소프트 센서와 식물 특성 모델을 사용하여 설명됩니다.

- **Performance Highlights**: KF-K-PCR과 KF-PLS 방법의 성능은 비선형 회귀(non-linear regression) 기술과 비교되었으며, 공하 대조군을 통해 각 방법의 우수성을 평가했습니다. 이 연구는 하이퍼스펙트럴 데이터의 효율적인 활용을 통해 환경 모니터링의 가능성을 확장하고자 합니다.



### Interaction Asymmetry: A General Principle for Learning Composable Abstractions (https://arxiv.org/abs/2411.07784)
Comments:
          Preprint, under review

- **What's New**: 이 연구에서는 개념의 분리된 표현(Disentangled Representations) 학습을 위한 새로운 원리인 상호작용 비대칭(Interaction Asymmetry)을 제안했습니다. 이 원리는 동일한 개념의 부분들이 서로 다른 개념의 부분들보다 더 복잡한 상호작용을 갖는다고 설명합니다.

- **Technical Details**: 우리는 이 원리를 개념을 관측된 데이터로 변환하는 생성기(Generator)의 $(n+1)$차 미분에 대한 블록 대각 조건(Block Diagonality Conditions)을 통해 정 형화합니다. 여기서 서로 다른 '복잡성'의 차수는 서로 다른 $n$에 대응합니다.

- **Performance Highlights**: 합성 이미지 데이터셋에서는 제안된 모델이 더 명시적인 객체 중심 선행 조건을 사용하는 기존 모델과 비교해 유사한 객체 분리 성능을 달성할 수 있음을 입증하였습니다.



### Automatic Album Sequencing (https://arxiv.org/abs/2411.07772)
Comments:
          presented as a late breaking demo in the 25th International Society for Music Information Retrieval Conference; 3 pages in main text, 3 figures in main text; source code available at this https URL

- **What's New**: 앨범 시퀀싱(album sequencing) 과정에서 사용자 친화적인 웹 기반 도구가 도입되었습니다. 이 도구를 통해 비전문가도 쉽게 음악 트랙을 업로드하고, 한 번의 클릭으로 시퀀싱 기법을 실행하여 결과를 시각화할 수 있습니다.

- **Technical Details**: 이 연구에서는 Transformer를 기반으로 한 새로운 앨범 시퀀싱 방법을 소개합니다. 이는 이전 연구의 복잡한 파이프라인을 단순화하여 하나의 모델로 대체하였으며, 알고리즘은 FMA 데이터셋을 기반으로 하여 두 층의 완전 연결 신경망과 두 층의 인코더-디코더 Transformer 모델을 사용합니다.

- **Performance Highlights**: 새로운 방법은 무작위 베이스라인(random baseline)보다 뛰어난 성능을 보였지만, 이전의 내러티브 본질(narrative essence) 접근법에는 미치지 못했습니다. 이 연구의 모든 구현은 공개적으로 제공되며, 비전문가도 사용할 수 있는 사용자 인터페이스가 제공됩니다.



### ASER: Activation Smoothing and Error Reconstruction for Large Language Model Quantization (https://arxiv.org/abs/2411.07762)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 양자화(Quantization)를 다루며, 특히 Post-Training Quantization 중의 오류 분포를 분석합니다. 새로운 알고리즘 ASER(Activation Smoothing and Error Reconstruction)가 소개되어, 양자화 오류를 보정하는 방법을 제시합니다.

- **Technical Details**: ASER는 (1) 오류 재구성(Error Reconstruction)과 (2) 활성화 부드럽게 하기(Activation Smoothing)를 포함합니다. 오류 재구성에서는 LoRA 스타일 행렬을 이용한 저계수 보정을, 활성화 부드럽게 하기는 이상치 분석을 통해 수행됩니다.

- **Performance Highlights**: ASER는 일반적인 LLM을 저비트로 양자화하는 데 효과적이며, W4A8 스킴에서 정확도를 유지할 수 있습니다. 실험 결과 ASER는 최첨단 양자화 알고리즘들과 비교해 경쟁력이 있음이 입증되었습니다.



### Navigation with QPHIL: Quantizing Planner for Hierarchical Implicit Q-Learning (https://arxiv.org/abs/2411.07760)
Comments:
          Under review. Code will be released upon acceptance

- **What's New**: 이번 연구에서는 Offline Reinforcement Learning (RL)의 새로운 접근 방식인 QPHIL을 제안합니다. QPHIL은 사전 기록된 시연을 활용하여 상태 공간의 이산 표현 및 시간적으로 일관된 표현을 학습합니다.

- **Technical Details**: QPHIL은 계층적 구조를 가진 목표 조건 강화 학습 방법입니다. 학습된 양자화기(quantizer)를 통해 공간을 분할하고, 이로 인해 하위 목표(subgoal) 계획이 단순화됩니다. 이 접근 방식은 복잡한 경로 추적(low-level path following)과 경로 계획(high-level path planning)을 분리하여 효율적인 환경 내 탐색이 가능하게 합니다.

- **Performance Highlights**: QPHIL은 복잡한 장거리 탐색 환경에서 최첨단 성능을 달성하였으며, 인간-영감을 받은 방식으로 하위 정책을 최종 목표로 유도합니다. 이 방법은 장기 탐색 벤치마크에서 뛰어난 결과를 보여주며, 오프라인 RL 설정에서의 계획 및 디지털화의 향후 조치를 위한 유망한 방향을 제시합니다.



### Spatially Regularized Graph Attention Autoencoder Framework for Detecting Rainfall Extremes (https://arxiv.org/abs/2411.07753)
- **What's New**: 본 연구에서는 인도의 시공간 강수 데이터를 활용하여 스케일러블한 이상 감지를 위한 새로운 Graph Attention Autoencoder (GAE)를 소개합니다. 이 모델은 Graph Attention Network (GAT)를 사용하여 데이터의 공간적 의존성과 시간적 동적성을 포착하며, 지리적 일관성을 보장하는 공간 정규화 항을 결합합니다.

- **Technical Details**: Spatially Regularized Graph Attention Autoencoder (SRGAttAE) 모델은 4827개의 노드와 75,000-85,000개의 엣지를 포함하는 그래프 구조에서 공간적 의존성을 효과적으로 감지하도록 설계되었습니다. 모델은 인코딩 및 디코딩 단계로 나뉘며, 각 단계는 두 개의 GAT 레이어로 구성되어 있습니다.

- **Performance Highlights**: SRGAttAE 모델은 인도 전체에서 이상 강수 패턴을 효과적으로 식별함을 보여주었으며, 기후 변화에 대한 준비와 대응 전략을 개선하는 데 기여할 것으로 기대됩니다.



### Exploring the loss landscape of regularized neural networks via convex duality (https://arxiv.org/abs/2411.07729)
- **What's New**: 이 논문은 정규화된 신경망의 손실 경관(loss landscape)에 대한 여러 측면을 논의합니다. 특히, 정적 점(stationary points)의 구조, 최적 솔루션(optimal solutions)의 연결성(connectivity), 손실이 감소하는 경로(path with nonincreasing loss) 및 최적 솔루션의 비유일성(nonuniqueness)에 대해 깊이 있는 탐구를 합니다.

- **Technical Details**: 이 연구는 두 개의 층으로 구성된 신경망에서 시작하여, 문제를 동등한 볼록 문제(convex problem)로 변환하고, 그 이중 문제(dual)를 고려하여 손실 경관을 분석합니다. 이 과정에서 볼록 문제의 해 집합(solution set)을 특성화하고, 모든 정적 점을 정의합니다. 또한 신경망의 폭(width)이 변할 때, 전역 최적(global optima)의 위상(topology)이 상전이(phase transition)를 경험한다는 것을 보여줍니다.

- **Performance Highlights**: 논문은 일반적인 두 층의 벡터 값 신경망(two-layer vector-valued neural networks)과 병렬된 세 층의 신경망(parallel three-layer neural networks)과 같은 다양한 구조에 대해서도 솔루션 집합의 특성화 및 연결성 결과가 확장될 수 있음을 제시하고 있습니다. 이로 인해 최적 솔루션의 연속적인 존재에 대한 반례(counterexamples)를 구성하여, 최적 솔루션의 다수성을 강조합니다.



### Convergence Rate Analysis of LION (https://arxiv.org/abs/2411.07724)
- **What's New**: LION(evoLved sIgn mOmeNtum) 최적화 기법이 대규모 신경망 훈련에서 인상적인 성능을 보인다는 연구 결과가 발표되었습니다. 이 논문은 LION의 수렴 속도와 관련된 포괄적인 분석을 제공합니다.

- **Technical Details**: LION은 Karush-Kuhn-Tucker(KKT) 포인트로의 수렴을 보이며, 이 속도는 O(√d K^{-1/4})로 측정됩니다. 여기서 d는 문제의 차원, K는 반복 단계의 수입니다. 제약조건을 제거한 후에도 같은 속도로 일반 비제약 문제의 임계점까지 수렴합니다.

- **Performance Highlights**: LION은 기존의 SGD(Standard Gradient Descent)보다 더 낮은 손실과 높은 성능을 달성하며, 실험적으로는 gradient ℓ₁/ℓ₂ 노름 비율이 Θ(√d)에 맞춰져 있음을 확인하였습니다.



### OWLed: Outlier-weighed Layerwise Pruning for Efficient Autonomous Driving Framework (https://arxiv.org/abs/2411.07711)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 본 논문에서는 자율 주행 시스템에 효과적인 LLM(대규모 언어 모델) 지원 프레임워크인 OWLed를 제안합니다. OWLed는 아울라이어(Outlier)를 중량화하여 레이어별 희소성을 적용하여 모델의 크기를 줄이고, 고급 추론 기능을 유지하면서 자율 주행의 효율성을 크게 향상시킵니다.

- **Technical Details**: OWLed는 비균일 레이어별 희소 비율을 적용하여 아울라이어 기능의 분포에 따라 다양한 레이어에 비균형적인 희소성을 할당합니다. 이 방법은 모델 압축 과정에서 드라이빙 환경 데이터를 포함시킴으로써 자율 주행 작업에 잘 적응할 수 있도록 합니다.

- **Performance Highlights**: OWLed는 지각(perception), 행동 예측(action prediction), 언어 이해(language understanding)에서 기존 방법보다 우수한 성능을 보이며, 상당히 낮은 계산 요구 사항을 유지합니다. 논문에서는 OWLed가 복잡한 시나리오를 처리할 수 있는 효율적이고 강력한 자율 주행 시스템 개발의 가능성을 강조합니다.



### Test Where Decisions Matter: Importance-driven Testing for Deep Reinforcement Learning (https://arxiv.org/abs/2411.07700)
- **What's New**: 이 논문에서는 Deep Reinforcement Learning (RL)에서 정책의 결정이 안전성과 성능에 미치는 영향을 평가하기 위해 새로운 방식의 모델 기반 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 상태 공간의 모든 상태에서 상태 중요성이랭킹을 엄격하게 계산합니다. 테스트 프레임워크는 각 반복(iteration)에서 낙관적(optimistic) 및 비관적(pessimistic) 안전 추정치를 계산하여 정책 실행에 대한 기대 결과의 하한과 상한을 제공합니다. 정책의 약점을 명확히 하기 위해 상태 공간을 안전한 영역과 안전하지 않은 영역으로 나눕니다.

- **Performance Highlights**: 여러 사례에 대한 자세한 평가 결과를 통해, 제안된 방법이 적은 테스트 노력으로 안전하지 않은 정책 행동을 발견할 수 있음을 보여주었습니다.



### What Do Learning Dynamics Reveal About Generalization in LLM Reasoning? (https://arxiv.org/abs/2411.07681)
- **What's New**: 이 논문에서는 현대 대형 언어 모델(LLMs)의 문제 해결 능력에 대한 기전이 여전히 불투명함을 밝히고, 학습 동역학(fin tuning)과 하향식 일반화(downstream generalization) 사이의 관계를 탐구합니다. 특히, 사전 기억 훈련 정확도(pre-memorization train accuracy)라는 새로운 훈련 지표를 제안하며, 이를 통해 모델의 일반화 행동을 효과적으로 예측할 수 있음을 보여줍니다.

- **Technical Details**: 연구는 수학적 추론 문제를 중심으로 진행되며, 각 모델의 훈련 중 정확도와 최종 해결책의 정확도를 비교하여 일반화 가능성을 분석합니다. 사전 기억 훈련 정확도는 훈련 데이터의 정확한 기억 단계에 도달하기 전의 모델 샘플의 정확도를 측정하며, 이로 인해 테스트 성과와의 높은 상관관계를 보입니다. 다양한 모델(Llama3 8B, Gemma2 9B) 및 데이터셋(GSM8k, MATH)에서 R^2 값이 0.9 이상에 달하는 성과를 기록했습니다.

- **Performance Highlights**: 사전 기억 훈련 정확도는 개별 모델 예측의 강건성(robustness)을 나타내는 지표로 활용되며, 정확도가 낮은 훈련 사례에 소량의 변경을 가하면 예측 성능이 크게 저하되는 반면, 높은 정확도를 가진 사례의 예측은 안정적입니다. 데이터 선별(data curation) 과정에서 이 지표를 활용하여 낮은 사전 기억 정확도를 가진 예제를 우선적으로 선택하면, 샘플 효율성이 1.5-2배 향상되는 결과를 보였으며, 기존의 선별 기술보다 더 나은 성과를 보였습니다.



### Safe Exploitative Play with Untrusted Type Beliefs (https://arxiv.org/abs/2411.07679)
Comments:
          26 pages, NeurIPS 2024

- **What's New**: 이번 연구에서는 다수의 에이전트가 존재하는 게임에서 타입 예측의 정확성이 에이전트의 이익(payoff)에 미치는 영향을 분석합니다. 이상적인 타입 믿음과 실제 타입 믿음 간의 차이를 정량화하여 위험(risk)과 기회(opportunity) 간의 균형을 탐구합니다.

- **Technical Details**: Bayesian 게임(Bayesian games)에서는 에이전트가 다른 에이전트의 유형에 대한 믿음을 형성하고, 관찰된 행동에 따라 이 믿음을 업데이트하며, 이를 바탕으로 자신의 행동을 선택합니다. 본 연구에서는 Bayesian 학습(Bayesian learning)을 기반으로 하는 다양한 알고리즘을 검토하며, 잘못된 믿음이 에이전트의 성과에 미치는 영향을 분석합니다.

- **Performance Highlights**: 연구 결과는 정상형(normal-form) 및 확률적(stochastic) Bayesian 게임에서의 위험과 기회 간의 무역오프(tradeoff)를 형성하는 상한선과 하한선을 제시합니다. 이로써 에이전트가 최적의 전략을 사용했을 때와 비교하여 잠재적인 손실과 이익을 평가할 수 있는 근거를 제공합니다.



### Rethinking Structure Learning For Graph Neural Networks (https://arxiv.org/abs/2411.07672)
- **What's New**: 본 논문은 Graph Structure Learning (GSL)의 효과를 재평가하고, GNN(Graph Neural Networks) 성능 향상에 대한 기여를 분석합니다. 이를 위해 새로운 GSL 프레임워크를 제안하며, GSL의 기초 구축 및 구조 생성 과정을 3단계로 나눕니다.

- **Technical Details**: 제안된 GSL 프레임워크는 (1) GSL 기초(GSL bases) 생성, (2) 새로운 구조 생성, (3) 뷰 융합(view fusion) 세 가지 단계로 구성됩니다. GSL 기초는 고정 또는 학습 가능한 매개변수를 갖는 그래프 인식 또는 비인식 모델로 처리된 노드 임베딩을 의미하며, 새로운 구조는 유사성 기반, 구조 기반, 최적화 기반 접근 방식을 사용하여 만듭니다. 최후에, 여러 GSL 그래프를 융합하기 위해 다양한 뷰 융합 전략이 적용됩니다.

- **Performance Highlights**: 실험 결과, GSL이 GNN 성능을 일관되게 향상시키지 못하고, GSL 기초가 GNN 성능을 높이는 주요 요소임을 확인했습니다. 대부분의 GSL 방법이 불필요하다는 점과, 동일한 하이퍼파라미터 튜닝 설정 하에서도 모델 성능에 큰 차이가 없음을 보여주어 GSL의 필요성을 재조명합니다.



### Is Graph Convolution Always Beneficial For Every Feature? (https://arxiv.org/abs/2411.07663)
- **What's New**: 본 논문에서는 Graph Neural Networks (GNNs)의 기능을 향상시키기 위해 Topological Feature Informativeness (TFI)라는 새로운 메트릭을 도입하고 이를 기반으로 한 Graph Feature Selection (GFS) 방법을 제안합니다. GFS는 GNN에 유리한 특성과 불리한 특성을 구분하여 각각 적합한 모델로 처리함으로써 성능을 크게 향상시킵니다.

- **Technical Details**: TFI는 집합된 노드 특징과 레이블 간의 상호정보(mutual information)를 측정하여 GNN이 선호하는 특성과 비선호하는 특성을 식별합니다. GFS는 TFI를 활용해 GNN에 유리한 특성은 GNN을 통해, GNN에 불리한 특성은 Multi-Layer Perceptrons (MLPs)로 처리한 후 두 모델의 임베딩을 결합하여 최종 노드 표현을 생성합니다.

- **Performance Highlights**: GFS를 사용하여 8개의 기준 및 최첨단 GNN 아키텍처를 10개의 데이터셋에 적용한 결과, 83.75%의 경우에서 성능 향상이 나타났습니다. 또한 TFI는 기존의 다른 특성 선택 방법들에 비해 우수한 성능을 보였습니다. 이 연구는 GFS와 TFI의 효과성을 실증적으로 검증하며, GFS가 하이퍼파라미터 조정에 대해 강건하다는 점을 강조합니다.



### Top-$n\sigma$: Not All Logits Are You Need (https://arxiv.org/abs/2411.07641)
- **What's New**: 이 논문에서는 LLMs에서의 표준 샘플링 기법인 greedy decoding 및 저온 샘플링의 관례에 도전하며, top-nσ라는 새로운 샘플링 기법을 소개합니다. 이 방법은 pre-softmax logits를 기반으로 하여 통계적 임계값을 이용하여 작동합니다.

- **Technical Details**: top-nσ 메소드는 logit 분포에서 Gaussian 분포와 정보성 있는 영역을 구분하며, 복잡한 확률 조작 없이도 효율적인 토큰 필터링을 가능하게 합니다. 이 알고리즘은 고온에서도 안정적인 샘플링 공간을 유지하여 test-time scaling 기술과의 통합이 용이합니다. 이 방법은 정렬 또는 추가적인 softmax 변환이 필요하지 않아 계산 효율성 또한 높습니다.

- **Performance Highlights**: 네 개의 추론 중심 데이터셋에서 우리의 방법은 기존의 샘플링 접근 방식뿐만 아니라 greedy decoding을 초월하는 성능을 보여줍니다. 특히, 높은 온도에서도 일관된 성능을 유지하며, 생성 품질의 유의미한 향상을 입증했습니다.



### Circuit Complexity Bounds for RoPE-based Transformer Architectur (https://arxiv.org/abs/2411.07602)
- **What's New**: 이번 연구는 Rotary Position Embedding (RoPE)을 사용하는 Transformer 아키텍처의 표현력을 제한하는 엄격한 회로 복잡도 경계(circuit complexity bounds)를 설정합니다. 이 연구를 통해 RoPE 기반 Transformer의 근본적인 한계를 이론적으로 밝혀냈습니다.

- **Technical Details**: RoPE (Rotation Position Embedding)는 절대 및 상대 위치 정보를 인코딩하여 Transformer의 성능을 향상시키는 기술입니다. 연구에서는 RoPE 기반 아키텍처의 각 구성 요소에 대한 회로 복잡도를 체계적으로 조사하였고, 이 모델들이 TC⁰ 회로들로 시뮬레이션될 수 있음을 증명했습니다. 또한, TC⁰ = NC¹이 아닌 경우 poly(n) 정밀도와 O(1) 레이어, 그리고 d ≤ O(n) 조건 하에서 RoPE 기반 Transformer가 산술 문제 또는 부울 포뮬라 값 문제를 해결할 수 없음을 보여주었습니다.

- **Performance Highlights**: RoPE 기반 Transformer는 일반 Transformer 모델에 비해 더 높은 일반화 능력을 나타내며, 긴 컨텍스트 정보 처리에서 우수한 성능을 발휘합니다. 최근의 실험적 결과들은 RoPE가 긴 문서 요약 및 지속적인 대화 등 긴 컨텍스트 작업에서 탁월한 능력을 발휘함을 보여주고 있습니다.



### Entropy Controllable Direct Preference Optimization (https://arxiv.org/abs/2411.07595)
- **What's New**: 본 연구에서는 Direct Preference Optimization (DPO) 방식의 수정인 H-DPO를 제안합니다. H-DPO는 정책의 엔트로피를 조정할 수 있어 분포의 선명도를 높이고, 효과적인 mode-seeking fitting을 가능하게 합니다.

- **Technical Details**: H-DPO는 기존 DPO의 손실 계산 수식을 단순히 수정하여 엔트로피 조정을 통해 성능을 향상시킬 수 있도록 설계되었습니다. 이 접근법에서는 손실 함수의 정규화 항을 변경하여 분포의 엔트로피 H(π)를 직접 제어하게 됩니다.

- **Performance Highlights**: 실험 결과, H-DPO는 다양한 작업에서 DPO보다 우수한 성능을 보여주었으며, 특히 수학적 과제의 pass@$k$ 평가에서 더욱 뛰어난 성과를 나타냈습니다.



### Overcoming the Curse of Dimensionality in Reinforcement Learning Through Approximate Factorization (https://arxiv.org/abs/2411.07591)
Comments:
          61 pages, 10 figures

- **What's New**: 이번 연구는 고차원 문제에서 발생하는 차원의 저주(curse of dimensionality)를 해결하기 위해 MDP(Markov Decision Process)를 약한 분해(approximate factorization)하여 더 작은 MDP로 나누는 접근 방식을 제안합니다. 이 방법은 실제 환경의 의존 구조를 활용하여 샘플 효율성을 개선할 수 있습니다.

- **Technical Details**: 이 논문에서는 원래의 MDP를 약하게 분해하여 고차원 구성 요소로 분해하는 새로운 방식을 제안합니다. 이 과정에서 개발된 모델 기반 RL 알고리즘은 동기식 샘플링을 사용하여 더 낮은 차원의 구조를 활용하고 문제에 따라 샘플 복잡도를 개선합니다. 또한, 모델 프리(model-free) 알고리즘인 분산 감소 Q-learning(variance-reduced Q-learning, VRQL-AF)도 개발하여 샘플 복잡도를 보장합니다.

- **Performance Highlights**: 제안된 알고리즘은 모델 기반 및 모델 프리 환경 모두에서 기존의 최적 샘플 복잡도를 초과하는 성능을 보여줍니다. 특히, VRQL-AF는 기존의 MDP에 대해 초기 성과와 비교하여 준 최적 샘플 복잡도 보장을 달성하며, 여러 구성 요소를 통한 통계적 상관관계를 관리하여 샘플 복잡도를 크게 줄이는 방법을 제시하였습니다.



### Disentangling Tabular Data towards Better One-Class Anomaly Detection (https://arxiv.org/abs/2411.07574)
- **What's New**: 이 논문은 Tabular anomaly detection(표 형태의 이상 탐지)에서의 one-class classification(단일 클래스 분류) 접근법을 발전시킵니다. 기존의 learnable mask 전략으로는 mask가 균일하게 생성될 위험이 있었는데, 이 문제를 해결하기 위해 두 개의 non-overlapping(비포괄적) 그리고 correlated(상호연관된) attribute subsets(속성 집합)인 CorrSets를 도입하여 이상을 효과적으로 탐지합니다.

- **Technical Details**: 이 연구에서 제안된 새로운 방법론인 Disent-AD는 two-head self-attention module을 활용하여 latent space(잠재 공간)에서 두 개의 CorrSets를 효과적으로 분리합니다. 이 과정에서 attention maps(어텐션 맵)을 통해 속성의 상관관계를 포착하고, reconstruction task(재구성 작업)를 통해 원본 데이터를 복원합니다. 이를 통해 model은 normal samples(정상 샘플) 간의 내부 상관관계를 잘 학습합니다.

- **Performance Highlights**: 20개의 표 데이터셋에서 진행된 실험 결과, 이 방법은 AUC-PR(Area Under the Curve - Precision-Recall)에서 평균 6.1% 및 AUC-ROC(Area Under the Curve - Receiver Operating Characteristic)에서 평균 2.1% 성능 향상을 보여주며, 최신 기법들을 크게 초월하는 성능을 입증했습니다.



### Zer0-Jack: A Memory-efficient Gradient-based Jailbreaking Method for Black-box Multi-modal Large Language Models (https://arxiv.org/abs/2411.07559)
Comments:
          Accepted to Neurips SafeGenAi Workshop 2024

- **What's New**: 본 논문에서는 Zer0-Jack이라는 새로운 방법을 제안합니다. 이 방법은 Multi-modal Large Language Models (MLLMs)의 안전 시스템을 우회하기 위해 zeroth-order optimization을 활용하며, 기존의 white-box 접근 방식 없이도 작동합니다.

- **Technical Details**: Zer0-Jack은 patch coordinate descent를 사용하여 블랙박스 MLLMs를 직접 공격할 수 있는 악의적인 이미지 입력을 효율적으로 생성하며, 이에 따라 메모리 사용량을 줄입니다. 이 방법은 고차원 입력에서 높은 추정 오류 문제를 완화하여 특정 이미지 부분만 최적화합니다.

- **Performance Highlights**: Zer0-Jack은 다양한 모델에서 높은 공격 성공률을 달성하였으며, MiniGPT-4에 대해 95%의 성공률을 기록했습니다. 또한, 상업용 MLLMs인 GPT-4o를 직접 공격할 수 있는 가능성을 보여줍니다.



### Unraveling the Gradient Descent Dynamics of Transformers (https://arxiv.org/abs/2411.07538)
- **What's New**: 이 연구는 Transformer 아키텍처의 최적화 동태를 이론적으로 분석하고, Gradient Descent (GD)를 통한 수렴 보장에 필요한 조건을 제시했습니다. 특히 Softmax와 Gaussian 주의 커널을 비교함으로써, 각 커널 유형의 성능 차이를 규명했습니다.

- **Technical Details**: 본 연구는 Softmax와 Gaussian attention 커널을 사용하여 Transformer 모델의 손실 경량을 분석하였습니다. 연구 결과, 적절한 가중치 초기화와 더불어 입력 임베딩 차원이 큰 경우 GD를 통해 Transformer 모델이 전역 최적 솔루션을 달성할 수 있음을 보여줍니다. 그러나 Softmax attention 커널이 지역 최적 해에 수렴할 수 있는 위험이 있음을 지적했습니다.

- **Performance Highlights**: 경험적으로 Gaussian attention 커널이 손실이 0에 수렴하는 반면, Softmax 커널은 지역 최적 해로 빠지는 경우가 있음을 발견했습니다. 성능 비교 결과, Softmax attention을 사용하는 Transformer가 Gaussian 커널보다 느리게 수렴하고 더 도전적인 학습 환경을 제공함을 확인했습니다.



### Accident Impact Prediction based on a deep convolutional and recurrent neural network mod (https://arxiv.org/abs/2411.07537)
Comments:
          28 pages, 18 figures

- **What's New**: 이 연구에서는 교통사고로 인한 영향을 예측하기 위해 'cascade model'이라는 새로운 딥러닝 모델을 제안합니다. 이 모델은 Los Angeles 카운티의 실시간 데이터를 활용하여 사고 발생 후의 영향을 예측하고, 기존 모델의 두 가지 주요 단점을 해결합니다.

- **Technical Details**: 제안된 모델은 두 가지 구성요소인 Long Short-Term Memory (LSTM)와 Convolutional Neural Network (CNN)으로 이루어져 있습니다. LSTM은 시간적 패턴을 포착하고, CNN은 스파스한 사고 데이터에서 패턴을 추출합니다. 또한, 외부 교통 혼잡 데이터셋을 포함하여 '사고 영향'이라는 새로운 특성을 도출합니다.

- **Performance Highlights**: 실험 결과, 제안하는 하이브리드 기계 학습 방법이 현재 최첨단 모델들에 비해 사고 발생 후의 영향을 예측하는 데 있어 더 높은 정밀도와 재현율을 보였습니다. 특히, 사고가 보고되지 않은 최소 영향의 경우 더 높은 정밀도를, 더 큰 영향을 가진 사고의 경우 더 높은 재현율을 달성했습니다.



### Model Stealing for Any Low-Rank Language Mod (https://arxiv.org/abs/2411.07536)
- **What's New**: 이 논문에서는 모델 스틸링(Model Stealing)의 이론적 기초를 탐구하며, Hidden Markov Models (HMMs) 및 저차원 언어 모델(low-rank language models)의 도용을 위한 효율적인 알고리즘을 제시합니다.

- **Technical Details**: 저자들은 조건부 쿼리 모델(conditional query model)을 활용하여 저차원 확률 분포를 학습하는 알고리즘을 개발했습니다. 이 알고리즘은 특정 시점에서의 조건부 분포를 나타내기 위해 바리센트릭 스파너(barycentric spanners)를 구성하고, 오류를 방지하기 위해 상대 엔트로피(relative entropy)를 포함하는 볼록 최적화(convex optimization) 문제를 반복적으로 해결합니다.

- **Performance Highlights**: 이 연구는 Kakade et al.의 이전 결과를 개선하여, 높은 '신뢰도'가 필요한 복잡한 조건을 제거하고, 저차원 출력 분포를 가진 모든 언어 모델을 도용할 수 있게 되었습니다. 이론적으로, ML 모델이 추론 시 더 복잡한 문제를 해결할 수 있도록 하는 것이 성능 향상에 기여할 수 있다는 흥미로운 예시로 작용합니다.



### Collaborative and Federated Black-box Optimization: A Bayesian Optimization Perspectiv (https://arxiv.org/abs/2411.07523)
- **What's New**: 이번 연구는 협업적이고 분산된 블랙박스 최적화(collaborative and federated black-box optimization, BBOpt)에 대한 새로운 접근 방식을 제안합니다. 특히, 이 연구는 Bayesian optimization 관점에서 분산 실험, 이질성(heterogeneity), 프라이버시(privay) 관련 문제를 해결하기 위한 세 가지 통합 프레임워크를 제시합니다.

- **Technical Details**: 제시된 세 가지 프레임워크는 다음과 같습니다: (i) 중앙 조정(global framework)에서 실험이 진행되는 프레임워크, (ii) 최소한의 공유 정보에 기반하여 결정할 수 있는 로컬 프레임워크(local framework), (iii) 지역 대리모델을 협업을 통해 개선하여 의사결정을 향상시키는 예측 프레임워크(predictive framework)입니다. 연구는 기존 방법들을 이러한 프레임워크 내에서 분류하며, BBOpt의 잠재력을 완전히 발휘하기 위해 해결해야 할 주요 질문을 강조합니다.

- **Performance Highlights**: 연구는 에이전트들이 협력하여 실험 노력을 분산시키는 BBOpt의 장점을 효율적으로 활용할 수 있는 가능성을 제시합니다. 다른 에이전트의 원시 데이터를 공유하지 않고도 최적 설계를 신속하게 찾을 수 있으며, 이로 인해 trial & error의 비용과 시간을 대폭 줄일 수 있는 잠재력을 가지고 있습니다.



### Bayesian Deep Learning Approach for Real-time Lane-based Arrival Curve Reconstruction at Intersection using License Plate Recognition Data (https://arxiv.org/abs/2411.07515)
Comments:
          accepted by T-ITS

- **What's New**: 본 연구는 차량의 차선 선택 패턴과 불확실성을 모두 특징짓는 Bayesian 딥러닝 접근법을 제안하여, 실시간 차선 기반 도착 곡선의 재구성을 가능하게 합니다. 이는 기존의 연구들이 다루지 않은 실시간 링크 기반 도착 차선 선택을 정량화합니다.

- **Technical Details**: 본 접근법은 부분적으로 관찰된 링크 기반 도착과 차선 기반 도착 간의 관계를 효과적으로 포착하여, 이를 차선 선택 비율로 해석할 수 있도록 설계되었습니다. Bayesian 파라미터 추론 기법을 사용하여 차선 선택의 불확실성을 모델링하고, LPR 데이터 매칭 비율이 낮은 조건에서도 도착 곡선 재구성의 불확실성을 최소화합니다.

- **Performance Highlights**: 실제 데이터셋을 바탕으로 수행된 실험 결과는 제안된 접근법이 기존의 방법들에 비해 우수하고, 차선 선택 모델링의 필요성을 강조하는 결과를 보여줍니다.



### Robust Offline Reinforcement Learning for Non-Markovian Decision Processes (https://arxiv.org/abs/2411.07514)
- **What's New**: 이 논문은 비마르코프(Non-Markovian) 강화 학습 분야에서 불확실성 집합(uncertainty set) 내의 최악의 환경에서 최적 정책을 학습하는 새로운 접근법을 제안합니다. 특히, 낮은 순위 구조(low-rank structure)를 가진 명목 모델을 사용하는 오프라인 강화 학습 알고리즘을 개발하였습니다.

- **Technical Details**: 제안된 알고리즘은 두 가지 유형의 불확실성 집합인 𝒯-type과 𝒫-type을 고려하여, 새로운 데이터셋 증류(dataset distillation) 기법과 Robust 값의 하한 신뢰 구간(lower confidence bound, LCB) 설계를 포함합니다. 또한, 오프라인 저랭크 비마르코프 의사결정 과정에 맞춘 새 유형의 concentrability coefficient 를 도입하여 알고리즘의 샘플 효율성을 보장합니다.

- **Performance Highlights**: 본 연구의 알고리즘은 O(1/ε^2) 개의 오프라인 샘플을 사용해 ε-최적(ε-optimal) 강인 정책을 찾을 수 있으며, 일반적인 비마르코프 RL에서도 폴리노미얼샘플 효율성을 제공합니다. 논문에서 제안한 조건이 충족된다면 샘플 수가 유한한 경우에도 Near-optimal Robust policy 를 효율적으로 배울 수 있음을 보여주었습니다.



### FM-TS: Flow Matching for Time Series Generation (https://arxiv.org/abs/2411.07506)
- **What's New**: FM-TS라는 새로운 Time Series generation 프레임워크가 소개되었습니다. 이 프레임워크는 Flow Matching에 기반하여 시간 경과에 따른 데이터의 생성 과정을 간소화하고, Conditional 및 Unconditional 설정 모두에서 최적화할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: FM-TS는 Rectified Flow Matching을 이용하여 시간 series 생성의 효율성과 성능을 높입니다. 이 모델은 반복적인 샘플링이나 복잡한 노이즈 일정이 필요하지 않으며, 단일 프레임 전달을 통해 샘플링을 수행합니다. 특히, 본 연구는 Rectified Flow Matching을 시간 series 생성에 처음 적용한 사례입니다.

- **Performance Highlights**: FM-TS는 다양한 데이터셋에 대해 기존 방법보다 우수한 성능을 보이며, Sines, Stocks, ETTh, MuJoCo, Energy, fMRI와 같은 데이터셋에서 차별적인 점수(discriminative score)를 달성했습니다. 예를 들어, Stock 데이터셋에서는 0.019, ETTh에서는 0.011의 점수를 기록하며 이전 최고의 성능인 0.067을 크게 초과하였습니다. 또한, Solar forecasting 및 MuJoCo imputation 과제에서도 성능이 크게 향상되었습니다.



### LAUREL: Learned Augmented Residual Layer (https://arxiv.org/abs/2411.07501)
Comments:
          Accepted at the 2nd Efficient Systems for Foundation Models Workshop at the International Conference on Machine Learning (ICML) 2024

- **What's New**: 이 논문에서는 전통적인 잔여 연결(residual connection)의 일반화된 형태인 학습된 증강 잔여 레이어(Learned Augmented Residual Layer, LAuReL)를 소개합니다. LAuReL은 모델 품질과 메모리 사용량 모두에서 기존 방법을 초월하는 것을 목표로 합니다.

- **Technical Details**: LAuReL의 주요 아이디어는 잔여 연결을 다음과 같이 재구성하는 것입니다: α는 학습된 스칼라 매개변수이며, g(⋅)는 학습된 선형 함수입니다. 이 함수는 잔여 연결의 출력을 입력으로 사용하여 더 복잡한 정보 흐름을 형성합니다. LAuReL은 모델의 크기와 지연(latency) 측면에서 경량화된 방식으로 이러한 잔여 흐름을 학습할 수 있습니다.

- **Performance Highlights**: ResNet-50과 ImageNet 1K 작업에서 LAuReL은 추가 레이어를 추가했을 때의 성능 향상 중 60%를 달성하면서도 파라미터 수는 0.003%만 증가했습니다. 이는 LAuReL이 적은 파라미터로도 높은 성능을 보장한다는 것을 보여줍니다.



### Enhancing Link Prediction with Fuzzy Graph Attention Networks and Dynamic Negative Sampling (https://arxiv.org/abs/2411.07482)
- **What's New**: 본 논문에서는 전통적인 Graph Neural Networks(GNNs)가 무작위 negative sampling에 의존하는 한계를 보완하기 위해 Fuzzy Graph Attention Networks(FGAT)를 제안합니다. 이 접근법은 fuzzy rough set을 통합하여 동적인 negative sampling과 향상된 노드 특징 집계를 가능하게 합니다.

- **Technical Details**: 본 연구에서는 Fuzzy Negative Sampling(FNS)을 통해 fuzzy 유사도 기반의 고품질 negative edges 선택하는 메커니즘을 도입합니다. FGAT 레이어는 fuzzy rough set 원리를 통합하여 강력하고 구분 가능한 노드 표현을 가능하게 합니다. 이를 통해 GNN의 전반적인 학습 효율성을 높이고 있습니다.

- **Performance Highlights**: 실험 결과, FGAT는 두 개의 연구 협력 네트워크에서 기존의 최첨단 방법들보다 우수한 링크 예측 정확도를 보여주었습니다. 특히, fuzzy rough set의 힘을 활용하여 효과적인 negative sampling과 노드 특징 학습을 구현함으로써 성능이 개선되었습니다.



### Machines and Mathematical Mutations: Using GNNs to Characterize Quiver Mutation Classes (https://arxiv.org/abs/2411.07467)
- **What's New**: 본 논문에서는 기계 학습(machine learning)을 활용하여 quiver mutation(퀴버 변형)의 동등성 기준을 탐구합니다. 이는 cluster algebras(군집 대수) 이론에 중앙으로 연결되며, 이전에는 알려지지 않은 $	ilde{D}_n$ 유형의 quiver에 대한 결과를 제공합니다.

- **Technical Details**: 연구에서는 약 70,000개의 quiver 데이터를 사용하고, graph neural networks(GNN)를 훈련시킵니다. 모델의 예측 정확도를 높이는 과정에서, D 유형 quiver와 관련된 알려진 조건들을 확인합니다. 이와 함께 D 유형 quiver의 새로운 변형 동등성을 규명하는 정리를 증명하는 데 기여합니다.

- **Performance Highlights**: 모델은 높은 정확도를 기록할 뿐만 아니라, 수학적 이론과 일치하는 구조를 모델의 숨겨진 표현에서 포착합니다. 이는 현대 기계 학습 모델이 수학적 데이터로부터 추상적이며 일반적인 규칙을 학습할 수 있다는 주요 증거 중 하나입니다.



### Fast unsupervised ground metric learning with tree-Wasserstein distanc (https://arxiv.org/abs/2411.07432)
- **What's New**: 본 연구는 Wasserstein singular vectors (WSV)를 사용하여 비지도 학습 방식으로 지표(metric)를 학습하는 새로운 방법을 제안합니다. 기존 WSV 방법의 복잡도를 개선하여 $	ext{O}(n^3)$의 계산 복잡도로 Tree-Wasserstein distance (TWD)를 계산합니다.

- **Technical Details**: 제안된 방법은 샘플과 특징을 트리 구조에 임베딩하여 TWD를 계산하는 것으로, 이를 통해 WSV 접근법보다 더 나은 근사치를 제공합니다. 기존의 방법이 $	ext{O}(n^5)$의 복잡도를 요구하는 반면, 본 알고리즘은 $	ext{O}(n^3)$ 복잡도로 더 효율적입니다.

- **Performance Highlights**: 본 연구는 여러 단일 세포 RNA 시퀀싱 유전체 데이터셋에서 TWD 알고리즘을 활용하여 비지도 셀 타입 클러스터링 문제에 대한 확장성 및 유용성을 입증하였습니다.



### Predicting BWR Criticality with Data-Driven Machine Learning Mod (https://arxiv.org/abs/2411.07425)
- **What's New**: 이 논문은 수조(boiling water) 원자로에서 과도한 비판성을(data-driven deep learning model) 추정하는 혁신적인 방법을 제안합니다.

- **Technical Details**: 기존 방법과는 달리, 이 접근법은 머신러닝 기법을 활용하여 원자로 주기(cycle) 동안 필요한 연료량을 정확히 결정하는 데 도움을 줍니다. 이 연구에서는 원자로의 비판 성능(critical performance) 예측을 위한 데이터를 분석하여 훈련된 딥러닝(dedep learning) 모델을 사용합니다.

- **Performance Highlights**: 이 방법은 원자로가 주기의 끝까지 비판 상태를 유지하도록 필요한 연료량을 최적화함으로써, 조기 감속(coastdown) 및 부족한 연료로 인한 비용 손실을 방지할 수 있습니다.



### Comparing Targeting Strategies for Maximizing Social Welfare with Limited Resources (https://arxiv.org/abs/2411.07414)
- **What's New**: 이 논문은 리소스가 제한된 개입에서 머신러닝을 이용해 개입을 받을 이들을 선택하는 방법에 대한 새로운 분석을 제공합니다. 기존에 사용되던 '위험 기반(targeting based on risk)' 접근법이 실제로 효과적인지 검토하며, biased한 치료 효과 추정치를 활용하는 것이 더 나은 접근법임을 보입니다.

- **Technical Details**: 이 연구는 5개의 RCT (Randomized Controlled Trials) 데이터를 사용하여 위험 기반 타겟팅과 비대칭적인 치료 효과 추정치 기반 타겟팅을 비교합니다. 머신러닝 기법을 활용해 치료 효과의 이질성을 더 잘 학습하는 것이 어렵고, 이는 고해상도 데이터가 필요하기 때문입니다. 연구는 정책 입안자들이 효율적으로 자원을 할당하는 데 있어 실용적인 통찰력을 제공합니다.

- **Performance Highlights**: 연구 결과, 위험 기반 타겟팅이 치료 효과에 대한 비의도치적 추정치에 비해 거의 항상 효과적이지 않다는 것을 발견했습니다. 특히, 정책 입안자가 불평등 완화 선호를 가질 때도 치료 효과 기반 타겟팅이 더 유리한 결론을 도출했습니다.



### ODEStream: A Buffer-Free Online Learning Framework with ODE-based Adaptor for Streaming Time Series Forecasting (https://arxiv.org/abs/2411.07413)
- **What's New**: 이번 논문에서는 streaming time series 데이터 처리의 제한을 극복하기 위한 ODEStream이라는 새로운 모델을 소개했습니다. 이는 복잡한 프레임워크 없이도 변화에 잘 적응할 수 있는 능력을 가지고 있으며, ODE(Ordinary Differential Equations) 방식을 활용하여 메모리 없는 온라인 예측을 가능하게 합니다.

- **Technical Details**: ODEStream은 버퍼링 없이 temporal isolation layer를 포함한 지속적인 학습 프레임워크로, 데이터 내의 시간적 의존성을 통합합니다. 이 방법은 불규칙하게 샘플링된 시퀀스를 처리하고 연속적인 데이터 표현을 생성할 수 있는 기능을 가지고 있으며, 데이터 스트리밍 시나리오에서 변화하는 동적에 무리 없이 적응할 수 있습니다.

- **Performance Highlights**: 다양한 벤치마크 실세계 데이터 세트에 대한 평가 결과, ODEStream은 최신 온라인 학습 및 스트리밍 분석 기준을 능가하며, 긴 시간 동안 정확한 예측을 제공하고 성능 저하를 최소화하는 데 성공하였습니다.



### Federated Learning Client Pruning for Noisy Labels (https://arxiv.org/abs/2411.07391)
- **What's New**: 이 논문은 Federated Learning(FL) 환경에서 노이즈가 있는 레이블 문제를 해결하기 위해 ClipFL(석방된 훈련 클라이언트 제외)이라는 새로운 접근 방식을 제시합니다.

- **Technical Details**: ClipFL은 세 가지 단계로 구성되어 있습니다: (1) 클라이언트의 성능을 기준으로 노이즈 클라이언트를 식별하고 Noise Candidacy Score(NCS)를 계산합니다. (2) NCS가 가장 높은 클라이언트의 일부를 제외합니다. (3) 남은 클라이언트에 대해 표준 FL을 수행하여 모델을 정제합니다.

- **Performance Highlights**: ClipFL은 다양한 데이터셋과 노이즈 수준에서 우수한 성능을 나타내어, 노이즈가 있는 클라이언트를 80% 이상의 정확도로 식별하고, 기존 FL 최적화 방법들보다 개선된 성능과 더 빠른 수렴을 보였습니다. 또한 통신 비용이 줄어드는 효과도 확인되었습니다.



### Identifying Differential Patient Care Through Inverse Intent Inferenc (https://arxiv.org/abs/2411.07372)
- **What's New**: 이 논문에서는 세균혈증(sepsis) 환자 관리를 위한 최적 정책을 학습하기 위해 행동 복제(behavioral cloning), 모방 학습(imitation learning), 역 강화 학습(inverse reinforcement learning) 등 다양한 강화 학습(reinforcement learning) 기법을 적용합니다. 이를 통해 실제 정책과의 차이를 비교하여 치료의 차별성을 파악하고자 합니다.

- **Technical Details**: 세균혈증 환자의 최적 치료 정책을 학습하기 위해 MIMIC-IV 데이터와 Mass General Brigham 의료 시스템의 임상 데이터를 사용했습니다. 이를 통해 의료 통계 및 정책을 분석하고, 각 환자 집단의 치료 정책의 차이를 식별하는 데 중점을 두었습니다. 정책 학습은 기초적인 RL 기법을 통해 수행되었습니다.

- **Performance Highlights**: 이 연구의 결과는 성별 및 인종/민족 그룹에 따른 치료 정책의 차별성을 확인하는 데 중요한 도움이 될 것입니다. 이를 통해 향후 공공 건강을 위한 개입의 타겟을 정할 수 있으며, 국가의 세균혈증 치료 가이드라인 변화에 따른 치료 패턴의 변화를 정량화할 수 있습니다.



### Exploring Variational Autoencoders for Medical Image Generation: A Comprehensive Study (https://arxiv.org/abs/2411.07348)
Comments:
          for associated mpeg file, see this https URL

- **What's New**: 본 논문은 의료 이미지 생성 분야에서 Variational Autoencoder (VAE)의 연구 동향을 종합적으로 리뷰하고 있습니다. 특히 VAE가 실제 데이터와 유사한 합성 이미지를 생성하는 능력에 집중하며, 데이터 증강(data augmentation)으로의 활용 가능성을 강조합니다.

- **Technical Details**: VAE는 작은 데이터셋이나 클래스 불균형이 있는 데이터셋에서 샘플을 추가하여 데이터셋을 개선하는 장점을 가지고 있습니다. 본 논문에서는 의료 이미지를 위한 VAE의 주요 구조와 방법론, 그리고 GANs와 같은 다른 생성 모델과의 비교를 다룹니다.

- **Performance Highlights**: 최근 의료 분야에서의 응용을 통해 VAE가 분할(segmentation) 및 분류(classification) 정확도를 개선할 수 있는 능력을 강조합니다.



### Warmstarting for Scaling Language Models (https://arxiv.org/abs/2411.07340)
- **What's New**: 본 연구는 대형 모델을 훈련하기 위한 비용을 줄이기 위해 소형 모델에서 시작하여 대형 모델을 학습하는 방법인 warmstarting을 활용하여 최적의 하이퍼파라미터를 유지할 수 있는지를 탐구합니다.

- **Technical Details**: 연구에서는 \,\mu Transfer을 활용하여 최적 하이퍼파라미터의 제로샷 전이를 이론적으로 동기화된 방법으로 적용할 수 있는 간단한 작업을 탐색합니다. 또한, warmstarting을 통한 수렴 속도 향상과 안정적인 훈련 역학의 유지 요인에 대해 조사합니다.

- **Performance Highlights**: 작은 모델의 가중치를 줄이고 제로 패딩을 적용하며, \mu Transfer로부터 확장된 초기화로 큰 모델을 섞는 방법을 통해 효과적인 warmstarting을 달성했습니다.



### Multimodal Fusion Balancing Through Game-Theoretic Regularization (https://arxiv.org/abs/2411.07335)
Comments:
          21 pages, 6 figures, 4 tables, 1 algorithm

- **What's New**: 이 논문에서는 Multimodal Competition Regularizer (MCR)이라는 새로운 손실 함수를 소개하고, 이를 통해 멀티모달 훈련에서 경쟁으로 인한 부작용을 방지할 수 있도록 설계하였습니다. MCR은 서로 다른 모달리티 간의 의존성을 분해하여 학습의 효율성을 높이고자 합니다.

- **Technical Details**: MCR은 서로 경쟁하는 모달리티가 최종 결과에 미치는 영향을 최대화하는 게임 이론적 원칙을 도입하여 두 가지 상하한을 설정합니다. 이를 통해 학습 과정에서 각 모달리티의 기여도를 조정하고,조건부 MI(Mutual Information)의 추정에 대한 잠재 공간의 변환을 제안하여 계산 효율성을 크게 개선합니다.

- **Performance Highlights**: MCR은 기존의 훈련 전략들보다 뛰어난 성능을 발휘하며, 단순한 앙상블 기준을 넘어 멀티모달 학습을 일관되게 개선하는 첫 번째 방법으로, 합성 및 실제 데이터셋에서 모두 성능 향상을 입증합니다. MCR은 특히 AVE, UCF, CREMA-D, CMU-MOSI, CMU-MOSEI 등 다양한 데이터셋에서 우수한 성능을 기록하였습니다.



### SynRL: Aligning Synthetic Clinical Trial Data with Human-preferred Clinical Endpoints Using Reinforcement Learning (https://arxiv.org/abs/2411.07317)
- **What's New**: 이 논문에서는 환자 데이터 생성기의 성능을 향상시키고, 사용자가 지정한 요구사항에 맞춘 합성 환자 데이터를 맞춤형으로 생성하기 위한 새로운 방법인 SynRL을 제안합니다. 기존의 방법들과 달리, SynRL은 강화학습(reinforcement learning)을 활용하여 데이터 생성 과정에 사용자의 피드백을 통합합니다.

- **Technical Details**: SynRL에서는 데이터 품질을 평가하기 위한 데이터 가치 비평가(data value critic) 기능을 사용하여, 사용자의 특정 요구에 따라 생성된 데이터의 품질을 평가하고, 강화학습을 통해 데이터 생성기를 사용자 선호에 맞추어 조정합니다. 이 시스템은 TVAE(가변 오토인코더)와 CTGAN(생성적 적대 신경망)과 같은 기초 생성기 모델을 사용합니다.

- **Performance Highlights**: 실험 결과, SynRL은 네 가지 임상 시험 데이터셋에서 생성된 합성 데이터의 품질을 향상시키는 동시에 개인 정보 보호 위험을 최소화하는 데 성공했습니다. 이 방법은 다양한 유형의 합성 데이터 생성기의 데이터 생성을 맞춤형으로 조정할 수 있는 범용 프레임워크로 활용될 수 있습니다.



### Anomaly Detection in OKTA Logs using Autoencoders (https://arxiv.org/abs/2411.07314)
Comments:
          11 pages, 3 tables, 8 figures, Databricks AI Summit 2024

- **What's New**: 이 논문은 기존의 Okta 로그 분석 방식의 한계점을 극복하기 위해 비지도 학습(unsupervised techniques)인 오토인코더(autoencoders)를 활용하는 방법을 제안합니다. 기존의 규칙 기반(rule-based) 모델은 한정된 시간 동안의 데이터에만 의존하여 사이버 보안 이벤트를 탐지하기 때문에, 이는 종종 잘못된 경고(false positives)를 초래할 수 있습니다.

- **Technical Details**: Okta의 사용자 행동 이상 탐지 기능은 사용자의 로그인 행동을 분석합니다. 이를 통해 사용자의 일반적인 패턴과의 차이를 여러 필드를 기반으로 정상 패턴에서 벗어나는 경우를 탐지하여 이상 행위를 파악하게 됩니다. 오토인코더는 심층 신경망(deep neural network)으로, 입력을 복제하려는 과정을 통해 비정상적인 행동을 식별하는 메커니즘을 제공합니다.

- **Performance Highlights**: 운영 환경에서 Okta System Log의 데이터셋을 활용하여, 기존의 한정된 시간대 대신 전체 사용자 로그인 이력을 통해 사용자 행동의 변화를 식별할 수 있는 강력한 모델을 개발했습니다. 이를 통해 사이버 공격 탐지 및 대응 능력을 향상시킬 수 있습니다.



### Multi-hop Upstream Preemptive Traffic Signal Control with Deep Reinforcement Learning (https://arxiv.org/abs/2411.07271)
Comments:
          5 tables, 12 figures. arXiv admin note: text overlap with arXiv:2409.00753

- **What's New**: 본 논문은 마르코프 체인 이론(Markov chain theory)을 기반으로 하는 새로운 개념인 다중 홉 업스트림 압력(multi-hop upstream pressure)을 소개하여 기존의 단기적 트래픽 신호 제어 방식의 한계를 극복하고자 한다. 이 접근 방식은 현재 위치에서 즉각적인 링크만 고려하는 대신, 더 넓은 범위의 트래픽 조건을 반영하여 신호 타이밍을 최적화할 수 있도록 한다.

- **Technical Details**: 다중 홉 업스트림 압력 개념은 주변 환경의 트래픽 조건을 종합적으로 고려하여 트래픽 신호 제어의 효과성을 높인다. 이를 위해 강화학습(deep reinforcement learning) 에이전트는 현재 대기열을 미리 정리하도록 안내되며, 시뮬레이션 결과는 이 새로운 지표가 네트워크의 전체 지연을 감소시키는 데 효과적임을 보여준다.

- **Performance Highlights**: 시뮬레이션(예: 토론토 시나리오) 결과, 다중 홉 업스트림 압력을 활용하는 컨트롤러는 넓은 범위의 선행 혼잡을 이해하여 트래픽 흐름을 우선시함으로써 전체 네트워크 지연을 상당히 줄이는 것으로 나타났다.



### Learning From Graph-Structured Data: Addressing Design Issues and Exploring Practical Applications in Graph Representation Learning (https://arxiv.org/abs/2411.07269)
Comments:
          arXiv admin note: text overlap with arXiv:2205.11691, arXiv:2304.14621

- **What's New**: 이번 논문에서는 그래프 표현 학습(Graph Representation Learning) 및 그래프 신경망(Graph Neural Networks, GNNs)의 최신 발전에 대한 포괄적인 리뷰를 제공합니다. 특히, 복잡한 노드 상호작용을 포착할 수 있는 고차 풀링(high-order pooling) 함수를 갖춘 GNN을 소개하며, 이는 노드 및 그래프 수준의 작업에서 GNN의 효능을 크게 향상시킵니다. 또한, GNN을 기반으로 한 분자 그래프 생성 모델도 제안합니다.

- **Technical Details**: GNN은 그래프 구조 데이터를 처리하도록 설계된 신경망으로, 여러 단계의 메시지 전파를 통해 이웃으로부터 정보를 집계하여 노드 표현을 반복적으로 업데이트합니다. 본 연구에서는 대칭 텐서 분해(symmetric tensor decomposition)를 기반으로 한 집계 함수를 설계하여 비선형 고차 곱셈 상호작용을 모델링하고, 이는 수치적으로 효율적인 방식으로 비가변(permutation-invariant) 멀티선형 함수들을 처리할 수 있습니다. 마지막으로, CP 레이어(CANDECOMP/PARAFAC decomposition)는 그래프의 전체 표현을 계산할 수 있게 해 주며, 이를 통해 비선형 고차 상호작용을 효과적으로 모델링 할 수 있습니다.

- **Performance Highlights**: 신뢰할 수 있는 방법들과의 철저한 실험 평가 및 비교를 통해, 제안된 모델들이 다양한 데이터셋을 사용하여 여러 실제 문제를 해결하는 데 있어 뛰어난 성능을 보임을 입증했습니다. 특히, GNN은 분자 그래프 생성 작업에 강력한 기반 모델로 활용되어, 화합물의 구조를 정확히 재현할 수 있음을 보여 주었습니다.



### Analysis and Forecasting of the Dynamics of a Floating Wind Turbine Using Dynamic Mode Decomposition (https://arxiv.org/abs/2411.07263)
- **What's New**: 본 논문은 Dynamic Mode Decomposition (DMD)을 기반으로 하는 헥사플로트 부유식 해상 풍력 터빈의 동역학에 대한 데이터 기반의 비방정식 모델링을 제시합니다. 연구진은 예측 알고리즘을 개발하여 터빈의 운동, 가속도 및 작용하는 힘, 도착하는 파도의 높이, 풍속 및 터빈이 추출한 전력을 평가합니다.

- **Technical Details**: Hankel-DMD라는 방법론적 확장을 사용하여 지연된 상태 복사본을 포함하는 보강 상태 벡터를 통해 알고리즘의 하이퍼파라미터 두 가지(지연 복사본 수, 관측 시간 길이)를 변동시키고, 세 가지 서로 다른 오류 지표를 이용하여 예측 품질을 평가합니다. Stochastic Hankel-DMD 공식을 도입하여 하이퍼파라미터를 확률 변수로 설정했습니다.

- **Performance Highlights**: 결과는 시스템 상태의 단기 예측을 위한 방법의 능력을 보여주며, 이는 실시간 예측 및 제어에 활용 가능하다는 점에서 의미가 있습니다. 확률적 버전의 방법은 예측의 불확실성을 포함하여 정량적 성능을 개선시키며, 결정론적 방법과 비교하여 정상화된 평균 제곱근 오차를 최대 10% 감소시켰습니다.



### Ozone level forecasting in Mexico City with temporal features and interactions (https://arxiv.org/abs/2411.07259)
Comments:
          11 pages, 5 figures

- **What's New**: 이번 연구에서는 멕시코시티의 오존 수준을 예측하기 위해 다양한 회귀 모델의 정확도를 비교하였으며, 시간적 특성과 변수 상호작용을 추가함으로써 모델 정확도를 향상시킬 수 있음을 보였습니다.

- **Technical Details**: 연구는 2015년 1월 1일부터 2023년 5월 31일까지의 데이터를 사용하여, CO, NOx, PM10, PM2.5, SO2 등의 변수를 포함한 다수의 회귀 모델을 분석했습니다. 시간적 특성과 변수 간 상호작용을 모델에 포함시키면서 성능을 향상시키고, 랜덤 포레스트를 이용하여 중요한 변수를 선택했습니다.

- **Performance Highlights**: 모델에 시간적 특성과 상호작용을 추가함으로써, 오존 수준 예측의 정확도가 유의미하게 높아졌음을 확인하였습니다. 이로 인해 오존 동역학에 대한 이해도가 높아지고, 향후 보다 효과적인 규제 정책 수립에 도움이 될 것으로 기대됩니다.



### Leonardo vindicated: Pythagorean trees for minimal reconstruction of the natural branching structures (https://arxiv.org/abs/2411.08024)
Comments:
          22 pages, lots of hi res figures I had to reduce quality of, submitting as a requirement to the Theory of Computing Journal

- **What's New**: 최근 Pythagorean tree(피타고라스 나무) 구조의 다양한 변형을 조사하여 자연에서 관찰되는 나뭇가지 구조와 유사한 형상을 찾고 설명하는 연구가 진행되었습니다. 이 연구는 나무의 구조적 아름다움과 그 공학적 최적성을 동시에 탐구하고 있습니다.

- **Technical Details**: Pythagorean tree는 정삼각형의 변에 사각형을 두어 구성된 프랙탈 디자인으로, 나뭇가지의 지점에서 자연적인 나무의 분기 구조를 모방하는 알고리즘을 개발하였습니다. 이를 통해 다양한 혼합 파라미터를 조정하여 나무의 구조를 시뮬레이션하고 CNN(Convolutional Neural Networks)을 이용하여 나무의 진짜 이미지를 분류하는 과정에서 자연적인 나무 구조를 재현하는 데 성공하였습니다.

- **Performance Highlights**: 이 연구에서 생성된 프랙탈 트리는 CNN의 분류 정확도를 높이는 결과를 가져왔으며, 이는 Leonardo da Vinci의 분기 규칙과 황금비를 기반으로 한 나무의 구조적 원리를 뒷받침하는 것으로 나타났습니다. 이를 통해 인공적으로 생성된 트리 모델이 다양한 나무 종의 탐지를 위한 강력한 훈련 데이터로 사용될 수 있음을 주장합니다.



### Language Models as Causal Effect Generators (https://arxiv.org/abs/2411.08019)
- **What's New**: 이 논문에서는 큰 언어 모델(LLM)에 기반한 데이터 생성 프레임워크를 제시하며, 특정한 인과 구조를 제어할 수 있는 방법을 설명합니다. 이 방법은 언어 모델과 방향 비순환 그래프(DAG)를 결합하여 순차적으로 구동되는 구조적 인과 모델(SD-SCM)을 생성합니다.

- **Technical Details**: SD-SCM은 사용자 정의 구조와 LLM 정의 구조 방정식을 포함하는 인과 모델을 나타냅니다. SD-SCM을 사용하면 관찰적, 개입적, 반사적 분포에서 샘플링할 수 있는 방법을 제공합니다. 이 모델을 통해 개별 반사적 데이터를 자동으로 생성할 수 있으며, 이는 기존의 기능적 관계를 수동으로 명시하지 않아도 가능하게 합니다. 코드와 데이터셋은 GitHub에서 접근할 수 있습니다.

- **Performance Highlights**: SD-SCM을 활용하여 생성된 데이터셋에 대해 여러 평균처리효과(ATE), 조건부 평균처리효과(CATE), 개별처리효과(ITE)를 추정하는 방법을 테스트했습니다. 이 절차는 LLM이 잘못된 정보, 차별, 또는 기타 바람직하지 않은 행동을 감지하는 데도 사용할 수 있어 LLM 감사를 위한 기반이 될 수 있습니다.



### Wavelet Latent Diffusion (Wala): Billion-Parameter 3D Generative Model with Compact Wavelet Encodings (https://arxiv.org/abs/2411.08017)
- **What's New**: 본 논문에서는 Wavelet Latent Diffusion (WaLa)라는 새로운 접근 방식을 소개하여 3D 형태를 wavelet 기반의 압축된 잠재 인코딩으로 인코딩합니다. 이를 통해 $256^3$의 signed distance field를 $12^3 	imes 4$의 잠재 그리드로 압축하여 2427배의 압축 비율을 달성했습니다.

- **Technical Details**: WaLa는 압축 과정에서 정보 손실 없이 wavelet 표현을 더욱 압축하여, diffusion 기반의 생성 모델을 효율적으로 확장할 수 있도록 합니다. 구체적으로는 convolution 기반의 VQ-VAE 모델을 사용하여 압축을 진행하며, 이는 약 10억 개의 매개변수를 포함하고 있습니다.

- **Performance Highlights**: WaLa는 고해상도 3D 생성에서 최첨단 성능을 보여주며, 다양한 입력 모달리티를 지원합니다. 모델의 생성 속도는 2~4초이며, 제어된 생성 또한 가능하여 복잡한 기하학, 신뢰할 수 있는 구조와 세밀한 토폴로지를 가진 3D 형태를 생성합니다.



### Investigating the Effectiveness of Explainability Methods in Parkinson's Detection from Speech (https://arxiv.org/abs/2411.08013)
Comments:
          The first two authors contributed equally to this research: author order is alphabetical

- **What's New**: 이 연구는 파킨슨병(Parkinson's Disease, PD) 진단을 위한 음성 기초 모델의 해석 가능성을 높이기 위한 설명 가능성 방법을 체계적으로 평가합니다. 이는 PD에 특화된 음성 특징을 식별하고, 임상 의사결정에 있어 정확하고 해석 가능한 모델의 개발을 지원하는 것을 목표로 합니다.

- **Technical Details**: 연구 방법론은 (i) 주류 해석 가능성 기술을 사용하여 속성 및 주목도 맵(attribution and saliency maps)을 획득하고, (ii) 이러한 맵의 충실성을 정량적으로 평가하며, (iii) 보조 분류기로부터 PD 탐지를 위한 주목도 맵이 전하는 정보를 평가하는 과정을 포함합니다. Saliency, SmoothGrad, Integrated Gradients, Guided GradCAM 등의 다양한 해석 가능성 기법이 사용되었습니다.

- **Performance Highlights**: 결과적으로, 설명은 분류기의 예측과 일치는 하나, 도메인 전문가들에게 진정으로 유용한 통찰을 제공하지 못하는 경우가 많았습니다. 이는 기존의 해석 가능성 방법들이 실제적인 사용에 필요한 수준의 해석 가능성을 결여하고 있음을 강조합니다.



### Derivational Morphology Reveals Analogical Generalization in Large Language Models (https://arxiv.org/abs/2411.07990)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)의 언어적 일반화(linguistic generalization) 메커니즘을 분석했다. 특히, 규칙 기반(rule-based) 접근 방식과 유사성 기반(analogical) 접근 방식의 둘 다 설명할 수 있는 다양한 문법 현상 중 영어 형용사의 명사화(adjective nominalization)를 조사했다.

- **Technical Details**: 본 연구는 GPT-J 모델을 중심으로 하여, 규칙 기반 모델인 Minimal Generalization Learner (MGL)와 유사성 모델인 Generalized Context Model (GCM)을 비교 분석하였다. 이 과정에서는 4가지 형용사 클래스에 대해 -ity와 -ness의 선호도를 분석하며, 각 모델의 예측 결과를 검토하였다. 연구는 모델이 유형 빈도(type frequency)와 토큰 빈도(token frequency) 의 영향을 어떻게 반영하는지를 평가하였다.

- **Performance Highlights**: 연구 결과, GPT-J는 정규적인 명사화 패턴에 대해서는 MGL과 GCM 모두와 비슷한 예측을 수행한다. 하지만, 다양한 명사화 패턴을 가진 형용사에서는 GCM의 유사성 모델이 GPT-J의 행동을 더 잘 설명한다는 사실이 드러났다. 이는 GPT-J가 규칙이 아닌 유사한 예를 바탕으로 언어적 일반화를 이루고 있음을 시사한다.



### Doubly Robust Regression Discontinuity Designs (https://arxiv.org/abs/2411.07978)
- **What's New**: 이 연구는 Regression Discontinuity (RD) 디자인을 위한 doubly robust (DR) 추정기를 소개합니다. 이 방법은 두 가지 서로 다른 추정기를 결합하여 조건부 기대 결과를 추정함으로써 치료 효과를 보다 안정적으로 추정할 수 있도록 합니다.

- **Technical Details**: DR-RD 추정기는 두 개의 비모수 (nonparametric) 회귀 추정기를 결합하며, 둘 중 하나라도 일관성이 있을 경우 치료 효과 추정량이 일관성을 유지합니다. 또한, 제안된 추정기는 두 회귀 추정기가 특정한 조건을 만족할 경우 $	ext{√n}$-일관성을 달성합니다.

- **Performance Highlights**: 이 방법은 RD 디자인에서의 비모수 수렴 속도의 한계를 극복하고, 통계적 추론을 단순화하는 데 기여합니다. 연구는 경제학, 정치학 및 역학과 같은 다양한 분야에서 RD 디자인의 응용 가능성을 보여줍니다.



### Optimal Control of Mechanical Ventilators with Learned Respiratory Dynamics (https://arxiv.org/abs/2411.07971)
Comments:
          2024 IEEE 37th International Symposium on Computer-Based Medical Systems (CBMS), 7 pages, 3 figures

- **What's New**: 본 연구에서는 ARDS 환자의 인공호흡기 관리 문제를 상태 전이 모델인 Markov Decision Process (MDP)로 정의하고, 이를 통해 다양한 인공호흡기 제어 전략을 비교합니다. 특히, 인공지능의 신경망 및 최적 제어 이론을 활용하여 전통적인 규칙 기반 접근법 없이도 효과적인 인공호흡기 관리 전략을 자동으로 발견할 수 있음을 입증합니다.

- **Technical Details**: 연구는 Pulse Physiology Engine을 사용하여 ARDS 환자의 호흡 역학을 시뮬레이션하고, MDP를 통해 환자의 상태(S), 인공호흡기 동작(A), 보상 함수(R) 등을 정의합니다. 인공호흡기의 입력값으로는 FiO2 (Fraction of Inspired Oxygen), inspiratory pressure, inspiratory time 등을 포함하여 총 6개의 변수를 고려하고, 환자의 건강상태에 따른 전이확률을 모델링합니다.

- **Performance Highlights**: 실험 결과, 신경망 및 최적 제어 기술을 활용한 접근법이 기존의 ARDSnet 프로토콜과 비교하여 환자의 호흡률, 산소화 수치, 생체신호 개선에서 향상된 결과를 보임을 확인했습니다. 특히, 인공지능 기반 전략이 인공호흡기의 관리 프로세스에 대한 명시적 지침 없이도 효과적인 솔루션을 제공할 수 있음을 보여줍니다.



### Tukey g-and-h neural network regression for non-Gaussian data (https://arxiv.org/abs/2411.07957)
- **What's New**: 이 논문은 비가우시안 회귀(non-Gaussian regression)에 대한 신경망(neural networks) 접근 방식을 제안합니다. Tukey g-and-h 변환을 이용하여 신경망 모델이 회귀 프레임워크에서 Tukey g-and-h 분포의 매개변수를 예측하도록 훈련하는 방법을 설명합니다.

- **Technical Details**: Tukey g-and-h 분포는 두 개의 매개변수 g와 h를 사용하는 유연한 모수(parametric) 변환으로, skewness와 kurtosis를 도입하여 일반적으로 Tukey g-and-h 분포로 알려진 분포를 생성합니다. 이 연구에서는 비가우시안 특성을 모델링하기 위해 신경망을 훈련시키며, log-likelihood의 음수 최소화를 통해 회귀 모델을 최적화합니다. 이를 위해 Tukey g-and-h 변환의 역함수를 제공하기 위한 binary search 방법이 사용됩니다.

- **Performance Highlights**: 모의 데이터 실험을 통해 제안된 방법론의 효율성을 입증하였으며, 여러 작물의 글로벌 수확량 데이터셋에 이 방법을 적용하여 실제 회귀 문제에서의 성능을 보여줍니다. 또한, Pytorch 구현이 Github에 공개되어 있습니다.



### Towards Low-bit Communication for Tensor Parallel LLM Inferenc (https://arxiv.org/abs/2411.07942)
- **What's New**: 이 논문에서는 tensor parallelism의 커뮤니케이션 비용을 줄이기 위한 새로운 양자화(quantization) 방법을 제안합니다. 기존 tensor parallelism의 양자화 방법은 주로 높은 정밀도로 출력 기능을 유지하는 것을 목표로 했으나, 본 연구는 양자화를 통해 커뮤니케이션을 평균 16비트에서 4.2비트로 줄이고, 원래 모델의 성능을 거의 유지하는 것을 이룹니다.

- **Technical Details**: 제안된 방법은 각 디바이스 간의 tensor parallelized attention 및 feedforward 블록에서 저비트(low-bit) 출력을 효과적으로 통신할 수 있도록 설계되었습니다. 주요 아이디어는 아웃라이어(outlier) 기능을 정적으로 선택하여 BF16으로 유지하고, 나머지는 4비트로 양자화(quantization)하는 것입니다. 이 과정에서 원래 가중치(weights)에 교란을 주지 않는 특징이 있습니다.

- **Performance Highlights**: 이 방법은 Gemma 2 27B 모델의 경우 원래 성능의 약 98%, Llama 2 13B 모델의 경우 99.5%를 유지하며, 통신하는 정보의 양은 약 1/4로 줄어듭니다. 이러한 성능 유지는 재구성 오류를 최소화하며 고성능의 대규모 언어 모델을 위한 전략적인 접근법을 제공합니다.



### Prediction of Acoustic Communication Performance for AUVs using Gaussian Process Classification (https://arxiv.org/abs/2411.07933)
- **What's New**: 이 논문에서는 자율 수중 차량(AUV) 간의 신뢰할 수 있는 통신을 예측하기 위해 새로운 접근법인 확률 기반 통신 맵을 제안합니다. 기존의 보수적인 통신 범위 가정에서 벗어나, 각 AUV의 위치에 기반하여 성공적인 통신 확률을 모델링합니다.

- **Technical Details**: 이 연구에서는 Gaussian process binary classification을 활용하여 통신 성공 확률을 예측하기 위한 방법을 개발했습니다. AUV 간의 통신 성능을 모델링하고, 통신 실패 이벤트와 송신기 위치의 불확실성을 고려하여 통신 성능 예측을 개선합니다.

- **Performance Highlights**: 실험적으로 Virginia Tech 690 AUVs를 사용하여 GP classification이 GP regression보다 우수한 예측 성능을 보여주었으며, 송신기 위치의 불확실성을 고려했을 때 추가적인 성능 향상이 관찰되었습니다.



### INTRABENCH: Interactive Radiological Benchmark (https://arxiv.org/abs/2411.07885)
Comments:
          Undergoing Peer-Review

- **What's New**: IntRaBench는 3D 의료 영상에서의 인터랙티브 세분화 방법을 효과적으로 평가할 수 있는 새로운 벤치마크 프레임워크입니다. 이 프레임워크는 다양한 데이터셋과 세분화 모델을 포함하며, 임상에서의 실제 사용을 고려하여 개발되었습니다.

- **Technical Details**: IntRaBench는 2D 및 3D 인터랙티브 세분화 방법의 공정하고 재현 가능한 평가를 지원합니다. 특정한 프로트핑(prompting) 및 수정(refinement) 전략을 통해 2D 모델에서도 사용자 상호작용을 간소화하고, 대시(board)에서 인간의 노력을 최소화합니다. 이 벤치마크는 10개의 데이터셋과 7개의 모델을 포함하며, 모두 공개되어 있어 사용자가 쉽게 다운로드 및 전처리할 수 있습니다.

- **Performance Highlights**: IntRaBench는 최초로 2D와 3D 인터랙티브 세분화 방법 간의 공정한 비교를 가능하게 합니다. 연구자들은 이 프레임워크를 이용하여 새로운 방법을 평가하고, 지속적이고 투명한 세분화 모델 평가를 통해 3D 의료 영상 세분화 분야에서의 진전을 추적할 수 있습니다.



### CDXFormer: Boosting Remote Sensing Change Detection with Extended Long Short-Term Memory (https://arxiv.org/abs/2411.07863)
- **What's New**: 본 논문에서는 성능과 효율성을 균형 있게 고려한 새로운 접근 방식인 CDXFormer를 제안합니다. CDXFormer는 XLSTM 기반의 강력한 Feature Enhancer 레이어를 핵심 요소로 사용하여, 공간적 맥락(spatial context) 인식 및 변화 감지를 설명 가능하게 합니다.

- **Technical Details**: CDXFormer는 Scale-specific Feature Enhancer가 포함된 구조로, Cross-Temporal Global Perceptron (CTGP)와 Cross-Temporal Spatial Refiner (CTSR)로 구성됩니다. CTGP는 심층의 영역에서 의미적 차이를 강화하며, CTSR은 얕은 영역에서 세부 정보를 보강합니다. Cross-Scale Interactive Fusion (CSIF) 모듈이 결합되어 글로벌 변화 표현과 공간 응답을 상호작용하게 합니다.

- **Performance Highlights**: CDXFormer는 세 가지 기준 데이터셋에서 이전의 최첨단(SOTA) 접근법을 능가하며, 효율성과 정확성의 뛰어난 균형을 제공합니다.



### Tucano: Advancing Neural Text Generation for Portugues (https://arxiv.org/abs/2411.07854)
- **What's New**: 본 연구는 포르투갈어의 신경 텍스트 생성을 위한 새로운 자원을 소개하고 있습니다. GigaVerbo라는 대규모의 포르투갈어 데이터셋을 구축하여, 이를 활용해 Tucano라는 디코더-트랜스포머 모델을 훈련하였습니다.

- **Technical Details**: GigaVerbo는 2000억 개의 토큰으로 구성된 중복 제거된 포르투갈어 텍스트 코퍼스의 집합으로, 여기서 Tucano 모델을 훈련시켰습니다. 이 모델은 여러 포르투갈어 벤치마크에서 다른 포르투갈어 및 다국어 모델들과 동등하거나 우수한 성능을 보였습니다.

- **Performance Highlights**: Tucano 모델은 기존의 포르투갈어 NLP 커뮤니티에서 사용되는 벤치마크와의 성능 평가에서도 좋은 성과를 냈으며, 특히 기존 모델과의 성능 상관관계의 한계를 드러냈습니다.



### PatchCTG: Patch Cardiotocography Transformer for Antepartum Fetal Health Monitoring (https://arxiv.org/abs/2411.07796)
- **What's New**: 이 논문에서는 Antepartum Cardiotocography (CTG) 분석을 위한 최신 Transformer 기반 모델인 PatchCTG를 소개합니다. 기존의 CTG 분석 방법은 해석의 일관성이 부족하여 오류가 발생하기 쉬운데, PatchCTG는 패치 기반 토크나이제이션과 인스턴스 정규화, 채널 독립 처리 기술을 이용하여 이러한 한계를 극복하고자 합니다.

- **Technical Details**: PatchCTG는 CTG 데이터를 패치로 나누고, 각 패치마다 채널 독립적으로 처리하여 로컬 및 글로벌 시간 종속성(temporal dependencies)을 포착합니다. 또한 인스턴스 정규화를 통해 데이터의 분포 변화(distribution shifts)를 관리하고, FHR(심박수) 및 자궁 수축 패턴을 더 정확하게 모델링할 수 있도록 설계되었습니다. 이를 통해 PatchCTG는 임신 중 다양한 임상 요구에 적합한 분석을 수행할 수 있습니다.

- **Performance Highlights**: PatchCTG는 Oxford Maternity (OXMAT) 데이터 세트를 활용하여 검증되었습니다. 실험 결과, PatchCTG는 AUC 77%, 특이도(specificity) 88%, 민감도(sensitivity) 57%를 기록했으며, 특히 출산 직전 데이터를 활용한 미세 조정(fine-tuning)에서 민감도 52% 및 특이도 88%를 달성했습니다. 이러한 성과는 PatchCTG가 안정적이고 신뢰할 수 있는 임신 중 건강 상태 평가 도구로 사용될 가능성을 시사합니다.



### Likelihood as a Performance Gauge for Retrieval-Augmented Generation (https://arxiv.org/abs/2411.07773)
Comments:
          Under review at NAACL 2025. Code is available at this https URL

- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LMs)에서의 retrieval-augmented generation(RAG) 과정 중 문서의 순서가 결과에 미치는 영향을 분석합니다. 특히, 질문의 likelihood가 모델 성능을 예측할 수 있는 지표가 될 수 있음을 보여주며, 이를 바탕으로 더 나은 성능을 위한 프롬프트 최적화 방법을 제안합니다.

- **Technical Details**: 본 연구는 NQ-Open과 ELI5 두 가지 질문-답변 데이터셋에서 다양한 최첨단 LMs(LLaMA-2, LLaMA-3, LLaMA-3.1, Mistral-v0.3, MPT)를 활용하여 질문의 likelihood와 답변 정확도 간의 상관관계를 조사하였습니다. 입력 프롬프트의 세 가지 구성 요소인 컨텍스트, 질문, 금지 답변(gold answer)의 log-likelihood를 분석하여, 높은 log-likelihood를 가진 질문에 대해 LMs가 더 나은 답변을 할 수 있음을 발견했습니다.

- **Performance Highlights**: 제안된 방법은 질문 likelihood를 기반으로 한 프롬프트 최적화를 통해 두 데이터셋에서 답변 정확도를 개선했습니다. 효율적인 계산 방식이 특징적이며, LM 응답을 생성하기 위해 여러 번 실행할 필요가 적어 계산 비용을 절감할 수 있습니다.



### EMPERROR: A Flexible Generative Perception Error Model for Probing Self-Driving Planners (https://arxiv.org/abs/2411.07719)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 새로운 트랜스포머 기반의 생성형 인식 오류 모델(PEM)인 Emperror를 제안합니다. 이를 통해 경험 기반의 데이터로부터 자율 주행 플래너를 스트레스 테스트할 수 있는 틀을 제공합니다.

- **Technical Details**: Emperror는 트랜스포머 아키텍처를 기반으로 구축되어, 현대의 객체 탐지기와의 불일치를 더 정밀하게 모방합니다. 이러한 PEM은 다양한 오류 패턴을 모델링하고, 자율 주행 플래너의 강인성을 시험할 수 있는 도전적인 샘플을 생성하는 역할을 합니다.

- **Performance Highlights**: Emperror를 사용하여 생성된 노이즈 입력으로는 플래너의 충돌률이 최대 85%까지 증가하여, 자율 주행 플래너의 평가에 있어 유용한 도구로 기능함을 보였습니다.



### Understanding Audiovisual Deepfake Detection: Techniques, Challenges, Human Factors and Perceptual Insights (https://arxiv.org/abs/2411.07650)
- **What's New**: 이 연구는 복합적인 심층fake (deepfake)를 탐지하기 위해 음향(오디오)과 시각(비주얼) 모달리티를 함께 분석하는 접근 방식을 다룬 최초의 종합적인 조사입니다.

- **Technical Details**: 이 연구는 네 가지 유형의 deepfake를 정의하고 각각의 생성 기법 및 최신 Deep Learning (DL) 기술을 분석합니다. 특히, Generative Adversarial Networks (GAN)과 Variational Autoencoders (VAE)를 사용하여 음향과 시각의 결합된 디지털 조작 방법을 활용합니다.

- **Performance Highlights**: 음향 및 시각 모달리티를 활용한 기존 탐지 방법의 한계를 지적하고, 오디오 및 비주얼 deepfake 탐지에 대한 기존 연구의 격차를 해소하기 위한 연구 방향을 제시합니다. 또한, 이러한 방법들을 훈련시키기 위한 공개 데이터 세트에 대해서도 자세히 분석하였습니다.



### xCG: Explainable Cell Graphs for Survival Prediction in Non-Small Cell Lung Cancer (https://arxiv.org/abs/2411.07643)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 11 pages

- **What's New**: 이번 논문에서는 그래프 신경망(Graph Neural Networks)을 활용하여 폐 선암 환자의 생존 예측을 위한 설명 가능한 세포 그래프(xCG) 접근법을 소개합니다. 이는 정밀 의학 데이터 기반의 결정에 기여할 수 있습니다.

- **Technical Details**: 저자들은 다중 조직 샘플 및 그래프를 처리할 수 있는 GNN 프레임워크를 제안하며, 세포 수준의 다양한 특징 영역(예: marker 표현과 임상 메타데이터)을 통합합니다. 이를 통해 생존 회귀(survival regression)와 분류(classification)가 가능합니다. 또한 grid 기반의 layer-wise importance propagation (LRP) 방법을 도입하였습니다.

- **Performance Highlights**: 제안된 xCG 방법은 416명의 폐 선암 환자에 대한 이미징 질량 세포 분석(IMC) 데이터로 검증되었으며, 암 병기와 모델 앙상블을 결합함으로써 리스크 추정의 품질을 개선하는 데 중요한 요소로 작용했습니다.



### Exploring Multi-Agent Reinforcement Learning for Unrelated Parallel Machine Scheduling (https://arxiv.org/abs/2411.07634)
Comments:
          11 pages, 5 figures, 4 tables, article submitted to a journal

- **What's New**: 이번 연구는 Multi-Agent Reinforcement Learning (MARL) 접근법을 통해 Unrelated Parallel Machine Scheduling Problem (UPMS)을 다루며, 기존 Single-Agent 알고리즘과 비교하여 MARL의 효능을 입증합니다. 특히, MASKABLE PPO 알고리즘은 Single-Agent 시나리오에서 우수한 성능을 보여주고, Multi-Agent 환경에서는 협동 학습의 도전 과제를 선보입니다.

- **Technical Details**: 본 논문은 세팅 시간과 자원 활용을 고려하여 Unrelated Parallel Machine Scheduling 문제를 정의하고 최적 작업 스케줄링을 위한 MARL 환경을 설정합니다. 다양한 딥 뉴럴 네트워크 정책을 적용하여 Single-Agent 및 Multi-Agent 접근법의 성과를 비교 분석합니다.

- **Performance Highlights**: 실험 결과, Single-Agent 알고리즘은 제한된 시나리오에서 적절한 성능을 보였으나, Multi-Agent 접근법은 협동 학습의 도전 과제를 드러내면서도 확장 가능한 능력을 지니고 있음을 확인했습니다. 이 연구는 알고리즘의 정교함과 스케일러블한 특성을 균형있게 고려하여 지능형 스케줄링 솔루션을 위한 MARL 기술의 적용에 대한 통찰력을 제공합니다.



### CJST: CTC Compressor based Joint Speech and Text Training for Decoder-Only ASR (https://arxiv.org/abs/2411.07607)
Comments:
          submitted to ICASSP2025

- **What's New**: 본 연구에서는 CTC compressor를 이용한 새로운 연합 음성 및 텍스트 훈련(CJST) 프레임워크를 제안합니다. 이 프레임워크는 디코더 전용 자동 음성 인식(ASR) 모델의 성능을 향상시키기 위해 음성과 텍스트를 양 방향으로 매칭합니다.

- **Technical Details**: CJST는 간단한 모달리티 어댑터와 CTC compressor의 여러 기능(시퀀스 압축, 실시간 강제 정렬 및 CTC 클래스 임베딩 등)을 결합하여 구현됩니다. CTC compressor는 오디오 인코더의 출력에 직접 적용되어 CTC 확률을 바탕으로 압축된 출력을 추가 디코딩 모델에 전달합니다.

- **Performance Highlights**: 실험 결과, 제안된 CJST는 Librispeech 및 TED-LIUM2 데이터셋을 활용하였으며, 효과적인 텍스트 주입을 통해 도메인 내 및 크로스 도메인 시나리오에서 최고의 성능을 달성했습니다. 또한 CTC compressor의 다양한 압축 모드와 노이즈 데이터를 포함한 포괄적인 연구를 제공하여 디코더 전용 모델에 가장 적합한 설정을 규명하였습니다.



### SegQC: a segmentation network-based framework for multi-metric segmentation quality control and segmentation error detection in volumetric medical images (https://arxiv.org/abs/2411.07601)
Comments:
          28 pages, 9 figures

- **What's New**: SegQC는 볼륨 의료 이미지에서 세그멘테이션(segmentation) 품질 추정 및 오류 감지를 위한 새로운 프레임워크입니다. 이 프레임워크는 의료 실습에서의 세그멘테이션 오류 검출 및 모델 개발을 용이하게 합니다.

- **Technical Details**: SegQC는 다음의 주요 요소들로 구성됩니다: 1. SegQC-Net - 스캔과 세그멘테이션 마스크를 입력으로 받아 각 복셀(voxel)에 대한 세그멘테이션 오류 확률을 출력하는 딥 네트워크(deep network); 2. 세그멘테이션 오류 확률을 바탕으로 계산된 세 가지 새로운 세그멘테이션 품질 메트릭(metrics) - 두 개의 오버랩(overlap) 메트릭과 구조 크기 메트릭(structure size metric); 3. 스캔 슬라이스에서 가능한 세그멘테이션 오류를 감지하는 새로운 방법.

- **Performance Highlights**: SegQC는 198개의 태아 MRI 스캔에서 태아 뇌, 태아 몸, 태반의 세 구조에 대해 시험되었습니다. SegQC는 Pearson correlation과 MAE(Mean Absolute Error) 측면에서 TTA 기반 품질 추정보다 더 우수한 성능을 보였습니다. 세그멘테이션 오류 감지 방법은 태아 몸과 태아 뇌의 경우 각각 0.77 및 0.48, 0.74 및 0.55의 재현율(recall)과 정밀도(precision) 비율을 달성하였습니다.



### Decision Feedback In-Context Symbol Detection over Block-Fading Channels (https://arxiv.org/abs/2411.07600)
- **What's New**: 본 논문에서는 	extit{DE}cision 	extit{F}eedback 	extit{IN}-Cont	extit{E}xt 	extit{D}etection (DEFINED)라는 새로운 와이어리스 수신기 설계를 제안하며, 기존의 채널 추정을 우회하고 제한된 파일럿 데이터로 직접 기호 검출을 수행하는 모델을 개발했습니다.

- **Technical Details**: DEFINED 모델은 ICL(In-Context Learning)에서 결정 피드백 메커니즘을 적용하여 자동으로 기호를 연속적으로 통합하여 이후의 기호 검출 개선을 꾀합니다. 모델은 혼합 훈련 프로세스를 통해 제한된 파일럿(때로는 단일 파일럿)로도 높은 성능을 달성하고, 충분한 파일럿이 있을 때 정확도를 유지합니다.

- **Performance Highlights**: 광범위한 와이어리스 통신 설정에서 DEFINED는 상당한 성능 개선을 보여주었습니다. 일부 경우에는 단일 파일럿 쌍만으로도 우수한 성능을 달성하며, 일반적인 신호 대 잡음비(SNR) 환경에서도 효과적입니다.



### Uncertainty-Aware Test-Time Adaptation for Inverse Consistent Diffeomorphic Lung Image Registration (https://arxiv.org/abs/2411.07567)
Comments:
          5 pages, 4 figures

- **What's New**: 본 연구에서는 불확실성을 고려한 테스트 시간 적응 프레임워크를 제안하여 폐 이미지의 역일관성 있는 차원화 방법을 개선하고자 하였습니다. 이 방법은 Monte Carlo (MC) dropout을 활용하여 공간적 불확실성(map) 지도를 생성하고 이를 통해 모델 성능을 향상시킵니다.

- **Technical Details**: 우리는 변형될 이미지 맞춤(matching)을 위해 변형적 이미지 등록(Deformable Image Registration, DIR) 프레임워크를 사용하였습니다. 고정 이미지와 이동 이미지를 정렬하는 최적의 변환을 찾는 문제로, 이는 미분 가능하고 역변환 가능성의 보장을 포함하는 LDDMM(Large Deformation Diffeomorphic Metric Mapping) 프레임워크를 바탕으로 하였습니다. 몬테카를로 dropout을 통해 불확실성 지도를 생성하고 이를 이용해 모델을 적응 및 최적화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 폐 경계에서 Dice 유사도 지수(DSC) 0.966을 달성하였으며, 이는 기존의 VoxelMorph(0.953) 및 TransMorph(0.953)보다 높은 수치입니다. 양방향 등록에서도 역등록 방향에 대한 일관된 개선이 관찰되었습니다.



### Exogenous Randomness Empowering Random Forests (https://arxiv.org/abs/2411.07554)
Comments:
          103 pages, 10 figures

- **What's New**: 이 논문에서는 훈련 데이터와 독립적인 트리 구축 규칙을 가진 랜덤 포레스트의 효과에 대한 외부 난수성(exogenous randomness)의 영향을 이론적 및 경험적으로 살펴봅니다.

- **Technical Details**: 우리는 외부 난수성의 개념을 공식적으로 도입하고, 특성 서브샘플링(feature subsampling)에서 발생하는 Type I과 트리 구축 과정에서의 동점 해결(tie-breaking)에서 발생하는 Type II의 두 가지 일반적인 난수성을 식별합니다. 또한, 개별 트리와 랜덤 포레스트의 평균 제곱 오차(mean squared error, MSE)에 대한 비대칭(non-asymptotic) 확장을 개발하고, 일관성(consistency)을 위한 충분 및 필요 조건을 확립하였습니다. 독립적 특성을 가진 선형 회귀 모델의 특수 예제를 통해 MSE 확장은 더 명시적이며, 랜덤 포레스트의 메커니즘에 대한 이해를 제공합니다.

- **Performance Highlights**: 특성 서브샘플링이 개별 트리에 비해 랜덤 포레스트의 편향(bias)과 분산(variance)을 모두 줄이는 것으로 나타났습니다. 또한, 노이즈 특성이 랜덤 포레스트의 성능을 향상시키는 ‘축복(blessing)’으로 작용하는 흥미로운 현상도 발견했습니다.



### Effective Virtual Reality Teleoperation of an Upper-body Humanoid with Modified Task Jacobians and Relaxed Barrier Functions for Self-Collision Avoidanc (https://arxiv.org/abs/2411.07534)
Comments:
          XR & Robotics Workshop, IROS 2022

- **What's New**: 본 논문에서는 상용 Virtual Reality (VR) 트래커를 활용하여 상체를 가진 휴머노이드 로봇을 효과적으로 원거리 조작(teleoperate)할 수 있는 새로운 접근 방식을 제안합니다. 이 접근법은 각 조인트에 대한 트래커의 적절한 배정을 통해 자가 충돌(self-collision) 없는 동작을 보장합니다.

- **Technical Details**: 수정된 Task Jacobian을 통해 각 트래커가 어떤 조인트에 할당되는지를 정의함으로써 여러 문제를 해결합니다. 자가 충돌 회피를 위해 Relaxed Barrier Functions을 사용하고, Inverse Kinematics (IK) 문제를 해결하는 과정에서 이들 기능을 통합합니다.

- **Performance Highlights**: 실험적으로 Apptronik의 Astro 하드웨어에서 탁자 위의 물체 조작 및 두 손으로 상자를 들어주고 전달하는 작업을 수행하여 제안된 방법의 효용성을 입증하였습니다.



### SecEncoder: Logs are All You Need in Security (https://arxiv.org/abs/2411.07528)
- **What's New**: 이번 논문에서는 보안 로그(security logs)를 사용하여 사전 훈련된 SecEncoder라는 특화된 작은 언어 모델(small language model)을 소개합니다. 이는 일반적인 언어 모델들이 가지고 있는 도메인 특정 제한사항을 해결하고, 보안 로그에서 발견되는 고유한 언어와 패턴에 집중하기 위해 설계되었습니다.

- **Technical Details**: SecEncoder는 보안 로그 데이터 세트를 기반으로 사전 훈련된 인코더 전용 모델입니다. 이 모델은 다양한 보안 사건과 관련된 이벤트 및 활동을 포착하는 로그를 분석하고, 이상 탐지(anomaly detection), 로그 검색(log search), 사건 분류(incident classification)와 같은 작업에서 평가됩니다.

- **Performance Highlights**: SecEncoder는 BERTlarge, DeBERTa-v3-large 및 OpenAI의 Embedding(textembedding-ada-002) 모델보다 다양한 작업에서 우수한 성능을 보였습니다. 보안 로그에만 주로 사전 훈련되었음에도 불구하고, 사고 우선순위 설정과 위협 인텔리전스 문서 검색과 같은 로그 분석을 넘는 작업에서도 더 나은 성능을 나타냈습니다. 이는 보안 로그로의 도메인 특정 사전 훈련이 LMs의 성능을 상당히 향상시킬 수 있음을 시사합니다.



### AdaS&S: a One-Shot Supernet Approach for Automatic Embedding Size Search in Deep Recommender System (https://arxiv.org/abs/2411.07504)
- **What's New**: 본 논문에서는 AdaS&S라는 새로운 프레임워크를 제안하여 기존의 Automatic Embedding size Search (AES) 방법의 여러 문제를 해결하고자 하였습니다. 이 방법은 다양한 후보 임베딩을 포함하는 슈퍼넷(supernet)을 구성하고, 이를 통해 안정적이고 효과적인 임베딩 크기를 추출할 수 있도록 합니다.

- **Technical Details**: AdaS&S 프레임워크는 두 단계로 구성됩니다: 첫 번째 단계에서는 파라미터 훈련과 임베딩 크기 검색을 분리하여 Adaptive Sampling 방법을 통해 잘 훈련된 슈퍼넷을 생성합니다. 두 번째 단계에서는 강화 학습(Reinforcement Learning) 기반의 검색 과정을 통해 모델 성능을 향상시키는 임베딩 크기를 도출하며, 자원 제약(resource constraint)에 맞추기 위해 자원 경쟁 패널티를 도입합니다.

- **Performance Highlights**: AdaS&S 방법은 공공 데이터셋에서 실험을 통해 AUC를 약 0.3% 개선하고, 모델 파라미터를 약 20% 절감하는 성과를 보였습니다. 또한, 검색 결과의 안정성이 다른 방법들에 비해 현저히 뛰어난 것으로 나타났습니다.



### A Novel Automatic Real-time Motion Tracking Method for Magnetic Resonance Imaging-guided Radiotherapy: Leveraging the Enhanced Tracking-Learning-Detection Framework with Automatic Segmentation (https://arxiv.org/abs/2411.07503)
- **What's New**: 이번 연구에서는 MRI 유도 방사선 치료(MRIgRT)의 정확한 모션 트래킹(motion tracking)을 보장하기 위해 자동 실시간 트래킹 방법을 개선했습니다. ETLD(Enhanced Tracking-Learning-Detection) 프레임워크와 자동 분할(automatic segmentation)을 결합하여 ETLD+ICV(Improved Chan-Vese model)라는 새로운 방식을 구현하였습니다.

- **Technical Details**: ETLD+ICV 방법은 두 가지 주요 방법을 통합한 것입니다. TLD 프레임워크는 실시간 cine MRI에 적합하도록 업그레이드되었으며, 고급 이미지 전처리, 비참조 이미지 품질 평가, 향상된 메디안 흐름 추적기(median-flow tracker), 동적 검색 영역 조정이 가능한 정제된 탐지기를 포함합니다. ICV는 타겟 볼륨(target volume)을 정밀하게 커버하기 위해 사용하는 것으로, 트래킹 결과를 기반으로 분할된 영역을 프레임 별로 개선합니다.

- **Performance Highlights**: 106,000 프레임을 77개의 치료 분획(fraction)에 걸쳐 평가한 결과, 모든 대상에서 서브 밀리미터(less than 0.8mm)의 트래킹 오류와 99% 이상의 정밀도 및 98%의 재현율(recall)을 달성했습니다. ETLD+ICV는 모든 대상에서 82% 이상의 Dice global score를 기록하였으며, 이는 제안된 방법의 확장성과 정밀한 타겟 볼륨 커버리지를 잘 보여줍니다.



### ADMM for Structured Fractional Minimization (https://arxiv.org/abs/2411.07496)
- **What's New**: 이 논문은 구조화된 분수 최소화 문제(class of structured fractional minimization problems)에 대해 새로운 알고리즘인 FADMM을 소개합니다. 이 방법은 기존의 느린 수렴 속도와 수치적 안정성 문제를 해결합니다.

- **Technical Details**: FADMM(Alternating Direction Method of Multipliers)은 원래 문제를 선형화된 근접 하위 문제(linearized proximal subproblems)로 분리하고, Dinkelbach의 매개변수화 방법({\sf FADMM-D})와 이차 변환 방법({\sf FADMM-Q})의 두 가지 변형이 있습니다. 새로운 Lyapunov 함수를 도입함으로써 FADMM이 문제의 $\	ext{\epsilon}$-근사 임계점(critical points)에 수렴함을 $\	ext{\mathcal{O}(1/\epsilon^{3})}$의 오라클 복잡도(oracle complexity) 내에서 증명하였습니다.

- **Performance Highlights**: FADMM은 희소 피셔 판별 분석(sparse Fisher discriminant analysis), 강건 샤프 비율 최소화(robust Sharpe ratio minimization), 강건 희소 복구(robust sparse recovery) 문제에 대한 실험에서 효과성을 입증했습니다.



### Quantifying Knowledge Distillation Using Partial Information Decomposition (https://arxiv.org/abs/2411.07483)
Comments:
          Accepted at NeurIPS 2024 Machine Learning and Compression Workshop

- **What's New**: 본 논문은 Knowledge Distillation의 정보 이론적 한계를 규명하기 위한 새로운 메트릭을 도입했습니다. 이를 통해 교사 모델(teacher model)의 지식을 학생 모델(student model)과 특정 다운스트림 작업(downstream task)에 맞춰 정량화하는 방법을 제안합니다.

- **Technical Details**: 우리의 연구에서는 Partial Information Decomposition (PID)을 활용하여 교사가 제공할 수 있는 정보에 대한 새로운 양적 지표를 정의했습니다. 이 지표는 Task에 대한 교사만의 유일한 정보를 나타내며, 이로써 지식 증류 과정에서 필터를 통한 정보 정제 및 개선이 가능합니다.

- **Performance Highlights**: CIFAR10 데이터셋에 대한 실험을 통해, 새로운 Knowledge Distillation 프레임워크인 Redundant Information Distillation (RID)이 기존의 Variational Information Distillation (VID)보다 성능이 우수하다는 것을 입증했습니다.



### Privacy-Preserving Verifiable Neural Network Inference Servic (https://arxiv.org/abs/2411.07468)
Comments:
          This paper is to appear at the Annual Computer Security Applications Conference (ACSAC) 2024. The source code for our implementation can be found at $\href{this https URL}{this http URL}$

- **What's New**: 이 논문에서는 고객 데이터 개인정보 보호와 추론 결과의 verifiability를 보장하는 새로운 ML 추론 방식인 vPIN을 제안합니다.

- **Technical Details**: vPIN은 부분 동형 암호화(Partial Homomorphic Encryption)와 커밋-앤-프루브 방법론(Succinct Non-interactive Argument of Knowledge, SNARK)을 사용하여 클라이언트 데이터의 개인 정보를 보호하고 서버 계산의 무결성을 보장합니다. 또한, 새로운 최적화 기법을 도입하여 동형 추론 평가를 위한 프로빙 회로를 최소화하여 효율성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, vPIN은 증명 시간, 검증 시간 및 증명 크기 측면에서 높은 효율성을 달성했습니다. 이 방법은 MNIST 및 CIFAR-10와 같은 표준 데이터세트에서 테스트되었습니다.



### IdentifyMe: A Challenging Long-Context Mention Resolution Benchmark (https://arxiv.org/abs/2411.07466)
Comments:
          9 pages, 5 figures

- **What's New**: 이번 연구에서는 LLMs(대규모 언어 모델)의 핵심 연합 해결(coreference resolution) 능력을 평가하기 위한 새로운 벤치마크인 IdentifyMe를 소개합니다. 이 벤치마크는 다국적 선택형 질문(MCQ) 형식으로 제공되어 LLM의 참조 이해도를 보다 효과적으로 측정할 수 있도록 합니다.

- **Technical Details**: IdentifyMe는 문서에서 언급된 객체를 식별하는 MCQ 기반 벤치마크로, LitBank와 FantasyCoref라는 두 가지 장문 코어페런스 데이터셋에서 파생된 언급을 사용합니다. 벤치마크는 특정 유형의 언급(대명사 및 명사)을 필터링하고 난이도 조정을 위한 휴리스틱(heuristics)을 적용하여 모델이 보다 복잡한 문제를 해결하도록 만듭니다.

- **Performance Highlights**: 가장 높은 점수를 기록한 모델인 GPT-4o는 81.9%의 정확도를 달성하였으며, 이는 현재 LLMs의 참조 능력이 상당히 우수하지만 여전히 개선의 여지가 있음을 보여주고 있습니다. 또한, 모델은 대명사 언급을 해결하는 데 더 큰 어려움을 겪었으며 이는 표면 정보가 제한적이기 때문입니다.



### BudgetMLAgent: A Cost-Effective LLM Multi-Agent system for Automating Machine Learning Tasks (https://arxiv.org/abs/2411.07464)
Comments:
          Presented at AIMLSystems '24

- **What's New**: 이번 연구에서는 저비용 및 무비용 모델을 활용하여 복잡한 머신 러닝 (ML) 작업을 해결하기 위한 멀티 에이전트 시스템을 제안합니다. 이전 시스템들이 고비용의 대형 모델에 의존했던 반면, 이 새로운 접근법은 비용 효율성을 강조합니다.

- **Technical Details**: 다양한 LLM 전문가의 조합을 이용한 Multi-Agent 시스템을 사용하며, 여기에는 프로파일링, 과거 관찰의 효율적인 검색, LLM cascade, 간헐적인 전문가 호출을 포함합니다. 실험은 MLAgentBench 벤치마크에서 수행됩니다.

- **Performance Highlights**: 본 시스템은 GPT-4 기반 단일 에이전트 시스템에 비해 평균 94.2%의 비용 절감과 32.95%의 성공률을 기록했습니다. 이는 단일 에이전트 GPT-4 시스템의 22.72%의 성공률에 비해 월등히 높은 수치입니다.



### MSEG-VCUQ: Multimodal SEGmentation with Enhanced Vision Foundation Models, Convolutional Neural Networks, and Uncertainty Quantification for High-Speed Video Phase Detection Data (https://arxiv.org/abs/2411.07463)
Comments:
          Under Review in EAAI

- **What's New**: MSEG-VCUQ는 VideoSAM을 통해 고속 비디오 단계 감지(HSV PD) 분할을 위한 혁신적인 하이브리드 프레임워크를 제시합니다. 이 모델은 CNN과 transformer 기반 비전 모델을 결합하여 다중 모드 데이터에서의 분할 정확성을 향상시킵니다.

- **Technical Details**: VideoSAM은 U-Net(CNN)과 Segment Anything Model(SAM)을 통합하여 다양한 HSV PD 양식에서 고급 특징 추출 및 분할을 수행합니다. 이 프레임워크는 치수 기반 불확실성 정량화(UQ)도 포함하여 실험적 조건에 따른 신뢰할 수 있는 메트릭을 제공합니다.

- **Performance Highlights**: VideoSAM은 복잡한 위상 경계, 겹치는 기포 및 동적 액체-증기 상호작용이 있는 환경에서도 SAM 및 특정 모드 CNN 모델보다 뛰어난 분할 정확도를 보여줍니다. 이 모델은 다양한 데이터 세트에 효과적으로 적응하여 신뢰성이 높은 결과를 제공합니다.



### Optimizing Data Delivery: Insights from User Preferences on Visuals, Tables, and Tex (https://arxiv.org/abs/2411.07451)
- **What's New**: 이 연구는 사용자가 특정 질문에 대해 차트(chart), 테이블(table), 또는 텍스트(text) 중 어떤 형식으로 결과를 보고자 하는지를 조사합니다. 이를 통해 특정 질문에 대해 사용자에게 가장 적합한 결과 형식을 제시하는 방법을 이해합니다.

- **Technical Details**: 사용자 연구(user study)를 수행하여 사용자의 선호 데이터를 수집하였으며, 다양한 사용자 특성이 데이터 출력에 미치는 영향을 분석했습니다. 본 연구는 LLMs(대형 언어 모델)의 사용을 통해 사용자 선호를 복제할 수 있는 정도에 대해서도 탐구합니다.

- **Performance Highlights**: 본 연구는 사용자 특성이 데이터 출력 선호도에 미치는 영향을 분석하고, 특정 데이터 질문에 대해 사용자가 선호하는 출력 형식을 정량적으로 파악하는 데 기여합니다. LLM을 활용하여 사용자 선호도를 예측하는 가능성을 제시하며, 사용자 맞춤형 LLM의 효과성도 강조합니다.



### Just Label the Repeats for In-The-Wild Audio-to-Score Alignmen (https://arxiv.org/abs/2411.07428)
Comments:
          25th International Society for Music Information Retrieval Conference, San Francisco, 2024

- **What's New**: 본 연구에서는 오프라인 환경에서의 고품질 오디오와 악보의 정렬을 위한 효율적인 워크플로우를 제안합니다. 기존 연구는 dynamic time warping (DTW) 기법을 사용해 반복 기호가 포함된 악보의 점프를 처리하지만, 종종 낮은 품질의 정렬 결과를 제공합니다. 대신, 사용자가 반복 기호를 클릭해 점프를 빠르게 주석(annotation)할 수 있는 인터페이스를 도입하여, 인간의 감독을 최소화하면서도 훨씬 더 높은 품질의 정렬을 가능하게 합니다.

- **Technical Details**: 이 연구에서는 (1) 악보 기능 표현에 측정 지점 검출(measure detection)을 통합하고, (2) 피아노 롤 대신 음악 필기 모델의 원시 onset 예측 확률(raw onset prediction probabilities)을 오디오 기능 표현으로 활용하여 정렬 품질을 향상시켰습니다. 또한, 정렬 정확도를 평가하기 위한 새로운 평가 프로토콜을 제안하며, 이를 통해 추정된 정렬과 실제 정렬 간의 거리를 측정합니다.

- **Performance Highlights**: 제안된 점프 주석 워크플로우와 개선된 기능 표현들은 기존 연구보다 정렬 정확도를 상대적으로 150% 향상시켜 기존 33%에서 82%로 증가시킵니다. 실제로 재생 반복이 있는 악보를 대상으로 할 경우, 정확도는 20%에서 83%로 상승하는 결과를 보였습니다.



### Factorised Active Inference for Strategic Multi-Agent Interactions (https://arxiv.org/abs/2411.07362)
- **What's New**: 이 연구는 Active Inference framework(AIF)와 게임 이론을 통합하여 개인 에이전트가 다른 에이전트의 내부 상태에 대한 명확한 신념을 바탕으로 전략적 의사 결정을 내리는 방법을 제안합니다. 이를 통해 비정상적인 사회적 맥락에서의 에이전트의 적응 과정을 설명할 수 있습니다.

- **Technical Details**: 제안된 모델은 두 개 또는 세 개의 플레이어를 위한 반복적인 일반합 게임에 적용되어, 에이전트의 선호가 시간에 따라 어떻게 변화하는지를 연구합니다. 이 과정에서 사용하는 주요 요소는 변분 자유 에너지(Variational Free Energy, VFE)와 기대 자유 에너지(Expected Free Energy, EFE)입니다. EFE를 통해 다수의 Nash 균형을 특징짓는 매력의 분산(basins of attraction)을 분석하고, 집합 수준의 최소화가 항상 이루어지지 않음을 발견했습니다.

- **Performance Highlights**: 제안된 모델은 AIF와 게임 이론의 통합을 통해 지능 집단이 동적 환경에서 어떻게 행동을 최적화하고 학습하는지를 보다 깊이 있게 이해할 수 있도록 돕습니다. 연구 결과는 협력적 또는 비협력적 상황 모두에서 에이전트가 변화하는 사회적 맥락에 적응하는 방식을 설명하는 데 기여합니다.



### Richer Output for Richer Countries: Uncovering Geographical Disparities in Generated Stories and Travel Recommendations (https://arxiv.org/abs/2411.07320)
Comments:
          Submitted to ARR - October 2024

- **What's New**: 이번 연구는 대형 언어 모델이 지리적 지식에 대한 편향을 분석하며, 여행 추천과 지리 기반 이야기 생성의 두 가지 일반적인 시나리오를 탐구합니다. 특히, 저소득 국가에 대한 추천이 상대적으로 덜 독창적이며 빈곤과 슬픔의 감정을 더 많이 포함하고 있음을 발견했습니다.

- **Technical Details**: 연구는 ShareGPT 데이터를 기반으로 1.7%의 쿼리가 여행 추천, 1.5%가 이야기 생성에 관한 것임을 파악했습니다. 444개의 모델에서 300K 이상의 응답을 분석했으며, 이는 전 세계 150K 이상의 장소에 걸쳐 있습니다. 각 모델에 대해 평균 독창성과 감정 표현을 비교했습니다.

- **Performance Highlights**: 부유한 국가에 비해 저소득 국가에서 생성된 이야기의 65%가 더 많은 고난의 정서를 담고 있으며, 여행 추천은 평균적으로 40% 이상의 독창성 차이를 보였습니다. 이러한 결과는 현재 모델들이 서구 중심적 내용을 생성하고 있음을 나타내며, 다양한 인구 집단에 대한 서비스를 보장하기 위한 더 큰 노력이 필요함을 강조합니다.



### Merit-Based Sortition in Decentralized Systems (https://arxiv.org/abs/2411.07302)
Comments:
          8 pages, 4 figures; appeared in ADI (October 2024)

- **What's New**: 이 논문에서 제안된 'merit-based sortition' 알고리즘은 비활성 참가자에게도 무수히 많은 기회를 제공하며, 성능 최적화를 통해 활성 집합의 품질을 두 배 이상 증가시킵니다.

- **Technical Details**: 'Merit-based sortition' 알고리즘은 참가자의 품질이 활성 집합에 초대될 확률에 영향을 미치도록 설계되었습니다. 이 알고리즘은 비활성 참가자에게도 актив 집합으로 초대될 기회를 무한히 제공하여, 성능을 높이는 동시에 대표성도 유지합니다.

- **Performance Highlights**: 실험 결과, 이 알고리즘은 활성 집합의 성능을 2배 이상 증가시키며 이를 통해 성능 최적화를 필요로 하는 분산 시스템의 요구를 만족시킵니다.



### Artificial Intelligence Ecosystem for Automating Self-Directed Teaching (https://arxiv.org/abs/2411.07300)
Comments:
          13 pages, 15 figures, 12 references and 1 table

- **What's New**: 이 연구에서는 인공지능(AI) 기반 교육 개념이 도입되었습니다. 이는 개인 맞춤형 강의 제공 및 자동화된 교수 지원을 통해 자율 학습(self-directed learning)을 최적화하는 것을 목표로 합니다.

- **Technical Details**: 시스템은 세밀하게 조정된 AI 모델을 활용하여 맞춤형 로드맵(customized roadmaps), 자동 프레젠테이션 생성(automated presentation generation), 복잡한 개념 시각화를 위한 3D 모델링(three-dimensional modeling) 등으로 구성된 적응형 학습 환경을 제공합니다. 실시간 가상 지원(real-time virtual assistance)을 통합하여 학습자의 즉각적인 교육 필요를 충족하며 자율 학습을 촉진합니다.

- **Performance Highlights**: 이 연구는 자율 학습의 심리적 장점을 탐구하고 AI 자동화가 개인화된 콘텐츠 제공과 상호작용 지원 메커니즘을 통해 교육 결과를 향상시킬 수 있음을 보여줍니다. 예비 결과는 이 접근 방식이 다양한 학습 스타일을 지원하고 자율적이며 독립적인 학습 방법론을 강조함으로써 학생의 참여도와 지식 유지력을 강화함을 시사합니다.



### The Surprising Effectiveness of Test-Time Training for Abstract Reasoning (https://arxiv.org/abs/2411.07279)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 모델의 추론 과정에서 입력 데이터에 기반한 손실을 사용하여 모델 파라미터를 임시로 업데이트하는 테스트 타임 트레이닝(test-time training, TTT)이 모델의 추론 능력을 향상시키는 효과를 탐구했습니다. 특히 Abstraction and Reasoning Corpus (ARC)를 기준으로 효율적인 TTT의 세 가지 주요 요소를 식별했습니다.

- **Technical Details**: TTT의 성공을 위해 세 가지 주 구성 요소가 필요합니다: (1) 유사한 작업에서의 초기 파인튜닝, (2) 보조 작업 형식 및 데이터 증강(augmentation), (3) 인스턴스별 훈련. 이 방법은 LMs가 기존의 Fine-tuned 모델보다 최대 6배 더 높은 정확도를 달성하도록 돕습니다. 8B 파라미터 언어 모델에 TTT를 적용했을 때, ARC의 공개 검증 세트에서 53%의 정확도를 달성했습니다.

- **Performance Highlights**: 우리의 연구 결과, TTT를 통해 모델이 네오-심볼릭 접근 방식의 성능에 버금가는 결과를 도출할 수 있음을 보여주었습니다. 최근의 프로그램 생성 접근 방식과 조합하여, ARC 검증 세트에서 61.9%의 최첨단(public validation SoTA) 정확도를 달성했습니다. 이는 평균 인간 성능에 해당하는 수치입니다.



### Constructing Gaussian Processes via Samplets (https://arxiv.org/abs/2411.07277)
- **What's New**: 이 논문은 Gaussian Processes의 두 가지 주요 문제, 즉 대규모 데이터셋을 위한 모델 구축과 최적 모델 선택을 다룹니다.

- **Technical Details**: 저자는 최근의 수렴(results) 결과를 검토하여 최적 수렴 속도를 가진 모델을 식별하고 필수 파라미터(parameters)를 정립했습니다. 이 모델을 기반으로 샘플트(samplet)-기반 접근법을 제안하여 Gaussian Processes를 효율적으로 구축하고 훈련하는 방법을 제공합니다.

- **Performance Highlights**: 이 방법은 계산 복잡성을 세제곱(cubic)에서 로그-선형(log-linear) 규모로 줄이며, 최적 회귀(optimal regression)를 가능하게 하면서 효율적인 성능을 유지합니다.



### Empirical Quantum Advantage Analysis of Quantum Kernel in Gene Expression Data (https://arxiv.org/abs/2411.07276)
Comments:
          5 pages

- **What's New**: 이 논문에서는 기계학습 분류 모델에 양자 원리를 도입하여 데이터에서 패턴을 추출하는 능력을 보여줍니다. 특히, 양자 기계학습의 유리성을 확인할 수 있는 데이터셋 선정과 전통적 및 양자 방법으로 선택한 특징의 관련성을 평가하는 데 중점을 두었습니다.

- **Technical Details**: 본 연구는 Gene Expression 데이터셋을 사용하여 전통적인 방법과 양자 방법의 효과성을 비교하였습니다. 데이터 전처리 시 Quantile Normalization을 적용하였고, Lasso 정규화(L1) 기법을 활용해 20개의 중요한 특징을 추출했습니다. 양자 특징 선택에는 D-Wave의 하이브리드 양자-고전적 프레임워크를 이용하여 QUBO 문제를 정의했습니다. 또한, SVM을 이용하여 분류 작업을 수행하고, 양자 커널 추정 방법을 적용하여 결정을 내렸습니다.

- **Performance Highlights**: 클래식 커널은 14개의 특징과 25개의 샘플에서 F1 점수 .93을 기록하였고, 양자 커널은 8개의 특징과 57개의 샘플에서 .85의 점수를 보였습니다. 결과적으로, 양자 커널이 성능 면에서 전통적인 방법에 비해 낮은 정확도를 보였습니다. 그러나 다양한 구성에서 양자와 고전적 방법의 교차점을 분석하여 최적의 결정 경계 생성을 위한 커널의 효율성에 대한 통찰을 제공하였습니다.



### ASTD Patterns for Integrated Continuous Anomaly Detection In Data Logs (https://arxiv.org/abs/2411.07272)
- **What's New**: 이 논문은 데이터 로그에서 앙상블 이상 탐지를 위한 ASTD 언어의 사용을 조사합니다. 슬라이딩 윈도우 기법을 활용하여 지속적인 학습을 수행하며, 각 윈도우가 완료될 때마다 학습 모델을 업데이트하여 정확한 탐지를 유지하고 현재 데이터 트렌드에 맞춥니다.

- **Technical Details**: ASTD 언어는 이상 탐지 시스템을 추상화하고 모듈화할 수 있는 능력을 강조합니다. 논문은 모델을 결합하기 위한 새로운 ASTD 연산자인 Quantified Flow를 제안하여 다양한 모델을 세련되게 결합할 수 있도록 합니다. 이 시스템은 학습 모델의 자동 재훈련 및 이벤트에 대한 모델 결합을 포함한 기능을 갖추고 있습니다.

- **Performance Highlights**: 논문에서는 데이터 로그에서 예상치 못한 이벤트를 탐지하기 위한 사례 연구를 통해 자동 교육 갱신과 비지도 모델의 결합의 효과를 보여줍니다. 성능 평가 결과, ASTD 언어의 사용은 이상 탐지 과정의 복잡성을 줄이고 개발의 용이성을 향상시킴을 나타냅니다.



### High quality ECG dataset based on MIT-BIH recordings for improved heartbeats classification (https://arxiv.org/abs/2411.07252)
Comments:
          4 pages, 5 figures, 5 tables, presented during IEEE COINS 2023 Berlin. link to ieeexploere: this https URL

- **What's New**: 본 논문에서는 MIT-BIH 기록을 기반으로 하여 새로운 고품질 심박수 데이터세트를 생성하는 방법론을 제안합니다. 이 방법론은 이상치를 제거하고 10초 창(window)에서 평균 값을 계산하여 최적의 심박수 크기를 산출하는 방식으로, 연속된 심박수 혼합 문제를 피할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법론은 IQR 방법론을 사용하여 이상 심박수를 제거하며, 현재 10초 창에서 RR 시간 간격의 평균 값에 따라 적응적인 심박수 크기를 계산합니다. 이는 QRS 중심의 심박수를 생성하게 하여 연속적인 심박수 혼합 문제를 해결합니다. 연구팀은 또한 1-D ResNet 아키텍처 모델을 개발하여 새로운 데이터세트의 성능을 평가하였습니다.

- **Performance Highlights**: 개발된 1-D ResNet 모델은 99.24%의 정확도를 달성하였으며, 이는 기존 방법에 비해 5.7% 향상된 수치입니다. 데이터세트를 다운샘플링하여 모델의 실행 시간을 33% 단축하고 메모리 사용량도 3배 감소시켰습니다.



### SPDIM: Source-Free Unsupervised Conditional and Label Shifts Adaptation in EEG (https://arxiv.org/abs/2411.07249)
- **What's New**: 이번 논문에서는 비표기 조정이 필요한 EEG 기반의 신경기술에 대한 문제를 해결하기 위해 새로운 기하학적 딥러닝 프레임워크(SPDIM)를 제안합니다. 기존의 표준 Riemannian 통계 정렬 방법은 레이블 변화(label shift)에 대해 일반화 능력을 떨어뜨리는 반면, 제안된 SPDIM은 이러한 변화를 효과적으로 보상할 수 있습니다.

- **Technical Details**: 제안된 SPDIM은 정보 최대화(information maximization) 원리를 활용하여 각 목표 도메인에 대해 단일 SPD(대칭 양의 정부호 symmetric positive definite) 매니폴드 제약(parameter) 특정 오버 보정을 조정하는 기법을 제공합니다. 기존의 방법들은 주로 시커먼 행동 또는 조건적 분포 변화에 대한 조정에 중점을 두고 있지만, SPDIM은 레이블 변화를 내포하는 여러 분포 변화에 맞출 수 있습니다.

- **Performance Highlights**: 시뮬레이션과 공개 EEG 기반 뇌-컴퓨터 인터페이스 및 수면 단계 데이터세트를 활용하여 SPDIM이 기존 접근 방식(domain adaptation approaches)보다 성능이 우수함을 입증하였습니다. 이를 통해 EEG 기반 기술의 적용 가능성과 확장성을 높이는 데 기여할 것으로 기대됩니다.




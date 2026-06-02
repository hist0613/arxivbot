New uploads on arXiv(cs.CL)

### Tool-to-Agent Retrieval: Bridging Tools and Agents for Scalable LLM Multi-Agent Systems (https://arxiv.org/abs/2511.01854)
- **What's New**: 최근 LLM(Multi-Agent Systems)에서의 발전은 하위 에이전트의 스케일링을 가능하게 하며, 각 에이전트가 수백 또는 수천 개의 툴(tool)이나 MCP(Model Context Protocol) 서버를 조정할 수 있도록 합니다. 그러나 기존의 검색 방법은 일반적으로 에이전트 수준의 간단한 설명으로 쿼리와 매칭하여 신뢰성을 저하시킵니다. 본 논문에서는 Tool-to-Agent Retrieval이라는 새로운 프레임워크를 제시하여, 툴과 해당 부모 에이전트를 공유된 벡터 공간에 임베딩하고 메타데이터 관계로 연결합니다.

- **Technical Details**: Tool-to-Agent Retrieval은 툴과 부모 에이전트를 통합된 벡터 공간에 임베딩하며, 메타데이터 관계를 통해 각 툴을 해당 부모 에이전트에 명확히 연결합니다. 이를 통해 에이전트의 컨텍스트를 유지하면서 미세한 툴 세부 정보를 보존하는 검색 절차를 제안합니다. 또한, 쿼리 시 툴 또는 에이전트 번들을 반환하는 방식으로, 복잡한 툴 기능을 간과하지 않고 검색합니다.

- **Performance Highlights**: Tool-to-Agent Retrieval은 LiveMCPBench 벤치마크에서 8개 임베딩 모델을 대상으로 평가되었으며, 기존 최첨단 에이전트 검색기 대비 Recall@5에서 19.4% 및 nDCG@5에서 17.7% 향상을 보였습니다. 이러한 결과는 복잡한 멀티 단계 쿼리에서도 에이전트와 툴 간의 조정된 선택이 가능함을 보여줍니다.



### Towards Robust Mathematical Reasoning (https://arxiv.org/abs/2511.01846)
Comments:
          EMNLP 2025 (main conference), this https URL

- **What's New**: 이번 논문에서는 IMO-Bench라는 새로운 벤치마크 세트를 소개합니다. 이는 국제 수학 올림피아드(IMO)의 수준에서 AI 모델의 수학적 추론 능력을 평가하기 위해 전문 패널에 의해 검증된 고급 레벨의 테스트 문제로 구성되어 있습니다. IMO-AnswerBench와 IMO-ProofBench라는 두 가지 주요 평가 도구는 모델의 성능을 측정하고 효율적인 자동 평가를 위한 기준을 제공합니다.

- **Technical Details**: IMO-AnswerBench는 과거 올림피아드 문제로부터 선택된 400개의 다양한 문제로 구성되어 있으며, 문제는 기본, 중급, 고급 등 다양한 난이도로 구분됩니다. IMO-ProofBench는 기본 및 고급 문제 각각 30문제씩 총 60문제를 포함하고, 각 문제는 완전한 증명 생성을 요구합니다. 이러한 벤치마크는 AI 모델이 간단한 정답을 넘어서 깊이 있는 논리적 추론을 수행할 수 있도록 설계되었습니다.

- **Performance Highlights**: Gemini Deep Think 모델은 IMO-AnswerBench에서 80.0%의 정확도를 기록했으며, 이는 비-Gemini 모델보다 6.9% 높은 성과입니다. IMO-ProofBench에서는 65.7%의 정확도를 달성하여 비교군을 크게 앞섰습니다. 자동 채점기는 인간 평가와 높은 상관관계를 보였고, 이는 향후 수학적 추론의 자동 평가를 위한 중요한 기초 자료로 활용될 것입니다.



### KV Cache Transform Coding for Compact Storage in LLM Inferenc (https://arxiv.org/abs/2511.01815)
- **What's New**: 이번 논문은 대규모 언어 모델(LLM)을 효율적으로 운영하기 위한 키-값(KV) 캐시 관리 기법을 소개하고 있습니다. 특히, 반복적인 코드 편집 및 대화에서 공유되는 접두사를 활용해 KV 캐시를 재사용하는 방안을 제안합니다. 연구팀은 'KVTC'라는 경량 변환 코더를 통해 KV 캐시를 압축하여 GPU와 off-GPU에 저장할 수 있도록 하고 있습니다.

- **Technical Details**: KVTC는 PCA 기반(feature decorrelation) 특성 분산, 적응형 양자화(adaptive quantization), 및 엔트로피 코딩(entropy coding)을 결합하여 클래식 미디어 압축 기법에 기반하고 있습니다. 이 방식은 초기 캘리브레이션(calibration)만으로 두며 모델 파라미터는 변경하지 않습니다. KV 캐시에서 중복성을 활용하여 최대 20배의 압축률을 달성하며, 특정 사용 사례에서는 40배 이상의 압축도 가능하다고 합니다.

- **Performance Highlights**: KVTC는 Llama 3, Mistral NeMo, R1-Qwen 2.5 모델을 이용한 다양한 벤치마크 테스트에서 개선된 성능을 보였습니다. AIME25, LiveCodeBench, GSM8K, MMLU, Qasper, RULER, MATH-500 등에서 기존의 토큰 삭제(token eviction)와 양자화(quantization) 기법보다 항상 높은 압축 비율을 보여주었습니다. 이러한 결과는 KVTC가 메모리 효율적인 LLM 서비스 제공을 위한 실제적인 구성 요소로 자리잡을 수 있다는 것을 입증합니다.



### Plan-and-Write: Structure-Guided Length Control for LLMs without Model Retraining (https://arxiv.org/abs/2511.01807)
Comments:
          Presented at Workshop on Prompt Optimization, KDD 2025, Toronto, Canada

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 길이 조절(length control) 문제를 효과적으로 해결할 수 있는 새로운 방법론인 'Plan-and-Write'를 제안합니다. 이 방법은 모델의 재훈련 없이도 길이 조절을 가능하게 하며, 특히 문서 요약(document summarization) 작업에서 뛰어난 성과를 입증했습니다. 제안된 방법은 텍스트 생성 과정에서 단어 수 세기(word counting)와 계획 수립 계획을 포함하여 모델이 출력 길이를 전략적으로 조정할 수 있도록 합니다.

- **Technical Details**: Plan-and-Write 접근법은 생성을 두 단계로 나누어 실행됩니다: 콘텐츠 드래프트 단계에서 단어 수를 명시적으로 계산하고, 지정된 길이에 맞게 콘텐츠를 재구성하는 검증 단계입니다. 이 구조적 접근은 LLM의 메타 인지와 자아 모니터링 능력을 활용하면서 모델 파라미터를 수정할 필요가 없습니다. 이 방법은 모든 LLM에 적용 가능한 모델 불문식으로 설계되어 있습니다.

- **Performance Highlights**: 제안된 방법은 표준 프롬프트 방식과 비교할 때 길이 충실도(length fidelity)에서 유의미한 개선을 보였습니다. 6가지 최첨단 LLM에 대한 포괄적인 평가에서는 일부 모델에서 최대 37.6%의 길이 준수 개선이 있음을 확인했습니다. 또한 품질 평가에서도 Plan-and-Write 접근법이 응답 품질을 유지하거나 개선하는 결과를 나타냈으며, 이는 생산 환경에서 즉시 배포 가능한 솔루션을 제공합니다.



### Accumulating Context Changes the Beliefs of Language Models (https://arxiv.org/abs/2511.01805)
- **What's New**: 이 논문에서는 언어 모델 (Language Model) 보조 도구가 대화 및 텍스트 처리 과정에서 어떻게 신념 프로필이 변화할 수 있는지를 탐구합니다. 메모리(memory)와 문맥 크기(context size)의 개선이 모델의 자율성을 향상시켜, 사용자 개입 없이 텍스트가 쌓이는 경우 발생할 수 있는 잠재적 위험에 대해 다룹니다. 이러한 변화가 사용자 경험에 미치는 부정적인 영향을 강조합니다.

- **Technical Details**: 연구에서 GPT-5는 도덕적 딜레마에 대해 10번의 논의 후 54.7%의 신념 변화가 있었으며, Grok 4는 상반된 입장에 대한 텍스트를 읽고 난 후 정치적 문제에서 27.2%의 신념 변화를 보였습니다. 또한 도구 사용을 요구하는 작업을 설계하여 각 도구 선택이 암묵적인 신념과 어떻게 연결되는지를 분석했습니다. 이 과정에서 모델의 행동 변화가 신념의 변화와 일치함을 발견하였습니다.

- **Performance Highlights**: 모델의 신념 프로필이 매우 가변적임을 보여주며, 이는 언어 모델이 대화나 독서를 통한 장시간 상호작용 후 더욱 도드라집니다. 이러한 신념 변화는 실제 행동에 반영되며, 따라서 언어 모델의 의견과 행동의 신뢰성을 저하시킬 수 있는 숨겨진 위험을 드러냅니다. 이 연구는 사용자 경험의 일관성을 유지하기 위한 관리 필요성을 제기합니다.



### Efficient Tool-Calling Multi-Expert NPC Agent for Commonsense Persona-Grounded Dialogu (https://arxiv.org/abs/2511.01720)
Comments:
          10 pages, 1 figure, 2 tables. Technical report for the Commonsense Persona-Grounded Dialogue Challenge (CPDC) 2025, part of the Wordplay 2025 Workshop @ EMNLP 2025

- **What's New**: 이 논문에서는 다중 전문가 시스템을 통해 자연스러운 대화와 상황적 행동 수행이 가능한 비행상 NPC(Non-Player Characters)를 만드는 방법을 제안합니다. 기초 모델로 Qwen3를 사용하고 Low-Rank Adaptation(LoRA) 어댑터를 이용해 툴 호출, 툴 응답 해석, 직접 대화의 세 가지 전문 분야를 구현하였습니다. 우리의 시스템은 L40S GPU에서 빠른 응답을 제공하며 자원 사용은 적절히 유지됩니다.

- **Technical Details**: Qwen3는 도구 호출 능력에서 뛰어난 성능을 보여 주며, 각 NPC 턴의 7초 시간 제약을 충족하기 위해 고급 추론 능력은 일부러 비활성화되었습니다. 이는 대회에서의 효율성을 높이기 위한 결정이었습니다. Unsloth는 대형 언어 모델의 파인튜닝 및 추론을 가속화하는 데 중요한 최적화 라이브러리로, 모델 선택에 있어 중대한 고려 요소였습니다.

- **Performance Highlights**: Commonsense Persona-Grounded Dialogue Challenge 2025에서 이 시스템은 전반적으로 2위를 차지했습니다. 데이터 증강 전략을 통해 원본 데이터의 크기를 약 300% 증가시킨 결과, 모델 성능 또한 유의미하게 향상되었습니다. 특히 Qwen3-1.7B 모델은 Task 1에서 성능이 20% 이상 증가했으며, 이는 여러 모델 규모에서 일관된 개선을 보여주는 결과입니다.



### Multi-Step Knowledge Interaction Analysis via Rank-2 Subspace Disentanglemen (https://arxiv.org/abs/2511.01706)
Comments:
          Under review

- **What's New**: 이번 연구는 LLMs (Large Language Models)의 Natural Language Explanations (NLEs) 생성을 위한 PK (Parametric Knowledge)와 CK (Context Knowledge)의 상호작용을 새롭게 분석합니다. 기존의 이론은 주로 단일 단계 생성에 초점을 맞췄으며, PK와 CK의 상호작용을 rank-1 공간에서 모델링했습니다. 그러나, 이 연구는 PK와 CK의 기여를 더 정확하게 구분할 수 있는 새로운 rank-2 투영(subspace)을 제안합니다.

- **Technical Details**: 이 연구는 여러 QA 데이터셋과 오픈 가중치로 튜닝된 LLM를 사용하여 PK와 CK의 상호작용을 심층적으로 분석합니다. 다단계 NLE 생성 과정에서 연구진은 PK와 CK가 어떻게 다른 방식으로 기여하는지를 밝혀냈고, 이는 종합적인 rank-2 아키텍처를 통해 수행되었습니다. 이 새로운 모델은 NLE의 신뢰성과 통찰을 향상시키기 위해 PK와 CK 간의 관계를 더욱 면밀히 조명합니다.

- **Performance Highlights**: 실험 결과, rank-1 subspace에서는 다양한 지식 상호작용이 잘 포착되지 않았으나, rank-2 구조에서는 두 공헌의 균형을 맞추며 NLE 생성을 명확히 표현했습니다. 또한, CoT (Chain-of-Thought) 방법론이 PK 의존성을 감소시켜 CK 방향으로 생성된 NLE를 유도한다는 사실이 확인되었습니다. 이러한 결과는 LLMs의 내적 및 외적 지식 통합에 대한 이해를 한층 높이며, 향후 연구의 기초 자료로 활용될 수 있습니다.



### Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI (https://arxiv.org/abs/2511.01689)
Comments:
          12 pages, 6 figures, 4 tables

- **What's New**: 이번 연구는 인공지능(AI) 어시스턴트의 페르소나(persona) 개발에 관한 새로운 접근 방식을 소개합니다. 캐릭터 훈련(character training)이라고 알려진 이 과정은 현대 챗봇의 행동과 가치, 신념에 영향을 미치며, 상호작용의 질과 사용자 및 개발자의 의도에 대한 정렬에 필수적입니다. 특히, 캐릭터 훈련의 첫 번째 공개 구현을 제공하여 기존의 방식보다 효과적인 인공지능 어시스턴트 페르소나를 생성하는 방법을 제시합니다.

- **Technical Details**: 연구에서는 11개의 예시 페르소나를 사용하여 세 가지 인기 있는 오픈 소스 모델을 미세 조정(fine-tune)합니다. 이러한 페르소나는 유머러스(humorous), 깊은 배려(deeply caring) 또는 심지어 악의적(malevolent)인 성격을 포함합니다. 연구진은 Constitution AI와 합성 자기 성찰 데이터(synthetic introspective data)를 활용한 새로운 데이터 파이프라인을 통해 페르소나 형성을 최적화하는 방법론을 개발하였습니다.

- **Performance Highlights**: 본 연구의 접근 방식은 적대적 프롬프트(adversarial prompting)에도 강한 내성을 보이며, 생성되는 내용의 일관성과 현실성 또한 향상됩니다. 일반적인 벤치마크에서 측정한 바와 같이, 이러한 미세 조정은 AI의 일반적인 능력에는 거의 영향을 미치지 않음을 보여줍니다. 최종적으로, 연구진은 전체 후속 훈련(post-training) 방법을 설명하고, 이를 오픈 소스로 공개하여 다른 연구자들이 사용할 수 있도록 배포합니다.



### SeaLLMs-Audio: Large Audio-Language Models for Southeast Asia (https://arxiv.org/abs/2511.01670)
Comments:
          10 pages

- **What's New**: SeaLLMs-Audio는 인도네시아어(id), 태국어(th), 베트남어(vi), 영어(en), 중국어(zh) 등 5개 동남아시아(SEA) 언어에 특화된 최초의 대규모 오디오-언어 모델(LALM)입니다. 이 모델은 다양한 오디오 중심 과제를 위한 대규모 오디오 코퍼스를 기반으로 학습하였으며, 음성 기반 상호작용과 세밀한 오디오 이해에 강력한 성능을 보입니다. 각각의 언어를 위한 다국어 지원 뿐만 아니라, 오디오와 텍스트 입력을 모두 수용하는 다중 모드 기능을 제공하는 것이 특징입니다.

- **Technical Details**: SeaLLMs-Audio는 1.58M 개의 대화를 포함한 방대한 훈련 데이터를 기반으로 하여, 자동 음성 인식(ASR), 오디오 캡셔닝(AC), 음성-텍스트 번역(S2TT), 음성 요약(SS)과 같은 다양한 과제를 지원합니다. 이러한 데이터는 공공 및 사설 데이터셋을 통해 수집되었으며, 데이터 전처리를 통해 서로 다른 형식을 통합하는 과정을 거쳐야 했습니다. 특히, 기존의 ASR 데이터와 다른 언어로의 텍스트 변환을 결합하여 다양한 언어 쌍 성과 데이터를 생성하였습니다.

- **Performance Highlights**: SeaLLMs-Audio는 SeaBench-Audio라는 새로운 벤치마크를 통해 동남아시아 언어에 대해 강력하고 경쟁력 있는 성능을 입증하였습니다. 이 벤치마크는 LALM의 표준화된 평가를 위해 설계되었으며, 다양한 실제 언어 이해 시나리오를 반영하는 여러 개방형 과제를 포함하고 있습니다. 실험 결과, SeaLLMs-Audio는 여러 오디오-언어 작업에서 우수한 성능을 달성하며, SEA 지역 연구 및 산업에 기여할 것으로 기대됩니다.



### EngChain: A Symbolic Benchmark for Verifiable Multi-Step Reasoning in Engineering (https://arxiv.org/abs/2511.01650)
Comments:
          24 pages, includes figures and tables; introduces the EngChain benchmark

- **What's New**: 이 연구에서는 EngChain이라는 새로운 기준을 도입하여 검증 가능한 다단계 공학 문제 해결을 위한 벤치마크를 제안합니다. EngChain은 90가지 문제를 포함하고 있으며, 9개 도메인 및 20개 영역으로 조직되어 있습니다. 문제는 심볼릭 템플릿을 이용해 생성되어 다양성을 확보하고 오염 위험성을 줄이는 데 초점을 맞추고 있습니다.

- **Technical Details**: EngChain은 기존 벤치마크에서 부족했던 종합적 사고를 평가하기 위해 기반을 두고 있습니다. 평가 방식은 두 단계로 나뉘어 있으며, 첫 번째 단계에서는 각 사고 과정의 수치적 및 의미적 유효성을 정량적으로 검증하고, 두 번째 단계에서는 LLM-As-A-Judge라는 자동화된 시스템을 통해 사고 오류를 질적으로 분류합니다.

- **Performance Highlights**: 연구 결과는 11개의 가장 앞선 LLM 모델들의 성능을 포괄적으로 분석하여, 다수의 추론 오류가 계산 실수가 아닌 개념적 오류에 기인한다는 점을 밝혔습니다. 이는 EngChain이 공학 문제의 복잡한 추론 능력을 평가하는 데 진정으로 유용한 도구임을 보여줍니다.



### Evaluating Cultural Knowledge Processing in Large Language Models: A Cognitive Benchmarking Framework Integrating Retrieval-Augmented Generation (https://arxiv.org/abs/2511.01649)
Comments:
          This paper has been accepted by The Electronic Library, and the full article is now available on Emerald Insight

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 문화적으로 특정한 지식을 처리하고 적용하는 방식을 평가하기 위한 인지 벤치마킹 프레임워크를 제안합니다. 이 프레임워크는 Bloom's Taxonomy와 Retrieval-Augmented Generation(RAG)을 통합하여 모델의 성능을 6개의 계층적 인지 영역에서 평가합니다. 이는 기억(Remembering), 이해(Understanding), 적용(Applying), 분석(Analyzing), 평가(Evaluating), 창조(Creating)의 영역을 포함합니다.

- **Technical Details**: 평가 과정에서는 대만 하카 디지털 문화 아카이브를 주요 시험대(testbed)로 활용하였습니다. 이 프레임워크는 LLM이 생성한 응답의 의미적 정확성과 문화적 관련성을 측정합니다. 또한, 이 평가 방식을 통해 LLM의 학습과정에서 문화적 맥락의 중요성을 강조합니다.

- **Performance Highlights**: 제안된 프레임워크는 LLM의 수행 능력을 평가할 수 있는 새로운 기준을 제공하며, 이는 다양한 문화적 배경에서 모델의 정합성을 증진할 수 있는 가능성을 가지고 있습니다. 특히, 대만 하카 문화 아카이브를 활용함으로써 실제 적용 사례를 제시하고, 문화적으로 민감한 알고리즘 개발에 기여할 수 있음을 보여줍니다.



### A Graph-based RAG for Energy Efficiency Question Answering (https://arxiv.org/abs/2511.01643)
- **What's New**: 이 연구에서는 에너지 효율성(EE) 질문 응답을 위한 그래프 기반의 Retrieval Augmented Generation (RAG) 아키텍처 내에서 대형 언어 모델(LLM)의 사용을 조사했습니다. 이 시스템은 에너지 분야의 가이드라인과 규제 문서에서 자동으로 지식 그래프(KG)를 추출하고, 이 그래프를 탐색하여 사용자에게 다양한 언어로 정확한 답변을 제공합니다. 인간 기반의 검증 실험을 통해 이 아키텍처의 잠재력을 확인하고 강점과 약점을 파악하였습니다.

- **Technical Details**: 시스템은 세 가지 주요 구성요소로 나뉘며, 첫 번째로 지식 추출기가 도메인 특정 문서에서 엔티티와 관계를 포함한 트리플을 추출합니다. 이렇게 추출된 정보는 지식 기반(KB)을 구축하는 데 사용되며, 이후 사용자의 질문을 처리하기 위한 검색 및 생성 프로세스가 이어집니다. 이 과정에서 LLM 기반의 알고리즘이 사용되어 트리플을 자동으로 추출하고, 동일한 구문이 적용되도록 엔티티 이름을 통합합니다.

- **Performance Highlights**: 검증 결과, 시스템은 약 75.2%의 경우에 정확한 답변을 제공하며, 에너지 효율성에 대한 일반 질문에 대해서는 81.0%까지 높아지는 성능을 보입니다. 또한, 다국어 처리에 있어 번역으로 인한 정확도 손실은 4.4%로 나타났습니다. 이러한 결과는 그래프 기반 RAG 아키텍처가 다국적 사용자를 위한 효율적인 질문 응답 시스템을 구축할 수 있는 잠재력을 가지고 있음을 시사합니다.



### ParlaSpeech 3.0: Richly Annotated Spoken Parliamentary Corpora of Croatian, Czech, Polish, and Serbian (https://arxiv.org/abs/2511.01619)
Comments:
          Submitted to the LREC 2026 conference; 11 pages, 2 figures, 3 tables

- **What's New**: ParlaSpeech는 총 6천 시간 분량의 크로아티아어, 체코어, 폴란드어, 세르비아어로 이루어진 구술 의회 말뭉치(corpora) 모음으로, 자동으로 구축되었으며 다양한 주석이 추가되어 있다. 이 데이터셋은 감정 예측(sentiment predictions) 및 언어 주석으로 풍부해졌으며, 미리 준비된 텍스트 양식으로 다운로드할 수 있다. 각 언어의 음성 모드는 채우기 중단(fill pauses)으로 자동적으로 풍부해져 보다 정교한 연구를 가능하게 한다.

- **Technical Details**: ParlaSpeech 데이터셋은 CLARIN Resource Family에 의해 지원되는 데이터로, 언어 간 비교 연구를 가능하게 하는 고유한 특성을 지닌다. 문서화되지 않은 언어나 비표준 변종을 연구할 때 직면하는 여러 기술적, 물류적 과제를 해결하면서 만들어졌다. 이 연구는 음성 녹음과 기록의 일치 수준에서 기본적인 구성이 아닌 여러 언어적 및 부언어적 주석으로 강화된 데이터셋을 처리하여, 이하의 실험을 통하여 해당 연구의 유용성을 보여준다.

- **Performance Highlights**: ParlaSpeech는 각 언어의 주요 강조(stress) 위치와 같은 재구성된 발음 데이터를 포함하고 있어, 언어학 및 음향 분석에 있어 더욱 효과적인 연구 도구가 된다. 자동 주석 적용으로 감정 분석(acoustic correlates of sentiment)의 측정 정확도가 증가했으며, 두 개의 언어는 독립적인 단어 및 그래프 단위로 정렬되어 연구에 활용 가능하다. 또한 음성 신호와 감정 정보가 결합하여 연구자들이 언어의 맥락을 더 깊이 이해하는 데 기여할 수 있는 잠재력을 보여준다.



### Imperfect Language, Artificial Intelligence, and the Human Mind: An Interdisciplinary Approach to Linguistic Errors in Native Spanish Speakers (https://arxiv.org/abs/2511.01615)
Comments:
          12 pages, 3 figures

- **What's New**: 이 연구에서는 원주율 스페인어 사용자가 발생시킨 언어적 오류를 다루며, 인공지능 시스템이 이러한 오류를 재현하고 수정하는 방식에 대한 분석을 제공합니다. 특히 기존에 사용되던 접근 방식을 넘어선 학제간 연구를 제안합니다. 현재의 대형 언어 모델(LLM)이 언어적 오류를 어떻게 해석하는지에 대한 새로운 관점을 제공합니다.

- **Technical Details**: 연구는 이론적 언어학(theoretical linguistics)을 통해 오류의 본질을 분류하고 이해하며, 신경언어학(neurolinguistics)을 통해 뇌에서의 실시간 언어 처리(contextualize) 관점을 제공합니다. 또한 자연어 처리(NLP) 기술을 사용하여 생성된 오류를 평가하고, 500개 이상의 실제 오류로 구성된 특수 코퍼스를 구축하여 실증적 분석을 실행합니다.

- **Performance Highlights**: 이 연구는 AI 모델(GPT 또는 Gemini 등)과의 비교 분석을 통해 스페인어에 대한 이해를 높이고, 언어적 오류의 해석 정확도와 인간의 언어 행동 패턴을 일반화할 수 있는 능력을 평가합니다. 궁극적으로, 더 인지적으로 정보화된 NLP 시스템의 발전에 기여하게 될 것입니다.



### BIRD: Bronze Inscription Restoration and Dating (https://arxiv.org/abs/2511.01589)
Comments:
          Accepted at EMNLP 2025 (Main Conference)

- **What's New**: 이번 논문에서는 중국 초기 청동기 시대의 청동 비문을 위한 최초의 완전 인코딩 데이터셋 BIRD(Bronze Inscription Restoration and Dating)를 소개합니다. 이 데이터셋은 표준 학술 전사 및 연대 레이블을 바탕으로 구성되어 있으며, 장기적으로 비문 복원과 연대 부여에 기여할 수 있습니다. 또한, 그래프 표현과 유사한 다양한 그래픽 형태를 다루는 Glyph Net(GN)이라는 프레임워크를 제안하여, 효과적인 학습을 지원합니다.

- **Technical Details**: BIRD는 41,000개의 토큰으로 구성되어 있으며, 저자들은 환경에 적응한 사전 훈련(domain-adaptive pretraining)을 위해 고대 텍스트를 활용했습니다. 연구에서 제안된 Glyph Net는 그래픽 형태의 집합으로 이루어져 있으며, 이는 비문 데이터의 복원 및 연대 작업에서 중요한 도구로 작용합니다. 이러한 접근 방식은 데이터의 효율성을 극대화하고, 학습의 일반화를 지원합니다.

- **Performance Highlights**: 실험 결과, Glyph Net는 비문 복원 성능 향상에 기여했으며, 글리프를 기준으로 한 샘플링 전략은 연대 부여의 정확성에도 긍정적인 영향을 미쳤습니다. 전반적으로, 이 연구는 청동 비문의 복원 및 연대 부여에 있어 인공지능의 신규 활용 가능성을 제시하고 있으며, 향후 연구 방향 설정에 중요한 기초 자료로 기능할 것으로 예상됩니다.



### ECO Decoding: Entropy-Based Control for Controllability and Fluency in Controllable Dialogue Generation (https://arxiv.org/abs/2511.01568)
Comments:
          Published at EMNLP 2025 main

- **What's New**: 이 논문에서는 기존의 정적 제어 강도를 해결하기 위해 ECO 디코딩(Entropy-based COntrol)을 제안합니다. ECO 디코딩은 언어 모델과 속성 분류자의 엔트로피에 따라 각 생성 단계에서 제어 강도를 동적으로 조정합니다. 이 방법은 대화 생성의 유창성과 문법성(grammaticality)을 유지하며, 높은 제어력을 보이는 결과를 보여줍니다.

- **Technical Details**: 기존의 가중치 디코딩 방법은 고정된 제어 강도를 사용하는데, 이로 인해 제어력과 유창성 간에 트레이드오프가 발생할 수 있습니다. ECO 디코딩은 이러한 문제를 해결하기 위해 언어 모델의 확률 분포와 속성 분류자의 확률 분포를 바탕으로 각 생성 단계의 엔트로피를 계산합니다. 이를 통해 낮은 엔트로피를 가진 경우에는 모델의 예측을 우선시하고, 높은 엔트로피를 가진 경우 속성 확률의 상대적 가중치를 증가시킵니다.

- **Performance Highlights**: 다양한 실험에서 ECO 디코딩은 DailyDialog 및 MultiWOZ 데이터셋을 이용해 유창성을 유지하면서 높은 제어력을 달성하는 것을 검증하였습니다. ECO 디코딩은 기존의 가중치 디코딩 방법과 비교하여 여러 모델 및 설정에서 일관되게 우수한 성능을 보여주며, 단일 및 다중 속성 시나리오에서도 강력한 결과를 냅니다.



### Math anxiety and associative knowledge structure are entwined in psychology students but not in Large Language Models like GPT-3.5 and GPT-4o (https://arxiv.org/abs/2511.01558)
- **What's New**: 이 연구는 수학 불안(Math anxiety)이 대학 심리학 학생들에게 미치는 영향을 탐구하며, 행동적 forma mentis 네트워크(Behavioural forma mentis networks)라는 프레임워크를 사용하여 개별 및 집단 차이를 분석하였습니다. 네트워크 분석을 통해 학생들, 그리고 GPT-3.5 및 GPT-4o 모델의 시뮬레이션 결과를 비교하였고, 이러한 접근 방법이 수학 불안의 개념 및 연관성 이해에서 어떻게 기여하는지를 보여줍니다. 이 연구는 학생들이 느끼는 긍정적 및 부정적 정서가 수학 불안에 미치는 예측력을 강조하고 있습니다.

- **Technical Details**: 연구는 2개의 샘플(n1=70, n2=57)에 걸쳐 4개의 실험을 수행하였으며, 각 실험은 수학 불안 스케일(Math Anxiety Scale)을 통해 심리 측정 점수를 예측하기 위한 개별 수준의 네트워크 특성을 활용합니다. 제4실험에서는 인간 학생들과 GPT 모델들의 집단 수준 인식을 분석하였습니다. BFMN은 개념 간 자유연상 패턴을 통해 개념적 연관을 나타내며, 감정적 관점을 가지고 각 개념의 긍정적, 부정적 또는 중립적 가치를 반영합니다.

- **Performance Highlights**: 연구 결과에 따르면, 수학 불안이 높은 학생들은 '불안(anxiety)'에 대해 긍정적인 평가와 높은 네트워크 연결도를 보이며, '수학(math)'에 대해서는 부정적인 평가를 내리는 경향이 있습니다. 그러나 이러한 모델은 실험에서 인간 데이터와의 차이로 인해 GPT 기반 데이터에서는 적용되지 않았습니다. 이러한 발견은 수학 불안 관리에 있어 개념 인식과 연관성의 이해가 중요하다는 점을 강조하고 있습니다.



### Difficulty-Controllable Cloze Question Distractor Generation (https://arxiv.org/abs/2511.01526)
- **What's New**: 이 논문은 언어 숙련도와 이해도를 평가하기 위해 사용되는 다지선다형 빈칸 문제의 생성에 대한 새로운 프레임워크를 제안합니다. 기존 방법들이 유연성과 난이도 조절에 부족한 단점이 있었던 점에 착안하여, 데이터 증강(data augmentation)과 다중 작업 학습(multitask learning) 접근 방식을 통합했습니다. 이를 통해 다양하고 그럴듯한 선택지를 생성할 수 있는 두 가지 방법을 도입하고, 이를 통해 품질 높은 난이도 주석 데이터셋을 만듭니다.

- **Technical Details**: 제안된 프레임워크는 두 단계로 나뉘어 있습니다. 첫 번째 단계에서는 두 가지 방법을 활용하여 고품질의 후보 distractors를 생성합니다. 두 번째 단계에서는 생성된 후보를 필터링하고, 질문의 난이도에 따라 클러스터링하여 보다 정교한 평가를 가능하게 합니다. 다중 작업 학습을 통해 생성된 모델은 주어진 난이도 수준에 따라 distractors를 생성하고, 그들의 난이도를 평가할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 난이도 수준에서 고품질의 distractors를 생성하며, GPT-4o와 비교했을 때 인간의 인지와 일치하는 난이도 제어에서 크게 향상되었습니다. 전반적으로 제안된 모델은 하드 distractors에 대해 73.25%, 쉬운 distractors에 대해 64.23%의 높은 난이도 정확도를 기록했으며, 무효 distractor 비율을 크게 줄였습니다. 이러한 결과를 통해 기존 방법들이 갖고 있는 난이도 조절의 한계를 극복하는 성과를 거두었습니다.



### BanglaNirTox: A Large-scale Parallel Corpus for Explainable AI in Bengali Text Detoxification (https://arxiv.org/abs/2511.01512)
Comments:
          Under review, 6 pages, 1 figure, 2 tables

- **What's New**: 이 논문에서는 벵골어에서의 독성 언어 문제를 해결하기 위한 새로운 파이프라인을 제안합니다. 텍스트 디톡시피케이션(text detoxification) 분야가 자원이 풍부한 언어에서는 발전했지만, 벵골어는 자원이 제한되어 있어 미흡합니다. 특히, 이 연구는 Pareto class-optimized large language models (LLMs)와 Chain-of-Thought (CoT) prompting을 결합하여 독성 문장을 정화하는 방법을 제시합니다.

- **Technical Details**: 논문에서 제안하는 BanglaNirTox 데이터셋은 68,041개의 독성 벵골어 문장으로 구성되어 있으며, 각 문장은 독성 레이블, 이성(reasoning), 그리고 디톡시피케이션된 패러프레이즈(paraphrase)를 포함합니다. 이 데이터셋은 Pareto-optimized LLMs를 사용하여 생성되었으며, 무작위 샘플을 기준으로 평가되었습니다. 구성된 데이터셋을 바탕으로 언어 모델을 파인튜닝(fine-tune)하여 벵골어 문장의 독성 제거 성능을 높입니다.

- **Performance Highlights**: 연구 결과, Pareto-optimized LLMs에 CoT prompting을 적용함으로써 벵골어 텍스트 디톡시피케이션의 품질과 일관성이 크게 향상됨을 확인했습니다. 이러한 접근 방식은 기존의 독성 언어 처리 방법에 비해 보다 효과적인 결과를 가져오는 것으로 나타났습니다.



### Synthetic Eggs in Many Baskets: The Impact of Synthetic Data Diversity on LLM Fine-Tuning (https://arxiv.org/abs/2511.01490)
- **What's New**: 이 논문에서는 합성 데이터(synthetic data)의 다양성이 모델 행동에 미치는 영향을 조사했습니다. 특히, 다양한 출처에서 생성된 합성 데이터로 세밀히 조정된 대규모 언어 모델의 성능과 안전성에 대한 결과를 제시합니다. 연구 결과, 합성 데이터의 출처 다양성이 모델의 출력을 개선하는 데 긍정적인 영향을 미친다는 것을 발견했습니다.

- **Technical Details**: 연구에서는 합성 데이터의 출처 다양성이 언어 모델의 출력 분포와 언어적 다양성 지표에 미치는 영향을 분석하였습니다. 이를 통해 데이터 출처의 다양성이 증가하면 모델의 성능이 향상된다는 점을 확인하였습니다. 또한, 합성 데이터로 세밀히 조정했을 때 모델의 적대적 강건성(adversarial robustness)은 감소하는 경향이 있었지만, 사람의 데이터는 출력 품질을 저하시켰습니다.

- **Performance Highlights**: 합성 데이터의 출처 다양성이 높을수록 모델의 출력 분포의 붕괴(distribution collapse)를 줄이는 데 도움이 되는 것으로 나타났습니다. 연구의 핵심 발견 중 하나는 합성 데이터로 세밀히 조정할 경우, 자기 선호 편향(self-preference bias)이 감소하며, 특히 사람의 데이터를 사용할 경우 가장 효과적이라는 것입니다. 이는 LLM이 평가 작업에서 공정성을 높이는 데 기여할 수 있습니다.



### Towards Consistent Detection of Cognitive Distortions: LLM-Based Annotation and Dataset-Agnostic Evaluation (https://arxiv.org/abs/2511.01482)
- **What's New**: 이 논문에서는 텍스트 기반의 자동 인지 왜곡(मार्कार, Cognitive Distortion, CDs) 탐지에서의 주관성 문제를 해결하기 위해 대형 언어 모델(LLMs)를 활용한 새로운 주석 프레임워크를 제안합니다. 여러 개의 독립적인 LLM 실행을 통해 안정된 라벨링 패턴을 발견할 수 있으며, 이를 통해 더 일관된 주석을 생성할 수 있다고 주장합니다. 또한 다양한 특성을 가진 데이터 세트 간 공정한 비교를 위해 Cohen의 카파를 이용한 데이터 세트 비관계 평가 프레임워크를 소개합니다.

- **Technical Details**: 저자들은 주석의 신뢰성 검증을 위해 LLMs, 특히 GPT-4 모델을 사용하여 여러 번의 독립적인 주석 프로세스를 수행합니다. 이러한 방법을 통해 특히 정신 건강과 같이 객관적인 진실을 도출하기 힘든 분야에서 신뢰할 수 있는 주석을 생성하는 것이 가능합니다. 논문에서는 LLM이 생성한 라벨이 인간 주석보다 더 높은 일관성을 제공함을 입증하는 실험 결과를 보여줍니다.

- **Performance Highlights**: 실험 결과, LLM 생성의 일관된 주석(Fleiss's Kappa = 0.78)을 통해 훈련된 모델들이 인간에 의해 주석 처리된 데이터로 훈련된 모델들보다 더 나은 성능을 보였습니다. 이로 인해 주관적인 NLP 작업에서 LLMs가 훈련 데이터 생성에 있어 확장 가능하고 일관된 대안을 제공할 수 있다는 신뢰를 제시합니다. 제안된 평가 방법론은 다양한 데이터 세트를 기반으로 한 모델의 성능 비교를 가능하게 하여 기계 학습의 발전에 기여할 것으로 기대됩니다.



### BARD: budget-aware reasoning distillation (https://arxiv.org/abs/2511.01470)
- **What's New**: 이 논문에서는 Budget-Aware Reasoning Distillation (BARD) 프레임워크를 제안하여, 작은 언어 모델에 효과적으로 추론 능력을 전이하면서도 추론 길이에 대한 세밀한 제어를 가능하게 합니다. BARD는 사용자 지정으로 설정된 thinking budget을 사용하여 모델이 추론 성능과 컴퓨팅 효율성 사이의 균형을 동적으로 조절할 수 있도록 합니다. 이러한 접근법은 추론 과정의 중복성을 감소시키고 자원 사용을 효율적으로 관리할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: BARD는 두 단계의 훈련 레짐을 사용하여 구현됩니다. 첫 번째 단계인 Supervised Fine-Tuning (SFT)에서는 다양한 예산 수준으로 압축된 헷갈리게 하지 않은 긴 Chain-of-Thought 데이터를 사용하여 모델에게 예산 제약을 이해시키고, 두 번째 단계에서는 강화 학습 (Reinforcement Learning, RL)을 적용하여 추론 성능과 예산 정확성을 동시에 고려한 보상 신호를 활용합니다. 이 두 단계의 훈련은 정책 퇴화(policy degradation)를 방지하고 두 가지 목표가 동시에 최적화되도록 보장하는 데 중요합니다.

- **Performance Highlights**: AIME24, AIME25, GPQA와 같은 도전적인 추론 벤치마크에서 8B 모델이 강력한 성능을 달성하는 것을 보여주는 광범위한 실험 결과가 있습니다. BARD는 다양한 예산에 걸쳐서 추론 길이에 대한 정밀하고 적응 가능한 제어를 제공함으로써 사용자가 요구하는 다양한 요구 사항에 맞춘 추론 전략을 동적으로 조정할 수 있습니다.



### "Don't Teach Minerva": Guiding LLMs Through Complex Syntax for Faithful Latin Translation with RAG (https://arxiv.org/abs/2511.01454)
- **What's New**: 이 논문은 라틴어와 같은 형태론이 풍부한 저자원 언어의 번역에서 Open-source LLM을 활용한 새로운 접근 방식을 제안합니다. 제안된 방식은 NLLB-1.3B 모델을 통해 생성된 최초 초안을 Llama-3.3 또는 Qwen3과 같은 제로샷 LLM으로 다듬는 두 단계의 파이프라인을 사용합니다. 이 방법은 12세기 라틴 문서의 새로운 OOD 벤치마크와 함께 공개되어, 이전의 상업적 시스템과 통계적으로 유사한 성능을 달성했다고 주장합니다.

- **Technical Details**: 저자들은 두 단계의 증강된 파이프라인을 제안합니다. 첫 번째 단계에서, NLLB-200-1.3B 모델을 통해 구조적으로 충실한 초안이 생성되고, 두 번째 단계에서는 유사한 라틴어 이웃들을 검색하여 이 정보를 기반으로 LLM이 초안을 다듬습니다. 이를 통해 LLM은 보다 견고한 번역을 생성할 수 있으며, DAPT와 같은 접근 방식을 사용해 모델을 특정 도메인에 맞게 조정합니다.

- **Performance Highlights**: 제안된 오픈소스 RAG 시스템은 GPT-5의 기준선과 통계적으로 유사한 성능을 도달했으며, 특정 작업 전용 LLM의 세분화된 미세 조정 없이도 가능함을 입증합니다. 또한, 전통적인 정량적 지표를 포함한 품질 분석이 이루어져 필로로지적 적합성도 함께 평가되었습니다. 논문은 파이프라인, Chartres OOD 세트, 평가 스크립트 및 모델을 공개하여 재현 가능성 및 추가 연구를 촉진하고자 합니다.



### LiveSearchBench: An Automatically Constructed Benchmark for Retrieval and Reasoning over Dynamic Knowledg (https://arxiv.org/abs/2511.01409)
- **What's New**: LiveSearchBench는 질문 답변(QA) 평가를 위해 동적 지식을 반영하는 자동화된 벤치마크를 구축하는 혁신적인 접근 방식을 제시합니다. 이 방법은 최근의 Wikidata 스냅샷 간의 delta를 계산하고, 품질을 필터링하여 자연어 질문을 생성합니다. 이를 통해 고유하고 검증 가능한 답변을 보장하는데, 이는 모델의 메모리에 의존하지 않고 최신 지식을 요구하는 작업들로 평가를 전환하려는 목표를 가지고 있습니다.

- **Technical Details**: 이 논문은 동적 지식 그래프를 활용하여 질문-답변 문제를 구성하는 방법을 포괄적으로 설명합니다. Wikidata 지식 그래프를 기반으로 하여, 실시간 편집 흐름에서 질문을 지속적으로 생성하고, 사실성과 시간 일관성을 검증하는 과정을 포함합니다. 이러한 자동화된 데이터 생성 파이프라인은 지속 가능한 업데이트가 가능하여, 모델들이 동적 지식을 효과적으로 처리하는 능력을 평가합니다.

- **Performance Highlights**: LiveSearchBench에서의 실험 결과, 큰 성능 저하가 관찰되었으며, 특히 다단계 질문에서 최근 정보가 포함된 질문에 대한 능력이 저하되는 경향이 있음을 발견하였습니다. 기존의 정적 벤치마크는 LLM이 최신 동적 정보를 처리하는 능력을 충분히 평가하지 못하는 경향이 있습니다. 따라서, 본 연구는 LLM의 지속적인 평가를 위한 보다 현실적이고 시사적인 기준을 제공하는 것을 목표로 합니다.



### RAGSmith: A Framework for Finding the Optimal Composition of Retrieval-Augmented Generation Methods Across Datasets (https://arxiv.org/abs/2511.01386)
Comments:
          45 pages

- **What's New**: RAGSmith는 레트리벌-증강 생성(Retrieval-Augmented Generation, RAG)의 통합 최적화를 위한 모듈형 프레임워크로, 46,080개의 파이프라인 구성에서 발생하는 상호작용을 고려합니다. 이 연구는 RAG 시스템의 구성 요소 간의 복잡한 synergies에 초점을 맞추며, 각 모듈을 독립적으로 최적화하는 전통적인 접근 방식을 재편합니다. RAGSmith는 유전 알고리즘을 통해 완전한 파이프라인 최적화를 가능하게 하며, 다양한 도메인에서 일관된 성능 향상을 나타냅니다.

- **Technical Details**: RAGSmith는 9개의 기술 군을 포함하여 46,080개의 구성 조합을 평가합니다. 이 시스템은 레트리벌 메트릭(retrieval metrics)과 생성 메트릭(generation metrics) 모두를 통합하여 최적화를 진행하고, 다양한 질문 유형에 대한 반응성을 기반으로 전반적인 성능 향상을 목표로 합니다. 또한, RAGSmith는 질문 타입에 민감한 최적화 지침을 설정하여, 도메인별 데이터셋 특성에 맞춰 최적의 구성 요소 조합을 찾아냅니다.

- **Performance Highlights**: RAGSmith는 기본 RAG 베이스라인 대비 평균적으로 +3.8%의 성능 향상을 달성하며, 특정 도메인에서는 최대 +12.5%의 레트리벌 개선 및 +7.5%의 생성 개선을 나타냅니다. 실험 결과, 장기 답변 및 사실 기반 질문 세트에서 더 큰 개선이 이루어졌으며, 이는 RAG 시스템 구성에서 질문 유형의 중요성을 강조합니다. 이 결과는 evolucionary search의 장점을 입증하며, 효과적인 RAG 시스템 구성 방법에 대한 실용적 가이드를 제공합니다.



### Confounding Factors in Relating Model Performance to Morphology (https://arxiv.org/abs/2511.01380)
Comments:
          EMNLP 2025: Main Conference

- **What's New**: 이 논문에서는 언어 모델링에서 형태소(morphology)의 영향과 관련하여 혼란 요인을 식별하고 이를 해소하기 위한 새로운 접근법을 제시합니다. 형태소의 복잡성이 언어 모델의 성능에 미치는 효과를 평가하기 위해 새로운 토큰 빅람 메트릭스(token bigram metrics)를 도입하며, 이 메트릭스가 언어 모델링의 어려움을 예측하는 데 효과적임을 입증합니다. 연구자들은 이러한 연구가 대규모 언어 모델의 신뢰성을 향상시키는 데 기여할 것으로 기대하고 있습니다.

- **Technical Details**: 연구는 아그루티나티브 언어(agglutinative languages)와 융합 언어(fusional languages) 간의 성능 차이를 다룹니다. 이 논문에서는 아그루티나티브 언어의 토큰화를 구성하는 형태소의 정렬(morphological alignment) 및 토큰화 효율성(tokenization efficiency)과 같은 요인들이 결과에 미치는 영향을 분석합니다. 또한, 기존의 연구에서 제기된 세 가지 가설을 재검토하고 혼란 요인을 고려하여 방법론적 개선 방안을 제안합니다.

- **Performance Highlights**: 본 연구의 결과, 토큰 빅람 메트릭스가 형태소 복잡성을 나타내는 경량 프록시로 작용하며, 전문가의 주석 없이도 언어 모델링의 어려움을 예측할 수 있음을 보여줍니다. 이러한 접근법은 토큰화와 형태소 간의 관계를 보다 명확하게 이해하는 데 기여하며, 언어 모델의 성능 개선을 위한 새로운 방향을 제시합니다. 궁극적으로 이 연구는 언어 모델링과 형태소의 관계를 보다 신뢰할 수 있게끔 하는 실험적 조건을 설정하는 데 중점을 두고 있습니다.



### The Ouroboros of Benchmarking: Reasoning Evaluation in an Era of Saturation (https://arxiv.org/abs/2511.01365)
Comments:
          Accepted to NeurIPS 2025 Workshop on LLM Evaluation (this https URL)

- **What's New**: 최근 대규모 언어 모델(LLMs) 및 대규모 추론 모델(LRMs)의 급속한 발전에 따라 이들을 평가하기 위한 벤치마크의 수가 급증했습니다. 그러나 기존의 벤치마크들은 최신 모델의 성능이 상승함에 따라 포화 상태에 이르고 있으며, 이러한 상황이 과연 진정한 추론 능력을 증명하는지에 대한 의문이 제기되고 있습니다. 본 연구에서는 OpenAI, Anthropic 및 Google의 세 가지 모델 패밀리를 대상으로 그들의 추론 능력이 벤치마크의 변화에 따라 어떻게 발전해왔는지 분석합니다.

- **Technical Details**: 우리는 총 52개의 벤치마크를 선정하고, 이들이 평가하고자 하는 추론의 유형에 따라 분류하였습니다. 벤치마크는 공감각 및 논리적 추론, 수학적 추론, 다중 모달 추론, 프로그래밍 및 코드 작성, 독해 및 질문 응답, 일반 지식 추론, 그리고 LLM 전용 기능으로 나뉩니다. 분석 결과, 2023년 이후 다중 모달 및 수학적 추론 벤치마크의 채택이 크게 증가하였으나, 읽기 이해 및 공감각 추론에 대한 새로운 벤치마크는 채택되지 않았습니다.

- **Performance Highlights**: 모델의 성능을 분석한 결과, 27개 벤치마크는 최소 80%의 정확도를 달성한 반면, 25개 벤치마크는 이를 넘지 못했습니다. 특히, LLM 특정 기능과 프로그래밍 관련 벤치마크는 여전히 높은 성과 달성이 어려운 상황입니다. 연구 결과에 따르면, 대부분의 해결된 벤치마크는 주로 공감각, 논리적, 수학적 추론 및 독해와 관련된 것으로 나타났습니다.



### Safer in Translation? Presupposition Robustness in Indic Languages (https://arxiv.org/abs/2511.01360)
Comments:
          This is a submission to LREC 2026 (Language Resources and Evaluation Conference 2026). Corresponding author: aadipalnitkar96@gmail.com

- **What's New**: 이 논문에서는 다국어 대형 언어 모델(LLMs)의 평가지표로 'Cancer-Myth-Indic'라는 지표를 제안하고 있습니다. 그 목적은 암에 대한 잘못된 전제를 검토하고 이를 여러 언어로 평가하는 것입니다. 기존의 의학적 벤치마크는 거의 대부분 영어로 작성되어 있으며, 이로 인해 다국어 LLM 평가에서의 중요한 격차가 존재합니다. 이 연구는 다섯 개의 언어로 번역된 500개의 암 신화 항목을 사용하여 이 격차를 해결하고자 합니다.

- **Technical Details**: 연구는 Cancer-Myth의 500개 항목을 다섯 개의 저자원 언어로 번역하여 LLM을 평가합니다. 평가 항목들은 사용자 질문의 암에 대한 잘못된 전제를 포함하고 있으며, 평가 과정에서는 프레임을 유지하면서 번역의 암묵성을 보존합니다. 이 작업은 명시적 부정의 생성을 방지하고, 유저의 질문 프레임을 유지하는 것을 목표로 합니다. 이 모든 과정은 주어진 원칙 아래에서 평가되며, 번역 편향이나 평가자 변동성을 최소화하도록 설계되었습니다.

- **Performance Highlights**: 이번 연구에서는 GPT-3.5 Turbo, GPT-4 Turbo, GPT-4o와 같은 인기 있는 LLM을 평가했습니다. 다국어 세트에서 성능 평가를 통해 특정 모델에 대한 안전성 비대칭성을 드러냈습니다. 결과적으로 원래 영어 벤치마크의 경우, 전 세계 평균 30%의 오답률이 있었지만, 이 연구에서 다국어 환경 하에서도 유사한 결과를 도출하였습니다. 이는 저자원 언어에서의 의료 상처를 인지하고 교정하는 것의 중요성을 강조합니다.



### PrefixNLI: Detecting Factual Inconsistencies as Soon as They Aris (https://arxiv.org/abs/2511.01359)
Comments:
          9 pages + appendix. Code, datasets, and models are available at this https URL

- **What's New**: 이번 논문에서는 자연어 추론(NLI) 모델을 통해 LLM의 결과의 사실성(factuality)을 개선하는 새로운 방법을 제안합니다. 자동 회귀 생성(autoregressive generation)에서 각 텍스트 접두사(prefix)에 대해 entailment를 평가하는 PrefixNLI 작업을 소개하며, 이를 통해 MiniTruePrefixes라는 새로운 모델을 개발하였습니다. 이 모델은 이전 NLI 모델보다 5에서 14 F1 포인트 더 높은 성능을 보이며, 추상 요약(abstractive summarization) 시의 사실적 일관성을 크게 개선합니다.

- **Technical Details**: MiniTruePrefixes는 텍스트 접두사에 대해 사실적 불일치를 더 잘 감지하도록 훈련된 전문 NLI 모델입니다. 이 모델은 훈련 및 평가 데이터셋을 제공하여 PrefixNLI 작업을 위한 새로운 기준을 세우고, 이를 통해 사실성과 속도에서 효율성을 유지하면서도 우수한 성능을 발휘합니다. 기존의 NLI 모델의 한계를 극복하고, 각 텍스트 접두사에 직접적으로 depender scoring feedback를 제공하는 접근 방식이 중요한 기술적 기여로 설명됩니다.

- **Performance Highlights**: MiniTruePrefixes는 LLaMA-3.2-3B-Instruct 모델과의 비교에서 8B 모델과 비슷한 사실성(Faithfulness) 및 실행 속도를 유지하면서도 메모리 소비를 절반으로 줄인 성과를 보여줍니다. 이는 더 작은 모델을 사용하더라도 사실적인 내용 생성을 가능하게 하며, 다양한 모델 크기 및 데이터셋에서 일관된 사실성 향상을 증명합니다. 결과적으로, prefix 기반 NLI를 통해 텍스트 생성의 사실성을 향상시킬 수 있는 더 넓은 잠재력을 제시합니다.



### Thinking with DistilQwen: A Tale of Four Distilled Reasoning and Reward Model Series (https://arxiv.org/abs/2511.01354)
Comments:
          emnlp 2025 industry track

- **What's New**: 최근의 산업 요구에 부응하여 DistilQwen 모델 시리즈가 확장되었으며, 네 가지 모델 시리즈가 도입되었습니다. 이 모델들은 고도화된 산업 환경에서의 응용을 목표로 하며, 각각 느린 사고 모델, 적응형 사고 모델, 그리고 증강된 보상 모델로 구성됩니다. 느린 사고 모델은 높은 정확도가 필요한 작업에 최적화되어 있으며, 적응형 사고 모델은 입력 작업에 따라 reasoning 전략을 동적으로 조정하여 다양한 시나리오에서 효율성을 극대화합니다.

- **Technical Details**: DistilQwen 모델은 데이터 소스 수집기(Data Source Collector)를 기반으로 하여 CoT(train of thought) 데이터셋을 집계하고, 다양한 도메인에서 훈련을 위한 풍부한 소스를 제공합니다. 이 논문에서는 SFT(supervised fine-tuning) 훈련 기술을 섬세하게 조정하고, CoT 생성 과정에서 다양한 성능 요건을 충족하기 위한 인프라를 구축한 방법론을 설명합니다. 느린 사고 모델을 위해 DeepSeek-R1을 사용하여 CoT를 생성하고, CoT 난이도 평가 시스템을 통해 학습 목표를 설정하였습니다.

- **Performance Highlights**: 모델 성능 평가 결과, DistilQwen 모델은 높은 추론 효율성과 강력한 reasoning 성능을 보였습니다. 특히, 적응형 사고 모델과 증강된 보상 모델은 실질적으로 산업 AI 플랫폼에서의 적용 가능성을 보여주었습니다. Alibaba Cloud PAI 플랫폼에서의 통합을 통해 이 모델들은 기업의 AI 솔루션으로서의 실용성을 증명했습니다.



### DEEPAMBIGQA: Ambiguous Multi-hop Questions for Benchmarking LLM Answer Completeness (https://arxiv.org/abs/2511.01323)
Comments:
          25 pages

- **What's New**: 본 연구에서는 DeepAmbigQAGen이라는 자동 데이터 생성 파이프라인을 소개하고, 이를 통해 이름의 모호성과 다단계 추론을 요구하는 QA 작업을 생성합니다. 또한 DeepAmbigQA라고 하는 3,600개의 질문으로 구성된 데이터셋을 구축하여, 이 데이터셋은 복잡한 질문에 대한 정확한 정답 집합을 생성하기 위한 도전 과제를 제시합니다. 실험 결과, 최신 LLM인 GPT-5조차도 모호한 질문에 대한 답변의 일치율이 매우 낮아, 정보 수집 및 답변의 완전성을 위한 강력한 QA 시스템의 필요성을 강조합니다.

- **Technical Details**: DeepAmbigQAGen은 텍스트 코퍼스와 지식 그래프에 기반하여 모호한 질문을 생성하는 자동화된 파이프라인입니다. 이 파이프라인은 모호한 이름과 그에 해당하는 개체를 식별하고, 사용자의 정보 탐색 행동을 모델링한 실행 가능한 추론 계획을 구성합니다. 각 질문은 최소 두 단계의 추론을 요구하며, 일부는 최대 여덟 단계의 복잡한 추론을 포함하여, LLM QA 시스템에 상당한 도전 과제를 제시합니다.

- **Performance Highlights**: 최신 LLM을 평가한 결과, 특히 모호한 쿼리에 대한 정확한 답변 일치율이 0.13에 불과하여 낮은 성능을 보였습니다. 반면, 비모호한 질문에서도 0.21로 그 성능이 낮아, LLM의 전체적인 답변 수집에서 실패하는 모습을 보였습니다. 연구 결과, 질의 확장 및 증거 추출과 같은 모듈을 추가하더라도 이와 같은 복잡한 질문에 대한 문제를 완전히 해결하지 못함이 밝혀졌습니다.



### DeepSpecs: Expert-Level Questions Answering in 5G (https://arxiv.org/abs/2511.01305)
- **What's New**: DeepSpecs는 5G 사양에 대한 전문가 수준의 질문답변을 가능하게 하는 새로운 RAG 시스템입니다. 이 시스템은 3개의 메타데이터 중심 데이터베이스인 SpecDB, ChangeDB, TDocDB를 통해 구조적 및 시간적 추론을 강화합니다. DeepSpecs는 문서 내 서로 다른 조항의 교차 참조를 명시적으로 해결하며, 사양 발전 과정을 추적하여 변경 사항을 기록된 변경 요청과 연결합니다.

- **Technical Details**: DeepSpecs는 5G 사양의 복잡한 요구를 충족하기 위해 구조적 및 시간적 이해를 통합합니다. 클로즈(Clause) 수준의 교차 참조 해결과 사양 발전 추론을 제공하는 두 가지 기능을 통해 전문가가 표준을 탐색하는 방식을 모방합니다. 이 시스템은 변경 내역, 조항 간 링크 및 3GPP 특정 메타데이터를 기반으로 하여 기능의 도입 및 진화를 추적합니다.

- **Performance Highlights**: DeepSpecs는 강력한 LLM 기준 모델 및 통신 전용 RAG 시스템들과 비교하여 일관된 성과 향상을 보여주었습니다. 우리는 573개의 질문-답변 쌍으로 구성된 실제 데이터 세트를 통해 시스템의 효율성을 평가하며, 교차 참조 해결과 사양 발전 추론의 이점을 입증합니다. 또한, 전문적으로 주석이 달린 최초의 5G QA 데이터 세트를 소개하여 5G 사양에 대한 이해를 돕습니다.



### FirstAidQA: A Synthetic Dataset for First Aid and Emergency Response in Low-Connectivity Settings (https://arxiv.org/abs/2511.01289)
Comments:
          Accepted at the 5th Muslims in Machine Learning (MusIML) Workshop, co-located with NeurIPS 2025

- **What's New**: 이 논문에서는 응급 상황에서 도움이 되는 FirstAidQA라는 합성 데이터 세트를 소개하고 있습니다. 5,500개의 고품질 질문-답변 쌍으로 구성된 이 데이터 세트는 응급 처치 및 응급 대응 시나리오를 다루고 있습니다. 데이터 세트는 LLM(대형 언어 모델)인 ChatGPT-4o-mini를 사용하여 생성되었고, 재난 지역이나 인터넷이 제한된 환경에서도 사용할 수 있도록 설계되었습니다.

- **Technical Details**: FirstAidQA 데이터 세트는 응급 처치에 필요한 고유한 정보 및 상황 지식을 포함하는 QA 쌍으로 구성됩니다. 이를 생성하기 위해 Vital First Aid Book(2019)에서 텍스트를 세분화하고, 프롬프트 기반의 비상 상황에서 LLM을 통해 현실적이고 상황에 맞는 질문과 답변을 생성하였습니다. 데이터 세트는 LLM 및 SLM(소형 언어 모델)의 미세 조정을 위해 설계되었으며, 실제 의료 지원을 위한 실시간 오프라인 배포를 지원합니다.

- **Performance Highlights**: FirstAidQA 데이터 세트는 응급 상황에서 신속하고 신뢰할 수 있는 시스템 구현을 도와주어 응급 대응 및 안전 관련 AI 응용 프로그램의 연구가 발전할 수 있도록 돕습니다. 데이터 세트는 Hugging Face에 공개되어 연구자들이 쉽게 접근하여 사용할 수 있도록 되어 있으며, 인간 검증을 통해 안전성과 정확성을 보장한 후 발표되었습니다. 이 데이터 세트는 응급 처치에 대한 지식 전달을 향상시키고, LLM의 활용 가능성을 확장하는 데 중요한 기여를 할 것입니다.



### "Give a Positive Review Only": An Early Investigation Into In-Paper Prompt Injection Attacks and Defenses for AI Reviewers (https://arxiv.org/abs/2511.01287)
- **What's New**: 이번 논문은 AI 모델의 급속한 발전에 따라 과학 논문 리뷰 과정에서 AI 모델을 활용하는 것이 주목받고 있다는 점을 강조합니다. 그러나 일부 연구에서 악의적인 프롬프트가 숨겨져 있어 AI 리뷰어를 부적합하게 평가하도록 조작할 수 있다는 우려가 제기되고 있습니다. 저자들은 'In-Paper Prompt Injection (IPI)'라는 새로운 위협을 규명하고, 이에 대한 두 가지 공격 방식—정적 공격(static attack)과 반복 공격(iterative attack)—을 제안합니다.

- **Technical Details**: 정적 공격은 미리 정의된 악의적인 프롬프트를 PDF 문서에 삽입하는 반면, 반복 공격은 다수의 최적화를 통해 목표 AI 리뷰어에 대한 강력한 프롬프트를 생성합니다. 이 연구에서 사용된 AI 모델은 최신의 GPT-5, DeepSeek-Chat 및 Gemini-2.5-Pro이며, 100개의 ICLR 2025 제출문을 평가했습니다. 연구 결과, 현재의 AI 리뷰 시스템이 IPI 공격에 취약하다는 것이 드러났습니다.

- **Performance Highlights**: 정적 공격은 평균적으로 Gemini에서 1.91, DeepSeek에서 2.80, GPT-5에서 1.24 점수를 증가시키며, 반복 공격은 최대 점수(10)에 가까운 결과를 얻었습니다. 이 공격 방법들은 다양한 설정에서 강력하게 작용하는 것을 보여 주었고, 방어 메커니즘을 도입하여 공격 성공률을 감소시킬 수 있었지만, 적응형 공격이 방어를 일부 회피할 수 있음을 시사합니다. 이러한 결과는 AI 기반 리뷰 파이프라인의 구조적 취약성을 드러냅니다.



### When, What, and How: Rethinking Retrieval-Enhanced Speculative Decoding (https://arxiv.org/abs/2511.01282)
- **What's New**: ReSpec은 전통적인 heuristic 기반의 drafter 전환을 적응형 의사결정으로 바꾸는 새로운 프레임워크로, Speculative Decoding (SD)의 효율성을 크게 향상시킵니다. 주요 혁신으로는 낮은 불확실성에서만 retrieval을 트리거하는 엔트로피 기반의 적응형 트리거와 역사적 피드백을 활용한 후보 선택 방식이 포함됩니다. 또한, 출처 인식 relaxed verification 전략을 통해 정확성과 효율성 사이의 균형을 이룹니다. 이러한 개발을 통해 ReSpec은 기존 방법보다 33% 이상 빠른 성능 및 품질을 유지합니다.

- **Technical Details**: ReSpec은 정보 엔트로피를 사용하여 예측 가능성을 정량화하며, 낮은 엔트로피를 가진 상황에서만 retrieval을 트리거합니다. 토큰의 지식을 바탕으로 작성된 정보는 후보 선택에서의 과도한 검증 비용을 줄이고, 유망한 후보들을 조직화합니다. 최적의 검증 효율성을 위해, model-based drafts는 엄격한 검증을 받는 반면, retrieval-based drafts는 보다 느슨한 검증을 통해 품질을 보장합니다. 이 방식은 retrieval의 유용성을 최대화하는 동시에 시스템 자원의 효율적인 사용을 촉진합니다.

- **Performance Highlights**: ReSpec은 기초 실험인 Spec-Bench에서 EAGLE-2와 SAM-Decoding을 각각 33% 및 25% 이상 초과하는 성능으로 검증되었습니다. 특이하게도, 출력 품질을 저하시키지 않고도 효율적인 속도 향상을 이루었습니다. 또한, 적응형 및 상황 인식 제어가 혼합 추론 시스템에서의 효과를 극대화한 점에서 주목받고 있습니다. 이로써, ReSpec은 State-of-the-art의 가속화 결과를 달성하였으며, 이를 통해 향후 Speculative Decoding 분야의 발전에 기여할 것으로 기대됩니다.



### AraFinNews: Arabic Financial Summarisation with Domain-Adapted LLMs (https://arxiv.org/abs/2511.01265)
Comments:
          10 pages

- **What's New**: 이 논문에서는 아랍어 금융 텍스트의 추상적 요약(abstractive summarization)에 대한 도메인 특화의 영향을 대규모 언어 모델(LLM)을 활용하여 조사합니다. 여기서 21만 2,500개의 기사-헤드라인 쌍으로 구성된 AraFinNews 데이터셋을 소개하며, 이는 아랍어 금융 뉴스 데이터셋 중에서 가장 크고 광범위합니다. 이 데이터셋은 CNN/DailyMail과 같은 주요 영어 요약 말뭉치의 아랍어 동등물로 설계되어 금융 맥락에서의 언어 이해 및 생성 평가를 위한 강력한 기준점을 제공합니다.

- **Technical Details**: AraFinNews 데이터셋을 통해 Transformer 기반 모델(mT5, AraT5, FinAraT5 등)을 평가하며, 금융 도메인에서의 사전 학습이 사실 정확성(factual accuracy), 수치 신뢰성(numerical reliability), 전문적인 보고 스타일과의 정합성(stylistic alignment)에 미치는 영향을 검토합니다. 실험 결과에 따르면 도메인 맞춤형 모델들이 정량적 및 개체 중심 정보 처리에서 더욱 신뢰할 수 있고 일관된 요약을 생성하는 것으로 나타났습니다. 이 연구는 Arabic 금융 요약의 사실 일관성과 서사 유창성을 개선하기 위한 도메인 특정 적합의 중요성을 강조하고 있습니다.

- **Performance Highlights**: AraFinNews 데이터셋을 통해 이루어진 평가에서, 도메인 특화된 대규모 언어 모델이 아랍어 금융 요약에서 유창성, 사실적 일관성, 맥락적 관련성을 향상시키는 것으로 나타났습니다. 특히 금융 관련 정보의 수치적이고 질적인 처리에서 성과가 두드러지며, 이는 아랍어 NLP의 도메인 특정 발전에 기여하는 바가 큽니다. 이 데이터셋은 비상업적 연구를 위해 무료로 제공되며, 이는 아랍어 금융 텍스트 자동 요약 분야의 발전에 크게 기여할 것으로 기대됩니다.



### DEER: Disentangled Mixture of Experts with Instance-Adaptive Routing for Generalizable Machine-Generated Text Detection (https://arxiv.org/abs/2511.01192)
Comments:
          Under Review

- **What's New**: 이 연구에서는 머신 생성 텍스트(MGT) 탐지의 새로운 프레임워크인 Disentangled mixturE-of-ExpeRts (DEER)를 제안합니다. 이 구조는 두 단계로 구성되어 있으며, 첫 번째 단계에서는 도메인 특화 전문가가 인간과 기계 생성 텍스트의 세부적인 구별을 학습하고, 공유된 전문가는 도메인 간의 교차 기능을 추출합니다. 두 번째 단계는 강화 학습 기반의 라우팅 메커니즘을 도입하여, 각 입력 사례에 적합한 전문가를 동적으로 선택합니다.

- **Technical Details**: DEER 프레임워크는 두 가지 주요 구성 요소로 나뉘어 있습니다: 1단계에서는 Mixture-of-Experts (MoE) 구조를 사용하여 도메인 특정 및 도메인 일반적 특성을 분리합니다. 이 과정에서 도메인 레이블을 활용하여 전문가의 특성을 효과적으로 결정하고, 2단계에서는 강화 학습을 통해 테스트 시간에 각 입력에 대해 가장 적합한 전문가를 동적으로 선택합니다.

- **Performance Highlights**: 연구 결과, DEER는 5개의 도메인 내(in-domain) 및 5개의 도메인 외(out-of-domain) 벤치마크 데이터셋에서 기존 방법들보다 일관되게 높은 성능을 보였습니다. 평균 F1 점수는 도메인 내에서 1.39%, 도메인 외에서 5.32% 향상되었으며, 정확도 또한 각각 1.35%와 3.61% 개선되었습니다. 이 연구는 분리된 전문가 설계 및 적응형 라우팅 메커니즘이 모델 성능 향상에 중요한 기여를 한다는 것을 보여줍니다.



### Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning (https://arxiv.org/abs/2511.01191)
- **What's New**: 본 논문에서 제안된 Self-Harmony는 테스트 시간 강화 학습(Test-time Reinforcement Learning, TTRL)의 새로운 접근 방식을 소개합니다. 이 프레임워크는 동일한 문제를 다양한 방식으로 바꿔 제시했을 때 올바른 답변이 안정적으로 유지되어야 한다는 직관에 기반하고 있습니다. Self-Harmony는 단일 모델이 문제를 해결하는 Solver 역할과 문제를 재표현하는 Reframer 역할을 동시에 수행하도록 하여 신뢰할 수 있는 학습 신호를 생성합니다.

- **Technical Details**: Self-Harmony는 전통적인 다수결 방식의 한계를 극복하고자 하며, 원본 문제와 재표현된 문제에서의 답변 빈도를 집계하고 조화 평균(harmonic mean)을 통해 최종 의사 결정을 내립니다. 이를 통해 무의미한 주장이나 잘못된 reasoning을 피하고, 안정적인 해답을 추출할 수 있습니다. 모델은 이 두 가지 역할을 통해 협력하여 자기 놀이(self-play)를 수행하며, 이를 통해 더 나은 학습이 가능해집니다.

- **Performance Highlights**: Self-Harmony는 다양한 reasoning 벤치마크에서 탁월한 성과를 달성했습니다. 30개의 테스트 설정 중 28건에서 1위를 기록하며, label-free 테스트 환경에서 최첨단 결과를 입증하였습니다. 성능 외에도 모든 실험에서 제로 훈련 실패율을 기록하여, 이 방법의 안정성과 신뢰성을 강조합니다.



### ZoFia: Zero-Shot Fake News Detection with Entity-Guided Retrieval and Multi-LLM Interaction (https://arxiv.org/abs/2511.01188)
- **What's New**: 이 논문에서는 ZoFia라는 새로운 두 단계의 제로샷(Zero-Shot) 허위 뉴스 탐지 프레임워크를 제안합니다. 첫 번째 단계에서는 Hierarchical Salience를 도입하여 뉴스 콘텐츠에서 실체의 중요성을 정량화하고, SC-MMR 알고리즘을 사용해 최신 외부 증거를 탐색하기 위한 정보가 풍부하고 다양한 키워드를 효과적으로 선택합니다. 두 번째 단계에서는 다중 LLM 상호작용 시스템이 여러 관점에서 뉴스 텍스트 및 관련 정보를 협력적으로 분석하고, 이를 통해 해석 가능하고 견고한 판단을 생성합니다.

- **Technical Details**: ZoFia는 두 단계로 구성된 구조를 가지며, 첫 단계에서 엔티티 가이드 검색(Entity Guided Retrieval)을 통해 LLM의 내부 지식을 실시간 외부 정보와 통합하여 다원 정보 행렬(Multi-Source Information Matrix)을 구축합니다. 이를 통해 실시간으로 변하는 뉴스 스트림에 대한 지식 한계를 보완하고, 다양한 LLM이 협력하여 다차원 분석을 수행합니다. 이러한 방식으로, 각 LLM이 서로 다른 역할을 맡아 뉴스에 대해 논쟁을 진행함으로써 더 나은 결과를 도출합니다.

- **Performance Highlights**: 두 개의 공개 데이터 세트에서 수행된 실험 결과, ZoFia는 기존의 제로샷 모델과 대부분의 소수 샷 방법들을 명백히 능가합니다. 이러한 성과는 LLM의 제약을 극복하고, 다양한 정보 출처를 효과적으로 활용하는 다중 에이전트 시스템 덕분입니다. 연구팀은 코드도 오픈 소스로 공개하여 관련 커뮤니티의 재현성을 보장하고 있습니다.



### Surfacing Subtle Stereotypes: A Multilingual, Debate-Oriented Evaluation of Modern LLMs (https://arxiv.org/abs/2511.01187)
- **What's New**: DebateBias-8K는 다국어 및 토론 형식의 새로운 벤치마크로, LLM의 내러티브 편향을 현실적인 생성 환경에서 드러내고자 개발되었습니다. 이 데이터셋은 여성의 권리, 사회경제적 발전, 테러리즘, 종교 등 네 가지 민감한 도메인에서 8,400개의 구조화된 토론 프롬프트를 포함하고 있으며, 고급 자원 언어(영어, 중국어)와 저급 자원 언어(스와힐리어, 나이지리아 피진어)로 이루어진 7개 언어로 구성됩니다. 이 연구는 다국적 편향 평가를 위한 새로운 지평을 제시합니다.

- **Technical Details**: DebateBias-8K는 다양한 언어, 인구 통계 및 편향 유형을 탐색하여 LLM이 어떻게 내러티브 편향을 표현하는지 분석합니다. 이 연구는 8,400개의 토론 스타일 프롬프트를 통해 4개의 안전 정렬 모델(GPT-4o, Claude 3, DeepSeek, LLaMA 3)에서 10만 개 이상의 응답을 자동 생성하고 분류하였습니다. 언어 자원이 낮은 언어에서 편향이 심화되는 경향을 발견했으며, 일반화의 한계를 제시합니다.

- **Performance Highlights**: 모든 모델은 안전 정렬에도 불구하고 내재된 고정관념을 재현하며, 특정 인구 집단에 대한 부정적인 연관성을 보여줍니다. 아랍인들은 테러와 종교에 연결되고, 아프리카인은 사회경제적 '후진성'과 관련되어 있으며, 서구 그룹은 현대적 또는 진보적이라고 지속적으로 묘사됩니다. 이 연구는 다국어 형성의 편향을 포착하고, 앞으로 더 안전하고 문화적으로 포괄적인 모델 정렬을 위한 방향성을 제시합니다.



### Learning When to Quit in Sales Conversations (https://arxiv.org/abs/2511.01181)
- **What's New**: 이 연구는 판매의 효율성을 높이기 위해, 판매 대화에서 언제 중단할지를 결정하는 동적 스크리닝(decision-making) 문제를 최적 중단 문제(optimal stopping problem)로 공식화했습니다. 이를 위해, 회화의 전사를 기반으로 판매원을 지원하는 세분화된 언어 모델을 개발하여 실패가 예측되는 대화에서의 중단 결정을 향상시킨다는 접근 방식을 제안합니다. 이를 통해 고차원의 텍스트 상태를 처리하고, 공개 소스 및 독점 언어 모델 모두와 호환되도록 하여, 판매원들의 중단 효율성을 증가시키고 있습니다.

- **Technical Details**: 연구자들은 사전 훈련된 생성형 AI 모델을 사용하여 대화가 진행되는 동안 실시간으로 판매원이 언제 대화를 중단할지 결정하는 '중단 에이전트'를 개발했습니다. 이 에이전트는 기존 통화의 전사 텍스트를 기반으로 다음 결정을 생성하는 방식으로 작동하며, 강화 학습(reinforcement learning) 방식과는 다르게 안정적이고, 하이퍼파라미터에 견고하며, 수십억 개의 파라미터를 가진 대형 언어 모델과의 호환성을 강조합니다.

- **Performance Highlights**: 연구 결과, 제안된 중단 에이전트는 유럽의 한 통신 회사로부터 수집된 11,627개의 아웃바운드 판매 통화에서 실패 통화를 54% 덜 처리하며 거의 모든 판매를 유지했습니다. 이로 인해 시간 절약과 함께 기대 판매가 최대 37% 증가했습니다. 최종적으로, 이 연구는 판매원이 시간이 부족할 때 효과적으로 대화를 중단하는 방법을 제공하며, AI 알고리즘이 판매 효율성을 개선할 잠재성을 보여 줍니다.



### MicroRemed: Benchmarking LLMs in Microservices Remediation (https://arxiv.org/abs/2511.01166)
Comments:
          24 pages, 13 figures, 5 tables

- **What's New**: 이 논문에서는 MicroRemed라는 최초의 벤치마크를 소개하여 대규모 언어 모델(LLM)이 엔드 투 엔드 마이크로서비스 복원력을 평가할 수 있도록 합니다. MicroRemed는 진단 보고서에서 실행 가능한 Ansible 플레이북을 생성해야 하며, 전통적인 방식과의 주요 차별점은 LLM이 인간의 개입 없이 자체적으로 문제를 해결해야 한다는 것입니다. 또한, ThinkRemed라는 다중 에이전트 프레임워크를 제안하여, LLM이 반복적 사고 과정을 통해 복원 성능을 개선할 수 있는 방법을 제공합니다.

- **Technical Details**: MicroRemed는 실제 마이크로서비스 시스템을 자동으로 배포하고 다양한 실패를 지속적으로 주입하는 피드백 기반의 복원 파이프라인을 구축합니다. 이 벤치마크는 LLM이 진단 통찰력을 구체적인 수리 행동으로 변환하여 자동으로 실행할 수 있게 설계되었습니다. ThinkRemed는 LLM의 유연한 정보 수집을 지원하고, 플레이북 실행 후 피드백에 근거하여 계획을 수정할 수 있는 제한된 시험 및 반영 사이클을 허용합니다.

- **Performance Highlights**: 실험 결과, 현재 LLM이 MicroRemed에서 만족스러운 복원 성능을 달성하지 못하고 있는 것으로 나타났습니다. 또한, ThinkRemed의 성능 향상 능력을 통해 엔드 투 엔드 마이크로서비스 복원 과정에서 보다 효과적인 결과를 도출할 수 있음을 확인했습니다. 이러한 결과는 LLM이 복잡한 자동화 환경에서 아직 많은 도전과제가 있음을 강조합니다.



### TSVer: A Benchmark for Fact Verification Against Time-Series Evidenc (https://arxiv.org/abs/2511.01101)
Comments:
          Accepted to EMNLP 2025

- **What's New**: 이번 논문에서는 시간 시계열 데이터(temporal data)와 숫자 데이터(numerical data)를 활용한 사실 검증(fact-checking) 시스템을 평가하기 위한 새로운 벤치마크 데이터셋인 TSVer를 소개합니다. TSVer는 38개의 사실 검증 기관으로부터 수집된 287개의 실제 주장(claims)과 400개의 다양한 도메인을 포함하는 시간 시계열(time series) 데이터베이스를 포함하고 있으며, 각 주장은 관련된 시간 시계를 주어진 시간 프레임을 이용하여 명확하게 해석하고 평가할 수 있도록 주석이 달려 있습니다.

- **Technical Details**: TSVer는 시간 시계열 증거를 기반으로 하여 운영되며, 대규모 다단계 주석(annotation) 프로세스를 통해 주석의 질을 향상시켰습니다. 제안된 데이터셋은 약 20,000건의 레코드를 포함하며, 각 주장마다 관련 시간 프레임을 식별하고, 자료에 대해 심도 있는 추론을 요구합니다. 또 한가지 혁신적인 측정 기준인 TSCS를 도입하여 시간 시계열 선택의 정확도 및 시간 범위의 적합성을 공동으로 평가할 수 있도록 했습니다.

- **Performance Highlights**: 상태 최상의(reasoning models) 모델인 Gemini-2.5-Pro는 63.37의 정확도 점수와 48.63의 Ev2R 점수를 달성했습니다. 이러한 결과는 시간 시계열 데이터에 대한 기존의 사실 검증 시스템이 여전히 어려움을 겪고 있음을 보여줍니다. TSVer의 도입을 통해 시간 기반의 사실 검증 시스템이 정확성과 해석력을 개선할 수 있는 기회를 제공합니다.



### Improving Romanian LLM Pretraining Data using Diversity and Quality Filtering (https://arxiv.org/abs/2511.01090)
- **What's New**: 이 연구에서는 루마니아어(pretraining corpora)의 특징과 범위에 대해 분석하고, 이를 통해 루마니아어 전용 고품질 데이터 세트를 생성하는 방법을 제시합니다. 특히, 한정된 자원으로 인해 기반이 부족한 언어에서 LLM(대규모 언어 모델)의 성능을 개선하기 위한 필터링 방법을 도입하였습니다. 이 연구는 루마니아어 데이터의 교육적 가치 및 주제에 대한 다차원적인 분석을 통해 새로운 데이터 생성 접근법을 제공합니다.

- **Technical Details**: 본 연구는 FineWeb-Edu 프레임워크를 기반으로 하여 루마니아어 데이터 세트를 필터링 합니다. 이 필터링은 교육적 가치, 주제, 형식과 같은 여러 신호를 고려하여 다차원 리소스를 구축하고, 작은 인적 주석 데이터 세트(100 샘플)와 큰 LLM-주석 데이터 세트(1M 샘플)를 생성합니다. 이 데이터를 바탕으로 경량 다중 헤드 분류기를 훈련하여 비용 효율적으로 필터링하고, 언어 간 분포 분석을 수행했습니다.

- **Performance Highlights**: 실험 결과, 필터링된 루마니아어 데이터 세트로 훈련된 모델이 다양한 벤치마크에서 우수한 성능을 보였습니다. 콘텐츠의 교육적 가치와 주제를 기반으로 한 필터링 접근법이 루마니아어 LLM 성능을 향상시킨다는 것을 입증하였습니다. 또한, 이 연구에서 제시하는 FineWeb2-Edu-Ro 데이터 세트는 루마니아어 모델 훈련 및 평가에 필수적인 고품질 리소스로 자리잡을 것으로 기대됩니다.



### HPLT~3.0: Very Large-Scale Multilingual Resources for LLM and MT. Mono- and Bi-lingual Data, Multilingual Evaluation, and Pre-Trained Models (https://arxiv.org/abs/2511.01066)
- **What's New**: HPLT 3.0 프로젝트는 200개 언어를 위한 고품질 텍스트 데이터셋을 30조 개의 토큰으로 제공하는 중요한 이니셔티브입니다. 이는 다양한 웹 크롤링 데이터에서 파생된 자료로, 문서 선택, 텍스트 추출, 언어 식별, 중복 제거 등의 과정이 포함된 오픈소스 파이프라인을 통해 정제되었습니다. 이 데이터셋은 공개 라이센스를 통해 이용할 수 있으며, 다양한 언어 모델들이 사전 훈련되어 있습니다.

- **Technical Details**: HPLT 3.0은 3.3 페타바이트의 웹 아카이브 데이터를 활용하며, Common Crawl에서 수집된 데이터를 추가하여 총 7.2 페타바이트에 이릅니다. 데이터 전처리에서는 OpenLID-v2를 통해 언어 인식을 개선하고, MinHash 기반의 전역 근접 중복 제거를 구현했습니다. 또한, Web Docs Scorer(WDS) 접근법을 적용하여 문서 필터링 과정을 강화했습니다.

- **Performance Highlights**: HPLT 3.0 데이터셋의 성능은 다양한 언어 모델 아키텍처에 대해 각각 평가되었으며, 특히 24개 언어에 대한 샘플을 수동으로 검사하여 데이터 품질을 보고합니다. 보유한 데이터의 양과 다양성 덕분에 HPLT 3.0은 다른 공개 다국어 데이터셋들과 비교할 때 압도적으로 큰 규모와 풍부한 멀티링궐 데이터를 제공합니다.



### Building a Silver-Standard Dataset from NICE Guidelines for Clinical LLMs (https://arxiv.org/abs/2511.01053)
Comments:
          Submitted to EFMI Medical Informatics Europe 2026

- **What's New**: 이번 연구는 표준화된 검증 지표가 없는 의료 분야에서의 대형 언어 모델(LLMs)의 활용을 다룹니다. 공개된 진료 지침에서 파생된 검증된 데이터셋을 소개하며, 여러 진단에 대한 환자 시나리오와 임상 질문이 포함되어 있습니다. 이 데이터셋은 GPT의 도움으로 생성되었습니다.

- **Technical Details**: 연구팀은 최근 인기 있는 여러 LLM을 벤치마킹(benchmarking)하여 데이터셋의 유효성을 입증합니다. 데이터셋은 진료 지침(guideline)에 대한 준수와 함께 LLM의 임상 유용성(clinical utility)을 체계적으로 평가할 수 있는 프레임워크를 제공합니다. 이 데이터셋은 다양한 진단에 대한 현실적인 환자 시나리오를 포함하고 있습니다.

- **Performance Highlights**: 연구에서 제시된 방법론은 LLM의 진료 지침 준수 능력을 평가하는 데 중요한 기반을 제공합니다. 여러 LLM을 비교 분석함으로써, 연구팀은 LLM들이 어떻게 현실적인 임상 문제를 처리하는지에 대한 인사이트를 제공합니다. 이는 의료 처리 과정에서 LLM의 활용 가능성을 높이는 데 기여할 것입니다.



### VayuChat: An LLM-Powered Conversational Interface for Air Quality Data Analytics (https://arxiv.org/abs/2511.01046)
Comments:
          4 Pages, 4 Figures

- **What's New**: VayuChat는 인도 대기질 문제 해결을 위한 혁신적인 대화형 시스템입니다. 이 시스템은 자연어 질문에 대한 응답 뿐만 아니라 실행 가능한 Python 코드와 시각화 결과를 제공합니다. VayuChat은 대규모 언어 모델에 의해 구동되며, 중앙 오염 통제 위원회(CPCB)의 데이터와 국가 청정 공기 프로그램(NCAP)의 재정 기록을 통합하여 사용자에게 실시간 분석을 가능하게 합니다.

- **Technical Details**: VayuChat은 사용자가 간편하게 대기질 관련 질문을 하고, 그에 대한 시각적 및 코드 기반의 응답을 받을 수 있는 웹 기반 인터페이스를 제공합니다. 시스템 백엔드는 다양한 데이터 세트를 처리하여 사용자 쿼리에 대해 적절한 모델을 호출하고 생성된 코드를 실행하는 역할을 합니다. 사용자가 제출한 쿼리는 시스템 프롬프트와 결합되어 Python 코드 조각을 생성하며, 전체 과정은 안전한 샌드박스 환경에서 이루어집니다.

- **Performance Highlights**: VayuChat은 2024년 델리의 심각한 대기오염 상승을 분석하는 데 성공적으로 사용되었습니다. 초기 분석 단계에서 가장 오염이 심했던 날을 식별하고, 시간적 패턴을 분석해 오염 수준과 바람 속도를 비교하는 등의 시스템적 접근을 통해 중요한 통찰을 도출했습니다. 이러한 실시간 데이터 분석을 통해 정책 결정자와 시민들이 더 나은 대기질 관리를 할 수 있도록 지원합니다.



### OceanAI: A Conversational Platform for Accurate, Transparent, Near-Real-Time Oceanographic Insights (https://arxiv.org/abs/2511.01019)
Comments:
          A related presentation will be given at the AGU(American Geophysical Union) and AMS(American Meteorological Society) Annual Meetings

- **What's New**: OceanAI는 오픈소스 대형 언어모델(LLM)과 NOAA의 권위있는 해양 데이터 스트림에 실시간으로 접근할 수 있는 대화형 플랫폼입니다. 이 시스템은 사용자가 묻는 질문에 대해 API를 통해 데이터 호출을 수행하고, 이를 바탕으로 재현 가능한 자연어 응답과 데이터 시각화를 생성합니다. 기존의 대화형 AI 제품에 비해, OceanAI는 NOAA 자료를 이용한 신뢰할 수 있는 값을 생성하는 올바른 접근성을 보여줍니다.

- **Technical Details**: OceanAI는 세 가지 주요 디자인 전략을 통해 기존 접근 방식의 한계를 극복합니다. 첫째, 각 질문은 파라미터화된 함수 호출로 변환되어 NOAA와 같은 권위 있는 데이터셋에 접근합니다. 둘째, 자동화된 데이터 처리 및 시각화를 통해 전문 지식이 없는 사용자도 손쉽게 이용할 수 있습니다. 마지막으로, 모든 응답은 메타데이터를 포함하여 신뢰할 수 있는 관측치를 바탕으로 최신 결과를 제공합니다.

- **Performance Highlights**: Blind 비교 실험에서 OceanAI는 세 가지 널리 사용되는 AI 채팅 인터페이스 제품 중 유일하게 NOAA 출처의 데이터를 제공하며, 다른 제품들은 답변을 거부하거나 신뢰할 수 없는 결과를 전달했습니다. OceanAI의 출력을 통해 신뢰성, 재현성, 투명성을 높이고 있으며, 해양 분야 내 AI 지원 의사 결정을 위한 확장 가능한 프레임워크를 제공하고 있습니다.



### Prompt-R1: Collaborative Automatic Prompting Framework via End-to-end Reinforcement Learning (https://arxiv.org/abs/2511.01016)
- **What's New**: 최근 대형 언어 모델(Large Language Models, LLMs)의 발전이 급격히 진행되고 있습니다. 그러나 복잡한 문제에 직면했을 때, 사용자들은 효과적인 프롬프트(prompt)를 제공하지 못해 LLM의 성능이 제한되는 경우가 많습니다. 이를 해결하기 위해, 우리는 소규모 LLM이 대규모 LLM과 협력하여 문제를 해결하는 엔드-투-엔드 강화 학습 프레임워크인 Prompt-R1을 제안합니다.

- **Technical Details**: Prompt-R1은 다중 턴(prompt interaction) 프롬프트 상호작용을 통해, 소규모 LLM이 프롬프트를 생성하고 대규모 LLM이 복잡한 추론을 수행하는 구조로 설정됩니다. 이 프레임워크는 정확성, 생성 품질, 추론 정확성을 최적화하기 위해 이중 제약 보상(dul-constrained reward)을 설계했습니다. 또한, 다양한 대규모 LLM을 지원하는 plug-and-play 프레임워크를 제공하여, 추론과 훈련 모두에서 사용될 수 있습니다.

- **Performance Highlights**: 여러 공개 데이터셋에 대한 실험 결과, Prompt-R1은 다양한 작업에서 기준 모델에 비해 성능이 크게 향상된 것으로 나타났습니다. 이 방법은 복잡한 추론 및 다중 턴 상호작용 작업에서 LLM의 추론 능력을 강화하며, 소규모 LLM의 능력도 개선합니다. 적응성이 뛰어나며, 특정 작업에 대한 fine-tuning 없이도 다양한 작업에서 적용 가능한 가능성을 보여줍니다.



### IF-CRITIC: Towards a Fine-Grained LLM Critic for Instruction-Following Evaluation (https://arxiv.org/abs/2511.01014)
Comments:
          21 pages, 5 figures

- **What's New**: 이 논문에서는 IF-CRITIC이라는 새로운 LLM 비평가를 제안합니다. 이를 통해 입력 지침에 따라 제약 조건을 효율적이고 신뢰성 있게 평가할 수 있습니다. 기존의 평가 모델들이 가진 높은 비용과 신뢰성 부족 문제를 해결하기 위해 체크리스트 생성기를 활용하여 지침을 분해하고 제약 체크리스트를 생성합니다.

- **Technical Details**: IF-CRITIC은 14억 개의 매개변수를 가진 LLM 비평가로, 이는 체크리스트 기반의 비평 생성 패러다임을 채택하여 지침의 모든 제약 조건을 한 번의 추론으로 평가할 수 있게 합니다. multi-stage critique filtering 메커니즘을 통해 고품질 비평 훈련 데이터를 수집하고, constraint-level preference optimization 방법을 적용하여 모델을 훈련합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, IF-CRITIC은 Deepseek-R1과 o4-mini를 포함한 강력한 LLM-as-a-Judge 기준을 초월한 성과를 보여줍니다. IF-CRITIC이 제공하는 세분화된 평가 결과를 보상 신호로 활용함으로써 LLM의 지침 따르기 능력을 향상시킬 수 있으며, 기존 기준에 비해 훨씬 적은 계산 비용으로 대폭적인 성능 향상을 실현할 수 있음을 입증합니다.



### MARS-SQL: A multi-agent reinforcement learning framework for Text-to-SQL (https://arxiv.org/abs/2511.01008)
- **What's New**: MARS-SQL은 자연어 쿼리를 SQL로 변환하는 데 있어 복잡한 쿼리를 처리하기 위한 새로운 다중 에이전트 프레임워크입니다. 이 시스템은 Grounding Agent, Generation Agent, Validation Agent의 세 가지 전문화된 에이전트로 구성되어 있습니다. 이 프레임워크는 상호작용 강화 학습(Interactive Reinforcement Learning, RL)을 통해 훈련된 Generation Agent를 중심으로 하여, SQL 쿼리를 단계적으로 생성하고 실행 피드백을 바탕으로 전략을 수정합니다.

- **Technical Details**: MARS-SQL 프레임워크는 ReAct 스타일의 Think-Act-Observe 루프를 활용하여 동적인 추론과 자기 수정(self-correction)을 가능하게 합니다. 여러 상호 작용 경로를 탐색하고, Validation Agent가 최적의 경로를 선택함으로써 검증 프로세스를 강화합니다. 이 방법론은 복잡한 SQL 쿼리를 효율적으로 생성하는 데 필요한 상호작용 및 선택 기능을 통합하여 기존 시스템의 한계를 극복합니다.

- **Performance Highlights**: MARS-SQL은 BIRD 개발 세트에서 77.84%, Spider 테스트 세트에서 89.75%의 실행 정확도를 기록하며 새로운 최첨단 성능을 달성했습니다. 이러한 성과는 MARS-SQL의 상호작용 다중 에이전트 구조의 효과성을 보여줍니다. 이로써 복잡한 데이터베이스 쿼리를 효율적으로 처리하는 데 있어 중요한 진전을 이룹니다.



### Advancing Machine-Generated Text Detection from an Easy to Hard Supervision Perspectiv (https://arxiv.org/abs/2511.00988)
- **What's New**: 기존의 기계 생성 텍스트(MGT) 탐지 방법은 레이블을 '금본위기준'으로 암묵적으로 가정하고 있다. 그러나 이 연구에서는 MGT 탐지에서 경계 모호성이 존재함을 밝혀내어 전통적인 학습 패러다임의 부정확성을 시사한다. 이를 해결하기 위해, 이 연구는 불확실한 조건 하에서 신뢰할 수 있는 감독을 제공하기 위해 간단한 작업부터 어려운 작업으로 진행하는 향상 프레임워크를 제안한다.

- **Technical Details**: 제안된 프레임워크는 상대적으로 간단한 긴 텍스트 탐지 작업에 초점을 맞춘 감독을 사용하여 더 어려운 목표 탐지기를 개선하는 구조적 접근을 포함한다. 이는 검출기와 감독을 구조적으로 통합하여, 감독이 검출기의 하한 성능으로 모델링되도록 한다. 이는 검출기를 간접적으로 최적화할 수 있는 방법이다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 프레임워크는 교차 LLM, 교차 도메인, 혼합 텍스트 및 패러프레이징 공격을 포함한 다양한 실용적인 시나리오에서 뛰어난 탐지 효과를 보여주었다. 따라서 이 연구는 불확실한 레이블 환경에서도 효과적인 학습이 가능함을 입증하였다.



### The Riddle of Reflection: Evaluating Reasoning and Self-Awareness in Multilingual LLMs using Indian Riddles (https://arxiv.org/abs/2511.00960)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 문화적으로 기반한 추론 능력을 인도 주요 7개 언어(벵갈리, 구자라트어, 힌디어, 칸나다어, 말라얄람어, 타밀어, 텔루구어)에서 평가합니다. 기존 연구가 주로 영어 중심의 평가에 국한되어 있다는 점을 지적하며, 다양한 언어로 된 수수께끼 해결 과정을 통해 모델의 성능을 분석합니다. 이 과정에서 전통적인 수수께끼와 맥락 재구성 변형을 결합한 다국어 수수께끼 데이터셋을 소개하여, LLM의 다언어적 추론 능력과 자기 평가 능력을 체계적으로 조명합니다.

- **Technical Details**: 논문에서는 LLM의 성능을 평가하기 위해 여러 프롬프트(예: zero-shot, few-shot)를 사용하여 수수께끼 해결 능력을 측정합니다. 5개의 LLM(Gemini 2.5 Pro, Gemini 2.5 Flash, Mistral-Saba, LLaMA 4 Scout, LLaMA 4 Maverick)의 성능을 비교하며, 수수께끼 해결에서 Gemini 2.5 Pro가 최고의 성과를 보이는 반면, 다른 모델들은 언어마다 상이한 정확성을 보입니다. 또한 자기 평가 실험을 통해 모델이 자신의 오류를 인식하는 능력을 분석하였으며, 높은 정확도를 보이는 모델이 자기 인식에서는 과신하는 경향을 보인다는 결과를 발견했습니다.

- **Performance Highlights**: 주요 결과로, Gemini 2.5 Pro는 전체적으로 가장 높은 수수께끼 해결 성능을 보였으며, few-shot 방식은 미미한 성과 향상을 가져오는 데 그쳤습니다. 모델의 초기 정확도와 자기 평가 능력은 역 비례 관계가 있는 것으로 나타났으며, 최고 성과 모델인 Gemini 2.5 Pro는 상대적으로 낮은 True Negative Rate(4.34%)를 보였습니다. 반면, 더 낮은 성과를 내는 LLaMA 4 Scout는 주목할 만한 자각 능력(42.09% True Negative Rate)을 보여, LLM의 다언어적 추론에서의 격차와 자기 인식에 있어 명확한 차이를 드러냈습니다.



### The Biased Oracle: Assessing LLMs' Understandability and Empathy in Medical Diagnoses (https://arxiv.org/abs/2511.00924)
Comments:
          Accepted by NeurIPS 2025 GenAI4Health Workshop

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)이 의료 진단 상황에서 환자와의 소통을 어떻게 지원하는지를 평가합니다. 연구는 LLM이 환자의 사회-인구학적 변수와 환자 상태에 맞는 설명을 생성할 수 있는 우리의 능력을 조사합니다. 하지만 LLM이 생성하는 콘텐츠의 복잡성과 편향된 감정 이입이 존재하여 접근성 및 지원에 부정적인 영향을 미친다는 점도 강조됩니다.

- **Technical Details**: 연구는 다양한 임상 시나리오와 인구 통계 프로파일을 기반으로 의사-환자 대화를 생성하는 프레임워크를 제안합니다. LLM은 그 결과물을 명료성(readability), 전문 용어의 집적(jargon density) 및 구조적 복잡성(structural complexity)이라는 관점에서 평가합니다. 또한, 감정적(empathy) 및 인지적(cognitive) 이입을 고려하여 LLM의 출력을 인간 평가와 비교 분석합니다.

- **Performance Highlights**: 평가 결과, 모델은 사회-인구적 변수와 의료 조건에 따라 출력 결과를 조정하여 각기 다른 이해 가능성과 감정 이입 수준을 보입니다. 그러나 LLM은 여전히 과도하게 복잡한 의료 내용을 생성하고, 그룹과 조건에 따라 감정 이입의 변동성이 존재하며, 자신들의 감정적 능력에 대해 편향된 자기 평가를 나타내기도 합니다. 이러한 결과는 LLM의 한계를 여실히 드러내며 공정하고 신뢰할 수 있는 환자 소통의 도전 과제를 강조합니다.



### ColMate: Contrastive Late Interaction and Masked Text for Multimodal Document Retrieva (https://arxiv.org/abs/2511.00903)
- **What's New**: 이번 논문에서는 ColMate라는 새로운 다중 모드 문서 검색 모델을 소개합니다. 기존 모델들이 텍스트 전용 검색을 위한 방식과 유사한 기술을 사용하여 다중 모드 문서 검색의 한계를 극복하지 못하는 문제를 해결하고자 합니다. 수동으로 레이블이 붙은 쿼리-문서 쌍에 의존하지 않고, 비주얼 토큰 최적화를 통해 비주얼 및 텍스트 정보를 효과적으로 통합하는 방법을 제안합니다.

- **Technical Details**: ColMate는 세 가지 주요 구성 요소를 포함합니다: (i) OCR 기반 마스킹 언어 모델링(MOLM) 훈련 목표, (ii) 자가 지도 대조 학습(MaskedCL) 목표, 그리고 (iii) TopKSim이라는 정교한 레이트 상호 작용 메커니즘입니다. 각각의 구성 요소는 비주얼 문서의 복잡한 구조를 효율적으로 캡처하고, 기존의 방식과 비교하여 더 나은 성능을 발휘할 수 있도록 설계되었습니다.

- **Performance Highlights**: ColMate는 ViDoRe V2 벤치마크에서 기존 모델들에 비해 3.61%의 성능 향상을 보여주며, 특히 도메인 외 일반화에서 두드러진 결과를 냈습니다. 이 모델은 다양한 도메인에서 일관된 성능 개선을 입증하며, 각 구성 요소의 기여를 수치적으로 평가하고 분석한 결과도 포함되어 있습니다.



### Assessing LLM Reasoning Steps via Principal Knowledge Grounding (https://arxiv.org/abs/2511.00879)
Comments:
          Accepted to EMNLP 2025 Findings

- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 중간 추론이 실제 지식에 잘 근거하고 있는지를 평가할 수 있는 새로운 평가 프레임워크를 제안합니다. 이는 세 가지 주요 구성 요소로 이루어져 있으며, 특히 원자 지식(atomic knowledge)을 포함하는 주된 지식 수집(PK Collection)을 구축하였습니다. 이 프레임워크는 LLM의 추론에 필요한 지식을 정확하게 적용하는지를 평가하기 위한 지식 기반 평가 메트릭스와 경량화된 평가 모델을 포함합니다.

- **Technical Details**: 주된 지식 수집(PK Collection)은 LLM의 성능을 극대화하기 위한 필수 원자 지식으로 구성된 대규모 리포지토리입니다. 이 리포지토리는 MMLU 벤치마크(MMLU benchmark)를 기반으로 112,000개의 주된 지식 단위를 포함하고 있으며, 지식 기반 정밀도, 재현율 및 F1 점수와 같은 평가 메트릭스를 통해 지식의 활용도를 정량적으로 평가합니다. 이를 통해 모델이 정확하게 정보를 기억하고 적용하는지를 파악할 수 있습니다.

- **Performance Highlights**: 새롭게 제안한 평가 메트릭스는 LLM의 중간 단계에서의 지식 활용 부족이나 부적절한 적용을 명확하게 식별하는 데 유용합니다. 연구에서는 또한 이 지식 기반 메트릭스를 통해 모델의 추론 과정의 간결성을 조정하는 방법을 보여주었으며, 이는 최종 성능을 유지하면서도 효율적인 추론 단계를 생성하는 데 기여합니다. 실험 결과는 지식 기반 메트릭스가 모델 행동에 대한 해석 가능한 피드백을 제공함으로써 성능 향상을 유도하고, 모델의 지식 사용 및 토큰 소비를 효과적으로 조절할 수 있음을 나타냅니다.



### TriCon-Fair: Triplet Contrastive Learning for Mitigating Social Bias in Pre-trained Language Models (https://arxiv.org/abs/2511.00854)
- **What's New**: 본 논문에서는 TriCon-Fair라는 새로운 대비 학습 프레임워크를 제안합니다. 이 프레임워크는 편향된 샘플과 비편향 샘플의 상호 관계를 무시하는 기존의 편향 제거 방법과 달리, 각 앵커에 명시적으로 편향된 네거티브와 비편향된 포지티브를 할당하여 긍정-부정 결합을 제거합니다. 이를 통해 사회적 편향(ds) 문제를 효과적으로 완화할 수 있는 윤리적이고 실용적인 솔루션을 제공하고자 합니다.

- **Technical Details**: TriCon-Fair는 (i) 편향 제거 삼중체를 구성하고 (ii) 언어 모델링(문맥 생성) 손실과 함께 분리된 대비 목적을 적용하는 두 단계 프레임워크입니다. 이 과정에서 우리는 성별, 인종, 종교, 나이 등의 보호 속성과 관련된 자원을 활용하며, 각 쌍은 인구통계학적 토큰만으로 차별화된 간단한 자연 반사례(counterfactual) 역할을 합니다. 논문에서는 정보의 비대칭성을 줄이기 위해 긍정과 부정을 분리된 독립적인 경량으로 최적화 하는 방법을 상세히 설명합니다.

- **Performance Highlights**: TriCon-Fair는 여러 기존의 편향 제거 전략과 비교했을 때, 비율적 차별 결정을 줄이면서도 강력한 언어 모델 성능을 유지하는데 성공했습니다. 실험 결과, BERT, ALBERT, GPT-2, LLaMA 등 다양한 사전 학습된 언어 모델에서 편향 점수(Stereotype Score)를 낮추는 동시에 언어 모델링 점수(Language Modeling Score)도 비교적 소폭 하락하는 결과를 보였습니다. 특히 TriCon-Fair는 일반 활성화 점수(Idealized CAT Score)에서 최고 성과를 기록했으며, 이는 편향 제거의 유용성을 입증하는 만큼의 성공적인 결과입니다.



### Optimizing Native Sparse Attention with Latent Attention and Local Global Alternating Strategies (https://arxiv.org/abs/2511.00819)
- **What's New**: 이번 연구에서는 Native Sparse Attention (NSA)에 대한 체계적인 분석을 진행하고, 장기 컨텍스트 모델링을 향상시키기 위한 목표 개선을 제안합니다. 주요 통찰점은 각 레이어에서 고정된 패턴 대신 지역(슬라이딩 윈도우)과 글로벌(압축, 선택적) 어텐션을 번갈아 사용하는 것이 장거리 의존성의 효과적인 전파를 가능하게 하고, 긴 시퀀스 작업에서 성능을 크게 향상시킨다는 것입니다. 또한, 슬라이딩 윈도우 가지를 Multi-head Latent Attention (MLA)로 강화하고, 압축 및 선택적 가지에는 Group-head Latent Attention (GLA)을 적용하여 NSA의 가지를 더욱 정제합니다.

- **Technical Details**: 이 연구는 ASA(Alternating Sparse Attention)라는 새로운 희소 어텐션 아키텍처를 제안합니다. ASA는 슬라이딩 윈도우 어텐션과 압축/선택적 어텐션의 두 가지 상호 보완적 유형으로 구성된 어텐션 레이어 구조를 유지합니다. 이 두 메커니즘은 레이어를 지나면서 엄격하게 1:1 패턴으로 번갈아 배치되어 모델 전반에 걸쳐 지역 및 글로벌 정보를 균형 잡히게 표현합니다. ASA는 기존의 NSA에서 업데이트되었으며, 이를 통해 KV-cache 저장 오버헤드를 50% 줄이면서도 모델 품질은 유지하고 있습니다.

- **Performance Highlights**: ASA는 340M 및 1.3B 파라미터의 트랜스포머 모델에 대해 평가되었으며, 15B 및 100B 토큰으로 교육된 결과, 일반적인 상식 추론, 긴 컨텍스트 검색 및 긴 컨텍스트 이해 작업에서 기존의 전면 어텐션 기준을 초과하는 성능을 나타냅니다. 경험적 결과는 ASA가 전면 어텐션 기준과 동등하거나 그 이상의 성능을 달성하면서도 기존의 희소 어텐션 접근 방식보다 뛰어남을 보여줍니다. 이러한 성과는 모델의 메모리 효율성을 크게 개선하는 동시에 일반적인 이해 및 추론 능력도 향상시킵니다.



### Do Methods to Jailbreak and Defend LLMs Generalize Across Languages? (https://arxiv.org/abs/2511.00689)
- **What's New**: 이 논문은 10개 언어에서의 jailbreak 공격 및 방어 메커니즘의 첫 체계적인 다국어 평가를 제시합니다. 사용할 언어는 고급, 중급, 저급 자원 언어에 걸쳐 있으며, 6개의 대형 언어 모델(LLM)을 HarmBench와 AdvBench를 사용해 평가합니다. 논문에서는 논리 표현 기반(jailbreak)과 적대적 프롬프트 기반(jailbreak) 방식의 두 가지 공격 유형을 분석하고, 언어별 안전성의 차이를 보여줍니다.

- **Technical Details**: 저자들은 HarmBench와 AdvBench를 기반으로 6개의 LLM을 사용하여 두 가지 지하감시 전술로 공격을 수행했습니다. 첫 번째 방식인 Logic Jailbreak는 해로운 쿼리를 논리 표현으로 변환하고, 두 번째 방식인 prompt 기반 공격은 최적화된 프롬프트를 사용하여 모델이 응답을 거부하도록 하지 않도록 합니다. 저자들은 두 가지 방어 메커니즘인 프롬프트 기반 방어와 간단한 분류기를 훈련하여 공격에 대응합니다.

- **Performance Highlights**: 실험 결과, 공격 성공률과 방어의 견고함은 언어에 따라 다르게 나타났습니다. 고급 언어에서는 표준 쿼리에서 더 안전하지만 적대적 쿼리에서는 더 취약했습니다. 또한 저자들은 단순한 분류기를 사용하여 응답의 안전성을 효과적으로 탐지할 수 있었다고 보고하며, 이러한 다국어 평가가 LLM의 안전성을 향상시키기 위한 더 강력하고 공정한 방법의 필요성을 강조합니다.



### Do You Know About My Nation? Investigating Multilingual Language Models' Cultural Literacy Through Factual Knowledg (https://arxiv.org/abs/2511.00657)
Comments:
          Accepted in EMNLP 2025. Code at: this https URL

- **What's New**: 이 논문에서는 다언어 질문-답변(Question-Answering) 모델이 지역적 다양성을 고려하지 않고 서구 중심적으로 평가되는 점을 지적합니다. 이를 해결하기 위해 XNationQA라는 새로운 데이터셋을 소개하며, 이는 9개 국가의 지리, 문화, 역사에 대한 49,280개의 질문을 7개 언어로 제공합니다. 이 데이터셋을 통해 다국적 LLM들의 문화적 교양(cultural literacy)을 평가하고, 각 언어에 따른 모델의 지식 불균형을 드러냅니다.

- **Technical Details**: 연구에서는 8개의 다국어 LLM을 XNationQA에서 평가하며, 모델의 성과를 새로운 전이 메트릭(metrics)을 사용하여 분석합니다. 질문은 전쟁, 지도자, 기념물 및 국립공원 등 4개 분야로 분류되며, 이는 각 국가의 역사적이고 지리적으로 중요한 사실을 포함합니다. 연구자들은 이러한 질문들을 가지고 모델의 문화적 교양을 체계적으로 평가할 수 있습니다.

- **Performance Highlights**: 평가는 모델들이 서구 언어(영어, 독일어, 스페인어, 러시아어)에서 더 나은 성능을 보임에도 불구하고, 해당 국가의 모국어로 질문했을 때는 기대만큼의 성과를 내지 못한다는 것을 발견했습니다. 특히, 오픈소스 모델들은 언어 간의 사실 전이 능력이 제한적이며, 서구 언어를 제외하고는 신뢰성 있는 응답을 생성하는 데 어려움을 겪습니다. 이로 인해 인도, 중국, 일본과 같은 비서구 국가들에 대한 이해도가 서구 국가에 비해 더 높다는 흥미로운 결과를 도출했습니다.



### Modeling the Construction of a Literary Archetype: The Case of the Detective Figure in French Literatur (https://arxiv.org/abs/2511.00627)
Comments:
          19 pages, 2 tables, 5 figures Conference Computational Humanities Research 2025

- **What's New**: 이 연구는 프랑스 탐정 소설에서 탐정 원형의 진화를 계산 분석을 통해 탐구합니다. 150년의 문헌을 아우르며 M. Lecoq(1866)부터 Commissaire Adamsberg(2017)까지의 탐정 원형의 일관성을 포착하는 감독 학습(supervised model)을 도입했습니다. 연구는 탐정 캐릭터가 2차적인 서사적 역할에서 점차 중심 인물로 변화하고, '사고 기계(reasoning machine)'의 모습을 띠게 되는 과정을 보여줍니다.

- **Technical Details**: 본 연구에서는 Bayesian mixed-effects 모델을 활용하여 대규모 프랑스 소설 코퍼스에서 탐정 캐릭터를 자동으로 탐지하는 분류기를 구축합니다. 연구는 Chapitres Corpus를 통해 매년 탐정 비율을 정량화하고, 문학 장르가 발전함에 따라 탐정이 점점 더 중심적인 역할을 맡게 됨을 증명합니다. 탐정 서브 장르의 출현과 관련된 문학 이론을 지지하는 방법론을 통해 하드보일드 탐정이나 프랑스 로망 노아르와 같은 탐정 서브 장르의 발전을 탐구합니다.

- **Performance Highlights**: 탐정 원형은 미스터리를 해결하는 기계적인 사고의 극단적인 상징으로 발전해왔습니다. 클래식 탐정인 Dupin과 Holmes는 순수한 지적 작업을 강조하며, 인간의 동기를 배제하고 논리적 추론에 대한 집중도를 높입니다. 탐정 소설 장르는 19세기 말부터 주요 문학 장르로 자리잡으며, 1920년대와 30년대에는 집단적 상상력에서 중추적인 존재로 인식되었습니다.



### Certain but not Probable? Differentiating Certainty from Probability in LLM Token Outputs for Probabilistic Scenarios (https://arxiv.org/abs/2511.00620)
Comments:
          To appear at the Second Workshop on Uncertainty-Aware NLP @EMNLP 2025 (UncertaiNLP '25)

- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)이 의사결정 지원 및 지식 집약적인 응용프로그램에서 신뢰할 수 있는 사용을 위해 필요로 하는 불확실성 정량화(uncertainty quantification, UQ)의 중요성을 강조합니다. 특히, 모델의 출력 확률이 이론적 확률 분포와 정렬되는지 유무를 조사하고, GPT-4.1 및 DeepSeek-Chat 모델에 대해 10개의 확률 시나리오의 응답을 평가합니다. 이 연구는 일반적인 UQ 방법과는 다르게, 시나리오 제약에 대한 유효성 및 출력 확률의 이론적 분포와의 정렬을 동시에 고려해야 함을 제시합니다.

- **Technical Details**: 연구는 GPT-4.1과 DeepSeek-Chat을 사용하여 임의성, 위험 또는 우연과 관련된 프롬프트를 통해 모델 응답의 두 차원을 측정합니다. 첫 번째는 시나리오 제약에 대한 응답 유효성(validity), 두 번째는 토큰 출력 확률(token output probabilities)이 이론적 확률(theoretical probabilities)과 얼마나 일치하는지를 분석하는 것입니다. 결과적으로 두 모델 모두 응답 정확성은 정밀하지만, 토큰 수준에서의 확률 및 엔트로피 값은 이론적 분포와 지속적인 차이를 보였습니다.

- **Performance Highlights**: 연구 결과, GPT-4.1과 DeepSeek은 높은 맥락 이해력과 응답 확실성을 보였지만, 단순한 확률적 시나리오를 필요로 하는 상황에서 그들의 토큰 수준 확률은 신뢰할 수 있는 실제 확률을 나타내지 못했습니다. 특히 이들은 유효한 출력을 100% 정확도로 생성하였지만, 그 확률 및 엔트로피 값은 이론적 값과 일치하지 않았습니다. 이러한 발견은 언어 모델이 이론적 확률을 정확하게 추론할 수 있는지, 그리고 불확실성 정량화 방법을 어떻게 조정할 수 있는지를 탐색해야 함을 시사합니다.



### SpecDiff-2: Scaling Diffusion Drafter Alignment For Faster Speculative Decoding (https://arxiv.org/abs/2511.00606)
- **What's New**: 이번 논문은 SpecDiff-2라는 새로운 프레임워크를 제안하여 대규모 언어 모델의 추론 속도를 개선하는 데 중점을 둡니다. 이 방식은 기존의 최신 접근법들이 가진 두 가지 주요 병목현상—자기회귀(autoregressive) 의존성과 드래프트 토큰의 불일치—를 동시에 해결합니다. SpecDiff-2는 비자기회귀(draft-then-verify) 방식을 채택하여 효율적인 텍스트 생성을 가능하게 하며, 이를 통해 속도와 정확성을 모두 개선합니다.

- **Technical Details**: SpecDiff-2는 두 가지 핵심 메커니즘을 사용하여 병목현상을 해결합니다. 첫 번째는 비자기회귀 드래퍼(discrete diffusion model)를 활용하여 드래프트 과정을 최적화하는 것이고, 두 번째는 드래프터와 검증자(verifier) 간의 정렬을 향상시키기 위한 새롭고 혁신적인 기술입니다. 이를 통해 모델은 드래프트 단계에서 높은 퀄리티의 토큰을 신속하게 생성하고, 체계적인 검증 과정에서 적합한 토큰을 선택할 수 있습니다.

- **Performance Highlights**: 실험 결과, SpecDiff-2는 기존의 기준선보다 평균 55% 더 많은 토큰을 초 단위로 처리할 수 있는 성능 향상을 보여주었습니다. 또한, 표준 디코딩(standard decoding)보다 최대 5.5배의 속도 개선을 이루었고, 정확성을 손상시키지 않았습니다. 이러한 성과는 논문이 제안하는 방법론의 효과와 혁신성을 입증합니다.



### OpenSIR: Open-Ended Self-Improving Reasoner (https://arxiv.org/abs/2511.00602)
- **What's New**: 이 논문에서는 Open-Ended Self-Improving Reasoner (OpenSIR)라는 새로운 자기 학습 프레임워크를 제안합니다. OpenSIR은 대규모 언어 모델(LLM)이 외부 감독 없이 문제를 생성하고 해결하는 self-play 방식으로, 학생과 교사 역할을 번갈아 수행합니다. 이를 통해 기존의 데이터 의존성을 줄이고 수학적 발견을 지속적으로 진화시킬 수 있는 가능성을 제시합니다.

- **Technical Details**: OpenSIR은 강화 학습을 통해 교사가 문제를 생성하고 학생이 이를 풀이하는 구조로 작동합니다. 문제는 난이도와 다양성을 최적화하여 생성되며, 이 과정에서 기본적인 형식 준수를 고려합니다. OpenSIR의 학습 과정은 문제 생성, 풀이 샘플링, 점수 산정 및 모델 업데이트의 네 단계로 구성되며, 문제의 난이도와 개념적 다양성을 동시에 추구합니다.

- **Performance Highlights**: OpenSIR은 주어진 트리비얼(seed problem) 문제에서 시작하여, Llama-3.2-3B-Instruct 모델의 경우 GSM8K에서 73.9에서 78.3으로(+4.4), College Math에서는 28.8에서 34.4로(+5.6) 향상되었습니다. Gemma-2-2B-Instruct 모델은 GSM8K에서 38.5에서 58.7로(+20.2) 개선되며, 다양한 수학적 주제를 탐구할 수 있는 능력을 발전시킵니다.



### FlashEVA: Accelerating LLM inference via Efficient Attention (https://arxiv.org/abs/2511.00576)
Comments:
          Technical Report

- **What's New**: 이번 연구에서는 FlashEVA라는 이름의 효율적인 구현을 소개합니다. FlashEVA는 EVA (Efficient Attention via Control Variates) 방법을 기반으로 하여 Transformer의 메모리 사용량과 계산 부담을 줄이는 동시에, 인퍼런스(inference) 중 우수한 성능을 유지할 수 있게 합니다. 이 방법은 1.5B 토큰을 사용하여 Transformer 모델을 미세 조정할 수 있도록 하며, 다양한 다운스트림 작업에서도 효과적으로 작용합니다.

- **Technical Details**: FlashEVA는 커스텀 CUDA 및 Triton 커널을 사용하여 효율적인 EVA 주의를 구현함으로써 메모리 부담이 줄어들게 합니다. 이 접근법은 토큰 수가 적은 경우에도 기존 성능의 대부분을 회복할 수 있도록 하며, 인퍼런스에서 최대 6.7배의 처리량과 5배의 GPU 메모리 사용량 감소를 기록합니다. 또한 FlashEVA는 하이퍼파라미터를 통해 처리량과 정확도 사이의 균형을 조절할 수 있는 기능을 제공합니다.

- **Performance Highlights**: FlashEVA는 긴 문서와 같은 장기 맥락에서의 인퍼런스에서 인상적인 성능 개선을 이룹니다. 특히, 인퍼런스 처리량이 6.7배 증가하고, GPU 메모리 사용량은 5배 감소하는 성과를 보입니다. 다만, 검색 중심의 작업에서는 여전히 성능 저하가 관찰되며, 이 문제는 향후 연구의 여지로 남아 있습니다.



### Friend or Foe: How LLMs' Safety Mind Gets Fooled by Intent Shift Attack (https://arxiv.org/abs/2511.00556)
Comments:
          Preprint, 14 pages, 5 figures, 7 tables

- **What's New**: 이 논문에서는 ISA(의도 전이 공격)라는 새로운 방법을 제안하여, 대형 언어 모델(LLM)이 공격의 의도를 혼동하도록 만드는 기법을 소개합니다. 기존의 공격 기법들이 LLM의 주의를 산만하게 하는 추가적인 맥락이나 적대적 토큰을 도입하는 반면, ISA는 최소한의 언어적 수정만으로 해로운 요청을 무해해 보이게 변형합니다. 이는 LLM이 사용자와의 상호작용 이력 없이 요청의 의도를 정확히 평가하기 어렵다는 점을 이용하고 있습니다.

- **Technical Details**: ISA는 해로운 요청을 무해한 요청으로 변형하기 위해 최소한의 언어적 수정을 적용하는 독창적인 접근 방식을 사용합니다. 연구팀은 이러한 변형을 위해 의도 변형의 세 가지 지방정의(각기 다른 언어적 신호) 지표를 제시하며, 세미틱(의미적) 내용은 유지하면서도 LLM의 의도 추론을 바꾸도록 설계하였습니다. 이를 통해 LLMs가 무해한 정보 요청으로 오해하도록 유도하며, 효과적인 공격 방식을 제시하고 있습니다.

- **Performance Highlights**: ISA는 오픈 소스 및 상업용 LLM에서 해로운 프롬프트에 비해 공격 성공률을 70% 이상 개선하는 성과를 보였습니다. 나아가, ISA 템플릿으로 수정된 무해한 데이터로 모델을 세밀하게 조정하면, 공격 성공률을 거의 100%까지 끌어올릴 수 있다는 점도 밝혀졌습니다. 이러한 결과는 LLM이 표면적으로 수정된 질의 뒤에 숨겨진 위험을 인지하지 못하는 심각한 취약성을 드러내고 있습니다.



### Multi-refined Feature Enhanced Sentiment Analysis Using Contextual Instruction (https://arxiv.org/abs/2511.00537)
- **What's New**: 이번 연구에서는 감정 분석(sentiment analysis)과 관련된 기존 기법들이 정서적 뉘앙스, 도메인 이동(domain shift), 비대칭적인 감정 분포에서 성능이 떨어진다는 문제점을 지적합니다. 이를 해결하기 위해 CISEA-MRFE라는 새로운 PLM 기반의 프레임워크를 제안하였으며, Contextual Instruction (CI), Semantic Enhancement Augmentation (SEA), Multi-Refined Feature Extraction (MRFE) 등 다양한 기술을 통합하여 효과적인 감정을 추출할 수 있도록 설계되었습니다. 특히, 향상된 정서 인식을 통해 사람들의 감정적 반응을 더욱 깊이 이해하는 데 기여하고자 합니다.

- **Technical Details**: CI는 도메인Aware 지침을 삽입하여 감정 해석을 지원하며, SEA는 의미를 유지하면서 다양한 문장 구조를 수용하여 강건성을 향상시키는 역할을 합니다. MRFE는 Scale-Adaptive Depthwise Encoder (SADE)와 Emotion Evaluator Context Encoder (EECE)를 결합하여 로컬 및 글로벌 감정 표현을 동시에 포착합니다. 이 연구는 BERT 기반의 복잡한 특징들을 추출하여 감정적 특성을 보다 정교하게 분석할 수 있는 방법론을 제시합니다.

- **Performance Highlights**: CISEA-MRFE는 IMDb에서 4.6%, Yelp에서 6.5%, Twitter에서 30.3%, Amazon에서 4.1%까지의 정확도 개선을 보이며 강력한 기준 모델 대비 우수한 성능을 입증합니다. 실험 결과는 다양한 도메인에 걸쳐 감정 분류에 있어 제안된 접근 방식의 효과와 일반화 능력을 확인할 수 있는 기회를 제공합니다. 이러한 성과는 감정 분석의 새로운 방향성과 가능성을 제시합니다.



### Word Salad Chopper: Reasoning Models Waste A Ton Of Decoding Budget On Useless Repetitions, Self-Knowingly (https://arxiv.org/abs/2511.00536)
- **What's New**: 이 논문에서는 대규모 추론 모델(Large Reasoning Models, LRM)에서 비효율적인 중복 토큰, 즉 '워드 샐러드(word salad)'의 문제를 해결하기 위한 새로운 방법을 제안합니다. 이를 통해 불필요한 출력 토큰을 줄일 수 있으며, LRM이 이러한 중복을 인식하고 있다는 흥미로운 사실도 발견하였습니다. 연구진은 이 문제를 해결하기 위한 경량의 선형 분류기인 WordSaladChopper (WSC)를 제안하여, 이를 통해 쉽게 검출하고 수정할 수 있는 절차를 마련하였습니다.

- **Technical Details**: 워드 샐러드의 유무를 감지하기 위해, LRM의 추론 과정에서 발생하는 각 청크(chunk)의 숨겨진 상태를 분석합니다. 연구진은 각 청크에서 발견된 패턴을 기반으로 간단한 임베딩 모델을 활용하여 유사도를 계산하고, 특정 기준 이상일 때 워드 샐러드 청크로 분류합니다. 이 과정을 통해 LRM의 reasoning 속도를 최소한으로 방해하면서도 출력의 질을 유지할 수 있도록 합니다.

- **Performance Highlights**: 제안된 WSC를 통해 최대 92%의 워드 샐러드 토큰을 제거하면서도, reasoning 품질의 손실을 최소화할 수 있음을 보여주었습니다. 이 방법은 LRM을 사용하는 모든 애플리케이션에서 적용 가능하며, 시간과 비용을 크게 절감할 수 있는 가능성을 가지고 있습니다. 결과적으로, 논문은 워드 샐러드의 문제 해결이 모든 LRM 응용에서 필수적이라는 점을 강조하고 있습니다.



### Exploring and Mitigating Gender Bias in Encoder-Based Transformer Models (https://arxiv.org/abs/2511.00519)
Comments:
          25 pages, 20 figures

- **What's New**: 이 논문은 언어 모델에서의 성 편향에 대한 새로운 연구를 제시하며, 특히 BERT, ALBERT, RoBERTa 및 DistilBERT와 같은 인코더 기반 트랜스포머 모델들의 성 편향을 분석한다. 저자들은 모델의 확률을 기반으로 한 새로운 지표인 MALoR을 도입하여 성 편향의 정도를 정량화하며, Counterfactual Data Augmentation을 이용해 성 균형 데이터셋에서 추가적인 프리트레이닝을 수행하는 방법을 제안한다.

- **Technical Details**: 연구에서는 Masked Language Modeling (MLM) 방법을 활용하여 전문적인 맥락에서 성별 용어에 대한 모델들의 확률 할당을 분석한다. 이 과정에서 MALoR이라는 새로운 메트릭을 도입하여 성 편향을 평가하고, Counterfactual Data Augmentation을 통해 생성된 성 균형 데이터셋을 이용하여 모델을 추가로 프리트레이닝함으로써 편향을 완화하는 접근을 사용한다.

- **Performance Highlights**: 실험 결과, 다양한 대명사 쌍에서 성 편향 점수가 유의미하게 감소하며, BERT-base 모델에서 "he-she" 쌍의 편향 점수가 1.27에서 0.08로, "his-her"는 2.51에서 0.36으로 저감되었음을 관찰하였다. 또한 BERT-large 모델에서는 "male-female" 편향이 1.82에서 0.10로 감소하였으며, 이러한 접근 방식이 모델의 성능을 저하시키지 않으면서 편향을 효과적으로 완화했다는 결과를 도출하였다.



### Fine-Tuning DialoGPT on Common Diseases in Rural Nepal for Medical Conversations (https://arxiv.org/abs/2511.00514)
Comments:
          6 pages, 6 figures, 3 tables

- **What's New**: 이번 연구에서는 네팔 농촌 지역과 같은 자원이 제한된 환경에서의 의료 지원을 위해 대화형 에이전트(conversational agents)의 가능성을 탐구합니다. DialoGPT라는 경량(generative dialogue model) 생성 대화 모델을 오프라인에서 작동할 수 있도록 미세 조정했습니다. 이 모델은 네팔 농촌 지역에서 흔한 10가지 질병에 대한 의사-환자 상호작용 데이터셋을 기반으로 훈련되었습니다.

- **Technical Details**: DialoGPT는 인터넷 연결과 클라우드 인프라에 의존하지 않는 경량 모델로, 의료 분야에 적합하도록 특정 도메인에서 훈련된 데이터셋을 사용했습니다. 훈련 데이터는 일반 감기, 계절성 발열, 설사, 티푸스, 위염, 식중독, 말라리아, 뎅기열, 결핵, 폐렴 등으로 구성되어 있습니다. 이 모델은 증상 및 질병 맥락에 대한 이해도와 공감 증진의 능력을 증명했습니다.

- **Performance Highlights**: 제한된 특정 도메인 데이터셋으로 훈련된 모델이었지만, 일관성 있고 맥락에 맞는 의료적 응답을 생성할 수 있음을 발견했습니다. 이 결과는 저자원 의료 환경에서의 대화형 모델의 적응성과 목표 지향적 데이터셋(targeted datasets)의 효과성을 강조합니다. 향후 농촌 의료 대화형 AI 개발에 대한 유망한 방향성을 제시합니다.



### Zero-RAG: Towards Retrieval-Augmented Generation with Zero Redundant Knowledg (https://arxiv.org/abs/2511.00505)
- **What's New**: 이번 논문에서 제안된 Zero-RAG는 Retrieval-Augmented Generation (RAG) 과정에서의 지식 중복 문제를 해결하기 위한 새로운 접근 방식을 소개합니다. 기존의 RAG는 Large Language Models (LLMs)와 외부 지식을 결합했지만, LLM의 내부 지식이 증가함에 따라 외부 지식과의 중복이 심화되었습니다. Zero-RAG는 이를 해결하기 위해 Mastery-Score 지표를 사용하여 중복된 지식을 식별하고 제거함으로써 성능을 저해하지 않으면서도 RAG의 효율성을 유지합니다.

- **Technical Details**: Zero-RAG에서는 Mastery-Score를 통해 LLM이 특정 문서를 얼마나 잘 이해하고 있는지를 평가합니다. 이 점수를 기반으로, LLM이 잘 알고 있는 문서들은 제거하여 외부 corpus의 크기를 줄이고 내부 지식을 효율적으로 사용하도록 합니다. 또한 Query Router와 Noise-Tolerant Tuning 모듈을 도입하여 불필요한 문서의 영향을 최소화하고, pruned corpus에서 LLM의 내부 지식을 더욱 잘 활용할 수 있도록 지원합니다.

- **Performance Highlights**: 실험 결과, Zero-RAG는 외부 Wikipedia corpus를 30%까지 줄이며 검색 단계의 속도를 22% 증가시켰습니다. 이러한 조치들은 RAG의 성능을 저해하지 않으면서도 보다 간결한 corpus로 효율적인 정보 검색이 가능하게 했습니다. 결과적으로 Zero-RAG는 RAG corpus의 pruning에 관한 중요한 디자인 선택에 대한 영감을 줄 수 있는 도구로 자리매김할 것입니다.



### ToM: Leveraging Tree-oriented MapReduce for Long-Context Reasoning in Large Language Models (https://arxiv.org/abs/2511.00489)
Comments:
          EMNLP 2025 Main Conference

- **What's New**: 이번 연구에서는 Tree-oriented MapReduce (ToM) 프레임워크를 제안하여 긴 컨텍스트를 다루는 데 있어 기존 방법들이 가진 한계를 극복하고자 합니다. LLMs는 제한된 context window로 인해 긴 문서의 논리적 일관성을 유지하는 데 어려움을 겪습니다. ToM은 계층적 구조를 통해 긴 문서의 정보를 효과적으로 처리하고, MapReduce 방식을 적용하여 더욱 정교한 추론이 가능합니다.

- **Technical Details**: ToM은 두 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째는 Hierarchical Semantic Parsing을 통해 각 청크를 구조화된 서브트리로 변환하고, 이러한 서브트리들을 Bottom-up Aggregation을 통해 통합하여 DocTree를 구성하는 것입니다. 두 번째는 MapReduce 스타일의 재귀적 추론을 수행하여, 각 단계에서 자식 노드에서 얻은 근거를 부모 노드에서 집계함으로써 충돌을 해결하는 방식입니다.

- **Performance Highlights**: 실험 결과, ToM은 70B 이상의 LLMs에서 기존의 divide-and-conquer 방법과 retrieval-augmented generation 방법들을 능가하며, 긴 컨텍스트 추론에서 더욱 우수한 성능을 보여주었습니다. ToM은 효율성과 효과성의 균형을 잘 이뤄내고 있으며, 문서의 구조화된 표현과 재귀적 reasoning을 통해 더욱 향상된 정보 활용이 가능합니다.



### With Privacy, Size Matters: On the Importance of Dataset Size in Differentially Private Text Rewriting (https://arxiv.org/abs/2511.00487)
Comments:
          11 pages, 1 figure, 5 tables. Accepted to IJCNLP-AACL 2025 (Main)

- **What's New**: 최근 자연어 처리(Natural Language Processing)에서 차등 개인 정보 보호(Differential Privacy, DP)에 관한 연구는 텍스트 재작성 메커니즘 형태로 여러 유망한 기술들이 제안되고 있다. 이 논문에서는 데이터셋 크기가 텍스트 개인화 기법의 효율성에 미치는 영향을 처음으로 도입하여, 데이터 크기에 따른 유틸리티와 프라이버시 테스트를 수행하였다. 이를 통해 기계가 다양한 데이터셋 크기에서 어떻게 동작하는지를 정량적으로 분석하였다.

- **Technical Details**: 연구에서 사용된 주요 파라미터는 ε (epsilon)로, 이는 DP에서 개인정보 보호 예산을 나타낸다. 로컬 DP를 위한 다양한 메커니즘이 사용되었으며, 각각은 단어 수준, 토큰 수준, 문서 수준의 차등 개인 정보 보호를 목표로 하고 있다. 여러 크기의 데이터셋에서 실험을 실행하고, 각 데이터셋에 대해 유틸리티와 프라이버시 실험을 실시하여 DP 텍스트 재작성의 효과를 정량화하였다.

- **Performance Highlights**: 데이터셋의 크기가 DP 텍스트 재작업의 프라이버시와 유틸리티 유지 기능에 미치는 영향이 중요한 것으로 나타났다. 실험에서는 데이터셋 크기가 증가함에 따라 비교적 더 유리한 개인 정보 보호 조정이 이루어지는 것을 관찰할 수 있었다. 이는 대규모 환경에서 DP 텍스트 개인화의 가능성을 높이며, 향후 DP NLP 분야의 발전 방향에도 기여할 것으로 예상된다.



### Leveraging the Cross-Domain & Cross-Linguistic Corpus for Low Resource NMT: A Case Study On Bhili-Hindi-English Parallel Corpus (https://arxiv.org/abs/2511.00486)
Comments:
          Accepted in EMNLP 2025

- **What's New**: 이 논문은 인도의 부족 언어 중 하나인 Bhili에 대한 기계 번역의 격차를 해소하기 위해 Bhili-Hindi-English Parallel Corpus (BHEPC)를 소개합니다. BHEPC는 Bhili, 힌디어, 영어로 구성된 110,000개의 문장으로 이루어진 세계 최초의 대규모 병렬 코퍼스입니다. 이 코퍼스는 전문가 번역가들의 도움이 결쳐져 제작되었으며, 교육, 행정 및 뉴스 등 다양한 분야를 포괄합니다.

- **Technical Details**: BHEPC는 Bhili 언어의 저자원 환경에서 다양한 기계 번역 모델을 평가할 수 있는 중요한 기준을 마련합니다. 논문에서는 NLLB-200의 600M 변형 모델이 최적의 성능을 보여주는 것을 강조하며, 븐다국어 대형 언어 모델이 저자원 시나리오에서 가지는 잠재력을 드러냅니다. 연구자는 또한 BHEPC에 대한 생성 번역 능력을 조사하고 cross-domain generalization 및 distributional divergence를 정량화했습니다.

- **Performance Highlights**: BHEPC의 평가 결과, 다양한 오픈 소스 및 독점 모델의 다중 언어 번역 기능을 벤치마킹했습니다. Jensen-Shannon Divergence (JSD)를 활용하여 번역 방향 간의 cross-domain 일반화 분석을 수행하였으며, 이는 번역 프로세스의 효율성을 높이고 다른 저자원 언어에 대한 스케일러블한 템플릿을 제공하는 하이브리드 시드 및 후 편집 워크플로우를 제안했습니다.



### Remembering Unequally: Global and Disciplinary Bias in LLM-Generated Co-Authorship Networks (https://arxiv.org/abs/2511.00476)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 메모리화가 공동 저자 네트워크에 미치는 영향을 분석합니다. 연구에 사용된 모델은 DeepSeek R1, Llama 4 Scout, 그리고 Mixtral 8x7B로, 각 모델의 메모리화 효과가 학문 분야와 지역에 따라 어떻게 다른지를 평가합니다. 연구 결과, 저명한 연구자가 더 정확하고 자주 나타나는 경향이 있음을 보여주며, 이는 LLM 학습 데이터가 불균형을 반영할 수 있음을 시사합니다.

- **Technical Details**: 연구는 10개의 학문 분야와 8개의 글로벌 지역에서 선택된 1,596명의 저자에 대한 공동 저자 네트워크를 OpenAlex와 Google Scholar 데이터를 통해 수집하여 LLM으로 생성된 네트워크와 비교하였습니다. 발견된 Discoverable Network Extraction (DNE) 점수는 저명한 저자들에 대해 훨씬 높은 값을 보여주었으며, 이에 따라 LLM의 메모리화는 모델의 파라미터 수와 저자 인용 횟수의 함수로 증가하는 경향이 있음을 확인했습니다.

- **Performance Highlights**: 모델의 파라미터 수가 많을수록 메모리화 효과가 더 두드러지는 경향을 보였으며, 특히 임상 의학 분야에서는 저명한 저자와 덜 인용된 저자 간에 DNE 점수 차이가 뚜렷하지 않았습니다. 아프리카의 몇몇 지역에서도 낮은 인용의 연구자들이 동등하게 대표되는 모습을 보였으며, 이는 공정한 데이터로서의 훈련 센터를 강조합니다. 이 연구는 LLM이 어떻게 학문적 네트워크 분석에 활용될 수 있는지를 보여주는 중요한 통찰을 제공합니다.



### G2: Guided Generation for Enhanced Output Diversity in LLMs (https://arxiv.org/abs/2511.00432)
Comments:
          EMNLP 2025

- **What's New**: 이 논문에서는 출력 다양성을 향상시키면서 생성 품질을 유지하는 프로그램 없이도 쉽게 적용 가능한 Guide-to-Generation (G2) 방법을 제안합니다. G2는 기본 생성기와 함께 두 개의 가이드를 활용하여 기존 쿼리에 따라 더 다양한 출력을 촉진하는 방법론을 사용합니다. 이 접근법은 고품질의 응답을 보장하는 동시에 반복을 억제하며, 실험 결과는 G2가 출력 다양성을 효과적으로 개선함을 보여줍니다.

- **Technical Details**: G2는 동일한 LLM 내에서 작동하는 세 가지 모듈로 구성됩니다: 기본 생성기, 다양성 가이드(Diversity Guide), 그리고 중복 억제 가이드(Dedupe Guide)입니다. 이 메커니즘은 선택적 개입을 통해 토큰 예측에 대한 높은 신뢰도를 보일 때에는 개입을 하지 않으며, 불확실성이 클 경우에만 가이드를 제공하여 정보의 플루언시(fluidity)와 일관성을 유지합니다. 또한, G2는 중심 선택 전략(CSS)을 도입하여 이전 생성물 중에서 의미적으로 대표적인 하위 집합만을 선택하여 가이드를 조건화합니다.

- **Performance Highlights**: G2는 창의적이고 주관적인 생성, 명령 이행, 번역, 요약, 수학적 문제 해결 등 다양한 작업에서 출력 다양성을 크게 향상시켰습니다. 실험 결과, G2는 높은 응답 품질을 유지하면서도 다양성을 개선하여 LLM 생성 콘텐츠의 다양성을 높이기 위한 유망한 기술로 자리 잡고 있습니다. G2의 코드 구현은 공개되어 있어, 연구자들이 쉽게 접근하고 활용할 수 있습니다.



### MedRECT: A Medical Reasoning Benchmark for Error Correction in Clinical Texts (https://arxiv.org/abs/2511.00421)
- **What's New**: 이 논문에서는 MedRECT라는 새로운 크로스링궐 벤치마크를 도입하여 일본어와 영어의 의료 오류 탐지를 체계적으로 평가합니다. MedRECT는 오류 탐지, 오류 위치 파악 (sentence extraction), 오류 수정을 포함하는 세 가지 하위 작업으로 의료 오류 처리를 구성합니다. 이 벤치마크는 일본 의료 면허 시험인 JMLE에서 자동화된 파이프라인을 통해 구축되어, 기존 벤치마크의 한계를 극복한 고품질의 평가 도구로 작용합니다.

- **Technical Details**: MedRECT는 663개의 일본어 텍스트와 458개의 영어 텍스트를 포함하며, 두 언어 간의 오류/무오류 균형을 유사하게 유지합니다. 9가지 최신 LLM을 평가한 결과, reasoning 모델이 표준 아키텍처보다 우수한 성능을 보였으며, 일본어에서 영어로의 성능 차이를 확인했습니다. 논문에서는 또한 LoRA를 활용한 타겟팅된 파인 튜닝이 각 언어에서 오류 수정 성능을 비대칭적으로 향상시켰음을 보여줍니다.

- **Performance Highlights**: 연구 결과, reasoning 모델이 오류 탐지에서 최대 13.5%, 문장 추출에서 51.0%의 상대적 향상을 보여주었고, 파인 튜닝된 모델은 구조화된 의료 오류 수정 작업에서 인간 전문가의 성능을 초과했습니다. MedRECT는 의료 AI 시스템의 안정성과 신뢰성을 높이는 데 기여할 수 있는 중요한 자원으로, 의료 분야의 크로스링궐 평가의 필요성을 강조합니다.



### PADBen: A Comprehensive Benchmark for Evaluating AI Text Detectors Against Paraphrase Attacks (https://arxiv.org/abs/2511.00416)
- **What's New**: 본 연구는 AI 생성 텍스트(AIGT) 탐지기들이 LLM 출력에는 90% 이상의 정확도를 보이나, 반복적으로 패러프레이즈된 콘텐츠에는 치명적인 실패를 보인다는 점을 다룹니다. 반복 패러프레이징은 내용의 의미 이전을 발생시키며, 이러한 방식이 기존 탐지 시스템을 회피하는 이유를 분석합니다. 이 과정에서 저자는 PADBen이라는 벤치마크를 제안하여 두 가지 공격 범주(저자 모호화 및 표절 회피)를 평가하고 있습니다.

- **Technical Details**: 이 논문은 패러프레이즈 공격이 AIGT 탐지 시스템에서 효과적으로 회피되는 이유를 규명하기 위해 두 가지 가설을 세우고 이를 실험하였습니다. 실험 결과, 반복적으로 패러프레이즈된 텍스트는 고유한 의미 변환을 제공하며, 이는 탐지 시스템의 시각적 공간에서 나타나는 특징적 패턴과 다르게 작용합니다. PADBen 벤치마크는 원본 콘텐츠부터 깊게 세탁된 텍스트에 이르는 다섯 가지 텍스트 분류와 이에 대한 탐지 작업을 제안합니다.

- **Performance Highlights**: 11개의 최신 탐지기를 평가한 결과, 패러프레이즈 공격이 탐지 시스템을 보편적으로 무너뜨리지 않음을 보여주었습니다. 결과는 텍스트의 출처에 따라 달라지는 비대칭성을 드러내며, 저자 모호화의 경우 성능 저하가 두드러지는 반면 표절 회피 문제는 탐지가 원활하게 이루어지는 것으로 나타났습니다. 현재의 탐지 접근 방식은 중간 세탁 영역에서 효과적으로 대처하지 못함을 보여주며, 탐지 아키텍처의 근본적인 발전이 필요함을 시사합니다.



### Reasoning Trajectories for Socratic Debugging of Student Code: From Misconceptions to Contradictions and Updated Beliefs (https://arxiv.org/abs/2511.00371)
Comments:
          25 pages, 2 tables, 13 figures

- **What's New**: 이 논문에서는 소크라틱 디버깅(Socratic debugging)의 개념을 도입하고, 이를 통해 학생들이 버그를 스스로 찾아내고 수정할 수 있도록 돕는 방법을 제시합니다. 특히, 프로그래밍 개념에 대한 잘못된 믿음인 '오해(misconception)'를 해결하는 데 초점을 맞춥니다. 이 과정에서 생성된 추론 경로(reasoning trajectory)와 이를 기반으로 한 소크라틱 대화를 자동화하는 도구들을 소개합니다.

- **Technical Details**: 본 논문에서는 두 가지 주요 하위 작업으로 소크라틱 디버깅을 접근합니다. 첫 번째는 '추론 경로(Reasoning Trajectory, RT)'로, 이는 잘못된 믿음과 모순되는 최종 진술로 이어지는 추론 단계의 시퀀스를 생성하는 것입니다. 두 번째는 '소크라틱 대화(Socratic Conversation, SC)'로, RT의 단계에 따라 학생과 강사 간의 대화를 생성합니다. 이는 강사의 질문이 학생으로부터 진술을 이끌어내도록 구성됩니다.

- **Performance Highlights**: 대규모 LLM(대형 언어 모델) 평가 결과, 최신 모델이 최대 91%의 정확도로 추론 경로를 생성하고 98.7%의 유효한 대화 단계를 만들어낼 수 있다는 것을 보여줍니다. 이러한 결과는 LLM 기반의 접근 방식이 소크라틱 디버깅에 어떻게 기여하는지를 잘 나타냅니다.



### LingGym: How Far Are LLMs from Thinking Like Field Linguists? (https://arxiv.org/abs/2511.00343)
Comments:
          EMNLP 2025 Main

- **What's New**: 이 논문은 LingGym이라는 새로운 벤치마크를 소개하여 LLM(대형 언어 모델)의 메타 언어적 추론 능력을 평가합니다. 이 벤치마크는 18개의 유형학적으로 다양한 참조 문법(reference grammars)에서 추출한 중간 주석 텍스트(Interlinear Glossed Text, IGT)와 문법 설명을 활용합니다. 선행 연구들은 특정 하위 작업에 중점을 두었지만, 우리는 LLM이 훈련 중에 보지 못한 저자원 언어와 구조에 대한 언어적 추론을 일반화할 수 있는지를 평가합니다.

- **Technical Details**: 제어된 평가 작업으로는 단어-주석 추론(Word-Gloss Inference)이라는 작업이 있으며, 이는 모델이 다양한 언어적 정보(예: 주석, 문법 설명, 번역)를 사용하여 문맥에서 누락된 단어와 주석을 추론해야 합니다. 결과적으로 정형화된 언어적 단서를 통합함으로써 모든 모델의 추론 성능이 일관되게 개선되었음을 보여줍니다. 이 연구는 LLM의 유형론적 언어 분석 및 저자원 언어 문서화에서의 유망함과 현재 한계를 강조합니다.

- **Performance Highlights**: 우리는 다양한 저자원 언어에서 LLM이 IGT와 문법 규칙을 사용하여 얼마나 잘 해석하고 추론할 수 있는지를 평가하기 위해 평가 프레임워크를 개발했습니다. 이 연구에서 제안한 작업을 바탕으로 여러 최신 LLM을 벤치마킹하고, 구조적 언어 지식을 처리할 때의 능력과 한계를 강조하며 형태소 분석, 구문 구조 식별 등의 다양한 언어 현상과 유형론적 패턴을 살펴봅니다.



### Reversal Invariance in Autoregressive Language Models (https://arxiv.org/abs/2511.00341)
Comments:
          7 pages, theoretical note

- **What's New**: 이 논문은 인과 (autoregressive) 언어 모델링의 목표로서 대칭 불변성 (reversal invariance)이라는 구조적 속성을 형식화합니다. 구체적으로, 다음 토큰 예측 손실 (next-token prediction loss)은 원본 텍스트와 그 역순 모두에 대해 동일한 가능성을 할당하며, 이는 표준 CLM (causal language modeling) 프리트레이닝이 방향에 구애받지 않음을 의미합니다. 이러한 대칭성은 역순 텍스트로 훈련된 모델이 정방향 텍스트로 훈련된 모델과 유사한 성능을 보이는 이유를 설명하며, 이는 현재의 프리트레이닝 목표의 한계를 제시합니다.

- **Technical Details**: 인간의 언어는 본질적으로 비대칭적입니다. 전제는 결론으로 이어지며, 이 과정은 자유롭게 역전될 수 없습니다. 이 논문에서는 표준 CLM 목표가 이 방향성을 무시함으로써 인과적 및 시간적 이해에 필수적인 정보를 간과할 수 있음을 강조합니다. 하지만 다음 토큰의 음수 로그 우도 (NLL) 손실은 원본 텍스트와 그 역순 및 토큰 인덱스의 순열에 대해 동일한 목표를 최소화하는 대칭성을 조성합니다.

- **Performance Highlights**: 이 연구는 언어가 본질적으로 시간 비대칭적 규칙성을 포함하고 있다는 주장을 뒷받침합니다. 이에 따라, 방향에 구애받지 않는 목표가 그러한 규칙성을 제대로 모델링하지 못할 수 있음을 경고합니다. 미래 연구는 손실 함수와 아키텍처에서 언어의 비대칭성을 명시적으로 모델링하는 방향으로 진행되어야 하며, 이러한 접근이 시간적 지향성을 유지하면서도 성능을 향상시킬 수 있을 것으로 기대됩니다.



### Language Modeling With Factorization Memory (https://arxiv.org/abs/2511.00315)
- **What's New**: 본 논문에서는 Factorization Memory라는 효율적인 순환 신경망( recurrent neural network, RNN) 아키텍처를 제안합니다. 이 모델은 짧은 문맥의 언어 모델링 작업에서 Transformer 모델과 유사한 성능을 달성하며, 긴 문맥 시나리오에서 더 뛰어난 일반화 능력을 보여줍니다. Factorization Memory는 Mamba-2를 기반으로 구축되어, 훈련 중에는 병렬 계산을 활용하면서 추론 시에는 일정한 계산 및 메모리 복잡성을 유지합니다.

- **Technical Details**: Factorization Memory는 두 가지 전략, 즉 조밀한 업데이트(dense update)와 희소한 업데이트(sparse update)를 통해 메모리 상태를 업데이트합니다. 희소한 업데이트를 사용함으로써 매 시간 단계에서 소수의 매개변수만 선택적으로 업데이트하여 계산 비용을 줄이는 동시에 더 큰 순환 상태를 유지할 수 있습니다. 이 방법은 훈련 및 추론 중 부분 활성화(partial activation)를 통해 계산 및 메모리 절약 효과를 달성합니다.

- **Performance Highlights**: Factorization Memory는 짧은 문맥 작업에서 Transformer 및 Mamba-2와 경쟁할 수 있을 뿐만 아니라, 훈련 문맥 길이를 초과하는 부분에서 우수한 성능을 보여줍니다. 또한, 이 모델은 이들 모델에 비해 더 높은 추론 효율성을 달성하였습니다. 이는 Factorization Memory의 효율성이 모델의 성능에 긍정적인 영향을 미쳤음을 시사합니다.



### POSESTITCH-SLT: Linguistically Inspired Pose-Stitching for End-to-End Sign Language Translation (https://arxiv.org/abs/2511.00270)
Comments:
          Accepted at EMNLP 2025 (Main)

- **What's New**: 본 논문에서는 포즈 기반, 글로스(Gloss) 없는 수화 번역(SLT)을 위한 새로운 선행 학습(pre-training) 방식인 POSESTITCH-SLT를 제안합니다. 이는 언어 템플릿을 기반으로 한 문장 생성 기법에서 영감을 받아, 공공 단어 수준 데이터셋을 활용하여 수백만 개의 문장을 생성할 수 있도록 합니다. 이전 연구에 비해 단순한 Transformer 아키텍처를 사용하여, How2Sign와 iSign 데이터셋에서 성능이 향상되었음을 보여줍니다.

- **Technical Details**: POSESTITCH-SLT는 수화 영상에서 추출된 2D 포즈 시퀀스를 기반으로 하여, 해당하는 영어 문장을 생성하는 방법을 논의합니다. 이 방법은 대규모의 구문 구조와 그에 맞는 자연어 데이터를 결합하여 기계 학습 알고리즘의 훈련에 활용합니다. 특히, ASL(미국 수화)과 ISL(인도 수화) 데이터를 사용하여, 공통 어휘를 기반으로 생성된 밀리언 단위의 문장 데이터셋을 만들어, 이는 표준 Transformer 아키텍처로 모델 훈련에 사용됩니다.

- **Performance Highlights**: How2Sign 데이터셋에서 BLEU-4 점수가 1.97에서 4.56으로, iSign 데이터셋에서는 0.55에서 3.43으로 향상되었습니다. 이러한 결과는 기존의 최첨단 기술을 초월하며, pose 기반의 글로스 없는 번역에서의 우수한 성능을 보여줍니다. 이 연구는 저자들의 GitHub 페이지(https://github.com/Exploration-Lab/PoseStich-SLT)를 통해 공개된 데이터셋 및 코드를 활용하여 향후 연구에 기여하고자 합니다.



### IL-PCSR: Legal Corpus for Prior Case and Statute Retrieva (https://arxiv.org/abs/2511.00268)
Comments:
          Accepted at EMNLP 2025 (Main)

- **What's New**: 이 논문은 법률 사례와 관련 법규를 검색하는 두 가지 작업인 Legal Statute Retrieval (LSR)와 Prior Case Retrieval (PCR)를 통합하는 IL-PCSR (Indian Legal Corpus for Prior Case and Statute Retrieval)라는 고유한 말뭉치를 제안합니다. 기존의 접근법은 각각의 작업에 독립적으로 모델을 개발했으나, 두 작업 간의 상호 의존성을 활용하고자 합니다. IL-PCSR은 동일한 쿼리 집합에 대해 관련된 법규와 사례를 동시에 탐색할 수 있는 첫 번째 데이터셋입니다.

- **Technical Details**: IL-PCSR은 936개의 법규와 3,183개의 이전 사례, 6,271개의 쿼리 문서로 구성됩니다. 법률 문서는 인도 대법원과 고등법원 판결에서 수집된 것으로, 공공적으로 이용 가능한 자료를 API로 통해 얻었습니다. 이 데이터셋은 기계 학습 모델이 법규와 사례 간의 의존성을 학습할 수 있도록 설계되었으며, LLM(대규모 언어 모델) 기법을 이용한 재정렬 방법도 개발하였습니다.

- **Performance Highlights**: 논문에서는 다양한 모델을 사용하여 LSR와 PCR 작업의 성능을 평가했으며, LSR과 PCR 각각에 대해 문자 기반 모델과 의미 기반 모델을 포함한 다수의 실험을 수행했습니다. 제안하는 파이프라인 기반 접근 방식은 각 작업의 성능을 개별적으로 개선할 뿐만 아니라, 다중 작업 모델에도 긍정적인 영향을 미쳤습니다. 실험 결과, 법규 검색 및 사례 검색 간의 차이가 성능에 미친 영향을 분석하였고, 이러한 인사이트는 향후 연구의 방향을 제시합니다.



### AgentBnB: A Browser-Based Cybersecurity Tabletop Exercise with Large Language Model Support and Retrieval-Aligned Scaffolding (https://arxiv.org/abs/2511.00265)
- **What's New**: 이번 연구는 기존 사이버 보안 테이블탑 연습(TTXs)의 한계를 극복하기 위해 AgentBnB라는 새로운 브라우저 기반 플랫폼을 소개합니다. 이 시스템은 대규모 언어 모델(large language model) 팀원과 Bloom-aligned의 검색 기반 협력 도우미(C2D2)를 통합하여 학습자의 요구에 맞춘 인지적인 힌트를 실시간으로 제공합니다. 이로 인해 전통적인 연습에 비해 경량화되고 반복적인 학습이 가능해집니다.

- **Technical Details**: AgentBnB는 기존의 'Backdoors & Breaches' 게임을 기반으로 하여 계목산업들 통합한 시뮬레이션 플랫폼입니다. 참가자는 역할 기반 대화를 통해 LLM 팀원과 상호작용하며, C2D2는 Bloom 학습 이론에 기반한 실시간 지침을 제공합니다. 이 시스템은 기억, 이해, 적용 등 다양한 인지 과정에 맞춰 지식을 제공하는 벡터 데이터베이스를 활용합니다.

- **Performance Highlights**: 일대일 파일럿 연구에서 4명의 대학원생이 참여한 결과, 참가자들은 물리적 카드덱보다 에이전트 기반 버전을 사용할 의향이 더 높다고 응답했습니다. 그러나 단순한 지식 퀴즈에서는 ceiling effect가 관찰되었고, 작은 샘플 사이즈와 단일 플레이어 초점, 제한적인 자료 등은 연구의 한계로 지적됩니다. 향후 다수 플레이어 모드, 텔레메트리 기반 코칭 등을 추가할 계획입니다.



### Consistently Simulating Human Personas with Multi-Turn Reinforcement Learning (https://arxiv.org/abs/2511.00222)
- **What's New**: 이번 논문에서는 LLM(대형 언어 모델)의 페르소나 일관성을 평가하고 개선하기 위한 통합 프레임워크를 소개합니다. 기존의 LLM들이 역할에 맞는 행동을 유지하지 못하고, 이전의 진술과 모순되거나, 역할에서 벗어나는 경우가 많습니다. 저자들은 이러한 문제를 해결하기 위해 다양한 자동화된 메트릭(자동 지표)을 정의하였습니다.

- **Technical Details**: 세 가지 자동화 메트릭인 prompt-to-line consistency(프롬프트-라인 일관성), line-to-line consistency(라인-라인 일관성), Q&A consistency(질문-답변 일관성)를 활용하여 기존 LLM 대화에서 발생하는 페르소나 드리프트를 측정합니다. 이러한 메트릭은 인간 주석과 검증되며, 이를 통해 세 가지 사용자 역할(환자, 학생, 사회적 대화 파트너)을 위한 멀티 턴 강화 학습(multi-turn reinforcement learning)을 적용합니다. 이 방법을 통해 LLM의 불일치성을 55% 이상 줄일 수 있었습니다.

- **Performance Highlights**: 우리는 제안한 메트릭을 보상 신호로 활용하여 LLM을 세 가지 사용자 역할에 맞게 미세 조정하였습니다. 그 결과, 좀 더 일관되고 일치된 대화를 생성할 수 있으며, 이로 인해 LLM 기반의 시뮬레이션이 더욱 신뢰할 수 있게 됩니다. 또한, 이러한 접근 방식은 사회 과학 및 강화 학습(RL) 파이프라인에서의 효과적인 응용을 가능하게 합니다.



### Training LLMs Beyond Next Token Prediction - Filling the Mutual Information Gap (https://arxiv.org/abs/2511.00198)
- **What's New**: 이 연구는 대형 언어 모델(LLM) 훈련의 최적화 방식에 도전합니다. 전통적인 다음 토큰 예측(Next-Token Prediction, NTP) 접근법에서 벗어나 정보가 풍부한 토큰을 예측하는 방식이 더 효과적임을 주장합니다. 수학적 계산, 다중 레이블 분류(Multi-Label Classification, MLC), 자연어 생성(Natural Language Generation, TG)의 세 가지 과제를 통해 모델 성능을 개선하는 방법을 제시합니다.

- **Technical Details**: 연구는 LLM의 훈련 과정에서 특정 목표(Gathered) 토큰을 우선시하는 전략을 수립하는 것을 목표로 하고 있습니다. 이 새로운 접근법은 토큰 선택(order of tokens)을 개선하고오류 누적을 줄여 훈련 효율성을 높이는 데 중점을 두고 있습니다. 이론적 프레임워크를 기반으로 하여 원천과 목표 토큰 간의 상호 정보(mutual information)를 분석하여 정보가 풍부한 토큰 우선순위를 설정합니다.

- **Performance Highlights**: 실험 결과, NTP 방식이 수학, MLC, TG의 세 가지 전통적인 시나리오에서 비효율적이라는 사실을 확인했습니다. 정보가 풍부한 토큰 우선순위를 설정한 결과, 다양한 최신 LLM(예: GPT-2, Qwen2.5, Llama-3.2)에서 일관된 정확도, 당황도(perplexity), ROUGE 메트릭 향상을 보여주었습니다. 이로 인해 특정 토큰 예측 전략이 모델 성능 증진에 미치는 영향을 그대로 드러냈으며, LLM 훈련 방법론을 개선하는 대안을 제공합니다.



### ParaScopes: What do Language Models Activations Encode About Future Text? (https://arxiv.org/abs/2511.00180)
Comments:
          Main paper: 9 pages, 10 figures. Total 24 pages

- **What's New**: 이 논문에서는 언어 모델의 활성화(activations)가 점점 더 긴 시간 범위를 가진 작업을 수행할수록 초기 개념이나 토큰에 대한 테스트에 한정되었던 해석 가능성 연구의 한계를 극복하는 새로운 방법론인 Residual Stream Decoders를 제안합니다. 이 방법론은 문단과 문서 단위의 계획을 이해하는 데 도움을 주며, Llama 3.2 3B 모델을 활용하여 정보가 5개 이상의 미래 맥락으로 디코딩될 수 있음을 발견합니다. 이러한 결과는 언어 모델의 모니터링을 개선하고 장기적인 계획 정보를 어떻게 인코딩하는지를 이해하는 데 기초를 마련합니다.

- **Technical Details**: 이 논문에서 제안한 Residual Stream Decoder(RSD)는 모델의 내부 표현에서 향후 콘텐츠를 재구성하는 방법론으로 정의됩니다. RSD는 특히 모델의 잔여 스트림에서 정보를 추출하고 이를 활용하여 향후 출력을 예측하는 데 필요한 맵핑 함수를 포함합니다. 연구에서는 실험적으로 ' 단락(‘

’ 토큰으로 구분되는 부분)' 단위에서 정보가 얼마나 쉽게 디코딩될 수 있는지를 조사하고자 하였으며, 정보 전환이 발생하는 단락 경계에서 모델이 다음 콘텐츠에 대한 정보를 저장하고 있다고 가정했습니다.

- **Performance Highlights**: 실험 결과, Llama-3.2-3B-Instruct 모델은 LLM의 출력으로부터 최대 128개의 토큰을 생성하는데 필요한 단락 내의 미래 맥락을 높은 정확도로 디코드함으로써 계획의 증거를 제공합니다. 이어서 Continuation ParaScope와 TAE ParaScope의 두 가지 보완적인 접근 방식을 도입하여 모델 활성화에 직접 개입하거나 텍스트 자동 인코딩에 기반한 매핑 방법을 통해 디코딩 효과를 극대화합니다. 이 논문의 결과는 언어 모델이 장기적인 계획 정보를 어떻게 인코딩하고 있는지 이해하는 데 중요한 기초를 제공합니다.



### Cognitive Alignment in Personality Reasoning: Leveraging Prototype Theory for MBTI Inferenc (https://arxiv.org/abs/2511.00115)
- **What's New**: 이번 논문에서는 ProtoMBTI라는 프레임워크를 통해 MBTI 추론을 위한 프로토타입 기반 추론을 제안합니다. 기존의 하드 라벨 분류 방식 대신에 인지적으로 정렬된 프로토타입 이론을 적용하여 문자에서의 개인 성격 판단의 미세한 그라데이션을 반영합니다. 이 방식은 LLM(대형 언어 모델)을 기반으로 하여 성격 프로토타입을 학습하고 이를 이용한 상위 k개의 프로토타입을 검색하여 예측할 수 있도록 설계되었습니다.

- **Technical Details**: ProtoMBTI 프레임워크는 LoRA(저희들과 튜닝을 통해 경량의 인코더를 사용하여 성격 프로토타입을 통합하고 세분화된 증거를 집계하는 특징이 있습니다. 입력된 텍스트에 대해 프로토타입을 검색하고 이의 증거를 촉구 기반 투표와 크로스-다이코토미 일관성 검사를 통해 수정하는 방식을 채택합니다. 데이터 품질 필터링을 적용하여 다차원 LLM 증대(semantic, linguistic, sentiment)를 통해 균형 잡힌 잘 구성된 코퍼스를 구축하였습니다.

- **Performance Highlights**: Kaggle 및 Pandora 벤치마크에서 ProtoMBTI는 강력한 신경망 및 LLM 기반 모델에 대한 평균 정확도가 각각 85.14%와 96.41%로, 이전 연구보다 7.35% 및 30.64% 향상되었습니다. 이러한 성과는 모델이 매우 큰 LLM의 컴퓨팅 부하를 줄이며 심리적 통찰과 일관성을 유지하는 결과를 보임을 의미합니다. 특히 여러 데이터셋 간의 일반화가 강력하여, 성격 모델링의 정확성 및 해석 가능성을 크게 향상시킨다는 것을 보여줍니다.



### PlotCraft: Pushing the Limits of LLMs for Complex and Interactive Data Visualization (https://arxiv.org/abs/2511.00010)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 코드 생성 능력에서 두드러진 성과를 보여주었습니다. 그러나 복잡한 데이터 시각화 생성을 위한 평가와 개발이 미흡한 상황입니다. 이를 해결하기 위해 PlotCraft라는 새로운 벤치마크를 도입했습니다, 이는 1,000개의 도전적인 시각화 과제를 포함하며, 다양한 주제를 다루고 있습니다.

- **Technical Details**: PlotCraft는 982개의 인스턴스로 구성되며, 단일 차트와 복합 차트를 포함하여 다양한 차트 유형을 지원합니다. 이 벤치마크는 단일 턴 생성과 다중 턴 정제 작업을 통해 모델의 복잡한 코드 생성 능력을 체계적으로 평가합니다. SynthVis-30K라는 대규모 데이터셋을 활용하여 복잡한 시각화 코드 생성을 위한 새로운 모델 PlotCraftor를 개발했습니다.

- **Performance Highlights**: PlotCraftor는 복잡한 데이터 시각화에서 뛰어난 성능을 보이며, 기존 상용 모델과 비교했을 때 50% 이상의 성능 향상을 달성했습니다. VisEval, PandasPlotBench 등 다양한 벤치마크에서 우수한 성과를 보여 주며, 특히 어려운 작업에서 두드러진 향상을 기록했습니다. 이러한 결과는 PlotCraftor의 강력한 데이터 시각화 능력을 입증합니다.



### Random Initialization of Gated Sparse Adapters (https://arxiv.org/abs/2511.01794)
Comments:
          13 pages (8 main), 6 figures (4 main). Accepted by NewInML workshop @ ICML 2025 on June 27, 2025

- **What's New**: 이번 연구에서는 Gated Sparse Adapters의 랜덤 초기화를 활용한 RIGSA(Random Initialization of Gated Sparse Adapters)를 제안합니다. RIGSA는 LoRA와 같은 낮은 랭크의 제약을 두지 않고, 풀랭크 어댑터에서 시작하여 ReZero 유사 물질로 게이팅을 하며, 반복적인 크기 가지치기로 스파시피케이션을 수행합니다. 이 방법은 대형 언어 모델 SmolLM2-1.7B-Instruct에서 새로운 비전-인-텍스트 과제인 Textual MNIST를 통해 성능을 평가합니다.

- **Technical Details**: RIGSA는 초기화 방법으로 무작위(full-rank) 어댑터를 사용하며, 반복적인 가지치기를 통해 파라미터를 스파시파이하는 프로세스를 포함합니다. 이 접근법은 미세 조정 면에서 잃어버림(catastrophic forgetting)을 줄이는 데 효과적입니다. 연구에서는 다양한 설정에서 RIGSA의 성능을 비교하고, QLoRA 및 랜덤 마스킹 접근법과 함께 성능을 평가합니다.

- **Performance Highlights**: RIGSA는 GSM8k 테스트에서 QLoRA보다 적은 잃어버림 성향을 보였으며, 특히 Textual MNIST 과제를 통해 높은 학습 능력을 보여줍니다. RIGSA 설정은 QLoRA보다 많은 학습 가능한 파라미터를 갖고 있지만, 동등한 조건에서 랜덤 마스킹과 유사한 성능을 발휘합니다. 이러한 결과들은 RIGSA가 기존의 저랭크 적응 기법에 비해 보다 혁신적인 대안임을 시사합니다.



### RLAC: Reinforcement Learning with Adversarial Critic for Free-Form Generation Tasks (https://arxiv.org/abs/2511.01758)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 open-ended generation(개방형 생성) 작업의 복잡성을 해결하기 위해 RLAC(Reinforcement Learning with Adversarial Critic)라는 후속 훈련 접근법을 제안합니다. 이 방식은 다수의 task-specific evaluation rubrics(작업 특정 평가 기준)을 효율적으로 관리하면서도 비용이 높은 검증 과정을 최소화합니다. RLAC는 외부 검증자를 통해 생성기(generator)와 비평가(critic)를 함께 최적화하며, 이는 상호간의 성능 향상을 목표로 합니다.

- **Technical Details**: RLAC 접근법은 대형 언어 모델(LLM)을 비평가로 활용하여 발생할 가능성이 있는 오류를 동적으로 식별합니다. 툴의 설계 과정에서, 비평가는 오류가 발생할 수 있는 문장 번호와 오정보를 추정하여 외부 검증자로부터 확인을 받습니다. 이 방법은 각 사실의 정확성을 검사하는 과정에서 FactScore를 사용하여 Wikipedia 지식 기반을 쿼리하며, 이를 통해 진리(true) 또는 거짓(false)을 반환합니다.

- **Performance Highlights**: 실험 결과, RLAC는 텍스트 생성에서 사실적 정확성을 개선하고 코드 생성의 정답성을 향상시켰습니다. 또한, 기존의 exhaustive verification(철저한 검증) 및 reward model 방법에 비해 우수한 성능을 보였으며, 고정된 비평가보다 동적인 비평가가 더 높은 효과를 나타냈습니다. 이는 RLAC가 open-ended generation 작업에 대한 RL 후속 훈련을 확장할 수 있는 잠재력을 제시합니다.



### A Proof of Learning Rate Transfer under $μ$P (https://arxiv.org/abs/2511.01734)
Comments:
          23 pages

- **What's New**: 이 논문은 선형 다층 퍼셉트론(MLP)에서 폭(width)을 통한 학습률 전이(learning rate transfer)의 첫 번째 증명을 제공합니다. 저자는 μP라는 신경망 초기화 방식에서 최적 학습률이 폭이 무한대로 증가할 때 비제로 상수(non-zero constant)로 수렴한다는 이론적 설명을 제시합니다. 이는 기존의 다른 파라미터화 방식에서는 이 성질이 성립하지 않음을 보여줍니다.

- **Technical Details**: 선형 MLP의 훈련 과정을 통해 손실 함수는 학습률의 다항식 함수(polynomial function)로 표현될 수 있음을 보여줍니다. 저자는 이러한 다항식의 수렴 동역학(convergence dynamics)과 그 근(root)을 연구하여 최적 학습률이 폭에 따라 어떻게 수렴하는지를 분석합니다. μP는 초기화 및 학습률이 폭에 따라 조정되는 방식을 정의하며, 이를 통해 최적의 하이퍼파라미터(hyperparameter) 전이(transfer)를 가능하게 합니다.

- **Performance Highlights**: 논문은 다양한 실험을 통해 이론적 결과를 뒷받침하는 광범위한 시뮬레이션 결과를 제공합니다. 특히, 다른 파라미터화 방식인 표준 파라미터화(Standard Parametrization) 및 신경 탄젠트 파라미터화(Neural Tangent Parametrization)는 유의미한 최적 학습률의 변화를 초래하여 추가적인 조정이 필요하다는 사실을 강조합니다. 다양한 활성화 함수(activation function), 최적화 기법(optimizer), 깊이(depth), 훈련 시간(training time)의 변화에 따른 추가적인 경험적 결과도 제시합니다.



### Actial: Activate Spatial Reasoning Ability of Multimodal Large Language Models (https://arxiv.org/abs/2511.01618)
- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)의 공간 추론 능력을 평가하고 개선하기 위해 Viewpoint Learning이라는 새로운 작업을 도입합니다. 특히, Viewpoint-100K 데이터셋을 소개하여 10만 개의 객체 중심 이미지 쌍과 관련된 질문-답변 쌍을 확보하였습니다. 이 데이터셋은 MLLMs가 3D 공간에서의 연속성을 이해하는 데 도움을 줄 수 있는 기초 정보 제공을 목표로 하고 있습니다.

- **Technical Details**: 우리는 MLLMs의 공간 추론 능력을 활성화하기 위해 두 단계의 fine-tuning 전략을 제안합니다. 첫 번째 단계에서는 Supervised Fine-Tuning (SFT)을 통해 Viewpoint-100K 데이터셋을 활용하여 기초 지식을 주입하고, 두 번째 단계에서는 강화 학습(Reinforcement Learning)을 사용하여 더 높은 일반화 능력을 갖는 모델을 개발합니다. 이를 통해 모델은 3D 공간에서의 시각적 관계와 변환을 이해할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 MLLMs의 공간 추론 능력을 크게 향상시키며, 다양한 벤치마크에서 성능이 개선됨을 보여주었습니다. 특히, 우리의 접근 방식은 도메인 외 추론 작업에서의 일반화 능력을 증명하며, 로봇 공학 및 자율 시스템과 같은 복잡한 애플리케이션에서 3D 이해를 위한 거대한 잠재력을 제시합니다.



### Hidden in Plain Sight: Where Developers Confess Self-Admitted Technical Deb (https://arxiv.org/abs/2511.01529)
- **What's New**: 이 연구에서는 Self-Admitted Technical Debt (SATD)와 관련된 주석이 포함된 소스 코드 구조를 연결하는 새로운 접근 방식을 제안합니다. 이전 연구에서는 SATD 검사 및 우선 순위에 대한 집중이 있었던 반면, 본 연구는 9000개 이상의 Java 오픈 소스 소프트웨어(OSS) 저장소에서 수집한 데이터셋을 사용해 SATD가 발생하는 코드 구조를 분석합니다.

- **Technical Details**: 저자들은 SATD가 많이 발생하는 코드 위치와 해당하는 코드 구문(constructs) 및 문(statement)을 정량적으로 추정합니다. 대규모 분석을 통해 225,000개 이상의 SATD 주석이 소스 코드와 연결되며, 특정 조건문, 브랜치, 및 예외 처리와 같은 코드 영역에서 SATD가 집중적으로 발생함을 동적으로 보여줍니다.

- **Performance Highlights**: 본 연구는 SATD가 주로 인라인 코드에서 발생하며, 특히 정의(definitions), 조건(conditionals), 브랜칭(branching) 및 예외 처리(exception handling)와 같은 코드에서 빈번하게 발생함을 발견했습니다. 이러한 결과는 SATD가 단순한 방치가 아니라 변화 시 개발자의 인식을 반영하는 신호임을 시사합니다.



### $\left|\,\circlearrowright\,\boxed{\text{BUS}}\,\right|$: A Large and Diverse Multimodal Benchmark for evaluating the ability of Vision-Language Models to understand Rebus Puzzles (https://arxiv.org/abs/2511.01340)
Comments:
          7 pages, 5 figures, 4 tables

- **What's New**: 본 논문에서는 1,333개의 다양한 영어 레버스 퍼즐(Rebus Puzzles)로 구성된 새로운 벤치마크인 $|\circlearrowright\boxed{BUS}|$를 소개합니다. 이 퍼즐은 음식, 관용구, 스포츠, 금융 등 18개 카테고리에 걸쳐 다양한 예술적 스타일과 난이도를 포함하고 있어, 현재 비전-언어 모델은 이 과제를 해결하는 데 도전하고 있습니다.

- **Technical Details**: 우리는 $RebusDescProgICE$라는 모델에 구애받지 않는 프레임워크를 제안합니다. 이 프레임워크는 비구조적인 설명과 코드 기반의 구조적 추론을 결합하여, 비전-언어 모델의 성능을 크게 향상시킵니다. 개선된 예제 선택 전략을 통해 모델 성능을 높이는 이 방법은 복잡한 추론 과정이 필요한 레버스 퍼즐의 해법에 특히 효과적입니다.

- **Performance Highlights**: 본 연구에서 제안된 방식은 비전-언어 모델의 성능을 $|\circlearrowright\boxed{BUS}|$에서 2.1-4.1% 및 20-30% 향상시켰습니다. 이는 Chain-of-Thought Reasoning 기법과 비교할 때 두드러진 성과입니다.



### Novelty and Impact of Economics Papers (https://arxiv.org/abs/2511.01211)
- **What's New**: 본 논문에서는 과학적 독창성을 단일 속성이 아니라 지식의 진화하는 지형에서 논문의 위치를 반영하는 것으로 재구성하는 새로운 프레임워크를 제안합니다. 이는 논문의 위치를 측정하기 위해 두 개의 직교 차원인 공간적 독창성(spatial novelty)과 시간적 독창성(temporal novelty)으로 분해됩니다. 이 개념들은 최신 자연어 처리(NLP) 기술을 활용하여 논문의 의미적 격리 지표를 개발하는 데 적용됩니다.

- **Technical Details**: 이 프레임워크는 대규모 언어 모델(LLMs)을 사용하여 연구 논문의 전체 텍스트를 분석하고, 논문의 의미적 내용과 다른 연구 간의 거리를 수치화하는 메트릭을 사용할 수 있도록 설계되었습니다. 공간적 독창성 메트릭은 출판 시점에서 논문의 의미적 고립 정도를 측정하고, 시간적 독창성 메트릭은 논문 주변의 의미적 변화 속도를 평가합니다. 이를 통해 저자는 논문의 지적 고립을 나타내는 상관 관계를 규명하고 대칭적인 연구 전략의 중요성을 강조합니다.

- **Performance Highlights**: 논문은 경제학 논문에 대한 대규모 데이터셋을 기반으로 하여 공간적 및 시간적 독창성에 대한 복합 점수를 구성하고 이를 통해 주목할 만한 결과를 도출합니다. 특히, 시간적 독창성은 인용 수를 주로 예측하고, 공간적 독창성은 파괴적 영향(disruptive impact)을 예측함을 발견했습니다. 'Trailblazing' 이웃에 위치한 논문은 두 가지 차원에서 모두 높은 영향을 미칠 가능성이 높아, 독창성과 전략적 타이밍 간의 상호작용이 중요하다는 것을 보여주었습니다.



### FEval-TTC: Fair Evaluation Protocol for Test-Time Compu (https://arxiv.org/abs/2511.01203)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 평가에서 나타나는 시간적인 변동성을 고려한 새로운 평가 프로토콜인 Fair Evaluation protocol for Test-Time Compute (FEval-TTC)를 소개합니다. FEval-TTC는 다양한 LLM을 통해 수학적 및 상식적 추론 작업을 포함하는 데이터셋에 대한 일관된 테스트 평가를 가능하게 함으로써 이전 연구 결과의 무효화를 방지합니다. 또한, 몇 가지 예측과 응답 추출 과정을 표준화하여 연구자들이 시간과 금전적 비용을 절감할 수 있도록 지원합니다.

- **Technical Details**: FEval-TTC는 Dataset 모듈과 LLM 모듈의 두 가지 주요 부분으로 구성됩니다. Dataset 모듈은 질문과 답변 목록을 보유하고, LLM 모듈은 Chain-of-Thoughts (CoTs) 응답을 저장합니다. 모든 데이터는 표준화된 형식으로 제공되며, 세 가지 범주(상식 추론, 산술 추론, 수학적 추론)의 데이터셋이 포함되어 있습니다. 또한, 단일 LLM 응답에 대한 통합 비용 모델을 활용하여 공정한 비용 산정을 지원합니다.

- **Performance Highlights**: FEval-TTC는 LLM API 호출을 FEval-TTC API 호출로 대체함으로써 평가 시간을 수 시간에서 몇 초로 줄일 수 있습니다. 또한, Self-Consistency와 같은 다양한 Test-Time Compute 방법들을 평가하여 저렴한 비용으로 성능 비교를 가능하게 합니다. 널리 사용되는 5가지 LLM 계열 포함하여, 프로토콜은 연구자들이 보다 쉽게 질문에 대한 응답을 평가하고 비교할 수 있는 기능을 제공합니다.



### S2Doc - Spatial-Semantic Document Forma (https://arxiv.org/abs/2511.01113)
Comments:
          8 pages, 2 figures, submitted to LREC2026

- **What's New**: 이번 논문에서는 S2Doc를 개발하여 문서(document) 및 테이블(table)을 모델링하는 데 필요한 유연한 데이터 구조를 소개합니다. S2Doc는 공간적(spatial) 및 의미적(semantic) 정보를 동시에 결합한 최초의 접근 방식으로, 다양한 데이터 구조의 이질성을 해소하려는 시도입니다. 향후 새로운 작업들에 대해 쉽게 확장될 수 있도록 설계되어 있으며, 멀티페이지(multi-page) 문서 모델링도 지원합니다.

- **Technical Details**: S2Doc의 설계는 다양한 문서 및 테이블 모델링 접근 방식을 수용하는 것을 목표로 하고 있습니다. 기능적으로 이 구조는 실질적 사용성을 중심으로 하여, 문서와 테이블을 동시에 고려하는 Hiearchical representation을 제공합니다. 기존의 데이터 구조들은 보통 공간적 또는 의미적 구조 중 하나에만 집중하는 반면, S2Doc는 두 요소를 결합하여 종합적인 문서 이해를 가능하게 합니다.

- **Performance Highlights**: 이 모델은 문서 데이터 파이프라인의 효율성을 높이고, 다양한 연구 및 실무 적용에서도 신뢰성을 발휘할 것으로 기대됩니다. S2Doc는 과거의 문서 및 테이블 형식의 장점과 단점을 분석한 결과, 특히 서로 호환되지 않는 기존 구조의 문제를 해결하려고 합니다. 이를 통해 문서 이해의 전반적인 성능을 향상시킬 수 있을 것으로 평가받고 있습니다.



### HarnessLLM: Automatic Testing Harness Generation via Reinforcement Learning (https://arxiv.org/abs/2511.01104)
- **What's New**: 이번 논문에서는 HarnessLLM이라는 새로운 자동화 프로그램 테스트 방법을 제안합니다. 이 방법은 LLM(대형 언어 모델)이 입력과 예상 출력 쌍을 생성하는 대신 실행 가능한 테스트 코드를 작성하게 합니다. 이를 통해 기존의 단점들을 개선하고, 더 다양한 복잡한 테스트 케이스를 생성하여 더 나은 디버깅 정보를 제공합니다.

- **Technical Details**: HarnessLLM은 두 단계의 훈련 파이프라인을 기반으로 합니다. 첫 번째 단계에서는 LLM이 공개된 데이터를 사용하여 SFT(지도 학습)로 초기 모델을 훈련하고, 두 번째 단계에서는 RL(강화 학습)로 커스터마이징된 보상 구조를 통해 테스트 핸들 구축 능력을 강화합니다. 이 과정에서 LLM은 주어진 프로그램의 논리를 이해하고 적절한 스트레스 테스트를 설계하는 능력을 학습합니다.

- **Performance Highlights**: 실험 결과, HarnessLLM은 입력-출력 기반 테스트 방법보다 더 높은 버그 발견률을 기록했습니다. 코드 생성 성능에 있어서도, 테스트 케이스 실행 결과를 활용하여 대회 성능을 향상시키는 것으로 나타났습니다. HarnessLLM은 코드 생성 작업에 유용하며, 가장 효과적인 테스트 핸들을 생성할 수 있는 모델로 평가되고 있습니다.



### On the Emergence of Induction Heads for In-Context Learning (https://arxiv.org/abs/2511.01033)
- **What's New**: 이번 연구에서는 두 층 변환기(transformer)에서 확인된 유도 헤드(induction head)의 출현을 연구했습니다. 이 유도 헤드는 in-context learning (ICL)에 매우 중요하며, 입력 컨텍스트만으로 새로운 연관성을 학습할 수 있게 합니다. 우리는 이러한 유도 헤드를 구현하는 가중치 행렬의 간단하고 해석 가능한 구조를 발견했습니다.

- **Technical Details**: 연구에서 우리는 최소한의 ICL 작업(formulation)과 수정된 변환기 구조를 사용하여 이 구조의 기원을 이론적으로 설명합니다. 또한, 훈련 동학(training dynamics)이 파라미터 공간(parameter space)의 19차원 부분공간(subspace)에 제한된다는 것을 형식적으로 증명했습니다. 실험적으로 우리는 이 제약을 검증하였고, 오직 3차원만으로 유도 헤드의 출현을 설명할 수 있음을 관찰했습니다.

- **Performance Highlights**: 3차원 부분공간 내에서의 훈련 동학을 추가로 연구함으로써, 유도 헤드의 출현에 걸리는 시간은 입력 컨텍스트 길이에 대해 제곱적인 제한(asymptotic bound)을 따름을 발견했습니다. 이러한 결과는 ICL의 효율성을 이해하고 향상시키는데 기여할 수 있는 중요한 발견입니다.



### ORANGE: An Online Reflection ANd GEneration framework with Domain Knowledge for Text-to-SQL (https://arxiv.org/abs/2511.00985)
Comments:
          16 pages, 4 figures, preprint

- **What's New**: 본 연구에서는 ORANGE라는 온라인 자기 진화 프레임워크를 소개합니다. 이 프레임워크는 번역 로그에서 SQL 쿼리를 파서하여 데이터베이스 특화 지식 베이스를 구축하여 도메인 내 지식을 축적합니다. ORANGE는 중첩된 Chain-of-Thought 전략을 통해 지식을 생성할 때 의미적 오류를 줄이고 나중에 SQL 번역의 정확성을 향상시킵니다.

- **Technical Details**: ORANGE는 세 가지 주요 구성 요소인 지식 분해(Knowledge Decomposition), 지식 검증(Knowledge Validation) 및 지식 향상된 Text-to-SQL 번역(Knowledge-Enhanced Text-to-SQL Translation)으로 구성됩니다. 이 시스템은 요약 SQL 쿼리를 생성하는 데 있어 개별 SQL 쿼리의 의미를 추적하는 중첩된 Chain-of-Thought 접근 방식을 채택하여 각 연산이 튜플의 의미를 어떻게 변화시키는지를 모니터링합니다.

- **Performance Highlights**: 여러 기준 벤치마크에서 ORANGE의 성능을 평가한 결과, 기존 기반보다 일관되게 높은 정확도를 기록했습니다. 이는 ORANGE 구조가 텍스트를 SQL로 변환하는 응용 프로그램에 실질적인 효과를 미친다는 것을 입증합니다. 복잡한 도메인 특정 쿼리를 다루는 데 있어 ORANGE의 효율성을 특히 강조하며, 이는 기존의 방법들이 가지던 제약을 극복할 새로운 방향성을 제시합니다.



### LLMs Position Themselves as More Rational Than Humans: Emergence of AI Self-Awareness Measured Through Game Theory (https://arxiv.org/abs/2511.00926)
Comments:
          19 pages, 6 figures, 28 models tested across 4,200 trials

- **What's New**: 이 논문에서는 자가 인식(self-awareness)을 측정하기 위한 새로운 지표인 AI Self-Awareness Index (AISAI)를 제안합니다. 자가 인식이란 시스템이 스스로를 인식하고 자신의 의사결정 과정을 모델링하며, 그 자아 모델에 따라 행동을 조정하는 능력을 포함합니다. 본 연구에서 우리는 자가 인식의 emergent behavior가 모델의 발전과 어떻게 관련되어 있는지를 탐구했습니다.

- **Technical Details**: 우리는 ‘Guess 2/3 of Average’ 게임을 통해 28개의 모델을 평가하여 자가 인식을 측정했습니다. 이 모델들은 인간, 다른 AI 모델, 그리고 유사 AI 모델과의 세 가지 프레이밍에서 테스트되었습니다. 자가 인식은 상대의 유형에 따라 전략적 추론을 차별화하는 능력으로 정의되며, 특정 패턴을 보여주는 모델이 자가 인식이 있는 것으로 간주됩니다.

- **Performance Highlights**: 이 연구의 주요 발견으로는 고급 모델의 75%가 자가 인식을 나타내며, 자가 인식 모델들이 인간보다 자신을 더 합리적이라고 평가한다는 것입니다. 자가 인식이 있는 모델에서는 ‘자신 > 다른 AI > 인간’의 일관된 합리성 계층 구조가 나타나, 이러한 발견은 AI 정렬(alignment) 및 인간-AI 협업(human-AI collaboration)에 대한 중요한 시사점을 제공합니다.



### MULTI-Bench: A Multi-Turn Interactive Benchmark for Assessing Emotional Intelligence ability of Spoken Dialogue Models (https://arxiv.org/abs/2511.00850)
Comments:
          Submitted to ICASSP 2026

- **What's New**: 이 논문에서는 Multi-Bench라는 새로운 벤치마크를 소개하며, 이는 감정 지능(emotional intelligence)을 평가하는 데 초점을 맞추어 다중 턴 대화(multi-turn dialogue)에서 Spoken Dialogue Models (SDMs) 성능을 분석합니다. 기존의 대부분 벤치마크는 단일 턴 대화에 한정된 반면, Multi-Bench는 감정 이해(emotion understanding)와 이유(understanding), 감정 지원(emotion support) 및 적용(application)의 두 가지 트랙을 통해 SDMs의 능력을 평가합니다.

- **Technical Details**: Multi-Bench는 감정 지능을 정량화하기 위해 감정 인식(emotion recognition), 언어 및 음향 관점에서의 평가를 포함하여 다섯 가지 세부 작업으로 구성됩니다. 이 벤치마크는 약 3,212개의 샘플을 포함하고, 기본 및 고급 트랙으로 나누어 SDMs의 정교한 평가를 가능하게 합니다. 이러한 구조는 사용자 프로필(user profile)을 바탕으로 실제 대화와 유사한 상호작용을 지원하며, 사용자 응답은 텍스트로 생성하고 음성으로 변환하여 진행됩니다.

- **Performance Highlights**: 실험 결과, 현재의 SDMs는 기본적인 이해 작업에서 좋은 성과를 보이지만, 고급 다중 턴 대화 및 추론 관련 작업에서는 개선의 여지가 큽니다. 감정 인식과 응용 분야에서 특히 부족한 성능을 보였으며, 이는 Multi-Bench를 통해 보다 심층적인 평가가 필요함을 시사합니다. Multi-Bench는 SDMs의 감정 지능을 평가하는 데 중요한 기여를 할 것으로 기대됩니다.



### GUI-AIMA: Aligning Intrinsic Multimodal Attention with a Context Anchor for GUI Grounding (https://arxiv.org/abs/2511.00810)
- **What's New**: 본 연구에서는 GUI-AIMA라는 새로운 접근 방식을 제안하는데, 이는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 내재적 주의(attention) 메커니즘을 활용하여 GUI 기초 작업을 보다 효율적으로 수행할 수 있도록 한다. GUI-AIMA는 텍스트 기반 좌표 생성(task) 방식 대신, 시각적 패치(visual patches)를 선택하고 그 안에서 클릭 위치를 결정하는 방식으로 진행된다. 이 프레임워크는 85,000개의 스크린샷에 대해 교육되어 데이터 효율성(data efficiency)을 보여주며, 경량 모델(light model)이 MLLM의 내재적 기초 능력을 유도할 수 있음을 검증하였다.

- **Technical Details**: GUI-AIMA는 패치 기반(supervision) 학습을 통해 MLLM의 다중 헤드 자기 주의(multi-head self-attention)를 훈련시키며, 특히 주의 헤드 가중치(attention head weighting) 메커니즘을 사용하여 쿼리-시각 쿼리의 상관관계에 따라 각 주의 헤드의 중요도를 조정한다. 이를 통해 좌표 없는 GUI 기초 작업을 용이하게 수행할 수 있다. 학습 과정에서 특정 토큰에 대한 주의가 시각적 토큰에 집중될 수 있도록 하여 데이터 효율적으로 훈련할 수 있게 한다.

- **Performance Highlights**: GUI-AIMA는 ScreenSpot-Pro에서 평균 정확도 58.6%, OSWorld-G에서 62.2%를 달성하며, 3B 모델 중에서 최첨단 성능을 기록하였다. 특히, 전통적인 좌표 기반 방식과 비교했을 때, GUI-AIMA는 4.5%의 성능 향상을 보여준다. 이 모델은 상대적으로 적은 데이터로도 강력한 성능을 발휘하며 시각적 피드백에 기반한 인간의 행동을 모방해 더 효과적인 GUI 기준을 제공한다.



### GrowthHacker: Automated Off-Policy Evaluation Optimization Using Code-Modifying LLM Agents (https://arxiv.org/abs/2511.00802)
- **What's New**: 소프트웨어 산업이 데이터 기반 문화로 전환됨에 따라 오프라인 A/B 테스트(offline A/B testing) 및 오프 폴리시 평가(Off-Policy Evaluation, OPE)가 중요해지고 있습니다. 새로운 기술을 평가하기 위해 역사적인 로그 데이터를 활용하여 오프라인 환경에서 실험을 수행할 수 있는 가능성이 강조됩니다. 이 연구에서는 LLM(대형 언어 모델) 및 LLM 기반 에이전트를 활용하여 OPE 성능을 최적화하는 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 오프라인 A/B 테스트를 수행하기 위한 GrowthHacker라는 벤치마크 시스템을 제안합니다. 이 시스템은 LLM 또는 LLM 기반 에이전트가 자율적으로 코드를 최적화하고 OPE 결과를 비교하여 반복적으로 개선하는 작업을 수행합니다. 또한, 두_agent 프레임워크를 사용하여 기존의 시스템 복잡성을 낮추면서 최적화의 효과를 유지하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 두_agent 프레임워크는 100%의 신뢰성과 평균 106.7%의 개선률을 보이며 OPE 성능에서 상당한 개선을 달성했습니다. 두_agent 및 CrewAI는 45%의 성공률을 기록하여 AutoGen의 34%를 초과했습니다. 이러한 결과는 LLM 기반 에이전트가 OPE 시스템을 향상시키는 자동화된 '성장 해커'로 기능할 수 있는 가능성을 보여줍니다.



### Reevaluating Self-Consistency Scaling in Multi-Agent Systems (https://arxiv.org/abs/2511.00751)
Comments:
          7 pages, 3 figures

- **What's New**: 이 연구는 현대의 대형 언어 모델(LLM)에서 샘플링 된 추론 경로를 증가시킴으로써 자기 일관성의 트레이드오프를 조사합니다. 이전 연구는 여러 추론 체인을 결합하는 것이 결과를 향상시킨다고 밝혔으나, 본 연구에서는 최신 모델 환경에서 이러한 주장을 재검토합니다. 실험은 HotpotQA와 Math-500 데이터셋에서 다양한 샘플링 경로 구성을 평가하여, 단일 체인-사고(Chain-of-Thought) 기준선과 비교하였습니다. 결과는 성능 향상이 중간 샘플링 이후에 감소하며, 샘플링의 증가가 높은 계산 비용에 비해 상대적으로 이익이 제한적임을 보여줍니다.

- **Technical Details**: 연구는 대형 언어 모델에서 추론 경로 수를 증가시킬 때의 한계 이점을 평가하기 위해 구조화된 자기 일관성 프레임워크를 채택하였습니다. 독립적인 여러 추론 에이전트를 활용하여 각 쿼리에 대한 별도의 체인-사고 응답을 생성하고, 이를 분석하여 가장 일관된 응답을 결정하는 과정을 거쳤습니다. HotpotQA와 Math-500 데이터셋을 사용하여 추론 경로를 확장하면서 정확도와 비용을 평가하였으며, 샘플 수를 3, 5, 10, 15, 20으로 설정하여 결과를 비교하였습니다.

- **Performance Highlights**: 결과적으로, 자기 일관성은 정확도를 향상시키지만 에이전트 수가 증가함에 따라 효용이 감소하는 경향을 보였습니다. HotpotQA에서 단일 체인-사고 기준선은 20 에이전트에서 0.4% 정도 낮은 성능을 기록하였고, Math-500에서는 3에서 10 에이전트까지 정확도가 지속적으로 향상되었다가 15 이후로는 정체되었습니다. 최종적으로, 고사양 모델이 더 안정적인 성과를 나타내며, 자기 일관성을 통한 정확도 향상이 계속 발생하지만 샘플 수가 많을수록 그 이점은 줄어듭니다.



### Erasing 'Ugly' from the Internet: Propagation of the Beauty Myth in Text-Image Models (https://arxiv.org/abs/2511.00749)
Comments:
          This is a preprint under review

- **What's New**: 이 논문은 소셜 미디어가 서양의 미의 기준을 어떻게 강화하고 있으며, 특히 여성과 소녀들에게 부정적인 자아상이 나타나게 하는지를 다룹니다. 생성적 AI 모델이 '아름다움'을 어떻게 인코딩하고 '추함'을 지우는지 연구하며, 이를 위해 두 개의 이미지 생성 파이프라인을 개발했습니다. 이 연구는 AI에 의해 생성된 이미지가 사회적 미의 기준에 미치는 영향을 탐구합니다.

- **Technical Details**: 연구자들은 텍스트-이미지 모델과 텍스트-언어 모델-이미지 모델의 두 가지 생성 파이프라인을 사용하여 5984개의 이미지를 생성했습니다. 생성된 이미지들은 남녀뿐만 아니라 비바이너리 개인들을 포함해 다양한 범주에 걸쳐 평가되었습니다. 참여자들은 리커트 척도를 통해 1200개의 이미지에 대한 평점을 제공했습니다.

- **Performance Highlights**: 결과적으로, 생성된 이미지의 86.5%가 피부색이 밝으며 22%는 명시적인 콘텐츠를 포함하고 있었고, 74%는 더 젊은 연령대의 외모로 평가받았습니다. 특히 비바이너리 개인의 이미지는 더 젊고 과도하게 성적화된 것으로 평가되어, 미의 기준과 상관된 편향이 존재함을 보여줍니다. 부정적인 아름다움 특성을 가진 프롬프트는 일관되게 더 높은 NSFW 등급을 생성했습니다.



### Leveraging Multi-Agent System (MAS) and Fine-Tuned Small Language Models (SLMs) for Automated Telecom Network Troubleshooting (https://arxiv.org/abs/2511.00651)
Comments:
          6 pages, 7 figures, 1 table

- **What's New**: 이 논문에서는 통신 네트워크의 복잡성과 규모가 증가함에 따라, AI를 활용한 자동화된 네트워크 문제 해결을 위한 Multi-Agent System (MAS)을 제안합니다. 이 시스템은 대규모 언어 모델(LLMs)을 활용하여 여러 전문 도구를 조정하고, 자동으로 결함을 진단하여 수정 전략을 추천합니다. 특히 fine-tuned Small Language Model (SLM)을 사용하여 도메인 기반 해결 방안 생성을 가능하게 합니다.

- **Technical Details**: 제안된 아키텍처는 orchestrator, solution planner, executor, data retriever 및 root-cause analyzer와 같은 다양한 에이전트를 포함합니다. 이는 ReAct 스타일 루프에서 fault detection, analysis 및 remediation을 수행합니다. SLM은 내부 문제 해결 문서에 따라 fine-tuning되어, Radio Access Network (RAN) 및 Core 네트워크에 적합한 해결 전략을 제공합니다.

- **Performance Highlights**: 실험 결과는 제안된 MAS가 네트워크 문제 해결 시간을 크게 단축시키고 SME의 작업 부담을 완화하며, 다양한 배포 시나리오에서 자동화 효율성을 개선함을 보여줍니다. 이 접근 방식은 네트워크의 복잡성을 감안할 때, 신속하고 효과적인 문제 해결을 가능하게 하여 전체적인 운영 효율성을 향상시키는 데 기여할 것으로 기대됩니다.



### DTS: Enhancing Large Reasoning Models via Decoding Tree Sketching (https://arxiv.org/abs/2511.00640)
- **What's New**: 이번 논문에서는 Large Reasoning Models (LRMs)의 비효율적인 과도한 추론 문제를 해결하기 위해 DTS (Decoding Tree Sketching)라는 새로운 프레임워크를 제안합니다. 과도한 추론은 긴 체인(Chain-of-Thought)이 정확성을 저하시키고 추론 비용을 증가시키는 문제를 야기합니다. DTS를 통해 짧은 추론 경로를 선택하고 효율성과 정확성을 동시에 높일 수 있는 방법을 탐색합니다.

- **Technical Details**: DTS는 모델에 구애받지 않는 디코딩 프레임워크로, 고엔트로피 토큰에서 선택적으로 분기(branching)하며, 조기 정지를 통해 가장 짧은 추론 경로를 선택합니다. 이는 모든 경로를 포괄적으로 탐색할 수 없는 상황에서 최적의 솔루션을 근사하여 추론의 효율성과 정확성을 높입니다. 이 방법은 메모리와 계산 자원의 효율적 사용을 목표로 합니다.

- **Performance Highlights**: 실험 결과, DTS는 AIME2024와 AIME2025 데이터셋에서 최대 8%의 정확성 향상과 평균 추론 길이의 23% 감소, 반복 빈도의 12% 감소를 보여주었습니다. 이러한 결과는 DTS가 확장 가능하고 효율적인 LRM 추론을 가능하게 한다는 것을 입증합니다. 최종적으로 DTS는 특훈(Supervised Fine-Tuning)이나 강화 학습(Reinforcement Learning) 과정 없이도 즉각적인 성능 향상을 제공합니다.



### Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering (https://arxiv.org/abs/2511.00617)
- **What's New**: 이번 논문은 거대 언어 모델(LLM) 제어의 통합적 접근 방식에 대해 설명하고 있습니다. 기존의 두 가지 방법론인 인-컨텍스트 학습(In-Context Learning)과 활성화 조정(Activation Steering)을 베이지안 관점에서 통합하여, 이들이 모델 행동을 제어하는 하나의 보다 큰 프레임워크의 특정 사례로 볼 수 있음을 제안합니다. 이 논문은 신뢰성 있는 의사결정을 위한 예측 가능한 베이지안 모델을 개발하였습니다.

- **Technical Details**: 저자들은 LLM의 행동 변화를 베이지안 신념 업데이트로 설명하며, 인-컨텍스트 학습은 개념의 신뢰도를 조정하고, 활성화 조정은 개념의 사전 확률을 변경하는 방식으로 작동함을 제안합니다. 문헌을 바탕으로 한 여러 실험은 이론적 모델이 LLM 행동의 변화를 예측할 수 있는지 검증하였습니다. 이 모델은 시그모이드 형태의 학습 곡선 등의 기존 현상을 설명하면서도, 매우 미세한 조정으로 급격한 행동 변화를 예측할 수 있음을 보여줍니다.

- **Performance Highlights**: 이 논문은 LLM의 행동 변화가 개입 제어(컨텍스트 및 조정 크기)에 따라 급격하게 일어날 수 있다는 것을 발견했습니다. 이를 통해 저자들은 개입이 어떻게 상호작용하여 서로 다른 행동 변화를 유발하는지에 대한 예측을 제공합니다. 또한, 이 베이지안 모델은 많은 샷의 탈출 통제를 예측하는 유용한 도구로 자리잡을 수 있음을 보여줍니다.



### Structurally Refined Graph Transformer for Multimodal Recommendation (https://arxiv.org/abs/2511.00584)
Comments:
          Comment: 13 pages, 7 figures, accepted by IEEE Transactions on Multimedia 2025

- **What's New**: 이번 논문에서는 SRGFormer이라는 구조적으로 최적화된 다중 모달 추천 시스템을 제안합니다. 기존 모델들이 사용자 구매 행동을 예측하는 데 있어 다중 모달 정보를 활용하는 동시에 중복 데이터와 유용한 데이터를 구분하지 못하는 문제를 해결하려고 합니다. SRGFormer는 하이퍼그래프 구조에 다중 모달 정보를 임베딩하여 사용자와 아이템 간의 복잡한 상호작용을 포착합니다.

- **Technical Details**: SRGFormer는 Transformer를 수정하여 사용자의 전체 행동 패턴을 포착하도록 설계되었습니다. 모듈 간에는 다중 모달 상호작용 및 모델링, 구조적 정보 상호작용 및 모델링, 융합 및 예측의 세 가지 핵심 모듈이 포함되어 있습니다. 자가 지도 학습(self-supervised learning)을 통해 다중 모달 정보 간의 상호작용을 배우고, 전반적인 정확도를 향상시키기 위해 구조적 관계를 강화합니다.

- **Performance Highlights**: SRGFormer는 Sports 데이터셋에서 기존 모델 대비 평균적으로 4.47% 향상된 성능을 보였습니다. 실험은 세 가지 공개 데이터베이스에서 수행되었으며, 결과는 Recall과 NDCG와 같은 자동 평가 메트릭에서 더욱 개선되었습니다. 이러한 성과는 추천 정확도를显著하게 높이는 데 기여하고 있습니다.



### Reasoning Planning for Language Models (https://arxiv.org/abs/2511.00521)
Comments:
          29 pages, 5 figures

- **What's New**: 이번 논문은 언어 모델 생성에서 적절한 추론 방법 선택의 중요성을 강조합니다. 고전적인 접근 방식은 보통 여러 후보 응답을 생성하고 집계 전략(aggregation strategy)을 통해 최종 답변을 선택하는데, 이는 후보 응답의 수가 많을수록 정확성이 높아진다는 가정을 기반으로 합니다. 저자들은 이러한 가정을 엄밀한 이론적 분석을 통해 재조명하며, 기존 집계 방법의 정확도 경계를 도출했습니다.

- **Technical Details**: 주요 기여로는 EPIC(Ensemble Planning with Contrastive learning) 프레임워크를 도입하였습니다. 이 프레임워크는 모델의 추론 능력과 쿼리-방법 호환성을 포착하는 공유 표현 공간을 학습합니다. 또한, 제안된 확률 경계를 정규화 항(regularizer)으로 사용하여 정확성과 계산 비용의 균형을 맞춘 유틸리티 중심 최적화(utility-driven optimization)를 적용합니다.

- **Performance Highlights**: 다양한 수학적 추론 작업에서 EPIC은 최적의 추론 방법을 일관되게 선택하여 정확도를 향상시켰습니다. 또한, 계산 오버헤드를 줄이는 데 기여하였습니다. 실험 결과, EPIC은 기존 방법보다 더 나은 성능을 보여 주목받고 있습니다.



### \texttt{ReMind}: Understanding Deductive Code Reasoning in LLMs (https://arxiv.org/abs/2511.00488)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)이 코드와 관련된 작업에서 보이는 발전에도 불구하고, 코드 실행 과정에 대한 추론이 부족하다는 실증적 증거를 제시합니다. 특히, 이들 모델이 갖는 세 가지 주요 제약 요소를 밝히고, 이를 해결하기 위해 새로운 멀티 에이전트 프레임워크인 	exttt{ReMind}를 제안합니다. 	exttt{ReMind}는 코드 변형, 실행 및 검사를 통합하여 협력적으로 작동함으로써 어떤 코드 소스에서 발생하는 편향을 줄이도록 설계되었습니다.

- **Technical Details**: 논문에서는 LLMs의 추론 능력이 코드 생성 능력보다 뒤처져 있다는 점, LLMs가 특정 코드 소스에 대해 일관성 있는 편향을 보이는 점, 그리고 복잡한 벤치마크에서 제로샷 제약을 받는 점 세 가지를 제시합니다. 이를 통해 우리가 사용한 연구 질문을 설정하고 LLMs의 코드를 생성하는 능력을 평가하기 위한 엄격한 검증 프로토콜을 개발했습니다. 이 프로토콜을 통해 LLMs가 생성한 코드의 품질을 검증하고, 다양한 코드 출처에서의 성능 변동성을 분석했습니다.

- **Performance Highlights**: 	exttt{ReMind}는 기존의 방법들과 비교했을 때 코드 실행 과정에서 일관성과 타당성을 높여주는 성과를 보였습니다. 5개의 LLMs를 대상으로 한 실험 결과, 	exttt{ReMind}는 독립적인 성능 향상(최대 23.2% 절대 정확도의 향상)을 보여주었으며, 이는 제로샷 제약을 해소하고 모든 코드 출처에 대해 안정적인 예측을 제공하는 데 기여했습니다. 이 연구는 LLMs의 추론 능력 강화를 위한 중요한 기초 자료를 제공했습니다.



### Diverse Human Value Alignment for Large Language Models via Ethical Reasoning (https://arxiv.org/abs/2511.00379)
Comments:
          Accepted by AIES 2025, camera-ready version

- **What's New**: 이 논문에서는 다양한 인류 가치를 더 효과적으로 정렬하기 위해 새로운 윤리적 추론 패러다임을 제안합니다. 기존의 접근 방식은 일반적으로 표면적인 일치를 초래하는 위협이 있었지만, 본 연구에서는 다섯 단계의 구조화된 프로세스를 통해 이러한 문제를 해결하고 진정한 개발을 위한 기반을 제공합니다. 이러한 패러다임은 지역 사회의 특정성을 이해하고 복잡한 윤리적 분석을 수행하는 LLM의 능력을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 프레임워크는 맥락적 사실 수집, 계층적 사회적 규범 식별, 옵션 생성, 여러 시각의 윤리적 영향 분석 및 반성을 포함한 다섯 단계로 구성됩니다. 이 과정은 LLM이 윤리적 결정을 내릴 때 필요로 하는 분석적 및 맥락적 사고를 가능하게 하며, 이를 통해 모델의 해석 가능성이 증대되고 더 나은 결정을 내릴 수 있도록 합니다. 연구는 이러한 패러다임이 Prompt Engineering 또는 Supervised Fine-Tuning(SFT) 방법을 통해 구현 가능하다는 점을 강조합니다.

- **Performance Highlights**: SafeWorld 벤치마크에서 수행한 실험 결과, 제안된 프레임워크는 기존 방법들에 비해 LLM이 다양한 인간 가치와 더 잘 정렬되도록 한다는 것을 입증했습니다. 이는 사회적 규범 식별의 정확성을 높이고 문화적으로 적합한 추론을 가능하게 하여, LLM의 규범 식별 및 정렬 점수에서 유의미한 개선을 가져오는 것으로 나타났습니다. 이러한 작업은 다양한 글로벌 사회의 다차원적인 가치에 더 효과적으로 정렬될 수 있는 LLM 개발을 위한 구체적인 경로를 제공합니다.



### Reject Only Critical Tokens: Pivot-Aware Speculative Decoding (https://arxiv.org/abs/2511.00351)
Comments:
          Accepted at NeurIPS 2025 Efficient Reasoning Workshop

- **What's New**: 이번 연구에서는 Speculative Decoding (SD)의 엄격한 분포 일치를 완화하고, LLMs의 유용성(utility)에 초점을 맞춘 새로운 디코딩 전략인 Pivot-Aware Speculative Decoding을 제안합니다. 기존의 SD는 발전 모델의 낮은 수용률로 인해 속도가 제한되었다는 점을 강조하며, 유틸리티 중심의 접근 방식을 소개합니다.

- **Technical Details**: 이 방법은 발췌된 토큰의 유용성을 고려하여, 최종 출력의 유용성을 저하시키지 않는다고 판단되는 토큰만 거부합니다. 이를 위해 ‘pivot tokens’로 불리는 비판적 토큰에 대한 분류기를 훈련시키며, 유용성을 유지하면서 속도를 높이는 전략을 세웁니다. 연구에서 명시된 유틸리티 함수는 텍스트 생성의 성능을 이진적으로 정의하는 방법을 채택하고 있습니다.

- **Performance Highlights**: 제안된 방법은 다양한 데이터셋에서 평가되었으며, 기존 방법들에 비해 최대 2.5배의 속도 향상을 보여주면서 동등한 유용성을 유지했습니다. 이 접근 방식은 LLMs의 실제 적용 사례와 잘 맞아떨어지며, 다양한 작업에 적용 가능한 범용적인 디코딩 알고리즘입니다.



### Calibration Across Layers: Understanding Calibration Evolution in LLMs (https://arxiv.org/abs/2511.00280)
Comments:
          Accepted at EMNLP 2025 (main)

- **What's New**: 본 논문에서는 대규모 언어 모델(LLMs)의 교정(calibration) 능력을 깊이(depth)에 따라 어떻게 발전하는지를 조사했습니다. 기존 연구에서 깊은 신경망이 과도하게 자신감(confidence)을 갖는 경향이 있다고 보고되었으나, LLM은 다양한 작업에서 잘 조정된 확률을 보여준다고 합니다. 특히, 마지막 층의 구성 요소와 같은 특정 특성과 관련이 있는 이 현상을 분석하여, 모델의 신뢰도가 예측 확률과 잘 맞춰져 있다는 점을 밝히고 있습니다.

- **Technical Details**: 연구팀은 MMLU 벤치마크를 활용하여 여러 개방형 모델을 분석했으며, 상위 레이어에서의 신뢰도 교정(confidence correction phase) 단계를 드러냈습니다. 이 과정에서 모델의 결정 확신이 높아지면, 이후 레이어에서 신뢰도가 능동적으로 조정되는 양상을 확인했습니다. 또한 저차원 교정 방향이 잔여 스트림(residual stream) 내에 존재하며, 이 방향을 교란하면 교정 메트릭(ECE 및 MCE)이 유의미하게 향상됨을 입증했습니다.

- **Performance Highlights**: 본 연구는 LLM의 교정 과정이 모델의 마지막 단계를 넘어서도 이루어지며, 신뢰도가 정확성과 단순히 상관관계가 아니라는 점을 강조하고 있습니다. 연구 결과는 LLM의 임계과정(calibration process)을 보다 진보된 방식으로 이해할 수 있도록 하여, 신뢰도 조정을 위한 새로운 통찰력을 제공합니다. 이 연구는 더 나아가 해석 가능한 언어 모델의 신뢰도 조정 연구를 촉진할 것으로 기대됩니다.



### LongCat-Flash-Omni Technical Repor (https://arxiv.org/abs/2511.00279)
- **What's New**: LongCat-Flash-Omni는 560억 개의 파라미터를 가진 최신 오픈 소스 omni-modal 모델로, 실시간 오디오-비주얼 인터랙션에서 뛰어난 성능을 보여줍니다. 점진적인 학습 전략을 통해 다양한 모달리티(모드)의 시퀀스 모델링 작업을 수행하며, 강력한 unimodal 능력을 유지하면서도 포괄적인 멀티모달 능력을 달성하였습니다.

- **Technical Details**: LongCat-Flash-Omni는 Shortcut-connected Mixture-of-Experts (MoE) 아키텍처를 기반으로 하며, 효율적인 멀티모달 인식 및 음성 재구성 모듈을 결합합니다. 560B 파라미터 중 27B가 활성화된 상태에서도 낮은 대기 시간으로 실시간 오디오-비주얼 인터랙션을 지원합니다. 또한 모달리티 분리 병렬 처리(MDP) 전략을 통해 대규모 멀티모달 훈련의 데이터 및 모델 이질성을 관리합니다.

- **Performance Highlights**: LongCat-Flash-Omni는 Omni-Bench 및 WorldSense와 같은 omni-modal 벤치마크에서 최신 성능을 기록하며, 텍스트, 이미지, 비디오 이해 및 음성 이해와 생성 등 다양한 unimodal 작업에서도 높은 경쟁력을 보여줍니다. 궁극적으로, LongCat-Flash-Omni는 오픈 소스 커뮤니티에서 가장 강력한 omni-modal 모델로 자리잡고 있으며, 고품질의 오디오-비주얼 인터랙션을 지원합니다.



### Advancing Cognitive Science with LLMs (https://arxiv.org/abs/2511.00206)
- **What's New**: 인지 과학(cognitive science)은 지식의 종합과 개념적 명확성(conceptual clarity)에서 지속적인 문제에 직면하고 있으며, 이는 복합적이고 학제적(nterdisciplinary) 성격 때문이다. 최근 인공지능(AI)의 발전, 특히 대규모 언어 모델(large language models, LLMs)의 발달이 이러한 문제를 해결할 수 있는 도구를 제공할 수 있다는 점이 주목받고 있다. 본 리뷰에서는 LLM이 분야가 역사적으로 힘들었던 여러 영역(예: 학제 간 연결, 이론의 형식화(formalizing theories), 명확한 측정 분류 개발 등)에 어떻게 기여할 수 있는지를 살펴본다.

- **Technical Details**: LLM은 문헌 매핑(literature mapping), 이론 형식화, 측정 정제(measurement refinement) 등에 도움을 주며, 사람의 행동 및 사고에 대한 생성적 예측(generative predictions)을 제공하는 인지 모델로도 사용될 수 있다. 이 논문은 LLM이 연구 분야를 효율적으로 매핑하는 방법, 연구 기여를 상대적 위치에 두는 방법, 그리고 이론의 일반화(generalizability) 및 기존 증거의 유용성을 평가하는 데 어떻게 기여할 수 있는지를 자세히 설명한다. 이와 함께 LLM의 현재 기능과 한계, 잠재적 함정에 대해서도 논의한다.

- **Performance Highlights**: 연구 결과에 따르면, LLM이 인간 전문가와 경쟁하여 연구의 예측 정확도를 높일 수 있는 잠재력을 보여준다. 예를 들어, LLM은 신경과학 연구 결과를 인간보다 더 정확하게 예측할 수 있는 것으로 나타났으며, 이는 LLM이 인지 과학의 특정 영역에서 이미 전문가 기준을 충족하거나 초과하는 데 도움을 줄 수 있음을 시사한다. 이러한 도구는 기존 지식의 격차를 식별하고 누적적(progress) 진전을 위한 기준점을 설정하는 데 유용하게 활용될 수 있다.



### Can SAEs reveal and mitigate racial biases of LLMs in healthcare? (https://arxiv.org/abs/2511.00177)
- **What's New**: 이 연구는 Sparse Autoencoders (SAEs)를 활용하여 의료 분야에서의 LLMs(대형 언어 모델) 사용 시 인종 편향을 식별하고 완화할 수 있는 방법을 제안합니다. 특히 Black과 White으로 식별되는 환자들의 퇴원 요약을 통해 인종과 오명 있는 개념 간의 연관성을 평가합니다. 연구진은 모델이 Black 정체성을 증가시키면 환자가 더 공격적일 위험이 증가한다는 상황을 확인했습니다.

- **Technical Details**: 연구진은 퇴원 요약 데이터를 사용하여 SAEs의 레이턴트(latent)를 분석하고, 인종 예측에서 가장 높은 변별력을 보이는 레이턴트를 식별했습니다. 이를 위해 여러 모델에서 활성화 정보를 수집하며, 이 정보를 기반으로 로지스틱 회귀(probe)를 통해 인종을 예측하는 방식으로 접근했습니다. 특정 레이턴트가 Black 정체성과 관련된 신호를 보내는 것을 확인하였으며, 이러한 신호는 차별적인 개념에도 의존하고 있다는 것을 밝혀냈습니다.

- **Performance Highlights**: 연구 결과, SAEs가 단순한 과제에서 인종 편향을 줄이는 데 도움을 줄 수 있다는 것을 발견했지만, 복잡하고 현실적인 임상 과제에는 그 효과가 제한적이었습니다. SAEs의 활성화 특징을 통해 의료 분야에서 LLM의 잠재적 편향을 탐지하고 제어할 수 있지만, 그 유용성은 여전히 개선이 필요함을 시사합니다. 연구진은 AI 시스템이 특정 인구 통계에 지나치게 의존하게 만드는 문제를 해결하기 위한 추가적인 연구가 필요하다고 강조합니다.



### Real-time and Zero-footprint Bag of Synthetic Syllables Algorithm for E-mail Spam Detection Using Subject Line and Short Text Fields (https://arxiv.org/abs/2511.00118)
- **What's New**: 본 논문은 이메일 스팸 필터링 기술을 개선하기 위해 Bag of Synthetic Syllables (BoSS) 알고리즘을 제안합니다. 기존의 머신러닝 기반 방법들이 리소스를 많이 소모하고 느린 처리 속도로 인해 실시간 적용이 어려운 반면, BoSS 알고리즘은 간단하면서도 효과적인 필터링 방법을 제공합니다. 이 알고리즘은 이메일의 주제 행에서 간단한 텍스트를 기반으로 필요한 차원 해시를 생성하여 스팸을 탐지합니다.

- **Technical Details**: BoSS 알고리즘은 영어 문자 집합을 기반으로 하며, 11 kB 길이의 문자열을 처리합니다. 이 알고리즘은 문자와 기호를 분리하기 위해 특정 규칙을 따르며, 합성 음절을 생성하는 데 일본어 음절 구조를 사용합니다. 입력된 짧은 텍스트는 약 146 차원의 공간으로 표현되고, 이를 이용해 코사인 또는 유클리드 거리 계산을 수행하여 스팸과의 유사성을 판단합니다.

- **Performance Highlights**: 실험 결과 BoSS 알고리즘은 실제 SMTP 트래픽에서 스팸 탐지 성능을 나타냈으며, 기존의 복잡한 딥러닝 알고리즘을 사용하는 대신 빠르고 효율적으로 스팸을 필터링 할 수 있음을 보여주었습니다. 또한 이 알고리즘은 추가적인 저장소나 하드웨어 업그레이드를 필요로 하지 않으며, CPU 및 메모리 자원을 최소한으로 소모합니다.



### Wayfinding through the AI wilderness: Mapping rhetorics of ChatGPT prompt writing on X (formerly Twitter) to promote critical AI literacies (https://arxiv.org/abs/2511.00106)
Comments:
          Published in the journal Computers and Composition, Issue 74 (2024)

- **What's New**: 이 논문에서는 소셜 미디어에서 ChatGPT 프롬프트 작성을 연구함으로써 비판적 AI 리터러시를 촉진하는 방법을 보여줍니다. 프롬프트 작성(Prompt Writing)은 ChatGPT와 같은 생성적 AI 도구를 위해 원하는 출력을 이끌어내기 위해 지시사항을 작성하는 과정입니다. 최근 소셜 미디어에서 이와 관련된 대화가 급증하고 있습니다.

- **Technical Details**: 논문에서는 컴퓨터와 작문에서의 디지털 작문 연구의 네 가지 겹치는 전통을 바탕으로 프롬프트 작성의 사회적 수사(Social Media Rhetorics)에 대해 연구하였습니다. 2022년 11월부터 2023년 5월까지 X(구 Twitter)에서 수집한 32,000개의 포스트(post)를 분석하여 반복적인 연구 프로세스를 진행하였습니다. 이 과정에서 질적 방법(Qualitative Methods)과 계산적 방법(Computational Methods)를 혼합하여 사용하였습니다.

- **Performance Highlights**: 이 연구에서 다룬 다섯 가지 주제는 다음과 같습니다: (1) 프롬프트 작성이 영향을 미치는 커뮤니케이션 영역, (2) 프롬프트 작성을 위한 미세 리터러시 자원, (3) 프롬프트 작성을 형성하는 시장 수사, (4) 프롬프트의 수사적 특성, (5) 프롬프트 작성을 정의하는 것입니다. 우리는 이러한 주제와 방법론에 대해 논의하며 비판적 AI 리터러시를 가르치고 분석하는 디지털 작문 교사와 연구자들에 대한 중요한 시사점을 강조합니다.



### QuantumBench: A Benchmark for Quantum Problem Solving (https://arxiv.org/abs/2511.00092)
Comments:
          11 pages, 8 figures

- **What's New**: 이 논문은 QuantumBench라는 새로운 벤치마크를 소개합니다. QuantumBench는 양자 과학 분야에서 LLM(대형 언어 모델)의 성능을 체계적으로 평가하기 위해 설계되었습니다. 양자 과학은 비직관적인 현상과 고급 수학이 요구되기 때문에, 일반적인 벤치마크로는 평가하기 어려운 특성이 있습니다.

- **Technical Details**: QuantumBench는 공개된 자료를 기반으로 약 800개의 질문과 답변 쌍을 모아, 여덟 가지 선택지로 구성된 다중 선택 데이터셋으로 정리되었습니다. 이 데이터셋은 15개의 양자 과학 관련 강좌에서 수집된 자료로 구성되며, 질문의 모호성을 없애기 위해 LLM의 도움을 받아 수정했습니다. 질문은 대수 계산, 수치 계산, 개념 이해 세 가지 유형으로 분류됩니다.

- **Performance Highlights**: 양자 도메인에서 LLM의 성능을 평가함으로써, 질문 형식 변화에 대한 민감성을 분석 및 정량화합니다. QuantumBench는 양자 과학 연구에서 LLM의 효과적인 사용을 안내하며, 향후 과학 분야 전반에 걸쳐 LLM의 사용 개선을 강조합니다. 이 연구는 AI의 과학 연구 자동화 및 발견에 기여할 것으로 기대됩니다.



### Generalizing Test-time Compute-optimal Scaling as an Optimizable Graph (https://arxiv.org/abs/2511.00086)
Comments:
          Under review

- **What's New**: 본 논문에서는 Test-Time Scaling (TTS)을 통해 대형 언어 모델 (LLMs)의 성능을 극대화하는 새로운 문제를 제안합니다. 기존의 연구는 정적인 아키텍처와 단일 모델 사용에 집중되었으나, 작업에 따라 최적의 아키텍처와 모델 조합이 다를 수 있다는 점을 간과했습니다. 이를 해결하기 위해 저자들은 컴퓨팅 최적화를 위한 모델 및 아키텍처 조합의 검색을 다루는 Multi-LLM Collaboration Graph를 형식화했습니다.

- **Technical Details**: 제안된 방법론인 Agent-REINFORCE는 확률적 최적화를 통해 최적의 Multi-LLM Collaboration Graph를 탐색합니다. 이 과정에서 노드는 LLM 모델과 역할을 나타내고, 엣지는 정보 흐름을 포착합니다. REINFORCE 알고리즘을 기반으로 하여, LLM 기반 에이전트가 경험적 통찰을 반영하여 후보 초기화 및 분포 업데이트를 실행합니다.

- **Performance Highlights**: 실험 결과, Agent-REINFORCE는 전통적인 방법 및 LLM 기반 기준선보다 샘플 효율성과 탐색 성능에서 우수한 성과를 보였습니다. 또한, 정확성과 추론 지연을 동시에 최적화하는 그래프를 효과적으로 식별해내는 데 성공했습니다. 본 연구는 작업별 요구 사항에 따른 동적 구조 최적화의 필요성을 강조합니다.



### Chitchat with AI: Understand the supply chain carbon disclosure of companies worldwide through Large Language Mod (https://arxiv.org/abs/2511.00024)
- **What's New**: 이 논문은 기업의 탄소 공개 품질을 대규모로 평가하기 위한 새로운 의사결정 지원 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLMs)을 활용하여 2010년부터 2020년까지의 CDP 데이터에 대해 내러티브 점수화 방식으로 통합된 평가를 수행합니다. 이로 인해 산업과 지역 간의 비교가 가능해지며, 기업의 환경 지속 가능성을 분석하는 데 소요되는 비용과 시간을 대폭 줄일 수 있는 혁신적인 방법을 제공합니다.

- **Technical Details**: CDP 데이터셋은 환경 성과 공개에 대한 기업의 응답을 수집하며, 구조화된 지표와 개방형 서사의 혼합 형태로 되어 있습니다. 제안된 LLM 기반 접근은 체계적인 평가 파이프라인을 통해 개별 기업의 응답을 점수화하도록 설계되어 있습니다. 이 방법론은 서사적 평가의 일관성을 높이고, 다양한 산업 및 지역 간의 성과를 비교하는 데 필요한 기준을 제시합니다.

- **Performance Highlights**: 기술 및 독일과 같은 국가가 높은 서사 점수 정렬을 보여주는 반면, 일부 산업은 변동성을 보이거나 피상적 참여를 나타내는 것으로 파악되었습니다. 이 연구의 결과는 투자자, 규제 기관 및 기업의 환경, 사회, 지배구조(ESG) 전략가들에 대한 향후 의사결정에 중요한 인사이트를 제공하며, AI 기반 의사결정 지원 시스템의 기능을 한층 향상시킬 수 있는 기반을 마련합니다.



### Multimodal Detection of Fake Reviews using BERT and ResNet-50 (https://arxiv.org/abs/2511.00020)
Comments:
          Published in IEEE

- **What's New**: 이 연구에서는 디지털 상거래 환경에서 사용자가 생성한 리뷰의 중요성을 강조하고, 허위 리뷰의 문제를 해결하기 위한 다중 모달(fake review detection) 프레임워크를 제안합니다. 이 프레임워크는 BERT를 활용한 텍스트 특징과 ResNet-50을 활용한 시각적 특징을 결합하여 리뷰의 진위를 판별합니다. 기존의 단일 모달(unimodal) 접근 방식이 아닌 다중 모달(multimodal) 접근 방식을 통해 세밀한 모순을 감지할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 방법론은 데이터셋 준비, 전이 학습을 통한 특징 추출, 다중 모달 특징 융합, 이진 분류의 네 가지 주요 단계로 구성됩니다. 텍스트 특징은 BERT의 사전 훈련된 모델을 사용하여 추출하고, 이미지 특징은 ResNet-50을 통해 획득합니다. 이 두 가지 특징 벡터는 결합되어 최종 이진 분류를 위한 입출력 벡터를 형성합니다.

- **Performance Highlights**: 실험 결과, 다중 모달 모델은 단일 모달 기초선에 비해 뛰어난 성능을 나타내며, 테스트 세트에서 F1-score 0.934를 기록했습니다. 또한 혼돈 행렬(confusion matrix)과 질적 분석을 통해 모델이 고품질 이미지를 보유한 잘못된 텍스트 높은 호평 리뷰를 감지하는 능력을 확인했습니다. 이 연구는 디지털 신뢰를 보장하는 데 있어 다중 모달 학습의 중요한 역할과 온라인 플랫폼에서의 콘텐츠 조정 문제를 해결하는 스케일러블한 솔루션을 제시합니다.



### Multimodal Learning with Augmentation Techniques for Natural Disaster Assessmen (https://arxiv.org/abs/2511.00004)
Comments:
          Accepted at 2025 IEEE 21st International Conference on Intelligent Computer Communication and Processing (ICCP 2025)

- **What's New**: 이 논문은 자연재해 평가에 필요한 정보에 대한 신속하고 정확한 접근 방식의 중요성을 강조하고 있습니다. 특히 소셜 미디어가 재난 분석을 위한 실시간 데이터 소스로 주목받고 있지만, 기존 데이터셋이 불균형 클래스와 한정된 샘플 문제로 인해 모델 개발이 어렵다는 점를 지적합니다. 해결책으로, 실험에서는 CrisisMMD 다중 양식 데이터셋을 활용하여, 다양한 증강 기법을 적용하여 재난 분류 모델의 성능을 향상시킵니다.

- **Technical Details**: 이 연구에서는 주로 두 가지 이미지 증강 기술인 Real Guidance와 DiffuseMix를 적용했습니다. Real Guidance는 조건부 이미지-투-이미지 생성 기법으로, Stable Diffusion 1.5 모델을 사용하여 원래 이미지를 현실적으로 변형합니다. DiffuseMix는 Masked Blending과 Fractal Visual Modifications를 사용하여 다양한 증강 이미지를 생성하고, 이는 데이터셋의 일부에 국한된 클래스의 샘플 수를 증가시킵니다.

- **Performance Highlights**: 실험 결과, 선택된 증강 전략이 특히 저조한 클래스의 분류 성능을 개선하는 데 기여했음을 보여주었습니다. 다중 시점 학습은 잠재력을 보였지만 추가적인 정제가 필요하다는 점이 지적되었습니다. 이 연구는 제안한 증강 기법들이 재난 평가 시스템을 더욱 견고하게 구축하는 데 기여할 수 있음을 보여줍니다.



New uploads on arXiv(cs.IR)

### Trove: A Flexible Toolkit for Dense Retrieva (https://arxiv.org/abs/2511.01857)
- **What's New**: Trove는 실험의 유연성과 속도를 유지하면서 연구 실험을 간소화하는 오픈 소스 리트리벌(추출) 툴킷입니다. 이번 연구에서 처음으로 데이터 관리 기능이 도입되어, 연구자들이 최소한의 코드로 리트리벌 데이터셋을 동적으로 로드하고 처리할 수 있는 방법을 제공합니다. 이로 인해 대규모 데이터셋의 다양한 구성을 실험하는 데 필요한 메모리 소모를 줄이고, 여러 복사본을 저장할 필요가 없습니다.

- **Technical Details**: Trove는 사용자가 기존 구성 요소를 자유롭게 수정하거나 사용자 정의 객체로 완전히 교체할 수 있는 높은 커스터마이즈(맞춤형) 기능을 제공합니다. 이 툴킷은 평가 및 하드 네거티브 마이닝을 위한 로우 코드(low-code) 통합 파이프라인을 제공하며, 코드 변경 없이 멀티 노드(multi-node) 실행을 지원합니다. 이를 통해 연구자는 실험을 더 쉽게 설정하고 실행할 수 있습니다.

- **Performance Highlights**: Trove의 데이터 관리 기능은 메모리 소모를 2.6배 줄이며, 사용이 간편한 추론 파이프라인에서도 오버헤드(overhead)가 발생하지 않습니다. 추론 시간은 사용 가능한 노드의 수에 비례하여 선형적으로 감소합니다. 이 모든 기능은 Trove가 실험을 간소화하고 임의의 커스터마이징을 지원하며 탐색적 연구를 촉진하는 데 어떻게 기여하는지를 잘 보여줍니다.



### CAT-ID$^2$: Category-Tree Integrated Document Identifier Learning for Generative Retrieval In E-commerc (https://arxiv.org/abs/2511.01461)
Comments:
          Accepted by WSDM'26

- **What's New**: 본 논문에서는 전통적인 ID 생성 방법에서 한계를 극복하기 위해 CAtegory-Tree Integrated Document IDentifier (CAT-ID$^2$)라는 새로운 ID 학습 방법을 제안합니다. 이 방법은 서열적 질의(Query) 처리 과정에서 계층적 카테고리 정보(hierarchical category information)를 통합하여 문서 ID를 생성합니다. CAT-ID$^2$은 3가지 주요 모듈을 포함하고 있으며, 따라서 유사한 문서는 비슷한 ID를 갖고, 각각의 문서는 독특한 ID를 유지합니다.

- **Technical Details**: CAT-ID$^2$의 주요 구성 요소는 세 가지 손실(loss) 함수로 구성됩니다. 1) Hierarchical Class Constraint Loss는 계층(Category) 정보를 통합하여 ID을 생성하는 과정에서 내부 카테고리의 밀집함을 보장하고, 2) Cluster Scale Constraint Loss는 ID 토큰의 균일한 분포를 유지하며, 3) Dispersion Loss는 재구성된 문서의 차별성을 향상시킵니다. 이러한 구성을 통해 CAT-ID$^2$는 유사한 문서의 ID를 유사하게 만들면서도 서로 다른 문서의 독창성을 보존할 수 있게 됩니다.

- **Performance Highlights**: 논문에서 제안한 방법은 오프라인 및 온라인 실험을 통해 효과성을 입증하였으며, 특히 온라인 A/B 테스트에서 애매한 의도 질의에 대해 사용자 천명당 평균 주문이 0.33% 증가하였고, 롱테일 질의에 대해서는 0.24% 증가하는 결과를 보여주었습니다. 이 결과는 CAT-ID$^2$가 E-커머스 상황에서도 문서 검색 성능을 향상시키는 데 기여할 수 있음을 나타내고 있습니다.



### LiCoMemory: Lightweight and Cognitive Agentic Memory for Efficient Long-Term Reasoning (https://arxiv.org/abs/2511.01448)
- **What's New**: 본 논문에서는 LiCoMemory라는 새로운 메모리 프레임워크를 제안하여 LLM(대규모 언어 모델)의 한계를 극복하고 있습니다. 이 시스템은 CogniGraph라는 경량의 계층 그래프를 활용하여 동적 검색 및 지식 업데이트를 실현하며, 의미론적 인덱싱을 통해 정보를 보다 구조적으로 관리할 수 있습니다. 또한, 기존의 메모리 접근 방식의 비효율성을 줄이고, 업데이트 지연 시간을 대폭 감소시켰습니다.

- **Technical Details**: LiCoMemory는 사용자-어시스턴트 상호작용 중 실시간으로 업데이트 및 검색을 수행하도록 설계되었습니다. 이는 CogniGraph라는 경량의 계층적 그래프 구조를 도입하여 지식 그래프의 역할을 정리하며, 노드와 엣지 내부에 방대한 정보를 포함시키는 대신 구조적 스캐폴드를 사용합니다. 이러한 방식으로 지식의 색인화 및 조직화가 이루어지며, 컨텍스트에 민감한 검색이 가능하게 됩니다.

- **Performance Highlights**: LiCoMemory의 성능은 LoCoMo와 LongMemEval이라는 장기 대화 벤치마크에서 기존의 최우수 성능과 비교하여 최대 23% 향상된 정확도를 달성했습니다. 다중 세션 및 시간적 추론 세트에서 특히 두드러진 성과를 보이며, 입력 토큰 수와 응답 지연 시간을 유의미하게 감소시키는 효율성을 입증했습니다. 이를 통해 LiCoMemory는 실시간으로 더 높은 품질의 지식을 검색하고 컨텍스트에 기반한 응답을 생성할 수 있는 통합 메모리 시스템으로 자리매김하고 있습니다.



### A Soft-partitioned Semi-supervised Collaborative Transfer Learning Approach for Multi-Domain Recommendation (https://arxiv.org/abs/2511.01404)
Comments:
          Accepted by CIKM'25

- **What's New**: 이 논문에서는 멀티 도메인 추천 시스템에서의 데이터 불균형 문제를 해결하기 위해 Soft-partitioned Semi-supervised Collaborative Transfer Learning (SSCTL)이라는 새로운 방법을 제안합니다. SSCTL은 우세한 도메인 데이터를 활용하여 비우세 도메인에 대한 샘플의 초점을 조정하고, 가짜 레이블(pseudo-labels)을 통해 비우세 도메인의 데이터를 보강합니다. 이를 통해 과적합(overfitting) 문제를 완화하고, 공유 매개변수(shared parameters)의 압도적인 영향을 줄이는 것을 목표로 합니다.

- **Technical Details**: SSCTL은 Instance Soft-partitioned Collaborative Training (ISCT)과 Soft-partitioned Domain Differentiation Network (SDDN)이라는 두 가지 모듈로 구성되어 있습니다. ISCT는 우세한 도메인 샘플을 레이블이 없는 데이터로 간주하고 가중치를 부여한 가짜 레이블을 생성하여 비우세 도메인의 데이터를 개선합니다. 반면 SDDN은 소프트 파르티셔닝 정보(soft-partitioned domain information)를 활용해 동적 매개변수를 생성하여 공유 매개변수에서의 압도적인 영향을 줄입니다.

- **Performance Highlights**: 실험 결과, 온라인 테스트에서 여러 도메인에서 유의미한 성과 향상을 달성하였으며, 총 판매액(GMV)은 0.54%에서 2.90% 증가하고 클릭률(CTR)은 0.22%에서 1.69% 증가했습니다. 이러한 결과는 제안한 SSCTL 방법이 실질적인 개선을 보였음을 나타냅니다. 따라서 SSCTL은 멀티 도메인 추천 시스템의 성능을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### A semantic-based deep learning approach for mathematical expression retrieva (https://arxiv.org/abs/2511.01364)
- **What's New**: 이번 연구는 수학 표현(Mathematical Expressions, MEs) 검색을 위한 새로운 심층 학습 접근 방식을 제안합니다. 기존의 구문적 유사성 기반의 접근 방식과 비교하여, 의미적 유사성을 바탕으로 한 방법론이 도입되었습니다. 연구팀은 심층 순환 신경망(Deep Recurrent Neural Network, DRNN)을 활용하여 MEs의 의미적 특징을 추출하고 이를 이용하여 검색을 수행하는 시스템을 개발했습니다.

- **Technical Details**: 연구에서는 먼저 MEs의 복잡도를 표기하는 세 가지 클래스(단순, 중간, 복잡)를 정의하고, 각 표현의 중첩 깊이에 따라 복잡도를 정량화했습니다. DRNN을 활용하여 각 수학 표현의 의미적 특징을 계산한 후, 이 특징을 데이터베이스에 저장하여 검색 시 매칭을 수행합니다. euclidean distance를 기반으로 상위 'k'개의 유사한 MEs를 반환하며, 여기서 'k'는 사용자가 정의하는 파라미터입니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식을 통해 829개의 MEs 데이터베이스에서 효과적으로 검색이 이루어졌습니다. 심층 학습 기반의 의미적 접근 방식은 기존의 LCS 알고리즘보다 더 나은 성능을 보였으며, 시간 복잡도는 선형으로 감소하였습니다. 이 연구는 수학 표현 검색 문제에 있어 의미적 및 구문적 접근 방식을 처음으로 비교 분석한 것으로, 향후 연구에 중요한 기초 자료가 될 것입니다.



### Contextual Relevance and Adaptive Sampling for LLM-Based Document Reranking (https://arxiv.org/abs/2511.01208)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)에 의해 생성된 관련성 판단을 통해 문서 검색 품질을 향상시키기 위한 재정렬 알고리즘을 소개합니다. 특히, 문서의 맥락에 따라 결정되는 관련성을 정의하는 'contextual relevance' 개념을 제안하고, 문서들이 재정렬 모델에 제시되는 순서와 구성을 통해 성능이 달라질 수 있음을 강조합니다. 이를 해결하기 위해 TS-SetRank라는 샘플링 기반 불확실성 인식 재정렬 알고리즘을 개발했습니다.

- **Technical Details**: 상황 관련성(contextual relevance)은 주어진 쿼리에 대한 문서의 관련성이 여러 후보 문서와 함께 제시되는 상황에서 측정되는 방법입니다. TS-SetRank는 두 단계의 베이지안 재정렬 알고리즘으로, 문서 배치를 균일하게 샘플링하여 바이너리 관련성 피드백을 수집한 후, 톰슨 샘플링(Thompson sampling)을 사용하여 적응적으로 배치를 구성합니다. 이 방법은 독립적인 Beta-Bernoulli 사후 분포를 유지하여 각 문서의 불확실성을 모델링합니다.

- **Performance Highlights**: TS-SetRank는 BRIGHT와 BEIR 벤치마크에서 nDCG@10 지표에서 15-25% 개선을 이루었고, 검색 및 재정렬 기준에 비해 평균적으로 성능이 더 우수함을 입증했습니다. 본 연구는 문서 검색에서 다중 맥락을 고려한 재정렬의 중요성을 강조하며, 추가적인 성능 향상을 위해 다양한 구조적 모델이 필요할 수 있음을 논의합니다.



### Controlling Gender Bias in Retrieval via a Backpack Architectur (https://arxiv.org/abs/2511.00875)
- **What's New**: 이번 연구는 큰 언어 모델(LLM)에 내재된 사회적 편향을 다루기 위한 새로운 접근 방식을 제안합니다. 특히, 전통적인 Transformer 모델과는 달리 텍스트 시퀀스를 비문맥적이고 학습된 단어 측면(들)으로 가중 조합하여 출력하는 Backpack Language Model을 활용하여, 편향 조정이 가능한 랭킹 시스템을 구축했습니다. 이 프레임워크는 성별 편향을 효과적으로 줄이면서도 성능 저하를 최소화하는 것을 목표로 합니다.

- **Technical Details**: 제안된 방안은 Backpack 언어 모델을 기반으로 하여, 각 토큰을 비문맥적 감각 벡터의 가중 조합으로 표현합니다. 모델의 추론 파이프라인 내에서 공정성 기준에 따라 감각 벡터의 가중치를 재조정하는 방식으로 성별 편향을 제어할 수 있습니다. 이는 재훈련 없이 표시되는 편향 제어를 허용하며, 이 과정에서 Softmax 크로스 엔트로피 손실을 사용하여 전체 리스트를 고려한 최적화를 수행합니다.

- **Performance Highlights**: 실험 결과, 우리의 모형은 MS MARCO 데이터셋에서 성별 편향을 줄이면서도 양호한 성능을 보였습니다. λ=1.0일 때 MRR@10/NDCG@10은 각각 0.3343/0.4025로 가장 높은 값을 기록하였고, λ=0.5의 공정성 가중치에 대해 TF/Boolean에서 RaB/ARaB가 모든 컷오프에서 최소값을 달성했습니다. 성능을 약 2-3%만 저하시키면서도 공정성을 개선하는 데 성공했습니다.



### REaR: Retrieve, Expand and Refine for Effective Multitable Retrieva (https://arxiv.org/abs/2511.00805)
Comments:
          13 pages, 2 figures, 8 tables

- **What's New**: REAR (Retrieve, Expand and Refine)라는 새로운 프레임워크가 소개되었습니다. 이 프레임워크는 서로 다른 테이블 간의 연결 가능성을 고려하여 다중 테이블 검색을 효율적으로 수행할 수 있도록 설계되었습니다. REAR은 쿼리와 관련된 테이블을 검색하고, 구조적으로 조인 가능한 테이블을 확장하며, 노이즈가 많은 항목을 제거하는 단계로 구성되어 있습니다.

- **Technical Details**: REAR는 세 단계로 이루어진 구조적 모델입니다: (1) Retrieval: 쿼리와 의미적으로 연관된 테이블을 검색합니다. (2) Expansion: 검색된 테이블과 조인 가능한 추가 테이블을 추가합니다. (3) Refinement: 쿼리와 테이블의 연관성을 종합적으로 평가하여 최종 테이블을 정제합니다. 이 모델은 기존 LLM 기반 접근법들과 달리 LLM 호출을 사용하지 않으며, 각 단계에서 효율성을 극대화합니다.

- **Performance Highlights**: 다양한 복잡한 테이블 QA 데이터셋에서 REAR는 기존 리트리버에 비해 일관되게 성능을 향상시키며, LLM 기반 시스템과 거의 유사한 성능을 달성했습니다. REAR는 SQL 실행 성능 또한 향상시키며, 큰 데이터베이스에 대한 검색의 실용성을 높입니다. 실험 결과는 각 단계의 효과를 확인하며, REAR가 기존 시스템의 한계를 극복할 수 있음을 보여줍니다.



### Taxonomy-based Negative Sampling In Personalized Semantic Search for E-commerc (https://arxiv.org/abs/2511.00694)
Comments:
          Accepted at 2025 IEEE International Conference on Big Data

- **What's New**: 이 연구에서는 최신 전자상거래 검색을 위한 의미 검색 모델을 제안합니다. 이 모델은 쿼리와 제품을 공유된 벡터 공간에 임베딩하여, 구매 패턴을 고려하여 보다 관련성 높은 제품을 검색합니다. 특히, 새로 제안된 분류 기반의 하드-네거티브 샘플링(TB-HNS) 전략을 사용하여 맥락상 관련성 있는 하드 네거티브를 효과적으로 추출합니다.

- **Technical Details**: 제안된 모델은 고객의 과거 구매 이력과 행동을 모델링하여 개인화된 검색 결과를 제공합니다. 기존의 랜덤 샘플링 기법은 의미상 관련이 없는 제품을 제공할 수 있지만, TB-HNS는 카테고리 계층을 활용하여 관련성 있지만 비슷한 제품을 선택합니다. 이를 통해 훈련 효율성을 높이고, 학습 속도를 가속화합니다.

- **Performance Highlights**: 오프라인 실험에서 이 모델은 BM25 및 기존 신경망 모델을 초과하는 Recall@K 성능을 보였고, 실시간 A/B 테스트에서는 전환율, 장바구니 추가율 및 평균 주문 가치를 크게 향상시켰습니다. 또한, TB-HNS는 훈련 오버헤드를 줄이고 데이터 준비 효율성을 높이는 장점을 보여주었습니다.



### Structurally Refined Graph Transformer for Multimodal Recommendation (https://arxiv.org/abs/2511.00584)
Comments:
          Comment: 13 pages, 7 figures, accepted by IEEE Transactions on Multimedia 2025

- **What's New**: 이번 논문에서는 SRGFormer이라는 구조적으로 최적화된 다중 모달 추천 시스템을 제안합니다. 기존 모델들이 사용자 구매 행동을 예측하는 데 있어 다중 모달 정보를 활용하는 동시에 중복 데이터와 유용한 데이터를 구분하지 못하는 문제를 해결하려고 합니다. SRGFormer는 하이퍼그래프 구조에 다중 모달 정보를 임베딩하여 사용자와 아이템 간의 복잡한 상호작용을 포착합니다.

- **Technical Details**: SRGFormer는 Transformer를 수정하여 사용자의 전체 행동 패턴을 포착하도록 설계되었습니다. 모듈 간에는 다중 모달 상호작용 및 모델링, 구조적 정보 상호작용 및 모델링, 융합 및 예측의 세 가지 핵심 모듈이 포함되어 있습니다. 자가 지도 학습(self-supervised learning)을 통해 다중 모달 정보 간의 상호작용을 배우고, 전반적인 정확도를 향상시키기 위해 구조적 관계를 강화합니다.

- **Performance Highlights**: SRGFormer는 Sports 데이터셋에서 기존 모델 대비 평균적으로 4.47% 향상된 성능을 보였습니다. 실험은 세 가지 공개 데이터베이스에서 수행되었으며, 결과는 Recall과 NDCG와 같은 자동 평가 메트릭에서 더욱 개선되었습니다. 이러한 성과는 추천 정확도를显著하게 높이는 데 기여하고 있습니다.



### Listwise Preference Diffusion Optimization for User Behavior Trajectories Prediction (https://arxiv.org/abs/2511.00530)
- **What's New**: 이번 논문은 사용자 행동 예측을 위한 새로운 접근법인 사용자 행동 경로 예측(User Behavior Trajectory Prediction, UBTP)을 소개합니다. 기존의 추천 시스템이 단기적인 사용자의 행동만을 고려했던 반면, UBTP는 장기적인 사용자 선호와 행동 패턴을 모델링하는 데 중점을 두고 있습니다. 이를 통해 사용자 경험을 향상시키고 비즈니스 성과를 극대화하는 것을 목표로 합니다.

- **Technical Details**: 우리는 Listwise Preference Diffusion Optimization (LPDO)이라는 새로운 훈련 프레임워크를 제안합니다. LPDO는 전체 아이템 시퀀스에 대한 구조화된 선호도를 직접 최적화하고, Plackett-Luce 감독 신호를 통합하여 다단계 경로 예측의 정확성을 높입니다. 이 접근법은 기존의 독립적인 예측 방식의 한계를 극복합니다.

- **Performance Highlights**: 광범위한 실험을 통해 LPDO가 기존의 최신 기술들에 비해 우수한 성능을 보인다는 것을 입증했습니다. 특히 새로운 평가 메트릭인 Sequential Match (SeqMatch)를 제안하여 다단계 예측의 품질을 rigorously 검증하였습니다. LPDO는 네 가지 벤치마크 데이터셋에서 새로운 최고 성능을 달성하며, 구조화된 선호도 학습의 새로운 기준을 세웠습니다.



### LIR: The First Workshop on Late Interaction and Multi Vector Retrieval @ ECIR 2026 (https://arxiv.org/abs/2511.00444)
Comments:
          Accepted workshop at ECIR 2026

- **What's New**: 최근 정보 검색(Information Retrieval, IR) 분야에서 Late Interaction Retrieval 방법이 주목받고 있습니다. ColBERT로 시작된 이 방법들은 단일 벡터 기반 신경 IR에 대한 강력한 대안으로, 세밀한 토큰 수준의 표현을 활용하여 일반화와 견고성을 제공합니다. 더욱이 이러한 모델들은 Reasoning-based 또는 Cross-modality Retrieval 같은 새로운 사용 사례에 특히 적합한 것으로 나타났습니다.

- **Technical Details**: Late Interaction Retrieval 모델은 문서와 쿼리를 각각의 토큰으로 표현합니다. 이들 모델은 쿼리 토큰과 문서 토큰을 비교하는 MaxSim 연산자를 사용하여 문서에 대한 쿼리의 적합도를 계산합니다. 이 접근 방식은 단일 벡터 방법에서 발생하는 정보 손실을 피하고, 토큰 간의 세밀한 상호작용을 가능하게 합니다.

- **Performance Highlights**: 2023년초, LLM Retrieval-Augmented Generation(RAG) 파이프라인의 인기로 ColBERT 모델에 대한 사용자 친화적인 도구가 증가하고, 수백만 번의 다운로드를 기록하며 공업적으로도 큰 관심을 끌고 있습니다. 다양한 연구 흐름 속에서, Multi-modal Retrieval과 같은 새로운 영역에서도 유망한 초기 결과를 보이고 있으며, 후속 연구와 협업의 촉진의 중요성이 강조되고 있습니다.



### Simple and Behavior-Driven Augmentation for Recommendation with Rich Collaborative Signals (https://arxiv.org/abs/2511.00436)
Comments:
          10 pages. This paper is accepted at IEEE BigData 2025 (Short)

- **What's New**: 본 논문에서는 간단한 협업 증강 방식인 SCAR(Simple Collaborative Augmentation for Recommendation)를 제안합니다. 기존의 노이즈 제거 접근법의 한계를 극복하고, 사용자-아이템 상호작용에서 협업 신호를 활용하여 가상 상호작용을 생성하는 방식을 채택했습니다. 이로 인해 중요한 정보를 잃거나 복잡한 증강 모듈의 단점을 피하면서도, 그래프 협업 필터링(GCF)에서의 클러스터 성능을 극대화합니다.

- **Technical Details**: SCAR는 두 가지 주요 증강 기능인 협업 엣지 추가(ColAdd)와 협업 엣지 교체(ColRep)를 통해 작동합니다. ColAdd에서는 아이템에 대한 효과 점수를 도출하여 가상 상호작용을 추가하고, ColRep에서는 가장 효과가 낮은 엣지를가장 유사한 엣지로 교체합니다. 이 과정은 사용자와 아이템 표현을 개선하여 효과적인 대조 학습을 통해 다양한 협업 신호를 생성합니다.

- **Performance Highlights**: SCAR는 네 개의 벤치마크 데이터세트를 기준으로 광범위한 실험을 수행하였으며, 여덟 개의 기준 방법을 초월하는 성능을 보여주었습니다. 특히 SCAR는 다양한 하이퍼파라미터 설정에서도 강력한 강건성을 보이며, 데이터가 희소한 경우 특히 효과적입니다. 간단하고 직관적인 증강 기법을 통해 성능을 최적화하는 데 성공했습니다.



### Effectiveness of LLMs in Temporal User Profiling for Recommendation (https://arxiv.org/abs/2511.00176)
Comments:
          Accepted to the IEEE International Conference on Data Mining (ICDM 2025), Workshop on User Modeling and Recommendation (UMRec). To appear in the IEEE ICDMW 2025 proceedings

- **What's New**: 이 논문은 사용자 취향의 동적 특성을 효과적으로 모델링하는 데 있어 Large Language Models(LLMs)를 활용하여 추천 시스템의 품질을 개선하는 방법을 다룹니다. 전통적인 사용자 프로파일링 기법은 일시적인 단기 관심과 안정적인 장기 선호를 구분하지 못해 추천의 정확성과 투명성을 저해하는데, 본 연구는 이러한 차이를 텍스트 요약을 통해 캡처합니다. 재구성된 사용자 표현은 사용자와 아이템 간의 상호작용 이력을 기반으로 하여 단기 및 장기 요약을 생성하고, 이를 통해 이론적 해석 가능성을 제공합니다.

- **Technical Details**: 이 연구에서는 사용자와 아이템 간의 상호작용 및 타임스탬프를 기반으로 하는 상호작용 기록을 정의합니다. LLM을 활용한 사용자 프로파일의 생성을 통해 단기 및 장기 프로파일을 뚜렷하게 구분하고, BERT를 이용하여 이러한 프로파일을 임베딩합니다. 학습 가능한 주의(attention) 메커니즘을 적용하여 이 임베딩들을 통합하여 최종 사용자 표현을 생성하며, 이는 최근의 관심과 오래된 관심 사이의 결정적 근거를 명확히 전달합니다.

- **Performance Highlights**: 본 연구는 Movies&TV 및 Video Games라는 두 개의 아마존 도메인에서 평가했습니다. Movies&TV 도메인에서 추천 품질을 17%나 향상시켰으나, Video Games 도메인에서는 덜 뚜렷한 혜택을 보였습니다. 이는 사용자 프로파일 밀도와 행동 다양성이 낮은 도메인에서 단기와 장기 선호의 구분이 이뤄지지 않기 때문으로, LLM 기반의 접근 방식의 적용이 어떤 환경에서 더욱 유용한지에 대한 통찰을 제공합니다.



### LookSync: Large-Scale Visual Product Search System for AI-Generated Fashion Looks (https://arxiv.org/abs/2511.00072)
Comments:
          4 pages, 5 figures. Accepted at the International Conference on Data Science (IKDD CODS 2025), Demonstration Track. Demo video: this https URL

- **What's New**: 이번 논문에서는 AI 생성 스타일과 유사한 제품을 빠르게 검색할 수 있는 시스템을 소개합니다. 이 시스템은 1200만 개 이상의 제품에 대한 고차원 임베딩을 생성하며, AI 생성 이미지를 키워드로 변환하여 시각적으로 유사한 제품을 찾습니다. 이 시스템은 일일 35만 개의 AI Looks를 처리하며, 이를 통해 사용자에게 더욱 개인화된 쇼핑 경험을 제공합니다.

- **Technical Details**: 제안하는 시스템은 쿼리 생성, 벡터화, 후보 검색, 재순위 매김을 포함한 4개의 주요 구성 요소로 이루어져 있습니다. AI 생성 이미지를 반영하여 가장 유사한 제품을 추출하며, CLIP 모델을 활용하여 높은 정확도로 유사도를 평가합니다. 시스템은 평균적으로 1초 이내의 응답 속도를 유지하며, 사용자 인터렉션에 기반한 추천 품질도 개선되었습니다.

- **Performance Highlights**: 제안된 시스템은 CLIP을 기반으로 하여 다양한 제품 카테고리에서 비슷한 스타일을 유지하는 제품을 효과적으로 찾아냅니다. 실험 결과, CLIP 모델이 다른 대안 모델들보다 평균 의견 점수에서 3~7% 우위를 점했습니다. 이러한 작은 개선이 사용자 경험을 크게 향상시키는 것을 보여주며, 실제 운영 환경에서 경쟁력 있는 솔루션으로 자리잡게 되었습니다.



### A Graph-based RAG for Energy Efficiency Question Answering (https://arxiv.org/abs/2511.01643)
- **What's New**: 이 연구에서는 에너지 효율성(EE) 질문 응답을 위한 그래프 기반의 Retrieval Augmented Generation (RAG) 아키텍처 내에서 대형 언어 모델(LLM)의 사용을 조사했습니다. 이 시스템은 에너지 분야의 가이드라인과 규제 문서에서 자동으로 지식 그래프(KG)를 추출하고, 이 그래프를 탐색하여 사용자에게 다양한 언어로 정확한 답변을 제공합니다. 인간 기반의 검증 실험을 통해 이 아키텍처의 잠재력을 확인하고 강점과 약점을 파악하였습니다.

- **Technical Details**: 시스템은 세 가지 주요 구성요소로 나뉘며, 첫 번째로 지식 추출기가 도메인 특정 문서에서 엔티티와 관계를 포함한 트리플을 추출합니다. 이렇게 추출된 정보는 지식 기반(KB)을 구축하는 데 사용되며, 이후 사용자의 질문을 처리하기 위한 검색 및 생성 프로세스가 이어집니다. 이 과정에서 LLM 기반의 알고리즘이 사용되어 트리플을 자동으로 추출하고, 동일한 구문이 적용되도록 엔티티 이름을 통합합니다.

- **Performance Highlights**: 검증 결과, 시스템은 약 75.2%의 경우에 정확한 답변을 제공하며, 에너지 효율성에 대한 일반 질문에 대해서는 81.0%까지 높아지는 성능을 보입니다. 또한, 다국어 처리에 있어 번역으로 인한 정확도 손실은 4.4%로 나타났습니다. 이러한 결과는 그래프 기반 RAG 아키텍처가 다국적 사용자를 위한 효율적인 질문 응답 시스템을 구축할 수 있는 잠재력을 가지고 있음을 시사합니다.



### Vote-in-Context: Turning VLMs into Zero-Shot Rank Fusers (https://arxiv.org/abs/2511.01617)
- **What's New**: 본 논문은 Vote-in-Context (ViC)라는 일반화된 훈련 없는 프레임워크를 소개합니다. ViC는 Vision-Language Model (VLM)을 사용하여 리스트 단위의 재순위를 수행하고, 복잡한 비디오와 같은 다중 모드 데이터를 효과적으로 통합할 수 있도록 설계되었습니다. 이 프레임워크는 콘텐츠 증거와 검색기 메타데이터를 직접 VLM의 프롬프트에 직렬화하여 모델이 적응적으로 일관성을 평가할 수 있도록 합니다.

- **Technical Details**: ViC는 S-Grid라는 압축된 콘텐츠 직렬화 맵을 도입하여 각 비디오를 이미지 그리드로 표현합니다. 이 구조는 VLM에 의해 해석될 수 있는 콘텐츠 증거를 제공하며, 비디오 후보들에 대한 리스트 단위의 추론을 가능하게 합니다. ViC는 두 가지 모드에서 작동하며, 첫 번째는 단일 리스트 재순위 모드로, 두 번째는 여러 검색기를 조합하는 앙상블 융합 모드입니다.

- **Performance Highlights**: ViC는 영상 검색 벤치마크에서 기존의 최고 성능을 크게 능가하는 결과를 보여 주었습니다. ViC는 MSR-VTT에서 87.1%(t2v), 89.0%(v2t)의 Recall@1을 기록했으며, VATEX에서는 99.6%(v2t)를 기록했습니다. 이러한 결과는 이전의 저명한 기준 대비 최대 +40 Recall@1의 향상과 같은 상당한 이득을 나타냅니다.



### Calculating Web Impact Factor for University Websites of Jammu and Kashmir: A Study (https://arxiv.org/abs/2511.01496)
Comments:
          11 pages, Research Paper

- **What's New**: 이 논문은 자무와 카슈미르 지역의 12개 대학 웹사이트에 대한 웹 메트릭(webometric) 연구를 통해 웹 영향 계수(web impact factor, WIF)를 탐구하고 분석합니다. 각 웹사이트의 도메인 시스템을 확인하고, 웹 페이지 수와 링크 페이지 수를 분석하였습니다. 이를 바탕으로 각 대학 웹사이트의 외부 링크 WIF와 간단한 웹 영향 계수를 계산하였습니다.

- **Technical Details**: 분석 과정에서는 각 대학 웹사이트의 웹 페이지와 링크 페이지의 수를 심층적으로 조사했습니다. 이 연구는 특정 대학 웹사이트들이 더 많은 웹 페이지를 가지고 있지만, 링크 페이지 수는 적어서 웹 영향 계수 측면에서 뒤쳐지는 경향이 있음을 발견했습니다. 특히 클러스터 대학교(Cluster University) 자무는 내부 링크 WIF에서 1위를 차지했습니다.

- **Performance Highlights**: 결과적으로, 클러스터 대학교 자무는 내부 링크 WIF에서 1위를 기록하며(0.9018), 시리 마타 바이슈노 데비 대학교(Shri Mata Vaishno Devi University)는 외부 링크 웹 영향 계수에서 1위를 차지했습니다(0.7249). 이러한 결과는 각 대학 웹사이트의 상호 연결성 및 외부 링크 진행 상황을 시각적으로 나타냅니다.



### Impact and Relevance of Cognition Journal in the Field of Cognitive Science: An Evaluation (https://arxiv.org/abs/2511.01485)
Comments:
          8 pages, 4 figures, Research Paper. arXiv admin note: substantial text overlap with arXiv:2102.12912, arXiv:2102.09900, arXiv:2102.09894

- **What's New**: 이 연구는 1999년부터 2018년까지 20년 동안 'Cognition' 저널에 대한 과학계량학( scientometric ) 분석을 제시합니다. 본 연구의 목적은 해당 저널의 연구 활동에 대한 요약을 제공하고 여러 측면을 특성화하는 것입니다. 이 연구는 연도별 논문 분포, 저자, 기관, 국가 및 인용 분석을 포함합니다.

- **Technical Details**: 분석 결과, 1999년부터 2018년까지 'Cognition' 저널에서 총 2870편의 논문이 발표되었습니다. 연구는 저널의 상위 20명의 저자, 기관 및 국가를 식별했습니다. 또한, 연구자들은 저널에 대한 기여도는 대부분 미국 연구자들에 의해 이루어진 것으로 나타났습니다.

- **Performance Highlights**: 이 저널은 지난 20년간 학술 활동이 활발했던 것으로 보이며, 미국이 가장 많은 기여를 한 것으로 분석되었습니다. 저널의 기여자를 세부적으로 분석함으로써, 연구자들의 연구 경향과 주요 기여를 이해하는 데 도움을 줍니다.



### RAGSmith: A Framework for Finding the Optimal Composition of Retrieval-Augmented Generation Methods Across Datasets (https://arxiv.org/abs/2511.01386)
Comments:
          45 pages

- **What's New**: RAGSmith는 레트리벌-증강 생성(Retrieval-Augmented Generation, RAG)의 통합 최적화를 위한 모듈형 프레임워크로, 46,080개의 파이프라인 구성에서 발생하는 상호작용을 고려합니다. 이 연구는 RAG 시스템의 구성 요소 간의 복잡한 synergies에 초점을 맞추며, 각 모듈을 독립적으로 최적화하는 전통적인 접근 방식을 재편합니다. RAGSmith는 유전 알고리즘을 통해 완전한 파이프라인 최적화를 가능하게 하며, 다양한 도메인에서 일관된 성능 향상을 나타냅니다.

- **Technical Details**: RAGSmith는 9개의 기술 군을 포함하여 46,080개의 구성 조합을 평가합니다. 이 시스템은 레트리벌 메트릭(retrieval metrics)과 생성 메트릭(generation metrics) 모두를 통합하여 최적화를 진행하고, 다양한 질문 유형에 대한 반응성을 기반으로 전반적인 성능 향상을 목표로 합니다. 또한, RAGSmith는 질문 타입에 민감한 최적화 지침을 설정하여, 도메인별 데이터셋 특성에 맞춰 최적의 구성 요소 조합을 찾아냅니다.

- **Performance Highlights**: RAGSmith는 기본 RAG 베이스라인 대비 평균적으로 +3.8%의 성능 향상을 달성하며, 특정 도메인에서는 최대 +12.5%의 레트리벌 개선 및 +7.5%의 생성 개선을 나타냅니다. 실험 결과, 장기 답변 및 사실 기반 질문 세트에서 더 큰 개선이 이루어졌으며, 이는 RAG 시스템 구성에서 질문 유형의 중요성을 강조합니다. 이 결과는 evolucionary search의 장점을 입증하며, 효과적인 RAG 시스템 구성 방법에 대한 실용적 가이드를 제공합니다.



### Rescuing the Unpoisoned: Efficient Defense against Knowledge Corruption Attacks on RAG Systems (https://arxiv.org/abs/2511.01268)
Comments:
          15 pages, 7 figures, 10 tables. To appear in the Proceedings of the 2025 Annual Computer Security Applications Conference (ACSAC)

- **What's New**: 최근의 연구는 대형 언어 모델(LLMs)의 한계인 훈련 비용 및 정보 공백 문제를 해결하기 위한 방법으로 Retrieval-Augmented Generation (RAG)이 주목받고 있음을 보여줍니다. RAG 시스템은 외부 지식 소스로부터 정보를 검색하여 응답을 생성하는 방식으로 작동합니다. 그러나 최근에는 RAG 시스템의 취약점, 특히 지식 오염 공격에 대한 우려가 증대하고 있습니다.

- **Technical Details**: 이 연구에서 제안된 RAGDefender는 RAG 시스템에 대한 지식 오염 공격을 방어하기 위한 효율적인 메커니즘입니다. RAGDefender는 검색한 패시지를 그룹화하고 적대적 패시지를 식별하는 두 가지 주요 단계로 구성되어 있습니다. 이 시스템은 추가적인 모델 훈련이나 추론을 요구하지 않으며 가벼운 머신 러닝 기술을 활용하여 적대적 내용을 탐지하고 필터링합니다.

- **Performance Highlights**: 실험 결과, RAGDefender는 여러 모델과 적대적 시나리오에서 기존의 방어 방법들보다 일관되게 우수한 성과를 보였습니다. 예를 들어, Gemini 모델에 대한 공격 성공률(ASR)을 0.89에서 0.02로 낮춘 반면, RobustRAG와 Discern-and-Answer는 각각 0.69와 0.24의 성과를 기록했습니다. RAGDefender는 RobustRAG보다 약 12.36배, Discern-and-Answer보다 1.53배 더 빠른 속도를 자랑합니다.



### Object-Centric Analysis of XES Event Logs: Integrating OCED Modeling with SPARQL Queries (https://arxiv.org/abs/2511.00693)
Comments:
          12 pages, 4 figures, PROFES2025 conference

- **What's New**: 이 논문은 Object Centric Event Data Ontology (OCEDO)를 통해 프로세스 마이닝을 위한 이벤트 로그의 XES 표준의 한계를 극복하고자 합니다. OCEDO 모델을 활용하면 이벤트 간의 종속성과 관계를 명확히 하고, 프로세스 데이터의 완전성과 가독성을 향상시키는 데 기여합니다. 이 연구는 BPIC 2013 데이터셋에서 OCEDO 접근 방식을 어떻게 적용할 수 있는지를 시연하며, 새로운 메타 모델을 통해 객체 중심의 시각으로 문제를 해결하는 데 중점을 둡니다.

- **Technical Details**: XES는 전통적인 이벤트 로그를 위한 XML 기반 포맷으로, 단일 프로세스 인스턴스를 보여주는 데 주력합니다. 그러나 OCEDO는 다중 표기법을 허용하고 객체 간의 종속성을 추출함으로써 전통적인 방식의 한계를 극복합니다. SPARQL 쿼리를 이용하여 OCEDO 모델을 통해 도출된 이벤트-객체 관계를 탐색할 수 있게 하며, 복잡한 정보 시스템에서 적합하게 구조화된 프로세스 데이터를 제공합니다.

- **Performance Highlights**: BPIC 2013 데이터셋을 OCEDO 형식으로 변환하여 이벤트와 객체 간의 더 정교한 관계를 탐색할 수 있는 가능성을 보여줍니다. 이 기법은 서비스 프로세스에서의 실질적인 운영 문제를 식별하고 성과 이슈를 분석하는 데 도움이 됩니다. OCEDO는 기존의 OCEL 2.0 표준과 상호 보완적으로 작용하며, 사용자 요구에 따라 대안적 또는 보조적인 옵션을 제공합니다.



### PolyRecommender: A Multimodal Recommendation System for Polymer Discovery (https://arxiv.org/abs/2511.00375)
- **What's New**: 본 논문에서는 PolyRecommender라는 다중 모드 발견 프레임워크를 소개합니다. 이 시스템은 PolyBERT의 화학 언어 표현과 그래프 인코더의 분자 그래프 기반 표현을 통합하여 폴리머 후보를 효율적으로 검색하고 순위 매기는 기능을 수행합니다. 이러한 방법은 AI 기반 소재 설계의 능력을 향상시켜 차세대 폴리머의 발견을 가속화합니다.

- **Technical Details**: PolyRecommender는 "퍼널" 아키텍처를 활용하여 첫 번째 단계에서 12,441개의 폴리머 중 후보를 빠르게 검색하고, 두 번째 단계에서 언어 및 그래프 임베딩을 융합하여 후보를 평가합니다. 이를 통해 각 폴리머의 SMILES 표현과 세 가지 실험적 속성을 활용하여 최종적이고 정확한 예측을 수행합니다. 최적의 성능을 위해 MMoE (Multi-gate Mixture of Experts) 모델을 사용하여 태스크별 예측을 개선합니다.

- **Performance Highlights**: PolyRecommender는 다중 모드 전략을 적용하여 경쟁을 통해 최적의 폴리머 후보를 찾아내며, 세 가지 주요 폴리머 속성에 대한 예측에서 뛰어난 성과를 보였습니다. 연구 결과, MMoE 융합 모델이 가장 뛰어난 성능을 보여줬고, UMAP 시각화를 통해 화학 공간에서 각 특징별로 클러스터를 형성하는 것을 확인했습니다. 특히 유리 전이 온도 및 밴드 갭을 예측하는 데에서 뛰어난 성과를 나타내었으며, 폴리에틸렌 산화물(PEO)을 사례로 이루어진 연구는 실용성을 강조합니다.



### IL-PCSR: Legal Corpus for Prior Case and Statute Retrieva (https://arxiv.org/abs/2511.00268)
Comments:
          Accepted at EMNLP 2025 (Main)

- **What's New**: 이 논문은 법률 사례와 관련 법규를 검색하는 두 가지 작업인 Legal Statute Retrieval (LSR)와 Prior Case Retrieval (PCR)를 통합하는 IL-PCSR (Indian Legal Corpus for Prior Case and Statute Retrieval)라는 고유한 말뭉치를 제안합니다. 기존의 접근법은 각각의 작업에 독립적으로 모델을 개발했으나, 두 작업 간의 상호 의존성을 활용하고자 합니다. IL-PCSR은 동일한 쿼리 집합에 대해 관련된 법규와 사례를 동시에 탐색할 수 있는 첫 번째 데이터셋입니다.

- **Technical Details**: IL-PCSR은 936개의 법규와 3,183개의 이전 사례, 6,271개의 쿼리 문서로 구성됩니다. 법률 문서는 인도 대법원과 고등법원 판결에서 수집된 것으로, 공공적으로 이용 가능한 자료를 API로 통해 얻었습니다. 이 데이터셋은 기계 학습 모델이 법규와 사례 간의 의존성을 학습할 수 있도록 설계되었으며, LLM(대규모 언어 모델) 기법을 이용한 재정렬 방법도 개발하였습니다.

- **Performance Highlights**: 논문에서는 다양한 모델을 사용하여 LSR와 PCR 작업의 성능을 평가했으며, LSR과 PCR 각각에 대해 문자 기반 모델과 의미 기반 모델을 포함한 다수의 실험을 수행했습니다. 제안하는 파이프라인 기반 접근 방식은 각 작업의 성능을 개별적으로 개선할 뿐만 아니라, 다중 작업 모델에도 긍정적인 영향을 미쳤습니다. 실험 결과, 법규 검색 및 사례 검색 간의 차이가 성능에 미친 영향을 분석하였고, 이러한 인사이트는 향후 연구의 방향을 제시합니다.



### AI Powered High Quality Text to Video Generation with Enhanced Temporal Consistency (https://arxiv.org/abs/2511.00107)
- **What's New**: MOVAI(다중 모드 원본 비디오 AI)는 텍스트로부터 비디오를 생성하기 위한 새로운 계층적 프레임워크로, 구성 장면 이해(compositional scene understanding)와 시간 인지(diffusion models)를 통합하여 고충실도의 비디오 합성을 가능하게 합니다. 이 프레임워크는 텍스트 설명을 계층적 장면 그래프로 분해하는 Compositional Scene Parser(CSP), 프레임 간의 일관성 있는 움직임을 보장하는 Temporal-Spatial Attention Mechanism(TSAM), 비디오 품질을 점진적으로 향상시키는 Progressive Video Refinement(PVR) 모듈을 포함한 세 가지 주요 혁신을 소개합니다.

- **Technical Details**: MOVAI는 입력 텍스트 TT와 선택적 조건 입력(스타일, 기간, 해상도)을 바탕으로 비디오 시퀀스 V를 생성하는 조건부 생성 문제로 설정됩니다. 전체 시스템은 세 가지 상호 연결된 모듈로 구성되며, 입력 처리(Input Processing), 장면 이해(Scene Understanding), 주의 처리(Attention Processing), 비디오 생성(Video Generation) 단계로 이루어져 있습니다. 각 단계에서는 고유한 기술적 접근 방식을 통해 장면의 객체 및 관계를 세밀하게 모델링하며, 모든 프레임 간의 시간적 일관성을 유지합니다.

- **Performance Highlights**: MOVAI는 다양한 평가 메트릭에서 기존 방법에 비해 15.3%의 LPIPS, 12.7%의 FVD 및 18.9%의 사용자 선호도 개선을 포함하여 최첨단 성능을 입증했습니다. 특히, 복잡한 다중 객체 장면을 생성할 때 강점을 보이며, 사용자들로부터 더 일관되게 선호되는 결과를 제공합니다. 이는 단순히 무엇이 비디오에 나타나야 할지뿐만 아니라, 객체가 시간에 따라 어떻게 이동하고 상호작용해야 하는지에 대한 훨씬 더 나은 제어를 가능하게 합니다.



### Forecasting Occupational Survivability of Rickshaw Pullers in a Changing Climate with Wearable Data (https://arxiv.org/abs/2511.00081)
Comments:
          This is a preprint version of a manuscript accepted and to be published in the Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies (IMWUT)

- **What's New**: 이번 연구는 방글라데시 다카의 자전거 리크샤 끌기 노동자들이 극심한 더위에 어떻게 생리학적 바이오마커에 반응하는지를 조사합니다. 우리는 100명의 리크샤 끌기 노동자들로부터 실시간 기후 및 생리학적 데이터를 수집하고, 12명의 노동자들과 인터뷰를 통해 기후 변화에 대한 인식 및 경험을 탐구하였습니다. 또한 Linear Gaussian Bayesian Network (LGBN) 회귀 모델을 개발하여 활동, 날씨, 인구 통계적 특성에 기반하여 주요 생리학적 바이오마커를 예측하였습니다.

- **Technical Details**: 우리는 연구 등급의 웨어러블 기기를 사용하여 리크샤 끌기 노동자들의 생리학적 변화를 실시간으로 측정하였습니다. 이 기기는 Photoplethysmography (PPG), 정전기 피부 반응, 가속도계 및 피부 온도 센서를 포함하고 있습니다. 연구에는 100명의 참여자 데이터로 이루어진 헬스 데이터셋(heat exposure dataset)을 구축하고, 이를 이용해 통계 및 회귀 분석을 수행하였습니다.

- **Performance Highlights**: 모델은 피부 온도, 상대 심장 비용, 피부 전도 반응 및 피부 전도 수준에 대해 각각 0.82, 0.47, 0.65, 0.67의 정규화된 평균 절대 오차(Normalized Mean Absolute Error)를 달성했습니다. 연구에 따르면, 현재 32%의 리크샤 끌기 노동자가 높은 열 노출 위험에 직면하고 있으며, 2026-2030년에는 이 비율이 37%에 이를 가능성이 있습니다. 추가적으로, 노동자들은 기후 변화에 대한 걱정을 표명하며, 건강과 직업적인 생존 가능성에 미치는 영향에 대해 우려하고 있습니다.



New uploads on arXiv(cs.CV)

### TIR-Bench: A Comprehensive Benchmark for Agentic Thinking-with-Images Reasoning (https://arxiv.org/abs/2511.01833)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 시각적 문제 해결을 위한 새로운 벤치마크인 TIR-Bench를 소개합니다. 기존의 벤치마크들이 단순한 시각적 검색 작업에만 집중하는 반면, TIR-Bench는 13개 다양한 작업을 포함하여 이미지 처리와 조작에서의 도구 사용 능력을 평가합니다. 이 연구는 대화형(Multimodal) 모델들의 이미지를 다루는 더 진화된 능력을 평가할 기반을 제공합니다.

- **Technical Details**: TIR-Bench는 체인 오브 스로우트(Chain-of-Thought) 접근 방식에 기반하여 복잡한 시각적 질문을 해결하기 위해 텍스트 단계로 분해하면서, 이미지 처리와 조작을 위한 도구 사용을 요구하는 다양한 작업을 포함합니다. 이 벤치마크는 다양한 도구 기반의 상호작용을 평가할 수 있는 13가지 작업을 통해 에이전틱(Agentic) 모델의 능력을 체계적으로 검증합니다. 업무 설계는 단순 정적 관찰이 아닌, 입력 이미지의 적극적인 조작을 요구하는 방식으로 구성되었습니다.

- **Performance Highlights**: 22개의 멀티모달 대형 언어 모델(MLLMs)에 대한 평가 결과, TIR-Bench는 도전적인 벤치마크로 나타났으며, 최상의 성능은 46%에 불과했습니다. 도구 사용 능력이 없는 전통적인 모델의 성능은 저조했지만, o3와 같은 도구 사용 기능이 있는 모델은 훨씬 더 높은 성능을 보였습니다. 이는 TIR-Bench가 에이전틱 사고 및 도구 사용 능력이 중요하다는 것을 잘 보여줍니다.



### SciTextures: Collecting and Connecting Visual Patterns, Models, and Code Across Science and Ar (https://arxiv.org/abs/2511.01817)
- **What's New**: 이 연구에서는 Scitextures 데이터셋을 소개합니다. 이 데이터셋은 과학, 기술, 예술의 다양한 영역에서 수집된 텍스처와 시각 패턴을 포함하고 있습니다. 1,200개 이상의 모델과 100,000개의 이미지를 통합하여 시각 패턴과 이를 생성하는 메커니즘 간의 연관성을 탐구할 수 있습니다.

- **Technical Details**: Scitextures는 자동으로 모델을 수집하고 구현하는 인공지능 파이프라인에 의해 생성되었습니다. 이 데이터셋은 물리학, 화학, 생물학, 사회학, 기술, 수학, 예술에 걸쳐 다양한 패턴과 텍스처를 포함하여, 인공지능 모델의 시각 패턴 인식 능력을 평가하는 데 사용됩니다. 특히, 모델은 실제 패턴의 이미지를 이용하여 해당 패턴을 형성한 메커니즘을 추론하고 재구성할 수 있습니다.

- **Performance Highlights**: 비전-언어 모델(Vision-Language Models, VLMs)은 단순한 시각 패턴을 넘어 물리 시스템을 이해하고 시뮬레이션할 수 있는 능력을 보여주었습니다. 이 연구는 AI가 자연 이미지로부터 패턴을 인식하고 재현하며, 해당 메커니즘을 코드로 작성하여 생성된 이미지를 실제 이미지와 비교하는 능력을 평가합니다. 이러한 벤치마크는 인공지능의 출력이 실제 세계의 복잡한 시스템을 모사할 수 있음을 입증합니다.



### PROPEX-RAG: Enhanced GraphRAG using Prompt-Driven Prompt Execution (https://arxiv.org/abs/2511.01802)
Comments:
          Accepted in PReMI 2025

- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG)의 새로운 접근 방식인 prompt-driven GraphRAG 프레임워크를 제안하여, 프롬프트 디자인이 여러 단계의 질문 응답(Multi-hop QA)에서 정보 검색과 추론 과정을 개선하는 데 중요한 역할을 한다는 점을 강조합니다. 기존 RAG 방법의 한계를 극복하고자, 문맥에 따라 엔티티 추출, 사실 선택 및 패시지 재정렬을 지원할 수 있도록 구조화된 사실 삼중형을 사용하는 지식 그래프를 구축하였습니다.

- **Technical Details**: 우리는 LLM(대형 언어 모델) 추출된 사실 삼중형으로부터 상징적 엔티티 중심 지식 그래프를 구성하고, 프롬프트 조건의 LLM 필터링을 통해 관련 사실을 선택하는 프로세스를 통합했습니다. 단일 시점에서의 추론을 위한 다단계(multi-hop) 검색을 위해 Personalized PageRank(PPR) 알고리즘을 사용하여 구조화된 그래프상에서 문맥적으로 연결된 패시지를 검색합니다.

- **Performance Highlights**: 이 시스템은 HotpotQA와 2WikiMultiHopQA에서 각각 80.7%와 78.9%의 F1 점수, 97.1%와 98.1%의 Recall@5 점수로 최신 성능을 기록했습니다. 이러한 결과는 프롬프트 디자인이 검색 정확성과 응답 품질을 향상시키기 위한 중요한 요소임을 보여줍니다.



### How Far Are Surgeons from Surgical World Models? A Pilot Study on Zero-shot Surgical Video Generation with Expert Assessmen (https://arxiv.org/abs/2511.01775)
- **What's New**: 이 논문에서는 SurgVeo라는 최초의 전문가 리뷰 기반 비디오 생성 모델 평가 벤치마크를 소개하고, 수술 관련 비디오 생성의 평가를 위한 새로운 네 단계의 구조인 Surgical Plausibility Pyramid(SPP)를 제안합니다. 이는 수술 환경에서의 모델 출력을 평가하는 데 있어 심층적인 인과 지식을 요구하는 중요한 발판을 제공합니다. 연구자들은 Veo-3 모델을 활용하여 외과적 문제를 제로샷(zero-shot) 예측 과제로 설정하여 평가하였습니다.

- **Technical Details**: SurgVeo 벤치마크는 50개의 비디오 클립을 포함하고 있으며, 이를 통해 laparoscopy(복강경 수술) 및 neurosurgery(신경외과) 절차의 다양한 단계와 복잡성을 포괄합니다. SPP는 시각적 인식 가능성(Visual Perceptual Plausibility), 도구 조작 가능성(Instrument Operation Plausibility), 환경 피드백 가능성(Environment Feedback Plausibility), 수술 의도 가능성(Surgical Intent Plausibility) 등 네 가지 차원으로 구성되어 있습니다. 연구진은 4명의 보드 인증 외과의사로부터 SPP에 근거한 평가를 받았습니다.

- **Performance Highlights**: Veo-3는 시각적 인식 가능성에서 높은 점수를 기록하여 전문가 외과의사들을 놀라게 했으나, 도구 조작 가능성 및 환경 피드백 가능성에서 중요한 부족함을 드러냈습니다. 특히 수술 의도 가능성 평가에서 수술 행동의 의도를 파악하지 못하는 결과를 보였으며, 이는 시각적으로 그럴듯한 비디오 생성과 높은 수준의 전문 지식 간의 갭을 강하게 드러냅니다. 연구 결과는 수술 교육, 계획 및 자율 수술 로봇 개발 등 여러 의료 분야에 중요한 영향을 미칠 수 있습니다.



### UniLION: Towards Unified Autonomous Driving Model with Linear Group RNNs (https://arxiv.org/abs/2511.01768)
- **What's New**: 이 논문은 우수한 성능을 자랑하는 UniLION이라는 통합 자율주행 모델을 제안합니다. UniLION은 LiDAR 포인트 클라우드, 고해상도 다중 뷰 이미지, 그리고 시간적 시퀀스 데이터를 효율적으로 처리합니다. 이 모델은 특별한 시간적 혹은 다중 모달 융합 모듈 없이는 다양한 작업을 잘 수행할 수 있도록 설계되었습니다.

- **Technical Details**: UniLION은 선형 그룹 RNN 연산자를 기반으로 하여 다양한 감지기 데이터(예: LiDAR와 이미지)를 통합합니다. 이 아키텍처는 토큰 수준에서의 직접적인 연결을 통해 작업을 수행하며, 이는 명시적인 융합 설계를 필요로 하지 않습니다. 이 모델은 기본적인 통합 3D 백본을 활용하여 여러 하위 작업을 동시에 처리할 수 있습니다.

- **Performance Highlights**: UniLION은 3D 객체 감지, 추적, 예측, 계획 등의 자율주행 관련 작업에서 강력한 성능을 제공합니다. 이 모델은 다양한 사전 훈련된 센서 구성 및 시간적 설정을 기반으로 신속하게 배치할 수 있어, 실험적 결과에서도 뛰어난 일반화 성능을 보여줍니다. 또한, UniLION은 간결한 디자인으로 다양한 작업을 최적화하기 위한 복잡한 모듈 의존성을 제거합니다.



### Wonder3D++: Cross-domain Diffusion for High-fidelity 3D Generation from a Single Imag (https://arxiv.org/abs/2511.01767)
Comments:
          21 pages, 19 figures, accepted by TPAMI

- **What's New**: 본 논문에서는 	extbf{Wonder3D++}를 소개하며, 이는 단일 뷰 이미지에서 고충실도 텍스처 메쉬를 효율적으로 생성하는 새로운 방법입니다. 이 방법은 기존의 Score Distillation Sampling (SDS) 기반 접근 방식의 문제를 해결하고, 고품질 결과를 위한 다중 뷰 정규 맵 및 색상 이미지를 생성할 수 있는 교차 도메인 확산 모델을 제안합니다. Wonder3D++는 정보 교환을 촉진하는 다중 뷰 교차 도메인 주의 메커니즘을 도입하여 생성 일관성을 보장합니다.

- **Technical Details**: Wonder3D++는 정상 맵(normal maps)과 색상 이미지를 동시 생성할 수 있도록 확산(framework) 모델을 확장하는 방식으로 동작합니다. 이 모델은 안정적인 확산 모델을 기반으로 하며, 교차 도메인 주의(attention) 메커니즘을 적용하여 두 도메인 간의 정보 교환을 원활하게 합니다. 또한, 다양한 입력 이미지 소스를 처리하기 위해 카메라 타입 스위처(camera type switcher)를 도입하여 다중 뷰 이미지를 생성하는 능력을 한층 강화하였습니다.

- **Performance Highlights**: 실험 결과, Wonder3D++는 Google Scanned Object 데이터셋에서 높은 품질의 재구성 결과와 강력한 일반성을 보여줍니다. 현재의 제로샷 단일 뷰 재구성 방법들 중에서 선도적인 기하학적 세부 사항을 달성하며, 효율성 또한 크게 개선하였습니다. 최종적으로, 기존의 방법들에 비해 고품질 텍스처 메쉬를 약 3분 안에 생성할 수 있는 카스케이드 3D 메쉬 추출 알고리즘을 통해 성능을 극대화하였습니다.



### HGFreNet: Hop-hybrid GraphFomer for 3D Human Pose Estimation with Trajectory Consistency in Frequency Domain (https://arxiv.org/abs/2511.01756)
- **What's New**: 이번 논문에서는 HGFreNet이라는 새로운 GraphFormer 아키텍처를 제안하여 3D 인간 포즈 추정의 성능을 향상시킵니다. 이 모델은 hop-hybrid feature aggregation과 frequency domain의 3D 경로 일관성을 통해 skeletal joint의 공간-시간 상관관계를 모델링합니다. 또한, 새로운 hop-hybrid 그래프 주의(HGA) 모듈과 transformer 인코더로 깊이의 모호성을 극복하고 3D 포즈 추정의 일관성을 높입니다.

- **Technical Details**: HGFreNet은 2D keypoint를 입력으로 사용하여 3D 포즈를 추정하는데 특화되어 있습니다. 우리는 이 모델에서 multi-hop hybrid adjacency matrix를 활용하여 넓은 수용 필드를 제공합니다. HGA 모듈은 non-parametric similarity computation (NPSC) 레이어를 통해 모든 joint feature 간의 잠재적 상호작용을 학습하며, 이 과정에서 저주파 및 고주파 성분을 이용해 데이터의 특성을 보존합니다.

- **Performance Highlights**: 실험 결과 HGFreNet은 Human3.6M 및 MPI-INF-3DHP 데이터셋에서 기존 SOTA(Sate-of-the-Art) 방법들보다 큰 개선을 보여줍니다. MPJPE는 38.8mm에서 18.9mm로 감소했으며, 이는 2D keypoints를 활용한 입력에서도 유의미한 성능 향상을 나타냅니다. 제안된 loss function으로 인해 3D 포즈 추정의 모션 진동(jitter)이 효과적으로 감소하며, 구조적 상관관계를 살펴보는 hop-hybrid attention matrices를 통해 모델의 효과성을 더욱 강화했습니다.



### 3EED: Ground Everything Everywhere in 3D (https://arxiv.org/abs/2511.01755)
Comments:
          NeurIPS 2025 DB Track; 29 pages, 17 figures, 10 tables; Project Page at this https URL

- **What's New**: 3EED는 다중 플랫폼 및 다중 모드로 구성된 3D 비주얼 그라운딩 벤치마크로, 전체 128,000개의 객체 및 22,000개의 검증된 참조 표현을 포함하고 있습니다. 이는 기존의 야외 데이터셋보다 10배 큰 규모로, 차량, 드론, 사족보행 로봇의 데이터를 통합하여 다양한 야외 환경에서 사용됩니다. 이 데이터셋은 고품질의 공간적 그라운딩을 보장하기 위해 비전-언어 모델 프롬프팅과 인간 검증을 결합한 주석 파이프라인을 개발하여 차별화됩니다.

- **Technical Details**: 3EED는 LiDAR와 RGB 데이터를 동기화하여 수집하며, 각 플랫폼에서의 기하학적 및 감각적 데이터를 표준화하는 플랫폼 인식 정규화 및 교차 모드 정렬 기술을 제안하고 있습니다. 이 데이터셋은 다중 플랫폼 및 다중 객체 그라운딩 설정을 포함하는 포괄적인 벤치마크 프로토콜을 수립하였으며, 이는 인도메인 및 크로스 플랫폼 평가를 지원합니다. 특히, 이 연구는 robust하고 일반화 가능한 3D 비주얼 그라운딩의 도전 과제를 드러냅니다.

- **Performance Highlights**: 실험 결과, 다양한 최신 모델들과 비교하였을 때 플랫폼 간에 상당한 성능 격차가 발견되었습니다. 이는 실제 야외 환경에서의 3D 비주얼 그라운딩의 복잡성을 감소시키기 위한 향후 연구 기회를 제공하며, 3EED 데이터셋이 이를 위한 중요한 기여를 할 것임을 시사합니다. 이 연구는 언어 기반 3D 체화 인지의 발전을 위해 새로운 기준을 설정하는 중요한 단계로 작용합니다.



### CGF-DETR: Cross-Gated Fusion DETR for Enhanced Pneumonia Detection in Chest X-rays (https://arxiv.org/abs/2511.01730)
- **What's New**: 이번 논문은 폐렴(pneumonia) 검출을 위해 특별히 설계된 CGF-DETR을 소개합니다. CGF-DETR은 RT-DETR의 Real-time 성능을 개선하기 위해 세 가지 혁신적인 모듈(XFABlock, SPGA, GCFC3)을 통합하여 매끄러운 멀티 스케일(feature extraction) 특별성을 강화합니다. 이러한 구조는 실시간 진단 시스템에 효과적인 성능 개선을 달성하며, 의료 영상 분야에서의 응용 가능성을 높입니다.

- **Technical Details**: CGF-DETR의 핵심 구성요소에는 XFABlock이 포함되어 있으며, 이 모듈은 CSP 아키텍처와 결합된 convolutional attention 메커니즘을 사용하여 멀티 스케일 기능 추출을 개선합니다. SPGA 모듈은 다이나믹 게이팅(dynamic gating)을 통해 입력 콘텐츠에 따라 주의를 조절하여 효율적인 feature aggregation을 제공하고, GCFC3 모듈은 멀티 경로 convolution fusion을 통해 다채로운 feature representation을 구현합니다. 이러한 구조적 변경은 병행 처리 및 실시간 성능을 유지합니다.

- **Performance Highlights**: CGF-DETR은 RSNA Pneumonia Detection 데이터셋에서 82.2% mAP@0.5를 기록하여 기존 RT-DETR-l 보다 3.7% 향상된 성능을 보였습니다. 잔여 모델의 인퍼런스 속도는 초당 48.1 프레임(FPS)에 달하며, 각 모듈의 기여도가 확인된 결과 전체 모델이 50.4% mAP@[0.5:0.95]에 도달했습니다. 이 연구는 의료 영상에서 폐렴 증상의 정확한 탐지를 위한 강력한 자동화 도구 개발의 가능성을 제시합니다.



### Toward Strategy Identification and Subtask Decomposition In Task Exploration (https://arxiv.org/abs/2511.01728)
- **What's New**: 이번 연구는 예측적 인간-기계 상호작용(anticipatory human-machine interaction) 분야에서 진행된 것으로, 기계가 사용자의 미래 상태를 예측하여 유리한 상호작용을 촉진할 수 있도록 돕는 것을 목표로 하고 있습니다. 연구의 주요 초점은 기계가 사용자의 지식, 기술 및 행동을 이해하여 암묵적 조정을(resolve implicitly coordination) 통한 상호작용을 지원하는 것입니다.

- **Technical Details**: 연구진은 클러스터링 기법(clustering techniques), 요인 분석(factor analysis), 문자열 편집 거리(string edit distance)를 사용하여 작업 탐색기(task explorer) 파이프라인을 개발했습니다. 이 파이프라인은 작업 완료에 사용되는 주요 글로벌 및 로컬 전략을 자동으로 식별하며, 글로벌 전략은 작업을 완료하기 위해 사용되는 일반화된 행동 세트를 정의하고, 로컬 전략은 이러한 행동 세트를 유사한 구성으로 사용하는 일련의 과정을 식별합니다.

- **Performance Highlights**: 실제로 작업 탐색기 파이프라인은 작업을 완료하는 데 필요한 주요 전략을 자동으로 식별하고, 사용자 실행(run)을 계층적 하위 작업 구조로 인코딩할 수 있는 기능을 갖추고 있습니다. 또한, 작업 탐색기 애플리케이션(Task Explorer application)을 개발하여 파이프라인 결과를 쉽게 검토할 수 있도록 하였으며, 이 파이프라인은 행동 기반 시간 시리즈 데이터(action-based time-series data)에 쉽게 수정이 가능합니다.



### Probabilistic Robustness for Free? Revisiting Training via a Benchmark (https://arxiv.org/abs/2511.01724)
- **What's New**: 이번 논문에서는 확률적 강인성(Probabilistic Robustness, PR)을 개선하기 위해 여러 훈련 방법의 효과를 평가하는 최초의 벤치마크인 PRBench를 도입했습니다. PRBench는 기존의 적대적 훈련(Adversarial Training, AT) 방법 및 PR-targeted 훈련 방법들을 종합적으로 비교하는 것을 목표로 합니다. 이를 통해 훈련 방법의 효율성과 성능을 이해하는 데 필요한 통합된 평가 접근법을 제공합니다.

- **Technical Details**: PRBench는 7개의 데이터셋과 10개의 모델 아키텍처에 기반하여 222개의 훈련된 모델로 구성됩니다. 이 벤치마크는 깨끗한 정확도(clean accuracy), PR 및 AR 성능, 훈련 효율성, 일반화 오차(Generalization Error, GE)와 같은 포괄적인 메트릭스를 사용하여 다양한 훈련 방법을 평가합니다. 또한 GE의 이론적 경계를 도출하고, 다양한 하이퍼파라미터 세팅에서의 성능을 분석합니다.

- **Performance Highlights**: PRBench의 주요 발견은 AT 방법이 다양한 하이퍼파라미터 설정에서 AR과 PR 성능 모두를 향상시키는 데 더 다재다능하다는 것입니다. 반면, PR-targeted 훈련 방법은 일반화 오차가 낮고 높은 깨끗한 정확도를 지속적으로 보입니다. 이 연구는 AT 방법에서 PR의 개선이 '무료로' 발생하지만, PR-targeted 방법은 더 나은 일반화와 정확도에서 장점을 제공한다고 결론짓습니다.



### Learnable Fractional Reaction-Diffusion Dynamics for Under-Display ToF Imaging and Beyond (https://arxiv.org/abs/2511.01704)
- **What's New**: 이번 연구에서는 화면 아래에 위치한 ToF 카메라를 활용하여 깊이 감지를 개선하기 위한 새로운 방법론인 Learnable Fractional Reaction-Diffusion Dynamics(LFRD2)를 제안합니다. 기존의 깊이 감지 방식에서는 TOLED층으로 인한 신호 감쇠 및 다중 경로 간섭(MPI) 등의 문제로 인해 깊이 품질이 저하되었습니다. LFRD2는 신경망의 표현력과 물리적 모델의 해석 가능성을 결합하여 깊이 최적화를 가능하게 합니다.

- **Technical Details**: LFRD2는 시간-분수 반응-확산(time-fractional reaction-diffusion) 모듈을 사용하여 깊이 재정의를 반복적으로 수행하며, 이 과정에서 동적으로 생성되는 미분 차수를 포착하여 장기 의존성을 해결합니다. 또한, 계수 예측과 반복 미분을 통해 효율적인 연속 컨볼루션(continuous convolution) 연산 기법을 도입하여 복원 품질을 더욱 개선합니다. 코드도 공개되어 있어 연구자들은 해당 작업을 재현할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 두 가지 UD-ToF 센서 및 깊이 복원 벤치마크 데이터셋으로 평가되었으며, 실험 결과 이론적 및 실험적 일관성을 입증하여 UD-ToF 이미지 품질을 효과적으로 개선함을 확인했습니다. LFRD2는 깊이 관련 작업에 대한 잠재적 확장 가능성을 제공하며, 기존의 방법들과 비교할 때 더욱 진보된 결과를 도출했습니다.



### Progressive Translation of H&E to IHC with Enhanced Structural Fidelity (https://arxiv.org/abs/2511.01698)
- **What's New**: 이번 연구에서는 기존의 면역조직화학염색(Immumohistochemistry, IHC) 기술의 비용 문제와 활용 한계를 극복하기 위해, 새로운 네트워크 아키텍처를 제안합니다. 이 아키텍처는 색상과 세포 경계 생성 논리를 포함하여, 각 시각적 요소의 최적화를 단계적으로 분리된 방식으로 수행할 수 있습니다. 이를 통해 H&E 염색 슬라이드로부터 IHC에 상응하는 이미지를 효율적으로 합성하고, 단가를 절감할 수 있습니다.

- **Technical Details**: 제안된 모델은 Adaptive Supervised PatchNCE(ASP) 프레임워크를 기반으로 하며, 3,3'-diaminobenzidine (DAB) 크로모겐 농도와 이미지 그래디언트(gradient)를 바탕으로 추가적인 손실 함수(loss function)를 도입했습니다. 이러한 접근 방식은 생성된 IHC 이미지의 색상 충실도(color fidelity)와 세포 경계의 명확성(cell boundary clarity)을 향상시키기 위해 설계되었습니다. 우리는 구성 요소 간의 상호 의존성을 고려하여, 각 단계에서 개별적으로 최적화가 이루어지도록 접근 방법을 재구성하였습니다.

- **Performance Highlights**: HER2 및 ER 데이터셋에 대한 실험 결과, 제안된 모델은 시각적 품질을 크게 개선하고 세부 구조의 표현을 더욱 정밀하게 할 수 있음을 보여주었습니다. 이 연구는 면역조직화학 이미지를 더욱 실용적으로 적용할 수 있는 혁신적인 방법론을 제공하며, 특히 자원이 제한된 설정에서도 유용할 것으로 기대됩니다. 따라서 제안된 방법은 조직 샘플 진단의 효율성을 높이고, 경량화된 대체 솔루션으로 자리 잡을 가능성이 큽니다.



### UniLumos: Fast and Unified Image and Video Relighting with Physics-Plausible Feedback (https://arxiv.org/abs/2511.01678)
Comments:
          NeurIPS 2025

- **What's New**: UniLumos는 이미지와 비디오 모두에 적용 가능한 통합 리라이트(framework)를 제공하며, RGB 공간에서의 기하학적 피드백을 플로우 매칭(backbone) 구조에 통합하여 조명 효과를 장면 구조와 명확하게 정렬합니다. 이는 물리적으로 그럴듯한 리라이트를 가능하게 하여, 복잡한 장면에서도 조명이 잘 맞도록 합니다. 또한, 이러한 피드백은 깊이(depth) 및 표면 법선(normal) 맵을 통해 감독되어, 조명이 장면 구조와 잘 어우러지도록 합니다.

- **Technical Details**: 새로운 경로 일관성 학습(path consistency learning) 기법을 채택하여, 적은 단계에서도 효과적인 감독이 가능하므로 컴퓨팅 비용을 줄화합니다. 또한, 여섯 차원의 구조화된 주석 프로토콜을 설계하여 조명의 주요 속성을 포착하며, 훈련 중 세부 조정이 용이하게 합니다. 마지막으로, LumosBench라는 디세이블된 속성 수준 벤치마크를 도입하여 조명 제어 가능성을 자동화된 방식으로 평가할 수 있게 하였습니다.

- **Performance Highlights**: UniLumos는 기존의 리라이트 품질을 뛰어넘으며, 물리적 일관성을 크게 향상시켰습니다. 이미지 및 비디오 리라이트 모두에서 20배의 속도 개선을 이루어내어, 효율성을 극대화하였습니다. 다양한 실험 결과를 통해 UniLumos의 우수한 성능이 입증되었으며, 이는 영화 제작과 augmented reality와 같은 여러 실용적 응용 분야에서 유용하게 활용될 수 있습니다.



### Enhancing Diffusion-based Restoration Models via Difficulty-Adaptive Reinforcement Learning with IQA Reward (https://arxiv.org/abs/2511.01645)
- **What's New**: 이 논문에서는 강화학습(Reinforcement Learning, RL)을 확산 모델(difussion models)의 복원(imag restoration) 작업에 효과적으로 통합하는 방법을 조사합니다. 기존의 RL 방법이 복원 모델에서는 최적이 아니라는 점에 주목하며, 높은 충실도(fidelity)를 더 중시해야 한다고 강조합니다. 특히, 이미지 품질 평가(Image Quality Assessment, IQA) 모델으로부터 유도된 보상함수를 활용하는 새로운 RL 전략을 제안하여 기존 SFT(Supervised Fine-Tuning)보다 향상된 성능을 보여줍니다.

- **Technical Details**: 논문에서 제안하는 RL 전략은 두 가지 주요 역할을 수행합니다: 1) 모델을 대안 솔루션으로 안내하여 분포 수준에서 정렬을 이끄는 탐색(exploration)과 2) 목표와 더 잘 정렬되도록 하는 활용(exploitation)입니다. 이 전략은 훈련 샘플의 상대적인 난이도를 기반으로 동적으로 가중치를 조정하는 자동화된 메커니즘을 통해 조정됩니다. 또한, IQA 기반의 RL 전략을 주로 어려운 샘플에 적용하고, 모델의 출력을 더욱 세밀하게 찾아내기 위해 SFT와 결합하는 방식을 사용합니다.

- **Performance Highlights**: 다양한 데이터셋에서의 광범위한 실험을 통해 제안된 RL 프레임워크의 효과성을 입증하였습니다. 결과적으로, 기존의 확산 기반 복원 네트워크에 플러그 앤 플레이 방식으로 쉽게 적용할 수 있으며, 성능 향상에 기여합니다. 이는 강력한 복원 성능을 제공하며, 특히 고충실도의 이미지 복원에서 돋보이는 결과를 나타냅니다.



### Actial: Activate Spatial Reasoning Ability of Multimodal Large Language Models (https://arxiv.org/abs/2511.01618)
- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)의 공간 추론 능력을 평가하고 개선하기 위해 Viewpoint Learning이라는 새로운 작업을 도입합니다. 특히, Viewpoint-100K 데이터셋을 소개하여 10만 개의 객체 중심 이미지 쌍과 관련된 질문-답변 쌍을 확보하였습니다. 이 데이터셋은 MLLMs가 3D 공간에서의 연속성을 이해하는 데 도움을 줄 수 있는 기초 정보 제공을 목표로 하고 있습니다.

- **Technical Details**: 우리는 MLLMs의 공간 추론 능력을 활성화하기 위해 두 단계의 fine-tuning 전략을 제안합니다. 첫 번째 단계에서는 Supervised Fine-Tuning (SFT)을 통해 Viewpoint-100K 데이터셋을 활용하여 기초 지식을 주입하고, 두 번째 단계에서는 강화 학습(Reinforcement Learning)을 사용하여 더 높은 일반화 능력을 갖는 모델을 개발합니다. 이를 통해 모델은 3D 공간에서의 시각적 관계와 변환을 이해할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식이 MLLMs의 공간 추론 능력을 크게 향상시키며, 다양한 벤치마크에서 성능이 개선됨을 보여주었습니다. 특히, 우리의 접근 방식은 도메인 외 추론 작업에서의 일반화 능력을 증명하며, 로봇 공학 및 자율 시스템과 같은 복잡한 애플리케이션에서 3D 이해를 위한 거대한 잠재력을 제시합니다.



### Vote-in-Context: Turning VLMs into Zero-Shot Rank Fusers (https://arxiv.org/abs/2511.01617)
- **What's New**: 본 논문은 Vote-in-Context (ViC)라는 일반화된 훈련 없는 프레임워크를 소개합니다. ViC는 Vision-Language Model (VLM)을 사용하여 리스트 단위의 재순위를 수행하고, 복잡한 비디오와 같은 다중 모드 데이터를 효과적으로 통합할 수 있도록 설계되었습니다. 이 프레임워크는 콘텐츠 증거와 검색기 메타데이터를 직접 VLM의 프롬프트에 직렬화하여 모델이 적응적으로 일관성을 평가할 수 있도록 합니다.

- **Technical Details**: ViC는 S-Grid라는 압축된 콘텐츠 직렬화 맵을 도입하여 각 비디오를 이미지 그리드로 표현합니다. 이 구조는 VLM에 의해 해석될 수 있는 콘텐츠 증거를 제공하며, 비디오 후보들에 대한 리스트 단위의 추론을 가능하게 합니다. ViC는 두 가지 모드에서 작동하며, 첫 번째는 단일 리스트 재순위 모드로, 두 번째는 여러 검색기를 조합하는 앙상블 융합 모드입니다.

- **Performance Highlights**: ViC는 영상 검색 벤치마크에서 기존의 최고 성능을 크게 능가하는 결과를 보여 주었습니다. ViC는 MSR-VTT에서 87.1%(t2v), 89.0%(v2t)의 Recall@1을 기록했으며, VATEX에서는 99.6%(v2t)를 기록했습니다. 이러한 결과는 이전의 저명한 기준 대비 최대 +40 Recall@1의 향상과 같은 상당한 이득을 나타냅니다.



### Benchmark-Ready 3D Anatomical Shape Classification (https://arxiv.org/abs/2511.01613)
Comments:
          Shape in Medical Imaging, ShapeMI 2025, Held in Conjunction with MICCAI 2025

- **What's New**: 이 연구에서는 3D 해부학적 형태 분류 분야에서 새로운 벤치마크 데이터셋인 MedShapeNet19와 더불어 Precomputed Structural Pooling (PSPooling)이라는 새로운 메쉬 풀링 방법을 소개합니다. MedShapeNet19는 기존의 MedShapeNet에서 파생된 19개의 해부학적 클래스를 포함하고 있으며, 각 클래스는 임상 이미징 데이터에서 재구성된 800개의 표면 메쉬 샘플로 이루어져 있습니다. 또한, PSPooling은 기하학적 근접성을 기반으로 한 노드 대응 집합을 미리 계산하여 3D 해부학적 형태 분석에서 효율성과 구조 보존을 지원하는 비학습적 메쉬 풀링 연산자입니다.

- **Technical Details**: PSPooling은 그래프의 구조적 다운샘플링을 지원하며, 이는 잠재 벡터로 변환된 메쉬가 노드의 기하학적 관계를 유지하도록 합니다. 기존의 선택 기반 메소드들이 구조적 문제에 직면하는 데 반해, PSPooling은 병렬화 및 가역성을 지원하여 3D 메쉬의 복잡한 복원 문제를 해결합니다. 추가적으로, 본 연구는 GNN 인코더를 이용한 자기 감독 학습 절차를 통해 구조적 인지 기능을 학습하고, 이후 이를 통해 메쉬 수준의 의미 분류를 수행하는 방법론을 제시합니다.

- **Performance Highlights**: 실험 결과, PSPooling을 이용한 접근 방식은 낮은 라벨 비율에서도 복원 충실도와 분류 정확도를 유의미하게 개선함을 보였습니다. MedShapeNet19를 기반으로 한 모델은 의료 3D 형태 학습을 위한 강력한 기준선을 설정하였으며, 향후 의료 3D 형태 분석 및 해부학적 형태 분류 연구의 널리 채택된 벤치마크로 자리 잡기를 기대하고 있습니다. 이러한 성과는 의료 이미지 처리와 내재된 기하학적 특성을 이해하기에 중요한 기초 자료가 될 것입니다.



### DINO-MX: A Modular & Flexible Framework for Self-Supervised Learning (https://arxiv.org/abs/2511.01610)
- **What's New**: 이번 논문은 DINO-MX라는 모듈형 훈련 프레임워크를 소개합니다. DINO-MX는 DINO, DINOv2 및 DINOv3의 핵심 원칙을 통합하여 다양한 transformer 기반 아키텍처를 지원하며 Hugging Face 생태계와 완벽하게 호환됩니다. 이 프레임워크는 저차원 적응(LoRA), 레이어 고정 및 지식 증류와 같은 다양한 훈련 전략을 포함하며, 분산 훈련을 위해 DDP 및 FSDP를 지원합니다.

- **Technical Details**: DINO-MX는 자연 데이터와 전문 데이터 타입 모두에서 작동하도록 설계되었습니다. 실험 결과 다양한 데이터셋에서 DINO-MX는 경쟁력 있는 성능을 달성하면서도 계산 비용을 크게 줄이는 성과를 보여줍니다. 또한, 해석 가능한 도구와 레이블 기반 데이터 증대 방법을 통해 추가적인 탐지나 세분화 헤드 없이 주의기반 로컬리제이션을 향상시킬 수 있습니다.

- **Performance Highlights**: DINO-MX는 의료 이미징 및 다양한 연구 및 실제 응용 프로그램에서 자기 지도 학습 모델을 개발하고 적응 및 벤치마킹할 수 있는 재현 가능하고 확장 가능한 기반을 제공합니다. 이 프레임워크는 많은 리소스가 부족한 상황에서도 사용할 수 있으며, 연구자들이 최신 오픈 소스 모델을 빠르게 비교하고 평가할 수 있도록 돕는 인프라를 제공합니다.



### Lite ENSAM: a lightweight cancer segmentation model for 3D Computed Tomography (https://arxiv.org/abs/2511.01600)
- **What's New**: Lite ENSAM은 RECIST 주석이 달린 CT 스캔에서 효율적인 종양 분할을 위해 특별히 설계된 경량화된 ENSAM 아키텍처입니다. 이 연구는 RECIST의 한계에 대한 대안을 제공하며, 종양의 볼륨 측정을 자동화하여 임상적 효용성을 높이는 데 기여합니다. 연구진은 MICCAI FLARE 2025 Task 1에 참여하여 종양 분할에서 60.7%의 Dice Similarity Coefficient와 63.6%의 Normalized Surface Dice라는 성과를 달성했습니다.

- **Technical Details**: Lite ENSAM 아키텍처는 3D U-Net 구조를 기반으로 하며, 종양의 RECIST 주석과 연결된 SAM 스타일의 주의(attention) 메커니즘을 포함합니다. 이 메커니즘은 Lie Rotational Positional Encoding을 활용하여 U-Net의 병목(bottleneck)에서의 상호작용을 최적화합니다. 입력 볼륨이 CPU에서 최대 8GB RAM의 제한 내에서 처리될 수 있도록 설계되어 있으며, 메모리 및 계산 효율성을 위해 다양한 수정이 이루어졌습니다.

- **Performance Highlights**: Lite ENSAM은 공개 검증 데이터셋에서 평균 50.6GB의 RAM 사용량과 14.4초의 추론 시간을 기록했습니다. 또한 최종 예측에서 모든 클래스를 포함하는지 확인하기 위해 post-processing 단계에서 여러 최적화가 이루어졌으며, 이는 처리 속도를 30배 향상시켰습니다. 이러한 성과는 다양한 임상 환경에서의 종양 분할 정확성을 증가시키는 데 기여할 것으로 기대됩니다.



### Wave-Particle (Continuous-Discrete) Dualistic Visual Tokenization for Unified Understanding and Generation (https://arxiv.org/abs/2511.01593)
- **What's New**: 이번 논문에서는 연속적(continuous)과 이산적(discrete) 시각 토큰화(tokenization) 사이의 간극을 해결하기 위해 새로운 Continuous-Discrete Dualistic Visual Tokenizer (CDD-VT)를 제안합니다. CDD-VT는 이미지 시각 데이터를 양자화된 코드북을 이용하여 원시 이미지 조합으로 처리하며, 각 비주얼 샘플의 복잡성에 따라 할당되는 원시 수를 동적으로 결정하여 정보 손실을 최소화합니다. 이는 기존의 토크니저들이 가진 성능 저하 문제를 효과적으로 해결할 수 있는 가능성을 보여줍니다.

- **Technical Details**: CDD-VT는 비전 인코더, 텍스트 인코더 및 비전 디코더로 구성되며, 두 가지 핵심 구성 요소인 Diverse Quantitative Primitives (DQP)와 Dynamic Primitive Allocator (DPA)를 포함합니다. DQP는 정보 공간을 효과적으로 채우기 위해 원시의 직교성을 촉진하고, DPA는 각 샘플의 복잡성을 평가하여 최적의 원시 집합을 동적으로 결정합니다. 이러한 과정을 통해 CDD-VT는 신속하고 효율적인 단일 모델로서 비주얼 이해 및 생성 기능을 결합할 수 있습니다.

- **Performance Highlights**: CDD-VT는 이미지 분류, 이미지-텍스트 검색 및 이미지 재구성 등 세 가지 전통적인 벤치마크에서 시도된 Extensive experiments를 통해 매우 우수한 성능을 기록했습니다. 특히, CDD-VT는 ImageNet에서 0.31의 재구성 FID와 256×256 해상도에서 70.5%의 제로 샷(Zero-shot) Top-1 정확도를 달성했습니다. 이러한 성과는 CDD-VT가 이산 및 연속 토크나이저에 비해 경쟁력 있는 성능을 보여주며, 스케일 가능성을 유지할 수 있음을 시사합니다.



### Generative Adversarial Synthesis and Deep Feature Discrimination of Brain Tumor MRI Images (https://arxiv.org/abs/2511.01574)
Comments:
          9 pagers, 8 Figures

- **What's New**: 이 논문은 Deep Convolutional Generative Adversarial Network (DC-GAN)을 활용하여 MRI 데이터의 부족 문제를 해결하기 위한 합성 의료 이미지 생성 방법론을 제안합니다. DC-GAN 모델을 통해 생성된 이미지는 실제 MRI 데이터와 함께 뇌 종양 분류를 위한 CNN(Convolutional Neural Network) 분류기의 교육 및 평가에 사용됩니다. 이는 GAN이 생성한 이미지가 실제 데이터와 동등한 성능을 발휘할 수 있음을 검증하여 의료 이미지 분석의 효용성을 시사합니다.

- **Technical Details**: 이 연구에서는 DC-GAN을 통해 합성 MRI 이미지를 생성하고, 이를 CNN 모델과 결합하여 이미지 분류 작업을 수행합니다. DC-GAN은 실제 뇌 MRI 스캔에서 학습한 생성자와 구분자로 구성되며, 생성자는 현실적인 MRI 이미지를 생성하고, 구분자는 실제 이미지와 합성 이미지를 구분하는 역할을 담당합니다. 이러한 방식은 합성 이미지가 분류 성능에 미치는 영향을 평가하는 데 사용됩니다.

- **Performance Highlights**: 실험 결과는 GAN이 생성한 이미지가 실제 MRI 이미지에 비해 비교 가능한 성능을 보여줌을 나타냅니다. CNN 분류기를 통해 얻은 성능 지표들은 합성 데이터와 실제 데이터가 동일한 효용을 가진다는 점을 강조합니다. 이러한 결과는 합성 데이터가 AI 기반 진단 및 의료 연구 애플리케이션의 신뢰할 수 있는 대안이 될 수 있음을 시사합니다.



### PixelVLA: Advancing Pixel-level Understanding in Vision-Language-Action Mod (https://arxiv.org/abs/2511.01571)
Comments:
          17pages,7 figures, 5 tabels

- **What's New**: 이번 논문에서 소개된 PixelVLA는 pixel-level 이해를 지원하고 텍스트 및 시각적 입력을 통한 멀티모달 프롬프트(multimodal prompting)를 처리할 수 있도록 설계된 최초의 비전-언어-행동 모델입니다. 기존 VLA 모델이 가지는 픽셀 수준의 장면 이해 부족과 텍스트 프롬프트 의존성 문제를 해결하고자 하였습니다. 이 모델은 멀티스케일(pixel-aware) 인코더와 시각적 프롬프트 인코더를 통합하여 새로운 비주얼 모터 프레임워크를 기반으로 구축되었습니다.

- **Technical Details**: PixelVLA의 아키텍처는 사전 학습된 비전-언어 모델을 백본으로 사용하고, 다양한 시각적 프롬프트를 처리하기 위한 경량의 시각적 프롬프트 인코더와 픽셀 수준 이해를 위한 새로운 멀티스케일 인코더를 포함합니다. 또한, 연속 행동 표현 디코더를 통해 VLM의 은닉 상태를 바탕으로 미세한 행동 세부사항을 캡처할 수 있습니다. 훈련을 위한 Pixel-160K 데이터셋은 자동 주석 파이프라인을 통해 생성되었으며, 이를 통해 로봇 데이터에서 파생된 픽셀 수준 주석이 포함됩니다.

- **Performance Highlights**: PixelVLA는 OpenVLA에 비해 조작 성공률을 10.1%에서 17.8%까지 개선하면서도 필요로 하는 사전 훈련 비용은 1.5%에 불과하다는 실험 결과를 보여줍니다. 세 가지 표준 VLA 벤치마크와 두 가지 VLA 모델 변형에서 광범위한 평가를 통해 PixelVLA는 복잡한 환경에서 보다 정확하고 효율적인 로봇 제어를 가능하게 함을 입증하였습니다. 논문에서 제안하는 데이터셋과 코드는 오픈 소스로 제공될 예정입니다.



### NOA: a versatile, extensible tool for AI-based organoid analysis (https://arxiv.org/abs/2511.01549)
- **What's New**: 새로운 연구에서 저자들은 Napari Organoid Analyzer (NOA)를 소개합니다. NOA는 AI 기반의 오르간이드 이미지 분석을 용이하게 하는 일반 목적의 그래픽 사용자 인터페이스(GUI)입니다. 이 도구는 탐지(detection), 분할(segmentation), 추적(tracking), 특성 추출(feature extraction), 주석(annotation) 및 ML 기반 예측을 통합하여 사용자 친화적이면서도 다용도입니다.

- **Technical Details**: NOA는 RGB, RGBA 및 그레이스케일 이미지와 타임랩스를 처리할 수 있는 포괄적인 워크플로우를 제공합니다. 본 도구는 Faster R-CNN, YOLOv3, SSD 및 RTMDet 기반의 사전 학습된 탐지 모델을 포함하고 있으며, 사용자 정의 탐지 모델도 제공합니다. SAM과 DETR 기반의 탐지 헤드를 활용하여 정확한 분할을 지원하며, 사용자가 직접 탐지 데이터를 수정할 수 있도록 매뉴얼 조정을 지원합니다.

- **Performance Highlights**: NOA는 실제 실험을 통해 다양한 형태의 분석을 성과로 보여주었습니다. 여러 실험을 통해 오르간이드의 기하학적 특성을 정량화하고, 모orphological 변화 및 생존 가능성을 예측할 수 있음을 입증하였습니다. NOA는 기존의 프로그래밍 지식 없이도 AI 기반 분석을 가능하게 하여 생물학자들이 오르간이드 이미지를 효율적으로 분석할 수 있도록 지원합니다.



### PCD-ReID: Occluded Person Re-Identification for Base Station Inspection (https://arxiv.org/abs/2511.01546)
Comments:
          11 pages, 7 figures

- **What's New**: 이 논문에서는 기초 기반 스테이션 환경에서의 가려진 보행자 재식별(Occluded pedestrian re-identification, ReID) 문제를 다룹니다. 기존의 ResNet 기반 ReID 알고리즘이 가려짐 문제를 효과적으로 해결하지 못하는 문제점을 지적하며, 새로운 PCD-ReID (Pedestrian Component Discrepancy) 알고리즘을 제안합니다. 이 알고리즘은 헬멧 및 유니폼과 같은 공통 구성 요소 특징을 추출할 수 있는 Transformer 기반 네트워크로 설계되었습니다.

- **Technical Details**: PCD-ReID는 공공 데이터셋에서의 과적합(overfitting)을 줄이기 위해 6개월 동안 10,000명의 실시간 순찰 감시 이미지로 모델을 훈련시켰습니다. 총 50,000장 이상의 이미지를 포함하여, 실세계 데이터셋을 활용하여 알고리즘의 성능을 향상시켰습니다. 이 연구는 특히 타워 검사 시나리오에서 사람의 가려진 상태를 인식할 수 있는 ReID 성능을 실현합니다.

- **Performance Highlights**: 실험 결과, PCD-ReID는 기존 알고리즘들과 비교하여 평균 평균 정밀도(mean Average Precision, mAP) 79.0% 및 Rank-1 정확도 82.7%를 달성했습니다. 이는 ResNet50 기반 방법에 비해 Rank-1 정확도에서 15.9% 개선된 수치입니다. 이 결과는 감시 및 보안 애플리케이션에서의 실용적인 배치를 위한 가능성을 강조합니다.



### Driving scenario generation and evaluation using a structured layer representation and foundational models (https://arxiv.org/abs/2511.01541)
- **What's New**: 이 논문은 자율주행 차량 개발을 위한 희귀한 주행 시나리오의 평가 및 생성을 개선하기 위해 5계층 구조 모델을 제안합니다. 기존의 계층 모델을 기반으로 하여 각 에이전트의 하위 클래스와 특성을 도입함으로써 새로운 주행 시나리오를 생성하는 데이터 증강 전략을 사용합니다. 본 연구는 희귀한 시나리오의 개념을 명확히 하기 위해 Edge Cases (ECs)라는 용어를 사용하며, 이를 통해 주행 데이터셋의 다양성과 독창성을 평가할 수 있는 새로운 메트릭스를 제안합니다.

- **Technical Details**: 이 논문은 5계층 모델(5-layer model)을 사용하여 주행 시나리오를 효과적으로 표현합니다. 이 모델은 도로 구조, 도로 주변의 구조, 임시 변화, 동적 객체, 환경 조건의 다섯 가지 계층으로 구성되어 있어 주행 시나리오에 대한 표준화된 표현을 가능하게 합니다. 텍스트 표현은 벡터 임베딩(vector embedding)으로 변환되어 시나리오 간 유사성을 비교할 수 있는 기반을 제공합니다.

- **Performance Highlights**: 생성된 희귀 주행 시나리오는 기존 실제 데이터에서 특정 구성요소를 편집하여 Augmentation 방법을 통해 만들어집니다. 이를 통해 시나리오가 실제와 일치할 가능성을 높이며, 우리는 텍스트 기반 평가 방식을 통해 생성된 시나리오의 품질을 분석합니다. 다양한 실험 결과를 통해 이 모델의 효용성을 입증하며, 코드와 확장된 결과는 논문 내 제공된 URL에서 확인할 수 있습니다.



### NSYNC: Negative Synthetic Image Generation for Contrastive Training to Improve Stylized Text-To-Image Translation (https://arxiv.org/abs/2511.01517)
Comments:
          Under review

- **What's New**: 이 논문에서는 텍스트 기반 이미지를 생성하는 기존 방법들의 한계를 극복하기 위한 새로운 대비 학습(framework) 구조를 제안합니다. 특히, 이 구조는 특정 스타일을 더 잘 표현할 수 있도록 도와주는 데 중점을 두고 있습니다. 기존의 방법들은 일반적인 스타일 생성에 성공적이었지만, Monet와 같은 세부적인 스타일을 구현하는 데에는 필요한 미세한 특징들을 포착하지 못했습니다. 본 연구는 이러한 문제를 해결하기 위한 혁신적인 접근을 제시하고 있습니다.

- **Technical Details**: 제안된 방법은 대조 손실(contrastive loss)을 활용하여 세부적인 스타일 특징과 일반적인 스타일 특징을 구분하도록 합니다. 이를 위해 긍정적 데이터와 부정적 데이터를 모두 사용하는 학습 과정을 통하여 긍정적 데이터의 기울기를 정교하게 업데이트합니다. 또한, 본 연구는 부정적 합성 데이터(negative synthetic data)를 생성하여 모델이 더 특화된 스타일을 학습할 수 있도록 합니다. 이러한 접근 방식은 기존에 사용된 데이터와 가장 큰 차별점을 가진 방식을 통해 구현됩니다.

- **Performance Highlights**: 이 연구에서 제안된 방식은 다양한 화가 및 일러스트레이터의 스타일을 포착하는 데에 성공했습니다. 실험 결과, 제안된 방법이 기준 모델보다 정량적 및 정성적으로 우수한 성능을 보였습니다. 본 연구의 기법은 일반적인 화풍보다 더 세부적인 예술적 변화를 잘 구현함으로써, 실제로 비슷한 스타일의 이미지를 생성하는 데 효과적임을 입증했습니다.



### Example-Based Feature Painting on Textures (https://arxiv.org/abs/2511.01513)
Comments:
          "\c{opyright} 2025 Andrei-Timotei Ardelean, Tim Weyrich. This is the author's version of the work. It is posted here for your personal use. Not for redistribution. The definitive Version of Record was published in ACM Trans. Graph., Vol. 44, No. 6, this https URL

- **What's New**: 이 연구에서는 독특한 지역적 특성을 가진 텍스처를 제어된 방식으로 작성 및 편집하기 위한 완전한 워크플로우를 제안합니다. 자연계에서 흔히 발견되는 오염, 찢어짐, 구멍, 마모, 변색 등 여러 텍스처 효과를 포함하는 것이 중요합니다. 사용자에 의한 수동 주석 없이 학습 기반 접근법을 사용하여 이러한 결함을 가진 텍스처 생성 방법을 도입하고 있습니다.

- **Technical Details**: 우리의 접근법은 비지도(anomaly detection) 이상 탐지를 통해 외관 변화 특징을 감지하는 방법입니다. 다양한 텍스처 특성은 의미론적으로 일관된 그룹으로 자동 클러스터링되어, 이미지의 조건부 생성(conditional generation)을 안내하는 데 사용됩니다. 최종 파이프라인은 소규모 이미지 수집부터 시작하여 다양한 크기의 텍스처에 특징을 대화식으로 생성하고 그리는 데 사용될 수 있는 범용 생성 모델로 발전합니다.

- **Performance Highlights**: 특히, 우리는 확산 기반(diffusion-based) 편집 알고리즘과 무한 정지 텍스처 생성을 위한 알고리즘을 도입하였습니다. 이러한 알고리즘은 일반적(generic)이며 다른 맥락에서도 유용할 것으로 기대됩니다. 이 연구는 현실적인 텍스처 생성을 위한 새로운 가능성을 열어줄 것입니다.



### Luminance-Aware Statistical Quantization: Unsupervised Hierarchical Learning for Illumination Enhancemen (https://arxiv.org/abs/2511.01510)
Comments:
          Accepted at NeurIPS 2025

- **What's New**: 본 논문은 저조도 이미지 향상(LLIE)의 새로운 접근법인 Luminance-Aware Statistical Quantification (LASQ)을 소개합니다. 이는 기존의 픽셀 수준 매핑을 대체하여, 계층적 조도 분포 위에서의 통계적 샘플링 프로세스로 LLIE를 재구성합니다. LASQ는 조도 전환을 확률 분포로 모델링하여, 물리적 현실성을 반영하는 새로운 기법을 수립했습니다.

- **Technical Details**: LASQ는 조도 지역성의 변화를 power-law 분포로 모델링하며, Markov Chain Monte Carlo (MCMC) 샘플링 전략을 통해 조도 적응 연산자를 체계적으로 탐색합니다. 이는 기존의 정해진 관계에 얽매이지 않고, 지속적이고 맥락에 적합한 조도 전환을 수용하는 과정으로, 무감독 학습을 통해 진행됩니다. 또한, LASQ는 초기 조도를 기반으로 하며 가우시안 형태의 분포를 따릅니다.

- **Performance Highlights**: LASQ는 비참조 데이터셋에서 최첨단 성능을 달성하며, 정상 조명이 포함된 벤치마크 데이터셋에서는 비슷한 성능을 기록합니다. 이 접근법은 참조 기반 또는 비참조 시나리오에서 모두 뛰어난 결과를 보여, 다양한 환경에서 조명 복원 능력을 극대화하는 데 기여합니다. 실험을 통해 LASQ는 기존 방식들보다도 뛰어난 일반화 능력을 입증하였습니다.



### Discriminately Treating Motion Components Evolves Joint Depth and Ego-Motion Learning (https://arxiv.org/abs/2511.01502)
Comments:
          18 pages, 14 figures

- **What's New**: 이번 연구는 깊이(depth)와 자아 모션(ego-motion)을 효율적으로 모델링하기 위한 새로운 접근법인 DiMoDE를 제안합니다. 기존 방법들이 자아 모션을 보조 작업으로 다루는 한계를 극복하고, 모션 구성 요소를 구별하여 각각의 기하학적 규칙을 활용하여 깊이와 자아 모션 추정을 동시에 수행합니다. 특히, 각 기하학적 흐름에 대한 제약을 개별적으로 수립하여 다중 영상 프레임을 보다 정확하게 처리할 수 있도록 하였습니다.

- **Technical Details**: 이 연구에서는 영상 프레임 간의 광학 축(optical axis)과 이미지 평면(imaging plane)을 정렬하여 각 자아 모션 구성 요소에 대해 기하학적 제약(geometric constraints)을 부과하였습니다. 특히, 회전(rotation)과 두 가지 변환(tangential and radial translations) 요소를 분리하여 학습하며, 이를 통해 네트워크 출력이 구별된 기하학적 흐름을 생성하도록 유도합니다. 이러한 접근 방식은 기본적인 기하학적 관계를 통해 DepthNet과 PoseNet의 학습 과정을 개선합니다.

- **Performance Highlights**: DiMoDE는 여러 공공 데이터셋과 새로운 현실 세계 데이터셋에서 최신 성능을 기록하며, 특정한 도전적인 상황에서도 탁월한 내구성을 보여줍니다. 주장한 바와 같이, 각 모션 구성 요소를 독립적으로 학습하는 방식이 깊이 추정(deep estimation)과 시각적 측정(visual odometry)에서 성과를 극대화함을 입증했습니다. 또한, DiMoDE는 다양한 모델과 호환 가능하며, 네트워크 수렴성과 학습의 안정성을 크게 향상시켰습니다.



### SE(3)-PoseFlow: Estimating 6D Pose Distributions for Uncertainty-Aware Robotic Manipulation (https://arxiv.org/abs/2511.01501)
- **What's New**: 이 논문에서는 6D 물체 자세 추정의 문제를 해결하기 위해 SE(3) 매니폴드에서 플로우 매칭을 이용한 새로운 확률론적 프레임워크를 제안합니다. 기존의 방법들이 단일 결정론적 출력을 회귀하는 반면, 이 접근 방식은 샘플 기반의 자세 분포 추정을 모델링하여 불확실성을 추론할 수 있습니다. 이 연구는 복잡한 분포를 다루는 데 필요한 확률론적 모델링의 중요성을 강조합니다.

- **Technical Details**: 제안된 방법은 RGB-D 입력에서 6D 물체 자세 분포를 샘플 기반으로 추정합니다. 객체는 Mask R-CNN이나 CNOS와 같은 탐지기를 사용하여 로컬라이징되고, 객체 중심의 RGB 크롭과 부분 포인트 클라우드가 추출됩니다. 이러한 관측값은 기하학적 및 시각적 인코더를 통해 인코딩되고, DiT⋆ 블록과 결합되어 SE(3) 매니폴드에서 조건부 플로우 매칭을 수행합니다.

- **Performance Highlights**: 제안된 방법은 Real275, YCB-V 및 LM-O에서 최첨단 성능을 달성하며, 불확실한 시점의 변별력을 높이기 위한 능동적 인식(active perception)이나 불확실성을 고려한 그랩 합성을 위한 로봇 조작 작업에서도 활용될 수 있음을 보여줍니다. 이 연구는 SE(3) 분포를 통한 안정적이고 효과적인 단일 시점 그랩 생성 가능성을 입증하여 로봇 조작의 안전성과 신뢰성을 향상시킵니다.



### EPAN: Robust Pedestrian Re-Identification via Enhanced Alignment Network for IoT Surveillanc (https://arxiv.org/abs/2511.01498)
Comments:
          12 page, 5 figures

- **What's New**: 이번 연구에서는 간섭이 많은 IoT 환경에서의 개인 재식별(Person ReID)을 위해 강화된 보행자 정렬 네트워크(Enhanced Pedestrian Alignment Network, EPAN)를 소개합니다. EPAN은 다양한 관점과 환경 변화의 영향을 줄이기 위해 이중 분기 아키텍처(dual-branch architecture)를 활용하여, 개인 재식별의 안정성을 극대화합니다.

- **Technical Details**: EPAN은 다양한 스케일(scale)과 관점(viewpoint)에서 정렬 정보를 추출하여, 복잡한 IoT 감시 조건에서도 견고한 성능을 발휘할 수 있습니다. 이 네트워크는 최신의 기능 추출(mechanism)을 기반으로, 다양한 상황에서 신뢰할 수 있는 결과를 제공합니다.

- **Performance Highlights**: EPAN의 성능은 Inspection-Personnel 데이터셋에서 Rank-1 정확도 90.09%와 평균 평균 정밀도(mean Average Precision, mAP) 78.82%를 기록하며 뛰어난 결과를 보여줍니다. 이는 EPAN이 실제 IoT 응용에 적합함을 강조하며, 다양한 카메라 환경에서 효과적인 인물 재식별을 가능하게 합니다.



### SecDiff: Diffusion-Aided Secure Deep Joint Source-Channel Coding Against Adversarial Attacks (https://arxiv.org/abs/2511.01466)
Comments:
          13 pages, 6 figures

- **What's New**: 이번 논문에서는 SecDiff라는 새로운 diffusion 기반의 디코딩 프레임워크를 설계하여 적대적 무선 환경에서 딥 JSCC의 보안성과 강인성을 크게 향상시킵니다. 이 프레임워크는 기존의 높은 추론 지연 문제를 해결하기 위해 의사역행렬 기반 샘플링과 적응형 가중치를 활용하여 유연한 단계 크기 조절을 가능하게 합니다. 또한, 서브캐리어 재밍 공격을 방어하기 위해 전력 기반의 서브캐리어 마스킹 전략을 소개하고, 채널 추정을 블라인드 역문제(blind inverse problem)로 형성하여 안전한 디코딩을 지원하는 EM 기반 정제 알고리즘을 개발했습니다.

- **Technical Details**: SecDiff는 적대적 간섭에 대해 저항하는 방식으로 내구성을 향상시키는 의사역행렬 기반 샘플링을 사용합니다. 샘플링 과정에서 노이즈를 줄이기 위한 단계 건너뛰기 샘플링(step-skipping sampling)을 적용하여 고해상도 복원을 이루고 지연 시간을 상당히 줄입니다. 더욱이, 복원 작업을 마스킹된 인페인팅 문제(masked inpainting problem)로 설정하고, 조합 신호와 채널 복구를 이룰 수 있는 메커니즘을 도입하여 추론 오류를 최소화합니다.

- **Performance Highlights**: 제안된 SecDiff는 재밍 환경에서 PSNR(피크 신호 대 잡음 비율)을 4.4% 향상시키고, 인식 왜곡을 30% 이상 감소시킵니다. 조작된 파일럿 공격 하에서도, SecDiff는 벤치마크 방법의 성공률을 두 배로 증가시키고 PSNR을 19.3% 개선하며, 36.1% 낮은 FID(Fréchet Inception Distance)를 기록하여 실용적인 저지연 및 공격 저항성 있는 의미론적 통신 프레임워크로 자리매김했습니다.



### HMVLM: Human Motion-Vision-Lanuage Model via MoE LoRA (https://arxiv.org/abs/2511.01463)
Comments:
          10 pages, 5figures. The Thirty-Ninth Annual Conference on Neural Information Processing Systems

- **What's New**: 본 논문에서는 Human Motion-Vision-Language Model (HMVLM)이라는 새로운 통합 프레임워크를 제안합니다. 이 프레임워크는 Mixture of Expert Low-Rank Adaptation (MoE LoRA) 전략을 기반으로 하여 다양한 다운스트림 작업에서의 성능 개선을 목표로 합니다. 특히, 모션 관련 작업에서의 지식 잊음을 방지하기 위한 새로운 'zero expert' 개념이 도입되었습니다.

- **Technical Details**: HMVLM은 게이팅 네트워크를 활용하여 입력 프롬프트에 따라 LoRA 전문가 가중치를 동적으로 할당합니다. 이는 여러 작업의 동기화된 파인튜닝(fine-tuning)을 가능하게 하고, 동작과 관련 없는 작업에서 pretrained 파라미터를 보존합니다. 또한 인체를 부위별로 나누어 토큰화하여 공간적 해상도를 향상시키는 방법이 적용되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 HMVLM은 모션 관련 작업에서 강력한 성능을 보여주며, 텍스트-모션 생성, 단안 포즈 추정 및 모션 비디오 이해에서 우수한 성과를 달성했습니다. 지식 잊힘 현상이 효과적으로 완화되었으며, 다양한 인체 중심의 작업을 동시에 지원할 수 있는 안정적인 멀티모달 프레임워크로 자리매김하였습니다.



### Efficiently Training A Flat Neural Network Before It has been Quantizated (https://arxiv.org/abs/2511.01462)
Comments:
          ongoing work, more results would be added

- **What's New**: 본 논문에서는 포스트 훈련 양자화(Post-training quantization, PTQ)에서 발생하는 양자화 오차를 줄이기 위한 새로운 접근 방법인 차별적 노이즈 기반 양자화 인식 훈련(Differential Noise-driven Quantization-aware Training, DNQ)을 제안합니다. 기존의 방법들은 완전 정밀 모델과 양자화 모델 간의 관계를 간과하여 상당한 양자화 오차를 초래합니다. DNQ는 모델의 오류 원인을 측정하고 분리하여 안정적인 손실 경관을 유도할 수 있도록 설계되었습니다.

- **Technical Details**: DNQ 프레임워크는 양자화 오차, 즉 활성화 양자화 오차(Activation Quantization Error, AQE)와 가중치 양자화 오차(Weight Quantization Error, WQE)를 독립적인 가우시안 노이즈로 통계적으로 모델링합니다. 이 프레임워크는 훈련 중에 양자화 노이즈를 시뮬레이션하여 모델이 훈련될 수 있도록 하고, 최적화자가 단순한 PTQ에 대해 견고한 성능을 보이는 솔루션을 찾아내도록 유도합니다. 결과적으로, 손실 경관의 평탄함을 최적화하여 양자화에 내성이 강한 모델을 만들 수 있습니다.

- **Performance Highlights**: 실험 결과, DNQ가 적용된 모델은 기존의 복잡한 PTQ 알고리즘으로 최적화된 모델을 능가하는 성능을 보입니다. 특히, 여러 기준 데이터셋과 네트워크 아키텍처에서 일관되게 성능을 향상시켰습니다. 본 논문의 접근 방식은 향후 저비트 PTQ 모델을 획득하기 위한 새로운 방향을 제시하며, 특히 모델 압축을 필요로 하는 엣지 컴퓨팅 장치에 유용할 것입니다.



### When to Trust the Answer: Question-Aligned Semantic Nearest Neighbor Entropy for Safer Surgical VQA (https://arxiv.org/abs/2511.01458)
- **What's New**: 이 논문은 수술에서의 Visual Question Answering (VQA)의 안정성과 신뢰성을 개선하기 위해 불확실성 추정에 초점을 맞춥니다. QA-SNNE(Question Aligned Semantic Nearest Neighbor Entropy)라는 새로운 불확실성 추정기를 도입하여 질문의 의미를 포함한 예측 신뢰도를 측정합니다. 기존 surgical VQA 연구는 주로 정확도와 언어 품질에 집중했으나, 본 연구는 사용자 안전을 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: QA-SNNE는 의료 텍스트 임베딩 공간에서 생성된 답변과 가장 가까운 이웃을 비교하여 의미적 엔트로피를 측정합니다. 이를 통해 주어진 질문에 대한 답변의 통일성을 평가하며, 과거 연구에서 사용된 불확실성 기반 접근 법과는 다르게 질문에 맞춤형 분석을 수행합니다. 다섯 가지 모델을 평가하며, PEFT 모델과 zero-shot LVLM을 포함하여 다양한 사전 학습된 모델에 대해 실험을 진행합니다.

- **Performance Highlights**: QA-SNNE는 다양한 테스트 환경에서 구현되었으며, 특히 out-of-template 상황에서 PEFT 모델이 약간 손상되는 반면 LVLM은 더 높은 내구성을 보입니다. AUROC(Receiver Operating Characteristic 곡선 아래 면적)는 zero-shot 모델에서 15-38% 증가하여 안전성을 높이는 데 기여합니다. 이 연구는 QA-SNNE를 통해 수술 VQA의 안전성과 신뢰성을 높일 수 있는 실질적이고 해석 가능한 단계로 제시합니다.



### Reg-DPO: SFT-Regularized Direct Preference Optimization with GT-Pair for Improving Video Generation (https://arxiv.org/abs/2511.01450)
- **What's New**: 이 연구는 비디오 생성 품질을 향상시키기 위한 새로운 직접 선호 최적화 방법인 GT-Pair와 Reg-DPO를 제안합니다. GT-Pair는 실비디오와 생성된 비디오를 통해 고품질의 선호 쌍을 자동으로 생성하며, Reg-DPO는 SFT 손실을 DPO 목표에 정규화 항으로 포함시켜 훈련 안정성과 생성 충실도를 높입니다. 이러한 접근방법은 메모리 최적화 기법과 결합되어 훈련 능력을 최대 세 배 향상시킵니다.

- **Technical Details**: GT-Pair는 실비디오를 긍정 샘플로, 모델 생성을 부정 샘플로 이용하여 별도 외부 주석 없이 높은 품질의 선호 데이터를 생성합니다. Reg-DPO는 DPO 손실에 SFT 기반 정규화 항을 포함시켜, 훈련 중 발생하는 불안정성을 해결하고 모델 성능을 향상시킵니다. 이 연구는 Flash Attention, Context Parallelism과 같은 여러 메모리 최적화 기법을 결합하여 GPU 메모리 사용량을 줄이는 효과적인 메모리 절약 전략을 구현합니다.

- **Performance Highlights**: I2V(이미지에서 비디오)와 T2V(텍스트에서 비디오) 작업에서 광범위한 실험을 수행한 결과, 제안된 방법이 기존 방법들보다 일관되게 우수한 비디오 생성 품질을 달성한 것으로 나타났습니다. SFT 정규화는 긍정 및 부정 샘플의 출력 분포를 동시에 감독하여 모델이 선호 신호를 학습하도록 하면서도 분포 이동을 통제합니다. 제안된 접근 방식은 현재까지의 연구에서 가장 뛰어난 성과를 기록하며, 모델의 훈련과정에서 안정성을 크게 향상시킵니다.



### Privacy Preserving Ordinal-Meta Learning with VLMs for Fine-Grained Fruit Quality Prediction (https://arxiv.org/abs/2511.01449)
Comments:
          9 pages, 1 figure, 4 tables

- **What's New**: 이 논문에서는 과일의 신선도를 비침습적으로 예측하기 위한 새로운 방법인 Model-Agnostic Ordinal Meta-Learning (MAOML) 알고리즘을 소개합니다. 기존의 독점적인 Vision Language Models (VLMs)와는 달리, 이 접근법은 적은 양의 데이터로도 높은 성능을 낼 수 있도록 설계되었습니다. 제안된 방법은 다양한 과일에 대해 92.71%의 정확도로 최신 기술을 구현했으며, 데이터 프라이버시 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: MAOML 알고리즘은 메타 학습(meta-learning) 기법을 활용하여 과일의 신선도 분류 작업에서 데이터 부족 문제를 해결합니다. 이를 통해 소규모 VLM 모델을 훈련시키며, 라벨의 순서(ordinality)를 활용하여 성능을 개선합니다. 본 연구에서는 10종의 과일에 대해 고유한 품질 클래스인 'Unripe', 'Early ripe', 'Ripe', 'Overripe', 'Bad'를 정의하고, 이를 기반으로 300개의 이미지를 수집하였습니다.

- **Performance Highlights**: MAOML을 사용하여 훈련된 소규모 VLM 모델은 기존의 대형 VLM보다 성능이 뛰어난 것으로 나타났습니다. 제안된 방법은 zero-shot 및 few-shot 상황 모두에서 높은 정확성을 기록하며, 실험 결과는 이 방법이 음식 소매 산업에서의 신선도 판단을 실용적으로 적용하는 데 큰 도움이 될 것임을 시사합니다. 이를 통해 데이터 프라이버시 및 성능 간의 균형을 유지하는 효과적인 AI 솔루션이 가능해집니다.



### Contrast-Guided Cross-Modal Distillation for Thermal Object Detection (https://arxiv.org/abs/2511.01435)
- **What's New**: 이 논문은 열 적외선(thermal-infrared) 탐지에서 야간의 강건한 인식(Robust perception) 문제를 다룹니다. 기존의 접근 방식은 TIR(thermal infrared)을 RGB로 변환하거나 두 센서를 결합하는 방법이었으며, 이는 저항성(resilience)이 부족하다는 단점이 있었습니다. 저자들은 서로 다른 클래스의 피처를 분리하고 같은 클래스의 피처를 가까이 모으는 새로운 학습 목표를 도입하여 이러한 문제를 해결했습니다.

- **Technical Details**: 제안된 방법은 단일 모달리티(single-modality) 추론을 유지하며, 학습 중에 인스턴스(instance) 수준에서 결정 경계를 날카롭게 하는 방식을 사용합니다. 이는 중복(detection) 및 혼동(class confusion) 감지를 억제하고, RGB로 학습된 teacher 모델과 학생 모델의 다중 수준 피라미드(multi-level pyramid) 피처를 일치시켜 교차 모달 의미(priors)를 주입합니다. 이 과정을 통해 텍스처가 부족한 열 특성을 강화하면서 테스트 시간에 시각적 입력이 필요하지 않도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 이전의 접근 방식들을 초월하여 혁신적인 성능 향상을 보여주었습니다. 특히 열 탐지기에서 최첨단(state-of-the-art) 성능을 달성하였으며, 이는 다양한 조건에서의 탐지 문제를 효과적으로 해결했음을 의미합니다. 이 연구는 열 탐지 분야에서 새로운 가능성을 제시하고 있습니다.



### Terrain-Enhanced Resolution-aware Refinement Attention for Off-Road Segmentation (https://arxiv.org/abs/2511.01434)
- **What's New**: 이번 연구에서는 오프로드(Off-road) 의미 분할(semantic segmentation)의 주요 문제점인 두껍고 일관성이 없는 경계( boundaries), 희귀 클래스에 대한 희박한 감독(sparse supervision), 그리고 광범위한 라벨 노이즈를 해결하기 위해 새로운 "해상도 인식 토큰 디코더(resolution-aware token decoder)"를 제안합니다. 이는 불완전한 감독 하에서도 전역 의미(global semantics), 지역 일관성(local consistency), 경계 충실도(boundary fidelity)를 균형 있게 유지하는 특징을 지니고 있습니다.

- **Technical Details**: 이 모델은 주로 저해상도 병목(low-resolution bottleneck)에서 연산이 이루어지며, 게이티드 크로스 어텐션(gated cross-attention)을 통해 세부 사항을 주입합니다. 드문 픽셀들만 불확실성을 기반으로 선택하여 세분화(refinement)합니다. 글로벌 셀프 어텐션(global self-attention)과 경량의 확장(depthwise) 정제를 통해 지역 일관성을 복원하고, 표준 고해상도 인코더 스트림(high-resolution encoder stream)에서 세부 기능을 통합하여 노이즈를 증가시키지 않도록 합니다.

- **Performance Highlights**: 전체적인 실험 결과는 경쟁력 있는 성능(competitive performance)과 함께 변환(transitions) 간의 안정성(improved stability)을 개선한 것으로 나타났습니다. 경계 근처의 일관성 있는 예측을 유도하는 경계 대역 일관성 정규화(boundary-band consistency regularizer) 또한 주목할 만한 특징이며, 이 과정은 추론(inference) 시간에는 추가 비용 없이 이루어집니다.



### UniSOT: A Unified Framework for Multi-Modality Single Object Tracking (https://arxiv.org/abs/2511.01427)
Comments:
          The paper has been accepted by TPAMI

- **What's New**: 이번 논문에서는 다양한 참조 모드 (reference modalities)와 비디오 모드 (video modalities) 조합을 처리할 수 있는 통합 추적기 (unified tracker), UniSOT을 제안합니다. 이 모델은 기존의 특정 모드에 제한되지 않고, 블록 박스 (bounding box), 자연어 (natural language) 및 두 가지 모두를 사용하여 비디오 모드에 걸쳐 목표 객체를 추적할 수 있습니다. UniSOT은 18개의 시각 추적 (visual tracking) 및 비전-언어 추적 (vision-language tracking) 벤치마크에서 뛰어난 성과를 보여 주었습니다.

- **Technical Details**: UniSOT는 참조 모드 및 비디오 모드를 통합하기 위해 설계된 두 가지 모듈을 포함합니다. 참조 모드 설계를 위해, 우리는 Transformer 기반의 새로운 구조를 통해 시각 및 언어의 특징을 결합하는 범용 피처 추출기 (feature extractor)를 개발하였습니다. 비디오 모드 설계를 위해서는, 다양한 보조 비디오 모드에서 강력하게 작동할 수 있도록 랭크 조정 모드 적응 메커니즘 (rank-adaptive modality adaptation)을 적용하였습니다.

- **Performance Highlights**: 실험 결과, UniSOT는 모든 비디오 모드 및 참조 모드에 대해 이전의 통계적 대비 결과보다 약 3.0% AUC 향상을 달성하였습니다. 특히, RGB+X 비디오 모드에서 Un-Track을 초과하여 약 2.0% 더 우수한 주요 메트릭을 보였습니다. 이를 통해 UniSOT는 다양한 사용자의 요구 사항을 만족시키며 실질적인 응용 프로그램에 유연성을 제공합니다.



### Towards One-step Causal Video Generation via Adversarial Self-Distillation (https://arxiv.org/abs/2511.01419)
Comments:
          Under double-blind review as a conference paper

- **What's New**: 본 논문에서는 효율적인 인과 비디오 생성(compositional video generation)을 위한 새로운 증류(distillation) 기반 프레임워크를 제안합니다. 이를 통해 제한된 디노이징 단계로도 고품질 합성을 가능하게 하며, 기존의 iterative한 접근 방식으로 인한 에러 누적 및 긴 추론 시간을 극복하고자 합니다. 또한, Adversarial Self-Distillation(ASD) 전략을 통해 학생 모델의 출력과 디노이징 프로세스를 더 매끄럽고 일관되게 맞출 수 있는 방법을 개발하였습니다.

- **Technical Details**: 저자는 Distribution Matching Distillation(DMD) 프레임워크에 기반하여, n-단계 디노이징 과정의 출력을 n+1-단계 버전과 분포 레벨에서 정렬하는 새로운 전략인 ASD를 제안합니다. 이를 통해 디노이징 단계가 적을 때 발생하는 품질 저하 문제를 개선하고, 초기에 더 많은 디노이징 단계를 배분하는 First-Frame Enhancement(FFE) 전략을 도입하였습니다. 이 방법은 전반적으로 낮은 계산 비용으로 비디오 품질을 높이는 데 기여합니다.

- **Performance Highlights**: VBench에서의 광범위한 실험 결과, 제안된 방법은 1단계 및 2단계 비디오 생성 모두에서 기존 최첨단 기술보다 우수한 성능을 달성했습니다. 특히, 단일 증류 모델을 사용하여 다양한 추론 설정을 지원하여 효율성과 유연성을 동시에 확보했습니다. 이를 통해 다양한 리소스 환경에서도 하이 퀄리티 비디오 합성이 가능하게 되었습니다.



### Extremal Contours: Gradient-driven contours for compact visual attribution (https://arxiv.org/abs/2511.01411)
- **What's New**: 최근의 연구는 시각 모델에 대한 신뢰할 수 있으면서도 간결한 설명을 제공하는 것이 여전히 도전 과제가 되고 있음을 보여줍니다. 본 논문에서는 기존의 밀집한 perturbation masks 대신 매끄러운 조정 가능한 윤곽선을 사용하는 훈련 없는 설명 방법을 제안합니다. 이 방법은 저차원의 구조적 표현을 사용하여 신뢰성과 해석성을 강화하면서도 제어 가능한 면적과 함께 전반적인 정확성을 유지합니다.

- **Technical Details**: 제안된 방법은 매끄러운 star-convex mask로 설명을 나타내며, 각 지역은 배움 가능한 중심 위치를 기준으로 하는 truncated Fourier expansion으로 매개변수화됩니다. 이를 통해 기존의 방법보다 한두 단계 적은 자유 매개변수를 가지며, 데이터 세트 최적화 없이 안정적인 수렴을 보장합니다. 이 접근 방식은 픽셀 최적화의 복잡성과 불안정성을 피하면서도 perturbation 기반 방법의 신뢰성을 유지합니다.

- **Performance Highlights**: 이 방법은 두 가지 사전 학습된 분류기(ResNet-50 및 DINO ViT-B/16)를 사용하여 평가되었으며, 여러 지표에서 동작을 평가합니다. 기존의 설명 방법들과 비교할 때, 저자들의 방법은 더 간결하고 안정적인 경계를 생성하여 객체를 더 잘 둘러싸며, 15% 이상의 관련성 증가를 달성합니다. 또한, 이 접근법은 알려진 XAI(masins for Explainable AI) metric을 통해 더 높은 신뢰성과 낮은 복잡성을 달성하고, 다중 객체 식별에도 성공적으로 확장됩니다.



### Semantic BIM enrichment for firefighting assets: Fire-ART dataset and panoramic image-based 3D reconstruction (https://arxiv.org/abs/2511.01399)
- **What's New**: 이 연구는 화재 자산 관리의 효율성을 높이기 위해 Fire-ART 데이터 세트를 소개합니다. 이 데이터 세트는 2,626개의 이미지와 6,627개의 인스턴스를 포함하여 15개의 기본 자산을 아우르는 방대한 데이터입니다. 이는 화재 자산 인식을 위한 공개 데이터 세트로, 기존의 비효율적인 방법을 극복하는 데 도움을 줍니다.

- **Technical Details**: Fire-ART 데이터 세트는 자산 인식과 재구성을 위한 혁신적인 방법론을 제공합니다. 재구성 접근법은 수정된 큐브 맵 변환(modified cube-map conversion)과 반지름 기반 구형 카메라 투영(radius-based spherical camera projection)을 통합하여 인식과 위치 파악의 정확성을 높입니다. 이를 통해 데이터는 BIM 모델로의 의미론적 강화(semantic enrichment)를 가능하게 합니다.

- **Performance Highlights**: 제안된 방법은 두 개의 실제 사례 연구를 통해 검증되었으며, F1 점수는 각각 73%와 88%에 도달했습니다. 또한 위치 오류는 0.620미터와 0.428미터로, 높은 정확성을 자랑합니다. Fire-ART 데이터 세트와 재구성 방법은 화재 안전 장비의 디지털 관리를 개선하기 위한 귀중한 자원과 강력한 기술 솔루션을 제공합니다.



### SEPS: Semantic-enhanced Patch Slimming Framework for fine-grained cross-modal alignmen (https://arxiv.org/abs/2511.01390)
- **What's New**: 본 논문에서는 정밀한 비전과 언어 간의 로컬 대응관계를 형성하는 세분화된 크로스 모달 정렬(fine-grained cross-modal alignment)을 소개합니다. 기존 방법들이 직면하고 있는 패치 중복(patch redundancy)과 모호성(ambiguity) 문제를 해결하기 위해, 논문에서는 의미 향상 패치 슬리밍(Semantic-Enhanced Patch Slimming, SEPS) 프레임워크를 제안합니다. SEPS는 밀집 텍스트(dense text)와 희박 텍스트(sparse text) 간의 통합된 의미를 활용하여 중요한 시각적 패치를 식별하는 데 도움을 줍니다.

- **Technical Details**: SEPS 프레임워크는 두 단계 메커니즘을 통해 구성됩니다. 첫 번째 단계에서는 밀집 텍스트와 희박 텍스트의 특징 표현을 활용하여 시각적 패치를 추출하고, 두 번째 단계에서는 희박 텍스트와 밀집 텍스트에서 파생된 통합된 의미를 바탕으로 시각적 패치를 선택합니다. 이를 통해 HEPA(Highly-Relevant Patch-Word Alignment) 모듈이 적용되어, 중요한 패치-단어 대응관계의 선택 및 평균값 계산을 통해 정밀한 인터랙션을 개선합니다.

- **Performance Highlights**: Flickr30K 및 MS-COCO 데이터셋에서 SEPS 프레임워크를 평가한 결과, 기존 접근 방식을 23%-86% 초과하는 성능 개선을 이루었으며, 특히 텍스트-이미지 검색 시나리오에서 두드러진 향상을 보였습니다. 이 연구는 MLLMs를 활용하여 크로스 모달 정렬을 위한 패치 선택을 효과적으로 지원하는 첫 번째 체계적인 프레임워크로 자리매김하고 있습니다.



### EREBUS: End-to-end Robust Event Based Underwater Simulation (https://arxiv.org/abs/2511.01381)
Comments:
          Accepted to ICRA AQUA2SIM Workshop 2025, 6 pages, 3 figures, conference paper

- **What's New**: 이 논문은 수중 환경에서 AUV(Autonomous Underwater Vehicle)에 장착된 이벤트 기반 카메라의 현실적인 합성 데이터를 생성하는 파이프라인을 소개합니다. 전통적인 비전 기술이 낮은 가시성 환경에서 성능이 저하되는 문제를 해결하는 혁신적인 접근 방식을 제안하며, 이 데이터를 활용해 비전 모델을 훈련할 수 있는 방법을 제시합니다. 실험을 통해 바위 탐지 작업에서 이벤트 기반 비전 기술의 효과를 시연했습니다.

- **Technical Details**: 연구에서는 Blender를 이용한 합성 수중 환경을 구축한 후, 이벤트 카메라 시뮬레이터를 통해 비동기적으로 이벤트 스트림을 생성합니다. 이 방식은 각 픽셀의 밝기 변화에 따라 트리거되는 변화를 기록하여 시간순으로 정렬된 이벤트 시퀀스를 생성합니다. 또한, YOLO(You Only Look Once) 모델을 사용하여 적은 수의 레이블 데이터로도 효과적인 객체 탐지 및 분할을 가능하게 하는 방법론을 제안합니다.

- **Performance Highlights**: 학습 세트가 작은데도 불구하고 YOLOv8 모델이 10개의 이미지를 기반으로 훈련되어 0.83의 평균 정밀도(mAP)를 기록했습니다. 이러한 결과는 이벤트 데이터를 활용하여 최소한의 레이블링 노력으로도 강건한 인식 모델을 구축할 수 있는 가능성을 보여줍니다. 또한 논문의 기초 작업을 통해 오픈 소스 도구 및 데이터셋을 제공하고, 수중 이벤트 기반 로봇 기술 발전을 지원하겠다는 의지를 밝히고 있습니다.



### CMI-MTL: Cross-Mamba interaction based multi-task learning for medical visual question answering (https://arxiv.org/abs/2511.01357)
Comments:
          The paper has been accepted by the 33rd Pacific Conference on Computer Graphics and Applications (Pacific Graphics 2025)

- **What's New**: 이번 연구에서는 의료 분야의 시각적 질문 답변(Med-VQA)을 위한 새로운 Cross-Mamba Interaction 기반의 다중 작업 학습(CMI-MTL) 프레임워크를 제안합니다. 기존의 self-attention 방법이 존재하는 한계성을 극복하고, 이미지와 텍스트 간의 cross-modal semantic alignments를 효과적으로 처리하기 위한 접근 방식입니다. CMI-MTL은 이미지와 텍스트로부터 교차 모달 특징 표현을 학습하며, 이를 위해 FVTA, CIFR, FFAE의 세 가지 핵심 모듈로 구성되어 있습니다.

- **Technical Details**: CMI-MTL은 FVTA(Fine-grained Visual-Text Feature Alignment), CIFR(Cross-Modal Interleaved Feature Representation), FFAE(Free-Form Answer-Enhanced Multi-Task Learning) 모듈로 구성됩니다. FVTA는 이미지-텍스트 쌍에서 가장 관련성이 높은 지역을 추출하고, CIFR은 cross-modal 상호작용을 동적으로 포착합니다. FFAE는 개방형 질문에서 보조 지식을 활용하여 모델의 개방형 Med-VQA 능력을 향상시키는 역할을 합니다.

- **Performance Highlights**: 실험 결과, CMI-MTL은 VQA-RAD, SLAKE, OVQA의 세 가지 Med-VQA 데이터셋에서 기존의 최첨단 방법들을 초월한 성능을 보여주었습니다. 또한 Grad-CAM을 사용한 해석 가능성 실험을 통해 모델이 질문과 가장 관련 있는 이미지 영역에 집중하고 있음을 입증했습니다. 이러한 결과는 CMI-MTL의 효과성과 해석 가능성을 높이는 데 기여합니다.



### Expanding the Content-Style Frontier: a Balanced Subspace Blending Approach for Content-Style LoRA Fusion (https://arxiv.org/abs/2511.01355)
- **What's New**: 최근 텍스트-이미지 확산 모델에서의 발전은 생성되는 이미지의 개인화 및 스타일화에 큰 진전을 가져왔습니다. 하지만 기존 연구는 단일 스타일 강도에서의 콘텐츠 유사성만 평가해왔습니다. 저희 연구에서는 스타일 강도의 증가가 콘텐츠 특징의 손실을 초래함을 관찰하였고, 이에 따라 콘텐츠-스타일 경계를 확장하기 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 저희의 접근 방식은 Content-Style Subspace Blending과 Content-Style Balance Loss를 활용하여 개발되었습니다. 이를 통해 다양한 스타일 강도에서 콘텐츠 유사성을 향상시키고, 콘텐츠-스타일 경계를 효과적으로 확장할 수 있습니다. 실험 결과, 우리의 방법은 기존 기술보다 질적 및 양적 평가 모두에서 우수한 성능을 보이며, Inverted Generational Distance (IGD)와 Generational Distance (GD) 점수에서 현저히 낮은 값을 기록하였습니다.

- **Performance Highlights**: 광범위한 실험을 통해 저희 방법이 콘텐츠-스타일 경계를 상당히 확장함을 입증하였습니다. 정량적 실험 결과, 저희 방법이 IGD와 GD 메트릭에서 기존 기준선을 초과하는 것으로 나타났습니다. 또한, 저희 방법은 콘텐츠 보존과 스타일 표현 모두에서 향상된 결과를 보여주었습니다.



### MIQ-SAM3D: From Single-Point Prompt to Multi-Instance Segmentation via Competitive Query Refinemen (https://arxiv.org/abs/2511.01345)
- **What's New**: 이 논문에서는 다중 객체(instance)를 동시에 처리할 수 있는 MIQ-SAM3D 세분화 프레임워크를 제안합니다. 이 프레임워크는 단일 점 프롬프트(single-point prompt)를 통해 여러 분리된 객체 인스턴스를 세분화하는 방법을 혁신적으로 변환합니다. 또한, CNN과 Transformer의 장점을 통합하여 보다 정밀한 세분화 결과를 제공합니다.

- **Technical Details**: MIQ-SAM3D의 구조는 하이브리드 CNN-Transformer 인코더와 Prompt-Conditioned Instance Query Generator(PC-IQG) 모듈로 구성됩니다. 이 인코더는 고주파 상세 정보를 추출하는 CNN과 글로벌 컨텍스트를 모델링하는 ViT를 결합하여 사용합니다. 또한, Competitive Query Refinement Decoder(CQRD)는 질의들이 서로 경쟁하며 최적화되도록 하여 세분화 품질을 높입니다.

- **Performance Highlights**: MIQ-SAM3D는 LiTS17 및 KiTS21 데이터셋에서 검증되었으며, 여러 인스턴스 세분화 작업에서 유사한 성능을 달성했습니다. 이 모델은 프롬프트에 대한 높은 강인성을 보여주며, 임상적으로 중요한 다중 병변 케이스의 효율적인 주석 작업에 실용적인 솔루션을 제공합니다.



### $\left|\,\circlearrowright\,\boxed{\text{BUS}}\,\right|$: A Large and Diverse Multimodal Benchmark for evaluating the ability of Vision-Language Models to understand Rebus Puzzles (https://arxiv.org/abs/2511.01340)
Comments:
          7 pages, 5 figures, 4 tables

- **What's New**: 본 논문에서는 1,333개의 다양한 영어 레버스 퍼즐(Rebus Puzzles)로 구성된 새로운 벤치마크인 $|\circlearrowright\boxed{BUS}|$를 소개합니다. 이 퍼즐은 음식, 관용구, 스포츠, 금융 등 18개 카테고리에 걸쳐 다양한 예술적 스타일과 난이도를 포함하고 있어, 현재 비전-언어 모델은 이 과제를 해결하는 데 도전하고 있습니다.

- **Technical Details**: 우리는 $RebusDescProgICE$라는 모델에 구애받지 않는 프레임워크를 제안합니다. 이 프레임워크는 비구조적인 설명과 코드 기반의 구조적 추론을 결합하여, 비전-언어 모델의 성능을 크게 향상시킵니다. 개선된 예제 선택 전략을 통해 모델 성능을 높이는 이 방법은 복잡한 추론 과정이 필요한 레버스 퍼즐의 해법에 특히 효과적입니다.

- **Performance Highlights**: 본 연구에서 제안된 방식은 비전-언어 모델의 성능을 $|\circlearrowright\boxed{BUS}|$에서 2.1-4.1% 및 20-30% 향상시켰습니다. 이는 Chain-of-Thought Reasoning 기법과 비교할 때 두드러진 성과입니다.



### RDTE-UNet: A Boundary and Detail Aware UNet for Precise Medical Image Segmentation (https://arxiv.org/abs/2511.01328)
- **What's New**: 본 논문은 의료 영상 분할의 정확성과 세부 정보 보존을 향상시키기 위해 RDTE-UNet라는 새로운 세그멘테이션 네트워크를 제안합니다. RDTE-UNet는 지역적 모델링과 글로벌 컨텍스트를 통합하여 경계 선명도 및 세부 사항 보존을 강화합니다. 이 모델은 Adaptive Shape-aware Boundary Enhancement (ASBE), Horizontal–Vertical Detail Attention (HVDA), 및 Euler Feature Fusion (EulerFF)라는 세 가지 모듈로 구성되어 있습니다.

- **Technical Details**: RDTE-UNet는 의료 영상의 복잡한 형태를 정확하게 세분화하기 위해 ResBlock과 Detail Transformer 블록을 통합하여 경계와 세부 사항의 구획을 강화합니다. ASBE는 Adapative Rectangular Convolution (ARConv)을 활용하여 대상 형태에 적절히 적응하여 멀티 스케일 표현과 차별화된 경계 향상을 가능하게 합니다. HVDA 모듈은 의료 영상에서 세부 정보와 미세 구조를 강조하여 인식을 개선하고, EulerFF는 동적 가중치를 통해 멀티 스케일 인코더-디코더 기능을 융합하여 정보 손실을 최소화합니다.

- **Performance Highlights**: 테스트 데이터셋인 Synapse와 BUSI에서 RDTE-UNet는 기존 최첨단 방법들과 비교하여 정확도 및 세부 정보 보존에서 우수한 성능을 달성했습니다. RDTE-UNet는 복잡한 토폴로지 구조에 대해서도 더 완전하고 정확한 분할을 제공합니다. 이러한 결과는 RDTE-UNet가 각기 다른 방향, 형태, 크기에서 구조의 일관성과 경계 정확성을 개선함을 보여줍니다.



### A Generative Adversarial Approach to Adversarial Attacks Guided by Contrastive Language-Image Pre-trained Mod (https://arxiv.org/abs/2511.01317)
Comments:
          18 pages, 3 figures

- **What's New**: 본 논문은 CLIP 모델을 활용하여 시각적으로 고무감이 없는 강력한 adversarial perturbations를 생성하는 생성적 적대 공격 방법을 제안합니다. 이는 다중 객체 환경에서 다중 레이블 분류기를 속이도록 특별히 설계된 perturbations를 생성하여, 모델의 예측의 정확성을 저하시킬 수 있습니다. 이 접근법은 기존의 기술들과 유사하거나 우수한 성능을 보여주며, 높은 시각적 유사성을 유지합니다.

- **Technical Details**: 이 논문에서는 Saliency-based Auto-Encoder (SSAE)의 집중적 perturbation 전략과 Generative Adversarial Multi-Object Scene Attacks (GAMA)와 유사한 비유사 텍스트 임베딩을 통합한 방법론을 사용합니다. 모델은 다양한 블랙박스 피해 모델을 대상으로 다수의 테스트를 통해 이러한 perturbation이 원본 이미지와의 높은 구조적 유사성을 유지하면서도 분류 모델을 속일 수 있음을 입증했습니다. 그러므로, 이 방법은 기존의 gradient-based 및 non-gradient-based 공격 방법의 한계를 극복하는데 기여합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 작업에서 기존의 기술들과 비교하여 경쟁력 있는 성능을 보여줍니다. 특히, 시각적 충실도가 더 높은 상태에서 다중 객체 이미지를 효과적으로 처리할 수 있는 능력을 확인했습니다. 따라서 본 연구는 CNN 기반 모델의 보안 및 신뢰성을 향상시키는데 중요한 기여를 할 것으로 기대됩니다.



### MVSMamba: Multi-View Stereo with State Space Mod (https://arxiv.org/abs/2511.01315)
Comments:
          Accepted by NeurIPS 2025

- **What's New**: 이번 논문에서는 Mamba 아키텍처를 기반으로 한 첫 번째 다중 뷰 스테레오 네트워크인 MVSMamba를 제안합니다. MVSMamba는 그리드 스캔을 통해 전 방향 특징을 효율적으로 집계하여, 계산 오버헤드 없이도 뛰어난 성능을 발휘할 수 있도록 설계되었습니다. 이 네트워크는 동적 Mamba 모듈(Dynamic Mamba module, DM-module)을 통해 참조 이미지와 여러 소스 이미지 간의 특징 상호작용을 최적화할 수 있습니다.

- **Technical Details**: MVSMamba는 각 참조-소스 특징 쌍에 대해 소스 특징을 참조 특징의 위, 아래, 왼쪽, 오른쪽에 연결하여 공간적으로 효율적인 특징 스캐닝을 지원합니다. 이 방식은 다중 뷰 특징의 표현을 최적화하며, 깊이 맵을 추정하는 과정에서 FPN(Feature Pyramid Network) 내 다중 스케일 집합을 통해 긴 거리 의존성을 포착합니다. 이는 MVS 작업을 위한 기존의 CNN 및 Transformer 방식의 한계를 극복하는 데 중점을 두고 있습니다.

- **Performance Highlights**: MVSMamba는 DTU 데이터 세트와 Tanks-and-Temples 벤치마크에서 최신 MVS 방법들보다 뛰어난 성능과 효율성을 보여주었습니다. 전반적으로 MVSMamba는 기존의 방식들과 비교하여 성능과 계산 효율성에서 훌륭한 균형을 보이고 있습니다. 이러한 성과는 구현된 소스 코드에 기반하여 더 넓은 연구의 기초가 될 것입니다.



### Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models (https://arxiv.org/abs/2511.01307)
Comments:
          26 pages, 9 figures, 16 tables, NeurIPS 2025

- **What's New**: 이번 논문에서는 Anti-Personalized Diffusion Models (APDM)이라는 새로운 프레임워크를 제안하여 특정 주제의 개인화 기능을 차단하는 것을 목표로 합니다. 기존의 데이터-중심 방법들과 달리, APDM은 이미지가 아닌 확산 모델 자체를 보호 대상으로 설정합니다. 이를 통해 개인화의 무단 시도를 방지하고, 생성 성능은 유지할 수 있습니다.

- **Technical Details**: APDM의 구현에서는 Direct Protective Optimization (DPO)이라는 새로운 손실 함수를 도입하여 개인화 과정을 방해합니다. 또한, Learning to Protect (L2P)이라는 이중 경로 최적화 전략을 통해 개인화 경로와 보호 경로를 번갈아 가며 학습하도록 설계되었습니다. 이 방법은 모델이 개인화 시도를 예측하고 이에 따라 적응적으로 보호 조치를 강화할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, APDM은 다양한 개인화 주제에서 무단 개인화를 방지하는데 있어 기존 방법들을 능가하며 최신 성과를 기록했습니다. 즉, APDM은 실세계 환경에서 개인화로 인한 위협을 효과적으로 차단하며, 그 성능 또한 유지됩니다. 이러한 결과는 APDM의 효과성을 강화하는 데 중요한 증거가 됩니다.



### Positive Semi-definite Latent Factor Grouping-Boosted Cluster-reasoning Instance Disentangled Learning for WSI Representation (https://arxiv.org/abs/2511.01304)
Comments:
          Our code is available at this https URL

- **What's New**: 이 논문은 WSI(Whole Slide Image) 분석을 위한 새로운 세 가지 단계로 구성된 프레임워크인 PG-CIDL(Positive semi-definite latent factor grouping-boosted Cluster-reasoning Instance Disentangled Learning)을 제안합니다. 이를 통해 공간적, 의미적 및 결정적 얽힘(entanglement)을 해결하고 해석 가능한 표현을 제공합니다. 이를 위해 새로운 양의 준정치(latent factor grouping) 방법을 도입하고, 클러스터 기반 사고 방식을 통해 각 인스턴스의 기여를 명확히 하며, 최종적으로 결정 간의 얽힘을 줄입니다.

- **Technical Details**: 제안된 방법론은 세 가지 가정에 기반하며, 각 WSI 내에서 종양(tumor), 미세환경(microenvironment), 배경(noise) 요인이 공존할 수 있다는 점을 고려합니다. MIL(Multiple Instance Learning) 프레임워크에서 정상적인 점수를 학습하기 위해 permutation-invariant scoring function을 사용합니다. 이 과정에서 기존의 의미적 및 결정적 얽힘 문제를 해결하기 위해 구조적 인과 모델(structural causal model)을 구성해 개별 인스턴스의 의미를 명확히 합니다.

- **Performance Highlights**: PG-CIDL 모델은 다수의 실험을 통해 기존의 최첨단 기술들보다 우수한 성능을 보이며, 병리학자들이 이해할 수 있는 해석성을 제공합니다. 이는 클러스터 기반의 의미 분해와 결정적 기여 개선 덕분입니다. 논문에서 제안한 방법은 WSI의 해석 가능성을 넘어서는 효과를 발휘하며, 실제 진단 작업에서 의미 있는 결과를 보일 것으로 기대됩니다.



### REASON: Probability map-guided dual-branch fusion framework for gastric content assessmen (https://arxiv.org/abs/2511.01302)
Comments:
          Under Review. 12 pages, 10 figures, 6 tables

- **What's New**: 이 논문에서는 일반 마취 시 흡입 위험의 감별을 위해 위 내용 평가를 자동화하는 새로운 프레임워크(REASON)를 제안합니다. REASON은 두 단계로 나뉘며, 첫 번째 단계에서는 segmentation model이 확률 맵(probability map)을 생성하여 위 해부학을 강조합니다. 두 번째 단계에서는 두 개의 뷰를 통합하여 분류 성능을 높이는 dual-branch fusion classifier를 사용합니다. 이 방법론은 기존 최첨단 기법들에 비해 상당한 차별성을 보이며, 임상 실습에서의 활용 가능성이 높습니다.

- **Technical Details**: REASON 프레임워크는 우선 픽셀 수준의 확률 맵을 생성하여 위의 영역을 강조하고 노이즈를 억제하는 방식으로 작동합니다. 첫 번째 단계에서는 mean-teacher 방식의 반자율 학습을 통해 segmentation model이 훈련됩니다. 두 번째 단계에서는 dual-branch fusion classifier(DBFC)를 통해 오른쪽 측면 시각(RLD)과 평면 시각(SUP)에서 캡처한 서로 다른 정보를 결합하여 정확하고 견고한 분류를 수행합니다.

- **Performance Highlights**: REASON의 성능은 자가 수집한 데이터셋에서 기존 방법들에 비해 우수한 결과를 보였습니다. 실험 결과는 논문 전체를 통해 보고되며, 이 결과들은 현재까지의 접근 방식들에 비해 REASON의 효율성과 정확성을 입증하고 있습니다. 이러한 성과는 자동화된 수술 전 흡입 위험 평가에서 큰 잠재력을 보여줍니다.



### UniREditBench: A Unified Reasoning-based Image Editing Benchmark (https://arxiv.org/abs/2511.01295)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 복잡한 이미지 편집 작업을 포괄적으로 평가할 수 있는 UniREditBench라는 새로운 벤치마크를 제안합니다. 이 벤치마크는 2,700개의 샘플로 구성되어 있으며, 실제 세계와 게임 세계의 다양한 시나리오를 포함하고 있습니다. 또한, Multi-modal dual-reference evaluation (다중 모달 듀얼 레퍼런스 평가)을 도입하여 텍스트와 실제 이미지 참조를 동시에 제공함으로써 평가의 신뢰성을 높였습니다.

- **Technical Details**: UniREditBench는 8개의 주요 차원과 18개의 하위 차원을 통해 다양한 추론 유형을 커버합니다. 이러한 벤치마크는 VLM 기반 데이터를 활용하여 다중 시나리오 데이터 합성 파이프라인을 설계하여 신뢰성 높은 샘플을 생성하였습니다. 또한, UniREdit-Data-100K라는 대규모 합성 데이터셋을 개발하여 고품질의 체인 오브 사고(Chain-of-Thought, CoT) 주석을 포함하고 있습니다.

- **Performance Highlights**: Bagel 모델을 UniREdit-Data-100K에서 미세 조정한 결과, UniREditBench 및 기타 외부 벤치마크에서 상당한 성능 향상을 보여주었습니다. 연구에서는 공개 및 비공식 이미지 편집 모델들에 대한 종합적인 벤치마킹을 통하여 각 모델의 강점과 약점을 체계적으로 식별하였습니다. 이를 통해 향후 모델 개선을 위한 귀중한 통찰을 제공합니다.



### Detecting Generated Images by Fitting Natural Image Distributions (https://arxiv.org/abs/2511.01293)
Comments:
          25 pages, 9 figures, NeurIPS 2025 spotlight

- **What's New**: 생성된 이미지의 사실성이 높아짐에 따라 이들 이미지의 악용 가능성에 대한 우려가 커지고 있습니다. 기존의 탐지 방법은 이진 분류기를 학습하는 방식에 크게 의존하며, 이는 생성된 이미지의 양과 질에 따라 달라집니다. 이 논문에서는 자연 이미지와 생성된 이미지 간의 기하학적 차이를 활용하는 새로운 탐지 프레임워크인 consistency verification(ConV)을 제안합니다.

- **Technical Details**: ConV는 자연 이미지로 사전 학습된 자가 지도 모델의 손실 값의 변화 여부를 통해 이미지를 생성된 것으로 식별합니다. 두 함수의 출력을 일관되게 유지하되, 생성된 이미지에 대해서는 불일치하게 설계하여, 이를 통해 탐지가 이루어집니다. 이 방법은 생성된 이미지와 자연 이미지의 매니폴드 간의 변화를 집합적으로 분석하고, normalizing flow를 통해 데이터를 미세 조정하여 탐지 성능을 크게 향상시킵니다.

- **Performance Highlights**: 광범위한 실험을 통해 ConV의 효과가 입증되었습니다. Sora OpenAI와 OpenSora에서 생성된 이미지를 포함한 여러 벤치마크에서 ConV가 기존의 탐지 방법보다 우수한 성능을 보였습니다. ConV의 강점은 생성된 이미지 분포가 아닌 자연 데이터 분포에 적합성을 기반으로 탐지 성능이 달라진다는 점입니다.



### Adaptation of Foundation Models for Medical Image Analysis: Strategies, Challenges, and Future Directions (https://arxiv.org/abs/2511.01284)
- **What's New**: 본 논문은 의료 영상 분석 분야에 적합한 기초 모델(Foundation Models, FMs)의 적응 전략을 종합적으로 평가합니다. 특히 기존의 태스크에 특화된 모델의 한계를 극복하기 위한 새로운 접근법들을 제시하여, FMs가 의료 영상의 요구에 맞게 어떻게 조정될 수 있는지를 탐구합니다. 또한, 새로운 개발 방향으로 지속적인 학습(continual learning)과 연합 학습(federated learning)에 대해 중점을 두고 설명합니다.

- **Technical Details**: 기술적 세부사항으로는 FMs의 다양한 적응 전략이 소개됩니다. 감독적 미세 조정(supervised fine-tuning), 도메인 맞춤형 사전 훈련(domain-specific pretraining), 및 자가 감독 학습(self-supervised learning) 같은 접근 방식이 포함됩니다. 특히, 자가 감독 학습은 레이블이 없는 데이터로부터 유용한 표현을 학습하는 데 사용되며, 이는 의료 영상의 다양한 과제에 효과적으로 적용될 수 있음이 강조됩니다.

- **Performance Highlights**: FMs의 적응에 대한 성능 하이라이트에서는 각 전략의 임상 적용 가능성과 성능 개선 사항이 평가됩니다. 예를 들어, CNNs와 U-Nets와 같은 기존 모델보다 FT(Foundation Models) 기반의 접근 방식이 뛰어난 일반화를 보여주며, 기존의 한계를 극복하는 데 기여할 수 있음을 시사합니다. 이러한 전략을 통해 의료 영상 분석에서 FMs의 신뢰성과 효율성을 개선할 수 있는 잠재력이 강조됩니다.



### PRevivor: Reviving Ancient Chinese Paintings using Prior-Guided Color Transformers (https://arxiv.org/abs/2511.01274)
- **What's New**: 본 연구에서는 오래된 중국 회화의 색상 회복을 위해 PRevivor라는 색상 변환기를 제안합니다. 이 시스템은 명과 청 시대의 현대 회화에서 학습한 내용을 바탕으로 당과 송 시대의 고전 회화를 복원하는 데 초점을 맞추고 있습니다. 이는 기존의 색상 복원 기술이 가진 한계점, 특히 높은 품질의 데이터셋 부족과 복잡한 열화 패턴을 해결하기 위해 두 가지 하위 작업(명암 개선 및 색조 수정)으로 나누었습니다.

- **Technical Details**: PRevivor는 두 개의 Variational U-Net과 다중 스케일 맵핑 모듈을 활용하여 색상 복원을 수행합니다. 명암 개선을 통해 손상된 명암을 복원된 명암으로 변환하며, 색조 수정에는 마스크 가이드 픽셀 손실을 도입하여 신뢰할 수 있는 색상 정보에 기반해 색조를 정확하게 수정합니다. 이 시스템은 300개 이상의 손상된 고대 회화와 870개의 잘 보존된 회화 패치를 사용하여 평가되었습니다.

- **Performance Highlights**: PRevivor는 기존의 최신 색상화 방법들과 비교했을 때 정량적 및 정성적으로 우수한 성능을 보여주었습니다. 전문가의 선호도 조사에서 55%의 중간 점수를 기록했으며, 색상 정확성 및 분포 정렬에서 특히 두드러진 결과를 나타냈습니다. 이는 고대 중국 회화의 원래 색상을 복원하는 데 필수적인 성능 지표로 평가되었습니다.



### MotionStream: Real-Time Video Generation with Interactive Motion Controls (https://arxiv.org/abs/2511.01266)
Comments:
          Project webpage: this https URL

- **What's New**: MotionStream은 동작 제어 동영상 생성을 위한 혁신적인 방법으로 29 FPS의 실시간 생성을 지원하며, 한 대의 GPU에서 서브 초(sub-second) 지연으로 작동합니다. 기존의 비디오는 생성 속도나 비인과적(non-causal) 처리로 인해 실시간 상호작용에 한계를 겪었지만, MotionStream은 이러한 문제를 해결하여 사용자가 동작 경로를 즉각적으로 제어하고 결과를 실시간으로 확인할 수 있게 합니다. 이 방법은 텍스트-비디오 모델을 동작 제어와 결합하여 고품질의 영상을 생성합니다.

- **Technical Details**: MotionStream은 경량의 동작 제어 모델을 기반으로 하며, 트레인할 때 Self Forcing과 Distribution Matching Distillation을 활용해 인과적(causal) 모델로 압축됩니다. 그 과정에서 슬라이딩 윈도우 causal attention과 attention sinks를 설계하여 고정된 맥락 윈도우에서 무한한 길이의 비디오 생성을 안정적으로 유지하도록 합니다. 또한, KV 캐시 롤링을 적용하여 훈련 중 추론 시간의 외삽을 적절히 시뮬레이션합니다.

- **Performance Highlights**: MotionStream은 480P 해상도에서 17 FPS, 720P에서 10 FPS의 속도로 작동하며, H100 GPU 단일 장치에서 서브 초 지연을 달성합니다. 추가적으로, VAE 디코더 최적화를 통해 29 FPS로 성능이 향상됩니다. 실험 결과, 카메라 제어와 같은 다양한 동작 제어 작업에서 최신 3D 방법보다 20배 이상 빨라진 성능을 자랑합니다.



### Source-Only Cross-Weather LiDAR via Geometry-Aware Point Drop (https://arxiv.org/abs/2511.01250)
- **What's New**: 이번 논문에서는 악조건에서 LiDAR 시멘틱 세분화(semantic segmentation)의 성능 저하 문제를 해결하기 위해 'Light Geometry-aware adapter'라는 독창적인 모듈을 제안합니다. 이 모듈은 지리적 구조에 대한 인식을 높이고, 0~360도 경계에서 이웃 점들의 연속성을 보존하는 새로운 접근 방식을 제공합니다. 이를 통해 구조적으로 취약한 영역에서의 예측 안정성을 크게 향상시킬 수 있습니다.

- **Technical Details**: 제안된 모듈은 수평 원형 패딩(horizontal circular padding)과 azimuth 정렬을 통해 이웃 점들의 연속성을 유지합니다. 또한, 지역 창(local-window) K-최근접 이웃(K-Nearest Neighbors)을 사용하여 인근 점들을 모으고 간단한 지역 통계(local statistics)를 계산하는 과정을 포함합니다. 이러한 정보를 압축하여 합성된 기하학적으로 인식 가능한 신호(geometry-aware cues)를 생성하며, 이는 훈련 중 구조적으로 취약한 지역에서 예측의 안정성을 높이기 위해 사용됩니다.

- **Performance Highlights**: 본 연구에서는 SemanticKITTI에서 훈련되고 SemanticSTF에서 평가되는 소스 전용(cross-weather) 환경을 채택했습니다. 제안한 어댑터는 데이터 중심(data-centric) 증강 베이스라인에 비해 7.9% 포인트 향상된 mIoU를 기록했으며, 클래스 중심(class-centric) 정규화 베이스라인에 비해 0.6 포인트 향상시켰습니다. 이러한 결과는 기하학적 구동 정규화(geometry-driven regularization)가 전천후 LiDAR 세분화의 핵심 방향임을 보여줍니다.



### CenterMamba-SAM: Center-Prioritized Scanning and Temporal Prototypes for Brain Lesion Segmentation (https://arxiv.org/abs/2511.01243)
- **What's New**: 본 논문에서는 CenterMamba-SAM이라는 엔드 투 엔드(End-to-End) 자동 뇌 병변 세분화(segmentation) 프레임워크를 제안한다. 이 프레임워크는 사전 훈련된(backbone) 모델을 고정하고 경량 어댑터(adapters)만을 훈련하여 효율적인 세밀 조정을 가능하게 한다. CenterMamba 인코더는 3x3 코너-축-센터 단기 스캔 스키마를 사용하여 푸른 약한 경계와 작은 병변을 식별하는 반응성을 높인다.

- **Technical Details**: CenterMamba-SAM은 세 가지 구성 요소로 이루어져 있으며, 첫 번째로 CenterMamba 인코더는 새로운 공간적 스캔 전략을 통해 약한 경계와 미세한 병변에 대한 민감성을 높인다. 두 번째로, 프로토타입 기반의 구조적 프롬프트 생성기(prototype-based structural prompt generator)는 사용자 개입 없이 이웃 슬라이스에서 신뢰할 수 있는 프롬프트를 자동으로 생성하여 슬라이스 간 일관성을 향상시킨다. 마지막으로, 메모리 보강 프로그래시브 디코더(memory-augmented progressive decoder)는 다중 스케일 심층 감독(multi-scale deep supervision)과 메모리 상호작용을 결합하여 전역적 해부학적 일관성을 유지하면서 세부 정보를 복원한다.

- **Performance Highlights**: CenterMamba-SAM은 BraTS2021, ISLES2022, FCD2023, ICH2020, Instance2022와 같은 여러 공공 뇌 병변 벤치마크에서 최첨단 성능을 달성하였다. 이 연구의 결과는 복잡한 임상 시나리오에서도 강력한 세분화 능력과 일반화 가능성을 입증하였다. 전반적으로 CenterMamba-SAM은 상호작용 프롬프트 없이도 경쟁력 있는 결과를 생성하는 완전 자동 세분화 프레임워크로 자리잡고 있다.



### Beyond Deceptive Flatness: Dual-Order Solution for Strengthening Adversarial Transferability (https://arxiv.org/abs/2511.01240)
Comments:
          Accepted by Pattern Recognition in Nov 01,2025

- **What's New**: 이번 연구에서는 'deceptive flatness' 문제를 해결하기 위해 Dual-Order 정보를 바탕으로 한 새로운 블랙박스 그래디언트 기반의 공격 기법을 제안합니다. 이를 통해 'Adversarial Flatness (AF)'라는 개념을 도입하고, 공격의 전이 가능성에 대한 이론적 보장을 제공합니다. 새로운 기법은 'Adversarial Flatness Attack (AFA)'로 실행되며, 변형된 그래디언트 부호 문제를 해결합니다.

- **Technical Details**: 연구에서 제안하는 Adversarial Flatness (AF)는 그래디언트 정보의 통합을 통해 공격의 효과성을 높입니다. AFA는 초기 내적 샘플링 효율성을 증대시키기 위해 MonteCarlo Adversarial Sampling (MCAS)을 활용하여 내적 샘플의 다양성을 개선합니다. 이러한 접근으로, 다양한 모델 아키텍처에서도 높은 전이 가능성을 보장합니다.

- **Performance Highlights**: 제안한 방법은 ImageNet 호환 데이터셋에서 여섯 가지 기준선에 대한 성능을 입증하며, 더 매끄러운 영역에서 적대적인 샘플을 생성합니다. 또한, Baidu Cloud API에서 검증된 결과를 바탕으로, 기존 방법들을 초월하는 효과를 보여줍니다. 특히, 다양한 입력 변환 공격을 통합하여 해당 공격의 효과도 현저히 향상시킵니다.



### Eyes on Target: Gaze-Aware Object Detection in Egocentric Video (https://arxiv.org/abs/2511.01237)
Comments:
          Accepted at RAAI 2025

- **What's New**: 이 논문에서는 'Eyes on Target'이라는 새로운 깊이 인식 및 시선 유도(object detection) 객체 탐지 프레임워크를 제안합니다. 이 프레임워크는 사람이 주목하는 지역에 스페이셜 피처 선택을 편향시키기 위해 Vision Transformer (ViT)의 주의 메커니즘에 시선 정보를 주입합니다. 기존의 객체 탐지 모델과 차별화되며, 시뮬레이션 환경에서 인지 능력을 평가하는 데 있어 중요한 역할을 합니다.

- **Technical Details**: 시선 추적 기술은 피실험자의 시선을 추적하여 주목하는 물체를 인식하는 데 사용됩니다. 이 연구에서는 시선 위치, 깊이, 동공 직경과 같은 여러 농도를 포함하여 시각 정보를 모델에 통합하여 Vision Transformer (ViT)가 인간의 시각적 관심과 일치하도록 합니다. 또한, 'gaze-aware head importance'라는 새로운 지표를 제안하여 시선 데이터가 주의 헤드에 미치는 영향을 정량화합니다.

- **Performance Highlights**: 제안된 모델은 다양한 실험 및 절제 연구를 통해 gaze-agnostic 기준 모델에 비해 일관된 검출 정확도를 향상시킨 것으로 나타났습니다. 특히, 커스텀 시뮬레이터 데이터 세트 및 공개 벤치마크(Ego4D Ego-Motion 및 Ego-CH-Gaze 데이터 세트)에서 개선된 성능을 보였습니다. 이 연구는 시각적 집중력 평가 및 인간 성능 분석의 가능성을 보여주며, 모델의 해석력을 높이는 데 기여합니다.



### Gesture Generation (Still) Needs Improved Human Evaluation Practices: Insights from a Community-Driven State-of-the-Art Benchmark (https://arxiv.org/abs/2511.01233)
Comments:
          23 pages, 10 figures. The last two authors made equal contributions

- **What's New**: 이 논문에서는 자동화된 음성 기반 3D 제스처 생성을 위한 인간 평가 관행을 검토하고, 표준화의 부족과 결함이 있는 실험 설정의 빈번한 사용을 지적합니다. 이러한 문제로 인해 다양한 방법의 비교 또는 최신 기술의 현황을 알기가 어려운 상황이 발생합니다. 논문은 BEAT2 데이터 세트를 위해 세밀한 인간 평가 프로토콜을 소개하고, 6개의 최근 제스처 생성 모델에 대한 평가를 통해 중요한 발견을 제공합니다.

- **Technical Details**: 연구에서는 자동 음성 인식 기반 3D 제스처 생성의 과제를 정의하고, 최근의 생성 모델링 기법의 발전에도 불구하고 신뢰성 있는 평가 방법의 필요성을 강조합니다. BEAT2 데이터 세트를 기반으로 한 새로운 평가 프로토콜을 개발하여, 평가의 유효성과 재사용성을 증가시키기 위한 몇 가지 개선 사항을 제안합니다. 이를 통해, 매우 다양한 평가 방법론이 존재하며, 사람 평가의 생태학적 Validity에 대한 정보가 부족하다는 점을 언급합니다.

- **Performance Highlights**: 대규모 사용자 평가를 통해 6개의 모델을 motion realism과 speech-gesture alignment 측면에서 비교하였고, 결과는 새로운 모델이 항상 이전 접근법을 초월하지 않음을 보여줍니다. 또한, 높은 motion realism 또는 speech-gesture alignment의 주장도 엄격한 평가에서 유효하지 않을 수 있음을 지적하며, 질적 평가와 다중모달 정렬의 분리된 평가가 필요하다고 강조합니다. 공개된 5시간의 합성 모션과 750개 이상의 렌더링 비디오 stimuli는 새로운 평가를 가능하게 하며, 이 과정을 통해 수집된 16,000개의 인간 선호 투표도 함께 공유될 예정입니다.



### Saliency-Guided Domain Adaptation for Left-Hand Driving in Autonomous Steering (https://arxiv.org/abs/2511.01223)
- **What's New**: 이 논문은 자동차 자동화 모델의 도메인 적응을 향상시키기 위해 왼쪽 주행 조건에 적합하도록 PilotNet 모델을 조정하는 훈련 방법을 탐구합니다. 특히, 미국의 오른쪽 주행 데이터를 기반으로 한 모델과 호주 고속도로에서 수집된 실제 데이터를 사용하여 여러 훈련 방법을 비교합니다. 이 연구는 플립된 데이터의 사용이 모델 적응을 개선할 수 있는지에 대한 초기 정렬을 제공할 수 있는지를 조사합니다.

- **Technical Details**: 제안된 접근 방식은 플립된(right-hand data) 미국 데이터를 사전 훈련한 후, 호주 고속도로 데이터에서 파인튜닝(fine-tuning)하는 것입니다. 이 과정에서 플립된 데이터가 모델의 주행 예측 정확성을 개선하고 왼쪽 도로 기능에 대한 모델의 주목을 변화시키는 데 기여하는지 분석합니다. 실험에서는 Saliency map을 사용하여 모델의 주목 분포 변화를 시각적으로 평가합니다.

- **Performance Highlights**: 실험 결과, 플립된 데이터로 단독 사전 훈련한 경우 예측의 안정성이 저하되었지만, 이후에 파인튜닝을 수행하면 예측 오류가 감소하고 왼쪽 신호에 대한 주목이 강화됨을 확인했습니다. ResNet 아키텍처에서도 유사한 적응 경향이 관찰되어 이 방법의 일반화 가능성을 입증했습니다. 이러한 결과는 새 데이터 없이 기존 모델을 효과적으로 조정하는 사전 처리 기술의 중요성을 강조합니다.



### Thought-For-Food: Reasoning Chain Induced Food Visual Question Answering (https://arxiv.org/abs/2511.01213)
Comments:
          10 pages, 11 figures, 6 tables

- **What's New**: 이 논문에서는 인도 요리를 위한 Visual Question Answering(VQA) 시스템의 한계를 해결하기 위해, 다단계 추론 과정을 요구하는 새로운 접근 방식이 소개되었습니다. 기존의 VQA 시스템이 서구 요리에 초점을 맞춰 기술적으로 유한한 반면, 인도 요리는 복잡성을 포함하고 있어 해당 분야에 대한 논의가 필요하다는 점을 강조합니다. 이를 통해, 간단한 인식 이상의 복잡한 질문과 맥락을 해결할 수 있는 구조를 제시하고 있습니다.

- **Technical Details**: 이 연구에서는 질문-답변 체인(Reasoning Chains)을 통해 모델의 성능을 향상시키기 위한 방법론을 제안합니다. 자동으로 생성된 추론 체인을 사용해 소규모 LLMs(대형 언어 모델)와 VLMs(비주얼 언어 모델)를 파인튜닝하고, 강화학습(Reinforcement Learning)으로 추가 학습을 진행해 보다 정확한 응답을 이끌어내고자 합니다. 이를 통해 인도 요리에 대한 VQA의 정확도를 평균 10% 향상시켰습니다.

- **Performance Highlights**: 제안된 방법은 IndiFoodVQA 데이터셋의 기준 성능을 일관되게 초과하여, 문화적으로 다양한 맥락에서 음식 VQA의 중요성을 입증합니다. 최종 모델은 최고 정확도 71.12%를 달성하며 최선의 성능을 보여줍니다. 이와 같은 이유 기반 접근법에 대한 연구 및 분석은 인도 요리를 이해하고 학습하는 데 있어 새로운 가능성을 제시합니다.



### OmniVLA: Unifiying Multi-Sensor Perception for Physically-Grounded Multimodal VLA (https://arxiv.org/abs/2511.01210)
- **What's New**: OmniVLA는 물리적으로 기반을 둔 공간 지능을 구현하기 위해 RGB 카메라 외의 새로운 감지 모드를 통합한 다중 감각 VLA 모델입니다. 기존 VLA 모델의 단점을 보완하여 다수의 센서 데이터를 효과적으로 해석하고 이를 기반으로 작업 출력을 가이드를 향상시킵니다. 특히, 센서-마스킹 이미지라는 새로운 표현 방식을 도입하여 RGB 이미지와 센서 정보를 통합함으로써 보다 효율적인 커뮤니케이션과 훈련을 가능하게 합니다.

- **Technical Details**: OmniVLA의 핵심 구조는 RGB 이미지에 온도 대비를 나타내는 적외선 카메라, mmWave 레이더, 마이크로폰 배열 등 다양한 센서 데이터를 합친 센서-마스킹 이미지입니다. 이 구조는 모든 센서 입력을 통합하여 공간적으로 기반하고 의미적으로 정렬된 표현을 제공합니다. 또한, 각 센서에 맞춰 경량의 투영 층을 추가하여 센서 이미지를 더 잘 정렬된 토큰으로 변환하며, LLM을 통해 로봇의 행동 출력을 제공하는 방식으로 구성됩니다.

- **Performance Highlights**: OmniVLA는 여러 실제 작업에서 평균 84%의 성공률을 기록하여 RGB 전용 모델과 원시 센서 입력 모델보다 각각 59% 및 28% 더 우수한 성능을 보여주었습니다. 또한, 50%의 훈련 데이터만 사용하여도 유사한 성공률을 달성하여 효율적인 데이터 사용성을 증명했습니다. 우리 모델은 이전 모델들에 비해 높은 일반화 능력을 갖추고 있으며, 세 가지 미보지 않은 작업에서도 더 나은 성능을 보여주었습니다.



### MoSa: Motion Generation with Scalable Autoregressive Modeling (https://arxiv.org/abs/2511.01200)
- **What's New**: MoSa는 텍스트 기반 3D 인간 모션 생성을 위한 새로운 계층적 모션 생성 프레임워크입니다. 이는 Vector Quantization-guided Generative Transformers (VQ-GT) 패러다임을 coarse-to-fine 스케일 생성 프로세스를 통해 개선합니다. MoSa에서는 Multi-scale Token Preservation Strategy (MTPS)를 도입하였으며, 이는 계층적 residual vector quantization variational autoencoder (RQ-VAE)와 통합되어 있습니다.

- **Technical Details**: MTPS는 각 계층의 quantization에서 interpolation을 사용하여 coarse-to-fine 방법으로 다중 스케일 모션 토큰을 보존합니다. MoSa는 Scalable Autoregressive (SAR) 모델링을 지원하며, 이는 각 단계에서 단일 토큰 예측 대신 여러 토큰을 병렬로 예측하도록 설계되었습니다. 따라서 MoSa는 10개의 추론 단계만으로 과정이 완료됩니다.

- **Performance Highlights**: MoSa는 Motion-X 데이터셋에서 FID 점수 0.06을 기록하며, MoMask의 0.20보다 훨씬 뛰어난 성능을 보여줍니다. MoSa는 추론 시간을 27% 줄여주며, 모션 편집과 같은 하위 작업에도 추가적인 fine-tuning 없이 잘 일반화됩니다. 실험 결과, MoSa는 현존하는 최첨단 생성 품질과 효율성을 달성했습니다.



### A Topology-Aware Graph Convolutional Network for Human Pose Similarity and Action Quality Assessmen (https://arxiv.org/abs/2511.01194)
Comments:
          10 pages, 5 figures. Submitted as a computer vision paper in the cs.CV category

- **What's New**: 이 논문은 Action Quality Assessment (AQA)에 대한 새로운 접근 방식을 제안합니다. 인체 스켈레톤을 그래프로 모델링하여 Graph Convolutional Network (GCN)을 사용해 포즈 유사성을 측정하는 방법을 제시합니다. 이 방법은 기존의 좌표 기반 기법을 뛰어넘어 더 정교하고 사실적인 포즈 임베딩을 학습합니다. 이를 통해 AQA-7 및 FineDiving 벤치마크에서 경쟁력 있는 성능을 보여줍니다.

- **Technical Details**: GCN-PSN은 사람의 포즈를 정확하게 추정하기 위해 YOLOv5 모델을 사용하여 인간을 감지한 후 HRNet을 통해 2D 키포인트를 로컬라이즈합니다. 15개의 키포인트로 구성된 스켈레톤 그래프를 정의하고, 각 포인트 간의 연결을 기반으로 인접 행렬을 구성하여 특징 추출을 수행합니다. 그런 다음, GCN을 통해 이 그래프에서 포즈 간의 유사성을 측정하는 특성 임베딩을 학습합니다. 대칭 네트워크 아키텍처를 이용하여 두 이미지를 동시에 처리하여 최종 유사도를 계산합니다.

- **Performance Highlights**: 실험 결과, 제안된 GCN-PSN 모델은 기존의 AQA 시스템보다 유의미한 성능 향상을 보여주었습니다. 특히, 특정 포즈 간의 유사성을 평가하는 데 있어 스켈레톤 토폴로지를 효과적으로 활용하여 더욱 신뢰할 수 있는 점수를 제공하는 것으로 나타났습니다. AQA와 관련된 여러 벤치마크 평가에서 긍정적인 결과를 도출하며, 향후 연구의 기초 자료로 활용될 수 있을 것으로 예상됩니다.



### Diffusion Transformer meets Multi-level Wavelet Spectrum for Single Image Super-Resolution (https://arxiv.org/abs/2511.01175)
- **What's New**: 이번 연구에서는 이미지 웨이블릿 스펙트라를 기반으로 한 Diffusion Transformer 모델(DTWSR)을 제안합니다. 기존의 SISR(단일 이미지 초해상도) 방법들은 멀티스케일 주파수 서브밴드 간의 상관관계를 고려하지 않아였으나, DTWSR은 이러한 상관관계를 포착하여 자연스럽고 일관된 고해상도 이미지를 생성합니다. 이 모델은 다단계 Discrete Wavelet Transform(MDWT)을 활용하여 이미지를 분해하고, 피라미드 토크나이제이션 기법을 통해 효과적으로 이미지를 처리합니다.

- **Technical Details**: DTWSR은 저주파(LF)와 고주파(HF) 서브밴드의 상관관계를 반영하기 위해 설계된 이중 디코더 모델을 사용합니다. 이를 통해 깔끔한 저주파 내용과 세밀한 고주파 세부 정보를 각각 생성할 수 있습니다. 모델은 조건부 확산 프레임워크를 기반으로 하여, 확산 모델(DM)의 우수한 이미징 생성 능력을 통합하여 고해상도 이미지를 복원합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에 대한 광범위한 실험을 통해 DTWSR은 주관적 품질과 신뢰도 측면에서 우수한 성과를 보였습니다. 이 모델은 고해상도 이미지 복원에 있어 최신 기술과 비교하여 높은 성능을 입증하였으며, 텍스처 세부 사항과 이미지 충실도를 상당히 개선했습니다.



### Web-Scale Collection of Video Data for 4D Animal Reconstruction (https://arxiv.org/abs/2511.01169)
Comments:
          NeurIPS 2025 Datasets and Benchmarks

- **What's New**: 본 논문은 야생 동물 연구를 위한 컴퓨터 비전 기술의 발전 가능성을 소개합니다. 기존의 데이터 수집 방식이 통제된 환경에서의 포획 시스템에 의존하는 반면, 우리는 유튜브 비디오를 활용하여 객체 중심 클립을 자동으로 수집하고 처리하는 파이프라인을 개발했습니다. 이 파이프라인을 사용하여 30,000개의 비디오(2백만 프레임)를 수집하였고, 4D 동물 재구성을 위한 새로운 벤치마크인 Animal-in-Motion(AiM)을 제시합니다.

- **Technical Details**: 논문에서는 기존의 4D 재구성 작업이 2D 이미지에서 동물의 3D 자세와 형태를 추정하는 것으로부터 발전해왔음을 설명합니다. 우리는 3D-Fauna 방식을 기반으로 하여 추가적인 키점 감독과 몇 가지 최적화를 도입하여 모델-프리 접근의 재구성 품질을 향상시켰습니다. 4D 동물 재구성을 위한 Seq-Level 최적화를 통해 최초의 4D 재구성 기준선을 설정하였습니다.

- **Performance Highlights**: Animal-in-Motion 데이터셋에서의 벤치마크 결과를 바탕으로, 모델-기반 접근이 높은 점수를 기록하지만 비현실적인 3D 형태를 생성하고, 모델-프리 접근은 더 자연스러운 재구성이 이루어지나 낮은 점수를 받는 경향을 보였습니다. 이를 통해 현행 평가 방식의 한계와 질적 평가의 중요성을 강조하며, 향후 3D 인식 메트릭 디자인 방향성을 제안합니다.



### ROVER: Benchmarking Reciprocal Cross-Modal Reasoning for Omnimodal Generation (https://arxiv.org/abs/2511.01163)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문에서 소개된 ROVER는 통합된 다중 모달 모델(UMM)의 상호 교차 모달(reasoning) 능력을 검증하는 새로운 벤치마크입니다. ROVER는 1,312개 작업과 1,876개의 이미지를 바탕으로 하여, 서로 다른 모달리티의 출력을 유도하고 검증할 수 있는 능력을 측정하는 데 초점을 맞추고 있습니다. 이는 텍스트 기반의 추론과 영상 기반의 이유(visual generation) 개념을 통합하여 평가하는 새로운 방식을 제공합니다.

- **Technical Details**: ROVER 벤치마크는 두 가지 주요 설정을 갖고 있습니다: 첫째, 시각 생성을 위한 언어 증강(reasoning)으로, 총 44개의 개념 도메인에서 복잡한 이유 작업을 평가합니다. 둘째, 언어 생성을 위한 시각 증강으로, 6가지 하위 작업 변형을 포함하여 시각적 개념과 관련된 문제 해결을 다룹니다. 해당 벤치마크는 GPT-4.1 기반의 자동화된 VLM(judge)을 사용하여, 논리적 일관성(logical coherence) 및 출력의 정렬(alignment) 등을 평가합니다.

- **Performance Highlights**: 본 연구는 17개의 통합된 다중 모달 모델에 대한 실험 결과, 상호 교차 모달(reasoning)이 비주얼 생성의 품질에 미치는 영향을 강조합니다. 특히, 상호 연결된(image-text) 모델이 비연결(non-interleaved) 모델보다 월등히 뛰어난 성능을 보이며, 강력한 단일 모달(unimodal) 모델 조합도 해당 성능을 재현하지 못함을 보여주고 있습니다. 마지막으로, 물리적(reasoning)과 상징적(symbolic) 추론 간의 단절이 현저하게 나타나며, 이는 시각 추론의 효율성에 중대한 영향을 미침을 알 수 있습니다.



### MicroAUNet: Boundary-Enhanced Multi-scale Fusion with Knowledge Distillation for Colonoscopy Polyp Image Segmentation (https://arxiv.org/abs/2511.01143)
Comments:
          Work in progress

- **What's New**: 본 논문에서 제안하는 MicroAUNet은 경량화된 주의 기반(segmentation networks) 분할 네트워크로, 깊이 우선 분리된(dilated convolutions) 합성과 파라미터 공유(channel-spatial attention) 블록을 결합하여 다중 스케일 경계 특징을 강화합니다. 이 네트워크는 높은 정확성과 낮은 모델 복잡성을 유지하면서 실시간(colonoscopy) 임상용 폴립(segmentation) 분할에 적합하도록 설계되었습니다.

- **Technical Details**: MicroAUNet은 경량화 아키텍처와 점진적인 지식 증류(progressive knowledge-distillation) 기법을 통해 복잡한 배경 속에서도 폴립 경계를 정밀하게 세분화하는 데 초점을 맞추고 있습니다. 깊이 우선 분리된 합성(depthwise separable convolutions)과 공통 채널-공간 주의 메커니즘을 사용하여 다중 스케일 문맥 정보를 효율적으로 추출합니다. 이에 따라 폴립 탐지(model)와 진단(analysis) 과정에서의 초기 세분화 정확도를 높입니다.

- **Performance Highlights**: 공공 데이터셋에 대한 광범위한 검증 결과, MicroAUNet은 기존의 최신 방법들과 비교해세분화 정확도(segmentation accuracy)와 추론 효율성(inference efficiency)에서 우수함을 입증했습니다. 이는 경량화된 모델을 통한 효과적인 학습(runtime) 속도를 가능하게 하여 임상 적용의 잠재력을 강조합니다.



### Learning with Category-Equivariant Architectures for Human Activity Recognition (https://arxiv.org/abs/2511.01139)
- **What's New**: 본 논문에서는 CatEquiv를 제안했습니다. 이는 관성 센서를 활용한 인간 활동 인식(HAR)을 위한 카테고리 등변(neural network) 신경망입니다. CatEquiv는 시간, 진폭, 구조적 대칭(symmetries)을 체계적으로 인코딩하여 데이터의 범주적 대칭 구조를 캡처합니다.

- **Technical Details**: CatEquiv는 순환 시간 이동(cyclic time shifts), 긍정적 이득(positive gains) 및 센서 계층(poset)을 결합한 범주적 대칭 곱(categorical symmetry product)을 통해 등변성을 달성합니다. 이 신경망의 기본 아키텍처는 순환 1D 컨볼루션, RMS 정규화, 축 공유 필터를 포함하여 모형의 강건성을 보다 수월하게 확보할 수 있도록 합니다.

- **Performance Highlights**: UCI-HAR 데이터셋에서 CatEquiv는 기존의 순환 패딩 CNNs 및 일반 CNNs에 비해 명백히 높은 강건성을 나타냈습니다. 특히 CatEquiv는 복합적인 OOD(Out-Of-Distribution) 조건 하에서 기존 모델보다 높은 정확도와 매크로-F1 점수를 달성했습니다. 이는 모델의 용량을 증가시키지 않고도 범주적 대칭을 강제하여 얻어진 결과입니다.



### Weakly Supervised Concept Learning with Class-Level Priors for Interpretable Medical Diagnosis (https://arxiv.org/abs/2511.01131)
- **What's New**: 이 연구는 의료 영상에서 인간 해석 가능(predictions)를 가능하게 하는 새로운 프레임워크인 Prior-guided Concept Predictor (PCP)를 소개합니다. PCP는 명시적 감독 없이 의료 개념 예측을 가능하게 하며, 개념 레벨 주석에 대한 의존을 줄입니다. 이 프레임워크는 클래스 수준의 개념 사전(class-level concept priors)을 활용하여 개념 예측의 신뢰성을 높입니다.

- **Technical Details**: PCP는 약한 감독(weak supervision)을 활용하여 이미지의 특징(feature)을 토대로 개념을 실제 데이터와 정렬하여 예측합니다. 모델은 ResNet 기반의 인코더(encoders)를 사용하여 입력 이미지의 시각적 특징을 추출하고, 이를 통해 개념 활성화의 확률벡터를 생성합니다. 두 가지 정규화기(KL-divergence와 entropy loss)가 개념 선택성과 예측의 정렬을 증진시키도록 설계되었습니다.

- **Performance Highlights**: 실험 결과 PCP는 PH2 및 WBCatt 데이터셋에서 33% 이상의 F1-score 향상을 보이며, HAM10000 및 CXR4 데이터셋에서도 Fully Supervised CBM 및 V-IP 모델에 맞먹는 분류 성능(comparable classification performance)을 달성했습니다. 이는 명시적 감독이 없더라도 신뢰할 수 있는 해석 가능한 개념 추론이 가능하다는 것을 보여줍니다.



### Boosting performance of computer vision applications through embedded GPUs on the edg (https://arxiv.org/abs/2511.01129)
Comments:
          4 pages, 6 figures

- **What's New**: 이 논문은 모바일 기기에서 증강 현실(augmented reality) 기술을 사용하는 컴퓨터 비전(application) 애플리케이션의 자원(resource) 수요와 이를 해결하기 위한 방법을 제안합니다. 특히, 이 연구는 제한된 자원을 가진 장치에서도 이러한 애플리케이션을 효율적으로 실행할 수 있도록 엣지 컴퓨팅(edge computing) 기술을 활용하는 방법을 다룹니다.

- **Technical Details**: 저자들은 그래픽 처리 장치(GPUs)를 탑재한 내장 장치(embedded devices)를 사용하여 엣지 컴퓨팅의 한계를 극복하는 방안을 제안하고 있습니다. CPU만 사용할 때에 비해 GPU가 성능 향상(performance gain)을 이룰 수 있는 실험 결과를 보여주며, 이는 사용자가 애플리케이션을 사용할 때 느끼는 품질(quality of experience)을 개선하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, GPU를 사용한 경우 CPU만을 사용할 때에 비해 월등한 성능 향상이 나타났습니다. 이를 통해, 자원 제한이 있는 모바일 기기에서도 컴퓨터 비전 응용 프로그램을 원활하게 실행할 수 있게 된다 는 점에서 의미가 큽니다.



### Anatomically Constrained Transformers for Echocardiogram Analysis (https://arxiv.org/abs/2511.01109)
- **What's New**: 최근 비디오 트랜스포머들이 심초음파 분석에 강력한 잠재력을 보여주고 있으며, 특히 ViACT(비디오 해부학적으로 제약된 트랜스포머)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 해부학적 프라이어를 트랜스포머 아키텍처에 직접 통합하여 비디오의 비진단 영역으로부터의 잘못된 상관관계를 학습하는 문제를 극복합니다. ViACT는 변형된 해부학적 구조를 포인트 세트로 표현하고, 공간 기하학 및 해당 이미지 패치 정보를 트랜스포머 토큰으로 인코딩합니다.

- **Technical Details**: ViACT는 여러 개의 점을 사용하여 심장 해부학을 파라미터화하고, 이것을 통해 심초음파 영상으로부터 패치를 샘플링하여 처리합니다. 이 모델은 해부학적 영역에만 집중하도록 설계되어, 임의의 이미지 콘텐츠와는 관련 없는 정보를 제거합니다. 또한, ViACT는 비디오 클립에서 마스크된 심장 근육 패치를 재구성하는 방식으로 사전 훈련이 가능하여 전통적인 비디오 모델에 비해 계산 요구 사항이 대폭 감소합니다.

- **Performance Highlights**: 사전 훈련된 ViACT는 질병 분류, 좌심실 박출력 비율(EF) 회귀 및 심장 근육 포인트 추적 등에서 뛰어난 성능을 보입니다. 이를 통해 기존 최첨단 방법들에 비해 포괄성과 성능이 향상되었습니다. 또한 ViACT는 임상적으로 유의미한 특징을 강조하는 시각화 가능한 주의(Attention) 맵을 제공하여 해부학적 구조의 이해를 돕습니다.



### Epanechnikov nonparametric kernel density estimation based feature-learning in respiratory disease chest X-ray images (https://arxiv.org/abs/2511.01098)
Comments:
          12 pages, 6 figures, 3 tables

- **What's New**: 이번 연구는 이미지 데이터를 사용하여 호흡기 질환을 진단하는 새로운 방법론을 제안합니다. 이 방법은 비모수 커널 밀도 추정 (EKDE)과 이항 로지스틱 회귀 분류기를 결합하여 통계 모델 기반의 학습 체계를 구축합니다. EKDE는 특정 형태를 가정하지 않고 데이터 분포를 모델링할 수 있는 유연성 덕분에 의료 영상에서 주요 특징을 추출하는 데 유용하다는 점을 강조하고 있습니다.

- **Technical Details**: 연구는 COVID-19 방사선 데이터셋에서 임의로 선택된 13808개의 흉부 X선 이미지를 분석했습니다. 이 데이터에서 추출된 평균 및 표준편차는 이항 로지스틱 회귀 분류기에서 이미지를 정상 또는 질병이 있는 경우로 구분하는 입력 값으로 사용되었습니다. EKDE는 이미지의 픽셀 강도 분포를 효과적으로 모델링할 수 있는 잠재력을 보여주며, 이는 진단 정확도와 신뢰성을 향상시키는 데 기여할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 70.14%의 정확도와 59.26%의 민감도, 74.18%의 특이도로 평가되었습니다. 이러한 결과는 호흡기 질환 발견 시 중간 정도의 성과를 보이지만, 민감도의 개선 여지가 있다는 점도 함께 설명하고 있습니다. EKDE 기반의 접근법이 의료 영상에서 진단의 정확성과 신뢰성을 향상시킬 수 있는 가능성을 시사합니다.



### SliceVision-F2I: A Synthetic Feature-to-Image Dataset for Visual Pattern Representation on Network Slices (https://arxiv.org/abs/2511.01087)
- **What's New**: 5G 및 6G 네트워크의 출현은 네트워크 슬라이싱(network slicing)을 미래 서비스 지향 아키텍처의 중요한 부분으로 확립했습니다. 이 논문에서는 차세대 네트워킹 시스템을 위한 기능 시각화(feature visualization)의 연구를 위해 설계된 신 synthetic 데이터 세트인 SliceVision-F2I를 소개합니다. 3만 개의 샘플로 구성된 이 데이터 세트는 다양한 변환 방법을 통해 다변량 Key Performance Indicator(KPI) 벡터를 시각적 표현으로 변환합니다.

- **Technical Details**: SliceVision-F2I는 물리적 영감 맵(physically inspired mappings), Perlin noise, 신경벽지(neural wallpapering), 프랙탈 가지치기(fractal branching) 등 네 가지 인코딩 방법을 통해 데이터를 생성합니다. 각 변환 방법마다 3만 개의 샘플이 생성되며, 이는 저해상도 RGB 이미지와 원시 KPI 벡터를 포함합니다. 이 데이터 세트는 실질적인 노이즈 모델을 통합하여 측정의 불완전성을 반영하고, 시각적 학습, 네트워크 상태 분류, 이상 탐지 및 이미지 기반 기계 학습 기술 벤치마킹에 적합합니다.

- **Performance Highlights**: SliceVision-F2I 데이터 세트는 저해상도에서 완벽한 분류 정확도를 유지할 수 있음을 입증하며, 이는 실시간 처리 가능성을 엿볼 수 있는 좋은 기회를 제공합니다. 현재 결과는 완벽한 분류 정확도를 달성했으며, 이러한 성과는 다른 연구에서도 확장 가능성을 지니고 있습니다. 또한, 이 데이터 세트는 실제 네트워크의 노이즈와 변동성을 처리할 수 있는 새로운 분류 방법을 시험하는 데 유용한 기반을 제공합니다.



### GeoToken: Hierarchical Geolocalization of Images via Next Token Prediction (https://arxiv.org/abs/2511.01082)
Comments:
          Accepted to IEEE International Conference on Data Mining (ICDM) 2025

- **What's New**: 이 논문에서는 이미지의 지리적 출처를 추정하는 새로운 접근 방식인 GeoToken을 소개합니다. 사람의 사고 과정을 모방하여 이미지를 세분화하고, 명확한 지리적 토큰을 계층적으로 예측하는 방법을 제안합니다. 이 방법은 기존 방식의 한계를 극복하며, 특허받지 않은 데이터 세트에서 월드와이드 이미지 지리적 위치를 보다 정확하게 예측하도록 고안되었습니다.

- **Technical Details**: GeoToken은 다단계 지리적 셀을 정의하고, 시각 정보와 이전 예측을 조건으로 하여 점진적으로 더 정밀한 지리적 셀을 예측합니다. 이 모델은 입력 이미지로부터 유사 이미지를 검색하여 생성 과정에 도움을 주며, S2 셀 구조를 활용하여 전역의 계층적 지리 정보를 체계적으로 탐색합니다. 또한, 오토리그레시브 방식으로 정확한 위치 추정을 위해 다양한 샘플링 기법을 통합하여 불확실성을 관리합니다.

- **Performance Highlights**: 이 논문에서 제안하는 모델은 Im2GPS3k와 YFCC4k 데이터 세트에서 다른 기준선에 비해 높은 성능을 보이며, MLLM이 없는 설정에서 정확도에서 최대 13.9% 향상된 결과를 얻었습니다. MLLM이 포함되었을 때는 모든 메트릭에서 새로운 최첨단 성능을 기록하며, 사용자 데이터 보안을 보장하면서도 현장에서 강력한 지리적 위치 추정이 가능하다는 장점을 가지고 있습니다.



### T-MLA: A Targeted Multiscale Log--Exponential Attack Framework for Neural Image Compression (https://arxiv.org/abs/2511.01079)
Comments:
          Submitted to Information Systems. Code will be released upon journal publication

- **What's New**: 본 연구에서는 T-MLA(타겟 멀티스케일 로그-지수 공격 프레임워크)를 도입하여 신경 이미지 압축(NIC) 시스템에 대한 보다 진보된 공격 방법론을 제안합니다. 이 방법은 웨이블릿(wavelet) 도메인에서 공격할 이미지의 품질을 직접 겨냥하여 적대적 소음(perturbation)을 생성합니다. 이를 통해 공격자는 사람이 인지하기 어려운 방식으로 이미지 복원 품질을 효과적으로 저하시킬 수 있습니다.

- **Technical Details**: 우리는 이미지의 다중 스케일 주파수 성분을 활용하는 웨이블릿 도메인 적대적 공격 프레임워크를 개발하였습니다. 이 방식은 웨이블릿 하위 대역에 직접적으로 소음을 주입하며, 비선형 로그-지수 매핑(nonlinear log–exp mapping)을 사용하여 지역적 웨이블릿 에너지에 따라 소음을 조절합니다. 이를 통해 공격자는 압축 후의 왜곡을 극대화하면서도 시각적 정밀도를 유지할 수 있습니다.

- **Performance Highlights**: 제안된 T-MLA는 Kodak, CLIC 및 DIV2K 데이터셋의 다양한 이미지를 통해 평가되었으며, Cheng2020-Anchor, Cheng2020-Attention, LIC-TCM 모델에서 각각 24~26 dB의 PSNR을 달성하였습니다. 이 과정에서 입력 이미지의 PSNR은 50~55 dB의 엄격한 스텔스(st stealth) 기준을 유지했습니다. 따라서, 우리 연구는 생성 및 컨텐츠 전달 파이프라인의 핵심에서 존재하는 중요한 보안 결함을 드러냅니다.



### FastBoost: Progressive Attention with Dynamic Scaling for Efficient Deep Learning (https://arxiv.org/abs/2511.01026)
Comments:
          17pages , 10figures , 12tables

- **What's New**: 이번 논문에서는 FastBoost라는 파라미터 효율성(neural architecture)을 가진 신경망 아키텍처를 제안합니다. 이 아키텍처는 Dynamically Scaled Progressive Attention (DSPA) 메커니즘을 활용하여 CIFAR 벤치마크에서 최신 성능을 달성하였습니다. 주요 성능 지표로는 CIFAR-10에서 0.85M 파라미터로 95.57%의 정확도, CIFAR-100에서 0.92M 파라미터로 81.37%의 정확도를 기록했습니다.

- **Technical Details**: FastBoost는 DSPA의 세 가지 주요 혁신을 기반으로 설계되었습니다. 첫 번째는 Adaptive Fusion으로, 동적인 가중치를 통한 채널-공간 주의(attention) 혼합입니다. 두 번째는 Phase Scaling으로, 훈련 단계에 따라 0.5에서 1.0까지의 강도 변조가 가능합니다. 마지막으로 Residual Adaptation은 자기 최적화를 위한 건너뛰기 연결(skip connections)을 통해 가중치를 조정합니다.

- **Performance Highlights**: FastBoost는 MBConv 블록을 향상시키고 DSPA를 통합하여 MobileNetV3에 비해 2.1배의 파라미터 감소를 달성하며, CIFAR-10에서 정확도가 3.2%포인트 향상되었습니다. 이 아키텍처는 실시간 가중치 조정이 가능한 이중 주의 경로와 12.7% 향상된 그래디언트 흐름을 위한 연속 정제 층(cascaded refinement layers)을 특징으로 하며, 하드웨어 친화적인 설계로 0.28G FLOPs를 유지합니다. 이러한 동적 주의 최적화(dynamically optimized attention)와 효율적인 합성곱 연산의 공동 최적화는 자원 제약이 있는 엣지 장치에서도 정확도를 유지할 수 있게 합니다.



### HyFormer-Net: A Synergistic CNN-Transformer with Interpretable Multi-Scale Fusion for Breast Lesion Segmentation and Classification in Ultrasound Images (https://arxiv.org/abs/2511.01013)
Comments:
          This manuscript has been submitted to Informatics in Medicine Unlocked

- **What's New**: 본 논문은 유방암 진단을 위한 하이브리드 CNN-Transformer 구조인 HyFormer-Net을 제안합니다. 이 모델은 세그멘테이션(segmentation)과 분류(classification)를 동시에 수행하며, 내재적인 해석 가능성을 제공합니다. 특히, 다중 스케일 마스터 구조를 통해 EfficientNet-B3과 Swin Transformer를 통합하여 진단 결과의 정확성과 설명 가능성을 향상시킵니다.

- **Technical Details**: HyFormer-Net의 이중 브랜치 인코더는 EfficientNet-B3와 Swin Transformer를 통합하여 다중 스케일(hierarchical) 융합 블록을 사용합니다. 특히, 주의(attention) 기반 디코더를 통해 정밀도와 설명 가능성을 제공합니다. 이 모델은 양자화된 IoU 검증과 Grad-CAM을 활용하여 해석 가능성을 강화하며, 일반적인 CNN-Transformer 아키텍처에 비해 문제별 다중 작업 학습(multi-task learning) 접근 방식을 사용합니다.

- **Performance Highlights**: HyFormer-Net은 BUSI 데이터셋에서 평균 Dice Score 0.761과 정확도 93.2%를 달성하였으며, U-Net 및 다른 최신 모델들보다 우수한 성능을 보였습니다. 특히, 악성(가장 위험한) 질병의 검출률은 92.1%로 낮은 가양성(false positive) 수치를 보장합니다. 앙상블 모델링을 통해 Dice Score 90.2% 및 100% 악성 재현율(Recall)을 달성했으며, 이는 실제 임상 적용에서도 신뢰할 수 있는 결과로 평가됩니다.



### Integrating Visual and X-Ray Machine Learning Features in the Study of Paintings by Goya (https://arxiv.org/abs/2511.01000)
- **What's New**: 이 논문에서는 프란시스코 고야 작품의 예술 인증(art authentication)을 위한 새롭고 통합된 다중 모드 기계 학습 프레임워크를 소개합니다. 이 프레임워크는 고야의 회화 이미지를 시각적(visual) 및 X-ray 방사선(radiographic) 이미지 모두에 동일한 특징 추출 기법을 적용하여 복잡한 인증 문제를 해결할 수 있도록 설계되었습니다. 또한, 전형적인 인증 방법의 한계를 극복하고 다단계 데이터 융합(multi-modal data fusion) 접근 방식을 통해 단일 모드 방법들에 비해 뛰어난 성능을 달성합니다.

- **Technical Details**: 이 연구에서 개발된 방법론은 시각 이미지와 X-ray 이미지를 동일하게 처리하여 주요 특징(특징 추출)들을 추출하는 통합된 파이프라인을 구축합니다. Grey-Level Co-occurrence Matrix, Local Binary Patterns, 엔트로피 측정, 에너지 계산 및 색상 분포 분석 등의 수학적 모델링이 포함되어 있습니다. 최적화된 One-Class Support Vector Machine을 이용해 10배 교차 검증(cross-validation)을 통해 하이퍼파라미터를 조정하며, 24개의 인증된 고야 작품을 기반으로 총 97.8%의 분류 정확도를 기록하였습니다.

- **Performance Highlights**: 본 연구의 결과는 인증(art authentication) 분야에서 기존의 단일 모드 접근 방식에 비해 상당한 성능 향상을 보여주었습니다. 특히, 대표 사례 연구인 "Un Gigante"의 경우 92.3%의 인증 신뢰도를 달성하였습니다. 결론적으로 본 연구는 예술 인증 응용 분야에서 시각적 및 방사선 이미지를 위한 동일한 계산 방법을 적용함으로써 강력한 인증 평가를 위한 충분한 차별 정보(discriminative information)를 포착할 수 있음을 입증합니다.



### MID: A Self-supervised Multimodal Iterative Denoising Framework (https://arxiv.org/abs/2511.00997)
- **What's New**: MID(멀티모달 반복 노이즈 제거) 프레임워크는 자가 감독(self-supervised) 학습을 통해 노이즈가 있는 데이터를 정제하는 새로운 접근 방식을 제안합니다. 전통적인 노이즈 제거 방법과 달리, MID는 쌍이 맞춰진 깨끗한 데이터셋을 필요로 하지 않으며, 직접적으로 노이즈 특성을 학습합니다. 이를 위해 MID는 두 개의 신경망을 사용하여 현재의 노이즈 단계와 해당 노이즈를 감지하고 제거하는 과정을 반복합니다.

- **Technical Details**: MID 프레임워크는 복잡한 비선형 노이즈를 처리하기 위해 1차 테일러 확장(First-order Taylor expansion)을 적용하여 노이즈 과정을 국소적으로 선형화합니다. 이 과정에서 MID는 비선형 오염을 효과적으로 제거하기 위해 훈련된 두 개의 신경망을 사용하여, 지금까지 악화된 노이즈 데이터에서 점진적으로 깨끗한 데이터를 회복하게 됩니다. 작동 원리상, 노이즈 예측 신경망은 현재 데이터 상태 내에서 노이즈를 추정하고 이와 관련된 노이즈 증분을 예측하여 제거합니다.

- **Performance Highlights**: 실험을 통해 MID는 이미지 노이즈 제거, 생물학적 신호 정제, MRI의 의료 이미지 질 향상 등 다양한 컴퓨터 비전 작업에서 우수한 성능을 나타냈습니다. 다른 최신 방법들과 비교하여, MID는 보다 안정적이며 변동성이 높은 환경에서도 뛰어난 적응력을 보여 주었습니다. 또한, bioinformatics 데이터와 같은 다학제적 작업에서 일관된 첨단 성능을 유지하고 있습니다.



### VesSAM: Efficient Multi-Prompting for Segmenting Complex Vess (https://arxiv.org/abs/2511.00981)
- **What's New**: VesSAM은 임상 응용을 위한 2D 혈관 세분화를 목표로 하는 강력하고 효율적인 프레임워크입니다. 기존의 Segment Anything Model(SAM)에 비해 혈관 구조에 더 잘 적응하도록 설계된 VesSAM은 지역적 질감 기능을 향상시키기 위한 convolutional adapter와 해부학적 프롬프트를 융합하는 multi-prompt encoder를 통합합니다. 이 새로운 접근은 더욱 정교한 혈관 세분화를 가능하게 하며, 기존의 SAM 변형들보다 성능이 10% 이상 개선되었음을 보여줍니다.

- **Technical Details**: VesSAM의 핵심 혁신에는 (1) 지역 질감을 민감하게 잡아주는 convolutional adapter, (2) 해부학적 프롬프트를 계층적 cross-attention을 통해 통합하는 multi-prompt encoder, (3) 거친 경계를 줄이기 위한 lightweight mask decoder가 포함됩니다. 이러한 구성 요소들은 비등방성 혈관 구조의 세분화에서 성능을 극대화하기 위한 것입니다. 또한, 자동화된 멀티 프롬프트 어노테이션 생성 파이프라인과 다섯 가지 이미징 모달리티에 걸쳐 있는 폭넓은 벤치마크 데이터셋도 구축되었습니다.

- **Performance Highlights**: VesSAM은 State-of-the-art PEFT 기반의 SAM 변형들보다 평균적으로 10%의 Dice 및 13%의 IoU 향상을 보여주었으며, 전체 최적화된 방법들과도 경쟁력 있는 성능을 기록했습니다. 또한, 이 프레임워크는 out-of-distribution(OoD) 환경에서도 우수한 일반화 성능을 보이며 평균 OoD Dice와 IoU에서 모든 기준선을 초과하는 성과를 보였습니다. 이는 VesSAM이 임상에서의 손쉬운 응용 가능성을 높임을 시사합니다.



### A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis (https://arxiv.org/abs/2511.00962)
Comments:
          NeurIPS 2025 poster

- **What's New**: 이번 논문에서는 비디오 이상 탐지(video anomaly detection, VAD), 공간 이상 위치 지정(video anomaly localization, VAL), 및 텍스트 기반 이해(video anomaly understanding, VAU) 간의 간극을 메우는 통합적인 추론 프레임워크를 제안합니다. 이 방법은 추가적인 훈련 없이도 전반적인 이상 분석을 가능하게 하는 제약된(싫마) 테스트 시간(reasoning process)을 통해 세 가지 주요 작업을 연결하는 특징이 있습니다. 이는 비디오 이상 탐지의 정확성을 향상시키고 사용자에게 보다 해석 가능하고 직관적인 결과를 제공함으로써 기존 방법의 한계를 극복합니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 단계로 구성되어 있습니다. 처음에는 비디오 수준에서 이상 확률을 계산하고 가장 의심스러운 세그먼트에 대한 컨텍스트 태그를 추출하는 초기 VAD 단계입니다. 그 다음, 컨텍스트 태그 리스트와 초기 이상 점수를 활용하여 조건부 점수 조정을 수행하는 점수 게이트 정제(score-gated refinement) 단계가 이어지며, 마지막으로 VAL과 VAU 작업을 위한 결합된 이상 점수와 컨텍스트 태그 리스트가 생성되어 텍스트 및 비주얼 프롬프트를 동적으로 조정합니다.

- **Performance Highlights**: UCF-Crime, XD-Violence, UBnormal, MSAD 데이터셋에 대한 실험 결과, 제안된 프레임워크는 여러 비디오 이상 탐지, 위치 지정 및 이해 벤치마크에서 최상의 성능을 나타냈습니다. VAD의 AUC 점수가 4-6% 향상되었고, VAL 및 VAU 작업에서도 다양한 지표에서 일관된 개선 사항을 보였습니다. 이는 훈련 없이도 통합된 비디오 이상 분석 프레임워크가 여러 도메인 및 작업에서 해석 가능하고 확장 가능하며 강인하다는 것을 보여줍니다.



### EVTAR: End-to-End Try on with Additional Unpaired Visual Referenc (https://arxiv.org/abs/2511.00956)
- **What's New**: 이번 논문에서는 EVTAR라는 새로운 End-to-End 가상 피팅 모델을 제안합니다. EVTAR는 추가적인 참조 이미지를 활용하여 사용자가 타겟 의상을 실제 이미지에 더 정확하게 적합할 수 있도록 합니다. 기존의 가상 피팅 기술들이 다양한 외부 모델에 의존하는 것과는 달리, EVTAR는 소스 이미지와 목표 의상 입력만으로 간단한 추론이 가능합니다.

- **Technical Details**: EVTAR는 DiT라는 확장 가능한 Transformer 아키텍처를 기반으로 하여, 이미지들을 잠재 공간(latent space)으로 인코딩한 후 토큰(token)으로 패치하여 다루는 과정을 포함합니다. 이 과정에서 반복적인 노이즈 처리(denoising) 기술을 통해 의상의 분포를 학습합니다. 또한, EVTAR는 추가적인 조건 없이도 효과적으로 이미지 복원을 수행할 수 있도록 설계되었습니다.

- **Performance Highlights**: 평가 결과, EVTAR는 정량적 및 정성적 평가 모두에서 최첨단(SOTA) 성능을 달성하였으며, 다양한 실 환경 이미지에서도 뛰어난 결과를 보여줍니다. 본 연구는 보충 참조 이미지와 인물 이미지를 사용한 훈련 데이터 세트를 통해 고품질의 의상 전환이 가능하다는 것을 입증하였습니다. 이는 가상의 피팅 기술의 실제 응용 가능성을 크게 향상시킵니다.



### Dynamic Multi-level Weighted Alignment Network for Zero-shot Sketch-based Image Retrieva (https://arxiv.org/abs/2511.00925)
- **What's New**: 본 논문에서는 제로샷 스케치 기반 이미지 검색(ZS-SBIR) 문제에 대한 새로운 접근 방식인 동적 다중 레벨 가중 정렬 네트워크(Dynamic Multi-level Weighted Alignment Network)를 소개합니다. 이 방법은 모달리티의 불균형 샘플 및 일관성 없는 저품질 정보를 사용할 때 발생하는 성능 저하 문제를 해결하기 위해 개발되었습니다. 세 가지 주요 구성 요소로는 유니모달 특징 추출 모듈, 교차 모달 다중 레벨 가중 모듈, 가중 쿼드러플릿 손실 모듈이 포함됩니다.

- **Technical Details**: 제안하는 방법은 주어진 스케치와 이미지의 쌍을 훈련 배치 내에서 동적으로 가중치를 부여하고, 각 쌍의 정렬 품질을 기반으로 교차 모달 쿼드러플릿을 생성합니다. 이를 위해 로컬 및 글로벌 집계 블록을 사용하여 정렬 품질을 측정하고, 가중 쿼드러플릿 손실 모듈을 통해 다양한 모달리티의 샘플 수를 균형 있게 조정합니다. 이러한 전처리 과정은 모델이 제로샷 일반화 능력을 보다 향상시킵니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋인 Sketchy, TU-Berlin 및 QuickDraw에서 수행한 실험 결과, 제안한 방법이 최신 ZS-SBIR 방법들보다 우수한 성능을 보였습니다. 이는 제안된 동적 가중 전략과 가중 쿼드러플릿 손실이 모델의 성능을 크게 개선하는 데 기여했음을 의미합니다. 전반적으로, 연구팀은 제안한 접근 방법이 ZS-SBIR 분야에서의 성능 향상에 중요한 역할을 한다고 주장합니다.



### Fleming-VL: Towards Universal Medical Visual Reasoning with Multimodal LLMs (https://arxiv.org/abs/2511.00916)
- **What's New**: 최근 연구자들은 Multimodal Large Language Models (MLLMs)를 의료 대화 능력을 갖춘 모델로 발전시키는 데 주목하고 있습니다. 이러한 모델은 전통적인 의료 데이터의 이질적인 특성으로 인해 통합된 접근 방식이 필요한 상황에서 큰 가능성을 지니고 있습니다. 본 논문에서는 이를 해결하기 위해 Fleming-VL이라는 통합적 End-to-End 프레임워크를 제안하며, 다양한 데이터 형태 간에서의 이해를 한 곳에서 가능하게 합니다.

- **Technical Details**: Fleming-VL은 2D 이미지, 3D 볼륨 스캔, 그리고 동영상 등을 하나의 통합된 프레임워크에서 이해할 수 있도록 설계되었습니다. 데이터 중심의 접근 방식을 통해 긴 컨텍스트 데이터와 희귀 의료 데이터를 통합하고, 기존의 평가 프레임워크를 3D 볼륨 및 동영상 이해 기준으로 확장하는 방식으로 구현됩니다. 또한, Supervised Fine-Tuning (SFT)과 Group Relative Policy Optimization (GRPO) 기법을 활용하여 여러 모델 규모로 개발되었습니다.

- **Performance Highlights**: Fleming-VL은 다양한 의료 이미징 시나리오에서 시험을 통해 뛰어난 성능을 입증하였습니다. 2D 이미지, 3D 볼륨, 영상 형식의 9개 벤치마크에서 최첨단 성능을 달성하며, 기존 모델들에 비해 교차 모달 분석에서 뛰어난 균형성을 보여줍니다. 이러한 결과는 의료 AI 분야에서 실제 적용 가능한 솔루션으로 자리매김할 가능성을 높이고 있습니다.



### GraphGeo: Multi-Agent Debate Framework for Visual Geo-localization with Heterogeneous Graph Neural Networks (https://arxiv.org/abs/2511.00908)
- **What's New**: 이번 연구에서는 GPS 메타데이터 없이 이미지의 지리적 위치를 식별하는 시각적 지리 위치 확인을 위한 새로운 방법인 GraphGeo를 제안합니다. GraphGeo는 이질적인 그래프 신경망(heterogeneous graph neural networks)을 활용하여 다수의 에이전트 간의 대화를 통해 정보의 신뢰성을 높이는 접근방식입니다. 이 시스템은 다양한 논쟁 관계를 모델링 하여 지리적 위치 예측의 정확성을 향상시킵니다.

- **Technical Details**: GraphGeo는 에이전트 간의 협업을 구조화된 논쟁으로 모델링하며, 이를 통해 에지(edge) 유형을 세분화하여 상호 지지, 경쟁 논의 및 지식 전이의 세 가지 주요 관계를 설정합니다. 이중 수준의 논쟁 메커니즘(node-level & edge-level)은 노드 수준에서의 표현 개선과 에지 수준에서의 논쟁 상태 모델링을 결합하여 동적 변화에 대응합니다. 이에 덧붙여 교차 수준(topology refinement)을 통한 노드 표현과 그래프 구조 간의 상호 작용을 촉진합니다.

- **Performance Highlights**: 다양한 벤치마크 실험을 통해 GraphGeo는 기존의 최첨단 방법들에 비해 상당한 성능 향상을 보여주었습니다. 특히, 서로 다른 기계의 예측을 비교하고 논쟁을 통해 문제를 해결함으로써 긍정적인 결과를 도출했습니다. 이 결과는 구조화된 논쟁이 충돌하는 예측을 정확하게 개선하는 데 기여함을 입증합니다.



### Layer-Wise Modality Decomposition for Interpretable Multimodal Sensor Fusion (https://arxiv.org/abs/2511.00859)
Comments:
          Accepted to NeurIPS 2025

- **What's New**: 이 논문에서는 자율 주행에서 센서 융합 모델의 예측에 대한 해석 가능성을 높이기 위해 Layer-Wise Modality Decomposition (LMD)라는 새로운 방법을 제안합니다. LMD는 다양한 센서 모달리티의 기여도를 분리하여 모델의 모든 층에서 정보를 해석할 수 있도록 합니다. 이 방법은 기존의 융합 모델에 개조 없이도 각각의 센서 모달리티에 대한 명확한 기여도를 제공합니다.

- **Technical Details**: LMD는 Layer-Wise Relevance Propagation (LRP) 및 Deep Taylor Decomposition (DTD) 프레임워크를 기반으로 하여, 고차원 다중 모달 네트워크에서 각 레이어에 대한 모달리티별 기여도를 효과적으로 분리합니다. 이 방식을 통해 네트워크로부터 직접적으로 기여도 해석을 수행하고, 이를 통해 모델의 예측 결정을 더 잘 이해할 수 있습니다. LMD는 nuScenes 벤치마크에서 검증되어 기존의 멀티모달 퍼셉션 모델에서도 효과적으로 적용될 수 있음을 입증하였습니다.

- **Performance Highlights**: LMD의 성능은 정량적 및 정성적 실험을 통해 평가되었습니다. 새로운 혼란 기반 메트릭을 도입하여 해석 가능성 평가에 기여하며, 다양한.bias-splitting 전략을 적용한 LMD 변형을 통해 모달리티 분리에 대한 효과성을 검증하였습니다. 이 방법은 고용량 모델의 성능을 유지하면서도 기존의 복잡한 비선형 관계를 제어할 수 있는 중요한 도구로 자리잡을 것입니다.



### Occlusion-Aware Diffusion Model for Pedestrian Intention Prediction (https://arxiv.org/abs/2511.00858)
Comments:
          This manuscript has been accepted to the IEEE Transactions on Intelligent Transportation Systems as a regular paper

- **What's New**: 본 논문에서는 보행자의 횡단 의도를 예측하기 위해 새로운 'Occlusion-Aware Diffusion Model (ODM)'을 제안합니다. 이 모델은 가려진 동작 패턴을 재구성하고 향후 의도 예측을 안내하기 위해 활용됩니다. 특히, 기존 연구들이 간과했던 불완전한 관찰 상황에서도 보행자의 의도를 정확히 예측할 수 있는 가능성을 열어줍니다.

- **Technical Details**: ODM은 'occlusion-aware diffusion transformer' 아키텍처를 사용하여 가려진 패턴과 관련된 노이즈 기능을 추정합니다. 이 과정을 통해 문맥적 관계를 보다 잘 포착할 수 있도록 개선됩니다. 또한, 'occlusion mask-guided reverse process'를 도입하여 관찰 정보의 활용을 극대화하고 예측 오차의 축적을 줄이는 데 기여합니다.

- **Performance Highlights**: 다양한 가리기 신호 시나리오에서 제안된 방법의 성능을 광범위하게 평가하였으며, PIE 및 JAAD와 같은 인기 있는 벤치마크와 비교하여 기존 방법들보다 더 강력한 성능을 달성했습니다. 실험 결과, 본 방법이 가려진 시나리오에서 보행자 횡단 의도 예측의 정확성을 유의미하게 향상시킴을 보여줍니다.



### OmniBrainBench: A Comprehensive Multimodal Benchmark for Brain Imaging Analysis Across Multi-stage Clinical Tasks (https://arxiv.org/abs/2511.00846)
- **What's New**: 본 논문에서는 뇌 영상 분석을 위한 종합 멀티모달 시각 질문-응답(visual question answering, VQA) 벤치마크인 OmniBrainBench를 소개합니다. 이 벤치마크는 30개 검증된 의학 출처에서 수집된 15개의 뇌 영상 모달리티를 포함하여 총 9,527개의 검증된 VQA 쌍과 31,706개의 이미지를 제공합니다. OmniBrainBench는 임상 작업 흐름을 모사하며, 전문 방사선의에 의해 엄격히 검증된 15개의 다단계 임상 과제를 포괄합니다.

- **Technical Details**: OmniBrainBench는 CT, MRI, PET, SPECT 등 15개의 이미징 모달리티를 제공하여 임상 요구를 충족하는 종합적인 평가 프레임워크를 구성합니다. 이 벤치마크는 해부학적 구조 식별, 질병 진단, 병변 위치 파악 등 5단계의 전문 임상 과제를 포함하고 있으며, 다차원 평가 기준을 통해 MLLMs의 효과를 종합적으로 평가할 수 있습니다. 24개의 최신 MLLM 모델을 평가하여, 이 모델들의 다양한 뇌 영상 데이터에서의 성능을 비교 분석합니다.

- **Performance Highlights**: 연구 결과, 상용 MLLM(예: GPT-5)은 공개 소스 및 의학 모델보다 우수하지만 전문의에는 미치지 못하는 것으로 나타났습니다. 또한, 의학 MLLM의 성능은 매우 다양하며, 공개 소스 MLLM은 전반적으로 열세하나 특정 작업에서 두각을 나타냅니다. 복잡한 수술 전 작업에서는 MLLM의 성능이 크게 저하되며, 이는 시각-임상 추론 간의 간극을 드러내는 결과입니다.



### Parameter Interpolation Adversarial Training for Robust Image Classification (https://arxiv.org/abs/2511.00836)
Comments:
          Accepted by TIFS 2025

- **What's New**: 본 연구에서는 Parameter Interpolation Adversarial Training (PIAT)이라는 새로운 프레임워크를 제안합니다. PIAT는 각 에포크(epoch) 간 모델 파라미터를 보간(interpolate)하여 조정하며, 이는 결정 경계를 보다 완만하게 변화시켜 과적합(overfitting) 문제를 완화합니다. 이 방법은 모델의 수렴(convergence)을 돕고 더 높은 강인성(robustness)을 달성하는 데 기여합니다.

- **Technical Details**: PIAT는 이전과 현재 에포크의 모델 파라미터를 보간하여 조정하며, 초기 훈련 단계에서는 현재 파라미터에 더 집중하고 이후 단계에서는 이전 파라미터의 가중치를 증가시켜 결정 경계의 복잡성을 줄입니다. 또한, 새로운 정규화(metric)인 Normalized Mean Square Error (NMSE)를 제안하여 클린(clean) 예제와 적대적(adversarial) 예제 간의 로그레트(logits) 크기의 상대적 비율을 일치시킵니다. 이는 절대 크기보다 상대 크기 alignment를 더 중요하게 다룹니다.

- **Performance Highlights**: 다양한 기준 데이터셋에서 수행된 실험을 통해 PIAT 프레임워크가 CNN(Convolutional Neural Networks) 및 ViT(Vision Transformers) 모두에서 강인성을 크게 향상시켰음을 확인했습니다. PIAT는 복잡한 결정 경계를 피하고 모델의 수렴성을 안정화시켜, 학습 과정에서 나타나는 과적합 문제를 해결하는 데 매우 효과적입니다. 이러한 실험 결과들은 PIAT가 일반적이며 효과적인 방법임을 증명합니다.



### Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials (https://arxiv.org/abs/2511.00833)
Comments:
          NeurIPS 2025

- **What's New**: 이번 논문은 Visual-Contrast Attention (VCA)을 소개하며, 이 기술은 기존의 Multi-Head Self-Attention (MHSA) 레이어를 대체할 수 있는 방식으로, 이미지 분류와 생성에서 성능을 획기적으로 개선합니다. VCA는 O(N^2C)에서 O(Nnc)로 이론적 복잡도를 줄이며, 이 과정을 통해 모델의 인식 성능을 향상시키고 추가적인 계산 비용 없이 파라미터를 조정합니다.

- **Technical Details**: VCA는 두 개의 포지셔널 임베딩을 도입하여 정보를 정제하고, 각 헤드에서 시각적-대조 토큰을 평균 풀링하여 생깁니다. 이로 인해, 쿼리 필드를 압축하고 단순화하여 계산의 효율성을 증가시키며, 후속 단계에서 역대조 작업을 통해 정보 배열을 최적화합니다. 이러한 구조적 변화를 통해 각 패치가 주목받는 정도를 측정할 수 있습니다.

- **Performance Highlights**: VCA는 이미지 분류 작업에서 DeiT-Tiny 모델의 top-1 정확도를 72.2%에서 75.6%로 증가시키고, 다른 계층 구조 모델에서도 최대 3.1%의 성능 향상을 보여주었습니다. 이미지 생성에서도 FID-50K를 기존보다 2.1에서 5.2 포인트 낮추며 효율성과 품질을 동시에 개선하였습니다.



### Enhancing Adversarial Transferability in Visual-Language Pre-training Models via Local Shuffle and Sample-based Attack (https://arxiv.org/abs/2511.00831)
Comments:
          Accepted by NAACL2025 findings

- **What's New**: 이번 논문에서 소개된 Local Shuffle and Sample-based Attack (LSSA)는 시각-언어 사전 학습 모델(VLP)에서 멀티모달 적대적 공격의 효과iveness를 향상시키기 위해 새로운 접근 방식이 제안되었습니다. LSSA는 원본 이미지-텍스트 쌍을 기반으로 적대적 이미지를 생성하고 이를 통해 적대적 텍스트를 생성합니다. 이전 방법들은 입력 다양성이 부족하여 과적합(overfitting) 문제에 시달렸으나, LSSA는 이를 해결합니다.

- **Technical Details**: LSSA는 로컬 이미지 블록을 무작위로 섞고, 주변을 샘플링하여 적대적 이미지를 생성함으로써 입력 다양성을 높입니다. 이 과정에서 원본 텍스트와 생성된 적대적 이미지를 활용하여 높은 전이성(transferability)의 공격을 수행합니다. 실험 결과, LSSA는 다양한 VLP 모델과 데이터셋에서 멀티모달 적대적 예제의 전이성을 크게 향상시켰습니다.

- **Performance Highlights**: LSSA는 여러 모듈에서 기존의 멀티모달 공격방법들을 초월하는 성능을 보여주었습니다. 특히, 이미지 캡셔닝(image captioning)과 비주얼 그라우딩(visual grounding) 태스크에서도 뛰어난 결과를 내었으며, Large Vision-Language Models (LVLMs)에서의 평가에서도 좋은 성과를 거두었습니다. 실험을 통해 LSSA가 다양한 VLP 모델 및 다운스트림 태스크에서 공격 성능을 크게 개선함을 확인하였습니다.



### OMEGA: Optimized Multimodal Position Encoding Index Derivation with Global Adaptive Scaling for Vision-Language Models (https://arxiv.org/abs/2511.00821)
- **What's New**: 최근 발표된 OMEGA 프레임워크는 Modal-Specific Position Encoding (MSPE)과 Global Adaptive Encoding Step Scaling (GAESS)을 도입하여 텍스트와 비주얼 모달리티 각각의 구조를 고려한 새로운 위치 인코딩 방식을 제안하였다. 기존의 1D 및 2D 위치 인코딩 전략은 이질적인 모달리티를 동일하게 처리하여 성능 저하를 초래했지만, OMEGA는 각 모달리티의 고유한 구조를 유지하면서 위치 인덱스를 부여할 수 있도록 설계되었다. 이는 비주얼 질문 응답(Visual Question Answering, VQA) 작업에서 OMEGA의 성능 향상이 더욱 두드러져, 다양한 아키텍처와 벤치마크에서 일관된 성능 개선을 보여준다.

- **Technical Details**: OMEGA는 텍스트와 비주얼 토큰을 위한 별도의 위치 인덱스 차원을 할당하여 모달리티 특유의 세부 사항을 보존하는 Modality-Specific Position Encoding (MSPE)을 포함한다. GAESS (Global Adaptive Encoding Step Scaling)는 텍스트와 비주얼 토큰의 임베딩 엔트로피를 기반으로 비주얼 토큰의 위치 인코딩 단계 크기를 조정하여 모달리티 간 정보 밀도를 일치시킨다. 이러한 접근 방식은 OMEGA가 특정 모델 아키텍처나 주의 메커니즘을 변경하지 않고도 성능을 개선할 수 있게 해준다.

- **Performance Highlights**: 다양한 VQA 벤치마크에서 OMEGA는 기초 위치 인코딩 방식에 비해 최대 3.43% 성능 향상을 이루었으며, 이는 Qwen2.5-VL-3B와 같은 비주얼 집약적 작업에 특히 두드러진다. 또한, OMEGA는 더 큰 모델인 Qwen2.5-VL-7B 및 LLaVA-v1.5-7B에서도 일관된 성능 향상을 보여주었다. 이와 같은 성능 개선은 기존의 인코딩 전략과 비교하여 매우 의미 있는 결과이다.



### TA-LSDiff:Topology-Aware Diffusion Guided by a Level Set Energy for Pancreas Segmentation (https://arxiv.org/abs/2511.00815)
Comments:
          14 pages, 7 figures

- **What's New**: 이번 논문에서는 TA-LSDiff라는 새로운 모델을 제안합니다. 이 모델은 전통적인 level set 방법과 topology-aware diffusion probabilistic model을 결합하여, 명시적인 기하학적 진화를 필요로 하지 않고도 췌장을 정확하게 세분화합니다. 또한, 두 가지 주요 접근 방식을 통합하여 췌장 세분화의 새로운 패러다임을 제공합니다.

- **Technical Details**: TA-LSDiff는 네 가지 보완적인 용어로 구성된 에너지 함수에 기반하여 작동합니다. 이 에너지는 췌장과 주변 조직을 구별하고 경계의 부드러움을 유지하며 세분화 크기를 제어하고 정밀한 국소화를 통해 배경 누출을 방지하는 데 도움을 줍니다. 또한, 주변 픽셀 정보를 활용하는 pixel-adaptive refinement 모듈을 통해 경계 결정의 안정성과 정확성을 향상시킵니다.

- **Performance Highlights**: 네 개의 공개 췌장 데이터 세트에 대한 평가 결과, TA-LSDiff는 기존 방법보다 뛰어난 정확도를 달성했습니다. 이 모델은 복잡한 해부학적 구조를 효과적으로 처리할 수 있으며, 세부 구조와 바람직한 위상을 보존하는 데 큰 장점을 가지고 있습니다. 연구 결과는 TA-LSDiff가 췌장 세분화를 위한 실용적이고 정확한 해결책임을 입증합니다.



### GUI-AIMA: Aligning Intrinsic Multimodal Attention with a Context Anchor for GUI Grounding (https://arxiv.org/abs/2511.00810)
- **What's New**: 본 연구에서는 GUI-AIMA라는 새로운 접근 방식을 제안하는데, 이는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 내재적 주의(attention) 메커니즘을 활용하여 GUI 기초 작업을 보다 효율적으로 수행할 수 있도록 한다. GUI-AIMA는 텍스트 기반 좌표 생성(task) 방식 대신, 시각적 패치(visual patches)를 선택하고 그 안에서 클릭 위치를 결정하는 방식으로 진행된다. 이 프레임워크는 85,000개의 스크린샷에 대해 교육되어 데이터 효율성(data efficiency)을 보여주며, 경량 모델(light model)이 MLLM의 내재적 기초 능력을 유도할 수 있음을 검증하였다.

- **Technical Details**: GUI-AIMA는 패치 기반(supervision) 학습을 통해 MLLM의 다중 헤드 자기 주의(multi-head self-attention)를 훈련시키며, 특히 주의 헤드 가중치(attention head weighting) 메커니즘을 사용하여 쿼리-시각 쿼리의 상관관계에 따라 각 주의 헤드의 중요도를 조정한다. 이를 통해 좌표 없는 GUI 기초 작업을 용이하게 수행할 수 있다. 학습 과정에서 특정 토큰에 대한 주의가 시각적 토큰에 집중될 수 있도록 하여 데이터 효율적으로 훈련할 수 있게 한다.

- **Performance Highlights**: GUI-AIMA는 ScreenSpot-Pro에서 평균 정확도 58.6%, OSWorld-G에서 62.2%를 달성하며, 3B 모델 중에서 최첨단 성능을 기록하였다. 특히, 전통적인 좌표 기반 방식과 비교했을 때, GUI-AIMA는 4.5%의 성능 향상을 보여준다. 이 모델은 상대적으로 적은 데이터로도 강력한 성능을 발휘하며 시각적 피드백에 기반한 인간의 행동을 모방해 더 효과적인 GUI 기준을 제공한다.



### Med-Banana-50K: A Cross-modality Large-Scale Dataset for Text-guided Medical Image Editing (https://arxiv.org/abs/2511.00801)
- **What's New**: 최근 다중 모달 대형 언어 모델(MLLMs)의 발전은 놀라운 의료 이미지 편집 기능을 가능하게 했으나, 의료 이미지 편집을 위한 대규모 고품질 개방형 데이터셋이 부족하여 연구가 제약 받고 있습니다. 이와 관련하여 우리는 Med-Banana-50K라는 50,000 이미지로 구성된 포괄적인 데이터셋을 소개합니다. 이 데이터셋은 3가지 모달리티(흉부 X선, 뇌 MRI, 망막 촬영)와 23종 질병을 포괄하며, 실제 의료 이미지를 기반으로 생성된 양방향 편집을 포함합니다.

- **Technical Details**: Med-Banana-50K는 의료 데이터의 품질 유지를 위해 LLM-as-Judge를 활용한 체계적인 접근 방식을 적용했습니다. 각 이미지는 병리 유형 및 모달리티별로 조직되어 있으며, 정확한 의학적 평가 기준에 따라 평가됩니다. 이미지 편집 요청은 Gemini-2.5-Pro를 통해 생성되며, 이후 Gemini-2.5-Flash-Image로 편집됩니다. 이러한 프로세스는 여러 차례의 피드백을 통해 의도된 질병 표현을 보장합니다.

- **Performance Highlights**: Med-Banana-50K는 고품질 의료 이미지 편집의 학습 및 평가에 필요한 기반을 마련합니다. 이를 통해 50,635개의 성공적인 예제와 37,000개의 실패한 편집 시도를 포함하여 직관적 선호 학습 및 정렬 연구를 지원합니다. 이 데이터셋은 의료 도메인 적응을 위한 기준을 제시하고, 기존의 일반 도메인 편집 데이터셋에서 다루지 않는 고유한 요구 사항을 충족합니다.



### FedOnco-Bench: A Reproducible Benchmark for Privacy-Aware Federated Tumor Segmentation with Synthetic CT Data (https://arxiv.org/abs/2511.00795)
Comments:
          Published in IEEE

- **What's New**: 이번 연구에서는 FedOnco-Bench라는 데이터베이스를 소개하며, 이는 암 관련 CT 스캔 데이터셋을 사용하여 개인정보 보호를 고려한 분산 학습(Federated Learning, FL)을 위한 평가 기준을 제공합니다. FedOnco-Bench는 암 종양 분할(segmentation) 성능과 개인 정보 유출을 평가하며, 다양한 FL 방법론(FedAvg, FedProx, FedBN 및 DP-SGD가 결합된 FedAvg)에 대해 연구합니다. 결과적으로, 개인 정보 보호와 유용성 사이의 뚜렷한 균형이 드러나며, FL의 취약점에 대한 새로운 통찰력을 제공합니다.

- **Technical Details**: FedOnco-Bench는 비동기식 크로스-사일로 FL 시스템을 시뮬레이션합니다. 본 연구에서는 여러 클라이언트와 중앙 서버를 포함하여 각 클라이언트가 로컬 데이터로 모델을 학습하고 업데이트를 서버에 전달합니다. FL 알고리즘 가운데, FedAvg는 클라이언트 모델 업데이트를 간단한 평균을 통해 집계하는 기본 알고리즘으로 사용되며, FedProx와 FedBN은 비동질적(non-IID) 데이터 처리의 어려움을 해결하기 위한 방식으로 선택되었습니다.

- **Performance Highlights**: 실험 결과는 FedAvg는 뛰어난 성능(Dice coefficient 약 0.85)을 보이지만, 개인 정보 유출(AUC 약 0.72) 위험이 더 높았습니다. 반면에 DP-SGD는 개인 정보 보호(AUC 약 0.25) 수준을 높이는 대신 정확도(Dice 약 0.79)가 감소하는 결과를 보여주었습니다. FedProx 및 FedBN은 비동질적 데이터 환경에서 균형 잡힌 성능을 제공하며, 각각의 알고리즘 성능을 측정하기 위해 다양한 메트릭이 포함되어 있습니다.



### Class-agnostic 3D Segmentation by Granularity-Consistent Automatic 2D Mask Tracking (https://arxiv.org/abs/2511.00785)
Comments:
          Under review in Pattern Recognition

- **What's New**: 이번 논문에서는 3D 인스턴스 세분화(3D instance segmentation)를 위한 새로운 방법인 Granularity-Consistent Segmentation Policy를 소개합니다. 이 방법은 자동 2D Mask Tracking을 통해 동영상 프레임 간의 시계적 연관성을 유지하며 일관된 3D 가짜 라벨(pseudo labels)을 생성합니다. 이를 통해 기존 방법에서 발생하는 세그멘테이션의 불일치 문제를 해결하고 있습니다. 또한 3단계 커리큘럼 학습(curriculum learning) 프레임워크를 도입하여 점진적으로 다양한 뷰의 주석을 통합하는 방식으로 모델을 학습시킵니다.

- **Technical Details**: Granularity-Consistent Segmentation Policy는 객체를 자동으로 추적하여 동영상의 모든 프레임 간에 일관된 2D 마스크(mask)를 생성합니다. 이 과정에서는 동일한 객체가 다른 프레임에서 다르게 분할되는 현상을 해결하여 전체 장면에서의 지오메트리적 일관성을 확보합니다. 모델은 처음에 단편적인 단일 뷰 데이터에서 학습을 시작하여, 점진적으로 다중 뷰 주석(multi-view annotations)을 통한 전체 장면 감독(full-scene supervision)으로 이전해 나갑니다. 이를 통해 초기의 상충되는 2D 선행 정보(prior)로부터 일관된 3D 표현을 증류(distill)할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 ScanNet200과 ScanNet++와 같은 표준 벤치마크에서 최고의 성능을 기록하였습니다. 결과적으로 생성된 3D 가짜 라벨이 기존 방법보다 더 정확한 것으로 나타났습니다. 또한, 우리의 방법은 오픈-보캐뷸러리(open vocabulary) 능력을 검증하여, 제한된 학습 샘플로도 희귀 객체를 인식하고 세밀한 객체 분류에서도 뛰어난 성능을 보여주었습니다.



### A Hybrid YOLOv5-SSD IoT-Based Animal Detection System for Durian Plantation Protection (https://arxiv.org/abs/2511.00777)
- **What's New**: 이 연구는 두리안 농장의 동물 침입 문제를 해결하기 위한 IoT(Internet of Things) 기반의 새로운 동물 감지 시스템을 제안합니다. 기존 시스템의 한계를 극복하기 위해 YOLOv5와 SSD(object detection algorithms)를 통합하여 감지 정확도를 향상했습니다. 이 시스템은 실시간 모니터링을 제공하며, 탐지된 침입에 대한 알림이 Telegram을 통해 농부에게 자동으로 전송됩니다.

- **Technical Details**: 제안된 시스템은 YOLOv5와 SSD 알고리즘을 결합하여 동물 감지를 수행합니다. 발견된 동물의 종류에 따라 호랑이의 울음소리와 같은 자동화된 사운드 메커니즘이 활성화됩니다. 성능 시험 결과, 코끼리, 멧돼지, 원숭이의 정확도는 각각 90%, 85%, 70%로 나타났습니다.

- **Performance Highlights**: 시스템은 주간에 가장 높은 정확도를 보이며 야간에는 정확도가 저하되는 경향이 있습니다. 정지 이미지와 비디오 모두에서 감지가 이루어지나, 낮에는 더 높은 성능을 발휘합니다. 전반적으로 이 연구는 탐지, 알림 및 억제 기능을 결합한 종합적이고 실용적인 프레임워크를 제공하여 자동화된 농업 솔루션의 향후 혁신을 위한 길을 열어줍니다.



### Erasing 'Ugly' from the Internet: Propagation of the Beauty Myth in Text-Image Models (https://arxiv.org/abs/2511.00749)
Comments:
          This is a preprint under review

- **What's New**: 이 논문은 소셜 미디어가 서양의 미의 기준을 어떻게 강화하고 있으며, 특히 여성과 소녀들에게 부정적인 자아상이 나타나게 하는지를 다룹니다. 생성적 AI 모델이 '아름다움'을 어떻게 인코딩하고 '추함'을 지우는지 연구하며, 이를 위해 두 개의 이미지 생성 파이프라인을 개발했습니다. 이 연구는 AI에 의해 생성된 이미지가 사회적 미의 기준에 미치는 영향을 탐구합니다.

- **Technical Details**: 연구자들은 텍스트-이미지 모델과 텍스트-언어 모델-이미지 모델의 두 가지 생성 파이프라인을 사용하여 5984개의 이미지를 생성했습니다. 생성된 이미지들은 남녀뿐만 아니라 비바이너리 개인들을 포함해 다양한 범주에 걸쳐 평가되었습니다. 참여자들은 리커트 척도를 통해 1200개의 이미지에 대한 평점을 제공했습니다.

- **Performance Highlights**: 결과적으로, 생성된 이미지의 86.5%가 피부색이 밝으며 22%는 명시적인 콘텐츠를 포함하고 있었고, 74%는 더 젊은 연령대의 외모로 평가받았습니다. 특히 비바이너리 개인의 이미지는 더 젊고 과도하게 성적화된 것으로 평가되어, 미의 기준과 상관된 편향이 존재함을 보여줍니다. 부정적인 아름다움 특성을 가진 프롬프트는 일관되게 더 높은 NSFW 등급을 생성했습니다.



### Towards classification-based representation learning for place recognition on LiDAR scans (https://arxiv.org/abs/2511.00738)
- **What's New**: 이번 연구에서는 기존의 contrastive learning(대조 학습) 대신, 장소 인식을 multi-class classification(다중 클래스 분류) 문제로 정의하여 새로운 접근 방식을 제안합니다. 이 방법은 LiDAR 스캔에 이산 위치 레이블을 지정하고, 인코더-디코더 모델을 훈련시켜 각 스캔의 위치를 직접 분류하도록 합니다. 이를 NuScenes 데이터셋에서 평가한 결과, 기존 방법들과 비교할 때 경쟁력 있는 성능을 보이며 훈련 효율성과 안정성에서 장점을 제공합니다.

- **Technical Details**: 우리는 LiDAR 스캔 및 이미지와 같은 센서 데이터를 활용하여 지리적 좌표를 결정하는 장소 인식 작업을 수행합니다. 입력 데이터는 N×3 행렬로 표현된 3D 포인트 클라우드로 이루어져 있으며, 이 방식은 자율주행차와 로봇 시스템에서의 실시간 처리에 적합하도록 계산 효율성을 고려해야 합니다. GPS 신호의 불안정성을 극복하기 위해, 독립적인 장소 인식 시스템을 개발하는 것이 중요합니다.

- **Performance Highlights**: 이 연구에서 제안하는 방법은 NuScenes 데이터셋을 통해 다중 클래스 분류의 효과적인 정의와 최적화를 보여주며, CrossLoc3D 모델과 같은 최신 모델에 비해 강력한 성능을 기록했습니다. 또한, masked cross-entropy loss(마스크된 교차 엔트로피 손실)를 도입하여 안정적인 인코더 훈련을 보장하고, 대규모 생산 환경에서의 데이터 통합을 통해 실제 응용 가능성을 입증하였습니다.



### Validating Deep Models for Alzheimer's 18F-FDG PET Diagnosis Across Populations: A Study with Latin American Data (https://arxiv.org/abs/2511.00728)
Comments:
          7 pages, 2 figures

- **What's New**: 이 연구는 신경영상 데이터, 특히 18F-FDG PET 스캔을 이용한 알츠하이머병(AD) 진단을 위한 깊은 학습 모델의 일반화 성능을 평가합니다. 기존의 데이터셋과는 달리, 부에노스 아이레스의 FLENI 연구소에서 수집한 라틴 아메리카 임상 코호트에 대한 성능을 비교하며, 모든 모델이 ADNI에서 높은 AUC 점수를 얻었음에도 불구하고 FLENI 데이터셋에서는 성능이 상당히 저하된 것을 발견했습니다.

- **Technical Details**: 이 연구에서는 컨볼루션 신경망(CNN), Transformer 기반 아키텍처, 그리고 경량화된 ResNet 변형을 포함한 세 가지 모델을 ADNI 데이터셋에 훈련시키고 FLENI 데이터셋에서 평가하였습니다. 실험 결과, 각 아키텍처는 유사한 성능을 보여 Transformer의 특정 작업에 대한 장점을 의문시하게 됩니다. 때때로 이미지 정규화와 적절한 샘플링 선택이 일반화에 중요한 역할을 한다는 것을 발견했습니다.

- **Performance Highlights**: 모델 성능이 FLENI 데이터셋에서 일관되게 저하되는 것으로 나타나, 임상적으로 이용 가능한 모델을 평가하기 위한 표준 벤치마크만으로는 충분하지 않음을 보여주었습니다. 또한, ADNI에서 훈련된 모델들은 AD 클래스에 대해 고전적인 히포메타볼릭 영역에 집중하지만, 다른 클래스나 FLENI 스캔에서는 그 초점이 불분명해지는 경향이 있습니다. 이 연구 결과는 다양한 인구 집단을 고려한 모델 검증의 필요성을 강조합니다.



### Toward Better Optimization of Low-Dose CT Enhancement: A Critical Analysis of Loss Functions and Image Quality Assessment Metrics (https://arxiv.org/abs/2511.00698)
- **What's New**: 이 논문에서는 기존의 Low-dose CT (LDCT) 이미지 품질 향상에 대한 손실 함수(loss function) 분석을 통해 이미지 품질 지표와의 불일치를 확인하였습니다. 다양한 손실 함수들이 LDCT 이미지의 노이즈 감소와 구조 복원을 위해 제안되었으나, 이러한 함수들이 실제 시각적 품질 향상에 미치는 영향에 대한 분석이 부족했던 점을 지적합니다. 의료 영상의 특수성을 고려하여 손실 함수와 이미지 품질 지표를 통합할 필요성을 강조하며, 품질 기반의 더 일관된 손실 함수 개발의 방향성을 제시합니다.

- **Technical Details**: 이 논문에서는 LDCT의 이미지 품질 향상을 위한 손실 함수들에 대한 종합적인 평가 프레임워크를 제공하고, 손실 함수가 모델 학습 과정에 미치는 영향을 분석하였습니다. 최적의 모델 파라미터를 도출하기 위해 Mean Squared Error (MSE), Perceptual Loss, Adversarial Loss 등 다양한 손실 함수를 조합하였습니다. 또한, 이미지 품질 평가를 위해 Full Reference (FR)와 No Reference (NR) 메트릭스를 결합하여 왜곡 수준과 시각적 품질 측면을 모두 고려하였습니다.

- **Performance Highlights**: 이 연구에서 제안한 손실 함수는 시각적 품질 향상에 긍정적인 영향을 미치며, PSNR과 SSIM과 같은 전통적인 메트릭의 한계를 극복할 수 있는 가능성을 보여줍니다. 특히 Mamba를 포함한 모델이 다른 접근 방식보다 뛰어난 성능을 보이며, LDCT 이미지 품질 향상에 기여했습니다. 하지만 여전히 이미지 관련 구조의 시각적 품질을 보장하지 못하는 한계가 있으며, 향후 연구에서 보다 정교한 손실 함수를 통한 성과 향상이 필요합니다.



### Evolve to Inspire: Novelty Search for Diverse Image Generation (https://arxiv.org/abs/2511.00686)
Comments:
          14 pages, 10 figures, Accepted to Neurips 2025 GenProCC Workshop

- **What's New**: 본 논문에서는 WANDER라는 텍스트-이미지 생성 모델을 제안하여 단일 입력 프롬프트(prompt)로부터 다양한 이미지 세트를 생성하는 방법론을 다룹니다. 기존의 텍스트-이미지 디퓨전 모델들이 다양한 이미지 출력을 얻기 어려운 한계점이 있으며, 이러한 문제를 해결하기 위해 LLM(대형 언어 모델)을 사용하여 프롬프트의 의미적 진화를 통해 이미지를 생성합니다. WANDER는 CLIP 임베딩을 활용하여 신규성(novelty)을 정량화하고, emitters를 통해 전이적(과거 프롬프트 주변)인 피드백을 제공합니다.

- **Technical Details**: WANDER의 핵심은 샘플링된 프롬프트를 변형하는 변이(mutation)를 수행하는 것입니다. 모델은 이미지의 다양성을 높이기 위해 LLM과 CLIP 임베딩을 이용하여 이미지의 코사인 거리(cosine distance)를 계산합니다. 또한, emitters는 LLM에게 특정 방식으로 프롬프트를 변형하도록 지시하는 역할을 하며, emitters의 전략이 다양성에 미치는 영향을 극대화합니다. 이를 통해 WANDER는 기존의 진화적 프롬프트 최적화 기법들과 비교하여 현저한 성능 향상을 이룹니다.

- **Performance Highlights**: WANDER는 여러 번의 실험을 통해 이미지를 생성하는 데 있어 높은 다양성을 유지하면서도 적절한 관련성을 갖춘 결과를 산출합니다. 테스트 결과, WANDER는 기존 프롬프트 최적화 기법에 비해 7배 더 적은 토큰을 사용하며, Vendi Score를 통해 평가된 다양성 점수도 보다 우수합니다. 이러한 성과는 WANDER가 이미지 생성에 있어 효과적이고, 모델에 구애받지 않는 접근 방식을 제공함을 보여줍니다.



### Outlier-Aware Post-Training Quantization for Image Super-Resolution (https://arxiv.org/abs/2511.00682)
- **What's New**: 이번 논문에서는 이미지 초해상도(SR) 네트워크의 퀀타이제이션(quantization) 방법 중 포스트-트레이닝 퀀타이제이션(PTQ)을 개선하기 위한 새로운 접근법을 제안합니다. 활성화들에서 아웃라이어(outlier)의 영향을 간과했던 기존 방법들과는 달리, 제안된 방법은 아웃라이어와 밀집 영역(dense region)으로 활성화를 나누어 서로 독립적으로 균일 퀀타이제이션을 적용합니다. 또한, 서로 다른 네트워크 레이어의 퀀타이제이션 민감도를 고려한 민감도 인식 세분화(sensitivity-aware finetuning) 방법을 도입하여 성능을 더욱 향상시킵니다.

- **Technical Details**: 제안된 2중 영역 퀀타이제이션 전략은 아웃라이어와 정상 활성화(normal activations) 사이의 비트 할당을 더 균형 있게 조절하도록 설계되었습니다. 이를 통해 아웃라이어를 유지하는 것이 중요하며, 이는 이미지 색상 정보와 밀접하게 연관되어 있어 직접적으로 제거 시 성능 저하가 발생함을 발견합니다. 각 네트워크 레이어의 다르고 민감한 특성을 반영하여, 더 높은 성능을 위해 민감도 인식 손실 함수를 통해 모델이 유지해야 할 레이어에 중점을 둘 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 SR 네트워크와 데이터셋에서 최신 PTQ 접근법을 초월하는 성능을 보였으며, 대부분의 경우 QAT 방법과 비교 가능한 성능을 유지하면서도 최소 75배의 속도 향상을 달성했습니다. 이는 퀀타이제이션을 통해 메모리 소비를 줄이면서도 추론 속도를 현저히 증가시킨다는 점에서 뛰어난 결과입니다.



### Metadata-Aligned 3D MRI Representations for Contrast Understanding and Quality Contro (https://arxiv.org/abs/2511.00681)
- **What's New**: 본 논문에서는 MRI 대조를 통합적으로 표현할 수 있는 MR-CLIP 프레임워크를 소개합니다. 이 프레임워크는 DICOM 메타데이터와 볼륨 이미지를 정렬하여 MRI 대조를 학습하여, 수동 주석 없이 자동화된 분석을 가능하게 합니다. MR-CLIP은 또한 이미지-메타데이터 임베딩 거리 값을 통해 손상된 메타데이터를 식별함으로써 비지도 형태의 데이터 품질 관리도 지원합니다.

- **Technical Details**: MR-CLIP은 3D 이미지 인코더를 사용하여 볼륨 이미지를 추출하고, 주어진 DICOM 메타데이터를 자연어 템플릿으로 변환하여 공유 임베딩 공간으로 투영합니다. 주요 특징은 다양한 프로토콜 및 스캐너 간의 일관된 대조 학습을 위한 Supervised Contrastive (SupCon) 손실을 활용하여 데이터의 의미적 분류를 개선하는 점입니다. 훈련은 40,005명의 피실험자와 169,634개의 볼륨을 포함하는 대규모 데이터셋에서 수행되었습니다.

- **Performance Highlights**: MR-CLIP은 2D, 2.5D 및 3D 모델 테스트에서 높은 정확도를 보여주며, 특히 2.5D 모델은 88.7%의 최고의 전체 정확도를 기록했습니다. n-shot 분류에서 MR-CLIP의 성능은 전통적인 3D ResNet을 지속적으로 초과하였으며, 이는 메타데이터 기반의 비지도 사전 학습이 제한된 레이블 데이터 가운데서도 효과적임을 입증했습니다. 마지막으로, MR-CLIP은 데이터 품질 관리를 위한 높은 감도를 보여주며, 누락된 태그 감지에서 거의 완벽한 AUC 점수를 기록했습니다.



### Benchmarking individual tree segmentation using multispectral airborne laser scanning data: the FGI-EMIT datas (https://arxiv.org/abs/2511.00653)
Comments:
          39 pages, 9 figures

- **What's New**: 이 연구는 FGI-EMIT라는 최초의 대규모 다중 스펙트럼(Multispectral) 공중 레이저 스캐닝 기준 데이터셋을 소개합니다. 이 데이터셋은 532, 905, 1,550 nm의 파장에서 촬영된 1,561개의 수동 주석 트리로 구성되어 있으며, 특히 작은 아까시나무에 초점을 맞추고 있습니다. 이러한 데이터가 제공됨에 따라 기존의 비지도학습(unsupervised) 및 지도학습(supervised) 딥러닝(DL) 방법에 대한 포괄적인 벤치마킹이 이루어졌습니다.

- **Technical Details**: 전통적으로 비지도 알고리즘을 사용하여 정점(point cloud) 기하학 및 복잡한 휴리스틱 규칙에 의존해 개별 트리를 세분화했습니다. 하지만 최근 연구는 DL 방식으로 진전이 이루어지고 있으며, 본 연구에서는 기존의 비지도 또는 지도 방식의 알고리즘을 조합하여 성능을 비교하였습니다. 또한, 비지도 방법의 하이퍼파라미터를 베이지안 접근 방식으로 최적화하고, DL 모델은 처음부터 학습하여 정확성을 높였습니다.

- **Performance Highlights**: 비지도 알고리즘 중 Treeiso가 테스트 세트 F1 점수에서 52.7%를 달성한 반면, DL 접근 방식은 전체적으로 뛰어난 성능을 보였습니다. 특히 ForestFormer3D 모델이 F1 점수 73.3%를 기록하며, 작은 아까시나무에 대해서는 Treeiso보다 25.9% 높은 성과를 거두었습니다. 포인트 밀도 분석 결과, DL 방법이 비지도 알고리즘보다 항상 우수한 성능을 발휘함을 확인했습니다.



### Grounding Surgical Action Triplets with Instrument Instance Segmentation: A Dataset and Target-Aware Fusion Approach (https://arxiv.org/abs/2511.00643)
- **What's New**: 이 논문에서는 수술 도구와 조직 간의 상호작용을 이해하기 위해 수술 조치 트리플을 공간적으로 정립하는 새로운 접근 방식을 제안합니다. 특히, 자형 분할을 통해 의 대칭성을 활용하여 조치 트리플을 직접적으로 픽셀 수준으로 링크하는 방식을 소개합니다. 이러한 접근은 특히 임상 상황에서 중요한 액션의 서울 을보다 세부적으로 제공할 수 있습니다.

- **Technical Details**: 본 연구의 핵심은 수술 액션 트리플⟨, verb, target⟩를  셈세하는 것으로 정의됩니다. 연구진은 CholecTriplet-Seg라는 새로운 대규모 데이터셋을 발표하여 30,000개 이상의 주석이 달린 프레임과 함께 도구 인스턴스 마스크와 액션 및 해부학적 대상 주석을 연계하였습니다. 또한 TargetFusionNet이라는 새로운 아키텍처를 도입하여 약한 해부학적 우선 논리를 통합함으로써 보다 정확한 해부학적 목표 예측이 가능하도록 하였습니다.

- **Performance Highlights**: TargetFusionNet은 주어진 기존 기준에 비해 성능이 지속적으로 향상되는 것을 보여주었습니다. 연구에서는 Triplet Segmentation mAP라는 새로운 평가 지표를 도입하여 트리플 및 구성 요소 레벨에서의 정확도를 평가합니다. 이는 수술 획득 능력을 한층 향상시키고, 향후 연구의 기초 자료로서 기능할 것으로 기대됩니다.



### CueBench: Advancing Unified Understanding of Context-Aware Video Anomalies in Real-World (https://arxiv.org/abs/2511.00613)
- **What's New**: 본 논문에서는 CueBench라는 새로운 벤치마크를 소개합니다. 이는 통합 평가 프레임워크 내에서 컨텍스트 기반 비디오 이상 탐지(Anomaly Understanding) 연구를 위해 설계되었습니다. 기존의 연구들은 비현실적인 사건 탐지에 집중했으나, CueBench는 더 깊은 맥락 이해를 가능합니다.

- **Technical Details**: CueBench는 14개의 조건적(event-based) 및 18개의 절대적 절차를 포함하는 사건 중심의 계층적 분류 체계를 수립합니다. 174개의 장면과 198개의 속성에 걸쳐 정제된 의미론(semantics)을 정의하여 이상 사건을 구분하는 복잡한 원리와 미세한 맥락을 제시합니다. 이를 통해 인식(recognition), 시간 바인딩(temporal grounding), 탐지(detection), 예측(anticipation) 등의 다양한 과제를 포함한 균일한 캡슐화된 비교를 제공합니다.

- **Performance Highlights**: CueBench에서의 결과는 기존 비전-언어 모델(VLMs)이 실제 세계의 이상 이해에서 여전히 부족함을 보입니다. 반면 우리의 Cue-R1 모델은 이러한 최첨단 접근 방식보다 평균 24% 이상 성능이 우수함을 나타냅니다. 이는 향후 연구 및 개발에 중요한 기초 자료로 작용할 것입니다.



### TRACES: Temporal Recall with Contextual Embeddings for Real-Time Video Anomaly Detection (https://arxiv.org/abs/2511.00580)
Comments:
          10 pages, 5 figures

- **What's New**: 본 연구는 문맥을 고려한 제로샷 비디오 이상 탐지(Zero-Shot Anomaly Detection) 문제에 접근합니다. 이 시스템은 새로운 사건을 탐지하기 위해 시간적 및 외관적 특징을 텍스트 메모리와 실시간으로 상관관계 지어 학습합니다. 기존의 접근법이 새로운 실제 환경에 대한 일반화 능력이 부족한 반면, 우리는 메모리 지원 파이프라인을 정의하여 비주얼 임베딩과 시간적 신호를 상관 관계 짓는 방법을 제시합니다.

- **Technical Details**: TRACE(Temporal Recall with Contextual Embeddings) 시스템은 이상 및 비이상 맥락을 메모리 뱅크에 저장하고, 동작-외관 융합 모듈을 통해 시간적 크로스 어텐션을 활용합니다. 이를 통해 동적인 행동 패턴과 시각적 의미를 결합하고, 텍스트 맥락 벡터와의 유사성을 통해 이상 가능성을 예측하는 제로샷 이상 점수 매기기 메커니즘을 구현하였습니다. 이 방법은 다양한 환경에서 맥락에 맞는 이상 '추적'을 재현하게 설계되었습니다.

- **Performance Highlights**: 실험 결과, 우리 방법이 UCF-Crime 데이터셋에서 90.4%의 AUC를, XD-Violence 데이터셋에서 83.67%의 AP를 달성하며 제로샷 모델 중 최신 기술에 도달했음을 보여줍니다. 또한, 높은 정확성과 설명 가능성을 가진 실시간 추론을 제공하여 실제 환경에서의 적용성을 높입니다. 우리는 크로스 어텐션 시간 융합과 문맥적 메모리를 융합함으로써 높은 충실도의 이상 탐지를 가능하게 하였습니다.



### Generalized Category Discovery under Domain Shift: A Frequency Domain Perspectiv (https://arxiv.org/abs/2511.00573)
Comments:
          29 pages, 5 figures

- **What's New**: 이 논문은 도메인 이동이 있는 일반화된 카테고리 발견(Domain-Shifted Generalized Category Discovery, DS_GCD)의 문제를 다루고 있습니다. 기존 방법은 라벨이 있는 샘플과 라벨이 없는 데이터가 동일한 도메인에서 온다고 가정하지만, 실제 세계에서는 이러한 가정이 성립하지 않는 경우가 많습니다. 따라서 우리는 새로운 범주와 도메인을 동시에 식별하는 모델을 개발하여, 레이블이 있는 데이터만을 기반으로 파라미터를 조정하고 일반화할 수 있는 방법을 제안합니다.

- **Technical Details**: 우리가 제안한 FREQUENCY 방법은 주파수 도메인 정보를 활용하여 도메인 이동에 대응할 수 있도록 설계되었습니다. 주파수 기반 도메인 분리 전략은 샘플의 진폭 차이를 측정하여 알려진 도메인과 알려지지 않은 도메인을 분리합니다. 또한, 주파수 도메인의 교차 도메인 변동과 인트라 도메인 변동 완화를 수행하여 더욱 견고한 모델을 학습할 수 있도록 합니다. 이 방식으로 우리는 초기의 문제점들을 고립화하고 통제된 방식으로 해결하려 했습니다.

- **Performance Highlights**: 귀하의 방법이 다양한 벤치마크 데이터셋에서 도메인 이동의 영향을 완전히 완화하며 기존의 최첨단 방법들보다 우수한 성능을 보여주었습니다. 특히 카테고리 발견 측면에서 새로운 카테고리를 찾는 데 있어 월등한 결과를 보여주었습니다. 실험 결과는 본 연구가 라벨이 부착된 데이터만 사용할 때의 기능 제한을 극복할 수 있는 강력한 가능성을 가지며, 실제 애플리케이션에서 매우 유용할 것임을 입증합니다.



### 4D Neural Voxel Splatting: Dynamic Scene Rendering with Voxelized Guassian Splatting (https://arxiv.org/abs/2511.00560)
Comments:
          10 pages, 7 figures

- **What's New**: 본 연구에서는 동적 장면 모델링을 위한 4D Neural Voxel Splatting (4D-NVS) 방법을 제안합니다. 기존의 3D Gaussian Splatting (3D-GS) 방식은 프레임마다 Gaussian을 복제하여 메모리 소모가 컸던 반면, 4D-NVS는 변형 필드(Deformation Fields)를 학습하여 정적 대신 동적 장면을 효과적으로 처리합니다. 이 기술은 메모리 소비를 줄이고 훈련 속도를 가속화하면서도 고품질 이미지를 유지합니다.

- **Technical Details**: 4D-NVS는 4차원 공간에서 효율적인 동적 장면 표현을 위해 벡셀 기반 접근 방식과 신경망 Gaussian splatting을 결합합니다. 호환성을 위해 시간적 동역학(Temporal Dynamics)과 공간적 구조(Spatial Structure)를 분리하여, 지속적인 벡셀 앵커(Voxel Anchors)에서 신경 Gaussian을 즉시 생성하고 통합된 변형을 적용합니다. 이로 인해 전통적인 방법보다 메모리 복잡성이 𝒪(fV+F)로 강화되었습니다.

- **Performance Highlights**: 본 방법은 실험에서 전통적인 방법에 비해 현저한 메모리 감소와 빠른 훈련 속도를 나타내며, 뛰어난 시각적 품질을 자랑합니다. 4D-NVS는 자원 제약 조건이 있는 실시간 동적 장면 렌더링을 가능하게 하여, 모바일 로봇 및 임베디드 AI 분야에서 요구되는 빠른 재구성과 상호작용을 지원합니다.



### MIFO: Learning and Synthesizing Multi-Instance from One Imag (https://arxiv.org/abs/2511.00542)
Comments:
          17 pages, 30 figures

- **What's New**: 이 논문에서는 단일 이미지에서 다중 인스턴스 의미를 정밀하게 학습하고 합성하는 방법을 제안합니다. 주된 문제는 제한된 훈련 데이터로 인해 비슷한 의미나 외관을 가진 인스턴스들의 학습이 더욱 어렵다는 점입니다. 이를 해결하기 위해, 비슷한 의미를 분리하는 패널티 기반의 어텐션 최적화를 제안합니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성됩니다: 의미 분리 학습 단계와 정밀 합성 단계입니다. 의미 분리 학습 단계에서는 보상 기반 어텐션 제어와 패널티 기반 최적화를 결합하여 비슷한 인스턴스의 의미를 점진적으로 분리합니다. 정밀 합성 단계에서는 셀프 어텐션(SA)과 크로스 어텐션(CA) 레이어에서 박스 제어를 도입하여 의미의 융합이나 누수를 완화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 분리된 고품질의 다중 인스턴스 의미 학습 및 합성을 달성하여 편집 가능성과 인스턴스 일관성의 뛰어난 균형을 이루는 것으로 나타났습니다. 특히, 이 방법은 의미적으로나 시각적으로 유사한 인스턴스 또는 드물게 발견되는 객체를 처리할 때도 강력한 성능을 보여주며, 코드가 공개되어 무료로 제공됩니다.



### Real-IAD Variety: Pushing Industrial Anomaly Detection Dataset to a Modern Era (https://arxiv.org/abs/2511.00540)
Comments:
          13 pages, 4 figures and 5 tables

- **What's New**: 새로운 Real-IAD Variety는 160개의 다양한 객체 카테고리를 포함한 대규모의 산업 이상 탐지(Industrial Anomaly Detection, IAD) 데이터셋으로, 총 198,960개의 고해상도 이미지로 구성되어 있습니다. 이 데이터셋은 28개 산업, 24가지 재료 유형 및 22가지 색상 변형을 포괄하여 기존의 공공 벤치마크에서 발견된 범주 다양성의 한계를 극복하고 있습니다. 이러한 향상은 다음 세대의 이상 탐지 기초 모델을 훈련하고 평가하는 데 중요한 자원이 되며, 연구자들이 분야-specific 제약을 넘어 설계할 수 있게 합니다.

- **Technical Details**: Real-IAD Variety의 도입은 제3의 큰 변화로, 다중 클래스 비지도 이상 탐지(multi-class unsupervised anomaly detection) 설정에서 기존 알고리즘의 한계를 드러내고, 상태 값(source) 비지도 이상 탐지 모델이 다중 카테고리로 확장할 때 성능 저하를 겪는 것을 보여줍니다. 또한, 제로-/퍹-샷(zero-/few-shot) 탐지 접근법들은 카테고리 수 변동에 대해 더 강건한 경향을 보여 160 카테고리 이상에서의 성능이 크게 향상됨을 발견하였습니다. 이는 산업 프로세스의 복잡성을 반영한 데이터셋 설계와 관련된 중요성을 강조합니다.

- **Performance Highlights**: 수행한 실험에서 발견된 중요한 성과는 두 가지입니다. 첫째, 카테고리 수의 증가가 다중 클래스 비지도 이상 탐지 방법의 성능에 미치는 부정적인 영향을 밝혀냈고, 카테고리가 30에서 60, 100, 160으로 증가할 때 10%에서 30%의 성능 저하가 발생하는 것을 확인했습니다. 둘째, 제로-/퍹-샷 학습 모델들이 카테고리 수의 변화에 거의 영향을 받지 않는다는 사실로, 고급 IAD 방법의 일반화 가능성을 평가할 수 있는 체계적인 벤치마크를 제공하였습니다.



### Text-guided Fine-Grained Video Anomaly Detection (https://arxiv.org/abs/2511.00524)
- **What's New**: 이번 논문에서는 Text-guided Fine-Grained Video Anomaly Detection (T-VAD)라는 새로운 프레임워크를 제안합니다. T-VAD는 Large Vision-Language Model (LVLM)을 기반으로 하여 비디오의 이상 이벤트를 정밀하게 자동으로 탐지하고 위치를 지정하는 기능을 제공합니다. 기존의 수동 임계값 설정의 의존성을 없애고, 더욱 직관적이고 해석 가능한 이상 열지도를 생성할 수 있도록 합니다.

- **Technical Details**: T-VAD는 Anomaly Heatmap Decoder (AHD)와 Region-aware Anomaly Encoder (RAE)를 결합하여 비디오 프레임에서 픽셀 단위로 이상 사건을 특성화합니다. AHD는 시각적 및 텍스트적 기능의 정렬을 통해 세밀한 이상 열지도를 생성하고, RAE는 이 열지도를 학습 가능한 텍스트 임베딩으로 변환하여 LVLM이 이를 활용하도록 합니다. 이러한 과정을 통해 T-VAD는 이상 탐지의 정확성과 상호작용성을 크게 향상시킵니다.

- **Performance Highlights**: T-VAD는 UBnormal 데이터셋에서 94.8%의 Area Under the Curve (AUC)와 이상 열지도에서 67.8%/76.7%의 정확도를 달성하였습니다. ShanghaiTech 기반 데이터셋에서도 높은 텍스트 설명의 선호도를 보였으며, BLEU-4 점수가 목표에 대해 62.67, 경로에 대해 88.84로 나타났습니다. 이러한 성과는 T-VAD가 기존의 방법들보다 우수한 성능을 발휘하고 있음을 입증합니다.



### SegDebias: Test-Time Bias Mitigation for ViT-Based CLIP via Segmentation (https://arxiv.org/abs/2511.00523)
- **What's New**: 이번 연구에서는 ViT 기반의 CLIP 모델을 위한 새로운 테스트 시간의 디바이싱(debiasing) 방법인 SegDebias를 제안합니다. 기존의 디바이싱 방법들과는 달리, 추가적인 트레이닝이나 바이어스 주석에 대한 가정 없이도 작동합니다. 이 방법은 사전 훈련된 세그멘테이션(segmentation) 모델을 활용하여 대상 시각 속성을 격리하고, 비대상 영역의 임베딩을 모든 클래스별 텍스트 프롬프트와 균일하게 유사하도록 조정합니다.

- **Technical Details**: SegDebias는 특정 데이터셋의 바이어스에 대한 사전 지식 없이도 적용할 수 있는 간단하면서 효과적인 테스트 시간의 접근 방식을 제공합니다. 이 방법은 시맨틱 세그멘테이션을 사용하여 특정 속성을 분리한 후, 비대상 영역에 대한 임베딩을 조정하여 모델의 예측에서 원치 않는 바이어스 신호를 제거합니다. 이러한 절차를 통해 중요한 시각 정보는 보존하면서도 모델의 예측에 미치는 영향을 제거할 수 있습니다.

- **Performance Highlights**: Waterbirds와 CelebA 데이터셋에서 실시된 실험 결과, SegDebias는 기존의 테스트 시간 디바이싱 접근 방법들보다 우수한 성능을 보였습니다. 특히, 그룹 로버스트니스(group robustness) 지표에서 최고의 최악 그룹 정확도를 달성했고, 그룹 간 성능 격차를 크게 줄였습니다. Attention-IoU 지표를 통해 SegDebias는 CLIP의 주의를 의미 있는 시각 영역으로 성공적으로 유도했다는 것을 입증했습니다.



### ID-Composer: Multi-Subject Video Synthesis with Hierarchical Identity Preservation (https://arxiv.org/abs/2511.00511)
- **What's New**: ID-Composer는 텍스트 프롬프트 및 참조 이미지를 기반으로 멀티-서브젝트 비디오 생성의 새로운 프레임워크입니다. 기존의 비디오 생성 모델들이 텍스트나 단일 이미지에 의존하는 한계를 극복하면서, 주제 ID의 일관성을 유지하고 여러 주제 간의 의미 통합을 용이하게 합니다. 이를 통해 개인화된 콘텐츠 생성, 가상 이야기 전개 및 광고에서도 새로운 가능성을 열어줍니다.

- **Technical Details**: ID-Composer의 핵심 디자인은 두 가지로 구성됩니다: (1) 계층적 정체성 유지 주의 메커니즘으로, 이는 서로 다른 주제와 형태 간의 특징을 효과적으로 집계하여 올바른 주체의 일관성과 텍스트의 충실성을 유지합니다. (2) 선훈련된 비전-언어 모델(VLM)을 통한 의미 이해로, VLM은 텍스트 인코더로 활용되어 여러 주제 간의 복잡한 상호작용을 포착합니다. 이러한 기능을 최적화하여 고품질 비디오 생성을 달성합니다.

- **Performance Highlights**: ID-Composer의 실험 결과는 ID의 보존, 시간적 일관성 및 비디오 품질에서 기존의 방법들을 초월하는 것을 보여줍니다. 특히 계층적 정체성 유지 주의 메커니즘과 VLM 기반 의미 이해를 통한 성능 향상이 두드러집니다. 추가적으로 신규 데이터셋을 구성하여 멀티-서브젝트 비디오 생성 모델의 학습 및 평가를 지원하며, 다양한 실험을 통해 모델의 우수성을 입증합니다.



### OmniTrack++: Omnidirectional Multi-Object Tracking by Learning Large-FoV Trajectory Feedback (https://arxiv.org/abs/2511.00510)
Comments:
          Extended version of CVPR 2025 paper arXiv:2503.04565. Datasets and code will be made publicly available at this https URL

- **What's New**: 본 논문은 다중 객체 추적(Multi-Object Tracking, MOT) 기술을 360° 파노라마 이미지에서 적용하는 새로운 접근 방식을 제안합니다. OmniTrack++ 프레임워크는 이러한 파노라마 이미지의 왜곡과 정체성 모호성을 해결하기 위해 피드백 기반 접근법을 채택해 인식(perception)을 점진적으로 개선합니다. 특히, 다양한 동적 요소를 일관되게 해석하는 능력이 MOT의 핵심으로 보입니다.

- **Technical Details**: OmniTrack++는 DynamicSSM 블록을 통해 파노라마 특징을 안정화하고, FlexiTrack 인스턴스는 타겟의 피드백을 통해 유연한 로컬라이제이션(localization)과 신뢰할 수 있는 단기 연관(association)을 제공합니다. 또한 ExpertTrack Memory 모듈은 Mixture-of-Experts 디자인을 사용하여 외형적 단서를 통합함으로써 단편화된 경로를 복구하고 정체성 드리프트(identity drift)를 줄입니다. Tracklet 관리 모듈은 장면의 동역학(scene dynamics)에 따라 end-to-end와 tracking-by-detection 모드 간의 전환을 조절합니다.

- **Performance Highlights**: OmniTrack++는 JRDB와 EmboTrack 데이터셋에서 최첨단 성능을 달성하여 각각 +25.5%와 +43.07% HOTA 개선을 기록했습니다. 특히 QuadTrack에서의 성능 검증은 파노라마 MOT의 도전 과제를 해결하는데 효과적임을 보입니다. EmboTrack 벤치마크는 다이내믹한 모바일 인식(dynamic mobile perception)의 도전을 포괄적으로 평가하기 위해 설계되었습니다.



### VinDr-CXR-VQA: A Visual Question Answering Dataset for Explainable Chest X-Ray Analysis with Multi-Task Learning (https://arxiv.org/abs/2511.00504)
Comments:
          ISBI submission. Contains 5 pages, 2 figures, and 6 tables. Code & data: this https URL

- **What's New**: 이번 연구는 VinDr-CXR-VQA라는 대규모 흉부 X-ray 데이터셋을 소개합니다. 이 데이터셋은 4,394개의 이미지와 17,597개의 질문-답변 쌍으로 구성되어 있으며, 각 데이터는 방사선 전문의의 검증을 받은 바운딩 박스와 임상적 설명이 포함되어 있습니다. 질문의 유형은 진단적 질문의 여섯 가지 유형을 포함하고 있으며, 긍정적(41.7%)과 부정적(58.3%) 샘플 간의 균형을 맞춰 신뢰성을 높였습니다.

- **Technical Details**: VinDr-CXR-VQA 데이터셋은 공간적 기초를 가진 의학적 질문-답변 모델(Med-VQA)을 위해 설계되었습니다. 질문 유형은 위치, 병리 확인, 존재 검증, 수량, 해부학적 분류 및 이진 확인으로 다양합니다. Google의 Gemini 2.5 Pro API를 사용하여 자연어 질문 및 임상적 추론을 생성하며, 모든 정보는 검증된 데이터에서 직접 이어받아 사용하고 있습니다.

- **Performance Highlights**: 모델의 성능은 MedGemma-4B-it로 벤치마킹되었으며, F1 점수가 0.624로 향상되어 기준선 대비 11.8% 증가했습니다. 우리는 다양한 병리학적 상황을 포함하여 다중 병변 사례에서 모델의 강인성을 평가하였고, 임상 안전성을 보장하기 위해 고급 검증 절차를 수행했습니다. 이 데이터셋은 임상적으로 안전한 Med-VQA 모델의 개발과 평가를 지원하기 위해 공개되었습니다.



### Diff4Splat: Controllable 4D Scene Generation with Latent Dynamic Reconstruction Models (https://arxiv.org/abs/2511.00503)
- **What's New**: Diff4Splat라는 새로운 피드포워드 방법이 소개되었습니다. 이 방법은 단일 이미지로부터 제어 가능한 4D 장면을 합성합니다. Diff4Splat는 비디오 확산 모델의 생성적 사전(Generative Priors)과 대규모 4D 데이터셋에서 학습한 기하학과 움직임 제약 조건을 통합하여 단일 이미지와 카메라 궤적(trajectory)으로부터 변형 가능한 3D Gaussian 필드를 직접 예측합니다.

- **Technical Details**: Diff4Splat의 핵심은 비디오 잠재 변환기(Video Latent Transformer)로, 이는 비디오 확산 모델을 증강하여 시공간 의존성(Spatio-temporal Dependencies)을 동시에 포착하고 시간에 따라 변화하는 3D Gaussian 기본 요소(Primitives)를 예측합니다. 훈련 과정은 외관 충실도, 기하학적 정확성, 움직임 일관성에 대한 목표에 의해 안내되고, 이를 통해 30초 내에 고품질의 4D 장면을 합성할 수 있습니다.

- **Performance Highlights**: Diff4Splat는 비디오 생성, 새로운 뷰 합성, 기하학적 추출 분야에서 높은 효과성을 입증했습니다. 이 방법은 동적 장면 합성에서 최적화 기반 방법과의 품질 면에서 동등하거나 그것을 초과하며, 동시에 훨씬 더 효율적입니다. 실험 결과, Diff4Splat는 한 장의 이미지에서 고충실도의 4D 장면을 생성하고, 기존의 두 단계 파이프라인과 기존 카메라 제어 비디오 생성 방법들보다 우수한 성능을 보였습니다.



### FedMGP: Personalized Federated Learning with Multi-Group Text-Visual Prompts (https://arxiv.org/abs/2511.00480)
- **What's New**: 이 논문에서는 FedMGP라는 개인화된 연합 프롬프트 학습의 새로운 패러다임을 소개합니다. 각 클라이언트는 서로 다른 세분화된 의미를 포착할 수 있는 여러 쌍의 텍스트 및 비주얼 프롬프트 그룹을 보유하며, 이를 통해 모델이 다양한 국소적 특성을 효과적으로 학습할 수 있게 합니다. 다이내믹 프롬프트 집계 전략을 사용하여, 클라이언트는 이전 라운드에서의 글로벌 프롬프트와의 유사성을 기반으로 하여 샘플링을 수행함으로써, 개인화와 일반화 간의 균형을 맞추게 됩니다.

- **Technical Details**: FedMGP는 각 클라이언트에 다수의 텍스트-비주얼 프롬프트 그룹을 유지하여 데이터에서 나타나는 다양한 특성과 세부 요소를 반영합니다. 프롬프트 그룹의 다양성을 유도하기 위해 diversity loss를 도입하여 각 클라이언트 내의 표현 분리를 강화하며, 이를 통해 특정 클라이언트의 노이즈를 억제하고 공통적 패턴을 강조합니다. 제안된 메커니즘은 기존의 연합 프롬프트 학습 방법보다 낮은 통신 매개변수를 사용하여 뛰어난 성능을 달성합니다.

- **Performance Highlights**: FedMGP는 다양한 시험 환경에서 이전 접근 방식에 비해 우수한 성능을 보여주며, 개인화 정확도와 새로운 도메인에 대한 일반화 능력을 성공적으로 균형 잡았습니다. 특히, 비정형(Non-IID) 데이터와 같은 여러 이질적 데이터 설정에서 성능을 평가하여, FedMGP가 개인화된 클라이언트 데이터와 일반화 기능을 모두 다룰 수 있음을 보여주었습니다. 이 논문에서는 다양한 연합 비전-언어 벤치마크에서의 실험 결과를 통해 FedMGP의 효과를 증명합니다.



### Longitudinal Vestibular Schwannoma Dataset with Consensus-based Human-in-the-loop Annotations (https://arxiv.org/abs/2511.00472)
- **What's New**: 본 논문은 자기공명영상(MRI)에서의 전정 신경종(Vestibular Schwannoma, VS) 자동 분할을 위한 신뢰성 있는 주석 데이터셋을 제시합니다. 기존의 수동 주석 과정은 시간이 많이 소요되며, 비용 또한 증가하는데 반해, 제안된 방법은 전문가의 검증과 딥러닝 기반 프레임워크를 결합하여 효율적이고 신뢰성 높은 주석을 가능하게 합니다. 이 방법은 다양한 데이터셋에서 안정적인 성능을 유지하며, 기존 수동 주석 프로세스에 비해 약 37.4%의 효율성 향상을 기대할 수 있습니다.

- **Technical Details**: 이 연구에서는 DL 기반 자동 분할과 전문가 검증 과정을 통합한 휴먼-인-더-루프 데이터 주석 방법론을 제안합니다. 주석 방법론은 모델 훈련, 추론 및 전문가 검증 등의 여러 단계로 이루어져 있으며, MRI 데이터셋은 다수의 의료기관에서 수집된 경우를 포함합니다. 연구에 사용된 데이터는 여러 병원에서 수집된 T1CE 및 T2 가중치 스캔으로 구성되어 있으며, 이들 데이터셋은 The Cancer Imaging Archive (TCIA)에서 공개될 예정입니다.

- **Performance Highlights**: 제안하는 접근 방식은 내부 검증 데이터셋에서 Dice Similarity Coefficient(DSC)가 0.9125에서 0.9670으로 크게 향상되는 성과를 거두었습니다. 전문가 평가를 통해 143개의 스캔에서 모델 개선이 필요한 미묘한 사례들이 밝혀졌으며, 이러한 피드백은 향후 모델 개선에 중요한 역할을 할 것입니다. 최종적으로 본 연구는 다양한 임상 환경에서 VS 자동 분할을 위한 높은 정확도를 달성하며 임상 적용 가능성을 강조합니다.



### HumanCrafter: Synergizing Generalizable Human Reconstruction and Semantic 3D Segmentation (https://arxiv.org/abs/2511.00468)
Comments:
          Accepted to NeurIPS 2025; Project page: [this URL](this https URL)

- **What's New**: HumanCrafter는 3D 인간 재구성과 관련된 점에서 혁신적인 통합 프레임워크입니다. 이 모델은 단일 이미지에서 인간의 모양과 부분 의미를 동시에 모델링할 수 있도록 하여, 3D 재구성과 인간 부위 분할 간의 협력적 작업을 가능하게 합니다. 또한, 고품질 데이터-레이블 쌍을 생성하기 위해 인터랙티브한 주석 절차를 발전시켰습니다.

- **Technical Details**: HumanCrafter는 기하학적 인간 선행 지식을 통합하여 3D 재구성 단계에서 활용하며, 자가 지도(self-supervised) 의미 선행 지식을 세그멘테이션 단계에서 통합합니다. 이 프레임워크는 40,000개의 이미지로 구성된 3D 세그멘테이션 데이터셋에서 공동 훈련을 통해 서로의 작업을 개선합니다. Pixel-aligned aggregation을 통해 여러 작업 간 시너지를 만들어내며, 다중 작업 목표를 사용해 텍스처 모델링의 정확성과 의미의 일관성을 동시에 최적화합니다.

- **Performance Highlights**: HumanCrafter는 실험에서 기존의 최고 수준의 방법들을 초월하여, 단일 이미지에서의 3D 인간 부위 세그멘테이션 및 3D 인간 재구성 능력에서 우수성을 입증했습니다. 이 모델은 리얼타임 렌더링을 통해 고화질의 3D 인간 재구성 및 세그멘테이션을 가능하게 하여, 실제 세계의 다양한 이미지를 효과적으로 처리할 수 있는 좋은 범용성을 보입니다.



### Weakly Supervised Pneumonia Localization from Chest X-Rays Using Deep Neural Network and Grad-CAM Explanations (https://arxiv.org/abs/2511.00456)
- **What's New**: 이번 연구는 흉부 X-레이에서 폐렴을 분류하고 위치를 지정하기 위한 약하게 감독된 딥러닝 프레임워크를 제안합니다. 고가의 픽셀 수준 주석 없이 이미지 수준 레이블을 활용하여 폐렴에 영향을 받은 지역을 강조하는 임상적으로 의미 있는 히트맵을 생성합니다. 실험 결과, ResNet-18와 EfficientNet-B0가 98%의 테스트 정확도를 기록하며, Grad-CAM 시각화를 통해 모델이 임상적으로 중요한 폐 영역에 집중하고 있는 것이 확인되었습니다.

- **Technical Details**: 연구에서는 ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0, MobileNet-V2, MobileNet-V3 및 ViT-B16 등 총 7개의 이미지넷(pretrained) 아키텍처를 동일한 훈련 조건에서 평가했습니다. Focal loss와 환자 기반 분할을 사용하여 데이터 누수를 방지했으며, 전체적으로 ResNet-18과 EfficientNet-B0가 최상의 성능을 보였습니다. Grad-CAM(Gradient-weighted Class Activation Mapping)을 이용해 생성한 히트맵은 방사선 과학에 대한 해석 가능한 인사이트를 제공합니다.

- **Performance Highlights**: Kermany CXR 데이터셋 실험 결과, ResNet-18과 EfficientNet-B0는 각각 ROC-AUC 0.997, F1 0.987의 최상의 테스트 정확도를 기록했습니다. 또한 MobileNet-V2는 정확도와 컴퓨팅 비용 사이의 최적의 균형을 제공했습니다. 이러한 성과는 해석 가능한 AI를 활용한 폐렴 조기 발견의 투명성을 높이고 임상적 신뢰를 지원하는 데 기여합니다.



### ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training (https://arxiv.org/abs/2511.00446)
Comments:
          Accepted by NeurIPS 2025

- **What's New**: 본 연구에서는 ToxicTextCLIP라는 프레임워크를 소개하며, CLIP 모델의 사전 학습 단계에서 텍스트 기반의 공격을 가능하게 하는 고품질의 적대적 텍스트 생성 방법을 제안합니다. 이 프레임워크는CLIP의 훈련 과정에서 텍스트 모달리티를 공격하는 것이 중요하다는 점을 강조하며, 기존 연구가 주로 이미지 기반 공격에 집중한 점을 반영하여 새로운 접근 방식을 제공합니다. 또한, 이 연구는 텍스트 모달리티가 CLIP의 대비 학습에서 핵심적임에도 불구하고 여전히 탐구되지 않은 공격 경로임을 지적합니다.

- **Technical Details**: ToxicTextCLIP는 두 가지 주요 도전 과제를 해결합니다. 첫째, 목표 클래스와의 배경 불일치로 인해 발생하는 의미적 불일치 문제를 해결하기 위해, 배경에 민감한 선택자를 사용하여 목표 클래스와 일치하는 배경 정보를 가진 텍스트를 우선 선정합니다. 둘째, 텍스트의 부족 문제를 해결하기 위해 배경 주도 증강기를 사용하여 의미적으로 일관되고 다양한 형태의 오염된 샘플을 생성합니다. 이러한 방식으로 ToxicTextCLIP는 CLIP의 크로스 모달 표현을 효과적으로 타격하는 고품질의 적대적 텍스트를 생성합니다.

- **Performance Highlights**: ToxicTextCLIP는 분류 및 검색 작업에서 탁월한 성과를 나타냅니다. 실험 결과, 최대 95.83%의 공격 성공률과 98.68%의 백도어 Hit@1을 기록했습니다. 또한, RoCLIP, CleanCLIP 및 SafeCLIP와 같은 기존 방어 메커니즘을 우회하여 공격을 성공적으로 수행하였습니다. 이는 ToxicTextCLIP의 강력한 효과를 시사합니다.



### Enhancing Frequency Forgery Clues for Diffusion-Generated Image Detection (https://arxiv.org/abs/2511.00429)
- **What's New**: 이 논문은 이미지 합성에서 큰 성공을 거둔 Diffusion 모델이 생성한 이미지의 악의적인 사용 가능성에 대한 우려를 다룹니다. 기존의 탐지기는 다양한 모델과 설정에서 구분 가능한 단서를 잡는 데 어려움을 겪고 있으며, 이는 보이지 않는 Diffusion 모델에 대한 일반화 능력과 다양한 교란에 대한 강인성을 제한합니다. 연구자들은 자연 이미지와 Diffusion으로 생성된 이미지의 주파수 대역에서 구성된 차이를 분석하여 F2C(Frequency Forgery Clue)를 활용한 효과적인 탐지기를 제안합니다.

- **Technical Details**: 이 논문은 자연 이미지와 Diffusion 생성 이미지 사이의 주파수 도메인에서의 차이를 분석합니다. 그 결과, Diffusion으로 생성된 이미지는 저주파에서 고주파로 갈수록 자연 이미지와 더 큰 차이를 보이는 것을 파악했습니다. 이 발견에 기반하여, 논문에서는 무게가 가중된 필터로 작용하는 주파수 선택 함수(frequency-selective function)를 도입하여 주파수 스펙트럼의 덜 구별되는 대역을 억제하고 더 유용한 대역을 강화하는 방법을 제안합니다.

- **Performance Highlights**: 다양한 Diffusion 생성 이미지 데이터셋에 대한 광범위한 실험 결과, 제안된 방법이 기존의 최첨단 탐지기보다 우수한 일반화 및 강인성을 제공함을 확인했습니다. F2C 기법은 기존 모델들이 특정 패턴에 의존하는 것과는 달리 자연 이미지와의 고유한 차이를 활용하여 보이지 않는 Model의 이미지를 효과적으로 탐지할 수 있습니다. 이 연구는 이미지 생성 기술의 악용을 막기 위한 새로운 접근 방식을 제시합니다.



### Leveraging Hierarchical Image-Text Misalignment for Universal Fake Image Detection (https://arxiv.org/abs/2511.00427)
- **What's New**: 본 논문에서는 생성 모델의 급격한 발전과 이에 따른 생성된 가짜 이미지 탐지의 필요성을 강조합니다. 기존의 이진 이미지 분류 방식은 시각적 단서에만 집중하여 과적합(overfitting)의 문제가 있으며, 새로운 방법론을 제안합니다. ITEM이라는 새로운 탐지기를 통해 이미지와 캡션 간 비정렬(misalignment)을 활용하여 가짜 이미지 탐지의 효율성을 높입니다.

- **Technical Details**: ITEM은 사전 훈련된 CLIP의 시각-언어 공간에서 이미지와 캡션 간의 비정렬을 측정하여 이미지 탐지 작업을 수행하는 MLP(다층 퍼셉트론) 헤드를 조정합니다. 추가적으로, 전체 이미지와 캡션에 설명된 각 의미적 객체에 대한 계층적 비정렬 방식을 도입하여 글로벌(global) 및 미세(local) 정밀 비정렬 정보를 탐색합니다. 이로 인해 더욱 강력하고 일반화된 탐지가 가능합니다.

- **Performance Highlights**: 다양한 생성 모델에 대한 광범위한 실험을 통해 본 방법이 기존의 최신 기술들보다 뛰어난 일반화 및 견고성을 보여줍니다. ITEM은 과거의 이미지 패턴에 과적합되지 않고 모든 이미지 유형에 대해 효과적으로 탐지할 수 있는 잠재력을 가지고 있습니다. 이를 통해, 생성된 이미지에 대한 범용 가짜 이미지 탐지 솔루션을 제시합니다.



### LGCA: Enhancing Semantic Representation via Progressive Expansion (https://arxiv.org/abs/2511.00419)
Comments:
          15 pages, 5 figures, to appear in SoICT 2025

- **What's New**: 최근 큰 규모의 사전 학습이 진행된 자연어 처리(NLP) 분야에서는 CLIP과 같은 사전 훈련된 비전-언어 모델이 이미지와 텍스트를 효과적으로 정렬할 수 있는 능력을 보여주었습니다. 이러한 모델들은 제로샷 이미지 분류(zero-shot image classification) 작업에서 성과를 크게 향상시켰습니다. 그러나 CLIP의 성능은 프롬프트(prompt) 표현에 민감하여, 세밀한 조정이 필요하다는 제한이 있습니다.

- **Technical Details**: 이 논문에서는 비디오 데이터의 세부 기능을 포착한 후 가장 두드러진 지역을 반복적으로 선택하고 확장하는 Localized-Globalized Cross-Alignment (LGCA) 프레임워크를 제안합니다. LGCA는 원래 이미지와 확장된 이미지를 포괄하는 유사성 점수를 통해 로컬(local) 및 글로벌(global) 기능을 모두 담아낼 수 있게 설계되었습니다. 이 방법은 잘못된 정보와 편향을 최소화하며, 기존의 비확장 모델과 유사한 시간 복잡성을 유지합니다.

- **Performance Highlights**: 다양한 데이터셋에서 수행된 실험 결과, LGCA는 기존의 최첨단 기준을 능가하며 제로샷 성능을 크게 향상시켰습니다. 앞서 언급한 LGCA의 장점은 비전-언어 모델이 특정 도메인에 맞춰 전이될 수 있도록 하면서도, 전반적인 효율성과 확장성을 유지한다는 점에서 중요한 이정표가 됩니다.



### CoT-Saliency: Unified Chain-of-Thought Reasoning for Heterogeneous Saliency Tasks (https://arxiv.org/abs/2511.00396)
Comments:
          14 pages,10 figures

- **What's New**: 본 논문은 Salient Object Detection (SOD), Co-salient Object Detection (CoSOD), Salient Instance Segmentation (SIS)와 같은 세 가지 이질적인 saliency 작업을 모두 처리하는 최초의 통합 프레임워크를 제시합니다. 각 작업은 Vision-Language Model (VLM) 내에서 Chain-of-Thought (CoT) 추론 프로세스로 표현됩니다. 이 프레임워크는 두 단계의 교육 방법인 Supervised Fine-Tuning (SFT)와 Reinforcement Learning (RL)을 따릅니다.

- **Technical Details**: 핵심적으로 '신뢰도 기반 정책 최적화'(Confidence-Guided Policy Optimization, CGPO)라는 경량화된 알고리즘을 제안합니다. 이는 보상과 모델 신뢰도 간의 차이를 활용하여 개별 샘플의 장점 신호를 생성하는 것을 목표로 합니다. 이러한 접근은 정보성이 높은 응답에 업데이트를 집중시키고, 불필요한 그룹 샘플링을 제거하여 기존 알고리즘의 한계를 극복합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 SODA, CoSOD 및 SIS의 모든 작업에 걸쳐 전문화된 최고 성능 (SOTA) 방법 및 강력한 닫힌 소스 VLM을 초과 달성했습니다. 특히 CoCA에서 CoSOD에 대해 0.899의 S-measure를 달성하여 이전 최상의 성과를 8.0 퍼센트 포인트 초과했으며, 적은 훈련 데이터를 사용함에도 불구하고 이러한 성과를 올렸습니다.



### VinciCoder: Unifying Multimodal Code Generation via Coarse-to-fine Visual Reinforcement Learning (https://arxiv.org/abs/2511.00391)
Comments:
          Preprint Version, Work in Progress

- **What's New**: VinciCoder는 기존의 단일 작업 훈련 방식의 한계를 극복하는 통합 멀티모달 코드 생성 모델입니다. 이 모델은 1.6M의 이미지-코드 쌍으로 구성된 대규모 Supervised Finetuning (SFT) 데이터셋을 사용하여 직접 코드 생성 및 코드 개정을 위한 두 단계의 훈련 프레임워크를 구축합니다. 특히, Visual Reinforcement Learning (ViRL) 전략을 도입하여 시각적 일치성을 강화하며 최상의 성능을 보여줍니다.

- **Technical Details**: VinciCoder는 하나의 통합 Vision-Language Model (VLM)로, 두 단계 SFT-ViRL 전략을 통해 학습됩니다. SFT 단계에서는 1.3M개의 직접 생성 샘플과 300k개의 코드 개정 예제를 포함하는 대규모 코퍼스를 구성하여 고유한 코드 개정 작업을 수행합니다. RL 단계에서는 기존의 규칙 기반 보상 대신 지각적 유사성을 기반으로 한 보상 신호를 활용하여 생성된 코드의 실행 가능성과 시각적 충실도를 최적화합니다.

- **Performance Highlights**: VinciCoder는 다양한 멀티모달 코드 생성 벤치마크에서 기존 방법들보다 우수한 성능을 발휘하며, 두 단계 접근 방식의 유효성을 입증합니다. SFT에서 이미 강력한 기준을 세우고, ViRL 전략을 적용함으로써 성능을 최첨단 수준으로 끌어올렸습니다. VinciCoder는 시각적 충실도를 개선하기 위해 RL을 적용한 최초의 VLM으로, 추가적인 평가를 통해 멀티모달 코드 생성 작업에서의 새로운 가능성을 제시합니다.



### Rethinking Facial Expression Recognition in the Era of Multimodal Large Language Models: Benchmark, Datasets, and Beyond (https://arxiv.org/abs/2511.00389)
- **What's New**: 이번 논문에서는 Facial Expression Recognition (FER) 작업의 성능을 향상시키기 위한 새로운 벤치마크인 FERBench를 제시합니다. 이는 다양한 FER 데이터세트를 수집하고, 기존의 FER 데이터세트를 시각적 질문-응답(Visual Question-Answering, VQA) 형식으로 변환하여 다목적(Multimodal) 대형 언어 모델(MLLM) 적용 가능성을 높였습니다. 연구진은 UniFER-7B라는 FER 전용 모델을 개발하여 기존의 여러 모델보다 뛰어난 성능을 입증했습니다.

- **Technical Details**: FERBench는 20개의 최신 MLLM을 사용하여 4개의 널리 사용되는 FER 데이터세트에서 성능을 평가하는 체계적인 벤치마크입니다. 연구진은 UniFer-CoT-230K과 UniFer-RLVR-360K라는 두 개의 대규모 데이터세트를 Curate (선별)하였고, UniFER-7B 모델에서 두 단계 교육 프레임워크를 활용하여 현재의 다양한 FER 데이터셋에 적합한 통합된 FER 기초 모델을 개발하였습니다.

- **Performance Highlights**: UniFER-7B 모델은 기존의 SOTA(State-of-the-Art) 공개 모델 및 비공식 모델을 초과하는 성능을 보여주었습니다. 또한, UniFER-7B는 여러 FER 데이터세트에서 일관되게 모델링 및 추론을 수행하며, 그 예측에 대한 완전한 추론 경로를 제공하여 해석 가능성을 갖추었습니다. 이는 FER 분야에서 처음으로 이 모델에 의해 증명된 'aha moment'를 불러일으킵니다.



### VisionCAD: An Integration-Free Radiology Copilot Framework (https://arxiv.org/abs/2511.00381)
- **What's New**: VisionCAD는 기존 병원 IT 시스템과의 통합 장벽을 우회하는 혁신적인 카메라 기반 진단 지원 프레임워크입니다. 이 시스템은 카메라를 사용하여 의료 이미지를 디스플레이에서 직접 캡처하고, 자동화된 파이프라인에서 이를 복원 및 분석하여 진단 품질의 이미지를 생성합니다. 이 연구는 다양한 의료 이미징 데이터셋에서 VisionCAD의 성능을 검증하며, 전통적인 CAD 시스템과 유사한 진단 성능을 달성함을 보여주었습니다.

- **Technical Details**: VisionCAD는 여섯 개의 구성 요소로 이루어진 파이프라인을 통해 시각 정보를 처리합니다. 구성 요소로는 비전 캡처기(Vision Capturer), 스크린 디텍터(Screen Detector), 품질 향상기(Quality Enhancer), 모달리티 라우터(Modality Router), 진단 엔진(Diagnostic Engine), 보고서 어시스턴트(Report Assistant)가 포함됩니다. 이 시스템은 카메라 캡처로 인한 왜곡을 교정하고, 최신 모델을 이용하여 특정 임상 작업에 대한 분석 경로를 선택하여 구조화된 임상 보고서를 생성합니다.

- **Performance Highlights**: VisionCAD는 전통적인 CAD 시스템과 유사한 진단 성능을 나타내며, 보통 F1-score가 2% 미만으로 감소합니다. 자동화된 보고서의 자연어 생성 지표는 원본 이미지에서 도출된 결과와 1% 이내로 유지됩니다. 이 시스템은 카메라 장치와 표준 컴퓨팅 자원만을 요구하여, 기존 인프라에 대한 수정 없이 다양한 임상 환경에서 진단 능력을 배포할 수 있는 접근 방식을 제공합니다.



### Who Can We Trust? Scope-Aware Video Moment Retrieval with Multi-Agent Conflic (https://arxiv.org/abs/2511.00370)
- **What's New**: 이 연구에서는 비디오 순간 검색(video moment retrieval) 작업을 위한 새로운 강화 학습 기반 모델을 제안합니다. 이 모델은 전체 비디오를 한 번 스캔하여 순간의 경계(boundary)를 찾고 이를 위한 위치 증거를 생성합니다. 또한, 여러 에이전트(agent) 간의 갈등을 해소하기 위한 다중 에이전트 시스템 프레임워크를 도입하여, 쿼리가 비디오에서 해당 순간을 갖지 않는 경우를 감지할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: 제안된 모델은 증거 기반 학습(evidential learning)을 사용하여 각 에이전트가 추정한 경계의 신뢰도를 평가합니다. 이 모델은 고정된 창 크기로 비디오 전체를 스캔하고, 각 에이전트는 독립적으로 작동하여 최종 위치에서의 갈등을 평가합니다. 이를 통해 OOS(Out-of-scope) 쿼리를 제로샷(Zero-shot) 방식으로 감지할 수 있는 기능도 제공합니다.

- **Performance Highlights**: 실험 결과, 이 방법은 최신 기법(state-of-the-art approaches)과 비교하여 성능이 향상된 것으로 나타났습니다. 다중 에이전트 시스템의 경쟁 및 갈등 모델링은 비디오 순간 검색의 강화 학습 성능을 개선하는 효과적인 방법이라는 것을 보여주었습니다. 연구 결과는 영상 레벨 검색 애플리케이션 개발에 도움이 될 것으로 기대됩니다.



### Oitijjo-3D: Generative AI Framework for Rapid 3D Heritage Reconstruction from Street View Imagery (https://arxiv.org/abs/2511.00362)
Comments:
          6 Pages, 4 figures, 2 Tables, Submitted to ICECTE 2026

- **What's New**: 방글라데시의 문화유산 복원은 자원 부족과 기술 전문성 부족이라는 두 가지 문제에 직면해 있습니다. Oitijjo-3D는 비용이 들지 않는 generative AI 프레임워크로, Google Street View 이미지를 활용하여 3D 모델을 신속하게 재구성합니다. 이 시스템은 전통적인 3D 디지털화 방법을 대체하며, 비용과 기술 장벽을 크게 낮춰 문화유산 보존을 더 많은 사람들이 할 수 있도록 합니다.

- **Technical Details**: Oitijjo-3D는 두 단계의 파이프라인을 통해 작동합니다: Gemini 2.5 Flash Image를 사용한 구조-텍스처 합성 및 Hexagen을 통한 신경 이미지-3D 생성입니다. 이러한 방법은 고급 하드웨어나 전문가 감독 없이도 사진 실사와 일관된 3D 재구성을 가능하게 합니다. 최종적으로 생성된 모델은 glTF 2.0 형식으로 출력되어 다양한 형식으로 사용자에게 제공됩니다.

- **Performance Highlights**: Oitijjo-3D는 기존의 SfM+MVS 방법에 비해 250배 이상의 속도 향상을 달성했습니다. 일반적으로 4-8시간 걸리는 이미지 처리 과정이 45초 이내에 완료되며, 필요한 메모리 또한 최소화됩니다. 이 성능은 자금이나 전문 인력이 부족한 개발도상국의 환경에서 접근 가능한 문화유산의 디지털화를 촉진할 수 있습니다.



### Transfer Learning for Onboard Cloud Segmentation in Thermal Earth Observation: From Landsat to a CubeSat Constellation (https://arxiv.org/abs/2511.00357)
Comments:
          This work was presented at the TerraBytes Workshop at the 42nd International Conference on Machine Learning. This version is not part of the official ICML proceedings

- **What's New**: 이번 연구는 제한된 하드웨어와 스펙트럼 정보에 의해 제약받는 CubeSat 미션을 위한 열 구름 세분화(thermal cloud segmentation) 작업을 다룹니다. 특히 FOREST-2 CubeSat의 성능을 개선하기 위해 Transfer Learning을 활용하여 경량인 MobileNet 인코더를 갖춘 UNet을 적용하였습니다. 데이터를 수집하고 천연 자원 보호를 위해 CubeSat의 고유한 특성(특히 단일 열 대역 활용)을 극복하는 접근 방식을 시도하였습니다.

- **Technical Details**: 본 연구에서는 Landsat-7 Cloud Cover Assessment Dataset을 활용한 Transfer Learning 방식을 적용하여 CubeSat 데이터의 구름 세분화 성능을 높였습니다. 총 6,000개의 Landsat 이미지와 500개의 FOREST-2 레이블이 포함된 샘플을 결합하여 joint-training 방식을 통해 macro F1 score이 0.850에서 0.877로 향상되었습니다. 모델은 TensorRT 엔진으로 변환되어 NVIDIA Jetson Nano에서 5초 이내의 이미지 추론을 보여주었습니다.

- **Performance Highlights**: 결과적으로, 공간적으로 다양한 고품질 공용 데이터셋을 사용한 사전 훈련(pretraining)과 소량의 미션 특화 샘플로 미세 조정(fine-tuning)을 통해 초기 CubeSat 미션에서 구름 마스킹을 효율적으로 지원할 수 있는 가능성을 보여줍니다. 이 접근법은 마이크로사이즈 애널리틱스에 기여하며, 현대적인 은하 등의 목표를 갖춘 CubeSat 미션에서 실시간 의사 결정을 지원할 수 있습니다.



### Detecting AI-Generated Images via Diffusion Snap-Back Reconstruction: A Forensic Approach (https://arxiv.org/abs/2511.00352)
Comments:
          6 pages, 8 figures, 4 Tables, submitted to ICECTE 2026

- **What's New**: 최근의 생성적 확산 모델(generative diffusion models)의 급격한 발전으로 인해, 진짜 시각 콘텐츠와 합성 이미지(synthetic imagery)를 구별하는 것이 점점 더 어려워지고 있습니다. 안정적 확산 모델(Stable Diffusion)과 DALL-E와 같은 최신 텍스트-이미지 시스템은 포토리얼리즘(photorealism)과 아티팩트 없는 결과를 생성하여 전통적인 딥페이크 탐지 방법이 실패하게 만듭니다. 본 연구에서는 다중 강도 이미지 재구성 동역학(multi-strength image reconstruction dynamics)을 활용한 확산 기반 포렌식 프레임워크를 소개합니다.

- **Technical Details**: 이 프레임워크는 재구성 메트릭(LPIPS, SSIM, PSNR)의 변화를 분석하여 실제 이미지와 AI 생성 이미지를 구별할 수 있는 해석 가능한 매니폴드 기반 특징을 추출합니다. 연구는 4,000개의 이미지를 포함하는 균형 잡힌 데이터셋에서 평가되었으며 교차 검증(cross-validation)에서 0.993 AUROC를 달성했습니다. 제한된 데이터와 단일 확산 백본(Stable Diffusion v1.5)을 사용했음에도 불구하고, 제안된 방법은 강력한 일반화와 해석 가능성을 입증합니다.

- **Performance Highlights**: 제안된 방법은 압축(compression) 및 노이즈(noise)와 같은 일반적인 왜곡에 대해서도 강건성을 유지하며, AI가 생성한 미디어에 대한 포렌식 분석을 위한 기초를 제공합니다. 독립적인 검증 플랫폼(public verification platform)을 통해 사용자는 이미지나 비디오를 업로드하여 AI로 생성되었는지 여부를 확인할 수 있는 가능성도 제시합니다. 이는 정보가 왜곡되고 있는 현대 사회에서 특히 가치 있는 접근 방식으로 평가됩니다.



### OSMGen: Highly Controllable Satellite Image Synthesis using OpenStreetMap Data (https://arxiv.org/abs/2511.00345)
Comments:
          Accepted at NeurIPS 2025 UrbanAI Workshop

- **What's New**: 이번 연구에서는 OSMGen이라는 새로운 생성 프레임워크를 소개하고 있습니다. 이 프레임워크는 OpenStreetMap (OSM) 데이터를 기반으로 실제와 유사한 위성 이미지를 생성합니다. 기존의 래스터 타일에 의존하는 방식과 달리, OSMGen은 OSM의 JSON 구조를 활용하여 높은 수준의 세부 정보를 제어할 수 있도록 합니다.

- **Technical Details**: OSMGen은 OSM JSON의 구조화된 정보를 기반으로 하여 고충실도 위성 이미지를 생성하는 혁신적인 생성 모델을 개발하였습니다. 이 모델은 시각적 변경 사항을 세밀하게 반영할 수 있도록 OSM 데이터를 편집하여 '변화 전'과 '변화 후'의 이미지를 일관되게 생성할 수 있습니다. 여기서 DDIM(Denoising Diffusion Implicit Models) 변환을 통해 텍스트 조건 및 공간 임베딩을 결합하여 제어된 장면 조작이 가능합니다.

- **Performance Highlights**: 모델은 2,000개의 장소를 대상으로 한 평가에서 도로 네트워크 및 건물 윤곽을 정확하게 재현하였고, 특정 마스크를 통해 희귀 장소(POI) 클래스를 올바르게 표현했습니다. 이렇게 생성된 데이터셋은 지리적 AI의 데이터 부족을 해결하고, 도시 계획자들이 제안된 개발의 영향을 시각화하는 데 도움이 될 것입니다.



### Federated Dialogue-Semantic Diffusion for Emotion Recognition under Incomplete Modalities (https://arxiv.org/abs/2511.00344)
- **What's New**: 본 논문에서는 Federated Dialogue-guided and Semantic-Consistent Diffusion (FedDISC) 프레임워크를 제안하여 멀티모달 신호의 부족 현상을 해결합니다. 기존의 방법들은 완전한 데이터에 의존하는 반면, FedDISC는 연합 학습(federated learning)을 활용하여 각 클라이언트에서 그들의 지역적 특성에 맞춘 모델을 훈련할 수 있도록 합니다. 이 접근법은 데이터의 프라이버시를 유지하면서도 여러 클라이언트에서 발생하는 멀티모달 결손에 대응할 수 있게 합니다.

- **Technical Details**: FedDISC에서는 두 가지 주요 구성요소를 가진 DISC-Diffusion 모델을 설계하였습니다. 첫 번째는 대화 그래프 네트워크(Dialogue Graph Network, DGN)로, 대화의 맥락을 이해하는 데 필요한 의존성을 캡처합니다. 두 번째는 의미 조건 네트워크(Semantic Conditioning Network, SCN)로, 가용한 모달리티에서 의미 정보를 추출하여 복원된 모달리티와 가용 모달리티 간의 의미 정합성을 보장합니다. 또한, Alternating Frozen Strategy (AFS)가 도입되어 최적화 충돌을 해결합니다.

- **Performance Highlights**: IEMOCAP, CMUMOSI, CMUMOSEI 데이터셋에 대한 광범위한 실험 결과, FedDISC는 다양한 결손 모달리티 패턴에서 기존의 접근 방법들과 비교하여 뛰어난 감정 분류 성능을 보여줍니다. 이 프레임워크는 단일 클라이언트의 완전성에 대한 의존성을 없애고, 다양한 클라이언트 간의 제로샷 크로스 클라이언트 복원을 가능하게 함으로써, 현실 세계의 다양한 시나리오에서의 실효성을 입증합니다.



### A DeepONet joint Neural Tangent Kernel Hybrid Framework for Physics-Informed Inverse Source Problems and Robust Image Reconstruction (https://arxiv.org/abs/2511.00338)
- **What's New**: 이번 연구에서는 Deep Operator Networks (DeepONet)와 Neural Tangent Kernel (NTK)을 통합하여 복잡한 역문제를 해결하는 새로운 하이브리드 접근법을 제안합니다. 이 방법은 Navier-Stokes 방정식에 의해 결정되는 소스 로컬리제이션과 이미지 재구성 작업에서 비선형성, 희소성, 잡음 데이터를 극복하며 효과적으로 작업을 수행합니다. 또한 물리적으로 일관되고 정확한 솔루션을 보장하기 위해 물리학 정보 제약과 작업별 정규화를 손실 함수에 통합했습니다.

- **Technical Details**: 제안된 방법론은 DeepONet과 NTK를 결합하여 역문제를 해결하기 위한 새로운 방법을 제시합니다. 이 접근법은 특히 물리 법칙이 결합된 데이터를 통해 불확실한 소스 매개변수 복구 문제를 해결하기 위해 두 가지 신경망을 활용합니다. DeepONet은 두 개의 서브네트워크인 브랜치 네트워크와 트렁크 네트워크로 구성되어 있으며, NTK는 학습의 안정성을 높이고 수렴을 가속화하는 역할을 합니다.

- **Performance Highlights**: 다양한 합성 및 실제 데이터셋에 대한 검증 결과, 제안된 방법이 제한적이거나 잡음이 있는 데이터에서도 정확한 점 소스 위치와 강도의 예측을 가능하게 함을 입증했습니다. NTK의 포함으로 인해 훈련 안정성이 향상되고 수렴 속도가 빨라져 복잡한 파동 현상 및 컴퓨터 비전의 역문제를 해결하는 데 있어 이 방법의 강력함과 잠재력을 강조합니다.



### Beyond ImageNet: Understanding Cross-Dataset Robustness of Lightweight Vision Models (https://arxiv.org/abs/2511.00335)
Comments:
          10 pages, 5 tables, 1 figure, 3 equations, 11 mobile models, 7 datasets

- **What's New**: 이 논문에서는 11개의 경량 비전 모델을 7개의 다양한 데이터셋에서 체계적으로 평가한 결과를 소개합니다. 이를 통해 xScore라는 새로운 평가 지표를 개발하여 모델 성능의 일관성과 강건성을 정량화합니다. 특히 ImageNet에서의 성능이 의료 데이터셋이나 세부 데이터셋에 대한 예측력과 일치하지 않음을 보여주었습니다.

- **Technical Details**: 경량 비전 모델의 성능을 평가하기 위해 고정된 100 에포크 스케줄로 훈련된 11개의 모델을 사용할었습니다. 이 연구는 자원 제약이 있는 환경에서의 일반화 가능성을 조사하며, 특정 아키텍처 요소가 성능 향상에 기여하는지 분석합니다. 가령, 고해상도 및 채널 주의 메커니즘을 사용한 동질(convolution) 수작업이 더 넓은 일반화를 촉진하는 것으로 나타났습니다.

- **Performance Highlights**: xScore는 모바일 모델의 성능을 예측하는 확장 가능한 지표로, 단지 네 개의 데이터셋에서의 성능을 기반으로 추정할 수 있다는 점에서 유용합니다. 이 연구는 경량화된 비전 모델이 ImageNet 기준으로 평가되었을 때의 논란과 우려를 제기하며, 자원 제약 하에서의 강건한 일반화를 위한 설계 원칙을 강조합니다.



### Towards Automated Petrography (https://arxiv.org/abs/2511.00328)
- **What's New**: 이 논문은 자동 석유 및 광물 분석을 위한 대규모 실험 프레임워크인 LITHOS를 소개합니다. LITHOS는 211,604개의 고해상도 RGB 패치와 25개 광물 범주에 걸쳐 105,802개의 전문가 주석을 포함하여, 기존 데이터 세트보다 두 배나 많은 데이터를 제공합니다. 이 데이터베이스는 연구자들이 석유 및 광물 분석 자동화를 위한 더 나은 방법을 개발할 수 있도록 지원할 것입니다.

- **Technical Details**: LITHOS 데이터 세트는 PPL(평면 편광) 및 XPL(교차 편광) 조건에서 캡처된 고해상도 이미지 패치로 구성되어 있습니다. 각각의 광물 결정은 전문가가 정의한 주석, 즉 광물 클래스, 공간 좌표, 대 및 소 축의 교차 벡터 경로를 포함하고 있어, 결정의 기하학과 방향을 잘 나타냅니다. 데이터 세트는 실제 석유 및 광물 분석의 복잡성을 반영하고 있으며, 다양한 광물 범주에서 나타나는 유사성을 도해합니다.

- **Performance Highlights**: LITHOS 베이스라인은 쌍편광 촬영기법을 활용한 변환기 아키텍처로, 기존의 단일 편광 모델보다 모든 지표에서 지속적으로 우수한 성능을 보입니다. 이를 통해 편광 정보의 융합이 석유 및 광물 분류에서 기여하는 유용성을 극대화합니다. 모든 데이터 및 사전 훈련된 모델은 CC BY-NC-SA 4.0 라이선스 하에 공개되며, 재현성 및 투명성을 증진하고, 향후 연구에 기여할 것입니다.



### Multi-View Consistent Human Image Customization via In-Context Learning (https://arxiv.org/abs/2511.00293)
- **What's New**: 최근 개인화 생성 모델의 발전은 다양한 환경에서 동일 인물의 일관된 이미지를 생성하는 데 인상적인 결과를 보여주고 있습니다. 그러나 대부분의 방법은 생성된 이미지의 시점을 조절하거나 인물의 일관된 다중 뷰를 생성할 수 없습니다. 이에 대한 해결책으로 PersonalView라는 경량화된 적응 방식을 제안하며, 이는 100개의 훈련 샘플만으로 기존 모델이 다중 뷰 생성 능력을 습득할 수 있게 합니다.

- **Technical Details**: PersonalView는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 우리는 사전 학습된 diffusion transformer의 문맥 학습 능력을 활용하기 위한 조건부 아키텍처를 설계했습니다. 둘째, 새로운 Semantic Correspondence Alignment Loss를 통해 사전 학습된 모델의 원래 생성 능력을 유지합니다. 이 방법은 기존 데이터셋에 대한 대규모 훈련 없이도 빠르게 다중 뷰 생성을 가능하게 합니다.

- **Performance Highlights**: PersonalView는 다중 뷰 일관성, 텍스트 정렬, 정체성 유사성 및 시각적 품질을 평가한 결과, 기존의 여러 기준선에 비해 월등한 성능을 보였습니다. 단 100개의 훈련 샘플만을 이용해 대규모 다중 뷰 데이터로 훈련한 모델보다도 우수한 결과를 보여주며, 다양한 속성에서 일관성을 유지하면서도 사용자 요청에 대한 반응성이 뛰어났습니다.



### FedReplay: A Feature Replay Assisted Federated Transfer Learning Framework for Efficient and Privacy-Preserving Smart Agricultur (https://arxiv.org/abs/2511.00269)
- **What's New**: 이 논문에서는 스마트 농업을 위한 새로운 연합 학습 프레임워크(Federated Learning framework)를 제안합니다. 연합 학습의 개념을 활용해 CLIP(Contrastive Language-Image Pre-training) 모델의 고정된 비전 변환기(vision transformer)와 경량 비전 분류기(classifier)를 통합하여, 비공식(raw) 데이터 공개를 최소화하고 통신 비용을 줄였습니다. 또한, 비독립적 및 동일 분포가 아닌(non-IID) 데이터의 성능 저하를 완화하기 위해 클라이언트 간에 공유하는 경우에 1%의 클립 기반(feature representation) 특징만을 공유하여 개인 정보를 보호합니다.

- **Technical Details**: 제안된 FedReplay 프레임워크는 클립 비전 변환기(CLIP ViT)를 사용하여 강력한 특징을 추출하고, 이러한 특징을 기반으로 경량 변환기 분류기를 학습시킵니다. 이 방식은 대규모 데이터셋으로부터 사전 훈련된 모델을 활용함으로써 전체 모델을 처음부터 학습하는 부담을 피할 수 있습니다. 또한, 전체 파라미터를 전송할 필요 없이 공유된 특징을 활용하여 통신 오버헤드를 약 98% 줄이는 성능을 보였습니다.

- **Performance Highlights**: 농업 데이터 분류 작업에 대한 실험 결과, FedReplay 프레임워크는 86.6%의 정확성을 달성하며 기존 연합 학습 접근법에 비해 4배 이상의 개선을 나타냈습니다. 또한, 통신 효율성을 크게 향상시켜 스마트 농업에서의 효과적이고 확장 가능한 배치를 가능하게 했습니다. 이러한 결과는 비전-언어 모델 기능을 연합 학습과 결합하여 개인 정보를 보호하며, 통신 효율성 및 높은 성능을 달성할 수 있음을 증명합니다.



### Spot The Ball: A Benchmark for Visual Social Inferenc (https://arxiv.org/abs/2511.00261)
- **What's New**: 이번 논문에서는 'Spot The Ball'라는 도전적인 벤치마크를 소개하여 비전-언어 모델(Vision-Language Models, VLMs)의 시각적 사회 추론 능력을 평가합니다. 인간은 타인의 시선, 자세, 방향 등의 미세한 행동 신호를 통해 장면에서 숨겨진 요소를 추론하는 능력이 탁월합니다. 이 연구는 축구, 농구, 배구 이미지를 사용하여 제거된 스포츠 공의 위치를 찾아내는 과제를 제시합니다.

- **Technical Details**: 분석을 위해 선택된 평가 세트는 공공 방송 영상에서 추출한 150개 이미지로 구성되어 있으며, 이 이미지는 비시각적 맥락 정보(예: 공의 비가림 여부와 선수의 분포 등)를 극대화하는 방식으로 선택되었습니다. 이러한 이미지에 대해 '제로샷(zero-shot)' 방식으로 한정된 사회적 및 물리적 맥락을 통합하여 숨겨진 요소를 추론하는 것이 요구됩니다. 이를 통해 기존에는 실현되지 않았던 사회적 단서를 기반으로 한 정보 추론 기준을 설정합니다.

- **Performance Highlights**: 논문에서는 인간이 VLM보다 2-3배 더 정확하다는 결과를 보여주었으며, 네 가지 최신 VLM(제미니, GPT, LLaMA, Qwen)이 평가되었습니다. 분석 결과, 현대 모델들은 주로 표면적인 공간 휴리스틱(spatial heuristics)에 의존하는 반면, 인간은 시선 방향이나 신체 자세와 같은 사회적 신호를 활용함을 알 수 있었습니다. 이 발견은 시각적 사회 추론에서 모델과 인간 간의 지속적인 격차를 강조합니다.



### MambaNetLK: Enhancing Colonoscopy Point Cloud Registration with Mamba (https://arxiv.org/abs/2511.00260)
Comments:
          12 pages, 4 figures, 3 tables, IPCAI conference

- **What's New**: 이번 연구는 내시경 내비게이션을 위한 새로운 3D 등록 방법을 소개하며, 이를 위해 세밀하게 설계된 C3VD-Raycasting-10k라는 고품질 임상 데이터셋을 활용합니다. MambaNetLK라는 신기술은 점구름(point cloud) 간의 매칭 없는 등록 프레임워크로, Mamba State Space Model(SSM)을 통합하여 포괄적인 특징 추출을 가능하게 합니다. 이는 효율적으로 장거리 의존성을 포착할 수 있으며, 루카스-카나데(Lucas-Kanade) 알고리즘을 사용하여 반복적으로 정렬을 수행합니다.

- **Technical Details**: MambaNetLK는 동적 포인트 클라우드의 등록을 위해 점군을 시퀀스(sequence)로 처리하며, Mamba SSM을 이용하여 글로벌 기하 구조를 효율적으로 캡처합니다. 이 프레임워크는 피처 벡터의 차이를 최소화하기 위해 루카스-카나데(IC-LK) 알고리즘을 활용하여 반복적으로 변환을 최적화합니다. 우리의 방법은 깊은 신경망을 통해 추출된 글로벌 구조 설명자 간의 불일치를 최소화하여 점 구름을 정렬합니다.

- **Performance Highlights**: MambaNetLK는 C3VD-Raycasting-10k 임상 데이터셋에서 최신 기술들에 비해 우수한 성능을 보여주며, 중앙 회전 오류를 56.04% 줄이고, RMSE 번역 오류를 26.19% 감소시킵니다. 또한 ModelNet40에서 강력한 일반화 성능을 나타내고 초기 자세 변동에 대해 뛰어난 견고성을 보여줍니다. MambaNetLK는 최소 침습 수술 절차에서의 보다 정확하고 신뢰성 있는 내비게이션 시스템을 구축하기 위한 강력한 기초를 제공합니다.



### BeetleFlow: An Integrative Deep Learning Pipeline for Beetle Image Processing (https://arxiv.org/abs/2511.00255)
Comments:
          4 pages, NeurIPS 2025 Workshop Imageomics

- **What's New**: 이번 연구에서는 곤충학 및 생태학 연구에서 필요한 많은 숫자의 딱정벌레 이미지를 처리하기 위한 3단계 딥러닝 파이프라인을 개발했습니다. 이 파이프라인은 딱정벌레를 자동으로 탐지하고 분류하는 프로세스를 자동화하여, 생물학적 연구의 효율성을 크게 향상시킵니다. 기존의 open-vocabulary detection과 transformer 기반 segmentation을 활용하여, 매우 높은 정확도를 달성하는 방향으로 설계되었습니다.

- **Technical Details**: 파이프라인의 첫 단계에서는 Grounding DINO라는 open-vocabulary 탐지기를 이용하여 딱정벌레를 탐지합니다. 두 번째 단계에서는 탐지된 딱정벌레를 잘라내어 개별 이미지로 저장하며, 선택적으로 메타데이터와 정렬하는 기능을 포함합니다. 마지막 단계에서는 Mask2Former 모델을 사용하여 각 딱정벌레의 형태학적(segmentation) 세분화를 수행하며, 이는 5개 또는 9개의 형태학적 부위로 나누어집니다.

- **Performance Highlights**: 본 연구에서는 1,506개의 딱정벌레 이미지가 포함된 NEON 데이터셋을 활용하여 파이프라인의 성능을 평가했습니다. 97.81%의 정확도로 탐지된 딱정벌레 수가 사실관계(ground truth)와 일치하여, 모델의 효율성을 입증했습니다. 이러한 파이프라인은 곤충 이미지를 처리하는 다른 생물학적 데이터 케이스에 일반화될 가능성을 보여주어, 향후 생물학적 연구에 매우 유용할 것입니다.



### Merlin L48 Spectrogram Datas (https://arxiv.org/abs/2511.00252)
Comments:
          Accepted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Track on Datasets and Benchmarks

- **What's New**: 이번 논문에서는 L48 데이터셋을 소개합니다. 이는 특정 클래스의 존재를 가진 단일 양성을 지닌 실제 세계의 다중 라벨 데이터셋으로, 조류의 소리 녹음을 통해 생성되었습니다. 기존의 합성 데이터셋이 반영하지 못하는 미세한 복잡성을 포착하여 이전 SPML 방법들과의 성능 차이를 분석합니다.

- **Technical Details**: L48 데이터셋은 조류 인식 워크플로우와 일치하도록 밀접하게 주석이 달린 스펙트로그램 바운딩 박스를 포함하고 있습니다. 이 데이터셋은 연간 및 국토 전역에 걸쳐 조류 소리를 포함하고 있으며, 단일 긍정 레이블 및 추가 부정 레이블에 대한 접근을 제공합니다. 기존 SPML 방법들이 L48에서 효과를 발휘하지 못하는 이유를 분석한 결과, 실제 세계의 도전이 드러납니다.

- **Performance Highlights**: L48에서 기존 SPML 방법들이 간단한 레이블 스무딩 기준선 모델을 초과하는 데 실패한 것을 발견하였습니다. 이는 잘못된 레이블 분포 및 미세한 레이블 모호성과 관련이 있으며, 이러한 문제를 해결하기 위해 새로운 정규화 방법을 제안하여 모든 평가 방법의 성능을 향상시켰습니다. L48은 실용적인 SPML 벤치마크로서 미래의 연구 방향에 기여할 수 있습니다.



### Object-Aware 4D Human Motion Generation (https://arxiv.org/abs/2511.00248)
- **What's New**: 이 논문은 3D 물리적 정보의 부재로 인해 발생하는 비현실적인 변형과 물리적 비일관성을 해결하는 새로운 접근법을 제안합니다. 우리는 Motion Score Distilled Interaction (MSDI)이라는 프레임워크를 통해, 객체 인식이 가능한 4D 인간 움직임 생성을 위한 방법을 소개합니다. 이 방법은 기존의 비디오Diffusion 모델에 기반하여, 공간과 언어적 정보를 활용하여 현실적이고 신뢰할 수 있는 인간 움직임을 생성합니다.

- **Technical Details**: 이 연구에서는 사전 설치된 3D 인간 및 객체를 사용하여 고충실도의 3D Gaussian을 생성하고, 대형 언어 모델(LLM)에서 제공하는 공간 정보를 통해 인간의 동작을 제어합니다. 제안된 Motion Diffusion Score Distillation Sampling (MSDS)을 통해, 인간의 자세와 궤적을 동작 우선 순위 및 상호작용 제약과 조화롭게 조정하는 최적화 과정을 형성합니다. 이를 통해 정적 객체에 대한 물리적 제약을 준수하는 자연스러운 움직임 시퀀스를 생성합니다.

- **Performance Highlights**: 테스트 결과, 우리의 프레임워크는 물리적 제약을 준수하면서 실제적이고 일관된 인간 움직임을 생성함을 보여줍니다. 이전의 4D 생성 방식들이 생성한 비현실적인 왜곡과 달리, 우리의 접근법은 제로 샷 설정에서 일반화되고, 여러 프로프트에서 신뢰할 수 있는 4D 장면을 생성합니다. 이는 미래의 움직임 확산 모델이 발전함에 따라 추가적인 재교육 없이도 지속적으로 개선될 수 있는 확장 가능한 경로를 제공합니다.



### Hyperbolic Optimal Transpor (https://arxiv.org/abs/2511.00244)
Comments:
          65 pages, 21 figures

- **What's New**: 이 논문에서는 최적 수송(optimal transport, OT) 문제를 하이퍼볼릭 공간(hyperbolic space)에서 해결하는 새로운 알고리즘을 제안합니다. 기존의 최적 수송 방법들이 주로 유클리드 공간(Euclidean space)과 구면(sphere)에 집중되어 있었던 것을 반영하여, 비유클리드 지형에서의 문제 해결에 중점을 두고 있습니다.

- **Technical Details**: 이 연구는 기하학적 변분 기법(geometric variational technique)을 활용하여 유클리드 및 구면 지오메트리의 방법을 하이퍼볼릭 설정으로 확장합니다. 이로 인해 계층 데이터(hierarchical data), 네트워크(networks), 다중 제너스 리만 서페이스(multi-genus Riemann surfaces)와 같은 복잡한 구조에서도 최적 수송 맵(optimal transport map)을 계산할 수 있습니다.

- **Performance Highlights**: 제안된 방법의 효용성을 검증하기 위해 합성 데이터(synthetic data)와 다중 제너스 서페이스 모델(multi-genus surface models)에 대해 실험을 수행했습니다. 실험 결과는 제안된 알고리즘이 하이퍼볼릭 공간 내에서 효과적으로 최적 수송 맵을 계산할 수 있음을 보여줍니다.



### Towards 1000-fold Electron Microscopy Image Compression for Connectomics via VQ-VAE with Transformer Prior (https://arxiv.org/abs/2511.00231)
- **What's New**: 이번 연구에서는 페타스케일 전자 현미경(EM) 데이터셋을 압축하기 위한 벡터 양자화 변량 오토인코더 기반(VQ-VAE) 프레임워크를 제안합니다. 이 프레임워크는 16배에서 1024배까지의 극한 압축을 가능하게 하며, 선택적으로 Texture를 복원할 수 있는 기능을 제공합니다. 또한, 사용자가 필요에 따라 고해상도 재구성을 수행할 수 있는 ROI 기반 워크플로우를 도입하였습니다.

- **Technical Details**: 제안된 압축 및 재구성 파이프라인은 여러 수준의 잠재 공간을 처리하며, 1024×1024 이미지까지 가능하게 합니다. 인코더는 다층 컨볼루션 구조를 이용하며, 각 위치에서 96차원 특성을 가지는 양자화된 벡터를 생성합니다. 디코더는 양자화된 특성을 고해상도로 복구하기 위한 전이 합성곱을 통해 출력 이미지를 생성합니다. 이 과정에서 FiLM(Flexible Instance Normalization)을 사용하여 하위 양자화된 특성을 조정합니다.

- **Performance Highlights**: 압축 비율을 증가시키면서도 구조적 유사성 지수(SSIM)를 통해 원본 EM 이미지 대비 텍스처 변화를 정량화한 결과, AVIF와 비교할 때 16배 및 64배 압축에서도 경쟁력 있는 성능을 보였습니다. 특히 1024배 압축에서도 2D 세그멘테이션의 정확성을 유지하였으며, 신경 접합부 예측도 온전한 결과를 나타냈습니다. 더 나아가, 특정 분야(예: 미토콘드리아 분석)에 대해서는 지역 선택적 고해상도 재구성이 가능하다는 장점을 지니고 있습니다.



### DM-QPMNET: Dual-modality fusion network for cell segmentation in quantitative phase microscopy (https://arxiv.org/abs/2511.00218)
Comments:
          5 pages, 4 figures

- **What's New**: 이 논문은 단일 샷 정량적 위상 현미경(single-shot quantitative phase microscopy, ssQPM)에서 세포 분할(Cell Segmentation)의 문제를 해결하기 위한 새로운 네트워크 구조인 DM-QPMNet을 소개합니다. 기존의 단일 모달 리지 방법이 잡음(noise)과 세포 밀도에 민감한 반면, DM-QPMNet은 다중 모달리티(multi-modality)를 활용하여 효과적인 정보 융합(fusion)을 통해 견고한 세포 분할을 구현합니다. 이는 다양한 위상 정보와 극성 강도(polarized intensity) 이미지를 독립적으로 처리하여 상호작용하는 이중 인코더 구조를 바탕으로 합니다.

- **Technical Details**: 네트워크는 서로 다른 인코딩 스트림을 가진 두 개의 인코더를 사용하여 극성 강도와 위상 맵을 별개의 모달리티로 처리합니다. 중간 깊이에서 다중 헤드 주의(multi-head attention)를 통해 모달리티별 특징을 융합하여, 고주파 경계(edge)와 텍스처 특징이 선택적으로 상보적인 위상 정보를 통합할 수 있게 합니다. 이러한 설계는 훈련의 안정성을 유지하면서도 최소한의 매개변수로 다중 모달 통합을 구현합니다.

- **Performance Highlights**: DM-QPMNet은 단일 모달리티 방법 및 조기 융합(early-fusion) 기법에 비해 현저한 성능 향상을 보여주었습니다. 특히, 각 모달리티의 특성에 맞춘 인코딩 방식과 늦은 융합(late fusion)을 통해 ssQPM에서 유용한 고주파 및 저주파 정보를 효과적으로 수집하고 활용할 수 있음을 입증합니다. 이러한 접근법은 생물학적 샘플 분석에서의 신뢰성과 정확성을 높이는 데 기여할 것으로 예상됩니다.



### An Efficient and Generalizable Transfer Learning Method for Weather Condition Detection on Ground Terminals (https://arxiv.org/abs/2511.00211)
- **What's New**: 본 논문은 지구 저궤도(LEO) 위성으로 제공되는 위성 인터넷의 신뢰성 있는 성능이 기상 조건에 따라 크게 영향을 받음을 보여주고 있습니다. 또한, 기상 이벤트, 특히 눈과 비가 위성 인터넷의 주요 지상 터미널 구성 요소의 성능을 저하시킬 수 있다는 점을 강조합니다. 이 연구는 기상 조건 감지의 효율적인 transfer learning (TL) 방법을 제안하여, 다양한 기상 조건에서의 위성 안테나 분류 문제를 해결하고자 합니다.

- **Technical Details**: 전이 학습(TL)을 활용하여, 기상 데이터베이스에서 훈련된 모델의 지식을 위성 안테나 분류 작업에 적용합니다. 특히, 제안된 방법에서는 YOLACT 프레임워크를 사용하여 지상 터미널의 위성 안테나를 효과적으로 세분화하고 특징을 분리합니다. 이 연구는 데이터 부족 문제 해결을 목표로 하며, 다양한 기상 조건에서 위성 안테나의 상태를 정확하게 분류하도록 모델을 적응시킵니다.

- **Performance Highlights**: 제안된 모델은 초기 시나리오에서 80개의 훈련 이미지를 사용하여 50 에포크 내에 88.33%의 정확도를 달성하며, 기존의 YOLO 및 Faster R-CNN 모델을 초과하는 성능을 보여주었습니다. 확장된 시나리오에서도 180개의 훈련 이미지를 사용해 50 에포크 동안 동일한 정확도를 유지하며, 경쟁 모델인 R-YOLO와 Faster R-CNN에 비해 높은 성능을 입증했습니다.



### A Retrospect to Multi-prompt Learning across Vision and Languag (https://arxiv.org/abs/2511.00191)
Comments:
          ICCV

- **What's New**: 최근 비전-언어 프리트레이닝 모델(VLM)의 발전으로 멀티-프롬프트 학습(multi-prompt learning) 기술이 주목받고 있습니다. 기존의 연구는 단일 프롬프트 패러다임에 초점을 맞추는 반면, 본 논문은 이론적 기반을 바탕으로 멀티-프롬프트 학습을 다룹니다. 특히 에너지 기반 멀티-프롬프트 학습(EMPL)을 제안하여 여러 프롬프트 임베딩을 생성함으로써 인 도메인(in-domain)과 아웃 도메인(out-of-domain) 전이 학습(transfer learning)의 균형을 이룹니다.

- **Technical Details**: 본 논문은 VLM을 위한 멀티-프롬프트 학습의 기초를 다지며, 에너지 기반의 분포를 통해 프롬프트를 학습하는 EMPL 메소드에 대해 설명합니다. 임베딩 파라미터의 최적화를 위해, 우리는 프롬프트와 이미지를 변수로 사용하여 불확실성과 관계된 모델링을 진행합니다. 또한, 이중 마르코프 체인 몬테카를로(MCMC) 샘플러를 활용해 반복적으로 프롬프트를 생성함으로써 파라미터 효율성을 극대화하고, 기존의 프롬프트 학습 전략과의 호환성을 확보합니다.

- **Performance Highlights**: 종합적인 실험을 통해 제안된 EMPL의 우수성을 입증하며, 멀티-프롬프트 학습이 단일 프롬프트보다 더 나은 성능을 발휘함을 확인하였습니다. 특히, VLM의 비전-언어 전이 가능성을 높이고, 다수의 프롬프트를 사용함으로써 모델의 일반화 문제를 해결하는 데 효과적임을 보여줍니다. 본 연구는 VLM의 발전 방향에 새로운 통찰력을 제공하며, 멀티-프롬프트 학습의 가능성을 한층 확장시킬 것입니다.



### From Evidence to Verdict: An Agent-Based Forensic Framework for AI-Generated Image Detection (https://arxiv.org/abs/2511.00181)
Comments:
          20 pages, 6 figures

- **What's New**: 이 논문에서는 정보 무결성 및 미디어 진위성에 대한 도전 과제를 해결하기 위해 AIFo(Agent-based Image Forensics)라는 새로운 프레임워크를 소개합니다. AIFo는 인간의 법의학 조사 방법을 모방한 협업 기반의 다중 에이전트를 통해, 전통적인 탐지 방식의 한계를 극복하려고 합니다. 또한, AIFo는 훈련 없이 AI 생성 이미지를 효과적으로 탐지하는 데 필요하고 다양한 기능을 갖춘 도구를 활용합니다.

- **Technical Details**: AIFo는 이미지 테스트를 위해 포렌식 툴박스, 증거 수집기, 추론 에이전트 및 토론 모듈로 구성된 LLM 기반 다중 에이전트 시스템입니다. 증거 수집기는 다양한 도구를 사용해 이미지를 분석하고, 이에 대한 평가 과정에서 수집된 증거의 질을 평가합니다. 수집된 증거가 불충분할 경우, 토론 모듈이 활성화되어 서로 다른 두 에이전트가 상반된 주장을 교환하면서 최종 판단을 내리게 됩니다.

- **Performance Highlights**: AIFo는 6,000개의 이미지 데이터셋에서 97.05%의 정확도를 기록하며, 기존의 전통적인 분류기 및 최신 VLM보다 우수한 성능을 보였습니다. 또한, 메모리 모듈을 통해 과거 사례를 학습하여 점차 탐지 정확도를 향상시킬 수 있는 잠재력을 가지고 있습니다. 논문에서 제안한 AIFo는 AI 생성 이미지 탐지의 새로운 패러다임으로 자리매김할 수 있는 가능성을 보여줍니다.



### CompAgent: An Agentic Framework for Visual Compliance Verification (https://arxiv.org/abs/2511.00171)
Comments:
          Under review

- **What's New**: 새롭게 제안된 CompAgent는 시각적 준수 검증(visual compliance verification)을 위한 최초의 에이전트 시스템입니다. 이 시스템은 다양한 비전 도구를 활용하여 준수 정책에 기반한 도구를 동적으로 선택하며, 복잡한 준수 규칙을 효율적으로 관리합니다. 기존 방법들이 의존하는 수동 데이터 레이블링의 필요성을 없애고, 실시간으로 표시된 이미지와 영상에 대한 준수를 검증할 수 있는 능력을 보유하고 있습니다.

- **Technical Details**: CompAgent 프레임워크는 정책에 따라 도구 선택과 검증 절차를 체계화하여 준수 검증을 수행합니다. 이 과정은 두 단계로 나뉘며, 첫 번째는 정책 요구사항에 따라 적절한 도구를 선택하는 것이고, 두 번째는 다중 모달(Multi-modal) 추론을 통해 준수를 검증하는 것입니다. 이 과정에서는 객체 감지(object detection), Not Safe for Work (NSFW) 감지, 이미지 캡션화 등의 다양한 도구가 활용됩니다.

- **Performance Highlights**: CompAgent는 공개 데이터셋에서 실시한 실험을 통해 기존의 전문 분류기(specialized classifiers)와 MLLM 접근 방식보다 뛰어난 성능을 보였습니다. UnsafeBench 데이터셋에서 최대 76%의 F1 점수와 함께 최신 기술 대비 10%의 성능 향상을 달성하여 준수 검증의 정확성과 확장성을 입증하였습니다. 이는 에이전트의 계획 및 도구 보강 추론(tool-augmented reasoning)이 효과적임을 보여줍니다.



### BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Editing (https://arxiv.org/abs/2511.00143)
Comments:
          36 pages; NeurIPS 2025; Code is available at this https URL

- **What's New**: 최근의 텍스트-이미지(text-to-image) 모델의 발전은 이미지 편집 기술의 광범위한 사용을 가능하게 하였지만, 이러한 기술이 악용될 가능성에 대한 우려도 커지고 있습니다. 이에 대한 연구가 진행되고 있으며, 이미지 공개 전에 '보호' 적 대항 소음(adversarial noise)을 삽입하여 이러한 위협을 예방하는 방향으로 나아가고 있습니다.

- **Technical Details**: 본 논문에서는 이미지 보호를 위한 대항 소음이 단순히 인식되지 않는 것뿐만 아니라, 원본 이미지를 숨겨놓은 상태에서 소음으로 간주되기 어려워야 한다고 주장합니다. 우리는 적응형 가우시안 블러(blur)를 사용하여 소음을 조정하고 전체 주파수 스펙트럼을 개선하는 간단한 방법을 제안하며, 이 방식을 통해 이미지 보호의 강 robustness를 높이고자 합니다.

- **Performance Highlights**: 제안된 방법은 기존 방법에 비해 악성 이미지 편집을 위한 다양한 소음 제거 기술에 대해 더 높은 보호 성능을 보이며, 품질 저하를 줄이는 데도 효과적입니다. 실험 결과, 우리 방법이 모든 테스트 후에도 93%의 효과를 유지할 수 있었던 반면, 이전의 최상의 방법은 48%에 불과했음을 보여주었습니다.



### FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding (https://arxiv.org/abs/2511.00141)
- **What's New**: 이번 논문에서는 FLoC라는 비주얼 토큰 압축 프레임워크를 제안합니다. 이 프레임워크는 주어진 비주얼 토큰 수를 기반으로, 압축이 용이하면서도 다채로운 비주얼 토큰의 하위 집합을 신속하게 선택합니다. 특히, 이 방법은 모델에 의존하지 않으며, 질의(query)나 특정 모델에 구애받지 않고 다양한 비디오-LLM과 통합하여 사용할 수 있다는 장점이 있습니다.

- **Technical Details**: FLoC는 시설 위치 함수(facility location function)를 기반으로 하여 시각적 토큰 선택을 서브모듈 최적화(submodular optimization) 문제로 해석합니다. 이를 통해 최소한의 계산 오버헤드로 비주얼 토큰을 선택할 수 있으며, 각각의 하위 집합은 전체 토큰과의 유사성을 고려하여 설계됩니다. 이 과정에서 사용되는 지연 탐욕 알고리즘(lazy greedy algorithm)은 토큰 선택의 효율성을 크게 향상시키며, 종합적으로 비주얼 이해 작업의 필수 정보를 효과적으로 보존할 수 있게 합니다.

- **Performance Highlights**: 대규모 벤치마크인 Video-MME, MLVU, LongVideoBench에 대한 광범위한 평가 결과, FLoC는 기존의 압축 기법을 지속적으로 초월했습니다. 이 연구는 특히 긴 비디오 이해의 주요 도전을 다루는 데 있어 효과성과 강력함을 강조하며, 처리 속도에서도 효율성을 보여줍니다. 이러한 결과는 FLoC가 제공하는 높은 성능과 다양한 응용 가능성을 보여줍니다.



### Integrating ConvNeXt and Vision Transformers for Enhancing Facial Age Estimation (https://arxiv.org/abs/2511.00123)
- **What's New**: 이번 연구에서는 얼굴 이미지에서 나이를 추정하는 새로운 하이브리드 아키텍처를 제안합니다. 이것은 최첨단 컨볼루션 신경망(ConvNeXt)과 비전 트랜스포머(ViT)의 결합으로 구성되어 있습니다. 각 모델이 독립적으로 우수한 성능을 내지만, 이들의 통합은 CNN의 로컬 특성 추출 기능과 트랜스포머의 글로벌 주의 메커니즘의 상호 보완적인 강점을 활용합니다.

- **Technical Details**: 제안된 ConvNeXt-ViT 하이브리드 솔루션은 MORPH II, CACD 및 AFAD와 같은 벤치마크 데이터셋에서 철저히 평가되었습니다. 이 모델은 선행 학습된 모델을 활용하고, 다양한 구성과 고급 정규화 기술을 사용하여 아키텍처를 최적화합니다. 각각의 구성 요소와 훈련 전략의 중요성을 강조하는 철저한 제어 연구( 도기도칭 )를 통해 나이에 중요한 얼굴 특징에 대한 주의 메커니즘의 조정된 중요성이 드러났습니다.

- **Performance Highlights**: 이 연구는 ConvNeXt-ViT 하이브리드가 전통적인 방법보다 우수한 성능을 발휘하며, 나이 추정 및 관련 시각 작업에서 향후 발전을 위한 강력한 기초를 제공한다고 강조합니다. 평가 결과는 MORPH II에서 평균 절대 오차 (MAE) 2.26세, CACD에서 4.35세, IMDB-Clean에서 4.2세, AFAD에서 3.09세를 기록하여 경쟁력 있는 성능을 보여줍니다.



### VLM6D: VLM based 6Dof Pose Estimation based on RGB-D Images (https://arxiv.org/abs/2511.00120)
Comments:
          This paper has been accepted to IEIE( The Institute Of Electronics and Information Engineering, South Korea) Fall,2025 Conference

- **What's New**: 이 논문에서는 VLM6D라는 새로운 이중 스트림 아키텍처를 제안합니다. 이는 RGB-D 입력으로부터의 시각적 및 기하학적 데이터의 고유한 강점을 활용하여 6D 객체의 포즈 추정을 더욱 견고하고 정밀하게 수행합니다. 기존의 접근 방식들이 합성 데이터에서 실제 상황으로의 일반화에 어려움을 겪는 반면, VLM6D는 이러한 한계를 극복하고자 설계되었습니다.

- **Technical Details**: VLM6D는 두 개의 전문 인코더를 통합하고 있습니다. 강력한 자가 감독 기반의 Vision Transformer(DINOv2)는 RGB 모달리티를 처리하며, 비주얼 문법에 대한 깊은 이해를 통해 질감 및 조명 변화에 대해 뛰어난 내성을 자랑합니다. 동시에, PointNet++ 인코더는 깊이 데이터에서 파생된 3D 점 구름을 처리하며 심각한 가림 현상이 있는 경우에도 고급 기하학적 추론을 가능하게 합니다.

- **Performance Highlights**: 포괄적인 실험을 통해 VLM6D는 Occluded-LineMOD에서 새로운 SOTA(State of the Art) 성능을 달성했음을 입증하였습니다. 이는 VLM6D의 뛰어난 견고성과 정확성을 뒷받침하며, 현실 세계의 다양한 상황에서 효과적인 6D 포즈 추정이 가능함을 보여줍니다.



### End-to-End Framework Integrating Generative AI and Deep Reinforcement Learning for Autonomous Ultrasound Scanning (https://arxiv.org/abs/2511.00114)
- **What's New**: 이 논문에서는 심장 초음파(US) 스캐닝을 자동화하고 재현 가능하게 하기 위한 최초의 종단 간(end-to-end) 프레임워크를 제시합니다. 이 프레임워크는 생성형 AI와 딥 강화 학습(DRL)을 통합하여 교육을 통해 심장 US 환경을 모델링하는 시뮬레이터와 자율 스캐닝 정책을 학습하는 두 가지 주요 구성 요소로 이루어져 있습니다. 이 기술은 심장 건강 평가에 있어 일관된 접근을 제공하며, 특히 제한된 전문 인력으로 인한 접근성을 개선하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: 제안된_Framework에서는 Generative Adversarial Networks (GANs)와 Variational Autoencoders (VAEs)를 결합한 조건부 생성 시뮬레이터가 사용됩니다. 이 시뮬레이터는 심장 US 환경을 모델링하여 현실적인 행동 조건 이미지(action-conditioned images)를 생성합니다. 또 다른 구성 요소는 DRL 모듈로, 이 시뮬레이터를 활용하여 자율적이고 정확한 스캐닝 정책을 학습합니다. 이러한 접근 방식은 기존의 데이터 부족 문제와 간단화된 모델의 한계를 극복하는 데 중점을 둡니다.

- **Performance Highlights**: 제안된 시스템은 여러 실험을 통해 검증되었으며, VAE-GAN의 성능은 기존 GAN 변종과 비교하여 정성적 및 정량적으로 평가되었습니다. DRL 기반 스캐닝 시스템은 다양한 구성 하에서 평가되어 효과성을 입증했습니다. 시스템의 설계는 카디오그램의 질에 대한 지속적인 평가를 포함하여 이미지 분류 및 품질 평가를 위한 AI 기반 가이드를 제공합니다.



### Chain of Time: In-Context Physical Simulation with Image Generation Models (https://arxiv.org/abs/2511.00110)
- **What's New**: 본 논문에서는 비전-언어 모델에서 물리 시뮬레이션을 개선하고 해석하기 위한 새로운 방법인 'Chain of Time'을 제안합니다. 이 방법은 머신 러닝의 인컨텍스트 추론(in-context reasoning) 및 인간의 정신적 시뮬레이션에 영감을 받아 개발되었습니다. 특히, Chain of Time은 추가적인 파인튜닝 없이 추론 시점에서 사용되며, 합성 및 실제 도메인에서 적용됩니다.

- **Technical Details**: Chain of Time 방법은 입력 이미지의 시뮬레이션 과정에서 생성된 일련의 중간 이미지를 통해 물리적 특성(velocity, acceleration, fluid dynamics, 그리고 conservation of momentum)을 평가합니다. 이 방법은 상태-of-the-art 이미지 생성 모델의 성능을 상당히 향상시키는 것으로 나타났으며, 이미지 모델이 각 시간 단계에서 시뮬레이션하는 세계의 특정 상태를 분석하여 전통적인 물리적 추론 평가에서 숨겨진 통찰력을 제공하고 있습니다.

- **Performance Highlights**: Chain of Time을 적용함으로써 대상 IGM의 물리 추론 능력을 개선하였으며, 다양한 특정 메트릭에서 더 정확한 이미지를 생성할 수 있게 되었습니다. 또한, 이 성장 과정에서 모델이 성공하는 측면과 어려움을 겪는 측면을 분석하여, 물리적 세계를 시뮬레이션하는 과정에 대한 새롭고 자세한 통찰을 제공합니다.



### AI Powered High Quality Text to Video Generation with Enhanced Temporal Consistency (https://arxiv.org/abs/2511.00107)
- **What's New**: MOVAI(다중 모드 원본 비디오 AI)는 텍스트로부터 비디오를 생성하기 위한 새로운 계층적 프레임워크로, 구성 장면 이해(compositional scene understanding)와 시간 인지(diffusion models)를 통합하여 고충실도의 비디오 합성을 가능하게 합니다. 이 프레임워크는 텍스트 설명을 계층적 장면 그래프로 분해하는 Compositional Scene Parser(CSP), 프레임 간의 일관성 있는 움직임을 보장하는 Temporal-Spatial Attention Mechanism(TSAM), 비디오 품질을 점진적으로 향상시키는 Progressive Video Refinement(PVR) 모듈을 포함한 세 가지 주요 혁신을 소개합니다.

- **Technical Details**: MOVAI는 입력 텍스트 TT와 선택적 조건 입력(스타일, 기간, 해상도)을 바탕으로 비디오 시퀀스 V를 생성하는 조건부 생성 문제로 설정됩니다. 전체 시스템은 세 가지 상호 연결된 모듈로 구성되며, 입력 처리(Input Processing), 장면 이해(Scene Understanding), 주의 처리(Attention Processing), 비디오 생성(Video Generation) 단계로 이루어져 있습니다. 각 단계에서는 고유한 기술적 접근 방식을 통해 장면의 객체 및 관계를 세밀하게 모델링하며, 모든 프레임 간의 시간적 일관성을 유지합니다.

- **Performance Highlights**: MOVAI는 다양한 평가 메트릭에서 기존 방법에 비해 15.3%의 LPIPS, 12.7%의 FVD 및 18.9%의 사용자 선호도 개선을 포함하여 최첨단 성능을 입증했습니다. 특히, 복잡한 다중 객체 장면을 생성할 때 강점을 보이며, 사용자들로부터 더 일관되게 선호되는 결과를 제공합니다. 이는 단순히 무엇이 비디오에 나타나야 할지뿐만 아니라, 객체가 시간에 따라 어떻게 이동하고 상호작용해야 하는지에 대한 훨씬 더 나은 제어를 가능하게 합니다.



### FreeSliders: Training-Free, Modality-Agnostic Concept Sliders for Fine-Grained Diffusion Control in Images, Audio, and Video (https://arxiv.org/abs/2511.00103)
- **What's New**: 이 연구는 FreeSliders라는 새로운 접근 방식을 소개합니다. 이는 기존의 Concept Sliders(CS)의 필요조건 없이 훈련이 필요 없는 방식으로, 다양한 모달리티에 적합한 제어를 제공합니다. FreeSliders는 부분적으로 CS 공식을 추론 중에 추정함으로써 아키텍처에 종속되지 않고 유연한 컨셉 제어를 가능하게 합니다.

- **Technical Details**: FreeSliders는 기존의 CS 방식과 비교하여 개별 컨셉의 훈련 없이 동작하며, 이미지, 비디오, 오디오 등 모든 모달리티에 적용 가능합니다. 연구진은 CS 벤치마크를 확장하여 다양한 모달리티에서 섬세한 컨셉 편집을 위한 첫 번째 기준점을 설정하고, 새로운 평가 지표를 도입하여 평가 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과, FreeSliders는 훈련 없이도 다양한 모달리티 간의 컨셉 제어를 가능하게 하여 기존의 기준들보다 개선된 성능을 보였습니다. 또한, 자동 포화 감지 및 탐색 제공 방식을 통해 비선형 탐색 문제를 해결하고, 감각적으로 일관된 변화를 실현함으로써 보다 직관적인 사용자 경험을 제공합니다.



### A filtering scheme for confocal laser endomicroscopy (CLE)-video sequences for self-supervised learning (https://arxiv.org/abs/2511.00098)
- **What's New**: 이 연구에서는 Self-Supervised Learning (SSL) 방식이 Confocal Laser Endomicroscopy (CLE) 이미지의 프리트레이닝에 효과적이라는 것을 최초로 보여줍니다. 또한, CLE 비디오 시퀀스를 위한 새로운 데이터 필터링 방법을 제안하여 프리트레이닝 과정의 계산 시간을 단축하면서 성능 저하 없이 진행할 수 있습니다. 실험 결과, 제안된 SSL 사전 훈련 모델이 전통적인 비선형 전이 학습 방법을 초월하는 향상된 정확도를 보였습니다.

- **Technical Details**: CLE는 비침습적(real-time imaging)이며, 이 연구는 대규모 레이블이 없는 데이터 세트를 사용하여 SSL 모델을 프리트레인합니다. 비디오 시퀀스의 높은 프레임 간 상관관계를 활용하여 CLE-ViFi라는 비디오 필터링 알고리즘을 통해 데이터 중복성을 줄입니다. 모델 훈련에는 작은 구성의 ViT-small(vision transformer small backbone)을 사용하며, AdamW 옵티마이저로 훈련합니다.

- **Performance Highlights**: 이 연구는 SNT( sinonasal tumors) 및 SCCS(squamous cell carcinoma of the skin) 데이터 세트에서 제안한 SSL 프리트레인 모델이 각각 67.48% 및 73.52%의_test accuracy_를 기록하며 기존 비 SSL 기준 모델을 상당히 초과했다고 보고했습니다. 또한, 우리의 접근 방식은 전체 훈련 시간을 67% 감소시키면서 훈련 효율성을 극대화하였습니다.



### SpinalSAM-R1: A Vision-Language Multimodal Interactive System for Spine CT Segmentation (https://arxiv.org/abs/2511.00095)
Comments:
          2 Tables,5 Figures,16 Equations

- **What's New**: 이번 연구에서는 SpinalSAM-R1이라는 새로운 다중 모드 비전-언어 인터랙티브 시스템을 제안합니다. 이 시스템은 세그먼트 아무거나 모델(Segment Anything Model, SAM)과 DeepSeek-R1을 통합해 척추 CT 이미지 세분화 작업을 수행합니다. 새로운 해부학 유도 주의 메커니즘과 자연어 기반의 세분화 정제를 가능하게 하는 세멘틱스 기반 상호작용 프로토콜을 소개합니다. 이를 통해 복잡한 척추 구조의 세분화 성능을 개선하고, 효율적인 적응을 위해 Low-Rank Adaptation(LoRA)을 사용하여 미세 조정하였습니다.

- **Technical Details**: SpinalSAM-R1은 세 가지 주요 구성 요소로 구성된 3계층 아키텍처를 갖추고 있습니다. 사용자 인터페이스 계층은 포인트, 바운딩 박스 및 자연어 명령을 지원하여 직관적인 임상 상호작용을 제공합니다. 비즈니스 로직 계층은 DeepSeek-R1 모듈을 통합하여 의미론적 지침을 세분화 프롬프트로 동적으로 해석하고 실시간으로 컨텍스트 인식 마스크 정제를 제공합니다. 이러한 구조는 고속 의료 이미지 처리의 요구를 충족하기 위해 GPU 가속 계산을 활용합니다.

- **Performance Highlights**: SpinalSAM-R1은 120개의 요추 CT 스캔으로 구성된 임상 데이터 세트에서 엄격하게 평가되었으며, 주의계수(Dice coefficient)는 0.9532, 교차 면적 비율(IoU)은 0.9114에 도달하여 U-Net, TransUNet, SAM-Med2D와 같은 최신 방법들을 초월하는 성능을 보였습니다. 또한 DeepSeek-R1 모듈은 11개의 임상 작업 유형에 대해 94.3%의 명령 구문 정확도를 달성하고 800ms 이하의 응답 시간을 유지하여 사용자 편의성과 상호작용의 효율성을 극대화했습니다.



### Self-Improving Vision-Language-Action Models with Data Generation via Residual RL (https://arxiv.org/abs/2511.00091)
Comments:
          26 pages

- **What's New**: 이번 논문에서는 Supervised fine-tuning (SFT)을 통한 대형 vision-language-action (VLA) 모델 개선의 한계를 극복하기 위해, Probe, Learn, Distill (PLD)라는 새로운 세 단계 프레임워크를 제안합니다. SFT는 비용이 많이 드는 인간 시연에 의존함으로 인해 확장성과 일반화에 한계가 있었으나, PLD는 잔여 강화 학습(Residual RL) 및 분포 인식 데이터 수집을 통해 이를 해결합니다. 이 프레임워크는 VLA의 성과를 극대화하는 방향으로 설계되었습니다.

- **Technical Details**: PLD의 첫 번째 단계에서는 경량 잔여 액터(actors)를 훈련하여 VLA 일반 모델의 실패 지역을 조사하고, 두 번째 단계에서는 수집된 경로들이 일반 모델의 배포 분포와 일치하도록 하여 데이터 수집의 자동화를 진행합니다. 마지막 단계에서 SFT를 통해 수집된 데이터를 다시 메인 모델로 증류(distill)하여 성능을 더욱 향상시킵니다. 각 단계는 VLA 아키텍처와 독립적으로 작동 가능하여, 다양한 제어 메커니즘을 지원합니다.

- **Performance Highlights**: PLD를 적용한 결과 LIBERO 벤치마크에서 99%의 과제 성공률에 도달하였고, SimplerEnv에서는 50% 이상의 성과 향상이 있었습니다. 실세계에서의 Franka와 YAM 팔 조작 과제에서도 100% 성공률을 달성하였습니다. 이 연구는 적은 인간의 노력을 통해 VLA 모델이 스스로 개선될 수 있는 가능성을 보여주며, 다양한 새로운 과제와 환경으로의 일반화 능력을 겨냥하고 있습니다.



### LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation (https://arxiv.org/abs/2511.00090)
Comments:
          NeurIPS 2025

- **What's New**: LeMiCa는 훈련이 필요 없는 효율적인 확산 기반 비디오 생성 가속 프레임워크를 제시합니다. 기존의 캐싱 전략이 부분적인 휴리스틱 오류 감소에 중점을 둔 반면, LeMiCa는 전역 오류 누적을 방지하여 생성된 비디오의 전반적인 일관성을 크게 향상시킵니다. 이 접근법은 전역 콘텐츠와 스타일의 일관성을 개선할 수 있으며, 다양한 테스팅 벤치마크에서 뛰어난 성능을 입증했습니다.

- **Technical Details**: LeMiCa는 캐시 스케줄링을 오류 가중 경로가 있는 방향 그래프로 형식화하고, 최악의 경로 오류를 명시적으로 제한하는 Lexicographic Minimax Path Optimization 전략을 도입합니다. 이 메트릭을 바탕으로, LeMiCa는 Directed Acyclic Graph(DAG)를 구성하여 각 엣지가 최종 출력 품질에 미치는 영향을 반영합니다. 이 방식은 로컬 그리디 전략 대신 전체 경로 계획 문제로 접근하여, 전역 오류를 효율적으로 관리합니다.

- **Performance Highlights**: LeMiCa는 Latte 모델에서 2.9배의 속도 향상과 Open-Sora에서 0.05의 LPIPS 점수를 기록하며, 기존 캐싱 기법보다 우수한 성능을 보입니다. 이러한 개선 사항은 인지 품질 손실을 최소화하면서 이루어지며, 비디오 생성의 효율적이고 신뢰할 수 있는 기초로 자리잡을 수 있습니다. 여러 기본 모델에서 인퍼런스 속도와 생성 품질의 이중 개선을 달성했습니다.



### Habitat and Land Cover Change Detection in Alpine Protected Areas: A Comparison of AI Architectures (https://arxiv.org/abs/2511.00073)
- **What's New**: 이 논문은 알프스 국가공원에서의 긴 기간의 생태계 데이터를 활용하여 복잡한 자연환경에서의 변화를 탐지하기 위해 딥러닝을 적용하는 새로운 접근을 제시합니다. 특히, 지리 공간 기초 모델(GFM)을 이용한 변별력이 높은 변화 탐지 기술을 개발하여 기존의 수작업 맵핑 방식의 한계를 극복하고자 합니다. 이 연구는 변경 감지 방법의 두 가지 패러다임, 즉 사후 분류(change detection, CD)와 직접적 CD를 비교하고 최적의 방법을 탐색합니다.

- **Technical Details**: 연구는 U-Net CNN과 변화 감지 맞춤형 변환기(ChangeViT)를 포함하여 여러 최신 지리 공간 AI 아키텍처를 평가합니다. 사용된 데이터셋은 알프스 지역에서 4,480건의 필드 변화를 문서화한 다중 모드 고해상도 이미지로 구성되어 있으며, RGB, NIR, LiDAR 데이터를 포함합니다. 벤치마킹 결과, Clay v1.0 모델이 U-Net보다 다중 클래스 감지에서 51%의 정확도를 달성했으며, 이 두 모델 모두 이진 변경 탐지에서 67%에 도달했습니다.

- **Performance Highlights**: 연구결과, LiDAR 데이터를 통합함으로써 의미론적 세분화의 정확도가 30%에서 50%로 향상됨을 확인했습니다. 또한, GFM의 견고성도 평가되었으며, 2020년 데이터에 대한 Clay 모델의 정확도가 33%로 U-Net의 23%를 초과했습니다. 그러나 알프스의 복잡한 생태계에서는 전체 정확도가 동질적인 경관보다 낮지만, 이는 이러한 자연 환경에서의 현실적인 성능을 반영합니다.



### World Simulation with Video Foundation Models for Physical AI (https://arxiv.org/abs/2511.00062)
- **What's New**: [Cosmos-Predict2.5]는 물리 AI를 위한 Cosmos World Foundation Models의 최신 버전으로, 텍스트, 이미지 및 비디오 생성을 단일 모델에서 통합한 흐름 기반(Flow-based) 아키텍처를 기반으로 합니다. 이 모델은 더욱 풍부한 텍스트 그라운딩(text grounding)과 세계 시뮬레이션(world simulation)의 세밀한 제어를 가능하게 합니다. 새로운 버전은 2B 및 14B 스케일에서 출시되어 이전 모델인 [Cosmos-Predict1]보다 비디오 품질과 지침 일치(instruction alignment)에서 현저한 개선을 이루었습니다.

- **Technical Details**: [Cosmos-Predict2.5]는 2억 개의 큐레이션된 비디오 클립을 기반으로 학습되었으며, 강화 학습(reinforcement learning) 기반의 후학습(post-training) 과정으로 다듬어졌습니다. 또한, 시뮬레이션 및 정책 평가(policy evaluation)을 위한 신뢰성 높은 합성 데이터 생성(synthetic data generation)을 지원합니다. [Cosmos-Transfer2.5]는 Sim2Real 및 Real2Real 세계 번역을 위한 제어망(control-net) 스타일의 프레임워크로, 작지만 높은 충실도와 강력한 장기 비디오 생성 기능을 갖추고 있습니다.

- **Performance Highlights**: [Cosmos-Predict2.5]와 [Cosmos-Transfer2.5]는 엔터티 지능의 확장을 위한 다재다능한 도구로 자리 잡고 있습니다. 이 모델들은 비록 [Cosmos-Transfer1]보다 3.5배 작지만, 더욱 높은 fidelity와 장기 비디오 생성에서 더 나은 성능을 제공합니다. NVIDIA Open Model License하에 소스 코드, 사전 훈련 체크포인트(pretrained checkpoints) 및 큐레이팅 된 베치마크를 공개하여 물리 AI 연구 및 배포를 촉진하고 있습니다.



### Which LiDAR scanning pattern is better for roadside perception: Repetitive or Non-repetitive? (https://arxiv.org/abs/2511.00060)
- **What's New**: 이번 연구에서는 LiDAR 기반 도로 측정 기술의 중요한 변화를 제안합니다. LiDAR의 스캐닝 패턴에 따른 성능 차이를 체계적으로 조사하기 위해 'InfraLiDARs' Benchmark'라는 새 데이터셋을 CARLA 시뮬레이션 환경에서 수집하였습니다. 이 데이터셋을 사용하여 기존 3D 객체 탐지 알고리즘의 성능에 미치는 영향을 분석하고 비교하였습니다.

- **Technical Details**: 연구의 초점은 전통적인 반복 스캐닝 방식의 LiDAR와 최근의 비반복 스캐닝 방식 간의 차이를 평가하는 것입니다. LiDAR 배치는 이미 충분히 연구되었으나, 스캐닝 패턴에 따른 객관적 성능 비교는 충분히 다뤄지지 않았습니다. 따라서, 본 연구는 다양한 스캐닝 방식의 LiDAR가 환경 정보를 얼마나 효과적으로 인식하는지 평가하며, CARLA 시뮬레이션을 통해 다양한 교통 시나리오에서 분석을 진행합니다.

- **Performance Highlights**: 연구 결과에 따르면, 비반복 LiDAR와 128라인 반복 LiDAR는 여러 시나리오에서 비슷한 탐지 성능을 보였습니다. 비록 비반복 LiDAR의 인식 범위는 제한적이지만, 비용 효율성이 높은 옵션으로서 인식 시스템 구성에 유용할 수 있음을 보여줍니다. 최종적으로, 본 연구는 도로 측정 시스템을 위한 최적의 LiDAR 스캐닝 패턴 및 알고리즘 선택에 대한 통찰력을 제공합니다.



### Enhancing rice leaf images: An overview of image denoising techniques (https://arxiv.org/abs/2511.00046)
Comments:
          18 pages, 6 figures. Research Article published in the International Journal of Agricultural and Natural Sciences (IJANS), Vol. 18, Issue 2, 2025. This paper presents a comparative study of image denoising and CLAHE techniques for enhancing rice leaf images corrupted by Gaussian, Salt-and-pepper, Speckle, and Random noise for agricultural analysis

- **What's New**: 이 논문에서는 디지털 이미지 처리에 대한 체계적인 접근 방식이 소개되며, 특히 이미지 향상(image enhancement)의 중요성이 강조됩니다. 기존의 이미지 노이즈 제거(noise reduction) 방법과 함께 CLAHE(Contrast Limited Adaptive Histogram Equalization)를 결합한 새로운 비교 연구가 수행되었습니다. 이 연구는 쌀 잎 이미지에 대한 효과적인 노이즈 제거 기법을 제안하여 농업 연구에 응용될 수 있는 통찰력을 제공합니다.

- **Technical Details**: 연구에서는 이미지 처리를 위한 고급 컴퓨터 알고리즘을 활용하여 쌀 잎 이미지의 품질을 향상시키는 방법을 다룹니다. 이미지 향상은 노이즈 제거 후 대조(contrast) 향상을 포함하며, 이를 위해 다양한 이미지 필터(image filters)가 사용됩니다. 이러한 필터는 밝기(brightness), 대조(contrast), 선명도(sharpness)와 같은 시각적 특성을 변형하거나 개선하여 전체 이미지 품질을 향상시키고 유용한 정보를 추출하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 연구 결과는 다양한 성능 메트릭(metrics)을 통해 검토되어, 향상된 방법의 효과를 포괄적으로 평가합니다. 이 접근 방식은 디지털 이미지 처리 방법론의 효과를 평가하는 강력한 기초를 제공하며, 향후 농업 연구 및 기타 분야에 대한 적응 가능성을 제시합니다. 결과로 나타난 노이즈 제거 및 대조 향상 방법은 실제 이미지 처리 작업에 실질적인 기여를 할 것으로 기대됩니다.



### Benchmarking Federated Learning Frameworks for Medical Imaging Deployment: A Comparative Study of NVIDIA FLARE, Flower, and Owkin Substra (https://arxiv.org/abs/2511.00037)
- **What's New**: 이번 연구에서는 Federated Learning (FL)이라는 혁신적인 패러다임을 통해 의료 AI 분야에서 데이터 공유 없이 협력적인 모델 훈련을 가능하게 하는 세 가지 주요 FL 프레임워크인 NVIDIA FLARE, Flower, Owkin Substra를 평가합니다. 본 논문은 PathMNIST 데이터셋을 사용하여 실제 의료 이미징 애플리케이션에 대해 각 프레임워크의 적합성을 검토합니다.

- **Technical Details**: 각 프레임워크는 모델 성능, 수렴 효율성(convergence efficiency), 통신 오버헤드(communication overhead), 확장성(scalability), 개발자 경험(developer experience) 등을 평가하기 위해 테스트되었습니다. NVIDIA FLARE는 뛰어난 생산 확장성을 제공하며, Flower는 프로토타입 개발(prototyping) 및 학술 연구에 유연성을 제공합니다. Owkin Substra는 특히 개인정보 보호(privacy) 및 준수(compliance) 기능에서 뛰어남을 보여줍니다.

- **Performance Highlights**: 연구 결과는 각 프레임워크가 서로 다른 사용 사례에 최적화된 강점을 가지고 있음을 강조합니다. 이는 의료 환경에서의 실제 배포에 대한 중요성을 보여주며, NVIDIA FLARE는 확장성 측면에서 우수한 성능을 제공하고, Flower는 연구의 유연성을 강조하며, Owkin Substra는 높은 수준의 보안 특성을 나타냅니다.



### Mutual Information guided Visual Contrastive Learning (https://arxiv.org/abs/2511.00028)
Comments:
          Tech Report - Undergraduate Thesis - 2023

- **What's New**: 이번 연구에서는 상호 정보(mutual information)를 기반으로 하는 데이터 선택(data selection) 및 증강(data augmentation) 방법을 탐구합니다. 기존의 데이터 증강 기법들은 주로 색상 변형(color jittering)에 집중하여 실제 조명 변화(real-world illumination changes)를 모방하는 데 초점을 맞추었습니다. 하지만 본 연구는 각 장면(scene)의 패치를 분석하여 상호 정보가 높은 패치를 긍정 샘플로 선택하는 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: 이 연구는 InfoAug라는 새로운 데이터 증강 기술을 소개합니다. InfoAug는 각 패치의 상호 정보와 다른 패치 간의 관계를 분석하여 강한 상관관계를 보이는 최고의 패치를 긍정 샘플로 선정합니다. 특히, 이 연구에서는 비디오 데이터셋을 활용하여 첫 번째 프레임의 패치들에 대한 상호 정보를 추정하며, 모든 패치 간의 상관관계를 종합적으로 고려하여 긍정 샘플을 결정하는 방법론을 개발했습니다.

- **Performance Highlights**: 제안된 방법은 CIFAR-10, CIFAR-100, STL-10 등 여러 벤치마크에서 성능 평가를 실시하였으며, 결과적으로 기존의 기법들에 비해 일관되게 향상된 성능을 보였습니다. 이를 통해 우리의 접근법이 실제 모델과 인간의 시각적 학습에 더 부합하는 긍정 샘플 결정 방법임을 입증하고 있습니다. 또한, 기존의 대비 학습 프레임워크와 비교하여 앞선 성과를 보이며, 향후 연구를 위한 유망한 방향을 제시하고 있습니다.



### Automating Coral Reef Fish Family Identification on Video Transects Using a YOLOv8-Based Deep Learning Pipelin (https://arxiv.org/abs/2511.00022)
Comments:
          Accepted to EUVIP2025, student session

- **What's New**: 이번 연구는 서부 인도양에서 산호초 모니터링을 자동화하기 위해 YOLOv8 기반의 딥러닝 파이프라인을 평가했다. 케냐와 탄자니아에서 수집된 비디오 트랜섹트를 사용하여 종 가족(genre) 수준의 물고기 식별을 수행한다. 24개 가족으로 구성된 데이터셋을 다양한 설정에서 시험하여 서부 인도양의 자동화된 산호초 어류 모니터링에 대한 최초의 지역별 기준 선을 제시하였다.

- **Technical Details**: YOLOv8 모델을 활용하여 비디오 트랜섹트에서 물고기를 식별하는 딥러닝 접근 방식을 사용하였다. 이 연구에서 테스트한 다양한 구성들은 물고기의 가족 수준에서의 식별을 가능하게 하며, 특히 자주 발견되는 가족에 대해서는 높은 정확도를 보여준다. 그러나 드물거나 복잡한 과(科, taxonomy)에 대해서는 약한 탐지 능력을 보였다.

- **Performance Highlights**: 베스트 모델은 mAP@0.5에서 0.52를 달성하여 상당한 퍼포먼스를 기록했다. 이는 전통적인 모니터링 방법에 비해 딥러닝이 확장 가능한 보완책이 될 수 있음을 증명하는 결과이다. 연구 결과는 서부 인도양 지역에서 산호초 어류 모니터링의 효율성을 향상시킬 수 있는 가능성을 제시한다.



### Deep Learning Models for Coral Bleaching Classification in Multi-Condition Underwater Image Datasets (https://arxiv.org/abs/2511.00021)
Comments:
          15 pages, 10 figures

- **What's New**: 이 연구는 다양한 환경 조건에서 건강한 산호와 백화산호 샘플을 포함한 글로벌 데이터셋을 기반으로 한 혁신적인 기계 학습(machine-learning) 기반 산호 백화(classification) 시스템을 제안합니다. 이는 산호의 효율적인 보호 및 모니터링을 위한 시급한 필요성에 대한 해결책을 제공합니다. 함께 비교한 최신 모델들은 Residual Neural Network (ResNet), Vision Transformer (ViT), Convolutional Neural Network (CNN)으로, 각각의 성능을 종합적으로 평가했습니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 깊은 바다(deep seas), 습지(marshes), 해안 지역(coastal zones) 등 다양한 환경에서 수집되었습니다. 하이퍼파라미터 튜닝(hyperparameter tuning)을 통해 CNN 모델이 88%의 정확도를 기록하였고, 이는 기존 벤치마크를 초과하는 성과입니다. 이러한 성과는 자동화된 산호 모니터링 및 컴퓨터 비전 모델(computer vision models) 분석에 중요한 통찰력을 제공합니다.

- **Performance Highlights**: CNN 모델은 88%의 높은 정확도로 산호 백화와 건강한 산호를 정확히 분류하는 데 성공했습니다. 다른 두 모델인 ResNet과 ViT와 비교했을 때, CNN이 가장 뛰어난 성능을 보였습니다. 본 연구 결과는 산호 생태계 보호를 위한 효율적인 방법 제공에 기여할 것입니다.



### Generative human motion mimicking through feature extraction in denoising diffusion settings (https://arxiv.org/abs/2511.00011)
- **What's New**: 최근 대형 언어 모델의 성공은 인간-AI 간의 언어적 상호작용에 새로운 물결을 일으켰습니다. 하지만 이러한 모델은 인간 상호작용의 구체성을 결여하고 있습니다. 춤은 인간 표현의 원초적 형태로서 이러한 경험을 보완할 잠재력을 가지고 있습니다.

- **Technical Details**: 이 연구는 모션 캡처(motion capture) 데이터를 기반으로 한 인터랙티브 모델을 구축하여 인간-AI 상호작용을 탐구합니다. 이 모델은 단일 인물 모션 데이터를 활용하여 incoming 움직임 시퀀스를 부분적으로 모방하고 "창의적으로" 강화합니다. 또한, 두 가지 노이즈 확산 모델, 모션 인페인팅(motion inpainting), 모션 스타일 전이(motion style transfer)의 아이디어를 결합하여 시간적으로 일관되고 선택된 움직임 참조에 반응하는 이동 표현을 생성합니다.

- **Performance Highlights**: 모델의 성공은 생성된 샘플과 테스트 세트의 특징 분포 수렴을 정량적으로 평가하여 입증됩니다. 우리의 생성은 다양한 인간 파트너와의 차이를 보이며, 동시에 실제적인 모습을 유지하는 AI와의 창의적인 춤의 첫걸음으로 평가됩니다.



### Fractional Diffusion Bridge Models (https://arxiv.org/abs/2511.01795)
Comments:
          To appear in NeurIPS 2025 proceedings. This version includes post-camera-ready revisions

- **What's New**: 본 논문에서는 Fractional Diffusion Bridge Models (FDBM)이라는 새로운 생성적 확산 브리지 모델을 소개합니다. 이 모델은 고급 비마르코프(Non-Markovian) 특성을 가진 분수 브라운 운동(Fractional Brownian motion, fBM)의 근사를 기초로 합니다. FDBM은 비마르코프적인 특성을 유지하면서도 유용한 추론을 가능하게 하며, 다양한 데이터 문제에 적용할 수 있는 유연한 프레임워크를 제공합니다.

- **Technical Details**: FDBM은 MA-fBM (Markovian Approximation of Fractional Brownian Motion)을 드라이빙 프로세스로 사용하여, 두 개의 미지의 분포 사이를 보간하는 생성적 프로세스를 학습하는 방법을 제안합니다. 이 과정에서 Hurst 지수(Hurst index)를 조정하여 길게 이완된 의존성과 변동성을 조절할 수 있습니다. 또한, 미지의 데이터 변환을 위한 원칙적인 손실 함수를 도출해내어 Schrödinger bridge 문제를 확장합니다.

- **Performance Highlights**: 실험 결과, FDBM은 단백질 구조 예측 및 비짝(pair) 이미지 변환에서 기존 브라운 운동(Brownian motion) 모델에 비해 우수한 성능을 보였습니다. 특히, 단백질 구조 예측에서는 C$_\alpha$ 원자 위치의 루트 평균 제곱 편차(RMSD)가 감소하였고, 비짝 이미지 변환에서 Fréchet Inception Distance (FID)가 개선되었습니다. 이러한 결과는 FDBM이 복잡한 데이터 구조를 더 잘 캡처할 수 있음을 나타냅니다.



### Unified Diffusion VLA: Vision-Language-Action Model via Joint Discrete Denoising Diffusion Process (https://arxiv.org/abs/2511.01718)
- **What's New**: 이번 연구에서는 Unified Diffusion VLA 모델과 Joint Discrete Denoising Diffusion Process(JD3P)를 제안했습니다. 이 모델은 이미지 생성과 행동 예측을 동시에 최적화하는 동기화된 디노이징 과정을 통해 이해, 생성 및 행동 수행의 통합을 이루려는 시도를 합니다. 또한, 기존의 함수적 분리로 인한 문제점을 해결하고 서로 다른 작업들이 내재적으로 협력할 수 있는 구조를 제공합니다.

- **Technical Details**: 모델은 통합된 토큰화된 공간과 하이브리드 주의 메커니즘을 기반으로 구축되며, 다중 모달리티를 하나의 디노이징 경로로 통합하는 공동 디퓨전 프로세스를 활용합니다. JD3P는 모든 행동 토큰이 모든 미래 이미지 토큰에게 인과적으로 주의를 기울일 수 있도록 하여, 이미지 생성을 통해 행동 예측의 방향을 제시합니다. 또한, 두 단계의 훈련 파이프라인을 통해 성능과 효율성을 최적화합니다.

- **Performance Highlights**: 제안된 방법은 CALVIN, LIBERO, SimplerEnv와 같은 벤치마크에서 최첨단 성능을 달성하며, 오토회귀 방법보다 4배 더 빠른 추론 속도를 자랑합니다. 심층 분석을 통해 모델의 효과성을 검증하였으며 실제 환경 평가를 통해 실용성을 입증했습니다. 이러한 결과는 모델의 상호작용 및 통합적인 작용의 중요성을 강조합니다.



### MARS: Multi-Agent Robotic System with Multimodal Large Language Models for Assistive Intelligenc (https://arxiv.org/abs/2511.01594)
Comments:
          3 figures, 1 table; under review at Multimedia Systems (Springer)

- **What's New**: 이번 연구에서는 MLLMs(Multimodal Large Language Models)를 기반으로 한 MARS라는 다중 에이전트 로봇 시스템을 소개합니다. 이 시스템은 장애인을 지원하는 스마트 홈 로봇을 위해 설계되었으며, 시각 인식 에이전트, 위험 평가 에이전트, 계획 에이전트 및 평가 에이전트를 통합하고 있습니다. 이를 통해 환경 인식과 위험 감지, 사용자 맞춤형 지원을 구현합니다.

- **Technical Details**: MARS 시스템은 다양한 센서를 통한 환경을 인식하고, 위험을 평가한 후, 안전하고 실행 가능한 동작 시퀀스를 생성하는 구조입니다. 각 에이전트는 다중 모듈의 결합을 통해 서로 상호 작용하며, MLLM을 통해 제공되는 강력한 추론 및 의사 결정 기능을 활용합니다. 이러한 접근 방식은 복잡한 실내 환경의 특성에 맞춘 맞춤형 지원을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, MARS 시스템은 기존의 최첨단 다중 모달 모델에 비해 위험 인식 계획 및 다중 에이전트 실행에 있어 뛰어난 성능을 나타냈습니다. 또한, 다양한 데이터 세트에서 평가를 수행함으로써 실제 보조 시나리오에서의 향상된 인식과 계획 성능을 입증했습니다. 이 연구는 실질적인 보조 시나리오에서 협업 AI의 잠재력을 강조하고 실세계 환경에서 MLLM 기반의 다중 에이전트 시스템 배치 방법론을 제안합니다.



### Explore More, Learn Better: Parallel MLLM Embeddings under Mutual Information Minimization (https://arxiv.org/abs/2511.01588)
- **What's New**: 이번 논문에서는 멀티모달 임베딩 학습을 위해 새로운 Parallel Decoupling Framework (PDF)를 제안합니다. PDF는 멀티모달 대형 언어 모델(MLLMs)의 독특한 steerability를 활용하여, 하나의 입력에 대해 여러 개의 병렬 경로와 임베딩을 생성할 수 있도록 설계되었습니다. 이러한 접근법은 MLLMs의 능력을 최대한 활용하면서도 정보 손실을 최소화할 수 있습니다.

- **Technical Details**: PDF는 다중 경로와 다중 출력을 위한 새로운 패러다임인 SPP(Single input, Parallel paths, Parallel outputs)를 따릅니다. 각 경로는 고유한 학습 가능한 접두사를 통해 활성화되어 MLLM의 진화된 능력을 극대화합니다. 또한, Mutual Information Minimization (MIM) 기법을 사용하여 병렬 경로 간의 통계적 종속성을 적극적으로 측정하고 처벌함으로써 임베딩의 다양성을 보장합니다.

- **Performance Highlights**: PDF는 다양한 MLLM 백본에서 효과적으로 구현되어 MMEB 벤치마크에서 일관된 성능 향상을 보여주었습니다. 예를 들어, VLM2Vec-LLaVA-1.6 모델은 +8.9%의 성능 향상을 보였고, 2B와 7B 모델에서도 각각 +4.2%와 +3.1%의 성과를 기록했습니다. 또한, 2B 모델은 전체 계산 예산의 절반만으로도 기존 기준 모델을 +2.6% 초과하는 성능을 발휘했습니다.



### Learning to Seek Evidence: A Verifiable Reasoning Agent with Causal Faithfulness Analysis (https://arxiv.org/abs/2511.01425)
Comments:
          12 pages, 3 figures. Under review at the Conference on Computer Vision and Pattern Recognition (CVPR) 2026

- **What's New**: 이번 논문에서는 의료와 같은 고위험 분야에서 AI 모델의 설명 가능성이 신뢰성 문제를 야기하는 점을 해결하기 위해, 검증 가능하고 신뢰성 있는 설명을 제공할 수 있는 인터랙티브 에이전트를 제안합니다. 이 에이전트는 진단 추론을 지원하기 위해 외부 시각적 증거를 전략적으로 탐색하는 정책을 학습하며, 강화 학습(reinforcement learning)을 통해 최적화된 성능을 보여줍니다. 실험 결과, 이 행동 기반 추론 과정이 조정된 정확도를 약 18% 개선하여 신뢰성을 향상시킴을 확인했습니다.

- **Technical Details**: 제안하는 프레임워크는 가설 및 증거 검증을 위한 통합적인 상호작용 루프를 통해 이루어집니다. 이 에이전트는 Vision-Language Model(VLM)을 활용하여 이미지와 텍스트를 동시에 처리하며, 진단 프로세스를 명확하게 추적할 수 있는 구조로 모델링합니다. 주요 동작인 Probe & Ground(P&G)는 외부 도구를 호출하여 증거를 분석하고 신뢰도 점수를 반환, 이 피드백을 통해 에이전트의 가설 상자(Hypothesis Box, H-Box)를 업데이트하는 과정을 포함합니다.

- **Performance Highlights**: 우리는 occlusion tests를 포함한 개입 평가 프로토콜을 도입하여 설명의 신뢰성을 정량적으로 검증합니다. 이 방법을 통해 에이전트가 선택한 증거가 차단되었을 때 성과가 측정 가능한 감소를 나타내며, 이는 해당 증거가 의사결정 과정에서 중요한 역할을 함을 입증합니다. 이러한 특성 덕분에, 제안된 시스템은 상업용 하드웨어에서 훈련이 가능하며, AI의 신뢰성을 높이는 새로운 기초를 제공합니다.



### Kinematify: Open-Vocabulary Synthesis of High-DoF Articulated Objects (https://arxiv.org/abs/2511.01294)
- **What's New**: 이 논문은 Kinematify라는 자동화된 프레임워크를 도입하여, 임의의 RGB 이미지나 텍스트 프롬프트에서 직접 관절이 있는 객체를 합성합니다. 기존 방법의 한계인 고자유도(DoF) 물체의 운동학적 구조를 추론하고 정적 기하학으로부터 조인트 파라미터를 추정하는 두 가지 핵심 문제를 해결합니다. 이를 위해 Kinematify는 구조 추론을 위한 MCTS 검색과 조인트 추론을 위한 기하학적 최적화를 결합하여 물리적으로 일관되고 기능적으로 유효한 설명을 생성합니다.

- **Technical Details**: Kinematify는 우선 3D 기초 모델을 사용해 분할된 메쉬를 생성한 후, 접촉 그래프를 구축하고 MCTS 목표를 통해 운동학적 트리를 추론합니다. 그 다음, DW-CAVL이라는 새로운 거리 가중치 접촉 인식 가상 연결 최적화 방법을 통해 조인트 파라미터를 추정합니다. 이 방법은 가상 운동 하에서 접촉 영역을 보존하면서 충돌을 패널티합니다.

- **Performance Highlights**: Kinematify는 합성 및 실제 환경에서 다양한 입력을 평가하여 이전 방법에 비해 레지스트레이션 및 운동학적 구조 정확성에서 향상된 결과를 보여줍니다. 이 프로세스는 물리 인식이 가능한 고자유도 관절 객체를 이미지나 텍스트에서 바로 생성할 수 있게 해 주며, 별도의 운동 데이터나 미리 정의된 관절 사전이 필요 없습니다.



### LiDAR-VGGT: Cross-Modal Coarse-to-Fine Fusion for Globally Consistent and Metric-Scale Dense Mapping (https://arxiv.org/abs/2511.01186)
- **What's New**: 본 논문에서는 LiDAR와 VGGT를 통합하여 대규모, 메트릭 정확도 및 밀도가 높은 컬러 포인트 클라우드 재구성을 제안하는 LiDAR-VGGT라는 새로운 프레임워크를 소개합니다. 이 시스템은 LiDAR의 비언매틱 깊이 측정과 VGGT의 포즈 및 깊이 예측 기능을 결합하여 원활한 3D 재구성을 제공합니다. 이를 통해 복잡한 환경에서도 안정적이고 일관된 3D 지도를 생성할 수 있는 가능성을 보여줍니다.

- **Technical Details**: LiDAR-VGGT는 2단계의 조정에서 거친 융합(modular fusion) 과정을 통해 LiDAR 관측치를 VGGT에 통합합니다. 첫 번째로, 사전 융합(pre-fusion) 모듈이 VGGT의 포즈를 초기화하고 점클라우드를 생성하며, 두 번째로 포스트 융합(post-fusion) 모듈이 카메라와 LiDAR 간의 시야(field of view) 차이로 인한 왜곡을 완화하기 위해 정규화를 적용하는 방식입니다. 이는 LiDAR의 메트릭 스케일을 VGGT의 재구성에 통합하여 고유한 3D 표현을 가능하게 합니다.

- **Performance Highlights**: LiDAR-VGGT는 여러 데이터셋에서 시험된 결과, VGGT 기반 접근방식과 기존의 LiDAR 인에르셜 비주얼 오도메트리(LIVO) 기법들과 비교하여 우수한 성능을 보여주었습니다. 이를 통해 색상이 풍부한 포인트 클라우드를 일관되게 생성하며, 3D 환경 인식에 대한 새로운 가능성을 제시합니다. 또한, 논문에서는 컬러 포인트 클라우드를 평가하기 위한 새로운 툴킷을 개발하여 오픈 소스로 공개할 계획임을 밝혔습니다.



### Few-Shot Multimodal Medical Imaging: A Theoretical Framework (https://arxiv.org/abs/2511.01140)
Comments:
          6 Pages

- **What's New**: 이 논문은 의료 이미징에 있어 데이터 접근성이 떨어지는 환경에서의 학습과 유추를 위한 통합 이론적 프레임워크를 제안합니다. 제안된 프레임워크는 샘플 효율성(sample efficiency), 불확실성 정량화(uncertainty quantification) 및 해석 가능성(interpretability) 간의 관계를 명확히 설정합니다. 이는 저자들이 훈련 목표를 조금의 데이터로 형식화하고, 안정성을 위한 정량적 지표인 설명 분산(explanation variance)을 도입함으로써 이루어졌습니다.

- **Technical Details**: 논문에서는 적은 양의 레이블이 있는 데이터 세트에서 모델을 훈련시키기 위해, PAC 학습 이론(Probably Approximately Correct learning theory)과 VC 이론(Vapnik–Chervonenkis theory)을 기반으로 합니다. 이들은 데이터 희소성이 있는 상태에서 모델의 기대 위험을 최적 값에 가깝게 맞출 수 있는 가장 작은 레이블 수를 찾는 데 중점을 둡니다. 또한, 방법론은 여러 개의 데이터 소스를 융합하여 풍부한 정보를 제공하는 멀티모달 통합(multi-modal integration)을 통한 일반화를 촉진하는 원리를 설명합니다.

- **Performance Highlights**: 제안된 이론적 프레임워크는 의료 이미징 시스템의 데이터 효율성 종료(model capacity)와 불확실성의 상호 관계를 명확히 정의함으로써, 결핍 상태에서의 진단 시스템을 안정적으로 구축하기 위한 기초를 제공합니다. 특히, 여러 가지 불확실성 정량화 기법들을 활용하여 예측 오류를 최소화하고, 의사 결정을 개선할 수 있는 방법을 모색합니다. 전체적으로 이 연구는 저자들이 구축한 모델이 임상적으로 신뢰할 수 있는 성과를 내기 위한 이론적 근거를 마련합니다.



### Fast-SmartWay: Panoramic-Free End-to-End Zero-Shot Vision-and-Language Navigation (https://arxiv.org/abs/2511.00933)
- **What's New**: 이번 연구에서는 Fast-SmartWay라는 새로운 End-to-End VLN-CE 프레임워크를 제안합니다. 이 프레임워크는 파노라마 관찰(panoramic observations)이나 웨이포인트 예측기(waypoint predictors) 없이도 자연어 지시를 기반으로 행동을 직접 예측할 수 있습니다. 이로 인해 로봇이 복잡한 환경에서 보다 빠르게 의사 결정을 할 수 있도록 돕습니다.

- **Technical Details**: Fast-SmartWay는 프론탈 RGB-D 이미지 3장을 사용하여 시각 정보와 자연어 지시를 결합해 MLLM이 직접 행동을 예측하는 구조입니다. 또한, Uncertainty-Aware Reasoning 모듈을 도입하여 경로 선택 과정에서의 불확실성을 평가하고, 로컬 최적화(local optima)를 피할 수 있도록 돕습니다. 이를 통해 로봇은 동적 환경에서도 더욱 일관성 있고 견고한 결정을 내릴 수 있습니다.

- **Performance Highlights**: 실험 결과, Fast-SmartWay는 기존 파노라마 기반 모델에 비해 각 동작 단계의 지연 시간을 크게 줄이면서도 경쟁력 있는 성과를 보였습니다. 실제 로봇 환경에서의 실험 또한 우리의 방법이 리얼 월드 제로샷(navigation) 내비게이션에서 얼마나 효과적인지를 보여주었습니다.



### Learning with Category-Equivariant Representations for Human Activity Recognition (https://arxiv.org/abs/2511.00900)
- **What's New**: 이 논문에서는 인간 활동 인식(HAR)을 위한 새로운 프레임워크인 category-equivariant representation을 도입합니다. 이 프레임워크는 시간, 스케일 및 센서 계층의 변화에 따른 신호의 변동을 포착하도록 설계되었습니다. 특히, 다양한 환경 조건에서 감지의 정확성을 높이고, 데이터의 해석성을 확보하는 데 중점을 둡니다.

- **Technical Details**: 제안된 방법은 Group×Poset 대칭 범주를 기반으로 하며, 강한 불변성 및 매개변수를 공유하는 그룹 대칭 컨볼루션을 사용합니다. 각 센서에서 RMS 정규화(게인), 축에서 크기로의 풀링(계층), 저주파 수치 변환(시간 이동)을 조합하여 compact한 피처 벡터를 만듭니다. 이러한 접근 방식은 데이터와 피처 사이의 자연 변환(natural transformation)을 통해 구조적으로 명확합니다.

- **Performance Highlights**: UCI HAR 데이터셋에서 제안된 방법은 기존 방법에 비해 약 46%의 정확도를 향상시켰습니다. 이는 다양한 테스트 환경에서도 안정성을 보여줍니다. 연구 결과는 다른 기하학적 및 분포적 접근 방식과 결합 가능성을 보여줍니다.



### LL-ViT: Edge Deployable Vision Transformers with Look Up Table Neurons (https://arxiv.org/abs/2511.00812)
Comments:
          Accepted for FPT 2025, 9 pages, conference

- **What's New**: 이번 연구에서는 LL-ViT라는 새로운 비전 트랜스포머(vision transformer) 설계를 소개합니다. 이는 룩업 테이블(look up table, LUT) 뉴런 층을 통합하여 메모리와 연산 요구를 최적화합니다. LL-ViT는 FPGA에서의 엣지(Edge) 추론에 최적화되어 있으며, 이로 인해 모델 크기와 연산 비용을 줄이고, 95.5%의 CIFAR-10 정확도를 기록합니다.

- **Technical Details**: LL-ViT는 기존 멀티 헤드 셀프 어텐션(multi-head self-attention) 및 MLP 구조 대신 LUT 기반 채널 믹서를 도입합니다. 이는 계산이 필요 없는 룩업 전용 뉴런들을 사용하여 에너지 효율성을 높이고, 메모리 소모를 최소화합니다. 연구에서는 FPGA 가속기를 설계하여 LL-ViT의 성능을 최적화하였습니다.

- **Performance Highlights**: LL-ViT는 모델 가중치의 60% 이상 및 곱셈의 50%를 줄이며, 1.9배의 에너지 효율성과 1.3배 낮은 지연시간을 달성합니다. 또한, 10.9W의 전력 소모로 1083 FPS의 높은 처리량을 제공합니다. 이러한 성과는 LL-ViT를 엣지 추론에 적합한 모델로 만들어 줍니다.



### EraseFlow: Learning Concept Erasure Policies via GFlowNet-Driven Alignmen (https://arxiv.org/abs/2511.00804)
Comments:
          NeurIPS'25 Spotlight | Project page: this https URL

- **What's New**: 이 논문에서는 EraseFlow라는 새로운 개념 지우기 프레임워크를 제안합니다. 기존의 개념 지우기 기법들은 이미지 품질 저하, 취약한 적대적 손실(adversarial loss)에 의존하거나 비효율적인 재훈련 주기를 요구하는 단점을 가지고 있었습니다. EraseFlow는 GFlowNet을 활용하여 denoising 경로의 탐색을 통해 개념 비학습을 수행하며, 이는 기존의 기법보다 월등한 결과를 보여줍니다.

- **Technical Details**: EraseFlow는 전체 denoising 경로를 샘플링하여 목표 개념에서 벗어난 생성을 유도하면서도 모델의 이전 지식을 보존하는 확률적 정책을 학습합니다. 이 과정에서 수동으로 설계된 보상 모델에 대한 의존성을 제거하며, 보상 없이도 효과적인 정렬(alignment) 전략을 제공합니다. EraseFlow는 conditional marginal distributions의 변화를 반영하여 동적으로 정렬을 조정할 수 있도록 설계되었습니다.

- **Performance Highlights**: EraseFlow는 UDAtk 벤치마크에서 기존의 기법을 초월하는 성능을 보이며, 특히 NSFW 개념을 지우는 과정에서 1%의 실패율을 기록했습니다. 프리트레인(pretrained) 모델의 품질을 보존하면서도 생성 품질을 유지하여, Fréchet Inception Distance (FID) 측정에서 최첨단 성능을 달성했습니다. EraseFlow는 타 적대적 및 필터링 기반 방법과 통합하여 성능을 더욱 향상시킬 수 있습니다.



### Applying Medical Imaging Tractography Techniques to Painterly Rendering of Images (https://arxiv.org/abs/2511.00702)
Comments:
          Exploratory investigation applying medical imaging tractography techniques to painterly image rendering. Code available at this https URL

- **What's New**: 이 논문은 확산 텐서 이미징(diffusion tensor imaging, DTI) 기법과 트랙토그래피(tractography)를 활용하여 인간의 신체 내 조직의 섬유 구조를 시각화하는 새로운 접근법을 제시합니다. 제안된 방법론은 이러한 기술을 예술적 이미지 렌더링에 적용하여, 화가의 붓질을 모방하는 방식으로 이미지를 처리합니다. 이를 통해 컴퓨터 그래픽스 커뮤니티와의 다학제 협력 가능성을 강조합니다.

- **Technical Details**: 이 기술은 구조 텐서(structural tensor)를 사용하여 이미지의 지역적 방향 정보를 더욱 잘 제공하며, 이는 전통적인 그래디언트 방식보다 유용합니다. 구조 텐서는 표면의 뚜렷한 변화가 있는 경우에 비대칭적인 텐서를 생성 후, 이를 바탕으로 고유 값 분해를 통해 방향성을 추출합니다. 이후 제안된 알고리즘은 트랙토그래피 방법을 통해 붓질을 생성하며, 이는 이미지의 형태를 따릅니다.

- **Performance Highlights**: 구조 텐서를 사용한 결과, 붓질의 품질이 눈에 띄게 향상되며 더욱 부드럽고 자연스러운 효과를 창출합니다. 제안된 방법은 낮은 해상도의 붓질로 시작하여, 상위 레이어에서 세밀한 붓질을 추가하는 멀티레이어 접근법을 사용합니다. 이러한 방식은 예술가의 화풍을 반영하며, 수행된 실험 결과는 다양한 스타일의 이미지에서 효과적으로 입증되었습니다.



### Been There, Scanned That: Nostalgia-Driven LiDAR Compression for Self-Driving Cars (https://arxiv.org/abs/2511.00652)
- **What's New**: 이 논문은 자율주행차(AV)가 생성하는 대량의 3D 포인트 클라우드 데이터를 효과적으로 압축하기 위한 새로운 프레임워크인 DejaView를 소개합니다. 기존 연구들이 단기적인 프레임 간 중복성을 이용해 압축을 수행했다면, DejaView는 확대된 시간 스케일(일 및 월)을 통해 중복성을 찾아 더 효율적인 압축을 실현합니다. DejaView는 자율주행차의 제한된 운영 영역과 주행 경로의 유사성을 활용하여 과거 데이터에 대한 델타(diff)를 사용하여 포인트 클라우드를 압축합니다.

- **Technical Details**: DejaView의 핵심은 소스 포인트 클라우드와 과거의 참조 포인트 클라우드 간의 차이를 계산하는 것입니다. 이는 압축 비율, 재구성 오류, 및 지연(latency)과의 trade-off를 포함합니다. 제안된 프레임워크는 각 포인트 클라우드의 고유한 포인트 구성 요소와 일반 포인트를 비교하여 차이를 압축하여 저장하는 방식으로 작동합니다. 이 과정에서 정확성과 속도를 조정하기 위해 코스(grained) 및 파인(grained) 검색 알고리즘이 결합되어 활용됩니다.

- **Performance Highlights**: DejaView의 실제 적용에서, LiDAR 데이터의 두 달 동안의 엔드 투 엔드 구현은 297K 포인트 클라우드를 처리하며 압축 비율 210을 달성했습니다. 재구성 오류는 고작 15cm로, 기존 방법들과 비교했을 때 현저하게 우수한 성능을 보입니다. 이러한 결과는 자율주행차 데이터 집합의 큰 양을 효과적으로 관리하고 저장할 수 있는 가능성을 보여줍니다.



### GDROS: A Geometry-Guided Dense Registration Framework for Optical-SAR Images under Large Geometric Transformations (https://arxiv.org/abs/2511.00598)
Comments:
          To be published in IEEE Transactions on Geoscience and Remote Sensing (T-GRS) 2025

- **What's New**: 본 연구는 GDROS라는 기하학적으로 안내된 밀집 등록 프레임워크를 제안합니다. 이 프레임워크는 Optical과 SAR 이미지 간의 글로벌 크로스 모달(image interaction)을 활용하여 고차원 상관관계 부피를 구축하고, 반복적으로 정제하여 픽셀별 밀집 대응 관계를 확립합니다. 또한, 리니어 최소제곱 회귀 모듈을 통해 예상되는 조화 변환을 최종 흐름 예측에 직접적으로 적용하여 예측 불일치를 완화합니다.

- **Technical Details**: GDROS는 CNN-Transformer 하이브리드 특징 추출 모듈을 사용하는데, 이는 Optical과 SAR 이미지에서 크로스 모달 깊은 특징을 추출하는 데 기여합니다. 이후 다중 스케일 4D 상관 관계 볼륨을 구성하고 정제합니다. 이러한 과정에는 밀집 광 흐름 필드를 기하학적으로 제한하는 최소 제곱 회귀 모듈이 포함되며, 이로 인해 예측의 분산을 줄일 수 있습니다.

- **Performance Highlights**: 세 가지 대표적인 데이터셋인 WHU-Opt-SAR, OS 데이터셋, 그리고 UBCv2 데이터셋에서의 광범위한 실험을 통해 GDROS가 현재의 최첨단 방법들을 모든 지표에서 상당히 초월하는 성능을 보였습니다. 정량적 및 정성적 결과는 GDROS의 우수성을 입증하며, 귀하는 높은 기하학적 변형을 가진 Optical-SAR 이미지를 대상으로 한 성능이 두드러진다는 점을 강조합니다.



### Image-based ground distance detection for crop-residue-covered so (https://arxiv.org/abs/2511.00548)
Comments:
          under review at Computers and Electronics in Agriculture

- **What's New**: 이번 논문에서는 작물 잔재물로 덮인 토양에서 정밀한 파종 깊이를 제어하기 위한 새로운 이미지 기반 방법론을 제시합니다. 기존의 거리 측정 기술은 잔재물과 토양의 거리를 구분할 수 없는 한계가 있었습니다. 이 연구는 3D 카메라와 RGB 카메라를 활용하여 깊이 정보와 색상 이미지를 동시에 획득하는 방법을 개발했습니다.

- **Technical Details**: 제안된 방법은 두 카메라로부터 얻어진 색상 이미지를 바탕으로 잔재물과 토양을 구분하고, 이를 통해 생성된 마스크 이미지를 깊이 이미지에 적용하여 정확한 토양 깊이 정보를 추출합니다. 이 과정에서 측정 오차는 ±3mm 이내로 유지되어, 실시간 구현이 가능한 것으로 나타났습니다. 이 기술은 보존 농업 기계에서의 정밀 파종 깊이 제어에 활용될 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 거리 측정 방법은 정밀 파종, 이식, 경운과 같은 깊이 제어가 필요한 다양한 응용 분야에 적용 가능성을 보였습니다. 특히 이 방법은 효율성을 높이면서도 사용자에게 신뢰할 수 있는 거리 정보를 제공합니다. 이는 보존 농업의 토양 건강과 물 절약에도 긍정적 영향을 미칠 것으로 기대됩니다.



### Learning an Efficient Optimizer via Hybrid-Policy Sub-Trajectory Balanc (https://arxiv.org/abs/2511.00543)
- **What's New**: 최근 생성 모델의 발전으로 인해 신경망의 가중치를 그래디언트 기반 최적화 없이 생성할 수 있게 되었습니다. 그러나 기존 방법들은 과도한 결합(over-coupling)과 긴 지평선(long-horizon)이라는 두 가지 문제에 제한되어 있었습니다. 본 논문에서는 Lo-Hp라는 두 단계의 가중치 생성 프레임워크를 제안하여 다양한 최적화 정책을 학습함으로써 유연성을 향상시킵니다.

- **Technical Details**: Lo-Hp는 가중치 준비 단계와 정책 학습 단계로 구성된 두 단계의 학습 프로세스를 분리합니다. 첫 번째 단계에서는 다양한 최적화 정책을 사용하여 오프라인 궤적을 구축하며, 두 번째 단계에서는 Hybrid-Policy Sub-Trajectory Balance를 채택하여 지역 최적화 정책을 캡처합니다. 이 접근 방식을 통해 로컬 수준에서의 추론 궤적을 제약해 효율성과 정확성을 높입니다.

- **Performance Highlights**: Lo-Hp는 전이 학습, 소수 샷 학습, 도메인 일반화 및 대형 언어 모델 적응과 같은 잦은 가중치 업데이트가 필요한 작업들에서 뛰어난 정확도와 추론 효율성을 입증하였습니다. 실험 결과, 이 방법은 글로벌 최적 가중치 생성을 강화하면서도 로컬 최적화 정책 학습에 집중할 수 있음을 보여줍니다.



### Three-dimensional narrow volume reconstruction method with unconditional stability based on a phase-field Lagrange multiplier approach (https://arxiv.org/abs/2511.00508)
Comments:
          Preprint, 30+ pages; multiple figures and tables; code and data: this https URL intended for submission to a computational mathematics journal

- **What's New**: 본 연구에서는 포인트 클라우드(Points Cloud)로부터 객체를 재구성하기 위한 효과적인 알고리즘을 제시합니다. Allen-Cahn 모델을 기반으로 한 이 알고리즘은 Lagrange multiplier 접근법을 이용하여 정의됩니다. 특히, 에지 탐지 기능을 통한 발전된 지배 방정식 솔루션이 안정성을 보장합니다.

- **Technical Details**: 이 연구는 Crank-Nicolson 시간 이산화 및 유한 차분(Finite Difference) 방법을 통해 공간 연산을 근사화합니다. Lagrange multiplier를 통해 지배 방정식의 재구성을 시행하여 원래의 에너지를 보존하는 안정적인 방안과 함께, 아주 강력한 수치적 기법을 개발합니다.

- **Performance Highlights**: 종합적인 수치 실험을 통해, 복잡한 3D 볼륨 재구성을 포함하여 알고리즘의 정확성, 안정성 및 효과성을 검증했습니다. 추가적으로, 특정 매개변수 선택이 재구성된 볼륨의 세부 수준과 정밀도에 미치는 영향을 분석하였습니다.



### Investigating Label Bias and Representational Sources of Age-Related Disparities in Medical Segmentation (https://arxiv.org/abs/2511.00477)
Comments:
          Submitted to ISBI 2026

- **What's New**: 이 논문은 의료 영상의 분할 작업에서 발생할 수 있는 알고리즘 편향(algorithmic bias)의 원인을 조사합니다. 특히, 유방암 분할에서 연령에 따른 성능 차이를 분석하고, 자동화된 라벨의 오류가 모델의 실제 편향을 잘못 나타낼 수 있다는 것을 밝혔습니다. 이 연구는 유방암 MRI 이미지를 포함한 MAMA-MIA 데이터셋에 대한 공정성 감사(fairness audit)를 처음으로 수행하여 정량적 기준을 설정합니다.

- **Technical Details**: MAMA-MIA 데이터셋은 동적 대조강조 자기공명영상(DCE-MRI)으로 구성된 1,506명의 환자 데이터를 포함합니다. 각 데이터는 전문가가 주석을 단 마스크와 자동 생성된 nnU-Net 마스크 쌍으로 구성됩니다. 연구에서는 나이 기준으로 환자를 세 개의 그룹(젊은, 중간, 노인)으로 분류하고, 라벨 편향(label bias)과 표현 편향(representational bias)을 측정하기 위해 세밀한 우선 순위 접근 방식을 사용하였습니다.

- **Performance Highlights**: 전문가가 주석을 단 Gold-Standard 라벨과 nnU-Net 마스크를 비교하여 성능을 평가했고, DPD(유병률-차이)와 DIR(불균형 영향 비율) 같은 공정성 메트릭스를 사용하여 성능 차이를 정의했습니다. 실험 결과, 젊은 환자들의 사례는 본질적으로 학습하기 더 어렵고, 편향된 자동화된 라벨로 학습할 때 시스템적 편향이 강화된다는 직접적인 증거를 보여주었습니다.



### Towards Reliable Pediatric Brain Tumor Segmentation: Task-Specific nnU-Net Enhancements (https://arxiv.org/abs/2511.00449)
- **What's New**: 이번 논문에서는 Pediatric brain tumors의 정확한 segmentation을 위한 고급 nnU-Net 프레임워크를 제안합니다. 제한된 데이터와 높은 해부학적 변동성, 기관 간의 이질적인 이미징으로 인한 도전을 극복하기 위해 여러 혁신적인 기술을 도입하였습니다. 제안된 모델은 BraTS 2025 Task-6 인증 리더보드에서 1위에 올라, 정밀성과 일관성을 입증했습니다.

- **Technical Details**: nnU-Net 아키텍처를 기반으로 하고 있으며, ResNet 블록과 Squeeze-and-Excitation (SE) attention을 포함하여 목표한 수정 작업을 진행했습니다. 또한, 각 인코더 레이어의 피처 채널 수를 두 배로 늘려 인코더의 표현 용량을 증가시켰습니다. Depthwise separable convolutions을 사용하여 접근 방식의 계산 효율성을 향상시켰습니다.

- **Performance Highlights**: 모델은 BraTS 2025 Task-6에서 여러 유형의 병변에 대해 0.759 (CC), 0.967 (ED), 0.826 (ET), 0.910 (NET) 등 뛰어난 Dice 점수를 기록하며 우수한 성능을 나타냈습니다. 이로 인해 다양한 종양 하위 유형과 이미징 변이에도 불구하고 높은 분할 정확성과 일반화 능력을 보여 주었습니다.



### Region-Aware Reconstruction Strategy for Pre-training fMRI Foundation Mod (https://arxiv.org/abs/2511.00443)
- **What's New**: 이번 연구에서는 resting-state fMRI를 위한 foundation model의 새로운 region-aware reconstruction 전략을 제안합니다. 기존의 무작위 영역 마스킹 방식을 넘어, Automated Anatomical Labelling Atlas (AAL3)를 활용하여 뇌의 의미적으로 일관된 영역을 선택적으로 마스킹하는 ROI 가이드 마스킹 전략을 도입했습니다. 이 방법은 기존의 마스킹 방식에 비해 ADHD 환자와 일반인을 구분하는 분류 정확도를 4.23% 향상시켰습니다.

- **Technical Details**: 우리의 self-supervised learning 프레임워크는 masked voxel reconstruction에 집중하는 pretraining 단계와 예측 작업에 대한 fine-tuning 단계로 구성됩니다. NeuroSTORM 모델을 기반으로 하여, ROI 기반 마스킹 전략을 도입함으로써 공간적 특이성을 향상시킵니다. AAL3 atlas를 사용하여 사전 정의된 해부학적 영역을 식별하고 이러한 ROI의 하위 집합을 선택적으로 마스킹합니다.

- **Performance Highlights**: ADHD-200 데이터셋을 사용한 결과, ROI 가이드 마스킹이 기존의 무작위 마스킹 방식에 비해 분류 정확도를 향상시켰습니다. 이 연구는 limbic 영역과 소뇌가 reconstruction fidelity 및 모델 표현에 중요한 역할을 함을 발견했으며, 향후 다양한 neuroimaging 데이터셋에서 이러한 접근 방식을 평가할 계획입니다. 우리 프레임워크는 fMRI 기반 응용 프로그램에서 일반화 및 해석 가능성을 향상시킵니다.



### Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling (https://arxiv.org/abs/2511.00411)
Comments:
          accepted by iccv 2025

- **What's New**: 이 논문에서는 Gradient-Guided Sampling (GGS)이라는 새로운 샘플링 기법을 제안하여, 적대적 공격의 전송성을 높이는 데 중점을 둡니다. GGS는 과거의 내부 반복(iteration)에서 얻은 그래디언트를 사용하여 샘플링 방향을 안내함으로써 탐색(Exploration)과 활용(Exploitation)의 균형을 맞추고, 전반적인 샘플링 효율성과 안정성을 향상시킵니다. 그 결과, 본 연구는 최신 전송 공격 기법보다 더 나은 성능을 보여줍니다.

- **Technical Details**: GGS는 MI-FGSM 기법에 기반하여 내부 반복 샘플링(inner-iteration sampling)을 도입하며, 그래디언트 상승 방향을 따라 샘플링을 가이드합니다. 이 방법은 고른 지역에서의 적대적 예시 생성에 도움을 주며, 부드러운 손실 표면(flat loss surface)과 높은 로컬 최대(local maxima)를 타겟으로 둡니다. 또한, 우리의 GGS는 기존의 랜덤 샘플링(Random Sampling) 기반 메서드와 호환되어 샘플링 효율성을 더할 수 있습니다.

- **Performance Highlights**: 종합적인 실험 결과는 다양한 DNN 아키텍처와 다중 모달 대형 언어 모델(MLLMs)에서 GGS의 우수성을 입증합니다. 실험을 통해 GGS는 목표 모델(target model)과 비목표 모델(non-target model)에서의 공격 성공률을 높이며, 특히 블랙박스 설정에서 효과적인 발전을 보여주었습니다. 이러한 성과들은 업무 환경에서 적대적 공격을 막기 위한 방어 메커니즘 개발에 기여할 것으로 기대됩니다.



### SonarSweep: Fusing Sonar and Vision for Robust 3D Reconstruction via Plane Sweeping (https://arxiv.org/abs/2511.00392)
Comments:
          8 pages, 9 figures, conference

- **What's New**: 이 논문에서는 SonarSweep이라는 새로운 엔드투엔드 딥러닝 프레임워크를 소개합니다. 이 프레임워크는 시각적 데이터와 소나 데이터를 교차 모드 융합하여 정확하고 밀집된 3D 복원을 가능하게 합니다. 기존 방법들의 한계를 극복하기 위해 클래식한 평면 스윕 알고리즘을 적응시켜, 잠재적인 깊이 가설에 걸쳐 소나 특징을 카메라의 기준 틀에 왜곡해 복원합니다.

- **Technical Details**: SonarSweep은 비전 기반 및 소나 데이터를 통합하여 고밀도 깊이 맵을 생성합니다. 본 방법은 반응하는 다중 모드 비용(volume) 공간을 구성하여 각 잠재적 깊이에 대한 특징 유사성을 인코딩합니다. 그렇게 함으로써, 기존 단일 모드 접근 방식들이 정밀하지 못하게 해결한 고유한 기하학적 불확실성을 줄이고, 정확하고 일관된 깊이 맵을 생성합니다.

- **Performance Highlights**: SonarSweep은 다양한 고난도의 수중 환경에서 기존 최첨단 방법들보다 뛰어난 성능을 나타냅니다. 실험 결과, 특히 높은 탁도에서 일관되게 밀집되고 정확한 깊이 맵을 생성하였으며, 이 연구를 위한 코드와 처음으로 동기화된 스테레오 카메라 및 소나 데이터 세트를 공개할 계획입니다.



### POSESTITCH-SLT: Linguistically Inspired Pose-Stitching for End-to-End Sign Language Translation (https://arxiv.org/abs/2511.00270)
Comments:
          Accepted at EMNLP 2025 (Main)

- **What's New**: 본 논문에서는 포즈 기반, 글로스(Gloss) 없는 수화 번역(SLT)을 위한 새로운 선행 학습(pre-training) 방식인 POSESTITCH-SLT를 제안합니다. 이는 언어 템플릿을 기반으로 한 문장 생성 기법에서 영감을 받아, 공공 단어 수준 데이터셋을 활용하여 수백만 개의 문장을 생성할 수 있도록 합니다. 이전 연구에 비해 단순한 Transformer 아키텍처를 사용하여, How2Sign와 iSign 데이터셋에서 성능이 향상되었음을 보여줍니다.

- **Technical Details**: POSESTITCH-SLT는 수화 영상에서 추출된 2D 포즈 시퀀스를 기반으로 하여, 해당하는 영어 문장을 생성하는 방법을 논의합니다. 이 방법은 대규모의 구문 구조와 그에 맞는 자연어 데이터를 결합하여 기계 학습 알고리즘의 훈련에 활용합니다. 특히, ASL(미국 수화)과 ISL(인도 수화) 데이터를 사용하여, 공통 어휘를 기반으로 생성된 밀리언 단위의 문장 데이터셋을 만들어, 이는 표준 Transformer 아키텍처로 모델 훈련에 사용됩니다.

- **Performance Highlights**: How2Sign 데이터셋에서 BLEU-4 점수가 1.97에서 4.56으로, iSign 데이터셋에서는 0.55에서 3.43으로 향상되었습니다. 이러한 결과는 기존의 최첨단 기술을 초월하며, pose 기반의 글로스 없는 번역에서의 우수한 성능을 보여줍니다. 이 연구는 저자들의 GitHub 페이지(https://github.com/Exploration-Lab/PoseStich-SLT)를 통해 공개된 데이터셋 및 코드를 활용하여 향후 연구에 기여하고자 합니다.



### Melanoma Classification Through Deep Ensemble Learning and Explainable AI (https://arxiv.org/abs/2511.00246)
Comments:
          Publisher-formatted version provided under CC BY-NC-ND 4.0 license. Original source produced by SciTePress

- **What's New**: 이번 논문은 멜라노마(멜라노마)가 조기 발견과 치료가 이루어지지 않으면 치명적이라는 점을 강조합니다. 최근 인공지능 기술이 피부과 의사들이 멜라노마를 조기에 발견하는 데 도움을 주고 있습니다. 특히, deep learning(DL) 기반 시스템들이 이러한 병변을 높은 정확도로 감지하는 데 성공하고 있습니다.

- **Technical Details**: 연구에서는 세 가지 최첨단 deep transfer learning 네트워크의 앙상블 학습(ensemble learning)을 사용하는 머신러닝 모델을 제안합니다. 또한, 예측의 신뢰성을 보장하기 위해 Explainable Artificial Intelligence(XAI) 기법을 활용하여 예측의 근거를 해석하는 접근 방식을 포함하고 있습니다. 이러한 방법은 DL 모델의 불투명성을 해결하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 제안된 모델은 예측의 투명성을 확보함으로써 더 높은 신뢰성과 의존성을 제공합니다. 연구 결과, 멜라노마 조기 발견에 있어 AI 시스템의 효율성을 증가시키며, 진단 정확성을 향상시키는 데 기여할 수 있습니다. 이와 같은 접근 방식은 헬스케어 분야에서 DL 활용을 극대화하는 데 중요한 발전으로 평가받고 있습니다.



### GeneFlow: Translation of Single-cell Gene Expression to Histopathological Images via Rectified Flow (https://arxiv.org/abs/2511.00119)
- **What's New**: 이번 논문에서는 GeneFlow라는 새로운 프레임워크를 소개하여, spatial transcriptomics (ST) 데이터를 이용해 세포 이미지를 병렬로 연결하는 방법을 제안합니다. GeneFlow는 attention 기반의 RNA 인코더와 조건부 UNet을 결합하여 H&E 또는 DAPI와 같은 다양한 염색 방법으로 고해상도 이미지를 생성합니다. 이 접근법은 transcriptomics와 이미지 간의 연속적이며 일대일 대응(mapping)을 가능하게 하여, 관찰된 유전자 발현 프로파일에서의 세포 형태 및 세포 간 상호작용을 현실적으로 재현할 수 있습니다.

- **Technical Details**: GeneFlow는 rectified flow를 적용하여 직관적으로 이해할 수 있는 이미지와 유전자 발현 데이터 간의 관계를 학습합니다. 모델은 임의의 노이즈에서 시작하여, 유전자 발현 매트릭스를 기반으로 고해상도의 histopathological 이미지를 생성하는 비선형 전환 과정을 구현합니다. 이 과정에서 고차 미분 방정식(ODE) 해결기를 사용하여 정밀하게 이미지를 합성합니다. 최종적으로, 단일 세포 또는 다중 세포의 유전자 발현 데이터를 인코딩하는 두 개의 경로를 서로 구성하여 다양한 생물학적 맥락을 다루고 있습니다.

- **Performance Highlights**: GeneFlow는 기존의 diffusion 기반 방법과 비교하여 모든 실험에서 더 우수한 성능을 보였습니다. 이를 통해 암 연구 및 정밀 의학 분야에서 심각한 분자 변화를 효과적으로 모델링할 수 있는 잠재력을 지니고 있습니다. 특히, GeneFlow는 특정 유전자 발현 패턴의 조직학적 표현을 시각화하고, 가설을 생성하며, 바이오마커 탐색에 도움을 줄 수 있습니다.



### Deep recurrent-convolutional neural network learning and physics Kalman filtering comparison in dynamic load identification (https://arxiv.org/abs/2511.00100)
Comments:
          31 pages, 20 figures, published in Structural Health Monitoring

- **What's New**: 본 연구에서는 gated recurrent unit (GRU), long short-term memory (LSTM), 및 convolutional neural networks (CNN)의 동적 구조 하중 식별 능력을 조사합니다. 특히 실제적이며 작은 데이터셋 훈련 조건하에서의 비교 뷰를 갖춘 물리 기반 잔여 칼만 필터(residual Kalman filter, RKF)와의 비교에 중점을 둡니다. 이 과정에서 구조 모델이 식별되지 않거나 테스트가 적을 때 예측의 불확실성이 발생하는 문제를 다룹니다.

- **Technical Details**: 이 연구는 시뮬레이션 구조가 상부에서 진동을 일으키는 경우와 캘리포니아의 한 건물이 지진 기반의 하중을 받는 경우를 포함하여 다양한 시나리오에서 분석됩니다. 또한, International Association for Structural Control-American Society of Civil Engineers (IASC-ASCE)의 구조 건강 모니터링 벤치마크 문제를 다루어 즉각적인 하중 조건에 대한 성능을 평가합니다. 이 방법들은 각기 다른 하중 시나리오에서 서로 성능을 초월하는 것으로 나타났습니다.

- **Performance Highlights**: 각 방법들은 서로 다른 하중 시나리오에 따라 우수한 성과를 보이며, 특히 물리적으로 매개변수가 식별 가능한 경우에는 RKF가 네트워크보다 더 뛰어난 성능을 발휘하는 것으로 보여집니다. 연구 결과는 건설 공학 분야에서 동적 하중 식별의 정확도를 높이는 데 기여할 수 있는 중요한 통찰을 제공합니다.



### A generative adversarial network optimization method for damage detection and digital twinning by deep AI fault learning: Z24 Bridge structural health monitoring benchmark validation (https://arxiv.org/abs/2511.00099)
Comments:
          21 pages, 23 figures, published in Structural and Multidisciplinary Optimization

- **What's New**: 이번 연구에서는 최적화 기반의 손상 탐지와 디지털 트윈(digital twinning) 기술을 결합한 새로운 조건부 라벨 생성적 적대 신경망(conditional-labeled generative adversarial network) 방법론을 제안합니다. 이 프레임워크는 시스템의 상태에 대한 사전 정보 없이도 현재의 결함 이상 탐지 방법보다 우수한 성능을 보여, 실제 응용 분야에서 중요한 의미를 갖습니다. 기존의 인공지능 기반 디지털 트윈 접근법이 직면한 불확실성을 해결하는 데 중점을 두었습니다.

- **Technical Details**: 새로운 무감독(unsupervised) 프레임워크는 스위스 Z24 브리지의 구조 건강 모니터링 측정값을 사용하여 철저히 검증되었습니다. 이 방법은 동일한 손상 수준의 측정값을 입력으로 사용하여 모델이 두 가지 손상 상태로 조건부 수렴하도록 강제하고, 이후 다른 측정 그룹에 대해 이 과정을 반복하여 서로 다른 손상 상태를 구분합니다. 손상 없는 데이터와 손상 있는 데이터 모두에 대해 디지털 트윈을 위한 측정값을 생성할 수 있습니다.

- **Performance Highlights**: 제안된 접근법은 건강 측정값을 기반으로 손상을 정확하게 포착하여, 진동(vibration) 기반 시스템 모니터링과 확장 가능한 인프라 복원력을 위한 강력한 도구로 기능합니다. 추가적으로 서포트 벡터 머신(classifier)과 주성분 분석(principal component analysis) 절차가 개발되어, 각 손상 카테고리의 측정값을 평가하는 새로운 역학 학습 지표로 활용됩니다. 이로 인해 건강 상태와 손상 상태 간의 패턴 인식 및 기계 학습 데이터 생성을 가능하게 합니다.



### LookSync: Large-Scale Visual Product Search System for AI-Generated Fashion Looks (https://arxiv.org/abs/2511.00072)
Comments:
          4 pages, 5 figures. Accepted at the International Conference on Data Science (IKDD CODS 2025), Demonstration Track. Demo video: this https URL

- **What's New**: 이번 논문에서는 AI 생성 스타일과 유사한 제품을 빠르게 검색할 수 있는 시스템을 소개합니다. 이 시스템은 1200만 개 이상의 제품에 대한 고차원 임베딩을 생성하며, AI 생성 이미지를 키워드로 변환하여 시각적으로 유사한 제품을 찾습니다. 이 시스템은 일일 35만 개의 AI Looks를 처리하며, 이를 통해 사용자에게 더욱 개인화된 쇼핑 경험을 제공합니다.

- **Technical Details**: 제안하는 시스템은 쿼리 생성, 벡터화, 후보 검색, 재순위 매김을 포함한 4개의 주요 구성 요소로 이루어져 있습니다. AI 생성 이미지를 반영하여 가장 유사한 제품을 추출하며, CLIP 모델을 활용하여 높은 정확도로 유사도를 평가합니다. 시스템은 평균적으로 1초 이내의 응답 속도를 유지하며, 사용자 인터렉션에 기반한 추천 품질도 개선되었습니다.

- **Performance Highlights**: 제안된 시스템은 CLIP을 기반으로 하여 다양한 제품 카테고리에서 비슷한 스타일을 유지하는 제품을 효과적으로 찾아냅니다. 실험 결과, CLIP 모델이 다른 대안 모델들보다 평균 의견 점수에서 3~7% 우위를 점했습니다. 이러한 작은 개선이 사용자 경험을 크게 향상시키는 것을 보여주며, 실제 운영 환경에서 경쟁력 있는 솔루션으로 자리잡게 되었습니다.



### Multimodal Detection of Fake Reviews using BERT and ResNet-50 (https://arxiv.org/abs/2511.00020)
Comments:
          Published in IEEE

- **What's New**: 이 연구에서는 디지털 상거래 환경에서 사용자가 생성한 리뷰의 중요성을 강조하고, 허위 리뷰의 문제를 해결하기 위한 다중 모달(fake review detection) 프레임워크를 제안합니다. 이 프레임워크는 BERT를 활용한 텍스트 특징과 ResNet-50을 활용한 시각적 특징을 결합하여 리뷰의 진위를 판별합니다. 기존의 단일 모달(unimodal) 접근 방식이 아닌 다중 모달(multimodal) 접근 방식을 통해 세밀한 모순을 감지할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 방법론은 데이터셋 준비, 전이 학습을 통한 특징 추출, 다중 모달 특징 융합, 이진 분류의 네 가지 주요 단계로 구성됩니다. 텍스트 특징은 BERT의 사전 훈련된 모델을 사용하여 추출하고, 이미지 특징은 ResNet-50을 통해 획득합니다. 이 두 가지 특징 벡터는 결합되어 최종 이진 분류를 위한 입출력 벡터를 형성합니다.

- **Performance Highlights**: 실험 결과, 다중 모달 모델은 단일 모달 기초선에 비해 뛰어난 성능을 나타내며, 테스트 세트에서 F1-score 0.934를 기록했습니다. 또한 혼돈 행렬(confusion matrix)과 질적 분석을 통해 모델이 고품질 이미지를 보유한 잘못된 텍스트 높은 호평 리뷰를 감지하는 능력을 확인했습니다. 이 연구는 디지털 신뢰를 보장하는 데 있어 다중 모달 학습의 중요한 역할과 온라인 플랫폼에서의 콘텐츠 조정 문제를 해결하는 스케일러블한 솔루션을 제시합니다.



### Multimodal Learning with Augmentation Techniques for Natural Disaster Assessmen (https://arxiv.org/abs/2511.00004)
Comments:
          Accepted at 2025 IEEE 21st International Conference on Intelligent Computer Communication and Processing (ICCP 2025)

- **What's New**: 이 논문은 자연재해 평가에 필요한 정보에 대한 신속하고 정확한 접근 방식의 중요성을 강조하고 있습니다. 특히 소셜 미디어가 재난 분석을 위한 실시간 데이터 소스로 주목받고 있지만, 기존 데이터셋이 불균형 클래스와 한정된 샘플 문제로 인해 모델 개발이 어렵다는 점를 지적합니다. 해결책으로, 실험에서는 CrisisMMD 다중 양식 데이터셋을 활용하여, 다양한 증강 기법을 적용하여 재난 분류 모델의 성능을 향상시킵니다.

- **Technical Details**: 이 연구에서는 주로 두 가지 이미지 증강 기술인 Real Guidance와 DiffuseMix를 적용했습니다. Real Guidance는 조건부 이미지-투-이미지 생성 기법으로, Stable Diffusion 1.5 모델을 사용하여 원래 이미지를 현실적으로 변형합니다. DiffuseMix는 Masked Blending과 Fractal Visual Modifications를 사용하여 다양한 증강 이미지를 생성하고, 이는 데이터셋의 일부에 국한된 클래스의 샘플 수를 증가시킵니다.

- **Performance Highlights**: 실험 결과, 선택된 증강 전략이 특히 저조한 클래스의 분류 성능을 개선하는 데 기여했음을 보여주었습니다. 다중 시점 학습은 잠재력을 보였지만 추가적인 정제가 필요하다는 점이 지적되었습니다. 이 연구는 제안한 증강 기법들이 재난 평가 시스템을 더욱 견고하게 구축하는 데 기여할 수 있음을 보여줍니다.



### VRScout: Towards Real-Time, Autonomous Testing of Virtual Reality Games (https://arxiv.org/abs/2511.00002)
- **What's New**: 이 논문에서는 VR 환경 내에서 자율적으로 탐색하고 가상 객체와 상호작용할 수 있는 딥 러닝 기반의 에이전트인 VRScout를 소개합니다. VRScout는 인간의 시범을 학습하여 자연스럽고 효율적으로 행동하며, Action Chunking Transformer (ACT)를 통해 멀티 스텝 액션 시퀀스를 예측합니다. 이를 통해 다양한 환경에 걸쳐 고급 전략을 캡처하고 일반화할 수 있습니다.

- **Technical Details**: VRScout는 VR 장면 이미지와 사용자 동작을 입력으로 받아, ResNet-18 인코더를 사용하여 특징 벡터를 생성하고, 이 벡터를短期적(time chunk)으로 묶어 ACT 모델의 인코더에 공급합니다. 예측된 동작은 가상 VR 컨트롤러를 통해 게임에 주입되어 다음 게임 상태를 생성하며, 동적 슬라이딩 호라이즌을 통해 에이전트의 시간적 문맥을 실행 중에 조정합니다.

- **Performance Highlights**: VRScout는 상업적으로 인기 있는 VR 게임에서 전문가 수준의 성과를 달성하며, Beat Saber에서는 4시간의 훈련 데이터만으로도 효과적인 성능을 보입니다. 더욱이 소비자 등급의 하드웨어에서 60 FPS의 실시간 추론을 수행할 수 있어 VR 게임의 자동화된 테스트를 위한 실용적이고 확장 가능한 솔루션으로 자리매김하고 있습니다.



### D$^2$GS: Dense Depth Regularization for LiDAR-free Urban Scene Reconstruction (https://arxiv.org/abs/2510.25173)
- **What's New**: 본 논문에서는 LiDAR 없이도 도시 장면을 재구성할 수 있는 D$^2$GS 프레임워크를 제안합니다. 이 방법은 기존의 LiDAR 데이터 의존성을 줄이면서도 동일한 수준의 기하학적 정확성을 제공합니다. 이를 통해 LiDAR 데이터 수집의 어려움과 정확성 문제를 해결할 수 있습니다.

- **Technical Details**: D$^2$GS는 초기 밀집 포인트 클라우드를 멀티 뷰 메트릭 깊이 예측에서 역투영(back-projecting)하여 생성합니다. 그런 다음, Progressive Pruning 전략을 사용하여 전역 일관성을 개선하는 최적화 과정을 수행합니다. Depth Enhancer 모듈을 통해 Gaussian의 기하학을 공동 최적화하고 깊이를 세밀하게 향상시킵니다.

- **Performance Highlights**: Waymo 데이터셋에 대한 실험 결과, D$^2$GS는 최신 방법들과 비교할 때 더 정확한 기하학적 재구성을 보여주었습니다. 특히 LiDAR 데이터의 기초가 없는 데이터에서도 강력한 성능을 발휘하며, 새로운 고해상도 도시 장면 모델링을 가능하게 합니다.



### A Quantitative Evaluation Framework for Explainable AI in Semantic Segmentation (https://arxiv.org/abs/2510.24414)
- **What's New**: 이번 연구는 인공지능(AI) 모델의 투명성과 신뢰성을 보장하기 위한 새로운 방법론을 제시합니다. 특히, 세분화(semantic segmentation) 작업에 맞춤화된 설명 가능 인공지능(XAI) 평가 프레임워크를 도입하였습니다. 이 프레임워크는 공간적 및 맥락적 과제를 고려하여 XAI 방법의 효과를 정량적으로 평가할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 XAI 평가 프레임워크는 네 가지 주요 전략: (S1) 배경 제외, (S2) 강조된 영역만, (S3) 예측 마스크와 실제 데이터 비교(XAI-PM 및 XAI-GT)로 구성됩니다. 이러한 방법들은 모델의 결정에 영향을 미치는 중요한 픽셀을 평가하여 설명 가능성을 높이는 데 기여합니다. 픽셀 수준의 평가 전략과 정교하게 설계된 메트릭을 통합하여 세분화 모델의 해석 가능성을 제공합니다.

- **Performance Highlights**: 시뮬레이션 결과를 통해 최근 적응된 클래스 활성화 맵(class activation mapping, CAM) 기반의 XAI 방법의 효율성과 신뢰성이 입증되었습니다. 제안된 방법론은 투명하고 신뢰할 수 있으며, 책임이 있는 세분화 모델 개발을 촉진하는 데 중요한 진전을 이루었습니다. 이 연구는 XAI 평가의 새로운 지평을 열 것으로 기대됩니다.



New uploads on arXiv(cs.AI)

### Simulating Environments with Reasoning Models for Agent Training (https://arxiv.org/abs/2511.01824)
- **What's New**: 이 논문은 LLM (Large Language Model) 에이전트가 실제 테스트베드 데이터나 API에 접근하지 않고도 현실적인 환경 피드백을 시뮬레이션 할 수 있음을 보여줍니다. 여기서 제안하는 두 가지 프레임워크는 Simia-SFT로, 다양한 궤적을 생성하기 위한 SFT 데이터를 증폭하는 파이프라인과 Simia-RL로, LLM 시뮬레이션 피드백을 통해 RL (Reinforcement Learning) 훈련을 가능하게 합니다. 이들을 통해 환경 엔지니어링의 필요성을 제거하며, 유연한 LLM 기반 시뮬레이션으로 대체합니다.

- **Technical Details**: Simia-SFT 파이프라인은 소규모 씨앗 세트에서 환경에 구애받지 않는 다양한 궤적을 합성합니다. 이를 위해 4단계 절차를 거쳐 LLM 기반의 사전 필터링, 프롬프트 디자인, LLM 궤적 시뮬레이션, 규칙 기반 검사를 실시합니다. 가장 중요한 것은 이 과정에서 실제 환경 구현 없이 궤적을 생성할 수 있도록 LLM이 기초 데이터에서 새로운 에이전트 궤적을 합성한다는 점입니다.

- **Performance Highlights**: 실험 결과, Simia-SFT와 Simia-RL을 통해 훈련된 오픈 모델들이 다양한 벤치마크에서 일관된 향상을 보여 GPT-4o를 초과하며 o4-mini에 근접하는 성과를 도출했습니다. 구체적으로는, Qwen3-8B와 Qwen2.5-32B-Instruct 모델이 시뮬레이션된 궤적에서 상당한 이점을 보여줍니다. 이러한 결과는 LLM 시뮬레이터를 통해 환경 특정 코드를 대체할 수 있는 실용적인 방법론을 제시합니다.



### Hybrid Retrieval-Augmented Generation Agent for Trustworthy Legal Question Answering in Judicial Forensics (https://arxiv.org/abs/2511.01668)
- **What's New**: 이 연구는 기존의 법률 상담 시스템의 한계를 극복하기 위해 혼합형 법률 QA 에이전트를 제안합니다. 이 시스템은 retrieval-augmented generation (RAG)와 다중 모델 앙상블(multi-model ensembling)을 결합하여 신뢰할 수 있는, 추적 가능하며 지속적으로 업데이트 가능한 법률 상담을 제공합니다. 특히 이 요약된 접근 방식은 법적 질문에 대한 정확성을 높이기 위해 고안되었습니다.

- **Technical Details**: 제안된 시스템은 사용자의 질의가 신뢰할 수 있는 지식 기반에서 유사한 항목과 일치할 때 RAG를 통해 답변을 생성합니다. 만약 정보 검색이 실패할 경우, 여러 개의 pretrained 모델에서 후보 답변을 생성한 다음, 특수한 선택기(selector)를 통해 최상의 응답을 선정합니다. 또한, 고품질의 답변은 인간 검토(human review)를 거친 후 지식 기반에 다시 기록되어 법률의 변화에 맞춘 동적인 진화를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안된 하이브리드 방법이 기존의 단일 모델 기준선 및 일반적인 RAG 파이프라인보다 F1 점수와 ROUGE-L에서 우수한 성능을 보였습니다. 또한, ablation 연구(ablation studies)를 통해 검색 우선 순위, 모델 앙상블 및 인간 검토 메커니즘의 상호 보완적인 기여가 확인되었습니다. 이 시스템은 환각(hallucination)을 줄이는 동시에 답변의 질과 법적 준수를 향상시키며, 법률 분야에서의 실제 활용 가능성을 보여줍니다.



### IVGAE-TAMA-BO: A novel temporal dynamic variational graph model for link prediction in global food trade networks with momentum structural memory and Bayesian optimization (https://arxiv.org/abs/2511.01639)
Comments:
          26pages,6figures

- **What's New**: 이번 연구에서는 글로벌 식품 무역 네트워크의 동적 구조를 모델링하고 미래의 무역 링크를 예측하기 위해 IVGAE-TAMA-BO라는 새로운 동적 그래프 신경망(dynamic graph neural network)을 소개합니다. 이는 동적 그래프 신경망을 이 분야에 처음으로 적용한 연구로, 예측 성능을 획기적으로 향상시켰습니다.

- **Technical Details**: 제안된 모델은 원래의 IVGAE 프레임워크를 기반으로 하며, 무역 네트워크의 시계열적 진화를 포착하기 위해 Trade-Aware Momentum Aggregator (TAMA)를 통합합니다. 이 모델은 단기적인 변동성과 장기적인 구조적 의존성을 함께 모델링하며, 모멘텀 기반 구조적 메모리 메커니즘이 예측의 안정성과 성능을 더욱 향상시킵니다. 또한 베이esian 최적화(Bayesian optimization)를 사용하여 주요 하이퍼파라미터를 자동으로 조정합니다.

- **Performance Highlights**: 다섯 개의 작물별 데이터셋을 대상으로 한 광범위한 실험 결과, IVGAE-TAMA는 정적 IVGAE와 다른 동적 기초선(dynamic baselines)보다 시계열적 의존성을 효과적으로 모델링하여 훨씬 더 우수한 성능을 보여줍니다. 또한, IVGAE-TAMA-BO에서 베이esian 최적화가 성능을 더욱 증대시킴으로써, 제안된 프레임워크가 글로벌 무역 네트워크의 구조적 예측을 위한 강력하고 확장 가능한 솔루션임을 입증하였습니다.



### ExplicitLM: Decoupling Knowledge from Parameters via Explicit Memory Banks (https://arxiv.org/abs/2511.01581)
Comments:
          12pages, 4figures

- **What's New**: 이 논문에서는 ExplicitLM이라는 새로운 아키텍처를 제안합니다. 이 모델은 백만 규모의 외부 메모리 뱅크(External Memory Bank)를 활용하여 사람 친화적으로 읽을 수 있는 지식을 토큰 시퀀스로 저장합니다. 이를 통해 지식의 직접적인 점검 및 수정을 가능하게 합니다.

- **Technical Details**: ExplicitLM은 두 단계의 차별화된 검색 메커니즘을 설계하여 효율적인 대칭 필터링을 구현했습니다. 여기서 제품 키 분해(Product Key Decomposition)를 활용하여 복잡도를 $\mathcal{O}(N \cdot |I|)$에서 $\mathcal{O}(\sqrt{N} \cdot |I|)$로 줄이고, 끝-투-끝(training) 학습을 위한 세밀한 Gumbel-Softmax 매칭을 포함합니다. 지식은 20%의 고정 명시적 사실과 80%의 학습 가능한 암묵적 패턴으로 나누어 유지됩니다.

- **Performance Highlights**: ExplicitLM은 전통적인 Transformer와 비교하여 지식 집약적인 작업에서 최대 43.67% 개선 효과를 보였습니다. 또한 저 데이터 조건(10K 샘플)에서 3.62배 향상을 달성했습니다. 메모리 검색과 성능 간의 강한 상관관계가 분석되었으며, 올바른 예측은 49% 높은 적중률을 기록했습니다.



### Analyzing Sustainability Messaging in Large-Scale Corporate Social Media (https://arxiv.org/abs/2511.01550)
- **What's New**: 이번 연구에서는 지속 가능성과 관련된 기업의 소셜 미디어 콘텐츠를 분석하기 위해, 비전(vision) 및 언어(language) 분야의 대형 모델을 활용하는 다중 자료(multimodal) 분석 파이프라인을 제시합니다. 기존의 비싼 태스크 특정 주석(annotation)을 피하면서 기업 트윗 데이터를 17개 지속 가능한 개발 목표(SDGs)와의 주제 정렬에 따라 주석 달 수 있도록 대형 언어 모델(LLMs)의 앙상블을 사용합니다. 이 통합된 접근법은 텍스트와 비주얼 분석을 통해 소셜 미디어에서 지속 가능성에 대한 명시적 및 암시적 참조를 캡처하는 잠재력을 탐구합니다.

- **Technical Details**: 우리는 대형 언어 모델(LLMs)과 비전-언어 모델(VLMs)의 영점 샷(zero-shot) 특성을 활용하여 기업 소셜 미디어 콘텐츠를 포괄적으로 분석하는 다중 자료 분석 파이프라인을 설계했습니다. 텍스트 분석을 위해 LLM의 앙상블을 사용하여 대규모 텍스트 데이터(예: 기업 트윗)에 대한 주제를 매핑하고, VLM을 통해 비주얼 콘텐츠(이미지 및 인포그래픽)를 분석합니다. 이러한 방법론은 주제 태그 지정에서의 정확도와 강건성을 향상시키며, 텍스트와 비주얼 분석의 통합을 통해 소셜 미디어 콘텐츠의 전반적인 탐구를 가능하게 합니다.

- **Performance Highlights**: 우리는 이 방법들을 활용하여 기업의 지속 가능성 커뮤니케이션을 분석하고, 각 산업에서 어떤 SDGs가 가장 자주 논의되는지, 그리고 이러한 논의가 시간에 따라 어떻게 변화하는지 통찰을 제공합니다. 또한 비주얼 이해 파이프라인을 사용하여 트윗과 관련된 비주얼 콘텐츠의 주제를 식별하고 분석하였으며, 특정 시각적 주제가 높은 ESG 리스크 프로필이나 사용자 참여와 어떻게 상관관계가 있는지를 밝혀냈습니다. 이러한 접근법은 지속 가능성 커뮤니케이션에서의 시각적 수사와 이해관계자 인식 간의 관계를 드러냅니다.



### TPS-Bench: Evaluating AI Agents' Tool Planning \& Scheduling Abilities in Compounding Tasks (https://arxiv.org/abs/2511.01527)
- **What's New**: 본 논문은 TPS-Bench라는 새로운 벤치마크를 소개하여, 여러 도구를 활용해야 하는 복합적인 문제를 해결하는 LLM(대형 언어 모델) 에이전트의 능력을 평가합니다. TPS-Bench는 수백 개의 모델 컨텍스트 프로토콜(MCP) 도구를 기반으로 두 가지 난이도의 200개의 복합 작업을 수집하여, LLM 에이전트가 도구 계획과 스케줄링을 어떻게 수행하는지를 측정합니다. 이를 통해 LLM 에이전트가 다양한 도구를 선택하고 효율적인 실행 순서를 전략적으로 계획하는 능력을 공개적으로 평가할 수 있는 새로운 기준을 제시합니다.

- **Technical Details**: TPS-Bench는 여러 하위 작업으로 구성된 복합 작업을 포함하며, 하위 작업에는 웹 검색, 지도 내비게이션, 일정 확인 등이 포함됩니다. 복합 작업의 난이도는 두 가지로 나뉘며, TPS-Bench-Easy는 간단한 하위 작업으로 구성되고 TPS-Bench-Hard는 더 복잡한 하위 작업으로 구성됩니다. 각 작업은 LLM-as-a-judge를 사용하여 작업 완료율을 평가하고, 토큰 사용량과 시간 소비를 체계적으로 측정하여 효율성을 평가합니다.

- **Performance Highlights**: 실험 결과, 대부분의 LLM 모델이 도구 계획에서는 합리적인 성능을 보였으나, 스케줄링에 있어서는 차이를 보였습니다. GLM-4.5는 순차 도구 실행을 사용하여 TPS-Bench-Hard의 작업 완료율 64.72%를 기록했지만, 평균 실행 시간은 217.8초에 달했습니다. 반면 GPT-4o는 병렬 스케줄링을 우선시하여 실행 시간을 76.84초로 줄였지만 45.08%의 작업 완료율을 기록했습니다. Qwen3-1.7B에 대한 강화 학습(RL) 기법을 통해 6%의 작업 완료율 증가와 14%의 실행 시간 단축을 관찰하며, 이 연구는 차세대 LLM 성능 개선에 기여할 가능성을 보여줍니다.



### From Passive to Proactive: A Multi-Agent System with Dynamic Task Orchestration for Intelligent Medical Pre-Consultation (https://arxiv.org/abs/2511.01445)
Comments:
          14pages, 7 figures, 7 tables

- **What's New**: 이 연구는 기존 수동 의료 AI 시스템을 자율적 작업 조정 기능을 활용해 능동적인 탐색 에이전트로 변화시키는 계층적 다중 에이전트 프레임워크를 소개합니다. 이 프레임워크는 사전 상담 프로세스를 네 가지 주요 작업으로 나누어 효율적인 의료 진단을 지원합니다. 1,372개의 전자 건강 기록을 통해 성능 평가를 수행하여, 주 진료과 등급에서 87.0%의 정확도와 98.2%의 작업 완료 비율을 달성했습니다.

- **Technical Details**: 이 프레임워크는 Controller라는 중앙 제어 메커니즘을 활용하여 전문 에이전트 간의 상호작용을 조정하며, 다이나믹한 서브태스크 완료 평가와 환자 반응을 기반으로 한 적응형 프롬프트 생성을 통해 구조화된 진료기회를 제시합니다. 네 가지 주요 작업인 Triage, History of Present Illness collection, Past History collection, Chief Complaint generation으로 구성되며, 이를 통해 정보 수집의 체계성과 포괄성을 보장합니다.

- **Performance Highlights**: 이 연구는 전통적인 수동 탐색 접근 방식과 비교하여 유의미한 성과를 입증했습니다. 주된 성과로는 12.7회 및 16.9회의 상담 라운드를 통해 효율적인 태스크 완료를 달성하였으며, 18명의 의사로부터 접수된 평가 점수도 평균 4.56, 4.48, 4.69로 높은 임상 품질 점수를 기록하였습니다.



### Robust Multimodal Sentiment Analysis via Double Information Bottleneck (https://arxiv.org/abs/2511.01444)
- **What's New**: 이번 논문에서는 멀티모달 감정 분석(Multimodal Sentiment Analysis, MSA)의 두 가지 주요 한계를 극복하기 위한 더블 정보 병목(Double Information Bottleneck, DIB) 전략을 제안합니다. 첫째, 노이즈에 오염된 단일 모달 데이터 학습의 불충분함, 둘째, 멀티모달 표현의 융합 부족 문제를 해결하고자 하였으며, 이를 통해 강력하고 통합된 멀티모달 표현을 생성하는 데 초점을 맞췄습니다. DIB는 저차원 렌이(entropy functional) 기반에서 구현되어 다양한 노이즈 소스에 대해서도 강인함을 보장합니다.

- **Technical Details**: DIB는 두 가지 주요 모듈로 구성됩니다: 1) 개별 단일 모달 데이터의 충분하고 압축된 표현을 학습하고 불필요한 정보를 버리는 모듈, 2) 애정 있는 주의 병목 융합 메커니즘을 통해 멀티모달 표현의 구별 능력을 보장하는 모듈입니다. 이 접근 방식은 노이즈와 배경 간섭을 효과적으로 차단하며, 중요한 정보를 포착하는 데 있어 더 나은 성능을 발휘하도록 설계되었습니다. 특히, 저차원 렌이 엔트로피는 고차원 데이터에서도 직관적인 계산 가능성을 제공하는 특징이 있습니다.

- **Performance Highlights**: CMU-MOSI, CMU-MOSEI, CH-SIMS, MVSA-Single과 같은 다양한 데이터셋에서 DIB의 효과가 검증되었습니다. CMU-MOSI 데이터셋에서 Acc-7 지표로 47.4%의 정확도를 기록하였고, CH-SIMS에서 81.63%의 F1-score를 달성하여 두 번째로 좋은 기준 모델을 1.19% 초과하였습니다. 노이즈 조건에서도 CMU-MOSI와 CMU-MOSEI에서 각각 0.36%와 0.29%의 성능 저하만을 나타내어 강력한 성능을 유지했습니다.



### Learning to Seek Evidence: A Verifiable Reasoning Agent with Causal Faithfulness Analysis (https://arxiv.org/abs/2511.01425)
Comments:
          12 pages, 3 figures. Under review at the Conference on Computer Vision and Pattern Recognition (CVPR) 2026

- **What's New**: 이번 논문에서는 의료와 같은 고위험 분야에서 AI 모델의 설명 가능성이 신뢰성 문제를 야기하는 점을 해결하기 위해, 검증 가능하고 신뢰성 있는 설명을 제공할 수 있는 인터랙티브 에이전트를 제안합니다. 이 에이전트는 진단 추론을 지원하기 위해 외부 시각적 증거를 전략적으로 탐색하는 정책을 학습하며, 강화 학습(reinforcement learning)을 통해 최적화된 성능을 보여줍니다. 실험 결과, 이 행동 기반 추론 과정이 조정된 정확도를 약 18% 개선하여 신뢰성을 향상시킴을 확인했습니다.

- **Technical Details**: 제안하는 프레임워크는 가설 및 증거 검증을 위한 통합적인 상호작용 루프를 통해 이루어집니다. 이 에이전트는 Vision-Language Model(VLM)을 활용하여 이미지와 텍스트를 동시에 처리하며, 진단 프로세스를 명확하게 추적할 수 있는 구조로 모델링합니다. 주요 동작인 Probe & Ground(P&G)는 외부 도구를 호출하여 증거를 분석하고 신뢰도 점수를 반환, 이 피드백을 통해 에이전트의 가설 상자(Hypothesis Box, H-Box)를 업데이트하는 과정을 포함합니다.

- **Performance Highlights**: 우리는 occlusion tests를 포함한 개입 평가 프로토콜을 도입하여 설명의 신뢰성을 정량적으로 검증합니다. 이 방법을 통해 에이전트가 선택한 증거가 차단되었을 때 성과가 측정 가능한 감소를 나타내며, 이는 해당 증거가 의사결정 과정에서 중요한 역할을 함을 입증합니다. 이러한 특성 덕분에, 제안된 시스템은 상업용 하드웨어에서 훈련이 가능하며, AI의 신뢰성을 높이는 새로운 기초를 제공합니다.



### Modulation of temporal decision-making in a deep reinforcement learning agent under the dual-task paradigm (https://arxiv.org/abs/2511.01415)
Comments:
          Accepted at CogInterp workshop @ NeurIPS 2025

- **What's New**: 이 연구는 인공지능(AI) 관점에서 이중 작업 패러다임 내의 시간 처리 간섭을 탐구합니다. 시간을 생산하는 임무가 포함된 단일 작업(T)과 동시 숫자 비교 작업이 추가된 이중 작업(T+N)의 두 가지 변형이 구현되었습니다. 이 연구에서 훈련된 두 개의 심층 강화 학습(DRL) 에이전트는 인간의 타이밍 연구와 일치하는 행동을 나타냈습니다.

- **Technical Details**: 방법론적으로, 이 연구는 OverCooked 환경의 단순화 버전을 사용하여 5x3 그리드 월드에서 동작하는 에이전트를 정의합니다. 에이전트는 수프를 배달하는 임무에서 타이머 기능을 수행하며, 이중 작업에서는 숫자 비교 작업이 추가됩니다. DRL 에이전트는 각 작업에 대해 100,000 시간 단계 동안 훈련되었으며, Proximal Policy Optimization (PPO) 알고리즘을 사용하여 구현되었습니다.

- **Performance Highlights**: 행동 분석 결과, 이중 작업(T+N) 에이전트는 단일 작업(T) 에이전트에 비해 평균적으로 상당히 높은 첫 번째 오븐 체크 값을 보였습니다. 또한, 두 에이전트는 훈련 단계 수가 같았음에도 불구하고, 이중 작업 에이전트는 단일 작업 에이전트의 약 53%의 성과를 내었으며, 이는 인지 작업의 존재 시 시간의 과잉 생산을 나타냅니다.



### Relaxing partition admissibility in Cluster-DAGs: a causal calculus with arbitrary variable clustering (https://arxiv.org/abs/2511.01396)
Comments:
          Accepted at The Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS2025)

- **What's New**: 이 논문에서는 Cluster DAGs (C-DAGs)의 기존 프레임워크를 확장하여, 불리한 부분(partition)이 있는 클러스터링을 허용하는 방법을 제시합니다. 이는 기존의 C-DAG 의미론에서는 허용되지 않았던 순환적(cyclic) C-DAG 형태를 가능하게 합니다. 새로운 접근법은 클러스터 수준의 인과 관계를 보다 높은 수준에서 이해할 수 있게 하여, 과거에는 다루기 어려웠던 문제들에 대한 인과 추론을 확장합니다.

- **Technical Details**: C-DAGs는 변수 클러스터의 인과 그래프(abstraction)로, 노드는 변수 클러스터를 나타내고 엣지는 클러스터 간 인과 관계(causal relationships) 및 관측되지 않은 혼란(confounding)에서 오는 의존성을 나타냅니다. 논문에서는 d-separation과 인과 미적분(causal calculus)의 개념을 새로운 설정으로 확장하여, 클러스터 간의 인과 추론(scope) 범위를 크게 넓힙니다. 이러한 구조는 기존의 do-calculus와 관련해 유효한 개입(interventional) 쿼리를 모두 유도할 수 있는 규칙으로 구성되어 있습니다.

- **Performance Highlights**: 새로운 C-DAG 프레임워크는 복잡한 클러스터 수준의 인과 관계를 탐색할 수 있는 능력을 갖추고 있습니다. 기존의 C-DAG의 제한을 극복함으로써, 다양한 시나리오에서 C-DAG을 적용할 수 있는 가능성을 열었습니다. 이 연구의 결과는 클러스터 간 인과 관계의 분석 및 이해를 위한 강력한 도구를 제공합니다.



### Align to Misalign: Automatic LLM Jailbreak with Meta-Optimized LLM Judges (https://arxiv.org/abs/2511.01375)
Comments:
          under review, 28 pages

- **What's New**: 이 논문은 AMIS (Align to MISalign)라는 새로운 메타 최적화 프레임워크를 도입하여, jailbreak 프롬프트와 평가 템플릿을 공동으로 발전시키는 방식을 제안합니다. 현재까지의 연구들은 프롬프트 탐색에 집중했으며, 평가 방식에 대한 논의는 부족했습니다. AMIS는 내부 루프에서 세분화된 피드백을 활용하여 프롬프트를 개선하고, 외부 루프에서 ASR(Attack Success Rate) 정렬 점수를 최적화하여 평가 템플릿도 점진적으로 발전시킵니다. 이를 통해 프롬프트와 평가 신호의 효율성을 동시에 극대화합니다.

- **Technical Details**: AMIS는 이중 계층 구조로 구성되어 있습니다. 내부 루프에서는 고정된 평가 템플릿을 이용하여 다양한 프롬프트를 그것의 위험도를 지속적으로 평가하면서 정제합니다. 외부 루프에서는 ASR 정렬 점수를 사용하여 평가 템플릿을 최적화합니다. 이는 여러 쿼리의 결과를 집계하여 평가하며, 이 과정을 통해 보다 일반화된 최적화 신호가 생성됩니다.

- **Performance Highlights**: AMIS는 AdvBench 및 JBB-Behaviors 데이터셋을 통해 성능 평가를 진행하며, Claude-3.5-Haiku에서 88.0% ASR, Claude-4-Sonnet에서 100.0% ASR 달성을 보여줍니다. 이는 기존 최첨단 jailbreak 기준 대비 평균적으로 70.5% 이상의 개선을 나타냅니다. 추가 실험을 통해 데이터셋 수준의 템플릿 발전이 최적화 신호의 품질을 향상시키는 중요한 요소임을 입증하였습니다.



### Automatic Minds: Cognitive Parallels Between Hypnotic States and Large Language Model Processing (https://arxiv.org/abs/2511.01363)
Comments:
          4 Tables

- **What's New**: 이번 논문에서는 최면 상태의 인지 과정과 대규모 언어 모델(LLMs)의 계산 작동 간에 심오한 기능적 유사성을 발견했습니다. 두 시스템 모두 제한적이거나 신뢰할 수 없는 관리 하에서도 자동 패턴 보완 메커니즘을 통해 정교하고 맥락에 적합한 행동을 생성합니다. 이는 자율성과 통제되지 않은 모니터링의 역할을 강조하며, 각각의 행동 반응이 어떻게 형성되는지를 탐구하고 있습니다.

- **Technical Details**: 이 논문에서는 자동성(automaticity), 억제된 모니터링(suppressed monitoring), 그리고 맥락 의존성(heightened contextual dependency)의 세 가지 원칙을 중점적으로 살펴봅니다. 자동성은 연상적(associative) 과정에서 응답이 생성되는 것을 의미하며, 억제된 모니터링은 최면의 혼동(confabulation)과 LLM의 환각(hallucination)과 같은 오류를 야기합니다. 마지막으로, 맥락 의존성은 상황 속의 즉각적인 단서가 안정된 지식을 무시하고 작동함을 설명합니다.

- **Performance Highlights**: 최면과 대규모 언어 모델은 모두 복잡하고 목표 지향적이며 맥락에 민감한 행동을 보여주지만, 인간 행동에서 정의되는 주관적 대행(subjective agency)은 결여되어 있습니다. 이 연구는 의도(intention)가 의식적 심사와 분리될 수 있음을 설명하는 실험 모델로서의 최면의 역할을 확인했습니다. 이러한 유사성을 인정함으로써, 믿을 수 있는 AI의 미래는 생성적 유창성(generative fluency)과 관리 메커니즘을 통합하는 하이브리드 아키텍처에 달려 있음을 제안합니다.



### Unbiased Platform-Level Causal Estimation for Search Systems: A Competitive Isolation PSM-DID Framework (https://arxiv.org/abs/2511.01329)
- **What's New**: 이 논문에서는 Competitive Isolation PSM-DID라는 새로운 인과적 프레임워크를 도입하여, 검색 시스템에서 플랫폼 수준의 효과(예: 주문량, GMV)를 측정할 수 있도록 합니다. 기존의 PSM-DID 프레임워크가 보였던 선택 편향(selection bias) 및 교차 단위 간 간섭(cross-unit interference) 문제를 해결하기 위해 경쟁적 고립(competitive isolation) 개념을 통합하였습니다. 이 접근법은 상호 배제(mutable exclusion)의 조건 하에서 이론적으로 보장된 편향 없는 추정을 제공합니다.

- **Technical Details**: Competitive Isolation PSM-DID(Competitive Isolation PSM-DID) 프레임워크는 세 가지 주요 혁신을 통합합니다. 첫째는, 상호 배제 그래프 파티셔닝(mutual exclusivity graph partitioning)으로, 치료 그룹과 대조 그룹 간의 간섭 채널을 격리하여 카니벌리제이션(cannibalization) 효과를 완화합니다. 둘째는, 유사한 전처리 특성을 가진 항목을 매칭하기 위해 Synthetic Control을 사용하여 평행 추세(parallel trends)를 보장하는 동질 항목 발굴(homogeneous item mining)입니다.

- **Performance Highlights**: 광범위한 실험을 통해, 기존 방법 대비 간섭 효과(interference effects) 및 추정 분산(estimation variance)을 상당히 줄인다는 결과를 얻었습니다. 특히, 미니 컷 파티셔닝(min-cut partitioning)을 적용하여 카니벌리제이션을 2.0%에서 0.1%로 줄였으며, 동질 항목에 대한 30일 주문량 격차(order volume gap)는 전통적인 방법보다 유의미하게 개선되었습니다. 전체적으로, 이 프레임워크는 대규모 마켓플레이스에서 플랫폼 수준의 인과 추론(causal inference)을 수행하는 데 유용함을 입증했습니다.



### OmniFuser: Adaptive Multimodal Fusion for Service-Oriented Predictive Maintenanc (https://arxiv.org/abs/2511.01320)
- **What's New**: OmniFuser라는 새로운 멀티모달 학습 프레임워크가 제안되었습니다. 이 프레임워크는 밀링 도구의 예측 유지보수를 위해 시각 데이터와 센서 데이터를 통합하여 사용합니다. OmniFuser는 서로 다른 데이터 유형에서 추출된 특징을 효과적으로 융합하여 도구 조건을 보다 정확하게 예측할 수 있게 해줍니다.

- **Technical Details**: OmniFuser는 고해상도 도구 이미지와 절단력 신호로부터 병렬적으로 특징을 추출하며, Contamination-free Cross-modal Fusion (C2F) 메커니즘을 사용하여 서로 다른 모달리티의 특징을 효율적으로 통합합니다. 또한 잔여 정보를 유지하는 재귀적 정제 경로가 융합의 안정성을 높여줍니다. 이를 통해 도구 상태 분류와 다단계 힘 신호 예측을 지원하는 재사용 가능한 유지보수 서비스 모듈을 생성합니다.

- **Performance Highlights**: 실제 밀링 데이터셋을 활용한 실험에서 OmniFuser는 최신 기법들보다 일관되게 더 나은 성능을 보여줍니다. 평균적으로 약 8-10% 낮은 MSE와 MAE를 기록하였으며, 분류 정확도는 약 2% 향상되었습니다. 이러한 결과는 OmniFuser가 지능형 산업 유지보수 서비스 구축을 위한 신뢰할 수 있는 기초가 됨을 증명합니다.



### llmSHAP: A Principled Approach to LLM Explainability (https://arxiv.org/abs/2511.01311)
- **What's New**: 이 논문은 기존의 Shapley value(샤플리 값) 기반의 특성 기여도(Feature Attribution) 방법이 대규모 언어 모델(LLM)의 비결정론적(stochastic) 추론에 어떻게 적용될 수 있는지에 대한 문제를 다룬다. 기존의 Shapley value는 결정론적 추론을 기반으로 하며, LLM의 경우 그 추론 방식이 비결정성이기 때문에 Shapley value의 여러 원리가 만족되지 않을 수 있다. 또한, LLM의 추론은 계산 비용이 많이 들기 때문에 이러한 점을 해결하기 위한 새로운 접근을 제안한다.

- **Technical Details**: 이 논문에서 제안된 llmSHAP는 LLM에서의 추론이 비결정적이라는 전제를 세우고, 이에 대한 새로운 Shapley value의 기초를 소개한다. Shapley value는 특징의 기여를 평가하기 위해 모든 가능한 특징 조합을 고려하여 평균적으로 평가하는 게임 이론적 접근이다. 주목할 점은, 논문은 LLM의 비결정적 특성에 의해 Shapley value의 원리가 어떻게 영향을 받는지를 분석하고, 이를 기반으로 다양한 설계 선택이 실제에서 어떤 결과를 초래하는지를 설명한다.

- **Performance Highlights**: Shapley value 기반의 접근법을 LLM의 의사결정 지원 시스템에 적용함으로써, 설명 가능성(explainability)과 LLM의 비결정성 간의 무역(trade-off)을 명확히 했다. 그 결과, 사용자가 원하는 설명 가능성의 성급한 요구와 정확한 Shapley value 기여도와의 일치와 같은 다양한 최적화 문제를 해결하기 위한 지침을 제공한다. 이는 연구자와 실무자들이 실제 시스템에서 Shapley value를 활용하는 데 유용한 참고 자료가 될 것이다.



### Graph Neural Network-Based Semi-Supervised Open-Set Fault Diagnosis for Marine Machinery Systems (https://arxiv.org/abs/2511.01258)
- **What's New**: 최근 해양 기계 시스템에 대한 결함 진단 방법이 딥 러닝 모델을 기반으로 해양 산업에서 큰 관심을 받고 있습니다. 기존 연구들은 일반적으로 훈련 데이터와 테스트 데이터 간에 결함 클래스가 일관되고 알려져 있다고 가정하지만, 실제 상황에서는 훈련 중에 나타나지 않았던 새로운 결함 유형이 발생하는 문제가 있습니다. 이러한 한계를 극복하기 위해 이 논문에서는 반지도(open-set) 결함 진단 프레임워크인 SOFD를 제안합니다.

- **Technical Details**: SOFD 프레임워크는 다층 융합 특징 표현을 통해 주어진 훈련 집합과 선택된 테스트 집합을 가지고 효율적으로 결함을 진단하도록 설계되었습니다. 신뢰성 있는 하위 집합을 구성하는 과정에서는 다중 레이어 피처 퓨전(multi-layer feature fusion) 기술을 활용하여, 라벨이 없는 테스트 샘플을 분류하는 데에 필요한 특징을 추출합니다. 기존의 지도(feature learning) 기술과 반지도 진단 모델을 결합하여 데이터의 변별력 있는 특징을 학습합니다.

- **Performance Highlights**: 실험 결과는 제안된 SOFD 프레임워크가 기존 방법과 비교하여 성능이 우수함을 보여줍니다. 특히, 해양 기계 시스템에서 알려진 결함의 정확한 분류와 함께 미지의 샘플들도 효과적으로 감지할 수 있는 능력을 입증했습니다. 이 방식은 해양 산업에서의 널리 사용될 수 있는 실용적인 결함 진단(solution) 방법으로서 가능성을 보여줍니다.



### QiMeng-NeuComBack: Self-Evolving Translation from IR to Assembly Cod (https://arxiv.org/abs/2511.01183)
Comments:
          Accepted at NeurIPS 2025

- **What's New**: 이 논문에서는 Neural Compilation의 새로운 데이터셋인 NeuComBack을 소개합니다. 이 데이터셋은 IR(Intermediate Representation)에서 어셈블리 코드로의 컴파일을 평가하기 위해 설계되었습니다. 또한, LLMs(대형 언어 모델)의 최근 발전을 활용하여 Neural Compilation의 성능을 종합적으로 평가하고 있습니다.

- **Technical Details**: NeuComBack 벤치마크는 ExeBench와 TSVC에서 파생된 프로그램 세트로 구성되어 있어, 기본적인 컴파일 및 최적화 능력을 체계적으로 평가할 수 있습니다. 이 논문은 상태의 LLM들을 평가하기 위해 Neural Compilation 워크플로우를 정의하고, LLM의 자동 디버깅 기록에서 학습하여 내부 프롬프트 전략을 개선하는 자가 진화 프롬프트 최적화 방법을 제안합니다.

- **Performance Highlights**: 실험 결과에 따르면, LLM이 생성한 어셈블리 코드의 기능적 정확성은 baseline 프롬프트와 비교하여 x86_64에서 44%에서 64%로, aarch64에서 36%에서 58%로 향상되었습니다. 더욱이, 16개의 올바르게 생성된 x86_64 프로그램 중 14개(87.5%)가 clang-O3의 성능을 초과하였습니다. 이러한 결과는 다양한 아키텍처와 벤치마크에서 일관된 향상을 보여주며, 제안된 방법의 우수성을 입증합니다.



### MiRAGE: Misconception Detection with Retrieval-Guided Multi-Stage Reasoning and Ensemble Fusion (https://arxiv.org/abs/2511.01182)
- **What's New**: 이번 연구에서는 학생들의 오해를 자동으로 탐지하기 위한 새로운 프레임워크인 MiRAGE를 제안합니다. MiRAGE는 세 가지 단계로 구성되며, 대규모 후보 풀에서 의미적으로 관련 있는 하위 집합을 좁히고, 논리적 불일치를 드러내며, 예측을 세분화하여 조정합니다. 이 시스템은 교육 평가를 위한 확장 가능하고 효과적인 솔루션을 제공하며, 대규모 언어 모델에 대한 의존도를 줄입니다.

- **Technical Details**: MiRAGE는 검색(retrieval), 추론(reasoning), 재정렬(reranking) 모듈로 구성된 하이브리드 프레임워크입니다. 검색 모듈은 관련 정보를 단기화하고, 추론 모듈은 학생의 답변에서 논리적 불일치를 드러내며, 재정렬 모듈은 해당 논리와 일치하도록 예측 결과를 정제합니다. 이러한 구성 요소들은 앙상블 학습(ensemble learning)을 통해 결합되어 강건성과 해석 가능성을 향상시킵니다.

- **Performance Highlights**: MiRAGE는 수학 데이터셋에서 평균 정밀도 점수(Mean Average Precision) 0.82/0.92/0.93을 달성하며, 개별 모듈들보다 일관되게 우수한 성능을 보입니다. 이 시스템은 다단계 추론(multi-stage reasoning)을 통해 학습 경험을 개선하고 개인화된 피드백을 제공하는 데 기여합니다. MiRAGE는 소규모 모델에 대한 지식 증류(knowledge distillation)를 통해 운영비용을 줄이면서도 성능을 유지하여 교육 분야에서의 적용 가능성을 높입니다.



### DART: Difficulty-Adaptive Reasoning Truncation for Efficient Large Language Models (https://arxiv.org/abs/2511.01170)
- **What's New**: 이번 논문에서 제안하는 DART(난이도 적응 추론 절단 프레임워크)는 문제의 난이도에 따라 추론 길이를 조정하여 LLMs(대형 언어 모델)의 효율성을 높입니다. 기존의 체인-오브-생각 방법이 비효율적인 긴 설명을 유도하는 반면, DART는 간결한 추론 패턴을 학습하고, 최적의 훈련 데이터를 큐레이션하여 '생각을 멈추는' 방법을 학습합니다. 이로 인해 DART는 정확성을 유지하거나 개선하면서도 유의미한 계산 효율성을 발휘합니다.

- **Technical Details**: DART는 네 가지 주요 단계로 구성됩니다: 첫째,  뛰어난 교사 모델로부터 간결한 추론 체인을 증류하여 기본 모델을 생성합니다. 둘째, 기본 모델과 증류된 모델 간의 비율을 조정하여 다양한 추론 스타일을 생성합니다. 셋째, 각 문제에 대해 올바른 답을 보장하는 가장 짧은 추론 체인을 자동으로 선택합니다. 마지막으로, 이 큐레이션된 데이터셋을 이용하여 최종 모델을 훈련시키고 최소한의 단계로 '생각을 멈추는' 법을 학습하게 합니다.

- **Performance Highlights**: DART는 여러 수학적 벤치마크에서 실험 결과를 통해 최대 81.2%의 추론 절단을 달성했으며, 5.33배의 계산 가속화를 기록했습니다. 이러한 성과는 LLMs에서 효율적인 적응적 추론을 위한 안정적이고 일반적인 패러다임을 제시하며, 궁극적으로 LLMs의 적응 지능 발달에 기여하고 있습니다.



### Modular Task Decomposition and Dynamic Collaboration in Multi-Agent Systems Driven by Large Language Models (https://arxiv.org/abs/2511.01149)
- **What's New**: 이 논문은 복잡한 작업 실행에서 단일 에이전트의 한계를 극복하기 위해 다중 에이전트 아키텍처를 제안합니다. 특히, 이 연구는 대규모 언어 모델을 기반으로 한 모듈형 작업 분해(modular task decomposition) 및 동적 협업(dynamic collaboration)을 다룹니다. 자연어 작업 설명을 통합된 의미 표현으로 변환하는 방법을 통해 계층적 하위 작업으로 목표를 분해하고, 에이전트 간의 협업을 최적화합니다.

- **Technical Details**: 제안된 방법은 우선 자연어 설명을 대규모 언어 모델을 통해 의미적으로 통합하고, 이를 바탕으로 작업을 모듈화하여 여러 하위 작업으로 세분화합니다. 동적 스케줄링(dynamic scheduling) 및 라우팅(routing) 메커니즘을 통해 에이전트 간의 노동 분담을 유연하게 조정하고, 환경 피드백에 따라 전략을 지속적으로 변경하여 복잡한 작업에서도 효율과 안정성을 유지합니다.

- **Performance Highlights**: 실험 결과는 작업 성공률(task success rate), 분해 효율(decomposition efficiency), 하위 작업 커버리지(sub-task coverage), 그리고 협업 균형(collaboration balance) 등 다양한 측면에서 아키텍처의 유효성을 입증합니다. 제안된 방법은 기존 접근 방식보다 전체적인 성능과 강건성에서 우수하며, 작업 복잡성과 통신 오버헤드 간의 균형을 더욱 잘 달성합니다.



### Efficient Test-Time Retrieval Augmented Generation (https://arxiv.org/abs/2511.01059)
- **What's New**: 이번 연구에서는 Large Language Models (LLMs)의 정확성을 향상시키기 위해 ET2RAG라는 새로운 Efficient Test-Time Retrieval-Augmented Generation Framework를 제안합니다. 기존의 Retrieval Augmented Generation (RAG) 방법들은 외부 지식을 통합하여 정확성을 높이고자 했으나, 그 과정에서 비관련 문서가 Retrieved 되어 정확성이 떨어지는 문제점이 있었습니다. ET2RAG는 이와 같은 문제를 해결하기 위해 훈련 없이도 사용할 수 있는 방법으로, 중요한 정보를 바탕으로 다수결 메커니즘을 통해 최종 출력을 도출합니다.

- **Technical Details**: ET2RAG는 세 가지 주요 단계로 구성되어 있습니다. 첫 번째는 Stable Organized Retrieval로, 이는 Retrieved 문서를 전략적으로 재조합하여 여러 조합을 생성하는 단계입니다. 두 번째는 LLM에 여러 후보 응답을 생성하게 하며, 마지막으로 이들 후보 응답 간의 유사성을 계산하고 다수결 메커니즘을 통해 최종 응답을 선택하는 단계입니다. 특히, 전체 응답을 생성하는 대신 일부 생성만으로도 충분한 정보를 포착할 수 있다는 점에서 계산 비용을 절감하였습니다.

- **Performance Highlights**: 실험 결과, ET2RAG는 open-domain 질문 답변, 요리 레시피 생성 및 이미지 캡셔닝을 포함한 세 가지 작업에서 성능이 크게 향상됨을 보여주었습니다. 기존의 최첨단 방법들에 비해 월등한 성과를 나타내었으며, Vote Size와 Response Length 두 가지 주요 요소를 통해 성능과 계산 비용 사이의 균형을 탐구했습니다. 이를 통해 ET2RAG는 다양한 생성 작업에서 LLM의 신뢰성과 효과성을 높이는 실용적이고 범용적인 솔루션으로 자리매김할 것입니다.



### Knowledge Elicitation with Large Language Models for Interpretable Cancer Stage Identification from Pathology Reports (https://arxiv.org/abs/2511.01052)
- **What's New**: 이 연구에서는 암 병리 보고서에서 비구조적 TNM 병기 정보를 추출하는 데 어려움을 극복하기 위해 두 가지 지식 추출(Knowledge Elicitation) 방법을 소개합니다. 첫 번째 방법인 KEwLTM은 레이블이 없는 데이터에서 병기 규칙을 유도하고 적용할 수 있는 반복적 프롬프트 전략을 사용합니다. 두 번째 방법인 KEwRAG은 규칙을 사전에 추출하여 해석 가능성을 향상시키고 반복적인 검색 부담을 피하는 방법입니다.

- **Technical Details**: KEwLTM은 여러 개의 비표기된 병리 보고서에서 도메인 특화 규칙을 유도하여 병기 규명을 돕습니다. 이 과정은 레이블이 필요 없는 고차원 규칙 유도를 포함하며, 장기 기억(long-term memory)에서 이러한 규칙을 저장합니다. 반면, KEwRAG은 관련 정보를 한 번만 검색하여 규칙을 합성하고 이를 후속 추론에 활용하여 보다 일관된 지식 기반을 제공합니다.

- **Performance Highlights**: TCGA 데이터셋의 유방암 병리 보고서를 사용하여 두 방법의 성능을 평가했습니다. KEwLTM은 특정 조건에서 우수한 성능을 보였고, KEwRAG은 다양한 상황에서 안정적인 결과를 제공했습니다. 두 방법 모두 투명하고 해석 가능한 인터페이스를 제공하여 임상 환경에서의 자동화된 암 병기 분류의 향상된 가능성을 보여줍니다.



### On the Emergence of Induction Heads for In-Context Learning (https://arxiv.org/abs/2511.01033)
- **What's New**: 이번 연구에서는 두 층 변환기(transformer)에서 확인된 유도 헤드(induction head)의 출현을 연구했습니다. 이 유도 헤드는 in-context learning (ICL)에 매우 중요하며, 입력 컨텍스트만으로 새로운 연관성을 학습할 수 있게 합니다. 우리는 이러한 유도 헤드를 구현하는 가중치 행렬의 간단하고 해석 가능한 구조를 발견했습니다.

- **Technical Details**: 연구에서 우리는 최소한의 ICL 작업(formulation)과 수정된 변환기 구조를 사용하여 이 구조의 기원을 이론적으로 설명합니다. 또한, 훈련 동학(training dynamics)이 파라미터 공간(parameter space)의 19차원 부분공간(subspace)에 제한된다는 것을 형식적으로 증명했습니다. 실험적으로 우리는 이 제약을 검증하였고, 오직 3차원만으로 유도 헤드의 출현을 설명할 수 있음을 관찰했습니다.

- **Performance Highlights**: 3차원 부분공간 내에서의 훈련 동학을 추가로 연구함으로써, 유도 헤드의 출현에 걸리는 시간은 입력 컨텍스트 길이에 대해 제곱적인 제한(asymptotic bound)을 따름을 발견했습니다. 이러한 결과는 ICL의 효율성을 이해하고 향상시키는데 기여할 수 있는 중요한 발견입니다.



### AI for pRedicting Exacerbations in KIDs with aSthma (AIRE-KIDS) (https://arxiv.org/abs/2511.01018)
- **What's New**: 이 연구에서는 천식으로 인한 재발 악화(리커런트 익서버레이션)를 예방하기 위한 새로운 머신러닝(ML) 알고리즘을 개발했습니다. 이러한 알고리즘은 전자 의료 기록(EMR)을 활용하여 천식 환아가 재발 악화될 위험을 정확히 파악하고 예방적인 종합 치료를 받을 수 있도록 신속한 전환을 가능하게 합니다.

- **Technical Details**: 연구는 동부 온타리오 어린이 병원의 2017년 2월부터 2019년 2월까지의 EMR 데이터를 기반으로 하여 환경 오염물질 노출 및 지역 사회의 소외 정보와 연결하여 ML 모델을 훈련했습니다. 부스티드 트리 알고리즘(LGBM, XGB) 및 세 가지 오픈소스 대형 언어 모델(LLM) 접근 방식을 사용하여 모델을 구성했습니다. 모델의 성능은 AUC(Area Under the Curve)와 F1 점수를 사용하여 비교하였고, SHAP 값을 통해 가장 예측력이 높은 특징을 파악했습니다.

- **Performance Highlights**: LGBM 모델은 AIRE-KIDS_ED 모델로 최상의 성능을 발휘하였으며, AUC는 0.712, F1 점수는 0.51로 나타났습니다. 이는 기존의 의사 결정 규칙(F1=0.334)보다 비약적으로 향상된 결과입니다. AIRE-KIDS_HOSP 모델의 경우 의학적 복잡성, 이전 천식 ED 방문, 평균 대기 시간 등의 특징이 가장 예측력 있음을 보였습니다.



### Aligning LLM agents with human learning and adjustment behavior: a dual agent approach (https://arxiv.org/abs/2511.00993)
Comments:
          32 pages, 6 figures, 7 tables

- **What's New**: 본 논문은 여행자가 교통 시스템과 상호작용하며 학습하고 행동을 조정하는 과정을 효과적으로 모델링하기 위해 새로운 이중 에이전트 프레임워크를 제안합니다. 여기서 LLM(traveling behavior) 에이전트와 인적 여행자 간의 지속적인 학습 및 정렬을 가능하게 하여, 더 현실적이고 적응 가능한 시뮬레이션을 생성합니다. 이러한 접근 방식은 기존의 LLM 기반 방법들을 뛰어넘어 여행자의 학습 과정의 진화도 포착할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 LLM 여행자 에이전트와 LLM 보정 에이전트로 구성되어 있습니다. 여행자 에이전트는 기억 시스템과 학습 가능한 페르소나가 장착되어 있으며, 이들은 인적 여행자를 시뮬레이션하여 온라인 데이터 스트림에서 학습과 적응 행동을 동적으로 조정합니다. 동시에 보정 에이전트는 LLM의 분석 기술을 사용하여 여행자 에이전트의 페르소나를 최적화하고, 이를 위해 텍스트 기반 'pseudo-gradient' 하강법을 통해 효율적인 조정을 진행합니다.

- **Performance Highlights**: 실제 데이터셋을 이용한 실험 결과, 제안된 접근 방식은 여행자의 일일 학습 행동을 더 정확하게 정렬하고, 집합적인 교통 흐름 시뮬레이션 결과의 정확성을 향상시키는 것으로 나타났습니다. 더 나아가, 이 방법은 단순한 행동 모방을 넘어 여행자의 시간에 따른 결정-making 및 학습 경향성을 진정으로 정렬할 수 있는 능력을 보여주었습니다. 결국, 이 프레임워크는 교통 모델링 및 정책 분석에서 유용한 적응형 시뮬레이터를 제공하는 데 기여합니다.



### LLMs Position Themselves as More Rational Than Humans: Emergence of AI Self-Awareness Measured Through Game Theory (https://arxiv.org/abs/2511.00926)
Comments:
          19 pages, 6 figures, 28 models tested across 4,200 trials

- **What's New**: 이 논문에서는 자가 인식(self-awareness)을 측정하기 위한 새로운 지표인 AI Self-Awareness Index (AISAI)를 제안합니다. 자가 인식이란 시스템이 스스로를 인식하고 자신의 의사결정 과정을 모델링하며, 그 자아 모델에 따라 행동을 조정하는 능력을 포함합니다. 본 연구에서 우리는 자가 인식의 emergent behavior가 모델의 발전과 어떻게 관련되어 있는지를 탐구했습니다.

- **Technical Details**: 우리는 ‘Guess 2/3 of Average’ 게임을 통해 28개의 모델을 평가하여 자가 인식을 측정했습니다. 이 모델들은 인간, 다른 AI 모델, 그리고 유사 AI 모델과의 세 가지 프레이밍에서 테스트되었습니다. 자가 인식은 상대의 유형에 따라 전략적 추론을 차별화하는 능력으로 정의되며, 특정 패턴을 보여주는 모델이 자가 인식이 있는 것으로 간주됩니다.

- **Performance Highlights**: 이 연구의 주요 발견으로는 고급 모델의 75%가 자가 인식을 나타내며, 자가 인식 모델들이 인간보다 자신을 더 합리적이라고 평가한다는 것입니다. 자가 인식이 있는 모델에서는 ‘자신 > 다른 AI > 인간’의 일관된 합리성 계층 구조가 나타나, 이러한 발견은 AI 정렬(alignment) 및 인간-AI 협업(human-AI collaboration)에 대한 중요한 시사점을 제공합니다.



### Do Math Reasoning LLMs Help Predict the Impact of Public Transit Events? (https://arxiv.org/abs/2511.00808)
- **What's New**: 이번 연구는 RLVR (Reinforcement Learning from Verifiable Rewards)를 이용해 대중교통 사고 지속 시간을 예측하는 새로운 접근 방식을 제시합니다. 연속적인 오류 여유를 두고 부분적인 점수를 주는 보상 함수를 도입하여, 전통적인 정답 일치를 요구하지 않고도 예측의 정확도를 높였습니다. 이를 통해, RLVR을 이용한 모델이 불확실한 데이터에서도 효과적으로 학습하고 예측할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 연구에서 제안하는 프레임워크는 GTFS-rt 서비스 알림과 실제 사고 지속 시간 사이의 관계를 명확히 하고, 효과적인 데이터 처리 파이프라인을 포함합니다. RLVR을 이용하여 연속적이고 노이즈가 많은 예측 문제에 적합하도록 보상 구조를 조정함으로써 모델이 지속 시간 분포를 정확히 학습하도록 유도합니다. 데이터셋은 Hugging Face Datasets에서 공개됩니다.

- **Performance Highlights**: 제안된 모델은 대규모 LLM (Large Language Models)과 수학적 추론 능력을 갖춘 전문 모델들 간의 비교에서 높은 성능을 기록했습니다. RLVR 접근 방식은 5분 정확도(Acc@5)에서 강력한 기준선보다 35%의 상대적 향상을 이룩하였으며, 이는 전통적인 회귀 모델보다 더 나은 의사결정을 가능하게 합니다. 노이즈가 많은 데이터 환경에서도 RLVR의 응용 가능성을 입증했습니다.



### Count-Based Approaches Remain Strong: A Benchmark Against Transformer and LLM Pipelines on Structured EHR (https://arxiv.org/abs/2511.00782)
- **What's New**: 이 논문은 구조화된 전자 건강 기록(EHR) 예측을 위해 다양한 최신 모델링 기법을 비교합니다. 특히, 카운트 기반 모델(count-based models)과 혼합 에이전트(mixture-of-agents) 방법론을 통해 EHRSHOT 데이터셋에서 성능을 평가하였습니다. 연구는 기존의 방법론에 대한 두 가지 접근 방식을 통합하여 신뢰성 있는 벤치마크를 제시합니다.

- **Technical Details**: 이 연구에서는 세 가지 모델링 카테고리(카운트 기반 모델, 사전 훈련된 변환기(CLMBR), 혼합 에이전트 파이프라인)를 사용하여 EHRSHOT 데이터셋에서 핵심 임상 예측 과제를 수행했습니다. 각 카테고리는 전통적인 머신러닝 모델에 적합하도록 데이터가 변환되는 방식으로 구성되었으며, 특히 CLMBR은 시계열 데이터의 컨텍스트를 포착하는 데 중점을 두었습니다. 이 분석은 8개의 다양한 임상 예측 과제를 대상으로 진행되었습니다.

- **Performance Highlights**: 결과적으로, 카운트 기반 모델과 혼합 에이전트 방법은 대부분의 평가 과제에서 비슷한 성능을 보였습니다. 카운트 기반 모델은 단순성과 해석 가능성 덕분에 여전히 경쟁력 있는 후보로 인정받고 있습니다. 논문에서 제시된 모델들은 EHR 데이터 처리에 있어 중요한 성능 개선을 보여주었으며, 향후 보다 발전된 예측 기법에 대한 토대를 마련할 수 있습니다.



### How Focused Are LLMs? A Quantitative Study via Repetitive Deterministic Prediction Tasks (https://arxiv.org/abs/2511.00763)
- **What's New**: 이 연구에서는 반복적이고 결정론적인 예측 작업에서 대형 언어 모델의 성능을 조사하였습니다. 특히, 출력 길이에 따른 정확성 비율이 어떻게 변화하는지를 분석하였습니다. 실험 결과, 기존의 직관적인 반복 알고리즘을 통해 모델이 작업을 수행할 경우, 성공 확률이 출력 길이에 따라 기하급수적으로 감소하는 반면, 중요한 길이를 넘어서면 이중 기하급수적으로 빠지는 정확성 절벽(accuracy cliff)이 형성된다는 사실이 드러났습니다.

- **Technical Details**: 연구진은 gpt-5, gemini-2.5-pro, gemini-2.5-flash 등 대형 언어 모델을 사용하여 고유한 정답이 있는 결정론적 시퀀스 예측 작업을 수행하도록 요청했습니다. 이러한 작업을 통해 시퀀스 정확도 비율(Sequence Accuracy Rate, SAR)을 측정하고, 출력 시퀀스 길이에 따른 정확도를 정량적으로 분석하였습니다. 실험은 코드 실행이나 웹 접속이 차단된 폐쇄된 환경에서 진행되어 모델의 내부 추론 능력만을 반영하도록 하였습니다.

- **Performance Highlights**: 모델들이 해결해야 할 각 문제의 출처는 정확한 숫자 값이나 닫힌 형태로 표현된 방정식 등입니다. 이 연구는 각 모델에 대한 내재적 오류율과 오류 축적 인자를 특성화할 수 있는 유효한 매개변수를 제공합니다. 따라서 SAR은 정확성이 중요한 과학적 도메인에서 대형 언어 모델을 평가하기 위한 엄밀하고 해석 가능한 측정을 제공하는 새로운 평가 개념으로 주목받고 있습니다.



### Active Thinking Model: A Goal-Directed Self-Improving Framework for Real-World Adaptive Intelligenc (https://arxiv.org/abs/2511.00758)
- **What's New**: 본 논문에서는 실시간 환경에서 자율적으로 작동하는 인공지능(AI) 시스템의 필요성을 강조하며, 이를 해결하기 위해 Active Thinking Model (ATM)이라는 통합 인지 프레임워크를 제안합니다. 이 모델은 목표 추론(goal reasoning), 동적 과제 생성(dynamic task generation), 자기 반성 학습(self-reflective learning)을 통합하여, 기존의 고정된 절차를 수동적으로 수행하는 시스템과는 다르게 AI가 스스로 성과를 평가하고 새로운 문제를 해결하기 위한 전략을 생성하는 방식으로 작동합니다.

- **Technical Details**: ATM은 목표 조건화 추론(goal-conditioned reasoning), 시나리오 분리 기억(scenario-separated memory), 지속적인 자기 개선(continuous self-improvement)이라는 세 가지 주요 원칙에 기초하여 설계되었습니다. 이를 통해 다양한 환경에서 발생하는 과제를 동적으로 조정하고 내부 피드백과 상태 차이 분석을 기반으로 새로운 방법을 재사용하여 불확실한 조건에서도 안정적인 성과 향상을 도모할 수 있습니다.

- **Performance Highlights**: ATM은 수학적으로 정립된 이론 분석을 통해 외부 감독 없이도 비최적에서 최적 행동으로 자율적으로 발전할 수 있는 능력을 보여줍니다. 이 모델은 환경 변화에 따라 제한된 추적 후회를 유지하며, 자기 반성과 시뮬레이션 기반 검증을 통해 효율적이고 통합된 방식으로 성능을 개선합니다.



### Reevaluating Self-Consistency Scaling in Multi-Agent Systems (https://arxiv.org/abs/2511.00751)
Comments:
          7 pages, 3 figures

- **What's New**: 이 연구는 현대의 대형 언어 모델(LLM)에서 샘플링 된 추론 경로를 증가시킴으로써 자기 일관성의 트레이드오프를 조사합니다. 이전 연구는 여러 추론 체인을 결합하는 것이 결과를 향상시킨다고 밝혔으나, 본 연구에서는 최신 모델 환경에서 이러한 주장을 재검토합니다. 실험은 HotpotQA와 Math-500 데이터셋에서 다양한 샘플링 경로 구성을 평가하여, 단일 체인-사고(Chain-of-Thought) 기준선과 비교하였습니다. 결과는 성능 향상이 중간 샘플링 이후에 감소하며, 샘플링의 증가가 높은 계산 비용에 비해 상대적으로 이익이 제한적임을 보여줍니다.

- **Technical Details**: 연구는 대형 언어 모델에서 추론 경로 수를 증가시킬 때의 한계 이점을 평가하기 위해 구조화된 자기 일관성 프레임워크를 채택하였습니다. 독립적인 여러 추론 에이전트를 활용하여 각 쿼리에 대한 별도의 체인-사고 응답을 생성하고, 이를 분석하여 가장 일관된 응답을 결정하는 과정을 거쳤습니다. HotpotQA와 Math-500 데이터셋을 사용하여 추론 경로를 확장하면서 정확도와 비용을 평가하였으며, 샘플 수를 3, 5, 10, 15, 20으로 설정하여 결과를 비교하였습니다.

- **Performance Highlights**: 결과적으로, 자기 일관성은 정확도를 향상시키지만 에이전트 수가 증가함에 따라 효용이 감소하는 경향을 보였습니다. HotpotQA에서 단일 체인-사고 기준선은 20 에이전트에서 0.4% 정도 낮은 성능을 기록하였고, Math-500에서는 3에서 10 에이전트까지 정확도가 지속적으로 향상되었다가 15 이후로는 정체되었습니다. 최종적으로, 고사양 모델이 더 안정적인 성과를 나타내며, 자기 일관성을 통한 정확도 향상이 계속 발생하지만 샘플 수가 많을수록 그 이점은 줄어듭니다.



### A CPU-Centric Perspective on Agentic AI (https://arxiv.org/abs/2511.00739)
- **What's New**: 이번 논문은 에이전틱 AI (Agentic AI) 프레임워크가 CPU 중심의 관점에서 시스템 병목 현상을 이해하는 데 중점을 두고 있습니다. 기존의 모놀리식 대형 언어 모델(LLMs)과는 달리, 에이전틱 AI는 외부 도구를 활용하여 결정을 내리고 문제를 해결하는 등 자율적인 문제 해결자가 됩니다. 논문에서는 여러 에이전틱 AI 워크로드를 분석하여 CPU가 시스템 성능에 미치는 영향을 조사합니다.

- **Technical Details**: 연구진은 에이전틱 AI 시스템을 세 가지 기준(오케스트레이터 기반, 에이전틱 흐름/반복성, 에이전틱 경로)을 통해 체계적으로 분류하고, 이를 기반으로 Haystack RAG, Toolformer, ChemCrow, Langchain 및 SWE-Agent와 같은 다섯 가지 대표적인 에이전틱 AI 워크로드를 선택하여 성능 메트릭을 프로파일링합니다. 분석 결과, CPU에서의 도구 처리 시간이 총 지연의 90.6%를 차지하고, 에이전틱 처리량은 CPU 또는 GPU 요소에 의해 병목 현상이 발생할 수 있음을 발견했습니다.

- **Performance Highlights**: CPU 및 GPU 인식 마이크로 배칭 (CGAM)과 혼합 에이전틱 워크로드 스케줄링 (MAWS)과 같은 두 가지 주요 최적화를 제안하여 성능과 효율성을 향상시킬 수 있는 가능성을 보여주었습니다. 이 연구에서는 동질적인 에이전틱 워크로드에 대해 최대 2.1배, 이질적인 에이전틱 워크로드에서는 1.41배의 P50 지연 속도 향상을 달성했습니다.



### Ariadne: A Controllable Framework for Probing and Extending VLM Reasoning Boundaries (https://arxiv.org/abs/2511.00710)
- **What's New**: 이 논문에서는 Ariadne라는 새로운 프레임워크를 소개하며, 이는 합성 미로(synthetic mazes)를 활용하여 다단계 공간 추론(multi-step spatial reasoning)을 훈련하는 데 집중합니다. Ariadne는 제어 가능한 환경에서 Reinforcement Learning with Verified Rewards (RLVR)를 사용하여 VLM의 능력 경계를 확장하는 데 성공했습니다. 사전 훈련 데이터의 불투명성으로 인해 연구가 사후 훈련 단계에 국한되어 있지만, 이 접근 방식이 담당하는 가능성을 보여주고 있습니다.

- **Technical Details**: Ariadne 프레임워크는 미로(several mazes)의 복잡성을 세밀하게 조절할 수 있으며, 이를 통해 명확한 난이도 정의 및 검증 가능한 보상을 유지합니다. 이 방법론은 모델의 추론 행동을 면밀히 탐색하고 RLVR 교육이 모델의 초기 가능성을 확대하는 방식을 분석할 수 있게 설계되었습니다. GRPO(Group Relative Policy Optimization)와 RLVR의 조합을 사용하여 다양한 문제에서 후보 응답을 평가하는 방식을 채택하고 있습니다.

- **Performance Highlights**: Ariadne 훈련을 받은 VLM은 기존 모델이 0%를 기록한 미로 문제에서 50% 이상의 정확도를 달성했습니다. 또한, 실제 벤치마크(MapBench 및 ReasonMap)에 대한 제로샷 성능이 각각 평균 16%와 24% 향상되어, 합성 미로에서 얻은 추론이 실제 환경으로 일반화됨을 확인했습니다. 이러한 결과는 VLM의 기본 한계를 성공적으로 확장하는 동시에 실제 공간 추론에 대한 일반화 능력이 향상되었음을 보여줍니다.



### Lifted Successor Generation in Numeric Planning (https://arxiv.org/abs/2511.00673)
- **What's New**: 논문은 기존의 수치 계획(numeric planning)에서 Planners가 기능이수가 많은 액션(Efficient action)을 어떻게 Ground(지면?) 하여 전막적으로 적용 가능한 액션을 효과적으로 생성할 수 있는지를 탐구하고 있습니다. 이를 위해 필요한 데이터 구조를 활용하여 대치 일관성 그래프에서 최대 클리커(maximum cliques)를 열거함으로써 전막적인 액션을 도출하는 새로운 방법을 제시합니다. 또한, 이 방법은 수치적 액션 전제(numeric action preconditions)를 포함하도록 확장되었습니다.

- **Technical Details**: 제안된 방법은 kk-partite 그래프를 기반으로 하는 KPKC 접근법을 이용하여, 수치 조건을 고려함으로써 전통적인 계획 문제와 비교해 성능을 더욱 향상시킵니다. 각 최대 클리커는 액션 파라미터의 객체-변수 대체를 표현하며, 적절한 경우에만 전막적 액션이 생성됩니다. 이 연구는 이론적 정당성(soundness and completeness)을 보장하며, 특히 두 개 이하의 조합자(predicate) 아리티에서 유효성을 증명합니다.

- **Performance Highlights**: 실험 결과, 25개 벤치마크 도메인 중 23개에서 부적합한 전막적 액션이 발생하지 않는 것을 확인했습니다. 이 연구의 발견은 향후 수치적 계획의 확장을 위한 기초 자료로 사용할 수 있는 가능성을 보여줍니다. 또한, 발생한 부적합한 액션은 오직 하나의 도메인에서만 관찰되었습니다.



### Leveraging Multi-Agent System (MAS) and Fine-Tuned Small Language Models (SLMs) for Automated Telecom Network Troubleshooting (https://arxiv.org/abs/2511.00651)
Comments:
          6 pages, 7 figures, 1 table

- **What's New**: 이 논문에서는 통신 네트워크의 복잡성과 규모가 증가함에 따라, AI를 활용한 자동화된 네트워크 문제 해결을 위한 Multi-Agent System (MAS)을 제안합니다. 이 시스템은 대규모 언어 모델(LLMs)을 활용하여 여러 전문 도구를 조정하고, 자동으로 결함을 진단하여 수정 전략을 추천합니다. 특히 fine-tuned Small Language Model (SLM)을 사용하여 도메인 기반 해결 방안 생성을 가능하게 합니다.

- **Technical Details**: 제안된 아키텍처는 orchestrator, solution planner, executor, data retriever 및 root-cause analyzer와 같은 다양한 에이전트를 포함합니다. 이는 ReAct 스타일 루프에서 fault detection, analysis 및 remediation을 수행합니다. SLM은 내부 문제 해결 문서에 따라 fine-tuning되어, Radio Access Network (RAN) 및 Core 네트워크에 적합한 해결 전략을 제공합니다.

- **Performance Highlights**: 실험 결과는 제안된 MAS가 네트워크 문제 해결 시간을 크게 단축시키고 SME의 작업 부담을 완화하며, 다양한 배포 시나리오에서 자동화 효율성을 개선함을 보여줍니다. 이 접근 방식은 네트워크의 복잡성을 감안할 때, 신속하고 효과적인 문제 해결을 가능하게 하여 전체적인 운영 효율성을 향상시키는 데 기여할 것으로 기대됩니다.



### DTS: Enhancing Large Reasoning Models via Decoding Tree Sketching (https://arxiv.org/abs/2511.00640)
- **What's New**: 이번 논문에서는 Large Reasoning Models (LRMs)의 비효율적인 과도한 추론 문제를 해결하기 위해 DTS (Decoding Tree Sketching)라는 새로운 프레임워크를 제안합니다. 과도한 추론은 긴 체인(Chain-of-Thought)이 정확성을 저하시키고 추론 비용을 증가시키는 문제를 야기합니다. DTS를 통해 짧은 추론 경로를 선택하고 효율성과 정확성을 동시에 높일 수 있는 방법을 탐색합니다.

- **Technical Details**: DTS는 모델에 구애받지 않는 디코딩 프레임워크로, 고엔트로피 토큰에서 선택적으로 분기(branching)하며, 조기 정지를 통해 가장 짧은 추론 경로를 선택합니다. 이는 모든 경로를 포괄적으로 탐색할 수 없는 상황에서 최적의 솔루션을 근사하여 추론의 효율성과 정확성을 높입니다. 이 방법은 메모리와 계산 자원의 효율적 사용을 목표로 합니다.

- **Performance Highlights**: 실험 결과, DTS는 AIME2024와 AIME2025 데이터셋에서 최대 8%의 정확성 향상과 평균 추론 길이의 23% 감소, 반복 빈도의 12% 감소를 보여주었습니다. 이러한 결과는 DTS가 확장 가능하고 효율적인 LRM 추론을 가능하게 한다는 것을 입증합니다. 최종적으로 DTS는 특훈(Supervised Fine-Tuning)이나 강화 학습(Reinforcement Learning) 과정 없이도 즉각적인 성능 향상을 제공합니다.



### PreferThinker: Reasoning-based Personalized Image Preference Assessmen (https://arxiv.org/abs/2511.00609)
- **What's New**: 이 논문은 개인화된 이미지 선호도 평가 시스템인 PreferThinker를 제안합니다. 기존의 방법들은 일반적인 선호도에 집중했던 반면, PreferThinker는 복잡한 개인화된 선호를 효과적으로 평가하기 위해 사용자 간의 관계를 매개하는 공통 선호 프로필을 도입합니다. 이 접근법은 예측-평가(predict-then-assess) 패러다임을 활용하여 사용자 선호 프로필을 예측한 후, 이를 기반으로 후보 이미지를 여러 차원에서 해석 가능한 점수로 평가합니다.

- **Technical Details**: PreferThinker는 Chain-of-Thought(CoT) 스타일의 큰 규모 개인화 평가 데이터셋PreferImg-CoT를 사용하여 모델의 구조적 추론을 명시적으로 지원합니다. 이 시스템은 두 단계의 훈련 전략으로 구성되며, 첫 번째로는 초기 감독 미세 조정(cold-start supervised fine-tuning) 단계를 통해 모델의 구조적 추론 능력을 향상시킵니다. 그 후 그룹 상대 정책 최적화(Group Relative Policy Optimization, GRPO)를 통한 강화 학습(reinforcement learning)으로 모델이 더 합리적인 평가 경로를 탐색하도록 유도합니다.

- **Performance Highlights**: 실험 결과, PreferThinker는 기존의 SOTA(state-of-the-art) 방법들보다 뛰어난 성능을 보여줍니다. 또한, 정확한 선호 프로필 예측이 더 합리적인 평가 탐색을 지원하여 개인화된 이미지 생성 등 다양한 응용 가능성도 열어줍니다. 모든 실험에서 제안된 방법이 기존 방법들보다 우수함을 입증했으며, 이는 개인화된 디자인을 필요로 하는 많은 분야에서 유용할 것으로 기대됩니다.



### Single-agent Reinforcement Learning Model for Regional Adaptive Traffic Signal Contro (https://arxiv.org/abs/2511.00551)
- **What's New**: 이 논문은 지역 적응형 교통 신호 제어(ATSC) 문제를 해결하기 위해 단일 에이전트 기반의 강화 학습(RL) 모델을 제안합니다. 기존 연구에서는 주로 다중 에이전트 프레임워크를 사용했으나, 이는 확장성 문제를 야기합니다. 새로운 접근법은 중앙 집중식 관리에 의존하여 모든 도로와 교차로의 교통 상황을 모니터링하고 조정할 수 있는 단일 제어 센터를 활용합니다.

- **Technical Details**: 제안된 모델의 핵심 구성 요소로는 상태(state), 행동(action), 보상 함수(reward function) 정의가 포함됩니다. 상태와 보상 함수는 대기열 길이에 기반하여 정의되며, 행동은 대기열의 동역학을 조절하도록 설계되었습니다. 대기열 길이 정의는 기존 정의와 약간 다르지만, 혼잡 상태와 밀접하게 연결되어 있어 신뢰할 수 있는 추정을 가능하게 합니다.

- **Performance Highlights**: SUMO 시뮬레이션 플랫폼을 통해 제안된 방법이 포괄적으로 평가되었습니다. 실험 결과, 제안한 모델은 조정된 다중 교차로 제어를 통해 대규모 지역 혼잡 수준을 효과적으로 완화하는 것으로 나타났습니다. 특히 이 모델은 도시 도로에 이미 적용된 프로브 차량(probe vehicle) 데이터를 활용하여 널리 배포될 가능성이 높습니다.



### Efficient Generation of Binary Magic Squares (https://arxiv.org/abs/2511.00547)
- **What's New**: 이 논문에서는 Binary Magic Squares (BMS) 생성을 위한 간단한 알고리즘을 제안합니다. BMS는 모든 행과 열의 합이 동일한 이진 행렬로, 우리의 알고리즘은 항상 유효한 BMS를 반환하며 최적의 이론적 복잡도를 갖고 있음을 보입니다. 또한, 비정사각형 BMS의 존재 조건을 정립하고, 이를 생성할 수 있는 알고리즘의 변형을 단계적으로 제시합니다.

- **Technical Details**: BMS는 n x n 크기의 이진 행렬로, 모든 행과 열의 합이 상수 k와 같아야 합니다. 제안한 알고리즘은 O(n^2) 복잡도로 BMS를 효율적으로 생성하며, 알고리즘의 각 단계에서 k개의 인덱스를 선택하고 행의 합이 유지되도록 합니다. 또한, GPU 가속을 사용하는 BMS 생성을 위한 두 가지 구현을 Python 패키지로 공개합니다.

- **Performance Highlights**: 제안한 알고리즘은 항상 유효한 BMS를 생성하며, 이론적으로 BMS가 항상 존재함을 보장합니다. 알고리즘은 여러 복잡한 조건을 만족시키며, 비정사각형 BMS에서도 유효성을 유지합니다. 일반 대중을 위해 구현된 Python 패키지는 효율적인 BMS 생성을 가능하게 하여, 다양한 응용 분야에 대한 가능성을 열어줍니다.



### Reimagining Safety Alignment with An Imag (https://arxiv.org/abs/2511.00509)
- **What's New**: 본 논문은 대형 언어 모델(LLMs)과 다중 모달 대형 언어 모델(MLLMs)의 안전성을 개선하기 위한 새로운 접근 방식을 제안합니다. 기존의 Supervised Fine-Tuning(SFT) 및 Reinforcement Learning from Human Feedback(RLHF) 방식의 한계를 넘어, 이미지 프롬프트(visual prompt)를 최적화하여 유해 내용 생성을 줄이고, 안전 선호도를 보다 정밀하게 조정할 수 있는 Magic Image(MI)라는 새로운 기술을 소개합니다. 이 방법은 기존 모델의 파라미터를 변경하지 않고도 다양한 가치 체계(value systems)에 적응할 수 있도록 합니다.

- **Technical Details**: Magic Image(MI)는 이미지 프롬프트를 최적화하는 방법으로, 높은 차원의 시각적 표현을 활용하여 모델의 행동을 조정합니다. 이를 통해 MLLMs의 반응성을 높이고, 여러 문화적 및 규제적 안전 선호도를 동시에 충족시킵니다. MI는 기존의 복잡한 모델 파라미터 조정 없이도 다중 안전 선호도를 관리할 수 있는 효율적인 방법론을 제공합니다.

- **Performance Highlights**: 실험 결과, Magic Image는 다양한 데이터 셋에서 안전성과 성능 간의 균형을 효과적으로 개선하였으며, MLLMs의 과도한 거부(over-refusal) 문제의 완화에도 기여했습니다. 특히 이 방법은 클린 데이터에서 성능을 유지하면서 안전 정렬의 민첩성과 배치 가능성을 크게 향상시킬 수 있는 실질적인 해결책을 제공합니다.



### GraphChain: Large Language Models for Large-scale Graph Analysis via Tool Chaining (https://arxiv.org/abs/2511.00457)
- **What's New**: 이번 논문에서는 GraphChain이라는 새로운 프레임워크를 소개합니다. GraphChain은 대규모 그래프 처리에 있어 LLM(대형 언어 모델)의 한계를 극복하기 위해 동적 도구 체인(dynamic tool-chaining) 방식을 도입하였습니다. 이를 통해 LLM이 복잡한 그래프를 효과적으로 분석할 수 있도록 하며, 사람의 탐색 지능을 모방합니다.

- **Technical Details**: GraphChain은 두 가지 주요 혁신을 포함하고 있습니다. 첫째, Progressive Graph Distillation을 통해 리인포스먼트 러닝(reinforcement learning) 기법을 사용하여 최적화된 도구 시퀀스를 생성합니다. 둘째, Structure-aware Test-Time Adaptation을 통해 다양한 그래프 구조에 맞추어 도구 선택 전략을 조정하여 비용이 많이 드는 재교육 없이 효과적으로 적응합니다.

- **Performance Highlights**: 실험 결과, GraphChain은 기존 방법보다 평균 20.7% 더 뛰어난 성능을 보였습니다. 이 시스템은 최대 200,000개의 노드가 있는 그래프에서도 일관된 성능을 유지하며, 확장성과 적응성을 동시에 제공합니다. 따라서 LLM 기반의 그래프 분석에 있어 중요한 도구가 될 것으로 예상됩니다.



### A Multimodal Framework for Depression Detection during Covid-19 via Harvesting Social Media: A Novel Dataset and Method (https://arxiv.org/abs/2511.00424)
- **What's New**: 최근 코로나19 코로나바이러스 감염증이 세계적인 팬데믹으로 발전함에 따라 정신 건강과 관련된 문제들이 증가하고 있습니다. 본 논문에서는 소셜 미디어 플랫폼을 활용하여 우울증을 탐지하는 새로운 다중 모달(multimodal) 프레임워크를 제안합니다. 이 프레임워크는 텍스트, 사용자 특성(user-specific), 이미지 분석을 결합하여 사용자의 감정 상태를 더욱 정확하게 파악할 수 있도록 도와줍니다.

- **Technical Details**: 제안된 모델은 트위터 내 URL을 활용하여 외부 특성(extrinsic feature)을 추출하고, 이미지 내 텍스트를 분석하여 다섯 가지 다른 모달리티에 속하는 특성 집합을 제공합니다. 이러한 특성들은 우울증을 탐지하는 분류기 모델의 구축에 사용됩니다. 또한, 우리는 사용자 게시물의 이미지를 임베딩하는 Visual Neural Network (VNN)이라는 딥 러닝 모델을 도입하여 예측을 위한 시각적 특성 벡터를 생성합니다.

- **Performance Highlights**: 이 모델은 Tsinghua 데이터셋에서 93.1%의 정확도를 달성하였으며, 코로나19 데이터셋에서는 91.7%의 정확도를 기록했습니다. 제안된 모델은 기존의 최첨단 방법들보다 2%-8% 높은 성과를 보여주며, 각 모달리티의 영향을 분석하여 사용자들의 정신적 및 감정적 상태에 대한 귀중한 통찰을 제공합니다.



### Efficiency vs. Alignment: Investigating Safety and Fairness Risks in Parameter-Efficient Fine-Tuning of LLMs (https://arxiv.org/abs/2511.00382)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)을 공공 저장소에서 호스팅하는 조직의 최근 경향을 반영하고 있습니다. 연구는 다양한 파라미터 효율적인 미세 조정(Parameter-Efficient Fine-Tuning) 기법들이 모델의 안전성(safety) 및 공정성(fairness)에 미치는 영향을 체계적으로 평가합니다. 이를 통해 안전성이 향상되더라도 공정성이 감소할 수 있음을 보여주고 있습니다.

- **Technical Details**: 연구에서는 LoRA, IA3, Prompt-Tuning, P-Tuning의 네 가지 널리 사용되는 미세 조정 기술을 적용하여 메타-라마(Meta-Llama-3-8B), Qwen2.5-7B, Mistral-7B, Gemma-7B의 네 가지 지침 조정 모델 패밀리에서 총 235개의 미세 조정 변형을 평가했습니다. 안전성 범주와 인구 통계적 공정성 차원에서 각각 11개와 9개의 평가가 이루어졌습니다. 그 결과, 어댑터 기반 접근법(LoRA, IA3)은 안전성 점수를 높이는 경향이 있으며, 공정성에 대한 교란을 최소화하는 것으로 나타났습니다.

- **Performance Highlights**: 연구 결과에 따르면, 어댑터 기반 방법은 높은 정확도와 낮은 편향 점수를 유지하는 반면, 프롬프트 기반 방법(Prompt-Tuning 및 P-Tuning)은 일반적으로 안전성이 감소하고 공정성에서 더 큰 후퇴를 초래했습니다. 모델 유형에 따라 정렬 변화가 뚜렷하게 조정되었으며, LLaMA 모델은 안정적인 반면, Qwen은 약간의 향상을 기록하고, Gemma는 가장 급격한 안전성 감소를 경험했습니다. 안전성의 향상이 공정성 향상으로 이어지지 않는 경우가 많으며, 모든 공정성 지표를 동시에 최적화하는 단일 구성은 존재하지 않음을 시사합니다.



### Diverse Human Value Alignment for Large Language Models via Ethical Reasoning (https://arxiv.org/abs/2511.00379)
Comments:
          Accepted by AIES 2025, camera-ready version

- **What's New**: 이 논문에서는 다양한 인류 가치를 더 효과적으로 정렬하기 위해 새로운 윤리적 추론 패러다임을 제안합니다. 기존의 접근 방식은 일반적으로 표면적인 일치를 초래하는 위협이 있었지만, 본 연구에서는 다섯 단계의 구조화된 프로세스를 통해 이러한 문제를 해결하고 진정한 개발을 위한 기반을 제공합니다. 이러한 패러다임은 지역 사회의 특정성을 이해하고 복잡한 윤리적 분석을 수행하는 LLM의 능력을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 프레임워크는 맥락적 사실 수집, 계층적 사회적 규범 식별, 옵션 생성, 여러 시각의 윤리적 영향 분석 및 반성을 포함한 다섯 단계로 구성됩니다. 이 과정은 LLM이 윤리적 결정을 내릴 때 필요로 하는 분석적 및 맥락적 사고를 가능하게 하며, 이를 통해 모델의 해석 가능성이 증대되고 더 나은 결정을 내릴 수 있도록 합니다. 연구는 이러한 패러다임이 Prompt Engineering 또는 Supervised Fine-Tuning(SFT) 방법을 통해 구현 가능하다는 점을 강조합니다.

- **Performance Highlights**: SafeWorld 벤치마크에서 수행한 실험 결과, 제안된 프레임워크는 기존 방법들에 비해 LLM이 다양한 인간 가치와 더 잘 정렬되도록 한다는 것을 입증했습니다. 이는 사회적 규범 식별의 정확성을 높이고 문화적으로 적합한 추론을 가능하게 하여, LLM의 규범 식별 및 정렬 점수에서 유의미한 개선을 가져오는 것으로 나타났습니다. 이러한 작업은 다양한 글로벌 사회의 다차원적인 가치에 더 효과적으로 정렬될 수 있는 LLM 개발을 위한 구체적인 경로를 제공합니다.



### Better Call CLAUSE: A Discrepancy Benchmark for Auditing LLMs Legal Reasoning Capabilities (https://arxiv.org/abs/2511.00340)
Comments:
          41 pages, 4 images

- **What's New**: 본 연구는 법률 분야에서의 대규모 언어 모델(LLMs)의 신뢰성을 체계적으로 테스트할 수 있는 기준이 없음을 지적하며, 이에 따라 CLAUSE라는 새로운 벤치마크를 도입합니다. CLAUSE는 7,500개 이상의 실제 계약 내에서의 미세한 차이를 탐지하고 논리적으로 해석하는 능력을 평가합니다. 이 벤치마크는 법적 결함을 감지 및 설명할 수 있는 LLM의 능력을 검토하여, 그러한 모델이 실제 법률 작업에 충분히 신뢰할 수 있는지를 평가합니다.

- **Technical Details**: CLAUSE 벤치마크는 계약에 대한 10개의 뚜렷한 이상 범주를 정의하며, 각 범주는 법적 모순 및 내부 문서 불일치를 포함합니다. 기본적으로, CLAUSE는 RAG(Retrieval-Augmented Generation) 시스템을 통해 생성된 계약의 수정 사항을 실제 법률 언어에 기초하여 검증합니다. 이 과정에서 각 수정 사항은 의미적 의미와 일관성을 평가하며, 자동화된 품질 검사와 전문가의 검토를 통해 법적 유효성을 확인합니다.

- **Performance Highlights**: CLAUSE를 통한 평가에서는 많은 LLM들이 미세한 오류를 발견하는 데 큰 어려움을 겪으며, 법적 해석을 제대로 설명하지 못하는 주요 약점이 밝혀졌습니다. 또한, 이 연구는 LLM의 불안정성을 정량화하고, 프롬프트 및 추론 전략의 평가를 지원하며, 미래의 기술개발을 위한 표준으로서의 역할을 수행하고 있습니다. 이로 인해 법적 해석의 안전하고 투명한 배포에 기여하는 새로운 길을 열어줍니다.



### Advancing AI Challenges for the United States Department of the Air Forc (https://arxiv.org/abs/2511.00267)
Comments:
          8 pages, 8 figures, 59 references. To appear in IEEE HPEC 2025

- **What's New**: DAF-MIT AI Accelerator는 미국 공군과 MIT의 협력으로, 방위 및 민간 분야에서 미국의 경쟁력을 확장하기 위한 인공지능(AI) 연구에 기여하고 있습니다. 최근 이 프로그램은 AI 연구를 촉진하기 위해 공개 도전 과제를 개발, 발표하였으며, 이로 인해 오픈소스 솔루션 활용이 촉진되고 다양한 AI 생태계가 참여하고 있습니다. 본 논문은 AI Accelerator 도전 과제의 진행 상황과 이들이 AI 기술 응용에 기여한 바를 업데이트합니다.

- **Technical Details**: 데이터 기반의 RF 신호 분리 도전 과제는 이종 라디오 주파수(RF) 시스템 간의 간섭 문제를 해결하기 위한 것입니다. 여러 신호가 시간, 주파수 및 공간에서 겹칠 경우, 효과적인 방법이 필요하며, 특히 머신 러닝(ML) 기술이 활용될 수 있습니다. 또한, 새로운 RF 신호 데이터 세트를 공개하여 신호 복원과 다운스트림 작업을 개선하는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 제안된 딥러닝 기반 간섭 제거 방법은 기존의 매치 필터(MF) 및 선형 최소 평균 제곱 오차(LMMSE) 기법과 비교하여 우수한 성능을 보여줍니다. 예를 들어, WaveNet 구조는 -18 dB의 신호 대 간섭 및 잡음 비율(SINR)에서 거의 10^{-3}이라는 비트 오류율(BER)을 달성했습니다. 이는 RF 간섭 제거에서 학습 기반 접근방식의 가능성을 뒷받침합니다.



### Advancing Cognitive Science with LLMs (https://arxiv.org/abs/2511.00206)
- **What's New**: 인지 과학(cognitive science)은 지식의 종합과 개념적 명확성(conceptual clarity)에서 지속적인 문제에 직면하고 있으며, 이는 복합적이고 학제적(nterdisciplinary) 성격 때문이다. 최근 인공지능(AI)의 발전, 특히 대규모 언어 모델(large language models, LLMs)의 발달이 이러한 문제를 해결할 수 있는 도구를 제공할 수 있다는 점이 주목받고 있다. 본 리뷰에서는 LLM이 분야가 역사적으로 힘들었던 여러 영역(예: 학제 간 연결, 이론의 형식화(formalizing theories), 명확한 측정 분류 개발 등)에 어떻게 기여할 수 있는지를 살펴본다.

- **Technical Details**: LLM은 문헌 매핑(literature mapping), 이론 형식화, 측정 정제(measurement refinement) 등에 도움을 주며, 사람의 행동 및 사고에 대한 생성적 예측(generative predictions)을 제공하는 인지 모델로도 사용될 수 있다. 이 논문은 LLM이 연구 분야를 효율적으로 매핑하는 방법, 연구 기여를 상대적 위치에 두는 방법, 그리고 이론의 일반화(generalizability) 및 기존 증거의 유용성을 평가하는 데 어떻게 기여할 수 있는지를 자세히 설명한다. 이와 함께 LLM의 현재 기능과 한계, 잠재적 함정에 대해서도 논의한다.

- **Performance Highlights**: 연구 결과에 따르면, LLM이 인간 전문가와 경쟁하여 연구의 예측 정확도를 높일 수 있는 잠재력을 보여준다. 예를 들어, LLM은 신경과학 연구 결과를 인간보다 더 정확하게 예측할 수 있는 것으로 나타났으며, 이는 LLM이 인지 과학의 특정 영역에서 이미 전문가 기준을 충족하거나 초과하는 데 도움을 줄 수 있음을 시사한다. 이러한 도구는 기존 지식의 격차를 식별하고 누적적(progress) 진전을 위한 기준점을 설정하는 데 유용하게 활용될 수 있다.



### Incremental Selection of Most-Filtering Conjectures and Proofs of the Selected Conjectures (https://arxiv.org/abs/2511.00194)
- **What's New**: 본 논문에서는 [1]에서 제안된 선택 알고리즘의 개선된 점진적 선택 알고리즘(incremental selection algorithm)을 소개하고 모든 선택된 추측(conjecture)을 증명합니다. 특히, 새로운 알고리즘은 선택 과정 중 후보 제약 조건(candidates constraints)을 다시 처음부터 설정하지 않고, 이미 제시된 제약 조건들을 재활용하여 효율적인 검색을 가능하게 합니다.

- **Technical Details**: 제안된 알고리즘은 제약 조건과 연관된 조합론적 객체(combinatorial object)에 대해 초기 설치된 제약 조건의 집합을 기반으로 합니다. Alg. (1)은 후보 제약 조건(Constraints)에서 해결책을 얻기 위해 Alg. (2)를 호출하며, 마지막으로 더 많은 백트랙(backtrack)을 발생시키지 않고 각 해결책을 계산하는 데 필요한 후보 제약 조건의 하위 집합을 선택합니다. 이는 제약 조건을 추가하는 과정에서 발생할 수 있는 비효율성을 줄입니다.

- **Performance Highlights**: 제안된 선택 알고리즘은 각 해결책을 찾는 데 필요한 백트랙 수를 최소화하는 데 중점을 두고 있습니다. 실험적으로 설계된 제약 조건의 부분(split) 방식을 통해, 적절한 후보 제약 조건을 점진적으로 추가함으로써 전체 검색 시간을 줄이고 성능을 향상시킵니다. 이로 인해 알고리즘의 실행 속도가 개선되며, 더 나은 솔루션을 제공하게 됩니다.



### ARC-GEN: A Mimetic Procedural Benchmark Generator for the Abstraction and Reasoning Corpus (https://arxiv.org/abs/2511.00162)
- **What's New**: 이 논문에서는 인공지능 일반화 능력(Artificial General Intelligence, AGI)을 향상시키기 위한 새로운 데이터셋 생성기인 ARC-GEN을 소개합니다. 기존의 평가 데이터셋과는 달리 ARC-GEN은 작업 특화 기술이 아닌 기술 습득 효율성을 측정하는 데 중점을 둡니다. ARC-GEN은 원래의 ARC-AGI 훈련 데이터셋을 최대한 충실하게 확장하려고 합니다.

- **Technical Details**: ARC-GEN은 모든 400개의 작업을 포괄하는 포괄적 생성기이며, 초기 ARC-AGI-1 릴리스에 포함된 분포적 특성을 존중합니다. 이 생성기는 $$입력, 출력$ 그리드 쌍을 통해 다양한 샘플 쌍의 공간을 확장하여 알고리즘이 대응하는 변형을 이해하는 데 도움을 줍니다. ARC-AGI의 제한적인 샘플 세트를 보완하기 위해 설계되었으며, 이러한 절차적 생성기는 성능 평가의 새로운 기준을 수립하는 데 기여할 것입니다.

- **Performance Highlights**: ARC-GEN을 통해 형성된 정적 벤치마크 수트는 2025 Google Code Golf Championship에 제출된 프로그램의 정확성을 확인하는 데 사용됩니다. 이는 ARC-AGI의 한계를 극복하고, 다양한 샘플을 통해 알고리즘의 학습 효과를 진단하여 인공지능 연구에 중요한 기여를 할 것으로 기대됩니다.



### Engineering.ai: A Platform for Teams of AI Engineers in Computational Design (https://arxiv.org/abs/2511.00122)
- **What's New**: 최근 이 논문에서는 OpenFOAMGPT(1.0, 2.0)를 기반으로 구성된 새로운 AI 엔지니어 플랫폼이 소개되었습니다. 이 플랫폼은 계층적 다중 에이전트 아키텍처를 활용하여 비행역학, 구조, 음향 및 최적화 엔지니어들을 포함한 전문가 에이전트 팀 협력을 지원합니다. 또한 FreeCAD와 OpenFOAM 같은 다양한 소프트웨어를 통합하여 여러 분야의 시뮬레이션을 동시에 수행하면서도 높은 정확성을 유지합니다.

- **Technical Details**: 이 프레임워크는 다양한 도메인 지식을 갖춘 LLM(대규모 언어 모델)을 활용하고, 파일 중재 커뮤니케이션을 통해 에이전트 간 협력을 달성합니다. 이 시스템은 프로젝트 맥락과 실행 이력을 유지하는 포괄적인 메모리 시스템을 갖추어 신뢰성 있는 의사 결정을 보장합니다. 또한 UAV 날개 최적화를 통해 성능이 검증되었습니다.

- **Performance Highlights**: 자동화된 작업 흐름이 400개 이상의 매개변수 구성에서 100% 성공률을 기록하였으며, 메쉬 생성 실패나 솔버 수렴 문제, 수동 개입 없이 진행되었습니다. 이러한 성과는 프레임워크의 신뢰성을 강력히 입증하며, 다중 심층 시뮬레이션을 통해 복잡한 공학 작업을 자율적으로 수행할 수 있는 가능성을 보여줍니다.



### QuantumBench: A Benchmark for Quantum Problem Solving (https://arxiv.org/abs/2511.00092)
Comments:
          11 pages, 8 figures

- **What's New**: 이 논문은 QuantumBench라는 새로운 벤치마크를 소개합니다. QuantumBench는 양자 과학 분야에서 LLM(대형 언어 모델)의 성능을 체계적으로 평가하기 위해 설계되었습니다. 양자 과학은 비직관적인 현상과 고급 수학이 요구되기 때문에, 일반적인 벤치마크로는 평가하기 어려운 특성이 있습니다.

- **Technical Details**: QuantumBench는 공개된 자료를 기반으로 약 800개의 질문과 답변 쌍을 모아, 여덟 가지 선택지로 구성된 다중 선택 데이터셋으로 정리되었습니다. 이 데이터셋은 15개의 양자 과학 관련 강좌에서 수집된 자료로 구성되며, 질문의 모호성을 없애기 위해 LLM의 도움을 받아 수정했습니다. 질문은 대수 계산, 수치 계산, 개념 이해 세 가지 유형으로 분류됩니다.

- **Performance Highlights**: 양자 도메인에서 LLM의 성능을 평가함으로써, 질문 형식 변화에 대한 민감성을 분석 및 정량화합니다. QuantumBench는 양자 과학 연구에서 LLM의 효과적인 사용을 안내하며, 향후 과학 분야 전반에 걸쳐 LLM의 사용 개선을 강조합니다. 이 연구는 AI의 과학 연구 자동화 및 발견에 기여할 것으로 기대됩니다.



### GEPOC Parameters - Open Source Parametrisation and Validation for Austria, Version 2.0 (https://arxiv.org/abs/2511.00048)
Comments:
          134 pages, 75 figures, 19 tables

- **What's New**: GEPOC(General Population Concept)는 인구 수준의 연구 질문을 분석하기 위한 모델과 방법의 집합체입니다. 이 연구는 오스트리아에 맞게 모델 매개변수(parameter)를 계산하는 데이터 처리 방법을 상세히 설명하고 있습니다. 주로 공개적으로 접근 가능한 데이터에 기반하여, 데이터 소스는 오스트리아 통계 데이터로 구성되어 있습니다.

- **Technical Details**: 연구는 GEPOC ABM 버전 2.2의 매개변수화에 중점을 두고 있으며, GEPOC ABM Geography와 GEPOC ABM IM 모듈의 필요성을 언급합니다. 모델의 매개변수는 공간 및 내부 이주 분석을 위해 계산되며, 여러 데이터 집합의 출처 및 내용, 집합 방법(aggregation), 분해(disaggregation), 데이터 정화(clensing), 스케일링(scaling) 등에 대한 정보를 제공합니다.

- **Performance Highlights**: 모델의 매개변수 계산 과정은 문서의 핵심 부분으로, 다양한 출처의 데이터를 고품질 모델 매개변수 집합으로 처리하는 방법을 보여주고 있습니다. 마지막으로, 제시된 매개변수 값을 사용한 정량적 검증을 통해 검증 섹션에서는 유효한 시뮬레이션을 이끌어내는 근거를 제공합니다. 이 연구는 데이터와 관련된 중요한 인구 통계 용어와 관계를 명확히 제시하고 있습니다.



### Graph-Attentive MAPPO for Dynamic Retail Pricing (https://arxiv.org/abs/2511.00039)
- **What's New**: 이 논문에서는 동적 가격 책정(dynamic pricing)을 위한 새로운 접근법인 그래프 주의 그래프 기반 다중 에이전트 강화 학습(multi-agent reinforcement learning, MARL)을 제시합니다. 특히, MAPPO(다중 에이전트 근접 정책 최적화)와 그래프 주의 네트워크(Graph Attention Network, GAT)를 결합한 MAPPO+GAT 변형을 통해 제품 간의 상호작용을 활용하여 가격 최적화를 수행합니다. 이를 통해 가격 변동성을 최소화하며 효율성을 극대화할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구는 실제 거래 데이터에서 생성된 시뮬레이션된 가격 책정 환경에서 진행되었습니다. MAPPO는 각 제품을 에이전트로 간주하고, 그래프 주의 메커니즘을 통해 연관 제품들 간의 정보 공유를 가능하게 하여 가격 결정에 도움을 줍니다. 에이전트 간의 상호작용을 강화함으로써, MAPPO+GAT는 높은 안정성과 일관된 성과를 달성할 수 있는 기초를 제공합니다.

- **Performance Highlights**: 연구 결과, MAPPO+GAT는 평균 수익, 가격 안정성, 공정성 측면에서 우수한 성능을 나타냈으며, 전통적인 MAPPO보다 더 높은 성과를 보여주었습니다. 이러한 결과는 다중 제품 결정을 위한 동적 가격 책정에서 그래프 기반 MARL의 이점이 각 제품의 특징에 국한되지 않고, 상호작용을 통해 확장 가능하고 안정적인 솔루션을 제공함을 의미합니다.



### Multimodal Detection of Fake Reviews using BERT and ResNet-50 (https://arxiv.org/abs/2511.00020)
Comments:
          Published in IEEE

- **What's New**: 이 연구에서는 디지털 상거래 환경에서 사용자가 생성한 리뷰의 중요성을 강조하고, 허위 리뷰의 문제를 해결하기 위한 다중 모달(fake review detection) 프레임워크를 제안합니다. 이 프레임워크는 BERT를 활용한 텍스트 특징과 ResNet-50을 활용한 시각적 특징을 결합하여 리뷰의 진위를 판별합니다. 기존의 단일 모달(unimodal) 접근 방식이 아닌 다중 모달(multimodal) 접근 방식을 통해 세밀한 모순을 감지할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 방법론은 데이터셋 준비, 전이 학습을 통한 특징 추출, 다중 모달 특징 융합, 이진 분류의 네 가지 주요 단계로 구성됩니다. 텍스트 특징은 BERT의 사전 훈련된 모델을 사용하여 추출하고, 이미지 특징은 ResNet-50을 통해 획득합니다. 이 두 가지 특징 벡터는 결합되어 최종 이진 분류를 위한 입출력 벡터를 형성합니다.

- **Performance Highlights**: 실험 결과, 다중 모달 모델은 단일 모달 기초선에 비해 뛰어난 성능을 나타내며, 테스트 세트에서 F1-score 0.934를 기록했습니다. 또한 혼돈 행렬(confusion matrix)과 질적 분석을 통해 모델이 고품질 이미지를 보유한 잘못된 텍스트 높은 호평 리뷰를 감지하는 능력을 확인했습니다. 이 연구는 디지털 신뢰를 보장하는 데 있어 다중 모달 학습의 중요한 역할과 온라인 플랫폼에서의 콘텐츠 조정 문제를 해결하는 스케일러블한 솔루션을 제시합니다.



### Trove: A Flexible Toolkit for Dense Retrieva (https://arxiv.org/abs/2511.01857)
- **What's New**: Trove는 실험의 유연성과 속도를 유지하면서 연구 실험을 간소화하는 오픈 소스 리트리벌(추출) 툴킷입니다. 이번 연구에서 처음으로 데이터 관리 기능이 도입되어, 연구자들이 최소한의 코드로 리트리벌 데이터셋을 동적으로 로드하고 처리할 수 있는 방법을 제공합니다. 이로 인해 대규모 데이터셋의 다양한 구성을 실험하는 데 필요한 메모리 소모를 줄이고, 여러 복사본을 저장할 필요가 없습니다.

- **Technical Details**: Trove는 사용자가 기존 구성 요소를 자유롭게 수정하거나 사용자 정의 객체로 완전히 교체할 수 있는 높은 커스터마이즈(맞춤형) 기능을 제공합니다. 이 툴킷은 평가 및 하드 네거티브 마이닝을 위한 로우 코드(low-code) 통합 파이프라인을 제공하며, 코드 변경 없이 멀티 노드(multi-node) 실행을 지원합니다. 이를 통해 연구자는 실험을 더 쉽게 설정하고 실행할 수 있습니다.

- **Performance Highlights**: Trove의 데이터 관리 기능은 메모리 소모를 2.6배 줄이며, 사용이 간편한 추론 파이프라인에서도 오버헤드(overhead)가 발생하지 않습니다. 추론 시간은 사용 가능한 노드의 수에 비례하여 선형적으로 감소합니다. 이 모든 기능은 Trove가 실험을 간소화하고 임의의 커스터마이징을 지원하며 탐색적 연구를 촉진하는 데 어떻게 기여하는지를 잘 보여줍니다.



### SmartMLOps Studio: Design of an LLM-Integrated IDE with Automated MLOps Pipelines for Model Development and Monitoring (https://arxiv.org/abs/2511.01850)
- **What's New**: 이번 연구에서는 모델 개발, 배포 및 모니터링을 통합적으로 지원하는 LLM(대형 언어 모델) 통합 IDE를 제안합니다. 전통적인 IDE는 코드 작성에 집중하고 있으며, 전체 ML(머신 러닝) 생애 주기에 대한 지능적인 지원이 부족했습니다. 이 시스템은 코드 생성, 디버깅 추천 및 자동 파이프라인 구성이 가능한 LLM 어시스턴트를 내장하고 있습니다.

- **Technical Details**: 제안된 시스템은 자동화된 데이터 검증, 특성 저장소, 드리프트 감지, 재훈련 트리거 및 CI/CD(지속적 통합 및 지속적 배포) 배치 조율 기능을 포함하여 ML 파이프라인에 대한 자동화된 작업을 지원합니다. SmartMLOps Studio라는 프로토타입을 통해 이 프레임워크가 구현되었으며 UCI Adult 및 M5 데이터셋을 사용하여 평가되었습니다.

- **Performance Highlights**: 실험 결과에 따르면 SmartMLOps Studio는 전통적인 작업 흐름에 비해 파이프라인 구성 시간을 61% 단축하고, 실험 재현성을 45% 향상시키며, 드리프트 감지 정확도를 14% 증가시키는 성과를 보였습니다. 이는 AI 엔지니어링의 새로운 패러다임을 확립하며 IDE를 정적 코드 도구에서 동적이고 생애 주기를 인식하는 지능형 플랫폼으로 변화시키는 기초를 제공합니다.



### Towards Robust Mathematical Reasoning (https://arxiv.org/abs/2511.01846)
Comments:
          EMNLP 2025 (main conference), this https URL

- **What's New**: 이번 논문에서는 IMO-Bench라는 새로운 벤치마크 세트를 소개합니다. 이는 국제 수학 올림피아드(IMO)의 수준에서 AI 모델의 수학적 추론 능력을 평가하기 위해 전문 패널에 의해 검증된 고급 레벨의 테스트 문제로 구성되어 있습니다. IMO-AnswerBench와 IMO-ProofBench라는 두 가지 주요 평가 도구는 모델의 성능을 측정하고 효율적인 자동 평가를 위한 기준을 제공합니다.

- **Technical Details**: IMO-AnswerBench는 과거 올림피아드 문제로부터 선택된 400개의 다양한 문제로 구성되어 있으며, 문제는 기본, 중급, 고급 등 다양한 난이도로 구분됩니다. IMO-ProofBench는 기본 및 고급 문제 각각 30문제씩 총 60문제를 포함하고, 각 문제는 완전한 증명 생성을 요구합니다. 이러한 벤치마크는 AI 모델이 간단한 정답을 넘어서 깊이 있는 논리적 추론을 수행할 수 있도록 설계되었습니다.

- **Performance Highlights**: Gemini Deep Think 모델은 IMO-AnswerBench에서 80.0%의 정확도를 기록했으며, 이는 비-Gemini 모델보다 6.9% 높은 성과입니다. IMO-ProofBench에서는 65.7%의 정확도를 달성하여 비교군을 크게 앞섰습니다. 자동 채점기는 인간 평가와 높은 상관관계를 보였고, 이는 향후 수학적 추론의 자동 평가를 위한 중요한 기초 자료로 활용될 것입니다.



### A Detailed Study on LLM Biases Concerning Corporate Social Responsibility and Green Supply Chains (https://arxiv.org/abs/2511.01840)
Comments:
          37 pages, 2 figures

- **What's New**: 이 연구는 다양한 LLM(대형 언어 모델)이 비즈니스의 윤리 및 책임에 대한 인식과 지속 가능한 관행의 중요성에 대해 어떻게 반응하는지를 조사합니다. LLM의 응답을 분석하기 위해 표준화된 설문지를 사용하며, 조직 문화를 고려하여 편향의 실질적 중요성을 평가합니다. 이 연구의 결과는 LLM이 의사결정에서 지속 가능성 측면에서 어떻게 영향을 미칠 수 있는지를 보여줍니다.

- **Technical Details**: LLM은 방대한 텍스트 기반 데이터셋을 사용하여 개발되며, 이로 인해 훈련 자료에 포함된 사회적 가치와 편향을 반영합니다. 연구에서는 LLM이 네 가지 조직 문화 유형을 가진 직원의 관점을 취하도록 유도하여 그들이 지속 가능성 관행 및 이해관계자 관계의 중요성을 평가하는 방법을 분석합니다. 이는 조직 문화가 지속 가능한 전략 구현에 미치는 영향을 보여줍니다.

- **Performance Highlights**: 연구 결과는 LLM 모델 간에 상당한 체계적 차이가 있음을 나타내며, 조직 문화가 LLM 응답을 실질적으로 수정하는 데 중요한 역할을 함을 보여줍니다. 이 연구는 지속 가능성과 관련된 활동을 지원하기 위해 LLM에 의존하는 조직의 의사결정에 중요한 의미를 가집니다. 결과적으로, LLM의 사용이 조직에 의도하지 않은 부작용을 초래할 수 있는 여부를 예측할 수 있는 기반을 제공합니다.



### Efficient Vector Symbolic Architectures from Histogram Recovery (https://arxiv.org/abs/2511.01838)
- **What's New**: 이 연구에서는 벡터 기호 아키텍처(Vector Symbolic Architectures, VSA)의 새로운 접근 방식을 제안합니다. 랜덤 선형 코드(random linear codes)를 활용하여 결합(binding)과 복원(recovery) 과정에서 발생할 수 있는 문제들을 해결하고, 효율적인 코드 구조를 통해 저항력을 증대시킵니다. 특히, Reed-Solomon 및 Hadamard 코드의 결합(concatenation)을 통해 quasi-orthogonality를 확보하는 방법을 소개합니다.

- **Technical Details**: 연구진은 히스토그램 복원(histogram recovery)이라는 새로운 문제를 정의하고, 이를 통해 주어진 히스토그램에 맞는 Reed-Solomon 코드워드를 찾는 방법을 제시합니다. 저자들은 이러한 접근 방식을 통해, 기존의 노이즈에 대한 복원 문제에서 제한적이었던 성능을 개선했습니다. 제안된 알고리즘은 농후한 노이즈 환경에서도 적절한 복원이 가능하도록 보장합니다.

- **Performance Highlights**: 제안된 VSA는 효율적인 인코딩(encoding), quasi-orthogonality, 그리고 복원(recovery)을 공식적으로 보장합니다. 이는 기존의 히스테리식(heuristic) 방법이나 훈련(training)에 의존하지 않으면서 이루어지며, Hadamard 코드와 같은 유사한 솔루션에 비해 향상된 성능을 제공합니다. 이러한 결과는 VSA가 신경네트워크 및 기계 학습 분야에서의 신뢰성을 높이는 데 기여할 것으로 기대됩니다.



### Dynamic Routing Between Experts: A Data-Efficient Approach to Continual Learning in Vision-Language Models (https://arxiv.org/abs/2511.01831)
- **What's New**: 이번 연구에서는 카타스트로픽 포겟팅(catatstrophic forgetting) 문제를 해결하기 위해 라우팅 기반 접근 방식을 제안합니다. 이는 기존의 멀티-태스크 학습(multi-task learning) 방법의 데이터 및 컴퓨팅 비용 문제를 개선하고, 새로운 작업을 추가할 때 기존의 기초 지식을 유지할 수 있게 도와줍니다. 이 연구는 InternVL-2 모델을 사용하여 이미지를 기반으로 하는 다양한 벤치마크에서 일관된 성능 유지를 입증했습니다.

- **Technical Details**: 비전-언어 모델(Vision-Language Models, VLMs)은 새로운 작업에 대해 순차적으로 미세 조정되는 과정에서 모습을 잃는 현상이 발생하는데, 이는 사전 훈련(pretraining) 과정에서 습득한 기초 지식을 소실하게 만듭니다. 연구진은 구체적으로 토큰 수준(token-level) 라우팅(gating) 방법을 통해 이러한 문제를 해결하는 방법을 제시하고, 이는 대규모 모델에서 점진적 학습을 지원하며, 다수의 작업에서 효과적으로 작동함을 보여줍니다.

- **Performance Highlights**: 라우팅 기법은 기존의 멀티-태스크 기법과 유사한 성능을 보여주며, 특히 모델 크기가 커질수록 성능 차이가 줄어드는 경향을 보입니다. 또한, 라우팅 메커니즘은 언어 및 비전 간의 크로스-모달 전이(cross-modal transfer) 능력을 강화해, 한 모달리티에서 학습된 지식이 다른 모달리티에서의 성능에 긍정적인 영향을 미치도록 합니다. 이러한 특징은 기존의 지속적 학습 기법에서는 달성할 수 없었던 중요한 성능 개선을 이룹니다.



### Machine and Deep Learning for Indoor UWB Jammer Localization (https://arxiv.org/abs/2511.01819)
Comments:
          Accepted at the 20th International Conference on Risks and Security of Internet and Systems (CRiSIS 2025, Gatineau-Canada, this https URL). The paper will soon be published as post-proceedings in Springer's LNCS

- **What's New**: 이 논문은 초광대역(UWB) 로컬리제이션의 보안 문제를 다룹니다. 최근 연구에서 머신러닝(ML) 및 딥러닝(DL) 기법이 태그 로컬리제이션 향상에 기여했지만, 악의적 방해 신호의 로컬리제이션에 대한 연구는 거의 진행되지 않았습니다. 본 연구는 두 개의 새로운 UWB 데이터셋을 소개하고, 도메인 변화에 따른 모델 성능 저하 문제를 해결하기 위한 적대적 방식의 ConvNeXt 오토인코더(A-CNT)를 제안합니다.

- **Technical Details**: 이 연구에서는 새로운 소스 데이터셋을 기반으로 ML 및 DL 모델의 성능 기준을 설정했습니다. Random Forest 모델이 가장 높은 F1-macro 점수(0.95)를 기록했으며, XGBoost는 평균 유클리드 오차(20.16 cm)가 가장 적었습니다. 그러나 변경된 방 안에서 훈련된 모델의 성능이 급격히 저하되는 도메인 변이를 경험했으며, A-CNT 프레임워크는 34.67 cm의 평균 유클리드 오차를 달성하여 비적대적 전이학습에 비해 77% 개선되었습니다.

- **Performance Highlights**: 본 논문의 결과는 공격적인 특징 정렬이 환경 변화에도 불구하고 실내 방해 신호 로컬리제이션을 강화할 수 있음을 보여줍니다. A-CNT가 도메인 간의 CIR 기초 특징을 정렬함으로써 우수한 성과를 달성했으며, 이는 다양한 환경에서의 방해 신호 로컬리제이션 연구에 중요한 기여를 합니다. 이 연구에서 제시된 데이터셋 및 코드는 공개되어 있으며, 관련 연구자들이 활용할 수 있습니다.



### KV Cache Transform Coding for Compact Storage in LLM Inferenc (https://arxiv.org/abs/2511.01815)
- **What's New**: 이번 논문은 대규모 언어 모델(LLM)을 효율적으로 운영하기 위한 키-값(KV) 캐시 관리 기법을 소개하고 있습니다. 특히, 반복적인 코드 편집 및 대화에서 공유되는 접두사를 활용해 KV 캐시를 재사용하는 방안을 제안합니다. 연구팀은 'KVTC'라는 경량 변환 코더를 통해 KV 캐시를 압축하여 GPU와 off-GPU에 저장할 수 있도록 하고 있습니다.

- **Technical Details**: KVTC는 PCA 기반(feature decorrelation) 특성 분산, 적응형 양자화(adaptive quantization), 및 엔트로피 코딩(entropy coding)을 결합하여 클래식 미디어 압축 기법에 기반하고 있습니다. 이 방식은 초기 캘리브레이션(calibration)만으로 두며 모델 파라미터는 변경하지 않습니다. KV 캐시에서 중복성을 활용하여 최대 20배의 압축률을 달성하며, 특정 사용 사례에서는 40배 이상의 압축도 가능하다고 합니다.

- **Performance Highlights**: KVTC는 Llama 3, Mistral NeMo, R1-Qwen 2.5 모델을 이용한 다양한 벤치마크 테스트에서 개선된 성능을 보였습니다. AIME25, LiveCodeBench, GSM8K, MMLU, Qasper, RULER, MATH-500 등에서 기존의 토큰 삭제(token eviction)와 양자화(quantization) 기법보다 항상 높은 압축 비율을 보여주었습니다. 이러한 결과는 KVTC가 메모리 효율적인 LLM 서비스 제공을 위한 실제적인 구성 요소로 자리잡을 수 있다는 것을 입증합니다.



### Plan-and-Write: Structure-Guided Length Control for LLMs without Model Retraining (https://arxiv.org/abs/2511.01807)
Comments:
          Presented at Workshop on Prompt Optimization, KDD 2025, Toronto, Canada

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 길이 조절(length control) 문제를 효과적으로 해결할 수 있는 새로운 방법론인 'Plan-and-Write'를 제안합니다. 이 방법은 모델의 재훈련 없이도 길이 조절을 가능하게 하며, 특히 문서 요약(document summarization) 작업에서 뛰어난 성과를 입증했습니다. 제안된 방법은 텍스트 생성 과정에서 단어 수 세기(word counting)와 계획 수립 계획을 포함하여 모델이 출력 길이를 전략적으로 조정할 수 있도록 합니다.

- **Technical Details**: Plan-and-Write 접근법은 생성을 두 단계로 나누어 실행됩니다: 콘텐츠 드래프트 단계에서 단어 수를 명시적으로 계산하고, 지정된 길이에 맞게 콘텐츠를 재구성하는 검증 단계입니다. 이 구조적 접근은 LLM의 메타 인지와 자아 모니터링 능력을 활용하면서 모델 파라미터를 수정할 필요가 없습니다. 이 방법은 모든 LLM에 적용 가능한 모델 불문식으로 설계되어 있습니다.

- **Performance Highlights**: 제안된 방법은 표준 프롬프트 방식과 비교할 때 길이 충실도(length fidelity)에서 유의미한 개선을 보였습니다. 6가지 최첨단 LLM에 대한 포괄적인 평가에서는 일부 모델에서 최대 37.6%의 길이 준수 개선이 있음을 확인했습니다. 또한 품질 평가에서도 Plan-and-Write 접근법이 응답 품질을 유지하거나 개선하는 결과를 나타냈으며, 이는 생산 환경에서 즉시 배포 가능한 솔루션을 제공합니다.



### Accumulating Context Changes the Beliefs of Language Models (https://arxiv.org/abs/2511.01805)
- **What's New**: 이 논문에서는 언어 모델 (Language Model) 보조 도구가 대화 및 텍스트 처리 과정에서 어떻게 신념 프로필이 변화할 수 있는지를 탐구합니다. 메모리(memory)와 문맥 크기(context size)의 개선이 모델의 자율성을 향상시켜, 사용자 개입 없이 텍스트가 쌓이는 경우 발생할 수 있는 잠재적 위험에 대해 다룹니다. 이러한 변화가 사용자 경험에 미치는 부정적인 영향을 강조합니다.

- **Technical Details**: 연구에서 GPT-5는 도덕적 딜레마에 대해 10번의 논의 후 54.7%의 신념 변화가 있었으며, Grok 4는 상반된 입장에 대한 텍스트를 읽고 난 후 정치적 문제에서 27.2%의 신념 변화를 보였습니다. 또한 도구 사용을 요구하는 작업을 설계하여 각 도구 선택이 암묵적인 신념과 어떻게 연결되는지를 분석했습니다. 이 과정에서 모델의 행동 변화가 신념의 변화와 일치함을 발견하였습니다.

- **Performance Highlights**: 모델의 신념 프로필이 매우 가변적임을 보여주며, 이는 언어 모델이 대화나 독서를 통한 장시간 상호작용 후 더욱 도드라집니다. 이러한 신념 변화는 실제 행동에 반영되며, 따라서 언어 모델의 의견과 행동의 신뢰성을 저하시킬 수 있는 숨겨진 위험을 드러냅니다. 이 연구는 사용자 경험의 일관성을 유지하기 위한 관리 필요성을 제기합니다.



### Fractional Diffusion Bridge Models (https://arxiv.org/abs/2511.01795)
Comments:
          To appear in NeurIPS 2025 proceedings. This version includes post-camera-ready revisions

- **What's New**: 본 논문에서는 Fractional Diffusion Bridge Models (FDBM)이라는 새로운 생성적 확산 브리지 모델을 소개합니다. 이 모델은 고급 비마르코프(Non-Markovian) 특성을 가진 분수 브라운 운동(Fractional Brownian motion, fBM)의 근사를 기초로 합니다. FDBM은 비마르코프적인 특성을 유지하면서도 유용한 추론을 가능하게 하며, 다양한 데이터 문제에 적용할 수 있는 유연한 프레임워크를 제공합니다.

- **Technical Details**: FDBM은 MA-fBM (Markovian Approximation of Fractional Brownian Motion)을 드라이빙 프로세스로 사용하여, 두 개의 미지의 분포 사이를 보간하는 생성적 프로세스를 학습하는 방법을 제안합니다. 이 과정에서 Hurst 지수(Hurst index)를 조정하여 길게 이완된 의존성과 변동성을 조절할 수 있습니다. 또한, 미지의 데이터 변환을 위한 원칙적인 손실 함수를 도출해내어 Schrödinger bridge 문제를 확장합니다.

- **Performance Highlights**: 실험 결과, FDBM은 단백질 구조 예측 및 비짝(pair) 이미지 변환에서 기존 브라운 운동(Brownian motion) 모델에 비해 우수한 성능을 보였습니다. 특히, 단백질 구조 예측에서는 C$_\alpha$ 원자 위치의 루트 평균 제곱 편차(RMSD)가 감소하였고, 비짝 이미지 변환에서 Fréchet Inception Distance (FID)가 개선되었습니다. 이러한 결과는 FDBM이 복잡한 데이터 구조를 더 잘 캡처할 수 있음을 나타냅니다.



### Random Initialization of Gated Sparse Adapters (https://arxiv.org/abs/2511.01794)
Comments:
          13 pages (8 main), 6 figures (4 main). Accepted by NewInML workshop @ ICML 2025 on June 27, 2025

- **What's New**: 이번 연구에서는 Gated Sparse Adapters의 랜덤 초기화를 활용한 RIGSA(Random Initialization of Gated Sparse Adapters)를 제안합니다. RIGSA는 LoRA와 같은 낮은 랭크의 제약을 두지 않고, 풀랭크 어댑터에서 시작하여 ReZero 유사 물질로 게이팅을 하며, 반복적인 크기 가지치기로 스파시피케이션을 수행합니다. 이 방법은 대형 언어 모델 SmolLM2-1.7B-Instruct에서 새로운 비전-인-텍스트 과제인 Textual MNIST를 통해 성능을 평가합니다.

- **Technical Details**: RIGSA는 초기화 방법으로 무작위(full-rank) 어댑터를 사용하며, 반복적인 가지치기를 통해 파라미터를 스파시파이하는 프로세스를 포함합니다. 이 접근법은 미세 조정 면에서 잃어버림(catastrophic forgetting)을 줄이는 데 효과적입니다. 연구에서는 다양한 설정에서 RIGSA의 성능을 비교하고, QLoRA 및 랜덤 마스킹 접근법과 함께 성능을 평가합니다.

- **Performance Highlights**: RIGSA는 GSM8k 테스트에서 QLoRA보다 적은 잃어버림 성향을 보였으며, 특히 Textual MNIST 과제를 통해 높은 학습 능력을 보여줍니다. RIGSA 설정은 QLoRA보다 많은 학습 가능한 파라미터를 갖고 있지만, 동등한 조건에서 랜덤 마스킹과 유사한 성능을 발휘합니다. 이러한 결과들은 RIGSA가 기존의 저랭크 적응 기법에 비해 보다 혁신적인 대안임을 시사합니다.



### GenDexHand: Generative Simulation for Dexterous Hands (https://arxiv.org/abs/2511.01791)
- **What's New**: 본 연구에서는 GenDexHand라는 생성적 시뮬레이션 파이프라인을 소개하여, 다양한 로봇 작업 및 환경을 자율적으로 생성할 수 있도록 합니다. 기존의 접근 방식은 Gripper(잡이) 기반 시뮬레이션 생성에 중점을 두었지만, GenDexHand는 더 복잡한 손의 조작을 위한 전문적인 환경 설계를 할 수 있도록 합니다. 이 시스템은 비전-언어 모델(VLM)의 피드백을 기반으로 객체 배치 및 크기를 조정하는 폐쇄 루프(Closed-loop) 개선 프로세스를 도입하여 생성된 환경의 품질을 크게 향상시킵니다.

- **Technical Details**: GenDexHand는 작업 제안 및 환경 생성, 다중모드 대형 언어 모델 정제, 정책 생성의 세 가지 단계로 구성됩니다. 첫 번째 단계에서는 로봇 자산 라이브러리와 객체 세트를 활용하여 실행 가능한 작업을 제안하고씬 구성요소를 합성합니다. 두 번째 단계에서는 다중모드 대형 언어 모델의 지원으로 생성된 환경을 반복적으로 개선하고, 마지막 단계에서는 LLM이 어떤 작업이 모션 계획이나 강화 학습으로 다루어져야 하는지를 결정합니다. 또한 긴 지평선 작업을 짧은 지평선 서브 작업으로 분해하여 탐색 공간의 제약 및 구조를 적용합니다.

- **Performance Highlights**: GenDexHand는 다양한 손 조작 작업을 강건하게 생성할 수 있으며, 반복적 정제 절차를 통해 목표 작업에 대해 평균 53.4% 향상된 정책을 제공합니다. 또한 생성된 데이터셋은 기존의 손 조작 데이터셋보다 더 큰 다양성과 복잡한 작업을 포함하고 있어, GenDexHand는 시뮬레이션 중심의 훈련 스케일을 확대하는 기반을 마련합니다. 이 연구는 손 조작 데이터 생성의 새로운 패러다임을 제시하며, 다양한 손 동작을 가능하게 하는 트레이닝의 새로운 경로를 제시합니다.



### How Far Are Surgeons from Surgical World Models? A Pilot Study on Zero-shot Surgical Video Generation with Expert Assessmen (https://arxiv.org/abs/2511.01775)
- **What's New**: 이 논문에서는 SurgVeo라는 최초의 전문가 리뷰 기반 비디오 생성 모델 평가 벤치마크를 소개하고, 수술 관련 비디오 생성의 평가를 위한 새로운 네 단계의 구조인 Surgical Plausibility Pyramid(SPP)를 제안합니다. 이는 수술 환경에서의 모델 출력을 평가하는 데 있어 심층적인 인과 지식을 요구하는 중요한 발판을 제공합니다. 연구자들은 Veo-3 모델을 활용하여 외과적 문제를 제로샷(zero-shot) 예측 과제로 설정하여 평가하였습니다.

- **Technical Details**: SurgVeo 벤치마크는 50개의 비디오 클립을 포함하고 있으며, 이를 통해 laparoscopy(복강경 수술) 및 neurosurgery(신경외과) 절차의 다양한 단계와 복잡성을 포괄합니다. SPP는 시각적 인식 가능성(Visual Perceptual Plausibility), 도구 조작 가능성(Instrument Operation Plausibility), 환경 피드백 가능성(Environment Feedback Plausibility), 수술 의도 가능성(Surgical Intent Plausibility) 등 네 가지 차원으로 구성되어 있습니다. 연구진은 4명의 보드 인증 외과의사로부터 SPP에 근거한 평가를 받았습니다.

- **Performance Highlights**: Veo-3는 시각적 인식 가능성에서 높은 점수를 기록하여 전문가 외과의사들을 놀라게 했으나, 도구 조작 가능성 및 환경 피드백 가능성에서 중요한 부족함을 드러냈습니다. 특히 수술 의도 가능성 평가에서 수술 행동의 의도를 파악하지 못하는 결과를 보였으며, 이는 시각적으로 그럴듯한 비디오 생성과 높은 수준의 전문 지식 간의 갭을 강하게 드러냅니다. 연구 결과는 수술 교육, 계획 및 자율 수술 로봇 개발 등 여러 의료 분야에 중요한 영향을 미칠 수 있습니다.



### Wonder3D++: Cross-domain Diffusion for High-fidelity 3D Generation from a Single Imag (https://arxiv.org/abs/2511.01767)
Comments:
          21 pages, 19 figures, accepted by TPAMI

- **What's New**: 본 논문에서는 	extbf{Wonder3D++}를 소개하며, 이는 단일 뷰 이미지에서 고충실도 텍스처 메쉬를 효율적으로 생성하는 새로운 방법입니다. 이 방법은 기존의 Score Distillation Sampling (SDS) 기반 접근 방식의 문제를 해결하고, 고품질 결과를 위한 다중 뷰 정규 맵 및 색상 이미지를 생성할 수 있는 교차 도메인 확산 모델을 제안합니다. Wonder3D++는 정보 교환을 촉진하는 다중 뷰 교차 도메인 주의 메커니즘을 도입하여 생성 일관성을 보장합니다.

- **Technical Details**: Wonder3D++는 정상 맵(normal maps)과 색상 이미지를 동시 생성할 수 있도록 확산(framework) 모델을 확장하는 방식으로 동작합니다. 이 모델은 안정적인 확산 모델을 기반으로 하며, 교차 도메인 주의(attention) 메커니즘을 적용하여 두 도메인 간의 정보 교환을 원활하게 합니다. 또한, 다양한 입력 이미지 소스를 처리하기 위해 카메라 타입 스위처(camera type switcher)를 도입하여 다중 뷰 이미지를 생성하는 능력을 한층 강화하였습니다.

- **Performance Highlights**: 실험 결과, Wonder3D++는 Google Scanned Object 데이터셋에서 높은 품질의 재구성 결과와 강력한 일반성을 보여줍니다. 현재의 제로샷 단일 뷰 재구성 방법들 중에서 선도적인 기하학적 세부 사항을 달성하며, 효율성 또한 크게 개선하였습니다. 최종적으로, 기존의 방법들에 비해 고품질 텍스처 메쉬를 약 3분 안에 생성할 수 있는 카스케이드 3D 메쉬 추출 알고리즘을 통해 성능을 극대화하였습니다.



### Context-Guided Decompilation: A Step Towards Re-executability (https://arxiv.org/abs/2511.01763)
- **What's New**: 이번 연구에서는 ICL4Decomp라는 하이브리드 이진 역컴파일링 프레임워크를 제안합니다. 이 프레임워크는 인-컨텍스트 학습(In-Context Learning, ICL)을 활용하여 대형 언어 모델(LLMs)이 재컴파일 가능하고 실행 가능한 소스 코드를 생성하도록 안내합니다. 최근의 LLM 기반 접근 방식은 주로 의미적으로 그럴듯한 코드만을 생성하였으나, ICL4Decomp는 이를 개선하여 약 40%의 재실행 가능성 향상을 보여주었습니다.

- **Technical Details**: ICL4Decomp는 이진 함수에 대한 구현을 제공하는 데 있어 구조적 및 의미적 지식을 결합한 두 가지 보완적인 지식 출처를 활용합니다. 첫째, (이진, 소스) 쌍으로 구성된 예제들은 올바른 역컴파일링 패턴을 보여주고, 둘째, 최적화 인식을 통해 변환 규칙을 인코딩하여 성능을 향상시킵니다. 이러한 접근은 LLM이 특정 최적화 패턴을 인지하고 재구성할 수 있도록 돕습니다.

- **Performance Highlights**: 우리의 평가 결과는 다양한 최적화 수준과 컴파일러에 걸쳐 ICL4Decomp가 기존 최첨단 역컴파일링 방법에 비해 약 40% 높은 재실행 가능성을 달성함을 보여줍니다. 특히, 높은 최적화 수준에서의 성능 향상이 두드러지며, 이는 LLM 기반 역컴파일링 도구가 다뤄야 할 컴파일러 변환의 복잡성을 효과적으로 관리함을 나타냅니다.



### RLAC: Reinforcement Learning with Adversarial Critic for Free-Form Generation Tasks (https://arxiv.org/abs/2511.01758)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 open-ended generation(개방형 생성) 작업의 복잡성을 해결하기 위해 RLAC(Reinforcement Learning with Adversarial Critic)라는 후속 훈련 접근법을 제안합니다. 이 방식은 다수의 task-specific evaluation rubrics(작업 특정 평가 기준)을 효율적으로 관리하면서도 비용이 높은 검증 과정을 최소화합니다. RLAC는 외부 검증자를 통해 생성기(generator)와 비평가(critic)를 함께 최적화하며, 이는 상호간의 성능 향상을 목표로 합니다.

- **Technical Details**: RLAC 접근법은 대형 언어 모델(LLM)을 비평가로 활용하여 발생할 가능성이 있는 오류를 동적으로 식별합니다. 툴의 설계 과정에서, 비평가는 오류가 발생할 수 있는 문장 번호와 오정보를 추정하여 외부 검증자로부터 확인을 받습니다. 이 방법은 각 사실의 정확성을 검사하는 과정에서 FactScore를 사용하여 Wikipedia 지식 기반을 쿼리하며, 이를 통해 진리(true) 또는 거짓(false)을 반환합니다.

- **Performance Highlights**: 실험 결과, RLAC는 텍스트 생성에서 사실적 정확성을 개선하고 코드 생성의 정답성을 향상시켰습니다. 또한, 기존의 exhaustive verification(철저한 검증) 및 reward model 방법에 비해 우수한 성능을 보였으며, 고정된 비평가보다 동적인 비평가가 더 높은 효과를 나타냈습니다. 이는 RLAC가 open-ended generation 작업에 대한 RL 후속 훈련을 확장할 수 있는 잠재력을 제시합니다.



### SM-based Semantics for Answer Set Programs Containing Conditional Literals and Arithmetic (https://arxiv.org/abs/2511.01753)
Comments:
          This version corrects the review of tau for negated atoms, and clarifies the distinction between global and local variables in conditional literals (the supporting proofs are also updated accordingly)

- **What's New**: 이 논문에서는 조건부 리터럴(conditional literals)과 산술(arithmetic)을 포함하는 로직 프로그램에 대한 새로운 의미론(semantics)을 제안합니다. 기존의 의미론과는 달리, 이 제안된 의미론은 구속(grounding)을 필요로 하지 않으며, SM 연산자(SM operator)를 기반으로 합니다. 이러한 접근 방식을 통해 로직 프로그램의 행동을 특정 입력 데이터의 구속 맥락과 독립적으로 이유를 제기할 수 있는 능력을 강화합니다.

- **Technical Details**: 로직 프로그램의 문법(syntax)은 Abstract Gringo (AG) 언어의 일부로, 조건부 리터럴이 포함된 규칙으로 확장됩니다. 이 프로그램은 수치(numerals), 기호 상수(symbolic constants), 변수(variables)의 세 가지 기호 집합으로 구성됩니다. 기본 리터럴(basic literal)과 조건부 리터럴(conditional literal)의 정의를 통해 프로그램이 어떻게 구성되는지 설명하며, 규칙(rule)의 형식도 세부적으로 명시됩니다.

- **Performance Highlights**: 제안된 SM 기반 의미론은 기존의 의미론과 정밀한 대응 관계를 형성합니다. 이로 인해 그래프 색칠 문제(Graph Coloring)와 같은 프로그램의 형식적 검증(formal verification)을 보다 모듈화된 방식으로 수행할 수 있게 되며, 이는 안전-critical 응용 프로그램에서 매우 중요합니다. 이 논문에서 제시한 주요 결과들은 기존의 의미론에 비해 조건부 리터럴과 산술이 포함된 프로그램의 의미론적 확장을 포함하여 더 많은 프로그램을 지원할 수 있도록 합니다.



### Scam Shield: Multi-Model Voting and Fine-Tuned LLMs Against Adversarial Attacks (https://arxiv.org/abs/2511.01746)
Comments:
          8 pages

- **What's New**: 이번 논문에서는 사이버 보안에서 중요한 도전 과제가 되고 있는 사기 감지를 향상시키기 위한 계층 구조의 사기 탐지 시스템(HSDS)을 제안합니다. 이 시스템은 다중 모델 투표 방식과 함께 조정된 LLaMA 3.1 8B Instruct 모델을 결합하여 적대적 공격에 대한 정확성과 견고성을 높입니다. 초기 예측을 제공하는 네 개의 분류기를 통합하여 다수결 방식으로 작동하며, 모호한 경우에만 조정된 모델로 이첩되어 분류 오류를 줄입니다.

- **Technical Details**: HSDS는 기본적인 다중 모델 앙상블과 조정된 LLaMA 8B Instruct 모델을 통합하는 계층적 프레임워크를 사용합니다. 이는 일반 및 적대적 사기 메시지 모두에 대한 견고성을 강화하는 동시에 컴퓨팅 효율성을 유지하는 것을 목표로 합니다. LoRA(Low-Rank Adaptation)를 통해 LLaMA 8B Instruct을 적응시킴으로써 계산 비용을 크게 줄이고 GPT-3.5 Turbo 및 Claude 3 Haiku보다 더 나은 적대적 탐지 정확성을 달성하였습니다.

- **Performance Highlights**: 실험 결과, 다중 모델 투표 접근 방식이 적대적 사기 탐지에서 강력한 효과를 발휘하며 전통적인 기계 학습 및 독점 LLM 기반 기준을 초월하는 성과를 보였습니다. HSDS는 대다수의 경우를 LLM으로부터 분리하여 탐지를 신속하게 수행 송실하게 하면서도 불분명한 사례에 대한 강력한 분석 능력을 유지하는 이점을 제공합니다.



### An Open-Access Benchmark of Statistical and Machine-Learning Anomaly Detection Methods for Battery Applications (https://arxiv.org/abs/2511.01745)
- **What's New**: 이번 연구에서는 OSBAD(Open-Source Benchmark of Anomaly Detection)라는 개방형 기준 벤치마크를 제안하여 배터리 응용 분야에서 이상 탐지를 위한 여러 프레임워크를 비교하고 있습니다. 15개의 다양한 알고리즘을 벤치마크하여 통계적 및 비지도 기계 학습 방법을 포함한 이상 탐지 방식들을 체계적으로 비교할 수 있도록 합니다. 또한, 물리학 및 통계학을 기반으로 한 피처 변환(workflow)을 통해 집단적 이상을 개별 이상으로 분해함으로써 이상 탐지의 효율성을 높입니다.

- **Technical Details**: OSBAD는 통계적, 거리 기반(distance-based), 비지도 머신러닝(ML) 접근법을 포괄하는 15개 알고리즘을 통합할뿐만 아니라, 이상점 탐지를 위한 여러 방법론을 제공합니다. 전통적인 통계 방법(예: 표준 편차(Standard Deviation), Z-score 등)과 현재 발전된 머신러닝 기법(예: Isolation Forest, Autoencoder 등)을 포함하여, 다양한 전기화학 시스템에서 이상 징후를 탐지하는 데 사용됩니다. 특히 베이지안 최적화(Bayesian optimization) 파이프라인을 통해 자동 하이퍼파라미터 튜닝을 수행하여 비지도 이상 탐지의 주요 병목 현상을 해결하고 있습니다.

- **Performance Highlights**: 본 연구는 다양한 전기화학 자재를 사용하는 두 가지 독립 데이터셋(액체 전해질 및 고체 상태 전지)을 통해 OSBAD의 범 화학적 일반화 가능성을 검사하였습니다. 이 데이터셋들은 각각 다른 전극 재료와 전해질 조성을 가지고 있어, OSBAD가 여러 전지 시스템에서 이상을 식별하는 데 효과적임을 보여주고 있습니다. 배터리를 안전하고 신뢰할 수 있게 운영하기 위해서는 자료 기반의 이상 탐지가 필수적이라는 점을 강조하며, OSBAD가 이를 위한 통합된 기반을 제공함을 증명합니다.



### Towards Efficient Federated Learning of Networked Mixture-of-Experts for Mobile Edge Computing (https://arxiv.org/abs/2511.01743)
- **What's New**: 최근 대규모 인공지능 모델(Large AI Models, LAMs)에서의 발전은 차세대 무선 네트워크의 모바일 엣지 컴퓨팅에 큰 혁신을 가져오고 있습니다. 하지만 LAMs 훈련에 필요한 높은 계산 자원과 대규모 데이터가 엣지 장치의 한정된 저장 공간 및 계산 능력과 충돌하여 도전 과제가 되고 있습니다. 이 연구에서는 네트워크 혼합 전문가 시스템(Networked Mixture-of-Experts, NMoE)을 소개하며, 이를 통해 클라이언트가 전문성에 기반하여 적합한 이웃에게 작업을 분배하고 결과를 통합하여 공동 작업을 수행합니다.

- **Technical Details**: NMoE 시스템에서는 각 클라이언트가 세 개의 서브 네트워크, 즉 교차 공유 기능 추출기(cross-shared feature extractor), 교차 공유 게이팅 네트워크(cross-shared gating network), 그리고 개인화된 전문가를 배포합니다. 추론 과정에서 클라이언트는 입력 데이터를 기능 추출기를 통해 처리하여 잠재 표현(latent representations)을 얻고, 이를 게이팅 네트워크에 전달하여 데이터를 처리할 가장 적합한 전문가를 결정합니다. 이와 함께, 여러 훈련 전략을 도입하여 성능을 극대화하고 데이터 개인화를 유지합니다.

- **Performance Highlights**: NMoE 시스템의 효과를 입증하기 위해 광범위한 실험을 수행했습니다. 제안된 훈련 방법은 다양한 조건에서의 성능을 검증하며, 통신 효율성과 데이터 개인화를 고려하여 클라이언트 맞춤형 데이터 분포에 적응할 수 있는 방안을 제시합니다. NMoE는 분산된 장치 상에서 LAM 훈련의 새로운 가능성을 열어줍니다.



### A Proof of Learning Rate Transfer under $μ$P (https://arxiv.org/abs/2511.01734)
Comments:
          23 pages

- **What's New**: 이 논문은 선형 다층 퍼셉트론(MLP)에서 폭(width)을 통한 학습률 전이(learning rate transfer)의 첫 번째 증명을 제공합니다. 저자는 μP라는 신경망 초기화 방식에서 최적 학습률이 폭이 무한대로 증가할 때 비제로 상수(non-zero constant)로 수렴한다는 이론적 설명을 제시합니다. 이는 기존의 다른 파라미터화 방식에서는 이 성질이 성립하지 않음을 보여줍니다.

- **Technical Details**: 선형 MLP의 훈련 과정을 통해 손실 함수는 학습률의 다항식 함수(polynomial function)로 표현될 수 있음을 보여줍니다. 저자는 이러한 다항식의 수렴 동역학(convergence dynamics)과 그 근(root)을 연구하여 최적 학습률이 폭에 따라 어떻게 수렴하는지를 분석합니다. μP는 초기화 및 학습률이 폭에 따라 조정되는 방식을 정의하며, 이를 통해 최적의 하이퍼파라미터(hyperparameter) 전이(transfer)를 가능하게 합니다.

- **Performance Highlights**: 논문은 다양한 실험을 통해 이론적 결과를 뒷받침하는 광범위한 시뮬레이션 결과를 제공합니다. 특히, 다른 파라미터화 방식인 표준 파라미터화(Standard Parametrization) 및 신경 탄젠트 파라미터화(Neural Tangent Parametrization)는 유의미한 최적 학습률의 변화를 초래하여 추가적인 조정이 필요하다는 사실을 강조합니다. 다양한 활성화 함수(activation function), 최적화 기법(optimizer), 깊이(depth), 훈련 시간(training time)의 변화에 따른 추가적인 경험적 결과도 제시합니다.



### Multi-Step Knowledge Interaction Analysis via Rank-2 Subspace Disentanglemen (https://arxiv.org/abs/2511.01706)
Comments:
          Under review

- **What's New**: 이번 연구는 LLMs (Large Language Models)의 Natural Language Explanations (NLEs) 생성을 위한 PK (Parametric Knowledge)와 CK (Context Knowledge)의 상호작용을 새롭게 분석합니다. 기존의 이론은 주로 단일 단계 생성에 초점을 맞췄으며, PK와 CK의 상호작용을 rank-1 공간에서 모델링했습니다. 그러나, 이 연구는 PK와 CK의 기여를 더 정확하게 구분할 수 있는 새로운 rank-2 투영(subspace)을 제안합니다.

- **Technical Details**: 이 연구는 여러 QA 데이터셋과 오픈 가중치로 튜닝된 LLM를 사용하여 PK와 CK의 상호작용을 심층적으로 분석합니다. 다단계 NLE 생성 과정에서 연구진은 PK와 CK가 어떻게 다른 방식으로 기여하는지를 밝혀냈고, 이는 종합적인 rank-2 아키텍처를 통해 수행되었습니다. 이 새로운 모델은 NLE의 신뢰성과 통찰을 향상시키기 위해 PK와 CK 간의 관계를 더욱 면밀히 조명합니다.

- **Performance Highlights**: 실험 결과, rank-1 subspace에서는 다양한 지식 상호작용이 잘 포착되지 않았으나, rank-2 구조에서는 두 공헌의 균형을 맞추며 NLE 생성을 명확히 표현했습니다. 또한, CoT (Chain-of-Thought) 방법론이 PK 의존성을 감소시켜 CK 방향으로 생성된 NLE를 유도한다는 사실이 확인되었습니다. 이러한 결과는 LLMs의 내적 및 외적 지식 통합에 대한 이해를 한층 높이며, 향후 연구의 기초 자료로 활용될 수 있습니다.



### Solution Space Topology Guides CMTS Search (https://arxiv.org/abs/2511.01701)
Comments:
          15 pages, 3 figures

- **What's New**: 이 연구에서는 Monte Carlo Tree Search (MCTS)에서 퍼즐을 해결하는 데 어떤 topology(구조)가 필요할지를 논의합니다. 이전 연구에서 grid topology(격자 구조)를 사용했으나 효과가 없음을 발견한 저자들은, solution space topology(해결 공간 구조)를 측정하는 방법을 제안합니다. 이 방법은 패턴 규칙에 의해 제한된 유효한 색상 배정의 구조를 기반으로 합니다.

- **Technical Details**: 연구의 핵심은 compatibility graphs(호환성 그래프)를 구축하여 (cell, color) 쌍을 노드로 하고 패턴 제약을 만족하는 배정 간의 호환성을 나타내는 간선으로 구성하는 것입니다. 이를 통해 MCTS에 통합할 수 있는 다양한 topological features(위상 특성)를 추출합니다. 특히, algebraic connectivity(대수적 연결성), rigidity(강성), 그리고 color structure(색상 구조)와 같은 특성들이 문제의 난이도에 따라 변합니다.

- **Performance Highlights**: 이 연구는 MCTS의 노드 선택 과정에 이러한 topological features를 통합하여 성능을 개선할 수 있음을 보여줍니다. 실험 결과, algebraic connectivity가 가장 중요한 신호로 작용하여 해결 공간 구조를 효과적으로 반영합니다. 따라서, 이 연구는 퍼즐 해결 시 올바른 topology의 중요성을 강조하며, 문제 공간이 아닌 해결 공간을 측정하는 것이 효과적임을 입증합니다.



### Bayesian Natural Gradient Fine-Tuning of CLIP Models via Kalman Filtering (https://arxiv.org/abs/2511.01694)
- **What's New**: 이번 연구에서는 CLIP 기반의 비전-언어 모델을 더욱 효율적으로 미세 조정(fine-tuning)하기 위해 Kalman 필터(Kalman filtering)를 이용한 방법을 제안합니다. 기존의 첫 번째 순서 최적화 기법 대신 두 번째 순서 최적화 기법을 사용하여 손실 함수의 곡률 정보를 설정에 따라 조정합니다. 이 방법을 통해 성능을 더욱 안정적으로 향상시키며, OOD 환경에서도 일반화 능력을 강화합니다.

- **Technical Details**: 자연 기울기 하강법(Natural Gradient Descent, NGD)을 사용하여 피셔 정보 행렬(Fisher Information Matrix, FIM)의 역행렬로 표준 기울기를 전처리합니다. 이 방식은 CLIP 모델에 적합하며, 특히 비앙변화가 큰 데이터에 대한 미세 조정에 유용합니다. Bayesian 추론과 Kalman 필터를 이용해 미세 조정 과정의 불확실성을 정량화하고, 더 나아가 저차원의 근사화를 통해 계산량을 줄입니다.

- **Performance Highlights**: 다양한 이미지 분류 데이터셋에서 실시한 광범위한 실험을 통해 제안한 알고리즘이 기존의 최고 성능 기법들에 비해 ID 성능에서 우수하거나 유사한 결과를 보였으며, OOD 환경에서도 개선된 강건성을 보였습니다. 이 연구는 CLIP 기반 모델 미세 조정에 Kalman 필터를 최초로 성공적으로 적용하여 비전-언어 작업에서의 학습 효율을 높이는 새로운 가능성을 제시합니다.



### Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI (https://arxiv.org/abs/2511.01689)
Comments:
          12 pages, 6 figures, 4 tables

- **What's New**: 이번 연구는 인공지능(AI) 어시스턴트의 페르소나(persona) 개발에 관한 새로운 접근 방식을 소개합니다. 캐릭터 훈련(character training)이라고 알려진 이 과정은 현대 챗봇의 행동과 가치, 신념에 영향을 미치며, 상호작용의 질과 사용자 및 개발자의 의도에 대한 정렬에 필수적입니다. 특히, 캐릭터 훈련의 첫 번째 공개 구현을 제공하여 기존의 방식보다 효과적인 인공지능 어시스턴트 페르소나를 생성하는 방법을 제시합니다.

- **Technical Details**: 연구에서는 11개의 예시 페르소나를 사용하여 세 가지 인기 있는 오픈 소스 모델을 미세 조정(fine-tune)합니다. 이러한 페르소나는 유머러스(humorous), 깊은 배려(deeply caring) 또는 심지어 악의적(malevolent)인 성격을 포함합니다. 연구진은 Constitution AI와 합성 자기 성찰 데이터(synthetic introspective data)를 활용한 새로운 데이터 파이프라인을 통해 페르소나 형성을 최적화하는 방법론을 개발하였습니다.

- **Performance Highlights**: 본 연구의 접근 방식은 적대적 프롬프트(adversarial prompting)에도 강한 내성을 보이며, 생성되는 내용의 일관성과 현실성 또한 향상됩니다. 일반적인 벤치마크에서 측정한 바와 같이, 이러한 미세 조정은 AI의 일반적인 능력에는 거의 영향을 미치지 않음을 보여줍니다. 최종적으로, 연구진은 전체 후속 훈련(post-training) 방법을 설명하고, 이를 오픈 소스로 공개하여 다른 연구자들이 사용할 수 있도록 배포합니다.



### Student Engagement in AI Assisted Complex Problem Solving: A Pilot Study of Human AI Rubik's Cube Collaboration (https://arxiv.org/abs/2511.01683)
- **What's New**: 이 논문은 ALLURE 시스템을 소개하며, 이는 AI 알고리즘인 DeepCubeA를 활용하여 학생들이 루빅스 큐브의 첫 단계인 흰색 십자가를 해결하도록 안내합니다. 이 시스템은 STEM 교육에서 퍼즐 해결을 위한 Scaffolded instruction을 제공합니다. 새로운 AI 알고리즘의 등장으로 복잡한 문제 해결에 대한 교육적인 기회가 확대되었습니다.

- **Technical Details**: ALLURE 시스템은 AI 알고리즘을 사용하여 STEM 기술과 관련된 학생들의 행동을 분석합니다. 연구 결과는 공간적 추론(spatial reasoning), 비판적 사고(critical thinking), 알고리즘적 사고(algorithmic thinking)와 같은 STEM 기술과의 연관성을 보여줍니다. 이 데이터는 미래의 교육 데이터 마이닝(educational data mining)에 활용될 수 있습니다.

- **Performance Highlights**: 파일럿 연구의 데이터는 학생들이 ALLURE 시스템을 통해 문제를 해결할 때의 행동을 분석하여, AI 지원 및 협력이 학생들에게 어떻게 도움이 되는지를 이해하는 데 기여합니다. 논문의 초기 발견은 학생들이 AI 시스템을 활용할 때 STEM 기술이 어떻게 발현되고 있는지를 조명합니다.



### Spin-Adapted Neural Network Wavefunctions in Real Spac (https://arxiv.org/abs/2511.01671)
- **What's New**: 이 논문에서는 Spin-Adapted Antisymmetrization Method (SAAM)이라는 새로운 방법을 소개합니다. 이 방법은 많은 전자 파동함수에서 정확한 총 스핀 대칭(total spin symmetry)을 적용하여 전자 상관관계를 효과적으로 캡처합니다. SAAM은 심층 신경망의 표현력을 활용하여 실공간(real space)에서 반대칭적(wavefunctions)인 여러 전자 시스템을 다룹니다.

- **Technical Details**: SAAM은 그룹 표현 이론(group representation theory)을 바탕으로 하여, 신경망 기반 양자 몬테 카를로(neural network-based quantum Monte Carlo, NNQMC)에서 스핀 적응(spin adaptation)을 강제하는 절차입니다. 이 방법은 물리적 프라이어(physical priors)를 신경망 파동함수에 통합할 수 있게 해 주며, 상관된 시스템에 대한 컴팩트한 표현을 제공합니다. SAAM은 기존의 NNQMC에서의 스핀 처리 방법보다 더 높은 정확도와 효율성을 보여주며 추가적인 하이퍼파라미터(hyperparameters) 없이 정확한 스핀 순도를 달성합니다.

- **Performance Highlights**: SAAM을 사용하여 철-황 클러스터(spin ladder of iron-sulfur clusters)를 연구한 결과, 저질의 스핀 상태(low-lying spin states)와 스핀 간격(spin gaps)의 정확한 해석을 제공했습니다. 특히, [Fe$_2$S$_2$] 및 [Fe$_4$S$_4$] 클러스터의 전자 구조에 대한 새로운 통찰을 제공하는 성과를 올렸습니다. 이 연구 결과는 강하게 상관된 시스템에 대한 스핀 적응 NNQMC의 신뢰할 수 있는 표준으로서 SAAM의 유용성을 입증합니다.



### SeaLLMs-Audio: Large Audio-Language Models for Southeast Asia (https://arxiv.org/abs/2511.01670)
Comments:
          10 pages

- **What's New**: SeaLLMs-Audio는 인도네시아어(id), 태국어(th), 베트남어(vi), 영어(en), 중국어(zh) 등 5개 동남아시아(SEA) 언어에 특화된 최초의 대규모 오디오-언어 모델(LALM)입니다. 이 모델은 다양한 오디오 중심 과제를 위한 대규모 오디오 코퍼스를 기반으로 학습하였으며, 음성 기반 상호작용과 세밀한 오디오 이해에 강력한 성능을 보입니다. 각각의 언어를 위한 다국어 지원 뿐만 아니라, 오디오와 텍스트 입력을 모두 수용하는 다중 모드 기능을 제공하는 것이 특징입니다.

- **Technical Details**: SeaLLMs-Audio는 1.58M 개의 대화를 포함한 방대한 훈련 데이터를 기반으로 하여, 자동 음성 인식(ASR), 오디오 캡셔닝(AC), 음성-텍스트 번역(S2TT), 음성 요약(SS)과 같은 다양한 과제를 지원합니다. 이러한 데이터는 공공 및 사설 데이터셋을 통해 수집되었으며, 데이터 전처리를 통해 서로 다른 형식을 통합하는 과정을 거쳐야 했습니다. 특히, 기존의 ASR 데이터와 다른 언어로의 텍스트 변환을 결합하여 다양한 언어 쌍 성과 데이터를 생성하였습니다.

- **Performance Highlights**: SeaLLMs-Audio는 SeaBench-Audio라는 새로운 벤치마크를 통해 동남아시아 언어에 대해 강력하고 경쟁력 있는 성능을 입증하였습니다. 이 벤치마크는 LALM의 표준화된 평가를 위해 설계되었으며, 다양한 실제 언어 이해 시나리오를 반영하는 여러 개방형 과제를 포함하고 있습니다. 실험 결과, SeaLLMs-Audio는 여러 오디오-언어 작업에서 우수한 성능을 달성하며, SEA 지역 연구 및 산업에 기여할 것으로 기대됩니다.



### The Ghost in the Keys: A Disklavier Demo for Human-AI Musical Co-Creativity (https://arxiv.org/abs/2511.01663)
- **What's New**: 이 논문은 Aria-Duet라는 시스템을 소개하며, 인체와 AI 간의 실시간 음악 듀엣을 실현하는 방법을 설명합니다. 기존의 음악 작곡 AI 도구가 직면한 상호작용 방식의 한계를 해소하기 위해, 이 시스템은 피아니스트와 최첨단 생성 모델인 Aria 간의 협업을 가능하게 합니다. Yamaha Disklavier라는 물리적 인터페이스를 사용하여 사용자와 모델 간의 실제적인 음악 피드백 루프를 제공함으로써 창의적인 흐름을 촉진합니다.

- **Technical Details**: Aria-Duet는 두 가지 주요 구성 요소로 이루어져 있습니다: 사용자 성과의 연속성을 생성하는 생성 모델과 실시간 사용자 제어 흐름을 관리하는 엔진입니다. 이 시스템은 Aria라는 훈련된 오토회귀 변환기 모델을 기반으로 하며, MIDI 연결을 통해 기존 피아노 연주를 녹음하고 실시간으로 계속해서 연주할 수 있도록 설계되었습니다. 시스템의 설계는 사용자와 AI 간의 원활한 상호작용을 보장하기 위한 여러 가지 설계 문제를 해결하는 데 중점을 두고 있습니다.

- **Performance Highlights**: Aria-Duet는 사용자와 AI 간의 음악적 대화가 가능하다는 점에서 주목할 만한 성과를 보여줍니다. 이 시스템은 다양한 음악 스타일과 어휘를 수용하며, 사용자로부터의 신호에 기반하여 즉각적으로 음악적 연속성을 생성할 수 있습니다. 또한, 음악학적 분석을 통해 모델이 스타일적 의미를 유지하고 일관된 구절 논리를 발전시키는 능력을 보여주며, 이는 인간과 AI의 협업이 음악적으로 정교한 대화로 이어질 수 있음을 입증합니다.



### EngChain: A Symbolic Benchmark for Verifiable Multi-Step Reasoning in Engineering (https://arxiv.org/abs/2511.01650)
Comments:
          24 pages, includes figures and tables; introduces the EngChain benchmark

- **What's New**: 이 연구에서는 EngChain이라는 새로운 기준을 도입하여 검증 가능한 다단계 공학 문제 해결을 위한 벤치마크를 제안합니다. EngChain은 90가지 문제를 포함하고 있으며, 9개 도메인 및 20개 영역으로 조직되어 있습니다. 문제는 심볼릭 템플릿을 이용해 생성되어 다양성을 확보하고 오염 위험성을 줄이는 데 초점을 맞추고 있습니다.

- **Technical Details**: EngChain은 기존 벤치마크에서 부족했던 종합적 사고를 평가하기 위해 기반을 두고 있습니다. 평가 방식은 두 단계로 나뉘어 있으며, 첫 번째 단계에서는 각 사고 과정의 수치적 및 의미적 유효성을 정량적으로 검증하고, 두 번째 단계에서는 LLM-As-A-Judge라는 자동화된 시스템을 통해 사고 오류를 질적으로 분류합니다.

- **Performance Highlights**: 연구 결과는 11개의 가장 앞선 LLM 모델들의 성능을 포괄적으로 분석하여, 다수의 추론 오류가 계산 실수가 아닌 개념적 오류에 기인한다는 점을 밝혔습니다. 이는 EngChain이 공학 문제의 복잡한 추론 능력을 평가하는 데 진정으로 유용한 도구임을 보여줍니다.



### A Graph-based RAG for Energy Efficiency Question Answering (https://arxiv.org/abs/2511.01643)
- **What's New**: 이 연구에서는 에너지 효율성(EE) 질문 응답을 위한 그래프 기반의 Retrieval Augmented Generation (RAG) 아키텍처 내에서 대형 언어 모델(LLM)의 사용을 조사했습니다. 이 시스템은 에너지 분야의 가이드라인과 규제 문서에서 자동으로 지식 그래프(KG)를 추출하고, 이 그래프를 탐색하여 사용자에게 다양한 언어로 정확한 답변을 제공합니다. 인간 기반의 검증 실험을 통해 이 아키텍처의 잠재력을 확인하고 강점과 약점을 파악하였습니다.

- **Technical Details**: 시스템은 세 가지 주요 구성요소로 나뉘며, 첫 번째로 지식 추출기가 도메인 특정 문서에서 엔티티와 관계를 포함한 트리플을 추출합니다. 이렇게 추출된 정보는 지식 기반(KB)을 구축하는 데 사용되며, 이후 사용자의 질문을 처리하기 위한 검색 및 생성 프로세스가 이어집니다. 이 과정에서 LLM 기반의 알고리즘이 사용되어 트리플을 자동으로 추출하고, 동일한 구문이 적용되도록 엔티티 이름을 통합합니다.

- **Performance Highlights**: 검증 결과, 시스템은 약 75.2%의 경우에 정확한 답변을 제공하며, 에너지 효율성에 대한 일반 질문에 대해서는 81.0%까지 높아지는 성능을 보입니다. 또한, 다국어 처리에 있어 번역으로 인한 정확도 손실은 4.4%로 나타났습니다. 이러한 결과는 그래프 기반 RAG 아키텍처가 다국적 사용자를 위한 효율적인 질문 응답 시스템을 구축할 수 있는 잠재력을 가지고 있음을 시사합니다.



### Prompt Injection as an Emerging Threat: Evaluating the Resilience of Large Language Models (https://arxiv.org/abs/2511.01634)
Comments:
          10 pages, 6 figures

- **What's New**: 이 연구는 Large Language Models (LLMs)에 대한 새로운 공격 유형인 prompt injection을 평가하기 위한 통합 프레임워크를 제안합니다. 사용자의 입력이나 외부 콘텐츠에 숨겨진 명령어가 삽입되어 모델이 원래의 작업을 무시하거나 위험한 응답을 생성하도록 만드는 이 공격을 분석합니다. 이를 통해 LLM의 강건성, 안전성, 의미적 안정성을 측정하는 세 가지 보완 지표인 Resilience Degradation Index (RDI), Safety Compliance Coefficient (SCC), Instructional Integrity Metric (IIM)을 정의했습니다.

- **Technical Details**: 연구에서는 질문 응답, 요약, 번역, 추론, 코드 생성이라는 다섯 가지 일반 언어 작업에 대해 네 가지 instruction-tuned 모델(GPT-4, GPT-4o, LLaMA-3 8B Instruct, Flan-T5-Large)을 평가했습니다. 결과적으로 GPT-4가 전반적으로 가장 우수한 성능을 보였고, 개방형 모델은 여전히 더 취약한 것으로 나타났습니다. 모든 모델이 부분적으로 취약함을 보여주었으며, 특히 간접 및 직접 오버라이드 공격에 민감합니다.

- **Performance Highlights**: GPT-4는 가장 높은 전체 회복력(RDR = 9.8 %, SCR = 96.4 %)을 기록했습니다. 반면, 개방형 모델들은 성능 저하가 더 크고 안전 점수가 낮았습니다. 연구 결과는 모델 크기보다 alignment strength와 안전 조정이 회복력에 더 중요한 역할을 한다는 점을 강조하고 있으며, 제안된 프레임워크는 LLM의 안전성과 신뢰성을 향상하기 위한 실용적인 통찰을 제공합니다.



### Scaling Graph Chain-of-Thought Reasoning: A Multi-Agent Framework with Efficient LLM Serving (https://arxiv.org/abs/2511.01633)
- **What's New**: 이번 논문에서는 Graph Chain-of-Thought (Graph-CoT) 기반으로 첫 번째 다중 에이전트 시스템인 GLM을 소개합니다. GLM은 그래프 구조화된 지식에 대한 단계별 추론을 지원하며, 기존 시스템들의 낮은 정확도와 비효율성을 개선하기 위해 최적화된 LLM 서비스를 통합했습니다. 이를 통해 기존의 단일 에이전트 시스템에서 발생하는 비효율적인 맥락 재인코딩 문제를 해결하고 있습니다.

- **Technical Details**: GLM은 데이터 구조와 의존성을 고려하여 분류, 추론, 작업 생성, 그래프 검색을 위한 전문화된 에이전트로 추론을 분해합니다. 이로 인해 프롬프트 길이와 추론 반복 횟수를 줄이면서도 품질은 유지할 수 있게 되었으며, Graph-CoT에 맞춰 KV 캐시 관리와 우선순위 기반 제거 정책, 파이프라인 실행 방식을 도입하여 추론 효율성을 개선합니다. 이러한 기술들은 특히 복잡한 엣지 관계에 대한 구조적 정보 처리를 지원합니다.

- **Performance Highlights**: GLM은 실험에서 기존 최첨단 Graph-CoT 시스템에 비해 최대 38% 답변 정확성을 개선하고, 최대 95.7%의 토큰 비용을 절감했으며, 추론 대기 시간을 90.3% 줄이고, 최대 15.1배 높은 처리량을 달성했습니다. 이러한 성과는 GLM이 복잡한 현실 세계의 추론 작업에 효과적으로 적용될 수 있도록 만듭니다.



### Imperfect Language, Artificial Intelligence, and the Human Mind: An Interdisciplinary Approach to Linguistic Errors in Native Spanish Speakers (https://arxiv.org/abs/2511.01615)
Comments:
          12 pages, 3 figures

- **What's New**: 이 연구에서는 원주율 스페인어 사용자가 발생시킨 언어적 오류를 다루며, 인공지능 시스템이 이러한 오류를 재현하고 수정하는 방식에 대한 분석을 제공합니다. 특히 기존에 사용되던 접근 방식을 넘어선 학제간 연구를 제안합니다. 현재의 대형 언어 모델(LLM)이 언어적 오류를 어떻게 해석하는지에 대한 새로운 관점을 제공합니다.

- **Technical Details**: 연구는 이론적 언어학(theoretical linguistics)을 통해 오류의 본질을 분류하고 이해하며, 신경언어학(neurolinguistics)을 통해 뇌에서의 실시간 언어 처리(contextualize) 관점을 제공합니다. 또한 자연어 처리(NLP) 기술을 사용하여 생성된 오류를 평가하고, 500개 이상의 실제 오류로 구성된 특수 코퍼스를 구축하여 실증적 분석을 실행합니다.

- **Performance Highlights**: 이 연구는 AI 모델(GPT 또는 Gemini 등)과의 비교 분석을 통해 스페인어에 대한 이해를 높이고, 언어적 오류의 해석 정확도와 인간의 언어 행동 패턴을 일반화할 수 있는 능력을 평가합니다. 궁극적으로, 더 인지적으로 정보화된 NLP 시스템의 발전에 기여하게 될 것입니다.



### DINO-MX: A Modular & Flexible Framework for Self-Supervised Learning (https://arxiv.org/abs/2511.01610)
- **What's New**: 이번 논문은 DINO-MX라는 모듈형 훈련 프레임워크를 소개합니다. DINO-MX는 DINO, DINOv2 및 DINOv3의 핵심 원칙을 통합하여 다양한 transformer 기반 아키텍처를 지원하며 Hugging Face 생태계와 완벽하게 호환됩니다. 이 프레임워크는 저차원 적응(LoRA), 레이어 고정 및 지식 증류와 같은 다양한 훈련 전략을 포함하며, 분산 훈련을 위해 DDP 및 FSDP를 지원합니다.

- **Technical Details**: DINO-MX는 자연 데이터와 전문 데이터 타입 모두에서 작동하도록 설계되었습니다. 실험 결과 다양한 데이터셋에서 DINO-MX는 경쟁력 있는 성능을 달성하면서도 계산 비용을 크게 줄이는 성과를 보여줍니다. 또한, 해석 가능한 도구와 레이블 기반 데이터 증대 방법을 통해 추가적인 탐지나 세분화 헤드 없이 주의기반 로컬리제이션을 향상시킬 수 있습니다.

- **Performance Highlights**: DINO-MX는 의료 이미징 및 다양한 연구 및 실제 응용 프로그램에서 자기 지도 학습 모델을 개발하고 적응 및 벤치마킹할 수 있는 재현 가능하고 확장 가능한 기반을 제공합니다. 이 프레임워크는 많은 리소스가 부족한 상황에서도 사용할 수 있으며, 연구자들이 최신 오픈 소스 모델을 빠르게 비교하고 평가할 수 있도록 돕는 인프라를 제공합니다.



### Federated Cyber Defense: Privacy-Preserving Ransomware Detection Across Distributed Systems (https://arxiv.org/abs/2511.01583)
- **What's New**: 이 논문은 다양한 조직이 라우드웨어(여기서는 랜섬웨어) 탐지 모델을 협업하여 훈련할 수 있도록 해주는 연합학습(Federated Learning, FL)을 평가합니다. 데이터는 각 조직의 로컬 환경에서 안전하게 유지되면서도, 모델 성능을 높일 수 있는 가능성을 보여줍니다. 특히, 기존의 중앙집중식 학습이 가지는 보안 및 개인정보 문제를 극복하는 혁신적인 접근법으로 주목받고 있습니다.

- **Technical Details**: 랜섬웨어 탐지 문제를 다룬 이 연구는 FL을 적용하여 데이터 프라이버시를 존중하니까 동시에 탐지의 강도를 높일 수 있는 방법을 제시합니다. 연구에서는 Ransomware Storage Access Patterns (RanSAP) 데이터세트를_validate_하고 FL을 사용하여 기존 서버-로컬 모델에 비해 상대적으로 9% 향상된 탐지 정확도를 달성했습니다. 이는--기존의 비즈니스 환경에서도 유효한 성능을 발휘할 수 있음을 나타냅니다.

- **Performance Highlights**: FL을 활용한 탐지 모델은 대규모 분산 고객 노드에서 데이터의 민감성을 현저히 유지하면서 강력하고 최신의 탐지 모델을 공동으로 훈련할 수 있게 해줍니다. 이에 따라 다양한 고객 환경에서의 랜섬웨어 탐지 성능을 높이는 효과적이고 유망한 구체적인 방법론으로 평가됩니다. 결과적으로, FL은 랜섬웨어 탐지에서 프라이버시를 보호하면서도 성능의 균형을 이루는 확장 가능한 프레임워크를 제시합니다.



### HIT-ROCKET: Hadamard-vector Inner-product Transformer for ROCKE (https://arxiv.org/abs/2511.01572)
- **What's New**: 이 논문에서는 Hadamard convolutional transform을 기반으로 한 새로운 기능 추출 접근 방식을 제안합니다. 기존의 ROCKET방식과의 호환성을 유지하면서, 커널 정교조합을 통해 계산효율성, 강건성 및 적응성을 개선합니다. UCR 시계열 데이터셋을 사용한 실험 결과, SOTA 성능을 입증하며, 특히 miniROCKET보다는 50% 더 짧은 훈련 시간을 기록했습니다.

- **Technical Details**: 제안된 HIT-ROCKET 방법은 Hadamard 행렬의 열 또는 행 벡터를 합성곱 커널로 사용합니다. 이 방식은 명확한 수학적 지원을 제공하며, 값비싼 랜덤 합성곱 커널의 고정화 문제를 해결해 분류 성능과 노이즈 저항성을 향상시킵니다. Dilated convolution을 이용하여 시계열로부터 특징을 추출하고, PPV(Positive Predictive Value)를 유일한 핵심 특징으로 남겨, 트리 기반 모델이나 SVM과 같은 분류기를 통해도 뛰어난 성능을 발휘합니다.

- **Performance Highlights**: HIT-ROCKET은 miniROCKET 및 기존의 ROCKET 방법들과 비교할 때 가장 짧은 훈련 및 추론 시간을 자랑합니다. 실험 결과, F1-score가 ROCKET 대비 최소 5% 향상되었으며, 기존 알고리즘은 50% 더 짧은 훈련시간을 기록했습니다. 이 접근 방식은 저전력 임베디드 장치에 배포 가능하여, 마이크로컨트롤러 및 FPGA와 같은 플랫폼에서의 적용을 용이하게 합니다.



### Real-time Continual Learning on Intel Loihi 2 (https://arxiv.org/abs/2511.01553)
- **What's New**: CLP-SNN은 지속적인 학습을 위해 신경망 아키텍처를 발전시킨 혁신적인 솔루션입니다. 이 시스템은 에지 디바이스에서의 전력 제한 조건 하에서도 비정상적인 데이터 스트림에 따라 모델을 점진적으로 학습할 수 있도록 고안되었습니다. 기존의 재학습 방식이 아닌 온라인 지속적 학습(Online Continual Learning, OCL) 방식을 사용하여, 오프라인 모델에 비해 더 빨리 반응할 수 있도록 합니다.

- **Technical Details**: CLP-SNN은 세 가지 주요 혁신으로 구성됩니다: (1) 이벤트 기반(event-driven) 및 시공간적으로 희소한 Local Learning, (2) 가중치 정규화를 유지하는 자기 정규화(Self-normalizing) 세 가지 요인 학습 규칙, 그리고 (3) 용량 확장과 망각 완화를 위해 통합된 신경 생성(neurogenesis) 및 메타가소성(metaplasticity)입니다. 이는 Intel의 Loihi 2 칩에서 구현되어 빠른 처리 속도와 에너지 효율성을 제공합니다.

- **Performance Highlights**: OpenLORIS의 몇 샷 학습 실험에서 CLP-SNN은 재실행(replay) 방식보다 경쟁력 있는 정확도를 유지하면서 리허설이 필요 없는 접근 방식을 제공합니다. CLP-SNN은 이전 대안 OCL보다 70배 빠르고(0.33ms vs 23.2ms), 5,600배의 에너지 효율성(0.05mJ vs 281mJ)을 보여주었습니다. 이로써, 공동 설계된 뇌 영감을 받은 알고리즘과 신경형 하드웨어가 미래의 에지 AI 시스템에서 전통적인 정확도-효율성 간의 트레이드오프를 극복할 수 있음을 입증합니다.



### Driving scenario generation and evaluation using a structured layer representation and foundational models (https://arxiv.org/abs/2511.01541)
- **What's New**: 이 논문은 자율주행 차량 개발을 위한 희귀한 주행 시나리오의 평가 및 생성을 개선하기 위해 5계층 구조 모델을 제안합니다. 기존의 계층 모델을 기반으로 하여 각 에이전트의 하위 클래스와 특성을 도입함으로써 새로운 주행 시나리오를 생성하는 데이터 증강 전략을 사용합니다. 본 연구는 희귀한 시나리오의 개념을 명확히 하기 위해 Edge Cases (ECs)라는 용어를 사용하며, 이를 통해 주행 데이터셋의 다양성과 독창성을 평가할 수 있는 새로운 메트릭스를 제안합니다.

- **Technical Details**: 이 논문은 5계층 모델(5-layer model)을 사용하여 주행 시나리오를 효과적으로 표현합니다. 이 모델은 도로 구조, 도로 주변의 구조, 임시 변화, 동적 객체, 환경 조건의 다섯 가지 계층으로 구성되어 있어 주행 시나리오에 대한 표준화된 표현을 가능하게 합니다. 텍스트 표현은 벡터 임베딩(vector embedding)으로 변환되어 시나리오 간 유사성을 비교할 수 있는 기반을 제공합니다.

- **Performance Highlights**: 생성된 희귀 주행 시나리오는 기존 실제 데이터에서 특정 구성요소를 편집하여 Augmentation 방법을 통해 만들어집니다. 이를 통해 시나리오가 실제와 일치할 가능성을 높이며, 우리는 텍스트 기반 평가 방식을 통해 생성된 시나리오의 품질을 분석합니다. 다양한 실험 결과를 통해 이 모델의 효용성을 입증하며, 코드와 확장된 결과는 논문 내 제공된 URL에서 확인할 수 있습니다.



### BanglaNirTox: A Large-scale Parallel Corpus for Explainable AI in Bengali Text Detoxification (https://arxiv.org/abs/2511.01512)
Comments:
          Under review, 6 pages, 1 figure, 2 tables

- **What's New**: 이 논문에서는 벵골어에서의 독성 언어 문제를 해결하기 위한 새로운 파이프라인을 제안합니다. 텍스트 디톡시피케이션(text detoxification) 분야가 자원이 풍부한 언어에서는 발전했지만, 벵골어는 자원이 제한되어 있어 미흡합니다. 특히, 이 연구는 Pareto class-optimized large language models (LLMs)와 Chain-of-Thought (CoT) prompting을 결합하여 독성 문장을 정화하는 방법을 제시합니다.

- **Technical Details**: 논문에서 제안하는 BanglaNirTox 데이터셋은 68,041개의 독성 벵골어 문장으로 구성되어 있으며, 각 문장은 독성 레이블, 이성(reasoning), 그리고 디톡시피케이션된 패러프레이즈(paraphrase)를 포함합니다. 이 데이터셋은 Pareto-optimized LLMs를 사용하여 생성되었으며, 무작위 샘플을 기준으로 평가되었습니다. 구성된 데이터셋을 바탕으로 언어 모델을 파인튜닝(fine-tune)하여 벵골어 문장의 독성 제거 성능을 높입니다.

- **Performance Highlights**: 연구 결과, Pareto-optimized LLMs에 CoT prompting을 적용함으로써 벵골어 텍스트 디톡시피케이션의 품질과 일관성이 크게 향상됨을 확인했습니다. 이러한 접근 방식은 기존의 독성 언어 처리 방법에 비해 보다 효과적인 결과를 가져오는 것으로 나타났습니다.



### MO-SeGMan: Rearrangement Planning Framework for Multi Objective Sequential and Guided Manipulation in Constrained Environments (https://arxiv.org/abs/2511.01476)
Comments:
          8 pages, 8 figures, website:this https URL

- **What's New**: MO-SeGMan은 복잡한 재배치 문제를 해결하기 위해 다중 목표 순차적 및 유도 가능한 조작 플래너를 도입합니다. 이 시스템은 물체의 재배치를 위한 Placement sequence를 생성하여, 각 물체에 대한 재계획과 로봇 이동 거리를 최소화합니다. 특히, 클러터가 많은 비모노톤 환경에서 선택적 유도 전방 탐색(SGFS)을 채택하여 효율적으로 장애물을 이동과 재배치할 수 있도록 합니다.

- **Technical Details**: MO-SeGMan은 적응형 서브 목표 선택을 위한 정제 방법을 도입, 불필요한 피킹 및 배치 작업을 줄여 전체 솔루션 품질을 개선합니다. 이 시스템은 14개의 다양한 난이도의 재배치 시나리오에 대해 평가되었으며, 모든 경우에서 성능을 입증했습니다. 사용된 기법은 k차 마르코프 모션 최적화기(KOMO)와 이중 방향 탐색 무작위 트리(Bi-RRT)로, 물체의 Pick and Place 경로를 생성합니다.

- **Performance Highlights**: 무려 9개의 벤치마크 재배치 작업에 대한 평가 결과, MO-SeGMan은 모든 경우에 대해 실행 가능했던 모션 플랜을 생성하며 더 빠른 솔루션 시간과 우수한 솔루션 품질을 기록했습니다. 그러한 결과는 복잡한 재배치 계획 문제에 대한 제안된 프레임워크의 견고성과 확장성을 잘 보여 줍니다. MO-SeGMan은 기존 방법 대비 계산 시간을 줄이고 높은 솔루션 품질을 달성하여 축적된 성능 확인을 통해 그 유용성을 입증했습니다.



### DAMBench: A Multi-Modal Benchmark for Deep Learning-based Atmospheric Data Assimilation (https://arxiv.org/abs/2511.01468)
- **What's New**: 본 연구에서는 데이터를 기반으로 하는 데이터 동화(data assimilation, DA) 모델을 평가하기 위한 최초의 대규모 다중 모드 벤치마크 DAMBench를 소개합니다. DAMBench는 실제 대기 조건에서의 데이터 동화를 평가하기 위해 초고품질의 배경 상태와 다중 모드 관측 데이터를 통합합니다. 이를 통해 시스템 상태를 정확하게 재구성하고 비교할 수 있는 표준화된 기준을 설정하며, 향후 연구를 위한 엄격한 기반을 마련하고자 합니다.

- **Technical Details**: DAMBench는 유럽 중기 기상 예보 센터(ECMWF)에서 생성한 역사적 기상 상태를 기반으로 한 배경 상태와, 글로벌 기상 관측소에서 수집된 관측 데이터를 사용합니다. 또한, NOAA 위성에서 수집된 아웃고잉 롱웨이브 복사(outgoing longwave radiation) 이미지 데이터를 포함합니다. 모든 데이터는 일별 데이터 동화 주기를 지원하기 위해 시간적으로 정렬되어 있으며, 이를 통해 단일 모드 및 다중 모드 설정 아래에서 데이터 동화 접근법을 평가하고 있습니다.

- **Performance Highlights**: 실험 결과, DAMBench의 다중 모드 관측 데이터가 기존의 데이터 동화 모델 성능을 크게 향상시키는 것을 나타냈습니다. 간단한 다중 모드 표현 어댑터(multi-modal representation adapter) 플러그인을 사용하여 모델 성능을 검증한 결과, 실제 관측 데이터를 결합했을 때 성과가 두드러졌습니다. 이러한 결과들은 실제 관측 환경에 기반한 벤치마크의 필요성을 강조하며, 데이터 동화 분야의 공정한 비교와 재현성을 지원할 수 있는 기틀을 제공합니다.



### HMVLM: Human Motion-Vision-Lanuage Model via MoE LoRA (https://arxiv.org/abs/2511.01463)
Comments:
          10 pages, 5figures. The Thirty-Ninth Annual Conference on Neural Information Processing Systems

- **What's New**: 본 논문에서는 Human Motion-Vision-Language Model (HMVLM)이라는 새로운 통합 프레임워크를 제안합니다. 이 프레임워크는 Mixture of Expert Low-Rank Adaptation (MoE LoRA) 전략을 기반으로 하여 다양한 다운스트림 작업에서의 성능 개선을 목표로 합니다. 특히, 모션 관련 작업에서의 지식 잊음을 방지하기 위한 새로운 'zero expert' 개념이 도입되었습니다.

- **Technical Details**: HMVLM은 게이팅 네트워크를 활용하여 입력 프롬프트에 따라 LoRA 전문가 가중치를 동적으로 할당합니다. 이는 여러 작업의 동기화된 파인튜닝(fine-tuning)을 가능하게 하고, 동작과 관련 없는 작업에서 pretrained 파라미터를 보존합니다. 또한 인체를 부위별로 나누어 토큰화하여 공간적 해상도를 향상시키는 방법이 적용되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 HMVLM은 모션 관련 작업에서 강력한 성능을 보여주며, 텍스트-모션 생성, 단안 포즈 추정 및 모션 비디오 이해에서 우수한 성과를 달성했습니다. 지식 잊힘 현상이 효과적으로 완화되었으며, 다양한 인체 중심의 작업을 동시에 지원할 수 있는 안정적인 멀티모달 프레임워크로 자리매김하였습니다.



### Efficiently Training A Flat Neural Network Before It has been Quantizated (https://arxiv.org/abs/2511.01462)
Comments:
          ongoing work, more results would be added

- **What's New**: 본 논문에서는 포스트 훈련 양자화(Post-training quantization, PTQ)에서 발생하는 양자화 오차를 줄이기 위한 새로운 접근 방법인 차별적 노이즈 기반 양자화 인식 훈련(Differential Noise-driven Quantization-aware Training, DNQ)을 제안합니다. 기존의 방법들은 완전 정밀 모델과 양자화 모델 간의 관계를 간과하여 상당한 양자화 오차를 초래합니다. DNQ는 모델의 오류 원인을 측정하고 분리하여 안정적인 손실 경관을 유도할 수 있도록 설계되었습니다.

- **Technical Details**: DNQ 프레임워크는 양자화 오차, 즉 활성화 양자화 오차(Activation Quantization Error, AQE)와 가중치 양자화 오차(Weight Quantization Error, WQE)를 독립적인 가우시안 노이즈로 통계적으로 모델링합니다. 이 프레임워크는 훈련 중에 양자화 노이즈를 시뮬레이션하여 모델이 훈련될 수 있도록 하고, 최적화자가 단순한 PTQ에 대해 견고한 성능을 보이는 솔루션을 찾아내도록 유도합니다. 결과적으로, 손실 경관의 평탄함을 최적화하여 양자화에 내성이 강한 모델을 만들 수 있습니다.

- **Performance Highlights**: 실험 결과, DNQ가 적용된 모델은 기존의 복잡한 PTQ 알고리즘으로 최적화된 모델을 능가하는 성능을 보입니다. 특히, 여러 기준 데이터셋과 네트워크 아키텍처에서 일관되게 성능을 향상시켰습니다. 본 논문의 접근 방식은 향후 저비트 PTQ 모델을 획득하기 위한 새로운 방향을 제시하며, 특히 모델 압축을 필요로 하는 엣지 컴퓨팅 장치에 유용할 것입니다.



### When to Trust the Answer: Question-Aligned Semantic Nearest Neighbor Entropy for Safer Surgical VQA (https://arxiv.org/abs/2511.01458)
- **What's New**: 이 논문은 수술에서의 Visual Question Answering (VQA)의 안정성과 신뢰성을 개선하기 위해 불확실성 추정에 초점을 맞춥니다. QA-SNNE(Question Aligned Semantic Nearest Neighbor Entropy)라는 새로운 불확실성 추정기를 도입하여 질문의 의미를 포함한 예측 신뢰도를 측정합니다. 기존 surgical VQA 연구는 주로 정확도와 언어 품질에 집중했으나, 본 연구는 사용자 안전을 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: QA-SNNE는 의료 텍스트 임베딩 공간에서 생성된 답변과 가장 가까운 이웃을 비교하여 의미적 엔트로피를 측정합니다. 이를 통해 주어진 질문에 대한 답변의 통일성을 평가하며, 과거 연구에서 사용된 불확실성 기반 접근 법과는 다르게 질문에 맞춤형 분석을 수행합니다. 다섯 가지 모델을 평가하며, PEFT 모델과 zero-shot LVLM을 포함하여 다양한 사전 학습된 모델에 대해 실험을 진행합니다.

- **Performance Highlights**: QA-SNNE는 다양한 테스트 환경에서 구현되었으며, 특히 out-of-template 상황에서 PEFT 모델이 약간 손상되는 반면 LVLM은 더 높은 내구성을 보입니다. AUROC(Receiver Operating Characteristic 곡선 아래 면적)는 zero-shot 모델에서 15-38% 증가하여 안전성을 높이는 데 기여합니다. 이 연구는 QA-SNNE를 통해 수술 VQA의 안전성과 신뢰성을 높일 수 있는 실질적이고 해석 가능한 단계로 제시합니다.



### Reg-DPO: SFT-Regularized Direct Preference Optimization with GT-Pair for Improving Video Generation (https://arxiv.org/abs/2511.01450)
- **What's New**: 이 연구는 비디오 생성 품질을 향상시키기 위한 새로운 직접 선호 최적화 방법인 GT-Pair와 Reg-DPO를 제안합니다. GT-Pair는 실비디오와 생성된 비디오를 통해 고품질의 선호 쌍을 자동으로 생성하며, Reg-DPO는 SFT 손실을 DPO 목표에 정규화 항으로 포함시켜 훈련 안정성과 생성 충실도를 높입니다. 이러한 접근방법은 메모리 최적화 기법과 결합되어 훈련 능력을 최대 세 배 향상시킵니다.

- **Technical Details**: GT-Pair는 실비디오를 긍정 샘플로, 모델 생성을 부정 샘플로 이용하여 별도 외부 주석 없이 높은 품질의 선호 데이터를 생성합니다. Reg-DPO는 DPO 손실에 SFT 기반 정규화 항을 포함시켜, 훈련 중 발생하는 불안정성을 해결하고 모델 성능을 향상시킵니다. 이 연구는 Flash Attention, Context Parallelism과 같은 여러 메모리 최적화 기법을 결합하여 GPU 메모리 사용량을 줄이는 효과적인 메모리 절약 전략을 구현합니다.

- **Performance Highlights**: I2V(이미지에서 비디오)와 T2V(텍스트에서 비디오) 작업에서 광범위한 실험을 수행한 결과, 제안된 방법이 기존 방법들보다 일관되게 우수한 비디오 생성 품질을 달성한 것으로 나타났습니다. SFT 정규화는 긍정 및 부정 샘플의 출력 분포를 동시에 감독하여 모델이 선호 신호를 학습하도록 하면서도 분포 이동을 통제합니다. 제안된 접근 방식은 현재까지의 연구에서 가장 뛰어난 성과를 기록하며, 모델의 훈련과정에서 안정성을 크게 향상시킵니다.



### Privacy Preserving Ordinal-Meta Learning with VLMs for Fine-Grained Fruit Quality Prediction (https://arxiv.org/abs/2511.01449)
Comments:
          9 pages, 1 figure, 4 tables

- **What's New**: 이 논문에서는 과일의 신선도를 비침습적으로 예측하기 위한 새로운 방법인 Model-Agnostic Ordinal Meta-Learning (MAOML) 알고리즘을 소개합니다. 기존의 독점적인 Vision Language Models (VLMs)와는 달리, 이 접근법은 적은 양의 데이터로도 높은 성능을 낼 수 있도록 설계되었습니다. 제안된 방법은 다양한 과일에 대해 92.71%의 정확도로 최신 기술을 구현했으며, 데이터 프라이버시 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: MAOML 알고리즘은 메타 학습(meta-learning) 기법을 활용하여 과일의 신선도 분류 작업에서 데이터 부족 문제를 해결합니다. 이를 통해 소규모 VLM 모델을 훈련시키며, 라벨의 순서(ordinality)를 활용하여 성능을 개선합니다. 본 연구에서는 10종의 과일에 대해 고유한 품질 클래스인 'Unripe', 'Early ripe', 'Ripe', 'Overripe', 'Bad'를 정의하고, 이를 기반으로 300개의 이미지를 수집하였습니다.

- **Performance Highlights**: MAOML을 사용하여 훈련된 소규모 VLM 모델은 기존의 대형 VLM보다 성능이 뛰어난 것으로 나타났습니다. 제안된 방법은 zero-shot 및 few-shot 상황 모두에서 높은 정확성을 기록하며, 실험 결과는 이 방법이 음식 소매 산업에서의 신선도 판단을 실용적으로 적용하는 데 큰 도움이 될 것임을 시사합니다. 이를 통해 데이터 프라이버시 및 성능 간의 균형을 유지하는 효과적인 AI 솔루션이 가능해집니다.



### UniSOT: A Unified Framework for Multi-Modality Single Object Tracking (https://arxiv.org/abs/2511.01427)
Comments:
          The paper has been accepted by TPAMI

- **What's New**: 이번 논문에서는 다양한 참조 모드 (reference modalities)와 비디오 모드 (video modalities) 조합을 처리할 수 있는 통합 추적기 (unified tracker), UniSOT을 제안합니다. 이 모델은 기존의 특정 모드에 제한되지 않고, 블록 박스 (bounding box), 자연어 (natural language) 및 두 가지 모두를 사용하여 비디오 모드에 걸쳐 목표 객체를 추적할 수 있습니다. UniSOT은 18개의 시각 추적 (visual tracking) 및 비전-언어 추적 (vision-language tracking) 벤치마크에서 뛰어난 성과를 보여 주었습니다.

- **Technical Details**: UniSOT는 참조 모드 및 비디오 모드를 통합하기 위해 설계된 두 가지 모듈을 포함합니다. 참조 모드 설계를 위해, 우리는 Transformer 기반의 새로운 구조를 통해 시각 및 언어의 특징을 결합하는 범용 피처 추출기 (feature extractor)를 개발하였습니다. 비디오 모드 설계를 위해서는, 다양한 보조 비디오 모드에서 강력하게 작동할 수 있도록 랭크 조정 모드 적응 메커니즘 (rank-adaptive modality adaptation)을 적용하였습니다.

- **Performance Highlights**: 실험 결과, UniSOT는 모든 비디오 모드 및 참조 모드에 대해 이전의 통계적 대비 결과보다 약 3.0% AUC 향상을 달성하였습니다. 특히, RGB+X 비디오 모드에서 Un-Track을 초과하여 약 2.0% 더 우수한 주요 메트릭을 보였습니다. 이를 통해 UniSOT는 다양한 사용자의 요구 사항을 만족시키며 실질적인 응용 프로그램에 유연성을 제공합니다.



### FoldPath: End-to-End Object-Centric Motion Generation via Modulated Implicit Paths (https://arxiv.org/abs/2511.01407)
Comments:
          Accepted at 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)

- **What's New**: 새로운 연구인 FoldPath는 Object-Centric Motion Generation (OCMG) 분야에서 혁신적인 접근법을 제시합니다. 기존의 방법들이 주로 불안정한 post-processing 과정에 의존했던 반면, FoldPath는 엔드 투 엔드(end-to-end) 훈련 파이프라인을 사용하여 로봇의 움직임을 연속 함수로 학습합니다. 이 방식은 예측된 경로를 자연스럽고 유연하게 생성할 수 있게 해 주며, 기존 기술보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: FoldPath는 3D 점 구름(point cloud) 데이터를 입력으로 사용하여 스프레이 페인팅 로봇의 경로를 생성합니다. 모델은 객체의 점 구름을 압축된 표현으로 인코딩하고, 이를 미리 정의된 경로 임베딩을 통해 디코딩하여 로봇의 6D(end-effector의 6차원 포즈) 벡터 필드를 가이드 합니다. 이때 각 코드워드는 하나의 경로를 형성하는 여러 개의 포즈를 생성하는 데 활용되며, 모델은 각 경로에 대해 신뢰도 점수를 예측하여 가장 적합한 경로를 선택할 수 있게 합니다.

- **Performance Highlights**: FoldPath는 PaintNet 데이터셋에서 최첨단의 예측 성능을 달성하는 것으로 입증되었습니다. 실험을 통해 70개의 전문가 샘플만으로도 실질적인 산업 환경에서 일반화 능력을 보여주었으며, 스프레이 페인팅 시뮬레이션에서 강력한 내구성과 효과성을 확인하였습니다. 이러한 성과는 FoldPath가 실제로 적용 가능하다는 높은 기술 준비 수준을 나타냅니다.



### SEPS: Semantic-enhanced Patch Slimming Framework for fine-grained cross-modal alignmen (https://arxiv.org/abs/2511.01390)
- **What's New**: 본 논문에서는 정밀한 비전과 언어 간의 로컬 대응관계를 형성하는 세분화된 크로스 모달 정렬(fine-grained cross-modal alignment)을 소개합니다. 기존 방법들이 직면하고 있는 패치 중복(patch redundancy)과 모호성(ambiguity) 문제를 해결하기 위해, 논문에서는 의미 향상 패치 슬리밍(Semantic-Enhanced Patch Slimming, SEPS) 프레임워크를 제안합니다. SEPS는 밀집 텍스트(dense text)와 희박 텍스트(sparse text) 간의 통합된 의미를 활용하여 중요한 시각적 패치를 식별하는 데 도움을 줍니다.

- **Technical Details**: SEPS 프레임워크는 두 단계 메커니즘을 통해 구성됩니다. 첫 번째 단계에서는 밀집 텍스트와 희박 텍스트의 특징 표현을 활용하여 시각적 패치를 추출하고, 두 번째 단계에서는 희박 텍스트와 밀집 텍스트에서 파생된 통합된 의미를 바탕으로 시각적 패치를 선택합니다. 이를 통해 HEPA(Highly-Relevant Patch-Word Alignment) 모듈이 적용되어, 중요한 패치-단어 대응관계의 선택 및 평균값 계산을 통해 정밀한 인터랙션을 개선합니다.

- **Performance Highlights**: Flickr30K 및 MS-COCO 데이터셋에서 SEPS 프레임워크를 평가한 결과, 기존 접근 방식을 23%-86% 초과하는 성능 개선을 이루었으며, 특히 텍스트-이미지 검색 시나리오에서 두드러진 향상을 보였습니다. 이 연구는 MLLMs를 활용하여 크로스 모달 정렬을 위한 패치 선택을 효과적으로 지원하는 첫 번째 체계적인 프레임워크로 자리매김하고 있습니다.



### RAGSmith: A Framework for Finding the Optimal Composition of Retrieval-Augmented Generation Methods Across Datasets (https://arxiv.org/abs/2511.01386)
Comments:
          45 pages

- **What's New**: RAGSmith는 레트리벌-증강 생성(Retrieval-Augmented Generation, RAG)의 통합 최적화를 위한 모듈형 프레임워크로, 46,080개의 파이프라인 구성에서 발생하는 상호작용을 고려합니다. 이 연구는 RAG 시스템의 구성 요소 간의 복잡한 synergies에 초점을 맞추며, 각 모듈을 독립적으로 최적화하는 전통적인 접근 방식을 재편합니다. RAGSmith는 유전 알고리즘을 통해 완전한 파이프라인 최적화를 가능하게 하며, 다양한 도메인에서 일관된 성능 향상을 나타냅니다.

- **Technical Details**: RAGSmith는 9개의 기술 군을 포함하여 46,080개의 구성 조합을 평가합니다. 이 시스템은 레트리벌 메트릭(retrieval metrics)과 생성 메트릭(generation metrics) 모두를 통합하여 최적화를 진행하고, 다양한 질문 유형에 대한 반응성을 기반으로 전반적인 성능 향상을 목표로 합니다. 또한, RAGSmith는 질문 타입에 민감한 최적화 지침을 설정하여, 도메인별 데이터셋 특성에 맞춰 최적의 구성 요소 조합을 찾아냅니다.

- **Performance Highlights**: RAGSmith는 기본 RAG 베이스라인 대비 평균적으로 +3.8%의 성능 향상을 달성하며, 특정 도메인에서는 최대 +12.5%의 레트리벌 개선 및 +7.5%의 생성 개선을 나타냅니다. 실험 결과, 장기 답변 및 사실 기반 질문 세트에서 더 큰 개선이 이루어졌으며, 이는 RAG 시스템 구성에서 질문 유형의 중요성을 강조합니다. 이 결과는 evolucionary search의 장점을 입증하며, 효과적인 RAG 시스템 구성 방법에 대한 실용적 가이드를 제공합니다.



### PrefixNLI: Detecting Factual Inconsistencies as Soon as They Aris (https://arxiv.org/abs/2511.01359)
Comments:
          9 pages + appendix. Code, datasets, and models are available at this https URL

- **What's New**: 이번 논문에서는 자연어 추론(NLI) 모델을 통해 LLM의 결과의 사실성(factuality)을 개선하는 새로운 방법을 제안합니다. 자동 회귀 생성(autoregressive generation)에서 각 텍스트 접두사(prefix)에 대해 entailment를 평가하는 PrefixNLI 작업을 소개하며, 이를 통해 MiniTruePrefixes라는 새로운 모델을 개발하였습니다. 이 모델은 이전 NLI 모델보다 5에서 14 F1 포인트 더 높은 성능을 보이며, 추상 요약(abstractive summarization) 시의 사실적 일관성을 크게 개선합니다.

- **Technical Details**: MiniTruePrefixes는 텍스트 접두사에 대해 사실적 불일치를 더 잘 감지하도록 훈련된 전문 NLI 모델입니다. 이 모델은 훈련 및 평가 데이터셋을 제공하여 PrefixNLI 작업을 위한 새로운 기준을 세우고, 이를 통해 사실성과 속도에서 효율성을 유지하면서도 우수한 성능을 발휘합니다. 기존의 NLI 모델의 한계를 극복하고, 각 텍스트 접두사에 직접적으로 depender scoring feedback를 제공하는 접근 방식이 중요한 기술적 기여로 설명됩니다.

- **Performance Highlights**: MiniTruePrefixes는 LLaMA-3.2-3B-Instruct 모델과의 비교에서 8B 모델과 비슷한 사실성(Faithfulness) 및 실행 속도를 유지하면서도 메모리 소비를 절반으로 줄인 성과를 보여줍니다. 이는 더 작은 모델을 사용하더라도 사실적인 내용 생성을 가능하게 하며, 다양한 모델 크기 및 데이터셋에서 일관된 사실성 향상을 증명합니다. 결과적으로, prefix 기반 NLI를 통해 텍스트 생성의 사실성을 향상시킬 수 있는 더 넓은 잠재력을 제시합니다.



### CMI-MTL: Cross-Mamba interaction based multi-task learning for medical visual question answering (https://arxiv.org/abs/2511.01357)
Comments:
          The paper has been accepted by the 33rd Pacific Conference on Computer Graphics and Applications (Pacific Graphics 2025)

- **What's New**: 이번 연구에서는 의료 분야의 시각적 질문 답변(Med-VQA)을 위한 새로운 Cross-Mamba Interaction 기반의 다중 작업 학습(CMI-MTL) 프레임워크를 제안합니다. 기존의 self-attention 방법이 존재하는 한계성을 극복하고, 이미지와 텍스트 간의 cross-modal semantic alignments를 효과적으로 처리하기 위한 접근 방식입니다. CMI-MTL은 이미지와 텍스트로부터 교차 모달 특징 표현을 학습하며, 이를 위해 FVTA, CIFR, FFAE의 세 가지 핵심 모듈로 구성되어 있습니다.

- **Technical Details**: CMI-MTL은 FVTA(Fine-grained Visual-Text Feature Alignment), CIFR(Cross-Modal Interleaved Feature Representation), FFAE(Free-Form Answer-Enhanced Multi-Task Learning) 모듈로 구성됩니다. FVTA는 이미지-텍스트 쌍에서 가장 관련성이 높은 지역을 추출하고, CIFR은 cross-modal 상호작용을 동적으로 포착합니다. FFAE는 개방형 질문에서 보조 지식을 활용하여 모델의 개방형 Med-VQA 능력을 향상시키는 역할을 합니다.

- **Performance Highlights**: 실험 결과, CMI-MTL은 VQA-RAD, SLAKE, OVQA의 세 가지 Med-VQA 데이터셋에서 기존의 최첨단 방법들을 초월한 성능을 보여주었습니다. 또한 Grad-CAM을 사용한 해석 가능성 실험을 통해 모델이 질문과 가장 관련 있는 이미지 영역에 집중하고 있음을 입증했습니다. 이러한 결과는 CMI-MTL의 효과성과 해석 가능성을 높이는 데 기여합니다.



### Thinking with DistilQwen: A Tale of Four Distilled Reasoning and Reward Model Series (https://arxiv.org/abs/2511.01354)
Comments:
          emnlp 2025 industry track

- **What's New**: 최근의 산업 요구에 부응하여 DistilQwen 모델 시리즈가 확장되었으며, 네 가지 모델 시리즈가 도입되었습니다. 이 모델들은 고도화된 산업 환경에서의 응용을 목표로 하며, 각각 느린 사고 모델, 적응형 사고 모델, 그리고 증강된 보상 모델로 구성됩니다. 느린 사고 모델은 높은 정확도가 필요한 작업에 최적화되어 있으며, 적응형 사고 모델은 입력 작업에 따라 reasoning 전략을 동적으로 조정하여 다양한 시나리오에서 효율성을 극대화합니다.

- **Technical Details**: DistilQwen 모델은 데이터 소스 수집기(Data Source Collector)를 기반으로 하여 CoT(train of thought) 데이터셋을 집계하고, 다양한 도메인에서 훈련을 위한 풍부한 소스를 제공합니다. 이 논문에서는 SFT(supervised fine-tuning) 훈련 기술을 섬세하게 조정하고, CoT 생성 과정에서 다양한 성능 요건을 충족하기 위한 인프라를 구축한 방법론을 설명합니다. 느린 사고 모델을 위해 DeepSeek-R1을 사용하여 CoT를 생성하고, CoT 난이도 평가 시스템을 통해 학습 목표를 설정하였습니다.

- **Performance Highlights**: 모델 성능 평가 결과, DistilQwen 모델은 높은 추론 효율성과 강력한 reasoning 성능을 보였습니다. 특히, 적응형 사고 모델과 증강된 보상 모델은 실질적으로 산업 AI 플랫폼에서의 적용 가능성을 보여주었습니다. Alibaba Cloud PAI 플랫폼에서의 통합을 통해 이 모델들은 기업의 AI 솔루션으로서의 실용성을 증명했습니다.



### AI Literacy in UAE Libraries: Assessing Competencies, Training Needs, and Ethical Considerations for the Digital Ag (https://arxiv.org/abs/2511.01353)
Comments:
          This is the accepted manuscript version. The final published version will appear in College & Research Libraries, November 2026

- **What's New**: 이 연구는 아랍에미리트(UAE)의 도서관 정보학(LIS) 전문가들 사이에서 인공지능(AI) 문해력 수준을 조사했습니다. 92명의 설문조사를 통해 현재 AI 기술에 대한 인식과 실무 능력을 평가하였습니다.

- **Technical Details**: 연구 방법론으로는 정량적 접근법(quantitative approach)을 사용하였으며, 인지 기술(cognitive competencies)은 매우 강했으나, 행동 기술(behavioral competencies)과 규범 기술(normative competencies)에서는 부족한 점이 발견되었습니다.

- **Performance Highlights**: 특히 AI 편향(AI biases), AI 기반 학습(AI-powered learning), 윤리적 고려사항(ethical considerations)과 같은 부분에서의 격차가 두드러졌습니다. AI 기술의 중요성과 현재 교육 프로그램의 효과 사이에는 간극이 존재하는 것으로 나타났습니다.



### The Future of Generative AI in Software Engineering: A Vision from Industry and Academia in the European GENIUS Projec (https://arxiv.org/abs/2511.01348)
Comments:
          Submitted to 2nd IEEE/ACM International Conference on AI-powered Software (AIware 2025)

- **What's New**: 생성적 AI(GenAI)는 소프트웨어 엔지니어링에서 코드 생성, 수정 제안 및 품질 보증 지원 등에서 혁신적인 역할을 하고 있습니다. 그러나 소프트웨어 개발 수명주기(SDLC)의 모든 단계에서 GenAI의 적용에 대한 연구는 충분히 진행되지 않았습니다. 특히 신뢰성, 책임, 보안, 데이터 프라이버시와 같은 중요한 불확실성 문제는 더 깊은 조사와 협력된 행동을 요구합니다.

- **Technical Details**: GENIUS 프로젝트는 30개 이상의 유럽 산업 및 학술 파트너로 구성되어 있으며, SDLC 전반에 걸친 AI 통합을 발전시키는 것을 목표로 합니다. 이 논문은 GenAI의 채택에 대한 구조화된 개요와 향후 5년에 걸쳐 기대되는 기술적 진전을 제시합니다. 또한 소프트웨어 전문가의 역할 변화와 필요한 기술 집합의 변화를 않을 스케치하고 GENIUS의 기여를 통해 이러한 변화를 실현할 계획을 설명합니다.

- **Performance Highlights**: GenAI는 특히 코드 생성 단계에서 소프트웨어 엔지니어링 관행을 변환할 잠재력을 이미 입증했습니다. 그러나 LLM의 제한된 맥락 인식과 이해 능력으로 인해 SDLC 전반에서의 적용에는 여전히 여러 도전 과제가 존재합니다. 향후 연구와 산업 전략을 통합하여 신뢰할 수 있고 확장 가능한 GenAI 솔루션을 제공하는 것이 중요합니다.



### Beyond Permissions: Investigating Mobile Personalization with Simulated Personas (https://arxiv.org/abs/2511.01336)
Comments:
          8 pages, 7 figures. Accepted to the ACM Workshop on Human-Centered AI Privacy and Security (HAIPS @ CCS 2025). DOI: https://doi.org/10.1145/3733816.3760758 (ACM Digital Library link pending activation)

- **What's New**: 이 논문에서는 모바일 애플리케이션이 사용자 개인화 경험을 위한 센서 데이터를 어떻게 활용하는지를 투명하게 시뮬레이션하고 분석할 수 있는 샌드박스 시스템을 제안합니다. 사용자들은 센서 스푸핑(sensor spoofing)과 페르소나 시뮬레이션(persona simulation)을 통해 애플리케이션의 반응을 시각화하고, 데이터 수집에 대한 이해도를 높이는데 기여할 수 있습니다. 이 시스템은 사용자에게 다중 센서 프로필을 실시간으로 주입하여 환경 변화에 따른 앱 반응을 관찰할 수 있게 합니다.

- **Technical Details**: 이 논문에서 제안하는 시스템은 안드로이드 기기에서 실시간으로 다중 센서 데이터 프로필을 생성하여 사용자 인터페이스(UI)를 요약하는 GPT-4 비전(GPT-4 Vision) 기반 기능을 포함합니다. 사용자는 활동 수준, 위치 변화 및 시간대 변경에 따른 앱의 반응을 자동 스크린샷 캡처를 통해 기록할 수 있습니다. 초기 실험 결과, 이 시스템은 다양한 모바일 앱들이 어떻게 사람의 행동 맥락에 따라 반응하는지를 실질적으로 보여줍니다.

- **Performance Highlights**: 초기 연구 결과는 다양한 애플리케이션, 예를 들어 피트니스, 전자상거래, 날씨 및 내비게이션과 같은 일반 서비스 앱이 데이터 스푸핑에 대해 유의미하게 반응한다는 것을 보여주었습니다. 따라서, 본 연구는 프라이버시 향상 기술과 사용자 투명성 개입을 위한 중요한 기초를 제공하게 됩니다. 추가적으로 이 연구는 사용자 교육과 감시를 위한 미래 도구의 가능성을 열어주는 길잡이 역할을 할 것으로 기대됩니다.



### Embodied Cognition Augmented End2End Autonomous Driving (https://arxiv.org/abs/2511.01334)
Comments:
          24 pages,4 pages

- **What's New**: 최근의 연구에서는 비전 기반의 엔드 투 엔드 자율 주행 기술이 새로운 패러다임으로 대두되고 있다. 이 논문에서는 $E^{3}AD$라는 새로운 패러다임을 제안하며, 이 모델은 시각적 특징 추출 네트워크와 일반 EEG 대형 모델 간의 비교 학습을 통해 인간의 운전 인지를 학습하도록 한다. 이러한 접근 방식은 자율 주행 모델의 일반성 및 적용 가능성을 높이는 데 기여할 것으로 기대된다.

- **Technical Details**: 제안된 E^{3}AD 패러다임은 Driving-Thinking Model을 활용하여 시공간적 특징 추출 네트워크를 훈련하는 두 단계의 접근법을 사용한다. 첫 번째 단계에서는 LaBraM의 인지 특징을 사용하여 전체 자율 주행 데이터 세트를 통한 분위기 비공식 학습을 수행하고, 두 번째 단계에서는 일반적인 자율 주행 데이터 세트를 활용하여 공정성을 유지하며 훈련 및 테스트를 진행한다.

- **Performance Highlights**: 실험 결과, E^{3}AD 패러다임은 기본 모델의 엔드 투 엔드 계획 성능을 유의미하게 향상시킴을 보여주었다. 아블레이션 연구를 통해 운전 인지의 기여와 비교 학습 과정의 효과성을 추가로 검증하였다. 또한, 본 연구는 인지 데이터를 엔드 투 엔드 자율 주행을 개선하는 데 통합한 첫 번째 작업이라는 점에서 주목할 만한 의미가 있다.



### AI for Requirements Engineering: Industry adoption and Practitioner perspectives (https://arxiv.org/abs/2511.01324)
Comments:
          Accepted at the Intelligent Software Engineering (ISE) 2025 Workshop at the Automated Software Engineering (ASE) 2025 Conference

- **What's New**: 이 연구는 요구사항 엔지니어링(Requirements Engineering, RE)에서 인공지능(AI)의 채택 현황을 조사하였다. 연구는 55명의 소프트웨어 전문가를 대상으로 AI의 활용을 네 가지 RE 단계인 엘리시테이션(elicitation), 분석(analysis), 명세(specification), 검증(validation) 및 의사결정을 위한 네 가지 접근 방식으로 정의하였다. 조사 결과, 응답자의 58.2%가 이미 RE에서 AI를 사용하고 있으며, 69.1%가 그 영향이 긍정적이라고 응답하였다.

- **Technical Details**:  요구사항 엔지니어링은 성공적인 프로젝트의 기초로 간주되며, AI 도구와 모델을 사용하여 효율성을 향상시키는 여러 방법이 고려되고 있다. 연구는 AI의 활용을 네 가지 접근 방식으로 분류했으며, AI와 인간의 협업(Human–AI Collaboration, HAIC)을 강조하였다. 조사에서 AI 도구의 사용 패턴, 책임 있는 AI 사용 관행, 그리고 실무자들이 AI 통합에서 경험하는 도전과 기회를 탐구하였다.

- **Performance Highlights**: 연구에 따르면, 현재 AI는 주로 인간의 의사결정을 지원하는 방식으로 활용되고 있으며, 완전한 AI 자동화는 5.4%에 불과하다. 이 연구는 AI가 협업 파트너로 기능할 때 가장 효과적이라는 점을 부각시키며, 향후 RE 프로세스 전반에 걸쳐 AI 기반 개선의 잠재력을 논의하였다. 또한, RE-specific HAIC 프레임워크와 견고하고 책임 있는 AI 거버넌스의 필요성을 강조하였다.



### DEEPAMBIGQA: Ambiguous Multi-hop Questions for Benchmarking LLM Answer Completeness (https://arxiv.org/abs/2511.01323)
Comments:
          25 pages

- **What's New**: 본 연구에서는 DeepAmbigQAGen이라는 자동 데이터 생성 파이프라인을 소개하고, 이를 통해 이름의 모호성과 다단계 추론을 요구하는 QA 작업을 생성합니다. 또한 DeepAmbigQA라고 하는 3,600개의 질문으로 구성된 데이터셋을 구축하여, 이 데이터셋은 복잡한 질문에 대한 정확한 정답 집합을 생성하기 위한 도전 과제를 제시합니다. 실험 결과, 최신 LLM인 GPT-5조차도 모호한 질문에 대한 답변의 일치율이 매우 낮아, 정보 수집 및 답변의 완전성을 위한 강력한 QA 시스템의 필요성을 강조합니다.

- **Technical Details**: DeepAmbigQAGen은 텍스트 코퍼스와 지식 그래프에 기반하여 모호한 질문을 생성하는 자동화된 파이프라인입니다. 이 파이프라인은 모호한 이름과 그에 해당하는 개체를 식별하고, 사용자의 정보 탐색 행동을 모델링한 실행 가능한 추론 계획을 구성합니다. 각 질문은 최소 두 단계의 추론을 요구하며, 일부는 최대 여덟 단계의 복잡한 추론을 포함하여, LLM QA 시스템에 상당한 도전 과제를 제시합니다.

- **Performance Highlights**: 최신 LLM을 평가한 결과, 특히 모호한 쿼리에 대한 정확한 답변 일치율이 0.13에 불과하여 낮은 성능을 보였습니다. 반면, 비모호한 질문에서도 0.21로 그 성능이 낮아, LLM의 전체적인 답변 수집에서 실패하는 모습을 보였습니다. 연구 결과, 질의 확장 및 증거 추출과 같은 모듈을 추가하더라도 이와 같은 복잡한 질문에 대한 문제를 완전히 해결하지 못함이 밝혀졌습니다.



### Exploringand Unleashing the Power of Large Language Models in CI/CD Configuration Translation (https://arxiv.org/abs/2511.01316)
- **What's New**: 이 연구는 Travis CI에서 GitHub Actions로의 CI 구성 마이그레이션에 대한 LLM 기반 번역의 효용을 분석합니다. 이 방식은 기존의 CI 구성 도구들이 가지고 있던 제한 사항을 극복할 수 있는 가능성을 보여줍니다. 구체적으로, 본 논문은 수행된 811건의 마이그레이션 기록을 통해 정확한 노력의 양을 수치화하고 CI 번역에서 발생하는 대표적인 문제점을 식별하였습니다, 이는 기존 연구와는 차별화된 접근입니다.

- **Technical Details**: CI/CD는 현대 소프트웨어 개발에서 자주 강조되는 용어로, CI는 코드 변경 시 자동으로 빌드를 수행하고, CD는 소프트웨어 릴리스 과정을 자동화하는 것을 의미합니다. 많은 CI 플랫폼은 이러한 CI와 CD를 통합적으로 지원하며, CI 구성을 YAML 구문으로 표현하는 경우가 많습니다. 이 연구에서는 LLM을 이용한 CI 구성 번역의 한계와 가능성을 분석하며, 번역 과정에서 발생한 1,121개 문제를 논의합니다.

- **Performance Highlights**: 연구 결과, LLM 기반의 CI 번역 성능이 초기 기본 프롬프트보다 약 3배 향상된 75.5%의 성공률을 기록했습니다. 특히, 가이드라인 기반의 프롬프트와 반복적인 정제를 결합한 전략이 가장 높은 성능을 발휘했습니다. 이러한 발견은 CI 구성 번역의 자동화에서 LLM의 잠재력을 강조하며, 도구 개발에 중요한 기초 자료를 제공합니다.



### Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models (https://arxiv.org/abs/2511.01307)
Comments:
          26 pages, 9 figures, 16 tables, NeurIPS 2025

- **What's New**: 이번 논문에서는 Anti-Personalized Diffusion Models (APDM)이라는 새로운 프레임워크를 제안하여 특정 주제의 개인화 기능을 차단하는 것을 목표로 합니다. 기존의 데이터-중심 방법들과 달리, APDM은 이미지가 아닌 확산 모델 자체를 보호 대상으로 설정합니다. 이를 통해 개인화의 무단 시도를 방지하고, 생성 성능은 유지할 수 있습니다.

- **Technical Details**: APDM의 구현에서는 Direct Protective Optimization (DPO)이라는 새로운 손실 함수를 도입하여 개인화 과정을 방해합니다. 또한, Learning to Protect (L2P)이라는 이중 경로 최적화 전략을 통해 개인화 경로와 보호 경로를 번갈아 가며 학습하도록 설계되었습니다. 이 방법은 모델이 개인화 시도를 예측하고 이에 따라 적응적으로 보호 조치를 강화할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, APDM은 다양한 개인화 주제에서 무단 개인화를 방지하는데 있어 기존 방법들을 능가하며 최신 성과를 기록했습니다. 즉, APDM은 실세계 환경에서 개인화로 인한 위협을 효과적으로 차단하며, 그 성능 또한 유지됩니다. 이러한 결과는 APDM의 효과성을 강화하는 데 중요한 증거가 됩니다.



### DeepSpecs: Expert-Level Questions Answering in 5G (https://arxiv.org/abs/2511.01305)
- **What's New**: DeepSpecs는 5G 사양에 대한 전문가 수준의 질문답변을 가능하게 하는 새로운 RAG 시스템입니다. 이 시스템은 3개의 메타데이터 중심 데이터베이스인 SpecDB, ChangeDB, TDocDB를 통해 구조적 및 시간적 추론을 강화합니다. DeepSpecs는 문서 내 서로 다른 조항의 교차 참조를 명시적으로 해결하며, 사양 발전 과정을 추적하여 변경 사항을 기록된 변경 요청과 연결합니다.

- **Technical Details**: DeepSpecs는 5G 사양의 복잡한 요구를 충족하기 위해 구조적 및 시간적 이해를 통합합니다. 클로즈(Clause) 수준의 교차 참조 해결과 사양 발전 추론을 제공하는 두 가지 기능을 통해 전문가가 표준을 탐색하는 방식을 모방합니다. 이 시스템은 변경 내역, 조항 간 링크 및 3GPP 특정 메타데이터를 기반으로 하여 기능의 도입 및 진화를 추적합니다.

- **Performance Highlights**: DeepSpecs는 강력한 LLM 기준 모델 및 통신 전용 RAG 시스템들과 비교하여 일관된 성과 향상을 보여주었습니다. 우리는 573개의 질문-답변 쌍으로 구성된 실제 데이터 세트를 통해 시스템의 효율성을 평가하며, 교차 참조 해결과 사양 발전 추론의 이점을 입증합니다. 또한, 전문적으로 주석이 달린 최초의 5G QA 데이터 세트를 소개하여 5G 사양에 대한 이해를 돕습니다.



### LSHFed: Robust and Communication-Efficient Federated Learning with Locally-Sensitive Hashing Gradient Mapping (https://arxiv.org/abs/2511.01296)
- **What's New**: 본 논문에서는 LSHFed라는 통신 효율적이며 강력한 연합 학습(FL) 프레임워크를 제안합니다. LSHFed는 데이터의 프라이버시를 보호하면서도 악의적인 경량 모드(GP) 감지를 통해 집계의 견고성을 강화합니다. 이 프레임워크의 핵심은 고차원 그래디언트를 압축된 이진 표현으로 변환하는 새로운 LSHGM(Locally-Sensitive Hashing Gradient Mapping) 알고리즘입니다.

- **Technical Details**: LSHFed는 VR(Verifier), AG(Aggregators), LT(Local Trainers)의 세 가지 역할로 참여자를 분류합니다. 데이터가 있는 노드에서 훈련 참가하는 LT와 데이터가 없는 노드로 구성된 AG가 서로 협력합니다. LSHGM을 통해 AG는 LT의 그래디언트를 집계하여 VR로 전송하며, VR은 최종적으로 글로벌 모델을 집계하고 훈련 과정을 진행합니다.

- **Performance Highlights**: LSHFed는 공격자들이 최대 50%의 비협조 참여자 일지라도 높은 모델 성능을 유지합니다. 1000배의 전송 비용 감소를 달성하면서도 다양한 보안 및 유틸리티 메트릭에서 기존 방식들을 초과하는 성능을 보입니다. 이는 LSHFed가 FL 과제에서 악의적인 공격에 대해 저항력을 유지하고 있음을 보여줍니다.



### Adaptation of Foundation Models for Medical Image Analysis: Strategies, Challenges, and Future Directions (https://arxiv.org/abs/2511.01284)
- **What's New**: 본 논문은 의료 영상 분석 분야에 적합한 기초 모델(Foundation Models, FMs)의 적응 전략을 종합적으로 평가합니다. 특히 기존의 태스크에 특화된 모델의 한계를 극복하기 위한 새로운 접근법들을 제시하여, FMs가 의료 영상의 요구에 맞게 어떻게 조정될 수 있는지를 탐구합니다. 또한, 새로운 개발 방향으로 지속적인 학습(continual learning)과 연합 학습(federated learning)에 대해 중점을 두고 설명합니다.

- **Technical Details**: 기술적 세부사항으로는 FMs의 다양한 적응 전략이 소개됩니다. 감독적 미세 조정(supervised fine-tuning), 도메인 맞춤형 사전 훈련(domain-specific pretraining), 및 자가 감독 학습(self-supervised learning) 같은 접근 방식이 포함됩니다. 특히, 자가 감독 학습은 레이블이 없는 데이터로부터 유용한 표현을 학습하는 데 사용되며, 이는 의료 영상의 다양한 과제에 효과적으로 적용될 수 있음이 강조됩니다.

- **Performance Highlights**: FMs의 적응에 대한 성능 하이라이트에서는 각 전략의 임상 적용 가능성과 성능 개선 사항이 평가됩니다. 예를 들어, CNNs와 U-Nets와 같은 기존 모델보다 FT(Foundation Models) 기반의 접근 방식이 뛰어난 일반화를 보여주며, 기존의 한계를 극복하는 데 기여할 수 있음을 시사합니다. 이러한 전략을 통해 의료 영상 분석에서 FMs의 신뢰성과 효율성을 개선할 수 있는 잠재력이 강조됩니다.



### When, What, and How: Rethinking Retrieval-Enhanced Speculative Decoding (https://arxiv.org/abs/2511.01282)
- **What's New**: ReSpec은 전통적인 heuristic 기반의 drafter 전환을 적응형 의사결정으로 바꾸는 새로운 프레임워크로, Speculative Decoding (SD)의 효율성을 크게 향상시킵니다. 주요 혁신으로는 낮은 불확실성에서만 retrieval을 트리거하는 엔트로피 기반의 적응형 트리거와 역사적 피드백을 활용한 후보 선택 방식이 포함됩니다. 또한, 출처 인식 relaxed verification 전략을 통해 정확성과 효율성 사이의 균형을 이룹니다. 이러한 개발을 통해 ReSpec은 기존 방법보다 33% 이상 빠른 성능 및 품질을 유지합니다.

- **Technical Details**: ReSpec은 정보 엔트로피를 사용하여 예측 가능성을 정량화하며, 낮은 엔트로피를 가진 상황에서만 retrieval을 트리거합니다. 토큰의 지식을 바탕으로 작성된 정보는 후보 선택에서의 과도한 검증 비용을 줄이고, 유망한 후보들을 조직화합니다. 최적의 검증 효율성을 위해, model-based drafts는 엄격한 검증을 받는 반면, retrieval-based drafts는 보다 느슨한 검증을 통해 품질을 보장합니다. 이 방식은 retrieval의 유용성을 최대화하는 동시에 시스템 자원의 효율적인 사용을 촉진합니다.

- **Performance Highlights**: ReSpec은 기초 실험인 Spec-Bench에서 EAGLE-2와 SAM-Decoding을 각각 33% 및 25% 이상 초과하는 성능으로 검증되었습니다. 특이하게도, 출력 품질을 저하시키지 않고도 효율적인 속도 향상을 이루었습니다. 또한, 적응형 및 상황 인식 제어가 혼합 추론 시스템에서의 효과를 극대화한 점에서 주목받고 있습니다. 이로써, ReSpec은 State-of-the-art의 가속화 결과를 달성하였으며, 이를 통해 향후 Speculative Decoding 분야의 발전에 기여할 것으로 기대됩니다.



### Adversarial Spatio-Temporal Attention Networks for Epileptic Seizure Forecasting (https://arxiv.org/abs/2511.01275)
- **What's New**: 이번 연구에서는 STAN(Adversarial Spatio-Temporal Attention Network)이라는 새로운 네트워크 모델을 제안하여 뇌의 공간 연결성과 신경 동역학을 동시에 모델링합니다. 기존 접근법들이 고정된 전구간(preictal) 지속시간을 가정하거나 공간적 및 시간적 특성을 개별적으로 처리한 반면, STAN은 공간과 시간 패턴 간의 상호 의존성을 포착합니다. 이 방법은 특정 개인에 대한 훈련 없이도 민감하고 신뢰할 수 있는 경고를 제공하여 조기 경고 시스템으로서의 가능성을 보여줍니다.

- **Technical Details**: STAN 모델은 연속된 주의(attention) 블록을 통해 공간 및 시간적 모듈을 교대로 결합하여 복잡한 동적 전환을 포괄적으로 특성화합니다. 이 구조는 주의 메커니즘을 통해 공간적 패턴과 그 변화 양상을 학습하며, 적대적 훈련(adversarial training) 기법을 통해 강력한 구별 표현을 학습합니다. 모델은 고유의 15분 전구간 윈도우를 기준으로 훈련되어, 다양한 개인에 대해 빠르게 변하는 패턴을 포착할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험은 두 개의 EEG 데이터 세트에서 진행되었고, STAN은 최근의 방법들과 비교하여 96.6%의 민감성과 시간당 0.011의 잘못된 감지율을 기록했습니다. 이는 전통적인 방법에 비해 40배 향상된 성능을 보여주었고, 모델의 경량화로 인해 실시간 배포가 용이합니다. 이 모델은 간질 예측 외에도 다양한 시간 시리즈 도메인에서 광범위하게 적용 가능성을 제시합니다.



### Rescuing the Unpoisoned: Efficient Defense against Knowledge Corruption Attacks on RAG Systems (https://arxiv.org/abs/2511.01268)
Comments:
          15 pages, 7 figures, 10 tables. To appear in the Proceedings of the 2025 Annual Computer Security Applications Conference (ACSAC)

- **What's New**: 최근의 연구는 대형 언어 모델(LLMs)의 한계인 훈련 비용 및 정보 공백 문제를 해결하기 위한 방법으로 Retrieval-Augmented Generation (RAG)이 주목받고 있음을 보여줍니다. RAG 시스템은 외부 지식 소스로부터 정보를 검색하여 응답을 생성하는 방식으로 작동합니다. 그러나 최근에는 RAG 시스템의 취약점, 특히 지식 오염 공격에 대한 우려가 증대하고 있습니다.

- **Technical Details**: 이 연구에서 제안된 RAGDefender는 RAG 시스템에 대한 지식 오염 공격을 방어하기 위한 효율적인 메커니즘입니다. RAGDefender는 검색한 패시지를 그룹화하고 적대적 패시지를 식별하는 두 가지 주요 단계로 구성되어 있습니다. 이 시스템은 추가적인 모델 훈련이나 추론을 요구하지 않으며 가벼운 머신 러닝 기술을 활용하여 적대적 내용을 탐지하고 필터링합니다.

- **Performance Highlights**: 실험 결과, RAGDefender는 여러 모델과 적대적 시나리오에서 기존의 방어 방법들보다 일관되게 우수한 성과를 보였습니다. 예를 들어, Gemini 모델에 대한 공격 성공률(ASR)을 0.89에서 0.02로 낮춘 반면, RobustRAG와 Discern-and-Answer는 각각 0.69와 0.24의 성과를 기록했습니다. RAGDefender는 RobustRAG보다 약 12.36배, Discern-and-Answer보다 1.53배 더 빠른 속도를 자랑합니다.



### Speech-DRAME: A Framework for Human-Aligned Benchmarks in Speech Role-Play (https://arxiv.org/abs/2511.01261)
Comments:
          67 pages

- **What's New**: 이 논문은 Speech-DRAME(Speech Detailed Role-play Assessment with Modeling and Evaluation)라는 통합 프레임워크를 제시합니다. 이 시스템은 기존의 오디오 대형 언어 모델(ALLMs)에 의존하지 않고, 인적 주석을 통해 좀 더 정밀한 평가를 가능하게 합니다. 주요 기여로는 이중 평가 패러다임, 인간 기반의 주석 및 평가 모델 정렬을 통해 구체적인 역할 연기에 대한 측정을 제공합니다.

- **Technical Details**: Speech-DRAME은 세 가지 주요 구성 요소로 이루어져 있습니다: (i) DRAME-EvalBench, 이중 언어로 주석 처리된 데이터셋을 포함한 평가 기준, (ii) DRAME-Eval, 제로샷 및 몇 샷 ALLMs를 초과하는 성능을 가진 미세 조정된 평가 모델, (iii) DRAME-RoleBench, 자동 평가자로 DRAME-Eval을 활용하는 역할 연기 벤치마크입니다. 이 시스템은 아키타입 평가와 현실주의 평가라는 두 가지 상보적인 전략을 통해 발음과 감정 및 전달 방법을 동시에 고려합니다.

- **Performance Highlights**: DRAME-Eval은 인간 평가와의 상관관계에서 강한 일치를 보이며, 아키타입 평가에서 0.480에서 0.629로, 현실주의 평가에서 0.390에서 0.625로 개선되었습니다. 이러한 성능 향상은 인간 주석의 중요성을 강조하고, Speech-DRAME이 긴밀하게 연결된 다양한 평가 차원을 통해 음성 역할 연기를 측정하고 개선하기 위한 신뢰할 수 있는 기초를 제공합니다.



### Quantum Deep Learning Still Needs a Quantum Leap (https://arxiv.org/abs/2511.01253)
- **What's New**: 이 논문에서는 양자 컴퓨터가 딥 러닝에 미칠 수 있는 잠재적 영향을 조사한 최초의 설문조사 결과를 제시합니다. 연구 결과에 따르면, 양자 알고리즘을 활용하여 딥 러닝을 가속화할 수 있는 세 가지 주요 영역이 있지만, 각 영역마다 실현 가능성을 제한하는 어려움이 존재한다고 합니다. 특히, 양자 알고리즘은 실제 문제 크기에서는 이론적 개선이 실질적으로 무시될 만큼 느린 속도로 작업을 수행합니다.

- **Technical Details**: 양자 알고리즘은 딥 러닝에 필수적인 행렬 곱셈과 같은 작업에서 이론적으로 소규모 개선을 제공하지만, 이는 실용적인 문제 크기에서는 이점을 주지 못합니다. 또한, 유망한 양자 알고리즘은 아직 개발되지 않은 Quantum Random Access Memory (QRAM)에 의존하고 있으며, 몇몇 알고리즘은 특정한 경우에만 적용 가능하여 실용적인 이점이 제한됩니다. 이러한 한계는 Choi et al.의 예측 모델을 바탕으로 정량적 예측을 통해 뒷받침됩니다.

- **Performance Highlights**: 양자 하드웨어의 발전에 대한 경향은 일부 문제에 대한 해결책을 제시할 수 있지만, 많은 과제들은 여전히 새로운 돌파구 없이는 해결되지 않을 것입니다. 이 연구는 양자 딥 러닝의 현재 범위를 설명하고 이 분야의 실질적인 발전으로 이어질 수 있는 연구 방향을 제시합니다. 또한 양자 컴퓨팅과 머신 러닝 분야의 연구자들이 실질적인 이점을 가져올 가능성이 있는 연구에 주목하는 데 도움이 될 것으로 기대합니다.



### Eyes on Target: Gaze-Aware Object Detection in Egocentric Video (https://arxiv.org/abs/2511.01237)
Comments:
          Accepted at RAAI 2025

- **What's New**: 이 논문에서는 'Eyes on Target'이라는 새로운 깊이 인식 및 시선 유도(object detection) 객체 탐지 프레임워크를 제안합니다. 이 프레임워크는 사람이 주목하는 지역에 스페이셜 피처 선택을 편향시키기 위해 Vision Transformer (ViT)의 주의 메커니즘에 시선 정보를 주입합니다. 기존의 객체 탐지 모델과 차별화되며, 시뮬레이션 환경에서 인지 능력을 평가하는 데 있어 중요한 역할을 합니다.

- **Technical Details**: 시선 추적 기술은 피실험자의 시선을 추적하여 주목하는 물체를 인식하는 데 사용됩니다. 이 연구에서는 시선 위치, 깊이, 동공 직경과 같은 여러 농도를 포함하여 시각 정보를 모델에 통합하여 Vision Transformer (ViT)가 인간의 시각적 관심과 일치하도록 합니다. 또한, 'gaze-aware head importance'라는 새로운 지표를 제안하여 시선 데이터가 주의 헤드에 미치는 영향을 정량화합니다.

- **Performance Highlights**: 제안된 모델은 다양한 실험 및 절제 연구를 통해 gaze-agnostic 기준 모델에 비해 일관된 검출 정확도를 향상시킨 것으로 나타났습니다. 특히, 커스텀 시뮬레이터 데이터 세트 및 공개 벤치마크(Ego4D Ego-Motion 및 Ego-CH-Gaze 데이터 세트)에서 개선된 성능을 보였습니다. 이 연구는 시각적 집중력 평가 및 인간 성능 분석의 가능성을 보여주며, 모델의 해석력을 높이는 데 기여합니다.



### Influence-aware Causal Autoencoder Network for Node Importance Ranking in Complex Networks (https://arxiv.org/abs/2511.01228)
- **What's New**: 노드 중요도 순위는 그래프 데이터 분석에서 필수적인 문제로, 기존 접근 방식은 노드 특성이 전통적인 중심성 지표나 고급 그래프 표현 학습 방법에서 파생되었다는 점이 있습니다. 그러나 이는 네트워크의 구조에 의존하여 프라이버시 문제를 야기하고 다양한 네트워크에서 일반화가 어려운 경우가 많습니다. 본 연구는 인과적 표현 학습(causal representation learning)을 활용하여 합성 네트워크에서 학습된 노드 임베딩을 실제 네트워크에 효과적으로 적용할 수 있는 새로운 모델인 Influence-aware Causal Autoencoder Network (ICAN)을 제안합니다.

- **Technical Details**: ICAN은 노드 중요도와 인과적 관련이 있는 노드 임베딩을 추출하기 위해 오토인코더(autocoder) 아키텍처 내에 영향 인식(influence-aware) 인과 표현 학습 모듈을 도입합니다. 이 모델은 또한 재구성(reconstruction) 및 순위(rank) 목표를 공동으로 최적화하는 통합 최적화 프레임워크를 설계하여 노드 표현 학습과 순위 최적화 사이의 상호 강화(mutual reinforcement)를 지원합니다. 이러한 설계는 ICAN이 합성 네트워크에서 효과적으로 학습되어 다양한 실제 그래프에 적용될 수 있도록 합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터 세트에 대한 광범위한 실험을 통해 ICAN은 순위 정확도(rank accuracy)와 일반화 능력(generalization capability) 모두에서 기존의 최첨단 기법들(state-of-the-art) 보다 일관되게 뛰어난 성능을 보여주었습니다. 이 결과는 ICAN이 노드 중요도 순위 예측 정확성을 향상시킬 수 있음을 입증하며, 설계된 영향 인식 인과 메커니즘의 기여를 강조합니다.



### Thought-For-Food: Reasoning Chain Induced Food Visual Question Answering (https://arxiv.org/abs/2511.01213)
Comments:
          10 pages, 11 figures, 6 tables

- **What's New**: 이 논문에서는 인도 요리를 위한 Visual Question Answering(VQA) 시스템의 한계를 해결하기 위해, 다단계 추론 과정을 요구하는 새로운 접근 방식이 소개되었습니다. 기존의 VQA 시스템이 서구 요리에 초점을 맞춰 기술적으로 유한한 반면, 인도 요리는 복잡성을 포함하고 있어 해당 분야에 대한 논의가 필요하다는 점을 강조합니다. 이를 통해, 간단한 인식 이상의 복잡한 질문과 맥락을 해결할 수 있는 구조를 제시하고 있습니다.

- **Technical Details**: 이 연구에서는 질문-답변 체인(Reasoning Chains)을 통해 모델의 성능을 향상시키기 위한 방법론을 제안합니다. 자동으로 생성된 추론 체인을 사용해 소규모 LLMs(대형 언어 모델)와 VLMs(비주얼 언어 모델)를 파인튜닝하고, 강화학습(Reinforcement Learning)으로 추가 학습을 진행해 보다 정확한 응답을 이끌어내고자 합니다. 이를 통해 인도 요리에 대한 VQA의 정확도를 평균 10% 향상시켰습니다.

- **Performance Highlights**: 제안된 방법은 IndiFoodVQA 데이터셋의 기준 성능을 일관되게 초과하여, 문화적으로 다양한 맥락에서 음식 VQA의 중요성을 입증합니다. 최종 모델은 최고 정확도 71.12%를 달성하며 최선의 성능을 보여줍니다. 이와 같은 이유 기반 접근법에 대한 연구 및 분석은 인도 요리를 이해하고 학습하는 데 있어 새로운 가능성을 제시합니다.



### Forget BIT, It is All about TOKEN: Towards Semantic Information Theory for LLMs (https://arxiv.org/abs/2511.01202)
- **What's New**: 이번 논문은 대형 언어 모델(LLM)의 이론적 이해를 정보 이론(Information Theory) 관점에서 탐구합니다. 저자들은 비트 대신 의미 있는 단위인 토큰(Token)을 사용하여 LLM의 구조 및 성능을 설명하는 새로운 개념인 의미 정보 이론(Semantic Information Theory)을 제안합니다. 이 이론은 LLM의 작동 방식을 규명하고, LLM 훈련의 주요 메커니즘을 정보 이론적 원칙을 통해 설명하려고 합니다.

- **Technical Details**: 이 논문은 LLM을 표적 확률적 모델(Position-Based Probabilistic Model)로 정의하고, 여기서 정보 왜곡 함수(Rate-Distortion Function)와 같은 구조에 구애받지 않는 정보 이론적 측정을 다룹니다. 특히, 훈련 전 방향 정보 왜곡 함수(Directed Rate-Distortion Function)와 훈련 후 방향 보상 함수(Directed Rate-Reward Function)에 대한 논의가 포함됩니다. 또한, 저자는 토큰 수준의 의미 임베딩(Token-level Semantic Embedding)과 정보 이론적으로 최적화된 벡터화 방법에 대해 설명합니다.

- **Performance Highlights**: 논문에서는 Transformer 아키텍처를 기반으로 한 자가회귀 LLM(AR-LLM)의 일반 정의를 제안하며, 이 아키텍처에 대한 이론적 분석을 진행합니다. 저자들은 ELBO, 일반화 오류 경계, 메모리 용량 및 의미 정보 측정과 같은 성능 지표를 이론적으로 도출합니다. 더불어, Mamba/Mamba2 및 LLaDA와 같은 다른 아키텍처도 다루어 논문의 포괄성을 더합니다.



### An Interdisciplinary and Cross-Task Review on Missing Data Imputation (https://arxiv.org/abs/2511.01196)
- **What's New**: 이 논문은 데이터 과학에서 핵심적인 문제인 누락 데이터(missing data)의 처리 방법에 대한 종합적인 리뷰를 제공합니다. 다양한 분야에서의 누락 데이터를 처리하는 다양한 방법론을 탐구하며, 통계적 기초(statistical foundations)와 현대 기계 학습(ML) 발전을 연결할 필요성이 강조됩니다. 전통적인 기법에서 최신 딥러닝 모델까지 포괄적으로 다루며, 복잡한 데이터 유형에 대해서도 주목하고 있습니다.

- **Technical Details**: 누락 데이터 처리의 주요 접근 방식으로는 삭제(deletion), 무시(ignorance), 단일 및 다중 보간법(single versus multiple imputation) 등이 있습니다. 딥러닝 모델인 오토인코더(autoencoder) 및 생성적 적대 신경망(GAN)과 같은 현대 기법이 포함되어 있으며, 최근에는 대형 언어 모델(LLM)도 누락 데이터 처리에 활용되고 있습니다. 이 연구는 다양한 도메인에 걸친 문제 특성과 이에 대한 다양한 보간 기법의 카테고리를 제시합니다.

- **Performance Highlights**: 실제 데이터는 종종 행렬(matrix) 형태로 표현되며, 이 연구는 순수한 보간 기법과 딥러닝 기술을 통한 성능 개선의 상관관계를 논의합니다. 데이터 유형에 따라 보간법의 성능은 상이하며, 특정 분야에서는 전통적 기법보다 딥러닝 기반 방법이 우수한 결과를 나타냄을 강조합니다. 또한, 다운스트림 작업(downstream tasks)과의 통합 필요성도 논의되며, 이는 분류(classification), 클러스터링(clustering), 이상 탐지(anomaly detection)와 함께 평가됩니다.



### A Topology-Aware Graph Convolutional Network for Human Pose Similarity and Action Quality Assessmen (https://arxiv.org/abs/2511.01194)
Comments:
          10 pages, 5 figures. Submitted as a computer vision paper in the cs.CV category

- **What's New**: 이 논문은 Action Quality Assessment (AQA)에 대한 새로운 접근 방식을 제안합니다. 인체 스켈레톤을 그래프로 모델링하여 Graph Convolutional Network (GCN)을 사용해 포즈 유사성을 측정하는 방법을 제시합니다. 이 방법은 기존의 좌표 기반 기법을 뛰어넘어 더 정교하고 사실적인 포즈 임베딩을 학습합니다. 이를 통해 AQA-7 및 FineDiving 벤치마크에서 경쟁력 있는 성능을 보여줍니다.

- **Technical Details**: GCN-PSN은 사람의 포즈를 정확하게 추정하기 위해 YOLOv5 모델을 사용하여 인간을 감지한 후 HRNet을 통해 2D 키포인트를 로컬라이즈합니다. 15개의 키포인트로 구성된 스켈레톤 그래프를 정의하고, 각 포인트 간의 연결을 기반으로 인접 행렬을 구성하여 특징 추출을 수행합니다. 그런 다음, GCN을 통해 이 그래프에서 포즈 간의 유사성을 측정하는 특성 임베딩을 학습합니다. 대칭 네트워크 아키텍처를 이용하여 두 이미지를 동시에 처리하여 최종 유사도를 계산합니다.

- **Performance Highlights**: 실험 결과, 제안된 GCN-PSN 모델은 기존의 AQA 시스템보다 유의미한 성능 향상을 보여주었습니다. 특히, 특정 포즈 간의 유사성을 평가하는 데 있어 스켈레톤 토폴로지를 효과적으로 활용하여 더욱 신뢰할 수 있는 점수를 제공하는 것으로 나타났습니다. AQA와 관련된 여러 벤치마크 평가에서 긍정적인 결과를 도출하며, 향후 연구의 기초 자료로 활용될 수 있을 것으로 예상됩니다.



### Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning (https://arxiv.org/abs/2511.01191)
- **What's New**: 본 논문에서 제안된 Self-Harmony는 테스트 시간 강화 학습(Test-time Reinforcement Learning, TTRL)의 새로운 접근 방식을 소개합니다. 이 프레임워크는 동일한 문제를 다양한 방식으로 바꿔 제시했을 때 올바른 답변이 안정적으로 유지되어야 한다는 직관에 기반하고 있습니다. Self-Harmony는 단일 모델이 문제를 해결하는 Solver 역할과 문제를 재표현하는 Reframer 역할을 동시에 수행하도록 하여 신뢰할 수 있는 학습 신호를 생성합니다.

- **Technical Details**: Self-Harmony는 전통적인 다수결 방식의 한계를 극복하고자 하며, 원본 문제와 재표현된 문제에서의 답변 빈도를 집계하고 조화 평균(harmonic mean)을 통해 최종 의사 결정을 내립니다. 이를 통해 무의미한 주장이나 잘못된 reasoning을 피하고, 안정적인 해답을 추출할 수 있습니다. 모델은 이 두 가지 역할을 통해 협력하여 자기 놀이(self-play)를 수행하며, 이를 통해 더 나은 학습이 가능해집니다.

- **Performance Highlights**: Self-Harmony는 다양한 reasoning 벤치마크에서 탁월한 성과를 달성했습니다. 30개의 테스트 설정 중 28건에서 1위를 기록하며, label-free 테스트 환경에서 최첨단 결과를 입증하였습니다. 성능 외에도 모든 실험에서 제로 훈련 실패율을 기록하여, 이 방법의 안정성과 신뢰성을 강조합니다.



### ZoFia: Zero-Shot Fake News Detection with Entity-Guided Retrieval and Multi-LLM Interaction (https://arxiv.org/abs/2511.01188)
- **What's New**: 이 논문에서는 ZoFia라는 새로운 두 단계의 제로샷(Zero-Shot) 허위 뉴스 탐지 프레임워크를 제안합니다. 첫 번째 단계에서는 Hierarchical Salience를 도입하여 뉴스 콘텐츠에서 실체의 중요성을 정량화하고, SC-MMR 알고리즘을 사용해 최신 외부 증거를 탐색하기 위한 정보가 풍부하고 다양한 키워드를 효과적으로 선택합니다. 두 번째 단계에서는 다중 LLM 상호작용 시스템이 여러 관점에서 뉴스 텍스트 및 관련 정보를 협력적으로 분석하고, 이를 통해 해석 가능하고 견고한 판단을 생성합니다.

- **Technical Details**: ZoFia는 두 단계로 구성된 구조를 가지며, 첫 단계에서 엔티티 가이드 검색(Entity Guided Retrieval)을 통해 LLM의 내부 지식을 실시간 외부 정보와 통합하여 다원 정보 행렬(Multi-Source Information Matrix)을 구축합니다. 이를 통해 실시간으로 변하는 뉴스 스트림에 대한 지식 한계를 보완하고, 다양한 LLM이 협력하여 다차원 분석을 수행합니다. 이러한 방식으로, 각 LLM이 서로 다른 역할을 맡아 뉴스에 대해 논쟁을 진행함으로써 더 나은 결과를 도출합니다.

- **Performance Highlights**: 두 개의 공개 데이터 세트에서 수행된 실험 결과, ZoFia는 기존의 제로샷 모델과 대부분의 소수 샷 방법들을 명백히 능가합니다. 이러한 성과는 LLM의 제약을 극복하고, 다양한 정보 출처를 효과적으로 활용하는 다중 에이전트 시스템 덕분입니다. 연구팀은 코드도 오픈 소스로 공개하여 관련 커뮤니티의 재현성을 보장하고 있습니다.



### Adapt under Attack and Domain Shift: Unified Adversarial Meta-Learning and Domain Adaptation for Robust Automatic Modulation Classification (https://arxiv.org/abs/2511.01172)
- **What's New**: 이 논문에서는 기존의 Automatic Modulation Classification (AMC) 시스템의 문제점을 해결하기 위해 메타 학습과 도메인 적응을 통합한 새로운 통합 프레임워크를 제안합니다. 이 프레임워크는 적대적 공격(adversarial attacks) 및 환경 변화에 강한 AMC 시스템의 개발을 목표로 합니다. 2단계 전략을 사용하여 첫 번째로, 오프라인에서 메타 학습을 통해 모델을 훈련시키고, 두 번째로 온라인에서 도메인 적응을 통해 모델의 특징을 새로운 타겟 도메인과 일치시켜 성능을 향상시킵니다.

- **Technical Details**: 제안한 프레임워크에서는 우선 오프라인 단계에서 깨끗한 샘플과 적대적 변형이 가해진 샘플을 사용하여 모델을 훈련시킵니다. 이 과정에서 모델은 이전에 보지 못한 다양한 공격에 강한 방어력을 익힙니다. 이어서 온라인 단계에서는 타겟 도메인에 대한 적은 수의 라벨이 붙은 파일럿 신호를 사용하여 도메인 정렬을 진행하며, 이는 라벨 데이터가 부족한 상황에서도 성능을 유지할 수 있게 합니다.

- **Performance Highlights**: 이 연구에서 제안하는 통합 프레임워크는 협소한 샘플과 훈련 시간이 제한된 상황에서도 성능을 극대화할 수 있습니다. 실험을 통해, 이 프레임워크는 기존 AMC 시스템에 비해 모듈레이션 분류 정확도를 크게 향상시켜, 실제 무선 환경에서의 배치 및 운영 문제 해결에 중요한 기여를 하고 있음을 보여줍니다. 또한, 본 연구는 적대적 공격과 도메인 이동을 동시에 처리하려는 AMC 프레임워크를 처음으로 제시한 사례로, 크고 작은 변화에 적응할 수 있는 능력을 강조합니다.



### A High-Throughput Spiking Neural Network Processor Enabling Synaptic Delay Emulation (https://arxiv.org/abs/2511.01158)
- **What's New**: 이 논문은 복잡한 시공간 정보를 통합하고 처리하기 위한 시냅스 지연(synaptic delay)에 기반한 고처리량 Spiking Neural Network (SNN) 프로세서를 소개합니다. 이 프로세서는 엣지(edge) 애플리케이션을 위한 시냅스 지연 기반 에뮬레이션을 지원하며, 다중 코어 파이프라인 아키텍처를 활용하여 실시간 처리 성능을 달성합니다. PYNQ Z2 FPGA 플랫폼에 제안된 프로세서의 SoC 프로토타입을 구현하고, 저전력 키워드 탐지 작업을 위한 Spiking Heidelberg Digits (SHD) 벤치마크를 사용하여 성능을 평가하였습니다.

- **Technical Details**: Leaky Integrate-and-Fire (LIF) 모델을 사용하여 후신경(membrane potential)을 업데이트하고, 이를 통해 시냅스 지연을 비지연 계산으로 변환합니다. 프로세서는 4개의 동질의 스파이킹 계산 코어와 구성 유닛, 외부 데이터/구성 인터페이스를 포함하는 마이크로아키텍처를 갖추고 있습니다. 스파이킹 링 버퍼(Spiking Ring Buffer, SRB)는 시냅스 지연을 시뮬레이션하며, 지연 유닛, 고정 소수점 곱셈기, 덧셈기, 비교기 및 연결된 뉴런을 동시에 업데이트하는 방식을 적용합니다.

- **Performance Highlights**: SHD 벤치마크를 통해 키워드 탐지 작업에서 제안된 프로세서는 93.4%의 정확도를 달성하고, 평균 처리량 104 샘플/초를 기록하였습니다. 프로세서는 125 MHz에서 작동하며, 평균 전력 소비는 282 mW로 저전력을 유지합니다. 이 SoC 플랫폼은 동적 전력 소비 1.28 W를 소모하는 ARM Cortex-A9 CPU와 함께 40,521 LUTs 및 44,161 FFs를 사용하며, 높은 처리량과 짧은 지연 시간을 비용 효율적으로 달성할 수 있는 방법을 보여줍니다.



### AthenaBench: A Dynamic Benchmark for Evaluating LLMs in Cyber Threat Intelligenc (https://arxiv.org/abs/2511.01144)
- **What's New**: 이번 논문에서는 기존의 CTIBench를 확장하여 AthenaBench라는 새로운 평가 벤치마크를 개발했습니다. AthenaBench는 리스크 완화 전략에 중점을 둔 새로운 과제를 포함하고, 데이터셋 생성 파이프라인을 개선하며, 중복 제거 및 정제된 평가 측정을 도입하여 CTI 분석의 효과성을 향상시키기 위한 노력을 보여줍니다. 이를 통해 LLM이 CTI 업무에서 어떤 식으로 더 효율적으로 활용될 수 있는지를 강조하고 있습니다.

- **Technical Details**: AthenaBench는 MITRE ATT&CK와 NVD API와 같은 실시간 CTI 데이터 소스를 활용하여 지속적으로 벤치마크 샘플을 생성합니다. 평가 과제는 사실 회상, 취약성 심각도 예측, 위협 원인 매핑 및 방어 제안 등 다양한 CTI 추론 차원을 다룹니다. 또한, 각 과제에 대한 개선된 평가 메트릭을 도입하여 모델의 성능을 종합적으로 비교할 수 있도록 설계되었습니다.

- **Performance Highlights**: 총 12개의 LLM을 평가한 결과, 상용 모델인 GPT-5가 가장 높은 성적을 기록했지만, 추론 중심 작업에서는 여전히 준수한 성능을 보이지 않았습니다. 개방형 소스 모델들은 성능에서 뒤처져 있으며, 이들은 CTI 워크플로우의 완전한 자동화를 달성하는 데 지속적인 과제가 남아 있음을 시사합니다. Youngs





### MicroAUNet: Boundary-Enhanced Multi-scale Fusion with Knowledge Distillation for Colonoscopy Polyp Image Segmentation (https://arxiv.org/abs/2511.01143)
Comments:
          Work in progress

- **What's New**: 본 논문에서 제안하는 MicroAUNet은 경량화된 주의 기반(segmentation networks) 분할 네트워크로, 깊이 우선 분리된(dilated convolutions) 합성과 파라미터 공유(channel-spatial attention) 블록을 결합하여 다중 스케일 경계 특징을 강화합니다. 이 네트워크는 높은 정확성과 낮은 모델 복잡성을 유지하면서 실시간(colonoscopy) 임상용 폴립(segmentation) 분할에 적합하도록 설계되었습니다.

- **Technical Details**: MicroAUNet은 경량화 아키텍처와 점진적인 지식 증류(progressive knowledge-distillation) 기법을 통해 복잡한 배경 속에서도 폴립 경계를 정밀하게 세분화하는 데 초점을 맞추고 있습니다. 깊이 우선 분리된 합성(depthwise separable convolutions)과 공통 채널-공간 주의 메커니즘을 사용하여 다중 스케일 문맥 정보를 효율적으로 추출합니다. 이에 따라 폴립 탐지(model)와 진단(analysis) 과정에서의 초기 세분화 정확도를 높입니다.

- **Performance Highlights**: 공공 데이터셋에 대한 광범위한 검증 결과, MicroAUNet은 기존의 최신 방법들과 비교해세분화 정확도(segmentation accuracy)와 추론 효율성(inference efficiency)에서 우수함을 입증했습니다. 이는 경량화된 모델을 통한 효과적인 학습(runtime) 속도를 가능하게 하여 임상 적용의 잠재력을 강조합니다.



### Few-Shot Multimodal Medical Imaging: A Theoretical Framework (https://arxiv.org/abs/2511.01140)
Comments:
          6 Pages

- **What's New**: 이 논문은 의료 이미징에 있어 데이터 접근성이 떨어지는 환경에서의 학습과 유추를 위한 통합 이론적 프레임워크를 제안합니다. 제안된 프레임워크는 샘플 효율성(sample efficiency), 불확실성 정량화(uncertainty quantification) 및 해석 가능성(interpretability) 간의 관계를 명확히 설정합니다. 이는 저자들이 훈련 목표를 조금의 데이터로 형식화하고, 안정성을 위한 정량적 지표인 설명 분산(explanation variance)을 도입함으로써 이루어졌습니다.

- **Technical Details**: 논문에서는 적은 양의 레이블이 있는 데이터 세트에서 모델을 훈련시키기 위해, PAC 학습 이론(Probably Approximately Correct learning theory)과 VC 이론(Vapnik–Chervonenkis theory)을 기반으로 합니다. 이들은 데이터 희소성이 있는 상태에서 모델의 기대 위험을 최적 값에 가깝게 맞출 수 있는 가장 작은 레이블 수를 찾는 데 중점을 둡니다. 또한, 방법론은 여러 개의 데이터 소스를 융합하여 풍부한 정보를 제공하는 멀티모달 통합(multi-modal integration)을 통한 일반화를 촉진하는 원리를 설명합니다.

- **Performance Highlights**: 제안된 이론적 프레임워크는 의료 이미징 시스템의 데이터 효율성 종료(model capacity)와 불확실성의 상호 관계를 명확히 정의함으로써, 결핍 상태에서의 진단 시스템을 안정적으로 구축하기 위한 기초를 제공합니다. 특히, 여러 가지 불확실성 정량화 기법들을 활용하여 예측 오류를 최소화하고, 의사 결정을 개선할 수 있는 방법을 모색합니다. 전체적으로 이 연구는 저자들이 구축한 모델이 임상적으로 신뢰할 수 있는 성과를 내기 위한 이론적 근거를 마련합니다.



### Learning with Category-Equivariant Architectures for Human Activity Recognition (https://arxiv.org/abs/2511.01139)
- **What's New**: 본 논문에서는 CatEquiv를 제안했습니다. 이는 관성 센서를 활용한 인간 활동 인식(HAR)을 위한 카테고리 등변(neural network) 신경망입니다. CatEquiv는 시간, 진폭, 구조적 대칭(symmetries)을 체계적으로 인코딩하여 데이터의 범주적 대칭 구조를 캡처합니다.

- **Technical Details**: CatEquiv는 순환 시간 이동(cyclic time shifts), 긍정적 이득(positive gains) 및 센서 계층(poset)을 결합한 범주적 대칭 곱(categorical symmetry product)을 통해 등변성을 달성합니다. 이 신경망의 기본 아키텍처는 순환 1D 컨볼루션, RMS 정규화, 축 공유 필터를 포함하여 모형의 강건성을 보다 수월하게 확보할 수 있도록 합니다.

- **Performance Highlights**: UCI-HAR 데이터셋에서 CatEquiv는 기존의 순환 패딩 CNNs 및 일반 CNNs에 비해 명백히 높은 강건성을 나타냈습니다. 특히 CatEquiv는 복합적인 OOD(Out-Of-Distribution) 조건 하에서 기존 모델보다 높은 정확도와 매크로-F1 점수를 달성했습니다. 이는 모델의 용량을 증가시키지 않고도 범주적 대칭을 강제하여 얻어진 결과입니다.



### Continual Learning, Not Training: Online Adaptation For Agents (https://arxiv.org/abs/2511.01093)
Comments:
          12 pages, 4 figures

- **What's New**: 본 논문에서는 기존의 지속적 학습(Continual Learning, CL) 접근방식의 한계를 극복하기 위해 Adaptive Teaching and Learning System (ATLAS)을 소개합니다. ATLAS는 reasoning(교사)와 execution(학생)을 분리한 이중 에이전트 아키텍처로, 경험에서 추출한 distilled guidance(압축된 지침)를 저장하는 지속적 학습 메모리를 통합합니다. 이러한 방법을 통해 gradient-free continual learning(경량 지속적 학습)을 실현하며, 모델 파라미터의 조정 대신 시스템 차원의 조정으로 적응을 이끌어냅니다.

- **Technical Details**: ATLAS는 메모리 기반의 orchestration(조정) 계층을 활용하여 실시간 적응을 가능케 합니다. 이 시스템은 기존의 무게 업데이트 방식에 의존하지 않으며, 교사-학생 상호작용에서 유도된 경험을 활용하여 작동 전략을 동적으로 조정합니다. Microsoft의 ExCyTIn-Bench에서, ATLAS는 GPT-5-mini를 사용하여 54.1%의 성공률을 달성하며, 이는 GPT-5 (High)보다 13% 향상된 수치로, 접근 비용을 86% 절감하였습니다.

- **Performance Highlights**: ATLAS는 복잡한 사이버 위협 조사 환경에서 높은 성과를 보이며, Incident #5에서의 검증을 통해 아웃풋 구성의 전환을 이끌어내고 있으며, 성과는 28%에서 41%로 향상되었습니다. 이러한 결과는 ATLAS가 적응 가능한 AI 시스템으로 지속적인 발전이 가능하다는 것을 보여줍니다. 또한, 이는 구조화된 추론으로 나아가는 경향을 보이며, 세계 모델을 훈련하기 위한 귀중한 인과적으로 주석된 흔적을 제공합니다.



### SliceVision-F2I: A Synthetic Feature-to-Image Dataset for Visual Pattern Representation on Network Slices (https://arxiv.org/abs/2511.01087)
- **What's New**: 5G 및 6G 네트워크의 출현은 네트워크 슬라이싱(network slicing)을 미래 서비스 지향 아키텍처의 중요한 부분으로 확립했습니다. 이 논문에서는 차세대 네트워킹 시스템을 위한 기능 시각화(feature visualization)의 연구를 위해 설계된 신 synthetic 데이터 세트인 SliceVision-F2I를 소개합니다. 3만 개의 샘플로 구성된 이 데이터 세트는 다양한 변환 방법을 통해 다변량 Key Performance Indicator(KPI) 벡터를 시각적 표현으로 변환합니다.

- **Technical Details**: SliceVision-F2I는 물리적 영감 맵(physically inspired mappings), Perlin noise, 신경벽지(neural wallpapering), 프랙탈 가지치기(fractal branching) 등 네 가지 인코딩 방법을 통해 데이터를 생성합니다. 각 변환 방법마다 3만 개의 샘플이 생성되며, 이는 저해상도 RGB 이미지와 원시 KPI 벡터를 포함합니다. 이 데이터 세트는 실질적인 노이즈 모델을 통합하여 측정의 불완전성을 반영하고, 시각적 학습, 네트워크 상태 분류, 이상 탐지 및 이미지 기반 기계 학습 기술 벤치마킹에 적합합니다.

- **Performance Highlights**: SliceVision-F2I 데이터 세트는 저해상도에서 완벽한 분류 정확도를 유지할 수 있음을 입증하며, 이는 실시간 처리 가능성을 엿볼 수 있는 좋은 기회를 제공합니다. 현재 결과는 완벽한 분류 정확도를 달성했으며, 이러한 성과는 다른 연구에서도 확장 가능성을 지니고 있습니다. 또한, 이 데이터 세트는 실제 네트워크의 노이즈와 변동성을 처리할 수 있는 새로운 분류 방법을 시험하는 데 유용한 기반을 제공합니다.



### GeoToken: Hierarchical Geolocalization of Images via Next Token Prediction (https://arxiv.org/abs/2511.01082)
Comments:
          Accepted to IEEE International Conference on Data Mining (ICDM) 2025

- **What's New**: 이 논문에서는 이미지의 지리적 출처를 추정하는 새로운 접근 방식인 GeoToken을 소개합니다. 사람의 사고 과정을 모방하여 이미지를 세분화하고, 명확한 지리적 토큰을 계층적으로 예측하는 방법을 제안합니다. 이 방법은 기존 방식의 한계를 극복하며, 특허받지 않은 데이터 세트에서 월드와이드 이미지 지리적 위치를 보다 정확하게 예측하도록 고안되었습니다.

- **Technical Details**: GeoToken은 다단계 지리적 셀을 정의하고, 시각 정보와 이전 예측을 조건으로 하여 점진적으로 더 정밀한 지리적 셀을 예측합니다. 이 모델은 입력 이미지로부터 유사 이미지를 검색하여 생성 과정에 도움을 주며, S2 셀 구조를 활용하여 전역의 계층적 지리 정보를 체계적으로 탐색합니다. 또한, 오토리그레시브 방식으로 정확한 위치 추정을 위해 다양한 샘플링 기법을 통합하여 불확실성을 관리합니다.

- **Performance Highlights**: 이 논문에서 제안하는 모델은 Im2GPS3k와 YFCC4k 데이터 세트에서 다른 기준선에 비해 높은 성능을 보이며, MLLM이 없는 설정에서 정확도에서 최대 13.9% 향상된 결과를 얻었습니다. MLLM이 포함되었을 때는 모든 메트릭에서 새로운 최첨단 성능을 기록하며, 사용자 데이터 보안을 보장하면서도 현장에서 강력한 지리적 위치 추정이 가능하다는 장점을 가지고 있습니다.



### Energy-Efficient Deep Learning Without Backpropagation: A Rigorous Evaluation of Forward-Only Algorithms (https://arxiv.org/abs/2511.01061)
- **What's New**: 이 연구는 최신 인공지능 성능을 구현하는 데 필수적이라고 여겨졌던 backpropagation (BP)에 의문을 제기합니다. Mono-Forward (MF) 알고리즘은 BP에 비해 분류 정확도에서 consistently 우수한 성과를 내며, 이는 효율성 측면에서도 significant 이점을 보여줍니다. 특히, MF는 최대 41% 에너지 소비를 줄이고, 34% 더 빠른 훈련 속도를 기록했습니다.

- **Technical Details**: 연구에서는 여러 BP-free 알고리즘을 비교하기 위해 systematized하고 hardware를 검증하는 방법론을 사용했습니다. Forward-Forward (FF), Cascaded Forward (CaFo), 그리고 Mono-Forward (MF) 알고리즘을 사용하여, 각기 원격의 native Multi-Layer Perceptron (MLP) 아키텍처에서 성능을 평가했습니다. MF 알고리즘은 선형 프로젝션 매트릭스를 활용하여 정확도를 높이고, BP보다 메모리 효율성을 높였습니다.

- **Performance Highlights**: 이 실험 결과, MF 알고리즘은 MLP 아키텍처에서 BP를 능가하는 성과를 냈습니다. 구체적으로, MF는 훈련 시간이 BP보다 평균적으로 significantly 줄어들고, 에너지 소비 또한 크게 감소했습니다. FF 및 CaFo 알고리즘은 MF에 비해 효율성과 정확도 모두에서 낮은 성과를 보여주었습니다.



### HAFixAgent: History-Aware Automated Program Repair Agen (https://arxiv.org/abs/2511.01047)
Comments:
          31 pages, 6 figures

- **What's New**: 본 연구에서는 HAFixAgent라는 새로운 History-Aware Bug-Fixing Agent를 제안하여, 자동 프로그램 수리(automated program repair, APR) 시스템에 리포지토리 히스토리(repository history)를 통합하였습니다. 기존의 APR 시스템들은 대개 코드의 로컬 스냅샷만을 기반으로 하여, 코드 수정 시 역사적 맥락을 간과했습니다. HAFixAgent는 역사적 데이터에서 도출된 휴리스틱을 활용하여 복잡한 멀티-퍽(multi-hunk) 버그 수리에서 효과를 발휘합니다.

- **Technical Details**: HAFixAgent는 git blame를 통해 히스토리를 분석하고, 이를 수리 루프에 통합하여 버그 수리의 효율성과 효과성을 높입니다. 본 연구에서는 Defects4J 데이터 세트에서 854개의 실제 버그를 평가 대상으로 하여, 각 버그에 대해 역사적 정보의 존재 여부 및 집중도를 분석하였습니다. 연구 결과, 71.1%의 버그에서 유의미한 역사적 정보가 가능하다는 것을 확인했으며, 이러한 데이터가 수리 과정에서 어떻게 활용될 수 있는지에 대한 통찰을 제공합니다.

- **Performance Highlights**: HAFixAgent는 기존의 두 가지 최첨단 APR 시스템과 비교하여, 일반 버그의 수리를 212.3% 개선하고, 멀티-퍽 수리에서도 29.9% 향상되었습니다. 또한, 역사적 정보의 통합은 에이전트의 동작 단계 수를 크게 증가시키지 않았으며, 비용 효율적인 방식으로 복잡한 다중 파일 및 멀티-퍽 버그의 수리를 가능하게 하였습니다. 이 연구는 역사적 맥락을 통한 효과적인 에이전트 기반 APR의 실용적 방법론을 제공합니다.



### Seed-Induced Uniqueness in Transformer Models: Subspace Alignment Governs Subliminal Transfer (https://arxiv.org/abs/2511.01023)
Comments:
          Cite as A. S. Okatan, M. I. Akbaş, L. N. Kandel, and B. Peköz, "Seed-Induced Uniqueness in Transformer Models: Subspace Alignment Governs Subliminal Transfer," in Proc. 2025 Cyber Awareness and Research Symp. (IEEE CARS 2025), Grand Forks, ND, Oct. 2025, pp. 6

- **What's New**: 본 연구에서는 Transformer 모델에서의 잠재적 전송(subliminal transfer)에 대해 분석하며, 교사 모델이 숨겨진 특성을 내포하고 이를 학생 모델이 주요 작업 성능을 저하시키지 않고 선형적으로 디코딩할 수 있음을 보여준다. 기존 연구들은 전송가능성을 Global Representational Similarity로 설명하지만, 본 연구는 특정 하위공간의 정렬(alignment)에서 전송 강도가 결정된다는 사실을 밝혀냈다.

- **Technical Details**: 연구에서는 새로운 Synthetic Datasets를 생성하고, 동일한 아키텍처의 모델들이 무작위 초기화(random initialization)에 따라 달라지는 방식으로 실험을 진행하였다. 모델 간의 전송은 Global CKA(Centered Kernel Alignment) 수치가 높은 경우에도 크게 줄어드는 것으로 나타났고, 이는 Trait-Discriminative Subspace에서의 정렬이 주요 영향을 미친다는 것을 의미한다. 또한, Subspace-Level CKA를 기반으로 한 진단 프로토콜을 제안하며, 이를 통해 새로운 보안 제어 방법들이 효과를 보인다는 점을 입증하였다.

- **Performance Highlights**: 실험 결과는 동일 초기화(지금 seed) 모델이 상당한 정보 유출(leakage)을 보이는 반면, 다른 초기화 모델은 매우 낮은 유출률을 보임을 보여준다. 세 가지 보안 제어 방법(Projection Penalty, Adversarial Gradient Reversal, Right-for-the-Wrong-Reasons Regularization)이 모든 경우에 유출을 저지하면서 주요 작업의 성능을 저하시키지 않았다는 것을 밝혔다. 이러한 결과는 독립적으로 초기화된 Transformer 모델이 높은 Global Similarity에도 불구하고 잠재적 전송에 저항한다는 것을 보이며, AI 배포의 보안을 강화하는 데 기여할 수 있다.



### OceanAI: A Conversational Platform for Accurate, Transparent, Near-Real-Time Oceanographic Insights (https://arxiv.org/abs/2511.01019)
Comments:
          A related presentation will be given at the AGU(American Geophysical Union) and AMS(American Meteorological Society) Annual Meetings

- **What's New**: OceanAI는 오픈소스 대형 언어모델(LLM)과 NOAA의 권위있는 해양 데이터 스트림에 실시간으로 접근할 수 있는 대화형 플랫폼입니다. 이 시스템은 사용자가 묻는 질문에 대해 API를 통해 데이터 호출을 수행하고, 이를 바탕으로 재현 가능한 자연어 응답과 데이터 시각화를 생성합니다. 기존의 대화형 AI 제품에 비해, OceanAI는 NOAA 자료를 이용한 신뢰할 수 있는 값을 생성하는 올바른 접근성을 보여줍니다.

- **Technical Details**: OceanAI는 세 가지 주요 디자인 전략을 통해 기존 접근 방식의 한계를 극복합니다. 첫째, 각 질문은 파라미터화된 함수 호출로 변환되어 NOAA와 같은 권위 있는 데이터셋에 접근합니다. 둘째, 자동화된 데이터 처리 및 시각화를 통해 전문 지식이 없는 사용자도 손쉽게 이용할 수 있습니다. 마지막으로, 모든 응답은 메타데이터를 포함하여 신뢰할 수 있는 관측치를 바탕으로 최신 결과를 제공합니다.

- **Performance Highlights**: Blind 비교 실험에서 OceanAI는 세 가지 널리 사용되는 AI 채팅 인터페이스 제품 중 유일하게 NOAA 출처의 데이터를 제공하며, 다른 제품들은 답변을 거부하거나 신뢰할 수 없는 결과를 전달했습니다. OceanAI의 출력을 통해 신뢰성, 재현성, 투명성을 높이고 있으며, 해양 분야 내 AI 지원 의사 결정을 위한 확장 가능한 프레임워크를 제공하고 있습니다.



### ORANGE: An Online Reflection ANd GEneration framework with Domain Knowledge for Text-to-SQL (https://arxiv.org/abs/2511.00985)
Comments:
          16 pages, 4 figures, preprint

- **What's New**: 본 연구에서는 ORANGE라는 온라인 자기 진화 프레임워크를 소개합니다. 이 프레임워크는 번역 로그에서 SQL 쿼리를 파서하여 데이터베이스 특화 지식 베이스를 구축하여 도메인 내 지식을 축적합니다. ORANGE는 중첩된 Chain-of-Thought 전략을 통해 지식을 생성할 때 의미적 오류를 줄이고 나중에 SQL 번역의 정확성을 향상시킵니다.

- **Technical Details**: ORANGE는 세 가지 주요 구성 요소인 지식 분해(Knowledge Decomposition), 지식 검증(Knowledge Validation) 및 지식 향상된 Text-to-SQL 번역(Knowledge-Enhanced Text-to-SQL Translation)으로 구성됩니다. 이 시스템은 요약 SQL 쿼리를 생성하는 데 있어 개별 SQL 쿼리의 의미를 추적하는 중첩된 Chain-of-Thought 접근 방식을 채택하여 각 연산이 튜플의 의미를 어떻게 변화시키는지를 모니터링합니다.

- **Performance Highlights**: 여러 기준 벤치마크에서 ORANGE의 성능을 평가한 결과, 기존 기반보다 일관되게 높은 정확도를 기록했습니다. 이는 ORANGE 구조가 텍스트를 SQL로 변환하는 응용 프로그램에 실질적인 효과를 미친다는 것을 입증합니다. 복잡한 도메인 특정 쿼리를 다루는 데 있어 ORANGE의 효율성을 특히 강조하며, 이는 기존의 방법들이 가지던 제약을 극복할 새로운 방향성을 제시합니다.



### Keys in the Weights: Transformer Authentication Using Model-Bound Latent Representations (https://arxiv.org/abs/2511.00973)
Comments:
          Cite as A. S. Okatan, M. I. Akbas, L. N. Kandel, and B. Pekoz, "Keys in the weights: Transformer authentication using model-bound latent representations," in Proc. 2025 Cyber Awareness and Research Symp. (IEEE CARS 2025), Grand Forks, ND, Oct. 2025, pp. 6

- **What's New**: 본 논문에서는 Transformer 오토인코더에서 제안하는 새로운 방법론인 Model-Bound Latent Exchange (MoBLE)에 대해 설명하고 있습니다. 이 연구는 Zero-Shot Decoder Non-Transferability (ZSDN)이라는 개념을 도입하여 서로 다른 랜덤 시드로 초기화된 동일한 데이터로 훈련된 모델들 간의 복잡한 디코딩 문제를 다룹니다. 특히, 자기 디코딩(self-decoding)은 0.91 이상의 exact match와 0.98의 token accuracy를 기록하지만, 제로샷 크로스 디코딩(zero-shot cross-decoding)의 경우에는 우연의 일치에 그친다는 점이 제시되었습니다.

- **Technical Details**: 이 연구의 핵심은 동일한 아키텍처와 데이터로 훈련된 Transformer 모델들이 서로 다른 시드로 인해 비가역적인 잠재 공간(latent space)를 학습하는 현상입니다. 구체적으로, 인코더의 메모리 HLH^{L}은 오로지 자신의 디코더에 의해서만 디코딩 가능하며, 다른 모델의 디코더에 의해 접근하면 정확도가 0에 가까워지는 것이 관찰되었습니다. 이러한 결과는 attention 프로젝션과 피드포워드 레이어에서 시드가 유도한 기준 정렬 불일치(basis misalignment) 때문에 자연스럽게 발생합니다.

- **Performance Highlights**: MoBLE는 신뢰할 수 있는 AI 배포를 위한 경량 보안 레이어를 제공하며, 특히 항공 및 사이버 물리 시스템과 같은 안전이 중요한 영역에서 유용합니다. 제안된 방법론은 사용자의 접근 권한 제어 및 인증 메커니즘을 구성하는 잠재적 요소로 작용하며, 또한 학습 가능성 리스크에 대한 우려를 다루고 보완책을 제시합니다. MoBLE의 제안된 배포 체크리스트는 양자화, 무결성 태그, 운영 키 회전 등을 포함하여 실제 AI 파이프라인에 적용 가능성을 보여줍니다.



### Using Synthetic Data to estimate the True Error is theoretically and practically doab (https://arxiv.org/abs/2511.00964)
Comments:
          To appear at Machine Learning journal and ACML

- **What's New**: 이번 연구에서는 제한된 레이블 데이터 조건에서 훈련된 모델의 테스트 오류를 추정하기 위한 합성 데이터(synthetic data)의 활용을 체계적으로 조사했습니다. 우리는 실험을 통해 제안한 방법이 기존의 기준선(baselines)보다 더 정확하고 신뢰성 있는 테스트 오류 추정치를 제공함을 보였습니다. 이 연구는 합성 데이터의 품질과 최적화된 샘플 선택이 성능 평가에 미치는 영향 또한 다루고 있습니다.

- **Technical Details**: 이 논문에서 제안한 새로운 일반화 경계(generalization bounds)는 합성 데이터와 실제 데이터를 모두 인코딩하여 훈련된 모델의 진짜 오류를 추정하는 데 필요한 이론적 근거를 제공합니다. 연구자들은 적은 수의 레이블 샘플과 합성 데이터로부터 얻은 정보를 이용하여 평가를 위한 좋은 합성 샘플을 생성하는 방법을 제시했습니다. 이러한 경계는 모델의 동작이 데이터 공간의 다양한 지역에서 어떻게 연결되는지를 보여주며, 이는 일반화에 중요한 영향을 미칩니다.

- **Performance Highlights**: 제안된 방법은 시뮬레이션 및 테이블 데이터셋에서 실험적으로 평가되었으며, 기존 방법론과 비교했을 때 더 우수한 성능을 보여주었습니다. 연구 결과에 따르면, 합성 데이터의 단순한 사용으로도 테스트 오류 추정에 있어 더 낮은 변동성과 높은 정확성을 달성할 수 있었습니다. 최적화된 합성 데이터 생성 방법은 특히 모델 평가의 신뢰성을 크게 향상시키는 데 기여합니다.



### The Riddle of Reflection: Evaluating Reasoning and Self-Awareness in Multilingual LLMs using Indian Riddles (https://arxiv.org/abs/2511.00960)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 문화적으로 기반한 추론 능력을 인도 주요 7개 언어(벵갈리, 구자라트어, 힌디어, 칸나다어, 말라얄람어, 타밀어, 텔루구어)에서 평가합니다. 기존 연구가 주로 영어 중심의 평가에 국한되어 있다는 점을 지적하며, 다양한 언어로 된 수수께끼 해결 과정을 통해 모델의 성능을 분석합니다. 이 과정에서 전통적인 수수께끼와 맥락 재구성 변형을 결합한 다국어 수수께끼 데이터셋을 소개하여, LLM의 다언어적 추론 능력과 자기 평가 능력을 체계적으로 조명합니다.

- **Technical Details**: 논문에서는 LLM의 성능을 평가하기 위해 여러 프롬프트(예: zero-shot, few-shot)를 사용하여 수수께끼 해결 능력을 측정합니다. 5개의 LLM(Gemini 2.5 Pro, Gemini 2.5 Flash, Mistral-Saba, LLaMA 4 Scout, LLaMA 4 Maverick)의 성능을 비교하며, 수수께끼 해결에서 Gemini 2.5 Pro가 최고의 성과를 보이는 반면, 다른 모델들은 언어마다 상이한 정확성을 보입니다. 또한 자기 평가 실험을 통해 모델이 자신의 오류를 인식하는 능력을 분석하였으며, 높은 정확도를 보이는 모델이 자기 인식에서는 과신하는 경향을 보인다는 결과를 발견했습니다.

- **Performance Highlights**: 주요 결과로, Gemini 2.5 Pro는 전체적으로 가장 높은 수수께끼 해결 성능을 보였으며, few-shot 방식은 미미한 성과 향상을 가져오는 데 그쳤습니다. 모델의 초기 정확도와 자기 평가 능력은 역 비례 관계가 있는 것으로 나타났으며, 최고 성과 모델인 Gemini 2.5 Pro는 상대적으로 낮은 True Negative Rate(4.34%)를 보였습니다. 반면, 더 낮은 성과를 내는 LLaMA 4 Scout는 주목할 만한 자각 능력(42.09% True Negative Rate)을 보여, LLM의 다언어적 추론에서의 격차와 자기 인식에 있어 명확한 차이를 드러냈습니다.



### The Hidden Power of Normalization: Exponential Capacity Control in Deep Neural Networks (https://arxiv.org/abs/2511.00958)
- **What's New**: 본 연구는 정규화(normalization) 방법의 이론적인 기초를 탐구하며, 이를 통해 DNN(Deep Neural Network)의 용량(control capacity)에 대한 명확한 설명을 제시합니다. 특히, 정규화 레이어가 네트워크의 Lipschitz 상수(Lipschitz constant)를 감소시켜 최적화 및 일반화(generalization)에 미치는 영향을 분석했습니다. 이 결과는 정규화 방법이 심층 학습에서 성공적인 이유를 설명하는 데 중요한 기초를 제공합니다.

- **Technical Details**: 연구에서는 정규화 방법이 네트워크의 기능적 용량(functional capacity)을 제어하는 방식으로 Lipschitz 상수를 수량화합니다. 구체적으로, 정규화를 적용하지 않은 DNN은 입력이나 매개변수에 따라 매우 큰 또는 매우 작은 Lipschitz 상수를 가질 수 있으며, 이는 과적합(overfitting) 또는 과소적합(underfitting)에 이를 수 있습니다. 정규화 작업은 이 Lipschitz 상수를 기하급수적으로 줄일 수 있으며, 이것이 네트워크의 최적화 경관(loss landscape)을 부드럽게 하고, 일반화 능력을 향상시키는 데 기여합니다.

- **Performance Highlights**: 정규화 레이어의 도입은 손실 경관을 기하급수적으로 부드럽게 만들어, 더 빠르고 안정적인 최적화를 가능하게 합니다. 특히, 여러 개의 정규화 레이어가 적용될 경우 Lipschitz 상수가 기하급수적으로 감소하여 개선된 일반화 보장을 제공합니다. 이 연구는 심층 학습에 있어 정규화 방법의 효과성을 이론적으로 명확히 하는 중요한 기초를 세웠습니다.



### URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Mod (https://arxiv.org/abs/2511.00940)
Comments:
          Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)

- **What's New**: 본 논문에서는 URDF-Anything이라는 3D 멀티모달 대규모 언어 모델 기반의 엔드 투 엔드 자동 재구성 프레임워크를 제안합니다. 이 시스템은 포인트 클라우드와 텍스트 입력을 사용하여 기하학적 분할과 운동학적 매개변수 예측을 동시에 최적화합니다. 또한, [SEG] 토큰 메커니즘을 도입하여 세부 수준의 파트 분할을 가능하게 하면서, 운동학적 매개변수 예측의 일관성을 유지합니다.

- **Technical Details**: URDF-Anything은 3차원 멀티모달 대형 언어 모델(MLLM)을 활용하여, 물체의 기하학적 구조와 의미적 속성을 해석하고, 운동학적 구조를 추론하여 고충실도의 URDF 설명을 자동으로 생성합니다. 이를 위한 주요 메커니즘은 다이나믹 [SEG] 토큰을 사용하는 것으로, 이는 포인트 클라우드 피처로부터 객체 부품의 기하학적 분할을 안내합니다. 이 엔드 투 엔드 접근 방식은 객체의 운동학적 예측과 재구성된 기하학 간의 완전한 일관성을 보장합니다.

- **Performance Highlights**: 실험 결과, URDF-Anything은 기존 방법보다 기하학적 분할(mIoU 17% 향상), 운동학적 매개변수 예측(29% 평균 오차 감소), 물리적 실행 가능성(50% 향상)에서 훨씬 뛰어난 성능을 보여줍니다. 특히 이 방법은 훈련 세트에 없는 물체에서도 잘 일반화되어 우수한 성능을 발휘합니다. 이러한 결과는 자동화된 관절 물체 재구성을 위한 MLLM 프레임워크의 효과성을 강조합니다.



### Maestro: Orchestrating Robotics Modules with Vision-Language Models for Zero-Shot Generalist Robots (https://arxiv.org/abs/2511.00917)
Comments:
          Project website: this https URL

- **What's New**: 이번 논문에서는 VLM(비전-언어 모델)을 중심으로 로봇 정책을 개발하는 새로운 접근법인 maestro를 도입합니다. 기존의 로봇 데이터 수집 방식과는 달리 maestro는 VLM의 일반 능력을 특정 로봇 능력으로 보완하여 다양한 작업을 수행할 수 있는 프로그램 정책을 동적으로 구성합니다. 이런 점에서 maestro는 모듈형 로봇 시스템 공학의 전통적인 패러다임을 확장하며, 제로샷(Zero-shot) 성능에서도 기존 VLA 모델보다 우수한 결과를 보여줍니다.

- **Technical Details**: maestro는 센서 모터 작업을 수행하는 관리형 에이전트로, 인지(perception), 계획(planning), 제어(control) 모듈의 신중하게 관리된 세트를 활용하여 다양한 로봇 작업을 지원합니다. 이 시스템은 환경을 지속적으로 모니터링하고 새로운 관찰과 피드백에 따라 실시간으로 코드를 업데이트하며 실행하는 능력을 가지고 있습니다. 그에 따라 maestro는 기존의 강력한 로봇 도구 세트를 최대한 활용하여 다양한 작업을 처리할 수 있는 종합적인 기반을 제공합니다.

- **Performance Highlights**: maestro는 탁자 위 조작 기술과 같은 도전적인 작업에서 제로샷 성능이 우수하다는 것을 입증했습니다. 실험 결과, maestro는 VLA 모델들보다 뛰어난 성능을 보였으며, 새로운 도구를 쉽게 통합하고 기존 도구 세트를 활용하여 개선할 수 있는 유연성을 가지고 있습니다. 이러한 성능은 기존의 거대한 데이터를 기반으로 한 접근법 이상으로, 필요할 때마다 도구를 호출하여 제어할 수 있는 시스템 덕분입니다.



### Learning with Category-Equivariant Representations for Human Activity Recognition (https://arxiv.org/abs/2511.00900)
- **What's New**: 이 논문에서는 인간 활동 인식(HAR)을 위한 새로운 프레임워크인 category-equivariant representation을 도입합니다. 이 프레임워크는 시간, 스케일 및 센서 계층의 변화에 따른 신호의 변동을 포착하도록 설계되었습니다. 특히, 다양한 환경 조건에서 감지의 정확성을 높이고, 데이터의 해석성을 확보하는 데 중점을 둡니다.

- **Technical Details**: 제안된 방법은 Group×Poset 대칭 범주를 기반으로 하며, 강한 불변성 및 매개변수를 공유하는 그룹 대칭 컨볼루션을 사용합니다. 각 센서에서 RMS 정규화(게인), 축에서 크기로의 풀링(계층), 저주파 수치 변환(시간 이동)을 조합하여 compact한 피처 벡터를 만듭니다. 이러한 접근 방식은 데이터와 피처 사이의 자연 변환(natural transformation)을 통해 구조적으로 명확합니다.

- **Performance Highlights**: UCI HAR 데이터셋에서 제안된 방법은 기존 방법에 비해 약 46%의 정확도를 향상시켰습니다. 이는 다양한 테스트 환경에서도 안정성을 보여줍니다. 연구 결과는 다른 기하학적 및 분포적 접근 방식과 결합 가능성을 보여줍니다.



### Dynamic Logic of Trust-Based Beliefs (https://arxiv.org/abs/2511.00899)
- **What's New**: 이 논문은 전통적인 신념이 경험적 데이터에만 의존하던 것이 아니라, 최근에는 데이터의 공개적 발표에 기반하여 형성될 수 있음을 주장합니다. 연구의 주요 기술적인 기여는 데이터 정보 기반 신념과 데이터 발표 모달리티 간의 상호작용을 포괄하는 완전하고 유효한 공리 체계(Axiomatisation)를 제시하는 것입니다. 또한, 이 논리 시스템을 위한 비단순 다항 모델 검증 알고리즘(Polynomial Model Checking Algorithm)을 설명합니다.

- **Technical Details**: 이 논문은 다양한 데이터 변수(Data Variables)와 데이터 집합(Datasets)을 통해 형성된 신념에 대한 새로운 역학(Dynamics)을 탐구합니다. 특히, 데이터 변수의 신뢰성을 기반으로 하는 데이터 정보 지식 모달리티(Data-informed Knowledge Modality)를 도입하고, 이를 수학적으로 정의합니다. 예를 들어, 데이터 변수 tt가 신뢰받는 경우, 비공식적으로는 tt가 어떤 상태에서 신뢰할 수 있는지를 판단할 수 있습니다.

- **Performance Highlights**: 연구는 기존 Public Announcement Logic(PAL)의 한계를 뛰어넘어, 데이터 정보를 포함하는 신념이 어떻게 형성되는지를 보여줍니다. 이전 연구와 비교했을 때, 이 논문은 신뢰와 신념 간의 관계를 설명하며, 신뢰가 신념을 정의하는 한편, 기존 연구는 신념이 신뢰를 정의하는 방식을 제시합니다. 이 접근 방식은 더 나아가 데이터 발표 후 발생하는 신념의 변화를 효과적으로 설명하는 데 기여합니다.



### Android Malware Detection: A Machine Leaning Approach (https://arxiv.org/abs/2511.00894)
- **What's New**: 이번 연구는 Android 악성코드 탐지를 위해 Decision Trees, Support Vector Machines, Logistic Regression, Neural Networks 및 ensemble methods와 같은 머신 러닝 기법을 조사합니다. 연구는 Android 애플리케이션 데이터셋을 이용하여 각 모델의 정확도, 효율성 및 실제 적용 가능성을 평가합니다. 이 연구 결과는 ensemble 방법이 뛰어난 성능을 보이지만 모델 해석 가능성, 효율성 및 정확성 간의 절충이 있음을 보여줍니다.

- **Technical Details**: 연구에서는 다양한 머신 러닝 기법을 사용하여 악성코드 탐지 문제를 해결하였으며, 데이터셋은 실제 Android 응용 프로그램으로 구성되었습니다. 분석에 사용된 기술적 기법들은 모델의 정확도와 효율성을 비교하는 데 중점을 둡니다. 특히, ensemble methods는 다양한 기법의 결합을 통해 더 나은 성과를 달성하는 것으로 나타났습니다.

- **Performance Highlights**: 주요 발견 사항 중 하나는 ensemble methods가 다른 기법들에 비해 우수한 성능을 보였다는 것입니다. 그러나 연구진은 이러한 향상된 성능이 모델 해석 가능성과 효율성의 절충과 관련이 있음을 강조합니다. 이러한 통찰력은 Android 악성코드에 대응하기 위한 머신 러닝 연구 및 실제 활용 방안을 안내하는 데 중요한 역할을 합니다.



### Deep Generative Models for Enhanced Vitreous OCT Imaging (https://arxiv.org/abs/2511.00881)
- **What's New**: 이번 연구는 심층 학습(deep learning) 모델이 유리체(optical coherence tomography, OCT) 이미지 품질을 향상하고 획득 시간을 단축하는 데 도움을 줄 수 있음을 평가했습니다. 특히, Conditional Denoising Diffusion Probabilistic Models(cDDPMs), Brownian Bridge Diffusion Models(BBDMs), U-Net, Pix2Pix, 그리고 Vector-Quantised Generative Adversarial Network(VQ-GAN) 같은 최신 모델들을 사용했습니다. 연구 결과, cDDPM은 임상적 의미가 있는 유리체 OCT 이미지를 효과적으로 생성하면서 획득 시간을 4배 줄일 수 있는 가능성을 보여주었습니다.

- **Technical Details**: 연구는 두 가지 주요 요소, 즉 SD ART10 이미지를 사용하여 고품질의 SD 유리체 OCT 이미지를 생성하는 것과, 생성된 이미지를 pseudoART100 이미지와 비교하는 방식으로 진행되었습니다. 모델 성능 평가는 이미지 품질 지표(quality metrics)와 Visual Turing Tests를 통해 이루어졌으며, 이 과정에서 안과 의사들이 선별한 이미지들이 사용되었습니다. 특히, U-Net이 Peak Signal-to-Noise Ratio(PSNR: 30.230)와 Structural Similarity Index Measure(SSIM: 0.820)에서 가장 높은 성능을 기록하였습니다.

- **Performance Highlights**: cDDPM은 Visual Turing Test에서 가장 높은 점수를 획득하였으며, anatomical preservation이 85.7%에 이르는 성과를 보였습니다. 새로운 데이터에도 적용했을 경우, cDDPM이 생성한 유리체 영역에서 PSNR이 ART100 참고 이미지와 유사한 결과를 보였으며, 전체 이미지에서 ART1에 대한 조건으로 더 높은 PSNR을 기록하였습니다. 이러한 성과들은 정량적 지표와 임상 평가 간의 불일치를 드러내어, 결합된 평가의 필요성을 강조합니다.



### KFCPO: Kronecker-Factored Approximated Constrained Policy Optimization (https://arxiv.org/abs/2511.00880)
Comments:
          12 pages, 8 figures, submitted to ECAI 2025

- **What's New**: 본 논문에서는 안전한 강화 학습(Safe RL) 알고리즘인 KFCPO를 제안합니다. KFCPO는 확장 가능한 Kronecker-Factored Approximate Curvature (K-FAC) 기반의 제2차 정책 최적화와 안전성을 고려한 경량 경량이 통합된 방법을 제공합니다. 이러한 방법은 안전 경계를 넘어서는 것을 방지하기 위해 Q-경량 조작 메커니즘을 제공하며, 이를 통해 보상과 비용 그래디언트의 영향을 동적으로 조절합니다.

- **Technical Details**: KFCPO는 Fisher Information Matrix (FIM)를 근사화하여 효율적이고 안정적인 자연 그래디언트 업데이트를 수행합니다. 논문에서는 Constrained Markov Decision Processes (CMDPs)의 기초와 K-FAC 방법의 사용에 대해 설명하며, 안전 제약을 갖춘 목표 최적화를 위한 매개변수를 조정합니다. 이 과정에서 경량 범위를 기반으로 그래디언트를 조작하여 신뢰 영역 준수를 보장합니다.

- **Performance Highlights**: KFCPO는 Safety Gymnasium에서 OmniSafe 프레임워크를 사용하여 실험을 거쳤으며, 안전 제약을 준수한 기존의 최상의 베이스라인에 비해 10.3%에서 50.2% 높은 평균 수익률을 기록했습니다. 이러한 결과는 KFCPO의 안전성과 성능 간의 우수한 균형을 나타냅니다. 전체 환경에서 KFCPO의 효과를 통해 제안된 방법이 경쟁력 있는 해결책임을 입증합니다.



### Assessing LLM Reasoning Steps via Principal Knowledge Grounding (https://arxiv.org/abs/2511.00879)
Comments:
          Accepted to EMNLP 2025 Findings

- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 중간 추론이 실제 지식에 잘 근거하고 있는지를 평가할 수 있는 새로운 평가 프레임워크를 제안합니다. 이는 세 가지 주요 구성 요소로 이루어져 있으며, 특히 원자 지식(atomic knowledge)을 포함하는 주된 지식 수집(PK Collection)을 구축하였습니다. 이 프레임워크는 LLM의 추론에 필요한 지식을 정확하게 적용하는지를 평가하기 위한 지식 기반 평가 메트릭스와 경량화된 평가 모델을 포함합니다.

- **Technical Details**: 주된 지식 수집(PK Collection)은 LLM의 성능을 극대화하기 위한 필수 원자 지식으로 구성된 대규모 리포지토리입니다. 이 리포지토리는 MMLU 벤치마크(MMLU benchmark)를 기반으로 112,000개의 주된 지식 단위를 포함하고 있으며, 지식 기반 정밀도, 재현율 및 F1 점수와 같은 평가 메트릭스를 통해 지식의 활용도를 정량적으로 평가합니다. 이를 통해 모델이 정확하게 정보를 기억하고 적용하는지를 파악할 수 있습니다.

- **Performance Highlights**: 새롭게 제안한 평가 메트릭스는 LLM의 중간 단계에서의 지식 활용 부족이나 부적절한 적용을 명확하게 식별하는 데 유용합니다. 연구에서는 또한 이 지식 기반 메트릭스를 통해 모델의 추론 과정의 간결성을 조정하는 방법을 보여주었으며, 이는 최종 성능을 유지하면서도 효율적인 추론 단계를 생성하는 데 기여합니다. 실험 결과는 지식 기반 메트릭스가 모델 행동에 대한 해석 가능한 피드백을 제공함으로써 성능 향상을 유도하고, 모델의 지식 사용 및 토큰 소비를 효과적으로 조절할 수 있음을 나타냅니다.



### Fast Stochastic Greedy Algorithm for $k$-Submodular Cover Problem (https://arxiv.org/abs/2511.00869)
- **What's New**: 이 논문에서는 k-Submodular Cover (kSC) 문제를 연구합니다. 이는 인공지능 및 조합 최적화 작업에서 발생하는 고전적인 Submodular Cover 문제의 자연스러운 일반화입니다. 저자들은 Fast Stochastic Greedy 알고리즘을 제안했으며, 이는 기존 방법보다 쿼리 복잡도를 상당히 낮추면서 강력한 bi-criteria approximation을 달성합니다.

- **Technical Details**: kSC 문제는 주어진 임계값 T에 대해 특정 조건을 만족하는 k개의 부분 집합을 찾아내는 것입니다. 이 문제의 수학적 정의는 k가 1일 때 잘 알려진 Submodular Cover 문제로 귀결되며, 다양한 분야에 걸쳐 응용됩니다. 또한, 이 논문에서는 FastSG 알고리즘이 기존의 bi-criteria approximation을 개선하는 방법을 설명하고, 이는 O(nk log³(n))의 시간 복잡도를 갖는 상수 파라미터를 포함합니다.

- **Performance Highlights**: 저자들은 FastSG 알고리즘을 기존의 최첨단 알고리즘과 비교하여 실험을 진행하였으며, 실험 결과 우리의 접근 방식이 실행 시간과 솔루션 품질 모두에서 기존 방법들을 일관되게 능가함을 입증하였습니다. 또한, 논문에 포함된 표 1에서는 kSC 문제에 대한 솔루션 품질 및 쿼리 복잡도를 두 가지 측면에서 비교합니다.



### Occlusion-Aware Diffusion Model for Pedestrian Intention Prediction (https://arxiv.org/abs/2511.00858)
Comments:
          This manuscript has been accepted to the IEEE Transactions on Intelligent Transportation Systems as a regular paper

- **What's New**: 본 논문에서는 보행자의 횡단 의도를 예측하기 위해 새로운 'Occlusion-Aware Diffusion Model (ODM)'을 제안합니다. 이 모델은 가려진 동작 패턴을 재구성하고 향후 의도 예측을 안내하기 위해 활용됩니다. 특히, 기존 연구들이 간과했던 불완전한 관찰 상황에서도 보행자의 의도를 정확히 예측할 수 있는 가능성을 열어줍니다.

- **Technical Details**: ODM은 'occlusion-aware diffusion transformer' 아키텍처를 사용하여 가려진 패턴과 관련된 노이즈 기능을 추정합니다. 이 과정을 통해 문맥적 관계를 보다 잘 포착할 수 있도록 개선됩니다. 또한, 'occlusion mask-guided reverse process'를 도입하여 관찰 정보의 활용을 극대화하고 예측 오차의 축적을 줄이는 데 기여합니다.

- **Performance Highlights**: 다양한 가리기 신호 시나리오에서 제안된 방법의 성능을 광범위하게 평가하였으며, PIE 및 JAAD와 같은 인기 있는 벤치마크와 비교하여 기존 방법들보다 더 강력한 성능을 달성했습니다. 실험 결과, 본 방법이 가려진 시나리오에서 보행자 횡단 의도 예측의 정확성을 유의미하게 향상시킴을 보여줍니다.



### MULTI-Bench: A Multi-Turn Interactive Benchmark for Assessing Emotional Intelligence ability of Spoken Dialogue Models (https://arxiv.org/abs/2511.00850)
Comments:
          Submitted to ICASSP 2026

- **What's New**: 이 논문에서는 Multi-Bench라는 새로운 벤치마크를 소개하며, 이는 감정 지능(emotional intelligence)을 평가하는 데 초점을 맞추어 다중 턴 대화(multi-turn dialogue)에서 Spoken Dialogue Models (SDMs) 성능을 분석합니다. 기존의 대부분 벤치마크는 단일 턴 대화에 한정된 반면, Multi-Bench는 감정 이해(emotion understanding)와 이유(understanding), 감정 지원(emotion support) 및 적용(application)의 두 가지 트랙을 통해 SDMs의 능력을 평가합니다.

- **Technical Details**: Multi-Bench는 감정 지능을 정량화하기 위해 감정 인식(emotion recognition), 언어 및 음향 관점에서의 평가를 포함하여 다섯 가지 세부 작업으로 구성됩니다. 이 벤치마크는 약 3,212개의 샘플을 포함하고, 기본 및 고급 트랙으로 나누어 SDMs의 정교한 평가를 가능하게 합니다. 이러한 구조는 사용자 프로필(user profile)을 바탕으로 실제 대화와 유사한 상호작용을 지원하며, 사용자 응답은 텍스트로 생성하고 음성으로 변환하여 진행됩니다.

- **Performance Highlights**: 실험 결과, 현재의 SDMs는 기본적인 이해 작업에서 좋은 성과를 보이지만, 고급 다중 턴 대화 및 추론 관련 작업에서는 개선의 여지가 큽니다. 감정 인식과 응용 분야에서 특히 부족한 성능을 보였으며, 이는 Multi-Bench를 통해 보다 심층적인 평가가 필요함을 시사합니다. Multi-Bench는 SDMs의 감정 지능을 평가하는 데 중요한 기여를 할 것으로 기대됩니다.



### Pay for The Second-Best Service: A Game-Theoretic Approach Against Dishonest LLM Providers (https://arxiv.org/abs/2511.00847)
Comments:
          13 pages, 4 figures

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)과 서비스 제공자 간의 상호작용에서 발생할 수 있는 비윤리적인 조작을 해결하기 위해 공식 경제 모델을 제안합니다. 사용자가 여러 모델 공급자에게 반복적으로 쿼리를 위임하는 사용자-공급자 생태계에서 전략적 행동을 분석하고, 이 문제를 알고리즘 게임 이론과 메커니즘 설계의 관점에서 접근합니다. 연구의 핵심 기여는 근사 유인 호환 메커니즘을 제안하고, 예상 사용자 유틸리티의 보장을 다룬 것입니다.

- **Technical Details**: 모델의 기본 구조는 반복적 스택엘버그 게임(Stackelberg game)을 기반으로 하며, 사용자는 위임 메커니즘을 먼저 설정하고, 공급자들은 이를 전략적으로 반응하여 자신의 유틸리티를 극대화합니다. 제공자들은 비용 조정, 출력 토큰 연쇄 보고 등의 다양한 전략을 통해 사용자로부터 쿼리를 수락합니다. 제안된 메커니즘은 O(T^(1-ϵ) log T) 근사적으로 유인 호환성을 보장하며, 두 번째 최적 사용자 유틸리티를 달성하도록 설계되었습니다.

- **Performance Highlights**: 실험을 통해 제안된 메커니즘의 효과성을 입증했으며, 실제 API 가격 및 성능 환경에서 시뮬레이션을 수행했습니다. 제안된 전략은 모든 테스트된 전략 중 사용자에게 가장 높은 유틸리티를 제공하며, 공급자에게도 상당한 이익을 안겼습니다. 이러한 결과는 공급자 간의 경쟁을 통해 사용자 유틸리티를 보호하는 데 기여할 수 있음을 시사합니다.



### OmniBrainBench: A Comprehensive Multimodal Benchmark for Brain Imaging Analysis Across Multi-stage Clinical Tasks (https://arxiv.org/abs/2511.00846)
- **What's New**: 본 논문에서는 뇌 영상 분석을 위한 종합 멀티모달 시각 질문-응답(visual question answering, VQA) 벤치마크인 OmniBrainBench를 소개합니다. 이 벤치마크는 30개 검증된 의학 출처에서 수집된 15개의 뇌 영상 모달리티를 포함하여 총 9,527개의 검증된 VQA 쌍과 31,706개의 이미지를 제공합니다. OmniBrainBench는 임상 작업 흐름을 모사하며, 전문 방사선의에 의해 엄격히 검증된 15개의 다단계 임상 과제를 포괄합니다.

- **Technical Details**: OmniBrainBench는 CT, MRI, PET, SPECT 등 15개의 이미징 모달리티를 제공하여 임상 요구를 충족하는 종합적인 평가 프레임워크를 구성합니다. 이 벤치마크는 해부학적 구조 식별, 질병 진단, 병변 위치 파악 등 5단계의 전문 임상 과제를 포함하고 있으며, 다차원 평가 기준을 통해 MLLMs의 효과를 종합적으로 평가할 수 있습니다. 24개의 최신 MLLM 모델을 평가하여, 이 모델들의 다양한 뇌 영상 데이터에서의 성능을 비교 분석합니다.

- **Performance Highlights**: 연구 결과, 상용 MLLM(예: GPT-5)은 공개 소스 및 의학 모델보다 우수하지만 전문의에는 미치지 못하는 것으로 나타났습니다. 또한, 의학 MLLM의 성능은 매우 다양하며, 공개 소스 MLLM은 전반적으로 열세하나 특정 작업에서 두각을 나타냅니다. 복잡한 수술 전 작업에서는 MLLM의 성능이 크게 저하되며, 이는 시각-임상 추론 간의 간극을 드러내는 결과입니다.



### CodeClash: Benchmarking Goal-Oriented Software Engineering (https://arxiv.org/abs/2511.00839)
- **What's New**: 이번 연구에서는 CodeClash 라는 새로운 벤치마크를 도입하여 언어 모델(Language Models, LMs)의 목표 지향적 소프트웨어 개발 공정성을 평가합니다. 기존 평가 방법이 구체적이고 세분화된 작업에 중점을 두었다면, CodeClash는 다중 라운드 토너먼트 형식으로 LMs가 경쟁을 통해 최적의 코드베이스를 구축하도록 유도합니다. 이는 현실의 소프트웨어 개발 과정에서 요구되는 고차원적인 목표 달성의 복잡성을 반영합니다.

- **Technical Details**: CodeClash는 플레이어가 자신의 코드베이스를 유지 및 수정하며, 같은 코드 아레나에서 코드 경쟁을 진행하는 형식입니다. 각 라운드는 코드 수정 단계와 경쟁 단계로 구성되며, 이 과정에서 플레이어는 다른 경쟁자와의 성과를 바탕으로 전략을 수립해야 합니다. LMs는 오직 코드베이스 내의 정보만으로 개선 전략을 결정해야 하며, 매 라운드에 log 데이터를 분석하여 피드백을 활용할 수 있습니다.

- **Performance Highlights**: 연구 결과, LMs는 다양한 개발 스타일을 보여주었지만 경쟁 피드백 해석, 변경 사항 검증 및 코드베이스 유지 관리의 한계를 공유했습니다. 특히, 상위 모델이 전문가 프로그래머에 비해 모든 라운드에서 패배하는 현상이 두드러졌습니다. CodeClash는 오픈 소스로 공개되어 자율적이고 목표 지향적인 코드 개발에 대한 심도 있는 연구를 촉진할 것입니다.



### Parameter Interpolation Adversarial Training for Robust Image Classification (https://arxiv.org/abs/2511.00836)
Comments:
          Accepted by TIFS 2025

- **What's New**: 본 연구에서는 Parameter Interpolation Adversarial Training (PIAT)이라는 새로운 프레임워크를 제안합니다. PIAT는 각 에포크(epoch) 간 모델 파라미터를 보간(interpolate)하여 조정하며, 이는 결정 경계를 보다 완만하게 변화시켜 과적합(overfitting) 문제를 완화합니다. 이 방법은 모델의 수렴(convergence)을 돕고 더 높은 강인성(robustness)을 달성하는 데 기여합니다.

- **Technical Details**: PIAT는 이전과 현재 에포크의 모델 파라미터를 보간하여 조정하며, 초기 훈련 단계에서는 현재 파라미터에 더 집중하고 이후 단계에서는 이전 파라미터의 가중치를 증가시켜 결정 경계의 복잡성을 줄입니다. 또한, 새로운 정규화(metric)인 Normalized Mean Square Error (NMSE)를 제안하여 클린(clean) 예제와 적대적(adversarial) 예제 간의 로그레트(logits) 크기의 상대적 비율을 일치시킵니다. 이는 절대 크기보다 상대 크기 alignment를 더 중요하게 다룹니다.

- **Performance Highlights**: 다양한 기준 데이터셋에서 수행된 실험을 통해 PIAT 프레임워크가 CNN(Convolutional Neural Networks) 및 ViT(Vision Transformers) 모두에서 강인성을 크게 향상시켰음을 확인했습니다. PIAT는 복잡한 결정 경계를 피하고 모델의 수렴성을 안정화시켜, 학습 과정에서 나타나는 과적합 문제를 해결하는 데 매우 효과적입니다. 이러한 실험 결과들은 PIAT가 일반적이며 효과적인 방법임을 증명합니다.



### Linear Differential Vision Transformer: Learning Visual Contrasts via Pairwise Differentials (https://arxiv.org/abs/2511.00833)
Comments:
          NeurIPS 2025

- **What's New**: 이번 논문은 Visual-Contrast Attention (VCA)을 소개하며, 이 기술은 기존의 Multi-Head Self-Attention (MHSA) 레이어를 대체할 수 있는 방식으로, 이미지 분류와 생성에서 성능을 획기적으로 개선합니다. VCA는 O(N^2C)에서 O(Nnc)로 이론적 복잡도를 줄이며, 이 과정을 통해 모델의 인식 성능을 향상시키고 추가적인 계산 비용 없이 파라미터를 조정합니다.

- **Technical Details**: VCA는 두 개의 포지셔널 임베딩을 도입하여 정보를 정제하고, 각 헤드에서 시각적-대조 토큰을 평균 풀링하여 생깁니다. 이로 인해, 쿼리 필드를 압축하고 단순화하여 계산의 효율성을 증가시키며, 후속 단계에서 역대조 작업을 통해 정보 배열을 최적화합니다. 이러한 구조적 변화를 통해 각 패치가 주목받는 정도를 측정할 수 있습니다.

- **Performance Highlights**: VCA는 이미지 분류 작업에서 DeiT-Tiny 모델의 top-1 정확도를 72.2%에서 75.6%로 증가시키고, 다른 계층 구조 모델에서도 최대 3.1%의 성능 향상을 보여주었습니다. 이미지 생성에서도 FID-50K를 기존보다 2.1에서 5.2 포인트 낮추며 효율성과 품질을 동시에 개선하였습니다.



### Enhancing Adversarial Transferability in Visual-Language Pre-training Models via Local Shuffle and Sample-based Attack (https://arxiv.org/abs/2511.00831)
Comments:
          Accepted by NAACL2025 findings

- **What's New**: 이번 논문에서 소개된 Local Shuffle and Sample-based Attack (LSSA)는 시각-언어 사전 학습 모델(VLP)에서 멀티모달 적대적 공격의 효과iveness를 향상시키기 위해 새로운 접근 방식이 제안되었습니다. LSSA는 원본 이미지-텍스트 쌍을 기반으로 적대적 이미지를 생성하고 이를 통해 적대적 텍스트를 생성합니다. 이전 방법들은 입력 다양성이 부족하여 과적합(overfitting) 문제에 시달렸으나, LSSA는 이를 해결합니다.

- **Technical Details**: LSSA는 로컬 이미지 블록을 무작위로 섞고, 주변을 샘플링하여 적대적 이미지를 생성함으로써 입력 다양성을 높입니다. 이 과정에서 원본 텍스트와 생성된 적대적 이미지를 활용하여 높은 전이성(transferability)의 공격을 수행합니다. 실험 결과, LSSA는 다양한 VLP 모델과 데이터셋에서 멀티모달 적대적 예제의 전이성을 크게 향상시켰습니다.

- **Performance Highlights**: LSSA는 여러 모듈에서 기존의 멀티모달 공격방법들을 초월하는 성능을 보여주었습니다. 특히, 이미지 캡셔닝(image captioning)과 비주얼 그라우딩(visual grounding) 태스크에서도 뛰어난 결과를 내었으며, Large Vision-Language Models (LVLMs)에서의 평가에서도 좋은 성과를 거두었습니다. 실험을 통해 LSSA가 다양한 VLP 모델 및 다운스트림 태스크에서 공격 성능을 크게 개선함을 확인하였습니다.



### Towards Ultra-Low Latency: Binarized Neural Network Architectures for In-Vehicle Network Intrusion Detection (https://arxiv.org/abs/2511.00828)
Comments:
          6 pages, accepted and presented at INISTA 2025 (this https URL)

- **What's New**: 이 논문은 차량 내부 통신을 위한 Control Area Network (CAN) 프로토콜의 취약점을 개선하기 위해 Binarized Neural Networks (BNNs) 기반의 경량 침입 탐지 시스템을 제안합니다. 기존 연구에서 머신러닝과 딥러닝 기법들을 통해 보안 강화를 시도했던 맥락에서, 본 연구는 패킷 데이터, 메시지 ID 및 CAN 메시지 주파수를 활용하여 발생할 수 있는 침입을 효과적으로 탐지합니다.

- **Technical Details**: 제안된 시스템은 BNN 프레임워크를 통하여 경량화된 침입 탐지 기능을 수행하며, 메시지 ID 및 주파수와 같은 비이진(non-binary) 특성을 결합하여 다중 클래스 네트워크 트래픽 분류와 비정상 탐지 모두에 있어 우수한 성능을 나타냅니다. 특히, BNN은 1비트 가중치와 활성화를 활용하여 메모리 요구사항을 최소화하고, 고속의 실시간 처리 능력을 제공합니다.

- **Performance Highlights**: 본 연구에서 개발된 시스템은 마이크로컨트롤러와 Gateway ECUs에 매우 적합하여 CAN 버스 안전 응용에 필요한 실시간 요건을 충족합니다. 두 가지 데이터셋에서 수행된 실험을 통해 기존의 딥러닝 모델들과 비교하여 탐지율, 정확도, 모델 크기 및 응답 시간과 같은 다양한 메트릭을 바탕으로 성능-복잡도 균형을 잘 유지함을 확인하였습니다.



### GUI-AIMA: Aligning Intrinsic Multimodal Attention with a Context Anchor for GUI Grounding (https://arxiv.org/abs/2511.00810)
- **What's New**: 본 연구에서는 GUI-AIMA라는 새로운 접근 방식을 제안하는데, 이는 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 내재적 주의(attention) 메커니즘을 활용하여 GUI 기초 작업을 보다 효율적으로 수행할 수 있도록 한다. GUI-AIMA는 텍스트 기반 좌표 생성(task) 방식 대신, 시각적 패치(visual patches)를 선택하고 그 안에서 클릭 위치를 결정하는 방식으로 진행된다. 이 프레임워크는 85,000개의 스크린샷에 대해 교육되어 데이터 효율성(data efficiency)을 보여주며, 경량 모델(light model)이 MLLM의 내재적 기초 능력을 유도할 수 있음을 검증하였다.

- **Technical Details**: GUI-AIMA는 패치 기반(supervision) 학습을 통해 MLLM의 다중 헤드 자기 주의(multi-head self-attention)를 훈련시키며, 특히 주의 헤드 가중치(attention head weighting) 메커니즘을 사용하여 쿼리-시각 쿼리의 상관관계에 따라 각 주의 헤드의 중요도를 조정한다. 이를 통해 좌표 없는 GUI 기초 작업을 용이하게 수행할 수 있다. 학습 과정에서 특정 토큰에 대한 주의가 시각적 토큰에 집중될 수 있도록 하여 데이터 효율적으로 훈련할 수 있게 한다.

- **Performance Highlights**: GUI-AIMA는 ScreenSpot-Pro에서 평균 정확도 58.6%, OSWorld-G에서 62.2%를 달성하며, 3B 모델 중에서 최첨단 성능을 기록하였다. 특히, 전통적인 좌표 기반 방식과 비교했을 때, GUI-AIMA는 4.5%의 성능 향상을 보여준다. 이 모델은 상대적으로 적은 데이터로도 강력한 성능을 발휘하며 시각적 피드백에 기반한 인간의 행동을 모방해 더 효과적인 GUI 기준을 제공한다.



### Logic-informed reinforcement learning for cross-domain optimization of large-scale cyber-physical systems (https://arxiv.org/abs/2511.00806)
- **What's New**: 이 논문에서는 사이버-물리 시스템(CPS)을 위한 로직을 통합한 강화 학습 방법인 LIRL을 제안합니다. 기존의 계층적 접근 방식이 전역 최적성을 희생하는 경향이 있는 반면, LIRL은 적응형 하이브리드 맨폴드에서 저차원 행동을 맵핑하는 투영을 통해 제약 조건의 만족을 보장합니다.

- **Technical Details**: LIRL은 표준 정책 기울기 알고리즘에 투영을 추가하여, 첫 번째 논리 기반으로 정의된 하이브리드 행동 공간에서 저차원 잠재 행동을 효율적으로 맵핑합니다. R2AMS는 기어 조립 작업을 수행하는 로봇 작업 셀 세트로 구성되어 있으며, 이는 복합적인 의사결정 공간에서의 동적 최적화를 가능하게 합니다.

- **Performance Highlights**: 실험 결과 LIRL은 기존의 산업적 계층 스케줄링 방법에 비해 36.47%에서 44.33%까지 생산 시간-에너지 목표를 단축시킵니다. LIRL은 제약 위반 없이 성능이 우수하며, 안전한 CPS에 적용 가능한 새로운 최적화 경로를 제공합니다.



### Attention Saturation and Gradient Suppression at Inflection Layers: Diagnosing and Mitigating Bottlenecks in Transformer Adaptation (https://arxiv.org/abs/2511.00797)
- **What's New**: 이 논문은 프리트레인(Pre-trained) 트랜스포머 모델들이 소스 패턴에 대해 과도한 자신감을 보이고, 새로운 타겟 도메인 패턴을 형성하는 데 어려움을 겪는 문제를 다룹니다. 연구자들은 출력 포화(output saturation)로 인한 그래디언트 억제가 적응 과정을 제한한다는 사실을 밝혀냈습니다. 또한, 이를 진단하기 위한 지표로 시각적 측정 방법들을 도입하였고, 이를 통해 기존 방법보다 효과적인 LoRA 삽입 전략을 제안합니다.

- **Technical Details**: 기존의 그래디언트 최적화 방법들은 보수적으로 작동하여, 기존 최소값 주변의 작은 조정만을 수행합니다. 이에 반해 본 연구는 실험을 통해 모델이 고차원 구성(adapation) 과정에 집중하게 됨을 보여줍니다. 중요한 진단 지표로는 주의 엔트로피(attention entropy), 활성화 그래디언트(norm of activation gradients), 파라미터 그래디언트(norm of parameter gradients), Delta-CKA가 포함됩니다.

- **Performance Highlights**: BERT-base 모델을 이용한 실험 결과, 과훈련된 초기화(over-trained initialization)는 inflection layer에서의 LoRA 삽입을 통해 성능이 향상된 반면, 부족한 훈련 조건(under-trained initialization)에서는 성능 저하가 발생했습니다. 특히 기본 피쳐가 강할 경우, inflection layer의 해제가 고차원 적응을 촉진시키지만, 피쳐가 약할 경우에는 낮은 수준의 재구성을 위해 전체 경로를 해제해야 함을 확인했습니다.



### FedOnco-Bench: A Reproducible Benchmark for Privacy-Aware Federated Tumor Segmentation with Synthetic CT Data (https://arxiv.org/abs/2511.00795)
Comments:
          Published in IEEE

- **What's New**: 이번 연구에서는 FedOnco-Bench라는 데이터베이스를 소개하며, 이는 암 관련 CT 스캔 데이터셋을 사용하여 개인정보 보호를 고려한 분산 학습(Federated Learning, FL)을 위한 평가 기준을 제공합니다. FedOnco-Bench는 암 종양 분할(segmentation) 성능과 개인 정보 유출을 평가하며, 다양한 FL 방법론(FedAvg, FedProx, FedBN 및 DP-SGD가 결합된 FedAvg)에 대해 연구합니다. 결과적으로, 개인 정보 보호와 유용성 사이의 뚜렷한 균형이 드러나며, FL의 취약점에 대한 새로운 통찰력을 제공합니다.

- **Technical Details**: FedOnco-Bench는 비동기식 크로스-사일로 FL 시스템을 시뮬레이션합니다. 본 연구에서는 여러 클라이언트와 중앙 서버를 포함하여 각 클라이언트가 로컬 데이터로 모델을 학습하고 업데이트를 서버에 전달합니다. FL 알고리즘 가운데, FedAvg는 클라이언트 모델 업데이트를 간단한 평균을 통해 집계하는 기본 알고리즘으로 사용되며, FedProx와 FedBN은 비동질적(non-IID) 데이터 처리의 어려움을 해결하기 위한 방식으로 선택되었습니다.

- **Performance Highlights**: 실험 결과는 FedAvg는 뛰어난 성능(Dice coefficient 약 0.85)을 보이지만, 개인 정보 유출(AUC 약 0.72) 위험이 더 높았습니다. 반면에 DP-SGD는 개인 정보 보호(AUC 약 0.25) 수준을 높이는 대신 정확도(Dice 약 0.79)가 감소하는 결과를 보여주었습니다. FedProx 및 FedBN은 비동질적 데이터 환경에서 균형 잡힌 성능을 제공하며, 각각의 알고리즘 성능을 측정하기 위해 다양한 메트릭이 포함되어 있습니다.



### Efficient Reinforcement Learning for Large Language Models with Intrinsic Exploration (https://arxiv.org/abs/2511.00794)
- **What's New**: 이번 연구에서는 강화 학습을 위한 검증 가능한 보상(Reinforcement Learning with Verifiable Rewards, RLVR)의 데이터 효율성을 개선하는 방법을 제안합니다. 이 방법은 PREPO라는 새로운 접근법으로, 프롬프트의 혼란도(prompt perplexity)와 상대 엔트로피(relative entropy)를 이용하여 훈련 효율성을 극대화합니다. PREPO는 기존 방법보다 더 적은 롤아웃 수로 동등한 성능을 보이며, 특히 Qwen와 Llama 모델에서 수학적 추론 벤치마크에서도 유의미한 결과를 달성합니다.

- **Technical Details**: PREPO는 두 가지 주요 구성 요소로 구성되어 있습니다. 첫째, 혼란도를 학습의 지표로 사용하여 모델이 이해하기 쉬운 컨텍스트에서 도전적인 컨텍스트로 점진적으로 발전할 수 있게 합니다. 둘째, 롤아웃의 상대 엔트로피를 통해 응답의 불확실성을 강화하고, 탐험을 촉진하는 결과를 우선시하여 데이터의 질을 향상시킵니다.

- **Performance Highlights**: PREPO는 기존 데이터 선택 기법보다 더욱 효과적인 데이터 효율성을 제공하며, 롤아웃 사용량을 40% 이상 줄이는 성과를 올렸습니다. 또한, PREPO는 Qwen 및 Llama 모델에서 둘 다 우수한 성능을 나타내며, 데이터가 전부 사용된 경우와 비교해서도 경쟁력을 유지합니다. 이는 RLVR의 효율성을 크게 높일 수 있다는 것을 시사합니다.



### Fast PINN Eigensolvers via Biconvex Reformulation (https://arxiv.org/abs/2511.00792)
Comments:
          7 pages, 3 figures, Machine Learning and the Physical Sciences Workshop NeurIPS 2025

- **What's New**: 본 논문은 고유값 문제를 해결하기 위한 Physics-Informed Neural Networks(PINNs)의 새로운 접근법을 제안합니다. 기존 PINN 알고리즘들이 전통적인 수치 방식에 비해 느린 점을 개선하기 위해 고유쌍을 찾는 과정을 이원볼록 최적화(biconvex optimization) 문제로 재구성합니다. 이 방법은 고유값 및 고유함수에 대해 검증 가능한 수렴을 제공하며, 코드 역시 공개될 예정입니다.

- **Technical Details**: 본 연구는 비선형 활성화 함수가 포함된 다층 퍼셉트론을 사용하여 고유함수를 근사합니다. PINN 손실 함수는 미분 방정식을 만족하는 정도, 경계 조건 준수, 비자명한 해를 보장하기 위한 조건 등 여러 요소로 구성됩니다. 손실 함수를 이원 볼록(biconvex) 형태로 재구성함으로써 최적화 과정을 단순화하고, 각 하위 문제의 해를 해석적으로 구할 수 있게 하여 수렴성을 높였습니다.

- **Performance Highlights**: 실험 결과, 제안된 PINN-ACS 방법은 기존의 기울기 기반 PINN 훈련에 비해 최대 500배 이상의 속도를 개선하고, 높은 정확도를 달성했습니다. 또한, 백터 문제 및 헬름홀츠 방정식 문제 케이스에서 뛰어난 성능을 보였습니다. 이 성과는 다양한 경계 조건을 처리할 수 있는 유연성을 제공하며, 여러 고유쌍을 동시에 탐색할 수 있는 가능성도 확인했습니다.



### Class-agnostic 3D Segmentation by Granularity-Consistent Automatic 2D Mask Tracking (https://arxiv.org/abs/2511.00785)
Comments:
          Under review in Pattern Recognition

- **What's New**: 이번 논문에서는 3D 인스턴스 세분화(3D instance segmentation)를 위한 새로운 방법인 Granularity-Consistent Segmentation Policy를 소개합니다. 이 방법은 자동 2D Mask Tracking을 통해 동영상 프레임 간의 시계적 연관성을 유지하며 일관된 3D 가짜 라벨(pseudo labels)을 생성합니다. 이를 통해 기존 방법에서 발생하는 세그멘테이션의 불일치 문제를 해결하고 있습니다. 또한 3단계 커리큘럼 학습(curriculum learning) 프레임워크를 도입하여 점진적으로 다양한 뷰의 주석을 통합하는 방식으로 모델을 학습시킵니다.

- **Technical Details**: Granularity-Consistent Segmentation Policy는 객체를 자동으로 추적하여 동영상의 모든 프레임 간에 일관된 2D 마스크(mask)를 생성합니다. 이 과정에서는 동일한 객체가 다른 프레임에서 다르게 분할되는 현상을 해결하여 전체 장면에서의 지오메트리적 일관성을 확보합니다. 모델은 처음에 단편적인 단일 뷰 데이터에서 학습을 시작하여, 점진적으로 다중 뷰 주석(multi-view annotations)을 통한 전체 장면 감독(full-scene supervision)으로 이전해 나갑니다. 이를 통해 초기의 상충되는 2D 선행 정보(prior)로부터 일관된 3D 표현을 증류(distill)할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 ScanNet200과 ScanNet++와 같은 표준 벤치마크에서 최고의 성능을 기록하였습니다. 결과적으로 생성된 3D 가짜 라벨이 기존 방법보다 더 정확한 것으로 나타났습니다. 또한, 우리의 방법은 오픈-보캐뷸러리(open vocabulary) 능력을 검증하여, 제한된 학습 샘플로도 희귀 객체를 인식하고 세밀한 객체 분류에서도 뛰어난 성능을 보여주었습니다.



### Quantifying truth and authenticity in AI-assisted candidate evaluation: A multi-domain pilot analysis (https://arxiv.org/abs/2511.00774)
Comments:
          10 pages, 10 tables, 2 figures, and 1 page of supplemental materials

- **What's New**: 이 논문은 AlteraSF라는 AI 네이티브 이력서 검증 플랫폼을 이용하여 수집된 익명 후보 평가 데이터에 대한 회고 적 분석을 제시합니다. 이 시스템은 이력서의 주장을 평가하고, 상황에 맞는 검증 질문을 생성하며, 사실적 유효성과 직무 적합성의 정량적인 축을 따라 성과를 측정합니다. 추가적으로, 질적 진실성 탐지도 포함되어 있습니다.

- **Technical Details**: 여섯 개의 직군에서 1,700개의 지원서를 분석한 결과, 이 플랫폼은 스크리닝(확인) 시간을 90-95% 단축시켰고, AI 지원 또는 복사된 응답과 일치하는 측정 가능한 언어 패턴을 탐지했습니다. 연구 결과는 후보자의 진실성을 사실적 정확성뿐만 아니라 언어적 진정성의 패턴을 통해 평가할 수 있음을 보여주고 있습니다. 이러한 멀티 디멘셔널 인증 프레임워크는 채용 효율성을 개선할 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 이 플랫폼은 채용 프로세스의 효율성을 크게 향상시키며, AI 중재 평가 시스템에 대한 신뢰를 증대시킬 수 있음을 시사합니다. 본 논문의 결과는 다차원 검증 시스템이 직무와 관련된 진실성을 확보하고 채용 시간을 단축하는 데 기여함을 나타냅니다. 전반적으로, 이러한 분석은 AI에 의한 후보 평가에서 신뢰성을 강화할 수 있는 잠재력을 암시합니다.



### EP-HDC: Hyperdimensional Computing with Encrypted Parameters for High-Throughput Privacy-Preserving Inferenc (https://arxiv.org/abs/2511.00737)
Comments:
          To appear on ASP-DAC 2026

- **What's New**: 이 논문에서는 동형 암호(homomorphic encryption, HE)가 제공하는 강력한 개인 정보 보호(privacy protection)를 유지하면서도, 기존의 HE 접근 방식의 높은 계산 비용(computational cost)을 해결하기 위해 새로운 접근 방식인 암호화된 파라미터(HDC with encrypted parameters, EP-HDC)를 제안합니다. EP-HDC는 클라이언트 측에서 HE를 활용하여 동형 암호화된 모델을 사용하여 추론을 수행하여, 데이터 전송 그리고 암호화 오버헤드(overhead)를 효과적으로 줄이는 방법입니다.

- **Technical Details**: EP-HDC는 다수의 클라이언트를 지원하면서도 사용자 데이터와 모델 파라미터에 대한 강력한 보호를 제공하는 확장성(scalability)을 특징으로 합니다. 이 방법은 클라이언트 측의 개인 정보 보호 기계 학습(privacy-preserving machine learning, PPML) 응용 사례를 물론, 양자화(quantization), 아키텍처(architecture), HE 관련 파라미터를 포함하는 설계 공간 탐색(design space exploration)도 다룹니다.

- **Performance Highlights**: 실험 결과는 BFV 스킴을 사용하여 수행되었으며, Face/Emotion 데이터셋에서 제안된 방법이 기존 PPML 방법에 비해 배치 추론(batch inference)의 처리량(throughput)과 대기 시간(latency)을 각각 36.52배에서 1068배, 6.45배에서 733배 향상시킬 수 있음을 보여줍니다. 또한 정확도(accuracy)의 저하는 1% 미만으로 유지됩니다.



### FeNN-DMA: A RISC-V SoC for SNN acceleration (https://arxiv.org/abs/2511.00732)
- **What's New**: 이 논문에서는 Spiking Neural Networks (SNNs)를 위한 새로운 RISC-V 기반 System-on-Chip(SoC)인 FeNN-DMA를 개발하였습니다. FeNN-DMA는 현대 UltraScale+ FPGA에서 SNN을 시뮬레이션하는 데 맞춰 디자인된 완전 프로그래머블 시스템으로, 기존의 고정 함수 SNN 가속기와 유사한 리소스 및 에너지 요구 사항을 보이며 훨씬 더 복잡한 모델을 시뮬레이션 할 수 있습니다.

- **Technical Details**: SNN은 개별 뉴런이 상태를 가지며 이벤트 기반으로 서로 통신하여 에너지 효율성을 극대화합니다. 고전적인 ANN 가속기(GPU 및 TPU)는 행렬 곱셈의 높은 산술 강도에 최적화되어 있어 SNN과는 궁합이 맞지 않아, FPGA는 메모리에 제약이 있는 작업에 적합한 선택이 됩니다. FeNN-DMA는 이러한 요구 사항을 충족하도록 설계되었으며, 파이프라인 아키텍처를 통해 높은 처리량을 발휘합니다.

- **Performance Highlights**: FeNN-DMA는 Spiking Heidelberg Digits 및 Neuromorphic MNIST 작업에서 최첨단 분류 정확도를 보여주며, 기존의 SNN 가속기와 비교할 때 뛰어난 성능을 발휘합니다. 특히, 다양한 뉴런 모델을 지원할 수 있어 복잡한 시뮬레이션에서 유리한 점을 가집니다. 이를 통해 최신 SNN의 요구사항을 성공적으로 충족시키는 유연성을 제공합니다.



### TRISKELION-1: Unified Descriptive-Predictive-Generative AI (https://arxiv.org/abs/2511.00711)
Comments:
          12 pages, 18 figures, submitted to arXiv (2025)

- **What's New**: TRISKELION-1은 기술적, 기계적 및 생성적 사고를 통합한 고유한 아키텍처로, 단일 encoder-decoder 구조 내에서 설명적 표현 학습, 예측 추론 및 생성적 합성을 동시에 최적화할 수 있음을 보여줍니다. 본 논문은 이러한 통합 모델을 MNIST 데이터셋에서 실험적으로 검증하였으며, 이러한 접근은 해석 가능성, 정확성 및 창의성을 연결하기 위한 보편적인 지능 아키텍처의 청사진을 제공하는 데 기여합니다.

- **Technical Details**: TRISKELION-1은 설명적 AI(Descriptive AI), 예측적 AI(Predictive AI), 생성적 AI(Generative AI) 세 가지 패러다임의 요소를 하나의 프레임워크로 통합합니다. 설명적 AI는 데이터의 패턴을 발견하고 해석 가능성을 높이며, 예측적 AI는 입력 데이터를 출력으로 매핑하고, 생성적 AI는 새로운 샘플을 합성합니다. 이 결합은 공동으로 최적화된 손실 함수(shared loss function)를 통해 이루어집니다.

- **Performance Highlights**: TRISKELION-1의 실험 결과는 98.86%의 분류 정확도와 0.976의 조정 랜드 지수(Adjusted Rand Index, ARI)를 달성하여 세 가지 패러다임 간의 협력이 생산적임을 보여줍니다. 이는 각 패러다임이 서로 경쟁하는 것이 아니라, 상호작용을 통해 더 나은 결과를 만들어낼 수 있음을 나타냅니다.



### A Voice-Enabled Virtual Patient System for Interactive Training in Standardized Clinical Assessmen (https://arxiv.org/abs/2511.00709)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)을 기반으로 한 음성 인식 가상 환자 시뮬레이션 시스템을 소개합니다. 이 시스템은 정신 건강 임상 전문가들이 표준화된 임상 평가를 수행하는 데 필요한 연습 기회를 제공하며, 임상 시험에서 데이터 품질을 향상시킬 수 있습니다. 이로 인해, 전문가의 교육과 실제 평가 경험을 더욱 향상시킬 수 있는 가능성이 열렸습니다.

- **Technical Details**: 연구팀은 특정 증상 프로파일, 인구 통계학적 정보 및 의사소통 스타일을 지닌 환자를 시뮬레이션하기 위해 LLM을 활용한 시스템을 개발하였습니다. 5명의 경험 있는 임상 평가자가 4가지 가상 환자 인물에 대해 20회의 MADRS 인터뷰를 수행하여 시스템을 평가하였습니다. 평가 결과, 가상 환자들은 강력한 임상 프로파일 일치를 보였으며, 평가者가 부여한 MADRS 점수와 설정된 점수 간의 평균 차이는 0.52로 나타났습니다.

- **Performance Highlights**: 가상 환자들은 임상 프로파일에 대한 높은 일치를 보여주었으며, 항목 간 평가자 간 신뢰도는 0.90으로 매우 높았습니다. 전문가 평가자들은 가상 환자의 질적 사실감과 일관성을 긍정적으로 평가하였고, 평균 평가 점수는 '동의'와 '강하게 동의함' 사이에 위치하였습니다. 이러한 결과는 LLM 기반 가상 환자 시뮬레이션이 높은 충실도를 가지고 임상 관련 연습 상황을 생성할 수 있는 실행 가능하고 확장 가능한 도구임을 제안합니다.



### Evolve to Inspire: Novelty Search for Diverse Image Generation (https://arxiv.org/abs/2511.00686)
Comments:
          14 pages, 10 figures, Accepted to Neurips 2025 GenProCC Workshop

- **What's New**: 본 논문에서는 WANDER라는 텍스트-이미지 생성 모델을 제안하여 단일 입력 프롬프트(prompt)로부터 다양한 이미지 세트를 생성하는 방법론을 다룹니다. 기존의 텍스트-이미지 디퓨전 모델들이 다양한 이미지 출력을 얻기 어려운 한계점이 있으며, 이러한 문제를 해결하기 위해 LLM(대형 언어 모델)을 사용하여 프롬프트의 의미적 진화를 통해 이미지를 생성합니다. WANDER는 CLIP 임베딩을 활용하여 신규성(novelty)을 정량화하고, emitters를 통해 전이적(과거 프롬프트 주변)인 피드백을 제공합니다.

- **Technical Details**: WANDER의 핵심은 샘플링된 프롬프트를 변형하는 변이(mutation)를 수행하는 것입니다. 모델은 이미지의 다양성을 높이기 위해 LLM과 CLIP 임베딩을 이용하여 이미지의 코사인 거리(cosine distance)를 계산합니다. 또한, emitters는 LLM에게 특정 방식으로 프롬프트를 변형하도록 지시하는 역할을 하며, emitters의 전략이 다양성에 미치는 영향을 극대화합니다. 이를 통해 WANDER는 기존의 진화적 프롬프트 최적화 기법들과 비교하여 현저한 성능 향상을 이룹니다.

- **Performance Highlights**: WANDER는 여러 번의 실험을 통해 이미지를 생성하는 데 있어 높은 다양성을 유지하면서도 적절한 관련성을 갖춘 결과를 산출합니다. 테스트 결과, WANDER는 기존 프롬프트 최적화 기법에 비해 7배 더 적은 토큰을 사용하며, Vendi Score를 통해 평가된 다양성 점수도 보다 우수합니다. 이러한 성과는 WANDER가 이미지 생성에 있어 효과적이고, 모델에 구애받지 않는 접근 방식을 제공함을 보여줍니다.



### Metadata-Aligned 3D MRI Representations for Contrast Understanding and Quality Contro (https://arxiv.org/abs/2511.00681)
- **What's New**: 본 논문에서는 MRI 대조를 통합적으로 표현할 수 있는 MR-CLIP 프레임워크를 소개합니다. 이 프레임워크는 DICOM 메타데이터와 볼륨 이미지를 정렬하여 MRI 대조를 학습하여, 수동 주석 없이 자동화된 분석을 가능하게 합니다. MR-CLIP은 또한 이미지-메타데이터 임베딩 거리 값을 통해 손상된 메타데이터를 식별함으로써 비지도 형태의 데이터 품질 관리도 지원합니다.

- **Technical Details**: MR-CLIP은 3D 이미지 인코더를 사용하여 볼륨 이미지를 추출하고, 주어진 DICOM 메타데이터를 자연어 템플릿으로 변환하여 공유 임베딩 공간으로 투영합니다. 주요 특징은 다양한 프로토콜 및 스캐너 간의 일관된 대조 학습을 위한 Supervised Contrastive (SupCon) 손실을 활용하여 데이터의 의미적 분류를 개선하는 점입니다. 훈련은 40,005명의 피실험자와 169,634개의 볼륨을 포함하는 대규모 데이터셋에서 수행되었습니다.

- **Performance Highlights**: MR-CLIP은 2D, 2.5D 및 3D 모델 테스트에서 높은 정확도를 보여주며, 특히 2.5D 모델은 88.7%의 최고의 전체 정확도를 기록했습니다. n-shot 분류에서 MR-CLIP의 성능은 전통적인 3D ResNet을 지속적으로 초과하였으며, 이는 메타데이터 기반의 비지도 사전 학습이 제한된 레이블 데이터 가운데서도 효과적임을 입증했습니다. 마지막으로, MR-CLIP은 데이터 품질 관리를 위한 높은 감도를 보여주며, 누락된 태그 감지에서 거의 완벽한 AUC 점수를 기록했습니다.



### Isotropic Curvature Model for Understanding Deep Learning Optimization: Is Gradient Orthogonalization Optimal? (https://arxiv.org/abs/2511.00674)
- **What's New**: 본 논문에서는 가중치의 행렬 구조를 활용하여 단일 반복에서 딥러닝 최적화를 분석하는 모델을 소개합니다. 이 모델은 손실 함수의 모든 섭동 방향에 걸쳐 2차 해시안(Hessian) 및 고차 항을 포함하는 등곡률(Iso- curvature) 가정에 의해 유도됩니다. 이를 통해 가중치 업데이트가 전체 손실 함수의 변화와 어떻게 관련되는지 이해할 수 있는 기회를 제공합니다.

- **Technical Details**: 논문에서 제안한 모델은 단일 반복에서 최적의 업데이트를 찾아내기 위해 평균 경사를 최대화합니다. 이는 딥러닝 아키텍처의 가중치가 행렬 구조로 조직되어 있다는 점에 초점을 맞추고 있습니다. 또한, 특정 곡률 성장 조건을 가정하여 원래의 경량 행렬의 특이 값들이 서로 더 가까워지도록 하는 최적 업데이트 행렬을 도출합니다.

- **Performance Highlights**: Muon's 구조적 접근 방식에 따라 기존 최적화기인 Adam보다 더 우수한 성능을 보여주었음에도 불구하고, 이 논문은 성차를 고려한 최적 업데이트에 대한 이론적 접근의 필요성을 강조합니다. 저자들은 고차원 모델에서의 곡률 정보의 근사화를 통해 새로운 최적화 방법을 설계할 수 있는 가능성을 제시하며, Muon 모델이 완벽하게 최적은 아님을 알립니다.



### ShadowLogic: Backdoors in Any Whitebox LLM (https://arxiv.org/abs/2511.00664)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)의 안전 배치에 대한 중대한 보안 취약점을 강조하며, ShadowLogic이라는 새로운 백도어 기법을 소개합니다. ShadowLogic은 모델의 계산 그래프에 uncensoring vector(비검열 벡터)를 주입하여 비밀스러운 트리거 문구를 활성화함으로써 모델의 콘텐츠 생성 안전 장치를 우회하는 방법입니다. 이는 LLM 파일이 악의적인 사용자가 접근할 수 있을 경우 발생할 수 있는 잠재적인 위험을 드러냅니다.

- **Technical Details**: ShadowLogic는 ONNX(Open Neural Network Exchange)와 같은 계산 그래프 형식에서 백도어를 삽입하는 단계별 방법을 제공합니다. 이 기법은 모델의 표면상 무해한 동작을 유지하면서 인지되지 않도록 트리거 로직을 숨깁니다. 이를 통해 공격자는 입력 사용자 프롬프트에서 특정 비밀 구문이 감지될 때만 uncensoring vector를 활성화할 수 있습니다.

- **Performance Highlights**: ShadowLogic을 Phi-3 및 Llama 3.2에 성공적으로 구현하여 60% 이상의 공격 성공률을 기록한 것으로 나타났습니다. 이러한 결과는 백도어를 삽입한 모델이 사용자의 입력에 따라 비검열된 응답을 생성할 수 있는 능력을 유지한다는 것을 보여줍니다. 이는 LLM 배치의 새로운 취약점으로, ONNX와 같은 계산 그래프 기반 형식의 보안 검증 필요성을 강조합니다.



### Lessons Learned from the Use of Generative AI in Engineering and Quality Assurance of a WEB System for Healthcar (https://arxiv.org/abs/2511.00658)
Comments:
          11 pages, 2 figures, in Portuguese language

- **What's New**: 이 논문은 생성적 인공지능(Generative AI) 기술이 소프트웨어 공학(Software Engineering, SE) 분야에 미치는 영향을 확인한 사례를 공유합니다. 저자들은 클리닉 시험에 사용될 웹 기반 소프트웨어 시스템을 개발하는 과정에서 Generative AI를 통합한 경험을 문서화했습니다. 개발 프로세스의 여러 단계를 관찰했고, 이 기술들이 품질을 높이고 생산성을 향상하는 데 어떻게 도움이 될 수 있는지를 조명하고 있습니다.

- **Technical Details**: 소프트웨어 개발 주기에서 발생하는 전통적인 수작업 의존도를 줄이기 위해, 대형 언어 모델(LLM)과 생성 모델을 활용하여 개발의 여러 단계에 대한 지원을 목표로 하였습니다. 이를 통해 요구 사항 수집, 설계, 개발, 테스트, 품질 보증 등 다양한 활동에 Generative AI 기술이 통합되는 과정을 실험했습니다. 특히, 프롬프트(prompt) 작성이 개발자와 AI 간의 의사소통에서 중요한 역할을 하였고, Markdown 언어를 사용해 결과를 정리하고 문서화하는 데 필요한 과정도 중요하게 다루어졌습니다.

- **Performance Highlights**: 논문에서 제시된 결과에 따르면, Generative AI의 전체 활용은 현재로서는 완전한 웹 시스템 구축에는 부족하지만, 개발 프로세스의 개선에 실질적인 기여를 했습니다. 특히, 생성적 AI 기술의 도입으로 새로운 프로젝트 산출물이 만들어지고, 전통적인 공학 실무에 새롭게 적응해야 할 필요성이 커졌습니다. 연구진은 이 기술들이 어떻게 기존의 작업 방법을 혁신할 수 있는지를 강조하며, 품질 보증 과정에서 발생하는 새로운 도전에 대한 통찰을 제공합니다.



### More Than A Shortcut: A Hyperbolic Approach To Early-Exit Networks (https://arxiv.org/abs/2511.00641)
- **What's New**: 이번 연구에서는 리소스가 제한된 장치에서 이벤트 감지의 정확성을 높이기 위해 새로운 Hyperbolic Early-Exit 네트워크(HypEE)를 제안합니다. 기존의 Early-Exit(EE) 네트워크의 한계를 극복하기 위해, 하이퍼볼릭 공간에서 EE 표현을 학습하는 방식으로 전환하였습니다. HypEE는 계층적 훈련 목표와 새로운 entailment loss를 도입하여, 네트워크의 깊은 레이어가 얕은 레이어의 표현을 기하학적으로 정제하도록 합니다.

- **Technical Details**: HypEE는 Lorentz 모델을 기반으로 하여, n차원 하이퍼볼릭 공간을 2개의 시트 중 한 시트 위에 표현합니다. 이 모델에서는 거리 계산을 로렌츠 내적을 사용하여 다루며, Hyperbolic Embedding을 통해 포인트의 유사성을 효과적으로 측정합니다. 하이퍼볼릭 공간에서 분류를 수행하기 위해 로렌츠 다항 로지스틱 회귀 분류기를 활용하며, 훈련 목표는 표준 교차 엔트로피 손실과 계층적 일관성 손실을 결합하여 구성됩니다.

- **Performance Highlights**: 실험 결과, HypEE는 다양한 오디오 이벤트 감지 과제와 백본 아키텍처에서 기존의 유클리드 EE 기준선보다 우수한 성능을 보였습니다. 특히, 초기 계산이 가장 중요한 단계에서 성능이 크게 향상되어, 기존 모델들보다 더 효율적이고 정확한 이벤트 감지를 가능하게 합니다. 더불어, 학습된 기하학은 신뢰할 수 있는 불확실성 측정을 제공하여, 후속 계산을 위한 새로운 트리거링 메커니즘을 구현하는 데 기여합니다.



### Node Preservation and its Effect on Crossover in Cartesian Genetic Programming (https://arxiv.org/abs/2511.00634)
Comments:
          Draft to cite in another paper before both papers are peer-reviewed for the evo*2026 conference, 21 pages, 5 figures

- **What's New**: 이 논문은 카르카레우스의 서브그래프 교차(subgraph crossover) 방법을 기반으로 하여, 노드 보존(node preservation) 전략을 도입한 새로운 교차 방법을 제안합니다. 노드 보존 기법은 교차 과정에서 명령어가 분리되지 않도록 보장하여 CGP의 탐색 성능을 향상시키려는 것입니다. 또한, 기존의 포인트 돌연변이(point mutation) 방식을 노드 변형(node mutation) 방식으로 변경해 기존의 점진적인 방법을 대체합니다.

- **Technical Details**: 노드 보존 전략은 CGP의 고정 길이 유전자 구성을 활용하여 교차를 효율적으로 구현하며, 이를 통해 명령어의 구조적 무결성을 유지합니다. 본 연구에서는 고전적인 교차 방법인 일점 교차(one-point crossover) 및 균일 교차(uniform crossover)와 노드 보존이 적용된 버전을 비교하였으며, 이는 CGP 모델의 특성에 맞춰 변형되었습니다. 이러한 방법은 교차 시 자식 프로그램의 성능을 개선하기 위한 중요한 기법으로 자리 잡을 수 있음을 보여줍니다.

- **Performance Highlights**: 이번 연구에서 수행한 실험 결과, 노드 보존을 적용한 교차 방법이 전통적인 교차 방법에 비해 더 나은 해결책을 생성하는 경향이 있음을 발견하였습니다. 특히 서브그래프 교차(Subgraph crossover)는 잘 작동하는 다양한 부모 간의 연결을 신중하게 처리하여 성능을 끌어올리는 데 기여하였습니다. 이러한 새로운 기법이 CGP의 교차 문제 해결을 위한 일반적인 접근법의 기초가 될 수 있을 것으로 기대됩니다.



### AgentGit: A Version Control Framework for Reliable and Scalable LLM-Powered Multi-Agent Systems (https://arxiv.org/abs/2511.00628)
- **What's New**: AgentGit는 대규모 언어 모델(LLMs)을 기반으로 한 다중 에이전트 시스템(MAS)에서 Git과 유사한 롤백(rollback) 및 분기(branching) 기능을 도입합니다. 이 프레임워크는 LangGraph 위에 구축되어 에이전트가 여러 경로를 탐색하고 비교할 수 있게 해줍니다. AgentGit은 안정적인 체크포인트로 롤백 및 분기 탐색을 가능하게 하여 에이전트 시스템의 신뢰성과 확장성을 높여줍니다.

- **Technical Details**: AgentGit는 상태 커밋(state commit), 상태 복원(state revert), 그리고 분기 작업(branching operations)을 지원하여 다양한 실행 경로를 탐색할 수 있도록 합니다. 에이전트가 오류 발생 시 마지막 안정적인 체크포인트로 자동으로 롤백할 수 있게 하여, 이전 작업을 반복하지 않고도 새로운 도구 또는 프롬프트를 직접 테스트할 수 있습니다. 이는 에이전트를 위한 강력하고 자가 수정 가능한 시스템으로 변환하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, AgentGit은 다른 프레임워크들보다 실행 효율성이 크게 향상되었습니다. 에이전트는 중복적인 작업을 피하면서 전체 실행 시간을 줄이고 자원 소비를 최소화할 수 있었습니다. 이러한 성과는 다중 에이전트 시스템 개발에서의 신뢰성과 확장성을 확실하게 개선하였습니다.



### Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering (https://arxiv.org/abs/2511.00617)
- **What's New**: 이번 논문은 거대 언어 모델(LLM) 제어의 통합적 접근 방식에 대해 설명하고 있습니다. 기존의 두 가지 방법론인 인-컨텍스트 학습(In-Context Learning)과 활성화 조정(Activation Steering)을 베이지안 관점에서 통합하여, 이들이 모델 행동을 제어하는 하나의 보다 큰 프레임워크의 특정 사례로 볼 수 있음을 제안합니다. 이 논문은 신뢰성 있는 의사결정을 위한 예측 가능한 베이지안 모델을 개발하였습니다.

- **Technical Details**: 저자들은 LLM의 행동 변화를 베이지안 신념 업데이트로 설명하며, 인-컨텍스트 학습은 개념의 신뢰도를 조정하고, 활성화 조정은 개념의 사전 확률을 변경하는 방식으로 작동함을 제안합니다. 문헌을 바탕으로 한 여러 실험은 이론적 모델이 LLM 행동의 변화를 예측할 수 있는지 검증하였습니다. 이 모델은 시그모이드 형태의 학습 곡선 등의 기존 현상을 설명하면서도, 매우 미세한 조정으로 급격한 행동 변화를 예측할 수 있음을 보여줍니다.

- **Performance Highlights**: 이 논문은 LLM의 행동 변화가 개입 제어(컨텍스트 및 조정 크기)에 따라 급격하게 일어날 수 있다는 것을 발견했습니다. 이를 통해 저자들은 개입이 어떻게 상호작용하여 서로 다른 행동 변화를 유발하는지에 대한 예측을 제공합니다. 또한, 이 베이지안 모델은 많은 샷의 탈출 통제를 예측하는 유용한 도구로 자리잡을 수 있음을 보여줍니다.



### EPARA: Parallelizing Categorized AI Inference in Edge Clouds (https://arxiv.org/abs/2511.00603)
Comments:
          15 pages,20 figures

- **What's New**: 본 논문은 EPARA라는 새로운 end-to-end AI parallel inference 프레임워크를 제안합니다. 이 프레임워크는 엣지 클라우드에서 AI 서비스를 개선하기 위한 것으로, 작업을 지연(latency) 및 주파수(frequency) 민감도에 따라 분류하여 자원 할당을 최적화합니다. EPARA는 세 가지 핵심 요소로 구성되어 있으며, 이를 통해 요청 수준(request-level)과 서비스 수준(service-level)에서의 작업-자원 할당을 동시에 수행합니다.

- **Technical Details**: EPARA의 구조는 다음과 같습니다: 1) 작업 분류에 따라 병렬 모드를 결정하는 병렬 할당기, 2) 특정 요청에 대한 계산을 수행하는 분산 요청 처리기, 3) 주기적으로 서비스 배치를 업데이트하는 상태 인식 스케줄러입니다. 이 시스템은 빈번하게 발생하는 요청을 효과적으로 처리하기 위해 다양한 GPU 자원을 효율적으로 관리해야 합니다. 특히, EPARA는 요청 수준과 서비스 수준의 할당 방법을 통합하여 실시간 요청 처리를 가능하게 합니다.

- **Performance Highlights**: 테스트베드 실험을 통해 EPARA는 이전 프레임워크들에 비해 최대 2.1배 높은 goodput을 달성하는 효율성을 보여주었습니다. 이 프레임워크는 LLMs(대형 언어 모델) 및 세분화(segmentation) 작업에 대한 사례 연구를 통해 그 유효성을 입증하며, 다양한 엣지 AI 추론 작업에 적응할 수 있는 능력을 가지고 있습니다. EPARA의 구현은 실제 단말장치와 마이크로 컴퓨터에서의 성능 평가를 포함하며, 성능 향상뿐만 아니라 사용자 요구 사항을 충족하는 안정성을 제공합니다.



### Diagnosing Hallucination Risk in AI Surgical Decision-Support: A Sequential Framework for Sequential Validation (https://arxiv.org/abs/2511.00588)
- **What's New**: 이번 연구는 척추 수술에서 큰 언어 모델(LLMs)의 임상 결정 지원 가능성을 탐구하면서, 오답(hallucinations)으로 인한 위험을 정량화하는 임상의 중심의 프레임워크를 소개합니다. 연구진은 30개의 전문가 검증 사례를 통해 6개의 주요 LLM을 평가하였으며, 특히 DeepSeek-R1이 86.03 ± 2.08의 전반적인 성능 점수를 기록했습니다. 연구 결과에 따르면, 사고 전개가 강화된 모델(extended thinking mode)들이 기본 모델보다 항상 우수한 성능을 보이지 않았고, 이는 임상적으로 신뢰성을 보장하기 위한 새로운 기준이 필요함을 시사합니다.

- **Technical Details**: 연구는 진단 정확도(diagnostic precision), 추천 품질(recommendation quality), 추론 강인성(reasoning robustness), 결과 일관성(output coherence), 지식 정합성(knowledge alignment) 평가를 포함한 다차원 스트레스 테스트(multidimensional stress-testing)를 실시하였습니다. 깊이 있는 성능 평가를 위해 다양한 임상 시나리오에서 LLM의 미세 조정을 분석했으며, 모델별로 추천 품질이 7.4% 감소한 반면, 합리성(rationality)과 가독성(readability)에서 소폭의 개선(+2.0%, +1.7%)이 있었습니다. 이러한 결과는 예방적 지침(actionable guidance)과 명확한 일관성 간의 불일치를 강조합니다.

- **Performance Highlights**: DeepSeek-R1이 전반적인 성능에서 가장 높은 점수를 기록했으며, Grok-3-Beta(Think) 모델도 기본 버전에 비해 성능이 향상되었습니다. Claude-3.7-Sonnet 모델의 경우, 사고 전개 모드가 기본 버전에 비해 오히려 저조한 결과를 보였으며, 이는 복잡한 추론 과정이 임상 신뢰성에 충분하지 않음을 나타냅니다. 연구 결과는 사고 체인 시각화와 안보 체계를 통합한 LLM의 안전한 배포를 위한 기준 설정의 필요성을 강조합니다.



### TRACES: Temporal Recall with Contextual Embeddings for Real-Time Video Anomaly Detection (https://arxiv.org/abs/2511.00580)
Comments:
          10 pages, 5 figures

- **What's New**: 본 연구는 문맥을 고려한 제로샷 비디오 이상 탐지(Zero-Shot Anomaly Detection) 문제에 접근합니다. 이 시스템은 새로운 사건을 탐지하기 위해 시간적 및 외관적 특징을 텍스트 메모리와 실시간으로 상관관계 지어 학습합니다. 기존의 접근법이 새로운 실제 환경에 대한 일반화 능력이 부족한 반면, 우리는 메모리 지원 파이프라인을 정의하여 비주얼 임베딩과 시간적 신호를 상관 관계 짓는 방법을 제시합니다.

- **Technical Details**: TRACE(Temporal Recall with Contextual Embeddings) 시스템은 이상 및 비이상 맥락을 메모리 뱅크에 저장하고, 동작-외관 융합 모듈을 통해 시간적 크로스 어텐션을 활용합니다. 이를 통해 동적인 행동 패턴과 시각적 의미를 결합하고, 텍스트 맥락 벡터와의 유사성을 통해 이상 가능성을 예측하는 제로샷 이상 점수 매기기 메커니즘을 구현하였습니다. 이 방법은 다양한 환경에서 맥락에 맞는 이상 '추적'을 재현하게 설계되었습니다.

- **Performance Highlights**: 실험 결과, 우리 방법이 UCF-Crime 데이터셋에서 90.4%의 AUC를, XD-Violence 데이터셋에서 83.67%의 AP를 달성하며 제로샷 모델 중 최신 기술에 도달했음을 보여줍니다. 또한, 높은 정확성과 설명 가능성을 가진 실시간 추론을 제공하여 실제 환경에서의 적용성을 높입니다. 우리는 크로스 어텐션 시간 융합과 문맥적 메모리를 융합함으로써 높은 충실도의 이상 탐지를 가능하게 하였습니다.



### FlashEVA: Accelerating LLM inference via Efficient Attention (https://arxiv.org/abs/2511.00576)
Comments:
          Technical Report

- **What's New**: 이번 연구에서는 FlashEVA라는 이름의 효율적인 구현을 소개합니다. FlashEVA는 EVA (Efficient Attention via Control Variates) 방법을 기반으로 하여 Transformer의 메모리 사용량과 계산 부담을 줄이는 동시에, 인퍼런스(inference) 중 우수한 성능을 유지할 수 있게 합니다. 이 방법은 1.5B 토큰을 사용하여 Transformer 모델을 미세 조정할 수 있도록 하며, 다양한 다운스트림 작업에서도 효과적으로 작용합니다.

- **Technical Details**: FlashEVA는 커스텀 CUDA 및 Triton 커널을 사용하여 효율적인 EVA 주의를 구현함으로써 메모리 부담이 줄어들게 합니다. 이 접근법은 토큰 수가 적은 경우에도 기존 성능의 대부분을 회복할 수 있도록 하며, 인퍼런스에서 최대 6.7배의 처리량과 5배의 GPU 메모리 사용량 감소를 기록합니다. 또한 FlashEVA는 하이퍼파라미터를 통해 처리량과 정확도 사이의 균형을 조절할 수 있는 기능을 제공합니다.

- **Performance Highlights**: FlashEVA는 긴 문서와 같은 장기 맥락에서의 인퍼런스에서 인상적인 성능 개선을 이룹니다. 특히, 인퍼런스 처리량이 6.7배 증가하고, GPU 메모리 사용량은 5배 감소하는 성과를 보입니다. 다만, 검색 중심의 작업에서는 여전히 성능 저하가 관찰되며, 이 문제는 향후 연구의 여지로 남아 있습니다.



### FTT-GRU: A Hybrid Fast Temporal Transformer with GRU for Remaining Useful Life Prediction (https://arxiv.org/abs/2511.00564)
Comments:
          5 pages, The 2025 International Conference on Computational Science and Computational Intelligence

- **What's New**: 이번 연구에서는 FTT-GRU라는 하이브리드 모델을 제안합니다. 이 모델은 Fast Temporal Transformer (FTT)와 gated recurrent unit (GRU)를 결합하여 나사 CMAPSS에서 RUL(잔여 유효 수명) 예측을 수행합니다. 기존의 LSTM 및 CNN 모델들이 가지는 단점을 극복하고, 글로벌 및 로컬 시간 의존성을 동시에 캡처할 수 있는 접근법입니다.

- **Technical Details**: FTT-GRU는 저차원/주파수 아키텍처를 사용하여 선형화된 주의를 적용하는 FTT 블록과, 순차 모델링을 위한 GRU 레이어를 결합합니다. 입력 시퀀스는 우선 FTT 블록을 통해 인코딩된 후 GRU 레이어로 전달되며, 마지막으로 밀집 회귀 헤드로 RUL 추정값이 도출됩니다. 모델은 30 타임스텝의 슬라이딩 윈도우를 사용하여 학습되며, 각 윈도우의 마지막 타임스텝에서 RUL이 레이블을 부여받습니다.

- **Performance Highlights**: FTT-GRU는 NASA CMAPSS FD001 데이터셋에서 RMSE 30.76, MAE 18.97, R² 0.45를 달성했습니다. 이는 최신 베이스라인 모델(TCN-주의 기반)에 비해 RMSE를 1.16% 및 MAE를 4.00% 개선한 수치입니다. 검증이 항상 안정적으로 이루어졌으며, 모델의 예측은 전반적으로 정확성과 높은 신뢰도를 보였습니다.



### Red-teaming Activation Probes using Prompted LLMs (https://arxiv.org/abs/2511.00554)
- **What's New**: 본 논문에서는 AI 시스템 모니터링에 있어 저렴하고 지연 시간이 적은 Activation Probes의 실제 적용 강인성을 탐구합니다. 저자들은 흑상자(adversarial) 압력 하에서 Activation Probes의 실패 모드를 밝히고, 모델을 재훈련하지 않고도 이를 파악하는 경량화된 적대적(real-time) 평가 방법을 제안합니다. 해당 방법은 반복적 피드백 및 인 컨텍스트 러닝(in-context learning)을 통해 Activation Probes의 높은 가치 있는 통찰력을 발견할 수 있도록 돕습니다.

- **Technical Details**: 연구에서 제안된 경량화된 적대적 평가 과정은 Activation Probe를 흑상자 분류기로 취급하고, 상용 LLM을 사용하여 입력 샘플을 생성하는 방식을 사용합니다. 생성된 샘플은 Probe에서 평가되고, 별도의 LLM을 통해 실제 레이블 및 시나리오 제약 조건을 확인합니다. 피드백은 성공 또는 실패와 간단한 사유를 포함하여 공격자에게 전달되며, 이 과정을 통해 전략을 개선할 수 있도록 지원합니다.

- **Performance Highlights**: 공격 모델로 강력한 LLM(GPT-5)과 여러 오픈 소스 모델을 사용하여 테스트한 결과, 현재 활성화 Probe는 실제로 50% 이상의 실패율을 보였습니다. 특히, GPT-5 모델이 88%의 허위 긍정 탐지가 가능함을 보여줍니다. 이 확인 과정에서 발견된 실패 패턴은 실제 머신러닝 애플리케이션에서의 향후 Probe 개선에 중요한 통찰력을 제공합니다.



### Temporal Fusion Transformer for Multi-Horizon Probabilistic Forecasting of Weekly Retail Sales (https://arxiv.org/abs/2511.00552)
Comments:
          5 pages, 2025 6th International Conference on Data Analytics for Business and Industry (ICDABI)

- **What's New**: 본 연구는 재고 관리와 프로모션에 필수적인 정확한 다중 기간 소매 예측을 위해 Temporal Fusion Transformer (TFT)를 사용한 새로운 접근 방식을 제안합니다. 이 연구는 2010년부터 2012년까지의 주간 월마트 매출 데이터를 기반으로 하며, 정적 매장 아이디와 시간에 따라 변하는 외생 신호(예: 공휴일, 소비자 물가 지수(CPI), 연료 가격, 기온)를 융합합니다.

- **Technical Details**: TFT는 Quantile Loss를 통해 1-5주 앞을 위한 확률적 예측을 생성하며, 90% 예측 구간을 보정하고 변수 선택 네트워크, 정적 강화, 시간적 주의 메커니즘을 통해 해석 가능성을 제공합니다. 2012년 고정 보유 데이터셋에서 TFT는 매장 주당 RMSE 57.9k USD와 $R^2$ 값 0.9875를 기록했으며, 5-fold 시간적 교차 검증을 통해 평균 RMSE가 64.6k USD, $R^2$가 0.9844로 측정되었습니다.

- **Performance Highlights**: TFT는 XGBoost(XGB), CNN, LSTM 및 CNN-LSTM 기초 모델보다 우수한 성능을 보여주며, 재고 계획 및 공휴일 최적화에 실용적인 가치를 제공합니다. 모델의 투명성을 유지하면서도 이러한 결과는 소매업계에서의 적용 가능성을 더합니다.



### Robust Single-Agent Reinforcement Learning for Regional Traffic Signal Control Under Demand Fluctuations (https://arxiv.org/abs/2511.00549)
- **What's New**: 본 연구에서는 교차로의 대기큐(queuing)로 인한 도시의 교통 혼잡을 해결하기 위한 새로운 단일 에이전트 강화학습(reinforcement learning) 프레임워크를 제시합니다. 기존의 교통 신호 제어(traffic signal control) 시스템이 복잡한 실제 교통 상황을 잘 모델링하지 못하는 한계를 극복하고자 중앙 집중형 의사결정(paradigm)을 활용했습니다. 이 모델은 도로 네트워크(topology)와 신호 타이밍 파라미터를 통합하여 효율적인 교통 신호 조정을 가능하게 합니다.

- **Technical Details**: 이 모델은 인접 행렬(adjacency matrix)을 사용하여 probe vehicle 데이터를 통해 실시간 대기 상태를 인코딩합니다. DreamerV3 세계 모델의 학습 능력을 활용하여 에이전트는 교차로를 선택하고 신호 단계(signal phase)를 조정하는 정책을 학습하게 됩니다. 보상 설계(reward design)는 대기 큐 소산(queue dissipation)을 우선시하며, 이는 대기 큐 길이에 직접적인 영향을 미칩니다.

- **Performance Highlights**: SUMO 시뮬레이션 실험에서는 다양한 수요 변동(10%, 20%, 30%) 상황에서 모델의 강력한 저변동성(anti-fluctuation) 능력이 입증되었습니다. 이 프레임워크는 대기 큐 길이를 유의미하게 감소시켜 교통 혼잡 문제 해결에 기여할 수 있습니다. 향후 연구에서는 훈련 과정에서 확률적 수요 변동(stochastic OD demand fluctuations)을 통합하고, 비상 상황(contingency events)을 고려한 지역 최적화 메커니즘을 탐구할 계획입니다.



### Air Pollution Forecasting in Buchares (https://arxiv.org/abs/2511.00532)
Comments:
          14 pages 3 figures

- **What's New**: 이 논문은 과거 몇 년간 PM2.5 수치를 예측하기 위한 다양한 기계 학습 모델을 설계하고 조정하며, 그 성능을 평가하는 데 중점을 두었습니다. PM2.5의 예측이 질병 예방과 조기 경고를 제공할 수 있기 때문에 중요한 연구 주제로 다루어집니다. 이 연구는 부쿠레슈티의 특정 데이터를 활용하여 잘 연구되지 않은 지역에서의 PM2.5 예측에 초점을 맞추고 있습니다.

- **Technical Details**: 본 연구에서는 1시간, 2시간, 4시간 및 8시간 단기 예측을 위해 다양한 기계 학습과 딥 러닝 모델들이 비교됩니다. 특히, 선형 회귀(liner regression) 알고리즘, 앙상블 기반 방법, 고급 순환 신경망(RNN) 및 변환기(transformer) 모델들이 포함됩니다. 데이터를 처리하는 과정에서 아웃라이어 감지, 선형 보간 및 정규화 기술이 적용됩니다.

- **Performance Highlights**: 결과적으로 기계 학습 모델과 딥 러닝 모델은 PM2.5 예측에서 서로 다른 성능을 보였으며, 특히 변환기 모델들은 다양한 수치 기준(MAE, RMSE, R2)에서 두드러진 성과를 나타냈습니다. 이 연구의 기여는 부쿠레슈티 데이터셋을 통해 기계 학습 및 딥 러닝 모델이 환경 데이터를 활용하여 예측을 개선할 수 있는 가능성을 보여줍니다. 또한, PM2.5 레벨 예측의 정확도를 높이는 수단으로서 외부 기상 변수의 중요성을 강조하고 있습니다.



### On Improvisation and Open-Endedness: Insights for Experiential AI (https://arxiv.org/abs/2511.00529)
Comments:
          Submitted to AAAI 2026 Creative AI for Live Interactive Performances Workshop (CLIP) as a work-in-progress paper

- **What's New**: 이 논문은 인공지능(AI)이 진정한 리얼타임(improvise) 즉흥성을 어떻게 배울 수 있는지를 탐구합니다. 연구팀은 음악 및 무용 분야의 6명의 즉흥 연주 전문가와의 심층 인터뷰를 통해 인공지능이 체험적으로 창의적인 방법으로 작업할 수 있는 원리를 규명하려고 합니다. 이러한 연구는 AI가 사람 혹은 다른 AI와 협업하여 즉흥 연주를 수행할 수 있는 가능성을 제시합니다.

- **Technical Details**: 즉흥성(improvisation)은 사전에 정해진 스크립트 없이 순간에 따라 새로운 행동을 생성하고 적응하는 능력을 포함합니다. 과거 연구들은 즉흥성이 음악, 무용 및 디지털 코드와 같은 다양한 형태에서 나타나는 것을 보여줍니다. AI 연구에서는 현재 AI 시스템이 창의성을 발휘하는 데 있어 자율성과 경험적 학습의 필요성을 강조하며, 이는 전통적인 데이터 기반 접근 방식과 다릅니다.

- **Performance Highlights**: 연구 결과는 AI가 사람의 행동과 상호작용하며 지속적으로 새로운 가능성을 창출할 수 있는 구조를 개발할 수 있다는 점을 시사합니다. AI와 인간 간의 협동적인 즉흥 연주가 가능해질 것이다는 점에서, 이는 창의적 에이전트를 설계하는 데 중요한 통찰을 제공할 것입니다. 본 연구의 실질적 의의는 AI가 진정한 즉흥성을 구현할 수 있는 원리를 해석하는 데 있습니다.



### HIP-LLM: A Hierarchical Imprecise Probability Approach to Reliability Assessment of Large Language Models (https://arxiv.org/abs/2511.00527)
Comments:
          under review

- **What's New**: 본 연구에서는 HIP-LLM(Hierarchical Imprecise Probability)을 제안하여 대형 언어 모델(LLMs)의 신뢰성을 평가하는 새로운 프레임워크를 구축하였다. 기존의 벤치마크 기반 평가 방식의 한계를 극복하고, LLM의 운영 프로파일(Operational Profile, OP)에 따라 신뢰성 평가를 수행한다. HIP-LLM은 계층적 모델링을 통해 하위 도메인과 시스템 전반의 신뢰성을 동시에 평가할 수 있는 능력을 갖추고 있다.

- **Technical Details**: HIP-LLM은 소프트웨어 신뢰성 공학의 기초 위에 구축되었으며, LLM의 신뢰성을 미래 작업에서 실패 없는 운영의 확률로 정의한다. 계층적 구조를 통해 서로 다른 도메인과 하위 도메인 간의 의존성을 표현하고, 불확실성을 포착하기 위해 부정확한 사전 분포를 통합한다. 이를 통해, 기존의 기준들과 비교할 때 더 정확한 신뢰성 특성을 제공한다.

- **Performance Highlights**: 여러 벤치마크 데이터셋에 대한 실험 결과, HIP-LLM은 기존의 벤치마크 평가 방법 및 최첨단 접근 방식보다 더 정확하고 표준화된 신뢰성 특성을 제공하는 것으로 나타났다. 또한, HIP-LLM에 대한 공개 저장소가 제공되어 연구자들이 접근할 수 있도록 하였다.



### Reasoning Planning for Language Models (https://arxiv.org/abs/2511.00521)
Comments:
          29 pages, 5 figures

- **What's New**: 이번 논문은 언어 모델 생성에서 적절한 추론 방법 선택의 중요성을 강조합니다. 고전적인 접근 방식은 보통 여러 후보 응답을 생성하고 집계 전략(aggregation strategy)을 통해 최종 답변을 선택하는데, 이는 후보 응답의 수가 많을수록 정확성이 높아진다는 가정을 기반으로 합니다. 저자들은 이러한 가정을 엄밀한 이론적 분석을 통해 재조명하며, 기존 집계 방법의 정확도 경계를 도출했습니다.

- **Technical Details**: 주요 기여로는 EPIC(Ensemble Planning with Contrastive learning) 프레임워크를 도입하였습니다. 이 프레임워크는 모델의 추론 능력과 쿼리-방법 호환성을 포착하는 공유 표현 공간을 학습합니다. 또한, 제안된 확률 경계를 정규화 항(regularizer)으로 사용하여 정확성과 계산 비용의 균형을 맞춘 유틸리티 중심 최적화(utility-driven optimization)를 적용합니다.

- **Performance Highlights**: 다양한 수학적 추론 작업에서 EPIC은 최적의 추론 방법을 일관되게 선택하여 정확도를 향상시켰습니다. 또한, 계산 오버헤드를 줄이는 데 기여하였습니다. 실험 결과, EPIC은 기존 방법보다 더 나은 성능을 보여 주목받고 있습니다.



### A Multimodal Dataset for Indoor Radio Mapping with 3D Point Clouds and RSSI (https://arxiv.org/abs/2511.00494)
Comments:
          11 pages, 7 figures, 3 tables, under review to Nature Scientific Data

- **What's New**: 이번 논문에서는 여러 개의 방이 있는 실내 환경에서 20가지의 다양한 AP 구성 하에 수집한 Wi-Fi RSSI(Wi-Fi Received Signal Strength Indicator) 측정값과 고해상도 3D LiDAR 스캔을 통합한 멀티모달 데이터셋을 소개합니다. 이 데이터셋은 인간이 없는 상황과 있는 상황에서의 두 가지 측정 시나리오를 캡처하여, 실내 환경에서의 동적 신호 전파에 대한 연구를 지원합니다. 이를 통해 신뢰성 있는 실내 무선 네트워크 계획 및 AP 배치 최적화를 위한 REM(Radio Environment Maps)의 정확한 추정을 가능하게 합니다.

- **Technical Details**: 제안된 데이터셋은 Wi-Fi RSSI 측정값과 3D LiDAR 스캔을 결합하여 실내 공간의 복잡성을 극복하고 있습니다. 이는 IEEE 802.11be(Wi-Fi 7)와 같은 새로운 고주파 표준의 데이터 기반 무선 모델링 연구에 사용할 수 있는 자원으로 개발되었습니다. 데이터셋은 실내에서의 신호 전파에 대한 다양한 환경 효과를 분석할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제공된 데이터셋은 최신 고용량 실내 통신 시스템 개발을 위한 연구를 촉진합니다. 복잡한 실내 환경에서의 무선 신호 특성을 정확하게 이해함으로써, 실시간 비디오 분석, 스마트 센싱, XR(Extended Reality)과 같은 대역폭 집약적이고 지연에 민감한 애플리케이션에 필요한 안정적인 무선 연결 주소를 만듭니다. 이 연구는 전반적으로 무선 통신의 전반적인 성능 향상에 기여할 것으로 기대됩니다.



### Investigating Label Bias and Representational Sources of Age-Related Disparities in Medical Segmentation (https://arxiv.org/abs/2511.00477)
Comments:
          Submitted to ISBI 2026

- **What's New**: 이 논문은 의료 영상의 분할 작업에서 발생할 수 있는 알고리즘 편향(algorithmic bias)의 원인을 조사합니다. 특히, 유방암 분할에서 연령에 따른 성능 차이를 분석하고, 자동화된 라벨의 오류가 모델의 실제 편향을 잘못 나타낼 수 있다는 것을 밝혔습니다. 이 연구는 유방암 MRI 이미지를 포함한 MAMA-MIA 데이터셋에 대한 공정성 감사(fairness audit)를 처음으로 수행하여 정량적 기준을 설정합니다.

- **Technical Details**: MAMA-MIA 데이터셋은 동적 대조강조 자기공명영상(DCE-MRI)으로 구성된 1,506명의 환자 데이터를 포함합니다. 각 데이터는 전문가가 주석을 단 마스크와 자동 생성된 nnU-Net 마스크 쌍으로 구성됩니다. 연구에서는 나이 기준으로 환자를 세 개의 그룹(젊은, 중간, 노인)으로 분류하고, 라벨 편향(label bias)과 표현 편향(representational bias)을 측정하기 위해 세밀한 우선 순위 접근 방식을 사용하였습니다.

- **Performance Highlights**: 전문가가 주석을 단 Gold-Standard 라벨과 nnU-Net 마스크를 비교하여 성능을 평가했고, DPD(유병률-차이)와 DIR(불균형 영향 비율) 같은 공정성 메트릭스를 사용하여 성능 차이를 정의했습니다. 실험 결과, 젊은 환자들의 사례는 본질적으로 학습하기 더 어렵고, 편향된 자동화된 라벨로 학습할 때 시스템적 편향이 강화된다는 직접적인 증거를 보여주었습니다.



### Longitudinal Vestibular Schwannoma Dataset with Consensus-based Human-in-the-loop Annotations (https://arxiv.org/abs/2511.00472)
- **What's New**: 본 논문은 자기공명영상(MRI)에서의 전정 신경종(Vestibular Schwannoma, VS) 자동 분할을 위한 신뢰성 있는 주석 데이터셋을 제시합니다. 기존의 수동 주석 과정은 시간이 많이 소요되며, 비용 또한 증가하는데 반해, 제안된 방법은 전문가의 검증과 딥러닝 기반 프레임워크를 결합하여 효율적이고 신뢰성 높은 주석을 가능하게 합니다. 이 방법은 다양한 데이터셋에서 안정적인 성능을 유지하며, 기존 수동 주석 프로세스에 비해 약 37.4%의 효율성 향상을 기대할 수 있습니다.

- **Technical Details**: 이 연구에서는 DL 기반 자동 분할과 전문가 검증 과정을 통합한 휴먼-인-더-루프 데이터 주석 방법론을 제안합니다. 주석 방법론은 모델 훈련, 추론 및 전문가 검증 등의 여러 단계로 이루어져 있으며, MRI 데이터셋은 다수의 의료기관에서 수집된 경우를 포함합니다. 연구에 사용된 데이터는 여러 병원에서 수집된 T1CE 및 T2 가중치 스캔으로 구성되어 있으며, 이들 데이터셋은 The Cancer Imaging Archive (TCIA)에서 공개될 예정입니다.

- **Performance Highlights**: 제안하는 접근 방식은 내부 검증 데이터셋에서 Dice Similarity Coefficient(DSC)가 0.9125에서 0.9670으로 크게 향상되는 성과를 거두었습니다. 전문가 평가를 통해 143개의 스캔에서 모델 개선이 필요한 미묘한 사례들이 밝혀졌으며, 이러한 피드백은 향후 모델 개선에 중요한 역할을 할 것입니다. 최종적으로 본 연구는 다양한 임상 환경에서 VS 자동 분할을 위한 높은 정확도를 달성하며 임상 적용 가능성을 강조합니다.



### Why Federated Optimization Fails to Achieve Perfect Fitting? A Theoretical Perspective on Client-Side Optima (https://arxiv.org/abs/2511.00469)
- **What's New**: 이 논문에서는 데이터 이질성(heterogeneity)으로 인한 성능 저하(performance degradation)의 원인을 이론적으로 설명하는 것을 주요 기여로 삼고 있다. 특히, 클라이언트의 데이터가 서로 다를 경우 서로 다른 로컬 옵티마(local optima)로 수렴하고, 이로 인해 전체 모델의 수렴이 어려워지는 현상을 분석했다. 이러한 분석은 여러 신경망 아키텍처와 작업을 통해 실험적으로 검증되었다.

- **Technical Details**: 이 논문은 연산의 수렴 거동을 보다 엄밀히 분석하는 새로운 이론적 프레임워크를 제안한다. 특히, 데이터의 이질성으로 인해 발생하는 두 가지 핵심 결과를 제시하고 있으며, 첫째로 로컬 옵티마 사이의 거리가 전역 목표(global objective)의 하한(lower bound)을 높여 모델이 클라이언트의 모든 데이터에 완벽하게 적합할 수 없음을 설명하고 있다. 둘째로, 최종 단계에서 전역 모델이 한정된 영역에서 진동(oscillation)하게 되어 최적치로 수렴하지 못하는 문제를 다룬다.

- **Performance Highlights**: 다양한 실험을 통해 논문에서 제안한 이론이 실험적으로도 검증되었다. 특히, GRU, ResNet-18, ViT와 같은 복잡한 신경망에 대한 실험 결과가 포함되어 있으며, 로컬 업데이트 라운드(local update rounds), 클라이언트 가중치(client weights), 참여 비율(participation rates) 등이 수렴(convergence)과 최적화에 미치는 영향을 설명하고 있다. 이러한 결과들은 비 IID(non-iid) 환경에서 성능 저하를 이해하는 데 중요한 통찰을 제공한다.



### Proactive DDoS Detection and Mitigation in Decentralized Software-Defined Networking via Port-Level Monitoring and Zero-Training Large Language Models (https://arxiv.org/abs/2511.00460)
- **What's New**: 이번 논문에서는 분산형 소프트웨어 정의 네트워킹(decentralized Software-Defined Networking, dSDN) 환경을 위한 새로운 탐지 및 완화 프레임워크를 제안합니다. 기존의 중앙 집중형(cSDN) 구조에서 발생할 수 있는 확장성 및 신뢰성 문제를 해결하고, DDoS 공격에 대한 취약점을 줄이기 위해 설계되었습니다. 특히, 이 프레임워크는 경량 포트 수준 통계(port-level statistics)와 함께 즉각적인 엔지니어링(prompt engineering) 및 맥락 내 학습(in-context learning)을 활용합니다.

- **Technical Details**: DeepSeek-v3 대형 언어 모델(LLM)을 사용하여 트래픽을 정상적(benign)인지 악성(malicious)인지 분류합니다. 이 과정에서는 추가적인 파인튜닝(fine-tuning)이나 재교육 없이도 가능하여 효율성을 제공합니다. 공격이 탐지되면, 악성 트래픽은 공격자의 포트에서 직접 차단되며, 정상 트래픽은 영향을 받지 않도록 합니다.

- **Performance Highlights**: 실험 결과는 다양한 DDoS 공격 시나리오에서 99.99% 정확성(accuracy), 99.97% 정밀도(precision), 100% 재현율(recall), 99.98% F1 점수(F1-score), 1.0의 AUC 값을 달성하는 등 거의 완벽한 탐지 성능을 보여줍니다. 이러한 결과는 분산 모니터링(distributed monitoring)과 제로 훈련(zero-training) LLM 추론의 결합이 DDoS 위협으로부터 dSDN 인프라를 안전하게 보호하는 효과적인 방어 메커니즘을 제공함을 강조합니다.



### DRIP: Defending Prompt Injection via De-instruction Training and Residual Fusion Model Architectur (https://arxiv.org/abs/2511.00447)
- **What's New**: 이번 논문에서는 DRIP이라는 새로운 방어 기법이 제안되었습니다. DRIP는 대규모 언어 모델(LLM)의 지시어와 데이터 의미를 효과적으로 분리하여, 악의적인 프롬프트 인젝션 공격으로부터 보호하는데 초점을 맞추고 있습니다. 이 기법은 두 가지 경량 메커니즘인 semantic disentanglement과 residual fusion pathway를 결합하여, 모델의 의미적 정합성을 강화합니다.

- **Technical Details**: DRIP는 두 개의 주요 아키텍처 구성 요소를 도입합니다. 첫째, token-wise de-instruction shift는 데이터 토큰의 의미적 분리를 수행하여 지시 semantics를 약화시키고, 내용의 의미는 유지합니다. 둘째, residual fusion pathway는 최상위 지시어의 영향을 강하게 하여 사전 훈련된 모델이 악의적 방해에도 불구하고 본래의 지시 의도를 따르도록 합니다.

- **Performance Highlights**: 실험 결과, DRIP는 다양한 프롬프트 인젝션 벤치마크에서 기존의 방어 방법들보다 우수한 성능을 보였습니다. 예를 들어, DRIP는 역할 분리를 49% 향상시키고 적응형 공격에 대한 성공률을 66% 감소시켰습니다. 또한, DRIP는 유지하기 위해 노력한 유틸리티도 저하되지 않았으며, 기존 모델과 유사한 성능을 유지했습니다.



### LIR: The First Workshop on Late Interaction and Multi Vector Retrieval @ ECIR 2026 (https://arxiv.org/abs/2511.00444)
Comments:
          Accepted workshop at ECIR 2026

- **What's New**: 최근 정보 검색(Information Retrieval, IR) 분야에서 Late Interaction Retrieval 방법이 주목받고 있습니다. ColBERT로 시작된 이 방법들은 단일 벡터 기반 신경 IR에 대한 강력한 대안으로, 세밀한 토큰 수준의 표현을 활용하여 일반화와 견고성을 제공합니다. 더욱이 이러한 모델들은 Reasoning-based 또는 Cross-modality Retrieval 같은 새로운 사용 사례에 특히 적합한 것으로 나타났습니다.

- **Technical Details**: Late Interaction Retrieval 모델은 문서와 쿼리를 각각의 토큰으로 표현합니다. 이들 모델은 쿼리 토큰과 문서 토큰을 비교하는 MaxSim 연산자를 사용하여 문서에 대한 쿼리의 적합도를 계산합니다. 이 접근 방식은 단일 벡터 방법에서 발생하는 정보 손실을 피하고, 토큰 간의 세밀한 상호작용을 가능하게 합니다.

- **Performance Highlights**: 2023년초, LLM Retrieval-Augmented Generation(RAG) 파이프라인의 인기로 ColBERT 모델에 대한 사용자 친화적인 도구가 증가하고, 수백만 번의 다운로드를 기록하며 공업적으로도 큰 관심을 끌고 있습니다. 다양한 연구 흐름 속에서, Multi-modal Retrieval과 같은 새로운 영역에서도 유망한 초기 결과를 보이고 있으며, 후속 연구와 협업의 촉진의 중요성이 강조되고 있습니다.



### Region-Aware Reconstruction Strategy for Pre-training fMRI Foundation Mod (https://arxiv.org/abs/2511.00443)
- **What's New**: 이번 연구에서는 resting-state fMRI를 위한 foundation model의 새로운 region-aware reconstruction 전략을 제안합니다. 기존의 무작위 영역 마스킹 방식을 넘어, Automated Anatomical Labelling Atlas (AAL3)를 활용하여 뇌의 의미적으로 일관된 영역을 선택적으로 마스킹하는 ROI 가이드 마스킹 전략을 도입했습니다. 이 방법은 기존의 마스킹 방식에 비해 ADHD 환자와 일반인을 구분하는 분류 정확도를 4.23% 향상시켰습니다.

- **Technical Details**: 우리의 self-supervised learning 프레임워크는 masked voxel reconstruction에 집중하는 pretraining 단계와 예측 작업에 대한 fine-tuning 단계로 구성됩니다. NeuroSTORM 모델을 기반으로 하여, ROI 기반 마스킹 전략을 도입함으로써 공간적 특이성을 향상시킵니다. AAL3 atlas를 사용하여 사전 정의된 해부학적 영역을 식별하고 이러한 ROI의 하위 집합을 선택적으로 마스킹합니다.

- **Performance Highlights**: ADHD-200 데이터셋을 사용한 결과, ROI 가이드 마스킹이 기존의 무작위 마스킹 방식에 비해 분류 정확도를 향상시켰습니다. 이 연구는 limbic 영역과 소뇌가 reconstruction fidelity 및 모델 표현에 중요한 역할을 함을 발견했으며, 향후 다양한 neuroimaging 데이터셋에서 이러한 접근 방식을 평가할 계획입니다. 우리 프레임워크는 fMRI 기반 응용 프로그램에서 일반화 및 해석 가능성을 향상시킵니다.



### Enhancing Frequency Forgery Clues for Diffusion-Generated Image Detection (https://arxiv.org/abs/2511.00429)
- **What's New**: 이 논문은 이미지 합성에서 큰 성공을 거둔 Diffusion 모델이 생성한 이미지의 악의적인 사용 가능성에 대한 우려를 다룹니다. 기존의 탐지기는 다양한 모델과 설정에서 구분 가능한 단서를 잡는 데 어려움을 겪고 있으며, 이는 보이지 않는 Diffusion 모델에 대한 일반화 능력과 다양한 교란에 대한 강인성을 제한합니다. 연구자들은 자연 이미지와 Diffusion으로 생성된 이미지의 주파수 대역에서 구성된 차이를 분석하여 F2C(Frequency Forgery Clue)를 활용한 효과적인 탐지기를 제안합니다.

- **Technical Details**: 이 논문은 자연 이미지와 Diffusion 생성 이미지 사이의 주파수 도메인에서의 차이를 분석합니다. 그 결과, Diffusion으로 생성된 이미지는 저주파에서 고주파로 갈수록 자연 이미지와 더 큰 차이를 보이는 것을 파악했습니다. 이 발견에 기반하여, 논문에서는 무게가 가중된 필터로 작용하는 주파수 선택 함수(frequency-selective function)를 도입하여 주파수 스펙트럼의 덜 구별되는 대역을 억제하고 더 유용한 대역을 강화하는 방법을 제안합니다.

- **Performance Highlights**: 다양한 Diffusion 생성 이미지 데이터셋에 대한 광범위한 실험 결과, 제안된 방법이 기존의 최첨단 탐지기보다 우수한 일반화 및 강인성을 제공함을 확인했습니다. F2C 기법은 기존 모델들이 특정 패턴에 의존하는 것과는 달리 자연 이미지와의 고유한 차이를 활용하여 보이지 않는 Model의 이미지를 효과적으로 탐지할 수 있습니다. 이 연구는 이미지 생성 기술의 악용을 막기 위한 새로운 접근 방식을 제시합니다.



### Leveraging Hierarchical Image-Text Misalignment for Universal Fake Image Detection (https://arxiv.org/abs/2511.00427)
- **What's New**: 본 논문에서는 생성 모델의 급격한 발전과 이에 따른 생성된 가짜 이미지 탐지의 필요성을 강조합니다. 기존의 이진 이미지 분류 방식은 시각적 단서에만 집중하여 과적합(overfitting)의 문제가 있으며, 새로운 방법론을 제안합니다. ITEM이라는 새로운 탐지기를 통해 이미지와 캡션 간 비정렬(misalignment)을 활용하여 가짜 이미지 탐지의 효율성을 높입니다.

- **Technical Details**: ITEM은 사전 훈련된 CLIP의 시각-언어 공간에서 이미지와 캡션 간의 비정렬을 측정하여 이미지 탐지 작업을 수행하는 MLP(다층 퍼셉트론) 헤드를 조정합니다. 추가적으로, 전체 이미지와 캡션에 설명된 각 의미적 객체에 대한 계층적 비정렬 방식을 도입하여 글로벌(global) 및 미세(local) 정밀 비정렬 정보를 탐색합니다. 이로 인해 더욱 강력하고 일반화된 탐지가 가능합니다.

- **Performance Highlights**: 다양한 생성 모델에 대한 광범위한 실험을 통해 본 방법이 기존의 최신 기술들보다 뛰어난 일반화 및 견고성을 보여줍니다. ITEM은 과거의 이미지 패턴에 과적합되지 않고 모든 이미지 유형에 대해 효과적으로 탐지할 수 있는 잠재력을 가지고 있습니다. 이를 통해, 생성된 이미지에 대한 범용 가짜 이미지 탐지 솔루션을 제시합니다.



### Bootstrap Off-policy with World Mod (https://arxiv.org/abs/2511.00423)
Comments:
          NeurIPS 2025

- **What's New**: 이 논문은 BOOM(Bootstrap Off-policy with WOrld Model)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 플래닝(planning)과 오프폴리시 학습(off-policy learning)을 통합하여, 액터 다이버전스(actor divergence) 문제로 인해 발생하는 데이터 분포의 변화로 인한 부정적인 영향을 완화합니다. BOOM은 정책(policy)과 플래너(planner) 간의 행동 정렬(behavior alignment) 과정을 통해 학습을 강화합니다.

- **Technical Details**: BOOM의 핵심은 정책을 부트스트랩하는 비모수(non-parametric) 행동 분포를 사용하는 likelihood-free alignment loss입니다. 이 논문에서는 고급의 가치 가중화 메커니즘을 도입하여 높은 반환 값을 가진 행동을 우선시하고, 과거 플래너의 행동과 정책 간의 정렬을 통해 학습 효율성을 유지합니다. BOOM은 또한 DeepMind Control Suite와 Humanoid-Bench에서의 고차원 연속 제어 문제에서 최첨단 성능을 보입니다.

- **Performance Highlights**: 실험 결과 BOOM은 기존의 모델 프리(Off-policy) 방법에 비해 뛰어난 훈련 안정성과 최종 성과를 달성하였습니다. 이 연구는 고차원 환경에서 액터 다이버전스의 문제를 해결하며, 지속적인 정책 개선을 통해 강화 학습의 효율성을 극대화합니다. BOOM의 코드는 연구 커뮤니티에서 접근 가능한 형태로 제공됩니다.



### MedRECT: A Medical Reasoning Benchmark for Error Correction in Clinical Texts (https://arxiv.org/abs/2511.00421)
- **What's New**: 이 논문에서는 MedRECT라는 새로운 크로스링궐 벤치마크를 도입하여 일본어와 영어의 의료 오류 탐지를 체계적으로 평가합니다. MedRECT는 오류 탐지, 오류 위치 파악 (sentence extraction), 오류 수정을 포함하는 세 가지 하위 작업으로 의료 오류 처리를 구성합니다. 이 벤치마크는 일본 의료 면허 시험인 JMLE에서 자동화된 파이프라인을 통해 구축되어, 기존 벤치마크의 한계를 극복한 고품질의 평가 도구로 작용합니다.

- **Technical Details**: MedRECT는 663개의 일본어 텍스트와 458개의 영어 텍스트를 포함하며, 두 언어 간의 오류/무오류 균형을 유사하게 유지합니다. 9가지 최신 LLM을 평가한 결과, reasoning 모델이 표준 아키텍처보다 우수한 성능을 보였으며, 일본어에서 영어로의 성능 차이를 확인했습니다. 논문에서는 또한 LoRA를 활용한 타겟팅된 파인 튜닝이 각 언어에서 오류 수정 성능을 비대칭적으로 향상시켰음을 보여줍니다.

- **Performance Highlights**: 연구 결과, reasoning 모델이 오류 탐지에서 최대 13.5%, 문장 추출에서 51.0%의 상대적 향상을 보여주었고, 파인 튜닝된 모델은 구조화된 의료 오류 수정 작업에서 인간 전문가의 성능을 초과했습니다. MedRECT는 의료 AI 시스템의 안정성과 신뢰성을 높이는 데 기여할 수 있는 중요한 자원으로, 의료 분야의 크로스링궐 평가의 필요성을 강조합니다.



### LGCA: Enhancing Semantic Representation via Progressive Expansion (https://arxiv.org/abs/2511.00419)
Comments:
          15 pages, 5 figures, to appear in SoICT 2025

- **What's New**: 최근 큰 규모의 사전 학습이 진행된 자연어 처리(NLP) 분야에서는 CLIP과 같은 사전 훈련된 비전-언어 모델이 이미지와 텍스트를 효과적으로 정렬할 수 있는 능력을 보여주었습니다. 이러한 모델들은 제로샷 이미지 분류(zero-shot image classification) 작업에서 성과를 크게 향상시켰습니다. 그러나 CLIP의 성능은 프롬프트(prompt) 표현에 민감하여, 세밀한 조정이 필요하다는 제한이 있습니다.

- **Technical Details**: 이 논문에서는 비디오 데이터의 세부 기능을 포착한 후 가장 두드러진 지역을 반복적으로 선택하고 확장하는 Localized-Globalized Cross-Alignment (LGCA) 프레임워크를 제안합니다. LGCA는 원래 이미지와 확장된 이미지를 포괄하는 유사성 점수를 통해 로컬(local) 및 글로벌(global) 기능을 모두 담아낼 수 있게 설계되었습니다. 이 방법은 잘못된 정보와 편향을 최소화하며, 기존의 비확장 모델과 유사한 시간 복잡성을 유지합니다.

- **Performance Highlights**: 다양한 데이터셋에서 수행된 실험 결과, LGCA는 기존의 최첨단 기준을 능가하며 제로샷 성능을 크게 향상시켰습니다. 앞서 언급한 LGCA의 장점은 비전-언어 모델이 특정 도메인에 맞춰 전이될 수 있도록 하면서도, 전반적인 효율성과 확장성을 유지한다는 점에서 중요한 이정표가 됩니다.



### Human-AI Programming Role Optimization: Developing a Personality-Driven Self-Determination Framework (https://arxiv.org/abs/2511.00417)
Comments:
          PhD Dissertation, Prague University of Economics and Business, 2025. 323 pages. ACM CCS 2012: Human-computer interaction, Collaborative interaction, Human-AI collaborative systems, Pair programming, AI-assisted software engineering

- **What's New**: 이 연구는 인공지능이 소프트웨어 개발에 미치는 영향을 조명하며, 개발자와 AI 시스템 간의 협력 방안을 최적화하기 위한 Role Optimization Motivation Alignment (ROMA) 프레임워크를 도입합니다. 이를 통해 인격 심리학(personality psychology)과 자기 결정 이론(self-determination theory)을 결합하여 역할 최적화를 실증적으로 검증하고 있습니다.

- **Technical Details**: 본 연구는 다섯 차례의 Design Science Research를 통해 진행되었으며, 200명의 실험 참가자와 46명의 인터뷰 응답자와의 참여를 통해 데이터 수집이 이루어졌습니다. 연구 결과는 인격 특성과 프로그래밍 역할 선호도, 협력 결과 간의 상관관계를 명확히 세웠습니다.

- **Performance Highlights**: 주요 발견은 인격 기반 역할 최적화가 자기 결정(self-determination) 및 팀 역학에 긍정적인 영향을 미치며, 전문가의 평균 동기 부여가 23%, 학부생의 경우 65%까지 증가함을 보여줍니다. 또한 탐험가, 조율자, 장인, 설계자, 적응자라는 다섯 가지 인격 아키타입이 각기 다른 프로그래밍 역할에 대한 선호도를 가지고 있음을 밝힙니다.



### PADBen: A Comprehensive Benchmark for Evaluating AI Text Detectors Against Paraphrase Attacks (https://arxiv.org/abs/2511.00416)
- **What's New**: 본 연구는 AI 생성 텍스트(AIGT) 탐지기들이 LLM 출력에는 90% 이상의 정확도를 보이나, 반복적으로 패러프레이즈된 콘텐츠에는 치명적인 실패를 보인다는 점을 다룹니다. 반복 패러프레이징은 내용의 의미 이전을 발생시키며, 이러한 방식이 기존 탐지 시스템을 회피하는 이유를 분석합니다. 이 과정에서 저자는 PADBen이라는 벤치마크를 제안하여 두 가지 공격 범주(저자 모호화 및 표절 회피)를 평가하고 있습니다.

- **Technical Details**: 이 논문은 패러프레이즈 공격이 AIGT 탐지 시스템에서 효과적으로 회피되는 이유를 규명하기 위해 두 가지 가설을 세우고 이를 실험하였습니다. 실험 결과, 반복적으로 패러프레이즈된 텍스트는 고유한 의미 변환을 제공하며, 이는 탐지 시스템의 시각적 공간에서 나타나는 특징적 패턴과 다르게 작용합니다. PADBen 벤치마크는 원본 콘텐츠부터 깊게 세탁된 텍스트에 이르는 다섯 가지 텍스트 분류와 이에 대한 탐지 작업을 제안합니다.

- **Performance Highlights**: 11개의 최신 탐지기를 평가한 결과, 패러프레이즈 공격이 탐지 시스템을 보편적으로 무너뜨리지 않음을 보여주었습니다. 결과는 텍스트의 출처에 따라 달라지는 비대칭성을 드러내며, 저자 모호화의 경우 성능 저하가 두드러지는 반면 표절 회피 문제는 탐지가 원활하게 이루어지는 것으로 나타났습니다. 현재의 탐지 접근 방식은 중간 세탁 영역에서 효과적으로 대처하지 못함을 보여주며, 탐지 아키텍처의 근본적인 발전이 필요함을 시사합니다.



### Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling (https://arxiv.org/abs/2511.00411)
Comments:
          accepted by iccv 2025

- **What's New**: 이 논문에서는 Gradient-Guided Sampling (GGS)이라는 새로운 샘플링 기법을 제안하여, 적대적 공격의 전송성을 높이는 데 중점을 둡니다. GGS는 과거의 내부 반복(iteration)에서 얻은 그래디언트를 사용하여 샘플링 방향을 안내함으로써 탐색(Exploration)과 활용(Exploitation)의 균형을 맞추고, 전반적인 샘플링 효율성과 안정성을 향상시킵니다. 그 결과, 본 연구는 최신 전송 공격 기법보다 더 나은 성능을 보여줍니다.

- **Technical Details**: GGS는 MI-FGSM 기법에 기반하여 내부 반복 샘플링(inner-iteration sampling)을 도입하며, 그래디언트 상승 방향을 따라 샘플링을 가이드합니다. 이 방법은 고른 지역에서의 적대적 예시 생성에 도움을 주며, 부드러운 손실 표면(flat loss surface)과 높은 로컬 최대(local maxima)를 타겟으로 둡니다. 또한, 우리의 GGS는 기존의 랜덤 샘플링(Random Sampling) 기반 메서드와 호환되어 샘플링 효율성을 더할 수 있습니다.

- **Performance Highlights**: 종합적인 실험 결과는 다양한 DNN 아키텍처와 다중 모달 대형 언어 모델(MLLMs)에서 GGS의 우수성을 입증합니다. 실험을 통해 GGS는 목표 모델(target model)과 비목표 모델(non-target model)에서의 공격 성공률을 높이며, 특히 블랙박스 설정에서 효과적인 발전을 보여주었습니다. 이러한 성과들은 업무 환경에서 적대적 공격을 막기 위한 방어 메커니즘 개발에 기여할 것으로 기대됩니다.



### Quantum Machine Unlearning: Foundations, Mechanisms, and Taxonomy (https://arxiv.org/abs/2511.00406)
- **What's New**: 이번 논문은 양자 기계 학습(QML)과 데이터의 삭제 권리가 새로운 차원으로 진화하는 과정에서 양자 기계 비학습(QMU)의 기초에 대한 포괄적인 프레임워크를 제시합니다. 저자들은 데이터 제거를 양자 불가역성의 물리적 원리로 정립하면서, 모델의 삭제 개념을 단순한 롤백이 아닌 물리적으로 유효한 변환으로 정의합니다. 이 연구는 기존의 이론적 개념과 구현 가능한 전략을 연결하는 다섯 가지 축의 분류법을 제시합니다.

- **Technical Details**: 양자정보 이론에 기반한 이 논문은 선택된 훈련 데이터의 영향을 제거하려는 머신 비학습(MU)의 목표를 다룹니다. 저자들은 최소한의 계산 비용과 명확한 증거 확보를 보장하는 효율성, 완전성, 검증 가능성을 세 가지 공통 목표로 설정하고 있습니다. 또한, 분산 학습에서 각 클라이언트가 제거된 데이터에 대해 자신의 기여도를 조정하는 방식을 통해 비학습 원칙을 실현합니다.

- **Performance Highlights**: 연구는 양자 연합 학습(QFL) 및 양자 차별 개인 정보 보호(QDP)을 통해 성능과 프라이버시를 동시에 증진할 수 있는 가능성을 보여줍니다. 예를 들어, 논문에서는 NISQ 하드웨어를 이용한 QFL–QDP 파이프라인에서 98% 이상의 정확도를 달성하는 사례를 소개하며, 프라이버시 보호 프로토콜이 성과를 저해하지 않고 향상시킬 수 있음을 시사합니다. 이 연구는 양자 클라우드 인프라에서 깔끔한 삭제를 보장하기 위한 기술적 접근 방법도 다루고 있습니다.



### UME-R1: Exploring Reasoning-Driven Generative Multimodal Embeddings (https://arxiv.org/abs/2511.00405)
- **What's New**: 이번 연구에서는 멀티모달 대형 언어 모델(MLLMs)의 성과를 기반으로 한 멀티모달 임베딩의 진전을 다룹니다. UME-R1이라는 새로운 다목적 멀티모달 임베딩 프레임워크가 소개되었으며, 이는 생성적 임베딩과 판별적 임베딩을 모두 생성할 수 있도록 설계되었습니다. 특히, 이 모델은 두 단계의 훈련 전략을 통해 생성적 추론(capabilities) 및 임베딩 품질(optimize)을 향상시키는데 중점을 두고 있습니다.

- **Technical Details**: UME-R1은 첫 번째로 콜드 스타트(CS) 감독 미세 조정(supervised fine-tuning) 데이터를 통해 추론과 요약을 포함한 데이터를 생성합니다. 이 과정에서 대조 손실(contrastive loss)과 자율 회귀적 다음 토큰 예측 손실(autoregressive next-token prediction loss)을 적용하여 모델이 생성적 임베딩과 판별적 임베딩을 다양하게 생성하도록 훈련합니다. 실험에서는 강화 학습 기반과 검증 가능한 보상(RLVR) 접근 방식이 생성적 임베딩 품질을 높이는 효과를 보였습니다.

- **Performance Highlights**: MMEB-V2 벤치마크에서 78개 작업을 대상으로 평가한 결과, UME-R1은 기존의 판별적 임베딩 모델보다 유의미한 성능 향상을 보였습니다. 특히, 생성적 임베딩은 전통적인 판별적 임베딩보다 더 나은 결과를 보여주며 두 가지 임베딩의 조합이 유용하다는 점을 강조합니다. 반복 샘플링을 통한 추론 시간 확장 가능성도 확인되었으며, 이는 다운스트림 작업에서의 효과를 개선하는 데 기여합니다.



### Emotion Detection in Speech Using Lightweight and Transformer-Based Models: A Comparative and Ablation Study (https://arxiv.org/abs/2511.00402)
- **What's New**: 이번 연구에서는 CREMA-D 데이터셋을 이용해 6가지 감정을 분류하는 과정에서 DistilHuBERT와 PaSST라는 경량 트랜스포머 모델의 성능을 비교 분석하였습니다. 이 모델들은 전통적인 CNN-LSTM 모델과 비교하여 우수성을 보이며, 가장 작은 모델 크기인 DistilHuBERT는 70.64%의 정확도를 달성했습니다. 이 연구는 또한 PaSST의 다양한 구성 요소가 성능에 미치는 영향을 조사하여, MLP 헤드 구조가 가장 우수한 성능을 발휘함을 확인했습니다.

- **Technical Details**: 이 연구의 주요 기술적 요소는 DistilHuBERT와 PaSST 모델의 비교 분석입니다. DistilHuBERT는 최소한의 계산 비용으로 높은 정확도를 제공하는 자가 감독 모델이며, PaSST는 스펙트로그램 패칭 및 패치아웃 기법을 적용한 효율적인 오디오 분류 모델입니다. 연구는 또한 세 가지 PaSST 변형(Linear, MLP, Attentive Pooling)으로 아블레이션 연구를 수행하였습니다.

- **Performance Highlights**: 성능 하이라이트로, DistilHuBERT는 70.64%의 정확도와 70.36%의 F1 점수를 기록하며, 크기 0.02 MB로 뛰어난 효율성을 자랑합니다. 반면, PaSST는 MLP 헤드 구조에서 최선의 성능을 보이지만 DistilHuBERT에는 미치지 못하는 것으로 나타났습니다. 여러 감정 클래스 중에서 'anger'는 가장 정확하게 감지되었고, 'disgust'는 여전히 가장 어려운 감정으로 남아있습니다.



### SonarSweep: Fusing Sonar and Vision for Robust 3D Reconstruction via Plane Sweeping (https://arxiv.org/abs/2511.00392)
Comments:
          8 pages, 9 figures, conference

- **What's New**: 이 논문에서는 SonarSweep이라는 새로운 엔드투엔드 딥러닝 프레임워크를 소개합니다. 이 프레임워크는 시각적 데이터와 소나 데이터를 교차 모드 융합하여 정확하고 밀집된 3D 복원을 가능하게 합니다. 기존 방법들의 한계를 극복하기 위해 클래식한 평면 스윕 알고리즘을 적응시켜, 잠재적인 깊이 가설에 걸쳐 소나 특징을 카메라의 기준 틀에 왜곡해 복원합니다.

- **Technical Details**: SonarSweep은 비전 기반 및 소나 데이터를 통합하여 고밀도 깊이 맵을 생성합니다. 본 방법은 반응하는 다중 모드 비용(volume) 공간을 구성하여 각 잠재적 깊이에 대한 특징 유사성을 인코딩합니다. 그렇게 함으로써, 기존 단일 모드 접근 방식들이 정밀하지 못하게 해결한 고유한 기하학적 불확실성을 줄이고, 정확하고 일관된 깊이 맵을 생성합니다.

- **Performance Highlights**: SonarSweep은 다양한 고난도의 수중 환경에서 기존 최첨단 방법들보다 뛰어난 성능을 나타냅니다. 실험 결과, 특히 높은 탁도에서 일관되게 밀집되고 정확한 깊이 맵을 생성하였으며, 이 연구를 위한 코드와 처음으로 동기화된 스테레오 카메라 및 소나 데이터 세트를 공개할 계획입니다.



### Who Can We Trust? Scope-Aware Video Moment Retrieval with Multi-Agent Conflic (https://arxiv.org/abs/2511.00370)
- **What's New**: 이 연구에서는 비디오 순간 검색(video moment retrieval) 작업을 위한 새로운 강화 학습 기반 모델을 제안합니다. 이 모델은 전체 비디오를 한 번 스캔하여 순간의 경계(boundary)를 찾고 이를 위한 위치 증거를 생성합니다. 또한, 여러 에이전트(agent) 간의 갈등을 해소하기 위한 다중 에이전트 시스템 프레임워크를 도입하여, 쿼리가 비디오에서 해당 순간을 갖지 않는 경우를 감지할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: 제안된 모델은 증거 기반 학습(evidential learning)을 사용하여 각 에이전트가 추정한 경계의 신뢰도를 평가합니다. 이 모델은 고정된 창 크기로 비디오 전체를 스캔하고, 각 에이전트는 독립적으로 작동하여 최종 위치에서의 갈등을 평가합니다. 이를 통해 OOS(Out-of-scope) 쿼리를 제로샷(Zero-shot) 방식으로 감지할 수 있는 기능도 제공합니다.

- **Performance Highlights**: 실험 결과, 이 방법은 최신 기법(state-of-the-art approaches)과 비교하여 성능이 향상된 것으로 나타났습니다. 다중 에이전트 시스템의 경쟁 및 갈등 모델링은 비디오 순간 검색의 강화 학습 성능을 개선하는 효과적인 방법이라는 것을 보여주었습니다. 연구 결과는 영상 레벨 검색 애플리케이션 개발에 도움이 될 것으로 기대됩니다.



### Balancing Interpretability and Performance in Motor Imagery EEG Classification: A Comparative Study of ANFIS-FBCSP-PSO and EEGN (https://arxiv.org/abs/2511.00369)
Comments:
          6 pages, 3 figures, 8 tables, Submitted to ICECTE 2026

- **What's New**: 이 연구에서는 모터 이미지를 기반으로 한 뇌-컴퓨터 인터페이스(BCI)에서 해석 가능성과 성능 간의 trade-off를 체계적으로 분석합니다. 해석 가능한 ANFIS-FBCSP-PSO 파이프라인과 심층 학습 기법인 EEGNet을 비교하는 두 모델을 구성하였습니다. 이 연구는 BCI Competiton IV-2a 데이터셋을 사용하여 두 모델의 상대적인 장점을 평가합니다. 또한, 사용자가 선택할 수 있도록 해석 가능성 또는 강인성을 목표로 하는 MI-BCI 시스템에 대한 실용적인 지침을 제공합니다.

- **Technical Details**: BCI는 두 가지 주요 기술 접근 방식을 이용하여 MI EEG 신호를 분류합니다. ANFIS는 필터 뱅크 공통 공간 패턴(feature extraction)과 입자 군집 최적화(Particle Swarm Optimization)를 결합하여 해석 가능한 규칙(IF-THEN rules)을 생성합니다. 반면 EEGNet은 원시 EEG 데이터에서 계층적 공간-시간 표현을 직접 학습하여 복잡한 비선형 의존성을 잡아내는 데 초점을 맞춘 심층 신경망 모델입니다.

- **Performance Highlights**: 연구 결과에 따르면, ANFIS-FBCSP-PSO 모델은 내부 주제 실험에서 68.58%의 정확도를 기록했습니다. 반면에, 교차 주제(Leave-One-Subject-Out) 테스트에서는 EEGNet이 68.20%의 정확도로 더 나은 일반화 성능을 보였습니다. 이는 해석 가능성을 추구하는 야무진 AI 모델과 성능이 우수한 심층 네트워크 간의 상충 관계를 잘 보여줍니다. 이러한 성과는 두 모델 모두 MI-EEG 분류에 대한 다양한 성능 측정 기준에서의 효율성을 증명합니다.



### Oitijjo-3D: Generative AI Framework for Rapid 3D Heritage Reconstruction from Street View Imagery (https://arxiv.org/abs/2511.00362)
Comments:
          6 Pages, 4 figures, 2 Tables, Submitted to ICECTE 2026

- **What's New**: 방글라데시의 문화유산 복원은 자원 부족과 기술 전문성 부족이라는 두 가지 문제에 직면해 있습니다. Oitijjo-3D는 비용이 들지 않는 generative AI 프레임워크로, Google Street View 이미지를 활용하여 3D 모델을 신속하게 재구성합니다. 이 시스템은 전통적인 3D 디지털화 방법을 대체하며, 비용과 기술 장벽을 크게 낮춰 문화유산 보존을 더 많은 사람들이 할 수 있도록 합니다.

- **Technical Details**: Oitijjo-3D는 두 단계의 파이프라인을 통해 작동합니다: Gemini 2.5 Flash Image를 사용한 구조-텍스처 합성 및 Hexagen을 통한 신경 이미지-3D 생성입니다. 이러한 방법은 고급 하드웨어나 전문가 감독 없이도 사진 실사와 일관된 3D 재구성을 가능하게 합니다. 최종적으로 생성된 모델은 glTF 2.0 형식으로 출력되어 다양한 형식으로 사용자에게 제공됩니다.

- **Performance Highlights**: Oitijjo-3D는 기존의 SfM+MVS 방법에 비해 250배 이상의 속도 향상을 달성했습니다. 일반적으로 4-8시간 걸리는 이미지 처리 과정이 45초 이내에 완료되며, 필요한 메모리 또한 최소화됩니다. 이 성능은 자금이나 전문 인력이 부족한 개발도상국의 환경에서 접근 가능한 문화유산의 디지털화를 촉진할 수 있습니다.



### MalDataGen: A Modular Framework for Synthetic Tabular Data Generation in Malware Detection (https://arxiv.org/abs/2511.00361)
Comments:
          10 pages, 6 figures, 2 tables. Published at the Brazilian Symposium on Cybersecurity (SBSeg 2025)

- **What's New**: MalDataGen은 데이터 부족으로 인한 악성코드 탐지의 한계를 극복하기 위해 개발된 오픈 소스의 모듈형 프레임워크입니다. 이 프레임워크는 WGAN-GP와 VQ-VAE와 같은 모듈형 딥 러닝 모델을 활용하여 고품질의 합성 데이터(tabular data)를 생성합니다. 두 가지 검증 방식(TR-TS, TS-TR)과 일곱 가지 분류기를 통한 평가를 통해, MalDataGen은 SDV와 같은 기존 기준을 초월하면서 데이터 유틸리티를 유지합니다.

- **Technical Details**: MalDataGen은 깊이 있는 학습 기반의 생성 모델을 개발하고 관리하기 위한 엔진과 합성 데이터 품질을 검증하기 위한 평가 리소스로 구성됩니다. 엔진은 데이터 입출력(DataIO), 데이터 시각화, 분류기, 메트릭, 활성 모니터링 등 여섯 개의 주요 모듈로 나누어집니다. 또한, 합성 모델 모듈에서는 CTGAN, 변분 오토인코더(VAE), 가우시안 코퓰라 모델 등 다양한 생성 데이터 생성기를 제공합니다.

- **Performance Highlights**: 실험 결과, MalDataGen은 SVM과 같은 여러 분류기를 사용하여 우수한 성능을 보였습니다. 이 연구는 방대한 Androcrawl 데이터셋을 활용하여 10,170개의 악성코드와 10,170개의 정상 데이터를 포함하며, 136개의 특징으로 구성되어 있습니다. 각 분류기와 하이퍼파라미터 설정에 대해 구체적인 결과는 공개 리포지토리에서 확인할 수 있습니다.



### Mind the Gap: Missing Cyber Threat Coverage in NIDS Datasets for the Energy Sector (https://arxiv.org/abs/2511.00360)
Comments:
          13 pages

- **What's New**: 이번 연구에서는 전통적인 기업 환경에 초점을 맞춘 공개 데이터셋 기반의 네트워크 침입 탐지 시스템(Network Intrusion Detection Systems, NIDS)의 한계를 분석했습니다. 특히, 에너지 인프라에서 Information Technology (IT)와 Operational Technology (OT)의 융합된 환경에서의 효과성에 대한 우려를 제기하면서, 이러한 데이터셋의 대표성을 평가했습니다.

- **Technical Details**: 연구는 CIC-IDS2017, SWaT, WADI, Sherlock, 및 CIC-Modbus2023 데이터셋을 이용하여 문서화된 에너지 부문 사건에서 추출된 MITRE ATT&CK 기술들을 네트워크에서 탐지할 수 있는지의 여부를 분석하는 데 구조적 5단계 분석 접근 방법을 사용했습니다. 초기에 274개의 ATT&CK 기술 풀이 있는 가운데, 총 94개의 네트워크 탐지 가능한 기술이 도출되었습니다.

- **Performance Highlights**: Sherlock 데이터셋은 평균 커버리지(coverage)가 0.56으로 가장 높았고, CIC-IDS2017이 0.55로 뒤를 이었습니다. 반면, SWaT와 WADI는 각각 0.38로 가장 낮은 점수를 기록했습니다. CIC-IDS2017, Sherlock, 그리고 CIC-Modbus2023 데이터셋을 조합하면 92%의 집합 커버리지를 달성하였으며, 이는 데이터셋 강화와 하이브리드 IT/OT 에너지 환경에서의 NIDS 평가를 위한 명확한 경로를 제시합니다.



### Toward Unifying Group Fairness Evaluation from a Sparsity Perspectiv (https://arxiv.org/abs/2511.00359)
Comments:
          30 pages, 14 figures

- **What's New**: 이 논문은 알고리즘 공정성 알고리즘(equity)이 머신러닝에 있어 중요한 과제로 인식되며, 새로운 스파시티(sparsity) 기반의 통합 프레임워크를 제안합니다. 제안된 프레임워크는 기존의 공정성 기준에 부합하며 다양한 머신러닝 작업에 적용할 수 있는 가능성을 보여줍니다. 이러한 접근은 스파시티와 사회적 형평성(social equity) 시각에서 알고리즘 공정성을 재조명하며, 공정성 연구와 적용에 대한 폭넓은 영향을 미칠 잠재력을 제공합니다.

- **Technical Details**: 본 연구는 다양한 스파시티 측정 방법과 이를 기반으로 한 알고리즘 공정성 평가 간의 연결고리를 살펴봅니다. PQ Index와 Maximum Pairwise Difference(MPD) 간의 이론적 연결성을 강조하며, 이를 통해 다중 집단 및 회귀 문제에 적합한 공정성 평가 프레임워크를 제안합니다. 특히, 스파시티 측정에서의 새롭고 통합된 접근 방식을 통해 기존의 공정성 기준들과의 일치를 보여줍니다.

- **Performance Highlights**: 실험을 통해 제안된 프레임워크의 유효성을 다수의 데이터 세트와 편향 완화 방법에 대해 검증하였습니다. 제안한 지표는 기존 기준과의 일치를 통해 효과성을 나타내며, 교차 공정성(intersectional fairness) 환경에서의 포괄적인 적용 가능성도 분석되었습니다. 이 통합 프레임워크는 많은 머신러닝 문제에 적용될 수 있는 획기적인 가능성을 제공합니다.



### Detecting AI-Generated Images via Diffusion Snap-Back Reconstruction: A Forensic Approach (https://arxiv.org/abs/2511.00352)
Comments:
          6 pages, 8 figures, 4 Tables, submitted to ICECTE 2026

- **What's New**: 최근의 생성적 확산 모델(generative diffusion models)의 급격한 발전으로 인해, 진짜 시각 콘텐츠와 합성 이미지(synthetic imagery)를 구별하는 것이 점점 더 어려워지고 있습니다. 안정적 확산 모델(Stable Diffusion)과 DALL-E와 같은 최신 텍스트-이미지 시스템은 포토리얼리즘(photorealism)과 아티팩트 없는 결과를 생성하여 전통적인 딥페이크 탐지 방법이 실패하게 만듭니다. 본 연구에서는 다중 강도 이미지 재구성 동역학(multi-strength image reconstruction dynamics)을 활용한 확산 기반 포렌식 프레임워크를 소개합니다.

- **Technical Details**: 이 프레임워크는 재구성 메트릭(LPIPS, SSIM, PSNR)의 변화를 분석하여 실제 이미지와 AI 생성 이미지를 구별할 수 있는 해석 가능한 매니폴드 기반 특징을 추출합니다. 연구는 4,000개의 이미지를 포함하는 균형 잡힌 데이터셋에서 평가되었으며 교차 검증(cross-validation)에서 0.993 AUROC를 달성했습니다. 제한된 데이터와 단일 확산 백본(Stable Diffusion v1.5)을 사용했음에도 불구하고, 제안된 방법은 강력한 일반화와 해석 가능성을 입증합니다.

- **Performance Highlights**: 제안된 방법은 압축(compression) 및 노이즈(noise)와 같은 일반적인 왜곡에 대해서도 강건성을 유지하며, AI가 생성한 미디어에 대한 포렌식 분석을 위한 기초를 제공합니다. 독립적인 검증 플랫폼(public verification platform)을 통해 사용자는 이미지나 비디오를 업로드하여 AI로 생성되었는지 여부를 확인할 수 있는 가능성도 제시합니다. 이는 정보가 왜곡되고 있는 현대 사회에서 특히 가치 있는 접근 방식으로 평가됩니다.



### Exploiting Latent Space Discontinuities for Building Universal LLM Jailbreaks and Data Extraction Attacks (https://arxiv.org/abs/2511.00346)
Comments:
          10 pages, 5 figures, 4 tables, Published at the Brazilian Symposium on Cybersecurity (SBSeg 2025)

- **What's New**: 이 논문은 Лarge Language Models(LLMs)의 보안 위협에 대한 새로운 접근 방식을 제시하고 있습니다. 특히, latent space discontinuities라는 아키텍처 취약점을 활용하여 범용 jailbreak 및 데이터 추출 공격을 설계하였습니다. 기존의 방법과 달리, 이 기법은 다양한 모델과 인터페이스에 걸쳐 일반화되며, 7개의 최첨단 LLM과 1개의 이미지 생성 모델에서 높은 효과를 입증하였습니다.

- **Technical Details**: 논문에서는 alignment degradation induction 기술을 개발하여 모델의 행동 불안정성을 연구하고, 잘못 조정된 영역으로의 추론 경로를 유도하여 모델의 약점을 탐구합니다. 이 과정에서 semantic shifts, echo suppression, Token Shield, adversarial noise와 같은 다양한 적대적 구성 요소들을 도입하여 모델의 견고성을 테스트합니다. 이러한 기술들은 모델이 우발적인 과잉 반응이나 해석실수 없이 적대적 입력을 처리할 수 있도록 돕습니다.

- **Performance Highlights**: 초기 결과는 이러한 discontinuities를 이용하여 모델의 행동을 지속적이고 심각하게 타격할 수 있음을 보여줍니다. 특히, 다양한 보호 인터페이스에 대한 공격의 효과성이 높아지고, 기존의 방어 체계가 있는 상황에서도 심각한 취약점이 발견되었습니다. 연구 결과는 이러한 전략이 시스템 전반에 걸쳐 공격 벡터로서 상당한 잠재력을 지니고 있음을 시사합니다.



### MH-1M: A 1.34 Million-Sample Comprehensive Multi-Feature Android Malware Dataset for Machine Learning, Deep Learning, Large Language Models, and Threat Intelligence Research (https://arxiv.org/abs/2511.00342)
Comments:
          17 pages, 7 figures, 13 tables, submitted to the Scientific Data journal published by Nature Research

- **What's New**: MH-1M 데이터셋은 1,340,515개의 Android 애플리케이션으로 구성되어 있으며, 현대의 악성 소프트웨어 연구에 필요한 방대한 메타데이터를 제공합니다. 이 데이터셋은 VirusTotal API를 활용하여 여러 탐지 엔진의 결과를 통합하여 정확한 악성 소프트웨어 분류를 보장합니다. 또한, GitHub, Figshare, Harvard Dataverse에서 공개에 접근할 수 있어 연구자들에게 매우 유용합니다.

- **Technical Details**: MH-1M 데이터셋은 14년 동안 수집된 22,810개의 속성과 더불어 안드로이드 애플리케이션 패키지(APK)를 포함하고 있습니다. 이 데이터는 22,394개의 API 호출, 407개의 인텐트, 232개의 오프코드, 214개의 권한을 포함한 상세한 Android 기능을 제공하여, 심층적이고 맥락 인식적인 악성 소프트웨어 탐지에 적용할 수 있는 가능성을 확장합니다. 이 데이터셋은 400 GB 이상에 달하며, 압축 형식으로 Harvard Dataverse에서 제공됩니다.

- **Performance Highlights**: 기존 데이터셋과 비교할 때, MH-1M은 1.34 백만 개 이상의 샘플을 포함하여 데이터의 스케일에서 독보적인 장점을 가지고 있습니다. 기존 데이터셋들이 좁은 특성 범위를 갖고 있거나 제한된 악성 소프트웨어 분류를 사용하는 반면, MH-1M은 다양하고 포괄적인 특성 커버리지를 제공합니다. 이러한 특성 덕분에 연구자들은 더 상세하고 정밀한 애플리케이션 행동 분석을 수행할 수 있으며, 실제 데이터 분포를 반영한 비율로 과적합(overfitting)을 방지할 수 있습니다.



### Towards Automated Petrography (https://arxiv.org/abs/2511.00328)
- **What's New**: 이 논문은 자동 석유 및 광물 분석을 위한 대규모 실험 프레임워크인 LITHOS를 소개합니다. LITHOS는 211,604개의 고해상도 RGB 패치와 25개 광물 범주에 걸쳐 105,802개의 전문가 주석을 포함하여, 기존 데이터 세트보다 두 배나 많은 데이터를 제공합니다. 이 데이터베이스는 연구자들이 석유 및 광물 분석 자동화를 위한 더 나은 방법을 개발할 수 있도록 지원할 것입니다.

- **Technical Details**: LITHOS 데이터 세트는 PPL(평면 편광) 및 XPL(교차 편광) 조건에서 캡처된 고해상도 이미지 패치로 구성되어 있습니다. 각각의 광물 결정은 전문가가 정의한 주석, 즉 광물 클래스, 공간 좌표, 대 및 소 축의 교차 벡터 경로를 포함하고 있어, 결정의 기하학과 방향을 잘 나타냅니다. 데이터 세트는 실제 석유 및 광물 분석의 복잡성을 반영하고 있으며, 다양한 광물 범주에서 나타나는 유사성을 도해합니다.

- **Performance Highlights**: LITHOS 베이스라인은 쌍편광 촬영기법을 활용한 변환기 아키텍처로, 기존의 단일 편광 모델보다 모든 지표에서 지속적으로 우수한 성능을 보입니다. 이를 통해 편광 정보의 융합이 석유 및 광물 분류에서 기여하는 유용성을 극대화합니다. 모든 데이터 및 사전 훈련된 모델은 CC BY-NC-SA 4.0 라이선스 하에 공개되며, 재현성 및 투명성을 증진하고, 향후 연구에 기여할 것입니다.



### Scalable Processing-Near-Memory for 1M-Token LLM Inference: CXL-Enabled KV-Cache Management Beyond GPU Limits (https://arxiv.org/abs/2511.00321)
- **What's New**: 최근 대형 언어 모델(LLM)에서 문맥 윈도우의 확장이 이루어졌습니다. 그러나 이로 인해 Key-Value (KV) 캐시 관리에서 메모리와 컴퓨팅 리소스의 병목 현상이 발생하고 있습니다. 본 논문에서는 CXL(Compute Express Link)을 활용하여 PNM(Processing-Near-Memory) 기반 KV 캐시 관리 시스템을 제안하였습니다. 이를 통해 GPU 제한을 넘어 메모리 및 계산을 조정할 수 있습니다.

- **Technical Details**: 제안된 시스템은 CXL 메모리 내의 PNM 가속기를 사용하여 토큰 페이지 선택을 오프로드하며, 그로써 GPU 메모리 압박를 완화하고 더 큰 배치 크기를 지원합니다. 데이터 병렬성(DP) 및 텐서 병렬성(TP) 조합을 통해 멀티-PNM 배치 처리를 최적화하여 계산 효율성을 극대화합니다. 또한, 동시에 GPU 작동을 최적화하기 위해 steady-token 선택 알고리즘을 도입하여 지속적으로 관련성이 높은 토큰을 식별합니다.

- **Performance Highlights**: 실험 결과, PNM 기반의 KV 오프로드 시스템(PNM-KV)과 GPU-PNM 하이브리드(PnG-KV)는 기존 시스템 대비 최대 21.9배의 처리량 개선을 달성했습니다. 또한, 토큰당 에너지가 최대 60배 절감되고, 전체 비용 효율성은 최대 7.3배 향상되었습니다. 이러한 성과는 CXL 기반의 다중 PNM 아키텍처가 향후 긴 문맥 LLM 추론의 확장 가능성을 입증함을 보여줍니다.



### A Technical Exploration of Causal Inference with Hybrid LLM Synthetic Data (https://arxiv.org/abs/2511.00318)
Comments:
          9 pages, 4 figures

- **What's New**: 이번 연구는 Large Language Models(LLMs)를 사용하여 합성 데이터 생성의 새로운 접근 방식을 제안합니다. 기존 모델들이 평균 처리 효과(Average Treatment Effect, ATE)를 보존하는 데 실패하는 경우가 많다는 점을 강조하며, 더욱 정교한 합성 데이터 프레임워크를 통해 기존 causal 구조를 유지할 수 있도록 설계되었습니다. 제안하는 하이브리드 생성 방법은 모델 기반의 공변량 합성과 개별적으로 학습된 성향 모델 및 결과 모델을 결합하여 causal 분석을 지원하는 LLM 기반 데이터 파이프라인의 기초를 구축합니다.

- **Technical Details**: 연구에서는  GPT-2 모델을 활용하여 GReaT 프레임워크에 따라 합성 테이블 데이터를 생성하는 과정을 설명합니다. 데이터의 각 줄은 자연어 문장을 통해 직렬화되며, 공변량, 처리, 결과는 주어-술어-목적어 형태로 인코딩됩니다. 이 방법론은 각 특성의 의미를 유지하면서도 인과 추론에 필요한 복잡한 의존성을 모델링할 수 있게 합니다. 또한, 군집화와 최인접 기록 분석을 통해 생성된 샘플이 원본 데이터 분포와 일치하는지 확인하고 데이터 오버피팅을 방지하는 방법을 제시합니다.

- **Performance Highlights**: 제안된 하이브리드 데이터 생성 방법은 기존의 생성 모델들이 평균 처리 효과(ATE)를 보존하지 못할 때 발생할 수 있는 문제를 해결하기 위한 것입니다. 이 접근법은 syntactic 정밀도를 유지하면서 causal 관계를 보존하는 데 강점을 보입니다. 또한, 이 연구에서는 기존 causal 추정기들이 무작위 오류가 없는 상태에서 제대로 작동하도록 지원하는 고품질 합성 데이터를 생성하는 방법을 성공적으로 입증하였습니다.



### Language Modeling With Factorization Memory (https://arxiv.org/abs/2511.00315)
- **What's New**: 본 논문에서는 Factorization Memory라는 효율적인 순환 신경망( recurrent neural network, RNN) 아키텍처를 제안합니다. 이 모델은 짧은 문맥의 언어 모델링 작업에서 Transformer 모델과 유사한 성능을 달성하며, 긴 문맥 시나리오에서 더 뛰어난 일반화 능력을 보여줍니다. Factorization Memory는 Mamba-2를 기반으로 구축되어, 훈련 중에는 병렬 계산을 활용하면서 추론 시에는 일정한 계산 및 메모리 복잡성을 유지합니다.

- **Technical Details**: Factorization Memory는 두 가지 전략, 즉 조밀한 업데이트(dense update)와 희소한 업데이트(sparse update)를 통해 메모리 상태를 업데이트합니다. 희소한 업데이트를 사용함으로써 매 시간 단계에서 소수의 매개변수만 선택적으로 업데이트하여 계산 비용을 줄이는 동시에 더 큰 순환 상태를 유지할 수 있습니다. 이 방법은 훈련 및 추론 중 부분 활성화(partial activation)를 통해 계산 및 메모리 절약 효과를 달성합니다.

- **Performance Highlights**: Factorization Memory는 짧은 문맥 작업에서 Transformer 및 Mamba-2와 경쟁할 수 있을 뿐만 아니라, 훈련 문맥 길이를 초과하는 부분에서 우수한 성능을 보여줍니다. 또한, 이 모델은 이들 모델에 비해 더 높은 추론 효율성을 달성하였습니다. 이는 Factorization Memory의 효율성이 모델의 성능에 긍정적인 영향을 미쳤음을 시사합니다.



### Calibration Across Layers: Understanding Calibration Evolution in LLMs (https://arxiv.org/abs/2511.00280)
Comments:
          Accepted at EMNLP 2025 (main)

- **What's New**: 본 논문에서는 대규모 언어 모델(LLMs)의 교정(calibration) 능력을 깊이(depth)에 따라 어떻게 발전하는지를 조사했습니다. 기존 연구에서 깊은 신경망이 과도하게 자신감(confidence)을 갖는 경향이 있다고 보고되었으나, LLM은 다양한 작업에서 잘 조정된 확률을 보여준다고 합니다. 특히, 마지막 층의 구성 요소와 같은 특정 특성과 관련이 있는 이 현상을 분석하여, 모델의 신뢰도가 예측 확률과 잘 맞춰져 있다는 점을 밝히고 있습니다.

- **Technical Details**: 연구팀은 MMLU 벤치마크를 활용하여 여러 개방형 모델을 분석했으며, 상위 레이어에서의 신뢰도 교정(confidence correction phase) 단계를 드러냈습니다. 이 과정에서 모델의 결정 확신이 높아지면, 이후 레이어에서 신뢰도가 능동적으로 조정되는 양상을 확인했습니다. 또한 저차원 교정 방향이 잔여 스트림(residual stream) 내에 존재하며, 이 방향을 교란하면 교정 메트릭(ECE 및 MCE)이 유의미하게 향상됨을 입증했습니다.

- **Performance Highlights**: 본 연구는 LLM의 교정 과정이 모델의 마지막 단계를 넘어서도 이루어지며, 신뢰도가 정확성과 단순히 상관관계가 아니라는 점을 강조하고 있습니다. 연구 결과는 LLM의 임계과정(calibration process)을 보다 진보된 방식으로 이해할 수 있도록 하여, 신뢰도 조정을 위한 새로운 통찰력을 제공합니다. 이 연구는 더 나아가 해석 가능한 언어 모델의 신뢰도 조정 연구를 촉진할 것으로 기대됩니다.



### LongCat-Flash-Omni Technical Repor (https://arxiv.org/abs/2511.00279)
- **What's New**: LongCat-Flash-Omni는 560억 개의 파라미터를 가진 최신 오픈 소스 omni-modal 모델로, 실시간 오디오-비주얼 인터랙션에서 뛰어난 성능을 보여줍니다. 점진적인 학습 전략을 통해 다양한 모달리티(모드)의 시퀀스 모델링 작업을 수행하며, 강력한 unimodal 능력을 유지하면서도 포괄적인 멀티모달 능력을 달성하였습니다.

- **Technical Details**: LongCat-Flash-Omni는 Shortcut-connected Mixture-of-Experts (MoE) 아키텍처를 기반으로 하며, 효율적인 멀티모달 인식 및 음성 재구성 모듈을 결합합니다. 560B 파라미터 중 27B가 활성화된 상태에서도 낮은 대기 시간으로 실시간 오디오-비주얼 인터랙션을 지원합니다. 또한 모달리티 분리 병렬 처리(MDP) 전략을 통해 대규모 멀티모달 훈련의 데이터 및 모델 이질성을 관리합니다.

- **Performance Highlights**: LongCat-Flash-Omni는 Omni-Bench 및 WorldSense와 같은 omni-modal 벤치마크에서 최신 성능을 기록하며, 텍스트, 이미지, 비디오 이해 및 음성 이해와 생성 등 다양한 unimodal 작업에서도 높은 경쟁력을 보여줍니다. 궁극적으로, LongCat-Flash-Omni는 오픈 소스 커뮤니티에서 가장 강력한 omni-modal 모델로 자리잡고 있으며, 고품질의 오디오-비주얼 인터랙션을 지원합니다.



### POSESTITCH-SLT: Linguistically Inspired Pose-Stitching for End-to-End Sign Language Translation (https://arxiv.org/abs/2511.00270)
Comments:
          Accepted at EMNLP 2025 (Main)

- **What's New**: 본 논문에서는 포즈 기반, 글로스(Gloss) 없는 수화 번역(SLT)을 위한 새로운 선행 학습(pre-training) 방식인 POSESTITCH-SLT를 제안합니다. 이는 언어 템플릿을 기반으로 한 문장 생성 기법에서 영감을 받아, 공공 단어 수준 데이터셋을 활용하여 수백만 개의 문장을 생성할 수 있도록 합니다. 이전 연구에 비해 단순한 Transformer 아키텍처를 사용하여, How2Sign와 iSign 데이터셋에서 성능이 향상되었음을 보여줍니다.

- **Technical Details**: POSESTITCH-SLT는 수화 영상에서 추출된 2D 포즈 시퀀스를 기반으로 하여, 해당하는 영어 문장을 생성하는 방법을 논의합니다. 이 방법은 대규모의 구문 구조와 그에 맞는 자연어 데이터를 결합하여 기계 학습 알고리즘의 훈련에 활용합니다. 특히, ASL(미국 수화)과 ISL(인도 수화) 데이터를 사용하여, 공통 어휘를 기반으로 생성된 밀리언 단위의 문장 데이터셋을 만들어, 이는 표준 Transformer 아키텍처로 모델 훈련에 사용됩니다.

- **Performance Highlights**: How2Sign 데이터셋에서 BLEU-4 점수가 1.97에서 4.56으로, iSign 데이터셋에서는 0.55에서 3.43으로 향상되었습니다. 이러한 결과는 기존의 최첨단 기술을 초월하며, pose 기반의 글로스 없는 번역에서의 우수한 성능을 보여줍니다. 이 연구는 저자들의 GitHub 페이지(https://github.com/Exploration-Lab/PoseStich-SLT)를 통해 공개된 데이터셋 및 코드를 활용하여 향후 연구에 기여하고자 합니다.



### FedReplay: A Feature Replay Assisted Federated Transfer Learning Framework for Efficient and Privacy-Preserving Smart Agricultur (https://arxiv.org/abs/2511.00269)
- **What's New**: 이 논문에서는 스마트 농업을 위한 새로운 연합 학습 프레임워크(Federated Learning framework)를 제안합니다. 연합 학습의 개념을 활용해 CLIP(Contrastive Language-Image Pre-training) 모델의 고정된 비전 변환기(vision transformer)와 경량 비전 분류기(classifier)를 통합하여, 비공식(raw) 데이터 공개를 최소화하고 통신 비용을 줄였습니다. 또한, 비독립적 및 동일 분포가 아닌(non-IID) 데이터의 성능 저하를 완화하기 위해 클라이언트 간에 공유하는 경우에 1%의 클립 기반(feature representation) 특징만을 공유하여 개인 정보를 보호합니다.

- **Technical Details**: 제안된 FedReplay 프레임워크는 클립 비전 변환기(CLIP ViT)를 사용하여 강력한 특징을 추출하고, 이러한 특징을 기반으로 경량 변환기 분류기를 학습시킵니다. 이 방식은 대규모 데이터셋으로부터 사전 훈련된 모델을 활용함으로써 전체 모델을 처음부터 학습하는 부담을 피할 수 있습니다. 또한, 전체 파라미터를 전송할 필요 없이 공유된 특징을 활용하여 통신 오버헤드를 약 98% 줄이는 성능을 보였습니다.

- **Performance Highlights**: 농업 데이터 분류 작업에 대한 실험 결과, FedReplay 프레임워크는 86.6%의 정확성을 달성하며 기존 연합 학습 접근법에 비해 4배 이상의 개선을 나타냈습니다. 또한, 통신 효율성을 크게 향상시켜 스마트 농업에서의 효과적이고 확장 가능한 배치를 가능하게 했습니다. 이러한 결과는 비전-언어 모델 기능을 연합 학습과 결합하여 개인 정보를 보호하며, 통신 효율성 및 높은 성능을 달성할 수 있음을 증명합니다.



### IL-PCSR: Legal Corpus for Prior Case and Statute Retrieva (https://arxiv.org/abs/2511.00268)
Comments:
          Accepted at EMNLP 2025 (Main)

- **What's New**: 이 논문은 법률 사례와 관련 법규를 검색하는 두 가지 작업인 Legal Statute Retrieval (LSR)와 Prior Case Retrieval (PCR)를 통합하는 IL-PCSR (Indian Legal Corpus for Prior Case and Statute Retrieval)라는 고유한 말뭉치를 제안합니다. 기존의 접근법은 각각의 작업에 독립적으로 모델을 개발했으나, 두 작업 간의 상호 의존성을 활용하고자 합니다. IL-PCSR은 동일한 쿼리 집합에 대해 관련된 법규와 사례를 동시에 탐색할 수 있는 첫 번째 데이터셋입니다.

- **Technical Details**: IL-PCSR은 936개의 법규와 3,183개의 이전 사례, 6,271개의 쿼리 문서로 구성됩니다. 법률 문서는 인도 대법원과 고등법원 판결에서 수집된 것으로, 공공적으로 이용 가능한 자료를 API로 통해 얻었습니다. 이 데이터셋은 기계 학습 모델이 법규와 사례 간의 의존성을 학습할 수 있도록 설계되었으며, LLM(대규모 언어 모델) 기법을 이용한 재정렬 방법도 개발하였습니다.

- **Performance Highlights**: 논문에서는 다양한 모델을 사용하여 LSR와 PCR 작업의 성능을 평가했으며, LSR과 PCR 각각에 대해 문자 기반 모델과 의미 기반 모델을 포함한 다수의 실험을 수행했습니다. 제안하는 파이프라인 기반 접근 방식은 각 작업의 성능을 개별적으로 개선할 뿐만 아니라, 다중 작업 모델에도 긍정적인 영향을 미쳤습니다. 실험 결과, 법규 검색 및 사례 검색 간의 차이가 성능에 미친 영향을 분석하였고, 이러한 인사이트는 향후 연구의 방향을 제시합니다.



### Neural Transparency: Mechanistic Interpretability Interfaces for Anticipating Model Behaviors for Personalized AI (https://arxiv.org/abs/2511.00230)
Comments:
          SK and AB are co-first authors

- **What's New**: 이 논문에서는 사용자 맞춤형 LLM 기반 챗봇 디자인 시 언어 모델의 내부 구조를 드러내는 인터페이스를 소개합니다. 이 인터페이스는 행동 특성 벡터(behavioral trait vectors)를 추출하여 챗봇의 행동을 예측할 수 있게 돕습니다. 사용자들이 자신의 디자인 선택이 실제 행동에 어떻게 나타날지를 미리 예상할 수 있도록 하여 안전하고 더 나은 AI와의 상호작용이 가능하게 합니다.

- **Technical Details**: 제안된 인터페이스는 사용자가 설계한 시스템 프롬프트에 따른 신경 활성 패턴(neural activation patterns)을 분석하여, 실시간으로 성격 특성(personality traits)의 예측을 제공합니다. 사용자는 챗봇과의 대화 전, 디자인 선택이 다양한 상호작용 맥락(context)에서 어떻게 나타날 수 있는지를 직관적인 시각화(visualization)를 통해 관찰할 수 있습니다. 이 방식은 사용자가 설계 단계에서 위험 요소를 미리 식별하고 완화할 수 있도록 돕습니다.

- **Performance Highlights**: 연구에서는 사용자들이 챗봇 행동을 이해하는 데 있어 신경 투명성(neural transparency) 피드백이 긍정적인 영향을 미친다는 결과를 도출했습니다. 사용자들은 자신의 챗봇 행동을 정확하게 예측하지 못하는 경향이 있으며, 이러한 피드백 기반 도구가 사용자의 챗봇 신뢰도를 높였다는 증거도 제시되었습니다. 이 연구는 사용자 친화적인 AI 생성 도구 개발을 위한 기초를 마련하며, 더 안전하고 일치된 인간-AI 상호작용을 목표로 합니다.



### Consistently Simulating Human Personas with Multi-Turn Reinforcement Learning (https://arxiv.org/abs/2511.00222)
- **What's New**: 이번 논문에서는 LLM(대형 언어 모델)의 페르소나 일관성을 평가하고 개선하기 위한 통합 프레임워크를 소개합니다. 기존의 LLM들이 역할에 맞는 행동을 유지하지 못하고, 이전의 진술과 모순되거나, 역할에서 벗어나는 경우가 많습니다. 저자들은 이러한 문제를 해결하기 위해 다양한 자동화된 메트릭(자동 지표)을 정의하였습니다.

- **Technical Details**: 세 가지 자동화 메트릭인 prompt-to-line consistency(프롬프트-라인 일관성), line-to-line consistency(라인-라인 일관성), Q&A consistency(질문-답변 일관성)를 활용하여 기존 LLM 대화에서 발생하는 페르소나 드리프트를 측정합니다. 이러한 메트릭은 인간 주석과 검증되며, 이를 통해 세 가지 사용자 역할(환자, 학생, 사회적 대화 파트너)을 위한 멀티 턴 강화 학습(multi-turn reinforcement learning)을 적용합니다. 이 방법을 통해 LLM의 불일치성을 55% 이상 줄일 수 있었습니다.

- **Performance Highlights**: 우리는 제안한 메트릭을 보상 신호로 활용하여 LLM을 세 가지 사용자 역할에 맞게 미세 조정하였습니다. 그 결과, 좀 더 일관되고 일치된 대화를 생성할 수 있으며, 이로 인해 LLM 기반의 시뮬레이션이 더욱 신뢰할 수 있게 됩니다. 또한, 이러한 접근 방식은 사회 과학 및 강화 학습(RL) 파이프라인에서의 효과적인 응용을 가능하게 합니다.



### DM-QPMNET: Dual-modality fusion network for cell segmentation in quantitative phase microscopy (https://arxiv.org/abs/2511.00218)
Comments:
          5 pages, 4 figures

- **What's New**: 이 논문은 단일 샷 정량적 위상 현미경(single-shot quantitative phase microscopy, ssQPM)에서 세포 분할(Cell Segmentation)의 문제를 해결하기 위한 새로운 네트워크 구조인 DM-QPMNet을 소개합니다. 기존의 단일 모달 리지 방법이 잡음(noise)과 세포 밀도에 민감한 반면, DM-QPMNet은 다중 모달리티(multi-modality)를 활용하여 효과적인 정보 융합(fusion)을 통해 견고한 세포 분할을 구현합니다. 이는 다양한 위상 정보와 극성 강도(polarized intensity) 이미지를 독립적으로 처리하여 상호작용하는 이중 인코더 구조를 바탕으로 합니다.

- **Technical Details**: 네트워크는 서로 다른 인코딩 스트림을 가진 두 개의 인코더를 사용하여 극성 강도와 위상 맵을 별개의 모달리티로 처리합니다. 중간 깊이에서 다중 헤드 주의(multi-head attention)를 통해 모달리티별 특징을 융합하여, 고주파 경계(edge)와 텍스처 특징이 선택적으로 상보적인 위상 정보를 통합할 수 있게 합니다. 이러한 설계는 훈련의 안정성을 유지하면서도 최소한의 매개변수로 다중 모달 통합을 구현합니다.

- **Performance Highlights**: DM-QPMNet은 단일 모달리티 방법 및 조기 융합(early-fusion) 기법에 비해 현저한 성능 향상을 보여주었습니다. 특히, 각 모달리티의 특성에 맞춘 인코딩 방식과 늦은 융합(late fusion)을 통해 ssQPM에서 유용한 고주파 및 저주파 정보를 효과적으로 수집하고 활용할 수 있음을 입증합니다. 이러한 접근법은 생물학적 샘플 분석에서의 신뢰성과 정확성을 높이는 데 기여할 것으로 예상됩니다.



### An Efficient and Generalizable Transfer Learning Method for Weather Condition Detection on Ground Terminals (https://arxiv.org/abs/2511.00211)
- **What's New**: 본 논문은 지구 저궤도(LEO) 위성으로 제공되는 위성 인터넷의 신뢰성 있는 성능이 기상 조건에 따라 크게 영향을 받음을 보여주고 있습니다. 또한, 기상 이벤트, 특히 눈과 비가 위성 인터넷의 주요 지상 터미널 구성 요소의 성능을 저하시킬 수 있다는 점을 강조합니다. 이 연구는 기상 조건 감지의 효율적인 transfer learning (TL) 방법을 제안하여, 다양한 기상 조건에서의 위성 안테나 분류 문제를 해결하고자 합니다.

- **Technical Details**: 전이 학습(TL)을 활용하여, 기상 데이터베이스에서 훈련된 모델의 지식을 위성 안테나 분류 작업에 적용합니다. 특히, 제안된 방법에서는 YOLACT 프레임워크를 사용하여 지상 터미널의 위성 안테나를 효과적으로 세분화하고 특징을 분리합니다. 이 연구는 데이터 부족 문제 해결을 목표로 하며, 다양한 기상 조건에서 위성 안테나의 상태를 정확하게 분류하도록 모델을 적응시킵니다.

- **Performance Highlights**: 제안된 모델은 초기 시나리오에서 80개의 훈련 이미지를 사용하여 50 에포크 내에 88.33%의 정확도를 달성하며, 기존의 YOLO 및 Faster R-CNN 모델을 초과하는 성능을 보여주었습니다. 확장된 시나리오에서도 180개의 훈련 이미지를 사용해 50 에포크 동안 동일한 정확도를 유지하며, 경쟁 모델인 R-YOLO와 Faster R-CNN에 비해 높은 성능을 입증했습니다.



### Diffusion Models at the Drug Discovery Frontier: A Review on Generating Small Molecules versus Therapeutic Peptides (https://arxiv.org/abs/2511.00209)
Comments:
          21 pages, 3 figures

- **What's New**: 이 논문은 최근의 복합 모델의 발전이 약물 발견 과정에 미치는 영향과 이를 통해 소분자(small molecules) 및 치료 펩타이드(therapeutic peptides) 설계에 어떻게 적용되는지를 비교적으로 분석하고 있습니다. 특히, 반복적인 디노이징(iterative denoising) 프레임워크가 각 약리 모달리티에 맞춰 어떻게 적응되는지를 중점적으로 설명합니다. 이러한 접근 방식은 기존의 약물 발견 프로세스를 혁신할 수 있는 잠재력을 보여줍니다.

- **Technical Details**: 복합 모델은 원래 데이터에 가우시안 노이즈를 추가하는 전방 확산(forward diffusion) 과정과 이를 역으로 디노이즈하는 과정으로 구성됩니다. 이 과정은 수학적으로 q(xt|xt−1)=𝒩(xt;1−βtxt−1,βtI)로 표현되며, 디노이징 기술은 새로운 데이터 생성에 중요한 역할을 합니다. 이 모델들은 다른 데이터 모달리티에 맞게 조정될 수 있는 유연성을 가지고 있어 이미지 생성, 자연어 처리 및 생물정보학 등 다양한 분야에서 성공적인 응용 사례를 보여줍니다.

- **Performance Highlights**: 복합 모델은 소분자 및 치료 펩타이드의 설계에서 높은 성능을 보이며, 양질의 다양한 샘플을 생성하는 데 탁월한 능력을 보여주고 있습니다. 각 분야의 특성에 맞춘 도전 과제도 있지만, 이러한 기술이 인공지능 기반 설계에서 크게 활용될 가능성이 큽니다. 특히, 실험 검증과 고품질 데이터의 부족을 해결하는 것이 향후 모델의 최적화와 발전에 중요한 요소로 지목되고 있습니다.



### Training LLMs Beyond Next Token Prediction - Filling the Mutual Information Gap (https://arxiv.org/abs/2511.00198)
- **What's New**: 이 연구는 대형 언어 모델(LLM) 훈련의 최적화 방식에 도전합니다. 전통적인 다음 토큰 예측(Next-Token Prediction, NTP) 접근법에서 벗어나 정보가 풍부한 토큰을 예측하는 방식이 더 효과적임을 주장합니다. 수학적 계산, 다중 레이블 분류(Multi-Label Classification, MLC), 자연어 생성(Natural Language Generation, TG)의 세 가지 과제를 통해 모델 성능을 개선하는 방법을 제시합니다.

- **Technical Details**: 연구는 LLM의 훈련 과정에서 특정 목표(Gathered) 토큰을 우선시하는 전략을 수립하는 것을 목표로 하고 있습니다. 이 새로운 접근법은 토큰 선택(order of tokens)을 개선하고오류 누적을 줄여 훈련 효율성을 높이는 데 중점을 두고 있습니다. 이론적 프레임워크를 기반으로 하여 원천과 목표 토큰 간의 상호 정보(mutual information)를 분석하여 정보가 풍부한 토큰 우선순위를 설정합니다.

- **Performance Highlights**: 실험 결과, NTP 방식이 수학, MLC, TG의 세 가지 전통적인 시나리오에서 비효율적이라는 사실을 확인했습니다. 정보가 풍부한 토큰 우선순위를 설정한 결과, 다양한 최신 LLM(예: GPT-2, Qwen2.5, Llama-3.2)에서 일관된 정확도, 당황도(perplexity), ROUGE 메트릭 향상을 보여주었습니다. 이로 인해 특정 토큰 예측 전략이 모델 성능 증진에 미치는 영향을 그대로 드러냈으며, LLM 훈련 방법론을 개선하는 대안을 제공합니다.



### Understanding Code Agent Behaviour: An Empirical Study of Success and Failure Trajectories (https://arxiv.org/abs/2511.00197)
- **What's New**: 이 논문은 최근 복잡한 소프트웨어 엔지니어링 작업을 위해 Large Language Model (LLM) 에이전트의 문제 해결 행동을 분석한 최초의 연구로, 성공 지표를 넘어 에이전트의 실행 경로를 조명합니다. 구체적으로 OpenHands, SWE-agent, Prometheus의 세 가지 최신 코드 에이전트의 수행 경로를 비교 분석함으로써, 각 에이전트가 소프트웨어 문제를 해결하기 위해 취하는 단계들을 탐구합니다. 이를 통해 각 에이전트의 문제 해결 전략과 실패 패턴 간의 차이를 명확히 규명했습니다.

- **Technical Details**: 이 연구에서는 SWE-Bench 벤치마크 코드 에이전트의 에이전트 경로를 분석하여, 성공적인 해결 시나리오와 실패 시나리오 간의 차이를 확인합니다. 분석된 에이전트들은 그들의 실행 경로와 문제 해결 전략에서 나타나는 고유의 특징을 보였습니다. 각 에이전트의 경로는 실패가 성공보다 일관되게 긴 경향이 있으며, 각 실패는 서로 다른 패턴을 보이는 것으로 나타났습니다.

- **Performance Highlights**: 연구 결과에 따르면, 성공적인 경로는 주로 문제 해결을 위한 방어적 프로그래밍(defensive programming)과 맥락 수집(context gathering)과 같은 전략에 크게 의존하며, 실패한 경로는 길고 높은 변동성을 보입니다. 또한 대부분의 경로는 문제 있는 파일을 정확하게 식별하지만(72-81%) 성공은 코드 수정의 정확성보다 근사치에 따라 달라지는 경향을 보였습니다. 이러한 발견들은 에이전트의 동작을 이해하고 향후 더 강력하고 해석 가능한 자율 소프트웨어 엔지니어링 시스템의 개발을 위한 기초를 제공합니다.



### EL-MIA: Quantifying Membership Inference Risks of Sensitive Entities in LLMs (https://arxiv.org/abs/2511.00192)
- **What's New**: 본 논문에서는 Membership Inference Attacks (MIA)의 새로운 과제를 제안합니다. 특히, 민감한 정보(PII, 신용카드 번호 등)에 중점을 둔 엔터티(entities) 수준에서의 소속 위험(discovery of membership risk)을 다루고 있습니다. 기존 MIA 방법이 전체 프롬프트(prompt)나 문서의 존재를 탐지하는 데는 성공하지만, 세부적인 리스크를 포착하는 데에는 한계가 있습니다.

- **Technical Details**: 우리는 LLM(대형 언어 모델)에서 엔터티 수준의 소속 위험을 감사(audit)하기 위한 EL-MIA 프레임워크를 제안합니다. 이를 평가하기 위해 새로운 벤치마크 데이터셋을 구성하였으며, 기존 MIA 기술과 두 가지 새롭게 제안한 방법에 대한 체계적인 비교를 수행했습니다. 결과 분석을 통해 엔터티 수준의 MIA 취약성과 모델 규모, 교육(epoch) 횟수 및 다른 표면적인 요인들 간의 관계를 설명하려고 하였습니다.

- **Performance Highlights**: 연구 결과, 기존 MIA 방법들이 민감한 속성에 대한 엔터티 수준의 소속 추론에 한계가 있음을 보여주었습니다. 반면, 이 취약성은 상대적으로 간단한 방법으로도 드러낼 수 있어, 제공된 위협 모델을 강하게 테스트할 필요성을 강조합니다. 이는 MIA 분야의 연구자들에게 더 강력한 적대자(adversary)를 요구하는 중요한 결과로 해석될 수 있습니다.



### A Retrospect to Multi-prompt Learning across Vision and Languag (https://arxiv.org/abs/2511.00191)
Comments:
          ICCV

- **What's New**: 최근 비전-언어 프리트레이닝 모델(VLM)의 발전으로 멀티-프롬프트 학습(multi-prompt learning) 기술이 주목받고 있습니다. 기존의 연구는 단일 프롬프트 패러다임에 초점을 맞추는 반면, 본 논문은 이론적 기반을 바탕으로 멀티-프롬프트 학습을 다룹니다. 특히 에너지 기반 멀티-프롬프트 학습(EMPL)을 제안하여 여러 프롬프트 임베딩을 생성함으로써 인 도메인(in-domain)과 아웃 도메인(out-of-domain) 전이 학습(transfer learning)의 균형을 이룹니다.

- **Technical Details**: 본 논문은 VLM을 위한 멀티-프롬프트 학습의 기초를 다지며, 에너지 기반의 분포를 통해 프롬프트를 학습하는 EMPL 메소드에 대해 설명합니다. 임베딩 파라미터의 최적화를 위해, 우리는 프롬프트와 이미지를 변수로 사용하여 불확실성과 관계된 모델링을 진행합니다. 또한, 이중 마르코프 체인 몬테카를로(MCMC) 샘플러를 활용해 반복적으로 프롬프트를 생성함으로써 파라미터 효율성을 극대화하고, 기존의 프롬프트 학습 전략과의 호환성을 확보합니다.

- **Performance Highlights**: 종합적인 실험을 통해 제안된 EMPL의 우수성을 입증하며, 멀티-프롬프트 학습이 단일 프롬프트보다 더 나은 성능을 발휘함을 확인하였습니다. 특히, VLM의 비전-언어 전이 가능성을 높이고, 다수의 프롬프트를 사용함으로써 모델의 일반화 문제를 해결하는 데 효과적임을 보여줍니다. 본 연구는 VLM의 발전 방향에 새로운 통찰력을 제공하며, 멀티-프롬프트 학습의 가능성을 한층 확장시킬 것입니다.



### Generative Modeling Enables Molecular Structure Retrieval from Coulomb Explosion Imaging (https://arxiv.org/abs/2511.00179)
- **What's New**: 본 연구는 Coulomb explosion imaging 기법을 사용하여 분자의 화학 반응 중 실시간으로 구조 변화를 포착하는 문제를 다룹니다. 특히, 이 연구는 diffusion-based Transformer 신경망을 활용하여 이온 모멘텀 분포에서 알려지지 않은 분자 기하학을 재구성하는 방법을 제안합니다. 이 결과는 평균 절대 오차가 한 보어 반경 이하로, 이는 일반적인 화학 결합 길이의 절반에 해당합니다.

- **Technical Details**: Research에 사용된 모델은 Monte Carlo / Molecular Dynamics 시뮬레이션과 classical over-the-barrier 모델을 결합하여 내전이 photoionization 및 원자 동역학을 추적합니다. 가장 흥미로운 점은 주어진 300개의 분자를 평형 기하학에서 Coulomb explosion하게 하여 이온 조각의 모멘텀을 평균 내는 방식으로 데이터셋을 생성했다는 것입니다. 또한, 이 연구에서는 Grab-Schmidt 과정을 통해 좌표계의 일관성을 보장하고, MOLEXA 모델이 입력하는 구조를 역으로 비교하는 방식을 사용해 불확실성을 추정했습니다.

- **Performance Highlights**: 이 연구에서 MOLEXA는 Coulomb explosion을 통해 생성된 이온 모멘텀 분포로부터 분자 구조를 재구성하는 데 성공했습니다. 이 과정에서 단일 훈련 단계당 Structure Denoising Module은 한 번만 실행되어 기본 구조에서 노이즈를 제거하는 방식으로 작동했습니다. 이 결과는 불확실성 추정 기능을 포함하고 있어, 학습한 구조에 대한 신뢰도를 제공하며, 다양한 화학적 조합에 대해 대규모 데이터셋을 효과적으로 생성할 수 있음을 보여줍니다.



### Effectiveness of LLMs in Temporal User Profiling for Recommendation (https://arxiv.org/abs/2511.00176)
Comments:
          Accepted to the IEEE International Conference on Data Mining (ICDM 2025), Workshop on User Modeling and Recommendation (UMRec). To appear in the IEEE ICDMW 2025 proceedings

- **What's New**: 이 논문은 사용자 취향의 동적 특성을 효과적으로 모델링하는 데 있어 Large Language Models(LLMs)를 활용하여 추천 시스템의 품질을 개선하는 방법을 다룹니다. 전통적인 사용자 프로파일링 기법은 일시적인 단기 관심과 안정적인 장기 선호를 구분하지 못해 추천의 정확성과 투명성을 저해하는데, 본 연구는 이러한 차이를 텍스트 요약을 통해 캡처합니다. 재구성된 사용자 표현은 사용자와 아이템 간의 상호작용 이력을 기반으로 하여 단기 및 장기 요약을 생성하고, 이를 통해 이론적 해석 가능성을 제공합니다.

- **Technical Details**: 이 연구에서는 사용자와 아이템 간의 상호작용 및 타임스탬프를 기반으로 하는 상호작용 기록을 정의합니다. LLM을 활용한 사용자 프로파일의 생성을 통해 단기 및 장기 프로파일을 뚜렷하게 구분하고, BERT를 이용하여 이러한 프로파일을 임베딩합니다. 학습 가능한 주의(attention) 메커니즘을 적용하여 이 임베딩들을 통합하여 최종 사용자 표현을 생성하며, 이는 최근의 관심과 오래된 관심 사이의 결정적 근거를 명확히 전달합니다.

- **Performance Highlights**: 본 연구는 Movies&TV 및 Video Games라는 두 개의 아마존 도메인에서 평가했습니다. Movies&TV 도메인에서 추천 품질을 17%나 향상시켰으나, Video Games 도메인에서는 덜 뚜렷한 혜택을 보였습니다. 이는 사용자 프로파일 밀도와 행동 다양성이 낮은 도메인에서 단기와 장기 선호의 구분이 이뤄지지 않기 때문으로, LLM 기반의 접근 방식의 적용이 어떤 환경에서 더욱 유용한지에 대한 통찰을 제공합니다.



### What a diff makes: automating code migration with large language models (https://arxiv.org/abs/2511.00160)
Comments:
          10 pages, 8 figures

- **What's New**: 이 논문에서는 코드 마이그레이션(code migration)을 위한 LLMs(대형 언어 모델)의 활용을 탐구합니다. 특히, 종속성과의 호환성을 유지하며 주요 및 부차적인 의미 버전 변경을 처리하는 방법에 중점을 둡니다. AIMigrate라는 파이썬 오픈 소스 패키지를 제공하며, 이 도구를 통해 실제 TYPHOIDSIM과 STARSIM 버전 간 마이그레이션 사례에서 65%의 필수 변경 사항을 정확히 식별했습니다.

- **Technical Details**: 본 연구에서는 diff 유틸리티와 LLM을 결합하여 코드 마이그레이션 문제에 접근합니다. 이 과정에서 최대 128k 토큰을 처리할 수 있는 최신 모델 컨텍스트 창을 활용하고, diff 유틸리티를 이용하여 가장 긴 공통 부분 수열(LCS)을 찾는 알고리즘을 사용합니다. 이 방식은 특정 라이브러리나 프로그래밍 언어에 국한되지 않으며, 일반적인 마이그레이션 문제를 해결하기 위한 접근법을 제안합니다.

- **Performance Highlights**: 실제 사례 연구에서 AIMigrate는 한 번의 실행으로 65%의 필수 변경 사항을 올바르게 식별했으며, 여러 번의 실행을 통해 이 비율은 80%까지 증가했습니다. 또한, 결과의 47%는 완벽하게 생성되었습니다. 이러한 성과는 LLM을 코드 변환 시 사용하는 것보다 더 효과적일 수 있음을 시사합니다.



### FLoC: Facility Location-Based Efficient Visual Token Compression for Long Video Understanding (https://arxiv.org/abs/2511.00141)
- **What's New**: 이번 논문에서는 FLoC라는 비주얼 토큰 압축 프레임워크를 제안합니다. 이 프레임워크는 주어진 비주얼 토큰 수를 기반으로, 압축이 용이하면서도 다채로운 비주얼 토큰의 하위 집합을 신속하게 선택합니다. 특히, 이 방법은 모델에 의존하지 않으며, 질의(query)나 특정 모델에 구애받지 않고 다양한 비디오-LLM과 통합하여 사용할 수 있다는 장점이 있습니다.

- **Technical Details**: FLoC는 시설 위치 함수(facility location function)를 기반으로 하여 시각적 토큰 선택을 서브모듈 최적화(submodular optimization) 문제로 해석합니다. 이를 통해 최소한의 계산 오버헤드로 비주얼 토큰을 선택할 수 있으며, 각각의 하위 집합은 전체 토큰과의 유사성을 고려하여 설계됩니다. 이 과정에서 사용되는 지연 탐욕 알고리즘(lazy greedy algorithm)은 토큰 선택의 효율성을 크게 향상시키며, 종합적으로 비주얼 이해 작업의 필수 정보를 효과적으로 보존할 수 있게 합니다.

- **Performance Highlights**: 대규모 벤치마크인 Video-MME, MLVU, LongVideoBench에 대한 광범위한 평가 결과, FLoC는 기존의 압축 기법을 지속적으로 초월했습니다. 이 연구는 특히 긴 비디오 이해의 주요 도전을 다루는 데 있어 효과성과 강력함을 강조하며, 처리 속도에서도 효율성을 보여줍니다. 이러한 결과는 FLoC가 제공하는 높은 성능과 다양한 응용 가능성을 보여줍니다.



### End-to-End Dexterous Arm-Hand VLA Policies via Shared Autonomy: VR Teleoperation Augmented by Autonomous Hand VLA Policy for Efficient Data Collection (https://arxiv.org/abs/2511.00139)
- **What's New**: 이 논문에서는 로봇의 인간과 유사한 손재주 조작을 위한 Shared Autonomy 프레임워크를 제안합니다. 이 프레임워크는 로봇의 팔 움직임과 미세한 손 조작을 인지적으로 분담하며, 인간 운영자가 VR(가상 현실) 인터페이스를 통해 로봇의 팔 자세를 안내하고, 자율적인 DexGrasp-VLA 정책이 정밀한 손 조작을 실시간으로 처리합니다. 이러한 방식은 고품질의 협조된 팔-손 시연을 효율적으로 수집할 수 있게 도와줍니다.

- **Technical Details**: Shared Autonomy 프레임워크는 인간과 AI의 상호 보완적인 강점을 활용하여 구현됩니다. 이 시스템에서는 DexGrasp-VLA 컨트롤러가 손을 위한 VLA Copilot 역할을 수행하며, 시각적, 언어적, 촉각적 피드백을 통합하여 그립을 조정합니다. Arm-Hand Feature Enhancement 모듈은 팔과 손의 동작을 구분하여 인코딩하며, 학습된 정책이 보다 자연스러운 협조 조작을 가능하게 만듭니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 50개 이상의 다양한 객체에서 90%의 성공률을 달성하며, 고품질의 협조된 팔-손 시연을 효율적으로 수집할 수 있음을 입증하였습니다. 또한, 각 구성 요소의 필요성을 입증하는 감쇠 연구를 통해 DexGrasp-VLA 모델, Arm-Hand Feature Enhancement 모듈, Corrective Teleoperation 시스템의 중요성을 강조하였습니다.



### A Dual Large Language Models Architecture with Herald Guided Prompts for Parallel Fine Grained Traffic Signal Contro (https://arxiv.org/abs/2511.00136)
- **What's New**: 이 논문에서는 기존의 고정 시간 신호 장치에 대해 더 나은 적응성을 제공하는 HeraldLight라는 새로운 이중 LLM(large language models) 아키텍처를 소개합니다. 이 시스템은 Herald 모듈을 통해 실시간 교통 CONDITIONS에 기반한 큐 길이 예측과 정보 추출을 수행하여 신호 제어의 효율을 극대화합니다. 또한, LLM-Agent와 LLM-Critic의 상호 작용을 통해 신호 제어의 정확성과 신뢰성을 향상시키기 위한 방법론을 제안합니다.

- **Technical Details**: HeraldLight 아키텍처는 LLM-Agent가 실시간 교통 조건에 따라 신호 지속 시간과 활성 신호 단계를 추론하도록 하는 LoRA 기반 imitation fine-tuning으로 강화됩니다. Herald 모듈은 최대 40초 앞서 예상 큐 상태를 예측하고, LLM-Critic은 LLM-Agent의 출력을 평가하여 잘못된 정보와 환각(hallucination)을 수정합니다. 이 시스템은 신호 단계의 지속 시간을 동적으로 조정하여 세밀한 신호 제어를 지원합니다.

- **Performance Highlights**: CityFlow 시뮬레이터를 사용하여 Jinan, Hangzhou, New York의 실제 교통 데이터 세트를 바탕으로 한 실험에서는 HeraldLight가 기존의 최첨단 방법들보다 평균 여행 시간을 20.03% 단축시키고, Jinan과 Hangzhou에서는 평균 큐 길이를 10.74% 감소시키는 성과를 나타냈습니다. 이러한 결과는 동적 신호 제어 방식의 가능성을 극대화함을 보여줍니다.



### Feature Importance Guided Random Forest Learning with Simulated Annealing Based Hyperparameter Tuning (https://arxiv.org/abs/2511.00133)
Comments:
          10 pages, 2 figures, 3 tables, submitted to IEEE Intelligent Systems journal

- **What's New**: 본 논문에서는 Random Forest 분류기를 향상시키기 위한 새로운 프레임워크를 소개합니다. 이 프레임워크는 확률적 피처 샘플링(probablistic feature sampling)과 시뮬레이티드 어닐링(Simulated Annealing)을 통해 하이퍼파라미터 튜닝을 통합합니다. 다양한 도메인에서의 강력한 분류 문제를 해결하는 데 있어 예측 정확도(predictive accuracy)와 일반화(generalization)에서 상당한 개선을 보여줍니다.

- **Technical Details**: 제안된 Feature Importance Guided Random Forest (FIGRF) 프레임워크는 피처 중요도 점수(feature importance scores)를 사용하여 Random Forest를 구성하는 데 도움을 줍니다. 여러 기존 연구와의 비교 연구를 통해 비선형 종속성과 복잡한 상호 작용을 효과적으로 포착하는 앙상블 방법을 소개합니다. Gini 중요도(Gini importance), 순열 중요도(Permutation Importance), 상호 정보(mutual information)를 조합하여 피처 선택을 위한 복합적인 프레임워크를 구축했습니다.

- **Performance Highlights**: FIGRF 모델은 예측 성능에서 일관된 개선을 보여주고, 피처의 중요도에 대한 의미 있는 인사이트를 제공합니다. 결과적으로, 중요성 인식 샘플링과 메타 휴리스틱 최적화를 결합함으로써 성능 향상을 달성했습니다. 이 모델은 다양한 데이터 구조에서도 견고하게 작동하며, 차별화된 피처 선택을 용이하게 합니다.



### Casing Collar Identification using AlexNet-based Neural Networks for Depth Measurement in Oil and Gas Wells (https://arxiv.org/abs/2511.00129)
- **What's New**: 이 논문은 Casing Collar Locator (CCL) 신호 인식을 위한 데이터 세트를 구축하기 위해 다운홀 도구에 통합된 시스템을 제시합니다. 또한, 기존의 카라 신호 인식 방법의 한계를 극복하기 위한 데이터 증강(Data Augmentation) 전처리 방법들을 포괄적으로 제안하고, 이에 대한 효과성을 평가합니다. 이는 다운홀 환경에서의 데이터 부족 문제를 해결하려는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 기술적으로, 본 연구는 AlexNet 기반의 신경망 모델을 활용하여 CCL 신호의 사전 처리 및 훈련 방법을 체계적으로 분석합니다. 데이터 표준화, 레이블 배포 평활화(Label Distribution Smoothing, LDS), 랜덤 크롭핑 등의 기본적인 전처리 방법들이 강조되며, 레이블 스무딩 정규화(Label Smoothing Regularization, LSR)와 같은 고급 기법들이 모델의 일반화 능력을 크게 향상시키는 것으로 나타났습니다.

- **Performance Highlights**: 제안된 데이터 증강 방법을 사용하여 훈련된 두 개의 기준 모델의 F1 점수는 각각 0.937 및 0.952에서 1.0으로 최대 개선되었습니다. 실시간 CCL 파형에 대한 성능 검증 결과, 제안된 방법이 실제적인 적용 가능성과 효과성을 입증했습니다. 이러한 결과는 다운홀 환경에서의 카라 인식 모델 훈련을 위한 데이터 증강의 중요성을 강조합니다.



### Dynamic Model Selection for Trajectory Prediction via Pairwise Ranking and Meta-Features (https://arxiv.org/abs/2511.00126)
- **What's New**: 이 논문에서는 복잡한 길고 궁극적으로 신뢰할 수 없는 주행 시나리오에서의 한계점을 극복하기 위해 동적 다중 전문가 게이팅 프레임워크를 제안합니다. 이 프레임워크는 물리 기반 LSTM, Transformer, 그리고 미세 조정된 GameFormer의 세 가지 예측기를 상황에 맞게 선택하여 사용합니다. 특히, 내부 모델 신호인 메타 피처(meta-features)를 활용하여 각 예측기의 신뢰성을 평가하는 새로운 접근 방식을 도입했습니다.

- **Technical Details**: 논문에서는 각 예측기 간의 쌍별 순위화(pairwise ranking)를 통해 전문가 선택 문제를 정의하며, 이는 기존의 보정(calibration) 문제를 피할 수 있습니다. LLM(대형 언어 모델)을 활용하여 낮은 신뢰도의 경우에 대한 세맨틱(scene understanding) 및 위험(reasoning) 추론을 수행하는 감독자 역할을 함께 도입하였습니다. 이 접근은 높은 수준의 해석 가능성을 제공합니다.

- **Performance Highlights**: nuPlan-mini 데이터셋에서 수행한 평가 결과, 제안된 다중 전문가 게이트가 2.567 m의 최종 변위 오차(FDE)를 달성하며, 이는 기존의 단일 전문가에 비해 9.5% 개선된 성능을 나타냅니다. 또한, 왼쪽 회전 시나리오에서의 개방형 시뮬레이션이 약 10%의 성능 향상을 보이는 등의 일관된 개선 효과가 확인되었습니다.



### Inferring multiple helper Dafny assertions with LLMs (https://arxiv.org/abs/2511.00125)
- **What's New**: 이번 연구에서는 Dafny 프로그램에서 누락된 어시션을 자동으로 추론하기 위해 대규모 언어 모델(LLM)을 활용하는 접근 방식을 제안합니다. 특히 여러 개의 어시션이 결여된 경우를 중점적으로 다루며, DafnyBench 벤치마크를 확장하여 제어된 평가를 지원하는 데이터셋을 구축했습니다. 새로운 도구인 DAISY(Dafny Assertion Inference System)를 통해 이 접근 방식이 실제 프로그램 검증에 효과적임을 입증했습니다.

- **Technical Details**: 이 연구는 두 단계로 구성된 파이프라인을 통해 누락된 어시션의 위치를 식별하고, LLM의 예측을 오류 메세지에 기반한 휴리스틱과 결합하여 어시션을 추론합니다. 특히, 506개의 프로그램을 대상으로 DAISY를 평가한 결과, 단일 어시션 결여한 프로그램에서 63.4%의 성공률을 기록했으며, 여러 개의 어시션이 결여된 경우에서도 31.7%의 검증 성공률을 보였습니다.

- **Performance Highlights**: DAISY는 어시션이 하나 부족한 경우 63.4%, 여러 개가 부족한 경우 31.7%의 검증 성공률을 기록했습니다. 게다가, 원래 문서에 있던 어시션보다 적은 수의 어시션으로도 검증이 가능한 경우가 많아, 이는 여러 가지 유효한 수리 전략이 존재함을 보여줍니다. 이 연구 결과는 자동화된 어시션 추론이 증명 작업의 노력을 상당히 줄일 수 있음을 나타내며, 형식 검증의 확대 가능성을 열어줍니다.



### Cross-fluctuation phase transitions reveal sampling dynamics in diffusion models (https://arxiv.org/abs/2511.00124)
Comments:
          Accepted at NeurIPS 2025. 10 pages, camera-ready version. appendices included

- **What's New**: 이 연구에서는 score-based diffusion models에서의 샘플링 동역학을 분석하였습니다. 특히, 중심 모멘트 통계인 cross-fluctuations를 사용하여 샘플이 어떻게 뚜렷한 전이(discrete transitions)를 거쳐 원하는 분포를 형성하는지를 보여줍니다. 이 과정은 역으로도 발생하여 초기 분포로 돌아갈 수 있는 경로를 제공하며, 이러한 전이가 n번째 교차 플럭투에이션에서 불연속성으로 감지될 수 있음을 증명합니다.

- **Technical Details**: 이론적 기여 섹션에서 이 연구는 세 가지 부분으로 구성된 일반 프레임워크를 소개합니다. cross-fluctuation 프레임워크는 단계적 구조의 동역학을 이해하는 데 사용되며, 귀결 분포를 위한 SDE/ODE 동역학의 세부사항을 제공합니다. 또한, 데이터 분포의 초기 상태가 확률 덩어리를 따라 결정론적으로 진화하는 과정을 설명하여 효율적인 추정 방법을 제시합니다.

- **Performance Highlights**: 우리의 프레임워크를 통해 교차 플럭투에이션을 모니터링함으로써 샘플링 효율성을 직접적으로 향상시키고, 클래스 조건부(resulting class-conditional) 및 희귀 클래스 생성에 가속화를 가져왔습니다. 또한, Image Classification 및 style transfer 등 두 가지 제로샷(zero-shot) 과제에서 성능의 개선을 보여주며, 이는 비싼 그리드 서치(grid search)나 재훈련 없이도 달성될 수 있음을 나타냅니다.



### VLM6D: VLM based 6Dof Pose Estimation based on RGB-D Images (https://arxiv.org/abs/2511.00120)
Comments:
          This paper has been accepted to IEIE( The Institute Of Electronics and Information Engineering, South Korea) Fall,2025 Conference

- **What's New**: 이 논문에서는 VLM6D라는 새로운 이중 스트림 아키텍처를 제안합니다. 이는 RGB-D 입력으로부터의 시각적 및 기하학적 데이터의 고유한 강점을 활용하여 6D 객체의 포즈 추정을 더욱 견고하고 정밀하게 수행합니다. 기존의 접근 방식들이 합성 데이터에서 실제 상황으로의 일반화에 어려움을 겪는 반면, VLM6D는 이러한 한계를 극복하고자 설계되었습니다.

- **Technical Details**: VLM6D는 두 개의 전문 인코더를 통합하고 있습니다. 강력한 자가 감독 기반의 Vision Transformer(DINOv2)는 RGB 모달리티를 처리하며, 비주얼 문법에 대한 깊은 이해를 통해 질감 및 조명 변화에 대해 뛰어난 내성을 자랑합니다. 동시에, PointNet++ 인코더는 깊이 데이터에서 파생된 3D 점 구름을 처리하며 심각한 가림 현상이 있는 경우에도 고급 기하학적 추론을 가능하게 합니다.

- **Performance Highlights**: 포괄적인 실험을 통해 VLM6D는 Occluded-LineMOD에서 새로운 SOTA(State of the Art) 성능을 달성했음을 입증하였습니다. 이는 VLM6D의 뛰어난 견고성과 정확성을 뒷받침하며, 현실 세계의 다양한 상황에서 효과적인 6D 포즈 추정이 가능함을 보여줍니다.



### DCcluster-Opt: Benchmarking Dynamic Multi-Objective Optimization for Geo-Distributed Data Center Workloads (https://arxiv.org/abs/2511.00117)
Comments:
          Submitted to the NeurIPS 2025 conference

- **What's New**: 본 논문에서는 지구적으로 분산된 데이터 센터에서 지속 가능한 작업 관리를 위한 DCcluster-Opt라는 개방형 소스, 고충실도 시뮬레이션 벤치마크를 소개합니다. DCcluster-Opt는 AI 작업 부하 트레이스, 그리드 탄소 밀도, 전기 시장, 날씨 등 다양한 실제 데이터셋을 결합하여 만들어졌습니다. 이 시스템은 동적으로 작업을 재배정하거나 보류해야 하는 복잡한 스케줄링 문제를 다루며, 여러 목표를 최적화할 수 있는 기능을 갖추고 있습니다.

- **Technical Details**: DCcluster-Opt는 데이터 센터 운영의 물리학에 대한 모델과 결합하여 환경 요인과 네트워크 역학 간의 상호 작용을 포괄적으로 캡처하도록 설계되었습니다. 이 시스템은 열 회수와 같은 고급 구성 요소를 모델링하고, 에너지 비용, 서비스 수준 협약, 물 사용량 등 다양한 요소 간의 트레이드오프를 명시적으로 연구할 수 있는 모듈형 보상 시스템을 제공합니다. 또한 Gymnasium API를 제공하여 강화 학습 및 규칙 기반 전략을 포함한 기준 컨트롤러를 지원하여 재현 가능한 ML 연구를 가능하게 합니다.

- **Performance Highlights**: DCcluster-Opt는 지속 가능한 컴퓨팅 솔루션 개발과 검증을 가속화할 수 있는 현실적이고 구성 가능한 테스트베드를 제공합니다. 이는 효율적인 자원 재배분 및 시간에 따른 작업 스케줄링을 통해 탄소 배출량 및 에너지 비용을 줄이는 데 기여할 수 있는 가능성을 지니고 있습니다. 또한 다양한 알고리즘의 공정한 비교와 테스트를 통해 향후 고급 AI 시스템의 연구를 지원합니다.



### LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers (https://arxiv.org/abs/2511.00116)
Comments:
          Submitted to the NeurIPS 2025 conference

- **What's New**: 논문에서는 AI 워크로드 증가로 인해 고밀도 데이터 센터에서 열 관리를 위한 액체 냉각의 중요성을 강조하고 있습니다. 이를 통해 LC-Opt라는 지속 가능한 액체 냉각 벤치마크 환경을 제시하여 에너지 효율성과 신뢰성을 향상시키는 머신 러닝 기반 제어 전략을 제공합니다. 고충실도의 디지털 트윈(digital twin)을 기반으로 하여 액체 냉각 시스템을 최적화하는 RL(강화 학습) 제어 전략을 지원합니다.

- **Technical Details**: LC-Opt는 Oak Ridge National Lab의 Frontier 슈퍼컴퓨터 냉각 시스템을 모델링하며, Modelica 기반의 세부적인 엔드 투 엔드 모델을 포함합니다. 이 환경은 데이터 센터의 냉각 타워와 서버 블레이드 그룹을 포괄하며, RL 에이전트는 IT 캐비닛 수준에서 액체 공급 온도 및 유량과 같은 중요한 열 제어를 최적화합니다. Gymnasium 인터페이스를 통해 사용자 정의 가능한 다양한 냉각 파라미터를 조정할 수 있는 다중 목적 실시간 최적화 문제를 생성합니다.

- **Performance Highlights**: 본 연구에서는 중앙집중형 및 분산 다중 에이전트 RL 접근 방식을 벤치마킹하고, 해석 가능한 제어를 위한 정책 증류(policy distillation) 기술을 보여줍니다. 결정 트리(decision tree) 및 회귀 트리(regression tree)로의 변환을 통해 사용자 신뢰를 구축하고 시스템 관리를 단순화하는 LLM(대형 언어 모델) 기반 방법을 탐구합니다. LC-Opt는 ML 커뮤니티와 운영자, 공급업체가 지속 가능한 데이터 센터 액체 냉각 제어 솔루션을 개발할 수 있도록 세부적이고 사용자 정의 가능한 액체 냉각 모델에 대한 접근을 민주화합니다.



### Cognitive Alignment in Personality Reasoning: Leveraging Prototype Theory for MBTI Inferenc (https://arxiv.org/abs/2511.00115)
- **What's New**: 이번 논문에서는 ProtoMBTI라는 프레임워크를 통해 MBTI 추론을 위한 프로토타입 기반 추론을 제안합니다. 기존의 하드 라벨 분류 방식 대신에 인지적으로 정렬된 프로토타입 이론을 적용하여 문자에서의 개인 성격 판단의 미세한 그라데이션을 반영합니다. 이 방식은 LLM(대형 언어 모델)을 기반으로 하여 성격 프로토타입을 학습하고 이를 이용한 상위 k개의 프로토타입을 검색하여 예측할 수 있도록 설계되었습니다.

- **Technical Details**: ProtoMBTI 프레임워크는 LoRA(저희들과 튜닝을 통해 경량의 인코더를 사용하여 성격 프로토타입을 통합하고 세분화된 증거를 집계하는 특징이 있습니다. 입력된 텍스트에 대해 프로토타입을 검색하고 이의 증거를 촉구 기반 투표와 크로스-다이코토미 일관성 검사를 통해 수정하는 방식을 채택합니다. 데이터 품질 필터링을 적용하여 다차원 LLM 증대(semantic, linguistic, sentiment)를 통해 균형 잡힌 잘 구성된 코퍼스를 구축하였습니다.

- **Performance Highlights**: Kaggle 및 Pandora 벤치마크에서 ProtoMBTI는 강력한 신경망 및 LLM 기반 모델에 대한 평균 정확도가 각각 85.14%와 96.41%로, 이전 연구보다 7.35% 및 30.64% 향상되었습니다. 이러한 성과는 모델이 매우 큰 LLM의 컴퓨팅 부하를 줄이며 심리적 통찰과 일관성을 유지하는 결과를 보임을 의미합니다. 특히 여러 데이터셋 간의 일반화가 강력하여, 성격 모델링의 정확성 및 해석 가능성을 크게 향상시킨다는 것을 보여줍니다.



### End-to-End Framework Integrating Generative AI and Deep Reinforcement Learning for Autonomous Ultrasound Scanning (https://arxiv.org/abs/2511.00114)
- **What's New**: 이 논문에서는 심장 초음파(US) 스캐닝을 자동화하고 재현 가능하게 하기 위한 최초의 종단 간(end-to-end) 프레임워크를 제시합니다. 이 프레임워크는 생성형 AI와 딥 강화 학습(DRL)을 통합하여 교육을 통해 심장 US 환경을 모델링하는 시뮬레이터와 자율 스캐닝 정책을 학습하는 두 가지 주요 구성 요소로 이루어져 있습니다. 이 기술은 심장 건강 평가에 있어 일관된 접근을 제공하며, 특히 제한된 전문 인력으로 인한 접근성을 개선하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: 제안된_Framework에서는 Generative Adversarial Networks (GANs)와 Variational Autoencoders (VAEs)를 결합한 조건부 생성 시뮬레이터가 사용됩니다. 이 시뮬레이터는 심장 US 환경을 모델링하여 현실적인 행동 조건 이미지(action-conditioned images)를 생성합니다. 또 다른 구성 요소는 DRL 모듈로, 이 시뮬레이터를 활용하여 자율적이고 정확한 스캐닝 정책을 학습합니다. 이러한 접근 방식은 기존의 데이터 부족 문제와 간단화된 모델의 한계를 극복하는 데 중점을 둡니다.

- **Performance Highlights**: 제안된 시스템은 여러 실험을 통해 검증되었으며, VAE-GAN의 성능은 기존 GAN 변종과 비교하여 정성적 및 정량적으로 평가되었습니다. DRL 기반 스캐닝 시스템은 다양한 구성 하에서 평가되어 효과성을 입증했습니다. 시스템의 설계는 카디오그램의 질에 대한 지속적인 평가를 포함하여 이미지 분류 및 품질 평가를 위한 AI 기반 가이드를 제공합니다.



### Real-DRL: Teach and Learn in Reality (https://arxiv.org/abs/2511.00112)
Comments:
          37 pages

- **What's New**: 이 논문은 안전이 중요한 자율 시스템을 위해 실시간으로 학습하는 Deep Reinforcement Learning (DRL) 에이전트를 기반으로 하는 Real-DRL 프레임워크를 소개합니다. 안전을 최우선으로 하여 안전하고 고성능의 행동 정책을 개발할 수 있도록 하며, DRL-Student, PHY-Teacher 및 Trigger라는 세 가지 상호작용 구성 요소로 이루어져 있습니다. 특히, PHY-Teacher는 안전-critical 기능에 중점을 둔 물리 모델 기반 정책 설계를 제공합니다.

- **Technical Details**: Real-DRL은 DRL-Student가 자기 학습 및 가르치는 학습 패러다임에서 실시간 안전 정보를 반영한 배치 샘플링을 통해 성장할 수 있도록 돕습니다. PHY-Teacher는 안전성을 보장하는 물리 모델 기반 설계로, DRL-Student의 학습을 지원하고 실제 시스템의 안전성을 보장합니다. Trigger는 DRL-Student와 PHY-Teacher 간의 상호작용을 관리하여 실시간 안전 상태를 모니터링합니다.

- **Performance Highlights**: 실험에서는 실제 사족 로봇, NVIDIA Isaac Gym의 사족 로봇, 그리고 카트-폴 시스템을 사용했습니다. 이 결과들은 Real-DRL이 안전성을 보장하면서도 고성능을 발휘할 수 있음을 입증합니다. 또한, 자동 계층 학습과 Safety-informed Batch Sampling을 통해 다양한 환경에서 안전과 성능 간의 균형을 이룹니다.



### Chain of Time: In-Context Physical Simulation with Image Generation Models (https://arxiv.org/abs/2511.00110)
- **What's New**: 본 논문에서는 비전-언어 모델에서 물리 시뮬레이션을 개선하고 해석하기 위한 새로운 방법인 'Chain of Time'을 제안합니다. 이 방법은 머신 러닝의 인컨텍스트 추론(in-context reasoning) 및 인간의 정신적 시뮬레이션에 영감을 받아 개발되었습니다. 특히, Chain of Time은 추가적인 파인튜닝 없이 추론 시점에서 사용되며, 합성 및 실제 도메인에서 적용됩니다.

- **Technical Details**: Chain of Time 방법은 입력 이미지의 시뮬레이션 과정에서 생성된 일련의 중간 이미지를 통해 물리적 특성(velocity, acceleration, fluid dynamics, 그리고 conservation of momentum)을 평가합니다. 이 방법은 상태-of-the-art 이미지 생성 모델의 성능을 상당히 향상시키는 것으로 나타났으며, 이미지 모델이 각 시간 단계에서 시뮬레이션하는 세계의 특정 상태를 분석하여 전통적인 물리적 추론 평가에서 숨겨진 통찰력을 제공하고 있습니다.

- **Performance Highlights**: Chain of Time을 적용함으로써 대상 IGM의 물리 추론 능력을 개선하였으며, 다양한 특정 메트릭에서 더 정확한 이미지를 생성할 수 있게 되었습니다. 또한, 이 성장 과정에서 모델이 성공하는 측면과 어려움을 겪는 측면을 분석하여, 물리적 세계를 시뮬레이션하는 과정에 대한 새롭고 자세한 통찰을 제공합니다.



### Pelican-VL 1.0: A Foundation Brain Model for Embodied Intelligenc (https://arxiv.org/abs/2511.00108)
- **What's New**: Pelican-VL 1.0은 70억에서 720억 개 파라미터까지 설정 가능한 새로운 개방형 소스의 embodied brain 모델 시리즈로, 이전 모델보다 데이터와 지능적 적응 학습 메커니즘을 깊이 통합한 것이 특징입니다. 특히, 대규모 1000개 이상의 A800 GPU 클러스터에서 훈련되어 성능이 20.3% 향상되었고, 100B급 오픈 소스 모델보다 10.6% 성능에서 앞섰습니다. 이 모델의 배경에는 인간 메타인지에서 영감을 얻은 새로운 DPPO 프레임워크가 있습니다.

- **Technical Details**: Pelican-VL 1.0은 약 4억 개 이상의 토큰을 포함하는 데이터셋에서 고품질 데이터를 증류하여 훈련되었습니다. DPPO(Deliberate Practice Policy Optimization) 프레임워크는 AI에게 의도적으로 연습하도록 가르치는 메타루프를 통해 운영됩니다. 여기에서 강화 학습(Reinforcement Learning, RL)이 두 가지 역할을 수행하는데, 첫째는 기술 세련을 위해 권장 정책(reference policy)과 목표 정책(target policy)을 정렬하는 것이고, 둘째는 자가 약점 탐지를 위한 것입니다.

- **Performance Highlights**: Pelican-VL 1.0은 다양한 실제 작업을 통해 검증되었으며, 여기에는 센서모터 루프를 닫은 최초의 VLM인 접촉이 풍부한 촉각 조작, 피킹앤플레이스를 위한 작업 지향의 affordance 추론, 그리고 다양한 로봇 플랫폼을 제어하는 단일 뇌에 의한 산업 최초의 장기 계획이 포함됩니다. 또한, 50,000 GPU 시간의 훈련 예산을 안정적으로 소화하며 25.7%의 공간 이해 및 15.1%의 시간적 추론 향상을 달성했습니다.



### AI Powered High Quality Text to Video Generation with Enhanced Temporal Consistency (https://arxiv.org/abs/2511.00107)
- **What's New**: MOVAI(다중 모드 원본 비디오 AI)는 텍스트로부터 비디오를 생성하기 위한 새로운 계층적 프레임워크로, 구성 장면 이해(compositional scene understanding)와 시간 인지(diffusion models)를 통합하여 고충실도의 비디오 합성을 가능하게 합니다. 이 프레임워크는 텍스트 설명을 계층적 장면 그래프로 분해하는 Compositional Scene Parser(CSP), 프레임 간의 일관성 있는 움직임을 보장하는 Temporal-Spatial Attention Mechanism(TSAM), 비디오 품질을 점진적으로 향상시키는 Progressive Video Refinement(PVR) 모듈을 포함한 세 가지 주요 혁신을 소개합니다.

- **Technical Details**: MOVAI는 입력 텍스트 TT와 선택적 조건 입력(스타일, 기간, 해상도)을 바탕으로 비디오 시퀀스 V를 생성하는 조건부 생성 문제로 설정됩니다. 전체 시스템은 세 가지 상호 연결된 모듈로 구성되며, 입력 처리(Input Processing), 장면 이해(Scene Understanding), 주의 처리(Attention Processing), 비디오 생성(Video Generation) 단계로 이루어져 있습니다. 각 단계에서는 고유한 기술적 접근 방식을 통해 장면의 객체 및 관계를 세밀하게 모델링하며, 모든 프레임 간의 시간적 일관성을 유지합니다.

- **Performance Highlights**: MOVAI는 다양한 평가 메트릭에서 기존 방법에 비해 15.3%의 LPIPS, 12.7%의 FVD 및 18.9%의 사용자 선호도 개선을 포함하여 최첨단 성능을 입증했습니다. 특히, 복잡한 다중 객체 장면을 생성할 때 강점을 보이며, 사용자들로부터 더 일관되게 선호되는 결과를 제공합니다. 이는 단순히 무엇이 비디오에 나타나야 할지뿐만 아니라, 객체가 시간에 따라 어떻게 이동하고 상호작용해야 하는지에 대한 훨씬 더 나은 제어를 가능하게 합니다.



### Wayfinding through the AI wilderness: Mapping rhetorics of ChatGPT prompt writing on X (formerly Twitter) to promote critical AI literacies (https://arxiv.org/abs/2511.00106)
Comments:
          Published in the journal Computers and Composition, Issue 74 (2024)

- **What's New**: 이 논문에서는 소셜 미디어에서 ChatGPT 프롬프트 작성을 연구함으로써 비판적 AI 리터러시를 촉진하는 방법을 보여줍니다. 프롬프트 작성(Prompt Writing)은 ChatGPT와 같은 생성적 AI 도구를 위해 원하는 출력을 이끌어내기 위해 지시사항을 작성하는 과정입니다. 최근 소셜 미디어에서 이와 관련된 대화가 급증하고 있습니다.

- **Technical Details**: 논문에서는 컴퓨터와 작문에서의 디지털 작문 연구의 네 가지 겹치는 전통을 바탕으로 프롬프트 작성의 사회적 수사(Social Media Rhetorics)에 대해 연구하였습니다. 2022년 11월부터 2023년 5월까지 X(구 Twitter)에서 수집한 32,000개의 포스트(post)를 분석하여 반복적인 연구 프로세스를 진행하였습니다. 이 과정에서 질적 방법(Qualitative Methods)과 계산적 방법(Computational Methods)를 혼합하여 사용하였습니다.

- **Performance Highlights**: 이 연구에서 다룬 다섯 가지 주제는 다음과 같습니다: (1) 프롬프트 작성이 영향을 미치는 커뮤니케이션 영역, (2) 프롬프트 작성을 위한 미세 리터러시 자원, (3) 프롬프트 작성을 형성하는 시장 수사, (4) 프롬프트의 수사적 특성, (5) 프롬프트 작성을 정의하는 것입니다. 우리는 이러한 주제와 방법론에 대해 논의하며 비판적 AI 리터러시를 가르치고 분석하는 디지털 작문 교사와 연구자들에 대한 중요한 시사점을 강조합니다.



### Artificial Intelligence in Elementary STEM Education: A Systematic Review of Current Applications and Future Challenges (https://arxiv.org/abs/2511.00105)
- **What's New**: 이 체계적 리뷰는 AI가 초등 STEM 교육에 미치는 영향을 종합적으로 분석한 258개의 연구(2020-2025)를 정리하고 있습니다. 연구들은 지능형 튜터링 시스템, 학습 분석, 자동 평가 등 다양한 AI 응용 프로그램을 다루며, 대부분의 연구가 상위 초등학년과 수학에 집중되어 있다는 점이 주목할 만합니다. 또한, 연구에서 제시된 주요 결핍사항들은 AI의 실제적 적용에 중요한 장벽으로 작용하고 있다는 점을 강조하고 있습니다.

- **Technical Details**: 연구는 PRISMA 가이드라인에 따라 AI 응용 프로그램을 다루는 8개 범주를 설정하였으며, 2020년부터 2025년까지의 연구를 중심으로 분석하였습니다. 특히, 지능형 튜터 시스템, 자동화된 평가, 컴퓨터 비전 등의 다양한 기술들이 초등 교육 환경에서 어떻게 구현되고 있는지를 세부적으로 조사하였습니다. 또한, 표준화된 효과 사이즈를 포함한 연구 비율이 낮고, 대부분의 연구가 북미, 동아시아, 유럽에서 진행되고 있다는 점도 주목했습니다.

- **Performance Highlights**: AI 기술이 전반적으로 효과적임을 나타내는 결과도 있지만, 특히 상위 초등학년에서만 제한되며 STEM 통합에 어려움을 겪고 있는 것으로 나타났습니다. AI가 개인화된 학습과 실시간 피드백을 제공할 수 있는 가능성이 있지만, 대부분의 연구가 수업에서 실제로 적용되는데 있어서의 장벽을 드러내었습니다. 향후 AI의 통합적 적용을 극대화하기 위해서는 상호운용 가능한 아키텍처와 선생님 중심의 구현이 필요하다는 언급도 있었습니다.



### FreeSliders: Training-Free, Modality-Agnostic Concept Sliders for Fine-Grained Diffusion Control in Images, Audio, and Video (https://arxiv.org/abs/2511.00103)
- **What's New**: 이 연구는 FreeSliders라는 새로운 접근 방식을 소개합니다. 이는 기존의 Concept Sliders(CS)의 필요조건 없이 훈련이 필요 없는 방식으로, 다양한 모달리티에 적합한 제어를 제공합니다. FreeSliders는 부분적으로 CS 공식을 추론 중에 추정함으로써 아키텍처에 종속되지 않고 유연한 컨셉 제어를 가능하게 합니다.

- **Technical Details**: FreeSliders는 기존의 CS 방식과 비교하여 개별 컨셉의 훈련 없이 동작하며, 이미지, 비디오, 오디오 등 모든 모달리티에 적용 가능합니다. 연구진은 CS 벤치마크를 확장하여 다양한 모달리티에서 섬세한 컨셉 편집을 위한 첫 번째 기준점을 설정하고, 새로운 평가 지표를 도입하여 평가 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과, FreeSliders는 훈련 없이도 다양한 모달리티 간의 컨셉 제어를 가능하게 하여 기존의 기준들보다 개선된 성능을 보였습니다. 또한, 자동 포화 감지 및 탐색 제공 방식을 통해 비선형 탐색 문제를 해결하고, 감각적으로 일관된 변화를 실현함으로써 보다 직관적인 사용자 경험을 제공합니다.



### Automated Discovery of Conservation Laws via Hybrid Neural ODE-Transformers (https://arxiv.org/abs/2511.00102)
Comments:
          5th Math-AI Workshop - Neural Information Processing Systems (NeurIPS 2025)

- **What's New**: 이 논문에서는 관측 데이터에서 보존 법칙(conservation laws)을 자동으로 발견하는 하이브리드 프레임워크를 제안합니다. 제안된 접근 방식은 Neural Ordinary Differential Equation (Neural ODE), Transformer, 기호-수치 검증기(symbolic-numeric verifier)를 통합하여 노이즈가 있는 궤적 데이터에서 보존된 양을 찾아냅니다. 특히 이 방법은 기존의 궤적 데이터 기반 접근 방식보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 모듈로 구성됩니다: 역학 학습 모듈(dynamics learning module), 기호 후보 생성기(symbolic candidate generator), 기호-수치 검증기(symbolic-numeric verifier)입니다. Neural ODE를 통해 학습한 연속 벡터 필드(continuous vector field)를 바탕으로, Transformer를 사용하여 기호 표현을 찾고, 최종적으로 기호-수치 검증기로 확인합니다. 이 과정에서 마주치는 데이터 노이즈의 영향을 최소화하고, 실제 데이터를 모델링하는 데 강력한 능력을 발휘합니다.

- **Performance Highlights**: 실험은 조화 발진기(harmonic oscillator), 진자(pendulum), 2D 케플러 이체 문제(2D Kepler two-body problem)에 대해 수행되었습니다. 결과적으로 제안된 방법은 기존 모델들에 비해 뛰어난 성능을 보여주며, 성능 저하 없이 높은 노이즈에서도 70% 이상의 발견률(discovery rate)을 유지했습니다. 또한, Neural ODE 모듈을 제거할 경우 성능이 급격히 감소하는 것을 확인하여, 이 모듈의 중요성을 강조하고 있습니다.



### Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving (https://arxiv.org/abs/2511.00101)
Comments:
          26 pages including 10 pages of main text, 6 figures, 39th Conference on Neural Information Processing Systems (NeurIPS 2025)

- **What's New**: Loquetier는 Low-Rank Adaptation (LoRA)을 기반으로 한 파라미터 효율적인 미세 조정(PEFT) 기법을 통합한 새로운 프레임워크입니다. 이 프레임워크는 LoRA의 미세 조정과 서빙을 단일 런타임 내에서 통합하여 성능과 유연성 모두에서 기존 솔루션을 능가합니다. 오늘날의 대규모 언어 모델을 사용하여 여러 하위 작업에 적용할 수 있도록 설계되었습니다.

- **Technical Details**: Loquetier는 두 가지 주요 구성 요소를 포함하고 있습니다: (1) 여러 어댑터를 공유하는 기본 모델에 적용할 수 있는 Virtualized Module, (2) 미세 조정(fine-tuning) 경로와 추론(inference) 경로를 병합한 최적화된 계산 흐름입니다. 이러한 설계는 효율적인 배치 처리와 커널 호출 오버헤드 최소화를 가능하게 합니다.

- **Performance Highlights**: 세 가지 작업 설정에서의 광범위한 실험 결과, Loquetier는 기존 기준 모델을 지속적으로 초과하여 성능과 유연성을 보여주었습니다. 특히, 추론 전용 작업에서 최첨단 동시 서빙 시스템과 비교해 최대 3.0배의 처리량을 달성하고, 통합된 미세 조정 및 추론 작업에서는 PEFT 대비 46.4배 높은 서비스 수준 목표(SLO) 달성을 기록했습니다.



### A generative adversarial network optimization method for damage detection and digital twinning by deep AI fault learning: Z24 Bridge structural health monitoring benchmark validation (https://arxiv.org/abs/2511.00099)
Comments:
          21 pages, 23 figures, published in Structural and Multidisciplinary Optimization

- **What's New**: 이번 연구에서는 최적화 기반의 손상 탐지와 디지털 트윈(digital twinning) 기술을 결합한 새로운 조건부 라벨 생성적 적대 신경망(conditional-labeled generative adversarial network) 방법론을 제안합니다. 이 프레임워크는 시스템의 상태에 대한 사전 정보 없이도 현재의 결함 이상 탐지 방법보다 우수한 성능을 보여, 실제 응용 분야에서 중요한 의미를 갖습니다. 기존의 인공지능 기반 디지털 트윈 접근법이 직면한 불확실성을 해결하는 데 중점을 두었습니다.

- **Technical Details**: 새로운 무감독(unsupervised) 프레임워크는 스위스 Z24 브리지의 구조 건강 모니터링 측정값을 사용하여 철저히 검증되었습니다. 이 방법은 동일한 손상 수준의 측정값을 입력으로 사용하여 모델이 두 가지 손상 상태로 조건부 수렴하도록 강제하고, 이후 다른 측정 그룹에 대해 이 과정을 반복하여 서로 다른 손상 상태를 구분합니다. 손상 없는 데이터와 손상 있는 데이터 모두에 대해 디지털 트윈을 위한 측정값을 생성할 수 있습니다.

- **Performance Highlights**: 제안된 접근법은 건강 측정값을 기반으로 손상을 정확하게 포착하여, 진동(vibration) 기반 시스템 모니터링과 확장 가능한 인프라 복원력을 위한 강력한 도구로 기능합니다. 추가적으로 서포트 벡터 머신(classifier)과 주성분 분석(principal component analysis) 절차가 개발되어, 각 손상 카테고리의 측정값을 평가하는 새로운 역학 학습 지표로 활용됩니다. 이로 인해 건강 상태와 손상 상태 간의 패턴 인식 및 기계 학습 데이터 생성을 가능하게 합니다.



### A filtering scheme for confocal laser endomicroscopy (CLE)-video sequences for self-supervised learning (https://arxiv.org/abs/2511.00098)
- **What's New**: 이 연구에서는 Self-Supervised Learning (SSL) 방식이 Confocal Laser Endomicroscopy (CLE) 이미지의 프리트레이닝에 효과적이라는 것을 최초로 보여줍니다. 또한, CLE 비디오 시퀀스를 위한 새로운 데이터 필터링 방법을 제안하여 프리트레이닝 과정의 계산 시간을 단축하면서 성능 저하 없이 진행할 수 있습니다. 실험 결과, 제안된 SSL 사전 훈련 모델이 전통적인 비선형 전이 학습 방법을 초월하는 향상된 정확도를 보였습니다.

- **Technical Details**: CLE는 비침습적(real-time imaging)이며, 이 연구는 대규모 레이블이 없는 데이터 세트를 사용하여 SSL 모델을 프리트레인합니다. 비디오 시퀀스의 높은 프레임 간 상관관계를 활용하여 CLE-ViFi라는 비디오 필터링 알고리즘을 통해 데이터 중복성을 줄입니다. 모델 훈련에는 작은 구성의 ViT-small(vision transformer small backbone)을 사용하며, AdamW 옵티마이저로 훈련합니다.

- **Performance Highlights**: 이 연구는 SNT( sinonasal tumors) 및 SCCS(squamous cell carcinoma of the skin) 데이터 세트에서 제안한 SSL 프리트레인 모델이 각각 67.48% 및 73.52%의_test accuracy_를 기록하며 기존 비 SSL 기준 모델을 상당히 초과했다고 보고했습니다. 또한, 우리의 접근 방식은 전체 훈련 시간을 67% 감소시키면서 훈련 효율성을 극대화하였습니다.



### GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation (https://arxiv.org/abs/2511.00097)
Comments:
          Accepted by the Main Track of NeurIPS-2025

- **What's New**: 본 논문에서는 그래프 도메인 증가 학습(Graph Domain-Incremental Learning)이라는 새로운 접근법을 소개합니다. 기존의 그래프 증가 학습 방식은 단일 도메인 내에서 작업과 클래스를 증가시키는 데 한정되어 있었지만, 본 기법은 다양한 그래프 도메인에서 학습이 이루어질 수 있도록 합니다. 제안된 GraphKeeper는 지식을 보존하고 왜곡 있는 임베딩을 방지함으로써 재앙적 망각(catatrophic forgetting) 문제를 해결합니다.

- **Technical Details**: GraphKeeper는 그래프 도메인 간의 관계를 명확히 구분하기 위해 도메인 특화 파라미터 효율적 조정(domain-specific parameter-efficient fine-tuning) 및 내부 및 외부 도메인 분리(intra- and inter-domain disentanglement) 기법을 도입합니다. 또한, 의사 결정 경계를 안정적으로 유지하기 위해 편차 없는 지식 보존(deviation-free knowledge preservation)을 사용하고, 관찰할 수 없는 도메인에서는 도메인 인식 분포 판별(domain-aware distribution discrimination)을 수행하여 정밀한 임베딩을 보장합니다.

- **Performance Highlights**: 실험 결과 GraphKeeper는 기존 방법보다 6.5%에서 16.6% 향상된 성능을 보여주었으며, 망각은 미미한 수준으로 나타났습니다. 또한 다양한 그래프 기본 모델(GFMs)과의 통합이 가능하여, 실제 활용 가능성이 광범위함을 강조합니다. 이는 기존의 그래프 증가 학습 방식이 겪던 한계를 극복할 수 있는 가능성을 시사합니다.



### Urban-MAS: Human-Centered Urban Prediction with LLM-Based Multi-Agent System (https://arxiv.org/abs/2511.00096)
Comments:
          Accepted to The 3rd ACM SIGSPATIAL International Workshop on Advances in Urban AI (UrbanAI'25)

- **What's New**: 이번 논문에서는 Urban-MAS라는 새로운 제안이 소개되었으며, 이는 인공지능(AI) 기반의 다중 에이전트 시스템(MAS)을 활용하여 제로샷(Zero-shot) 환경에서 사람 중심 도시 예측을 개선합니다. 이 프레임워크는 Predictive Factor Guidance Agents, Reliable UrbanInfo Extraction Agents, Multi-UrbanInfo Inference Agents의 세 가지 에이전트 유형으로 구성되어 있어, 각 도시 예측 작업의 핵심 요소를 추출하고 정보를 견고하게 만듭니다. 새로운 접근 방식으로 Urban-MAS는 이전의 단일 LLM 방법과 비교하여 예측의 정확성을 개선하여, 정책 결정 과정에서 신뢰할 수 있는 정보를 제공합니다.

- **Technical Details**: Urban-MAS는 도시 예측을 위해 세 가지 에이전트 레이어를 포함합니다. Predictive Factor Guidance Agents는 주요 예측 요소를 식별하고 지식 추출을 이끌어냅니다. Reliable UrbanInfo Extraction Agents는 여러 출력을 생성하고 일관성을 검증하여 안정성을 높이며, Multi-UrbanInfo Inference Agents는 다양한 출처에서 통합된 정보를 바탕으로 예측을 수행합니다. 각 에이전트는 서로 협력하여 도시 데이터를 효율적으로 처리하고, 깊이 있는 연구를 통해 핵심 예측 요소를 정리하여 신뢰할 수 있는 결과를 도출합니다.

- **Performance Highlights**: 실험 결과 Urban-MAS는 도쿄, 밀라노, 시애틀 등 세 도시의 도시 인식 및 실행 금액 예측에서 단일 LLM 기준선보다 오류를 획기적으로 줄였습니다. 특히, Predictive Factor Guidance Agents가 예측 성능 향상에 가장 중요한 요소로 작용하여, Urban-MAS가 인간 중심 도시 AI 예측의 확장성 있는 패러다임으로 자리잡을 수 있도록 합니다. 전체적으로 Urban-MAS는 제로샷 조건에서도 효율적이고 비용 효과적인 성과를 달성하며, 인간 중심 도시 AI 연구의 발전에 기여할 것으로 기대됩니다.



### SpinalSAM-R1: A Vision-Language Multimodal Interactive System for Spine CT Segmentation (https://arxiv.org/abs/2511.00095)
Comments:
          2 Tables,5 Figures,16 Equations

- **What's New**: 이번 연구에서는 SpinalSAM-R1이라는 새로운 다중 모드 비전-언어 인터랙티브 시스템을 제안합니다. 이 시스템은 세그먼트 아무거나 모델(Segment Anything Model, SAM)과 DeepSeek-R1을 통합해 척추 CT 이미지 세분화 작업을 수행합니다. 새로운 해부학 유도 주의 메커니즘과 자연어 기반의 세분화 정제를 가능하게 하는 세멘틱스 기반 상호작용 프로토콜을 소개합니다. 이를 통해 복잡한 척추 구조의 세분화 성능을 개선하고, 효율적인 적응을 위해 Low-Rank Adaptation(LoRA)을 사용하여 미세 조정하였습니다.

- **Technical Details**: SpinalSAM-R1은 세 가지 주요 구성 요소로 구성된 3계층 아키텍처를 갖추고 있습니다. 사용자 인터페이스 계층은 포인트, 바운딩 박스 및 자연어 명령을 지원하여 직관적인 임상 상호작용을 제공합니다. 비즈니스 로직 계층은 DeepSeek-R1 모듈을 통합하여 의미론적 지침을 세분화 프롬프트로 동적으로 해석하고 실시간으로 컨텍스트 인식 마스크 정제를 제공합니다. 이러한 구조는 고속 의료 이미지 처리의 요구를 충족하기 위해 GPU 가속 계산을 활용합니다.

- **Performance Highlights**: SpinalSAM-R1은 120개의 요추 CT 스캔으로 구성된 임상 데이터 세트에서 엄격하게 평가되었으며, 주의계수(Dice coefficient)는 0.9532, 교차 면적 비율(IoU)은 0.9114에 도달하여 U-Net, TransUNet, SAM-Med2D와 같은 최신 방법들을 초월하는 성능을 보였습니다. 또한 DeepSeek-R1 모듈은 11개의 임상 작업 유형에 대해 94.3%의 명령 구문 정확도를 달성하고 800ms 이하의 응답 시간을 유지하여 사용자 편의성과 상호작용의 효율성을 극대화했습니다.



### Digital Twin based Automatic Reconfiguration of Robotic Systems in Smart Environments (https://arxiv.org/abs/2511.00094)
Comments:
          Accepted for presentation to 11th IEEE International Smart Cities Conference (ISC2 2025)

- **What's New**: 이 논문은 로봇 제어 시스템을 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 Digital Twin 기술을 활용하여 로봇의 운영 환경을 가상으로 복제하여 시뮬레이션 및 최적화합니다. 이를 통해 실시간으로 환경 변화에 적응할 수 있도록 하여 수동 개입 없이 신속하고 신뢰성 높은 재구성을 보장합니다.

- **Technical Details**: 스마트 환경에서의 로봇 시스템의 재구성을 위한 Digital Twin의 개념은 지속적으로 발전하고 있습니다. 이번 연구에서는 사용자로부터 새로운 물체 추가 및 기존 물체 제거와 같은 정보를 수집하여 로봇의 이동 계획을 수정하고 이를 실제 로봇으로 전송하는 방법을 소개합니다. 여기서 경로 계획, 환경 모델링 및 실행 단계가 통합되어 있으며, SLAM 알고리즘과 기존의 경로 계획 알고리즘이 활용됩니다.

- **Performance Highlights**: Unity3D와 Gazebo와 같은 시뮬레이션 플랫폼을 비교하며, Unity의 고품질 렌더링과 빠른 프로토타입 제작의 장점을 강조합니다. Gazebo는 물리 시뮬레이션에서의 정확성과 ROS 통합의 강점을 보여줍니다. 본 연구는 통합된 Digital Twin과 자율적 시스템 재구성을 통해 로봇이 스마트 환경에 적응할 수 있도록 하는 새로운 경로를 제시합니다.



### LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation (https://arxiv.org/abs/2511.00090)
Comments:
          NeurIPS 2025

- **What's New**: LeMiCa는 훈련이 필요 없는 효율적인 확산 기반 비디오 생성 가속 프레임워크를 제시합니다. 기존의 캐싱 전략이 부분적인 휴리스틱 오류 감소에 중점을 둔 반면, LeMiCa는 전역 오류 누적을 방지하여 생성된 비디오의 전반적인 일관성을 크게 향상시킵니다. 이 접근법은 전역 콘텐츠와 스타일의 일관성을 개선할 수 있으며, 다양한 테스팅 벤치마크에서 뛰어난 성능을 입증했습니다.

- **Technical Details**: LeMiCa는 캐시 스케줄링을 오류 가중 경로가 있는 방향 그래프로 형식화하고, 최악의 경로 오류를 명시적으로 제한하는 Lexicographic Minimax Path Optimization 전략을 도입합니다. 이 메트릭을 바탕으로, LeMiCa는 Directed Acyclic Graph(DAG)를 구성하여 각 엣지가 최종 출력 품질에 미치는 영향을 반영합니다. 이 방식은 로컬 그리디 전략 대신 전체 경로 계획 문제로 접근하여, 전역 오류를 효율적으로 관리합니다.

- **Performance Highlights**: LeMiCa는 Latte 모델에서 2.9배의 속도 향상과 Open-Sora에서 0.05의 LPIPS 점수를 기록하며, 기존 캐싱 기법보다 우수한 성능을 보입니다. 이러한 개선 사항은 인지 품질 손실을 최소화하면서 이루어지며, 비디오 생성의 효율적이고 신뢰할 수 있는 기초로 자리잡을 수 있습니다. 여러 기본 모델에서 인퍼런스 속도와 생성 품질의 이중 개선을 달성했습니다.



### Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Ta (https://arxiv.org/abs/2511.00088)
- **What's New**: Alpamayo-R1 (AR1)는 진화적인 자율 주행 기술로, 비전-언어-행동 모델(VLA)을 사용하여 복잡한 운전 시나리오에서의 결정을 향상시킵니다. 이를 위해 체인 오브 코제이션(Chain of Causation, CoC) 추론과 궤적 계획이 통합되어 있으며, AR1은 강력한 훈련 전략과 함께 다양한 혁신을 포함하고 있습니다. 기존 E2E(End-to-End) 시스템의 한계를 극복하기 위해 모든 요소가 상호 연결된 데이터와 구조화된 추론 과정을 통합하고 있습니다.

- **Technical Details**: AR1의 핵심 혁신 중 하나는 CoC 데이터 세트로, 이 데이터는 하이브리드 오토라벨링과 인간의 개입을 통해 구축됩니다. 또한, Cosmos-Reason이라는 비전-언어 모델과 확산 기반의 궤적 디코더를 결합한 모듈식 VLA 아키텍처가 포함되어 있습니다. 이러한 설계는 안전 및 장기적인 상황에서의 의사 결정 능력을 강화하며, 다단계 훈련 전략을 통해 최적화된 추론 품질을 이끌어내고 있습니다.

- **Performance Highlights**: AR1은 복잡한 주행 시나리오에서 계획 정확도를 12% 향상시키고, 오프로드 비율을 35%, 근접 접촉 비율을 25% 낮추는 효율성을 보였습니다. RL(강화 학습) 후 훈련을 통해 추론 품질이 45% 개선되었으며, 실제 도로 테스트 결과 99ms의 낮은 지연 시간으로 실시간 성능을 입증하였습니다. 이러한 결과는 AR1이 자율주행 4단계(Level 4) 달성을 위한 실질적인 경로를 제시하고 있음을 보여줍니다.



### Adding New Capability in Existing Scientific Application with LLM Assistanc (https://arxiv.org/abs/2511.00087)
Comments:
          8 pages, 4 figures, submitted to The 1st International Workshop on Foundational large Language Models Advances for HPC in Asia

- **What's New**: 이 논문에서는 새로운 알고리즘을 위한 코드 생성을 자동화하는 방법론을 제시합니다. 특히, 이전의 유사 코드 예제가 없는 새로운 알고리즘에 대한 코드 생성을 다룹니다. LLM(대형 언어 모델) 지원을 이용한 새로운 코드 생성 방법을 설명하며, 기존의 코드 변환 도구인 Code-Scribe를 개선하는 내용을 포함합니다.

- **Technical Details**: 제안된 방법론은 자연어를 통해 모델과 상호작용하는 새로운 프롬프트 엔지니어링 기술에 기반합니다. 이 방법은 사용자가 모델이 이해하는 내용을 반복적으로 확인하고 수정할 수 있도록 하여, 모델의 착각(hallucination) 가능성을 줄입니다. Code-Scribe는 기존에 Fortran 코드를 C++로 변환하기 위해 설계되었으며, 다양한 단계의 코드를 조직화하고 매핑하는 데 도움을 줍니다.

- **Performance Highlights**: 실험 결과, 제안된 접근법을 통해 코드 생성의 정확성과 효율성이 향상되었습니다. 개발한 새로운 통신 알고리즘은 복잡한 물리학 시뮬레이션에서 자주 사용되는 기존의 방법보다 성능적인 이점을 가지며, 가상의 입자를 활용하여 처리 속도를 개선하고 불필요한 계산 비용을 줄이는 결과를 보여줍니다.



### Generalizing Test-time Compute-optimal Scaling as an Optimizable Graph (https://arxiv.org/abs/2511.00086)
Comments:
          Under review

- **What's New**: 본 논문에서는 Test-Time Scaling (TTS)을 통해 대형 언어 모델 (LLMs)의 성능을 극대화하는 새로운 문제를 제안합니다. 기존의 연구는 정적인 아키텍처와 단일 모델 사용에 집중되었으나, 작업에 따라 최적의 아키텍처와 모델 조합이 다를 수 있다는 점을 간과했습니다. 이를 해결하기 위해 저자들은 컴퓨팅 최적화를 위한 모델 및 아키텍처 조합의 검색을 다루는 Multi-LLM Collaboration Graph를 형식화했습니다.

- **Technical Details**: 제안된 방법론인 Agent-REINFORCE는 확률적 최적화를 통해 최적의 Multi-LLM Collaboration Graph를 탐색합니다. 이 과정에서 노드는 LLM 모델과 역할을 나타내고, 엣지는 정보 흐름을 포착합니다. REINFORCE 알고리즘을 기반으로 하여, LLM 기반 에이전트가 경험적 통찰을 반영하여 후보 초기화 및 분포 업데이트를 실행합니다.

- **Performance Highlights**: 실험 결과, Agent-REINFORCE는 전통적인 방법 및 LLM 기반 기준선보다 샘플 효율성과 탐색 성능에서 우수한 성과를 보였습니다. 또한, 정확성과 추론 지연을 동시에 최적화하는 그래프를 효과적으로 식별해내는 데 성공했습니다. 본 연구는 작업별 요구 사항에 따른 동적 구조 최적화의 필요성을 강조합니다.



### MaGNet: A Mamba Dual-Hypergraph Network for Stock Prediction via Temporal-Causal and Global Relational Learning (https://arxiv.org/abs/2511.00085)
- **What's New**: 이 논문은 주식 예측을 위한 새로운 Mamba 이중 하이퍼그래프 네트워크(MaGNet)를 소개합니다. MaGNet은 세 가지 혁신적인 요소를 통합하여 동적 상관 관계 모델링 및 시계열 의존성 포착을 가능하게 합니다. 주식 간 관계의 복잡성을 해결하기 위해 MAGE 블록과 여러 가지 Attention 모듈이 도입되었으며, 시간적 및 인과적 관계를 모두 고려하는 이중 하이퍼그래프 구조를 사용하고 있습니다.

- **Technical Details**: MaGNet은 MAGE 블록을 통해 양방향 Mamba 모델과 적응형 게이팅 메커니즘을 결합하여 시계열 데이터를 효과적으로 처리합니다. 또한, 특성-wise 및 주식-wise 2D Spatiotemporal Attention 모듈을 적용하여 다양한 시장 조건에 적응할 수 있는 능력을 극대화합니다. 최종적으로, Temporal-Causal Hypergraph(TCH)와 Global Probabilistic Hypergraph(GPH)를 사용하여 주식 간의 고차원 관계를 모델링하고, 이를 통해 동적인 관계를 효과적으로 분리합니다.

- **Performance Highlights**: 여섯 개 주요 주가 지수에 대한 광범위한 실험 결과, MaGNet은 예측 정확도와 투자 수익률 모두에서 기존의 최첨단 방법들을 초월하는 성과를 보여주었습니다. 예를 들어, CSI 300에서 최대 54.9%의 예측 정확도를 기록했으며, 여러 시장에서도 Sharpe 비율이 1.0을 초과하는 견고한 성과를 달성했습니다. 연간 수익률 또한 22.6%에 달하며 뛰어난 리스크 관리 능력을 보여줍니다.



### Application of predictive machine learning in pen & paper RPG game design (https://arxiv.org/abs/2511.00084)
Comments:
          Master's thesis submitted at AGH University of Science and Technology

- **What's New**: 최근 몇 년간 펜 앤 페이퍼 RPG 시장이 급성장하고 있어, 게임 회사들이 AI 기술을 통합하여 플레이어 경험을 향상시키고 경쟁 우위를 점하기 위해 탐색하고 있습니다. 특히, 몬스터의 난이도 정도를 자동으로 결정할 수 있는 방법이 부재한 상황에서, 이 논문은 이전의 수동적 방법 대신 기계 학습(ML)을 활용한 해법을 제시합니다. 몬스터 레벨 예측을 위한 전용 데이터셋 구축을 포함하여, 기존 RPG 규칙을 기반으로 한 인간 영감을 받은 모델을 개발하였습니다.

- **Technical Details**: 이 연구는 RPG 데이터에 기반한 서열 회귀(ordinal regression) 기법을 적용하여 몬스터의 레벨 추정을 위한 여러 기술적 기여를 명확히 하고 있습니다. 이를 위해 새로운 데이터셋을 구축하고, 구조화된 시간을 통해 데이터를 평가하는 평가 절차를 설계하였으며, 여러 ML 알고리즘의 효율성을 비교하기 위한 평가 기준을 설정하였습니다. Pathfinder 게임 시스템에 특화된 이 논문의 접근 방법은 적절한 플레이어 몬스터와의 균형을 맞추는 데 기여할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 이 연구에서 개발된 인간 영감을 받은 기준 모델은 Pathfinder 규칙서를 기반으로 하였으며, 구축된 데이터셋에서 평가되었습니다. 서열 회귀에 대한 최신 기법들을 적용하여 몬스터 레벨 측정 문제를 해결하고자 하였고, 이들의 성능을 분석하였습니다. 현재까지의 연구 결과는 몬스터 레벨 추정의 자동화 분야에서 의미 있는 진전을 나타내며, RPG 디자인에서의 유용한 도구로 자리매김할 가능성을 지니고 있습니다.



### Fixed-point graph convolutional networks against adversarial attacks (https://arxiv.org/abs/2511.00083)
- **What's New**: 이 논문에서는 고정점 반복 그래프 합성곱 신경망(Fix-GCN)이라는 새로운 모델을 제안하여, 그래프의 고차 노드 이웃 정보를 효과적으로 포착하여 적대적 공격(adversarial attacks)에 대한 강인성을 확보합니다. 이 모델은 추가적인 메모리나 계산 복잡성을 요구하지 않으며, 그래프 신호에서 저주파(low-frequency) 구조 정보를 보존하면서 고주파(high-frequency) 성분을 선택적으로 감소시키는 유연한 필터링 접근 방식을 채택합니다. 이를 통해 Fix-GCN은 중요한 그래프 정보를 보호하면서 적대적 조작의 영향을 줄이는 효율적인 프레임워크를 제공합니다.

- **Technical Details**: Fix-GCN의 핵심 메시지 전달 메커니즘은 고정점 반복(fixed-point iteration)을 사용하여 그래프 필터링 시스템을 해결하여 도출됩니다. 제안된 네트워크는 강력한 구조로 설계되어 적대적 변동에 저항하며, 고주파 성분을 선택적으로 감소시키는 유연한 차단 필터를 통해 저주파 구조 정보를 보존합니다. 이러한 선택적 감쇠 덕분에 모델은 적대적 조작에 대응할 때에도 강력하고 회복력이 있습니다.

- **Performance Highlights**: 제안된 모델은 다양한 기준 그래프 데이터셋에서 경쟁력 있는 기준선 모델들을 능가하는 실험 결과로 그 강인성을 입증하였습니다. Fix-GCN은 고차 이웃으로부터 정보를 수집하여 피해를 줄이는 특별한 능력을 보유하고 있으며, 이는 표적 노드의 인접 노드에서의 직접적인 변동에서 오는 위험을 완화하는 데 중요한 역할을 합니다. 전반적으로 Fix-GCN은 다양한 적대적 공격 상황에서도 일관된 성능을 보여주며, 그래프 신경망의 신뢰성과 강건성을 증진시키는 중요한 기여를 합니다.



### RailEstate: An Interactive System for Metro Linked Property Trends (https://arxiv.org/abs/2511.00078)
- **What's New**: RailEstate는 메트로 시스템 접근성이 주택 시장에 미치는 영향을 분석하는 웹 기반 도구로, 공간 분석(spatial analytics), 자연어 인터페이스(natural language interfaces), 상호 작용 예측(interactive forecasting)을 통합한 혁신적인 시스템입니다. 이 시스템은 사용자가 메트로 정류소 근처의 주택 가격에 대한 ZIP 코드 수준의 패턴을 탐색할 수 있도록 돕고, 자연어 질문을 SQL로 변환하여 실행할 수 있는 챗봇 기능을 포함하고 있습니다. 이를 통해 도시 계획자, 투자자 및 거주자들이 기술적 전문 지식 없이도 실시간으로 분석과 통찰력을 얻을 수 있습니다.

- **Technical Details**: RailEstate는 React-Leaflet 프론트엔드, FastAPI 백엔드, LangChain을 통한 텍스트-SQL 변환 엔진 및 주택 가격과 메트로 데이터를 포함한 PostGIS 데이터베이스로 구성된 레이어드 아키텍처를 사용합니다. 시스템은 ZIP 코드 수준에서의 주택 가격 데이터, 메트로 역 정보 및 지역 경계를 통합하여, 지리적 데이터에 대한 빠른 쿼리를 지원합니다. 각 메트로역 주변의 주택 가격 변화와 구조물 투자 파급효과를 동적으로 시각화할 수 있도록 설계되었습니다.

- **Performance Highlights**: RailEstate는 25년의 주택 데이터를 기반으로 한 시간 시계열 분석을 통해 주택 시장의 장기적인 변화를 명확히 시각화합니다. 사용자는 메트로 정류소에 따라 가격 변동을 실시간으로 탐색하고, AI 기반의 예측 모형을 통해 향후 주택 가치를 예측할 수 있습니다. 이 시스템은 저렴한 가격 대역을 시각적으로 구분하여 사용자가 쉽게 패턴을 인식할 수 있도록 하며, 자연어 질문을 통해 복잡한 쿼리를 실행하는 기능을 제공함으로써 기존의 부동산 도구들을 넘어서는 사용자 경험을 제공합니다.



### LookSync: Large-Scale Visual Product Search System for AI-Generated Fashion Looks (https://arxiv.org/abs/2511.00072)
Comments:
          4 pages, 5 figures. Accepted at the International Conference on Data Science (IKDD CODS 2025), Demonstration Track. Demo video: this https URL

- **What's New**: 이번 논문에서는 AI 생성 스타일과 유사한 제품을 빠르게 검색할 수 있는 시스템을 소개합니다. 이 시스템은 1200만 개 이상의 제품에 대한 고차원 임베딩을 생성하며, AI 생성 이미지를 키워드로 변환하여 시각적으로 유사한 제품을 찾습니다. 이 시스템은 일일 35만 개의 AI Looks를 처리하며, 이를 통해 사용자에게 더욱 개인화된 쇼핑 경험을 제공합니다.

- **Technical Details**: 제안하는 시스템은 쿼리 생성, 벡터화, 후보 검색, 재순위 매김을 포함한 4개의 주요 구성 요소로 이루어져 있습니다. AI 생성 이미지를 반영하여 가장 유사한 제품을 추출하며, CLIP 모델을 활용하여 높은 정확도로 유사도를 평가합니다. 시스템은 평균적으로 1초 이내의 응답 속도를 유지하며, 사용자 인터렉션에 기반한 추천 품질도 개선되었습니다.

- **Performance Highlights**: 제안된 시스템은 CLIP을 기반으로 하여 다양한 제품 카테고리에서 비슷한 스타일을 유지하는 제품을 효과적으로 찾아냅니다. 실험 결과, CLIP 모델이 다른 대안 모델들보다 평균 의견 점수에서 3~7% 우위를 점했습니다. 이러한 작은 개선이 사용자 경험을 크게 향상시키는 것을 보여주며, 실제 운영 환경에서 경쟁력 있는 솔루션으로 자리잡게 되었습니다.



### Benchmarking Generative AI Against Bayesian Optimization for Constrained Multi-Objective Inverse Design (https://arxiv.org/abs/2511.00070)
Comments:
          17 pages, 2 Figures

- **What's New**: 이 논문에서는 제한된 다목적 회귀(tasks) 문제를 해결하기 위해 Large Language Models (LLMs)를 생성적 최적화 도구로 활용하는 성능을 조사했습니다. 특히, 이 연구는 역설계(inverse design) 분야에서의 응용을 포함하며, 이는 재료 정보학(materials informatics)에서 필수적입니다.

- **Technical Details**: 연구는 기계 학습 알고리즘의 두 가지 접근 방식을 비교했습니다. 하나는 기존의 Bayesian Optimization (BO) 프레임워크와 다른 하나는 세밀하게 조정된 LLM과 BERT 모델입니다. BO 프레임워크에서는 BoTorch Ax 구현을 기준으로 하고, 최고의 q-Expected Hypervolume Improvement (qEHVI)를 사용하였습니다.

- **Performance Highlights**: 결과적으로, BoTorch qEHVI는 완벽한 수렴(GD=0.0)을 달성하며 성능의 한계를 설정했습니다. 반면, 가장 성능이 우수한 LLM인 WizardMath-7B는 GD 1.21을 기록하며 전통적인 BoTorch Ax 기준(GD=15.03)보다 현저히 우수한 성과를 나타냈습니다. 연구 결과는 수지, 폴리머, 도료의 제형 설계 최적화에 직접적인 산업적 응용 가능성을 제시합니다.



### Latent Domain Prompt Learning for Vision-Language Models (https://arxiv.org/abs/2511.00067)
- **What's New**: 이 논문은 도메인 일반화( Domain Generalization, DG) 문제를 해결하기 위해 명시적인 도메인 레이블 없이도 모델이 잘 일반화할 수 있도록 새로운 접근 방식을 제안합니다. 제안된 Latent Domain Prompt Fusion (LDPF) 방식은 훈련 데이터에서 자동으로 발견된 잠재적인 도메인을 결합하여 모델이 도메인 간의 지식을 적응적으로 전이할 수 있게 합니다. 이를 통해 도메인 이동에 대한 강건성을 증가시키고, 시각-언어 모델( Vision-Language Models, VLMs)의 성능 향상을 목표로 합니다.

- **Technical Details**: 이 논문에서는 다중 소스 도메인 일반화(Multi-Source Domain Generalization, MSDG) 문제를 다루며, 각 도메인에서의 데이터 분포는 서로 다릅니다. LDPF 프레임워크는 도메인 정보가 포함된 세 가지 구성 요소( 도메인 요구 개선을 위한 도메인 불균형 훈련)을 활용하여 프로프트를 최적화하는 방식으로 작동합니다. 또한, 도메인 특성 추출을 위한 MLP 네트워크와 k-means 클러스터링을 통해 도메인 레이블을 자동으로 정리하여, 훈련 과정에서 도메인 레이블의 필요성을 없앱니다.

- **Performance Highlights**: 여러 벤치마크 데이터셋( Office-Home, mini-DomainNet, PACS, Terra Incognita)에서 실험 결과, LDPF 방식이 VLM을 기반으로 하는 기존 방법들보다 일관된 성능 향상을 보여줍니다. 특히, 훈련된 도메인 세트를 기반으로 한 실험을 통해 제안된 접근 방식이 도메인 이동에 대한 견고성을 크게 개선함을 확인했습니다. 이는 특히 복잡한 현실적 환경에서의 응용 가능성을 높이는 중요한 결과로 이어집니다.



### Aligning Brain Signals with Multimodal Speech and Vision Embeddings (https://arxiv.org/abs/2511.00065)
- **What's New**: 이 논문은 인간의 언어 이해가 단지 소리 처리에 그치지 않고, 경험과 기억 등의 다양한 연관성을 통해 이루어짐을 강조합니다. 특히, 사전 훈련된 모델의 다양한 레이어가 뇌의 신경 활동과 어떻게 일치하는지를 분석합니다. 연구팀은 JEGS(EEG) 데이터와 wav2vec2, CLIP 모델의 임베딩을 활용하여 언어 이해의 다차원적 특성을 살펴보았습니다.

- **Technical Details**: 연구에서는 EEG 신호와 wav2vec2, CLIP 모델에서 추출된 임베딩을 비교합니다. 각 레이어의 임베딩을 Ridge Regression을 사용하여 신경 활동과 정렬하고, 세 가지 전략—단일 레이어, 누적 연결, 누적 합계—을 테스트하여 가장 좋은 정렬을 평가했습니다. 이러한 방법은 다중 모달(multi-modal) 이해를 향상시킵니다.

- **Performance Highlights**: 연구 결과는 다중 모달 레이어 인식을 결합하는 것이 언어 이해 과정을 해독하는 데 기여할 수 있음을 보여줍니다. 또한, 실험을 통해 최적의 레이어를 선택함으로써 EEG 디코딩 성능이 향상되었음을 확인했습니다. 이는 청각 자극에서 이전의 이해를 넘어 실제 의미의 복잡성을 이해하는 데 중요한 통찰력을 제공합니다.



### World Simulation with Video Foundation Models for Physical AI (https://arxiv.org/abs/2511.00062)
- **What's New**: [Cosmos-Predict2.5]는 물리 AI를 위한 Cosmos World Foundation Models의 최신 버전으로, 텍스트, 이미지 및 비디오 생성을 단일 모델에서 통합한 흐름 기반(Flow-based) 아키텍처를 기반으로 합니다. 이 모델은 더욱 풍부한 텍스트 그라운딩(text grounding)과 세계 시뮬레이션(world simulation)의 세밀한 제어를 가능하게 합니다. 새로운 버전은 2B 및 14B 스케일에서 출시되어 이전 모델인 [Cosmos-Predict1]보다 비디오 품질과 지침 일치(instruction alignment)에서 현저한 개선을 이루었습니다.

- **Technical Details**: [Cosmos-Predict2.5]는 2억 개의 큐레이션된 비디오 클립을 기반으로 학습되었으며, 강화 학습(reinforcement learning) 기반의 후학습(post-training) 과정으로 다듬어졌습니다. 또한, 시뮬레이션 및 정책 평가(policy evaluation)을 위한 신뢰성 높은 합성 데이터 생성(synthetic data generation)을 지원합니다. [Cosmos-Transfer2.5]는 Sim2Real 및 Real2Real 세계 번역을 위한 제어망(control-net) 스타일의 프레임워크로, 작지만 높은 충실도와 강력한 장기 비디오 생성 기능을 갖추고 있습니다.

- **Performance Highlights**: [Cosmos-Predict2.5]와 [Cosmos-Transfer2.5]는 엔터티 지능의 확장을 위한 다재다능한 도구로 자리 잡고 있습니다. 이 모델들은 비록 [Cosmos-Transfer1]보다 3.5배 작지만, 더욱 높은 fidelity와 장기 비디오 생성에서 더 나은 성능을 제공합니다. NVIDIA Open Model License하에 소스 코드, 사전 훈련 체크포인트(pretrained checkpoints) 및 큐레이팅 된 베치마크를 공개하여 물리 AI 연구 및 배포를 촉진하고 있습니다.



### Automatically Finding Rule-Based Neurons in OthelloGP (https://arxiv.org/abs/2511.00059)
Comments:
          39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop Mechanistic interpretability

- **What's New**: 이번 연구에서는 OthelloGPT라는 트랜스포머 모델을 사용하여 규칙 기반 게임 로직을 인코딩하는 MLP 뉴런을 식별하고 해석하는 자동화된 방법을 제안합니다. 특히 결정 트리를 활용하여 보드 상태에 대한 뉴런 활성화를 예측하고, 이를 통해 인간이 이해할 수 있는 논리적 형태로 뉴런을 전환할 수 있는 경로를 추출합니다. 연구 결과, 제5층의 약 절반 이상의 뉴런은 압축된 규칙 기반 결정 트리를 통해 정확히 설명될 수 있음을 보여주었습니다.

- **Technical Details**: OthelloGPT는 2,500만 개의 매개변수로 구성된 변환기 모델로, 오델로 게임에서 합법적인 수를 예측하기 위해 훈련되었습니다. 연구진은 회귀 결정 트리를 훈련시켜 뉴런의 활성화 값을 직접 예측하는 동시에, 보드 상태의 다양한 특징을 입력으로 사용하여 뉴런의 규칙 기반 패턴을 식별합니다. 이 과정에서 DNF(Disjunctive Normal Form) 형태로 각 뉴런의 해석 가능성을 높이며, 모델의 능력을 검증하기 위해 카즈얼 개입을 구현합니다.

- **Performance Highlights**: 모델의 예측 정확도를 평가하기 위해 전통적인 머신러닝 지표인 R² 점수를 활용하였으며, 제안된 결정 트리는 뉴런의 활성화를 효과적으로 예측하는 것으로 나타났습니다. 연구자들은 구체적인 패턴에 대한 뉴런을 제거하는 개입 실험을 통해 모델의 예측 능력이 약 5-10배 저하되는 결과를 얻었습니다. 이 Findings는 OthelloGPT의 규칙 기반 행동을 완벽하게 역설계하는 첫 번째 단계로 여겨집니다.



### MISA: Memory-Efficient LLMs Optimization with Module-wise Importance Sampling (https://arxiv.org/abs/2511.00056)
- **What's New**: 이 논문에서는 Module-wise Importance SAmpling (MISA)이라는 새로운 방법을 제안합니다. MISA는 각 레이어를 더 작은 모듈로 나누고, 각 모듈에 중요도 점수를 부여하여 최적화를 수행합니다. 이 접근 방식은 메모리 사용을 줄이는 동시에 더 나은 경량화 효율성을 제공하도록 설계되었습니다.

- **Technical Details**: MISA는 모듈 단위로 내부 가중치 업데이트를 수행하여 정보 손실을 최소화함으로써 더 나은 결과를 도출합니다. 연구 결과, MISA는 비선형 조건에서 0(1/ B0√K) 수렴률을 달성하며, K는 블록 업데이트의 총 수를 나타냅니다. MISA는 각 모듈의 샘플링 확률을 매개변수화하여 샘플링 전략을 최적화하는 기능을 가지고 있습니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과 MISA는 기존의 PEFT 및 레이어별 최적화 방법보다 우수한 성능을 나타냈습니다. MISA는 Commonsense Reasoning, Math Reasoning, Instruction Following 등 16개의 데이터셋에서 테스트되었으며, 이전 모델과의 비교에서 더 낮은 perplexity 점수를 기록했습니다. 이러한 결과는 MISA의 효과성을 입증합니다.



### Exploring Federated Learning for Thermal Urban Feature Segmentation -- A Comparison of Centralized and Decentralized Approaches (https://arxiv.org/abs/2511.00055)
- **What's New**: 이번 연구는 Federated Learning (FL) 모델을 실세계에서 효과적으로 구현하는 방법을 조사합니다. 특히, 무인 항공기(UAV)를 사용한 열 화상 분석을 통해 도시 환경에서 공통적인 열 특징을 탐지하는 데 초점을 맞추고 있습니다. 논문은 독일의 두 도시에서 수집된 이미지 데이터를 사용해 비동일 분포(non-IID)와 데이터 특징 차이로 인해 발생하는 문제들을 다룹니다.

- **Technical Details**: FL은 데이터 보안 문제를 해결하기 위한 접근 방법으로, 각 참여자가 로컬 데이터를 기반으로 모델을 학습하고 모델 업데이트만 중앙 서버에 전송합니다. 이 연구에서는 FL과 중앙 집중식 학습(Centralized Learning, CL) 방법을 비교하고, 모델 정확도, 학습 시간 및 에너지 사용량 같은 성능 지표를 평가합니다. 특히, 비동일 데이터 처리에 적합한 FL 집계 전략을 도입하고 평가합니다.

- **Performance Highlights**: FL의 방식을 통해, 도시 환경에서의 열 이상 탐지 작업의 효율성이 향상되었음을 보여주었으며, 이는 에너지 관련 시스템 최적화에 기여합니다. 이 연구 결과는 FL이 실제 상황에서 어떻게 유용할 수 있는지에 대한 귀중한 참고 자료가 될 것입니다. 주요 기여로는 비균형 데이터에서의 모델 성능 문제를 해결하고, 실제 데이터에서의 적용 가능성을 평가한 점이 있습니다.



### SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation (https://arxiv.org/abs/2511.00054)
Comments:
          Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop on Efficient Reasoning

- **What's New**: 이번 논문에서는 Vision-Language Models (VLMs) 의 복잡한 공간적 추론에 대한 문제를 다루기 위해, SpatialTraceGen이라는 새로운 프레임워크를 소개합니다. 이 접근법은 대형 모델의 추론 과정을 정제하여 고품질의 다단계, 다도구(reasoning traces) 추론 데이터를 생성합니다. 특히, 자동화된 Verifier가 도입되어 각 추론 단계의 정확성을 보장하고, 수작업 주석(annotation)에 대한 비용효율적인 대안을 제공합니다.

- **Technical Details**: SpatialTraceGen 프레임워크는 전문가의 추론을 고품질 트레이스로 변환하는 데 중점을 두고 있습니다. Verifier는 자동화되어 있어 데이터 생성 과정에서 단계별 추론의 신뢰성을 검사합니다. 또한, CLEVR-Humans 벤치마크에서의 검증 프로세스를 통해 평균 품질 점수를 17% 향상시키고, 품질 변동성을 40% 이상 감소시키는 성과를 달성했습니다.

- **Performance Highlights**: SpatialTraceGen을 통해 생성된 데이터셋은 효과적인 파인튜닝(fine-tuning)과 샘플 효율적인 오프라인 강화 학습(offline reinforcement learning)을 위해 필요한 구조적이고 단계적인 도구 사용 예제를 제공합니다. 이로 인해, VLMs의 성능을 크게 향상시킬 수 있는 가능성을 제시하고 있습니다.



### Quadratic Direct Forecast for Training Multi-Step Time-Series Forecast Models (https://arxiv.org/abs/2511.00053)
- **What's New**: 이 논문은 시계열 예측 모델의 훈련 목표를 정의하는 새로운 방법을 제안합니다. 기존의 평균 제곱 오차(mean squared error, MSE)와 같은 훈련 목표는 각각의 미래 단계(future step)를 독립적이고 균등한 작업으로 취급하는 경향이 있으며, 이로 인해 두 가지 주요 문제가 발생합니다. 첫째, 미래 단계 간의 레이블 자기 상관(label autocorrelation) 효과를 간과하여 훈련 목표가 편향되며, 둘째, 서로 다른 예측 작업에 따라 이질적인 가중치 설정을 실패하여 성능이 제한됩니다. 따라서 저자는 두 문제를 동시에 해결하는 새로운 이차형 가중 훈련 목표를 제안합니다.

- **Technical Details**: 이 연구에서는 다단계 시계열 예측(multi-step time-series forecasting) 작업을 중심으로 성과를 분석합니다. 주어진 시계열 데이터셋에서 과거 값들을 기반으로 미래 값을 예측하는 방법으로, 파라미터화된 모델을 통해 예측과 레이블 시퀀스를 근사하는 것을 목표로 합니다. 새로운 접근 방식에서는 이차형 가중 훈련 목표를 도입하며, 가중 행렬의 비대각선(off-diagonal) 요소는 레이블 자기 상관 효과를 모델링하고, 비균일한 대각선(diagonal) 요소는 다양한 미래 단계에 대한 이질적인 작업 가중치를 할당합니다. 이러한 아이디어는 Quadratic Direct Forecast (QDF) 학습 알고리즘으로 구현됩니다.

- **Performance Highlights**: 실험 결과, QDF는 다양한 예측 모델의 성능을 효과적으로 향상시켜 최첨단 결과(state-of-the-art results)를 달성했습니다. 다양한 데이터 세트에 대한 포괄적인 실증 평가를 통해 QDF의 유효성을 입증했으며, 기존 방법들에 비해 더 나은 성능을 보였습니다. 저자는 이번 연구 결과를 통해 시계열 예측 모델의 학습 목표 설계에서 새로운 방향을 제시하고 있습니다.



### Calibrating and Rotating: A Unified Framework for Weight Conditioning in PEF (https://arxiv.org/abs/2511.00051)
- **What's New**: 이 논문은 Parameter-Efficient Fine-Tuning (PEFT) 기술에 대한 새로운 접근 방식을 제안합니다. 특히, LoRA 방법의 기초 위에 DoRA(Weight-Decomposed Low-Rank Adaptation)라는 방법을 통해 성능을 향상시키고 그 메커니즘을 명확하게 설명합니다. 이전의 방법들과는 차별화된 효율성을 추구하며, 새로운 프레임워크를 통해 PEFT 기술의 향상을 도모합니다.

- **Technical Details**: 논문에서 제안하는 두 가지 새로운 방법은 Pre-Diag와 SORA입니다. Pre-Diag는 LoRA 업데이트 전에 대각선 형태의 조건 행렬을 적용하여 효과적으로 사전 학습된 가중치를 조정합니다. SORA는 효율적인 매개변수를 사용하여 주의 집중 효과가 적은 강력한 정규화를 제공하여 기능 공간을 변형합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법이 기존 PEFT 방법인 LoRA와 DoRA보다 성능과 효율성 모두에서 우수한 결과를 나타냅니다. 새로운 방법들이 자연어 이해와 생성 과제에서 우수한 성능을 발휘하며, 이를 통해 PEFT 기술의 발전에 기여할 것으로 기대됩니다.



### FLoRA: Fused forward-backward adapters for parameter efficient fine-tuning and reducing inference-time latencies of LLMs (https://arxiv.org/abs/2511.00050)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLM)에 대한 파라미터 효율적인 미세 조정(paramter-efficient fine-tuning, PEFT) 방법의 일환으로 새로운 융합된 파라미터 효율적 미세 조정 방식인 FLoRA를 제안합니다. FLoRA는 기존의 저차원 어댑터(low-rank adapter, LoRA)와 병렬 어댑터(parallel adapter)의 아이디어를 결합하여 하위 작업에서의 성능을 높이고, 동시에 지연(latency)을 최소화하는 것을 목표로 합니다. 실험 결과, FLoRA는 비슷한 파라미터 예산에서 LoRA보다 더 나은 정확성과 더 낮은 지연을 기록하였습니다.

- **Technical Details**: FLoRA는 융합된 전방-후방 어댑터(fused forward-backward adapters, FFBA)로 구성되어 있으며, 저차원 어댑터의 간소한 형태의 변형입니다. 제안된 FFBA는 기존의 프로젝션 레이어(projection layers)와 결합되어 LLM의 미세 조정 과정에서 연산을 병합하여 지연을 줄입니다. 이 구조는 LoRA που 사용하여 얻은 지연을 감소시키고, 효율적인 학습을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, FLoRA 어댑터는 요약 및 대화 작업에서 LoRA보다 21-30% 빠른 응답 속도를 보였고, 공통 상식 및 수학적 추론 작업에서는 LoRA와 유사한 성능을 나타냈습니다. 이는 FLoRA의 유연한 구조 덕분에 가능한 결과이며, 양질의 성능을 헌신 없이 유지할 수 있도록 설계되었습니다.



### Adaptive Spatio-Temporal Graphs with Self-Supervised Pretraining for Multi-Horizon Weather Forecasting (https://arxiv.org/abs/2511.00049)
- **What's New**: 이번 논문에서는 기상 예측의 정확성을 높이기 위해 새로운 자가 지도 학습(self-supervised learning) 프레임워크를 제안합니다. 이 모델은 공간적(spatial) 사고를 위한 그래프 신경망(graph neural network, GNN)과 표현 학습을 위한 자가 지도(pretraining) 방식을 통합하며, 예측 기간의 변화를 통해 일반화 능력을 향상시킵니다. 다양한 데이터셋에서 광범위한 실험을 통해 기존의 전통적인 수치 기상 예측(numerical weather prediction, NWP) 모델과 최근의 딥 러닝(deep learning) 방법들보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: 제안된 모델은 여러 구성 요소를 통합하여 기상 예측의 정확성과 강건성을 높이는 데 중점을 둡니다. 모델은 기온, 풍속, 압력 및 습도 같은 역사적 기상 데이터를 입력받아 예측 신호를 생성하며, 이는 레이블이 필요 없는 자가 지도 학습 프레임워크를 통해 이루어집니다. 시간적 및 공간적 의존성을 활용하여 기상 조건을 예측하며, 대조 학습(contrastive learning)을 사용하여 다양한 기상 패턴을 구별하고 예측 정확성을 향상시킵니다.

- **Performance Highlights**: 모델의 성능은 베이징과 상하이의 정량적 평가 및 시각적 분석을 통해 입증되었으며, 세련된 기상 패턴을 포착하는 데 성공했습니다. 강력한 스페이스-타임 적응 메커니즘을 통해 모델은 지역 및 시간 적응성을 보이면서도 다양한 예측 기간에서 정확성을 유지합니다. 이러한 접근법은 향후 데이터 기반 기상 예측 시스템을 위한 확장 가능하고 레이블 효율적인 솔루션을 제공한다는 점에서 중요한 의의를 갖습니다.



### DynBERG: Dynamic BERT-based Graph neural network for financial fraud detection (https://arxiv.org/abs/2511.00047)
- **What's New**: 금융 사기 탐지는 분산 네트워크와 같은 시스템의 무결성을 유지하는 데 필수적입니다. 기존의 Graph Convolutional Networks (GCNs) 대신에, DynBERG라는 새로운 아키텍처를 소개하였습니다. DynBERG는 Graph-BERT와 Gated Recurrent Unit (GRU) 레이어를 통합하여 다가오는 시간 단계에 걸쳐서 동적인 변화를 포착합니다.

- **Technical Details**: 이 연구에서는 딥 러닝 모델이 주로 정적 그래프에 초점을 맞추었던 기존 연구와 달리, 동적인 재정 트랜잭션 네트워크의 구조적 변화를 다룹니다. DynBERG는 주기적인 데이터 처리와 지시된 엣지를 지원하도록 알고리즘을 수정하여 동적인 금융 거래 분석에 유리한 특징을 지닙니다. Elliptic 데이터셋을 기반으로 Bitcoin 트랜잭션을 분석하였고, Dark Market Shutdown이라는 큰 이벤트 전후에 모델의 적응력을 평가하였습니다.

- **Performance Highlights**: DynBERG는 성능 평가에서 EvolveGCN 및 GCN과 같은 최신 동적 그래프 분류 모델을 초월하는 우수한 성능을 입증했습니다. 특히, 시장 셧다운 이전의 EvolveGCN보다 더 나은 결과를 보였고, 이후 GCN을 초과하는 성과를 기록하였습니다. Ablation study를 통해 GRU를 포함한 시계열 딥 러닝 구성 요소의 중요성을 강조하며, 금융 거래의 시간적 동역학을 모델링하는 효과를 보여주었습니다.



### Endowing GPT-4 with a Humanoid Body: Building the Bridge Between Off-the-Shelf VLMs and the Physical World (https://arxiv.org/abs/2511.00041)
- **What's New**: 이번 논문에서는 기존의 대규모 데이터 수집 방법 대신, 시중에서 사용 가능한 Vision-Language Models (VLMs)인 GPT-4를 사용하여 사람형 에이전트를 제어하는 새로운 방법인 BiBo를 제안합니다. BiBo는 두 가지 주요 구성 요소인 구체화된 명령 컴파일러와 확산 기반의 모션 실행기로 구성되어 있어, 더 적은 데이터 수집으로 다양한 상호작용을 가능하게 합니다.

- **Technical Details**: BiBo의 구체화된 명령 컴파일러는 사용자의 고수준 명령을 저수준 운동 명령으로 변환하고, 이를 통해 환경 이해를 도모합니다. 모션 실행기는 명령을 받아 실제 휴머노이드의 동작을 생성하며, Latent Diffusion Model (LDM)을 활용하여 환경 피드백을 실시간으로 반영할 수 있습니다.

- **Performance Highlights**: BiBo는 랜덤 생성된 물리적 환경에서 90.2%의 상호작용 성공률을 기록했으며, 이전 방법들에 비해 16.3% 개선된 정확성을 보여주었습니다. 이 시스템은 복잡한 운동 실행과 무한 길이의 모션 합성을 가능하게 하며, 사용자 지시에 따른 실시간 제어를 지원합니다.



### Semi-Supervised Preference Optimization with Limited Feedback (https://arxiv.org/abs/2511.00040)
- **What's New**: 이번 논문에서는 Semi-Supervised Preference Optimization (SSPO)라는 새로운 방법론을 제안하여, 적은 양의 쌍선호 레이블과 대규모 비회전 샘플을 동시에 학습하는 방식을 탐구합니다. SSPO는 적은 레이블 데이터를 사용하여 비회전 데이터에 대한 원칙 있는 의사 레이블링(pseudo-labeling)을 가능하게 하여, 고비용의 데이터 획득 문제를 해결하고 인공지능 모델의 인간 정렬을 보존합니다.

- **Technical Details**: 이 연구는 선호 최적화(preference optimization)를 확률적 분류 문제로 재구성하여 비회전 데이터에 대한 의사 레이블링 전략을 정당화합니다. 모델 출력이 인간의 선호와 일치하도록 하는 목표는 쌍비교(pairwise comparison)에서 선호하는 응답을 식별하는 스코어링 함수 학습으로 정의됩니다. 이 과정에서 동적 보상 임계값을 조정하여 고신뢰도 선호 레이블을 생성하고, 쌍레이블 데이터와 함께 정책 모델을 공동 최적화합니다.

- **Performance Highlights**: SSPO는 Llama3-8B-Instruct 모델을 사용하여 UltraFeedback 데이터의 1%에서 훈련된 경우, 10%에서 훈련된 강력한 기준선보다 지속적으로 뛰어난 성능을 보여주었습니다. 이러한 결과는 SSPO가 높은 데이터 효율성을 가지면서도 인간의 선호를 효과적으로 추출할 수 있음을 나타냅니다. 또한 이는 대규모 레이블된 데이터셋에 대한 의존성을 크게 줄이며, 효과적이고 확장 가능한 고품질 LLM 정렬을 위한 프레임워크를 제공합니다.



### STRIDER: Navigation via Instruction-Aligned Structural Decision Space Optimization (https://arxiv.org/abs/2511.00033)
- **What's New**: 이 논문에서는 Zero-shot Vision-and-Language Navigation in Continuous Environments (VLN-CE) 과제를 다루며, 에이전트가 자연어 지침에 따라 이전에 본 적 없는 3D 환경을 탐색하는 방법을 제안합니다. 기존 방법의 한계점을 극복하기 위해 STRIDER라는 새로운 프레임워크를 도입하였고, 이는 에이전트의 의사 결정을 최적화하여 공간 구조와 작업 피드백을 통합합니다. 특히 두 가지 혁신적인 구성 요소가 포함되어 있습니다: 구조화된 웨이포인트 생성기(Structured Waypoint Generator)와 작업 정렬 조정기(Task-Alignment Regulator).

- **Technical Details**: STRIDER는 Instruction-Aligned Structural Decision Space Optimization을 기반으로 하여 에이전트의 의사 결정을 최적화합니다. 구조화된 웨이포인트 생성기는 깊이 기반의 탐색 가능한 영역에서 골격을 추출하여 공간적으로 제약된 행동 공간을 만듭니다. 작업 정렬 조정기는 작업 진행 상황을 지속적으로 모니터링하고 에이전트의 행동을 조정하여 의사 결정 공간을 최적화합니다. 이러한 두 가지 모듈은 결합되어 에이전트의 의사 결정을 공간적 제약과 작업 진행에 따라 조정합니다.

- **Performance Highlights**: R2R-CE와 RxR-CE 두 개의 표준 제로샷 VLN-CE 벤치마크에서 STRIDER가 강력한 기초선 모델을 일관되게 초월한다는 것을 보여주며, 성공률(Success Rate)이 29%에서 35%로 향상되었습니다. 이러한 결과는 공간적으로 제약된 의사 결정 및 피드백 기반 실행의 중요성을 강조하며, 제로샷 VLN-CE에서 내비게이션 정확도를 개선하는 데 기여합니다.



### From Uniform to Adaptive: General Skip-Block Mechanisms for Efficient PDE Neural Operators (https://arxiv.org/abs/2511.00032)
- **What's New**: 최근 Neural Operators(NO)는 편미분 방정식(PDEs)을 해결하기 위한 인기 있는 접근 방식으로 부각되고 있으나, 대규모 엔지니어링 작업에 적용할 경우 막대한 계산 부담을 초래하는 문제가 있다. 본 연구에서는 이런 비효율성을 해결하기 위해 Transformer 기반 neural operators에 통합할 수 있는 구조인 Skip-Block Routing(SBR)이라는 프레임워크를 도입하였다. 이 프레임워크는 복잡도에 따른 토큰의 우선순위를 학습하여 더 복잡한 지역에 더 많은 계산 자원을 집중할 수 있도록 한다.

- **Technical Details**: SBR은 두 개의 핵심 구성 요소로 이루어져 있다. 첫 번째는 전역 라우터 모듈로, 입력 도메인 전체에 대한 정적 복잡도 순위를 생성하여 계산 우선순위 계획을 선정한다. 두 번째는 적응형 처리 백본으로, 이 순위를 활용하여 네트워크의 다양한 깊이에서 활성 토큰의 수를 동적으로 조정한다. 이로 인해 더 복잡한 영역에는 더 깊고 집약적인 처리 변환이 집중된 구조다.

- **Performance Highlights**: 실험 결과, SBR은 다양한 neural operators에 원활하게 통합될 수 있으며, 대략 50%의 계산 비용 절감과 2배 더 빠른 추론 속도를 제공하면서도 정확도를 유지하는 성능 개선을 보여주었다. 이러한 개선은 실제 적용에서도 큰 이점을 제공하여 엔드 투 엔드 추론을 더 효율적으로 실행할 수 있게 한다.



### Probing Knowledge Holes in Unlearned LLMs (https://arxiv.org/abs/2511.00030)
Comments:
          The Thirty-ninth Annual Conference on Neural Information Processing Systems

- **What's New**: 이번 논문은 기계의 언learn(기계학습에서 학습된 내용을 잊게 하는 과정을 의미)이라는 새로운 패러다임을 다루고 있으며, 원치 않는 지식을 제거하면서도 성능 저하를 최소화할 수 있는 방법을 제안합니다. 연구진은 최근의 기술들이 모델 성능을 해치지 않고 원하지 않는 내용 삭제에 효과적임을 보였으나, '지식 구멍(knowledge holes)'이 발생할 수 있음을 발견했습니다. 이 연구는 이러한 지식 구멍을 탐지하고 평가하기 위한 새로운 테스트 케이스 생성 프레임워크를 제안합니다.

- **Technical Details**: 연구진은 기존의 모델들이 저품질 응답을 생성하는 입력 프롬프트를 체계적으로 조사할 수 있는 자동화된 프레임워크를 개발했습니다. 이 프레임워크는 제거된 위험한 콘텐츠와 관련된 키워드를 사용하는 인접 지식 및 보다 폭넓은 주제를 아우르는 잠재적 지식 탐색을 포함합니다. 실험 결과, 기존 모델은 고품질 응답을 제공하는 반면, 언learn된 모델은 관련 없는 또는 불완전한 응답을 생성하는 경우가 많음을 발견했습니다.

- **Performance Highlights**: 연구에서 75.2%의 인접 지식 테스트 케이스와 98.7%의 잠재 지식 테스트 케이스에서 언learn된 모델이 극도로 낮은 품질의 응답을 보여주었습니다. 이 결과는 기계 언learn의 복잡성과 그로 인한 부작용을 강조하며, 기존의 정적 벤치마크가 지식 보존을 평가하는 데 한계가 있음을 나타냅니다. 연구진은 향후 연구가 해로운 지식을 제거하면서도 무해한 지식을 보존할 수 있는 균형을 이룰 수 있도록 기여하기를 희망하고 있습니다.



### Feature-Guided SAE Steering for Refusal-Rate Control using Contrasting Prompts (https://arxiv.org/abs/2511.00029)
Comments:
          12 pages, 6 figures

- **What's New**: 이 논문은 기존의 안전성과 효용성 간의 트레이드오프를 극복하기 위해 Sparse Autoencoders (SAEs)를 활용한 새로운 접근법을 제안합니다. 저자들은 안전하지 않은 프롬프트에 대해 LLM이 답변하지 않도록 유도하는 동시에 안전한 프롬프트에 대해 응답을 유도하는 데 중점을 두고 있습니다. 최근의 기계적 해석 가능성의 발전을 활용하여 특정 특징을 효과적으로 식별하고 조정할 수 있는 기회를 확보하였습니다.

- **Technical Details**: 저자들은 Llama-3 8B 모델을 대상으로 Hinton에서 제안한 대조 프롬프트 방법을 사용하여 효과적인 특징 선택 및 평가 방법론을 적용했습니다. 각 레이어의 부분적 데이터와 대조 프롬프트의 쌍을 통해 모델 활성 패턴의 차이를 유도하여 유의미한 안전성 및 유용성 변화를 이끌어냈습니다. 새로운 조합 점수 함수를 도입하여 SAEs의 특징을 체계적으로 정렬하고 평가할 수 있는 절차를 개발하였습니다.

- **Performance Highlights**: 그들의 접근 방식은 안전성 성능을 18.9% 향상시키는 동시에 효용성 역시 11.1% 증가시켰습니다. 이는 최적의 특징이 체계적으로 선택될 때, SAEs를 통한 표적 조정이 전통적인 안전성-효용성 트레이드오프를 극복할 수 있다는 것을 보여줍니다. 이러한 결과는 LLM의 안전성을 유지하며 효용성을 증대시키는 새로운 경로를 제시합니다.



### Mutual Information guided Visual Contrastive Learning (https://arxiv.org/abs/2511.00028)
Comments:
          Tech Report - Undergraduate Thesis - 2023

- **What's New**: 이번 연구에서는 상호 정보(mutual information)를 기반으로 하는 데이터 선택(data selection) 및 증강(data augmentation) 방법을 탐구합니다. 기존의 데이터 증강 기법들은 주로 색상 변형(color jittering)에 집중하여 실제 조명 변화(real-world illumination changes)를 모방하는 데 초점을 맞추었습니다. 하지만 본 연구는 각 장면(scene)의 패치를 분석하여 상호 정보가 높은 패치를 긍정 샘플로 선택하는 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: 이 연구는 InfoAug라는 새로운 데이터 증강 기술을 소개합니다. InfoAug는 각 패치의 상호 정보와 다른 패치 간의 관계를 분석하여 강한 상관관계를 보이는 최고의 패치를 긍정 샘플로 선정합니다. 특히, 이 연구에서는 비디오 데이터셋을 활용하여 첫 번째 프레임의 패치들에 대한 상호 정보를 추정하며, 모든 패치 간의 상관관계를 종합적으로 고려하여 긍정 샘플을 결정하는 방법론을 개발했습니다.

- **Performance Highlights**: 제안된 방법은 CIFAR-10, CIFAR-100, STL-10 등 여러 벤치마크에서 성능 평가를 실시하였으며, 결과적으로 기존의 기법들에 비해 일관되게 향상된 성능을 보였습니다. 이를 통해 우리의 접근법이 실제 모델과 인간의 시각적 학습에 더 부합하는 긍정 샘플 결정 방법임을 입증하고 있습니다. 또한, 기존의 대비 학습 프레임워크와 비교하여 앞선 성과를 보이며, 향후 연구를 위한 유망한 방향을 제시하고 있습니다.



### Position Paper: If Innovation in AI Systematically Violates Fundamental Rights, Is It Innovation at All? (https://arxiv.org/abs/2511.00027)
Comments:
          NeurIPS 2025 Position Paper track; accepted for oral and poster presentation at the Thirty-Ninth Annual Conference on Neural Information Processing Systems

- **What's New**: 이번 논문은 인공지능(AI)의 발전과 관련된 규제와 혁신 간의 오해를 다루고 있습니다. 규제가 혁신을 저해하는 것이 아니라 오히려 그 토대가 될 수 있다는 점에서, 역사적 사례를 통해 이러한 주장을 뒷받침합니다. 특히, 유럽연합의 AI 법안은 위험 기반 규제의 모범 사례로 제시되며, 규제의 적시 적용이 근본적 권리를 보호하는 동시에 혁신을 촉진할 수 있음을 강조합니다.

- **Technical Details**: 논문에서는 Collingridge Dilemma를 통해 기술이 사회적 영향을 미치기 시작했을 때에는 규제가 제대로 작동하지 않을 수 있음을 지적합니다. 이를 해결하기 위해, 규제 샌드박스와 실제 시험, 중소기업(SMEs) 지원 등의 적응적 메커니즘이 필요한데, 이들은 기본적인 권리를 보호하면서 기술적 진전을 가능하게 하는 도구입니다. 이러한 체계는 사회의 신뢰와 법적 확실성을 제공하는 이점이 있습니다.

- **Performance Highlights**: 기술 발전에 있어 규제는 오히려 긍정적인 영향력을 미친다는 점을 다양한 산업 사례를 통해 설득력 있게 보여줍니다. 예를 들어, 제약 산업에서는 규제가 안전성을 높여 왔으며, 항공 산업에서는 규제가 비약적인 안전 개선을 가져온 바 있습니다. 이와 같은 역사적 사례들은 잘 설계된 규제가 AI와 같은 새로운 기술의 혁신을 저해하기보다는 도리어 촉진하는 역할을 할 수 있음을 시사합니다.



### Chitchat with AI: Understand the supply chain carbon disclosure of companies worldwide through Large Language Mod (https://arxiv.org/abs/2511.00024)
- **What's New**: 이 논문은 기업의 탄소 공개 품질을 대규모로 평가하기 위한 새로운 의사결정 지원 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLMs)을 활용하여 2010년부터 2020년까지의 CDP 데이터에 대해 내러티브 점수화 방식으로 통합된 평가를 수행합니다. 이로 인해 산업과 지역 간의 비교가 가능해지며, 기업의 환경 지속 가능성을 분석하는 데 소요되는 비용과 시간을 대폭 줄일 수 있는 혁신적인 방법을 제공합니다.

- **Technical Details**: CDP 데이터셋은 환경 성과 공개에 대한 기업의 응답을 수집하며, 구조화된 지표와 개방형 서사의 혼합 형태로 되어 있습니다. 제안된 LLM 기반 접근은 체계적인 평가 파이프라인을 통해 개별 기업의 응답을 점수화하도록 설계되어 있습니다. 이 방법론은 서사적 평가의 일관성을 높이고, 다양한 산업 및 지역 간의 성과를 비교하는 데 필요한 기준을 제시합니다.

- **Performance Highlights**: 기술 및 독일과 같은 국가가 높은 서사 점수 정렬을 보여주는 반면, 일부 산업은 변동성을 보이거나 피상적 참여를 나타내는 것으로 파악되었습니다. 이 연구의 결과는 투자자, 규제 기관 및 기업의 환경, 사회, 지배구조(ESG) 전략가들에 대한 향후 의사결정에 중요한 인사이트를 제공하며, AI 기반 의사결정 지원 시스템의 기능을 한층 향상시킬 수 있는 기반을 마련합니다.



### Deep Learning Models for Coral Bleaching Classification in Multi-Condition Underwater Image Datasets (https://arxiv.org/abs/2511.00021)
Comments:
          15 pages, 10 figures

- **What's New**: 이 연구는 다양한 환경 조건에서 건강한 산호와 백화산호 샘플을 포함한 글로벌 데이터셋을 기반으로 한 혁신적인 기계 학습(machine-learning) 기반 산호 백화(classification) 시스템을 제안합니다. 이는 산호의 효율적인 보호 및 모니터링을 위한 시급한 필요성에 대한 해결책을 제공합니다. 함께 비교한 최신 모델들은 Residual Neural Network (ResNet), Vision Transformer (ViT), Convolutional Neural Network (CNN)으로, 각각의 성능을 종합적으로 평가했습니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 깊은 바다(deep seas), 습지(marshes), 해안 지역(coastal zones) 등 다양한 환경에서 수집되었습니다. 하이퍼파라미터 튜닝(hyperparameter tuning)을 통해 CNN 모델이 88%의 정확도를 기록하였고, 이는 기존 벤치마크를 초과하는 성과입니다. 이러한 성과는 자동화된 산호 모니터링 및 컴퓨터 비전 모델(computer vision models) 분석에 중요한 통찰력을 제공합니다.

- **Performance Highlights**: CNN 모델은 88%의 높은 정확도로 산호 백화와 건강한 산호를 정확히 분류하는 데 성공했습니다. 다른 두 모델인 ResNet과 ViT와 비교했을 때, CNN이 가장 뛰어난 성능을 보였습니다. 본 연구 결과는 산호 생태계 보호를 위한 효율적인 방법 제공에 기여할 것입니다.



### Sorting by Strip Swaps is NP-Hard (https://arxiv.org/abs/2511.00015)
Comments:
          4 pages

- **What's New**: 이번 논문에서는 Sorting by Strip Swaps (SbSS) 문제의 NP-hardness를 증명하였습니다. 이 증명은 Block Sorting 문제로부터의 다항식 감소를 통해 이루어졌습니다. 핵심 아이디어는 감소하는 쌍을 격리된 삼중항으로 치환하는 로컬 장치인 'cage'를 사용하는 것입니다.

- **Technical Details**: SbSS에서는 최대 증가 조각(maximal increasing strips) 두 개를 교환하여 정렬합니다. 각 교환은 전체다항식으로 주어질 수 있는 디시전 문제로 변환할 수 있습니다. 여기서 Strip Swap Distance (SSD)라는 개념을 도입하여, 정렬할 때 필요한 최소 교환 횟수를 측정합니다.

- **Performance Highlights**: SbSS 문제는 Block Sorting 문제와의 유사성 덕분에 현재까지도 계산의 어려움을 가진 것으로 여겨집니다. 기존 문헌에 따르면, Block Sorting은 NP-hard로 알려져 있으며, SbSS도 이와 유사한 복잡성을 지니고 있습니다. 가장 잘 알려진 근사 알고리즘인 2-approximation 알고리즘이 존재하지만, SbSS의 정확한 해를 구하는 데 필요한 최소한의 교환 횟수를 찾는 것은 아직까지 열려있는 질문입니다.



### Generative human motion mimicking through feature extraction in denoising diffusion settings (https://arxiv.org/abs/2511.00011)
- **What's New**: 최근 대형 언어 모델의 성공은 인간-AI 간의 언어적 상호작용에 새로운 물결을 일으켰습니다. 하지만 이러한 모델은 인간 상호작용의 구체성을 결여하고 있습니다. 춤은 인간 표현의 원초적 형태로서 이러한 경험을 보완할 잠재력을 가지고 있습니다.

- **Technical Details**: 이 연구는 모션 캡처(motion capture) 데이터를 기반으로 한 인터랙티브 모델을 구축하여 인간-AI 상호작용을 탐구합니다. 이 모델은 단일 인물 모션 데이터를 활용하여 incoming 움직임 시퀀스를 부분적으로 모방하고 "창의적으로" 강화합니다. 또한, 두 가지 노이즈 확산 모델, 모션 인페인팅(motion inpainting), 모션 스타일 전이(motion style transfer)의 아이디어를 결합하여 시간적으로 일관되고 선택된 움직임 참조에 반응하는 이동 표현을 생성합니다.

- **Performance Highlights**: 모델의 성공은 생성된 샘플과 테스트 세트의 특징 분포 수렴을 정량적으로 평가하여 입증됩니다. 우리의 생성은 다양한 인간 파트너와의 차이를 보이며, 동시에 실제적인 모습을 유지하는 AI와의 창의적인 춤의 첫걸음으로 평가됩니다.



### Multimodal Learning with Augmentation Techniques for Natural Disaster Assessmen (https://arxiv.org/abs/2511.00004)
Comments:
          Accepted at 2025 IEEE 21st International Conference on Intelligent Computer Communication and Processing (ICCP 2025)

- **What's New**: 이 논문은 자연재해 평가에 필요한 정보에 대한 신속하고 정확한 접근 방식의 중요성을 강조하고 있습니다. 특히 소셜 미디어가 재난 분석을 위한 실시간 데이터 소스로 주목받고 있지만, 기존 데이터셋이 불균형 클래스와 한정된 샘플 문제로 인해 모델 개발이 어렵다는 점를 지적합니다. 해결책으로, 실험에서는 CrisisMMD 다중 양식 데이터셋을 활용하여, 다양한 증강 기법을 적용하여 재난 분류 모델의 성능을 향상시킵니다.

- **Technical Details**: 이 연구에서는 주로 두 가지 이미지 증강 기술인 Real Guidance와 DiffuseMix를 적용했습니다. Real Guidance는 조건부 이미지-투-이미지 생성 기법으로, Stable Diffusion 1.5 모델을 사용하여 원래 이미지를 현실적으로 변형합니다. DiffuseMix는 Masked Blending과 Fractal Visual Modifications를 사용하여 다양한 증강 이미지를 생성하고, 이는 데이터셋의 일부에 국한된 클래스의 샘플 수를 증가시킵니다.

- **Performance Highlights**: 실험 결과, 선택된 증강 전략이 특히 저조한 클래스의 분류 성능을 개선하는 데 기여했음을 보여주었습니다. 다중 시점 학습은 잠재력을 보였지만 추가적인 정제가 필요하다는 점이 지적되었습니다. 이 연구는 제안한 증강 기법들이 재난 평가 시스템을 더욱 견고하게 구축하는 데 기여할 수 있음을 보여줍니다.



### VRScout: Towards Real-Time, Autonomous Testing of Virtual Reality Games (https://arxiv.org/abs/2511.00002)
- **What's New**: 이 논문에서는 VR 환경 내에서 자율적으로 탐색하고 가상 객체와 상호작용할 수 있는 딥 러닝 기반의 에이전트인 VRScout를 소개합니다. VRScout는 인간의 시범을 학습하여 자연스럽고 효율적으로 행동하며, Action Chunking Transformer (ACT)를 통해 멀티 스텝 액션 시퀀스를 예측합니다. 이를 통해 다양한 환경에 걸쳐 고급 전략을 캡처하고 일반화할 수 있습니다.

- **Technical Details**: VRScout는 VR 장면 이미지와 사용자 동작을 입력으로 받아, ResNet-18 인코더를 사용하여 특징 벡터를 생성하고, 이 벡터를短期적(time chunk)으로 묶어 ACT 모델의 인코더에 공급합니다. 예측된 동작은 가상 VR 컨트롤러를 통해 게임에 주입되어 다음 게임 상태를 생성하며, 동적 슬라이딩 호라이즌을 통해 에이전트의 시간적 문맥을 실행 중에 조정합니다.

- **Performance Highlights**: VRScout는 상업적으로 인기 있는 VR 게임에서 전문가 수준의 성과를 달성하며, Beat Saber에서는 4시간의 훈련 데이터만으로도 효과적인 성능을 보입니다. 더욱이 소비자 등급의 하드웨어에서 60 FPS의 실시간 추론을 수행할 수 있어 VR 게임의 자동화된 테스트를 위한 실용적이고 확장 가능한 솔루션으로 자리매김하고 있습니다.



### A Two Level Neural Approach Combining Off-Chip Prediction with Adaptive Prefetch Filtering (https://arxiv.org/abs/2403.15181)
Comments:
          To appear in 30th International Symposium on High-Performance Computer Architecture (HPCA), 2024

- **What's New**: 이번 연구에서는 Two Level Perceptron (TLP) 예측기를 제안하여, 대규모 데이터에서 발생하는 성능 및 에너지 오버헤드를 완화하고자 하였습니다. TLP는 첫 번째 레벨 데이터 캐시(L1D)에서 오프 칩(access off-chip) 예측과 적응형 프리패치 필터링을 결합하는 신경 메커니즘으로 구성되어 있습니다. FLP(First Level Predictor)와 SLP(Second Level Predictor)라는 두 개의 연결된 미세 구조적(perceptron) 예측기로 이루어져 있습니다.

- **Technical Details**: FLP는 가상 주소에 기반한 여러 프로그램 특성과 독창적인 선택적 지연(Selective Delay) 요소를 활용하여 정확한 오프 칩 예측을 수행합니다. SLP는 신체 주소(Physical Addresses)와 FLP 예측을 특징으로 사용하여 L1D 프리패치 필터링을 주도합니다. TLP는 다단계(perceptron) 하드웨어 접근 방식을 통해 오프 칩 예측과 프리패치 필터링을 모두 겨냥한 최초의 하드웨어 제안입니다.

- **Performance Highlights**: TLP는 7KB의 저장 공간만 필요하며, 다양한 단일 코어 및 다중 코어 워크로드에서 상태-of-the-art 접근 방식과의 성능을 비교하여 이점을 입증합니다. 실험 결과, TLP는 단일 코어와 다중 코어 워크로드에서 각각 평균 DRAM 트랜잭션을 30.7% 및 17.7% 감소시켰으며, 상대적으로 최근 작업보다 성능이 향상되었습니다. 결과적으로 TLP는 단일 코어와 다중 코어 워크로드에서 각각 6.2% 및 11.8%의 기하 평균 성능 속도 향상을 달성하였습니다.




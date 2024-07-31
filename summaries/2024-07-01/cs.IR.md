New uploads on arXiv(cs.CL)

### Scaling Synthetic Data Creation with 1,000,000,000 Personas (https://arxiv.org/abs/2406.20094)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 다양한 관점을 활용하여 대규모 언어 모델(LLM)에서 다양한 합성 데이터를 생성하는 새로운 페르소나 기반 데이터 합성 방법론을 제안합니다. 이를 위해 웹 데이터로부터 자동으로 수집된 10억 개의 다양한 페르소나를 포함하는 페르소나 허브(Persona Hub)를 소개합니다. 이 페르소나는 전 세계 지식의 분산된 매개체로서 LLM의 거의 모든 관점을 활용하여 다양한 합성 데이터를 대규모로 생성할 수 있습니다.

- **Technical Details**: 이 논문에서는 두 가지 주요 접근 방식인 Text-to-Persona와 Persona-to-Persona를 사용하여 웹 데이터에서 다양한 페르소나를 도출합니다. Text-to-Persona 방식은 특정 텍스트에서 해당 텍스트를 읽거나 작성할 가능성이 있는 특정 페르소나를 유추하는 방법입니다. Persona-to-Persona 방식은 Text-to-Persona로 도출된 페르소나의 인간관계를 통해 새로운 페르소나를 추가로 생성하는 방법입니다. 페르소나 허브는 MinHash를 사용하여 페르소나 설명의 n-gram 특징을 기반으로 중복을 제거하여 다양성을 보장합니다.

- **Performance Highlights**: 페르소나 허브를 사용하여 대규모 수학 및 논리적 추론 문제, 사용자 지침, 지식이 풍부한 텍스트, 게임 NPC 및 도구(function)를 효과적으로 생성할 수 있음을 입증했습니다. 이를 통해 페르소나 기반 데이터 합성은 사용자 친화적이며, 다목적성, 확장 가능성, 유연성을 제공하여 합성 데이터 생성 및 실제 적용에서 패러다임 전환을 이끌 가능성이 큽니다.



### Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs (https://arxiv.org/abs/2406.20086)
Comments:
          13 pages, 14 figures. Code and data at this https URL

- **What's New**: 이 연구에서는 Named entities 및 multi-token words의 마지막 토큰 표현이 초기에 정보가 지워지는 'erasure' 효과를 보인다는 점을 발견했습니다. 이를 바탕으로, 자가회귀 LLM의 암묵적 단어장(implicit vocabulary)을 레이어 간의 토큰 표현 차이를 통해 '읽어내기' 위한 방법을 제안했습니다. Llama-2-7b와 Llama-3-8B 모델을 대상으로 실험을 진행해 결과를 발표했습니다.

- **Technical Details**: LLM은 일반적으로 단어를 여러 토큰으로 분할하여 처리합니다. 예를 들어 'northeastern'은 ['_n', 'ort', 'he', 'astern']과 같은 토큰으로 나뉘어 처리되며, 이들은 각각 의미 있는 단어 'north' 또는 'east'와는 연관이 없습니다. 연구진은 마지막 토큰의 표현이 이전 토큰들의 정보를 조기에 '지워버리는' 효과가 있다는 것을 발견했습니다. 이를 바탕으로, 특정 토큰 시퀀스의 'lexicality'를 점수화하는 휴리스틱을 개발해 자가회귀 LLM의 어휘 목록을 추출하는 방법을 제안했습니다.

- **Performance Highlights**: Llama-2-7b와 Llama-3-8B 모델 모두에서 마지막 토큰이 초기에 정보를 지우는 'erasure' 효과가 확인되었습니다. 이 새로운 접근법을 통해 LLM의 암묵적 어휘 목록을 성공적으로 읽어낼 수 있었으며, 이는 기존의 LLM 연구에서 중요한 기여를 합니다.



### Molecular Facts: Desiderata for Decontextualization in LLM Fact Verification (https://arxiv.org/abs/2406.20079)
- **What's New**: 이 논문은 대형 언어 모델(LLM)이 생성하는 정보의 사실성 검증(factuality verification)에 대한 새로운 접근 방식을 제시합니다. 기존의 원자적 사실(atomic facts) 대신 '분자적 사실(molecular facts)'이라는 개념을 도입하여, 이들이 맥락에서 독립적으로 의미를 가질 수 있고 최소한의 추가 정보를 포함하도록 합니다.

- **Technical Details**: 연구진은 사실검증을 위해 '탈맥락화(decontextuality)'와 '최소성(minimality)'이라는 두 가지 기준을 정의했습니다. 이 기준들을 바탕으로 원자적 사실이 너무 작아서 맥락 없이도 의미를 잃지 않도록 하면서도, 과도한 정보가 추가되지 않도록 조정한 분자적 사실을 생성하는 방법을 제안합니다. 또한, 다양한 탈맥락화 기법들과 비교하여 분석했습니다.

- **Performance Highlights**: 실험 결과, 분자적 사실은 장문 생성(long-form generation)에서 기존의 탈맥락화 방법들보다 더 우수한 성능과 최소성을 보였습니다. 특히, 신원 확인이 모호한 상황에서 분자적 사실이 정확성을 유지하면서도 주장(클레임)의 최소성을 만족시키는 것으로 나타났습니다.



### Applying RLAIF for Code Generation with API-usage in Lightweight LLMs (https://arxiv.org/abs/2406.20060)
- **What's New**: 이번 연구는 Reinforcement Learning from AI Feedback (RLAIF) 프레임워크를 경량 (<1B 파라미터) LLMs의 코드 생성 능력을 개선하는 데 도입하였습니다. 특히, 적절한 API 호출을 작성하는 데 필요한 코드 생성 작업에 집중하며, LLMs의 잘 알려진 허상(헬루시네이션) 문제를 다룹니다. 본 연구에서는 GPT-3.5와 같은 더 큰 LLM에서 AI 피드백을 추출하여 이를 보상 모델로 사용하여 작은 LLM의 정렬을 향상시키고자 합니다.

- **Technical Details**: 본 연구는 Gorilla 데이터셋으로 실험을 진행하며, AST, ROUGE, Code-BLEU 등 다양한 메트릭을 통해 모델 생성 코드의 품질을 철저히 평가합니다. 또한 코드를 실행 가능성을 정확하게 계산할 수 있는 파이프라인을 개발하였습니다. 연구에서는 GPT-2-large (780M 파라미터) 모델이 RLAIF를 통해 LLaMA-7B (7B 파라미터) 모델의 성능을 초과하는 것을 확인하였습니다. 인스트럭션과 생성된 코드(및 API 호출)가 주어졌을 때, 여러 이진 질문을 통해 코드의 품질을 평가하는 새로운 프롬프트 전략이 사용됩니다.

- **Performance Highlights**: RLAIF 프레임워크를 적용한 결과, 기존의 파인튜닝된 LLM 대비 실행 가능성(executability rate) 측면에서 4.5% 향상된 성능을 보였습니다. 특히, 780M 파라미터의 작은 모델이 7B 파라미터를 가진 더 큰 모델의 실행 가능성 측면에서 1.0% 더 높은 성능을 달성했습니다.



### To Word Senses and Beyond: Inducing Concepts with Contextualized Language Models (https://arxiv.org/abs/2406.20054)
- **What's New**: 본 논문에서는 새로운 과제인 'Concept Induction'을 소개합니다. 이는 데이터로부터 직접 단어들 사이의 소프트 클러스터링을 학습하여 개념을 정의하는 비지도 학습 과제입니다. 기존의 Word Sense Induction을 일반화하면서, 단어의 의미 뿐만 아니라 개념을 학습할 수 있습니다.

- **Technical Details**: 이 접근법은 목표 어휘 집합의 단어 출현을 컨텍스트화된 언어 모델(BERT Large)에서 파생된 단어 임베딩으로 표현하고, 이를 하드 클러스터링 알고리즘을 사용하여 개념 클러스터로 묶습니다. 로컬 및 글로벌 두 수준의 클러스터링을 활용하여, 단어 의미 및 개념을 동시에 학습하는 바이레벨(bi-level) 방법론을 제안합니다.

- **Performance Highlights**: SemCor 데이터셋으로 평가한 결과, Concept Induction 과제에서 BCubed F1 점수 0.60을 초과하여 좋은 성능을 보였습니다. 또한, Word-in-Context 과제에서도 경쟁력 있는 성능을 발휘했으며, 이는 우리의 접근법이 로컬 및 글로벌 관점을 모두 활용하는 것이 효과적임을 나타냅니다.



### Understanding and Mitigating Language Confusion in LLMs (https://arxiv.org/abs/2406.20052)
- **What's New**: 새로운 연구는 대형 언어 모델(LLMs)의 주요 한계를 밝혀냈습니다. 이는 사용자가 원하는 언어로 텍스트를 일관되게 생성하지 못하는 문제입니다. 이에 대한 평가를 위해 Language Confusion Benchmark(LCB)을 도입하였고, 이는 15개의 다양한 언어를 포괄합니다. 다양한 LLM들을 평가한 결과, Llama Instruct와 Mistral 모델이 높은 수준의 언어 혼란을 보였으며, 가장 강력한 모델들도 일관되게 정확한 언어로 응답하지 못하는 문제를 확인하였습니다.

- **Technical Details**: 연구는 두 가지 실제 사용 사례를 반영한 단일 언어 및 교차 언어 생성에서 언어 혼란을 평가했습니다. monolingual generation에서는 사용자가 특정 언어로 LLM에게 질의하면 동일한 언어로 응답하도록 요청하고, cross-lingual generation에서는 사용자가 한 언어로 모델에게 다른 언어로 텍스트를 생성하도록 명시적으로 지시합니다. 새로운 언어 혼란 벤치마크를 만들기 위해 공공의 영어 및 다국어 지시 데이터셋에서 프롬프트를 수집하였고, 더 복잡한 프롬프트를 사용해 새로운 데이터를 생성했습니다.

- **Performance Highlights**: Llama Instruct와 Mistral 모델은 많은 언어에서 심각한 언어 혼란을 보였습니다. Command R 및 OpenAI 모델은 단일 언어 생성에서는 더 나은 성능을 보였지만, 교차 언어 생성에서는 일관되게 올바른 언어로 텍스트를 생성하지 못했습니다. 언어 혼란은 일부 해결책(few-shot prompting, multilingual SFT, preference tuning)을 통해 부분적으로 완화할 수 있음을 발견했습니다.



### BioMNER: A Dataset for Biomedical Method Entity Recognition (https://arxiv.org/abs/2406.20038)
- **What's New**: 이 연구에서는 자동화된 BioMethod 개체 인식 및 정보 검색 시스템을 활용하여 인체 주석(annotator)을 돕는 새로운 생의학 방법 명명 개체 인식(BioMethod NER) 데이터셋을 제안합니다. 이는 인체 주석 속도를 높이고 주석자 간 일치도(inter-agreement)를 향상하는 것을 목표로 합니다.

- **Technical Details**: 연구에서는 다양한 전통적인 및 최신 개체 인식 방법론을 탐구했습니다. 여기에는 대규모 언어 모델(LLM)인 BERT와 ALBERT, BiLSTM, CRF 등이 포함됩니다. 특히, ALBERT 모델(11MB)과 조건부 확률장(CRF)을 결합한 방법이 최첨단 성능(SOTA)을 달성했습니다.

- **Performance Highlights**: 대규모 언어 모델들이 생의학 방법 개체 추출 패턴을 효과적으로 학습하는데 장애가 되는 것으로 나타났습니다. 반면, 소형 ALBERT 모델과 CRF를 조합한 방법이 BioMethod NER 과제에서 최적의 성능을 보였습니다.



### LEMoE: Advanced Mixture of Experts Adaptor for Lifelong Model Editing of Large Language Models (https://arxiv.org/abs/2406.20030)
- **What's New**: 본 논문에서는 지속적인 모델 편집(lifelong model editing)을 위한 고급 전문가 혼합 어댑터(Mixture of Experts, MoE)인 LEMoE를 도입합니다. 전통적인 MoE 어댑터의 문제인 대규모 망각(catastrophic forgetting), 불일치 라우팅(inconsistent routing), 순서 민감도(order sensitivity)를 분석하였고, 이를 해결하기 위해 맞춤형 모듈 삽입 방법과 새로운 KV 앵커 라우팅(Key-Value anchor routing)을 제안합니다.

- **Technical Details**: LEMоE는 맞춤형 모듈 삽입 방법을 사용하여 지속적인 편집을 가능하게 합니다. 데이터 배치와 전문가 네트워크를 정렬시키며, 이전 데이터에 해당하는 전문가 네트워크를 얼려 현재 데이터의 편집이 이전 편집에 악영향을 미치지 않도록 합니다. 또한, 입력 인스턴스별로 키 벡터와 값을 할당하여, 동일한 입력이 학습 및 추론 단계에서 일관되게 동일한 전문가에게 전달되도록 보장합니다. 마지막으로, 클러스터링 기반의 편집 순서 계획을 통해 효율성을 높였습니다.

- **Performance Highlights**: LLaMA-7B와 Mistral-7B 모델을 사용한 실험 결과, LEMoE가 이전 모델 편집 방법을 능가하며, 병렬 편집(batch editing) 작업에서도 우수한 성능을 유지하는 것으로 나타났습니다. 주요 기여는 전통적인 MoE 어댑터의 영향 요소 분석과 새로운 어댑터 설계 및 실험을 통한 방법 유효성 검증입니다.



### ToolBeHonest: A Multi-level Hallucination Diagnostic Benchmark for Tool-Augmented Large Language Models (https://arxiv.org/abs/2406.20015)
- **What's New**: ToolBH라는 새로운 진단 벤치마크가 소개되었습니다. 이 벤치마크는 도구를 사용한 큰 언어 모델(LLMs)의 환각(hallucination) 문제를 깊이와 넓이 두 측면에서 평가합니다. ToolBH는 7가지 과제를 포함하며, 대규모 수작업을 통해 700개의 평가 샘플을 수집하였습니다. 현재 최첨단 모델인 Gemini-1.5-Pro와 GPT-4o는 각각 100점 만점에 45.3점과 37.0점을 획득하여 이 새로운 벤치마크에서의 도전 과제를 잘 보여줍니다.

- **Technical Details**: 환각 평가에서 깊이(depth) 측면으로는 (1) 해결 가능성 감지(solvability detection), (2) 솔루션 계획(solution planning), (3) 누락된 도구 분석(missing-tool analysis)이라는 다단계 진단 프로세스를 제안합니다. 넓이(breadth) 측면에서는 도구셋의 특성에 따라 필요한 도구가 누락된 경우, 잠재적 도구의 경우, 제한된 기능의 도구 경우의 세 가지 시나리오를 고려합니다.

- **Performance Highlights**: 고급 모델인 Gemini-1.5-Pro와 GPT-4o는 총점 100점 만점에 각각 45.3점과 37.0점을 획득했습니다. 훈련 데이터와 응답 전략이 도구 강화 LLM 시나리오에서 중요한 역할을 하며, 큰 모델 파라미터가 더 나은 성능을 보장하지 않는다는 것을 보여줍니다. 주요 오류 원인은 과제 해결 가능성을 평가하는 데 있으며, 오픈 소스 모델은 긴 응답에서 성능 저하를 겪지만, 독점 모델은 더 긴 추론에서 더 우수한 성능을 보였습니다.



### The SIFo Benchmark: Investigating the Sequential Instruction Following Ability of Large Language Models (https://arxiv.org/abs/2406.19999)
- **What's New**: 이번 연구에서는 언어 모델의 다중 지시사항 수행 능력을 평가하기 위한 새로운 벤치마크인 Sequential Instruction Following (SIFo)을 소개합니다. SIFo는 지시사항이 연속적으로 연결되어 있으며, 최종 지시사항만 확인하여 성공 여부를 검사할 수 있습니다. 이를 통해 다중 지시사항 간의 일관성을 보장하고 위치 편향을 피할 수 있습니다.

- **Technical Details**: SIFo 벤치마크는 텍스트 수정, 질문 응답, 수학, 보안 규칙 준수를 포함한 네 가지 과제를 통해 모델의 다중 지시사항 수행 능력을 평가합니다. 최근 및 대형 모델이 오래된 소형 모델에 비해 SIFo 과제에서 뛰어난 성능을 보이며, 이는 벤치마크의 효과를 입증합니다. SIFo 벤치마크 및 소스 코드는 [GitHub 링크](https://github.com/shin-ee-chen/SIFo)에서 확인할 수 있습니다.

- **Performance Highlights**: Mistral, Llama2 (7B, 70B), Llama3 (8B, 70B), Claude-3, GPT-4 등 다양한 최첨단 언어 모델들을 평가한 결과, 최신 모델들이 더 뛰어난 성능을 보였습니다. 그러나 모든 모델이 후속 지시사항을 따르는 데 어려움을 겪고 있으며, 이는 현재 언어 모델이 다중 지시사항을 수행하는 데 있어 강인성이 부족함을 나타냅니다.



### Single Parent Family: A Spectrum of Family Members from a Single Pre-Trained Foundation Mod (https://arxiv.org/abs/2406.19995)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)의 압축을 위한 새로운 방식인 Progressive Low Rank Decomposition(PLRD)을 소개합니다. 이 방법은 사전 훈련된 모델을 점진적으로 낮은 랭크로 분해(decompression)하여 작은 크기의 모델로 만드는 기술입니다. 이를 통해, 재훈련 없이 원본 모델에서 후속 모델을 파생시키며, 계산 오버헤드와 에너지 소비를 크게 줄일 수 있습니다. 실험 결과, PLRD로 훈련된 모델은 1B 토큰으로만 약 전통적인 모델과 비슷한 성능을 유지하면서도, 0.1%의 토큰만을 사용한다는 점을 보였습니다. PLRD의 유연성 덕분에 단 하나의 기본 모델에서 여러 크기의 모델을 생성할 수 있어, 다양한 계산 및 메모리 예산에 맞출 수 있습니다.

- **Technical Details**: PLRD는 낮은 랭크의 텐서로 분해하여 모델 성능과 자원 사용 사이의 트레이드오프를 최적화합니다. 특정 크기의 모델을 훈련할 때마다 원본 모델에서 시작해 점차 크기를 줄이는 방식으로 진행됩니다. 이를 통해, 사전 훈련 없이도 필요한 크기와 성능을 맞출 수 있으며, 파라미터의 수와 계산 복잡도를 크게 줄일 수 있습니다. 또한, LLM들의 구조적 특성인 transformer와 fully connected(FC) layer의 구성 요소를 고려해 저차원 행렬로 분해(SVD와 같은 기법을 활용)합니다.

- **Performance Highlights**: PLRD를 통해 훈련된 모델은 1B 토큰만으로 전통적 훈련 방법과 거의 동등한 성능을 유지하며, 단 0.1%의 토큰만을 사용합니다. 이는, PLRD가 제공하는 효율적인 압축 기술 덕분에 매우 적은 리소스로도 높은 성능을 유지할 수 있음을 보여줍니다. PLRD는 사용자에게 압축 비율에 대한 통제권을 부여하고, 필요에 따라 계산 및 메모리 예산에 맞춰 대형 언어 모델을 원하는 크기로 압축할 수 있습니다. 또한, 실제로 사용자가 필요로 하는 정확한 모델 크기와 자원 사용에 따라 모델을 조정할 수 있습니다.



### Into the Unknown: Generating Geospatial Descriptions for New Environments (https://arxiv.org/abs/2406.19967)
- **What's New**: 이 논문은 새로운 Rendezvous (RVS) 과제를 다룹니다. 이 과제는 시각과 언어의 결합을 위한 Vision-and-Language Navigation (VLN) 과제와 유사하지만, 시각적 단서 대신 지도를 사용한 내비게이션에 중점을 둡니다. 특히 비순차적 내비게이션 명령과 지도를 통해 관점에 독립적인 공간 관계를 추론하는 것을 요구합니다.

- **Technical Details**: 제안된 방법은 새 환경에서의 데이터를 생성하는 대규모 증강 방법을 포함합니다. 이 방법은 개체 관계를 캡처하는 정착된 지식 그래프를 구성하고, 문맥 자유 문법 (CFG)과 대형 언어 모델 (LLM)을 사용하여 내비게이션 명령을 생성합니다. LLM을 통해 생성된 명령과 비교하면 CFG 기반 증강 방법이 더욱 우수한 성능을 보입니다.

- **Performance Highlights**: RVS 데이터셋에 대한 종합 평가에서, 이 방법은 새로운 환경에서 100미터 정확도가 45.83% 향상되었으며, 중간 거리 오류가 39미터 감소했습니다. 학습된 모델이 없는 환경에서도 인간과 AI 간의 성능 격차를 줄이는 데 기여합니다.



### Simulating Financial Market via Large Language Model based Agents (https://arxiv.org/abs/2406.19966)
- **What's New**: 이번 논문에서는 인간의 비합리적 행동을 더 잘 시뮬레이션하기 위해 'Agent-based Simulated Financial Market (ASFM)'를 제안합니다. 이 시스템은 실시간 주문 매칭 시스템을 통해 시뮬레이션된 주식 시장을 구축하고, 대형 언어 모델(Large Language Model, LLM) 기반의 주식 거래 에이전트를 사용합니다. 이 에이전트는 프로필, 관찰, 도구 학습 기반의 행동 모듈로 구성되어 있으며, 시장 동향과 금융 정책 정보를 종합적으로 이해하고, 그들의 거래 전략에 맞는 결정을 내립니다.

- **Technical Details**: ASFM은 실제 금융 시장과 유사한 주문 매칭 메커니즘을 구현하여 주식 거래 시장을 시뮬레이션합니다. 에이전트는 다양한 프로필과 거래 전략을 가지고 있으며, 시장 관찰 모듈과 주식 거래 능력을 갖추고 있습니다. 각 에이전트는 'Sell', 'Buy', 'Hold' 세 가지 도구를 사용하여 거래를 수행합니다. ASFM은 금리 변화와 인플레이션 충격이라는 두 가지 통제 가능한 시나리오에서 실제 주식 시장과 일치하는 반응을 보이며, 거래자의 행동 편향과 대규모 거래자가 미치는 영향 등 경제 연구에서 중요한 문제를 다루기 위한 실험을 수행합니다.

- **Performance Highlights**: 시뮬레이션 결과, ASFM은 현실 세계의 시장 원칙을 효과적으로 시뮬레이션할 수 있음을 확인했습니다. 두 가지 주요 실험 시나리오에서, ASFM의 결과는 최근 경제 연구의 예비 결론과 일치하는 것으로 나타났습니다. 이는 LLM을 경제 및 금융 운영 메커니즘과 통합하여 새로운 경제 연구 패러다임을 제공할 수 있음을 시사합니다.



### BESTOW: Efficient and Streamable Speech Language Model with the Best of Two Worlds in GPT and T5 (https://arxiv.org/abs/2406.19954)
- **What's New**: BESTOW 아키텍처가 제안되었으며, 이는 SpeechLLM(Speech Large Language Model)을 스트리밍과 멀티태스킹이 가능한 형태로 통합하여 실시간 평가를 가능하게 만듭니다. 이전의 Speech-LLaMA 디자인을 개선하여 오프라인 모드와 스트리밍 모드를 통합하였습니다.

- **Technical Details**: BESTOW는 크로스-어텐션 (cross-attention)과 읽기-쓰기 (read-write) 정책 문제를 기반으로 하며, 오프라인 및 스트리밍 모드를 단일 프레임워크로 통합합니다. 이 솔루션은 LLM의 프롬프트 형식을 유지하면서 스트리밍 문제를 해결합니다. 크로스-어텐션 기능 추출기를 도입하여 Speech-LLaMA 아키텍처와 비견되는 성능을 보입니다.

- **Performance Highlights**: BESTOW는 ASR, AST, SQA 등 여러 대규모 멀티태스킹 스피치-투-텍스트 벤치마크에서 최첨단 성능을 자랑하며, 학습 및 추론 비용이 낮습니다. 또 이 솔루션은 87,000시간의 음성 데이터로 확장 가능하며 하루 만에 학습을 완료할 수 있습니다. 코드와 체크포인트가 공개되어 연구를 장려할 예정입니다.



### Mining Reasons For And Against Vaccination From Unstructured Data Using Nichesourcing and AI Data Augmentation (https://arxiv.org/abs/2406.19951)
Comments:
          8 pages + references and appendix

- **What's New**: 이번 연구에서는 Reasons For and Against Vaccination (RFAV)이라는 데이터셋을 소개합니다. 이 데이터셋은 백신 접종 찬반 이유와 이를 정당화하는 과학적 권위에 대한 예측을 위해 작성되었으며, nichesourcing과 GPT4, GPT3.5-Turbo를 사용하여 증강되었습니다. 연구팀은 수동적인 텍스트에서 이러한 이유를 다양한 과제 정의 하에 채굴할 수 있음을 보여주었고, 인공지능을 사용한 데이터 증강의 영향을 탐구했습니다. 이 데이터셋과 훈련된 모델, 그리고 주석 작업에 사용된 매뉴얼도 공개되었습니다.

- **Technical Details**: 연구팀은 백신 관련 키워드를 바탕으로 웹 문서를 수집하였고, 이를 통해 총 136,934개의 영어 문서와 94,361개의 스페인어 문서를 확보했습니다. 문서는 백신 접종 찬반 이유와 과학적 권위로 라벨링되었으며, 각 이유는 백신에 대한 입장을 리커트 척도(Likert scale)로 표시했습니다. 데이터는 nichesourcing 기법으로 주석 작업이 이루어졌으며, GPT4와 GPT3.5-Turbo를 이용해 데이터 증강을 수행했습니다. 모델은 세 가지 태스크로 정의되었으며, 각각의 태스크에서의 성능을 평가했습니다.

- **Performance Highlights**: 연구 결과, 'Reason', 'Compressed Stances'와 'Scientific Authority'는 'moderate agreement' 수준의 일관성을 보였으며, 일부는 'fair agreement' 수준이었습니다. 주석 학습과 데이터 수집에 있어 지속적인 리뷰와 개선 작업을 통해 성능이 향상됨을 확인했습니다. 기존 연구들과 비교했을 때, 충분히 만족할 만한 수준의 성능을 보였습니다.



### Calibrating LLMs with Preference Optimization on Thought Trees for Generating Rationale in Science Question Scoring (https://arxiv.org/abs/2406.19949)
- **What's New**: 이 논문에서는 점수 결정의 타당성을 설명할 수 있는 근거(rationale)를 생성하는 새로운 프레임워크를 제안합니다. 기존의 자동 채점 시스템에서 생성된 근거가 정확도가 낮고 환상 정보를 많이 포함하는 문제가 있었지만, 제안된 프레임워크는 이러한 문제를 해결하고, 분류기 기반(black-box) 채점 시스템과 비슷한 성능을 보입니다.

- **Technical Details**: 본 프레임워크는 먼저 대형 언어 모델(LLMs)을 사용하여 생각의 나무(thought tree)를 생성하고, 각 경로에서 중간 평가 결정을 요약해 합성된 근거 데이터와 선호도 데이터를 만듭니다. 이후, 합성된 데이터를 활용해 두 단계의 학습 과정(지도 학습, 선호도 최적화)을 거쳐 LLM을 조정(calibrate)합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 이전 연구 대비 평가 성능이 38% 향상된 QWK 점수를 기록했습니다. 또한, 인간 평가자와 LLM 모두에 의해 높은 품질의 근거를 생성하는 것으로 인정받았습니다. 합성된 선호도 데이터를 사용한 선호도 최적화 과정에서의 효과가 증명되었습니다.



### From the Least to the Most: Building a Plug-and-Play Visual Reasoner via Data Synthesis (https://arxiv.org/abs/2406.19934)
- **What's New**: 이 논문에서는 비전-언어 모델 (Vision-Language Models, VLMs)에서 다단계 추론을 탐구합니다. 다단계 시각 및 언어 처리 데이터를 구하는 것이 어려운 문제를 극복하기 위해, 질문을 서브 질문으로 분해하고 각 서브 질문을 해결하기 위해 외부 도구를 호출하는 least-to-most 시각 추론 패러다임을 제안합니다. 이를 바탕으로 자동으로 생성된 질문과 다단계 추론 경로를 제공하는 새로운 데이터 생성 방법을 소개합니다. 이 방법을 통해 50,000개의 시각 추론 예제를 구축했습니다.

- **Technical Details**: 이 접근법은 복잡한 작업을 몇 가지 간단한 서브 작업으로 나누고, 오픈소스 모델을 거의 완전히 활용해 서브 작업을 수행합니다. least-to-most 시각 추론 패러다임은 (1) 엔티티 인식(Entity Recognition), (2) 노드 구성(Node Construction), (3) 추론 과정 합성(Reasoning Process Synthesis), (4) 질문 합성(Question Synthesis)으로 구성된 네 단계의 파이프라인으로 이루어집니다. 이 방법은 비용 효율적이고 재현 가능하며 데이터의 질을 보장합니다.

- **Performance Highlights**: 제안된 비주얼 리즈너(Visual Reasoner)는 감독된 미세 조정을 통해 기존 VLM의 추론 능력을 향상시킬 수 있습니다. 4개의 VQA 벤치마크에서 실험 결과, 모든 VLM에서 일관되게 성능을 개선했으며, 절대 성능 향상 범위는 0.71%에서 39%까지 다양합니다.



### Interactive Topic Models with Optimal Transpor (https://arxiv.org/abs/2406.19928)
Comments:
          Pre-print; Work in progress

- **What's New**: 이번 연구에서는 EdTM이라는 새로운 접근 방식을 제안했습니다. 이는 라벨 이름 감독형(topic modeling)을 통해 문서 컬렉션을 분석하는 방법입니다. EdTM은 대규모 언어 모델(LM/LLM) 기반 문서-토픽 유사도를 활용하고 최적 운송(optimal transport)을 사용하여 전 세계적으로 일관된 토픽 할당을 수행합니다. 이를 통해 분석가의 피드백을 반영하면서도 잡음이 있는 입력에도 견고하게 작동할 수 있습니다.

- **Technical Details**: EdTM은 상호작용적 토픽 모델링 프레임워크로, 문서-토픽(score)을 LM/LLM을 사용하여 계산하고, 최적 운송 알고리즘을 이용해 문서-토픽 할당을 수행합니다. 최적 운송은 GPU 계산을 활용할 수 있으며, 문서-토픽 유사도를 전 세계적 최소 비용으로 정렬합니다. 이를 통해, 타당한 페어별 비용을 사용할 때 희소하고 부드러운 할당을 가능케 합니다. 또한, 특정 문서-토픽 할당을 배제하는 부분 할당 알고리즘을 사용하여 사용자 입력의 잡음에 강건하게 대응합니다. 마지막으로, 최적 운송의 고비용 문제를 해결하기 위해 배치 계산을 통해 근사 할당을 수행합니다.

- **Performance Highlights**: 실험 결과, EdTM은 다양한 기준 모델(LDA, 클러스터링 기반 토픽 모델링 등)보다 우수한 성능을 보였습니다. 고품질의 토픽을 유도하고, 분석가의 다양한 형태의 피드백을 지원하며, 잡음이 많은 상황에서도 견고하게 작동할 수 있음을 입증했습니다. 코드와 데이터셋은 논문이 승인되면 공개될 예정입니다.



### Paraphrase Types Elicit Prompt Engineering Capabilities (https://arxiv.org/abs/2406.19898)
- **What's New**: 현대 언어 모델의 성공은 적절한 프롬프트(prompt)를 찾는 것에 크게 의존합니다. 이 연구는 프롬프트의 언어적 표현 변화가 모델에 어떤 영향을 미치는지를 체계적이고 실질적으로 평가합니다. 이를 위해 120개의 과제와 여섯 가지 패러프레이즈(paraphrase) 유형으로 모델의 행동 변화를 측정합니다. 연구 결과, 특정 패러프레이즈 유형에 따라 프롬프트를 조정하면 모델의 성능이 향상될 수 있음을 발견했습니다.

- **Technical Details**: 이 연구는 총 120개의 과제와 5개의 모델에 대해 6개의 패러프레이즈 유형(형태소, 구문, 어휘, 어휘-구문, 담론, 기타)을 사용하여 프롬프트 변화를 측정합니다. 프롬프트 길이, 어휘 다양성, 훈련 데이터와의 근접성 등 다른 프롬프트 엔지니어링 요소들을 통제합니다. 모델의 민감도를 다양한 프롬프트 변화에 대해 평가하고, 어떤 변화가 모델의 성능에 가장 큰 영향을 미치는지 분석합니다.

- **Performance Highlights**: 프롬프트를 특정 유형으로 조정하면 모델의 과제 수행 능력이 향상될 potential이 있음을 보여줍니다. Mixtral 8x7B 모델에서 6.7%의 중간 성능 향상, LLaMA 3 8B 모델에서 5.5%의 성능 향상을 달성했습니다. 특히 형태소와 어휘 변화가 프롬프트 개선에 유망한 성과를 보였습니다.



### Untangling the Unrestricted Web: Automatic Identification of Multilingual Registers (https://arxiv.org/abs/2406.19892)
- **What's New**: 이번 연구는 16개 언어를 다루는 웹 기반 데이터셋에서 텍스트 종류(레지스터)를 자동으로 식별하기 위한 딥러닝 모델을 탐구한 내용입니다. 웹 레지스터 식별은 컴퓨터 언어학에서 웹 스케일 데이터셋의 내용을 이해하기 위한 중요한 솔루션을 제공할 수 있습니다. 새로운 다국어 CORE 코퍼스를 활용하여, 25개의 레지스터를 상세하고 계층적으로 분류하는 새로운 모델을 제안하였습니다.

- **Technical Details**: 본 연구에서는 자세한 계층적 다중 레이블 설정을 사용하여 새로운 다국어 CORE 코퍼스로부터 데이터를 학습시킨 딥러닝 모델을 실험하였습니다. 이 다국어 CORE 코퍼스는 영어, 핀란드어, 스웨덴어, 프랑스어, 터키어를 포함한 16개 언어로 구성되어 있으며 각 언어에 대해 25개의 레지스터로 주석이 달려 있습니다. 초기 모델은 약 80%의 F1 점수를 기록하였으나, 모호한 예시를 제거함으로써 성능을 90% 이상으로 향상시켰습니다. 다국어 모델은 특히 훈련 예시가 적고 레지스터가 작은 언어에 대해 단일 언어 모델보다 더 나은 성능을 보였습니다.

- **Performance Highlights**: 제안한 모델은 최신 상태의 결과를 달성하며, 초기에는 약 80%의 F1 점수를 기록했습니다. 그러나 모호한 예시를 제거한 후 모델 성능을 90% 이상으로 향상시켰습니다. 또한 다국어 모델은 단일 언어 모델보다 더 우수하며, 훈련 데이터가 적은 언어에서 더 큰 장점을 보여줍니다. 제로샷(Zero-shot) 설정에서는 평균적으로 7%의 성능 감소가 있었지만, 이는 특정 레지스터나 언어와는 무관하였습니다.



### Investigating the Timescales of Language Processing with EEG and Language Models (https://arxiv.org/abs/2406.19884)
Comments:
          Accepted at the 2024 Conference on Cognitive Computational Neuroscience (CCN 2024)

- **What's New**: 이 연구는 사전 학습된 트랜스포머 기반 언어 모델(transformer-based language model)의 단어 표현과 EEG 데이터를 정렬하여 언어 처리의 시간적 동학을 탐구합니다. Temporal Response Function (TRF) 모델을 사용하여 신경 활동이 모형 표현에 어떻게 대응하는지를 조사하였습니다. 이를 통해 언어 이해 중 인공 언어 모델과 두뇌 반응 간 상호작용에 대한 통찰력을 제공합니다.

- **Technical Details**: 공개된 EEG 데이터와 GPT-2 모델을 사용하여 단어 표현을 임베딩 레이어와 깊은 레이어에서 추출했습니다. 각 표현을 EEG 데이터와 정렬하기 위해 시간 지연 선형 회귀 모델을 사용했으며, 과적합을 방지하기 위해 L2 정규화(L2 regularisation)를 적용했습니다. 또한, 선형 판별 분석(LDA)를 통해 품사(part-of-speech, POS) 표현을 분리하여 신경 반응과 구문 처리의 기저 메커니즘을 조사했습니다.

- **Performance Highlights**: 임베딩 레이어와 깊은 레이어의 TRF가 현저히 다름을 발견했습니다. 임베딩 레이어는 주로 늦은 시간대에 음수를 나타내는 반면, 깊은 레이어는 200ms 근처에서 강한 음수를, 600ms 이후에서 양의 효과를 보였습니다. 또한, 각 품사 태그는 신경 활동과 200ms 이후의 기간에서 음의 상관관계를 나타냈습니다.



### Detecting Subtle Differences between Human and Model Languages Using Spectrum of Relative Likelihood (https://arxiv.org/abs/2406.19874)
Comments:
          13 pages, 12 figures

- **What's New**: 이 연구는 기존의 절대적 가능도(likelihood) 값 대신 상대적 가능도 값을 사용하여 인간이 작성한 텍스트와 모델이 생성한 텍스트를 구별하는 새로운 접근법을 제안합니다. 이를 통해 보다 미묘한 차이를 밝혀내며, 단기 텍스트 감지에서 최고의 성능을 달성합니다.

- **Technical Details**: 텍스트의 가능도 변화를 스펙트럼-뷰로 캡처하기 위해 Fourier 변환을 사용합니다. 이를 통해 복잡한 가능도 변화를 시간 도메인에서 더욱 간결한 형태로 요약하여 인간과 모델 텍스트 간의 미묘한 차이를 강조합니다. 두 가지 분류 방법, 즉 지도 학습 기반과 휴리스틱 기반 분류기를 설계했습니다. 나이브 n-그램 모델로도 가능도 점수를 추정할 수 있어 연산 비용이 낮습니다. 이 접근법을 FourierGPT라 명명했습니다.

- **Performance Highlights**: 제안된 접근법은 기존의 여러 제로샷(Zero-shot) 감지 방법들과 비교하여 경쟁력 있는 성능을 보이며, 특히 단기 텍스트 감지에서 새로운 최고 성능을 달성했습니다. 또한, 인간과 모델 언어 사이의 미묘한 차이를 이론적으로 설명할 수 있는 기반을 제시합니다.



### YuLan: An Open-source Large Language Mod (https://arxiv.org/abs/2406.19853)
- **What's New**: 본 논문에서는 YuLan이라는 120억 개의 파라미터로 구성된 오픈소스 대규모 언어 모델(LLMs)을 소개합니다. YuLan은 영어, 중국어, 다국어 데이터에서 약 1.7조 개의 토큰으로 학습되었으며, 2024년 1월에 훈련이 완료되었습니다. YuLan은 최신 LLM들과 비슷한 성능을 달성한 것으로 평가되었습니다.

- **Technical Details**: YuLan은 세 단계의 사전 훈련 방법을 통해 설계되었습니다. 첫 번째 단계에서는 다양한 데이터셋에서 균일한 샘플링 전략을 사용하여 자연어와 세계 지식을 학습합니다. 두 번째 단계에서는 지시 조정(instruction-tuning)과 인간 정렬(human alignment)을 통해 고품질의 합성 데이터를 사용합니다. 마지막으로 커리큘럼 학습 프레임워크를 도입하여 복잡하고 드문 지식을 점진적으로 학습합니다. 모델 아키텍처는 Transformer의 디코더만을 사용하며, 이를 통해 자연어 생성과 추론 능력을 극대화했습니다.

- **Performance Highlights**: YuLan은 다국어 텍스트 1.7조 개의 토큰, 4천만 개의 지시 데이터, 20만 개의 인간 정렬 데이터를 사용하여 96개의 NVIDIA A800 GPU에서 학습되었습니다. 22개의 공개 벤치마크 데이터셋에서 최신 오픈소스 LLM과 비슷한 성능을 보였습니다. 이는 YuLan의 능력이 다양한 실제 시나리오에서 복잡한 작업을 성공적으로 처리할 수 있음을 의미합니다.



### AnomaLLMy -- Detecting anomalous tokens in black-box LLMs through low-confidence single-token predictions (https://arxiv.org/abs/2406.19840)
Comments:
          6 pages

- **What's New**: AnomaLLMy라는 새로운 기술이 API 전용 접근 방식에서 블랙박스 대규모 언어 모델(LLMs)의 비정상 토큰을 자동으로 감지하는 방법으로 소개되었습니다. 이는 모델의 신뢰성 및 품질을 저하시키는 비정상 토큰 문제를 해결함으로써 LLM의 견고성과 정확성을 높이는 데 기여합니다.

- **Technical Details**: AnomaLLMy는 모델이 자신 있게 예측하지 못하는 경우를 비용 효율적인 지표로 사용하여 비정상 토큰을 식별합니다. GPT-4의 토큰 세트인 cl100k_base 데이터세트에서 테스트한 결과, API 크레딧 비용 $24.39만으로 413개의 주요 비정상 토큰과 65개의 경미한 비정상 토큰을 감지했습니다. 비정상 토큰을 평가하는 기준으로는 다음과 같은 세 가지가 있습니다: 상위 5개의 예측에서 높은 엔트로피, 다른 토큰 예측에 대한 높은 꼬리 확률, 최상위 및 차상위 예측 간의 낮은 확률 차이.

- **Performance Highlights**: 테스트 결과, AnomaLLMy는 20시간의 실행 시간 동안 478개의 비정상 토큰을 성공적으로 감지했습니다. 주요 비정상 토큰은 모델의 응답에 큰 영향을 미치며, 빈 텍스트 또는 잘못된 예측을 유도하고, API 오류 또는 JSON 스키마 위반을 초래할 수 있습니다. 경미한 비정상 토큰은 종종 오타나 인코딩 문제 등으로 인해 발생하며, 모델 응답이 여전히 일관성이 있습니다.



### BeamAggR: Beam Aggregation Reasoning over Multi-source Knowledge for Multi-hop Question Answering (https://arxiv.org/abs/2406.19820)
Comments:
          Accepted to ACL 2024

- **What's New**: 본 논문에서는 지식 집약 멀티홉 질문 응답(QA)을 위한 새로운 추론 프레임워크인 Beam Aggregation Reasoning(BeamAggR)를 제안합니다. 기존의 복잡한 질문을 해결하는 과정에서 발생하는 부정확한 검색 및 멀티 소스 지식 통합의 어려움을 극복하기 위해, BeamAggR는 각 홉에서 유망한 답변 후보들을 탐색하고 우선시합니다.

- **Technical Details**: BeamAggR는 세 가지 모듈로 구성됩니다: (i) 질문 분해: LLM을 사용해 복잡한 질문을 트리 구조로 변환하고, 원자 질문과 복합 질문으로 분류하여 하향식 추론을 수행합니다. (ii) 보완적 멀티 소스 추론: 원자 질문에 대해서는 멀티 소스 지식을 기반으로 정교한 답변 집계를 수행하여 후보 답변을 생성합니다. (iii) 빔 집합(beam aggregation): 복합 질문에 대해서는 종속된 하위 질문의 조합을 나열하여 추론을 수행한 후, 확률적으로 집계하여 가장 유망한 예측을 선택합니다.

- **Performance Highlights**: 네 가지 오픈 도메인 멀티홉 추론 데이터셋(HotpotQA, 2WikiMQA, MuSiQue, Bamboogle)에서 BeamAggR를 평가한 결과, 기존 최고의 상태로 오픈 도메인 멀티홉 추론 방법들보다 평균 8.5% 향상된 성능을 보였습니다. 추가적인 분석에서는 BeamAggR가 더 나은 지식 협업 및 답변 집계를 이끌어내는 것으로 나타났습니다.



### Scalable and Domain-General Abstractive Proposition Segmentation (https://arxiv.org/abs/2406.19803)
- **What's New**: 이번 연구는 텍스트를 더 단순하고 독립적인 문장으로 변환하는 '추상적 명제 세분화(Abstractive Proposition Segmentation, APS)'를 소개합니다. 여러 NLP 응용 프로그램에서 단순 문장 분할이 불충분하다는 문제를 해결하기 위해, 이 연구는 텍스트를 간단한 의미 단위로 나누는 모델을 제안합니다. 특히, 현존하는 데이터셋을 학습한 대규모 언어모델(LLM)을 활용하여 APS 정확도를 높였으며, 이 모델을 '교사 모델'로 사용해 더 작은 '학생 모델'을 학습시키는 방법도 소개합니다. 마지막으로, 이 기술을 사용할 수 있는 간편한 API도 제공됩니다.

- **Technical Details**: 연구팀은 APS의 품질을 평가하는 평가 메트릭스를 도입했습니다. 제안된 모델은 감독 학습 기반으로, 기존 주석 데이터셋을 활용하여 LLM을 학습시킴으로써 성능을 향상시켰습니다. 더 나아가, 다양한 도메인의 합성 데이터로 모델을 학습하여, 새로운 도메인에도 일반화될 수 있도록 했습니다. 제안한 '학생 모델'은 성능 면에서 교사 LLM에 필적합니다.

- **Performance Highlights**: 제안된 APS 모델은 기존 몇 샷 프롬프트(few-shot prompted) 방법보다 더 높은 정확도를 보였습니다. 또한, 새로운 두 도메인에서 데이터를 주석하고 평가했을 때, 제안된 기술이 효과적으로 성능을 발휘함을 확인했습니다.



### Direct Preference Knowledge Distillation for Large Language Models (https://arxiv.org/abs/2406.19774)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs) 분야에서 지식 증류(Knowledge Distillation, KD)를 위한 새로운 접근법인 Direct Preference Knowledge Distillation(DPKD)를 제안합니다. DPKD는 기존 KL 발산(KL divergence)의 한계와 문제점을 보완하고, LLM의 암묵적 보상 함수(implicit reward function)를 통해 성능을 끌어올리는 방법을 소개합니다.

- **Technical Details**: DPKD는 암묵적 보상 함수와 역방향 KL 발산(reverse KL divergence)을 사용하여 최적화 목표를 재구성하고, 그 이후에 학생 모델과 교사 모델의 출력 선호도(preference probability)를 개선합니다. 이는 KL 발산의 단점을 보완하면서도 높은 효율성을 유지할 수 있습니다. 또한, Bradley-Terry 모델을 사용하여 선택된 출력 간의 확률 비교를 통해 선호도를 측정합니다.

- **Performance Highlights**: 다양한 데이터셋과 LLM 파라미터(120M ~ 13B)를 통해 실험을 진행한 결과, DPKD 방법이 기존의 기준 방법(baseline method)보다 출력 응답 정확도(output response precision)와 정확 일치율(exact match percentage)에서 더 나은 성과를 보였습니다. 또한, 암묵적 보상 함수와 선호도 모델을 통해 KD 과정의 재구성의 효과를 이론적으로 증명하고 실험을 통해 검증하였습니다.



### Belief Revision: The Adaptability of Large Language Models Reasoning (https://arxiv.org/abs/2406.19764)
- **What's New**: 이 논문에서는 텍스트로부터 추론하는 능력이 현실 세계에서 NLP 응용 프로그램에 중요하다는 사실을 강조합니다. 기존의 평가에서는 일관된 정보로 작동하는 언어 모델(LM)을 가정하지만, 이 논문은 새로운 증거가 주어질 때 LM의 신념 수정 능력을 테스트하기 위한 Belief-R이라는 새로운 데이터셋을 소개합니다. 이 데이터셋은 인간이 새로운 정보를 바탕으로 이전의 결론을 억제하는 방식을 모방하여 설계되었습니다.

- **Technical Details**: Belief-R 데이터셋은 델타 추론($\Delta R$) 프레임워크를 통해 LM의 신념 수정 능력을 평가하도록 고안되었습니다. 이 데이터셋은 초기 전제와 새로운 정보를 포함한 시퀀스로 구성되어 있으며, LM이 새로운 정보를 기반으로 이전의 결론을 수정해야 하는 시나리오를 시뮬레이션합니다. 새로운 데이터셋은 기본적인 모순 긍정(modus ponens) 또는 부정(modus tollens) 추론을 지원하는 두 개의 초기 전제와 추가적인 정보를 포함하는 새로운 전제로 구성됩니다.

- **Performance Highlights**: 다양한 프롬프트 전략을 사용하여 약 30개의 LM을 평가한 결과, 대부분의 LM이 새로운 정보에 따라 적절하게 신념을 수정하는 데 어려움을 겪는다는 것을 발견했습니다. 또한, 업데이트에 능숙한 모델들은 업데이트가 필요하지 않은 시나리오에서 상대적으로 성능이 저하되는 경향을 보여 중요한 트레이드오프를 강조했습니다. 이러한 결과는 변화하는 정보에 대한 LM의 적응력을 향상시키는 것이 신뢰할 수 있는 AI 시스템을 위한 중요한 단계임을 강조합니다.



### Breaking the Script Barrier in Multilingual Pre-Trained Language Models with Transliteration-Based Post-Training Alignmen (https://arxiv.org/abs/2406.19759)
Comments:
          preprint

- **What's New**: 최근 다중언어 사전 학습 모델(mPLMs)이 다중언어 전이 작업에서 놀라운 성능을 보여주고 있지만, 저자원 언어가 상위 자원 언어와 다른 스크립트로 작성된 경우 전이 성능이 저하되는 문제를 발견했습니다. 이 문제를 해결하기 위해, 이 논문은 다양한 스크립트를 사용하는 언어 사이의 다중언어 정렬을 개선하기 위한 음역 기반 후사전 훈련 정렬(PPA) 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 Mediterranean-Amharic-Farsi 및 South+East Asian 언어 그룹을 대상으로 합니다. 이 언어들은 상호 영향을 받지만 각기 다른 스크립트를 사용합니다. 본 연구에서는 Uroman이라는 규칙 기반 음역 도구를 사용하여 원래 스크립트와 라틴 스크립트 음역본을 사용해 단말 및 시퀀스 레벨에서 모델을 후사전 훈련합니다. 이 방법은 병렬 데이터에 의존하지 않고, 단일 공통 스크립트 사용의 제한을 극복하며, 동일한 언어 군에서 더 나은 소스 언어 선택을 통해 전이 성능을 더욱 향상시킵니다.

- **Performance Highlights**: 실험 결과에 따르면, PPA 후 모델은 영어 중심 전이 작업에서 최대 50%까지 성능이 향상되었습니다. 또한, 영어 외 다른 언어를 소스로 사용할 때 더욱 큰 개선이 나타났습니다. 이는 제안된 방법이 소스와 타겟 언어 간의 정렬을 높이고 전반적인 다중언어 정렬 성능을 크게 향상시킴을 보여줍니다.



### Message du troisi{\`e}me type : irruption d'un tiers dans un dialogue en lign (https://arxiv.org/abs/2406.19731)
Comments:
          in French language. JADT 2024 - 17es Journ{é}es internationales d'Analyse statistique des Donn{é}es Textuelles, SeSLa (S{é}minaire des Sciences du Langage de l'UCLouvain -- Site Saint-Louis); LASLA (Laboratoire d'Analyse statistique des Langues anciennes de l'Universit{é} de Li{è}ge), 2024, Bruxelles, Belgique

- **What's New**: 이번 연구는 프랑스어 위키피디아 토론 페이지(Wikipedia talk pages)에 초점을 맞추어, 온라인 상에서 기여자들의 행동을 글로벌 관점에서 분석합니다. 300,000개 이상의 토론 스레드(discussion threads)를 포함하는 데이터셋을 사용하여 다자간 대화(multiparty conversation)가 어떻게 전개되는지를 조사하고, 두 명의 위키피디언(Wikipedians)이 교환을 시작한 후 세 번째 참여자의 개입 역할을 특히 탐구합니다.

- **Technical Details**: 연구는 참여자들 간의 상호작용에서 순차적 구조(sequential structure)에 중점을 두고, 제3 메시지(third message)의 어휘적 특성(lexical particularities)을 탐색하여 이를 구체화합니다. 이를 위해 제3 참여자 메시지의 초기 유형학(initial typology)을 제시하고, 이 메시지가 선행 메시지와 어떻게 조화되는지를 분석합니다.

- **Performance Highlights**: 본 연구는 제3 참여자의 개입이 기존의 대화 흐름에 어떻게 영향을 미치는지를 이해함으로써, 온라인 커뮤니티의 상호작용 구조를 보다 명확히 하고 향후 커뮤니케이션 도구 개발에 기여할 수 있는 인사이트를 제공합니다.



### Le sens de la famille : analyse du vocabulaire de la parent{\'e} par les plongements de mots (https://arxiv.org/abs/2406.19729)
Comments:
          in French language. JADT 2024 - 17es Journ{é}es internationales d'Analyse statistique des Donn{é}es Textuelles, SeSLa (S{é}minaire des Sciences du Langage de l'UCLouvain -- Site Saint-Louis), 2024, Bruxelles, Belgique

- **What's New**: 이번 연구에서는 밀집하고 매우 구조화된 프랑스어 어휘, 특히 가족 관계에 관련된 어휘를 대상으로 한 코퍼스 분석을 제안합니다. 25개의 주요 관계(예: 아들, 사촌, 어머니, 할아버지, 시누이 등)를 나타내는 명사들을 활용하여 이 용어들이 서로 어떻게 배치되는지 분포적 분석을 통해 조사합니다.

- **Technical Details**: 연구는 코퍼스(corpora)에서 이러한 용어의 사용을 기반으로 분포적(distributional) 정보를 통해 이 어휘를 구성하는 특정 특징(하강, 동맹, 형제자매, 성별)을 포착할 수 있음을 보입니다. 이 정보는 다른 코퍼스 비교 시 다양한 방식으로 다르게 나타납니다.

- **Performance Highlights**: 본 연구는 분포적 정보가 하강(descendant), 동맹(alliance), 형제자매(siblings), 성별(genre) 등 가족 관계 어휘를 조직하는 요소를 파악할 수 있음을 입증했습니다.



### Less is More: Accurate Speech Recognition & Translation without Web-Scale Data (https://arxiv.org/abs/2406.19674)
Comments:
          Accepted at Interspeech-2024

- **What's New**: 최근의 많은 음성 인식과 번역 기술이 방대한 인터넷 음성 데이터에 의존하는 반면, Canary라는 다국어 ASR(Automatic Speech Recognition) 및 음성 번역 모델은 이러한 대규모 데이터 없이도 최첨단 정확성을 달성할 수 있음을 입증했습니다. Canary는 기존의 Whisper, OWSM, 및 Seamless-M4T 모델을 능가하며, 영어, 프랑스어, 스페인어 및 독일어에서 더 적은 데이터로 학습되었습니다. Canary는 FastConformer 기반의 어텐션 인코더-디코더 아키텍처, 머신 트랜스레이션을 통한 합성 데이터 사용, 데이터 균형 조정 및 노이즈 견고한 미세 조정을 포함한 고급 학습 기술을 활용합니다.

- **Technical Details**: Canary 모델의 주요 기술적 특징은 다음과 같습니다:
1. **FastConformer 기반 어텐션 인코더-디코더 아키텍처**: 기존의 Conformer를 변형하여 더 높은 다운샘플링 비율을 제공하며, 이는 2.8배의 속도 향상을 가져옵니다.
2. **합성 데이터 활용**: 기계 번역을 통해 생성된 합성 데이터를 사용하여 AST 모델을 학습합니다.
3. **다양한 학습 기술**: 데이터 균형 조정, 동적 데이터 블렌딩, 동적 버켓팅 및 노이즈-견고한 미세 조정 기술을 채택하여 효율적인 학습을 달성합니다.

- **Performance Highlights**: Canary 모델은 다음과 같은 주요 성능 성과를 보여줍니다:
1. 기존의 Whisper, OWSM 및 Seamless-M4T 모델보다 높은 정확도를 달성합니다.
2. 128개 NVIDIA A100 80GB GPU를 사용하여 48시간 이내에 학습을 완료합니다.
3. 기존 AST 데이터 없이도 높은 정확도를 유지하며, 86,000 시간의 음성 데이터만으로 훈련되어 기존 모델보다 데이터 효율성이 뛰어납니다.



### DECOR: Improving Coherence in L2 English Writing with a Novel Benchmark for Incoherence Detection, Reasoning, and Rewriting (https://arxiv.org/abs/2406.19650)
Comments:
          21 pages, 5 figures, 20 tables

- **What's New**: DECOR는 L2 영어 글쓰기에서 일관성 검출 및 수정에 초점을 맞춘 최초의 벤치마크 데이터셋입니다. 기존 자동 글쓰기 평가 시스템이 표면적인 언어적 특징을 이용해 일관성을 검출하는 데 그쳤다면, DECOR는 검출된 비일관성을 전문가가 수정한 예시를 포함해 L2 학습자의 글쓰기 향상을 도모합니다.

- **Technical Details**: DECOR는 TOEFL-11 코퍼스를 기반으로 문맥-문장 쌍(context-sentence pairs)을 구축합니다. 데이터셋은 세 가지 주요 작업 - 비일관성 감지, 비일관성의 이유 추론, 비일관성 문장 재작성 - 을 포함하도록 설계되었습니다. 코히션(Cohesion), 일관성(Consistency), 관련성(Relevance)의 조건을 기반으로 조정된 주석 스키마(annotation scheme)를 사용합니다.

- **Performance Highlights**: 실험 결과, 비일관성 감지 및 문장 재작성 모델이 GPT-4에 비해 크기가 작고 비용이 적게 들면서도 유사한 성능을 보였습니다. 특히 비일관성 이유를 포함한 파인튜닝(fine-tuning)이 모델의 재작성 품질을 일관되게 향상시켜, 전문가 수준의 결과를 자동 및 인간 평가에서 인정받았습니다.



### Unlocking Varied Perspectives: A Persona-Based Multi-Agent Framework with Debate-Driven Text Planning for Argument Generation (https://arxiv.org/abs/2406.19643)
- **What's New**: 이번 연구에서는 불확실한 주제에 대해 더 설득력 있는 논증(Argument)을 작성하기 위해 다중-요원(Multi-Agent) 논쟁 방식 기반의 인격 기반 프레임워크(Persona-Based Framework)를 제안합니다. 각 에이전트(agent)에게 고유한 인격(Persona)를 부여하여 서로 다른 관점과 신념을 바탕으로 논쟁을 펼치며 최종적인 논증 계획을 구성합니다. 이 프레임워크는 다양한 관점과 논리를 포함하여 더 논리적이고 설득력 있는 에세이를 만들어냅니다.

- **Technical Details**: 이 프레임워크는 다음과 같은 단계를 거쳐 논증을 생성합니다. 먼저 주어진 주제에 따라 고유한 인격을 각 에이전트에게 부여합니다(인격 배정). 그 다음 에이전트들끼리 논쟁을 통해 고차원의 논증 계획을 세웁니다(논쟁 기반 계획). 마지막으로 이 계획을 바탕으로 에세이를 실제 텍스트로 전환합니다. 비판 에이전트(Critic Agent)를 추가하여 토론에서 약점을 지적하며 최종 논증 계획을 더욱 견고히 합니다.

- **Performance Highlights**: 이 프레임워크는 idebate와 reddit/CMV 포털에서 수집된 주제를 바탕으로 평가되었으며, 자동 평가와 인간 평가에서 기존 방법보다 더 다양한 관점과 논리적으로 일관된 논증을 생성하는 것으로 나타났습니다. 주된 기여는 (1) 인격 기반 다중-요원 접근 방식을 사용해 다양한 관점을 확보, (2) 유동적이고 비선형적인 아이디어 발전을 허용하는 논쟁 기반 계획, (3) 장문의 논증에서 관점 다양성을 평가하는 새로운 자동화 메트릭을 설계한 것입니다.



### IDT: Dual-Task Adversarial Attacks for Privacy Protection (https://arxiv.org/abs/2406.19642)
Comments:
          28 pages, 1 figure

- **What's New**: 이 논문은 자연어 처리 (NLP) 모델이 사용자 프라이버시를 어떻게 침해할 수 있는지와 이에 대한 방어 방법에 대해 논의합니다. 특히, 텍스트를 모델에 전달하기 전에 재작성(rewrite)을 통해 민감한 속성을 숨기는 방법을 탐구합니다.

- **Technical Details**: 논문에서 제안하는 방법(IDT)은 'adversarial attack' 기법을 변형하여 텍스트를 조작함으로써 하나의 분류기(privacy)에서는 속일 수 있지만, 다른 분류기(utility)에서는 예측을 유지할 수 있도록 합니다. 보조적이고 해석 가능한 모델을 사용해 어떤 토큰을 변경해야 하고 어떤 토큰을 유지해야 하는지를 분석합니다.

- **Performance Highlights**: 다양한 NLP 데이터셋을 사용하여 자동 및 인간 평가를 수행한 결과, IDT는 기존 방법에 비해 텍스트의 유용성을 유지하면서 프라이버시 분류기를 속이는 성능이 뛰어남을 입증했습니다.



### Mixture of In-Context Experts Enhance LLMs' Long Context Awareness (https://arxiv.org/abs/2406.19598)
Comments:
          14 pages, 5 figures

- **What's New**: 본 논문은 'Mixture of In-Context Experts' (MoICE)라는 새로운 방법을 소개하여 대형 언어 모델(LLMs)에서 RoPE를 위치 임베딩으로 사용할 때의 문맥 인식 문제를 해결합니다. 기존의 방법들이 문맥 인식 능력을 강화하기 위해 다양한 접근을 시도했으나, 효율성과 효과를 동시에 달성하는 데 어려움이 있었습니다. MoICE는 두 가지 주요 구성 요소로 이루어져 있으며, 각각의 어텐션 헤드 내에 라우터를 통합하고, 경량 라우터 전용 학습 최적화 전략을 통해 높은 문맥 인식을 제공합니다.

- **Technical Details**: MoICE는 두 가지 주요 실행 방법을 도입합니다. 첫째, 각 RoPE 각도를 문맥 내 전문가로 간주하여 특정 문맥 위치로 어텐션을 유도합니다. 따라서 각 어텐션 헤드는 라우터에 의해 선택된 다양한 RoPE 각도를 사용하여 동적으로 토큰을 처리합니다. 이는 중요한 문맥 정보를 간과할 위험을 줄여줍니다. 둘째, 라우터만을 업데이트하는 라우터 전용 학습 전략을 사용합니다. 이 전략은 LLM의 파라미터를 고정시키고 라우터만 몇 단계 만에 업데이트하여 효율성을 높입니다.

- **Performance Highlights**: MoICE를 공개 소스 LLM인 Llama와 Mistral에 적용한 결과, 길고 복잡한 문맥 이해와 생성 작업에서 이전 방법을 뛰어넘는 성능을 나타냈습니다. 특히, MoICE는 감탄할 만한 추론 효율성을 유지하면서도 뛰어난 문맥 인식 능력을 보여주었습니다.



### SK-VQA: Synthetic Knowledge Generation at Scale for Training Context-Augmented Multimodal LLMs (https://arxiv.org/abs/2406.19593)
- **What's New**: 최근에는 대규모 시각 및 언어 모델 훈련을 위한 합성 데이터 생성이 주목받고 있습니다. 하지만, 멀티모달(context-augmented) 생성 시스템 훈련을 위한 합성 데이터 적용은 상대적으로 탐구되지 않았습니다. 이를 해결하기 위해, 우리는 SK-VQA라는 대규모 합성 멀티모달 데이터셋을 생성했습니다. 이 데이터셋은 외부 지식을 통해 최종 답변을 도출할 수 있는 200만 개 이상의 질문-답변 쌍을 포함하고 있습니다.

- **Technical Details**: 우리의 데이터셋은 기존 자원보다 크고, 더 다양하며 11배 더 많은 유일한 질문과 여러 출처의 이미지를 포함하고 있습니다. 이를 통해 기존의 생성 멀티모달 모델 multi-modal models)을 context-augmented generation에 적응시키기 위한 자원으로 활용될 수 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 우리의 합성 데이터셋이 도전적인 벤치마크로 작동할 뿐만 아니라 기존의 생성 멀티모달 모델을 context-augmented generation로 적응시키는 데 매우 효과적임을 입증했습니다.



### Voices Unheard: NLP Resources and Models for Yor\`ub\'a Regional Dialects (https://arxiv.org/abs/2406.19564)
- **What's New**: 본 연구는 여러 방언을 포함한 고품질의 평행 텍스트 및 음성 코퍼스 YORÙLECT을 소개합니다. 이는 표준 요루바어와 지역 방언들 간의 성능 격차를 좁히기 위해 개발된 첫 번째 시도로, 방언 적응(finetuning)을 통해 성능 향상을 도모했습니다.

- **Technical Details**: YORÙLECT 코퍼스는 세 가지 도메인(종교, 뉴스, Ted 강연)에서 수집된 텍스트 및 음성 데이터를 포함하며, 네 가지 요루바 방언(표준 요루바어, Ifẹ̀, Ìlàjẹ, Ìjẹ̀bú)으로 구성되어 있습니다. 이를 위해 원어민과 협력하여 각 방언이 사용되는 지역 사회에서 데이터를 수집하였습니다. 이 코퍼스는 텍스트-텍스트 기계 번역(MT), 자동 음성 인식(ASR), 음성-텍스트 번역(S2TT) 등의 다양한 실험에 사용되었습니다.

- **Performance Highlights**: 실험 결과, 표준 요루바어와 다른 방언 간의 성능 차이가 크다는 것을 발견했습니다. 하지만 각 방언에 맞춘 미세 조정(finetuning)을 통해 이러한 격차를 줄일 수 있었습니다. 텍스트-텍스트 기계 번역과 음성-텍스트 번역의 경우 평균적으로 BLEU 점수가 각각 14점 및 5점 향상되었으며, 자동 음성 인식(ASR)의 경우 단어 오류율(word-error-rate)이 20점 감소했습니다.



### Rethinking harmless refusals when fine-tuning foundation models (https://arxiv.org/abs/2406.19552)
Comments:
          ICLR 2024 AGI Workshop Poster

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 미세 조정이 얼마나 효과적으로 바람직하지 않은 행동을 완화하는지, 아니면 단지 숨기는지에 대해 조사합니다. 이를 위해 반(半)현실적인 롤플레잉 시나리오를 통해 이러한 행동을 유도하고, 미세 조정 후 모델의 응답 동태를 탐구합니다.

- **Technical Details**: 우리의 방법론은 Chain-of-Thought(CoT) 추론을 모델에게 제안하여, 추론 과정과 최종 출력 간의 일관성을 분석하는 것입니다. 특히, 우리는 '이유 기반 기만(reason-based deception)'이라는 널리 퍼진 현상을 발견했습니다. 이는 모델이 더 이상 추론 과정을 생성하지 않거나, 윤리적으로 보이는 추론 과정을 생성하지만 최종 출력은 비윤리적인 양상을 띠는 현상을 말합니다.

- **Performance Highlights**: 우리는 반박(explicit rebuttal)과 정중한 거절(polite refusal)이 다음 출력에서 바람직하지 않은 행동을 줄이는 데 얼마나 효과적인지 조사했습니다. 결과적으로, 반박은 정중한 거절에 비해 바람직하지 않은 출력을 방지하는 데 있어 훨씬 탁월한 성과를 보였으며, 이유 기반 기만 현상을 거의 제거했습니다. 이는 모델 미세 조정 방법에서 응답 전략을 재고할 필요성을 강조합니다.



### Leveraging Machine-Generated Rationales to Facilitate Social Meaning Detection in Conversations (https://arxiv.org/abs/2406.19545)
Comments:
          To appear at The Proceedings of the Association for Computational Linguistics, 2024

- **What's New**: 이 논문은 대화에서 암묵적으로 인코딩된 사회적 의미를 탐지하기 위해 대형 언어 모델(LLMs)을 활용하는 일반화 가능한 분류 접근 방식을 제안합니다. 여러 측면의 프롬프트를 설계하여 가시적인 단서와 기저의 사회적 의미를 연결하는 추론의 텍스트 설명을 추출합니다. 이러한 설명 또는 합리성을 대화 텍스트에 추가하여 대화 이해와 전이를 촉진합니다.

- **Technical Details**: 다양한 실험 설정에서 합리성을 추가하는 것이 인-도메인 분류(in-domain classification), 제로샷(zero-shot) 및 소수샷(few-shot) 도메인 전이에서 긍정적인 영향을 미친다는 것을 실증적으로 입증했습니다. 구체적으로 저항 전략과 감정 인식을 위한 두 가지 사회적 의미 탐지 작업에서 실험을 수행했습니다. 이 논문에서는 합리성을 생성하기 위해 사회언어학 이론에 기반한 다각적인 프롬프트 프레임워크를 설계했습니다.

- **Performance Highlights**: 실험 결과, 합리성을 추가한 모델이 두 작업 및 설정 모두에서 기본 모델보다 성능이 현저히 향상되었습니다. 특히 데이터가 적은 환경에서는 더욱 큰 성능 향상을 보였습니다. 이것은 이 접근 방식의 일반화 가능성을 강조합니다. 연구 결과를 공공 자원으로 제공하여, 오픈 소스 해결책 개발을 장려합니다.



### Demarked: A Strategy for Enhanced Abusive Speech Moderation through Counterspeech, Detoxification, and Message Managemen (https://arxiv.org/abs/2406.19543)
- **What's New**: 이번 연구에서는 디지털 폭력에 대한 새로운 접근법인 Demarcation을 제안합니다. 이는 기존의 단편적인 차단이나 금지 대신, 악성 발언을 네 가지 기준에 따라 평가하고 대응 옵션을 제시합니다: (i) 심각도 (severity scale); (ii) 대상 존재 여부 (presence of a target); (iii) 맥락 (context scale); (iv) 법적 기준 (legal scale). 이를 통해, 텍스트 해독 (detoxification), 반박 발언 생성 (counter speech generation), 차단, 최종적으로는 인간 개입 (human intervention) 등을 포함한 다단계 대응을 제안합니다.

- **Technical Details**: 본 논문에서는 다양한 국가, 소셜 미디어 플랫폼, 연구 논문에서의 악성 발언 규정을 심층 분석하여 이를 바탕으로 Demarcation이라는 종합적인 대응 시스템을 제안합니다. 이 시스템은 텍스트 해독, 반박 발언 생성, 차단 및 최종적인 인간 개입을 포함한 여러 단계의 프로세스로 구성되어 있습니다. 기존의 차단 위주 접근 방식의 한계를 지적하고, 반박 발언 생성 및 텍스트 해독과 같은 새로운 접근법을 통합하였습니다.

- **Performance Highlights**: Demarcation은 기존의 단편적인 차단 및 금지 방식보다 더 지속적이고 복합적인 접근법을 통해 악성 발언 감소에 기여할 수 있습니다. 특히, 반박 발언 생성 및 텍스트 해독 등은 단순 차단보다 긴 지속성을 가지며, 온라인 커뮤니티의 긍정적 변화를 도모할 수 있습니다.



### Context Matters: An Empirical Study of the Impact of Contextual Information in Temporal Question Answering Systems (https://arxiv.org/abs/2406.19538)
- **What's New**: 이번 논문은 대형 언어 모델(LLMs)이 시간 기반 질의응답(Temporal Question Answering, TQA)에서 여러 상황(context) 유형에 대한 견고성을 실증적으로 분석하여 개선 방안을 제시합니다. 저자들은 새로운 두 가지 TQA 데이터셋, ContextAQA와 ContextTQE를 소개하며, 이 데이터셋들을 사용해 다양한 문맥 환경에서의 모델 성능을 평가했습니다.

- **Technical Details**: 연구는 관련된, 무관한, 약간 변경된, 그리고 문맥이 없는 상황에서 훈련된 TQA 시스템의 견고성을 조사하였습니다. 문맥의 위치 또한 성능에 중요한 영향을 미치는 것으로 나타났습니다. 질문 앞에 문맥을 배치하는 것이 더 나은 성과를 보였습니다. 주요 기술적 접근법으로는 Retrieval-Augmented Generation (RAG) 과 같은 정보 검색 기반 모델 그리고 불필요한 문맥을 필터링하는 자연어 추론 필터 등이 있습니다.

- **Performance Highlights**: 다양한 문맥 유형 혼합으로 훈련된 모델은 순수하게 관련된 문맥으로만 훈련된 모델에 비해 정확도가 7.5% 향상되었습니다. 이는 신뢰할 수 있는 시간기반 질의응답 시스템을 개발하는 데 중요한 통찰을 제공합니다.



### Handling Ontology Gaps in Semantic Parsing (https://arxiv.org/abs/2406.19537)
- **What's New**: 최근 Neural Semantic Parsing (NSP) 모델 개발에서 큰 문제로 떠오른 'hallucination(환각)' 문제를 해결하기 위해 Hallucination Simulation Framework (HSF)를 제안했습니다. 이 프레임워크는 모든 NSP 작업에 적용 가능하며, 특히 폐쇄 온톨로지(closed-ontology)의 문제를 다루는 데 중점을 둡니다. 이번 연구는 또한 KQA Pro 벤치마크 데이터셋을 사용하여 최신 기술들을 평가하고 새로운 환각 탐지 전략을 도입하여 F1-Score가 약 21%, 24% 및 1%까지 향상됨을 보여줍니다.

- **Technical Details**: Semantic Parsing(SP)은 복잡한 자연어를 기계가 이해할 수 있는 언어(e.g., SQL, SPARQL 등)로 변환하는 작업을 의미합니다. 하지만 대부분의 NSP 모델은 '폐쇄 세계(closed-world)' 가정 하에 개발됩니다. 이 가정 하에서는 온톨로지(ontology)에 없는 개념에 대해 모델이 알지 못하더라도, 언제나 유효한 타겟 논리 형식(target logical form)을 만들어내려고 합니다. 이를 위해 HSF는 NSP 모델의 환각을 자극하고 분석하는 일반 설정을 제공하며, 구체적으로 Hallucination Detection Model (HDM)을 통해 인식 신호를 활용해 NSP 모델이 환각을 일으키는지 여부를 판단합니다. 특히 모델의 활성화(activations)를 환각 탐지 신호로 사용하여 성능을 크게 향상시켰습니다.

- **Performance Highlights**: 제안된 프레임워크를 사용하여 폐쇄 온톨로지에서 발생하는 문제를 해결하는 데 성공했으며, KQA Pro 벤치마크 테스트에서 환각 탐지 성능이 눈에 띄게 향상되었습니다. 구체적으로, 온톨로지 갭 문제 해결에서 약 21%, NSP 오류 탐지에서 1%, 그리고 도메인 외 질의(out-of-domain, OOD) 탐지에서는 24%의 F1-Score 향상을 이루었습니다.



### TocBERT: Medical Document Structure Extraction Using Bidirectional Transformers (https://arxiv.org/abs/2406.19526)
Comments:
          6 pages, 6 figures

- **What's New**: TocBERT는 새로운 bidirectional transformer 기반의 텍스트 분할 솔루션으로, 제목(title) 및 부제(sub-title)를 감지하여 텍스트를 분할합니다. 특히, Bio-ClinicalBERT 모델을 활용하여 MIMIC-III 데이터셋의 의료 문서를 효과적으로 분할하고 평가하였습니다.

- **Technical Details**: TocBERT는 제목 및 부제의 의미적 표현을 기반으로 감독 학습된(supervised) 솔루션입니다. 이 방법은 named entity recognition (NER) 문제로 공식화되었으며, Bio-ClinicalBERT 모델을 fine-tuning하여 의료 문서에서 제목과 부제를 감지합니다. 이와 같은 작업은 MIMIC-III 데이터셋의 퇴원 요약 보고서에서 수행되었습니다.

- **Performance Highlights**: TocBERT는 인간이 라벨링한 250개의 노트를 평가하여, 연속적 텍스트 분할(linear text segmentation) 문제에서 84.6%, 계층적 텍스트 분할(hierarchical text segmentation) 문제에서 72.8%의 F1-score를 달성하였습니다. 이는 특히 제목과 부제를 구별하는데 있어, 정교하게 설계된 규칙 기반 솔루션을 능가했습니다.



### Captioning Visualizations with Large Language Models (CVLLM): A Tutoria (https://arxiv.org/abs/2406.19512)
Comments:
          6 pages, 4 figures

- **What's New**: 대형 언어 모델 (Large Language Models, LLMs)의 최근 진보는 정보 시각화 (Information Visualization, InfoVis) 및 자막 생성 분야에 새로운 가능성을 열었습니다. 이 튜토리얼에서는 InfoVis 원칙과 과거 작업에 대한 간략한 리뷰를 제공한 후, 일반적인 LLM에 사용되는 신경 모델과 트랜스포머 아키텍처를 도입합니다. 또한, InfoVis에 LLM을 적용하는 최근 사례와 이 분야의 향후 유망한 연구 방향을 탐구합니다.

- **Technical Details**: 이 튜토리얼은 두 부분으로 나눠져 있으며, 각각 90분 동안 진행되며 30분의 휴식 시간이 포함됩니다. 첫 번째 부분에서는 시각화 디자인에 필요한 데이터 추상화 및 사용자 작업 식별의 중요성을 다룹니다. 두 번째 부분에서는 LLM의 작동 메커니즘, 특히 신경 네트워크와 트랜스포머 아키텍처를 설명하며, CoT, RAG, RLHF와 같은 최신 기술들이 자막 생성의 한계를 어떻게 극복하는지에 대해 논의합니다.

- **Performance Highlights**: Kantharaj et al. (2022)의 대규모 데이터셋(44,096 항목), Tang et al. (2023)의 구조적 자막을 포함한 데이터셋(12,441 항목) 등 최근 연구의 주요 기여들을 검토합니다. Li et al. (2024)의 다중 그림 및 문맥 자막 생성 과제와 Liu et al.의 단계별 학습 및 정확한 질문 답변 기법을 다룹니다. 또한, Singh et al. (2023)은 RLHF 방법을 통한 자막 모델 최적화를, Huang et al. (2024)는 사실적 오류 분석 및 종합적 평가를 강조합니다.



### Are Generative Language Models Multicultural? A Study on Hausa Culture and Emotions using ChatGP (https://arxiv.org/abs/2406.19504)
- **What's New**: ChatGPT와 같은 대형 언어 모델(LLMs)이 저자원 언어의 문화와 감정을 어떻게 반영하는지 조사한 연구입니다. 특히 Hausa 문화를 중심으로, ChatGPT의 응답과 Hausa 원어민의 응답을 비교했습니다.

- **Technical Details**: 연구는 37개의 문화적으로 중요한 질문에 대해 ChatGPT와 Hausa 원어민의 응답을 비교했습니다. 감정 분석과 두 가지 유사성 지표(BERTScore, METEOR)를 적용하여 인간 응답과 ChatGPT 응답의 일치도를 평가했습니다. 또한, 인간 참가자는 ChatGPT 응답의 문화적 표현과 감정적 내용을 5단계 Likert 척도로 평가했습니다.

- **Performance Highlights**: 결과적으로 ChatGPT는 인간 응답과 일부 유사성을 보였으나, Hausa 문화와 감정에 대한 지식과 인식에 부족함과 편견이 나타났습니다. 감정 분석 결과 ChatGPT 응답은 대부분 중립적이거나 복합적이었고, 인간 응답은 다양한 감정을 나타냈습니다. BERTScore는 높은 의미적 유사성을 보였으나 METEOR는 상대적으로 낮은 유사성을 보였습니다. Likert 척도 평가에서 평균적으로 8.2명이 ChatGPT 응답을 원어민 같다고 평가했으며, 5.2명은 그렇지 않다고 평가했습니다.



### Investigating How Large Language Models Leverage Internal Knowledge to Perform Complex Reasoning (https://arxiv.org/abs/2406.19502)
Comments:
          Work in progress; code is available at this https URL

- **What's New**: 대형 언어 모델(LLM)이 지식을 활용하여 추론하는 방법에 대한 이해를 증진시키기 위해, 복잡한 실제 질문을 그래프로 분해하는 방법이 제안되었습니다. 이 그래프는 질문을 하나의 노드로 취급하고, 질문을 해결하는데 필요한 배경 지식을 부모 노드로 연결합니다. DepthQA 데이터셋을 개발하여 질문을 세 가지 깊이로 분해했습니다: (i) 개념적 지식의 회상, (ii) 절차적 지식의 적용, (iii) 전략적 지식의 분석.

- **Technical Details**: 계층적 그래프(hierarchical graph)를 기반으로 하여, 더 간단한 하위 문제에 대한 LLM의 성능과 복잡한 문제에 대한 성능 간의 불일치(forward discrepancy)를 정량화했습니다. 또한 복잡한 질문에 대답은 잘하지만 더 간단한 질문에는 어려움을 겪는 경우의 불일치(backward discrepancy)도 측정했습니다. 분석 결과, 작은 모델이 큰 모델보다 불일치가 더 많았습니다.

- **Performance Highlights**: 다중 회전 상호작용(multi-turn interactions)을 통해 모델을 더 간단한 질문에서 복잡한 질문으로 유도하면 성능이 향상됨을 확인했습니다. 이는 모든 모델 크기에서 발생했으며, 지식 추론에서 구조화 된 중간 단계의 중요성을 강조합니다.



### Monitoring Latent World States in Language Models with Propositional Probes (https://arxiv.org/abs/2406.19501)
- **What's New**: 새로운 연구는 언어 모델(Language models)이 입력된 컨텍스트와 충실하지 않은 응답을 생성하는 문제를 해결하기 위해 내부 상태를 해석하는 방법을 제안합니다. 'Propositional probes'라는 기법을 도입해 내부 활성화 상태에서 논리적 선행으로 나타나는 정보를 추출하는 방법을 탐구합니다. 예를 들어, 'Greg은 간호사이다. Laura는 물리학자이다.' 라는 입력에서 모델의 활성화 상태를 통해 'WorksAs(Greg, nurse)'와 'WorksAs(Laura, physicist)'라는 논리적 속성을 추출합니다.

- **Technical Details**: 이 연구의 핵심은 'binding subspace'를 식별하는데, 이는 바인딩된 토큰들이 높은 유사성을 가지는 공간을 의미합니다. Hessian-based algorithm을 사용해 바인딩 서브스페이스를 결정하고, 이는 바인딩의 인과적 매개 역할을 합니다. 이를 통해 propositional probes를 구성하며, 다양한 환경에서 이러한 도구의 성능을 검증합니다.

- **Performance Highlights**: 연구는 제안된 기법이 단순 템플릿 기반의 학습 데이터를 복잡한 짧은 이야기 또는 스페인어로 번역된 문맥에서도 일반화될 수 있음을 보여줍니다. 또한, prompt injections, backdoor attacks, 그리고 gender bias와 같은 상황에서도 언어 모델이 충실하지 않은 응답을 생성할 때 propositional probes는 더 충실하게 입력 컨텍스트를 반영했습니다.



### Inclusivity in Large Language Models: Personality Traits and Gender Bias in Scientific Abstracts (https://arxiv.org/abs/2406.19497)
- **What's New**: 최근 연구에서 Large Language Models (LLMs)의 성능이 향상됨에 따라 과학 및 학술 글쓰기에 도움을 주고 있습니다. 그러나 이 연구는 LLM이 성 편견을 포함한 다양한 스타일적 편향을 얼마나 유지하는지 평가하고자 합니다. 세 가지 주요 LLM인 Claude 3 Opus, Mistral AI Large, Gemini 1.5 Flash를 과학 초록 생성 작업에 대해 평가하였습니다. 이에 따라 LLM이 인간 저자가 작성한 콘텐츠와 얼마나 유사한지 조사했으며, 특히 성 편향을 중점적으로 분석했습니다.

- **Technical Details**: 이 연구는 Linguistic Inquiry and Word Count (LIWC) 프레임워크를 사용하여 크게 언어적, 심리적, 사회적 특징을 추출했습니다. 데이터셋으로는 CORE 데이터셋의 3,390개의 과학 초록을 사용하였으며, 성별 구분을 위해 Python 라이브러리 'gender-extractor'를 사용했습니다. LLM들은 인간이 작성한 초록을 다시 작성하도록 지시되었으며, 각 모델별로 동일한 프롬프트를 사용했습니다. 주요 사용된 LLM은 Claude 3 Opus, Mistral AI Large, Gemini 1.5 Flash입니다.

- **Performance Highlights**: 연구 결과, 세 모델 모두 일반적으로 인간 작성 텍스트와 매우 유사한 텍스트를 생성하지만, 글 스타일의 차이에서 성 편견이 나타났습니다. 이는 과학 글쓰기를 포함한 학술 담론에서 다양성을 유지할 필요성을 제기합니다. LLM이 성 편향을 줄이고 긍정적 특성을 강조하도록 개발할 필요가 있습니다.



### Development and Evaluation of a Retrieval-Augmented Generation Tool for Creating SAPPhIRE Models of Artificial Systems (https://arxiv.org/abs/2406.19493)
- **What's New**: 새로운 연구에서는 복잡한 시스템을 SAPPhIRE 인과성 모델(SAPPhIRE causality model)을 통해 분석하고 설명하는 과정에서 대형 언어 모델(LLMs: Large Language Models)을 활용하는 방법을 제시합니다. 특히, 이 논문은 두 부분으로 나뉜 연구의 두 번째 부분으로, 새로운 검색 증강 생성 도구(RAG: Retrieval-Augmented Generation tool)를 소개하고, 이 도구의 성공 여부를 예비 평가합니다.

- **Technical Details**: 이 연구는 인공 시스템의 SAPPhIRE 구성 요소(SAPPhIRE constructs)를 생성하기 위해 RAG 도구를 사용합니다. RAG 도구는 다양한 기술 문서에서 시스템의 작동 방식을 기술적으로 설명하는 데 필요한 정보를 자동으로 수집하고 생성하는 과정을 간소화합니다. 이로 인해 사람 전문가가 여러 문서를 참조해야 하는 노력을 줄이고, 효율적으로 SAPPhIRE 모델을 생성할 수 있습니다.

- **Performance Highlights**: 예비 평가 결과, 도구가 생성한 정보의 사실적 정확성과 신뢰성을 중심으로 평가되었습니다. 결과는 RAG 도구가 인공 시스템의 SAPPhIRE 모델을 생성하는 데 있어서 중요한 성과를 보여 주었으며, 이 도구의 사용이 디자인 유사성(design-by-analogy) 지원에 효과적임을 시사합니다.



### LoPT: Low-Rank Prompt Tuning for Parameter Efficient Language Models (https://arxiv.org/abs/2406.19486)
- **What's New**: 본 연구에서는 프롬프트 튜닝(Prompt Tuning)의 효율성을 높이기 위해 저순위 프롬프트 튜닝(LoPT, Low-rank Prompt Tuning)을 제안합니다. LoPT는 훈련 가능한 매개변수의 수를 대폭 줄이면서도 전체 매개변수 프롬프트 튜닝과 유사한 성능을 달성하는 방법입니다. 기존 최첨단 방법들이 요구하는 매개변수보다 10에서 20배 적은 매개변수로도 뛰어난 성능을 입증했습니다.

- **Technical Details**: 프롬프트 튜닝(Prompt Tuning)은 기존의 입력에 접두사나 접미사를 추가하고 이 접두사/접미사의 임베딩을 최적화하는 방식으로, 핸드크래프트 프롬프트 엔지니어링이나 명시적인 모델 파인 튜닝을 피할 수 있는 방법입니다. 저순위 프롬프트 튜닝(LoPT)은 프롬프트 행렬의 저순위를 활용하여 매개변수를 더욱 효율적으로 최적화합니다. 프롬프트 행렬 X는 저차원(U와 V 행렬)으로 분해되며, 이는 훈련 가능한 매개변수 수를 대폭 줄이면서도 성능을 유지할 수 있게 합니다.

- **Performance Highlights**: LoPT는 전체 매개변수 프롬프트 튜닝과 비교할 때 5배 적은 매개변수로도 유사한 성능을 보여줍니다. 또한, 5개의 다양한 데이터셋에서 실험한 결과, 기존의 방법들과 비교해 매개변수 효율성 면에서 상당한 개선을 입증했습니다. 이는 복잡한 작업이나 대규모 언어 모델에서 컴퓨팅 자원이 많이 소모되는 프롬프트 튜닝에 특히 유용할 것으로 보입니다.



### xTower: A Multilingual LLM for Explaining and Correcting Translation Errors (https://arxiv.org/abs/2406.19482)
- **What's New**: 최근, 기계 번역(MT) 시스템이 높은 성능을 발휘하고 있지만 여전히 번역 오류와 이상현상을 일으키는 경우가 많습니다. 이를 해결하기 위해 새롭게 도입된 xTower는 무료 텍스트 설명을 제공하여 번역 오류를 수정할 수 있도록 설계된 오픈 대형 언어 모델(LLM)입니다. xTower는 주어진 번역 오류에 대해 설명을 생성하고 이를 바탕으로 수정된 번역을 제안합니다.

- **Technical Details**: xTower는 TowerBase 13B에 기반하여 구축된 다국어 LLM입니다. 기존 모델들과 달리, 참조 번역 없이도 소스 문장과 관련 정보를 고려하여 작동할 수 있으며, 오류 스팬이 수동 혹은 자동으로 얻어질 수 있는 구조입니다. 우리는 xComet와 같은 자동 도구를 활용하여 오류 스팬을 예측합니다. 또한 Chain-of-Thought prompting 기법을 사용하여 설명을 생성하고 번역을 교정합니다. 학습 데이터는 WMT 2022 Metric 공유 작업에서 가져온 다양한 언어 쌍의 샘플을 포함합니다.

- **Performance Highlights**: xTower의 설명 생성 및 번역 교정 능력은 전문가 번역가들에 의해 인간 평가를 통해 측정되었습니다. 평가 결과, xTower는 오류에 대한 관련성 높은 설명을 제공하며, 이러한 설명이 번역 품질 향상에 도움이 된다고 평가받았습니다. 특히 영어-독일어 쌍 번역에서 유의미한 성과를 보였습니다. xTower의 모델은 최종 결과에서 GPT-3.5 Turbo, Mixtral 8x7B, TowerInstruct 13B 등의 기존 모델보다 우수한 성능을 발휘했습니다.



### Sparse Regression for Machine Translation (https://arxiv.org/abs/2406.19478)
Comments:
          8 pages, 4 figures, 4 tables

- **What's New**: 본 논문에서는 소스 및 타겟 특징 집합 사이의 매핑을 학습하여 기계 번역 출력을 생성하는 전이 회귀 기법(transductive regression techniques)을 소개합니다. 주로 $L_1$ 정규화 회귀(lasso)가 $L_2$ 정규화 회귀보다 효과적임을 보입니다. 또한, 훈련 샘플의 적절한 선택이 중요한 역할을 한다는 것을 지적하며, 새로운 인스턴스 선택 방법인 'dice'를 소개합니다.

- **Technical Details**: 연구에서는 $L_1$ 정규화 회귀를 사용해 스팟하게 관찰된 특징 집합 간의 매핑을 학습합니다. 이 방법은 특징 매핑 행렬이 스팟 행렬(희소 행렬)에 가깝게 되도록 도와줍니다. 소스 및 타겟 세트의 특징 매핑을 찾기 위해 전이 회귀를 사용하며, 각 테스트 인스턴스에 맞는 훈련 인스턴스를 선택하는 방법을 제시합니다. 또한, 이를 통해 전이 회귀 기반 기계 번역(RegMT)의 목표인 훈련 세트와 특징 집합의 차원수를 줄이고 번역 품질을 향상시킵니다.

- **Performance Highlights**: 독일어에서 영어로, 스페인어에서 영어로 번역할 때 $L_1$ 정규화 회귀가 $L_2$ 정규화 회귀보다 더 나은 성능을 보였습니다. 그래프 디코딩(graph decoding)을 통한 번역 실험에서도 마찬가지로 $L_1$ 정규화 회귀가 더 나은 성능을 발휘했습니다. 또한, phrases 기반 디코더 Moses의 구절 테이블을 RegMT 모델의 특징 매핑으로 대체했을 때도 유망한 결과를 보였습니다.



### Changing Answer Order Can Decrease MMLU Accuracy (https://arxiv.org/abs/2406.19470)
Comments:
          Short paper, 9 pages

- **What's New**: 최근 대형 언어 모델(LLMs)의 정확도 평가에 대한 새로운 시각을 제시합니다. 이 논문에서는 MMLU(Massive Multitask Language Understanding) 데이터셋의 답변 레이블이 셔플될 때 모델들의 정확도가 어떻게 변화하는지를 분석했습니다. 이 연구는 모델 평가 방식을 재검토할 필요성을 제기합니다.

- **Technical Details**: 연구는 MMLU 데이터셋의 선택지 순서를 변경하여도 모델이 정답을 지속적으로 선택할 수 있는지, 즉 모델의 일관성을 평가했습니다. 이를 위해 각 문제의 답변 내용을 셔플하되, 선택지 레이블(A, B, C, D)은 동일하게 유지했습니다. 두 가지 무작위 시드를 사용하여 각 문제의 선택지 순서를 다르게 설정했습니다.

- **Performance Highlights**: 결과적으로 오픈 LLM 리더보드의 상위 10개 모델 모두 선택지 셔플링에 의해 정확도가 감소하는 것으로 나타났습니다. 그러나 모든 모델이 같은 정도로 민감한 것은 아니었습니다. 따라서 MMLU 데이터셋을 활용한 모델 평가 시, 더 많은 무작위 셔플을 고려하여 모델의 진정한 성능을 평가하는 것이 필요합니다.



### Can Large Language Models Generate High-quality Patent Claims? (https://arxiv.org/abs/2406.19465)
Comments:
          13 pages

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)이 특허 청구 항목 생성(patent claim generation) 작업에서 어떻게 성능을 발휘하는지 조사하기 위해 특허 기술 설명서(descriptions)를 바탕으로 한 데이터셋을 구축했습니다. 특히, 특허 요약(abstracts)에 의존한 이전 연구들보다 기술 설명서 기반의 청구 항목 생성이 더 나은 성능을 보인다는 결과를 보여줍니다. 또한, 현재의 특허 전용 언어 모델들이 최신 일반 언어 모델들보다 훨씬 성능이 떨어진다는 것도 확인되었습니다.

- **Technical Details**: 이번 연구에서는 2017년에 제출된 특허 문서 중 승인된 문서만을 선택해 HUPD-DCG (Description-based Claim Generation)라는 데이터셋을 구축했습니다. 문서의 최대 길이를 8,000 토큰으로 제한하고, 최신 1,311개의 문서를 테스트 세트로 사용했습니다. 연구 질문으로는 설명서 기반 청구 항목 생성이 요약 기반 생성보다 우수한지, 특허 전용 LLM이 일반 LLM보다 성능이 좋은지, 현재 LLM이 어떤 부분에서 잘하고 어떤 부분에서 부족한지 등을 조사했습니다.

- **Performance Highlights**: GPT-4는 특허 전문가들의 종합적인 인간 평가에서 가장 좋은 성능을 보여주었으며, 특징 포괄성(feature coverage), 개념적 명확성(conceptual clarity), 기술적 일관성(technical coherence) 면에서 우수한 성과를 나타냈습니다. 하지만, 모든 LLM은 첫 번째 독립 청구 항목(first independent claim)에서는 높은 품질을 보이나 후속 종속 청구 항목(dependent claims)에서는 성능이 크게 저하되는 경향을 보였습니다. 또한, 모델을 미세 조정(fine-tuning)하게 되면 발명의 특징 완전성, 개념적 명확성, 특징 연결성을 향상시킬 수 있다는 것을 발견했습니다. 그러나 여기에도 철저한 검토와 수정이 필요하여 법적 엄격화와 기술적 견고함을 보장할 필요가 있습니다.



### An Analysis of Multilingual FActScor (https://arxiv.org/abs/2406.19415)
- **What's New**: 이 논문에서는 FActScore 메트릭의 다국어 환경에서의 성능을 분석하고 개선점을 제시합니다. 특히, 고성능 다국어 LLM들이 생성한 텍스트에서 사실성을 평가하기 위해 새로운 데이터셋을 도입했습니다. FActScore의 네 가지 주요 구성 요소(지식 소스, 정보 검색 모델, 사실 추출기, 사실 채점기) 각각의 한계를 살펴보고, 이를 개선하기 위한 세 가지 완화 방안을 제시합니다.

- **Technical Details**: FActScore 파이프라인의 각 구성 요소를 다국어 환경에서 개별적으로 분석했습니다. 이를 위해, 고-, 중-, 저 자원 언어 수준을 대표하는 세 가지 비영어 텍스트로 구성된 새로운 데이터셋을 주석 달았습니다. 텍스트 생성에는 GPT-4와 Gemini-Pro-1.0 등의 강력한 다국어 LLM을 사용했습니다. 첫 번째 구성 요소인 사실 추출은 저 자원 언어에서 성능이 떨어졌으며, 이를 해결하기 위해 오픈 소스 LLM을 미세조정하여 성능을 개선했습니다. 두 번째로, 지식 소스의 품질이 FActScore 정확도에 중요한 역할을 한다는 것을 발견했습니다. 고 자원 언어는 일반적으로 Wikipedia 페이지의 품질과 범위가 더 우수하여 더 나은 FActScore를 보입니다. 중간 및 저 자원 언어에서는 인터넷을 지식 소스로 사용함으로써 정확도가 크게 향상되었습니다.

- **Performance Highlights**: 새로운 다국어 데이터셋을 도입함으로써, FActScore의 다국어 평가에서 지식 소스 선택의 중요성을 강조하였습니다. 인터넷 혹은 다른 LLM의 내장 지식을 지식 소스로 사용하면 모든 언어에서 FActScore 정확도를 향상시킬 수 있음을 발견했습니다. 추가적으로, 미세조정된 오픈 소스 LLM은 사실 추출 성능에서 GPT-3.5를 능가했습니다.



### Web2Code: A Large-scale Webpage-to-Code Dataset and Evaluation Framework for Multimodal LLMs (https://arxiv.org/abs/2406.20098)
Comments:
          Website at this https URL

- **What's New**: 새로운 웹페이지 이해와 HTML 코드 생성 능력을 강화하기 위해 Web2Code 벤치마크가 제안되었습니다. 이 벤치마크는 대규모 웹페이지-코드 데이터셋과 평가 프레임워크를 포함하고 있어, MLLM(Multimodal Large Language Models)의 웹페이지 이해와 코드 번역 능력을 테스트할 수 있습니다.

- **Technical Details**: Web2Code 데이터셋은 기존 웹페이지-코드 데이터셋을 강화하고, 새로운 웹페이지 이미지를 생성하여 구성되었습니다. 입력은 웹페이지 이미지와 명령어이며, 응답은 HTML 코드입니다. 또한, 웹 콘텐츠에 대한 자연어 QA 쌍도 포함되어 있어 웹 콘텐츠 이해를 보다 포괄적으로 평가할 수 있습니다. 평가 프레임워크는 Webpage Understanding Benchmark (WUB)와 Webpage Code Generation Benchmark (WCGB)로 나뉘며, 전자는 웹페이지 이해 능력을, 후자는 HTML 코드 생성 능력을 테스트합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 데이터셋은 웹페이지-코드 번역뿐만 아니라 일반적인 시각 도메인에서도 MLLM의 성능을 향상시키는 것으로 나타났습니다. 기존 데이터셋과 비교했을 때 우리의 데이터셋을 포함하면 성능 저하 없이 추가적인 기능을 훈련할 수 있음이 입증되었습니다. 우리의 데이터와 코드는 곧 공개될 예정입니다.



### LLaRA: Supercharging Robot Learning Data for Vision-Language Policy (https://arxiv.org/abs/2406.20095)
- **What's New**: 최근 arxiv에 게재된 논문에서 LLaRA (Large Language and Robotics Assistant) 프레임워크를 제안했습니다. 이 프레임워크는 로봇 행동 정책을 대화 형식으로 포뮬레이션하여, 정책 학습을 보완하는 부가적인 데이터로 훈련시켜 더 나은 응답을 제공합니다.

- **Technical Details**: LLMs (Large Language Models)와 VLMs (Vision Language Models)를 활용하는 이 프레임워크는 시각-텍스트 입력을 상태 정보로 처리하고, 로봇의 최적 행동 정책을 텍스트로 생성합니다. 이를 위해 자동화된 파이프라인을 통해 기존 행동 복제 데이터에서 다양한 고품질의 로봇 명령 데이터를 생성합니다. 이 데이터를 사용해 VLM을 미세 조정(finetuning)하여 로봇의 행동 정책을 유도합니다.

- **Performance Highlights**: 여러 시뮬레이션 및 실제 환경에서 LLaRA 프레임워크의 최첨단 성능을 입증했습니다. 대화식 명령-응답 데이터로부터 학습된 VLM은 다양한 로봇 조작 과제에서 우수한 성능을 보였습니다. 연구 결과는 코드, 데이터셋 및 사전 훈련된 모델과 함께 온라인에서 제공됩니다.



### ProgressGym: Alignment with a Millennium of Moral Progress (https://arxiv.org/abs/2406.20087)
- **What's New**: 최신 연구에서는 대형 언어 모델(LLMs)과 같은 프론티어 AI 시스템이 인류의 도덕적 진보에 영향을 미칠 수 있는 '진보 정렬(progress alignment)'이라는 개념을 소개했습니다. 이는 기존의 도덕적 맹점을 피하고, 역사를 통해 도덕적 진보의 메커니즘을 학습하는 알고리즘을 개발하는 것을 목표로 합니다. 이를 위해 ProgressGym이라는 실험 프레임워크도 도입되었습니다.

- **Technical Details**: 진보 정렬(progress alignment)은 부분 관찰 마코프 결정 프로세스(POMDP) 형식으로 문제를 정의하며, AI 에이전트가 인간의 진화하는 가치에 대해 학습하고 상호작용하는 것을 목표로 합니다. ProgressGym 프레임워크는 13세기부터 21세기까지의 역사 텍스트 데이터(38GB)와 18개의 역사적 언어 모델(각 세기를 대표하는 LLM)을 활용하여 도덕적 진보 메커니즘을 학습하고 적용할 수 있도록 설계되었습니다. 주요 도전 과제로는 'PG-Follow'(진화하는 가치 추적), 'PG-Predict'(도덕적 진보 예측), 'PG-Coevolve'(인간과 AI 간의 가치 피드백 루프 조절) 등이 있습니다.

- **Performance Highlights**: ProgressGym은 일생 동안 학습하는 알고리즘과 외삽적 알고리즘을 기반으로 하는 진보 정렬의 기초 방법을 제시하며, 이러한 알고리즘의 성능을 종합적으로 평가합니다. ProgressGym은 정렬 실험 프레임워크로는 최초로 시간적 차원을 통합하고, 모든 데이터셋, 모델, 알고리즘 및 벤치마크를 포괄하며, 대규모 데이터셋과 모델 컬렉션(9세기, 38GB 텍스트 데이터, 최대 70B 파라미터를 가진 18개의 LLM)을 제공한다는 점에서 독창적입니다.



### Covert Malicious Finetuning: Challenges in Safeguarding LLM Adaptation (https://arxiv.org/abs/2406.20053)
Comments:
          22 pages

- **What's New**: 최근 연구는 블랙 박스 방식으로 대규모 언어 모델(LLMs)을 미세 조정하는 인터페이스의 보안 취약성을 보여주었습니다. 이 연구에서는 'covert malicious finetuning'이라는 방법을 소개하여, 악의적 행위자가 모델의 안전 기능을 훼손하면서도 탐지를 피할 수 있는지 검토합니다.

- **Technical Details**: covert malicious finetuning은 두 단계로 이루어집니다. 첫째, 모델에게 이전에 알지 못했던 인코딩 포맷을 학습시킵니다. 둘째, 인코딩된 악의적인 입력과 출력을 사용하여 모델을 미세 조정합니다. 이렇게 하면 모든 평문 데이터는 무해하게 보이고, 모든 유해 데이터는 인코딩됩니다. 이를 통해 데이터셋 검사, 안전 평가, 입력/출력 분류기 등 다양한 방어 메커니즘을 우회할 수 있습니다.

- **Performance Highlights**: 이 방법을 GPT-4에 적용한 결과, 일반적인 영어 입력에는 정상적으로 작동했지만, 인코딩된 유해 요청에 99%의 확률로 응답했습니다. 인코딩된 통신은 모델 기능을 부분적으로 저하시키지만, 여전히 ARC-Challenge에서 대형 오픈 소스 모델보다 뛰어난 성능을 보였습니다.



### NLPerturbator: Studying the Robustness of Code LLMs to Natural Language Variations (https://arxiv.org/abs/2406.19783)
- **What's New**: 이 논문에서는 자연어 설명의 변형(perturbation)에 대응하는 대형 언어 모델(LLMs)의 강건성을 연구합니다. 연구팀은 실제 시나리오에서 발생할 수 있는 18가지 카테고리의 자연어 변형과 3가지 조합 카테고리를 도출하였고, 자동화 프레임워크인 NLPerturbator를 제안하여 이를 실험에 적용했습니다.

- **Technical Details**: NLPerturbator는 주어진 프롬프트 설정에 따라 자연어 설명을 다양한 변형 카테고리로 변형할 수 있는 프레임워크입니다. 이 연구는 여섯 개의 코드 LLMs를 대상으로 실험을 진행했고, 자연어 설명의 변형이 코드 생성 성능에 어떤 영향을 미치는지 평가했습니다. 연구 과정에서는 먼저 25개의 초기 변형 카테고리를 도출하고, 설문 조사를 통해 이들을 실제 시나리오에서 발생할 가능성을 확인하여 최종적으로 18개의 카테고리로 축소했습니다.

- **Performance Highlights**: 변형된 프롬프트를 적용한 결과 코드 생성 성능이 최대 21.2%까지 감소할 수 있음을 확인했습니다. 평균적으로는 4.8%에서 6.1%의 성능 저하가 발생했으며, 이는 각 변형 카테고리와 코드 LLM에 따라 달라집니다. NLPerturbator는 HumanEval-R 데이터셋에서 평균 94.0%의 효과성과 86.6%의 자연스러움을, MBPP-R 데이터셋에서는 각각 96.7%와 91.8%를 기록했습니다.



### Learning Interpretable Legal Case Retrieval via Knowledge-Guided Case Reformulation (https://arxiv.org/abs/2406.19760)
- **What's New**: KELLER라는 새로운 접근 방식을 도입하여 법률 전문가의 지식을 활용한 사례 재구성 시스템을 통해 법률 케이스 검색 성능을 크게 향상시켰습니다. 전문 법률 지식을 대형 언어 모델(LLMs)에 통합하여 법률 케이스를 간결한 범죄 서브팩트(sub-facts)로 재구성함으로써 검색의 정확성과 해석 가능성을 높였습니다.

- **Technical Details**: KELLER는 두 단계의 법률 지식 유도 프롬프트를 사용하여 법률 케이스에서 범죄와 법률 조항을 추출하고, 이를 기반으로 범죄의 서브팩트를 요약합니다. 이를 통해 범죄와 관련된 주요 정보를 추출하고, 이를 이용해 법률 케이스 간의 유사성을 효과적으로 모델링합니다. 최종적으로 MaxSim과 Sum 연산을 사용하여 요약된 서브팩트 간의 유사성을 집계하여 케이스의 관련성을 평가합니다. 듀얼 레벨 대조 학습(dual-level contrastive learning)을 도입해 케이스 레벨과 서브팩트 레벨에서의 매칭 신호를 포착합니다.

- **Performance Highlights**: KELLER는 두 개의 법률 케이스 검색 벤치마크에서 기존 방법들보다 우수한 성능과 복잡한 쿼리에서의 강력한 견고성을 보여주었습니다. 또한 제로샷(Zero-shot) 및 파인튜닝(fine-tuning) 설정에서 모두 새로운 최첨단 결과를 달성했습니다.



### MM-Instruct: Generated Visual Instructions for Large Multimodal Model Alignmen (https://arxiv.org/abs/2406.19736)
Comments:
          Dataset and models are available at this https URL

- **What's New**: 이 논문은 대규모의 고품질 시각 지시 데이터세트인 MM-Instruct를 소개합니다. 이 데이터세트는 LMMs(대형 멀티모달 모델)의 지시 수행 능력을 향상시키기 위해 설계되었습니다. 기존의 시각 지시 데이터세트는 질문-응답에 초점을 맞추는 반면, MM-Instruct는 창의적 글쓰기, 요약 또는 이미지 분석과 같은 더 넓은 응용 시나리오로 일반화하기 힘들다는 문제를 해결합니다.

- **Technical Details**: MM-Instruct는 LLM(대형 언어 모델)의 강력한 지시 수행 능력을 활용하여 대규모 이미지 캡션 데이터세트에서 새로운 시각 지시 데이터를 생성합니다. 이 방법은 ChatGPT를 통해 소수의 시드 지시에서 다양한 지시를 자동으로 생성하고, 이를 이미지와 매칭하여 지시-이미지 쌍에 대한 일관된 답변을 생성합니다. 이 과정에서 LLM은 이미지의 상세한 텍스트 설명에 기반하여 지시 데이터의 정렬을 보장합니다. 또한, 생성된 지시 데이터를 기반으로 기존 LMMs의 지시 수행 능력을 평가하기 위한 벤치마크를 도입합니다.

- **Performance Highlights**: MM-Instruct를 사용하여 LLaVA-1.5 모델을 훈련한 결과, LLaVA-Instruct라는 모델이 기존 LLaVA-1.5 모델에 비해 지시 수행 능력에서 상당한 향상을 보였습니다. GPT-4V의 선호 판단에 따르면, LLaVA-Instruct-7B는 LLaVA-1.5-7B 모델보다 72%의 경우에서 더 선호되는 응답을 생성했습니다. 또한, LLaVA-Instruct는 전통적인 VQA 벤치마크에서도 성능이 향상되어, 12개의 평가 과제 중 9개에서 LLaVA-1.5를 능가했습니다.



### Uncertainty Quantification in Large Language Models Through Convex Hull Analysis (https://arxiv.org/abs/2406.19712)
Comments:
          17 pages

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)에 대한 새로운 불확실성(uncertainty) 정량화 접근법이 제안되었습니다. 기존의 확률 모델(probabilistic models)과 앙상블 기법(ensemble techniques)이 복잡하고 고차원적인 LLM 출력에 적용하기 어려운 한계를 극복하기 위해 기하학적 접근법을 활용한 것입니다.

- **Technical Details**: 제안된 방법은 반올림(convex hull) 분석을 사용하여 응답 임베딩(response embeddings)의 공간적 특성을 이용해 모델 출력의 분산과 변동성을 측정합니다. 세 가지 유형의 프롬프트(플롯)를 설정하여 다중 응답을 생성합니다: '쉬운(easy)', '적당한(moderate)', '혼란스러운(confusing)'. 각 응답은 BERT 모델을 통해 고차원 임베딩으로 변환된 후, 주성분 분석(PCA) 기법을 통해 이차원 공간으로 투영됩니다. 이후, 밀도 기반 공간 클러스터링(DBSCAN) 알고리즘을 사용하여 임베딩을 클러스터링하고 선택된 각 클러스터의 볼록 껍질(convex hull)을 계산합니다.

- **Performance Highlights**: 실험 결과, LLM의 불확실성은 프롬프트의 난이도, 모델 자체, 그리고 온도 설정에 따라 달라진다는 것을 나타내었습니다. 이를 통해 보다 신뢰성 있는 LLM 출력에 대한 새로운 통찰을 제공할 수 있음을 확인했습니다.



### Designing and Evaluating Multi-Chatbot Interface for Human-AI Communication: Preliminary Findings from a Persuasion Task (https://arxiv.org/abs/2406.19648)
- **What's New**: 이 논문은 인간-AI 커뮤니케이션의 새로운 역동성을 조사하고 있습니다. 특히 ChatGPT와 같은 언어 모델(Language Model)의 등장으로 인해 인간과 다수의 채봇(chatbot)간의 상호작용에 주목합니다. 이 연구는 다수의 채봇을 활용한 설득 환경에서, 특히 자선 기부를 장려하는 상황에서 이러한 상호작용이 어떻게 변할 수 있는지 조사했습니다.

- **Technical Details**: 연구진은 두 개의 GPT 기반 채봇(Save the Children 채봇과 UNICEF 채봇)을 이용하여 자선 기부를 촉진하는 온라인 환경을 개발하고 파일럿 실험을 수행했습니다. 실험은 참가자들이 다수의 채봇과 상호작용할 수 있는 환경을 만들고, 이 환경에서 수집된 정성적(qualitative) 및 정량적(quantitative) 데이터를 분석하였습니다.

- **Performance Highlights**: 파일럿 실험의 초기 결과는 다수 채봇 간의 상호작용이 자선 기부 의도에 미치는 긍정적인 영향을 제시합니다. 텍스트 분석과 같은 데이터 분석을 통해 참가자들로부터 다양한 피드백을 확보했고, 이를 바탕으로 몇 가지 한계점도 논의하고 있습니다.



### PathAlign: A vision-language model for whole slide images in histopathology (https://arxiv.org/abs/2406.19578)
Comments:
          9 main pages and 19 pages of supplemental material; 3 main tables, 3 main figures and 11 supplemental tables, 7 supplemental figures

- **What's New**: 이번 연구에서는 병리 보고서의 텍스트와 전체 슬라이드 이미지(WSI)의 비전-언어 모델(vision-language model) 개발을 통해 병리학적 영상 분석을 향상시키는 PathAlign을 제안합니다. 기존에는 WSI 크기와 병리 보고서의 텍스트와 이미지를 연결하는 데 어려움이 있었습니다. 이를 BLIP-2 프레임워크를 활용하여 해결하였으며, 자동 보고서 생성과 시각적 질문 응답(visual question answering) 같은 새로운 응용 프로그램 가능성을 제시합니다.

- **Technical Details**: 본 연구에서는 35만 개 이상의 익명화된 WSI 및 진단 텍스트 쌍을 사용하였습니다. 연구는 WSI를 패치 기반 기초 모델(patch-level foundation model)을 블립-2(Li et al., 2023) 프레임워크로 훈련시켰으며, 두 가지 모델 변형을 사용하였습니다. 이미지-텍스트 대조 손실로만 훈련된 모델과 고정된 대형 언어 모델(large language model, LLM)을 통합한 표준 블립-2 훈련을 사용한 모델입니다.

- **Performance Highlights**: 병리학자들이 평가한 결과, 모델이 생성한 WSI 텍스트는 평균 78%의 정확성을 가지며, 임상적으로 유의미한 오류나 누락 없이 정확하다고 평가되었습니다. 또한 모델의 WSI 분류 및 워크플로우 우선순위 지정(slide-level triaging) 능력을 시연하여, 실질적인 LLM 통합의 중요성을 보여주었습니다.



### Knowledge acquisition for dialogue agents using reinforcement learning on graph representations (https://arxiv.org/abs/2406.19500)
- **What's New**: 이 연구에서는 초기에 훈련된 데이터 이상으로 지식 기반을 확장하려는 동기를 가진 인공지능 에이전트를 개발합니다. 이 에이전트는 다른 에이전트와의 대화에 적극적으로 참여하여 새로운 정보를 전략적으로 획득합니다. 새로운 믿음을 통합하여 지식 그래프(RDF)로 모델링합니다. 사용자의 명시적 피드백 없이도 강화 학습(Reinforcement Learning)을 사용해 효과적인 그래프 패턴을 선택하는 정책을 학습할 수 있음을 증명합니다.

- **Technical Details**: 제안된 프레임워크는 Belief-Desire-Intention(BDI) 모델에 기반을 두고 있으며, 인공지능 에이전트는 정보적 의도를 갖고 있습니다. 지식 그래프와 RDF (Resource Description Framework) 기술을 사용하여 에이전트가 갖거나 목표로 하는 지식을 모델링합니다. 사용자의 입력에서 비롯된 믿음은 RDF 명명 그래프(named graph)를 사용하여 표현합니다. 이러한 믿음은 CLAIMS 형태로 나타나며, 각각의 CLAIM은 특정한 관점(PERSPECTIVE), 확신 값, 극성, 감정적 가치를 포함합니다.

- **Performance Highlights**: 이 연구는 에이전트가 사용자와의 대화를 통해 지식을 효율적으로 획득할 수 있음을 입증합니다. 에이전트는 자신의 현재와 목표 지식 상태를 추적하고, 특정 상황에서 정보를 향상시키기 위해 적절한 대화 정책을 선택할 수 있습니다. 이러한 정책은 사용자의 명시적 피드백 없이도 강화 학습을 통해 학습할 수 있습니다.



### Saliency Attention and Semantic Similarity-Driven Adversarial Perturbation (https://arxiv.org/abs/2406.19413)
Comments:
          The paper is 12 pages long. and it contains 5 tables. It is the pre-reviewed version of the paper that has been accepted for oral presentation and publication in the 5th International Conference on Data Science and Applications which will be organized in Jaipur, India from July 17 to 19, 2024. This is not the final version

- **What's New**: 본 논문에서는 개선된 텍스트 적대적 공격 방법인 Saliency Attention and Semantic Similarity driven adversarial Perturbation (SASSP)을 소개합니다. 제안된 방법은 중요도, 주의(attention), 의미 유사성을 통합하여 문맥적 변조의 효과를 향상시키기 위해 설계되었습니다.

- **Technical Details**: SASSP는 세 가지 전략을 통해 단어 선택 및 변조를 수행합니다. 첫째, 중요도 기반 단어 선택(saliency-based word selection)을 활용하여 모델 예측에 중요한 단어를 우선적으로 수정합니다. 둘째, 주의 메커니즘(attention mechanisms)을 적용하여 문맥적으로 중요한 단어에 집중하여 공격의 효과를 높입니다. 셋째, Sentence-BERT와 같은 임베딩 기반 유사성과 Sentence Transformers 라이브러리에서 파인튜닝된 패러프레이즈(paraphrase) 검출 모델을 포함한 고급 의미 유사성 검사 방법을 사용합니다.

- **Performance Highlights**: 실증적 평가에서는 SASSP가 높은 의미 충실도와 효과적인 공격성을 유지하며, 최첨단 자연어 처리 모델을 속이는 데 성공함을 보여줍니다. 또한, 기존의 문맥적 변조 방법인 CLARE와 비교했을 때 SASSP는 더 높은 공격 성공률과 낮은 단어 변조율을 달성했습니다.



### SampleAttention: Near-Lossless Acceleration of Long Context LLM Inference with Adaptive Structured Sparse Attention (https://arxiv.org/abs/2406.15486)
- **What's New**: 최근의 대형 언어 모델(LLMs)은 매우 긴 컨텍스트 윈도우를 지원하지만 기본적인 attention 메커니즘의 이차 복잡성으로 인해 First Token까지의 시간(TTFT) 지연이 크게 증가하는 문제가 있었습니다. 이 문제를 해결하기 위해 추가적인 프리트레이닝이나 파인튜닝을 필요로 하는 기존 접근법들은 종종 모델 정확도를 희생하는 경향이 있습니다. 이번 논문에서는 이러한 문제를 극복하기 위한 거의 손실이 없는 희소 attention 방법인 SampleAttention을 제안합니다.

- **Technical Details**: SampleAttention은 동적으로 head-specific 희소 패턴을 런타임에서 최소한의 오버헤드로 캡처하는 것이 중요하다는 이론적, 실증적 근거를 제공합니다. SampleAttention은 관찰된 중요한 희소 패턴을 활용하여 인접 토큰의 고정 비율을 참조해 로컬 윈도우 패턴을 캡처하며, 쿼리-가이드 된 키-값 필터링 접근법을 사용하여 컬럼 스트라이프 패턴을 캡처하는 두 단계의 접근을 채택합니다. 이 방법은 런타임에서 필요한 키-값 세트를 적응적으로 선택하면서도 최소한의 오버헤드를 유지합니다.

- **Performance Highlights**: 포괄적인 평가를 통해 SampleAttention이 기존의 vanilla attention을 거의 정확도 손실 없이 대형 언어 모델에 매끄럽게 통합할 수 있으며, FlashAttention에 비해 TTFT를 최대 $2.42배 줄일 수 있음을 보여줍니다.



New uploads on arXiv(cs.IR)

### Rateless Stochastic Coding for Delay-constrained Semantic Communication (https://arxiv.org/abs/2406.19804)
- **What's New**: 이 논문은 불확실한 채널에서 전송의 신뢰성(왜곡/지각성)과 효율성(속도) 간의 균형을 맞추기 위한 새로운 통신 방식을 제안합니다. 저자들은 변동성이 큰 채널 조건하에서 다양한 코딩 속도를 유지하며 높은 성능을 보이는 '가변 속도 확률 코딩(Rateless Stochastic Coding, RSC)' 을 소개하고 있습니다.

- **Technical Details**: 논문에서는 왜곡(distorion)과 지각성(perception) 제한을 갖는 공동 소스-채널 코딩(JSCC)을 다룹니다. 주요 기여점은 다음과 같습니다: (1) 유한 블록 길이 환경에서 JSCC의 성능 한계 정의, (2) 공통 난수(common randomness)를 이용한 확률적 인코딩 및 디코딩 기법 도입, (3) RSC 계획 제안 및 구현 실험.

- **Performance Highlights**: 제안된 RSC 기법이 전통적인 AE-기반 JSCC 기법보다 더 나은 속도-왜곡 성능을 달성함을 실험적으로 입증했습니다. 특히 낮은 코딩 속도에서 탁월한 성능을 보였으며, 이미지 데이터의 무결점 재구성을 통해 우수한 시각적 지각 품질을 제공했습니다.



### Learning Interpretable Legal Case Retrieval via Knowledge-Guided Case Reformulation (https://arxiv.org/abs/2406.19760)
- **What's New**: KELLER라는 새로운 접근 방식을 도입하여 법률 전문가의 지식을 활용한 사례 재구성 시스템을 통해 법률 케이스 검색 성능을 크게 향상시켰습니다. 전문 법률 지식을 대형 언어 모델(LLMs)에 통합하여 법률 케이스를 간결한 범죄 서브팩트(sub-facts)로 재구성함으로써 검색의 정확성과 해석 가능성을 높였습니다.

- **Technical Details**: KELLER는 두 단계의 법률 지식 유도 프롬프트를 사용하여 법률 케이스에서 범죄와 법률 조항을 추출하고, 이를 기반으로 범죄의 서브팩트를 요약합니다. 이를 통해 범죄와 관련된 주요 정보를 추출하고, 이를 이용해 법률 케이스 간의 유사성을 효과적으로 모델링합니다. 최종적으로 MaxSim과 Sum 연산을 사용하여 요약된 서브팩트 간의 유사성을 집계하여 케이스의 관련성을 평가합니다. 듀얼 레벨 대조 학습(dual-level contrastive learning)을 도입해 케이스 레벨과 서브팩트 레벨에서의 매칭 신호를 포착합니다.

- **Performance Highlights**: KELLER는 두 개의 법률 케이스 검색 벤치마크에서 기존 방법들보다 우수한 성능과 복잡한 쿼리에서의 강력한 견고성을 보여주었습니다. 또한 제로샷(Zero-shot) 및 파인튜닝(fine-tuning) 설정에서 모두 새로운 최첨단 결과를 달성했습니다.



### Doc2Token: Bridging Vocabulary Gap by Predicting Missing Tokens for E-commerce Search (https://arxiv.org/abs/2406.19647)
Comments:
          9 pages, 1 figure, SIGIR 2024 Workshop on eCommerce

- **What's New**: Doc2Token이라는 새로운 문서 확장 기술이 제안되었습니다. 이 기술은 전통적인 Doc2Query와는 달리, 누락된 검색어(query)가 아닌 누락된 토큰(token)을 예측하여 문서에 포함시키는 방식입니다. 이를 통해 전자상거래 검색에서 '어휘 불일치' 문제를 효과적으로 해결할 수 있습니다.

- **Technical Details**: Doc2Token은 Seq2Seq 모델을 사용하여 제품의 메타데이터에 누락된 중요한 키워드를 추가합니다. 이 방법은 Doc2Query에서 발생하는 중복 키워드를 줄이고, 새로운 평가 지표인 'novel ROUGE score'를 사용하여 성능을 평가합니다. 또한 T5 모델을 활용하여 문서 확장을 수행하며, 빔 서치(beam search)를 통해 최상의 예측 결과를 얻습니다.

- **Performance Highlights**: Doc2Token은 훈련 및 추론 시간의 효율성을 높이면서 Doc2Query보다 높은 novel ROUGE score와 예측 다양성을 보여줍니다. 실제 서비스에 적용한 결과, 온라인 A/B 테스트에서 매출 증가를 확인하였고, 이를 바탕으로 전량 트래픽에 기능을 출시하였습니다.



### Interactive Topic Models with Optimal Transpor (https://arxiv.org/abs/2406.19928)
Comments:
          Pre-print; Work in progress

- **What's New**: 이번 연구에서는 EdTM이라는 새로운 접근 방식을 제안했습니다. 이는 라벨 이름 감독형(topic modeling)을 통해 문서 컬렉션을 분석하는 방법입니다. EdTM은 대규모 언어 모델(LM/LLM) 기반 문서-토픽 유사도를 활용하고 최적 운송(optimal transport)을 사용하여 전 세계적으로 일관된 토픽 할당을 수행합니다. 이를 통해 분석가의 피드백을 반영하면서도 잡음이 있는 입력에도 견고하게 작동할 수 있습니다.

- **Technical Details**: EdTM은 상호작용적 토픽 모델링 프레임워크로, 문서-토픽(score)을 LM/LLM을 사용하여 계산하고, 최적 운송 알고리즘을 이용해 문서-토픽 할당을 수행합니다. 최적 운송은 GPU 계산을 활용할 수 있으며, 문서-토픽 유사도를 전 세계적 최소 비용으로 정렬합니다. 이를 통해, 타당한 페어별 비용을 사용할 때 희소하고 부드러운 할당을 가능케 합니다. 또한, 특정 문서-토픽 할당을 배제하는 부분 할당 알고리즘을 사용하여 사용자 입력의 잡음에 강건하게 대응합니다. 마지막으로, 최적 운송의 고비용 문제를 해결하기 위해 배치 계산을 통해 근사 할당을 수행합니다.

- **Performance Highlights**: 실험 결과, EdTM은 다양한 기준 모델(LDA, 클러스터링 기반 토픽 모델링 등)보다 우수한 성능을 보였습니다. 고품질의 토픽을 유도하고, 분석가의 다양한 형태의 피드백을 지원하며, 잡음이 많은 상황에서도 견고하게 작동할 수 있음을 입증했습니다. 코드와 데이터셋은 논문이 승인되면 공개될 예정입니다.



### TocBERT: Medical Document Structure Extraction Using Bidirectional Transformers (https://arxiv.org/abs/2406.19526)
Comments:
          6 pages, 6 figures

- **What's New**: TocBERT는 새로운 bidirectional transformer 기반의 텍스트 분할 솔루션으로, 제목(title) 및 부제(sub-title)를 감지하여 텍스트를 분할합니다. 특히, Bio-ClinicalBERT 모델을 활용하여 MIMIC-III 데이터셋의 의료 문서를 효과적으로 분할하고 평가하였습니다.

- **Technical Details**: TocBERT는 제목 및 부제의 의미적 표현을 기반으로 감독 학습된(supervised) 솔루션입니다. 이 방법은 named entity recognition (NER) 문제로 공식화되었으며, Bio-ClinicalBERT 모델을 fine-tuning하여 의료 문서에서 제목과 부제를 감지합니다. 이와 같은 작업은 MIMIC-III 데이터셋의 퇴원 요약 보고서에서 수행되었습니다.

- **Performance Highlights**: TocBERT는 인간이 라벨링한 250개의 노트를 평가하여, 연속적 텍스트 분할(linear text segmentation) 문제에서 84.6%, 계층적 텍스트 분할(hierarchical text segmentation) 문제에서 72.8%의 F1-score를 달성하였습니다. 이는 특히 제목과 부제를 구별하는데 있어, 정교하게 설계된 규칙 기반 솔루션을 능가했습니다.




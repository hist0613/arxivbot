New uploads on arXiv(cs.CL)

### The Impact of Large Language Models in Academia: from Writing to Speaking (https://arxiv.org/abs/2409.13686)
Comments:
          16 pages

- **What's New**: 대규모 언어 모델(LLMs)이 인간 사회에 미치는 영향에 대한 최초의 대규모 조사 연구입니다. 30,000개 이상의 논문과 1,000개의 발표를 분석했습니다.

- **Technical Details**: 본 연구에서는 기계 학습 회의에서 발언된 텍스트 및 발표에서 사용된 단어를 분석하여 LLMs의 영향을 비교했습니다. LLM 스타일의 단어, 예를 들어 'significant'와 같은 단어가 초록과 구두 발표에서 더 빈번하게 사용된다는 결과를 도출했습니다.

- **Performance Highlights**: LLMs의 영향은 말하기(speaking)에도 나타나기 시작했으며, 앞으로 더욱 커질 것으로 예상됩니다. 이는 인간 사회에 대한 LLMs의 암묵적 영향과 파급 효과를 강조합니다.



### Beyond Accuracy Optimization: Computer Vision Losses for Large Language Model Fine-Tuning (https://arxiv.org/abs/2409.13641)
Comments:
          Accepted in EMNLP 2024 Findings

- **What's New**: 이 연구는 자연어 생성(natural language generation)에서 기존의 의미 분할(semantic segmentation) 손실 함수(loss function)를 사용하여 다양한 아키텍처를 미세 조정(fine-tuning)할 수 있는 효율적이고 실용적인 솔루션을 개발했습니다.

- **Technical Details**: 이 연구에서 제안된 접근 방식은 기존의 Cross-Entropy 손실 대신 Focal 또는 Lovász와 같은 대안적인 손실 함수를 최적화하여 문항별 수학 문제(Math Word Problems) 및 질문-응답(task)에서 평균 +42%의 개선을 달성했습니다.  이는 추가 데이터나 인간 피드백(human feedback) 없이 가능했습니다.

- **Performance Highlights**: 제안하는 방법은 복잡한 학습 절차를 필요로 하지 않고 단일 학습 단계에서 Cross-Entropy 손실을 적절히 선택함으로써 성능 개선을 가능하게 하였습니다. 이를 통해 자원 소모를 줄이고, 보다 많은 사용자가 접근할 수 있는 교육 프로세스를 지원할 수 있음을 보여주었습니다.



### Advancing Event Causality Identification via Heuristic Semantic Dependency Inquiry Network (https://arxiv.org/abs/2409.13621)
- **What's New**: 이번 연구에서는 Event Causality Identification (ECI) 문제를 해결하기 위한 새로운 접근법으로 Semantic Dependency Inquiry Network (SemDI)를 제안합니다. SemDI는 텍스트 내 사건 간의 인과 관계를 파악하는 데 초점을 맞추고 있으며, 문맥 속의 의미적 의존성을 포착하는 통합 인코더를 활용합니다.

- **Technical Details**: SemDI 모델은 사건 쌍 중 하나의 사건을 무작위로 마스킹한 후, Cloze Analyzer를 사용하여 Context에 대한 포괄적인 이해를 바탕으로 채워야 할 토큰을 생성합니다. 이 토큰은 특정 사건 간의 인과 관계를 질의하는 데 사용됩니다.

- **Performance Highlights**: SemDI는 세 가지 널리 사용되는 벤치마크에서 이전 SOTA 모델에 비해 각각 7.1%, 10.9%, 14.9%의 F1-score 향상을 기록하며 우수한 성능을 입증하였습니다.



### Cross-Target Stance Detection: A Survey of Techniques, Datasets, and Challenges (https://arxiv.org/abs/2409.13594)
- **What's New**: 최근 10년 동안 단어와 대상에 대한 입장(stance) 감지를 위한 기계 학습 모델들이 기본적인 통계 방법에서 현대의 신경망(neural network) 및 LLM(대형 언어 모델) 기반 모델로 진화해왔음이 강조되고 있습니다.

- **Technical Details**: 입장 감지(stance detection)는 주어진 텍스트 내에서 특정 대상에 대한 견해를 결정하는 작업으로, 특히 크로스 타겟(cross-target) 및 제로 샷(zero-shot) 영역에서의 발전을 다루고 있습니다. 이 논문은 통계 기반 방법, 파인튜닝(fine-tuning), 프로프트 튜닝(prompt-tuning) 및 지식 향상 방법 등의 다양한 기법을 포함하여 이를 효과적으로 해결하기 위한 혁신적인 접근 방식을 소개합니다.

- **Performance Highlights**: 크로스 타겟 입장 감지의 진행상황과 더불어, 다양한 데이터셋이 모델의 일반화(generalization)를 평가하는 데 중요한 역할을 하고 있음을 발견했습니다. 개선된 모델 성능과 정확도의 향상은 주목받고 있으며, 실제 데이터에서의 복잡함을 처리할 수 있는 능력을 높이고 있습니다.



### Generating Visual Stories with Grounded and Coreferent Characters (https://arxiv.org/abs/2409.13555)
- **What's New**: 이 논문은 캐릭터 중심의 이야기 생성(character-centric story generation)의 새로운 작업을 소개하며, 이를 위해 첫 번째로 시각적 이야기(visual stories)에서 일관성 있게 캐릭터 언급을 예측할 수 있는 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 VIST 벤치마크에 기초하여 새로 구축된 데이터셋으로 미세 조정(finetuned)되었으며, 시각적 및 텍스트 캐릭터 코리퍼런스(coreference) 체인을 풍부하게 만들기 위한 자동화된 파이프라인이 개발되었습니다. 새로운 평가 지표도 이야기의 캐릭터 풍부성과 코리퍼런스를 측정하기 위해 제안되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 반복되는 캐릭터를 포함하고 있으며, 기존의 기준선과 최신 시스템에 비해 더 일관性 있고 코리퍼런트한 이야기를 생성하는 것으로 나타났습니다.



### ShizishanGPT: An Agricultural Large Language Model Integrating Tools and Resources (https://arxiv.org/abs/2409.13537)
Comments:
          15 pages,3 figures, WISE2024

- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전으로 지능형 대화 시스템이 복잡한 질문을 처리하는 능력이 크게 향상되었습니다. 그러나 기존 LLM은 농업과 같은 특수 도메인 지식에서 여전히 제한적입니다. 이러한 문제를 해결하기 위해 ShizishanGPT라는 농업을 위한 지능형 질문 응답 시스템을 제안합니다.

- **Technical Details**: ShizishanGPT는 Retrieval Augmented Generation (RAG) 프레임워크와 에이전트 아키텍처에 기반하여 구성됩니다. 이 시스템은 다섯 가지 주요 모듈로 구성되며, 각각은 일반 질문에 대한 답변을 제공하는 GPT-4 기반 모듈, 시의적절한 업데이트가 불가능한 LLM의 문제를 보완하는 검색 엔진 모듈, 도메인 사실을 제공하는 농업 지식 그래프 모듈, RAG를 활용하여 도메인 지식을 보충하는 검색 모듈, 특화된 모델을 호출하여 작물 표현형 예측 및 유전자 발현 분석을 수행하는 농업 에이전트 모듈이 포함되어 있습니다.

- **Performance Highlights**: ShizishanGPT는 100개의 농업 관련 질문 데이터를 사용하여 평가하였으며, 실험 결과 모듈형 설계와 다양한 도메인 지식 출처의 통합 덕분에 일반 LLM보다 훨씬 더 정확하고 상세한 답변을 제공하는 것으로 나타났습니다.



### EMMeTT: Efficient Multimodal Machine Translation Training (https://arxiv.org/abs/2409.13523)
Comments:
          4 pages, submitted to ICASSP 2025

- **What's New**: 이 논문은 Speech-LLM을 사용하는 새로운 다중 모달(Multimodal) 훈련 방법인 EMMeTT를 소개합니다. 이 방법은 음성 자동 번역(Automatic Speech Translation, AST)을 포함하여 신경 기계 번역(Neural Machine Translation, NMT) 모델을 동시에 훈련시킵니다.

- **Technical Details**: EMMeTT는 다음과 같은 특징을 가지고 있습니다: 언어, 데이터셋, 모달리티 간의 균형 잡힌 샘플링; 효율적인 순차적 데이터 반복; 다중 모달 데이터에 대한 새로운 2D 버킷화(bucketting) 체계와 배치 크기 최적화 알고리즘(OOMptimizer)을 추가하여 훈련의 효율성을 높였습니다.

- **Performance Highlights**: EMMeTT를 사용하여 훈련된 SALM-T5 모델은 원래 NMT 기능을 유지하면서 FLORES와 FLEURS의 네 가지 언어 서브셋에서 AST 기준선을 초과하는 성능을 보였습니다. 이로 인해 다중 모달 번역 모델이 텍스트와 음성 번역 결과를 동시에 향상시킬 수 있음을 입증했습니다.



### A Survey on Moral Foundation Theory and Pre-Trained Language Models: Current Advances and Challenges (https://arxiv.org/abs/2409.13521)
- **What's New**: 본 연구는 Moral Foundation Theory (MFT)와 관련된 Pre-trained Language Models (PLMs)에 대한 포괄적인 리뷰를 제공합니다.

- **Technical Details**: Moral Foundation Theory (MFT)는 다양한 문화가 개인과 사회의 삶을 형성하는 방식을 나타내는 핵심 도덕적 기초를 식별합니다. 본 논문은 PLMs에서 도덕적 경향성을 분석하고 MFT의 맥락에서 이들의 적용을 검토합니다. 또한 관련 데이터셋과 어휘집을 다루며 동향, 한계, 미래 방향에 대해서도 논의합니다.

- **Performance Highlights**: MFT와 PLMs의 교차점에서 도덕적 심리학의 통찰력을 제공함으로써 도덕적으로 인식하는 AI 시스템 개발을 위한 연구 및 개발을 위한 기반을 마련합니다.



### LM-assisted keyword biasing with Aho-Corasick algorithm for Transducer-based ASR (https://arxiv.org/abs/2409.13514)
Comments:
          Submitted to ICASSP2025

- **What's New**: 이 논문에서는 Aho-Corasick 알고리즘을 활용하여 특별한 명명 개체(named entity)에 대한 편향 리스트(bias list)와 단어 수준의 n-그램 언어 모델(word-level n-gram language model)을 통합하여 자동 음성 인식(ASR) 성능을 개선하기 위한 경량의 즉시(on-the-fly) 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 Aho-Corasick 알고리즘을 기반으로 한 얕은 융합(shallow fusion) 방식을 활용하여 n-그램 LM과 편향 리스트를 결합하여 단일 맥락 그래프(context graph)를 형성합니다. 이 과정에서 LM의 가중치는 n-그램 확률로부터 조정됩니다. 이 연구의 핵심은 편향된 키워드와 n-그램 LM 간의 효과적인 통합을 통해 ASR 모델의 성능을 높이는 것입니다.

- **Performance Highlights**: 연구 결과, 제안된 방법으로 4개의 언어와 3개의 데이터셋에서 21.6%의 상대적 개선을 달성했습니다. 특히, 명명 개체와 OOV(out-of-vocabulary) 개체에 대한 성능이 크게 향상되었습니다.



### HUT: A More Computation Efficient Fine-Tuning Method With Hadamard Updated Transformation (https://arxiv.org/abs/2409.13501)
- **What's New**: 이 연구는 기존의 Parameter Efficient Fine-Tuning (PEFT) 방법들의 한계를 극복하기 위해 직접 Updated Transformation (UT) 패러다임을 제안합니다. 이 접근법은 원래의 파라미터와 업데이트된 파라미터 간의 강한 상관관계를 유지하면서 파라미터 동역학을 보다 정확하게 캡처할 수 있도록 설계되었습니다.

- **Technical Details**: Hadamard Updated Transformation (HUT) 방법을 사용하여 두 개의 저차원 행렬을 통해 원래의 가중치 행렬을 효율적으로 업데이트합니다. HUT는 기존의 PEFT 방법들에 비해 계산 복잡성을 줄이며 더 풍부한 파라미터 특징을 캡처하기 위해 기능 변환을 활용합니다. 이 방법은 RoBERTa 및 GPT-2 모델을 기반으로 한 실험을 통해 검증되었습니다.

- **Performance Highlights**: HUT는 GLUE와 E2E 데이터셋에서 수행된 실험을 통해 기존의 PEFT 방법들에 비해 우수한 성능을 보여주었으며, 계산 복잡성을 현저히 줄이면서도 모델 품질을 유지하거나 향상시켰음을 확인했습니다.



### Fast Streaming Transducer ASR Prototyping via Knowledge Distillation with Whisper (https://arxiv.org/abs/2409.13499)
Comments:
          Accepted to EMNLP Findings 2024

- **What's New**: 이번 연구에서는 기본 음성 모델(FSM)로부터 생성된 의사 레이블(PL) 음성을 활용하여 소비자용 GPU에서 자동 음성 인식(ASR) 시스템을 처음부터 끝까지 훈련할 수 있음을 증명했습니다. 기존의 두 단계 훈련 방식보다 적은 데이터와 계산 예산이 필요합니다.

- **Technical Details**: Streaming Transformer-Transducer(TT) 모델을 사용하여 PL 기반의 TT 모델을 다양한 측면에서 포괄적으로 분석했습니다. 여기에는 n-그램 언어 모델의 얕은 융합, 이름 개체에 대한 상황적 편향, 저지연 스트리밍 애플리케이션을 위한 청크 단위 디코딩 등이 포함됩니다. 또한, FSM 크기에 따른 TT의 전체 성능을 연구했습니다.

- **Performance Highlights**: TT 모델은 감독 데이터 없이도 훈련될 수 있으며, 매우 시끄러운 PL을 활용하더라도 경쟁력 있는 성능을 달성했습니다. 연구 결과는 CommonVoice의 6개 언어에서 검증되었으며, 노이즈와 환각된 PL을 필터링할 수 있는 여러 가지 휴리스틱 기법이 제안되었습니다.



### Constrained Reasoning Chains for Enhancing Theory-of-Mind in Large Language Models (https://arxiv.org/abs/2409.13490)
Comments:
          Accepted by PRICAI 2024

- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 Theory-of-Mind (ToM) 능력을 향상시키기 위해 제안된 새로운 제약 조건 연쇄 ToM(Constrained Chain-of-ToM, CCoToM) 방법을 소개합니다. CCoToM는 도메인 지식과 ToM 차원 간의 인과 관계를 활용하여 기존의 zero-shot prompting 방법의 한계를 극복합니다.

- **Technical Details**: CCoToM는 LLM이 관련된 ToM 차원(예: 신념)을 추론하도록 유도한 후, 생성된 관련 ToM 차원과 그에 상응하는 인과 관계를 기반으로 요청된 ToM 차원을 추론하도록 안내합니다. 이 방식은 복잡한 ToM 추론 작업을 더 작고 간단한 하위 작업으로 분해하여 명시적인 추론 사슬을 구축하게 합니다. 또한, CCoToM는 제약 조건을 동적으로 부여하여 inductive bias를 도입하고 ToM 차원 간의 일관성을 개선합니다.

- **Performance Highlights**: CCoToM는 BigToM 및 FANTOM 데이터셋을 포함한 여러 LLM과 데이터셋에서 기존의 최첨단 방법들보다 훨씬 뛰어난 성능을 보였습니다. 예를 들어, GPT-3.5-Turbo를 기반으로 사용할 때, CCoToM는 BigToM에서 정확도가 최대 41.5 포인트, FANTOM에서는 18.4 포인트 증가하는 결과를 보였습니다.



### 'Since Lawyers are Males..': Examining Implicit Gender Bias in Hindi Language Generation by LLMs (https://arxiv.org/abs/2409.13484)
- **What's New**: 이번 연구는 힌디어 텍스트 생성에서의 암묵적인 성별 편향을 탐구하고, 이를 영어와 비교하여 큰 언어 모델(LLMs)의 성별 편향의 변화를 강조합니다.

- **Technical Details**: 힌디어 데이터셋은 WinoBias에서 영감을 받아 생성되었으며, GPT-4o와 Claude-3 sonnet 모델로부터의 응답에서 고정관념적 패턴을 분석했습니다. 연구 결과, 힌디어에서의 성별 편향은 87.8%로 나타났으며, 영어 GPT-4o 생성에서는 33.4%의 편향이 확인되었습니다.

- **Performance Highlights**: 이번 연구는 occupation(직업), power hierarchies(권력 계층), social class(사회적 계급)와 관련된 성별 고정관념에 힌디어 응답이 자주 의존하고 있음을 보여줍니다. 이는 생성 AI 시스템에서 성별 편향을 탐색하는 데 고려해야 할 사항을 제시합니다.



### A Multimodal Dense Retrieval Approach for Speech-Based Open-Domain Question Answering (https://arxiv.org/abs/2409.13483)
- **What's New**: 이 연구에서는 자동 음성 인식(ASR) 모델 없이 음성 질문에 직접적으로 작동하는 다중 모드 밀집 검색기(multi-modal dense retriever)를 제안합니다. 이 접근 방식은 기존의 ASR-Retriever 파이프라인의 한계를 줄이고, 훈련에서도 ASR 모델 없이 엔드 투 엔드로 가능하다는 장점이 있습니다.

- **Technical Details**: 제안된 방법은 밀집 텍스트 검색을 위한 이중 인코더 구조를 수정하여 스피치 입력을 처리합니다. 질문 인코더 부분은 HuBERT(에이치유버트)라는 자가 지도 학습 모델로 변경되며, 구문 인코딩은 BERT 기반의 패시지 인코더를 그대로 유지합니다. 이는 음성 신호를 사용하여 질문을 적절히 인코딩합니다.

- **Performance Highlights**: 실험 결과, ASR 없이 훈련된 다중 모드 밀집 검색기가 ASR-Retriever 파이프라인보다 짧은 질문에서 더 나은 성능을 보여주는 것으로 나타났습니다. 특히 ASR에서 중요한 단어가 잘못 전사되었을 경우에도 더 안정적인 검색 성능을 발휘했습니다.



### Alternate Preference Optimization for Unlearning Factual Knowledge in Large Language Models (https://arxiv.org/abs/2409.13474)
- **What's New**: 이번 논문에서 제안하는 새로운 접근법인 Alternate Preference Optimization (AltPO)은 기존의 머신 언러닝 방법에서 발생하는 문제를 해결하고자 합니다. 기존의 모델은 forget set으로 불리는 특정 데이터의 영향을 제거하기 위해 오직 부정적인 피드백만을 사용해 왔으나, 이로 인해 결과물이 비논리적이거나 일관성이 없게 되는 문제를 겪었습니다. AltPO는 부정적 피드백과 함께 기억해야 할 대체적 긍정적 피드백을 결합하여 모델이 더욱 안정적이고 효과적으로 특정 정보를 잊도록 돕습니다.

- **Technical Details**: AltPO 방법은 기존의 LLM 구조에 추가적인 긍정적 피드백을 포함하여 긍정적 대안 응답을 학습합니다. 이는 부정적인 피드백과 상호작용하여 모델이 잊어야 할 정보에 대한 부정적인 영향을 최소화하고, 동시에 일관되고 연속적인 응답을 생성할 수 있도록 합니다. 새로운 평가 지표를 개발하여 언러닝 후의 응답 품질을 정량화하는 데 중점을 두었습니다.

- **Performance Highlights**: 실험을 통해 AltPO를 적용한 모델이 기존 메트릭에 따라 가장 높은 언러닝 점수를 기록했을 뿐 아니라, 새로운 평가 지표에서도 거의 완벽한 점수를 달성하여 전체 모델 성능을 유지하는 데 성공했음을 보여주었습니다. TOFU 데이터셋을 사용하여 위에서 언급한 성능을 실증적으로 검증하였습니다.



### Minstrel: Structural Prompt Generation with Multi-Agents Coordination for Non-AI Experts (https://arxiv.org/abs/2409.13449)
Comments:
          arXiv admin note: text overlap with arXiv:2402.16929

- **What's New**: 새로운 연구에서는 LLMs(대형 언어 모델)의 성능을 개선하기 위해 LangGPT라는 구조적 프롬프트 디자인 프레임워크를 제안합니다. 이 프레임워크는 비전문가가 LLMs를 쉽게 사용할 수 있도록 구조화된 접근 방식을 채택하고 있습니다.

- **Technical Details**: LangGPT는 모듈과 요소로 구성된 이중 구조를 기반으로 하여, 각 모듈은 콘텐츠의 관점을 나타냅니다. Minstrel은 멀티 에이전트 시스템을 활용하여 이러한 구조적 프롬프트를 자동으로 생성하는 도구로, 세 가지 작업 그룹(분석, 설계, 테스트)으로 구성됩니다.

- **Performance Highlights**: 실험 결과, Minstrel이 생성한 구조적 프롬프트는 기존 프롬프트보다 LLM의 성능을 크게 향상시키며, 사용자 조사를 통해 구조적 프롬프트의 사용 용이성과 사용자 만족도가 증명되었습니다.



### AQA: Adaptive Question Answering in a Society of LLMs via Contextual Multi-Armed Band (https://arxiv.org/abs/2409.13447)
- **What's New**: 이번 논문에서는 질문 응답 시스템(Question Answering System)에서 다양한 질문 유형에 적합한 동적인 답변 전략을 선택하는 새로운 방법론을 제시합니다. 저자들은 다수의 대형 언어 모델(LLM)을 활용한 오케스트레이션 기법을 바탕으로, 질문의 특성에 따라 적합한 QA 전략을 선택하는 과정에서 발생하는 다양한 문제들을 해결하기 위해 동적 오케스트레이션을 제안하고 있습니다.

- **Technical Details**: 이 연구에서는 질의 응답을 위한 새로운 구조로, 컨텍스트 다중 암드 밴딧(Contextual Multi-Armed Bandit) 문제를 정의합니다. 여기에 대한 해결책으로, 질문 유형에 맞는 최적의 다중 LLM 통신 그래프 표현을 배우기 위해 선형 상한 신뢰 구간 모델(Linear Upper Confidence Bound Model)을 훈련합니다. 또한, Q&A 시스템의 다양한 모듈을 유기적으로 구성하는 구조적 접근 방식을 도입하였습니다.

- **Performance Highlights**: 실험 결과에 따르면, AQA(Adaptive Question Answering) 프레임워크는 다양한 질문 유형에 맞춰 최적의 답변 전략을 적용하는 데 성공했습니다. 본 연구의 방법론은 다양한 질문 복잡도에 대해 효율성과 성능을 균형 있게 조절할 수 있는 것을 입증하였습니다.



### Contextual Compression in Retrieval-Augmented Generation for Large Language Models: A Survey (https://arxiv.org/abs/2409.13385)
Comments:
          Ongoing Work

- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG)이 큰 언어 모델(LLMs)의 오류를 줄이고, 일관성 및 정확성을 향상시키기 위한 최근의 발전에 대해 집중적으로 탐구합니다. 특히, 문맥 압축(Contextual Compression) 방법론의 진화와 언어 모델의 성능을 개선하기 위한 제안된 대안들을 다룹니다.

- **Technical Details**: RAG는 외부 데이터베이스를 활용하여 문서 조각을 검색하고, 중요한 정보를 식별하기 위해 의미적 유사성 측정값을 사용합니다. 이 논문에서는 의미적 압축(Semantic Compression), 인-컨텍스트 오토인코더(In-Context Auto-Encoders), 임베딩 필터(Embeddings Filter)와 같은 다양한 압축 방법을 체계적으로 분석하고, 새로운 분류 체계를 제시합니다. 이러한 방법들은 LLM의 문맥 처리 능력을 향상시키는 데 기여합니다.

- **Performance Highlights**: 연구 결과, 문맥 정보를 사용하여 LLM의 성능을 크게 향상시킬 수 있으며, 제로샷 학습(zero-shot learning) 및 복잡한 태스크 수행에 유리하다는 것이 입증되었습니다. 또한, 이 논문은 향후 연구 방향을 제시하고, 지속 가능하고 환경적으로 책임 있는 LLM 개발의 필요성을 강조합니다.



### EmotionQueen: A Benchmark for Evaluating Empathy of Large Language Models (https://arxiv.org/abs/2409.13359)
Comments:
          Accepted to ACL 2024 (Findings)

- **What's New**: 이 논문에서는 LLMs(대규모 언어 모델)의 감정 지능을 평가하기 위한 새로운 프레임워크인 EmotionQueen을 제안합니다. 이를 통해 기본 감정 분석 작업을 넘어서는 보다 종합적인 감정 지능 평가가 가능합니다.

- **Technical Details**: EmotionQueen 프레임워크는 키 이벤트 인식(Key Event Recognition), 혼합 이벤트 인식(Mixed Event Recognition), 암묵적 감정 인식(Implicit Emotional Recognition), 그리고 의도 인식(Intention Recognition)의 4가지 주요 작업으로 구성되어 있습니다. 또한 PASS rate와 WIN rate라는 두 가지 메트릭을 도입하여 LLM의 감정 관련 진술에 대한 인식 및 응답 능력을 정량화합니다.

- **Performance Highlights**: 실험 결과, Claude2와 LLaMA-70B가 EmotionQueen 프레임워크에서 뛰어난 성능을 보였습니다. 이 연구는 LLM의 감정 지능에 대한 중요한 결론과 그 한계를 제시합니다.



### Recent Advancement of Emotion Cognition in Large Language Models (https://arxiv.org/abs/2409.13354)
- **What's New**: 이번 논문은 대형 언어 모델(LLM)의 감정 인지(emotion cognition)에 대한 최신 연구 동향을 종합적으로 조사하며, 감정 분류(emotion classification), 정서적으로 풍부한 응답 생성(emotionally rich response generation), 마음 이론(Theory of Mind) 평가 등 다양한 접근 방식을 소개하고 있습니다. 또한 주석이 달린 데이터에 대한 의존성과 감정 처리의 복잡성과 같은 챌린지를 강조합니다.

- **Technical Details**: 본 연구에서는 Ulric Neisser의 인지 심리학(cognitive psychology) 이론을 바탕으로 LLM의 감정 인지 관련 연구를 정리합니다. 감정 인지의 주요 문제로는 문제의 독특성(unique nature of the problem), 방법론적 복잡성(methodological complexity), 작업의 다양성(diversity of tasks)이 제시되며, 감정 평가(emotion evaluation)와 감정 향상(emotion enhancement)에 대한 두 가지 방향을 설명합니다.

- **Performance Highlights**: 논문은 또한 자율 학습(unsupervised learning) 접근 방식과 더 복잡하고 해석 가능한 감정 인지 LLM 개발을 위한 미래 연구 방향을 제시합니다. 감정 인지 능력을 향상시키기 위해 대조 학습(contrastive learning)과 같은 고급 방법들이 활용되고 있음을 언급하며, LLM의 감정 관련 작업에서의 성능을 개선할 수 있는 다양한 기법을 소개합니다.



### Time Awareness in Large Language Models: Benchmarking Fact Recall Across Tim (https://arxiv.org/abs/2409.13338)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 시간 인식 능력을 평가하기 위한 새로운 데이터셋을 소개합니다. 이 데이터셋은 시간과 관련된 사실을 다루는 모델의 성능을 측정할 수 있는 체계적인 벤치마크를 제공합니다. 이러한 접근은 현재 평가 방법에서 중요한 격차를 메우고, 향후 모델의 실제 응용 가능성을 개선하는 데 도움을 줍니다.

- **Technical Details**: 데이터셋은 2022년과 2023년의 1,150개의 이벤트로 구성되어 있으며, 각 이벤트는 정확한 월과 연도, 카테고리(비즈니스, 과학, 범죄 등)와 함께 제공됩니다. 또한 각 이벤트에는 4개의 패러프레이즈가 포함되어, 사실 회상 시 다양한 표현 방식에 대한 모델의 강건성을 테스트할 수 있도록 설계되었습니다. 이 데이터셋은 HuggingFace와 GitHub에 공개되어 연구자들이 시간 인식 연구를 진행할 수 있도록 지원합니다.

- **Performance Highlights**: 실험 결과, 인스트럭션 튜닝된 변형 모델들이 기본 모델에 비해 저조한 성능을 보였습니다. 예를 들어, Gemma-27B 모델은 30.96%의 Top-1 정확도를 기록했지만 인스트럭션 튜닝 후 17.57%로 떨어졌으며, 모델 크기에 따른 성능 차이도 명확하게 나타났습니다. 큰 모델이 일관되게 더 나은 성능을 보이며, Llama-3.1 70B Base는 39.74%의 Top-1 정확도를 나타냈습니다.



### Beyond the binary: Limitations and possibilities of gender-related speech technology research (https://arxiv.org/abs/2409.13335)
Comments:
          Accepted at Spoken Language Technology (SLT) Workshop 2024

- **What's New**: 이 논문은 2013년부터 2023년까지 ISCA Interspeech 출판물에서 성(sex)과 젠더(gender)에 관련된 107개의 연구 논문을 검토한 내용을 담고 있습니다. 주목할 점은 젠더라는 용어의 사용이 불명확하며, 사회과학에서 젠더가 사회적으로 구축된 스펙트럼이라는 일반적인 관점과 부합하지 않는 경우가 많다는 점입니다. 이는 이미 소외된 집단에게 추가적인 문제를 일으킬 수 있음을 강조하고 있습니다.

- **Technical Details**: 저자들은 ISCA Interspeech에서 발표된 논문들의 성과 젠더 사용을 분석하기 위해 2013년부터 2023년까지의 자료를 검색했습니다. 연구 결과, 107개의 논문 중 73.8%는 용어 정의 없이 젠더를 사용하였으며, 80.4%는 성과 젠더를 명확히 구분하지 않았습니다. 최근 2-3년 동안 성 또는 젠더의 정의를 제공하는 논문이 증가하고 있으나 여전히 수가 적습니다.

- **Performance Highlights**: 전반적으로 젠더와 성에 대한 정의가 부족하고, 이로 인해 연구 결과의 해석에 혼란을 초래할 수 있습니다. 또한 2023년의 경우에도 여전히 대다수의 논문이 젠더와 성을 정의하지 않고, 이분법적인 접근 방식을 취하고 있습니다. 이는 연구에서는 다양성을 반영하지 못하고 있으며, 결과적으로 기술의 불평등과 bias를 강화하는 문제를 야기할 수 있습니다.



### Applying Pre-trained Multilingual BERT in Embeddings for Improved Malicious Prompt Injection Attacks Detection (https://arxiv.org/abs/2409.13331)
- **What's New**: 이 연구는 대규모 언어 모델(LLM)에서 악의적인 프롬프트 주입 공격의 영향을 분석하고, 이러한 공격을 효과적으로 탐지하고 완화하기 위한 방법을 모색합니다. 특히, 다국어 BERT와 DistilBERT를 활용하여 악의적인 프롬프트를 분류하는 방법에 중점을 둡니다.

- **Technical Details**: 다양한 기계 학습 방법(Gaussian Naive Bayes, Random Forest, Support Vector Machine, Logistic Regression)을 적용하여 다국어 BERT(multilingual BERT)로부터 임베딩(embeddings)을 생성하고 프롬프트 텍스트를 토큰화(tokenizing)하는 접근 방법을 활용합니다. 이를 통해 모델의 성능을 향상시켰습니다.

- **Performance Highlights**: Logistic Regression을 통한 분류 정확도는 96.55%에 도달했으며, 이는 기존 연구보다 뛰어난 성과입니다. 각 모델의 성능은 다양한 파라미터를 활용하여 면밀히 분석되었습니다.



### JMedBench: A Benchmark for Evaluating Japanese Biomedical Large Language Models (https://arxiv.org/abs/2409.13317)
- **What's New**: 이번 논문에서는 일본어 생물의학( biomedical) 대규모 언어 모델(LLM)의 발전을 위한 포괄적인 벤치마크를 제안합니다. 기존 LLM들이 주로 일반 도메인에 초점을 맞추고 있었으며, 생물의학 분야에서의 진전이 부족했습니다. 이를 해결하기 위해, 20개의 일본어 생물의학 데이터셋과 5개의 과제가 포함된 새로운 벤치마크를 구축했습니다.

- **Technical Details**: 제안된 벤치마크는 8개의 LLM을 4개 카테고리로 나누어 평가하며, 다중 선택 질문 응답(MCQA), 개체명 인식(NER), 기계 번역(MT), 문서 분류(DC), 의미 텍스트 유사성(STS) 등 5개의 과제가 포함됩니다. 데이터셋의 양과 질을 개선하기 위해 다른 언어에서 일본어로 번역한 대규모 데이터셋을 사용하였습니다.

- **Performance Highlights**: 실험 결과, 일본어와 생물의학 지식이 풍부한 LLM이 생물의학 과제에서 더 나은 성능을 보였으며, 일본어 생물의학 분야에 특화되지 않은 LLM도 예상 외의 성능을 발휘했습니다. 그러나 특정 과제에서는 기존 LLM에서 여전히 개선의 여지가 많이 남아 있습니다.



### GAProtoNet: A Multi-head Graph Attention-based Prototypical Network for Interpretable Text Classification (https://arxiv.org/abs/2409.13312)
Comments:
          8 pages, 5 figues, submitted to COLING 2025

- **What's New**: 본 연구에서는 LM 인코더로 구축된 텍스트 분류 모델의 결정을 설명하기 위해 새롭게 설계된 GAProtoNet을 소개합니다. GAProtoNet은 입력 벡터와 프로토타입을 그래프의 노드로 간주하고, 멀티 헤드 그래프 어텐션(multi-head graph attention)을 활용하여 해석 가능한 프로토타입 표현을 학습합니다.

- **Technical Details**: GAProtoNet은 텍스트 임베딩, 프로토타입 레이어 및 그래프 어텐션 메커니즘으로 구성됩니다. 프로토타입 레이어는 입력 텍스트의 다양한 의미적 측면을 캡슐화하는 여러 개의 프로토타입 벡터를 형성하고, 그래프 어텐션 메커니즘은 텍스트 임베딩 벡터와 이웃 프로토타입 벡터 간의 관련성을 효과적으로 학습합니다. 이로 인해 결정을 내릴 때 어텐션 점수에 의해 가중치가 주어진 프로토타입의 선형 결합을 기반으로 합니다.

- **Performance Highlights**: 여러 공공 데이터셋에서의 실험 결과, GAProtoNet은 원래의 블랙 박스 모델의 정확도를 유지하면서도 더 나은 성능을 달성했습니다. GAProtoNet은 모든 비교 실험에서 최고의 정확도와 F1 점수를 기록하며, 텍스트 분류에서 결정적인 해석성을 제공합니다.



### Unsupervised Domain Adaptation for Keyphrase Generation using Citation Contexts (https://arxiv.org/abs/2409.13266)
Comments:
          Accepted at EMNLP 2024 Findings

- **What's New**: 이 논문은 keyphrase generation 모델을 새로운 도메인으로 적응시키기 위한 새로운 비지도 학습 방법, silk을 제안합니다. 이 방법은 citation context에서 silver-standard keyphrases를 추출하여 합성 레이블 데이터를 생성하여 도메인 적응 문제를 해결하고자 합니다.

- **Technical Details**: silk 방법은 citation context에서 문서와 silver-standard keyphrase 쌍을 생성하여 키프레이즈 생성 모델을 새로운 도메인에 적응시키는 방식으로 작동합니다. 실험에서는 Natural Language Processing, Astrophysics, Paleontology의 3개 과학 도메인에서 이 방법을 적용하였고, 문서 제목, 초록, 해당 citation context에서 키프레이즈 후보를 추출합니다.

- **Performance Highlights**: silk에서 생성한 합성 샘플을 사용하여 몇 샷 파인튜닝을 실시한 결과, 기존 Baselines에 비해 도메인 내 성능이 현저히 향상되었습니다. 실험을 통해 생성된 합성 샘플의 질을 평가하였으며, 적응 모델이 처음 도메인에서의 기억 상실, 선호 편향에 대한 점검을 수행했습니다.



### Towards LifeSpan Cognitive Systems (https://arxiv.org/abs/2409.13265)
- **What's New**: 이 연구에서는 LifeSpan Cognitive System (LSCS)이라는 새로운 인공지능 시스템을 제안하며, 이는 복잡한 환경에서 지속적이고 고빈도로 상호작용하는 기능을 갖춘 인간처럼 행동하는 시스템을 목표로 합니다. LSCS는 경험을 흡수하고 응답을 생성하는 두 가지 핵심 프로세스를 통해 작동하며, 과거 경험을 정확하게 기억하고 회상할 수 있는 능력을 강조합니다.

- **Technical Details**: LSCS의 주요 기술적 도전 과제는 (1) 추상화 및 경험 병합, (2) 장기 보관 및 정확한 회상입니다. 이 시스템은 환경에서의 상호작용을 통해 경험을 획득하고 이를 기존 메모리와 통합하는 방법을 모색합니다. 연구에서는 저장 복잡성(Storage Complexity)을 바탕으로 네 가지 기술 클래스를 제안하고 있으며, 이를 통해 LSCS 구축을 위한 새로운 패러다임을 구체화합니다.

- **Performance Highlights**: 존재하는 기술들이 각각의 한계를 지니고 있지만, LSCS는 이러한 기술들을 통합하여 효율적으로 작동할 수 있는 혁신적인 방법론을 제시합니다. LSCS는 신속하고 점진적인 정보 업데이트를 가능하게 하여 동적인 환경에 적응할 수 있는 능력을 갖추게 됩니다.



### Large Language Model Should Understand Pinyin for Chinese ASR Error Correction (https://arxiv.org/abs/2409.13262)
- **What's New**: 이번 연구에서는 Pinyin(병음)을 활용한 생성적 오류 교정(generative error correction)인 PY-GEC를 제안합니다. 이는 한국어 ASR(Automatic Speech Recognition) 시스템에서 오류를 교정하기 위해 Pinyin을 보조 정보로 사용하여 성능을 향상시키는 방법입니다.

- **Technical Details**: PY-GEC는 멀티태스크 학습(multitask training) 접근 방식을 도입하여 Pinyin과 텍스트 간의 변환 작업을 수행합니다. 사용자 입력의 가장 좋은 가설(one-best hypothesis)만을 사용하며 훈련 시에는 합성 오류(synthetic errors)만을 이용합니다.

- **Performance Highlights**: Aishell-1과 Common Voice 데이터셋에서의 실험 결과, PY-GEC는 텍스트만을 사용한 GEC보다 연속적으로 더 나은 성능을 보여주었으며, CER(Character Error Rate)을 8.3% 감소시키고 엔티티 리콜(entity recall)을 3.9% 향상시켰습니다.



### Neural-Symbolic Collaborative Distillation: Advancing Small Language Models for Complex Reasoning Tasks (https://arxiv.org/abs/2409.13203)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)의 복잡한 추론 능력을 학습하기 위한 새로운 지식 증류 방법인 Neural-Symbolic Collaborative Distillation (NesyCD)을 제안합니다. 기존의 작은 언어 모델(SLMs)이 복잡한 추론 작업을 처리하는 데 어려움을 겪는 이유를 설명하고, LLMs의 일반 기능과 전문 지식을 효과적으로 증류하는 방법을 제시합니다.

- **Technical Details**: NesyCD는 LLMs에서 일반적인 추론 기능과 공통 지식을 SLMs로 전이하며, 전문적인 지식은 상징적 지식 기반(KB)에 저장합니다. 우리는 SLM의 효과성을 높이기 위해 CoT(Cross-of-Thought) 증류 과정에서 생성된 정답과 오류 사례를 수집하고, LLMs가 이를 분석하여 전문 지식을 추출하도록 합니다. 또한, 여러 보조 작업을 포함해 다중 작업 학습을 통해 탐색된 전문 지식을 활용합니다.

- **Performance Highlights**: NesyCD는 BBH, GSM8K와 같은 데이터셋에서 SLM의 복잡한 추론 성능을 크게 향상시킵니다. 특히, LLaMA3-8B와 Qwen2-7B 모델이 GPT-3.5-turbo를 초과하여 성능을 발휘하며, LLaMA3-70B와 가까운 성능을 발휘했습니다. 이를 통해, 향상된 1.1B TinyLLaMA 모델이 기존 LLaMA2-7B보다 우수한 결과를 보이고, 훈련 데이터의 1/4만 사용하여도 경쟁력 있는 성능을 달성합니다.



### CITI: Enhancing Tool Utilizing Ability in Large Language Models without Sacrificing General Performanc (https://arxiv.org/abs/2409.13202)
- **What's New**: 제안된 연구에서는 Large Language Models (LLMs)의 tool-utilizing 능력을 향상시키기 위해 Component Importance-based Tool-utilizing ability Injection (CITI) 방법론을 제시했습니다. 이 방법은 중요한 구성 요소를 최적화하면서도 모델의 일반 성능을 해치지 않도록 설계되었습니다.

- **Technical Details**: CITI는 Mixture-Of-LoRA (MOLoRA) 어댑터를 중요한 구성 요소에 적용하고, 상대적으로 덜 중요한 구성 요소의 파라미터를 전체적으로 튜닝하여 모델의 성능을 향상시킵니다. 연구에서는 hidden representation의 변화와 구성 요소의 gradient-based 중요성 점수를 분석하여, LLM의 일반 성능과 tool-utilizing 능력 간의 trade-off을 이해합니다.

- **Performance Highlights**: 실험 결과, CITI는 API-Bank 데이터셋에서 LoRA보다 7.59% 더 우수하고, 전체 파라미터 튜닝보다 31.95% 더 나은 성과를 보였으며, ToolAlpaca 데이터셋에서도 각각 8.96% 및 29.03% 우수를 기록했습니다. 이는 모델의 tool-utilizing 능력을 효과적으로 향상시키면서도 일반 성능을 유지할 수 있음을 나타냅니다.



### CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information (https://arxiv.org/abs/2409.13199)
Comments:
          Work in progress

- **What's New**: CFSP(Compressed Fine-structured Sparse Pruning)라는 새로운 구조적 프루닝 프레임워크를 소개하며, 불필요한 매개변수를 제거하여 LLM의 효율성을 높입니다. 이 방식은 Coarse(거칠게)와 Fine-grained(세밀하게) 활성 정보 모두를 활용하여 프루닝을 안내합니다.

- **Technical Details**: CFSP는 단일 전방 패스를 통해 특징 활성화를 계산하며, 블록 간의 중요성에 따라 희소성 예산을 할당합니다. 각각의 블록 내에서는 상대적인 활성화와 가중치를 곱하여 불필요한 부분을 제거합니다. 이를 통해 트레이닝 오버헤드를 줄이고 성능을 향상시키는 회복 파인 튜닝 방법을 도입합니다.

- **Performance Highlights**: 실험 결과 CFSP는 다양한 모델과 희소성 예산에서 기존의 방법들을 초월하는 성능을 보여주며, 높은 희소성 수준에서도 도전적인 작업에서 우수한 결과를 나타냅니다.



### Exploring Scaling Laws for Local SGD in Large Language Model Training (https://arxiv.org/abs/2409.13198)
Comments:
          Technical Report

- **What's New**: 이 논문은 LLM(대규모 언어 모델) 훈련에서의 local SGD(국소 확률적 경사 하강법) 스케일링 법칙을 조사합니다. 이 방법은 느슨하게 연결된 장치에서의 훈련을 용이하게 하는 분산 최적화 알고리즘입니다.

- **Technical Details**: 실험을 통해 local SGD가 동일한 모델 파라미터(model parameters), 데이터셋(dataset), 그리고 컴퓨팅 자원(computational resources)을 고려했을 때 기존 방법들과 경쟁력을 가진 결과를 보여주었습니다. 또한, multi-cluster(다중 클러스터) 환경과 edge computing(엣지 컴퓨팅) 환경에서의 응용 가능성도 탐구했습니다.

- **Performance Highlights**: 우리의 연구 결과는 효과적인 multi-cluster LLM 훈련을 위한 필수 조건들을 명확히 하고, LLM 훈련 과정에서 엣지 컴퓨팅 자원을 활용하는 것의 잠재력과 한계를 조사했습니다. 이는 단일 대형 클러스터 훈련에 대한 대안으로서의 가능성을 입증합니다.



### An adapted large language model facilitates multiple medical tasks in diabetes car (https://arxiv.org/abs/2409.13191)
- **What's New**: 이 연구에서는 당뇨병 특화된 대규모 언어 모델(LLM)을 개발하고 평가하기 위한 새로운 프레임워크를 소개했습니다. 특히, 다양한 당뇨병 작업에 대한 효과성을 검증하는 데 중점을 두었습니다.

- **Technical Details**: 이 연구는 데이터 수집, 필터링, 증강(augmentation) 및 정제를 포함하는 포괄적인 데이터 처리 파이프라인을 개발했습니다. 이를 통해 고품질의 당뇨병 전용 데이터셋을 생성하고 다양한 평가 기준을 처음부터 만들었습니다. 수집된 교육 데이터를 활용하여, 당뇨병 수치 특화 LLM 가족을 세밀하게 조정하여 다양한 당뇨병 작업을 처리하는 데 있어서 최첨단 성능을 보였습니다.

- **Performance Highlights**: 임상 연구 결과, 우리의 모델은 개인 맞춤형 의료 제공, 의료 교육 지원, 임상 작업의 효율성 향상 등 당뇨병 관리에 있어 다양한 응용 가능성을 보여주었습니다. 또한, 모든 사용자에게 데이터 기반 지원을 제공하여 임상 실습을 개선할 수 있는 잠재력이 강조되었습니다.



### $\textit{SKIntern}$: Internalizing Symbolic Knowledge for Distilling Better CoT Capabilities into Small Language Models (https://arxiv.org/abs/2409.13183)
- **What's New**: 이 논문은 Small Language Models (SLMs)의 reasoning 능력을 향상시키기 위한 새로운 방법인 ‘SKIntern’을 제안합니다. SKIntern은 symbolic knowledge와 few-shot examples를 점진적으로 internalize 하여, inference 과정에서 컴퓨팅 오버헤드를 줄이고 reasoning 프로세스를 가속화합니다.

- **Technical Details**: SKIntern은 curriculum learning에 따라 미리 정의된 linear decay 일정에 의해 지도되며, 모델 fine-tuning 과정에서 symbolic knowledge를 internalize합니다. 이 방법은 초기에는 LLMs로부터 rationale과 symbolic knowledge를 생성하여, cosine similarity를 사용하여 가장 관련성이 높은 정보를 선택하는 두 단계로 구성됩니다. 그런 다음, token-level symbolic knowledge 압축과 instance-level example pruning이 수행되어 최적의 정보가 SLM을 fine-tuning하는 데 사용됩니다.

- **Performance Highlights**: SKIntern은 TinyLLaMA와 LLaMA2-7B 모델을 대상으로 한 여러 reasoning benchmarks에서 기존의 강력한 기준선보다 5% 이상 우수한 성능을 보였고, inference 비용(FLOPs 기준)을 최대 4배 줄였습니다. 이러한 결과는 SKIntern이 SLM의 inference 성능을 비용 효율적으로 최적화하는 데 매우 적합함을 보여줍니다.



### RRM: Robust Reward Model Training Mitigates Reward Hacking (https://arxiv.org/abs/2409.13156)
- **What's New**: 본 논문은 기존의 보상 모델(Reward Model, RM) 훈련에서 발생하는 주요 제약사항을 드러내고, 이를 해결하기 위한 인과적 프레임워크(causal framework)를 제안합니다. 이를 통해 비문맥적 아티팩트(irrelevant artifacts)로부터 독립적으로 선호를 학습할 수 있게 됩니다.

- **Technical Details**: 우리는 인과 그래프(causal graph)를 통해 인간 선호를 모델링하며, 데이터 증강(data augmentation) 기법을 도입하여 보상 모델 훈련 시 아티팩트를 효과적으로 제거합니다. 이를 통해 RM과 개선된 보상 모델(Robust Reward Model, RRM)을 훈련시켜, 보상의 진정한 품질을 파악하고자 합니다.

- **Performance Highlights**: RRM은 Gemma-2-9b-it에서 훈련된 쌍 간 보상 모델의 정확도를 80.61%에서 84.15%로 증가시키는 성과를 보였으며, 새롭게 훈련된 DPO 정책들은 MT-Bench에서 7.27에서 8.31로 향상되었고, AlpacaEval-2에서 길이 조절 승률이 33.46%에서 52.49%로 증가했습니다.



### Are Large Language Models Good Essay Graders? (https://arxiv.org/abs/2409.13120)
- **What's New**: 이번 연구에서는 자동화된 에세이 채점 자동화 시스템(AES)에서 두 가지 대형 언어 모델(Large Language Models, LLMs)인 ChatGPT와 Llama의 성능을 평가하였습니다. LLMs의 점수가 인간 채점자와 얼마나 일치하는지를 중심으로 다루고 있으며, 제로샷(zero-shot) 및 퓨샷(few-shot) 학습 접근 방식과 다양한 프롬프트 기법들을 고려하였습니다.

- **Technical Details**: 이 연구에서는 ASAP 데이터셋을 사용하여 LLM이 제공하는 수치 점수를 인간 평가자의 점수와 비교하였습니다. ChatGPT는 Llama에 비해 전반적으로 더 낮은 점수를 부여하며, 인간 평가자와의 상관관계도 낮은 것으로 나타났습니다. 이 에세이 채점 과정에서 글자 수, 연결어 사용, 문법 오류 수 등의 요소가 효과가 없음을 발견하였습니다.

- **Performance Highlights**: Llama 3의 성능이 이전 모델보다 일반적으로 더 좋다는 결과를 보고하며, LLM들이 인간 채점자와의 일관성을 보이지 않으나, 향후 에세이 채점을 보조하는 도구로서 사용 가능성이 있다는 점에서 긍정적인 결론을 내립니다.



### Guided Profile Generation Improves Personalization with LLMs (https://arxiv.org/abs/2409.13093)
Comments:
          EMNLP 2024 Findings

- **What's New**: 이 연구에서는 Guided Profile Generation (GPG)이라는 새로운 방법론을 제시하여 개인화(Personalization) 작업에서 LLMs의 성능을 개선하는 방법에 대해 다룹니다. GPG는 자연어로 개인 프로필을 생성하는 일반적인 방법으로, LLM이 개인 맥락을 해석하고 이를 기반으로 고품질의 자연어 개인 프로필을 생성할 수 있도록 돕습니다.

- **Technical Details**: GPG는 두 가지 주요 과제를 해결하는 데 초점을 맞추고 있습니다. 첫 번째는 개인 맥락의 복잡성과 핵심 정보의 희소성(sparsity) 문제입니다. 두 번째는 일반화(generalization)와 개인화 사이의 균형을 유지하는 것입니다. GPG는 개인 맥락을 분석하고 특정 질문에 답하여 자연어 개인 프로필을 생성하는 과정을 포함한 다단계 프로세스를 사용하며, 앙상블 모델을 통해 요구되는 작업에 대해 응답을 합니다.

- **Performance Highlights**: GPG는 다양한 작업에서 LLM의 개인화 성능을 향상시켰습니다. 특히, 온라인 구매에서 개인 선호도 예측 정확도를 37% 개선하였고, 트윗의 텍스트 패러프레이징에서 METEOR 점수를 2.24 향상시켰습니다. 이러한 결과는 GPG가 LLM의 개인화 능력을 크게 개선할 수 있음을 보여줍니다.



### LLM Surgery: Efficient Knowledge Unlearning and Editing in Large Language Models (https://arxiv.org/abs/2409.13054)
- **What's New**: 본 논문에서는 LLM(대형 언어 모델)의 문제적 및 구식 정보를 효과적으로 잊고 새로운 지식을 통합할 수 있는 방법, 즉 LLM 수술(LLM Surgery)이라는 프레임워크를 제안합니다.

- **Technical Details**: LLM 수술은 (1) 문제적 정보가 포함된 비학습 데이터셋에서 역경량화를 수행하고, (2) 새로운 정보가 포함된 업데이트 데이터셋에 대해 경량하강을 수행하며, (3) 변하지 않은 작은 하위 집합인 유지 데이터셋에서 KL 발산(KL divergence)을 최소화하는 세 가지 목표 함수(component objective function)를 최적화합니다. 이를 통해 모델의 성능을 보존하면서도 효과적으로 학습할 수 있도록 합니다.

- **Performance Highlights**: Llama2-7B 모델을 사용하여 우리의 방법론이 unlearn 집합에서 큰 잊기를 달성하고, update 집합에서 20%의 정확도 향상 및 retain 집합에서는 성능 유지에 성공했음을 보여줍니다.



### TACO-RL: Task Aware Prompt Compression Optimization with Reinforcement Learning (https://arxiv.org/abs/2409.13035)
Comments:
          Submitted to COLING 2025

- **What's New**: 본 논문에서는 기존의 비효율적인 프롬프트 압축 기법들을 개선하기 위해 강화학습을 기반으로 한 새로운 태스크 인지 프롬프트 압축 방법인 TACO-RL을 제안합니다. 이 방법은 태스크 특화된 보상 신호를 활용하여 모델을 학습시킴으로써 높은 성능을 유지하며 입력 토큰 수를 최소화하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 연구는 Transformer 기반의 토큰 분류 모델을 사용하여 입력 토큰의 태스크 특이적 보상 신호에 따라 학습 과정을 안내합니다. 또한, REINFORCE 알고리즘을 경량화하여 지연 시간(latency) 요건을 충족시키며, 세 가지 다양한 작업(텍스트 요약, 질문 응답, 코드 요약)에 대한 성능을 평가합니다.

- **Performance Highlights**: TACO-RL 방법은 LLMLingua-2 및 다른 최신 압축 기법들과 비교하여 세 가지 작업에서 각각 8%에서 260%의 성능 향상을 보여주었습니다. 이는 동일한 압축률과 지연 시간 요건을 유지하면서 이루어진 결과입니다.



### Seeing Through Their Eyes: Evaluating Visual Perspective Taking in Vision Language Models (https://arxiv.org/abs/2409.12969)
- **What's New**: 이 논문은 시각적 관점 이해(Visual Perspective-Taking, VPT) 능력의 평가를 위한 새로운 데이터셋인 Isle-Bricks와 Isle-Dots를 소개합니다. 이는 최근에 개발된 비전 언어 모델(Vision Language Models, VLMs)의 VPT 능력을 테스트하는 데 사용됩니다.

- **Technical Details**: 연구에서는 12개의 VLM을 평가했으며, 모델들은 VPT를 요구할 때 성능이 크게 하락하는 경향을 보였습니다. 또한, 객체 탐지(object detection) 성능은 VPT 성능과 잘 상관되지 않아 기존의 벤치마크가 이 문제를 충분히 이해하는 데 부족할 수 있음을 보여줍니다.

- **Performance Highlights**: 모델은 여러 사람이 있는 장면에서 더 크게 성능이 떨어지며, 다수의 사람들이 있는 이미지에서 VPT를 수행하는 데 어려움을 겪었습니다. 모든 모델에서 97% 이상 이해 가능한 답변을 제공했지만, 'Chain-of-Thought' 기술을 사용할 때는 때때로 비이해 가능 답변이 늘어나는 경향이 있었습니다.



### ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robot Navigation (https://arxiv.org/abs/2409.13682)
- **What's New**: 이 연구에서는 로봇이 긴 시간 동안 상호작용하며 다양한 질문에 대답할 수 있는 새로운 시스템인 Retrieval-augmented Memory for Embodied Robots(ReMEmbR)를 소개합니다. ReMEmbR는 로봇 내비게이션을 위한 긴 시간 비디오 질문 응답 시스템으로, 나아가 이를 평가하기 위해 NaVQA 데이터셋을 제안합니다.

- **Technical Details**: ReMEmbR는 기억 구축 및 질의 단계로 구성되며, 로봇의 역사적 데이터를 효과적으로 처리하기 위해 temporal information, spatial information, 그리고 이미지를 활용합니다. 이 시스템은 로봇이 긴 시간 동안 축적한 데이터를 바탕으로 다양한 질문에 대답하고 행동을 수행할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, ReMEmbR는 기존의 LLM과 VLM 기반 시스템보다 뛰어난 성능을 보여주었으며, 낮은 지연 시간으로 효과적인 장기적 추론이 가능함을 입증했습니다. 실제 로봇에 ReMEmbR를 배포한 결과, 다양한 질문에 대한 대응 능력을 보여줍니다.



### MaPPER: Multimodal Prior-guided Parameter Efficient Tuning for Referring Expression Comprehension (https://arxiv.org/abs/2409.13609)
Comments:
          EMNLP 2024

- **What's New**: 이 논문에서는 기존의 Referring Expression Comprehension (REC) 작업을 위한 새로운 효율적인 파라미터 전송 학습 방법인 MaPPER를 소개합니다. 이 방법은 Multimodal Prior-guided Parameter Efficient Tuning을 기반으로 하며, 구체적으로는 Dynamic Prior Adapters와 Local Convolution Adapters를 사용하여 지역적 의미를 추출하고 시각적 인식을 향상시킵니다.

- **Technical Details**: MaPPER는 Dynamic Prior Adapter (DyPA)와 Local Convolution Adapter (LoCA)를 포함합니다. DyPA는 정렬된 prior에 의해 각 토큰의 중요성을 계산해 동적으로 조정하며, LoCA는 다중 스케일의 지역적 지식을 통합하여 비주얼 트랜스포머의 표현력을 향상시킵니다. Prior-Guided Text 모듈을 도입하여 텍스트와 비주얼 정보를 융합합니다.

- **Performance Highlights**: 실험 결과, MaPPER는 RefCOCO, RefCOCO+, RefCOCOg와 같은 세 가지 벤치마크에서 이전의 SOTA 방법들보다 뛰어난 성능을 보이며, 단 1.41%의 튜닝 파라미터로 최고의 정확도를 달성하였습니다.



### YesBut: A High-Quality Annotated Multimodal Dataset for evaluating Satire Comprehension capability of Vision-Language Models (https://arxiv.org/abs/2409.13592)
Comments:
          EMNLP 2024 Main (Long), 18 pages, 14 figures, 12 tables

- **What's New**: 이번 논문에서는 이미지에서 풍자를 탐지하고 이해하는 어려운 작업을 제안하며, 이를 평가하기 위한 고품질 데이터셋인 YesBut을 발표합니다. 이 데이터셋은 2,547개의 이미지로 구성되어 있으며, 풍자적 이미지(1,084개)와 비풍자적 이미지(1,463개)가 포함되어 있습니다.

- **Technical Details**: 우리는 (1) Satirical Image Detection (풍자적 이미지 탐지), (2) Satirical Image Understanding (풍자적 이미지 이해), (3) Satirical Image Completion (풍자적 이미지 완성)라는 세 가지 벤치마크 작업을 제안합니다. 이 작업들은 VL(Vision-Language) 모델의 일반적인 기능을 초월하여 풍자의 맥락을 이해해야 합니다.

- **Performance Highlights**: 현재 SOTA VL 모델이 YesBut 데이터셋에서 풍자적 이미지 탐지, 이해, 완성 작업에서에서도 열악한 성능을 보였습니다. 특히 제로샷 환경에서는 풍자적 이미지 탐지가 매우 어려웠습니다.



### Demystifying and Extracting Fault-indicating Information from Logs for Failure Diagnosis (https://arxiv.org/abs/2409.13561)
Comments:
          This paper has been accepted by the 35th IEEE International Symposium on Software Reliability Engineering (ISSRE'2024)

- **What's New**: 본 논문은 로그 분석을 통한 고장 진단을 자동화하기 위해 LoFI라는 새로운 접근 방식을 제안합니다. 이 방법은 로그에서 고장 표시 정보를 자동으로 추출하는 두 가지 주요 단계를 포함합니다.

- **Technical Details**: LoFI의 첫 번째 단계에서는 의미적 유사성을 기반으로 고장과 관련된 로그를 수집하는 coarse-grained filtering을 수행합니다. 두 번째 단계에서는 미리 훈련된 언어 모델을 기반으로 한 novel prompt-based tuning 방법을 활용하여 수집된 로그에서 섬세한 정보를 추출합니다.

- **Performance Highlights**: LoFI는 Apache Spark 및 CloudA에서 수집된 데이터셋을 사용하여 평가되었으며, FID/FIP 추출에서 각각 87.4/80.6 점을 기록하며 기존 방법들보다 평균 81% 이상 높은 성능을 보여주었습니다.



### Contextualized Data-Wrangling Code Generation in Computational Notebooks (https://arxiv.org/abs/2409.13551)
Comments:
          To appear at ASE 2024

- **What's New**: 이 논문은 데이터 과학에서 중요한 단계인 데이터 조작(data wrangling)을 자동화하기 위한 코드 생성 방법을 탐구합니다. 특히, CoCoMine이라는 자동화된 접근 방식을 제안하며, 명확한 다중 모달(contextualized multi-modal) 맥락을 갖춘 데이터 조작 코드 생성 예제를 수집하여 훈련 모델을 개선하고자 합니다.

- **Technical Details**: 논문에서는 CoCoMine를 통해 데이터 흐름 분석(data flow analysis)을 통해 데이터 조작 코드 블록을 식별하고, 실행된 결과를 추적하여 맥락화된 데이터 조작 코드 예제를 정리합니다. 이를 통해 CoCoNote라는 58,221개의 예제를 포함하는 데이터셋을 생성하고, DataCoder라는 새로운 인코더-디코더 모델을 제안하여 코드, 텍스트 및 데이터 맥락을 별도로 인코딩합니다.

- **Performance Highlights**: 실험 결과, 데이터 맥락이 데이터 조작 코드 생성에서 중요한 요소임을 확인했습니다. 특히, 출력 예제가 입력 예제보다 더 효과적이며, GPT-4 모델이 50.6%의 실행 정확도(execution accuracy)를 달성하여 해당 작업에서 더 향상될 여지가 있음을 보여주었습니다.



### Sketching With Your Voice: "Non-Phonorealistic" Rendering of Sounds via Vocal Imitation (https://arxiv.org/abs/2409.13507)
Comments:
          SIGGRAPH Asia 2024

- **What's New**: 이 논문은 인간의 음성을 흉내 내는 방법을 자동으로 생성하는 시스템을 소개합니다. 이를 통해 목소리의 조절 매개변수를 조정하고, 인지 통신 이론을 적용하여 발음할 소리를 보다 직관적으로 일치시킬 수 있게 됩니다. 특히, 이는 시각적 표현의 '스케치'와 유사한 개념으로 청각적 경험을 전달하는 방법을 탐구합니다.

- **Technical Details**: 연구진은 인간의 발화 과정을 모델링하기 위해 간단한 제어 가능한 인간 발음 기관 모델을 사용합니다. 목표 소리에 맞춰 저수준 청각 특성에 기반하여 발음 기관 제어를 최적화합니다. 이 과정에서 인지 과학 이론을 통해 소리를 명확하게 흉내 내는 데 필요한 커뮤니케이션 추론 모듈을 통합하였습니다.

- **Performance Highlights**: 이 시스템은 다양한 소리에 대해 인간과 유사한 음성 모사를 수행할 수 있습니다. 사용자 연구 결과, 사람들은 주어진 소리의 음성 모사에서 이 시스템의 결과를 실제 인간의 생성한 음성과 비교했을 때 더 선호하는 것으로 나타났습니다. 이 방법은 새로운 제약에도 유연하게 적응할 수 있으며, 사용자가 요청한 음성 모사를 기반으로 소리를 검색하는 데 활용될 수 있습니다.



### Selective Exploration and Information Gathering in Search and Rescue Using Hierarchical Learning Guided by Natural Language Inpu (https://arxiv.org/abs/2409.13445)
Comments:
          Pre-print version of the accepted paper to appear in IEEE International Conference on Systems, Man and Cybernetics (SMC) 2024

- **What's New**: 본 논문에서는 SAR(검색 및 구조) 로봇의 의사결정 과정을 향상시키기 위해 대규모 언어 모델(LLM)과 계층적 강화 학습(HRL) 프레임워크를 통합한 시스템을 개발하였습니다. 이 시스템은 인적 자원의 입력을 기반으로 구체적인 전략 조정 및 행동 수행을 가능하게 합니다.

- **Technical Details**: 제안된 시스템은 LLM을 통해 사람의 언어적 입력을 해석하고, 이를 강화 학습(RL) 인사이트로 전환하여 SAR 로봇의 작업을 지원합니다. Context Extractor, Information Space, Strategic Decision Engine(SDE), Attention Space, Worker 등의 구성 요소로 이루어져 있으며, 이를 통해 로봇은 기본적인 명령 수행을 넘어 상황에 맞춘 의사결정을 수행할 수 있습니다.

- **Performance Highlights**: 이 연구를 통해 SAR 로봇은 인간의 제공하는 정보를 실시간으로 수집, 처리, 활용할 수 있게 되며, 정보의 분산이 비효율성으로 이어지는 문제를 해결합니다. 새로운 정보, 예를 들어 위험 요소나 피해자 위치 업데이트 등의 환경에 대한 정보는 로봇의 작업 완료 성공률을 유의미하게 높이는데 기여합니다.



### LLMs Still Can't Plan; Can LRMs? A Preliminary Evaluation of OpenAI's o1 on PlanBench (https://arxiv.org/abs/2409.13373)
- **What's New**: OpenAI의 최신 모델 o1 (Strawberry)은 기존의 LLM과 구별되는 새로운 reasoning 모델로 평가되고 있으며, 이는 Chain-of-Thought (CoT) 기반 추론이 특징이다. 이 논문은 o1의 성능을 PlanBench에서 평가하고 그 한계를 탐구한다.

- **Technical Details**: PlanBench는 LLM의 계획 능력을 평가하는 벤치마크로, 600개의 블록 문제를 포함하며, o1은 이러한 평가에서 상당한 성과를 달성했다. o1은 강화 학습 기반(pre-training RL)으로 훈련되어 있으며, 유사한 모델인 AlphaGo와 유사한 메커니즘을 가질 가능성이 있다. 저자들은 효율성, 비용, 보증을 포함하여 LRM의 추론 능력을 측정하는 새로운 접근 방식이 필요하다고 주장한다.

- **Performance Highlights**: o1의 성능은 PlanBench에서 경쟁 모델보다 뛰어나지만 여전히 완전한 결정을 내리기에는 부족하다. LLaMA 3.1 405B가 Blocksworld에서 62.6%의 정확도로 최고 성과를 기록했으나, Mystery Blocksworld에서는 성능이 대폭 저조하였다. 이 연구는 LLM의 계획 능력에 대한 지속적인 한계를 명확히 보여준다.



### SLaVA-CXR: Small Language and Vision Assistant for Chest X-ray Report Automation (https://arxiv.org/abs/2409.13321)
- **What's New**: 대규모 언어 모델(LLMs)의 성공에 영감을 받아 의료 분야에서 임상 의사들을 지원하기 위한 LLM 개발에 대한 연구가 증가하고 있습니다. 그러나 상용 모델을 사용하는 것에는 개인정보 보호 문제가 있으며, 오픈 소스 모델은 많은 계산 자원을 필요로 합니다. 이를 해결하기 위해 우리는 가슴 엑스레이(CXR) 보고서 자동화를 위한 오픈 소스 소형 언어 및 비전 도우미(SLaVA-CXR)를 제안합니다.

- **Technical Details**: 제안한 SLaVA-CXR는 Re3Training 방법론을 통해 훈련됩니다. 이 방법론은 방사선 의사의 인지 발달을 모사하며, Recognition(인식), Reasoning(추론), Reporting(보고) 세 가지 단계로 구성됩니다. 또한 RADEX라는 데이터 합성 방법을 통해 고품질의 다양한 훈련 데이터를 합성하여 개인정보 보호 규정을 준수합니다.

- **Performance Highlights**: 2.7B 백본을 기반으로 구축된 SLaVA-CXR는 기존의 대형 모델에 비해 6배 빠른 추론 효율을 달성하는 동시에 CXR 보고서 생성 및 요약 작업에서 최고의 성능을 보입니다.



### RLHFuse: Efficient RLHF Training for Large Language Models with Inter- and Intra-Stage Fusion (https://arxiv.org/abs/2409.13221)
- **What's New**: RLHFuse는 RLHF(Reinforcement Learning from Human Feedback) 훈련의 세밀한 최적화를 통해 GPU 활용도를 극대화하는 새로운 접근 방식을 제안합니다. 이를 위해 기존의 태스크를 더욱 세분화된 하위 태스크로 나누어 각 단계의 최적화를 수행합니다.

- **Technical Details**: RLHFuse는 두 가지 핵심 아이디어를 기반으로 작동합니다. 첫째, 생성 생성 및 추론 태스크를 샘플 수준의 하위 태스크로 나누어 inter-stage fusion을 통해 긴 꼬리 샘플에 의해 발생하는 병목 현상을 완화합니다. 둘째, 훈련 태스크는 마이크로 배치의 하위 태스크로 나누어 intra-stage fusion을 수행하여 파이프라인 버블을 감소시킵니다. RLHFuse는 이러한 최적화를 통해 전통적인 RLHF 프레임워크에서 발생하는 문제들을 해결합니다.

- **Performance Highlights**: RLHFuse는 다양한 유명 LLMs에서 평가되었으며, 기존의 최첨단 시스템에 비해 훈련 처리량이 최대 3.7배 증가함을 보여주었습니다.



### ChemDFM-X: Towards Large Multimodal Model for Chemistry (https://arxiv.org/abs/2409.13194)
Comments:
          19 pages, 7 figures, 11 tables

- **What's New**: 이 연구에서는 화학 데이터의 다양한 모드와 작업을 처리하는 첫 번째 교차 모드 화학 대화 기초 모델인 ChemDFM-X를 소개합니다. ChemDFM-X는 여러 화학 모드를 이해하고 해석할 수 있는 능력을 갖추고 있으며, 모든 모드에서 단일 모델 가중치 세트로 다양한 하위 작업을 수행할 수 있습니다.

- **Technical Details**: ChemDFM-X는 약 7.6M 개의 교차 모드 데이터를 포함하는 지침 튜닝 데이터세트를 활용하여 다중 모드 입력을 이해하고 추론할 수 있는 능력을 갖추고 있습니다. 이 데이터는 SMILES에서 생성된 대량 생성 및 작업 특화 모델 예측을 통해 보완됩니다.

- **Performance Highlights**: ChemDFM-X는 기존 전문 모델이나 다중 모드 모델(의 성능을 초월하여 고유의 화학 작업을 효과적으로 처리할 수 있으며, 대부분의 일반적인 화학 모드를 잘 다루는 것으로 입증되었습니다. 이를 통해 ChemDFM-X는 화학 분야에서 다중 모드 지능(CG)의 중요한 이정표를 나타냅니다.



### Personalized Speech Recognition for Children with Test-Time Adaptation (https://arxiv.org/abs/2409.13095)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 어린이 음성 인식(ASR)에서 새로운 테스트 시 적응(test-time adaptation, TTA) 기법을 적용함으로써, 성인 음성을 바탕으로 사전 훈련된 ASR 모델이 어린이 음성에 지속적으로 적응할 수 있도록 하는 혁신적인 파이프라인을 제안합니다.

- **Technical Details**: 우리의 제안된 ASR 시스템은 사전 훈련된 wav2vec 2.0 ASR 모델과 두 가지 최신 TTA 기법(SUTA, SGEM)을 결합하여 어린이 음성을 인식합니다. TTA 방법은 추가적인 인간 주석 없이 새로운 사용자 데이터에 ASR 모델을 적응시킬 수 있습니다.

- **Performance Highlights**: TTA 방법을 적용한 ASR 모델은 일반적으로 사용되는 ASR 기준 모델에 비해 통계적으로 유의미한 성능 향상을 보였으며, 각 어린이 화자의 음성 데이터 간의 데이터 도메인 변화를 분석하여 TTA의 필요성을 확인했습니다.



### Embedding Geometries of Contrastive Language-Image Pre-Training (https://arxiv.org/abs/2409.13079)
Comments:
          ECCV 2024 - Beyond Euclidean Workshop

- **What's New**: 이 논문은 CLIP의 전통적인 설계 방식인 L2 정규화와 코사인 유사성의 로그 확률을 재검토하고, 더 직관적인 유클리드 기하학을 기반으로 한 Euclidean CLIP (EuCLIP)을 제안합니다. EuCLIP은 CLIP의 성능을 일치시키거나 초과하며, 더 복잡한 하이퍼볼릭 대안들보다도 계층적 관계를 잘 지원합니다.

- **Technical Details**: 이 연구에서는 다양한 임베딩 기하학을 체계적으로 테스트했습니다. 여기에는 코사인 유사성(Cosine Similarity)을 사용하는 ELIP와 음의 거리 제곱 로그릿(Negative Distance Squared Logit)이 포함됩니다. 실험 결과, 시각 및 텍스트 변환기(Vision and Text Transformers)의 최종 레이어 정규화(LayerNorm)는 성능을 저하시키며, Euclidean 기하학 및 하이퍼볼릭 기하학 모두에서 EuCLIP이 CLIP와 유사한 또는 우수한 성능을 보여주었습니다.

- **Performance Highlights**: EuCLIP은 기존 CLIP 모델과 비교하여 제로 샷 이미지 분류(zero-shot image classification) 및 검색(retrieval) 능력을 유지하면서, 더 복잡한 모델인 MERU와 동등한 또는 더 나은 성능을 달성했습니다.



### Natural Language Processing Methods for the Study of Protein-Ligand Interactions (https://arxiv.org/abs/2409.13057)
Comments:
          52 Pages and 3 Figures

- **What's New**: 최신 연구에서는 단백질-리간드 상호작용(PLIs)을 예측하기 위한 효과적인 방법들이 소개되고 있으며, 이는 약물 발견 및 단백질 공학에 매우 중요합니다. 특정 단백질-리간드 쌍에 대한 실험적 데이터를 생성하는 기존의 접근 방식 대신 기계 학습(Machine Learning)과 자연어 처리(Natural Language Processing, NLP) 기술이 활용되고 있습니다.

- **Technical Details**: NLP는 언어 구조를 분석하고 조작하여 컴퓨터 자동화를 위한 방법론을 제공합니다. 최근 ChatGPT와 같은 NLP 도구들이 폭넓게 사용되고 있으며, 이러한 기술들은 단백질과 리간드의 텍스트 표현을 분석하고 예측하는 데 사용됩니다. 본 논문에서는 긴 짧은 메모리(Long Short-Term Memory), 트랜스포머(Transformers), 주의(attention)와 같은 모델들이 PLI 연구에 어떻게 적용되는지를 설명합니다.

- **Performance Highlights**: NLP 기법을 통해 단백질 응집 문제를 해결하고 PLI 연구에서 예측 모델이 유용하다는 사실이 강조되었습니다. 특히, AlphaFold와 같은 기술은 단백질의 3D 구조를 예측하는 데 성공하여 큰 주목을 받았습니다. 그러나 이들 모델 구축에는 상당한 계산적 부담이 따르고, 입력 데이터의 어떤 요소가 예측 성공에 기여하는지 이해하는 것이 어렵다는 한계가 있습니다.



### CraftRTL: High-quality Synthetic Data Generation for Verilog Code Models with Correct-by-Construction Non-Textual Representations and Targeted Code Repair (https://arxiv.org/abs/2409.12993)
- **What's New**: 이번 논문에서는 하드웨어 설명 언어인 Verilog에서 코드 생성을 위한 대규모 언어 모델(LLM)의 성능을 개선하기 위한 새로운 방법을 제시합니다. 특히, 비텍스트적 요소(예: Karnaugh 맵, 상태 전이 다이어그램, 파형) 처리의 어려움과 훈련 중 발생하는 랜덤한 실수 문제를 해결하기 위해 데이터 선별 방법을 향상시켰습니다.

- **Technical Details**: 연구진은 비텍스트적 표현을 처리하기 위한 정합성을 갖춘 데이터를 생성하고, 다양한 모델 체크포인트에서 에러 리포트를 자동 생성하여 이를 공개 소스 코드에 주입하는 프레임워크를 도입하였습니다. Starcoder2-15B 모델을 통해 VerilogEval-Machine, VerilogEval-Human, RTLLM에서 각각 3.8%, 10.9%, 6.6%의 성과 향상을 기록했습니다.

- **Performance Highlights**: 이번 결과는 Starcoder2-15B 모델이 기존의 최첨단 성능을 초과함을 보여주며, 특히 각 평가 벤치마크에서 더 높은 정확성을 보였습니다. 이러한 성과는 Verilog 코드 생성의 정확성 향상에 크게 기여했습니다.



New uploads on arXiv(cs.IR)

### Data Augmentation for Sequential Recommendation: A Survey (https://arxiv.org/abs/2409.13545)
- **What's New**: 이 논문은 Sequential Recommendation(SR)의 데이터 희소성 문제를 해결하기 위한 Data Augmentation(DA) 방법에 대한 포괄적인 리뷰를 제공합니다.

- **Technical Details**: Sequential Recommendation은 사용자의 역사적 시퀀스 데이터를 학습하여 미래의 상호작용을 예측합니다. DA 방법은 원본 데이터의 다양성을 증가시키거나 품질을 향상시키는데 초점을 맞추며, 주로 Heuristic-based(휴리스틱 기반)와 Model-based(모델 기반) 방법으로 분류됩니다. 휴리스틱 기반 방법은 Data-level(데이터 수준)과 Representation-level(표현 수준)으로 나뉘며, 모델 기반 방법은 Sequence Extension(시퀀스 확장) 및 Generation(생성)으로 나뉩니다.

- **Performance Highlights**: 이 논문은 다양한 DA 방법의 장단점을 비교하고, SR 모델의 성능 향상을 위한 대표적인 실험 결과를 제시합니다. 이는 향후 연구 방향을 제시함과 동시에 SR 분야의 데이터 중심 AI(Data-centric AI) 발전에 기여할 것으로 기대됩니다.



### Procedure Model for Building Knowledge Graphs for Industry Applications (https://arxiv.org/abs/2409.13425)
- **What's New**: 이번 논문에서는 다양한 산업 분야에서 사용할 수 있는 RDF 지식 그래프(knowledge graph)를 구축하기 위한 단계별 절차 모델을 제안합니다. 이 모델은 비즈니스 및 데이터 이해를 시작으로, 온톨로지 모델링과 그래프 설정을 포함하여, 평가 및 배포의 프로세스 단계로 구성되어 있습니다.

- **Technical Details**: 해당 절차 모델은 데이터 마이닝을 위한 크로스 산업 표준 프로세스(CRISP-DM)를 기반으로 하며, RDF(Resource Description Framework), RDF Schema(RDFS), 웹 온톨로지 언어(OWL) 등을 사용합니다. 이 모델은 산업 응용 프로그램의 복잡한 프로세스와 다양한 데이터 사일로를 고려하여 설계되었습니다.

- **Performance Highlights**: 이 연구에서는 산업 응용 프로그램을 위한 지식 그래프 구축이 조직의 비즈니스 데이터와 지식을 통합하여 인사이트를 제공하고, 70%의 데이터 분석 효율 향상을 이끄는 사례를 제시합니다. 특히, 비즈니스 목표와 요구사항을 충족하는 데 중요한 역할을 하는 ' competency questions (CQ)'의 활용이 강조됩니다.



### More Clustering Quality Metrics for ABCDE (https://arxiv.org/abs/2409.13376)
- **What's New**: 이 논문에서는 클러스터링의 품질과 차이를 평가하기 위한 ABCDE(‘Application-Based Cluster Diff Evals’) 기법을 확장하여 새로운 메트릭 IQ를 도입하고, DeltaRecall 측정 기법을 제안합니다. 이로 인해 클러스터링 변화의 질적 변화를 평가하는 데 도움이 됩니다.

- **Technical Details**: ABCDE는 두 가지 클러스터링(Baseline 클러스터링과 Experiment 클러스터링)의 차이를 Impact와 Quality 메트릭을 사용하여 평가합니다. 기존 품질 메트릭에는 GoodSplitRate, BadSplitRate, GoodMergeRate, BadMergeRate, DeltaPrecision이 포함됩니다. 이 논문에서는 ΔRecall의 특성을 파악하고, 클러스터링 변화가 품질 개선으로 이어지는 정도를 나타내는 새로운 메트릭 IQ를 제안합니다.

- **Performance Highlights**: 새롭게 도입된 메트릭 IQ는 클러스터링의 질이 변화함에 따라 얼마나 개선되었는지를 정량적으로 측정하게 하여, 사용자는 최적의 클러스터링을 선택하는 데 필요한 인사이트를 제공합니다.



### TRACE: Transformer-based user Representations from Attributed Clickstream Event sequences (https://arxiv.org/abs/2409.12972)
Comments:
          RecSys Workshop on Recommenders in Tourism (RecTour 2024), October 14th-18th, 2024, co-located with the 18th ACM Conference on Recommender Systems, Bari, Italy

- **What's New**: TRACE는 여러 사용자 세션에서의 클릭스트림 데이터를 기반으로 사용자 임베딩을 생성하기 위한 새로운 변환기 기반 접근 방식입니다. 기존 연구와 달리 단일 세션 제품 시퀀스에 주로 집중하는 대신, TRACE는 총체적인 사용자 선호도와 의도를 캡처합니다.

- **Technical Details**: TRACE는 다중 작업 학습(multi-task learning) 프레임워크를 활용하여 다수의 사용자 참여 대상을 예측하는 경량 변환기 인코더를 훈련합니다. 각 사용자의 방문 기록은 페이지 뷰 이벤트로 기록되며, 세션 간 연속된 페이지 조회를 모델링하여 풍부한 사용자 여정 표현을 생성합니다.

- **Performance Highlights**: TRACE는 대규모 여행 전자상거래 데이터셋을 통한 광범위한 실험에서 기존의 변환기(Transformer) 및 LLM 스타일 아키텍처에 비해 우수한 성능을 보여주었습니다. 학습된 임베딩의 시각화는 잠재적인 사용자 상태와 행동에 해당하는 의미 있는 클러스터를 드러내어 추천 시스템 향상 가능성을 강조합니다.



### Beauty Beyond Words: Explainable Beauty Product Recommendations Using Ingredient-Based Product Attributes (https://arxiv.org/abs/2409.13628)
Comments:
          18th ACM Conference on Recommender Systems, Workshop on Strategic and Utility-aware REcommendation

- **What's New**: 이 논문에서는 뷰티 제품의 성분을 기반으로 뷰티 관련 속성을 정확하게 추출하기 위한 새로운 시스템을 제시합니다. 이 시스템은 엔드 투 엔드 지도 학습(end-to-end supervised learning) 접근 방식을 사용하며, 에너지 기반의 암묵적 모델 아키텍처(energy-based implicit model architecture)를 핵심으로 하여 속도의 향상, 설명 가능성, 강인성, 유연성을 제공합니다.

- **Technical Details**: 우리의 모델은 BERT와 유사한 양방향 Transformer 인코더 네트워크를 기반으로 구축되며, 최종 어텐션 레이어에 약간의 수정을 가했습니다. 모델은 사용자 쿼리 속성(query attribute), 재료 리스트(ingredients), 제품 제목(product title)을 입력으로 받아 해당 속성의 확률을 출력합니다. 이는 제품의 성분에 따라 주어진 속성이 맞는지 여부를 예측하는 방식으로 작동합니다.

- **Performance Highlights**: 이 모델은 전통적인 키워드 기반 솔루션과 비교했을 때 뛰어난 정확도와 정밀도를 보여줍니다. 또한, 모델은 시각적 해석(attention weights 분석)을 통해 설명 가능성을 제공하며, 저자원 환경에서도 강인성을 유지하고, 새 레이블에 대해 이전에 훈련된 모델을 쉽게 미세 조정할 수 있는 유연성을 특징으로 합니다.



### Advancing Event Causality Identification via Heuristic Semantic Dependency Inquiry Network (https://arxiv.org/abs/2409.13621)
- **What's New**: 이번 연구에서는 Event Causality Identification (ECI) 문제를 해결하기 위한 새로운 접근법으로 Semantic Dependency Inquiry Network (SemDI)를 제안합니다. SemDI는 텍스트 내 사건 간의 인과 관계를 파악하는 데 초점을 맞추고 있으며, 문맥 속의 의미적 의존성을 포착하는 통합 인코더를 활용합니다.

- **Technical Details**: SemDI 모델은 사건 쌍 중 하나의 사건을 무작위로 마스킹한 후, Cloze Analyzer를 사용하여 Context에 대한 포괄적인 이해를 바탕으로 채워야 할 토큰을 생성합니다. 이 토큰은 특정 사건 간의 인과 관계를 질의하는 데 사용됩니다.

- **Performance Highlights**: SemDI는 세 가지 널리 사용되는 벤치마크에서 이전 SOTA 모델에 비해 각각 7.1%, 10.9%, 14.9%의 F1-score 향상을 기록하며 우수한 성능을 입증하였습니다.



### A Multimodal Dense Retrieval Approach for Speech-Based Open-Domain Question Answering (https://arxiv.org/abs/2409.13483)
- **What's New**: 이 연구에서는 자동 음성 인식(ASR) 모델 없이 음성 질문에 직접적으로 작동하는 다중 모드 밀집 검색기(multi-modal dense retriever)를 제안합니다. 이 접근 방식은 기존의 ASR-Retriever 파이프라인의 한계를 줄이고, 훈련에서도 ASR 모델 없이 엔드 투 엔드로 가능하다는 장점이 있습니다.

- **Technical Details**: 제안된 방법은 밀집 텍스트 검색을 위한 이중 인코더 구조를 수정하여 스피치 입력을 처리합니다. 질문 인코더 부분은 HuBERT(에이치유버트)라는 자가 지도 학습 모델로 변경되며, 구문 인코딩은 BERT 기반의 패시지 인코더를 그대로 유지합니다. 이는 음성 신호를 사용하여 질문을 적절히 인코딩합니다.

- **Performance Highlights**: 실험 결과, ASR 없이 훈련된 다중 모드 밀집 검색기가 ASR-Retriever 파이프라인보다 짧은 질문에서 더 나은 성능을 보여주는 것으로 나타났습니다. 특히 ASR에서 중요한 단어가 잘못 전사되었을 경우에도 더 안정적인 검색 성능을 발휘했습니다.



### Contextual Compression in Retrieval-Augmented Generation for Large Language Models: A Survey (https://arxiv.org/abs/2409.13385)
Comments:
          Ongoing Work

- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG)이 큰 언어 모델(LLMs)의 오류를 줄이고, 일관성 및 정확성을 향상시키기 위한 최근의 발전에 대해 집중적으로 탐구합니다. 특히, 문맥 압축(Contextual Compression) 방법론의 진화와 언어 모델의 성능을 개선하기 위한 제안된 대안들을 다룹니다.

- **Technical Details**: RAG는 외부 데이터베이스를 활용하여 문서 조각을 검색하고, 중요한 정보를 식별하기 위해 의미적 유사성 측정값을 사용합니다. 이 논문에서는 의미적 압축(Semantic Compression), 인-컨텍스트 오토인코더(In-Context Auto-Encoders), 임베딩 필터(Embeddings Filter)와 같은 다양한 압축 방법을 체계적으로 분석하고, 새로운 분류 체계를 제시합니다. 이러한 방법들은 LLM의 문맥 처리 능력을 향상시키는 데 기여합니다.

- **Performance Highlights**: 연구 결과, 문맥 정보를 사용하여 LLM의 성능을 크게 향상시킬 수 있으며, 제로샷 학습(zero-shot learning) 및 복잡한 태스크 수행에 유리하다는 것이 입증되었습니다. 또한, 이 논문은 향후 연구 방향을 제시하고, 지속 가능하고 환경적으로 책임 있는 LLM 개발의 필요성을 강조합니다.



### A Unified Causal Framework for Auditing Recommender Systems for Ethical Concerns (https://arxiv.org/abs/2409.13210)
Comments:
          28 pages

- **What's New**: 추천 시스템에 대한 감사(auditing) 접근 방식을 인과적(causal) 관점에서 제안하며, 사용자 주체성(user agency)을 측정하기 위한 새로운 감사 메트릭을 정의한 것이 주목할 만한 새로운 점입니다.

- **Technical Details**: 기존 감사 메트릭을 분류하고, 추천 시스템의 다단계 동적 과정을 고려하여 사용자 주체성 측정을 위한 두 가지 메트릭인 과거 및 미래 도달 가능성(past- and future-reachability)과 안정성(stability)을 제안하였습니다. 이 메트릭을 계산하기 위한 그래디언트 기반(gradient-based) 및 블랙박스(black-box) 접근 방식을 제공.

- **Performance Highlights**: 실험 결과, 추천 시스템의 스토캐스틱성(stochasticity)이 증가할 경우 사용자 추천의 안정성은 높아지나 도달 가능성은 낮아지는 경향을 보였습니다. 특히 행렬 분해(matrix factorization) 기반 추천 시스템은 더 나은 도달 가능성을 제공했지만 안정성 면에서는 덜 안정적임을 확인했습니다.



### RPAF: A Reinforcement Prediction-Allocation Framework for Cache Allocation in Large-Scale Recommender Systems (https://arxiv.org/abs/2409.13175)
- **What's New**: 이 논문에서는 기존의 캐시 할당 문제에서 발생하는 두 가지 주요 과제를 다루고, 강화 학습 기반의 예측-할당 프레임워크(RPAF)를 제안합니다. 특히, RPAF는 가치-전략 의존성(value-strategy dependency)과 스트리밍 할당 문제(streaming allocation problem)를 해결하기 위해 설계되었습니다.

- **Technical Details**: RPAF는 예측 단계와 할당 단계로 구성된 2단계 프레임워크입니다. 예측 단계에서는 강화 학습을 활용하여 다양한 캐시 선택의 값을 추정하며, 할당 단계에서는 각 사용자 요청에 따라 캐시 선택을 결정합니다. 또한, 제약 조건을 해결하기 위한 완화된 로컬 할당기(Relaxed Local Allocator, RLA)와 스트리밍 방식의 PoolRank 알고리즘을 사용하여 예산 제약을 준수하는 최적의 할당 전략을 제공합니다.

- **Performance Highlights**: 실험 결과, RPAF는 계산 예산 제약 하에서도 사용자 참여도를 현저히 개선한 것으로 나타났습니다. 오프라인 실험과 온라인 A/B 테스트를 통해 RPAF의 효과가 입증되었으며, 실제 응용 프로그램에 배포되어 눈에 띄는 개선을 가져왔습니다.



New uploads on arXiv(cs.CV)

### Colorful Diffuse Intrinsic Image Decomposition in the Wild (https://arxiv.org/abs/2409.13690)
Comments:
          12 pages, 12 figures. Accepted to SIGGRAPH Asia 2024 (TOG). Webpage: this https URL

- **What's New**: 본 연구는 입력 이미지를 확산 알베도(difffuse albedo), 다채로운 확산 음영(colorful diffuse shading), 및 반사 성분(specular residual)으로 분리하는 방법을 제시합니다. 기존 방법들이 단일 색상 조명과 램버티안 세계 가정을 사용한 데 반해, 우리 방법은 이러한 가정을 점진적으로 제거하여 더 복잡한 실제 장면의 이미지를 처리할 수 있게 했습니다.

- **Technical Details**: 우리의 방법은 먼저 장면의 전역 맥락을 사용하여 음영의 색조(chroma)를 추정한 다음, 희소한 확산 알베도를 생성합니다. 이후 주어진 확산 알베도를 바탕으로 음영을 확산 성분과 반사 성분으로 추가 분해합니다. 이 방법은 복잡한 비구조적 작업을 여러 개념적으로 간단한 하위 문제로 나누어 복잡한 장면에 일반화할 수 있도록 합니다.

- **Performance Highlights**: 우리는 다양한 벤치마크 및 실제 환경에서 정성적 및 정량적으로 우리의 방법의 유효성을 광범위하게 평가했습니다. 또한, 픽셀별 화이트 밸런싱 및 반사 제거와 같은 여러 조명 인식(image editing) 응용 프로그램을 통해 본 연구의 유용성을 입증합니다.



### Temporally Aligned Audio for Video with Autoregression (https://arxiv.org/abs/2409.13689)
Comments:
          Submitted to ICASSP 2025. Project page https://v-aura.notion.site

- **What's New**: V-AURA는 비디오-오디오 생성에서 높은 시간 동기화(temporal alignment)와 관련성을 달성한 최초의 자기 회귀 모델입니다. 고프레임 비주얼 피처 추출기(high-framerate visual feature extractor)와 크로스 모달 오디오-비주얼 피처 융합(cross-modal audio-visual feature fusion) 전략을 활용하여 세밀한 비주얼 모션 이벤트를 포착하고 정밀한 시간 동기화를 보장합니다. 또한, 비디오와 오디오 간의 높은 관련성을 갖는 벤치마크 데이터셋인 VisualSound를 제안합니다.

- **Technical Details**: V-AURA는 고충실도 신경 오디오 코덱(neural audio codec)을 사용하여 비주얼 스트림에 따라 의미적 및 시간적으로 정렬된 오디오를 생성합니다. 입력 비디오에서 세밀한 비주얼 및 모션 특징을 추출하고 이를 오디오의 시간 차원에 맞게 업샘플링합니다. 이 때, 오디오와 비주얼 특징이 융합되어 자연스러운 오디오-비디오 동시 발생을 강조합니다. V-AURA는 VisualSound 데이터셋에서 훈련되어 강력한 오디오-비주얼 관련성을 보장합니다.

- **Performance Highlights**: V-AURA는 현재의 최첨단 모델들과 비교하여 시간 동기화 및 의미적 관련성에서 뛰어난 성능을 보여주며 오디오 품질 또한 비교 가능한 수준을 유지하고 있습니다. 모델은 비디오-오디오 생성에서 강력한 관련성과 시간 동기화를 달성하는 첫 번째 자기 회귀 모델로 자리 잡았습니다. 또한, 데이터셋 수가 줄어들면서 훈련 시간이 크게 감소하였습니다.



### Morphological Detection and Classification of Microplastics and Nanoplastics Emerged from Consumer Products by Deep Learning (https://arxiv.org/abs/2409.13688)
- **What's New**: 이 논문에서는 새로운 마이크로 및 나노플라스틱(MiNa, micro- and nanoplastics) 데이터셋을 소개하여 자동 검출 및 분류를 가능하게 하였습니다. 이 데이터셋은 실제 수중 조건에서 시뮬레이션된 스캐닝 전자 현미경 이미지를 포함하고 있으며, 다양한 크기와 폴리머 종류에 따라 플라스틱을 분류합니다.

- **Technical Details**: MiNa 데이터셋은 고급 물체 검출 알고리즘을 이용한 자동 검출 및 분류를 지원하며, 각기 다른 물체 검출 기술들을 적용하여 마이크로 및 나노플라스틱의 특성을 평가합니다. 사용된 주요 모델들은 Mask R-CNN, Faster R-CNN, YOLOv10 등입니다.

- **Performance Highlights**: 이 데이터셋은 4가지 주요 폴리머 타입을 포함하고 있으며, 기존 데이터셋 보다 더 넓은 크기 범위를 다루고 있습니다. 공개된 데이터셋으로서 마이크로플라스틱 연구에 크게 기여할 수 있는 가능성을 보여줍니다.



### A Bottom-Up Approach to Class-Agnostic Image Segmentation (https://arxiv.org/abs/2409.13687)
- **What's New**: 이번 연구는 클래스 비의존적인 이미지 분할을 위한 새로운 하향식(bottom-up) 접근 방식을 제시합니다. 기존의 방법들은 일반적으로 상향식(top-down) 형식을 따르며, 객체 탐지가 객체 분할보다 우선시되지만, 이 논문에서는 역으로 접근하여 새로운 분할 기법을 개발하였습니다.

- **Technical Details**: 제안된 방법은 특성 공간의 투영 구(sphere)에서 네트워크를 직접 감독합니다. 이는 메트릭 학습(metric learning) 문헌에서 영감을 받은 손실(loss) 함수와 새로운 분할 공간 표현에서 정의된 손실 함수를 사용하는 방식입니다. 최종 세분화(segmentation) 결과는 추정된 특성의 평균 이동 클러스터링(mean-shift clustering)을 통해 획득됩니다.

- **Performance Highlights**: 제안된 하향식 방법은 클래스 기반 분할을 위해 설계된 데이터셋을 사용해도 뛰어난 일반화 능력을 보여줍니다. 특히, 세포 및 핵(segmentation of cell and nucleus) 분할과 같은 복잡한 작업에서 효과성을 입증하였습니다. 따라서 이번 연구는 다양한 분할 문제에 대한 귀중한 통찰을 제공할 것입니다.



### V^3: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians (https://arxiv.org/abs/2409.13648)
- **What's New**: V³(Viewing Volumetric Videos)는 동적 Gaussian의 스트리밍을 통해 모바일 장치에서 고품질 렌더링을 가능하게 하는 혁신적인 접근 방식입니다. 이는 2D 비디오처럼 동적 3DGS를 시각화하여 하드웨어 비디오 코덱을 활용할 수 있게 합니다.

- **Technical Details**: V³의 주요 기술적 혁신은 두 단계의 훈련 전략을 도입하여 저장 요구 사항을 줄이고 빠른 훈련 속도를 구현합니다. 첫 번째 단계에서는 해시 인코딩과 얕은 MLP를 사용해 모션을 학습하고, Gaussian 수를 줄이기 위해 프루닝(pruning)을 적용합니다. 두 번째 단계에서는 잔여 엔트로피 손실과 시간 손실을 통해 Gaussian 속성을 미세 조정하여 시간 연속성을 개선합니다.

- **Performance Highlights**: V³는 기존 방법들보다 우수한 성능을 보여주며, 일반적인 모바일 장치에서 고화질 렌더링과 스트리밍이 가능합니다. 사용자는 매끄럽게 스크롤하고 즉시 공유할 수 있는 전례 없는 볼륨 비디오 경험을 제공합니다.



### Exploring Fine-Grained Image-Text Alignment for Referring Remote Sensing Image Segmentation (https://arxiv.org/abs/2409.13637)
- **What's New**: 본 연구에서는 지리적 대상으로 특정 언어 표현을 기반으로 원격 탐사 이미지를 세분화하는 새로운 방법인 FIANet을 제안합니다. 기존의 RRSIS (referring remote sensing image segmentation) 방식이 간단한 정렬 방법을 사용했던 반면, FIANet은 세밀한 이미지-텍스트 정렬(fine-grained image-text alignment)을 통해 멀티모달(multi-modal) 정보를 더 잘 추출할 수 있음을 주장합니다.

- **Technical Details**: FIANet에서는 세 가지 주요 구성 요소를 사용합니다: Fine-grained Image-text Alignment Module (FIAM)은 이미지와 해당 텍스트의 특징을 동시에 활용하여 더 나은 세분화 표현(discriminative representation)을 학습합니다. Text-aware Multi-scale Enhancement Module (TMEM)은 다양한 스케일에서 이미지를 처리할 수 있도록 설계되어 있습니다. 이 두 모듈은 각각 텍스트-이미지 조정 및 멀티 스케일 조정을 통해 원격 탐사 이미지에서 다양한 지표 대상들을 효과적으로 분리하는 데 기여합니다.

- **Performance Highlights**: FIANet은 RefSegRS와 RRSIS-D라는 두 개의 공개 데이터셋에서 평가된 결과, 기존의 여러 최신 기법들에 비해 우수한 성능을 발휘했습니다. 특히, 제안된 방법은 더 세분화된 이미지-텍스트 정렬을 통해 복잡한 배경과 다양한 공간 스케일을 가진 지면 대상들을 효과적으로 식별할 수 있음을 보여주었습니다.



### FIHA: Autonomous Hallucination Evaluation in Vision-Language Models with Davidson Scene Graphs (https://arxiv.org/abs/2409.13612)
- **What's New**: 최근 대규모 비전-언어 모델(Large Vision-Language Models, LVLMs)의 발전에도 불구하고 환각 문제(hallucination)로 인한 평가의 중요성이 더욱 커지고 있습니다. 이러한 문제를 해결하기 위해 FIHA(Fine-grained Hallucination Evaluation)라는 자동 평가 프레임워크를 제안하며, 이는 LLM(대규모 언어 모델)이나 주석(annotation) 없이 다양한 환각을 평가할 수 있습니다.

- **Technical Details**: FIHA는 이미지 또는 자막을 입력으로 받아 Q&A(Quitions and Answers) 쌍을 생성합니다. 이 프레임워크는 BLIP-2를 통한 자막 생성, Fast R-CNN을 통한 특징 추출, 질문 생성을 위한 템플릿을 통합하여 LLM이나 수동 주석에 의존하지 않고 완전 자동화된 Q&A 생성을 가능하게 합니다. 또한, 질문과 답변 쌍은 Davidson Scene Graph(DSG)를 사용하여 구조적으로 조직됩니다.

- **Performance Highlights**: FIHA-v1 벤치마크를 통해 MSCOCO와 Foggy 데이터셋에서 다양한 질문을 평가하였으며, 여러 주요 LVLM 모델의 한계와 도전 과제를 강조했습니다. 이 프레임워크는 평가 과정에서 신뢰성을 높이고, 다양한 유형의 질문을 포함하여 모델의 이해를 포괄적으로 평가할 수 있도록 돕습니다.



### MaPPER: Multimodal Prior-guided Parameter Efficient Tuning for Referring Expression Comprehension (https://arxiv.org/abs/2409.13609)
Comments:
          EMNLP 2024

- **What's New**: 이 논문에서는 기존의 Referring Expression Comprehension (REC) 작업을 위한 새로운 효율적인 파라미터 전송 학습 방법인 MaPPER를 소개합니다. 이 방법은 Multimodal Prior-guided Parameter Efficient Tuning을 기반으로 하며, 구체적으로는 Dynamic Prior Adapters와 Local Convolution Adapters를 사용하여 지역적 의미를 추출하고 시각적 인식을 향상시킵니다.

- **Technical Details**: MaPPER는 Dynamic Prior Adapter (DyPA)와 Local Convolution Adapter (LoCA)를 포함합니다. DyPA는 정렬된 prior에 의해 각 토큰의 중요성을 계산해 동적으로 조정하며, LoCA는 다중 스케일의 지역적 지식을 통합하여 비주얼 트랜스포머의 표현력을 향상시킵니다. Prior-Guided Text 모듈을 도입하여 텍스트와 비주얼 정보를 융합합니다.

- **Performance Highlights**: 실험 결과, MaPPER는 RefCOCO, RefCOCO+, RefCOCOg와 같은 세 가지 벤치마크에서 이전의 SOTA 방법들보다 뛰어난 성능을 보이며, 단 1.41%의 튜닝 파라미터로 최고의 정확도를 달성하였습니다.



### Towards Child-Inclusive Clinical Video Understanding for Autism Spectrum Disorder (https://arxiv.org/abs/2409.13606)
Comments:
          5 pages, 2 figures, 2 tables

- **What's New**: 이 연구는 발달 장애 중 하나인 자폐 스펙트럼 장애(ASD)에 대한 진단 세션을 비디오, 음성, 텍스트의 세 가지 모달리티를 사용하여 분석하는 방법론을 제안합니다. 특히, 대규모 언어 모델을 reasoning agent로 활용하여 다양한 모달리티를 통합하는 방법론을 개발했습니다.

- **Technical Details**: 제안된 방법론은 대화 중 아동의 행동을 정량화하기 위한 자동 분석을 가능하게 하여 연구자와 임상가가 아동의 행동을 보다 잘 이해하고 평가할 수 있도록 돕습니다. 이 연구에서는 Activity Recognition(활동 인식)과 Abnormal Behavior Detection(비정상 행동 탐지)의 두 가지 주요 작업을 통해 성능을 평가했습니다.

- **Performance Highlights**: 모달리티별 한계에 대한 강건성을 제공하며, 단일 모달리티 환경 대비 20% 정도의 성능 향상을 보여주었습니다. 이는 자폐 아동의 진단 세션 분석에서 더 나은 결과를 도출할 수 있는 가능성을 제시합니다.



### MeLIAD: Interpretable Few-Shot Anomaly Detection with Metric Learning and Entropy-based Scoring (https://arxiv.org/abs/2409.13602)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이번 논문에서는 해석 가능한 (interpretable) 이상 탐지 (anomaly detection)를 위한 새로운 방법론인 MeLIAD를 제안합니다. MeLIAD는 기존 방법들과는 다르게 metric learning에 기반하여, 사전 분포 가정에 의존하지 않고도 해석 가능성을 제공합니다.

- **Technical Details**: MeLIAD는 훈련을 위해 몇 개의 이상 샘플만 필요하며, 데이터 증강 (augmentation) 기법을 사용하지 않으면서 본질적으로 해석 가능한 (interpretable) 방법입니다. 이 방법은 이상 사례를 식별하고 지역화하기 위한 새로운 trainable entropy-based scoring component와 metric learning 목표와 함께 이상 점수 구성 요소를 최적화하는 새로운 손실 함수 (loss function)를 도입했습니다.

- **Performance Highlights**: 다섯 개의 공개 벤치마크 데이터 세트를 활용한 실험 결과, MeLIAD는 기존 최고 성능 방법들과 비교하여 이상 탐지 및 지역화 성능에서 향상된 결과를 보여주었습니다.



### YesBut: A High-Quality Annotated Multimodal Dataset for evaluating Satire Comprehension capability of Vision-Language Models (https://arxiv.org/abs/2409.13592)
Comments:
          EMNLP 2024 Main (Long), 18 pages, 14 figures, 12 tables

- **What's New**: 이번 논문에서는 이미지에서 풍자를 탐지하고 이해하는 어려운 작업을 제안하며, 이를 평가하기 위한 고품질 데이터셋인 YesBut을 발표합니다. 이 데이터셋은 2,547개의 이미지로 구성되어 있으며, 풍자적 이미지(1,084개)와 비풍자적 이미지(1,463개)가 포함되어 있습니다.

- **Technical Details**: 우리는 (1) Satirical Image Detection (풍자적 이미지 탐지), (2) Satirical Image Understanding (풍자적 이미지 이해), (3) Satirical Image Completion (풍자적 이미지 완성)라는 세 가지 벤치마크 작업을 제안합니다. 이 작업들은 VL(Vision-Language) 모델의 일반적인 기능을 초월하여 풍자의 맥락을 이해해야 합니다.

- **Performance Highlights**: 현재 SOTA VL 모델이 YesBut 데이터셋에서 풍자적 이미지 탐지, 이해, 완성 작업에서에서도 열악한 성능을 보였습니다. 특히 제로샷 환경에서는 풍자적 이미지 탐지가 매우 어려웠습니다.



### Portrait Video Editing Empowered by Multimodal Generative Priors (https://arxiv.org/abs/2409.13591)
Comments:
          Accepted by SIGGRAPH Asia 2024. Project Page: this https URL

- **What's New**: 우리는 PortraitGen을 소개합니다. 이는 멀티모달(Multimodal) 프롬프트를 통해 일관성 있고 표현력이 뛰어난 초상 영상 편집 방법을 제공합니다. 기존의 초상 영상 편집 방식은 3D와 시간적 일관성에서 어려움을 겪어왔고, 일반적으로 렌더링 품질과 효율성이 부족했습니다.

- **Technical Details**: 본 시스템은 초상 영상 프레임을 통합된 동적 3D Gaussian 필드로 전환하여 구조적 및 시간적 일관성을 보장합니다. 또한, 새로운 Neural Gaussian Texture 메커니즘을 설계하여 복잡한 스타일 편집을 가능하게 하고 100FPS 이상의 렌더링 속도를 달성합니다. 이 방법은 대규모 2D 생성 모델에서 지식을 추출하여 멀티모달 입력을 통합합니다.

- **Performance Highlights**: 실험 결과, 우리의 방법은 시간적 일관성, 편집 효율성, 우수한 렌더링 품질을 보여주었으며, 텍스트 기반 편집, 이미지 기반 편집, 그리고 재조명(Relighting) 등 다양한 응용 분야에서 우수성을 입증했습니다.



### Region Prompt Tuning: Fine-grained Scene Text Detection Utilizing Region Text Promp (https://arxiv.org/abs/2409.13576)
- **What's New**: 본 논문에서는 scene text detection에서의 섬세한 text feature를 포착하기 위해 새로운 방법인 region prompt tuning (RPT)을 제안합니다.

- **Technical Details**: RPT 프레임워크는 general text prompt와 region text prompt를 사용하여 각각 고정된 word embedding과 섬세한 local features를 동시에 처리합니다. 특히, character-token correspondence를 위한 position embedding 공유와 bidirectional distance loss를 도입하여 region text prompt와 detection target을 정렬합니다.

- **Performance Highlights**: 실험 결과, ICDAR2015, TotalText, CTW1500 벤치마크에서 RPT 방법이 scene text detection에서 뛰어난 성능을 보여주며, 세부적인 feature를 포착하는 데 효과적임을 입증했습니다.



### Tackling fluffy clouds: field boundaries detection using time series of S2 and/or S1 imagery (https://arxiv.org/abs/2409.13568)
Comments:
          under review

- **What's New**: 이 연구는 기존의 광학 원거리 감지 방법들이 구름 영향을 받기 쉬운 문제를 해결하기 위해, Sentinel-2 (S2)와 Sentinel-1 (S1) 이미지를 이용한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 본 연구에서는 위성 이미지 시계열에 적합한 3D Vision Transformer 아키텍처를 도입하며, 메모리 효율적인 주의(attention) 메커니즘을 포함합니다. 두 가지 모델, 즉 S2 또는 S1 데이터를 독립적으로 처리하는 PTAViT3D와 두 데이터셋을 융합하여 정확성을 높이는 PTAViT3D-CA를 제시합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델들은 부분 구름(S2 및 S1 데이터 융합) 또는 밀집된 구름(S1) 하에서도 효과적으로 분야 경계를 구분할 수 있음을 입증하며, S1 기반 모델은 공간 해상도 측면에서 S2 이미지와 비교되는 성능을 제공합니다.



### Efficient Visualization of Neural Networks with Generative Models and Adversarial Perturbations (https://arxiv.org/abs/2409.13559)
Comments:
          4 pages, 3 figures

- **What's New**: 이 논문은 생성 네트워크(Generative Network)를 통한 새로운 심층 시각화(Deep Visualization) 접근 방식을 제시하며, 기존 방법들보다 개선된 성능을 보여줍니다. 이 모델은 기존에 사용되던 여러 네트워크의 수를 줄여 단일 생성기(Generator)와 판별기(Discriminator)만을 필요로 하며, 비대립적 훈련 과정을 통해 훈련됩니다.

- **Technical Details**: 이 연구의 핵심 기여는 특정 클래스 레이블(Class Labels)에 부합하는 세밀한 시각화 이미지를 생성하는 능력입니다. 모델은 레이블 유도 이미지 생성(Label-directed Image Generation)을 향상시키기 위해 고유한 스킵 연결(Skip-connection)에서 영감을 받은 블록 디자인을 통합하였습니다. 판별기는 생성된 이미지의 진위 여부를 판단하는 역할을 하며, 훈련 과정에서 실제 데이터셋을 기반으로 사전 훈련된 후 생성기를 지도합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 적대적 예제 생성 기술보다 우수한 성능을 보이며, 목표 공격(Targeted Attacks) 및 비목표 공격(Non-targeted Attacks)에서 최대 94.5%의 혼란률(Fooling Rate)을 달성하며, 최소한의 변형으로 효과적으로 DNN을 속일 수 있음을 입증하였습니다. 이를 통해 시각화 품질을 평가하는 정량적 지표로서의 혼란률의 가능성을 제시합니다.



### Trustworthy Hate Speech Detection Through Visual Augmentation (https://arxiv.org/abs/2409.13557)
- **What's New**: 새로운 hate speech detection methods(증오 발언 탐지 방법)인 TrusV-HSD를 제안하며, 이는 시각적 보강을 통해 의미 정보를 강화하고 신뢰할 수 있는 손실 함수(trustworthy loss)를 통해 불확실성을 감소시킵니다.

- **Technical Details**: TrusV-HSD는 세 가지 주요 모듈로 구성됩니다: visual cues generation module(시각적 단서 생성 모듈), hate speech detector(증오 발언 탐지기) 및 trustworthy loss(신뢰할 수 있는 손실). 이 방법은 이미지와 텍스트가 쌍이 되어있지 않은 데이터를 사용하여 시각적 단서를 생성하고, 이를 통해 의미적 학습과 멀티모달 연결성을 효과적으로 수행합니다.

- **Performance Highlights**: 실험 결과, TrusV-HSD는 기존의 방법들에 비해 뛰어난 성능 향상을 보여줍니다. 이 접근 방식은 파트너 데이터의 필요 없이 다양한 길이의 발언에 대해서도 강력한 성능을 보장합니다.



### A preliminary study on continual learning in computer vision using Kolmogorov-Arnold Networks (https://arxiv.org/abs/2409.13550)
- **What's New**: 최근 Liu et al. (2024)이 발표한 Kolmogorov-Arnold Networks (KAN)이 새로운 다층 퍼셉트론(MLP)의 대안으로 부각되고 있습니다. KAN은 MLP가 갖는 재난적 망각(catastrophic forgetting) 문제를 해결하는 데 중점을 두고 있으며, 기존 연구와는 달리 1D 데이터셋 외에도 MNIST 데이터셋을 활용하여 성능을 평가한 것이 특징입니다.

- **Technical Details**: KAN은 Kolmogorov-Arnold Theorem (KAT)을 기반으로 하며, 여러 변수 연속 함수를 단일 변수 함수의 초합(superposition)으로 표현할 수 있습니다. KAN은 가중치(weight) 대신 독립 함수(univariate functions)를 사용하여 입력과 출력 간의 관계를 모델링합니다. 이는 MLP의 고정된 활성화 함수에 비해 유연성을 제공합니다.

- **Performance Highlights**: 연구 결과, KAN의 효율적인 변형이 전통적인 MLP와 원래 KAN 구현 모두를 초월하는 성능을 보여주었습니다. 또한 하이퍼파라미터가 KAN과 MLP의 성능에 미치는 영향과 특정 학습 가능한 파라미터(바이어스(bias) 및 스케일 가중치(scale weights))의 영향을 분석하였습니다.



### FullAnno: A Data Engine for Enhancing Image Comprehension of MLLMs (https://arxiv.org/abs/2409.13540)
Comments:
          7 pages, 5 figures, 2 tables

- **What's New**: 이번 논문에서는 Multimodal Large Language Models (MLLMs)의 성능을 향상시키기 위해 FullAnno 시스템을 제안하여, 고품질의 대규모 이미지 주석을 자동으로 생성하는 방법을 소개합니다. 이 시스템은 객체의 범주와 위치, 지역 설명, 텍스트 정보 및 밀도 있는 캡션을 포함한 세부적인 이미지 주석을 생성합니다.

- **Technical Details**: FullAnno는 여러 전문 모델의 cascade annotation 과정으로 구성되며, LLM을 활용하여 고밀도 이미지 캡션을 생성합니다. 이 과정에는 객체 범주, 위치, 속성, OCR 및 지역 설명에 대한 풍부한 정보가 포함된 프롬프트를 사용하여 LLaVA-v1.5 모델을 통해 이미지 내용을 철저하게 설명하는 이미지 번역 작업이 포함됩니다. 재주석 처리된 COCO 및 Visual Genome 데이터셋에서 3배 증가한 객체 주석과 15배 긴 캡션 길이를 기록했습니다.

- **Performance Highlights**: FullAnno 시스템을 통해 재주석 처리된 데이터는 LLaVA-v1.5의 여러 벤치마크에서 성능을 현저하게 향상시켰으며, 특히 생성된 캡션의 정확도가 크게 개선되었습니다. 데이터 재주석 처리 전후의 성능 비교 실험에서 고품질 이미지 캡션이 MLLMs의 능력을 어떻게 강화하는지를 입증합니다.



### First Place Solution to the Multiple-choice Video QA Track of The Second Perception Test Challeng (https://arxiv.org/abs/2409.13538)
- **What's New**: 이번 보고서에서는 제2회 Perception Test Challenge의 Multiple-choice Video QA 트랙에서 1위를 차지한 솔루션을 제공합니다. 본 대회는 복잡한 비디오 이해 과제를 제시하며, QwenVL2 모델을 활용하여 학습 세트에서 세부 조정을 하여 성능을 극대화했습니다.

- **Technical Details**: 우리는 QwenVL2 (7B) 모델을 사용했으며, 나이브 동적 해상도 입력(Naive Dynamic Resolution)과 다중모달 로터리 포지션 인베딩(Multimodal Rotary Position Embedding, M-ROPE) 같은 최신 기술을 통합하였습니다. 추가적으로, 5%의 데이터로 검증 세트를 나누어 모델을 훈련했고, Tensor 처리에 DeepSpeed ZeRO-2와 Low-Rank Adaptation (LoRA)을 사용했습니다.

- **Performance Highlights**: 최종 앙상블 결과로 Top-1 Accuracy 0.7647을 기록하였으며, 이는 다수 투표 및 테스트 시간 증강(Test Time Augmentation)을 통해 도출된 성과입니다. 이 모든 방법들은 복잡한 비디오 이해 과제에서 모델의 성능을 개선하는 데 중요한 역할을 했습니다.



### Formula-Supervised Visual-Geometric Pre-training (https://arxiv.org/abs/2409.13535)
Comments:
          Accepted to ECCV2024

- **What's New**: 이번 연구에서는 이미지를 처리하는 모달리티(visual)와 포인트 클라우드를 처리하는 모달리티(geometric)를 통합하여 이미지 및 3D 객체 인식에서 효과를 극대화하는 새로운 방법인 Formula-Supervised Visual-Geometric Pre-training (FSVGP)을 제안합니다.

- **Technical Details**: FSVGP는 단일 트랜스포머 모델을 기반으로 하여, 수학적 공식을 사용해 자동으로 정렬된 합성 이미지와 포인트 클라우드를 생성합니다. 이를 통해 시각-기하학적 표현의 슈퍼 비전(pre-training)을 가능하게 하며, VG-FractalDB라는 데이터베이스를 통해 자기조화된 프랙탈 이미지를 생성합니다.

- **Performance Highlights**: FSVGP는 이미지 및 3D 객체 분류, 탐지 및 세분화의 6개 작업에서 기존의 VisualAtom 및 PC-FractalDB 방법보다 더 효과적인 성과를 보였습니다. 이는 FSVGP가 우수한 일반화 능력을 갖추고 있음을 시사합니다.



### Boosting Federated Domain Generalization: The Role of Advanced Pre-Trained Architectures (https://arxiv.org/abs/2409.13527)
- **What's New**: 이 연구는 Federated Domain Generalization (FDG)을 향상시키기 위해 Vision Transformers (ViT), ConvNeXt 및 Swin Transformers와 같은 고급 사전 훈련 아키텍처의 효능을 탐구합니다.

- **Technical Details**: 이 연구에서는 ImageNet-1K, ImageNet-21K, JFT-300M 및 ImageNet-22K와 같은 대규모 데이터셋을 사용하여 다양한 변형의 아키텍처를 체계적으로 평가합니다. 자가 감독(self-supervised) 및 감독(supervised) 사전 훈련 전략을 비교하여 FDG 성능에 미치는 영향을 분석하였습니다.

- **Performance Highlights**: Office-Home 및 PACS 데이터셋에 대한 평가에서, 고급 아키텍처를 사용하면 각기 다른 데이터셋에서 평균 정확도 84.46% 및 92.55%를 달성하며, 파라미터 수가 적은 특정 모델이 기존의 ResNet 모델보다 성능이 우수해졌습니다.



### Efficient and Discriminative Image Feature Extraction for Universal Image Retrieva (https://arxiv.org/abs/2409.13513)
- **What's New**: 본 연구에서는 다양한 도메인에서 강력한 의미론적 이미지 표현을 제공하는 범용 특징 추출기를 위한 효율적인 학습 프레임워크를 개발하였습니다. 이를 위해 M4D-35k라는 다중 도메인 학습 데이터셋을 구축하여 리소스 효율적인 학습을 가능하게 하였습니다.

- **Technical Details**: 연구에서는 CBIR(content-based image retrieval) 시스템의 도메인 특이성 문제를 해결하기 위해, 여러 최신 시각-의미 모델과 서브헤드가 아닌 마진 기반 메트릭 학습 손실 함수에 대한 평가를 실시하였습니다. M4D-35k 데이터셋은 328,000개의 이미지를 포함하며, 35,000개의 클래스에 걸쳐 균형잡힌 분포를 유지합니다.

- **Performance Highlights**: 제한된 계산 자원에도 불구하고, Google Universal Image Embedding Challenge에서 0.721의 mMP@5로 근사 최첨단 결과를 달성하였으며, 이는 리더보드에서 2위에 해당하며, 최상위 방법에 비해 0.7% 포인트 차이로 근접합니다. 또한 전체 파라미터 수는 32% 적고, 훈련 가능한 파라미터 수는 289배 적습니다.



### DAP-LED: Learning Degradation-Aware Priors with CLIP for Joint Low-light Enhancement and Deblurring (https://arxiv.org/abs/2409.13496)
- **What's New**: 이번 논문에서는 저조도 환경에서의 이미지 복원 및 해상도 향상을 위해, Vision-Language 모델인 CLIP을 활용한 새로운 접근 방식을 제안합니다. 제안된 DAP-LED 프레임워크는 CLIP을 통해 저조도 및 블러링 문제를 동시에 해결하며, 이러한 방식이 다양한 다운스트림 작업에 유리하다는 점을 확인하였습니다.

- **Technical Details**: DAP-LED 프레임워크는 Transformer 기반의 Joint Learning 구조로, CLIP을 통해 야간 이미지의 저하 수준을 적응적으로 학습합니다. 이 과정에서 CLIP-guided Cross-fusion Module(CCM)을 도입하여, 이미지 임베딩에서 다중 스케일 패치 단위의 저하 히트맵을 생성합니다. 이후 이 히트맵은 설계된 CLIP-enhanced Transformer Blocks(CeTBs)를 통해 통합되어 유용한 저하 정보를 입력 데이터로부터 유지합니다.

- **Performance Highlights**: 실험 결과, DAP-LED는 기존 방법들에 비해 저조도 환경에서 최첨단 성능을 달성하였습니다. 또한, 이미지 복원 효과가 뛰어나며, 깊이 추정, 세분화 및 검출과 같은 다운스트림 작업에 효과적임을 입증하였습니다.



### PLOT: Text-based Person Search with Part Slot Attention for Corresponding Part Discovery (https://arxiv.org/abs/2409.13475)
- **What's New**: 본 논문에서는 이미지 컬렉션에서 개인을 식별하기 위한 텍스트 기반(person search) 검색의 새로운 프레임워크를 제안합니다. 이 프레임워크는 슬롯 어텐션(slot attention)에 기반한 부분(part) 탐지 모듈을 활용하여 시각적(visual) 표현과 텍스트적(textual) 표현 간의 정렬을 개선합니다.

- **Technical Details**: 기존 방법들이 부분(feature) 추출과 정렬에서 어려움을 겪는 이유로는 직접적인 부분 수준(part-level) 감독(supervision)의 부족과 휴리스틱(heuristic) 특징에 의존하기 때문입니다. 제안된 방법은 명시적인 부분 대응(supervision) 없이도 독특한 부분을 자율적으로 식별하고 정렬하는 데 중점을 둡니다.

- **Performance Highlights**: 세 가지 공개 벤치마크에서 평가한 결과, 제안된 방법은 기존 방법들에 비해 상당한 성능 향상을 보였습니다.



### Robust Salient Object Detection on Compressed Images Using Convolutional Neural Networks (https://arxiv.org/abs/2409.13464)
- **What's New**: 이 논문은 압축 이미지에서의 CNN 기반 두드러진 객체 탐지(Salient Object Detection, SOD)를 체계적으로 벤치마킹하고 분석하는 데 초점을 맞췄습니다. 기존의 SOD 모델들이 압축 이미지에서 성능 병목 현상을 잘 드러내며, 특징적인 압축 이미지 특성을 통해 모델의 강건성을 향상시키기 위한 새로운 접근 방식인 Hybrid Prior Learning(HPL)을 제안합니다.

- **Technical Details**: 압축 이미지는 제한된 대역폭으로 인해 다양한 형태로 변형된 이미지를 형성합니다. 본 연구에서는 C-DUT-OMRON, C-DUTS, C-HKU-IS, C-PASCAL-S 및 C-ECSSD라는 다양한 CI SOD 데이터셋을 구축하였으며, 각 데이터셋에 대해 CNN 기반 SOD 모델의 강건성을 평가했습니다. 연구의 주요 초점은 손실 압축으로 인한 구조적 불연속성과 고주파 정보 손실을 다루는 것이었습니다.

- **Performance Highlights**: 제안된 HPL은 압축 이미지의 특징을 모방하여 압축 이미지의 강건한 특징 학습을 개선하는 효과를 보여주었습니다. 실험 결과는 압축 이미지에서의 다양한 수준의 손상에도 불구하고 경쟁력 있는 정확도를 유지하면서도, 기존 최첨단 방식에 비해 엄청난 향상을 입증하였습니다.



### Concept-Based Explanations in Computer Vision: Where Are We and Where Could We Go? (https://arxiv.org/abs/2409.13456)
- **What's New**: 본 논문은 C-XAI(Concept-based Explainable AI) 방법을 종합적으로 검토하여 이 분야의 발전과 연구 방향을 제안합니다. 특히, 개념 선택, 개념 표현 방법, 개념 제어 방식에 대해 논의하며, 지식 표현 및 학습(KR) 분야에서 영감을 얻어 새로운 기법을 제안합니다.

- **Technical Details**: C-XAI는 신경망 모델의 중간 층에서 입력을 어떻게 표현하는지를 설명하며, 사용자가 이해할 수 있는 의미론적으로 의미 있는 개념을 사용합니다. 본 논문에서는 개념의 중요성, 개념 표현 방법 및 개념 제어 메커니즘을 다루어 C-XAI 연구의 미진한 영역을 조명합니다.

- **Performance Highlights**: 기존 방법론에서 간과된 C-XAI의 중요한 도전 과제를 제시하며, 이를 해결하기 위한 기법을 제안합니다. 이러한 접근은 C-XAI 분야의 향후 발전 가능성을 크게 열어줄 것입니다.



### Towards the Discovery of Down Syndrome Brain Biomarkers Using Generative Models (https://arxiv.org/abs/2409.13437)
- **What's New**: 이번 연구에서는 인공지능(AI)을 활용하여 다운 증후군(Down syndrome, DS) 환자들의 뇌 변화를 탐지하는 새로운 접근법을 소개합니다. 특히, 알츠하이머(Alzheimer's disease, AD)로 인한 신경퇴행의 다양한 정도를 고려하여 생성 모델(generative models)을 적용한 것이 주목할 만합니다.

- **Technical Details**: 연구에서는 Variational Autoencoders (VAE), Vector Quantized Variational Autoencoders (VQ-VAE), Denoising Diffusion Implicit Models (DDIM)와 같은 최첨단 뇌 이상 탐지 모델을 사용하여 차세대 뇌 자기공명영상(MRI) 스캔 분석을 진행했습니다. 연구에 사용된 방법은 1) 전문가의 질적 평가, 2) 생성 모델의 정량적 및 정성적 재구성 충실도 연구, 3) 히스토그램 후처리(histogram post-processing)의 효과를 분석한 절제 연구, 4) 피질 아래 구조의 정량적 볼륨 분석을 포함합니다.

- **Performance Highlights**: 연구 결과, 특정 모델들이 다운 증후군의 주요 뇌 해부학적 변화를 효과적으로 탐지할 수 있음을 보여주었습니다. 예를 들어, 소뇌의 크기가 작고, 뇌실이 확대되며, 대뇌 피질이 축소되고, 알츠하이머로 인한 두정엽 변화가 나타났습니다.



### Leveraging Text Localization for Scene Text Removal via Text-aware Masked Image Modeling (https://arxiv.org/abs/2409.13431)
Comments:
          Accepted by ECCV 2024

- **What's New**: 이 논문에서는 Scene Text Removal (STR) 작업의 훈련 데이터 부족 문제를 해결하기 위해 Text-aware Masked Image Modeling (TMIM) 알고리즘을 도입했습니다. TMIM은 저비용의 텍스트 검출 레이블(예: 텍스트 바운딩 박스)을 활용하여 STR 모델의 프리트레이닝을 가능하게 합니다.

- **Technical Details**: TMIM은 두 개의 병렬 스트림으로 구성됩니다: Background Modeling (BM) 스트림은 비텍스트 배경 생성 규칙을 학습하고, Text Erasing (TE) 스트림은 STR 학습을 위한 의사 레이블을 활용합니다. 이를 통해 TMIM은 저비용의 STD 데이터셋을 사용하여 직접 STR 모델을 훈련할 수 있습니다.

- **Performance Highlights**: TMIM은 SCUT-EnsText 데이터셋에서 37.35 PSNR로 최고 성능을 기록하며, 기존의 프리트레인 방법을 능가했습니다. 또한 STD 레이블만으로도 36.62 PSNR을 기록하여 비교한 모든 완전 감독 모델을 초월했습니다.



### CVT-Occ: Cost Volume Temporal Fusion for 3D Occupancy Prediction (https://arxiv.org/abs/2409.13430)
- **What's New**: 본 논문에서는 CVT-Occ라는 새로운 방법론을 소개합니다. 이는 시간을 기준으로 복셀 간의 기하학적 유사성을 활용하여 3D 점유 예측의 정확도를 향상시키고자 합니다. 이 방법은 복셀의 시선 방향을 따라 샘플링한 점들을 통해 역사적 프레임의 기능을 통합하여 비용 볼륨 특징 맵을 구성하여 현재 볼륨의 특징을 개선합니다.

- **Technical Details**: CVT-Occ는 각 복셀에 대한 시선 방향을따라 점을 샘플링하고, 이 점들의 과거 프레임에서의 3D 위치를 확인하여 비용 볼륨 특징 맵을 구성합니다. 이 특징 맵은 현재 볼륨의 특징을 정제하여 점유 예측을 향상시키는 데 사용됩니다. 기존 방법들과 비교했을 때, CVT-Occ는 과거 관측의 시차 정보를 활용하여 계산 부하를 최소화하면서도 우수한 성능을 발휘합니다.

- **Performance Highlights**: CVT-Occ는 Occ3D-Waymo 데이터셋에서 엄격한 실험을 통해 검증되었으며, 최신 기술들과 비교했을 때 3D 점유 예측 성능에서 가장 우수한 결과를 도출했습니다. 추가적인 계산 비용이 거의 없는 상태에서 성능을 향상시키는 데 성공했습니다.



### HMD$^2$: Environment-aware Motion Generation from Single Egocentric Head-Mounted Devic (https://arxiv.org/abs/2409.13426)
- **What's New**: 본 논문은 헤드 마운트 장치(head-mounted device)에서 발생하는 시각적 SLAM을 활용하여 실제적인 풀바디(human motion) 모션을 온라인으로 생성하는 새로운 시스템인 HMD$^2$를 소개합니다. 이 시스템은 모션 재구성과 생성 사이의 균형을 맞출 수 있도록 설계되었습니다.

- **Technical Details**: HMD$^2$ 시스템은 여러 모달리티의 조건부 모션 Diffusion 모델을 사용하여 모션의 시간적 일관성을 유지하며, 자가 회귀적 인페인팅(autoregressive in-painting) 기법을 통해 최소한의 지연(0.17초)으로 온라인 모션 추론을 지원합니다. 입력 스트림에서 헤드 모션, SLAM 포인트 클라우드(point cloud), 이미지 임베딩을 추출하여 다양한 환경에서 작동할 수 있습니다.

- **Performance Highlights**: HMD$^2$는 실세계 200시간의 데이터셋인 Nymeria 데이터셋에서 평가되었으며, 단일 HMD에서의 모션 생성에 대해 최신 성능(state-of-the-art performance)을 기록했습니다. 이 시스템은 다양한 실내 및 외부 활동을 포함하여 100명 이상의 피험자로부터 수집된 데이터를 활용하였습니다.



### Occupancy-Based Dual Contouring (https://arxiv.org/abs/2409.13418)
Comments:
          Accepted to SIGGRAPH Asia (conference) 2024. Code: this https URL

- **What's New**: 본 연구에서는 옥시피던시 함수(occupancy functions)의 최첨단 성능을 제공하는 이중 윤곽화 방법(Dual Contouring Method)을 소개합니다. 이 방법은 GPU 병렬화를 극대화하도록 설계되었으며, 학습이 필요하지 않아서 몇 초 안에 계산할 수 있는 속도를 자랑합니다.

- **Technical Details**: 본 연구에서는 Manifold Dual Contouring (MDC) 기반의 Occupancy-Based Dual Contouring (ODC) 방법을 제안합니다. 네트워크 기반 함수에 인코딩된 실제 형상을 드러내기 위해, 우리는 거리 정보를 사용하지 않고 1D 포인트와 3D 포인트의 계산을 수정합니다. 추가적인 2D 포인트를 도입하여 지역 표면 법선(local surface normals)을 계산할 수 있도록 하였으며, 이를 통해 Quadric Error Function (QEF)을 사용하여 3D 포인트를 식별합니다.

- **Performance Highlights**: 여러 3D 신경 생성 모델과 3D 메쉬 데이터 세트에서의 실험 결과, 우리 방법은 이전 작업보다 뛰어난 진실성을 보여주었습니다. 계산 시 GPU 병렬화를 완벽히 활용하여 해상도가 128^3일 때도 계산 시간을 5초 이내로 유지하였습니다.



### Sine Wave Normalization for Deep Learning-Based Tumor Segmentation in CT/PET Imaging (https://arxiv.org/abs/2409.13410)
Comments:
          Report for Team WukongRT in the AutoPET III Challenge

- **What's New**: 이 보고서는 CT/PET 스캔에서 자동 종양 분할을 위한 정규화 블록을 소개하고 있습니다. 주요 혁신은 SineNormal의 도입으로, 이는 PET 데이터에 주기적인 Sine 변환을 적용하여 병변 탐지를 향상시키는 것을 목표로 합니다.

- **Technical Details**: SineNormal은 PET 이미지의 대사 활동을 개선하기 위해 특별히 설계된 모듈입니다. 복잡한 강도 패턴을 갖는 PET 이미지에 비선형 변환을 적용하여 미세한 대사 변화를 강조하고, 특히 종양 경계와 대사 이질성을 강조합니다. nnUNet ResEnc(M) 계획을 기반으로 한 네트워크 아키텍처를 사용하여, 6단계로 구성돼 점진적으로 특징 채널을 증가시킵니다. 네트워크는 3D CNN을 사용하고, SineNormalBlock을 통해 정상화된 PET 데이터가 입력됩니다.

- **Performance Highlights**: 훈련 데이터셋(n=1611)을 사용하여, 배치 사이즈 8과 패치 사이즈 112×160×128로 훈련하였습니다. 최적의 성능 모델은 1050 에폭에서 확보되었으며, 효율성을 극대화하기 위해 동적 슬라이딩 윈도우 접근방식과 테스트 시간 증강(TTA)을 구현하여 처리 속도를 향상시켰습니다.



### Evaluating the plausibility of synthetic images for improving automated endoscopic stone recognition (https://arxiv.org/abs/2409.13409)
Comments:
          8 pages, 6 figures, 1 table, conference paper

- **What's New**: 본 논문에서는 Morpho-Constitutional Analysis (MCA) 및 Endoscopic Stone Recognition (ESR)와 같은 신장 결석 진단 방법이 인공지능(AI)을 통해 개선될 수 있는 가능성을 다루며, 특히 기존의 ex-vivo 데이터셋을 증강하기 위한 확산(diffusion) 기반 방법을 제안합니다.

- **Technical Details**: 이 연구는 합성 이미지 생성 모델인 SinDDM(Single Image Denoising Diffusion Model)을 사용하여 high-resolution charge-coupled device (CCD) 이미지를 바탕으로 신장 결석의 합성 이미지를 생성하고, 이를 모델의 pre-training에 활용합니다. 이를 통해 자동화된 ESR 시스템의 정확도를 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, ImageNet에서만 pre-trained된 baseline 모델에 비해 정확도가 10% 향상되었으며, CCD 이미지만으로 훈련된 모델에 비해 표면 이미지의 경우 6%, 단면 이미지의 경우 10% 증가함을 보였습니다.



### Instruction-guided Multi-Granularity Segmentation and Captioning with Large Multimodal Mod (https://arxiv.org/abs/2409.13407)
Comments:
          Code and dataset will be released at this https URL. 7 pages, 4 figures with Supplementary Material

- **What's New**: 이번 연구에서는 Multi-Granularity Large Multimodal Model (MGLMM)을 소개하고, 이를 통해 세분화(segmentation) 및 설명(captioning) 작업을 세밀하게 조정할 수 있는 능력을 보여줍니다. 이 모델은 사용자의 지침에 따라 panoptic SegCap에서 fine-grained SegCap으로의 원활한 변환이 가능합니다.

- **Technical Details**: MGLMM은 분할(segmentation) 및 설명(captioning)을 통합하는 새로운 작업인 Multi-Granularity Segmentation and Captioning (MGSC)을 수행하며, 사용자 지침에 따라 세분화의 세밀도를 조정할 수 있습니다. 보조적으로, MGSC를 위한 벤치마크 데이터셋인 MGSCData를 구축하여, 10,000개의 이미지와 30,000개 이상의 이미지-질문 쌍을 포함하고 있습니다. 또한, Unified SegCap Data Format (USCDF)을 제안하여 각기 다른 세분화 데이터셋의 통합을 용이하게 합니다.

- **Performance Highlights**: MGLMM은 8개 이상의 하위 작업에서 뛰어난 성능을 보이며, MGSC, GCG(image captioning), 다양한 세분화 작업에서 최신 기술(state-of-the-art) 성능을 달성했습니다. 이를 통해 다중 모달 연구를 발전시킬 잠재력을 지니고 있습니다.



### Validation & Exploration of Multimodal Deep-Learning Camera-Lidar Calibration models (https://arxiv.org/abs/2409.13402)
Comments:
          8 pages, 10 figures

- **What's New**: 이 연구는 깊이 학습 아키텍처를 활용하여 다중 모달 센서 시스템의 교정을 탐색하고 평가하며 구현하는 혁신적인 방법을 제시합니다. 특히 3D LiDAR와 2D 카메라 센서 간의 실시간 정렬을 목표로 하는 센서 융합(sensor fusion) 기법을 강조합니다.

- **Technical Details**: 전통적인 정적 교정(static calibration) 방법은 번거롭고 시간이 많이 소요되므로, CNN(Conventional Neural Networks)과 기하정보 학습(geometrically informed learning)을 결합하여 이 문제를 해결하려고 합니다. 연구는 RegNet, CalibNet, LCCNet과 같은 외부 LiDAR-카메라 교정 도구의 기초 원리를 활용합니다. 각 프레임워크에 대해 소스 코드를 조정하고, 미세 조정(fine-tuning), 훈련(training), 검증(validation), 테스트(testing)를 수행하여 공정한 비교를 위한 시각적이고 측정 가능한 결과를 도출합니다.

- **Performance Highlights**: 일련의 실험을 통해 각 모델의 단점을 분석하고 개선 가능성을 탐색합니다. 검증된 모든 모델 중에서 LCCNet이 가장 우수한 결과를 나타내는 것으로 확인되었습니다.



### PointSAM: Pointly-Supervised Segment Anything Model for Remote Sensing Images (https://arxiv.org/abs/2409.13401)
Comments:
          15 pages

- **What's New**: 이 연구에서는 전통적인 방법과 달리 Segment Anything Model (SAM)을 사용하여 더 편리하면서도 도전적인 포인트 주석(point annotations)을 통해 SAM을 미세 조정하는 방법에 초점을 맞추었습니다. Pseudo-labels의 세미-자기 훈련(self-training) 접근을 통해 안정적으로 모델을 개선할 수 있도록 하였습니다.

- **Technical Details**: PointSAM은 Prototype-based Regularization (PBR)라는 메소드를 통해 원본과 타겟 모델의 특성을 정렬하고, Hungarian algorithm을 활용하여 예측 프로토타입과 타겟 프로토타입 간의 매칭을 수행하여 훈련 중 오류가 발생하는 것을 방지합니다. 추가적으로, Negative Prompt Calibration (NPC) 방식을 사용하여 마스크 예측 과정에서의 정확성을 높였습니다.

- **Performance Highlights**: NWPU VHR-10, WHU, HRSID 등 3개의 RSI 데이터셋에서 실험을 수행하여, PointSAM이 기존 SAM, SAM2 및 다른 비교 방법들에 비해 우수한 성능을 보이는 것을 확인하였습니다. PointSAM은 포인트-박스 변환기(point-to-box converter)로서도 성공적인 결과를 보여 주며, 이는 추가적인 포인트 주석 기반 작업에 확장될 수 있는 가능성을 시사합니다.



### Elite-EvGS: Learning Event-based 3D Gaussian Splatting by Distilling Event-to-Video Priors (https://arxiv.org/abs/2409.13392)
- **What's New**: 이번 논문에서는 새로운 이벤트 기반 3D Gaussian Splatting 프레임워크인 Elite-EvGS를 제안합니다. 이 프레임워크는 이벤트 카메라의 데이터 사용을 극대화하여 3D 장면을 더욱 효율적으로 재구성할 수 있도록 합니다.

- **Technical Details**: Elite-EvGS 프레임워크는 이벤트 카메라에서 발생하는 드문 이벤트 스트림을 활용하여 3DGS 최적화를 수행합니다. 이 과정에서 warm-up initialization 전략을 도입하여 E2V 모델로부터 생성된 프레임을 사용해 초기 3DGS를 최적화한 후, 이벤트 데이터를 활용해 세부 정보를 보완합니다. 또한, progressive event supervision 전략을 통해 감독을 받기 위해 필요한 이벤트의 수를 점진적으로 줄이는 방식으로 최적화를 강화합니다.

- **Performance Highlights**: 실험 결과, Elite-EvGS는 다양한 데이터셋에서 3D 장면을 재구성할 때 텍스처 및 구조적 디테일에서 우수한 성능을 보여주었습니다. 특히, 실세계 데이터에서도 다양한 어려운 조건, 예를 들어 빠른 움직임 및 저조도 환경에서도 신뢰할 수 있는 성능을 발휘했습니다.



### Feature-Centered First Order Structure Tensor Scale-Space in 2D and 3D (https://arxiv.org/abs/2409.13389)
- **What's New**: 본 연구에서는 구조 텐서 방법을 단순화하여 사용자에게 필요한 매개변수 선택을 보다 쉽게 만들었습니다. Gaussian 통합/스무딩 대신 링 필터 단계를 도입하여 파생 필터의 너비를 이미지 특징의 크기와 직접 연결합니다.

- **Technical Details**: 링 필터를 사용하여 피처 가장자리에서 중심으로 응답을 더 정확하게 전이시킵니다. 구조적 측정을 사용하여 스케일 맵의 정확성을 보정함으로써 2D 및 3D 피처 크기를 신뢰성 있게 표현할 수 있습니다.

- **Performance Highlights**: 전통적인 1차 구조 텐서 방법에 비해 우리의 방법은 훨씬 더 정확하며 최소한의 사용자 입력으로 넓은 범위의 구조적 매개변수를 추출하기 위한 기본적인 솔루션으로 제공될 수 있습니다.



### RingMo-Aerial: An Aerial Remote Sensing Foundation Model With A Affine Transformation Contrastive Learning (https://arxiv.org/abs/2409.13366)
- **What's New**: 본 논문에서는 Aerial Remote Sensing (ARS) 비전 분야를 위한 최초의 기초 모델 RingMo-Aerial을 소개합니다. 이 모델은 기존 공간 임지 원거리 센서(model)와 relevant 비교되며, 항공 독자 꿈을 위해 개선되었습니다.

- **Technical Details**: 이 모델은 Frequency-Enhanced Multi-Head Self-Attention (FE-MSA) 메커니즘과 affine transformation을 기반으로 한 Contrastive Learning (CL) 프리트레이닝 방법을 도입하여, ARS 특유의 기울어진 시점에서 작은 목표물 감지 능력을 향상시킵니다. 또한 ARS-Adapter라는 효율적인 파라미터 미세 조정 방법이 제안되어, 다양한 ARS 비전 작업에서 모델의 적응성과 효과성을 개선합니다.

- **Performance Highlights**: RingMo-Aerial은 여러 다운스트림 작업에서 SOTA (State Of The Art) 성능을 달성했으며, 이는 이 모델이 ARS 비전 작업의 성능 향상에 있어 실용적이고 효과적임을 나타냅니다.



### ID-Guard: A Universal Framework for Combating Facial Manipulation via Breaking Identification (https://arxiv.org/abs/2409.13349)
- **What's New**: 본 논문에서는 facial manipulation을 방지하기 위해 ID-Guard라는 새로운 프로ACTIVE defense framework를 제안합니다. 이 프레임워크는 단일 encoder-decoder 네트워크의 forward pass를 통해 특정 facial 이미지에 해당하는 일반적인 adversarial perturbation을 생성합니다.

- **Technical Details**: ID-Guard는 Identity Destruction Module (IDM)을 도입하여 가짜 이미지에서 식별 가능한 정보를 목표로 파괴합니다. 또한, perturbation 생성을 위한 multi-task learning 문제로서의 최적화를 통해 다양한 facial 조작에 대한 disruption을 개선합니다.

- **Performance Highlights**: ID-Guard는 여러 인기 있는 facial manipulation 알고리즘에 대한 방어 성능에서 뛰어난 결과를 보이며, 변조된 얼굴 이미지에서 식별 가능한 지역을 효과적으로 왜곡하여 관찰자가 개인의 정체성을 인식할 수 없도록 만듭니다.



### Imagine yourself: Tuning-Free Personalized Image Generation (https://arxiv.org/abs/2409.13346)
- **What's New**: 이번 연구에서는 기존 개인화된 이미지 생성 방식을 넘어서는 혁신적인 모델인 'Imagine yourself'를 소개합니다. 이 모델은 각 개별 사용자에 대한 조정 없이 모든 사용자가 공유하는 프레임워크에서 작동합니다.

- **Technical Details**: 'Imagine yourself'는 1) 이미지 다양성을 촉진하는 새로운 합성 쌍 데이터 생성 메커니즘, 2) 세 가지 텍스트 인코더와 완전하게 학습 가능한 비전 인코더를 포함하는 완전한 병렬 주의(Attention) 아키텍처, 3) 시각 품질의 경계를 점진적으로 향상시키는 새로운 거칠게부터 미세 조정까지의 다단계(Multi-stage) 방법론 등을 포함하고 있습니다.

- **Performance Highlights**: 'Imagine yourself' 모델은 개인화된 이미지 생성의 모든 측면에서 기존 SOTA(SOTA: State of the Art) 모델을 초월하여, 특히 아이덴티티 보존, 시각적 품질, 및 텍스트 정렬에서 두드러진 성능 개선을 보였습니다.



### A Novel Adaptive Fine-Tuning Algorithm for Multimodal Models: Self-Optimizing Classification and Selection of High-Quality Datasets in Remote Sensing (https://arxiv.org/abs/2409.13345)
- **What's New**: 이 논문에서는 멀티모달 대형 모델을 위한 적응형 미세 조정(Adaptive Fine-Tuning) 알고리즘을 제안합니다. 이 알고리즘은 데이터 집합을 의미 벡터 공간(Semantic Vector Space)으로 투영하는 첫 번째 단계와 MiniBatchKMeans 알고리즘을 사용한 자동 클러스터링 단계를 포함합니다.

- **Technical Details**: 데이터는 의미적 유사성이 높은 클러스터로 분류되어, 각 클러스터 내의 원본 데이터와 변형된 데이터 간의 변환 차이(Translational Difference)를 계산합니다. 이 차이는 데이터의 일반화 메트릭(Generalization Metric)으로 사용되어 고유한 일반화 잠재력을 가진 데이터를 선택하여 훈련에 활용합니다.

- **Performance Highlights**: 이 알고리즘을 사용하여 InternLM-XComposer2-VL-7B 모델을 훈련한 결과, 전체 데이터셋과 비교하여 다양한 원격 감지 지표에서 성능이 1%만 감소했습니다. 이 방법은 훈련 시간을 68.2% 단축시키면서 일반적인 능력을 크게 보존하였습니다. 모델은 UCMerced 및 AID 평가 데이터셋에서 각각 89.86 및 77.19의 점수를 기록하여 GeoChat 데이터셋보다 각각 5.43 및 5.16점 높았습니다.



### Enhancing Fruit and Vegetable Detection in Unconstrained Environment with a Novel Datas (https://arxiv.org/abs/2409.13330)
Comments:
          24 pages, 8 figures, 6 tables, Scientia Horticulturae

- **What's New**: 본 논문에서는 FRUVEG67이라는 신규 데이터셋을 소개하며, 이는 67종의 과일과 채소가 포함된 이미지로 구성되어 있습니다. 이 데이터셋은 비제한적인 환경에서 캡처된 이미지들을 포함하고 있어, 농업의 자동화 및 효율성을 높이는 데 기여할 수 있습니다.

- **Technical Details**: 반자동적으로 데이터를 주석(annotation) 달도록 설계된 준지도 학습 알고리즘(SSDA)을 사용하여 비주석 이미지에 대한 객체 바운딩 박스를 생성합니다. 감지 기술로는 YOLOv7의 앙상블 버전인 FVDNet을 제안하며, 3개의 서로 다른 그리드 구성을 사용하였고, 제안된 모델은 Jensen-Shannon Divergence(JSD)와 집중 손실(focal loss)을 결합하여 작은 객체의 감지를 개선합니다.

- **Performance Highlights**: FVDNet은 mAP(Mean Average Precision) 점수 0.78을 기록하며, 이전 YOLO 모델보다 현저한 성능 향상을 보여주었습니다. 또한 개방형 카테고리 냉장고 이미지를 사용하여 FVDNet의 효용성을 평가한 결과도 유망한 성과를 보였습니다.



### Towards Semi-supervised Dual-modal Semantic Segmentation (https://arxiv.org/abs/2409.13325)
- **What's New**: 이번 연구에서는 PD-Net이라는 병렬 이중 스트림 네트워크를 제안하여, 제한된 라벨 데이터와 많은 비라벨 데이터를 활용해 3D 포인트 클라우드와 2D 이미지를 동시에 세미-슈퍼바이즈드(semantically-supervised) 세그멘테이션을 수행할 수 있도록 합니다.

- **Technical Details**: PD-Net은 원래 스트림(original stream)과 가짜 라벨 예측 스트림(pseudo-label prediction stream)이라는 두 개의 병렬 스트림으로 구성되어 있습니다. 각 스트림은 3D 및 2D 데이터 각각에 대해 인코더-디코더 구조를 가지며, 다중 이중 모드 융합 모듈을 통해 이중 모드(feature) 특성을 융합합니다. 이 모델은 Pseudo-Label Optimization Module을 통해 생성된 가짜 라벨의 품질을 향상시킵니다.

- **Performance Highlights**: 공식 데이터 세트에서 실험한 결과, PF-Net은 기존의 세미-슈퍼바이즈드 방법보다 더 나은 성능을 보였으며, 여러 경우에서 완전 슈퍼바이즈드 방법과도 경쟁력을 가진 성과를 나타냈습니다.



### Localized Gaussians as Self-Attention Weights for Point Clouds Correspondenc (https://arxiv.org/abs/2409.13291)
- **What's New**: 이 연구에서는 Transformer 아키텍처에서 고정된 attention weights를 사용하여 point cloud matching 문제를 해결하는 새로운 접근법을 제안합니다. 본 논문은 Gaussian 함수 형태의 패턴을 attention heads에 통합하여 모델의 성능을 향상시킬 수 있는 가능성을 확인했습니다.

- **Technical Details**: 연구에서는 고정된 Gaussian 함수로 self-attention heads의 가중치를 설정하고, 두 가지 변형(하나는 고정된 분산 값 사용, 다른 하나는 학습 가능한 파라미터로 취급) 비교를 통해 성능을 평가했습니다. 잡음이 있는 데이터에 대한 성능 분석과 robustness 향상 가능성도 탐구했습니다.

- **Performance Highlights**: 상대적으로 짧아진 훈련 시간과 안정적인 최적화를 통해 모델의 성능이 향상되었습니다. Gaussian 정보를 일부 self-attention heads에 주입함으로써 최적화가 안정되고 파라미터 수가 줄어들었으며, correspondences qualit가 개선된 것을 확인했습니다.



### Adaptive Margin Global Classifier for Exemplar-Free Class-Incremental Learning (https://arxiv.org/abs/2409.13275)
- **What's New**: 본 연구는 Exemplar-free class-incremental learning (EFCIL)에서의 분산 기반의 새로운 분류 기법인 Adaptive Margin Global Classifier (AMGC)를 제안합니다. 이는 데이터 효율성을 높이고, 새로운 클래스 학습 중 모델의 편향을 줄이는 데 초점을 맞추고 있습니다.

- **Technical Details**: AMGC는 분포 기반의 전역 분류기(Distribution-Based Global Classifier, DBGC)를 기반으로 하여 데이터 불균형, 샘플링 과거의 편향 요인을 감소시키고, 고전적인 분류 손실(softmax cross entropy) 대신 적응적 마진 손실(Adaptive Margin Softmax Cross Entropy, AMarX)을 제공합니다. AMarX는 오래된 클래스의 분포 변화를 시뮬레이션하여 분류 마진을 조절합니다.

- **Performance Highlights**: AMGC는 EFCIL 설정에서 뛰어난 성능을 이루어냈으며, 기존의 방법들과 비교해도 뛰어난 이미지 분류 결과를 보였습니다. 실험 결과는 AMGC가 EFCIL의 도전적인 환경에서도 경쟁력 있는 성능을 유지하고 있음을 입증해 줍니다.



### JoyHallo: Digital human model for Mandarin (https://arxiv.org/abs/2409.13268)
- **What's New**: 이 연구에서는 Mandarin(만다린) 오디오 기반 비디오 생성에 대한 도전 과제를 다룹니다. 전체적이고 다양한 Mandarin 데이터셋(데이터 세트) 수집의 어려움을 해결하기 위해 JD Health International Inc. 직원으로부터 29시간 분량의 Mandarin 음성 비디오를 수집하여 jdh-Hallo 데이터셋을 만들었습니다.

- **Technical Details**: JoyHallo 모델은 Mandarin을 위한 중국의 wav2vec2(audio feature embedding) 모델을 활용하여 적응하였으며, 반-decoupled structure(반 분리 구조)를 제안하여 입술, 표정, 포즈 특성 간의 상관 관계를 캡처합니다. 이 구조는 교차 주의(focus attention) 메커니즘을 갖추고 있으며, 비디오 품질을 향상시키고 정보 활용 효율성을 개선합니다.

- **Performance Highlights**: 실험 결과, JoyHallo는 비디오 품질, 운동 부드러움, 주제 일관성에서 개선을 보여주었으며, 이전 모델인 Hallo와 비교해 메모리 요구량이 2.5% 감소하고 추론 시간도 14.3% 단축되었습니다. 또한 JoyHallo는 Mandarin 및 English(영어) 비디오 생성에서 모두 강력한 성능을 보여줍니다.



### T2M-X: Learning Expressive Text-to-Motion Generation from Partially Annotated Data (https://arxiv.org/abs/2409.13251)
Comments:
          10 pages, 4 figures, conference paper

- **What's New**: 본 논문에서는 T2M-X라는 새로운 두 단계의 방식으로 텍스트에서 모션을 생성하는 접근 방식을 제안합니다. 이 방법은 부분적으로 주석이 달린 데이터를 학습하여 더욱 표현력 있는 애니메이션을 생성할 수 있도록 하며, 기존의 한계인 얼굴 표정과 손 움직임이 포함된 전신 모션 데이터 생성에 초점을 맞추고 있습니다.

- **Technical Details**: T2M-X는 각각 신체, 손, 얼굴을 위한 세 가지 별도의 Vector Quantized Variational AutoEncoders (VQ-VAEs)를 학습하여 고품질 모션 출력을 보장하며, 모션 생성과 다양한 신체 부위 간의 조정을 위해 모션 일관성 손실(Motion consistency loss)을 갖춘 Multi-indexing Generative Pretrained Transformer (GPT) 모델을 활용합니다.

- **Performance Highlights**: 실험 결과, T2M-X는 기존의 모델들에 비해 양적 및 질적으로 상당한 개선을 보여주며, 데이터 세트의 한계에 대한 강건성을 입증합니다.



### Deep Generative Adversarial Network for Occlusion Removal from a Single Imag (https://arxiv.org/abs/2409.13242)
- **What's New**: 이 연구에서는 강철 울타리와 그와 유사한 장애물로 인한 영상의 가시성 저하 문제를 다룹니다. 제안된 방법은 완전 자동화된 두 단계의 신경망으로 이루어져 있으며, 첫 번째 단계에서는 장애물 분할, 두 번째 단계에서는 분할된 장애물의 배경을 복구합니다.

- **Technical Details**: 이 방법은 UNet 아키텍처를 기반으로 하는 두 단계의 완전 컨볼루셔널 네트워크를 사용합니다. 첫 번째 단계는 단일 이미지로부터 장애물 마스크를 생성하는 분할 네트워크이며, 두 번째 단계는 생성된 마스크를 이용해 누락된 부분을 채우는 인페인팅 네트워크입니다. 이들은 GAN(Generative Adversarial Network)을 활용하여 구조와 질감을 함께 생성합니다.

- **Performance Highlights**: 제안된 방법은 IITKGP_Fence 데이터셋에서 성능을 평가하여 관찰한 바와 같이 기존의 전통적인 방법들보다 더 우수한 결과를 보였습니다. 새로운 접근 방식은 여러 형태와 크기의 장애물에도 효과적으로 적용될 수 있으며, 제안된 네트워크는 의미론적으로 일관된 예측을 생성하여 인페인팅 결과의 품질을 대폭 향상시켰습니다.



### 3D-GSW: 3D Gaussian Splatting Watermark for Protecting Copyrights in Radiance Fields (https://arxiv.org/abs/2409.13222)
- **What's New**: 이 논문에서는 3D Gaussian splatting(3D-GS)의 렌더링 과정에 물리적 상징을 안전하게 통합하는 새로운 물리적 상징 방법을 제안합니다. 이 방법은 사전 훈련된 3D-GS 모델을 미세 조정하여 다양한 시점에서 출력된 이미지가 주어진 메시지를 숨길 수 있도록 설계되었습니다.

- **Technical Details**: 주요 기술로는 Frequency-Guided Densification(FGD)과 Adaptive Gradient Mask가 있습니다. FGD는 고주파 영역에서 3D Gaussian을 분할하여 더 작은 크기로 만들어 물리적 상징을 삽입할 수 있는 기회를 높입니다. DWT(Discrete Wavelet Transform)를 사용하여 주파수 도메인에서 물리적 상징을 더욱 안전하게 구현합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 물리적 상징의 용량과 강인성을 크게 향상시키는 동시에 시각적으로 감지할 수 없는 방식으로 3D Gaussian에 물리적 상징을 효과적으로 임베드합니다. 다른 방법들과 비교했을 때 최첨단 성능을 입증했습니다.



### Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models (https://arxiv.org/abs/2409.13174)
- **What's New**: 최근 다중모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 발전으로 비전-언어-행동 모델(Vision Language Action Models, VLAMs)이 로봇 조작 작업의 오픈 어휘(open-vocabulary) 상황에서 성능을 향상시키기 위한 새로운 접근 방식으로 제안되고 있습니다. 이 연구에서는 MLLMs 안전성 연구와 물리적 세계에서의 조작 작업의 특정 적용 시나리오를 통합하여 VLAMs의 물리적 위협에 대한 종합적인 평가를 제공합니다.

- **Technical Details**: 본 논문에서는 Physical Vulnerability Evaluating Pipeline (PVEP)을 제안하여 VLAMs의 물리적 강인성을 평가하고자 합니다. PVEP는 Out-of-Distribution (OOD), Typography-based Visual Prompt (VP), Adversarial Patch (AdvP) 공격과 같은 다양한 시각적 물리적 위협을 포함합니다. 평가 결과에 따라 VLAMs의 성능 변동을 분석하여 다양한 물리적 보안 위협에 대한 일반izable 분석을 제시합니다.

- **Performance Highlights**: 실험 결과에 따르면 VLAMs는 OOD 공격, Typography 기반 시각 프롬프트, 적대적 패치 공격에 따라 성능이 크게 저하되며, 이러한 변동은 사용된 MLLMs의 유형에 따라 달라지는 것으로 나타났습니다. 주요 발견은 VLAMs가 물리적 보안 위협에 취약하다는 것이며, 이 연구는 VLAMs의 시각적 모달리티 물리적 보안 평가를 위한 PVEP를 통해 현재까지 가장 포괄적인 강인성 성능 평가를 수행한 것입니다.



### Bilateral Sharpness-Aware Minimization for Flatter Minima (https://arxiv.org/abs/2409.13173)
- **What's New**: 본 연구에서는 Sharpness-Aware Minimization (SAM)의 일반화 성능을 개선하기 위해 Bilateral SAM (BSAM) 방법을 제안합니다. SAM의 Flatness Indicator Problem (FIP) 문제를 해결하며, 더 나은 Flatness Indicator (FI)를 활용해 최적화합니다.

- **Technical Details**: 이 논문은 Max-Sharpness (MaxS)와 Min-Sharpness (MinS)를 결합하여 BSAM을 만들고, 불리한 면에서 충분한 flatness를 고려하지 못하는 SAM의 한계를 극복합니다. BSAM은 손실의 훈련 손실과 이웃 지역의 최소 손실 차이를 사용하여 최적화합니다.

- **Performance Highlights**: 다양한 작업 (분류, 전이 학습, 인간 포즈 추정 및 네트워크 양자화)에서 BSAM은 기존 SAM보다 훨씬 더 나은 일반화 성능과 강인성을 제공합니다.



### Towards Zero-shot Point Cloud Anomaly Detection: A Multi-View Projection Framework (https://arxiv.org/abs/2409.13162)
- **What's New**: 이번 연구에서는 Multi-View Projection (MVP) 프레임워크를 소개하여 비지도 학습의 제한을 극복하고 이미 학습된 Vision-Language Models (VLMs)를 활용하여 포인트 클라우드의 이상 탐지(anomaly detection)를 수행합니다. MVP는 포인트 클라우드 데이터를 다중 시점 깊이 이미지로 변환하여 이미지 이상 탐지로 문제를 재구성합니다.

- **Technical Details**: MVP는 점검할 포인트 클라우드를 다중 시점 깊이 이미지로 변환한 후, 이 이미지에 대해 기존의 제로샷(image anomaly detection) 방법을 적용합니다. 프레임워크의 주요 개선사항으로는 VLM의 텍스트 프롬프트(text prompting) 및 시각적 프롬프트(visual prompting)를 통합하여 성능을 극대화하는 것입니다. 특히, VLM인 CLIP을 활용하여 MVP-PCLIP라는 개선된 방법을 제안합니다.

- **Performance Highlights**: MVP 프레임워크는 MVTec 3D-AD 및 Real3D-AD 데이터셋에 대한 광범위한 실험에서 제로샷 이상 탐지 성능이 뛰어난 것으로 나타났습니다. 자동차 플라스틱 부품 검사와 같은 실제 평가에서도 이전에 보지 못한 시나리오에 일반화될 수 있는 가능성을 보여주었습니다.



### High-Fidelity Mask-free Neural Surface Reconstruction for Virtual Reality (https://arxiv.org/abs/2409.13158)
- **What's New**: 이 논문에서는 Hi-NeuS라는 새로운 신경 임피시트 서피스 재구성 프레임워크를 제안합니다. 기존 방법은 객체 마스크를 필요로 했으나, Hi-NeuS는 다중 뷰 이미지를 통해 객체의 표면을 정확하게 복원할 수 있도록 설계되었습니다.

- **Technical Details**: Hi-NeuS는 렌더링 중 겹치는 영역을 활용하여 사용자가 원하는 객체를 간접적으로 식별합니다. 이 정보는 다중 뷰 렌더링 가중치를 사용해 signed distance functions (SDF)을 조정하는 자기 지도 방식으로 활용되며, SDF의 편향을 정규화하여 기하학적 일관성을 유지합니다.

- **Performance Highlights**: DTU 데이터셋에서 Hi-NeuS는 표면 노이즈를 약 20% 줄이고, unmasked Chamfer Distance (CD)를 약 30% 개선하여 우수한 표면 세부 사항을 달성했습니다. 또한, BlendedMVS와 휴대용 카메라 캡처에서도 Hi-NeuS의 우수성을 검증하였습니다.



### Beyond Skip Connection: Pooling and Unpooling Design for Elimination Singularities (https://arxiv.org/abs/2409.13154)
- **What's New**: 논문에서는 깊은 Convolutional Neural Networks (CNNs)의 학습에서 발생하는 다양한 문제를 해결하기 위해 'Pool Skip'이라는 새로운 구조적 모듈을 제안하고 있습니다. 이 모듈은 Max Pooling, Max Unpooling, 3x3 convolution 및 skip connection 을 전략적으로 결합하여 시도합니다.

- **Technical Details**: Pool Skip 모듈은 elimination singularities (ES)를 완화하기 위한 Weight Inertia 가설을 기반으로 하며, convolutional layers와 ReLU 함수 사이에 배치되어 작동합니다. 이 모듈은 neuron의 활동을 증가시키고, 각 층 간의 feature 전송의 무결성을 유지하도록 설계되었습니다. 또한, 이 논문에서는 Pool Skip의 affine compensation과 dimensional compensation이 역전파(backpropagation) 과정에서 gradient 변동을 최적화하여 성능 저하 문제를 해결하는 방법을 수학적으로 입증합니다.

- **Performance Highlights**: Pool Skip 모듈은 CIFAR-10, CIFAR-100과 같은 자연 이미지 분류 벤치마크, Pascal VOC 및 Cityscapes와 같은 세분화(segmentation) 작업, 그리고 BTCV 및 AMOS와 같은 의료 이미징 도전 과제에서 평가되었습니다. 이 결과들은 Pool Skip이 elimination singularities를 줄이고 모델의 일반화(generalization) 및 성능을 향상시키는 데 효과적임을 입증합니다.



### Learning Visual Information Utility with PIXER (https://arxiv.org/abs/2409.13151)
- **What's New**: 이번 논문에서는 PIXER라는 새로운 이미지 이해 프레임워크를 소개하며, 픽셀 단위의 시각적 유용성(visual utility)과 함께 신뢰성(uncertainty)을 예측하는 방법을 제안합니다. 이는 기존의 방법들이 특정 특징 알고리즘의 처리 전 시각 정보의 유용성을 측정하지 못한 점을 보완합니다.

- **Technical Details**: PIXER는 스토캐스틱 Bayesian Neural Networks(BNNs)에 대한 일반화를 통해, 픽셀 단위의 예측을 가능하게 하고, 단일 프로세스에서 확률(probability)과 불확실성(uncertainty)을 동시에 출력할 수 있도록 설계되었습니다. 이는 Monte Carlo 샘플링과 같은 비용이 큰 작업을 피하며, 다양한 어플리케이션에 맞춘 사용자 정의 가능한 featureness 정의를 제공합니다.

- **Performance Highlights**: PIXER를 시각적 오도메트리(visual odometry) 작업에 적용한 결과, 평균 31%의 RMSE 개선과 49% 더 적은 특징 사용으로 성능을 개선하였습니다. 이는 세 개의 데이터셋과 여덟 개의 특징 알고리즘에서 측정된 결과입니다.



### UniTabNet: Bridging Vision and Language Models for Enhanced Table Structure Recognition (https://arxiv.org/abs/2409.13148)
- **What's New**: UniTabNet, 새로운 table structure parsing 프레임워크를 도입하여 기존의 시각적 접근 방식을 넘어 텍스트 의미를 효과적으로 이해할 수 있도록 하였습니다. 이 모델은 'divide-and-conquer' 전략을 사용하여 테이블 셀을 분리하고, 물리적 및 논리적 디코더를 통합하여 완전한 테이블 구조를 재구성합니다.

- **Technical Details**: UniTabNet은 이미지 내의 각 테이블 셀을 분리하기 위해 이미지-텍스트(model) 프레임워크를 채택하며, 이를 통해 논리적 속성(행 및 열 범위 정보)과 물리적 속성(바운딩 박스 좌표)을 독립적으로 처리합니다. Vision Guider와 Language Guider 모듈을 통해 모델의 예측 정확도를 향상시킵니다.

- **Performance Highlights**: PubTabNet, PubTables1M, WTW, iFLYTAB과 같은 주요 테이블 구조 데이터셋에서 UniTabNet은 최신 성능을 달성하여 기존 방법에 대한 우수성을 입증하였습니다. 오픈소스로 코드도 공개될 예정입니다.



### Interpret the Predictions of Deep Networks via Re-Label Distillation (https://arxiv.org/abs/2409.13137)
Comments:
          Published by IEEE ICME 2021

- **What's New**: 본 연구에서는 딥 네트워크의 예측을 해석하기 위한 're-label distillation' 접근법을 제안합니다. 이 방법은 최신 Self-supervision 방식을 통해 입력과 예측 간의 직접적인 지도를 학습합니다.

- **Technical Details**: 이 방법은 VAE(Variational Autoencoder) 서브스페이스에 이미지를 투사하여 잠재 벡터를 무작위로 교란하여 합성 이미지를 생성합니다. 생성된 합성 이미지는 딥 네트워크가 예측한 레이블을 기준으로 둘 중 하나의 클래스에 주석을 달리 합니다. 이후 레이블링된 합성 이미지를 기반으로 선형 학생 모델을 학습합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법의 효과가 정성적 및 정량적으로 검증되었습니다. 이 're-label distillation' 접근법은 딥 네트워크 예측의 해석 가능성을 높이는 데 탁월한 성능을 보입니다.



### BGDB: Bernoulli-Gaussian Decision Block with Improved Denoising Diffusion Probabilistic Models (https://arxiv.org/abs/2409.13116)
- **What's New**: 새로운 연구에서는 단일 모델 학습에서 파생된 확률 정보를 사용하여 여러 개의 훈련 세션에 상응하는 데이터를 생성할 수 있는 가능성을 제시합니다. 이는 기존의 생성 모델의 한계를 극복하고, Central Limit Theorem (CLT)의 원리를 활용하여 보다 정확한 분류 확률을 제공합니다.

- **Technical Details**: 연구에서는 Bernoulli-Gaussian Decision Block (BGDB)라는 새로운 모듈을 제안하여, Improved Denoising Diffusion Probabilistic Models (IDDPM)을 활용하여 Bernoulli Trials의 확률 분포를 생성하고 반영합니다. 이러한 접근 방식은 분류기에서 feature reconstruction에서 logit reconstruction으로 초점을 이동시킵니다.

- **Performance Highlights**: 실험을 통해 Cityscapes, ISIC, Pascal VOC와 같은 다양한 데이터셋에서 분류와 세분화 작업에서 눈에 띄는 성능 향상을 입증하였습니다.



### Evolution and challenges of computer vision and deep learning technologies for analysing mixed construction and demolition was (https://arxiv.org/abs/2409.13112)
- **What's New**: 건설 및 철거 폐기물(C&DW)의 자동 및 신속 인식 개선을 위해 컴퓨터 비전(Computer Vision), 인공지능(AI), 로보틱스(Robotics), 사물인터넷(IoT) 기술이 통합되고 있습니다. 이 연구는 상업적 환경에서 혼합되고 고도로 오염된 C&DW의 인식 성능에 대한 부족한 연구를 보완하고 있습니다.

- **Technical Details**: 이 논문은 시드니의 C&DW 자원 회수 시설(MRF)에서의 경험을 바탕으로 하여, 혼합 C&DW 관리 시스템 개발의 도전 과제와 기회를 탐구합니다. 여러 C&DW 분석 기법을 리뷰한 결과, 딥러닝(DL) 기반의 비주얼 방법이 최적의 솔루션으로 여겨졌으며, C&DW 분석을 위한 센서 및 카메라 기술, 객체 탐지(Object Detection) 및 물질 분할(Material Segmentation)에 중점을 둔 DL 알고리즘의 발전도 분석하였습니다.

- **Performance Highlights**: C&DW 데이터셋의 생성과 관리, 기술적 및 상업적 문제들, 그리고 혼합 C&DW 분석의 연구 동향과 미래 방향성을 다루며, C&DW 관리의 효율성을 개선하기 위한 통찰을 공유합니다. 이 연구는 해당 분야의 지속적인 연구 및 개발에 중요한 기여를 할 것입니다.



### UL-VIO: Ultra-lightweight Visual-Inertial Odometry with Noise Robust Test-time Adaptation (https://arxiv.org/abs/2409.13106)
- **What's New**: 본 논문에서는 UL-VIO라는 초경량 visual-inertial odometry (VIO) 네트워크를 제안했습니다. 이 네트워크는 1M 미만의 파라미터를 가지며, 테스트 시 적응(test-time adaptation, TTA)을 통해 비주얼-관성 일관성에 기반한 성능을 유지합니다.

- **Technical Details**: UL-VIO 네트워크는 모델 압축을 통해 BatchNorm 파라미터를 포함한 저수준 인코더를 보존하면서 리소스 효율적인 TTA를 가능하게 합니다. 이 네트워크는 최신 기술보다 36배 작은 크기를 유지하면서도 KITTI 데이터셋에서 1%의 오차 증가로 비슷한 성능을 달성합니다. TTA를 위해 관성 출력 결과를 의사 레이블(pseudo labels)로 사용하여 BatchNorm 파라미터를 업데이트합니다.

- **Performance Highlights**: KITTI, EuRoC, Marulan 데이터셋에서 다양한 테스트 환경에서의 성능을 검증한 결과, 동적 도메인 변화 하에서도 네트워크는 최대 45%의 번역 RMSE(평균 18%) 감소를 달성했습니다. 이 연구는 노이즈 강건한 TTA를 VIO에 적용한 최초의 사례입니다.



### ERIC: Estimating Rainfall with Commodity Doorbell Camera for Precision Residential Irrigation (https://arxiv.org/abs/2409.13104)
Comments:
          BuildSys 2024

- **What's New**: 본 연구에서는 기존의 기상 기반 관개 시스템에 대한 개선안을 제시합니다. 특히, 이 시스템은 기성품 초인종 카메라를 활용하여 정확한 지역 강수량을 측정합니다. 이러한 접근법은 추가 하드웨어 배포 없이 저비용 솔루션을 제공합니다.

- **Technical Details**: 본 논문에서 제안한 ERIC 시스템은 가벼운 신경망 모델을 사용하여 반사 기반 비주얼 피처 및 오디오 피처를 결합해 강수량을 추정합니다. 시스템은 Raspberry Pi 4를 기반으로 하여 설계되었으며, 750시간 이상의 영상을 수집하여 데이터의 정확성을 검증합니다.

- **Performance Highlights**: ERIC 시스템은 초당 약 5mm의 정확도로 강수량을 추정하며, 월 9,112갤런의 물을 절약하여 약 28.56달러의 유틸리티 비용 절감을 달성했습니다.



### Interpretable Action Recognition on Hard to Classify Actions (https://arxiv.org/abs/2409.13091)
Comments:
          5 pages, This manuscript has been accepted at the Human-inspired Computer Vision (HCV) ECCV 2024 Workshop. arXiv admin note: text overlap with arXiv:2107.05319

- **What's New**: 이번 연구는 비디오 활동 인식 분야에서 인간과 유사한 해석 가능한 모델을 통해 비디오 내 복잡한 활동을 이해하는 방법을 탐구합니다. 기존 모델에서 3D 정보를 추가하여 향상된 정확도를 달성하였으며, 특히 'Depth relations' 추가가 성능 개선에 기여하였습니다.

- **Technical Details**: 모델은 객체 및 손의 위치와 움직임을 사용하여 활동을 인식합니다. 두 가지 주요 확장으로는: (1) 'Container'와 'NotContainer' 구분을 위한 최첨단 물체 탐지 모델을 재훈련했고, (2) 개별 물체의 깊이를 추정하기 위한 'depth estimation' 모델을 적용하여 기존의 관계를 확대했습니다.

- **Performance Highlights**: 'Depth relations'를 추가했음에도 불구하고 전체적인 성능은 주류 딥러닝 접근 방식에는 미치지 못했습니다. 실험 결과, 'Container' 탐지기는 성능을 개선하지 못했지만 깊이 관계의 추가가 유의미한 성과로 나타났습니다.



### Real-time estimation of overt attention from dynamic features of the face using deep-learning (https://arxiv.org/abs/2409.13084)
Comments:
          10 pages, 3 figures

- **What's New**: 코로나19 팬데믹으로 인해 원격 학습 환경에서 학생들의 집중도 추적이 어려워졌습니다. 본 연구에서는 웹캠을 통해 얼굴과 머리 움직임을 분석하여 학생의 집중도를 예측하는 딥 러닝 모델을 제안합니다.

- **Technical Details**: 이 연구는 MediaPipe FaceLandmarker 모델을 사용하여 얼굴 랜드마크를 실시간으로 추적하고, 10초의 윈도우를 통한 Inter-Subject Correlation (ISC)을 기반으로 학생의 집중도를 평가합니다. 이 방법은 비침습적이며, 추가적인 설정이나 참조 신호 없이 작동합니다.

- **Performance Highlights**: 모델은 알려지지 않은 데이터에서 $R^2$=0.38, 알려지지 않은 피험자들에 대해서는 $R^2$=0.26-0.30의 성능을 보였습니다. 이는 다양한 비디오와 새로운 참가자에 대해 일반화 가능성을 보여줍니다.



### Cross-Chirality Palmprint Verification: Left is Right for the Right Palmprin (https://arxiv.org/abs/2409.13056)
- **What's New**: 이 논문은 Cross-Chirality Palmprint Verification (CCPV) 프레임워크를 소개하며, 기존의 전통적인 손바닥 인식 시스템의 한계를 극복하고 있습니다. 기존 시스템들은 일반적으로 사용자로부터 좌우 손바닥을 모두 저장해야 하는 반면, CCPV는 한 손바닥만으로 검증할 수 있도록 하여 데이터 저장 공간을 절약합니다.

- **Technical Details**: CCPV 프레임워크의 핵심은 신중하게 설계된 매칭 규칙으로, 갤러리 및 쿼리 손바닥의 이미지를 뒤집고, 각 쌍 간의 평균 거리를 최종 매칭 거리로 계산합니다. 이러한 접근 방식은 매칭 분산을 효과적으로 줄이고 전반적인 시스템의 강건성을 향상시킵니다. 또한 교차 변화 손실 함수(Cross-Chirality Loss)는 왼손, 오른손, 뒤집힌 왼손, 뒤집힌 오른손의 네 가지 손바닥 변형 간의 일관성을 유지하도록 합니다.

- **Performance Highlights**: 다양한 공개 데이터셋을 사용한 광범위한 실험 결과, 제안된 CCPV 프레임워크가 우수한 성능을 보였으며, 실제 손바닥 인증 시스템에 적용 가능성이 매우 높음을 입증했습니다. 제안된 방법은 닫힌 집합(closed-set) 및 열린 집합(open-set) 설정에서 모두 높은 효과를 나타냈습니다.



### TACE: Tumor-Aware Counterfactual Explanations (https://arxiv.org/abs/2409.13045)
Comments:
          The paper has been accepted at Italian Workshop on Neural Networks (WIRN) 2024

- **What's New**: 이 연구에서는 의료 영상을 위한 신뢰할 수 있는 counterfactual 설명을 생성하는 Tumor Aware Counterfactual Explanations (TACE) 프레임워크를 제안합니다.

- **Technical Details**: TACE는 종양 특정 특징을 수정하는 데 집중하면서 전체 장기 구조를 변경하지 않도록 설계되었습니다. 이는 ROI (Region of Interest)만 수정하는 추가 단계를 포함하여 더 신뢰할 수 있는 counterfactual을 생성할 수 있게 합니다.

- **Performance Highlights**: TACE는 유방 촬영 사진(mammography) 및 뇌 MRI에서 기존의 최신 기술들을 크게 초월하는 성능을 보였습니다. 유방암의 분류 성공률이 10.69% 증가하고, 뇌 종양의 경우 98.02%의 성과 향상을 보여주었습니다.



### DNI: Dilutional Noise Initialization for Diffusion Video Editing (https://arxiv.org/abs/2409.13037)
Comments:
          17 pages, 11 figures, ECCV 2024

- **What's New**: 논문에서는 Dilutional Noise Initialization (DNI) 프레임워크를 제안하여 비구조적 편집(non-rigid editing)을 포함한 다양한 비디오 편집을 가능하게 합니다. 이 프레임워크는 비디오의 특정 영역에 대한 구조적 강직성을 완화하기 위해 추가적인 노이즈를 도입합니다.

- **Technical Details**: DNI는 초기 잠재 노이즈(latent noise) zᵢ를 입력으로 받아, 비구조적 편집이 필요한 구역에서 기존의 노이즈를 희석(dilution)하여 더 유연한 편집이 가능하게 합니다. 이러한 과정은 (1) 초기 노이즈를 시각적 브랜치와 노이즈 브랜치로 분리하고, (2) 추가적인 Gaussian 노이즈를 시각적 브랜치에 결합함으로써 수행됩니다.

- **Performance Highlights**: DNI 프레임워크는 모델에 구애받지 않고 다양한 비디오 편집 시스템에서 효과적인 편집을 가능하게 하여, 기존의 비디오 편집 벤치마크(DAVIS, TVGVE)에서도 뛰어난 성능을 입증했습니다.



### A New People-Object Interaction Dataset and NVS Benchmarks (https://arxiv.org/abs/2409.12980)
- **What's New**: 최근 인간-객체 상호작용 장면에서 새로운 People-Object Interaction Dataset(사람-객체 상호작용 데이터셋)을 소개합니다. 이 데이터셋은 다중 시점의 4K RGB-D 비디오 시퀀스로 구성되어 있어 기계 학습 연구에 유용합니다.

- **Technical Details**: 이 데이터셋은 30개의 Kinect Azure를 사용하여 촬영된 30VIEW의 RGB-D 비디오 시퀀스를 포함합니다. 각 비디오 시퀀스는 25 FPS로 1초에서 최대 19초까지 지속되며, 카메라 파라미터, 전경 마스크, SMPL 모델 및 포인트 클라우드 등이 추가로 제공됩니다.

- **Performance Highlights**: 우리는 인공지능 모델인 SOTA NVS 모델들을 평가하여 이 데이터셋에서 NVS 벤치마크를 수립했습니다. 결과는 공개적으로 접근 가능한 코드와 표준 파라미터 설정을 통해 재현 가능합니다. 연구자들은 이 데이터셋을 무료로 사용할 수 있습니다.



### Semantic Meta-Split Learning: A TinyML Scheme for Few-Shot Wireless Image Classification (https://arxiv.org/abs/2409.12978)
- **What's New**: 이 논문은 분할 학습(split-learning)과 메타 학습(meta-learning)을 통합하여 몇 개의 샷(few-shot) 무선 이미지 분류를 위한 TinyML 기반의 의미적(Semantic) 통신 프레임워크를 제안합니다. 이 시스템은 특정 작업에 필요한 정보만을 전송하는 효율적인 방법을 사용하여 에너지와 통신 효율을 높입니다.

- **Technical Details**: 제안된 Semantic-MSL은 마지막 층(output layer)의 결과만을 전송하고 학습기와 집계기 간의 평균화된 데이터만 교환하여 계산량을 줄이는 분할 학습을 기반으로 합니다. 메타 학습은 적은 양의 데이터로 빠른 훈련을 가능하게 하여 데이터 가용성을 극복합니다. 예를 들어, 손으로 쓴 글자의 이미지 데이터 세트를 사용하여 성능을 평가합니다.

- **Performance Highlights**: 시뮬레이션 결과, Semantic-MSL은 기존의 방법들에 비해 20%의 분류 정확도 향상을 달성하며, 적은 데이터 포인트로도 더 낮은 에너지 소비를 달성합니다. 또한, 불확실성 예측을 위한 변환 예측(conformal prediction) 분석을 통해 예측의 신뢰성을 평가합니다.



### Surveying You Only Look Once (YOLO) Multispectral Object Detection Advancements, Applications And Challenges (https://arxiv.org/abs/2409.12977)
- **What's New**: 이번 논문은 다스펙트럼 이미지 처리(Multispectral Imaging)와 딥러닝(Deep Learning)의 결합을 통해 다양한 분야에서의 응용 사례를 지원하는 최신 연구를 종합적으로 검토하였습니다.

- **Technical Details**: 총 400개의 논문을 고려하였으며, 그 중 200개의 논문을 세부적으로 분석하여 다스펙트럼 이미지 기술과 YOLO(You Only Look Once) 방법의 진화에 대한 권위 있는 메타 리뷰를 제공합니다. 논문에서 검토한 자료의 63%는 지상 기반 수집 방법을 사용하였고, 2020년 이후 비인간 항공 시스템(UAS)에서의 YOLO-다스펙트럼 응용도 두 배 증가했습니다. RGB(Long-Wave Infrared) 센서 융합(Fusion)가 39%로 가장 많이 사용되었습니다.

- **Performance Highlights**: YOLOv5는 다스펙트럼 응용에 가장 많이 사용되는 변형으로, 검토된 모든 YOLO 모델의 33%를 차지합니다. 다스펙트럼 YOLO 연구의 58%는 중국에서 진행되고 있으며, 저널 임팩트 팩터가 4.45로, 다른 국가의 4.36과 유사한 연구 품질을 보이고 있습니다.



### The Era of Foundation Models in Medical Imaging is Approaching : A Scoping Review of the Clinical Value of Large-Scale Generative AI Applications in Radiology (https://arxiv.org/abs/2409.12973)
Comments:
          25 pages,3 figures, 4 tables, submitted to NPJ imaging

- **What's New**: 최근 방대한 규모의 생성을 위한 AI (Generative AI)가 의료 영상 처리에서의 활용 가능성을 보여주고 있으며, 이 연구는 그 발전 현황과 주요 문제들을 체계적으로 정리하고 있습니다.

- **Technical Details**: 본 연구는 PCC 가이드라인을 따르며, PubMed, EMbase, IEEE-Xplore, Google Scholar의 네 개 데이터베이스에서 체계적인 검색을 진행하였습니다. 15개의 연구가 선정되었으며, 이들 대부분은 보고서 생성 과정의 효율성을 높이거나 환자의 이해를 돕기 위한 번역에 초점을 맞추었습니다. 연구들은 주로 LLMs (Large Language Models)를 활용하였으며, 다중 모달 모델 (multi-modal models)을 사용하는 연구는 세 건에 불과했습니다.

- **Performance Highlights**: 대부분의 연구가 GPT를 활용했으며, 의료 영상 도메인에 특화된 모델은 거의 사용되지 않았습니다. LLMs와 multi-modal 모델 모두 특정 영역에서 우수한 결과를 보였지만, 아직까지 진단 성능에서 방사선 전문의를 능가하지는 못했습니다. 이 연구는 의료 영상 분야에서의 대규모 생성 AI 활용의 현재 상태와 한계를 제시하며, 향후 의료 영상 기초 모델들이 임상 실무를 혁신적으로 변모시킬 것이라는 전망을 제공합니다.



### SoloParkour: Constrained Reinforcement Learning for Visual Locomotion from Privileged Experienc (https://arxiv.org/abs/2409.13678)
Comments:
          CoRL 2024. Project website: this https URL

- **What's New**: 이 연구는 제한된 감각 입력을 기반으로 복잡한 환경에서 기민하고 안전한 다리 로봇의 보행 능력을 배양하기 위한 새로운 방법을 제시합니다. 이를 위해 심층 픽셀에서 로봇 제어 명령까지 전방향 비주얼 정책을 훈련시키는 방법을 도입했습니다.

- **Technical Details**: 이 방법은 제한된 강화 학습(Constrained Reinforcement Learning, CRL) 문제로 공원(parkour) 학습을 형성합니다. 이를 통해 RL 에이전트는 로봇의 전체 능력을 탐색하는 동시에 위험한 행동을 방지할 수 있도록 보장합니다. 정책 훈련 과정에서 사이드 정보(privileged information)를 적용하며, 이를 통해 샘플 효율적인 오프-정책 RL 알고리즘에서 비주얼 정책으로 전이합니다.

- **Performance Highlights**: 우리 방법의 효과성을 실제 Solo-12 로봇에 배포하여 다양한 공원 기술을 수행하는 모습을 시연했습니다. 로봇은 자신의 높이보다 1.5배 큰 장애물을 극복하는 등의 성능을 보여주어 공원 실험에서 사용되던 다른 로봇에 비해 경량임에도 불구하고 우수한 능력을 발휘했습니다.



### Beyond Accuracy Optimization: Computer Vision Losses for Large Language Model Fine-Tuning (https://arxiv.org/abs/2409.13641)
Comments:
          Accepted in EMNLP 2024 Findings

- **What's New**: 이 연구는 자연어 생성(natural language generation)에서 기존의 의미 분할(semantic segmentation) 손실 함수(loss function)를 사용하여 다양한 아키텍처를 미세 조정(fine-tuning)할 수 있는 효율적이고 실용적인 솔루션을 개발했습니다.

- **Technical Details**: 이 연구에서 제안된 접근 방식은 기존의 Cross-Entropy 손실 대신 Focal 또는 Lovász와 같은 대안적인 손실 함수를 최적화하여 문항별 수학 문제(Math Word Problems) 및 질문-응답(task)에서 평균 +42%의 개선을 달성했습니다.  이는 추가 데이터나 인간 피드백(human feedback) 없이 가능했습니다.

- **Performance Highlights**: 제안하는 방법은 복잡한 학습 절차를 필요로 하지 않고 단일 학습 단계에서 Cross-Entropy 손실을 적절히 선택함으로써 성능 개선을 가능하게 하였습니다. 이를 통해 자원 소모를 줄이고, 보다 많은 사용자가 접근할 수 있는 교육 프로세스를 지원할 수 있음을 보여주었습니다.



### Improved Unet brain tumor image segmentation based on GSConv module and ECA attention mechanism (https://arxiv.org/abs/2409.13626)
Comments:
          9 pages; Accepted by CONF-CDS 2024 conference already

- **What's New**: 이번 논문에서는 뇌 종양의 의료 이미지 분할을 위한 개선된 모델을 제안합니다. 기존의 U-Net 아키텍처를 기반으로 GSConv 모듈과 ECA 주의(attention) 메커니즘을 도입하여 성능을 향상시킵니다.

- **Technical Details**: 개선된 U-Net 모델은 다중 스케일(multi-scale) 특징을 보다 효율적으로 추출하고, 중요한 채널에 유연하게 초점을 맞출 수 있어 의료 이미지 분할 작업에서 성능이 크게 개선되었습니다. 이 모델은 8번째 에폭(epoch) 이후 손실(loss) 값이 신속하게 최소점에 도달하고 점차적으로 수렴(stabilize)함을 보여주어 우수한 학습 능력 및 일반화 능력을 입증합니다.

- **Performance Highlights**: 개선된 U-Net 모델은 35번째 에폭 이후 평균 교차 비율(mIoU)이 0.8에 근접하여 안정적임을 보여주었으며, 전통적인 U-Net에 비해 분할 효과에서 분명한 장점을 입증했습니다. 특히 뇌 종양 이미지의 가장자리를 처리하는 데 있어, 보다 정확한 분할 결과를 제공합니다.



### Analyzing the Effect of $k$-Space Features in MRI Classification Models (https://arxiv.org/abs/2409.13589)
- **What's New**: 본 연구에서는 의료 이미징을 위한 설명 가능한 AI 방법론을 개발하였습니다. 이 방법은 Convolutional Neural Network (CNN)를 사용하여 MRI 이미지를 분석하고, Uniform Manifold Approximation and Projection (UMAP)을 통해 입력 임베딩을 시각화하는 과정을 포함하여 모델의 해석력을 높입니다.

- **Technical Details**: 우리의 연구는 k𝑘쪽(k-space) 데이터와 공간 영역 표현을 통합하여 MRI 분석을 진행하였습니다. Fourier 변환(FFT)을 사용하여 이미지를 k𝑘쪽 데이터로 변환하고, 세 가지 채널(공간 도메인 이미지 및 Fourier 변환된 데이터의 실제 및 허수 부품)을 처리하는 CNN 모델 아키텍처를 개발하였습니다.

- **Performance Highlights**: 분석 결과, frequency-augmented CNN 모델은 기본 이미지 도메인 CNN과 비교했을 때 false positive 비율 개선이 있었으나, Non Dementia와 Very Mild Dementia 간 구분에서 어려움을 겪었습니다. UMAP 비주얼라이제이션을 통해 클러스터의 명확한 분리가 관찰되어 모델의 학습 전략을 효과적으로 반영하였습니다.



### Data Diet: Can Trimming PET/CT Datasets Enhance Lesion Segmentation? (https://arxiv.org/abs/2409.13548)
- **What's New**: 이번 연구에서는 autoPET3 데이터 중심 트랙에서 모델 성능을 개선하는 새로운 접근 방식을 제시합니다. 기존의 상식에서는 더 큰 데이터셋이 더 나은 모델 성능으로 이어진다고 하지만, 특정 훈련 샘플을 제외함으로써 모델의 정확도를 향상시킬 수 있음을 보여주고 있습니다.

- **Technical Details**: 본 연구에 사용된 기본 모델은 DynUnet으로, PSMA-PET와 FDG-PET의 세분화 문제를 다루고 있습니다. 훈련 과정에서 손실(loss) 값을 기준으로 가장 쉬운 샘플을 훈련 데이터셋에서 제외하고, 이 과정을 통해 false negative 비율 및 dice score를 향상시킵니다. 전체적으로 1,611개의 이미지 중 597개는 PSMA-PET 이미지입니다.

- **Performance Highlights**: 3%의 쉬운 PSMA 이미지를 제외함으로써 false negative volume이 감소하고, Dice score는 개선되었습니다. QQ-plot 분석 결과 또한 false positive 예측의 수가 줄어든 것을 확인했습니다. 이 연구는 데이터 중심 접근 방식이 PSMA-PET 모델의 과도한 신뢰성을 줄여줄 수 있음을 나타냅니다.



### Physics-Informed Latent Diffusion for Multimodal Brain MRI Synthesis (https://arxiv.org/abs/2409.13532)
- **What's New**: 이 논문에서는 브레인 MRI(자기 공명 영상) 데이터를 위한 새로운 물리 기반(generative) 모델을 제안합니다. 이 모델은 기존 데이터셋에 존재하지 않는 다양한 MRI 모달리티를 합성할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: 제안된 모델은 latent diffusion model을 활용하며, 두 단계 생성 프로세스를 포함합니다. 첫 번째 단계에서는 관측되지 않은 물리적 조직 특성 맵을 생성하고, 두 번째 단계에서 물리 신호 모델을 결합하여 최종 MRI 스캔을 생성합니다. 이를 통해 물리적인 타당성을 유지하면서 이전 교육 데이터에는 없는 MR 대비를 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과, 저자들은 생성된 조직 특성의 분포가 실제 뇌 조직에서 측정된 결과와 유사함을 확인하였으며, 이는 다중 모달 의료 이미지에 대한 통합 생성 모델에 대한 향후 연구 가능성을 시사합니다.



### A Deep Learning Approach for Pixel-level Material Classification via Hyperspectral Imaging (https://arxiv.org/abs/2409.13498)
Comments:
          13 pages, 15 figures, 6 equations

- **What's New**: 이 연구는 전통적인 RGB 시스템의 한계를 극복하고, 고급 물질 특성을 요구하는 산업 분야에서 적용 가능한 하이퍼스펙트럴 (Hyperspectral) 이미징과 딥 러닝 (Deep Learning)의 결합 가능성을 평가합니다. 이 연구에서 제안된 P1CH 분류기는 경량의 CNN (Convolutional Neural Network) 아키텍처로 고해상도 물질 분류를 수행합니다.

- **Technical Details**: 하이퍼스펙트럴 이미징 기술은 900nm에서 1700nm 범위의 스펙트럼 정보를 포착하여 각 픽셀에 대한 세밀한 스펙트럼 데이터를 제공합니다. P1CH 모델은 이러한 스펙트럼 데이터를 활용하여, 99.94%의 정확도로 픽셀 단위의 물질 분류를 수행합니다. 실험 설계에는 HS 카메라, 컨베이어, 제어된 조명 장치가 포함되었으며 반 자동화된 마스크 생성 기능도 포함됩니다.

- **Performance Highlights**: P1CH 분류기의 성능 분석 결과, 색상, 크기, 형태 변형에 대한 강인성을 입증하며, 다양한 플라스틱 샘플에서 매우 높은 정확도를 기록했습니다(99.94%). 다만, 검정색 플라스틱과 같은 특정 물질에서는 여전히 도전과제가 존재합니다. 이러한 연구는 하이퍼스펙트럴 이미징을 통한 정확한 물질 분류의 새로운 가능성을 열어주는 동시에 향후 여러 산업에서의 광범위한 응용 가능성을 보여줍니다.



### A Plug-and-Play Method for Guided Multi-contrast MRI Reconstruction based on Content/Style Modeling (https://arxiv.org/abs/2409.13477)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이번 연구에서는 MRI (Magnetic Resonance Imaging)에서 여러 개의 대비를 통해 얻어진 정보를 활용하여 하위 샘플링된 대비의 재구성을 가이드하는 모듈형 2단계 접근 방식을 제안합니다. 제안된 방법은 PnP-MUNIT이라고 하며, 기존 방법의 한계를 극복하기 위해 대규모의 쌍 데이터 세트 없이도 유용하게 작동합니다.

- **Technical Details**: PnP-MUNIT는 이미지 간(content/style) 모델링을 통해 대비 독립적 및 대비 특이적 정보를 명시적으로 나타내고, 반복 재구성에 적용하는 방식으로 작동합니다. 이 연구는 unpaired 방식으로 두 대비 MRI 데이터의 콘텐츠 및 스타일을 학습하며, 이후 이를 기반으로 콘텐츠 복원 연산자를 정의합니다. 이는 최종적으로 ПnP-MUNIT이라는 모듈형 알고리즘으로 결합됩니다.

- **Performance Highlights**: 다양한 실험을 통해 PnP-MUNIT은 NYU fastMRI DICOM 데이터셋 및 자체 데이터셋에서 발생하는 여러 가지 매개변수에서 최대 32.6%의 속도 향상을 보였습니다. 진단 품질의 임상 재구성과 비교하여 PnP-MUNIT는 33.3% 더 빠른 재구성을 가능하게 하여 임상에 실질적인 유용성을 입증했습니다.



### Dermatologist-like explainable AI enhances melanoma diagnosis accuracy: eye-tracking study (https://arxiv.org/abs/2409.13476)
- **What's New**: 이 연구에서는 피부과 의사들이 인공지능(AI) 및 설명 가능한 인공지능(XAI) 도구를 사용하는 방식을 객관적으로 평가할 필요성을 강조하고 있습니다.

- **Technical Details**: 76명의 피부과 의사가 16개의 피부 병변 이미지를 XAI 시스템을 사용하여 진단하였으며, 이 시스템은 도메인별 설명을 제공합니다. 연구에서는 Eye-tracking 기술을 활용하여 의사들의 상호작용을 평가하였습니다. 표준 AI 시스템과 비교하여 진단 성능을 분석하였습니다.

- **Performance Highlights**: XAI 시스템이 표준 AI에 비해 조화로운 진단 정확도를 2.8% 향상시키는 것으로 나타났습니다. 또한, AI/XAI 시스템 및 복잡한 병변과의 진단 불일치는 인지적 부담(cognitive load)을 증가시키는 것으로 나타났습니다.



### Classification of 4 types of White blood cell images (https://arxiv.org/abs/2409.13442)
- **What's New**: 이 논문은 백혈구(white blood cells)의 자동 분류를 위한 새로운 프레임워크를 제안하고 있습니다. 기존의 CNN(pre-train models) 모델을 활용하여 자동화된 분류 시스템을 구현하였습니다.

- **Technical Details**: 백혈구는 주로 단핵구(monocytes), 림프구(lymphocytes), 호산구(eosinophils), 호중구(neutrophils) 등 네 가지로 분류됩니다. 본 연구에서는 ResNet-50, InceptionV3, VGG16, MobileNetV2와 같은 여러 가지 CNN(pre-train) 모델을 사용하였으며, 카글(Kaggle) 데이터셋을 통해 이미지를 분석하였습니다.

- **Performance Highlights**: 이 연구를 통해 우리는 각각 99.57%와 98.67%의 정확도를 달성하였으며, 이는 기존 문헌에서 보고된 결과들과 비교해도 경쟁력 있는 성과입니다.



### Longitudinal Segmentation of MS Lesions via Temporal Difference Weighting (https://arxiv.org/abs/2409.13416)
Comments:
          Accepted at MICCAI 2024 LDTM

- **What's New**: 본 논문은 다발성 경화증(Multiple Sclerosis, MS) 병변의 정확한 분할(segmentation)을 위해 longitudinal MRI 스캔에서 시간 포인트 간의 차이를 명시적으로 통합하는 새로운 방법론인 Difference Weighting Block을 제안합니다.

- **Technical Details**: 제안된 방법은 두 개의 시간 포인트에서 특징을 통합하여 병변 분할을 수행하며, 이전 스캔과 현재 스캔 간의 차이를 강조합니다. 이는 latent space 내에서 각 이미지의 차이를 계산하고, Attention Map을 통해 중요도를 조정을 하여 두 이미지의 정보를 통합합니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 두 개의 데이터 셋에서 기존의 단일 시간 포인트 모델 및 최신 기술을 초월하여 우수한 성능을 보여주었으며, 픽셀 기반 Dice 점수와 병변 기반 F1 점수 모두에서 향상된 결과를 기록했습니다.



### MCICSAM: Monte Carlo-guided Interpolation Consistency Segment Anything Model for Semi-Supervised Prostate Zone Segmentation (https://arxiv.org/abs/2409.13371)
Comments:
          13 pages, 5 figures

- **What's New**: 본 논문에서는 종래의 Segment Anything Model (SAM)을 의료 영상에 적용하기 위한 새로운 방법론인 MCICSAM을 제안합니다. 이 모델은 의료 이미지 주석을 위한 데이터 부족 문제를 해결하고, SAM의 강력한 기능 추출 능력을 최대한 활용합니다.

- **Technical Details**: MCICSAM은 저계수 적응(LoRA) 및 몬테카를로 유도 보간 일관성(MCIC)이라는 반지도 학습 기법을 활용합니다. 이 방법을 통해, 비주석 데이터에 대해 입력 데이터의 두 가지 보간 변환을 수행하고, 예측의 일관성을 유지하도록 모델을 강제합니다. 이러한 일관성 제약은 주석이 없는 데이터 분포에 잘 맞도록 모델을 적응시킵니다.

- **Performance Highlights**: MCICSAM은 Dice 지표에서 79.38%와 89.95%의 성능을 보였으며, 전이 영역(transition zone)에서는 HD95 값이 각각 3.12와 2.27로 개선되었습니다. 또한, MCICSAM은 뛰어난 일반화 능력을 갖추고 있어 프로타타 이미지 세분화 분야에 새로운 가능성을 제시할 것으로 기대됩니다.



### V-Hands: Touchscreen-based Hand Tracking for Remote Whiteboard Interaction (https://arxiv.org/abs/2409.13347)
- **What's New**: 본 논문에서는 capacitive 비디오 프레임에서 양손의 3D 포즈를 실시간으로 정확하게 추적하는 새로운 방법을 제안합니다. 기존의 방법은 손 제스처를 캡처하기 위한 복잡한 장치나, capacitive 이미지를 통한 손 포즈 추적에 한계를 보였습니다.

- **Technical Details**: 이 방법은 깊은 신경망(deep neural network)을 활용하여 손을 식별하고 손 관절 위치를 추론 후, 이 정보를 사용하여 제약 조건을 갖춘 역기구학적 솔버(constrained inverse kinematic solver)를 통해 3D 손 포즈를 복원합니다. 또한 향상된 품질의 손-스크린 상호작용 데이터를 캡처하기 위한 장치 구성을 설계하였습니다.

- **Performance Highlights**: 우리의 방법은 capacitive 프레임에 대한 3D 손 추적의 정확성과 안정성을 개선하면서도 원격 통신을 위한 소형 장치 구성을 유지합니다. 실험 결과, 사용자들이 상호작용 콘텐츠에 더욱 집중할 수 있도록 하여 상호작용 효율성이 향상되었음을 보여주었습니다.



### SLaVA-CXR: Small Language and Vision Assistant for Chest X-ray Report Automation (https://arxiv.org/abs/2409.13321)
- **What's New**: 대규모 언어 모델(LLMs)의 성공에 영감을 받아 의료 분야에서 임상 의사들을 지원하기 위한 LLM 개발에 대한 연구가 증가하고 있습니다. 그러나 상용 모델을 사용하는 것에는 개인정보 보호 문제가 있으며, 오픈 소스 모델은 많은 계산 자원을 필요로 합니다. 이를 해결하기 위해 우리는 가슴 엑스레이(CXR) 보고서 자동화를 위한 오픈 소스 소형 언어 및 비전 도우미(SLaVA-CXR)를 제안합니다.

- **Technical Details**: 제안한 SLaVA-CXR는 Re3Training 방법론을 통해 훈련됩니다. 이 방법론은 방사선 의사의 인지 발달을 모사하며, Recognition(인식), Reasoning(추론), Reporting(보고) 세 가지 단계로 구성됩니다. 또한 RADEX라는 데이터 합성 방법을 통해 고품질의 다양한 훈련 데이터를 합성하여 개인정보 보호 규정을 준수합니다.

- **Performance Highlights**: 2.7B 백본을 기반으로 구축된 SLaVA-CXR는 기존의 대형 모델에 비해 6배 빠른 추론 효율을 달성하는 동시에 CXR 보고서 생성 및 요약 작업에서 최고의 성능을 보입니다.



### Time Distributed Deep Learning models for Purely Exogenous Forecasting. Application to Water Table Depth Prediction using Weather Image Time Series (https://arxiv.org/abs/2409.13284)
- **What's New**: 이번 연구에서는 Grana-Maira 유역의 수위를 예측하기 위해 외부 기상 정보(시간에 따른 이미지 시계열)를 사용하는 두 가지 딥러닝 모델(TDC-LSTM 및 TDC-UnPWaveNet)을 제안했습니다. 특히, 이전의 지표 데이터에 비해 국소적이고 공간적으로 분포된 데이터를 활용했습니다.

- **Technical Details**: 제안된 모델은 Time Distributed Convolutional Neural Network (TDC)와 함께 LSTM 계층으로 구성된 TDC-LSTM 모델과 새로운 WaveNet 아키텍처를 응용한 TDC-UnPWaveNet 모델로 나뉩니다. TDC-UnPWaveNet은 Channel Distributed layer를 사용하여 입력 채널 각각에 대해 동일한 연산을 적용하여 시퀀스 길이를 조정합니다.

- **Performance Highlights**: TDC-LSTM과 TDC-UnPWaveNet 모델 모두 뛰어난 예측 결과를 보였으며, TDC-LSTM은 편향을 줄이는 데 집중한 반면, TDC-UnPWaveNet은 시간적 동역학을 극대화하며 상관관계와 KGE를 최적화하는 데 중점을 두었습니다.



### Understanding Stain Separation Improves Cross-Scanner Adenocarcinoma Segmentation with Joint Multi-Task Learning (https://arxiv.org/abs/2409.13246)
- **What's New**: COSAS(과 조직 및 스캐너 간 분할) 도전 과제는 진단 도구를 위해 디지털 병리학의 정확성을 높이기 위한 염색 분리를 활용한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 우리는 다중 과제 학습 구조 내에서 염색 행렬과 밀도를 분리하는 비지도 학습을 사용하여 다양한 스캐너 간 색상 변동성을 처리하고 재현성을 극대화합니다. 모델은 U-net 구조를 사용하여 아데노시암종 분할을 수행하며, 여러 염색 증강 기법과 결합하여 훈련되었습니다.

- **Performance Highlights**: 우리의 모델은 4겹 분할된 교차검증에서 평균 COSAS 메트릭 0.846을 기록했으며, Dice 점수는 0.887, IoU 점수는 0.805에 달합니다. 최종 평가에서는 COSAS 점수가 0.792로 안정화되었습니다.



### Multiscale Encoder and Omni-Dimensional Dynamic Convolution Enrichment in nnU-Net for Brain Tumor Segmentation (https://arxiv.org/abs/2409.13229)
Comments:
          9 pages, 3 figures. Accepted at MICCAI 2023, to be published in Springer LNCS. GitHub: this https URL

- **What's New**: 본 연구에서는 수정된 nnU-Net 아키텍처를 활용한 새로운 뇌종양 세분화(segmentation) 알고리즘을 소개합니다. 이 알고리즘은 옴니차원 동적 합성곱(Omni-dimensional Dynamic Convolution) 레이어를 통합하여 개선된 기능 표현력을 제공합니다.

- **Technical Details**: 제안된 알고리즘은 nnU-Net 아키텍처를 수정한 것으로, 두 개의 인코더와 3D 이미지에 적응된 ODConv3D 레이어로 구성되어 있습니다. ODConv는 혁신적인 다차원 주의 메커니즘(multi-dimensional attention mechanism)을 사용하여 네 개의 뚜렷한 주의(attention) 형태를 획득합니다.

- **Performance Highlights**: 제안한 모델은 BraTS-2023 챌린지의 다양한 데이터셋에서 nnU-Net 아키텍처의 성능을 크게 개선했습니다. 특히, BraTS Africa 데이터셋의 검증 과정에서 우수한 정확도를 기록했습니다.



### Deep Learning based Optical Image Super-Resolution via Generative Diffusion Models for Layerwise in-situ LPBF Monitoring (https://arxiv.org/abs/2409.13171)
- **What's New**: 이번 연구에서는 저비용 저해상도 이미지를 고해상도 이미지와 연결하기 위해 조건부 잠재 확률적 확산 모델을 활용하여 고해상도 모니터링을 비용 효율적으로 수행할 수 있는 방법을 제안합니다.

- **Technical Details**: 이 연구에서는 Denoising Diffusion Probabilistic Model (DDPM)을 적용하여 저해상도 이미지를 고해상도 이미지로 변환합니다. 모델은 Gaussian 분포를 활용하여 샘플을 반복적으로 변형하는 방식으로 훈련됩니다.

- **Performance Highlights**: 모델의 성능은 PSNR 및 SSIM 지표를 통해 평가되었으며, 출력된 고해상도 이미지의 품질은 기존 저해상도 이미지를 기반으로 하여 개선되었습니다.



### GASA-UNet: Global Axial Self-Attention U-Net for 3D Medical Image Segmentation (https://arxiv.org/abs/2409.13146)
- **What's New**: 이번 논문에서는 GASA-UNet라는 새롭고 개선된 U-Net 모델을 소개하였으며, 이는 Global Axial Self-Attention (GASA) 블록을 통합하여 의료 이미지 내 여러 기관을 정확하게 분할하고 병리 조직을 분별하는 데 도움을 줍니다.

- **Technical Details**: GASA 블록은 이미지 데이터를 3D 개체로 처리하고, 각 2D 평면은 다양한 해부학적 단면을 나타냅니다. 이 모델은 Multi-Head Self-Attention (MHSA) 메커니즘을 활용하여 추출된 1D 패치 간의 연결을 용이하게 하며, Positional Embeddings (PE)를 통해 voxel 특성을 공간적 맥락으로 풍부하게 합니다.

- **Performance Highlights**: GASA-UNet은 BTCV, AMOS, KiTS23 세 가지 벤치마크 데이터셋에서 Dice 점수와 Normalized Surface Dice (NSD)의 향상을 통해 특히 작은 해부학적 구조에 대한 분할 성능이 향상된 것으로 나타났습니다.



### Score-Based Multibeam Point Cloud Denoising (https://arxiv.org/abs/2409.13143)
Comments:
          Accepted to 2024 IEEE OES AUV Symposium

- **What's New**: 본 논문에서는 멀티빔 에코 사운더(MBES) 데이터를 위한 새로운 노이즈 제거 및 이상값 탐지 네트워크를 제안합니다. 기존의 방법들과 비교하여 우수한 성능을 보이며, 기존 MBES 표준 작업 흐름에 쉽게 통합될 수 있습니다.

- **Technical Details**: 제안된 네트워크는 포인트 클라우드 노이즈 제거 커뮤니티의 아이디어를 바탕으로 하고, 스코어 기반(point cloud denoising) 방법을 적용하여 MBES 포인트 클라우드의 노이즈를 제거합니다. 이 연구에서는 노이즈가 있는 MBES 포인트 클라우드를 입력으로 받아 청정한 포인트 클라우드를 복원하는 것을 목표로 합니다. 네트워크는 로그 확률 함수의 그래디언트를 사용하여 노이즈가 있는 포인트를 청정 표면에 가깝게 조정합니다.

- **Performance Highlights**: 제안된 방법은 기존의 클래식 방법에 비해 더 나은 성능을 보였으며, 고전적인 방법에 비해 노이즈 제거 및 이상값 탐지에서 더 효율적으로 작용합니다. 코드와 사전 훈련된 모델이 공개되어 향후 연구를 위한 접근성을 제공합니다.



### Federated Learning with Label-Masking Distillation (https://arxiv.org/abs/2409.13136)
Comments:
          Accepted by ACM MM 2023

- **What's New**: 이번 논문은 Federated Learning(FL)에서 label distribution skew 문제를 해결하기 위해 새로운 접근 방식인 Label-Masking Distillation(FedLMD)를 제안합니다. 기존의 방법들은 다수의 label에 의존하여 최적화에서 부정확한 결과를 초래하는 경우가 많았습니다. FedLMD는 각 클라이언트의 다양한 label 배포를 이해하고, 소수 label 지식을 보존하는데 중점을 둡니다.

- **Technical Details**: FedLMD는 majority label과 minority label을 분류하여 majority label의 예측을 마스크하여 클라이언트 모델이 minority label 지식에 집중하도록 합니다. 이를 통해 클라이언트 모델은 로컬 데이터를 통해 majority label 지식을 배우며, global 모델로부터는 masked minority label 지식을 학습합니다. 추가로, FedLMD-Tf라는 변형을 통해 teacher 모델이 필요 없는 경량화된 구조를 제안하였습니다.

- **Performance Highlights**: 제안된 FedLMD는 분류 정확성과 수렴 속도에서 기존의 최첨단 방법들보다 우수한 성능을 보여주었습니다. 또한 FedLMD-Tf는 이전의 경량화된 Federated Learning 방법을 뛰어넘는 결과를 보였습니다.



### Personalized 2D Binary Patient Codes of Tissue Images and Immunogenomic Data Through Multimodal Self-Supervised Fusion (https://arxiv.org/abs/2409.13115)
- **What's New**: 이번 연구는 MarbliX라는 새로운 멀티모달 프레임워크를 소개합니다. 이 프레임워크는 병리 이미지와 면역유전체 시퀀싱 데이터를 결합하여, 환자 정보를 압축된 이진 코드인 '모노그램'으로 만들어냅니다.

- **Technical Details**: MarbliX는 심층 학습(deep learning) 기법을 활용하여 병리학적 전체 슬라이드 이미지(whole slide images, WSIs)와 면역 세포 시퀀싱 데이터를 통합합니다. 이러한 통합은 두 가지의 데이터 모달리티를 아우르는 고차원의 정보를 다루기 위한 효율적인 방법으로 작동합니다.  시스템은 TCGA(The Cancer Genome Atlas)로부터 수집된 데이터셋을 통해 평가되었으며, 이는 535개의 폐 선암 및 510개의 폐 편평세포암(LUSC) 사례를 포함합니다.

- **Performance Highlights**: MarbliX의 실험 결과, 정밀한 진단을 가능하게 하고 평가의 변동성을 줄이며 맞춤 치료 옵션을 확대할 수 있는 잠재력을 지니고 있음을 보여주었습니다. 특히 폐와 신장 암에 대한 데이터 세트를 통해 이 시스템의 정확성과 효율성이 입증되었습니다.



### DenoMamba: A fused state-space model for low-dose CT denoising (https://arxiv.org/abs/2409.13094)
- **What's New**: 이번 연구에서는 DenoMamba라는 새로운 저선량 컴퓨터 단층 촬영(DLCT) 이미지 잡음 제거 방법을 소개합니다. 이 방법은 상태-공간 모델링(SSM)을 활용하여 의학 이미지에서 단기 및 장기 컨텍스트를 효율적으로 캡처합니다.

- **Technical Details**: DenoMamba는 인코더-디코더 구조를 가진 모래시계 아키텍처를 따르며, 공간 SSM 모듈과 새로운 채널 SSM 모듈을 통해 각각의 컨텍스트를 인코딩합니다. 두 모듈의 특징 맵은 저수준 입력 특징과 함께 합쳐집니다. DenoMamba는 전통적인 CNN 모델과 유사한 모델 복잡도로 고해상도 CT 이미지의 복잡한 특성을 효율적으로 처리합니다.

- **Performance Highlights**: DenoMamba는 25% 및 10%의 방사선 용량 감소로 수집된 LDCT 데이터셋에 대한 포괄적인 실험을 통해, 평균 PSNR(1.4dB), SSIM(1.1%), RMSE(1.6%)의 성능 향상을 보여주며 기존 최첨단 방법들보다 우수한 이미지 품질을 제공합니다.



### Embedding Geometries of Contrastive Language-Image Pre-Training (https://arxiv.org/abs/2409.13079)
Comments:
          ECCV 2024 - Beyond Euclidean Workshop

- **What's New**: 이 논문은 CLIP의 전통적인 설계 방식인 L2 정규화와 코사인 유사성의 로그 확률을 재검토하고, 더 직관적인 유클리드 기하학을 기반으로 한 Euclidean CLIP (EuCLIP)을 제안합니다. EuCLIP은 CLIP의 성능을 일치시키거나 초과하며, 더 복잡한 하이퍼볼릭 대안들보다도 계층적 관계를 잘 지원합니다.

- **Technical Details**: 이 연구에서는 다양한 임베딩 기하학을 체계적으로 테스트했습니다. 여기에는 코사인 유사성(Cosine Similarity)을 사용하는 ELIP와 음의 거리 제곱 로그릿(Negative Distance Squared Logit)이 포함됩니다. 실험 결과, 시각 및 텍스트 변환기(Vision and Text Transformers)의 최종 레이어 정규화(LayerNorm)는 성능을 저하시키며, Euclidean 기하학 및 하이퍼볼릭 기하학 모두에서 EuCLIP이 CLIP와 유사한 또는 우수한 성능을 보여주었습니다.

- **Performance Highlights**: EuCLIP은 기존 CLIP 모델과 비교하여 제로 샷 이미지 분류(zero-shot image classification) 및 검색(retrieval) 능력을 유지하면서, 더 복잡한 모델인 MERU와 동등한 또는 더 나은 성능을 달성했습니다.



### What does guidance do? A fine-grained analysis in a simple setting (https://arxiv.org/abs/2409.13074)
- **What's New**: 이 논문에서는 diffusion 모델에 대한 가이드(guide)의 개념이 실제로 의도된 tilted distribution에서 샘플링하는 데 실패한다는 것을 엄격히 증명합니다. 일반적으로 사용되는 가이드는 이론적으로 왜 왜곡된 생성물(distorted generations)을 만들어내는지를 설명합니다.

- **Technical Details**: 저자들은 가이드의 동역학(dynamics)을 두 가지 경우(1) 제한된 지지(mixture of compactly supported distributions) 및 (2) 가우시안 혼합(mixture of Gaussians)에 대해 정교하게 특징지었습니다. 연구 결과, 가이드 파라미터가 증가함에 따라 모델은 조건부 분포의 경계에서 더 많이 샘플링하게 됩니다. 또한 스코어 추정 오류(score estimation error)가 비제로인 경우, 충분히 큰 가이드는 지원 영역(support)에서 멀어지는 샘플링을 초래합니다.

- **Performance Highlights**: 본 연구는 합성 데이터셋(synthetic settings)에서 이론적 통찰을 검증했으며, 이를 통해 실무 배치(deployment)에서 유용한 처방을 제공하는 방법을 제시합니다. 특히, 가이드 파라미터가 증가함에 따라 생성된 샘플의 다양성이 감소하고, 생성물은 조건부 분포의 극단적 포인트로 수렴하는 경향이 나타납니다.



### MGSO: Monocular Real-time Photometric SLAM with Efficient 3D Gaussian Splatting (https://arxiv.org/abs/2409.13055)
Comments:
          Paper Contribution to the ICRA 2025 Conference. Currently being reviewed

- **What's New**: 이 논문에서는 Monocular GSO (MGSO)라는 새로운 실시간 SLAM 시스템을 소개합니다. MGSO는 포토메트릭 SLAM과 3D Gaussian Splatting (3DGS)를 통합하여 최초의 풍부한 3D 맵 생성을 시도하며, 단일 모노큘러 카메라로도 동작합니다.

- **Technical Details**: MGSO는 포토메트릭 SLAM을 사용하여 3DGS 초기화를 수행하고, 이를 통해 실시간 밀집 3D 맵을 생성합니다. 기존의 SLAM 시스템들은 각각의 요구에 대해 최적화를 이루지 못했으나, MGSO는 Dense SLAM과 SLAM 모듈을 위해 효율적인 밀집 포인트 클라우드를 생성하여 성능 향상을 이룹니다.

- **Performance Highlights**: MGSO는 Replica, TUM-RGBD, EuRoC 데이터셋에서 실험을 통해 경쟁 시스템들을 초월하는 성능을 보였으며, 랩탑 하드웨어에서도 뛰어난 성능을 유지합니다. 이를 통해 로봇 공학, 증강 현실(A/R) 등 다양한 실시간 애플리케이션에서 실용적인 솔루션이 될 수 있음을 입증했습니다.



### DiffSSD: A Diffusion-Based Dataset For Speech Forensics (https://arxiv.org/abs/2409.13049)
Comments:
          Submitted to IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2025

- **What's New**: 본 논문은 새로운 Diffusion-Based Synthetic Speech Dataset (DiffSSD)을 제안하며, 약 200시간 분량의 라벨이 붙은 음성을 포함하고 있습니다. 이 데이터셋은 8개의 오픈소스와 2개의 상업적 음성 생성기로부터 생성된 합성 음성을 포함하고 있어 최신의 고품질 음성 복제기에서 생성된 음성을 탐지하기 위한 중요한 기초 자료로 사용될 수 있습니다.

- **Technical Details**: DiffSSD 데이터셋은 합성 음성을 학습하기 위한 훈련, 검증 및 테스트 세트로 나누어져 있으며, 각 세트는 고정된 화자 정체성과 내용을 기준으로 합성 음성의 변별성을 극대화하고 있습니다. 기존 탐지 방법들인 GMM, ResNet, Transformer 네트워크를 사용하여 합성 음성을 평가하였습니다.

- **Performance Highlights**: 기존의 합성 음성 탐지기는 전통적인 생성기에서 생성된 음성을 기반으로 학습되어 최근의 diffusion 기반 생성기에서의 탐지 성능이 저하되는 경향이 있습니다. 실험 결과, 제안된 DiffSSD 데이터셋의 중요성이 강조되며, 이 데이터셋을 통해 detection 알고리즘의 성능을 개선할 수 있는 가능성을 보여주었습니다.



### AutoPET III Challenge: PET/CT Semantic Segmentation (https://arxiv.org/abs/2409.13006)
- **What's New**: 이번 연구에서는 AutoPET III 챌린지를 위해 PET/CT 이미지에서 병변(lesion)을 분할(segmentation)하는 두 단계의 딥러닝(Deep Learning) 기반 접근법을 구현하였습니다.

- **Technical Details**: 첫 번째 단계에서는 DynUNet 모델을 사용하여 거칠게(segmentation) 넓은 관심 영역을 식별하였고, 두 번째 단계에서는 SwinUNETR, SegResNet, UNet 모델의 앙상블(ensemble)을 사용하여 이 분할을 정교화 하였습니다. 이미지의 공통 해상도로 재샘플링(resampling)하고 정규화(normalization)하는 전처리 과정이 있었으며, 모델 일반화를 보강하기 위해 아핀 변환(affine transformations)과 강도 조정(intensity adjustments)과 같은 데이터 증강(Data Augmentation) 기법이 적용되었습니다.

- **Performance Highlights**: 데이터셋은 80% 훈련(training)과 20% 검증(validation)으로 나누어졌으며, 건강한 사례는 제외되었습니다. 이 방법은 다단계(segmentation)와 모델 앙상블을 활용하여 정확한 병변 분할을 달성하고, 강인함(robustness)과 전반적인 성능을 향상시키는 것을 목표로 합니다.



### Across-Game Engagement Modelling via Few-Shot Learning (https://arxiv.org/abs/2409.13002)
Comments:
          17 pages, accepted for publication at ECCV 2024 CV2 Workshop

- **What's New**: 이번 논문에서는 비디오 게임에서의 사용자 경험 모델링을 위한 새로운 프레임워크를 소개합니다. 이는 여러 도메인에서 사용자 경험을 효과적으로 모델링할 수 있도록 Few-Shot Learning (FSL) 기술을 적용합니다.

- **Technical Details**: 제안된 프레임워크는 각 게임을 별도의 도메인으로 분류하고, 교차 도메인 문제를 다루기 위해 도메인별 클래스를 분해합니다. 이를 통해 FSL을 이용해 제한된 데이터로 효과적으로 학습할 수 있습니다. GameVibe Few-Shot (GVFS) 데이터셋을 이용하여, 30개의 1인칭 슈팅 게임에서의 사용자 참여 예측 모델을 실험합니다.

- **Performance Highlights**: FSL 기반의 모델이 전통적인 모델보다 높은 정확도를 기록했습니다. 특히 기존의 도메인 불가지론적 모델링 기법 대비, 제안한 FSL 접근 방식이 다양한 실험에서 뛰어난 성과를 나타냈습니다. 이는 향후 비디오 게임을 넘어서 다양한 다중 도메인 문제에 적용될 수 있는 가능성을 제시합니다.



### Semi-overcomplete convolutional auto-encoder embedding as shape priors for deep vessel segmentation (https://arxiv.org/abs/2409.13001)
Comments:
          5 pages, 4 figures, conference

- **What's New**: 이 논문에서는 혈관의 자동 분할을 위한 새로운 방법을 제안합니다. 기존의 U-Net 기반 아키텍처가 혈관 시스템을 자동으로 delineate하는 데 어려움을 겪고 있으며, 우리는 이를 극복하기 위해 Semi-Overcomplete Convolutional Auto-Encoder (S-OCAE)에서 도출된 shape priors를 통합했습니다.

- **Technical Details**: 제안된 방법은 데이터의 고차원 투영을 통해 작은 구조를 더 잘 특성화하는 과정을 포함합니다. S-OCAE는 다중 경로 인코더를 활용하여 다중 스케일 혈관 나무 기하학을 캡처합니다. 또한, 두 가지 임상 데이터셋(DRIVE, 3D-IRCADb)에 대해 실험을 수행하여 U-Net과 비교하여 효과성을 검증했습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 기존의 CAE에서 shape priors를 사용한 U-Net보다 더 높은 성능을 보였으며, 특히 망막 및 간 혈관 추출에서 뛰어난 성과를 나타냈습니다.



### Seeing Through Their Eyes: Evaluating Visual Perspective Taking in Vision Language Models (https://arxiv.org/abs/2409.12969)
- **What's New**: 이 논문은 시각적 관점 이해(Visual Perspective-Taking, VPT) 능력의 평가를 위한 새로운 데이터셋인 Isle-Bricks와 Isle-Dots를 소개합니다. 이는 최근에 개발된 비전 언어 모델(Vision Language Models, VLMs)의 VPT 능력을 테스트하는 데 사용됩니다.

- **Technical Details**: 연구에서는 12개의 VLM을 평가했으며, 모델들은 VPT를 요구할 때 성능이 크게 하락하는 경향을 보였습니다. 또한, 객체 탐지(object detection) 성능은 VPT 성능과 잘 상관되지 않아 기존의 벤치마크가 이 문제를 충분히 이해하는 데 부족할 수 있음을 보여줍니다.

- **Performance Highlights**: 모델은 여러 사람이 있는 장면에서 더 크게 성능이 떨어지며, 다수의 사람들이 있는 이미지에서 VPT를 수행하는 데 어려움을 겪었습니다. 모든 모델에서 97% 이상 이해 가능한 답변을 제공했지만, 'Chain-of-Thought' 기술을 사용할 때는 때때로 비이해 가능 답변이 늘어나는 경향이 있었습니다.



New uploads on arXiv(cs.AI)

### Dermatologist-like explainable AI enhances melanoma diagnosis accuracy: eye-tracking study (https://arxiv.org/abs/2409.13476)
- **What's New**: 이 연구에서는 피부과 의사들이 인공지능(AI) 및 설명 가능한 인공지능(XAI) 도구를 사용하는 방식을 객관적으로 평가할 필요성을 강조하고 있습니다.

- **Technical Details**: 76명의 피부과 의사가 16개의 피부 병변 이미지를 XAI 시스템을 사용하여 진단하였으며, 이 시스템은 도메인별 설명을 제공합니다. 연구에서는 Eye-tracking 기술을 활용하여 의사들의 상호작용을 평가하였습니다. 표준 AI 시스템과 비교하여 진단 성능을 분석하였습니다.

- **Performance Highlights**: XAI 시스템이 표준 AI에 비해 조화로운 진단 정확도를 2.8% 향상시키는 것으로 나타났습니다. 또한, AI/XAI 시스템 및 복잡한 병변과의 진단 불일치는 인지적 부담(cognitive load)을 증가시키는 것으로 나타났습니다.



### A User Study on Contrastive Explanations for Multi-Effector Temporal Planning with Non-Stationary Costs (https://arxiv.org/abs/2409.13427)
- **What's New**: 이번 논문에서는 스마트 홈의 시간 계획을 위한 사용자 애플리케이션 내에서 contrastive explanations를 채택하였습니다. 이는 사용자들이 에너지 요금에 따라 지불하고, 높은 용량의 배터리 저장소를 이용하며, 에너지를 그리드에 판매할 수 있도록 돕는 방법입니다.

- **Technical Details**: 이 연구는 Cuttlefish라 불리는 시스템을 통해 다수의 장치 동시 스케줄링을 해결하는 다중 효과기 계획 문제(multi-effector planning problem)를 다룹니다. 기존의 PDDL 기반 플래너는 이러한 비국소 비용이 있는 문제를 해결할 수 없으므로, 도메인 종속 플래너를 설계하여 합리적인 수치로 확장 가능한 기능을 구현했습니다. A* 알고리즘과 도메인 종속 히uristic를 사용하는 상태 공간 검색 플래너도 개발되었습니다.

- **Performance Highlights**: 128명의 참가자를 대상으로 한 통제된 사용자 연구에서, contrastive 질문과 설명을 제공받은 참가자들은 추천된 AI 일정에 대해 더 높은 만족도와 이해도를 보였으며, 도움을 더 유용하다고 평가했습니다.



### LLMs Still Can't Plan; Can LRMs? A Preliminary Evaluation of OpenAI's o1 on PlanBench (https://arxiv.org/abs/2409.13373)
- **What's New**: OpenAI의 최신 모델 o1 (Strawberry)은 기존의 LLM과 구별되는 새로운 reasoning 모델로 평가되고 있으며, 이는 Chain-of-Thought (CoT) 기반 추론이 특징이다. 이 논문은 o1의 성능을 PlanBench에서 평가하고 그 한계를 탐구한다.

- **Technical Details**: PlanBench는 LLM의 계획 능력을 평가하는 벤치마크로, 600개의 블록 문제를 포함하며, o1은 이러한 평가에서 상당한 성과를 달성했다. o1은 강화 학습 기반(pre-training RL)으로 훈련되어 있으며, 유사한 모델인 AlphaGo와 유사한 메커니즘을 가질 가능성이 있다. 저자들은 효율성, 비용, 보증을 포함하여 LRM의 추론 능력을 측정하는 새로운 접근 방식이 필요하다고 주장한다.

- **Performance Highlights**: o1의 성능은 PlanBench에서 경쟁 모델보다 뛰어나지만 여전히 완전한 결정을 내리기에는 부족하다. LLaMA 3.1 405B가 Blocksworld에서 62.6%의 정확도로 최고 성과를 기록했으나, Mystery Blocksworld에서는 성능이 대폭 저조하였다. 이 연구는 LLM의 계획 능력에 대한 지속적인 한계를 명확히 보여준다.



### HeadCT-ONE: Enabling Granular and Controllable Automated Evaluation of Head CT Radiology Report Generation (https://arxiv.org/abs/2409.13038)
- **What's New**: 이번 연구에서는 Head CT 보고서 생성을 평가하기 위한 새로운 메트릭인 HeadCT-ONE을 제시합니다. 이 메트릭은 특정 도메인에 기반한 온톨로지를 통한 엔티티(entities) 및 관계(relation) 추출을 통해 현재의 정보 추출 메트릭을 개선합니다.

- **Technical Details**: HeadCT-ONE은 의료 용어의 표준화된 프레임워크를 제공함으로써, 세분화된 개념 분류와 다양한 엔티티에 대한 커스터마이징된 가중치를 부여할 수 있는 기능을 갖추고 있습니다. 세 가지 건강 시스템에서 수집한 Head CT 보고서를 통해 엔티티 정규화 및 가중치 적용 방법이 정상과 비정상 보고서를 더 잘 구별하고, 방사선 전문의의 임상적으로 중요한 오류 평가와 일치함을 보여줍니다.

- **Performance Highlights**: HeadCT-ONE은 동일한 의미를 가진 보고서 간의 유사성을 효과적으로 포착하며, 보고서 생성에서의 정상 및 비정상 사항 간의 차별화가 더 잘 이루어집니다. 이는 특히 AI 생성 보고서의 특수한 측면에 맞추어 조정 가능하다는 점에서 유리합니다.



### Morphological Detection and Classification of Microplastics and Nanoplastics Emerged from Consumer Products by Deep Learning (https://arxiv.org/abs/2409.13688)
- **What's New**: 이 논문에서는 새로운 마이크로 및 나노플라스틱(MiNa, micro- and nanoplastics) 데이터셋을 소개하여 자동 검출 및 분류를 가능하게 하였습니다. 이 데이터셋은 실제 수중 조건에서 시뮬레이션된 스캐닝 전자 현미경 이미지를 포함하고 있으며, 다양한 크기와 폴리머 종류에 따라 플라스틱을 분류합니다.

- **Technical Details**: MiNa 데이터셋은 고급 물체 검출 알고리즘을 이용한 자동 검출 및 분류를 지원하며, 각기 다른 물체 검출 기술들을 적용하여 마이크로 및 나노플라스틱의 특성을 평가합니다. 사용된 주요 모델들은 Mask R-CNN, Faster R-CNN, YOLOv10 등입니다.

- **Performance Highlights**: 이 데이터셋은 4가지 주요 폴리머 타입을 포함하고 있으며, 기존 데이터셋 보다 더 넓은 크기 범위를 다루고 있습니다. 공개된 데이터셋으로서 마이크로플라스틱 연구에 크게 기여할 수 있는 가능성을 보여줍니다.



### The Impact of Large Language Models in Academia: from Writing to Speaking (https://arxiv.org/abs/2409.13686)
Comments:
          16 pages

- **What's New**: 대규모 언어 모델(LLMs)이 인간 사회에 미치는 영향에 대한 최초의 대규모 조사 연구입니다. 30,000개 이상의 논문과 1,000개의 발표를 분석했습니다.

- **Technical Details**: 본 연구에서는 기계 학습 회의에서 발언된 텍스트 및 발표에서 사용된 단어를 분석하여 LLMs의 영향을 비교했습니다. LLM 스타일의 단어, 예를 들어 'significant'와 같은 단어가 초록과 구두 발표에서 더 빈번하게 사용된다는 결과를 도출했습니다.

- **Performance Highlights**: LLMs의 영향은 말하기(speaking)에도 나타나기 시작했으며, 앞으로 더욱 커질 것으로 예상됩니다. 이는 인간 사회에 대한 LLMs의 암묵적 영향과 파급 효과를 강조합니다.



### The FIX Benchmark: Extracting Features Interpretable to eXperts (https://arxiv.org/abs/2409.13684)
- **What's New**: 이 논문에서는 모델 예측을 설명하기 위한 기존의 피처 기반 방법들이 해석 가능한 피처가 쉽게 접근 가능하다는 전제를 갖고 있다고 지적합니다. 하지만 고차원 데이터에서는 이러한 피처를 명확하게 정의하기 어려운 경우가 많습니다. FIX (Features Interpretable to eXperts)라는 새로운 벤치마크를 제안하여, 전문가 지식과 맞는 피처들을 자동으로 추출할 수 있는 방법론의 필요성을 강조하고 있습니다.

- **Technical Details**: FIX 벤치마크는 전문가의 지식과 해석 가능성을 측정하기 위한 통합 평가 체계를 제안합니다. 이 프레임워크는 다양한 실제 환경에서 전문가와 협력하여 개발된 피처 해석 가능성 목표를 포함하고, 우주론, 심리학, 의학 등 다양한 분야에서 수집된 6개의 데이터셋을 사용합니다. 또한, 전문가가 쉽게 이해할 수 있는 높은 수준의 피처로 그룹화된 관련 저수준 피처를 추출하는 방법론에 초점을 맞추고 있습니다.

- **Performance Highlights**: 기존의 피처 기반 설명 방법들이 전문가의 지식과 잘 맞지 않음을 발견하여, 전문가가 이해할 수 있는 피처를 자동으로 추출하는 새로운 방법론을 개발할 필요성이 분명해졌습니다. FIXScore라는 새로운 메트릭을 통하여, 다양한 실제 환경에서 피처의 해석 가능성을 평가할 수 있는 기반을 마련했습니다.



### ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robot Navigation (https://arxiv.org/abs/2409.13682)
- **What's New**: 이 연구에서는 로봇이 긴 시간 동안 상호작용하며 다양한 질문에 대답할 수 있는 새로운 시스템인 Retrieval-augmented Memory for Embodied Robots(ReMEmbR)를 소개합니다. ReMEmbR는 로봇 내비게이션을 위한 긴 시간 비디오 질문 응답 시스템으로, 나아가 이를 평가하기 위해 NaVQA 데이터셋을 제안합니다.

- **Technical Details**: ReMEmbR는 기억 구축 및 질의 단계로 구성되며, 로봇의 역사적 데이터를 효과적으로 처리하기 위해 temporal information, spatial information, 그리고 이미지를 활용합니다. 이 시스템은 로봇이 긴 시간 동안 축적한 데이터를 바탕으로 다양한 질문에 대답하고 행동을 수행할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, ReMEmbR는 기존의 LLM과 VLM 기반 시스템보다 뛰어난 성능을 보여주었으며, 낮은 지연 시간으로 효과적인 장기적 추론이 가능함을 입증했습니다. 실제 로봇에 ReMEmbR를 배포한 결과, 다양한 질문에 대한 대응 능력을 보여줍니다.



### A sound description: Exploring prompt templates and class descriptions to enhance zero-shot audio classification (https://arxiv.org/abs/2409.13676)
Comments:
          DCASE 2024 - 9th Workshop on Detection and Classification of Acoustic Scenes and Events, Oct 2024, Tokyo, Japan

- **What's New**: 이번 연구는 제로샷 오디오 분류(zero-shot audio classification)를 위한 새로운 템플릿을 탐색하며, 보다 높은 성능을 내는 옵션이 존재함을 보여줍니다. 특히, 적절한 형식의 프롬프트(prompts) 사용이 결과에 미치는 영향과 오디오 중심 설명으로 클래스 레이블(class labels)을 보강하는 방법을 제안합니다.

- **Technical Details**: 저자들은 클래스를 임의로 정의하는 대신 LLMs(Large Language Models)를 통해 오디오 중심의 설명을 자동으로 생성하는 방법을 소개합니다. 이 방법은 수작업으로 설명을 수집하는 데 수반되는 노력 없이, 자가 주의(self-attention) 기반 모델을 사용하여 데이터 간의 중복성을 해결합니다.

- **Performance Highlights**: 제안된 접근법을 통해 zero-shot 오디오 분류에서 최신 기술(state-of-the-art) 결과를 달성합니다. 이 방법은 추가적인 훈련 없이도 높은 성능을 유지하며, 주요 환경 사운드 데이터셋에서 효율적인 결과를 입증했습니다.



### OATS: Outlier-Aware Pruning Through Sparse and Low Rank Decomposition (https://arxiv.org/abs/2409.13652)
- **What's New**: 본 논문은 OATS(Outlier-Aware Pruning Through Sparse and Low Rank Decomposition)라는 새로운 방법론을 소개하며, 대규모 모델을 압축할 때 재훈련 없이도 성능 저하 없이 최첨단 성능을 달성할 수 있음을 입증합니다.

- **Technical Details**: OATS는 입력 임베딩의 두 번째(moment) 정보를 활용하여 모델 가중치를 희소(sparse) 행렬과 저랭크(low-rank) 행렬의 합으로 분해합니다. OATS는 Phi-3 및 Llama-3와 같은 대형 언어 모델 및 ViT, DINOv2와 같은 비전 변환기에서 성능을 높입니다.

- **Performance Highlights**: OATS는 최대 60%의 압축률을 달성하며, 기존의 방법들과 비교했을 때 CPU 가속도가 최대 1.37배 향상됩니다. 그래픽스 및 개념적으로 효율적인 완벽한 균형을 이루면서 다양한 성능 지표에서 새로운 최첨단 성능을 보여주고 있습니다.



### Advancing Event Causality Identification via Heuristic Semantic Dependency Inquiry Network (https://arxiv.org/abs/2409.13621)
- **What's New**: 이번 연구에서는 Event Causality Identification (ECI) 문제를 해결하기 위한 새로운 접근법으로 Semantic Dependency Inquiry Network (SemDI)를 제안합니다. SemDI는 텍스트 내 사건 간의 인과 관계를 파악하는 데 초점을 맞추고 있으며, 문맥 속의 의미적 의존성을 포착하는 통합 인코더를 활용합니다.

- **Technical Details**: SemDI 모델은 사건 쌍 중 하나의 사건을 무작위로 마스킹한 후, Cloze Analyzer를 사용하여 Context에 대한 포괄적인 이해를 바탕으로 채워야 할 토큰을 생성합니다. 이 토큰은 특정 사건 간의 인과 관계를 질의하는 데 사용됩니다.

- **Performance Highlights**: SemDI는 세 가지 널리 사용되는 벤치마크에서 이전 SOTA 모델에 비해 각각 7.1%, 10.9%, 14.9%의 F1-score 향상을 기록하며 우수한 성능을 입증하였습니다.



### MaPPER: Multimodal Prior-guided Parameter Efficient Tuning for Referring Expression Comprehension (https://arxiv.org/abs/2409.13609)
Comments:
          EMNLP 2024

- **What's New**: 이 논문에서는 기존의 Referring Expression Comprehension (REC) 작업을 위한 새로운 효율적인 파라미터 전송 학습 방법인 MaPPER를 소개합니다. 이 방법은 Multimodal Prior-guided Parameter Efficient Tuning을 기반으로 하며, 구체적으로는 Dynamic Prior Adapters와 Local Convolution Adapters를 사용하여 지역적 의미를 추출하고 시각적 인식을 향상시킵니다.

- **Technical Details**: MaPPER는 Dynamic Prior Adapter (DyPA)와 Local Convolution Adapter (LoCA)를 포함합니다. DyPA는 정렬된 prior에 의해 각 토큰의 중요성을 계산해 동적으로 조정하며, LoCA는 다중 스케일의 지역적 지식을 통합하여 비주얼 트랜스포머의 표현력을 향상시킵니다. Prior-Guided Text 모듈을 도입하여 텍스트와 비주얼 정보를 융합합니다.

- **Performance Highlights**: 실험 결과, MaPPER는 RefCOCO, RefCOCO+, RefCOCOg와 같은 세 가지 벤치마크에서 이전의 SOTA 방법들보다 뛰어난 성능을 보이며, 단 1.41%의 튜닝 파라미터로 최고의 정확도를 달성하였습니다.



### MeLIAD: Interpretable Few-Shot Anomaly Detection with Metric Learning and Entropy-based Scoring (https://arxiv.org/abs/2409.13602)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이번 논문에서는 해석 가능한 (interpretable) 이상 탐지 (anomaly detection)를 위한 새로운 방법론인 MeLIAD를 제안합니다. MeLIAD는 기존 방법들과는 다르게 metric learning에 기반하여, 사전 분포 가정에 의존하지 않고도 해석 가능성을 제공합니다.

- **Technical Details**: MeLIAD는 훈련을 위해 몇 개의 이상 샘플만 필요하며, 데이터 증강 (augmentation) 기법을 사용하지 않으면서 본질적으로 해석 가능한 (interpretable) 방법입니다. 이 방법은 이상 사례를 식별하고 지역화하기 위한 새로운 trainable entropy-based scoring component와 metric learning 목표와 함께 이상 점수 구성 요소를 최적화하는 새로운 손실 함수 (loss function)를 도입했습니다.

- **Performance Highlights**: 다섯 개의 공개 벤치마크 데이터 세트를 활용한 실험 결과, MeLIAD는 기존 최고 성능 방법들과 비교하여 이상 탐지 및 지역화 성능에서 향상된 결과를 보여주었습니다.



### YesBut: A High-Quality Annotated Multimodal Dataset for evaluating Satire Comprehension capability of Vision-Language Models (https://arxiv.org/abs/2409.13592)
Comments:
          EMNLP 2024 Main (Long), 18 pages, 14 figures, 12 tables

- **What's New**: 이번 논문에서는 이미지에서 풍자를 탐지하고 이해하는 어려운 작업을 제안하며, 이를 평가하기 위한 고품질 데이터셋인 YesBut을 발표합니다. 이 데이터셋은 2,547개의 이미지로 구성되어 있으며, 풍자적 이미지(1,084개)와 비풍자적 이미지(1,463개)가 포함되어 있습니다.

- **Technical Details**: 우리는 (1) Satirical Image Detection (풍자적 이미지 탐지), (2) Satirical Image Understanding (풍자적 이미지 이해), (3) Satirical Image Completion (풍자적 이미지 완성)라는 세 가지 벤치마크 작업을 제안합니다. 이 작업들은 VL(Vision-Language) 모델의 일반적인 기능을 초월하여 풍자의 맥락을 이해해야 합니다.

- **Performance Highlights**: 현재 SOTA VL 모델이 YesBut 데이터셋에서 풍자적 이미지 탐지, 이해, 완성 작업에서에서도 열악한 성능을 보였습니다. 특히 제로샷 환경에서는 풍자적 이미지 탐지가 매우 어려웠습니다.



### ChainBuddy: An AI Agent System for Generating LLM Pipelines (https://arxiv.org/abs/2409.13588)
Comments:
          12 pages, 5 figures, pre-print

- **What's New**: ChainBuddy는 사용자가 LLM(대형 언어 모델) 평가를 시작할 수 있도록 돕는 AI 기반의 어시스턴트로, ChainForge 플랫폼에 통합되어 있습니다. 이 어시스턴트는 사용자가 특정 업무에 맞춤형 LLM 파이프라인(‘플로우’)을 생성할 수 있는 간편한 방법을 제공합니다.

- **Technical Details**: ChainBuddy는 기존의 오픈 소스 비주얼 환경인 ChainForge 위에 구축되어 있으며, 사용자가 초기 프롬프트를 제공하면 자동으로 시작할 수 있는 LLM 파이프라인을 생성합니다. 이를 통해 사용자는 더 이상 '백지 문제(blank page problem)'에 직면하지 않고, LLM의 다양한 행동을 평가하는 데 필요한 구조와 지침을 받게 됩니다.

- **Performance Highlights**: ChainBuddy 사용자는 AI 지원을 통해 작업량이 감소하고 LLM 평가 파이프라인 설정에 있어 자신감이 향상된다고 보고했습니다. 실험 결과에서도 ChainBuddy는 사용자 노력의 감소 및 요구 수집 기능에 대한 긍정적인 반응을 이끌어냈으며, 참가자 대다수가 어시스턴트의 능력에 놀라움을 표했습니다.



### Neurosymbolic Conformal Classification (https://arxiv.org/abs/2409.13585)
Comments:
          10 pages, 0 figures. arXiv admin note: text overlap with arXiv:2404.08404

- **What's New**: 본 논문은 neurosymbolic AI 및 conformal prediction 접근 방식을 결합하여 머신러닝(ML) 시스템의 약점을 완화하고 신뢰할 수 있는 AI를 설계하는 방법을 제안합니다.

- **Technical Details**: Neurosymbolic Artificial Intelligence (NeSy AI)는 신경 네트워크의 학습 능력과 상징 체계의 추론 능력을 결합하는 연구 분야입니다. Conformal prediction은 ML 시스템의 불확실성을 고려하여 단일 예측을 '신뢰 구간'으로 변환하는 기술이며, 이는 통계적 보증을 동반합니다. 두 접근 방식 모두 분포에 구애받지 않으며 모델에 독립적입니다.

- **Performance Highlights**: 이 논문에서는 다중 레이블 conformal 분류를 위한 새로운 방법을 소개하고, 과거 지식을 통합하는 두 가지 기술을 개발하여 성능 개선을 목표로 하고 있습니다.



### Time and Tokens: Benchmarking End-to-End Speech Dysfluency Detection (https://arxiv.org/abs/2409.13582)
- **What's New**: 이번 연구에서는 자동 음성 인식(ASR) 문제로서 담화의 비유창성(dysfluency) 문제를 새로운 관점에서 접근하였습니다. 우리는 비유창성을 토큰화(tokenizing)하고 이를 기반으로 한 통합 모형을 구축하여 실질적인 성능을 확보했습니다.

- **Technical Details**: 우리는 텍스트와 음성을 통한 비유창성 모델링을 위해 텍스트 시뮬레이터와 음성 시뮬레이터를 사용했습니다. 새로운 데이터셋 VCTK-Token을 생성하고 Whisper 아키텍처를 적용하여 비유창성 예측 시스템을 개발했습니다.

- **Performance Highlights**: 시간 기반 방법들보다 토큰 기반 방법이 대부분의 측정 지표에서 더 나은 성능을 보였으며, 연구 커뮤니티와의 협력을 위해 이 모든 리소스를 오픈 소스로 공개했습니다.



### Region Prompt Tuning: Fine-grained Scene Text Detection Utilizing Region Text Promp (https://arxiv.org/abs/2409.13576)
- **What's New**: 본 논문에서는 scene text detection에서의 섬세한 text feature를 포착하기 위해 새로운 방법인 region prompt tuning (RPT)을 제안합니다.

- **Technical Details**: RPT 프레임워크는 general text prompt와 region text prompt를 사용하여 각각 고정된 word embedding과 섬세한 local features를 동시에 처리합니다. 특히, character-token correspondence를 위한 position embedding 공유와 bidirectional distance loss를 도입하여 region text prompt와 detection target을 정렬합니다.

- **Performance Highlights**: 실험 결과, ICDAR2015, TotalText, CTW1500 벤치마크에서 RPT 방법이 scene text detection에서 뛰어난 성능을 보여주며, 세부적인 feature를 포착하는 데 효과적임을 입증했습니다.



### Scalable Multi-agent Reinforcement Learning for Factory-wide Dynamic Scheduling (https://arxiv.org/abs/2409.13571)
- **What's New**: 본 논문에서는 기존의 인간 제작 결정 규칙에 의존하지 않고, 리더-추종자 (leader-follower) 구조의 멀티 에이전트 강화 학습 (MARL)을 적용하여 대규모 공장 내 실시간 스케줄링 문제를 해결하는 혁신적인 접근 방식을 소개합니다.

- **Technical Details**: 다양한 운영 요구를 수용하기 위해, 스케줄링 문제를 여러 개의 하위 문제로 분해하고 각 개별 에이전트가 이를 처리하며, 이는 에이전트 간의 협업을 통해 이루어집니다. 이를 위해, 에이전트의 잘못된 결정으로 인한 생산 용량 손실을 방지하기 위해 규칙 기반의 전환 알고리즘을 제안합니다.

- **Performance Highlights**: 제안된 모델은 기존의 최첨단 심층 강화 학습 기반 스케줄링 모델들과 비교하여 교육 중 성능 개선이 두드러지며, 높은 수요 상황에서도 상대적으로 성능 저하가 적어 10.4%의 지연 감소 및 31.4%의 완료율 향상을 기록했습니다.



### Deep Learning and Machine Learning, Advancing Big Data Analytics and Management: Tensorflow Pretrained Models (https://arxiv.org/abs/2409.13566)
Comments:
          This book contains 148 pages and 7 figures

- **What's New**: 이 책은 TensorFlow의 사전 훈련된 모델을 딥러닝에서 효과적으로 사용하는 방법을 상세히 안내합니다. 이미지 분류 및 객체 감지와 같은 다양한 작업을 수행하기 위한 현대적인 아키텍처(architectures)인 ResNet, MobileNet, EfficientNet의 실제 구현을 다룹니다.

- **Technical Details**: TensorFlow는 Google이 개발한 오픈 소스 플랫폼으로, 기계 학습 및 딥러닝 애플리케이션에 사용됩니다. TensorFlow의 주요 구성 요소로는 TensorFlow Core(모델 설계 및 실행을 위한 저수준 API), Tensors(n차원 배열을 나타내는 데이터 구조), Graph(연산을 표현하는 계산 그래프), Session(그래프 내에서의 연산을 관리하는 세션), Eager Execution(즉시 연산을 실행하는 모드), Estimators와 Keras(모델 구축 및 훈련 단순화를 위한 고수준 API)가 포함됩니다.

- **Performance Highlights**: 전이 학습(Transfer Learning)을 통해 사전 훈련된 모델을 새로운 관련 작업에 쉽게 적응시킬 수 있습니다. 이러한 접근 방식은 대규모 라벨 데이터 및 계산 자원 없이도 모델 성능을 향상시킬 수 있습니다. 사전 훈련된 모델은 이미지 분류, 객체 감지, 자연어 처리(NLP)와 같은 다양한 작업에 광범위하게 사용되고 있으며, 사용자가 복잡한 기계 학습 작업을 신속하게 시작할 수 있도록 돕습니다.



### Efficient Visualization of Neural Networks with Generative Models and Adversarial Perturbations (https://arxiv.org/abs/2409.13559)
Comments:
          4 pages, 3 figures

- **What's New**: 이 논문은 생성 네트워크(Generative Network)를 통한 새로운 심층 시각화(Deep Visualization) 접근 방식을 제시하며, 기존 방법들보다 개선된 성능을 보여줍니다. 이 모델은 기존에 사용되던 여러 네트워크의 수를 줄여 단일 생성기(Generator)와 판별기(Discriminator)만을 필요로 하며, 비대립적 훈련 과정을 통해 훈련됩니다.

- **Technical Details**: 이 연구의 핵심 기여는 특정 클래스 레이블(Class Labels)에 부합하는 세밀한 시각화 이미지를 생성하는 능력입니다. 모델은 레이블 유도 이미지 생성(Label-directed Image Generation)을 향상시키기 위해 고유한 스킵 연결(Skip-connection)에서 영감을 받은 블록 디자인을 통합하였습니다. 판별기는 생성된 이미지의 진위 여부를 판단하는 역할을 하며, 훈련 과정에서 실제 데이터셋을 기반으로 사전 훈련된 후 생성기를 지도합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 적대적 예제 생성 기술보다 우수한 성능을 보이며, 목표 공격(Targeted Attacks) 및 비목표 공격(Non-targeted Attacks)에서 최대 94.5%의 혼란률(Fooling Rate)을 달성하며, 최소한의 변형으로 효과적으로 DNN을 속일 수 있음을 입증하였습니다. 이를 통해 시각화 품질을 평가하는 정량적 지표로서의 혼란률의 가능성을 제시합니다.



### Trustworthy Hate Speech Detection Through Visual Augmentation (https://arxiv.org/abs/2409.13557)
- **What's New**: 새로운 hate speech detection methods(증오 발언 탐지 방법)인 TrusV-HSD를 제안하며, 이는 시각적 보강을 통해 의미 정보를 강화하고 신뢰할 수 있는 손실 함수(trustworthy loss)를 통해 불확실성을 감소시킵니다.

- **Technical Details**: TrusV-HSD는 세 가지 주요 모듈로 구성됩니다: visual cues generation module(시각적 단서 생성 모듈), hate speech detector(증오 발언 탐지기) 및 trustworthy loss(신뢰할 수 있는 손실). 이 방법은 이미지와 텍스트가 쌍이 되어있지 않은 데이터를 사용하여 시각적 단서를 생성하고, 이를 통해 의미적 학습과 멀티모달 연결성을 효과적으로 수행합니다.

- **Performance Highlights**: 실험 결과, TrusV-HSD는 기존의 방법들에 비해 뛰어난 성능 향상을 보여줍니다. 이 접근 방식은 파트너 데이터의 필요 없이 다양한 길이의 발언에 대해서도 강력한 성능을 보장합니다.



### Generating Visual Stories with Grounded and Coreferent Characters (https://arxiv.org/abs/2409.13555)
- **What's New**: 이 논문은 캐릭터 중심의 이야기 생성(character-centric story generation)의 새로운 작업을 소개하며, 이를 위해 첫 번째로 시각적 이야기(visual stories)에서 일관성 있게 캐릭터 언급을 예측할 수 있는 모델을 제안합니다.

- **Technical Details**: 제안된 모델은 VIST 벤치마크에 기초하여 새로 구축된 데이터셋으로 미세 조정(finetuned)되었으며, 시각적 및 텍스트 캐릭터 코리퍼런스(coreference) 체인을 풍부하게 만들기 위한 자동화된 파이프라인이 개발되었습니다. 새로운 평가 지표도 이야기의 캐릭터 풍부성과 코리퍼런스를 측정하기 위해 제안되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 반복되는 캐릭터를 포함하고 있으며, 기존의 기준선과 최신 시스템에 비해 더 일관性 있고 코리퍼런트한 이야기를 생성하는 것으로 나타났습니다.



### Certified Adversarial Robustness via Partition-based Randomized Smoothing (https://arxiv.org/abs/2409.13546)
- **What's New**: 이 논문에서는 Gaussian smoothing 방식으로는 고차원 이미지 데이터셋에서 인증된 정확도가 낮아질 수 있는 문제를 지적하고, Pixel Partitioning-based Randomized Smoothing (PPRS) 방법론을 제안하여 이미지의 가시성을 높이고 인증된 예측 반경을 증대시키는 방안을 소개합니다.

- **Technical Details**: PPRS 방법은 입력 이미지의 픽셀을 파티셔닝하고, 파티션 내의 평균 픽셀 강도를 각 픽셀에 할당하여 노이즈가 추가된 이미지의 가시성을 개선합니다. 이 과정에서 섬세한 파티션을 통해 Gaussian 노이즈의 유효 분산이 감소하여 신호 대 노이즈 비율이 향상됩니다. PPRS는 표준 슈퍼픽셀 방법을 적용하여 이미지 픽셀의 의미론적 파티셔닝을 수행합니다.

- **Performance Highlights**: MNIST, CIFAR-10, ImageNet 데이터셋을 사용한 수치적 연구 결과, PPRS는 인증된 정확도와 예측 모델의 안정성을 크게 향상시킴을 보여줍니다. PPRS를 적용한 결과, 이미지의 시각적 품질이 향상되어 기존의 무작위 스무딩 기법에 비해 더 강한 내구성 인증서를 제공함을 시각적으로 입증하였습니다.



### First Place Solution to the Multiple-choice Video QA Track of The Second Perception Test Challeng (https://arxiv.org/abs/2409.13538)
- **What's New**: 이번 보고서에서는 제2회 Perception Test Challenge의 Multiple-choice Video QA 트랙에서 1위를 차지한 솔루션을 제공합니다. 본 대회는 복잡한 비디오 이해 과제를 제시하며, QwenVL2 모델을 활용하여 학습 세트에서 세부 조정을 하여 성능을 극대화했습니다.

- **Technical Details**: 우리는 QwenVL2 (7B) 모델을 사용했으며, 나이브 동적 해상도 입력(Naive Dynamic Resolution)과 다중모달 로터리 포지션 인베딩(Multimodal Rotary Position Embedding, M-ROPE) 같은 최신 기술을 통합하였습니다. 추가적으로, 5%의 데이터로 검증 세트를 나누어 모델을 훈련했고, Tensor 처리에 DeepSpeed ZeRO-2와 Low-Rank Adaptation (LoRA)을 사용했습니다.

- **Performance Highlights**: 최종 앙상블 결과로 Top-1 Accuracy 0.7647을 기록하였으며, 이는 다수 투표 및 테스트 시간 증강(Test Time Augmentation)을 통해 도출된 성과입니다. 이 모든 방법들은 복잡한 비디오 이해 과제에서 모델의 성능을 개선하는 데 중요한 역할을 했습니다.



### ShizishanGPT: An Agricultural Large Language Model Integrating Tools and Resources (https://arxiv.org/abs/2409.13537)
Comments:
          15 pages,3 figures, WISE2024

- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전으로 지능형 대화 시스템이 복잡한 질문을 처리하는 능력이 크게 향상되었습니다. 그러나 기존 LLM은 농업과 같은 특수 도메인 지식에서 여전히 제한적입니다. 이러한 문제를 해결하기 위해 ShizishanGPT라는 농업을 위한 지능형 질문 응답 시스템을 제안합니다.

- **Technical Details**: ShizishanGPT는 Retrieval Augmented Generation (RAG) 프레임워크와 에이전트 아키텍처에 기반하여 구성됩니다. 이 시스템은 다섯 가지 주요 모듈로 구성되며, 각각은 일반 질문에 대한 답변을 제공하는 GPT-4 기반 모듈, 시의적절한 업데이트가 불가능한 LLM의 문제를 보완하는 검색 엔진 모듈, 도메인 사실을 제공하는 농업 지식 그래프 모듈, RAG를 활용하여 도메인 지식을 보충하는 검색 모듈, 특화된 모델을 호출하여 작물 표현형 예측 및 유전자 발현 분석을 수행하는 농업 에이전트 모듈이 포함되어 있습니다.

- **Performance Highlights**: ShizishanGPT는 100개의 농업 관련 질문 데이터를 사용하여 평가하였으며, 실험 결과 모듈형 설계와 다양한 도메인 지식 출처의 통합 덕분에 일반 LLM보다 훨씬 더 정확하고 상세한 답변을 제공하는 것으로 나타났습니다.



### Contextualized AI for Cyber Defense: An Automated Survey using LLMs (https://arxiv.org/abs/2409.13524)
Comments:
          8 pages, 2 figures, 4 tables, accepted into 17th International Conference on Security of Information and Networks (SINCONF 2024)

- **What's New**: 이 논문은 2015년부터 2024년까지의 사이버 방어 능력을 향상시키기 위한 contextualized AI의 잠재력에 대한 연구 성장 추세를 보여줍니다. 연구는 강건성(robustness), 신뢰성(reliability), 통합 방법(integration methods)에 중점을 두었으며, 조직적 신뢰(organizational trust)와 거버넌스 프레임워크(governance frameworks)에서의 격차를 지적합니다.

- **Technical Details**: 본 연구에서는 LLM(대형 언어 모델)을 활용한 두 가지 문헌 조사 방법론을 제시합니다: (A) 초기 탐색을 위한 ChatGPT 4, (B) 문헌 필터링을 위한 Gemma 2:9b와 Claude 3.5 Sonnet을 통한 전체 텍스트 분석. 첫 번째 방법은 GPT-4를 사용해 주요 주제를 파악하는 반면, 두 번째 방법은 Gemma 2:9b로 효율적인 문헌 스크리닝을 하고 Claude 3.5 Sonnet으로 심층 분석을 수행합니다.

- **Performance Highlights**: GPT-4를 통한 초기 탐색에서는 34개의 독창적인 자료가 반환되었으며, 그 중 52.9%가 학술 출판물로 나타났습니다. Gemma 2:9b를 사용한 문헌 스크리닝은 논문의 관련성과 기여도를 점수화하여 '필독' 논문을 선별했습니다. 이 연구는 AI의 사이버 보안 적용을 위한 여러 관점과 전략적 방안을 제시하며, 협업 및 윤리적 고려사항의 중요성을 강조합니다.



### A Survey on Moral Foundation Theory and Pre-Trained Language Models: Current Advances and Challenges (https://arxiv.org/abs/2409.13521)
- **What's New**: 본 연구는 Moral Foundation Theory (MFT)와 관련된 Pre-trained Language Models (PLMs)에 대한 포괄적인 리뷰를 제공합니다.

- **Technical Details**: Moral Foundation Theory (MFT)는 다양한 문화가 개인과 사회의 삶을 형성하는 방식을 나타내는 핵심 도덕적 기초를 식별합니다. 본 논문은 PLMs에서 도덕적 경향성을 분석하고 MFT의 맥락에서 이들의 적용을 검토합니다. 또한 관련 데이터셋과 어휘집을 다루며 동향, 한계, 미래 방향에 대해서도 논의합니다.

- **Performance Highlights**: MFT와 PLMs의 교차점에서 도덕적 심리학의 통찰력을 제공함으로써 도덕적으로 인식하는 AI 시스템 개발을 위한 연구 및 개발을 위한 기반을 마련합니다.



### SatFed: A Resource-Efficient LEO Satellite-Assisted Heterogeneous Federated Learning Framework (https://arxiv.org/abs/2409.13503)
Comments:
          11 pages, 12 figures

- **What's New**: 본 논문에서는 저궤도(LEO) 위성을 활용하여 기존의 지상 네트워크에 의존하는 연합 학습(FL)의 한계를 극복하고, 자원 제약이 있는 이기종 환경에서의 FL을 지원하는 SatFed라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: SatFed는 최신 모델의 전달을 보장하기 위해 Freshness-based model prioritization queues를 구현하고, 데이터 분포, 전송 대역폭, 계산 능력 간의 관계를 실시간으로 포착하여 이기종 간의 상호 작용을 최적화하는 multigraph를 사용합니다.

- **Performance Highlights**: 실제 LEO 위성 네트워크를 사용한 포괄적인 실험 결과, SatFed는 기존의 최고 성능 벤치마크에 비해 우수한 성능과 강인성을 보여 주었습니다.



### HUT: A More Computation Efficient Fine-Tuning Method With Hadamard Updated Transformation (https://arxiv.org/abs/2409.13501)
- **What's New**: 이 연구는 기존의 Parameter Efficient Fine-Tuning (PEFT) 방법들의 한계를 극복하기 위해 직접 Updated Transformation (UT) 패러다임을 제안합니다. 이 접근법은 원래의 파라미터와 업데이트된 파라미터 간의 강한 상관관계를 유지하면서 파라미터 동역학을 보다 정확하게 캡처할 수 있도록 설계되었습니다.

- **Technical Details**: Hadamard Updated Transformation (HUT) 방법을 사용하여 두 개의 저차원 행렬을 통해 원래의 가중치 행렬을 효율적으로 업데이트합니다. HUT는 기존의 PEFT 방법들에 비해 계산 복잡성을 줄이며 더 풍부한 파라미터 특징을 캡처하기 위해 기능 변환을 활용합니다. 이 방법은 RoBERTa 및 GPT-2 모델을 기반으로 한 실험을 통해 검증되었습니다.

- **Performance Highlights**: HUT는 GLUE와 E2E 데이터셋에서 수행된 실험을 통해 기존의 PEFT 방법들에 비해 우수한 성능을 보여주었으며, 계산 복잡성을 현저히 줄이면서도 모델 품질을 유지하거나 향상시켰음을 확인했습니다.



### A Deep Learning Approach for Pixel-level Material Classification via Hyperspectral Imaging (https://arxiv.org/abs/2409.13498)
Comments:
          13 pages, 15 figures, 6 equations

- **What's New**: 이 연구는 전통적인 RGB 시스템의 한계를 극복하고, 고급 물질 특성을 요구하는 산업 분야에서 적용 가능한 하이퍼스펙트럴 (Hyperspectral) 이미징과 딥 러닝 (Deep Learning)의 결합 가능성을 평가합니다. 이 연구에서 제안된 P1CH 분류기는 경량의 CNN (Convolutional Neural Network) 아키텍처로 고해상도 물질 분류를 수행합니다.

- **Technical Details**: 하이퍼스펙트럴 이미징 기술은 900nm에서 1700nm 범위의 스펙트럼 정보를 포착하여 각 픽셀에 대한 세밀한 스펙트럼 데이터를 제공합니다. P1CH 모델은 이러한 스펙트럼 데이터를 활용하여, 99.94%의 정확도로 픽셀 단위의 물질 분류를 수행합니다. 실험 설계에는 HS 카메라, 컨베이어, 제어된 조명 장치가 포함되었으며 반 자동화된 마스크 생성 기능도 포함됩니다.

- **Performance Highlights**: P1CH 분류기의 성능 분석 결과, 색상, 크기, 형태 변형에 대한 강인성을 입증하며, 다양한 플라스틱 샘플에서 매우 높은 정확도를 기록했습니다(99.94%). 다만, 검정색 플라스틱과 같은 특정 물질에서는 여전히 도전과제가 존재합니다. 이러한 연구는 하이퍼스펙트럴 이미징을 통한 정확한 물질 분류의 새로운 가능성을 열어주는 동시에 향후 여러 산업에서의 광범위한 응용 가능성을 보여줍니다.



### DAP-LED: Learning Degradation-Aware Priors with CLIP for Joint Low-light Enhancement and Deblurring (https://arxiv.org/abs/2409.13496)
- **What's New**: 이번 논문에서는 저조도 환경에서의 이미지 복원 및 해상도 향상을 위해, Vision-Language 모델인 CLIP을 활용한 새로운 접근 방식을 제안합니다. 제안된 DAP-LED 프레임워크는 CLIP을 통해 저조도 및 블러링 문제를 동시에 해결하며, 이러한 방식이 다양한 다운스트림 작업에 유리하다는 점을 확인하였습니다.

- **Technical Details**: DAP-LED 프레임워크는 Transformer 기반의 Joint Learning 구조로, CLIP을 통해 야간 이미지의 저하 수준을 적응적으로 학습합니다. 이 과정에서 CLIP-guided Cross-fusion Module(CCM)을 도입하여, 이미지 임베딩에서 다중 스케일 패치 단위의 저하 히트맵을 생성합니다. 이후 이 히트맵은 설계된 CLIP-enhanced Transformer Blocks(CeTBs)를 통해 통합되어 유용한 저하 정보를 입력 데이터로부터 유지합니다.

- **Performance Highlights**: 실험 결과, DAP-LED는 기존 방법들에 비해 저조도 환경에서 최첨단 성능을 달성하였습니다. 또한, 이미지 복원 효과가 뛰어나며, 깊이 추정, 세분화 및 검출과 같은 다운스트림 작업에 효과적임을 입증하였습니다.



### 'Since Lawyers are Males..': Examining Implicit Gender Bias in Hindi Language Generation by LLMs (https://arxiv.org/abs/2409.13484)
- **What's New**: 이번 연구는 힌디어 텍스트 생성에서의 암묵적인 성별 편향을 탐구하고, 이를 영어와 비교하여 큰 언어 모델(LLMs)의 성별 편향의 변화를 강조합니다.

- **Technical Details**: 힌디어 데이터셋은 WinoBias에서 영감을 받아 생성되었으며, GPT-4o와 Claude-3 sonnet 모델로부터의 응답에서 고정관념적 패턴을 분석했습니다. 연구 결과, 힌디어에서의 성별 편향은 87.8%로 나타났으며, 영어 GPT-4o 생성에서는 33.4%의 편향이 확인되었습니다.

- **Performance Highlights**: 이번 연구는 occupation(직업), power hierarchies(권력 계층), social class(사회적 계급)와 관련된 성별 고정관념에 힌디어 응답이 자주 의존하고 있음을 보여줍니다. 이는 생성 AI 시스템에서 성별 편향을 탐색하는 데 고려해야 할 사항을 제시합니다.



### Deterministic versus stochastic dynamical classifiers: opposing random adversarial attacks with nois (https://arxiv.org/abs/2409.13470)
- **What's New**: 이번 논문에서는 생물학적 뉴런의 동적 상호작용을 설명하는 데 널리 사용되는 Continuous-Variable Firing Rate (CVFR) 모델을 실제로 사용할 수 있는 동적으로 보조된 분류기(dynamically assisted classifier)로 훈련시키고 테스트했습니다. 이 연구는 주어진 평형점에 대한 유도(basin of attraction)를 조각하여 클래스에 따른 대상을 적절히 분류할 수 있도록 설계되었습니다.

- **Technical Details**: CVFR 모델은 스펙트럼 분해(spectral decomposition)를 통해 구성이 되며, 서로 연결된 노드 간의 결합 행렬에 자생적으로 심어진 집합의 내용을 포함합니다. 훈련 과정에서는 입력 벡터가 해당하는 평형 상태로 진화하도록 유도하여 입력과 출력 간의 관계를 학습합니다. 또한, CVFR 모델의 비확률적 버전과 확률적 버전이 제안되어, 후자는 랜덤 공격에 대한 강건성을 보여줍니다.

- **Performance Highlights**: 제안된 방법론은 복잡한 분류 작업에서도 최첨단 DNN 구현과 유사한 정확도를 기록했습니다. 특히, MNIST와 같은 데이터셋에서 성능을 검증하였고, CVFR 모델이 비오모방적(biomimetic) 영감을 받았음에 따라 비정상적인 난이도의 분류 작업에서도 안정성을 유지함을 확인하였습니다.



### Global Outlier Detection in a Federated Learning Setting with Isolation Fores (https://arxiv.org/abs/2409.13466)
Comments:
          Accepted for publication at FLTA 2024: The 2nd IEEE International Conference on Federated Learning Technologies and Applications

- **What's New**: 이 논문에서는 서로 다른 두 서버를 사용하는 방법을 통해 연합 학습(federated learning) 환경에서 글로벌 이상치(global outliers)를 탐지하는 새로운 전략을 제시합니다. 특히, 클라이언트로부터 마스킹된(local data masking) 데이터를 서버에 전송하는 방식으로 민감한 정보의 노출을 방지하면서 이상치를 탐지할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 Isolation Forest(IF) 알고리즘을 사용하여 마스킹된 데이터에서 이상치를 탐지하고, 비식별화된 결과를 클라이언트에 통신하여 클라이언트가 자신의 로컬 데이터셋에서 이상치를 제거할 수 있도록 합니다. 이 과정에서, 클라이언트 간의 개인정보를 보호하기 위해 순열(permutation) 메커니즘이 추가되어, 서버는 어떤 클라이언트의 데이터 포인트가 마스킹되었는지 알 수 없습니다.

- **Performance Highlights**: 이 방법은 중앙화된 환경에서의 Isolation Forest 알고리즘과 유사한 결과를 제공합니다. 이는 연합 학습 모델 훈련 개선에 기여하며, 민감한 데이터의 보호 및 데이터 이동량 감소라는 두 가지 목표를 충족합니다.



### Differentially Private Multimodal Laplacian Dropout (DP-MLD) for EEG Representative Learning (https://arxiv.org/abs/2409.13440)
- **What's New**: 최근 다중 모드 전기뇌파(EEG) 학습이 질병 탐지에서 높은 가능성을 보여주고 있습니다. 이 논문에서는 둘 이상의 데이터 모드를 통합하여 EEG 데이터를 처리하기 위한 새로운 방법인 차별적으로 비공개된 다중 모드 라플라스 드롭아웃(DP-MLD)을 제안합니다. 특히, 이 방법은 언어 모델과 비전 변환기를 통합하여 데이터의 모달 리프레젠테이션을 처리하는 혁신적 접근을 사용합니다.

- **Technical Details**: 제안하는 DP-MLD 모델은 EEG 데이터를 텍스트로 처리하고, 다른 모달의 데이터를 비전 변환기로 이미지로 처리합니다. 또한, 교차 주의(cross-attention) 메커니즘을 설계하여 모달 데이터 간의 특성을 효과적으로 추출하고 통합합니다. 개인 정보 보호를 위한 라플라스 드롭아웃 메커니즘을 적용하여 각 특성에 랜덤 드롭아웃을 할당하고, 프라이버시 예산 내에서 성능을 최적화하도록 설계되었습니다. Gumbel-Softmax 기법을 활용하는 두 단계 최적화가 포함됩니다.

- **Performance Highlights**: 파킨슨병(PD) 환자에서의 동작 중단 냉각(FoG) 탐지를 위한 데이터셋을 사용한 실험 결과, 제안된 방법은 98.8%의 분류 정확도를 달성하며, 기존의 최첨단 방법보다 약 4% 향상되었습니다. 이는 개인 정보 보호를 고려하지 않은 접근 방식을 초월한 결과로, 다중 모드 EEG 학습에서 새로운 기준을 설정하고 있습니다.



### CVT-Occ: Cost Volume Temporal Fusion for 3D Occupancy Prediction (https://arxiv.org/abs/2409.13430)
- **What's New**: 본 논문에서는 CVT-Occ라는 새로운 방법론을 소개합니다. 이는 시간을 기준으로 복셀 간의 기하학적 유사성을 활용하여 3D 점유 예측의 정확도를 향상시키고자 합니다. 이 방법은 복셀의 시선 방향을 따라 샘플링한 점들을 통해 역사적 프레임의 기능을 통합하여 비용 볼륨 특징 맵을 구성하여 현재 볼륨의 특징을 개선합니다.

- **Technical Details**: CVT-Occ는 각 복셀에 대한 시선 방향을따라 점을 샘플링하고, 이 점들의 과거 프레임에서의 3D 위치를 확인하여 비용 볼륨 특징 맵을 구성합니다. 이 특징 맵은 현재 볼륨의 특징을 정제하여 점유 예측을 향상시키는 데 사용됩니다. 기존 방법들과 비교했을 때, CVT-Occ는 과거 관측의 시차 정보를 활용하여 계산 부하를 최소화하면서도 우수한 성능을 발휘합니다.

- **Performance Highlights**: CVT-Occ는 Occ3D-Waymo 데이터셋에서 엄격한 실험을 통해 검증되었으며, 최신 기술들과 비교했을 때 3D 점유 예측 성능에서 가장 우수한 결과를 도출했습니다. 추가적인 계산 비용이 거의 없는 상태에서 성능을 향상시키는 데 성공했습니다.



### Sine Wave Normalization for Deep Learning-Based Tumor Segmentation in CT/PET Imaging (https://arxiv.org/abs/2409.13410)
Comments:
          Report for Team WukongRT in the AutoPET III Challenge

- **What's New**: 이 보고서는 CT/PET 스캔에서 자동 종양 분할을 위한 정규화 블록을 소개하고 있습니다. 주요 혁신은 SineNormal의 도입으로, 이는 PET 데이터에 주기적인 Sine 변환을 적용하여 병변 탐지를 향상시키는 것을 목표로 합니다.

- **Technical Details**: SineNormal은 PET 이미지의 대사 활동을 개선하기 위해 특별히 설계된 모듈입니다. 복잡한 강도 패턴을 갖는 PET 이미지에 비선형 변환을 적용하여 미세한 대사 변화를 강조하고, 특히 종양 경계와 대사 이질성을 강조합니다. nnUNet ResEnc(M) 계획을 기반으로 한 네트워크 아키텍처를 사용하여, 6단계로 구성돼 점진적으로 특징 채널을 증가시킵니다. 네트워크는 3D CNN을 사용하고, SineNormalBlock을 통해 정상화된 PET 데이터가 입력됩니다.

- **Performance Highlights**: 훈련 데이터셋(n=1611)을 사용하여, 배치 사이즈 8과 패치 사이즈 112×160×128로 훈련하였습니다. 최적의 성능 모델은 1050 에폭에서 확보되었으며, 효율성을 극대화하기 위해 동적 슬라이딩 윈도우 접근방식과 테스트 시간 증강(TTA)을 구현하여 처리 속도를 향상시켰습니다.



### Validation & Exploration of Multimodal Deep-Learning Camera-Lidar Calibration models (https://arxiv.org/abs/2409.13402)
Comments:
          8 pages, 10 figures

- **What's New**: 이 연구는 깊이 학습 아키텍처를 활용하여 다중 모달 센서 시스템의 교정을 탐색하고 평가하며 구현하는 혁신적인 방법을 제시합니다. 특히 3D LiDAR와 2D 카메라 센서 간의 실시간 정렬을 목표로 하는 센서 융합(sensor fusion) 기법을 강조합니다.

- **Technical Details**: 전통적인 정적 교정(static calibration) 방법은 번거롭고 시간이 많이 소요되므로, CNN(Conventional Neural Networks)과 기하정보 학습(geometrically informed learning)을 결합하여 이 문제를 해결하려고 합니다. 연구는 RegNet, CalibNet, LCCNet과 같은 외부 LiDAR-카메라 교정 도구의 기초 원리를 활용합니다. 각 프레임워크에 대해 소스 코드를 조정하고, 미세 조정(fine-tuning), 훈련(training), 검증(validation), 테스트(testing)를 수행하여 공정한 비교를 위한 시각적이고 측정 가능한 결과를 도출합니다.

- **Performance Highlights**: 일련의 실험을 통해 각 모델의 단점을 분석하고 개선 가능성을 탐색합니다. 검증된 모든 모델 중에서 LCCNet이 가장 우수한 결과를 나타내는 것으로 확인되었습니다.



### Audio Codec Augmentation for Robust Collaborative Watermarking of Speech Synthesis (https://arxiv.org/abs/2409.13382)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이 논문은 협업 수watermarking 기법을 사용하여 전통적인 오디오 코덱와 신경 오디오 코덱에서 더 쉽게 감지할 수 있도록 하는 방법을 제안합니다. 이 접근 방식은 특히 디지털 음성 합성에서 발생하는 생성된 음성을 감지하는 데 혁신적입니다.

- **Technical Details**: 이 연구에서는 'channel augmentation' 기술을 이용하여 비차별적 전통 오디오 코덱 및 신경 오디오 코덱에서 협업 수watermarking의 성능을 평가합니다. 이 방법은 음성 신호의 생성 및 감지 과정에서 강력한 전송성을 제공하며, 'straight-through estimator'를 사용하여 각 과정에서의 그래디언트를 근사합니다.

- **Performance Highlights**: 실험 결과에 따르면, 협업 수watermarking은 높은 비트레이트 코덱과 DAC에서 거의 무시할 수 있는 지각적 저하를 초래하면서 신뢰성 있게 증가할 수 있습니다. 이 연구는 다양한 설정에서 코덱 비트레이트의 영향을 검토했으며, 높은 비트레이트에서는 EER이 5% 이하로 측정되었습니다.



### RingMo-Aerial: An Aerial Remote Sensing Foundation Model With A Affine Transformation Contrastive Learning (https://arxiv.org/abs/2409.13366)
- **What's New**: 본 논문에서는 Aerial Remote Sensing (ARS) 비전 분야를 위한 최초의 기초 모델 RingMo-Aerial을 소개합니다. 이 모델은 기존 공간 임지 원거리 센서(model)와 relevant 비교되며, 항공 독자 꿈을 위해 개선되었습니다.

- **Technical Details**: 이 모델은 Frequency-Enhanced Multi-Head Self-Attention (FE-MSA) 메커니즘과 affine transformation을 기반으로 한 Contrastive Learning (CL) 프리트레이닝 방법을 도입하여, ARS 특유의 기울어진 시점에서 작은 목표물 감지 능력을 향상시킵니다. 또한 ARS-Adapter라는 효율적인 파라미터 미세 조정 방법이 제안되어, 다양한 ARS 비전 작업에서 모델의 적응성과 효과성을 개선합니다.

- **Performance Highlights**: RingMo-Aerial은 여러 다운스트림 작업에서 SOTA (State Of The Art) 성능을 달성했으며, 이는 이 모델이 ARS 비전 작업의 성능 향상에 있어 실용적이고 효과적임을 나타냅니다.



### FPBoost: Fully Parametric Gradient Boosting for Survival Analysis (https://arxiv.org/abs/2409.13363)
- **What's New**: 본 연구에서는 기존 생존 분석 모델에서의 가정(strict assumptions)을 극복할 수 있는 혁신적인 생존 모델 FPBoost(Fully Parametric Gradient Boosting)를 제안합니다. 이 모델은 개별 완전 파라메트릭 위험 함수를 가중합(weighted sum)하여 구성합니다.

- **Technical Details**: FPBoost는 위험 함수(hazard function)를 모델링하기 위해 다수의 완전 파라메트릭 함수(heads)를 결합하며, 생존 가능성(survival likelihood)을 직접적으로 최적화합니다. 이는 그래디언트 부스팅(gradient boosting) 기법을 활용하여 이뤄집니다. FPBoost는 특히 우측 검열(right-censored) 환경에서 뛰어난 성능을 보입니다.

- **Performance Highlights**: FPBoost는 기존의 최신 생존 모델들과의 비교 실험에서 일관성(concordance) 및 보정(calibration) 지표 모두에서 모델의 성능을 개선하는 결과를 보였습니다. 다양한 데이터셋에서 FPBoost는 다수의 경우 다른 모델을 초초과 성능을 발휘했으며, 동등한 성능을 보여줬습니다.



### EmotionQueen: A Benchmark for Evaluating Empathy of Large Language Models (https://arxiv.org/abs/2409.13359)
Comments:
          Accepted to ACL 2024 (Findings)

- **What's New**: 이 논문에서는 LLMs(대규모 언어 모델)의 감정 지능을 평가하기 위한 새로운 프레임워크인 EmotionQueen을 제안합니다. 이를 통해 기본 감정 분석 작업을 넘어서는 보다 종합적인 감정 지능 평가가 가능합니다.

- **Technical Details**: EmotionQueen 프레임워크는 키 이벤트 인식(Key Event Recognition), 혼합 이벤트 인식(Mixed Event Recognition), 암묵적 감정 인식(Implicit Emotional Recognition), 그리고 의도 인식(Intention Recognition)의 4가지 주요 작업으로 구성되어 있습니다. 또한 PASS rate와 WIN rate라는 두 가지 메트릭을 도입하여 LLM의 감정 관련 진술에 대한 인식 및 응답 능력을 정량화합니다.

- **Performance Highlights**: 실험 결과, Claude2와 LLaMA-70B가 EmotionQueen 프레임워크에서 뛰어난 성능을 보였습니다. 이 연구는 LLM의 감정 지능에 대한 중요한 결론과 그 한계를 제시합니다.



### Recent Advancement of Emotion Cognition in Large Language Models (https://arxiv.org/abs/2409.13354)
- **What's New**: 이번 논문은 대형 언어 모델(LLM)의 감정 인지(emotion cognition)에 대한 최신 연구 동향을 종합적으로 조사하며, 감정 분류(emotion classification), 정서적으로 풍부한 응답 생성(emotionally rich response generation), 마음 이론(Theory of Mind) 평가 등 다양한 접근 방식을 소개하고 있습니다. 또한 주석이 달린 데이터에 대한 의존성과 감정 처리의 복잡성과 같은 챌린지를 강조합니다.

- **Technical Details**: 본 연구에서는 Ulric Neisser의 인지 심리학(cognitive psychology) 이론을 바탕으로 LLM의 감정 인지 관련 연구를 정리합니다. 감정 인지의 주요 문제로는 문제의 독특성(unique nature of the problem), 방법론적 복잡성(methodological complexity), 작업의 다양성(diversity of tasks)이 제시되며, 감정 평가(emotion evaluation)와 감정 향상(emotion enhancement)에 대한 두 가지 방향을 설명합니다.

- **Performance Highlights**: 논문은 또한 자율 학습(unsupervised learning) 접근 방식과 더 복잡하고 해석 가능한 감정 인지 LLM 개발을 위한 미래 연구 방향을 제시합니다. 감정 인지 능력을 향상시키기 위해 대조 학습(contrastive learning)과 같은 고급 방법들이 활용되고 있음을 언급하며, LLM의 감정 관련 작업에서의 성능을 개선할 수 있는 다양한 기법을 소개합니다.



### ID-Guard: A Universal Framework for Combating Facial Manipulation via Breaking Identification (https://arxiv.org/abs/2409.13349)
- **What's New**: 본 논문에서는 facial manipulation을 방지하기 위해 ID-Guard라는 새로운 프로ACTIVE defense framework를 제안합니다. 이 프레임워크는 단일 encoder-decoder 네트워크의 forward pass를 통해 특정 facial 이미지에 해당하는 일반적인 adversarial perturbation을 생성합니다.

- **Technical Details**: ID-Guard는 Identity Destruction Module (IDM)을 도입하여 가짜 이미지에서 식별 가능한 정보를 목표로 파괴합니다. 또한, perturbation 생성을 위한 multi-task learning 문제로서의 최적화를 통해 다양한 facial 조작에 대한 disruption을 개선합니다.

- **Performance Highlights**: ID-Guard는 여러 인기 있는 facial manipulation 알고리즘에 대한 방어 성능에서 뛰어난 결과를 보이며, 변조된 얼굴 이미지에서 식별 가능한 지역을 효과적으로 왜곡하여 관찰자가 개인의 정체성을 인식할 수 없도록 만듭니다.



### Imagine yourself: Tuning-Free Personalized Image Generation (https://arxiv.org/abs/2409.13346)
- **What's New**: 이번 연구에서는 기존 개인화된 이미지 생성 방식을 넘어서는 혁신적인 모델인 'Imagine yourself'를 소개합니다. 이 모델은 각 개별 사용자에 대한 조정 없이 모든 사용자가 공유하는 프레임워크에서 작동합니다.

- **Technical Details**: 'Imagine yourself'는 1) 이미지 다양성을 촉진하는 새로운 합성 쌍 데이터 생성 메커니즘, 2) 세 가지 텍스트 인코더와 완전하게 학습 가능한 비전 인코더를 포함하는 완전한 병렬 주의(Attention) 아키텍처, 3) 시각 품질의 경계를 점진적으로 향상시키는 새로운 거칠게부터 미세 조정까지의 다단계(Multi-stage) 방법론 등을 포함하고 있습니다.

- **Performance Highlights**: 'Imagine yourself' 모델은 개인화된 이미지 생성의 모든 측면에서 기존 SOTA(SOTA: State of the Art) 모델을 초월하여, 특히 아이덴티티 보존, 시각적 품질, 및 텍스트 정렬에서 두드러진 성능 개선을 보였습니다.



### A Novel Adaptive Fine-Tuning Algorithm for Multimodal Models: Self-Optimizing Classification and Selection of High-Quality Datasets in Remote Sensing (https://arxiv.org/abs/2409.13345)
- **What's New**: 이 논문에서는 멀티모달 대형 모델을 위한 적응형 미세 조정(Adaptive Fine-Tuning) 알고리즘을 제안합니다. 이 알고리즘은 데이터 집합을 의미 벡터 공간(Semantic Vector Space)으로 투영하는 첫 번째 단계와 MiniBatchKMeans 알고리즘을 사용한 자동 클러스터링 단계를 포함합니다.

- **Technical Details**: 데이터는 의미적 유사성이 높은 클러스터로 분류되어, 각 클러스터 내의 원본 데이터와 변형된 데이터 간의 변환 차이(Translational Difference)를 계산합니다. 이 차이는 데이터의 일반화 메트릭(Generalization Metric)으로 사용되어 고유한 일반화 잠재력을 가진 데이터를 선택하여 훈련에 활용합니다.

- **Performance Highlights**: 이 알고리즘을 사용하여 InternLM-XComposer2-VL-7B 모델을 훈련한 결과, 전체 데이터셋과 비교하여 다양한 원격 감지 지표에서 성능이 1%만 감소했습니다. 이 방법은 훈련 시간을 68.2% 단축시키면서 일반적인 능력을 크게 보존하였습니다. 모델은 UCMerced 및 AID 평가 데이터셋에서 각각 89.86 및 77.19의 점수를 기록하여 GeoChat 데이터셋보다 각각 5.43 및 5.16점 높았습니다.



### Time Awareness in Large Language Models: Benchmarking Fact Recall Across Tim (https://arxiv.org/abs/2409.13338)
- **What's New**: 이 논문에서는 대형 언어 모델(LLM)의 시간 인식 능력을 평가하기 위한 새로운 데이터셋을 소개합니다. 이 데이터셋은 시간과 관련된 사실을 다루는 모델의 성능을 측정할 수 있는 체계적인 벤치마크를 제공합니다. 이러한 접근은 현재 평가 방법에서 중요한 격차를 메우고, 향후 모델의 실제 응용 가능성을 개선하는 데 도움을 줍니다.

- **Technical Details**: 데이터셋은 2022년과 2023년의 1,150개의 이벤트로 구성되어 있으며, 각 이벤트는 정확한 월과 연도, 카테고리(비즈니스, 과학, 범죄 등)와 함께 제공됩니다. 또한 각 이벤트에는 4개의 패러프레이즈가 포함되어, 사실 회상 시 다양한 표현 방식에 대한 모델의 강건성을 테스트할 수 있도록 설계되었습니다. 이 데이터셋은 HuggingFace와 GitHub에 공개되어 연구자들이 시간 인식 연구를 진행할 수 있도록 지원합니다.

- **Performance Highlights**: 실험 결과, 인스트럭션 튜닝된 변형 모델들이 기본 모델에 비해 저조한 성능을 보였습니다. 예를 들어, Gemma-27B 모델은 30.96%의 Top-1 정확도를 기록했지만 인스트럭션 튜닝 후 17.57%로 떨어졌으며, 모델 크기에 따른 성능 차이도 명확하게 나타났습니다. 큰 모델이 일관되게 더 나은 성능을 보이며, Llama-3.1 70B Base는 39.74%의 Top-1 정확도를 나타냈습니다.



### SLaVA-CXR: Small Language and Vision Assistant for Chest X-ray Report Automation (https://arxiv.org/abs/2409.13321)
- **What's New**: 대규모 언어 모델(LLMs)의 성공에 영감을 받아 의료 분야에서 임상 의사들을 지원하기 위한 LLM 개발에 대한 연구가 증가하고 있습니다. 그러나 상용 모델을 사용하는 것에는 개인정보 보호 문제가 있으며, 오픈 소스 모델은 많은 계산 자원을 필요로 합니다. 이를 해결하기 위해 우리는 가슴 엑스레이(CXR) 보고서 자동화를 위한 오픈 소스 소형 언어 및 비전 도우미(SLaVA-CXR)를 제안합니다.

- **Technical Details**: 제안한 SLaVA-CXR는 Re3Training 방법론을 통해 훈련됩니다. 이 방법론은 방사선 의사의 인지 발달을 모사하며, Recognition(인식), Reasoning(추론), Reporting(보고) 세 가지 단계로 구성됩니다. 또한 RADEX라는 데이터 합성 방법을 통해 고품질의 다양한 훈련 데이터를 합성하여 개인정보 보호 규정을 준수합니다.

- **Performance Highlights**: 2.7B 백본을 기반으로 구축된 SLaVA-CXR는 기존의 대형 모델에 비해 6배 빠른 추론 효율을 달성하는 동시에 CXR 보고서 생성 및 요약 작업에서 최고의 성능을 보입니다.



### GAProtoNet: A Multi-head Graph Attention-based Prototypical Network for Interpretable Text Classification (https://arxiv.org/abs/2409.13312)
Comments:
          8 pages, 5 figues, submitted to COLING 2025

- **What's New**: 본 연구에서는 LM 인코더로 구축된 텍스트 분류 모델의 결정을 설명하기 위해 새롭게 설계된 GAProtoNet을 소개합니다. GAProtoNet은 입력 벡터와 프로토타입을 그래프의 노드로 간주하고, 멀티 헤드 그래프 어텐션(multi-head graph attention)을 활용하여 해석 가능한 프로토타입 표현을 학습합니다.

- **Technical Details**: GAProtoNet은 텍스트 임베딩, 프로토타입 레이어 및 그래프 어텐션 메커니즘으로 구성됩니다. 프로토타입 레이어는 입력 텍스트의 다양한 의미적 측면을 캡슐화하는 여러 개의 프로토타입 벡터를 형성하고, 그래프 어텐션 메커니즘은 텍스트 임베딩 벡터와 이웃 프로토타입 벡터 간의 관련성을 효과적으로 학습합니다. 이로 인해 결정을 내릴 때 어텐션 점수에 의해 가중치가 주어진 프로토타입의 선형 결합을 기반으로 합니다.

- **Performance Highlights**: 여러 공공 데이터셋에서의 실험 결과, GAProtoNet은 원래의 블랙 박스 모델의 정확도를 유지하면서도 더 나은 성능을 달성했습니다. GAProtoNet은 모든 비교 실험에서 최고의 정확도와 F1 점수를 기록하며, 텍스트 분류에서 결정적인 해석성을 제공합니다.



### OMG-RL:Offline Model-based Guided Reward Learning for Heparin Treatmen (https://arxiv.org/abs/2409.13299)
- **What's New**: 본 연구에서는 의료에서의 개인화된 의사결정을 위한 의약품 복용량 조정 문제에 대한 새로운 접근법을 제시합니다. 이는 전문가의 의도를 반영한 보상 함수(reward function)를 개발하고, Offline Model-based Guided Reward Learning (OMG-RL)이라는 기법을 도입하여 오프라인 강화학습을 진행합니다.

- **Technical Details**: OMG-RL은 제한된 데이터에서 전문가의 의도를 포함한 매개변수화된 보상 함수를 학습하여 에이전트의 정책(policy)을 향상시킵니다. 이 과정에서 역강화학습(inverse reinforcement learning, IRL) 기법을 사용하여, 다양한 상황과 환자 맞춤형 요소를 통합한 보상 함수를 효과적으로 정의합니다. 상태 전이를 다루기 위해 동적 모델(dynamic model)을 도입하여 시뮬레이션 및 예측 능력을 개선합니다.

- **Performance Highlights**: 제안된 방법론은 heparin 복용량 조정 문제를 해결하는 데 검증되었으며, 에이전트의 정책 구현이 크게 개선되었고, 활성화된 부분 트롬 보조 시간(aPTT) 지표에서 유의미한 긍정적 강화가 확인되었습니다. 이 접근법은 heparin 복용량 문제에 국한되지 않고 강화학습에 기반한 전반적인 약물 복용량 조정 작업에 광범위하게 적용될 수 있습니다.



### Time Distributed Deep Learning models for Purely Exogenous Forecasting. Application to Water Table Depth Prediction using Weather Image Time Series (https://arxiv.org/abs/2409.13284)
- **What's New**: 이번 연구에서는 Grana-Maira 유역의 수위를 예측하기 위해 외부 기상 정보(시간에 따른 이미지 시계열)를 사용하는 두 가지 딥러닝 모델(TDC-LSTM 및 TDC-UnPWaveNet)을 제안했습니다. 특히, 이전의 지표 데이터에 비해 국소적이고 공간적으로 분포된 데이터를 활용했습니다.

- **Technical Details**: 제안된 모델은 Time Distributed Convolutional Neural Network (TDC)와 함께 LSTM 계층으로 구성된 TDC-LSTM 모델과 새로운 WaveNet 아키텍처를 응용한 TDC-UnPWaveNet 모델로 나뉩니다. TDC-UnPWaveNet은 Channel Distributed layer를 사용하여 입력 채널 각각에 대해 동일한 연산을 적용하여 시퀀스 길이를 조정합니다.

- **Performance Highlights**: TDC-LSTM과 TDC-UnPWaveNet 모델 모두 뛰어난 예측 결과를 보였으며, TDC-LSTM은 편향을 줄이는 데 집중한 반면, TDC-UnPWaveNet은 시간적 동역학을 극대화하며 상관관계와 KGE를 최적화하는 데 중점을 두었습니다.



### A generalizable framework for unlocking missing reactions in genome-scale metabolic networks using deep learning (https://arxiv.org/abs/2409.13259)
- **What's New**: 이 논문에서는 GEnome-scale Metabolic 모델(GEMs)의 결측 반응 예측을 위한 새로운 딥러닝 기반 도구인 CLOSEgaps를 소개합니다. CLOSEgaps는 GEMs의 간극 문제를 하이퍼엣지(hyperedge) 예측 문제로 모델링하여 해결하는 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: CLOSEgaps는 메타볼릭 네트워크를 하이퍼그래프(hypergraph)로 매핑하고, 그 하이퍼탑올로지(hyper-topology)를 학습하여 가상의 반응(hypothetical reactions)을 활용해 결측 반응을 식별합니다. CLOSEgaps의 단계 과정은 GEM을 하이퍼그래프로 매핑, 부정적인 반응 샘플링, 특징 초기화, 특징 정제, 예측 또는 랭킹으로 이루어집니다.

- **Performance Highlights**: CLOSEgaps는 다양한 GEM에 대해 인위적으로 추가된 간극의 96% 이상을 정확히 메우며, 24개의 GEM에서 표현형 예측을 향상시키고 두 유기체에서 네 가지 주요 대사물질(젖산, 에탄올, 프로피온산, 석신산)을 생산하는 데 있어 상당한 개선을 보였습니다. CLOSEgaps는 여러 평가 지표에서 기존 모델보다 20% 이상 성능을 향상시켰으며, GEM 결측 반응 예측의 정확성을 증명했습니다.



### Emergent Collective Reproduction via Evolving Neuronal Flocks (https://arxiv.org/abs/2409.13254)
Comments:
          9 pages, 10 figures, conference

- **What's New**: 본 연구는 개체성의 진화적 전이(Evolutionary Transitions in Individuality, ETIs)를 이해하기 위한 새로운 인공 생명 프레임워크인 VitaNova를 소개합니다. VitaNova는 자기 조직화와 자연 선택이 복잡한 생식 집단의 출현을 시뮬레이션하도록 결합되어 있습니다.

- **Technical Details**: VitaNova는 Neural Networks에 의해 조정되는 'boid'라 불리는 개별 요원들이 환경에서 포식자와 공간적 제약에 대처하며 진화하는 과정을 모델링합니다. 이 시뮬레이션은 자기 조직화와 자연 선택의 힘이 결합하여 집합적 생식이 가능한 통합 단위를 형성하게 되는 과정을 보여줍니다.

- **Performance Highlights**: VitaNova 시뮬레이션에서 나타난 고리 구조는 자기 복제의 능력을 보여주며, 이러한 복잡한 집단 행동이 어떻게 자발적으로 발생하는지를 탐구합니다. 단순한 boid의 상호작용으로부터 자발적으로 복합적인 집단 구조가 Emergence되는 과정을 시뮬레이션을 통해 관찰했습니다.



### Leveraging Knowledge Graphs and LLMs to Support and Monitor Legislative Systems (https://arxiv.org/abs/2409.13252)
- **What's New**: 이 논문에서는 법률 지식 그래프(Legal Knowledge Graphs, LKs)와 대형 언어 모델(Large Language Models, LLMs)의 결합이 법률 프로세스를 지원할 수 있는 방법을 탐구합니다. 이 연구는 이탈리아 입법에 초점을 맞춘 Legis AI Platform을 개발하여 비전문가도 쉽게 사용할 수 있는 플랫폼을 제공합니다.

- **Technical Details**: 제안된 접근 방식은 도메인 특정 법률 지식 그래프를 사용하여 복잡한 입법 정보의 통합 및 검색을 가능하게 합니다. 이 시스템은 양적 분석 및 상호작용적 분석을 지원하며, LLM을 활용하여 이루어진 텍스트 품질의 자동 분석 및 보고서 생성을 포함합니다. 특히, 법률 텍스트의 메타데이터를 결합해 법률 품질 측정을 위한 언어적 지표를 계산합니다.

- **Performance Highlights**: Legis AI Platform은 법률 텍스트의 분석 뿐 아니라 새로운 법의 초안 작성을 지원하며, 전반적인 입법 시스템의 품질 및 복잡성을 분석하는 데 도움을 줍니다. 이 플랫폼은 사용자가 비교할 법률 집합을 선택하여 최신 법률의 품질을 평가할 수 있는 기능을 제공합니다.



### From Cognition to Precognition: A Future-Aware Framework for Social Navigation (https://arxiv.org/abs/2409.13244)
Comments:
          Social Navigation; Trajectory Prediction; Auxiliary Tasks

- **What's New**: 이 논문에서는 로봇의 사회적 내비게이션(SocialNav)을 개선하기 위해 Falcon이라는 강화 학습 아키텍처를 소개합니다. Falcon은 인간의 동작 예측을 통하여 로봇의 내비게이션을 더욱 안전하고 효율적으로 만드는 것을 목표로 합니다. 또한 SocialNav의 실제 평가를 위해 새로운 벤치마크인 Social-HM3D 및 Social-MP3D를 도입하였습니다.

- **Technical Details**: Falcon은 로봇이 사회적 적합성을 유지하며 인간의 경로를 차단하지 않도록 행동을 제안하는 미래 인식(Future-aware) SocialNav 프레임워크입니다. 주요 기술적 요소는 Social Cognition Penalty와 Spatial-Temporal Precognition Module로, 이는 로봇이 사회적 규범을 준수하고 잠재적인 충돌을 피하도록 학습하도록 합니다. 이를 통해 로봇은 한정된 공간에서의 동적 내비게이션 문제를 해결합니다.

- **Performance Highlights**: Falcon은 제안된 SocialNav 벤치마크에서 55%의 작업 성공률을 달성하며 약 90%의 개인 공간 준수를 유지하는 성과를 보였습니다. 이는 기존 방법론에 비해 뛰어난 성능을 나타냅니다.



### Relationship between Uncertainty in DNNs and Adversarial Attacks (https://arxiv.org/abs/2409.13232)
Comments:
          review

- **What's New**: 이번 논문에서는 Deep Neural Networks (DNNs)의 결과에 대한 불확실성과 적대적 공격(adversarial attack) 간의 관계를 분석합니다. DNNs가 많은 분야에서 최첨단 결과를 달성했지만, 이와 동시에 발생하는 결과의 불확실성 문제를 조명합니다.

- **Technical Details**: DNN의 불확실성은 모델 및 데이터의 제약으로부터 파생되며, 이는 적대적 공격에 의해 더욱 악화될 수 있습니다. 적대적 공격은 DNN에 왜곡된 입력을 제공하여 잘못된 예측을 유도하거나 모델의 불확실성을 증가시킵니다.

- **Performance Highlights**: DNNs는 자연어 처리, 패턴 인식, 예측 및 제어 최적화와 같은 다양한 분야에서 인간의 정확도를 초과하는 성과를 거두었지만, 이러한 불확실성을 해결하는 것은 여전히 도전 과제로 남아 있습니다.



### Redefining Data Pairing for Motion Retargeting Leveraging a Human Body Prior (https://arxiv.org/abs/2409.13208)
Comments:
          8 pages, 5 Figures, Accepted at IROS 2024

- **What's New**: MR.HuBo (Motion Retargeting leveraging a HUman BOdy prior)라는 새로운 방법을 제안하여 고품질의 로봇-인간 자세 데이터 수집을 편리하고 비용 효과적으로 구현했습니다. 기존 방법과는 다르게, 인간의 Motion Capture(MoCap) 자세를 로봇 자세로 변환하는 대신, 임의의 로봇 자세에서 시작하여 이를 인간 자세로 변환합니다.

- **Technical Details**: MR.HuBo는 대량의 인간 자세 데이터로부터 훈련된 인간 신체 우선(prior)을 활용하여 극단적인 자세를 필터링하는 기법을 포함합니다. 이 방법은 다양한 휴머노이드 로봇에 적용할 수 있으며, 시스템의 하이퍼파라미터를 설계 또는 최적화함으로써 구현할 수 있습니다. 학습된 두 단계의 모션 리타게팅 신경망이 고품질의 쌍 데이터를 통해 훈련되어, 실시간으로 동작 리타게팅이 가능합니다.

- **Performance Highlights**: 우리는 설계한 기법이 기존의 비(非)지도 학습 방법에 비해 성능이 뛰어남을 보여주었고, 특히 데이터 필터링 방법이 원시 및 노이즈가 있는 데이터로 훈련할 때보다 더 좋은 리타게팅 결과를 가져온다는 것을 입증했습니다.



### An adapted large language model facilitates multiple medical tasks in diabetes car (https://arxiv.org/abs/2409.13191)
- **What's New**: 이 연구에서는 당뇨병 특화된 대규모 언어 모델(LLM)을 개발하고 평가하기 위한 새로운 프레임워크를 소개했습니다. 특히, 다양한 당뇨병 작업에 대한 효과성을 검증하는 데 중점을 두었습니다.

- **Technical Details**: 이 연구는 데이터 수집, 필터링, 증강(augmentation) 및 정제를 포함하는 포괄적인 데이터 처리 파이프라인을 개발했습니다. 이를 통해 고품질의 당뇨병 전용 데이터셋을 생성하고 다양한 평가 기준을 처음부터 만들었습니다. 수집된 교육 데이터를 활용하여, 당뇨병 수치 특화 LLM 가족을 세밀하게 조정하여 다양한 당뇨병 작업을 처리하는 데 있어서 최첨단 성능을 보였습니다.

- **Performance Highlights**: 임상 연구 결과, 우리의 모델은 개인 맞춤형 의료 제공, 의료 교육 지원, 임상 작업의 효율성 향상 등 당뇨병 관리에 있어 다양한 응용 가능성을 보여주었습니다. 또한, 모든 사용자에게 데이터 기반 지원을 제공하여 임상 실습을 개선할 수 있는 잠재력이 강조되었습니다.



### Cooperative Resilience in Artificial Intelligence Multiagent Systems (https://arxiv.org/abs/2409.13187)
Comments:
          Supplementary material in this https URL

- **What's New**: 본 논문은 협력적 인공지능(cooperative AI) 분야에서의 '협력적 지속성(cooperative resilience)'의 개념을 명확히 정의하고, 이를 정량적으로 측정할 수 있는 방법론을 제안합니다. 이러한 접근은 인공지능 시스템의 지속적이고 적응 가능한 특성을 보장하기 위한 필수적인 발판이 됩니다.

- **Technical Details**: 협력적 지속성의 정의는 다음과 같습니다: 시스템은 개인들의 집합적 행동을 포함하여, 혼자 또는 함께, 방해가 되는 사건에 대처하고, 대비하고, 저항하고, 복구하며, 변화할 수 있는 능력입니다. 본 연구에서 제안된 방법론은 RL(강화 학습) 기반과 LLM(대형 언어 모델) 증강 자율 에이전트를 활용하여 시험됩니다.

- **Performance Highlights**: 실험 결과, 제안된 지속성 지표는 에이전트들의 저항, 적응 및 변화 능력을 평가하는 데 유용하며, 이는 기존의 다른 지표가 간과할 수 있는 요소입니다. 이러한 발견은 협력적 지속성의 정의와 측정 방법론을 수립하는 데 중요한 기초 데이터를 제공합니다.



### FreeAvatar: Robust 3D Facial Animation Transfer by Learning an Expression Foundation Mod (https://arxiv.org/abs/2409.13180)
Comments:
          11 pages, 11 figures

- **What's New**: 이번 논문에서는 기존의 방법이 해소하지 못한 미세한 감정을 포착하기 위해 기하학적 제약 없이 표현 표현만을 사용하여 3D 얼굴 애니메이션 전송을 가능하게 하는 새로운 방법, FreeAvatar를 제안합니다. 이 방법은 얼굴 이미지를 통해 학습한 표현의 표현력 있는 기초 모델과 애니메이션 전환 모델의 두 가지 주요 구성 요소로 이루어져 있습니다.

- **Technical Details**: FreeAvatar는 Masked Autoencoder (MAE)와 대비 학습을 통해 얼굴 특징을 학습해 정교한 표현 공간을 구축합니다. 이후 Expression-driven Multi-avatar Animator를 통해 이 표현을 3D 아바타의 얼굴 제어 매개변수로 변환하고, 훈련된 신경 렌더러를 통해 이를 이미지로 변환하는 과정을 구현합니다. 독립적인 동적 정체성 주입 모듈을 도입하여 여러 아바타를 단일 네트워크로 공동 훈련할 수 있도록 합니다.

- **Performance Highlights**: FreeAvatar는 실제 환경에서 수집된 이미지를 기반으로 한 실험에서 최첨단 성능을 달성하였으며, 다양한 시나리오에 유연하게 적용 가능하도록 설계되었습니다. 이 방법은 3D 얼굴 애니메이션 전송의 새로운 기준을 제시합니다.



### Unsupervised Attention-Based Multi-Source Domain Adaptation Framework for Drift Compensation in Electronic Nose Systems (https://arxiv.org/abs/2409.13167)
- **What's New**: 본 논문은 전자 코 시스템(E-nose)에서의 드리프트 보상을 위한 새로운 비지도 주의 기반 다중 소스 도메인 공유-개인 특징 융합 적응 프레임워크(AMDS-PFFA)를 제안합니다. 이 모델은 다양한 소스 도메인에서 수집된 라벨이 있는 데이터를 활용하여 라벨이 없는 타겟 도메인에서의 가스를 정확하게 식별할 수 있습니다.

- **Technical Details**: AMDS-PFFA 모델은 여러 소스 도메인에서 라벨이 있는 데이터를 이용해 등록하고, 공유-개인 특징 융합(shared-private feature fusion)을 통해 가스 센서 드리프트를 효과적으로 상쇄합니다. 이는 로컬 최대 평균 불일치(local maximum mean discrepancy, LMMD)를 정제하고 분류기 간 확률 예측 결과를 조화롭게 만들어 정확도를 향상시킵니다.

- **Performance Highlights**: AMDS-PFFA 모델은 UC Irvine (UCI) 표준 드리프트 가스 데이터셋과 자가 개발한 E-nose 시스템의 드리프트 신호 데이터에서 83.20%와 93.96%의 최대 평균 가스 인식 정확도를 달성하며, 기존 드리프트 보상 방법들과 비교해서도 가장 높은 성과를 보였습니다.



### Morphology and Behavior Co-Optimization of Modular Satellites for Attitude Contro (https://arxiv.org/abs/2409.13166)
Comments:
          The paper was accepted as an oral presentation by the 75th International Astronautical Congress, Milan, Italy

- **What's New**: 본 연구에서는 모듈형 위성의 형상(morphology)과 제어(control) 정책을 동시에 최적화하는 새로운 그래디언트 기반 접근 방식을 소개합니다. 이는 태세 제어(attitude control) 미션의 성능과 효율성을 향상시키는데 기여합니다.

- **Technical Details**: 연구에서는 마르코프 결정 프로세스(MDP)를 기반으로 한 RL(강화학습) 접근법을 통하여 모듈형 위성의 디자인과 제어를 통합하여 최적화합니다. TD3(Twin Delayed Deep Deterministic Policy Gradient) 알고리즘을 사용하여 연속 행동 공간에서 모듈형 위성의 디자인을 최적화합니다.

- **Performance Highlights**: 몬테카를로 시뮬레이션 결과, 제안된 공동 최적화(co-optimization) 접근 방식이 기존의 진화 기반 접근 방식보다 더 나은 미션 성능을 보였습니다. 이는 모듈형 위성의 차별화된 기능과 향상된 유지 관리 가능성을 입증합니다.



### Towards Efficient Neuro-Symbolic AI: From Workload Characterization to Hardware Architectur (https://arxiv.org/abs/2409.13153)
Comments:
          14 pages, 11 figures, 7 tables; IEEE Transactions on Circuits and Systems for Artificial Intelligence (TCASAI), 2024

- **What's New**: 이 논문은 신경-상징 AI(Neuro-Symbolic AI)의 작업 부하 특성과 잠재적 아키텍쳐를 체계적으로 분석하고 평가하여, 다음 세대 인지 AI 시스템의 개발을 위한 기반을 마련합니다. 특히, 신경과 상징 기반 메서드를 결합하여 해석 가능성, 견고성 및 신뢰성을 향상시키고, 더욱 적은 데이터로 학습할 수 있는 가능성을 탐구합니다.

- **Technical Details**: 연구에서는 신경-상징 AI 알고리즘을 분류하고, CPU, GPU 및 엣지 SoC에서의 런타임, 메모리, 계산 연산자, 희소성(sparsity) 및 시스템 특성을 실험적으로 평가합니다. 신경-상징 모델은 메모리 바운드(memory-bound)와 데이터 종속성(data dependencies) 문제로 인해 기존 하드웨어에서 비효율성을 겪습니다. 이 논문은 이러한 문제를 해결하기 위해 크로스 레이어 최적화(cross-layer optimization) 솔루션을 제안하고 있습니다.

- **Performance Highlights**: 신경-상징 아키텍처는 공간-시간 추론(task)에서 98.8%의 정확성을 달성하여 인간 성능(84.4%) 및 다른 AI 모델들을 초월하는 뛰어난 성능을 보였습니다. 또한, 이 시스템은 협업 로봇, 혼합 현실 시스템, 인간-AI 상호작용 등 다양한 응용 분야에서 신뢰성과 해석 가능성을 증진할 수 있는 잠재력을 가지고 있습니다.



### The Impact of Feature Embedding Placement in the Ansatz of a Quantum Kernel in QSVMs (https://arxiv.org/abs/2409.13147)
Comments:
          9 pages including references and appendix, 7 figures

- **What's New**: 이 논문에서는 Quantum Embedding Kernels (QEKs)의 아키텍처 설계에 대한 중요한 논의가 이루어졌으며, 기존의 아키텍처 스타일이 예상과 다르게 작동함을 보여주고 새로운 데이터-위빙(data-weaved) 커널 아키텍처를 제안합니다.

- **Technical Details**: 본 연구는 QEK의 다양한 아키텍처 스타일을 분석하고, 데이터-우선(data-first) 아키텍처와 데이터-후순위(data-last) 아키텍처를 비교합니다. 입력 데이터의 분포에 따라 분류 결과에 영향을 미치는 양자 커널의 구조적 구성요소인 매개변수화된 레이어와 특성 종속 레이어의 배열 방식을 논의합니다. '게이트 소거 문제(gate erasure bug)'를 해결하기 위해 제안된 새로운 데이터-위빙 아키텍처는 기존 구조에 비해 더 적은 게이트 수를 유지하면서도 동등한 성능을 제공합니다.

- **Performance Highlights**: 데이터-위빙 아키텍처는 다양한 벤치마크 테스트에서 기존 QEK 아키텍처보다 탁월한 성능을 보였습니다. 이 새로운 접근법은 최적화 과정에서의 속도 향상 및 낮은 게이트 수 측면에서 QML 모델 개선을 가능하게 하여 기존의 머신 러닝 모델에 대해 잠재적 이점을 제공합니다.



### Learning to Compare Hardware Designs for High-Level Synthesis (https://arxiv.org/abs/2409.13138)
Comments:
          Published in MLCAD 2024

- **What's New**: compareXplore는 고급 합성(High-Level Synthesis, HLS)에서 하드웨어 디자인을 효과적으로 비교하여 최적화하는 혁신적인 접근 방식입니다. 이 방법은 상대적 선호와 절대 성능을 모두 포착할 수 있는 하이브리드 손실 함수와 디자인 간의 정보 차이를 강조하는 노드 차이 주의 모듈을 도입합니다.

- **Technical Details**: compareXplore는 두 단계의 디자인 공간 탐색(DSE) 접근 방식을 사용하며, 첫 번째 단계에서는 점별 성능 예측 모델을 활용하여 하위 최적 디자인을 신속하게 제거하고, 두 번째 단계에서는 쌍별 비교 모델을 통해 남은 후보 디자인의 성능을 정밀 검증합니다. 이 모델은 그래프 신경망(Graph Neural Networks, GNNs) 기반으로 하드웨어 디자인을 학습합니다.

- **Performance Highlights**: compareXplore는 기존의 SOTA(state-of-the-art) 방법에 비해 순위 메트릭에서 상당한 향상을 이뤘으며, 선택된 디자인에 대해 높은 품질의 HLS 결과를 생성했습니다. 실험 결과 compareXplore는 하드웨어 디자인의 질에서 일관되게 더 나은 성능을 보였습니다.



### Interpret the Predictions of Deep Networks via Re-Label Distillation (https://arxiv.org/abs/2409.13137)
Comments:
          Published by IEEE ICME 2021

- **What's New**: 본 연구에서는 딥 네트워크의 예측을 해석하기 위한 're-label distillation' 접근법을 제안합니다. 이 방법은 최신 Self-supervision 방식을 통해 입력과 예측 간의 직접적인 지도를 학습합니다.

- **Technical Details**: 이 방법은 VAE(Variational Autoencoder) 서브스페이스에 이미지를 투사하여 잠재 벡터를 무작위로 교란하여 합성 이미지를 생성합니다. 생성된 합성 이미지는 딥 네트워크가 예측한 레이블을 기준으로 둘 중 하나의 클래스에 주석을 달리 합니다. 이후 레이블링된 합성 이미지를 기반으로 선형 학생 모델을 학습합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법의 효과가 정성적 및 정량적으로 검증되었습니다. 이 're-label distillation' 접근법은 딥 네트워크 예측의 해석 가능성을 높이는 데 탁월한 성능을 보입니다.



### Are Large Language Models Good Essay Graders? (https://arxiv.org/abs/2409.13120)
- **What's New**: 이번 연구에서는 자동화된 에세이 채점 자동화 시스템(AES)에서 두 가지 대형 언어 모델(Large Language Models, LLMs)인 ChatGPT와 Llama의 성능을 평가하였습니다. LLMs의 점수가 인간 채점자와 얼마나 일치하는지를 중심으로 다루고 있으며, 제로샷(zero-shot) 및 퓨샷(few-shot) 학습 접근 방식과 다양한 프롬프트 기법들을 고려하였습니다.

- **Technical Details**: 이 연구에서는 ASAP 데이터셋을 사용하여 LLM이 제공하는 수치 점수를 인간 평가자의 점수와 비교하였습니다. ChatGPT는 Llama에 비해 전반적으로 더 낮은 점수를 부여하며, 인간 평가자와의 상관관계도 낮은 것으로 나타났습니다. 이 에세이 채점 과정에서 글자 수, 연결어 사용, 문법 오류 수 등의 요소가 효과가 없음을 발견하였습니다.

- **Performance Highlights**: Llama 3의 성능이 이전 모델보다 일반적으로 더 좋다는 결과를 보고하며, LLM들이 인간 채점자와의 일관성을 보이지 않으나, 향후 에세이 채점을 보조하는 도구로서 사용 가능성이 있다는 점에서 긍정적인 결론을 내립니다.



### Personalized 2D Binary Patient Codes of Tissue Images and Immunogenomic Data Through Multimodal Self-Supervised Fusion (https://arxiv.org/abs/2409.13115)
- **What's New**: 이번 연구는 MarbliX라는 새로운 멀티모달 프레임워크를 소개합니다. 이 프레임워크는 병리 이미지와 면역유전체 시퀀싱 데이터를 결합하여, 환자 정보를 압축된 이진 코드인 '모노그램'으로 만들어냅니다.

- **Technical Details**: MarbliX는 심층 학습(deep learning) 기법을 활용하여 병리학적 전체 슬라이드 이미지(whole slide images, WSIs)와 면역 세포 시퀀싱 데이터를 통합합니다. 이러한 통합은 두 가지의 데이터 모달리티를 아우르는 고차원의 정보를 다루기 위한 효율적인 방법으로 작동합니다.  시스템은 TCGA(The Cancer Genome Atlas)로부터 수집된 데이터셋을 통해 평가되었으며, 이는 535개의 폐 선암 및 510개의 폐 편평세포암(LUSC) 사례를 포함합니다.

- **Performance Highlights**: MarbliX의 실험 결과, 정밀한 진단을 가능하게 하고 평가의 변동성을 줄이며 맞춤 치료 옵션을 확대할 수 있는 잠재력을 지니고 있음을 보여주었습니다. 특히 폐와 신장 암에 대한 데이터 세트를 통해 이 시스템의 정확성과 효율성이 입증되었습니다.



### Evolution and challenges of computer vision and deep learning technologies for analysing mixed construction and demolition was (https://arxiv.org/abs/2409.13112)
- **What's New**: 건설 및 철거 폐기물(C&DW)의 자동 및 신속 인식 개선을 위해 컴퓨터 비전(Computer Vision), 인공지능(AI), 로보틱스(Robotics), 사물인터넷(IoT) 기술이 통합되고 있습니다. 이 연구는 상업적 환경에서 혼합되고 고도로 오염된 C&DW의 인식 성능에 대한 부족한 연구를 보완하고 있습니다.

- **Technical Details**: 이 논문은 시드니의 C&DW 자원 회수 시설(MRF)에서의 경험을 바탕으로 하여, 혼합 C&DW 관리 시스템 개발의 도전 과제와 기회를 탐구합니다. 여러 C&DW 분석 기법을 리뷰한 결과, 딥러닝(DL) 기반의 비주얼 방법이 최적의 솔루션으로 여겨졌으며, C&DW 분석을 위한 센서 및 카메라 기술, 객체 탐지(Object Detection) 및 물질 분할(Material Segmentation)에 중점을 둔 DL 알고리즘의 발전도 분석하였습니다.

- **Performance Highlights**: C&DW 데이터셋의 생성과 관리, 기술적 및 상업적 문제들, 그리고 혼합 C&DW 분석의 연구 동향과 미래 방향성을 다루며, C&DW 관리의 효율성을 개선하기 위한 통찰을 공유합니다. 이 연구는 해당 분야의 지속적인 연구 및 개발에 중요한 기여를 할 것입니다.



### ERIC: Estimating Rainfall with Commodity Doorbell Camera for Precision Residential Irrigation (https://arxiv.org/abs/2409.13104)
Comments:
          BuildSys 2024

- **What's New**: 본 연구에서는 기존의 기상 기반 관개 시스템에 대한 개선안을 제시합니다. 특히, 이 시스템은 기성품 초인종 카메라를 활용하여 정확한 지역 강수량을 측정합니다. 이러한 접근법은 추가 하드웨어 배포 없이 저비용 솔루션을 제공합니다.

- **Technical Details**: 본 논문에서 제안한 ERIC 시스템은 가벼운 신경망 모델을 사용하여 반사 기반 비주얼 피처 및 오디오 피처를 결합해 강수량을 추정합니다. 시스템은 Raspberry Pi 4를 기반으로 하여 설계되었으며, 750시간 이상의 영상을 수집하여 데이터의 정확성을 검증합니다.

- **Performance Highlights**: ERIC 시스템은 초당 약 5mm의 정확도로 강수량을 추정하며, 월 9,112갤런의 물을 절약하여 약 28.56달러의 유틸리티 비용 절감을 달성했습니다.



### DenoMamba: A fused state-space model for low-dose CT denoising (https://arxiv.org/abs/2409.13094)
- **What's New**: 이번 연구에서는 DenoMamba라는 새로운 저선량 컴퓨터 단층 촬영(DLCT) 이미지 잡음 제거 방법을 소개합니다. 이 방법은 상태-공간 모델링(SSM)을 활용하여 의학 이미지에서 단기 및 장기 컨텍스트를 효율적으로 캡처합니다.

- **Technical Details**: DenoMamba는 인코더-디코더 구조를 가진 모래시계 아키텍처를 따르며, 공간 SSM 모듈과 새로운 채널 SSM 모듈을 통해 각각의 컨텍스트를 인코딩합니다. 두 모듈의 특징 맵은 저수준 입력 특징과 함께 합쳐집니다. DenoMamba는 전통적인 CNN 모델과 유사한 모델 복잡도로 고해상도 CT 이미지의 복잡한 특성을 효율적으로 처리합니다.

- **Performance Highlights**: DenoMamba는 25% 및 10%의 방사선 용량 감소로 수집된 LDCT 데이터셋에 대한 포괄적인 실험을 통해, 평균 PSNR(1.4dB), SSIM(1.1%), RMSE(1.6%)의 성능 향상을 보여주며 기존 최첨단 방법들보다 우수한 이미지 품질을 제공합니다.



### Guided Profile Generation Improves Personalization with LLMs (https://arxiv.org/abs/2409.13093)
Comments:
          EMNLP 2024 Findings

- **What's New**: 이 연구에서는 Guided Profile Generation (GPG)이라는 새로운 방법론을 제시하여 개인화(Personalization) 작업에서 LLMs의 성능을 개선하는 방법에 대해 다룹니다. GPG는 자연어로 개인 프로필을 생성하는 일반적인 방법으로, LLM이 개인 맥락을 해석하고 이를 기반으로 고품질의 자연어 개인 프로필을 생성할 수 있도록 돕습니다.

- **Technical Details**: GPG는 두 가지 주요 과제를 해결하는 데 초점을 맞추고 있습니다. 첫 번째는 개인 맥락의 복잡성과 핵심 정보의 희소성(sparsity) 문제입니다. 두 번째는 일반화(generalization)와 개인화 사이의 균형을 유지하는 것입니다. GPG는 개인 맥락을 분석하고 특정 질문에 답하여 자연어 개인 프로필을 생성하는 과정을 포함한 다단계 프로세스를 사용하며, 앙상블 모델을 통해 요구되는 작업에 대해 응답을 합니다.

- **Performance Highlights**: GPG는 다양한 작업에서 LLM의 개인화 성능을 향상시켰습니다. 특히, 온라인 구매에서 개인 선호도 예측 정확도를 37% 개선하였고, 트윗의 텍스트 패러프레이징에서 METEOR 점수를 2.24 향상시켰습니다. 이러한 결과는 GPG가 LLM의 개인화 능력을 크게 개선할 수 있음을 보여줍니다.



### Interpretable Action Recognition on Hard to Classify Actions (https://arxiv.org/abs/2409.13091)
Comments:
          5 pages, This manuscript has been accepted at the Human-inspired Computer Vision (HCV) ECCV 2024 Workshop. arXiv admin note: text overlap with arXiv:2107.05319

- **What's New**: 이번 연구는 비디오 활동 인식 분야에서 인간과 유사한 해석 가능한 모델을 통해 비디오 내 복잡한 활동을 이해하는 방법을 탐구합니다. 기존 모델에서 3D 정보를 추가하여 향상된 정확도를 달성하였으며, 특히 'Depth relations' 추가가 성능 개선에 기여하였습니다.

- **Technical Details**: 모델은 객체 및 손의 위치와 움직임을 사용하여 활동을 인식합니다. 두 가지 주요 확장으로는: (1) 'Container'와 'NotContainer' 구분을 위한 최첨단 물체 탐지 모델을 재훈련했고, (2) 개별 물체의 깊이를 추정하기 위한 'depth estimation' 모델을 적용하여 기존의 관계를 확대했습니다.

- **Performance Highlights**: 'Depth relations'를 추가했음에도 불구하고 전체적인 성능은 주류 딥러닝 접근 방식에는 미치지 못했습니다. 실험 결과, 'Container' 탐지기는 성능을 개선하지 못했지만 깊이 관계의 추가가 유의미한 성과로 나타났습니다.



### FedAT: Federated Adversarial Training for Distributed Insider Threat Detection (https://arxiv.org/abs/2409.13083)
Comments:
          10 pages, 7 figures

- **What's New**: 이번 연구는 다수의 장소에서의 Insider Threat Detection (ITD)을 위한 Federated Learning (FL) 기반의 새로운 다중 클래스 ITD 패러다임을 제안합니다. 이 패러다임은 비독립적이고 동일하게 분포하지 않은 (non-IID) 데이터 분포를 고려하여 각 클라이언트에서의 내부 위협을 탐지합니다.

- **Technical Details**: 제안된 접근법은 Generative Adversarial Network (GAN) 기반의 Federated Adversarial Training (FedAT) 방법을 사용하여 각 클라이언트 간의 비균형한 데이터 분포에서 발생하는 극단적인 데이터 왜곡을 완화합니다. 또한, Self-normalized Neural Network 기반의 Multi-Layer Perceptron (SNN-MLP) 모델을 사용하여 ITD의 효율성을 증가시킵니다.

- **Performance Highlights**: 다양한 FedAvg 및 FedProx와 같은 연합 최적화 알고리즘을 비교하여 제안된 FedAT 기반 ITD 모델의 성능 향상을 입증하였습니다. 이 연구는 FL과 GAN을 결합하여 데이터 분포의 불균형 문제와 프라이버시 문제를 모두 해결하는 방향으로 기여하고 있습니다.



### AutoVerus: Automated Proof Generation for Rust Cod (https://arxiv.org/abs/2409.13082)
- **What's New**: 이 논문은 Rust 코드의 정당성을 자동으로 증명하는 AutoVerus를 제안합니다. 이는 기존의 LLM 기반 코드 생성 방식에 비해 정당성 증명 생성의 발전이 더디기 때문에, 새로운 가능성을 열어주는 중요한 연구로 평가됩니다.

- **Technical Details**: AutoVerus는 LLM을 이용해 Rust 코드에 대한 정당성 증명을 생성합니다. 주요 특징으로는 Verus라는 검증 도구와의 호환성을 가지며, LLM 에이전트 네트워크를 통해 증명 생성, 수정 및 디버깅 단계를 모방합니다. 국제적으로 150개의 비사소적인 증명 작업에 대한 벤치마크 스위트를 구축하여 성능을 평가했습니다.

- **Performance Highlights**: 평가 결과, AutoVerus는 90% 이상의 증명 작업에서 자동으로 올바른 증명을 생성할 수 있으며, 그 중 절반 이상은 30초 또는 3번의 LLM 호출 이내에 해결되었습니다.



### Fear and Loathing on the Frontline: Decoding the Language of Othering by Russia-Ukraine War Bloggers (https://arxiv.org/abs/2409.13064)
Comments:
          15 pages

- **What's New**: 이번 연구에서는 'othering'(타자화)의 개념을 분석하여, 특정 집단을 본질적으로 다르게 묘사하는 행위가 어떻게 존재론적 위협으로 이어지는지를 다룹니다. 특히, 온라인 언어 사용과 선전에서 이 현상을 정량화할 수 있는 새로운 컴퓨테이셔널 프레임워크(computational framework)를 소개합니다.

- **Technical Details**: 이 연구는 대형 언어 모델(large language models, LLMs)을 활용하여 다양한 맥락에서의 타자화 현상을 정량화합니다. 기존의 적대감의 언어적 지표를 넘어서, 갈등 중 타자화가 어떻게 확대되는지, 도덕적 언어(moral language)와 어떻게 상호작용하는지를 분석합니다. 또한, 텔레그램(Telegram) 전쟁 블로거와 갭(Gab)에서의 정치적 논의와 같은 실제 데이터를 적용하여 연구하였습니다.

- **Performance Highlights**: 본 프레임워크는 타자화의 역학을 심층적으로 이해하는 데 중점을 두며, 사회적 응집력에 미치는 부정적인 영향을 완화하기 위한 필수 도구를 제공합니다. 특히, 위기 상황에서 타자화가 큰 주목을 받음을 발견하였으며, 이를 통해 관련된 문제들을 보다 효과적으로 실시간으로 대응할 수 있는 가능성을 제시합니다.



### Comprehensive Overview of Artificial Intelligence Applications in Modern Industries (https://arxiv.org/abs/2409.13059)
- **What's New**: 본 논문은 인공지능(AI)의 최신 응용을 의료, 금융, 제조, 소매의 네 가지 주요 산업 부문에서 분석하고 있습니다.

- **Technical Details**: 각 산업 부문은 특정한 도전과제가 있으며, 이를 해결하기 위해 사용되는 AI 기술들은 데이터 분석(data analysis), 기계 학습(machine learning), 그리고 예측 모델링(prediction modeling) 등을 포함하고 있습니다. 논문은 이러한 기술들이 비즈니스 결과와 사회적 복지(social welfare)에 미치는 측정을 통해 그 영향을 논의합니다.

- **Performance Highlights**: AI의 통합은 윤리적 고려사항(ethical considerations) 및 앞으로의 AI 개발 미래(future trajectory of AI development)와 같은 다양한 함의를 제시하면서, 경제 성장(economic growth)을 견인할 잠재력을 지니고 있으나, 신중하게 관리해야 할 도전 과제도 존재함을 강조하고 있습니다.



### LLM Surgery: Efficient Knowledge Unlearning and Editing in Large Language Models (https://arxiv.org/abs/2409.13054)
- **What's New**: 본 논문에서는 LLM(대형 언어 모델)의 문제적 및 구식 정보를 효과적으로 잊고 새로운 지식을 통합할 수 있는 방법, 즉 LLM 수술(LLM Surgery)이라는 프레임워크를 제안합니다.

- **Technical Details**: LLM 수술은 (1) 문제적 정보가 포함된 비학습 데이터셋에서 역경량화를 수행하고, (2) 새로운 정보가 포함된 업데이트 데이터셋에 대해 경량하강을 수행하며, (3) 변하지 않은 작은 하위 집합인 유지 데이터셋에서 KL 발산(KL divergence)을 최소화하는 세 가지 목표 함수(component objective function)를 최적화합니다. 이를 통해 모델의 성능을 보존하면서도 효과적으로 학습할 수 있도록 합니다.

- **Performance Highlights**: Llama2-7B 모델을 사용하여 우리의 방법론이 unlearn 집합에서 큰 잊기를 달성하고, update 집합에서 20%의 정확도 향상 및 retain 집합에서는 성능 유지에 성공했음을 보여줍니다.



### iCost: A Novel Instance Complexity Based Cost-Sensitive Learning Framework for Imbalanced Classification (https://arxiv.org/abs/2409.13007)
- **What's New**: 이번 연구에서 제안하는 iCost는 인스턴스 복잡성 기반의 비용 민감 학습 알고리즘으로, 불균형한 데이터에 대해 더 효과적인 접근 방식을 제공합니다.

- **Technical Details**: 제안된 방법은 소수 클래스 인스턴스를 난이도에 따라 분류하고, 교차 클래스 객체의 수를 기준으로 비용을 고르게 분배하는 것입니다. 이를 통해 연결된 샘플의 모양을 유지하고, 더 높은 분류 오차 비용을 부여하여 어려운 인스턴스에 더 집중하게 합니다.

- **Performance Highlights**: 66개의 불균형 데이터셋에서의 실험 결과, 기존의 비용 민감 학습 기법들보다 상당한 성능 향상을 보여주었습니다.



### Introducing the Large Medical Model: State of the art healthcare cost and risk prediction with transformers trained on patient event sequences (https://arxiv.org/abs/2409.13000)
Comments:
          10 pages, 10 figures

- **What's New**: 이번 논문은 미국의 헬스케어 지출이 5조 달러에 이르는 가운데, 25%가 낭비로 추정되는 상황에서 환자 관리와 헬스케어 관리의 예측 능력을 향상시키기 위해 고안된 Large Medical Model (LMM)이라는 Generative Pre-trained Transformer (GPT) 모델을 소개합니다.

- **Technical Details**: LMM은 1억 4천만 개의 장기적인 환자 청구 기록에서 수집된 의료 사건 시퀀스에 대한 모델로, 의료 용어 시스템에서 구축된 전문 용어를 사용하여 훈련되었습니다. 이 모델은 의료 비용 예측 및 잠재적 위험 요소 식별에서 뛰어난 성능을 보여주는 실험 및 검증이 이루어졌습니다.

- **Performance Highlights**: LMM은 기존 상용 모델에 비해 비용 예측에서 14.1%, 최고의 연구 트랜스포머 모델에 비해 만성 질환 예측에서 1.9% 향상된 성능을 보여주며, 복잡한 의료 조건 내의 정교한 패턴을 인식하고 환자 관리에서 새로운 관계를 파악할 수 있는 능력을 가지고 있습니다.



### VCAT: Vulnerability-aware and Curiosity-driven Adversarial Training for Enhancing Autonomous Vehicle Robustness (https://arxiv.org/abs/2409.12997)
Comments:
          7 pages, 5 figures, conference

- **What's New**: 이번 논문에서는 자율주행차(AV)가 복잡한 교통 환경에서 안전한 작동을 보장하기 위한 새로운 접근 방식을 제안합니다. 기존의 적대적 훈련(adversarial training) 방식의 한계를 극복하기 위해 VCAT(Vulnerability-aware and Curiosity-driven Adversarial Training)라는 프레임워크를 도입하여 AV의 내재적인 취약점을 탐구하며, 공격자가 미지의 영역을 탐색하도록 유도합니다.

- **Technical Details**: VCAT 프레임워크는 두 단계로 구성되어 있습니다: 공격 훈련과 방어 훈련. 공격자는 surrogate network를 통해 AV의 취약성을이고, 무작위 네트워크 증류(random network distillation)를 사용하여 새로운 환경을 탐색하는 내재적 보상을 형성합니다. 이는 AV가 이전에 경험하지 못한 상황에서 더 나은 방어 전략을 수립하는데 기여합니다.

- **Performance Highlights**: 실험 결과, VCAT 훈련 방법론은 기존의 방법들보다 AV의 제어 능력을 현저히 향상시켰고, 충돌률(crash rates)을 크게 감소시켰습니다. 이 연구는 자율주행차의 안전성을 높이는 데 기여할 것으로 기대됩니다.



### pyrtklib: An open-source package for tightly coupled deep learning and GNSS integration for positioning in urban canyons (https://arxiv.org/abs/2409.12996)
- **What's New**: 이 논문은 pyrtklib이라는 파이썬 바인딩을 소개하여, 전통적인 GNSS 알고리즘을 현대의 딥러닝 기술과 통합하는 혁신적인 접근 방식을 제공합니다. RTKLIB의 모든 기능을 Python에서 활용할 수 있게 하여, 딥러닝 기반 GNSS 알고리즘의 프로토타입 및 구현을 신속하게 진행할 수 있습니다.

- **Technical Details**: pyrtklib는 RTKLIB의 오픈소스 기능을 Python으로 연결하여, C의 속도와 Python의 편리함을 결합합니다. 또한, 딥러닝 서브시스템을 통합하여 GNSS 위치 결정 과정에서 가중치와 바이어스를 예측합니다. 메타 프로그래밍 기법을 이용하여 RTKLIB의 헤더 파일을 Python 바인딩 코드로 자동으로 변환합니다.

- **Performance Highlights**: pyrtklib를 사용한 연구에서, GNSS 측정 정확도를 개선하기 위해 가중치 및 바이어스를 예측하는 네트워크를 개발하였으며, 이 예측 값을 가중 최소제곱 곱셈 과정에 적용하여 측정의 정확도를 향상시켰습니다. 이를 통해 GNSS 위치 결정의 정확성을 획기적으로 높일 수 있는 가능성을 보여주고 있습니다.



### Improving generalisability of 3D binding affinity models in low data regimes (https://arxiv.org/abs/2409.12995)
Comments:
          17 pages, 10 figues

- **What's New**: 이번 연구에서는 PDBBind 데이터셋의 새로운 분할 방법을 도입하여, 훈련 세트와 시험 세트 간의 유사성 누수를 최소화하고 다양한 모델 아키텍처 간의 공정하고 직접적인 비교를 가능하게 했습니다. 특히, 3D 글로벌 모델이 단백질-국소 모델보다 낮은 데이터 상황에서 성능이 우수하다는 것을 보여주었습니다.

- **Technical Details**: 본 연구에서는 GNN(그래프 신경망)을 활용하여, 양자역학적 데이터에 의한 감독형 사전 훈련(supervised pre-training)과 작은 분자의 확산을 통한 비감독형 사전 훈련(unsupervised pre-training)을 통해 성능을 향상시키는 방법을 제안했습니다. 또한, 입력 그래프에서 수소 원자를 명시적으로 모델링하는 것이 중요한 요소로 밝혀졌습니다.

- **Performance Highlights**: 낮은 데이터 환경에서 3D 글로벌 모델은 단백질에 특정한 국소 모델보다 현저히 우수한 성능을 보였으며, 더 많은 데이터가 주어질수록 국소 모델의 성능이 빠르게 향상되었습니다. 특히, 사전 훈련 기법을 통해 GNN의 성능이 개선됨을 확인했습니다.



### Performance and Power: Systematic Evaluation of AI Workloads on Accelerators with CARAML (https://arxiv.org/abs/2409.12994)
Comments:
          To be published in Workshop Proceedings of The International Conference for High Performance Computing Networking, Storage, and Analysis (SC-W '24) (2024)

- **What's New**: 이 논문은 CARAML 벤치마크 스위트를 소개하며, 이는 다양한 하드웨어 가속기에서 대형 언어 모델과 컴퓨터 비전 모델의 훈련 성능과 에너지 소비를 평가하는 데 사용된다. NVIDIA, AMD, Graphcore와 같은 시스템에서의 평가를 포함해야 한다.

- **Technical Details**: CARAML은 ML(머신러닝) 작업을 다양한 하드웨어 아키텍처에서 평가하기 위한 Compact, Automated, Reproducible Assessment framework이다. jpwr라는 사용자 지정 전력 측정 도구를 사용하여 에너지를 측정하며, 이는 PyTorch로 구현된 GPT 기반의 대형 언어 모델 훈련과 TensorFlow로 구현된 ResNet50 모델 훈련에서 수집된 성능 데이터와 에너지 측정을 포함한다.

- **Performance Highlights**: CARAML을 통해 수집된 데이터는 다양한 배치 크기에 대해 처리량(
images per second, tokens per second) 및 에너지 효율성(
images per Wh, tokens per Wh)을 측정하며, 하드웨어 및 소프트웨어 적합성과 비교 가능성에 대한 문제를 해결한 경험을 공유한다.



### DiffEditor: Enhancing Speech Editing with Semantic Enrichment and Acoustic Consistency (https://arxiv.org/abs/2409.12992)
- **What's New**: 본 논문에서는 DiffEditor라는 새로운 음성 편집 모델을 소개합니다. 이 모델은 OOD(out-of-domain) 텍스트 시나리오에서 성능을 향상시키기 위해 고안되었습니다.

- **Technical Details**: DiffEditor는 편집된 음성의 이해도를 높이기 위해 미리 훈련된 언어 모델에서 추출한 단어 임베딩을 통합하여 음소(phoneme) 임베딩의 의미 정보를 풍부하게 만듭니다. 또한, 음향 일관성(acoustic consistency)을 모델링하기 위해 프레임 간(Interframe) 스무딩 속성이 중요함을 강조하며, 편집 경계에서 부드러운 전환을 촉진하는 1차 손실 함수(first-order loss function)를 제안합니다.

- **Performance Highlights**: 실험 결과, DiffEditor는 인 도메인(in-domain) 및 OOD 텍스트 시나리오 모두에서 최첨단(state-of-the-art) 성능을 달성하는 것을 입증하였습니다.



### Hyperbolic Brain Representations (https://arxiv.org/abs/2409.12990)
Comments:
          8 pages, 4 figures

- **What's New**: 이 논문에서는 인공지능(AI)에서 인공신경망(ANN)의 발전을 줄여진 기하학의 복잡성을 통해 향상시키기 위한 접근법으로 하이퍼볼릭 기하학(hyperbolic geometry)의 사용을 제안합니다. 인간 뇌의 구조와 기능을 연구하여, 하이퍼볼릭 기하학이 인간 지능에 미치는 중요한 역할을 설명합니다.

- **Technical Details**: 하이퍼볼릭 기하학은 비유클리드 기하학(non-Euclidean geometry)으로서, 데이터의 복잡한 계층적 구조를 효과적으로 모델링할 수 있습니다. 논문에서는 하이퍼볼릭 공간이 대규모 계층 구조를 자연스럽게 표현하는 방법을 다루고, ANN의 성능 향상에 기여할 수 있음을 논의합니다. 이 기하학의 모델링 방법론 또한 소개됩니다.

- **Performance Highlights**: 하이퍼볼릭 신경망은 자연어 처리(natural language processing), 컴퓨터 비전(computer vision), 복잡한 네트워크 분석에서 유클리드 모델(Euclidean models)보다 우수하며, 적은 수의 매개변수로 더 나은 일반화(generalisation)를 보여줍니다. 이러한 성능적 강점이 하이퍼볼릭 기하학을 AI 연구의 매력적인 접근법으로 만든다고 강조합니다.



### Can we only use guideline instead of shot in prompt? (https://arxiv.org/abs/2409.12979)
- **What's New**: 연구는 FGT 프레임워크를 통해 태스크에 특화된 가이드라인을 자동으로 학습하는 새로운 방법론을 제안합니다. 기존의 shot 방법 대신 가이드라인만을 사용하는 가능성을 탐구하였습니다.

- **Technical Details**: FGT 프레임워크는 피드백 에이전트, 가이드라인 에이전트, 트리 수집 에이전트로 구성됩니다. 피드백 에이전트는 각 Q&A의 결과를 평가하여 효과적인 최적화 전략을 이끌어내며, 가이드라인 에이전트는 이 피드백을 바탕으로 가이드라인을 도출하여 로컬 메모리에 저장합니다. 마지막으로, 트리 수집 에이전트는 모든 가이드라인을 계층적으로 집계하여 최종적으로 유용한 가이드라인을 취합합니다.

- **Performance Highlights**: 실험 결과, FGT 프레임워크에서 생성된 가이드라인 기반의 프롬프트가 여러 태스크에서 뛰어난 성능을 달성하였고, 가이드라인을 사용한 프롬프트의 효과성을 강조합니다.



### The Era of Foundation Models in Medical Imaging is Approaching : A Scoping Review of the Clinical Value of Large-Scale Generative AI Applications in Radiology (https://arxiv.org/abs/2409.12973)
Comments:
          25 pages,3 figures, 4 tables, submitted to NPJ imaging

- **What's New**: 최근 방대한 규모의 생성을 위한 AI (Generative AI)가 의료 영상 처리에서의 활용 가능성을 보여주고 있으며, 이 연구는 그 발전 현황과 주요 문제들을 체계적으로 정리하고 있습니다.

- **Technical Details**: 본 연구는 PCC 가이드라인을 따르며, PubMed, EMbase, IEEE-Xplore, Google Scholar의 네 개 데이터베이스에서 체계적인 검색을 진행하였습니다. 15개의 연구가 선정되었으며, 이들 대부분은 보고서 생성 과정의 효율성을 높이거나 환자의 이해를 돕기 위한 번역에 초점을 맞추었습니다. 연구들은 주로 LLMs (Large Language Models)를 활용하였으며, 다중 모달 모델 (multi-modal models)을 사용하는 연구는 세 건에 불과했습니다.

- **Performance Highlights**: 대부분의 연구가 GPT를 활용했으며, 의료 영상 도메인에 특화된 모델은 거의 사용되지 않았습니다. LLMs와 multi-modal 모델 모두 특정 영역에서 우수한 결과를 보였지만, 아직까지 진단 성능에서 방사선 전문의를 능가하지는 못했습니다. 이 연구는 의료 영상 분야에서의 대규모 생성 AI 활용의 현재 상태와 한계를 제시하며, 향후 의료 영상 기초 모델들이 임상 실무를 혁신적으로 변모시킬 것이라는 전망을 제공합니다.



### TRACE: Transformer-based user Representations from Attributed Clickstream Event sequences (https://arxiv.org/abs/2409.12972)
Comments:
          RecSys Workshop on Recommenders in Tourism (RecTour 2024), October 14th-18th, 2024, co-located with the 18th ACM Conference on Recommender Systems, Bari, Italy

- **What's New**: TRACE는 여러 사용자 세션에서의 클릭스트림 데이터를 기반으로 사용자 임베딩을 생성하기 위한 새로운 변환기 기반 접근 방식입니다. 기존 연구와 달리 단일 세션 제품 시퀀스에 주로 집중하는 대신, TRACE는 총체적인 사용자 선호도와 의도를 캡처합니다.

- **Technical Details**: TRACE는 다중 작업 학습(multi-task learning) 프레임워크를 활용하여 다수의 사용자 참여 대상을 예측하는 경량 변환기 인코더를 훈련합니다. 각 사용자의 방문 기록은 페이지 뷰 이벤트로 기록되며, 세션 간 연속된 페이지 조회를 모델링하여 풍부한 사용자 여정 표현을 생성합니다.

- **Performance Highlights**: TRACE는 대규모 여행 전자상거래 데이터셋을 통한 광범위한 실험에서 기존의 변환기(Transformer) 및 LLM 스타일 아키텍처에 비해 우수한 성능을 보여주었습니다. 학습된 임베딩의 시각화는 잠재적인 사용자 상태와 행동에 해당하는 의미 있는 클러스터를 드러내어 추천 시스템 향상 가능성을 강조합니다.



### MITHOS: Interactive Mixed Reality Training to Support Professional Socio-Emotional Interactions at Schools (https://arxiv.org/abs/2409.12968)
- **What's New**: MITHOS는 교사들을 위한 혁신적인 Mixed Reality(MR) 훈련 시스템으로, 교실 내 갈등 해결 능력을 극대화하기 위해 설계되었습니다. 이 시스템은 사회적 정서적 자기 인식, 관점 취하기, 긍정적 인식을 배양하는 4단계 과정을 제공하여 교사가 갈등 상황을 효과적으로 관리하고 감정을 조절할 수 있도록 돕습니다.

- **Technical Details**: MITHOS는 가상 환경에서 상호작용하는 학생 에이전트(SIA)와 교사가 실제적으로 사회적 상호작용을 훈련할 수 있도록 지원합니다. 첫번째 단계에서 교사는 다양한 학생 반응에 자유롭게 상호작용하며 갈등 상황을 경험하고, 이후에는 자기 인식 및 공감 능력을 향상시키는 훈련 단계를 진행합니다. 시스템은 Wizard-of-Oz(WoZ) 모형을 사용하여 데이터를 수집하고, 기계 학습 및 모델 기반 접근 방식을 통해 향후 전자동 시스템을 개발합니다.

- **Performance Highlights**: MITHOS의 초기 평가에서는 시나리오의 사실성과 교사 자기 인식의 선행 요인으로서 아바타 유사성의 효과가 체계적으로 테스트되었습니다. 훈련에 참여한 교사들은 정서적 자기 인식과 공감적 긍정적 태도를 바탕으로 더 효과적인 갈등 해결 방안을 개발할 수 있었습니다.



### OpenRANet: Neuralized Spectrum Access by Joint Subcarrier and Power Allocation with Optimization-based Deep Learning (https://arxiv.org/abs/2409.12964)
- **What's New**: Open RAN(오픈 무선 접속망)에서 AI-native 인터페이스와 딥러닝(Deep Learning) 통합으로 무선 셀룰러 네트워크의 비선형 최적화 문제를 해결하는 새로운 모델 OpenRANet을 제안합니다.

- **Technical Details**: OpenRANet은 비선형 문제를 볼록(Convex) 하위 문제로 변환하여 해결하는 최적화 기반의 딥러닝 모델입니다. 변환 과정에서 변수 변화 및 이완 기법을 활용하여 볼록 최적화 계층을 통합합니다. 이는 데이터 전송 속도 요구를 충족하면서 총 전력 소모를 최소화하도록 설계되었습니다.

- **Performance Highlights**: OpenRANet은 다른 최신 최적화 기반 기법 및 머신러닝 모델과 비교하여 성능을 평가하였으며, 제시된 해결책이 큰 정확도 향상과 계산 효율성을 제공함을 보였습니다.



### Improve Machine Learning carbon footprint using Parquet dataset format and Mixed Precision training for regression models -- Part II (https://arxiv.org/abs/2409.11071)
Comments:
          35 pages, 16 tables, 19 figures. arXiv admin note: substantial text overlap with arXiv:2409.07853

- **What's New**: 본 논문은 머신러닝(Machine Learning) 회귀 모델을 훈련하는 과정에서 CSV와 Parquet 데이터셋 포맷을 비교하고, 기본 부동 소수점(32bit)과 Nvidia 혼합 정밀도(16bit 및 32bit) 사용 시의 전력 소비를 분석했습니다.

- **Technical Details**: 실험은 Deep Neural Networks (DNN)를 구축하기 위해 다양한 ML 하이퍼파라미터(예: 배치 크기, 뉴런 수, 에포크)를 선택하고, 기본 하이퍼파라미터 값을 사용한 벤치마크 테스트를 참조하여 진행되었습니다. 결과는 Excel에 기록되었고, 집단 간 평균을 계산하여 그래프와 표로 비교했습니다. 주의할 점은 배치 크기와 뉴런 수가 증가할 경우 전력 소비에 부정적인 영향을 미칠 수 있다는 것입니다.

- **Performance Highlights**: 혼합 정밀도 사용 및 특정 하이퍼파라미터 조합 시, 회귀 모델의 전력 소비가 7에서 11 와트 감소했습니다. 하지만 하이퍼파라미터를 신중히 고려해야 하며, ANOVA 및 T-검정을 통한 추론 통계가 필요했습니다. 결과적으로, ML 기술이나 Parquet 데이터셋 포맷의 선택이 전반적인 연산 전력 소비 및 ML 탄소 발자국에 개선을 가져오지 못했습니다. 다만, GPU 클러스터를 통해 더 큰 샘플 사이즈를 확보할 수 있다는 가능성이 제기되었습니다.




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



### h4rm3l: A Dynamic Benchmark of Composable Jailbreak Attacks for LLM Safety Assessmen (https://arxiv.org/abs/2408.04811)
- **What's New**: 이 논문은 컴퓨터 프로그램 합성을 이용하여 다양한 새로운 제한 해제 공격을 생성하는 새로운 벤치마크(benchmark)를 제안합니다. 이 벤치마크는 정적 데이터셋이 아닌 동적 데이터셋을 사용하기 때문에 끊임없이 진화하고 있는 제한 해제 공격에 대해 더 효과적으로 대처할 수 있습니다. 또한, 이 논문은 이러한 공격을 공식적으로 표현할 수 있는 도메인 특정 언어(DSL)를 개발하였습니다. 이 DSL은 제한 해제 공격을 파라미터화된 문자열 변환 함수(string transformation function)의 조합으로 표현합니다.



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




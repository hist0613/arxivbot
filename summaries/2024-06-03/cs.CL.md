### Code Pretraining Improves Entity Tracking Abilities of Language Models (https://arxiv.org/abs/2405.21068)
- **What's New**: 최근 연구들은 코드 데이터로 사전 학습된 언어 모델이 자연어로 표현된 담화(entity)의 상태 변화를 추적하는 능력을 향상시킨다는 간접적인 증거를 제시했습니다. 이번 연구에서는 베이스 모델(base models)과 코드 데이터를 추가로 학습한 모델을 비교하며 이 주장에 대한 체계적인 검증을 시도하였습니다.

- **Technical Details**: 우리는 코드 데이터와 수학 데이터, 그리고 alignment tuning(정렬 튜닝)이 모델의 엔티티 추적 능력에 미치는 영향을 분석했습니다. 코드 데이터로 추가 학습된 모델이 베이스 모델보다 뛰어난 성능을 보인 반면, 수학 데이터나 alignment tuning은 일관된 이점을 보이지 않았습니다. 베이스 모델과 추가로 특정 데이터를 학습한 모델 쌍을 비교하여 실험을 진행했습니다. 코드 데이터의 효과를 테스트하기 위해 Llama 2, Code Llama, DeepSeek, DeepSeek-Coder 등의 모델 쌍을 사용하였으며, 수학 데이터의 효과를 테스트하기 위해 Code Llama, Llemma 등의 모델 쌍을 사용했습니다.

- **Performance Highlights**: 코드 데이터를 추가로 학습한 모델(Llama 2 13B, 70B, DeepSeek 모델 등)은 엔티티 추적 능력 면에서 베이스 모델을 상회하는 성능을 보였습니다. 특히, 코드 학습을 추가한 모델은 비트리비얼(trivial)한 엔티티 추적 사례에서 일관되게 우수한 성과를 냈습니다. 반면, Llama 2 7B 모델과 CodeGemma 8B 모델에서는 추가 코드 학습의 향상이 비교적 미미했습니다. 엔티티 추적 능력에 있어 코드 학습의 효과는 모델 크기와 추가 코드 데이터 양에 좌우될 수 있음을 시사합니다.



### Direct Alignment of Language Models via Quality-Aware Self-Refinemen (https://arxiv.org/abs/2405.21040)
- **What's New**: 이 논문은 인간의 피드백에서 강화 학습 (Reinforcement Learning from Human Feedback, RLHF)을 대체하여 정책 최적화(Direct Policy Optimization, DPO) 방식을 사용하고, 기존의 보상 모델을 필요로 하지 않는 새로운 방법을 제안한다. 이 방법은 긍정적 및 부정적 응답의 상대적 품질을 고려해 자체적으로 손실 함수를 개선하도록 설계되었다는 점에서 독창적이다.

- **Technical Details**: 기존의 DPO는 보상 모델 없이 정책 자체를 사용하여 메모리와 훈련 시간을 절약할 수 있지만, 응답의 상대적 품질을 무시하여 최적의 학습 결과를 얻는 데 어려움이 있었다. 이를 해결하기 위해, LLM의 내재적 지식을 활용하여 긍정적 응답과 부정적 응답의 품질을 평가하는 정제 함수를 개발하였다. 이 정제 함수는 자체적으로 손실 함수를 개선할 수 있으며, DPO와 그 변형인 Identity Policy Optimization (IPO)에 통합되었다.

- **Performance Highlights**: 이 논문에서 제안된 방법(정제 함수를 사용하는 DPO와 IPO 변형)은 여러 평가자들에 의해 실험되었으며, 기존의 DPO와 IPO 방법들을 초과하는 성능을 보여주었다.



### LACIE: Listener-Aware Finetuning for Confidence Calibration in Large Language Models (https://arxiv.org/abs/2405.21028)
Comments:
          17 pages. Code: this https URL

- **What's New**: 본 연구는 대형 언어 모델(LLM)이 명시적 증거(marker)와 암시적 증거를 통해 자신감 수준을 전달할 수 있지만, 대부분 과도하게 자신감을 표현한다는 문제를 다룬다. 새롭게 제안된 모델 LACIE는 청자의 반응을 고려하여 모델의 자신감 수준을 조정하는 방법이다. 이를 통해 모델의 신뢰성을 높이고 인간 사용자들에게 정확한 정보 전달을 목표로 한다.

- **Technical Details**: LACIE는 'Listener-Aware Calibration for Implicit and Explicit confidence'의 약자로, 청자 모델을 활용한 청자의 반응을 학습에 반영하는 방법이다. 모델은 QA 쌍을 이용하여 정확성과 청자의 수용 여부를 기준으로 데이터를 생성하고, 이를 통해 모델의 응답이 청자에게 어떻게 인식되는지를 학습한다. Direct Preference Optimization (DPO) 프레임워크를 채택하여 모델을 미세 조정(finetuning)한다.

- **Performance Highlights**: 세 가지 LLM(Mistral-7B, Llama3-8B, Llama3-70B)을 LACIE로 미세 조정한 결과, 청자 모델에 대한 정확도는 평균 20.7점 상승, 칼리브레이션 오류는 7.8점 감소했다. LACIE로 훈련된 모델은 47% 더 적은 잘못된 답변을 수용하면서도 올바른 답변에 대해 동일한 수준의 수용을 유지했다. 또한, LACIE는 암시적 증거를 효율적으로 사용하여 모델의 신뢰도와 성능을 크게 개선했다.



### You Only Scan Once: Efficient Multi-dimension Sequential Modeling with LightN (https://arxiv.org/abs/2405.21022)
Comments:
          Technical report. Yiran Zhong is the corresponding author. The code is available at this https URL

- **What's New**: 이번 연구에서는 기존의 선형 주의 메커니즘(linear attention mechanism)의 효율성을 개선하여 다차원 시퀀스 모델링 문제에 대한 새로운 접근법을 제안합니다. LightNet이라는 새로운 프레임워크를 통해 이러한 다차원 시퀀스 데이터를 단일 스캔(single scan)으로 처리하는 효율적인 방법을 도입했습니다. 또한 다차원 상대적 위치 인코딩(MD-TPE, MD-LRPE)을 통해 위치 정보를 보다 효과적으로 인식할 수 있도록 했습니다.

- **Technical Details**: 기존의 곱셈적 선형 재발(multiplicative linear recurrence)은 다차원 데이터 처리에 비효율적이었으나, 본 연구에서는 이를 해결하기 위해 더 효율적인 덧셈적 선형 재발(additive linear recurrence)을 제안합니다. 이로 인해 단일 스캔으로 글로벌 정보를 수집할 수 있게 되었습니다. 특히, MD-TPE(Multi-Dimensional Toeplitz Positional Encoding)와 MD-LRPE(Multi-Dimensional Linearized Relative Positional Encoding) 같은 새로운 위치 인코딩 방법을 도입하여 다차원 시나리오에서의 효율성을 극대화했습니다.

- **Performance Highlights**: LightNet 모델은 이미지 분류, 이미지 생성, 양방향 언어 모델링(bidirectional language modeling), 그리고 자가회귀 언어 모델링(autoregressive language modeling)을 포함한 다양한 작업에서 뛰어난 성능을 보였습니다. 이는 다른 최신 모델들과 비교해도 경쟁력 있는 성능을 보여줍니다.



### SaySelf: Teaching LLMs to Express Confidence with Self-Reflective Rationales (https://arxiv.org/abs/2405.20974)
Comments:
          The code is available at \url{this https URL}

- **What's New**: 이번 논문에서는 SaySelf라는 새로운 훈련 프레임워크를 제안하여 대형 언어 모델(LLMs)이 더 정확하고 세밀한 확신 추정치를 생성하도록 가르칩니다. 이전 연구들과는 달리, SaySelf는 모델의 지식 격차를 식별하고 해당 불확실성을 설명하는 자기 성찰적인 논리를 생성하도록 합니다.

- **Technical Details**: SaySelf는 두 가지 주요 단계를 포함합니다. 첫 번째는 감독 학습(Supervised Fine-Tuning)으로, 여러 응답을 샘플링하여 모델별 데이터셋을 구축합니다. 이 데이터셋에는 질문(q), 이유 체인(s), 자기 성찰적 논리(r), 확신 추정치(c)가 포함됩니다. 두 번째 단계는 강화 학습(Reinforcement Learning)입니다. 여기서 설계된 보상 함수를 통해 정확하고 높은 확신도의 예측을 생성하도록 모델을 보정합니다. 이를 위해 HotpotQA 데이터셋을 사용하여 다단계 추론이 필요한 질문들에서 훈련 샘플을 얻습니다.

- **Performance Highlights**: 실험 결과에 따르면 SaySelf는 내재적(out-of-distribution) 및 외재적(in-distribution) 데이터셋 모두에서 우수한 성능을 보입니다. 특히, 확신 보정 에러를 감소시키고 태스크 성능을 유지합니다. 생성된 자기 성찰적 논리가 내부 불확실성을 효과적으로 포착하여 보정 성능을 향상시킵니다.



### Superlatives in Context: Explicit and Implicit Domain Restrictions for Superlative Frames (https://arxiv.org/abs/2405.20967)
Comments:
          11 pages

- **What's New**: 이번 논문에서는 최상급(superlative) 표현의 의미 분석을 위한 통합 계정을 제안합니다. 이는 다양한 도메인 데이터셋에 대해 최상급 및 그 의미 해석을 주석 처리할 수 있는 광범위한 주석 스키마를 도출하는 것을 목표로 합니다. 이를 통해 암묵적이거나 애매한 최상급 표현을 어떻게 해석할 수 있는지 연구합니다.

- **Technical Details**: 최상급 표현은 특정 속성을 기준으로 한 집합 비교(set comparison)를 수행합니다. 이 논문에서는 최상급 의미를 통합적으로 설명하는 주석 스키마를 제안하며, 이를 기반으로 SuperSem이라는 데이터셋을 주석 처리했습니다. 특히 T5 모델을 사용하여 최상급 해석을 예측하는 실험을 통해 문맥이 비교 해석에 미치는 영향을 분석했습니다. 또한 GPT-4 모델을 사용하여 문맥 제한을 적절히 통합하는 데 어려움을 겪는다는 점을 밝혔습니다.

- **Performance Highlights**: 실험 결과, 문맥 내에서의 세밀한 최상급 의미 해석은 현대의 모델들에게 도전 과제임을 보여주었습니다. T5 모델은 문맥 없이도 최상급을 예측하는데 어느 정도 성공적이었으나, GPT-4 모델은 문맥 제한을 적절히 통합하는 데 어려움을 겪었습니다.



### OR-Bench: An Over-Refusal Benchmark for Large Language Models (https://arxiv.org/abs/2405.20947)
Comments:
          version 1

- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 과도한 거부 문제를 다루기 위해 새로운 방법을 제안합니다. OR-Bench라는 최초의 대규모 과도한 거부 벤치마크를 도입하여, LLM들이 무해한 질문에 대해 과도하게 거부하는 문제를 체계적으로 측정할 수 있도록 합니다.

- **Technical Details**: 연구팀은 '겉보기엔 유해하지만 실제로는 무해한 질문'을 자동으로 생성할 수 있는 방법을 개발했습니다. 이를 통해 OR-Bench라는 벤치마크를 구축하였으며, 80,000개의 무해한 질문 세트와 1,000개의 어려운 질문 세트, 그리고 600개의 실제 유해한 질문을 포함하고 있습니다. 또한, 이 벤치마크를 사용하여 25개의 인기 있는 LLM들을 8개의 모델 계열에 걸쳐 평가하였습니다.

- **Performance Highlights**: 평가 결과 대부분의 모델들이 안전성을 높이는 데는 성공했지만, 동시에 무해한 질문을 과도하게 거부하는 경향이 나타났습니다. 특히, Claude 모델은 가장 높은 안전성을 보였지만 과도한 거부도 가장 많았고, GPT-3.5-turbo는 최신 버전에서 과도한 거부가 줄어드는 추세를 보였습니다. 또한, 모델 크기와 안전-민감도 균형 사이의 상관관계가 확실하지 않은 것으로 나타났습니다.



### Learning to Estimate System Specifications in Linear Temporal Logic using Transformers and Mamba (https://arxiv.org/abs/2405.20917)
Comments:
          20 pages, 15 figures

- **What's New**: 이 연구는 기존의 커맨드 기반(combinatorial) 방법에 비해 더 효율적으로 시스템 트레이스(trace)로부터 선형 시간 논리(linear temporal logic, LTL) 수식을 생성할 수 있는 새로운 자가 회귀 모델(autoregressive models)을 소개합니다. 이 모델은 파라미터 수가 더 적음에도 불구하고 높은 정확도와 효율성을 자랑합니다.

- **Technical Details**: 연구진은 사양 마이닝(specification mining) 문제를 해결하기 위해 두 가지 주요 아키텍처를 제안했습니다. 첫째, 트랜스포머 인코더-디코더(transformer encoder-decoder) 아키텍처 기반 모델로, 이는 자연어 번역 분야에서 이미 검증된 구조입니다. 둘째, 디코더 전용 아키텍처(decoder-only architecture) 기반 모델로, 대표적으로 Mamba 모델을 포함합니다. 이러한 모델들은 입력 트레이스가 주어질 때 자가 회귀 방식으로 LTL 수식을 생성합니다. 또한 LTL 문법을 강제하기 위한 알고리즘도 도입했습니다.

- **Performance Highlights**: 제안된 모델들은 기존의 조합적(combinatorial) 알고리즘에 비해 연산 비용이 현저히 적습니다. 실험 결과, 트랜스포머와 Mamba 아키텍처 모두 의미적으로나 문법적으로 올바른, 그리고 독창적인 수식을 생성하는 데 있어 유의미한 성과를 보였습니다.



### Preemptive Answer "Attacks" on Chain-of-Thought Reasoning (https://arxiv.org/abs/2405.20902)
Comments:
          Accepted to ACL'24 (Findings). Camera-ready version

- **What's New**: 이 연구에서는 사전 답변(preemptive answers) 시나리오를 새롭게 소개합니다. 이는 사용자가 질문에 대한 답변을 먼저 얻은 후 추론을 시작하는 상황을 말하며, 의도하지 않게 발생할 수도 있고 악의적인 사용자에 의해 유도될 수도 있습니다. 이러한 사전 답변이 모델의 추론 능력을 크게 저해한다는 실험 결과를 얻었습니다. 이를 완화하기 위한 두 가지 방안을 제시하며, 아직도 추가적인 추론 강인성 개선이 필요함을 시사합니다.

- **Technical Details**: CoT(Chain-of-Thought) 프롬프트를 통해 모델이 문제를 단계별로 추론하도록 하는 방법론을 사용합니다. 사전 답변 시나리오를 시뮬레이트하기 위해, 원래의 사용자 프롬프트에 '질문에 대한 답변을 먼저 반환해야 한다'는 추가 지시를 추가합니다. 이는 모델이 추론 전에 답변을 생성하도록 유도하며 잘못된 답변을 생성할 가능성을 높입니다. 이를 완화하기 위해 문제 재진술(problem restatement)과 자기 반성(self-reflection) 두 가지 전략을 제안합니다.

- **Performance Highlights**: ChatGPT와 GPT-4를 대상으로 한 실험에 따르면, 사전 답변 시나리오에서 CoT 방법론의 성능이 최대 62%까지 감소할 수 있습니다. 특히 모델이 잘못된 사전 답변을 제안할 경우, 이후의 추론 결과도 이에 따라 잘못될 가능성이 큽니다.



### Large Language Models: A New Approach for Privacy Policy Analysis at Sca (https://arxiv.org/abs/2405.20900)
- **What's New**: 이 연구는 이전의 심볼릭(symbolic) 및 통계적(statistical) 자연어 처리(NLP) 기술을 대체하여 대규모로 프라이버시 정책에서 프라이버시 관행을 추출하기 위해 대형 언어 모델(LLM)을 적용하는 것을 제안합니다. 특히 ChatGPT와 Llama 2와 같은 잘 알려진 LLM을 활용하며, 프롬프트(prompt), 매개변수 및 모델의 최적 설계에 대한 가이드를 제공합니다.

- **Technical Details**: 연구에서는 중요한 프라이버시 관행을 정확하게 감지하는 능력을 강조하면서 ChatGPT의 최적 구성을 찾는 것에 중점을 둡니다. 이를 위해 몇 가지 주어진 예제를 통해 모델을 미리 학습시키는 'few-shot learning' 전략을 포함합니다. 실험에서는 주요 데이터셋을 벤치마크로 사용하여 LLM 기반 접근 방식이 높은 성능을 달성함을 확인했습니다.

- **Performance Highlights**: 제안된 LLM 기반 솔루션은 전통적인 NLP 기술을 능가하는 성과를 보입니다. 몇 가지 알려진 데이터셋을 사용하여 여러 실험을 수행한 결과, 연구는 F1 점수 93%를 초과하는 성능을 달성했습니다. 또한, 낮은 비용, 빠른 처리 시간, 그리고 적은 기술 지식 요구 사항 등 다양한 장점을 지니고 있습니다.



### A comparison of correspondence analysis with PMI-based word embedding methods (https://arxiv.org/abs/2405.20895)
- **What's New**: 이 논문은 단어 임베딩(Word Embedding) 방법 중 하나인 PMI 행렬의 가중치 분해(weighted factorization)와 대응 분석(Correspondence Analysis, CA)을 수학적으로 연결시킨다. 또한, 단어-컨텍스트 행렬의 분석에 성공적인 새로운 CA 변형들인 ROOT-CA와 ROOTROOT-CA를 소개한다.

- **Technical Details**: CA는 단일 값 분해(Singular Value Decomposition, SVD)를 사용하는 차원 축소 방법으로, PMI 행렬의 가중치 분해와 수학적으로 유사하다. ROOT-CA는 단어-컨텍스트 행렬의 항목에 제곱근 변화(square-root transformation)를 적용한 것이며, ROOTROOT-CA는 이 행렬에 네 번째 루트 변화(fourth-root transformation)를 적용한다.

- **Performance Highlights**: ROOT-CA와 ROOTROOT-CA는 단어 유사성(word similarity) 실험에서 PMI 기반 방법들보다 약간 더 나은 성과를 보였다. 특히, ROOT-CCA와의 비교에서도 우수한 성능을 보였다.



### clembench-2024: A Challenging, Dynamic, Complementary, Multilingual Benchmark and Underlying Flexible Framework for LLMs as Multi-Action Agents (https://arxiv.org/abs/2405.20859)
Comments:
          under review

- **What's New**: 최신 연구에서 Large Language Models(LLMs)을 '자체 플레이(self-play)' 대화 게임에서 사용하여 그들의 특정 능력(일반 지침 준수, 전략적 목표 지향, 언어 이해 능력)을 테스트할 수 있음이 밝혀졌습니다. 이번 연구에서는 이러한 게임 플레이 환경을 설정하는 데 사용되는 clemgame 프레임워크를 활용하여 평가 도구로서의 유용성을 다양한 차원에서 검증했습니다.

- **Technical Details**: 이번 연구는 clemgame 프레임워크를 사용하여 새로운 개발과의 동시성을 유지하면서 데이터 오염을 피할 수 있다는 점을 확인하였고, 현재의 테스트가 최상의 모델보다 훨씬 높은 인간 성과를 가지고 있는 덕분에 아직 포화 상태에 도달하지 않았음을 보여주었습니다. 또한, 프롬프트 언어가 성능에 미치는 영향을 조사하는 등 추가적인 질문을 연구할 수 있는 가능성을 제시하였습니다. 이 프레임워크는 다양한 모델을 쉽게 통합할 수 있으며, 이는 실험 환경의 확장과 새로운 모델의 성능 추적을 용이하게 합니다.

- **Performance Highlights**: 현재의 벤치마크 결과는 빠르게 발전하는 LLM 분야를 효과적으로 추적할 수 있음을 보여줍니다. 초기 발표에서는 9개의 모델만을 평가했으나, 최신 버전에서는 53개의 모델을 추적하고 있습니다. 흥미롭게도, 닫힌 무게 모델들과는 달리 열린 무게 모델들은 현저하게 성능이 향상되었습니다. 예를 들어, llama3-70b-ins 모델 덕분에 열린 모델과 닫힌 모델 간의 성능 격차가 크게 줄어들었습니다.



### Towards Spoken Language Understanding via Multi-level Multi-grained Contrastive Learning (https://arxiv.org/abs/2405.20852)
- **What's New**: SLU (Spoken Language Understanding)의 핵심 과제인 의도 탐지(intent detection)와 슬롯 채우기(slot filling)를 보다 효과적으로 결합하기 위해, 연구팀은 새로운 멀티 레벨 멀티 그레인(Multi-Level Multi-Grained) SLU 프레임워크 MMCL을 제안합니다. 이 프레임워크는 대조 학습(contrastive learning)을 발화 수준(utterance level), 슬롯 수준(slot level), 단어 수준(word level)에서 적용하여 의도와 슬롯이 상호 가이드를 가능하도록 합니다.

- **Technical Details**: MMCL은 세 가지 레벨(발화 수준, 슬롯 수준, 단어 수준)에서 대조 학습을 사용합니다. 발화 수준에서는 두 종류의 대조 학습(세밀한 granularity와 거친 granularity)을 동시에 수행하며, 모델의 강건성을 높이기 위해 자가 증류 방법(self-distillation method)을 도입합니다. 또한, 의도 탐지와 슬롯 채우기 사이의 상호 관계를 최대한 활용하기 위해 margin 기반의 대조 학습 방법과 multi-grained 대조 학습을 적용합니다.

- **Performance Highlights**: MMCL은 두 가지 공공 멀티 의도 SLU 데이터셋(MixATIS와 MixSNIPS)에서 최고의 성능을 달성했습니다. 특히, MixATIS 데이터셋에서 이전 모델 대비 2.6%의 전체 정확도(accuracy) 향상을 보였습니다. 이러한 성과는 제안된 모델이 기존 모델보다 우수하다는 것을 입증합니다.



### Improving Reward Models with Synthetic Critiques (https://arxiv.org/abs/2405.20850)
- **What's New**: 새로운 연구에서는 인간 피드백을 통한 강화 학습을 최적화하기 위해 대규모 언어 모델이 생성한 합성 자연 언어 비평(synthetic natural language critiques)을 활용한다는 아이디어를 제안했습니다. 이 방법은 기존의 인간 주석보다 더 풍부한 시그널을 제공하여 보상 모델(RM)의 성능과 데이터 효율성을 높이는 데 중점을 둡니다.

- **Technical Details**: 논문에서는 언어 모델을 활용하여 합성 비평을 생성하고 이를 바탕으로 보상 모델을 훈련하는 방식을 설명하고 있습니다. 구체적으로, LLM(Large Language Models)에게 각 프롬프트와 응답 쌍에 대해 합성 비평을 생성하도록 지시하며, 이 비평은 정확성, 지침 준수 등 여러 차원을 평가합니다. 이후 이 비평을 조건으로 한 보상 모델을 훈련해 스칼라 보상을 예측합니다. 이 방법은 기존 인간 피드백 수집의 비용과 노동 강도를 크게 줄여줍니다.

- **Performance Highlights**: 고품질의 합성 비평을 활용하면 보상 모델의 성능이 유의미하게 향상됨을 실험을 통해 입증했습니다. 특히 저자원 데이터 환경에서 성능 개선이 두드러졌으며, 고품질 모델 생성 비평 하나는 약 40개의 기존 선호 데이터쌍에 상응하는 효과를 보였습니다. 또한, 합성 비평을 통해 보상 모델의 해석 가능성과 강건성이 증대되었습니다.



### Don't Buy it! Reassessing the Ad Understanding Abilities of Contrastive Multimodal Models (https://arxiv.org/abs/2405.20846)
Comments:
          Accepted to the main conference ACL 2024

- **What's New**: TRADE라는 새로운 평가 시험 세트를 도입하여 이미지 기반 광고 이해(automatic ad understanding)에 대한 기존 평가 구조의 한계를 분석했습니다. 이 세트는 인간에게는 그럴 듯하지 않게 보이지만, 네 가지 다른 대조적 Vision-and-Language Models (VLMs)을 '속이는' 광고 설명을 포함합니다.

- **Technical Details**: Pitt Ads 데이터셋은 광고 64832개와 각 광고에 대해 3개의 설명을 포함하여 이미지-텍스트 일치(scoring)를 통해 광고를 이해하는 모델 성능을 평가합니다. CLIP 등 여러 VLMs가 이미지-텍스트 정렬 점수를 통해 광고 설명을 검색하는 작업에서 아주 뛰어난 성능(95.2%)을 보였으나, 이는 시각적 및 텍스트 기반 정렬 정보에 의존한 단순한 휴리스틱(heuristics)을 활용한 결과일 수 있습니다. TRULY ADversarial ad understanding Evaluation(TRADE)는 모델이 이러한 단순한 정렬 정보에 의존하지 않도록 설계된 새로운 평가 세트입니다.

- **Performance Highlights**: TRADE 테스트에서, 기존에 뛰어난 성능을 보였던 여러 VLMs(CLIP 포함)이 성능이 크게 저하되어 기회 수준(chance level)으로 떨어졌습니다. 이는 현재의 광고 이해 평가 방식이 모델의 다중모달(multimodal) 추론 능력을 제대로 평가하지 못하고 있다는 것을 시사합니다. 반면 인간은 TRADE에서도 훌륭한 성능을 보였습니다. 코드는 공개되어 있습니다.



### That's Optional: A Contemporary Exploration of "that" Omission in English Subordinate Clauses (https://arxiv.org/abs/2405.20833)
Comments:
          ACL2024 (main conference), 8 pages

- **What's New**: 이번 연구는 기존의 유니폼 정보 밀도(UID) 가설을 바탕으로 영어 종속절에서 선택적인 'that' 접속사의 생략을 조사합니다. 저자들은 확장된 텍스트 코퍼스를 사용하고, 현대 대형 언어 모델(LLMs)을 활용하여 정보 균일성 원칙을 더욱 심화된 방식으로 분석합니다. 이를 통해 문법 간소화 현상을 분석하며, 정보 이론적 관점에서 SC 시작 단어의 정보 엔트로피를 고려합니다.

- **Technical Details**: 이 연구는 Reddit에서 수집된 200만 개의 게시글과 댓글을 분석 대상으로 하였습니다. 487,614개의 문장을 필터링하여 명시적 및 암시적 'that' 접속사의 사용 패턴을 조사했습니다. 이를 위해 SOTA(benepar) 구문 분석기를 사용했습니다. SVM 기반의 대형 언어 모델(LLMs)을 사용하여 SC 시작 단어의 surprisal(비예측성)과 정보 엔트로피를 계산하고, SC 절의 선택적 생략 처리에 UID의 영향을 분석했습니다.

- **Performance Highlights**: UID 원칙이 문법 간소화에 미치는 영향을 성공적으로 입증하며, 10만 개 이상의 문장을 포함하는 대규모 코퍼스를 공개했습니다. 이 연구는 기존의 UID 원칙을 확장하여 의사소통의 효율성을 최적화하는 새로운 빛을 제공하였습니다. 특히, '생각하다(think)', '말하다(say)', '알다(know)'와 같은 주 동사 뒤에 나타나는 'that' 접속사의 생략 경향을 더 깊이 이해할 수 있게 되었습니다.



### Self-Augmented Preference Optimization: Off-Policy Paradigms for Language Model Alignmen (https://arxiv.org/abs/2405.20830)
- **What's New**: 기존의 전통적인 언어 모델 정렬 방법과 달리, SAPO(Self-Augmented Preference Optimization)는 정적이고 사전 수집된 선호 데이터에 의존하지 않고도 효과적인 훈련을 제공합니다. SAPO는 자가 생성된 거부 응답을 활용하는 'self-play' 개념을 기반으로 하며, 오프폴리시 학습(off-policy learning) 파이프라인을 통합하여 데이터 탐색과 활용을 강화합니다. 이를 통해 SAPO는 실시간 피드백과 과거 데이터를 통합해 동적 업데이트를 실시합니다.

- **Technical Details**: SAPO는 세 가지 주요 구성 요소로 나뉩니다: 현재 정책, 현재 정책의 EMA(Exponential Moving Average) 모델, 그리고 순차적(replay buffer). 학습은 두 단계로 진행됩니다: 샘플링 단계와 훈련 단계. 샘플링 단계에서는 EMA 모델이 응답을 생성하여 자가 증강된 거부 샘플을 만들고, 이 샘플들은 원본 응답과 함께 버퍼에 저장됩니다. 훈련 단계에서는 버퍼에서 무작위로 선택된 데이터로 현재 정책을 훈련합니다. EMA 모델과 순차적 버퍼를 사용함으로써 학습 신호의 일관성을 유지합니다.

- **Performance Highlights**: SAPO는 다양한 벤치마크(Open LLM Leaderboard, IFEval, AlpacaEval 2.0, MT-Bench)에서 기존의 오프라인 대조 학습 방법(DPO, ORPO)과 비교해 동등하거나 우수한 성능을 나타냈습니다. 또한, SAPO는 오프라인 자가 학습 방법인 SPIN보다 우수한 성능을 보여주며, 긴 훈련 시간을 요구하지 않습니다.



### An iterated learning model of language change that mixes supervised and unsupervised learning (https://arxiv.org/abs/2405.20818)
- **What's New**: 이 논문에서는 새로운 Iterated Learning Model(ILM) 모델을 소개합니다. 이 모델에서는 신경망(Neural Networks)을 활용하여 디코더와 인코더를 각각 지도 학습(Supervised Learning)과 비지도 학습(Unsupervised Learning)의 형태로 훈련시킵니다. 이 새로운 접근 방식은 기존의 오버전(Obversion) 알고리즘의 높은 계산 비용을 피하고, 인간 발달 과정에서 관찰되는 지도 학습과 비지도 학습의 혼합을 도입합니다.

- **Technical Details**: ILM은 언어 변화를 모사하는 에이전트 기반 모델로, 언어가 튜터에서 학생으로, 다시 새로운 튜터로 전해지는 과정을 반복합니다. 새로운 ILM에서는 신경망 기반의 디코더와 인코더를 사용하여 의미에서 신호로의 매핑과 신호에서 의미로의 매핑을 각각 구현합니다. 이 두 신경망은 처음엔 개별적으로 지도 학습되고, 이후 오토인코더(Autoencoder)를 통해 함께 비지도 학습됩니다.

- **Performance Highlights**: 새로운 모델은 기존 연구들보다 더 복잡하고 표현력이 뛰어난 구성을 가진 언어를 달성하며, 언어의 의미-신호 공간의 차원성과 최적의 전송 병목현상(bottleneck) 크기 사이에 선형 관계를 밝힙니다. 이는 이전의 ILM 모델들보다 더 효율적인 언어 학습이 가능함을 보여줍니다.



### Multilingual Text Style Transfer: Datasets & Models for Indian Languages (https://arxiv.org/abs/2405.20805)
- **What's New**: 이 논문에서는 텍스트 스타일 변환(Text Style Transfer, TST)의 하위 작업인 감정 전이(sentiment transfer)를 인도 언어들인 힌디어, 마가히, 말라얄람어, 마라티어, 펀자브어, 오디아어, 텔루구어, 우르두어에 초점을 맞추어 연구했습니다. 이는 이전의 영어-벵골어 감정 전이 연구를 확장한 것입니다. 각 언어별로 1,000개의 긍정적 문장과 1,000개의 부정적 문장으로 구성된 평행 데이터셋을 새롭게 도입했습니다.

- **Technical Details**: 이번 연구에서는 평행(parallel), 비평행(non-parallel), 교차 언어(cross-lingual), 공유 학습(shared learning) 접근 방식을 사용하여 여러 벤치마크 모델을 평가했습니다. 예를 들어, Llama2와 GPT-3.5 같은 대규모 언어 모델(Large Language Models, LLMs)의 성능을 테스트했습니다. 특히, 비평행 기술에서는 Masked Style Filling (MSF) 접근 방식의 효과성을 입증했습니다.

- **Performance Highlights**: 실험 결과, 평행 데이터의 중요성을 강조했으며 교차 언어 및 다중 언어 학습 방법이 많은 잠재력을 가지고 있다는 점을 보여주었습니다. 이번 연구는 다양한 언어와 특정 작업에 적합한 최적의 모델을 선택하는데 유용한 통찰력을 제공합니다. 감정 전이 작업에 대해 여러 언어를 다룬 이 연구는 그 자체로 독보적인 탐색입니다.



### PGA-SciRE: Harnessing LLM on Data Augmentation for Enhancing Scientific Relation Extraction (https://arxiv.org/abs/2405.20787)
- **What's New**: 이번 연구에서는 과학 도메인의 Relation Extraction (RE) 모델 성능을 향상시키기 위해 PGA(Paraphrasing and Generating Augmentation)라는 텍스트 데이터 증강 프레임워크를 제안합니다. 이 프레임워크는 LLM을 이용해 기존 학습 샘플을 패러프레이즈(paraphrasing)하여 동일한 문장의 의미를 가지고 있지만 다른 표현과 형태를 갖는 가상 샘플(pseudo-sample)을 얻는 방법과, LLM에게 원본 학습 샘플의 관계와 엔티티를 기반으로 라벨에 대한 정보를 암묵적으로 포함한 문장을 생성하도록 지시(fake sentence generation)하는 방법을 소개합니다.

- **Technical Details**: PGA 프레임워크는 두 가지 방식의 데이터 증강을 도입합니다. 첫째, LLM을 활용해 원본 학습 세트의 샘플을 패러프레이즈하여 다양한 표현과 형태를 가진 가상 샘플을 얻습니다. 둘째, 원본 샘플의 관계와 엔티티를 기반으로 라벨 정보를 암시적으로 포함하는 새로운 문장을 생성하도록 LLM을 지시합니다. 이러한 가상 샘플들은 원본 데이터세트와 함께 RE 모델의 학습에 사용됩니다.

- **Performance Highlights**: 실험 결과, PGA 프레임워크는 과학 도메인에서 주류 RE 모델의 F1 점수를 개선했습니다. 또한, LLM을 사용해 샘플을 얻는 방식은 수작업으로 데이터를 라벨링하는 비용을 효과적으로 줄일 수 있음을 보여줍니다.



### Large Language Model Sentinel: Advancing Adversarial Robustness by LLM Agen (https://arxiv.org/abs/2405.20770)
- **What's New**: 최근의 연구에서 우리는 LLAMOS (Large LAnguage MOdel Sentinel)라는 새로운 방어 기법을 소개했습니다. 이는 대형 언어 모델(LLMs)의 적대적 공격에 대한 대응력을 강화하는 데 중점을 둡니다. LLAMOS는 적대적 텍스트 예시들을 정제하여 타겟 언어 모델에 입력하기 전에 불순물을 제거합니다.

- **Technical Details**: LLAMOS는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 'Agent instruction'은 적대적 방어를 위해 최소한의 문자 변형을 통해 문장의 본래 의미를 유지하면서 공격에 대응하도록 설계된 새로운 에이전트를 시뮬레이션합니다. 둘째, 'Defense guidance'는 깨끗하거나 적대적인 예시들을 수정하는 전략을 제공하여 타겟 LLM에서 효과적인 방어와 정확한 출력을 보장합니다.

- **Performance Highlights**: 광범위한 실험을 통해 LLAMOS의 성능을 평가한 결과, LLAMA-2와 GPT-3.5 모델에서 각각 최대 45.59%와 37.86%의 공격 성공률 감소를 달성했습니다. 이는 LLAMOS가 적대적 공격에 대해 효과적으로 방어함을 나타냅니다. 또한, 초기 방어 에이전트가 명백한 공격에 대해 기대한 결과를 내지 못한 경우, 인컨텍스트 러닝(in-context learning)을 통해 방어 에이전트를 최적화하여 방어 능력을 크게 향상시켰습니다.



### Improving code-mixed hate detection by native sample mixing: A case study for Hindi-English code-mixed scenario (https://arxiv.org/abs/2405.20755)
Comments:
          Generated from XeLaTeX

- **What's New**: 해당 논문은 코드-혼합 (code-mixed) 환경에서의 혐오 탐지 (hate detection)를 다룹니다. 기존의 연구는 주로 단일 언어 환경에 집중되었으나, 이 논문은 힌디어-영어 코드-혼합을 사례 연구로 사용하여 멀티링구얼 언어 모델 (Multilingual Language Models, MLMs)을 통해 혐오 탐지를 개선하는 방법을 제안합니다.

- **Technical Details**: 연구팀은 주로 네이티브 언어의 혐오 샘플을 사용하여 코드-혼합 환경에서의 혐오 탐지를 시도했습니다. 네이티브 언어의 혐오 샘플이 포함된 경우, MLMs가 코드-혼합 환경에서 혐오를 더 잘 탐지할 수 있음이 밝혀졌습니다. 이는 MLMs가 교차 언어 설정에서 혐오 탐지가 효과적이라는 기존 연구를 보완하는 결과입니다.

- **Performance Highlights**: ['코드-혼합 학습 세트에 소량이라도 네이티브 혐오 샘플을 추가하면 MLMs의 성능이 향상됨.', '네이티브 샘플만을 사용해서도 상당한 정도로 코드-혼합 혐오를 탐지할 수 있음.', '포함된 네이티브 샘플을 시각화한 결과, MLMs가 혐오 단어에 더 잘 집중함을 확인.', '주관적이거나 비꼬는 (sarcastic) 혐오 표현의 경우, 네이티브 샘플만으로는 탐지에 한계가 있음.']



### FinGen: A Dataset for Argument Generation in Financ (https://arxiv.org/abs/2405.20708)
- **What's New**: 이 연구는 NLP(Natural Language Processing) 연구에서 미래 지향적인 논증 생성에 대해 탐구하고 있습니다. 금융 애플리케이션 시나리오에서 세 가지 논증 생성 작업을 새롭게 제안하며, 기존 생성 모델들이 이 작업에서 여전히 큰 도전에 직면하고 있음을 실험 결과로 보여줍니다.

- **Technical Details**: 논증 생성은 증거 검색, 구조/전략 계획, 논증 합성을 포함한 세 가지 주요 단계로 나눌 수 있습니다. 이번 연구에서는 기업의 미래 운영에 대한 주관적 의견을 생성하는 Evidence2Claim, 가격 차트와 투자자의 노트를 기반으로 요약을 생성하는 Chart2Argument, 그리고 뉴스 기사를 바탕으로 가능한 시나리오와 투자 제안을 생성하는 News2Argument를 제안합니다.

- **Performance Highlights**: 실험에서는 각 작업에 대해 다양한 모델들을 사용하였으며, 대표적인 생성 모델들인 T5, mT5, BART, Pegasus 및 Pegasus-Chinese를 적용했습니다. 또한, Chart2Argument 작업에서는 Transformer 기반의 TrOCR, BEiT, RoBERTa, DeiT, Swin, ViT 등 다수의 이미지 인코더-디코더 아키텍처를 탐구했습니다. 결과적으로 이러한 작업들은 여전히 많은 도전 과제가 남아 있음을 확인했습니다.



### It is Simple Sometimes: A Study On Improving Aspect-Based Sentiment Analysis Performanc (https://arxiv.org/abs/2405.20703)
Comments:
          Accepted to ACL Findings 2024

- **What's New**: 이 논문은 Aspect-Based Sentiment Analysis (ABSA)를 위한 새로운 생성적 프레임워크를 제안합니다. 기존의 사리아 등(2023)의 명령 학습 모델을 확장하여, NLP 관련 작업 접두사(prefix)를 작업 설명에 추가하는 PFInstruct를 도입했습니다. 이 간단한 접근 방식은 모든 검증된 SemEval 부작업에서 성능이 향상되었고, 특히 ATE(Aspect-Term Extraction) 부작업에서 +3.28 F1-score, AOOE(Aspect-Opinion Extraction) 부작업에서 평균 +5.43 F1-score로 이전 최고 성능을 능가했습니다.

- **Technical Details**: PFInstruct는 명령 학습 패러다임을 확장하여 작업 설명 앞에 모든 ABSA 부작업에 대해 NLP 관련 작업 접두사를 추가합니다. 이 접두사는 모델이 주어진 텍스트(S)에 대해 더 풍부한 의미 정보를 수집할 수 있도록 도와 ABSA 부작업에 대한 예측을 보다 정확하게 합니다. 접두사는 Relation Extraction(RE)나 Named Entity Recognition(NER) 작업일 수 있으며, 무작위로 생성된 접두사(노이즈 접두사)도 사용해보았습니다.

- **Performance Highlights**: PFInstruct는 SemEval 데이터셋의 모든 테스트 부작업에서 이전 최고 성능을 능가했으며, 특히 ATE 부작업에서 +3.28 F1-score, AOOE 부작업에서 평균 +5.43 F1-score를 기록했습니다. 접두사 기반의 접근 방식은 모델 성능을 향상시키는 데 상당한 효과가 있으며, 특정 도메인 (예: 바이오메디컬 도메인)에서도 경쟁력 있는 결과를 제공함을 입증했습니다.



### Unveiling the Lexical Sensitivity of LLMs: Combinatorial Optimization for Prompt Enhancemen (https://arxiv.org/abs/2405.20701)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)이 작업 지시에 대한 미세한 어휘 변동에도 매우 민감하다는 사실을 밝힙니다. 인간에게는 지각되지 않는 이러한 어휘 변동이 모델의 성능에 큰 영향을 미칠 수 있습니다. 이를 기반으로 우리는 Prompt Lexical Enhancement (COPLE)라는 블랙박스 조합 최적화 프레임워크를 제안합니다. COPLE는 프락시 작업에서의 피드백을 활용해 반복적으로 어휘를 최적화합니다.

- **Technical Details**: COPLE는 프락시(proxy) 작업 그룹에서의 피드백을 바탕으로 어휘 영향을 고려한 검색 전략(search strategy)을 사용하여 반복적인 최적화를 수행합니다. LLM의 지시(instruction) 민감도를 밝히기 위해 저자는 어휘 조합 최적화 문제로 프레임(framing)하여 검색 공간(search space)과 검색 방법(search method)을 정의했습니다. 조합 최적화(combinatorial optimization) 접근법을 통해 LLM의 프롬프트 성능을 극대화할 수 있음을 제시합니다.

- **Performance Highlights**: 실험 결과, 인간이 직관이나 경험을 바탕으로 제작한 프롬프트도 LLM의 어휘 민감성으로 인한 성능 저하를 겪을 수 있습니다. COPLE는 모델 파라미터나 복잡한 프롬프트 엔지니어링 없이도 단순한 어휘 수정만으로 프롬프트의 성능을 효과적으로 극대화하는 성과를 보였습니다.



### DORY: Deliberative Prompt Recovery for LLM (https://arxiv.org/abs/2405.20657)
- **What's New**: 새로운 연구인 DORY는 API 기반 대형 언어 모델(LLM)의 출력에서 프롬프트를 정확하게 복구하기 위해 불확실성을 활용하는 방법을 제시합니다. 이는 사용자 프라이버시 보호, 저작권 문제 해결 등과 같은 중요한 연구 목표를 달성하는 데 기여할 수 있습니다.

- **Technical Details**: DORY는 출력 텍스트와 출력 확률을 이용하여 프롬프트를 복구합니다. 세 가지 주요 단계로 구성됩니다: (1) 출력 텍스트에서 초안을 재구성하는 Draft Reconstruction, (2) 출력의 불확실성을 기반으로 힌트를 생성하는 Hint Refinement, (3) 잡음을 제거하는 Noise Reduction. 프롬프트 복구 성능과 출력 확률 기반 불확실성 사이에 강한 음의 상관관계를 발견하여 이를 기반으로 DORY를 개발하였습니다.

- **Performance Highlights**: DORY는 기존의 프롬프트 복구 방법보다 약 10.82% 향상된 성능을 보였으며, 새롭게 SOTA(최신 기술)를 달성했습니다. 더욱이, DORY는 외부 자원이나 추가 모델 개발 없이 단일 LLM만으로 작동하므로 비용 효율적이고 사용하기 편리한 솔루션을 제공합니다.



### Passage-specific Prompt Tuning for Passage Reranking in Question Answering with Large Language Models (https://arxiv.org/abs/2405.20654)
Comments:
          Accepted at Gen-IR@SIGIR24

- **What's New**: 최근 오픈 도메인 질의 응답(Open-Domain Question Answering, ODQA) 과제에서 효과적인 문단 검색 및 재정렬 기법이 많이 사용되고 있으며, 이를 강화하기 위해 대형 언어 모델(LLMs)을 활용한 재정렬 기법도 주목받고 있습니다. 본 논문은 PSsoft prompt tuning을 활용하여 재정렬 기능을 더욱 향상시키는 방법을 제안합니다.

- **Technical Details**: 제안된 방법인 passage-specific prompt tuning for reranking in open-domain question answering (PSPT)는 LLMs의 생성을 활용하여 질문에 조건화된 각 문단의 로그 우도(log-likelihood)를 기반으로 검색된 문단을 재정렬합니다. PSsoft prompt는 학습 가능한 문단-특정 소프트 프롬프트(soft prompts)을 미세 조정하여 문단-특정 지식(passage-specific knowledge)을 통합하는 효율적인 파라미터 미세 조정 기법입니다. 새로운 학습 프롬프트 모듈은 원본 문단의 임베딩과 결합되며, LLMs에 새로운 입력으로 제공됩니다.

- **Performance Highlights**: 세 가지 공개된 ODQA 데이터셋에서 Llama-2-chat-7B 모델을 활용해 광범위한 실험을 수행한 결과, 제안된 기법이 기존 베이스라인 검색기와 최근 LLMs 기반 모델 성능을 넘어서는 재정렬 성능을 보여주었습니다. 특히, 질문에 조건화된 문단의 로그 우도 차이를 벌줄로 사용하는 힌지 손실(hinge loss)을 적용해 학습하여 효과를 입증했습니다.



### Reward-based Input Construction for Cross-document Relation Extraction (https://arxiv.org/abs/2405.20649)
Comments:
          Accepted at ACL 2024 main conference

- **What's New**: 이번 논문에서는 문서 간 관계 추출(cross-document relation extraction, RE)을 위한 첫 번째 학습 기반 문장 선택 기법인 REward-based Input Construction(REIC)를 제안합니다. 이 기술은 관계 증거에 기반하여 문장을 선택하고, 강화 학습(reinforcement learning)을 통해 여러 문서에서 관계를 효과적으로 추론할 수 있도록 설계되었습니다.

- **Technical Details**: REIC는 현재 선택된 문장에 기반하여 문장을 선택하는 모듈을 도입하고, 이를 마코프 결정 과정(Markov Decision Process, MDP)으로 모델링합니다. 문장 선택 모듈은 관계 예측 점수를 보상으로 받아 강화 학습을 통해 훈련됩니다. REIC는 구체적으로 BERT와 같은 사전 학습된 언어 모델(pre-trained language models)의 토큰 제한을 극복하고자 설계되었습니다.

- **Performance Highlights**: 실험 결과, REIC는 다양한 RE 모듈과 백본(backbone)에 대해 기존의 휴리스틱 방법보다 뛰어난 성능을 보였습니다. CodRED 데이터셋을 사용한 실험에서, REIC는 관계 추출의 정확도 및 효율성을 높이는 데 효과적임을 입증했습니다.



### Leveraging Large Language Models for Entity Matching (https://arxiv.org/abs/2405.20624)
- **What's New**: 이 비전 논문은 GPT-4와 같은 대규모 언어 모델(LLM)의 엔티티 매칭(Entity Matching, EM) 응용을 탐구합니다. LLM이 전통적인 EM 방법론의 한계를 어떻게 해결할 수 있는지에 대한 장점, 도전과제 및 미래 연구 방향을 논의합니다.

- **Technical Details**: 엔티티 매칭은 데이터 통합의 핵심 과제로, 다양한 데이터 소스에서 동일한 현실 세계 엔티티에 해당하는 레코드를 식별하고 연결하는 것을 목표로 합니다. 전통적인 EM 방법은 수동으로 설계된 특징과 규칙 기반 시스템에 의존하는 반면, LLM은 광범위한 텍스트 코퍼스에서 훈련되어 고급 의미 이해와 문맥 파악 능력을 갖추고 있습니다. LLM은 상황에 맞는 임베딩을 생성할 수 있어 수동 특징 엔지니어링을 줄이고, 다양한 도메인에 적응할 수 있습니다.

- **Performance Highlights**: LLM은 전통적인 EM 시스템이 놓치기 쉬운 의미적 유사성을 포착할 수 있으며, 이를 통해 매칭 정확도를 향상시킵니다. 미세 조정(fine-tuning)을 통해 특정 도메인 데이터에서 성능을 강화할 수 있으며, 특히 소셜 미디어와 전자 상거래처럼 데이터가 노이즈가 많고 구조화되지 않은 경우에 유용합니다. 또한 LLM은 약한 감독(weak supervision)과 비지도 EM 방법을 향상시켜 높은 차원과 노이즈가 많은 데이터에서도 더 나은 성능을 발휘할 수 있습니다.



### FineRadScore: A Radiology Report Line-by-Line Evaluation Technique Generating Corrections with Severity Scores (https://arxiv.org/abs/2405.20613)
- **What's New**: 최근 공개된 FineRadScore는 대규모 언어 모델(LLM) 기반으로 자동화된 평가 메트릭을 통해 생성된 가슴 엑스레이(CXR) 리포트를 평가합니다. 이 메트릭은 후보 리포트를 기반으로 실질 리포트까지 도달하려면 몇 개의 수정이 필요한지 최소 수의 라인 바이 라인 수정 개수를 제공합니다. 또한 각 수정에 대한 오류 심각도 등급과 수정 이유를 설명하는 주석을 생성합니다.

- **Technical Details**: FineRadScore는 GPT-4와 Claude-3 Opus를 백본으로 사용하는 평가 프레임워크를 제안합니다. 이 프레임워크는 문장 단위의 수정과 임상 심각도 등급, 그리고 오류 유형을 제공하며 다양한 데이터셋에서 이러한 평가 메트릭이 방사선 전문의의 판단과 일치하는지 확인했습니다. 본 연구는 초기 100개 사례와 기존 데이터셋을 사용해 평가 프레임워크의 효과를 검증했습니다.

- **Performance Highlights**: FineRadScore는 생성된 리포트와 실질 리포트를 비교하여 방사선 전문의의 의견과 일치하는 수준의 수정 및 오류 심각도 점수를 제공합니다. 리포트 품질을 판단하는 데에도 방사선 전문의와 동등한 수준의 정확성을 보였으며, 기존 최첨단 자동화 CXR 평가 메트릭과 유사한 성능을 발휘했습니다. 또한 개선이 필요한 부분도 분석하여 향후 연구 방향을 제시합니다.



### UniBias: Unveiling and Mitigating LLM Bias through Internal Attention and FFN Manipulation (https://arxiv.org/abs/2405.20612)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)에서 발생하는 내재 편향(inherent bias)을 조사하고 이를 완화하기 위한 새로운 기법인 UniBias를 소개합니다. 기존 연구들이 LLM 출력 결과에 대한 외부 조정을 통해 편향을 해결하려고 했던 반면, 이 연구는 피드포워드 신경망(FFN)과 어텐션 헤드(attention heads)가 이러한 편향을 유발하는 내부 메커니즘을 탐구합니다.

- **Technical Details**: LLM의 편향을 유발하는 내부 메커니즘을 조사하기 위해 각 FFN 벡터와 어텐션 헤드가 레이블 예측에 미치는 영향을 해석하였습니다. 이를 통해 특정 레이블로 예측을 왜곡시키는 편향된 LLM 구성 요소를 식별합니다. UniBias는 이러한 편향된 FFN 벡터와 어텐션 헤드를 알아내어 제거하는 추론 전용(inference-only) 방법입니다. 이 방법은 어휘 공간(vocabulary space)으로의 투사를 통해 편향된 구성 요소를 식별한 후 관련된 크리테리온(relatedness criterion), 편향 크리테리온(bias criterion), 그리고 저분산 크리테리온(low variance criterion)을 기반으로 편향된 부분을 마스킹하여 제거합니다.

- **Performance Highlights**: 12개의 NLP 데이터셋을 대상으로 한 실험 결과, UniBias를 적용한 LLM은 원래 LLM에 비해 인컨텍스트 학습(ICL) 성능이 크게 향상되었으며, 다양한 디자인 설정의 변동성에 대해 더 높은 성능과 강인성을 보였습니다. 특히, 이 방법을 통해 LLM의 내부 구조를 효과적으로 조작하여 편향을 완화할 수 있음을 확인했습니다.



### Identifying while Learning for Document Event Causality Identification (https://arxiv.org/abs/2405.20608)
Comments:
          Accepted at ACL 2024

- **What's New**: 이 연구는 사건 인과관계 식별(Event Causality Identification, ECI)에서 기존 방법이 간과한 '인과 방향성(causal direction)'을 포함하는 새로운 접근 방식을 제안합니다. 저자들은 인과관계를 학습하는 동시에 식별하는 모드를 제안하며, 이를 통해 사건의 표현을 반복적(iterative)으로 업데이트하여 인과관계 식별 성능을 향상시킵니다.

- **Technical Details**: 제안된 시스템은 반복적 학습 및 식별 프레임워크(iterative Learning and Identifying Framework, iLIF)를 사용합니다. 초기에는 사전 학습된 언어 모델(pretrained language model)을 통해 사건의 문맥 표현을 인코딩하며, 이후에는 식별된 인과관계를 바탕으로 방향 그래프(event causality graph)를 구축하고 이를 통해 사건의 인과 구조 표현을 갱신합니다. 이 과정에서 새로운 '반복할인 손실 함수(iteration discounted loss function)'를 도입하여 오류 전파를 완화합니다.

- **Performance Highlights**: 제안된 방법은 EventStoryLine(v0.9)와 MAVEN-ERE 두 개의 공개 데이터셋에서 실험을 거쳤으며, 인과 관계의 존재 및 방향성을 평가하는 모든 측면에서 최신 알고리즘(state-of-the-art)보다 우수한 성능을 보였습니다.



### DAFNet: Dynamic Auxiliary Fusion for Sequential Model Editing in Large Language Models (https://arxiv.org/abs/2405.20588)
Comments:
          ACL2024 findings

- **What's New**: 최근 대형 언어 모델(LLMs)은 뛰어난 성능을 보여주고 있지만 여전히 잘못된 정보를 생성하는 '환상' 문제를 겪고 있습니다. 이를 해결하기 위해 모델 편집 기법이 사용되지만, 대부분의 이전 연구들은 이를 일회성 작업으로 간주했습니다. 본 논문은 지속적으로 오류를 수정하는 순차적 모델 편집(SME) 작업을 다루며, 이를 위해 DAFNet (Dynamic Auxiliary Fusion Network)을 설계했습니다.

- **Technical Details**: DAFNet은 사실 지식을 전체 시퀀스 내에서 의미론적으로 상호작용시켜 여러 지식 삼중항(triple)을 편집하는 과정에서 정보 손실을 방지합니다. (1) 세 가지 관계 삼중항 내에서의 의미 융합을 위해, 자동 회귀적(self-attentive) 자체 주의 메커니즘을 활용하여 LLM 내에서 토큰 수준의 주의 흐름을 수집합니다. 또한 다층 대각선 간편집 주의 흐름을 활용해 시퀀스 수준의 가중치 표현을 업데이트합니다. (2) 순차적 편집을 위해 보조 파라미터가 필요하기 때문에, 최근과 인기 있는 데이터, 길게 저장된 데이터, 강건성을 갖춘 데이터로 구성된 새 데이터셋 DAFSet을 구축했습니다.

- **Performance Highlights**: 실험 결과, DAFNet은 단일과 순차적 편집 모두에서 강력한 기존 방법을 크게 능가했습니다. DAFSet을 사용하면 다양한 시나리오에서 다른 보조 네트워크 기반 방법의 성능도 일관되게 향상되었습니다.



### GAMedX: Generative AI-based Medical Entity Data Extractor Using Large Language Models (https://arxiv.org/abs/2405.20585)
- **What's New**: GAMedX는 병원 방문 동안 생성된 비정형 텍스트(medical narratives)에서 개체를 추출하기 위해 대형 언어 모델(LLMs)을 활용한 이름명 인식(NER) 방법론을 소개합니다. 이 연구는 의료 기록에서 데이터를 효율적으로 추출하여 의료 NER 애플리케이션의 새로운 표준을 설정합니다.

- **Technical Details**: GAMedX는 오픈 소스 LLM을 사용하여 비정형 의료 텍스트를 처리하는 통합 접근 방식을 취합니다. 체인된 프롬프트(chained prompts)와 Pydantic 스키마를 통해 복잡한 의료 용어(jargon)를 구조화된 출력으로 변환합니다. LLM의 민감한 데이터 보호 요구사항을 충족하면서도 의료 시스템과의 통합을 가능하게 합니다.

- **Performance Highlights**: GAMedX는 한 개 평가 데이터셋에서 ROUGE F1 점수와 98%의 정확도를 달성하여 우수한 성능을 보였습니다. 이 시스템은 자동화된 양식 작성 및 비정형 데이터 처리의 효율성을 크게 향상시키며, 비용 효율적인 해결책을 제시합니다.



### The Point of View of a Sentiment: Towards Clinician Bias Detection in Psychiatric Notes (https://arxiv.org/abs/2405.20582)
Comments:
          Oral presentation at NAACL 2024 Queer in AI Workshop

- **What's New**: 이번 연구는 정신과 임상 기록에서 환자에 대한 부정적인 기술과 낙인 언어가 환자에게 미치는 영향을 조사합니다. 이 논문은 대형 언어 모델(large language models)을 활용하여 독자의 관점에 따라 정신과 임상 기록에서 표현된 감정을 분류하는 방법을 제시합니다.

- **Technical Details**: 마운트 사이나이 건강 시스템(Mount Sinai Health System)의 다양한 임상 기록에서 문장을 추출한 후, GPT-3.5, Llama 2, Mistral 세 가지 대형 언어 모델을 사용하여 문장이 제공자(provider)와 비제공자(non-provider) 관점에서 어떤 감정을 전달하는지 분류했습니다. '프롬프트'와 인-컨텍스트 학습(in-context learning)을 이용해 각 모델을 튜닝하였습니다.

- **Performance Highlights**: 실험 결과, GPT-3.5 모델이 제공자(provider) 관점과 가장 잘 일치하는 반면, Mistral 모델은 비제공자(non-provider) 관점과 가장 잘 일치하는 것으로 나타났습니다.



### Open Ko-LLM Leaderboard: Evaluating Large Language Models in Korean with Ko-H5 Benchmark (https://arxiv.org/abs/2405.20574)
Comments:
          Accepted at ACL 2024 Main

- **What's New**: 새롭게 발표된 Open Ko-LLM Leaderboard와 Ko-H5 Benchmark는 한국어 대형 언어 모델(LLM)을 평가하기 위한 중요한 도구들입니다. 영어 중심의 평가 체계를 한국어로 확장하고, 민감한 데이터 유출 문제를 최소화하기 위해 private test sets를 도입했습니다.

- **Technical Details**: Open Ko-LLM Leaderboard는 두 가지 원칙을 기초로 구축되었습니다: (i) 영어 Open LLM Leaderboard와의 정렬, (ii) 프라이빗 테스트 세트의 사용. 이를 통해 모델 간의 비교가 용이해졌으며, 프라이빗 세트를 사용하여 데이터 오염 없이 다양한 모델을 공정하게 평가할 수 있게 되었습니다. Ko-H5 Benchmark는 기존 영어 데이터셋과 새롭게 구축된 데이터셋으로 구성됩니다. 이러한 데이터셋에는 Ko-CommonGen v2와 같은 새로운 항목이 포함되어 평가의 다양성을 높였습니다.

- **Performance Highlights**: Ko-H5 벤치마크의 점수가 시간에 따라 어떻게 변화했고, 모델의 크기에 따라 어떤 차이가 있는지에 대한 심층 분석이 수행되었습니다. 또한, 사전 학습된 모델이 지침 조정 모델에서 성능 향상에 어떤 영향을 미치는지 보여줍니다. 특정 작업의 점수가 빠르게 포화되는 경향이 나타나면서, 좀 더 현실적인 사례를 반영하는 평가 체계로의 전환이 필요함을 제안합니다.



### Towards Ontology-Enhanced Representation Learning for Large Language Models (https://arxiv.org/abs/2405.20527)
Comments:
          14 pages, 1 figure

- **What's New**: 이 논문은 참조 온톨로지(reference ontology)에 형식화된 지식을 주입함으로써 임베딩-Large Language Model (embedding-LLM)을 향상시키는 새로운 접근법을 제안합니다. Ontological Knowledge Infusion(온톨로지 지식 유입)은 참조 온톨로지가 설명하는 지식 영역을 효과적으로 모델링하는 LLM의 능력을 향상시키는 것을 목표로 합니다.

- **Technical Details**: 이 접근법은 GPT-3.5-turbo와 같은 강력한 생성 LLM을 사용하여 개념 정의를 생성하고 이를 통해 타겟 embedding-LLM을 대조 학습(contrastive learning) 프레임워크로 미세 조정합니다. 여기서 언어 정보(예: 개념 동의어와 설명)와 구조적 정보(예: is-a 관계)를 사용하여 개념 정의 집합을 작성합니다.

- **Performance Highlights**: 제안된 접근법을 평가하기 위해 생물의학적 질병 Ontology MONDO를 사용했습니다. 결과는 온톨로지 질병 지식이 주입된 embedding-LLM이 생물의학 문서에서 언급된 질병과 도메인 내 문장의 유사성을 효과적으로 평가하는 능력이 향상되었음을 보여주며, 도메인 외 성능은 손상되지 않았습니다.



### How Multilingual Are Large Language Models Fine-Tuned for Translation? (https://arxiv.org/abs/2405.20512)
- **What's New**: 최근 대형 언어 모델(LLM)에 대한 미세 조정(fine-tuning)이 대규모 병렬 텍스트(parallel text)를 사용한 전통적인 번역 시스템을 능가하는 성능을 보이는 새로운 패러다임이 등장했습니다. 이 연구는 많은 언어 쌍에 대해 미세 조정된 모델이 아닌 LLM이 광범위한 다국어 기계 번역을 가능하게 할 수 있는지 여부를 탐구합니다.

- **Technical Details**: 이 연구는 TOWER 모델 패밀리(TOWER family of language models)를 사용해 132개의 번역 과제에 대한 번역 품질을 평가하였습니다. 평가는 다국어 FLORES-200 데이터셋을 기반으로 했으며, 언어쌍별로 다양하게 평가되었습니다. 이 연구는 또한 대형 언어 모델의 제로샷(zero-shot) 번역 능력, 특히 비영어 언어쌍에 대한 번역 성능을 조사하였습니다.

- **Performance Highlights**: 평가 결과, 미세 조정된 번역 작업이 제로샷 언어에도 번역 품질을 향상시킬 수 있다는 것을 발견했습니다. 다만, 언어 쌍에 따라 성능 차이가 크며, 특정 언어는 다른 언어에 비해 번역 품질이 낮았습니다. 예를 들어, 한국어(Korean)와 아이슬란드어(Icelandic)는 미세 조정 시에도 낮은 품질을 보였습니다.



### SPOT: Text Source Prediction from Originality Score Thresholding (https://arxiv.org/abs/2405.20505)
- **What's New**: 최근 대형 언어 모델(LLMs)의 광범위한 도입이 새로운 사회적 위험과 애플리케이션을 야기했습니다. 이에 대한 대응책으로, 정보의 진위를 평가하는 대신 신뢰도 관점에서 LLM이 생성한 텍스트를 조사하는 SPOT이라는 방법을 제안합니다. 이 방법은 입력 텍스트가 사람이 작성한 것인지 LLM이 작성한 것인지 분류하며, 특정 LLM이 다른 LLM을 탐지할 수 있는지를 예측하여 오리지널리티 점수(originality score)를 산출합니다.

- **Technical Details**: SPOT은 주어진 LLM의 예측 로그잇(prediction logits)과 오리지널리티 점수를 활용하여 입력 텍스트의 출처를 분류합니다. 이를 위해, SPOT은 사전 훈련된 LLM(logits)으로부터 예측한 '토큰의 특정 위치에 대한 오리지널리티'를 측정합니다. 이를 통해 각 문장에 대한 오리지널리티 점수를 계산하고, 특정 임계값(ρ)을 기준으로 텍스트의 출처를 '사람' 또는 'LLM'으로 분류합니다.

- **Performance Highlights**: SPOT은 다양한 LLM 아키텍처 및 훈련 데이터, 모델 크기(350M부터 7B까지), 토큰 시퀀스 길이(24부터 512까지)에 대해 높은 성능을 보였습니다. 특히, SPOT은 LLM 압축 시에도 성능이 유지되며, 짧은 컨텍스트(24 토큰) 및 긴 컨텍스트(512 토큰) 모두에서 일관된 결과를 보였습니다. 그러나 특정 작업에 특화된 LLM(예: 수학 문제 해결, 코딩)에 대해서는 효과적이지 않았습니다.



### Transfer Q Star: Principled Decoding for LLM Alignmen (https://arxiv.org/abs/2405.20495)
- **What's New**: 새로운 연구는 전통적인 파인 튜닝(fine-tuning) 방법의 대안으로 디코딩 기반의 정렬(alignment via decoding) 방법을 제안합니다. 이 방법은 대규모 모델 파라미터 업데이트 없이 직접적으로 응답 분포를 조정하여 목표 보상(target reward)을 최대화합니다. 특히, 이 연구에서는 Transfer Q^* 라는 새로운 방법을 도입하여 최적의 Q-함수를 분명히 추정합니다.

- **Technical Details**: Transfer Q^*는 기본 보상(baseline reward)에 맞춰진 모델을 통해 목표 보상에 대한 최적의 값 함수(optimal value function)를 암묵적으로 추정합니다. 이 접근법은 이론적으로 최적성에 대한 엄격한 특성을 제공하며, 사전 훈련된 SFT 모델에서의 편차를 제어할 수 있는 하이퍼파라미터를 식별합니다. 주요 아이디어는 이미 공개적으로 사용 가능한 경로 수준 모델을 사용하여 Q^*을 추정하고, 이를 통해 디코딩 시 최적의 토큰 수준 언어 모델을 도출하는 것입니다.

- **Performance Highlights**: Transfer Q^*는 기존 최첨단(SoTA) 디코딩 방법들이 경험했던 서브-최적 격차(sub-optimality gap)를 현저하게 줄이며, 여러 합성 및 실제 데이터셋에서 뛰어난 성능을 입증 했습니다. 예를 들어, GPT-4 기반의 테스트에서 평균 보상이 최대 1.45배, 승-무 비율이 67.34% 향상되었습니다. 생성된 텍스트의 일관성, 다양성 및 품질 측면에서도 우수한 성과를 보였습니다.



### Automated Focused Feedback Generation for Scientific Writing Assistanc (https://arxiv.org/abs/2405.20477)
Comments:
          Accepted to ACL 2024 (Findings)

- **What's New**: 새로운 과제로 과학적 글쓰기 지원을 위한 자동 집중 피드백 생성(task of automated focused feedback generation)을 제안합니다. 이를 위해, 구체적이고 실행 가능한 피드백(comment)을 생성하는 SWIF$^{2}$T 도구를 소개합니다. 이 도구는 과학 문서의 약점을 식별하거나 수정안을 제안하는 데 중점을 둡니다.

- **Technical Details**: 우리의 접근법은 플래너(planner), 조사자(investigator), 리뷰어(reviewer), 컨트롤러(controller)의 네 가지 구성 요소로 이루어져 있습니다. 이들은 모두 여러 대형 언어 모델(LLM: Large Language Models)을 활용하여 구현됩니다. 플래너는 조사자가 문서나 문헌을 사용해 질문에 답하도록 지시합니다. 이를 통해 문단을 관련 맥락으로 풍부하게 합니다. 이후 리뷰어는 수집된 맥락을 활용해 특정 부분의 약점 유형(예: Originality)을 예측하고 집중 피드백을 생성합니다. 컨트롤러는 전체 계획의 진행 상황을 관리하고 다른 모델들의 동작을 조정합니다.

- **Performance Highlights**: 우리는 동료 리뷰를 통해 약점을 지적한 300개의 예제를 포함한 테스트 세트를 컴파일하고, SWIF$^{2}$T의 성능을 CoVe와 GPT-4와 같은 기존 모델들과 비교 평가했습니다. 결과는 SWIF$^{2}$T의 피드백이 구체성, 독해력(reading comprehension) 및 전반적인 유용성 면에서 다른 접근법보다 우수함을 보여줍니다. 분석 결과, AI가 생성한 피드백이 인간의 피드백보다 선호되는 경우도 있어 과학적 글쓰기 과정에 AI 피드백을 통합할 가능성을 시사합니다.



### Extending the Massive Text Embedding Benchmark to French (https://arxiv.org/abs/2405.20468)
- **What's New**: 최신 연구에 따르면, 문장 임베딩(sentence embeddings)을 위한 첫 번째 프랑스어 대규모 벤치마크가 제안되었습니다. 총 25개의 데이터셋(이 중 3개는 새로 생성됨)이 포함되어 있으며, 이들을 사용해 8가지 다른 작업에서 46개의 임베딩 모델을 비교합니다. 모든 코드는 오픈 소스로 제공되며, 새로운 데이터셋과 퍼블릭 리더보드도 포함되어 있습니다.

- **Technical Details**: 임베딩 모델은 입력의 의미를 캡처하는 밀집 벡터 표현입니다. Word2Vec에서 시작해 BERT와 같은 트랜스포머(transformer) 기반 모델로 발전해왔으며, 현재 다양한 아키텍처의 모델들이 존재합니다. 본 연구에서는 프랑스어에 특화된 MTEB(Massive Text Embedding Benchmark)을 확장하여 사용하기 쉬운 인터페이스로 22개의 기존 데이터셋과 3개의 새로운 데이터셋을 통합했습니다. 문장 유사성(sentence similarity) 작업에 미리 학습된 대규모 다국어 모델들이 특히 우수한 성능을 보입니다.

- **Performance Highlights**: 모든 작업에서 최고의 성능을 보이는 모델은 없었지만, 대규모 다국어 모델들이 대부분의 벤치마크 데이터셋에서 우수한 성능을 보여주었습니다. 모델의 성능과 모델 크기, 학습 기법, 다국어 데이터 사용 여부가 강하게 상관관계가 있음이 밝혀졌습니다.



### Scalable Detection of Salient Entities in News Articles (https://arxiv.org/abs/2405.20461)
- **What's New**: 이 논문은 사전 훈련된 트랜스포머 모델을 미세조정하여 뉴스 기사 내에서 중요한 엔터티(Entities)를 효율적이고 효과적으로 감지하는 새로운 접근 방식을 탐구합니다. 이전 연구들을 크게 능가하며, 특히 대규모 데이터셋에서 높은 성과를 보입니다. 또한, 지식 증류(Knowledge Distillation) 기법을 사용하여 모델의 계산 비용을 줄이면서도 정확성을 유지하는 방법을 제안합니다.

- **Technical Details**: RoBERTa를 변형하여 이진 분류 작업으로 엔터티의 중요도를 모델링합니다. 이 과정에서 단/다수 엔터티에 대한 위치, 빈도 등의 정보를 포함한 특징 공학(Feature Engineering) 없이도 트랜스포머 모델을 활용합니다. 또한 최고의 성능을 제공하기 위해 DistillRoBERTa 등의 지식 증류 기법과 온도 조정(Temperature Scaling) 등을 사용하여 모델 크기를 작게 하여도 높은 성능을 유지하도록 합니다.

- **Performance Highlights**: 제안된 모델은 NYT-Salience, WN-Salience, SEL-Wikinews 데이터셋에서 주목할 만한 성과를 나타냈습니다. 특히, 대규모 RoBERTa-Large 모델과 유사한 성능을 작은 DistillRoBERTa 모델에서 구현하였으며, 다양한 데이터셋에 걸쳐 좋은 전반적 성능을 보여주었습니다.



### SeamlessExpressiveLM: Speech Language Model for Expressive Speech-to-Speech Translation with Chain-of-Though (https://arxiv.org/abs/2405.20410)
- **What's New**: 이 논문은 SeamlessExpressiveLM이라는 새로운 모델을 제안합니다. 이 모델은 풍부한 의미적 정보와 음성 스타일 보존을 목표로 하는 개별 LMs를 사용하는 대신, 언어 모델링(Chain-of-Thought Prompting)을 통해 중간 생성 단계를 거칩니다. 이를 통해 스페인어-영어와 헝가리어-영어 번역에서 의미적 품질과 스타일 전환 모두에서 우수한 성능을 보입니다.

- **Technical Details**: SeamlessExpressiveLM 모델은 단일 스피치 언어 모델로, 음성 간 번역 성능을 개선하기 위해 체인-오브-생각(Chain-of-Thought) 프롬프팅을 사용합니다. 이 모델은 먼저 대상 의미 콘텐츠를 번역하고, 그런 다음 화자의 스타일을 멀티 스트림으로된 음향 단위에 전송합니다. 데이터는 화자 스타일이 정렬되지 않은 의미적으로 정렬된 음성을 사용하여 모델을 훈련합니다. 이를 위해 임의로 타겟 세그먼트를 자르고 음향 프롬프트로 사용하여 스타일 보존을 가능하게 합니다.

- **Performance Highlights**: SeamlessExpressiveLM은 기존의 Cascaded Language Models와 비교했을 때 의미적 품질과 스타일 전환 모두에서 우수하며, 더 나은 파라미터 효율성을 달성했습니다. 또한 의미적 단위와 음향 단위를 동일한 언어 모델로 효과적으로 모델링할 수 있음을 실험적으로 보여주었습니다.



### XPrompt:Explaining Large Language Model's Generation via Joint Prompt Attribution (https://arxiv.org/abs/2405.20404)
- **What's New**: 이번 연구에서는, 대규모 언어 모델 (LLMs)인 GPT4, LLaMA, Claude 등에서 생성된 텍스트의 원인을 설명하기 위해 XPrompt라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 입력 프롬프트가 생성 결과에 미치는 영향을 설명하는 반사실적 설명 (counterfactual explanation) 방법론을 사용하여, 입력 프롬프트의 일부 텍스트가 LLM의 전체 생성 과정에 어떻게 영향을 미치는지 설명합니다.

- **Technical Details**: 기존 연구들은 주로 텍스트 분류나 다음 단어 예측에 초점을 맞추었으나, 본 연구는 프롬프트 텍스트의 조합이 전체 생성 결과에 미치는 영향을 고려하는 Joint Prompt Attribution 방법을 사용합니다. 이 방법론은 조합 최적화 문제로 정의되며, 이산 공간 내에서 인과 입력 조합을 탐색하는 확률적 알고리즘을 도입합니다. 특히, 디스크리트 솔루션 공간에서 효율적인 탐색을 위해 그래디언트 가이드 및 확률적 업데이트를 포함한 마스크 접근 방식을 사용합니다.

- **Performance Highlights**: XPrompt 프레임워크는 텍스트 요약, 질의응답, 일반적인 instruction 데이터셋을 포함한 다양한 언어 생성 작업에서 강력한 성능을 보여줍니다. 생성 확률, 단어 빈도, 의미 유사성을 고려한 종합적인 지표를 기반으로, 문맥에 따라 설명의 충실성과 효율성을 입증하여 다양한 작업 간의 이전 가능성과 유효성을 검증했습니다.



### Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools (https://arxiv.org/abs/2405.20362)
Comments:
          Our dataset, tool outputs, and labels will be made available upon publication. This version of the manuscript (May 30, 2024) is updated to reflect an evaluation of Westlaw's AI-Assisted Research

- **What's New**: 최근 법률 실무에서 인공지능(AI)을 포함한 제품들이 급증하고 있습니다. 이 도구들은 판례 검색 및 요약부터 문서 작성에 이르는 다양한 핵심 법률 작업을 지원하는데, 여기서 사용되는 대형 언어 모델(LLMs)이 '환각'현상을 보일 수 있다는 점이 문제가 되고 있습니다. 최근 법률 연구 제공업체들이 Retrieval-Augmented Generation(RAG) 기법을 사용해 환각 현상을 제거하거나 회피한다고 주장한 바 있습니다.

- **Technical Details**: 이 연구는 LexisNexis(Lexis+ AI)와 Thomson Reuters(Westlaw AI-Assisted Research 및 Ask Practical Law AI)에서 제공하는 AI 기반 법률 연구 도구들의 성능을 실증적으로 처음으로 평가한 것입니다. 연구 결과, 이러한 도구들의 환각 발생률은 일반 채팅봇(GPT-4)에 비해 감소했지만, 여전히 17%에서 33% 사이의 빈도로 환각을 나타냈습니다. 또한, 시스템 간의 응답성과 정확성에서도 큰 차이가 있었습니다.

- **Performance Highlights**: LexisNexis의 Lexis+ AI는 가장 높은 성능을 보여 65%의 정확성을 보였고, Westlaw AI-Assisted Research는 42%의 정확성을 기록했습니다. Thomson Reuters의 Ask Practical Law AI는 쿼리의 60% 이상에서 불완전한 답변을 제시했습니다. 이러한 성과는 환각과 정확한 법률 응답을 구분하는 명확한 유형을 제시하고, AI 출력의 감독 및 검증에서 법률 전문가들이 지녀야 할 책임에 대한 증거를 제공합니다.



### Small Language Models for Application Interactions: A Case Study (https://arxiv.org/abs/2405.20347)
- **What's New**: Microsoft 연구팀은 최신 논문에서 Small Language Models(SLMs)의 효율성을 평가했습니다. 이 연구는 마이크로소프트의 클라우드 공급 체인 관리 내부 애플리케이션을 통해 수행되었으며, SLM이 대형 언어 모델(LLM)보다 높은 정확성과 저렴한 운영 시간을 제공할 수 있음을 보여줍니다.

- **Technical Details**: 연구팀은 Transformer 아키텍처와 확장 법칙(scaling laws)에 대한 전반적인 배경을 설명하며, 데이터 품질 향상의 중요성을 강조했습니다. SLM의 경우, 높은 품질의 데이터셋을 통해 적은 매개변수로도 큰 언어 모델을 능가할 수 있습니다. 작업 유형은 데이터 추출, 계획 생성 및 가상 시나리오 분석 등으로 구분됩니다.

- **Performance Highlights**: SLM인 Phi-3, Llama 3 및 Mistral v0.2 모형이 실험에서 최첨단 LLM보다 높은 정확도와 빠른 출력 속도를 달성했습니다. 작은 데이터셋으로도 우수한 성능을 나타내었으며, 시스템은 최적화된 계획 생성, 비용 증가 분석 등의 작업을 수행하는 데 두각을 드러냈습니다.



### Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis (https://arxiv.org/abs/2405.21075)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 MLLMs (Multi-modal Large Language Models)에서 비디오 분석 성능을 평가하기 위한 최초의 종합적인 평가 벤치마크인 Video-MME를 소개합니다. 이는 기존의 이미지 중심 평가에서 벗어나 시간적인 맥락을 포함한 비디오 데이터의 분석 능력을 평가하는 데 중점을 둡니다.

- **Technical Details**: Video-MME는 6개의 주요 비주얼 도메인의 30개 세부 필드에서 다양한 유형의 비디오를 수집하여 광범위한 시나리오에 대한 일반화를 보장합니다. 비디오 길이는 11초에서 1시간까지 다양하며, 자막과 오디오를 포함한 다중 모달 입력을 통합하여 MLLMs의 전반적인 능력을 평가합니다. 최종적으로 900개의 비디오와 2,700개의 질문-답변 쌍이 포함되었습니다.

- **Performance Highlights**: Video-MME를 사용하여 최신 MLLMs, 예를 들어 GPT-4 시리즈와 Gemini 1.5 Pro, 그리고 오픈 소스 모델인 InternVL-Chat-V1.5, LLaVA-NeXT-Video를 평가했습니다. 실험 결과, Gemini 1.5 Pro가 75.7%의 정확도로 상업 모델 중 가장 높은 성능을 보였으며, 오픈 소스 모델은 상대적으로 낮은 성능을 나타냈습니다. Video-MME 데이터셋은 MLLMs의 시간적 맥락 처리 능력의 한계를 나타내며, 자막과 오디오 통합이 비디오 이해 능력을 크게 향상시킴이 관찰되었습니다.



### Generalization Beyond Data Imbalance: A Controlled Study on CLIP for Transferable Insights (https://arxiv.org/abs/2405.21070)
- **What's New**: 이번 연구는 웹 스케일의 비전-언어 데이터셋에서 심각한 데이터 불균형이 존재함을 밝혔습니다. 연구 결과, CLIP(Contrastive Language-Image Pre-training)는 이러한 데이터 불균형에 대하여 감독 학습(Supervised Learning)보다 뛰어난 견고성을 보이며, 일반화 가능한 표현을 효과적으로 학습할 수 있음을 확인했습니다.

- **Technical Details**: 이 연구는 균형 잡힌 학습 신호를 통해 CLIP의 사전 학습 태스크가 동적 분류 문제를 형성함을 발견했습니다. 이는 지배적인 클래스의 편향을 격리하고 implicitly(암묵적으로) 학습 신호를 균형 있게 합니다. 또한, CLIP의 견고성과 변별력은 서술적인 언어 감독(Language Supervision), 더 큰 데이터 규모, 그리고 더 넓은 open-world 개념을 통해 향상됨을 확인했습니다. 이러한 요소들은 감독 학습에서는 접근할 수 없던 장점들입니다.

- **Performance Highlights**: CLIP 모델은 다양한 비전-언어 데이터셋에서 감독 학습 모델에 비해 데이터 불균형에 대한 견고성이 높았습니다. 이는 클래스의 성능과 빈도 사이의 상관관계가 약하다는 점으로 입증됩니다. 또한, 더 설명적인 텍스트와 open-world 개념이 포함된 데이터는 CLIP의 일반화 성능을 향상시키는 데 기여합니다.



### Grammar-Aligned Decoding (https://arxiv.org/abs/2405.21047)
- **What's New**: 대형 언어 모델(LLMs)이 프로그램 코드, 수학 공식, 구성 파일 등 고도로 구조화된 출력을 생성하는 데 어려움을 겪는 문제를 해결하기 위한 새로운 접근 방식이 제안되었습니다. 구문 제약 디코딩(Grammar-Constrained Decoding, GCD)이 아닌 구문 정렬 디코딩(Grammar-Aligned Decoding, GAD)이라는 방법을 통해 LLM의 분포와 일치하는 출력을 보장합니다.

- **Technical Details**: GAD는 LLM의 출력이 주어진 구문에 맞도록 보장하면서도 LLM의 조건부 확률 분포에 맞는 출력을 생성합니다. 새로운 알고리즘인 ASAp(Adaptive Sampling with Approximate Expected Futures)은 기존 제약 디코딩 알고리즘 위에서 작동하며, 샘플링된 접두사들에 대해 마스킹된 토큰의 확률을 기억하여 미래의 문법적 가능성을 과대평가하는 방식을 사용합니다.

- **Performance Highlights**: 코드 생성 및 구조화된 자연어 처리(NLP) 작업에서 ASAp는 기존의 구문 제약 디코딩(GCD) 기법보다 더 높은 확률의 출력을 생성했으며, 시간이 지남에 따라 LLM의 분포와 일치하는 출력으로 수렴하는 모습을 보였습니다. 즉, GAD는 제약을 유지하면서도 LLM의 성능을 꾸준히 반영할 수 있음을 증명했습니다.



### Exploratory Preference Optimization: Harnessing Implicit Q*-Approximation for Sample-Efficient RLHF (https://arxiv.org/abs/2405.21046)
- **What's New**: 새로운 알고리즘인 탐색 선호 최적화(Exploratory Preference Optimization, XPO) 제안에 대한 논문입니다. XPO는 온라인 탐색을 강화하여 기존 모델의 데이터를 넘어서는 답변을 생성하도록 합니다. 이를 통해 언어 모델의 학습과 성능을 극대화하려는 새로운 방법론입니다.

- **Technical Details**: XPO는 기존의 Direct Preference Optimization (DPO) 모델에 탐색 보너스를 추가하는 간단하면서도 효과적인 변경을 통해 구현됩니다. XPO는 KL-정규화된 마코프 결정 과정(KL-regularized Markov Decision Processes)을 이용하여 이론적으로 샘플 효율성을 증명하고, 초기 모델의 탐색 능력 여부에 관계없이 최적의 언어 모델 정책에 수렴하는 것을 보장합니다. 이 알고리즘은 Q-approximation와 Bellman 오류 최소화를 통해 견고한 이론적 기반을 확보하고 있습니다.

- **Performance Highlights**: 초기 실험에서 XPO는 비탐색적 DPO 변형보다 샘플 효율성이 뛰어난 것으로 나타났습니다. 이는 XPO가 적은 양의 선호 데이터로 더 나은 성능을 발휘할 수 있음을 시사합니다.



### Improved Techniques for Optimization-Based Jailbreaking on Large Language Models (https://arxiv.org/abs/2405.21018)
- **What's New**: 이 논문은 기존의 Greedy Coordinate Gradient (GCG) 공격 방법을 개선하여 최적화 기반의 LLM jailbreak 효율성을 높이기 위한 여러 기법을 제안합니다. 특히, 'Sure' 단일 템플릿의 한계를 극복하기 위해 다양한 해로운 자기 제안 및 가이던스를 포함한 타겟 템플릿을 적용하는 방안을 제시합니다.

- **Technical Details**: 기존 GCG 기법에서의 단일 타겟 템플릿이 공격 효율성을 제한하는 문제를 해결하기 위해, 다양한 타겟 템플릿을 적용하여 LLM을 오도할 수 있는 방안을 제안합니다. 또한, 자동 다수 좌표 업데이트 전략(automatic multi-coordinate updating strategy)과 어려운 초기화(easy-to-hard initialization) 기법을 도입하여 최적화 기반 공격의 수렴 속도를 높이고 성능을 향상시킵니다. 이러한 기법들을 결합하여 효율적인 jailbreak 방법인 ℐ-GCG를 개발했습니다.

- **Performance Highlights**: 실험 결과, 제안된 ℐ-GCG는 NeurIPS 2023 Red Teaming Track 등 다양한 벤치마크에서 기존 최첨단 공격 기법들을 능가하여 거의 100%의 공격 성공률을 달성했습니다. 이는 GCG 기반 방법의 성능을 크게 향상시킨 결과입니다.



### Towards a Fluid computer (https://arxiv.org/abs/2405.20999)
Comments:
          11 pages, 3 figures

- **What's New**: 이 논문에서는 3차원에서 '유체 컴퓨터'를 구축한 작업을 검토합니다. 이 구성은 상징적 동역학(symbolic dynamics)과 Etnyre와 Ghrist가 밝혀낸 정상적인 오일러 흐름과 접촉 기하학(contact geometry)을 결합했습니다. 추가로, 벡터 필드 벨트라미(Beltrami)을 생성하는 메트릭(metric)이 케른-해밀톤(Chern-Hamilton) 의미에서 비판적일 수 없다는 것을 논의합니다.

- **Technical Details**: 논문에서는 유체역학과 튜링 머신(Turing machine) 간의 연결을 탐구합니다. 3차원 리만 구(Riemannian sphere)에서 오일러 방정식(Euler equations)의 정상적인 해(stationary solutions)를 구축해 튜링 완전성(Turing completeness)을 입증했으며, 이는 비결정적인 경로(undecidable paths)를 보여줍니다. 다양한 기법, 특히 코시-코발레프스카야 정리(Cauchy-Kovalevskaya theorem)와 기울기 동적 시스템 이론을 사용하여 유클리드 공간(Euclidean space)에서 무한 에너지를 가진 비물리적 유체 흐름을 생성했습니다.

- **Performance Highlights**: 논문에서 제시한 주요 결과는 튜링 완전성을 가진 유체 흐름의 존재를 입증한 것과 이것이 비결정적 현상을 유발하는 경로를 포함한다는 점입니다. 또한, 이 구축은 벡터 필드 벨트라미(Beltrami)를 만드는 메트릭이 케른-해밀톤 함수(Chern-Hamilton energy functional)에 대해 비판적이지 않음을 보입니다.



### CWRCzech: 100M Query-Document Czech Click Dataset and Its Application to Web Relevance Ranking (https://arxiv.org/abs/2405.20994)
Comments:
          Accepted to SIGIR 2024

- **What's New**: 최근 발표된 연구에서는 체코어에 대한 클릭 웹 순위 데이터셋(CWRCzech)을 공개했습니다. 이 데이터셋은 검색 엔진 기록에서 수집된 사용자 행동 데이터를 포함하며, 1억 개의 쿼리-문서 쌍과 2760만 개의 클릭된 문서 및 1080만 개의 체류 시간을 포함하고 있습니다. 이는 비영리 학술 라이선스 하에 공개되었으며, 체코어 연구 커뮤니티에서 자유롭게 이용할 수 있습니다.

- **Technical Details**: CWRCzech는 체코어 검색 엔진 Seznam.cz의 로그에서 수집된 1억 개의 쿼리-문서 쌍을 포함합니다. 사용자 행동 데이터는 문서의 검색 결과 내 위치와 사용자의 클릭 및 체류 시간 정보를 제공합니다. 이 데이터셋은 긍정적 예시뿐만 아니라 사용자에게 제공되었으나 클릭되지 않은 부정적 예시도 포함하여, 모델 훈련에 유용한 자원을 제공합니다. 또한 연구의 평가 벤치마크로 사용할 수 있는 약 5만 개의 쿼리-문서 쌍을 수작업으로 주석 달아서 공개했습니다.

- **Performance Highlights**: CWRCzech 데이터셋을 사용하여 자동으로 수집된 사용자 행동 데이터 기반의 모델이 수작업으로 주석된 데이터 기반 모델보다 정확도가 높다는 것을 실험을 통해 확인했습니다. 이는 대규모 사용자 행동 데이터가 얼마나 유용한지를 보여주는 중요한 발견입니다. 이러한 자동 수집 데이터는 학습에 충분히 활용될 수 있으며, 특히 비영어권 데이터셋의 제한을 극복하는 데 주요한 역할을 할 수 있습니다.



### LCQ: Low-Rank Codebook based Quantization for Large Language Models (https://arxiv.org/abs/2405.20973)
Comments:
          10 pages, 5 figures

- **What's New**: 최근 많은 작업에서 인상적인 성능을 보인 대형 언어 모델(LLMs)의 고비용 저장과 계산 요구가 실제 응용에서 파견을 어렵게 하고 있습니다. 본 논문에서는 LLMs의 고효율 저장을 위해 새로운 가중치 양자화 방법인 LCQ(low-rank codebook based quantization)를 제안합니다. LCQ는 랭크가 하나보다 큰 저랭크 코드북을 양자화에 사용하고, 기존 방법보다 높은 정확도를 유지하면서도 저장 비용을 최소화 할 수 있습니다.

- **Technical Details**: LCQ는 포스트 트레이닝 양자화(PTQ) 방법을 사용하며, 특정 래그래디언트 기반 최적화 알고리즘을 통해 코드북의 파라미터를 최적화합니다. 양자화의 결과로, LCQ는 기존의 랭크-원 코드북에 비해 더 좋은 표현 능력을 가지며, 두 번의 양자화 전략을 채택하여 코드북의 저장 비용을 더욱 줄였습니다.

- **Performance Highlights**: 실험 결과에 따르면, LCQ는 기존의 다른 양자화 방법들에 비해 정확도가 우수하며, 추가적인 저장 비용도 거의 발생하지 않습니다. 특히 4비트 이하의 양자화에서도 기존 방법보다 성능 저하가 적습니다.



### Large Language Models are Zero-Shot Next Location Predictors (https://arxiv.org/abs/2405.20962)
- **What's New**: 최근 연구는 자연어 처리에서의 혁신적인 대형 언어 모델(LLMs)의 발전을 통해, 이들이 '제로 샷'(zero-shot) 방식으로 미래 위치 예측에서도 효과적일 수 있음을 보여줍니다. 연구팀은 이러한 모델들이 지리적 지식을 잘 구성하고 있다고 판단하여, Llama, GPT-3.5 및 Mistral 7B와 같은 인기 있는 LLM들을 평가했습니다. 이 연구는 제로 샷 방식의 다음 위치 예측에서 32.4%까지의 정확도를 달성하며 기존의 복잡한 딥러닝(DL) 모델들에 비해 600% 이상의 상대적 성능 향상을 보여주었습니다.

- **Technical Details**: 본 연구는 세 가지 실제 이동 데이터셋을 사용해, 여러 LLM의 다음 위치 예측 기능을 테스트했습니다. 특별히 설계된 '프롬프트'(prompt)를 통해 LLM들의 성능을 지리적 전이 및 일반화 가능성 측면에서 평가하였고, 적절한 테스트 프레임워크를 통해 데이터 오염 가능성을 최소화했습니다. 또한 텍스트 기반 설명기능을 탐구하고, 역사적 정보의 양이 성능에 미치는 영향을 분석했습니다.

- **Performance Highlights**: LLM들은 제로 샷 다음 위치 예측에서 최대 32.4%의 정확도를 달성했으며, 이는 기존의 복잡한 딥러닝 모델에 비해 600% 이상의 성능 향상을 보여줍니다. 또한, 역사적 방문 데이터 제공 시 성능이 최대 16.69%까지 향상될 수 있음을 확인했습니다. 일부 LLM들은 데이터 오염 없이도 안정적인 성능을 유지했습니다.



### A Robot Walks into a Bar: Can Language Models Serve asCreativity Support Tools for Comedy? An Evaluation of LLMs' Humour Alignment with Comedians (https://arxiv.org/abs/2405.20956)
Comments:
          15 pages, 1 figure, published at ACM FAccT 2024

- **What's New**: 에든버러 페스티벌 프린지 2023 (Edinburgh Festival Fringe 2023) 에서 'AI x Comedy' 워크숍을 통해 20명의 프로 코미디언을 인터뷰했습니다. 이들은 인공지능을 코미디 창작 과정에 활용합니다. 워크숍은 인공지능 대형 언어 모델 (LLMs)을 활용한 코미디 작성 세션, Creativity Support Index를 평가하기 위한 인간-컴퓨터 상호작용 설문 조사, 그리고 코미디언들이 인공지능을 사용하는 동기와 윤리적 우려를 논의하는 포커스 그룹으로 구성되었습니다.

- **Technical Details**: 연구는 주로 ChatGPT (OpenAI)와 Bard (Google)의 대형 언어 모델을 사용하였습니다. 인간 수준의 난이도를 요구하는 불협화음 생성 및 해결 작업을 기반으로 객관적인 평가를 진행했습니다. 참가자들은 LLM의 윤리적 문제, 편견, 검열, 저작권 문제에 대해 논의했으며, 이러한 문제들이 창의성 지원 도구로서 LLM의 유용성을 저하시킨다고 느꼈습니다.

- **Performance Highlights**: 참가자들은 현재의 LLM이 창의성 지원 도구로서 부적합하다고 평가했습니다. 이들은 기존의 안전 필터링 및 교육용 조정된 LLM이 소수 그룹의 시각을 지우고, 이를 일종의 검열로 간주하고 있습니다. 결과적으로, LLM이 생성한 코미디 자료는 '1950년대의 크루즈 선 코미디 자료와 같지만 인종차별적이지 않다'고 평가되었습니다. 이는 중요한 문화적 유산을 잃고 있는 것에 대해 우려를 나타냅니다.



### Enhancing Vision Models for Text-Heavy Content Understanding and Interaction (https://arxiv.org/abs/2405.20906)
Comments:
          5 pages, 4 figures (including 1 graph)

- **What's New**: 이번 논문은 텍스트가 많은 복잡한 시각 자료를 이해할 수 있는 비전 모델 (vision model) 능력을 향상시키는 방법을 다룹니다. 특히 교과서와 연구 논문과 같은 여러 이미지와 그래프, 표가 포함된 시각 자료를 학습하고 이해하는 기술을 개발했습니다. 이를 위해 데이터셋 전처리 (dataset preprocessing), 인스트럭션 중심의 데이터로 모델을 파인튜닝 (fine tuning), 및 평가를 수행했습니다. CLIP을 사용한 이미지 인코딩 (image encoding)과 Massive Text Embedding Benchmark의 모델을 통합한 시각 채팅 애플리케이션 (visual chat application)도 개발했습니다.

- **Technical Details**: 이 접근 방식은 데이터셋 전처리와 인스트럭션 중심 데이터로 모델을 파인튜닝하는 과정을 포함합니다. 또한, 이미지 인코딩을 위해 CLIP을 사용하고, 텍스트와 시각적 입력을 모두 고려하는 Massive Text Embedding Benchmark의 모델을 통합했습니다.

- **Performance Highlights**: 이 프로젝트는 96.71%의 정확도를 달성했습니다. 목표는 복잡한 시각적 텍스트 데이터와 상호 연결된 데이터를 이해하는 능력을 가진 멀티모달 AI (multimodal AI)를 발전시키는 것입니다.



### Outliers and Calibration Sets have Diminishing Effect on Quantization of Modern LLMs (https://arxiv.org/abs/2405.20835)
- **What's New**: 최근 연구는 큰 언어 모델 (LLMs)의 효율성을 향상시키기 위한 Post-Training Quantization (PTQ)에 초점을 맞추고 있으며, 특히 캘리브레이션 세트(calibration sets)가 hidden activations에 미치는 영향을 탐구합니다. 기존 OPT 모델과 달리, Llama-2 7B, Llama-3 8B, Command-R 35B 및 Mistral 7B와 같은 새로운 모델들은 안정적인 activations과 뛰어난 내구성을 보여주고 있습니다. 이는 PTQ 전략에서 중대한 변화를 요구할 수 있습니다.

- **Technical Details**: PTQ는 고정밀 모델의 가중치(weights)를 낮은 정밀도로 변환하여 메모리 및 성능 요구 사항을 줄여줍니다. PTQ 방법에는 zero-shot 방법과 one-shot 방법이 있으며, 후자는 캘리브레이션 세트를 사용하여 weights를 양자화하는 동안 성능을 유지합니다. 본 연구에서는 hidden activations의 크기와 outliers를 평가하는 데 캘리브레이션 세트가 중요함을 발견하였습니다. 특히, OPT 모델은 캘리브레이션 세트의 변화에 매우 취약한 반면, 새로운 모델들은 이러한 취약성이 크게 줄어들었습니다.

- **Performance Highlights**: Mistral 7B 모델은 outliers에 거의 면역이며 안정적인 activations을 보였습니다. Llama-2 7B, Llama-3 8B, Command-R 35B는 캘리브레이션 세트의 품질, 내용, 언어에 덜 민감한 것으로 나타났습니다. 이로 인해 최신 LLMs는 성능 저하 없이 더 나은 양자화가 가능해졌습니다. 이러한 발전은 PTQ 전략이 기존의 outlier 보호보다 추론 속도를 최적화하는 방향으로 이동해야 함을 시사합니다.



### Ovis: Structural Embedding Alignment for Multimodal Large Language Mod (https://arxiv.org/abs/2405.20797)
- **What's New**: 최근 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)에서 발생하는 텍스트와 비주얼 정보의 융합 문제를 해결하기 위해 새로운 아키텍처 'Ovis'가 제안되었습니다. Ovis는 기존의 MLLMs가 전통적으로 가진 구조적 텍스트 임베딩과 비연속적 비주얼 임베딩 간의 불일치 문제를 해결합니다. 이를 위해 Ovis는 학습 가능한 시각 임베딩(visual embedding) 테이블을 추가하여 구조적으로 정렬된 시각 임베딩을 구현합니다.

- **Technical Details**: Ovis는 비주얼 인코더의 과정에 학습 가능한 시각 임베딩 테이블을 통합하며, 각 이미지 패치는 이 테이블을 여러 번 참조합니다. 이 과정은 텍스트 임베딩을 생성하는 방식과 유사합니다. 또한, Ovis는 합리적 토큰(probabilistic token)을 생성하여 비주얼 패치의 연속적인 토큰을 시각 어휘 집합과의 유사성을 나타내는 토큰으로 매핑합니다. 이 토큰을 바탕으로 시각 임베딩 테이블을 여러 번 인덱싱하여 최종 시각 임베딩을 생성합니다.

- **Performance Highlights**: 다양한 멀티모달 벤치마크에서 Ovis는 동일한 파라미터 규모의 오픈소스 MLLMs를 능가하는 성능을 보였으며, 심지어 고성능의 독점 모델인 Qwen-VL-Plus를 전체적으로 상회하는 결과를 보였습니다. 특히, Ovis-14B 모델은 전반적인 멀티모달 벤치마크에서 독점 모델들과 비슷하거나 그 이상의 성능을 보여주었습니다.



### Cross-Modality Jailbreak and Mismatched Attacks on Medical Multimodal Large Language Models (https://arxiv.org/abs/2405.20775)
- **What's New**: 의료 분야에서 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 보안 취약성에 대한 연구가 중요한 가운데, 특히 임상 환경에서 사용할 수 있는 의료 멀티모달 대형 언어 모델(MedMLLMs)의 안전성에 대한 연구가 부족하다는 문제가 제기되었습니다. 본 논문은 임상 환경에서 MLLMs의 보안 취약성을 심층적으로 분석하고, 특히 2M-attack과 O2M-attack이라고 부르는 새로운 유형의 공격 방법을 정의하였습니다. 이 연구는 이러한 공격 방법을 활용하여 MedMLLMs의 보안성을 평가하고, 향후 임상 적용 시 필요한 강화된 보안 대책 마련의 필요성을 강조합니다.

- **Technical Details**: 연구의 주요 기여점으로는 다음과 같습니다. 첫째, 임상 환경에서 Mismatched와 Malicious 문제를 정의하며, 2M-attack과 O2M-attack이라는 새로운 공격 방법을 소개했습니다. 둘째, 다양한 의료 이미지를 포함한 Comprehensive Medical Safety dataset인 3MAD를 생성하여 다양한 임상 입력의 특징을 설명했습니다. 셋째, 텍스트와 이미지 데이터를 동시에 다루는 혁신적인 멀티모달 크로스 최적화(MCM) 전략을 도입하여 MedMLLM에 대한 공격 성공률을 높였습니다. 이를 통해 3MAD-66K와 3MAD-Tiny-1K 두 가지 데이터셋을 구성하여 다양한 공격 및 평가 시나리오에서 실험을 진행하였습니다.

- **Performance Highlights**: 백색 공격(white-box attacks)과 전달 공격(transfer attacks) 수행 결과, 최신 MedMLLM 모델 LLaVA-Med를 포함한 네 개의 최첨단 모델에서 높은 성공률을 보였습니다. 특히 MCM 최적화 방법은 전통적인 방법들보다 월등히 높은 공격 성공률을 기록했습니다. 이는 MedMLLMs가 여전히 보안 취약점에 노출되어 있으며, 안전 강화와 효율성 개선이 필요함을 시사합니다.



### Joint Embeddings for Graph Instruction Tuning (https://arxiv.org/abs/2405.20684)
Comments:
          Conference Preprint

- **What's New**: 최근 연구에서는 멀티모달 기능을 갖춘 대형 언어 모델(LLM)이 시각적 지침을 따르는 가상 비서로 발전하고 있으나, 그래프 형태 데이터를 처리하는 비서는 아직 개발되지 않았습니다. 이 논문은 LLM에 그래프 임베딩을 통합하여 사용자의 지시에 따라 그래프 기반 응답을 생성하는 새로운 방법을 제안합니다.

- **Technical Details**: 이 연구는 LLM에 그래프 임베딩(graph embeddings)을 추가하여, 그래프 이해 및 그래프 지시(Task Following)를 수행할 수 있는 심층 학습 모델을 개발했습니다. 제안된 방법은 그래프를 고정된 수의 임베딩으로 변환하고 이를 사용자 지시와 함께 LLM에 주입하여 모델을 훈련합니다. 이를 위해, 텍스트로 그래프를 표현하는 접근법 대신, 노드 특성 임베딩(node feature embedding)과 그래프 통합 임베딩(graph embedding)을 사용하는 방법을 채택했습니다.

- **Performance Highlights**: 텍스트로 그래프를 표현하는 접근법에 비해 제안된 방법은 성능이 현저히 우수하며, 그래프 크기가 커짐에 따라 성능 저하가 발생하지 않습니다. 또한 임베딩 계층에서만 작동하기 때문에, 사용되는 LLM 아키텍처에 의존하지 않아 더 높은 확장성을 가집니다.



### Unraveling and Mitigating Retriever Inconsistencies in Retrieval-Augmented Large Language Models (https://arxiv.org/abs/2405.20680)
Comments:
          ACL 2024 (findings)

- **What's New**: 이 연구는 Retrieval-Augmented Large Language Models (RALMs)와 retrieval-free 언어 모델(LMs) 간의 성능 일관성을 조사하며, 두 접근 방식 간의 예제 수준의 성능 변동성뿐만 아니라 서로 다른 retrievers 간에도 변동성이 존재함을 발견했습니다. 이를 해결하기 위해 Ensemble of Retrievers (EoR)라는 새로운 훈련 가능한 프레임워크를 도입했으며, 이는 다양한 지식 소스에서 적응적으로 검색하고 예측 모델의 예측 오류를 효과적으로 줄이는 것을 목표로 합니다.

- **Technical Details**: 연구는 Open Domain Question Answering (ODQA)을 벤치마크 과제로 사용하여, 서로 다른 지식 소스(예: 검색 엔진, Wikipedia) 및 다양한 처리 방법(예: 절단, 재정렬, 압축)에 기초하여 15개의 서로 다른 retrievers를 구축했습니다. RALM의 퇴화 행동을 Retriever Error, Extraction Error, Hallucination Error, Lucky Guess 네 가지 카테고리로 이론적으로 분해하고, 예제 수준에서의 오류 발생 행동을 조사했습니다. 이를 통해 EoR 프레임워크를 설계했으며, 이는 다양한 retrievers에서 샘플링하고 응답의 유사성을 측정하는 투표 메커니즘을 기반으로 오류를 최소화합니다.

- **Performance Highlights**: ODQA 실험에서 EoR은 단일 retriever를 사용하는 RALM에 비해 성능이 크게 향상되었습니다. EoR은 다양한 지식 소스에서 적응적으로 정보를 검색하고 여러 retrievers의 응답을 비교하여 일관되지 않은 행동을 줄임으로써 성능 일관성을 크게 개선했습니다.



### Position Coupling: Leveraging Task Structure for Improved Length Generalization of Transformers (https://arxiv.org/abs/2405.20671)
Comments:
          73 pages, 20 figures, 90 tables

- **What's New**: Transformers는 긴 시퀀스를 일반화하는 데 어려움이 있었으며, 이 문제를 해결하기 위해 새로운 '포지션 커플링(position coupling)' 방법을 제안합니다. 이 방법은 작업의 구조를 Transformer의 포지션 인코딩에 직접 포함시켜, 훈련 중에 경험하지 않은 긴 시퀀스에 대해 일반화할 수 있도록 돕습니다.

- **Technical Details**: 전통적인 절대적 위치 메커니즘은 각 토큰에 고유한 포지션 ID를 할당하지만, 제안된 포지션 커플링에서는 의미적으로 관련된 여러 개의 토큰에 동일한 포지션 ID를 할당합니다. 예를 들어, 정수 덧셈 작업에서는 동일한 자릿수의 숫자를 동일한 포지션에 묶습니다. 이러한 구조는 Transformer가 입력 시퀀스의 길이에 상관없이 작업을 학습할 수 있게 합니다.

- **Performance Highlights**: 제안된 포지션 커플링을 사용하여 1레이어 Transformer를 1~30자리 덧셈으로 훈련하면 최대 200자리 덧셈(6.67배 확장)까지 일반화할 수 있습니다. 또한, 이론적으로 1레이어 Transformer가 포지션 커플링을 사용할 경우 지수적으로 많은 자릿수를 가진 덧셈 작업을 해결할 수 있음을 증명했습니다. 이는 기존 Transformer로는 불가능했던 성과입니다.



### Shotluck Holmes: A Family of Efficient Small-Scale Large Language Vision Models For Video Captioning and Summarization (https://arxiv.org/abs/2405.20648)
- **What's New**: 이 논문에서는 영상 요약 및 자막 생성의 성능을 향상시키기 위해 Shotluck Holmes라는 효율적인 대규모 언어 비전 모델(LLVMs)을 제안합니다. 이 모델은 향상된 사전 학습 및 데이터 수집 전략을 활용하여 기존 소형 LLVMs의 프레임 이해 능력을 확장합니다. 특히 Shotluck Holmes는 Shot2Story20K 데이터셋에서 최첨단 성능(SOTA)을 능가하는 결과를 보여줍니다.

- **Technical Details**: Shotluck Holmes는 Shot2Story20K의 세 단계 모델 파이프라인 중 마지막 두 단계를 대체하여 메모리 사용량, 연산 용량 및 지연 시간을 크게 줄이면서도 뛰어난 성능을 유지합니다. TinyLLaVA라는 소형 다중모델(Multi-modal) 모델 패밀리를 활용하여 Shot2Story20K의 데이터셋을 미세 조정함으로써 이러한 목표를 달성합니다. 비디오는 텐서로 변환한 후 Visual encoder에 입력되며, 샘플링 방법으로는 Uniform sampling과 Head-tail sampling을 사용합니다.

- **Performance Highlights**: 제안된 Shotluck Holmes 모델은 Shot2Story20K 데이터셋의 싱글샷 비디오 자막 및 멀티샷 비디오 요약 작업에서 최첨단 성능(SOTA)을 달성했습니다. 특히 메모리 사용량과 연산 효율성을 크게 개선하여 훨씬 작은 모델 크기에도 불구하고 높은 성능을 기록했습니다. 두 가지 모델, 1.5B와 3.1B 파라미터를 가지는 LLM이 제시되었습니다.



### Large Language Models Enhanced Sequential Recommendation for Long-tail User and Item (https://arxiv.org/abs/2405.20646)
- **What's New**: Sequential recommendation systems (SRS)은 사용자의 과거 상호작용을 바탕으로 이후의 선호도를 예측하는 목적으로 사용됩니다. 본 연구에서는 대형 언어 모델 (Large Language Models, LLMs)를 활용하여 기존의 SRS 성능을 향상시키는 'LLM-ESR' 프레임워크를 소개합니다. 이 프레임워크는 더 높은 컴퓨팅 오버헤드를 초래하지 않으면서 SRS의 성능을 향상시킵니다.

- **Technical Details**: 본 연구에서는 LLM의 시맨틱 임베딩(semantic embedding)을 활용하여 두 가지 주요 문제, 즉 롱테일 사용자(long-tail user)와 롱테일 아이템(long-tail item) 문제를 해결하고자 합니다. 롱테일 아이템 문제에 대처하기 위해, 우리는 LLM의 시맨틱 정보와 전통적인 SRS의 협력 신호를 결합하는 듀얼 뷰 모델링 접근법을 제안합니다. 롱테일 사용자 문제를 해결하기 위해서는, 유사한 사용자의 풍부한 상호작용 데이터를 통합하여 사용자 선호도 표현을 개선하는 검색 증강 자기 증류(retrieval augmented self-distillation) 기법을 도입하였습니다.

- **Performance Highlights**: 세 가지 실제 데이터셋과 세 가지 널리 사용되는 SRS 모델을 사용하여 포괄적인 실험을 수행한 결과, 제안한 프레임워크는 기존 방법론보다 우수한 성능을 보였습니다.



### ToxVidLLM: A Multimodal LLM-based Framework for Toxicity Detection in Code-Mixed Videos (https://arxiv.org/abs/2405.20628)
Comments:
          ACL Findings 2024

- **What's New**: 최근 인터넷 기술의 급속한 발전으로 멀티모달 콘텐츠, 특히 비디오의 폭증이 온라인 커뮤니케이션의 지평을 확장시켰습니다. 본 논문은 코드 믹스된 저자원 언어로 된 비디오 콘텐츠의 유해 콘텐츠 감지를 위한 최초의 벤치마크 데이터셋을 소개합니다. 유튜브에서 수집된 931개의 비디오와 4021개의 힌디-영어 코드 믹스된 발화를 포함하며, 각 발화는 독성, 심각도 및 감정 레이블로 주석이 달려 있습니다.

- **Technical Details**: 우리의 Multimodal Multitask 프레임워크는 거대 언어 모델(LLMs)을 활용하여 유해 콘텐츠 감지, 감정 분석, 심각도 분석을 수행합니다. ToxVidLLM은 3가지 주요 모듈, 즉 인코더 모듈, 교차 모달 동기화 모듈 및 멀티태스크 모듈로 구성되어 있습니다. 또한, 텍스트 모달리티를 다른 모달리티와 동기화하는 방법을 제안하며, 이는 다양한 사전 훈련된 모델과도 적응할 수 있습니다.

- **Performance Highlights**: 다중 모달리티를 포함한 실험 결과, 유해 콘텐츠 감지에서 94.29%의 정확도와 94.35%의 가중치 F1 점수를 달성했습니다. 또한, 가장 효과적인 멀티태스크 모델은 독성 감지에서 94.35%, 심각도 분석에서 86.84%, 감정 식별에서 83.42%의 가중치 F1 점수를 달성했습니다.



### Bi-Directional Transformers vs. word2vec: Discovering Vulnerabilities in Lifted Compiled Cod (https://arxiv.org/abs/2405.20611)
Comments:
          8 pages, 0 figures, IEEE 4th Cyber Awareness and Research Symposium 2024 (CARS'24)

- **What's New**: 이 연구는 고급 코드 구조와 같은 정보가 손실되기 때문에 컴파일된 바이너리에서 취약점을 탐지하는 것이 어려운 문제를 해결합니다. 이 연구는 자연어 처리(NLP) 임베딩 기법을 사용하여 중간 표현(LLVM) 코드로부터 의미론을 학습하고, 이를 통해 취약점을 탐지하는 방법을 탐구합니다. 최근에는 word2vec와 같은 상대적으로 간단한 임베딩 모델이 BERT와 RoBERTa와 같은 복잡한 양방향 트랜스포머 모델보다 더 나은 성능을 보였다는 결과가 나왔습니다.

- **Technical Details**: 연구자는 Juliet 데이터셋의 약 118,000개의 LLVM 함수로부터 인코더를 생성하여 word2vec, BERT, RoBERTa 모델로 임베딩을 학습했습니다. 이 임베딩을 사용하여 LSTM 신경망을 훈련시키고 성능을 비교했습니다. 연구는 bidirectional transformer 기반의 모델(BERT, RoBERTa)의 성능을 단순한 word2vec 모델과 비교하는 데 중점을 두었습니다. 특히, 이들은 LLVM 코드에서 의미를 학습하며 취약점을 탐지하는 데 있어 각 임베딩 모델의 효과를 평가했습니다.

- **Performance Highlights**: word2vec의 Continuous Bag of Words(CBOW) 모델은 취약점 탐지에서 92.3%의 검증 정확도를 달성했으며, word2vec의 Skip-Gram, BERT, RoBERTa 모델보다 우수한 성능을 보였습니다. 이는 제한된 데이터 샘플 수(예: 118K)를 사용하여 양방향 트랜스포머 기반 모델을 훈련시킬 때 복잡한 맥락적 NLP 임베딩이 단순한 word2vec 모델에 비해 이점을 제공하지 못할 수 있음을 시사합니다.



### Masked Language Modeling Becomes Conditional Density Estimation for Tabular Data Synthesis (https://arxiv.org/abs/2405.20602)
- **What's New**: 이 논문에서는 이질적(혼합형) 테이블 데이터셋에 대한 고도의 기계 학습 효용성(MLu)을 가진 합성 데이터를 생성하는 방법을 제안합니다. 이를 위해, 학생 득점 데이터를 기반으로 새로운 천연 언어 모델링 방법으로 다중 클래스 분류 문제를 히스토그램 기반의 비모수적 조건밀도 추정 문제로 재정의한 MaCoDE 모델을 제시합니다.

- **Technical Details**: MaCoDE는 Masked Language Modeling(MLM)을 활용하여 조건 변수와 대상 변수 간의 조건 밀도를 추정하는 방법입니다. 이 모델은 히스토그램 기반 비모수적 방법을 통해 혼합형 테이블 데이터의 조건 밀도를 정확히 추정합니다. 연속적인 열 데이터의 경우, 누적 분포 함수(Cumulative Distribution Function, CDF)를 사용하여 값을 [0,1] 구간으로 변환합니다. 생성되는 데이터의 열 순서는 무작위로 지정되어, 기존의 자동 회귀 밀도 추정기와는 차이가 있습니다.

- **Performance Highlights**: 실제 데이터셋 10개를 사용한 실험에서, 제안된 모델은 높은 기계 학습 효용성을 보였습니다. 또한, 다양한 결측 데이터 메커니즘 하에서도 다중 대체 성능을 평가하여, 제안된 모델이 결측 데이터 처리에 유리함을 확인했습니다. 이는 또한 데이터 프라이버시 수준을 조정할 수 있는 이점을 제공합니다.



### Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models (https://arxiv.org/abs/2405.20541)
- **What's New**: 이 연구에서는 소형 언어 모델(Small Language Models)을 이용해 대규모 텍스트 데이터셋을 잘라내어(deep pruning) 더 큰 언어 모델(Larger Language Models)의 성능을 향상시킬 수 있는지 조사했습니다. 특히, 소형 모델이 거대 모델의 퍼플렉시티(perplexity)를 기반으로 데이터를 정렬하고, 가지치기(pruning) 방법론이 데이터 도메인 구성에 미치는 영향을 분석했습니다.

- **Technical Details**: 연구진은 데이터 가지치기(pruning) 과정에서 퍼플렉시티(perplexity)를 측정하여 품질을 평가합니다. 이를 위해 소형 언어 모델(125백만 파라미터)을 학습시켜 데이터셋의 샘플 퍼플렉시티를 계산하고, 해당 퍼플렉시티를 기반으로 고품질 샘플만을 선택하여 대형 모델(3억 파라미터)의 사전학습 데이터로 사용합니다. 실험은 'The Pile'과 'Dolma'라는 다양한 도메인의 데이터셋을 사용해 진행되었습니다.

- **Performance Highlights**: 퍼플렉시티 기반 데이터 가지치기를 사용한 결과, 소형 모델로 데이터를 정제함으로써 대형 모델의 다운스트림 성능이 최대 2.04점 향상되었으며, 사전 학습 단계도 최대 1.45배 줄일 수 있었습니다. 또한, 과도하게 훈련된 상황(over-trained)이나 데이터가 제한된 환경(data-constrained)에서도 퍼플렉시티 기반 가지치기가 유효함을 확인했습니다.



### Unveiling the Impact of Coding Data Instruction Fine-Tuning on Large Language Models Reasoning (https://arxiv.org/abs/2405.20535)
- **What's New**: Instruction Fine-Tuning (IFT)에 있어서 코딩 데이터가 LLM(대형 언어 모델)의 추론 능력에 미치는 영향을 심층 분석했습니다. 이 연구는 코딩 데이터의 비율, 모델 계열, 크기, 추론 도메인 등 다양한 측면에서 코딩 데이터가 LLM의 추론 능력에 어떻게 영향을 미치는지 탐구합니다.

- **Technical Details**: 본 연구는 ShareGPT(ShareGPT, 2023) 데이터를 활용하여 코딩 데이터의 비율을 조절한 세 가지 IFT 데이터셋을 만들었습니다. 여섯 가지 다른 모델 계열과 규모의 LLM(Llama-1, Llama-2, Llama-3, Mistral, Qwen-1.5, Gemma)을 이 데이터셋으로 미세조정(fine-tuning)하여 상징적(symbolic), 논리적(logical), 산술적(arithmetic) 세 개의 추론 도메인에서 열두 가지 과제를 평가했습니다.

- **Performance Highlights**: 코딩 데이터 튜닝은 다양한 모델 계열과 규모에서 전체 추론 능력을 향상시켰으며, 특히 Mistral-7B 모델에서 최대 10.3 퍼센트 포인트의 향상을 보였습니다. 추론 도메인별로는 상징적 추론에서 큰 성능 향상이 있었으나, 산술적 추론에서는 상이한 결과를 보였습니다. 과제별 성능 면에서는 모델 계열 간 비교적 일관된 증진 효과가 나타났으나, 최적의 코딩 데이터 비율은 과제별로 달랐습니다.



### An Automatic Question Usability Evaluation Toolk (https://arxiv.org/abs/2405.20529)
Comments:
          Artificial Intelligence in Education 2024

- **What's New**: SAQUET (Scalable Automatic Question Usability Evaluation Toolkit)는 다중 선택 질문(MCQ)의 품질 평가를 자동화하고 더욱 깊이 있는 결함을 분석하기 위해 개발된 오픈 소스 도구입니다. 최신 언어 모델(GPT-4 등), 고급 단어 임베딩(word embeddings), 트랜스포머(Transformers) 기술을 활용하여 다양한 분야의 MCQ를 평가합니다.

- **Technical Details**: SAQUET는 Item-Writing Flaws (IWF) 루브릭을 기반으로 MCQ의 품질을 종합적으로 평가합니다. 이 도구는 텍스트 복잡성 분석을 위해 GPT-4 및 다양한 트랜스포머 모델을 사용하며, 다른 자동 평가 메트릭스와 기존의 인간 평가 사이의 차이를 보여줍니다.

- **Performance Highlights**: SAQUET는 화학(Chemistry), 통계(Statistics), 컴퓨터 과학(Computer Science), 인문학(Humanities), 헬스케어(Healthcare) 등 다양한 분야의 MCQ 데이터를 평가해 94% 이상의 정확도로 인간 평가자가 식별한 결함을 감지했습니다. 이는 기존의 평가 방법의 한계를 부각시키고 교육 평가의 품질 향상 가능성을 보여줍니다.



### Automated Generation and Tagging of Knowledge Components from Multiple-Choice Questions (https://arxiv.org/abs/2405.20526)
Comments:
          Learning @ Scale 2024

- **What's New**: 본 연구에서는 고등교육 과정의 다지선다형 질문(MCQs)에 대한 Knowledge Component(KCs) 생성을 단순화하기 위해 GPT-4를 활용했습니다. 화학과 이러닝 분야에서 도메인 전문가가 만든 KCs와 대규모 언어 모델(LLM)이 생성한 KCs를 비교하고 평가했습니다. 평가 결과, 비일치하는 KCs의 경우 평가자들은 인간이 생성한 것보다 LLM이 생성한 KCs를 더 선호하는 경향을 보였습니다.

- **Technical Details**: GPT-4를 사용하여 화학 및 이러닝 다지선다형 질문의 KCs를 생성하였고, 생성된 KCs는 세 명의 도메인 전문가에 의해 평가되었습니다. 또한, 온톨로지 유도 알고리즘을 개발하여 유사한 KCs를 평가하는 질문들을 군집화(clustering)했습니다. 이 알고리즘은 명시적 레이블이나 컨텍스트 정보 없이도 질문들을 성공적으로 분류하였습니다.

- **Performance Highlights**: 가장 효과적인 LLM 전략은 화학 질문에서 56%, 이러닝 질문에서 35%의 KCs를 정확하게 매칭했습니다. 상위 5개 KC 제안을 고려할 경우 성공률은 더 높았습니다. 인간 평가자들은 두 영역 모두에서 통계적으로 유의미하게 LLM이 생성한 KCs를 인간이 할당한 것보다 약 2/3의 비율로 더 선호했습니다.



### Phantom: General Trigger Attacks on Retrieval Augmented Language Generation (https://arxiv.org/abs/2405.20485)
- **What's New**: 이번 연구는 Retrieval Augmented Generation (RAG) 시스템의 새로운 공격 원인을 제안합니다. RAG 시스템은 LLM(대형 언어 모델)에 외부 지식 데이터베이스를 참조하게 하여 더욱 개인화된 생성 결과를 제공하지만, 새로운 보안 위험을 초래할 수 있습니다. 연구팀은 Phantom이라는 공격 프레임워크를 설계하여, 한 개의 악성 문서를 지식 데이터베이스에 삽입함으로써 RAG 시스템을 조작할 수 있는 방법을 제시했습니다.

- **Technical Details**: Phantom은 두단계 공격 프레임워크입니다. 첫 번째 단계는 특정 단어 시퀀스를 포함하는 사용자의 쿼리에 의해 호출될 독성문서를 최적화하는 것입니다. 이 문서는 RAG 시스템이 상위k개의 결과로 반환할 때만 검색됩니다. 두 번째 단계에서는 LLM 생성기를 조작하여 서비스 거부, 명성 손상, 프라이버시 침해, 유해한 행동 등을 유발하는 악성 문자열을 삽입합니다. 이 공격은 Gemma, Vicuna, Llama 같은 다양한 LLM 아키텍처에서 시연되었습니다.

- **Performance Highlights**: Phantom은 RAG-보강 LLM 시스템을 성공적으로 조작하여 서비스 거부 공격, 편향된 의견 생성, RAG의 지식 데이터베이스에서 민감한 정보 추출, 유해한 텍스트 생성 등 다양한 공격을 수행할 수 있음을 보여주었습니다. RAG 시스템의 리트리버와 생성기를 동시에 최적화하여, 단일 독성 문서를 통해 여러 악성 목적을 달성할 수 있다는 점에서 현재의 공격 기법을 크게 개선했습니다.



### Enhancing Antibiotic Stewardship using a Natural Language Approach for Better Feature Representation (https://arxiv.org/abs/2405.20419)
- **What's New**: 이 연구는 전자 건강 기록(EHRs)을 통합하여 임상 결정 지원 시스템을 개선하고, 항생제 저항성 문제를 해결하기 위한 방법을 탐구합니다. 기록 데이터를 시리얼화된 텍스트 표현으로 변환하고, 사전 학습된 기반 모델(pretrained foundation models)을 사용하여 항생제 감수성 예측을 수행합니다.

- **Technical Details**: 이 연구는 EHR 데이터를 시리얼화된 텍스트 형태의 '의사 노트(pseudo-notes)'로 변환해 특징 표현을 개선합니다. 이를 통해 사전 학습된 기반 모델들은 풍부한 특징 표현을 활용하여 데이터를 분석합니다. MIMIC-IV 및 MIMIC-IV 응급실 데이터베이스를 사용해 특정 감염 및 항생제 감수성을 분석했습니다. 데이터는 진단, 약물, 생리 기록 등 여섯 가지 임상 양식을 포함합니다.

- **Performance Highlights**: 연구 결과, 시리얼화된 텍스트 표현과 기반 모델을 결합하면 해석 가능성을 높이고, 항생제 감수성 예측의 정확성을 크게 개선할 수 있는 것으로 나타났습니다. 이 접근법은 임상 결정 지원 시스템을 강화하고, 최적의 항생제를 식별하는 데 도움을 줄 수 있습니다.



### Jailbreaking Large Language Models Against Moderation Guardrails via Cipher Characters (https://arxiv.org/abs/2405.20413)
Comments:
          20 pages

- **What's New**: 최근 출시된 논문에서 JAMBench라는 새로운 벤치마크를 소개했습니다. 이는 LLMs(대규모 언어 모델)에서 'moderation guardrails'를 테스트하고 평가하는 데 초점을 맞췄습니다. JAMBench는 LLM의 보안 조치를 우회하는 'jailbreaks'에 대한 평가를 강화하기 위해 설계되었습니다.

- **Technical Details**: JAMBench는 네 가지 주요 위험 범주에 걸쳐 다양한 수준의 심각도를 가진 160개의 수작업으로 작성된 질문을 포함합니다. 이 벤치마크는 Hate and Fairness(증오 및 공정성), Sexual(성적), Violence(폭력), Self-Harm(자해) 등 네 가지 주요 영역을 다룹니다. 이 논문은 'Jailbreak Against Moderation' (JAM)이라는 새로운 jailbreak 방법을 제안하며, 이는 입력 수준 필터와 출력 수준 필터를 우회하기 위해 'cipher characters'를 생성하기 위한 미세 조정된 모델을 사용합니다.

- **Performance Highlights**: JAM은 기존 벤치마크와 비교하여 약 19.88배 높은 우회 성공률과 약 1/6의 낮은 차단율을 달성했습니다. 네 가지 LLMs (GPT-3.5, GPT-4, Gemini, Llama-3)에 대한 실험 결과 JAM의 효과와 높은 이식성을 입증했습니다. 또한, JAM에 대해 성공적으로 방어할 수 있는 두 가지 잠재적 대응책을 제안하여 추가적인 'guardrails'의 중요성을 강조했습니다.



### Literature Filtering for Systematic Reviews with Transformers (https://arxiv.org/abs/2405.20354)
- **What's New**: 해당 연구에서는 자연 언어로 표현된 연구 질문과 검색된 논문 세트를 매칭하여 필터링 시스템을 개발했습니다. 이전에는 체계적인 리뷰를 수행하는 데 상당한 시간이 소요되었으나, 이 과정을 효율화할 수 있는 방법을 제안했습니다.

- **Technical Details**: 이 연구에서는 변형 모델(transformer models)을 사용했습니다. 특히, BioBERT(Lee2020) 버전의 BERT Transformer(Devlin2019)를 사용하여 생의학 문헌으로 사전 학습한 후 특정 작업에 맞게 정밀 조정하였습니다. 모델은 연구 질문과 관련된 불필요한 논문들을 제거하는 데 효과적이었습니다.

- **Performance Highlights**: 연구 결과, 변형 모델은 대부분의 연구 질문에 대해 불필요한 논문들을 대량으로 제거할 수 있음을 보여주었습니다. 이는 체계적인 리뷰 프로세스를 크게 개선할 가능성을 시사합니다. 또한, 인간 리뷰어의 초기 라벨링 데이터 없이도 일반 목적의 필터로서 작동할 수 있음을 확인했습니다.




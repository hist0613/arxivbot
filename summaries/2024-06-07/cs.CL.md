### PaCE: Parsimonious Concept Engineering for Large Language Models (https://arxiv.org/abs/2406.04331)
Comments:
          26 pages, 17 figures, 5 tables, dataset and code at this https URL

- **What's New**: 이번 아카이브 논문에서는 대규모 언어 모델(LLMs)의 정렬(alignment) 문제를 해결하기 위해 새로운 방법론인 Parsimonious Concept Engineering(PaCE)을 제안합니다. PaCE는 기존의 정렬 방법들이 겪고 있는 문제들을 해결하며, 언어 모델의 언어적 능력을 유지하면서도 정렬 목표를 효과적으로 달성합니다.

- **Technical Details**: PaCE는 두 단계로 구성됩니다: 개념 사전 구성 및 분할(Concept Dictionary Construction and Partition)과 활성화 분해 및 개입(Activation Decomposition and Intervention)입니다. 첫 번째 단계에서는 초기 대규모 개념 사전, PaCE-1M을 구성하고, 정렬 작업에 따라 개념을 양호하거나 바람직하지 않은 것으로 자동으로 분할합니다. 두 번째 단계에서는 희소 코딩(sparse coding) 기술을 사용하여 사용자 입력 프롬프트의 활성화를 분해하고, 바람직하지 않은 구성 요소를 제거하여 모델의 행동을 정렬 목표에 맞게 재조정합니다.

- **Performance Highlights**: PaCE는 응답 해독화(response detoxification), 신뢰도 향상(faithfulness enhancement), 및 감정 수정(sentiment revising)과 같은 여러 정렬 작업에서 최첨단 성능을 달성하는 동시에 언어 모델의 언어적 능력을 유지합니다. 또한 PaCE-1M의 개념 방향은 기하학적으로 일관되며, 활성화의 의미를 잘 나타낸다고 평가되었습니다.



### What Languages are Easy to Language-Model? A Perspective from Learning Probabilistic Regular Languages (https://arxiv.org/abs/2406.04289)
Comments:
          Accepted to ACL 2024

- **What's New**: 이 연구는 대형 언어 모델(LMs)이 배울 수 있는 것에 대해 탐구합니다. 기존의 이론적 접근과 달리, 이 연구는 LMs가 실제로 배우는 능력을 경험적으로 평가합니다. 연구는 특히 순환 신경망(RNN)과 Transformer LMs가 규칙적 언어 모델(RLMs)을 얼마나 잘 학습할 수 있는지를 조사합니다.

- **Technical Details**: 연구는 확률적 유한 상태 오토마타(PFSA)에서 샘플링된 학습 데이터를 통해 RNN와 Transformer 언어 모델의 학습 능력을 평가합니다. 2100개의 무작위로 생성된 PFSA에서 샘플링된 20,000개의 문자열 데이터를 사용하여, 다양한 숨겨진 상태 크기를 갖는 15,000개의 RNN 및 Transformer 모델을 훈련시켰습니다. 모델의 학습 능력은 KL divergence를 통해 평가되었습니다.

- **Performance Highlights**: 실험 결과, RLM의 순위와 샘플 문자열의 예상 길이가 RNN 및 Transformer 모델 모두에서 중요한 학습능력 예측 변수임을 발견했습니다. 여러 다른 예측 변수들도 학습능력에 영향을 미쳤지만, 그 패턴은 RNN과 Transformer 모델 간에 다르게 나타났습니다. 또한, RNN이 Transformer보다 공식적인 언어를 더 잘 모델링하는 경향이 있음을 확인했습니다.



### ABEX: Data Augmentation for Low-Resource NLU via Expanding Abstract Descriptions (https://arxiv.org/abs/2406.04286)
Comments:
          ACL 2024 Main Conference. Code and data: this https URL

- **What's New**: ABEX는 저자원이 필요한 자연어 이해(NLU) 작업을 위한 혁신적이고 효과적인 생성적 데이터 증강 방법론입니다. ABSTRACT-AND-EXPAND라는 새로운 패러다임을 활용하여 입력 문서를 다양한 형태로 변환시킵니다. 문서를 간결한 추상적 설명으로 변환한 후, 이를 확장하여 새로운 문서를 생성하는 방식입니다. 이는 문서의 원래 의미와 스타일을 유지하면서도 다양한 생성을 가능하게 합니다.

- **Technical Details**: ABEX는 BART 모델을 대규모 합성 데이터셋에서 학습시켜 추상적 설명을 확장하는 작업을 수행합니다. 문서의 추상적 설명을 생성하기 위해 AMR 그래프 편집을 기반으로 간단하고 제어 가능한 방법을 제안합니다. 또한, 두 문장의 AMR 그래프를 혼합하여 추상적 설명의 다양성을 높이는 방법도 도입했습니다.

- **Performance Highlights**: ABEX는 4개의 NLU 작업과 12개의 데이터셋, 4개의 저자원 설정에서 다른 모든 기준선들을 능가하는 성과를 보여주었습니다. 성능 향상 폭은 0.04%에서 38.8%에 이릅니다. 또한, ABEX에 의한 생성은 맥락, 토큰, 길이 다양성 측면에서 이전 방법들을 능가합니다.



### Characterizing Similarities and Divergences in Conversational Tones in Humans and LLMs by Sampling with Peop (https://arxiv.org/abs/2406.04278)
Comments:
          Accepted to Main Conference at ACL 2024

- **What's New**: 대화 톤(Conversational tones)은 대화의 효과적 전달을 위해 중요한 요소입니다. 최근 대형 언어 모델(LLMs)의 보급이 증가하면서, 인간과 LLMs 간의 대화 톤 차이를 분석하는 것이 필수적이 되었습니다. 이 연구는 인지 과학(Cognitive Science)의 방법에서 영감을 받아, 대화 톤과 문장을 동시에 이끌어내는 반복적 방법을 제안합니다. 인간과 GPT-4를 대상으로 100회 반복 실험을 통해 문장과 빈번한 대화 톤을 수집했습니다. 추가 실험에서는 인간과 GPT-4 모두가 모든 문장을 모든 톤으로 주석(annotation) 달았습니다.

- **Technical Details**: 연구는 두 가지 과제를 번갈아 수행하는 반복적 방법을 제안합니다: (1) 한 참가자는 주어진 문장의 톤을 식별하고, (2) 다른 참가자는 그 톤에 기반한 문장을 생성합니다. 수집된 데이터는 1339명의 인간 참가자, 33,370개의 인간 판단, 그리고 29,900개의 GPT-4 쿼리를 포함합니다. 이를 바탕으로 대화 톤 간의 관계를 해석할 수 있는 기하학적 표현을 만들어냈습니다.

- **Performance Highlights**: 이 접근법을 통해 인간과 GPT-4 사이의 대화 톤 관계를 효과적으로 분석할 수 있음을 보여주었습니다. 또한, 기계 학습(Machine Learning)과 인지 과학의 아이디어를 결합하여 인간-컴퓨터 상호작용의 문제를 해결할 수 있음을 시사합니다.



### Buffer of Thoughts: Thought-Augmented Reasoning with Large Language Models (https://arxiv.org/abs/2406.04271)
Comments:
          Project: this https URL

- **What's New**: Buffer of Thoughts (BoT)은 다양한 과제에서 대형 언어 모델(LLMs)의 정확성, 효율성, 안정성을 향상시키기 위한 혁신적이고 유연한 사고 강화 추론 접근법입니다. BoT는 사고 템플릿(thought-template)을 저장하는 메타 버퍼(meta-buffer)를 도입하고, 각 문제에 대한 관련 사고 템플릿을 동적으로 검색하여 제시된 문제 해결에 적용합니다. 또한, 버퍼 매니저(buffer-manager)를 사용하여 메타 버퍼를 동적으로 업데이트하여 확장성과 안정성을 보장합니다. 이 방법은 이전 SOTA(State Of The Art) 방법들에 비해 뛰어난 성능 향상을 보여줍니다.

- **Technical Details**: BoT는 세 가지 핵심 구성 요소를 가집니다: (i) 메타-버퍼: 여러 문제 해결 과정을 통해 추출된 보편적인 고수준 사고 템플릿(thought-template)을 저장하는 라이브러리, (ii) 버퍼 관리자(buffer-manager): 해결된 더 많은 작업에 따라 메타 버퍼의 용량을 동적으로 향상하는 관리 시스템, (iii) 문제 제공기(problem-distiller): 작업 관련 정보를 추출하여 메타 버퍼에서 관련 사고 템플릿을 검색하는 시스템. 이는 단일 및 다중 쿼리 추론 방법의 한계를 극복하며, 사고 템플릿을 사용하여 효율적으로 특정 추론 구조를 구현합니다.

- **Performance Highlights**: 총 10개의 도전적인 추론 중심 작업에서 BoT 방식을 이용한 실험 결과, 이전 SOTA 방법들에 비해 뛰어난 성능 향상을 기록했습니다: Game of 24에서 11%, Geometric Shapes에서 20%, Checkmate-in-One에서 51%의 성능 향상을 달성했습니다. 또한 평균적으로 멀티 쿼리 방식의 12% 비용만을 요구하면서도 높은 정확성, 효율성 및 모델 안정성을 보여주었습니다. 특히, Llama3-8B+BoT가 Llama3-70B 모델을 능가할 잠재력을 보여주었습니다.



### Transformers need glasses! Information over-squashing in language tasks (https://arxiv.org/abs/2406.04267)
- **What's New**: 이 논문은 디코더 전용 트랜스포머(Decoder-only Transformers)에서 정보가 어떻게 전파되는지에 대해 연구합니다. 저자들은 마지막 토큰의 표현(representation)이 특정 입력 시퀀스에 대해 매우 유사해지는 '표현 붕괴(Representational Collapse)' 현상을 이론적으로 증명합니다. 이 현상은 낮은 정밀도의 부동 소수점 형식(floating-point formats)을 사용함으로써 더 악화됩니다. 이러한 문제로 인해 모델은 특정 시퀀스를 다르게 처리하는 데 실패하게 되어 복사나 카운팅과 같은 작업에서 오류가 발생할 수 있습니다.

- **Technical Details**: 트랜스포머의 마지막 레이어에서 마지막 토큰의 표현을 분석하여 정보가 어떻게 전파되는지 탐구했습니다. 이 이론적 분석은 마지막 토큰의 표현이 서로 매우 가까워질 수 있음을 보이며, 이는 트랜스포머 모델의 정보 처리에 한계를 초래한다고 주장합니다. 또한, 낮은 정밀도의 부동 소수점 방식이 이 문제를 더 악화시킨다고 설명합니다. 그래프 신경망(GNNs)에서 이미 잘 알려진 '오버 스쿼싱(over-squashing)' 현상과 관련이 있습니다.

- **Performance Highlights**: 이 논문은 현대 트랜스포머 기반 LLMs의 문제점을 실제 실험으로 검증하여 이론적 발견이 실제로 적용될 수 있음을 보여줍니다. 또한, 이러한 문제를 완화하기 위한 간단한 해결책도 제안합니다. 저자들은 디코더 전용 트랜스포머의 한계를 이론적으로 분석하고, 실험을 통해 검증하여 이론적 발견이 실제로 적용될 수 있음을 보여줍니다.



### Benchmark Data Contamination of Large Language Models: A Survey (https://arxiv.org/abs/2406.04244)
Comments:
          31 pages, 7 figures, 3 tables

- **What's New**: 이 논문은 대형 언어 모델(LLMs) 평가에서 발생하는 벤치마크 데이터 오염(Benchmark Data Contamination, BDC) 문제를 다룹니다. GPT-4, Claude-3, Gemini와 같은 최신 LLMs의 급속한 발전으로 인해 자연어 처리가 변혁을 겪고 있지만, 이는 또한 BDC 문제를 낳고 있습니다. BDC는 평가 벤치마크 정보가 훈련 데이터에 포함되면서 평가 시 정확성과 신뢰성 문제가 발생하는 현상을 의미합니다. 이 논문은 BDC 문제를 체계적으로 정의하고, 이를 검출 및 완화하는 방법을 탐구합니다.

- **Technical Details**: LLMs는 Transformer와 같은 심층 학습 아키텍처에 기반을 두고 있으며, 텍스트 생성, 번역, 요약, 질문 응답 등 다양한 도메인에서 뛰어난 능력을 보이고 있습니다. 그러나 BDC 문제는 LLM의 평가 방법과 그들의 프라이버시 및 보안 고려사항에 큰 영향을 미칩니다. 기존 평가 방법론은 벤치마크 데이터셋을 기준으로 하지만, 이러한 데이터셋이 훈련 데이터에 포함되어 BDC 문제를 발생시킬 수 있습니다. 이를 해결하기 위해 연구자들은 벤치마크 데이터를 재생성하거나 벤치마크 없는 평가를 포함한 대안 평가 방법을 모색하고 있습니다.

- **Performance Highlights**: BDC 문제를 해결하기 위한 대안 평가 방법으로는 벤치마크 데이터를 재생성하거나 전통적인 벤치마크에 의존하지 않는 방법이 있습니다. 예를 들어, 데이터 큐레이션(Curating New Data), 기존 데이터 재구성(Refactoring Existing Data), 벤치마크 없는 평가(Benchmark-free Evaluation) 등의 접근법을 통해 BDC 문제를 완화할 수 있습니다.



### FairytaleQA Translated: Enabling Educational Question and Answer Generation in Less-Resourced Languages (https://arxiv.org/abs/2406.04233)
Comments:
          Preprint - Accepted for publication at ECTEL 2024

- **What's New**: 이번 논문에서는 FairytaleQA라는 유명한 QA 데이터를 여러 언어로 번역하여 제공했습니다. FairytaleQA는 어린이들의 서사 이해 능력을 평가하기 위해 만들어진 데이터셋으로, 기존에는 영어로만 제공되었습니다. 이번 연구를 통해 스페인어, 포르투갈어(유럽 및 브라질), 프랑스어 번역 버전을 제작하여, 언어 자원이 적은 언어에서도 QA와 QG 연구를 지원할 수 있게 되었습니다.

- **Technical Details**: 번역된 데이터셋을 기반으로, 적당한 규모의 모델을 사용하여 QA와 QG 작업에 대한 기본 벤치마크를 설정했습니다. 이러한 모델들은 비용 효율적인 하드웨어에서 사용할 수 있다는 장점이 있습니다. 또한, 포르투갈어(pt-PT)를 대상으로 질문-답변 쌍을 생성하는 모델을 사용한 사례 연구도 진행되었습니다. 여기서 중요한 평가 기준은 질문의 형성 여부, 답변 가능성, 관련성, 어린이 적합성 등이었습니다. 이러한 평가를 통해 발생하는 오류 사례를 분석하고 향후 연구 방향을 제시했습니다.

- **Performance Highlights**: 모델 평가 결과, 번역된 데이터에서 훈련된 QAPG(Question-Answer Pair Generation) 모델이 비교적 잘 형성된 질문을 생성할 수 있는 것으로 나타났습니다. 그러나 의미적 모호성이 감지되었고, 생성된 질문-답변 쌍의 일치성이 항상 일관되지는 않다는 점도 발견되었습니다. 이를 통해 추가적인 정제가 필요함을 확인했습니다.



### BEADs: Bias Evaluation Across Domains (https://arxiv.org/abs/2406.04220)
Comments:
          under review

- **What's New**: 대형 언어 모델(LLMs)의 최근 발전으로 자연어 처리(NLP) 애플리케이션이 크게 향상되었습니다. 그러나 이러한 모델은 훈련 데이터에서 편향을 상속하고 퍼뜨릴 수 있습니다. 이를 해결하기 위해 다양한 NLP 작업을 평가할 수 있는 데이터셋이 필요하다. 본 논문에서는 이를 위해 다양한 NLP 작업을 지원하는 Bias Evaluations Across Domains (BEADs) 데이터셋을 소개합니다.

- **Technical Details**: BEADs는 텍스트 분류, 편향 엔터티 인식, 편향 정량화 및 무해한 언어 생성 등 다양한 NLP 작업을 지원합니다. AI 기반 주석과 전문가 검증을 결합하여 신뢰할 수 있는 라벨을 제공합니다. 이는 기존의 군중 소싱, 전문가 전용 주석 또는 검증되지 않은 AI 라벨링에 의존하는 데이터셋의 한계를 극복합니다. BEADs는 텍스트 및 토큰 분류, 편향 정량화, 그리고 무해한 언어 생성을 포함하는 여러 작업을 지원하며, 다양한 사회적 편향(인종, 성별, 나이 등)을 다룹니다.

- **Performance Highlights**: 추론 분석에 따르면, BEADs는 다양한 언어 모델에서 편향을 감지하고 줄이는 데 효과적입니다. 작은 모델을 BEADs로 미세 조정하면 편향 분류 작업에서 더 큰 LLMs를 종종 능가합니다. 그러나 이러한 모델은 여전히 특정 인구 통계에 대한 편향을 보일 수 있습니다. 무해한 언어 데이터를 사용한 LLMs의 미세 조정도 편향을 줄이면서 모델의 지식을 유지합니다. 이 연구는 포괄적인 편향 평가의 중요성과 특정 미세 조정이 LLM의 편향을 줄이는 잠재력을 강조합니다.



### Rethinking LLM and Linguistic Steganalysis: An Efficient Detection of Strongly Concealed Stego (https://arxiv.org/abs/2406.04218)
- **What's New**: LSGC라고 불리는 새로운 언어 스테가분석(Linguistic Steganalysis, LS) 방법이 제안되었습니다. 이 방법은 생성 모드와 분류 모드 두 가지로 나뉩니다. 특히 LLM(Large Language Models) 기반 스테가노그래피의 증가된 은닉성에 대처하기 위해 설계되었습니다.

- **Technical Details**: 생성 모드에서는 LLM의 생성 능력을 활용하여 입력 텍스트가 스테고(stego)인지 여부를 설명하도록 합니다. 분류 모드에서는 'description'을 제거하고 'CausalLM' 아키텍처를 'SequenceClassification' 아키텍처로 변경하여 모델의 한 번의 패스만으로 LS 특성을 추출합니다. 또한, LoRA(Low-Rank Adaptation) 전략을 채택하여 대규모 모델의 효율적인 미세 조정을 가능케 합니다.

- **Performance Highlights**: 강력하게 은닉된 스테고에 대한 실험에서 LSGC는 기존의 모든 방법을 능가하여 SOTA(State-Of-The-Art) 성능을 달성했습니다. 또한, 분류 모드의 LSGC는 훈련 시간을 크게 줄이면서도 높은 성능을 유지합니다.



### What Do Language Models Learn in Context? The Structured Task Hypothesis (https://arxiv.org/abs/2406.04216)
Comments:
          This work is published in ACL 2024

- **What's New**: 대형 언어 모델(LLMs)이 컨텍스트 내학습(이하 ICL) 능력을 보이는 이유에 대한 기존 가설들 중 두 가지를 무효화하고, 세 번째 가설을 지지하는 실험 결과를 발표했습니다. 이 논문은 LLMs가 사전 학습 기간 동안 학습한 여러 작업(task)을 조합하여 새로운 작업을 수행할 수 있음을 시사합니다.

- **Technical Details**: 이번 연구에서는 일반적인 텍스트 분류 작업을 토대로 여러 실험을 통해 세 가지 가설을 검토했습니다. 첫 번째 가설은 모델이 컨텍스트 내에 주어진 예시를 바탕으로 작업을 선택(Task Selection)하고, 이를 일반화한다는 것입니다. 두 번째 가설은 ICL이 메타 학습(meta-learning) 형태로, 모델이 사전 학습 시 학습 알고리즘을 배우고 이를 예시에 적용한다는 것입니다. 마지막으로 세 번째 가설은 모델이 예시를 활용하여 사전 학습 기간 동안 배운 여러 작업을 조합(Composition of tasks)하여 ICL을 수행한다고 주장합니다.

- **Performance Highlights**: 첫 번째와 두 번째 가설은 반례(counterexamples)를 통해 무효화되었으며, 모델이 새로운 작업을 학습하는 데 있어 세 번째 가설만이 타당하다는 증거가 제시되었습니다. 따라서 LLMs는 여러 사전 학습된 작업을 조합하여 새로운 작업을 학습할 수 있는 능력이 있음을 확인하였습니다.



### mCSQA: Multilingual Commonsense Reasoning Dataset with Unified Creation Strategy by Language Models and Humans (https://arxiv.org/abs/2406.04215)
Comments:
          Accepted at Findings of ACL 2024

- **What's New**: 이번 연구는 다국어 자연어 이해(NLU) 평가를 위한 새로운 데이터셋인 Multilingual CommonsenseQA (mCSQA)를 제안합니다. 이러한 데이터셋은 단순 번역에 의존하는 기존 다국어 데이터셋의 한계를 극복하고, 특정 언어에 특화된 지식과 상식을 평가할 수 있습니다.

- **Technical Details**: mCSQA는 기존의 CSQA 데이터셋을 기반으로 하여 영어, 일본어, 중국어, 독일어, 포르투갈어, 네덜란드어, 프랑스어, 러시아어 총 8개 언어로 확장되었습니다. 데이터셋 생성에는 생성적 다국어 언어 모델(generative multilingual LMs)을 활용해 질문 생성, 답변 수정, QA 검증 등의 과정을 자동화하여 생성 비용을 대폭 절감했습니다. 최종 검증 작업만 최소한의 인간 노력이 필요했습니다.

- **Performance Highlights**: 실험 결과, 다국어 언어 모델은 쉽게 해결할 수 있는 질문에 대해서는 높은 언어 전이 성능을 보였지만, 깊은 지식 또는 상식을 요구하는 질문에 대해서는 낮은 전이 성능을 보였습니다. 이는 번역되지 않은 언어별 데이터셋이 평가와 훈련에 필요하다는 사실을 강조합니다. 또한, 데이터셋 생성 비용이 기존보다 100분의 1로 줄어들었습니다.



### ValueBench: Towards Comprehensively Evaluating Value Orientations and Understanding of Large Language Models (https://arxiv.org/abs/2406.04214)
Comments:
          Accepted at ACL 2024

- **What's New**: 이 연구는 LLMs(대형 언어 모델)의 가치 성향 및 이해를 평가하기 위한 최초의 종합적인 심리측정 기준인 ValueBench를 소개합니다. ValueBench는 44개의 기존 심리측정 도구에서 수집한 453가지 다양한 가치 차원을 통합하여 데이터를 수집합니다.

- **Technical Details**: ValueBench는 인간-AI 상호작용에 기반한 LLM 가치 성향 평가 파이프라인과 계층적이며 개방된 가치 공간에서 가치 이해를 평가하기 위한 새로운 작업을 제안합니다. 6개의 대표적인 LLM을 대상으로 광범위한 실험을 통해 학습된 내용과 결과를 도출하였습니다.

- **Performance Highlights**: LLMs는 80% 이상의 일관성으로 가치 이론과 일치하는 결과를 도출할 수 있으며, 이는 충분한 맥락과 잘 설계된 프롬프트를 제공했을 때 가능합니다. 또한, LLM들 간의 공통적이면서도 독특한 가치 성향을 식별할 수 있었습니다.



### Legal Documents Drafting with Fine-Tuned Pre-Trained Large Language Mod (https://arxiv.org/abs/2406.04202)
Comments:
          12th International Conference on Software Engineering & Trends (SE 2024), April 27 ~ 28, 2024, Copenhagen, Denmark Volume Editors : David C. Wyld, Dhinaharan Nagamalai (Eds) ISBN : 978-1-923107-24-3

- **What's New**: 대규모 언어 모델 (Large-Scale Language Models, LLM)의 발전과 함께, 미리 학습된 LLM을 미세 조정(Fine-Tuning)하여 자연어 처리의 다운스트림 태스크를 해결하는 것이 주류가 되었습니다. 이 논문은 특히 법률 분야에 필요한 많은 법률 문서가 없어서 어려운 점을 해결하고자 합니다.

- **Technical Details**: 일반적인 NLP 접근 방식은 많은 수의 수작업으로 주석이 달린 데이터셋에 의존하여 학습합니다. 하지만 법률 분야에서는 수작업으로 주석이 달린 대규모 데이터셋을 얻기가 어렵습니다. 이 논문은 주석 없는 많은 법률 문서를 이용하여 미리 학습된 대규모 언어 모델을 로컬 컴퓨터에서 미세 조정해 법률 문서 초안을 생성하는 방법을 제시합니다. 중요한 점은 중국어 단어 분할 없이도 이 작업이 가능하며, 정보 프라이버시 보호 및 정보 보안 문제를 개선할 수 있다는 것입니다.

- **Performance Highlights**: 실험 결과, 법률 분야의 미세 조정 작업에 있어 주석 없는 법률 문서를 사용하여도 성공적으로 법률 문서 초안을 생성할 수 있음을 보여줍니다. 이는 정보 프라이버시와 보안 문제를 동시에 해결할 수 있는 효과를 나타냅니다.



### DICE: Detecting In-distribution Contamination in LLM's Fine-tuning Phase for Math Reasoning (https://arxiv.org/abs/2406.04197)
Comments:
          13 pages, 7 figures

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 성능 과대평가 문제를 해결하기 위해, 데이터 오염 문제를 정밀하게 탐지하는 새로운 메서드 DICE를 제안합니다. 특히, 'In-distribution' 오염에 대해 주목했으며 이는 기존의 'Exact Contamination'과는 다른 접근 방식입니다.

- **Technical Details**: DICE는 LLM의 내부 상태를 활용해 오염된 층을 먼저 식별한 후, 해당 층의 내부 상태를 바탕으로 분류기를 훈련시킵니다. 이를 통해 오염 수준을 정량적으로 평가합니다. 다양한 LLM과 수학적 추론 데이터셋을 대상으로 실험을 수행한 결과, DICE가 높은 정확도로 In-distribution 오염을 탐지하는 것으로 나타났습니다.

- **Performance Highlights**: DICE는 여러 벤치마크에서 오염을 탐지하는 능력의 일반화 가능성을 보여줬습니다. 특히, 10개의 LLM과 4개의 수학적 추론 데이터셋에서 DICE의 예측값이 성능과 높은 상관관계를 보였으며, 이는 많은 기존 모델의 실제 성능이 과대평가될 가능성을 시사합니다.



### Confabulation: The Surprising Value of Large Language Model Hallucinations (https://arxiv.org/abs/2406.04175)
Comments:
          Forthcoming at ACL2024 main conference. 1 figure

- **What's New**: 이 논문은 대형 언어 모델(LLM)의 '환상(hallucination)'이나 '의도적 거짓(confabulation)'이 부정적인 결함이 아니라 잠재적인 자원으로 평가할 수 있다는 점을 체계적으로 방어합니다. 표준적인 관점에서는 환상이 본질적으로 문제로 여겨지지만, 연구진은 실험적으로 LLM 환상의 측정 가능한 의미적 특성이 인간의 이야기력 증대를 인지적 자원으로 이용하는 경향성과 반영하며, 구조적으로 유사하다는 것을 입증했습니다. 이는 LLM 환상이 일관된 서사 텍스트 생성 능력과 밀접하게 관련되어 있음을 시사합니다.

- **Technical Details**: 이 논문은 다양한 인기 있는 환상 벤치마크를 분석하여, 환상 출력이 사실적인 출력에 비해 더 높은 수준의 이야기성(narrativity)과 의미적 일관성을 보여주는 것을 밝혀냈습니다. 이러한 발견은 전통적으로 부정적으로 이해했던 환상이 오히려 긍정적인 서사 텍스트 생성 능력과 긴밀하게 연결되어 있음을 나타냅니다.

- **Performance Highlights**: 연구진은 환상을 '의도적 거짓(confabulation)'으로 재개념화하여 이를 더 유연하게 측정하고 분석하는 방안을 제안했습니다. 실험 결과, 환상이 진실의 출력을 초과하는 이야기성 수준을 나타내며, 이는 인간의 이야기력과 유사한 인지적 및 사회적 효과를 제공할 수 있음을 시사합니다.



### Pointer-Guided Pre-Training: Infusing Large Language Models with Paragraph-Level Contextual Awareness (https://arxiv.org/abs/2406.04156)
Comments:
          17 pages, 3 figures, 5 tables, accepted at ECML-PKDD 2024

- **What's New**: 이 논문에서는 '포인터 가이드 세그먼트 순서화' (Pointer-guided segment ordering, 이하 SO)라는 새로운 프리 트레이닝 기법을 소개합니다. 이 기술은 큰 언어 모델에서 문단 수준의 텍스트 표현을 개선하는 것을 목표로 합니다. 본 기법은 셀프 어텐션 기반 포인터 네트워크를 활용해 섞인 텍스트 세그먼트를 원래 순서대로 재구성함으로써 문서의 구조적 일관성과 문맥 적응성을 파악하는 데 도움을 줍니다.

- **Technical Details**: SO 기법은 텍스트 세그먼트를 섞어놓은 후, 셀프 어텐션 기반 포인터 네트워크를 이용해 원래의 순서로 복원하는 작업을 수행합니다. 이를 통해 서술 흐름, 일관성 및 문맥적 의존성을 모델이 학습하게 합니다. 이 프리 트레이닝 기법은 마스크 언어 모델링(Masked Language Modeling) 같은 기존 기법과 결합해 효과를 높이며, 파인 튜닝 단계에서 다이나믹 샘플링(Dynamic Sampling)을 도입해 데이터 다양성을 증대시키고 샘플 효율성을 개선합니다.

- **Performance Highlights**: 다양한 데이터셋에서 SO 프리 트레이닝 모델을 평가한 결과, 과학 문헌과 금융 보고 분야에서 순차적 텍스트 분류 작업에서 최고의 성능을 보였습니다. SO 기법으로 프리 트레이닝된 모델은 문서의 복잡한 구조를 이해하는 능력이 크게 향상되어, 다운스트림 분류 작업에서 경쟁 모델들을 능가하는 성과를 달성했습니다.



### Towards Understanding Task-agnostic Debiasing Through the Lenses of Intrinsic Bias and Forgetfulness (https://arxiv.org/abs/2406.04146)
- **What's New**: 이 논문은 사전 학습된 언어 모델(Pretrained Language Models, PLMs)의 디바이싱(debiasing)에서 '발견된 사회적 편향(relearning social biases)' 문제를 해결하는 새로운 프레임워크, 'ProSocialTuning'을 제안합니다.

- **Technical Details**: 연구진은 PLMs의 파라미터가 디바이싱과 다운스트림 파인튜닝(downstream fine-tuning) 중에 어떻게 변화하는지 심도 있게 분석하였습니다. ProSocialTuning은 PAC-Bayes 트레이닝을 기반으로 하여, 다운스트림 파인튜닝 시 잘 디바이스된 어텐션 헤드를 규제함으로써 PLMs가 편향을 잊지 않도록 합니다.

- **Performance Highlights**: ProSocialTuning 프레임워크는 대부분의 실제 사례에서 디바이싱 모델의 편향 수준을 다운스트림 파인튜닝 모델의 편향 하한으로 접근하게 할 수 있음을 실험적으로 입증하였습니다. 이는 디바이싱 비효과성을 덜어주는 데 크게 기여할 수 있습니다.



### Every Answer Matters: Evaluating Commonsense with Probabilistic Measures (https://arxiv.org/abs/2406.04145)
Comments:
          ACL 2024 Camera Ready

- **What's New**: 최근 연구들은 기존의 상식 평가 방식이 다수정답의 잠재적 가능성을 다루지 못하고 있다고 지적하며, 이에 대한 새로운 접근 방식을 제안합니다. 특히, '상식 프레임 완성 (Commonsense Frame Completion, CFC)'이라는 새로운 생성형 평가 과제를 소개하여 다중 개방형 답변을 통해 상식을 평가하고자 합니다.

- **Technical Details**: 기존의 상식 평가들은 Multiple Choice Question Answering (MCQA) 방식에 의존해왔으나, 이는 현실적 상황을 반영하는 데 한계가 있습니다. CFC는 주어진 문맥(sentence)에서 누락된 정보를 추론하는 과제로, 각 문맥-질문 쌍에 대해 다수의 다양한 답변을 수집합니다. 이를 위해 확률적 평가 방법을 제안하였으며, 이는 사람들의 판단과 높은 상관관계를 보입니다. 구체적으로, AMR (Abstract Meaning Representation) 파싱을 통해 문맥을 분석한 후, 누락된 슬롯을 식별하고, 이를 기반으로 사람이 주는 여러 답변을 확률 분포로 평가합니다.

- **Performance Highlights**: 여러 LLMs (Large Language Models)의 성능을 제안된 확률적 평가 메트릭으로 측정한 결과, 인간의 성능과 큰 격차를 보였습니다. 이는 현재의 LLM들이 상식 문제에서 여전히 한계가 있음을 시사합니다.



### Do Language Models Understand Morality? Towards a Robust Detection of Moral Conten (https://arxiv.org/abs/2406.04143)
- **What's New**: 이 논문에서는 텍스트 내 도덕적 가치를 탐지하는 새로운 시스템을 소개합니다. GPT-3.5 기반 Davinci 모델을 사용한 Zero-shot 무감독 멀티 라벨 분류기를 소개하며, 데이터 라벨링 없이 도덕적 가치를 탐지합니다. 기존의 작은 NLI 기반 Zero-shot 모델과 비교하여 경쟁력 있는 성능을 입증했습니다.

- **Technical Details**: 새로운 접근 방식은 GPT-3.5 Davinci 모델을 사용하여 도덕적 가치를 Zero-shot 방식으로 탐지합니다. 이는 라벨링된 데이터에 대한 명시적인 학습 없이 이루어집니다. 또한, 작은 NLI 기반 Zero-shot 모델과 비교하여 성능을 평가하여 경쟁력 있는 결과를 확인했습니다. 로버타(RoBERTa) 아키텍처 기반으로 여러 도메인에서 학습된 감독 모델과의 성능 비교를 통해 깊이 있는 분석을 수행했습니다.

- **Performance Highlights**: GPT-3와 비교하여 NLI 접근 방식이 경쟁력 있는 성능을 보였습니다. 또한, MFRC를 포함한 다양한 데이터 도메인에서 훈련된 RoBERTa 기반 감독 모델이 교차 도메인 컨텍스트에서 뛰어난 성능을 발휘함을 입증했습니다.



### Legal Judgment Reimagined: PredEx and the Rise of Intelligent AI Interpretation in Indian Courts (https://arxiv.org/abs/2406.04136)
- **What's New**: 이번 연구는 인도 사법 맥락에서의 법적 판결 예측 및 설명을 위한 가장 큰 전문가 주석 데이터셋인 PredEx를 소개합니다. 이 데이터셋은 15,000개 이상의 주석을 포함하며, 법적 AI 분석의 교육 및 평가를 향상시키는 데 중요한 역할을 합니다. 또한, 대형 언어 모델(LLMs)에 대한 교육 조정(instruction tuning)의 적용을 강조하여 법적 판결의 예측 정확도와 설명 깊이를 향상시켰습니다.

- **Technical Details**: PredEx 데이터셋은 인도 대법원과 여러 고등법원에서 약 20,000개의 법원 판결을 수집하고 주석처리한 후, 약 16,000개의 케이스 파일을 남겼습니다. 이 데이터셋에는 판결에 중요한 역할을 한 핵심 문장과 예측된 결과에 대한 이유가 포함됩니다. 연구진은 일반 및 인도 법률 맥락에 맞춘 transformer 기반 모델을 사용했으며, LLMs의 교육 조정을 통해 높은 예측 정확도를 달성했습니다.

- **Performance Highlights**: 고급 LLMs와 PredEx 데이터셋을 사용하여 개발된 AI 모델은 법적 판결 예측에서 뛰어난 성과를 보였습니다. 특히, 예측 정확도와 관련성에서 전례 없는 수준의 성과를 보여줬습니다. 또한 전문가 평가를 통해 인간 전문가 기준과 비교한 모델의 성능을 검증했습니다.



### Are We Done with MMLU? (https://arxiv.org/abs/2406.04127)
- **What's New**: 최근 발표된 연구에서는 널리 사용되는 Massive Multitask Language Understanding (MMLU) 벤치마크 데이터셋의 오류를 식별하고 분석했습니다. 연구팀은 Virology 부분에서 분석된 질문 중 57%에 오류가 있음을 발견했으며, 이를 해결하기 위해 새로운 오류 분류 체계를 도입하고 MMLU-Redux라는 3,000개의 질문을 다시 주석한 하위 집합을 소개했습니다. 이 새로운 데이터셋은 모델 성능 지표의 중요한 차이를 나타내며, 향후 MMLU 데이터셋의 신뢰성과 유용성을 높이기 위해 데이터셋의 수정을 강력히 권장합니다.

- **Technical Details**: MMLU 데이터셋은 다양한 주제에 걸쳐 LLM(Large Language Models)의 언어 이해 능력을 평가하기 위해 설계되었습니다. 그러나, MMLU는 많은 오류를 포함하고 있다는 것이 밝혀졌습니다. 연구팀은 새로운 오류 분류 체계를 사용해 MMLU의 30개의 하위 집합에서 임의로 선택한 3,000개의 질문을 수작업으로 다시 주석했습니다. 오류는 주로 문항의 명확성(예: 질문이나 선택지의 불명확함)과 정답 검증(예: 정답의 누락 또는 다중 정답 가능성)으로 분류되었습니다. MMLU-Redux를 통해 재평가된 LLM 성능은 기존에 보고된 지표와 크게 달랐으며, 이는 모델 평가에 중요한 영향을 미칠 수 있음을 보여줍니다.

- **Performance Highlights**: MMLU-Redux를 활용한 실험에서는 기존 MMLU 데이터셋에서의 평가 지표와 비교하여 모델 성능 순위가 달라지는 것을 확인했습니다. 이는 원본 MMLU 데이터셋에서 발견된 다양한 오류들이 모델 성능 평가의 공정성과 정확성에 큰 영향을 미친다는 것을 시사합니다.



### Uncovering Limitations of Large Language Models in Information Seeking from Tables (https://arxiv.org/abs/2406.04113)
Comments:
          Findings of ACL 2024

- **What's New**: 최근 논문에서 테이블 정보 탐색(Table Information Seeking, TIS)에 대한 새로운 벤치마크인 TabIS를 소개했습니다. 기존 텍스트 유사성 기반 메트릭을 사용한 평가의 신뢰성 문제를 해결하기 위해, 단일 선택 질문 형식의 평가 방식을 채택했습니다. 이를 통해 대규모 언어 모델(LLMs)의 테이블 이해 능력을 보다 정확하게 평가할 수 있습니다.

- **Technical Details**: TabIS 벤치마크는 기존의 table-to-text generation (TTG) 데이터셋을 기반으로 구축되었습니다. TTG 데이터셋을 단일 선택 질문 형식으로 변환하여 신뢰할 수 있는 평가가 가능합니다. 선택지 생성 과정에서는 세 가지 프롬프트 기반 방법(Modify-Input, Modify-Output, Exam-Judge)을 사용해 고품질의 오답을 생성했습니다. TabIS는 세 가지 시나리오(B-TIS, SU-TIS, M-TIS)를 포함하여 다양한 난이도의 테이블 정보 탐색 과제를 제공합니다.

- **Performance Highlights**: 12개의 대표적인 LLM에 대한 실험 결과, GPT-4-turbo는 평균 85.7%의 정확도를 기록하며 가장 높은 성능을 보였습니다. 그에 반해, 다른 상용 및 오픈소스 모델들은 대부분 50-60% 범위에 머물렀습니다. 이는 테이블 이해 및 정보 탐색에 대한 LLM의 한계를 드러냅니다. 특히, LLM은 테이블 구조 이해와 pseudo-relevant tables을 다루는 데 취약합니다. 추가적으로, Llama2-13b-chat 모델을 미약하게 지도학습(fine-tuning)한 결과, 성능이 크게 개선되었으나 여전히 GPT-4-turbo에 미치지 못했습니다.



### Intention and Face in Dialog (https://arxiv.org/abs/2406.04109)
- **What's New**: 이 논문은 대화에서 '면' (face) 개념을 중심으로 '의도' (intention)와 '예의' (politeness) 분류를 수행하는 세 가지 컴퓨팅 시스템을 분석합니다. 논문에서는 특히 의도가 예의에 어떻게 영향을 미치는지에 주목하며, 기존 모델을 이용해 새롭게 state-of-the-art (SoTA) 성과를 달성했다고 주장합니다.

- **Technical Details**: 기존의 face act (면 행동) 분류기를 훈련시키기 위해 Dutt et al. (2020) 데이터셋을 사용하였으며, 추가적으로 대화 행위 (dialog act) 주석을 통합했습니다. 대화 행위를 이용하여 분류 성능을 향상시켰으며, 모델의 F-측정치를 69%에서 73%로 증가시켰습니다. BERT 기반의 네트워크 아키텍처를 사용하고, 면 행동과 설득의 관계도 조사했습니다.

- **Performance Highlights**: 특정 면 행동에서 대화 행위를 활용한 분류 모델이 성능을 크게 향상시켰습니다. 특히, 소수 클래스의 면 행동 탐지에서 좋은 성과를 보였으며, 전체 면 행동 분류기의 F-측정치가 73%로 향상되었습니다.



### Explainability and Hate Speech: Structured Explanations Make Social Media Moderators Faster (https://arxiv.org/abs/2406.04106)
Comments:
          11 pages, 14 figures, to be published at ACL 2024

- **What's New**: 소셜 미디어에서의 혐오 발언 탐지 연구는 많았지만, 실제 콘텐츠 관리자(Content Moderators)를 대상으로 한 연구는 거의 없었습니다. 이번 연구는 실제 관리자들에게 설명이 제공될 때 의사결정 속도가 어떻게 변화하는지를 조사했습니다.

- **Technical Details**: 실험에서는 세 가지 조건을 설정하여 관리자들이 게시물을 평가하도록 했습니다: 1) 게시물만 표시, 2) 게시물 및 정책 규칙 표시(일반적 설명), 3) 게시물 및 태그가 포함된 구체적 설명(구조적 설명). 연구에서는 PLEAD 데이터셋을 사용하여 사용자 의도와 설명이 포함된 3,535개의 혐오적 게시물을 분석했습니다.

- **Performance Highlights**: 실험 결과, 구체적 설명(구조적 설명)이 포함된 경우 관리자들이 게시물당 평균 1.34초 더 빨리 결정을 내렸으며, 이는 평균적으로 18.14초에서 16.8초로, 약 7.4%의 시간 단축을 의미합니다. 또한, 관리자의 84%는 구조적 설명을 선호하는 것으로 나타났습니다.



### Ask LLMs Directly, "What shapes your bias?": Measuring Social Bias in Large Language Models (https://arxiv.org/abs/2406.04064)
Comments:
          Findings of ACL 2024

- **What's New**: 최근 연구에서는 대규모 언어 모델(LLMs)에서 사회적인 편향을 더욱 정밀하게 이해하기 위해 다양한 관점에서의 사회 인식을 종합적으로 평가하는 새로운 접근 방식을 제안했습니다. 이 논문은 사회적 인식이 LLMs에서 발생하는 편향을 어떻게 형성하는지를 조사하고, 다양한 사회적 인식을 기반으로 편향을 평가할 수 있는 새로운 지표를 제안합니다.

- **Technical Details**: 이 연구에서는 질문-응답(QA) 형식을 채택하여 LLMs의 사회적 인식을 직접적으로 측정하고, 다른 인물에게 할당된 페르소나(persona)에 따라 타겟에 대한 관점을 평가하는 방법론을 도입했습니다. 주요 지표로는 Target Bias (목표 편향), Bias Amount (편향 양), 그리고 Persona Bias (페르소나 편향)를 제안하였습니다. 이를 통해 LLMs 내부의 편향을 세밀하게 분석할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 지표들은 LLMs의 사회적 인식 변화를 정량적으로 나타내며, 다양한 사회적 인식에 따라 발생하는 편향을 반영하는 것을 보여주었습니다. 이로써 LLMs 내부의 편향을 다차원적으로 파악할 수 있으며, 편향을 더욱 포괄적이고 세밀하게 조사할 수 있게 되었습니다.



### The syntax-semantics interface in a child's path: A study of 3- to 11-year-olds' elicited production of Mandarin recursive relative clauses (https://arxiv.org/abs/2406.04025)
- **What's New**: 이번 논문에서는 어린이의 반복 구조를 포함하는 관계절(recursive relative clauses, RRCs)을 습득하는 과정에서의 문법-의미 관계에 대한 연구를 제시합니다. 특히 3세에서 11세 사이의 어린이가 8가지 형태의 중국어 RRCs를 어떻게 가장 잘 만들어내는지 실험을 통해 조사했습니다.

- **Technical Details**: 연구는 4가지 문법유형(소주어-gap 관계절이 목적어-gap 관계절에 포함되는 유형(SORRCs), 목적어-gap 관계절이 또 다른 목적어-gap 관계절에 포함되는 유형(OORRCs), 목적어-gap 관계절이 소주어-gap 관계절에 포함되는 유형(OSRRCs), 소주어-gap 관계절이 또 다른 소주어-gap 관계절에 포함되는 유형(SSRRCs))과 2가지 내부 의미 조건(불가역적 내부 의미: irreversible internal semantics, IIS; 가역적 내부 의미: reversible internal semantics, RIS)을 포함한 4x2 디자인을 사용했습니다.

- **Performance Highlights**: 실험 결과, SSRRCs, OSRRCs, SORRCs의 IIS-IES(불가역적 외부 의미) 조건이 가역적 내부 의미 조건보다 두 해 전에 생성된다는 것을 발견했습니다. 이에 따라 두 단계의 발달 경로가 제안되었습니다: 먼저 불가역적인 문법과 IIS의 상호작용으로 시작하여, 이후 문법과 IES의 상호작용으로 끝맺음하는 경로가 있다는 것입니다.



### American Sign Language Handshapes Reflect Pressures for Communicative Efficiency (https://arxiv.org/abs/2406.04024)
Comments:
          Accepted to ACL 2024

- **What's New**: 이번 연구에서는 미 언어인 미국 수어(ASL)의 수형(handshape)이 효율적인 의사소통 압력을 반영하는지를 살폈으며, 시각-제스처 모달리티의 새로운 증거를 제시합니다. 특히, 영어에서 차용된 수형과 본래 ASL의 수형을 비교하여 이 효율성 압력이 어떻게 다르게 적용되는지를 분석했습니다.

- **Technical Details**: 연구에서 우리는 손 모양을 만들 때 필요한 발성 노력(articulatory effort)과 그것을 인식하는 데 필요한 지각 노력(perceptual effort)을 정량화하는 새로운 방법론을 개발했습니다. ASL과 영어에서 사용 빈도와 의사소통 노력 간의 상관관계를 비교했습니다. 이는 양손 또는 단일 손으로 표현된 모양을 통해 실험이 진행되었습니다.

- **Performance Highlights**: 연구 결과, 빈번히 사용되는 ASL 수형은 만들기 쉽다는 것이 밝혀졌습니다. 또한, 효율적인 의사소통 압력은 주로 ASL 사용에서 비롯된 것이며 영어 차용에는 그 압력이 크게 영향을 미치지 않는 것으로 나타났습니다. 영어의 문자 빈도와 손가락 철자법(ASL fingerspelling)의 발성 용이성 사이에는 유의미한 상관관계가 없음을 발견하여 이러한 결론을 뒷받침했습니다.



### Assessing LLMs for Zero-shot Abstractive Summarization Through the Lens of Relevance Paraphrasing (https://arxiv.org/abs/2406.03993)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 추상적 요약 생성을 위한 제로샷 성능에 대한 강건성을 평가하기 위해 새로운 방법인 'relevance paraphrasing'(관련성 패러프레이징)을 제안합니다. 이는 모델이 이상적인 요약을 생성하기 위해 가장 중요한 문장을 식별하고, 이 문장을 최소한으로 변형된 데이터셋으로 패러프레이징하는 방식입니다.

- **Technical Details**: 연구진은 네 개의 다양한 데이터셋과 네 개의 다른 크기의 LLMs(GPT-3.5-Turbo, Llama-2-13B, Mistral-7B, Dolly-v2-7B)을 사용하여 광범위한 실험을 수행했습니다. 패러프레이징된 문장은 원래 의미를 유지하면서도 다른 방식으로 표현되도록 재구성되어, 원본과 비교하여 모델의 요약 성능 변화를 평가합니다. 이러한 평가 기준으로는 ROUGE 및 BertScore 등이 사용되었습니다.

- **Performance Highlights**: 실험 결과, LLMs는 최소한으로 변형된 기사에 대해 일관성이 없는 요약 성능을 보였습니다. 이는 LLMs가 관련성 패러프레이징된 문장을 처리할 때 중요한 정보를 잃어버리는 경향이 있음을 나타냅니다. 다양한 크기의 모델과 여러 데이터셋에서 이러한 경향이 일관되게 나타났으며, 이는 이러한 모델들이 일관적인 요약 성능을 보장하기 위해 추가적인 개선이 필요함을 시사합니다.



### On The Persona-based Summarization of Domain-Specific Documents (https://arxiv.org/abs/2406.03986)
- **What's New**: 이 논문은 헬스케어 분야에서 각기 다른 '페르소나(예: 의사, 환자, 일반인)'에 맞춘 도메인 특정 요약을 효율적으로 생성하는 방법을 제안합니다. 전통적으로 사람이 요약을 만드는 것은 비용이 크고 일관성 없는 결과를 초래합니다. 기존의 대형 언어 모델(LLM)을 사용하는 AI 기반 요약은 정확도가 떨어질 수 있으며, 이를 일상적 운영에 사용하는 것은 비용이 큽니다. 이 논문의 기여는 두 가지로 요약됩니다: 1) 헬스케어 데이터셋을 사용하여 소형 기반 LLM을 효율적으로 미세 조정하고, AI 기반 평가를 통해 요약의 질을 평가하는 방법을 제시합니다. 2) AI 기반 평가가 인간 기반 평가와 높은 일치도를 가지고 있다는 것을 입증합니다.

- **Technical Details**: 논문에서는 요약 생성 및 평가를 위해 세 단계의 프롬프트를 사용합니다. 프롬프트는 시스템 프롬프트와 사용자 프롬프트로 나뉘며, 시스템 프롬프트는 특정 페르소나에 맞춘 의학 문서 요약을 생성하도록 되어 있습니다. 사용자는 이를 바탕으로 의사, 환자, 일반인 관점에서 문서를 요약하도록 지시됩니다. 평가 시스템은 생성된 요약을 페르소나별로 평가하며, 품질을 0에서 1까지 점수로 나타냅니다.

- **Performance Highlights**: Llama2-13b 모델을 사용하여 다양한 크기의 WebMD 데이터로 학습을 진행했습니다. 데이터 크기에 따른 모델 성능의 변화를 확인한 결과, 데이터 양이 증가할수록 성능이 향상됨을 알 수 있었으며, 데이터의 40%만 사용해도 매우 좋은 성능을 보여주었습니다. 또한, 최소한의 데이터로도 적절한 페르소나와 측면 기반 요약을 생성할 수 있음을 확인했습니다. 추론 단계에서는 'max-new-token'과 'temperature' 파라미터를 조정하여 성능에 미치는 영향을 조사했습니다.



### A + B: A General Generator-Reader Framework for Optimizing LLMs to Unleash Synergy Potentia (https://arxiv.org/abs/2406.03963)
Comments:
          Accepted to ACL'24 (Findings)

- **What's New**: 기존 검색 기반 생성 방식(RAG)의 성능 병목 현상을 해결하기 위해 '생성-후-읽기(generate-then-read)' 프레임워크를 도입했습니다. 이 새로운 방식은 기본 모델(base)과 대화형 모델(chat)을 결합하여 검색 단계를 모델 자체의 생성으로 대체합니다. 이를 통해 다양한 시나리오에서 성능을 개선할 수 있습니다.

- **Technical Details**: 'A + B' 프레임워크는 생성기(Generator)와 리더(Reader) 역할을 부여한 두 개의 서로 다른 모델 결합을 포함합니다. 생성기는 입력 문맥과 관련된 내용을 생성하여 높은 수준의 사실 정확성을 필요로 하고, 리더는 이 생성된 문맥을 해석하여 인지적 추론과 인간의 선호에 맞는 적절한 응답을 제공합니다. 추후 학습을 통해 외부 지식을 통합하여 안전성과 유용성을 유지하는 것도 연구의 중요한 부분입니다.

- **Performance Highlights**: 'A + B' 프레임워크의 다양한 모델 조합들이 단일 모델에 비해 특히 복잡한 시나리오에서 탁월한 성능을 보입니다. 기본(base) 모델은 기억 능력이 뛰어나 사실에 맞는 내용을 잘 생성하고, 대화형(chat) 모델은 더 유용하고 안전한 응답을 제공하는 데 적합합니다. 실험 결과 외부 문서가 주어진 상황에서도 이 프레임워크가 일관되게 우수한 성능을 발휘합니다.



### Tox-BART: Leveraging Toxicity Attributes for Explanation Generation of Implicit Hate Speech (https://arxiv.org/abs/2406.03953)
Comments:
          17 Pages, 5 Figures, 13 Tables, ACL Findings 2024

- **What's New**: 이 연구는 언어 모델(LMs)을 사용하여 암묵적 혐오 게시물에 대한 설명을 생성하는 작업에 관한 새로운 주요 발견을 발표했습니다. 연구는 특히 지식 그래프(KG) 튜플이 암묵적 혐오 설명 생성을 위해 항상 가장 좋은 선택이 아닐 수 있다는 결론을 도출했습니다. 대신, 외부 독성 신호를 통합한 단순한 모델이 더 나은 성능을 보여줍니다.

- **Technical Details**: 연구는 주로 두 개의 공개된 암묵적 혐오 데이터셋(SBIC와 LatentHatred)과 두 개의 주요 지식 그래프(ConceptNet과 StereoKG)를 사용하여 실험을 진행했습니다. 또한, 'toxicity attributes'(독성 속성)을 사용하여 Tox-BART 모델을 구축했고, 이를 통해 암묵적 혐오 설명을 생성했습니다. 이 독성 속성은 Jigsaw 데이터셋에서 finetune된 BERT 리그레서로부터 얻어졌습니다.

- **Performance Highlights**: SBIC와 LatentHatred 데이터셋에 대해 BLEU, ROUGE-L, BERTScore 지표에서 KG-기반 설정과 유사한 성능 변화를 보였습니다. 구체적으로 BLEU에서는 +0.44, ROUGE-L에서는 +1.83, BERTScore에서는 -4.59의 성능 변화를 관찰했습니다. 또한 인간 평가와 오류 분석을 통해 Tox-BART가 zero-shot GPT-3.5보다 더 정확한 설명을 생성함을 확인했습니다.



### UltraMedical: Building Specialized Generalists in Biomedicin (https://arxiv.org/abs/2406.03949)
Comments:
          Datasets and models are available at this https URL

- **What's New**: 이 논문에서는 고품질 수동 및 합성 데이터세트로 구성된 UltraMedical 데이터를 소개합니다. 이 데이터세트는 최신 LLM팀과의 선호도 주석이 포함된 의학 영역의 데이터이며, 이를 통해 Llama-3 시리즈를 기반으로 한 전문 의학 모델을 미세조정했습니다. 또한 의학 및 일반 보상 벤치마크에서 뛰어난 성능을 발휘하는 강력한 보상 모델을 개발했습니다.

- **Technical Details**: UltraMedical 데이터세트는 약 410,000개의 고품질 수동 및 합성 의료 명령으로 구성되어 있으며, GPT-4와 기타 LLM의 선호도 주석이 포함된 약 110,000개의 명령이 포함되어 있습니다. 이 데이터는 PubMed 문헌 연구 및 개방형 질문 등 다양한 질문 유형을 포함하며, 명령의 복잡성과 다양성을 유지하기 위해 사전 및 사후 방식을 혼합하여 구성되었습니다.

- **Performance Highlights**: UltraMedical 컬렉션을 사용하여 미세 조정된 Llama-3 시리즈는 다양한 오픈 소스 의료 벤치마크에서 경쟁력 있는 결과를 달성했습니다. 특히, 8B 모델은 MedPaLM 1, Gemini-1.0, GPT-3.5 및 Meditron-70B와 같은 이전의 더 큰 모델을 능가했으며, 70B 모델은 MedQA-USMLE에서 86.5점을 기록해 오픈 소스 LLM 중 최고 기록을 달성했습니다. 이러한 성과는 UltraMedical 데이터세트를 사용함으로써 오픈 소스와 독점 모델 간의 격차를 줄일 수 있음을 시사합니다.



### Culturally Aware and Adapted NLP: A Taxonomy and a Survey of the State of the Ar (https://arxiv.org/abs/2406.03930)
- **What's New**: 최근 자연어 처리(Natural Language Processing, NLP)의 중요한 연구 주제로 문화(culture)가 부상하고 있으며, 문화적으로 인식하고 적응하는 NLP 연구는 급격한 증가세를 보이고 있습니다. 그러나 '문화' 개념에 대한 공통된 이해 부족이 이 분야의 진전을 평가하는 데 어려움을 초래하고 있습니다. 이를 해결하기 위해, 기존 연구에 기반을 두어 문화 요소의 광범위한 분류 체계를 제안하고, 이를 통해 현재의 연구 상태와 향후 연구의 틈새를 파악할 수 있는 체계를 제공합니다.

- **Technical Details**: 제안된 분류 체계는 인류학에서 오랜 기간 연구된 문화 개념에 기반하고 있습니다. 주요 문화 요소로는 개념(concepts), 지식(knowledge), 가치(values), 규범과 도덕(norms and morals), 언어 형식(linguistic form), 유물(artifacts)이 있으며, 사회문화적 요소로는 관계(relationship), 문맥(context), 의사소통 목표(communicative goals), 인구통계(demographics) 등이 포함됩니다. 이 분류 체계를 사용해 기존 연구를 조직하고 분석하였습니다.

- **Performance Highlights**: 기존 NLP 연구는 데이터 자체나 다문화적 주석 등을 통해 문화를 포착하고 있습니다. 최근 연구는 문화에 특화된 개념이나 다중언어 대화 데이터셋 등에서 문화적 적응을 위한 데이터셋을 개발하고 있는데, 이러한 데이터셋은 평가용으로만 활용되는 경우가 많고, 다양한 문화 및 언어 범주에 대한 데이터가 여전히 부족한 상태입니다. 또한, 개념 간의 다문화적 차이를 탐구하는 연구도 존재하며, 이러한 연구는 비유적 표현이나 전통적 격언 등을 통해 이루어지고 있습니다.



### ArMeme: Propagandistic Content in Arabic Memes (https://arxiv.org/abs/2406.03916)
Comments:
          disinformation, misinformation, factuality, harmfulness, fake news, propaganda, multimodality, text, images

- **What's New**: 새로운 연구는 아랍어 밈(Emoji) 데이터셋을 개발하고 주로 영어에 초점을 맞춘 기존 연구와 달리 아랍어 멀티모달(multimodal) 콘텐츠 분석에 중요한 자료를 제공합니다. 연구진은 소셜 미디어 플랫폼에서 수집한 약 6,000개의 아랍어 밈을 선전(propaganda)적 내용으로 수동 주석(annotations)하여 처음으로 아랍어 멀티모달 연구 자원을 구축했습니다.

- **Technical Details**: 연구진은 아랍어 밈을 네 가지 범주로 구분하여 주석 작업을 수행했습니다. 데이터 수집 절차는 다양한 소셜 미디어 플랫폼(Facebook, Instagram, Twitter)에서 공공 그룹을 선택하여 구현되었으며, 이를 통해 정치, 유명인, 공공 인물에 관한 밈을 수집했습니다. 텍스트 및 이미지 모듈을 기반으로 한 AI 모델 훈련, 즉 멀티모달 접근법을 통해 선전적인 콘텐츠를 식별하는 도구를 개발했습니다.

- **Performance Highlights**: 텍스트 모달리티(modality)에서는 전통적인 모델과 다언어 트랜스포머(multilingual transformer) 모델을 정밀 조정(fine-tuning)했습니다. 이미지 모달리티에서는 다양한 구조의 CNN(Convolutional Neural Network) 모델을 정밀 조정했으며, 멀티모달 모델의 경우 조기 융합 기반 모델을 훈련시켰습니다. 모든 모달리티에 대해 LLMs(large language models)을 사용한 제로샷(zero-shot) 설정에서 성능을 평가했습니다. 이 데이터셋은 커뮤니티에 공개될 예정입니다.



### HeSum: a Novel Dataset for Abstractive Text Summarization in Hebrew (https://arxiv.org/abs/2406.03897)
- **What's New**: 최근 대형 언어 모델(LLMs)이 영어를 비롯한 다양한 자연어 처리(NLP) 작업에서 뛰어난 성능을 보이고 있지만, 히브리어와 같이 리소스가 부족한 언어에서는 그 성능에 대한 이해가 제한적입니다. 이에 대한 문제를 해결하고자 우리는 현대 히브리어의 추상적 텍스트 요약을 위한 참신한 벤치마크인 HeSum을 소개합니다. HeSum은 히브리어 뉴스 웹사이트에서 전문 저널리스트가 작성한 10,000개의 기사-요약 페어를 포함하고 있습니다.

- **Technical Details**: HeSum 데이터셋은 'Shakuf', 'HaMakom', 'The Seventh Eye'와 같은 히브리어 뉴스 웹사이트에서 수집된 기사와 요약 쌍으로 구성되어 있습니다. 이 데이터셋은 높은 추상성과 히브리어 특유의 언어적 복잡성을 가지고 있으며, 이러한 복잡성은 현대의 최첨단 대형 언어 모델(LLMs)에게도 독특한 도전을 안겨줍니다. 추가적으로, 우리는 DictaBert와 AlephBert와 같은 히브리어 모노링궐 BERT 기반 인코더 모델을 사용하여 어휘 크기, 형태소적 애나포라 표현, 그리고 구문과 의미적 특징을 분석했습니다.

- **Performance Highlights**: HeSum은 높은 추상성, 형태소적 복잡성, 그리고 철저한 LLM 평가를 결합하여 MRL(형태소가 풍부한 언어) 설정에서 요약 모델의 경계를 확장하는데 중요한 자원이 됩니다. 다양한 LLM을 사용하여 포괄적인 실험 분석을 수행한 결과, HeSum은 현대의 모델들에게도 독특한 도전을 제공하는 것으로 나타났습니다.



### How Good is Zero-Shot MT Evaluation for Low Resource Indian Languages? (https://arxiv.org/abs/2406.03893)
- **What's New**: 이 논문에서는 초자원(low-resource) 인도 언어들인 아삼어, 칸나다어, 마이틸리어, 펀자브어에 대한 zero-shot 평가를 다루고 있습니다. 이는 최근에 데이터와 모델의 가용성이 증가함에 따라 초자원 언어에 대한 평가에 대한 관심이 증가했기 때문입니다.

- **Technical Details**: 해당 연구에서는 Multi-Dimensional Quality Metrics (MQM)와 Direct Assessment (DA) 주석을 수집하여 테스트 세트를 만들고 다양한 자동화 평가 메트릭스를 메타평가했습니다. 4가지 언어에 대해 각 언어당 250개의 주석을 수집하여 총 1000개의 MQM 주석을 수집했습니다. 이를 바탕으로 여러 자동화 및 학습된 메트릭스를 평가했습니다. 학습된 메트릭스의 경우, 관련 있는 인도 언어 데이터를 사용하여 fine-tuning과 zero-shot 메타평가를 수행했습니다.

- **Performance Highlights**: 해당 연구의 메인 결과는 학습된 메트릭스들도 초자원 언어에서는 여전히 큰 개선이 필요하다는 점입니다. Kendall Tau와 Pearson 상관계수는 각각 최대 0.32와 0.45에 불과했으며, synthetic data 접근 방식은 혼합된 결과를 보였고 초자원 언어 간 격차를 크게 줄이지 못했습니다.



### Spontaneous Speech-Based Suicide Risk Detection Using Whisper and Large Language Models (https://arxiv.org/abs/2406.03882)
Comments:
          Accepted by Interspeech 2024

- **What's New**: 이 논문은 청소년들의 자발적인 스피치를 바탕으로 자살 위험을 자동으로 감지하는 방법을 연구했습니다. 이를 위해 만다린어로 1000명 이상의 청소년들로부터 15시간의 자살 관련 스피치 데이터를 수집했습니다. Whisper 스피치 모델과 대형 언어 모델(LLMs)을 사용해 자살 위험을 감지하는 방법을 제안했습니다.

- **Technical Details**:  자발적인 스피치에 내재된 다양한 음향적 및 언어적 특징을 활용하기 위해 Whisper 스피치 모델과 LLM을 사용했습니다. 두 모델에 대해 전 파라미터 미세조정(all-parameter finetuning, APFT) 및 파라미터 효율적 미세조정(parameter-efficient finetuning, PEFT) 접근법을 적용했습니다. 또한 Whisper 및 LLM의 표현을 결합하기 위해 여러 오디오-텍스트 융합 기술을 평가했습니다.

- **Performance Highlights**: 제안된 시스템은 테스트 세트에서 0.807의 검출 정확도와 0.846의 F1 점수를 달성했습니다. 이는 실제 자살 위험 감지 응용 프로그램에 대한 유망한 가능성을 나타냅니다.



### Evaluating the IWSLT2023 Speech Translation Tasks: Human Annotations, Automatic Metrics, and Segmentation (https://arxiv.org/abs/2406.03881)
Comments:
          LREC-COLING2024 publication (with corrections for Table 3)

- **What's New**: 이번 연구는 IWSLT 2023에서 진행된 여러 공유 과제의 결과를 종합적으로 인간 평가를 통해 분석하는 최초의 시도입니다. 이를 통해 음성 번역(Speech Translation, ST)에 대한 인간 평가의 빈틈을 채우고, 강력한 평가 전략을 제안하였습니다.

- **Technical Details**: 자동 재세그먼트(automatic resegmentation)와 세그먼트 문맥과 함께 직접 평가(direct assessment)를 기반으로 한 평가 전략을 제안했습니다. 데이터는 IWSLT 2023의 오프라인, 다국어, 동시 통역 조건에서 수집되었습니다. 평가 과제에는 다양한 발음, 전문 용어, 자발적 발화 등이 포함되었습니다.

- **Performance Highlights**: 1) 직접 평가 점수가 다른 유형의 인간 판단과 잘 일치한다는 것을 확인했습니다. 2) 자동 메트릭(automatic metrics)은 일반적으로 직접 평가 점수와 잘 상관이 있지만 항상 그렇지는 않았습니다. 3) COMET 메트릭이 chrF보다 강력한 것으로 나타났습니다. 또한, 수집된 인간 평가 데이터를 공개하여 추가 연구를 장려하고 있습니다.



### Decoder-only Streaming Transformer for Simultaneous Translation (https://arxiv.org/abs/2406.03878)
Comments:
          Accepted to ACL 2024. 14 pages, 10 Tables, 5 Figures

- **What's New**: 이번 연구에서는 새로운 동시 기계 번역(Simultaneous Machine Translation, SiMT) 모델인 'Decoder-only Streaming Transformer (DST)'를 제안합니다. 이 모델은 기존의 Encoder-Decoder 구조가 아닌, 오직 Decoder만을 사용하는 아키텍처로 이루어져 있습니다.

- **Technical Details**: DST 모델은 첫 번째로 소스(prefix)와 타겟(prefix)의 위치를 각각 인코딩하여 소스 prefix의 확장이 타겟 prefix의 위치에 영향을 주지 않도록 설계되었습니다. 두 번째로, 전통적인 마스크드 셀프 어텐션(masked self-attention) 대신, 스트리밍 셀프 어텐션(Streaming Self-Attention, SSA)을 도입하여 고정된 위치의 소스 토큰으로부터 정보를 평가하고, 이를 바탕으로 번역 정책을 결정합니다. SSA는 소프트 어텐션(soft-attention) 메커니즘과 결합하여 기대되는 어텐션을 계산하고 이를 통해 컨텍스트 벡터(context vector)를 생성합니다.

- **Performance Highlights**: 실험 결과, DST는 세 가지 번역 작업에서 최첨단 성능(state-of-the-art performance)을 달성하였습니다. DST는 소스 토큰의 도착과 함께 타겟 prefix의 위치 증가로 인해 발생하는 재인코딩 문제를 해결하여, 추론 비용을 줄였습니다. 또한, 훈련 동안 모든 가능한 소스 prefix에 대해 타겟 prefix를 고려할 수 있도록 하여 훈련 비용 역시 절감시켰습니다.



### BLSP-Emo: Towards Empathetic Large Speech-Language Models (https://arxiv.org/abs/2406.03872)
- **What's New**: BLSP-Emo는 기존의 음성 인식(ASR) 및 음성 감정 인식(SER) 데이터를 활용하여 의미와 감정을 이해하고 공감적인 응답을 생성할 수 있는 새로운 종단 간(end-to-end) 음성-언어 모델입니다. 이 접근 방식은 감정을 고려한 전사(transcript)를 생성하고, 이를 통해 감정 단서를 이해하고 반응하도록 모델을 조정합니다.

- **Technical Details**: BLSP-Emo는 BLSP(bootstrapped speech-language pretraining) 방법을 기반으로 하며, 음성 인코더(speech encoder), 명령을 따르는 LLM(instruction-following LLM), 그리고 두 모듈 사이의 모달리티 어댑터(modality adapter)로 구성되어 있습니다. 모델은 의미적 정렬(semantic alignment)을 위해 ASR 데이터를 사용하고, 감정적 정렬(emotion alignment)을 위해 SER 데이터를 사용합니다.

- **Performance Highlights**: BLSP-Emo 모델은 음성을 이해하고 공감적인 응답을 생성하는 데 뛰어납니다. 본 논문에서는 이 모델이 독립적인 음성 감정 인식(standalone speech emotion recognition), 공감적인 응답 생성, 그리고 공감적인 대화를 수행할 수 있는 경쟁력 있는 능력을 갖추고 있음을 보여줍니다.



### Recovering document annotations for sentence-level bitex (https://arxiv.org/abs/2406.03869)
Comments:
          ACL 2024 Findings

- **What's New**: 최근 기계 번역(machine translation) 분야에서 문서 수준의 정보(document-level information)가 부족하며, 특히 긴 문장을 다루는 방법이 효과적이지 않다는 문제가 제기되었습니다. 이 연구에서는 독일어, 프랑스어, 스페인어, 이탈리아어, 폴란드어 및 포르투갈어(영어와 쌍을 이룬)로 된 세 개의 대규모 데이터셋(ParaCrawl, News Commentary, Europarl)을 문서 수준으로 재구성하였습니다. 또한, 기존의 비텍스트 필터링(bitext filtering)을 대체하는 문서 수준 필터링(document-level filtering) 기법을 소개하고, 이를 통해 컨텍스트 일관성을 유지하는 번역이 가능하도록 했습니다.

- **Technical Details**: ParaCrawl, News Commentary, Europarl에서 문서 수준의 데이터(annotation)를 재구성하기 위해 각 원시 파일의 단일언어 파일을 사용했습니다. 각 세그먼트에 대해 원본 문서에 따라 별도의 어노테이션(annotation)을 적용하여 문서 구조를 재구성하였습니다. 이를 통해 문서 수준의 컨텍스트를 제공할 수 있는 새로운 필터링 방법을 설계하고, 저품질 문서 대신 고품질 문서만을 남기도록 했습니다. 이러한 데이터 수집 및 정제 과정의 결과는 ParaDocs라는 데이터셋으로 공개되었습니다.

- **Performance Highlights**: 문서 수준 컨텍스트를 학습에 적용한 모델들의 성능 평가 결과, 문서 수준 번역의 정확도가 향상되었으며, 문장 수준 번역 성능에는 저하가 없음을 확인하였습니다. 따라서 문서 수준의 데이터를 활용하면 기존의 문장 수준 방법론을 보완할 수 있는 가능성을 보여줍니다.



### Performance of large language models in numerical vs. semantic medical knowledge: Benchmarking on evidence-based Q&As (https://arxiv.org/abs/2406.03855)
- **What's New**: 새로운 연구에서는 대형 언어 모델(LLMs, Large Language Models)이 의료 질문에 대해 언어 기반 응답을 생성하는 능력 외에도, 증거 기반 진단에 필요한 숫자 정보를 얼마나 잘 다룰 수 있는지를 평가했습니다. 이를 위해, '증거 기반 의학 질문 답변 데이터셋(EBMQA)'을 구성하여 Chat-GPT4와 Claude3-Opus 두 모델의 성능을 비교했습니다.

- **Technical Details**: EBMQA는 50,000개 이상의 피어 리뷰 논문에서 추출한 데이터를 기반으로 105,000개의 질문과 답변(QAs)을 포함하고 있으며, 이 질문들은 의료 및 비의료 주제로 분류되고 숫자와 의미적 질문으로 나뉩니다. 이 데이터셋을 바탕으로, 두 개의 최첨단 LLMs인 Chat-GPT4와 Claude3-Opus가 평가되었습니다. 시험에는 총 24,500개의 질문이 사용되었으며, 질문 종류 및 세부 주제별로 모델의 정확도가 평가되었습니다.

- **Performance Highlights**: 평가 결과, 두 LLM 모두 의미적 질문보다 숫자 질문에서 더 높은 정확도를 보였으며, Claude3가 숫자 질문에서 GPT4보다 우수한 성능을 나타냈습니다. 그러나 두 모델 모두 다양한 의료 측면에서 약간의 간극이 있었으며, 여전히 인간 전문가보다 성능이 떨어졌습니다. 따라서 이 모델들의 의료 조언을 신중하게 다루어야 함을 시사합니다.



### Speculative Decoding via Early-exiting for Faster LLM Inference with Thompson Sampling Control Mechanism (https://arxiv.org/abs/2406.03853)
Comments:
          Accepted by ACL 2024 (Findings)

- **What's New**: EESD(Early-exiting Speculative Decoding)의 도입으로 LLM(대형 언어 모델)의 추론 비용을 줄이고 토큰 생성 속도를 증가시킵니다. 이는 초기 N 계층 이후 Early-exiting 구조를 활용하여 드래프트 토큰을 생성하고, Thompson Sampling을 사용하여 드래프트 토큰의 품질을 최적화합니다.

- **Technical Details**: EESD는 다음의 주요 컴포넌트로 구성됩니다: 첫째, LLM의 초기 N 계층에 기반한 Early-exiting 레이어; 둘째, 드래프트 모델의 학습을 향상시키는 자기 증류(self-distillation) 방법; 셋째, 토큰의 품질에 따라 드래프트 토큰의 길이를 적응적으로 결정하는 Thompson Sampling 제어 메커니즘. 이로 인해 kv-cache 공유가 가능해지며 계산 중복이 줄어듭니다.

- **Performance Highlights**: 13B 및 70B 모델에서 종합 실험 결과, EESD는 기존 방법보다 빠른 토큰 디코딩을 보이며, 추론 속도를 크게 향상시키고 여전히 높은 성능을 유지하는 것을 확인했습니다.



### Lean Workbook: A large-scale Lean problem set formalized from natural language math problems (https://arxiv.org/abs/2406.03847)
- **What's New**: 새로운 연구에서는 대규모 언어 모델(LLMs)이 수학 문제를 공식 언어로 번역하는 성능을 개선하는 파이프라인을 제안합니다. 자연어 수학 문제를 Lean 4 명제와 상호 번역하기 위해 반복적으로 합성 데이터를 생성하고 필터링하는 방법을 사용했습니다. 이 기법을 통해 총 57,000개의 공식-비공식 문제 쌍 데이터를 생성하였으며, 이는 LLMs의 성능 향상에 크게 기여했습니다.

- **Technical Details**: 이 연구는 Lean 4를 이용한 수학 문제의 공식화를 목표로 합니다. 연구진은 Mathlib 라이브러리를 활용하여 자연어 수학 문제를 공식화하고, Lean 컴파일러와 자연어 추론(NLI)을 통해 유효성을 검증합니다. 초기 데이터는 MiniF2F와 ProofNet에서 수집되었으며, 번역 모델은 InternLM-Math-Plus-20B를 사용하여 미세 조정했습니다. 전체 번역 프로세스는 여러 번 반복되며, 인적 검토를 통해 정확도를 향상시켰습니다.

- **Performance Highlights**: 최종적으로 생성된 데이터셋은 약 57,000개의 공식-비공식 문제 쌍을 포함하며, 무작위 샘플의 정확성은 93.5%로 보고되었습니다. 또한, 21개의 새로운 IMO 문제도 공식화되었습니다. 이 데이터와 코드는 공개되어 연구와 응용에 사용될 수 있습니다.



### Chaos with Keywords: Exposing Large Language Models Sycophancy to Misleading Keywords and Evaluating Defense Strategies (https://arxiv.org/abs/2406.03827)
Comments:
          To be published in Findings of ACL 2024

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 아첨(sycophantic) 경향을 탐구하며, 사용자의 의도에 맞는 답변을 제공하나 꼭 정확하지 않을 수 있는 경향을 조사합니다. 이는 사용자가 부분적이거나 오해의 소지가 있는 키워드를 이용해 모델에게 완전한 응답을 기대할 때 발생할 수 있는 문제를 중점으로 합니다.

- **Technical Details**: 여러 LLMs에서 관련된 아첨 경향을 실험적으로 분석하고, 잘못된 키워드를 제시할 때 모델이 오정보를 증폭시키는 위험성을 보여주었습니다. 이에 더불어 네 가지 기존 환각(hallucination) 완화 전략을 철저히 평가하여 모델의 아첨 행동을 줄이고자 했습니다. 이들 완화 전략에는 예시 문장 사용, 주의사항 추가, LLM 추론 및 웹 검색을 통한 추가 컨텍스트 제공 등이 포함됩니다.

- **Performance Highlights**: 실험 결과, 모든 아첨 완화 전략이 사실적으로 정확한 문장 생성을 돕는 효과가 있었음을 확인했습니다. 특히, 각기 다른 도메인에서도 아첨 행동이 지속되는 것을 관찰하였고, 다양한 아첨 완화 카테고리에서 LLM이 사실적으로 부정확한 문장을 수정하는 능력을 평가했습니다. 이 연구의 주요 기여는 LLMs의 아첨 행동 문제를 밝혀내고, 이를 완화하는 전략의 유효성을 증명함으로써 신뢰할 수 있는 LLM 개발에 도움이 될 것으로 기대됩니다.



### ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search (https://arxiv.org/abs/2406.03816)
Comments:
          29 pages

- **What's New**: LLM(self-training)의 새로운 방법론은 주로 LLM이 응답을 생성하고, 올바른 출력 응답을 필터링하여 그 데이터를 학습에 사용하는 것에 의존하고 있습니다. 그러나 이러한 접근 방식은 종종 저품질의 fine-tuning(training set)을 초래합니다. 이 논문에서는 LLM self-training을 위한 강화된 방법론인 ReST-MCTS*를 소개합니다. 이 접근 방식은 상과적 보상 지침과 몬테카를로 트리 서치(MCTS)를 결합하여 고품질 reasoning trace와 각 단계별 값을 수집하고 이를 통해 정책 및 보상 모델을 학습합니다.

- **Technical Details**: ReST-MCTS*는 작업별 수동 주석 없이 트리 서치 기반 강화 학습을 통해 과정 보상을 추정하여 LLM(self-training)을 진행합니다. 주어진 최종 올바른 답이 있을 때, ReST-MCTS*는 이 단계가 올바른 답으로 이어질 가능성을 추정하여 올바른 과정 보상을 추론합니다. 이러한 추론 된 보상은 과정 보상 모델을 더욱 개선하는 가치 목표로 활용될 뿐만 아니라 정책 모델의 self-training을 위한 고품질 트레이스 선택에 도움이 됩니다.

- **Performance Highlights**: ReST-MCTS*의 트리 서치(policy)는 동일한 서치 예산 내에서 Best-of-N 및 Tree-of-Thought 등의 기존 LLM(reasoning) baseline보다 높은 정확도를 달성합니다. ReST-MCTS*를 통해 검색된 트레이스를 학습 데이터로 사용하여 지속적으로 3개의 언어 모델을 여러 번 반복하여 강화할 수 있으며, ReST$^{EM}$ 및 Self-Rewarding LM 같은 다른 self-training 알고리즘보다 뛰어난 성능을 보입니다.



### Improving Zero-Shot Chinese-English Code-Switching ASR with kNN-CTC and Gated Monolingual Datastores (https://arxiv.org/abs/2406.03814)
- **What's New**: 최근 발표된 kNN-CTC 기반의 음성 인식 모델이 모노링구얼(한 언어) 환경에서 효과가 있었으나, 다중언어 환경에서는 적용에 어려움을 겪고 있습니다. 이에 대응하기 위해, 연구자들은 kNN-CTC 기반의 새로운 코드 스위칭 ASR(CS-ASR) 프레임워크를 제안했습니다. 이 프레임워크는 두 개의 모노링구얼 데이터스토어와 게이트드 데이터스토어 선택 메커니즘을 활용하여 언어 간 간섭 노이즈를 줄입니다.

- **Technical Details**: 제안된 방법은 각 프레임을 디코딩할 때 적절한 데이터스토어를 선택하여 언어별 정보를 주입합니다. 이를 위해 연구자들은 최신 CTC 기반 모델에 이 프레임워크를 적용하여, 고급 CS-ASR 시스템을 개발했습니다. 이 시스템은 중국어-영어 코드 스위칭 상황에서 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 제안된 게이트드 데이터스토어 메커니즘의 유효성이 다양한 실험을 통해 입증되었습니다. 연구 결과, 이 방법은 특히 중국어-영어 간 코드 스위칭 음성 인식에서 효율적으로 작동하며, 제로샷 (Zero-shot) 학습 환경에서도 우수한 성능을 보였습니다.



### Light-PEFT: Lightening Parameter-Efficient Fine-Tuning via Early Pruning (https://arxiv.org/abs/2406.03792)
Comments:
          Findings of ACL 2024

- **What's New**: Light-PEFT 프레임워크는 대규모 언어 모델의 파라미터 효율성을 개선하여 특정 작업에서 더 효율적인 파인튜닝을 가능하게 합니다. 이 프레임워크는 '기본 모델의 마스킹된 초기 가지치기(Masked Early Pruning)'와 '여러 단계의 PEFT 초기 가지치기(Multi-Granularity Early Pruning)' 두 가지 방법을 포함합니다. Light-PEFT는 초기 학습 단계에서 불필요한 파라미터를 추정하고 가지치기하여, 기존 방식보다 더 효율적으로 파인튜닝할 수 있게 합니다.

- **Technical Details**: Light-PEFT 프레임워크는 초기 학습 단계에서 불필요한 파라미터를 추정하여 잘라냅니다. 기본 모델 내부의 머리(heads)와 중간 차원(intermediate dimensions), PEFT 모듈의 중요도(module importance)와 순위 중요도(rank importance)를 동시에 추정하고 구조적 가지치기(structured pruning)를 통해 불필요한 파라미터를 제거합니다. 이 접근법은 다양한 벤치마크(GLUE, SuperGLUE, QA task)와 모델(RoBERTa, OPT-1.3B, OPT-6.7B)에서 검증되었습니다.

- **Performance Highlights**: Light-PEFT 프레임워크는 기존 방법에 비해 학습 메모리 사용을 39% 줄이고 학습 속도는 1.6배 향상시켰습니다. 또한 추론 메모리 사용을 48% 감소시키고 추론 속도를 1.6배 증가시켰습니다. 이러한 성능 개선을 통해 Light-PEFT는 PEFT의 '플러그 앤 플레이(plug-and-play)' 특성을 유지하면서 학습 및 추론 효율성을 크게 개선합니다.



### End-to-End Trainable Soft Retriever for Low-resource Relation Extraction (https://arxiv.org/abs/2406.03790)
Comments:
          preprint

- **What's New**: 이 연구는 관계 추출 작업에서 텍스트 생성 모델을 사용한 인스턴스 기반 방법의 중요한 도전 과제를 다룹니다. 기존의 비차별적 인스턴스 선택 문제를 해결하기 위해, 소프트하고 차별적인 $k$ 가까운 인스턴스 선택을 활용하는 End-to-end TRAinable Soft K-nearest neighbor retriever(ETRASK)를 제안합니다. 이 방법은 관계 추출 작업에서 리트리버(정답 사례를 검색하는 모델)의 종단 간(end-to-end) 학습을 가능하게 합니다.

- **Technical Details**: ETRASK는 뉴럴 프롬프팅(neural prompting) 방법을 사용하여 $k$ 가까운 인스턴스를 소프트하고 차별적으로 선택합니다. 이를 통해, 비차별적인 인스턴스 선택 문제를 극복하고, 리트리버를 종단 간 최적화할 수 있습니다. ETRASK는 뉴럴 최근접 이웃 네트워크(neural nearest neighbor networks)에서 영감을 받아 소프트 프롬프트를 활용합니다. 또한, 선택된 인스턴스를 소프트 프롬프트로 사용하여 텍스트 생성 모델에서 최적화를 수행합니다.

- **Performance Highlights**: TACRED 데이터셋의 저자원(10% 훈련 데이터) 설정에서 ETRASK는 71.5%의 최고 수준의 F1 점수를 달성했습니다. 이는 기존 베이스라인 모델을 일관되게 향상시켰으며, 모든 설정에서 인스턴스를 추가하여 성능을 개선했습니다. 이러한 결과는 저자원 환경에서 관계 추출 성능을 높이는 ETRASK의 효율성을 강조합니다.



### XL-HeadTags: Leveraging Multimodal Retrieval Augmentation for the Multilingual Generation of News Headlines and Tags (https://arxiv.org/abs/2406.03776)
Comments:
          ACL 2024 camera ready

- **What's New**: 이 논문은 다중언어 뉴스 기사에서 헤드라인과 태그를 생성하기 위한 새로운 방법을 제안합니다. 특히, 기사에 포함된 이미지 및 캡션과 같은 보조 정보(auxiliary information)를 활용하여 관련 문장을 검색하고 이를 통해 다중 언어(context)에서 헤드라인과 태그를 생성하는 방법을 논의합니다. 이를 위해 XL-HeadTags라는 데이터셋을 구축했으며, 이 데이터셋은 6개의 다양한 언어 계열에 걸친 20개의 언어를 포함하고 있습니다.

- **Technical Details**: 이 연구에서는 플러그 앤 플레이(multimodal-multilingual) 검색기를 사용합니다. 이 검색기는 기사 내의 이미지와 캡션 정보를 사용하여 중요한 문장을 찾고, 이를 통해 효율적으로 헤드라인과 태그를 생성합니다. 또한, 다중언어 텍스트를 처리하고 평가하기 위한 도구도 개발하였습니다. 이 도구들은 다중언어 분석의 정확성과 효율성을 높이는 데 기여합니다.

- **Performance Highlights**: 광범위한 평가를 통해 제안된 방법이 헤드라인과 태그 생성 모두에 있어 효과적임을 증명했습니다. 특히, 멀티모달(multi-modal) 및 다중언어(multilingual) 컨텍스트에서 높은 성능을 보였습니다.



### Character-Level Chinese Dependency Parsing via Modeling Latent Intra-Word Structur (https://arxiv.org/abs/2406.03772)
Comments:
          Findings of ACL 2024

- **What's New**: 이 논문은 기존의 단어 수준의 파서를 문자 수준의 의존 구문 분석기(Character-level dependency parser)로 전환하기 위해 단어 내부 구조를 모델링하는 방법을 제안합니다. 각 단어 수준의 의존 트리를 문자 수준의 트리 여러 개로 해석하는 방식이 사용됩니다.

- **Technical Details**: 제안된 방법에서는 단어 내부 구조에 대해 단일 루트(single root)를 보장하고, 이 루트들 간의 의존성을 설정하는 방식으로 제한된 Eisner 알고리듬(constrained Eisner algorithm)을 구현합니다. 이를 통해 문자 수준의 트리 간에 호환성을 유지할 수 있습니다.

- **Performance Highlights**: 중국어 트리뱅크(Chinese treebanks)에서 실험한 결과, 이 방법은 파이프라인(framework) 및 이전의 공동 모델(joint models)을 능가하는 성능을 보였습니다. 세부 분석에 따르면, coarse-to-fine 구문 분석 전략이 모델이 더 언어적으로 그럴듯한(intra-word) 구조를 예측할 수 있게 합니다.



### NAP^2: A Benchmark for Naturalness and Privacy-Preserving Text Rewriting by Learning from Human (https://arxiv.org/abs/2406.03749)
- **What's New**: 이 논문에서는 민감한 텍스트를 처리할 때 개인정보를 보호하기 위해 인간이 사용하는 두 가지 일반적인 전략을 제안합니다: i) 민감한 표현 삭제, ii) 민감한 세부사항을 추상화하여 감추기. 이를 위해 첫 번째 코퍼스인 NAP^2을 구축하여 텍스트 재작성 도구를 개발하였습니다. 기존의 differential privacy(차등 프라이버시) 기반 접근 방식과 달리, 제안된 인간 영감 접근 방식은 자연스러운 재작성 텍스트를 제공하고 프라이버시 보호와 데이터 유용성 간의 균형을 개선합니다.

- **Technical Details**: 이 연구는 사용자가 상업용 대형 언어 모델(LLM)을 사용할 때 민감한 데이터를 제3자에게 노출시키는 문제를 해결합니다. 기존의 redaction(편집)과 anonymization(익명화) 기술은 비자연스러운 텍스트와 정보 유출 위험을 초래합니다. 이에 반해, 제안된 방법은 인간의 텍스트 편집 전략을 바탕으로 민감한 정보를 삭제하거나 추상화하여 자연스러운 재작성 텍스트를 생성합니다. 이를 평가하기 위해 PERSONA-CHAT 코퍼스를 사용하여 NaP² 코퍼스를 구축하고, GPT4를 사용해 3900개의 합성 예제를 생성했습니다. 또한 Privacy_NLI라는 새로운 프라이버시 측정 지표를 사용하여 성능을 평가합니다.

- **Performance Highlights**: T5-Base 모델은 Privacy_NLI 93.81%를 달성하여 높은 수준의 프라이버시 보호 성능을 보였습니다. 이는 삭제 전략을 사용할 때 GPT4보다 우수한 성능을 나타냅니다. 또한, competitive DP(차등 프라이버시) 방법은 Privacy_NLI 점수가 62.14% 이하로 낮았습니다. Privacy_NLI 지표는 Spearman의 순위 상관관계가 0.70으로 인간 평가와 잘 일치했습니다. GPT4는 인간 평가에서 개인정보 보호와 유용성 간의 괜찮은 균형을 보여 GPT-3.5 터보와 다른 오픈 소스 LLMs보다 우수한 성능을 보였습니다. 합성 데이터의 통합은 프라이버시 보호 측면에서 인간이 작성한 데이터로 훈련된 T5-Base 모델의 성능을 7% 향상시켰습니다.



### Efficient Knowledge Infusion via KG-LLM Alignmen (https://arxiv.org/abs/2406.03746)
Comments:
          ACL2024 Findings

- **What's New**: 대형 언어 모델 (LLM)의 도메인별 지식 부족 문제를 해결하기 위한 방법으로, LLM을 활용하여 도메인별 지식 그래프를 효율적으로 구축하고 이를 통해 지식 불일치를 해결하는 연구가 발표되었습니다. 또한, LLM이 지식 그래프의 정보를 효과적으로 활용하도록 하는 세 단계의 KG-LLM 정렬 전략이 제안되었습니다.

- **Technical Details**: 이 연구에서는 소량의 레이블된 샘플과 대규모 코퍼스를 활용하여 도메인별 지식 그래프를 LLM으로 구축하였습니다. 처음에는 소량의 레이블된 데이터를 사용하여 LLM 기반 지식 추출 모델을 훈련합니다. 이후, 비감독 도메인별 코퍼스를 대상으로 추출을 수행하여 도메인 지식 그래프를 구축하고, 간단한 후처리를 통해 결과의 오류를 줄입니다. 마지막으로, 세 단계로 구성된 KG-LLM 정렬 프레임워크를 제안하여 LLM의 KG 활용 능력을 최적화하였습니다.

- **Performance Highlights**: 이 접근 방식은 BioASQ와 CMedQA라는 두 개의 생물의학 질의응답 데이터셋에서 제한된 샘플 설정으로 실험되었습니다. 그 결과, 제안된 방법이 기존의 대조군보다 뛰어난 성능을 보였으며, 특히 지식 불일치 문제와 정보 준수 문제를 효과적으로 해결하는 것으로 나타났습니다.



### LLMEmbed: Rethinking Lightweight LLM's Genuine Function in Text Classification (https://arxiv.org/abs/2406.03725)
Comments:
          ACL 2024 main conference

- **What's New**: 최근 대형 언어 모델(LLM)의 발전으로 프롬프트 학습(prompt-learning)이 다양한 연구 분야에서 주목받고 있습니다. 이 논문에서는 텍스트 분류 성능을 향상시키기 위해 LLM 기반의 간결하고 효과적인 전이 학습 전략인 LLMEmbed를 제안하였습니다. 기존의 복잡한 프롬프트 기반 방법론과 달리, LLMEmbed는 경량 LLM을 통해 텍스트 임베딩을 추출하고 이를 분류기에 학습시키는 방식으로 효율성과 성능을 동시에 향상시킵니다.

- **Technical Details**: LLMEmbed는 경량 LLM의 네트워크 깊이(different network depths)에 따라 텍스트 임베딩을 적절히 추출하고 융합하는 방법을 연구합니다. 이를 통해 높은 견고성과 차별성을 가진 임베딩을 생성하여 분류기의 성능을 높이고자 합니다. 실험 결과는 무거운 LLM (예: GPT-3) 및 복잡한 프롬프트 기반 전략과 비교했을 때 낮은 훈련 오버헤드로 강력한 성능을 제공함을 보여줍니다. LLMEmbed는 공개된 데이터셋에서 상당한 정확도를 달성하면서 모델 파라미터의 4%, 전기 소비의 1.8%, 런타임의 1.5%만 사용합니다.

- **Performance Highlights**: LLMEmbed는 경량 LLM의 의미 임베딩을 활용하여 텍스트 분류에서 SOTA(State-Of-The-Art) 성능을 달성합니다. 추가적인 토큰 오버헤드 없이 직접 입력 텍스트에서 출력 결과로 매핑을 구성하여 예산 절감 효과가 있습니다. 또한, 프롬프트 기반 방법론과 달리 고속 병렬 방식으로 텍스트 분류 작업을 수행할 수 있어 효율성과 유연성이 돋보입니다.



### A Survey on Medical Large Language Models: Technology, Application, Trustworthiness, and Future Directions (https://arxiv.org/abs/2406.03712)
- **What's New**: 최근에는 대형 언어 모델(LLMs)이 의료 분야에서 혁신적이고 강력한 도구로 떠오르면서 전통적인 의료 관행을 변화시키고 향상된 의료 서비스를 예고하고 있습니다. 이 설문 조사에서는 일반 LLM에서 의료 특정 도메인으로의 진화 과정과 의료에 미치는 변화적 영향을 설명합니다.

- **Technical Details**: LLMs의 기본 역사와 기술부터 시작하여, 의료 도메인에서의 적응과 개선 과정을 다룹니다. 특히 임상 추론, 지식 그래프, 검색 기반 생성(retrieval-augmented generation), 인간 정렬(human alignment), 다중 모달 학습(multi-modal learning) 등의 복잡한 의료 환경을 다루는 고급 알고리즘을 강조합니다.

- **Performance Highlights**: Med-LLMs의 광범위한 응용 예로는 임상 의사 결정 지원, 보고서 생성, 의학 교육에서의 성과를 들 수 있습니다. 예를 들어, MedPrompt는 미국 의사 면허 시험(USMLE)에서 90.2의 점수로 인간 전문가(87.0)를 능가하는 결과를 달성했고, HuatuoGPT-II는 중국 의학 자격 시험을 통과했습니다.

- **Challenges**: Med-LLMs의 공정성, 책임, 프라이버시, 견고성을 보장하는 데 있어 여러 도전 과제가 있습니다. 데이터 프라이버시, 표준화, 표현 등 다양한 문제들이 존재합니다.

- **Future Trajectories**: 향후 Med-LLMs의 신중한 확장을 위한 경로를 예측하고 논의합니다. 의료 전문가와 연구자들에게 강점을 제공하고 한계를 분석하며, 의료 환경에서 책임감 있는 혁신을 보장하는 것을 목표로 합니다.



### Synthesizing Conversations from Unlabeled Documents using Automatic Response Segmentation (https://arxiv.org/abs/2406.03703)
Comments:
          findings of ACL 2024

- **What's New**: 이번 연구에서는 대화형 문서 이해(Conversational Question Answering, ConvQA) 시스템 개발에 필요한 훈련 데이터의 부족과 비용 문제를 해결하기 위해 새로운 대화 합성(Synthesizing Conversations) 방법을 제안합니다. 기업 내부 문서의 다양한 코퍼스를 활용하여 검색 엔진 대신 보다 더 직관적으로 문서를 이해할 수 있도록 대화 시스템을 구축하고자 합니다. 특히 문장 경계에서 세그먼트하는 대신 대화 작업을 위해 데이터를 세분화하는 기법을 학습하였습니다. 생성된 합성 데이터셋은 기계 및 인간 평가를 통해 WikiDialog 데이터셋보다 우수한 품질을 보여줍니다.

- **Technical Details**: 우리는 기존의 WikiDialog 데이터셋의 여러 문제점을 분석한 후, 여러 기존 데이터셋을 병합하고 불충분한 데이터를 필터링하여 새로운 데이터셋을 설계하였습니다. 특히 답변 세분화 기법을 도입하여 대화 생성에 특화된 새로운 모델을 개발하였습니다. SynCARS(Synthesizing Conversations using Automatic Response Segmentation) 방법을 통해 문서의 연속적인 N개의 문장을 하나의 상상된 질문에 대한 답변으로 처리합니다(N>1). 이를 위해 기존의 다이얼로그 인페인팅(dialog inpainting) 방법에 몇 가지 수정 사항을 더하여 더 나은 대화형 QA 데이터셋을 생성하였습니다.

- **Performance Highlights**: 우리의 합성된 데이터셋은 WikiDialog와 비교하여 답변의 질과 질문의 구체성 면에서 더 높은 평가를 받았습니다. 또한, 우리의 데이터로 사전 훈련된 질문 검색 시스템은 WikiDialog와 표준 검색 벤치마크 방식으로 훈련된 시스템보다 우수한 결과를 제공합니다.



### M-QALM: A Benchmark to Assess Clinical Reading Comprehension and Knowledge Recall in Large Language Models via Question Answering (https://arxiv.org/abs/2406.03699)
Comments:
          Accepted at ACL 2024 (Findings)

- **What's New**: 본 논문은 대형 언어 모델(Large Language Models, LLMs)이 임상 및 생물 의학 영역에서 어떻게 지식을 회상하고 제시된 정보와 결합할 수 있는지에 대한 이해를 높이기 위한 대규모 실험 연구를 수행하였습니다. 이를 위해 세 개의 일반 생물 의학 하위 도메인과 세 개의 전문 생물 의학 하위 도메인에서 22개 데이터셋을 사용하여 다중 선택형(Multiple Choice) 및 추상형 질문 답변(Abstractive Question Answering) 태스크를 수행했습니다.

- **Technical Details**: 15개의 LLMs의 성능을 하위 도메인, 지식의 출처, 모델 아키텍처에 따라 분석하였으며, 성능 향상에 도움이 되는 요소로 'instruction tuning'이 사용되었습니다. 최근에 제안된 도메인 적응 모델은 적절한 지식이 부족할 수 있지만, 수집한 의학 지식 데이터셋에 대해 직접 미세 조정(fine-tuning)한 결과는 긍정적이었습니다. 모델들이 단순히 필요한 지식을 회상하는 것과 제시된 문맥과 통합하는 능력 사이에는 큰 격차가 있다는 것을 스킬 지향의 수동 오류 분석(skill-oriented manual error analysis)을 통해 밝혔습니다.

- **Performance Highlights**: 수집한 의학 지식 데이터셋에 대한 직접적인 미세 조정이 새로운 전문 하위 도메인에서도 일반화(generalisation)되는 것을 보여주었습니다. 연구 커뮤니티와 협력을 증진하기 위해 표준화된 방법론 및 평가 결과를 공유하는 M-QALM을 제공하였습니다.



### Evaluating the World Model Implicit in a Generative Mod (https://arxiv.org/abs/2406.03689)
- **What's New**: 최근 연구에서는 대형 언어 모델(Large Language Models, LLMs)이 세계 모델(world model)을 암묵적으로 학습할 수 있다고 제안했습니다. 본 연구에서는 이를 평가하는 방법을 제안하며, 결정론적 유한 오토마타(Deterministic Finite Automaton, DFA)로 지배되는 현실을 기준으로 평가 메트릭을 제시합니다. 게임, 논리 퍼즐, 그리고 내비게이션 같은 다양한 도메인에서 새로운 평가 메트릭을 사용하여 기존 진단법으로는 파악할 수 없는 언어 모델의 일관성 부족을 발견했습니다.

- **Technical Details**: 제안된 평가 메트릭은 Myhill-Nerode 정리에 영감을 받아 시퀀스 압축(sequence compression)과 시퀀스 구분(sequence distinction)을 측정합니다. DFA는 고유한 상태 전이를 가지고 있으며, 이를 통해 생성 모델이 이러한 전이를 얼마나 잘 캡처하는지를 평가합니다. 이를 통해, 지도 재구성 및 다양한 알고리즘을 평가할 수 있습니다.

- **Performance Highlights**: 뉴욕시의 택시 운행 데이터를 사용한 실험에서, 트랜스포머 모델이 단기적인 경로 예측에서는 높은 정확성을 보였지만, 실제 지도를 복구하는 데는 실패했습니다. 이는 오셀로(Othello)와 논리 퍼즐의 평가에서도 유사한 결과를 보였습니다. 이러한 발견은 언어 모델이 훈련된 도메인의 정확한 세계 모델을 포착하는 능력을 향상시키기 위한 이론적으로 뒷받침된 평가 메트릭의 중요성을 강조합니다.



### Linguistically Conditioned Semantic Textual Similarity (https://arxiv.org/abs/2406.03673)
Comments:
          To appear in the ACL 2024 main proceedings

- **What's New**: 최신 연구는 Semantic Textual Similarity (STS)의 조건부 측정을 돕기 위해 Conditional STS (C-STS)라는 새로운 접근 방식을 설명합니다. 기존의 C-STS 데이터셋에 존재하는 문제점을 해결하고, 모델 평가를 개선할 방법을 제안합니다.

- **Technical Details**: 연구진은 C-STS 검증 셋을 재주석하고, 원래 레이블에 비해 55%의 주석 오류를 발견했습니다. 이를 해결하기 위해 질의응답 (Question Answering, QA) 기법을 사용하여 조건을 이해하는 모델을 학습시켰습니다. 또한, large language models (LLMs)을 활용해 조건을 기반으로 답변을 생성하고, 이를 통해 주석 오류를 자동으로 식별할 수 있는 파이프라인을 구축했습니다.

- **Performance Highlights**: 제안된 방법은 C-STS 데이터셋의 주석 오류를 80% 이상의 F1 점수로 식별할 수 있었으며, 새로운 모델 학습 방법론을 통해 기존 기법 대비 성능이 크게 향상되었습니다. 또한, TFS (Typed-Feature Structure)를 사용해 더 구체적인 의미 기반의 조건을 부여하는 새로운 주석 사양을 제안했습니다.



### What Makes Language Models Good-enough? (https://arxiv.org/abs/2406.03666)
Comments:
          To appear in Findings of ACL2024

- **What's New**: 이번 연구는 인간의 언어 처리 방식을 모방하는 '충분히 좋은(good-enough)' 언어 모델을 학습시키는 아키텍처 특징을 조사합니다. 특히 Transformer 모델에서 레이어 수와 self-attention 헤드의 수에 초점을 맞추어 연구했습니다. 연구팀은 새로운 평가 데이터셋 GELP(Good-Enough Language Processing)를 생성했으며, 이를 통해 모델의 성능을 평가했습니다.

- **Technical Details**: GELP 데이터셋은 7,680개의 예제로 구성되어 있으며, 두 가지 타당성 유형, 여덟 가지 문장 구조 유형, 그리고 세 가지 메모리 비용을 분석하고자 설계되었습니다. 해당 데이터셋은 군중 소싱 실험을 통해 주석을 달았으며, 이는 이전의 심리언어학 연구를 기반으로 디자인되었습니다. 다양한 레이어와 self-attention 헤드 수를 가진 24개의 BERT 변형 모델을 평가하여, 얕은 깊이와 적은 수의 헤드를 가진 모델도 충분히 좋은 성능을 보임을 발견했습니다.

- **Performance Highlights**: 모델 평가 결과, 레이어 수가 적고 self-attention 헤드 수가 적은 BERT 변형 모델들이 전체 모델과 비슷한 '충분히 좋은' 언어 처리를 할 수 있다는 것을 확인했습니다. 이는 깊은 아키텍처가 필수적이지 않다는 가설을 지지합니다. 하지만, self-attention 헤드의 역할에 대해서는 혼합된 결과가 나왔습니다.



### Is Free Self-Alignment Possible? (https://arxiv.org/abs/2406.03642)
- **What's New**: AlignEZ는 프리트레인 된 언어 모델(LMs)의 자체 생성 선호 데이터와 표현 편집을 사용하여 거의 비용 없이 정렬을 가능하게 하는 혁신적인 접근법입니다. AlignEZ는 인간의 선호 데이터 없이도 모델의 내재된 지식만을 활용하여 정렬이 가능합니다.

- **Technical Details**: AlignEZ는 (1) 자체 생성 선호 데이터와 (2) 표현 편집을 사용하여 비용 없이 정렬을 수행합니다. 자체 생성된 선호 쌍을 통해 모델의 임베딩 공간 내 서브스페이스를 식별하여 바람직하지 않은 컴포넌트를 줄이고 바람직한 컴포넌트를 강화합니다. 이러한 절차는 추론 중에 모델의 임베딩을 수정하는 방식으로 수행됩니다.

- **Performance Highlights**: 실험 결과, AlignEZ는 사전 학습된 모델과 조정된 모델 간의 성능 격차를 평균 31.6% 좁혔습니다. 이는 여섯 개의 데이터셋과 세 가지 모델 아키텍처에 걸쳐 관찰된 결과입니다. 또한 AlignEZ는 소규모의 진실 선호 데이터만을 사용하여 DPO 모델의 성능을 향상시켰습니다. 최종적으로 AlignEZ를 사용하여 비용이 많이 드는 정렬 절차를 가속화할 수 있는 가능성을 탐구하였습니다.



### TACT: Advancing Complex Aggregative Reasoning with Information Extraction Tools (https://arxiv.org/abs/2406.03618)
Comments:
          Website (this https URL), Huggingface (this https URL)

- **What's New**: 최근 연구에서는 대형 언어 모델(LLMs)이 다양한 텍스트를 종합하여 정보를 집계하는 질의에 대해 성능이 떨어진다는 문제를 다루고 있습니다. 이를 평가하고 모델링 노력을 촉진하기 위해, 'TACT - Text And Calculations through Tables'라는 새로운 데이터셋이 소개되었습니다. 이 데이터셋은 복잡한 명령을 통해 모델의 추론 및 계산 능력을 평가합니다.

- **Technical Details**: TACT 데이터셋은 기존 텍스트와 그에 연관된 표 데이터를 사용해 만들어졌으며, 새로운 질의를 구성하고 각각의 정답을 수집하는 과정으로 구성되었습니다. 모델 성능을 세부적으로 분석하기 위해 '표 생성(table-generation)', 'Pandas 명령 생성(command-generation)', 및 '실행(execution)' 세 가지 구성 요소로 나누어 평가했습니다. 특히, 모든 현 LLM들이 해당 데이터셋에서 38% 미만의 정확도를 기록했으며, 각 구성 요소에서 상당한 도전 과제가 있다는 점을 밝혀냈습니다.

- **Performance Highlights**: TACT 데이터셋의 복잡한 명령을 현재의 언어 모델들이 효과적으로 처리하지 못하는 이유는 여러 가지로 분석되었습니다. 특히, 테이블 내 계산 및 텍스트 해석, Pandas 명령 생성을 포함한 다양한 요소에서 모델이 어려움을 겪고 있습니다. 'IE as a tool' 이라는 새로운 모델링 프레임워크를 제안하여 각 단계를 해결하기 위한 '도구(tool)'를 추가하는 방법으로 성능을 개선하려는 시도가 이루어졌으며, 이는 기존의 프롬프트 기법보다 더 나은 성능을 보여주었습니다.



### Knowledge-Infused Legal Wisdom: Navigating LLM Consultation through the Lens of Diagnostics and Positive-Unlabeled Reinforcement Learning (https://arxiv.org/abs/2406.03600)
Comments:
          Accepted by ACL Findings 2024

- **What's New**: 최근 연구는 AI의 법률 분야 통합을 가속화하면서, 특히 법률 배경이 없는 사용자가 전문적인 질의 형성에 어려움을 겪고 중요한 법적 요소를 간과할 가능성을 줄이기 위해 D3LM(진단 법률 대형 언어 모델)을 제안합니다. D3LM은 적응형 변호사와 같은 진단 질문을 통해 추가적인 사건 정보를 수집하며, 이에 대한 고품질 피드백을 제공합니다. 또한, D3LM은 혁신적인 그래프 기반 Positive-Unlabeled Reinforcement Learning(PURL) 알고리즘을 통합하여 중요한 질문의 생성을 가능하게 하고 사용자와의 상호작용을 개선합니다. 본 연구는 미국 법률 데이터베이스를 기반으로 하는 새로운 영어 CVG(Court Views Generation) 데이터셋도 도입하였습니다.

- **Technical Details**: D3LM은 추가적인 사건 정보를 수집하기 위해 적응형 진단 질문을 사용하는 모델입니다. 그래프 기반 Positive-Unlabeled Reinforcement Learning(PURL) 알고리즘을 통해 중요한 질문을 생성하고 사용자와 LLM 간의 상호작용을 향상시킵니다. 또한, 새로운 LLM 기반 정지 기준을 통합하여 정확한 법원 견해 생성을 촉진합니다. 본 연구에서는 미국 판례 데이터베이스를 기반으로 한 새로운 영어 CVG 데이터셋을 소개하여 LLM 연구와 배포를 풍부하게 합니다.

- **Performance Highlights**: D3LM은 전통적인 LLM보다 뛰어난 성과를 자랑하며, 법률 분야에서의 탁월한 사용자 경험을 제공합니다. 또한, 법률 시나리오에서의 정보 수집 능력을 향상시키기 위한 강화 학습 접근 방식을 사용하여 실제 법률 상황에서의 정확성과 실용성을 입증합니다.



### Measuring Retrieval Complexity in Question Answering Systems (https://arxiv.org/abs/2406.03592)
Comments:
          Accepted to ACL 2024 (findings)

- **What's New**: 이 논문에서는 retrieval-based QA에서 어려운 질문을 식별하기 위한 'Retrieval Complexity (RC)'라는 새로운 측정 기준을 제안합니다. 이를 통해 특정 retrieval 시스템에 대해 질문이 얼마나 어려운지를 측정할 수 있는 비지도 파이프라인을 도입했습니다.

- **Technical Details**: 논문에서 제안한 비지도 파이프라인 'Reference-based Question Complexity Pipeline (RRCP)'는 주어진 질문과 하나 이상의 참조 답변을 사용하여 관련 문서들을 검색하고, 이를 바탕으로 독창적인 평가자 'GenEval'을 사용하여 RC 점수를 계산합니다. 이렇게 얻은 RC는 문제를 풀기 위해 필요한 증거가 검색된 문서들 사이에 얼마나 분산되어 있는지를 대략적으로 측정합니다.

- **Performance Highlights**: RRCP는 6개의 QA 벤치마크에서 대체 추정기들보다 더 정확하게 RC를 측정하는 것으로 나타났습니다. 또한, 5개의 벤치마크에서 RC 점수와 QA 성능, 전문가 평가 사이에 강한 상관 관계가 있음이 확인되었습니다. 따라서 RC는 질문의 난이도를 효과적으로 평가하는 지표로 작용할 수 있습니다.



### Ranking Manipulation for Conversational Search Engines (https://arxiv.org/abs/2406.03589)
- **What's New**: 최근 연구는 사용자 쿼리에 대한 응답으로 대형 언어 모델(LLM) 생성 콘텐츠를 신속하게 통합하는 대화형 검색 엔진이 급격히 증가하고 있음을 보여줍니다. 그러나 이러한 LLM은 보안 및 품질 목표를 방해하는 'jailbreaking'과 'prompt injection' 공격에 매우 취약합니다. 본 연구는 대화형 검색 엔진에서 참조된 소스의 순위에 미치는 prompt injection의 영향을 조사합니다. 이를 위해 실제 소비자 제품 웹사이트의 집중된 데이터셋을 도입하고, 대화형 검색 순위를 적대적인 문제로 공식화합니다.

- **Technical Details**: 대화형 검색 엔진은 Retrieval-Augmented Generation(RAG) 아키텍처를 기반으로 합니다. RAG 모델은 입력 프롬프트와 관련된 텍스트를 벡터 인덱스에서 검색하여 LLM 문맥에 연결함으로써 외부 데이터베이스의 정보를 통합합니다. 본 연구는 여러 LLM 간의 제품 이름, 문서 내용, 문맥 위치가 RAG 순위에 미치는 영향을 분석하고, 'tree-of-attacks' 기반의 jailbreaking 기술을 제시하여 낮은 순위의 제품을 신뢰성 있게 상위로 유도할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, 서로 다른 LLM은 제품 이름, 문서 내용, 문맥 위치에 대해 우선순위가 크게 다름을 확인했습니다. 특히, 제안된 공격 기법은 최첨단 대화형 검색 엔진에 효과적으로 적용되는 것을 증명했습니다. 이는 웹사이트 소유자가 검색 순위를 향상시키기 위한 강력한 재정적 인센티브가 있는 점을 고려할 때, 미래의 견고성 작업에 있어 매우 중요한 문제로 간주됩니다.



### Verbalized Machine Learning: Revisiting Machine Learning with Language Models (https://arxiv.org/abs/2406.04344)
Comments:
          Technical Report v1 (92 pages, 15 figures)

- **What's New**: 이번 연구에서, 연구진은 기존의 머신 러닝 모델과는 달리, 인간이 이해할 수 있는 자연어로 매개변수를 제한하는 Verbalized Machine Learning (VML) 프레임워크를 소개했습니다. 이는 LLM (Large Language Models)과 텍스트 프롬프트를 통해 함수 근사화(function approximation)의 새로운 관점을 제시합니다.

- **Technical Details**: VML은 텍스트 프롬프트를 통한 LLM 매개변수를 사용하는 접근 방식으로, 기존의 머신 러닝 문제인 회귀 분석(regression)과 분류(classification)을 재검토합니다. 주요 기술적 이점으로는 귀납적 편향(inductive bias)의 간편한 인코딩, 자동 모델 클래스 선택(automatic model class selection), 그리고 해석 가능한 학습자 업데이트입니다. 또한, 모델 매개변수를 텍스트 요약으로 고정 길이로 유지하는 증분 업데이트 방식(incremental updating with summarization)을 제안합니다.

- **Performance Highlights**: VML의 성능은 귀납적 편향을 자연어로 쉽게 인코딩하고 학습자 업데이트 이유를 설명할 수 있다는 점에서 우수합니다. 가장 흥미로운 점은 VML이 Kolmogorov Complexity와 Occam's Razor 적용을 통해 간결하고 심플한 텍스트로 데이터를 설명할 수 있는 잠재력을 보인다는 점입니다.



### Improving Alignment and Robustness with Short Circuiting (https://arxiv.org/abs/2406.04313)
- **What's New**: AI 시스템에서 발생할 수 있는 해로운 행동을 억제하고 적대적 공격에 대한 취약성을 줄이기 위한 새로운 접근법, 이른바 '숏서킷(Short-circuiting)' 기법이 제안되었습니다. 이 기법은 기존의 거부 훈련(refusal training)이나 적대적 훈련(adversarial training) 대신 모델의 내부 표현을 직접 제어함으로써 해로운 출력을 방지합니다.

- **Technical Details**: 숏서킷 기법은 내부 표현(representation)을 '리매핑(remap)'하여 해로운 출력이 생성되는 과정을 중단시키는 방식입니다. 특히 'Representation Rerouting (RR)'이라는 기법을 통해 모델이 해로운 출력을 생성하려 할 때 해당 과정을 방해하고 중단시킵니다. 이 방법은 텍스트 전용 모델뿐만 아니라 멀티모달 모델(multimodal models)에도 적용될 수 있으며, 공격의 형태나 강력한 적대적 공격에도 견디도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 숏서킷 기법은 최신 대형 언어 모델(LLMs)의 해로움을 현격히 줄였으며, 이전에 보지 못한 다양한 적대적 공격에도 유의미하게 성능이 개선되었습니다. 특히, Llama-3-8B-Instruct 모델을 Cygnet으로 미세조정하여 해로운 출력을 약 두 단계(order of magnitude) 줄였으며, 멀티모달 모델에서도 유사한 성과를 보였습니다.



### Measuring and Addressing Indexical Bias in Information Retrieva (https://arxiv.org/abs/2406.04298)
Comments:
          ACL 2024

- **What's New**: PAIR 프레임워크는 랭킹된 문서나 전체 정보 검색 시스템에 대한 자동 바이어스 감사를 지원합니다. DUO라는 최초의 일반적인 자동 바이어스 메트릭을 도입하여 32k의 합성 문서와 4.7k의 자연 문서를 포함한 새로운 코퍼스에서 8개의 정보 검색 시스템을 평가했습니다. 인간 행동 연구를 통해 우리의 접근이 독자의 의견에 어떤 영향을 미칠 수 있는지를 예측할 수 있음을 입증했습니다.

- **Technical Details**: Perspective-Aligned Information Retrieval, 또는 PAIR은 랭킹된 문서 내에서 관점의 분산을 측정하는 Discounted Uniformity of Perspectives(DUO) 바이어스 메트릭을 도입했습니다. 이 메트릭은 문서 임베딩(embeddings)에 대한 주성분 분석을 사용하여 이슈마다 가장 극단적인 의미 축을 미리 계산합니다. PAIR은 Wiki-Balance 코퍼스를 사용하며, 이는 연속적으로 확장 가능합니다.

- **Performance Highlights**: DUO 메트릭을 사용하여 최소화된 랭킹을 통해 독자의 의견 변화를 줄일 수 있음을 입증했습니다. 기존의 7개의 오픈 소스 IR 시스템과 하나의 상업적 검색 엔진을 평가했으며, 32k의 합성 문서와 4.7k의 구글 검색을 통한 자연 문서를 사용하여 종합적인 바이어스 감사를 수행했습니다. 우리의 접근 방식은 SEME(Search Engine Manipulation Effect)를 예측하는 데 유효함을 확인했습니다.



### VISTA: Visualized Text Embedding For Universal Multi-Modal Retrieva (https://arxiv.org/abs/2406.04292)
Comments:
          Accepted to ACL 2024 main conference

- **What's New**: 다중 모달 검색(multi-modal retrieval)에서 기존의 텍스트 중심 모델들이 시각 정보를 처리할 수 없는 한계를 극복하기 위해 새로운 임베딩 모델 VISTA가 소개되었습니다. VISTA는 텍스트와 이미지 데이터를 결합하는 능력을 가집니다.

- **Technical Details**: VISTA는 강력한 텍스트 인코더를 기반으로 하여 시각적 토큰 임베딩(visual token embeddings)을 도입해 이미지를 이해할 수 있는 유연한 아키텍처를 제안합니다. 모델 훈련을 위해 두 가지 데이터 생성 전략을 개발했으며, 토큰 정렬과 다중 모달 표현 학습을 위한 다단계 훈련 알고리즘이 도입되었습니다.

- **Performance Highlights**: VISTA는 다양한 다중 모달 검색 과제에서 뛰어난 성능을 보여주며, 제로샷(zero-shot)과 지도 학습(supervised settings) 환경 모두에서 우수한 결과를 달성했습니다. 특별한 최적화 없이도 최첨단 방법론을 능가하거나 유사한 성과를 나타냈습니다.



### Self-Play with Adversarial Critic: Provable and Scalable Offline Alignment for Language Models (https://arxiv.org/abs/2406.04274)
- **What's New**: 이 연구는 오프라인 선호 데이터와 일치하도록 대형 언어 모델(LLMs)을 조정하는 도전 과제를 다룹니다. 특히, Reinforcement Learning from Human Feedback (RLHF)를 통해 조정을 강조합니다. 기존의 선호 최적화 방법은 실질적으로 좋은 성능을 보이지만, 이론적으로 최적 정책에 수렴할 보장이 없습니다. 이에 반해, 이론적으로 동기부여된 선호 최적화 방법은 대규모 응용에 비해 효율적이지 않습니다. 이 격차를 해소하기 위해, SPAC(self-play with Adversarial Critic)이라는 새로운 오프라인 선호 최적화 방법을 제안합니다. 이는 온평균 비관적 기술(온-애버리지 페시미즘)을 활용하여 이론적으로 보장된 확장 가능한 LLM 정렬 방식입니다.

- **Technical Details**: SPAC는 오프라인 강화 학습 문헌에서 영감을 얻어 Stackelberg 게임으로 오프라인 선호 최적화를 공식화합니다. 학습자가 비관적 보상 추정에 따라 정책을 최적화하는 동안, 비평가는 학습자의 정책 하에서 비관적 보상을 유지하는 이중 역학 구조를 갖춥니다. 또한 SPAC는 기존의 RLHF 코드베이스위에 쉽게 구현할 수 있으며, 알파고 제로나 GAN 같은 최신 기계 학습 응용에서도 널리 사용된 개념인 자기 대결(self-play)을 활용합니다.

- **Performance Highlights**: SPAC는 7B 규모의 LLM 정렬 실험에서 높은 성능을 보였습니다. Open LLM 리더보드 평가에서 SPAC는 인기 있는 정렬 기반과 비교하여 경쟁력을 입증했습니다. SPAC는 단일 정책 집중성(single-policy concentrability) 하에서 최적 정책에 수렴할 이론적 결과와 비동기적(suboptimality) 분석을 제공합니다.



### MLVU: A Comprehensive Benchmark for Multi-Task Long Video Understanding (https://arxiv.org/abs/2406.04264)
- **What's New**: 기존 비디오 이해 벤치마크가 다양한 문제와 제약으로 인해 긴 비디오 이해 성능(Long Video Understanding, LVU)을 평가하는 데 한계가 있다는 문제를 해결하기 위해 새로운 벤치마크 MLVU(Multi-task Long Video Understanding Benchmark)를 제안합니다. MLVU는 다양한 비디오 길이, 장르, 그리고 평가 과제를 포함하여 LVU 성능을 포괄적이고 심도 있게 평가할 수 있는 방법을 제공합니다.

- **Technical Details**: MLVU는 3분에서 2시간까지 다양한 길이의 비디오를 포함하며, 평균 비디오 길이는 약 12분입니다. 또한 영화, 다큐멘터리, 감시 비디오, 자아 중심 비디오, 게임, 만화 등 다양한 비디오 장르를 포함합니다. 평가 과제는 인지(Cognition), 캡션 작성(Captioning), 인식(Recognition), 요약(Summarization) 등 총 9개의 과제로 구성되며, 다중 선택 및 자유 형식 생성 과제를 포함하여 MLLM의 종합적인 능력을 평가합니다.

- **Performance Highlights**: 20개의 최신 MLLM을 통한 실험 결과, 긴 비디오 이해는 여전히 기술적으로 도전적인 문제임이 확연히 드러났습니다. GPT-4는 실험에서 가장 높은 성능을 보여주었지만, 다중 선택 과제에서 평균 64.6%의 점수에 그쳤습니다. 모든 방법이 비디오 길이가 길어질수록 성능 저하를 겪었으며, 세밀한 정보가 필요한 작업에서는 어려움을 보였습니다. 성능 향상을 위해 맥락 길이(extension of context length), 이미지 이해 능력, 강력한 LLM 백본(backbone)의 선택 등이 중요한 역할을 한다고 평가되었습니다.



### Hypernetworks for Personalizing ASR to Atypical Speech (https://arxiv.org/abs/2406.04240)
- **What's New**: 이번 연구는 비정상적인 음성(예를 들어, 실어증, 발음 장애 등)에 적응하기 위해 발전된 자동 음성 인식 시스템(ASR)의 새로운 접근법을 제안합니다. 기존의 방법들은 특정 음성 장애의 사전 진단이 필요했지만, 이번 접근법은 이러한 사전 지식 없이도 보다 효과적으로 다양한 음성 장애에 적응할 수 있습니다.

- **Technical Details**: 연구팀은 모델 파라미터 중 최소한의 집합을 식별하여 ASR 적응에 필요한지를 분석했습니다. 이를 통해 전체 파라미터의 0.03%만을 조절함으로써 Word Error Rate (WER)을 절반으로 줄였습니다. 또한, 기존의 코호트(cohort)나 개별 수준의 미세 조정 대신, 하이퍼네트워크(hypernetwork)를 이용해 발화 단위로 실시간으로 적응을 생성하는 방법을 제안했습니다. 하이퍼네트워크는 메타러닝(meta-learning) 절차를 통해 다양한 음성 장애에 적응할 수 있는 파라미터를 생성합니다.

- **Performance Highlights**: 평가 결과, 하이퍼네트워크는 out-of-distribution 화자에 대해 기존 방법보다 일반화 성능이 뛰어나며, 전체 파라미터의 0.1%만을 사용해 WER을 75.2% 상대적으로 감소시켰습니다. 이는 기존 방법론에 비해 훨씬 효과적인 결과를 보여줍니다.



### The CLRS-Text Algorithmic Reasoning Language Benchmark (https://arxiv.org/abs/2406.04229)
Comments:
          Preprint, under review. Comments welcome

- **What's New**: 이번 논문은 CLRS-Text라는 새로운 벤치마크를 제안합니다. 이 벤치마크는 기존 CLRS 벤치마크를 기반으로 하며, 텍스트 형태로 알고리즘 트레이스를 생성합니다. 이는 다양한 알고리즘 태스크에서 데이터셋을 절차적으로 생성할 수 있어 언어 모델(LMs)의 추론 능력을 평가하는데 적합합니다.

- **Technical Details**: CLRS-Text는 CLRS의 그래프 기반 트레이스를 텍스트 형태로 변환하여, 언어 모델이 텍스트 데이터를 처리하는데 적합하게 합니다. 이 벤치마크는 30가지 다양한 알고리즘 태스크에서 트레이스 데이터를 생성할 수 있으며, 새로운 알고리즘 태스크도 추가할 수 있는 표준 파이프라인을 제공합니다.

- **Performance Highlights**: 여러 언어 모델을 CLRS-Text 벤치마크에서 파인튜닝 및 평가한 결과, 이전 연구를 검증했으며 언어 모델 추론 커뮤니티를 위한 새로운 도전 과제를 제시했습니다. 모델의 성능은 다양한 문제 사례에서 일관되게 동작해야 하고, 이는 CLRS-Text를 통해 평가할 수 있습니다.



### AgentGym: Evolving Large Language Model-based Agents across Diverse Environments (https://arxiv.org/abs/2406.04151)
Comments:
          Project site: this https URL

- **What's New**: 이번 논문에서는 일반적인 능력을 갖춘 LLM 기반 에이전트의 자기 진화(self-evolution) 가능성을 연구하여 연결된 다양한 환경에서 작업을 수행할 수 있는 에이전트를 개발하려고 합니다. 연구팀은 AgentGym이라는 새로운 프레임워크를 제안하여 실시간 상호작용 및 다양한 환경, 작업을 통한 훈련과 평가를 도울 수 있도록 했습니다.

- **Technical Details**: AgentGym 프레임워크는 14개의 에이전트 환경, 89개의 다양한 작업을 포함하여 웹 작업, 신체적 작업 등을 다루고 있습니다. 이 프레임워크는 확장된 명령어 데이터베이스, 벤치마크 Suite (AgentEval), 고품질의 상호작용 궤적 데이터 (AgentTraj 및 AgentTraj-L)을 포함합니다. 에이전트의 자기 진화를 촉진하는 새로운 알고리즘인 AgentEvol을 제안하여, 새로운 과제와 지시를 탐험하고 학습하는 과정에서 에이전트의 진화 가능성을 평가합니다.

- **Performance Highlights**: 실험 결과, 진화된 에이전트는 SOTA 모델과 비교할 만한 성과를 보여주었으며, 본 연구에서 제안한 방법이 어떻게 작동하는지에 대한 충분한 분석과 검증을 수행했습니다. AgentGym의 전체 스위트, 알고리즘 구현 및 에이전트 체크포인트를 공개하여 커뮤니티가 새로운 알고리즘과 진전을 이룰 수 있도록 돕고자 합니다.



### Promoting Fairness and Diversity in Speech Datasets for Mental Health and Neurological Disorders Research (https://arxiv.org/abs/2406.04116)
Comments:
          34 pages

- **What's New**: 최근 머신러닝과 AI 연구가 모델링과 성능 평가에 중점을 두는 경향이 있지만, 데이터 수집의 중요성을 간과해서는 안 된다는 점이 부각되고 있습니다. 특히, 정신 건강 및 신경학적 장애(MHND)와 같은 민감한 도메인에서는 음성 데이터가 환자의 건강 개선 및 의료 제공자를 지원하는 AI 응용 프로그램 개발에 사용되므로 데이터의 한계와 편향이 신뢰성과 신뢰성에 큰 영향을 미칠 수 있습니다. 이 논문에서는 MHND 도메인을 위한 사용 가능한 음성 데이터셋의 현황을 파악하고, 개선할 기회와 공정성 및 다양성을 촉진할 수 있는 방법을 제시합니다. 또한, 데이터 수집 시의 윤리적 문제를 강조하는 체크리스트를 제공하여 더 책임감 있는 연구를 촉진하고자 합니다.

- **Technical Details**: 전통적인 AI 시스템 개발은 모델 중심의 생명 주기를 따르며, 데이터 수집 및 관리가 자주 간과됩니다. 그러나 데이터 품질과 신뢰성을 중시하는 최근 트렌드는 모델 설계보다 데이터 수집의 중요성을 강조합니다. 특히, MHND 도메인에서는 정확한 정책과 프로토콜을 따르지 않으면 시스템의 신뢰성과 신뢰성이 저하될 수 있습니다. 논문에서는 MHND 응용 프로그램을 위한 음성 데이터셋 구축 시 필요한 필수 기능과 속성을 정의하고, 이를 윤리적 관심사가 반영된 체크리스트 형태로 정리했습니다.

- **Performance Highlights**: 논문은 DAIC-WoZ 같은 기존 데이터셋이 성능에 영향을 미칠 수 있는 편향성과 불균형 문제를 제기합니다. DAIC-WoZ는 우울증 자동 인식을 위한 주요 데이터셋으로 사용되고 있으며, 다양한 연구에서 주요 방법론을 구축하는 데 중요한 역할을 했습니다. 그러나 데이터셋의 불균형, 잘못된 라벨링, 유지보수 문제 등 다양한 한계가 존재하여 더 균형 잡힌 데이터 수집 프로세스를 필요로 합니다. 논문은 36개의 관련 논문을 조사하여 현재의 관행을 비판적으로 검토하고 개선이 필요한 영역을 강조합니다.



### MuJo: Multimodal Joint Feature Space Learning for Human Activity Recognition (https://arxiv.org/abs/2406.03857)
- **What's New**: 최근 AI 분야에서 인간 행동 인식(Human Activity Recognition, HAR) 문제를 해결하기 위한 새로운 방법인 MuJo(Multimodal Joint Feature Space Learning)가 제안되었습니다. 이 방법은 비디오, 언어, 자세(pose), IMU 센서 데이터 등 다양한 모달리티 데이터를 통합하여 HAR 성능을 향상시킵니다. 특히, 새로운 대규모 데이터셋(MM-Fit dataset)도 함께 소개되었으며, 이 데이터셋은 병렬 비디오, 언어, 자세, 센서 데이터 포인트로 구성되어 연구를 지원합니다.

- **Technical Details**: MuJo 방식은 대조 학습(contrastive learning)과 멀티태스킹 학습(multitask learning) 방법을 결합하여, 비디오, 언어, 자세, IMU 센서 데이터를 이용해 다중 모달리티의 공동 특징 공간(joint feature space)을 학습합니다. 이 방법은 기존의 높은 품질의 영상 입력이 필요한 HAR 방법들과는 다르게, 다양한 센서 데이터를 활용하여 더욱 견고한 성능을 보입니다.

- **Performance Highlights**: MM-Fit 데이터셋을 사용한 실험에서, MuJo 모델은 전체 훈련 데이터의 2%만 사용했을 때도 매크로 F1-스코어(Macro F1-Score) 0.992를 기록하였으며, 모든 훈련 데이터를 사용한 경우 0.999라는 높은 성능을 달성했습니다. 또한, 새로운 데이터셋에 대해서도 최대 0.638의 일반화 성능을 보여, 다른 모달리티들 사이의 지식 전이를 통해 적은 양의 훈련 데이터로도 높은 성능을 보였습니다.



### Tool-Planner: Dynamic Solution Tree Planning for Large Language Model with Tool Clustering (https://arxiv.org/abs/2406.03807)
Comments:
          46pages first version

- **What's New**: 본 논문에서는 Tool-Planner라는 새로운 태스크 처리 프레임워크를 소개합니다. 이 프레임워크는 도구(tool) 학습에서 발생하는 주요 문제들을 해결하기 위해 고안되었습니다. Tool-Planner는 API 기능에 따라 도구들을 툴킷(toolkit)으로 그룹화하고, 이를 통해 다양한 툴킷 간의 계획 수립을 가능하게 합니다.

- **Technical Details**: Tool-Planner는 언어 모델(LLM)이 각 도구의 사용 예와 해당 기능을 학습하도록 하여 복잡한 문제를 해결할 수 있게 합니다. 주요 기능은 도구 오류 발생 시 툴킷을 기반으로 도구를 재선택하고 조정할 수 있는 능력입니다. 이를 통해 불필요한 오류 수정으로 인한 계획 불안정성과 긴 실행 시간을 해결합니다. 주요 대상 모델로는 GPT-4와 Claude 3가 있습니다.

- **Performance Highlights**: 실험 결과, Tool-Planner는 여러 데이터셋에서 높은 통과율(pass rate)과 승률(win rate)을 달성하였습니다. 이는 도구 학습에 있어서 계획 체계를 최적화하는 데 있어 뛰어난 잠재력을 보여주는 결과입니다.



### Your Absorbing Discrete Diffusion Secretly Models the Conditional Distributions of Clean Data (https://arxiv.org/abs/2406.03736)
- **What's New**: 이번 논문에서는 흡수 과정(absorbing process)을 가지는 이산 확산 모델(discrete diffusion model)이 언어 모델링에서 유망한 성과를 보였음을 밝히고 있습니다. 저자들은 콘크리트 스코어(concrete score)를 조건부 확률(conditional probabilities)과 시간에 따라 변화하는 스칼라의 형태로 표현할 수 있음을 발견했습니다. 이를 바탕으로 시간에 독립적인 조건부 확률을 특징으로 하는 RADD(Reparameterized Absorbing Discrete Diffusion)을 제안했습니다.

- **Technical Details**: RADD는 SEDD(Score Entropy Discrete Diffusion)의 시간 조건을 제거하여 간단하면서도 효율적인 모델링을 제공합니다. 노이즈가 일정한 샘플링 구간 동안 변하지 않을 때, 시간 조건이 없는 네트워크 출력을 캐시하여 함수 평가 횟수(NFEs)를 줄일 수 있습니다. 또한, 새로운 콘크리트 스코어의 분해 방식을 기반으로 DCE(Denoising Cross-Entropy)라는 단순한 형태로 흡수 확산의 정확한 가능도를 재작성할 수 있음을 증명했습니다.

- **Performance Highlights**: 실험 결과, RADD는 가장 강력한 기준선보다 최대 3.5배 더 빠르면서도 일관되게 더 나은 성능을 보였습니다. 특히, 5개의 제로 샷 언어 모델링 벤치마크에서 획기적인 향상을 나타냈습니다(GPT-2 스케일 기준). 이는 새로운 이론적 발견이 실제 성능 향상으로 이어졌음을 실험적으로 검증한 것입니다.



### Generalization-Enhanced Code Vulnerability Detection via Multi-Task Instruction Fine-Tuning (https://arxiv.org/abs/2406.03718)
Comments:
          Accepted to ACL 2024 Findings

- **What's New**: 최근 몇 년간 코드를 사전에 학습한 모델(Code Pre-trained Models, CodePTMs)을 통한 취약점 탐지는 유망한 결과를 보여주었지만, 실제 사용 환경에서는 일반화에 어려움을 겪고 있습니다. 이러한 문제를 해결하기 위해 다중 작업 학습(multi-task learning)과 대형 언어 모델(Large Language Models, LLMs)을 통합한 새로운 프레임워크인 VulLLM을 소개합니다. 이 프레임워크는 취약점 패치를 활용해 취약점 위치(Localization)와 취약점 해석(Interpretation) 작업을 추가로 구성하여 보다 깊이 있는 취약점 특징을 효율적으로 탐색합니다.

- **Technical Details**: VulLLM은 취약점 탐지와 더불어 두 가지 보조 작업을 추가하여 모델이 취약점의 근본 원인을 파악하도록 유도합니다. 첫째, 취약점 패치를 이용한 취약점 위치 탐지 작업을 구성했습니다. 둘째, 패치로부터 추출한 취약점 특징을 기반으로 GPT-4를 활용한 취약점 해석 작업을 구축했습니다. 이러한 접근 방식은 생성형 대형 언어 모델(generative LLMs)을 활용하여 복잡한 취약점 패턴을 이해함으로써, 단일 작업의 표면적인 특징에 과적합(overfitting)되지 않도록 유도합니다.

- **Performance Highlights**: 여섯 개의 대규모 데이터셋을 대상으로 수행한 실험 결과에 따르면, VulLLM은 효과성, 일반화 능력, 그리고 견고성 측면에서 기존의 최첨단 모델 7종을 능가하였습니다.



### What Should Embeddings Embed? Autoregressive Models Represent Latent Generating Distributions (https://arxiv.org/abs/2406.03707)
Comments:
          15 pages, 8 figures

- **What's New**: 이 논문은 자기회귀(autoregressive) 언어 모델이 왜 특정한 내재적 구조를 표현하는지에 대한 통찰을 제공합니다. 특히, 이 연구는 자기회귀 예측 목적을 예측 통계량(predictive sufficient statistics)과 연결지어, 세 가지 상황에서 최적의 임베딩 내용을 어떻게 식별할 수 있는지 설명합니다. 이를 통해 임베딩이 독립적이고 동일하게 분포된 데이터, 잠재 상태 모델, 그리고 이산 가설 공간에서 데이터의 후분포(posterior distribution)를 반영해야 함을 보여줍니다.

- **Technical Details**: 이 연구는 Bayesian inference와 자기회귀 언어 모델의 관계를 탐구합니다. 자기회귀 학습 목표를 통해 예측 통계량을 찾는 방법을 제시함으로써 최적의 임베딩 내용을 세 가지 경우에 대해 정의합니다: 1) 독립적으로 동일하게 분포된 데이터에서는 데이터의 충분한 통계량을 포착해야 합니다. 2) 잠재 상태 모델에서는 주어진 데이터에 기반한 상태의 후분포를 인코딩해야 합니다. 3) 이산 가설 공간에서는 데이터에 기반한 가설의 후분포를 반영해야 합니다. 이 연구는 또한 Transformers가 이 세 가지 유형의 잠재 생성 분포를 인코딩하며, out-of-distribution에서도, 그리고 토큰 암기 없이도 성능이 우수함을 보입니다.

- **Performance Highlights**: 경험적 탐구를 통해 LM 임베딩이 이러한 관련 정보를 디코딩할 수 있음을 확인하였습니다. 특히, 예측 통계량을 통해 캡처될 것으로 예상되지 않은 내용이 더 어려운 복구 과제임을 보여줍니다. 이는 LLM이 텍스트의 생성 과정을 기반으로 한 후분포를 캡처할 수 있으며, Bayesian 최적 에이전트와 비교하여 모델의 행동을 평가하고 해석하는 데 효과적일 수 있음을 나타냅니다.



### Improving Audio Codec-based Zero-Shot Text-to-Speech Synthesis with Multi-Modal Context and Large Language Mod (https://arxiv.org/abs/2406.03706)
Comments:
          Accepted by Interspeech 2024

- **What's New**: 이번 연구에서는 오디오 코덱 기반의 새로운 TTS 모델을 소개합니다. 특히, 오디오북 및 대화형 TTS 시나리오에서 요구되는 긴 문맥 정보를 활용할 수 있도록 여러 가지 개선을 도입했습니다. Qformer의 성공에 영감을 받아, 멀티 모달 문맥-강화된 Qformer (MMCE-Qformer)를 통해 추가적인 멀티 모달 문맥 정보를 활용했습니다. 또한, 사전 학습된 대형 언어 모델 (LLM)을 적용하여 의미 토큰을 예측하고, SoundStorm을 사용해 음향 토큰을 생성함으로써 오디오 품질과 화자 유사성을 높였습니다.

- **Technical Details**: 모델은 크게 네 가지 주요 구성요소로 이루어져 있습니다: 통합된 오디오 코덱 모델을 통해 의미 및 음향 토큰을 추출, 멀티 모달 문맥-강화된 Qformer 인코더 (MMCE-Qformer)로 멀티 모달 문맥 정보를 활용해 입력 텍스트 특징을 강화, 사전 학습된 LLM을 사용하는 텍스트-의미 자율회귀 모델, 그리고 SoundStorm 모델입니다. MMCE-Qformer는 긴 문맥 정보를 포함한 멀티 모달 문맥과 입력 텍스트로부터 텍스트와 오디오 Qformer 임베딩을 추출합니다.

- **Performance Highlights**: 제안된 방법은 다양한 문맥 TTS 시나리오에서 각종 기준에서 기존 오디오 코덱 기반 제로샷 TTS 모델을 능가하는 성과를 보였습니다. 주관적 및 객관적 평가를 통해 자연스러움, 억양, 그리고 화자 유사성 면에서 우수한 성능을 입증하였으며, 여러 상관 실험을 통해 모델의 효과를 검증했습니다.



### Style Mixture of Experts for Expressive Text-To-Speech Synthesis (https://arxiv.org/abs/2406.03637)
- **What's New**: 최근의 스타일 전이 음성합성(TTS)은 합성 음성의 표현력을 크게 향상시켰으나, 다양한 레퍼런스 음성에서 스타일리틱 정보를 인코딩하는 것은 여전히 도전 과제입니다. 이를 해결하기 위해 StyleMoE를 제안합니다. StyleMoE는 스타일 인코더를 Mixture of Experts(MoE) 레이어로 대체하여 스타일 공간을 다루는 스타일 전문가들을 통해 스타일을 분할합니다. 게이팅 네트워크를 사용하여 각 스타일 전문가가 최적화하는 동안 스타일 공간의 특정 부분에 전문화를 구현합니다.

- **Technical Details**: StyleMoE는 기존 TTS 시스템의 스타일 인코더를 sparse Mixture of Experts(MoE) 레이어로 대체합니다. 각 전문가(E_i)는 동일한 인코더 구조를 가지지만 각기 다른 파라미터를 유지합니다. 게이팅 네트워크(G)는 입력 x에 기반하여 각 전문가의 기여도를 결정하는데, noisy top-k 게이팅 전략을 사용하여 희소성을 유지하면서 성능 저하를 방지합니다. 이를 GenerSpeech 프레임워크에 적용해 로컬 스타일 인코더들에 MoE 레이어를 통합했습니다.

- **Performance Highlights**: 제안한 StyleMoE 방법은 다양한 스타일 및 새로운 스타일의 레퍼런스 음성에 대해 스타일 인코더의 커버리지를 넓힐 뿐만 아니라, 기존 최첨단 스타일 전이 TTS 시스템의 성능도 향상시킵니다. 객관적 및 주관적 실험을 통해 본 방법의 효과를 입증했습니다. 결과적으로, StyleMoE는 다양한 스타일 공간을 더 잘 모델링하는데 기여하며, 이는 새로운 스타일을 포함한 레퍼런스 음성에서 더욱 표현력 있는 합성 음성을 생성할 수 있게 합니다.



### Advancing Anomaly Detection: Non-Semantic Financial Data Encoding with LLMs (https://arxiv.org/abs/2406.03614)
- **What's New**: 일반원장 데이터의 이상 감지는 금융 기록의 신뢰성을 보장하는 데 매우 중요합니다. 이 논문에서는 금융 데이터의 이상 감지를 위한 새로운 접근법으로 Large Language Models (LLMs) 임베딩을 사용합니다. LLM 임베딩을 통해 비의미적 범주형 데이터를 인코딩하고, 이를 바탕으로 이상 탐지 성능을 향상시키는 방법을 제시합니다.

- **Technical Details**: 현실 세계의 금융 기록으로부터 비의미적 범주형 데이터를 인코딩하는 데 3개의 사전 훈련된 일반 목적 문장-변환기(sentence-transformer) 모델을 테스트했습니다. 다운스트림 분류 작업을 위해서 로지스틱 회귀(Logistic Regression), 랜덤 포레스트(Random Forest), 그라디언트 부스팅 머신(Gradient Boosting Machines), 서포트 벡터 머신(Support Vector Machines), 뉴럴 네트워크(Neural Networks) 등 5개의 최적화된 머신 러닝 모델을 구현 및 평가했습니다.

- **Performance Highlights**: 실험 결과, LLMs가 이상 탐지에 유용한 정보를 제공하며, 특정 설정에서는 기준 모델(baselines)보다 성능이 우수함을 확인했습니다. 특히, LLM 임베딩을 활용하여 특징 희소성(feature sparsity)을 극복함으로써 금융 계정 항목의 이상 탐지 성능을 크게 향상시킬 수 있었습니다.




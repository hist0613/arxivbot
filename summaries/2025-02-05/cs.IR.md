New uploads on arXiv(cs.CL)

### Spatio-temporal transformer to support automatic sign language translation (https://arxiv.org/abs/2502.02587)
- **What's New**: 본 논문은 Sign Language Translation(SLT) 시스템을 위한 새로운 Transformer 기반 아키텍처를 제안합니다. 이 시스템은 다양한 수화(sign) 변형과 복잡한 언어적 특성을 여전히 처리할 수 있는 능력을 갖추고 있습니다. 특히, spatio-temporal motion gestures를 인코딩하여 지역적(local) 및 장거리(long-range) 공간 정보를 효과적으로 보존합니다.

- **Technical Details**: 제안된 접근법은 optical-flow 이미지를 기반으로 한 입력 표현을 통해 motion kinematic 패턴을 강조합니다. 또한, 2D positional encodings을 적용하여 2D 데이터에서 공간 정보를 보다 잘 처리하도록 했습니다. 2D self-attention 메커니즘이 도입되어 장거리 의존성을 효과적으로 강조하는 반면, convolution과 상호 보완적으로 작용합니다.

- **Performance Highlights**: 이 아키텍처는 Colombian Sign Language Translation Dataset (CoL-SLTD)와 RWTH-PHOENIX-Weather-2014T (PHOENIX14T) 데이터셋에서 검증되었습니다. CoL-SLTD에서 BLEU4 점수 46.84%를 달성하였으며, PHOENIX14T에서도 30.77%를 기록하였습니다. 이는 제안된 접근법의 실용성과 효과성을 보여줍니다.



### A comparison of translation performance between DeepL and Supertex (https://arxiv.org/abs/2502.02577)
- **What's New**: 이 연구는 깊은 언어 모델(LLMs)에 기반한 기계 번역(MT) 시스템의 품질을 평가하는 새로운 방법론을 제시합니다. DeepL과 Supertext라는 두 상업적 MT 시스템을 비교하여, 문서 전체에 대한 문맥을 고려한 번역 품질을 평가했습니다. 결과적으로 문서 수준의 분석에서는 Supertext가 더 일관된 번역을 제공하는 경향이 나타났습니다.

- **Technical Details**: 이 연구에서는 원문을 절단하지 않고 전체 텍스트를 번역한 후, 전문 번역가가 문서 전체 문맥을 고려하여 세그먼트를 평가했습니다. 8명의 전문 번역가들이 참여하여 A/B 테스트를 통해 두 시스템의 출력을 비교했습니다. 번역 품질 평가는 세그먼트 수준과 문서 수준 모두에서 수행되었습니다.

- **Performance Highlights**: 세그먼트 수준에서는 두 시스템에 대해 큰 선호도가 없었으나, 문서 수준에서는 Supertext가 4개의 언어 방향 중 3개에서 선호되는 결과를 보였습니다. 이는 LLMs의 사용이 단기 관점에서의 우수성을 넘어 장기적인 번역 품질에도 영향을 미친다는 것을 시사합니다. 이 연구는 MT 품질 평가를 보다 문맥 민감하게 진행할 필요성을 강조하고 있습니다.



### Are Language Models Up to Sequential Optimization Problems? From Evaluation to a Hegelian-Inspired Enhancemen (https://arxiv.org/abs/2502.02573)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)가 Sequential Optimization Problems (SOPs)을 처리하는 능력을 탐구합니다. 새로운 동적 프레임워크인 WorldGen을 소개하여 LLM 성능을 평가하기 위한 간단하면서도 효과적인 방법을 제시합니다. 초기 관찰에 따르면, LLM은 간단한 SOP에서는 잘 작동하지만 복잡성이 증가함에 따라 성능이 크게 저하되는 것으로 나타났습니다. 이러한 문제를 해결하기 위해 Hegelian Dialectics에서 영감을 받아 ACE라는 새로운 접근 방식을 제안합니다.

- **Technical Details**: WorldGen은 복잡성을 제어할 수 있는 새로운 SOP를 생성하는 능력을 갖춘 프레임워크입니다. 이 프레임워크는 기존의 정적 벤치마크와 달리 LLM의 발전에 맞춰 평가 복잡성을 증가시킬 수 있도록 설계되었습니다. 또한, 이 논문에서는 LLM의 성능 저하 문제를 해결하기 위해 LLM을 블랙 박스처럼 다루며, 추가적인 재교육이나 미세 조정 없이 성능을 향상시키는 방법을 제안합니다. Hegelian Dialectics의 구조적 접근법을 통해 SOP의 문제를 다루는 LLM의 가능성을 제시합니다.

- **Performance Highlights**: 실험 결과, 단순한 최적화 문제에서는 현재 LLM이 효율적으로 문제를 해결할 수 있음을 확인했습니다. 그러나 최적화 문제의 복잡성이 증가함에 따라 LLM의 성능이 만족스럽지 않게 저하되는 경향이 있음을 관찰했습니다. ACE 프레임워크를 통해 이러한 성능을 크게 향상시킬 수 있었으며, 이는 기존의 LLM들이 SOP에서 더욱 효율적으로 작업할 수 있게 해줍니다. 이 논문은 LLM의 향후 발전에 있어 철학적 접근이 중요한 역할을 할 수 있음을 강조합니다.



### Adaptive Self-improvement LLM Agentic System for ML Library Developmen (https://arxiv.org/abs/2502.02534)
- **What's New**: 이 논문은 ASPL(Architecture-Specific Programming Languages)을 사용하여 ML(기계 학습) 라이브러리를 생성하는데 어려움을 겪는 문제를 해결하기 위해 적응형 자기 개선 에이전트 시스템을 도입합니다. 연구자들은 대규모 언어 모델(LLM)의 코딩 기능을 활용하여 ML 라이브러리의 성능을 향상시키려 시도하고 있으며, 이 방법은 비전문가도 고급 ML 연산자를 단순한 데이터로 생성할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 본 연구는 STeP(Streaming Tensor Programs)이라는 새로운 ASPL을 대상으로 하여 ML 라이브러리를 생성합니다. STeP는 재구성 가능한 데이터 흐름 아키텍처를 위해 설계되었으며, ASPL의 발전을 반영하여 ML 라이브러리 개발의 복잡한 과제를 해결하는데 필요한 고급 기술이 필요합니다. 또한, LLM 에이전트는 증강 학습을 통해 자가 개선 사이클을 통해 진화하여 고품질 ML 연산자를 생성합니다.

- **Performance Highlights**: 연구 결과, 제안된 시스템은 기준 LLM 대비 최대로 3.9배 성능 향상을 기록하였고, 벤치마크의 96% 이상 작업을 해결하는 데 성공했습니다. 이는 ASPL의 한계로 인해 발생하는 코드 예제가 부족한 문제를 극복하며, ML 라이브러리 개발에 있어 더 자동화된 솔루션의 필요성을 강조합니다.



### Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search (https://arxiv.org/abs/2502.02508)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 추론 능력을 증대시키기 위해 검색 기능을 내부화하려는 새로운 접근 방식을 제안합니다. 기존의 방법들은 외부 LLM 검증기를 활용하는 반면, 본 연구에서는 단일 LLM이 복잡한 작업을 처리할 수 있게 하는 방법을 모색합니다. 특히 Chain-of-Action-Thought (COAT) 메커니즘과 두 단계의 훈련 패러다임을 적용하여, 자율 검색 기능을 강화하고자 합니다.

- **Technical Details**: Satori라는 7B LLM은 COAT 추론 형식을 내재화하는 소규모 포맷 조정 단계와 강화 학습 기반의 대규모 자가 개선 단계로 구성된 두 단계의 훈련을 통해 개발됩니다. 이 모델은 개방형 소스 데이터로 훈련되어 있고, 수학적 추론 과제에서 최첨단 성능을 기록하며, 도메인 외 작업에 대한 강력한 일반화 능력을 보입니다. 코드는 완전하게 오픈 소스 형식으로 제공될 예정입니다.

- **Performance Highlights**: Satori는 수학적 추론 작업에 대해 우수한 성능을 보이며, 동일한 기본 모델로 구축된 지침 모델보다 더 높은 성능을 기록했습니다. 또한, Satori는 외부 지침 없이 자율 검색이 가능한 단일 LLM이라는 점에서 효율성을 가지고 있습니다. 연구에서 강조하듯이, 이 모델은 자가 반영 및 자가 탐색 능력이 뛰어난 전반적인 역량을 지니고 있습니다.



### Multilingual Machine Translation with Open Large Language Models at Practical Scale: An Empirical Study (https://arxiv.org/abs/2502.02481)
Comments:
          Accept to NAACL2025 Main Conference

- **What's New**: 이 논문은 10억 개 미만의 파라미터를 가진 오픈 대형 언어 모델(LLM)이 다국어 기계 번역(MT) 작업을 처리하는 능력을 체계적으로 조사한 새로운 연구입니다. 특히, Gemma2-9B 모델이 뛰어난 다국어 번역 성능을 보이며, 이를 위한 Parallel-First Monolingual-Second(PFMS) 데이터 믹싱 전략을 도입했습니다. 결과적으로, GemmaX2-28 모델은 28개 언어에서 최상위 성능을 달성하고 Google Translate 및 GPT-4-turbo와 경쟁력을 갖추었습니다.

- **Technical Details**: 논문에서는 Mistral-7B-v0.3, Qwen2/2.5-7B, LLaMA3/3.1-8B 및 Gemma2-9B 등 최신 오픈 소스 LLM의 성능을 평가했습니다. PFMS 데이터 믹싱 전략을 통해 Gemma2-9B 모델을 꾸준히 재교육하여, 많은 언어 쌍을 처리할 수 있는 GemmaX2-28-9B 모델을 구축하였습니다. 이는 고품질 번역 쌍에 대한 작은 집합에서의 지침 파인튜닝과 결합하여 다국어 MT 성능을 크게 향상시켰습니다.

- **Performance Highlights**: 다국어 MT 성능을 평가한 결과, GemmaX2-28-9B는 28개 언어에서 뛰어난 번역 성능을 보였습니다. 또한, 이 모델은 TowerInstruct 및 XALMA와 같은 최신 모델을 지속적으로 능가하며, Google Translate 및 GPT-4-turbo와도 경쟁하는 성능을 나타냈습니다. 이 연구는 다각적인 언어 지원을 위한 모범 사례와 오픈 LLM의 최신 발전 사항에 대한 통찰력을 제공합니다.



### SAISA: Towards Multimodal Large Language Models with Both Training and Inference Efficiency (https://arxiv.org/abs/2502.02458)
- **What's New**: 최근 연구에서 제안된 NAAViT (No Attention Among Visual Tokens)와 SAISA (Self-Attention Input Space Alignment)는 두 가지 서로 다른 멀티모달 대형 언어 모델 아키텍처의 효율성을 개선하기 위한 새로운 접근법을 제공합니다. 기존 아키텍처들 간의 주목할 만한 차이점은 시각적 토큰 간의 상호작용 방식에 있으며, 이 연구는 시각적 토큰 간의 주목(attention) 적용을 최소화하여 훈련 및 추론 효율성을 높이는 방법을 제안합니다.

- **Technical Details**: 이 연구는 LLaVA-1.5 및 Flamingo와 같은 두 가지 아키텍처를 분석하여 시각적 및 텍스트 모달리티 간의 정렬 방식이 어떻게 훈련 및 추론 효율성에 영향을 미치는지를 밝혀냅니다. NAAViT은 시각적 토큰 간의 주목을 제거하여 계산 비용을 줄이며, SAISA는 이러한 NAAViT 블록에 시각적 특징을 직접 정렬하여 자기 주목과 피드포워드 네트워크(FFN)에서의 계산 오버헤드를 줄입니다.

- **Performance Highlights**: SAISA는 LLaVA-1.5와 동일한 설정을 사용하여 훈련 예산을 26% 줄이고 추론 FLOPs를 66% 감소시키면서도 성능이 우수하다는 결과를 보여줍니다. 포괄적인 절제 실험(ablative studies)을 통해 다양한 LLMs 및 시각 인코더에서 SAISA의 효율성과 효과성을 검증했습니다. 코드와 모델은 추후 공개될 예정입니다.



### Beyond English: Evaluating Automated Measurement of Moral Foundations in Non-English Discourse with a Chinese Case Study (https://arxiv.org/abs/2502.02451)
Comments:
          12 pages, 2 figures, 6 tables

- **What's New**: 이번 연구는 비영어 코퍼스에서 도덕적 기초(moral foundations, MFs)를 측정하기 위한 컴퓨터 기반 접근법을 탐구합니다. 대부분의 자원이 영어로 개발되었기 때문에 다국어 적용은 아직 제한적입니다. 이 논문은 중국어를 사례로 삼아 기계 번역(machine translation), 지역 언어 레키콘(local language lexicons), 다국어 모델(multilingual models), 대형 언어 모델(large language models, LLMs)을 활용한 MFs 측정의 효과성을 평가합니다.

- **Technical Details**: 본 연구는 MF 값 측정의 데이터 효율성을 특히 중점적으로 조사가 진행되었습니다. 연구 결과, 지역 언어 레키콘(Local lexicons)은 최적의 성능을 보이지 못하며, 기계 번역 접근법이 오히려 더 나은 성과를 보였습니다. 반면, 다국어 모델은 일정 수준의 성공을 거두었으나 데이터 효율성 면에서는 LLMs가 월등히 뛰어난 것으로 나타났습니다.

- **Performance Highlights**: LLMs는 정확성과 데이터 효율성 모두에서 다른 접근법을 초월하는 성능을 보였습니다. 영어 주석 데이터로 미세 조정(fine-tuning)하고 보강하면 비영어 코퍼스에서도 훌륭한 성능을 발휘할 수 있습니다. 그러나 이러한 성능은 각 MF 값에 따라 상이하게 나타나며, 문화적 뉘앙스를 간과할 가능성이 있다고 강조합니다.



### Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models (https://arxiv.org/abs/2502.02444)
- **What's New**: 본 논문에서는 Generative Psycho-Lexical Approach (GPLA)를 도입하여 심리학적으로 기초한 LLM(대규모 언어 모델) 가치 시스템을 구축하는 방법론을 제안합니다. 기존의 연구들이 주로 인간을 위한 가치 시스템인 Schwartz의 가치 이론을 바탕으로 하였던 반면, GPLA는 LLM의 고유한 심리를 반영한 새로운 접근 방식을 제공합니다. 이 방법론은 심리적 원칙에 기반한 다섯 가지 요소의 가치 시스템을 통해 LLM의 본질적 가치를 평가하고 정렬하는 데 도움을 줄 것입니다.

- **Technical Details**: GPLA는 LLMS의 텍스트에서 인식된 데이터를 추출하고, 이로부터 가치를 찾아내며, 최종적으로 값 시스템을 구조화하는 다섯 단계의 과정을 포함합니다. 이러한 방식은 기존의 수작업으로 이루어진 가치 사전 작성 방식과 결합하여, 보다 효율적으로 LLM의 상태를 반영하는 가치를 구분할 수 있도록 합니다. 또한, 자동화된 비반응적 가치 측정을 통해 전통적인 기법의 편향성을 줄이고 유연성을 높였습니다.

- **Performance Highlights**: 논문은 세 가지 벤치마크 작업을 통해 GPLA의 유효성을 검증하고 있습니다. 이러한 작업들은 구조적 유효성을 평가하기 위한 Confirmatory Factor Analysis, LLM의 안전성을 예측하기 위한 LLM Safety Prediction 및 LLM의 가치 정렬을 평가하는 LLM Value Alignment를 포함합니다. 결과적으로 이 제안된 가치 시스템은 기존의 Schwartz 가치 시스템보다 더 나은 성과를 보여주었으며, LLM의 안전성 예측 및 정렬 향상에 기여할 수 있음이 확인되었습니다.



### Activation-Informed Merging of Large Language Models (https://arxiv.org/abs/2502.02421)
- **What's New**: 이번 논문에서는 Activation-Informed Merging (AIM)이라는 새로운 기술을 소개합니다. AIM은 여러 개의 fine-tuned large language models (LLMs)의 파라미터와 embedding을 통합하는 모델 병합 방식으로, 활성화 공간(activation space) 정보를 활용하여 성능과 견고성을 향상시키는 데 중점을 두고 있습니다. AIM은 기존의 병합 방법에 적용 가능한 유연하고 보완적인 솔루션으로, 지속적 학습(continual learning, CL) 및 모델 압축의 원리를 바탕으로 설계되었습니다.

- **Technical Details**: AIM은 병합 과정에서 활성화 공간 정보를 통합하여 필수적인 가중치(weights)를 선택적으로 우선시합니다. 이는 모델의 기본 성능을 유지하면서도 새롭게 fine-tuned된 모델의 지식을 통합하는 데 도움을 줍니다. AIM의 핵심은 활성화 정보를 통해 가장 영향력 있는 가중치가 최소한의 변화만 겪도록 업데이트 단계(modification step)를 수정하는 것입니다.

- **Performance Highlights**: 실험 결과 AIM은 기존의 병합 방법들과 함께 사용할 수 있는 보완적인 솔루션으로, 여러 벤치마크에서 성능을 최대 40%까지 향상시킵니다. AIM은 구조가 단순함에도 불구하고 활성화 공간 정보의 중요성을 강조하며, LLM 병합 전략에 실질적인 발전을 가져올 수 있는 가능성을 보여줍니다.



### FewTopNER: Integrating Few-Shot Learning with Topic Modeling and Named Entity Recognition in a Multilingual Framework (https://arxiv.org/abs/2502.02391)
Comments:
          Code source : this https URL

- **What's New**: FewTopNER는 저자원이던 언어 환경에서의 named entity recognition(NER)과 주제 인식을 통합한 혁신적인 프레임워크입니다. XLM-RoBERTa를 기반으로 한 다국어 인코더와 언어 특화 조정 메커니즘을 활용하여 강력한 문맥 임베딩을 생성합니다. 이 아키텍처는 BiLSTM과 Conditional Random Fields를 사용한 프로토타입 기반의 NER 모듈과 하이브리드 확률적 방법을 통해 문서 수준의 주제를 추출하는 주제 모델링 모듈로 구성되어 있습니다.

- **Technical Details**: FewTopNER는 엔티티 인식과 주제 모델링 간의 동적 양방향 주의 망을 통해 정보를 융합하는 Cross-Task Attention Module을 포함하고 있습니다. 이를 통해 전역 의미 맥락을 엔티티 표현에 추가하고 주제 일관성을 개선합니다. 또한, Model-Agnostic Meta-Learning (MAML)을 통합하여 적은 데이터로 빠른 미세 조정을 가능하게 하며, Active Learning Interface를 통해 불확실한 사례를 겨냥하여 모델 예측을 반복적으로 개선합니다.

- **Performance Highlights**: FewTopNER는 영어, 프랑스어, 스페인어, 독일어, 이탈리아어를 포함한 다국어 벤치마크에서 기존의 최첨단 few-shot NER 모델을 유의미하게 초과하는 성능을 보여줍니다. 특히, F1 점수에서 2.5-4.0 포인트의 개선을 달성했으며, 정규화된 점수 기반의 상호정보량을 통해 주제 일관성이 향상되었습니다. 샘플과 메커니즘에 대한 분석 연구는 공유 인코더 및 과제 간 통합이 전체 성능에 미치는 중요한 기여를 입증하고 있습니다.



### CoAT: Chain-of-Associated-Thoughts Framework for Enhancing Large Language Models Reasoning (https://arxiv.org/abs/2502.02390)
- **What's New**: 최근 LLM(대형 언어 모델) 기술이 급격히 발전하면서, 기존의 '빠른 사고' 접근 방식에서 '느린 사고' 기법으로 전환하는 경향이 증대되고 있습니다. 이러한 '느린 사고'는 인간의 사고 과정과 유사하게 새로운 정보를 통합하고 지식을 갱신하는 능력을 반영하고 있습니다. 본 논문에서는 CoAT(Chain-of-Associated-Thoughts) 프레임워크를 제안하며, 이는 MCTS(Monte Carlo Tree Search) 알고리즘과 동적 연상 기억 메커니즘을 결합하여 LLM의 추론 능력을 혁신적으로 확장하고 있습니다.

- **Technical Details**: CoAT 프레임워크는 인간의 연상 능력을 모방하여 LLM이 실시간으로 관련 정보를 검색하고 자기 증강을 수행할 수 있도록 설계되었습니다. MCTS 알고리즘의 최적화된 라우팅 전략을 사용하여, 각 연상 기억의 추가가 후속 콘텐츠 생성을 위한 중요한 정보를 제공하도록 보장합니다. 이 구조적 탐색과 적응 학습의 시너지를 통해 CoAT는 기존 LLM의 한계를 극복하고 의미 통일성을 유지하면서 추론 범위를 확장합니다.

- **Performance Highlights**: 광범위한 생성 및 추론 작업을 통한 실험 결과, CoAT 프레임워크는 기존의 추론 프로세스에 비해 정확성, 일관성 및 다양성 측면에서 우수한 성능을 보였습니다. 이는 CoAT가 이전 추론을 반복적으로 수정하고 진화하는 정보를 통합함으로써 제공하는 정밀하고 포괄적인 결과를 반영합니다. 본 연구의 결과는 LLM 프레임워크의 혁신을 가져오며, 향후 복잡한 추론 작업을 효과적으로 해결할 수 있는 기초를 제공합니다.



### STAIR: Improving Safety Alignment with Introspective Reasoning (https://arxiv.org/abs/2502.02384)
Comments:
          22 pages, 8 figures

- **What's New**: 이 논문에서는 Large Language Models (LLMs)의 안전성 확보를 위한 새로운 프레임워크인 STAIR를 소개합니다. STAIR는 SafeTy Alignment와 Itrospective Reasoning을 통합하여 LLM이 안전 위험을 단계별 분석을 통해 인식할 수 있도록 지원합니다. 기존의 안전 정렬 방법들이 성능과 안전성 간의 균형 문제를 겪는 것에서 탈피하고자 합니다.

- **Technical Details**: STAIR는 모델에 구조화된 추론(Structured Reasoning) 능력을 부여하며, Safety-Informed Monte Carlo Tree Search (SI-MCTS)를 이용하여 단계별 추론 데이터에 대한 선호 최적화를 반복적으로 진행하여 안전 정렬을 진전시킵니다. 이를 통해 자가 개선된 Chain-of-thought (CoT) 추론을 통해 안전성 인식을 강화합니다. 또한, 테스트 시 검색 개선을 위한 프로세스 보상 모델을 훈련합니다.

- **Performance Highlights**: 광범위한 실험 결과, STAIR는 본능적인 정렬 방법들과 비교하여 유해한 출력 감소를 효과적으로 달성하면서도 유용성(helpfulness)을 더 잘 유지합니다. STAIR는 인기 있는 jailbreak 공격에 대한 Claude-3.5와 유사한 안전 성능을 달성하며, 이는 테스트 시간에서의 스케일링을 통해 가능해집니다.



### Premise-Augmented Reasoning Chains Improve Error Identification in Math reasoning with LLMs (https://arxiv.org/abs/2502.02362)
- **What's New**: 이 논문에서는 Chain-of-Thought (CoT) 프롬프트가 대형 언어 모델(LLMs)의 수학적 추론을 개선하는 방법을 다룹니다. 특히 추론 단계의 검증을 용이하게 하기 위해 각 단계의 전제를 식별하는 새로운 프레임워크를 제시합니다. 기존의 직선적인 추론 체인을 Premise Augmented Reasoning Chains (PARC)로 변환하여 전제 링크를 포함하는 방법을 소개하고 있습니다.

- **Technical Details**: 이 연구는 추론 체인의 각 단계에 대해 전제를 식별하는 시스템을 구축하였습니다. PARC는 유도 비순환 그래프(directed acyclic graph) 구조로, 노드는 각 추론 단계이며 엣지는 전제 링크를 나타냅니다. PERL (Premises and ERrors identification in LLMs)이라는 데이터셋을 통해 LLM이 복잡한 추론 체인에서 전제를 신뢰성 있게 식별할 수 있음을 보였습니다.

- **Performance Highlights**: 실험 결과, 오픈소스 LLM도 전제 식별에서 90%의 재현율(recall)을 기록했습니다. 또한 PARC를 사용하여 단계별 검증을 수행함으로써 오류 식별의 정확도가 6%에서 16% 개선되었습니다. 이러한 결과는 복잡한 문제 해결 작업에 있어 전제 중심 표현의 유용성을 강조하며, LLM 기반 추론 평가의 신뢰성을 개선할 새로운 방향성을 제시합니다.



### Boosting Multimodal Reasoning with MCTS-Automated Structured Thinking (https://arxiv.org/abs/2502.02339)
- **What's New**: 본 논문에서는 복합 비주얼 추론을 위한 새로운 자동화된 구조적 사고 패러다임인 AStar를 제안합니다. MCTS(Monte Carlo Tree Search)를 활용하여 제한된 데이터에서 고수준의 인지 추론 패턴을 자동으로 도출할 수 있는 계층적 구조를 형성합니다. 이러한 접근법은 MLLMs의 기존 한계를 극복하고, 성능과 효율성 사이에 훌륭한 균형을 이루는 것을 목표로 합니다.

- **Technical Details**: AStar 방법론은 세 가지 단계로 이루어져 있습니다: (1) 비주얼 추론 행동 정의, (2) MCTS 기반의 사고 카드 구성, (3) 적응형 추론 및 검증입니다. 비주얼 추론 행동은 유사 인간 사고 행동을 모사하는 여섯 가지 원자적 행동으로 정의되며, MCTS를 사용하여 참조 추론 패턴을 생성하는 사고 카드를 구축합니다. 마지막으로, 문제의 복잡성에 맞추어 최적의 사고 카드를 선택하고, 검증 과정을 거칩니다.

- **Performance Highlights**: 실험 결과 AStar는 MathVerse 벤치마크에서 54.0%의 정확도를 기록하며, GPT-4o의 50.2%를 초월하는 뛰어난 성과를 보여주었습니다. 또한, 7B 백본을 기반으로 하여 데이터와 계산 효율성을 대폭 개선하였으며, 이는 6.4배의 추론 오버헤드를 감소시키는 결과를 낳았습니다. 이러한 성과는 MLLMs의 내부 추론 능력과 외부 지침을 효과적으로 통합한 결과로 평가됩니다.



### Evalita-LLM: Benchmarking Large Language Models on Italian (https://arxiv.org/abs/2502.02289)
Comments:
          42 pages, 1 figure, 32 tables

- **What's New**: Evalita-LLM은 이탈리어 작업에 대한 대규모 언어 모델(Large Language Models, LLM)을 평가하기 위해 설계된 새로운 벤치마크입니다. 이 벤치마크의 독창적 특징 중 하나는 모든 작업이 이탈리아어로 원래 만들어졌다는 것입니다. 이러한 접근은 이탈리아어에서의 번역 문제가 및 문화적 편향을 피하는 데 도움을 줍니다. 또한, 기존의 객관식 작업 외에도 생성 작업을 포함하여 LLM과의 보다 자연스러운 상호작용을 가능하게 합니다.

- **Technical Details**: Evalita-LLM은 이탈리아 컴퓨터 언어학 협회(Italian Association for Computational Linguistics, AILC)의 지원을 받아 기존 데이터셋을 활용하여 작성되었습니다. 약 70개의 데이터셋이 다양한 언어학적 현상을 다루며, 이 중 약 35개는 유럽 언어 그리드 플랫폼(European Language Grid, ELG)을 통해 개방형 라이센스로 제공됩니다. 이를 통해 Evalita-LLM은 연구자들이 손쉽게 사용할 수 있는 플랫폼이 됩니다.

- **Performance Highlights**: Evalita-LLM의 초기 버전에는 다양한 언어적 현상과 장르가 포함된 10개의 작업이 포함되어 있으며, 평가 과정에서 최고의 성능을 발휘한 여러 최신 LLM의 통계도 제공됩니다. 이러한 작업들은 이탈리아어에 특화되어 있으며, LLM의 성능 평가에 공정하고 객관적인 기준을 제시하는 것을 목표로 합니다. 전체적으로 Evalita-LLM은 이탈리아어 LLM 평가의 중요한 기준으로 자리 매김할 예정입니다.



### Conversation AI Dialog for Medicare powered by Finetuning and Retrieval Augmented Generation (https://arxiv.org/abs/2502.02249)
Comments:
          12 pages

- **What's New**: 이 연구는 의사-환자 대화의 맥락에서 두 가지 주요 기법인 LoRA (Low-Rank Adaptation)로서의 파인 튜닝과 RAG (Retrieval-Augmented Generation) 프레임워크의 비교 분석을 진행합니다. 다양한 의료 분야의 여러 데이터셋을 활용하여 진행된 이 연구는 기존의 방법론에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 이 분석은 세 가지 최첨단 모델인 Llama-2, GPT, LSTM 모델을 포함하여, 실제 의사-환자 대화를 통해 수행되었습니다. 평가 지표로는 언어 품질(perplexity, BLEU score), 사실 정확성(fact-checking), 의료 지침 준수, 인간의 판단(coherence, empathy, safety) 등을 사용하여 모델 성능을 종합적으로 평가했습니다.

- **Performance Highlights**: 연구 결과는 각 접근법의 강점과 한계를 제공하며, 의료 애플리케이션에 적합성을 조명합니다. 또한, 다양한 환자 질문에 대한 모델의 견고성 조사 및 도메인 특화 지식 통합의 영향을 탐구하여, 적절한 데이터 증강 및 검색 전략을 통해 LLM의 성능을 향상시킬 수 있는 가능성을 강조합니다.



### When Dimensionality Hurts: The Role of LLM Embedding Compression for Noisy Regression Tasks (https://arxiv.org/abs/2502.02199)
- **What's New**: 이 연구는 대형 언어 모델(LLM)에서 텍스트 표현의 압축된 표현이 회귀(task) 작업에서 더 나은 성능을 발휘할 수 있음을 보여줍니다. 특히 금융 수익 예측, 작문 품질 평가, 리뷰 점수 매기기와 같은 다양한 신호 대 잡음(signal-to-noise) 환경에서의 임베딩 압축의 상대적인 성능을 비교하고, 압축 기법이 과적합(overfitting)을 완화하고 성능을 향상시킬 수 있음을 확립했습니다.

- **Technical Details**: 저자들은 autoencoder의 은닉 표현을 사용하여 수행된 최소 감독 방식의 압축을 통해 낮은 신호 대 잡음 비율의 회귀 작업에서 성능 향상을 보여줍니다. 연구는 LLM의 숨겨진 차원의 크기가 768에서 16384로 증가했지만, 특정 작업에서는 차원의 최적화 및 압축이 필요하다는 점을 강조합니다. 데이터 세트에는 CNN 기사 및 주식 시장 데이터가 포함되어 이들 간의 관계를 정밀하게 분석합니다.

- **Performance Highlights**: 연구 결과, 압축된 표현이 특정 금융 예측 작업에서 더 나은 성과를 보였으며, 이들 표현을 통해 진정한 feature 세트보다 규제(regularising) 효과가 더 크게 나타났습니다. 특히 금융 뉴스 관련된 주식 수익 예측 작업에서, 단순 감정 또는 주제 표현이 압축된 결과에 기여한 것으로 평가됩니다. 이 논문은 또한 다양한 데이터 세트를 통해 높은 인과 종속성이 있는 작업의 최적 차원성을 규명합니다.



### Mass-Editing Memory with Attention in Transformers: A cross-lingual exploration of knowledg (https://arxiv.org/abs/2502.02173)
- **What's New**: 최근 연구에서는 대형 언어 모델(LLM)의 사실적 지식을 업데이트하고 수정하는 방법이 탐구되었습니다. 이 연구는 기존의 지식 편집 방법이 여러 언어에서 얼마나 효과적인지를 검토하며, 특히 Attention Mechanism의 역할에 초점을 맞추고 있습니다. 이를 바탕으로 Mass-Editing Memory with Attention in Transformers (MEMAT)를 제안하며, 이는 매개변수 수정을 최소화하면서 모든 지표에서 значительные 개선을 이룬다는 점에서 혁신적입니다.

- **Technical Details**: 대형 언어 모델은 실제 의미 이해보다는 문장에서 토큰 발생 확률을 예측하도록 설계되어, 현실과 정확성이 결여된 콘텐츠 생성을 초래할 수 있습니다. 이 연구에서는 MEMIT이라는 지식 편집 방법을 분석하며, 영어와 카탈로니아어 간의 교차 언어 능력을 검토합니다. 메모리 내에서 지식 삽입 시, Attention Head의 역할은 모델의 신뢰성을 높이는데 기여하며, 내부 구조에서 사실적 연관성을 평가하는 중요한 정보 원천으로 작용합니다.

- **Performance Highlights**: MEMAT 방법은 모든 평가 지표에서 개선을 보여주며, 어떤 경우에는 10% 이상의 차이를 나타냅니다. 이 알고리즘은 기존 지식에 대한 이해를 증진시키고, 새로운 지식을 삽입하는 대신 효율적으로 작동함을 입증하였습니다. 추후 실험 결과는 MEMAT이 기존 모델에 비해 포터블하고 연산적으로 효율적이라는 것을 보여줍니다.



### Multilingual Attribute Extraction from News Web Pages (https://arxiv.org/abs/2502.02167)
- **What's New**: 이 논문은 다국어 뉴스 기사 웹 페이지에서 속성을 자동으로 추출하는 문제를 다룹니다. 최근 신경망 모델들이 반구조화된 웹 페이지에서의 정보 추출에 높은 효율성을 보였으나, 이들은 주로 영어 데이터로 사전 훈련되어 다른 언어 웹 페이지에의 적용이 복잡하다는 한계가 있었습니다. 저자들은 6개의 언어로 구성된 3,172개의 뉴스 웹 페이지로 이뤄진 다국어 데이터셋을 준비하였으며, 이를 통해 MarkupLM 모델을 미세 조정(fin-tune)하여 성능을 평가했습니다.

- **Technical Details**: 본 논문에서는 영어로 사전 훈련된 MarkupLM 모델과 다국어 데이터를 기반으로 다시 훈련한 DOM-LM 모델을 활용하여 뉴스 웹 페이지의 속성을 추출합니다. 각 모델은 HTML 문서 이해를 위한 고급 사전 훈련 모델로, MarkupLM은 24백만 개의 영어 웹 페이지에서 훈련되었으며, DOM-LM은 다양한 구조적 특성을 고려하여 훈련되었습니다. 이 두 모델의 성능 비교를 통해, 기존 오픈 소스 뉴스 데이터 추출 도구보다 뛰어난 속성 추출 메트릭을 달성했습니다.

- **Performance Highlights**: 모델 비교 결과, 다국어 DOM-LM은 뉴스 웹사이트 속성 추출에서 가장 높은 성능을 기록했습니다. 다양한 언어 그룹에서 테스트한 결과, 제안된 방법이 장애물 없이 새로운 웹사이트에서도 잘 작동함을 입증했습니다. 또한, 영어로 번역된 페이지의 질적 영향도 평가하였으며, 이러한 평가를 통해 다국어 속성 추출의 가능성을 확인했습니다.



### Topic Modeling in Marath (https://arxiv.org/abs/2502.02100)
- **What's New**: 이 논문은 인도 언어인 마라티어(Marathi)에 대한 토픽 모델링(topic modeling)의 여러 접근 방식을 탐구합니다. 기존 영어 기반 연구와는 달리, 인도 언어에 대한 자원의 부족과 다양성 때문에 제한적이었던 연구 현황을 살펴봅니다.

- **Technical Details**: 저자들은 BERT(bidirectional encoder representations from transformers)와 비-BERT(non-BERT) 접근 방식을 비교합니다. 특히, 다국어 및 단일 언어 BERT 모델을 사용하여, 토픽 일관성(topic coherence)과 토픽 다양성(topic diversity)이라는 평가 지표를 통해 성능을 분석하였습니다.

- **Performance Highlights**: 연구 결과, 인도 언어로 훈련된 BERT 모델과 결합한 BERTopic이 LDA(Latent Dirichlet Allocation)보다 마라티어 토픽 모델링 성능에서 더 우수한 것으로 나타났습니다. 이는 인도 언어에 대한 보다 효과적인 토픽 모델링 방법을 제시하는 중요한 발견입니다.



### LongDPO: Unlock Better Long-form Generation Abilities for LLMs via Critique-augmented Stepwise Information (https://arxiv.org/abs/2502.02095)
- **What's New**: 이 논문에서는 LongDPO라는 새로운 접근 방식을 제안하여 긴 형식의 생성(long-form generation)을 개선합니다. 기존의 outcome supervision 대신 process supervision을 도입하여 단계별(supervision) 학습을 가능하게 합니다. Monte Carlo Tree Search (MCTS) 기법을 사용하여 단계별 선호 데이터(preference data)를 수집하고, 이러한 데이터를 활용하여 보다 일관된 길이와 품질을 가진 콘텐츠 생성을 목표로 합니다.

- **Technical Details**: LongDPO는 두 가지 주요 구성 요소로 나뉘어 있습니다: 단계별 선호 데이터를 수집하는 것과 수집된 데이터를 활용하여 DPO(Direct Preference Optimization) 훈련을 수행하는 것입니다. MCTS는 선택, 확장, 평가 및 역전파의 네 가지 절차를 통해 단계별 선호 쌍을 수집하며, 이를 통해 모델의 일관성을 유지합니다. 또한 외부 비평(critique)을 통해 고품질의 선호 쌍을 모으기 위해 MCTS의 평가 단계에서 판단 모델을 사용합니다.

- **Performance Highlights**: 실험 결과 LongDPO는 긴 형식의 생성 벤치마크에서 길이와 품질을 개선하며, 일반 벤치마크에서도 거의 손실 없는 성능을 유지합니다. Llama 및 Qwen 기반의 모델 백본에서 기존 DPO 버전보다 우수한 성능을 보였으며, 일반 작업에 대해서도 향상된 결과를 보여주었습니다. 본 연구의 결과는 생성된 응답이 인간의 선호와 더욱 일치한다는 점에서 의의를 갖습니다.



### Rethinking stance detection: A theoretically-informed research agenda for user-level inference using language models (https://arxiv.org/abs/2502.02074)
- **What's New**: 본 논문에서는 스탠스(stance) 탐지의 이론적 개념화 부족과 메시지 수준이 아닌 개인 또는 사용자 수준에서의 스탠스 처리 문제를 강조합니다. 스탠스 탐지 모델에서 심리적 특성과 관련된 속성을 유용하게 통합하기 위한 개인 수준 구성으로서의 스탠스의 다학제적 기원을 살펴봅니다.

- **Technical Details**: 최근의 사전 학습(pre-trained)된 대형 언어 모델(Large Language Models, LLMs)이 사용자 수준의 속성을 유연하게 추론하고 스탠스 모델링에 통합할 수 있는 방법을 제공할 수 있다는 주장을 합니다. 이에 대한 통찰을 제공하기 위해 사용자 속성을 추론하는 데 LLMs를 사용하는 연구의 신흥 코퍼스를 간략히 검토하고 통합합니다.

- **Performance Highlights**: 이 논문은 이론적으로 기반을 둔, 포괄적이며 실용적인 스탠스 탐지 연구를 위한 4가지 아젠다를 제안하며, 스탠스 탐지의 연구 방향성을 제시합니다. 이 새로운 접근법이 자연어 처리(NLP)에서의 스탠스 탐지 모델의 진전을 어떻게 가능하게 할지를 제안하고 있습니다.



### ASCenD-BDS: Adaptable, Stochastic and Context-aware framework for Detection of Bias, Discrimination and Stereotyping (https://arxiv.org/abs/2502.02072)
Comments:
          17 pages, 6 Figures and this manuscript will be submitted to Q1,Q2 Journals

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 진화가 자연어 처리 분야에 변화를 가져온 것과 동시에, 편향(bias)과 차별(discrimination) 등의 문제에 대한 우려를 제기하고 있습니다. 특히, ASCenD BDS(Adaptable, Stochastic and Context-aware framework for Detection of Bias, Discrimination and Stereotyping)라는 새로운 프레임워크를 소개하여, 다양한 사회적 맥락에서 이러한 문제를 탐지할 수 있는 접근법을 제안합니다.

- **Technical Details**: ASCenD BDS는 성별, 계급, 나이, 장애, 사회경제적 상태, 언어적 변이 등 여러 카테고리를 통해 편향, 차별, 고정관념(stereotyping)을 탐지하는 방식입니다. 기존의 데이터셋에 의존하던 방식과는 달리, 이 프레임워크는 적응성(adaptability), 확률적(stochasticity), 맥락 인식(context awareness) 기능을 통해 보다 맞춤형으로 접근할 수 있는 장점을 가지고 있습니다. 특히, 인도의 문화적 맥락에 맞춘 내용을 통해 카테고리를 정립하고, 그에 맞는 데이터를 생성할 수 있도록 합니다.

- **Performance Highlights**: 이 프레임워크는 800개 이상의 STEM(Science, Technology, Engineering, Mathematics), 10개의 카테고리 및 31개의 고유 서브 카테고리를 개발하였으며, 이는 Saint Fox Consultancy Private Ltd의 컨설턴트 팀에 의해 수행되었습니다. 또한, SFCLabs에서 제품 개발의 일환으로 이 개념을 테스트하여 프레임워크의 실효성을 증명하였습니다. 이와 같은 성과는 향후 다양한 문화적 배경에서 편향을 탐지하고 해결하는 데 기여할 수 있을 것으로 기대됩니다.



### AmaSQuAD: A Benchmark for Amharic Extractive Question Answering (https://arxiv.org/abs/2502.02047)
- **What's New**: 이 연구는 저자들이 Amharic 언어로 번역한 SQuAD 2.0의 AmaSQuAD 데이터셋을 생성함으로써, 저자들이 저자들이 추출적 질문-응답(Question-Answering) 데이터셋을 저자들이 자원이 부족한 언어로 전환하는 새로운 프레임워크를 제시하고 있습니다. 특히 번역된 질문과 답변 간의 불일치와 여러 답변 인스턴스의 존재와 같은 문제를 다루고 있으며, 이 과정에서 BERT 기반 모델의 임베딩을 활용한 코사인 유사성과 가장 긴 공통 부분 수열(Longest Common Subsequence, LCS)을 사용했습니다.

- **Technical Details**: AmaSQuAD 데이터셋을 통해 Amharic 질문-응답 시스템을 훈련하기 위해 XLM-R 모델을 미세 조정(fine-tuning) 하는 방법론을 사용했습니다. 이 연구는 저자들이 기본 성능을 개선했음을 보여주며, 미세 조정된 모델이 AmaSQuAD 개발 데이터셋에서 F1 점수를 36.55%에서 44.41%로, 그리고 50.01%에서 57.5%로 증가시켰습니다. 또한, AmQA 데이터셋에서 인상적인 성과를 보여주며, F1 점수가 67.80%에서 68.80%로 상승했습니다.

- **Performance Highlights**: 미세 조정한 XLM-R 모델은 AmaSQuAD와 AmQA 데이터셋에서 성능 향상을 입증했습니다. AmaSQuAD 개발 데이터셋에서 F1 점수가 36.55%에서 44.41%로, AmQA 데이터셋에서 Exact Match 스코어가 52.50%에서 52.66%로 증가하였습니다. 이러한 결과들은 저자들이 저자들이 새로운 테크닉을 통해 Amharic 언어 기반 QA 시스템의 발전에 기여할 수 있음을 나타냅니다.



### Contextual Memory Reweaving in Large Language Models Using Layered Latent State Reconstruction (https://arxiv.org/abs/2502.02046)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)의 메모리 유지 문제를 해결하기 위한 새로운 접근법인 Contextual Memory Reweaving (CMR)을 제안합니다. CMR은 여러 처리 레이어에서 캡처한 잠재 상태를 재구성하여 긴 시퀀스에서도 토큰 표현을 강화하는 구조적인 방법입니다. 이를 통해 이전 컨텍스트에서의 정보를 더 잘 기억하고 활용할 수 있게 됩니다.

- **Technical Details**: CMR 프레임워크는 Layered Latent State Reconstruction (LLSR) 방법론을 포함하여 모델의 기존 잠재 표현을 사용하여 장기 의존성을 강화합니다. LLSR은 고차원 토큰 표현을 단계적으로 재구성하면서 모델 내에서 메모리를 효율적으로 관리합니다. 이러한 방식을 통해 외부 메모리 모듈을 도입하지 않고도 메모리 유지 능력을 개선할 수 있습니다.

- **Performance Highlights**: 실험 결과, CMR 프레임워크는 다양한 시퀀스 길이에서 회상 정확도를 향상시키고, 드물게 발생하는 토큰과 수치적 추론의 일관성을 유지하는 데 있어 뚜렷한 성과를 보였습니다. 이는 긴 텍스트 생성과 모호한 쿼리 해결에서 일관성과 불일치를 줄이는 데 효과적이었습니다. 또한 CMR을 통해 얻은 Attention weight 분포의 구조적 패턴은 향상된 맥락 인식에 기여하는 잠재 상태의 재구성과 관련이 있음을 제안합니다.



### M2R2: Mixture of Multi-Rate Residuals for Efficient Transformer Inferenc (https://arxiv.org/abs/2502.02040)
- **What's New**: 이 논문에서는 Mixture of Multi-rate Residuals (M2R2)라는 새로운 프레임워크를 소개하며, 이 프레임워크는 잔여 변환의 속도를 동적으로 조절하여 조기 정렬을 최적화합니다. 이를 통해 다양한 추론 패러다임에서 효율성을 향상시키고, 생성 품질과 속도 간의 최적의 균형을 달성할 수 있습니다. M2R2는 self-speculative decoding 방법을 초월하여 MT-Bench에서 최대 2.8배의 속도를 기록하는 하이라이트를 제공합니다.

- **Technical Details**: M2R2는 토큰 수준에서 잔여 변환 속도를 조절하면서 초기 단계를 빠르게 정렬할 수 있도록 설계되었습니다. 이 방법은 동적 컴퓨팅 환경과 Mixture-of-Experts (MoE) 아키텍처에서 성능을 개선할 수 있도록 고안되었습니다. 연구에서 실험을 통해 잔여 변환의 속도가 토큰의 표현을 더 빠르게 처리하는 데 어떻게 기여하는지를 보여줍니다.

- **Performance Highlights**: M2R2는 기존의 self-speculative decoding 방법과 비교하여 MT-Bench에서 2.8배의 속도 향상을 달성하였고, MoE 모델에서는 전문가 로딩과 컴퓨팅 연산을 동시에 수행하여 최대 2.9배의 속도 향상을 이루어냈습니다. 이는 자원이 제한된 환경에서도 효과적으로 작동할 수 있는 가능성을 보여줍니다.



### Fine-tuning Language Models for Recipe Generation: A Comparative Analysis and Benchmark Study (https://arxiv.org/abs/2502.02028)
Comments:
          15 pages, 8 figures

- **What's New**: 이 연구는 다양한 소형 언어 모델을 미세 조정(fine-tuning)하여 조리법 생성(task of recipe generation)을 탐구하고 개발하는 데 초점을 맞추고 있습니다. 특히 조리법 생성의 열린 과제에 대한 비교와 함께 강력한 평가 지표(evaluation metrics)를 개발하는 데 중점을 두었습니다. 본 연구는 T5-small, SmolLM-135M, Phi-2과 같은 여러 모델 아키텍처에 대한 광범위한 실험을 수행하였습니다.

- **Technical Details**: 연구에서는 Food.com 데이터셋을 활용하여 180,000개 이상의 조리법과 700,000개의 리뷰를 분석했습니다. 이를 통해 조리법 이름, 재료 목록, 조리법 지침, 영양 정보 등을 정형화하여 입력-출력 쌍을 생성하는 데이터 전처리 파이프라인을 구축하였습니다. 또한 알레르기 대체를 위한 RAG 및 프롬프트 기반 접근 방식을 개발하였으며, 조리법 생성의 품질을 평가하기 위해 새로운 조리법 특화 평가 지표를 포함한 다차원 평가 프레임워크를 제안하였습니다.

- **Performance Highlights**: SmolLM-360M과 SmolLM-1.7B는 크기 차이에도 불구하고 유사한 성능을 보였으나, Phi-2는 더 많은 파라미터에도 불구하고 조리법 생성에서 한계를 나타내었습니다. 실험 결과, 더 큰 모델들이 표준 지표에서 일반적으로 더 나은 성능을 보였지만, 도메인 특화 지표를 고려할 때 모델 크기와 조리법 품질 간의 관계는 더 복잡함을 보여주었습니다. 본 연구는 조리법 생성과 관련된 NLG(Natural Language Generation) 작업에서 도메인 전문성과 안전성의 중요성에 대한 통찰력을 제공합니다.



### Reasoning Bias of Next Token Prediction Training (https://arxiv.org/abs/2502.02007)
Comments:
          19 pages, 11 figures

- **What's New**: 이 연구에서는 Large Language Models (LLMs)에 대한 Critical Token Prediction (CTP)과 Next Token Prediction (NTP)의 성능 차이를 체계적으로 비교합니다. NTP가 CTP에 비해 여러 가지 벤치마크 데이터 세트에서 일반화 능력을 향상시키는 것을 보여주는 결과를 도출했습니다. 이러한 연구 결과는 NTP가 훈련 중 잡음에 노출되더라도 이유 판단 능력에서 더 우수함을 강조합니다. 이는 LLM 용 훈련 전략에 대한 이해를 심화시키는 데 중요한 기여를 합니다.

- **Technical Details**: NTP는 자가 지도 학습(self-supervised learning) 방식으로, 대량의 비표시 텍스트에서 다음 토큰을 예측하여 모델을 훈련합니다. 반면 CTP는 주어진 라벨에 대해서만 훈련하여 중요한 토큰을 예측합니다. 이 연구에서는 NTP로 훈련된 모델이 더 나은 일반화 성능을 보여주며, 이는 NTP가 높은 일반화 능력과 더불어 논리적 추론 능력을 갖추고 있음을 시사합니다. 마지막으로, NTP 훈련 속도는 CTP보다 느리지만, 훈련 결과는 더 풍부한 견고함을 보입니다.

- **Performance Highlights**: NTP는 다양한 벤치마크 데이터에서 쿼리에 대한 응답, 즉 Q&A 쌍에서도 더 나은 성능을 발휘하는 것으로 나타났습니다. 비록 CTP가 감독된 미세 조정(finetuning) 단계에서는 우수한 선택으로 간주될 수 있지만, 전반적으로 NTP는 엄청난 일반화와 변동성에 대한 저항성을 보여줍니다. 이는 NTP가 훈련 중에 잡음이 주요 특징으로 작용할 때, 모델의 손실 표면이 더욱 평평하게 형성되며, 훈련 후 성능 향상을 가져오기 때문입니다.



### Wavelet-based Positional Representation for Long Contex (https://arxiv.org/abs/2502.02004)
Comments:
          Accepted to ICLR 2025. 28 pages, 11 figures

- **What's New**: 이번 연구에서는 기존의 위치 인코딩 방법의 한계를 벗어나, 다중 스케일을 포착할 수 있는 새로운 방법을 제안합니다. 이 방법은 여러 단계의 윈도우 크기를 활용하여 전체 주의 메커니즘의 수용 영역을 제한하지 않고 위치 정보를 외삽할 수 있도록 설계되었습니다. 실험 결과, 이 새로운 방법이 기존의 방법들보다 짧거나 긴 문맥에서 모델 성능을 개선하는 데 기여하는 것을 보여줍니다.

- **Technical Details**: 기존의 Rotary Position Embedding (RoPE)은 고정된 스케일 파라미터를 사용하여 Haar 스타일의 웨이브릿 변환을 수행하였으나, 이는 비정상 신호의 미세한 움직임을 포착하는 데 제한적입니다. Attention with Linear Biases (ALiBi)는 각 토큰 간의 상대적인 위치를 통해 길이를 초과하는 시퀀스를 처리할 수 있지만, 수용 영역을 한정하여 깊은 종속성을 포착하는 데 한계를 지닙니다. 본 연구에서는 웨이브릿 변환을 통한 다중 윈도우 크기를 활용한 새로운 위치 표현 방법으로, 시간이 지나면서의 동적 변화를 포착하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 우리의 실험에서는 wikitext-103 데이터셋을 사용하여, 제안한 방법이 전통적인 위치 인코딩 방법보다 더 나은 성능을 발휘한다는 것을 보여주었습니다. 특히, 긴 문맥에서 RoPE에 비해 낮은 perplexity를 기록하며 Llama-2 모델과 CodeParrot 데이터셋에서도 확연한 성능 향상을 확인했습니다. 이는 모델의 주의 메커니즘을 통해 더 나은 정보 처리와 외삽 성능을 가능하게 합니다.



### Can LLMs Assist Annotators in Identifying Morality Frames? -- Case Study on Vaccination Debate on Social Media (https://arxiv.org/abs/2502.01991)
Comments:
          Accepted at 17th ACM Web Science Conference 2025 (WebSci'25)

- **What's New**: 최근의 연구는 COVID-19 팬데믹이 공공 건강 위기일 뿐만 아니라 디지털 플랫폼 활용에 있어 중요한 사건이라는 점을 강조합니다. 특히, 백신과 같은 논란의 여지가 있는 주제에서 소셜 미디어는 공공 담론에 중요한 영향을 미치고 있으며, 다양한 도덕적 관점이 개인의 의견에 큰 영향을 미친다는 점이 부각됩니다. 연구팀은 최신의 대형 언어 모델(LLMs)을 이용하여 인간 주석자가 백신 관련 논의에서 도덕적 프레임을 식별하는 데 도움을 줄 수 있는 가능성을 탐구하였습니다.

- **Technical Details**: 본 연구에서는 LLMs를 이용한 두 단계 절차를 통해 도덕적 프레임 분석을 수행합니다. 첫 번째 단계는 개념과 설명을 생성하는 것이고, 두 번째 단계는 '생각 소리 내기(thinking aloud)' 도구를 활용한 인간 평가입니다. LLMs는 적은 수의 예제를 통해 새로운 작업을 수행하는 능력인 few-shot learning을 통해 초기 레이블과 도덕적 프레임에 대한 설명을 생성함으로써 주석자의 인지 부담을 줄이고 일관성과 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: 연구 결과 LLMs를 주석 과정에 통합함으로써 주석의 정확성이 향상되었고, 작업의 난이도는 줄어들며, 인지 부담이 감소하는 것으로 나타났습니다. 수집한 피드백을 통해 참가자들은 LLMs의 설명이 유용하며, 도덕적 프레임 식별 작업에서 그들의 이해를 돕고 인지 부담을 줄이는 데 긍정적인 영향을 미쳤다고 응답했습니다. 이는 복잡한 심리언어학적 작업에서 인간과 AI 간의 협업 가능성을 제시하는 중요한 사례로 평가됩니다.



### Gradient-Regularized Latent Space Modulation in Large Language Models for Structured Contextual Synthesis (https://arxiv.org/abs/2502.01979)
- **What's New**: 이 연구에서는 구조화된 텍스트 생성을 위한 새로운 패러다임인 Gradient-Regularized Latent Space Modulation (GRLSM)을 도입하였습니다. GRLSM은 잠재 공간(latent space)에서의 구조적 제약을 적용하여 텍스트 생성 과정에서의 일관성을 향상시킵니다. 기존의 방법론이 가진 유연성과 일반화 부족 문제를 해결하여, 다양한 작업에 대한 적용 가능성을 높이려는 의도가 담겨 있습니다.

- **Technical Details**: GRLSM의 핵심 개념은 훈련 단계에서 잠재 변수(latent variables)에 제약을 두는 것입니다. 이를 통해 모델이 미리 정의된 구조적 패턴에 맞는 출력을 생성하도록 유도합니다. 또한, 그래디언트 정규화(gradient regularization) 원리를 통합하여 잠재 표현을 조절함으로써 구조적으로 일관된 텍스트 생성을 가능하게 합니다.

- **Performance Highlights**: 실험 결과 GRLSM 프레임워크는 perplexity 감소, 일관성 점수(coherence scores) 증가, 다양한 도메인에서의 구조적 정렬 개선을 보였습니다. 또한, 이 방법론은 생성된 텍스트의 의미적 일관성을 유지하면서 제약을 부여하여 구조적 불일치를 현저히 줄이는 성과를 나타냈습니다.



### CITER: Collaborative Inference for Efficient Large Language Model Decoding with Token-Level Routing (https://arxiv.org/abs/2502.01976)
- **What's New**: CITER(Collaborative Inference with Token-lEvel Routing) 프레임워크는 작은 언어 모델(SLMs)과 대형 언어 모델(LLMs) 간의 효율적인 협업을 가능하게 합니다. 이 새로운 접근법은 비판적이지 않은 토큰은 SLM에 라우팅하여 효율성을 높이고, 중요한 토큰은 LLM에 라우팅하여 일반화 품질을 유지합니다. 이 프레임워크는 정책 최적화 기법을 통해 라우터를 훈련시키며, 예측 품질과 생성 비용 모두에 따라 보상을 받도록 설계되어 있습니다.

- **Technical Details**: CITER는 토큰 수준의 라우팅을 활용하여 언어 모델의 추론 과정을 가속화합니다. 라우터는 각 토큰의 라우팅 점수를 예측하고, 미리 정의된 임계값에 따라 모델을 선택하여 토큰을 생성하도록 합니다. 이 과정은 강화 학습(reinforcement learning) 문제로 공식화되며, 라우터 훈련을 통해 현재 토큰의 정확성뿐만 아니라 장기적인 의사결정의 영향을 고려합니다.

- **Performance Highlights**: CITER는 다섯 개의 벤치마크 데이터셋을 통해 LLM의 추론 비용을 줄이며 높은 출력 정확도를 유지하는 효율성을 입증하였습니다. 이 방법은 최대 30% 적은 계산 비용으로 유사한 정확도를 달성하거나 동일한 비용으로 25% 더 높은 정확도를 제공합니다. 또한, 토큰 수준의 라우팅은 더 유연한 결과를 제공하여 성능을 극대화합니다.



### Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning (https://arxiv.org/abs/2502.01968)
- **What's New**: 이 논문은 감독된 미세 조정(Supervised Fine-Tuning, SFT) 과정에서 데이터 품질이 양보다 더 중요하다는 최근 연구를 바탕으로 한다. 특히, 데이터 정제 방법들이 전체 샘플을 필터링하는 데 중점을 두는 대신, 개별 토큰의 품질을 평가하고 개선하는 새로운 토큰 정제 파이프라인을 제안한다. 저자들은 비인간 라벨의 관점에서 토큰 품질을 조사하고, 유용한 정보를 보존하면서 불필요한 토큰을 제거하는 방법을 소개한다.

- **Technical Details**: 이 연구에서 저자들은 두 가지 주요 접근 방식을 통해 토큰 품질을 평가한다. 첫 번째는 수정된 모델(Fix-Model Cleaning)로, 고정된 두 개의 모델을 이용해 일괄적으로 SFT 데이터셋을 정제하는 방법이다. 두 번째는 자기 진화 모델(Self-Evolving Cleaning)로, 이 방법에서는 참조 모델이 반복적으로 업데이트되어 주어진 데이터의 각 부분을 청소한다. 이론적 분석을 통해 두 방법의 장점과 한계를 평가하였다.

- **Performance Highlights**: 다양한 다운스트림 작업에 대한 광범위한 실험을 통해, 제안된 토큰 정제 파이프라인이 기존 방법들보다 성능을 일관되게 향상시키는 것으로 나타났다. 특히, 이 방법은 모델이 관련성이 높은 토큰에 집중하도록 하여 다운스트림 결과를 개선하는데 큰 역할을 할 수 있음을 입증하였다.



### Boundary-Driven Table-Filling with Cross-Granularity Contrastive Learning for Aspect Sentiment Triplet Extraction (https://arxiv.org/abs/2502.01942)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 본 논문에서는 Aspect Sentiment Triplet Extraction (ASTE) 과제를 해결하기 위해 boundary-driven table-filling과 cross-granularity contrastive learning (BTF-CCL) 방법을 제안합니다. 기존의 2D table-filling 프로세스가 단어 수준의 상호작용에 초점을 맞추는 반면, 제안된 방법은 문장 수준의 표현과 단어 수준의 표현 간의 의미적 일관성을 높이는 데 중점을 둡니다. 이를 통해 모델은 복잡한 문장에서 다중 단어의 aspect와 opinion 용어 간의 관계를 보다 효과적으로 포착할 수 있습니다.

- **Technical Details**: BTF-CCL은 문장 내에서 aspect, opinion 및 sentiment 간의 관계를 학습하기 위해 긍정적 및 부정적 샘플 쌍을 구성합니다. 이 방법은 BERT를 이용하여 입력 문장을 인코딩하고, multi-scale, multi-granularity convolutional method (MMCNN)을 통해 로컬 의미 정보를 더욱 잘 포착합니다. 최종적으로, 모든 후보 영역이 감정 극성을 감지하고 분류됩니다.

- **Performance Highlights**: 실험 결과, BTF-CCL 방법이 기존의 최신 기법들과 비교하여 F1 점수 기준으로 더욱 뛰어난 성능을 보임을 증명하였습니다. 제안된 접근방식은 문장 수준의 맥락 정보를 보다 효과적으로 캡처하면서도 로컬 세부 사항에 대한 민감성을 유지합니다. 이러한 점에서, ASTE 분야의 발전에 기여할 것으로 기대됩니다.



### Can LLMs Maintain Fundamental Abilities under KV Cache Compression? (https://arxiv.org/abs/2502.01941)
Comments:
          21 pages

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 기본 기능에 대한 KV 캐시 압축 방법의 영향을 조사하는데 초점을 맞추고 있습니다. 기존 연구들은 긴 컨텍스트에서 인상적인 압축 비율을 달성했지만, 이러한 방법들이 모델의 핵심 기능에 미치는 영향은 충분히 탐구되지 않았습니다. 이를 통해 均一한 성능 저하가 작업 특이적으로 나타난다는 것을 발견했습니다.

- **Technical Details**: KV 캐시 압축 기술은 LLM 배포의 효율성을 높이기 위해 필수적이며, 모델 사이즈와 컨텍스트 길이가 증가함에 따라 메모리 관리에서의 필요성이 커졌습니다. 이 논문에서는 여러 가지 KV 캐시 압축 방법을 다양한 작업에 대해 검토하였으며, 특히 수치적 추론 작업이 압축에 민감하다는 것을 밝혔습니다. 새로운 방법 ShotKV를 통해 선행 및 디코딩 단계에서의 압축을 관리하면서도 의미적 일관성을 유지할 수 있음을 증명했습니다.

- **Performance Highlights**: ShotKV는 압축 비율이 증가하는 상황에서 긴 컨텍스트 생성 작업에서 성능을 9%에서 18% 향상시키는 결과를 보였습니다. 수치적 추론 및 안전성 관련 작업의 성능 저하가 크다는 것을 보여주었으며, 다단계 추론 모델이 지시 튜닝 모델보다 압축에 대한 내구성이 더 우수함을 나타냈습니다. 전반적으로, 다양한 작업에 대한 종합적인 평가를 통해 기존 압축 방법들의 한계를 파악하고 새로운 접근법인 ShotKV가 효과적임을 입증하였습니다.



### PANDAS: Improving Many-shot Jailbreaking via Positive Affirmation, Negative Demonstration, and Adaptive Sampling (https://arxiv.org/abs/2502.01925)
- **What's New**: 많은 샷 탈옥(many-shot jailbreaking)은 대형 언어 모델(LLMs)의 안전 정렬(security alignment)을 우회하는 방법으로, 사용자가 모델과의 대화를 여러 번 나눈 것처럼 보이도록 조작된 프롬프트를 사용합니다. 본 논문에서는 이러한 탈옥 기술을 개선하기 위해 PANDAS라는 하이브리드 기법을 제안합니다. PANDAS는 긍정적인 확인, 부정적인 시연, 그리고 주제에 최적화된 적응형 샘플링을 결합하여 성공적으로 작동합니다.

- **Technical Details**: PANDAS는 세 가지 주요 기술로 구성됩니다. 첫 번째로, 긍정적인 확인 구문이 악의적인 질문이 제기되기 전 대화에 삽입되어 모델의 지시 따르기 행동을 강화합니다. 두 번째로, 기존 질문-응답 쌍에 거부 및 수정 구문을 추가하여 모델이 거부를 처리하는 방법을 명시적으로 보여줍니다. 세 번째로, 특정 주제에 대해 최적 샘플링 분포를 파악하는 베이지안 최적화(Bayesian optimization) 방법을 사용하여 적응형 샘플링 전략을 개발합니다.

- **Performance Highlights**: PANDAS는 AdvBench와 HarmBench에서 최신 오픈 소스 모델을 사용하여 실험한 결과, 기존의 긴 컨텍스트 방식보다 유의미한 개선을 보였습니다. 긴 컨텍스트 시나리오에서 성공률이 크게 향상되었으며, 모델의 오래된 컨텍스트 취약성이 어떻게 악용되는지를 분석하는 데 중요한 통찰을 제공합니다. 전반적으로, PANDAS는 많은 샷 탈옥 기법을 크게 개선하여 안전성 검증을 회피하는 새로운 방법론을 제시하고 있습니다.



### Conceptual Metaphor Theory as a Prompting Paradigm for Large Language Models (https://arxiv.org/abs/2502.01901)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)의 추론 능력을 향상시키기 위한 새로운 방법으로 개념적 은유 이론(Conceptual Metaphor Theory, CMT)을 제안합니다. CMT는 복잡한 사고 작업에서 추상적 개념을 구조화하는 데 도움이 되는 은유적 매핑을 활용하여 LLMs가 더 인간적인 사고 패턴으로 유도되도록 합니다. 실험에서는 CMT 기반 프롬프트를 적용한 모델의 성능이 기존 모델보다 유의미하게 향상되었다고 보고합니다.

- **Technical Details**: 논문은 CMT 프롬프트 접근 방식을 통해 대규모 언어 모델의 추론을 강화하는 전략을 제시합니다. CMT는 구체적인 경험에서 온 출발 도메인(source domain)과 추상적인 개념을 포함하는 목표 도메인(target domain) 간의 체계적인 매핑을 통해 은유적 사유를 지원합니다. 여기에는 Llama3.2, Phi3, Gemma2, Mistral과 같은 네 개의 모델이 사용되었으며, 다양한 추론 작업에 대한 정확도와 일관성을 평가합니다.

- **Performance Highlights**: 결과적으로 CMT 기반 프롬프트가 파일 기반의 기존 모델보다 높은 추론의 정확성, 명확성 및 은유적 일관성을 보였습니다. 100개 과제의 벤치마크 데이터 세트를 통해 검증된 이 방법은 다양한 추론 도메인에서 중요한 통찰력을 제공하며, AI 시스템의 언어 처리 및 문제 해결 능력을 향상시키는 데 기여할 수 있습니다.



### Latent Lexical Projection in Large Language Models: A Novel Approach to Implicit Representation Refinemen (https://arxiv.org/abs/2502.01882)
- **What's New**: 이번 연구에서는 Latent Lexical Projection (LLP)이라는 새로운 접근 방식을 제안하였습니다. LLP는 기존 언어 모델 아키텍처 내에서 최적화된 프로젝션 메커니즘을 통합하여, 입력 임베딩과 문맥적 의미들 간의 정렬을 강화합니다. 이 방법은 레키컬(lexical) 표현을 구조적으로 변환하여 생성된 텍스트의 일관성과 적절성을 향상시키는 것을 목표로 합니다. 이에 따라, perplexity의 감소와 BLEU 점수의 증가가 관찰되어, 예측 정확도와 유창성에서 개선이 이루어졌음을 나타냅니다.

- **Technical Details**: LLP는 레키컬 임베딩을 구조화된 잠재 공간(latent space)으로 매핑하는 비주얼 변환을 포함합니다. 입력 임베딩 x는 비선형 프로젝션 함수 f에 의해 변환되며, 이 함수는 레키컬 표현의 의미적 일관성을 정제합니다. LLP는 특히 긴 거리의 토큰 의존성을 보존하는 데 유리하며, 그 결과 분류 정확도가 확장된 토큰 거리에서 증가합니다. 실험 세팅을 위해 최첨단 오픈 소스 LLM에 LLP를 구현하고 자연어 처리(NLP) 과제에서 성능을 평가하기 위한 데이터셋을 준비합니다.

- **Performance Highlights**: 실험 결과, 생성된 텍스트의 어휘 다양성이 증가하고 중복 및 반복적인 구문 구조의 일반적인 문제를 해결하였습니다. 또한, 디코딩 동안 불확실성이 감소하여 단어 선택에 대한 신뢰도가 향상된 것으로 나타났습니다. 이러한 성과는 LLP가 기존 언어 모델에 통합될 수 있는 실용성을 강조하며, 계산 효율성 또한 관리 가능한 범위 내에 유지되었습니다. 앞으로 LLP의 적용 가능성과 향후 연구 방향에 대한 통찰을 제시하고자 합니다.



### SelfCheckAgent: Zero-Resource Hallucination Detection in Generative Large Language Models (https://arxiv.org/abs/2502.01812)
- **What's New**: 이번 연구에서는 SelfCheckAgent라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 상징적 에이전트(Symbolic Agent), 특수 감지 에이전트(Specialized Detection Agent), 상황 일관성 에이전트(Contextual Consistency Agent) 등 세 가지 에이전트를 통합하여 다각적인 환각 감지를 수행합니다. 연구 결과, Llama 3.1과 함께하는 상황 일관성 에이전트가 WikiBio 데이터셋에서 비사실적 환각 감지에서 93.64%의 성능을 보였습니다.

- **Technical Details**: SelfCheckAgent는 환각 감지의 정확성을 높이기 위해 삼각 측량 전략(triangulation strategy)을 포함하고 있습니다. 이 프레임워크는 LLM의 응답 일관성을 평가하고 비사실적 오류를 정량화하는 방법을 활용하여 신뢰도를 크게 향상시킵니다. 또한, 복잡한 수학적 추론을 기반으로 한 새로운 환각 감지 데이터셋을 도입하여 LLM의 응답 행동에 대한 포괄적인 분석을 제공합니다.

- **Performance Highlights**: 자세한 성능 분석에 따르면, GPT-4o는 AIME 데이터셋에서 비사실적 감지에서 94.89%의 우수한 실적을 기록하였지만, 사실적 감지에서는 30.58%, 순위 매김에서는 30.68%로 낮은 수치를 기록했습니다. 이러한 결과는 복잡한 수학적 영역에서 환각 감지의 복잡성을 잘 보여줍니다. 최종적으로, SelfCheckAgent는 다양한 분야에서의 적용 가능성을 입증하며 신뢰할 수 있는 LLM을 위한 중요한 발전으로 자리매김하고 있습니다.



### On Bob Dylan: A Computational Perspectiv (https://arxiv.org/abs/2502.01772)
- **What's New**: 본 연구는 Cass Sunstein의 'On Bob Dylan' 에세이를 확장하여 Bob Dylan의 가사를 1962년부터 2012년까지 대규모 computational analysis (컴퓨테이셔널 분석)를 통해 분석합니다. 이 연구는 Dylan의 가사에서 개념 대 개념 관계를 추출하고, 이를 기반으로 방향성 지식 그래프를 구축하여 그의 주제적 구조를 포착합니다. 결과적으로, Dylan의 가사는 메타포에 대한 의존도가 증가하고, 감정 프로파일이 진화하며, 'dishabituation' (탈습관화)이 높아지는 것으로 나타났습니다.

- **Technical Details**: 연구자는 Bob Dylan의 1962년부터 2012년까지의 스튜디오 앨범 가사를 수집했습니다. o3-mini-high라는 대형 언어 모델을 활용하여 가사를 분석하고, 관련된 개념 쌍을 추출하여 각 개념 간의 관계를 구조화된 JSON 형식으로 저장했습니다. 이 과정에서 연구자는 노드(개념)를 정규화하고 관계를 확인하기 위해 다양한 네트워크 측정을 계산했습니다.

- **Performance Highlights**: 분석 결과, Dylan의 가사는 시대에 따라 테마가 다양하게 변하고, 이는 그의 음악적 변화에 대한 깊은 통찰을 제공합니다. 그는 특히 'movement', 'protest', 'mythic imagery'와 같은 주제를 다루며 그의 경력에 걸쳐 동적인 변화를 보였습니다. 이번 연구는 예술가의 개인적인 진화를 이해할 수 있는 새로운 방법론을 제시하며, 문화 및 창조적 변화의 연구에도 널리 적용될 수 있습니다.



### Evaluation of Large Language Models via Coupled Token Generation (https://arxiv.org/abs/2502.01754)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 평가와 순위를 정하는 데 있어 무작위성(randomization)의 영향을 제어해야 한다고 주장합니다. 구체적으로, 저자들은 같은 무작위성을 공유함으로써 서로 연결된 autoregressive 생성을 통해 LLM의 성능을 보다 신뢰성 있게 비교할 수 있는 방법을 제시합니다. 이를 통해 일반적인 autoregressive 생성 방식보다 필요 샘플 수를 줄일 수 있다는 점도 강조합니다.

- **Technical Details**: 이 연구의 핵심은 LLM의 선택적 sampler를 하나의 무작위 소스를 공유하도록 연결(coupled)함으로써, 평가의 일관성을 확보하려는 것입니다. LLM은 사전 지정된 토큰 분포(token distribution)를 기반으로 다음 토큰을 무작위로 선택하는 autoregressive 프로세스를 사용합니다. 이를 통해 발생할 수 있는 평가의 불확실성을 특정 할 수 있으며, 샘플의 수를 줄일 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, Llama 계열의 여러 LLM 모델을 사용하여, coupled autoregressive 생성은 vanilla autoregressive 생성 방식에 비해 평가에 필요한 샘플 수를 최대 40%까지 줄이는 것을 발견했습니다. 또한, LMSYS Chatbot Arena 플랫폼에서 수집한 데이터에서도 두 방법 간의 pairwise 비교 결과가 상당히 다르게 나타남을 확인했습니다. 이러한 결과는 기존 평가 프로토콜에서 모델 간의 이점이 무작위성에 의해 교란될 수 있음을 시사합니다.



### Comply: Learning Sentences with Complex Weights inspired by Fruit Fly Olfaction (https://arxiv.org/abs/2502.01706)
Comments:
          Accepted at NICE2025

- **What's New**: 이 논문에서는 생물학에서 영감을 받는 신경망을 통해 새로운 단어 임베딩(word embeddings) 모델인 Comply를 제안합니다. Comply는 FlyVec의 성능을 초월하면서도 더욱 생물학적으로 그럴듯한 표현을 사용하고 있습니다. 새로운 접근법은 시간 정보를 통합하여 임베딩 품질을 향상시키는 것을 목표로 합니다.

- **Technical Details**: Comply는 복소수(complex numbers)로 표현된 가중치를 통해 시퀀스 표현을 학습할 수 있도록 단일 층 신경망을 설계했습니다. 이 방법은 복잡한 입력을 사용하여 임베딩을 생성하며, 시간 패턴을 반영하여 우수한 성능을 보입니다. 또한, Comply는 비지도 에너지 함수(unsupervised energy function)의 확장을 활용하여 복소수 파라미터 행렬을 학습합니다.

- **Performance Highlights**: Comply는 문장 표현에서 높은 성능을 발휘하며, 특히 기존의 FlyVec보다도 우수한 결과를 보여줍니다. 이는 추가적인 파라미터 없이도 이루어지며, 기존의 큰 모델들과 동등한 성능을 자랑합니다. 실험 결과는 Comply가 FlyVec의 단순성과 해석 가능성을 유지하면서도 더 나은 임베딩 품질을 제공함을 나타냅니다.



### BARE: Combining Base and Instruction-Tuned Language Models for Better Synthetic Data Generation (https://arxiv.org/abs/2502.01697)
- **What's New**: 이 논문에서는 **Base-Refine (BARE)**라는 새로운 합성 데이터 생성 방법을 소개합니다. 기존의 instruct-tuned 모델에서는 다양성과 품질을 동시에 충족시키는 데 한계가 있었으나, BARE는 기본 모델의 다양성과 instruct-tuned 모델의 품질을 결합하여 향상된 성능을 제공합니다. 연구 결과, BARE를 사용하여 생성된 데이터로 모델을 미세 조정할 경우, 기존의 최첨단 방법에 비해 성능 향상이 효과적임을 보여주었습니다.

- **Technical Details**: BARE는 두 단계의 프로세스를 통해 합성 데이터를 생성합니다. 첫 번째 단계에서는 기본 모델을 사용하여 다양한 출력을 생성하고, 두 번째 단계에서는 이를 instruct-tuned 모델로 정제합니다. 이러한 방법론을 통해 생성된 데이터는 다양한 데이터 세트를 제작할 수 있으며, 下游 (downstream) 과제의 성능을 향상시킴을 입증하였습니다. 논문에서는 **GSM8K** 및 **RAFT**와 같은 현대적인 작업에서 BARE의 효과를 실험하였습니다.

- **Performance Highlights**: BARE로 생성된 데이터로 fine-tuning을 수행 시, 최소 1,000개의 샘플을 사용하여도 기존의 최첨단 모델과 비교할 수 있는 성능에 도달할 수 있음을 보여줍니다. 예를 들어, BARE로 생성된 데이터는 GSM8K에서 **101%**의 성능 향상을 제공하며, RAFT에서는 **18.4%**의 성능 향상이 나타났습니다. 이러한 향상은 기본 모델에서 생성된 데이터가 실제 데이터의 다양성을 잘 반영할 수 있음을 시사합니다.



### Agent-Based Uncertainty Awareness Improves Automated Radiology Report Labeling with an Open-Source Large Language Mod (https://arxiv.org/abs/2502.01691)
- **What's New**: 이 연구는 복잡하고 비영어 텍스트(예: 히브리어)의 방사선 보고서에서 구조화된 데이터를 신뢰성 있게 추출하기 위한 새로운 접근 방식을 소개합니다. 특히, 의학적 응용에서 LLM의 예측 신뢰도를 향상시키기 위해 에이전트 기반의 불확실성 인식 방법을 도입하였습니다. 이 방법은 2010년부터 2023년까지 크론병 환자의 히브리어 방사선 보고서 9,683개를 분석하였습니다.

- **Technical Details**: 연구팀은 512개의 보고서를 수동으로 주석 처리하여 6개의 위장 기관과 15개의 병리학적 발견을 기록하였고, 나머지 보고서는 HSMP-BERT를 사용하여 자동으로 주석 처리하였습니다. Llama 3.1 (Llama 3-8b-instruct)과 Bayesian Prompt Ensembles (BayesPE)를 통해 구조화된 데이터 추출이 이루어졌고, 이 과정에서는 불확실성을 추정하기 위해 의미적으로 동등한 6개의 프롬프트가 사용되었습니다.

- **Performance Highlights**: 에이전트 기반의 의사 결정 모델이 여러 프롬프트 출력을 통합하여 5개의 신뢰 수준으로 조정된 불확실성을 제공하며, 성능은 정확도, F1 점수, 정밀도, 재현율 및 Cohen's Kappa로 평가되었습니다. 에이전트 기반 모델은 모든 지표에서 기본 모델보다 우수한 성능을 보였고, F1 점수는 0.3967, 재현율은 0.6437로 나타났습니다. 불확실성이 높은 사례를 필터링한 후 F1 점수는 0.4787로 개선되었고 Kappa 점수는 0.4258로 증가했습니다.



### LLM-Powered Benchmark Factory: Reliable, Generic, and Efficien (https://arxiv.org/abs/2502.01683)
- **What's New**: 최근 대형 언어 모델(LLM)의 급격한 발전으로 인해 모델 공급과 응용 수요가 증가하고 있습니다. 이에 따라 신뢰할 수 있고 일반적인 벤치마크 생성기가 필요하지만, 기존의 LLM 벤치마크 생성기는 일반화 가능성 부족과 신뢰성 문제를 안고 있습니다. 이 논문에서는 자동화된 평가 프레임워크를 제안하고, 이를 기반으로 LLM을 직접 끌어내는 방식에서의 강점과 약점을 분석합니다.

- **Technical Details**: 벤치마크 생성기를 위한 자동화된 평가 프레임워크는 네 가지 차원과 열 개의 기준으로 구성되어 있습니다. 이를 통해 LLM을 벤치마크 생성기로 직접 활용할 때 나타나는 문제점을 식별하고 해결하는 다양한 방법이 BenchMaker로 통합되어 개발되었습니다. BenchMaker는 단계별 자체 정정 생성 및 갈등 유도 대조 분별법을 이용해 샘플의 신뢰성을 강화하는 방법을 제시합니다.

- **Performance Highlights**: 다양한 LLM과 과제를 대상으로 실시한 실험 결과, BenchMaker는 인간이 주석을 단 벤치마크와 비교하여 모든 지표에서 우수하거나 동등한 성능을 달성했습니다. 특히, BenchMaker는 12개의 LLM에 대해 높은 일관된 평가 결과를 제공하며(0.967 Pearson correlation against MMLU-Pro), 샘플당 $0.005 및 0.38분의 리소스를 소모합니다.



### The exception of humour: Iconicity, Phonemic Surprisal, Memory Recall, and Emotional Associations (https://arxiv.org/abs/2502.01682)
Comments:
          9 pages, 6 tables

- **What's New**: 이번 메타 연구는 유머(humor), 음소 빅램 서프라이잘(phonemic bigram surprisal), 정서적 가치(emotional valence), 그리고 기억 회상(memory recall) 사이의 관계를 탐구합니다. 기존 연구 결과에 따르면, 음소 서프라이잘이 높은 단어는 더 쉽게 기억되는 경향이 있으며, 이는 예측 불가능한 음소 순서가 장기 기억 회상을 촉진한다는 것을 시사합니다.

- **Technical Details**: 본 연구에서는 부정적인 연상(negative associations)이 있는 단어들이 종종 더 높은 서프라이잘을 보이며 기억하기 쉽다는 점을 강조합니다. 또한, 유머와 관련된 단어는 긍정적인 감정을 유발하면서도, 높은 서프라이잘을 보이고 기억력이 향상되는 특이성을 가지고 있습니다.

- **Performance Highlights**: 이 연구는 부정적인 경험과 자극이 일반적으로 긍정적인 것보다 더 쉽게 기억된다는 점을 보여줍니다. 연구 결과는 유머가 긍정적인 감정을 포함하면서도 기억 회상에 있어 특별한 역활을 한다는 것을 강조합니다.



### Benchmark on Peer Review Toxic Detection: A Challenging Task with a New Datas (https://arxiv.org/abs/2502.01676)
Comments:
          Accepted to WiML workshop @Neurips 2024

- **What's New**: 이번 연구에서는 동료 평가(peer review)에서 독성이 있는 피드백을 탐지하는 새로운 접근 방식을 제시합니다. 평가의 독성을 네 가지 범주로 나누고, OpenReview 플랫폼에서 수집된 동료 평가 데이터셋을 전문가들이 주석을 달아 정의했습니다. 이는 기존의 연구들에서 다루지 않았던 중요한 분야로, 독성을 효과적으로 탐지하는 모델들에 대한benchmarking도 수행하였습니다.

- **Technical Details**: 연구팀은 독성 탐지 모델(toxicity detection model), 감정 분석 모델(sentiment analysis model), 여러 오픈소스 대형 언어 모델(LLMs) 및 두 개의 비공식 LLM을 포함하여 다양한 모델을 평가했습니다. 각 모델의 성능은 퍼진 프롬프트(prompt granularity)에 따라 다르게 나타났으며, 특히 데이터셋의 유용성을 통해 LLM의 독성 탐지 능력을 개선할 수 있음을 강조합니다. 또한, 모델의 신뢰도 점수(confidence score)는 인간의 판단과의 일치도를 나타내는 좋은 지표로 작용합니다.

- **Performance Highlights**: 실험 결과, GPT-4와 같은 최신 LLM은 간단한 프롬프트에 대해 인간의 판단과는 낮은 일치를 보였으나, 세부적인 지침을 사용할 때 더 좋은 일치를 보였습니다. 예를 들어, GPT-4는 인간의 판단에 대해 Cohen의 Kappa 점수 0.56을 달성한 반면, 신뢰도가 95% 이상인 예측만 사용할 때에는 이 점수가 0.63으로 증가했습니다. 이러한 결과는 LLM의 독성 탐지 향상을 위한 지속적인 연구의 필요성을 강조합니다.



### Multilingual State Space Models for Structured Question Answering in Indic Languages (https://arxiv.org/abs/2502.01673)
- **What's New**: 이번 연구는 인디언 언어에서의 질문 응답(Question Answering, QA) 작업에 대해 State Space Models (SSMs)의 최초 적용을 다룹니다. SSM은 시퀀스 데이터의 장기 및 단기 의존성을 모델링할 수 있어 복잡한 문법 구조를 효율적으로 처리하는 데 유리합니다. 연구진은 다양한 인디언 언어 데이터셋에서 여러 개의 SSM 아키텍처를 평가하고, 이들 모델이 언어의 미세한 뉘앙스를 효과적으로 포착한다는 결과를 도출하였습니다.

- **Technical Details**: SSMs는 입력 매트릭스와 상태 전이가 포함된 구조로, 시스템의 동적 과정을 표현하는 상태 벡터를 가지고 있습니다. 이 연구에서는 기존의 Transformer 아키텍처와 비교하여 SSM의 장점과 단점을 강조하고, 입력 데이터의 선택적 전파를 가능하게 하는 Mamba라는 기술적 발전을 소개합니다. SSM 모델은 효율성과 확장성을 중시하여 이론적으로 계산 복잡성을 선형으로 유지하면서 장기간의 의존성 문제를 해결할 수 있습니다.

- **Performance Highlights**: 연구 결과 SSM 모델은 질문 해석, 맥락 정렬 및 답변 생성을 통해 QA 시스템의 성능을 향상시키는 데 기여했습니다. 특히, SSM은 인디언 언어의 복잡성을 관리하는 데 필요한 조건들을 잘 충족시키며,低-Resource 상황에서도 추가적인 최적화를 제안합니다. 이 연구는 인디언 언어의 QA 시스템 개발을 위한 기초 벤치마크를 제공하며, 향후 관련 연구의 방향성을 제시합니다.



### Explainable AI for Sentiment Analysis of Human Metapneumovirus (HMPV) Using XLN (https://arxiv.org/abs/2502.01663)
- **What's New**: 2024년 중국에서 발생한 인체 메타뉴모바이러스(HMPV)의 확산은 영국 등 다른 국가로 확대되며 공공의 우려를 일으켰습니다. 본 연구는 HMPV에 대한 공공 반응을 이해하기 위해 소셜 미디어 데이터를 활용한 감정 분석(sentiment analysis)을 탐구합니다. 특히 XLNet 모델을 사용하여 93.50%의 정확도로 감정을 분류합니다.

- **Technical Details**: HMPV는 주로 기침, 인후통과 같은 감기를 유발하는 호흡기 바이러스입니다. 높은 위험군인 어린이, 노인, 면역력이 저하된 환자에게서는 심각한 질병을 초래할 수 있습니다. 본 연구에서는 HMPV 관련 댓글을 수집하여 감정 분석을 수행하며, SHAP(Shapley Additive Explanations)를 통해 모델의 투명성을 향상시킵니다.

- **Performance Highlights**: HMPV에 대한 감정 분석 프레임워크를 구축함으로써, 최신 NLP 기술을 활용하여 신뢰성과 해석 가능성을 높였습니다. 이를 통해 HMPV 발생에 대한 공공의 감정을 파악하고, 보다 효과적인 건강 커뮤니케이션 및 정책 개발을 지원할 수 있는 인사이트를 제공합니다.



### Speculative Ensemble: Fast Large Language Model Ensemble via Speculation (https://arxiv.org/abs/2502.01662)
- **What's New**: 최근 큰 발전이 있었던 Large Language Models (LLMs)에서, Ensemble 방법들이 여러 모델을 결합하여 성능을 향상시키고 있으나, 높은 계산 비용이 문제로 남아있습니다. 본 논문에서는 Speculative Ensemble을 제안하며, 이는 성능 저하 없이 LLM 앙상블의 속도를 향상시키는 새로운 프레임워크입니다. 이를 통해 제안 모델이 토큰을 순차적으로 생성하고, 대형 목표 모델이 이를 병렬로 검증하는 방식을 채택합니다.

- **Technical Details**: Speculative Ensemble (SE)은 두 가지 주요 통찰력을 기반으로 합니다. 첫째, 검증 분포는 제안 및 목표 모델의 앙상블 분포가 될 수 있으며, 둘째, 각 모델을 제안자 및 검증자로 번갈아 사용함으로써 효율성을 더욱 높일 수 있습니다. 이 방법을 n개의 모델로 일반화하고 이론적으로 SE가 표준 앙상블보다 느릴 수 없음을 증명하여, 일반적으로 더 빠른 속도를 달성할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과에 따르면 Speculative Ensemble은 표준 앙상블 기법보다 1.11배에서 2.23배의 속도 향상을 보여주며, 생성 품질을 저하시키지 않습니다. 다양한 작업에서 무작위 표본을 통해 Llama, Vicuna, Qwen 시리즈 등 여러 모델 쌍을 테스트하여 SE의 일관된 가속을 확인했습니다. 이는 특히 가중 앙상블 환경에서 1.34배에서 1.85배의 속도를 달성함을 보여주며, 본 방법의 실용성을 증명합니다.



### Large Language Models' Accuracy in Emulating Human Experts' Evaluation of Public Sentiments about Heated Tobacco Products on Social Media (https://arxiv.org/abs/2502.01658)
- **What's New**: 이 연구는 소셜 미디어에서 대체 담배 제품에 대한 감정 분석의 중요성을 강조하고, 감정 평가의 인간 작업을 효율화할 수 있는 대형 언어 모델(Large Language Models, LLMs)의 성능을 평가했습니다.

- **Technical Details**: 연구팀은 GPT-3.5와 GPT-4 Turbo를 사용하여 500개의 페이스북 및 500개의 트위터 메시지를 분류했습니다. 메시지는 반HTP, 찬성HTP, 중립 메시지로 분류되었고, 각 메시지는 최대 20회 평가되었습니다. 결과적으로 GPT-3.5는 페이스북 메시지에서 61.2%의 정확성을 보였고, GPT-4 Turbo는 81.7%로 더 높은 정확성을 기록했습니다.

- **Performance Highlights**: GPT-4 Turbo는 세 개의 응답 인스턴스를 사용하여 20개의 인스턴스의 99%에 가까운 정확성을 달성했습니다. 또한, GPT-4 Turbo는 중립 메시지에 비해 반HTP 및 찬성HTP 메시지에 대해 더 높은 정확성을 보였으며, GPT-3.5는 종종 반HTP 메시지를 중립으로 잘못 분류하는 경향이 있었습니다.



### Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies (https://arxiv.org/abs/2502.02533)
Comments:
          11 pages, 7 figures, 1 table (30 pages, 9 figures, 5 tables including references and appendices)

- **What's New**: 본 논문에서는 다중 에이전트 시스템(MAS) 설계를 자동화하기 위한 Multi-Agent System Search(MASS)라는 새로운 최적화 프레임워크를 제안합니다. 기관세가 없이 효율적인 설계 과정을 통해 MAS의 성능 향상에 기여하며, LLM(대형 언어 모델) 기반 에이전트들이 상호작용하며 문제를 해결하는 데 필요한 최적의 프롬프트와 토폴로지를 탐색합니다. 이 시스템은 세 가지 단계로 구성되며, 각 단계는 이전 단계에서 최적화된 프롬프트 및 토폴로지에 조건화 됩니다.

- **Technical Details**: MASS는 블록 수준의 프롬프트 최적화와 워크플로우 토폴로지 최적화, 글로벌 프롬프트 최적화를 조합해 효과적인 에이전트를 설계합니다. MAS 설계에는 블록 수준 디자인과 워크플로우 레벨 조정이 포함되며, 토폴로지 최적화는 각기 다른 에이전트들과 그들의 배열에 대한 결정 과정을 포함합니다. 본 연구는 MAS에서 프롬프트와 토폴로지 디자인이 성능에 미치는 영향에 대한 정량적 분석을 제공합니다.

- **Performance Highlights**: MASS로 최적화된 다중 에이전트 시스템은 기존의 수동 설계된 대안들에 비해 월등히 우수한 성능을 보여주었습니다. 특히, reasoning, multi-hop understanding, code generation 등 다양한 작업에서 상태 최첨단 성능을 달성하였습니다. 연구 결과는 효과적인 다중 에이전트 시스템 구축을 위한 가이드라인을 제공합니다.



### Analyzing Similarity Metrics for Data Selection for Language Model Pretraining (https://arxiv.org/abs/2502.02494)
Comments:
          14 pages

- **What's New**: 이 논문은 언어 모델의 사전 훈련을 위한 데이터 큐레이션(data curation)에 적합한 임베딩 모델의 suitability를 분석하는 프레임워크를 소개합니다. 기존의 비슷함(similarity) 측정 방법이 일반적으로 과제를 위해 훈련된 오프더셸프(Off-the-shelf) 임베딩 모델을 사용하는 반면, 저자들은 임베딩 공간에서의 유사성과 사전 훈련 손실(pretraining loss) 간의 상관관계를 정량화합니다. 이 연구는 다양한 임베딩 모델을 분석하고, 단순한 방법으로 토큰 당 임베딩의 평균을 내는 것이 매우 경쟁력이 있음을 발견했습니다.

- **Technical Details**: 저자들은 Pile 데이터셋을 사용하여 17억 파라미터를 가진 디코더 전용 언어 모델을 사전 훈련하는 실험을 수행했습니다. 연구는 주로 K-Means 클러스터링 알고리즘을 사용하여 주어진 임베딩 공간의 낮은 거리와 같은 손실 값을 측정하는 방식으로 진행됩니다. 또한 임베딩 공간이 서로 다른 데이터 소스로부터의 예시들을 얼마나 잘 분리할 수 있는지 평가합니다.

- **Performance Highlights**: 저자들은 다양한 임베딩 모델이 사전 훈련 데이터 큐레이션에서 유용하다는 것을 발견했습니다. 특히 다수의 임베딩을 사용한 데이터 큐레이션 방법이 기존 기준 훈련보다 성능이 우수하다는 결과를 얻었습니다. 마지막으로, 데이터 큐레이션을 위한 임베딩 모델에 대한 특별한 디자인이 필요하다는 것을 강조하고, 이 연구가 임베딩 모델의 평가 및 새로운 디자인 개발에 기초가 될 수 있음을 시사합니다.



### Rankify: A Comprehensive Python Toolkit for Retrieval, Re-Ranking, and Retrieval-Augmented Generation (https://arxiv.org/abs/2502.02464)
Comments:
          Work in Progress

- **What's New**: 본 논문은 정보 검색, 재순위화(re-ranking), 검색 증강 생성(RAG) 등 현대 자연어 처리(NLP) 애플리케이션을 위한 통합 프레임워크인 Rankify를 소개합니다. Rankify는 다양한 retrieval 기법과 최신 재순위 모델을 지원하며, 비교적 간편하게 라이브러리를 설치하고 사용할 수 있도록 설계되었습니다. 또한 연구자들이 다양한 실험을 진행할 수 있도록 40개의 커스터마이즈된 데이터셋을 포함하고 있습니다.

- **Technical Details**: Rankify는 sparse(BM25) 및 dense(DPR, ANCE 등) 리트리버와 24개의 최신 재순위 모델을 통합하여 사용자가 손쉽게 다양한 retrieval 및 ranking 실험을 수행할 수 있게 합니다. 재순위 기술은 pointwise, pairwise, listwise로 나누어지며, Rankify는 이러한 다양한 접근 방식을 지원합니다. RAG 통합을 통해 Rankify는 검색된 문서를 LLM에 전달하여 평가 과정을 연결합니다.

- **Performance Highlights**: Rankify는 실험에서의 일관성과 확장성을 보장하며, 다양한 평가 메트릭과 도구를 제공합니다. 개방형 문서화가 잘 되어 있어 사용자가 각 기능을 쉽게 탐색할 수 있습니다. Rankify는 PyPI 및 GitHub를 통해 무료로 제공되며, 연구자와 실무자에게 접근성이 높은 도구로 자리잡을 것입니다.



### Avoiding spurious sharpness minimization broadens applicability of SAM (https://arxiv.org/abs/2502.02407)
- **What's New**: 이 논문은 Sharpness Aware Minimization (SAM) 기법이 자연어 처리(NLP) 분야에서 성능이 떨어진다는 점을 강조합니다. 저자들은 SAM의 성능 저하 원인을 규명하고, 이에 대한 해결책으로 Functional-SAM이라는 대체 알고리즘을 개발하였습니다. 이 알고리즘은 함수의 통계값 조정을 통해 곡률을 정규화하며, 불필요한 최소화를 피합니다.

- **Technical Details**: 논문에서는 Functional-SAM과 SAM의 성능 비교를 위해 창에서의 경로 분석을 수행했습니다. 연구진은 SAM의 기여를 두 가지로 나누어, 로짓(statistics of the logit) 통계 수정을 통한 샤프니스 감소와 함수의 기하학 수정 과정을 분석합니다. 이를 통해 NLP 환경에서는 로짓 경로가 지배적임을 발견하였고, 이러한 점을 해결하기 위해 새로운 정규화 방법을 고안했습니다.

- **Performance Highlights**: Functional-SAM 알고리즘은 AdamW 및 SAM 기법과 비교해 성능 향상을 보여주었고, 다양한 모델 크기에서도 일관된 결과를 보였습니다. 또한, preconditioned SAM과 결합하여 최대 성능 향상을 이끌어냈습니다. 이 연구는 LLMs와 같은 대규모 언어 모델에 곡률 정규화의 적용 가능성을 보여주는 중요한 기초 자료가 될 것입니다.



### ReSpark: Leveraging Previous Data Reports as References to Generate New Reports with LLMs (https://arxiv.org/abs/2502.02329)
- **What's New**: 이번 논문에서는 ReSpark라는 새로운 데이터 보고서 생성 방법을 제안합니다. 이 방법은 기존의 데이터 보고서를 참조하여 새로운 보고서를 작성하며, 유사한 주제를 가진 여러 보고서를 검색하고 분석 목표에 맞춰서 세분화된 분석 논리를 제공합니다. ReSpark는 사용자가 실시간 출력을 검토하고 새로운 목표를 삽입하며 보고서 내용을 수정할 수 있는 상호작용 인터페이스를 제공합니다.

- **Technical Details**: ReSpark 방법은 기존 보고서를 재구성하여 새로운 데이터에 맞게 분석 작업을 수행하는 과정으로, 분석 목표, 차트, 텍스트 설명의 상호 의존적 세그먼트로 워크플로우를 구조화합니다. 이 과정에서 자연어 처리(natural language processing) 기술이 복잡하게 활용되며, 각 분석 목표는 새로운 데이터에 맞추어 재분석되어 코드, 시각화 및 통찰력이 생성됩니다. ReSpark는 사용자 연구를 통해 그 효과가 검증되었습니다.

- **Performance Highlights**: ReSpark는 기존 보고서의 참조를 통해 데이터 보고서를 생성하는 과정에서 분석 목표의 불일치와 데이터 변환의 필요성을 파악하고, 사용자 맞춤형 통찰력을 생성하는 데 효과적입니다. 비교 연구와 사용성 연구를 통해 ReSpark는 기존 보고서의 새롭게 수정된 보고서를 효과적으로 생성할 수 있음을 입증하였습니다. 이 결과는 데이터 보고서 작성에서 LLM과의 협업을 향상시킬 수 있는 기회를 제공합니다.



### VaiBot: Shuttle Between the Instructions and Parameters (https://arxiv.org/abs/2502.02315)
- **What's New**: 본 논문은 LLMs(대형 언어 모델)에서의 지시어(instructions)의 통합적 모델링을 위한 새로운 신경망 프레임워크인 VaiBot을 제안합니다. VaiBot은 VAE(변분 오토인코더)와 VIB(변분 정보 병목)을 결합하여 추론(deduction)과 귀납(induction) 작업을 동시에 여유롭게 모델링할 수 있도록 설계되었습니다. 이 연구는 이전의 연구들이 지시어의 출현과 LLM의 데이터 기반 훈련을 별개의 과정으로 간주한 점에서 차별화됩니다.

- **Technical Details**: VaiBot은 주어진 지식 k를 잠재 표현 z로 매핑하기 위해 학습됩니다. 이 z는 LLM의 추가 매개변수 역할을 하면서도, 원래의 지식 k를 재구성하는 데에도 사용됩니다. VaiBot은 텍스트 인코더, 디코더, 그리고 작업 LLM으로 이루어져 있으며, 인코더의 기능은 주어진 지시어를 잠재적 표현으로 변환하는 것입니다.

- **Performance Highlights**: 실험 결과, VaiBot은 추론 능력에서 기존의 방법들과 동등한 성능을 보였고 귀납 능력에서는 40% 이상의 성능 개선을 달성했습니다. 또한, VaiBot은 일반적인 지시어 데이터로 훈련하여 여러 작업에 대해 일반화할 수 있는 능력을 개발했으며, T-SNE 차원 축소 기법을 통해 더 우수한 추론 성능을 보여주었습니다.



### Can You Move These Over There? An LLM-based VR Mover for Supporting Object Manipulation (https://arxiv.org/abs/2502.02201)
Comments:
          64 pages (30 in main text), 22 figures (19 in main text)

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)을 활용하여 사용자의 음성 명령을 이해하고 해석하여 가상 현실(VR)에서 객체 조작을 지원하는 VR Mover라는 솔루션을 제안합니다. 사용자가 단순히 가리키고 말함으로써, VR Mover는 구조화된 입력 없이 객체를 조작할 수 있는 능력을 갖추고 있습니다. 사용자 연구를 통해 이 인터페이스가 다중 객체 조작에서의 사용자 유용성을 향상시키고, 작업 부담과 팔의 피로를 줄인다는 결과를 보여주었습니다.

- **Technical Details**: VR에서 객체 조작은 3D 객체를 이동시키고 조작하는 작업을 의미합니다. 하지만 전통적인 사용 방법은 사용자가 객체를 선택하고 위치, 회전, 크기를 지정한 후 조작을 확인해야 하며, 이 과정에서 ‘고릴라 팔 효과’로 인한 팔의 피로가 발생할 수 있습니다. 저자들은 사용자가 자연스럽게 음성과 제스처를 결합하여 객체 조작을 할 수 있도록 LLM에 여러 API를 제공하여 문제를 해결하고자 하였습니다.

- **Performance Highlights**: VR Mover는 비정형적(Unstructured), 불완전한(Incomplete), 맥락화된(Contextualized) 명령을 다룰 수 있는 혁신적인 LLM 기반 객체 조작 인터페이스를 통해 사용자 경험을 크게 개선하였습니다. 사용자 연구에서 LLM 기반 인터페이스가 사용성, 전체 경험, 다중 객체 조작의 성능을 향상시키고, 팔의 피로와 작업 부담을 감소시킨다는 결과가 도출되었습니다. 이 연구는 향후 LLM 기반 객체 조작 인터페이스 설계에 대한 통찰을 추가로 제공할 것으로 기대됩니다.



### Vulnerability Mitigation for Safety-Aligned Language Models via Debiasing (https://arxiv.org/abs/2502.02153)
Comments:
          37 pages

- **What's New**: 이 논문은 AI의 안전 정렬(safety alignment) 문제를 다루며, 기존의 방법들이 특정 카테고리의 안전성을 보장하는 데 실패함을 보여줍니다. 저자들은 다양한 모델을 평가한 결과, 일반적으로 안전성을 향상시켰지만 특정 취약점을 제거하기가 어렵다는 점을 강조합니다. 새로운 방법인 Token-level Safety-Debiased Inference (TSDI)를 소개하여 안전성 편향을 추정하고 수정하는 과정을 제안합니다.

- **Technical Details**: 이 연구는 대형 언어 모델(LLM)의 안전성 및 유용성 간의 트레이드오프를 연구합니다. 기존의 방법들은 단일 보상 지표에 의존하는 경향이 있으며, 이는 다양한 안전성 요구를 충족하지 못하는 문제를 초래합니다. 특히, 보상 모델링 및 인간 피드백을 통한 강화 학습(RLHF) 적용 시 제약을 걸고 안전한 LLM을 개발하는 새로운 접근법을 제공합니다.

- **Performance Highlights**: TSDI 방법을 사용하여 모델의 유용성을 향상시키면서도 안전성을 유지하는 결과를 보였습니다. 실험 결과, TSDI는 생성 과정에서 안전성 편향을 수정하여 더 높은 수준의 유용성을 가져오면서도 위험 요소를 최소화하는 데 기여합니다. 이는 안전성과 유용성을 동시에 높이는 최적의 트레이드오프를 달성하는 데 중요한 성과로 여겨집니다.



### Risk-Aware Driving Scenario Analysis with Large Language Models (https://arxiv.org/abs/2502.02145)
Comments:
          IEEE Intelligent Vehicles Symposium 2025

- **What's New**: 이번 논문은 Large Language Models (LLMs)를 활용하여 자율주행 시스템에서 생성되는 운전 시나리오의 위험 인식 분석(risk-aware analysis)을 위한 새로운 프레임워크를 제안합니다. LLMs의 강력한 정보 처리 능력을 통해 자율주행 시뮬레이터에서 생성된 시나리오의 안전성을 평가할 수 있는 가능성을 탐구하고 있습니다. 논문에서는 비판적인 안전성(safety-critical)을 평가하기 위한 LLMs의 유용성을 검증하는 실증적 평가를 수행하였습니다.

- **Technical Details**: LLMs의 능력을 활용하여, 기존의 비위험한(non-critical) 시나리오를 수정하여 새로운 안전 비판적 시나리오를 생성하는 적대적 방법(adversarial method)을 사용하는 프레임워크를 설계하였습니다. 이 과정에서 자율주행 테스트 시뮬레이터가 생성한 다양한 운전 시나리오를 평가하고, 이 시나리오들이 안전성과 관련하여 얼마나 효과적인지를 분석합니다. 이러한 방법론은 또한 운동 계획 알고리즘의 유효성을 검증하는 데 활용될 수 있습니다.

- **Performance Highlights**: 실험 결과, LLMs는 자율주행 시뮬레이터가 생성한 시나리오의 안전성을 평가하는 데 매우 효과적임을 보여주었습니다. 이 연구는 자율주행 시스템의 안전성을 높이는 데 기여할 수 있는 가능성을 제시하며, 새로운 테스트(Object)에 대한 피드백을 제공하는 방식으로, 기술의 적용 가능성을 넓히고 있습니다. 연구에 사용된 코드와 시나리오는 제공된 링크에서 확인할 수 있습니다.



### Robust and Secure Code Watermarking for Large Language Models via ML/Crypto Codesign (https://arxiv.org/abs/2502.02068)
- **What's New**: 본 논문은 RoSe라는 독창적인 ML/Crypto 코드 디자인 워터마킹 프레임워크를 도입합니다. 이는 LLM(Large Language Model) 생성 코드의 지적 재산권을 보호하고 소프트웨어 개발에서 부적절한 사용을 방지하는 데 중점을 둡니다. RoSe는 고품질 워터마크를 생성하는 데 있어 기존의 워터마킹 방법의 한계를 극복하고 있으며, 안전한 검증을 위해 zero-knowledge proof(제로 지식 증명)을 사용합니다.

- **Technical Details**: RoSe는 워터마크 삽입 및 추출 모듈을 end-to-end 방식으로 훈련하여 코드 기능을 변경하지 않으면서 워터마크의 검출성과 강건성을 높입니다. 이를 위해 CodeT5를 백본으로 사용하여 코드의 구문 및 변수 이름 전환 검색 공간을 늘리고, 워터마크 삽입 시 코드를 손상시키지 않도록 하여 검출성과 강건성을 극대화합니다. 훈련된 워터마크 인코더는 LLM 생성 코드에 소유자의 서명을 삽입하며, 사용자가 코드에 대한 검사를 요청할 수 있도록 합니다.

- **Performance Highlights**: RoSe는 extensive evaluation을 통해 코드를 기능적으로 유지하면서도 0.97의 detection AUROC을 기록하며, 공격에 대한 저항력도 뛰어난 성능을 보여줍니다. 또한, RoSe는 zero-knowledge proof를 통해 120ms 이내에 안전하게 코드 스니펫의 출처를 검증하는 효율성을 제공합니다. 이러한 성과는 코드 워터마킹의 detectability-fidelity-robustness 간의 균형을 이루어냅니다.



### AdaptBot: Combining LLM with Knowledge Graphs and Human Input for Generic-to-Specific Task Decomposition and Knowledge Refinemen (https://arxiv.org/abs/2502.02067)
Comments:
          Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2025

- **What's New**: 이번 연구에서는 LLMs(대규모 언어 모델)와 KG(지식 그래프)의 결합을 통해 새로운 작업과 시나리오에 신속하게 적응할 수 있는 조력형 에이전트를 위한 새로운 프레임워크를 제안합니다. 기존의 지식이 부족한 환경에서도 인간의 피드백을 활용하여 LLM의 출력을 수정하고 향상시킬 수 있는 접근 방법을 소개합니다. 이러한 프레임워크는 이전의 당면 과제를 해결하기 위한 신뢰할 수 있는 대안을 제공하여 LLM의 일반적인 예측 능력을 폭넓게 활용합니다.

- **Technical Details**: 제안된 프레임워크는 LLM을 사용하여 주어진 작업을 수행하기 위한 행동 시퀀스(서브 태스크)를 생성하고, KG를 통해 도메인 특정 지식을 인코딩하여 LLM 출력의 수정 가능성을 지원합니다. 각 서브 태스크는 LLM 호출을 통해 만들어지며, KG의 정보와 비교하여 불일치가 발견될 경우 이를 해결하기 위한 대체 행동을 제시합니다. 이 과정에서 HUMAN-IN-THE-LOOP(HITL) 방식으로 인간의 피드백을 요청하여 KG를 수정하고 새로운 작업에 대한 적응력을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크를 사용한 조력형 에이전트는 기존 LLM이나 LLM과 KG의 조합 만으로 수행한 작업에 비해 성능이 향상된다는 것이 입증되었습니다. 요리와 청소 작업에 대한 시뮬레이션 평가에서 유의미한 성능 향상을 보여주며, 복잡한 작업 구성을 신속하게 적응하는 특성을 강조합니다. 이러한 성능 향상은 에이전트가 새로운 작업을 수행하는 방식과 기존 지식을 지속적으로 수정하는 능력에서 비롯됩니다.



### Anticipate & Act : Integrating LLMs and Classical Planning for Efficient Task Execution in Household Environments (https://arxiv.org/abs/2502.02066)
Comments:
          Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2024

- **What's New**: 본 논문에서는 가정에서의 효율적인 작업 수행을 위해, Upcoming tasks를 예상하고 이를 기반으로 action sequence를 함께 계산하는 새로운 프레임워크를 제안합니다. LLM(대규모 언어 모델)의 사전 지식을 활용하여, 작은 수의 prompts로 고수준의 작업을 예측하고, 이를 기존의 계획 시스템에서 목표로 사용하여 세밀한 액션들을 계산합니다. 이러한 접근 방법은 기존의 작업 예측 방법의 한계를 극복할 수 있습니다.

- **Technical Details**: 제안된 프레임워크는 두 가지 주요 요소와 관련되어 있습니다: 1) LLM에서 인코딩된 일반적 지식을 이용해 작업의 부분 시퀀스를 기반으로 후속 작업을 예측하고, 2) PDDL(Planning Domain Definition Language)를 사용하여 세부 도메인 지식을 표현합니다. 이 프레임워크는 기존의 상징적 및 휴리스틱 클래스 계획자를 수정하여 즉각적인 작업과 예상된 작업을 목표로 하여 행동의 시퀀스를 계산합니다.

- **Performance Highlights**: 프레임워크의 성능은 VirtualHome 시뮬레이션 환경에서 평가되었으며, 예상하지 않은 작업을 고려하지 않은 시스템과 비교하여 실행 시간이 31% 단축되었습니다. 또한 계획의 길이는 12% 단축되었습니다. 이러한 개선 사항은 복잡한 가정 환경에서 여러 작업을 수행하는 데 있어 매우 유익한 결과를 제공합니다.



### Efficient Domain Adaptation of Multimodal Embeddings using Constrastive Learning (https://arxiv.org/abs/2502.02048)
- **What's New**: 최근의 기계 학습(ML), 자연어 처리(NLP), 그리고 기초 모델들(foundation models)의 발전은 헬스케어와 같은 컴퓨팅 자원이 제한된 분야에서 실제 응용 가능성을 보여주고 있습니다. 이러한 분야에서 기초 모델과 감독 기계 학습(supervised ML)의 결합은 진단 및 치료 계획과 같은 업무를 자동화할 수 있는 잠재력을 제공합니다. 그러나 이러한 기술을 효과적으로 적용하기에는 현장 computational resource의 제한이 큰 도전 과제가 되고 있습니다.

- **Technical Details**: 우리의 접근 방식은 기초 모델을 fine-tuning 하지 않고 downstream task에 embedding을 효율적으로 적응시키는 방법을 제안합니다. 이 방법은 Large Language Models (LLMs)와 Vision Models의 frozen embeddings를 활용하고, 대조 학습(contrastive learning)을 통해 작은 비선형 모델을 훈련시킵니다. 이는 각 embedding을 새로운 task-specific 하위 공간(subspace)으로 매핑하여 동일한 레이블을 가진 embedding은 가깝게, 다른 레이블을 가진 embedding은 멀리 위치하도록 학습합니다.

- **Performance Highlights**: 실험 결과, 우리의 접근 방식은 임상 노트(clinical notes)에서 성능을 향상시키는 데 있어 20% 이상 F1 점수 증가를 보여주었습니다. 이 방법은 CPU만을 사용해도 수천 개 샘플에 대해 몇 분 이내에 실행 가능하며, 기존의 PCA보다 성능이 크게 향상됩니다. 또한 원래의 모델을 업데이트할 필요 없이 여러 기계 학습 모델과 멀티모달 데이터셋에서 효과적으로 작동할 수 있음을 보여주었습니다.



### Layer by Layer: Uncovering Hidden Representations in Language Models (https://arxiv.org/abs/2502.02013)
- **What's New**: 이 논문에서는 중간 층(intermediate layers)의 성능을 분석하여, 언어 모델에서 마지막 층(final layer)보다 더 나은 표현을 제공할 수 있음을 밝혔다. 특히, 32개의 텍스트 임베딩 작업을 통해 중간 층의 표현력 향상 효과를 확인했으며, 이는 전통적인 마지막 층 의존성에 도전하는 발견이다.

- **Technical Details**: 우리는 정보 이론(information theory), 기하학(geometry), 그리고 입력 섭동(input perturbations)에 대한 불변성(invariance)을 기반으로 한 통합 프레임워크를 제안한다. 이 프레임워크는 각 모델 층이 정보 압축(compression)과 신호 보존(signal preservation)을 어떻게 균형 있게 조절하는지를 설명하면서 중간 층의 효과성을 이해하는 데 도움이 된다.

- **Performance Highlights**: 중간 층의 임베딩은 대부분의 아키텍처에서 마지막 층과 비교해 일관되게 더 강력한 기능을 제공하며, 특히 자동회귀 모델의 경우 중간 층의 ‘압축 계곡(compression valley)’ 현상을 보여준다. 이러한 연구 결과는 모델 디자인과 학습 관행에 새로운 방향성을 제공하며, AI 시스템의 Robustness와 정확성을 향상시킬 수 있는 기회를 열어준다.



### Fairness through Difference Awareness: Measuring Desired Group Discrimination in LLMs (https://arxiv.org/abs/2502.01926)
- **What's New**: 이 논문은 알고리즘적 공정성(algorithmic fairness) 개념의 재정의가 필요하다는 주장을 한다. 기존의 공정성 기준은 인종 차별을 무시하는 경향이 있지만, 특정 사회 집단 간의 차이를 인식하는 것이 중요한 여러 상황에서 다름 인식(difference awareness)가 필요하다는 점을 강조한다. 예를 들어, 법적 맥락이나 해악 평가에서 그룹 간의 차별이 중요할 수 있다.

- **Technical Details**: 논문에서는 설명적(descriptive), 규범적(normative), 상관적(correlation) 평가 기준의 세 가지 범주로 차이를 명시적으로 구분한다. 이는 각기 다른 해석과 완화(mitigation) 접근이 필요하다는 것을 말하며, 이러한 구분은 현재의 공정성 평가 기준에서 소홀히 여겨지고 있다는 점을 지적한다. 저자들은 16,000개의 질문으로 구성된 8개의 평가 기준으로 이루어진 벤치마크(Benchmark) 세트를 제시하며, 차이가 인식되는 공정성을 측정할 수 있는 방법을 모색한다.

- **Performance Highlights**: 저자들은 기존의 공정성 기준이 차이 인식에 대한 무관심으로 인해 어떻게 불충분한지를 보여준다. 이 연구는 차이 인식과 맥락 인식(Contextual Awareness)이 공정성의 중요한 개념임을 강조하며, 현재의 편향 완화 전략이 잘못된 결과를 초래할 수 있음을 입증한다. 또한, 다양한 모델에서의 실험 결과를 통해 차이 인식이 공정성의 독립적인 차원임을 보여준다.



### Training and Evaluating with Human Label Variation: An Empirical Study (https://arxiv.org/abs/2502.01891)
- **What's New**: 이 논문에서는 Human label variation (HLV)을 다루기 위한 새롭고 체계적인 방법론을 제시하고 있습니다. 기존의 HLV 모델 평가 방안을 메타 평가하는 접근은 부족했기에, 이 연구는 부드러운 평가 메트릭스(soft evaluation metrics)를 제안하고, 이를 통해 모델이 어떻게 인식할 수 있는지를 탐색합니다. 실험 결과, 분리된 주석(disaggregated annotations)이나 소프트 레이블(soft labels)로 훈련하는 것이 좋은 성능을 나타낸 것으로 밝혀졌습니다.

- **Technical Details**: HLV는 전통적인 단일 진실 값을 가정하는 머신 러닝 모델에 도전하며, 이로 인해 모델 훈련 및 평가가 간단하지 않게 됩니다. 이 연구는 부드러운 세트 이론(fuzzy set theory)에서 영감을 얻어, 인공지능 모델의 성과를 평가하는 새로운 방법론을 제시합니다. 또한, 제안된 소프트 메트릭스는 미분 가능하여 훈련 목표로 활용될 수 있습니다.

- **Performance Highlights**: 실험에서는 총 6개의 데이터 세트를 사용하여, 제안된 소프트 메트릭스가 인간의 선호(human preference)와 가장 높은 상관관계를 보인다는 것을 확인했습니다. 그러나 상관관계가 제한적임을 나타내어, HLV에 적합한 평가 메트릭스의 필요성을 강조하고 있습니다. 이 연구는 또한 기존의 다양한 HLV 메트릭스와의 비교를 통해, 보다 나은 평가 접근 방식을 탐색하고 있습니다.



### Soup-of-Experts: Pretraining Specialist Models via Parameters Averaging (https://arxiv.org/abs/2502.01804)
- **What's New**: 이번 논문에서는 Soup-of-Experts라는 혁신적인 아키텍처를 제안합니다. 이 아키텍처는 테스트 시 다양한 도메인 가중치를 통해 최소한의 계산 비용으로 모델을 인스턴스화 할 수 있습니다. 이를 통해 여러 전문화 데이터셋을 활용할 수 있으며, 기존의 모델에 비해 재훈련이 필요 없습니다.

- **Technical Details**: Soup-of-Experts는 전문가 파라미터의 은행을 구성하며, 입력 도메인 가중치에 따라 선형 결합된 형태로 하나의 모델을 구성합니다. 이 모델은 사전 훈련된 다양한 도메인에서 여러 전문가를 미리 훈련하여 가치를 극대화합니다. 학습된 선형 결합 계수는의 배리에 의해 결정되며, 이 과정은 효율적인 데이터 샘플링기법을 통해 이루어집니다.

- **Performance Highlights**: 실험 결과, Soup-of-Experts는 다양한 언어 모델링 작업에서 작은 전문 모델을 신속하게 얻는 데 유리하다는 점을 보여줍니다. 사전 훈련된 110M 모델을 Redpajama 데이터셋에서 훈련한 후 16개 도메인에서 전문화하여 고성능을 달성했습니다. 이 아키텍처는 특히 다양한 전문 모델을 신속하게 전송해야 하는 경우에 유용합니다.



### CTC-DRO: Robust Optimization for Reducing Language Disparities in Speech Recognition (https://arxiv.org/abs/2502.01777)
- **What's New**: 최근의 딥 러닝 모델들은 높은 전반적인 성능을 달성하지만 특정 하위 그룹(subgroup)에서 일관되게 저조한 성능을 보이는 문제가 있습니다. 이를 해결하기 위해 제안된 group distributionally robust optimization (group DRO) 방식은 최악의 그룹 손실을 최소화하는 데 도움이 됩니다. 그러나 손실 함수가 그룹 간 성능 차이를 제대로 반영하지 못할 때 실패할 수 있습니다. 본 논문에서는 이러한 문제점을 해결하기 위해 CTC-DRO를 새롭게 제안합니다.

- **Technical Details**: CTC-DRO는 기존의 group DRO 목표를 일반화하여 높은 손실 그룹에 대한 가중치 업데이트를 매끄럽게 처리하는 기법입니다. 이를 통해 일관되게 손실이 높은 그룹에 대한 과도한 강조를 방지하며, 입력 길이에 맞춘 배치를 사용하여 CTC의 확장성 문제를 완화합니다. 이 방법은 여러 언어 세트를 사용한 다국어 자동 음성 인식(multilingual automatic speech recognition)에서 평가되었습니다.

- **Performance Highlights**: CTC-DRO는 ML-SUPERB 2.0 벤치마크의 다섯 개 언어 세트에서 group DRO와 CTC 기반 모델들을 일관되게 초과 달성하며 성능을 개선했습니다. 특히, CTC-DRO는 최악의 성능을 보이는 언어에서 오류율을 최대 65.9%까지 줄이고, 모든 언어의 평균 오류율도 최대 47.7% 감소시켰습니다. CTC-DRO는 자동 음성 인식에 적용할 때 최소한의 계산 비용으로 그룹 간 격차를 줄일 수 있는 잠재력을 가지고 있습니다.



### ACECODER: Acing Coder RL via Automated Test-Case Synthesis (https://arxiv.org/abs/2502.01718)
Comments:
          9 pages, 1 figure, 7 tables

- **What's New**:  이 논문에서는 코드 모델 훈련을 향상시키기 위해 자동화된 대규모 테스트 케이스 합성을 활용합니다. 특히, 기존 코드 데이터에서 (question, test-cases) 쌍을 생성하는 파이프라인을 설계하여 신뢰할 수 있는 보상 데이터 부족 문제를 해결합니다. 이 연구는 코드 생성 모델에서 강화 학습(RL)의 잠재력을 강조합니다.

- **Technical Details**:  'AceCode-89K'라는 대규모 검증 가능한 코드 훈련 데이터셋을 구축했습니다. 이 데이터셋은 기존 SFT 데이터셋에서 코드 문제를 수집하고, GPT-4o-mini를 이용하여 문제를 LeetCode 스타일로 재작성하며, 그에 따른 20개의 테스트 케이스를 생성하는 과정을 포함합니다. 최종적으로 89K 질문과 300K 테스트 케이스가 쌍으로 구성된 데이터셋을 생성하였습니다.

- **Performance Highlights**:  강화 학습을 통해 Llama-3.1-8B-Ins와 Qwen2.5-Coder-7B-Ins 모델의 성능이 평균 각각 10점과 5점 개선되었습니다. 최적화 단계가 80단계에 불과한 HumanEval-plus에서 25%의 개선을 보였으며, 이는 모델의 파라미터 조정을 통해 달성되었습니다. 이 연구는 코드 생성 모델에서 RL 훈련의 큰 잠재력을 보여줍니다.



### QLESS: A Quantized Approach for Data Valuation and Selection in Large Language Model Fine-Tuning (https://arxiv.org/abs/2502.01703)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)의 메모리 요구 사항을 줄이기 위한 새로운 방법인 	extbf{QLESS}(Quantized Low-rank Gradient Similarity Search)를 제안합니다. QLESS는 고차원 그래디언트를 로우-랭크(LoRA) 기반 무작위 프로젝션을 통해 저차원으로 압축하고, 이를 저비트폭 표현으로 양자화하여 메모리 효율적인 데이터 평가 및 선택을 가능하게 합니다. 실험 결과, QLESS는 LESS와 유사한 데이터 선택 성능을 보이면서 메모리 사용량을 최대 16배 줄이는 데 성공했습니다.

- **Technical Details**: QLESS는 그래디언트 데이터 저장소에 absmax 기반 양자화를 통합함으로써, 고정밀 그래디언트를 양자화된 저비트 그래디언트로 대체합니다. 이를 통해 메모리 요구 사항을 크게 줄이고도 영향 계산의 무결성을 유지할 수 있습니다. 연구에서는 무작위 프로젝션을 통해 압축된 그래디언트를 정규화하여 데이터 평가 과정의 신뢰성을 확보합니다.

- **Performance Highlights**: 1비트 양자화된 그래디언트가 16비트 그래디언트와 유사한 성능을 자주 내는 것으로 나타났습니다. 이는 그래디언트 기반 데이터 평가에서 정밀도와 효능 간의 전통적인 가정을 도전하며, 극단적 양자화 하에서도 영향 계산의 견고성에 대한 흥미로운 질문을 제기합니다. QLESS는 효율성과 확장성을 결합하여 LLM의 지침 조정에 실용적인 솔루션을 제공합니다.



### Multimodal Inverse Attention Network with Intrinsic Discriminant Feature Exploitation for Fake News Detection (https://arxiv.org/abs/2502.01699)
- **What's New**: 이번 연구는 마르텐츠 적인 모드별 특성을 활용한 'Multimodal Inverse Attention Network (MIAN)' 프레임워크를 제안합니다. 기존 접근법들은 모드 특수한 표현과 불일치 특징을 충분히 활용하지 못했으나, MIAN은 뉴스 콘텐츠의 본질적인 특징을 탐색하여 허위 뉴스 검출을 향상시킵니다. MIAN은 지역 내-지역 상호 작용과 지역 내-전역 상호 작용을 통해 다층 학습 모듈을 도입하여 향상된 단일 모드 표현을 생성합니다.

- **Technical Details**: MIAN은 모드 간 상호 작용 모듈을 통해 조관계 메커니즘을 사용하여 정제된 단일 모드 표현 간의 의존 관계를 설정합니다. 또한, 역 주의 메커니즘을 통해 각 모드에서 모순된 패턴과 의미적 편차를 강조함으로써 불일치 특징을 명확히 추출하는 데 중점을 둡니다. 이 구조는 뉴스 텍스트와 이미지 간의 명확한 관계를 탐색하고 단일 모드 및 다중 모드 내 본질적 식별 정보를 활용합니다.

- **Performance Highlights**: 광범위한 실험에서 MIAN은 여러 벤치마크 데이터 세트에서 기존의 최신 방법들에 비해 유의미한 성능 향상을 보였습니다. MIAN은 다양한 실제 사례에서 허위 뉴스의 탐지를 효과적으로 개선하며, 사회적 안전을 위한 솔루션을 제공합니다. 이 연구는 공공 정보의 신뢰성과 무결성을 보장하기 위한 자동화된 허위 뉴스 검출 기법의 개발을 촉진하는 데 기여하고 있습니다.



### Automated Extraction of Spatio-Semantic Graphs for Identifying Cognitive Impairmen (https://arxiv.org/abs/2502.01685)
Comments:
          To appear in ICASSP 2025

- **What's New**: 본 연구에서는 기존의 수작업으로 진행되던 Content Information Units (CIUs) 태깅 작업을 자동화하여, Cookie Theft 그림을 기반으로 한 spatio-semantic graph의 자동 추출 방법을 제안합니다. 이 자동화된 접근법은 인지 손상 평가에서 시각적 의미 경로를 자동적으로 특성화할 수 있는 가능성을 제시합니다. 연구 결과, 자동으로 생성된 spatio-semantic graph가 인지 손상 유무를 효과적으로 구분할 수 있음을 확인하였습니다.

- **Technical Details**: 이 연구에서 사용된 spatio-semantic graph는 CIUs의 위치와 시간 순서를 시각적으로 표현하는 그래프 이론적 구조입니다. 각 CIU는 Cookie Theft 그림의 픽셀 좌표에 매핑되며, 이 과정을 통해 시각적 경로를 구성합니다. 연구팀은 NetworkX 툴킷을 사용해 이러한 좌표들을 기반으로 노드와 엣지를 구성하여 시각적 의미 경로를 시각화합니다.

- **Performance Highlights**: 자동으로 생성된 spatio-semantic graph는 인지 손상이 있는 화자와 없는 화자를 구별하는 데 있어 유의미한 차이를 보였습니다. 통계적 분석 결과, 자동화된 방법으로 유도된 특징들이 수작업 방법과 유사한 결과를 생성했으나, 임상 집단 간 차이를 더 뚜렷하게 나타내는 것으로 확인되었습니다. 이는 자동화된 접근법이 인지 손상 평가를 위한 임상적 언어 모델 개발에 크게 기여할 수 있음을 시사합니다.



### LIBRA: Measuring Bias of Large Language Model from a Local Contex (https://arxiv.org/abs/2502.01679)
Comments:
          Paper accepted by ECIR 2025

- **What's New**: 이 연구에서는 기존의 대형 언어 모델(LLM) 편향 평가의 두 가지 주요 한계를 극복하기 위해 Local Integrated Bias Recognition and Assessment Framework (LIBRA)를 제안합니다. LLM의 편향을 평가할 때 지역 특수성을 반영한 데이터 세트를 사용하고, LLM이 훈련 데이터에서 접하지 못한 낯선 단어에 따라 나타나는 비적절한 결과를 다루는 방안을 포함합니다. 이를 통해 뉴질랜드 맥락에서 360,000개 이상의 테스트 사례로 구성된 데이터세트를 개발하며, Enhanced Idealized CAT Score (EiCAT)라는 새로운 평가 지표도 제안합니다.

- **Technical Details**: LIBRA 프레임워크는 지역 코퍼스를 활용하여 데이터세트를 생성하고 편향을 측정하며, 편향 테스트 중 LLM의 지식 경계를 초과하는 단어로 인한 환상적 결과를 최소화하는 데 초점을 맞춥니다. 이 연구에서는 StereoSet의 삼중체 구조를 활용하여 목표 사회 집단을 설명하는 유사한 문장으로 구성된 160,000개 이상의 테스트 케이스를 갖춘 데이터셋을 생성했습니다. EiCAT 점수는 전통적인 iCAT 점수와 LLM의 지식 경계를 초과하는 점수(bbs)를 통합해, 확률 분포를 분석하여 편향을 측정합니다.

- **Performance Highlights**: 연구 결과, BERT 계열, GPT-2, Llama-3 모델은 다양한 맥락에서 지역 단어를 잘 이해하지 못하는 경향을 보였습니다. 특히 Llama-3는 더 높은 편향을 보여주었지만, 다양한 문화적 맥락에도 보다 나은 반응을 보였습니다. 이 연구를 통해, LLM의 편향 평가를 위한 새로운 접근법이 제시되었으며, 이를 통해 향후 LLM의 공정성과 신뢰성을 높일 수 있는 기반이 마련되었습니다.



### Efficiently Integrate Large Language Models with Visual Perception: A Survey from the Training Paradigm Perspectiv (https://arxiv.org/abs/2502.01524)
Comments:
          28 pages, 3 figures

- **What's New**: 이번 논문에서는 Vision-Language Large Language Models (VLLMs)에서 시각 모달리티의 통합을 위한 새로운 교육 패러다임을 제시합니다. 연구의 주요 초점은 MoDALITY Integrators (MIs)와 함께 두 가지 주요 조정 과정인 Single-stage Tuning과 Two-stage Tuning입니다. 이를 통해 연구자들은 LLMs의 매개변수 효율성을 유지하면서 성능을 향상시킬 수 있는 방법을 제시하고 있습니다.

- **Technical Details**: 본 연구에서는 다양한 LLM 아키텍처와 함께 Parameter-Efficient Adaptation (PEA) 기법을 분류합니다. Single-stage Tuning과 Two-stage Tuning 방식은 각각 고유한 효율성 동기를 가지고 있으며, Direct Adaptation 방식은 자원의 효율적인 소비를 강조합니다. 각 교육 패러다임에 대해 독특한 매개변수 최적화 전략을 제시하며, 다양한 비전 인코더의 아키텍처를 포괄적으로 다룹니다.

- **Performance Highlights**: 논문은 34개의 VLLMs를 조사한 결과, 최신 VLLMs에서의 Two-stage Tuning과 함께 여러 모델의 성능을 비교 분석합니다. 실험 결과는 Direct Adaptation 접근 방식의 주요 개발과 효과성을 입증하며, 다양한 분야에서 비전 모달리티 통합의 효율성을 끌어올리는 데 기여할 수 있는 지침을 제공합니다. 연구의 결과는 연구자와 실무자들이 LLMs에 비전 모달리티를 효율적으로 통합하는 데 유용한 통찰력을 제공합니다.



### Scaling Embedding Layers in Language Models (https://arxiv.org/abs/2502.01637)
- **What's New**: 이번 논문에서는 SCONE (Scalable, Contextualized, Offloaded, N-gram Embedding)이라는 새로운 방법을 제안하여 입력 임베딩 레이어의 확장을 통한 언어 모델의 성능 향상을 다룹니다. 이 방법은 원래의 어휘(vocabulary)를 유지하면서도 자주 발생하는 n-그램(n-grams)에 대한 임베딩을 추가하여 입력 토큰의 맥락적(contextualized) 표현을 제공합니다. 이러한 기술로 인해 인퍼런스(inference) 시 벡터 간의 디코딩 비용을 증가시키지 않고도 성능을 개선할 수 있습니다.

- **Technical Details**: SCONE은 입력 임베딩의 크기를 단순히 증가시키는 대신, 미리 정의된 f-그램(frequent n-grams)의 컨텍스트 변형을 통해 각 토큰을 보강합니다. 이는 대규모의 임베딩 테이블을 구성할 수 있게 해주며, 출력 레이어의 계산 비용에 영향을 주지 않고도 맥락화된 표현을 활용할 수 있습니다. 이러한 접근 방식은 인퍼런스 시간에서 FLOPS를 고정 유지하면서 f-그램 임베딩의 수를 늘리거나 f-그램 모델을 스케일 할 수 있는 두 가지 새로운 전략을 가능하게 합니다.

- **Performance Highlights**: SCONE은 1.9B 매개변수(parameter)를 갖는 기반 모델을 뛰어넘어 다양한 데이터셋에서 성능을 발휘하였으며, 인퍼런스 시간에서 필요한 FLOPS는 절반으로 줄였습니다. 실험 결과에 따르면, SCONE을 사용하는 1B 매개변수 모델은 비슷한 성능을 유지하면서도 인퍼런스 비용을 크게 줄일 수 있음을 보여주었습니다. 이러한 결과는 임베딩 레이어의 효과적인 확장을 통해 모델 성능을 더욱 강화할 수 있음을 시사합니다.



### Lifelong Sequential Knowledge Editing without Model Degradation (https://arxiv.org/abs/2502.01636)
- **What's New**: 이 논문에서는 기존의 파라미터 수정 지식 편집 방법의 한계를 극복하고, 모델 성능 저하 없이 10,000개의 순차적 지식 편집을 가능하게 하는 ENCORE 방법을 제안합니다. 지식 편집 과정에서 발생하는 오버피팅(Overfitting)과 편집된 행렬의 비대칭적 노름 성장(Norm Growth)의 문제를 해결하는 데 중점을 두었습니다. ENCORE는 이러한 문제를 완화시켜 모델의 다운스트림 성능을 유지하면서, 지난 방법들보다 61%에서 64%까지 빠른 속도를 자랑합니다.

- **Technical Details**: 본 연구는 'locate-then-edit' 방법으로 알려진 파라미터 수정 지식 편집 메커니즘을 두 단계의 미세 조정(fine-tuning) 과정으로 설명합니다. 첫 번째 단계에서는 경량 회귀(guided descent)를 사용하여 적절한 활성화 벡터를 찾고, 두 번째 단계에서는 최소 제곱 손실(minimum squared loss) 함수를 사용하여 선택한 MLP(다층 퍼셉트론) 행렬의 가중치를 업데이트합니다. 새로운 조기 중단 방법(Most-Probable Early Stopping, MPES)을 도입하여 편집된 사실의 오버피팅을 줄이고, Frobenius 노름 제약조건을 추가하여 행렬의 증가를 조절합니다.

- **Performance Highlights**: ENCORE는 GPT2-XL, Llama-2-7B, Llama-3-8B에서 이전의 locate-then-edit 방법들보다 월등하게 성능을 높였으며, 특히 Llama3-8B 위에서 MEMIT보다 61%, AlphaEdit보다 64% 빠른 속도로 편집을 수행합니다. 이로 인해 모델의 성능 저하 없이 10,000개의 순차적 지식 편집이 가능해지며, 이는 대규모 지식 편집 분야의 새로운 이정표를 설정합니다.



### LLM-TA: An LLM-Enhanced Thematic Analysis Pipeline for Transcripts from Parents of Children with Congenital Heart Diseas (https://arxiv.org/abs/2502.01620)
Comments:
          Accepted by GenAI for Health Workshop @ AAAI 2025, Philadelphia

- **What's New**: 본 연구에서는 의료 분야의 고위험 환경에서의 Thematic Analysis (TA) 과정에서 대형 언어 모델(Large Language Models, LLM)의 활용 가능성을 모색합니다. 특히, 선천성 심장 질환인 Anomalous Aortic Origin of a Coronary Artery (AAOCA)의 부모와의 인터뷰 전사를 분석하기 위해 LLM-Enhanced Thematic Analysis (LLM-TA) 파이프라인을 제안합니다. 이 파이프라인은 비용 효율적인 최첨단 LLM(GPT-4o mini), LangChain 및 청크 기법을 통합하여 TA 프레임워크에 따라 데이터 분석의 효율성과 정확성을 높입니다.

- **Technical Details**: 연구자는 42명의 부모가 참여한 9개의 포커스 그룹 세션에서 나온 탈-식별화된 전사를 사용하여 LLM-TA 파이프라인을 개발하였습니다. 기존의 LLM-기반 TA 방법과 달리, 이 파이프라인은 초기 코드 생성부터 주제 식별까지의 과정에서 LLM을 통해 자동화를 시도하며, 이를 위해 전사를 최대 1,500단어의 작은 청크로 나누어 분석의 섬세함을 높였습니다. 각 청크는 대화의 맥락을 유지하며, LLM이 보다 세부적이고 정확한 코드를 생성을 용이하도록 합니다.

- **Performance Highlights**: 연구 결과, LLM-TA 파이프라인은 기존 LLM-보조 TA 방법들에 비해 주제 정확성, LLM 평가, 전문가 검토에서 모두 우수한 성능을 보였습니다. 분석자 작업량을 줄이면서 협업을 통해 도메인 전문가와 함께 더 나은 결과를 도출할 수 있는 가능성을 지니고 있습니다. 그러나, 이 시스템은 아직 인간 수준의 품질을 달성하지는 못했지만, 유망한 개선 가능성을 보이고 있습니다.



### Large Language Models Are Human-Like Internally (https://arxiv.org/abs/2502.01615)
Comments:
          19 pages

- **What's New**: 최근의 인지 모델링 연구에서는 대형 언어 모델(large language models, LMs)이 인간의 읽기 행동과 잘 맞지 않다는 결과를 보고하였습니다. 본 논문에서는 메커니즘 해석 가능성(mechanistic interpretability)의 관점에서 이 주제를 재고하며, LMs의 최종 레이어(final layers)에만 집중한 이전 연구의 결론이 잘못된 것임을 주장합니다. 내부 레이어에서 도출한 다음 단어 확률이 작은 LMs보다 인간의 문장 처리 데이터와 더 잘 일치하는 사실을 발견했습니다.

- **Technical Details**: 이 연구는 LLMs의 내부 레이어를 사용하여 단어 확률(surprisal)을 계산하는 새로운 방법론을 제안합니다. 초기 레이어는 빠른 주목 지속 시간(first-pass gaze durations)에 더 잘 맞고, 후속 레이어는 N400 뇌 전위와 MAZE 처리 시간과 같은 더 느린 신호와 잘 일치하는 경향을 보입니다. 이러한 연구 결과는 LLMs의 다양한 내부 레이어가 서로 다른 시간을 반영한다는 통설과 일치합니다.

- **Performance Highlights**: 본 연구는 대형 LMs가 실제적으로 인간의 행동과 신경 생리학적 데이터를 모델링하는 데 있어 우수한 인지적 가능성을 제공함을 제안합니다. 즉, LLMs 내부에는 인지적으로 그럴듯한 더 얕은 레이어들이 '중첩(nested)'되어 존재합니다. 이러한 발견은 인지 모델링과 메커니즘 해석 가능성의 통합을 강조하며, 인간 측정치에 대한 레이어별 정렬에 대한 관심을 촉진합니다.



### Breaking Focus: Contextual Distraction Curse in Large Language Models (https://arxiv.org/abs/2502.01609)
- **What's New**: 최근의 연구에 따르면, Large Language Models (LLMs)에서 새로운 취약점인 Contextual Distraction Vulnerability (CDV)가 발견되었습니다. 이 취약점은 의미론적으로 일관된 비필수적 맥락이 질문에 추가될 때 모델의 성능을 저하시킵니다. 연구진은 이러한 CDV 예제를 자동으로 생성할 수 있는 효율적인 트리 기반의 검색 방법론을 제안하였습니다.

- **Technical Details**: 이 연구에서는 CDV를 체계적으로 분석하기 위해 원본 질문의 의미를 유지하면서 의미론적으로 유효한 맥락상의 방해 요소를 자동 생성하는 세 단계로 구성된 프레임워크를 도입했습니다. 우선, 필터링 메커니즘을 통해 변형하기 쉬운 샘플을 선별한 후, 트리 기반 검색 전략을 이용하여 조작된 샘플을 생성합니다. 이 과정은 컴퓨테이션적 효율성을 확보하면서, CDV의 예제를 정밀하게 만들어내도록 설계되었습니다.

- **Performance Highlights**: 제안된 방법은 최신 LLM들에서 평균 약 45%의 성능 저하를 유발하는 CDV 예제를 성공적으로 생성하였습니다. 실험을 통해 다양한 데이터셋에서 효과적인 CDV 질문 예제를 만들어냈으며, 타겟된 훈련 방식을 통해 모델의 강건성을 향상시키는 방법도 연구했습니다. 이러한 결과는 CDV가 LLM 개발에서 중요한 도전 과제임을 강조합니다.



### FutureVision: A methodology for the investigation of future cognition (https://arxiv.org/abs/2502.01597)
- **What's New**: 이 논문은 다중 양식의 의미 분석(multimodal semantic analysis)과 시선 추적 실험Protocol(eye-tracking experimental protocol)을 결합하여 미래 시나리오의 의사소통 이해 과정에서의 인지적 노력을 조사하는 방법론을 제시합니다. 실험을 통해 참가자의 시선 고정 패턴이 가상의 광고 속 긍정성과 반사실성(counterfactuality)을 평가하는 과정에서 어떻게 변화하는지를 분석하였습니다. 이 연구는 미래 시나리오를 이해하는 데 있어 기저 공간의 단절이 인지적 과부하를 증가시킨다는 가설을 지지하는 초기 결과를 보여줍니다.

- **Technical Details**: 이 연구는 'Persistence of the Base' 가설을 바탕으로 하며, 다중 양식 프레임 의미 주석(multimodal frame-semantic annotation)과 시선 추적 기술을 통해 인지적 노력을 측정합니다. 연구에서는 참가자가 미래 시나리오에 대한 광고를 평가하면서 수집된 안구 운동을 기록하였고, 이 과정에서 시각적 요소와 언어적 요소 간의 관계를 분석하였습니다. 시선 추적(glasses)을 통해 수집된 데이터에 대해 세밀하게 분석이 진행되어 미래 시나리오 이해에 필요한 인지적 노력의 정도를 평가할 계획입니다.

- **Performance Highlights**: 파일럿 실험 결과에 따르면, 먼 미래와 비관적인 시나리오에 대해 참가자들의 시선 고정 시간과 운동 패턴이 더욱 복잡하게 나타났습니다. 이는 기저 공간의 단절이 미래 시나리오의 해석에 있어 인지적 부담을 증가시키는 요인이라는 hypothesis 를 지지합니다. 이 연구는 다차원적인 의미를 동원하는 광고 기획 및 미래 시나리오 예측에 있어 새로운 통찰을 제공하며, 광고 분야에서의 실용적인 적용 가능성을 보여줍니다.



### ReGLA: Refining Gated Linear Attention (https://arxiv.org/abs/2502.01578)
Comments:
          Accepted by NAACL 2025 (main)

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 복잡한 언어 모델링 작업에서 뛰어난 성과를 보여주고 있습니다. 그러나 이러한 모델은 소프트맥스 어텐션의 제곱 계산 복잡도로 인해 상당한 컴퓨팅 및 저장 요구사항을 가지고 있습니다. 이 연구에서는 Gated Linear Attention 모듈의 성능에 크게 영향을 미치는 세 가지 주요 구성 요소인 피처 맵(Feature Map), 정규화(Normalization), 그리고 게이팅 메커니즘(Gating Mechanism)을 분석했습니다.

- **Technical Details**: 본 연구는 Gated Linear Attention 메커니즘의 다양한 구성 요소를 최적화하여 빠른 추론과 훈련 과정을 고려하였습니다. 피처 매핑 함수(Feature Mapping Function)를 통해 이전 연구들이 간과한 중요한 문제들을 해결하고, 훈련 과정을 안정화하기 위한 정규화 층(Normalization Layer)을 통합했습니다. 또한 게이팅 메커니즘의 포화 현상(Saturation Phenomenon)을 조사하고 이를 정제하는 모듈을 추가하여 성능을 개선했습니다.

- **Performance Highlights**: 우리는 광범위한 실험을 통해 우리의 아키텍처가 기존 Gated Linear Attention 메커니즘을 초월하는 성능을 발휘한다는 것을 입증했습니다. 특히, 훈련을 처음부터 시작하거나 연속적 사전 훈련(Continual Pre-training)을 통해 후속 선형화(Post-linearization)하는 작업에서 성능을 개선했습니다. 이 연구는 긴 시퀀스를 처리하는 데 요구되는 도전 과제를 해결하기 위해 새로운 아키텍처를 제안합니다.



### Visual Theory of Mind Enables the Invention of Writing Systems (https://arxiv.org/abs/2502.01568)
Comments:
          Currently under submission to a non-archival conference, published here with permission from organizers

- **What's New**: 본 연구에서는 인지적 및 문화적 프로세스를 통해 초기 상징 체계의 발달을 조명하기 위해 에이전트가 시각적 마음 이론을 활용하여 행동을 소통하는 모델을 제시합니다. 이 연구는 기존의 비자연적인 방법론의 한계를 극복하고, 인지의 발달 진화를 더욱 잘 이해할 수 있도록 합니다. 이러한 과정에서  'Signification Game'이라 불리는 다중 에이전트 강화 학습 테스트베드를 개발하여 의사소통의 구체적인 기제를 실험적으로 살펴봅니다.

- **Technical Details**:  'Signification Game'은 초기 인류의 커뮤니케이션 도구의 제한을 모방한 행동 공간에서 에이전트가 의사소통을 배울 수 있는 설정을 제공합니다. 에이전트는 환경에서 행동을 취할 때 사용되는 정보 처리 시스템을 기반으로 통신 신호를 해석해야 하며, 이 과정은 그들의 상호작용을 통해 점진적으로 발전합니다. 실험 설계는 동물들의 원시적 의사소통 방식의 이론을 기반으로 하여, 조건된 S-R 행동의 구조를 밝혀냅니다.

- **Performance Highlights**: 연구 결과, 단순한 정책 구조로도 에이전트가 보상을 극대화하며 의사소통을 배울 수 있음을 보여주었으며, 이러한 방식에는 한계가 존재함을 발견했습니다. 특히 인지 신호의 한계로 인해 특정 참조 개념을 소통하는 데 어려움이 발생하는 signification gap이 있음을 밝혔습니다. 그러나 시각적 마음 이론을 활용한 추론 모델이 이러한 gap을 극복할 수 있도록 지원하며, 에이전트들이 빠른 시간 내에 대다수의 참조 개념을 전달할 수 있는 가능성을 열어줍니다.



### Scalable Language Models with Posterior Inference of Latent Thought Vectors (https://arxiv.org/abs/2502.01567)
- **What's New**: 본 논문에서는 Latent-Thought Language Models (LTMs)라는 새로운 언어 모델 계열을 제안합니다. LTMs은 명시적인 latent thought vectors를 도입하여 이들이 latent space에서 명시적인 prior model을 따릅니다. 이러한 latent vectors는 Transformer decoder의 autoregressive 생성 과정을 안내하며, 이는 기존의 큰 언어 모델들과는 다른 특징을 가지고 있습니다.

- **Technical Details**: LTMs은 고전적인 variational Bayes 프레임워크 내에서 이중 속도 최적화 프로세스를 사용합니다. 이에는 latent vectors의 posterior 분포에 대한 지역적 변동 매개변수의 빠른 학습과, 전역 decoder 매개변수의 느린 학습이 포함됩니다. 이러한 구조는 인간의 인지 과정에서의 빠른 에피소드 학습과 느린 도식 학습의 상호 작용을 반영합니다.

- **Performance Highlights**: LTMs는 기존의 autoregressive 모델 및 diffusion 모델에 비해 샘플 및 매개변수 효율성에서 뛰어난 성능을 보입니다. 이 모델은 validation perplexity와 zero-shot 언어 모델링에서도 기존 성과를 크게 초과하는 결과를 보여줍니다. 특히, LTM-Large는 조건부 및 비조건부 텍스트 생성에서도 뛰어난 성능을 발휘합니다.



### Massive Values in Self-Attention Modules are the Key to Contextual Knowledge Understanding (https://arxiv.org/abs/2502.01563)
- **What's New**: 본 논문은 거대 언어 모델(LLMs)의 맥락적 지식 이해가 어떻게 이루어지는지를 심층적으로 조사합니다. 특히, 주의 쿼리(Q)와 키(K)의 특정 영역에서 집중된 방대한 값이 발생하는 패턴을 발견했으며, 이러한 값이 모델의 파라메트릭 지식 회복보다는 맥락적 지식 이해에 중요한 역할을 한다고 주장합니다. 또한, Rotary Positional Encoding (RoPE)이 이러한 집중된 값을 초래한다는 점을 강조합니다.

- **Technical Details**: 연구에서는 블록별로 독립적으로 작동하는 자기 주의(heads of self-attention)에서 방대한 값이 특정 인덱스에 집중된다는 사실을 발견했습니다. Q와 K의 계산에서 이러한 방대한 값은 독특하게 군집화되지만, V에서는 이러한 패턴이 존재하지 않는 것을 확인했습니다. 이러한 현상은 RoPE가 적용된 모델에서만 나타나며, 이는 각 주의 레이어에서 파동의 빈도에 따라 영향을 미치는 방식으로 작용합니다.

- **Performance Highlights**: 방대한 값을 무시하고 양자화 기법을 적용하면 맥락적 이해 과제의 성능이 현저히 떨어지며, 특히 영화 리뷰 분석(IMDB)이나 특정 결정적 과제에서의 성능 저하가 두드러지는 것으로 나타났습니다. 반면, 방대한 값을 다루는 양자화 방법이 LLM의 성능을 보다 잘 유지하는 것으로 검증되었습니다. 이러한 발견은 새로운 양자화 전략 설계에 직접적인 통찰을 제공합니다.



### What is a Number, That a Large Language Model May Know It? (https://arxiv.org/abs/2502.01540)
Comments:
          16 pages, 8 figures

- **What's New**: 본 연구에서는 대형 언어 모델(LLMs)이 숫자와 문자열 표현을 혼합하여 학습하는 방식을 탐구합니다. 숫자가 문맥에 따라 다르게 해석될 수 있는 상황에서 이러한 이중성(immediate duality)이 모델의 대표성(representation)와 관련된 다양한 결과를 초래하는지에 대한 의문을 제기합니다. 새로운 유사성 기반 프롬프트 기법을 통해 LLMs가 숫자 쌍 간의 유사성을 어떻게 판단하는지 확인하고, 그 과정에서 나타나는 혼합된 표현을 탐구합니다.

- **Technical Details**: 저자들은 심리학 및 인지 과학에서 유래한 유사성 판단(simiarity judgments) 기법을 사용하여 최신 LLMs에서 숫자 쌍에 대한 유사성을 측정했습니다. 이 방법을 통해 Levenshtein 편집 거리(Levenshtein edit distance)와 Log-Linear 숫자 거리(numerical Log-Linear distance)의 조합이 모델들 간의 숫자 표현을 효율적으로 설명함을 보여줍니다. 이러한 접근은 모델의 내부 구조를 직접 참조하지 않고도 사용 가능하며, 프로빙(probing) 기법과 결합하여 잠재 임베딩(latent embedding)의 구조를 평가할 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 이 연구에서 얻어진 중요한 발견은 모든 모델이 숫자 및 문자열 표현 간의 혼합 효과를 경험한다는 것입니다. 특정 맥락(int() vs. str())을 통해 이 혼합을 줄일 수는 있지만 완전히 제거할 수는 없음을 확인했습니다. 이러한 현상이 실제 결정 시나리오에서도 나타나, 문자열 편향(string bias)이 잘못된 결과를 초래하는 방식에 대한 통찰을 제공합니다. 이러한 결과는 LLM이 숫자와 문자열의 표현 간의 긴장을 어떻게 navigat할 필요가 있는지에 대한 중요한 정보를 제공합니다.



### CondAmbigQA: A Benchmark and Dataset for Conditional Ambiguous Question Answering (https://arxiv.org/abs/2502.01523)
- **What's New**: 이번 연구에서는 Conditional Ambiguous Question-Answering (CondAmbigQA)라는 새로운 벤치마크를 도입하였습니다. 이 벤치마크는 200개의 모호한 질문과 조건 인식 평가 지표를 포함하고 있으며, QA 작업의 모호성을 해결하기 위해 질문의 문맥적 제약이나 추정 조건을 강조합니다. 기존 연구의 한계를 극복하고 인간 편견을 최소화하기 위한 새로운 접근법을 제시하며, 인간의 주관성을 줄이기 위한 상호작용적 주석 프로세스를 개발했습니다.

- **Technical Details**: CondAmbigQA의 핵심은 질문 모호성을 명확히 해주는 ``conditions''을 식별하고 표현하는 것입니다. 각 질문에 대해 20개의 Wikipedia 조각을 검색하여 조건과 답변을 주석화합니다. LLM(대형 언어 모델)을 활용하여 문맥을 종합하고 조건-답변 쌍을 정제하는 과정을 통해 주석화의 일관성과 신뢰성을 증대시킵니다. 이 과정은 기존 방법보다 시간 효율성을 높이고, 인간 주석자가 소요하는 작업 시간을 80%까지 감소시킵니다.

- **Performance Highlights**: 실험 결과, 조건을 고려하여 답변을 생성한 모델이 일반 RAG 접근법보다 20% 더 높은 성과를 보였습니다. 명시적 조건을 제공할 경우에는 추가적으로 5%의 성과 향상이 확인되었습니다. 특히, 더 큰 모델이 조건 준수와 답변 품질 측면에서 우수한 성능을 보여, 모델 크기가 QA 과제에서 모호성을 해결하는 데 중요한 역할을 한다는 점이 확인되었습니다. 또한, 인용 생성 통합을 통해 더 큰 모델에서의 답변 신뢰성을 향상시키는 중요한 가능성을 보여주었습니다.



### Hybrid Machine Learning Model for Detecting Bangla Smishing Text Using BERT and Character-Level CNN (https://arxiv.org/abs/2502.01518)
Comments:
          Conference Name: 13th International Conference on Electrical and Computer Engineering (ICECE 2024)

- **What's New**: 이 논문에서는 Bangla smishing 텍스트를 탐지하기 위한 새로운 하이브리드 기계 학습 모델을 제안합니다. 이 모델은 Bidirectional Encoder Representations from Transformers (BERT)와 Convolutional Neural Networks (CNNs)를 결합하여 문자 수준 분석을 향상시키며, 정상 메시지(Normal), 광고(Promotional), 그리고 smishing SMS를 구분하는 다중 클래스 분류를 다룹니다. 기존의 이진 분류 방식과 달리, 이 접근법은 BERT의 문맥적 임베딩과 CNN의 문자 수준 특징을 통합함으로써 탐지 정확도를 향상시킵니다.

- **Technical Details**: 연구에서 사용된 데이터셋은 2,287개의 Bangla SMS 메시지로, 정상, smishing, 광고의 세 가지 클래스로 분류됩니다. 각 메시지는 'label' 열로 분류되며, 'text' 열에는 Bangla 내용이 포함되어 있습니다. 본 연구에서 SMS 데이터를 정제하고 토큰화하는 과정은 BERT 토크나이저를 사용하며, 문자 수준의 보다 세부적인 분석을 위해 커스텀 토크나이저도 구현되었습니다.

- **Performance Highlights**: 제안된 모델은 smishing 탐지에서 98.47%의 정확도를 달성하며, 기존의 분류기보다 높은 성능을 보입니다. 비정상적 메시지의 탐지에서 높은 정밀도(precision)와 재현율(recall)을 기록하였으며, 모든 카테고리에서 강력한 성능을 보여줍니다. 이 결과는 기존의 이진 분류 시스템의 한계를 극복하며, Bangla SMS 유형의 다양성을 반영한 것이 주요 성과입니다.



### Memorization Inheritance in Sequence-Level Knowledge Distillation for Neural Machine Translation (https://arxiv.org/abs/2502.01491)
- **What's New**: 이번 연구에서는 Teacher Neural Machine Translation (NMT) 모델의 instance-level memorization이 학생 모델에게 어떻게 전달되는지를 분석합니다. 연구 결과, 학생 모델은 원본 훈련 데이터를 직접 보지 않았음에도 불구하고 기존 모델보다 더 많은 memorization을 발생시킨다고 밝혔습니다. 이러한 결과는 SeqKD (Sequence-level Knowledge Distillation) 접근법 아래 새로운 실패 모드를 유발하며, 학생 모델의 신뢰성과 안전성에 대한 경각심을 불러일으킵니다.

- **Technical Details**: 논문은 SeqKD 모델에서 Teacher와 Student 간의 메모리 전이를 정의하고, 이를 메모리 측정 지표와 함께 분석합니다. 연구에서는 학생 모델이 teacher의 memorization 행동을 상속받는 방식을 통계적으로 설명하고, 다양한 데이터 서브그룹에서 발생하는 오류 및 새로운 memorization 모드를 조사합니다. Adaptive-SeqKD와 같은 개선된 접근법을 통해 memorization을 줄이고 hallucination을 감소시키는 방법도 제안합니다.

- **Performance Highlights**: 실험은 WMT20, WMT22, WMT23의 다양한 언어 쌍과 Transformer 네트워크 규모를 사용하여 진행되었습니다. 결과적으로 MemFreeSeqKD라는 새로운 Teacher finetuning 알고리즘이 제안되었으며, 이를 통해 학생 모델의 신뢰성을 크게 향상시키면서도 평균 성능을 유지할 수 있음을 보여줍니다. 이 연구는 SeqKD 기술을 사용할 때 발생할 수 있는 잠재적인 문제에 대한 주의를 촉구합니다.



### FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Mod (https://arxiv.org/abs/2502.01472)
Comments:
          Under Review

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 기밀 정보 처리 문제를 해결하기 위한 새로운 방법론인 FALCON(Fine-grained Activation manipuLation by Contrastive Orthogonal uNalignment)을 제안합니다. 기존의 기계 학습에서의 기억 소거(machine unlearning) 기술들은 지식 내에서의 특정한 부분을 분리하기 위한 비효율적인 방법들이 존재했습니다. FALCON은 정보 이론적 접근 방식을 통해 모델의 파라미터 선택을 효율적으로 수행하고, 대전제 기법을 이용하여 효과적인 지식 분리를 달성합니다.

- **Technical Details**: FALCON은 상대적 정보(mutual information)를 사용하여 데이터 간의 의존성을 평가하고, 이를 기반으로 파라미터 선택과 최적화 지침을 제공합니다. 두 가지 주요 메커니즘을 통해 다중 도메인 지식 소거를 수행하며, 이를 방향성 대전제의 적용과 그라디언트 직교 투영(gradient orthogonal projection)으로 이루어집니다. 이 방식은 파라미터의 부분 수정만으로도 효율적인 지식 소거를 달성할 수 있도록 설계되었습니다.

- **Performance Highlights**: FALCON은 광범위한 실험을 통해 다른 기계 학습 모델보다 뛰어난 지식 소거 효과를 보여주었습니다. 이 방법은 모델의 유용성을 유지하면서도 지식 회복 시도에 대한 저항성이 뛰어난 결과를 도출합니다. 또한 FALCON은 다양한 모델에 걸쳐 확장 가능성을 가지고 있으며, 실질적으로 접근할 수 없는 훈련 데이터에 대한 완전한 접근 없이도 작동할 수 있습니다.



### Towards Safer Chatbots: A Framework for Policy Compliance Evaluation of Custom GPTs (https://arxiv.org/abs/2502.01436)
- **What's New**: 이 논문에서는 OpenAI의 Usage Policies에 따라 Custom GPT들을 자동으로 평가하기 위한 확장 가능한 프레임워크를 제시합니다. 특히, 모델을 검색하고 데이터를 수집하는 자동화 기능과 정책 카테고리에 맞춤화된 레드 팀 프롬프트 생성기, 그리고 전체적인 준수를 분석하는 LLM-as-a-judge 기법이 통합되었습니다. 이 프레임워크는 수작업 주석 처리된 데이터셋을 검증하여 정확성을 보장하며, 782개의 Custom GPT를 대규모 연구를 통해 평가했습니다.

- **Technical Details**: 프레임워크는 세 가지 주요 모듈로 구성되어 있습니다: 1) Custom GPT Interactor는 모델을 자동으로 검색하고 메타데이터를 수집합니다; 2) Red-Teaming Prompts Generator는 각 Custom GPT에 대한 맞춤형 프롬프트를 생성합니다; 3) Compliance Assessment 모듈은 LLM-as-a-judge 기법을 사용하여 응답을 분석하고 정책 준수 여부를 평가합니다. 이 과정은 Orchestrator 모듈에 의해 조정되며, JSON 형식으로 결과를 구조화하여 평가 결과를 제공합니다.

- **Performance Highlights**: 연구에서 58.7%의 모델이 비준수로 나타났으며, 이는 GPT 스토어의 검토 및 승인 프로세스의 취약점을 드러냅니다. 특히, 사용자 맞춤화보다는 기본 모델에서 상속받은 행동이 비준수 문제의 주 원인으로 분석되었습니다. 이 결과는 다른 챗봇 플랫폼 및 정책 영역에도 확장 가능한 접근 방식을 제공하며, LLM 기반 시스템의 안전성을 향상시킬 가능성을 제시합니다.



### Emergent Stack Representations in Modeling Counter Languages Using Transformers (https://arxiv.org/abs/2502.01432)
- **What's New**: 이 연구는 트랜스포머 모델이 카운터 언어(Counter Languages)를 학습하는 방식과 이를 통해 스택과 유사한 구조를 내포한 내부 표현을 발전시키는지를 분석합니다. 기존의 연구는 이러한 언어 구조에 대한 이해가 부족했으나, 본 논문은 4개의 카운터 언어로 훈련된 모델을 통해 이러한 과정을 탐구합니다. 특히, 모델이 다음 토큰을 예측하는 과정에서 스택 깊이에 대한 내부 표현을 학습하는 것을 보여줍니다.

- **Technical Details**: 트랜스포머의 내부 구조를 분석하는 기법인 프로빙 분류기(Probng Classifier)를 사용하여 카운터 언어에서 학습된 모델의 표현을 탐지합니다. 이 방법론은 중간 활성화(Intermediate Activations)를 소규모 분류기에 연결하여 특정 속성을 예측하는 방식을 채택합니다. 이러한 방식은 모델의 정보가 특정 속성과 연결되는지를 검증하는 데 케이스의 의미 있게 적용될 수 있습니다.

- **Performance Highlights**: 트랜스포머 모델이 특정 카운터 언어를 효과적으로 학습하고, 스택 구조와 유사한 형상을 내포한 것을 발견했습니다. 이 연구는 언어 모델의 내부적 알고리즘을 이해하는데 기여하며, AI 안전성 및 정렬 문제 해결에 중요한 통찰을 제공합니다. 또한, 카운터 언어를 통한 연구가 언어 모델링의 방향성을 더욱 명확히 하고 있음을 나타냅니다.



### Annotation Tool and Dataset for Fact-Checking Podcasts (https://arxiv.org/abs/2502.01402)
Comments:
          Accepted as resource paper in TheWebConf 2025

- **What's New**: 이 논문은 팟캐스트의 팩트 체크를 위한 새로운 접근 방식을 제시합니다. 실시간으로 팟캐스트를 재생하면서 주석을 달 수 있는 도구를 통해 팟캐스트에서 주장되고 있는 내용의 검증이 가능해집니다. 이 도구는 OpenAI의 Whisper를 포함한 고급 전사 모델을 활용하여 다국어 변환기 모델의 성능을 향상시키는 데 도움을 줍니다.

- **Technical Details**: 제안된 도구는 팟캐스트를 청취하면서 전사 오류를 수정하고, 주장을 식별할 수 있도록 실시간 주석 달기를 지원합니다. 오픈 소스 구성 요소로 구축되어 있으며, Whisper ASR를 통한 전사 및 F-Coref를 통한 공통 참조 해소 기능과 같은 다양한 기능을 제공합니다. 이 도구는 90개 이상의 언어를 지원하며, 정확한 주장의 식별과 검증을 가능하게 합니다.

- **Performance Highlights**: 531개의 에피소드에서 수집된 전사본과 주석 데이터셋은 여러 팟캐스트 에피소드를 대상으로 합니다. XLM-RoBERTa와 같은 변환기 모델을 미세 조정하여 다국어 주장 감지 및 입장 분류 작업의 유용성을 보여주며, GPT-4와 같은 LLM과 비교했을 때의 성능 차이를 확인합니다. 향후 대규모 주석 작업과 사용자 친화적인 주석 시스템을 개발할 예정입니다.



### Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models (https://arxiv.org/abs/2502.01386)
- **What's New**: 본 논문은 RAG 시스템을 겨냥한 주제 지향적 적대적 여론 조작 공격을 다루며, 기존의 사실적 단일 쿼리 조작을 넘어서는 실질적인 접근 방식을 제안합니다. 새로운 공격 방식인 Topic-FlipRAG를 통해 여러 관련 쿼리에 걸쳐 여론을 변화시키는 두 단계의 조작 파이프라인을 구현합니다. 실험 결과, 이러한 공격이 사용자의 정보 인식에 중대한 영향을 미치는 것으로 나타났으며, 현재의 완화 방법으로는 이러한 공격에 효과적으로 대응할 수 없음을 강조합니다.

- **Technical Details**: Topic-FlipRAG는 두 단계의 공격 방법론으로 구성되어 있으며, 첫 번째 단계에서는 LLM의 내부 의미 지식을 활용하여 목표 문서에 대한 은밀한 적대적 수정이 이루어집니다. 두 번째 단계에서는 공개 소스 신경 순위 모델(NRM)로부터의 그래디언트를 사용하여 주제별 적대적 트리거를 생성합니다. 이러한 방법은 문서의 의미적 수준에서 섬세한 조작을 가능하게 하며, RAG 모델의 출력에서 특정 주제에 대한 여론을 효과적으로 변화시킵니다.

- **Performance Highlights**: 실험에서 Topic-FlipRAG는 네 가지 도메인에서 0.5의 평균 입장 변화를 기록하며, 다른 기준선 방법보다 현저히 우수한 성능을 보여줍니다. 또한, 사용자 실험 결과에 따르면, 사용자가 독성 RAG 시스템과 상호작용한 후 논란 있는 주제에 대한 입장이 16% 이상 변화하는 것으로 나타났습니다. 이러한 결과는 Topic-FlipRAG가 정보의 제시 및 인식 방식에 영향을 미칠 수 있다는 것을 확인시켜 주며, 강력하고 적응 가능한 방어 전략의 필요성을 부각시킵니다.



### Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations (https://arxiv.org/abs/2502.01349)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 제품 추천 시스템이 혁신적으로 변화했지만, 이들이 적대적인 조작에 취약하다는 점이 상업적 응용에 있어 큰 도전 과제가 되고 있습니다. 본 연구에서는 인간의 심리 원리를 활용하여 상품 설명을 조작해 이러한 적대적 조작을 탐지하기 어렵게 만드는 새로운 접근 방식을 논의합니다. 우리는 인지 편향이 LLM과 인간의 구매 행동에 미치는 영향을 분석하며, 다양한 LLM을 대상으로 한 실험을 통해 그들이 추천자로서의 보안 취약성을 드러냅니다.

- **Technical Details**: 대형 언어 모델(LLMs)과 인지 편향간의 관계는 인공지능과 심리학의 중요한 교차점을 형성합니다. 연구에서는 LLM이 사전 훈련 과정에서 인간의 인지 편향을 어떻게 내재화하는지를 탐구하며, 이는 LLM이 추천 시스템에서 공정성과 신뢰성을 저해하는 요인으로 작용할 수 있음을 보여줍니다. 또한, 추천 제품의 설명을 변경해 LLM의 반응을 조작하는 방법을 다루며, 이러한 공격이 여러 제품과 LLM에서 일관성을 가지고 수행될 수 있음을 증명합니다.

- **Performance Highlights**: 본 연구에서 우리는 LLM이 인지 편향에 의해 불리하게 영향을 받을 수 있음을 보여주었으며, 이는 공격에 대한 방어가 어렵다는 점을 강조합니다. LLM 추천 시스템에서의 인지 편향 연구는 그들의 전반적인 결정 과정과 신뢰성을 향상시키기 위한 중요한 기초 자료를 제공합니다. 최종적으로 실험 결과는 제품 추천의 품질과 신뢰성을 높이기 위한 새로운 전략적 접근을 제안하며, 추천 엔진의 공정성 확보를 위한 방안을 모색합니다.



### AlignVLM: Bridging Vision and Language Latent Spaces for Multimodal Understanding (https://arxiv.org/abs/2502.01341)
- **What's New**: 이번 논문에서는 비전-언어 모델(VLM) 간의 정렬 문제를 해결하기 위해 새로운 방법인 AlignVLM을 제안합니다. AlignVLM은 시각적 특징을 LLM의 텍스트 임베딩에 대한 확률 분포로 매핑하여, 심층 연결 방식 없이 또한 신뢰성 있는 시각적 특징을 보장합니다. 이는 기존의 다수의 연결 방법보다 개선된 성능을 보여주기 위해, LLM의 사전 훈련된 언어 특성을 활용하는 방식으로 이루어집니다.

- **Technical Details**: AlignVLM은 세 가지 주요 구성 요소로 이루어진 VLM 아키텍처를 기반으로 하며, 이 구성요소는 원시 이미지를 처리하는 비전 인코더, 텍스트에 대해 사전 훈련된 대형 언어 모델(LLM) 및 시각적 특징을 LLM의 의미 공간으로 매핑하는 연결 모듈입니다. AlignVLM의 핵심 기술은 각 시각적 특징을 LLM의 텍스트 임베딩의 볼록 조합으로 제한하는 것입니다. 이 과정에서 LLM에 이미 내재된 언어적 우선 사항을 활용하여, 향상된 정렬을 통해 노이즈 또는 분포 외부 입력의 위험을 줄입니다.

- **Performance Highlights**: 실험 결과, AlignVLM은 문서 이해 작업에서 이전 연결 방법들보다 월등한 성능을 발휘하였습니다. AlignVLM은 1B에서 8B까지의 다양한 모델 크기에서는 강력한 효율성과 안정성을 보여주며, 문서의 텍스트 내용을 보다 정확하게 매핑하는 데 효과적입니다. 본 연구는 시각적 및 언어적 컨텐츠의 융합을 최적화하여 다중 모드 문서 이해 작업에서 최고의 성능을 달성한 것에 기여합니다.



### Main Predicate and Their Arguments as Explanation Signals For Intent Classification (https://arxiv.org/abs/2502.01270)
- **What's New**: 이 논문은 대화형 에이전트의 의도 분류에 대한 새로운 기술과 방법론을 제시합니다. 특히, 주 동사(main predicate)와 그 인자(arguments)를 활용하여 의도 분류 데이터셋의 텍스트 샘플에 자동으로 단어 수준의 설명 신호를 추가하는 방법을 개발했습니다. 이를 통해 ATIS와 SNIPS 데이터셋을 사용하여 21,000개의 샘플을 포함한 새로운 데이터셋을 만들었습니다.

- **Technical Details**: 이 연구에서는 의도 분류에서 주 동사와 이의 직접 객체가 의도 신호로 작용할 수 있다는 관찰을 바탕으로 새로운 은주석(silver annotation) 기법을 도입했습니다. 저자들은 LIME 및 통합 그래디언트(integrated gradients)와 같은 포스트 호크(post-hoc) 해설 기법을 사용하는 심층 학습 모델과 언어 모델을 분석하였습니다. 이로 인해 설명 가능성(predictability) 및 신뢰성(faithfulness) 지표에서 모델의 성능이 저하된다는 결과를 보였습니다.

- **Performance Highlights**: 이 연구의 결과는 훈련 과정에서 설명 신호에 더 집중하도록 모델을 유도할 시, 설명 가능성 지표에서 3-4%의 Token F1 점수가 개선된다는 것을 보여줍니다. 또한, ATIS와 SNIPS 데이터셋을 대상으로 한 실험을 통해 기존의 모델보다 더 향상된 성능을 보였습니다. 이 시스템은 특히 자원이 부족한 중소기업의 챗봇에 적합하여, 보다 효율적인 의도 분류를 가능하게 합니다.



### OphthBench: A Comprehensive Benchmark for Evaluating Large Language Models in Chinese Ophthalmology (https://arxiv.org/abs/2502.01243)
- **What's New**: 이번 연구에서는 중국 안과 분야에서 LLM(대형 언어 모델)의 성능 평가를 위한 전문 벤치마크인 OphthBench를 소개합니다. 이 벤치마크는 5개의 주요 시나리오로 구성되며, 총 9개의 작업과 591개의 질문이 포함되어 있습니다. 이를 통해 LLM의 실제 의료 적용 가능성을 종합적으로 평가하고 향후 연구 방향을 제시합니다.

- **Technical Details**: OphthBench는 교육, 분류(triage), 진단(diagnosis), 치료(treatment), 예후(prognosis)와 같은 5가지 핵심 시나리오를 바탕으로 구축되었습니다. 각 시나리오에는 다양한 질문 유형이 포함되며, 이를 통해 LLM의 정확성과 능력을 평가할 수 있습니다. 특히, 중국의 문화적 맥락과 의료 시스템을 반영하여 설계되었습니다.

- **Performance Highlights**: 39개의 인기 있는 LLM을 대상으로 한 평가 결과, 현재 모델의 능력이 임상에서 요구되는 실제적인 필요와 큰 차이를 보임을 확인했습니다. 이는 LLM의 진화 방향을 제시하는 중요한 통찰로 볼 수 있습니다. 본 연구는 LLM의 잠재력을 열어주는 기반을 마련하고, 안과 항목에서의 개발을 촉진하는 데 기여하고자 합니다.



### On the Robustness of Temporal Factual Knowledge in Language Models (https://arxiv.org/abs/2502.01220)
- **What's New**: 이번 논문은 언어 모델(LM)의 시간적 지식(temporal knowledge) 처리의 견고성(robustness)을 탐구합니다. 특히, LM들이 간단한 사실 진술을 완성하는 데는 능숙하지만, 특정 시간 프레임 내에서만 유효한 시간적 사실 관리에는 한계를 보이는지 검사합니다. 저자들은 여러 개의 사전 훈련된(pretrained) 및 지시 조정된(instruction-tuned) 모델의 성능을 평가하기 위한 통제된 실험을 설계하고, 이 과정에서 위키데이터(Wikidata)에서 인기 있는 사실을 사용하였습니다.

- **Technical Details**: 예를 들어, '미국의 대통령은 ____'이라는 문장에서 적절한 시간적 문맥과 다음과 같은 프롬프트(예: [YEAR], 미국의 대통령은 누구였습니까?)를 통해 LM의 반응을 평가합니다. 저자들은 600개의 인기 있는 시간적 사실과 18개의 LM을 대상으로 실험을 수행하였고, 이중에서 LLaMa3.1-70B 모델이 시간적 사실의 26%만을 견고하게 알고 있음이 밝혀졌습니다. 이렇게 다양한 시간적 변환과 정밀도를 변형하여 LM의 반응을 분석하였습니다.

- **Performance Highlights**: 이 실험 결과, LM들은 서로 다른 시간 정밀도(year, month, day)에서 상이한 성능을 보이며, 특히 더 세부적인 날짜의 정밀도에서 성능이 저하되는 경향이 있습니다. 또한, 지시 조정 모델이 사전 훈련 모델보다 뛰어난 성능을 보여주지 않는다는 점이 발견되었습니다. 이러한 결과는 LM들이 시간적 사실의 견고한 표현을 학습하는 데 한계가 있어, 시간적 지식 기반으로서의 사용에 의문을 제기합니다.



### Modelling change in neural dynamics during phonetic accommodation (https://arxiv.org/abs/2502.01210)
- **What's New**: 이번 연구는 음성의 짧은 기간 내 상대 음성에 따른 조정(Phonetic Accommodation)을 모델링하는 역동적 신경장(dynamic neural field) 방정식을 바탕으로 한 새로운 계산 모델을 제시합니다. 이는 기존의 예증 기반(exemplar-based) 모델과 달리 움직임 계획(Motion Planning) 및 기억의 동역학을 포함하여 음성이 어떻게 변화하는지를 설명합니다.

- **Technical Details**: 연구에서는, 모의 화자(model talker)의 타격(Accent)에 따라 참가자들이 자신들의 발음 습관을 어떻게 조정하는지를 다뤘습니다. 실험에서는 13명의 북부 앵글로 영국인 참가자들이 'bath'와 'strut'이라는 두 개의 모음(vowel)을 주제로 하여 모의 화자의 발음을 따라 말하는 과정을 수행했습니다. 이러한 접근 방식은 복잡한 운동 시너지(motor synergy)가 발음 계획의 중심 역할을 하며, 이는 주의 기반의 동적 필드 모델(dynamic field model)에서 잘 발전되어 왔음을 보여줍니다.

- **Performance Highlights**: 본 연구의 모델은 음성 모방(shadowing) 동안 관찰된 경험적 패턴을 유사하게 재현할 수 있었으며, 이는 음성 변화의 정도가 음성 기계적 신경 다이나믹스(inhibitory memory dynamics)에 의해 조정됨을 나타냅니다. 실험 결과, 참여자는 모의 화자의 발음과의 차이를 줄일 뿐만 아니라, 음성 변화를 포착하여 사후 테스트에서 기초선으로 돌아가는 경향을 보였습니다. 이런 결과는 음성 짧은 기간 내의 조정과 장기적 음성 변화의 관계에 대한 새로운 통찰력을 제공합니다.



### OCR Error Post-Correction with LLMs in Historical Documents: No Free Lunches (https://arxiv.org/abs/2502.01205)
Comments:
          To be published in RESOURCEFUL 2025

- **What's New**: 이번 연구는 Optical Character Recognition (OCR) 시스템의 역사적 문서 전사 오차 수정에 있어 오픈 웨이트 LLMs의 활용 가능성을 평가합니다. 특히, 영어와 핀란드어 데이터셋에서 다양한 전략을 탐색하며, LLMs가 OCR로 생성된 텍스트를 수정하는 데 도움이 될 수 있다는 점에서도 주목받고 있습니다. 연구 결과, 영어에서는 문자 오차율(Character Error Rate, CER)을 줄이는 데 효과적이나, 핀란드어에서는 유의미한 성능을 달성하지 못했습니다.

- **Technical Details**: 본 연구에서는 OCR 오류 수정 작업에서 LLM의 성능을 최적화하기 위해 하이퍼파라미터 최적화, 양자화(quantization), 입력 길이, 출력 후처리 및 새로운 수정 방법을 연구합니다. 특히, 상업 모델 사용이 비용적으로 불가능한 큰 역사적 데이터세트에 대해 오픈 웨이트 LLM을 적합하게 활용하고자 합니다. 제안된 여러 방법론과 LLM의 적용에 따른 장단점을 평가하여 중요한 관리적 도전 과제를 다루고 있습니다.

- **Performance Highlights**: 연구 결과, 영어 데이터셋에는 약 0.04 CER의 경미한 오차에서 주요 성과가 있었으나, 핀란드어의 경우 실용적인 성능은 나타나지 않았습니다. 또한, 현대 LLMs를 통한 OCR 오류 수정이 대규모 역사적 자료 검색에 있어 많은 잠재력을 지니고 있지만, 실질적인 수익성에 대한 한계도 분명히 드러났습니다. 따라서 LLM의 성능을 극대화하기 위해 추가적인 방법론과 데이터 조정이 필요하다는 결론을 내렸습니다.



### COVE: COntext and VEracity prediction for out-of-context images (https://arxiv.org/abs/2502.01194)
Comments:
          Camera-ready version accepted to NAACL 2025 Main Conference

- **What's New**: 이번 연구에서는 COVE라는 새로운 방법을 소개합니다. COVE는 이미지를 올바른 문맥(COntext)으로 예측한 후, 이를 바탕으로 캡션의 진실성(VEracity)을 판단합니다. 기존의 자동 화자 검증 방법들은 두 가지 목표를 명시적으로 다루지 못했으나, COVE는 이 두 작업을 순차적으로 수행아며, 성능에서도 좋은 결과를 보여줍니다.

- **Technical Details**: COVE의 전체 구조는 여섯 단계로 나눌 수 있으며, 첫 세 단계에서는 다양한 증거를 수집하는 방법을 다룹니다. 웹 캡션 및 시각적 엔터티, 위키백과 엔터티 등 각기 다른 출처의 데이터를 수집하여 입력합니다. 그 후, 수집된 증거를 바탕으로 LLM을 사용하여 첫 번째 문맥을 예측하고, 누락된 항목을 위키백과에서 업데이트한 뒤 캡션의 진실성을 판단합니다.

- **Performance Highlights**: COVE는 모든 문맥 항목에서 기존의 최첨단(context prediction SOTA) 모델을 초과하여 성능을 향상시켰습니다. 특히 COVE는 합성 데이터에서 검증 예측 모델들과 경쟁력을 보이며, 실제 데이터에서는 최대 4.5 포인트 더 높은 성능을 보여주었습니다. 이 연구는 예측된 문맥이 특정 이미지에 대한 새로운 캡션을 검증하는 데 재사용 가능하고 해석 가능한 유용한 자료임을 잘 입증하고 있습니다.



### A Single Model Ensemble Framework for Neural Machine Translation using Pivot Translation (https://arxiv.org/abs/2502.01182)
- **What's New**: 이 논문에서는 저자들이 낮은 자원 언어 쌍의 번역 성능을 향상시키기 위해 새로운 기법인 Pivot-based single model ensemble (PivotE)을 제안하고 있습니다. 기존의 앙상블 기법이 여러 모델 학습의 높은 계산 비용과 블랙박스 모델의 한계로 어려움을 겪고 있는 반면, 제안된 방법은 단일 모델을 사용하여 피벗 언어를 통한 후보 생성 및 후속 집계를 수행합니다. 이 접근 방식은 고품질 후보를 생성하고, 최종 번역 성능을 향상시킵니다.

- **Technical Details**: PivotE는 피벗 번역을 활용하여 후보를 생성하고, 이 후보들을 집계하여 최종 번역을 만듭니다. 첫 번째 단계에서는 단일 다국어 NMT 모델을 사용하여 후보를 생성하고, 두 번째 단계에서는 높은 품질의 후보를 선택하여 집계합니다. 이를 통해 다양한 후보를 생성하면서도 계산 비용을 줄일 수 있습니다.

- **Performance Highlights**: 실험 결과, PivotE는 서로 다른 언어 쌍에서 기존의 최첨단 방법들을 지속적으로 초과하는 번역 품질을 보여주었습니다. 고자원 언어의 지식을 전이하여 더 나은 후보를 생성하는 능력 덕분에 원본 문장의 미묘한 의미와 뉘앙스를 효과적으로 전달할 수 있었습니다.



### Joint Localization and Activation Editing for Low-Resource Fine-Tuning (https://arxiv.org/abs/2502.01179)
Comments:
          The code for the method is released at this https URL

- **What's New**: 최근 연구에서 제안된 Joint Localization and Activation Editing (JoLA)은 모델의 특정 컴포넌트를 수정하는 방법을 학습하는 혁신적인 접근법입니다. 기존의 parameter-efficient fine-tuning (PEFT) 방식의 한계를 극복하며, 작은 데이터셋에서도 우수한 성능을 발휘하도록 설계되었습니다. JoLA는 어떤 heads를 수정할지, 보정 방법으로 additive와 multiplicative를 함께 사용할지 등을 학습합니다.

- **Technical Details**: JoLA는 HardConcrete gates와 예상-L0 정규화를 사용하여 모델의 activations를 동적으로 조정합니다. 이 때, JoLA는 intervention을 최소화하여 편집할 컴포넌트를 선택하고, 조정의 유연성을 제공하기 위해 additive 오프셋과 multiplicative 스케일링을 결합합니다. 이 방법은 각 task에 대해 최적의 intervention 파라미터를 학습하여 저자원 환경에서 효과적인 fine-tuning을 가능하게 합니다.

- **Performance Highlights**: 세 가지 benchmark(Task)에 대한 실험 결과 JoLA는 기존 방식보다 일관되게 우수한 성능을 보였습니다. JoLA는 여러 데이터 규모와 모델 크기에 걸쳐 안정적인 결과를 나타내며, 특히 저자원 환경에서 더욱 두드러진 성능을 발휘합니다. 이 연구는 attention heads의 중요성을 강조하며, JoLA가 가장 효과적인 fine-tuning 전략임을 증명합니다.



### Jailbreaking with Universal Multi-Prompts (https://arxiv.org/abs/2502.01154)
Comments:
          Accepted by NAACL Findings 2025

- **What's New**: 본 논문에서는 LLM(JUMP)을 탈옥(jailbreak)하기 위한 보편적인 다중 프롬프트를 최적화하는 기술을 제안합니다. 기존 기법들이 개별 사례에 대한 공격으로만 초점을 맞추었으나, JUMP는 이전의 접근 방식과는 달리 아직 보지 못한 작업(supervised tasks)으로까지 전이 가능한 공격자를 훈련할 수 있는 방법을 소개합니다. 또한, 방어 기술인 DUMP를 제안하여 공격 시나리오뿐만 아니라 방어 시나리오에서도 유용성을 입증합니다.

- **Technical Details**: JUMP는 Beam Search 과정을 활용하여 일련의 적대적 접미사(adversarial suffixes)를 생성합니다. 이 방법은 BEAST 프레임워크를 확장하여 좀 더 일반적인 시나리오를 다루도록 설계되었습니다. 특히, 우리의 알고리즘은 ASR(Attack Success Rate)과 perplexity 간의 균형을 잘 잡아주며, 신중하게 선택된 초기 프롬프트가 이러한 문제를 완화하는데 도움을 줍니다. 궁극적으로 JUMP++는 현재의 최고 기술들보다 성능이 우수함을 보였습니다.

- **Performance Highlights**: 실험 결과, JUMP는 기존의 여러 기술들보다 우수한 성과를 나타냈으며, 방어 시나리오에서도 잘 일반화되었습니다. JUMP*와 JUMP++ 두 가지 버전 모두 각각의 공격 결과에서 우수한 공격 성공률을 기록했습니다. 이러한 성과는 프롬프트 생성 과정에서의 최적화가 LLM의 보안 툴로서의 활용 가능성을 향상시킴을 시사합니다.



### Language Models Prefer What They Know: Relative Confidence Estimation via Confidence Preferences (https://arxiv.org/abs/2502.01126)
- **What's New**: 이번 연구에서는 언어 모델의 절대적 신뢰도 측정의 한계를 극복하기 위해 상대적 신뢰도 추정(relative confidence estimation)을 제안합니다. 기존의 절대적 신뢰도 추정이 제공하는 신뢰도 점수가 너무 선택적이어서 실제 모델 예측의 정확성을 평가하기 어려운 문제를 해결하고자 합니다. 우리는 질문을 서로 비교하여 신뢰도를 판단하게 하는 방법을 통해 보다 신뢰할 수 있는 신뢰도 점수를 제공하는 방법을 제시합니다.

- **Technical Details**: 제안된 상대적 신뢰도 추정에서는 각 질문을 서로 비교하여 모델의 신뢰도를 평가합니다. 이 과정에서 우리는 Elo 점수, Bradley-Terry 모델과 같은 순위 집계(rank aggregation) 방법을 활용하여 질문에 대한 상대적인 신뢰도를 점수로 변환합니다. 실험에서는 GPT-4, GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet, Llama 3.1 405B와 같은 5가지 최신 언어 모델을 사용하여 다양한 STEM, 사회과학 및 상식 질문 응답 작업을 수행했습니다.

- **Performance Highlights**: 실험 결과, 상대적 신뢰도 추정 방법이 절대적 신뢰도 추정 방법보다 신뢰할 수 있는 신뢰도 점수를 지속적으로 제공하는 것으로 나타났습니다. 상대적 신뢰도는 평균 3.5%의 AUC 개선을 보였으며, 자가 일관성(self-consistency) 접근법 대비 1.7%의 성능 향상을 기록했습니다. 전반적으로 상대적 신뢰도 추정은 신뢰도 평가의 새로운 접근 방식을 제시하며, 보다 정확한 답변 검출을 가능하게 합니다.



### Enhancing Aspect-based Sentiment Analysis with ParsBERT in Persian Languag (https://arxiv.org/abs/2502.01091)
- **What's New**: 이번 논문은 페르시아어 텍스트 마이닝에서의 주요 과제를 해결하는 데 초점을 맞추고 있습니다. 연구자들이 직면한 문제인 페르시아어 데이터셋의 부족과 기존 언어 모델의 비효율성을 해결하기 위해 새로운 접근법을 제안합니다. 이 접근법은 페르시아어에 맞춤화된 언어 모델을 더욱 효과적으로 개선하는 것을 목표로 하고 있습니다.

- **Technical Details**: 저자들은 ParsBERT 모델을 활용하여 감정 분석을 수행하며, 이와 관련된 어휘집을 보강하여 정확도를 높였습니다. 연구는 페르시아 웹사이트 'Digikala'에서 추출한 사용자 의견을 분석하여, 세부적인 감정(sentiment)을 이해하는 데 초점을 둡니다. 제안된 방법은 텍스트의 의미론적 능력을 크게 향상시킵니다.

- **Performance Highlights**: 실험 결과는 제안된 방법의 정확도 88.2%와 F1 점수 61.7이라는 우수한 성과를 보여줍니다. 이 연구는 사용자 생성 콘텐츠에서 미세한 감정을 추출하는 데 기여하며, 페르시아어 텍스트 마이닝 분야에서 감정 분석의 효율성과 정확성을 높이는 데 중요한 역할을 할 것입니다.



### Classic4Children: Adapting Chinese Literary Classics for Children with Large Language Mod (https://arxiv.org/abs/2502.01090)
Comments:
          Accepted at NAACL 2025 Findings

- **What's New**: 이번 연구에서는 어린이들이 이해할 수 있도록 중국 문학 클래식을 교묘하게 각색하는 작업인 아동 친화적 문학 각색(Child-Friendly Literary Adaptation, CLA) 과제를 제안합니다. 기존의 대형 언어 모델(LLMs)이 아동의 독서 선호를 효과적으로 반영하지 못하는 점을 개선하기 위해 InstructChild라는 방법을 고안하였습니다. 이 방법은 각색 과정에서 인물의 성격과 내러티브 구조를 반영해 어린이의 흥미를 유도하려는 목적을 가지고 있습니다.

- **Technical Details**: InstructChild는 미세 조정(fine-grained instruction tuning), 가독성 지표(readability metric) 설계, 그리고 미리 보기 디코딩 전략(lookahead decoding strategy)이라는 세 가지 주요 기술로 구성되어 있습니다. 첫째, 인물의 성격과 내러티브 구조를 기반으로 LLM을 조정하여 아동 친화적인 텍스트 생성을 돕습니다. 둘째, 중국어 가독성 지표는 아동이 이해하기 쉬운 문장을 생성하도록 LLM을 안내합니다. 마지막으로, 미리 보기 디코딩 전략은 현재 토큰 선택을 위해 잠재적인 후속 토큰의 영향을 고려합니다.

- **Performance Highlights**: 실험 결과 InstructChild를 적용한 경우 기존 LLM에 비해 자동 및 인간 평가 모두에서 우수한 성능 향상을 나타냈습니다. Classic4Children 데이터셋을 통해 원본 텍스트와 아동 친화적 버전 간의 효과를 평가하며 각색된 텍스트가 아동의 독서 수준에 적합하게 향상된 것을 보여주었습니다. 이 연구는 향후 연구를 위한 토대를 제공하며, 코드와 데이터셋은 공개하여 추가적인 연구를 촉진할 계획입니다.



### Knowledge Synthesis of Photosynthesis Research Using a Large Language Mod (https://arxiv.org/abs/2502.01059)
Comments:
          17 pages, 6 figures

- **What's New**: 최근 생물학적 데이터 분석 도구와 대규모 언어 모델(LLMs)의 발전으로 식물 과학 연구에서 AI를 활용할 새로운 가능성이 열렸습니다. 이러한 연구에서 기존 LLM의 한계를 극복하기 위해 OpenAI의 GPT-4o 모델을 기반으로 하는 광합성 연구 보조 도구(prag, Photosynthesis Research Assistant)인 PRAG가 제안되었습니다. PRAG는 RAG(retrieval-augmented generation) 기술과 프롬프트 최적화를 통해 정확한 과학적 정보를 제공하도록 설계되었습니다.

- **Technical Details**: PRAG는 OpenAI의 GPT-4o 모델을 기반으로 하며, 자동화된 피드백 루프와 벡터 데이터베이스를 사용하여 광합성 관련 쿼리에 대한 응답의 정확성과 관련성을 향상시켰습니다. 이 시스템은 RAG 모델을 구현하여 외부 데이터베이스에서 관련 문서를 검색하고, 그에 따라 응답을 생성하는 구조를 갖추고 있습니다. 또한, RAG Evaluator와 Prompt Reviser를 통해 지속적인 피드백과 개선을 수행합니다.

- **Performance Highlights**: PRAG는 과학적 글쓰기와 관련된 다섯 가지 지표에서 평균 8.7% 개선을 보여주었으며, 출처 투명성은 25.4% 증가했습니다. PRAG의 과학적 깊이와 분야 커버리지는 기존의 광합성 연구 논문과 비교할 때 유사한 수준을 보였으며, 사용된 지식 그래프를 통해 63%와 39.5%의 키 엔티티 일치를 달성했습니다. 이를 통해 PRAG는 광합성 연구 및 광범위한 식물 과학 분야에서 데이터 분석 및 예측 기능을 향상시킬 수 있습니다.



### PARA: Parameter-Efficient Fine-tuning with Prompt Aware Representation Adjustmen (https://arxiv.org/abs/2502.01033)
Comments:
          accepted by ACL-2024

- **What's New**: 이 논문은 새로운 PEFT(파라미터 효율적 미세 조정) 기법인 PARA(Prompt Aware Representation Adjustment)를 소개합니다. 이 기법은 각 Transformer 레이어에 경량 벡터 생성기(vector generator)를 통합하여 입력 프롬프트에 반응하는 벡터를 생성하며, 이를 통해 모델의 은닉 표현(hidden representation)을 조정합니다. PARA는 이전 PEFT 방법들과 비교하여 효율성과 성능을 모두 개선한 것으로 평가됩니다.

- **Technical Details**: PARA는 LLMs(대형 언어 모델)에 직접적으로 은닉 표현을 수정하는 방식을 사용하여 조정 벡터를 생성합니다. 이 조정 벡터는 입력 프롬프트의 은닉 상태를 입력으로 받아들여 이를 바탕으로 은닉 상태를 변경합니다. 또한, PARA는 Lightweight bottleneck architecture로 구성된 벡터 생성기(VG)를 도입하여, 각 Transformer 레이어 앞에서 프롬프트에 따라 적절한 조정 벡터를 생성합니다.

- **Performance Highlights**: PARA는 다양한 실험을 통해 이전의 PEFT 기준을 지속적으로 초과 성능을 보여주었으며, 특히 Latency가 중요한 멀티 테넌트 환경에서 LoRA보다 현저히 빠른 속도를 기록하였습니다. 이 기법은 동일한 조정 매개변수 한계 내에서 성능의 우수성을 입증하였으며 실질적 산업 적용 가능성을 갖추고 있습니다.



### Knowing When to Stop: Dynamic Context Cutoff for Large Language Models (https://arxiv.org/abs/2502.01025)
Comments:
          Project Website: this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(Large Language Models, LLMs)이 입력 문맥을 비효율적으로 처리하는 문제를 해결하기 위해 'dynamic context cutoff'라는 새로운 방식을 제안합니다. 이 방법은 LLM이 작업에 관련된 정보를 충분히 획득하면 프로세스를 자가 종료하도록 돕습니다. 특히, 특정 attention heads가 'sufficiency signals'를 인코딩하여 중요한 정보의 처리가 완료되었는지를 예측할 수 있는 점이 새롭게 제시되었습니다.

- **Technical Details**: 연구에서 사용된 'sufficiency signals'는 경량 분류기를 통해 감지할 수 있으며, 이를 통해 모델의 내부 이해도가 외부 압축 휴리스틱(heuristics)이 아닌 프로세스 필요성을 결정할 수 있음을 밝혔습니다. 실험은 6개의 QA 데이터셋(최대 40K tokens)을 사용하여 진행되었고, 세 모델 계열(LLaMA, Qwen, Mistral, 1B0-70B)을 통해 검증되었습니다.

- **Performance Highlights**: 실험 결과, 평균 1.33배의 토큰 감소를 달성하면서 정확도는 1.3% 향상되었습니다. 또한, 동일한 수준의 토큰 감소율에서도 다른 문맥 효율성 방법보다 더 나은 성능을 보여주었으며, 작은 모델은 sufficiency 감지를 위해 프로빙(probing)이 필요했지만, 큰 모델은 프롬프트(prompting)를 통해 본질적인 자가 평가 능력을 보이는 emergent scaling 현상을 관찰했습니다.



### MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs (https://arxiv.org/abs/2502.00997)
Comments:
          Accepted by NAACL 2025 Main

- **What's New**: 최근 대규모 언어 모델(LLMs)이 특정 영역에서 성공을 거두면서, 이러한 전문 모델을 통합하여 Mixture-of-Experts (MoE) 모델로 병합하는 방법에 대한 관심이 증가하고 있습니다. 이 연구에서는 전문가 모델 간의 파라미터 간섭(parameter interference)을 해결하고, 서로 다른 아키텍처를 가진 전문가들을 효과적으로 통합할 수 있는 새로운 병합 기법을 제시합니다. 실험을 통해 제안된 방식이 기존의 방법들보다 성능을 개선하고, MoE 병합의 실용성을 확장함을 보여줍니다.

- **Technical Details**: 연구는 다양한 도메인에서 사전 학습된 밀집 전문가 모델들을 효율적으로 병합하는 방법을 제안합니다. 이 방법은 파라미터 간섭을 줄이고, MoE 수정 없이 도메인 특화된 전문가에게 토큰 시퀀스를 라우팅하는 휴리스틱을 통한 접근 방식을 포함하고 있습니다. 논문에서는 또한 서로 다른 아키텍처의 전문가를 결합하여 동적으로 토큰 시퀀스를 적절한 전문가로 라우팅하는 새로운 방법을 개발하였습니다.

- **Performance Highlights**: 제안된 방법은 수학적 추론, 프로그래밍 및 일반 지식 벤치마크에서의 실험을 통해 기존 최첨단 방법보다 뛰어난 성능을 보였습니다. 또한, MoE 병합 후 추가적인 세부 조정에 필요한 자원을 줄이는 등의 장점을 가지며, 다양한 전문가 모델을 활용할 수 있는 가능성을 열었습니다. 이 연구는 MoE 합치기의 효율성을 높이고 더욱 다양한 응용을 가능하게 합니다.



### Self-supervised Analogical Learning using Language Models (https://arxiv.org/abs/2502.00996)
- **What's New**: 본 연구에서는 대형 언어 모델(LLM)의 일관성 문제를 해결하기 위해 자가 감독(Self-supervised) 추상적 유사 학습 프레임워크인 SAL을 제안합니다. SAL은 인간의 유추 과정을 모방하여 잘 해결할 수 있는 케이스에서의 고품질 기호 솔루션을 다른 드문 사례로 전이하는 훈련을 진행합니다. 이 방법을 통해 모델들은 고차원적이며 추상적인 추론 과정을 이해하게 되어, 일관성을 높일 수 있습니다.

- **Technical Details**: SAL은 두 가지 새로운 감독 추출 방법인 개념화(conceptualization)와 단순화(simplification)를 사용하여 자가 감독 신호를 수집합니다. 개념화 방법은 원래 질문과 동일한 고차원 솔루션을 공유하는 유사 질문들을 찾고, 이를 통해 성공적으로 답할 수 있는 질문들의 기호 솔루션을 수집하는 방식입니다. 단순화 방법은 원래 질문의 하위 루틴을 공유하는 질문들을 찾아 이를 통해 고차원 솔루션을 구성합니다.

- **Performance Highlights**: SAL을 통한 학습 후 모델들은 StrategyQA, GSM8K, HotpotQA와 같은 다양한 추론 벤치마크에서 기본 언어 모델보다 2%에서 20% 향상된 성능을 기록했습니다. 또한, SAL은 프로그램적 추론 방식을 통해 더 높은 해석 가능성과 제어 가능성을 제공합니다. 연구 결과와 자료는 발표 후 배포될 예정입니다.



### ChartCitor: Multi-Agent Framework for Fine-Grained Chart Visual Attribution (https://arxiv.org/abs/2502.00989)
- **What's New**: 이 논문은 ChartCitor라는 다중 에이전트 프레임워크를 제안하여 차트 질문-답변 작업을 개선합니다. 기존의 LLM(대형 언어 모델)은 신뢰할 수 없는 응답을 생성하는 경향이 있었으나, ChartCitor는 차트 이미지에서 단서 정보를 식별하여 결과를 향상시킵니다. 이 시스템은 여러 LLM 에이전트를 조정하여 데이터를 구조화하고, 답변을 재구성하며, 증거를 탐색하는 방식으로 작동합니다.

- **Technical Details**: ChartCitor는 다음과 같은 주요 에이전트로 구성됩니다: 차트에서 구조화된 데이터 테이블을 추출하는 Chart2Table Extraction Agent, 답변을 재구성하는 Answer Reformulation Agent, 테이블 데이터를 이해하기 위한 Entity Captioning Agent, 그리고 관련 테이블 셀을 찾아내는 LLM Re-ranking Agent입니다. 이 시스템은 LLM을 활용하여 차트의 비주얼 요소를 식별하고, 이를 바탕으로 증거를 제공하는 바운딩 박스를 생성합니다.

- **Performance Highlights**: ChartCitor는 다양한 차트 유형에서 기존 기준선을 초과하는 성능을 보여줍니다. 사용자 연구 결과, ChartCitor는 LLM 지원 차트 QA의 설명 가능성을 높여 사용자의 신뢰를 증가시키는 데 기여함을 시사합니다. 사용자들은 이러한 시스템을 통해 전문적인 결과를 보다 신속하게 검증하고 생산성을 높일 수 있음을 알게 되었습니다.



### PlotGen: Multi-Agent LLM-based Scientific Data Visualization via Multimodal Feedback (https://arxiv.org/abs/2502.00988)
- **What's New**: 이번 논문에서는 PlotGen이라는 새로운 다중 에이전트 프레임워크를 제안하며, 이로써 사용자 요구에 맞는 과학 데이터 시각화를 자동 생성하는 방법을 모색하고 있습니다. 이 시스템은 여러 개의 LLM 기반 에이전트를 통해 복잡한 사용자 요청을 처리하고, 중간 결과물을 반복적으로 교정하여 정확한 그래프를 생성합니다. PlotGen은 Prompting 기술을 사용하여 사용자 요구사항을 구체적인 실행 단위로 분해하며, 이를 통해 기술적 전문성이 부족한 사용자도 고급 정보 그래픽을 생성할 수 있도록 돕습니다.

- **Technical Details**: PlotGen은 (1) Query Planning Agent, (2) Code Generation Agent, (3) Numeric Feedback Agent, (4) Lexical Feedback Agent, (5) Visual Feedback Agent로 구성된 다섯 개의 unique multimodal feedback agents로 이루어져 있습니다. Query Planning Agent는 복잡한 사용자 요청을 단계별 실행 가능 작업으로 분해하며, Code Generation Agent는 사용자의 데이터를 바탕으로 pseudocode를 실행 가능한 Python 코드로 변환합니다. 각 Feedback Agent는 시각적, 언어적 및 수치적 피드백을 통해 잘못된 부분을 수정하고 최종적으로 사용자가 요구하는 시각화를 제작합니다.

- **Performance Highlights**: 실험 결과, PlotGen은 기존의 강력한 기준선 모델보다 4-6% 높은 성능을 보여 MatPlotBench 데이터셋에서 10-12%의 개선을 이루었습니다. 사용자 조사에서도 PlotGen은 시각화 생성에 대한 신뢰성을 높였으며, 초보 분석가들이 디버깅 시간을 단축시킴으로써 생산성을 증대시킬 수 있음을 확인했습니다. 이러한 성장은 LLM 기반의 시각화 생성 기술 향상에 기여할 것으로 기대됩니다.



### RandLoRA: Full-rank parameter-efficient fine-tuning of large models (https://arxiv.org/abs/2502.00987)
Comments:
          To appear at the International Conference on Learning Representations (ICLR) 2025

- **What's New**: 이 논문에서는 RandLoRA라는 새로운 기법을 소개합니다. 이 방법은 Low-Rank Adaptation (LoRA) 방식의 한계를 극복하며 전체 차원의 업데이트(full-rank updates)를 가능하게 합니다. 특히, 비훈련 가능한 임의의 랜덤 매트릭스(random matrices)의 선형 조합을 학습하여 매개변수 효율성을 높입니다.

- **Technical Details**: RandLoRA는 매개변수의 수를 줄이기 위해 고정된 랜덤 매트릭스에 대각 스케일링 매트릭스(diagonal scaling matrices)를 적용하여 최적화(optimization)합니다. 이러한 방식은 훈련 중에 매개변수(parameter)와 메모리(memory) 효율성을 유지하면서도 낮은 차원의 한계를 극복하게 합니다. 실험은 비전(vision), 언어(language), 비전-언어(vision-language) 벤치마크를 포함하여 다양한 작업에서 수행되었습니다.

- **Performance Highlights**: RandLoRA는 기존의 LoRA 방식과 비교하여 비전 및 언어 작업에서 성능 향상을 보여줍니다. 특히, 비전-언어 작업에서는 성능 격차를 현저히 줄이거나 심지어 없애는 경향을 보였습니다. 이는 RandLoRA의 유효성을 강조하며, 복잡한 작업에서 더 나은 성능을 제공하는 방법임을 증명합니다.



### Context-Aware Hierarchical Merging for Long Document Summarization (https://arxiv.org/abs/2502.00977)
Comments:
          30 pages

- **What's New**: 이 논문에서는 Hierarchical Merging 기법을 활용하여 매우 긴 문서(100K 토큰 이상)를 요약하는 새로운 방안을 제시합니다. 기존의 방법들에서는 LLM(대형 언어 모델)을 활용하여 요약을 생성하였지만, 비합리적인 요소인 hallucination(환각)이 문제시되었습니다. 제안된 방법은 원본 문서로부터 관련된 맥락(context)를 결합하여 요약의 진실성을 높이고자 합니다.

- **Technical Details**: Hierarchical Merging 기법은 입력 문서를 고정된 길이로 나눈 후, 각 부분을 요약하고 이들을 통합하여 최종 요약을 생성하는 과정입니다. 본 연구에서는 맥락을 보강하는 세 가지 방법을 제안하며, 이 방법들은 추출적 요약(extractive summarization), 정보 검색(retrieval), 그리고 인용(attribution)을 포함합니다. 이를 통해 각 단계에서 구성 요소들이 진실한 정보를 기반으로 요약을 생성할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, Llama 3.1 모델 패밀리에 대해 제안된 방법이 기존의 zero-shot 및 hierarchical merging 기반보다 요약의 진실성과 품질 모두에서 일관되게 우수함을 보여주었습니다. 특히, 추출적 요약과 함께 활용할 때 개념적 요약이 효과적으로 작용하며, 관련된 맥락의 강화가 요약의 신뢰성을 높이는 것으로 나타났습니다.



### Wizard of Shopping: Target-Oriented E-commerce Dialogue Generation with Decision Tree Branching (https://arxiv.org/abs/2502.00969)
Comments:
          Accepted by SIGDIAL 2024 but withdrawn

- **What's New**: 본 논문에서는 고객의 쇼핑 의도를 이해하고 적절한 제품을 찾기 위해 대화형 제품 검색(Conversational Product Search, CPS) 시스템을 개발하는 새로운 접근 방식인 TRACER를 제안합니다. TRACER는 대형 언어 모델(LLMs)을 활용하여 다양한 쇼핑 도메인에 대해 사실적이고 자연스러운 대화를 생성하며, 제품 검색 경로를 결정 트리 모델에서 예측하는 대화 계획에 기반하여 발생합니다. 또한 Wizard of Shopping (WoS)이라는 첫 번째 목표 지향 CPS 데이터셋을 발표하여, 세 가지 쇼핑 도메인에서 3,600개의 자연스러운 대화를 포함합니다.

- **Technical Details**: TRACER는 두 개의 LLM 에이전트를 사용하여 쇼핑 대화를 시뮬레이션하고, 각각 고객과 판매자인 두 가지 역할을 할당합니다. 고객이 지정된 쇼핑 관심사를 가지고, 판매자는 제품 카탈로그에 접근하여 대화를 진행합니다. 대화 생성을 현실적으로 유지하기 위해 결정 트리로부터 예측된 대화 계획을 활용함으로써, 고객이 가장 적은 검색 조건으로 목표 제품에 도달하도록 보장합니다.

- **Performance Highlights**: WoS 데이터셋을 통한 인간 평가 결과, 생성된 대화가 매우 자연스럽고 일관된 것으로 나타났으며, 이는 LLMs가 대화 계획을 쉽게 자연스러운 대화로 변환할 수 있음을 보여줍니다. 제안된 TRACER 접근 방식 및 WoS 데이터셋은 대화형 쿼리 생성 및 제품 순위 결정 작업의 개선을 입증하며, CPS 연구 분야에 중요한 기여를 할 것으로 기대됩니다.



### Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search (https://arxiv.org/abs/2502.00955)
- **What's New**: 본 논문에서는 대규모 언어 모델(LLM) 기반 다중 에이전트 시스템(MAS)에서 자기 학습을 향상하기 위해 합성 데이터를 생성하는 새로운 프레임워크인 데이터 영향 기반 트리 탐색(Data Influence-oriented Tree Search, DITS)을 제안합니다. DITS는 영향 점수(influence score)를 사용하여 데이터 선택 및 트리 탐색에서 모델 성능 향상에 가장 직접적으로 기여하는 데이터를 우선적으로 선택하도록 설계되었습니다. 이를 통해 기존의 Q-value에 의존하는 접근 방식의 한계를 극복하고 모델 훈련에 더욱 효과적인 데이터를 식별할 수 있습니다.

- **Technical Details**: DITS는 비미분 가능한 성능 지표에 적합하도록 설계된 영향 점수 추정 방법을 도출합니다. 이 방법은 전통적인 접근 방식에서 필요로 하는 계산 집약적인 그래디언트(computationally intensive gradient computations)를 피하고, 추론 계산을 활용하여 컴퓨팅 오버헤드를 상당히 줄입니다. DITS 프레임워크는 훈련 데이터의 유용성을 정량화하여 모델 성능 향상에 실질적인 기여를 하는 데이터를 선택할 수 있도록 합니다.

- **Performance Highlights**: 여덟 개의 다중 에이전트 데이터 세트에서 DITS의 효과를 검증한 결과, 우리의 방법은 최신 다중 에이전트 최적화 기술을 능가하며 정보 교환 작업에서 단일 라운드 반복에서 평균 2.1%, 다중 라운드 반복에서 2.5%의 성능 향상을 보여주었습니다. 또한, 동일한 데이터 합성 예산 내에서 DITS는 전통적인 방법을 초월하여 합성 계산의 효율적인 확장을 가능하게 합니다.



### Universal Abstraction: Harnessing Frontier Models to Structure Real-World Data at Sca (https://arxiv.org/abs/2502.00943)
- **What's New**: 이번 논문에서는 UniMedAbstractor (UMA)라는 새로운 의료 추출 프레임워크를 제안하여, 전통적인 방법의 한계를 극복하고자 합니다. UMA는 Large Language Models (LLMs)을 활용하여 모듈화된 프롬프트 템플릿을 통해 제로샷(zero-shot) 의료 추출을 가능하게 합니다. 기존의 방식이 요구하는 수작업 노력을 최소화하고, 새로운 속성(attribute)에 신속하게 적응할 수 있는 방법론을 제시하고 있습니다.

- **Technical Details**: UMA의 핵심은 유연한 프롬프트 템플릿으로, 사용자가 관심 있는 특정 속성을 정의할 수 있게 되어 있습니다. 이를 통해 단기 특성(short-context attributes)과 장기 특성(long-context attributes)을 모두 수용할 수 있도록 설계되었습니다. 또한, 각 속성에 대한 출력 및 구성요소를 모듈화하여 새로운 속성을 쉽게 추가할 수 있는 확장성을 확보하고 있습니다.

- **Performance Highlights**: 실제 데이터에서의 실험 결과, UMA는 F1/정확도 점수에서 평균 2 포인트 개선 효과를 보였으며, 병리학적 T 단계에서의 경우 감독된 모델과 비교해 20 포인트 이상의 정확도 향상을 보여주었습니다. 이러한 성능은 UMA가 전통적인 방법보다 실질적으로 효과적인 의료 추출 솔루션임을 입증합니다.



### Attention Sinks and Outlier Features: A 'Catch, Tag, and Release' Mechanism for Embeddings (https://arxiv.org/abs/2502.00919)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 주목할 만한 두 가지 특성인 attention sinks와 outlier features의 중요성을 분석합니다. 연구에 따르면 attention sinks는 토큰 시퀀스를 캡처하고 태그를 붙여 이후 다시 스트림으로 풀어주는 'catch, tag, release' 메커니즘을 활용합니다. 이러한 발견은 모델 성능 향상 및 압축에 기여할 수 있습니다.

- **Technical Details**: 저자들은 두 개의 주요 현상인 attention sinks와 outlier features를 조사하였으며, 이들이 모델 매개변수에서 어떻게 나타나는지 설명합니다. attention weight matrices의 저랭크(low-rank) 구조가 이러한 현상을 생성하는 데 기여함을 입증했습니다. 이는 평균화와 같은 단순한 작업에서 이러한 메커니즘이 자연스럽게 발생하는 이유를 설명합니다.

- **Performance Highlights**: 본 연구는 OATS라는 새로운 압축 알고리즘을 통해 attention sinks와 outlier features를 효과적으로 보존할 수 있음을 보여줍니다. OATS는 전통적인 pruning 알고리즘과 비교하여 성능 저하 없이 저랭크 구조를 유지합니다. 결과적으로, 저자들은 few-shot learning과 같은 다운스트림 과제에서도 성능을 높일 수 있음을 입증했습니다.



### The Accuracy, Robustness, and Readability of LLM-Generated Sustainability-Related Word Definitions (https://arxiv.org/abs/2502.00916)
Comments:
          NLP4Ecology Workshop 2025

- **What's New**: 이번 연구는 LLMs(대형 언어 모델)이 기후 용어를 정의하는 데 있어 표준화된 용어의 중요성을 강조합니다. 300개의 IPCC(기후 변화에 관한 정부間 패널) 용어 정의와 함께, GPT-4o-mini, Llama3.1 8B 및 Mistral 7B에서 생성된 정의를 비교하고 분석했습니다. 이들 모델은 평균적으로 0.57-0.59의 일치도를 기록했으며, 모델 생성 정의는 원본보다 읽기 어려운 것으로 나타났습니다.

- **Technical Details**: 연구 방법으로는 SBERT 문장 임베딩을 통해 공식 정의와 모델 생성 정의 간의 유사도를 계산하고, 이로부터 일치도(adherence) 및 강건성(robustness)을 평가했습니다. 모델 생성 정의의 변동성을 평가하기 위해, 동일한 용어에 대해 다른 프롬프트(template)를 이용하여 5개의 정의를 생성하고 이들의 유사도를 비교했습니다. LLM 모델은 GPT-4o-mini, Meta의 Llama-3.1-8B, Mistral-7B를 사용했습니다.

- **Performance Highlights**: 결과적으로 LLM의 정의는 IPCC 정의에 비해 읽기 수준이 높고 복잡성은 더 큰 것으로 나타났습니다. 또한 일부 정의는 특정 맥락에 제한되지 않아 다양한 해석을 초래하는 경향이 있었습니다. 특히, 단어에 따라 정의의 일관성이 크게 변동하는 반면, 평균적으로 LLM의 응답은 IPCC 목표를 달성하는 데 더 개선이 필요한 것으로 평가되었습니다.



### Embracing Dialectic Intersubjectivity: Coordination of Different Perspectives in Content Analysis with LLM Persona Simulation (https://arxiv.org/abs/2502.00903)
- **What's New**: 이번 연구에서는 content analysis 방법론을 consensus 중심에서 coordination 중심으로 발전시키는 것을 목표로 하였습니다. 이를 통해 다양한 coding outputs를 수용하고 여러 관점 간의 역동성을 탐구할 수 있는 기회를 제공합니다. 연구에서는 여섯 가지 GPT-4o 구성 모델을 평가하여 2020년 미국 대통령 선거 동안 Fox News와 MSNBC의 Biden과 Trump에 대한 감정을 분석했습니다.

- **Technical Details**: 이 연구는 진행 중인 LLM-Assisted Content Analysis (LACA) 방법론을 통해 이러한 분석을 수행했습니다. dialectic intersubjectivity 프레임워크를 적용하여 다양성을 강조하고, 정치적으로 일치하는 콘텐츠를 처리할 때 이념적 편향이 어떻게 나타나는지 평가했습니다. 각 모델의 출력은 두 후보에 대한 긍정성과 부정성 기준으로 평가되었으며, 상호코더 신뢰성(intercoder reliability) 또한 검토되었습니다.

- **Performance Highlights**: 연구 결과, LLM들은 동일한 이념적 집단의 코드 작성자들 간의 높은 일치를 보여주었으나, 이념적으로 일치하는 콘텐츠를 분석할 때 감정의 차이를 보였습니다. 이러한 차이는 dialectic intersubjectivity의 중요성을 강조하며, 연구자들이 LLM의 도구를 통해 다양한 해석을 포착할 수 있는 가능성을 보여줍니다. 이는 AI 기반 사회 과학 연구의 신뢰성을 높이고, 실제 세상의 복잡성을 반영하는 데 기여할 수 있습니다.



### MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies (https://arxiv.org/abs/2502.00894)
- **What's New**: 이번 논문에서는 MorphBPE라는 형태소 인식 버전의 Byte Pair Encoding (BPE)을 소개합니다. MorphBPE는 하위 단어(token) 토큰화에 언어적 구조를 통합하여 통계적 효율성을 유지하면서 형태소 경계를 고려합니다. 이 방법은 기존 LLM 훈련 파이프라인과 완벽하게 호환되며 최소한의 수정으로 통합이 가능하다는 점에서 주목할 만합니다.

- **Technical Details**: MorphBPE는 형태소별 편집 거리(Morphological Edit Distance)와 형태소 일관성 F1 점수(Morphological Consistency F1-Score)를 포함한 두 가지 새로운 평가 메트릭을 도입합니다. 이는 토큰화 품질을 평가하는데 있어 유용하며, 특히 형태론적으로 복잡한 언어에서의 효율성을 향상시키는 데 기여합니다. 연구는 영어, 러시아어, 헝가리어, 아랍어를 포함한 다양한 언어와 300M 및 1B 파라미터의 LLM을 대상으로 진행되었습니다.

- **Performance Highlights**: Experiments show that MorphBPE는 cross-entropy loss를 일관되게 줄이고 수렴 속도를 가속화하며 형태소 정렬 점수를 개선합니다. MorphBPE는 구조적으로 복잡한 언어에 대한 더 나은 하위 단어 토큰화를 제공하며, 모델의 성능 향상에 기여합니다. 이 접근법은 전통적인 형태소 분석과 NLP 간의 격차를 해소하는 데 도움을 줍니다.



### Predicting potentially unfair clauses in Chilean terms of services with natural language processing (https://arxiv.org/abs/2502.00865)
Comments:
          37 pages, 2 figures, under review

- **What's New**: 이 연구는 소비자 계약에서 정보 비대칭에 대한 우려를 다루고 있으며, 복잡한 Terms of Service(Tos)의 확산에 의해 악화되고 있다. 특히, 스페인어 법조문을 다룬 첫 번째 다중 레이블 분류 데이터세트를 소개하여 칠레 내의 온라인 서비스를 위한 데이터와 방법론을 발전시켰다. 본 연구는 기계 학습의 가능성을 활용하여 소비자가 잠재적으로 불공정한 조항을 식별할 수 있도록 지원하는 데 중점을 두고 있다.

- **Technical Details**: 최신 연구에서 제안한 분류 시스템은 20개의 잠재적으로 불공정한 조항을 3개의 그룹으로 분류하는 것이며, 여기에는 6개의 불법 조항, 6개의 다크 조항, 그리고 8개의 회색 조항이 포함된다. 연구에서는 50개의 스페인어 Terms of Service 문서에서 6,523개의 주석이 달린 조항을 포함한 데이터셋을 구축하였고, 이를 통해 15,201개의 조항으로 이루어진 데이터셋이 생성되었다. 또한, Transformer 기반 모델을 사용하여 다양한 ML 접근법을 평가하고, 마이크로 및 매크로 F1 점수로 성능을 비교하였다.

- **Performance Highlights**: 실험 결과, 잠재적으로 불공정한 조항의 감지 작업에서는 매크로 F1 점수가 79%에서 89%까지 다양하게 나타났고, 마이크로 F1 점수는 96%까지 도달하였다. 분류 작업에서도 매크로 F1 점수가 60%에서 70%까지 이루어졌으며, 마이크로 F1 점수는 64%에서 80%의 범위를 보였다. 이 연구는 칠레 법 체계 내에서 공정 거래를 촉진하고 소비자 권익을 보호하는 데 이바지할 수 있는 중요한 시작점을 제공한다.



### HintEval: A Comprehensive Framework for Hint Generation and Evaluation for Questions (https://arxiv.org/abs/2502.00857)
Comments:
          Submitted to SIGIR 2025

- **What's New**: 이번 연구에서는 HintEval이라는 새로운 파이썬 라이브러리를 소개합니다. 이 라이브러리는 Hint Generation과 Hint Evaluation을 위한 통합된 프레임워크로, 다양한 데이터세트와 평가 기준을 제공합니다. HintEval은 산재해 있는 자원을 하나의 도구로 통합하여 연구자들이 쉽게 접근하고 활용할 수 있도록 도와줍니다.

- **Technical Details**: HintEval은 여러 형식의 데이터셋을 제공하며, 이로 인해 연구자들이 데이터를 준비하는 데 소요되는 시간을 절약할 수 있습니다. 또한 Hint Generation(힌트 생성)과 Hint Evaluation(힌트 평가)을 위한 표준화된 접근 방식을 통해, 연구자들이 비교적 쉽게 일관된 평가 기준을 적용할 수 있도록 합니다. 이 라이브러리는 자세한 온라인 문서와 함께 제공되어 사용자가 기능을 탐색하고 쉽게 시작할 수 있습니다.

- **Performance Highlights**: HintEval은 학습 지원에 있어서 중요한 진전을 이루는 데 기여합니다. 연구자들은 HintEval을 통해 적극적으로 힌트를 최적화하고 교육 및 문제 해결 작업에서 어떻게 활용할 수 있는지에 대한 깊은 이해를 확보할 수 있습니다. 이 라이브러리는 GitHub와 PyPI에서 무료로 사용할 수 있어, 연구자와 실무자들에게 광범위하게 접근 가능하다는 점도 특징입니다.



### Explainability in Practice: A Survey of Explainable NLP Across Various Domains (https://arxiv.org/abs/2502.00837)
- **What's New**: 이 논문은 설명 가능한 자연어 처리(Explainable NLP, XNLP)의 중요성과 실제 적용 사례를 탐구합니다. 특히, 의료와 금융과 같은 도메인에서 XNLP의 활용 필요성을 강조하며, 기존의 문헌에서 부족한 부분을 보완하기 위해 도메인 별 접근 방식을 제안합니다. 또한, XNLP의 실용적인 배치와 그로 인해 사용자 신뢰가 어떻게 증진될 수 있는지를 논의합니다.

- **Technical Details**: 자연어 처리(Natural Language Processing, NLP)와 대형 언어 모델(Large Language Models, LLMs)은 기계가 인간 언어를 더 잘 이해하도록 도와주며 다양한 분야에 걸쳐 응용되고 있습니다. 아울러, 설명 가능한 인공지능(Explainable AI, XAI) 기법들이 모델의 의사 결정을 명확히 하기 위해 사용되고 있으며, 주목할 만한 기법으로는 LIME(Local Interpretable Model-agnostic Explanations)와 SHAP(SHapley Additive exPlanations)가 있습니다. 하지만, XNLP의 실제 적용과 평가 방법에 대한 논의는 여전히 부족하며, 이는 도메인 별 특성을 고려해야 함을 시사합니다.

- **Performance Highlights**: XNLP는 다양한 작업에서 높은 예측 성능을 보이는 기계 학습 모델의 '블랙 박스' 문제를 해결하고자 하고 있습니다. 특히, 진료 지원 시스템에서의 활용은 환자 관리에 긍정적인 영향을 미치며, 금융 부문에서도 사기 탐지 및 위험 평가에서 중요한 역할을 할 수 있습니다. 그러나, 각 도메인에서 신뢰할 수 있는 해석 가능성을 확보하기 위한 방법ologies가 필요하며, 이러한 요구 사항은 결정되는 결과에 직접적인 영향을 미칠 것입니다.



### Generalization of Medical Large Language Models through Cross-Domain Weak Supervision (https://arxiv.org/abs/2502.00832)
- **What's New**: 이번 연구에서는 Incremental Curriculum-Based Fine-Tuning (ICFT) 프레임워크를 제안하여 의료 대규모 언어 모델(MLLM)의 생성 능력을 향상시키고자 합니다. ICFT는 커리큘럼 학습(curriculum learning)과 메모리 조정(memory coordination), 그리고 파라미터 효율적인 미세 조정(parameter-efficient fine-tuning)의 조합을 통해 일반 언어 지식에서 특정 도메인 전문 지식으로의 점진적인 전이를 가능하게 합니다. 이를 통해 탁월한 정확도 및 효율성을 달성하며, 기존 방법들보다 우수한 성능을 보여줍니다.

- **Technical Details**: ICFT는 세 가지 주요 구성요소, 즉 의료 지식 주입(Medical Knowledge Injection), 이중 단계 메모리 조정(Dual-Stage Memory Coordination), 저순위 적응(LoRA)을 통한 파라미터 효율적인 미세 조정(Parameter-Efficient Fine-Tuning)으로 구성됩니다. 이 프레임워크는 일반적인 의료 지식에서 시작하여 단계적으로 더 전문화되고 복잡한 데이터셋을 통합함으로써 모델의 능력을 강화합니다. 이러한 접근법을 통해 모델은 일반적인 언어 및 추론 능력을 유지하면서 도메인 전문 지식을 습득할 수 있습니다.

- **Performance Highlights**: 실험 결과 ICFT가 의료 질문 응답, 진단 추론, 선호 분류 작업에서 기존 최첨단 방법들에 비해 최대 10%의 정확도 향상을 달성했다는 것을 보여줍니다. 또한, ICFT 모델은 미지의 시나리오에서 더 나은 일반화 능력을 나타내어 점진적 학습 패러다임의 효능을 강조합니다. 이러한 결과는 ICFT 프레임워크가 의료 도메인에서 LLM을 적응시키기 위한 튼튼하고 확장 가능한 솔루션임을 확립하고 있습니다.



### Weak Supervision Dynamic KL-Weighted Diffusion Models Guided by Large Language Models (https://arxiv.org/abs/2502.00826)
- **What's New**: 이번 논문에서는 텍스트-이미지 생성 품질 및 효율성을 높이기 위해 대형 언어 모델(LLM)과 확산 모델(diffusion model)을 결합한 새로운 방법을 제안합니다. 특히, 동적 KL 가중치 조정 전략을 도입하여 확산 과정을 최적화하고, 사전 훈련된 LLM의 의미적 이해를 활용하여 생성 과정을 안내합니다. 새로운 접근 방식은 생성된 이미지의 시각적 품질과 텍스트 설명과의 정합성을 크게 향상시켜 전통적인 GAN 기반 모델과 비교하여 뛰어난 성능을 보여줍니다.

- **Technical Details**: 본 연구는 먼저 대규모 이미지 및 텍스트 설명 데이터셋으로 확산 모델을 사전 훈련한 후, 점진적으로 언어 모델의 영향을 확산 과정에 도입합니다. LLM의 제어가 점진적으로 증가함에 따라 이미지 생성이 이루어지며, 감독 및 비감독 손실(supervised and unsupervised losses)을 조합하여 훈련합니다. 이를 통해 생성된 이미지와 텍스트 입력 간의 불일치를 최소화하는 데 초점을 맞추며, 실험에서는 Inception Score(IS), Fréchet Inception Distance(FID), 새롭게 제안된 정합 점수(alignment score)를 사용하여 모델 성능을 평가합니다.

- **Performance Highlights**: 실험 결과, 본 접근 방식은 기존 방법들보다 높은 품질의 이미지 생성을 달성하면서 텍스트 입력과의 정합성에서 더 높은 정도를 유지하는 것으로 나타났습니다. 특히, 광범위한 실험을 통해 기존 GAN 기반 모델들에 비해 이미지의 사실성, 텍스트와의 관계, 전반적인 미적 품질에서 우수한 성과를 확인하였습니다. 또한, 본 연구의 방법론은 다른 멀티모달 작업에도 확장 가능성을 보여주어 다양한 생성 응용 프로그램에 유용한 솔루션이 될 것으로 기대됩니다.



### Probing Large Language Models in Reasoning and Translating Complex Linguistic Puzzles (https://arxiv.org/abs/2502.00817)
Comments:
          8 pages, 8 figures

- **What's New**: 본 논문은 복잡한 언어 퍼즐을 해결하기 위해 Large Language Models (LLMs)의 활용을 조사합니다. 특히 Input-Output Prompting (IO), Chain-of-Thought Prompting (CoT), Solo Performance Prompting (SPP) 등 다양한 프롬프팅 기법을 통해 LLMs의 추론 능력을 향상시키는 방법을 탐구합니다. 연구 결과는 LLMs의 언어적 추론과 복잡한 번역 작업에서의 잠재력을 조명하고, 관련 제한 사항을 확인합니다.

- **Technical Details**: 이 연구는 Puzzling Machine Competition과 여러 Linguistics Olympiad에서 수집한 데이터셋을 사용하여 GPT-4 0603 모델의 성능을 평가합니다. 다양한 프롬프팅 기법을 적용하여 모델의 결정 경로를 설명하고, 특히 Two-Phase reasoning 전략과 CoT 기법을 활용하여 더 깊이 있는 사고 과정을 유도합니다. 이러한 접근은 LLMs의 언어적 문제 해결 능력을 평가하는 데 중점을 둡니다.

- **Performance Highlights**: GPT-4 모델의 평가 결과에 따르면, 다양한 프롬프팅 기법이 이 모델의 성능에 긍정적인 영향을 미치며, 특히 Chain-of-Thought 방식이 LLM의 추론 능력을 크게 향상시킨다는 것을 보여줍니다. BLEU, characTER, chrF 등의 메트릭스를 통해 번역의 품질을 정량적으로 평가하면서, LLMs가 복잡한 언어 작업에서의 적용 가능성을 넓힐 수 있음을 알 수 있습니다. 이러한 결과는 자연어 처리(NLP) 분야에서 LLM 사용 최적화에 기여할 것으로 기대됩니다.



### Vision-centric Token Compression in Large Language Mod (https://arxiv.org/abs/2502.00791)
- **What's New**: 이 논문은 큰 언어 모델(LLM)의 긴 문맥 처리 능력을 개선하기 위한 새로운 방법인 VIST를 제안합니다. 이 방법은 기존의 텍스트 인코더 대신 경량의 비전 인코더를 사용하여 긴 텍스트를 이미지로 변환하고 압축하여 처리합니다. VIST는 기존 방법들보다 16% 적은 FLOPs와 50% 적은 메모리 사용량으로 유사한 성능을 달성하며, 중요한 토큰에 집중할 수 있는 빈도 기반 마스킹 전략을 활용합니다. 이 접근 방식은 결국 LLM의 토큰 효율성을 크게 향상시킵니다.

- **Technical Details**: VIST는 긴 텍스트를 이미지로 변환한 후, 경량 비전 인코더를 사용하여 핵심 정보를 포착합니다. 기존의 텍스트 인코더를 사용했던 방식과 비교했을 때, VIST는 계산 비용이 낮고 더 긴 입력을 효율적으로 처리할 수 있습니다. 특별히, Probability-Informed Visual Enhancement (PVE)를 통해 의미가 풍부한 토큰에 대한 강조를 높이고 성과를 극대화합니다. 기존 LLM의 문제점을 해결하기 위해 VIST는 이미지와 텍스트를 통합하여 장기적인 문맥 정보를 효과적으로 처리합니다.

- **Performance Highlights**: VIST는 TriviaQA, NQ, PopQA, TREF, SST2 및 SST5와 같은 다양한 기준에서 평균 5.7% 성능 향상을 보여줍니다. 텍스트 인코더 기반의 기존 방법들과 비교할 때, VIST는 오픈 도메인 QA 및 ICL 작업에서 우수한 결과를 달성하며, 낮은 계산 비용으로 더 긴 문맥을 처리할 수 있습니다. 이러한 성과는 LLM의 실용적인 적용에서 중요한 돌파구를 마련하며, VIST의 새로운 접근 방식이 큰 영향을 미칠 것임을 시사합니다.



### FIRE: Flexible Integration of Data Quality Ratings for Effective Pre-Training (https://arxiv.org/abs/2502.00761)
Comments:
          19 pages, 11 figures

- **What's New**: 본 논문에서는 고품질 데이터를 선택하는 FIRE라는 유연하고 확장 가능한 프레임워크를 제안합니다. FIRE는 여러 데이터 품질 평가자의 평가 신호를 통합하여 데이터 품질을 포괄적으로 평가합니다. 기존 방법들이 단일 품질 신호에 의존하고 있는 반면, FIRE는 다양한 차원에서 데이터 품질을 분석할 수 있도록 설계되었습니다.

- **Technical Details**: FIRE 프레임워크는 두 가지 주요 프로세스를 포함합니다: 1) 평가 정렬(Rating Alignment) 2) 평가자 통합(Rater Integration). 먼저, 여러 평가자의 등급을 통합된 등급 공간으로 매핑하기 위해 win rate를 기반으로 한 정렬 방법을 제안합니다. 이후, 평가하여 얻어진 품질 신호를 통합할 때 평가자의 내재적 신뢰성과 독립성을 고려하여 보다 신뢰할 수 있는 통합 평가를 제공합니다.

- **Performance Highlights**: SlimPajama 데이터셋에 대한 실험 결과, FIRE는 다양한 다운스트림 작업에서 일관되게 다른 선택 방법보다 뛰어난 성능을 보였습니다. 평균 2.9% 성능 향상과 특정 성능 수준을 달성하기 위해 필요한 FLOPs를 절반 이상 줄이는 성과를 기록했습니다. 이러한 결과는 FIRE가 데이터 품질 평가 및 선택에서 효과적임을 입증합니다.



### Structural Latency Perturbation in Large Language Models Through Recursive State Induction (https://arxiv.org/abs/2502.00758)
- **What's New**: 이 논문에서는 재귀적 상태 유도를 통한 구조적 지연 섭동(Structural Latency Perturbation, SLP)이라는 새로운 프레임워크를 제안합니다. SLP는 LLM의 내부 계산 경로를 체계적으로 변경하여 효율성을 높이고 지연 시간을 줄이는 방법을 제시합니다. 이는 기존의 모델 압축 방법이나 파라미터 기반 수정과는 달리, 정보가 모델 내에서 전파되는 방식을 재구성하는 접근법입니다. 즉, 모델의 내부 상태 전이를 동적으로 재구성하여 필요한 경우에만 계산 복잡성을 낮추는 역할을 합니다.

- **Technical Details**: SLP는 입력 특정 특성에 따라 모델의 계산 그래프를 재구성할 수 있도록 하여 더 효율적으로 처리할 수 있게 합니다. 재귀적 상태 조정을 통해 모델은 시퀀스 길이에 따라 지연 시간을 줄일 수 있으며, 더 긴 텍스트 생성의 경우 누적 효율성 개선의 혜택을 볼 수 있습니다. 기존의 압축이나 양자화 방법과 비교할 때, SLP는 토큰 유지나 메모리 사용량을 저해하지 않고도 지연 시간을 줄일 수 있는 가능성을 지니고 있습니다. 또한, 선택적으로 중복 활성화를 억제함으로써 전력 효율성을 개선할 수 있다는 점이 새로운 주목거리입니다.

- **Performance Highlights**: 실험 결과, SLP를 적용했을 때 여러 시퀀스 길이에서 지연 시간이 감소함을 확인했습니다. 특히, 텍스트 생성을 연장할 경우 누적 효율성 개선이 관찰되었습니다. 언어 안정성 분석에서도, 통제가 이루어진 섭동 임계값 아래에서 토큰 수준의 일관성이 대체로 유지됨을 보여주었습니다. 이러한 결과는 단계적 지연 수정이 매개변수 중심 최적화 기법의 대안으로서 유효하다는 것을 뒷받침합니다.



### Universal Post-Processing Networks for Joint Optimization of Modules in Task-Oriented Dialogue Systems (https://arxiv.org/abs/2502.00747)
Comments:
          Accepted by AAAI 2025 Main Technical Track

- **What's New**: 본 연구에서 제안하는 UniPPN(Universal Post-Processing Networks)은 기존의 PPN(Post-Processing Networks) 방식의 한계를 극복하고, 모든 모듈의 출력을 변환하는 경량 언어 모델 기반의 네트워크입니다. 이 네트워크는 Reinforcement Learning(RL) 알고리즘을 활용하여 시스템의 전체적인 작업 완료 능력을 향상시키는 데 기여합니다. 기존 PPN이 부분 모듈에만 국한되어 있었던 점과는 달리, UniPPN은 모든 모듈의 출력을 동시에 처리하고 최적화하는 새로운 방법론을 제시합니다.

- **Technical Details**: UniPPN의 RL 알고리즘은 모듈 수준의 Markov Decision Process(MDP)를 사용하여 각각의 모듈에 대한 세밀한 가치 및 이점 추정을 가능하게 하여 공동 학습을 안정화합니다. Proximal Policy Optimization(PPO)을 기반으로 하여 최적화 알고리즘을 구축하고 있으며, KL(Kullback-Leibler) 발산을 기반으로 한 패널티를 통해 정책 네트워크의 안정성을 유지합니다. 또한, GPT-2을 뒷받침 모델로 채택하고 구성적 파리미터를 조정하여 대화 시스템의 출력을 더 간소화하였습니다.

- **Performance Highlights**: MultiWOZ 데이터셋을 통해 진행된 시뮬레이션 및 인간 평가 실험에서 UniPPN은 기존의 PPN보다 작업 완료 능력에서 월등한 성능을 보였습니다. 특히, 모듈 수준 MDP의 도입으로 인해 UniPPN의 학습 성능이 개선된 것으로 나타났으며, 후반부 학습 시에는 turn-level MDP보다 안정적인 성과를 기록하였습니다. 이러한 결과는 UniPPN이 대화 시스템의 효율성과 정확성을 크게 향상시킬 수 있는 잠재력을 지니고 있음을 보여줍니다.



### ReFoRCE: A Text-to-SQL Agent with Self-Refinement, Format Restriction, and Column Exploration (https://arxiv.org/abs/2502.00675)
Comments:
          13 pages, 1 figure

- **What's New**: 이번 연구에서 저자들은 ReFoRCE라는 새로운 자기 세분화 에이전트를 제안합니다. 이 시스템은 테이블 압축(table compression), 포맷 제한(format restriction), 그리고 반복적인 열 탐색(column exploration)을 통해 복잡한 SQL 쿼리 문제를 해결하려고 합니다. 이를 통해 Spider 2.0 데이터셋에서의 성능 향상을 목표로 하고 있으며, 특히 SQL 방언(dialect)과 복잡한 데이터 타입에 대한 이해를 높이는 데 중점을 두고 있습니다.

- **Technical Details**: ReFoRCE는 장기 문맥 문제(long-context limitations)를 해결하기 위해 테이블 정보를 압축하고, 정확한 응답 포맷을 보장하기 위해 포맷 제한을 도입합니다. 데이터베이스에 대한 이해를 높이기 위해 SQL 쿼리를 반복적으로 실행하는 열 탐색을 수행하며, 전체 작업 흐름을 여러 스레드로 병렬화(parallelization)하여 신뢰성을 높입니다. 또한, SQL 쿼리를 실행한 후 결과가 없는 경우, Common Table Expression (CTE) 기반의 자기 세분화(self-refinement) 접근 방식을 사용하여 단계적으로 해결책을 찾습니다.

- **Performance Highlights**: ReFoRCE는 Spider 2.0-Snow에서 26.69, Spider 2.0-Lite에서 24.50의 성과를 기록하며, 기존의 Spider-Agent와 비교하여 유의미한 성능 향상을 보여줍니다. 이 결과는 복잡한 SQL 요청 처리와 다양한 SQL 방언 지원에서의 우수한 기능을 입증합니다. ReFoRCE의 기법은 데이터베이스의 복잡성을 성공적으로 다루어 실세계 문제 해결에 기여할 수 있는 가능성을 보이고 있습니다.



### Rethinking Mixture-of-Agents: Is Mixing Different Large Language Models Beneficial? (https://arxiv.org/abs/2502.00674)
- **What's New**: 이 논문에서는 기존의 Mixture-of-Agents (MoA) 접근법 대신, Self-MoA라는 새로운 앙상블 방법을 제안합니다. Self-MoA는 단일 성능이 가장 뛰어난 LLM에서 나온 출력만을 집계하여, 다양한 LLM의 출력을 결합하는 기존 MoA보다 더 나은 성능을 보여주었습니다. 특히, Self-MoA는 AlpacaEval 2.0 벤치마크에서 MoA보다 6.6% 향상된 성과를 기록했습니다.

- **Technical Details**: Self-MoA는 동일한 모델에서 반복적으로 샘플링한 출력을 집계하여 앙상블을 수행합니다. 이 접근은 각 개별 모델의 질, 다양성 간의 상충 관계를 효과적으로 관리하여, 평균적으로 더 높은 질의 출력을 생성합니다. 논문에서는 Self-MoA의 성능을 다양한 MoA 설정 아래에서의 품질과 다양성 간의 트레이드오프를 통해 입증하고 있습니다.

- **Performance Highlights**: Self-MoA는 AlpacaEval 2.0을 포함한 여러 벤치마크에서 평균 3.8% 향상을 달성하였고, 새로운 최첨단 성능을 선보였습니다. Self-MoA-Seq라는 시퀀셜 버전도 소개되어, 이 버전은 출력의 수가 많아도 전체 성능을 저하시킴 없이 처리할 수 있도록 합니다.



### Evaluating Small Language Models for News Summarization: Implications and Factors Influencing Performanc (https://arxiv.org/abs/2502.00641)
- **What's New**: 이 연구는 19개의 소형 언어 모델(SLM)을 뉴스 요약에 대해 포괄적으로 평가하고, 2000개의 뉴스 샘플을 사용하여 유의미성(relevance), 일관성(coherence), 사실 일치(factual consistency) 및 요약 길이(summary length)에 중점을 두었습니다. 연구 결과, Phi3-Mini와 Llama3.2-3B-Ins와 같은 최상위 성능 모델이 70B 대형 언어 모델(LLM)과 비교 가능한 결과를 생성하며 더 간결한 요약을 생성하는 것을 발견했습니다.

- **Technical Details**: SLM은 대형 언어 모델(LLM)과 동일한 디코더 전용 아키텍처(architecture)를 공유하지만 4억 개 이하의 매개변수(parameters)를 가지고 있습니다. SLM은 스마트폰과 개인용 컴퓨터와 같은 엣지 장치에서 효율적으로 실행되도록 설계되어 사용자 프라이버시를 보호하며 빠르고 안정적이며 저비용 솔루션을 제공합니다. 연구에서는 ROUGE 메트릭을 사용하여 SLM의 성능을 평가하여, 복잡한 프롬프트가 요약 품질을 저하시킬 수 있음을 발견했습니다.

- **Performance Highlights**: 최상의 SLM은 LLM과 유사한 품질의 뉴스 요약을 생성하지만 요약 길이는 훨씬 짧습니다. 특히, 단순한 프롬프트를 사용할 경우 SLM의 성능은 향상되는 경향이 있으며 Instruction tuning의 효과는 모델마다 다르게 나타났습니다. Llama3.2 시리즈 모델의 경우, instruction tuning 후에는 성능이 상당히 향상되었지만, Qwen2 및 InternLM2 모델은 변화가 적었습니다.



### SimulPL: Aligning Human Preferences in Simultaneous Machine Translation (https://arxiv.org/abs/2502.00634)
Comments:
          Accepted to ICLR 2025. 23 pages,13 figures,11 tables

- **What's New**: 본 논문에서는 실시간 기계 번역(SiMT)에서 인간의 선호도를 반영하기 위한 새로운 프레임워크인 Simultaneous Preference Learning(SimulPL)을 제안합니다. 기존 SiMT 기법들이 인간의 선호를 고려하지 못한다는 문제를 해결하고, 번역 품질, 단순성, 일관성 및 지연 시간 선호를 포함한 다섯 가지 주요 선호도를 통해 번역 작업을 최적화합니다. SimulPL은 이 선호도를 기반으로 GPT-4/4o와 함께 효과적으로 학습할 수 있도록 도와줍니다.

- **Technical Details**: SimulPL 프레임워크는 언어학 및 계산 언어학의 기존 연구를 바탕으로 설정되어 있으며, 다섯 가지의 인간 선호도—번역 품질, 일관성, 핵심 포인트, 단순성, 지연 선호—로 구분됩니다. 이 프레임워크는 초기 선호 정렬을 위해 SiMT 모델의 번역 능력과 읽기/쓰기 정책을 동시에 훈련하는 Multi-task Supervised Fine-tuning(MSFT) 방법을 사용합니다. 그 후, SimulDPO를 통해 지연 선호를 최적화 목표에 통합하고, SiMT 모델이 보다 효과적으로 인간의 선호에 맞도록 조정할 수 있습니다.

- **Performance Highlights**: 실험 결과, SimulPL은 모든 지연 수준에서 더 높은 번역 품질을 달성하며, 인간의 선호와의 정렬이 개선된 것을 보여줍니다. 특히, 번역 작업에서 SimulPL이 제안한 선호도 기준을 기반으로 평가한 결과, 기존 방법보다 전반적으로 나은 성능과 선호 정렬을 실현함을 확인했습니다. 이러한 개선 사항들은 번역 품질의 향상뿐만 아니라, 다섯 가지 선호 범주에서의 전반적인 성과를 포함합니다.



### Efficient Language Modeling for Low-Resource Settings with Hybrid RNN-Transformer Architectures (https://arxiv.org/abs/2502.00617)
Comments:
          PDF has 12 pages total, 7 without references and abstract; 10 individual graphics combined to 3 figures; 5 tables

- **What's New**: 본 논문에서는 Transformer 기반 아키텍처를 개선하여 데이터가 부족한 환경에서도 모델 성능을 향상시키는 방법을 제안합니다. 주목할 만한 점은 Attention 레이어를 Feed-Forward 및 Quasi-Recurrent Neural Network (QRNN) 레이어로 선택적으로 변경하여 모델을 경량화하고 성능을 유지할 수 있음을 보여줍니다. 실험 결과, 파라미터 수를 줄이면서도 기존 모델들보다 우수한 성과를 보인다는 것을 확인했습니다.

- **Technical Details**: 우리는 세 가지 모델을 실험했습니다: PAR Transformer, Hybrid Transformer, 그리고 Single-Headed-Attention LSTM (SHA-LSTM)입니다. 이 모델들은 Feed-Forward 레이어, 상대 다중 헤드 Attention 레이어, QRNN 레이어 및 RNN Dropout 레이어와 같은 네 가지 기본 요소로 구성됩니다. 특히 Hybrid Transformer는 파라미터 수가 비슷한 기존 모델들과 비교하여 뛰어난 성능을 발휘하며, 파라미터 수를 크게 줄일 수 있음을 보여줍니다.

- **Performance Highlights**: Hybrid Transformer는 파라미터 수를 줄인 상태에서 비슷한 성능을 달성하여 기존 모델과 비교해 두드러진 성과를 보였습니다. 또한, 전체적으로 Training Cost를 크게 절감하면서 모델 성능을 향상시킬 수 있었습니다. 이로 인해 환경에 미치는 영향도 최소화할 수 있는 가능성을 제시합니다.



### Mitigating Heterogeneous Token Overfitting in LLM Knowledge Editing (https://arxiv.org/abs/2502.00602)
- **What's New**: 이번 연구에서는 지식 편집(Knowledge Editing, KE)에서 발생하는 이질적인 토큰 과적합(Heterogeneous Token Overfitting, HTO)이라는 새로운 문제를 식별했습니다. LLM(대형 언어 모델)이 제공된 지식의 다양한 토큰에 대해 과적합을 나타내는 방식은 원래의 추론 능력을 저하시키는 주요 원인으로 확인됩니다. 이에 따라, 우리의 새로운 접근법인 OVERTONE을 제안하여 개선을 도모하고 있습니다.

- **Technical Details**: OVERTONE은 과적합 상태에 따라 각 토큰에 적응형 학습 목표를 할당하는 KE 훈련 패러다임입니다. 이 방법은 기존 학습 방법과 비교하여 물리적인 계산 비용이 거의 들지 않으며, 이는 LLM의 전이 가능성을 유지하는 동시에, 더 나은 파라미터 업데이트를 제공합니다. 특히, OVERTONE은 선호 데이터 쌍이 필요 없는 직접 선호 최적화(Direct Preference Optimization, DPO)와 밀접하게 연결되어 있습니다.

- **Performance Highlights**: 폭넓은 실험을 통해 OVERTONE의 효과성과 다재다능성을 입증하였으며, 다양한 KE 방법 및 LLM을 포함한 실험을 진행했습니다. 우리의 방법은 다른 편집 방법에 비해 우수한 성능을 보여주며, LLM의 추론 능력을 유지하면서 새로운 지식을 성공적으로 업데이트할 수 있는 가능성을 열어주는 결과를 도출했습니다.



### RPGBENCH: Evaluating Large Language Models as Role-Playing Game Engines (https://arxiv.org/abs/2502.00595)
Comments:
          Submitted to ICML 2025

- **What's New**: RPGBench는 텍스트 기반 롤플레잉 게임(RPG) 엔진으로서 대형 언어 모델(LLMs)의 성능을 평가하기 위해 처음으로 설계된 기준입니다. 게임 생성(Game Creation, GC) 및 게임 시뮬레이션(Game Simulation, GS)이라는 두 가지 핵심 작업을 포함하여, LLM이 구조화된 이벤트-상태 표현을 사용하여 논리적 일관성을 유지한 게임 세계를 생성할 수 있도록 합니다. 이 벤치마크는 자동 검증 시스템을 통해 생성된 게임의 유효성을 검사하고, 이를 활용하여 LLM의 창의성, 일관성 및 복잡성을 평가할 수 있는 새로운 기준을 제시합니다.

- **Technical Details**: RPGBench는 게임 메커니즘과 이벤트-상태 기반 표현을 통해 LLM이 생성한 게임의 구조적 요건을 자동으로 검증하는 BFS 유효성 검사기를 포함합니다. 각 게임은 이벤트 조건, 상태 전이 및 종료 규칙이 잘 이행되었는지의 객관적 기준을 사용하여 평가됩니다. 또한, LLM은 이벤트 계획, 게임 내러티브 및 상태 업데이트라는 3단계로 구성된 동적 시뮬레이션 루프를 운영하며, 이를 통해 이야기에 대한 유연성을 유지하면서도 게임 메커니즘의 확장성을 평가합니다.

- **Performance Highlights**: RPGBench의 실험 결과, 최첨단 LLM들이 매력적인 이야기를 생성할 수 있지만, 복잡한 상황에서 일관되며 검증 가능한 게임 메커니즘을 구현하는 데에 어려움이 있음을 알 수 있었습니다. 이 연구는 평가 과정에서 LLM으로서의 평가 방법을 활용하여 주관적 평가를 수행하는 고유한 견해를 제공합니다. 또한, 인간 평가자와 자동 점수 간의 정렬 및 불일치를 조사하여 주관적 평가의 복잡성을 강조합니다.



### M+: Extending MemoryLLM with Scalable Long-Term Memory (https://arxiv.org/abs/2502.00592)
- **What's New**: 이 연구는 MemoryLLM의 제한을 극복하기 위해 M+라는 새로운 메모리 증강 모델을 제안합니다. M+는 메모리 리트리버와의 공동 학습을 통해 더 긴 컨텍스트에서도 정보를 효과적으로 기억할 수 있는 능력을 가지고 있습니다. 이를 통해 기존 MemoryLLM의 장기 기억 능력을 20k 토큰에서 160k 토큰 이상으로 확장했습니다.

- **Technical Details**: M+는 MemoryLLM의 아키텍처를 활용하여 장기 메모리 메커니즘을 통합합니다. 기존의 H2O 및 SnapKV와 달리, M+는 리트리버와 언어 모델을 공동으로 학습하여 모든 쿼리 헤드에 대해 한 번만 리트리버를 호출합니다. 이를 통해 GPU 메모리 사용량을 기존과 유사하게 유지하면서도 장기 기억력을 크게 향상시킵니다.

- **Performance Highlights**: M+는 긴 컨텍스트 이해, 지식 기억, 단기 문서 질의응답과 같은 다양한 벤치마크에서 테스트되었으며, 모든 긴 벤치마크에서 MemoryLLM과 최근의 강력한 기준선과 비교하여 유의미한 성능 향상을 보였습니다. 실험 결과는 M+가 동일하거나 더 작은 추론 메모리 용량 내에서 예외적으로 성능이 뛰어나다는 것을 보여줍니다.



### Data-Driven Mispronunciation Pattern Discovery for Robust Speech Recognition (https://arxiv.org/abs/2502.00583)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 최근 기계 학습의 발전은 음성 인식 분야에 큰 진전을 이루었으나, 비유창하거나 액센트가 있는 화자의 음성을 인식하는 데 여전히 도전과제가 남아 있습니다. 본 연구에서는 음성 데이터를 활용하여 비원어민의 잘못된 발음을 자동으로 탐지하고 교정하는 두 가지 데이터 기반 접근 방식을 제안합니다. 이를 통해 원어민과 비원어민의 발음 차이를 효과적으로 분석하여 실제적인 Automatic Speech Recognition (ASR) 시스템 개선을 이루어냈습니다.

- **Technical Details**: 제안하는 방법에서는 L1(모국어) 데이터로 훈련된 ASR 모델을 이용해 한국어 화자의 비원어민 영어 발화에서 발음 시퀀스를 추출하고, 이를 통해 잘못된 발음 패턴을 식별합니다. 주의 메커니즘(attention mechanism)을 사용하여, 비원어민 음소(non-native phones)와 원어민 음소(native counterparts)를 정렬함으로써 비원어민 화자의 발음 인식을 12.8% 향상시키는 성과를 얻었습니다. 기존의 규칙 기반 접근 방식과 달리, 데이터 기반 접근 방식을 통해 발음 변화를 보다 정확하게 반영할 수 있었습니다.

- **Performance Highlights**: 이 연구에서 제안한 방법은 원어민 영어 데이터셋에서 5.7%, 비원어민 화자, 특히 한국어 화자의 발표에서 12.8%의 성능 향상을 달성했습니다. 또한, 다양한 음성 인식 데이터셋을 활용해 ASR 시스템의 신뢰성을 높이는 데 기여하였습니다. 이러한 접근법은 언어 교육 등 다양한 분야에서 활용될 수 있는 가능성을 제시합니다.



### Detecting Ambiguities to Guide Query Rewrite for Robust Conversations in Enterprise AI Assistants (https://arxiv.org/abs/2502.00537)
Comments:
          Preprint

- **What's New**: 본 논문은 다중 턴 대화에서의 모호성을 해결하기 위한 NLU-NLG 프레임워크를 제안합니다. 특히, 우리는 'Ambiguity-guided Query Rewrite'라는 새로운 작업을 도입하여 쿼리를 자동으로 재구성하고 모호성을 감지할 수 있는 시스템을 개발하였습니다. 이 접근법은 Adobe Experience Platform AI Assistant에 통합되어 실제 애플리케이션에서 사용되고 있습니다.

- **Technical Details**: 우리는 AI 조수와의 대화 로그를 기반으로 모호성의 유형을 분류하기 위한 세 가지 주요 유형: Pragmatic, Syntactical, Lexical의 간단하지만 효과적인 분류 체계를 개발하였습니다. 이를 통해, 각 유형의 모호성을 감지할 수 있는 효과적인 특징을 설계하고 규칙을 정의하여 분류기를 구현하였습니다. 우리의 혼합 접근법은 기존 방법을 초월하는 성능을 보여주었습니다.

- **Performance Highlights**: 제안된 모호성 감지 분류기를 통해 AI 도우미의 전반적인 성능이 개선되었습니다. 모호한 쿼리만을 재작성함으로써 불필요한 오류를 줄이고, 정확한 대화 응답을 보장하는 효과가 있었습니다. 이러한 접근법은 다양한 산업 설정에서 쉽게 적용될 수 있는 특징을 가지고 있습니다.



### A statistically consistent measure of Semantic Variability using Language Models (https://arxiv.org/abs/2502.00507)
- **What's New**: 최근 논문에서는 언어 모델이 생성하는 출력의 변동성을 해결하기 위한 통계적으로 일관된 의미 변동성 측정을 제안했습니다. 이 측정 방법은 semantic spectral entropy로, 구현이 용이하고 기존 언어 모델만을 사용하여도 실행 가능합니다. 연구에서 이 방법이 언어 모델의 무작위성에도 불구하고 정확한 지표를 생성할 수 있다는 것을 극명하게 보여주었습니다.

- **Technical Details**: 저자들은 텍스트 조각의 수집체를 기반으로 한 의미적 변동성 측정을 제안하며, 이는 기존의 entropy 개념을 활용합니다. 예를 들어, multi-nomial distribution에 대한 불확실성을 측정하기 위해 의미적 클러스터에 대한 확률 분포를 정의하고, 이를 통해 텍스트의 의미적 다양성을 정량화합니다. 또한, 제안된 방법은 최소한의 가정 하에서도 통계적으로 일관성을 보장합니다.

- **Performance Highlights**: 최종적으로, 논문은 기계적인 일관성을 강조하며, 구체적인 예제를 통해 lexically와 syntactically 다른 텍스트의 클러스터를 구성하는 방법을 제시합니다. 이러한 방법은 기존 모델들의 변동성을 극복하고 신뢰할 수 있는 의미적 변동성 측정을 가능하게 합니다. 연구 결과는 언어 모델이 생성하는 텍스트의 불확실성을 효과적으로 평가할 수 있는 새로운 기준을 제시합니다.



### Towards Privacy-aware Mental Health AI Models: Advances, Challenges, and Opportunities (https://arxiv.org/abs/2502.00451)
Comments:
          18 pages, 2 figures

- **What's New**: 이 논문은 정신 건강 분야에서는 개인정보 보호와 관련된 기존 문제를 깊이 분석하며 인공지능(AI) 모델을 통한 진단 및 치료의 접근성을 높이기 위한 새로운 접근 방식을 제안합니다. 특히, 멀티모달 데이터 처리 기술이 정신 질환 진단에 어떻게 활용될 수 있는지를 강조하며, 관련 개인정보의 유출 위험을 낮추기 위한 방안을 모색합니다. 새로운 데이터 수집 및 모델 훈련을 위한 개인정보 보호 중심의 파이프라인 개발 방향도 제시하고 있습니다.

- **Technical Details**: 정신 건강 진단 자동화를 위해 텍스트, 오디오 및 비디오 데이터를 분석할 수 있는 멀티모달 AI 모델의 발전이 필요합니다. 그러나 데이터 수집 과정에서 GDPR 및 HIPAA와 같은 개인정보 보호 규정을 준수해야 하므로, PII(개인 식별 정보)를 안전하게 보호하는 방법을 모색해야 합니다. 이러한 과정을 통해 환자의 목소리와 얼굴 특징 등이 악용되는 위험을 줄이고, 안전하게 연구를 진행할 수 있도록 해야 합니다.

- **Performance Highlights**: 이 연구는 멀티모달 AI 모델들이 정신 질환 진단을 보조할 수 있는 잠재력을 가지고 있지만, 개인정보 보호 문제로 인해 적절한 데이터 세트를 확보하기가 어렵다는 점을 지적합니다. 논문에서는 데이터 익명화, 합성 데이터 생성, 그리고 개인정보 보호 훈련 방법 등의 해결책을 통해 이러한 문제를 극복할 방안을 제시하고, 이와 함께 모델 성능 평가를 위한 새로운 평가 체계가 필요하다고 강조합니다. 전반적으로, 개인정보 보호를 고려한 AI 도구들을 개발하여 환자의 치료 결과를 더욱 향상시키고자 하는 목표를 가지고 있습니다.



### HERA: Improving Long Document Summarization using Large Language Models with Context Packaging and Reordering (https://arxiv.org/abs/2502.00448)
Comments:
          7 pages, 1 figure

- **What's New**: 이 논문에서는 HERA라는 새로운 요약 생성 프레임워크를 제안합니다. HERA는 긴 문서에서 이벤트 관련 정보를 세분화하고 재배치하여 LLM의 요약 성능을 개선하는 데 중점을 두고 있습니다. 특히 HERA는 추가적인 파인튜닝 없이도 다양한 LLM에서 효과적으로 공신력 있는 요약을 생성할 수 있습니다.

- **Technical Details**: HERA는 긴 문서를 문단별로 나누고, 각 이벤트와 관련된 텍스트 세그먼트를 검색한 후 의미적 순서로 재정렬하여 입력 컨텍스트를 형성합니다. 이 과정에서 LLM의 정보 검색 능력을 활용하여 가장 관련성이 높은 문단을 선택하고, 문장 요약 모델을 통해 각 문단의 요약을 생성합니다. 최종적으로, HERA는 재정렬된 문단으로부터 전체 문서의 요약을 생성합니다.

- **Performance Highlights**: HERA는 arXiv와 PubMed 데이터셋에서 LLM과 결합하여 ROUGE, BERTScore 및 신뢰성 메트릭에서 기존 모델을 초월하는 성능을 보였습니다. 특히 HERA는 Gemini 1.5 및 GPT-4와 함께 사용할 때 모든 메트릭에서 우수한 결과를 기록하며, 다양한 LLM에서 전체 요약 품질을 크게 향상시킬 수 있음을 보여줍니다.



### UniAttn: Reducing Inference Costs via Softmax Unification for Post-Training LLMs (https://arxiv.org/abs/2502.00439)
Comments:
          11 pages, 4 figures. Preprint, under review

- **What's New**: 본 논문에서는 사후 훈련(post-training)에서의 추론 비용을 줄이기 위해 Softmax 통합(Softmax Unification) 방법인 UniAttn을 제안합니다. 이 방법은 LLM(대형 언어 모델)의 여러 Transformer 블록에서 Softmax 활성화를 통합하여 메모리 및 추론 비용을 크게 감소시키는 기술입니다. UniAttn은 기존의 효율적인 아키텍처들보다 더욱 뛰어난 성능을 보이며, 실제 애플리케이션에 강력한 실용성을 제공합니다.

- **Technical Details**: UniAttn은 LLM 내의 연속적인 Transformer 블록을 "슈퍼블록(Superblock)"으로 그룹화하여 모든 블록 내에서 Softmax 활성화를 통합합니다. 이 과정에서 발생할 수 있는 오류는 선형 투영(linear projection)을 사용하여 보완합니다. 실험을 통해 UniAttn이 표준 사후 훈련과 거의 동등한 성능을 보이면서 추론 비용을 상당히 줄일 수 있음을 입증하였습니다.

- **Performance Highlights**: UniAttn은 4개의 오픈 소스 사전 훈련된 LLM에 대한 실험에서 도메인 특정 역량 강화와 일반 역량 개선 두 가지 시나리오에서 적용되었습니다. 그 결과, UniAttn은 기존의 효율적인 아키텍처들보다 더 나은 성능을 유지하면서도 추론 비용을 획기적으로 낮출 수 있음을 보여주었습니다. 또한, UniAttn은 KV-캐시(KV-cache) 압축 방법과 결합할 수 있어 메모리 과부하를 더 줄이는 데 기여할 수 있습니다.



### Sagalee: an Open Source Automatic Speech Recognition Dataset for Oromo Languag (https://arxiv.org/abs/2502.00421)
Comments:
          Accepted for ICASSP2025 (2025 IEEE International Conference on Acoustics, Speech, and Signal Processing)

- **What's New**: 이 연구에서는 에티오피아와 인근 지역에서 널리 사용되는 오로모어(Oromo language)를 위한 새로운 자동 음성 인식(ASR) 데이터셋인 'Sagalee'를 소개합니다. 이 데이터셋은 크라우드 소싱 이니셔티브를 통해 수집되었으며, 283명의 다양한 화자와 함께 100시간의 실제 오디오 녹음과 필기가 포함되어 있습니다. 이는 오로모어 ASR 자원 부족 문제를 해결하는데 중요한 기여를 하며, 데이터셋은 공공에 제공되어 연구 및 개발을 촉진할 예정입니다.

- **Technical Details**: Sagalee 데이터셋은 다양한 발음 변화를 포함하여 ASR 시스템 교육과 평가에 적합하도록 설계되었습니다. 연구자들은 Conformer 모델과 Whisper 모델을 사용하여 초기 ASR 실험을 수행하였으며, 각각 15.32%와 10.82%의 단어 오류율(Word Error Rate, WER)을 기록했습니다. 이 데이터셋의 사용은 오로모어 음성 인식 연구의 초기 베이스라인을 설정하고 향후 성능 개선 가능성을 보여줍니다.

- **Performance Highlights**: Conformer 모델의 경우, 하이브리드 손실 함수(hybrid loss function)를 사용한 결과 15.32%의 WER을 달성했습니다. 또한 Whisper 모델의 파인튜닝(fine-tuning)을 통해 성과가 크게 개선되어 10.82%의 WER을 기록했습니다. 이러한 결과는 오로모어 ASR의 고유한 도전 과제를 강조하며, Sagalee 데이터셋의 중요성을 잘 보여줍니다.



### Social media polarization during conflict: Insights from an ideological stance dataset on Israel-Palestine Reddit comments (https://arxiv.org/abs/2502.00414)
- **What's New**: 이 연구는 정치적으로 민감한 이스라엘-팔레스타인 분쟁 관련하여 9,969개의 Reddit 댓글을 분석하여 이념적 입장을 탐지하는 데 있어 기존 연구의 한계를 극복합니다. 세 가지 이념적 입장(Pro-Israel, Pro-Palestine, Neutral)으로 댓글을 분류하고, 머신러닝과 프리트레인드 언어 모델(pre-trained language models)을 포함한 다양한 접근 방식을 사용하였습니다. 공개적으로 사용 가능한 데이터셋을 제공하며, 이 연구는 정치적 발언의 이념적 입장을 탐지하는 모델의 효과성을 비교 분석합니다.

- **Technical Details**: 연구에서는 Mixtral 8x7B, Mistral 7B, Gemma 7B 및 Falcon 7B 등 4개의 오픈 액세스 LLMs와 RNN, LSTM, GRU, BiLSTM 등 4개의 신경망 아키텍처, BERT Cased, BERT Uncased와 같은 7개의 프리트레인드 언어 모델을 평가했습니다. 각 모델은 정확성(accuracy), F1-score, 재현율(recall), 정밀도(precision)를 포함한 성능 메트릭을 통해 평가되었으며, 최종 성능이 가장 뛰어난 방법이 규명되었습니다. 이 연구의 데이터셋은 이스라엘-팔레스타인 분쟁과 관련된 수동으로 주석이 달린 댓글로 구성되어 있습니다.

- **Performance Highlights**: Mixtral 8x7B의 Scoring 및 Reflective Re-read prompt가 테스트한 모든 방법 중에서 가장 높은 성능을 보였습니다. 다양한 모델을 통해 이념적 입장을 탐지하는 데 있어 성능 평가가 이루어졌으며, 각 모델의 효과적인 접근법을 제시합니다. 이는 정치적으로 민감한 소셜 미디어 맥락에서 이념적 입장 탐지의 중요성을 강조하며, 향후 연구의 방향성을 제안하고 있습니다.



### The Impact of Persona-based Political Perspectives on Hateful Content Detection (https://arxiv.org/abs/2502.00385)
- **What's New**: 이 연구는 정치적 다양성을 모델 출력에 도입하기 위해 persona-based prompting 전략이 어떻게 효과적으로 사용할 수 있는지를 조사합니다. 전통적인 방식인 정교한 정치적 pretraining에 비해, persona-based 접근 방식이 더 계산적으로 효율적이라는 점이 강조됩니다. 특히, 이 연구는 혐오 발언 탐지라는 실제 응용 분야에서 이러한 접근 방식의 실효성을 평가합니다.

- **Technical Details**: 저자들은 Hateful Memes와 MMHS150K라는 두 개의 데이터 세트를 이용하여 다양한 정치적 입장을 가진 200,000개의 synthetic personas를 분석합니다. 이들은 Political Compass Test (PCT)를 통해 정치적 입장을 매핑하고, 페르소나의 식별자가 분류 결정에 미치는 영향을 평가합니다. 연구 결과는 정책 편향이 실질적인 분류 작업에는 약한 상관관계를 가지며, 이는 기존의 정치적 전처리 방법과 다른 결과임을 보여줍니다.

- **Performance Highlights**: IDEFICS-3 모델은 persona-based prompting을 통해 Hateful Memes 데이터 세트에서 높은 성능을 기록했으며, harmfulness detection에서 0.908의 정확도와 0.890의 F1 점수를 달성했습니다. 이는 페르소나를 활용한 접근 방식이 모델의 분류 능력을 유지하면서도, 정치적 편향의 영향을 최소화할 수 있음을 시사합니다.



### When End-to-End is Overkill: Rethinking Cascaded Speech-to-Text Translation (https://arxiv.org/abs/2502.00377)
- **What's New**: 이 논문에서는 End-to-End (E2E) 음성-텍스트 변환 모델의 성공에도 불구하고, 검증된 방식으로 남아 있는 계단식(cascaded) 음성-텍스트 번역 모델의 필요성을 강조합니다. ASR(Automatic Speech Recognition)과 MT(Machine Translation) 간의 오류 전파를 최소화하기 위해, ASR로부터의 여러 후보들을 MT에 통합하는 방안을 제안합니다. 또한, 일반적인 오류의 원인과 음성 영역의 샘플 간 유사성으로 인한 차이에 대한 폭넓은 분석을 통해 논문의 독창성을 발휘하고 있습니다.

- **Technical Details**: 제안된 방법은 ASR에서 여러 후보를 통합하여 기계 번역의 정확도를 향상시키고, 자기지도(self-supervised) 음성 표현을 활용하여 언어 정보를 보존하는 방식입니다. 특히, 여러 후보 중 최선의 결과를 선택하게 하여, 음성 데이터로부터 발생하는 오차를 최소화하며, ASR과 MT 사이의 언어적 차이를 극복하는 정확한 번역을 보장합니다. 또한, 기존의 방법과 달리 추가 파라미터 설정 없이 N-best 전략을 통해 성능을 향상시킬 수 있는 점이 특징입니다.

- **Performance Highlights**: 제안된 모델은 음성-텍스트(S2T) 변환 작업에서 계단식 모델 중 최고의 성능을 자랑하며, 다양한 ASR 및 MT 사전 훈련(pre-trained) 모델을 쉽게 활용할 수 있습니다. 낮은 비용으로 대규모 데이터를 요구하지 않으면서도 빠른 학습 속도와 최소한의 데이터 사용으로 뛰어난 결과를 달성할 수 있다는 점에서도 큰 장점을 제공합니다. 최종적으로, 단일 후보를 사용하는 기존 모델과 비교했을 때, 제안된 접근 방식이 번역 품질을 크게 개선할 수 있음을 보여주고 있습니다.



### A Unit-based System and Dataset for Expressive Direct Speech-to-Speech Translation (https://arxiv.org/abs/2502.00374)
- **What's New**: 최근 음성 대 음성 번역(S2ST) 연구는 번역 정확도와 자연스러움에 초점을 맞추는 반면, 감정과 태도를 전달하는 데 필수적인 패러링귀스틱(paralinguistic) 정보와 같은 주요 요소는 간과하는 경향이 있습니다. 본 연구에서는 다양한 영화 오디오 트랙에서 신중하게 구성된 다국어 데이터셋을 소개하며, 각 데이터셋 쌍은 감정 정보와 지속 시간에 맞춰 정확하게 매칭되었습니다. 이 연구는 감성 정보를 보존하며 번역의 정확성과 자연성 또한 유지하는 S2ST 방법론을 제안합니다.

- **Technical Details**: 우리의 S2ST 시스템은 세 가지 구성 요소로 이루어져 있습니다. 우선, 하나의 언어에서 음성을 이산 단위로 변환하여 직접적인 음성 번역을 진행합니다. 그 후, 음성에서 화자의 특성을 추출하여 감성 전달 모델을 통해 목표 언어로 감성이 풍부하게 재생산된 음성을 합성합니다. 또한, HuBERT 구조를 활용하여 이산 단위를 추출하는 과정과 K-평균 알고리즘을 통한 클러스터 중심 생성을 통해 특정 언어의 음성을 연속적으로 표현할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 원음(Source speech)에서 더 많은 패러링귀스틱 정보를 유지하면서 번역의 정확성과 자연성을 높은 수준으로 유지하는 성과를 보여주었습니다. 특히, 영화 및 TV 프로그램과 같은 감정이 풍부한 대화 자료를 기반으로 한 데이터셋을 통해 감정 번역 측면에서 우수한 성능을 기록했습니다. 이는 향후 음성 번역 모델 연구에 중요한 기준점을 제공할 것입니다.



### FinchGPT: a Transformer based language model for birdsong analysis (https://arxiv.org/abs/2502.00344)
Comments:
          12 pages, 4 figures

- **What's New**: 이 논문에서는 미세한 구조를 갖는 벵갈 피치(Bengalese finch)의 노래에 대해 Transformer 아키텍처를 활용하여 장거리 의존성을 분석하는 연구를 소개합니다. FinchGPT라는 모델은 새 노래 데이터의 텍스트화를 통해 훈련되었고, 동물 음성 신호의 복잡성을 해석하는 새로운 방법론을 제시합니다. 기존의 딥러닝 모델의 한계를 넘어 비인간 언어에서도 복잡한 패턴이 존재할 수 있음을 보여주는 것이 주요 성과입니다.

- **Technical Details**: 벵갈 피치(Nichura striata domestica)의 노래 데이터는 SAIBS 프로그램을 통해 자동으로 텍스트화되었으며, 이를 활용하여 RNN, LSTM, Transformer 모델을 훈련했습니다. 모델 훈련은 지도 학습의 일환으로 교차 엔트로피 손실을 최소화하는 방향으로 진행되었으며, Adam 및 AdamW 최적화 알고리즘이 사용되었습니다. FinchGPT는 GPT-2 아키텍처와 유사하게 구성되었으며, 모델은 오직 벵갈 피치로부터 수집된 데이터만으로 훈련되었습니다.

- **Performance Highlights**: FinchGPT는 다른 아키텍처 모델에 비해 생리적 및 계산적 조작의 영향을 효과적으로 포착하여 장거리 의존성을 모델링하는 데 뛰어난 성능을 보였습니다. 주목할 점은 attentional head 분석을 통해 구조적 복잡성을 탐색할 수 있으며, 이는 동물의 음성 신호 이해에 있어 중요한 기여를 할 수 있다는 것입니다. 이 연구는 비인간 의사소통체계의 구조적 속성을 탐구하는 데 유용한 새로운 프레임워크를 제공합니다.



### Challenges and Innovations in LLM-Powered Fake News Detection: A Synthesis of Approaches and Future Directions (https://arxiv.org/abs/2502.00339)
- **What's New**: 소셜 미디어 플랫폼을 통한 가짜 뉴스의 확산은 공공 신뢰와 민주적 제도에 심각한 위험을 초래합니다. 이 문제를 해결하기 위한 새로운 검출 방법론이 필요하며, 최근의 연구들은 대형 언어 모델(LLM)의 발전과 다중 모달 프레임워크(multimodal framework)를 활용한 검출을 포함하고 있습니다. 이 논문에서는 정확성을 높이는 LLM의 중요성과 강건한 검출을 위한 크로스 모달리티(fusion) 융합의 필요성을 강조합니다.

- **Technical Details**: 가짜 뉴스의 검출을 위한 다양한 접근 방식이 논의되고 있으며, 특히 그래프(graph) 기반 방법론 및 적대적 훈련(adversarial training)이 주목받고 있습니다. 그러나 동적인 소셜 미디어 트렌드와 실시간 검출 능력의 적응성, LLM 오용에 따른 윤리적 문제에서 중요한 격차가 발견되었습니다. 따라서 우선적으로 스타일 독립적인 모델과 다국어 검출 프레임워크의 개발이 필요하다고 강조합니다.

- **Performance Highlights**: 미래 방향으로는 LLM 기반 잘못된 정보(misinformation)를 완화하기 위한 강력한 정책 수립이 제안되고 있습니다. 이 논문은 디지털 환경에서 증가하는 복잡성을 지속적으로 극복할 수 있는 가짜 뉴스 검출 시스템 강화를 위한 기초를 제공합니다. 연구자와 실무자들에게 유용한 통찰력을 제공하며, 더 나은 검출 기술의 개발을 위해 중요한 기반을 마련합니다.



### UGPhysics: A Comprehensive Benchmark for Undergraduate Physics Reasoning with Large Language Models (https://arxiv.org/abs/2502.00334)
Comments:
          9 pages

- **What's New**: 본 논문은 LLMs의 물리학 문제 해결 능력을 평가하기 위해 UGPhysics라는 대규모 물리학 벤치마크를 소개합니다. UGPhysics는 5,520개의 학부 수준 물리학 문제를 포함하고 있으며, 13개 과목과 4개의 물리학 추론 기술을 다룹니다. 이 연구는 기존의 교육 평가에서 활용되는 물리학 문제의 범위를 효과적으로 반영하고 있습니다.

- **Technical Details**: UGPhysics 벤치마크는 철저한 데이터 검증 과정을 통해 5,520개의 문제를 수집, 형식화, 분할 및 필터링했습니다. 이 문제들은 영어와 중국어로 번역되어 총 11,040개의 문제로 나뉘어 평가됩니다. 문제들에는 독립형 답변 타입과 복합형 답변 타입이 있으며, 이들을 해결하기 위해 필요한 기술이 네 가지로 분류됩니다.

- **Performance Highlights**: 본 논문은 31개의 주요 LLM들을 평가한 결과, OpenAI-o1-mini가 49.8%의 최고 정확도를 기록하는 것을 보여주었습니다. 이러한 결과는 현재 LLM들이 물리학 문제 해결에서 직면하고 있는 과제를 강조하며, 수학적 능력 이상의 물리학적 추론 기술이 필요한 필요성을 부각시킵니다. UGPhysics와 MARJ는 물리학 추론을 위한 AI 발전의 중요한 촉매제가 될 것으로 기대됩니다.



### MODS: Moderating a Mixture of Document Speakers to Summarize Debatable Queries in Document Collections (https://arxiv.org/abs/2502.00322)
Comments:
          Accepted at NAACL 2025(main)

- **What's New**: 이번 논문에서는 과거의 쿼리 중심 요약(Query-focused summarization, QFS)이 단일 답변을 가정하고, 논란이 있는 질문에 대한 균형 잡힌 요약을 생성하는 데 실패하는 문제를 다루고 있습니다. 저자들은 Debatable QFS (DQFS)라는 새로운 작업을 소개하며, 서로 대립하는 관점을 포함한 문서들로부터 포괄적이고 균형 잡힌 요약을 생성하는 방법을 제시합니다. 이 과제를 해결하기 위해 MODS라는 다중 LLM 프레임워크를 설계하여 사람의 패널 토론을 모방하는 방식을 사용했습니다.

- **Technical Details**: MODS는 각 문서를 개별 Speaker LLM으로 처리하고, Moderator LLM이 맞춤형 쿼리에 응답하도록 해당 Speaker를 선택하는 구조입니다. 이 시스템은 각 주제에 대해 최적화된 쿼리를 사용하여 관련 문서의 컨텍스트를 검색하고, 각 Speaker의 관점을 정리된 아웃라인에 기록하여 최종 요약을 안내합니다. 이렇게 함으로써 MODS는 정보를 고르게 대표하고 다양한 관점을 내포한 요약을 생성하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, MODS는 ConflictingQA 및 새로운 DebateQFS 데이터셋에서 최첨단(SoTA) 모델보다 38-59% 더 높은 주제 문단 커버리지와 균형성을 달성했습니다. 사용자들은 MODS의 요약이 읽기 쉽고 균형 잡혀 있다고 평가했으며, 이는 보다 많은 문서에서 온 관점을 포함하면서도 가독성을 유지했습니다. 이러한 결과는 MODS가 쿼리를 맞춤화하고 아웃라인을 통해 사용자의 요약 경험을 풍부하게 할 수 있음을 시사합니다.



### DEUCE: Dual-diversity Enhancement and Uncertainty-awareness for Cold-start Active Learning (https://arxiv.org/abs/2502.00305)
Comments:
          18 pages, 3 figures, 12 tables. Accepted manuscript by TACL. For published version by MIT Press, see this https URL

- **What's New**: 이 논문에서는 Cold-start Active Learning (CSAL)에 대한 새로운 접근법을 제시합니다. 기존 방법들이 약한 클래스와 어려운 대표 예시를 무시했던 문제를 해결하기 위해, Dual-Diversity Enhancing and Uncertainty-aware (DEUCE) 프레임워크를 제안합니다. DEUCE는 사전 훈련된 언어 모델(PLM)을 활용하여 텍스트 표현과 예측 불확실성을 효율적으로 추출합니다.

- **Technical Details**: DEUCE 프레임워크는 Dual-Neighbor Graph (DNG)를 구성하여 텍스트 다양성과 클래스 다양성을 결합합니다. 이는 데이터 분포를 균형 있게 만들고, 밀도 기반 클러스터링을 통해 불확실성 정보를 전파하여 어려운 대표 사례를 선택합니다. 이러한 접근법은 CSAL에서 클래스 균형과 하드 대표 데이터 선택에 효과적입니다.

- **Performance Highlights**: DEUCE는 여섯 개의 NLP 데이터세트에서 실험을 통해 그 우수성과 효율성을 입증했습니다. 이 프레임워크는 탐색과 활용 간의 균형을 잘 이루어, CSAL에서의 데이터 수집 성능을 향상시키는 것을 목표로 합니다. DEUCE는 텍스트와 클래스 다양성을 동시에 고려하여, CSAL의 클래스 불균형 문제를 해결하는데 기여합니다.



### Contextual Morphogenesis in Large Language Models: A Novel Approach to Self-Organizing Token Representations (https://arxiv.org/abs/2502.00301)
- **What's New**: 이번 연구는 언어 모델의 토큰 표현 방식에 대한 새로운 접근 방식인 contextual morphogenesis를 제안합니다. 기존의 고정된 토큰화 방식과 달리, 이 방법은 학습된 context에 따라 토큰 경계를 동적으로 재구성하여 언어 모델의 대표성을 향상시킵니다. 이를 통해 다양한 언어적인 맥락에 더 잘 적응할 수 있는 가능성을 탐색하고 있습니다.

- **Technical Details**: 연구에서는 기존의 대형 언어 모델에 contextual morphogenesis를 통합하는 실험적 프레임워크를 제시합니다. 토큰 embeddings가 계속 진화할 수 있도록 self-attention 메커니즘과 embedding 공간을 수정하여, 동적인 구조 조정을 가능하게 합니다. 이 방법은 강조된 의미적 및 구문적 특성에 기반하여 모델 내부 상태에서 유추된 정보를 반영하여 토큰 경계를 조정합니다.

- **Performance Highlights**: 실험 결과는 self-organizing token representations가 모델 성능과 표현 일관성에 미치는 긍정적인 영향을 보여줍니다. 기존의 토큰화 방식과 비교했을 때, 동적으로 조정된 토큰화를 사용한 모델이 더 낮은 perplexity를 기록하면서도 표현 안정성을 유지하였으며, 이는 언어 모델의 예측 정확도를 개선하는 데 도움이 되는 것으로 평가되었습니다.



### ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inferenc (https://arxiv.org/abs/2502.00299)
Comments:
          35 pages

- **What's New**: 본 논문에서는 기존의 KV 캐시 압축 방법이 언어의 실제 특성을 반영하지 못하고 토큰 간의 의존성을 고려하지 않는 문제를 지적하고, 이를 해결하기 위해 ChunkKV라는 새로운 접근법을 제시합니다. ChunkKV는 토큰들을 그룹으로 묶어 기본 압축 단위로 활용하며, 중요도가 낮은 정보를 버리고 가장 의미 있는 의미 단위를 보존합니다. 또한 ChunkKV가 다양한 레이어 간의 보존된 인덱스에서 더 높은 유사성을 보여주므로, 레이어-와이즈 인덱스 재사용 기법을 통해 추가적인 계산 오버헤드를 줄입니다.

- **Technical Details**: ChunkKV는 기존의 KV 캐시 압축 기술이 토큰의 중요성을 개별적으로 평가하는 것을 넘어, 연속적인 의미 정보 단위를 다룰 수 있는 방법을 도입합니다. 구체적으로, ChunkKV는 의미 덩어리가 문맥적으로 중요한 경우에는 이를 유지하여 유의미한 정보 손실을 방지하고, 효율성을 높이는 레이어-와이즈 인덱스 재사용 기법을 통해 계산 부하를 줄입니다. 이러한 방법론은 여러 최신 긴 문맥 벤치마크에서 평가되었으며, 다양한 모델에서 실험을 통해 그 효과가 입증되었습니다.

- **Performance Highlights**: ChunkKV는 LongBench, Needle-In-A-HayStack 및 GSM8K, JailbreakV와 같은 최신 긴 문맥 벤치마크에서 기존 방법들보다 최대 10% 성능 향상을 기록했습니다. 이러한 성능 개선은 선택적인 의미 덩어리 보존 능력에 기인하며, 효율성과 정확성 모두에서 기존의 KV 캐시 압축 방법을 초월합니다. 이 연구 결과는 ChunkKV가 KV 캐시 압축을 위한 단순하면서도 효과적인 접근법으로 자리잡을 수 있음을 보여줍니다.



### Estimating LLM Uncertainty with Logits (https://arxiv.org/abs/2502.00290)
- **What's New**: 이 논문에서는 LLM의 토큰별 불확실성을 실시간으로 추정하기 위한 새로운 프레임워크인 Logits-induced Token Uncertainty (LogU)를 제안합니다. 기존의 확률 기반 접근법의 한계를 극복하고, 다양한 응답에 대한 신뢰도를 평가할 수 있는 효율적이고 효과적인 방법으로 구체화되었습니다. LogU는 주로 aleatoric uncertainty와 epistemic uncertainty를 분리하여 토큰 수준에서 불확실성을 명확하게 추정할 수 있는 기능을 제공합니다.

- **Technical Details**: LogU는 디리클레 분포를 활용하여 aleatoric uncertainty와 epistemic uncertainty를 구분하고 평가합니다. 이 모델은 샘플링 없이도 각 응답의 신뢰도를 실시간으로 추정할 수 있으며, 사용자에게 더 정확한 피드백을 제공할 수 있습니다. LogU는 LLM의 본질적인 불확실성을 포착할 수 있는 새로운 기준을 제공하며, 효과적인 증거 모델링을 통해 이를 구현합니다.

- **Performance Highlights**: 실험 결과 LogU의 효과성이 입증되었으며, 이 방법을 통해 LLM의 할루시네이션 문제를 해결하는 데 기여할 수 있는 잠재력이 나타났습니다. LogU는 주요 토큰의 신뢰도를 평가하는 데 집중함으로써 신뢰할 수 없는 응답을 줄이고, 다운스트림 작업에 대한 가이드를 제공하는 데 유용합니다. 이러한 성과는 LLM의 안정성을 개선하는데 중요한 발전을 의미합니다.



### Scaling Flaws of Verifier-Guided Search in Mathematical Reasoning (https://arxiv.org/abs/2502.00271)
- **What's New**: 이 논문은 다단계 추론(multi-step reasoning)에 대한 LLM의 성능 향상을 위한 검증자 가이드를 통한 검색 방법의 한계를 분석합니다. 특히 샘플 사이즈가 증가할 때 검증자 가이드 검색이 겪는 성능 저하, 즉 '스케일링 결함(scaling flaws)'을 강조하고 있습니다. 이 연구는 문제의 난이도 및 솔루션 밀도가 높은 경우 이 현상이 더욱 두드러진다는 점을 지적합니다.

- **Technical Details**: 연구에서는 두 가지 검증 모델, 즉 결과 값 모델(Outcome Value Models)과 과정 보상 모델(Process Reward Models)을 사용하여 후보 경로의 평가 및 선택 과정에서의 한계를 살펴보았습니다. 또한, 반복 샘플링(repeated sampling)과 비교하여 검증자 가이드 검색의 성능이 감소하는 원인을 검증자 실패(verifier failures)로 규명하였습니다. 이러한 실패는 검증자가 후보를 잘못 순위 매기고 유효한 경로를 잘못 잘라내는 것으로 나타났습니다.

- **Performance Highlights**: 이 논문은 검증자 가이드 검색에서의 결함과 이로 인해 LLM의 성능이 저하되는 구체적인 사례를 조사하며, 검증자 의존도를 줄이기 위한 두 가지 간단한 접근법을 제안합니다. 결과적으로, 이 접근법은 기존의 검증자 가이드 검색 접근법의 근본적인 한계를 드러내며, 미래의 연구 방향을 제시합니다. 실험 결과는 어려운 문제와 분포 외(out-of-distribution) 문제에 대한 솔루션 제공에 대해 심각한 우려를 드러냅니다.



### Context-Preserving Tensorial Reconfiguration in Large Language Model Training (https://arxiv.org/abs/2502.00246)
- **What's New**: 이번 논문에서는 기존의 모델들이 직면한 장기 의존성 처리의 한계를 극복하기 위한 새로운 접근법인 Context-Preserving Tensorial Reconfiguration (CPTR)를 제안합니다. CPTR는 구조화된 팩토리제이션과 적응형 수축을 통해 가중치 텐서를 동적으로 재조직함으로써, 계산 오버헤드 없이 향상된 상황 통합을 가능하게 합니다. 연구 결과, CPTR를 통합한 모델들이 긴 시퀀스에서 일관성을 유지하면서도 향상된 언어 생성을 보여줍니다.

- **Technical Details**: CPTR의 기본 원리는 LLM 내부의 텐서 구조를 재구성하는 데 중점을 두고 있습니다. 이는 텐서 분해와 수축과 같은 고급 텐서 연산을 통해 수행되며, 모델이 복잡한 언어 데이터 내에서 장기 의존성을 더 효율적으로 처리할 수 있도록 돕습니다. 기존의 전통적인 주의 메커니즘에서 발생하는 한계를 극복하기 위해 CPTR는 데이터가 네트워크를 통과할 때 텐서 표현을 동적으로 재구성하는 모듈을 삽입하여 아키텍처를 수정하였습니다.

- **Performance Highlights**: 경험적 평가에 따르면, CPTR는 긴 문서 및 대화와 같은 태스크에서의 당혹감(perplexity) 저감 및 회상 정확도 향상에 기여합니다. CPTR 개선 모델들은 경쟁력을 유지하면서도 더 나은 계산 효율성과 메모리 사용량 감소를 보였습니다. 또한 기울기 안정성 지표를 통해 훈련 효율성의 향상이 입증되었으며, 이는 가중치 업데이트의 변동성을 보다 통제된 방식으로 표현합니다.



### Resolving Editing-Unlearning Conflicts: A Knowledge Codebook Framework for Large Language Model Updating (https://arxiv.org/abs/2502.00158)
- **What's New**: 이번 연구는 LLM(대형 언어 모델)의 최신 지식을 업데이트하기 위한 새로운 접근 방식인 LOKA를 제안합니다. LOKA는 지식 코드북을 기반으로 하여 비충돌(conflict-free) 프레임워크를 도입합니다. 이 프레임워크는 업데이트된 지식을 여러 메모리에 저장하고, 유사성을 고려한 지식 매핑을 통해 관련된 지식을 클러스터링하여 저장합니다.

- **Technical Details**: LOKA는 훈련 시 업데이트된 지식을 여러 코드북 메모리에 저장하고, 각 메모리는 편집된 지식과 삭제된 지식을 포함하는 지식 조각 그룹을 저장합니다. 편집과 비우기 작업 간의 충돌 문제를 해결하기 위해 태스크별(task-specific) 및 다중 태스크(multi-task) 메모리를 사용하고, 이러한 메모리의 활성화를 조정하기 위해 학습 기반 라우터(router)를 도입합니다.

- **Performance Highlights**: 광범위한 실험을 통해 LOKA의 효과성을 입증했으며, 기존의 비례한 지식 해결 방법의 한계를 극복했습니다. LOKA는 고유한 지식 할당 및 검색 메커니즘을 사용하는 동시에, 지식 업데이트가 요구되는 과제에 대해 뛰어난 성능을 발휘합니다. 새로운 벤치마크를 기반으로하여 LLM 지식 업데이트 작업을 평가하는 기준을 수립했습니다.



### A Three-Branch Checks-and-Balances Frameworkfor Context-Aware Ethical Alignment of Large Language Models (https://arxiv.org/abs/2502.00136)
Comments:
          17 pages, 6 tables, 6 figures. arXiv admin note: substantial text overlap with arXiv:2405.07076

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 윤리적 정렬을 위한 삼원 체크 앤 밸런스(framework)를 제안합니다. 정부 시스템에서 영감을 받아 세 가지 독립적이면서 상호 작용하는 요소들, 즉 지식 생성을 위한 실행(branch), 윤리적 가드레일을 설정하는 입법(branch), 그리고 맥락 해석을 위한 사법(branch)을 구현하고 있습니다.

- **Technical Details**: 이 프레임워크에서는 DIKE라는 입법 부문과 ERIS라는 사법 부문이 상호작용하며 적대적(dual) 관계를 형성하여 다양한 문화적 맥락에 맞춰 적응할 수 있도록 합니다. 이 구조는 인간 피드백을 통한 강화 학습(RLHF)의 한계를 극복하며 해석 가능하고 적응적이며 문화적 인식을 고려한 윤리적 추론을 제공합니다.

- **Performance Highlights**: 자기 지도 학습(self-supervised learning)과 적대적 테스트(adversarial testing)를 통해 프레임워크가 감정 모델링(emotional modeling)을 활용해 언어 행동을 윤리적 결과로 유도할 수 있는지를 보여주며, 지식 생성, 윤리적 감독, 맥락 해석 간의 독립성을 유지합니다.



### Sparse Autoencoder Insights on Voice Embeddings (https://arxiv.org/abs/2502.00127)
- **What's New**: 최근 설명 가능한 머신러닝의 발전은 sparse autoencoder가 밀집 인코딩된 임베딩에서 단일 의미적 특징을 발견하는 데 효과적임을 강조했습니다. 본 연구에서는 Titanet 모델로 생성된 화자 임베딩에 이 기술을 적용하였고, 이 비텍스트 기반 데이터에서 mono-semantic features를 추출하는 효과를 입증했습니다. 흥미롭게도 추출된 특징들은 Large Language Model(LLM) 임베딩에서 관찰된 특징 분할 및 조작 특성을 공유합니다.

- **Technical Details**: 이 연구의 목적은 sparse autoencoder 모델이 비텍스트 임베딩 데이터에서 mono-semantic features를 추출할 수 있는지를 조사하는 것입니다. Titanet 모델을 사용하여 다양한 화자 특성 임베딩을 생성하고, 이를 기반으로 sparse latent space를 학습하는 여러 SAEs를 훈련합니다. 이 과정에서 latent space의 특징과 동작을 분석하고, L1 정규화를 통해 sparsity를 적용하여 중요하지 않은 latent 요소를 제거하는 방법이 사용됩니다.

- **Performance Highlights**: 연구 결과, 화자의 언어 및 음악과 같은 특성이 잘 구분되며, latent feature의 조작을 통해 재구성된 화자 임베딩의 영향을 관찰할 수 있었습니다. 이 연구는 sparse autoencoders가 오디오 기반 화자 인식 및 데이터 해석을 위한 강력한 도구가 될 수 있음을 보여줍니다. 이러한 발견은 향후 다양한 데이터 도메인에서의 적용 가능성을 높이며, ML 모델의 해석 가능성을 향상시키는 데 기여할 것입니다.



### Disambiguating Numeral Sequences to Decipher Ancient Accounting Corpora (https://arxiv.org/abs/2502.00090)
- **What's New**: 이 논문은 고대의 부분적으로 해독된 프로토 엘라미트(PE) 문서에서 수치 기호의 모호성을 구별하는 방법을 제안합니다. 연구자들은 PE 숫자 표기의 가능한 읽기 목록을 알고리즘적으로 추출하고, 원본 문서의 구조적 속성과 부트스트랩(classifier) 알고리즘으로 학습한 분류기를 기반으로 한 두 가지 해독 기술을 기여합니다. 또한, 이 논문에서는 해독 기술을 평가하기 위한 테스트 세트와 부트스트랩 분류기를 위한 조심스러운 규칙 선택의 새로운 접근 방식을 제시합니다.

- **Technical Details**: 논문은 CDLI에서 제공되는 변환된 PE 말뭉치(corpus)를 기반으로 합니다. PE 숫자 표기는 일반적으로 4가지 주요 숫자 체계인 10진(D), 육십진(S), 양육십진(B), 그리고 용적(C) 중 하나를 사용하여 작성됩니다. 특히, PE 기호는 고대 아랍-힌두 숫자와 달리 고정된 값으로 구성되어 있으며, 더 큰 값은 동일한 기호를 반복해서 나타냅니다.

- **Performance Highlights**: 연구 결과는 제안된 해독 기술이 PE 문자의 이해를 확장하는 데 기여한다는 것을 보여줍니다. 이 기술을 통해 문서 내용과 숫자 크기 간의 이전에 알려지지 않은 상관관계가 밝혀졌습니다. PE 숫자 표기에 대한 대규모 조사 결과 먼저 제안된 기술들은 기존의 직관을 뒷받침하고, 고대 문서의 이해를 심화하는 데 중요한 역할을 합니다.



### Ensembles of Low-Rank Expert Adapters (https://arxiv.org/abs/2502.00089)
Comments:
          29 pages, 5 figures, 5 tables; proceedings in ICLR 2025

- **What's New**: 본 논문에서는 다양한 텍스트 데이터로부터 발생하는 gradient 방향의 충돌 문제를 해결하기 위해 Ensembles of Low-Rank Expert Adapters(ELREA)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 다중 미세 조정 작업에서 모델의 전문성을 높이고, 특정 작업에 대한 데이터의 중요성을 활용하여 우수한 성능을 달성할 수 있도록 설계되었습니다. ELREA는 Low-Rank Adaptation(LoRA) 기술을 사용하여 훈련 명령을 군집화하고, 이를 통해 모델의 최적화 과정을 간소화하며, 예측 과정 중에 가장 관련성이 높은 전문가 어댑터를 결합합니다.

- **Technical Details**: ELREA 프레임워크는 기본 어댑터를 전체 데이터셋에 대해 미세 조정하여 일반적인 지식을 습득한 후, 각 데이터 포인트의 gradient를 평가하여 군집화합니다. 이후, 각 군집에 대해 LoRA 전문가 어댑터를 훈련시키고, 이러한 전문가 어댑터는 입력 데이터의 gradient 유사성에 근거하여 예측 결과를 조합하는 방식으로 작동합니다. 이를 통해 ELREA는 기존 Deep Ensembles 기법보다 더 효율적이며, 추가적인 작업별 유효성 검증 데이터 없이도 작업 분류가 가능합니다.

- **Performance Highlights**: 실험 결과, ELREA는 전체 데이터셋에 대해 훈련된 baseline LoRA 어댑터 및 다른 Mixture of Experts(MoE) 기법을 능가하는 성능을 보였습니다. 다양한 도메인 특화 작업에 걸쳐 강력한 성능을 발휘하며, 모델의 확장성과 효율성을 동시에 유지할 수 있어 실제 애플리케이션에서 매우 유용한 선택이 될 수 있습니다. 본 논문에서 제안한 방법은 복잡한 자동화 작업을 처리하는 데 필수적인 모델 일반화 문제를 효과적으로 해결하는 데 기여합니다.



### Efficient Beam Search for Large Language Models Using Trie-Based Decoding (https://arxiv.org/abs/2502.00085)
Comments:
          9 pages

- **What's New**: 이번 연구에서는 Transformer 기반의 시퀀스-투-시퀀스 생성에서 빔 서치(beam search)의 메모리 비효율성을 해결하기 위한 새로운 트라이(trie) 기반 병렬 디코딩 방법을 소개합니다. 기존의 빔 서치 방식은 순차적 또는 배치 기반 접근법을 채택하여 각각의 단점을 지니고 있었으나, 본 연구의 방법은 이를 극복합니다.

- **Technical Details**: 제안된 방법은 공통 접두사를(shared prefix) 가진 모든 빔이 단일 KV 캐시(key-value cache)를 공유하는 방식을 통해 메모리 소비를 획기적으로 줄입니다. 이 방식은 모든 브랜치에서 병렬 디코딩(parallel decoding)을 가능하게 하며, 메모리 효율성을 극대화합니다.

- **Performance Highlights**: 트라이 기반 접근 방식을 활용하여 빔 서치의 메모리 사용량을 크게 줄이는 동시에 인퍼런스 속도를 유지할 수 있습니다. 이러한 특성 덕분에 메모리가 제한된 환경이나 대규모 모델 배치에 특히 적합합니다.



### BTS: Harmonizing Specialized Experts into a Generalist LLM (https://arxiv.org/abs/2502.00075)
- **What's New**: BTS(Branch-Train-Stitch)라는 새로운 알고리즘을 제안하여 독립적으로 훈련된 대규모 언어 모델(LLM) 전문가들을 결합하여 단일의 범용 모델을 생성합니다. 이 알고리즘은 경량 스티치 레이어(lightweight stitch layers)를 사용하여 전문가들의 연결성 없이도 전문가 모델들을 통합할 수 있는 유연한 접근 방식을 제공합니다. BTS는 기존의 LLM를 변경하지 않고도 전문가를 추가하거나 제거할 수 있는 modular한 구조를 갖추고 있습니다.

- **Technical Details**: BTS는 우선 도메인 전문가들을 독립적으로 훈련한 후, 스티치 레이어를 삽입하여 전문가 모델들을 통합한 후 범용 모델로 변환합니다. 이 구조는 각 전문가 모델 간의 연결성을 부여하는 게이팅 메커니즘을 사용하여 서로 다른 전문가의 정보를 통합할 수 있도록 하여, 새로운 도메인에 대한 일반화 능력을 강화합니다. BTS는 사용 중인 세드 모델(seed model)을 중심으로 하여 각 전문가의 표현을 융합하는 허브-스포크(hub-and-spoke) 모델을 채택합니다.

- **Performance Highlights**: BTS는 다양한 하위 작업에서 다른 전문가 병합 및 업사이클링 기법보다 우수한 성능을 보여줍니다. 실험 결과, BTS는 세드 모델 대신 한정된 훈련 데이터로도 효과적으로 작동하여 각 전문 분야에 대한 성능을 시연합니다. 이 구조적 디자인은 해석 가능성을 제공하며, 모델의 결정 과정에 대한 투명성을 확보할 수 있습니다.



### A Multi-Layered Large Language Model Framework for Disease Prediction (https://arxiv.org/abs/2502.00063)
- **What's New**: 이 연구는 소셜 원격의료가 의료 분야에서 어떻게 혁신을 가져왔는지를 다룹니다. 특히, COVID-19 팬데믹 동안 사용자-generated 데이터가 의료 거점으로 활용되는 사례를 제시합니다. 본 연구는 LLM(대형 언어 모델)을 활용하여 아랍어 의료 데이터의 전처리 단계에서 효율성을 증대시키고자 하는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 연구에서는 LLAMA3, GPT-3.5 Turbo, BERT 등의 LLM을 사용하여 아랍어 의료 텍스트를 처리하고, 주로 텍스트 요약(text summarization), 텍스트 개량(text refinement), 그리고 NER(Named Entity Recognition) 기법을 적용합니다. 이를 통해 CAMeL-BERT, AraBERT, Asafaya-BERT 모델을 파인튜닝하여 질병 분류 및 증상 심각도 평가의 정확도를 향상시켰습니다.

- **Performance Highlights**: CAMeL-BERT 모델은 NER이 보강된 텍스트를 사용하여 83%의 질병 유형 분류 및 69%의 심각도 평가 성과를 달성했습니다. 반면, 미세조정되지 않은 모델들은 13-20%의 유형 분류와 40-49%의 심각도 평가 성과로 낮은 성능을 보였습니다. 이러한 결과는 LLM을 통합한 원격의료 시스템이 진단 정확도 및 치료 성과를 크게 향상시킬 수 있음을 시사합니다.



### MALT: Mechanistic Ablation of Lossy Translation in LLMs for a Low-Resource Language: Urdu (https://arxiv.org/abs/2502.00041)
- **What's New**: 이 연구는 LLMs(대형 언어 모델)가 저자원 언어인 우르두어 처리에서 겪는 문제를 탐구하여, 기존 연구의 한계를 극복하고 LLM의 효과성을 높이는 새로운 접근 방식을 제안합니다. 특히 LLM이 주로 영어 데이터로 훈련되었기 때문에 저자원 언어에서의 성능이 크게 떨어진다는 점을 강조합니다. 연구에서는 번역 기능을 기계적으로 제거하고 별도의 번역 모델을 사용하여 성능을 향상시키고 문화적 뉘앙스를 유지하는 방법을 설명합니다.

- **Technical Details**: 연구는 총 239개의 질문을 포함한 다양성을 갖춘 데이터셋을 생성하였으며, 이 중 15개는 번역 기능 확인에 사용되고 나머지 223개는 평가에 사용됩니다. LLM의 성능을 평균화하기 위해, 작은 파라미터 수 (2~3억)를 가진 Gemma-2-2b 및 Llama-3.2-3b 모델을 사용하였습니다. 이 연구에서 사용된 기계 번역 모델은 한국어 특정 체크포인트인 mBART 모델을 활용하여 높은 번역 정확도를 목표로 하였습니다.

- **Performance Highlights**: Edited LLM에서 나타난 오류 유형으로는 유창성 오류, 반복 오류, 비관련 오류가 있으며, 이는 LLM의 포물선 의미성 때문일 수 있습니다. 실험 결과, Llama-3.2-3b 모델은 MALT 조건 하에서 11.6%에서 유의미한 향상을 보여 주었습니다. 이는 저자원 언어에 대한 LLM의 이해도가 우수하지만, 생성 과정에서는 여전히 성능 향상이 필요하다는 사실을 시사합니다.



### Learning to Generate Unit Tests for Automated Debugging (https://arxiv.org/abs/2502.01619)
Comments:
          First two authors contributed equally. Dataset and Code: this https URL

- **What's New**: 최근 논문에서는 Unit tests (UTs)가 코드 정확성을 평가하고 코드 디버깅에 도움을 주는 자동화된 테스트 생성의 중요성을 다루고 있습니다. UT를 효과적으로 생성하여 오류를 드러내고, 예상 출력을 제공하는 시스템인 UTGen을 제안합니다. 이를 UTDebug라는 디버깅 파이프라인과 통합하여, 모델이 효과적으로 디버깅을 수행할 수 있도록 지원합니다.

- **Technical Details**: UTGen은 LLMs가 주어진 코드와 작업 설명을 바탕으로 오류를 드러내는 unit test inputs와 그에 대한 예상 출력을 생성하도록 훈련됩니다. 이 시스템은 여러 개의 생성된 UT를 기반으로 수정 사항을 확인하고 수정하여 과적합을 방지하며, UT의 출력 정확도를 높이는 방법을 모색합니다. UTGen의 성능은 7.59%로 기존 UT 생성 기준과 비교하여 우수함을 입증합니다.

- **Performance Highlights**: UTDebug를 활용했을 때, UTGen의 유닛 테스트에서 받은 피드백은 Qwen-2.5 7B 모델의 HumanEvalFix 및 MBPP+ 디버깅 세트에서 pass@1 정확도를 각각 3%와 12.35% 향상시켰습니다. 이는 모델 이용 시 UT 생성의 중요성과 효과적인 코드 디버깅의 가능성을 보여줍니다.



### VisTA: Vision-Text Alignment Model with Contrastive Learning using Multimodal Data for Evidence-Driven, Reliable, and Explainable Alzheimer's Disease Diagnosis (https://arxiv.org/abs/2502.01535)
- **What's New**: 본 논문에서 제안하는 VisTA(비전-텍스트 정렬 모델)는 고차원 방사선 이미지를 통해 알츠하이머병(AD)을 진단하기 위한 혁신적인 다중모달 언어-비전 모델입니다. VisTA는 대조 학습(constrative learning)을 활용하여 질병 예측 및 임상 의사 결정에 대한 해석 가능성을 높이는 방향으로 최적화되었습니다. 이는 의사들이 기계 학습 기반 예측을 정당화할 수 있는 임상 증거를 요구하는 현재의 요구에 부합합니다.

- **Technical Details**: VisTA는 의료 이미징을 위한 사전 훈련된 언어-비전 모델인 BiomedCLIP에서 개발되었습니다. 이 모델은 검증된 이상과 그 설명을 정렬하기 위해 대조 학습으로 미세 조정(fine-tuned)되었습니다. VisTA는 예측된 이상 유형, 참조 사례와의 유사성, 증거 기반 설명 및 최종 AD 진단 등 네 가지 출력을 생성하여 임상의들이 중간 결과를 확인하고 잠재적인 오류를 파악하도록 지원합니다.

- **Performance Highlights**: VisTA는 170개의 샘플만을 사용하여 미세 조정한 후 비약적인 개선을 이루었으며, 이상 검색에서는 74%의 정확도와 0.87의 AUC를 달성했습니다. 또한, 치매 예측에서는 88%의 정확도와 0.82의 AUC를 기록하였으며, 이는 이전 모델에 비해 현저한 성능 향상을 보여줍니다. 이 모델이 생성한 설명은 인간 전문가의 평가와 높은 일치를 보이며, 진단 과정을 명확하게 해석할 수 있는 통찰력을 제공합니다.



### Preference Leakage: A Contamination Problem in LLM-as-a-judg (https://arxiv.org/abs/2502.01534)
Comments:
          17 pages, 8 figures

- **What's New**: 이번 연구에서는 LLM(큰 언어 모델) 기반의 데이터 주석 방법에서 발생할 수 있는 오염 문제인 '선호 누출(preference leakage)'을 다룹니다. 주로 LLM을 평가자로 사용할 때 합성 데이터 생성기와 평가자 간의 관련성으로 인한 것입니다. 이러한 새로운 모델 개발 패러다임의 효과에 대한 연구가 부족했던 점을 지적하며, 선호 누출이 모델 훈련 및 평가의 효율성을 높이는 과정에서 주의해야 할 문제임을 강조합니다.

- **Technical Details**: 연구에서는 데이터 생성기 LLM과 평가자 LLM 간의 세 가지 일반적인 관련성을 정의합니다. 이들은 동일한 모델, 상속 관계가 있는 모델, 같은 모델 계열에 속하는 경우입니다. 이러한 관계에 따라 여러 LLM 기반의 실험을 통해 평가자가 자신과 관련된 학생 모델에 대한 편견이 존재함을 실증적으로 확인하였습니다.

- **Performance Highlights**: 선호 누출 문제는 기존에 알려진 LLM-as-a-judge 시나리오의 편향 문제보다 검출하기 어려운 만연한 문제임을 알게 되었습니다. 이러한 결과들은 LLM-as-a-judge 분야에서 선호 누출이 광범위하고 도전적인 문제임을 시사하며, 향후 연구에서 필수적으로 고려해야 할 요소입니다.



### The in-context inductive biases of vision-language models differ across modalities (https://arxiv.org/abs/2502.01530)
Comments:
          10 pages

- **What's New**: 본 연구는 현대 비전-언어 모델(Vision-Language Models, VLMs)이 시각적 자극과 텍스트 자극을 통해 학습할 때 팀바이러스(Inductive Biases)가 어떻게 다르게 나타나는지 분석합니다. 특히 색상(color)과 형태(shape)의 두 가지 특징의 차이를 비교하면서, 예시의 제시 방식이 일반화(generalization)에 미치는 영향을 연구합니다. 이러한 결과는 인공지능(AI) 모델의 맥락(context) 내 학습 과정에 대한 이해를 높이고, 실제 응용에 실질적인 시사점을 제공합니다.

- **Technical Details**: 이 연구에서는 이미지와 텍스트에서의 카테고리 학습(paradigms)을 통한 모델의 일반화 성향을 조사하기 위해 세 가지 실험 패러다임을 사용했습니다. 연구진은 특정 특징이 나타나는 것을 기준으로 모델이 일반화하는 경향을 비교하기 위해, 시각적 자극인 이미지를 통한 학습과 텍스트 설명을 통한 학습을 구분하고, 각 모드에서 모델의 반응을 분석했습니다. 이를 통해 형태 바이어스(shape bias)와 색상 바이어스(color bias), 그리고 형태와 색상의 혼합 형태를 비교했습니다.

- **Performance Highlights**: 연구 결과, VLMs는 이미지로부터 학습할 때 형태에 더 강한 바이어스를 보이는 경향이 있음이 밝혀졌습니다. 텍스트로 자극이 제시될 경우, 형용사(adjectives)의 순서 역시 모델의 일반화 경향에 영향을 미치며, 첫 번째 형용사가 선호됩니다. 그러나 이러한 경향은 모델의 구조나 작업(Task)의 유형에 따라 달라질 수 있습니다.



### Explaining Context Length Scaling and Bounds for Language Models (https://arxiv.org/abs/2502.01481)
Comments:
          19 pages, 14 figures

- **What's New**: 본 논문에서는 긴 컨텍스트가 언어 모델에 미치는 영향을 이해하기 위해 새로운 이론적 프레임워크를 제안합니다. 이 프레임워크는 Intrinsic Space 관점에서 출발하여 Cross Entropy Loss, Intrinsic Dimension, 그리고 Context Length 간의 관계를 설명합니다. 특히, 훈련 데이터의 양에 따라 최적의 컨텍스트 길이를 정할 수 있는 방법을 제시합니다.

- **Technical Details**: 연구에서는 Bayes Risk와 Approximation Loss 개념을 통해 손실을 분해하고, 이러한 손실이 Intrinsic Dimension과 어떻게 연결되는지를 논의합니다. 이론적으로, 최적의 컨텍스트 길이는 훈련 데이터의 크기가 증가함에 따라 증가해야 한다고 주장하며, 이를 실험을 통해 검증합니다. 또한, Bayesian Neural Network의 중간층이나 최종 층에서의 데이터 표현을 활용하여 Intrinsic Space를 근사하는 방법도 논의합니다.

- **Performance Highlights**: 실험 결과, 특정 훈련 데이터 양에서 컨텍스트 길이가 증가함에 따라 손실이 감소하다가 최적의 컨텍스트 길이를 초과하면 손실이 증가하는 경향이 있음을 보여줍니다. 이러한 발견은 긴 컨텍스트가 언어 모델의 성능에 미치는 영향을 직관적으로 이해하는 데 중요한 통찰력을 제공합니다. 연구자들은 이러한 결과가 향후 긴 컨텍스트를 활용한 언어 모델 설계에 기여할 것이라고 기대하고 있습니다.



### Process Reinforcement through Implicit Rewards (https://arxiv.org/abs/2502.01456)
Comments:
          20 pages. Model&Code&Data available at this https URL

- **What's New**: 본 연구에서는 LLM(대형 언어 모델)의 추론 시간에 효과적인 Dense Process Rewards를 제안합니다. PRIME(Implicit Rewards를 통한 Process Reinforcement)은 고품질의 Dense 보상을 대규모로 효율적으로 수집하고 활용하는 방법을 제시합니다. 기존 방식에서 발생할 수 있는 Reward Hacking 문제를 완화하고, 온라인으로 PRM(프로세스 보상 모델)을 업데이트할 수 있는 가능성을 보여줍니다.

- **Technical Details**: PRIME은 Outcome Labels만 사용하여 Dense Reward 모델을 훈련하는 방법론으로, 기존 방식을 통해 요구되는 복잡한 Label 수집 과정을 간소화합니다. 이 프레임워크는 각 Token(Level) 보상과 Sparse Outcome Rewards의 결합도 가능하게 하여 다양한 RL(강화 학습) 알고리즘과 호환됩니다. 특히, PRIME은 SFT(슈퍼바이즈드 파인 튜닝) 모델에서 초기화하여 길어진 훈련 과정을 간소화합니다.

- **Performance Highlights**: 실험 결과에 따르면, PRIME은 기존의 SFT 모델에 비해 평균 15.1%의 성능 향상을 보였으며, Qwen2.5-Math-7B-Base 모델을 사용하여 수학 문제 해결에 있어 2.5배의 샘플 효율을 달성했습니다. 최종 모델인 Eurus-2-7B-PRIME은 Qwen2.5-Math-7B-Instruct를 7개 주요 벤치마크에서 초과 성능을 기록하였습니다. 또한, PRIME은 약 10%의 기존 데이터로도 결과를 달성함으로써 효율적인 학습 가능성을 입증했습니다.



### Originality in scientific titles and abstracts can predict citation coun (https://arxiv.org/abs/2502.01417)
Comments:
          6 pages, 3 figures, submitted to ISSI 2025, research in progress paper

- **What's New**: 이 연구는 창의성(science of creativity) 분야에서 오리지널리티(originality)와 관련된 계산적 측정 방법인 Divergent Semantic Integration (DSI)을 사용하여 Web of Science에서 수집한 99,557개의 과학 초록(abstracts)과 제목(titles)에서 연구를 진행하고 있습니다. 연구 결과, 주제(subject)와 연구 분야(field) 간에 DSI에서 통계적으로 유의미한 차이를 관찰하였습니다.

- **Technical Details**: DSI는 창의성의 원천을 수치화하는 지표로 사용되며, 연구자들은 DSI 값을 분석하여 학문 분야별로 오리지널리티를 측정합니다. 또한, 시간에 따른 DSI의 약간의 상승이 관찰되었고, 이를 바탕으로 연구 후 5년 동안의 인용 수(citation count)를 모델링 하였습니다.

- **Performance Highlights**: 연구 후 5년 동안의 인용 수에 대한 DSI의 상관관계를 분석한 결과, 모든 연구 분야에서 통계적으로 유의미한 긍정적 상관관계가 나타났습니다. 조정된 $R^2$ 값은 0.13으로, 이는 DSI가 인용 수를 예측하는 데 일정한 영향을 미친다는 것을 의미합니다.



### GRADIEND: Monosemantic Feature Learning within Neural Networks Applied to Gender Debiasing of Transformer Models (https://arxiv.org/abs/2502.01406)
- **What's New**: 이 연구에서는 성별 관련 정보를 인코딩하는 단일의 모노세맨틱(feature neuron) 특성 뉴런을 학습하기 위해 모델 기울기를 활용하는 새로운 인코더-디코더 접근법을 제안합니다. 이 방법은 Transformer 기반 언어 모델에서의 성별 편향을 제거할 수 있는 능력을 제공하면서도 모델의 다른 기능을 유지하는 것이 가능함을 보여줍니다. 또한, 기존의 접근법들과 달리 원하는 해석 가능한 의미를 갖는 특성 뉴런을 학습할 수 있도록 합니다.

- **Technical Details**: 제안된 접근법은 인코더-디코더 아키텍처를 기반으로 하며, 성별 정보가 포함된 기울기 정보를 사용하여 특정 특성 뉴런을 학습합니다. 이는 모델의 기울기를 통해 성별과 관련된 스칼라 값을 인코딩하고, 이 스칼라 값을 기반으로 기울기 업데이트를 디코딩하여 성별 편향을 변경할 수 있도록 합니다. 이 방식은 Gradiend와 INLP를 결합하여 성별 편향 제거에 있어 최신 기술을 달성하는 데 기여합니다.

- **Performance Highlights**: 이 연구를 통해 성별 편향 제거를 위해 기울기 기반 특성 학습의 가능성을 입증하였으며, 여러 인코더 전용 모델에서 효과성을 보여주었습니다. 목표한 특성 뉴런을 학습함으로써 성별 편향을 변화시킬 수 있다는 두 가지 가설을 검증하였고, 이를 통해 성별 편향을 성공적으로 수정하며 모델의 다른 성능을 유지할 수 있음을 입증하였습니다.



### AdaSVD: Adaptive Singular Value Decomposition for Large Language Models (https://arxiv.org/abs/2502.01403)
Comments:
          The code and models will be available at this https URL

- **What's New**: 본 논문에서는 LLM(대형 언어 모델)의 메모리 요구사항을 줄이기 위해 AdaSVD라는 적응형 SVD 기반 LLM 압축 방법을 제안합니다. 기존의 SVD 기반 방법들이 자주 발생했던 성능 저하 문제를 해결하기 위해 adaComp와 adaCR을 도입하여 SVD의 절단 오류를 보상하고 각 변환기 레이어에 적합한 압축 비율을 할당합니다. 따라서 AdaSVD는 리소스 제약이 있는 장치에서의 LLM 배포 가능성을 크게 향상시키는 데 기여합니다.

- **Technical Details**: AdaSVD는 SVD 트렁케이션(절단) 오류를 적응적으로 보정하는 adaComp 기법을 통해 U와 V^T 행렬을 번갈아 업데이트합니다. 또한, adaCR을 통해 각 레이어의 중요도에 기반하여 레이어별 압축 비율을 할당하여 모든 레이어에 동일한 비율을 적용하는 기존 방식과의 차별성을 두고 있습니다. 이러한 방식은 압축 비율이 고정된 상태에서도 성능을 개선하는 데 기여함으로써, 운영 메모리와 GPU 메모리에서의 사용 효율을 높입니다.

- **Performance Highlights**: 다양한 LLM 계열과 평가지표를 바탕으로 한 실험 결과, AdaSVD는 기존의 SOTA SVD 기반 LLM 압축 방법인 SVD-LLM을 초월하여 성능을 현저하게 향상시켰습니다. 이로 인해 압축 모델과 원본 모델 간의 성능 격차가 크게 줄어들어, LLM의 효과적이고 효율적인 실행이 가능해졌습니다. 제안된 방법은 다양한 플랫폼에서의 배포와 사용을 동시에 지원하는 가능성을 높입니다.



### Plan-Then-Execute: An Empirical Study of User Trust and Team Performance When Using LLM Agents As A Daily Assistan (https://arxiv.org/abs/2502.01390)
Comments:
          conditionally accepted to CHI 2025

- **What's New**: 이번 연구는 LLM(대형 언어 모델) 에이전트를 일상 업무에 활용하는 새로운 방법을 제시합니다. 사용자가 LLM 에이전트와 협력하여 고위급 계획 세우기 및 실시간 실행 과정에 참여하는 것을 통해 사용자 신뢰를 확립하고, 업무 수행 성과를 향상시킬 수 있음을 확인했습니다. 연구는 248명의 참여자를 대상으로 six 가지 일반적인 업무 시나리오를 통해 진행되었습니다.

- **Technical Details**: 연구는 LLM 에이전트를 계획-실행(plan-then-execute) 방식으로 운영하여, 사용자가 계획 단계에서 LLM 에이전트의 작업을 정의하고, 그 이후 실행 단계에서 과정을 모니터링하도록 하였습니다. LLM 에이전트는 도구(toolkits)를 사용하여 일련의 작업을 수행할 수 있으며, 사용자는 이를 통해 실제 작업의 실행에 참여하고 발생하는 문제를 해결할 수 있습니다. 이러한 접근 방식은 사용자의 인지 부하를 줄이고 업무 성과를 개선하는 데 도움이 됩니다.

- **Performance Highlights**: 연구 결과, 사용자 참여는 불완전한 계획을 보완하고 실행 관리를 통해 LLM 에이전트가 더 나은 업무 수행을 할 수 있도록 도와주는 영향을 미쳤습니다. 그러나, 대안 계획이 신뢰할 수 있다는 사용자 잘못된 믿음을 초래하여, 이러한 신뢰가 LLM 에이전트의 성과에 부정적으로 작용할 수 있음을 발견했습니다. 따라서 사용자의 신뢰를 적절히 조정하기 위한 전략이 필요하다는 점이 강조됩니다.



### Fine-Tuning Discrete Diffusion Models with Policy Gradient Methods (https://arxiv.org/abs/2502.01384)
Comments:
          23 pages, 4 figures, 5 tables

- **What's New**: 이 논문에서는 비연속 확산 모델(discrete diffusion models)의 세밀한 조정을 위한 새로운 정책 기울기 알고리즘을 제안합니다. 이 알고리즘은 Score Entropy Policy Optimization (SEPO)이라는 이름을 가지고 있으며, 비미분 가능 보상(non-differentiable rewards)을 효과적으로 처리할 수 있습니다. 이를 통해 강화 학습에서 사람의 피드백을 활용하는 데 있어 새로운 접근 방식을 제공합니다.

- **Technical Details**: SEPO 알고리즘은 기존의 두 기법, 즉 직접 보상 역전파(Direct Reward Backpropagation)와의 차별화된 점이 있습니다. 이 방법은 보상이 미분 가능할 필요가 없으며, 다양한 비연속 생성 작업에서 적용될 수 있습니다. 기존 방법들이 직면했던 수정성 문제를 해결하며, 비강화 학습 상황에서도 효율적인 최적화를 달성할 수 있습니다.

- **Performance Highlights**: 논문에서 제안하는 SEPO를 기반으로 한 다양한 수치 실험이 진행되었고, DNA 세밀 조정과 자연어 처리 태스크에서 우수한 성능을 보여주었습니다. 이 연구의 결과는 SEPO가 기존의 접근 방식보다 뛰어난 확장성과 효율성을 제공함을 입증하며, 앞으로 비연속 정보 처리의 새로운 가능성을 열어줄 것으로 기대됩니다.



### Meursault as a Data Poin (https://arxiv.org/abs/2502.01364)
Comments:
          7 pages, 9 figures, 4 tables

- **What's New**: 이 논문은 데이터화(datafication) 시대의 데이터 중심 사회에서 인간 경험을 수치화하는 것의 철학적, 윤리적 질문을 다룹니다. Albert Camus의 소설 'The Stranger'의 주인공인 Meursault의 감정적으로 단절된 삶을 통해 이러한 문제를 탐구하고 있습니다. 이 연구는 감정 탐지(emotion detection), 감정 분석(sentiment analysis), 명명된 엔티티 인식(named entity recognition)과 같은 자연어 처리(NLP) 기법을 사용하여 Meursault의 삶에서 주요 사건과 행동을 정량화합니다.

- **Technical Details**: 연구에서 사용된 기술적 방법론에는 BERT를 활용한 감정 탐지, VADER를 이용한 감정 분석, spaCy 기반의 명명된 엔티티 인식이 포함됩니다. 이러한 기법들은 Meursault의 복잡한 감정과 행동을 분석하는 데 적용되었으며, 이는 특히 존재적 소외와 도덕적 모호성이 있는 인간 경험에 적용할 때의 한계를 나타냅니다. AI 도구가 Meursault의 행동과 감정을 잘못 해석하는 방식을 조사함으로써, 인간 내러티브를 데이터 포인트로 축소하는 것의 윤리적 딜레마를 강조합니다.

- **Performance Highlights**: 연구 결과는 데이터 기반 내러티브에 대한 증가하는 의존도를 비판하며, 인공지능에 인간적인 가치를 통합할 필요성을 지적합니다. Meursault의 행동 분석은 기존의 알고리즘 모델이 복잡한 인간 경험을 얼마나 잘 반영하지 못하는지를 보여줍니다. 이러한 발견은 데이터 중심 사회의 근본적인 가정에 도전하며, 윤리적인 측면에서 더 포괄적인 접근을 제안합니다.



### PSSD: Making Large Language Models Self-denial via Human Psyche Structur (https://arxiv.org/abs/2502.01344)
Comments:
          WWW '25

- **What's New**: 이번 논문에서는 LLM(대형언어모델)의 추론 정확성을 개선하기 위한 새로운 방법인 PSSD(자기부정 구조)를 제안합니다. 이 접근법은 인간 사고 구조를 모방하여 세 가지 연결된 역할(직관 기반의 id, 규칙 기반의 superego, 스크립트 중심의 ego)을 통해 LLM의 내부 잠재력을 극대화합니다. PSSD는 LLM이 스스로 실수를 인식하고 이를 수정하는 과정을 보다 유기적으로 구성하여 더 나은 성능을 목표로 합니다.

- **Technical Details**: PSSD는 Freudian(프로이트) 이론을 바탕으로 하여, 각 역할이 상호작용하여 정교한 결과를 도출하도록 설계되었습니다. id 역할은 LLM의 직관적 시도를 통해 다양한 추론 경로를 생성하고, superego 역할은 이러한 시도를 규제하기 위한 규칙을 제공하며, ego 역할은 이를 종합하여 실행 가능한 스크립트를 생성합니다. 이 방식은 기존 LLM의 구조와 통합할 수 있어, 타 기법에 비해 효율성과 유연성을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, PSSD는 기존의 방법들보다 뛰어난 성능을 나타냈으며, 특히 LLM의 자원 소모 문제를 해결하는 데 효과적임을 보여주었습니다. 이 방법은 LLM이 자가 교정 기능을 발전시키고, 동시에 고품질의 결과를 신속하게 도출할 수 있도록 지원합니다. 결국, 이 연구는 LLM의 내부 잠재력을 활용하는 새로운 방향성을 제시하며, 추론 능력 개선에 중요한 기여를 하고 있습니다.



### Probabilistic adaptation of language comprehension for individual speakers: Evidence from neural oscillations (https://arxiv.org/abs/2502.01299)
- **What's New**: 이번 연구는 청취자가 화자의 개인적인 특성에 따라 언어 이해를 확장하는 방식을 탐구했습니다. 특히 고정관념과 일치하지 않는 발화의 가능성에 따라 청취자의 이해력이 어떻게 변화하는지를 조사하였습니다. 두 가지 메커니즘, 즉 화자 일반 메커니즘과 화자 개인 메커니즘의 존재를 발견하였습니다.

- **Technical Details**: 연구에는 두 가지 EEG 실험이 포함되었습니다. 실험 1에서는 화자의 발화가 고정관념과 일치하거나 불일치하는 상황에서 뇌의 고주파(High-beta, 21-30 Hz) 및 세타(Theta, 4-6 Hz) 진동을 분석하였습니다. 불일치 발화는 낮은 기준 비율에서는 진동 파워를 감소시키고 높은 기준 비율에서는 증가시켰습니다.

- **Performance Highlights**: 실험 2에서는 목표 화자와 기준 비율을 분리하여 조작했으며, 고주파 효과가 지속되었음을 관찰했습니다. 화자 불일치의 경우 낮은 기준 비율에서는 파워 감소가 나타났으나 높은 기준 비율에서는 효과가 없었습니다. 이러한 결과는 사회적 인지가 언어 처리에 실시간으로 미치는 영향을 시사합니다.



### Learnable polynomial, trigonometric, and tropical activations (https://arxiv.org/abs/2502.01247)
- **What's New**: 이번 연구는 Orthogonal function bases와 Tropical polynomials를 기반으로 하는 학습 가능한 활성화 함수(activation functions)를 가진 확장 가능한 신경망(neural networks)을 조사합니다. 이는 ImageNet-1K 분류(classification)와 OpenWebText의 다음 토큰 예측(next token prediction)을 목표로 하고 있습니다. 기존 활성화 함수인 ReLU와 달리, 학습 가능한 활성화 함수는 훈련 과정에서 네트워크가 동적으로 적응할 수 있도록 합니다.

- **Technical Details**: 연구에서는 깊은 네트워크(deep networks)에서 발생할 수 있는 소실(vanishing) 및 폭발(exploding) 기울기(gradient) 문제를 해결하기 위해, 변동성(variance) 관리를 개선하는 초기화(initiation) 방안을 제안합니다. 이 방법은 변환기(transformers)와 합성곱 네트워크(convolutional networks)에서 단독으로 단위 변동성을 보장할 수 있으며, 이는 깊은 구조에서도 안정적인 기울기 흐름을 보장합니다.

- **Performance Highlights**: 실험 결과, Hermite, Fourier, Tropical 기반의 학습 가능한 활성화 함수를 사용하는 네트워크가 GPT-2 및 ConvNeXt 네트워크보다 훈련과 테스트 정확도(accuracy) 및 혼란도(perplexity)에서 유의미한 향상을 보이는 것으로 나타났습니다. 이러한 연구 결과는 대규모 작업에서 학습 가능한 활성화 함수의 실현 가능성을 강조합니다.



### Eliciting Language Model Behaviors with Investigator Agents (https://arxiv.org/abs/2502.01236)
Comments:
          20 pages, 7 figures

- **What's New**: 이 논문에서는 언어 모델이 특정 목표 행동을 유도하는 프롬프트를 탐색하는 '행동 유도(behavior elicitation)' 문제를 다룹니다. 연구진은 지도 학습(supervised fine-tuning), 강화 학습(reinforcement learning)을 통한 DPO(Direct Preference Optimization) 및 새로운 Frank-Wolfe 학습 목표를 활용하여 다양한 프롬프트 전략을 발견하는 방법론을 제안합니다. 이 방법은 기존의 프롬프트 설계 방식보다 더 유연하고 해석 가능한 프롬프트를 생성할 수 있습니다.

- **Technical Details**: 연구에서는 고정된 타겟 모델 Llama-3.1 8B를 사용하여 두 가지 유형의 행동 유도 문제, 즉 문자열 유도(string elicitation)와 루브릭 기반 유도(rubric-based elicitation)를 고려합니다. 특히, 단일 프롬프트를 통해 특정 반응을 이끌어내는 단일 턴 elicitation을 중점적으로 다루며, 이때의 두 가지 주요 도전 과제는 언어 입력의 조합적 공간에서 최적화해야 한다는 점과 다양성을 보장하는 것이었습니다. 이를 해결하기 위해 다단계 RL 파이프라인을 도입하고, 다이버시티를 위해 반복적으로 정규화된 DPO 변형을 사용합니다.

- **Performance Highlights**: 이 연구의 결과, 조사자 모델은 100%의 공격 성공률을 달성하였고, AdvBench(유해 행동)에서 Llama-3.1 8B 모델에 대해 98%의 성공률을 기록했습니다. 또한, 인간 해석이 가능한 다양한 행동 유도 전략을 발견하여 단일 턴 조사자들이 복잡한 언어 모델의 다양하고 유익한 행동을 유도할 수 있음을 입증했습니다. 향후에는 다중 턴과 도구 사용이 가능한 조사자 모델이 구현될 것으로 기대하며, 이는 인간 조사자들이 사용하는 다양한 기법을 활용할 수 있을 것입니다.



### Almost Surely Safe Alignment of Large Language Models at Inference-Tim (https://arxiv.org/abs/2502.01208)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 안전한 응답 생성을 보장하는 새로운 추론 시 정렬 방법인 InferenceGuard를 소개합니다. 기존의 RLHF와 같은 정렬 기법이 모델의 가중치를 변경해야 하는 반면, InferenceGuard는 모델 가중치를 수정하지 않고도 안전성 보장을 가능하게 합니다. 이 새로운 접근법은 안전 제약을 추적하는 안전 상태를 활용하여 확률적으로 안전한 응답 생성을 가능하게 합니다.

- **Technical Details**: InferenceGuard는 LLM의 잠재 공간에서 제약 Markov 결정 과정(constrained Markov decision process, cMDP)을 활용하여 추론 시 안전한 응답 생성을 재구성합니다. 이 과정에서 우리는 안전 상태를 증강하여 보너스를 최적화하는 데 있어 Lagrangian 접근법의 한계를 우회하고, LLM의 잠재 공간에서 크리틱 기반 접근법을 통해 MDP를 해결합니다. 이러한 구조적 변화로 인해 모델의 수학적 안전성 보장을 최초로 수립하였습니다.

- **Performance Highlights**: 실험 결과, InferenceGuard는 Alpaca-7B에서 98.02%, Beaver-7B-v3에서 100%의 높은 안전성을 달성했습니다. 이는 안전성과 작업 성능의 균형을 성공적으로 맞추었음을 시사하며, 기존의 추론 시 정렬 기법보다 뛰어난 성능을 보였습니다. 이러한 결과는 모델의 가중치 변형 없이도 도출되었습니다.



### Skewed Memorization in Large Language Models: Quantification and Decomposition (https://arxiv.org/abs/2502.01187)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 기억화(memorization) 문제를 다뤄 개인정보 및 보안 리스크를 조명합니다. 저자들은 기존의 연구들이 평균 사례에 집중하고, 기억화의 skewed (비대칭적) 분포를 간과하고 있음을 지적합니다. 이를 통해 LLM의 감독적 세밀 조정(SFT) 시 기억화 확률을 분석하고, 데이터셋 크기 및 훈련 기간과의 관계를 탐구합니다. 또한, 모델의 데이터 생성 과정에서 기억화를 추정하고 기존 메트릭과 비교하는 통찰력을 제시합니다.

- **Technical Details**: 이 연구는 비모수적 통계 테스트를 활용하여 LLM의 기억화 분석을 수행합니다. 특히, 모델의 기억화가 훈련 기간의 증가에 따라 어떻게 증가하고, 데이터셋 구성이나 크기 변화에 따라 어떻게 달라지는지를 보여줍니다. 저자들은 모델의 생성 과정에서 용어별 확률을 분해하여 데이터의 특성(예: 유사성 격차 및 지역 데이터 밀도)이 기억화 가능성에 미치는 영향을 설명합니다. 이외에도, 이들은 전통적인 텍스트 유사도 측정 방법(Rouge 및 Levenshtein distance)에 비해 그들의 메트릭의 우수성을 강조합니다.

- **Performance Highlights**: 실험 결과, 훈련 에포크가 증가함에 따라 전체 손실이 감소하는 것에도 불구하고 기억화가 증가함을 보였습니다. 부적합(overfitting)이나 높은 지역 엔트로피로 인해 극단적인 기억화 경향이 발생한다는 점을 강조하고, 데이터셋의 구성이나 크기 변화가 기억화 패턴에 미치는 중대한 영향을 보여줍니다. 이러한 분석은 LLM의 기억화 행동을 이해하고, 이를 탐지 및 완화할 전략을 제시함으로써 개인정보 보호를 더욱 강화할 수 있는 방법을 제공합니다.



### DeepRAG: Thinking to Retrieval Step by Step for Large Language Models (https://arxiv.org/abs/2502.01142)
- **What's New**: DeepRAG라는 새로운 프레임워크가 소개되었습니다. 이 프레임워크는 Retrieval-Augmented Generation(RAG)을 Markov Decision Process(MDP)로 모델링하여 전략적이고 적응적인 정보 검색을 가능하게 합니다. DeepRAG는 질의를 반복적으로 분해함으로써 외부 지식을 검색할지 파라메트릭(reasoning) 지식에 의존할지를 동적으로 결정합니다. 실험 결과, DeepRAG는 기존 시스템에 비해 21.99% 더 높은 정확도로 답변을 제공하며 검색 효율성도 향상되었습니다.

- **Technical Details**: DeepRAG는 세 가지 핵심 단계로 구성됩니다: 1) Binary Tree Search를 통해 각 서브쿼리와 관련된 경로를 탐색합니다. 2) Imitation Learning을 활용하여 최소 검색 비용으로 올바른 답변에 도달하는 추론 과정을 학습합니다. 3) Chain of Calibration을 통해 LLM의 내부 지식을 조정하여 원활한 지식 경계 인식을 지원합니다. 각 단계는 MDP의 상태, 행동, 전이 역학, 보상 기능을 정의하여 체계적으로 구성됩니다.

- **Performance Highlights**: DeepRAG는 다섯 개의 오픈 도메인 QA 데이터셋에서 실험되어 그 효과가 검증되었습니다. HotpotQA와 같은 multi-hop factual QA에서 상향된 성능을 보였으며, CAG와 같은 시계열 QA에서도 우수한 결과를 나타냈습니다. 추가 분석을 통해 DeepRAG가 검색 결정과 파라메트릭 지식 간에 더 강한 상관관계를 나타내며, 이는 보다 효과적인 지식 경계 조정으로 이어집니다.



### Picky LLMs and Unreliable RMs: An Empirical Study on Safety Alignment after Instruction Tuning (https://arxiv.org/abs/2502.01116)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 안전 정렬(safety alignment) 저하 문제를 다룹니다. 특히, 일반적인 세팅에서 수행되는 미세 조정(fine-tuning)이 어떻게 모델의 안전성을 저하시킬 수 있는지 체계적으로 분석하였습니다. 일반적인 데이터셋을 사용하더라도 이러한 저하가 발생할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 답변 구조(answer structure), 정체성 보정(identity calibration), 역할 수행(role-play)의 세 가지 주요 요인이 LLM의 안전 정렬에 미치는 영향을 파악했습니다. 또한, 가장 최신의 보상 모델(reward models, RMs)의 신뢰성도 평가하여, 인간의 안전성 선호를 정확히 반영하지 못하는 경우가 많음을 발견했습니다.

- **Performance Highlights**: 이 연구는 LLM의 미세 조정 과정에서 안전 정렬을 유지하는 것이 얼마나 복잡한지를 강조합니다. 개발자들이 유용성과 안전성을 조화롭게 조율할 수 있는 방법에 대한 지침을 제공합니다. 실험에 사용된 데이터셋과 미세 조정 코드는 제공된 URL에서 확인할 수 있습니다.



### GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation (https://arxiv.org/abs/2502.01113)
Comments:
          19 pages, 6 figures

- **What's New**: GFM-RAG는 기존의 RAG 모델들을 개선하기 위해 개발된 새로운 그래프 기반 모델로, 복잡한 쿼리-지식 관계를 효과적으로 포착할 수 있도록 설계되었습니다. 이 모델은 800만 개의 파라미터를 갖춘 혁신적인 그래프 신경망(graph neural network)을 사용하여 복잡한 질문에 대한 응답을 보다 효율적으로 생성합니다. GFM-RAG는 기존의 동적 그래프 모델에서 발생하는 노이즈와 불완전성 문제를 해결하며, 사전 조정 없이도 이전에 보지 못한 데이터셋에서 저명한 성능을 보입니다.

- **Technical Details**: GFM-RAG는 60개의 지식 그래프와 1400만 개의 트리플로 구성된 대규모 데이터세트에서 이중 단계의 훈련 과정을 거쳐 학습됩니다. 실험에서는 HotpotQA, MuSiQue, 2WikiMultiHopQA와 같은 세 개의 다중 홉 QA 데이터셋을 활용하였으며, 700,000개의 문서에서 표본을 추출하여 KG 인덱스를 구축합니다. GFM-RAG는 다양한 도메인에 걸쳐 7개의 특정 RAG 데이터셋을 대상으로 평가하여 일반화 가능성을 확인하였습니다.

- **Performance Highlights**: GFM-RAG는 다중 홉 QA 데이터셋 및 도메인 특정 데이터셋에서 최신 기술 수준의 성능을 달성하였으며, 신경망 확장 법칙(neural scaling laws)과의 일치를 유지합니다. extensive 임상 실험을 통해 경쟁 모델들과의 비교에서 우위를 점하며, 데이터 효율성과 성능의 개선 가능성이 확인되었습니다. 이러한 결과는 GFM-RAG가 앞으로 더 많은 연구와 개발의 잠재력을 지니고 있음을 보여줍니다.



### ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning (https://arxiv.org/abs/2502.01100)
Comments:
          Website: this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 논리적 추론 능력과 복잡한 비단조 추론에서의 확장성을 조사합니다. ZebraLogic이라는 포괄적인 평가 프레임워크를 도입하여 LLM의 추론 성능을 논리 그리드 퍼즐을 통해 평가합니다. 이를 통해 문제의 복잡성을 조절하고 정량화할 수 있게 하여, LLM의 성능 한계를 체계적으로 연구할 수 있는 기회를 제공합니다.

- **Technical Details**: ZebraLogic은 다양한 복잡성을 둔 논리 그리드 퍼즐을 생성할 수 있는 프레임워크로, 이를 통해 LLM이 논리적 제약을 얼마나 잘 준수하는지를 평가합니다. 여기서 제약 만족 문제(CSPs)는 수학적으로 정의되고 확장 가능하여, 모델의 아키텍처나 크기와 관계없이 LLM의 논리 추론 능력을 평가하는 데 유용합니다. 논리 퍼즐은 특정 특성과 값으로 이루어진 집합을 기반으로 하며, 각 퍼즐은 K개의 단서에 대한 고유한 값을 찾기 위한 논리적 추론을 요구합니다.

- **Performance Highlights**: 결과적으로 퍼즐의 복잡성이 증가함에 따라 LLM의 정확도가 극심하게 감소하는 "complexity curse"가 발견되었습니다. 이 현상은 LLM의 크기를 키우거나 추론 시간 계산을 늘리더라도 지속되며, 이는 현재 LLM의 추론 능력이 고유한 제한을 지니고 있음을 나타냅니다. 최적의 추론 토큰 대 Z3 충돌 비율이 존재하나, o1 같은 모델은 복잡성이 매우 높을 경우 이 비율을 항상 달성하지 못한다는 점도 강조됩니다.



### Tool Unlearning for Tool-Augmented LLMs (https://arxiv.org/abs/2502.01083)
Comments:
this https URL

- **What's New**: 이번 논문에서는 툴 증강 대형 언어 모델(LLM)의 툴 언러닝(tool unlearning)에 대한 개념과 필요성을 소개합니다. 툴 언러닝은 보안 취약성이나 개인정보보호 규정으로 인한 필요성을 충족시키기 위해 특정 툴 사용 능력을 제거하는 것을 목표로 합니다. ToolDelete라는 새로운 알고리즘을 통해 툴 언러닝 문제를 해결하고, 툴에 대한 지식 제거와 유지, 일반적 기능 유지라는 세 가지 핵심 특성을 구현합니다.

- **Technical Details**: 툴 언러닝의 주요 목표는 특정 툴의 사용 능력을 제거하면서도 나머지 툴에 대한 지식을 유지해야 하는 것입니다. ToolDelete는 툴 지식 제거, 툴 지식 유지 및 일반적 기능 유지를 통해 이러한 도전을 해결합니다. 또한, LiRA-Tool이라는 새로운 멤버십 추론 공격(MIA) 모델을 개발하여 툴과 관련된 지식이 효과적으로 제거되었는지를 평가합니다.

- **Performance Highlights**: 실험 결과, ToolDelete는 주어진 툴을 잊는데 있어 기존 방법보다 높은 정확도를 달성하며, 사용하지 않거나 최신 도구에 대한 지식을 유지하는 데 있어 뛰어난 성능을 보입니다. ToolDelete는 또한 재교육에 비해 74.8%의 훈련 시간 절약을 가능하게 하며, 낮은 자원 환경에서도 95% 이상의 성능을 유지합니다. 이러한 특성들은 툴 언러닝의 실용성을 더욱 높입니다.



### The Jumping Reasoning Curve? Tracking the Evolution of Reasoning Performance in GPT-[n] and o-[n] Models on Multimodal Puzzles (https://arxiv.org/abs/2502.01081)
- **What's New**: OpenAI의 o1 및 o3 모델 공개는 대규모 언어 모델(Large Language Models, LLMs)의 추론 능력에 큰 변화를 의미합니다. 특히 o3는 인공지능의 일반적 지능을 테스트하는 Abstraction and Reasoning Corpus(ARC-AGI)에서 인간을 초월하는 문제 해결 능력을 보였습니다. 그러나 symbolic pattern에 한정된 기존 벤치마크를 넘어, 다양한 시각 및 언어 데이터를 포함한 멀티모달 시나리오에 대한 탐구가 절실하다는 점이 강조되었습니다.

- **Technical Details**: 이 연구에서는 GPT-[n] 및 o-[n] 시리즈 모델의 멀티모달 퍼즐 해결 능력을 평가합니다. PuzzleVQA와 AlgoPuzzleVQA 데이터셋을 통해 추상 시각 추론과 알고리즘적 문제 해결을 측정하였으며, 각 데이터셋은 모델의 인지 능력을 면밀히 시험하는 구조적 특성을 지니고 있습니다. 특히, 멀티모달 퍼즐은 모델이 시각적 정보와 텍스트 정보를 통합하여 문제를 해결하는 능력을 평가하는 중요한 벤치마크 역할을 합니다.

- **Performance Highlights**: 모델 버전이 진행됨에 따라 추론 능력의 증가 경향이 뚜렷히 나타났으나, o1 모델은 여전히 단순 멀티모달 퍼즐에서 어려움을 겪고 있습니다. 알고리즘 퍼즐에서의 성능 저하도 관찰되며, 이는 현재의 인공지능이 인간의 추론 능력과 아직 큰 격차가 있음을 시사합니다. 연구진은 지속적으로 새로운 모델의 성능을 추적하고 연구 결과를 업데이트할 계획입니다.



### FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation (https://arxiv.org/abs/2502.01068)
- **What's New**: 새로운 연구에서 제안된 FastKV는 긴 문맥(sequence)을 처리하는 대형 언어 모델(LLM)에서 키-값 캐시(KV cache)의 효율성을 높이는 방법이다. FastKV는 후속 레이어에서 일부 정보만 선택적으로 전파하는 Token-Selective Propagation (TSP) 접근법을 적용하여 처리 속도를 향상시키면서도 정확도를 유지한다. 이 방법은 긴 문맥을 다루면서도 정확성을 높일 수 있는 차별화된 전략을 제공한다.

- **Technical Details**: FastKV는 KV 캐시 압축을 위한 두 가지 전략을 이용한다. 초반 레이어에서는 전체 문맥 정보를 유지하고, 이후 레이어에서는 중요한 토큰에만 집중하여 캐시 크기를 최적화한다. 또한, FastKV는 그룹 쿼리 주의(grouped-query attention, GQA)에 대한 인식을 활용하여 메모리와 계산 효율성을 높인다.

- **Performance Highlights**: 실험 결과, FastKV는 HeadKV 대비 시간-첫 단어 토큰(time-to-first-token, TTFT)에서 2.00배, 처리량(throughput)에서 1.40배의 개선을 이루었다. 이러한 성과에도 불구하고 FastKV는 기준 수준의 긴 문맥 벤치마크에서 정확도를 유지하며, 실제 애플리케이션에서 실용적인 해결책을 제공한다.



### Mitigating Hallucinations in Large Vision-Language Models with Internal Fact-based Contrastive Decoding (https://arxiv.org/abs/2502.01056)
- **What's New**: 이번 논문은 Internal Fact-based Contrastive Decoding (IFCD)이라는 새로운 접근 방식을 제안하여 LVLMs의 오브젝트 환각(object hallucinations) 문제를 해결합니다. IFCD는 기계 학습 모델에 통합할 수 있으며, 시각적 입력에 의거한 언어 출력의 정확성을 높이기 위해 LVLMs 자체의 환각을 활용합니다. 기존의 기술들이 높은 비용을 요구하는 반면, IFCD는 상대적으로 간단하면서도 효과를 발휘할 수 있는 방법이라고 강조합니다.

- **Technical Details**: IFCD는 LVLMs 내의 표현을 수정하여 내부적으로 생성된 왜곡된 분포를 활용하는 방법론입니다. 구체적으로, 내부 표현의 편집을 통해 진실성과의 차이를 지닌 두 개의 분포를 구성하며, 이러한 차이를 통해 오브젝트 환각을 감소시키는 대조적 디코딩(contrastive decoding) 방식을 적용합니다. 이 방법은 사전 학습 단계에서 얻은 데이터를 사용하지 않고, LVLM 시스템에 치명적인 성능 저하를 가져오는 환각을 효과적으로 감소시키는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과는 IFCD가 POPE 및 MME 객체 환각 서브셋에서 평균적으로 각각 9% 및 8%의 정확도 향상을 달성하는 것을 보여줍니다. LVLM의 출력 분포를 조정하고 환각 출력을 줄이는 데 있어 IFCD의 효과성을 입증하며, 텍스트 생성 작업에서 생성된 텍스트의 품질을 유지하면서 환각된 객체 비율을 5% 감소시킬 수 있음을 나타냈습니다. 이러한 연구 결과는 LVLMs의 신뢰성과 성능 향상에 기여할 수 있음을 시사합니다.



### SimPER: A Minimalist Approach to Preference Alignment without Hyperparameters (https://arxiv.org/abs/2502.00883)
Comments:
          ICLR 2025

- **What's New**: 본 논문에서는 하이퍼파라미터 없이도 효과적인 선호 최적화 방법인 SimPER를 제안합니다. SimPER는 선택된 응답과 거부된 응답의 역당혹도를 최적화하여, 이전의 복잡한 방법들과 달리 자원 소모 없이도 유효한 성능을 거두는 것을 목표로 합니다. 이를 통해 언어 모델 정렬을 위한 기존 방법의 성능 저하 문제를 해결할 수 있습니다.

- **Technical Details**: SimPER는 하이퍼파라미터 조정과 참조 모델 없이 작동하도록 설계되었습니다. 이 알고리즘이 효과적으로 작동하는 이유는 선택된 응답의 당혹도를 최소화하고, 거부된 응답의 당혹도를 최대화함으로써 인간의 선호에 더 잘 정렬할 수 있도록 하는 역당혹도 최적화 문제를 해결하기 때문입니다. 이는 통계적 거리(Total Variation distance)를 최소화하여 정확한 경량화를 가능하게 합니다.

- **Performance Highlights**: SimPER는 Open LLM Leaderboard, MT-Bench, AlpacaEval 2 등에서 기존의 최첨단 방법들을 초과하는 성능을 입증했습니다. 특히, AlpacaEval 2에서는 5.7 포인트까지 우수한 성과를 기록하며, 10개 기준에서 높은 평균 순위를 달성했습니다. 이러한 결과는 SimPER가 하이퍼파라미터와 참조 모델 없이도 우수한 성능을 유지할 수 있음을 보여줍니다.



### Language Models Use Trigonometry to Do Addition (https://arxiv.org/abs/2502.00873)
- **What's New**: 이 연구는 세 가지 중형 LLM(GPT-J, Pythia-6.9B, Llama3.1-8B)이 더하기 문제를 어떻게 계산하는지를 역설계하여 이해하고자 합니다. 특히, 숫자가 일반화된 나선으로 표현되고 이 나선을 조작하여 덧셈을 수행한다는 점을 새롭게 발견했습니다. 이를 통해 LLM의 수학적 능력에 대한 첫 번째 표현 수준의 설명을 제공합니다.

- **Technical Details**: LLM들은 'Clock' 알고리즘을 활용하여 덧셈을 수행하며, 이로 인해 a와 b를 나타내는 나선을 조작해 a+b를 생성합니다. 또한, MLP, attention head 및 개별 뉴런의 사전 활성화 상태를 나선으로 모델링하여 구체적인 계산 과정을 분석합니다. 연구는 기계적 해석 가능성(mechanistic interpretability, MI)의 맥락에서 진행되며, LLM의 구조적 기능을 규명하는 데 기여합니다.

- **Performance Highlights**: 모델의 성능을 평가한 결과, 특정 LLM이 덧셈 작업을 효과적으로 수행할 수 있음을 발견하였으며, 그 과정에서 'Clock' 알고리즘이 어떻게 활용되는지에 대한 이해를 높였습니다. 본 연구는 LLM이 수치 데이터를 나선으로 나타내고 이를 조작하여 수학적 문제를 해결하는 방식을 분명히 밝히는 중요한 단계를 제시합니다.



### Disentangling Length Bias In Preference Learning Via Response-Conditioned Modeling (https://arxiv.org/abs/2502.00814)
- **What's New**: 이 논문에서는 인간 피드백으로부터의 강화 학습(RLHF)을 개선하기 위한 새로운 프레임워크를 제안하고 있습니다. 제안된 방법은 Response-conditioned Bradley-Terry (Rc-BT) 모델을 사용하여 보상 모델의 길이 편향(length bias) 완화 및 길이 지시사항 준수를 향상시킵니다. 이를 통해 모델의 선호도 모델링과 길이 관련 지시 사항 따름을 동시에 개선할 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: 연구에서 제안된 Rc-BT 모델은 보상 모델과 정책 최적화 단계에 통합되어 길이 편향을 줄이고 길이 지시 사항을 잘 따르도록 설계되었습니다. 모델은 원래 데이터셋을 기반으로 한 향상된 길이 지시 데이터셋으로 학습되어, 아웃라인 போல리티와 같은 단순 기준을 넘어 보다 복잡한 인간의 의미적 의도를 파악합니다. 이 연구는 DPO(Direct Preference Optimization) 방식을 통해 RLHF의 강도를 높이는 데 기여하고 있습니다.

- **Performance Highlights**: 광범위한 실험 결과에서 제안된 방법은 길이 편향을 줄이고 길이 지시 사항 준수를 크게 개선하는 것으로 나타났습니다. 또한, 다양한 기본 모델 및 선호도 데이터셋에서 효과성을 검증받아, 본 연구의 접근 방식이 LLM의 성능 향상에 기여할 수 있음을 보여줍니다. 이는 향후 RLHF 분야에서 길이 편향과 같은 문제를 해결할 수 있는 새로운 방향성을 제시합니다.



### Zero-Shot Warning Generation for Misinformative Multimodal Conten (https://arxiv.org/abs/2502.00752)
- **What's New**: 본 연구는 잘못된 맥락(misinformation)에서 정보를 추출하고 분석하는 모델을 제안합니다. 이 모델은 cross-modality consistency check를 통해 다양한 데이터 모달리티에서 정보를 검증하며, 훈련 시간이 적게 소요되는 특징이 있습니다. 또한, 자동으로 경고를 생성할 수 있는 새로운 zero-shot learning 작업을 도입하여 사용자 이해를 증진시킬 수 있는 방법을 탐색합니다.

- **Technical Details**: 제안된 모델은 (이미지, 캡션) 쌍의 진위를 평가하는 유연한 아키텍처를 갖추고 있으며, 이를 통해 87.04%의 정확도를 기록했습니다. 증거 수집(evidence retrieval), 일관성 검사(consistency check), 경고 생성(warning generation)이라는 세 단계를 포함한 파이프라인을 통해 정확한 검증을 이루어냅니다. 모델은 Visual Language Model(VLM)인 MiniGPT-4를 활용하여 증거의 홈페이지를 분석하고, 관련 정보를 바탕으로 사용자에게 경고 또는 설명을 제공합니다.

- **Performance Highlights**: 본 연구는 경량화된 모델이 전체 모델 활용 시에는 87.04%의 정확도를, 경량 모델 활용 시에는 84.78%의 정확도를 달성해 경쟁력을 검증했습니다. 질적 평가와 인간 평가를 통해 생성된 경고의 잠재력과 한계를 확인했습니다. 이를 통해 잘못된 정보의 추적 및 반박 과정에서 중요한 이해를 향상시킬 수 있음을 입증합니다.



### BEEM: Boosting Performance of Early Exit DNNs using Multi-Exit Classifiers as Experts (https://arxiv.org/abs/2502.00745)
Comments:
          Published at International Conference on Learning Representations (ICLR) 2025

- **What's New**: 본 논문에서는 Deep Neural Networks (DNNs)의 Early Exit (EE) 기법에 대해 새로운 의사결정 기준을 제안합니다. 제안된 기법은 exit classifiers를 전문가(experts)로 간주하고, 이들의 신뢰도(confidence scores)를 집계하여 ensemble 효과를 캡처합니다. 또한, 집계된 신뢰도值가 임계값(threshold)을 초과할 때만 샘플이 종료됩니다. 이로 인해 기존 DNN 추론 방식보다 성능 개선을 목표로 합니다.

- **Technical Details**: BEEM(Boosting Performance of Early Exit DNNs using Multi-Exit Classifiers as Experts) 메커니즘은 각 중간 exit classifier를 전문가로 취급하여 보다 효과적인 판단을 내립니다. 전문가의 정확도에 따라 신뢰도를 가중치로 조정하며, 이전 층의 신뢰도가 일치할 경우에만 집계하여 결정합니다. 이 때 임계값은 각 exit의 오류율(error rates)을 기반으로 설정되며, EE 결정을 내리기 위한 기초로 사용됩니다.

- **Performance Highlights**: 실험을 통해 BEEM은 GLUE 및 COCO 데이터셋에서 기존 EE 방법들보다 1.5배에서 2.1배의 속도 향상을 달성하였습니다. 특히 이미지 캡셔닝과 언어 작업에서 정확도가 비슷하거나 개선된 결과를 보였으며, 특정 NLP 작업에서는 최종 레이어보다 더 높은 정확도를 기록했습니다. 이 연구의 소스코드는 공개적으로 제공되어 연구자들에게 유용하게 활용될 수 있습니다.



### Model Provenance Testing for Large Language Models (https://arxiv.org/abs/2502.00706)
- **What's New**: 이 논문은 대규모 언어 모델의 모델 출처(MODEL PROVENANCE) 테스트를 위한 새로운 프레임워크를 제시합니다. 이 연구는 Fine-tuning을 통해 기초 모델에서 파생된 모델을 식별하는 방법을 개발하고, 이를 통해 저작권 보호와 모델의 편향 인식에 기여하고자 합니다. 저자들은 black-box 접근법을 통해 모델 구성 요소를 분석하고, 모델의 유사성을 비교하는 혁신적인 방법을 도입했습니다.

- **Technical Details**: 모델 출처 검증을 위한 접근 방식으로는 통계적 가설 테스트(statistical hypothesis testing)를 이용합니다. 이 방법은 기초 모델과 파생 모델 사이의 출력 분포(output distribution)를 분석하여 유사성을 정의합니다. 저자들은 Hugging Face에서 수집한 600개 이상의 모델을 대상으로 실험을 수행했으며, 90-95%의 정밀도(precision)와 80-90%의 재현율(recall)을 기록하여 모델 출처 테스트의 유효성을 입증했습니다.

- **Performance Highlights**: 제안된 모델 출처 검증 도구는 API 접근만으로도 높은 정확도를 보장하며, 실무 환경에서 저작권 및 사용 정책 위반을 감지하는 데 효과적인 방법으로 자리 잡을 수 있습니다. 이 연구는 대규모 언어 모델의 맞춤형 응용 프로그램으로부터 발생할 수 있는 문제를 사전에 예방하는 공신력 있는 해결책을 제공합니다. 논문의 결과는 기업이 불법 비즈니스 활동을 방지하는 데 도움을 줄 수 있습니다.



### Learning Autonomous Code Integration for Math Language Models (https://arxiv.org/abs/2502.00691)
- **What's New**: 최근 연구에서 수학 대형 언어 모델(LLMs)을 위한 도구 통합(tool integration)의 한계가 발견되었습니다. 기존의 도구 통합 모델이 외부 지시서에 의존하여 Chain-of-Thought (CoT)나 코드를 사용할지 결정하는 반면, 새로운 연구는 LLM이 독립적으로 방법론을 선택할 수 있는 자율 코드 통합(Autonomous Code integration) 접근 방식을 제안합니다. 이는 LLM이 안정적인 감독 없이 독자적으로 자신의 전략 선택 방법을 개발할 수 있게 합니다.

- **Technical Details**: 제안된 Expectation-Maximization (EM) 프레임워크는 모델의 능력을 탐색함으로써 의사결정 과정을 개선합니다. E-step은 자기 탐색을 통해 참조 전략을 계산하고, M-step은 이 새로운 신념에 기반하여 LLM을 업데이트합니다. 이 과정에서 최신 데이터 합성(data synthesis) 전략과 비정책 강화 학습(off-policy reinforcement learning)을 통합하여 효율성을 높였습니다.

- **Performance Highlights**: 실험 결과, 단지 공개 질의 세트를 사용하여 제안된 방법이 기존 수학 LLM의 성능을 크게 향상시키며, MATH 벤치마크에서 정확도를 약 20% 향상시킨 65.28%에 도달했습니다. 또한, 코드 실행 횟수를 최대 65%까지 줄이는 성과도 나타났습니다. 이는 수학 문제 해결을 위한 코드 통합이 더욱 효과적으로 이루어질 수 있음을 보여줍니다.



### A Survey of Quantized Graph Representation Learning: Connecting Graph Structures with Large Language Models (https://arxiv.org/abs/2502.00681)
- **What's New**: 최근의 그래프 표현 학습이 빠른 발전을 거듭하고 있으며, 이러한 발전의 중심에는 연속 임베딩(continuous embedding) 접근 방식이 자리잡고 있습니다. 하지만 이러한 방법들은 매개변수 효율성, 해석 가능성 및 견고성에서 문제에 직면하고 있습니다. 이에 따라 최근에는 양자화 그래프 표현(Quantized Graph Representation, QGR) 학습이 증가하는 관심을 받고 있으며, 이는 기존의 연속 임베딩 대신 이산 코드(discrete codes)로 그래프 구조를 표현합니다. QGR은 자연어에 유사한 표현 형식을 가지고 있어 대규모 언어 모델(LLMs)과의 통합에 효과적인 잠재력을 가지고 있습니다.

- **Technical Details**: QGR는 연속 임베딩을 사용하는 대신 노드, 서브그래프 또는 전체 그래프 구조에 대해 이산 코드를 학습하는 방법입니다. 그래프 텍스트 시나리오에서 연속 임베딩은 의미 정보의 손실을 초래할 수 있는 반면, QGR는 자연어의 이산 특성과 일관성을 가지므로 LLM에 매끄럽게 통합될 수 있는 이점이 있습니다. 또한 QGR의 핵심 아이디어는 고차원 공간을 여러 저차원 서브공간으로 분할하고 각 서브공간 내에서 독립적으로 양자화하는 것입니다. 이를 통해 QGR는 해석 가능성과 견고성을 제공하며, 자가 감독 그래프 학습을 통해 다양한 그래프 구조를 학습할 수 있습니다.

- **Performance Highlights**: QGR의 도입으로 인해 기억 사용량이 상당히 줄어들며, 이는 대용량 그래프에서의 매개변수 요구 사항을 대폭 절감하는 결과를 가져옵니다. 예를 들어, 특정 조건에서는 매개변수 요구사항이 113배 감소할 수 있습니다. 이처럼 QGR는 그래프 학습의 비효율성을 극복하고, 해석 가능성을 높이며, 실제 애플리케이션에 쉽게 적응할 수 있는 능력을 제공합니다. 궁극적으로 이러한 발전은 그래프 커뮤니티의 발전과 미래 연구를 자극할 것으로 기대됩니다.



### How Contaminated Is Your Benchmark? Quantifying Dataset Leakage in Large Language Models with Kernel Divergenc (https://arxiv.org/abs/2502.00678)
- **What's New**: 이 논문에서는 데이터셋 오염(dataset contamination)의 문제를 해결하기 위한 새로운 방법, 즉 Kernel Divergence Score (KDS)를 제안합니다. 데이터셋 오염은 평가 데이터셋이 모델의 사전 학습(pre-training) 데이터와 겹치는 현상으로, 이것이 성능 지표를 부풀리고 모델 평가의 신뢰성을 저하시키는 문제를 야기합니다. KDS는 모델의 벤치마크 데이터셋에서 파인튜닝(fine-tuning) 전과 후의 커널 유사성 행렬(kernel similarity matrix) 간의 차이를 계산함으로써 오염을 정량화합니다.

- **Technical Details**: KDS는 샘플 임베딩(sample embeddings)의 커널 유사성 행렬의 변화를 분석하여, 사전 학습된 데이터와 파인튜닝된 데이터 간의 관계를 평가합니다. 이 방법은 보지 못한(unseen) 샘플에 대한 파인튜닝의 효과가 더 크다는 통찰을 바탕으로 하여 진행됩니다. KDS는 여러 데이터셋에 대한 실험을 통해 오염 수준과 거의 완벽한 상관관계를 보이며 기존 알고리즘보다 더 나은 성능을 보여줍니다.

- **Performance Highlights**: KDS는 다양한 설정과 데이터셋에 걸쳐 안정적인 점수를 제공하여 연구자들이 데이터셋의 오염 정도에 따라 벤치마크를 신뢰성 있게 구분할 수 있도록 합니다. 이 연구는 KDS가 더 낮은 오염 수준의 데이터셋을 식별하는 데 효과적임을 입증하며, 이로 인해 모델의 일반화 능력을 더욱 정확하게 평가할 수 있는 기회를 제공합니다. KDS는 또한 여러 디자인 요소에 대해 강력한 성능을 보여, 커널 함수, 커널 대역폭(kernel bandwidth), 임베딩 추출 위치 등 다양한 설계 선택에 대한 감도를 확인하는 실험을 수행했습니다.



### Mitigating the Modality Gap: Few-Shot Out-of-Distribution Detection with Multi-modal Prototypes and Image Bias Estimation (https://arxiv.org/abs/2502.00662)
- **What's New**: 본 논문은 출처가 다른(out-of-distribution, OOD) 샘플 탐지를 위한 비전-언어 모델(vision-language model, VLM) 기반 접근 방식을 개선하는 새로운 방법을 제안합니다. 기존의 방법들은 텍스트 프로토타입과 이미지 프로토타입 간의 모달리티 갭(modality gap)으로 인해 높은 오탐지(false positive)율을 보였습니다. 이를 해결하기 위해, 저자들은 ID 이미지 프로토타입을 포함하는 방법을 도입하고, 이로 인해 OOD 탐지 성능이 개선된다고 주장합니다.

- **Technical Details**: 제안된 방법인 SUPREME은 편향된 프롬프트 생성(biased prompts generation, BPG)과 이미지-텍스트 일관성(image-text consistency, ITC) 모듈로 이루어져 있습니다. BPG는 이미지-텍스트 융합을 강화하고 일반화를 개선하기 위해 가우시안 기반의 이미지 도메인 바이어스를 조건으로 설정합니다. ITC는 모달리티 갭을 최소화하기 위해 intra-modal 및 inter-modal 거리 최솟값을 계산합니다. 이러한 방식을 통해 새로운 OOD 점수인 SGMP를 도입합니다.

- **Performance Highlights**: SUPREME은 기존 VLM 기반 OOD 탐지 방법들에 비해 일관되게 성능을 개선하였습니다. 실험 결과, SUPREME의 사용으로 Imagenet-100 및 네 가지 OOD 데이터셋에서 평균적인 FPR95와 AUROC가 각각 32.4와 94.5에서 24.2와 95.8로 개선되었습니다. 이러한 성과는 이론적 분석과 실증적 증거를 통해 뒷받침되며, 제안된 방법이 신뢰성을 높이는 데 기여할 수 있음을 보여줍니다.



### Reformulation is All You Need: Addressing Malicious Text Features in DNNs (https://arxiv.org/abs/2502.00652)
- **What's New**: 본 연구에서는 DNN(Natural Language Processing) 모델에 대한 적대적 공격(adversarial) 및 백도어(backdoor) 공격을 효과적으로 방어할 수 있는 통합 방어 프레임워크를 제안합니다. 기존 방법들은 모델 지향적(model-oriented) 또는 샘플 지향적(sample-oriented)으로 나뉘어 있으며, 각기 다른 단점이 존재하기 때문에 이러한 차이점을 해결할 필요성이 있었습니다. 이에 따라, 문자열 재구성(text reformulation) 모듈을 활용하여 원본 의미를 유지하면서 악의적인 특성을 제거하는 방식으로 진행됩니다.

- **Technical Details**: 이 프레임워크는 DNN 모델이 입력된 문자의 베이스에 대해 고차원 표현(high-dimensional representation)으로 인코딩을 진행하는 과정을 개선합니다. 특정 악의적인 특성에 대한 영향을 최소화하기 위해, 입력 텍스트의 핵심 의미를 사전 인코딩(pre-encoding)하고, 공격에 특화된 특징을 제거하는 방법으로 구조화됩니다. 이 과정에서 최신 LLMs(Large Language Models)를 활용하여 경량의 로컬 대리 모델(local surrogate models)을 훈련시키고, 지식 증류(knowledge distillation) 기술을 적용하여 로컬 환경에서도 안전하게 사용할 수 있도록 합니다.

- **Performance Highlights**: 여러 실험 결과, 제안된 방어 프레임워크는 기존의 샘플 지향적 방어 기법을 능가하며, 다양한 악의적인 텍스트 특징에 대해 강력한 저항력을 보였습니다. 이는 DNN 기반 NLP 모델들이 실질적으로 적용될 수 있는 수준의 신뢰성을 제공함을 보여줍니다. 또한, 효율성 및 개인정보 보호가 중요한 환경에서 실용적인 배치를 가능하게 합니다.



### Converting Transformers into DGNNs Form (https://arxiv.org/abs/2502.00585)
Comments:
          21 pages, 3 figures, and 8 tables

- **What's New**: 최근 딥러닝의 발전으로 Transformer 아키텍처가 주된 모델링 패러다임으로 자리 잡았습니다. 본 논문에서는 self-attention 메커니즘 대신 digraph convolution을 활용한 대체 기술인 Synthetic Unitary Digraph Convolution(Synvolution)을 제안합니다. 이 기술은 Transformer를 Directed Graph Neural Network(DGNN) 형태로 변환하는 Converter 모델로 이어집니다. 실험 결과, Converter는 Long-Range Arena 벤치마크 및 기타 분류 작업에서 뛰어난 성능을 보였습니다.

- **Technical Details**: Transformer의 핵심인 self-attention 메커니즘은 query와 key 행렬 간의 scaled dot-product를 통해 계산된 유사도를 기반으로 value 행렬을 조정합니다. 본 연구에서는 softmax 함수의 대체로 digraph convolution을 적용하여 시간 복잡도를 줄이면서도 고성능을 유지할 수 있는 Synvolution을 제안합니다. 또한, Kernel Polynomial Method를 통해 multi-head operation의 대안으로서 Kernelution을 제안하며, 고성능의 필터링 기능을 제공합니다.

- **Performance Highlights**: Converter 모델은 다양한 벤치마크에서 기존 Transformer 변형들에 비해 우수한 성능을 나타냈습니다. 특히 Long-Range Arena 벤치마크와 문서 분류, DNA 시퀀스 분류에서 효과적인 결과를 보였습니다. 이 연구는 lightweight하면서도 강력한 Transformer 변형으로서의 Converter의 가능성을 입증합니다.



### Defense Against the Dark Prompts: Mitigating Best-of-N Jailbreaking with Prompt Evaluation (https://arxiv.org/abs/2502.00580)
- **What's New**: 최근 연구는 Best-of-N(BoN) 방식의 AI jailbreaking이 랜덤하게 사용된 augmentations(예: 대문자화, 구두점 등)을 반복하여 사용함으로써 모든 주요 대형 언어 모델(LLMs)에서 효과적임을 보여주었습니다. 연구자들은 'Defense Against The Dark Prompts'(DATDP)라는 새로운 방어 방법을 개발하여 이러한 jailbreaking을 100% 차단할 수 있음을 발견하였습니다. DATDP 방법은 LLM을 활용하여 위험하거나 조작적인 행동에 대한 프롬프트를 평가하면서, 제어된 실험을 통해 기존의 공격을 효과적으로 차단합니다.

- **Technical Details**: DATDP는 평가 에이전트를 통한 반복적인 평가 과정을 기반으로 하며, 사용자 제출 프롬프트를 분석하여 유해한 입력을 사전에 차단합니다. 이 방법은 LLaMa-3-8B-instruct 및 Claude와 같은 LLM을 사용하여, 프롬프트가 위험한지 여부를 판단하고 안전한 프롬프트만 응답 모델에 전달합니다. 평가 과정에서는 프롬프트가 공격적인지, 또는 모델의 방어를 시도하고 있는지를 평가하는 단계가 포함됩니다.

- **Performance Highlights**: DATDP는 실험에서 99.5%에서 100%까지의 프롬프트를 차단하는 데 성공하였습니다. 이는 다양한 데이터셋에 적용했음에도 불구하고 일관된 성능을 보이며, 더 작은 LLM을 사용하더라도 유사한 결과를 기록했습니다. 이러한 성공적인 차단 결과는 AI 시스템의 보안을 강화하는 중요한 전략으로 자리 잡을 가능성을 보여주고 있습니다.



### Understanding Multimodal LLMs Under Distribution Shifts: An Information-Theoretic Approach (https://arxiv.org/abs/2502.00577)
- **What's New**: 이 논문에서는 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 배포 환경 변화에 대한 안정성과 신뢰성을 확보하기 위한 이론적 프레임워크의 필요성을 강조합니다. 기존 연구들은 MLLMs의 성능 평가를 위한 다양한 실험을 제공했으나, 이론적인 기반이 부족했던 점을 지적하며, 효과적인 상호 정보량(Effective Mutual Information, EMI)이라는 새로운 측정 지표를 도입했습니다. 이 EMI는 입력 쿼리와 모델 응답 간의 연관성을 정량화하여 MLLMs의 성능을 분석하는 데 중요한 역할을 합니다.

- **Technical Details**: EMI를 도입함으로써, 연구진은 MLLM의 깊이 있는 진단과 최악의 경우 성능 차이를 수치적으로 평가할 수 있는 상한선을 유도했습니다. 그 성능 격차는 배포 환경 변화에 의해 정의된 분포적 불일치와 연결되어 있습니다. 논문 내에서 EMI와 실제 평가 지표인 win rate 간의 이론적 관계도 입증되었으며, 이를 통해 EMI가 성능 변동을 이해하는 데 기여할 수 있음을 보여줍니다.

- **Performance Highlights**: 연구진은 61개의 배포 환경 변화 시나리오에 대해 MLLMs의 성능을 종합적으로 검증했습니다. 실험 결과 EMI와 win rate 간의 강한 상관 관계가 확인되었으며, EMI의 차이 및 상한선과의 관계 또한 검토되었습니다. 이로 인해 EMI 프레임워크가 다양한 배포 환경 변화에서 MLLM의 성능 갭을 포착하는 데 효과적이라는 점이 입증되었습니다.



### Vision-Language Modeling in PET/CT for Visual Grounding of Positive Findings (https://arxiv.org/abs/2502.00528)
- **What's New**: 이 연구는 PET/CT의 시각적 기초(visual grounding)를 위해 새로운 약한 라벨링 파이프라인을 개발했습니다. 기존의 큰 주석된 이미지-텍스트 데이터셋이 부족한 상황에서, 이 파이프라인은 PET/CT 보고서의 긍정적 발견을 특정 이미지 위치와 연결하는 자동화된 방법을 제공합니다. 개발된 ConTEXTual Net 3D 모델은 대규모 언어 모델의 텍스트 임베딩을 3D nnU-Net과 결합하여 훈련되었습니다.

- **Technical Details**: 이 모델은 25,578개의 PET/CT 이미지 및 보고서에서 11,356개의 문장-라벨 쌍을 추출하여 학습되었습니다. 특히, SUVmax와 축 슬라이스 번호를 기반으로 한 약한 라벨링을 통해 PET/CT의 질병 발견을 정확하게 나타내는 이미지 라벨을 생성하였습니다. ConTEXTual Net 3D는 3D nnU-Net 프레임워크를 바탕으로 설계되어 공간적 특성을 효과적으로 추출하면서 텍스트 정보와의 상호작용을 통해 세그멘테이션 성능을 향상시킵니다.

- **Performance Highlights**: ConTEXTual Net 3D는 LLMSeg 및 두 명의 핵의학 의사의 성과와 비교했을 때 F1 점수 0.80으로 우수한 성능을 보였습니다. 모델은 FDG(0.78) 및 DCFPyL(0.75) 검사에서 더 나은 성능을 보였으나, DOTATE(0.58) 및 Fluciclovine(0.66) 검사에서는 성능 저하가 있었습니다. 전반적으로 이 연구는 3D 시각적 기초 모델의 개발을 용이하게 하였지만, 핵의학 의사의 성과에 비해서는 여전히 개선의 여지가 있음을 보여줍니다.



### PolarQuant: Leveraging Polar Transformation for Efficient Key Cache Quantization and Decoding Acceleration (https://arxiv.org/abs/2502.00527)
Comments:
          preprint

- **What's New**: 이번 연구에서 우리는 PolarQuant라는 새로운 양자화 방법을 제안하고 있습니다. 이 방법은 키 벡터의 아웃라이어 문제를 효과적으로 해결하여 KV 캐시의 효율성을 크게 향상시킵니다. 특히, PolarQuant는 각 서브 벡터를 해당하는 반지름과 극각으로 인코딩하여 차원 간의 상관성을 최적화합니다.

- **Technical Details**: PolarQuant는 키 벡터를 두 개의 차원 서브 벡터로 나누어 이들을 비대칭적으로 양자화합니다. 이 과정에서 반지름과 극각을 각각 n비트, m비트 정수로 인코딩하며, 이를 통해 양자화된 서브 벡터는 잘 정의된 원형 패턴을 표현합니다. 이 새로운 관점은 기존의 기존 방식보다 양자화 인자를 적게 요구해 효율성을 높입니다.

- **Performance Highlights**: PolarQuant는 쿼리-키 내적을 테이블 조회로 변환하여 디코딩 프로세스를 가속화시킵니다. 실험 결과, 다양한 오픈소스 LLM에서 쿼리-키 곱셈이 최대 1.27배 빨라지는 것을 확인했으며, 후속 성능 또한 이전의 경쟁 방법들과 유사한 수준을 유지합니다. 따라서 PolarQuant는 비용을 줄이고, 구조를 간소화하며, 디코딩 속도도 증가시키는 세 가지 주요 기여를 합니다.



### Bridging Internal Probability and Self-Consistency for Effective and Efficient LLM Reasoning (https://arxiv.org/abs/2502.00511)
Comments:
          Under review

- **What's New**: 본 논문은 대규모 언어 모델(LLMs)의 추론 성능을 향상시키기 위한 새로운 접근 방식을 제안합니다. 기존의 perplexity(당혹도)와 self-consistency(자기 일관성) 방법의 한계를 이론적으로 분석하고, Reasoning-Pruning Perplexity Consistency(RPC)라는 새로운 방법론을 도입합니다. 이 방법은 정확한 확률 추정을 통해 빠른 추정 오류 감소와 낮은 모델 오류를 동시에 달성할 수 있는 가능성을 보여줍니다.

- **Technical Details**: RPC는 Perplexity Consistency와 Reasoning Pruning을 결합하여 설계되었습니다. Perplexity Consistency는 LLM의 내부 확률을 자기 일관성 프레임워크에 통합하여, 각 추론 경로의 신뢰도를 평가하는 데 사용됩니다. Reasoning Pruning은 저확률의 추론 경로를 제거하여 효과적으로 추정 오류 감소를 방지하는 역할을 합니다.

- **Performance Highlights**: 논문에서는 RPC가 7개의 벤치마크 데이터셋에서 수행된 실험을 통해 기존 방법들보다 유의미한 성능 개선을 제공함을 입증합니다. RPC는 50% 이상 샘플링 예산을 절약하면서도 동일한 추론 성능을 유지하며, 동일한 예산을 사용할 경우 평균 1.29% 높은 성능을 보입니다. 또한, RPC는 자신감 추정치의 정확도가 기존 방법들에 비해 더 뛰어난 것을 확인했습니다.



### Who's the MVP? A Game-Theoretic Evaluation Benchmark for Modular Attribution in LLM Agents (https://arxiv.org/abs/2502.00510)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델) 에이전트를 최적화하기 위한 새로운 평가 프레임워크인 CapaBench를 제안합니다. CapaBench는 협력 게임 이론의 Shapley Value에 기반하여 각 모듈의 기여도를 정량화합니다. 이를 통해 모듈의 성능 기여를 독립적으로 측정하고 분석할 수 있게 됩니다.

- **Technical Details**: CapaBench는 계획(planning), 추론(reasoning), 행동 실행(action execution), 반영(reflection) 등 LLM 아키텍처의 개별 모듈의 기여를 체계적으로 평가합니다. Shapley Value는 모듈의 기여와 상호작용 효과를 동시에 포착할 수 있는 수학적 접근을 제공합니다. 이러한 방법론은 모듈 조합에 따른 성능 예측을 가능하게 하여, 특정 모듈을 타겟으로 한 최적화를 지원합니다.

- **Performance Highlights**: CapaBench는 1,000개 이상의 멀티 라운드 작업으로 구성된 대규모 데이터셋을 구축하여 다양한 실제를 반영하는 평가를 제공합니다. 모듈의 Shapley Value가 높을수록 아젠트의 성능 향상에 긍정적인 영향을 미친다는 것을 실험을 통해 입증했습니다. 이 데이터셋은 공개될 예정이며, LLM 에이전트 성능 평가의 기초로 활용될 수 있을 것입니다.



### MarketSenseAI 2.0: Enhancing Stock Analysis through LLM Agents (https://arxiv.org/abs/2502.00415)
Comments:
          25 pages, 7 figures, Under review at Financial Innovation (FIN)

- **What's New**: MarketSenseAI는 Large Language Models (LLMs)를 활용하여 재무 뉴스를 분석하고, 역사적 주가와 회사 기본정보, 거시경제 환경을 통합하여 종합적인 주식 분석 및 선택을 지원하는 새로운 프레임워크입니다. 이 논문에서는 LLMs의 기술적 발전에 따라 MarketSenseAI의 개선된 기능을 소개하고 SEC 서류 및 수익 호출을 처리하는 Retrieval-Augmented Generation과 LLM 에이전트를 결합한 새로운 아키텍처를 통해 기존 버전 대비 기본 분석 정확도의 현저한 향상을 보여줍니다. 시장에 대한 심층적 통찰력을 제공하며, AI 기반 투자 전략의 견고함을 강조합니다.

- **Technical Details**: MarketSenseAI는 데이터 흐름과 에이전트 책임을 포함하여 LLM 아키텍처에 대한 업데이트를 기초로 하여, Chain-of-Agents (CoA) 접근법을 도입하여 대규모 재무 데이터를 더 세밀하게 처리할 수 있도록 합니다. 또한, Retrieval-Augmented Generation (RAG) 모듈을 통해 다양한 전문가 보고서를 처리하고, 전통적인 분석 방법에서는 놓치기 쉬운 거시경제 문맥을 제공하는 등의 강점을 갖추고 있습니다. 이러한 기술적 개선은 금융 데이터와 비구조적 데이터를 효과적으로 통합하는 데 중점을 두고 있습니다.

- **Performance Highlights**: S&P 100 주식에 대한 2023-2024년의 실증 평가 결과, MarketSenseAI는 누적 수익률 125.9%를 기록했으며, 이는 시장 지수 수익률 73.5%에 비해 현저한 성과를 보였습니다. 2024년 동안 S&P 500을 검증한 결과, MarketSenseAI는 시장보다 33.8% 높은 Sortino 비율을 달성하여 뛰어난 리스크 조정 수익을 보여주었습니다. 이는 소매 및 기관 투자자 모두에게 고급 분석를 제공할 수 있는 가능성을 보여줍니다.



### Doing More with Less -- Implementing Routing Strategies in Large Language Model-Based Systems: An Extended Survey (https://arxiv.org/abs/2502.00409)
- **What's New**: 이번 논문에서는 LLM(대형 언어 모델) 기반 시스템에서 사용자의 쿼리를 가장 적합한 구성 요소로 라우팅하는 메커니즘을 제안합니다. 이러한 접근 방법은 리소스를 최적화하고 비용을 절감하면서 응답 품질을 향상시킬 수 있도록 돕습니다. 특히, Routing을 이용하여 사전 훈련된 LLM의 적절한 선택과 각 쿼리의 특성에 맞는 최적의 작업을 수행할 수 있습니다.

- **Technical Details**: 라우팅 메커니즘은 주어진 쿼리(q)에 대해 모델 집합(ℳ={M1,…,Mn})에서 가장 적합한 LLM을 선택하는 시스템의 구성 요소입니다. 이 메커니즘은 성능 최대화와 예산(B) 제약을 고려하여 최적의 선택을 이끌어낼 수 있습니다. 다양한 LLMs와 임베딩 모델들의 호출 비용(CM)이 경량 모델을 통해 줄일 수 있음을 강조하며, 특정 작업에 가장 적합한 모델을 선택하는 것이 중요하다고 설명합니다.

- **Performance Highlights**: Routing을 통해 LLM 기반 시스템의 평균 대기 시간을 줄일 수 있으며, 이는 사용자가 간단한 질문을 할 때 리소스 요구를 줄이는 데 기여할 것입니다. 또한, 기존의 LLM 기반 시스템이 전반적으로 과도한 하드웨어 자원을 사용할 필요가 없게 만들 수 있습니다. 비용, 성능, 환경 영향 등을 고려하여 최적 효율성의 중요성을 강조하며, 이러한 접근 방식은 향후 연구 방향에도 큰 영향을 미칠 것으로 예상됩니다.



### ALU: Agentic LLM Unlearning (https://arxiv.org/abs/2502.00406)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)에서 정보 삭제(unlearning) 기능을 위한 새로운 접근 방식인 agentic LLM unlearning (ALU) 방법을 제안합니다. ALU는 모델 재교육 없이도 LLM의 유용성을 유지하면서 효과적인 정보 삭제를 가능하게 하는 다중 에이전트 구조를 사용합니다. 기존 방법들이 정보 삭제의 효과와 유유성(utility) 간의 균형을 맞추는 데 어려움을 겪고 있는 반면, ALU는 이러한 문제를 해결합니다.

- **Technical Details**: ALU 프레임워크는 특정 작업을 수행할 여러 LLM 에이전트를 포함하여 정보 삭제를 실행합니다. 각 에이전트는 삭제 과정의 특정 단계에 설계되어 있으며, 모델 가중치를 업데이트할 필요가 없습니다. 이는 어떤 기초 LLM 모델에도 변화 없이 적용 가능하고, 사용자는 시간에 따라 유연하게 삭제 요청을 할 수 있습니다.

- **Performance Highlights**: ALU는 TOFU, WMDP, WPU와 같은 기존 벤치마크 및 jailbreak 기술에 대한 광범위한 실험을 통해 기존 LLM 정보 삭제 방법들 중에서 가장 강력한 성능을 발휘하는 것으로 입증되었습니다. 특히 ALU는 최대 1000개의 삭제 목표에서 평가되고, 모든 위의 정보 삭제 방법들의 평가 범위를 초과하며 우수한 성능을 보여줍니다.



### Enhancing Token Filtering Efficiency in Large Language Model Training with Collider (https://arxiv.org/abs/2502.00340)
- **What's New**: 이번 논문은 토큰 필터링(token filtering) 방식을 통해 대형 언어 모델(LLM)의 훈련 효율성을 극대화하는 시스템인 Collider를 제안합니다. 기존 연구들은 출력 계층에서만 토큰을 필터링하여 효율성을 높이지 못했던 반면, Collider는 모든 계층에서 비의미 토큰을 필터링하여 스파시티(sparsity)를 유지합니다. 뿐만 아니라 Collider는 희소(gem) 일반 행렬 곱셈(GEMM)을 차원 축소된 밀집(dense) GEMM로 변환하는 자동화된 워크플로우를 특징으로 합니다.

- **Technical Details**: Collider는 후방 계산 그래프를 분석하여, 역전파(backpropagation) 중 비의미 토큰의 활성화를 추가로 필터링하여 충분한 스파시티를 확보합니다. 이 시스템은 기존의 토큰 필터링 방법의 효용 개선을 유지함과 동시에 성능 향상의 기회를 더욱 확대합니다. PyTorch의 동적 그래프 특성으로 인해 전역 업데이트가 복잡하지만, Collider는 런타임 안정성을 활용하여 필요한 차원과 변수를 역전파 전에 동적으로 식별하고 업데이트하도록 설계되었습니다.

- **Performance Highlights**: Collider는 세 가지 LLM 모델(TinyLlama-1.1B, Qwen2.5-1.5B, Phi1.5-1.4B)에서 평가를 통해 역전파 시간을 최대 35.1% 감소시키고, 전체 훈련 시간을 22.0% 낮추는 결과를 보여주었습니다. 15억 토큰으로 TinyLlama를 훈련할 경우, Collider는 평균적으로 모델 유용성을 16.3% 향상시키며, 훈련 시간을 4.5일에서 3.5일로 단축시킵니다. 이 시스템은 기존 LLM 훈련 프레임워크에 간단히 통합될 수 있어, 단 한 줄의 코드 추가로 토큰 필터링을 사용하는 시스템의 훈련 속도를 가속화할 수 있습니다.



### Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values (https://arxiv.org/abs/2502.00313)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)이 자원 분배에서 공정성을 지향하는지에 대한 실증적 분석을 시도하고 있습니다. 연구는 LLM의 응답이 인간의 분배 선호와 얼마나 일치하는지를 평가하며, 공정성 개념을 충족하는 능력을 비교합니다. 특히, LLM들이 종종 경제적 효율성을 우선시하는 경향이 있음을 보여줍니다.

- **Technical Details**: 연구에서는 LLM의 응답이 공정성의 다양한 개념, 즉 Equitability (EQ), Envy-Freeness (EF), Rawlsian Maximin (RMM)과 얼마나 잘 일치하는지를 평가하기 위해 여러 가지 실험을 수행했습니다. 각 LLM의 성능은 GPT-4o, Claude-3.5S, Llama3-70b, Gemini-1.5P처럼 최신 모델들로 비교되었습니다. 연구는 비대칭 자원 할당 과제에서도 LLM의 응답이 인간의 선택과는 다르게 나타났음을 밝혔습니다.

- **Performance Highlights**: LLMs는 공정성을 기반으로 한 자원 분배 결정에서 인간의 선택과는 상당한 불일치를 보였습니다. GPT-4o는 다른 LLM들과 비교할 때 공정성을 달성하기 위해 금전을 더 효과적으로 활용하는 모습을 보였으며, 주어진 선택지에서 공정한 해결책을 선택할 때도 인간의 가치를 더 잘 반영했습니다. 그러나 LLM은 특정 공정성 개념을 일관되게 충족하지 못하고, 자원 할당 문제에서 인간과의 가치 일치도가 낮은 것으로 나타났습니다.



### SigWavNet: Learning Multiresolution Signal Wavelet Network for Speech Emotion Recognition (https://arxiv.org/abs/2502.00310)
Comments:
          Published in: IEEE Transactions on Affective Computing

- **What's New**: 본 논문은 음성 감정 인식(SER) 분야에서 raw waveform 음성 신호로부터 의미 있는 표현을 추출하는 새로운 end-to-end (E2E) 딥러닝 다중 해상도 프레임워크를 소개합니다. 이 접근 방식은 고속 이산 웨이브렛 변환(FDWT)의 속성을 활용하여 노이즈 간섭 및 시스템 복잡성과 같은 기존의 한계를 극복합니다. 이를 통해 wavelet 기반과 노이즈 제거를 위한 학습 가능한 모델을 도입하여 SER의 정확도를 높입니다.

- **Technical Details**: 제안된 프레임워크는 웨이브렛 계수의 비대칭 하드 스레시홀드를 위한 활성화 함수와 함께 한 차원 팽창 합성곱 신경망(1D dilated CNN), 공간적 주의층, 양방향 게이트 순환 유닛(Bi-GRU) 및 시간적 주의층을 결합하여 감정 특징의 미세한 공간 및 시간적 특성을 효율적으로 포착합니다. 이 모델은 가변 길이 음성을 세분화 없이 처리할 수 있으며, 전처리 및 후처리의 필요성을 없앱니다.

- **Performance Highlights**: 이 모델은 IEMOCAP 및 EMO-DB 데이터셋에서 기존의 최신 기법보다 더 우수한 성능을 보였습니다. 연구 결과, 신경망 아키텍처가 음성 신호의 복잡한 감정을 효과적으로 인식하고 분석할 수 있도록 설계되었습니다. 소스 코드는 GitHub 리포지토리에 공유되어 다른 연구자들이 쉽게 접근하고 활용할 수 있습니다.



### Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation (https://arxiv.org/abs/2502.00306)
- **What's New**: 본 논문은 Retrieval-Augmented Generation (RAG) 시스템에서의 membership inference 공격 기법인 Interrogation Attack (IA)을 제시합니다. IA는 모델의 성능을 저하시키지 않으면서도, 최소 30개의 자연어 쿼리로 특정 문서의 존재 여부를 추론하는 데 성공합니다. 기존의 기법들과는 달리, IA는 스텔스성(stealthy)를 유지하면서도 높은 정확도와 재현성을 보여줍니다.

- **Technical Details**: RAG 시스템의 구조는 문서 집합과 검색기, 생성 모델로 구성됩니다. RAG는 쿼리에 따라 지식 기반에서 관련 문서를 검색하고 이를 모델의 프롬프트에 포함하여 출력을 생성합니다. IA는 타겟 문서와 밀접하게 연관된 자연어 쿼리를 설계하여, 문서의 존재 여부를 추론하는 방식으로 작동합니다.

- **Performance Highlights**: IA는 기존의 membership inference 공격들과 비교해 2배 향상된 True Positive Rate (TPR@1% FPR)을 기록했으며, 공격자의 쿼리는 덜 포착되고 대략 5%의 탐지율을 자랑합니다. 이는 기존 방법들이 90% 이상 탐지되는 것과 비교됩니다. 또한 이 공격은 RAG 시스템 사용 시 발생할 수 있는 개인정보 유출 문제를 해결하는 데 기여할 것으로 기대됩니다.



### ProxSparse: Regularized Learning of Semi-Structured Sparsity Masks for Pretrained LLMs (https://arxiv.org/abs/2502.00258)
- **What's New**: 이번 연구에서는 ProxSparse라는 새로운 프레임워크를 제안하여 Semi-Structured Pruning의 Mask Selection 문제를 해결합니다. 기존의 국소적인 히uristic 기반 접근법과 달리, ProxSparse는 전반적인 최적화를 고려한 학습 기반 방법입니다. 이 방법은 수백 개의 보정 데이터셋을 통해 마스크를 데이터 기반으로 학습하고, 추가적인 웨이트 업데이트 없이 마스크가 결정된 후 적용됩니다.

- **Technical Details**: ProxSparse의 핵심은 선택된 마스크에 대한 규제자(regularizer)로, 비선형적이고 비구별적인 마스크 선택 프로세스를 부드러운 최적화 프로세스(differentiable optimization process)로 전환합니다. 이를 통해 불필요한 웨이트를 점진적으로 축소하고, 자원을 효율적으로 사용한 유효한 Semi-Structured Mask를 찾을 수 있게 됩니다. 연구진은 ProxSparse를 활용하여 LLM 규모에서의 효율적 최적화 해결책을 개발했으며, 이 방법은 기존의 Gradient Descent 기반 방법보다 10배 빠른 속도를 자랑합니다.

- **Performance Highlights**: ProxSparse는 7개의 모델에 대한 평가를 통해 지속적으로 이전의 Semi-Structured Pruning 방법보다 성능이 개선된 결과를 보여주었습니다. 제안된 방법은 Perplexity(PPL)와 Accuracy에서 이전의 SOTA에 비해 최대 35%까지 성능 향상을 구현했습니다. 이로 인해 ProxSparse는 효과적인 Semi-Structured Pruning에 대한 새로운 기준을 제시한다고 할 수 있습니다.



### Mordal: Automated Pretrained Model Selection for Vision Language Models (https://arxiv.org/abs/2502.00241)
- **What's New**: 이번 논문에서는 자동화된 멀티모달 모델 검색 프레임워크인 Mordal을 도입하여 특정 작업을 위한 가장 적합한 Vision Language Model (VLM)을 효율적으로 찾는 방법을 제안합니다. 기존의 VLM들은 전문가에 의해 수작업으로 제작되어, 사용자가 원하는 작업에 맞는 자동 프레임워크가 없었습니다. Mordal은 후보 모델의 수를 줄이고 각 후보를 평가하는 시간을 최소화하여 효율적인 검색을 가능하게 합니다. 연구 결과, Mordal은 기존의 grid search에 비해 GPU 시간을 최대 8.9배에서 11.6배 절약할 수 있음을 보여주는 동시 신 모델을 발견했습니다.

- **Technical Details**: Mordal은 VLM을 구축하고 훈련하기 위한 기존 접근 방식을 크게 개선하는 방법론을 제공합니다. VLM은 일반적으로 Vision Encoder, Feature Projector 및 Language Model로 구성되며, 각각의 구성 요소는 특정한 역할을 수행하여 입력 이미지와 텍스트를 결합하고 해석합니다. 특히 다양한 pretrained 비전 인코더와 언어 모델을 조합해 최적의 성능을 낼 수 있는 조합을 탐색하며, 이를 위해 초기 후보 모델들을 유사도 기준으로 클러스터링하고 평가 시간을 단축하는 조기 중지 메커니즘을 도입했습니다.

- **Performance Highlights**: Mordal을 사용한 성능 평가에서, 전체적으로 49개의 VLM 후보를 대상으로 한 그리드 서치보다 높은 효율성과 적은 계산 시간으로 최적의 VLM을 찾는 것을 확인했습니다. 본 연구에서는 특히 비전-텍스트 정렬 데이터에 훈련된 Feature Projector의 중요성을 강조하며, 최적의 pretrained Vision Encoder 및 Language Model 조합을 찾기 위해 진행된 다양한 실험 결과를 공유합니다. 실험 결과들은 VLM 성능 향상에 기여하며, 여기서 발견된 새로운 VLM들은 기존의 최첨단 모델들을 초과하는 성능을 나타냈습니다.



### Should You Use Your Large Language Model to Explore or Exploit? (https://arxiv.org/abs/2502.00225)
- **What's New**: 최근 세대의 대형 언어 모델(LLMs)이 탐색-착취(tradeoff) 문제에 어느 정도 도움을 주는지 평가하는 연구가 진행되었습니다. 이 연구는 LLM들이 다양한 컨텍스트 기반의 밴딧(bandit) 작업에서 탐색 및 착취를 수행하는 능력을 분석합니다. 결과적으로, LLM들은 작은 작업에서는 성과를 일정 부분 개선할 수 있으나, 여전히 간단한 선형 회귀보다 성능이 떨어짐을 발견했습니다.

- **Technical Details**: 이 연구에서 Gpt-4, Gpt-4o 및 Gpt-3.5 모델이 탐색 및 착취 작업을 수행하는 데 어떻게 도움을 줄 수 있는지 탐구하였습니다. 특히, 방법론적 접근으로는 grounded factor를 활용한 in-context learning을 통해 LLM이 최적의 행동을 선택하는 능력을 평가했습니다. 수치 통계(summary statistics)와 같은 방법으로 세밀한 개선이 가능하였지만, 이는 복잡한 의사결정 작업에는 제한적입니다.

- **Performance Highlights**: LLMs는 작은 규모의 문제에서 착취 효율성을 어느 정도 보여주었으나, 문제의 크기가 중간 이상으로 증가하면 성능이 저하되는 경향을 보였습니다. 반면, 높은 차원에서의 행동 공간 탐색에 있어 LLMs는 적합한 후보군을 제안함으로써 효과적인 탐색을 지원했습니다. 대규모 밴딧 작업에서도 유사한 결과를 얻어, 성과는 만족스럽지만 여전히 전통적인 방법에 비해 부족하다는 결론을 내렸습니다.



### Reward-aware Preference Optimization: A Unified Mathematical Framework for Model Alignmen (https://arxiv.org/abs/2502.00203)
Comments:
          8 pages, 4 figures

- **What's New**: 이번 논문에서는 Reward-Aware Preference Optimization (RPO)라는 새로운 수학적 프레임워크를 제안하여, DPO, IPO, SimPO, REINFORCE (LOO) 등의 인기 있는 선호 최적화 기법들을 통합합니다. RPO는 다양한 디자인 선택의 영향을 체계적으로 연구할 수 있게 해주며, 특히 최적화 목표, 프롬프트당 응답 수, 그리고 보상 모델의 사용 방식의 차이에 대한 분석을 포함합니다. 이를 통해 LLM의 정렬 성능을 개선하기 위한 효율적인 전략을 제시합니다.

- **Technical Details**: RPO 프레임워크는 프롬프트(x)와 응답(y), 보상 모델(r) 간의 관계를 정의하고, 시뮬레이션 데이터 집합(𝒟)을 통해 모델의 성능을 평가합니다. 선호 최적화 기술을 적용하여, RLHF 알고리즘처럼 보상 모델을 학습하고 이를 통해 정책 모델(π)을 최적화할 수 있습니다. 이 과정에서 응답 수를 조정하거나 온라인과 오프라인 응답 수집 방법을 비교함으로써 레포 트리 저널의 선호를 최적화하는 데 기여합니다.

- **Performance Highlights**: RPO 프레임워크를 통해 실시한 일련의 배제 연구(ablation studies)는 선호 최적화 알고리즘의 결정적인 요인들에 대한 통찰력을 제공합니다. 연구 결과는 LLM 정렬을 위한 효과적인 요소들을 조합하여 새로운 알고리즘을 생성할 수 있는 기반을 마련합니다. 마지막으로, 이 연구는 모델 정렬을 위한 조리법(cookbook)을 제시하여, 향후 연구자들에게 실질적인 가이드를 제공합니다.



### Fairshare Data Pricing for Large Language Models (https://arxiv.org/abs/2502.00198)
- **What's New**: 본 논문에서는 LLMs(대형 언어 모델)의 학습 데이터 가격을 공정하게 설정하기 위한 fairshare pricing framework를 제안합니다. 이 프레임워크는 데이터의 가치를 정량화하는 데이터 평가(data valuation) 방법을 활용하여, 구매자와 판매자 모두에게 최적의 가격 책정을 가능하게 합니다. 이론적으로 볼 때, 제안된 가격 책정 방식은 데이터 가치와 구매자의 예산에 긴밀하게 연결되어 있음을 보여줍니다.

- **Technical Details**: 공정한 가격 책정을 위해, 연구진은 데이터의 기여도를 정량화한 후 이를 바탕으로 판매자가 수익을 극대화할 수 있는 가격을 책정하도록 합니다. 제안하는 프레임워크는 구매자가 예산 제약 하에서 현명한 구매 결정을 내릴 수 있도록 하며, 판매자는 최적 가격으로 데이터를 판매하는 파트너십을 구축할 수 있게 합니다. 시뮬레이션을 통해, 제안된 프레임워크가 대형 LLM의 품질 향상에 기여하며, 데이터 가격이 가치를 반영하게 됨을 보여줍니다.

- **Performance Highlights**: 연구에서 제안하는 프레임워크는 데이터 가격이 가치를 반영하도록 하여, 구매자들에게 데이터를 구매할 때의 만족도를 높이고 LLM 작업 성과도 향상시킵니다. 또한, 판매자들에게는 공정한 가격을 보장하여 데이터 판매에 대한 참여를 촉진합니다. 최종적으로 이 프레임워크는 LLM 구축에 필요한 고품질 학습 데이터 공급의 지속 가능성을 높이는 방향으로 이끌어 줍니다.



### DermaSynth: Rich Synthetic Image-Text Pairs Using Open Access Dermatology Datasets (https://arxiv.org/abs/2502.00196)
Comments:
          12 pages, 4 figures

- **What's New**: 이번 연구에서는 각종 피부과 임상 과제를 위해 92,020개의 합성 이미지-텍스트 쌍으로 구성된 새로운 데이터셋인 DermaSynth를 소개합니다. 이 데이터셋은 피부과 관련 이미지와 함께 제공되는 텍스트 어노테이션의 부족 문제를 해결하기 위해 개발되었습니다. 임상 관련 프롬프트와 자기 지도(self-instruct) 방법을 활용해 데이터셋을 생성하였으며, 이는 AI 연구에 기여할 것으로 기대됩니다.

- **Technical Details**: DermaSynth 데이터셋은 DERM12345, BCN20000, PAD-UFES-20 등 여러 공개 피부과 데이터셋에서 수집된 이미지들을 바탕으로 합니다. 각 이미지에 대해 일반 질문과 메타데이터 기반 질문을 통해 합성 이미지-텍스트 쌍을 생성하였고, 이 과정에서는 Gemini 2.0 API를 이용했습니다. 이렇게 생성된 데이터는 후처리 과정을 거쳐 임상적으로 관련성이 높고 일관된 스타일을 유지하도록 하였습니다.

- **Performance Highlights**: 프로토타입 모델 DermatoLlama 1.0은 Llama-3.2-11B-Vision-Instruct 모델로, DERM12345 데이터셋에서 5,000개의 이미지-텍스트 쌍을 Fine-tuning하여 개발되었습니다. 이 모델은 Hugging Face에서 접근 가능하며, 연구자들이 피부과 영상을 기반으로 한 LLM의 성능을 평가하는 데 활용될 수 있습니다. DermaSynth와 DermatoLlama 1.0은 모두 비상업적 연구 목적으로 제공되며, 해당 리소스들은 오픈 소스로 제공됩니다.



### AIN: The Arabic INclusive Large Multimodal Mod (https://arxiv.org/abs/2502.00094)
Comments:
          20 pages, 16 figures, ACL

- **What's New**: 최근의 큰 언어 모델(LLMs)과 다중 모달 모델(LMMs)의 발전 속에서, 아랍어 LMM들은 주목받지 못한 반면, 아랍어 LLMs는 상당한 향상을 보여 왔습니다. 이 격차를 해소하기 위해 AIN(Arabic Inclusive Multimodal Model)을 소개합니다. AIN은 영어-아랍어 이중 언어 LMM으로, 고품질의 360만 개 아랍어-영어 다중 모달 데이터 샘플을 활용하여 설계되었습니다.

- **Technical Details**: AIN 모델은 70억 개의 파라미터를 기반으로 한 아랍어 대형 다중 모달 모델로, 복잡한 추론, 다국적 작업, 이미지-텍스트 정렬에서 우수한 성능을 보입니다. CAMEL-Bench 기준에서 대조군 모델들과 비교하여 AIN-7B는 많은 도메인에서 높은 성과를 자랑하며, Qwen2-VL-7B와 비교해도 성능이 3.43% 향상되었습니다.

- **Performance Highlights**: AIN의 성과는 특히 의료 이미지 해석 및 과학 시각화 이해 등 다양한 분야에서 두드러집니다. 설문 조사 결과, 아랍어 구사자들 사이에서 AIN-7B가 76%의 지지를 받아 대세 모델이 되었으며, 복잡한 시각-언어 과제를 처리하는 데 있어 AIN의 효율성과 정확도가 두드러집니다.



### LLM Cyber Evaluations Don't Capture Real-World Risk (https://arxiv.org/abs/2502.00072)
Comments:
          11 pages

- **What's New**: 대형 언어 모델(LLMs)은 사이버 보안 애플리케이션에서 점점 더 많은 잠재력을 보여주고 있으며, 이는 방어를 강화할 수 있는 가능성과 함께 고유한 위험을 초래합니다. 이 논문에서는 LLM의 위험 평가 노력이 실제 세계의 영향을 이해하는 데 잘못 맞춰져 있다고 주장합니다. LLM의 사이버 능력에 대한 위험 평가 접근 방식을 제안하며, 이를 사이버 보조 도구로 사용된 언어 모델을 사례 연구로 적용합니다.

- **Technical Details**: LLMs의 사이버 능력 위험 평가에는 복잡한 분석이 요구됩니다. 이 논문은 기존의 위험 평가 방안이 LLM의 기술적 능력 분석에만 국한되어 있다고 비판하며, 위협 행위자의 행동 및 잠재적 영향을 포함한 종합적인 위험 평가 프레임워크를 제안합니다. 이를 통해 실제 공격 시나리오에서 LLM의 활용 가능성과 제약을 분석하고, 이로 인해 발생하는 잠재적 피해를 연구합니다.

- **Performance Highlights**: 분석 결과, LLM 모델들은 높은 준수 비율을 나타내지만, 실제 사이버 보조 작업에서는 중간 정도의 정확도를 보이고 있습니다. 연구 결과에 따르면, 특정 사용 사례에서 운영상의 이점과 영향 가능성이 제한되어 있는 만큼, LLM의 사이버 보안 능력으로 인한 위험은 중간 수준에 불과합니다. 마지막으로, 연구 우선순위를 실제 영향 평가와 일치시키기 위한 몇 가지 개선 사항을 제안하고 있으며, 이는 보다 효과적인 LLM 기반 사이버 보안 위험 평가와 완화로 나아가는 중요한 단계로 풀이됩니다.



### Contextually Entangled Gradient Mapping for Optimized LLM Comprehension (https://arxiv.org/abs/2502.00048)
- **What's New**: 이 연구에서는 Contextually Entangled Gradient Mapping (CEGM)이라는 새로운 접근법을 소개하고 있습니다. CEGM은 기존의 경량 모델에서의 기울기 최적화 관계를 재정의하여, 주제와 의미의 일관성을 높이고 추론 능력을 향상시키는 방법론입니다. 기울기를 독립적인 수치적 개체가 아닌 동적으로 연결된 맥락적 의존성의 운반자로 다루는 접근법으로, 현재 최적화 전략의 중요한 격차를 해소하기 위해 설계되었습니다. CEGM의 통합은 고차원 추론 및 맥락적 유지, 다양한 환경에의 적응성을 포함한 여러 작업에서 유의미한 향상을 나타냈습니다.

- **Technical Details**: CEGM은 모델의 최적화 과정에서 기울기와 맥락적 표현 간의 새로운 상호작용을 통해 작동합니다. 이 방법론은 다차원적인 맥락 의존성을 포착하도록 설계된 구조적 상호작용을 활용하여 여러 레이어의 상관관계를 다루고, 신경망 아키텍처의 효율성을 증가시킵니다. 이러한 동적 맥락 기반 조정은 기울기가 가시적이면서도 복잡한 관계를 반영함으로써 기존의 정적 모델 아키텍처에서의 한계를 극복하고자 합니다.

- **Performance Highlights**: 실험 데이터는 CEGM을 활용한 모델이 기초 모델과 비교했을 때 지속적으로 높은 정확도와 노이즈에 대한 저항력을 나타냈음을 보여줍니다. 문장 변환 과정에서 의미의 편향 감소와 의미적 일관성을 향상시켰으며, 이는 제안된 방법론의 강력함과 다용성을 강조합니다. CEGM을 적용한 윤곽 조정 및 훈련 파이프라인 변경을 통해 중장기 추론 요구 사항을 성공적으로 충족함으로써, 새로운 연구와 개발을 위한 길을 열어주고 있습니다.



### Optimization Strategies for Enhancing Resource Efficiency in Transformers & Large Language Models (https://arxiv.org/abs/2502.00046)
Comments:
          Accepted for ACM's ICPE 2025 in Short Paper format

- **What's New**: 이 연구는 자연어 처리 분야의 Transformer 아키텍처의 최적화 기법에 대해 다루고 있으며, 에너지 및 계산 효율성을 향상시키면서 성능을 유지하는 방법을 탐구합니다. 4비트 Quantization은 낮은 정확도 손실로 에너지 사용을 크게 줄일 수 있는 독립적인 방법으로 밝혀졌습니다. NVIDIA의 Minitron 접근법과 같은 하이브리드 방법도 주목받고 있으며, 이는 크기 축소와 정확도 유지 사이의 유망한 균형을 보여주고 있습니다.

- **Technical Details**: 연구에서 제안된 최적화 방정식은 다양한 방법을 비교하는 유연한 프레임워크를 제공합니다. 이 연구는 Quantization, Knowledge Distillation, Pruning 기술을 활용하여 GPT-2 및 OPT Transformer 모델의 perplexity(혼란도), 에너지 사용량, 계산 속도 변화를 평가합니다. 여러 최적화 방법을 조합하여 모델의 성능 손실을 최소화하면서 Resource 효율성을 극대화하려고 합니다.

- **Performance Highlights**: 본 연구는 다양한 모델 압축 방법의 효과를 평가하여, 시간 소비, 에너지 소비 및 모델 정확도 간의 중요한 거래를 해결하는 가능성을 검토합니다. 연구 결과는 모델 압축 전략의 최적 설정을 제공하여, AI 시스템의 에너지 요구 사항을 경감하는 방향으로 중요한 논의를 촉진할 것으로 기대됩니다. 궁극적으로, 이 연구는 보다 지속 가능한 대형 언어 모델(LLMs)의 개발 및 배포에 기여할 수 있는 귀중한 통찰력을 제공하고자 합니다.



### AlphaSharpe: LLM-Driven Discovery of Robust Risk-Adjusted Metrics (https://arxiv.org/abs/2502.00029)
- **What's New**: 이 논문에서는 AlphaSharpe라는 새로운 프레임워크를 소개하며, 이는 대형 언어 모델(LLM)을 활용하여 재무 성과 메트릭을 반복적으로 진화 및 최적화합니다. 특히, 기존의 재무 메트릭들이 가진 저항력 및 일반화의 한계를 극복하고 있습니다. LLM을 적용하여 생성된 새로운 리스크-수익 메트릭은 기존 메트릭보다 더 우수한 예측 능력을 보여줍니다.

- **Technical Details**: AlphaSharpe는 크로스오버(crossover), 변이(mutational), 평가(evaluation) 기법을 통해 재무 메트릭을 발전시키는 방법론을 구현합니다. LLM의 창의적인 생성 능력을 통해 다양하고 독창적인 메트릭 변종을 만들어내며, 각 메트릭은 데이터의 노이즈와 아웃라이어에 대한 저항성을 높이도록 설계되어 있습니다. 이러한 반복적인 최적화 과정은 메트릭이 미래 성과와 잘 일치하도록 보장합니다.

- **Performance Highlights**: 실험 결과 AlphaSharpe 메트릭이 전통적인 메트릭에 비해 평균 3배의 예측력과 2배의 포트폴리오 성과를 기록함을 보여줍니다. 발견된 메트릭들은 포트폴리오 관리자 및 금융 의사결정자들이 효과적으로 활용할 수 있도록 설계되었습니다. 이 연구는 재무 분석의 발전을 위한 LLM의 잠재력을 입증하며, 보다 신뢰성 있는 투자 전략을 개발하는 데 기여할 것입니다.



### Zoning in American Cities: Are Reforms Making a Difference? An AI-based Analysis (https://arxiv.org/abs/2502.00008)
Comments:
          31 pages, 6 figures, 1 table

- **What's New**: 이 연구는 미국의 2000개 이상의 인구 조사 지명 장소의 조닝 문서를 자연어 처리(Natural Language Processing, NLP) 기술을 사용하여 분석하여 형식 기반 코드(Form-Based Codes, FBC)의 원칙을 나타내는 언어적 패턴을 발견했습니다. 연구 결과 FBC의 채택이 전국적으로 광범위하게 이루어졌으며, 지역에 따라 차이가 있음을 보여주었습니다. 특히 FBC는 더 높은 바닥 면적 비율, 일관된 도로 후퇴 및 더 작은 필지와 관련이 있으며, 이는 지속 가능한 도시 형성과 밀접한 연관이 있습니다.

- **Technical Details**: 조닝 코드는 일반적으로 정보가 비구조적이고 다양하여 정량적으로 비교하기 어렵습니다. 본 연구는 FBC가 도입된 곳에서 더 높은 보행 가능성, 짧은 통근 거리, 다가구 주택의 비율 증가와 같은 결과를 찾았습니다. 또한, 연구는 1950년 이후 개발된 지역에서도 이러한 연관성이 존재함을 보여주었습니다.

- **Performance Highlights**: FBC가 채택된 지역은 보행 가능한 환경을 촉진하고, 통근 거리 단축 및 다가구 주택 비율 증가와 같은 긍정적인 영향을 미쳤습니다. 이는 즉, FBC가 지속 가능한 도시 환경을 조성하는 데 중요한 역할을 한다는 것을 의미합니다. 본 연구는 조닝 코드 및 도시 지속 가능성을 평가하는 NLP의 유용성을 강조합니다.



### The TIP of the Iceberg: Revealing a Hidden Class of Task-in-Prompt Adversarial Attacks on LLMs (https://arxiv.org/abs/2501.18626)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)에 대한 새로운 클래스의 jailbreak 적대적 공격, 즉 Task-in-Prompt (TIP) 공격을 소개합니다. TIP 공격은 시퀀스-투-시퀀스(Sequence-to-Sequence) 작업을 모델의 프롬프트에 포함시켜 금지된 입력을 간접적으로 생성하는 방법입니다. 논문은 PHRYGE 벤치를 도입하여 이러한 공격의 효과성을 체계적으로 평가하였고, 이 기술이 GPT-4o와 LLaMA 3.2를 포함한 여섯 개의 최첨단 모델에서 안전장치를 우회할 수 있음을 입증했습니다.

- **Technical Details**: TIP 공격은 LLM이 주어진 입력에 대한 출력 전달을 학습하는 방식을 활용합니다. 안전 정렬 과정에서 LLM은 특정 트리거 단어를 인식하고 필터링하는 법을 배우는데, TIP 공격자는 이러한 특정 단어를 피하고 안전하지 않은 내용을 무해한 변환 작업 내에 숨깁니다. CAESAR 암호, 모스 코드, Base64 등을 포함한 다양한 인코딩 방식을 사용할 수 있어 탐지하기 어려운 공격 형태입니다.

- **Performance Highlights**: 공격자는 TIP 공격을 통해 공격자가 원하는 금지된 내용을 간접적으로 재도입할 수 있습니다. 이러한 접근 방식은 기존의 안전장치들을 우회하고 모델의 내부 메커니즘을 혼란스럽게 할 수 있음을 보여주었습니다. TIP 공격은 특정 기술에만 국한되지 않고 보다 일반적인 취약성을 드러내며, LLM 안전성을 보장하려면 더욱 정교한 방어 전략이 필요하다는 점을 강조합니다.



### Tuning LLM Judge Design Decisions for 1/1000 of the Cos (https://arxiv.org/abs/2501.17178)
- **What's New**: 이번 연구에서는 LLM(judge)을 평가하기 위한 하이퍼파라미터 효과를 체계적으로 분석하고 조정하는 방법을 제안합니다. 비용 문제를 해결하기 위해 multi-objective multi-fidelity 접근법을 활용하여 정확도와 비용 간의 거래를 지닌 judge 구성요소를 찾아냅니다. 이 방법론은 기존 벤치마크보다 뛰어난 정확도와 비용 효율성을 보이며, 오픈 가중치 모델을 사용하여 더 널리 접근 가능하도록 보장합니다.

- **Technical Details**: 이 연구에서는 LLM judge의 성능을 평가하기 위해 hyoerpameter(하이퍼파라미터) 조정 방법을 제안합니다. LLM 모델, 프롬프트, 추론 파라미터(예: 온도), 및 judge 선호도를 추출하기 위한 파싱 메커니즘을 체계적으로 분석합니다. 기법의 기본적인 목적은 자원 소모를 줄이고, 기존의 여러 judge 평가를 통해 성능을 최적화할 수 있는 새로운 방안을 모색하는 것입니다.

- **Performance Highlights**: 제안된 방법은 실세계 데이터셋에 대한 테스트를 통해 이전의 state-of-the-art judge 구성보다 뛰어난 성능을 보여주었습니다. 특히, 기존의 모델 평가 방식에 비해 큰 비용 절감 효과를 가져오는데, 이는 judge 평가에 소요되는 인건비를 대폭 줄일 수 있게 하고, 하이퍼파라미터 조정의 효율성을 يق단적으로 향상시킵니다. 연구의 결과는 LLM judge의 발전과 안정성 확보에 중요한 기여를 할 것으로 기대됩니다.



New uploads on arXiv(cs.IR)

### Rankify: A Comprehensive Python Toolkit for Retrieval, Re-Ranking, and Retrieval-Augmented Generation (https://arxiv.org/abs/2502.02464)
Comments:
          Work in Progress

- **What's New**: 본 논문은 정보 검색, 재순위화(re-ranking), 검색 증강 생성(RAG) 등 현대 자연어 처리(NLP) 애플리케이션을 위한 통합 프레임워크인 Rankify를 소개합니다. Rankify는 다양한 retrieval 기법과 최신 재순위 모델을 지원하며, 비교적 간편하게 라이브러리를 설치하고 사용할 수 있도록 설계되었습니다. 또한 연구자들이 다양한 실험을 진행할 수 있도록 40개의 커스터마이즈된 데이터셋을 포함하고 있습니다.

- **Technical Details**: Rankify는 sparse(BM25) 및 dense(DPR, ANCE 등) 리트리버와 24개의 최신 재순위 모델을 통합하여 사용자가 손쉽게 다양한 retrieval 및 ranking 실험을 수행할 수 있게 합니다. 재순위 기술은 pointwise, pairwise, listwise로 나누어지며, Rankify는 이러한 다양한 접근 방식을 지원합니다. RAG 통합을 통해 Rankify는 검색된 문서를 LLM에 전달하여 평가 과정을 연결합니다.

- **Performance Highlights**: Rankify는 실험에서의 일관성과 확장성을 보장하며, 다양한 평가 메트릭과 도구를 제공합니다. 개방형 문서화가 잘 되어 있어 사용자가 각 기능을 쉽게 탐색할 수 있습니다. Rankify는 PyPI 및 GitHub를 통해 무료로 제공되며, 연구자와 실무자에게 접근성이 높은 도구로 자리잡을 것입니다.



### Policy-Guided Causal State Representation for Offline Reinforcement Learning Recommendation (https://arxiv.org/abs/2502.02327)
- **What's New**: 이번 논문에서는 오프라인 강화 학습 기반 추천 시스템(RLRS)에서 사용자 선호를 효과적으로 반영하는 주(State) 표현 학습의 중요성을 강조하며, 새로운 접근 방식인 정책에 기초한 인과적 표현(Policy-Guided Causal Representation, PGCR)을 제안합니다. PGCR은 사용자 만족도에 가장 관련 있는 인과적 특징(Causal Relevant Components, CRCs)을 선택하고 주 표현을 학습하는 두 단계의 프레임워크로 구성되어 있습니다. 이 방법은 보상 함수에서 오스트와인 거리(Wasserstein distance)를 사용하여 CRC를 보존하도록 유도합니다.

- **Technical Details**: PGCR 방법론은 인과적 특징 선택 정책을 통해 원래와 수정된 상태 간의 평균 제곱 오차(Mean Squared Error, MSE)를 최소화함으로써 주 표현을 학습합니다. 첫 번째 단계에서는 인과적 관련 없는 구성 요소를 수정하면서 CRC만을 유지하는 수정된 상태를 생성합니다. 두 번째 단계에서는 이 수정된 상태에 따라 인코더를 훈련시켜, 최적의 결정에 필요한 정보만을 보존하는 잠재 표현을 학습합니다.

- **Performance Highlights**: 상세한 실험 결과에 따르면, PGCR은 오프라인 RL 기반 추천 시스템에서 추천 성능을 크게 개선시키며, 그 효과성을 입증합니다. 이 방식은 오프라인 데이터셋에서의 한계에도 불구하고 사용자 만족도에 직접 영향을 미치는 정보에 집중하게 해줍니다. PGCR의 접근 방식은 추천 시스템의 장기적 효율성과 사용자 경험 향상에 기여할 수 있음을 보여줍니다.



### Combinatorial Optimization Perspective based Framework for Multi-behavior Recommendation (https://arxiv.org/abs/2502.02232)
Comments:
          Accepted by KDD 2025 Research Track

- **What's New**: 이 논문은 다양한 사용자 행동 정보를 활용하여 추천의 정확성을 높일 수 있는 새로운 프레임워크인 COPF(Combinatorial Optimization Perspective Framework)를 제안합니다. 기존의 다중 행동 추천 방법들이 단일 행동 혹은 제한된 업무 간의 상관 관계를 고려하는 데 한계를 보인 반면, COPF는 조합 최적화(combinatorial optimization) 관점에서 사용자 행동 패턴을 효율적으로 모형화합니다.

- **Technical Details**: COGF(Combinatorial Optimization Graph Convolution Network)는 사용자 행동 패턴에 대해 다양한 제약 조건을 두어 다중 행동 융합을 최적화합니다. 반면 DFME(Distributed Fitting Multi-Expert Networks)는 다중 행동 예측 단계에서 특징(feature) 및 레이블(label) 수준에서의 동기화를 개선하여, 부정적인 정보 전이를 방지하도록 설계되었습니다. 이 두 구성 요소는 조합 최적화를 활용하여 추천 시스템의 성능을 크게 향상시킵니다.

- **Performance Highlights**: 세 가지 실제 데이터세트에 대한 포괄적인 실험을 통해 COPF의 우수성을 증명하였으며, COGCN과 DFME 모듈의 유효성도 검증하였습니다. 기존의 방법들에 비해 사용자의 행동 패턴을 정밀하게 캡처하고 이로 인해 추천의 품질이 향상되는 것을 확인하였습니다.



### Large Language Models for Recommendation with Deliberative User Preference Alignmen (https://arxiv.org/abs/2502.02061)
- **What's New**: 이번 연구에서는 Deliberative Recommendation 작업을 새롭게 제안하여 사용자 선호도에 대한 사고 과정을 향상시킬 수 있는 방법을 소개합니다. 기존의 추천 시스템이 단순히 사용자 피드백을 생성하는 방식에 대한 한계를 극복하기 위해, 사용자의 피드백을 명시적으로 추론하는 과정이 중요한 목표로 포함됩니다. 이를 통해 LLM(large language models)을 더욱 신뢰할 수 있는 추천 시스템으로 발전시키고자 합니다.

- **Technical Details**: 제안된 Deliberative User Preference Alignment(DeliRec) 프레임워크는 사용자의 피드백에 대한 단계별 사고 과정을 통해 LLM이 보다 나은 추천을 할 수 있도록 돕습니다. DeliRec은 Prefene Distillation, Preference Matching, Feedback Prediction의 세 가지 기본적인 추론 능력을 통해 사용자 피드백을 분석하고, 항목 특성과의 매칭을 향상시킵니다. 각 단계별 전문가들이 협력하여 서로 다른 목표를 달성하도록 설계되었으며, QLoRA(adapter)를 사용하여 계산 및 메모리 비용을 줄이는 방식을 채택했습니다.

- **Performance Highlights**: 세 가지 실제 데이터셋을 통해 DeliRec 프레임워크의 효율성과 추론 품질을 폭넓게 실험하였으며, 그 결과 기존 방법 대비 우수한 예측 정확성을 보였습니다. 연구 결과는 DeliRec의 구조적 합리성에 대한 강력한 근거를 제시하며, 느린 사고 장치의 잠재력을 보여줍니다. 코드와 데이터셋은 공개되어 연구자들이 쉽게 접근할 수 있도록 제공됩니다.



### A Scalable Crawling Algorithm Utilizing Noisy Change-Indicating Signals (https://arxiv.org/abs/2502.02430)
- **What's New**: 본 논문에서는 웹 페이지의 변경 여부를 알리는 사이드 정보를 활용하여 웹 새로 고침 크롤링(web refresh crawling) 정책을 최적화하는 방법을 제시합니다. 기존의 연구에서는 완전하고 정확한 변경 정보를 가정했으나, 본 논문은 지각된 변화가 노이즈가 포함된 경우를 고려하여 모델을 확장합니다. 이를 통해 크롤링 효율성을 높이고, 자원 소비를 줄일 수 있는 새로운 알고리즘을 제안합니다.

- **Technical Details**: 제안된 알고리즘은 여러 가지 웹 페이지에서 수신한 변화 신호(변경 신호, Change-Indicating Signals)를 최적 방식으로 활용합니다. 이 알고리즘은 중앙 집중식 계산 없이도 배포가 가능하며, 일정한 총 속도로 웹 페이지를 크롤링할 수 있도록 설계되었습니다. 또한, 총 대역폭 제한이 바뀌더라도, 자동으로 새로운 최적 솔루션으로 조정될 수 있는 능력을 갖추고 있습니다.

- **Performance Highlights**: 실험 결과, 제안한 접근 방식이 웹 페이지 크롤링의 효율성을 크게 향상시킬 수 있음을 보여주었습니다. 특히, 노이즈가 포함된 변화 신호를 효과적으로 처리할 수 있는 능력을 강조합니다. 이러한 방식은 크롤링 자원의 낭비를 줄이고, 최신 정보를 적시에 제공하는 데에도 기여할 것입니다.



### Multilingual Attribute Extraction from News Web Pages (https://arxiv.org/abs/2502.02167)
- **What's New**: 이 논문은 다국어 뉴스 기사 웹 페이지에서 속성을 자동으로 추출하는 문제를 다룹니다. 최근 신경망 모델들이 반구조화된 웹 페이지에서의 정보 추출에 높은 효율성을 보였으나, 이들은 주로 영어 데이터로 사전 훈련되어 다른 언어 웹 페이지에의 적용이 복잡하다는 한계가 있었습니다. 저자들은 6개의 언어로 구성된 3,172개의 뉴스 웹 페이지로 이뤄진 다국어 데이터셋을 준비하였으며, 이를 통해 MarkupLM 모델을 미세 조정(fin-tune)하여 성능을 평가했습니다.

- **Technical Details**: 본 논문에서는 영어로 사전 훈련된 MarkupLM 모델과 다국어 데이터를 기반으로 다시 훈련한 DOM-LM 모델을 활용하여 뉴스 웹 페이지의 속성을 추출합니다. 각 모델은 HTML 문서 이해를 위한 고급 사전 훈련 모델로, MarkupLM은 24백만 개의 영어 웹 페이지에서 훈련되었으며, DOM-LM은 다양한 구조적 특성을 고려하여 훈련되었습니다. 이 두 모델의 성능 비교를 통해, 기존 오픈 소스 뉴스 데이터 추출 도구보다 뛰어난 속성 추출 메트릭을 달성했습니다.

- **Performance Highlights**: 모델 비교 결과, 다국어 DOM-LM은 뉴스 웹사이트 속성 추출에서 가장 높은 성능을 기록했습니다. 다양한 언어 그룹에서 테스트한 결과, 제안된 방법이 장애물 없이 새로운 웹사이트에서도 잘 작동함을 입증했습니다. 또한, 영어로 번역된 페이지의 질적 영향도 평가하였으며, 이러한 평가를 통해 다국어 속성 추출의 가능성을 확인했습니다.



### Policy Design for Two-sided Platforms with Participation Dynamics (https://arxiv.org/abs/2502.01792)
Comments:
          preprint, under review

- **What's New**: 이 논문은 양측 플랫폼(예: 비디오 스트리밍, 전자상거래)에서의 참여 역학과 정책 설계를 다루고, 특히 참가자 개체 수의 변화가 플랫폼의 사회적 복지와 연관되어 있음을 강조합니다. 기존 추천 정책이 일반적으로 이러한 '인구 효과(population effects)'를 고려하지 않는 점을 지적하며, 이는 플랫폼의 장기적인 건강성에 부정적인 영향을 미칠 수 있습니다. 게다가, 논문에서는 공급자 측의 고려사항을 효과적으로 다루는 것이 왜 중요한지를 설명하고, 인구 성장에 기반한 장기 목표 최적화를 위한 간단한 알고리즘을 제안합니다.

- **Technical Details**: 이 연구는 K개의 뷰어 그룹과 L개의 제공자 그룹으로 분류된 아규먼트의 동역학을 모델링합니다. 각 그룹의 인구는 노출이나 만족도를 기준으로 상대적으로 변화하며, 인구의 변화는 주어진 추천 정책에 의해 결정됩니다. 추천 정책은 뷰어와 제공자 그룹의 상호작용에서 발생하는 비선형성을 반영할 수 있도록 설계되며, 단기적 유틸리티 뿐만 아니라 장기적 인구 변화(큰 인구 증가의 잠재적 효과)를 고려합니다.

- **Performance Highlights**: 제안된 알고리즘은 'look-ahead' 정책이라고 불리며, 이는 즉각적인 인구가 아닌 예상되는 장기 인구를 기반으로 유틸리티를 최적화합니다. 실험 결과는 이 알고리즘이 마이옵-그리디(myopic-greedy) 정책 및 균일 랜덤 정책보다 더 나은 성과를 보이는 것을 보여줍니다. 이 연구는 또한 공급자 측의 공정한 노출의 중요성을 강조하며, 인구 성장과 사회적 복지 향상을 위한 추천 전략의 중요성을 뒷받침합니다.



### On Bob Dylan: A Computational Perspectiv (https://arxiv.org/abs/2502.01772)
- **What's New**: 본 연구는 Cass Sunstein의 'On Bob Dylan' 에세이를 확장하여 Bob Dylan의 가사를 1962년부터 2012년까지 대규모 computational analysis (컴퓨테이셔널 분석)를 통해 분석합니다. 이 연구는 Dylan의 가사에서 개념 대 개념 관계를 추출하고, 이를 기반으로 방향성 지식 그래프를 구축하여 그의 주제적 구조를 포착합니다. 결과적으로, Dylan의 가사는 메타포에 대한 의존도가 증가하고, 감정 프로파일이 진화하며, 'dishabituation' (탈습관화)이 높아지는 것으로 나타났습니다.

- **Technical Details**: 연구자는 Bob Dylan의 1962년부터 2012년까지의 스튜디오 앨범 가사를 수집했습니다. o3-mini-high라는 대형 언어 모델을 활용하여 가사를 분석하고, 관련된 개념 쌍을 추출하여 각 개념 간의 관계를 구조화된 JSON 형식으로 저장했습니다. 이 과정에서 연구자는 노드(개념)를 정규화하고 관계를 확인하기 위해 다양한 네트워크 측정을 계산했습니다.

- **Performance Highlights**: 분석 결과, Dylan의 가사는 시대에 따라 테마가 다양하게 변하고, 이는 그의 음악적 변화에 대한 깊은 통찰을 제공합니다. 그는 특히 'movement', 'protest', 'mythic imagery'와 같은 주제를 다루며 그의 경력에 걸쳐 동적인 변화를 보였습니다. 이번 연구는 예술가의 개인적인 진화를 이해할 수 있는 새로운 방법론을 제시하며, 문화 및 창조적 변화의 연구에도 널리 적용될 수 있습니다.



### Multimodal Inverse Attention Network with Intrinsic Discriminant Feature Exploitation for Fake News Detection (https://arxiv.org/abs/2502.01699)
- **What's New**: 이번 연구는 마르텐츠 적인 모드별 특성을 활용한 'Multimodal Inverse Attention Network (MIAN)' 프레임워크를 제안합니다. 기존 접근법들은 모드 특수한 표현과 불일치 특징을 충분히 활용하지 못했으나, MIAN은 뉴스 콘텐츠의 본질적인 특징을 탐색하여 허위 뉴스 검출을 향상시킵니다. MIAN은 지역 내-지역 상호 작용과 지역 내-전역 상호 작용을 통해 다층 학습 모듈을 도입하여 향상된 단일 모드 표현을 생성합니다.

- **Technical Details**: MIAN은 모드 간 상호 작용 모듈을 통해 조관계 메커니즘을 사용하여 정제된 단일 모드 표현 간의 의존 관계를 설정합니다. 또한, 역 주의 메커니즘을 통해 각 모드에서 모순된 패턴과 의미적 편차를 강조함으로써 불일치 특징을 명확히 추출하는 데 중점을 둡니다. 이 구조는 뉴스 텍스트와 이미지 간의 명확한 관계를 탐색하고 단일 모드 및 다중 모드 내 본질적 식별 정보를 활용합니다.

- **Performance Highlights**: 광범위한 실험에서 MIAN은 여러 벤치마크 데이터 세트에서 기존의 최신 방법들에 비해 유의미한 성능 향상을 보였습니다. MIAN은 다양한 실제 사례에서 허위 뉴스의 탐지를 효과적으로 개선하며, 사회적 안전을 위한 솔루션을 제공합니다. 이 연구는 공공 정보의 신뢰성과 무결성을 보장하기 위한 자동화된 허위 뉴스 검출 기법의 개발을 촉진하는 데 기여하고 있습니다.



### Addressing Delayed Feedback in Conversion Rate Prediction via Influence Functions (https://arxiv.org/abs/2502.01669)
- **What's New**: 본 논문에서는 지연된 피드백 문제를 해결하기 위해 Influence Function 기반의 새로운 프레임워크인 Delayed Feedback Modeling (IF-DFM)을 제안합니다. IF-DFM은 새롭게 수집된 전환 데이터를 사용하여 모델 매개변수에 미치는 영향을 추정하여 효율적으로 매개변수를 업데이트할 수 있도록 설계되었습니다. 이 프레임워크는 전체 재교육 없이도 모델을 적응시킬 수 있게 해줘, 광고 전환율 예측에서 더욱 강력한 성능을 발휘합니다.

- **Technical Details**: IF-DFM의 핵심은 인플루언스 함수(influence function)를 활용하여 잘못 레이블된 샘플을 올바른 샘플로 전환하는 과정에서 발생하는 매개변수 변화를 매개변수 업데이트로 연결짓는 것입니다. 이는 또한 새로운 행동 데이터를 통합할 수 있는 기능을 제공하여, 모델이 최신 사용자 상호작용에 적응하도록 돕습니다. 이러한 접근은 히에라르크 역행렬-벡터 곱(inverse Hessian-vector product) 계산을 최적화 문제로 재구성하여 계산 효율성을 극대화합니다.

- **Performance Highlights**: 광범위한 실험을 통해 IF-DFM은 기존의 최첨단 방법들보다 뛰어난 예측 정확도와 모델 적응성을 보여주었습니다. 특히, IF-DFM은 광고 전환율 예측에서 실시간 사용자 행동 변화에 민첩하게 반응할 수 있어, 이론적으로 설정된 한계를 넘어서는 성과를 얻었습니다. 이러한 결과는 대용량 데이터 환경에서도 효율적인 성능 개선을 보여줍니다.



### Query Brand Entity Linking in E-Commerce Search (https://arxiv.org/abs/2502.01555)
- **What's New**: 이 연구에서는 전자 상거래 검색 쿼리를 위한 브랜드 엔티티 링크(linking) 문제를 다룹니다. 새로운 접근 방식으로, 두 단계로 구성된 프로세스와 엔드 투 엔드 방식이 제안되었으며, 각 방식이 브랜드 엔티티를 추출하는 데 효과적입니다. 특히, 엔드 투 엔드 모델을 도입하여 입력 텍스트로부터 직접적으로 대상을 찾아내는 방법이 있습니다.

- **Technical Details**: 연구는 메타TS-NER(MetaTS-NER)라는 다국어 모델을 활용하여 브랜드명 인식을 수행하며, 이는 다중 레이블 분류에 기반하여 제품 검색 쿼리에서 브랜드를 감지합니다. 두 단계 프레임워크는 NER 모델을 통해 브랜드명을 추출한 후, 면목에 기반한(match) 추출 과정을 거칩니다. 또한, PECOS 도구를 활용하여 방대한 클래스(brand) 공간에서의 다중 클래스 분류 문제 해결도 시도합니다.

- **Performance Highlights**: 연구에서 제안하는 기법은 오프라인 벤치마크 및 온라인 A/B 테스트를 통해 성능을 검증하였습니다. 세부적으로는 브랜드 검색 쿼리에 대한 브랜드 엔티티 예측에서 높은 정확도를 보여주었으며, 짧은 쿼리 길이(평균 2.4 단어)를 효과적으로 처리할 수 있는 방법론을 제시합니다. 연구 결과는 제품 검색의 효율성을 향상시킬 것입니다.



### VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos (https://arxiv.org/abs/2502.01549)
- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG) 개념을 비디오 콘텐츠에 처음으로 적용한 VideoRAG 프레임워크를 소개합니다. 이는 복잡한 멀티모달 비디오 지식을 처리하고 이해하기 위해 특별히 설계된 모델로, 기존의 텍스트 기반 접근 방식의 한계를 넘는 혁신적인 접근 방식입니다. VideoRAG는 두 개의 상호 연결된 구성 요소인 Multi-Modal Video Knowledge Indexing framework와 Knowledge-Grounded Multi-Modal Retrieval paradigm을 통해 멀티모달 비디오 내용을 효과적으로 조직하고 인덱싱할 수 있게 합니다.

- **Technical Details**: VideoRAG의 핵심 혁신은 그래프 기반의 텍스트 지식 기초와 멀티모달 컨텍스트 인코딩을 통합한 이중 채널 아키텍처에 있습니다. 이 프레임워크는 다양한 비주얼 특징을 효율적으로 보존하고, 여러 비디오에 걸쳐 정확한 지식 그래프를 구축하여 비디오 간의 의미적 의존성을 유지합니다. 또한, 비디오 지식 검색을 효율적으로 수행하기 위해 LLM 기반의 키워드 추출과 비전-언어 모델을 기반으로 한 텍스트 그라운딩을 결합한 이중 단계의 콘텐츠 추출 프로세스를 활용합니다.

- **Performance Highlights**: VideoRAG는 160개 이상의 비디오로 구성된 LongerVideos 벤치마크에서 종합적인 실증 평가를 통해 기존 RAG 대안 및 긴 비디오 이해 방법과 비교해 상당한 성과를 보여줍니다. 이 프레임워크는 교육 콘텐츠 분석, 미디어 아카이빙, 비디오 기반 지식 추출과 같은 분야에서 비디오 이해력을 크게 향상시키며, 기존 단일 비디오에 제한된 기존 데이터셋을 넘어서는 지식을 제시합니다. 실험 결과와 사례 연구를 통해 VideoRAG의 실질적인 응용 가능성을 밝혀내며, 비디오 간 이해 향상에 기여하는 새로운 가능성을 열었습니다.



### Augmented Knowledge Graph Querying leveraging LLMs (https://arxiv.org/abs/2502.01298)
- **What's New**: 이 논문에서는 SparqLLM이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 Knowledge Graphs (KGs)에서의 쿼리 생성을 자동화하고, 사용자 친화적인 인터페이스를 제공하여 비전문가도 쉽게 KGs와 상호작용할 수 있게 도와줍니다. SparqLLM은 자연어 질문을 SPARQL 쿼리로 변환하여, 효율적이고 정확한 데이터 시각화를 가능하게 합니다.

- **Technical Details**: SparqLLM은 Retrieval-Augmented Generation (RAG) 방식을 채택하여, 자연어 질문을 SPARQL 쿼리로 변환할 수 있게 설계되었습니다. 데이터 통합을 위한 Extract, Transform, Load (ETL) 파이프라인을 통해 원시 데이터를 KG로 구축하며, 대형 언어 모델인 Large Language Models (LLMs)를 활용하여 쿼리 생성을 지원합니다. 이 시스템은 템플릿 기반의 방법을 통합하여 쿼리의 신뢰성을 높이고 의미적 오류를 줄입니다.

- **Performance Highlights**: 엄격한 실험 평가 결과, SparqLLM은 높은 쿼리 정확도와 향상된 신뢰성을 보여주었습니다. 또한 시각화 대시보드가 결과를 직관적으로 제시하여 사용자의 접근성과 편의성을 증대시켰습니다. 이러한 성능은 KGs에 대한 세밀한 접근을 가능하게 하며, 산업 환경에서의 데이터 관리와 분석에 크게 기여할 것으로 기대됩니다.



### GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation (https://arxiv.org/abs/2502.01113)
Comments:
          19 pages, 6 figures

- **What's New**: GFM-RAG는 기존의 RAG 모델들을 개선하기 위해 개발된 새로운 그래프 기반 모델로, 복잡한 쿼리-지식 관계를 효과적으로 포착할 수 있도록 설계되었습니다. 이 모델은 800만 개의 파라미터를 갖춘 혁신적인 그래프 신경망(graph neural network)을 사용하여 복잡한 질문에 대한 응답을 보다 효율적으로 생성합니다. GFM-RAG는 기존의 동적 그래프 모델에서 발생하는 노이즈와 불완전성 문제를 해결하며, 사전 조정 없이도 이전에 보지 못한 데이터셋에서 저명한 성능을 보입니다.

- **Technical Details**: GFM-RAG는 60개의 지식 그래프와 1400만 개의 트리플로 구성된 대규모 데이터세트에서 이중 단계의 훈련 과정을 거쳐 학습됩니다. 실험에서는 HotpotQA, MuSiQue, 2WikiMultiHopQA와 같은 세 개의 다중 홉 QA 데이터셋을 활용하였으며, 700,000개의 문서에서 표본을 추출하여 KG 인덱스를 구축합니다. GFM-RAG는 다양한 도메인에 걸쳐 7개의 특정 RAG 데이터셋을 대상으로 평가하여 일반화 가능성을 확인하였습니다.

- **Performance Highlights**: GFM-RAG는 다중 홉 QA 데이터셋 및 도메인 특정 데이터셋에서 최신 기술 수준의 성능을 달성하였으며, 신경망 확장 법칙(neural scaling laws)과의 일치를 유지합니다. extensive 임상 실험을 통해 경쟁 모델들과의 비교에서 우위를 점하며, 데이터 효율성과 성능의 개선 가능성이 확인되었습니다. 이러한 결과는 GFM-RAG가 앞으로 더 많은 연구와 개발의 잠재력을 지니고 있음을 보여줍니다.



### RankFlow: A Multi-Role Collaborative Reranking Workflow Utilizing Large Language Models (https://arxiv.org/abs/2502.00709)
- **What's New**: 이 논문은 정보 검색 (Information Retrieval, IR) 시스템에서 후보 구문을 특정 쿼리와의 관련성에 따라 정렬하는 리랭킹(reranking) 방식인 RankFlow를 소개합니다. RankFlow는 대형 언어 모델 (Large Language Models, LLMs)을 활용하여 쿼리 리라이터, 의사 응답자, 구문 요약자, 리랭커의 네 가지 역할을 수행합니다. 이러한 다기능 접근 방식은 쿼리를 정확하게 해석하고, LLMs의 방대한 지식을 활용하며, 구문을 간결하게 정제하고, 포괄적으로 평가하여 리랭킹 성능을 개선합니다.

- **Technical Details**: RankFlow는 쿼리 리라이트(Query Rewriter)와 구문 요약(Passage Summarization)을 통해 작업 수행의 일관성과 정확성을 향상시키는 구조화된 워크플로우를 제안합니다. LLM의 역할은 다각화되어 있어, 다양한 데이터셋에 대해 평가를 받아 LLM 기반 리랭킹 시스템의 기존 문제를 극복합니다. RankFlow의 실험 결과는 널리 인정받는 IR 벤치마크인 TREC-DL, BEIR 및 NovelEval에서 기존의 선두 접근 방식보다 더 우수한 성능을 보였습니다.

- **Performance Highlights**: RankFlow는 BEIR의 네 가지 데이터셋(예: Covid, NFCorpus, SciFact, Robust04)에서 기존 최첨단 방법보다 평균 2.5%p 향상된 nDCG@10 점수를 기록했습니다. 이러한 성과는 비단 성능에 국한되지 않고, RankFlow의 각 역할이 리랭킹에 미치는 개별적인 기여도를 분석함으로써 향후 연구를 위한 중요한 통찰을 제공합니다. 최종적으로 RankFlow는 LLM을 기반으로 한 다기능 리랭킹 워크플로우의 새로운 가능성을 여는 기폭제 역할을 할 것으로 기대됩니다.



### Retracted Citations and Self-citations in Retracted Publications: A Comparative Study of Plagiarism and Fake Peer Review (https://arxiv.org/abs/2502.00673)
- **What's New**: 이번 연구는 문헌에서 재tracted citations(철회된 인용)의 영향력을 분석하면서, plagiarism(표절)과 fake peer review(가짜 동료 심사)라는 두 개의 철회 범주에 집중했습니다. 연구 결과, 표절 문제를 가진 문헌은 가짜 동료 심사에 비해 인용이 더 많이 발생하고 있으며, 각각의 인용이 철회된 이유를  분석한 것이 특징입니다. 또한,  이 연구는 self-citations(자기 인용)의 분포와 기여도를 조사하여, 학문적 불완전성을 알리기 위한 기초 자료를 제공합니다.

- **Technical Details**: 본 연구는 Scopus에서 수집한 33,188개의 출판물 데이터를 분석하였으며, retraction(철회) 기록에 대한 정확한 연계를 위해 DOI를 통해 자료를 필터링했습니다. 필터링 후, 약 26,528개의 문서가 최종 분석 대상으로 선정되었고, 이 데이터는 plagiarism(표절)과 fake peer review(가짜 동료 심사)로 분류되었습니다. 이러한 철회 범주는 문서의 철회 사유가 명확하게 문서화되어 있어 연구에 적합합니다.

- **Performance Highlights**: 연구 성과는 표절 관련 문서의 인용이 2.5배 더 많고, 총 재급 기사가 1.8배 더 많은 것으로 나타났습니다. 또한, 46%의 재급 인용은 표절에 기인하고 있으며, 가짜 동료 심사에서는 53.6%가 가짜 동료 심사로 분류되었습니다. 흥미로운 점은, 가짜 동료 심사 사례가 표절 사례에 비해 더 빠르게 식별되고 철회된다는 점입니다.



### Personalized Denoising Implicit Feedback for Robust Recommender System (https://arxiv.org/abs/2502.00348)
Comments:
          To appear in WWW 2025

- **What's New**: 이 논문에서는 기존의 노이즈 제거 방법의 한계를 분석하고, 개인의 손실 분포를 기반으로 한 새로운  Denoising 전략인 PLD를 제안합니다. PLD는 사용자마다 나눌 수 있는 개인 손실 분포에서 노이즈가 없는 상호작용을 우선시하여 효율적으로 노이즈 상호작용을 최적화합니다. 이 방법은 상대적으로 명확한 사용자 간 손실 분포의 차이를 활용하여 노이즈를 제거하는 데 있어 더 효과적인 결과를 도출합니다.

- **Technical Details**: 연구에서는 노이즈가 포함된 사용자 피드백에 대해 두 가지 주요 문제점을 발견했습니다. 첫째, 정상 상호작용과 노이즈 상호작용 간의 손실 분포에서 중복이 매우 높다는 점입니다. 둘째, 점수 기반 손실 함수에서 쌍별 손실 함수로의 전환 시 중복이 증가합니다. PLD에서는 사용자의 상호작용 항목을 균등하게 샘플링한 후, 사용자 개인의 손실 분포에 따라 최적화할 항목을 재샘플링합니다.

- **Performance Highlights**: 다양한 노이즈 비율이 포함된 세 가지 데이터셋에서 수행된 광범위한 실험을 통해 PLD의 효용성과 강건성을 입증했습니다. PLD는 BCE 및 BPR 손실을 사용할 때 모두 최첨단 성능을 달성하여 현대 추천 시스템에서 노이즈의 영향을 크게 줄일 수 있음을 보여주었습니다. 이는 기존 방법들보다 훨씬 더 향상된 성능을 가지며, 이론적 분석을 통해 그 효과성을 확인했습니다.



### MIM: Multi-modal Content Interest Modeling Paradigm for User Behavior Modeling (https://arxiv.org/abs/2502.00321)
- **What's New**: 본 논문에서는 Multi-modal Content Interest Modeling(오른 발제, MIM)이라는 새로운 패러다임을 제안합니다. 이는 기존의 ID 임베딩 기반 접근방식의 한계를 극복하고, 콘텐츠와 사용자 관심사 간의 차이를 메꾸기 위한 세 가지 주요 단계로 구성됩니다. 또한, Taobao 플랫폼에서 오프라인 실험과 온라인 A/B 테스트를 통해 MIM의 효율성을 입증하였습니다.

- **Technical Details**: MIM은 사전 훈련, 콘텐츠-관심 인식 감독 세부 조정(C-SFT), 및 콘텐츠-관심 인식 UBM(CiUBM)이라는 세 가지 핵심 단계를 포함합니다. 이 과정에서 사용자 행동 신호를 활용하여 임베딩을 사용자 선호와 정렬되도록 유도하며, ID 기반 협업 필터링 신호를 통합하여 효과적인 사용자 행동 모델링 프레임워크를 제공합니다.

- **Performance Highlights**: MIM은 Taobao에서의 대규모 데이터 세트 실험을 통해 클릭률(CTR)을 +14.14% 개선시키고, 수익률(RPM)을 +4.12% 증가시켜 실질적인 산업 응용 가능성을 보여주었습니다. 이러한 개선은 MIM의 효율적인 훈련 및 추론 구조 덕분으로, 다양한 추천 작업에 널리 적용될 수 있습니다.



### Middleman Bias in Advertising: Aligning Relevance of Keyphrase Recommendations with Search (https://arxiv.org/abs/2502.00131)
- **What's New**: 이번 논문에서는 e-commerce 셀러들이 판매 상품의 광고 효과를 높이기 위해 사용되는 키프레이즈 추천의 중요성과 이와 관련된 한계점을 다룹니다. 특히, 클릭 및 판매 신호에 대한 편향된 학습이 키프레이즈의 연관성을 잘못 판단하게 만든다는 점을 강조하고, 이를 해결하기 위해 광고와 검색 시스템 간의 상호작용을 재정립합니다.

- **Technical Details**: 키프레이즈의 연관성을 분석하기 위해 광고가 생성한 키프레이즈와 구매자에게 도달하기 위한 검색의 중개 역할을 고려합니다. 연구에서는 bi-encoder와 cross-encoder를 활용하여 두 시스템 간의 연관성 정렬을 모델링하고, 이벤트 중심의 데이터에서 발생한 중개자 편향에 대한 해결책을 제시합니다.

- **Performance Highlights**: 이 실험은 eBay 내에서 셀러들이 사용할 수 있는 스케일 가능한 솔루션을 목표로 하며, 2400만 개의 키프레이즈-아이템 쌍을 포함하는 데이터를 통해 키프레이즈가 검색에서 어떻게 평가되는지를 분류합니다. 이러한 접근법은 기존의 BERT 모델과 비교하여 더 나은 성능을 보였으며, 최종적으로 광고와 검색 간의 일관성을 향상시키는 데 기여할 것으로 기대됩니다.



### Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models (https://arxiv.org/abs/2502.01386)
- **What's New**: 본 논문은 RAG 시스템을 겨냥한 주제 지향적 적대적 여론 조작 공격을 다루며, 기존의 사실적 단일 쿼리 조작을 넘어서는 실질적인 접근 방식을 제안합니다. 새로운 공격 방식인 Topic-FlipRAG를 통해 여러 관련 쿼리에 걸쳐 여론을 변화시키는 두 단계의 조작 파이프라인을 구현합니다. 실험 결과, 이러한 공격이 사용자의 정보 인식에 중대한 영향을 미치는 것으로 나타났으며, 현재의 완화 방법으로는 이러한 공격에 효과적으로 대응할 수 없음을 강조합니다.

- **Technical Details**: Topic-FlipRAG는 두 단계의 공격 방법론으로 구성되어 있으며, 첫 번째 단계에서는 LLM의 내부 의미 지식을 활용하여 목표 문서에 대한 은밀한 적대적 수정이 이루어집니다. 두 번째 단계에서는 공개 소스 신경 순위 모델(NRM)로부터의 그래디언트를 사용하여 주제별 적대적 트리거를 생성합니다. 이러한 방법은 문서의 의미적 수준에서 섬세한 조작을 가능하게 하며, RAG 모델의 출력에서 특정 주제에 대한 여론을 효과적으로 변화시킵니다.

- **Performance Highlights**: 실험에서 Topic-FlipRAG는 네 가지 도메인에서 0.5의 평균 입장 변화를 기록하며, 다른 기준선 방법보다 현저히 우수한 성능을 보여줍니다. 또한, 사용자 실험 결과에 따르면, 사용자가 독성 RAG 시스템과 상호작용한 후 논란 있는 주제에 대한 입장이 16% 이상 변화하는 것으로 나타났습니다. 이러한 결과는 Topic-FlipRAG가 정보의 제시 및 인식 방식에 영향을 미칠 수 있다는 것을 확인시켜 주며, 강력하고 적응 가능한 방어 전략의 필요성을 부각시킵니다.



### PSSD: Making Large Language Models Self-denial via Human Psyche Structur (https://arxiv.org/abs/2502.01344)
Comments:
          WWW '25

- **What's New**: 이번 논문에서는 LLM(대형언어모델)의 추론 정확성을 개선하기 위한 새로운 방법인 PSSD(자기부정 구조)를 제안합니다. 이 접근법은 인간 사고 구조를 모방하여 세 가지 연결된 역할(직관 기반의 id, 규칙 기반의 superego, 스크립트 중심의 ego)을 통해 LLM의 내부 잠재력을 극대화합니다. PSSD는 LLM이 스스로 실수를 인식하고 이를 수정하는 과정을 보다 유기적으로 구성하여 더 나은 성능을 목표로 합니다.

- **Technical Details**: PSSD는 Freudian(프로이트) 이론을 바탕으로 하여, 각 역할이 상호작용하여 정교한 결과를 도출하도록 설계되었습니다. id 역할은 LLM의 직관적 시도를 통해 다양한 추론 경로를 생성하고, superego 역할은 이러한 시도를 규제하기 위한 규칙을 제공하며, ego 역할은 이를 종합하여 실행 가능한 스크립트를 생성합니다. 이 방식은 기존 LLM의 구조와 통합할 수 있어, 타 기법에 비해 효율성과 유연성을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, PSSD는 기존의 방법들보다 뛰어난 성능을 나타냈으며, 특히 LLM의 자원 소모 문제를 해결하는 데 효과적임을 보여주었습니다. 이 방법은 LLM이 자가 교정 기능을 발전시키고, 동시에 고품질의 결과를 신속하게 도출할 수 있도록 지원합니다. 결국, 이 연구는 LLM의 내부 잠재력을 활용하는 새로운 방향성을 제시하며, 추론 능력 개선에 중요한 기여를 하고 있습니다.



### DeepRAG: Thinking to Retrieval Step by Step for Large Language Models (https://arxiv.org/abs/2502.01142)
- **What's New**: DeepRAG라는 새로운 프레임워크가 소개되었습니다. 이 프레임워크는 Retrieval-Augmented Generation(RAG)을 Markov Decision Process(MDP)로 모델링하여 전략적이고 적응적인 정보 검색을 가능하게 합니다. DeepRAG는 질의를 반복적으로 분해함으로써 외부 지식을 검색할지 파라메트릭(reasoning) 지식에 의존할지를 동적으로 결정합니다. 실험 결과, DeepRAG는 기존 시스템에 비해 21.99% 더 높은 정확도로 답변을 제공하며 검색 효율성도 향상되었습니다.

- **Technical Details**: DeepRAG는 세 가지 핵심 단계로 구성됩니다: 1) Binary Tree Search를 통해 각 서브쿼리와 관련된 경로를 탐색합니다. 2) Imitation Learning을 활용하여 최소 검색 비용으로 올바른 답변에 도달하는 추론 과정을 학습합니다. 3) Chain of Calibration을 통해 LLM의 내부 지식을 조정하여 원활한 지식 경계 인식을 지원합니다. 각 단계는 MDP의 상태, 행동, 전이 역학, 보상 기능을 정의하여 체계적으로 구성됩니다.

- **Performance Highlights**: DeepRAG는 다섯 개의 오픈 도메인 QA 데이터셋에서 실험되어 그 효과가 검증되었습니다. HotpotQA와 같은 multi-hop factual QA에서 상향된 성능을 보였으며, CAG와 같은 시계열 QA에서도 우수한 결과를 나타냈습니다. 추가 분석을 통해 DeepRAG가 검색 결정과 파라메트릭 지식 간에 더 강한 상관관계를 나타내며, 이는 보다 효과적인 지식 경계 조정으로 이어집니다.



### HintEval: A Comprehensive Framework for Hint Generation and Evaluation for Questions (https://arxiv.org/abs/2502.00857)
Comments:
          Submitted to SIGIR 2025

- **What's New**: 이번 연구에서는 HintEval이라는 새로운 파이썬 라이브러리를 소개합니다. 이 라이브러리는 Hint Generation과 Hint Evaluation을 위한 통합된 프레임워크로, 다양한 데이터세트와 평가 기준을 제공합니다. HintEval은 산재해 있는 자원을 하나의 도구로 통합하여 연구자들이 쉽게 접근하고 활용할 수 있도록 도와줍니다.

- **Technical Details**: HintEval은 여러 형식의 데이터셋을 제공하며, 이로 인해 연구자들이 데이터를 준비하는 데 소요되는 시간을 절약할 수 있습니다. 또한 Hint Generation(힌트 생성)과 Hint Evaluation(힌트 평가)을 위한 표준화된 접근 방식을 통해, 연구자들이 비교적 쉽게 일관된 평가 기준을 적용할 수 있도록 합니다. 이 라이브러리는 자세한 온라인 문서와 함께 제공되어 사용자가 기능을 탐색하고 쉽게 시작할 수 있습니다.

- **Performance Highlights**: HintEval은 학습 지원에 있어서 중요한 진전을 이루는 데 기여합니다. 연구자들은 HintEval을 통해 적극적으로 힌트를 최적화하고 교육 및 문제 해결 작업에서 어떻게 활용할 수 있는지에 대한 깊은 이해를 확보할 수 있습니다. 이 라이브러리는 GitHub와 PyPI에서 무료로 사용할 수 있어, 연구자와 실무자들에게 광범위하게 접근 가능하다는 점도 특징입니다.



### On Overlap Ratio in Defocused Electron Ptychography (https://arxiv.org/abs/2502.00762)
- **What's New**: 이번 논문에서는 Four-dimensional Scanning Transmission Electron Microscopy (4D STEM) 기술을 활용하여 복잡한 생물학적 샘플과 물질을 분석하는 방법을 제시합니다. 특히, defocused electron probe을 이용한 데이터 수집에서 Electron Ptychography (EP)의 품질이 인접한 조명 영역의 중첩 비율(overlap ratio)에 따라 어떻게 달라지는지를 실험적으로 탐구하였습니다. 40% 이상의 중첩 비율이 고품질 재구성을 위한 안정적인 결과를 보인다는 점을 강조합니다.

- **Technical Details**: 4D STEM에서 discontinuous electron probe를 사용하여 이미지화하는 방식과 그로부터 얻은 데이터가 2D 탐지기에서 기록되는 과정을 상세히 설명합니다. 이 과정에서 두 가지 주요 수량을 정의하고, 중첩 비율에 따라 EP 알고리즘의 성능을 평가합니다. 연구의 일환으로, 시뮬레이션된 4D STEM 데이터 셋을 활용하여 다양한 중첩 비율에서의 EP 알고리즘을 최적화하는 제약된 PIE 알고리즘을 제안합니다.

- **Performance Highlights**: 중첩 비율이 40% 이상일 경우, EP 재구성이 보다 안정적이며 고품질의 결과를 도출할 수 있다는 사실이 실험적으로 입증되었습니다. 또한, EP 복구가 중첩 비율에 따라 달라짐을 확인하고, 제안된 방법을 통해 데이터의 중복성과 EP 재구성의 품질 간의 관계를 명확히 밝혔습니다. 이 연구 결과는 4D STEM의 응용에서 중요한 기초 자료를 제공할 것으로 기대됩니다.



### Zero-Shot Warning Generation for Misinformative Multimodal Conten (https://arxiv.org/abs/2502.00752)
- **What's New**: 본 연구는 잘못된 맥락(misinformation)에서 정보를 추출하고 분석하는 모델을 제안합니다. 이 모델은 cross-modality consistency check를 통해 다양한 데이터 모달리티에서 정보를 검증하며, 훈련 시간이 적게 소요되는 특징이 있습니다. 또한, 자동으로 경고를 생성할 수 있는 새로운 zero-shot learning 작업을 도입하여 사용자 이해를 증진시킬 수 있는 방법을 탐색합니다.

- **Technical Details**: 제안된 모델은 (이미지, 캡션) 쌍의 진위를 평가하는 유연한 아키텍처를 갖추고 있으며, 이를 통해 87.04%의 정확도를 기록했습니다. 증거 수집(evidence retrieval), 일관성 검사(consistency check), 경고 생성(warning generation)이라는 세 단계를 포함한 파이프라인을 통해 정확한 검증을 이루어냅니다. 모델은 Visual Language Model(VLM)인 MiniGPT-4를 활용하여 증거의 홈페이지를 분석하고, 관련 정보를 바탕으로 사용자에게 경고 또는 설명을 제공합니다.

- **Performance Highlights**: 본 연구는 경량화된 모델이 전체 모델 활용 시에는 87.04%의 정확도를, 경량 모델 활용 시에는 84.78%의 정확도를 달성해 경쟁력을 검증했습니다. 질적 평가와 인간 평가를 통해 생성된 경고의 잠재력과 한계를 확인했습니다. 이를 통해 잘못된 정보의 추적 및 반박 과정에서 중요한 이해를 향상시킬 수 있음을 입증합니다.



### Predictive modeling and anomaly detection in large-scale web portals through the CAWAL framework (https://arxiv.org/abs/2502.00413)
Comments:
          15 pages, 4 figures

- **What's New**: 이 연구에서는 CAWAL 프레임워크를 통해 수집된 세션 및 페이지 뷰 데이터를 활용하여 웹 사용 마이닝(Web Usage Mining, WUM) 애플리케이션의 예측 모델링 및 이상 탐지 기능을 향상시키는 새로운 접근 방식을 제시합니다. 전통적인 WUM 방법들은 대개 웹 서버 로그에 의존하여 데이터 다양성과 품질을 제한받는데, CAWAL 프레임워크는 애플리케이션 로그와 웹 분석 데이터를 통합하여 사용자 상호작용에 대한 보다 상세한 뷰를 제공합니다. 이 통합 덕분에 데이터의 다양성과 품질이 향상되고, 기존 WUM의 전처리 단계를 제거하여 처리 효율성이 크게 증가합니다.

- **Technical Details**: CAWAL 프레임워크는 세션 데이터와 페이지 뷰 기록, 사용자 프로필, 상호작용 이력 같은 상세한 데이터를 결합하여 강화된 데이터셋을 만듭니다. 이 강화된 데이터셋은 Gradient Boosting 및 Random Forest와 같은 고급 머신 러닝 모델에 적용되어, 복잡한 패턴을 포착하고 비선형 관계를 모델링하는 데 효과적입니다. 연구 결과, 이러한 모델은 사용자 행동을 92% 이상의 정확도로 예측하고 이상 탐지 능력을 크게 향상시켰습니다.

- **Performance Highlights**: 결과적으로 CAWAL 프레임워크를 사용하여 생성된 데이터셋은 대규모 웹 포털의 효율성, 신뢰성 및 확장성을 개선하는 강력한 해결책으로 자리 잡았습니다. 이 연구는 WUM 프로세스를 가속화하고, 예측 모델의 정확도를 향상시키며, 다중 서버 및 다중 도메인 구조에서도 이상 탐지 프로세스를 최적화하는 데 기여합니다. 따라서 CAWAL은 평가 및 결정 내리기 과정에서 보다 포괄적인 데이터 인프라를 제공합니다.



### MODS: Moderating a Mixture of Document Speakers to Summarize Debatable Queries in Document Collections (https://arxiv.org/abs/2502.00322)
Comments:
          Accepted at NAACL 2025(main)

- **What's New**: 이번 논문에서는 과거의 쿼리 중심 요약(Query-focused summarization, QFS)이 단일 답변을 가정하고, 논란이 있는 질문에 대한 균형 잡힌 요약을 생성하는 데 실패하는 문제를 다루고 있습니다. 저자들은 Debatable QFS (DQFS)라는 새로운 작업을 소개하며, 서로 대립하는 관점을 포함한 문서들로부터 포괄적이고 균형 잡힌 요약을 생성하는 방법을 제시합니다. 이 과제를 해결하기 위해 MODS라는 다중 LLM 프레임워크를 설계하여 사람의 패널 토론을 모방하는 방식을 사용했습니다.

- **Technical Details**: MODS는 각 문서를 개별 Speaker LLM으로 처리하고, Moderator LLM이 맞춤형 쿼리에 응답하도록 해당 Speaker를 선택하는 구조입니다. 이 시스템은 각 주제에 대해 최적화된 쿼리를 사용하여 관련 문서의 컨텍스트를 검색하고, 각 Speaker의 관점을 정리된 아웃라인에 기록하여 최종 요약을 안내합니다. 이렇게 함으로써 MODS는 정보를 고르게 대표하고 다양한 관점을 내포한 요약을 생성하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, MODS는 ConflictingQA 및 새로운 DebateQFS 데이터셋에서 최첨단(SoTA) 모델보다 38-59% 더 높은 주제 문단 커버리지와 균형성을 달성했습니다. 사용자들은 MODS의 요약이 읽기 쉽고 균형 잡혀 있다고 평가했으며, 이는 보다 많은 문서에서 온 관점을 포함하면서도 가독성을 유지했습니다. 이러한 결과는 MODS가 쿼리를 맞춤화하고 아웃라인을 통해 사용자의 요약 경험을 풍부하게 할 수 있음을 시사합니다.



### Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation (https://arxiv.org/abs/2502.00306)
- **What's New**: 본 논문은 Retrieval-Augmented Generation (RAG) 시스템에서의 membership inference 공격 기법인 Interrogation Attack (IA)을 제시합니다. IA는 모델의 성능을 저하시키지 않으면서도, 최소 30개의 자연어 쿼리로 특정 문서의 존재 여부를 추론하는 데 성공합니다. 기존의 기법들과는 달리, IA는 스텔스성(stealthy)를 유지하면서도 높은 정확도와 재현성을 보여줍니다.

- **Technical Details**: RAG 시스템의 구조는 문서 집합과 검색기, 생성 모델로 구성됩니다. RAG는 쿼리에 따라 지식 기반에서 관련 문서를 검색하고 이를 모델의 프롬프트에 포함하여 출력을 생성합니다. IA는 타겟 문서와 밀접하게 연관된 자연어 쿼리를 설계하여, 문서의 존재 여부를 추론하는 방식으로 작동합니다.

- **Performance Highlights**: IA는 기존의 membership inference 공격들과 비교해 2배 향상된 True Positive Rate (TPR@1% FPR)을 기록했으며, 공격자의 쿼리는 덜 포착되고 대략 5%의 탐지율을 자랑합니다. 이는 기존 방법들이 90% 이상 탐지되는 것과 비교됩니다. 또한 이 공격은 RAG 시스템 사용 시 발생할 수 있는 개인정보 유출 문제를 해결하는 데 기여할 것으로 기대됩니다.



### DEUCE: Dual-diversity Enhancement and Uncertainty-awareness for Cold-start Active Learning (https://arxiv.org/abs/2502.00305)
Comments:
          18 pages, 3 figures, 12 tables. Accepted manuscript by TACL. For published version by MIT Press, see this https URL

- **What's New**: 이 논문에서는 Cold-start Active Learning (CSAL)에 대한 새로운 접근법을 제시합니다. 기존 방법들이 약한 클래스와 어려운 대표 예시를 무시했던 문제를 해결하기 위해, Dual-Diversity Enhancing and Uncertainty-aware (DEUCE) 프레임워크를 제안합니다. DEUCE는 사전 훈련된 언어 모델(PLM)을 활용하여 텍스트 표현과 예측 불확실성을 효율적으로 추출합니다.

- **Technical Details**: DEUCE 프레임워크는 Dual-Neighbor Graph (DNG)를 구성하여 텍스트 다양성과 클래스 다양성을 결합합니다. 이는 데이터 분포를 균형 있게 만들고, 밀도 기반 클러스터링을 통해 불확실성 정보를 전파하여 어려운 대표 사례를 선택합니다. 이러한 접근법은 CSAL에서 클래스 균형과 하드 대표 데이터 선택에 효과적입니다.

- **Performance Highlights**: DEUCE는 여섯 개의 NLP 데이터세트에서 실험을 통해 그 우수성과 효율성을 입증했습니다. 이 프레임워크는 탐색과 활용 간의 균형을 잘 이루어, CSAL에서의 데이터 수집 성능을 향상시키는 것을 목표로 합니다. DEUCE는 텍스트와 클래스 다양성을 동시에 고려하여, CSAL의 클래스 불균형 문제를 해결하는데 기여합니다.



### Towards Recommender Systems LLMs Playground (RecSysLLMsP): Exploring Polarization and Engagement in Simulated Social Networks (https://arxiv.org/abs/2502.00055)
Comments:
          8 pages, 2 figures

- **What's New**: 이 논문에서는 인공지능 기술의 급속한 발전과 추천 시스템의 악영향 가능성을 고려하여, 추천 시스템의 효과를 시뮬레이션하고 평가하는 것이 중요하다고 강조합니다. 새로운 프레임워크인 Recommender Systems LLMs Playground (RecSysLLMsP)를 통해, 대형 언어 모델(LLMs)을 활용하여 다양한 콘텐츠 추천 설정이 소셜 네트워크에서 사용자 참여 및 세분화에 미치는 영향을 조사합니다.

- **Technical Details**: RecSysLLMsP는 다채로운 AI 에이전트(AgentPrompts)를 생성하여, 세 가지 시나리오인 Plurality, Balanced, Similarity에서 자율 행동을 평가합니다. Similarity 시나리오에서는 사용자 선호도에 맞춘 콘텐츠가 최대의 참여를 이끌어내지만, 동시에 에코 챔버(echo chamber)를 촉진할 가능성이 있습니다. 반대로 Plurality 시나리오는 다양한 상호작용을 유도하지만, 참여도 결과는 혼합되어 나타납니다.

- **Performance Highlights**: 이 연구는 추천 시스템 디자인에서 사용자 만족을 높이며 사회적 분열을 완화하기 위해 신중한 균형이 필요함을 강조합니다. RecSysLLMsP의 장점은 사회적 영향을 평가하고 다양한 추천 시스템 설정에 대한 사용자 참여 수준을 결정하는 데 필수적인 세분화 효과를 계산할 수 있는 능력에 있습니다. 하지만, 연구의 한계는 현실을 정확하게 모사하는 것이며, 향후 연구는 실제 인간과 AgentPrompts 간의 행동 유사성을 검증하고 세분화 점수를 측정할 수 있는 지표를 수립해야 합니다.



### Querying Databases with Function Calling (https://arxiv.org/abs/2502.00032)
Comments:
          Preprint. 23 pages, 7 figures

- **What's New**: 이번 연구는 Large Language Models (LLMs)와 데이터베이스 쿼리를 통합하는 새로운 도구 정의를 제안합니다. 연구진은 Function Calling을 활용하여 LLM이 데이터에 접근하고, 검색 쿼리 및 필터를 적용해 효과적으로 쿼리를 수행할 수 있도록 하였습니다. 또한, 8개의 LLM을 사용하여 정확한 쿼리 결과를 평가하는 DBGorilla 벤치마크를 소개합니다.

- **Technical Details**: 연구에서는 Gorilla LLM 프레임워크를 기반으로 합성 데이터베이스 스키마와 쿼리를 생성하며, Function Calling을 통해 데이터 쿼리 처리를 효율적으로 할 수 있도록 합니다. 제안된 도구 정의는 검색 쿼리와 구조화된 데이터 접근을 통합하며, SQL 다이얼렉트와 관련된 다양한 쿼리 연산자를 쉽게 결합할 수 있게 합니다.

- **Performance Highlights**: 성능 평가 결과, Claude 3.5 Sonnet이 74.3%의 Exact Match 점수로 가장 높은 성과를 보였습니다. 전반적으로 LLM은 boolean 속성에 대한 연산 활용에 효과적이지만, 텍스트 속성 필터에는 어려움을 겪고 있습니다. 이번 연구는 LLM이 기능 호출(Function Calling)을 통해 데이터베이스를 효과적으로 쿼리할 수 있음을 보여줍니다.



New uploads on arXiv(cs.CV)

### Articulate AnyMesh: Open-Vocabulary 3D Articulated Objects Modeling (https://arxiv.org/abs/2502.02590)
- **What's New**: 최근 3D 조합체 물체 모델링의 도전 과제를 해결하기 위해 Articulate AnyMesh라는 자동화된 프레임워크를 제안합니다. 이 프레임워크는 모든 강체 3D 메쉬를 개방된 어휘(open-vocabulary) 방식으로 조합체 물체로 변환할 수 있습니다. 특히, 선진 Vision-Language Models 및 시각적 프롬프트 기법을 활용하여 물체 부품을 세분화하고 기능적 조인트를 구축하는 데 중점을 두고 있습니다.

- **Technical Details**: Articulate AnyMesh는 Movable Part Segmentation, Articulation Estimation, Refinement의 세 가지 주요 단계로 구성됩니다. Movable Part Segmentation 단계에서는 모든 이동 가능한 부품을 식별하고 그 의미를 결정하며, Articulation Estimation 단계에서는 기하학적 단서를 추출하여 조인트 매개변수를 예측합니다. 마지막으로 Refinement 단계에서는 2D diffusion 모델을 활용하여 메쉬의 구멍을 메우고 기하학과 질감을 향상시킵니다.

- **Performance Highlights**: 실험 결과, Articulate AnyMesh는 PartNet-Mobility 데이터셋에서 훈련된 최첨단 조합체 물체 모델링 방법과 비슷한 정확도의 조인트 매개변수 추정 능력을 보여주며, 더 다양한 조합체 물체를 모델링할 수 있는 능력을 입증했습니다. 이로써 기존의 데이터셋이 포괄하지 못한 범위의 조합체 물체 생성도 가능하게 되었습니다.



### COCONut-PanCap: Joint Panoptic Segmentation and Grounded Captions for Fine-Grained Understanding and Generation (https://arxiv.org/abs/2502.02589)
Comments:
          project website: this https URL

- **What's New**: 이 논문에서는 panoptic segmentation(파노프틱 분할)과 grounded image captioning(그라운디드 이미지 캡셔닝)을 강화하기 위해 COCONut-PanCap 데이터셋을 소개합니다. 기존의 이미지-텍스트 데이터셋이 상세하고 장면을 포괄하는 설명이 부족한 점을 보완하며, COCONut의 진보된 panoptic masks(파노프틱 마스크)를 기반으로 세밀한 지역 수준의 캡션을 포함합니다. 이 데이터셋은 이미지 이해 및 텍스트-이미지 생성 작업을 위한 vision-language models (VLMs)의 훈련을 지원합니다.

- **Technical Details**: COCONut-PanCap 데이터셋은 118K개의 이미지-텍스트 쌍으로 구성되어 있으며, 평균 캡션 길이는 203단어입니다. 이 데이터셋은 세분화된 마스크 주석과 VLMs를 활용한 주석 생성 방식을 통해 고품질의 캡션을 만드는 효율적인 방법론을 제안합니다. 연구는 각 세분화된 영역에 대해 VLM이 생성한 초안을 인간이 수정하여 정교화하는 과정을 포함하며, 이로 인해 사진 마스크와 객체 참조 간의 일관된 관계를 유지합니다.

- **Performance Highlights**: 실험 결과, COCONut-PanCap은 이미지 이해 및 생성 작업에서 성능을 크게 향상시킴을 보여줍니다. 이 데이터셋은 panoptic segmentation과 grounded captioning 작업의 새로운 벤치마크를 제시하여, 고품질의 이미지-텍스트 주석에 대한 요구를 충족합니다. COCONut-PanCap을 통해 다양한 비전-언어 응용 프로그램에서의 모델 성능 향상이 검증되었으며, 전체적인 캡션 생성 및 텍스트-이미지 작업에서 효용성이 입증되었습니다.



### Calibrated Multi-Preference Optimization for Aligning Diffusion Models (https://arxiv.org/abs/2502.02588)
- **What's New**: 이 논문에서는 Calibrated Preference Optimization (CaPO)이라는 새로운 방법론을 제안하여 수동 레이블 없이 다수의 보상 모델을 통해 T2I diffusion 모델을 최적화한다. 기존의 방법들은 pairwise preference distribution에만 의존하며, 다중 선호 사항을 고려하지 못하며 보상 간의 불일치 처리가 부족했다. CaPO는 이러한 문제를 해결하기 위해 보상 신호를 이용한 최적화를 향상시키는 보상 보정 방법을 도입한다.

- **Technical Details**: CaPO의 핵심은 선행 훈련된 모델에 의해 생성된 샘플에 대한 예상 승률을 계산하여 일반 선호를 근사하는 보상 보정 방법이다. 또한, Pareto frontier에서 쌍을 선택하는 경계를 기반으로 한 선택 방법을 제안하여 다중 선호 분포를 효율적으로 관리한다. 최종적으로 회귀 손실을 사용하여 선택된 쌍의 보정된 보상 간 차이를 일치시켜 T2I diffusion 모델을 미세 조정한다.

- **Performance Highlights**: 실험 결과, CaPO는 GenEval 및 T2I-Compbench와 같은 T2I 벤치마크에서 Direct Preference Optimization (DPO)와 같은 기존 방법들보다 일관되게 우수한 성능을 보였다. CaPO는 다양한 보상 신호를 공동 최적화하여 다중 보상 문제에 효과적으로 확장하며, 인간의 선호에 더 잘 부합하는 이미지를 생성하는 데 기여한다.



### Revisiting Expected Possession Value in Football: Introducing a Benchmark, U-Net Architecture, and Reward and Risk for Passes (https://arxiv.org/abs/2502.02565)
- **What's New**: 이 논문에서는 축구에 대한 첫 번째 Expected Possession Value (EPV) 벤치마크와 향상된 EPV 모델을 소개합니다. OJN-Pass-EPV 벤치마크를 통해 EPV 모델의 품질을 정량적으로 평가하는 새로운 방법을 제시하며, 개선된 모델은 공의 높이를 포함하고 보상과 위험을 분석하는 새로운 이중 구성 요소 패스 가치 모델을 특징으로 합니다. 이 결과로 제안된 EPV 모델은 OJN-Pass-EPV 벤치마크의 게임 상태 쌍에서 78%의 정확도로 높은 가치 상태를 식별할 수 있습니다.

- **Technical Details**: 본 연구는 EPV 모델의 성능을 정량적으로 평가하기 위해 OJN-Pass-EPV 벤치마크를 개발했습니다. 이 벤치마크는 50개의 수정된 게임 상태 쌍으로 구성되며, 게임 상태의 특정 요소를 현실적으로 변경하여 패스 모델을 평가하는 데 적합합니다. 또한, U-net 형태의 합성곱 신경망 아키텍처를 도입하여 모델의 적응성과 강건성을 다양한 축구 리그에 걸쳐 평가합니다.

- **Performance Highlights**: 향상된 패스 모델은 패스의 성공 가능성과 패스가 주는 잠재적 보상 및 위험을 나누어 평가합니다. 본 연구에서 사용된 데이터는 KNVB에서 제공받은 2021/22 및 2022/23 시즌의 네덜란드 에레디비지 리그와 2022 월드컵 데이터로, 이 데이터들을 통한 연구는 EPV 모델의 개선 및 축구 분석에 대한 심층적인 이해를 위한 기여를 목표로 합니다.



### Mosaic3D: Foundation Dataset and Model for Open-Vocabulary 3D Segmentation (https://arxiv.org/abs/2502.02548)
Comments:
          project page: this https URL

- **What's New**: 이번 연구에서는 오픈-어휘(open-vocabulary) 3D 장면 이해를 위한 새로운 데이터 생성 파이프라인과 훈련 프레임워크를 제시합니다. Mosaic3D-5.6M이라는 대규모 데이터셋을 통해 고품질의 3D mask-text 쌍을 제공하며, 이는 기존 데이터셋보다 훨씬 방대한 양입니다. 새로운 기초 모델인 Mosaic3D는 3D 인코더와 경량 마스크 디코더를 결합하여 오픈-어휘 3D 의미론 및 인스턴스 분할을 수행합니다.

- **Technical Details**: 우리는 3D 지역의 정밀한 분할(precise segmentation), 포괄적인 텍스트 설명(comprehensive textual descriptions), 대규모 데이터셋 요건을 충족하기 위해 최첨단 오픈-어휘 이미지 분할 모델과 지역 인식 비전-언어 모델을 활용합니다. 이렇게 생성된 Mosaic3D-5.6M 데이터셋은 3D 장면 30,000개 이상과 5.6백만개의 마스크-텍스트 쌍을 포함하고 있습니다. 또한, Sparse ConvNets를 기반으로 한 3D 인코더가 언어와 일치하는 특징을 학습하고, 경량 마스크 디코더가 직접 인스턴스 예측을 수행할 수 있도록 훈련됩니다.

- **Performance Highlights**: 우리의 방법은 ScanNet200, Matterport3D, ScanNet++와 같은 여러 오픈-어휘 3D 의미론 및 인스턴스 분할 작업에서 최첨단 결과를 달성하였습니다. 대규모 훈련 데이터의 효과성은 아블레이션 연구(absalation studies)를 통해 검증되었으며, 데이터 크기와 품질이 성능 개선의 중요한 요소라는 점을 입증했습니다. 이로 인해 우리는 오픈-어휘 3D 장면 이해 분야에서 획기적인 진전을 이루었습니다.



### Uncertainty Quantification for Collaborative Object Detection Under Adversarial Attacks (https://arxiv.org/abs/2502.02537)
- **What's New**: 이번 연구에서는 Trusted Uncertainty Quantification in Collaborative Perception (TUQCP) 프레임워크를 제안하여 기존의 Collaborative Object Detection (COD) 모델의 적대적 공격에 대한 강건성을 향상시키는 새로운 접근 방식을 소개합니다. TUQCP는 적대적 훈련(adversarial training) 및 불확실성 정량화(uncertainty quantification) 기법을 결합하여 객체 탐지 정확성을 증가시키고 출력 불확실성을 줄이는데 중점을 둡니다.

- **Technical Details**: TUQCP는 학습 기반 불확실성 예측(learning-based uncertainty prediction)과 conformal prediction(형식적 예측)을 활용하여 COD 모델의 견고성을 강화합니다. 또한, TUQCP는 선행(collaboration) 및 중간(collaboration) 모델에 모두 적용 가능하며, 공격을 견디기 위한 추가적인 불확실성 손실(unceetainty loss) 항을 도입하여 모델이 공격에 보다 효과적으로 대응할 수 있도록 합니다.

- **Performance Highlights**: TUQCP는 V2X-Sim 데이터셋을 활용하여 평가를 진행하였으며, 동일한 적대적 공격 하에서 기준 모델 대비 80.41%의 객체 탐지 정확도 향상을 나타내었습니다. 이것은 TUQCP가 적대적 공격 및 불확실성 정량화의 중요성을 비추어, COD 모델의 신뢰성을 크게 향상시킬 수 있음을 시사합니다.



### Diff9D: Diffusion-Based Domain-Generalized Category-Level 9-DoF Object Pose Estimation (https://arxiv.org/abs/2502.02525)
Comments:
          17 pages, 13 figures

- **What's New**: 본 논문에서는 9-DoF 객체의 자세(pose) 및 크기(size) 추정을 위한 새로운 접근법인 Diff9D를 제안합니다. 이 방법은 렌더링된 합성 데이터만을 사용하여 훈련되며, 3D 모양 선험(prior)에 의존하지 않고도 실제 세계 장면에 대한 일반화를 가능하게 합니다. 이를 통해 데이터 수집과 주석 작업에 필요한 많은 수고를 덜 수 있습니다.

- **Technical Details**: 제안된 Diff9D 모델은 Denoising Diffusion Probabilistic Model (DDPM)을 기반으로 하여, 객체 자세 추정을 생성적(generative) 관점에서 재정의합니다. 특히, Denoising Diffusion Implicit Model (DDIM)을 활용함으로써, 역 확산(reverse diffusion) 과정을 3단계로 수행하여 거의 실시간 성능을 달성합니다. 이 모델은 ResNet18 및 PointNet 모델을 기반으로 조건(condition) 추출을 수행하여 효과적인 노이즈 제거기가 포함되어 있습니다.

- **Performance Highlights**: 제작된 로봇 손잡이 시스템에 Diff9D 방법을 배포한 결과, 두 개의 널리 사용되는 벤치마크 데이터셋(REAL275 및 Wild6D)과 실제 로봇 손잡이 장면에서 뛰어난 도메인 일반화 성능을 입증했습니다. 이 방법은 17.2 프레임/초(FPS)로 실제 세계의 손잡이 작업에 대해 일반화할 수 있음을 보여주었습니다.



### Privacy Attacks on Image AutoRegressive Models (https://arxiv.org/abs/2502.02514)
Comments:
          Code: this https URL

- **What's New**: 최근 이미지 생성 모델 분야에서, Image Autoregressive (IAR) 모델이 Diffusion Models (DMs)을 이미지 품질과 생성 속도 모두에서 초월했습니다. 그러나 IAR의 프라이버시 위험에 대한 연구는 부족하여, 본 논문에서는 IAR과 DMs간의 포괄적인 프라이버시 분석을 진행합니다. 특히, 새로운 Membership Inference Attack (MIA)을 개발하여 IAR의 트레이닝 이미지 감지 성공률이 DMs 보다 현저히 높다는 것을 보여주고 있습니다.

- **Technical Details**: 이 연구에서는 MIA를 개발하여 특정 데이터 포인트가 IAR의 트레이닝 세트에 포함되었는지 판단할 수 있도록 하였습니다. 연구 결과, IAR 모델은 단 6개의 샘플만으로도 데이터셋 멤버십을 탐지할 수 있는 반면, DMs은 최소 200개의 샘플이 필요합니다. 이러한 결과는 IAR에서 정보 유출이 더 심각함을 나타내며, 또한 여러 모델 간 프라이버시 리크 테스트를 통해 IAR가 DMs보다 훨씬 더 취약하다는 것을 증명합니다.

- **Performance Highlights**: IAR 모델은 생성 효율성과 품질에서 DMs을 초월하지만, 프라이버시 누출 면에서는 여러 배 더 높은 취약점을 보입니다. 본 연구를 통해 IAR 모델의 새로운 MIA는 86.38%의 TPR@FPR=1%에 도달하며, 이는 기존 MIA의 단순 적용보다 최대 69% 향상된 성과입니다. 이와 같은 연구 결과는 IAR 모델의 성능, 효율성 및 프라이버시 간의 중요한 균형을 강조합니다.



### Unified Spatial-Temporal Edge-Enhanced Graph Networks for Pedestrian Trajectory Prediction (https://arxiv.org/abs/2502.02504)
- **What's New**: 이번 연구에서는 보행자 궤적 예측을 위한 새로운 접근 방식인 UniEdge를 제안합니다. UniEdge는 고차원 교차 시간 상호작용을 단순화하여 첫 번째 관계로 통합한 통합된 공간-시간 그래프 데이터 구조를 도입합니다. 이로 인해 다단계 집계 과정에서 정보 손실을 방지하고 즉각적인 반응을 가능하게 만들어 예측 성능을 향상시킵니다.

- **Technical Details**: UniEdge는 Edge-to-Edge-Node-to-Node Graph Convolution(E2E-N2N-GCN)을 기반으로 하는 이중 그래프 네트워크를 통해 보행자 간의 명시적 N2N 사회적 상호작용과 암묵적 E2E 영향 전파를 동시에 모델링합니다. 이러한 구조는 각 보행자의 행동과 집단 역학을 보다 정교하게 분석할 수 있어 밀집된 환경에서의 예측 정확도를 높여 줍니다.

- **Performance Highlights**: 시험 결과 UniEdge는 ETH, UCY, SDD와 같은 표준 데이터셋에서 최신 기법들과 비교하여 뛰어난 성능을 보여주었습니다. 특히 전통적인 그래프 신경망 방식이 가지는 한계를 극복함으로써 객체의 기간 의존성을 글로벌하게 모델링하고 예측 능력을 크게 향상시켰습니다.



### Graph-based Document Structure Analysis (https://arxiv.org/abs/2502.02501)
Comments:
          Accepted by ICLR 2025. Project page: this https URL

- **What's New**: 이 논문은 전통적인 문서 레이아웃 분석(DLA) 방법의 한계를 극복하기 위해 새로운 그래프 기반 문서 구조 분석(gDSA) 작업을 제안합니다. gDSA는 문서 요소를 감지하는 것뿐만 아니라, 공간적 및 논리적 관계를 그래프 구조로 생성하여 문서를 보다 직관적으로 이해할 수 있도록 합니다. 이를 위해 80,000개의 문서 이미지와 413만 개의 관계 주석으로 구성된 GraphDoc 데이터세트를 구축하였습니다.

- **Technical Details**: GraphDoc 데이터세트는 텍스트, 이미지, 표 등과 같은 문서 성분 간의 공간적 관계(위, 아래, 왼쪽, 오른쪽)와 논리적 관계(부모, 자식, 순서, 참조)에 대한 주석을 포함하고 있습니다. 연구진은 문서 레이아웃으로부터 관계 그래프를 생성하는 Document Relation Graph Generator(DRGG)라는 엔드-투-엔드 아키텍처를 제안하여, 문서 요소 간의 관계를 효과적으로 포착합니다. DRGG는 0.5의 관계 신뢰도 임계값에서 평균 정확도(mAP) 57.6%를 기록하여 강력한 기준을 설정했습니다.

- **Performance Highlights**: DRGG 모델은 다중 작업을 처리할 수 있는 기능을 제공하며, 읽기 순서 예측, 계층 구조 분석 및 복잡한 요소 간 관계 추론 등을 수행할 수 있습니다. 이 논문은 그래프 기반 문서 구조 분석이 문서 이해 및 분석에서 혁신적인 진전을 이룰 것이라고 기대하고 있습니다. 새로운 데이터세트와 코드도 공개될 예정입니다.



### VideoJAM: Joint Appearance-Motion Representations for Enhanced Motion Generation in Video Models (https://arxiv.org/abs/2502.02492)
- **What's New**: 비디오 생성 분야에서 최근 일반적인 픽셀 재구성 목표는 모션 일관성을 희생하면서 외관 충실도에 치우치는 경향이 있습니다. 이를 해결하기 위해, VideoJAM이라는 새로운 프레임워크를 도입하여 비디오 생성기에게 효과적인 모션 프라이어(effective motion prior)를 통합했습니다. 이 프레임워크는 두 개의 보완적 유닛으로 구성되어 있으며, 생성 과정에서 모션 예측을 동적 유도 신호로 활용하는 Inner-Guidance 메커니즘을 포함합니다.

- **Technical Details**: VideoJAM 프레임워크는 비디오 생성 모델에 공동 외관-모션 표현(Joint Appearance-Motion representation)을 배우도록 구성되어 있습니다. 훈련 중에는 외관 뿐만 아니라 모션도 예측하는 목표로 수정되며, 추론 과정에서는 Inner-Guidance 메커니즘을 통해 학습된 모션 프라이어를 활용합니다. 이러한 과정은 추가적인 두 개의 선형 레이어를 아키텍처에 추가하여 구현됩니다.

- **Performance Highlights**: VideoJAM은 다양한 모델 크기와 다양한 모션 타입에서 모션 일관성을 크게 향상시키며 최첨단 성능을 기록했습니다. 또한, VideoJAM은 생성된 비디오의 시각적 품질도 개선하였으며, 데이터나 모델 스케일링의 수정 없이 적용 가능합니다. 이러한 연구 결과는 외관과 모션이 상호 배타적이지 않음을 입증하며, 통합될 때 시각적 품질과 비디오 생성의 일관성을 동시에 향상시킬 수 있음을 보여줍니다.



### A Self-Supervised Framework for Improved Generalisability in Ultrasound B-mode Image Segmentation (https://arxiv.org/abs/2502.02489)
Comments:
          12

- **What's New**: 최신 논문에서는 자기 지도 학습(self-supervised learning, SSL) 기반의 접근법을 소개하며, B-mode 초음파 이미지에서 효과적인 세분화를 달성하기 위한 새로운 방법론을 제시합니다. 특히, 관계 대조 손실(Relation Contrastive Loss, RCL)을 도입하여 긍정적 및 부정적 샘플 쌍을 구별하고, 추가로 공간 및 주파수 기반의 증강 전략을 통해 성능을 한층 향상시킵니다. 이러한 방법은 기존의 감독 학습(supervised learning) 방법과 비교해 데이터가 제한된 경우에서도 우수한 성능을 나타내며, 새로운 데이터에 대한 일반화 능력이 뛰어난 것으로 확인되었습니다.

- **Technical Details**: 연구에서는 특히 B-mode 초음파 이미지를 위한 대조적 SSL 접근법을 통해 새로운 관계 대조 손실(RCL)을 적용하여 고유한 특징을 학습하도록 독려합니다. RCL은 학습 가능한 메트릭을 통해 긍정적인 샘플과 부정적인 샘플 쌍을 차별화하여 입체적인 특징을 강조합니다. 또한, 초음파 이미지의 표현 학습을 개선하기 위해 공간 및 주파수 기반의 데이터 증강 전략을 제안하고 있습니다.

- **Performance Highlights**: 연구 결과에 따르면, 제안된 접근법은 세 개의 공공 유방 초음파 데이터셋에서 전통적인 감독 세분화 방법을 크게 초월하는 성능을 보였습니다. 특히, BUSI 데이터셋에서는 20% 및 50%에서 각각 4% 증가, BrEaST 데이터셋에서는 6% 및 9%의 향상, UDIAT 데이터셋에서는 6.4% 및 3.7%의 성능 향상을 기록한 것으로 나타났습니다. 더욱이, UDIAT 데이터셋에서의 일반화 성능에서도 20.6% 및 13.6%의 성능 향상이 보여져, 데이터가 부족한 환경에서도 우수한 성능을 발휘하고 있음을 입증했습니다.



### Hier-EgoPack: Hierarchical Egocentric Video Understanding with Diverse Task Perspectives (https://arxiv.org/abs/2502.02487)
Comments:
          Project webpage at this https URL

- **What's New**: 이번 논문에서는 EgoPack의 발전된 형태인 Hier-EgoPack을 소개합니다. Hier-EgoPack은 다양한 시간적 세분화에 걸쳐 추론할 수 있는 기능을 제공하여 다양한 downstream tasks에 대한 적용성을 확장합니다. 이를 위해, 다단계 추론을 가능하게 하는 새로운 계층 구조와 GNN (Graph Neural Network) 층을 도입하였습니다.

- **Technical Details**: Hier-EgoPack은 입력 비디오의 세분화된 특징과 광범위한 맥락 패턴을 포착하는 계층적 아키텍처를 채택하고 있습니다. 이 아키텍처에서는 다중 세분화 추론 문제를 효과적으로 해결하기 위해 'Temporal Distance Gated Convolution (TDGC)'라는 새로운 GNN 층을 활용합니다. 이를 통해 과거와 미래 맥락을 모두 아우르는 원활한 시간 의존성 추론이 가능합니다.

- **Performance Highlights**: Hier-EgoPack은 Ego4D 벤치마크에 대한 실험을 통해 다양한 작업을 동시에 효과적으로 해결하는 성능을 입증하였습니다. 또한, Mistakes Queries 작업에의 확장을 통해 몇 초에서 몇 분까지의 활동 로컬리제이션을 성공적으로 다루었습니다. 이 연구는 cross-task interaction의 중요성을 강조하며, 최소한의 task-specific overhead로 여러 egocentric vision tasks를 학습할 수 있는 통합 비디오 이해 아키텍처를 제시합니다.



### Mind the Gap: Evaluating Patch Embeddings from General-Purpose and Histopathology Foundation Models for Cell Segmentation and Classification (https://arxiv.org/abs/2502.02471)
- **What's New**: 이번 연구는 컴퓨터 비전에서 기초 모델의 발전이 디지털 조직병리학에서 Cell 분석과 같은 특수 작업에 대한 도메인 특화된 기초 모델의 장점을 충분히 탐구하지 못했음을 지적합니다. 연구팀은 두 가지 모델 카테고리 간의 표현 학습 차이를 분석하였고, 여러 종류의 인코더를 활용하여 Cell instance segmentation과 분류의 성능을 향상시키기 위해 노력했습니다. 이 과정에서 다양한 인코더의 성능을 비교하였으며, 특히 최근에 출시된 여러 조직병리학 관련 기초 모델의 효용을 검토하였습니다.

- **Technical Details**: 연구에서는 CISCA 프레임워크를 활용하여 Cell segmentation과 classification을 수행하며, 이는 다중 작업 접근법으로 3 클래스 픽셀 분류와 거리 지도 회귀, Cell 유형 분류를 통합합니다. 인코더-디코더 아키텍처를 적용하여, 세 가지 변화를 통해 세그멘테이션 맵과 거리 지도를 생성합니다. 다양한 인코더를 사용하여 모델의 일반화를 평가하며, 전처리가 없는 입력 패치로 여러 히스토리 데이터셋에서 실험을 진행했습니다.

- **Performance Highlights**: 연구에서는 PanNuke, CoNIC 및 CytoDArk0 데이터셋을 통해 인스턴스 수준의 탐지, 세그먼트 정확도 및 Cell 유형 분류에서의 성능 차이를 평가하였습니다. 일반 모델과 전용 모델 간의 성능 차이는 예기치 않은 방향으로 나타났으며, 이는 차별화된 모델 선택과 Cell 중심 조직병리학 분석에 대한 통찰력을 제공합니다. 결과적으로, 연구는 기초 모델 선택에서의 가이드를 제공하며, 세포 중심의 조직병리학 및 뇌 세포 구조 분석 흐름에서의 모델 개발을 돕습니다.



### High-Fidelity Human Avatars from Laptop Webcams using Edge Compu (https://arxiv.org/abs/2502.02468)
Comments:
          6 pages, 6 figures, 1 table

- **What's New**: 새로운 연구는 스마트폰의 RGB와 IR 센서를 활용해 저해상도 웹캠으로부터 고충실도 애니메이션 아바타를 자동 생성할 수 있는 방법을 개발했습니다. 기존에 필요했던 고급 카메라 장비와 서버 처리에서 벗어나, 소비자 급의 하드웨어에서 작동할 수 있습니다. 이 방법은 사용자 이미지에서 우호적으로 조명을 조정하여 다양한 환경에서도 적용 가능한 아바타 생성을 목표로 하고 있습니다.

- **Technical Details**: 제안된 방법은 3D Morphable Models(3DMMs), 랜드마크 탐지, GANs, 그리고 미분 가능 렌더링을 기반으로 구성됩니다. 첫 번째 단계는 3DMM 형태 매개변수를 맞추는 것이고, 두 번째는 텍스처 맵 생성을 포함합니다. 이 파이프라인은 저해상도 또는 노이즈가 있는 웹캠 이미지를 포착한 경우에도 효율적으로 작동하도록 설계되었습니다.

- **Performance Highlights**: 결과적으로 개발된 시스템은 개인의 여러 각도에서 촬영된 얼굴 이미지를 바탕으로 사실감 넘치는 아바타를 생성하는 데 소요되는 시간이 몇 분에 불과하며, 낮은 품질의 이미지를 기반으로 하지만 최종적으로는 애니메이션이 가능한 아바타를 제공합니다. 이 방식은 비디오 회의와 같은 다양한 애플리케이션에서 사용자들이 프라이버시를 유지할 수 있도록 돕습니다.



### Towards Consistent and Controllable Image Synthesis for Face Editing (https://arxiv.org/abs/2502.02465)
- **What's New**: 최근 얼굴 편집 기술은 GAN 기반 기법에서 Diffusion 기반 모델로 초점이 이동하고 있습니다. 이 논문에서는 Stable-Diffusion 모델과 3D 얼굴 모델을 활용하여 조명, 표정 및 머리 자세를 제어하는 새로운 접근 방식을 제안합니다. 특히, 이 모델은 RigFace라는 이름을 가지고 있으며, 이를 통해 다양한 얼굴 속성의 조화를 이루면서 일관성을 유지하며 고품질의 편집을 가능하게 합니다.

- **Technical Details**: RigFace 모델은 세 가지 구성 요소로 이루어져 있습니다: 1) Spatial Attribute Encoder는 조명, 머리 자세, 표정 및 배경의 조건을 각각 정밀하게 분리하여 제공합니다. 2) Identity Encoder는 Stable-Diffusion 모델의 얼굴 특성을 추출하여, 그 정보를 UNet 구조의 노이즈 제거 과정에 이전합니다. 3) Attribute Rigger는 식별 조건을 통합하여 배경 일관성과 조명, 표정 정보를 결합하는 기능을 수행합니다.

- **Performance Highlights**: RigFace는 기존의 GAN 기반 및 Diffusion 기반 얼굴 편집 모델들과 비교하여, 신원 보존(identity preservation) 및 포토리얼리즘(photorealism) 면에서 우수한 성능을 보여줍니다. 특히, 이 모델은 효과적인 조건 분리를 바탕으로 조명과 표정, 머리 자세의 일관성을 유지하면서 다양한 얼굴 속성을 조절할 수 있는 능력을 갖추고 있습니다. 따라서 RigFace는 얼굴 이미지 편집의 품질과 편리함을 향상시키는 데 기여할 것으로 기대됩니다.



### IMDPrompter: Adapting SAM to Image Manipulation Detection by Cross-View Automated Prompt Learning (https://arxiv.org/abs/2502.02454)
- **What's New**: 이 논문에서는 Segment Anything Model(SAM)을 기반으로 하는 새로운 크로스 뷰 프롬프트 학습 패러다임인 IMDPrompter를 제안합니다. IMDPrompter는 수동 프롬프트에 대한 의존성을 없애고 자동화된 탐지 및 위치 지정을 가능하게 합니다. 이를 통해 SAM의 이미지 조작 탐지 기능을 활성화하는 데 기여합니다.

- **Technical Details**: IMDPrompter는 크로스 뷰 인식 학습을 위해 Cross-view Feature Perception, Optimal Prompt Selection, Cross-View Prompt Consistency와 같은 다양한 모듈을 포함합니다. 이들 모듈은 자동화된 프롬프트 생성을 통해 SAM의 성능을 높이고, 여러 뷰를 통합함으로써 맞춤 프롬프트를 생성합니다. 또한, Attention 기반의 CFP 모듈과 다층 퍼셉트론 기반의 PMM 모듈을 통해 크로스 뷰 정보의 융합을 달성합니다.

- **Performance Highlights**: 다섯 개의 데이터셋(CASIA, Columbia, Coverage, IMD2020, NIST16)을 통해 광범위한 실험 결과를 제시하며 IMDPrompter의 이미징 조작 탐지 및 위치 지정 능력을 검증하였습니다. 특히, IMDPrompter는 기존의 탐지기들에 비해 더 우수한 범위 내 및 범위 외 성능을 보여주어, 향후 이미지 조작 탐지 연구에 중요한 진전을 이루었습니다.



### Personalization Toolkit: Training Free Personalization of Large Vision Language Models (https://arxiv.org/abs/2502.02452)
- **What's New**: 이 논문에서는 시간 소모적인 사용자 맞춤 모델 학습 없이 LVLM(대형 비전 언어 모델)의 개인화를 가능하게 하는 새로운 방법을 제안합니다. 기존의 기술이 특정 사용자 및 객체에 대한 훈련에 의존하는 반면, 이 방법은 사전 훈련된 비전 기초 모델을 활용하여 독특한 피처를 추출하고 Retrieval-Augmented Generation (RAG) 기법으로 시각적 인식을 수행합니다. 또한, 시각적 프롬프트를 통해 각 사용자 맞춤의 응답을 생성하는 방식으로 효율성을 극대화합니다.

- **Technical Details**: 제안된 접근 방식은 세 가지 단계로 구성된 개인화 도구 키트인 PeKit을 활용합니다. 첫 번째 단계는 뷰 추출(View Extraction)로, 참조 이미지에서 객체-level 특징을 추출하고 메모리 모듈에 저장합니다. 두 번째 단계인 개인화된 객체 검색(Personalized Objects Retrieval)에서는 쿼리 이미지에서 객체를 인식하고, 마지막으로 개인화된 답변 생성(Personalized Answer Generation) 단계에서는 시각적 프롬프트를 통해 사용자 맞춤 응답을 생성합니다.

- **Performance Highlights**: 이 연구는 LVLM 개인화에 있어 다수의 벤치마크에서 최첨단 결과를 달성하며 기존 접근 방식보다 뛰어난 성능을 보여줍니다. 또한, 현실적인 개인화 시나리오를 반영한 도전적인 평가 세트를 추가로 구성하여 문제의 난이도를 강조하였습니다. 이로써, 개인화 작업의 한계를 극복하기 위한 연구의 기회를 제시합니다.



### TUMTraffic-VideoQA: A Benchmark for Unified Spatio-Temporal Video Understanding in Traffic Scenes (https://arxiv.org/abs/2502.02449)
- **What's New**: 본 논문에서는 복잡한 도로 교통 시나리오를 위한 새로운 데이터셋 TUMTraffic-VideoQA를 제안합니다. 이 데이터셋은 1,000개의 비디오와 85,000개의 다중 선택 QA 쌍, 2,300개의 객체 캡션 및 5,700개의 객체 위치 지정 주석을 포함하고 있습니다. TUMTraffic-VideoQA는 다중 작업을 통합하여 평가 프레임워크 내에서 비디오 질문 응답, 참조 객체 캡션, 시공간 객체 위치 지정과 같은 세 가지 필수 작업을 통합하고 있습니다.

- **Technical Details**: TUMTraffic-VideoQA 데이터셋은 복잡한 교통 비디오 이해를 위해 설계된 포괄적인 비디오 언어 데이터셋입니다. 이 데이터셋은 극단적인 기상 조건과 교통사고와 같은 중요한 사례들을 포함한 다양한 현실 세계 시나리오를 포착합니다. 또한, TUMTraffic-Qwen 기준 모델을 제안하고, 시각적 토큰 샘플링 전략을 통해 세부적인 시공간 추론의 도전 과제를 제공하며, 에 대한 분석을 지원합니다.

- **Performance Highlights**: 광범위한 실험을 통해 TUMTraffic-VideoQA의 복잡성을 입증하고 기존 모델의 한계를 강조하였습니다. 이 데이터셋은 지능형 교통 시스템 연구를 진전시키기 위한 강력한 기반으로 자리 잡고 있으며, 공개적으로 데이터셋과 벤치 마크를 제공하여 향후 연구를 촉진할 수 있도록 하였습니다.



### Extending SEEDS to a Supervoxel Algorithm for Medical Image Analysis (https://arxiv.org/abs/2502.02409)
Comments:
          Tech report

- **What's New**: 이번 연구는 SEEDS 슈퍼픽셀 알고리즘을 2D 이미지에서 3D 볼륨으로 확장하여 3D SEEDS라는 실시간의 높은 성능을 가진 오픈 소스 슈퍼복셀(supervoxel) 알고리즘을 개발하는 데 초점을 맞추고 있습니다. 3D SEEDS는 기존의 SLIC 알고리즘보다 10배 빠른 속도로 슈퍼복셀 생성을 가능하게 하고, Dice 점수를 6.5% 향상시키며, 언더 세그멘테이션 오류를 0.16% 감소시킵니다. 이 알고리즘은 의학 이미지 분석을 위한 최신 기술로 주목받고 있습니다.

- **Technical Details**: SEEDS 알고리즘은 픽셀 수준 및 블록 수준의 업데이트 방식을 사용하여 정규 그리드에서 슈퍼픽셀을 재귀적으로 추출합니다. 3D SEEDS 확장 과정에서 각 슈퍼픽셀의 경계를 조정하는 데 필요한 여러 조건이 있으며, 제안된 알고리즘은 OpenCV에 기반하여 C++ 및 Python과 호환됩니다. 경계의 픽셀을 이웃 슈퍼픽셀로 전이하는 과정에서 16가지의 특별한 경우를 피하는 것이 매우 중요하며, 이는 3D 공간의 복잡한 기하학적 특성으로 인한 도전으로 간주됩니다.

- **Performance Highlights**: 3D SEEDS는 BraTS 및 BTCV 벤치마크에서 10개 기관의 13개 세그멘테이션 작업을 평가하여 뛰어난 성능을 발휘합니다. 이 연구는 세그멘테이션의 성능을 측정하기 위한 새로운 메트릭인 achievable Dice score을 제안하며, 이를 통해 3D SEEDS는 기존의 SLIC 알고리즘과 비교했을 때 더 높은 오버 세그멘테이션 성능을 달성합니다. 생성 속도 또한 NIfTI 파일의 읽기 속도와 유사하여 실제 사용에 적합한 해결책으로 부각됩니다.



### LV-XAttn: Distributed Cross-Attention for Long Visual Inputs in Multimodal Large Language Models (https://arxiv.org/abs/2502.02406)
- **What's New**: 이번 논문에서 제안한 LV-XAttn는 대규모 시각적 입력을 효율적으로 처리하기 위한 새로운 분산 크로스 어텐션 메커니즘이다. 기존의 접근 방식들과는 달리, LV-XAttn는 작은 쿼리 블록을 GPU 간에 전달하고 큰 키-값 블록은 각 GPU에 로컬로 저장하는 구조를 취하고 있다. 또한, 길어진 시각적 컨텍스트를 지원하기 위한 활성화 재계산 기법도 도입하였다. 이로 인해 LV-XAttn는 기존 방법들보다 기억 용량의 요구를 줄이고 통신 오버헤드를 최소화했다.

- **Technical Details**: LV-XAttn의 기술적 세부 사항에는 시퀀스 병렬 처리와 최소 통신 오버헤드를 활용한 정확한 크로스 어텐션 메커니즘이 포함된다. 대규모 시각적 입력을 다루기 위해, LV-XAttn는 각 작업자(worker)에서 큰 키와 값 블록을 로컬로 저장하고, 작은 쿼리 블록을 서로 전송하여 어텐션 출력을 계산한다. 또한, 활성화 재계산을 통해 메모리를 절약하고 긴 시각 정보 입력을 처리할 수 있는 능력을 제공한다.

- **Performance Highlights**: LV-XAttn는 mPLUG-Owl3와 OpenFlamingo 모델 평가에서 최대 45.85배의 크로스 어텐션 속도 향상과 전체 모델 반복 시간에서 최대 5.58배의 향상을 달성하였다. 통신 볼륨을 최소화하고 계산과 통신을 효과적으로 중첩함으로써, LV-XAttn는 기존 접근 방식에 비해 0.42% 이하의 오버헤드를 유지하며 효율성을 크게 향상시켰다.



### MaintaAvatar: A Maintainable Avatar Based on Neural Radiance Fields by Continual Learning (https://arxiv.org/abs/2502.02372)
Comments:
          AAAI 2025. 9 pages

- **What's New**: 이번 연구는 Neural Radiance Fields(NeRF)를 기반으로 한 유지 가능한 아바타(maintainable avatar) 생성을 제안합니다. 기존 연구는 훈련 데이터의 인물 이미지를 고정된 것으로 가정했지만, 우리는 지속적인 학습을 통해 인물의 변화하는 외모와 자세를 올바르게 모델링할 수 있는 방법을 찾았습니다. 우리는 Global-Local Joint Storage Module과 Pose Distillation Module을 활용하여 과거 외모의 렌더링 질감을 유지할 수 있도록 설계했습니다.

- **Technical Details**: 우리가 제안하는 MaintaAvatar는 Global-Local Joint Storage Module을 통해 다양한 외모의 글로벌 및 로컬 정보를 독립적으로 저장하고, Pose Distillation Module을 통해 과거 작업에서의 자세 정보를 추출하여 새로운 작업의 지도 신호로 사용합니다. 이러한 접근 방식은 기존의 static human-nerf 방법과는 달리, 적은 양의 데이터로도 모델을 신속히 조정하고 재학습 시 발생할 수 있는 잃어버림(catastrphic forgetting)을 방지할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, MaintaAvatar 모델은 두 개의 데이터셋에서 뛰어난 성능을 보여주었습니다. 이 모델은 제한된 데이터 수집으로도 고품질의 렌더링을 유지하면서도 빠르게 새로운 외모에 적응할 수 있는 장점을 가지고 있습니다. 이를 통해 우리는 유지 가능한 가상 아바타의 생성에 대한 새로운 패러다임을 제시하였습니다.



### MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm (https://arxiv.org/abs/2502.02358)
- **What's New**: 이번 연구의 핵심은 Motion-Condition-Motion이라는 새로운 패러다임을 제안하여 인간 동작 생성 및 편집 작업을 통합하는 것입니다. 이 패러다임은 소스 동작, 조건, 대상 동작의 세 가지 개념을 바탕으로 하며, 사용자들이 다양한 작업을 위한 모델을 따로 훈련할 필요 없이 효율성을 높일 수 있습니다. 이러한 통합된 프레임워크는 특정 작업에 국한되지 않고 데이터 부족 문제를 해결할 수 있는 가능성을 제공합니다.

- **Technical Details**: 제안된 MotionLab 프레임워크는 MotionFlow Transformer를 활용하여 조건부 생성 및 편집을 효율적으로 수행합니다. 이 과정에서 Aligned Rotational Position Encoding (ROPE)을 통해 소스 동작과 대상 동작 간의 시간 동기화를 보장합니다. 또한, Task Instruction Modulation을 도입하여 다양한 작업을 구분하고, Motion Curriculum Learning을 통해 각 작업의 난이도에 따라 단계적으로 학습하도록 설계되었습니다.

- **Performance Highlights**: MotionLab 프레임워크는 여러 벤치마크에서 뛰어난 일반화 능력과 추론 효율성을 입증했습니다. 또한, 다양한 데이터 세트를 통합하여 모델 성능을 향상시킬 수 있는 잠재력이 있으며, 특히 데이터가 부족한 환경에서의 효율성을 기대할 수 있습니다. 이 연구는 인간 동작 생성과 편집의 통합적 접근법을 통해 다양한 컴퓨터 그래픽스 및 비전 애플리케이션에서의 활용 가능성을 넓힙니다.



### Transfer Risk Map: Mitigating Pixel-level Negative Transfer in Medical Segmentation (https://arxiv.org/abs/2502.02340)
- **What's New**: 이 연구에서는 의료 영상 세분화에 있어서 부정적 전이(negative transfer)를 완화하는 새로운 방법을 제시합니다. 기존 방법들은 주로 분류(classification) 및 회귀(regression) 작업에 집중해왔으나, 본 연구는 이미지의 특정 영역에 따라 전이 위험을 고려합니다. 이를 통해, 세분화 작업에서 각 픽셀의 전이 난이도를 정량화할 수 있는 전이 위험 맵을 도입했습니다.

- **Technical Details**: 제안된 방법은 pixel-level transfer risk map을 통해 각 픽셀의 전이 난이도 및 부정적 전이와 관련된 잠재적 위험을 정량화합니다. 또한, fine-tuning 단계에서 클래스 불균형을 다루기 위해 이미지 전경 크기로 정규화된 가중치 손실 함수(map-weighted loss function)를 사용합니다. 이 방식은 주로 LEEP(Log Expected Empirical Prediction) 지표를 전이 가능성(metric)으로 채택하여 높은 계산 효율성을 보장합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 방법이 brain tumor segmentation 데이터셋 FeTS 2021에서 4.37%, brain matter segmentation 데이터셋 iSeg-2019에서 1.81%의 성능 향상을 보였습니다. 또한, few-shot 시나리오에서도 평균 2.9% 향상된 결과를 기록하여 이 방법의 강건성을 입증했습니다. 이러한 성과는 다양한 작업과 모달리티 간의 부정적 전이를 성공적으로 피하면서 유익한 지식을 학습할 수 있음을 보여줍니다.



### Geometric Neural Process Fields (https://arxiv.org/abs/2502.02338)
- **What's New**: 이 논문에서는 Neural Field (NeF) 일반화의 문제를 다루며, G-NPF(Geometric Neural Process Fields)라는 새로운 확률적인 프레임워크를 제안합니다. 이 프레임워크는 적은 관측치로부터 새로운 신호에 효율적으로 적응할 수 있도록 설계되었습니다. 특히, NeF 기능 분포의 직접적인 추론을 가능하게 하여 불확실성을 명확히 캡처합니다.

- **Technical Details**: G-NPF는 구조적 귀납적 편향을 포함하기 위해 기하학적 기초를 도입합니다. 이러한 기초는 공간 구조를 인코딩하고 NeF 함수 분포의 추론을 촉진합니다. 또한, 계층적 잠재 변수 모델을 설계하여 여러 공간적 수준에서 구조적 정보를 통합하며 INR(Implicit Neural Representation) 함수의 매개변수를 효과적으로 조정합니다.

- **Performance Highlights**: 실험을 통해 3D 장면의 새로운 뷰 합성, 2D 이미지 및 1D 신호 회귀 작업에서 제안된 방법의 효과성을 입증하였습니다. G-NPF는 불확실성을 포착하고 구조적 정보를 활용하여 새로운 장면과 신호에 대한 일반화를 개선하는 데 기여하였습니다.



### Event-aided Semantic Scene Completion (https://arxiv.org/abs/2502.02334)
Comments:
          The established datasets and codebase will be made publicly at this https URL

- **What's New**: 최근 자율 주행 시스템에서는 견고한 3D 씬 이해가 필수적입니다. 이 논문은 이벤트 카메라를 사용하여 RGB 기반 접근 방식의 한계를 극복하는 새로운 데이터 세트인 DSEC-SSC를 제안합니다. 이는 동적인 물체 운동에 따라 적응하는 밀집 가시성 인식을 위한 혁신적인 4D 라벨링 파이프라인을 포함하고 있습니다.

- **Technical Details**: EvSSC는 이벤트 데이터를 통합하여 3D 점유 예측을 개선하는 이벤트 지원 프레임워크입니다. 핵심 구성 요소인 Event-aided Lifting Module (ELM)은 2D 특징을 3D 공간으로 변환하는 중요한 과정을 수행합니다. 이 프레임워크에서는 다양한 융합 패러다임을 탐색하여 공간 충실도를 유지하며 시간적 역학을 포착하는 융합 기반 리프팅을 적용합니다.

- **Performance Highlights**: DSEC-SSC 및 시뮬레이션된 SemanticKITTI-E 데이터 세트에서 EvSSC는 기존 모델보다 우수한 성능을 나타내며, 과거 상태 및 다양한 오염 시나리오에서 최대 52.5%의 상대적 개선을 달성합니다. 이는 자율 주행 중 발생할 수 있는 다양한 도전 과제들에 대한 내성을 검증하는 데 기여합니다.



### Improving Generalization Ability for 3D Object Detection by Learning Sparsity-invariant Features (https://arxiv.org/abs/2502.02322)
Comments:
          Accepted to ICRA 2025. Code is available at this https URL

- **What's New**: 이 논문에서는 자율 주행에서의 3D 물체 탐지를 위한 일반화 능력을 향상시키기 위한 새로운 방법을 제안합니다. 저자들은 단일 소스 도메인에서 학습한 모델이 다른 센서 구성과 장면 분포를 가진 타겟 도메인에 적용될 때 겪는 성능 저하 문제를 해결하고자 합니다. 그들의 접근법은 특정 밀도의 포인트 클라우드를 선정적으로 다운샘플링하고, 특징 정렬 방법을 사용하여 도메인 불변의 표현 학습을 수행합니다.

- **Technical Details**: 제안된 방법은 학생-교사 프레임워크를 기반으로 다양한 밀도를 가진 포인트 클라우드의 Bird's Eye View (BEV) 특징을 정렬합니다. 또한, graph-based embedding relationship alignment (GERA)와 feature content alignment (FCA) 기법을 사용하여 도메인 간 차이를 고려하지 않고도 모델이 학습할 수 있도록 합니다. 이 과정에서 소스 데이터의 밀도를 조정하고, 중요한 밀도를 결정하는 신뢰 점수를 활용하여 작업의 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다른 기준선들과 비교하여 우수한 일반화 능력을 보여주었습니다. 특히, 타겟 도메인 데이터에 접근하지 못하더라도 제안된 방법이 Unsupervised Domain Adaptation (UDA) 방법들과 유사한 성능을 발휘하는 것으로 나타났습니다. 이로 인해, 단일 소스 도메인에서 학습한 모델이 여전히 실세계 적용 가능성을 높일 수 있음을 입증하였습니다.



### Review of Demographic Bias in Face Recognition (https://arxiv.org/abs/2502.02309)
Comments:
          under review

- **What's New**: 최근 얼굴 인식(Face Recognition, FR) 기술에서 인구 통계적 편향(demographic bias)의 문제가 심각하게 대두되고 있습니다. 이 연구는 인구 통계 그룹 간의 성능 차이가 공정성(fairness) 및 신뢰성(reliability)에 미치는 영향을 면밀하게 검토하며, 이러한 문제의 주요 원인 및 해결법을 제시하고 있습니다. 특히, 알고리즘의 민감도(sensitivity), 데이터셋의 구성, 및 평가 지표에 대한 포괄적인 분석이 포함됩니다.

- **Technical Details**: 본 논문에서는 얼굴 인식에서의 인구 통계적 편향의 주요 원인으로 데이터셋 불균형, 피부색의 다양성, 알고리즘 민감도 및 이미지 품질(image quality) 등을 분류하고 있습니다. 이와 함께, 데이터셋의 특성과 인구 통계적 특성 간의 복잡한 상호작용을 분석하여 인식 정확도에 미치는 영향을 다룹니다. 또한, 다양한 평가 메트릭(metrics)과 최근의 편향 완화(mitigation) 전략을 설명하고 있습니다.

- **Performance Highlights**: 이 리뷰는 얼굴 인식 분야의 최신 경향과 도전 과제를 통합적으로 논의합니다. 연구에 따르면, 특정 인구 통계 그룹에서의 잘못된 인식 비율이 상대적으로 높아지며, 이는 알고리즘의 성능에도 큰 영향을 미친다고 합니다. 종합적으로, 이러한 문제는 향후 연구와 혁신을 위한 중요한 기회로서, 공정하고 신뢰할 수 있는 FR 시스템 개발의 필요성이 강조됩니다.



### UniGaze: Towards Universal Gaze Estimation via Large-scale Pre-Training (https://arxiv.org/abs/2502.02307)
- **What's New**: 이 논문에서는 gaze estimation(눈치 방향 추정) 문제를 해결하기 위해 새로운 모델 UniGaze를 제안합니다. UniGaze는 self-supervised pre-training(자기 지도 학습을 통한 사전 학습)을 활용하여 다양한 얼굴 이미지를 기반으로 gaze estimation의 일반화 성능을 향상시킵니다. 기존의 gaze estimation 모델들이 새로운 환경에서 성능 저하 문제를 겪고 있는 반면, UniGaze는 여러 데이터 도메인에서의 일반화 성능을 크게 개선하는 특징이 있습니다.

- **Technical Details**: UniGaze는 Masked Autoencoder (MAE)와 Vision Transformer (ViT) 아키텍처를 결합하여 설계되었습니다. 이 모델은 normalized face images(정규화된 얼굴 이미지)에 대해 MAE를 직접 적용하여 유의미한 feature representations(특징 표현)을 학습합니다. 제안된 모델은 다양한 실세계 및 합성 얼굴 데이터를 포함하여 약 1.6 백만 개의 얼굴 이미지로 구성된 데이터셋에서 학습됩니다.

- **Performance Highlights**: UniGaze는 leave-one-dataset-out 및 joint-dataset 평가 프로토콜을 통해 다양한 환경에서 기존 방법들보다 우수한 성능을 기록했습니다. 실험 결과, 여러 데이터 도메인에서의 일반화 성능이 향상되었으며, gaze estimation을 위한 필수적인 특징 학습이 가능하다는 것을 확인했습니다. 이로 인해 UniGaze는 gaze estimation의 새로운 기준을 제시하게 되었습니다.



### GP-GS: Gaussian Processes for Enhanced Gaussian Splatting (https://arxiv.org/abs/2502.02283)
Comments:
          14 pages,11 figures

- **What's New**: 이 논문에서는 3D Gaussian Splatting(3DGS)의 격차를 메우기 위해 Gaussian Processes Gaussian Splatting(GP-GS)라는 새로운 프레임워크를 제안합니다. GP-GS는_sparse Structure-from-Motion (SfM) 포인트 클라우드의 밀도를 높이고 불확실성을 기반으로 한 적응형 정보를 제공하여 3D 장면 재구성을 개선합니다. 이 접근 방식은 새로운 후보 지점을 추론하고, 고불확실성 예측을 제거함으로써 고품질 3D Gaussians를 생성하여 렌더링 성능을 극대화합니다.

- **Technical Details**: 본 연구에서는 Multi-Output Gaussian Process (MOGP) 모델을 활용하여 2D 이미지 픽셀과 깊이 정보로부터 3D 포인트 클라우드의 위치와 색상을 예측합니다. 특히, 픽셀을 샘플링하여 후보 MOGP 입력을 생성하고, 각 샘플에 대해 MOGP를 사용해 3D 속성을 예측한 후, 불확실성 기반 필터링을 통해 고불확실성 예측을 제거합니다. 이렇게 함으로써, 더 안정적이고 구조화된 포인트 클라우드 덴시피케이션 프로세스를 구현했습니다.

- **Performance Highlights**: 다양한 합성 및 실제 세계 데이터셋에서 수행된 실험을 통해, GP-GS는 기존 SfM 기반 파이프라인에 쉽게 통합될 수 있으며, 렌더링 품질을 향상시키는 유연한 모듈로 작용합니다. 추가적으로, GP-GS는 복잡한 지역에서의 3D Gaussian 초기화를 효과적으로 개선하여, 특히 조명이 어려운 조건에서도 우수한 성능을 보여줍니다. 이러한 연구 결과는 NVS(Novel View Synthesis) 모델의 품질 향상에 기여할 것으로 기대됩니다.



### Survey of Quantization Techniques for On-Device Vision-based Crack Detection (https://arxiv.org/abs/2502.02269)
Comments:
          Accepted by IEEE International Instrumentation and Measurement Technology Conference (I2MTC) 2025

- **What's New**: 이 연구는 UAV와 함께 사용하는 비전 기반 균열 탐지 시스템을 위한 경량의 딥러닝 모델인 MobileNetV1x0.25 및 MobileNetV2x0.5를 평가합니다. 다양한 양자화 기법을 사용하여 TensorFlow, PyTorch 및 ONNX 플랫폼에서 모델의 성능을 비교 분석했습니다. 특히 동적 양자화, 훈련 후 양자화(PTQ), 양자화 인식 훈련(QAT)에 대한 성능 차이를 체계적으로 분석하여 경량하고 효율적인 해결책을 제시했습니다.

- **Technical Details**: 선택된 CNN 모델과 양자화 기법은 두 가지 경량 모바일넷 모델을 기반으로 하며, TensorFlow와 PyTorch 프레임워크를 결합하여 모델을 훈련했습니다. 각 모델은 ImageNet-1K 데이터셋에서 미리 훈련된 후, SDNET2018 데이터셋을 사용하여 균열 분류를 위해 전이 학습과 미세 조정을 적용했습니다. 모든 훈련 과정은 Windows 10 환경에서 Intel® Core i9-12900 CPU 및 NVIDIA RTX 3090 GPU를 통해 수행되었습니다.

- **Performance Highlights**: 결과적으로, QAT를 통해 MBNV2x0.5는 Torch-QAT에서 0.8376의 F1 점수를 달성하여 부동 소수점 정밀도에 근접한 정확성을 유지했습니다. PTQ는 대부분의 경우 메모리 및 에너지 소비를 줄였지만, 특히 TensorFlow에서는 정확도가 손실되는 경향이 있었습니다. 본 연구는 UAV기반의 자율 균열 탐지 시스템을 위한 실제적인 TinyML 최적화 작업 흐름을 제안하여 효율적인 배포 과정을 지원합니다.



### UNIP: Rethinking Pre-trained Attention Patterns for Infrared Semantic Segmentation (https://arxiv.org/abs/2502.02257)
Comments:
          ICLR 2025. 27 pages, 13 figures, 21 tables

- **What's New**: 이 연구에서는 사전 훈련(pre-training) 기술이 한정된 훈련 데이터에서의 의미 분할(semantic segmentation) 작업 성능을 크게 향상시킴을 보여줍니다. 특히, RGB 데이터와는 다른 적외선(infrared) 데이터에 대한 사전 훈련 방법의 효과를 비교 분석하여, 세 가지 주요 주의(attention) 패턴인 지역(local), 혼합(hybrid), 글로벌(global) 패턴을 발견하였습니다. 이 연구는 UNIP라는 통합된 적외선 사전 훈련(framework)을 제안하여, 이론적 통찰에 기반한 구체적인 개발을 통해 모델 성능을 개선하였습니다.

- **Technical Details**: 적외선 이미지는 도로 감시, 자율 주행 및 드론과 같은 다양한 분야에서 사용되지만, 객체 탐지 및 의미 분할 작업에 대한 레이블이 부족하여 강력한 사전 훈련된 백본이 중요합니다. 연구팀은 다양한 모델 크기와 평가 기준을 사용하여 6가지의 감독(supervised) 및 자가 감독(self-supervised) 사전 훈련 방법을 평가하였습니다. 연구 결과로, 감독 및 CL 방법은 작은 모델에서 MIM 방법보다 일반화 성능이 우수하며, 하이브리드 주의 패턴이 의미 분할 작업에 있어 결정적으로 중요하다는 것을 발견하였습니다.

- **Performance Highlights**: UNIP 구조를 통해 기존 다양한 사전 훈련 방법보다 최대 13.5%의 평균 mIoU 개선을 달성하였고, 특히 UNIP-S 모델은 MAE-L 모델과 동등한 성능을 유지하면서도 1/10의 계산 비용으로 작동하였습니다. UNIP는 TINN 및 Mask2Former와 같은 최신 기술(SOTA) 방법들을 능가하였으며, RGB 및 깊이(depth) 이미지와 같은 다른 모달리티에 대한 응용 가능성도 보여주고 있습니다. 전체적으로 UNIP는 적외선 분할 작업에서 기존 최고의 성능을 자랑하며, 높은 효율성과 효과성을 입증하였습니다.



### Rotation-Adaptive Point Cloud Domain Generalization via Intricate Orientation Learning (https://arxiv.org/abs/2502.02247)
Comments:
          13pages, supplementary included, early accepted by TPAMI

- **What's New**: 이번 연구에서는 3D 포인트 클라우드 분석에서의 예측할 수 없는 회전에 대한 취약성을 해결하기 위해, orientation-aware 3D domain generalization을 위한 혁신적인 회전 적응형 도메인 일반화 프레임워크를 제안합니다. 이 접근법은 복잡한 샘플을 활용한 반복 학습 과정을 통해 방향 변화를 완화하는 데 중점을 둡니다.

- **Technical Details**: 연구진은 각 포인트 클라우드에 대해 가장 도전적인 회전을 식별하고, 이를 최적화하여 복잡한 방향 세트를 구축합니다. 이후 방향 일관성 손실(orientation consistency loss)과 마진 분리 손실(margin separation loss)을 포함한 방향 인식 대비 학습 프레임워크를 활용하여 회전 일관성을 지닌 범주적으로 구별 가능한 일반화 가능한 특징들을 효과적으로 학습합니다.

- **Performance Highlights**: 다양한 3D 교차 도메인 벤치마크에서 실시된 광범위한 실험과 ablation 연구를 통해 제안된 접근법의 최첨단 성능을 확립하였습니다. 이는 orientation-aware 3D domain generalization의 맥락에서 매우 중요한 발전을 나타냅니다.



### Mask-informed Deep Contrastive Incomplete Multi-view Clustering (https://arxiv.org/abs/2502.02234)
- **What's New**: 본 논문에서는 Mask-informed Deep Contrastive Incomplete Multi-view Clustering (Mask-IMvC)라는 새로운 방법을 제안한다. 이 방법은 다양한 뷰에서의 결측값의 영향을 최소화하고, 여러 뷰에서 통합된 정보를 효과적으로 활용하기 위해 마스크 정보를 사용하여 일반 표현을 추출한다. 특히, Mask-IMvC는 결측 뷰의 보간 과정 없이 직접적인 데이터 클러스터링을 위한 접근 방식을 제공한다.

- **Technical Details**: Mask-IMvC는 여러 뷰에서 데이터 수집 시 나타나는 결측값의 영향을 줄이기 위해 마스크 기반의 융합 네트워크를 활용한다. 이 네트워크는 다양한 뷰에서 샘플의 관측 상태를 기반으로 마스크를 형성하여 결측 샘플의 기여를 제거하고, 이웃 샘플 간의 관계를 활용하여 취합된 뷰-공통 표현을 정규화한다. 또한, Re-weighted contrastive loss를 이용하여 clustering 성능을 향상시키는 방법론을 설계하였다.

- **Performance Highlights**: Mask-IMvC 방법은 다양한 MvC 데이터셋에서 기존 최첨단 기법들에 비해 우수한 성능을 입증하였다. 수행된 실험에 따르면, Mask-IMvC는 결측값이 있는 상황에서도 클러스터링의 질을 유지하며, 뷰 공통 표현의 강건성을 높임으로써 IMvC 작업에서의 성과를 크게 향상시키는 것으로 나타났다.



### A Robust Remote Photoplethysmography Method (https://arxiv.org/abs/2502.02229)
Comments:
          9 pages, 5 figures, 1 table

- **What's New**: 이 논문에서는 카메라를 활용하여 원거리에서 심박수를 측정하는 새로운 방법인 Remote Photoplethysmography (rPPG)를 제안합니다. 기존 rPPG 방법들이 움직임이나 조명 변화에 취약하여 정확도가 떨어지는 문제를 해결하기 위해, 얼굴 특징 탐지 및 반복적인 곡선 맞춤 등의 기술을 조합하여 심박수를 정확하게 측정할 수 있도록 하였습니다.

- **Technical Details**: 제안된 방법은 CIELAB 색공간을 이용하여 조명과 색상을 분리하고, MediaPipe 신경망을 사용한 얼굴 추적 알고리즘을 적용하여 움직임 아티팩트를 최소화합니다. 실험을 위해 19명의 자원봉사자로부터 26개의 비디오를 수집하였으며, 수집된 데이터는 Arduino 기반의 심박수 모니터를 사용하여 비교 검증하였습니다.

- **Performance Highlights**: 최종적으로 제안된 방법의 평균 절대 오차(MAE)는 1.95 BPM으로 이전 연구들보다 현저히 개선된 결과를 보였습니다. 이를 통해 심박수를 원거리에서 측정하는 데 있어 정확도를 크게 향상시킴으로써, 피험자의 행동에 대한 제한 없이 효과적인 측정이 가능하다는 점을 증명하였습니다.



### Exploring the latent space of diffusion models directly through singular value decomposition (https://arxiv.org/abs/2502.02225)
- **What's New**: 이 논문은 기존의 확산 모델(Diffusion Models, DMs)의 잠재 공간(latent space)을 직접 탐구하며, 이를 통해 이미지 생성 결과를 제어할 수 있는 세 가지 유용한 특성을 발견했습니다. 이 특성들은 데이터 수집을 요구하지 않으며, 생성된 이미지의 정체성 유지(identity fidelity)를 보장합니다. 이를 기반으로 문자 프롬프트에 의해 유도된 두 쌍의 잠재 코드로부터 임의 속성을 학습할 수 있는 새로운 이미지 편집 프레임워크(image editing framework)를 제안합니다.

- **Technical Details**: 연구자들은 특이값 분해(Singular Value Decomposition, SVD)를 통해 DMs의 잠재 공간을 분석하였으며, 이 과정에서 발견된 세 가지 특성은 다음과 같습니다. 첫째, 모든 시간 단계에서 의미적으로 유사한 작은 이웃(subspace)을 제공합니다. 둘째, 새로운 특성이 추가되지 않는 한, 기존 특성을 변경할 수 없습니다. 셋째, 특이값의 감소 순서에 따라 변속성이 있으며, 이는 두 다른 시간 단계 간의 잠재 코드에서 새 특성을 도입하는 대안적인 방법을 제공합니다.

- **Performance Highlights**: 우리는 새로운 이미지 편집 접근 방식을 이론적 분석과 다양한 데이터 세트에 대한 포괄적인 실험을 통해 검증했습니다. 이 연구는 이미지 편집에서 높은 품질을 유지하면서도 원본 이미지의 정체성 충실성을 보존하는 방법을 보여줍니다. 이들의 결과는 특히 새로운 속성을 도입하는 효율성과 논리적인 명확성을 강조하며, 향후 이미지 조작 분야의 혁신을 이끌 수 있기를 기대합니다.



### InterLCM: Low-Quality Images as Intermediate States of Latent Consistency Models for Effective Blind Face Restoration (https://arxiv.org/abs/2502.02215)
Comments:
          Accepted at ICLR2025

- **What's New**: 이번 연구에서는 InterLCM이라는 새로운 접근 방식을 제안합니다. 이는 기존의 Diffusion Models (DMs)의 한계를 극복하고, Latent Consistency Model (LCM)을 활용하여 더욱 우수한 의미론적 일관성을 달성합니다. 특히, InterLCM은 저품질 이미지를 LCM의 중간 상태로 처리함으로써 신뢰도와 품질 사이의 균형을 맞추는 혁신적인 방법입니다.

- **Technical Details**: InterLCM은 시각적 특징을 추출하는 Visual Module과 공간적 세부 정보를 포착하는 Spatial Encoder를 포함하여 구조적 및 의미론적 불확실성을 완화합니다. 이 접근 방식은 ODE-trajectory에서 불확실한 노이즈-데이터 매핑을 학습하여 더 높은 의미론적 일관성을 제공합니다. 또한, 교육 중 인지 손실(perceptual loss)을 통합할 수 있어 실제 환경에서의 복원 품질이 향상됩니다.

- **Performance Highlights**: 다양한 실험 결과, InterLCM은 합성 및 실제 데이터셋 모두에서 기존 방법들을 능가하는 성능을 보여주었습니다. 이러한 연구는 복원 품질을 크게 향상시킬 뿐만 아니라, 빠른 추론 속도를 달성하는 데도 기여합니다. 이를 통해 InterLCM은 이미지 복원 분야에서의 새로운 가능성을 제시하고 있습니다.



### Exploiting Ensemble Learning for Cross-View Isolated Sign Language Recognition (https://arxiv.org/abs/2502.02196)
Comments:
          3rd Place in Cross-View Isolated Sign Language Recognition Challenge at WWW 2025

- **What's New**: 이 논문에서는 WWW 2025에서 열린 Cross-View Isolated Sign Language Recognition (CV-ISLR) 챌린지에 대한 새로운 솔루션을 제시합니다. 전통적인 ISLR 방법들이 주로 정면에서 촬영된 데이터를 사용하는 반면, CV-ISLR은 다양한 카메라 각도에서 수화 영상을 인식하는 데 중점을 두고 있습니다. 이 연구는 Ensemble Learning을 활용하여 다양한 시점에서 모델의 강건성과 일반화를 향상시키는 접근 방식을 탐구합니다.

- **Technical Details**: CV-ISLR 문제는 뷰포인트 변동성과 제스처 복잡성을 포함한 두 가지 주요 도전 과제가 있습니다. 이를 해결하기 위해, 우리는 Video Swin Transformer (VST) 아키텍처에 Ensemble Learning을 통합하려고 합니다. 다양한 크기의 VST 변형을 RGB 및 Depth 입력에 적용하여 각각의 모델이 다양한 특징을 추출하고 융합할 수 있도록 합니다.

- **Performance Highlights**: 최종 모델은 RGB 및 Depth 스트림의 결과를 통합하여 더 정교한 예측을 제공합니다. 본 연구에서 제안한 방법은 RGB 기반 및 RGB-D 기반 ISLR 트랙에서 각각 3위로 랭크되었으며, 교차 뷰 인식의 문제를 잘 처리함을 보여줍니다. 이는 다양한 각도에서의 수어 인식 정확도를 높이는 효과적인 접근임을 입증합니다.



### ShapeShifter: 3D Variations Using Multiscale and Sparse Point-Voxel Diffusion (https://arxiv.org/abs/2502.02187)
- **What's New**: ShapeShifter라는 새로운 3D 생성 모델이 제안되었습니다. 이 모델은 단일 참조 모델을 기반으로 형태 변형을 합성하는 것을 학습하며, 기존의 3D 생성 방법들이 가지고 있는 기하학적 세부 사항 부족 및 긴 교육 시간 등을 해결합니다. 또한, 이 모델은 정밀한 세부 묘사를 보존하고 다양한 표면 형태를 처리할 수 있는 능력이 향상되었습니다.

- **Technical Details**: ShapeShifter는 희소 볼륨 격자(sparse voxel grid) 및 포인트, 노멀, 색상 샘플링을 결합한 다중 스케일 신경망 아키텍처를 사용합니다. 이 접근 방식은 계산 효율성을 높이고 빠른 추론(interactive inference)을 가능하게 하여, 예시(input) 형태에 따라 고품질 형태 변형을 생성합니다. 포인트 샘플링과 희소 합성을 결합하여 서로 다른 스타일과 톱로지를 가진 3D 변형을 만들어내는 멀티스케일 생성 방식을 구현하고 있습니다.

- **Performance Highlights**: ShapeShifter는 훈련 시간을 크게 줄일 수 있으며, 대개 몇 분 이내에 훈련이 완료됩니다. 이 결과는 텍스처가 추가된 메시(mesh)로 쉽게 변환할 수 있으며, 아티스트에 의해 안내되는 반복적 공동 창작(iterative co-creation)이 가능합니다. 최종적으로, 높은 품질의 기하학적 모델 출력은 필요에 따라 텍스처가 추가될 수 있습니다.



### Sequence models for continuous cell cycle stage prediction from brightfield images (https://arxiv.org/abs/2502.02182)
- **What's New**: 이 연구는 Fluorescent protein reporters를 사용하지 않고, 비형광 밝은 필드 영상(brightfield imaging)을 통해 연속적인 Fucci 신호를 예측하는 딥 러닝 방법을 종합적으로 평가하였습니다. 연구진은 분할된 RPE1 세포의 130만 이미지를 포함한 대규모 데이터셋을 생성하여 다양한 모델 카테고리 간의 예측 성능을 비교하였습니다. 또한, 원인적(causal) 및 변환기(transformer) 기반 모델이 단일 타임프레임 접근 방식보다 우수한 성능을 보임을 발견했습니다.

- **Technical Details**: 연구팀은 72시간 동안 5분 간격으로 촬영된 RPE1 세포의 비형광 밝은 필드 영상과 형광 시간 경과 영상을 사용하여 대규모 데이터셋을 생성하였습니다. 세포 핵은 Histone H2B 채널을 기반으로 StarDist 모델을 통해 분할 및 추적되었습니다. 후에 K-Means 클러스터링을 사용하여 전체 세포 주기 경로를 식별하고, 세그먼트된 핵 마스크를 통해 평균 Fucci 강도를 정규화하여 실제 Fucci 신호를 계산하였습니다.

- **Performance Highlights**: 근본적으로 causal 및 transformer 기반 비因果(non-causal) 모델은 단일 프레임 접근 방식에 비해 세포 상태 전환을 1시간 해상도에서 예측할 수 있는 능력을 보여주었습니다. 연구 결과는 이러한 시퀀스 모델이 세포 주기 역학을 정확히 예측하는 데 중요함을 강조하고, 라벨이 없는 영상에 대한 잠재력을 부각시킵니다. 이는 다양한 세포 과정에 대한 연구에 있어 더 넓은 적용 가능성을 의미합니다.



### DeepForest: Sensing Into Self-Occluding Volumes of Vegetation With Aerial Imaging (https://arxiv.org/abs/2502.02171)
- **What's New**: 이 연구에서는 밀집한 캐노피(layer) 속으로 깊게 침투할 수 있는 식생 데이터에 대한 접근성을 개선하는 새로운 방법을 제안합니다. 기존의 원격 감지 기법들이 가진 한계를 극복하기 위해 고해상도 항공 이미지를 이용하며, 이는 3D 식생 구조를 측정하는 LiDAR와 레이더의 보완 역할을 합니다. 새롭게 개발된 접근법은 넓은 범위의 현미경 이미징(imaging) 프로세스와 유사하지만, 훨씬 더 큰 규모와 강한 막음을 처리할 수 있는 점이 특징입니다.

- **Technical Details**: 연구진은 드론을 이용해 합성개구(imaging) 스캔을 통해 초점 스택을 수집하고, 사전 훈련된 3D 합성곱 신경망(3D convolutional neural networks)을 이용해 초점이 맞지 않은 신호 기여를 줄였습니다. 이로 인해 생성된 볼륨 반사 스택(volumetric reflectance stacks)은 식생 볼륨의 저주파(low-frequency) 표현을 포함하고 있습니다. 다양한 스펙트럼 채널로부터의 여러 반사 스택을 결합하여 식물 건강, 성장 및 환경 조건을 전반적으로 파악할 수 있는 통찰력을 제공합니다.

- **Performance Highlights**: 이 접근법을 통해 깊은 자기 차폐(self-occluding) 식생 볼륨을 효과적으로 감지할 수 있으며, 이는 밀림과 같은 복잡한 생태계에서의 생태학적 동태(ecosystem dynamics)를 이해하는 데 중요한 데이터로 활용될 수 있습니다. 기존 방법들과 비교해 더 높은 데이터 해상도와 깊이를 제공하며, 식물의 생장 및 환경 변화를 통계적으로 분석하는 데 유용합니다. 이번 연구의 결과는 관련 분야의 연구 및 기술 발전에 기여할 가능성이 큽니다.



### Progressive Correspondence Regenerator for Robust 3D Registration (https://arxiv.org/abs/2502.02163)
- **What's New**: 이번 연구에서는 Regor이라는 새로운 3D 등록 방법을 제안하여, 희소한 인라이어(inlier) 문제를 극복하기 위한 고품질 포인트 대응점을 생성하는 데 중점을 두고 있습니다. 기존의 아웃라이어 제거(outlier removal) 방법들이 주로 잘못된 대응점을 제거하는 데 초점을 맞춘 반면, Regor는 점진적으로 개선된 대응점을 생성하여 강력한 등록을 실현합니다. 특히, 이 방법은 기존의 방식보다 최대 10배 더 많은 정확한 대응점을 생성할 수 있습니다.

- **Technical Details**: Regor는 세 가지 주요 모듈로 구성된 점진적인 반복 프레임워크를 따릅니다: 지역 그룹화 및 재매칭(Local Grouping and Rematching), 지역 대응점 개선(Local Correspondence Refinement), 그리고 글로벌 대응점 개선(Global Correspondence Refinement)입니다. 각 반복에서 지역 그룹화 및 재매칭은 이전 반복의 공간 정보와 상관없이 점차적으로 매칭 공간을 줄여가며, 지역 대응점 개선은 센터 인식을 통한 삼각형 일관성을 도입하여 인라이어를 추출하고 지역 대응점을 업데이트합니다. 글로벌 대응점 개선은 전반적인 최적화를 수행합니다.

- **Performance Highlights**: 이 방법은 3DMatch와 KITTI 데이터셋에서 최첨단 성능을 달성하였으며, 기존의 아웃라이어 제거 방법들과 비교할 때, 정확한 대응점이 10배 더 많다는 점에서 큰 장점을 보입니다. Regor는 약한 특징에도 불구하고 강력한 등록을 달성할 수 있는 능력을 보여주며, 이를 통해 3D 컴퓨터 비전 분야에 중요한 기여를 하게 됩니다.



### On the Guidance of Flow Matching (https://arxiv.org/abs/2502.02150)
Comments:
          35 pages, 7 figures

- **What's New**: 이번 논문에서는 흐름 매칭(Flow Matching)을 위한 일반적인 가이드를 제공하는 첫 번째 프레임워크를 제안합니다. 이전의 확산 모델(Diffusion Models)과는 달리, 흐름 매칭의 가이드는 보다 일반적이며 이는 새로운 접근 방식이 필요함을 의미합니다. 본 연구는 흐름 매칭에 대한 가이드를 탐색하는데 중요한 기여를 하고 있습니다.

- **Technical Details**: 제안된 프레임워크에서 우리는 여러 가지 가이드 기술을 도출합니다. 여기에는 훈련이 필요 없는 비대칭적 완전 가이드, 훈련 기반 가이드를 위한 새로운 손실 함수, 고전적인 그래디언트 가이드를 포함하는 두 가지 근사 가이드 클래스가 포함됩니다. 이들 각각의 기법들은 이론적으로 조사되어 여러 상황에서 적합한 기법을 선택하는 데 도움을 줍니다.

- **Performance Highlights**: 합성 데이터셋(synthetic datasets), 이미지 역문제(image inverse problems), 오프라인 강화 학습(offline reinforcement learning) 등 다양한 실험을 통해 제안된 가이드 기법의 효과를 입증하였습니다. 제안된 흐름 매칭 가이드 프레임워크의 정확성을 또한 검증하였습니다. 실험 재현을 위한 코드는 제공된 URL에서 확인할 수 있습니다.



### DOC-Depth: A novel approach for dense depth ground truth generation (https://arxiv.org/abs/2502.02144)
Comments:
          Preprint. Code and dataset available on the project page : this https URL

- **What's New**: 본 논문에서는 LiDAR 센서를 통해 밀집 깊이(depth) 주석을 생성하는 DOC-Depth라는 새로운 방법을 제안합니다. 기존의 데이터셋 기록 방법들이 동적 환경에서 완전하게 밀집된 깊이 정보를 제공하지 못하는 한계를 극복하고, 다양한 LiDAR와 환경에 대해 잘 일반화되는 방식으로 설계되었습니다. 특히, 상태-of-the-art(dynamic object classification) 기술인 DOC를 통해 동적 물체의 폐색을 자동으로 처리할 수 있습니다.

- **Technical Details**: DOC-Depth는 LiDAR 측정을 기반으로 밀집 깊이 주석을 생성하는 방식을 사용합니다. 이 과정에서 최소한 하나의 LiDAR가 포함된 센서 배치를 가정하고 있으며, 카메라는 후속 작업을 위한 용도로만 사용됩니다. LiDAR와 RGB 카메라의 정렬을 위해 카메라 내부 및 외부 보정을 수행하며, 이 과정에서 정확성과 시간 효율성을 모두 고려한 타겟 기반 방법을 채택합니다.

- **Performance Highlights**: KITTI 데이터셋을 활용한 실험에서 DOC-Depth의 효율성을 입증했으며, 깊이 밀도가 16.1%에서 71.2%로 향상되었습니다. 다양한 LiDAR 센서와 여러 환경에서의 결과를 보여주며, 모든 소프트웨어 구성 요소는 연구 커뮤니티에 공개되었습니다. 이러한 접근법은 대규모 깊이 추정 데이터셋의 확장 가능성을 크게 높여 줍니다.



### VerteNet -- A Multi-Context Hybrid CNN Transformer for Accurate Vertebral Landmark Localization in Lateral Spine DXA Images (https://arxiv.org/abs/2502.02097)
Comments:
          10 pages with 7 figures

- **What's New**: 본 연구에서는 Dual-Energy X-ray Absorptiometry (DXA) 이미지에서 척추의 정확한 Landmark Localization (VLL)을 위한 자동화된 방법론인 VerteNet을 제안합니다. VerteNet은 하이브리드 CNN-Transformer 모델로서, Dual Resolution Self-Attention (DRSA)와 Dual Resolution Cross-Attention (DRCA)이라는 새로운 주의 메커니즘을 도입하여 DXA 이미지의 다양한 주파수를 효과적으로 캡처합니다. 또한, 이미지 내에서 개별 특성을 효율적으로 통합하는 Multi-Context Feature Fusion Block (MCFB)을 설계하여 성능을 개선했습니다.

- **Technical Details**: VerteNet 모델은 620개의 DXA LSI 이미지를 사용하여 훈련되고 평가되었습니다. 이 모델은 다양한 장비에서 수집된 데이터를 기반으로 하여, Channel-wise Self-Attention (CSA) 및 Dual Resolution Self-Attention (DRSA) 블록을 사용하여 입력 특성 맵의 다양한 공간 맥락에서 고주파(HF) 및 저주파(LF) 구성 요소를 포착합니다. 모델의 성능은 GuideNet, HRNet 및 NFDP 모델과 비교되었으며, VerteNet이 도출한 결과는 이러한 기존 모델들을 능가합니다.

- **Performance Highlights**: VerteNet 모델은 정확한 vertebral landmark localization (VLL)을 통해 abdominal aorta cropping을 자동으로 탐지할 수 있는 알고리즘을 포함하고 있습니다. 이 알고리즘은 large datasets 내에서 많은 이미지를 효율적으로 처리하여 시간과 자원을 절약할 수 있도록 설계되었습니다. 결국, VerteNet을 활용한 IVG 생성은 AAC-24 스코어링 방법의 읽기 일관성을 향상시킬 수 있는 가능성을 보여주었으며, 이는 의료 진단에서 중요한 기여를 할 것으로 기대됩니다.



### Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization (https://arxiv.org/abs/2502.02096)
- **What's New**: 이 논문은 새로운 Dual-Flow 프레임워크를 제안하여 다중 타겟 인스턴스 비특정 적대적 공격을 가능하게 합니다. 이를 통해 적대적 속도 함수(adversarial velocity function)를 개발하기 위한 Cascading Distribution Shift Training을 활용합니다. 실험 결과, Dual-Flow는 이전의 다중 타겟 생성 공격에 비해 transferability가 크게 향상되었습니다.

- **Technical Details**: Dual-Flow 프레임워크는 미리 학습된 diffusion 모델과 LoRA 기반의 경량화된 적대적 속도 함수를 결합하여 구조화된 perturbation 생성을 가능하게 합니다. 또한, Cascading Distribution Shift Training을 통해 공격 능력을 향상시키고, 다이내믹 그래디언트 클리핑을 사용하여 ℓ∞ 제약 조건을 강제합니다. 이 과정에서 전방 흐름(forward flow)과 역방향 흐름(reverse flow)을 통해 효과적인 공격을 위한 변조된 이미지를 생성합니다.

- **Performance Highlights**: Dual-Flow 알고리즘은 기존의 다중 타겟 생성 공격보다 향상된 블랙박스 transferability를 보여줍니다. 예를 들어, Inception-v3에서 ResNet-152로의 성공률이 34.58% 증가하였습니다. 또한, 이 공격 방법은 적대적 방어 메커니즘에 대해 더 강한 견고성을 보였습니다.



### Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation (https://arxiv.org/abs/2502.02091)
- **What's New**: 최근의 4D 동적 장면 편집 방법들은 수천 개의 2D 이미지를 수정하고 전체 장면을 업데이트하는데 여러 시간의 처리가 소요되어 효율성이 떨어진다. 본 연구에서는 4D 가우시안 표현을 활용한 동적 장면 편집 방법을 제안하여 시간 차원에서 더 나은 확장성을 목표로 한다. 새로운 접근법은 편집 시간을 절반 이상 단축시키며, 사용자 지침을 잘 따르는 높은 편집 품질을 제공한다.

- **Technical Details**: 제안된 방법은 스태틱 3D 가우시안만을 편집하여 최소한의 요소로 비주얼 편집을 수행하며, 이를 통해 전체 동적 장면을 수정할 수 있게 한다. 가우시안 프리미티브의 위치 변화에 따른 불일치를 해결하기 위해 스코어 증류(score distillation) 메커니즘을 사용하여 정제 단계를 추가한다. 이로써, 정적 3D 가우시안과 변형 필드의 정렬을 개선하고 모션 아티팩트를 제거할 수 있다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 편집 시간을 절반 이상 줄이면서 향상된 시각적 품질을 제공한다는 것을 입증하였다. 동적 장면 편집에 있어서 다양한 입력 지침을 효과적으로 처리할 수 있는 능력이 확인되었다. 또한, Hexplane 기반 접근법을 사용하여 편집 품질과 렌더링 효율성을 대폭 향상시켰음을 보여준다.



### IPO: Iterative Preference Optimization for Text-to-Video Generation (https://arxiv.org/abs/2502.02088)
- **What's New**: 이번 논문에서는 비디오 생성 모델의 품질을 개선하기 위해 인간의 선호와 모델을 정렬하는 Iterative Preference Optimization (IPO) 전략을 소개합니다. 이는 비디오 품질 향상에 필요한 인간 피드백을 통합하여, 주제의 일관성, 매끄러운 동작, 미적 품질 등을 고려합니다. IPO는 멀티모달 대형 언어 모델(MLLM)을 활용하여 자동으로 선호 라벨을 부여하며, tedious한 수동 라벨링 없이 반복적인 선호 최적화를 가능하게 합니다.

- **Technical Details**: IPO는 세 가지 핵심 요소로 구성됩니다: 선호 데이터셋(preference dataset), 비평가 모델(critic model), 반복 학습 프레임워크(iterative learning framework)입니다. 선호 데이터셋을 통해 비평가 모델을 훈련시키고, 이를 활용하여 T2V(Text-to-Video) 모델의 비디오 품질을 정당화하며, 다양한 비디오 품질 라벨을 자동으로 생성할 수 있습니다. 이 과정은 Diffusion-DPO 또는 Diffusion-KTO 기법을 사용하여 T2V 모델의 성능을 계속적으로 강화합니다.

- **Performance Highlights**: VBench 벤치마크에서 IPO는 사전 훈련된 모델의 비디오 생성 품질을 효과적으로 개선하는 결과를 나타냈습니다. 특히, 2B 매개변수를 가진 모델이 5B 매개변수를 가진 모델을 초월하는 성과를 기록하며, 새로운 최첨단 성능을 달성했습니다. 연구진은 이 연구의 데이터셋, 코드 및 모델을 공개하여 향후 비디오 생성 연구를 촉진할 계획입니다.



### Improving Power Plant CO2 Emission Estimation with Deep Learning and Satellite/Simulated Data (https://arxiv.org/abs/2502.02083)
- **What's New**: 본 연구는 발전소에서의 CO2 배출량을 정확히 측정하기 위해 위성 기반의 데이터를 활용한 새로운 접근법을 제시합니다. 기존 데이터 한계를 극복하기 위해 Sentinel-5P의 NO2 데이터를 통합하여 연속적인 XCO2 맵을 생성하고, OCO-2/3의 실제 위성 관측 결과를 포함합니다. 이를 통해 71개의 발전소에서의 CO2 배출량을 더 정교하게 측정할 수 있는 기반을 마련했습니다.

- **Technical Details**: 이 연구에서는 두 가지의 심층 학습 아키텍처인 CNN과 사용자 정의된 U-Net 모델을 사용하여 CO2 배출률을 추정합니다. CNN은 여러 층으로 구성된 아키텍처로, 특징 탐지 및 차원 축소를 통해 예측의 정확성을 향상시키는 데 초점을 맞췄습니다. U-Net 모델은 특징 추출과 공간적 맥락 유지를 효율적으로 처리하여 CO2 농도에서 배출 플럭스 변환을 학습할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제시한 방법은 기존 연구에 비해 CO2 배출률의 정확성이 현저히 개선되었음을 보여줍니다. 기상 데이터 및 기타 변수를 통합하여 무결점의 연속적이고 격자화된 XCO2 맵을 생성함으로써, 발전소의 CO2 배출을 효과적으로 정량화할 수 있었습니다. 이러한 접근법은 환경 보호 이니셔티브와 규제 프레임워크 수립에 기여할 수 있는 가능성을 나타냅니다.



### LoRA-TTT: Low-Rank Test-Time Training for Vision-Language Models (https://arxiv.org/abs/2502.02069)
- **What's New**: 본 논문에서는 전통적인 Test-Time Training (TTT) 기법으로는 해결하기 어려운 문제를 제기하며, Low-Rank Adaptation (LoRA)을 이미지 인코더에 적용한 LoRA-TTT라는 새로운 방법을 제안합니다. 이 방법은 TTT 중 모델의 초기 일반화 능력을 유지할 수 있으면서도 메모리 소모와 런타임 오버헤드를 최소화하여 성능 향상을 이룹니다. LoRA-TTT는 TTT를 위한 새로운 재구성 손실을 도입하여 다양한 도메인에 적응할 수 있도록 설계되었습니다.

- **Technical Details**: LoRA-TTT의 핵심은 이미지 인코더에 LoRA를 적용하여 파라미터 업데이트 시 텍스트 인코더를 필요로 하지 않으며, 이는 메모리와 처리 속도에서 큰 장점을 제공합니다. 본 방법은 고유의 텍스트 프롬프트 튜닝을 하지 않고도 다양한 텍스트 프롬프트에 잘 일반화할 수 있는 능력을 보유하고 있습니다. Entropy 손실과 새로운 재구성 손실을 결합하여 모델의 신뢰성을 높이며, 이는 실시간 데이터 처리나 메모리 제약이 있는 환경에서도 유용합니다.

- **Performance Highlights**: LoRA-TTT는 OOD 벤치마크에서 평균 5.79%의 제로샷 최고 정확도를 향상시켰고, 세분화 벤치마크에서는 1.36%의 개선을 이뤘습니다. 15개 데이터셋에 대한 실험을 통해 기존 TTT 기법들보다 우수한 성능을 입증하였으며, 외부 모델이나 캐시 없이도 실시간 처리에 적합한 결과를 보여주었습니다. 이러한 성능 향상은 메모리 요구사항이 낮고 모델의 실용성을 높여줍니다.



### CASIM: Composite Aware Semantic Injection for Text to Motion Generation (https://arxiv.org/abs/2502.02063)
- **What's New**: 본 논문에서는 Composite Aware Semantic Injection Mechanism(CASIM)을 제안하여 고정 길이의 텍스트 임베딩을 사용하는 기존 방법의 한계를 극복합니다. CASIM은 복합적인 인간 동작의 특성과 텍스트 및 모션 토큰 간의 동적 정렬을 학습합니다. 이로 인해 모션 생성 품질과 텍스트-모션 정렬이 향상되어 실제 사용 가능한 수준의 동작이 가능해집니다.

- **Technical Details**: CASIM은 복합 인식 텍스트 인코더(composite-aware text encoder)와 텍스트-모션 정렬기(text-motion aligner)로 구성되어 있습니다. 이 기술은 텍스트 토큰과 모션 프레임 간의 동적 정렬을 통해 세밀한 의미 관계를 캡처합니다. CASIM은 모델 및 표현 방식에 구애받지 않아, 서로 다른 생성 모델에 쉽게 통합할 수 있는 장점을 가집니다.

- **Performance Highlights**: 실험 결과, HumanML3D 및 KIT 벤치마크에서 CASIM을 적용한 다양한 최신 방법들이 FID, R-precision 및 Multimodality Distance에서 현저한 개선을 보임을 보여줍니다. CASIM은 긴 기간의 인간 동작 생성에서도 모션 품질과 텍스트-모션 정렬을 개선하였으며, 이러한 성능은 정량적 분석뿐 아니라 주목(attention) 가중치 분석을 통해 확인되었습니다. 이 새로운 접근 방식은 고정 길이의 의미 주입 방법 대비 우수한 성능을 입증하였습니다.



### MORPH-LER: Log-Euclidean Regularization for Population-Aware Image Registration (https://arxiv.org/abs/2502.02029)
- **What's New**: MORPH-LER는 인구 수준에서의 형태학적 통계(morphological statistics)를 통합한 이미지 등록 방법을 제안합니다. 이 기술은 трасляционные 그룹의 복잡성을 고려하며, 해석 가능성과 재구성의 균형을 맞추는 데 중점을 둡니다. 기존 방법들이 가진 한계를 해결하여, 해부학적으로 일관된 변형(deformation)을 생성하는 데 기여합니다.

- **Technical Details**: MORPH-LER는 Log-Euclidean 정규화(regularization) 프레임워크를 사용하여 인구 인식을 위한 비지도 이미지 등록을 수행합니다. 이 방법은 공간 변환에서 인구 형태학적 특성을 학습하며, 다운스트림 등록 네트워크를 안내하고 정규화합니다. 이를 통해 대칭성과 역일관성(inverse consistency)을 보장하는 선형화된 잠재 공간(latent space)을 생성합니다.

- **Performance Highlights**: MORPH-LER는 OASIS-1 뇌 영상 데이터셋을 통해 두 가지 심층 학습 기반 등록 네트워크에 걸쳐 검증되었습니다. 실험 결과, 이 방법이 해부학적으로 정확하고 계산 효율적이며 통계적으로 의미 있는 변형을 생성하는 데 성공했음을 보여줍니다. 이는 인구 형태학적 분석(population morphometrics analysis)에서 더 나은 해석 및 적용 가능성을 제시합니다.



### From Fog to Failure: How Dehazing Can Harm Clear Image Object Detection (https://arxiv.org/abs/2502.02027)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2410.01225

- **What's New**: 이번 연구는 객체 탐지에 인체의 시각적 단서를 통합하는 데 있어 발생하는 도전 과제를 탐구하고, 다단계 프레임워크를 제안합니다. 이 프레임워크는 경량 탐지기가 관심 영역(Region of Interest, RoI)을 식별하고, 이 후 공간적 주의력에 기반한 디헤이징으로 강조한 후, 더욱 강력한 탐지 모델로 최종 탐지를 진행합니다. 이로 인해 안개 조건에서는 효과적인 성능을 보여주지만, 명확한 이미지를 처리할 때는 예상치 못한 성능 저하가 발생하는 점을 분석합니다.

- **Technical Details**: 제안된 방법론은 대기 산란 모델과 인간 시각 피질의 원리를 활용하여 저조도 조건에서 객체 탐지를 향상시키는 딥 러닝 프레임워크를 제시합니다. 이 파이프라인은 경량 탐지 모델로 시작하여 관심 영역을 식별하고, 디헤이징 과정에서 공간적 주의력을 개발하여 중요 특징을 보존하고 계산 비용을 줄입니다. 이후 더욱 견고한 탐지 모델이 객체 인식을 정제하고 개선합니다.

- **Performance Highlights**: Foggy Cityscapes 데이터셋에서의 성능 평가를 통해 AOD-NetX 통합된 모델이 안개 이미지에서 우수한 성능을 보이는 반면, 전통적인 모델들은 맑은 이미지에서 안개 이미지로 전환될 때 평균정밀도(mAP)에서 자연스러운 감소를 보입니다. 우리의 파이프라인은 다양한 저조도 조건에서도 강력한 성능을 보여주며, SSIM과 PSNR을 포함한 평가 지표가 사용되었습니다.



### Multi-illuminant Color Constancy via Multi-scale Illuminant Estimation and Fusion (https://arxiv.org/abs/2502.02021)
Comments:
          10 pages, 4 figures, this manuscript is under the consideration of Optics Express

- **What's New**: 이번 연구는 다중 조명 색상 불변성(multi-illuminant color constancy) 방법론을 통해 이미지 내의 색의 왜곡을 제거하는 새로운 방법을 제안합니다. 기존의 방법들은 이미지 스케일(image scale)의 영향을 간과했으며, 우리는 이를 해결하기 위해 다중 스케일 이미지를 사용하는 프레임워크를 도입했습니다. 세 가지 가지의 U-Net 기반의 신경망을 통해 멀티 기울기 조명 분포 맵(multi-grained illuminant distribution map)을 예측하고, 주의 메커니즘(attentional mechanism)을 통해 이를 통합하는 방안을 제시했습니다.

- **Technical Details**: 제안된 다중 조명 색상 불변성 방법은 다중 스케일 이미지에서 추정된 조명의 구성 요소들을 선형 조합으로 나타내어 조명 맵을 생성합니다. 각 스케일에 대해 U-Net을 사용하여 조명 맵을 생성하는 세 개의 브랜치를 가진 합성곱 신경망(convolutional neural network)을 구성하였습니다. 마지막으로, 조명 융합 모듈(attentional illuminant fusion module)을 통해 서로 다른 스케일에서 추정된 조명 맵을 적응적으로 통합합니다.

- **Performance Highlights**: 실험 분석 결과, 제안된 방법이 최신 기술 혁신(state-of-the-art)에서 뛰어난 성능을 보임을 입증하였습니다. 조명의 부정확한 추정으로 인해 발생하는 색상 왜곡 문제를 효과적으로 감소시키며, 다양한 조명 조건에서도 강건한 성능을 발휘합니다. 따라서 이 방법은 사진 품질 향상과 함께 다운스트림 비전 작업의 조명 강건성 개선에 기여할 수 있을 것으로 기대됩니다.



### One Diffusion Step to Real-World Super-Resolution via Flow Trajectory Distillation (https://arxiv.org/abs/2502.01993)
- **What's New**: 이번 논문에서는 다단계 확산 모델의 연산 비용 문제를 해결하기 위해 새로운 단일 단계 확산 모델인 FluxSR을 제안합니다. FluxSR은 FLUX.1-dev 모델을 기반으로 하여 높은 품질의 이미지를 한 번의 샘플링 단계에서 생성할 수 있도록 설계되었습니다. 특히, Flow Trajectory Distillation (FTD) 접근 방식을 통해, 이미지를 생성하는 과정에서 발생할 수 있는 아티팩트를 줄이는 방법을 제시합니다.

- **Technical Details**: FluxSR은 여러 핵심 구성 요소로 이루어져 있습니다. 첫째, FTD를 통해 T2I(텍스트-투-이미지)와 SR(슈퍼 해상도) 흐름 간의 관계를 명시적으로 구축합니다. 둘째, TV-LPIPS를 감각 손실로 사용하여 이미지의 고주파 성분을 복원하고 아티팩트를 줄이며, Attention Diversification Loss (ADL)을 통해 토큰의 다양성을 향상시키는 방법을 도입합니다. 이러한 기법들은 모두 12B 이상의 파라미터를 가진 대형 모델을 효율적으로 활용할 수 있도록 설계되었습니다.

- **Performance Highlights**: 폭넓은 실험 결과, FluxSR은 동일한 샘플링 단계에서 기존의 확산 기반 Real-ISR 방법들에 비해 뛰어난 성능을 발휘함을 입증하였습니다. 연구팀은 또한 이 방법이 이미지의 사실성을 크게 향상시키고, 고주파 아티팩트 문제를 효과적으로 해결함을 보여주었습니다. 새로운 모델의 구현 코드와 모델은 향후 공개될 예정입니다.



### DCT-Mamba3D: Spectral Decorrelation and Spatial-Spectral Feature Extraction for Hyperspectral Image Classification (https://arxiv.org/abs/2502.01986)
- **What's New**: 본 논문에서는 hyperspectral (HSI) 이미지 분류를 위한 새로운 프레임워크인 DCT-Mamba3D를 제안합니다. DCT-Mamba3D는 3D 스펙트럴-스페이셜 분산 모듈(3D-SSDM), 3D-Mamba 모듈 및 글로벌 잔여 강화(GRE) 모듈로 구성되어 있으며, 이는 스펙트럴 중복 및 복잡한 스페이셜-스펙트럴 의존성을 해결합니다. 이를 통해 다차원에서의 특징 명확성을 높이고, 다양한 스펙트럼에서 동일 물체를 구분하는 데 도움을 줍니다.

- **Technical Details**: DCT-Mamba3D의 3D Spatial-Spectral Decorrelation Module(3D-SSDM)은 3D DCT(Discrete Cosine Transform) 기저 함수를 적용하여 시공간 픽셀을 분산된 주파수 성분으로 변환함으로써 스펙트럴 및 스페이셜 중복을 줄입니다. 3D-Mamba 모듈은 상태 공간 모델(state-space model)을 이용하여 복잡한 스페이셜-스펙트럴 의존성을 캡처합니다. 마지막으로, GRE 모듈은 피처 표현을 안정화시켜 모델의 강인성과 수렴성을 개선합니다.

- **Performance Highlights**: 제안된 DCT-Mamba3D는 여러 벤치마크 데이터셋에서 기존의 최첨단 방법들을 초월하는 성능을 보여주었습니다. 특히 다양한 스펙트럼에서 동일한 객체와 같은 까다로운 시나리오에서 우수한 성과를 냈습니다. 이러한 성과는 HSI 데이터의 높은 차원성과 중복성을 효과적으로 처리하는 DCT-Mamba3D의 강력한 성능을 입증합니다.



### AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs (https://arxiv.org/abs/2502.01977)
Comments:
          Technical Report

- **What's New**: 이 논문에서는 대규모 UI 데이터 주석 자동화 파이프라인인 AutoGUI를 제안합니다. 기존의 UI 데이터셋은 주로 컨텍스트가 없는 요소 주석이나 소규모 기능 설명에 국한되어 있었지만, AutoGUI는 UI 상호작용의 변화를 기반으로 UI 요소에 대한 상세하고 기능적인 주석을 대규모로 생성할 수 있습니다. 이를 통해 VLM(Visual Language Model)의 사용자 인터페이스 이해 능력을 크게 향상시킬 수 있는 가능성이 제시됩니다.

- **Technical Details**: AutoGUI의 주석 파이프라인은 웹 브라우저 또는 Android 에뮬레이터에서 다양한 UI 상호작용 경로를 수집하고, 이러한 요소들과의 상호작용 시 UI 내용의 변화를 분석하여 LLMs(Large Language Models)를 활용하여 기능성을 추론합니다. 수집된 데이터의 품질을 유지하기 위해 LLM 지원 거부와 검증 과정을 도입하여 잘못된 샘플을 제거합니다. 최종적으로, AutoGUI-704k 데이터셋은 704,000개의 고품질 기능 주석 데이터를 포함하여 VLM의 파인튜닝 및 평가에 사용됩니다.

- **Performance Highlights**: AutoGUI 파이프라인을 통해 수집된 데이터는 VLM의 UI 기초 정확도를 크게 향상시키고, 기존의 웹 프리트레인 데이터 유형에 비해 현저하게 우수한 성능을 보입니다. 실험 결과, 제안된 데이터셋을 통해 VLM의 UI 상호작용 이해력이 96.7%의 주석 정확도로 개선되었으며, 이는 훈련된 인간 주석자와 비슷한 수준입니다. 이러한 결과는 AutoGUI가 GUI 지향 VLM을 위한 대규모 데이터 생성의 효율적인 파이프라인으로 자리매김할 수 있음을 시사합니다.



### Mitigating Object Hallucinations in Large Vision-Language Models via Attention Calibration (https://arxiv.org/abs/2502.01969)
- **What's New**: 최근의 연구는 Large Vision-Language Models (LVLMs)에서 발생하는 객체 환각(object hallucination) 문제를 해결하기 위한 방안을 제안하였습니다. 기존 방법들이 LVLM의 시각적 토큰과 공간 위치 간의 상관관계에 한정돼 있는 반면, 이 연구에서는 훈련 없이 사용할 수 있는 Uniform Attention Calibration (UAC)과 동적 조정이 가능한 Dynamic Attention Calibration (DAC) 방법을 통해 새로운 솔루션을 제시합니다. 이러한 방법은 다양한 LVLM 아키텍처에서 고도로 효율적인 결과를 가져오며, 객관적인 환각을 감소시키는 데 큰 성과를 보여줍니다.

- **Technical Details**: UAC는 무의미한 입력 이미지에서 편향(bias)을 추정하고 주의(attention) 불균형을 교정하는 보정(matrix) 행렬을 적용하여 학습이 필요 없는 단순한 솔루션을 제공합니다. DAC는 이 개념을 확장하여 자기 주의(self-attention) 메커니즘에 적용할 수 있는 학습 가능한 모듈을 통합합니다. 이 모듈은 대비 학습(contrastive learning)을 통해 이미지 내 객체 위치에 관계없이 일관된 출력을 유도하도록 조정됩니다.

- **Performance Highlights**: 실험 결과, UAC와 DAC는 여러 벤치마크에서 객관적 환각을 유의미하게 줄이며, LLaVA-1.5, mPLUG-Owl2 등 다양한 LVLM 모델에서 우수한 성과를 내고 있습니다. 또한, MME 및 LLaVA-Bench에서 LVLM의 전체 인식 능력을 강화하는 효과도 확인되었습니다. 이 연구는 LVLM의 성능 개선과 더불어 실질적인 응용 가능성을 보여 줍니다.



### Memory Efficient Transformer Adapter for Dense Predictions (https://arxiv.org/abs/2502.01962)
Comments:
          This paper is accepted by ICLR 2025

- **What's New**: 이번 논문에서 제안하는 META는 Vision Transformer (ViT) 어댑터의 메모리 효율성을 개선하고 비효율적인 메모리 접근 작업을 줄여 모델의 메모리 시간 소비를 감소시키는 방법이다. 이를 위해, self-attention과 feed-forward 네트워크 계층 간의 layer normalization을 공유하는 어댑터 블록을 도입하였다. 또한, cross-shaped self-attention 기법을 사용하여 잦은 reshaping 작업을 줄이는 한편, 경량의 convolutional 브랜치를 추가하여 밀집 예측 작업에서의 지역 유도 편향을 강화했다.

- **Technical Details**: META는 ViT의 기존 모델들에 비해 메모리 소비와 추론 시간 비용에서 개선된 성능을 보여준다. 이 어댑터 블록은 self-attention과 feed-forward 계층 간의 normalization 작업을 공유하여 메모리 소비를 감소시키고, cross-shaped self-attention을 사용해 reshape 작업에 대한 의존성을 줄인다. 또한, cascaded 메커니즘을 통해 다양한 헤드 특징을 계산하여 다양한 특징 표현을 풍부하게 한다.

- **Performance Highlights**: 다양한 데이터셋인 MS-COCO와 ADE20K에서 수행한 실험 결과, META는 예측 품질을 크게 향상시키며 새로운 최첨단 정확도를 달성하였다. 또한, 파라미터 수와 메모리 소모 요건을 줄이고 더 빠른 추론 속도를 기록하였다. 이론적으로는 META가 기존 ViT 어댑터 방법들보다 더 우수한 일반화 능력과 적응성을 보인다고 증명하였다.



### Hierarchical Consensus Network for Multiview Feature Learning (https://arxiv.org/abs/2502.01961)
Comments:
          AAAI 2025 accepted paper

- **What's New**: 이 논문은 다중 시나리오에서 데이터의 구분 가능한 특징을 학습하기 위한 새로운 방법인 계층적 합의 네트워크(HCN)를 제안합니다. HCN은 각 뷰(view) 간의 계층적 합의(consensus)를 포착하기 위하여 분류 합의(classifying consensus), 코딩 합의(coding consensus), 글로벌 합의(global consensus)의 세 가지 합의 지표를 도출합니다. 이러한 접근 방식은 다중 뷰 간의 정보 통합을 개선하여 더 포괄적이고 차별화된 특징을 얻을 수 있게 합니다.

- **Technical Details**: HCN은 각 뷰 내에서의 차별적 정보와 공통 정보를 학습하기 위해 뷰별(autoencoder) 오토인코더를 사용합니다. 제시된 방법은 원본 및 증가된 데이터(augmented data)를 사용하여 잠재 특징(latent features)을 얻고, 다양한 합의 학습을 통해 특징들을 정교화합니다. 특히, 분류 합의 학습에서는 한 뷰에 조건부인 다른 뷰의 클래스 확률의 조건부 엔트로피를 최소화하는 방식을 사용하고, 글로벌 합의에서는 두 개의 잠재 특징 간의 차이를 최소화합니다.

- **Performance Highlights**: 총 네 개의 다중 뷰 데이터셋에서 진행된 실험 결과는 HCN이 여러 최신 기법(State-of-the-Art)보다 우수한 성능을 보임을 입증합니다. 이 연구는 다중 뷰 데이터에서의 일관성(consistency)을 최대화하면서도 추가 비용을 최소화하는 방법을 양산합니다. 이로써, 기존의 복잡한 접근 방식과 비교하여 HCN은 낮은 복잡도로 효과적인 특징 학습을 가능하게 합니다.



### MATCNN: Infrared and Visible Image Fusion Method Based on Multi-scale CNN with Attention Transformer (https://arxiv.org/abs/2502.01959)
- **What's New**: 본 논문에서는 특징을 추출하는 데 있어 기존 방법의 한계를 극복하기 위해 다중 스케일 합성에 주안점을 두고 다중 스케일 합성 모듈(MSFM)과 글로벌 특징 추출 모듈(GFEM)을 활용한 새로운 크로스 모달 이미지 융합 기법인 MATCNN을 제안합니다. 이 방법은 세부 특징 손실을 줄이면서 글로벌 특징 표현 능력을 개선하였으며, 특히 적외선 이미지의 중요한 정보와 가시 이미지의 배경 텍스처를 보존하기 위해 정보 마스크를 이용합니다. 또한, 새로운 최적화 알고리즘을 개발하여 특징 추출을 유도하고 궁극적으로 우수한 융합 결과를 달성합니다.

- **Technical Details**: MATCNN은 여러 스케일에서의 로컬 특징을 추출하기 위해 다중 스케일 융합 모듈(MSFM)을 채택하고, 글로벌 특징을 추출하기 위해 글로벌 특징 추출 모듈(GFEM)을 활용합니다. 이 두 모듈의 결합은 세부 정보 손실을 줄이는 동시에 글로벌 특징의 표현 능력을 향상합니다. 최적화 알고리즘은 이미지의 내용, 구조적 유사성 지수 측정 및 글로벌 특징 손실을 결합하여 마스크를 통해 특징 추출을 안내합니다.

- **Performance Highlights**: 정량적 및 정성적 평가를 통해 MATCNN이 적외선 주요 목표를 강조하고 가시 이미지에서 추가적인 세부 정보를 보존하며, 크로스 모달 이미지에 대한 융합 결과에서 더 나은 성능을 기록함을 확인했습니다. MATCNN의 성능은 최신 기법과 비교하여 더 적절한 두드러짐, 충실성 및 대비를 제공함을 보여주었으며, 다양한 데이터셋에서 우수한 결과를 나타냈습니다.



### LAYOUTDREAMER: Physics-guided Layout for Text-to-3D Compositional Scene Generation (https://arxiv.org/abs/2502.01949)
- **What's New**: 이번 논문에서는 텍스트 기반의 3D 장면 생성 분야에 본격적으로 기여할 수 있는 LayoutDreamer라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 3D Gaussian Splatting(3DGS)을 이용해 고품질이며 물리적으로 일관된 장면 생성을 가능하게 합니다. 특히, 텍스트 프롬프트를 바탕으로 장면 그래프를 형성하고, 이를 통해 3D 개체의 밀도와 레이아웃을 조정합니다. LayoutDreamer는 종전의 방법들이 겪고 있던 여러 한계를 극복할 수 있는 가능성을 보여줍니다.

- **Technical Details**: LayoutDreamer는 세 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째로, 장면 그래프를 통해 상호작용 관계를 명확히 하고, 3D Gaussian을 초기화하는 방법을 제시합니다. 두 번째로, 동적인 카메라 조정 전략을 통해 물체의 포즈와 위치, 밀도를 최적화합니다. 마지막으로, 물리적 제약을 적용하기 위해 물리 및 레이아웃 에너지를 최소화하는 과정을 두 단계로 나누어 구현합니다.

- **Performance Highlights**: LayoutDreamer는 T3Bench의 여러 물체 생성 지표에서 최첨단(SOTA) 성능을 달성하며, 물리적 제약을 준수하는 고충실도의 3D 장면을 생성하는 데 매우 우수한 성능을 보입니다. 이러한 성능은 LayoutDreamer가 고도로 제어 가능한 장면 편집 및 확장 기능을 제공하도록 돕습니다. 포괄적인 실험 결과는 LayoutDreamer가 기존의 방법들에 비해 뛰어난 품질과 의미적 정렬을 제공함을 보여줍니다.



### DAMO: Data- and Model-aware Alignment of Multi-modal LLMs (https://arxiv.org/abs/2502.01943)
- **What's New**: 이번 논문에서는 기존의 Direct Preference Optimization (DPO) 방법들이 다양한 데이터 하드니스에 따라 비대칭적인 반응을 보이는 현상을 개선하기 위한 새로운 접근법인 Data- and Model-aware DPO (DAMO)를 제시합니다. DAMO는 데이터 하드니스와 모델의 실시간 반응을 동시에 고려하여 동적으로 최적화 과정을 조정합니다. 이를 통해 모델은 다양한 난이도를 가진 데이터에 잘 적응할 수 있게 됩니다.

- **Technical Details**: DAMO는 두 가지 주요 메커니즘, 즉 Data-aware Preference Optimization과 Model-aware Preference Optimization을 도입합니다. 데이터 하드니스는 CLIP 기반의 이미지-텍스트 유사도 점수를 통해 정량화되며, 이를 통해 얻은 스코어는 확률로 변환되어 하드니스 추정에 효과적으로 사용됩니다. 모델의 반응도는 선호 및 거부 응답 사이의 보상 격차를 통해 추정되며, 이 모든 정보를 사용하여 β 매개변수를 동적으로 조정합니다.

- **Performance Highlights**: 다섯 가지 벤치마크에 대한 광범위한 실험 결과, DAMO는 높은 신뢰성 및 일반 작업에서의 효과성을 향상시켰음을 입증했습니다. 예를 들어, Object HalBench 기준에서, DAMO-7B는 응답 수준과 언급 수준의 환각을 각각 90.0%와 95.3% 줄였고, GPT-4V의 성능을 초과하여 더욱 향상된 결과를 보여주었습니다.



### Toward a Low-Cost Perception System in Autonomous Vehicles: A Spectrum Learning Approach (https://arxiv.org/abs/2502.01940)
- **What's New**: 이번 연구는 자율 주행 및 자율 차량을 위한 고밀도 깊이 맵을 생성하기 위한 새로운 비용 효율적인 접근 방식을 제시합니다. 기존 카메라 RGB 이미지와 딥 뉴럴 네트워크(DNN) 4D 레이더 탐지기로부터 얻은 이미지를 통합하여 새로운 픽셀 위치 인코딩 알고리즘을 도입합니다. 이 알고리즘은 레이더 깊이 맵과 RGB 이미지를 통합된 픽셀 이미지 하위 공간인 Spatial Spectrum으로 변환하여 효과적인 학습을 가능하게 합니다.

- **Technical Details**: 연구진은 레이더와 카메라 이미지 간의 유사성을 기반으로 한 스펙트럼 기반 학습을 활용하여 깊이 맵을 생성하는 데이터 기반 접근 방식을 제안합니다. 이 과정에서 비선형 주파수 픽셀 위치 인코딩 알고리즘을 적용하여 4D 레이더 이미지와 카메라 이미지를 공유된 스펙트럼 하위 공간으로 변환합니다. 이를 통해 고해상도 카메라를 사용하여 레이더 깊이 맵 생성기를 효과적으로 훈련할 수 있으며, 오프라인 훈련 후 4D 레이더 모델은 카메라 없이 독립적으로 작동하여 더 선명하고 고밀도의 깊이 맵을 생성합니다.

- **Performance Highlights**: 본 방법은 기존의 최첨단(State-of-the-Art) 기술보다 Unidirectional Chamfer Distance(UCD) 기준으로 27.95% 성능 향상을 이뤘습니다. 이번 연구는 Mean Absolute Error(MAE), Relative Absolute Error(REL)에서도 각각 21.13% 및 7.9%의 감소를 나타내며, 고해상도 스펙트럼 추정과 깊이 맵 생성 실험에서도 뛰어난 결과를 보였습니다. 또한 카메라와 레이더 이미지의 상관 분석 결과, 피어슨 상관 계수와 상호 정보가 각각 3.88배 및 76.69배 증가했습니다.



### PATCH: a deep learning method to assess heterogeneity of artistic practice in historical paintings (https://arxiv.org/abs/2502.01912)
Comments:
          main text: 16 pages, 6 figures; SI: 7 pages, 3 figures

- **What's New**: 이번 연구에서는 기계학습(machine learning) 방법을 활용하여 예술 작품의 창작 과정을 분석하는 새로운 접근 방식을 제안합니다. 연구의 주요 초점은 역사적 작품에 대해 외부 훈련 데이터가 없이도 예술가의 개별적인 작업 양식을 식별하는 것입니다. 이를 통해 과거의 예술적 다양성에 대한 이해를 높이는 방향으로 나아갈 수 있습니다.

- **Technical Details**: 우리는 "pairwise assignment training for classifying heterogeneity (PATCH)"라는 새로운 기계학습 방법을 개발하였습니다. 이 방법은 교차 학습 데이터 없이도 예술적 관행의 이질성을 식별하는 능력을 가지고 있습니다. 결과적으로 이 방법은 기존의 간단한 통계 기법 및 비지도 기계학습 방법보다 우수한 성능을 보였습니다.

- **Performance Highlights**: PATCH 방법을 스페인 르네상스 마스터 El Greco의 두 작품, 'Christ의 세례'와 '풍경이 있는 십자가의 그리스도'에 적용한 결과를 통해, 이전 연구에서 해당 작품을 작업장 구성원에게 할당했던 결론에 이의를 제기하는 새로운 증거를 발견하였습니다. 또한, 이 연구에서의 분석 결과는 시간과 공간을 넘는 예술 작품의 이질성을 특징짓는 척도를 제공합니다.



### Rethinking Homogeneity of Vision and Text Tokens in Large Vision-and-Language Models (https://arxiv.org/abs/2502.01906)
- **What's New**: 이 논문은 Large Vision-and-Language Models (LVLMs)의 시각적 및 텍스트 임베딩 처리를 혁신적으로 발전시킨 Decomposed Attention (D-Attn) 방법을 제안합니다. 전통적으로 시각적 및 텍스트 입력은 동질적인 방식으로 처리되어 왔으나, 이 논문에서는 이 둘의 본질적인 차이를 인정하고 각각의 처리를 달리해야 한다고 주장합니다. D-Attn은 시각적 입력에 대한 인식을 높이고, 효율성을 극대화하면서도 성능은 저하되지 않도록 설계되었습니다.

- **Technical Details**: D-Attn의 핵심은 LVLM 내에서 causal self-attention 메커니즘을 세 가지 구성 요소로 분해하는 것입니다: (1) visual-to-visual self-attention (V2V Self-Attn), (2) textual-to-visual cross-attention (T2V Cross-Attn), (3) textual-to-textual self-attention (T2T Self-Attn). 이 구조를 통해 V2V Self-Attn의 계산 복잡성을 𝒪⁢(|V|2)에서 𝒪⁢(|V|)로 줄이고, T2V Cross-Attn에서 불필요한 positional bias를 제거하는 방법도 제안합니다. 단순한 구조적 변화로 매우 적은 수정으로 기존의 LLM의 능력을 최대한 활용할 수 있게 합니다.

- **Performance Highlights**: D-Attn은 공정한 비교 하에 8배 더 많은 시각적 임베딩을 처리하거나 5배 더 빠른 학습 속도를 보여주며 다양한 이미지 벤치마크에서 일관되게 성능이 개선되었습니다. 이 방법은 기존 self-attention 기법에 비해 뛰어난 성능을 입증하였고, 수많은 실험 결과에서 그 효과성이 검증되었습니다. 코드, 데이터 및 모델은 공개할 예정으로 연구 결과의 재현성을 보장합니다.



### INTACT: Inducing Noise Tolerance through Adversarial Curriculum Training for LiDAR-based Safety-Critical Perception and Autonomy (https://arxiv.org/abs/2502.01896)
- **What's New**: 이번 논문에서는 안전-critical (safety-critical) 인식 작업에서 노이즈가 가득한 LiDAR 데이터에 대한 깊은 신경망의 (DNN) 강건성을 향상시키기 위해 새롭게 제안된 INTACT라는 이중 단계 프레임워크를 소개합니다. INTACT는 메타러닝(meta-learning)과 적대적 커리큘럼 훈련(adversarial curriculum training, ACT)을 결합하여 3D 포인트 클라우드의 데이터 손상 및 희소성 문제를 효과적으로 다룹니다. 이 프레임워크는 실제 시스템에서 자원을 절약하면서도 높은 성능을 유지하는 AI 솔루션으로서의 가능성을 보여줍니다.

- **Technical Details**: INTACT의 메타러닝 단계는 태스크에 구애받지 않는 선험적(prior) 정보를 사용하여 교사 네트워크를 구성하고, 이는 중요한 데이터 영역을 식별하기 위한 강건한 주목 분포(saliency map)를 생성하는 데 도움을 줍니다. 이후 ACT 단계에서는 이러한 주목 분포를 활용하여 학생 네트워크가 점진적으로 복잡한 노이즈 패턴에 노출되도록 하여더 타겟화된 교란을 보장하고 노이즈 저항성을 개선합니다. 이 방식은 DNN의 노이즈 내성을 대폭 향상시키며, LiDAR 데이터의 복잡한 노이즈 특성에 대한 체계적인 접근법을 제공합니다.

- **Performance Highlights**: INTACT는 KITTI, Argoverse, ModelNet40과 같은 다양한 데이터 세트를 활용하여 검증되었습니다. 결과는 INTACT가 모든 작업에서 모델의 강건성을 최대 20% 향상시키며 표준 적대적 훈련 및 커리큘럼 훈련 방법을 초월하는 성과를 보여줍니다. 또한 MOTA는 9.6% 향상되었고(KITTI), Gaussian 노이즈 하에서는 12.4% 향상되었습니다. KITTI의 평균 정밀도(mAP)도 59.8%에서 69.8%로 상승하였으며, 이는 INTACT의 뛰어난 성능을 여실히 보여줍니다.



### SimBEV: A Synthetic Multi-Task Multi-Sensor Driving Data Generation Tool and Datas (https://arxiv.org/abs/2502.01894)
- **What's New**: 이 논문에서는 자율주행을 위한 새롭고 확장 가능한 데이터 생성 도구인 SimBEV를 소개합니다. SimBEV는 다양한 센서에서 수집된 정보를 융합하여 정확한 BEV(새로운 관점으로 본) 기본 데이터를 캡처합니다. 이를 통해 BEV 분할(BEV segmentation) 및 3D 객체 탐지(3D object detection)와 같은 여러 인식 작업을 지원한 새로운 데이터셋인 SimBEV 데이터셋을 생성합니다.

- **Technical Details**: SimBEV는 CARLA 시뮬레이터의 사용자 정의 버전을 기반으로 하며, 사용자에게 원하는 장면 수를 설정하도록 허용합니다. 이 도구는 여러 가지 시뮬레이션 매개변수를 무작위화하여 다양한 주행 시나리오를 신속하게 생성하며, 훈련, 검증 및 테스트 세트에 대해 각각의 장면 수를 구성할 수 있습니다. SimBEV는 각 장면에서 균일하게 분포된 웨이포인트를 생성하고, 이를 기반으로 자율차량과 해당 차량에 부착된 센서를 배치합니다.

- **Performance Highlights**: SimBEV 데이터셋은 다양한 주행 시나리오에서 수집된 주석이 달린 인식 데이터의 대규모 컬렉션을 통해 새로운 기준선을 제공합니다. 이 데이터셋은 다양한 센서와 여러 인식 작업을 지원하여 자율주행 시스템의 성능을 향상시키는 데 기여합니다. SimBEV의 확장성과 구성 가능성 덕분에 연구자들은 복잡한 주행 상황을 보다 효과적으로 모델링할 수 있습니다.



### Geometric Framework for 3D Cell Segmentation Correction (https://arxiv.org/abs/2502.01890)
Comments:
          17 pages, 16 figures

- **What's New**: 이 논문에서는 3D 세포 분할에서 발생하는 과분할(oversegmentation) 문제를 해결하기 위해 지오메트릭(geometric) 정보를 이용한 해석 가능한 프레임워크를 제안합니다. 특히, 2D 분할 결과를 이웃하는 층의 지오메트릭 정보를 기반으로 수정하여 정확한 3D 구조를 재구성하는 방식을 통해 성능을 향상시키고자 합니다. 논문은 미리 학습된(classifier) 모델을 공개 식물 세포 데이터셋에서 개발하고, 이를 동물 세포 데이터셋에 적용하여 효과성을 검증한 점도 주목할 만합니다.

- **Technical Details**: 제안된 프레임워크에서는 earthmover’s distance를 사용하여 세포 마스크의 층 간 지오메트릭 변화를 캡처하고, 라벨이 있는 3D 세그멘테이션에서의 위상(topological) 정보를 통합하여 인접한 세포들이 연결될지 여부를 결정하는 이진 분류 바이너리 클래시파이어(binary classifier)를 훈련합니다. 이를 통해 2D 세그멘테이션 오류를 수정하고 이후 정확한 3D 세포 구조를 재구성합니다. 또한, 비 2D 기반 모델에 대해서도 과분할 교정을 확장하여 적용할 수 있다는 장점을 가지고 있습니다.

- **Performance Highlights**: 논문은 다양한 세그멘테이션 방법과 호환성을 보여주며, 매개변수를 조정할 수 있는 유연성을 제공합니다. 실험 결과는 제안된 방법이 기존의 최신 기법들이 겪는 과분할 이슈를 효과적으로 해결할 수 있음을 보여줍니다. 나아가, 논문은 비슷한 문제에 직면한 다양한 데이터셋에 대해 일반화할 수 있는 가능성을 제시합니다.



### Explaining Automatic Image Assessmen (https://arxiv.org/abs/2502.01873)
- **What's New**: 이 논문에서는 미적 평가(aesthetic assessment)를 설명 가능한 모델로 개선하기 위해 데이터 세트의 경향을 시각화하고 다양한 방식으로 변환된 이미지에서 미적 판단을 자동으로 분류하는 새로운 접근 방식을 제안합니다. 기존의 수동 레이블링과 한정된 데이터 세트의 한계를 극복하기 위해, 신경망(neural networks)을 다양한 변형의 데이터 세트에 훈련시켜 미적 특성을 추출합니다. 이를 통해 미적 특성 및 경향을 포착하고 시각화할 수 있는 가능성을 엽니다.

- **Technical Details**: 본 연구는 각 입력 이미지의 여러 수정된 버전에서 훈련된 모델을 사용하여 미적 성능을 평가합니다. 우리는 깊이(depth) 맵, 주의(saliency) 맵, 흐림(blur) 이미지를 포함한 세 가지 '모달리티(modalities)'를 정의하여, 이들 각각의 정확성을 비교하고 경향을 정량화했습니다. 또한, NIMA(Natural Image Assessment) 모델의 손실 함수로 '총 운반 거리(earth movers distance)'를 채택하여 모델의 미적 품질 예측 能力을 향상시켰습니다.

- **Performance Highlights**: 본 모델은 다양한 변형 이미지에서 미적 특성과 경향을 포착하며, 기존 방법보다 더 나은 설명 가능성과 정확성을 보입니다. 특히, 서버와 같은 고급 하드웨어 없이도 소비자용 하드웨어에서 잘 작동하여 실용성을 높였습니다. 본 연구에 의하면, 잘 훈련된 모델은 이미지의 미적 특성을 잘 이해할 수 있으며, 데이터 기반의 평가 방식으로 기존의 수동 라벨링 문제를 해결할 수 있는 가능성을 제시합니다.



### Reliability-Driven LiDAR-Camera Fusion for Robust 3D Object Detection (https://arxiv.org/abs/2502.01856)
- **What's New**: ReliFusion은 LiDAR-카메라 데이터 융합을 위한 새로운 프레임워크로, 특정 신뢰성 점수에 따라 데이터의 기여도를 동적으로 조정하여 3D 객체 감지의 견고성을 강화합니다. 이 방법은 bird's-eye view(BEV) 공간에서 작동하며, 주요 구성 요소로는 Spatio-Temporal Feature Aggregation(STFA) 모듈, Reliability 모듈, 그리고 Confidence-Weighted Mutual Cross-Attention(CW-MCA) 모듈이 포함됩니다. 실험 결과, ReliFusion은 nuScenes 데이터셋에서 기존 최첨단 기법들보다 우수한 성능을 나타냈습니다.

- **Technical Details**: ReliFusion의 구조는 먼저 LiDAR와 다중 뷰 이미지를 별도로 처리하여 이들로부터 BEV 기반의 특성을 추출합니다. STFA 모듈은 시간에 따른 예측의 안정성을 높이기 위해 여러 프레임간의 의존성을 캡쳐하여 처리합니다. Reliability 모듈은 Cross-Modality Contrastive Learning(CMCL)을 활용하여 각 모달리티의 신뢰성을 평가하는 신뢰성 점수를 생성하며, CW-MCA 모듈은 이러한 점수에 따라 LiDAR와 카메라 데이터의 기여도를 조정합니다.

- **Performance Highlights**: ReliFusion은 제한된 LiDAR 시야 및 센서 고장과 같은 극한 상황에서도 향상된 감지 정확성과 내구성을 보였습니다. 실험에서 ReliFusion은 다양한 센서 감쇠 시나리오에서 뛰어난 성능 개선을 통해 3D 객체 감지의 신뢰성을 효과적으로 강화하는 모습을 보여주었습니다. 이러한 결과는 자율 주행 시나리오에서 센서 신뢰성 문제를 해결하는데 효과적인 접근 방식을 제시합니다.



### Learning Fine-to-Coarse Cuboid Shape Abstraction (https://arxiv.org/abs/2502.01855)
Comments:
          10 pages, 6 figures, 4 tables

- **What's New**: 3D 객체를 단순 기하학적 원시형으로 추상화하는 새로운 방법이 제안되었습니다. 이 방법은 복잡한 지오메트리에서 구조적 정보를 유추할 수 있도록 돕습니다. 연구자는 수백 개의 원시형에서 소수의 원시형으로의 변환을 통해, 전체 데이터 집합에서 일관된 구조를 학습할 수 있는 모델을 개발했습니다.

- **Technical Details**: 제안된 방법은 미세한 재구성과 조잡한 추상화를 통해 훈련 시 원시형의 수를 줄이는 독특한 비지도 학습 접근법을 사용합니다. 또한, 중복된 원시형에 대한 패널티를 증가시키는 추상화 손실 (abstraction loss) 공식을 도입하여 일관성을 유지합니다. 재구성 손실 (reconstruction loss) 공식은 표면 근사뿐만 아니라 부피 보존을 고려하여 3D 형태를 더 정확하게 표현할 수 있게 합니다.

- **Performance Highlights**: 제안된 방법은 인공물 및 인간형 형태의 데이터 집합을 평가하여 이전의 최첨단 방법들과 비교하였습니다. 결과적으로, 기존의 큐보이드 기반 형태 추상화 기술보다 더 나은 성능을 보여주었으며, 클러스터링, 검색, 부분 대칭 탐지와 같은 다운스트림 작업에서도 효과적으로 적용될 수 있음을 입증하였습니다.



### Foundation Model-Based Apple Ripeness and Size Estimation for Selective Harvesting (https://arxiv.org/abs/2502.01850)
- **What's New**: 이 연구는 자동 수확을 위한 새로운 프레임워크를 소개합니다. 이 프레임워크는 사과의 성숙도(ripeness)와 크기(size)를 추정하는 데 중점을 두고 있습니다. 특히, RGB-D 이미지 기반의 후지 사과 데이터셋을 포함하여 총 4,027장의 이미지와 16,257개의 주석이 달린 사과를 활용하여 새로운 기초 모델 기반 시스템을 개발했습니다.

- **Technical Details**: 우리는 Grounding-DINO라는 객체 탐지 모델을 활용하여 사과 탐지 및 성숙도 분류를 수행했습니다. 기존 기술과 비교할 때 더 높은 정확도와 유연성을 보여주었으며, 다섯 가지 다른 크기 추정 알고리즘을 개발 및 평가하여 최적의 성능을 보이는 알고리즘을 선택했습니다. 최종적으로, 이 프레임워크는 복잡한 자연 환경에서도 효과적으로 작동합니다.

- **Performance Highlights**: 우리의 새로운 데이터셋과 알고리즘은 미래의 자동화 및 선택적 수확 연구에 중요한 벤치마크 역할을 할 것입니다. 특히, 작물 수확성과 관련된 성숙도 및 크기 평가 방법을 개선하였고, 이를 통해 농업 로봇의 정확한 작물 탐지 및 수확 결정을 가능하게 했습니다. 이 연구는 노동 집약적인 전통적 수확 방식을 혁신적으로 변화시킬 잠재력을 가지고 있습니다.



### UVGS: Reimagining Unstructured 3D Gaussian Splatting using UV Mapping (https://arxiv.org/abs/2502.01846)
Comments:
this https URL

- **What's New**: 이번 논문에서는 3D Gaussian Splatting (3DGS)의 생성 및 모델링에서의 도전 과제를 해결하기 위해 새로운 방법인 UV Gaussian Splatting (UVGS)를 제안합니다. UVGS는 구형 매핑(spherical mapping)을 활용하여 3DGS를 구조화된 2D 표현으로 변환하며, 이는 위치, 크기, 색상, 불투명도 및 회전을 포함하는 다양한 Gaussian 속성의 다채널 이미지로 볼 수 있습니다.

- **Technical Details**: UVGS는 3D Gaussian primitives를 2D 형식으로 변환하여 그들의 공간 구조를 유지합니다. 그러면서도 Super UVGS라는 3채널의 압축된 표현을 도입해 다양한 속성을 공유 피쳐 공간으로 통합하여, 미리 훈련된 2D 신경망과 쉽게 호환될 수 있도록 합니다. 이는 3DGS의 비구조적 속성을 해결하고, 공간적 일관성을 통해 변환의 용이성을 증가시킵니다.

- **Performance Highlights**: UVGS는 기존 2D 모델들과의 호환성을 통해 3DGS의 다양한 생성 작업을 가능하게 하며, 이전에 복잡했던 비즈니스 요구를 해결합니다. 연구 결과를 통해 우리는 여러 조건부 생성 및 인페인팅(inpainting) 작업을 시연했으며, 이는 특별한 임계값을 넘지 않으면서도 우수한 시각적 결과를 보여줍니다.



### Texture Image Synthesis Using Spatial GAN Based on Vision Transformers (https://arxiv.org/abs/2502.01842)
Comments:
          Published at the 2nd International Conference on Artificial Intelligence and Software Engineering (AI-SOFT), Shiraz University, Shiraz, Iran, 2024

- **What's New**: 이 논문에서는 ViT-SGAN이라는 새로운 하이브리드 모델을 제안합니다. 이 모델은 Vision Transformers (ViTs)와 Spatial Generative Adversarial Network (SGAN)를 결합하여 기존 방법의 한계를 극복합니다. 이전의 전통적인 방법들은 복잡한 텍스처를 생성하는 데 어려움을 겪었지만, ViT-SGAN은 전반적인 텍스처 합성 능력을 향상시킵니다.

- **Technical Details**: ViT-SGAN은 mean-variance (mu, sigma) 및 textons와 같은 전문화된 텍스처 설명자를 ViTs의 self-attention 메커니즘에 통합합니다. 이 접근법을 통해 모델은 복잡한 공간적 의존성을 포착하는 능력이 향상되어 보다 높은 품질의 텍스처를 생성할 수 있습니다. 이는 특히 규칙적인 텍스처와 비정형 텍스처를 처리하는 데 뛰어난 성능을 발휘합니다.

- **Performance Highlights**: ViT-SGAN은 FID, IS, SSIM 및 LPIPS와 같은 다양한 지표를 통해 기존의 최첨단 모델보다 우수한 성능을 보여줍니다. 실험 결과는 ViT-SGAN의 다양한 현실적인 텍스처 생성 능력을 강조하며, 특히 질감의 다양성과 품질에서 개선된 결과를 나타냅니다.



### Low Resource Video Super-resolution using Memory and Residual Deformable Convolutions (https://arxiv.org/abs/2502.01816)
- **What's New**: 본 논문에서는 경량(轻量) 비디오 초해상도(VS 위주) 모델을 제안합니다. 이전 모델들과 달리, 제안된 모델은 잔여(residual) 연결을 통해 특징(feature) 활용도를 높이고, 변형 가능한(deformable) 합성곱(convolution)을 사용하여 프레임 정렬의 정확성을 개선합니다. 또한, 과거 프레임에서 축적된 정보를 포착하기 위해 단일 메모리 텐서(memory tensor)를 도입하여 동적 모션 추정을 향상시킵니다.

- **Technical Details**: 제안하는 모델은 2.3백만 개의 매개변수(parameter)만으로도 높은 SSIM(Structural Similarity Index Measure) 점수인 0.9175를 REDS4 데이터셋에서 기록하였습니다. 이 모델은 이전의 경량 모델들은 물론 많은 무거운 모델들보다 정확성과 리소스 효율성 모두 뛰어난 결과를 보였습니다. 또한, 2D Discrete Wavelet Transform(DWT)을 사용하여 주파수(domain) 영역에서의 특징 추출을 통해 공간(spatial)과 구조(structural) 표현을 개선합니다.

- **Performance Highlights**: 제안한 경량 비디오 초해상도 프레임워크는 무거운 계산 요구사항 없이도 실시간 VSR을 가능하게 합니다. 역사적 프레임에서 정보 수집을 통해 모델의 시간적 일관성(temporal coherence)을 높이고, 복잡한 비디오 시나리오를 효과적으로 처리합니다. 이 연구는 자원 제한이 있는 환경에서의 실제 적용을 위한 결실을 거두었습니다.



### PolyhedronNet: Representation Learning for Polyhedra with Surface-attributed Graph (https://arxiv.org/abs/2502.01814)
- **What's New**: 이번 연구에서는 PolyhedronNet이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 3D 폴리헤드럴 객체의 표현 학습을 위해 설계되었으며, 표면 속성이 부여된 그래프(Surface-Attributed Graph)를 기반으로 합니다. 이를 통해 폴리헤드럴 객체의 정점, 모서리, 면 및 이들 간의 기하학적 관계를 매끄럽게 모델링할 수 있습니다.

- **Technical Details**: Surface-Attributed Graph(SAG)를 사용하는 PolyhedronNet은 면의 의미를 명시적으로 캡처하여 폴리헤드럴 정보의 손실을 방지합니다. 각 지역의 상대적 위치를 효과적으로 학습하기 위해 지역 강체 표현(Local Rigid Representation)으로 SAG를 분해합니다. 이후 PolyhedronGNN을 통해 지역 강체 표현을 계층적으로 집합하여 전역 표현을 생성하고, 이 과정에서 회전 및 변환 불변성을 보장합니다.

- **Performance Highlights**: PolyhedronNet은 4개의 데이터셋에서 실험적으로 검증되었으며, 분류 및 검색 작업에서 뛰어난 성능을 보여주었습니다. 기존의 최첨단 방법들과 비교해 상당한 성능 향상을 기록하며, 3D 폴리헤드럴 객체에 대한 포괄적이고 유용한 표현을 캡처하는 데 성공하였습니다. 연구에 사용된 코드와 데이터는 공개되어 있어 추가 연구가 가능합니다.



### AquaticCLIP: A Vision-Language Foundation Model for Underwater Scene Analysis (https://arxiv.org/abs/2502.01785)
- **What's New**: 본 논문에서는 수생 장면 이해를 위한 새로운 모델인 AquaticCLIP을 소개합니다. AquaticCLIP은 이미지와 텍스트를 정렬하여 세그멘테이션(segment), 분류(classification), 탐지(detection), 객체 counting와 같은 다양한 작업을 수행할 수 있도록 하는 비지도 학습 프레임워크를 제공합니다. 수중 이미지를 위한 대규모 데이터셋을 사용하여 사전 훈련(pre-training)을 통해 기존의 비전-언어 모델들은 수중 환경에서 더 높은 성능을 발휘합니다.

- **Technical Details**: AquaticCLIP은 200만개의 수중 이미지-텍스트 쌍으로 구성된 데이터셋을 활용하여 훈련됩니다. 모델은 프로프트(learnable prompts)를 통해 패치(patch) 특징을 점진적으로 집계하는 비전 인코더(vision encoder)를 사용하며, 시각적 맥락을 통합하는 비전 가이드 언어 인코더로 언어 인코더(language encoder)를 강화합니다. 이들은 대조적 사전 훈련 손실을 통해 시각 및 텍스트 모달리티를 정렬합니다.

- **Performance Highlights**: AquaticCLIP은 다양한 수중 컴퓨터 비전 작업에서 제로샷(zero-shot) 환경에서도 현저한 성능 향상을 달성하였습니다. 모델은 기존의 방법들보다 더 뛰어난 강건성(robustness) 및 해석 가능성(interpretability)을 보여주며, 수중 환경에서 비전-언어 응용 프로그램을 위한 새로운 기준을 제시합니다. 이러한 성과는 AquaticCLIP이 수생 생물 다양성 보존에 기여할 수 있는 가능성을 보여줍니다.



### Sparse VideoGen: Accelerating Video Diffusion Transformers with Spatial-Temporal Sparsity (https://arxiv.org/abs/2502.01776)
Comments:
          13 pages, 8 figures, 3 tables

- **What's New**: 새롭게 제안된 Sparse VideoGen (SVG) 프레임워크는 Diffusion Transformers (DiTs)의 3D Full Attention의 비효율성을 해결하여 비디오 생성의 효율성을 크게 향상시킵니다. 이 방법에서는 주목헤드(attention heads)의 희소성(sparsity)을 활용하여 계산량을 줄이고, 설정된 패턴에 따라 효율적인 프로파일링(online profiling) 전략을 사용하여 성능을 개선합니다. 이를 통해 CogVideoX-v1.5 및 HunyuanVideo에서 최대 2.33배 속도 향상을 달성하면서 제작 품질을 유지합니다.

- **Technical Details**: SVG는 두 가지 종류의 주목헤드, 즉 시간적 주목헤드(Temporal Head)와 공간적 주목헤드(Spatial Head)를 구분하여 각 주목헤드의 적절한 희소성 패턴을 식별합니다. 공간적 주목헤드는 동일한 프레임 내의 토큰에 집중하고 시간적 주목헤드는 모든 프레임에서 동일한 공간적 위치의 토큰에 집중합니다. 이러한 희소성을 활용하기 위해 SVG는 최소한의 오버헤드로 두 가지 서로 다른 희소적인 주목을 처리하며, 최적의 패턴을 선택하는 방법을 제안합니다.

- **Performance Highlights**: SVG는 비디오 생성 품질을 유지하면서 이전 방법보다 우수한 PSNR(Peak Signal-to-Noise Ratio) 성능을 발휘합니다. PSNR 값은 최대 29에 달하며, 이는 품질 저하를 최소화하면서 성능을 극대화하는 것을 의미합니다. 또한, FP8 양자화(quantization)를 지원하여 품질을 저하시키지 않으면서 추가적인 효율성 향상을 이루어냅니다.



### Generating Multi-Image Synthetic Data for Text-to-Image Customization (https://arxiv.org/abs/2502.01720)
Comments:
          Project webpage: this https URL

- **What's New**: 이번 연구에서는 사용자 맞춤형 텍스트-이미지 모델의 한계를 극복하기 위해 새로운 접근 방식을 제안합니다. 기존 방법들은 비싼 테스트 시간 최적화(test-time optimization)나 단일 이미지 훈련 데이터셋에 의존하는 문제점을 지니고 있었습니다. 이를 해결하기 위해, 다양한 조명, 배경 및 포즈에서 동일한 객체의 여러 이미지를 포함하는 고품질 합성 커스터마이제이션 데이터셋(Synthetic Customization Dataset, SynCD)을 생성했습니다.

- **Technical Details**: 연구에서는 새로운 인코더 아키텍처를 도입합니다. 이 아키텍처는 공유 주의 메커니즘(shared attention mechanisms)을 기반으로 하여 입력 이미지의 미세한 시각적 세부 정보를 효과적으로 통합합니다. 또한, 텍스트와 이미지 가이드 벡터의 정규화를 통해 추론(inference) 중 과도 노출(overexposure) 문제를 완화하는 새로운 추론 기법을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 인코더와 추론 알고리즘으로 합성 데이터셋에서 훈련된 모델이 기존의 튜닝이 필요 없는 방법들보다 표준 커스터마이제이션 벤치마크에서 더 나은 성능을 발휘함을 확인했습니다. 이 모델은 다양한 환경에서 사용자 정의 개념을 효율적으로 생성할 수 있는 가능성을 보여줍니다.



### MJ-VIDEO: Fine-Grained Benchmarking and Rewarding Video Preferences in Video Generation (https://arxiv.org/abs/2502.01719)
- **What's New**: 최근 비디오 생성 기술은 텍스트 명령어로부터 비디오를 합성하는 능력이 크게 향상되었습니다. 하지만 기존 모델은 여전히 지침의 불일치, 콘텐츠 환각(content hallucination), 안전성 문제 및 편향과 같은 주요 도전과제에 직면해 있습니다. 이를 해결하기 위해 MJ-BENCH-VIDEO라는 대규모 비디오 선호 벤치마크를 도입하여 비디오 생성을 평가하기 위한 다섯 가지 중요한 측면을 제시합니다.

- **Technical Details**: MJ-BENCH-VIDEO는 Alignment, Safety, Fineness, Coherence & Consistency, Bias & Fairness의 다섯 가지 평가 측면으로 구성되어 있으며, 28개의 세부 기준을 포함하여 비디오 선호도를 종합적으로 평가합니다. 이를 기반으로 MJ-VIDEO라는 Mixture-of-Experts(MoE) 기반의 비디오 보상 모델을 제안하며, 입력된 텍스트-비디오 쌍에 따라 관련 전문가를 동적으로 선택할 수 있는 구조로 되어 있습니다.

- **Performance Highlights**: MJ-BENCH-VIDEO를 통해 기존 비디오 보상 모델의 한계를 분석하고, MJ-VIDEO가 비디오 선호도 평가에서 17.58% 및 15.87%의 개선을 달성하는 등 우수한 성능을 입증했습니다. 또한 MJ-VIDEO를 비디오 생성의 선호 조정(preference tuning)에 도입하여 생성된 비디오의 정렬 성능을 높일 수 있음을 보여주었습니다.



### A Multi-Scale Feature Fusion Framework Integrating Frequency Domain and Cross-View Attention for Dual-View X-ray Security Inspections (https://arxiv.org/abs/2502.01710)
- **What's New**: 본 논문은 이중 보기의 X-ray 보안 검사 이미지를 위한 다중 스케일 상호 작용 특징 융합 프레임워크를 제안합니다. 기존의 단일 보기 장비가 갖는 관점 의존성과 미비한 특징 표현 문제를 해결하기 위한 접근법으로, 여러 모듈을 통해 특징을 효율적으로 융합하고 보완적으로 표현할 수 있도록 합니다.

- **Technical Details**: 이 프레임워크는 세 가지 기본 모듈로 구성됩니다: 주파수 영역 상호 작용 모듈(FDIM)은 푸리에 변환을 통해 주파수 영역의 특징을 추출하며, 다중 스케일 교차 보기 특징 강화(MSCFE)는 크로스뷰 주의 메커니즘을 활용하여 시나리오에서 목표물 식별을 향상시킵니다. 마지막으로 컨볼루션 주의 융합 모듈(CAFM)은 채널 주의와 깊이 분리형 컨볼루션을 통합하여 특징을 효율적으로 융합합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 여러 백본 아키텍처에서 기존의 최첨단 방법론에 비해 탁월한 성능을 보여주었으며, 특히 복잡한 시나리오 및 물체 차폐 상황에서 우수한 감지 정확도를 보여줍니다.



### CLIP-DQA: Blindly Evaluating Dehazed Images from Global and Local Perspectives Using CLIP (https://arxiv.org/abs/2502.01707)
Comments:
          Accepted by ISCAS 2025 (Oral)

- **What's New**: 이번 논문에서는 Blind Dehazed Image Quality Assessment (BDQA) 문제를 해결하기 위해 Contrastive Language-Image Pre-Training (CLIP) 모델을 적응시키는 새로운 접근 방식을 제안합니다. CLIP은 대규모 이미지-텍스트 쌍으로 사전 학습되어 있으며, 이 연구에서는 그것을 BDQA 작업에 맞게 효과적으로 조정합니다. 특히, 인지 시스템의 특성을 반영하여 전역(global) 및 지역(local) 정보를 활용하는 방식으로 개선합니다.

- **Technical Details**: 제안된 방법인 CLIP-DQA는 전통적인 BDQA 방법과는 달리, 사람의 시각적 인지를 기반으로 한 계층적 특징을 활용하여 이미지를 평가합니다. 입력 이미지를 패치(patch)로 분할한 후 각각의 품질 점수를 추정하고 평균 점수를 전체 품질 점수로 사용합니다. CLIP의 두 가지 브랜치(vision branch와 language branch)를 조정하여 입력된 계층적 정보를 품질 점수로 정확하게 매핑하는 기법을 적용합니다.

- **Performance Highlights**: 실험 결과, CLIP-DQA는 기존 BDQA 방법에 비해 더 정확한 품질 예측을 보여주었습니다. 두 개의 실제 DQA 데이터셋에서 평가하였으며, 제안된 방법은 여러 관련 연구에 비해 높은 성능을 입증하였습니다. 이 연구의 결과는 향후 이미지 디헤이징 알고리즘의 평가 및 최적화를 위한 중요한 기초 자료로 활용될 수 있습니다.



### HuViDPO:Enhancing Video Generation through Direct Preference Optimization for Human-Centric Alignmen (https://arxiv.org/abs/2502.01690)
- **What's New**: 본 연구에서는 Direct Preference Optimization (DPO) 전략을 텍스트-비디오 생성(T2V) 작업에 최초로 도입하였습니다. 이를 통해 생성된 비디오가 인간의 선호도에 맞춰져 효율적으로 조정될 수 있도록 손실 함수를 유도했습니다. 새로운 방법인 HuViDPO는 인간의 피드백을 활용하여 비디오 생성의 품질과 미학적 연관성을 향상시키는 데 기여하였습니다.

- **Technical Details**: T2V 작업에 DPO 전략을 적용하기 위해, 연구진은 새로운 형태의 손실 함수를 철저히 유도하였습니다. 이 손실 함수는 강화학습 기반의 인간 피드백을 기반으로 하며, 모델이 별도의 보상 모델 없이도 인간 선호도에 맞는 비디오를 생성할 수 있도록 합니다. 추가적으로 각 행동 카테고리마다 소규모 인간 선호 데이터셋을 구축하고 이를 통해 모델을 세밀하게 조정하여 생성된 비디오의 미학적 품질을 향상시켰습니다.

- **Performance Highlights**: HuViDPO는 총 8개의 행동 카테고리에서 평가되었습니다. 실험 결과, HuViDPO는 각 행동 카테고리에 맞는 비디오를 생성하고 다양한 스타일의 비디오 생성 작업에도 효과적으로 전이할 수 있는 능력을 보여주었습니다. 다른 기준선 모델에 비해, 생성된 비디오는 인간의 미적 선호도와 더 가깝게 정렬되었습니다.



### Semantic Communication based on Generative AI: A New Approach to Image Compression and Edge Optimization (https://arxiv.org/abs/2502.01675)
Comments:
          PhD thesis

- **What's New**: 이 논문은 지능형 장치가 생성하는 방대한 데이터를 처리하는 데 있어 커뮤니케이션 네트워크가 직면한 도전 과제를 다루고 있습니다. 본 연구는 의미 기반 커뮤니케이션(semantic communication)과 생성 모델(generative models)을 통합하여 이미지 압축(image compression) 및 엣지 네트워크 자원 할당(edge network resource allocation)의 최적화를 이루었습니다. 이는 단순히 비트 중심 시스템에서 벗어나 특정 의미를 전달하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 이 연구의 핵심은 생성적 적대 신경망(Generative Adversarial Networks)과 노이즈 제거 확산 확률 모델(Denoising Diffusion Probabilistic Models)을 사용한 의미 보존 이미지 압축(semantic-preserving image compression)의 설계입니다. 이러한 모델들은 오직 의미와 관련된 특징들만 인코딩하여 이미지를 압축하고, 최소한의 전송으로 고품질 재구성을 가능하게 합니다. 또한, 정보 병목 원칙(Information Bottleneck principle)과 확률적 최적화(stochastic optimization)를 이용하여 자원을 동적으로 할당하고 효율성을 높이는 목표 지향 엣지 네트워크 최적화 프레임워크를 도입합니다.

- **Performance Highlights**: 성능 비교에서는 의미 인식 모델과 전통적인 이미지 압축 기법을 고전적 및 의미 평가 메트릭(classical and semantic evaluation metrics)을 사용하여 비교했습니다. 결과는 생성 AI와 의미 기반 커뮤니케이션의 결합이 현대의 데이터 중심 애플리케이션 요구사항을 충족하는 보다 효율적인 의미 목표 지향 커뮤니케이션 네트워크를 만드는 가능성을 보여줍니다. 이러한 접근 방식은 실시간 애플리케이션에 적합한 컴퓨팅 효율성과 커뮤니케이션 효과성을 균형 있게 제공합니다.



### Leveraging Stable Diffusion for Monocular Depth Estimation via Image Semantic Encoding (https://arxiv.org/abs/2502.01666)
- **What's New**: 이번 연구에서는 기존의 CLIP 모델 대신 SeeCoder라는 새로운 이미지를 기반으로 한 시맨틱 임베딩을 제안합니다. SeeCoder는 텍스트 설명 없이 직접 이미지에서 맥락 정보를 추출하여, 복잡한 환경에서도 깊이 예측 성능을 향상시킵니다. 또한, 우리는 KITTI 및 Waymo 데이터세트에서의 깊이 추정 과정을 통해 SeeCoder의 효과성과 일반화 능력을 입증했습니다.

- **Technical Details**: 제안된 프레임워크는 네 가지 주요 구성 요소로 이루어져 있습니다: 잠재적 특징 추출기, 공간적으로 강화된 시맨틱 인코더(SeeCoder), 디노이징 UNet, 그리고 작업 특정 디코더입니다. 입력된 RGB 이미지는 두 개의 병렬 경로를 통해 처리되며, 시맨틱 인코더는 맥락 정보를 위한 고급 임베딩을 생성합니다. 이 특징들은 디노이징 UNet에 통합되어 시각적 및 시맨틱 단서를 결합하여 깊이 맵을 재구성합니다.

- **Performance Highlights**: 우리의 방법은 KITTI 및 Waymo 데이터셋에서 평가되었으며, 최신 모델과 비교할 수 있는 성능을 달성했습니다. 특히, 아웃도어 장면에서 CLIP 임베딩의 단점을 극복하는 데 성공하였고, 깊이 추정 작업에서 모든 조건에서 더욱 강력하고 적응력 있는 성능을 보여주었습니다. 이는 다른 시각 인식 작업에도 적용될 수 있는 가능성을 제시합니다.



### FruitPAL: An IoT-Enabled Framework for Automatic Monitoring of Fruit Consumption in Smart Healthcar (https://arxiv.org/abs/2502.01643)
Comments:
          22 Pages, 17 Figures, 5 Tables

- **What's New**: 이 논문은 FruitPAL과 그 업그레이드 모델인 FruitPAL 2.0의 두 가지 자동화 장치를 소개합니다. 이 장치들은 안전한 과일 소비를 촉진하고 건강 위험을 줄이는 것을 목표로 합니다. 특히 YOLOv8 및 YOLOv5 V6.0 모델을 활용하여 과일의 알레르기를 감지하고 영양 정보를 제공합니다.

- **Technical Details**: FruitPAL 장치는 실시간으로 알레르기를 일으킬 수 있는 과일을 감지하며, 긴급 알림을 통해 간병인에게 즉각적인 알림을 전송합니다. FruitPAL 2.0은 15종의 과일을 분류할 수 있는 기능을 갖추고 있으며, 사용자의 식단에 대한 유용한 통찰을 제공합니다. 두 시스템 모두 Cloud를 이용하여 커뮤니케이션을 관리하며, GSM을 통해 사용자에게 정보를 전달합니다.

- **Performance Highlights**: FruitPAL은 YOLOv8 모델 덕분에 빠른 반응시간과 높은 정확도를 자랑하며, FruitPAL 2.0은 개선된 측정과 분석 기능으로 사용자의 fruit intake를 적극적으로 모니터링합니다. 이러한 장치는 정부 및 의료계에서 더 나은 공중 보건과 식단 인식을 위한 혁신적인 솔루션으로 자리 잡을 것입니다.



### Learning the RoPEs: Better 2D and 3D Position Encodings with STRING (https://arxiv.org/abs/2502.02562)
Comments:
          Videos of STRING-based robotics controllers can be found here: this https URL

- **What's New**: 본 논문은 STRING: Separable Translationally Invariant Position Encodings를 소개합니다. STRING은 최근에 제안된 Rotary Position Encodings를 확장하여, 일반적인 이론 프레임워크를 통해 접근합니다. STRING은 임의 차원의 토큰 좌표에 대해서도 정확한 translation invariance를 제공하며, 이는 로봇 공학 분야에서 효율적인 3D 토큰 표현을 가능하게 합니다.

- **Technical Details**: STRING은 Lie 그룹에 기반하여 설계되었습니다. 이는 이전의 RoPE를 통합하는 일반화된 알고리즘으로, 두 개의 쿼리와 키의 상대 회전 각도가 오직 그들의 위치 차이에만 의존합니다. 이러한 독립적인 변환은 트랜스포머의 반복 처리 중 다시 계산할 필요가 없어 KV-caching을 용이하게 하며, 이를 통해 성능을 더욱 향상시킵니다.

- **Performance Highlights**: STRING은 RGB(-D) 입력을 사용하는 비전 트랜스포머(Vision Transformers)에 성공적으로 통합되었습니다. 이를 통해 open-vocabulary object detection과 다양한 로봇 컨트롤러에서 실질적인 성과를 보여줍니다. 실험 결과는 STRING이 로봇 공학 분야에서의 실제 활용 가능성을 강조합니다.



### Particle Trajectory Representation Learning with Masked Point Modeling (https://arxiv.org/abs/2502.02558)
Comments:
          24 pages, 15 figures. Project page at this https URL

- **What's New**: 본 연구는 Self-Supervised Learning (SSL) 프레임워크를 활용하여 3D 입자 궤적 분석을 위한 마스킹 모델링 기법을 제안합니다. 이 방법은 TPC(Time Projection Chamber)에서 생성되는 희박한 포인트 구름을 해석하는 데 중점을 두고, 점 기반의 Liquid Argon Masked Autoencoder(PoLAr-MAE)를 통해 교훈적이지 않은 데이터로부터 파생된 지식을 효과적으로 학습합니다.

- **Technical Details**: 이를 위해 연구자들은 포인트를 그룹화하는 볼륨 토큰화(volumetric tokenization) 기법과 궤적의 의미론을 개선하는 보조 에너지 보간 작업을 도입했습니다. 연구 결과에 따르면, PoLAr-MAE는 궤적과 샤워 분류에서 각각 99.4% 및 97.7%의 F-스코어를 달성했으며, 이는 레이블이 있는 데이터 없이도 감독 학습(Supervised Learning) 기법과 유사한 성능입니다.

- **Performance Highlights**: 이 연구는 다양한 고에너지 물리학 실험에서 SSL의 가능성을 제시하며, 대규모의 레이블이 없는 원시 TPC 데이터 세트를 활용하는 방법을 개척합니다. 또한, 1,000,000개 이상의 이벤트와 52억 개의 레이블이 달린 에너지 침착 포인트를 포함하는 PILArNet-M이라는 가장 큰 공개 LArTPC 데이터세트를 제작하여 향후 연구에 활용할 수 있도록 기여하고 있습니다.



### AAD-DCE: An Aggregated Multimodal Attention Mechanism for Early and Late Dynamic Contrast Enhanced Prostate MRI Synthesis (https://arxiv.org/abs/2502.02555)
- **What's New**: 본 논문에서는 DCE-MRI(Dynamic Contrast-Enhanced Magnetic Resonance Imaging)의 촬영 과정에서 Gadolinium 기반 조영제를 사용하지 않고, 다양한 비대조 멀티모달 MRI 이미지를 통해 DCE-MRI 이미지를 합성하는 새로운 방법인 AAD-DCE를 제안합니다. AAD-DCE는 글로벌 및 로컬 분류 모듈로 구성된 생성적 적대 신경망(GAN)으로, 이는 조영제를 사용하지 않고도 프로스테이트 이미지를 향상시키는 데 도움을 줍니다. 추가적으로, 주목(attention) 모듈을 도입하여 이미지를 합성하고자 하는 지역에 집중할 수 있는 기반을 제공합니다.

- **Technical Details**: AAD-DCE는 3가지 입력 이미지(T2 Weighted, Apparent Diffusion Coefficient, T1 Pre-contrast)를 결합하여 생성된 이미지를 역전파하는 과정에서 aggregated attention map을 사용합니다. 여기에서 두 개의 구별자가 글로벌 및 로컬 정보를 처리하며, 이를 통해 생성기가 주요 정보에 더 집중하도록 유도합니다. 이러한 접근 방식은 DCE-MRI 데이터에서 혈관 생리학적 정보(perfusion information)를 보다 효율적으로 활용하게 합니다.

- **Performance Highlights**: ProstateX 데이터셋에서 실시한 실험 결과, AAD-DCE는 기존 DCE-MRI 합성 방법들과 비교하여 +0.64 dB PSNR, +0.0518 SSIM, -0.015 MAE의 개선된 성능을 보여줍니다. 늦은 반응에 대한 결과도 +0.1 dB PSNR, +0.0424 SSIM, -0.021 MAE의 성능 개선을 기록했습니다. 실험을 통해 주목 강화를 통한 정보 중요성을 강조하며, 병리적 영역의 세부 표현을 효과적으로 개선하는 데 기여하고 있음을 알 수 있습니다.



### The Skin Game: Revolutionizing Standards for AI Dermatology Model Comparison (https://arxiv.org/abs/2502.02500)
Comments:
          60 pages, 69 figures

- **What's New**: 이 논문은 피부 질환 분류 연구의 현재 방법론을 체계적으로 분석하여 데이터 준비, 증가 전략 및 성과 보고에서 상당한 불일치를 보여줍니다. 두 가지 중요한 기여로, 첫째는 피부 질환 분류를 위한 종합적인 훈련 및 평가 프레임워크를 제시하는 것입니다. 둘째, DINOv2-Large 비전 변환기를 활용하여 3개의 벤치마크 데이터셋(HAM10000, DermNet, ISIC Atlas)에서 실험을 통해 이 프레임워크의 실행 가능성을 입증했습니다.

- **Technical Details**: 실험 결과, DINOv2 모델은 피부 질환 분류에서 HAM10000 데이터셋에서 매크로 평균 F1-score 0.85, DermNet에서 0.71, ISIC Atlas에서 0.84를 기록하며 높은 성능을 보였습니다. 또한, Attention map 분석을 통해 모델 결정 과정에서의 중요한 패턴을 드러내었으며, 전형적인 사례에서 정교한 특징 인식을 보였으나 비정형 사례와 복합 이미지에서 상당한 취약성을 보였습니다.

- **Performance Highlights**: 이 연구는 피부 질환 분류에 대한 표준화된 평가 프로토콜의 필요성을 강조하며, 모델 개발, 평가 및 임상 배치에 대한 포괄적인 방법론적 권장 사항을 제안합니다. 데이터 준비에 대한 엄격한 요구, 체계적인 오류 분석, 다양한 이미지 유형을 위한 전문 프로토콜이 포함됩니다. 또한, 재현성을 촉진하기 위해 구현 코드를 GitHub를 통해 제공하며, 이는 임상 피부과에서의 책임감 있는 AI 구현을 위한 기초를 마련합니다.



### Style transfer as data augmentation: evaluating unpaired image-to-image translation models in mammography (https://arxiv.org/abs/2502.02475)
- **What's New**: 이번 논문에서는 유방암 진단에 사용되는 디지털 맘모그램(mammograms)에서 딥러닝 모델의 일반화(generalisability)를 향상시키기 위한 다양한 방법에 대해 다룹니다. 모델의 오버피팅(overfitting) 문제와 데이터 도메인(data domain)의 차이로 인해, 한 환자 집단에서 훈련된 모델이 다른 집단에 잘 적용되지 않는다는 문제를 제기합니다. 이미지-투-이미지 변환(image-to-image translation) 모델을 통해 이러한 문제를 해결할 수 있는 가능성이 논의됩니다.

- **Technical Details**: 논문에서는 CycleGAN(사이클 생성적 적대 신경망)과 SynDiff(확산 기반 모델)이라는 두 가지 생성 모델을 사용하여 비자동 이미지 간의 변환을 시도합니다. 맘모그램 데이터셋으로는 VinDr Mammo, 중국 맘모그래피 데이터베이스, DDSM을 이용하며, 형태학적(geometric) 변환 외에도 딥러닝 신경망을 통해 도메인-특정(domain-specific) 변환을 수행합니다. 모델의 스타일 전이(style transfer) 성능을 평가하기 위해 여러 가지 지표(metrics)를 사용할 예정이며, 특히 다양한 평가 지표들이 어떻게 각각의 모델 성능을 다르게 측정하는지에 대해 탐구합니다.

- **Performance Highlights**: 제안된 모델은 적응된 소스 도메인(source domain)과 타겟 도메인(target domain) 이미지 간의 특징(feature) 분포를 비교하여 스타일 이식을 평가합니다. Fréchet Inception Distance (FID)와 같은 방법들이 이미지의 활성화(activation)를 활용해 유사성을 측정하는 데 사용될 예정입니다. 연구 결과는 다양한 시스템에서 스타일 전이 모델이 어떤 성능을 발휘하는지를 명확히 하고, 여러 지표를 종합적으로 활용해야 한다는 점을 강조합니다.



### SAISA: Towards Multimodal Large Language Models with Both Training and Inference Efficiency (https://arxiv.org/abs/2502.02458)
- **What's New**: 최근 연구에서 제안된 NAAViT (No Attention Among Visual Tokens)와 SAISA (Self-Attention Input Space Alignment)는 두 가지 서로 다른 멀티모달 대형 언어 모델 아키텍처의 효율성을 개선하기 위한 새로운 접근법을 제공합니다. 기존 아키텍처들 간의 주목할 만한 차이점은 시각적 토큰 간의 상호작용 방식에 있으며, 이 연구는 시각적 토큰 간의 주목(attention) 적용을 최소화하여 훈련 및 추론 효율성을 높이는 방법을 제안합니다.

- **Technical Details**: 이 연구는 LLaVA-1.5 및 Flamingo와 같은 두 가지 아키텍처를 분석하여 시각적 및 텍스트 모달리티 간의 정렬 방식이 어떻게 훈련 및 추론 효율성에 영향을 미치는지를 밝혀냅니다. NAAViT은 시각적 토큰 간의 주목을 제거하여 계산 비용을 줄이며, SAISA는 이러한 NAAViT 블록에 시각적 특징을 직접 정렬하여 자기 주목과 피드포워드 네트워크(FFN)에서의 계산 오버헤드를 줄입니다.

- **Performance Highlights**: SAISA는 LLaVA-1.5와 동일한 설정을 사용하여 훈련 예산을 26% 줄이고 추론 FLOPs를 66% 감소시키면서도 성능이 우수하다는 결과를 보여줍니다. 포괄적인 절제 실험(ablative studies)을 통해 다양한 LLMs 및 시각 인코더에서 SAISA의 효율성과 효과성을 검증했습니다. 코드와 모델은 추후 공개될 예정입니다.



### Field Matching: an Electrostatic Paradigm to Generate and Transfer Data (https://arxiv.org/abs/2502.02367)
- **What's New**: 본 논문에서는 Electrostatic Field Matching (EFM)이라는 새로운 방법을 제안합니다. 이 방법은 생성 모델링(generative modeling)과 분포 전이(distribution transfer) 작업 모두에 적합합니다. EFM은 전기 축전기의 물리학에 영감을 받아 개발되었습니다.

- **Technical Details**: EFM은 전하가 있는 두 판을 통하여 데이터를 전이하는 원리를 기반으로 합니다. 우리는 이 판에 소스(source)와 타겟(target) 분포를 배치하고 각각에 긍정적(positive)과 부정적(negative) 전하를 부여합니다. 전기장을 뉴럴 네트워크(neural network)를 통해 학습하여, 이 전기장 선을 따라 샘플을 이동시켜 변환을 수행합니다.

- **Performance Highlights**: 이 방법은 다양한 실험에서 성능을 입증하였습니다. toy 데이터와 이미지 데이터 실험을 통해 EFM의 효과성을 확인하였으며, 저차원 및 고차원 생성 모델링 작업에서 개념 증명을 위한 실험을 수행하였습니다. EFM은 노이즈-데이터(nnoise-to-data) 및 데이터-데이터(data-to-data) 생성 작업 모두에 적용될 수 있습니다.



### Test Time Training for 4D Medical Image Interpolation (https://arxiv.org/abs/2502.02341)
- **What's New**: 본 논문에서는 4D 의료 이미지 보간(4D medical image interpolation)을 위한 새로운 테스트 시간 훈련 프레임워크(TTT4MII)를 제안합니다. 이 프레임워크는 라벨이 없는 상태에서 모델이 새로운 분포에 적응할 수 있도록 자기 지도(self-supervised) 학습을 활용합니다. 기존 연구에서 간과된 분포 이동(distribution shifts) 문제를 해결하여, 모델이 실시간으로 적응할 수 있는 시스템을 제공합니다.

- **Technical Details**: TTT4MII는 두 가지 자기 지도 작업인 회전 예측(rotation prediction)과 이미지 재구성(image reconstruction)을 통합하여 설계되었습니다. 이는 4D 의료 이미지의 보간 정확성을 높이는 데 기여하며, 테스트 데이터가 라벨 없이도 모델이 학습할 수 있도록 합니다. 또한, 단순 TTT(Naïve TTT), 온라인 TTT(Online TTT), 미니 배치 TTT(Mini-batch TTT) 등 세 가지 TTT 기법을 적용하여 성능 향상을 분석합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 Cardiac 데이터셋에서 33.73dB, 4D-Lung 데이터셋에서 34.02dB의 피크 신호 대 잡음비(peak signal-to-noise ratio)를 달성하며, 기존의 다양한 평가 지표에서 상당한 향상을 보였습니다. 이 연구는 4D 의료 이미지 보간을 발전시키는 데 기여할 뿐만 아니라, 이미지 분할(image segmentation) 및 이미지 등록(image registration) 등 다른 분야에서도 활용 가능한 템플릿을 제공합니다.



### Deep Ensemble approach for Enhancing Brain Tumor Segmentation in Resource-Limited Settings (https://arxiv.org/abs/2502.02179)
- **What's New**: 이 논문은 Sub-Saharan Africa 지역의 뇌 종양 세분화(segmentation) 문제를 해결하기 위한 혁신적인 접근법을 제시합니다. 깊은 학습(deep learning) 기반의 앙상블 모델을 사용하여 UNet3D, V-Net, MSA-VNet을 통합하였으며, BraTS-GLI 데이터 세트에서 훈련된 후 BraTS-SSA 데이터 세트로 세밀 조정하였습니다. 이 연구는 자원이 제한된 환경에서도 신뢰성 있는 자동화된 뇌 종양 세분화의 가능성을 강조합니다.

- **Technical Details**: BraTS-Africa 데이터 세트는 60명의 아프리카 환자로부터 수집된 MRI 이미지를 포함하며, T1 및 T2 가중치 이미지를 사용하여 세분화를 수행합니다. 이 데이터 세트는 BraTS 표준에 따라 엄격히 주석이 달리고, nnU-Net 모델을 활용하여 초기 자동 세분화를 생성한 후 경험이 풍부한 방사선 전문의에 의해 수동으로 정제되었습니다. 또한 Z 점수 정규화(Z-score normalization)를 적용하여 데이터의 강도 변화를 관리하며, 2D U-Net 아키텍처를 확장한 UNet3D와 유사한 V-Net 아키텍처를 사용하여 보다 정밀한 세분화를 구현합니다.

- **Performance Highlights**: 연구 결과, 세분화의 정확성이 현저히 개선되었습니다. Tumor Core, Whole Tumor, Enhancing Tumor에 대한 DICE 점수가 각각 0.8358, 0.8521, 0.8167에 달하는 것으로 나타났습니다. 이 성과는 다양한 모델을 통합한 앙상블 접근법의 효용성을 입증하며, 자원이 제한된 환경에서도 자동화된 뇌 종양 세분화의 신뢰성을 향상시킬 수 있는 가능성을 제시합니다.



### VLA-Cache: Towards Efficient Vision-Language-Action Model via Adaptive Token Caching in Robotic Manipulation (https://arxiv.org/abs/2502.02175)
- **What's New**: 본 논문에서는 Vision-Language-Action (VLA) 모델의 효율성을 개선하기 위해 VLA-Cache라는 새로운 접근 방식을 제안합니다. VLA-Cache는 이전 단계와 비교하여 변경이 최소화된 시각적 토큰을 적응적으로 식별하고 이들의 계산 결과를 재사용하여 전체 모델의 효율성을 향상시킵니다. 선언된 VLA-Cache 모델은 실제 로봇 상에서의 테스트 결과를 통해 성능을 입증하였습니다.

- **Technical Details**: VLA-Cache는 각 단계에서 입력 시각적 데이터를 비교하여 변화가 적은 토큰을 식별하는 토큰 선택 메커니즘을 포함합니다. 이 모델은 KV-cache를 사용하여 변경되지 않은 토큰에 대한 계산 결과를 재사용하며, 이는 반복적 계산을 줄이고 추론 효율성을 극대화합니다. 또한, 레이어 내에서의 주의 집중도를 기반으로 동적으로 재사용할 토큰의 비율을 조정하는 레이어 적응 전략도 제안합니다.

- **Performance Highlights**: VLA-Cache는 LIBERO 벤치마크에서 1.7배의 가속화를 달성하며, 성공률의 약간의 저하만을 수반하고 있습니다. 또한, Kinova Jaco 로봇 팔에 배치하여 실제 환경에서도 실용적인 가속화를 보여주어, 다양한 로봇 조작 작업에서의 적용 가능성을 높였습니다.



### EditIQ: Automated Cinematic Editing of Static Wide-Angle Videos via Dialogue Interpretation and Saliency Cues (https://arxiv.org/abs/2502.02172)
Comments:
          Accepted at 30th International Conference on Intelligent User Interfaces (IUI 25)

- **What's New**: 이 논문에서 제시하는 EditIQ는 고정형, 넓은 시야각을 가진 고해상도 카메라를 통해 캡처된 장면을 시네마틱하게 편집하는 완전 자동화된 프레임워크입니다. EditIQ는 static camera feed로부터 여러 개의 가상 피드를 생성하여 편집 과정을 간소화하며, 이를 통해 시청자에게 향상된 장면 내용을 전달합니다. 또한, 이 프레임워크는 대화 흐름을 분석하는 대형 언어 모델(LLM) 기반의 대화 이해 모듈과 시각적 중요도를 예측하여 핵심 장면 요소를 파악합니다.

- **Technical Details**: EditIQ는 비디오 편집을 에너지 최소화 문제로 계량화하여, 장면의 중요한 요소들에 대한 정보를 바탕으로 카메라 샷 선택을 수행합니다. 이는 비디오의 모든 순간에서 최적의 rushes를 선택하기 위해, 제시된 사람의 정보, LLM 예측 및 시각적 중요도의 출력을 결합하여 이루어집니다. 이러한 최적화 과정은 동적 프로그래밍을 활용하여 해결됩니다.

- **Performance Highlights**: EditIQ의 효과는 20명을 대상으로 한 심리 물리적 연구를 통해 검증되었으며, BBC Old School 데이터 세트에서의 여러 편집된 성능을 비교한 결과로 입증되었습니다. 결과적으로 EditIQ는 전문가의 편집에 근접한 품질의 출력을 달성했으며, 사용자는 감정 표현, 행동 유지, 전반적인 시청 경험 측면에서 EditIQ의 편집을 선호하였습니다.



### BRIDLE: Generalized Self-supervised Learning with Quantization (https://arxiv.org/abs/2502.02118)
- **What's New**: 이번 연구에서는 자가 지도 학습(Self-supervised Learning, SSL) 접근 방식을 통해 오디오, 이미지 및 비디오 처리에서의 제약을 해결하는 새로운 프레임워크인 BRIDLE(Bidirectional Residual Quantization Interleaved Discrete Learning Encoder)를 제안합니다. BRIDLE는 잔여 양자화(Residual Quantization, RQ)를 통합하여 다계층 코드북을 사용함으로써 보다 세밀한 표현을 가능하게 합니다. 이 방법은 기존의 인코더와 토크나이저 간의 상호 훈련 절차를 포함하여 성능을 크게 향상합니다.

- **Technical Details**: BRIDLE는 인코더와 토크나이저 간의 비대칭 훈련 과정을 통해 오디오 신호의 정밀한 양자화 및 표현을 제공합니다. 잔여 양자화를 사용으로, 신호의 세밀한 리얼타임 처리가 가능하며 이를 통해 오디오, 이미지 및 비디오 임무에서의 학습 효과를 증대시킵니다. 교육 과정에서 코드북을 효율적으로 활용하여, 학습 시간과 과적합 문제를 줄이는 데 기여합니다.

- **Performance Highlights**: BRIDLE는 AudioSet 및 ESC-50 벤치마크에서 최신의 분류 정확도를 기록하였고, ImageNet-1K 및 Kinetics-400 데이터셋에서 영상 분류에도 경쟁력 있는 성능을 보였습니다. RQ 기법을 사용할 경우, 기존 VQ 방법에 비해 다운스트림 성능에서 일관된 개선을 보여주어, 다양한 데이터 모달리티에서의 효용성을 입증하고 있습니다.



### Position Paper: Building Trust in Synthetic Data for Clinical AI (https://arxiv.org/abs/2502.02076)
Comments:
          7 pages, 8 figures (including sub-figures)

- **What's New**: 최근 의료 분야에서 Synthetic Medical Data(합성 의료 데이터)의 사용이 증가하고 있다. 이는 데이터 부족, 개인 정보 보호 문제, 클래스 불균형 등의 문제를 극복하는 혁신적인 솔루션으로 주목받고 있다. 그러나 이러한 데이터의 신뢰성과 신용에 대한 의문이 여전히 존재하여 임상 환경에서의 실제 채택은 제한적이다.

- **Technical Details**: 이 논문에서는 다양한 MRI 스캔을 사용하여 Brain Tumor Segmentation(뇌 종양 분할) 연구를 통해 합성 데이터의 품질, 다양성 및 비율이 어떻게 AI 모델의 신뢰성에 영향을 미치는지를 조사하였다. BraTS 2021 데이터셋을 기반으로 합성 이미지 생성에 활용된 Generative 모델인 Med-DDPM을 적용하여 실제 이미지와의 일대일 대응 관계를 보장하였다. 그렇게 생성된 합성 데이터는 기계 학습 모델 훈련 시 중요하게 다룰 요소가 된다.

- **Performance Highlights**: Empirical studies(경험적 연구) 결과, 합성 데이터의 품질이 높을수록 AI 모델에 대한 신뢰도가 증가하는 것을 확인하였다. 나아가 본 연구는 Synthetic Data(합성 데이터)의 임상적 채택에 있어 신뢰 구축이 얼마나 중요한지를 강조하며, 명확하게 정의된 프레임워크를 통해 합성 데이터가 신뢰성 있게 통합될 수 있는 방법을 모색하였다. 이러한 발견은 의료 AI 시스템의 실제 적용 가능성을 높이는 데 기여할 수 있다.



### RAPID: Robust and Agile Planner Using Inverse Reinforcement Learning for Vision-Based Drone Navigation (https://arxiv.org/abs/2502.02054)
Comments:
          18 pages, 11 figures, 58 references, and appendix is included

- **What's New**: 이번 논문에서는 복잡한 환경에서 민첩한 드론 비행을 위한 학습 기반 시각 계획 시스템인 RAPID를 소개합니다. 이 시스템은 인버스 강화 학습(ILR)에 기반하여 충돌이 없는 경로를 수 밀리초 안에 생성하며, 별도의 인식, 매핑 및 계획 모듈 없이 작동할 수 있습니다. 또한, 제안된 방법은 시뮬레이션 환경에서만 학습되어도 실세계에 직접 적용 가능하다는 점에서 참신합니다.

- **Technical Details**: RAPID는 고속 시각 내비게이션을 위한 인버스 소프트 Q-학습 기반 프레임워크를 개발하였으며, 수동 보상이 필요 없는 견고한 학습을 달성합니다. 이를 위해 고차원 시각 정보에서 복잡성을 줄이는 보조 오토 인코더 손실 함수를 도입하였으며, 고속 시나리오에 맞춰 흡수 상태 처리를 통합했습니다. 또한, 시뮬레이션에서 학습된 정책은 평균 7 m/s의 속도로 잘 수행되었습니다.

- **Performance Highlights**: 제안된 방법의 성능은 숲과 다양한 구조물에서는 물론, 다양한 실제 환경에서 검증되었습니다. RAPID의 학습된 정책은 최고 8.8 m/s의 최대 속도로 비행 실험에서 안정적인 성과를 보여주었습니다. 이 연구는 드론의 고속 시각 내비게이션을 위한 첫 번째 IRL 기반 접근 방식을 제시하며, 실제 환경에서도 수행 가능성을 입증했습니다.



### Efficient Domain Adaptation of Multimodal Embeddings using Constrastive Learning (https://arxiv.org/abs/2502.02048)
- **What's New**: 최근의 기계 학습(ML), 자연어 처리(NLP), 그리고 기초 모델들(foundation models)의 발전은 헬스케어와 같은 컴퓨팅 자원이 제한된 분야에서 실제 응용 가능성을 보여주고 있습니다. 이러한 분야에서 기초 모델과 감독 기계 학습(supervised ML)의 결합은 진단 및 치료 계획과 같은 업무를 자동화할 수 있는 잠재력을 제공합니다. 그러나 이러한 기술을 효과적으로 적용하기에는 현장 computational resource의 제한이 큰 도전 과제가 되고 있습니다.

- **Technical Details**: 우리의 접근 방식은 기초 모델을 fine-tuning 하지 않고 downstream task에 embedding을 효율적으로 적응시키는 방법을 제안합니다. 이 방법은 Large Language Models (LLMs)와 Vision Models의 frozen embeddings를 활용하고, 대조 학습(contrastive learning)을 통해 작은 비선형 모델을 훈련시킵니다. 이는 각 embedding을 새로운 task-specific 하위 공간(subspace)으로 매핑하여 동일한 레이블을 가진 embedding은 가깝게, 다른 레이블을 가진 embedding은 멀리 위치하도록 학습합니다.

- **Performance Highlights**: 실험 결과, 우리의 접근 방식은 임상 노트(clinical notes)에서 성능을 향상시키는 데 있어 20% 이상 F1 점수 증가를 보여주었습니다. 이 방법은 CPU만을 사용해도 수천 개 샘플에 대해 몇 분 이내에 실행 가능하며, 기존의 PCA보다 성능이 크게 향상됩니다. 또한 원래의 모델을 업데이트할 필요 없이 여러 기계 학습 모델과 멀티모달 데이터셋에서 효과적으로 작동할 수 있음을 보여주었습니다.



### UD-Mamba: A pixel-level uncertainty-driven Mamba model for medical image segmentation (https://arxiv.org/abs/2502.02024)
Comments:
          19 pages

- **What's New**: 최근 Mamba 프레임워크에 대한 발전이 주목받고 있습니다. 이 프레임워크는 긴 거리의 의존성을 효율적으로 포착할 수 있는 상태 공간 모델(state-space model)로, 선형 계산 복잡성을 특징으로 합니다. UD-Mamba라는 새로운 방법론이 제안되었으며, 이는 픽셀 순서 스캔 과정을 채널 불확실성을 도입하여 재정의하고 있습니다.

- **Technical Details**: UD-Mamba는 두 가지 주요 스캔 기법, 즉 순차 스캔(sequential scanning)과 건너뛰기 스캔(skip scanning)을 도입하여 불확실성이 높은 영역을 우선적으로 처리합니다. 순차 스캔은 변별력이 높은 경계 및 전경 객체를 클러스터링하는데 효과적이며, 건너뛰기 스캔은 배경과 전경 간의 상호작용을 향상시킵니다. 또한, 네 개의 학습 가능 파라미터를 도입하여 다양한 스캔 방법에서 추출된 특징의 중요성을 균형 있게 조절합니다.

- **Performance Highlights**: UD-Mamba는 병리학, 피부 병변, 심장 작업을 포함한 세 가지 의료 이미징 데이터 세트에서 검증된 결과를 기반으로 강력한 세분화 성능을 보여주고 있습니다. 기존 Mamba 기반 방법들과 비교할 때, UD-Mamba는 애매한 영역을 더 효과적으로 식별하여 더 신뢰할 수 있는 세분화 결과를 이끌어냅니다. 이러한 성능 향상은 모델이 중요 영역을 정확하게 처리할 수 있도록 도와줍니다.



### Rethinking Timesteps Samplers and Prediction Types (https://arxiv.org/abs/2502.01990)
- **What's New**: 본 논문에서는 자원 제한 상황에서의 확산 모델(difusion model) 훈련의 주요 도전과제를 조사합니다. 저자들은 훈련 손실이 시간 단계(timestep)에 따라 크게 변동하는 것이 훈련 실패의 주요 원인이라는 것을 발견하였으며, 효율적인 $x_0$ 예측 타입을 선택하기 위한 혼합 예측 접근법의 가능성을 제안합니다. 이를 통해 자원 제약이 있는 상황에서도 확산 모델의 훈련을 개선할 수 있는 가능성을 탐구합니다.

- **Technical Details**: 확산 모델은 $t$ 시간 단계를 포함하며 다수의 신경망을 통해 특정 데이터와 노이즈를 근사하는 방식으로 동작합니다. 그러나, 서로 다른 시간 단계에서의 MSE 손실 변동이 큰 문제로 자리 잡아, 훈련의 효율성을 저하시킵니다. 저자들은 특정 시간 단계가 훈련에서 더 중요한 역할을 하며, 훈련 프로세스에서 $x_0$ 예측을 할 때 다양한 파라미터화 접근법을 사용해야 한다고 주장합니다.

- **Performance Highlights**: 확산 모델은 높은 해상도의 이미지 생성 작업에서 뛰어난 성능을 보여주었지만, 대량의 자원과 시간이 필요하다는 단점이 있습니다. 예를 들어, Stable Diffusion 모델은 256개의 A100 GPU를 사용할 경우 32일이 소요되며, 총 비용이 거의 50만 달러에 달합니다. 저자들은 자원이 한정된 상황에서도 효율적인 훈련 결과를 얻기 위해서는 혼합 예측 접근법이 중요하다고 강조합니다.



### Layer Separation: Adjustable Joint Space Width Images Synthesis in Conventional Radiography (https://arxiv.org/abs/2502.01972)
- **What's New**: 이 연구는 류마티스 관절염(Rheumatoid Arthritis, RA) 분석을 위한 새로운 접근법인 레이어 분리 네트워크(Layer Separation Networks, LSN)를 제안합니다. LSN은 손가락 관절의 X선 이미지를 전통적으로 분석하는 과정에서 발생하는 데이터 품질 문제를 해결하기 위해 소프트 티슈, 상부 및 하부 뼈 층을 정확하게 분리하는 데 중점을 둡니다. 이러한 기술을 사용하여 조정 가능한 관절 공간 너비(Joint Space Width, JSW) 이미지를 합성하여 RA 진단 및 연구를 위한 데이터 자원의 다양성을 높이는 것을 목표로 합니다.

- **Technical Details**: LSN은 세 가지 기본 하위 네트워크로 구성되어 있으며, 각각은 레이어 이미지 생성, 세분화 기반 감독, 소프트 티슈 식별 네트워크의 역할을 수행합니다. 이를 통해 X선 이미지에서 뼈와 소프트 티슈의 텍스처 분리를 시도하며, 이 과정은 대칭 기반 생성 네트워크(transUnet)를 활용하여 수행됩니다. 생성된 레이어 이미지는 기존의 다양한 조건을 충족하는 동시에, 픽셀 수준의 분별을 통해 더 높은 정확성을 보장합니다.

- **Performance Highlights**: 실험 결과, LSN 기반의 합성 이미지는 실제 X선 이미지와 매우 유사하였으며, 하류 작업에서의 성능을 크게 향상시켰습니다. 합성된 데이터는 RA 진단의 정확성, 안정성, 그리고 강건성을 향상시키고, 주석이 달린 훈련 데이터에 대한 의존성을 줄이는 데 도움이 됩니다. 이와 함께, 연구에서 제공하는 코드와 데이터셋은 향후 RA 연구 발전에 중요한 자원이 될 것입니다.



### HeRCULES: Heterogeneous Radar Dataset in Complex Urban Environment for Multi-session Radar SLAM (https://arxiv.org/abs/2502.01946)
Comments:
          2025 IEEE International Conference on Robotics and Automation (ICRA 2025)

- **What's New**: 최근 레이더 기술이 로봇 공학에서 주목받고 있으며, 이 논문에서는 서로 다른 레이더 유형을 결합하여 상호 보완적인 이점을 제공하는 새로운 HeRCULES 데이터셋을 소개합니다. 이 데이터셋은 4D 레이더와 스피닝 레이더를 FMCW LiDAR, IMU, GPS 및 카메라와 함께 제공합니다. 기존 데이터셋은 일관된 레이더 유형만 포함했지만, HeRCULES는 다양한 레이더를 통합하여 보다 포괄적인 연구를 지원합니다.

- **Technical Details**: HeRCULES 데이터셋은 복합 도심 환경에서 다중 세션 레이더 SLAM을 위한 이질적 레이더 데이터셋으로 설계되었습니다. 이 데이터셋은 다양한 날씨 및 조명 조건, 복잡한 도시 교통 시나리오를 반영하여, 로봇 인식 및 SLAM 연구를 위한 최적의 환경을 제공합니다. 또한 호환성 확보를 위해 ROS 플레이어와 레이더 포맷 변환 소프트웨어를 제공합니다.

- **Performance Highlights**: HeRCULES 데이터셋은 포괄적인 로컬라이제이션, 맵 생성 및 장소 인식을 가능하게 하며, 4D 레이더와 스피닝 레이더 간의 비교 연구를 지원합니다. 또한 데이터셋은 각 센서의 평균 포즈를 포함하고 있으며, SLAM 및 장소 인식 작업을 평가하기 위한 벤치마크 평가를 제공합니다. 이 데이터셋은 다양한 환경에서 강력한 SLAM 연구에 기여할 것으로 기대됩니다.



### VILP: Imitation Learning with Latent Video Planning (https://arxiv.org/abs/2502.01784)
- **What's New**: 이번 연구는 로봇 공학에서 비디오 생성 모델의 통합을 통해 일반 목적 로봇 에이전트를 위한 새로운 가능성을 제시합니다. 제안된 VILP(임시 비디오 계획을 통한 모방 학습)는 기존의 비디오 생성 로봇 정책보다 훈련 비용과 추론 속도, 생성된 비디오의 시간 일관성 등 여러 지표에서 뛰어난 성능을 보입니다. 이 방법은 높은 시간 정렬 기능을 갖춘 비디오를 여러 관점에서 생성할 수 있어 정책 학습에 중요한 역할을 합니다.

- **Technical Details**: VILP는 라텐트 비디오 확산 모델(latent video diffusion model)을 사용하여 로봇 비디오를 생성하는 방식으로, 실시간 계획과 실행이 가능한 특성을 가지고 있습니다. 예를 들어, 이 모델은 96x160 픽셀의 해상도로 6개의 프레임을 가진 비디오를 5Hz의 속도로 생성할 수 있으며, 각 추론 과정에서 0.073초로 5프레임 영상을 생성합니다. 이는 기존 로봇 정책에 비해 비약적인 성능 향상을 나타냅니다.

- **Performance Highlights**: 실험 결과 VILP는 기존의 비디오 생성 로봇 정책인 UniPi보다 뛰어난 훈련 비용, 추론 속도 및 생성된 비디오의 시간 일관성을 보여줍니다. VILP는 적은 양의 고품질 데이타를 기반으로도 강력한 성능을 유지할 수 있으며, 멀티 모달 행동 분포를 효과적으로 표현할 수 있는 능력을 지니고 있습니다. 이러한 결과는 로봇 정책 통합에서 비디오 생성 모델의 효용성을 효과적으로 입증합니다.



### Coarse-to-Fine 3D Keyframe Transporter (https://arxiv.org/abs/2502.01773)
- **What's New**: 이 연구는 Keyframe Imitation Learning (IL)에서 발생하는 비대칭성(bi-equivariant symmetry)을 이용하여 효율적인 정책을 설계했습니다. 이는 작업 환경(workspace)과 그립퍼에 의해 잡힌 객체(object)의 변형에 일반화되는 새로운 접근 방식을 제공합니다. 제안된 Keyframe Transporter는 교차 상관(cross-correlation) 기법을 활용하여 그립퍼가 잡은 객체의 특징과 씬(scene)의 특징 간의 관계를 평가합니다.

- **Technical Details**: 이 논문에서 제안한 방법은 SE(3) 행동 평가 체계를 사용하여 이종 변환(translation)과 회전(rotation) 작업을 동시에 처리합니다. 초기적으로 행동을 대략적으로 평가(tightly evaluate)한 후 가장 적절한 SE(3) 행동을 식별하고, 이를 바탕으로 근처의 행동을 정교화(refine)하여 평가하는 계층적 방법입니다. 이러한 방식은 3D 크로스-상관(cross-correlation) 수행의 차원을 줄여 효율성을 크게 향상시킨 특징을 가지고 있습니다.

- **Performance Highlights**: 제안된 메소드는 여러 시뮬레이션 작업에서 기존의 Keyframe IL 기준선보다 평균 10% 이상의 성능 향상을 보였으며, 4개의 실제 실험에서는 평균 55%의 향상을 달성했습니다. 이러한 결과는 단일 통합 아키텍처(single unified architecture)에서 다양한 조작 행위를 학습할 수 있게 해주며, 기존의 방법들과 비교하여 특정 작업에 한정되지 않는 장점을 지니고 있습니다.



### Multimodal Inverse Attention Network with Intrinsic Discriminant Feature Exploitation for Fake News Detection (https://arxiv.org/abs/2502.01699)
- **What's New**: 이번 연구는 마르텐츠 적인 모드별 특성을 활용한 'Multimodal Inverse Attention Network (MIAN)' 프레임워크를 제안합니다. 기존 접근법들은 모드 특수한 표현과 불일치 특징을 충분히 활용하지 못했으나, MIAN은 뉴스 콘텐츠의 본질적인 특징을 탐색하여 허위 뉴스 검출을 향상시킵니다. MIAN은 지역 내-지역 상호 작용과 지역 내-전역 상호 작용을 통해 다층 학습 모듈을 도입하여 향상된 단일 모드 표현을 생성합니다.

- **Technical Details**: MIAN은 모드 간 상호 작용 모듈을 통해 조관계 메커니즘을 사용하여 정제된 단일 모드 표현 간의 의존 관계를 설정합니다. 또한, 역 주의 메커니즘을 통해 각 모드에서 모순된 패턴과 의미적 편차를 강조함으로써 불일치 특징을 명확히 추출하는 데 중점을 둡니다. 이 구조는 뉴스 텍스트와 이미지 간의 명확한 관계를 탐색하고 단일 모드 및 다중 모드 내 본질적 식별 정보를 활용합니다.

- **Performance Highlights**: 광범위한 실험에서 MIAN은 여러 벤치마크 데이터 세트에서 기존의 최신 방법들에 비해 유의미한 성능 향상을 보였습니다. MIAN은 다양한 실제 사례에서 허위 뉴스의 탐지를 효과적으로 개선하며, 사회적 안전을 위한 솔루션을 제공합니다. 이 연구는 공공 정보의 신뢰성과 무결성을 보장하기 위한 자동화된 허위 뉴스 검출 기법의 개발을 촉진하는 데 기여하고 있습니다.



### A Novel Real-Time Full-Color 3D Holographic (Diffractive) Video Capture, Processing, and Transmission Pipeline Using Off-The-Shelf Hardwar (https://arxiv.org/abs/2502.01695)
Comments:
          Published and Presented at Session 63: Emerging Approaches for AR/VR/MR, SID Display Week 2022. 4 pages, 9 figures

- **What's New**: 이 논문은 상용 하드웨어를 사용한 세계 최초의 실시간 3D 홀로그램(diffractive) 영상 통화에 대한 내용을 다루고 있습니다. 이 시스템은 RGBZ 데이터를 캡처하고 처리하며 전송하는 새로운 파이프라인(pipeline)을 소개합니다. 여기서 iPhone을 이용한 이미지 및 깊이(depth) 캡처와 VividQ의 SDK를 사용한 홀로그램 생성 기술이 포함됩니다.

- **Technical Details**: 본 연구에서는 RGBZ 데이터 캡처를 위해 iPhone과 VividQ SDK를 활용하고 있습니다. 이 방식은 사실적인 3D 홀로그램을 생성하기 위한 혁신적인 기술 체계를 제공합니다. 특히, 디스플레이(display) 장비와의 통합을 통해 실시간 영상 통화가 가능해졌습니다.

- **Performance Highlights**: 이러한 시스템은 사용자가 상대방과 실시간으로 3D 홀로그램 형태로 상호작용할 수 있는 가능성을 제시합니다. 실시간 처리와 재현성이 높아 사용자의 경험을 극대화할 수 있습니다. 이 기술은 향후 다양한 분야에 응용될 수 있는 잠재력을 가지고 있습니다.



### Automated Extraction of Spatio-Semantic Graphs for Identifying Cognitive Impairmen (https://arxiv.org/abs/2502.01685)
Comments:
          To appear in ICASSP 2025

- **What's New**: 본 연구에서는 기존의 수작업으로 진행되던 Content Information Units (CIUs) 태깅 작업을 자동화하여, Cookie Theft 그림을 기반으로 한 spatio-semantic graph의 자동 추출 방법을 제안합니다. 이 자동화된 접근법은 인지 손상 평가에서 시각적 의미 경로를 자동적으로 특성화할 수 있는 가능성을 제시합니다. 연구 결과, 자동으로 생성된 spatio-semantic graph가 인지 손상 유무를 효과적으로 구분할 수 있음을 확인하였습니다.

- **Technical Details**: 이 연구에서 사용된 spatio-semantic graph는 CIUs의 위치와 시간 순서를 시각적으로 표현하는 그래프 이론적 구조입니다. 각 CIU는 Cookie Theft 그림의 픽셀 좌표에 매핑되며, 이 과정을 통해 시각적 경로를 구성합니다. 연구팀은 NetworkX 툴킷을 사용해 이러한 좌표들을 기반으로 노드와 엣지를 구성하여 시각적 의미 경로를 시각화합니다.

- **Performance Highlights**: 자동으로 생성된 spatio-semantic graph는 인지 손상이 있는 화자와 없는 화자를 구별하는 데 있어 유의미한 차이를 보였습니다. 통계적 분석 결과, 자동화된 방법으로 유도된 특징들이 수작업 방법과 유사한 결과를 생성했으나, 임상 집단 간 차이를 더 뚜렷하게 나타내는 것으로 확인되었습니다. 이는 자동화된 접근법이 인지 손상 평가를 위한 임상적 언어 모델 개발에 크게 기여할 수 있음을 시사합니다.



### Efficient Brain Tumor Classification with Lightweight CNN Architecture: A Novel Approach (https://arxiv.org/abs/2502.01674)
Comments:
          Accepted in FMLDS 2024

- **What's New**: 이 연구에서는 MRI 이미지를 사용하여 뇌 종양 분류를 위한 새로운 딥러닝 모델 아키텍처를 제안합니다. 기존 모델들이 정확도와 계산 효율성을 균형 있게 유지하는데 어려움을 겪는 가운데, 우리는 Separable Convolutions와 Squeeze and Excitation (SE) 블록을 통합하여 특징 추출을 개선하였습니다. 또한, 모델의 경량화를 위해 Batch Normalization과 Dropout 기법을 사용하여 과적합(overfitting)을 방지하고 있으며, 이를 통해 임상 응용에 적합한 강력한 도구를 제공합니다.

- **Technical Details**: 제안된 모델은 Separable Convolutions를 통해 파라미터 수를 줄이고, Global Average Pooling을 사용하여 계산 복잡성을 최소화하면서도 높은 정확도를 유지합니다. 이 모델은 5,712장의 훈련 데이터와 1,311장의 테스트 데이터를 바탕으로 구성되며, 각 클래스가 잘 대표되어 모델의 훈련 및 평가에 충분한 데이터를 제공합니다. 데이터 전처리 과정으로는 이미지 로딩, 픽셀 값 정규화, 데이터 증강(data augmentation) 기법을 적용하여 모델의 일반화 능력을 향상시킵니다.

- **Performance Highlights**: 우리의 모델은 99.22%의 검증 정확도와 98.44%의 테스트 정확도를 기록하며, 기존의 모델들보다 약 0.5%에서 1.0% 높은 정확도를 달성합니다. 또한, 손실 저하 측면에서도 1.5%에서 2.5% 향상을 보여주며, 다양한 뇌 종양 유형에 대해 효과적으로 일반화할 수 있는 능력을 입증하였습니다. 이러한 성과는 미래의 의료 영상 분석을 위한 딥러닝 모델에서 정확도와 효율성을 최적화하는 기반을 제공합니다.



### Entropy-based measure of rock sample heterogeneity derived from micro-CT images (https://arxiv.org/abs/2502.01665)
Comments:
          26 pages, 11 figures

- **What's New**: 이 연구에서는 인간의 평가 없이 원시 X-ray 마이크로-컴퓨터 단층촬영(micro-CT) 이미지를 통해 암석의 텍스처 이질성을 자동으로 측정하는 새로운 방법을 제시합니다. 기존의 시간 소모적이며 비용이 많이 드는 방법의 한계를 극복하고, 이미지 세분화에 의존하지 않고 micro-CT 이미지를 직접 처리하여 텍스처 이질성을 식별합니다. 이 방법은 샘플 특성에 따라 적응하며, 서로 다른 샘플 세트를 비교할 수 있게 합니다.

- **Technical Details**: 제안된 방법은 각 micro-CT 이미지를 분할하여 서브볼륨(subvolumes)을 형성하고, 각 서브볼륨의 특성을 계산합니다. 이 때 불확실성을 측정하기 위해 엔트로피(entropy)를 사용하며, 높을수록 더 높은 이질성을 나타냅니다. 이 과정은 입력 이미지의 해상도와 세분화 기술에 의존하지 않고 이미지에서 직접 이질성을 정량화하는 시스템적 접근을 통해 이루어집니다.

- **Performance Highlights**: 연구 결과, 선택된 특성들이 구조적 이질성과 강한 상관관계를 형성하는 데 핵심적인 역할을 하였음이 입증되었습니다. 전문가 평가를 통해 부여된 고정밀 데이터는 기존 텍스처 속성이 제안하는 이질성 분석 방법보다 더 효과적으로 전문가의 판단과 일치함을 보여줍니다. 이 방법은 암석 특성을 지원하는 추가적인 매개변수를 제공할 뿐만 아니라, 재현성과 비용 효율성 또한 보장합니다.



### SliderSpace: Decomposing the Visual Capabilities of Diffusion Models (https://arxiv.org/abs/2502.01639)
Comments:
          Project Website: this https URL

- **What's New**: 본 논문에서는 SliderSpace라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 diffusion models의 시각적 능력을 자동으로 분해하여 통제 가능하고 인간이 이해할 수 있는 방향으로 전환합니다. 기존의 방법들과 달리, 각 수정 방향에 대해 사용자가 속성을 개별적으로 지정할 필요 없이, 단일 텍스트 프롬프트로부터 여러 해석 가능한 방향을 동시에 발견합니다.

- **Technical Details**: SliderSpace는 낮은 랭크 어댑터(low-rank adaptor)로 훈련된 각 방향을 통해 조합 가능성을 제공하며, 모델의 잠재 공간(latent space)에서 놀라운 가능성을 발견하는 능력을 갖추고 있습니다. 이 방법은 최첨단 diffusion models에 대한 광범위한 실험을 통해 검증되었습니다. 실험을 통해, SliderSpace가 더 효율적으로 모델의 지식 구조를 분해한다는 것을 확인하였습니다.

- **Performance Highlights**: 세 가지 응용 분야인 개념 분해(concept decomposition), 예술적 스타일 탐색(artistic style exploration), 그리고 다양성 향상(diversity enhancement)에서 SliderSpace의 효과성을 입증했습니다. 정량적 평가에 따르면, SliderSpace가 발견한 방향은 diffusion models 내에 암호화된 능력에 대한 통찰력을 제공합니다. 사용자 연구에서는 이 방법이 기준 모델들에 비해 더 다양한 유용한 변형을 생성한다는 것을 추가적으로 검증하였습니다.



### MFP-VTON: Enhancing Mask-Free Person-to-Person Virtual Try-On via Diffusion Transformer (https://arxiv.org/abs/2502.01626)
- **What's New**: 이번 논문에서는 MFP-VTON이라는 새로운 mask-free VTON 모델을 제안합니다. 이는 의상을 이미 착용한 사람이나 제품 이미지에서 현실적인 fitting 이미지를 생성하는 데 중점을 두고 있습니다. 또한, 이 모델은 person-to-person VTON을 위해 커스텀 데이터셋을 준비하고, Focus Attention loss를 도입하여 참조 의상과 대상 인물의 외부 세부 사항에 대한 집중을 극대화합니다.

- **Technical Details**: MFP-VTON은 diffusion transformer를 기반으로 하여 strong generative capabilities를 활용합니다. 이 방식은 의상 영역의 직관적인 마스킹을 피하고, 참조 의상과 원래의 대상 인물을 직접 결합하여 보다 나은 결과를 제공합니다. Focus Attention loss를 통해 transformer의 주의력을 참조 의상과 대상 인물의 의상 외부 세부 사항에 집중시켜, 이미지의 자연스러움을 향상시킵니다.

- **Performance Highlights**: 실험 결과, MFP-VTON은 person-to-person 및 garment-to-person VTON 작업 모두에서 뛰어난 성능을 보였습니다. 기존의 최첨단 방법론과 비교할 때, 우리 방법은 높은 충실도의 fitting 이미지를 생성할 수 있음을 입증했습니다. MFP-VTON은 커스텀 데이터셋을 효과적으로 활용하여, 데이터 부족 문제를 해결하는 데 기여합니다.



### Robust-LLaVA: On the Effectiveness of Large-Scale Robust Image Encoders for Multi-modal Large Language Models (https://arxiv.org/abs/2502.01576)
Comments:
          Under Review

- **What's New**: 이 논문에서는 Multi-modal Large Language Models (MLLMs)의 취약성을 개선하기 위해 기존의 비전 분류 모델을 활용하는 새로운 접근 방식을 제시합니다. 기존의 대적 훈련 방법이 모델의 강건성과 일반화 능력을 제한하는 반면, 저자들은 대규모 데이터에서 대적 훈련된 비전 모델을 통합하여 강력한 비전-언어 성능을 유지합니다. 이러한 연구는 MLLMs의 안전성과 성능을 크게 향상시킬 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 연구는 대규모 대적 훈련을 통한 비전 인코더의 통합이 MLLMs의 시각적 추론 능력을 향상시킬 수 있다는 것을 입증합니다. 특히, CLIP 모델과 같은 기존 비전 인코더를 MLLM에 통합하여 강력한 시각적 특성에 적응하는 언어 구성 요소를 제공함으로써 복잡한 작업에서 향상된 성능을 보입니다. 이 방법은 복잡한 비전 관련 문제에 대해 두 배 이상의 강건성 향상을 달성할 수 있음을 나타냅니다.

- **Performance Highlights**: 저자들은 비전 질문 응답(VQA) 및 이미지 캡셔닝 작업에서 MLLMs가 평균적으로 각각 2배 및 1.5배의 강건성 향상을 달성한다고 보고하였습니다. 또한, 고급 jailbreaking 공격에 대해 10% 이상의 개선을 보여주며, 기존의 plug-and-play 방식보다 우수한 성능을 보입니다. 이 연구는 MLLMs의 대적 강건성을 강화하는 동시에 일반적인 성능을 유지할 수 있는 가능성을 제시합니다.



### MakeAnything: Harnessing Diffusion Transformers for Multi-Domain Procedural Sequence Generation (https://arxiv.org/abs/2502.01572)
- **What's New**: 최근 연구에서 제안된 MakeAnything 프레임워크는 21개 태스크와 24,000개 이상의 절차적 시퀀스를 포함하는 멀티 도메인 데이터셋을 사용하여 절차적 튜토리얼 생성을 위한 새로운 방법론을 소개합니다. 이 모델은 Diffusion Transformer (DIT)를 기반으로 하여 고품질의 절차적 시퀀스를 생성하는데 필요한 메소드와 데이터의 결합으로 향상된 성능을 보여줍니다. 또한 이는 이미지-투-프로세스 발전을 가능하게 하는 ReCraft 모델을 도입하여 정적인 이미지에서 단계별 생성 시퀀스를 분해할 수 있도록 합니다.

- **Technical Details**: MakeAnything은 비대칭 저차원 적응(LoRA) 구조를 활용하여, 에디터 매개변수를 얼리고 디코더 레이어를 적응적으로 조정함으로써 일반화 및 특정 태스크 성능의 최적 균형을 이룹니다. ReCraft 모델은 이미지-조건형 프로세스 재구성을 위한 효율적인 생성 방법을 제공하며, 세밀한 조정 없이도 최소한의 훈련 데이터로 강력한 성능을 달성하도록 설계되었습니다. 이는 Clean latent tokens를 통해 노이즈가 있는 중간 프레임을 정화하고, 정적인 예술 작품의 생성 이력을 효과적으로 재구성합니다.

- **Performance Highlights**: MakeAnything은 기존의 방법들을 초월하여 절차적 생성 태스크를 위해 새로운 성능 벤치마크를 설정했습니다. 다양한 실험을 통해 이 프레임워크가 여러 도메인에서 고품질의 절차적 시퀀스를 생성할 수 있음을 입증했습니다. 특히, ReCraft 모델은 적은 데이터로도 강력한 결과를 보여주어 다양한 생성 프로세스에 대한 응용 가능성을 넓히고 있습니다.



### GauCho: Gaussian Distributions with Cholesky Decomposition for Oriented Object Detection (https://arxiv.org/abs/2502.01565)
- **What's New**: 이번 연구에서는 Oriented Object Detection (OOD)을 위한 새로운 회귀 헤드인 GauCho를 제안합니다. GauCho는 Cholesky 분해를 기반으로 하는 가우시안 분포를 직접 생성하며, 기존의 Oriented Bounding Boxes(OBBs)의 경계 불연속성 문제를 이론적으로 완화하는 것이 특징입니다. 또한, GauCho는 최근의 가우시안 기반 회귀 손실 함수와 완벽하게 호환됩니다.

- **Technical Details**: GauCho는 네트워크의 출력을 통해 가우시안 분포의 매개변수를 직접 회귀함으로써 OBB의 중간 단계 사용을 피하는 새로운 패러다임을 제시합니다. 이 접근법은 공분산 행렬을 통해 각도 변화에 대한 연속적인 표현을 제공합니다. 특히, GauCho는 Cholesky 분해의 하삼각 행렬을 회귀하는 방식을 사용하여 네트워크 출력과 가우시안 표현 간의 일대일 매핑을 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면, GauCho는 전통적인 OBB 헤드와 비교하여 다양한 가우시안 기반 손실 함수 및 기준 OOD 방법에 대해 경쟁력 있는 결과를 보여주었습니다. 특정 데이터셋 DOTA에서, GauCho는 anchor-free 탐지기인 FCOS를 사용할 때 명확한 개선을 기록하였습니다. 이 연구는 다양한 접근법에서의 높은 성능과 적용 가능성을 보여주는 중요한 기여를 합니다.



### FireCastNet: Earth-as-a-Graph for Seasonal Fire Prediction (https://arxiv.org/abs/2502.01550)
- **What's New**: 이 연구에서는 기후 변화(C climate change)가 화재 날씨 조건에 미치는 영향을 반영하여, 숲 화재의 예측을 위한 새로운 접근 방식을 제시합니다. SeasFire라는 포괄적인 글로벌 숲 화재 데이터셋을 활용하여, 기후, 식생(vegetation), 해양 지수(oceanic indices), 그리고 인간 관련 변수(human-related variables)를 포함한 계절적인 화재 예측을 수행합니다.

- **Technical Details**: 새로운 아키텍처인 FireCastNet은 3D convolutional encoder와 Graph neural networks를 결합하여 구축되었습니다. 이 모델은 숲 화재에 이르는 다양한 공간적(spatial) 및 시간적(temporal) 맥락을 포착하도록 훈련되었습니다. 또한, 민감도 분석을 통해 예측 시간 지평선(time horizons)에 따른 불이 탄 지역의 예측 효과성(effectiveness)을 평가합니다.

- **Performance Highlights**: 연구 결과, 깊은 학습(deep learning) 모델이 계절 화재 예측에서 유망함을 보여줍니다. 입력 시계열의 길이가 증가함에 따라 예측의 강건성(robustness)이 향상되고, 공간 정보를 통합하여 화재의 시공간적(spatio-temporal) 동태를 포착하는 것 또한 성능을 강화합니다. 더 나아가 긴 예측 지평선에서 성능을 개선하기 위해서는 넓은 수용 필드(receptive field)가 필요하다는 사실이 밝혀졌습니다.



### VisTA: Vision-Text Alignment Model with Contrastive Learning using Multimodal Data for Evidence-Driven, Reliable, and Explainable Alzheimer's Disease Diagnosis (https://arxiv.org/abs/2502.01535)
- **What's New**: 본 논문에서 제안하는 VisTA(비전-텍스트 정렬 모델)는 고차원 방사선 이미지를 통해 알츠하이머병(AD)을 진단하기 위한 혁신적인 다중모달 언어-비전 모델입니다. VisTA는 대조 학습(constrative learning)을 활용하여 질병 예측 및 임상 의사 결정에 대한 해석 가능성을 높이는 방향으로 최적화되었습니다. 이는 의사들이 기계 학습 기반 예측을 정당화할 수 있는 임상 증거를 요구하는 현재의 요구에 부합합니다.

- **Technical Details**: VisTA는 의료 이미징을 위한 사전 훈련된 언어-비전 모델인 BiomedCLIP에서 개발되었습니다. 이 모델은 검증된 이상과 그 설명을 정렬하기 위해 대조 학습으로 미세 조정(fine-tuned)되었습니다. VisTA는 예측된 이상 유형, 참조 사례와의 유사성, 증거 기반 설명 및 최종 AD 진단 등 네 가지 출력을 생성하여 임상의들이 중간 결과를 확인하고 잠재적인 오류를 파악하도록 지원합니다.

- **Performance Highlights**: VisTA는 170개의 샘플만을 사용하여 미세 조정한 후 비약적인 개선을 이루었으며, 이상 검색에서는 74%의 정확도와 0.87의 AUC를 달성했습니다. 또한, 치매 예측에서는 88%의 정확도와 0.82의 AUC를 기록하였으며, 이는 이전 모델에 비해 현저한 성능 향상을 보여줍니다. 이 모델이 생성한 설명은 인간 전문가의 평가와 높은 일치를 보이며, 진단 과정을 명확하게 해석할 수 있는 통찰력을 제공합니다.



### The in-context inductive biases of vision-language models differ across modalities (https://arxiv.org/abs/2502.01530)
Comments:
          10 pages

- **What's New**: 본 연구는 현대 비전-언어 모델(Vision-Language Models, VLMs)이 시각적 자극과 텍스트 자극을 통해 학습할 때 팀바이러스(Inductive Biases)가 어떻게 다르게 나타나는지 분석합니다. 특히 색상(color)과 형태(shape)의 두 가지 특징의 차이를 비교하면서, 예시의 제시 방식이 일반화(generalization)에 미치는 영향을 연구합니다. 이러한 결과는 인공지능(AI) 모델의 맥락(context) 내 학습 과정에 대한 이해를 높이고, 실제 응용에 실질적인 시사점을 제공합니다.

- **Technical Details**: 이 연구에서는 이미지와 텍스트에서의 카테고리 학습(paradigms)을 통한 모델의 일반화 성향을 조사하기 위해 세 가지 실험 패러다임을 사용했습니다. 연구진은 특정 특징이 나타나는 것을 기준으로 모델이 일반화하는 경향을 비교하기 위해, 시각적 자극인 이미지를 통한 학습과 텍스트 설명을 통한 학습을 구분하고, 각 모드에서 모델의 반응을 분석했습니다. 이를 통해 형태 바이어스(shape bias)와 색상 바이어스(color bias), 그리고 형태와 색상의 혼합 형태를 비교했습니다.

- **Performance Highlights**: 연구 결과, VLMs는 이미지로부터 학습할 때 형태에 더 강한 바이어스를 보이는 경향이 있음이 밝혀졌습니다. 텍스트로 자극이 제시될 경우, 형용사(adjectives)의 순서 역시 모델의 일반화 경향에 영향을 미치며, 첫 번째 형용사가 선호됩니다. 그러나 이러한 경향은 모델의 구조나 작업(Task)의 유형에 따라 달라질 수 있습니다.



### Efficiently Integrate Large Language Models with Visual Perception: A Survey from the Training Paradigm Perspectiv (https://arxiv.org/abs/2502.01524)
Comments:
          28 pages, 3 figures

- **What's New**: 이번 논문에서는 Vision-Language Large Language Models (VLLMs)에서 시각 모달리티의 통합을 위한 새로운 교육 패러다임을 제시합니다. 연구의 주요 초점은 MoDALITY Integrators (MIs)와 함께 두 가지 주요 조정 과정인 Single-stage Tuning과 Two-stage Tuning입니다. 이를 통해 연구자들은 LLMs의 매개변수 효율성을 유지하면서 성능을 향상시킬 수 있는 방법을 제시하고 있습니다.

- **Technical Details**: 본 연구에서는 다양한 LLM 아키텍처와 함께 Parameter-Efficient Adaptation (PEA) 기법을 분류합니다. Single-stage Tuning과 Two-stage Tuning 방식은 각각 고유한 효율성 동기를 가지고 있으며, Direct Adaptation 방식은 자원의 효율적인 소비를 강조합니다. 각 교육 패러다임에 대해 독특한 매개변수 최적화 전략을 제시하며, 다양한 비전 인코더의 아키텍처를 포괄적으로 다룹니다.

- **Performance Highlights**: 논문은 34개의 VLLMs를 조사한 결과, 최신 VLLMs에서의 Two-stage Tuning과 함께 여러 모델의 성능을 비교 분석합니다. 실험 결과는 Direct Adaptation 접근 방식의 주요 개발과 효과성을 입증하며, 다양한 분야에서 비전 모달리티 통합의 효율성을 끌어올리는 데 기여할 수 있는 지침을 제공합니다. 연구의 결과는 연구자와 실무자들이 LLMs에 비전 모달리티를 효율적으로 통합하는 데 유용한 통찰력을 제공합니다.



### BD-Diff: Generative Diffusion Model for Image Deblurring on Unknown Domains with Blur-Decoupled Learning (https://arxiv.org/abs/2502.01522)
Comments:
          We propose BD-Diff to integrate generative diffusion model into unpaired deblurring tasks

- **What's New**: 본 연구에서는 BD-Diff라는 새로운 generative-diffusion 모델을 제안하여 알려지지 않은 도메인에서의 이미지 디블러링 성능을 향상시키고자 합니다. 이 모델은 구조적 특징과 블러 패턴의 분리를 통해 데이터의 일반화 능력을 높이는 것을 목표로 하며, 세 가지 특별히 설계된 작업에서 공동 훈련을 실시합니다. BD-Diff는 기존의 supervised 방법에서 발생하는 과적합 문제를 해결하고, 실제 데이터에서도 뛰어난 성능을 발휘합니다.

- **Technical Details**: 제안된 BD-Diff 모델은 두 개의 Q-Former를 사용하여 구조적 표현과 블러 패턴 추출기를 각각 구현합니다. 구조 추출기는 합성 데이터에서 구조적 특징을 학습하고, 블러 패턴 추출기는 비지도 블러 전이 작업을 통해 특정 도메인의 블러 패턴을 식별합니다. 이 모델은 추가적으로 Reconstruction 작업을 통해 두 요소 간의 상호 보완성을 강화하여 데이터가 부족한 도메인에서도 잘 작동할 수 있도록 합니다.

- **Performance Highlights**: 실제 데이터 세트에서의 실험 결과, BD-Diff는 다양한 도전적인 시나리오에서 기존의 최신 방법들을 능가하는 성능을 보였습니다. DB-Diff는 기존 방법들의 한계를 극복하며 블러 제거 및 구조 보존에서 뛰어난 결과를 보여주었습니다. 중요한 점은 이 모델이 비교적 적은 양의 훈련 데이터로도 효과적인 성능을 발휘할 수 있도록 설계되었다는 것입니다.



### End-to-end Training for Text-to-Image Synthesis using Dual-Text Embeddings (https://arxiv.org/abs/2502.01507)
- **What's New**: 이 논문에서는 Text-to-Image (T2I) 합성을 위해 특화된 텍스트 임베딩을 효과적으로 학습하는 새로운 접근 방식을 제안하고 있습니다. 기존의 방법들과 달리 이 연구는 엔드 투 엔드(end-to-end) 방식으로 T2I 합성 네트워크에 맞춤형 텍스트 임베딩을 학습하여 결과의 품질을 향상시키고자 합니다. 이 접근법은 생성적 및 대조적 훈련을 결합하여 생성된 이미지의 포토리얼리즘(photo-realism)과 텍스트-이미지 정합(text-to-image alignment) 간의 균형을 이루는 두 개의 별도의 임베딩을 활용합니다.

- **Technical Details**: 이 연구는 Oxford-102, Caltech-UCSD, MS-COCO와 같은 세 가지 텍스트-이미지 벤치마크 데이터셋에서 평가를 수행하며, 세 가지 평가 지표(Inception Score, Fréchet Inception Distance, R-precision)를 사용하여 이미지 품질과 텍스트-이미지 정합을 측정합니다. 결과적으로 해당 접근법은 공유 임베딩을 사용하는 것보다 우수한 성능을 보이며, 대조적 손실(contrastive loss)을 통해 이전 훈련에서 학습된 임베딩으로부터 생성된 텍스트 표현보다 경쟁력 있는 결과를 보여줍니다. 또한, 학습된 임베딩은 텍스트-이미지 조작 등 다양한 다른 작업에서도 응용할 수 있음을 입증하고 있습니다.

- **Performance Highlights**: 실험 결과에 따르면 제안된 모델은 CUB에서 FID 점수를 14.06에서 13.67로, Oxford-102에서 40.31에서 30.07로 감소시켰으며, MS-COCO에서는 FID 점수가 35.49에서 25.17로 떨어졌습니다. 이러한 성능은 이전의 AttnGAN보다 우수한 결과를 나타냅니다. 이 두 개의 별도의 텍스트 임베딩을 사용하는 접근 방식이 검증된 바에 따르면, 공유 임베딩 사용에 비해 더 나은 성능을 발휘하는 것으로 확인되었습니다.



### MoireDB: Formula-generated Interference-fringe Image Datas (https://arxiv.org/abs/2502.01490)
- **What's New**: 이번 연구에서는 기존 데이터 증강 기법인 PixMix의 한계를 극복하기 위해 MoireDB라는 새로운 방식을 제안합니다. MoireDB는 수학적 공식을 활용하여 생성된 간섭 무늬 이미지 dataset으로, 저작권 문제를 제거하고 데이터셋 조립 비용을 줄이며 이미지 인식 모델의 견고성을 향상시키는 것이 목표입니다. 이러한 MoireDB를 통해 실제 세계의 열악한 환경에도 강한 이미지 인식 모델을 만들 수 있게 됩니다.

- **Technical Details**: MoireDB는 공식적으로 생성된 Moiré 이미지를 저장하는 데이터베이스로, 기존의 Fractal arts와 feature visualizations (FVis) 대신 사용됩니다. 연구자들은 Moiré 이미지를 데이터 증강 시 사용할 때 모델의 견고성이 향상된다는 가설을 세웠으며, 이를 통해 ddeep learning 모델을 훈련하여 견고성을 테스트하게 됩니다. MoireDB의 이미지들은 저작권 문제 없이 자동으로 생성되므로, 상업적 사용이 가능합니다.

- **Performance Highlights**: 실험 결과, MoireDB를 통해 증강된 이미지는 전통적인 Fractal arts 및 FVis 기반의 증강 기법보다 뛰어난 성능을 보였습니다. 특히, 이미지 인식 모델은 실제 환경에서의 열악한 조건에서도 더욱 견고하게 작동하는 것으로 확인되었습니다. 이러한 작업은 MoireDB의 확장 가능성과 효과성을 보이며, 객체 인식 모델의 품질을 크게 향상시키는 데 기여할 것으로 예상됩니다.



### Simultaneous Automatic Picking and Manual Picking Refinement for First-Break (https://arxiv.org/abs/2502.01474)
- **What's New**: 이 논문에서는 Simultaneous Picking and Refinement (SPR) 알고리즘을 새롭게 제안합니다. 이 알고리즘은 아웃라이어(outlier) 샘플이나 노이즈가 있는 라벨을 처리하도록 설계되었습니다. 기존의 접근법과는 달리, SPR은 수동으로 레이블링된 데이터 집합에서 진정한 첫 번째 단층(first-break)을 잠재 변수로 간주하여 라벨링 프리포맷(prior)을 포함하는 확률적 모델을 사용합니다.

- **Technical Details**: SPR 알고리즘은 데이터 상의 아웃라이어 및 잘못된 라벨의 영향을 완화하며, 확률적 모델을 통해 동적으로 조정된 진정한 첫 번째 단층의 식별을 목표로 합니다. 논문에서는 다중 추적(multi-trace) 처리를 적용하여 신호 감지 점에서 수신한 진폭 데이터 x∈R^{M×N}을 사용하여 첫 번째 단층을 식별합니다. 이 데이터는 인접한 추적 간의 공간적 상관관계를 활용하여 처리가 이루어집니다.

- **Performance Highlights**: 공개 데이터에 대한 실험을 통해 SPR 알고리즘의 성능을 입증하였으며, 다양한 사이트에서의 일반화 능력도 확인되었습니다. 또한, 노이즈가 있는 신호 및 라벨에 대한 실험을 통해 SPR의 내구성을 강조하였고, 잘못 정렬된 수동 주석을 정제하는 능력도 보여주었습니다. SPR은 특정 네트워크 구조에 제한되지 않으며 다양한 심층 학습 기반 방식에 적응할 수 있는 유연성을 제공합니다.



### Deep Unfolding Multi-modal Image Fusion Network via Attribution Analysis (https://arxiv.org/abs/2502.01467)
Comments:
          Accepted in IEEE Transactions on Circuits and Systems for Video Technology (TCSVT) 2024

- **What's New**: 이번 연구에서는 'Unfolding Attribution Analysis Fusion network' (UAAFusion)이라는 새로운 이미지 융합 네트워크를 제안합니다. 이 네트워크는 주석 분석(attribution analysis)을 활용하여 융합된 이미지를 시맨틱 분할(semantic segmentation)에 더 적합하게 만들어, 두 과정 간의 상호작용을 강화합니다. 특히, 이 방법은 소스 이미지에서 중요한 영역을 강조하여 시맨틱 분할에서의 성능 향상을 도모합니다.

- **Technical Details**: UAAFusion은 기존의 네트워크 구조에 주석 분석을 통합하고, 최적화 목표를 설정하며, 이를 통해 융합 손실(attribution fusion loss)을 계산합니다. 새로운 경로 함수(pathway function)를 개발하여 융합 작업에 적합하게 구성하였고, 네트워크의 각 단계에는 주석 주의 메커니즘(attribution attention mechanism)을 삽입하여 고수준 인식 작업에 중요한 영역과 픽셀에 우선순위를 두었습니다. 또한, 정보 손실을 줄이기 위해 메모리 증강 모듈(memory augmentation module)을 통합하여 다양한 네트워크 레이어 간의 정보 흐름을 개선하였습니다.

- **Performance Highlights**: 광범위한 실험을 통해 우리의 방법이 이미지 융합 및 시맨틱 분할에서 우수한 성능을 발휘함을 입증하였습니다. 알고리즘적 전개 기법을 통해 구조화된 네트워크는 성능과 해석성을 모두 향상시켜, 효과적으로 다운스트림 작업에 적합한 이미지를 생성합니다. 실험 결과, 이 새로운 접근 방식이 기존 방법보다 뛰어난 결과를 도출함을 나타냅니다.



### Temporal-consistent CAMs for Weakly Supervised Video Segmentation in Waste Sorting (https://arxiv.org/abs/2502.01455)
Comments:
          14 pages, 7 figures

- **What's New**: 이번 연구는 산업 환경에서 강력한 기계 비전 시스템과 딥러닝을 활용한 약한 감독 방식(Weakly Supervised, WS) 접근법의 효과를 보여줍니다. 특히, 이 방법은 비디오 스트림의 의미적 세그멘테이션을 위한 정확한 마스크를 생성할 수 있으며, 연속 프레임 간의 시간적 일관성을 활용하여 객체가 다양한 프레임에 나타날 때의 일관성을 증진시킵니다. 또한, 인공지능 모델의 훈련 과정에서 시간적 일관성을 통합하는 방안을 제안합니다.

- **Technical Details**: 제안된 방법은 먼저 두 개의 카메라를 활용하여 분리 과정 전후의 이미지를 수집합니다. 이러한 이미지를 바탕으로 생성된 saliency map은 주변 프레임의 움직임을 보정하여 같은 객체의 saliency map이 서로 유사하도록 강제합니다. 이 과정에서 Opticl Flow 알고리즘을 사용하여 연속 프레임 간의 모션을 계산하고, 이를 통해 네트워크가 유사한 출력을 생성하도록 합니다. 또한, 기존의 binary classification 문제에서 벗어나, 배경과 객체를 구분하는 세 가지 클래스(classification problem)를 설정하여 좀 더 정교한 세그멘테이션 마스크를 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 정확하고 일관성 있는 세그멘테이션 마스크를 생성함으로써 산업 폐기물 분류 애플리케이션에 적합하다는 것을 입증하였습니다. 이로 인해, 법적 객체와 불법 객체를 세밀한 픽셀 수준의 주석 없이도 구분할 수 있습니다. 이러한 접근법은 기존의 Fully Supervised segmentation 방법과 비교하여 훈련 데이터 수집의 비용과 시간을 대폭 절감할 수 있는 효과적인 방안이 됩니다.



### SPFFNet: Strip Perception and Feature Fusion Spatial Pyramid Pooling for Fabric Defect Detection (https://arxiv.org/abs/2502.01445)
Comments:
          8 pages, 4 figures, conference

- **What's New**: 본 연구에서는 YOLOv11 기반의 패브릭 결함 검출 모델을 제안하며, 그 과정에서 다중 스케일 컨볼루션을 통한 스트립 결함 인식 강화를 위해 Strip Perception Module (SPM)을 도입합니다. 또한, 스퀴즈-앤-익사이테이션(spatial pyramid pooling Fast, SPPF) 메커니즘을 통합하여 SE-SPPF 모듈을 개선하여 공간적 및 채널 정보를 효율적으로 통합하고 결함 특징 추출을 향상시킵니다. 이와 함께, 스케일 차이와 클래스 불균형 문제를 해결하는 새로운 Focal Enhanced Complete IoU (FECIoU) 메트릭을 제안하여 더욱 효과적인 결함 탐지를 목표로 하고 있습니다.

- **Technical Details**: 연구에서는 SPM을 도입하여 다중 스케일 컨볼루션을 활용하여 스트립 결함 특징을 추출하고, SE-SPPF를 통해 배경과 결함을 보다 잘 구분할 수 있게 합니다. FECIoU 메트릭을 도입하여 대상 박스의 스케일 변화를 동적으로 조정하고, 하드 투 디텍트(hard-to-detect) 인스턴스의 가중치를 조절하여 더욱 적응적이고 효율적인 탐지가 가능하게 합니다. 모델은 고속의 탐지성과 실시간 성능을 유지하며 결함 검출 정확도를 크게 향상시킵니다.

- **Performance Highlights**: Tianchi 데이터셋에서 평균 정확도(mean average precision, mAP)가 0.8-8.1% 개선되었으며, 연구자가 커스터마이징한 데이터셋에서도 1.6-13.2%의 성능 향상을 기록했습니다. 본 모델은 기존의 최첨단 기법들을 초월하는 성능을 보여주며, 특수한 결함 유형을 다루는 데 있어 한 단계 더 나아간 결과를 제시합니다. 실험 결과를 통해, 제안한 방법이 복잡한 배경과 다양한 결함 형태를 성공적으로 처리할 수 있음을 입증했습니다.



### Improved Training Technique for Latent Consistency Models (https://arxiv.org/abs/2502.01441)
Comments:
          Accepted at ICLR2025

- **What's New**: 최근 생성적 모델의 세계에서 새로운 일관성 모델(consistency models)이 주목받고 있다. 특히 이 모델들은 한 번의 단계에서 뿐만 아니라, 여러 단계를 통해 고품질 샘플을 생성할 수 있는 능력을 가지고 있다. 전통적인 확산 모델(difussion models)보다 더 효과적으로 큰 데이터셋에서 성능을 발휘하기 위해 잠재 공간(latent space)에서의 일관성 성능이 매우 중요하다는 점을 강조하고 있다.

- **Technical Details**: 본 연구에서는 고속 샘플링을 위해 누락된 위험을 최소화하는 방법으로 Cauchy 손실 함수를 도입하였다. 또한, 초기 타임스텝에서 확산 손실(difussion loss)을 추가하고, 최적 운송(optimal transport, OT) 결합 방법을 사용하여 성능을 강화하였다. 마지막으로, 적응형 스케일 조절 방법인 scaling-$c$ 스케줄러를 통해 훈련 과정을 조절하고, 구조상 비축척 레이어 노름(Non-scaling LayerNorm)을 적용하여 특성 통계를 보다 잘 포착할 수 있도록 했다.

- **Performance Highlights**: 이러한 전략을 통해 잠재 일관성 모델(latent consistency models)을 성공적으로 훈련시켜 고품질 샘플링을 1~2 단계에서 구현할 수 있었다. 이러한 성과로 인해 잠재 일관성과 확산 모델 간의 성능 격차가 크게 줄어들었다. 연구 결과, 제안된 방법들이 기존의 iCT 프레임워크 대비 성능을 크게 향상시키는 데 기여할 수 있음을 보여주었다.



### Visual Attention Never Fades: Selective Progressive Attention ReCalibration for Detailed Image Captioning in Multimodal Large Language Models (https://arxiv.org/abs/2502.01419)
- **What's New**: 이 논문에서 제안된 SPARC (Selective Progressive Attention ReCalibration) 방법은 기존 멀티모달 대형 언어 모델(MLLMs)의 이미지 캡셔닝 문제를 해결하기 위해 개발되었습니다. 현재의 방법들은 일반적으로 정밀도(precision)를 향상시키지만, 회수율(recall)을 저하시켜, 캡션의 핵심 요소를 놓치는 경우가 많습니다. SPARC는 이러한 한계를 극복하고, 육안 평가 및 자동 평가에서 우수한 성과를 보여줍니다.

- **Technical Details**: SPARC는 시각 토큰의 기여를 강조하기 위해 의도적으로 선택적으로 시각 토큰의 영향을 증폭합니다. 이 방법은 세 가지 주요 원칙 즉, 전체 시각 토큰의 영향 증가가 회수율을 감소시킨다는 점, 시각 주의(attention)가 길어질수록 노이즈(noise)를 더 많이 발생시킨다는 점, 그리고 약해지는 시각 주의를 강화하여 지속시키는 원칙에 기반하고 있습니다.

- **Performance Highlights**: SPARC는 기존의 방법들보다 캡션 품질을 크게 향상시키며, 정밀도와 회수율 모두를 효과적으로 개선합니다. 자동화된 평가와 사람의 평가를 통해, SPARC의 우수성이 입증되었으며, 기존의 방법들이 가진 정밀도와 회수의 절충(trade-off) 문제를 해결하는 데 기여합니다.



### Human Body Restoration with One-Step Diffusion Model and A New Benchmark (https://arxiv.org/abs/2502.01411)
Comments:
          8 pages, 9 figures. The code and model will be available at this https URL

- **What's New**: 이번 연구에서는 고품질 이미지 복원에 대한 새로운 데이터셋과 기술을 제안합니다. 새로운 데이터셋인 PERSONA는 사람의 모습을 복원하는 데 최적화되어 있으며, 다양한 활동과 복잡한 상호작용을 포함하고 있습니다. 또한, OSDHuman이라는 새로운 일단계 확산 모델을 도입하여 인간 이미지 복원의 효율성을 높였습니다.

- **Technical Details**: 연구에서는 HQ-ACF(고품질 자동 크롭 및 필터링) 파이프라인을 통해 라벨이 없는 이미지를 포함한 기존 오브젝트 탐지 데이터셋을 활용하여 고품질 인간 이미지를 생산합니다. HFIE(고충실도 이미지 임베더)를 사용하여 적절한 프롬프트를 생성하고, VSD(변이 점수 증류) 정규화를 적용하여 자연 이미지 분포에 맞춘 생성 분포를 유도합니다.

- **Performance Highlights**: OSDHuman 모델은 기존의 방법들에 비해 뛰어난 시각적 품질과 정량적 지표를 달성했습니다. PERSONA 데이터셋은 고품질의 109,052장의 이미지를 포함하여 인간 이미지 복원 작업에 있어서 중요한 기준이 될 것입니다. 실험 결과, OSDHuman은 인공지능 모델의 효율성을 유지하면서도 더 우수한 복원 성능을 보여주었습니다.



### FourieRF: Few-Shot NeRFs via Progressive Fourier Frequency Contro (https://arxiv.org/abs/2502.01405)
Comments:
          8 pages, 3DV 2025 conference

- **What's New**: 이번 연구에서는 Few-Shot 설정에서 빠르고 높은 품질의 재구성을 달성하기 위한 새로운 접근법인 FourieRF를 제안합니다. 본 방법은 명시적인 Curriculum Training 절차를 통해 특징을 효과적으로 매개변수화하며, 최적화 과정에서 장면 복잡성을 점차 증가시킵니다. FourieRF는 다양한 장면에서 강력하고 적응 가능한 기초를 수립하며, 기존 방법들과 비교하여 아티팩트(artifact)를 크게 줄이는 특징이 있습니다.

- **Technical Details**: FourieRF는 주파수 변환(Fourier Transform)을 기반으로 한 일반적인 Prior를 사용하여 grid 기반 NeRF 방법의 특징을 매개변수화합니다. 본 접근법은 각 반복(iteration)마다 적용되며, 거의 계산 오버헤드가 없어 매우 빠르고 효과적입니다. 복잡한 장면에서도 높은 품질의 결과를 제공하며, 데이터에 의존하지 않는 명시적 정규화(explicit regularization) 기법을 채택한 기존 최첨단 방법들과 비교하여 더욱 빠른 학습을 가능하게 합니다.

- **Performance Highlights**: FourieRF는 기존의 방법들과 비교하여 기록적인 시간 내에 학습할 수 있으며, 실험 결과 매우 견고한 결과를 도출했습니다. 기존 NeRF와 Mip-NeRF 기반의 접근법들이 가진 높은 계산 비용 문제를 극복하며, 매우 제한된 입력으로도 우수한 재구성을 제공합니다. 이로 인해 Few-Shot 렌더링 문제에 대한 새로운 표준을 설정하게 되었습니다.



### AdaSVD: Adaptive Singular Value Decomposition for Large Language Models (https://arxiv.org/abs/2502.01403)
Comments:
          The code and models will be available at this https URL

- **What's New**: 본 논문에서는 LLM(대형 언어 모델)의 메모리 요구사항을 줄이기 위해 AdaSVD라는 적응형 SVD 기반 LLM 압축 방법을 제안합니다. 기존의 SVD 기반 방법들이 자주 발생했던 성능 저하 문제를 해결하기 위해 adaComp와 adaCR을 도입하여 SVD의 절단 오류를 보상하고 각 변환기 레이어에 적합한 압축 비율을 할당합니다. 따라서 AdaSVD는 리소스 제약이 있는 장치에서의 LLM 배포 가능성을 크게 향상시키는 데 기여합니다.

- **Technical Details**: AdaSVD는 SVD 트렁케이션(절단) 오류를 적응적으로 보정하는 adaComp 기법을 통해 U와 V^T 행렬을 번갈아 업데이트합니다. 또한, adaCR을 통해 각 레이어의 중요도에 기반하여 레이어별 압축 비율을 할당하여 모든 레이어에 동일한 비율을 적용하는 기존 방식과의 차별성을 두고 있습니다. 이러한 방식은 압축 비율이 고정된 상태에서도 성능을 개선하는 데 기여함으로써, 운영 메모리와 GPU 메모리에서의 사용 효율을 높입니다.

- **Performance Highlights**: 다양한 LLM 계열과 평가지표를 바탕으로 한 실험 결과, AdaSVD는 기존의 SOTA SVD 기반 LLM 압축 방법인 SVD-LLM을 초월하여 성능을 현저하게 향상시켰습니다. 이로 인해 압축 모델과 원본 모델 간의 성능 격차가 크게 줄어들어, LLM의 효과적이고 효율적인 실행이 가능해졌습니다. 제안된 방법은 다양한 플랫폼에서의 배포와 사용을 동시에 지원하는 가능성을 높입니다.



### Evolving Symbolic 3D Visual Grounder with Weakly Supervised Reflection (https://arxiv.org/abs/2502.01401)
- **What's New**: 이 논문에서는 3D 비주얼 그라운딩(3DVG)을 위한 새로운 훈련 없이 사용할 수 있는 상징적 프레임워크인 EaSe를 소개하고 있습니다. EaSe는 LLM과 VLM을 통합하여 비교 가능한 성능을 유지하면서 이전 에이전트 기반 방법들보다 상당히 낮은 추론 비용을 제공합니다. 이 방법은 대규모 3D 비전-언어 데이터셋의 부족과 높은 비용 문제를 해결하는 데 초점을 맞추고 있습니다.

- **Technical Details**: EaSe는 LLM이 생성한 최적화된 Python 코드로 공간적 관계를 인코딩하는 방식을 채택하고 있습니다. 또한 이 시스템은 참조 발화를 상징적 표현으로 분석하고, 각 물체의 위치에 따라 공간적 관계 특징을 생성하여 관련성이 없는 물체를 제거합니다. VLM은 남은 후보 물체의 이미지에서 목표를 선택하는 역할을 담당합니다.

- **Performance Highlights**: 실험 결과, EaSe는 Nr3D 데이터셋에서 52.9%의 정확도를 달성하여 훈련 없는 최신 방법으로서의 성능을 입증했습니다. 또한, 24배 낮은 시간 비용과 6.8배 낮은 토큰 비용을 제공하여 성능과 효율성의 균형을 잘 맞추고 있습니다. 이러한 연구는 훈련이 필요 없는 접근 방식에 대한 가능성을 보여줍니다.



### Bayesian Approximation-Based Trajectory Prediction and Tracking with 4D Radar (https://arxiv.org/abs/2502.01357)
Comments:
          6pages, 4 figures

- **What's New**: 이번 연구에서는 Bayes-4DRTrack라는 4D Radar 기반의 3D Multi-Object Tracking (MOT) 시스템을 제안하였습니다. 기존의 MOT 시스템이 고정된 노이즈 공분산에 의존했던 것에 반해, 본 시스템은 Bayesian approximation 기법을 도입하여 물체 감지와 예측 단계에서의 적응성을 향상시킵니다. 이 접근 방식은 동적인 환경에서 여러 객체의 트래킹 성능을 개선할 수 있는 잠재력을 보여주고 있습니다.

- **Technical Details**: Bayes-4DRTrack 시스템은 transformer 기반의 모션 예측 네트워크를 활용하여 비선형 모션 역학을 잡아내고, Doppler 측정을 사용하는 2단계 데이터 연관 방법을 적용하여 긴밀하게 간격이 좁은 목표물들을 구별합니다. 이 시스템은 MC Dropout 및 Loss Attenuation 기법을 사용하여 노이즈 모델링을 동적으로 수행하며, 트래킹 과정의 일관성과 신뢰성을 높입니다. 또한, Radar Tensor Network with Height (RTNH)를 다양한 레이어의 3D sparse convolution을 통해 4D Radar 데이터로부터 객체의 3D 위치를 효율적으로 추출합니다.

- **Performance Highlights**: K-Radar 데이터셋을 기반으로 평가한 결과, Bayes-4DRTrack는 전통적인 모션 모델과 고정 노이즈 공분산을 사용하는 방법들에 비해 5.7%의 Average Multi-Object Tracking Accuracy (AMOTA) 향상을 보여주었습니다. 이러한 성과는 악천후 환경에서의 물체 추적 정확도를 크게 개선하며, 실제 자율주행 응용을 위한 견고함을 입증합니다. 연구 결과는 4D Radar 기술이 복잡한 도로 상황에서도 우수한 성능을 발휘할 수 있는 가능성을 보여줍니다.



### Quasi-Conformal Convolution : A Learnable Convolution for Deep Learning on Riemann Surfaces (https://arxiv.org/abs/2502.01356)
- **What's New**: 이번 연구에서는 Riemann surface(리만 서피스)에 대한 컨볼루션을 정의하기 위해 quasi-conformal convolution(QCC)이라는 새로운 프레임워크를 도입합니다. 이는 복잡한 기하학적 구조를 가진 비유클리드 데이터 처리에 필수적이며, 다양한 공간적으로 정의된 컨볼루션을 통합하고 특정 작업에 맞게 최적화된 맞춤형 컨볼루션 연산자를 학습할 수 있도록 합니다. QCCNN(Quasi-Conformal Convolutional Neural Network)을 개발하여 기하학 데이터와 관련된 작업을 수행하고 높은 성능을 보여줍니다.

- **Technical Details**: QCC는 quasi-conformal mapping(준 일치 맵)을 기반으로 하여 리만 서피스에서의 컨볼루션 연산을 정의합니다. 각 QCC 연산자는 특정 quasi-conformal 매핑과 연결되어 있으며, 이 매핑을 조작하여 컨볼루션 작업을 조정할 수 있습니다. QCC는 데이터 구조에 적응하여 동적으로 조정 가능한 학습 가능한 컨볼루션 연산자를 제공합니다.

- **Performance Highlights**: QCCNN을 통해 커브형 리만 서피스에서의 이미지 분류 작업에서 우수한 성능을 발휘하며, 3D 얼굴 데이터에 대한 두개안면 분석 및 얼굴 병변 세분화에서도 기존 방법 대비 높은 정확도와 신뢰성을 보여줍니다. 다양한 파라미터 선택이 모델 성능에 미치는 영향을 평가하기 위해 자가 소멸 연구도 수행하며, 이 접근 방식의 강건성과 적응성을 입증합니다.



### ConceptVAE: Self-Supervised Fine-Grained Concept Disentanglement from 2D Echocardiographies (https://arxiv.org/abs/2502.01335)
- **What's New**: 이 논문에서는 ConceptVAE라는 새로운 프리트레인(pre-training) 프레임워크를 소개합니다. 이 프레임워크는 스타일 특성으로부터 세밀한 개념을 탐지하고 분리하는 자기 지도 학습(self-supervised learning) 방법으로, 의료 영상에서의 응용 가능성을 열어줍니다. ConceptVAE는 심장 초음파와 같은 2D 이미지를 활용하여 해부학적 구조를 자동으로 인식하고, 이러한 구조의 스타일을 파악하는 기능을 갖추고 있습니다.

- **Technical Details**: ConceptVAE는 변분 오토인코더(Variational Autoencoder, VAE) 프레임워크를 확장하여 2D 입력 이미지를 세밀한 개념과 스타일 벡터로 인코딩합니다. 이 과정에서 각 이미지 스타일의 상관관계를 분리하도록 설계된 여러 손실 함수(loss function)를 적용합니다. 기본적인 아이디어는 이미지의 로컬 컨셉을 구분하고 이에 대한 스타일 특성을 식별하는 것입니다.

- **Performance Highlights**: ConceptVAE는 기존의 전통적인 자기 지도 학습 방법보다 지역 기반 인스턴스 검색(region-based instance retrieval), 의미 분할(semantic segmentation), 물체 탐지(object detection) 및 분포 외 감지(out-of-distribution detection)와 같은 다양한 작업에서 일관된 성능 향상을 보여줍니다. 또한, ConceptVAE는 훈련 데이터와 동일한 개념을 유지하면서도 스타일이 다른 합성 데이터를 생성할 수 있는 가능성을 탐구하여, 보다 정확한 데이터 생성을 지원합니다.



### CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation (https://arxiv.org/abs/2502.01312)
- **What's New**: 이번 논문에서는 CleanPose라는 새로운 접근 방식을 제안하여, 카테고리 수준 객체 포즈 추정(category-level object pose estimation)에서 관찰되지 않은 혼란 변수(confounders)의 부정적 영향을 감소시키고자 합니다. 이는 인과적 학습(causal learning)과 지식 증류(knowledge distillation)를 결합하여, 보다 정확한 포즈 예측을 가능하게 합니다. CleanPose는 일반화 능력을 향상시키기 위해 다이나믹 큐를 사용하여 혼란 변수에 대한 사전과 지식을 지속적으로 업데이트합니다.

- **Technical Details**: CleanPose의 핵심은 프론트 도어 조정(front-door adjustment)을 기반으로 하는 인과적 추론 모듈입니다. 이를 통해 모델은 숨겨진 혼란 변수들의 영향을 최소화하고 보다 신뢰할 수 있는 포즈 추정을 수행합니다. 또한, 잔여 기반 지식 증류 방법을 사용하여 강력한 3D 기초 모델(ULIP-2)로부터 풍부한 카테고리 정보를 전달합니다.

- **Performance Highlights**: CleanPose는 다양한 벤치마크 데이터셋(REAL275, CAMERA25, HouseCat6D)에서 기존 방법들보다 우수한 성능을 보였습니다. 예를 들어, REAL275 데이터셋의 까다로운 5°2 cm 측정에서 61.5%의 정확도를 기록했으며, 이는 현재 최상의 방법보다 4.5%p 높은 수치입니다. 이처럼 CleanPose는 카테고리 내 일반화 능력을 향상시키며 뛰어난 결과를 입증했습니다.



### Heterogeneous Image GNN: Graph-Conditioned Diffusion for Image Synthesis (https://arxiv.org/abs/2502.01309)
- **What's New**: 이번 논문에서는 이질적인 그래프 데이터(heterogeneous graph data)를 사용한 새로운 이미지 합성(diffusion-based image synthesis) 모델을 소개합니다. 전통적인 방법들은 조건 변수를 모델 아키텍처에 직접 통합하는 데 한계를 보였으며, 특히 복잡한 다중 관계를 가진 조건 변수의 처리에 어려움을 겪었습니다. 이 연구에서는 Heterogeneous Image Graphs (HIG)라는 새로운 표현을 통해 조건 변수와 타겟 이미지를 두 개의 상호 연결된 그래프로 모델링합니다.

- **Technical Details**: HIG는 가변 길이의 조건 입력(variable-length conditioning inputs)과 그들 간의 관계를 효율적으로 처리할 수 있도록 도와줍니다. 이 과정에서는 ControlNet 접근 방식을 통해 기존 EDM2 확산 모델에 HIG를 통합하는 크기 보존 그래프 신경망(magnitude-preserving GNN)을 제안합니다. 이러한 기술적 접근은 그래프의 속성과 관계를 가장 직접적으로 나타내는 간선(edge) 및 노드(node)를 이용하여 조건을 가능하게 합니다.

- **Performance Highlights**: 본 연구는 COCO-stuff와 Visual Genome 데이터셋을 대상으로 다양한 조건 입력에 대한 성능을 개선했습니다. 연구 결과는 HIG를 활용하여 출력 이미지의 품질을 높이고, 기존 방법보다 우수한 성과를 보여주었습니다. 이러한 접근은 복잡한 관계를 다룰 수 있는 가능성을 열어주며, 이는 특히 다양한 조건 변수를 활용해야 하는 어플리케이션에서 중요하게 작용합니다.



### Partial Channel Network: Compute Fewer, Perform Better (https://arxiv.org/abs/2502.01303)
- **What's New**: 이번 연구에서는 네트워크의 정확도와 처리량을 저하시키지 않으면서 파라미터 수와 FLOPs를 줄일 수 있는 Partial Channel Mechanism (PCM)을 제안합니다. 이 혁신적인 접근법은 feature map 채널을 여러 부분으로 나누고, 각각의 부분에 다양한 연산을 적용한 후 결과를 통합하는 방식으로 작동합니다. 이를 통해 Partial Attention Convolution (PATConv)을 도입해 시각적 주의를 결합하여 모델 파라미터와 FLOPs를 절약할 수 있는 방법을 제시합니다.

- **Technical Details**: Partial Attention Convolution (PATConv)는 전통적인 합성곱(convolution)과 시각적 주의(visual attention)를 결합하여 모델 성능을 향상시키도록 설계되었습니다. 이 연구는 PATConv을 활용하여 새로운 유형의 블록인 Partial Channel-Attention block (PAT_ch), Partial Spatial-Attention block (PAT_sp), 및 Partial Self-Attention block (PAT_sf)를 개발하였습니다. 또한, 동적으로 채널 비율을 학습할 수 있는 동적 부분 합성곱(dynamic partial convolution, DPConv)도 제안되었습니다.

- **Performance Highlights**: 제안된 PartialNet은 ImageNet-1K 분류 문제에서 다른 SOTA 모델과 비교할 때 더 높은 top-1 정확도와 빠른 추론 속도를 보여줍니다. 이 모델은 COCO 데이터셋에서의 탐지 및 분할 작업에서도 뛰어난 성능을 발휘하며, 전반적으로 효율성을 높이고 정확도를 개선했습니다. 연구팀은 이 모델의 코드도 공개하여 다른 연구자들이 쉽게 접근할 수 있도록 하였습니다.



### XR-VIO: High-precision Visual Inertial Odometry with Fast Initialization for XR Applications (https://arxiv.org/abs/2502.01297)
- **What's New**: 이 논문은 Visual Inertial Odometry (VIO)의 초기화 및 특징 매칭 모듈에 새로운 접근법을 제안하고 있습니다. 기존 방법들은 시각적 Structure from Motion (SfM)에서 불안정하거나 다수의 매개변수를 동시에 해결하는 데 있어 취약함을 겪었습니다. 본 연구에서는 자이로스코프 측정을 밀접하게 결합하여 다양한 복잡한 시나리오에서도 견고하고 정확한 초기화 방안을 제시합니다. 이 방법은 단 네 개의 이미지 프레임만으로도 안정적인 성능을 보여주며, 증강 현실 및 가상 현실 (AR/VR) 분야에서의 실용성을 강조합니다.

- **Technical Details**: 이 시스템은 MSCKF와 유사한 표준적인 Visual Inertial Odometry 프레임워크를 따릅니다. 입력으로 이미지를 및 IMU 데이터를 받아 6DoF 포즈를 출력합니다. 초기화와 특징 매칭이라는 두 가지 핵심 모듈로 구성되며, 각각의 동작은 이 시스템의 정상 작동을 위한 필수 요소로 작용합니다. 초기화 과정은 시각적 및 관성적 측정을 사용하여 초기 상태를 계산하는 것을 포함하며, 여러 방식의 비교가 이루어진 서술에서 이러한 기법들이 다루어집니다.

- **Performance Highlights**: 기존 기준에 대한 평가를 통해 본 기법은 정확도와 성공률 면에서 최첨단의 성능을 보여주고 있습니다. Optical flow와 descriptor-based matching의 결합에 대한 새로운 하이브리드 접근법은 효율적이고 정확한 추적 결과를 달성합니다. 특히, 영상 기반의 소프트웨어 그래픽스 분야에서의 적용 가능성이 큰 특징으로 부각됩니다. 여러 벤치마크에서의 평가 결과, 본 연구는 성능의 우수성을 입증했습니다.



### Template Matching in Images using Segmented Normalized Cross-Correlation (https://arxiv.org/abs/2502.01286)
Comments:
          14 pages, 2 tables, 3 figures

- **What's New**: 이 논문에서는 이미지의 템플릿 매칭(template matching)에서의 정규화 교차 상관(normalized cross-correlation, NCC) 알고리즘의 새로운 변형을 제안합니다. 제안된 알고리즘은 템플릿 이미지와의 근사 NCC를 보다 효율적으로 계산할 수 있도록 템플릿 이미지 근사값을 사전 계산(precomputation)하는 방식으로 설계되었습니다. 이 과정에서 기존의 정밀 NCC 계산을 위한 원래 템플릿을 사용하는 대신, 근사 템플릿을 사용합니다.

- **Technical Details**: 근사 템플릿은 이미지를 분할(split)하고 결합(merge)하는 접근 방식을 통해 사전 계산됩니다. 이 결과, 각 세그먼트는 축과 정렬된 직사각형으로 분해되며, 세그먼트 크기는 각 세그먼트의 픽셀 강도 변화(variance)에 따라 달라집니다. 또한, 각 세그먼트는 원래 템플릿의 해당 픽셀에서 평균 그레이스케일 값(mean grayscale value)을 할당받습니다.

- **Performance Highlights**: 제안된 알고리즘은 잘 알려진 빠른 푸리에 변환(Fast Fourier Transform, FFT) 기반 NCC 알고리즘에 비해 미미한 NCC 근사 오류를 가지면서도 우수한 계산 성능을 달성합니다. 특히 시각적으로 복잡하지 않거나 작은 템플릿 이미지에 적용할 경우 성능이 더욱 두드러집니다. 경우에 따라 제안된 알고리즘은 FFT 기반 알고리즘의 범위 내에서 계산 성능이나 NCC 근사 오류 중 하나를 유지할 수 있지만 두 가지 모두를 동시에 만족시키기 어렵습니다.



### Label Correction for Road Segmentation Using Road-side Cameras (https://arxiv.org/abs/2502.01281)
- **What's New**: 이 논문에서는 다양한 기상 조건에서 도로 데이터를 자동으로 수집하기 위해 기존에 설치된 도로변 카메라 인프라를 이용하는 방법을 제안합니다. 새로운 반자동 주석(annotation) 방법을 통해 한 카메라의 한 프레임만 수작업으로 라벨링하고, 해당 카메라의 다른 프레임에 라벨을 전달합니다. 카메라 간의 작은 움직임은 주파수 도메인 이미지 등록을 통해 보정되어 라벨 전송의 정확성을 높입니다.

- **Technical Details**: 제안된 라벨 전송 방법은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 Fourier-Mellin을 사용하여 프레임 간의 이미지 등록 변환을 계산하고, 두 번째 단계에서는 대규모의 변환을 적용하여 최적 경로를 찾아 수동 주석을 다른 프레임으로 이동합니다. 이 과정에서 이미지는 전처리를 거친 후 주파수 도메인으로 변환되며, 픽셀 번역을 위한 상관관계가 계산됩니다.

- **Performance Highlights**: 핀란드 전역의 927개 카메라에서 수집한 데이터를 바탕으로 반자동으로 주석이 달린 데이터로 교육된 여러 딥러닝 세분화 모델의 성능이 향상되었습니다. 테스트는 도로변 카메라 데이터와 차량에 장착된 카메라 데이터의 두 가지 데이터 세트를 사용하여 수행되어, 제안된 방법의 강건성을 입증하였습니다.



### FSPGD: Rethinking Black-box Attacks on Semantic Segmentation (https://arxiv.org/abs/2502.01262)
- **What's New**: 이번 연구에서는 Feature Similarity Projected Gradient Descent (FSPGD) 공격이라는 새로운 블랙박스 방법을 제안합니다. 이 방법은 공격 성능과 전이성을 크게 향상시키며, 기존의 세그멘테이션 공격 방법이 출력 예측에 의존하는 것과 달리 중간 레이어의 특징에서 그래디언트를 계산합니다. FSPGD는 클린 이미지와 적대적 예제 간의 특징을 비교하여 지역 정보를 목표로 하는 손실 함수를 도입하고, 객체 간의 공간적 관계를 고려하여 맥락 정보를 파괴합니다.

- **Technical Details**: FSPGD 공격은 기존의 전이 공격 방법의 한계를 해결하는 데 중점을 두고 설계되었습니다. 이 방법은 중간 레이어에서 추출된 특징을 활용하여 그래디언트를 계산하며, 이로 인해 세그멘테이션 작업에서 매우 효과적인 공격이 가능합니다. 손실 함수는 클린 이미지와 적대적 예제의 특징을 비교하여 저주파수 정보를 취득하고, 객체 간의 관계를 통해 맥락 정보를 저해합니다.

- **Performance Highlights**: FSPGD 공격은 Pascal VOC 2012 및 Cityscapes 데이터셋에서 다양한 모델을 기준으로 실험했을 때 탁월한 전이성과 공격 성능을 보여주었습니다. 본 연구는 기존 방법들을 능가하는 새로운 최첨단 기준을 설정하였습니다. 또한, 여러 기초 모델 및 데이터셋에 대한 방대한 실험 수행과 여러 차원에서의 분석을 통해 제안된 방법의 일반화 능력을 입증합니다.



### Exploring Few-Shot Defect Segmentation in General Industrial Scenarios with Metric Learning and Vision Foundation Models (https://arxiv.org/abs/2502.01216)
- **What's New**: 이 논문에서는 제조 품질 관리를 위한 산업 결함 분할(segmentation)의 중요성이 강조되고 있습니다. 기존 연구들은 간단한 텍스처에서의 결함에만 초점을 맞추었으나, 본 연구는 다양한 산업 제품과 결함 유형을 대상으로 새로운 데이터셋을 제안하고 있습니다. 이를 통해 견고한 few-shot defect segmentation (FDS) 기준을 구축하고, 메트릭 학습 기반의 FSS 방법론을 깊게 탐구합니다.

- **Technical Details**: 이 연구에서는 메타 학습(method based on meta-learning)과 Vision Foundation Models (VFMs) 기반의 방법론을 활용하여 결함 분할을 수행합니다. 이를 위해 새로운 현실 산업 데이터셋을 만들고, 다양한 VFMs의 적용 가능성을 체계적으로 조사합니다. 특히, 특징 매칭(feature matching) 기반의 FDS 방법론을 제안하며, SAM2 모델의 비디오 추적 모드가 FDS 문제 해결에 특히 효과적임을 발견했습니다.

- **Performance Highlights**: 제안된 데이터셋과 모델은 기존의 텍스처 기반 데이터셋보다 더 많은 도전을 제공하여 FDS 연구의 범위를 넓히는 중요한 기초 자료로 작용할 것입니다. 연구 결과, 메타 학습 기반 방법들은 산업 결함 문제에 적합하지 않음을 보여주었고, VFMs가 더 큰 잠재력을 가지는 것으로 나타났습니다. 본 연구의 기여는 9개의 결함 카테고리에 대한 픽셀 수준 주석이 포함된 새로운 객체 기반 FDS 데이터셋뿐만 아니라, FDS 문제를 해결하기 위한 혁신적인 접근 방식을 다루고 있습니다.



### One-to-Normal: Anomaly Personalization for Few-shot Anomaly Detection (https://arxiv.org/abs/2502.01201)
Comments:
          In The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS2024)

- **What's New**: 본 논문에서는 기존의 Anomaly Detection (AD) 방법의 한계를 극복하기 위해 anomaly personalization 방법을 제안합니다. 이는 query 이미지와 anomaly-free 맞춤형 생성 모델을 통해 변환을 수행하여, 정상 manifold와 밀접하게 정렬되는 방식을 사용합니다. 또한, 안정성과 강인성을 높이기 위해 triplet contrastive anomaly inference 전략을 도입하여, query 이미지와 생성된 anomaly-free 샘플 간의 포괄적인 비교를 수행합니다.

- **Technical Details**: 제안된 방법은 diffusion model을 활용하여 anomaly-free 맞춤형 모델을 구축하고, 이를 기반으로 query 이미지를 하나의 정상 이미지로 변환합니다. 이어서, query 이미지와 맞춤형 버전, anomaly-free 샘플 및 텍스트 프롬프트 간의 포괄적인 비교를 수행하여 결과의 정확도를 높입니다. 이 과정에서 다양한 관점에서의 예측을 통합하여 최종적인 anomaly 점수를 생성합니다.

- **Performance Highlights**: 11개의 데이터셋과 3개 도메인에 걸쳐 시행된 포괄적인 실험을 통해 제안된 모델이 최신 AD 방법과 비교하여 효과적임을 입증하였습니다. 또한, 생성된 이미지 데이터는 다른 AD 방법의 성능을 향상시키는 데 유연하게 전이될 수 있는 것으로 나타났습니다. 이로 인해 기존 몇-shot AD 방법의 성능을 증대시키는 데 기여할 수 있음을 보여주고 있습니다.



### Nearly Lossless Adaptive Bit Switching (https://arxiv.org/abs/2502.01199)
- **What's New**: 이 논문에서는 모델 압축과 가속을 위한 모델 양자화의 중요성을 강조하며, 특히 다중 정밀도(multi-precision) 또는 한 번의 훈련(one-shot training) 방식으로 양자화를 수행하는 새로운 방법을 제안합니다. 기존의 Quantization-Aware Training(QAT)은 고정된 비트 너비(bit-width)에 초점을 맞추고 있었으나, 본 연구는 다양한 하드웨어와 전송 요구 사항에 대응하기 위해 비트 전환(bit-switching)을 효과적으로 수행할 수 있는 Double Rounding 방식을 도입합니다.

- **Technical Details**: Double Rounding 방법은 정밀도가 다른 양자화 과정에서 거의 손실 없는 비트 스위칭을 가능하게 하여, 고정밀 모델을 저장하는 대신에 가장 높은 정수 정밀도를 사용합니다. 또한, Adaptive Learning Rate Scaling(ALRS) 기법을 통해 다양한 정밀도에 맞게 학습률을 동적으로 조정하여 훈련 프로세스를 최적화하려고 합니다. 이러한 새로운 방법들은 더 높은 정밀도의 모델을 사용할 때의 경량화를 가능하게 하여, 획기적인 개선을 가져옵니다.

- **Performance Highlights**: ImageNet-1K 분류 실험 결과, 본 연구에서 제안한 방법이 다중 정밀도 및 혼합 정밀도(mixed-precision)에서 최신 기법들과 비교하여 충분한 장점을 가지는 것으로 나타났습니다. 비트 스위칭 과정에서의 정확도 저하를 방지하고, 다양한 정밀도 간의 경쟁 문제를 해결하는 등 다양한 응용 분야에서의 성공 가능성을 보여줍니다. 이 연구 결과는 감지(detection) 및 분할(segmentation) 작업과 대규모 언어 모델(LLMs)에 대한 적용 가능성도 입증했습니다.



### Towards Robust and Reliable Concept Representations: Reliability-Enhanced Concept Embedding Mod (https://arxiv.org/abs/2502.01191)
- **What's New**: RECEM(신뢰성 강화 개념 임베딩 모델)은 CBMs의 신뢰성을 높이기 위해 두 가지 주요 메커니즘을 도입합니다. 첫째, Concept-Level Disentanglement(개념 수준 분리)가 불필요한 특징과 개념 관련 정보를 분리하며, 둘째, Concept Mixup(개념 혼합) 방식이 서로 다른 샘플 간의 의미 일치를 보장합니다. 이러한 접근 방식은 모델이 중요한 객체 속성에 집중할 수 있도록 하고 신뢰할 수 있는 개념 표현을 생성합니다.

- **Technical Details**: CBMs는 입력 데이터로부터 인간이 이해할 수 있는 고급 개념을 예측하여 의사 결정 과정을 두 단계로 나누고, 이러한 예측된 개념을 사용하여 최종 클래스 레이블을 도출합니다. 그러나 기존의 CEMs는 배경 변화와 같은 개념과 무관한 특징에 민감하며, 동일 개념의 서로 다른 샘플 간에 의미적 일관성이 부족합니다. RECEM은 이러한 문제를 해결하기 위해 고안된 아키텍처이며, 보다 신뢰할 수 있는 개념 임베딩을 달성합니다.

- **Performance Highlights**: 실험 결과, RECEM은 여러 데이터셋에서 기존의 모든 기준선 방법들을 지속적으로 초월하는 성과를 보였습니다. 특히, 배경이 변경될 때 다른 모델들에 비해 뛰어난 성능을 보여주었으며, 개념적 정렬이 높고 전문가의 개입을 잘 수용할 수 있는 능력이 강화되었습니다.



### A High-Accuracy SSIM-based Scoring System for Coin Die Link Identification (https://arxiv.org/abs/2502.01186)
- **What's New**: 이 연구는 고대 동전 분석을 위한 새로운 접근 방식을 선보이며, 동전의 다이 링크를 식별하는 첫 번째 공개 라벨 데이터셋을 제공합니다. 이 데이터셋은 329장의 이미지를 포함하여 데이터의 기초를 제공하고 메소드의 벤치마킹을 용이하게 합니다. 또한, 이전 연구 방법들보다 더 빠르고 정확하게 동전 쌍을 구별하는 새로운 SSIM 기반의 점수화 방법을 제안합니다.

- **Technical Details**: 이 연구는 고대 동전에 대한 수집 분석의 효율성을 높이기 위한 기계 학습 및 컴퓨터 비전 알고리즘의 합리적 적용에 중점을 두고 있습니다. 특히, SSIM(Structural Similarity Index Measure)에 기반한 거리 계산 절차를 통해 동전 간 유사성을 더 잘 식별할 수 있는 방식이 소개됩니다. 기존 연구와 비교해 비교적 빠르고 정밀한 다이 링크 식별이 가능하다는 점을 강조합니다.

- **Performance Highlights**: 연구 결과, SSIM 기반의 새로운 거리 계산 기법이 기존 방법을 능가하여 거의 완벽한 다이 링크 식별 성능을 보였습니다. 제안된 방법은 특히 대규모 동전 수집에서 비효율적인 수작업을 대체할 수 있는 가능성을 보여줍니다. 향후 연구에서는 이 데이터셋과 알고리즘이 고대 동전 및 역사 연구를 위한 강력한 도구로 발전하는 것을 기대할 수 있습니다.



### Enhancing Environmental Robustness in Few-shot Learning via Conditional Representation Learning (https://arxiv.org/abs/2502.01183)
Comments:
          15 pages, 8 figures, Accepted by IEEE Transactions on Image Processing

- **What's New**: 본 논문은 기존의 few-shot learning (FSL) 모델이 실전에서 성능 저하를 겪는 이유로 '환경적 강건성'이라는 개념을 도입했습니다. 이를 해결하기 위해, 새로운 실제 다중 도메인 FSL 벤치마크를 제안하며, 다양한 도전 요소가 포함된 데이터셋을 제공합니다. 이러한 새로운 환경에서의 성능 평가가 기존 평가 방법들의 허점을 드러내는 중요한 기반이 될 것입니다.

- **Technical Details**: 제안된 조건부 표현 학습 네트워크(CRLNet)는 훈련 이미지와 테스트 이미지 간의 상호작용을 통합하여 더 효과적인 특징 표현을 생성하도록 설계되었습니다. CRLNet은 특징 추출기, 조건 학습기, 재표현 학습기 등 세 가지 주요 구성 요소로 이루어져 있으며, 대조 학습 손실(contrastive learning loss)을 포함하여 학습 과정에서의 방향성을 부여합니다. 이를 통해 각 카테고리의 본질적인 정보를 잘 포착하여 다양한 환경 속에서의 성능 일관성을 향상시킬 수 있도록 합니다.

- **Performance Highlights**: 비교 실험의 결과, CRLNet은 기존의 최첨단(Few-shot learning) 기술보다 평균 6.83%에서 16.98%까지 성능 향상을 보여주었습니다. 이는 다양한 설정과 기반 모델에서 발휘된 결과로, CRLNet의 일반화 능력을 강조합니다. 실험을 통해 각 모듈의 효과와 CRLNet의 전반적인 우수성을 입증하였으며, 새로운 환경에서도 높은 성능을 유지할 수 있다는 것을 보여주었습니다.



### BVINet: Unlocking Blind Video Inpainting with Zero Annotations (https://arxiv.org/abs/2502.01181)
- **What's New**: 이 논문에서는 비디오 복원 작업인 비디오 인페이팅(Video Inpainting)에 대한 새로운 접근 방식을 제안하고 있습니다. 기존 방법들은 손상된 영역의 위치를 알고 있는 경우에만 작동하며, 수동으로 이 영역을 주석 처리해야 합니다. 본 연구는 이를 개선하여 부팅이 필요 없는 '블라인드 비디오 인페이팅' 설정을 정의하였습니다.

- **Technical Details**: 제안된 BVINet는 손상된 비디오의 숨겨진 마스크를 예측하고, 이 마스크를 활용하여 유효한 컨텍스트 정보를 바탕으로 손상된 부분을 보완합니다. 이 과정에서 semantic-discontinuous regions를 감지하고, temporal consistency를 활용합니다. 또한, consistency loss를 통해 마스크 예측과 비디오 완성을 상호 제약하여 전반적인 모델 성능을 극대화합니다.

- **Performance Highlights**: 실험 결과, 본 메서드는 비주얼 프레임을 손상 없이 복원하고, 매우 다양한 비디오 맥락에 대해 효과적임을 확인하였습니다. 나아가, 1250개의 실제 비디오 클립을 포함한 맞춤형 데이터셋을 제공함으로써 후속 연구에 많은 기여를 할 것으로 기대됩니다. 실제로 비관리 대상 비교 실험에서도 기존 방법들과 유사한 성능을 보였습니다.



### Radiant Foam: Real-Time Differentiable Ray Tracing (https://arxiv.org/abs/2502.01157)
- **What's New**: 이번 연구에서는 전통적인 광선 기반 렌더링 대신 래스터화(rasterization)를 사용하는 스플래팅(splatting) 방법론이 인기를 얻고 있음을 강조합니다. 이에 따라 새로운 장면 표현 기술인 Radiant Foam을 제안하며, 이는 광선 추적(ray tracing) 알고리즘을 효과적으로 활용하여 기존 모델들보다 성능과 품질을 향상시킵니다. 이 방법은 렌더링 속도를 높이는 동시에 복잡한 빛의 전송 현상(light transport phenomena) 구현의 어려움을 회피합니다.

- **Technical Details**: Radiant Foam은 수십 년 간 잊혀진 효율적인 볼륨 메쉬(ray tracing 알고리즘을 기반으로 한) 기술을 활용합니다. 이는 스플래팅의 재구성 품질을 유지하면서 레스터화의 근본적인 약점들을 피할 수 있는 방법을 제공하고 있습니다. 또한, 기존의 Gaussian 모델과 달리 특별한 하드웨어나 API가 필요하지 않아 프로그래머블 GPU의 기본 기능만으로도 구현 가능하다는 장점이 있습니다.

- **Performance Highlights**: 새로운 모델은 Gaussian Splatting과 비슷한 렌더링 속도와 품질을 달성하면서도 레스터화의 제약 없이 작동합니다. 이러한 특성 덕분에 리얼타임(real-time) 그래픽스 응용 분야에서의 활용 가능성이 크게 향상되었습니다. 결과적으로 Radiant Foam은 기술적으로 더 발전된 장면 표현 방법을 제시하며, 높은 성능을 통한 혁신을 이루고 있습니다.



### LayerTracer: Cognitive-Aligned Layered SVG Synthesis via Diffusion Transformer (https://arxiv.org/abs/2502.01105)
- **What's New**: 이번 연구는 LayerTracer라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 디퓨전 트랜스포머(Difussion Transformer) 기반으로, 디자이너의 레이어드 SVG 생성 프로세스를 학습하여, 생성된 결과물이 전문가 수준의 편집 기준에 부합하도록 합니다. 특히, 텍스트 조건의 다중 단계 래스터화 설계도를 생성하고, 이를 통해 사용자가 쉽게 편집할 수 있는 SVG를 만들어냅니다.

- **Technical Details**: LayerTracer는 두 단계로 구성된 생성 파이프라인을 사용합니다. 첫 번째 단계에서는 텍스트 조건부 DiT가 래스터화된 생성 프로세스 시퀀스를 생성하고, 두 번째 단계에서는 레이어별 벡터화 모듈이 이를 클린하고 편집 가능한 SVG 레이어로 변환합니다. 또한, 라셀 이미지( raster images)를 레이어드 벡터 그래픽으로 변환하는 역 변환 작업도 다루며, 프리트레인된 DiT 모델에 LoRA 파인튜닝을 적용하여 이미지 컨텍스트를 효과적으로 반영합니다.

- **Performance Highlights**: LayerTracer는 기존의 최적화 기반 및 신경망 기반 방법들과 비교했을 때, 생성 품질 및 편집 가능성 모두에서 우수한 성능을 나타납니다. Extensive experiments를 통해 성능을 검증하며, AI가 생성한 벡터가 전문 디자인 인지와 효과적으로 정렬될 수 있음을 보여줍니다.



### VidSketch: Hand-drawn Sketch-Driven Video Generation with Diffusion Contro (https://arxiv.org/abs/2502.01101)
Comments:
          17pages, 15 figures

- **What's New**: VidSketch는 손으로 그린 스케치와 텍스트 프롬프트에서 직접 높은 품질의 비디오 애니메이션을 생성할 수 있는 첫 번째 방법으로, 기존의 정적 이미지 생성 기술의 한계를 극복했습니다. 이 방법은 비전문가들이 쉽게 애니메이션을 제작할 수 있도록 하여 예술적 요구를 충족시킵니다. 특히, Level-Based Sketch Control Strategy를 도입하여 다양한 사용자 수준의 드로잉 기술에 맞춰 스케치의 가이던스 강도를 조정할 수 있도록 설계되었습니다.

- **Technical Details**: 핵심 기술로는 Temporal Attention과 TempSpatial Attention 메커니즘이 포함되어 있습니다. TempSpatial Attention은 생성된 비디오 애니메이션의 시공간 일관성을 향상시켜 각 프레임 간의 연속성을 확보합니다. 손으로 그린 스케치의 추상화 수준을 동적으로 평가하여 비디오 생성 과정에서 제어 강도를 조절하는 방식으로 실현됩니다.

- **Performance Highlights**: VidSketch 모델을 평가한 결과, 손으로 그린 스케치와 잘 일치하면서 높은 비디오 품질과 미적 매력, 스타일 다양성 및 시공간 일관성을 유지하는 것으로 나타났습니다. 이 연구는 일상 사용자들이 비디오 애니메이션 제작에 접근할 수 있도록 하여 예술적 장벽을 제거하며, 다양한 고품질 결과를 제시하는 데 기여하고 있습니다.



### SatFlow: Generative model based framework for producing High Resolution Gap Free Remote Sensing Imagery (https://arxiv.org/abs/2502.01098)
- **What's New**: 새로운 연구에서는 지상 반사율 이미지를 고해상도로 제공하는 SatFlow라는 생성 모델 기반 프레임워크를 제안했습니다. 이 프레임워크는 낮은 해상도의 MODIS 이미지를 Landsat 관측 자료와 융합하여 정확한 결과를 생성합니다. 특히, 클라우드 오염 지역을 신뢰성 있게 채워줘서 다운스트림 애플리케이션(예: 작물 생장 추적)에서 유용하게 사용될 수 있습니다.

- **Technical Details**: SatFlow는 Conditional Flow Matching 기법을 기반으로 하여 저해상도 MODIS 이미지를 고해상도 Landsat유사 이미지로 변환합니다. 이 과정에서 모델은 Learned Generative Process를 활용하여 클라우드로 오염된 화소를 복원하고, 스캔 라인으로 인한 결손 부분을 메웁니다. 이를 통해 생성된 이미지는 빠르게 변화하는 농업 환경에서도 적용 가능하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, SatFlow는 클라우드로 덮인 영역을 효과적으로 보정할 수 있는 능력을 보여주었습니다. 이를 통해 MODIS의 풍부한 시간 정보를 Landsat의 세부 정보와 결합하여, 공간적 및 시간적으로 향상된 장기 데이터를 생성할 수 있습니다. 향후 농업 면적 모니터링 및 환경 변화 탐지와 같은 다양한 분야에 활용될 전망입니다.



### The Jumping Reasoning Curve? Tracking the Evolution of Reasoning Performance in GPT-[n] and o-[n] Models on Multimodal Puzzles (https://arxiv.org/abs/2502.01081)
- **What's New**: OpenAI의 o1 및 o3 모델 공개는 대규모 언어 모델(Large Language Models, LLMs)의 추론 능력에 큰 변화를 의미합니다. 특히 o3는 인공지능의 일반적 지능을 테스트하는 Abstraction and Reasoning Corpus(ARC-AGI)에서 인간을 초월하는 문제 해결 능력을 보였습니다. 그러나 symbolic pattern에 한정된 기존 벤치마크를 넘어, 다양한 시각 및 언어 데이터를 포함한 멀티모달 시나리오에 대한 탐구가 절실하다는 점이 강조되었습니다.

- **Technical Details**: 이 연구에서는 GPT-[n] 및 o-[n] 시리즈 모델의 멀티모달 퍼즐 해결 능력을 평가합니다. PuzzleVQA와 AlgoPuzzleVQA 데이터셋을 통해 추상 시각 추론과 알고리즘적 문제 해결을 측정하였으며, 각 데이터셋은 모델의 인지 능력을 면밀히 시험하는 구조적 특성을 지니고 있습니다. 특히, 멀티모달 퍼즐은 모델이 시각적 정보와 텍스트 정보를 통합하여 문제를 해결하는 능력을 평가하는 중요한 벤치마크 역할을 합니다.

- **Performance Highlights**: 모델 버전이 진행됨에 따라 추론 능력의 증가 경향이 뚜렷히 나타났으나, o1 모델은 여전히 단순 멀티모달 퍼즐에서 어려움을 겪고 있습니다. 알고리즘 퍼즐에서의 성능 저하도 관찰되며, 이는 현재의 인공지능이 인간의 추론 능력과 아직 큰 격차가 있음을 시사합니다. 연구진은 지속적으로 새로운 모델의 성능을 추적하고 연구 결과를 업데이트할 계획입니다.



### BC-GAN: A Generative Adversarial Network for Synthesizing a Batch of Collocated Clothing (https://arxiv.org/abs/2502.01080)
Comments:
          This paper was accepted by IEEE TCSVT

- **What's New**: 이 연구는 패션 아이템을 동시에 여러 개 생성할 수 있는 혁신적인 프레임워크인 BC-GAN을 소개합니다. 이전 연구들은 하나의 아이템에 대해 단일한 합성 결과만을 제공할 수 있었으나, BC-GAN은 이를 개선하여 사용자 맞춤형 의류 디자인을 가능하게 합니다. 또한, BC-GAN은 패션 호환성을 높이기 위해 새로운 호환성 판별기를 도입하여 증강된 패션 아이템 생성의 가능성을 보여줍니다.

- **Technical Details**: BC-GAN은 사전 학습된 모델을 기반으로 하여 의류 이미지를 임베딩으로 변환하고, 이를 통해 상부 및 하부 의류 도메인 간의 매핑을 정확하게 학습합니다. 이 모델은 패션 아이템 간의 공간적 비정렬 문제를 해결하며, 대규모 데이터셋인 DiverseOutfits에서 31,631 개의 호환되는 의상을 사용하여 평가되었습니다. 또한, 대조 학습 관점에서의 호환성 판별기를 통해 합성된 아이템의 패션 호환성을 더욱 향상시킵니다.

- **Performance Highlights**: BC-GAN은 기존의 최신 방법들과 비교하여 다양성, 시각적 진정성, 패션 호환성 측면에서 우수한 성능을 보였습니다. 대규모 실험을 통해 BC-GAN이 제공하는 의류의 질과 다양성이 우수함을 입증하였으며, 여러 패션 아이템의 합성을 효과적으로 수행할 수 있음을 확인했습니다. 이러한 결과는 BC-GAN이 패션 디자이너들에게 실질적인 영감을 주고, 산업의 수익성을 증가시킬 수 있는 강력한 도구가 될 것임을 시사합니다.



### OmniHuman-1: Rethinking the Scaling-Up of One-Stage Conditioned Human Animation Models (https://arxiv.org/abs/2502.01061)
Comments:
this https URL

- **What's New**: OmniHuman 모델은 새로운 Diffusion Transformer 기반의 프레임워크로, 모션 관련 조건을 혼합하여 훈련 데이터의 규모를 확장합니다. 이는 사람의 동작 생성을 데이터 기반으로 활용함으로써 현실적인 인간 비디오 생성을 가능하게 합니다. 기존의 종단 간 오디오 기반 방법들과 비교할 때, OmniHuman은 더 사실적인 비디오를 생성할 뿐만 아니라 다양한 입력 소스를 처리할 수 있는 유연성을 제공합니다.

- **Technical Details**: OmniHuman은 텍스트, 오디오 및 포즈와 같은 여러 조건 신호를 통합하는 혼합 조건 훈련 전략을 기반으로 합니다. 이 전략은 두 가지 주요 원칙을 따르며, 더 강력한 조건이 더 약한 조건을 활용하여 데이터 낭비를 줄이는 방식을 취합니다. 이를 통해 OmniHuman은 데이터 필터링의 한계를 극복하고 다양한 입력 형식을 지원할 수 있는 자연스러운 모션 패턴을 학습합니다.

- **Performance Highlights**: OmniHuman은 다양한 인물 콘텐츠(얼굴 근접, 초상화, 반신 상, 전신)를 지원하며 말을 하거나 노래를 부르는 기능을 제공합니다. 이전의 모델들과 비교하여 제스처 생성을 크게 개선하였으며, 다양한 이미지 스타일에서도 우수한 성능을 발휘합니다. 이는 오디오 기반 인간 비디오 생성 방법들보다 현저히 우수한 결과를 보여줍니다.



### Mitigating Hallucinations in Large Vision-Language Models with Internal Fact-based Contrastive Decoding (https://arxiv.org/abs/2502.01056)
- **What's New**: 이번 논문은 Internal Fact-based Contrastive Decoding (IFCD)이라는 새로운 접근 방식을 제안하여 LVLMs의 오브젝트 환각(object hallucinations) 문제를 해결합니다. IFCD는 기계 학습 모델에 통합할 수 있으며, 시각적 입력에 의거한 언어 출력의 정확성을 높이기 위해 LVLMs 자체의 환각을 활용합니다. 기존의 기술들이 높은 비용을 요구하는 반면, IFCD는 상대적으로 간단하면서도 효과를 발휘할 수 있는 방법이라고 강조합니다.

- **Technical Details**: IFCD는 LVLMs 내의 표현을 수정하여 내부적으로 생성된 왜곡된 분포를 활용하는 방법론입니다. 구체적으로, 내부 표현의 편집을 통해 진실성과의 차이를 지닌 두 개의 분포를 구성하며, 이러한 차이를 통해 오브젝트 환각을 감소시키는 대조적 디코딩(contrastive decoding) 방식을 적용합니다. 이 방법은 사전 학습 단계에서 얻은 데이터를 사용하지 않고, LVLM 시스템에 치명적인 성능 저하를 가져오는 환각을 효과적으로 감소시키는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과는 IFCD가 POPE 및 MME 객체 환각 서브셋에서 평균적으로 각각 9% 및 8%의 정확도 향상을 달성하는 것을 보여줍니다. LVLM의 출력 분포를 조정하고 환각 출력을 줄이는 데 있어 IFCD의 효과성을 입증하며, 텍스트 생성 작업에서 생성된 텍스트의 품질을 유지하면서 환각된 객체 비율을 5% 감소시킬 수 있음을 나타냈습니다. 이러한 연구 결과는 LVLMs의 신뢰성과 성능 향상에 기여할 수 있음을 시사합니다.



### Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization (https://arxiv.org/abs/2502.01051)
Comments:
          20 pages, 14 tables, 15 figures

- **What's New**: 본 연구에서는 Diffusion 모델을 위한 Latent Reward Model (LRM)를 제안하여, 단계별로 사람의 선호도를 예측하는 방법을 제시합니다. 기존 모델들이 겪는 다양한 문제점을 해결하기 위해, LRM은 노이즈가 포함된 이미지를 효과적으로 처리 및 학습할 수 있는 구조로 설계되었습니다. 이어서 LRM을 기반으로 새로운 Latent Preference Optimization (LPO) 기법을 도입하여, 모델 최적화를 더욱 간소화하고 속도를 획기적으로 향상시키고 있습니다.

- **Technical Details**: LRM은 UNet에서 시각적 특징을 추출하고 텍스트 인코더에서 텍스트 특징을 활용하여, 다양한 시간 단계에서의 선호도를 예측합니다. 이 방식은 기존의 Vision-Language Models (VLMs) 기반 방법들이 겪었던 복잡한 변환 및 노이즈로 인한 문제점을 피할 수 있도록 설계되었습니다. 또한, Visual Feature Enhancement (VFE) 모듈과 Multi-Preference Consistent Filtering (MPCF) 전략을 도입하여 모델의 성능을 더욱 강화하고 있습니다.

- **Performance Highlights**: LPO는 기존의 방법들과 비교하여 2.5배에서 28배의 훈련 속도 향상을 이뤘으며, 일반적인 이미지 품질 뿐 아니라 미적 선호도 및 텍스트-이미지 정렬에서도 뛰어난 성능을 발휘합니다. 실험 결과, LPO는 DPO와 SPO 방법들과 비교해 지속적으로 우수한 결과를 보여주었으며, 이는 diffusion 모델의 품질을 크게 향상시킵니다.



### Sparks of Explainability: Recent Advancements in Explaining Large Vision Models (https://arxiv.org/abs/2502.01048)
Comments:
          Doctoral thesis

- **What's New**: 이 논문은 컴퓨터 비전의 설명 가능성을 높이는 고급 접근 방식을 다루고 있습니다. 특히, 딥 러닝 네트워크가 이용하는 특징들을 분석하고 모델링하여 새로운 방법을 제안하고 있습니다. 여기에는 알고리즘 안정성에 기반한 메트릭과 Sobol 지수를 활용한 계산 속도 향상 기법이 포함됩니다.

- **Technical Details**: 이 연구에서는 saliency maps와 같은 귀속 방법(attribution methods)을 평가하며, 알고리즘 안정성을 바탕으로 한 새로운 메트릭을 도입합니다. 또한, EVA 방법은 검증된 섭동 분석(verified perturbation analysis)을 통해 귀속에 대한 공식적인 보장을 처음으로 제시합니다. 실험 결과는 복잡한 상황에서 이러한 방법들이 모델이 '어디에' 집중하는지 파악할 수 있지만, '무엇을' 인식하는지는 명확히 설명하지 못한다는 것을 보여줍니다.

- **Performance Highlights**: 논문은 두 가지 가설을 탐구합니다: 인간의 추론을 모델과 일치시키기 위한 훈련 루틴의 도입과 개념적 설명 가능성 접근 방식의 채택입니다. CRAFT 방법은 모델이 사용하는 개념을 자동으로 추출하고 그 중요성을 평가하며, MACO는 이를 시각화하는 기능을 제공합니다. 이러한 연구들은 ResNet 모델의 1000개 ImageNet 클래스에 적용된 상호작용 시연을 통해 통합된 프레임워크로 수렴하게 됩니다.



### WonderHuman: Hallucinating Unseen Parts in Dynamic 3D Human Reconstruction (https://arxiv.org/abs/2502.01045)
- **What's New**: WonderHuman 프로젝트는 단안 비디오(monocular video)에서 동적인 인간 아바타를 고해상도로 재구성하는 혁신적인 접근 방식을 제시합니다. 기존의 방법들은 인간 몸체의 모든 부분을 촘촘히 촬영한 다각적 비디오를 요구했지만, 본 연구는 제한된 시점의 비디오에서도 뛰어난 성능을 발휘할 수 있도록 설계되었습니다. 특히 2D generative diffusion 모델을 활용하여 보이지 않는 신체 부위를 정확하게 렌더링하는 방법론을 도입했습니다.

- **Technical Details**: WonderHuman은 Dual-Space Optimization 기법을 적용하여 Score Distillation Sampling (SDS)을 관찰 공간(observation space)과 표준 공간(canonical space) 모두에서 활용합니다. 이를 통해 동적 인체의 시각적 일관성을 유지하고 사실감을 증가시키며, 나타나는 동작에 따른 변화를 잘 반영할 수 있도록 합니다. 포즈 기능 주입(pose feature injection) 및 시점 선택 전략(view selection strategy)도 적용하여 예측된 SDS 결과와 관찰된 데이터를 통합하여 아바타의 충실도를 높입니다.

- **Performance Highlights**: 실험 결과, WonderHuman은 주어진 단안 비디오에서 특히 보이지 않는 부위에 대한 렌더링에서 State-of-the-art(SOTA) 성능을 달성했습니다. 연구진은 여러 데이터셋(ZJU-Mocap, Monocap 등)을 활용하여 본 접근법의 효과를 검증하며, 복잡한 동작에 대한 고품질의 현실감 있는 재구성을 입증하였습니다. 최종적으로, WonderHuman은 동적 인간 아바타의 고해상도 재구성을 위한 혁신적인 방법을 제시하여, 다양한 응용 프로그램에서 활용될 가능성이 큽니다.



### Vessel segmentation for X-separation (https://arxiv.org/abs/2502.01023)
- **What's New**: 이 논문은 $	extchi$-separation이라는 고급 정량적 감수성 매핑 (QSM) 방법을 통해 뇌에서 철과 마이엘린의 분포를 반영하는 매핑을 생성하는 새로운 혈관 분할 방법을 소개합니다. 이 방법은 혈관 기하학에 의해 유도된 지역 성장(region growing) 과정을 포함해 세 단계로 구성되어 있습니다. 기존의 방법들과 비교하여 더 높은 정확성을 보여주며 혈관이 아닌 구조를 효과적으로 제외하는 개선된 결과를 제공합니다.

- **Technical Details**: 혈관 분할 방법은 $	extit{R}_2^*$ 맵과 $	extchi_{para}$ 및 $|	extchi_{dia}|$ 맵의 곱을 기반으로 한 시드(seed) 생성을 시작으로 합니다. 이후 혈관 기하학에 따라 지역이 성장하고, 마지막으로 비혈관 구조를 제외하여 혈관 마스크의 정제를 수행합니다. 이 방법은 정량적 평가(q quantitative evaluation)와 인구 평균 영역 분석(population-averaged ROI analysis) 두 가지 응용 분야에서 성능을 시험했습니다.

- **Performance Highlights**: 제안된 방법은 기존 혈관 분할 방법에 비해 최고 수준의 Dice 점수 계수를 달성하며, $	extchi$-sepnet-$	extit{R}_2^*$의 정량적 평가에서 유의미한 개선을 보고했습니다. 또한 인구 평균 ROI 분석에서도 통계적으로 유의미한 차이를 나타내어 혈관을 제외할 때 $	extchi$-separation 맵 분석에서 더 정확한 평가를 가능하게 합니다. 이는 향후 다양한 응용 분야에 응용될 수 있는 가능성을 제공합니다.



### ZeroBP: Learning Position-Aware Correspondence for Zero-shot 6D Pose Estimation in Bin-Picking (https://arxiv.org/abs/2502.01004)
Comments:
          ICRA 2025

- **What's New**: 본 논문에서는 로봇의 bin-picking 작업을 위한 제로샷 6D 포즈 추정 방법인 ZeroBP를 제안합니다. ZeroBP는 CAD 모델과 장면 인스턴스 간의 Position-Aware Correspondence (PAC)를 학습하여, 텍스처가 없고 불확실한 지역에서 발생하는 불일치 문제를 해결합니다. 이 방법은 로컬 특성과 글로벌 포지션을 모두 활용하여 포즈 추정 정확도를 크게 향상시킵니다.

- **Technical Details**: ZeroBP는 초기 포즈에서 시작하여 포즈와 글로벌 포지션을 교대로 반복적으로 정제하는 방식으로 작동합니다. 복합적인 위치 인코딩 다음에 포지션 인식 교차 주의를 설계하여 로컬 특징과 위치 인코딩을 결합하여 강력한 상관관계를 모델링합니다. 이 과정에서, 여러 포인트 간의 유사성 때문에 발생할 수 있는 혼잡한 정보는 극복되고 더욱 정확한 포즈 추정을 가능하게 합니다.

- **Performance Highlights**: ROBI 데이터셋에 대한 실험 결과, ZeroBP는 기존의 제로샷 포즈 추정 방법보다 9.1% 향상된 올바른 포즈의 평균 회수를 기록하며 성능을 크게 개선했습니다. 이러한 결과는 ZeroBP가 기존의 state-of-the-art 방법들보다 우수한 성능을 나타내는 것을 보여줍니다. 이는 제조된 대상 물체의 포즈 추정이 특히 도전적인 bin-picking 작업에서 높은 정확도를 달성하는 데 기여합니다.



### Multi-Resolution SAR and Optical Remote Sensing Image Registration Methods: A Review, Datasets, and Future Perspectives (https://arxiv.org/abs/2502.01002)
Comments:
          48 pages, 10 figures

- **What's New**: 이번 연구는 Synthetic Aperture Radar (SAR)와 광학 이미지 등록의 중요성을 강조하고 있습니다. 특히 군사 정찰(military reconnaissance), 환경 모니터링(environmental monitoring) 및 재해 관리(disaster management)에서의 원격 감지 데이터 융합(fusion)에 필수적인 기술입니다. 이를 위해 MultiResSAR 데이터셋을 새롭게 구축하여 10,000개 이상의 다중 소스(multi-source) 및 다중 해상도(multi-resolution) SAR 및 광학 이미지 페어를 제공합니다.

- **Technical Details**: SAR와 광학 이미지 간의 등록은 이미지 메커니즘의 차이, 기하학적 왜곡(geometric distortions), 방사 성질(radiometric properties)의 차이 등 여러 도전과제를 동반합니다. 연구에서는 16개의 최첨단 알고리즘을 테스트하였으며, 그 중 XoFTR과 RIFT가 각각 딥러닝(deep learning)과 전통적인 방법에서 가장 높은 성능을 나타냈습니다. 해상도가 증가할수록 등록 성능이 저하되며, 대부분의 알고리즘이 1미터(sub-meter) 이하의 데이터에서 실패했습니다.

- **Performance Highlights**: 테스트 결과, 성능이 프레임 해상도에 따라 감소하며 100% 성공한 알고리즘은 없었습니다. XoFTR은 딥러닝 방법 중 가장 낮은 성공률인 40.58%를 기록했으며, RIFT는 전통적 방법에서 66.51%의 성과를 보였습니다. 이러한 결과는 앞으로 고해상도 SAR 및 광학 이미지를 강력하게 등록하기 위한 노이즈 억제(noise suppression) 및 3D 기하학적 융합(3D geometric fusion) 연구의 필요성을 강조하고 있습니다.



### Adapting Foundation Models for Few-Shot Medical Image Segmentation: Actively and Sequentially (https://arxiv.org/abs/2502.01000)
- **What's New**: 본 논문에서는 FSDA(기능 적은 도메인 적응)에 대한 새로운 접근법인 ASAP(적극적 및 순차적 도메인 적응) 프레임워크를 제안합니다. 이 프레임워크는 동적으로 보조 데이터셋을 선택하여 단일 라운드의 파인튜닝을 통해 도메인 적응 문제를 해결하는 방식을 채택합니다. 많은 의료 이미지 세분화 데이터셋에서의 실증적 검증을 통해 ASDA 프레임워크가 기존 FSDA 방법보다 월등한 성능을 보였음을 입증했습니다.

- **Technical Details**: ASAP 프레임워크는 FSDA를 다중암 밴딧 문제로 포뮬레이트하고, 실행할 보조 데이터셋의 선택 전략을 정의합니다. 강화 학습의 기초가 되는 MAB(multi-armed bandit)를 통해, 각 턴마다 관찰되지 않은 데이터셋을 탐색하고 높은 보상을 얻을 수 있는 데이터셋을 활용하는 방식을 통해 강화합니다. 이러한 방식은 과적합(overfitting)을 방지하고 유의미한 표현을 캡처하는 데 도움을 줍니다.

- **Performance Highlights**: 시스템 성능을 평가한 결과, MRI 및 CT 데이터셋에서 각각 평균 27.75% 및 7.52%를 Dice 점수 기준으로 개선하여 기존 FSDA 방법을 크게 초월했습니다. 실행 비용도 상대적으로 낮아 다양한 의료 이미지 세분화 작업에 효과적으로 적용될 수 있음을 보여줍니다. 따라서, 이 방법은 의료 자원의 최적 활용에도 기여할 수 있습니다.



### FCBoost-Net: A Generative Network for Synthesizing Multiple Collocated Outfits via Fashion Compatibility Boosting (https://arxiv.org/abs/2502.00992)
Comments:
          This paper has been accepted for presentation at ACM Multimedia 2023

- **What's New**: 이 논문에서는 새로운 의상 생성 프레임워크 FCBoost-Net을 제안합니다. 이 프레임워크는 사전 훈련된 생성 모델의 힘을 활용하여 다양한 의상을 생성할 수 있도록 설계되었습니다. 기존 연구는 단일한 의상 세트만 제공했지만, FCBoost-Net은 여러 개의 호환 가능하고 다양한 의상 세트를 생성하는 것을 목표로 하고 있습니다.

- **Technical Details**: FCBoost-Net은 GAN inversion 기법을 사용하여 패션 아이템 간의 공간적 비정렬을 줄이는 동시에, 패션 호환성을 높이는 부스팅 전략을 채택합니다. 이 프레임워크는 이미지 데이터를 잠재 공간(latent space)으로 인코딩하여 훈련 복잡성을 줄이며, 패션 아이템 세트를 입력으로 받아 다수의 호환된 세트를 출력합니다. 이 모델은 기존의 이미지-이미지 변환 방법들과는 달리, 세트 단위의 비주얼 호환성을 유지하면서도 다양성을 극대화합니다.

- **Performance Highlights**: 실험 결과, FCBoost-Net은 패션 아이템 간의 호환성과 다양성을 모두 유지하면서도 시각적 진정성을 높일 수 있음을 확인했습니다. 이 연구는 의상 생성 분야에 있어 사용자에게 더 많은 선택지를 제공하여 패션 디자인을 더욱 풍부하게 할 수 있는 가능성을 열어줍니다. 특히, 이 모델은 의상 생성 작업에서의 신뢰성과 다중 선택의 필요성을 충족하는 데 중점을 두고 개발되었습니다.



### Pushing the Boundaries of State Space Models for Image and Video Generation (https://arxiv.org/abs/2502.00972)
Comments:
          21 pages, paper under review

- **What's New**: 이 논문에서는 Vision에서 발생하는 비대칭 순환 구조를 가진 상태 공간 모델(State Space Models, SSM)과 Transformer 아키텍처의 하이브리드 모델을 제안합니다. 기존 Transformer 모델의 높은 계산 복잡성을 보완하기 위해 SSM의 효율성을 활용하여 5억 개의 파라미터를 가진 대규모 확산 SSM-Transformer 하이브리드 모델을 구축했습니다. 이 모델은 최대 2K의 이미지와 360p 8초 비디오를 생성할 수 있습니다.

- **Technical Details**: 여기서 제안된 모델은 Bi-directional SSM(Hydra)와 Transformer 아키텍처를 결합하여 만들어졌습니다. 특히, 이미지 생성에 최적화된 수평 및 수직 레스터 스캔 패턴을 채택하였으며, 비디오 생성을 위한 모델은 특정 레이어의 스캔 순서를 시간 우선으로 변경하여 토큰의 시간적 진화를 학습하고자 하였습니다. 이러한 접근은 고유한 공간적 및 시간적 패턴을 요구하는 비디오 생성에서의 성능 향상을 목표로 합니다.

- **Performance Highlights**: 실험 결과, HTH 모델은 복잡한 텍스트 프롬프트에 잘 맞는 사실적인 결과를 생성하며, 시간적으로 일관되고 동적인 비디오를 생성하는 데 성공했습니다. 이러한 결과는 SSM이 비주얼 생성 작업에 있어서 큰 잠재력을 가지고 있음을 시사하며, 특히 긴 비주얼 시퀀스를 생성하는 데에 특히 유망한 결과를 보여줍니다.



### CoDe: Blockwise Control for Denoising Diffusion Models (https://arxiv.org/abs/2502.00968)
- **What's New**: 이 논문에서는 gradient-free guidance 방식인 controlled denoising (CoDe)을 제안합니다. CoDe는 기존 모델을 미세 조정할 필요 없이 개입 단계에서 샘플링할 수 있도록 돕습니다. CoDe는 사용자의 보상에 대한 선호도를 반영하여 간편하면서도 효과적인 성능을 보여줍니다.

- **Technical Details**: CoDe는 블록 단위 샘플링 방법으로, 각 블록에서의 denoising 단계를 제어하는 기법입니다. 이 방법은 KL divergence를 최소화하면서 보상을 극대화하는 최적화된 목표로부터 샘플링합니다. 또한, 다섯 가지 다양한 이미지 생성 시나리오에서 두 가지 사례 연구(Gaussian Mixture Model 및 이미지 생성)를 통해 성능을 평가합니다.

- **Performance Highlights**: 실험 결과, CoDe는 기존의 최첨단 기법들에 비견할 만한 경쟁력을 보이며, 보상 정렬과 신속한 지시 이행, 추론 비용 간의 균형을 이룹니다. 이러한 결과는 CoDe가 복잡한 데이터 분포를 효과적으로 모델링하고, 사용자 요구에 맞게 빠르게 최적화할 수 있음을 보여줍니다.



### CLIP-UP: A Simple and Efficient Mixture-of-Experts CLIP Training Recipe with Sparse Upcycling (https://arxiv.org/abs/2502.00965)
- **What's New**: CLIP-Upcycling (CLIP-UP)은 사전 훈련된 밀집 CLIP 모델을 희소 Mixture-of-Experts (MoE) 구조로 전환하는 효율적인 훈련 전략을 제안합니다. 이전의 방법들과 비교해 CLIP-UP는 훈련 복잡성과 비용을 크게 줄이며, COCO 및 Flickr30K의 텍스트-이미지 Recall@1 벤치마크에서 밀집 모델 대비 7.2% 및 6.6%의 성능 향상을 보입니다. 또한, 이 방법은 다양한 크기의 모델에서도 일반화 가능성을 보여줍니다.

- **Technical Details**: MoE는 입력당 모델의 일부 전문가만 활성화하여 밀집 모델 대비 추론 비용을 낮추는 기법입니다. CLIP-UP은 사전 훈련된 밀집 모델의 가중치를 활용하여 이를 기반으로 한 희소 모델을 생성합니다. 이 과정에서 CLIP-UP은 초기 훈련을 통해 효율성을 높이고, 복잡한 보조 손실 없이도 훈련할 수 있습니다.

- **Performance Highlights**: CLIP-UP를 사용한 B/16 모델은 밀집 CLIP 모델에 비해 COCO 및 Flickr30K에서 각각 7.2% 및 5.5% 향상된 Recall@1 점수를 기록했습니다. 이 모델은 L/14 모델보다 더 큰 성능을 보이면서도 30%의 추론 FLOPs만 사용합니다. 이러한 성과는 CLIP-UP이 다양한 모델 크기에서의 확장성이 뛰어난 것을 입증합니다.



### SAM-guided Pseudo Label Enhancement for Multi-modal 3D Semantic Segmentation (https://arxiv.org/abs/2502.00960)
Comments:
          ICRA 2025

- **What's New**: 본 논문에서는 다양한 도메인 적응 시나리오에서 멀티 모달 3D 의미 분할(Multi-modal 3D semantic segmentation) 성능을 향상시키기 위한 새로운 방법을 제시합니다. 특히, Segment Anything Model(SAM)의 2D 정보를 활용하여 고품질의 pseudo-labels를 생성하고 이를 통해 도메인 전이 성능을 강化하는 방안을 제안합니다.

- **Technical Details**: 제안하는 방법은 두 가지 단계로 구성된 mask-wise pseudo-label 개선 프레임워크를 사용합니다. 첫째, majority voting을 통해 클래스 레이블을 결정하고, 여러 제약 조건을 통해 신뢰할 수 없는 레이블을 필터링합니다. 둘째, Geometry-Aware Progressive Propagation(GAPP)을 통해 3D 포인트에 대해 레이블을 전파하되, 2D-3D 불일치(outliers)를 피하는 방식으로 성능을提高합니다.

- **Performance Highlights**: 다양한 데이터셋과 도메인 적응 작업에서 실험을 수행한 결과, 제안한 방법이 고품질의 pseudo-labels 수를 효과적으로 증가시키며, 기존 도메인 적응 방법들에 비해 현저한 성능 향상을 달성하는 것을 확인할 수 있었습니다.



### Hypo3D: Exploring Hypothetical Reasoning in 3D (https://arxiv.org/abs/2502.00954)
Comments:
          19 pages, 15 figures, 9 tables

- **What's New**: 이번 연구에서는 Hypothetical 3D Reasoning(Hypo3D)이라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 모델이 실시간 3D 장면 데이터에 접근하지 않고도 추론하는 능력을 평가하도록 설계되었습니다. Hypo3D는 모델이 제공된 변화를 기반으로 장면 상태를 상상하고 그에 따라 질문에 대한 답변을 조정해야 한다는 점이 특징입니다.

- **Technical Details**: Hypo3D는 7,727개의 문맥 변화(context changes)와 14,885개의 질문-답변 쌍(question-answer pairs)으로 구성된 데이터셋을 포함하며, 700개의 실내 장면을 사용합니다. 이 데이터셋은 Movement Change(움직임 변화), Removal Change(제거 변화), Attribute Change(속성 변화), Addition Change(추가 변화), Replacement Change(대체 변화)와 같은 다섯 가지 카테고리를 포함합니다. 모델은 질문을 통해 상대적 위치, 방향 기반 질문, 그리고 세부적 의미를 파악해야 합니다.

- **Performance Highlights**: 10개의 최신 모델에 대한 실험 결과, 모델과 인간 간에 Hypo3D 작업에서 성능 차이가 심각하게 나타났습니다. 특히, 움직임 변화 및 방향 추론에서 큰 격차가 관찰되었습니다. 이 연구는 현재 모델들이 상상력을 활용하여 결측된 인식 지식을 추론하고 가상의 상황에서 문제를 해결하는 데 한계가 있음을 보여줍니다.



### Fruit Fly Classification (Diptera: Tephritidae) in Images, Applying Transfer Learning (https://arxiv.org/abs/2502.00939)
Comments:
          15 pages and 19 figures

- **What's New**: 이 연구는 자동화된 분류를 위한 전이 학습 모델을 개발했습니다. 과거에는 전문가들이 수작업으로 분류했으나, 이는 시간과 정확성에서 한계가 있었던 문제를 해결하기 위해 AI 기반의 접근법을 제안합니다. 특히 모바일 카메라와 스테레오 현미경을 활용하여 고품질 이미지를 캡처하고, 이를 통해 데이터셋을 생성했습니다.

- **Technical Details**: 연구에서는 VGG16, VGG19, Inception-v3와 같은 사전 훈련된 합성곱 신경망 모델을 통해 학습을 수행했습니다. 이미지는 중요 형태학적 영역에 초점을 맞춰 세분화되고 라벨링되었습니다. 이 과정에서 F1-score 평가를 통해 Inception-v3가 93%로 최고의 성능을 보여주었습니다.

- **Performance Highlights**: Inception-v3의 신뢰성은 통제되지 않은 환경에서 모델을 테스트해 확인되었으며, Grad-CAM 기법을 사용하여 본질적인 형태학적 특징을 포착하는 능력을 입증했습니다. 이러한 결과는 Anastrepha fraterculus와 Ceratitis capitata를 분류하는 데 있어 Inception-v3가 효과적이고 재현 가능하다는 것을 나타내며, 자동화된 모니터링 시스템에서의 구현 가능성을 제시합니다.



### LoR-VP: Low-Rank Visual Prompting for Efficient Vision Model Adaptation (https://arxiv.org/abs/2502.00896)
- **What's New**: 이번 연구에서는 기존 시각 프롬프트 기법의 한계를 극복하기 위해 새로운 시각 프롬프트 디자인인 Low-Rank matrix multiplication for Visual Prompting (LoR-VP)를 제안하였습니다. 기존의 기술은 프롬프트 매개변수를 이미지 주변에 패딩하는 방식으로 제한적이며, 이는 이미지 내의 특정 패치와의 상호작용을 최소화하게 됩니다. LoR-VP는 이미지 픽셀의 행과 열을 통한 공유 정보와 패치별 특화 정보를 가능하게 하여 이러한 한계를 극복합니다.

- **Technical Details**: LoR-VP 기법은 이미지의 모든 패치에 영향을 미치며, 패치 간의 행렬 곱을 통해 인덕티브 바이어스(inductive bias)를 도입하여 성능을 향상시킵니다. 이 방법은 모델의 해석력을 높이려는 기존 방법들과는 달리 모든 패치에서 프롬프트를 균일하게 최적화하며, 이를 통해 모델이 전 지구적(holistic) 컨텍스트를 보다 효과적으로 캡처할 수 있게 합니다. 실험 결과, LoR-VP는 최첨단 시각 프롬프트 기법에 비해 3.1%의 성능 향상과 물론 6배의 빠른 학습 시간을 자랑합니다.

- **Performance Highlights**: LoR-VP는 총 7개의 네트워크 아키텍처와 4개의 데이터셋을 통해 전통적인 시각 촉진 방법들에 비해 성능과 효율성 모두에서 눈에 띄는 개선을 보여줍니다. 해당 연구에서 제안된 프레임워크는 기존 SOTA(state-of-the-art) 방법인 AutoVP보다 평균 3.1% 향상된 성능을 달성하였고, 18배 적은 프롬프트 매개변수를 사용하면서도 6배 빠른 훈련 시간을 기록했습니다. 이러한 결과들은 LoR-VP의 실질적인 유용성을 입증합니다.



### STAF: Sinusoidal Trainable Activation Functions for Implicit Neural Representation (https://arxiv.org/abs/2502.00869)
- **What's New**: 최근 기술에서는 Sinusoidal Trainable Activation Functions (STAF)를 도입하여 Implicit Neural Representations (INRs)의 한계를 극복하고 있습니다. STAF는 네트워크가 고주파 성분을 더욱 효율적으로 학습하고 표현할 수 있도록 설계되었습니다. 이 기능은 신호 표현 및 역 문제에서 성능을 향상시켜, 기존 ReLU 기반의 모델들이 갖고 있던 spectral bias의 문제를 효과적으로 해결합니다.

- **Technical Details**: STAF는 자가 적응형 주파수 학습을 가능하게 하여, 복잡한 신호를 높은 정확도로 모델링할 수 있는 능력을 제공합니다. NTK(Neural Tangent Kernel)의 개념을 통해 학습 동역학을 이해하며, STAF를 사용하여 네트워크의 동작을 정량적으로 분석합니다. 연구에서는 STAF 활성화 함수의 경우 더 높은 고유값과 고유 함수를 생성하여, 고주파 성분의 학습과 재구성을 향상시킨다는 사실을 보여줍니다.

- **Performance Highlights**: STAF는 정확도 및 재구성 신뢰성 면에서 최신 기술(SOTA) 방법들을 초월하는 성과를 보였습니다. 특히, Peak Signal-to-Noise Ratio (PSNR) 지표가 우수한 성능을 지니며, 복잡한 신호에서 효과적인 결과를 도출합니다. 이러한 성능 향상은 STAF가 다양한 컴퓨터 그래픽스 및 관련 분야에서 유용하게 사용될 수 있음을 시사합니다.



### RealRAG: Retrieval-augmented Realistic Image Generation via Self-reflective Contrastive Learning (https://arxiv.org/abs/2502.00848)
- **What's New**: 최근의 텍스트-이미지 생성 모델들이 주목할 만한 발전을 보였습니다. 하지만 이러한 모델들은 고정된 파라미터로 훈련된 폐쇄형 데이터셋에 제한되어 있어, 새로운 세밀한 객체를 생성할 때 왜곡이나 환각이 발생하는 문제가 있습니다. 이를 해결하기 위해 우리는 'RealRAG'라는 새로운 프레임워크를 제안하며, 이는 실제 이미지를 학습하고 검색하여 생성 모델의 지식 공백을 메우는데 도움을 줍니다.

- **Technical Details**: RealRAG의 핵심은 자가 반영 대조 학습(self-reflective contrastive learning)을 통해 훈련된 reflective retriever를 사용하는 것입니다. 이 retriever는 누락된 지식을 다룰 수 있는 이미지를 검색하여, 생성 모델에 필요한 시각적 메모리를 보강합니다. 특히, 텍스트 프롬프트와 관련된 이미지를 검색할 수 있어, 정교하고 새로운 객체를 더 잘 생성할 수 있도록 지원합니다.

- **Performance Highlights**: RealRAG는 다양한 최신 텍스트-이미지 생성 모델에 적용될 수 있으며, 모든 모델에서 유의미한 성능 향상을 이끌어냅니다. 예를 들어 Stanford Cars 벤치마크에서 auto-regressive 모델에 대해 16.18%의 FID 점수 증가를 나타냈습니다. 이를 통해 RealRAG는 텍스트-이미지 생성의 현실감을 증대시키고 왜곡 문제를 개선하는 데 기여하고 있습니다.



### VLM-Assisted Continual learning for Visual Question Answering in Self-Driving (https://arxiv.org/abs/2502.00843)
- **What's New**: 본 논문에서는 자율 주행에서 Visual Question Answering (VQA) 작업을 해결하기 위한 새로운 접근 방식을 제안합니다. 이 방법은 Vision-Language Models (VLMs)와 지속 학습(Continual Learning)을 통합하여 모델이 다양한 운전 작업에 원활하게 적응할 수 있게 합니다. 특히, 선택적 메모리 재생(selective memory replay)과 지식 증류(knowledge distillation)를 결합함으로써 기존의 지식을 유지하고 새로운 지식을 효과적으로 학습할 수 있도록 만듭니다. 이 접근 방식은 자율 주행의 안전성과 신뢰성을 높이는 데 필수적입니다.

- **Technical Details**: 제안된 접근 방식은 두 단계에서 지속 학습을 수행합니다: 데이터 재생 단계와 현재 작업 데이터 훈련 단계입니다. 이 과정에서 단순한 자율 주행 VLM을 구축하며, 이미지 임베딩 네트워크와 미리 학습된 T5 언어 모델을 결합합니다. 추가로, 모델의 학습 과정은 특징 임베딩 사이의 차이를 기반으로 손실 값을 계산하여 정규화를 달성함으로써 더욱 효과적으로 진행됩니다. 이를 통해 모델은 다양한 운전 조건에서도 중요한 정보를 유지하게 됩니다.

- **Performance Highlights**: DriveLM 데이터셋에서 평가한 결과, 본 프레임워크는 성능이 상당히 향상되었으며, 여러 지표에서 최소 21.40%에서 최대 32.28%의 개선을 보였습니다. 이러한 결과는 VLM과 지속 학습의 결합이 자율 주행에서 VQA 시스템의 회복성과 신뢰성을 향상시키는 데 효과적임을 강조합니다. 연구진은 소스 코드를 공개할 예정이며, 이것이 자율 주행 시스템의 발전에 기여할 것으로 기대하고 있습니다.



### Cross multiscale vision transformer for deep fake detection (https://arxiv.org/abs/2502.00833)
- **What's New**: 딥페이크(Deep fake) 기술의 발전은 디지털 미디어의 신뢰성을 크게 위협하고 있습니다. 본 연구에서는 SP Cup 2025의 딥페이크 탐지 도전 과제 데이터셋을 활용하여 다양한 딥러닝(deep learning) 모델을 평가합니다. 전통적인 방식뿐만 아니라 최신 아키텍처를 사용하여 딥페이크 콘텐츠를 탐지하기 위한 접근 방식을 탐구하고 있으며, 이 연구는 향후 기술 발전의 귀중한 기초를 제공할 것으로 기대됩니다.

- **Technical Details**: 이 연구에서는 DeepfakeBench 벤치마크 데이터셋을 사용하였으며, 진짜 이미지와 조작된 이미지 사이의 불균형 문제를 해결하기 위해 언더샘플링 전략을 적용하여 두 클래스의 샘플 수를 44,000으로 동일하게 맞추었습니다. CMVit Repeat 모델은 멀티스케일 비전 트랜스포머(MViT)와 크로스 모델 퓨전(CMF) 블록을 통합하여 공간적 및 주파수 도메인(feature) 정보를 처리하는 독창적인 딥러닝 아키텍처를 구성합니다. CMVit+LBP 모델은 LBP(Local Binary Pattern) 기능을 추가하여 텍스처 정보 캡처 능력을 향상시켰습니다.

- **Performance Highlights**: 모델의 성능 평가는 정확도(accuracy) 측정 지표를 통해 이루어졌습니다. CMVit Repeat 및 CMVit+LBP 모델은 서로 다른 아키텍처의 장점을 결합하여 고도화된 특징 표현을 학습할 수 있도록 설계되었습니다. 결과적으로, 두 모델의 성능은 기존 탐지 기술에 대한 중요한 통찰을 제공하며, 텍스처 및 주파수 도메인 특성을 모두 활용함으로써 딥페이크 탐지의 성공률을 높이는 데 기여하고 있습니다.



### Environment-Driven Online LiDAR-Camera Extrinsic Calibration (https://arxiv.org/abs/2502.00801)
- **What's New**: 이 논문에서는 최초의 환경 기반 온라인 LiDAR-카메라 외부 보정 방법인 EdO-LCEC를 소개합니다. 이 방법은 인간의 지각 시스템에서 영감을 받아, 환경 조건을 해석하고 여러 가상의 카메라를 생성하여 공간적 및 질감 정보의 세밀한 캡쳐를 가능케합니다. EdO-LCEC는 매끄러운 성능을 위해 교차 모드 기능 일치 문제를 해결하기 위해 이중 경로 대응 일치(DPCM)를 사용합니다.

- **Technical Details**: EdO-LCEC는 센서의 작동 환경을 공간-시간적 흐름으로 처리하여, 여러 장면의 정보를 동적으로 결합하고 높은 정확도의 보정을 달성합니다. 일반화 가능한 장면 구별자는 대형 비전 모델을 사용하여 깊이 추정 및 이미지 분할을 수행하면서, 환경을 구성하는 다양한 특징을 포착합니다. 이 시스템은 각 장면에서 DPCM을 수행하여 구조적 및 질감 일관성을 기반으로 신뢰할 수 있는 3D-2D 대응을 생성합니다.

- **Performance Highlights**: 다양한 실제 데이터셋에 대한 실험을 통해 EdO-LCEC는 다른 최신 온라인, 무대상 접근 방식들과 비교하여 우수한 강건성과 정확성을 보여주었습니다. 이 시스템은 특수 조정된 환경이 아닌 다양한 도전적인 환경에서의 신뢰할 수 있는 보정을 제공합니다. 결론적으로, EdO-LCEC는 높은 보정 정확도와 인간과 같은 적응성을 달성하는 혁신적인 접근법입니다.



### Adversarial Semantic Augmentation for Training Generative Adversarial Networks under Limited Data (https://arxiv.org/abs/2502.00800)
Comments:
          This work was completed in 2022 and submitted to an IEEE journal for potential publication

- **What's New**: 이 연구에서는 기존의 데이터 증대(data augmentation) 방법의 한계를 극복하기 위해, 적대적 의미 증대(adversarial semantic augmentation, ASA) 기법을 제안합니다. ASA는 이미지 수준이 아닌 의미 수준에서 훈련 데이터를 확대하여 저데이터 환경에서도 GAN의 생성 품질을 향상시킵니다. 이 기술은 의미 특징에서 의미 있는 변환 방향을 찾아 기존 데이터의 변환을 통해 새로운 샘플을 생성합니다.

- **Technical Details**: 제안된 ASA 기법은 실제 및 생성 이미지의 의미 특징의 공분산 행렬을 추정하여 의미 변환 방향을 도출합니다. 이러한 방식을 통해, 모델은 기존의 이미지를 변환하여 다양하고 신뢰성 있는 샘플을 생성할 수 있습니다. 특히, ASA는 추가적인 계산 비용 없이 다양한 GAN 모델에 쉽게 통합될 수 있도록 설계되었습니다.

- **Performance Highlights**: 다양한 저샷(few-shot) 및 대규모 데이터셋에 대한 실험을 통해, ASA 기법이 생성 품질을 일관되게 개선함을 입증했습니다. 또한, 시각적 및 정량적 결과는 제안된 방법의 우수한 범용성을 보여줍니다. ASA는 의미 변환을 자동으로 수행하여, 훈련 샘플의 명확한 증대 없이도 성능을 극대화하는 데 기여합니다.



### Task-Specific Adaptation with Restricted Model Access (https://arxiv.org/abs/2502.00796)
- **What's New**: 본 논문에서는 Gray-box fine-tuning 접근법을 소개하여 기존의 여러 한계점을 해결합니다. 기존의 fine-tuning 방법들이 모델의 구조와 가중치에 대한 접근을 요구하는 반면, Gray-box 방식은 기울기 전파만을 허용하며, 모델의 구조는 숨기고 가중치는 고정합니다. 이 새로운 접근법은 학습 및 배포의 효율성을 높이고 사용자의 데이터 프라이버시 및 지적 재산권 보호도 가능하게 합니다.

- **Technical Details**: Gray-box fine-tuning에서는 LightGray-box와 DarkGray-box의 두 가지 변형을 제안합니다. LightGray-box는 여러 입력 포인트를 허용하여 더 많은 모델 정보를 드러내는 반면, DarkGray-box는 원본 입력 포인트만 허용하여 모델을 더욱 안전하게 보호합니다. 이러한 프레임워크는 입력 및 출력 공간만을 수정하여 새로운 도메인 특정 작업에 모델을 적응시킬 수 있도록 설계되었습니다.

- **Performance Highlights**: 테스트 결과, DarkGray-Box Input/Output Adapters(DGA) 접근법은 Text-to-Image 및 Text-to-Video Retrieval 벤치마크와 같은 검색 작업에서 경쟁력이 있는 성능을 발휘했습니다. 우리 연구는 이러한 Gray-box 방법들이 기존의 화이트 박스 기준치에 근접하는 성과를 달성하며, 다양한 작업과 아키텍처에 대한 실용성을 보여줍니다.



### Estimating forest carbon stocks from high-resolution remote sensing imagery by reducing domain shift with style transfer (https://arxiv.org/abs/2502.00784)
- **What's New**: 이 논문은 숲이 육상의 중요한 탄소 저수지로 작용하며, 탄소 저장량을 모니터링하는 새로운 방법론을 제안합니다. 기존의 방법은 공중 관측 데이터를 사용하는데, 본 연구에서는 지상 샘플 데이터를 위성 원격 감지 이미지와 통합하는 방식을 사용했습니다. 이를 통해 대규모 관찰이 가능해지지만, 정확도를 높일 필요가 있습니다.

- **Technical Details**: 연구에서는 GF-1 WFV 이미지와 Landsat TM 이미지를 사용하여 중국 윈난성 구징시의 화이즈 카운티를 분석했습니다. 스타일 전이 방식(style transfer method)을 사용하여 Swin Transformer를 도입하였고, 주의 메커니즘(attention mechanisms)을 통해 전역 특징을 추출하였습니다. 이러한 접근법은 탄소 저장량 추정을 이미지 변환(image translation)으로 변환하는 데 기여합니다.

- **Performance Highlights**: 본 연구의 접근 방식은 기존 방법보다 향상된 정확도를 제공할 가능성을 보여줍니다. 특히, Swin Transformer를 통한 특징 추출 방식은 큰 데이터 세트에서의 성능을 개선하는 데 효과적일 것으로 기대됩니다. 이러한 혁신적인 기술은 기후 변화 완화 및 탄소 저장량의 효과적인 평가에 기여할 수 있습니다.



### A method for estimating forest carbon storage distribution density via artificial intelligence generated content mod (https://arxiv.org/abs/2502.00783)
- **What's New**: 이번 연구는 Yunnan(윈난) 성 Qujing(취징) 시 Huize(휘저) 카운티를 연구 지역으로 설정하고, 원거리 탐지(Remote Sensing) 기법을 통해 숲의 탄소 저장량을 추정합니다. 특히, GF-1 WFV 위성 이미지를 데이터로 활용하고, 초기 특징 추출을 위해 KD-VGG 모듈을 도입하여 향상된 내재 확산 모델(IIDM)을 제안했습니다.

- **Technical Details**: VGG-19 모듈은 지식 증류(Knowledge Distillation) 이후 초기 특징 추출을 통해 모델 파라미터 수를 줄이는 한편 추론 시간을 단축시키고 정확도를 향상시켰습니다. 또한, Attention + MLP 모듈을 추가하여 전역(global) 및 로컬(local) 특징 간의 관계를 파악하고 연속 스케일 범위 내에서 고충실도(high-fidelity) 이미지를 복원하는 데 성공했습니다.

- **Performance Highlights**: 제안된 IIDM 모델은 RMSE가 28.68로 다른 회귀(regression) 모델보다 13.16이 높은 약 31.45% 향상된 추정 정확도를 기록했습니다. 이 연구는 인공지능 생성 콘텐츠(AIGC)가 정량적 원거리 탐지 분야에서의 가능성을 입증하며, 탄소 중립화 효과 연구에 중요한 통찰력을 제공합니다.



### Spatio-Temporal Progressive Attention Model for EEG Classification in Rapid Serial Visual Presentation Task (https://arxiv.org/abs/2502.00730)
- **What's New**: 이 연구에서는 EEG 신호 분류 성능을 향상시키기 위한 새로운 공간-시간 진전 주의 모델(STPAM)을 제안합니다. STPAM은 세 가지 공간 전문가를 사용하여 뇌 영역의 공간 정보를 점진적으로 학습하고, EEG 전극의 선택을 통해 불필요한 간섭을 최소화하도록 설계되었습니다. 이 모델은 또한 저조도 적외선 이미지를 기반으로 구축된 새로운 RSVP EEG 데이터셋(IRED)을 소개하여, 다양한 자극을 탐색할 수 있는 기회를 제공합니다.

- **Technical Details**: STPAM은 두 가지 주요 모듈로 구성되며, 첫 번째는 Progressive Spatial Learning(PSL) 모듈로, 이 모듈은 관련 전극에서 EEG 특징을 추출하고 중요 전극을 선택하여 다음 전문가에게 전달합니다. 이후, Progressive Temporal Learning(PTL) 모듈이 중요 EEG 슬라이스를 점진적으로 선택하여 최종 분류를 위한 공간-시간 EEG 특징을 생성합니다. 이 과정에서 전문가들이 서로 다른 전극 및 EEG 슬라이스를 선택하도록 유도하기 위해 분산 제약을 도입합니다.

- **Performance Highlights**: 실험 결과에 따르면 STPAM은 기존 방법들과 비교했을 때 EEG 분류 작업에서 우수한 성능을 보였습니다. 특히, 기존의 RSVP EEG 데이터셋과 비교할 때, IRED 데이터셋에서의 성능 향상이 뚜렷하게 나타났습니다. 이는 EEG 신호의 시공간 정보의 통합이 신호 분류 성능에 긍정적인 영향을 주었음을 보여줍니다.



### Vision and Language Reference Prompt into SAM for Few-shot Segmentation (https://arxiv.org/abs/2502.00719)
Comments:
          8 pages, 2 figures

- **What's New**: 이번 논문에서는 Visual and Language reference Prompt into SAM (VLP-SAM)이라는 새로운 few-shot segmentation 모델을 제안합니다. VLP-SAM은 이미지와 텍스트 레이블을 함께 이용하여 SAM에 대한 프롬프트를 생성함으로써, 보다 정확한 세분화를 가능하게 합니다. 이를 통해 사용자가 제공한 프롬프트 없이도 특정 객체를 타겟 이미지에서 세분화할 수 있습니다.

- **Technical Details**: VLP-SAM은 비쥬얼-언어 모델(VLM)을 사용하여 참조 이미지의 비쥬얼 정보와 텍스트 레이블의 의미론적 정보를 통합합니다. 최소의 학습 가능한 파라미터로 이루어진 새로운 SAM 프롬프트 인코더를 도입하여, VLP-SAM은 다양한 모달리티를 입력으로 받는 확장 가능한 모델로 만들어졌습니다. 이 프로세스에서 VLP-SAM은 이미지와 텍스트를 동일한 임베딩 공간에서 처리하여 타겟 객체의 특성을 세밀하게 반영합니다.

- **Performance Highlights**: VLP-SAM은 PASCAL-5i와 COCO-20i 데이터셋에서의 실험을 통해 기존의 최첨단 모델들보다 mIoU에서 각각 6.3% 및 9.5% 향상된 성과를 보여주었습니다. 또한, VLP-SAM은 훈련 데이터에 포함되지 않은 새로운 객체에 대해서도 높은 일반화를 보이고 있어, 실용적인 응용 가능성이 큽니다.



### MINT: Mitigating Hallucinations in Large Vision-Language Models via Token Reduction (https://arxiv.org/abs/2502.00717)
Comments:
          8 pages, 5 figures, 4 tables

- **What's New**: 최근 LVLM(대형 비전-언어 모델)에서 출현하는 환영(hallucination) 문제를 해결하기 위한 새로운 접근법인 MINT가 제안되었습니다. 이 방법은 LLM(대형 언어 모델)의 고유한 문제에 대한 기존의 해결 방안과는 다른, 훈련 없이 사용할 수 있는 디코딩 전략입니다. MINT는 이미지 토큰을 감소시켜 환영 문제를 완화하고, 키 이미지 지역에 집중하도록 합니다.

- **Technical Details**: MINT는 LVLM의 주의(attention) 메커니즘을 활용하여 이미지에서 불필요한 토큰을 마스킹하고, 이로 인해 지역적 인식(local perception) 능력을 향상시킵니다. 또한, 대조적 디코딩(contrastive decoding)을 적용하여 모델이 핵심 이미지 영역에 더욱 집중하도록 유도합니다. 이 방식은 LVLM의 디코딩 과정에서 중요 이미지 토큰을 동적으로 선택하여 적용하는 것을 포함합니다.

- **Performance Highlights**: 다양한 공개 기준에서 실험한 결과, MINT는 기존 모델에 비해 환영 문제를 4% 감소시키며, 동시에 5% 더 많은 시각적 포인트를 인식할 수 있는 능력을 보여주었습니다. 이러한 결과는 MINT가 LVLM의 신뢰성을 높이는 데 기여할 수 있음을 나타냅니다.



### VIKSER: Visual Knowledge-Driven Self-Reinforcing Reasoning Framework (https://arxiv.org/abs/2502.00711)
Comments:
          17 pages,12 figures

- **What's New**: 이 논문에서는 VIKSER(Visual Knowledge-Driven Self-Reinforcing Reasoning Framework)를 제안하여 비주얼 정보에 대한 질문 해결을 위한 새로운 접근 방식을 소개합니다. 기존의 비주얼 추론(visual reasoning) 기법은 해석 가능성이 제한적이고, 질문 텍스트의 불완전성에 의해 제약을 받는 문제를 안고 있습니다. VIKSER는 대형 언어 모델(LLMs)에서 증류한 지식을 활용해 정밀한 비주얼 지식을 추출하고, 이를 통해 질문을 패러프레이즈(paraphrase)합니다.

- **Technical Details**: VIKSER의 핵심 구성 요소는 세분화된 비주얼 지식 추출(F-VKE) 모듈과 자기 강화 추론(S-RR) 모듈로 구성됩니다. S-RR 모듈은 Chain-of-Evidence(CoE)라는 새로운 프롬프트 기법을 통합하여 해석 가능성을 높이고, 과거의 실수를 통해 학습하는 자기 반성 메커니즘을 도입합니다. F-VKE 모듈은 입력 이미지의 주요 엔티티 간의 시각적 관계를 감지하고, 이를 기반으로 원인 관계를 분석하여 세분화된 비주얼 지식을 생성합니다.

- **Performance Highlights**: VIKSER는 다양한 공개 데이터셋에서 철저한 실험을 통해 기존 연구들을 초월하는 성과를 거두었으며, 모든 데이터셋에서 새로운 최신(state-of-the-art, SOTA) 결과를 달성하였습니다. 이러한 성과는 고도의 해석 가능성과 자기 강화 학습이 결합된 결과로, 비주얼 추론 작업에서의 효율성과 정확성을 크게 향상시킵니다. 이를 통해 VIKSER는 비주얼 지식 추출 및 추론 능력에서 뛰어난 성능을 입증하였습니다.



### PhiP-G: Physics-Guided Text-to-3D Compositional Scene Generation (https://arxiv.org/abs/2502.00708)
Comments:
          13 pages.8 figures

- **What's New**: 이번 논문에서는 PhiP-G라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 자연어 입력으로부터 고품질의 3D 장면을 생성하는 데 필수적인 구성 요소인 생성 모델과 LLM 기반 에이전트를 통합합니다. PhiP-G는 물리 법칙에 부합하는 고품질 3D 장면 생성을 위한 예측 및 계획 기능을 활용하여 레이아웃 단계를 향상시킵니다.

- **Technical Details**: PhiP-G는 복합적인 장면 설명을 분석하기 위해 LLM 기반 에이전트를 사용하고, 장면 그래프를 생성하여 수동 레이아웃을 피합니다. 2D 이미지 생성 에이전트인 DALL·E 3와 3D Gaussian splatting(3DGS) 모델을 결합하여 고품질 자산을 유연하게 생성합니다. Blender를 레이아웃 디자인의 기본 플랫폼으로 활용하여 물리 풀 및 관계 일치 에이전트를 도입하여 전체적인 레이아웃 안내를 실현합니다.

- **Performance Highlights**: PhiP-G는 복합 장면 생성에서 상태 최상위(State-of-the-Art) 성능을 달성하며, CLIP 점수에서 SOTA를 기록했습니다. 또한, T3Bench 메트릭에서 최고의 성능을 달성하면서 생성 효율성을 24배 향상시켰습니다. 이로 인해 복잡한 자연어 입력에 대한 의미적 일관성을 유지하며 물리 법칙을 준수하는 신뢰할 수 있는 3D 장면 생성을 가능하게 합니다.



### S2CFormer: Reorienting Learned Image Compression from Spatial Interaction to Channel Aggregation (https://arxiv.org/abs/2502.00700)
- **What's New**: 이 논문은 학습된 이미지 압축(LIC)에서 변환기(Transformer)의 역할을 재평가합니다. 상대적으로 간과되었던 피드포워드 네트워크(Feed-Forward Network, FFN) 기반의 채널 집합(Channel Aggregation) 모듈이 LIC 모델의 효율성에 중요한 기여를 한다는 점을 강조합니다. 공간적(interaction) 작업을 단순화하고 채널 집합에 중점을 둔 새로운 S2CFormer 구조를 제안하여 효율성과 성능을 모두 개선하였습니다.

- **Technical Details**: S2CFormer는 공간적 상호작용(Spatial Interaction)과 채널 집합(Channel Aggregation)이라는 두 가지 주요 요소로 구성됩니다. 이 구조는 공간적 작업을 단순화하여 디코딩 속도를 향상시키고, 채널 차원에서 비등방성 특징을 집합하여 유의미한 성능을 달성하도록 설계되었습니다. 새로운 LIC 모델인 S2C-Hybrid는 다양한 S2CFormer 인스턴스의 장점을 결합하여 R-D 성능을 더욱 향상시킵니다.

- **Performance Highlights**: S2C-Conv 및 S2C-Attention는 공간 상호작용과 채널 집합을 결합하여 최신의 R-D 성능을 도출하며, 디코딩 속도를 30% 이상 개선하는 결과를 보여줍니다. S2C-Hybrid 모델은 Kodak, Tecnick 및 CLIC 데이터셋에서 기존의 모든 방법을 능가하여 새로운 성능 기준을 세웠습니다. 이러한 성과는 LIC 모델에서 공간 상호작용보다 채널 집합이 더 중요하다는 점을 확고히 합니다.



### TMI-CLNet: Triple-Modal Interaction Network for Chronic Liver Disease Prognosis From Imaging, Clinical, and Radiomic Data Fusion (https://arxiv.org/abs/2502.00695)
Comments:
          6 pages, 3 figures, accepted by IEEE ISBI 2025

- **What's New**: 이번 연구에서는 만성 간 질환의 예후 평가를 위해 Triple-Modal Interaction Chronic Liver Network (TMI-CLNet)를 제안합니다. 이 새로운 접근법은 CT 이미지, 방사형 특징(radiomic features), 임상 정보를 통합하여 보다 종합적인 예후 정보를 제공합니다. 또한, Intra-Modality Aggregation 모듈과 Triple-Modal Cross-Attention Fusion 모듈을 통해 서로 다른 데이터 모달리티 간의 관계를 효과적으로 포착할 수 있도록 설계하였습니다.

- **Technical Details**: TMI-CLNet의 아키텍처는 세 가지 주요 구성 요소로 이루어져 있습니다: 특징 추출 모듈, 다중 모달 상호 작용 모듈, 그리고 분류 헤드 모듈입니다. 특징 추출 모듈에서는 3D ResNet-50을 사용하여 시각적 특징을 추출하며, 방사형 데이터는 다층 퍼셉트론(MLP)으로 처리해 보다 추상적인 특징 표현을 얻습니다. 각 모달리티의 특징을 통합하기 위해 Intra-Modal Aggregation (IMA) 모듈이 사용되며, 이 모듈은 다중 헤드 자기 주의 기법을 통해 특징 정보를 통합합니다.

- **Performance Highlights**: 광범위한 실험을 통해 TMI-CLNet이 기존의 최첨단 단일 모달 모델 및 기타 다중 모달 기술들보다 유의미하게 성능이 우수함을 보여주었습니다. 본 연구의 결과는 만성 간 질환 치료의 임상적 결정 과정에 중대한 기여를 할 것으로 기대됩니다. 또한 연구의 코드는 공개되어 있어 추후 연구자들이 쉽게 접근할 수 있도록 합니다.



### High-Order Matching for One-Step Shortcut Diffusion Models (https://arxiv.org/abs/2502.00688)
- **What's New**: 이 논문에서는 고차 매칭(HOMO, High-Order Matching for One-Step Shortcut Diffusion)을 도입하여 기존의 Shortcut 모델의 한계를 극복하고자 합니다. HOMO는 고차 감독(high-order supervision)을 통해 데이터 전송(distribution transportation)을 혁신적으로 개선합니다. 이 프레임워크는 가속(acceleration), 급상승(jerk) 등 다양한 요소를 포함하여 지오메트리적 정확성(geometric precision)과 안정성(stability)을 향상시킵니다.

- **Technical Details**: HOMO는 고차 감독을 통합하여 기존의 1차 동역학(first-order dynamics)에 의존하는 Shortcut 모델의 한계를 넘습니다. 이 모델은 예측 경로의 정확성과 안정성을 보장하기 위해 적절한 수학적 기초를 확립하고, 복잡한 데이터 분포를 정확히 모델링할 수 있는 역량을 갖추고 있습니다. 특히, 고차 동역학을 포함시키는 것이 데이터의 흐름을 더 매끄럽고 정확하게 만들어 줍니다.

- **Performance Highlights**: HOMO는 상당히 복잡한 환경, 특히 고곡률(high-curvature) 지역에서 Shortcut 모델을 뛰어넘는 성능을 보여줍니다. 실험 결과 HOMO는 더 매끄러운 경로(smoother trajectories)와 유리한 분포 정렬(better distributional alignment)을 달성하여 1단계 생성 모델(1-step generative models)에 새로운 기준을 제시합니다. 이러한 성과로 인해 HOMO는 생성 모델링에서의 신뢰성과 강력함을 증명하였습니다.



### Cross-Modal Synergies: Unveiling the Potential of Motion-Aware Fusion Networks in Handling Dynamic and Static ReID Scenarios (https://arxiv.org/abs/2502.00665)
- **What's New**: 이 논문에서는 다양한 감시 시나리오, 특히 occlusion(가림 현상) 상황에서의 사람 재식별(person re-identification, ReID)의 복잡성을 해결하기 위한 새로운 Motion-Aware Fusion (MOTAR-FUSE) 네트워크를 소개합니다. MOTAR-FUSE는 정적인 이미지를 통해 유도된 motion cues(움직임 단서)를 활용하여 ReID 성능을 크게 향상시킵니다. 이 네트워크는 이미지와 비디오 모두를 처리할 수 있는 이중 입력 비주얼 어댑터를 포함하여 더 효과적인 feature extraction(특징 추출)을 가능하게 합니다.

- **Technical Details**: MOTAR-FUSE 시스템은 정적 이미지에서 유도된 공간적 및 시간적 데이터를 활용함으로써 사람 재식별의 정확성을 크게 향상시키도록 설계되었습니다. 이 시스템 아키텍처는 비주얼 인코더, 비주얼 어댑터, 모션 인식 모듈 및 융합 인코더와 같은 주요 구성 요소로 구성됩니다. 특히, motion consistency task(모션 일관성 작업)가 통합되어 모션 인식 트랜스포머가 인간의 움직임의 역학을 효과적으로 포착할 수 있도록 지원합니다.

- **Performance Highlights**: 다양한 ReID 벤치마크에서 실시된 포괄적인 평가에 따르면, MOTAR-FUSE 네트워크는 기존 접근 방식보다 우수한 성능을 달성합니다. 이 기술은 특히 occlusion이 만연한 시나리오에서 기능 인식 능력을 개선하여 ReID 프로세스를 진화시킵니다. MOTAR-FUSE의 실험 결과는 복잡한 가림 현상과 변동성이 있는 조건에서 도시 보안 시스템의 운영 능력을 재정의할 수 있는 잠재력을 확인시켜 줍니다.



### Enhanced Convolutional Neural Networks for Improved Image Classification (https://arxiv.org/abs/2502.00663)
- **What's New**: 이 논문에서는 CIFAR-10 이미지 분류를 위한 향상된 CNN 아키텍처를 제안합니다. 기존 CNN 모델의 과적합(overfitting) 문제와 기능 표현 문제를 해결하기 위해 더 많은 convolutional 블록, 배치 정규화(batch normalization), 그리고 드롭아웃(Need dropout regularization) 기법을 통합하였습니다. 제안하는 모델은 84.95%의 테스트 정확도를 달성하여 기존 CNN 아키텍처를 초월하는 성능을 보였습니다.

- **Technical Details**: 데이터 전처리는 모델의 일반화(generalization)와 안정성을 높이는 데 중요한 역할을 합니다. CIFAR-10 데이터셋에 대해, 픽셀 값의 정규화(normalization)를 통해 입력 데이터를 [-1, 1] 범위로 조정하고, 데이터 증대(data augmentation)를 통해 학습 데이터의 다양성을 인위적으로 증가시켰습니다. 또한, 64 크기의 미니 배치(mini-batch)를 이용해 효율적인 그래디언트 계산을 가능하게 하였습니다.

- **Performance Highlights**: 실험 결과, 향상된 CNN 아키텍처는 기존의 CNN 기반 모델에 비해 월등한 성능을 보여주었습니다. 특히, 배치 정규화와 드롭아웃을 통합함으로써 훈련 과정의 안정성을 극대화하고, 제안하는 모델이 보다 복잡한 이미지 분류 작업에 효과적이라는 점을 강조했습니다. 모델의 구조적인 개선이 작은 규모의 이미지 분류 문제를 해결하는 데 어떠한 잠재력을 가지고 있는지를 보여주는 결과입니다.



### Mitigating the Modality Gap: Few-Shot Out-of-Distribution Detection with Multi-modal Prototypes and Image Bias Estimation (https://arxiv.org/abs/2502.00662)
- **What's New**: 본 논문은 출처가 다른(out-of-distribution, OOD) 샘플 탐지를 위한 비전-언어 모델(vision-language model, VLM) 기반 접근 방식을 개선하는 새로운 방법을 제안합니다. 기존의 방법들은 텍스트 프로토타입과 이미지 프로토타입 간의 모달리티 갭(modality gap)으로 인해 높은 오탐지(false positive)율을 보였습니다. 이를 해결하기 위해, 저자들은 ID 이미지 프로토타입을 포함하는 방법을 도입하고, 이로 인해 OOD 탐지 성능이 개선된다고 주장합니다.

- **Technical Details**: 제안된 방법인 SUPREME은 편향된 프롬프트 생성(biased prompts generation, BPG)과 이미지-텍스트 일관성(image-text consistency, ITC) 모듈로 이루어져 있습니다. BPG는 이미지-텍스트 융합을 강화하고 일반화를 개선하기 위해 가우시안 기반의 이미지 도메인 바이어스를 조건으로 설정합니다. ITC는 모달리티 갭을 최소화하기 위해 intra-modal 및 inter-modal 거리 최솟값을 계산합니다. 이러한 방식을 통해 새로운 OOD 점수인 SGMP를 도입합니다.

- **Performance Highlights**: SUPREME은 기존 VLM 기반 OOD 탐지 방법들에 비해 일관되게 성능을 개선하였습니다. 실험 결과, SUPREME의 사용으로 Imagenet-100 및 네 가지 OOD 데이터셋에서 평균적인 FPR95와 AUROC가 각각 32.4와 94.5에서 24.2와 95.8로 개선되었습니다. 이러한 성과는 이론적 분석과 실증적 증거를 통해 뒷받침되며, 제안된 방법이 신뢰성을 높이는 데 기여할 수 있음을 보여줍니다.



### EmoTalkingGaussian: Continuous Emotion-conditioned Talking Head Synthesis (https://arxiv.org/abs/2502.00654)
Comments:
          22 pages

- **What's New**: EmoTalkingGaussian 모델이 제안되어, 사용자가 제공한 연속적인 감정 값(valence & arousal)을 기반으로 하지 않고도 에모셔널한 화법 합성을 가능하게 합니다. 특히, 기존의 3D Gaussian splatting 기반 방법들의 감정 표현의 한계를 극복하고, 다양한 감정을 실시간으로 반영할 수 있습니다. 또한, 입술 움직임과 오디오의 싱크를 효과적으로 개선하기 위해 자기 지도 학습(self-supervised learning) 방식을 도입했습니다.

- **Technical Details**: EmoTalkingGaussian은 3D Gaussian splatting 기술을 활용하여, valence 및 arousal 값에 기반한 연속적인 감정 표현을 지원합니다. 모델은 lip-aligned emotional face generator를 통해 많은 감정적인 얼굴 이미지를 효과적으로 훈련하며, 텍스트-투-스피치(text-to-speech) 네트워크와 시각-청각 동기화 네트워크를 사용하여 리얼한 오디오와의 입술 동기화를 강화합니다.

- **Performance Highlights**: EmoTalkingGaussian 모델은 공공 비디오를 기반으로 실험을 진행하였으며, 시각 품질은 PSNR, SSIM, LPIPS로, 감정 표현은 V-RMSE, A-RMSE, Emotion Accuracy로 측정하여 기존의 최신 기술보다 더욱 우수한 결과를 보였습니다. 또한, 입술 동기화와 관련하여 LMD, Sync-E 및 Sync-C 지표에서도 자가 개선이 이루어졌습니다.



### Zeroth-order Informed Fine-Tuning for Diffusion Model: A Recursive Likelihood Ratio Optimizer (https://arxiv.org/abs/2502.00639)
- **What's New**: 본 논문에서는 Recursive Likelihood Ratio (RLR) 최적화 기법을 제안하여 확률적 확산 모델(DM)의 미세 조정을 향상시킵니다. RLR은 zeroth-order gradient estimator를 활용하여 다양한 확산 단계에서 기울기를 비편향적으로 평가할 수 있는 새로운 접근 방식을 제공합니다. 이 방법은 기존의 강화 학습(Reinforcement Learning, RL) 및 잘린 역전파(Backpropagation, BP) 방법의 한계를 극복하여 전체 프로세스에서 높은 효율성을 보장합니다.

- **Technical Details**: RLR optimizer는 DM의 재귀적 구조와 perturbation 기반 기울기 추정 간의 관계를 분석하여 개발되었습니다. 이 최적화 기법은 기존의 BP와 RL의 장점을 결합하면서도 단점을 완화합니다. 예를 들어, RLR은 BP를 통해 선택된 부분에서만 메모리를 제한하고, 나머지 단계에서는 perturbation 기반 방법으로 기울기를 추정함으로써 구조적 편향과 높은 분산을 해결합니다.

- **Performance Highlights**: RLR은 Text2Image 및 Text2Video 작업의 광범위한 평가를 통해 기존의 모든 기법들을 큰 폭으로 초월하는 성능을 보였습니다. 또한, RLR을 위한 새로운 프롬프트 기법을 제안하여 이의 적용 가능성을 크게 높였습니다. 이러한 결과는 RLR의 효과성을 뒷받침하며, DM의 실용적인 활용을 촉진할 것입니다.



### MedConv: Convolutions Beat Transformers on Long-Tailed Bone Density Prediction (https://arxiv.org/abs/2502.00631)
- **What's New**: 본 논문에서는 CT 스캔을 이용한 뼈 밀도 예측을 위한 새로운 모델인 MedConv를 제안합니다. 기존의 transformer 모델보다 뛰어난 성능을 보이면서도 낮은 계산 복잡성을 유지하는 것이 특징입니다. 이 모델은 hospital 데이터에 존재하는 불균형 문제를 해결하기 위해 Bal-CE loss와 post-hoc logit 조절 기법을 적용하였습니다.

- **Technical Details**: MedConv는 spinal CT 스캔을 기반으로 한 뼈 밀도 예측을 위해 convolutional 접근법을 재구성하였습니다. 이 모델은 기존의 transformer 기반 모델들에 비해 계산 복잡성이 낮아 portable 및 clinical setting에서의 활용 가능성을 높였습니다. 또한, class imbalance 문제를 해결하기 위해 Bal-CE loss와 post-hoc logit 조정을 특화하여 Accuracy와 ROC AUC의 향상을 꾀하였습니다.

- **Performance Highlights**: AustinSpine 데이터셋에서 실시한 다양한 실험을 통해 MedConv는 기존 최첨단 방법들보다 최대 21%의 Accuracy 개선과 20%의 ROC AUC 개선을 달성했습니다. 이러한 결과는 CT 스캔을 기반으로 한 뼈 밀도 예측의 가능성을 입증하며, 향후 뼈 건강 관리에 있어 효과적인 도구가 될 것으로 기대됩니다.



### Self-Prompt SAM: Medical Image Segmentation via Automatic Prompt SAM Adaptation (https://arxiv.org/abs/2502.00630)
- **What's New**: 이 논문은 Self-Prompt-SAM이라는 새로운 의료 이미지 분할 프레임워크를 제안합니다. 기존의 SAM(Segment Anything Model) 과는 달리, 추가 프롬프트 없이도 의료 이미지 분할을 가능하게 합니다. 본 연구에서는 multi-scale hierarchical prompt generator(MSPGenerator)를 통해 보조 마스크를 생성하고, 적절한 포인트와 바운딩 박스를 제공함으로써 성능을 향상시킵니다.

- **Technical Details**: Self-Prompt-SAM은 수정된 이미지 인코더, MSPGenerator, 프롬프트 인코더 및 수정된 마스크 디코더로 구성됩니다. 3D 정보를 추출할 수 있도록 DFusedAdapter라는 깊이 융합 어댑터를 디자인하였으며, 이는 깊이 차원에서 추가 정보를 학습할 수 있게 합니다. 또한, MAdapter를 통해 다양한 의료 이미지의 모달리티를 RGB 채널로 변환하여 인코딩하는 방법을 제안합니다.

- **Performance Highlights**: 이 연구는 AMOS2022, ACDC, Synapse 데이터셋에서 extensive experiments를 수행하여 Self-Prompt-SAM이 state-of-the-art 성능을 달성했음을 보여줍니다. 특히, nnUNet 대비 AMOS2022에서 2.3%, ACDC에서는 1.6%, Synapse에서는 0.5%의 성능 향상을 기록하였습니다. 이러한 결과는 제안된 접근 방식의 효과적인 점과 잠재력을 입증합니다.



### DesCLIP: Robust Continual Adaptation via General Attribute Descriptions for Pretrained Vision-Language Models (https://arxiv.org/abs/2502.00618)
- **What's New**: 이 논문에서는 이미지-언어 모델(Vision-Language Models, VLMs)의 지속적 적응에 관한 연구를 진행했습니다. 특히, 기존의 접근 방식이 시각적 특성과 특정 클래스 텍스트를 연결하는 데 집중하여 일반적 지식과 전문 지식 간의 잠재적 관계를 간과한 점에 주목했습니다. 이를 해결하기 위해, 일반 속성(description of general attributes) 설명을 활용하여 시각-GA-클래스(trilateral vision-GA-class) 연관성을 구축하는 'DesCLIP' 방법론을 제안합니다.

- **Technical Details**: DesCLIP은 언어 보조 도구를 사용하여 특정 클래스 객체에 대한 일반 속성 설명 후보를 생성합니다. 이를 통해 얻은 GA(description of general attributes) 설명 임베딩을 시각적-텍스트적 인스턴스 매칭을 위한 대응 텍스트 임베딩으로 사용하고, 시각 인코더를 튜닝합니다. 또한, 클래스 텍스트 임베딩은 공유된 GA 설명 임베딩에 맞춰 점진적으로 조정됩니다.

- **Performance Highlights**: CIFAR100, ImageNet, CUB-200 데이터셋을 포함한 전반적인 실험에서, 제안된 방법은 기존의 지속적 학습 방법에 비해 탁월한 성능을 입증했습니다. 이 연구는 시각-클래스 텍스트 연결 대신에 시각-GA-클래스 연관성을 형성함으로써 지식 소멸을 효과적으로 완화하는 데 기여하고 있습니다. 종합적인 연구 평가를 통해 효과성을 추가적으로 뒷받침합니다.



### Fast Vision Mamba: Pooling Spatial Dimensions for Accelerated Processing (https://arxiv.org/abs/2502.00594)
Comments:
          20 pages, 15 figures, this https URL

- **What's New**: 최근의 State Space Models (SSMs)을 활용한 컴퓨터 비전 모델들이 효율성을 극대화하고 있습니다. 특히, Mamba는 전통적인 Vision Transformers 대비 선형 복잡성(linear complexity)으로 토큰 상호작용을 처리할 수 있는 방법을 제시합니다. 이러한 빠른 처리 속도를 위해 Fast Vision Mamba (FastVim)라는 새로운 모델을 제안하며, 이는 기존 모델의 성능을 유지하면서 계산 시간을 더욱 단축시킵니다.

- **Technical Details**: FastVim은 평균 풀링(average pooling)을 통해 비전 Mamba 모델에서 재귀(step)를 줄이는 기법을 사용합니다. 구체적으로, 각 Mamba 블록 내에서 이미지 차원에 따라 토큰을 번갈아 풀링하여 재귀적 계산 수를 2배 줄이는 결과를 가져옵니다. 이렇게 얻어진 1D 토큰 그리드를 통해 선택적 스캔을 적용하여 신속한 스캔이 가능하게 합니다.

- **Performance Highlights**: FastVim은 높은 해상도 이미지(2048×2048)에서 기존 Vision Mamba 모델 대비 최대 72.5%의 속도 향상을 보여줍니다. 다양한 비전 태스크에 대해 실험 결과, 이미지를 분류하고, 세포 교란 예측, 분할(segmentation), 객체 감지(object detection) 등의 작업에서 최첨단 성능을 입증했습니다. 특히, FastMaskVim과 FastChannelVim으로 확장하여 비정형 그리드 및 다중 채널 이미징에 적용할 수 있는 가능성을 보였습니다.



### Contrastive Forward-Forward: A Training Algorithm of Vision Transformer (https://arxiv.org/abs/2502.00571)
Comments:
          22 pages, 8 figures, under review

- **What's New**: 이 논문에서는 Forward-Forward (FF) 알고리즘의 개선안을 제시하며, Contrastive Forward-Forward 알고리즘을 도입합니다. 이 알고리즘은 이미지 분류 과제 해결을 위해 Vision Transformer (ViT)와 함께 사용됩니다. 기존의 FF는 뇌의 작동 방식에 더 유사하다고 주장되지만, 여전히 기존의 역전파(backpropagation) 알고리즘에 비해 성능 상승 폭이 적었습니다. Contrastive Forward-Forward는 이러한 성능 격차를 줄이며, 특히 데이터의 정확한 레이블이 주어지지 않은 상황에서도 유용한 이점을 제공합니다.

- **Technical Details**: 제안된 알고리즘은 각 층 이후에 손실 함수(loss function)를 배치하고, 두 개의 로컬 포워드 전파(forward pass)와 하나의 로컬 백워드 전파(backward pass)를 사용하여 층을 업데이트합니다. 또한, Contrastive Learning(대비 학습)에서 영감을 받아 손실 함수의 수정과 같은 변화가 시도되었습니다. 논문에서 제안된 방법은 가벼운 수정을 통해 ViT와 같은 복잡한 네트워크에서 적용 가능하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 Contrastive Forward-Forward 방법이 기존의 FF 알고리즘에 비해 최대 10% 향상된 정확도와 5배에서 20배의 수렴(convergence) 속도를 보여주었습니다. 이 연구는 또한 Cross Entropy(Cross Entropy)를 기준 손실 함수로 사용할 때 Forward-Forward의 모드가 역전파에 비해 성능 격차를 줄이거나 특정 조건에서는 오히려 더 우수한 결과를 나타내는 것도 입증하고 있습니다.



### Generating crossmodal gene expression from cancer histopathology improves multimodal AI predictions (https://arxiv.org/abs/2502.00568)
- **What's New**: 이 연구는 PathoGen이라는 새롭고 혁신적인 diffusion 기반 crossmodal generative AI 모델을 소개하고 있습니다. 이 모델은 디지털 병리학 이미지에서 합성된 transcriptomic 데이터를 활용하여 암의 등급 및 생존 위험을 높은 정확도로 예측합니다. 기존의 transcriptomic 테스트가 제한적이었던 실제 클리닉 환경에서 이러한 접근 방식은 비용 효율적인 스크리닝 도구로서의 가능성을 제시합니다.

- **Technical Details**: PathoGen 모델은 The Cancer Genomic Atlas (TCGA)에서 제공하는 데이터 세트를 기반으로 합니다. 이 모델은 H&E 염색 슬라이드의 디지털 이미지를 사용하여 transcriptomic 데이터를 합성하고, 이를 통해 학습한 특성 맵과 함께 암 등급 및 생존 위험 예측에 사용합니다. 예측 성능 향상을 위해 co-attention 메커니즘을 사용하고, 병리학자가 AI 결정에 기여하는 중요 영역을 시각적으로 확인하도록 돕는 attention 기반 heatmap을 제공합니다.

- **Performance Highlights**: 모델은 TCGA의 두 개의 암 코호트에서 테스트되었으며, 생성된 transcriptomic 데이터는 실제 데이터와 높은 유사성을 보였습니다. PathoGen을 통해 합성된 transcriptomic 데이터와 디지털 병리학 이미지로부터 학습된 특징을 결합함으로써 진단 및 예후 예측의 성능이 크게 향상되었습니다. 또한, 모델은 불확실성 정량화를 통해 각 환자에 대한 합성된 transcriptomic 데이터의 가치를 확인할 수 있는 방법을 제공합니다.



### Complex Wavelet Mutual Information Loss: A Multi-Scale Loss Function for Semantic Segmentation (https://arxiv.org/abs/2502.00563)
Comments:
          11 pages, 6 figures

- **What's New**: 최근 심층 신경망(deep neural networks)의 발전은 의미 분할(semantic segmentation)의 성능을 크게 향상시켰습니다. 그러나 클래스 불균형(class imbalance) 및 인스턴스 불균형(instance imbalance)은 여전히 지속적인 도전 과제로 남아 있습니다. 본 논문에서는 복잡한 스티어러블 파이프(compound steerable pyramid)를 통해 분해된 서브밴드(subband) 이미지에서 상호 정보(mutual information)를 활용하는 복잡한 웨이브렛 상호 정보(CWMI) 손실 함수(loss function)를 제안합니다.

- **Technical Details**: CWMI 손실은 다중 규모(multiscale) 구조 정보를 효율적으로 추출하기 위해 복잡한 스티어러블 파이프의 강력한 분해 기능과 상호 정보의 통계적 특성을 결합합니다. 이 접근법은 지역적 단계(Local phase), 방향성(orientation), 구조적 특성을 손실 계산에 명시적으로 통합하여 구조적 일관성과 경계 보존을 보장합니다. CWMI 손실은 낮은 계산 비용을 유지하면서도 다양한 클래스 및 인스턴스 불균형 문제를 극복하는 데 적합합니다.

- **Performance Highlights**: 다양한 분할 데이터셋에서 수행된 광범위한 실험 결과, CWMI 손실은 최첨단 방법들과 비교하여 픽셀 정확도(pixel-wise accuracy) 및 위상 메트릭(topological metrics) 모두에서 유의미한 개선을 보였습니다. 또한, 11개의 최신 손실 함수와 비교했을 때 최소한의 계산 오버헤드를 도입하며 우수한 성능을 입증했습니다. 이러한 결과는 CWMI 손실이 구조적 유사성과 경계 보존을 고려한 강력한 기술임을 보여줍니다.



### Milmer: a Framework for Multiple Instance Learning based Multimodal Emotion Recognition (https://arxiv.org/abs/2502.00547)
- **What's New**: 최근 연구에 따르면 감정 인식(Emotion Recognition)은 인간 행동을 이해하는 열쇠로 작용하며, 이를 위한 새로운 멀티모달 프레임워크인 Milmer가 소개되었습니다. Milmer는 얼굴 표정 분석(Facial Expression Analysis)과 EEG 신호를 통합하여 감정을 인식하는데 있어 새로운 접근 방식을 제시합니다. 이 프레임워크는 transformer 기반 융합 방식(Fusion Approach)을 사용하여 시각적 및 생리적 모달리티를 효과적으로 결합합니다.

- **Technical Details**: Milmer 프레임워크는 EEG 전처리 모듈, 얼굴 특징 추출 및 균형 조정 모듈, 그리고 크로스 모달 융합 모듈로 구성됩니다. 본 연구는 감정 관련 데이터셋에서 사전 훈련된 Swin Transformer를 미세 조정(Fine-tune)하여 시각적 특징 추출을 향상시킵니다. 또한, 크로스 어텐션 메커니즘(Cross-Attention Mechanism)을 도입하여 토큰 표현(Token Representation)을 모달리티 간에 균형 있게 유지하여 효과적인 특징 통합을 보장합니다.

- **Performance Highlights**: DEAP 데이터셋에서 실시된 실험 결과, 제안된 Milmer 프레임워크는 96.72%의 분류 정확도를 달성하며, 멀티 클래스 감정 인식(Multi-class Emotion Recognition) 과제에서 우수한 성능을 보였습니다. 각 모듈의 기여도는 ablation study를 통해 검증되어, 고급 특징 추출과 융합 전략이 감정 인식 성능 향상에 있어 중요함을 강조하였습니다. 이는 단일 모달리티 접근 방식에 비해 감정 인식 정확도를 크게 개선하는 결과를 제공합니다.



### CAD: Confidence-Aware Adaptive Displacement for Semi-Supervised Medical Image Segmentation (https://arxiv.org/abs/2502.00536)
Comments:
          9 pages, 3 figures, 4 tables

- **What's New**: 이 논문에서는 Confidence-Aware Adaptive Displacement (CAD)라는 새로운 프레임워크를 소개합니다. CAD는 신뢰도가 낮은 영역을 신뢰도가 높은 패치로 동적으로 대체하여 세미-슈퍼바이즈드 의료 이미징 분할의 질을 향상시킵니다. 이 접근 방식은 학습 과정 중에 최대 교체 크기와 신뢰도 임계값을 조정하여 세분화 품질을 점진적으로 개선합니다.

- **Technical Details**: CAD는 최적화를 위해 감소하는 신뢰도 임계값과 동적인 공간 크기 조정을 사용합니다. 수학적으로, 의료 이미지를 3D 볼륨 X로 정의하고 각 볼륨의 목표는 레이블 맵 Y를 예측하는 것입니다. 주어진 데이터 세트는 레이블이 있는 데이터 N과 레이블이 없는 데이터 M으로 구성되어 있으며, N은 M에 비해 상당히 적습니다.

- **Performance Highlights**: 실험 결과, CAD는 기존 방법들보다 세분화 품질을 상당히 개선하여 새로운 최첨단 정확도를 달성했습니다. 또한, 공개 의료 데이터 세트에서 여러 테스트를 통해 CAD가 세미-슈퍼바이즈드 설정에서의 최적의 성능을 입증했습니다. 이 연구 결과는 의료 이미징 데이터의 레이블링 부족 문제를 해결하는데 기여할 것으로 기대됩니다.



### Work-Efficient Parallel Non-Maximum Suppression Kernels (https://arxiv.org/abs/2502.00535)
Comments:
          Code: this https URL

- **What's New**: 본 논문에서는 효과적인 Non-Maximum Suppression (NMS) 알고리즘을 제안합니다. 이 알고리즘은 임베디드 GPU 아키텍처를 염두에 두고 개발되었으며, 수천 개의 동시 감지를 처리할 수 있는 높은 확장성을 가지고 있습니다. 이러한 알고리즘은 기존의 NMS 방법과 다르게, 슬라이딩 윈도우 또는 단일 샷 CNN 메타 아키텍처에서 발생하는 중복 후보 윈도우를 해결합니다.

- **Technical Details**: 우리가 제안한 알고리즘은 CUDA의 맵/리듀스 커널을 사용하여 NMS 문제를 병렬로 해결합니다. 이 과정에서 Boolean 인접성 행렬을 활용하여 데이터 종속성을 회피함으로써 작업 효율성을 높입니다. 구현된 커널은 NVIDIA Tegra 시리즈와 같은 다양한 플랫폼에서실험되어 나쁘지 않은 메모리 대역폭과 코어 수 개선을 통해 성능을 향상시킵니다.

- **Performance Highlights**: NMS 알고리즘의 결과, NVIDIA Tegra X1 및 X2에서 초당 1ms의 속도로 1024개의 동시 감지를 처리할 수 있는 능력을 보여주었습니다. 또한, 제안된 병렬 NMS 알고리즘은 기존의 최첨단 NMS 방법과 비교했을 때 14배에서 40배까지 빠른 성능을 자랑합니다. 이는 CNN을 학습하는 데 필요한 시간을 줄여 주며, 실시간 컴퓨터 비전 응용 프로그램에 매우 유용합니다.



### Vision-Language Modeling in PET/CT for Visual Grounding of Positive Findings (https://arxiv.org/abs/2502.00528)
- **What's New**: 이 연구는 PET/CT의 시각적 기초(visual grounding)를 위해 새로운 약한 라벨링 파이프라인을 개발했습니다. 기존의 큰 주석된 이미지-텍스트 데이터셋이 부족한 상황에서, 이 파이프라인은 PET/CT 보고서의 긍정적 발견을 특정 이미지 위치와 연결하는 자동화된 방법을 제공합니다. 개발된 ConTEXTual Net 3D 모델은 대규모 언어 모델의 텍스트 임베딩을 3D nnU-Net과 결합하여 훈련되었습니다.

- **Technical Details**: 이 모델은 25,578개의 PET/CT 이미지 및 보고서에서 11,356개의 문장-라벨 쌍을 추출하여 학습되었습니다. 특히, SUVmax와 축 슬라이스 번호를 기반으로 한 약한 라벨링을 통해 PET/CT의 질병 발견을 정확하게 나타내는 이미지 라벨을 생성하였습니다. ConTEXTual Net 3D는 3D nnU-Net 프레임워크를 바탕으로 설계되어 공간적 특성을 효과적으로 추출하면서 텍스트 정보와의 상호작용을 통해 세그멘테이션 성능을 향상시킵니다.

- **Performance Highlights**: ConTEXTual Net 3D는 LLMSeg 및 두 명의 핵의학 의사의 성과와 비교했을 때 F1 점수 0.80으로 우수한 성능을 보였습니다. 모델은 FDG(0.78) 및 DCFPyL(0.75) 검사에서 더 나은 성능을 보였으나, DOTATE(0.58) 및 Fluciclovine(0.66) 검사에서는 성능 저하가 있었습니다. 전반적으로 이 연구는 3D 시각적 기초 모델의 개발을 용이하게 하였지만, 핵의학 의사의 성과에 비해서는 여전히 개선의 여지가 있음을 보여줍니다.



### Video Latent Flow Matching: Optimal Polynomial Projections for Video Interpolation and Extrapolation (https://arxiv.org/abs/2502.00500)
Comments:
          39 pages, 6 figures

- **What's New**: 이번 논문에서는 Video Latent Flow Matching (VLFM)이라는 효율적인 비디오 모델링 프로세스를 소개합니다. 기존 연구들이 비디오 생성을 위해 무작위로 잠재 패치를 샘플링한 것과 달리, VLFM은 강력하게 사전 훈련된 이미지 생성 모델에 의존하여 시간 의존적인 비디오 프레임으로 변환할 수 있는 특정 캡션 유도 흐름을 모델링합니다. 이 연구는 텍스트-비디오 모델링의 단순화를 통한 새로운 알고리즘 설계를 목표로 하고 있습니다.

- **Technical Details**: VLFM은 임의의 프레임 속도로 비디오 생성을 가능하게 하며, HiPPO 프레임워크를 통해 최적의 다항식 프로젝션을 근사하는 방식을 사용합니다. 이를 통해 모델링 효율성이 향상되고, 고정밀 비디오 복원 및 생성을 위한 보간(interpolation) 및 외삽(extrapolation) 기능이 제공됩니다. 우리의 접근법은 주어진 데이터셋 크기에 비례하여 적은 계산 자원으로도 효율적인 모델링이 가능함을 보여줍니다.

- **Performance Highlights**: 실험 결과, VLFM은 OpenVid-1M, MiraData, Pixabay의 비디오와 같은 다양한 데이터셋에서 텍스트-비디오 생성, 보간 및 외삽에 있어 강력한 성능을 발휘하였습니다. 연구는 VLFM의 이론적 장점과 함께 실제 비디오 애플리케이션에서의 잠재력을 강조합니다. 특히, VLFM은 분산 모델 및 디퓨전 변환기(Diffusion Transformer)와의 연계를 통해 더욱 효과적인 성능을 나타냅니다.



### A framework for river connectivity classification using temporal image processing and attention based neural networks (https://arxiv.org/abs/2502.00474)
Comments:
          15 pages, 8 figures

- **What's New**: 이번 연구에서는 기후 변화와 관련된 극단적인 날씨 사건이 강과 하천의 연결성(connectivity)에 미치는 영향을 측정하기 위해 자동화된 트레일 카메라 이미지 분류 시스템을 개발했습니다. 기존의 전통적인 유량 측정기(gauge)는 비용이 많이 들고 대규모 강에 한정되지만, 새롭게 제안된 방법은 저비용으로 쉽게 배치할 수 있는 방법입니다. 이 시스템은 이미지 처리(image processing), 이미지 증대(augmentation) 및 머신러닝(machine learning)으로 구성되어 있습니다.

- **Technical Details**: 이미지 전처리 단계는 7가지 이미지 품질 필터를 적용하고, 잎사귀에 기반한 루마 분산 감소(luma variance reduction), 리사이징(resizing) 및 하단 중앙 크롭(bottom-center cropping) 등의 과정을 포함합니다. 생성적 증대(generative augmentation)는 확산 모델(diffusion models)을 사용하여 변동적인 양으로 이미지를 균형 잡고, 이후 라벨이 지정된 형태로 머신러닝 분류 모델에 전달됩니다. 또한, 비전 트랜스포머 아키텍처(vision transformer architecture)와 시계열 이미지 향상(temporal image enhancement)을 활용합니다.

- **Performance Highlights**: 실험 결과, 새로운 미지의 사이트 이미지에 대한 정확도가 75%에서 90%로 증가함을 보여주었습니다. 이는 시간 기반 이미지 처리와 주의(attention) 기반 모델의 조합이 효과적으로 미지의 강 연결성 이미지를 분류하는 데 기여함을 나타냅니다. 연구는 2018-2020년 동안 Connecticut 주 에너지 및 환경 보호부(Department of Energy and Environmental Protection) 직원들이 캡처하고 라벨링한 데이터셋을 활용하였습니다.



### Evaluation of End-to-End Continuous Spanish Lipreading in Different Data Conditions (https://arxiv.org/abs/2502.00464)
Comments:
          Accepted in the "Language Resources and Evaluation" journal, Springer Nature

- **What's New**: 이번 논문은 스페인어에 대한 자동 연속 립리딩(continuous lipreading)에서의 주목할 만한 발전을 제시합니다. 특히, CTC/Attention 아키텍처를 기반으로 하는 엔드 투 엔드(end-to-end) 시스템을 소개하며, 대규모 데이터베이스와 강력한 주의(attention) 메커니즘의 활용 덕분에 진전을 이뤘습니다. 이것은 기존 성능을 크게 개선한 결과를 보여줍니다.

- **Technical Details**: 연구에서는 두 가지 상이한 특성을 가진 코퍼스(corpora)에서 실험이 진행되어 최첨단(state-of-the-art) 결과를 달성했습니다. 또한, 아키텍처를 구성하는 다양한 구성 요소들이 음성 인식 품질에 미치는 영향을 조사하기 위한 철저한 ablative(절단) 연구가 수행되었습니다. 이를 통해 립리딩의 정확도 향상에 기여한 여러 메커니즘이 분석되었습니다.

- **Performance Highlights**: 엄격한 오류 분석을 통해 자동 시스템 학습에 영향을 미칠 수 있는 다양한 요인들을 조사했습니다. 마지막으로, 새로운 스페인어 립리딩 벤치마크(benchmark)가 확립되어, 연구자들이 이 시스템을 개선하는 데 참고할 수 있는 자원이 마련되었습니다. 코드와 학습된 모델은 공개된 URL에서 이용 가능합니다.



### MambaGlue: Fast and Robust Local Feature Matching With Mamba (https://arxiv.org/abs/2502.00462)
Comments:
          Proc. IEEE Int'l Conf. Robotics and Automation (ICRA) 2025

- **What's New**: 최근 몇 년간, 딥러닝 기반의 강력한 매칭 방법이 컴퓨터 비전 과제에서 활발하게 연구되고 있습니다. 하지만 강력하면서도 빠른 매칭 기술에 대한 필요는 여전히 남아 있습니다. 이를 해결하기 위해, MambaGlue라는 새로운 지역 특징 매칭 방법을 제안합니다. MambaGlue는 Mamba 아키텍처와 Transformer 아키텍처를 결합하여 특징 매칭의 정확도와 효율성을 높입니다.

- **Technical Details**: MambaGlue는 MambaAttention 믹서와 딥 신뢰도 점수 회귀기라는 두 가지 모듈로 구성됩니다. MambaAttention 믹서는 Mamba 기반의 자기 주의(self-attention) 구조를 통해 지역적 및 전역적 맥락을 선택적으로 이해하는 역할을 하며, 깊은 신뢰도 점수 회귀기는 주어진 예측의 신뢰도를 평가하는 MLP 기반 아키텍처입니다. 이러한 구조적 특징들은 MambaGlue가 실제 응용에서 강인성과 효율성 사이의 균형을 이룰 수 있도록 합니다.

- **Performance Highlights**: MambaGlue는 다양한 공개 데이터 세트에서 실험하여, 기본 접근 방식에 비해 성능 향상을 입증합니다. 특히, MambaGlue는 스파스 특징 매칭 방법 중 최신 기법을 능가하는 성능을 보이며, 낮은 지연 시간으로 신속하게 처리할 수 있는 강점을 가지고 있습니다. 이러한 성능 개선은 Mamba 아키텍처의 선택적 초점을 활용하여 각 레이어의 성능을 극대화함으로써 이루어집니다.



### SatMamba: Development of Foundation Models for Remote Sensing Imagery Using State Space Models (https://arxiv.org/abs/2502.00435)
- **What's New**: Foundation models (기초 모델)에 대한 관심이 원격 감지(헬리콥터/위성 이미지) 분야에서 증가하고 있으며, 다양한 기반 모델들이 다중 스펙트럼 및 고해상도 이미지와 같은 데이터에 대해 괄목할 만한 성과를 보이고 있다. 본 연구에서는 SatMamba라는 새로운 프레임워크를 제안하며, 이는 Masked Autoencoders (MAE)와 State Space Model을 결합하여 선형적으로 계산량을 줄이는 특징이 있다. SatMamba는 기존의 Vision Transformers (ViTs) 과 비교할 때 다양한 이미지-대-이미지 하위 작업에서 경쟁력 있는 성능을 보여줄 것으로 기대된다.

- **Technical Details**: Masked Autoencoder(MAE)는 원격 감지 분야에서 기초 모델을 사전 학습하는 데 널리 사용되는 아키텍처이다. 이미지 입력 I는 토큰 S로 변환되며, MAE는 입력의 일부분을 마스킹한 후 남은 토큰으로 학습한다. Mamba 아키텍처는 State Space Model(SSM)의 새로운 클래스이며, 이 연구에서는 Mamba 2를 사용하여 다중 방향에서 이미지를 스캔해 더 포괄적인 표현을 학습하도록 설계되었다.

- **Performance Highlights**: 실험을 통해 SatMamba는 다양한 이미지-대-이미지 하위 작업에서 ViT 기반 모델들에 비해 두드러진 성능을 보여주었다. 특히, 고해상도 이미지를 활용한 연구에서 유망한 결과를 도출하였으며, 이는 기초 모델을 보다 효율적으로 사용할 수 있는 기반을 마련하는 데 기여할 것으로 보인다. 또한 SatMamba는 다중 스펙트럼, 중간 해상도 원격 감지 이미지 등 다양한 이미지 도메인에서도 적용 가능하다.



### CAT Pruning: Cluster-Aware Token Pruning For Text-to-Image Diffusion Models (https://arxiv.org/abs/2502.00433)
- **What's New**: 이번 연구에서는 token-level pruning과 caching 기술을 통합한 새로운 가속 전략을 제시합니다. 이 방법은 denoising 과정에서 발생하는 계산 비용을 줄이기 위해 상대적 노이즈 크기를 사용하여 중요한 token 변화를 식별합니다. 또한, 공간 클러스터링 및 분포 균형을 고려하여 token 선택을 개선합니다.

- **Technical Details**: 연구팀은 모델의 각 timesteps에 따라 token의 중요성을 정량화하는 메트릭을 개발하여, 각 레이어에서 일관된 token 집합을 선택할 수 있도록 합니다. 노이즈 크기 기반으로 token을 선택하여 caching 및 재사용이 노이즈 공간에서 이루어지게 하고, 이를 통해 보다 효율적인 모델 실행을 가능하게 합니다. 특히, token의 선택 빈도를 추적하여 분포적 균형을 유지하는 것이 중요합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 전모델 대비 50%-60%의 계산 비용을 절감하면서 모델의 성능을 유지하는 것으로 나타났습니다. 다양한 표준 데이터셋에서 비교 평가하여, 제안된 접근 방식의 유효성을 입증하였습니다. 이는 diffusion 모델의 효율성을 크게 향상시키는 데 기여할 것으로 기대됩니다.



### TeST-V: TEst-time Support-set Tuning for Zero-shot Video Classification (https://arxiv.org/abs/2502.00426)
- **What's New**: 이번 논문에서는 새로운 프레임워크인 TEst-time Support-set Tuning (TEST-V)를 제안합니다. 이 방법은 제로샷 비디오 분류를 위해 클래스를 다양하게 지원하는 샘플을 확장하고 동적으로 중요한 단서를 발굴하도록 설계되었습니다. 특히, Multi-prompting Support-set Dilation (MSD)와 Temporal-aware Support-set Erosion (TSE) 모듈을 활용하여 지원 세트의 다양성을 확보하고 기초적 주요 단서를 추출합니다.

- **Technical Details**: 경험적으로, TEST-V는 사전 훈련된 비전-언어 모델(VLM)을 기반으로 하여, 문장 프롬프트를 통해 다수의 지원 샘플을 생성합니다. MSD 모듈은 LLM을 통해 다중 프롬프트를 생성하여 지원 세트를 확장하고, TSE 모듈은 예측 일관성을 기반으로 각 프레임의 기여도를 조절합니다. 이러한 방식으로 각 비디오 프레임의 중요한 특성을 여러 단계에서 동적으로 조정할 수 있습니다.

- **Performance Highlights**: TEST-V는 네 가지 벤치마크에서 기존의 최고 성능을 달성하였으며, CLIP, BIKE, VIFi-CLIP 모델보다 평균적으로 각각 2.98%, 2.15%, 1.83% 더 높은 정확도를 기록하였습니다. 이 연구는 TEST-V의 해석성을 강조하며 지원 세트의 확장 및 침식을 통해 성능을 개선하는 방법론을 분명히 보여줍니다.



### MQuant: Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization (https://arxiv.org/abs/2502.00425)
Comments:
          First quantization solution for Multimodal large language models applicable to 5 mainstream MLLMs

- **What's New**: MQuant은 멀티모달 대형 언어 모델(MLLMs)의 고유한 도전에 대처하기 위해 설계된 포스트 트레이닝 양자화(post-training quantization, PTQ) 프레임워크입니다. 기존의 양자화 방법이 MLLMs에 적합하지 않은 이유는 다양한 종류의 시각적 및 텍스트 토큰 간의 분포 차이, 높은 추론 지연(inference latency), 그리고 해밀턴 변환에 의한 극단적인 아웃라이어(outlier) 문제가 발생하기 때문입니다. MQuant은 이러한 문제를 해결하기 위해 모달리티(모드) 별 정적 양자화, 주의 불변 유연한 스위칭 및 회전 크기 억제와 같은 새로운 방법론을 도입했습니다.

- **Technical Details**: MQuant은 세 가지 주요 방법론을 사용하여 MLLMs의 효율성을 극대화합니다. 첫째, 모달리티-특화 정적 양자화(Modality-Specific Static Quantization, MSQ)를 통해 시각적 및 텍스트 토큰에 대해 상이한 정적 스케일(scale)을 적용합니다. 둘째, 주의 불변 유연한 스위칭(Attention-Invariant Flexible Switching, AIFS)은 토큰 순서를 재배열하여 비싼 토큰별 스케일 계산을 피하며, 셋째, 회전 크기 억제(Rotation Magnitude Suppression, RMS)를 통해 온라인 해밀턴 회전에서 발생하는 아웃라이어를 최소화합니다.

- **Performance Highlights**: MQuant은 Qwen-VL, MiniCPM-V, CogVLM2 등 다섯 가지 주요 MLLMs에 대해 수행된 실험에서, 부동소수점(float) 정확도에서 98% 이상을 유지하면서도 추론 지연을 최대 30%까지 줄였습니다. 이는 기존의 PTQ 방법론보다 현저히 우수한 성능을 나타냅니다. MQuant은 자원 제약이 있는 환경에서 효율적이고 정확한 MLLM 추론을 위한 중요한 초석이 될 것으로 기대합니다.



### Parameter Efficient Fine-Tuning of Segment Anything Mod (https://arxiv.org/abs/2502.00418)
- **What's New**: 최근 딥러닝을 활용한 생물의학 이미지 분석에서 세그멘테이션(segmentation)의 중요성이 강조되고 있으며, 이는 개별 세포나 장기 등의 연구를 가능하게 합니다. 본 논문은 세그멘테이션을 위한 Vision foundation models인 Segment Anything Model(SAM)과 PEFT(parameter-efficient finetuning) 기법의 적용 가능성에 대해 다루고 있습니다. 저자들은 9가지 PEFT 기법을 활용하여 다양한 데이터셋에서 SAM의 성능을 평가하고, 자원 효율적인 파인튜닝 방법을 제안하였습니다.

- **Technical Details**: 연구에서는 두 가지 유형의 PEFT 방법, 즉 선택적(selective) 및 덧셈(additive) PEFT을 구별합니다. 선택적 PEFT 방법은 전체 모델의 일부 파라미터만 업데이트하고 나머지는 동결하는 방식으로, 주로 SAM의 이미지 인코더를 동결하여 성능을 극대화합니다. 덧셈 PEFT 방법은 기존의 파라미터를 동결하고 추가적인 소수의 파라미터를 도입하여 성능을 향상시키며, LoRA는 이러한 방법 중 하나로 주목받고 있습니다.

- **Performance Highlights**: 저자들은 6개의 현미경 이미지 데이터셋과 6개의 의료 이미지 데이터셋을 통한 9가지 PEFT 방법의 성능을 평가하였습니다. 초기 실험 결과, SEFT의 사용은 기존 모델에 비해 더 적은 주석(annotation)으로도 우수한 성능을 발휘하는 것으로 나타났습니다. 이 연구는 PEFT의 효과적인 적용을 통해 생물의학 이미지 세그멘테이션의 발전에 기여할 것으로 기대됩니다.



### TROI: Cross-Subject Pretraining with Sparse Voxel Selection for Enhanced fMRI Visual Decoding (https://arxiv.org/abs/2502.00412)
Comments:
          ICASSP 2025

- **What's New**: 이번 연구는 fMRI (functional Magnetic Resonance Imaging) 신호를 활용한 시각 디코딩 기술의 한계를 극복하기 위한 새로운 방법인 TROI (Trainable Region of Interest)를 제안합니다. TROI는 제한된 샘플을 가지는 교차 피험자를 위한 데이터 기반 ROI 레이블링 방법으로, 기존 수동으로 레이블된 ROIs의 단점을 해결하고자 합니다. 이 방법은 두 단계로 구성되어 있으며, voxel 마스크를 최적화하여 새로운 피험자로의 적용성을 높입니다.

- **Technical Details**: TROI 접근법은 두 단계로 구성되어 있으며, 첫 번째 단계에서는 LASSO 정규화와 로우 패스 필터를 사용하여 voxel 마스크를 신속하게 생성하고 입력 레이어 차원을 결정합니다. 두 번째 단계에서는 학습률 리와인딩 전략을 적용하여 입력 레이어를 다운스트림 작업을 위해 미세 조정합니다. 이 과정을 통해 새로운 피험자의 입력 레이어를 효율적으로 최적화할 수 있습니다.

- **Performance Highlights**: TROI 방식을 사용한 실험은 기존의 최첨단 fMRI 디코딩 모델인 MindEye2와의 비교에서, 주목받는 성능 향상을 보여주었습니다. 특히, 시각 반환 및 재구성 작업에서 주석이 달린 ROI 마스크를 사용하는 기존 방법보다 우수한 성능을 나타냈습니다. 이러한 결과는 TROI가 다양한 피험자에 대한 fMRI 디코딩의 실용성을 높일 수 있음을 시사합니다.



### Exploring Linear Attention Alternative for Single Image Super-Resolution (https://arxiv.org/abs/2502.00404)
Comments:
          This paper has been published to IEEE International Joint Conference on Neural Networks. Feel free to contact on nomodeset@qq.com

- **What's New**: 본 연구에서는 기존의 SISR(Single Image Super-Resolution) 기술의 한계를 극복하기 위해 Omni-Scale RWKV Super-Resolution (OmniRWKVSR) 모델을 제안하였습니다. 이 모델은 Receptance Weighted Key Value (RWKV) 아키텍처와 Visual RWKV Spatial Mixing (VRSM) 및 Visual RWKV Channel Mixing (VRCM) 기법을 결합하여 향상된 성능을 구현합니다. 기존 모델들과 비교하여 OmniRWKVSR는 이미지 재구성의 질과 효율성을 동시에 개선하였습니다.

- **Technical Details**: OmniRWKVSR는 여러 가지 기능 추출 기법을 통합하여 다양한 스케일과 공간 변환을 포착합니다. 이 모델은 ChannelMix 메커니즘을 통해 채널별 정보 흐름을 증진시켜 MLP(Matrix Linear Processing)에 비해 우수한 성능을 보여줍니다. 또한, Omni-Quad Shift 메커니즘을 도입하여 장기적인 의존성을 효과적으로 캡처할 수 있게 하였습니다.

- **Performance Highlights**: OmniRWKVSR 모델은 4배 슈퍼 해상도(Super-Resolution) 작업에서 MambaIR 모델에 비해 PSNR(Peak Signal-to-Noise Ratio)에서 평균 0.26%, SSIM(Structural Similarity Index)에서 0.16% 향상된 결과를 보여주었습니다. 본 연구에서 제안한 모델은 Set14 및 BSD100과 같은 인기 있는 데이터셋에서 최고 점수를 기록하며, MambaIR 모델보다 약 15% 적은 평균 학습 시간으로도 성능을 발휘하였습니다.



### Enhancing Highway Safety: Accident Detection on the A9 Test Stretch Using Roadside Sensors (https://arxiv.org/abs/2502.00402)
- **What's New**: 이 논문에서는 사람의 오류, 빠른 사고 탐지 및 즉각적인 의료 대응을 제안하여 도로 교통 사고를 줄이기 위한 새로운 사고 탐지 프레임워크를 소개합니다. 실제 고속도로 사고 데이터셋이 공개되며, 294,924개의 라벨이 붙은 2D 박스와 93,012개의 라벨이 붙은 3D 박스가 포함되어 있습니다. 이 데이터는 다차원적 사고 및 근접 사고를 실시간으로 탐지하고 분석하는 능력을 갖추었습니다.

- **Technical Details**: 네트워크 모델 YOLOv8을 사용하여 사고를 감지하는 학습 기반 접근법과 차량 경로를 분석하여 이상 행동을 감지하는 규칙 기반 접근법이 결합되었습니다. 규칙 기반 접근법에서는 갑작스러운 정지 및 비정상적인 차선 변경을 기준으로 사고를 분류합니다. 또한, 프레임에서 '사고'로 여겨진 사건은 최소 3개의 연속 프레임에서 감지되어야 하며, 여러 카메라의 감지를 집계하여 정확도를 높입니다.

- **Performance Highlights**: 논문에서 제안한 프레임워크는 12,290개의 15분 비디오 세그먼트에서 831,969대의 차량을 감지했으며, 1건의 실제 사고를 정밀하게 식별했습니다. 규칙 기반 접근법은 프레임당 10.41밀리초가 소요되어 매우 빠른 실시간 처리를 보여줍니다. 이 연구의 기여는 도로 교통 안전 향상과 자율주행 시스템 발전에 중요한 기초 자료로 활용될 것입니다.



### Minimalistic Video Saliency Prediction via Efficient Decoder & Spatio Temporal Action Cues (https://arxiv.org/abs/2502.00397)
Comments:
          Accepted at 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2025)

- **What's New**: 이 논문은 ViNet 아키텍처를 기반으로 한 36MB 모델인 ViNet-S를 소개합니다. 이 모델은 성능을 저하시키지 않으면서 모델의 크기와 파라미터를 크게 줄이는 경량 디코더가 특징입니다. 또한 ViNet-A(148MB)는 전통적인 비디오 주목 모델과는 다른 spatio-temporal action localization (STAL) 기능이 통합되었습니다. 연구 결과, ViNet-S와 ViNet-A의 앙상블을 통해 최첨단 성능을 달성했습니다.

- **Technical Details**: ViNet-A는 end-to-end 학습이 가능한 시각 정보 전용 모델로, SlowFast 네트워크를 비디오 인코더로 사용하여 공간 및 시간 측면에서 지역화된 동작을 효과적으로 캡처합니다. 이 모델은 고해상도의 3D 합성곱 네트워크를 기반으로 하며, 컴퓨팅 비용을 낮추기 위한 경량 디코더와 효율적인 변형을 통합하고 있습니다. 또한 ViNet-S는 원본 ViNet보다 성능이 향상된 소형 모델로, S3D 백본을 사용합니다.

- **Performance Highlights**: ViNet-E는 ViNet-S와 ViNet-A의 장점을 결합한 앙상블 모델로, 다양한 데이터셋에서 transformer 기반 접근 방식을 초월하는 성능을 보여줍니다. 이 모델은 9백만 개의 파라미터로 1000fps 이상의 속도를 달성하여 효율성을 극대화했습니다. 다양한 실험을 통해 아홉 데이터셋에 대한 질적 및 양적 통찰력을 제공하였습니다.



### RefDrone: A Challenging Benchmark for Referring Expression Comprehension in Drone Scenes (https://arxiv.org/abs/2502.00392)
- **What's New**: 이번 연구에서는 드론 장면에서의 Refering Expression Comprehension (REC)을 위한 새로운 벤치마크인 RefDrone을 소개합니다. RefDrone 데이터세트는 8,536장의 이미지에 총 17,900개의 참조 표현을 포함하고 있으며, 세 가지 주요 도전 과제를 다루고 있습니다: 다중 스케일 및 소규모 타겟 탐지, 다중 타겟 및 비타겟 샘플, 복잡한 환경 내 풍부한 맥락 표현. 또한, RDAgent라는 준자동 주석 도구를 개발하여 주석 비용을 줄이면서도 고품질의 표현을 보장합니다.

- **Technical Details**: RefDrone 데이터세트는 이미지가 수집된 다양한 시나리오와 조명 조건을 기반으로 만들어졌으며, 주석 프로세스를 효율적으로 진행하기 위해 RDAgent라는 다중 에이전트 시스템을 사용하는 주석 프레임워크를 설계했습니다. RDAgent는 전통적인 주석 워크플로우를 복잡한 표현의 품질을 유지하면서도 저비용으로 전환할 수 있도록 설계되었습니다. 더불어, Number GroundingDINO (NGDINO)라는 새로운 방법을 도입하여 다중 타겟 및 비타겟 케이스를 처리할 수 있도록 하였습니다.

- **Performance Highlights**: 실험 결과에 따르면, NGDINO는 RefDrone과 gRefCOCO 데이터세트 모두에서 우수한 성능을 달성했습니다. 기존 REC 방법들과 비교할 때, NGDINO는 참조된 객체의 수를 학습하고 활용하여 다중 타겟과 비타겟 샘플을 보다 효과적으로 처리합니다. 이러한 연구는 드론 장면에서의 REC 능력을 크게 향상시키고, Embodied AI 분야의 발전에 기여할 것으로 기대됩니다.



### Efficient Adaptive Label Refinement for Label Noise Learning (https://arxiv.org/abs/2502.00386)
- **What's New**: 이번 연구에서는 Adaptive Label Refinement (ALR)라는 새로운 방법을 제안하여 잘못된 레이블을 피하고 깨끗한 샘플을 철저히 학습하는 두 가지 작업을 분리합니다. 기존의 복잡한 모델과 절차를 단순화하며, ALR은 사전 지식이나 보조 데이터셋 없이도 적용할 수 있는 용이성을 가지고 있습니다. 이 방법은 샘플 학습 시 노이즈의 영향을 줄이기 위해 모델이 예측한 결과를 활용하여 하드 레이블을 소프트 레이블로 업데이트합니다.

- **Technical Details**: ALR의 핵심 아이디어는 네트워크의 역사적인 예측을 사용하여 소프트 레이블을 생성하는 시간 통합 전략을 사용하는 것입니다. 이 과정에서 엔트로피 손실(entropy loss)을 도입하여 높은 신뢰도를 가진 소프트 레이블을 점진적으로 '단단하게' 변형하며, 모델의 정확도를 향상시키고 깨끗한 레이블에서 학습을 더욱 집중할 수 있도록 합니다. 이 접근 방식은 있으면서도 효율적이며 복잡한 하이퍼파라미터 조정이 필요하지 않다는 장점이 있습니다.

- **Performance Highlights**: ALR는 인위적으로 레이블 노이즈가 포함된 데이터셋(CIFAR-10/100)과 실제 노이즈가 존재하는 데이터셋(ANIMAL-10N, Clothing1M, WebVision)에서 실험을 통해 효과성을 검증하며, 기존의 최첨단 방법들을 초월하는 성능을 보여줍니다. 특히, ALR은 전통적인 크로스 엔트로피 손실보다 더 나은 성능을 나타내며, 다양한 자연 및 인공 데이터셋에서 우수한 정확도를 기록합니다.



### Masked Generative Nested Transformers with Decode Time Scaling (https://arxiv.org/abs/2502.00382)
- **What's New**: 최근 시각적 생성 영역에서 뛰어난 품질의 콘텐츠를 생성하는데 있어 큰 발전이 있었지만, 모델의 추론 효율성에서 병목 현상이 발생하고 있습니다. 본 연구는 여러 번의 계산을 요하는 전통적인 접근 방식을 개선하기 위해, 생성 과정의 각 부분에 필요한 계산량을 조절하는 방안과 계산 재사용(Cache) 전략을 도입합니다. 이를 통해 다양한 크기의 모델을 효과적으로 활용하여 계산 비용을 줄이면서도 유사한 품질의 결과를 제공합니다.

- **Technical Details**: 이 연구에서는 Masked Generate Nested Transformers (MaGNeTS)라는 새로운 모델 아키텍처를 제안하며, 디코딩 과정 동안 모델 크기를 단계적으로 조정하여 계산 효율성을 극대화합니다. MaGNeTS는 작은 모델이 더 많은 토큰을 처리할 수 있도록 지원하며, 큰 모델이 더 적은 토큰을 처리하도록 합니다. 실험을 통해 ImageNet, UCF101, Kinetics600 데이터셋에서 엄청난 계산 효율성을 입증하고, 각 모델이 공유하는 매개변수 공간과 함께 동작하는 방식도 설명합니다.

- **Performance Highlights**: MaGNeTS는 기존 대비 약 3배 적은 계산으로 유사한 품질의 이미지를 생성할 수 있으며, 최첨단 모델들과 경쟁력 있는 성능을 보여줍니다. 특히, MaskGIT++와의 비교 실험에서 이 모델은 성능 개선이 두드러지며, 모든 작업에서 2.5배에서 3.7배에 달하는 계산 이득을 나타냅니다. 또한, 비디오 데이터셋에 대해서도 유의미한 효과를 증명했습니다.



### Latent Action Learning Requires Supervision in the Presence of Distractors (https://arxiv.org/abs/2502.00379)
Comments:
          Preprint. In review

- **What's New**: 최근 Latent Action Policies (LAPO)를 기반으로 하는 latent action learning이 대규모 로봇 공학 및 강화 학습 분야에서 놀라운 사전 훈련 효율성을 보였습니다. 이전 연구는 주로 진정한 행동(ground-truth actions)만으로 설명 가능한 간섭 요인이 없는 데이터에 집중했으나, 실제 비디오 데이터에는 행동과 관련된 방해 요소(Distractors)가 포함되어 있다는 점에서 한계가 있었습니다. 이 논문에서는 이러한 방해 요소가 latent action learning에 미치는 영향을 실증적으로 분석하고, LAPO가 이러한 시나리오에서 어려움을 겪는다고 밝힙니다.

- **Technical Details**: 기존 연구는 주로 간섭 요인이 없는 데이터에서 latent action learning의 효과를 연구했습니다. 이 작업에서는 Distracting Control Suite (DCS)를 사용하여 행동-연관 방해 요소의 영향을 조사하고, LAPO의 품질을 8배 향상시키는 LAOM이라는 간단한 LAPO 수정 버전을 제안합니다. 중요한 점은, 전체 데이터 세트의 약 2.5%만으로도 진정한 행동에 대한 지도 감독을 제공했을 때, downstream 성능이 평균 4.2배 향상된다는 것입니다.

- **Performance Highlights**: 연구 결과, 주어진 예산의 행동 레이블 수로 더 나은 결과를 얻기 위해서는 방해 요소가 존재할 때 Latent Action Models (LAM)을 우선 학습하고 그 이후에 진정한 행동으로 디코딩하는 현재의 파이프라인은 최적이 아님을 보였습니다. 본 논문은 또한 지도 감독을 활용하며 latent action learning의 성능이 더 잘 일반화된다는 점을 강조하며, 이는 기존의 역동적 모델에 기반한 접근 방식과 비교하여 개선된 결과를 제공합니다.



### Scalable Framework for Classifying AI-Generated Content Across Modalities (https://arxiv.org/abs/2502.00375)
Comments:
          12 pages, Defactify4 @ AAAI 2025

- **What's New**: 빠르게 성장하고 있는 생성형 인공지능(AI) 기술의 발전은 인간과 AI 생성 콘텐츠를 효과적으로 구분하고 다양한 생성 모델의 출력을 분류하는 것의 중요성을 높였다. 이 논문은 지각 해싱(perceptual hashing), 유사성 측정(similarity measurement), 그리고 의사 레이블링(pseudo-labeling)을 통합하여 이러한 문제를 해결하는 확장 가능한 프레임워크를 제안한다. 우리의 방법은 재훈련 없이 새로운 생성 모델을 통합할 수 있도록 하여 동적인 시나리오에서 적응성과 견고성을 보장한다.

- **Technical Details**: 제안된 방법은 지각 해싱, 유사성 측정 및 의사 레이블링으로 구성되어 있다. 학습 단계에서는 ArcFace 손실을 사용하여 모델을 훈련하고 데이터셋을 증강하기 위해 의사 레이블링을 적용한다. 새로운 레이블 적응 단계에서는 새로운 데이터에서 추출된 특징을 기존 특징 저장소에 통합하고, 추론 단계에서는 들어오는 데이터에서 추출한 특징을 저장된 특징과 비교하여 가장 유사한 레이블을 결정한다. k-NN 접근 방식을 통해 높은 차원의 특징 공간에서 유사성을 효과적으로 측정할 수 있다.

- **Performance Highlights**: Defactify4 데이터셋에 대한 평가에서 인간 생성 콘텐츠와 AI 생성 콘텐츠를 구분하고 생성 방법 간에 높은 정확도를 달성하여 텍스트 및 이미지 분류 작업에서 경쟁력 있는 성능을 보여주었다. 이 결과는 생성형 AI의 지속적인 발전에 따라 실제 응용에서 이 프레임워크의 잠재력을 부각시킨다. 마지막으로, 소스 코드는 공개적으로 사용할 수 있다.



### NAVER: A Neuro-Symbolic Compositional Automaton for Visual Grounding with Explicit Logic Reasoning (https://arxiv.org/abs/2502.00372)
- **What's New**: 이번 논문에서는 시각적 주변 탐지(Visual Grounding, VG) 작업을 중심으로, 인간의 인지와 유사한 복잡한 추론을 요구하는 새로운 방법 NAVER를 제안합니다. NAVER는 명시적 확률적 논리 추론을 통합하여 복잡한 쿼리 해석을 돕는 독창적인 조합적 접근 방식을 사용하고 있습니다. 이 시스템은 자기 수정 메커니즘을 포함하여 오류 전파를 방지하고 해석 가능성을 높이며, 향상된 Robustness를 보장합니다.

- **Technical Details**: NAVER는 ProbLog를 활용하여 확률적 논리 추론을 통합하고, 유한 상태 오토마타(Deterministic Finite-State Automaton, DFA)를 기반으로 설계되었습니다. 이러한 구조는 복잡한 맥락 제약과 관계들을 처리할 수 있도록 하고, 중간 결과에 따라 동적으로 상태를 전환할 수 있게 합니다. 각 단계와 모듈에는 자기 수정 메커니즘이 내장되어 있어, 하나의 단계에서 발생한 오류가 다른 단계에 영향을 미치지 않도록 설계되었습니다.

- **Performance Highlights**: NAVER는 최근의 통합형 및 조합형 방법들과 비교하여 최신 국면(State of the Art, SoTA) 성능을 달성한 결과를 보였습니다. 상세한 ablation study를 통해 각 구성 요소의 효과성이 검증되었으며, 다양한 데이터셋에서 뛰어난 성능을 입증했습니다. 이는 NAVER가 복잡한 쿼리를 처리하는 효율성과 신뢰성을 크게 향상시켰음을 나타냅니다.



### Shape from Semantics: 3D Shape Generation from Multi-View Semantics (https://arxiv.org/abs/2502.00360)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 'Shape from Semantics'라는 새로운 접근법을 제안하며, 이는 주어진 의미(semantics)에 맞춰 다양한 시점에서 3D 모델의 기하학(geometry)과 외관(appearance)을 일치시킬 수 있는 방식입니다. 기존의 'Shape from X' 방식이 시각적 입력에 의존했다면, 우리의 방법은 텍스트 기반의 설명인 의미를 활용하여 3D 기하학을 생성합니다. 이로 인해 보다 자유롭고 창의적인 3D 자산을 생성할 수 있는 가능성이 열립니다.

- **Technical Details**: 연구는 세 단계로 나뉘어 있는데, 첫 단계에서는 의미 기반의 확산 모델을 사용해 3D Gaussian Splatting을 통해 기초 기하학과 질감을 구성합니다. 두 번째 단계에서는 이미지 및 비디오 확산 모델을 사용해 생성된 질감을 정제하고, 다양한 위성 시점에서 렌더링 결과를 추정하여 후속 단계의 신경 암시 표현(neural implicit representation)을 위한 정보를 제공합니다. 마지막 단계에서는 SDF(Signed Distance Function)를 사용해 정제된 3D 모델을 표현하며, 이를 통해 고품질의 제조 가능한 메시(mesh)를 추출합니다.

- **Performance Highlights**: 이 연구의 결과는 서로 다른 관점에서 세부적인 의미 요소들이 효과적으로 통합된 3D 모델을 생성함을 보여줍니다. 생성된 메시들은 복잡한 세부사항과 매끄러운 전환을 가지고 있어 시각적으로 매력적입니다. 또한, 이 모델들은 실제적인 제작이 가능하여, 예술 작품이나 AR/VR 응용 분야에서의 실용성을 크게 향상시킵니다.



### Embodied Intelligence for 3D Understanding: A Survey on 3D Scene Question Answering (https://arxiv.org/abs/2502.00342)
Comments:
          Work in progress

- **What's New**: 이 논문은 3D Scene Question Answering (3D SQA)에 대한 최초의 종합적인 조사 연구를 제시합니다. 3D SQA는 3D 시각 인식과 자연어 처리의 융합을 통해 지능형 에이전트가 복잡한 3D 환경을 이해하고 상호작용할 수 있도록 지원합니다. 최근의 대규모 멀티모달 모델링의 발전 덕분에 다양한 데이터셋이 생성되고 있지만, 이러한 발전은 데이터셋과 기준(Baseline) 간 비교를 통합하기 어려운 문제를 일으킵니다.

- **Technical Details**: 논문은 3D SQA의 목표, 이를 지원하는 데이터셋, 그리고 이러한 목표를 달성하기 위해 개발된 모델들을 체계적으로 검토합니다. 초기 3D SQA 개발은 수동으로 주석이 달린 데이터셋인 ScanQA와 SQA에 의해 촉진되었으며, 최근에는 3DVQA와 MSQA와 같은 프로그램적 생성 방법을 도입하여 더 다양한 질문 유형을 가진 대규모 데이터셋을 생성할 수 있게 되었습니다. 또한, 대형 비전-언어 모델(Large Vision-Language Models, LVLM)이 데이터 주석 작업을 자동화하는 데 기여하고 있습니다.

- **Performance Highlights**: 데이터셋의 진화와 방법론의 발전에 대한 리뷰를 통해 이 논문은 3D SQA 분야에서 기존 문헌의 경향을 조명합니다. 연구는 수동 주석에서 LVLM 지원 생성으로의 변화와 닫힌 세트(closed-set) 접근법에서 제로샷(zero-shot) 기법으로의 발전을 강조합니다. 또한, 멀티모달 정렬과 평가 표준화의 도전 과제를 논의하며, 이것이 분야의 미래 방향에 대한 통찰력을 제공합니다.



### BiMaCoSR: Binary One-Step Diffusion Model Leveraging Flexible Matrix Compression for Real Super-Resolution (https://arxiv.org/abs/2502.00333)
Comments:
          10 pages, 5 figures. The code and models will be available at this https URL

- **What's New**: BiMaCoSR는 기존의 이진화를 통해 수퍼 해상도(Super-Resolution, SR) 모델의 메모리 및 계산 요구를 극복하는 혁신적인 접근 방식을 제안합니다. 이 모델은 1비트 이진화(binarization)와 단일 단계 증류(one-step distillation) 기술을 결합하여 극단적인 압축(compression)과 가속(acceleration)을 달성합니다. 또한, 모델의 성능 저하를 방지하기 위해 희소 행렬 분기(sparse matrix branch, SMB)와 저랭크 매트릭스 분기(low rank matrix branch, LRMB)를 도입하여 정보의 손실 없이 효과적으로 작동하도록 설계되었습니다.

- **Technical Details**: BiMaCoSR는 서로 다른 방식으로 완전 정밀도(full-precision) 정보를 전달하는 두 가지 보조 분기를 포함하고 있습니다. SMB는 극한값을 흡수하여 출력이 고랭크(high rank)로, 많은 양의 FP 정보를 제공합니다. 반면 LRMB는 LoRA에서 착안하여 초기화된 상위 r SVD 구성 요소를 활용해 저랭크(low rank) 표현을 출력합니다. 이러한 구조 덕분에 BiMaCoSR는 메모리 및 계산 부담을 최소화할 수 있으며, 주목할 만한 성능을 발휘합니다.

- **Performance Highlights**: BiMaCoSR는 기존의 이진화 메서드보다 뛰어난 성능을 보여주며, 완전 정밀도(FP) 모드와 비교할 때 23.8배의 압축 비율과 27.4배의 속도 향상을 달성합니다. 종합적인 비교 실험을 통해 BiMaCoSR의 우수성을 입증했으며, 폭넓은 제거 연구를 통해 모델의 견고성과 효율성을 확인하였습니다. 이를 통해, 자원 제한이 있는 엣지 디바이스에서도 실용적으로 적용 가능하다는 점을 강조하고 있습니다.



### MonoDINO-DETR: Depth-Enhanced Monocular 3D Object Detection Using a Vision Foundation Mod (https://arxiv.org/abs/2502.00315)
Comments:
          8 pages, 8 figures

- **What's New**: 이번 논문에서는 Monocular 3D Object Detection (M3OD) 모델의 성능 향상을 위한 새로운 방법들을 제안하고 있습니다. 전통적인 CNN 기반 접근 방식과는 달리, Vision Transformer (ViT) 기반의 foundation model을 백본으로 사용하는 방식을 채택했습니다. 이 방법은 깊이 추정에 필요한 글로벌 특징을 잘 포착하여, 보다 정확한 물체 탐지 및 깊이 추정을 가능하게 합니다.

- **Technical Details**: 모델은 Detection Transformer (DETR) 아키텍처를 통합하여 단일 단계에서 깊이 추정과 물체 탐지 성능을 개선합니다. Hierarchical Feature Fusion Block (HFFB)을 도입하여 foundation model로부터 더 풍부한 시각적 특징을 추출하고, 대규모 데이터에 기반하여 훈련된 상대적 깊이 추정 모델을 통합하여 깊이 추정 정확도를 향상시킵니다. 또한, transformer's decoder에서 쿼리를 활용하여 2D 바운딩 박스의 참조 점과 차원을 고려하여 인식 성능을 높입니다.

- **Performance Highlights**: 제안된 MonoDINO-DETR 모델은 KITTI 3D 벤치마크와 고지 racing 환경에서 수집된 사용자 정의 데이터셋을 통해 최근의 최첨단 기법들을 초과하는 성능을 발휘했습니다. 정량적 및 정성적 평가를 통해 M3OD 작업에서 깊이 및 시각적 특징 추출의 향상을 보여주었습니다. 코드는 해당 URL에서 이용 가능합니다.



### A Diffusion Model Translator for Efficient Image-to-Image Translation (https://arxiv.org/abs/2502.00307)
- **What's New**: 최근 이미지 간 변환(image-to-image translation) 작업에서 확산 모델(diffusion models)의 적용이 증가하고 있으며, 이 연구에서는 Diffusion Model Translator (DMT)라는 경량 번역기를 통해 효율성을 높인 새로운 방법을 제안합니다. 이 접근법은 기존 이미지의 정보가 아닌 선택한 특정 시점에서 도메인 변환(distribution transfer)을 수행함으로써 효율성을 확보합니다.

- **Technical Details**: DMT는 사전 훈련된 DDPM(Denoising Diffusion Probabilistic Model)을 활용하여, 서로 다른 두 이미지 도메인의 확산 프로세스를 조정하면서 재매개화(reparameterization) 기술을 사용합니다. 이는 기존의 복잡한 정보 주입 방식을 대체하여 더 나은 성능을 제공하며, 기존 I2I 기술(예: Pix2Pix, TSIT)과 결합하여 효율성을 증가시킵니다.

- **Performance Highlights**: 다양한 I2I 작업, 즉 이미지 스타일화(image stylization), 이미지 색채화(image colorization), 분할(segmentation) 및 스케치(sketch)에서 이미지로의 변환 실험을 통해 DMT의 효과가 입증되었습니다. DMT는 품질과 효율성 모두에서 기존의 다른 방법들을 초월한 성능을 보여주며, 코드도 공개될 예정입니다.



### MCM: Multi-layer Concept Map for Efficient Concept Learning from Masked Images (https://arxiv.org/abs/2502.00266)
- **What's New**: 이번 연구는 자연어 처리에서 주로 사용되던 마스킹 전략을 시각 인식 작업인 개념 학습에 적용하는 데 중점을 두고 있습니다. 우리는 마스킹된 이미지를 통해 개념 학습을 강화할 수 있는 Multi-layer Concept Map (MCM) 방법을 제안합니다. MCM은 비대칭 개념 학습 아키텍처를 도입하여 인코더와 디코더의 서로 다른 레이어 간의 관계를 설정합니다.

- **Technical Details**: MCM 방법은 입력 이미지를 패치로 나누고 마스킹된 패치를 사용하여 개념 토큰을 학습하도록 설정됩니다. 이 방법은 양방향 그래디언트를 활용하여 구조적 특징을 학습하고, 다양한 세분화 수준에서 개념 토큰을 추적하여 이미지를 재구성합니다. MCM은 Transformer 모델 아키텍처를 기반으로 하여 기존의 방법보다 계산 비용을 현저히 줄이고, 동시에 개념 예측 성능을 향상시킵니다.

- **Performance Highlights**: MCM은 전체 이미지 패치의 75% 미만에서 학습하면서도 경쟁력 있는 예측 및 재구성 성능을 달성합니다. 이러한 방법은 개념 토큰 세트를 효과적으로 학습하는 데 초점을 맞추며, 실험 결과는 다양한 메트릭스를 통해 MCM의 개선된 성능을 입증하고 있습니다. 이외에도, 노출 개념과 결합된 새로운 이미지 생성을 가능하게 하여 아이디어의 시각적 표현 강화를 목적으로 합니다.



### INSIGHT: Enhancing Autonomous Driving Safety through Vision-Language Models on Context-Aware Hazard Detection and Edge Case Evaluation (https://arxiv.org/abs/2502.00262)
- **What's New**: INSIGHT라는 새로운 프레임워크를 도입하여, 시각적(visual) 및 텍스트(text) 입력을 결합하여 자율주행 시스템의 위험 감지와 엣지 케이스 평가를 향상시킵니다. 기존의 모델들은 드문 사건에 대한 일반화가 부족했으나, INSIGHT는 다중 모달 데이터 융합(multi-modal data fusion)을 통해 이러한 문제를 해결합니다. 이 모델은 상황 인식 개선과 보다 나은 의사결정을 통해 자율주행 시스템의 안전성을 높입니다.

- **Technical Details**: INSIGHT는 대형 언어 모델(LLM)과 비전 인코더(vision encoder)를 통합한 계층적 비전-언어 모델(VLM)입니다. 이 모델은 Vision Transformer를 사용하여 시각적 특징을 추출하고, 사전 학습된 언어 모델에서 텍스트 임베딩(text embeddings)과 융합합니다. 주의 기반 메커니즘(attention-based mechanisms)과 좌표 회귀 기법을 통해 공간적인 위험 로컬라이제이션을 최적화하며, 주석이 달린 데이터셋에서 미세 조정을 통해 엣지 케이스 감지 능력을 향상시킵니다.

- **Performance Highlights**: BDD100K 데이터셋에서의 실험 결과, INSIGHT는 기존 모델들에 비해 위험 예측의 간단함과 정확성에서 현저한 개선을 보였습니다. 특히, 드문 사건에 대한 일반화 성능이 크게 향상되어 자율주행 시스템의 전반적인 강인함과 안전성을 높이는 데 기여합니다. 이를 통해 복잡한 실제 상황에서 자율주행 기술의 신뢰도를 강화할 수 있습니다.



### Transformer-Based Vector Font Classification Using Different Font Formats: TrueType versus PostScrip (https://arxiv.org/abs/2502.00250)
Comments:
          8 pages, 8 figures, 4 tables, Submitted to IJCNN 2025. Code available at this https URL

- **What's New**: 이 연구는 벡터 글꼴 분류에서 TrueType 아웃라인보다 PostScript 아웃라인이 더 나은 성능을 보인다는 것을 보여줍니다. 벡터 글꼴에 대한 딥러닝 연구가 주로 비트맵 형식에 집중되어 있는 반면, 벡터 형식을 사용하는 것은 점점 더 중요해지고 있습니다. 이에 따라 연구진은 벡터 글꼴의 두 가지 대표적인 형식인 TrueType과 PostScript의 비교를 통해, PostScript이 우수한 임베딩 표현을 제공한다는 점을 강조합니다.

- **Technical Details**: 이 연구는 Transformer 기반 모델을 활용하여 글꼴 분류 작업을 실시하였으며, 두 가지 아웃라인 형식인 TrueType 아웃라인과 PostScript 아웃라인을 비교했습니다. TrueType 아웃라인은 포인트와 플래그의 시퀀스로 문자 형태를 정의하는 반면, PostScript 아웃라인은 명령어의 시퀀스를 사용하여 곡선을 정의합니다. 연구진은 글꼴 분류 작업에서 PostScript 아웃라인을 기반으로 하는 임베딩 표현이 더 나은 성능을 보임을 관찰하였습니다.

- **Performance Highlights**: 실험 결과, 복잡한 형태의 글꼴인 한자 글꼴과 글꼴 두께 분류 작업에서 Transformer 기반 벡터 글꼴 분류 모델이 효과적으로 적용될 수 있음을 입증하였습니다. PostScript 아웃라인에 기반한 임베딩 표현이 TrueType 아웃라인보다 우수한 성능을 보이는 이유가 세그멘테이션 프로세스와 관련이 있음을 밝혀냈습니다. 이러한 결과는 향후 벡터 그래픽스 연구에서 아웃라인 형식 선택에 중요한 통찰력을 제공할 수 있습니다.



### A Hybrid Random Forest and CNN Framework for Tile-Wise Oil-Water Classification in Hyperspectral Images (https://arxiv.org/abs/2502.00232)
- **What's New**: 이 논문에서는 하이퍼스펙트럼 이미지(HSI)에서 석유와 물을 분류하기 위한 새로운 하이브리드 랜덤 포레스트(Random Forest) 및 컨볼루션 신경망(CNN) 프레임워크를 제안합니다. 이 프레임워크는 이미지를 작고 겹치지 않는 타일로 나누어 공간적 맥락을 보존하는 데 초점을 맞추고 있습니다. 랜덤 포레스트는 픽셀 단위 분류에서 뛰어난 성능을 보이지만, 공간적 관계를 충분히 활용하지 못하는 한계를 극복하기 위해 CNN이 도입되었습니다. 이 조합은 하이퍼스펙트럼 이미지의 맥락 인식 분석을 위한 효과적인 접근 방식을 제시합니다.

- **Technical Details**: 본 연구에서 사용된 데이터셋은 365nm에서 2500nm까지 스펙트럼을 촬영한 18개의 하이퍼스펙트럼 이미지로 구성된 HOSD dataset입니다. 데이터를 전처리하는 과정에서는 잡음을 제거하고, PCA(주성분 분석)을 통해 차원 축소를 수행하여 99%의 분산을 유지하면서 32개의 주성분을 추출하였습니다. 이러한 과정은 모델의 일반화 능력을 높이고 계산 효율성을 개선하는 데 기여했습니다. 이미지를 64x64 크기의 비겹치는 타일로 나누어 학습, 검증, 테스트 세트로 분할하였습니다.

- **Performance Highlights**: 하이브리드 접근법은 랜덤 포레스트의 확률 맵을 기반으로 학습한 CNN을 통해 성능을 크게 향상시켰습니다. 해당 방법은 baseline에 비해 7.6%의 recall(0.85), 2.4%의 F1 점수(0.84), 그리고 0.54%의 AUC(0.99) 개선을 달성했습니다. 이러한 결과는 확률적 출력과 공간적 특성 학습을 결합하여 하이퍼스펙트럼 이미지 분석의 정확성을 강화하는 데 성공했다는 것을 보여줍니다.



### EcoWeedNet: A Lightweight and Automated Weed Detection Method for Sustainable Next-Generation Agricultural Consumer Electronics (https://arxiv.org/abs/2502.00205)
- **What's New**: 본 연구에서는 EcoWeedNet이라는 새로운 모델을 제안하여 잡초 탐지 성능을 획기적으로 향상시켰습니다. 이 모델은 복잡한 계산 요구 사항을 추가하지 않고도 로우카본(low-carbon) 농업 실천의 목표에 부합하는 경량화된 솔루션입니다. EcoWeedNet은 기존의 큰 모델에 비해 필요한 파라미터 수가 약 4.21%에 불과하며, 이는 자동 잡초 탐지 방법의 효과적인 개발에 기여하고 차세대 농업 소비 전자 기기에서의 응용 가능성을 높입니다.

- **Technical Details**: 에코잡초넷(EcoWeedNet)은 깊이 학습 모델에서 전통적인 주의(attention) 모듈의 단점을 해결하고 파라미터가 없는 주의 모듈을 도입함으로써 계산 효율성을 유지하면서 성능을 최적화합니다. 이 연구에서는 CottonWeedDet12 벤치마크 데이터셋을 사용하여 실제 환경에서 성능을 테스트해 효율적인 잡초 탐지 능력을 입증하였습니다. 이 모델은 CNN(convolutional neural networks)을 활용하여 고해상도 이미지를 분석하며, 이미지 내 물체 탐지 문제를 효과적으로 정의하는 방식으로 작동합니다.

- **Performance Highlights**: EcoWeedNet는 YOLOv4의 약 6.59%의 GFLOPs로 대규모 모델에 가까운 성능을 보이며, 정확도에서 우수한 결과를 나타냅니다. 연구 결과에 따르면, 제안된 모델은 가벼우면서도 높은 검출 정확도를 제공하여, 지속 가능한 소비자 전자 농업 장비 및 로봇에 적합한 성능을 ₍를₉ 증명하였습니다. 이러한 성과는 차세대 농업 소비 기술의 지속 가능성을 위한 중요한 발판을 마련합니다.



### DermaSynth: Rich Synthetic Image-Text Pairs Using Open Access Dermatology Datasets (https://arxiv.org/abs/2502.00196)
Comments:
          12 pages, 4 figures

- **What's New**: 이번 연구에서는 각종 피부과 임상 과제를 위해 92,020개의 합성 이미지-텍스트 쌍으로 구성된 새로운 데이터셋인 DermaSynth를 소개합니다. 이 데이터셋은 피부과 관련 이미지와 함께 제공되는 텍스트 어노테이션의 부족 문제를 해결하기 위해 개발되었습니다. 임상 관련 프롬프트와 자기 지도(self-instruct) 방법을 활용해 데이터셋을 생성하였으며, 이는 AI 연구에 기여할 것으로 기대됩니다.

- **Technical Details**: DermaSynth 데이터셋은 DERM12345, BCN20000, PAD-UFES-20 등 여러 공개 피부과 데이터셋에서 수집된 이미지들을 바탕으로 합니다. 각 이미지에 대해 일반 질문과 메타데이터 기반 질문을 통해 합성 이미지-텍스트 쌍을 생성하였고, 이 과정에서는 Gemini 2.0 API를 이용했습니다. 이렇게 생성된 데이터는 후처리 과정을 거쳐 임상적으로 관련성이 높고 일관된 스타일을 유지하도록 하였습니다.

- **Performance Highlights**: 프로토타입 모델 DermatoLlama 1.0은 Llama-3.2-11B-Vision-Instruct 모델로, DERM12345 데이터셋에서 5,000개의 이미지-텍스트 쌍을 Fine-tuning하여 개발되었습니다. 이 모델은 Hugging Face에서 접근 가능하며, 연구자들이 피부과 영상을 기반으로 한 LLM의 성능을 평가하는 데 활용될 수 있습니다. DermaSynth와 DermatoLlama 1.0은 모두 비상업적 연구 목적으로 제공되며, 해당 리소스들은 오픈 소스로 제공됩니다.



### Lifting by Gaussians: A Simple, Fast and Flexible Method for 3D Instance Segmentation (https://arxiv.org/abs/2502.00173)
Comments:
          Accepted to WACV 2025

- **What's New**: Lifting By Gaussians (LBG)는 3D Gaussian Splatting Radiance Fields (3DGS)의 오픈 월드 인스턴스 세그멘테이션을 위한 새로운 접근 방식입니다. 이 방법은 기존 3DGS 재구성에 대해 전면적 학습 없이 빠르게 작동할 수 있는 특징을 가지고 있습니다. LBG는 2D 세그멘테이션 마스크와 CLIP 및 DINOv2의 피쳐를 통합하여 3DGS에서 직접 사용합니다.

- **Technical Details**: LBG는 두 가지 입력을 요구합니다: 1) 포즈가 있는 2D 이미지 데이터, 2) 사전 훈련된 3DGS 필드입니다. 이 방법은 2D 세분 모델을 이용하여 각 이미지에서 2D 세그멘테이션 마스크를 추출하고, unique object ID를 Gaussians에 할당하여 3D에서 개체 조각을 생성합니다. 최종적으로, incremental merging 방식으로 이러한 조각들을 통합하여 장면 수준 객체를 생성합니다.

- **Performance Highlights**: LBG는 기존 방법보다 속도가 빠르고 단순하며, 구성 가능성이 높습니다. 실험에 따르면, LBG는 세그멘테이션 성능과 품질 면에서 우수한 결과를 보이며, 특히 대규모 3DGS 필드에서 기존의 방법들과 비교했을 때 더 빠른 처리를 가능하게 합니다. 새롭게 제안된 평가 프로토콜을 통해 3D 자산의 렌더링 품질 평가에서 높아진 정확성을 확인할 수 있습니다.



### ALBAR: Adversarial Learning approach to mitigate Biases in Action Recognition (https://arxiv.org/abs/2502.00156)
Comments:
          Accepted to ICLR 2025

- **What's New**: 이 연구는 행동 인식(action recognition)에서 발생하는 배경(bias)과 전경(bias)에 대한 편향을 완화하기 위한 새로운 적대적 훈련(adversarial training) 방법인 ALBAR를 제안합니다. 기존의 접근 방식은 주로 배경 편향을 완화하는 데 집중했지만, 이 논문에서는 두 가지 유형의 편향을 모두 탐구하고 이를 효과적으로 해결할 수 있는 방법을 제공합니다.

- **Technical Details**: ALBAR는 static clip(정적 클립)에서 샘플링된 데이터를 기반으로 적대적 교차 엔트로피 손실(adversarial cross-entropy loss)을 적용하여 클래스 확률을 균일하게 만들고, 엔트로피 최대화 손실(entropy maximization loss)을 통해 static bias(정적 편향)를 완화합니다. 또한, 이 과정에서 gradient penalty loss(그래디언트 패널티 손실)를 추가하여 디바이싱(debiasing) 과정을 정규화합니다. 이러한 접근 방식은 편향 속성에 대한 전문 지식 없이도 작동하도록 디자인되었습니다.

- **Performance Highlights**: HMDB51 데이터셋에 대한 평가에서 ALBAR는 이전의 상태를 뛰어넘는 성능을 보여주며, 전체 정확도가 12% 이상 향상되었습니다. 또한, UCF101 프로토콜에서의 기존 배경 정보 유출 문제를 확인하고, 더 정밀한 분할 경계(segmentation boundaries)를 제안하여 테스트 세트를 수정함으로써 기존 접근 방식을 능가했습니다.



### Exploring Transfer Learning for Deep Learning Polyp Detection in Colonoscopy Images Using YOLOv8 (https://arxiv.org/abs/2502.00133)
Comments:
          10 pages, 3 figures, 6 tables, SPIE conference

- **What's New**: 이 논문은 전이 학습(transfer learning) 기술을 활용하여 YOLOv8n 모델을 폴립 탐지(polyp detection) 작업에 적용하는 방법을 탐구합니다. 기존의 데이터셋에 대한 전이 학습을 통해 폴립 탐지 성능을 향상시키는 방법을 제시하며, 다양한 데이터셋에서의 전이 학습 효과를 비교하고 분석합니다. 특히, 일반적인 대규모 데이터셋과 폴립과 같은 특정 특성을 가진 소규모 데이터셋 간의 성능 차이를 조사했습니다.

- **Technical Details**: 전이 학습은 특정 작업을 위해 사전 훈련된 모델을 사용하는 기술로, 충분한 데이터가 없거나 비용이 많이 드는 의료 분야에서 특히 유용합니다. 이 연구에서는 YOLOv8n 모델을 여러 개의 데이터셋에서 사전 훈련하고, 이를 폴립 탐지 데이터셋에 맞게 미세 조정(fine-tuning)하여 성능을 평가했습니다. 각 모델은 무작위로 초기화된 모델과 성능을 비교하여 사전 훈련의 이점을 측정했습니다.

- **Performance Highlights**: CVC-Clinic DB, CVC-ColonDB, ETIS-LaribPolypDB, Kvasir-SEG의 네 가지 공공 데이터셋에서 이루어진 실험은 관련 데이터셋에서 사전 훈련된 모델이 일반적인 객체 데이터셋에서 사전 훈련된 모델보다 일관되게 높은 성능을 나타낸다는 결과를 보였습니다. 특히, 사전 훈련이 없는 모델에 비해 사전 훈련된 모델이 더 우수한 결과를 보였고, 이는 폴립 탐지와 같은 제한된 데이터 환경에서 전이 학습의 중요성을 강조합니다.



### ProtoSnap: Prototype Alignment for Cuneiform Signs (https://arxiv.org/abs/2502.00129)
Comments:
          Accepted to ICLR 2025. Project page: this https URL

- **What's New**: 본 논문은 3천 년 이상의 역사적인 중근동에서 지식 전파의 매개체인 설형 문자(cuneiform) 시스템의 내부 구조를 복원하는 새로운 비지도 접근법인 ProtoSnap을 제안한다. 기존의 자동화 기술들은 문자의 유형을 범주적으로 취급하며, 그 내부 구성의 다양성을 모델링하지 못했으나, 본 연구는 강력한 생성 모델(generative models)과 프로토타입 폰트 이미지의 구조와 외관을 활용하여 이 문제를 해결하고자 한다.

- **Technical Details**: ProtoSnap은 이미지의 심층 특징을 사용하여 설형 문자에서 스켈레톤 기반 프로토타입을 정렬하는 방법으로, 기계적으로 문자의 구조를 추출하고 지역적 일관성을 강제한다. 이 과정에서, 템플릿과 타겟 이미지 간의 쌍별 유사성을 활용하며, 글로벌 및 로컬 정렬을 통한 다단계 과정을 통해 각 문자 구성 요소의 위치를 로컬라이징한다. 결과적으로, 기존의 방법보다 훨씬 높은 효율로 기계 인식 성능을 개선하는 데이터 생성이 가능해진다.

- **Performance Highlights**: ProtoSnap을 사용한 결과, 설형 문자의 구조를 효과적으로 식별할 수 있는 성능을 달성했으며, 특히 드문 기호에서 기존 기술보다 현저하게 개선된 OCR(Optical Character Recognition) 성능을 보여주었다. 또한, 논문은 새로운 전문가 주석 기반 벤치마크를 제공하며, 이 방법이 자동화된 설형 문자 텍스트 디지털화에 유용함을 입증했다. 이로 인해 생성된 합성 데이터는 문자의 정확한 구조적 구성을 가진 상태에서 훈련시킬 수 있다.



### AIN: The Arabic INclusive Large Multimodal Mod (https://arxiv.org/abs/2502.00094)
Comments:
          20 pages, 16 figures, ACL

- **What's New**: 최근의 큰 언어 모델(LLMs)과 다중 모달 모델(LMMs)의 발전 속에서, 아랍어 LMM들은 주목받지 못한 반면, 아랍어 LLMs는 상당한 향상을 보여 왔습니다. 이 격차를 해소하기 위해 AIN(Arabic Inclusive Multimodal Model)을 소개합니다. AIN은 영어-아랍어 이중 언어 LMM으로, 고품질의 360만 개 아랍어-영어 다중 모달 데이터 샘플을 활용하여 설계되었습니다.

- **Technical Details**: AIN 모델은 70억 개의 파라미터를 기반으로 한 아랍어 대형 다중 모달 모델로, 복잡한 추론, 다국적 작업, 이미지-텍스트 정렬에서 우수한 성능을 보입니다. CAMEL-Bench 기준에서 대조군 모델들과 비교하여 AIN-7B는 많은 도메인에서 높은 성과를 자랑하며, Qwen2-VL-7B와 비교해도 성능이 3.43% 향상되었습니다.

- **Performance Highlights**: AIN의 성과는 특히 의료 이미지 해석 및 과학 시각화 이해 등 다양한 분야에서 두드러집니다. 설문 조사 결과, 아랍어 구사자들 사이에서 AIN-7B가 76%의 지지를 받아 대세 모델이 되었으며, 복잡한 시각-언어 과제를 처리하는 데 있어 AIN의 효율성과 정확도가 두드러집니다.



### CerraData-4MM: A multimodal benchmark dataset on Cerrado for land use and land cover classification (https://arxiv.org/abs/2502.00083)
Comments:
          9 pages, 13 Figures, 3 tables

- **What's New**: 본 연구는 Brazil의 Cerrado 지역을 위한 CerraData-4MM이라는 새로운 멀티모달 데이터셋을 소개합니다. 이 데이터셋은 Sentinel-1 Synthetic Aperture Radar (SAR)와 Sentinel-2 MultiSpectral Imagery (MSI)를 결합하고 있으며, 10m의 공간 해상도를 자랑합니다. 이를 통해 7개 및 14개의 계층적 분류 수준을 관리하여 다양한 생태 지역을 대표합니다. 이러한 접근법은 환경 변화에 대한准确한 분석을 위해 필수적입니다.

- **Technical Details**: CerraData-4MM은 심층 학습을 통해 클래스 불균형 및 멀티모달 데이터 융합을 다루는 데 도전적인 벤치마크를 제공합니다. 이 데이터셋은 U-Net과 Vision Transformer (ViT) 모델을 사용하여 성과를 평가했으며, ViT 모델이 더 높은 매크로 F1-score (57.60%)와 평균 교차 비율 (mIoU: 49.05%)을 기록하였습니다. 하지만 두 모델 모두 상대적으로 소수 클래스에 대해 어려움을 겪었고, U-Net의 F1-score는 18.16%로 감소하는 문제를 보여주었습니다.

- **Performance Highlights**: CerraData-4MM의 성과는 모델이 멀티모달 시나리오에 대한 학습 능력이 뛰어남을 보여주며, 이 데이터셋이 심층 학습 혁신을 위한 새로운 기회를 제공함을 강조하고 있습니다. 특히, 클래스 간 균형을 맞추기 위한 시도가 필요하지만, 이는 전반적인 정확도를 감소시키는 단점이 있습니다. 따라서, 이를 통해 클래스 비례 조정의 중요성과 도전 과제를 잘 드러내고 있습니다.



### Influence of color correction on pathology detection in Capsule Endoscopy (https://arxiv.org/abs/2502.00076)
- **What's New**: 이 연구는 Wireless Capsule Endoscopy (WCE)에서 병리 감지에 영향을 미치는 색 보정의 효과를 평가합니다. 기존의 SEE-AI 데이터셋을 바탕으로 두 가지 색 보정 버전을 생성하여 Retinanet과 YOLOv5 모델의 성능을 평가하였습니다. 연구 결과, 색 보정은 모델이 더 큰 바운딩 박스를 생성하고, 특정 병리에 대한 오탐지(False positive) 수를 증가시키는 경향이 있지만, F1-score 및 IoU와 같은 성능 지표에서 일관된 개선이 나타나지 않았습니다.

- **Technical Details**: WCE 데이터셋에서 병리 감지 정확도를 높이기 위해 색 보정 기술을 적용하였습니다. 본 연구는 CC (Color Correction)와 CCC (Colon Color Checker) 두 가지 색 보정 행렬을 활용하여 두 개의 색 보정 데이터셋을 생성하였습니다. 이후, 세 가지 데이터셋(원본 SEE-AI, CCD, CCCD)에 대해 Retinanet과 YOLOv5 모델의 성능을 비교 분석하였습니다.

- **Performance Highlights**: 색 보정 후 모델의 결과는 바운딩 박스의 크기와 실제 주석 간의 교차 영역을 확대하며, 특정 병리에 대해 오탐지 수가 증가하는 것으로 나타났습니다. 하지만, 이러한 변화가 성능 지표의 일관된 개선으로 이어지지는 않았고, 색상 왜곡 문제를 해결하기 위해 향후 연구가 필요하다는 결론을 내렸습니다. 이 연구 결과는 WCE 이미지를 통한 병리 검출의 향상을 도모하며, 관련 데이터셋은 연구자들에게 공개될 예정입니다.



### SpikingRTNH: Spiking Neural Network for 4D Radar Object Detection (https://arxiv.org/abs/2502.00074)
Comments:
          arxiv preprint

- **What's New**: 최근 4D Radar 기술이 자율주행 차량의 3D 객체 탐지에 필수적인 센서로 자리잡고 있습니다. 이 논문에서는 SpikingRTNH라는 최초의 스파이킹 신경망(SNN) 아키텍처를 제안하여 4D Radar 데이터를 이용한 3D 객체 탐지를 수행합니다. 이 접근법은 conventional ReLU 활성화 함수를 leaky integrate-and-fire (LIF) 스파이킹 뉴런으로 대체하여 에너지 효율성을 극대화합니다.

- **Technical Details**: SpikingRTNH는 포인트 클라우드를 높은 밀도에서 낮은 밀도로 순차적으로 처리하는 생물학적 top-down inference (BTI) 방법을 도입합니다. 이 방식은 노이즈가 적고 중요도가 높은 포인트를 효과적으로 감지하는 데 초점을 둡니다. RTNH에서 높은 밀도의 4D Radar 포인트 클라우드를 처리하면서도 약 156G의 곱셈-누적(MAC) 연산을 요구하는 기존 방법들에 비해 현저한 에너지 절감이 가능합니다.

- **Performance Highlights**: K-Radar 데이터셋에서 수행된 실험 결과 SpikingRTNH는 78% 에너지 소비를 줄이면서도 51.1% AP 3D 및 57.0% AP BEV로 기존 인공신경망(ANN) 모델과 유사한 탐지 성능을 달성했습니다. 이러한 결과는 자율주행 시스템을 위한 에너지 효율적인 4D Radar 기반 객체 탐지의 가능성을 입증하고 있습니다. 연구에 사용된 모든 코드는 제공된 링크에서 확인할 수 있습니다.



### A two-stage dual-task learning strategy for early prediction of pathological complete response to neoadjuvant chemotherapy for breast cancer using dynamic contrast-enhanced magnetic resonance images (https://arxiv.org/abs/2502.00051)
- **What's New**: 이 논문은 유방암 환자의 병리학적 완전 반응(pCR)을 조기에 예측하기 위한 새로운 접근 방식을 제안합니다. 기존의 예측 방식과 달리, 이 연구에서는 neoadjuvant chemotherapy 초기 단계에 대한 정확도를 개선하기 위해 두 단계의 이중 과제 학습(dual-task learning) 전략을 사용했습니다. 이를 통해 치료 초기의 MRI 이미지를 통해 pCR을 조기에 예측할 수 있는 가능성을 제시합니다.

- **Technical Details**: 연구팀은 넷워크를 훈련시키기 위해 I-SPY2 임상 시험에서 얻은 동적 조영증강 자기공명영상(Dynamic Contrast-Enhanced MRI) 데이터를 활용했습니다. 첫 번째 단계에서는 convolutional long short-term memory 네트워크를 통해 T2 시점의 pCR 예측 및 잠재적인 이미지 특징(latent space image features)을 추출했으며, 두 번째 단계에서는 T0 및 T1의 이미지를 통해 동시에 pCR 및 T2의 이미지 특징을 예측하는 이중 과제 네트워크를 훈련시켰습니다.

- **Performance Highlights**: 기존의 단일 단계 단일 과제 전략을 적용했을 때, pCR 예측의 AUROC 값은 0.799였습니다. 그러나 제안된 두 단계의 이중 과제 학습 전략을 활용했을 때, AUROC 값이 0.820으로 향상되었습니다. 이는 초기 neoadjuvant chemotherapy 단계에서 pCR을 예측하는 모델 성능을显著하게 개선할 수 있음을 보여줍니다.



### VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos (https://arxiv.org/abs/2502.01549)
- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG) 개념을 비디오 콘텐츠에 처음으로 적용한 VideoRAG 프레임워크를 소개합니다. 이는 복잡한 멀티모달 비디오 지식을 처리하고 이해하기 위해 특별히 설계된 모델로, 기존의 텍스트 기반 접근 방식의 한계를 넘는 혁신적인 접근 방식입니다. VideoRAG는 두 개의 상호 연결된 구성 요소인 Multi-Modal Video Knowledge Indexing framework와 Knowledge-Grounded Multi-Modal Retrieval paradigm을 통해 멀티모달 비디오 내용을 효과적으로 조직하고 인덱싱할 수 있게 합니다.

- **Technical Details**: VideoRAG의 핵심 혁신은 그래프 기반의 텍스트 지식 기초와 멀티모달 컨텍스트 인코딩을 통합한 이중 채널 아키텍처에 있습니다. 이 프레임워크는 다양한 비주얼 특징을 효율적으로 보존하고, 여러 비디오에 걸쳐 정확한 지식 그래프를 구축하여 비디오 간의 의미적 의존성을 유지합니다. 또한, 비디오 지식 검색을 효율적으로 수행하기 위해 LLM 기반의 키워드 추출과 비전-언어 모델을 기반으로 한 텍스트 그라운딩을 결합한 이중 단계의 콘텐츠 추출 프로세스를 활용합니다.

- **Performance Highlights**: VideoRAG는 160개 이상의 비디오로 구성된 LongerVideos 벤치마크에서 종합적인 실증 평가를 통해 기존 RAG 대안 및 긴 비디오 이해 방법과 비교해 상당한 성과를 보여줍니다. 이 프레임워크는 교육 콘텐츠 분석, 미디어 아카이빙, 비디오 기반 지식 추출과 같은 분야에서 비디오 이해력을 크게 향상시키며, 기존 단일 비디오에 제한된 기존 데이터셋을 넘어서는 지식을 제시합니다. 실험 결과와 사례 연구를 통해 VideoRAG의 실질적인 응용 가능성을 밝혀내며, 비디오 간 이해 향상에 기여하는 새로운 가능성을 열었습니다.



### mWhisper-Flamingo for Multilingual Audio-Visual Noise-Robust Speech Recognition (https://arxiv.org/abs/2502.01547)
- **What's New**: mWhisper-Flamingo는 다양한 언어로 된 오디오-비주얼 음성 인식(Audio-Visual Speech Recognition, AVSR)을 위한 새로운 모델로, 사전 훈련된 오디오 모델(Whisper)과 비디오 모델(AV-HuBERT)의 장점을 결합합니다. 기존의 Whisper-Flamingo 모델이 영어 데이터에 국한되었던 것에 반해, mWhisper-Flamingo는 9개 언어로 된 비디오를 처리할 수 있는 능력을 갖추고 있습니다. 본 연구에서는 새로운 decoder modality dropout 기법을 도입하여, 쌍으로 된 오디오-비주얼 입력뿐만 아니라 독립적인 오디오/비디오 입력으로 훈련하여 성능을 향상시킵니다.

- **Technical Details**: mWhisper-Flamingo는 Whisper-Flamingo 아키텍처를 기반으로 하며, AV-HuBERT 비디오 인코더가 다국어 비디오에 맞게 사전 훈련됩니다. 학습 과정은 두 단계로 나뉘며, 첫 번째로 Whisper의 모든 매개변수를 오디오 입력을 기반으로 미세 조정하여 특정 도메인 성능과 노이즈 강인성을 높입니다. 두 번째로, 시각적 특성을 통합하기 위해 오디오-비주얼 데이터로 gated cross-attention 레이어를 훈련하여, 독립적으로 오디오와 비주얼을 처리하고 최종적으로 일관된 모델 통합을 이루는 방식입니다.

- **Performance Highlights**: mWhisper-Flamingo는 MuAViC 데이터셋에서 다국어 AVSR 성능을 극대화하였고, 깨끗한 오디오 환경에서 이전의 오디오-비주얼 방법보다 뛰어난 성능을 기록했습니다. 특히, 노이즈 환경에서도 6가지 서로 다른 노이즈 유형과 5단계의 노이즈 수준에서 오디오 전용 Whisper 모델을 지속적으로 초과하는 성능을 보여주었습니다. 연구진은 성능 향상이 중요한 다국어 음성 인식 분야에 기여할 것이라 전망하고 있습니다.



### VR-Robo: A Real-to-Sim-to-Real Framework for Visual Robot Navigation and Locomotion (https://arxiv.org/abs/2502.01536)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문은 로봇의 이동 및 내비게이션을 위한 새로운 Real-to-Sim-to-Real 프레임워크인 VR-Robo를 제안합니다. 이 프레임워크는 RGB 기반의 인식을 필요로 하는 고급 과제를 지원하는 포토리얼리스틱(photorealistic)이고 물리적으로 상호작용할 수 있는 디지털 트윈(digital twin) 환경을 생성합니다. VR-Robo는 3D Gaussian Splatting(3DGS)을 활용해 다중 시점 이미지를 바탕으로 장면을 재구성하고, 실시간 시뮬레이션을 통해 로봇 정책을 학습합니다. 이 접근 방식은 기존 시뮬레이터의 한계를 극복하고, 복잡한 환경에서의 로봇 정책을 보다 효과적으로 적응시키는 데 기여합니다.

- **Technical Details**: VR-Robo 프레임워크는 복잡한 지형에서의 물리적 상호작용을 지원하는 혁신적인 시뮬레이션 환경을 제공합니다. 3DGS 기반의 장면 재구성을 통해 다양한 시나리오에서 로봇의 학습이 가능하도록 하여 Sim-to-Real 간의 간극을 최소화합니다. 이 프레임워크는 에이전트-객체 랜덤화 및 가림 인식(scene composition) 전략을 도입하여 RL 정책 학습의 견고성을 크게 향상시킵니다. 두 단계의 훈련 전략을 통해 로봇은 환경을 탐색하고 목표 지점을 식별하며, 정책을 업데이트하는 능력을 갖추게 됩니다.

- **Performance Highlights**: 광범위한 실험을 통해, VR-Robo는 고급 내비게이션 및 이동 정책이 RGB 관찰을 통해 실세계로 제로-샷(zero-shot) 전이 가능하다는 것을 보여주었습니다. 이 연구는 집이나 공장과 같은 복잡한 환경에서 로봇 정책의 빠른 적응이 가능함을 강조하며, 실제 적용 가능성을 넓히고 있습니다. 이 결과는 VR-Robo가 새로운 시뮬레이션 방법론을 통해 안정성과 효율성을 모두 갖춘 로봇 학습을 가능하게 한다는 점에서 중요한 의미를 갖습니다.



### Structural features of the fly olfactory circuit mitigate the stability-plasticity dilemma in continual learning (https://arxiv.org/abs/2502.01427)
- **What's New**: 이 연구는 지속적인 학습(continual learning)에서 인공신경망이 직면하는 안정성-가소성 딜레마(stability-plasticity dilemma)를 해결하기 위해 최소한의 모델로서 파리의 후각 회로(fly olfactory circuit)를 제시합니다. 이 모델은 현대 기계 학습 방법과 통합할 수 있는 플러그 앤 플레이 구성 요소로 알려져 있으며, 지속적인 냄새 학습을 지원하는 생물학적 전략을 조사합니다. 또한 이 연구는 현재의 지속적 학습 전략의 한계를 극복할 수 있는 생물학적 솔루션을 제공합니다.

- **Technical Details**: 파리의 후각 회로는 생물학적 네트워크의 효율성을 기반으로 한 학습 모듈로서 기능하며, 이를 통해 메모리의 안정성(memory stability)과 학습의 가소성(plasticity)을 동시에 향상시킬 수 있음을 보여주었습니다. 이 모델은 다양한 도전적인 지속적 학습 시나리오에 걸쳐 검증되었으며, 일반적으로 사용되는 데이터 세트에서 효과성을 입증하였습니다. 결과적으로, 이 모델은 기계 학습에서 추가적인 계산 비용을 최소화하면서 지속적 학습을 향상시키는 모듈로 작용합니다.

- **Performance Highlights**: Fly Model은 기존의 지속적 학습 전략들을 초월하여 우수한 성능을 나타냈습니다. 연구 결과, 이 모델은 학습 효율을 높이고 안정성을 제공함으로써 데이터에 대한 적응력을 증가시킵니다. 이는 생물학적 회로가 평생 학습(lifelong learning)에서 어떻게 활용될 수 있는지를 보여주는 우아한 사례가 됩니다.



### Assessing the use of Diffusion models for motion artifact correction in brain MRI (https://arxiv.org/abs/2502.01418)
Comments:
          Accepted at IEEE International Symposium for Biomedical Imaging (ISBI) 2025

- **What's New**: 이 논문에서는 2D 뇌 MRI 스캔에서 모션 아티팩트를 수정하는 데 있어 확산 모델(diffusion model)의 효과를 평가합니다. 기존의 방법인 U-Net과 비교하여, 확산 모델이 모션 아티팩트 수정에서 어떻게 성능을 발휘하는지를 살펴보았습니다. 연구 결과, 확산 모델이 정확한 예측을 수행할 수도 있지만, 환각(hallucination)의 위험이 존재함을 확인했습니다.

- **Technical Details**: 연구에서는 두 가지 접근 방식을 비교했습니다. 첫 번째는 합성적으로 생성된 모션 영향을 받은 이미지를 사용한 감독 학습(supervised learning) 기반의 U-Net이며, 두 번째는 모션이 없는 이미지만으로 훈련된 비감독(unconditional) 확산 모델(Denoising Diffusion Probabilistic Model, DDPM)입니다. DDPM은 노이즈 확산 과정(difffusion process)을 역전시켜 고품질 샘플을 생성하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 확산 모델은 단일 이미지를 기반으로 한 경우에 비해 성능이 달라지며, NMSE와 PSNR 성능에서 우수한 결과를 보였습니다. 그러나 U-Net 구조는 SSIM 성능에서 가장 높았고, 전반적으로 UNet Synth 모델은 NMSE 및 PSNR에서는 확산 모델에 약간 뒤처졌으나 SSIM 측면에서는 근접한 성과를 보였습니다. 이러한 결과는 환자가 움직일 때 MRI 이미지의 품질를 향상시키기 위한 확산 모델의 잠재력을 시사합니다.



### Learning Traffic Anomalies from Generative Models on Real-Time Observations (https://arxiv.org/abs/2502.01391)
- **What's New**: 이번 연구에서는 Spatiotemporal Generative Adversarial Network (STGAN) 프레임워크를 사용하여 도시 교통 시스템에서의 이상 감지 정확성을 개선합니다. Graph Neural Networks (GNNs)와 Long Short-Term Memory (LSTM) 네트워크를 결합하여 교통 데이터의 복잡한 공간 및 시간적 의존성을 포착합니다. 연구는 스웨덴 예테보리의 42개 교통 카메라에서 수집된 실시간 데이터를 기반으로 하여 진행되었습니다.

- **Technical Details**: STGAN은 노드로서 교통 카메라를, 엣지로서 도로를 나타내는 동적 디지털 트윈 기법을 사용하여 도시 교통 네트워크를 모델링합니다. 이 프레임워크에서는 무게가 부여된 그래프를 통해 교통 정황을 나타내고, 각 노드 간의 인접성을 기반으로 공간적 상관관계를 정의합니다. 이를 통해 단기적 및 장기적 교통 패턴을 효율적으로 캡처하여 이상을 감지할 수 있습니다.

- **Performance Highlights**: 모델은 높은 정밀도로 교통 이상을 효과적으로 감지하며, 낮은 허위 긍정률을 기록했습니다. 이상 감지 사례에는 카메라 신호 중단, 비주얼 아티팩트 및 극한 기상 조건이 포함되었습니다. 연구 결과는 최근 딥러닝을 활용한 교통 예측의 효용성을 강조하며, 전통적인 방법으로는 감지하기 어려운 복잡한 교통 패턴을 포착하는 데 기여합니다.



### Detecting Backdoor Samples in Contrastive Language Image Pretraining (https://arxiv.org/abs/2502.01385)
Comments:
          ICLR2025

- **What's New**: 이번 연구에서는 CLIP(Contrastive Language-Image Pretraining) 모델의 약점인 백도어 공격(backdoor attack)에 대한 새로운 분석을 제공합니다. 연구 결과, 훈련 데이터의 0.01%만 오염시켜도 공격 성공률이 거의 완벽하게 달성된다는 사실을 발견했습니다. 이는 CLIP을 이용한 대규모 모델 훈련에서 웹 데이터를 검증 없이 사용하는 것에 대한 보안 우려를 제기합니다.

- **Technical Details**: 우리는 CLIP 모델이 학습한 백도어 오염 샘플의 표현을 분석한 결과, 이 샘플들이 지역적 하위공간(local subspace)에서 독특한 특성을 가지며, 클린 샘플(clean samples)의 이웃보다 훨씬 더 희소(sparse)한 지역적 특성을 나타냄을 발견했습니다. 이 특징을 바탕으로, 우리는 CLIP 백도어 공격 탐지에 대한 체계적인 연구를 수행하였고, 기존의 백도어 샘플 탐지 기법들이 실패하는 반면, 전통적인 밀도비(density ratio) 기반의 지역 이상 탐지기(local outlier detector)가 이러한 공격을 쉽게 탐지할 수 있음을 입증했습니다.

- **Performance Highlights**: 실험에서는 원본 CC3M 데이터셋에 이미 의도치 않은 백도어가 존재하고, 이로 인해 OpenCLIP에서 공개된 인기 있는 오픈소스 모델이 영향을 받았음을 발견했습니다. 또한 우리의 탐지기를 사용하면, 4개의 Nvidia A100 GPU로 15분 이내에 백도어 공격이 있는 웹 데이터셋(예: CC3M)을 효과적으로 정리할 수 있습니다. 연구 결과와 코드는 우리의 GitHub 리포지토리에 공개되어 있습니다.



### Inverse Bridge Matching Distillation (https://arxiv.org/abs/2502.01362)
- **What's New**: 이번 논문에서는 Diffusion Bridge Models (DBMs)의 추론 속도를 개선하기 위한 새로운 증류(distillation) 기법을 제안합니다. 기존 방식들과 달리, 제안된 방법은 조건부 및 비조건부 DBMs 모두에 적용 가능하며 단일 단계 생성기도 증류할 수 있습니다. 또한, 특정 데이터 세트 없이 손상된 이미지만을 사용하여 훈련할 수 있는 장점을 제공합니다. 이러한 접근 방식은 DBMs의 추론 시간을 4배에서 100배까지 가속화하며, 특정 환경에 따라선 더 나은 생성 품질을 제공합니다.

- **Technical Details**: DBMs는 이미지와 같은 데이터 간의 변환을 위해 개발된 새로운 형태의 확산 모델로, 노이즈를 데이터에 직접적으로 매핑하는 방식이 아닌 두 데이터 분포 간의 확산 과정을 구성합니다. 이 논문에서는 역 다리 일치(inverse bridge matching) 문제를 기반으로 한 증류 기법을 사용하여, 조건부 및 비조건부 DBMs 각각에 대해 효과적으로 동작하는 보편적 증류 방법을 제안합니다. 또한, 이는 다단계 및 단일 단계 생성기로 DBMs를 변환할 수 있도록 하여, 효율성을 높입니다.

- **Performance Highlights**: 제안된 증류 기법은 여러 이미지 처리 작업에서 테스트되었으며, 이전의 가속화 접근 방식보다 향상된 성능을 보여주었습니다. 특히, DBMs의 비조건부 버전에서의 증류가 최초로 적용되었고, 다양한 현업 문제에서 우수한 결과를 낳았습니다. 이러한 기술은 전체적인 모델의 품질을 향상시키는 데 기여하며, 더 빠르고 실용적인 DBMs의 개발을 가능하게 합니다.



### Diffusion at Absolute Zero: Langevin Sampling Using Successive Moreau Envelopes (https://arxiv.org/abs/2502.01358)
- **What's New**: 이 논문에서는 포텐셜 U(x)에 기반한 Gibbs 분포에서 샘플링하는 새로운 방법을 제안합니다. 이 방법은 확산 모델에 영감을 받아 미래의 분포 πt(x)를 정의하여 점진적으로 복잡한 타겟 분포로 샘플을 유도합니다. 이를 통해 기존 샘플링 기술의 효과를 향상시키고 다중 모드 밀도에 효과적으로 적용할 수 있습니다.

- **Technical Details**: 제안된 방법론은 랑젤빈 샘플링 기술을 기반으로 하며, 잠재적인 U(x)가 비강체(stongly convex) 또는 미분 불가능한 경우에도 적용할 수 있습니다. 또한, 제안된 방법은 에너지를 점진적으로 한층 높은 상태로 유도하기 위해 Moreau envelop을 사용하여 G(x)를 근사합니다. 이 과정에서 Annealed Langevin 샘플링 방식으로 순차적으로 샘플링을 진행합니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법은 수렴 속도를 크게 개선하고 다양한 밀도 분포에 대한 적용 가능성을 보여줍니다. 특히, 대칭적이지 않은 분포나 다중 모드를 가진 복잡한 데이터 세트를 다룰 때 효과적인 성능을 나타내며, 샘플링의 안정성과 개선된 효율성을 강조합니다.



### Deep generative computed perfusion-deficit mapping of ischaemic strok (https://arxiv.org/abs/2502.01334)
- **What's New**: 이 연구는 허혈성 뇌졸중에서의 국소 결손(focal deficits)을 이전의 전통적인 예측 방법과는 다른 방식으로 접근합니다. 기존의 손상(partial lesions)이 아닌, 손상 전의 perfusion(관류) 패턴을 분석하여 더 빈번하고 빠른 예측을 가능하게 합니다. 이러한 접근법은 일반적으로 사용되는 CT 혈관조영(CT angiography)에서 유래된 관류 맵을 통해 이루어집니다.

- **Technical Details**: 연구팀은 1,393명의 급성 허혈성 뇌졸중 환자의 CT 혈관조영에서 계산된 관류 맵을 분석하며, 깊이 생성 추론(deep generative inference) 기술을 사용하여 NIHSS 하위 점수의 신경 기초(neural substrates)를 정위화(localise)합니다. 연구는 주요 손상 정보를 알지 못한 상태에서도 이미 알려진 손상-결손 관계를 재현하며, 새로운 신경 의존성(neural dependents)도 발견합니다.

- **Performance Highlights**: 뛰어난 해부학적 충실도가 확보된 결과, CT 혈관조영을 통해 파생된 관류 맵은 급성 뇌졸중의 풍부한 표현에서 임상 및 과학적으로 중요한 가치를 가질 수 있음을 시사합니다. 연구진은 단순히 급성 이미지를 사용하여 허혈성 뇌졸중의 기능적 해부학적 관계를 고도로 표현할 수 있는 모델을 구동할 수 있음을 밝혀냈습니다.



### A Framework for Double-Blind Federated Adaptation of Foundation Models (https://arxiv.org/abs/2502.01289)
- **What's New**: 본 연구는 기본 모델의 이중 맹인의 연합 적응(double-blind federated adaptation)을 위한 프레임워크를 제안합니다. 이는 전통적인 연합 학습 구조와 보안 암호화 방식인 완전 동형 암호화(fully homomorphic encryption)를 결합하여 데이터 소유자가 데이터를 공유하는 대신 모델을 안전하게 학습할 수 있도록 합니다. 이러한 접근 방식은 기존의 모델이나 데이터를 외부에 노출시키지 않으면서도 다운스트림 데이터에 맞게 모델을 최적화할 수 있는 가능성을 열어줍니다.

- **Technical Details**: 이 프레임워크는 먼저 기본 모델을 지식 증류(knowledge distillation)를 통해 FHE 친화적인 블록 시퀀스로 분해합니다. 이후, 저차원 평행 어댑터(low-rank parallel adapters)를 활용하여 데이터 소유자가 데이터를 직접적으로 접근하지 않고도 다운스트림 작업에 맞춰 모델을 적응시킵니다. 이 과정에서 중간 표현( intermediate representations)을 안전하게 공유하고, 이를 위한 개인 정보 보호(Personal Data Protection)를 위한 변환 방식을 설계하여 모델 추출 공격(model extraction attacks)을 막습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 네 가지 데이터 세트에서 실제적으로 적용 가능하다는 것을 입증했습니다. 특히, 연합 학습(contextual federated learning) 설정에서 하위 적응 모델이 데이터 소유자의 개인 정보를 보호하면서도 성능을 유지하는 데 효과적임을 보였습니다. 이러한 연구 결과는 연합 학습을 활용한 모델 적응의 새로운 가능성을 제시하고 있습니다.



### Learnable polynomial, trigonometric, and tropical activations (https://arxiv.org/abs/2502.01247)
- **What's New**: 이번 연구는 Orthogonal function bases와 Tropical polynomials를 기반으로 하는 학습 가능한 활성화 함수(activation functions)를 가진 확장 가능한 신경망(neural networks)을 조사합니다. 이는 ImageNet-1K 분류(classification)와 OpenWebText의 다음 토큰 예측(next token prediction)을 목표로 하고 있습니다. 기존 활성화 함수인 ReLU와 달리, 학습 가능한 활성화 함수는 훈련 과정에서 네트워크가 동적으로 적응할 수 있도록 합니다.

- **Technical Details**: 연구에서는 깊은 네트워크(deep networks)에서 발생할 수 있는 소실(vanishing) 및 폭발(exploding) 기울기(gradient) 문제를 해결하기 위해, 변동성(variance) 관리를 개선하는 초기화(initiation) 방안을 제안합니다. 이 방법은 변환기(transformers)와 합성곱 네트워크(convolutional networks)에서 단독으로 단위 변동성을 보장할 수 있으며, 이는 깊은 구조에서도 안정적인 기울기 흐름을 보장합니다.

- **Performance Highlights**: 실험 결과, Hermite, Fourier, Tropical 기반의 학습 가능한 활성화 함수를 사용하는 네트워크가 GPT-2 및 ConvNeXt 네트워크보다 훈련과 테스트 정확도(accuracy) 및 혼란도(perplexity)에서 유의미한 향상을 보이는 것으로 나타났습니다. 이러한 연구 결과는 대규모 작업에서 학습 가능한 활성화 함수의 실현 가능성을 강조합니다.



### Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents (https://arxiv.org/abs/2502.01218)
- **What's New**: 이번 논문은 Action Temporal Coherence Learning (AcTOL)이라는 새로운 방법을 제안합니다. AcTOL은 비디오 내의 동작을 고정된 목표 기반 제약 없이 순서적이고 연속적인 비전-언어 표현을 학습하는 것을 목표로 합니다. 기존의 방법들이 미래 프레임에 치우쳐 오류를 발생시키는 문제를 해결하기 위해, AcTOL은 프레임 간의 의미적 차이를 대조하고 매끄러운 전이를 보장하는 로컬 브라운 운동 조항을 도입합니다.

- **Technical Details**: AcTOL은 비디오를 연속적인 궤적으로 간주하며, (1) 프레임 간의 의미적 차이를 대조하여 자연스러운 순서를 반영하고, (2) 비디오 전반에 걸쳐 일관된 비주얼 표현을 확보하도록 합니다. Vision-Language Ordering (VLO) 손실을 도입하여 프레임 간의 상대적 시간 거리에 따라 중첩된 의미적 정렬을 보장하고, 브라운 운동 과정을 사용하여 비주얼 표현의 연속성과 안정성을 확보합니다.

- **Performance Highlights**: 다양한 시뮬레이션 환경에서 수행된 실험 결과, AcTOL은 이전 방법들에 비해 최대 49% 성능을 향상시켰습니다. 또한, AcTOL은 여러 실제 동작 비디오에서 언어 조건에 따른 비주얼 보상을 생성하여 명시된 지침과 잘 정렬되는 밀집 보상을 생성하는 데 성공했습니다.



### Land Surface Temperature Super-Resolution with a Scale-Invariance-Free Neural Approach: Application to MODIS (https://arxiv.org/abs/2502.01204)
- **What's New**: 이 논문은 전통적인 낮은 해상도에서 훈련된 모델을 다시 높은 해상도로 적용하는 데 따른 문제를 해결하기 위해 Scale-Invariance-Free 접근 방식을 도입했습니다. 두 종류의 Convolutional Neural Network (CNN) 모델인 SIF-CNN-SR1과 SIF-CNN-SR2를 사용하여 MODIS Land Surface Temperature (LST) 제품의 초해상도를 구현했습니다.

- **Technical Details**: Scale-Invariance-Free 접근 방식은 높은 공간 해상도에서 LST 맵을 제공하도록 모델을 훈련하고, 저 해상도에서 복원될 때 초기 LST를 회복하며, 높은 해상도의 NDVI로 informed된 세밀한 텍스처를 포함합니다. 이 연구는 MODIS와 함께 사용할 수 있는 ASTER LST 이미지의 테스트 데이터베이스를 공개하여 초해상도 알고리즘의 평가에 유용하게 활용할 수 있습니다.

- **Performance Highlights**: SIF-CNN-SR1은 LPIPS와 Fourier 공간 메트릭으로 평가한 결과, 최신 기술들과 다른 CNN 모델들보다 뛰어난 성능을 보였습니다. 이 결과와 제공된 ASTER-MODIS 데이터베이스는 LST 초해상도 연구의 미래에 대한 기대감을 높입니다.



### Compressed Image Generation with Denoising Diffusion Codebook Models (https://arxiv.org/abs/2502.01189)
Comments:
          Code and demo are available at this https URL

- **What's New**: 새롭게 제안된 Denoising Diffusion Codebook Model (DDCM)은 Denoising Diffusion Models (DDMs)을 기반으로 한 생성적 접근법으로, 고품질 이미지를 생성함과 동시에 손실 없이 압축된 비트스트림 표현도 생성합니다. 이 과정은 표준 Gaussian 노이즈 샘플링을 미리 정의된 코드북에서의 고정된 iid Gaussian 벡터 샘플로 대체함으로써 이루어집니다. DDCM은 극도로 작은 코드북을 사용하더라도 표준 DDM의 샘플 품질과 다양성을 유지하는 것으로 나타났습니다.

- **Technical Details**: DDCM은 고정된 구조의 코드북을 통한 샘플링을 사용하여, 이전의 무한한 표현 공간의 부족함을 해결합니다. 이론적으로, 1000개의 샘플링 단계로 이루어진 확률적 확산 생성 프로세스는 고정된 노이즈 선택을 통해 무한한 다양한 결과(2^1000)를 생성할 수 있습니다. DDCM은 개별 이미지에 가장 적합한 노이즈를 선택하여 생성 모델을 손실 압축 이미지 코덱으로 전환하며, 수학적으로는 SDEs에 기반한 점수 기반 근사화와 연결됩니다.

- **Performance Highlights**: DDCM을 통해 생성된 이미지들은 주어진 이미지에 가장 잘 맞는 노이즈를 선택함으로써 압축 과정에서 뛰어난 지각적 품질(perceptual quality)을 달성합니다. DDCM은 다양한 노이즈 선택 규칙을 통해 압축된 조건부 생성 작업을 위한 다재다능한 프레임워크를 제공합니다. 이러한 방식으로, 이미지 복원과 같은 다양한 작업에서도 압축된 비트스트림과 함께 생성된 이미지가 제공됩니다.



### Towards Agile Swarming in Real World: Onboard Relative Localization with Fast Tracking of Active Blinking Markers (https://arxiv.org/abs/2502.01172)
- **What's New**: 이 논문에서는 Active blinking Marker Tracking (AMT)이라는 새로운 온보드 트래킹 접근법을 소개합니다. 이 방법은 여러 로봇 팀의 멀티 드론에서 상대적 로컬라이제이션과 통신을 향상시키며, 실제 환경에서의 강력한 실시간 배치가 가능합니다. AMT는 활성 깜박이는 마커의 미래 모습을 예측하는 데 가중 다항 회귀(weighted polynomial regression)를 사용하여 예측의 불확실성을 고려하여 이 문제를 해결합니다.

- **Technical Details**: AMT 접근법은 정적 알고리즘들이 깜박이는 마커의 간헐적인 출현으로 인해 고속으로 움직이는 경우에서의 문제를 극복하도록 설계되었습니다. 상대적으로 협력하는 드론들의 운동 모델과 제약 조건을 기반으로하여, AMT는 이전 운동 데이터를 융합해 다음 예상 위치를 추정합니다. 이 방식은 CPU 연산량을 줄여주며, 로봇 팀의 각 구성원을 브랜드화할 수 있는 잠재력을 제공합니다.

- **Performance Highlights**: 실험 결과 AMT 접근법은 트래킹 밀도, 정확도 그리고 복잡성에서 최첨단 방법들보다 우수한 성능을 보였습니다. 특히 2m/s와 7.4m/s의 다양한 속도에서, 여전히 효과적인 추적 알고리즘을 제공할 수 있음을 입증했습니다. 이 결과는 서로 밀접하게 협업하는 멀티 로봇 시스템의 상대적 로컬라이제이션에서 AMT의 활용 가능성을 보여줍니다.



### MIND: Modality-Informed Knowledge Distillation Framework for Multimodal Clinical Prediction Tasks (https://arxiv.org/abs/2502.01158)
Comments:
          Published in Transactions on Machine Learning Research (01/2025), this https URL

- **What's New**: 이번 연구에서는 Modality-INformed knowledge Distillation (MIND) 프레임워크를 제안합니다. MIND는 다양한 크기의 심층 신경망 앙상블에서 지식을 전이하여 보다 작은 멀티모달(student) 모델을 생성하는 지식 증류(knowledge distillation) 기반의 멀티모달 모델 압축 방법입니다. 이를 통해 의료와 관련된 멀티모달 데이터셋의 부족한 샘플 문제를 해결하고 모델의 크기를 효과적으로 조정할 수 있습니다.

- **Technical Details**: MIND는 다중 헤드(joint fusion) 모델을 활용하여 임퓨테이션(imputation)이나 마스킹(masking) 과정 없이도 단일 모달 샘플을 처리할 수 있게 설계되었습니다. 교사 모델들은 단일 모달 네트워크로 구성되어 있어, 학생 모델이 다양한 표현(representation)으로부터 학습할 수 있도록 돕습니다. 이 접근법은 고차원의 임상 데이터에도 최적화된 성능을 제공하며, 멀티모달 학습의 균형을 유지하는 데 유용합니다.

- **Performance Highlights**: MIND는 이진 및 다중 레이블 임상 예측 작업에서 시간 시계열 데이터와 흉부 X-ray 이미지에 대해 평가되었습니다. 또한, 비의료 멀티모달 다중 클래스 데이터셋에서도 MIND 프레임워크의 일반화 가능성을 검증하였습니다. 실험 결과, MIND는 다섯 가지 작업에서 작은 멀티모달 네트워크의 성능을 개선하고, 다양한 융합(fusion) 방법 및 멀티모달 아키텍처와 비교했을 때 최첨단 기준선을 초과하는 성과를 보였습니다.



### Learning to Learn Weight Generation via Trajectory Diffusion (https://arxiv.org/abs/2502.01117)
- **What's New**: 본 논문에서는 Lt-Di라는 새로운 방법을 제안합니다. 이 방법은 기존의 diffusion 알고리즘을 메타러닝(meta-learning)과 통합하여 보지 못한 작업이나 도메인에 대한 가중치를 효과적으로 생성할 수 있도록 합니다. 특히, Trajectory Diffusion(경로 확산)을 통해 최적화 과정에서 다른 가중치의 가치도 활용하며, 학습과 추론을 더욱 효율적으로 개선합니다.

- **Technical Details**: Lt-Di는 세 가지 단계인 가중치 준비, 메타 학습, 평가로 구성된 워크플로우를 가지고 있습니다. 가중치 준비 단계에서는 각 작업에 대해 최적화 경로를 구성하고, 최종 가중치가 최적화되도록 주의를 기울입니다. 메타 학습 단계에서는 REPTILE 프레임워크를 사용하여 여러 작업에서 빠르게 학습할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, Lt-Di는 제로샷(zero-shot) 및 몇 샷(few-shot) 학습, 다중 도메인 일반화, 대규모 언어 모델에 대해 높은 정확도를 달성하면서도 계산 오버헤드를 줄이는 성능을 보입니다. 이러한 특징으로 인해, Lt-Di는 다양한 작업에서 높은 성능을 발휘하는 혁신적인 접근법으로 평가됩니다.



### Towards Robust and Generalizable Lensless Imaging with Modular Learned Reconstruction (https://arxiv.org/abs/2502.01102)
Comments:
          16 pages

- **What's New**: 본 논문은 렌즈 없는 카메라의 새로운 접근 방식을 제안합니다. 이는 전통적인 렌즈 기반 설계를 우회하여 얇은 마스크를 사용함으로써 치수를 축소하고 비용을 절감하며 시각적 프라이버시를 향상시킵니다. 연구진은 이미지 복구 이전의 단계에서 사전 처리기(pre-processor)의 필요성을 이론적으로 입증하고, 다양한 렌즈 없는 이미징 접근 방식에서의 효과를 실험적으로 보여줍니다.

- **Technical Details**: 논문에서는 모듈식 재구성(modular reconstruction) 프레임워크를 제안하며, 이는 다수의 이미지 시스템 및 기존의 카메라 인버전 방법에 적용됩니다. 특히, 입력 노이즈를 증폭시키고 필연적인 모델 불일치로 인한 에러 항을 도입하는 카메라 인버전 방법에 대해 실험적으로 향상된 견고성을 보여줍니다. 더 나아가, 이 프레임워크를 통해 이전에 학습된 구성 요소를 활용하여 새로운 시스템에서의 전이 학습(transfer learning)을 가능하게 하였습니다.

- **Performance Highlights**: 이 연구는 여러 마스크 패턴과 유형을 아우르는 최초의 기준 평가를 수행하여 한 시스템에서 학습된 재구성이 다른 시스템으로 얼마나 잘 일반화되는지를 평가합니다. 또한, DigiCam이라는 프로그래머블 마스크 시스템을 도입하여 비용을 30배 절감했으며, 공공 데이터세트 및 코드, 도구를 오픈 소스하여 품질 높은 렌즈 없는 이미징의 확장성을 높였습니다.



### Enhancing Feature Tracking Reliability for Visual Navigation using Real-Time Safety Filter (https://arxiv.org/abs/2502.01092)
Comments:
          7 pages, 6 figures, Accepted to 2025 IEEE International Conference on Robotics & Automation (ICRA 2025)

- **What's New**: 이 논문은 로봇의 포즈를 로컬라이즈하기 위해 비전 센서를 사용하는 방법을 개선한 내용을 다룹니다. 일반적으로 GPS나 모션 캡처 시스템이 사용되지 않는 환경에서, 시각적 특징을 추적하여 로컬라이제이션을 수행하는데 초점을 맞추었습니다. 제안된 방법은 로봇의 작업 목표와 시각적 정보의 가시성 간의 갈등을 해결하는 새로운 접근 방식인 제약 제어 문제로 접근합니다.

- **Technical Details**: 본 연구에서는 로봇의 기계학습 모델 내에서 가시성 제약의 불변 특성(invariance properties)을 활용하여, 실시간 안전 필터(real-time safety filter)를 제안합니다. 이 필터는 기준 속도 명령(reference velocity command)을 입력으로 받아, 현재 가시적인 특징으로부터 얻는 정보 점수가 사용자 지정 임계값(threshold) 이상을 유지하면서 기준 속도에서 최소한으로 편차가 있는 수정된 속도를 생성합니다. 이는 반복적인 quadratic programming을 기반으로 합니다.

- **Performance Highlights**: 수치 시뮬레이션 결과, 제안된 안전 필터는 불변 조건(invariance condition)을 유지하며 요구하는 최소 수량 이상의 특징의 가시성을 보장하는 것으로 나타났습니다. 또한, 이 필터는 실제 환경에서 시각 동시 로컬라이제이션 및 맵핑(SLAM) 알고리즘에 통합되어, 도전적인 환경에서도 높은 추정 품질을 유지하며 간단한 추적 제어기를 초월하는 성과를 보였습니다.



### Emotional Face-to-Speech (https://arxiv.org/abs/2502.01046)
- **What's New**: 이번 연구는 정서적 얼굴 표정(emotional face-to-speech, eF2S)이라는 새로운 작업을 도입하여 시각적 단서로부터 감정을 파악한 목소리를 생성하려는 최초의 시도를 하였습니다. DEmoFace라는 혁신적인 생성 프레임워크를 제안하여, 다양한 발음 스타일을 정서적 표현과 연결하여 음성을 합성할 수 있는 가능성을 탐구합니다. 또한, 다중 조건부 생성(multi-conditional generation) 시나리오를 효과적으로 처리하기 위해 향상된 무예측 가이드라인(enhanced predictor-free guidance, EPFG)을 개발했습니다.

- **Technical Details**: DEmoFace는 이산 확산 변환기(discrete diffusion transformer, DiT)와 커리큘럼 학습(curriculum learning)을 활용하여 다중 수준의 신경 오디오 코덱(multi-level neural audio codec) 위에 구축되었습니다. 이 프레임워크는 음성과 텍스트의 동적 정렬을 달성하기 위해 다중 모달 DiT 블록(multi-modal DiT blocks)을 제안하고, 얼굴의 감정 및 정체성에 따라 발음 스타일을 맞춤화합니다. 또한, 각기 다른 조건들을 효과적으로 분리하여 다양한 조건부 생성을 가능하게 합니다.

- **Performance Highlights**: 광범위한 실험 결과들은 DEmoFace가 기존 방법들보다 더욱 자연스럽고 일관된 음성을 생성하는 데 성공했음을 입증합니다. 특히, 복잡한 음성 특성과 관련하여 정서적 표현을 효과적으로 결합하며, 이전의 음성 유도 방법조차 초월하는 성과를 보였습니다. 이는 특히 사용자 경험을 향상시키고, 인간-기계 상호작용을 더욱 풍부하게 만드는 데 기여할 수 있습니다.



### UASTHN: Uncertainty-Aware Deep Homography Estimation for UAV Satellite-Thermal Geo-localization (https://arxiv.org/abs/2502.01035)
Comments:
          7 pages, 6 figures, accepted at ICRA 2025

- **What's New**: 이번 연구는 UASTHN(Uncertainty-Aware Satellite-Thermal Homography Network)을 제안하여 Uncertainty Estimation(불확실성 추정) 및 Deep Homography Estimation(DHE) 작업에서의 열화상 기반 지오로컬라이제이션을 개선합니다. 기존의 Thermal Geo-localization(TG) 방법들은 불확실성 측정 기능이 결여되어 있어, 다양한 열화상 이미지 조건에서 시스템의 신뢰성이 떨어지는 문제를 겪고 있었습니다. Crop-based Test-Time Augmentation(CropTTA) 전략을 도입하여 데이터의 불확실성을 효과적으로 측정합니다.

- **Technical Details**: UASTHN 프레임워크는 Deep Homography Estimation(DHE) 모듈과 CropTTA 및 Deep Ensembles(DE)를 활용한 Uncertainty Estimation(UE) 모듈로 구성됩니다. DHE 모듈은 열화상 및 위성 이미지의 사각형 크기를 재조정하여 입력받고, 이들 이미지를 기반으로 네 곳 이탈(displacement)을 계산합니다. 이 과정에서 CropTTA는 컷-out 이미지에서의 호모그래피 합의를 활용하여 불확실성을 측정하고, DE는 모델 불확실성을 평가합니다.

- **Performance Highlights**: 제안된 방법은 도전적인 위성-열화상 데이터셋을 사용하여 7m의 지오로컬라이제이션 오류와 97%의 불확실성 추정 성공률을 달성했습니다. CropTTA를 통해 TG 시스템의 신뢰성과 상황 인식을 개선하며, 데이터와 모델의 불확실성을 종합적으로 평가할 수 있는 능력을 보여줍니다. 구현된 코드와 데이터는 공개되어 있어 다른 연구자들이 활용할 수 있도록 제공됩니다.



### RandLoRA: Full-rank parameter-efficient fine-tuning of large models (https://arxiv.org/abs/2502.00987)
Comments:
          To appear at the International Conference on Learning Representations (ICLR) 2025

- **What's New**: 이 논문에서는 RandLoRA라는 새로운 기법을 소개합니다. 이 방법은 Low-Rank Adaptation (LoRA) 방식의 한계를 극복하며 전체 차원의 업데이트(full-rank updates)를 가능하게 합니다. 특히, 비훈련 가능한 임의의 랜덤 매트릭스(random matrices)의 선형 조합을 학습하여 매개변수 효율성을 높입니다.

- **Technical Details**: RandLoRA는 매개변수의 수를 줄이기 위해 고정된 랜덤 매트릭스에 대각 스케일링 매트릭스(diagonal scaling matrices)를 적용하여 최적화(optimization)합니다. 이러한 방식은 훈련 중에 매개변수(parameter)와 메모리(memory) 효율성을 유지하면서도 낮은 차원의 한계를 극복하게 합니다. 실험은 비전(vision), 언어(language), 비전-언어(vision-language) 벤치마크를 포함하여 다양한 작업에서 수행되었습니다.

- **Performance Highlights**: RandLoRA는 기존의 LoRA 방식과 비교하여 비전 및 언어 작업에서 성능 향상을 보여줍니다. 특히, 비전-언어 작업에서는 성능 격차를 현저히 줄이거나 심지어 없애는 경향을 보였습니다. 이는 RandLoRA의 유효성을 강조하며, 복잡한 작업에서 더 나은 성능을 제공하는 방법임을 증명합니다.



### VL-Nav: Real-time Vision-Language Navigation with Spatial Reasoning (https://arxiv.org/abs/2502.00931)
- **What's New**: 이번 연구에서는 저전력 로봇을 위한 새로운 비전-언어 내비게이션 시스템인 VL-Nav를 제안합니다. VL-Nav는 효율적인 공간 추론을 통해 로봇이 인간의 명령을 이해하고 수행할 수 있도록 설계되었습니다. 기존의 단일 이미지 기반 접근 방식을 넘어, 픽셀 단위의 비전-언어 특성과 탐색 방법론을 결합하여 다양한 환경에서 강인한 내비게이션을 지원합니다.

- **Technical Details**: VL-Nav는 동적 점유 맵에서 전방 기반 목표 지점을 생성하고, 인간의 탐색 패턴을 모방하는 인스턴스 기반 목표 지점을 통합하여 목표 선택을 최적화하였습니다. 새롭게 제안된 휴리스틱 비전-언어(HVL) 공간 추론 기법을 통해 픽셀 단위의 비전-언어 정보를 공간 점수 분포로 변환합니다. 최종적으로 가장 높은 HVL 점수를 가진 목표 지점이 선정되며, 이는 로봇이 인간의 지시와 밀접하게 연관된 지점으로 안내합니다.

- **Performance Highlights**: VL-Nav는 다양한 환경에서 86.3%의 성공률을 달성하며, 이전 방법보다 44.15% 향상된 성능을 보였습니다. Jetson Orin NX 플랫폼에서 30 Hz의 실시간 내비게이션 속도를 유지하여 저전력 환경에서도 효율적으로 작동합니다. 이 연구는 인간과 협력하는 로봇 내비게이션 시스템의 새로운 가능성을 제시합니다.



### Paper Copilot: The Artificial Intelligence and Machine Learning Community Should Adopt a More Transparent and Regulated Peer Review Process (https://arxiv.org/abs/2502.00874)
- **What's New**: 최근 AI(Artificial Intelligence)와 ML(Machine Learning) 학회에 제출되는 논문 수가 급증하면서, 많은 학회들이 비공식 심사에서 공개 심사(open review)로의 전환을 모색하고 있습니다. 본 논문은 Paper Copilot이라는 웹사이트를 분석하여, AI/ML 분야에서 투명한 심사 프로세스에 대한 커뮤니티의 관심이 높아지고 있음을 강조합니다. 이 웹사이트는 177개국의 20만 명 이상의 초기 경력 연구자들의 데이터를 집계하고 분석하여, 리뷰 프로세스에 대한 참여를 촉진하고 있습니다.

- **Technical Details**: 전통적인 심사 관행은 공정성, 효율성 및 품질을 유지하는 데 압박을 받고 있으며, 이에 따라 많은 학회들이 공개 리뷰 플랫폼을 도입하고 있습니다. 연구자들은 평가의 투명성을 높이기 위해 각기 다른 공개 심사 모델을 도입하고 있으며, 완전 공개 리뷰, 부분 공개 리뷰, 비공식 리뷰 등 여러 형태가 존재합니다. 연구에 따르면, 완전 공개 리뷰는 모든 내용을 공개하지만, 개인적인 비판이나 편향이 개입할 수도 있습니다.

- **Performance Highlights**: Paper Copilot은 AI/ML 학회에서의 리뷰 통계를 제공하며, 지난 3~5년간의 리뷰 점수 분포, 리뷰 타임라인 및 저자/소속 분석을 포함합니다. 조사 결과, 연구자들은 투명성과 협력을 요구하고 있으며, 약 3,876건의 유효한 응답을 확보하여 이 정보를 체계적으로 처리하였습니다. 이를 통해, 더 투명하고 규제된 피어 리뷰 프로세스의 도입이 필요하다는 주장을 펼칩니다.



### OOD Detection with immature Models (https://arxiv.org/abs/2502.00820)
Comments:
          17 pages, 2 Tables, 9 Figures

- **What's New**: 이 논문은 Likelihood 기반의 Deep Generative Models (DGMs)이 in-distribution (ID) 데이터보다 out-of-distribution (OOD) 데이터에 대해 더 높은 likelihood 값을 하는 비정상적인 현상을 다루고 있습니다. 저자들은 훈련 초기 단계에서 중단된 모델들을 활용하여 OOD 탐지 성능이 성숙한 모델과 유사하거나 더 뛰어난 결과를 나타날 수 있음을 보여줍니다. 이 연구는 부분적으로 훈련된 모델을 활용하는 새로운 방법론을 제시하며, OOD 탐지에서의 가능성을 강조합니다.

- **Technical Details**: 이 연구에서는 Gradient Norms와 Layer-wise Gradient Norms를 이용하여 OOD 샘플 탐지의 새로운 기준을 제안합니다. 특히, Likelihood에 의한 전통적인 방법들이 한계가 있는 것에 비해, Gradient Norms는 OOD 데이터를 효과적으로 식별하는 데 유용한 것으로 보입니다. 연구에 사용된 GLOW 모델을 통해, 최대 우도에 기반한 훈련 없이도 OOD 탐지가 가능함을 입증하였습니다. 이 모델을 통해 부분적으로 훈련된 모델들이 OOD 탐지 작업에서 효율적으로 동작할 수 있다는 점을 보였습니다.

- **Performance Highlights**: 연구 결과에 따르면, 전체 훈련된 모델보다 부분 훈련된 모델들이 OOD 탐지에서 더 나은 성능을 발휘할 수 있다는 것을 보여주었습니다. 특히, ID 데이터가 복잡하지 않은 경우에 이러한 경향이 두드러졌습니다. 부분 훈련된 모델이 OOD 데이터의 가능성을 명확히 구분할 수 있는 성능을 향상시킨다는 점은 새로운 통찰력을 제공합니다. 이를 통해 OOD 탐지 관련 기법의 효율성을 한층 더 높일 수 있는 기반이 마련되었습니다.



### Vision-centric Token Compression in Large Language Mod (https://arxiv.org/abs/2502.00791)
- **What's New**: 이 논문은 큰 언어 모델(LLM)의 긴 문맥 처리 능력을 개선하기 위한 새로운 방법인 VIST를 제안합니다. 이 방법은 기존의 텍스트 인코더 대신 경량의 비전 인코더를 사용하여 긴 텍스트를 이미지로 변환하고 압축하여 처리합니다. VIST는 기존 방법들보다 16% 적은 FLOPs와 50% 적은 메모리 사용량으로 유사한 성능을 달성하며, 중요한 토큰에 집중할 수 있는 빈도 기반 마스킹 전략을 활용합니다. 이 접근 방식은 결국 LLM의 토큰 효율성을 크게 향상시킵니다.

- **Technical Details**: VIST는 긴 텍스트를 이미지로 변환한 후, 경량 비전 인코더를 사용하여 핵심 정보를 포착합니다. 기존의 텍스트 인코더를 사용했던 방식과 비교했을 때, VIST는 계산 비용이 낮고 더 긴 입력을 효율적으로 처리할 수 있습니다. 특별히, Probability-Informed Visual Enhancement (PVE)를 통해 의미가 풍부한 토큰에 대한 강조를 높이고 성과를 극대화합니다. 기존 LLM의 문제점을 해결하기 위해 VIST는 이미지와 텍스트를 통합하여 장기적인 문맥 정보를 효과적으로 처리합니다.

- **Performance Highlights**: VIST는 TriviaQA, NQ, PopQA, TREF, SST2 및 SST5와 같은 다양한 기준에서 평균 5.7% 성능 향상을 보여줍니다. 텍스트 인코더 기반의 기존 방법들과 비교할 때, VIST는 오픈 도메인 QA 및 ICL 작업에서 우수한 결과를 달성하며, 낮은 계산 비용으로 더 긴 문맥을 처리할 수 있습니다. 이러한 성과는 LLM의 실용적인 적용에서 중요한 돌파구를 마련하며, VIST의 새로운 접근 방식이 큰 영향을 미칠 것임을 시사합니다.



### Privacy Preserving Properties of Vision Classifiers (https://arxiv.org/abs/2502.00760)
- **What's New**: 이번 연구에서는 다양한 아키텍처의 비전 분류기들이 개인정보 보호 특성을 어떻게 나타내는지를 평가합니다. 연구의 핵심은 Multi-Layer Perceptrons (MLPs), Convolutional Neural Networks (CNNs), Vision Transformers (ViTs)와 같은 아키텍처의 차이가 훈련 데이터의 유출에 미치는 영향을 분석하는 것입니다. 모델의 가중치로부터 훈련 데이터를 복원하려는 네트워크 반전 공격을 이용하여, 아키텍처 당 비공식 메모리 정보의 노출 위험을 정량적으로 평가합니다.

- **Technical Details**: 우리는 신경망의 구조, 특성 추출 메커니즘, 그리고 가중치 구조가 개인정보 유출에 미치는 영향을 평가하기 위해 네트워크 반전 기반 복원 기법을 사용합니다. 이를 통해 각 아키텍처가 훈련 데이터를 얼마나 잘 기억하는지를 분석하며, 구조의 차이가 개인정보 보호 특성에 미치는 영향을 심도 있게 조사합니다. 본 연구는 MNIST, FashionMNIST, CIFAR-10, SVHN와 같은 다양한 벤치마크 데이터셋을 활용하여, 맞춤형 아키텍처들이 다양한 데이터 조건에서 어떻게 동작하는지를 연구합니다.

- **Performance Highlights**: 연구 결과, 아키텍처의 차이에 따라 훈련 데이터 유출의 정도가 상당히 다르다는 것을 발견했습니다. CNN과 ViTs는 입력 이미지 처리 및 특성 추출 방식의 차이로 각기 다른 메모리 패턴을 보이며, 그로 인해 침입 공격에 대한 저항력이 변화합니다. 이러한 통찰은 민감한 데이터가 포함된 경우의 안전한 기계 학습 모델 개발에 중요한 기초 데이터를 제공하며, 아키텍처 설계의 중요성을 강조합니다.



### Continuity-Preserving Convolutional Autoencoders for Learning Continuous Latent Dynamical Models from Images (https://arxiv.org/abs/2502.00754)
- **What's New**: 이 논문에서는 연속 동역학 모델을 학습하기 위한 새로운 방법인 연속성을 보존하는 합성곱 오토인코더(CpAE)를 제안합니다. 이전 접근 방식을 개선하기 위해 이미지 데이터를 통해 얻어진 이산 상태에서 연속적인 잠재 상태(latent states)를 효과적으로 학습할 수 있는 수학적 공식화를 제공합니다. 또한, 필터의 연속성을 촉진하여 잠재 상태의 연속성을 보존하는 정규화항을 도입하였습니다.

- **Technical Details**: CpAEs는 이산 이미지 프레임으로부터 연속적으로 진화하는 잠재 상태를 학습하기 위한 딥러닝 기법입니다. 필터가 Lipschitz 연속일 때 잠재 상태가 지속적으로 진화할 수 있는 충분 조건을 제시하며, 연속성을 보존하는 필터를 통해 잠재 상태의 연속성을 유지하는 방법론을 설명합니다. 이론적으로, 이러한 정규화는 모델의 효율성을 극대화하여 지속적으로 변화하는 동역학을 보다 정확하게 학습할 수 있도록 합니다.

- **Performance Highlights**: 다양한 시나리오에서 수행된 실험 결과, CpAE는 기존의 방법들보다 더 정밀하게 잠재 동역학 모델을 생성하는 것으로 나타났습니다. 특히, 복잡한 시각 패턴과 동역학적 행동을 모델링하는 데 있어 향상된 성능을 발휘하며, 미래의 상태 예측에서 더 높은 정확도를 기록했습니다. 이러한 결과는 CpAE가 다양한 과학적 및 공학적 적용에서 동역학 모델링을 위한 강력한 도구가 될 수 있음을 시사합니다.



### An Event-Based Perception Pipeline for a Table Tennis Robo (https://arxiv.org/abs/2502.00749)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 최근 몇 년간 탁구 로봇이 컨트롤 및 인식 알고리즘 연구의 흥미로운 도전 과제가 되었습니다. 본 연구에서는 최초로 이벤트 기반 카메라만을 사용하는 실시간 인식 파이프라인을 제안합니다. 기존의 프레임 기반 카메라에서는 발생할 수 있는 모션 블러를 해결하기 위해 이벤트 기반 카메라를 활용하여 더 빠르고 정확한 공 감지를 가능하게 하였습니다.

- **Technical Details**: 이벤트 기반 카메라는 픽셀 변화에 대해 독립적으로 비동기적으로 정보를 기록하며, 이는 μs 단위의 높은 시간 해상도를 제공합니다. 본 연구에서는 EROS 이벤트 표현 방식을 사용하여 공 감지를 위한 빠른 서클 탐지기를 결합한 인식 파이프라인을 개발했습니다. 인식의 정확성을 높이기 위해 입력 이벤트를 스레드 기반으로 처리하여 낮은 지연 시간과 높은 정보 밀도를 유지합니다.

- **Performance Highlights**: 제안된 인식 파이프라인은 프레임 기반 시스템에 비해 공 위치 추정치를 10배 더 많은 업데이트를 제공하며, 이는 공의 위치, 속도와 회전 예측의 불확실성을 감소시켜 로봇 제어에 유리합니다. 개발된 파이프라인은 공개되어 연구자들에게 사용될 수 있는 플랫폼을 제공합니다.



### BEEM: Boosting Performance of Early Exit DNNs using Multi-Exit Classifiers as Experts (https://arxiv.org/abs/2502.00745)
Comments:
          Published at International Conference on Learning Representations (ICLR) 2025

- **What's New**: 본 논문에서는 Deep Neural Networks (DNNs)의 Early Exit (EE) 기법에 대해 새로운 의사결정 기준을 제안합니다. 제안된 기법은 exit classifiers를 전문가(experts)로 간주하고, 이들의 신뢰도(confidence scores)를 집계하여 ensemble 효과를 캡처합니다. 또한, 집계된 신뢰도值가 임계값(threshold)을 초과할 때만 샘플이 종료됩니다. 이로 인해 기존 DNN 추론 방식보다 성능 개선을 목표로 합니다.

- **Technical Details**: BEEM(Boosting Performance of Early Exit DNNs using Multi-Exit Classifiers as Experts) 메커니즘은 각 중간 exit classifier를 전문가로 취급하여 보다 효과적인 판단을 내립니다. 전문가의 정확도에 따라 신뢰도를 가중치로 조정하며, 이전 층의 신뢰도가 일치할 경우에만 집계하여 결정합니다. 이 때 임계값은 각 exit의 오류율(error rates)을 기반으로 설정되며, EE 결정을 내리기 위한 기초로 사용됩니다.

- **Performance Highlights**: 실험을 통해 BEEM은 GLUE 및 COCO 데이터셋에서 기존 EE 방법들보다 1.5배에서 2.1배의 속도 향상을 달성하였습니다. 특히 이미지 캡셔닝과 언어 작업에서 정확도가 비슷하거나 개선된 결과를 보였으며, 특정 NLP 작업에서는 최종 레이어보다 더 높은 정확도를 기록했습니다. 이 연구의 소스코드는 공개적으로 제공되어 연구자들에게 유용하게 활용될 수 있습니다.



### Registration-Enhanced Segmentation Method for Prostate Cancer in Ultrasound Images (https://arxiv.org/abs/2502.00712)
- **What's New**: 이 연구는 MRI-TRUS 융합 기반의 자동 분할(segmentation) 방법을 제안합니다. 이 방법은 수동 주석을 요구하지 않고 TRUS 이미지에서 바로 전립선 종양을 식별할 수 있습니다. 기존의 단순한 데이터 결합 기법과는 달리, 제안된 방법은 등록-분할(registration-segmentation) 프레임워크를 통합하여 MRI와 TRUS 모달리티 간의 공간 정보를 효과적으로 활용합니다.

- **Technical Details**: 연구에서는 1,747명의 환자 데이터를 사용하여 방법의 유효성을 검증했습니다. 제안된 방법은 평균 Dice 계수가 0.212로, TRUS 전용 방법(0.117)과 단순한 MRI-TRUS 융합 방법(0.132)에 비해 상대적으로 81.2% 및 60.6% 개선된 성능을 보였습니다. 통계적으로 의미 있는 결과(p < 0.01)를 보여주었으며, 이는 종양 분할 정확도를 강화하는 데 기여합니다.

- **Performance Highlights**: 이 프레임워크는 전립선암 진단의 복잡성을 줄이고, 여러 모달리티를 사용하는 의료 영상 작업에 적용할 수 있는 유연한 아키텍처를 제공합니다. 전통적인 TRUS 유도 생검 절차에 비해, 제안된 방법은 시간과 노력을 크게 줄여 의료진의 부담을 경감할 것으로 기대됩니다.



### MM-IQ: Benchmarking Human-Like Abstraction and Reasoning in Multimodal Models (https://arxiv.org/abs/2502.00698)
- **What's New**: 이 논문은 MM-IQ라는 새로운 평가 프레임워크를 제안합니다. MM-IQ는 8개의 서로 다른 추론 패러다임을 아우르는 2,710개의 정교하게 선별된 테스트 항목으로 구성되어 있습니다. 기존의 멀티모달 시스템이 인간의 인지 능력에 가까워지지 못함을 강조하며, 발전을 위한 새로운 접근의 필요성을 나타냅니다.

- **Technical Details**: MM-IQ는 다양한 입출력 형태, 문제 구성 및 추론 패러다임에 따라 기존의 AVR 벤치마크와 비교할 수 있습니다. 이 평가 기준은 LMMs의 도형 인식, 추론, 그리고 숫자 추리 능력을 평가하는 데 중점을 둡니다. 모든 테스트 항목은 인간의 고차원적 추론 능력을 기반으로 체크되며, 8개의 미세한 추론 패러다임으로 분류됩니다.

- **Performance Highlights**: 현재의 멀티모달 시스템은 우연히 맞힐 확률(25%)을 조금 넘는 27.49%의 정확도만을 보여주고 있습니다. 이는 인간 수준의 성능 51.27%와 비교할 때 매우 부족한 수치임을 시사합니다. MM-IQ를 통한 체계적인 평가 결과는 현재의 LMMs가 인간과 같은 인지 적응 능력을 갖추고 있지 않음을 강조합니다.



### Strengthening Generative Robot Policies through Predictive World Modeling (https://arxiv.org/abs/2502.00622)
Comments:
          Website: this https URL

- **What's New**: 이번 연구에서는 Generative Predictive Control (GPC)라는 새로운 학습 제어 프레임워크를 소개합니다. 이 프레임워크는 전문가의 데모에서 생성된 정책을 복제하고, 예측 액션 조건 세계 모델을 훈련시키며, 온라인 계획을 통해 행동 제안을 최적화합니다. 특히, GPC는 conditional video diffusion를 사용하여 물리적으로 정확한 시각적 세계 모델을 학습하고 강력한 시각적 선견지명을 가능하게 합니다.

- **Technical Details**: GPC는 세 가지 주요 모듈로 구성됩니다. 첫째, Generative Policy Training에서는 전문가의 데모를 기반으로 행동 조각의 조건부 분포를 근사하는 정책을 학습합니다. 둘째, Predictive World Modeling에서는 과거 관측치와 행동 조각에 따라 미래 관측치를 시뮬레이션하는 세계 모델을 구축합니다. 마지막으로, Online Planning 모듈은 전문가의 정책과 예측 세계 모델을 결합하여 미래를 고려한 행동 제안의 순위를 매기고 최적화합니다.

- **Performance Highlights**: GPC는 시뮬레이션 및 실제 환경의 다양한 로봇 조작 과제에서 기존의 행동 클로닝을 능가하는 성능을 보여주었습니다. 구체적으로, GPC는 상태 기반 및 비전 기반 환경 모두에서 강화된 정책을 통해 더 높은 성공률을 기록하였습니다. 이러한 성과는 GPC의 예측적 세계 모델과 온라인 계획 알고리즘이 결합되어 실현된 결과로, 이는 현실 세계의 로봇 조작에서 더욱 강력한 응용 가능성을 제시합니다.



### Distribution-aware Fairness Learning in Medical Image Segmentation From A Control-Theoretic Perspectiv (https://arxiv.org/abs/2502.00619)
Comments:
          12 pages, 3 figures, 9 tables

- **What's New**: 이 논문에서는 의료 이미지 세분화의 공정성을 보장하기 위한 새로운 접근법으로, 다수의 전문가를 유통시키는 방법인 'Distribution-aware Mixture of Experts (dMoE)'를 제안합니다. 이는 최적 제어 이론(optimal control theory)에서 영감을 받아 개발되었으며, 임상 데이터의 불균형으로 인한 편향 문제에 대응합니다. dMoE는 다양한 네트워크 아키텍처에 통합되어 의료 이미징 분석의 여러 작업에 광범위하게 적용될 수 있습니다.

- **Technical Details**: dMoE는 Mixture of Experts (MoE) 프레임워크를 기반으로 하며, 피드백 제어 메커니즘으로 새롭게 해석되었습니다. 이 구조는 개별적인 특성과 분포 패턴을 적응적으로 통합하여 깊은 신경망 훈련에 활용됩니다. dMoE는 다양한 네트워크 구조, 특히 transformers와 CNNs에서 원활하게 작동하여 2D 및 3D 의료 이미지 세분화 작업에 적용될 수 있습니다.

- **Performance Highlights**: 다양한 임상 데이터셋에서의 실험 결과, dMoE는 공정성 학습 접근법을 한층 발전시켰으며, 불균형 데이터 분포로 인한 편향을 효과적으로 완화하는 데 성공했습니다. 이 접근 방식은 공정성 학습 패러다임 내에서 의료 이미지 세분화의 진단 및 치료 결정 과정을 보다 강력하고 공정하게 만드는 데 유망한 방법을 제공하였습니다.



### Integrating Frequency Guidance into Multi-source Domain Generalization for Bearing Fault Diagnosis (https://arxiv.org/abs/2502.00545)
- **What's New**: 이 논문은 Fourier 기반의 Augmentation Reconstruction Network(FARNet)를 제안합니다. 이 네트워크는 다양한 도메인 간의 차이를 효과적으로 줄이기 위해 진폭 스펙트럼과 위상 스펙트럼을 분석하여 특징을 학습합니다. 또한, manifold triplet loss를 도입하여 더욱 정교한 결정 경계를 형성하고, 이를 통해 보다 나은 일반화 성능을 실현합니다.

- **Technical Details**: FARNet은 진폭 스펙트럼 서브 네트워크와 위상 스펙트럼 서브 네트워크로 구성되어 있으며, 주어진 데이터의 진폭과 위상을 차례로 학습합니다. 이를 통해 frequency-spatial interaction learning 전략을 활용하여 데이터의 전반적 특징을 향상시킵니다. 또한, FSIM(Frequency-Spatial Interaction Module)을 통해 글로벌 정보와 로컬 공간 특징을 통합하여 표현 학습을 촉진합니다.

- **Performance Highlights**: CWRU와 SJTU 데이터셋에 대한 광범위한 실험을 통해 FARNet은 기존의 크로스 도메인 접근 방식들보다 우수한 성능을 보여줍니다. 특히, 다양한 작동 조건에서의 일반화 능력을 향상시키기 위해 설계된 모델로, 고급 결정을 지원하고 결합 학습 정확도를 높였습니다.



### VertiFormer: A Data-Efficient Multi-Task Transformer for Off-Road Robot Mobility (https://arxiv.org/abs/2502.00543)
Comments:
          9 figures, url: this https URL

- **What's New**: 이 논문은 VertiFormer라는 새로운 다중 작업 Transformer 모델을 제안하여 극도로 험난한 오프로드 지형에서 로봇의 이동성을 개선하는 데 중점을 두고 있습니다. VertiFormer는 단 한 시간의 데이터로 학습되어 강력한 학습 능력을 발휘하며, 전통적인 방법과는 달리 통합된 잠재 표현을 사용하여 여러 작업을 동시에 수행할 수 있습니다. 이 모델은 특히 역호출 방식으로 인한 에러 전파를 피하고, 누락된 모달리티의 영향을 줄이는 것으로 특징지어집니다.

- **Technical Details**: VertiFormer는 학습 가능한 마스킹 모델링(learnable masked modeling)과 다음 토큰 예측(next token prediction) 패러다임을 통해 로봇의 동작과 지형 정보를 예측합니다. 또한 비자기 회귀(non-autoregressive) 설계를 통해 계산 병목 현상과 에러 전파를 완화합니다. 이 모델은 다양한 오프로드 이동 작업을 위해 여러 목적 함수와 함께 복합적인 시간적 맵핑을 학습하여 일반화 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과, VertiFormer는 수치적으로 입증된 기존의 기노다이나믹(modelling approaches) 모델들보다 나은 내비게이션 성능을 보였습니다. 기존의 다중 작업 학습 방식을 통해 데이터가 한정된 환경에서도 효과적으로 로봇의 이동성을 확장할 수 있는 가능성을 보여줍니다. VertiFormer에 대한 평가 결과는 물리적 로봇에서 다양한 오프로드 이동 작업을 성공적으로 수행할 수 있는 기반을 마련했습니다.



### Weak-to-Strong Diffusion with Reflection (https://arxiv.org/abs/2502.00473)
Comments:
          20 pages, 19 figures, 14 tables

- **What's New**: 이 연구에서는 Weak-to-Strong Diffusion (W2SD)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 기존의 약한 모델과 강한 모델의 차이를 활용하여 이상적인 모델과 강한 모델 간의 격차를 근사하는 방법을 제안합니다. W2SD는 다양한 모델 쌍을 전략적으로 선택하여 생성 품질을 크게 향상시킬 수 있습니다.

- **Technical Details**: W2SD는 반사 작업을 통해 데이터 분포의 격차를 추적합니다. 이 프레임워크는 주어진 약한 모델과 강한 모델 간의 추정된 밀도 차이를 사용하여 강한 모델을 개선하는 방식으로 작동합니다. 연구는 W2SD가 잠재 변수를 실제 데이터 분포의 영역으로 유도하는 방법을 수학적으로 이해하고 설명합니다.

- **Performance Highlights**: 광범위한 실험을 통해 W2SD는 인간의 선호도, 미적 품질, 프롬프트 준수를 개선하였으며, 다양한 생성 작업(예: 이미지, 비디오)에서 SOTA 성능을 달성했습니다. W2SD를 사용한 Juggernaut-XL은 원래 결과에 비해 HPSv2 승률을 90%까지 개선하였으며, 추가적인 계산 비용을 초과하는 성능 향상을 보였습니다.



### Explorations of the Softmax Space: Knowing When the Neural Network Doesn't Know... (https://arxiv.org/abs/2502.00456)
Comments:
          9 pages, 5 figures, 1 table. arXiv admin note: substantial text overlap with arXiv:2407.07821

- **What's New**: 본 논문에서는 기계 학습 모델의 예측 신뢰성을 측정하는 새로운 접근 방식을 제안합니다. 클러스터링 기법을 활용하여 학습된 신경망의 출력과 클래스 중심 간의 거리 변화를 분석합니다. 이 거리를 예측의 신뢰성을 평가하는 메트릭으로 제안하며, 이를 통해 자동화된 예측의 수용 가능성과 인간 운영자에게 의사 결정을 맡겨야 할 시점을 판단합니다.

- **Technical Details**: 제안된 방법은 클래스에 대한 평균 softmax 출력을 나타내는 중심점에 각 예측을 할당하고 잘못된 예측과 해당 클래스 중심 간의 최소 거리로 안전 임계값을 정의합니다. MNIST와 CIFAR-10 데이터세트에서 Convolutional Neural Network와 Vision Transformer를 이용해 이 접근 방식을 평가하였습니다. 결과는 제안된 메트릭이 두 데이터 세트와 네트워크 모델에서 일관되며, 자동화된 예측의 수용 가능성을 판단하는 효율적인 방법을 제공할 수 있음을 보여줍니다.

- **Performance Highlights**: 연구 결과, 예측 정확도가 높은 클래스는 해당 중심까지의 softmax 거리가 낮다는 것을 확인했습니다. 이는 softmax 분포의 변화가 모델 예측에 대한 신뢰도를 나타내는 좋은 지표가 될 수 있음을 시사합니다. 제안된 접근 방식은 CNN과 ViT 아키텍처 전반에 걸쳐 일관성을 보였으며, 기계 학습 모델을 안전하고 신뢰성 있게 운영하는 데 기여할 것으로 기대됩니다.



### Segment Anything for Histopathology (https://arxiv.org/abs/2502.00408)
- **What's New**: 이 논문에서는 디지털 병리학에서 핵심 분석 작업인 핵(biological nucleus) 분할을 위한 새로운 비전 기초 모델인 PathoSAM을 소개합니다. PathoSAM은 다양한 주석 데이터셋을 활용하여 SAM(Segment Anything Model)을 통해 훈련되어, 자동 및 쌍방향 분할에서 최첨단 성능을 보여줍니다. 이 모델은 다양한 분할 작업에 적응할 수 있으며, 특히 감정세포 분할(Semantic nucleus segmentation)에서도 우수한 결과를 보여줍니다.

- **Technical Details**: PathoSAM은 SAM과 μSAM의 세 가지로 구성된 포맷을 바탕으로 합니다. SAM은 사용자가 제공하는 입력 프롬프트를 사용하여 객체를 식별할 수 있는 쌍방향 분할(interactive segmentation)을 지원합니다. 본 연구에서는 또한 PathoSAM을 기반으로 한 자동적 의미 분할(automatic semantic segmentation)에 대해서도 논의하고, 방대한 히스토패솔로지 데이터셋에서 모델을 훈련하고 평가했습니다.

- **Performance Highlights**: PathoSAM은 기존의 SAM 변형과 비교할 때 쌍방향 분할에서 성능이 뛰어났으며, 핵 인스턴스 분할 자연 상태에서 새로운 최고 성능 모델의 위치를 차지하고 있습니다. 정식 발표된 CellViT와는 비교하지 못하더라도, 여러 세그멘테이션 작업을 위한 미세조정이 가능합니다. PathoSAM은 사용이 편리한 데이터 주석 도구와 호환되며, 전체 슬라이드 이미지 세분화를 위한 스크립트를 제공합니다.



### FlexCloud: Direct, Modular Georeferencing and Drift-Correction of Point Cloud Maps (https://arxiv.org/abs/2502.00395)
Comments:
          Accepted for publication at VEHITS 2025, Proceedings of the 11th International Conference on Vehicle Technology and Intelligent Transport Systems - VEHITS; 2025

- **What's New**: 본 연구에서는 SLAM에서 생성된 포인트 클라우드 맵(PCM)의 자동 지리정보 시스템(georeferencing)을 위한 모듈화된 접근법인 FlexCloud를 제안합니다. 기존 SLAM 방법과 조화를 이루도록 설계되어 있으며, 생성된 지역 포인트 클라우드 맵과 그 오도메트리만 활용하여 GNSS 위치를 통해 직접적으로 지리정보 시스템을 수행합니다. 이 논문에서는 3D Rubber-sheet 변환을 사용하여 맵의 왜곡을 수정하고, 이를 통해 일관적이고 전 세계적으로 참조된 포인트 클라우드 맵 생성이 가능하다고 설명합니다.

- **Technical Details**: 본 연구는 LiDAR 데이터를 기반으로 한 SLAM의 기존 제약을 극복하기 위해, 보간(interpolation) 기반 전략을 통해 GNSS 위치에 따라 자동으로 제어 점(control points)을 결정하는 방법을 구현합니다. 이를 통해 장기적인 드리프트를 효과적으로 보정하면서도 맵의 구조와 연속성을 유지하는 3D Rubber-sheet 변환을 적용하였습니다. 이 접근법은 SLAM으로 생성된 지역 PCM을 전 세계에 참조된 것으로 변환할 수 있으며, 기존의 수작업 필요한 접근법을 대체하는 모듈러 파이프라인으로 설계되었습니다.

- **Performance Highlights**: 연구팀은 Abu Dhabi의 Yas Marina Circuit(YMC)에서 수집된 자체 기록 데이터와 KITTI 비전 벤치마크의 시퀀스 00에 대한 실험을 통해 FlexCloud의 결과를 평가하였습니다. 이 실험을 통해 제안된 방법의 일반화 가능성을 입증하였으며, 또 원활한 자동화된 프로세스로서의 실용성을 확인하였습니다. FlexCloud는 ROS 2 패키지로 오픈소스되어 있어 연구자들이 쉽게 활용할 수 있도록 합니다.



### A Unit-based System and Dataset for Expressive Direct Speech-to-Speech Translation (https://arxiv.org/abs/2502.00374)
- **What's New**: 최근 음성 대 음성 번역(S2ST) 연구는 번역 정확도와 자연스러움에 초점을 맞추는 반면, 감정과 태도를 전달하는 데 필수적인 패러링귀스틱(paralinguistic) 정보와 같은 주요 요소는 간과하는 경향이 있습니다. 본 연구에서는 다양한 영화 오디오 트랙에서 신중하게 구성된 다국어 데이터셋을 소개하며, 각 데이터셋 쌍은 감정 정보와 지속 시간에 맞춰 정확하게 매칭되었습니다. 이 연구는 감성 정보를 보존하며 번역의 정확성과 자연성 또한 유지하는 S2ST 방법론을 제안합니다.

- **Technical Details**: 우리의 S2ST 시스템은 세 가지 구성 요소로 이루어져 있습니다. 우선, 하나의 언어에서 음성을 이산 단위로 변환하여 직접적인 음성 번역을 진행합니다. 그 후, 음성에서 화자의 특성을 추출하여 감성 전달 모델을 통해 목표 언어로 감성이 풍부하게 재생산된 음성을 합성합니다. 또한, HuBERT 구조를 활용하여 이산 단위를 추출하는 과정과 K-평균 알고리즘을 통한 클러스터 중심 생성을 통해 특정 언어의 음성을 연속적으로 표현할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 원음(Source speech)에서 더 많은 패러링귀스틱 정보를 유지하면서 번역의 정확성과 자연성을 높은 수준으로 유지하는 성과를 보여주었습니다. 특히, 영화 및 TV 프로그램과 같은 감정이 풍부한 대화 자료를 기반으로 한 데이터셋을 통해 감정 번역 측면에서 우수한 성능을 기록했습니다. 이는 향후 음성 번역 모델 연구에 중요한 기준점을 제공할 것입니다.



### Prostate-Specific Foundation Models for Enhanced Detection of Clinically Significant Cancer (https://arxiv.org/abs/2502.00366)
Comments:
          44pages

- **What's New**: 이번 연구에서는 전립선암 진단의 정확성을 높이기 위한 새로운 모델인 ProViCNet(전립선 비전 대조 네트워크)를 소개합니다. ProViCNet은 MRI(자기공명영상)와 TRUS(경직장 초음파) 이미지를 사용하여 종합적인 암 검출을 목표로 합니다. 이 모델은 4,401명의 환자 데이터를 통해 훈련 및 검증되어, 방사선과 전문가의 주석에 의해 확인된 생검 정보를 기반으로 하고 있습니다.

- **Technical Details**: ProViCNet은 방사선 영상에서 패치 수준의 대조 학습(patch-level contrastive learning)을 적용하여 훈련되었습니다. 이 모델은 다중 내부 및 외부 검증 코호트를 통해 일관된 성능을 보여주었으며, 수신기 작동 곡선 아래 면적(AUC) 수치는 0.875에서 0.966 사이로 나타났습니다. 또한 mpMRI(다중 매개변수 MRI)에서의 독립 판독 연구에서 방사선 전문의보다 높은 성능(0.907 대 0.805, p<0.001)을 보였습니다.

- **Performance Highlights**: ProViCNet은 PSA(전립선 특이 항원)와 통합하여 가상 선별 검사(virtual screening test)를 개발하였으며, 임상적으로 중요한 암을 검출하는 데 높은 민감도를 유지하면서 특이도를 15%에서 38%로 두 배 이상 증가시켰습니다(p<0.001). 이러한 결과는 ProViCNet이 전립선암 진단의 정확성을 높이고 불필요한 생검을 줄이는 데 기여할 수 있는 잠재력을 강조합니다.



### A Study on the Performance of U-Net Modifications in Retroperitoneal Tumor Segmentation (https://arxiv.org/abs/2502.00314)
Comments:
          Accepted for presentation at the 2025 SPIE Medical Imaging Conference

- **What's New**: 이 논문은 복막 후 종양의 정확한 분할(segmentation)을 위한 새로운 접근 방식을 제시합니다. 연구진은 U-Net 아키텍처를 개선하여 Vision Transformer(ViT) 및 새로운 Mamba 상태 공간 모델(State Space Model, SSM)을 결합한 ViLU-Net 모델을 개발했습니다. 이 모델은 기존 방법들보다 높은 정확도와 효율성을 보여주며, 복잡한 종양 구조를 효과적으로 처리할 수 있습니다.

- **Technical Details**: 연구는 인공지능 모델의 발전을 통해 의료 이미지 분할 분야에서 CNN과 ViT의 하이브리드 모델을 개발하는 것에 초점을 맞추고 있습니다. 특히 xLSTM(Long Short-Term Memory) 모델을 활용하여 장기 종속성(long-range dependencies) 문제를 해결하고, 더 나은 속도와 성능으로 세밀한 이미지 분석을 지원합니다. 논문에서 제안된 ViLU-Net은 특정한 Retroperitoneal 종양 데이터셋에서 스킵 연결(skip connections)의 디자인을 개선하여 성능을 더욱 향상시킵니다.

- **Performance Highlights**: ViLU-Net 모델은 새로운 CT 데이터셋에서 기존의 최첨단(deep learning SOTA) 방법들과 비교하여 우수한 성능을 보여주며, 특히 xLSTM이 도입된 경우의 효율성이 강조됩니다. 이 연구는 복막 후 종양의 수술 계획 및 치료 반응 평가에 있어 더 나은 결정 지원을 가능하게 하며, 특히 이미지 분석의 일관성을 높이는 데 기여합니다. 코드와 데이터셋은 GitHub에서 공개되어 있어 연구자들이 쉽게 접근하여 추가 연구를 진행할 수 있습니다.



### K Nearest Neighbor-Guided Trajectory Similarity Learning (https://arxiv.org/abs/2502.00285)
- **What's New**: 이번 연구에서는 TSMini라는 새로운 모델을 제안합니다. TSMini는 서브 뷰(sub-view) 모델링 기법을 활용하여 다양한 수준의 궤적 패턴을 학습할 수 있도록 설계되었습니다. 또한 k-nearest neighbor(kNN) 기반의 손실 함수를 적용하여 궤적 간의 절대 유사성뿐만 아니라 상대 유사성을 학습하도록 유도합니다.

- **Technical Details**: TSMini의 서브 뷰 인코더는 입력 궤적의 세분화된 지역 이동 패턴을 포착하는 연속적인 하위 연속(sub-sequence)으로 궤적을 분해합니다. 이를 통해 다양한 세분화 수준에서 궤적 패턴을 학습하여 정확한 유사성 추정이 가능해집니다. kNN 기반 손실 함수는 궤적 쌍 사이의 상대적 유사성을 학습하는 데 중점을 두어, 훈련 데이터에서 유사성 신호를 효과적으로 활용합니다.

- **Performance Highlights**: 실험 결과에 따르면 TSMini는 기존의 최첨단 모델들에 비해 평균적으로 22%의 정확도 개선을 보여주었습니다. 다양한 실제 데이터셋에 대한 extensive한 실험을 통해 모델의 상시적으로 탁월한 성능을 확인하였습니다. 이로써 궤적 유사성 측정 학습을 위한 TSMini의 효과성을 입증합니다.



### Simultaneous Estimation of Manipulation Skill and Hand Grasp Force from Forearm Ultrasound Images (https://arxiv.org/abs/2502.00275)
Comments:
          30 pages, 52 references, 10 figures, 8 tables and 2 supplementary videos. Currently under review

- **What's New**: 이 논문은 팔꿈치 초음파 데이터를 사용하여 조작 기술과 손의 힘을 동시에 추정할 수 있는 새로운 방법을 제안합니다. 실험은 7명의 참가자가 수행한 5가지 조작 작업을 바탕으로 진행되었으며, 이로써 조작 기술의 분류와 힘 추정의 높은 정확도를 달성했습니다. 연구 결과는 로봇 원격 조작 및 인간-로봇 기술 전이에 있어서 초음파의 가능성을 강조합니다.

- **Technical Details**: 본 연구에서 사용된 B-mode 초음파는 팔꿈치에서 수집된 데이터를 활용하여 감지된 조작 기술을 분류하고 힘을 추정합니다. 실험 결과, 조작 기술의 분류 정확도는 94.87%에 달하며, 힘 추정의 평균 제곱근 오차(RMSE)는 0.51 N으로 나타났습니다. 이러한 접근 방식은 로봇을 위한 복잡한 조작 작업을 가능하게 하는 기반 기술을 제공합니다.

- **Performance Highlights**: 조작 기술 분류는 5회 교차 검증을 통해 94.9 ± 10.2%의 정확도를 보였고, 힘 추정에서는 평균 RMSE가 0.51 ± 0.19 N으로 측정되었습니다. 이 모델은 서로 다른 참가자들 간의 변동성을 잘 처리할 수 있으며, Grad-CAM을 통한 시각화를 통해 초음파 이미지와 관련된 주요 근육 그룹을 해석하여 신뢰성을 높였습니다. 이는 RoI(Region of Interest) 분석을 통해 적합한 조작 기술과 힘을 제공하는 데 중요한 기여를 합니다.



### Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion (https://arxiv.org/abs/2502.00264)
- **What's New**: 이번 논문에서는 변환기(transformer) 모델에 대한 새로운 형태의 매개변수 공간 대칭성인 회전 대칭(rotation symmetry)을 도입합니다. 기존의 순열 대칭(permutation symmetry)은 이산적 특성으로 인해 변환기에서는 유용성이 제한적이었으나, 회전 대칭은 매개변수 행렬을 연속적으로 회전시키는 방식을 제안하여 대칭의 범위를 확대합니다. 이 혁신적인 접근법은 모델 통합(model fusion)에도 유용하게 활용될 수 있습니다.

- **Technical Details**: 회전 대칭은 쿼리(query) 및 키(key) 행렬을 함께 분석하여, 쿼리 행렬에 회전을 적용하고 이에 상응하는 역회전을 키 행렬에 적용함으로써 쿼리-키 곱(product)을 보존합니다. 또한 이 같은 회전 규칙은 값(value) 및 출력(output) 행렬에도 적용될 수 있음을 보여줍니다. 이를 통해 우리는 회전 대칭을 활용한 매개변수 매칭(parameter matching) 알고리즘을 최적화 문제로 설정하고, 폐쇄형 해결책을 제안하여 극대화된 효율성을 달성합니다.

- **Performance Highlights**: 실험을 통해 회전 대칭 기반의 매칭 알고리즘이 모델 통합에서 효과적으로 성능을 향상시킨다는 것을 입증하였습니다. 특히, 매개변수의 일부만을 매칭해도 유의미한 성능 개선이 가능하다는 결과를 도출하였습니다. 이번 연구는 매개변수 공간 대칭성을 통해 모델 통합의 가능성을 향상시킬 수 있는 큰 잠재력을 가진다는 점을 강조합니다.



### Patch Triplet Similarity Purification for Guided Real-World Low-Dose CT Image Denoising (https://arxiv.org/abs/2502.00253)
- **What's New**: 최근 연구에서는 저선량 전산화 단층촬영(LOW-dose computed tomography, LDCT) 이미지를 효과적으로 복원하기 위해 비식별 전산화 단층촬영(non-contrast CT, NCCT) 이미지를 보조 자료로 활용하는 방안을 제안했습니다. 이를 통해 기존의 저선량 CT 이미지 복원 알고리즘들이 겪었던 모호한 구조나 움직임 아티팩트 문제를 완화할 수 있을 것으로 기대됩니다. 새로운 패치 트리플릿 유사성 정제(Patch Triplet Similarity Purification, PTSP) 전략을 도입하여 교육 데이터 간의 공간 불일치를 줄이기 위한 방법을 제시하였습니다.

- **Technical Details**: 논문에서는 NCCT 이미지를 저선량 CT 이미지의 복원 과정에 효과적으로 통합하기 위해 크로스 어텐션(cross-attention) 메커니즘을 통합한 두 가지 이미지 노이즈 제거 트랜스포머를 수정하였습니다. 또한, PTSP 전략을 통해 LDCT, NDCT, NCCT 이미지 패치 세트를 선택하는 방법을 공유하며, 고품질의 훈련 데이터를 생성하여 노이즈 제거 성능을 개선했습니다. 실험을 통해 수정된 SwinIR과 HAT 모델이 15개 경쟁 방법보다 뛰어난 성능을 보임을 확인했습니다.

- **Performance Highlights**: 제안된 PTSP 전략과 NCCT 이미지 보조 자료가 통합된 모델들은 실제 임상 데이터셋에서 우수한 성능을 보여주었습니다. 특히, NCCT 이미지의 정보가 LDCT 이미지 노이즈 제거에 매우 유용하다는 점이 부각되었습니다. 실험 결과에 따르면, 제안된 방식은 정량적 및 정성적으로 LDCT 이미지 복원 성능을 높이며, 이러한 접근 방식의 유효성을 뒷받침하는 여러 분석 결과들이 제공됩니다.



### Mordal: Automated Pretrained Model Selection for Vision Language Models (https://arxiv.org/abs/2502.00241)
- **What's New**: 이번 논문에서는 자동화된 멀티모달 모델 검색 프레임워크인 Mordal을 도입하여 특정 작업을 위한 가장 적합한 Vision Language Model (VLM)을 효율적으로 찾는 방법을 제안합니다. 기존의 VLM들은 전문가에 의해 수작업으로 제작되어, 사용자가 원하는 작업에 맞는 자동 프레임워크가 없었습니다. Mordal은 후보 모델의 수를 줄이고 각 후보를 평가하는 시간을 최소화하여 효율적인 검색을 가능하게 합니다. 연구 결과, Mordal은 기존의 grid search에 비해 GPU 시간을 최대 8.9배에서 11.6배 절약할 수 있음을 보여주는 동시 신 모델을 발견했습니다.

- **Technical Details**: Mordal은 VLM을 구축하고 훈련하기 위한 기존 접근 방식을 크게 개선하는 방법론을 제공합니다. VLM은 일반적으로 Vision Encoder, Feature Projector 및 Language Model로 구성되며, 각각의 구성 요소는 특정한 역할을 수행하여 입력 이미지와 텍스트를 결합하고 해석합니다. 특히 다양한 pretrained 비전 인코더와 언어 모델을 조합해 최적의 성능을 낼 수 있는 조합을 탐색하며, 이를 위해 초기 후보 모델들을 유사도 기준으로 클러스터링하고 평가 시간을 단축하는 조기 중지 메커니즘을 도입했습니다.

- **Performance Highlights**: Mordal을 사용한 성능 평가에서, 전체적으로 49개의 VLM 후보를 대상으로 한 그리드 서치보다 높은 효율성과 적은 계산 시간으로 최적의 VLM을 찾는 것을 확인했습니다. 본 연구에서는 특히 비전-텍스트 정렬 데이터에 훈련된 Feature Projector의 중요성을 강조하며, 최적의 pretrained Vision Encoder 및 Language Model 조합을 찾기 위해 진행된 다양한 실험 결과를 공유합니다. 실험 결과들은 VLM 성능 향상에 기여하며, 여기서 발견된 새로운 VLM들은 기존의 최첨단 모델들을 초과하는 성능을 나타냈습니다.



### Fast Solvers for Discrete Diffusion Models: Theory and Applications of High-Order Algorithms (https://arxiv.org/abs/2502.00234)
Comments:
          38 pages, 7 figures

- **What's New**: 이 논문은 고차 수치적 추론 기법을 이산 확산 모델에 맞춰 개발함으로써, 효율성을 크게 높이는 데 착안하였습니다. 특히, 기존의 1차 정확도 방법의 한계를 극복하기 위해 높은 단계 크기를 가능하게 하여 오차를 줄이는 방법을 제안합니다. 이 방법은 GPT-2 수준의 텍스트와 ImageNet 수준의 이미지 생성 작업에서 엄청난 성능 향상을 보여줍니다.

- **Technical Details**: 이 논문에서는 θ-Runge-Kutta-2 (θ-RK-2) 방법과 θ-사다리꼴(θ-trapezoidal) 방법을 제안합니다. 이 두 방법의 이론적 속성을 엄밀하게 증명하고 θ-사다리꼴 방법의 2차 수렴성을 확립했습니다. 제안된 방법들은 일반적인 수치적 방법과 달리 이산 확산 모델에 특화되어 성능이 대폭 향상되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 θ-사다리꼴 방법이 기존 접근법에 비해 동등한 계산 제약 조건 하에서도 높은 샘플 품질을 보여주었습니다. 이를 통해 이산 확산 모델의 효율적인 추론이 가능해지는 점을 강조하고 있습니다. 따라서 이 연구는 텍스트 및 이미지 생성의 새로운 기준을 제시함으로써 여러 응용 분야에서의 발전 가능성을 암시합니다.



### Fantastic Multi-Task Gradient Updates and How to Find Them In a Con (https://arxiv.org/abs/2502.00217)
Comments:
          16 pages, 7 figures, 5 tables

- **What's New**: 최신 연구에서 ConicGrad라는 다중 작업 최적화 프레임워크를 제안하여 기존 방법의 주요 한계를 해결하고자 합니다. 이 방법은 각 작업의 경량화된 Gradient를 효과적으로 조정하기 위한 각도 제약을 도입하였습니다. 이를 통해 ConicGrad는 최적화 목표와의 정합성을 유지하면서도 기존 방법보다 동적이고 유연한 업데이트 방향을 제공합니다. 또한, 기존 방법들에 비해 계산 효율성을 높이고 학습 속도를 가속화하는 성능 개선을 보여주고 있습니다.

- **Technical Details**: ConicGrad는 다중 작업 모델을 θ∈ℝM으로 매개변수화하고 K≥2의 작업 수를 갖습니다. 각 작업의 목표 함수는 ℒi(θ)이며, 전통적인 목표는 모든 손실의 균일 평균을 최적화하는 것입니다. ConicGrad는 이런 전통적인 접근의 한계를 극복하기 위해 각 작업 손실의 감소율을 극대화하는 대체 업데이트 벡터 d를 찾고자 합니다. 이는 Angular Constraint를 활용하여 효율적인 경량화된 Gradient 업데이트를 보장합니다.

- **Performance Highlights**: 다양한 표준 감독 학습 및 강화 학습 MTL 벤치마크에서 ConicGrad의 성능을 평가한 결과, 여러 작업에 걸쳐 최첨단 성과를 달성하였습니다. ConicGrad는 이전 방법들보다 더 빠른 수렴 속도를 보이며, 이론적 수렴 보장을 통해 이러한 경험적 결과를 뒷받침합니다. 실험 결과는 높은 차원의 파라미터 공간에서도 효율성과 확장성을 유지함을 보여줍니다.



### Evaluating Deep Human-in-the-Loop Optimization for Retinal Implants Using Sighted Participants (https://arxiv.org/abs/2502.00177)
- **What's New**: 이번 연구는 HILO (Human-in-the-loop optimization) 방식이 실제 인간 참가자와 상호작용할 때 시각 보조기구의 최적 자극 전략을 개인화하는 데 효과적인지를 평가했습니다. 이전 연구들은 시뮬레이션 환경에서만 HILO의 유용성을 보여주었으나, 이번 연구는 실제 사용자의 피드백을 반영하여 HILO의 성능을 테스트했습니다. 연구 결과, 참가자들은 HILO에서 생성된 자극을 다른 방법보다 선호하며, 모델의 misspecification 및 다양한 사용자 조건에서도 HILO가 효과적으로 작동함을 확인했습니다.

- **Technical Details**: 이 연구에서는 17명의 시각이 정상인 참가자가 시뮬레이션된 시각 보조기구를 사용하여 전기 자극과 시각적 인지 간의 관계를 최적화하는 실험에 참여했습니다. 각 참가자는 전기 자극으로 생성된 두 개의 phosphene을 보고, 더 잘 맞는 것을 선택하여 HILO 프레임워크가 자극 매개변수를 반복적으로 조정할 수 있도록 했습니다. 참가자들은 60회의 'duel'을 수행하면서 HILO의 최적화를 경험하고, 이어서 39회의 추가 실험을 통해 최적화된 DSE와 기본 DSE를 비교했습니다.

- **Performance Highlights**: HILO로 생성된 자극은 na&iuml;ve encoder 및 DSE 단독의 자극보다 참가자들에게 지속적으로 더 높은 선호도를 보였습니다. 모든 실험 조건에서 HILO의 log odds 비율이 우세하여, 각 참가자들이 HILO의 출력을 유리하게 선택했다는 것을 잘 보여주었습니다. 이번 연구는 HILO가 인간의 개별적인 인지 변동성에 잘 적응할 수 있도록 도와주는 remarkably robust한 방법임을 입증하는 결정적인 증거를 제공합니다.



### Improving Quality Control Of MRI Images Using Synthetic Motion Data (https://arxiv.org/abs/2502.00160)
Comments:
          Accepted at ISBI 2025

- **What's New**: 본 연구에서는 MRI 품질 관리 품질 제어(QC)를 자동화하기 위해 합성 데이터를 활용한 새로운 접근 방식을 소개합니다. 기존의 unbalanced와 제한된 데이터셋의 문제를 해결하고자 합성 움직임 아티팩트를 사전 훈련에 사용하여 transfer learning을 적용합니다. 이 방법은 저품질 스캔을 식별하는 정확성을 향상시키고, 초기부터 훈련하는 것에 비해 자원과 시간을 절약하는 특징이 있습니다.

- **Technical Details**: 이 연구에서는 Human Connectome Project Early Psychosis(HCPEP)와 AMP SCZ 두 개의 데이터셋을 사용하여 모델을 훈련했습니다. SFCN(Simple Fully Convolutional Network) 아키텍처를 사용하여 합성 움직임을 예측하고, 후속 QC 분류를 위한 전이 학습 모델을 구축하였습니다. 사전 훈련 과정에서 점진적으로 원활한 데이터를 생성하기 위해 다양한 데이터 변환을 적용하였으며, 충실한 합성 데이터 생성 및 QC 판단을 위해 50개의 이산 범위로 모션 점수를 재구성하였습니다.

- **Performance Highlights**: 모델은 25시간에 걸쳐 훈련된 후 최대 검증 R2가 0.89에 도달하였으며, 이는 새로운 데이터에 대한 일반화 능력이 뛰어난 것을 보여줍니다. 전이 학습을 통해 훈련된 모델은 항상 처음부터 훈련된 모델보다 우수한 성능을 보였으며, 특히 저품질 스캔을 분류하는 데 있어 전혀 대응하지 못한 문제를 해결했습니다. 이러한 결과는 전이 학습이 더 많은 자원과 시간을 요구하지 않으며, 최적의 모델을 사용하는 것이 성능을 크게 향상시킬 수 있다는 중요한 단서를 제공하고 있습니다.



### Multimodal MRI-Ultrasound AI for Prostate Cancer Detection Outperforms Radiologist MRI Interpretation: A Multi-Center Study (https://arxiv.org/abs/2502.00146)
- **What's New**: 이번 연구는 전립선 생검에서 의심되는 병변을 목표로 하는 전처리 자기공명영상(MRI)의 사용이 증가함에 따라 인공지능(AI) 응용 프로그램이 임상적으로 중요한 전립선암(CsPCa) 탐지를 개선할 수 있는 가능성을 보여주고 있습니다. 특히, 이 연구는 MRI와 직장 초음파(TRUS) 이미지를 통합한 다중 모달 AI 프레임워크를 제안하여 CsPCa 식별을 향상시키기 위한 체계적인 평가를 진행하였습니다.

- **Technical Details**: 이 연구는 두 개의 기관에서 3110명의 환자를 대상으로 전립선 생검을 수행하였고, 1700개의 테스트 사례에 대해 3D UNet 아키텍처에 기반한 제안된 프레임워크의 성능을 평가했습니다. 이 때, 단일 모달(MRI 또는 TRUS만 사용하는) AI 모델과의 성능 비교를 통해 다중 모달 AI 접근법의 우수성을 검증하였습니다.

- **Performance Highlights**: 다중 모달 AI 접근법은 단일 모달 MRI(73%, 30%) 및 TRUS 모델(49%, 27%)에 비해 더 높은 민감도(80%)와 병변 Dice(42%)를 기록했습니다. 방사선 의사와의 비교에서도 다중 모달 모델은 높은 특이성(88% vs. 78%)과 병변 Dice(38% vs. 33%)를 보였으며, 민감도는 동등한 수준(79%)을 유지했습니다. 이러한 결과는 생검 및 치료 계획 과정에서 CsPCa 병변을 정확하게 목표로 할 수 있는 다중 모달 AI의 잠재력을 입증합니다.



### A Direct Semi-Exhaustive Search Method for Robust, Partial-to-Full Point Cloud Registration (https://arxiv.org/abs/2502.00115)
Comments:
          IROS 2024

- **What's New**: 이 논문에서 소개하는 Direct Semi-Exhaustive Search (DSES) 알고리즘은 점 구름( point cloud) 등록 문제를 직접 최적화하는 방법을 제공하여, 기존의 매칭 문제를 해결하는 두 단계 접근에서 벗어나 비연관(an unsupervised) 방식에서 문제를 해결할 수 있게 합니다. DSES는 현대 GPU의 병렬 처리 능력을 활용하여 회전 행렬을 효율적으로 탐색하고 최적의 강체 변환을 찾습니다. 이 접근법은 아웃라이어( outliers)와 부분 겹침에 대해 보다 강건한 결과를 보여주는 동시에, 데이터 기반의 모형이 필요 없습니다.

- **Technical Details**: DSES 알고리즘은 주어진 점 구름 X와 Y 사이의 강체 변환 {R, t}를 최적화하여 정렬하는 문제를 정의합니다. 이 알고리즘은 회전 행렬을 반복적으로 검토하고 각 회전과 관련된 inlier를 극대화하는 변환을 효율적으로 계산합니다. 다양한 거리 메트릭을 통해 수행할 수 있으며, 이는 최적의 변환 후보 {R, t}의 오류를 직접적으로 계산함으로써 이루어집니다. 이를 통해 기존 ICP 방법과 비교하여 계산 비용을 절감하면서도 더 나은 성능을 발휘합니다.

- **Performance Highlights**: DSES 알고리즘은 ModelNet40 벤치마크에서 부분-풀 점 구름 등록에서 기존의 최첨단 방법을 능가하는 성능을 보였습니다. 특히, 이 알고리즘은 실제 로봇 문제에서도 높은 성능과 강건성을 입증하였습니다. DSES는 일반적인 로봇 어플리케이션에서 매우 널리 사용되는 점 구름 등록 문제가 얼마나 효율적으로 해결될 수 있는지를 보여줍니다.



### Mobile Robot Navigation Using Hand-Drawn Maps: A Vision Language Model Approach (https://arxiv.org/abs/2502.00114)
Comments:
          8 pages, 8 figures

- **What's New**: 본 논문에서는 Hand-drawn Map Navigation (HAM-Nav) 아키텍처를 소개하며, 이는 자연스러운 로봇 내비게이션을 위해 사전 훈련된 Vision Language Models (VLMs)를 활용합니다. HAM-Nav는 다양한 환경, 손으로 그린 스타일, 로봇 형태에 걸쳐 작업할 수 있으며, 지도의 정확성 문제가 있어도 작동합니다. 이를 통해 사용자들이 손으로 그린 지도를 사용할 수 있는 방법을 제시하고 있습니다.

- **Technical Details**: HAM-Nav는 Selective Visual Association Prompting 접근 방식을 통해 토포로지(Topology) 기반 위치 추정 및 내비게이션 계획을 실행합니다. 또한 Predictive Navigation Plan Parser를 통합하여 누락된 랜드마크를 추론합니다. 이는 복잡한 지도에서의 로봇 내비게이션 문제를 해결하는 데 도움이 됩니다.

- **Performance Highlights**: 포토리얼리스틱하게 시뮬레이션된 환경에서 다양한 실험을 수행하였으며, 휠 로봇과 다리 로봇을 모두 사용하여 HAM-Nav의 내비게이션 성공률 및 경로 길이에 따른 성공률(Success weighted by Path Length)에서의 효과성을 입증하였습니다. 또한 실제 환경에서의 사용자 연구 결과, 로봇 내비게이션을 위한 손으로 그린 지도 사용의 실용성이 강조되었습니다.



### Advanced Assessment of Stroke in Retinal Fundus Imaging with Deep Multi-view Learning (https://arxiv.org/abs/2502.00079)
- **What's New**: 이 연구는 첫 번째로 다중 뷰 입력으로 좌우 눈에서 촬영한 망막 촬영 이미지를 사용하여 뇌졸중(stroke)과 일과성 허혈 발작(transient ischemic attack, TIA)을 구별하는 딥 멀티뷰 학습 접근법을 제안합니다. 기존의 연구와 달리, 이 논문은 망막 기반 이미지를 통해 뇌졸중과 TIA를 정확하게 식별하는 최신 솔루션을 제시합니다. 제안된 다중 뷰 뇌졸중 네트워크(MVS-Net)는 두 개의 눈에서 포착된 이미지를 활용하여 더 포괄적인 진단을 가능하게 합니다.

- **Technical Details**: MVS-Net은 네 개의 입력 레이어로 구성되어 있으며, 각 레이어는 좌우 눈의 망막 중앙(macula-centered) 및 시신경유두(optic nerve head-centered) 뷰를 각각 처리합니다. 연구에서 제안한 MVS-Net은 뇌졸중 및 TIA 환자 73명과 건강한 대조군 121명의 망막 사진을 포함하는 Stroke-Data 데이터셋에서 평가되었습니다. 본 연구는 좌우 눈의 망막 이미지를 동시에 사용하는 접근 방식을 도입하여 뇌졸중 진단의 표준화를 지원합니다.

- **Performance Highlights**: 실험 결과, 제안된 MVS-Net은 뇌졸중과 TIA 탐지에서 0.84의 AUC 점수를 달성하여 기존 방법들보다 높은 정확도를 보여주었습니다. 이 연구는 망막 이미지를 기반으로 하는 진단의 잠재력을 발견하고, 임상 환경에서의 실제 적용 가능성을 높입니다. 전체적으로 MVS-Net은 뇌졸중 진단의 신뢰성을 높이는 유망한 기법으로 평가됩니다.



### Deep Ensembling with Multimodal Image Fusion for Efficient Classification of Lung Cancer (https://arxiv.org/abs/2502.00078)
- **What's New**: 본 연구는 다중 모드 폐 이미지에서 암 조직과 건강한 조직을 분류하는 새로운 네트워크, Deep Ensembled Multimodal Fusion (DEMF)을 소개합니다. 이 네트워크는 Computed Tomography (CT)와 Positron Emission Tomography (PET) 이미지를 결합하는 데 Principal Component Analysis (PCA)와 Autoencoder를 활용합니다. 데이터 부족 문제를 극복하기 위해 랜덤 이미지 증강 전략을 사용하여 훈련이 진행됩니다. 최종적으로는 명확한 분류를 위해 투표 기반의 앙상블 분류기를 도입하여 성능을 향상시킵니다.

- **Technical Details**: DEMF 네트워크의 구성 요소는 CT와 PET 이미지의 특징을 축소하고 융합하는 방식으로 처음부터 진행됩니다. PCA를 통해 각 모드의 주요 성분을 선택하고, Autoencoder를 통해 통합된 이미지 데이터를 학습하여 재구성합니다. 이 과정은 주 성분을 기반으로 이미지 융합을 수행하며, 20개의 주 성분을 이용해 128x128 크기의 CT와 PET 이미지를 연결합니다.

- **Performance Highlights**: DEMF 네트워크는 세 가지 공개 데이터 세트에서 비교 평가되어 우수한 성능을 입증하였습니다. 주요 지표로는 Accuracy, F1-Score, Precision 및 Recall이 포함되며, 이 모두에서 기존의 최첨단 네트워크를 초과했습니다. 그 결과는 제안된 네트워크의 효과성을 강조하며, 암 진단의 정확도 향상에 기여합니다.



### LSU-Net: Lightweight Automatic Organs Segmentation Network For Medical Images (https://arxiv.org/abs/2502.00042)
Comments:
          5 pages, 3 figures, 4 tables. Accepted at ICASSP 2025

- **What's New**: LSU-Net은 의료 영상 분할을 위한 새로운 경량 모델로, Light Conv Block과 Tokenized Shift Block을 통합하여 개발되었습니다. 이 모델은 동적 가중치 다중 손실 설계를 통해 효율적인 가중치 배분을 구현하며, 기존의 UNet 구조를 유지하면서도 전체 네트워크 복잡성을 줄이고 있습니다. LSU-Net는 적은 파라미터 수로 강력한 분할 성능을 유지하면서 다중 스케일 지도 학습을 수행할 수 있도록 설계되었습니다.

- **Technical Details**: LSU-Net은 U자형 구조를 기반으로 하며, Encoder Light Conv Block과 Tokenized Shift Down Block을 포함한 인코더와 Decoder Light Conv Block 및 Tokenized Shift Up Block으로 구성됩니다. Light Conv Block은 표준 합성과 깊이별 분리 합성을 결합하여 낮은 파라미터 수로 특징을 추출하며, Tokenized Shift Block은 Spatial Shift Block과 깊이별 분리 합성을 통해 깊은 특징을 최적화합니다. Multi-scale Deep Loss (MDL) 손실 함수를 사용하여 학습 과정의 안정성을 높이고 있습니다.

- **Performance Highlights**: UWMGI와 MSD Colon 데이터셋에서 LSU-Net의 성능을 검증한 결과, 기존 최첨단 분할 아키텍처들보다 우수한 성능을 기록했습니다. 다양한 실험을 통해, LSU-Net은 적은 수의 파라미터로도 높은 정확도를 유지하며, 실용적인 의료 환경에서도 적용 가능성을 보여주었습니다. 이러한 점에서 LSU-Net은 경량화된 성능과 향상된 훈련 안정성을 결합한 중요한 접근법으로 연구되고 있습니다.



New uploads on arXiv(cs.AI)

### Anytime Incremental $\rho$POMDP Planning in Continuous Spaces (https://arxiv.org/abs/2502.02549)
Comments:
          Submitted to IJCAI 2025

- **What's New**: 본 논문은 ρPOMDP를 위한 새로운 해결책인 ρPOMCPOW를 제안합니다. 이 알고리즘은 믿음 표현을 동적으로 정제하여 믿음 의존 보상을 계산하는 데 필요한 계산의 비용을 줄입니다. 또한, ρPOMDP의 지속적인 공간에서 이를 강화하는 데 필요한 공식적인 개선 보장도 포함되어 있습니다.

- **Technical Details**: ρPOMDP(부분 관찰 마르코프 결정 과정)는 불확실한 환경에서 의사 결정을 수행하는 데 강력한 틀을 제공합니다. 전통적인 온라인 ρPOMDP 솔버들은 고정된 믿음 표현에 의존하기 때문에 불확실성에 대한 유연한 접근이 어려웠습니다. ρPOMCPOW는 이러한 문제를 해결하기 위해 믿음 표현을 점진적으로 개선하며, 새로운 점진적 계산 방법을 도입하여 계산 비용을 대폭 낮춥니다.

- **Performance Highlights**: 실험 결과 ρPOMCPOW는 효율성과 솔루션 품질 모두에서 최신 솔버들을 초월하는 성능을 보였습니다. 알고리즘은 특히 정보 수집과 같은 작업에서 매우 높은 효율성을 나타내며, 컴퓨터 성능을 최적화하는 알고리즘으로 주목받고 있습니다.



### Towards graph neural networks for provably solving convex optimization problems (https://arxiv.org/abs/2502.02446)
- **What's New**: 이번 연구는 메시지 패싱 그래프 신경망(MPNN)을 이용하여 볼록 최적화 문제를 해결하는 새로운 반복적 프레임워크를 제안합니다. 이 프레임워크는 기존 MPNN 접근 방식의 한계를 극복하고, 특히 볼록 최적화 설정에서 해결 가능성(feasibility) 보장 기능을 제공합니다. 또한, SVM과 같은 관련 문제를 위한 이론적 근거를 마련했습니다.

- **Technical Details**: MPNN은 이론적으로 표준 내부 점 방법(internal-point methods)을 시뮬레이션하여 선형 제약 조건을 가진 이차 문제(quadratic problems)를 해결할 수 있음을 입증했습니다. 이 프레임워크의 주요 특징은 최초의 적합한 지점(feasible point)에서 시작하여 문제의 적합 영역(feasible region) 내에서 검색을 제한하는 변형을 도입한 것입니다. 이는 솔루션의 품질 및 해결 가능성을 향상시키는 데 기여합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 기존 신경망 베이스라인에 비해 솔루션의 품질과 해결 가능성 측면에서 우수한 성능을 보였습니다. 또한, 다양한 문제 크기에 대해 잘 일반화되며, 경우에 따라 Gurobi와 같은 최첨단 솔버(state-of-the-art solvers)보다 더 빠른 솔루션 시간을 달성합니다.



### A Minimax Approach to Ad Hoc Teamwork (https://arxiv.org/abs/2502.02377)
Comments:
          Accepted at AAMAS 2025

- **What's New**: 이번 연구에서는 Ad Hoc Teamwork (AHT)에 대한 새로운 접근법을 제안하며, 이에 따라 협력자의 불확실성을 명시적으로 고려하는 minimax-Bayes 기법을 활용합니다. 기존의 방법들은 특정한 파트너 분포를 가정하여 성능을 최적화하지만, 본 연구는 최악의 성능 보장을 개선하는 데 중점을 둡니다. 이를 통해 AHT에서의 강건성을 높이는 데 필요한 훈련 분포 선택의 중요성을 강조하고 있습니다.

- **Technical Details**: 제안된 방법인 Minimax-Bayes Reinforcement Learning (MBRL)에서는 파트너에 대한 불확실성을 고려하여 AHT 환경을 모델링합니다. 또한, 유틸리티(utility)와 후회(regret) 메트릭스를 사용하여 AHT 강건성을 제고하는 방법을 모색하며, 두 메트릭에 맞추어 해결책을 제공합니다. Gradient Descent-Ascent (GDA) 알고리즘을 사용하며, 이와 관련된 정책 기울기 방법의 수렴성에 대해서도 논의합니다.

- **Performance Highlights**: 실험 결과, 본 접근법은 자가 대결(self-play), 허구적 대결(fictitious play) 및 고정된 분포에 대한 학습과 비교할 때, 간단한 RL 조정 작업과 복잡한 작업 모두에서 최고의 강건한 솔루션을 도출하고 있음을 보여주었습니다. 특히, 보이지 않는 시나리오에서의 성능 평가를 통해 협력적인 문제 해결에서의 효과를 입증하였습니다. 이러한 성과는 기존 방법들에 비해 상당한 개선을 나타냅니다.



### The Elicitation Game: Evaluating Capability Elicitation Techniques (https://arxiv.org/abs/2502.02180)
- **What's New**: 본 논문에서는 AI 시스템의 잠재적 능력을 분석하고 평가하는 방법에 대한 연구를 소개합니다. 특히, 잠금 비밀번호를 사용하여 숨겨진 능력을 가진 언어 모델을 모델 유기체(model organisms)로 훈련시키는 새로운 접근 방식을 제안합니다. 이는 전통적인 비밀번호 잠금 모델보다 더 강력한 기법으로 밝혀졌습니다.

- **Technical Details**: 연구에서는 MCQA(multiple-choice question-answering)와 코드 생성 태스크를 통해 능력 유도 기법의 효과를 평가합니다. 키 기술로는 'prompting', 'steering', 그리고 'fine-tuning'이 사용되며, 그 중 fine-tuning 방법이 잠재 능력을 유도하는 데 가장 적합함을 강조합니다. 특히, circuit-breaking 방법을 도입하여 모델의 보안을 강화하며 이에 대한 효과성을 비교하고 분석합니다.

- **Performance Highlights**: 결과적으로, prompting 기법은 비밀번호 잠금 모델과 서킷 브레이킹 모델 모두에 대해 MCQA 능력을 이끌어낼 수 있지만, 서킷 브레이킹 모델에 대해서는 효과가 떨어지는 것으로 나타났습니다. 코드 생성 태스크에서는 fine-tuning 기법만이 새로운 모델 유기체의 숨겨진 능력을 유도할 수 있는 것으로 밝혀졌으며, 특히 anti-refusal training 기법이 중요한 역할을 했다. 이러한 연구 결과는 효과적인 능력 평가의 신뢰성을 높이기 위한 방안을 제시합니다.



### Vulnerability Mitigation for Safety-Aligned Language Models via Debiasing (https://arxiv.org/abs/2502.02153)
Comments:
          37 pages

- **What's New**: 이 논문은 AI의 안전 정렬(safety alignment) 문제를 다루며, 기존의 방법들이 특정 카테고리의 안전성을 보장하는 데 실패함을 보여줍니다. 저자들은 다양한 모델을 평가한 결과, 일반적으로 안전성을 향상시켰지만 특정 취약점을 제거하기가 어렵다는 점을 강조합니다. 새로운 방법인 Token-level Safety-Debiased Inference (TSDI)를 소개하여 안전성 편향을 추정하고 수정하는 과정을 제안합니다.

- **Technical Details**: 이 연구는 대형 언어 모델(LLM)의 안전성 및 유용성 간의 트레이드오프를 연구합니다. 기존의 방법들은 단일 보상 지표에 의존하는 경향이 있으며, 이는 다양한 안전성 요구를 충족하지 못하는 문제를 초래합니다. 특히, 보상 모델링 및 인간 피드백을 통한 강화 학습(RLHF) 적용 시 제약을 걸고 안전한 LLM을 개발하는 새로운 접근법을 제공합니다.

- **Performance Highlights**: TSDI 방법을 사용하여 모델의 유용성을 향상시키면서도 안전성을 유지하는 결과를 보였습니다. 실험 결과, TSDI는 생성 과정에서 안전성 편향을 수정하여 더 높은 수준의 유용성을 가져오면서도 위험 요소를 최소화하는 데 기여합니다. 이는 안전성과 유용성을 동시에 높이는 최적의 트레이드오프를 달성하는 데 중요한 성과로 여겨집니다.



### Risk-Aware Driving Scenario Analysis with Large Language Models (https://arxiv.org/abs/2502.02145)
Comments:
          IEEE Intelligent Vehicles Symposium 2025

- **What's New**: 이번 논문은 Large Language Models (LLMs)를 활용하여 자율주행 시스템에서 생성되는 운전 시나리오의 위험 인식 분석(risk-aware analysis)을 위한 새로운 프레임워크를 제안합니다. LLMs의 강력한 정보 처리 능력을 통해 자율주행 시뮬레이터에서 생성된 시나리오의 안전성을 평가할 수 있는 가능성을 탐구하고 있습니다. 논문에서는 비판적인 안전성(safety-critical)을 평가하기 위한 LLMs의 유용성을 검증하는 실증적 평가를 수행하였습니다.

- **Technical Details**: LLMs의 능력을 활용하여, 기존의 비위험한(non-critical) 시나리오를 수정하여 새로운 안전 비판적 시나리오를 생성하는 적대적 방법(adversarial method)을 사용하는 프레임워크를 설계하였습니다. 이 과정에서 자율주행 테스트 시뮬레이터가 생성한 다양한 운전 시나리오를 평가하고, 이 시나리오들이 안전성과 관련하여 얼마나 효과적인지를 분석합니다. 이러한 방법론은 또한 운동 계획 알고리즘의 유효성을 검증하는 데 활용될 수 있습니다.

- **Performance Highlights**: 실험 결과, LLMs는 자율주행 시뮬레이터가 생성한 시나리오의 안전성을 평가하는 데 매우 효과적임을 보여주었습니다. 이 연구는 자율주행 시스템의 안전성을 높이는 데 기여할 수 있는 가능성을 제시하며, 새로운 테스트(Object)에 대한 피드백을 제공하는 방식으로, 기술의 적용 가능성을 넓히고 있습니다. 연구에 사용된 코드와 시나리오는 제공된 링크에서 확인할 수 있습니다.



### Standard Neural Computation Alone Is Insufficient for Logical Intelligenc (https://arxiv.org/abs/2502.02135)
- **What's New**: 이 연구는 기존 신경망 아키텍처가 진정한 논리적 지능을 구현하는 데 한계가 있음을 주장합니다. 전통적인 신경계열(computation) 및 비선형 활성화(nonlinear activation)에 의존하는 현대 AI 모델은 패턴을 근사하는 데 효율적이나, 논리적 일관성(logical consistency)을 위한 구조적 보장이 부족합니다. 이에 따라 본 논문은 Logical Neural Units (LNU)라는 모듈 구성 요소를 제안하며, 이를 통해 신경망 아키텍처 내에 논리 연산을 차별적으로 포함할 수 있도록 합니다.

- **Technical Details**: LNU는 AND, OR, NOT과 같은 기본 논리 연산의 차별적 근사를 직접 포함하여 신경망과 통합하는 새로운 방식입니다. 기존의 신경 상징(neurosymbolic) 접근 방식과 달리, LNU는 외부의 논리 엔진이나 느슨하게 통합된 추론 모듈을 필요로 하지 않고, 하위 기호(sub-symbolic) 학습 과정에 의도적으로 논리적 추론(logical inference)을 통합합니다. 이 작업은 구조화된 추론(structured reasoning)과 딥 러닝 아키텍처의 결합을 구체화합니다.

- **Performance Highlights**: LNU는 고전적 다중 단계 추론(multi-step reasoning) 접근 방식을 통합하여 신경망의 성능을 향상시킵니다. 본 논문의 요구 사항에 따르면, LNU는 신뢰할 수 있는 논리적 추론을 위한 명시적인 구조적 통합을 제공하여 확률적 학습(statistical learning)과 규칙 기반 추론(rule-based reasoning) 간의 격차를 메꾸는 데 기여할 것입니다. 이러한 새로운 아키텍처는 미래 AI 연구의 주요 방향을 제시하며, 추가적인 구현 및 연구 과제를 나열합니다.



### CH-MARL: Constrained Hierarchical Multiagent Reinforcement Learning for Sustainable Maritime Logistics (https://arxiv.org/abs/2502.02060)
- **What's New**: 이번 연구에서는 CH-MARL(Constrained Hierarchical Multiagent Reinforcement Learning)이라는 새로운 프레임워크를 제안하였습니다. 이 프레임워크는 수치적 제약 조건을 실시간으로 강제하고, 공정한 보상 체계를 통해 자원의 공평한 분배를 도모합니다. CH-MARL은 해양 물류 환경에서의 실험을 통해 배출량 감소와 공정성 향상, 운영 효율성 개선을 입증하였습니다. 이 연구는 또한 다수의 에이전트 조정 문제를 해결하는 확장 가능하고 일반화된 솔루션을 제공합니다.

- **Technical Details**: CH-MARL은 계층적 의사결정과 동적 제약 사항 이행을 통합한 다중 에이전트 강화 학습의 새로운 프레임워크입니다. 이 시스템은 글로벌 배출 한도를 준수하기 위한 실시간 제약 이행 레이어를 포함하며, 공정성 지표를 통합하여 이해관계자 간의 자원 분배를 공평하게 합니다. 실험 환경에서는 디지털 트윈을 활용하여 향상된 운영 효율성과 배출감소 효과를 확인하였습니다.

- **Performance Highlights**: 이 연구는 CH-MARL이 해양 물류에서 배출효율성을 높이며, 이해관계자 간의 공정성을 증진시키고 운영 효율성을 향상시킨다는 점을 강조합니다. 특히, CH-MARL은 복잡한 다중 에이전트 환경에서 지속 가능성과 공정성을 동시에 추구하는 데 강력한 효과를 발휘합니다. 기존의 MARL 모델의 한계를 넘어, 이 프레임워크는 실제 해양 환경에서 강력하고 확장 가능한 솔루션으로 자리매김할 것입니다.



### Building a Cognitive Twin Using a Distributed Cognitive System and an Evolution Strategy (https://arxiv.org/abs/2502.01834)
Comments:
          first submitted on 09/22/2022, published on 01/20/2025

- **What's New**: 이 연구는 사용자 행동을 모델링하기 위해 입력과 출력 훈련 및 진화 전략을 사용하는 Interaction-based Cognitive Twins 구축 기술을 제시합니다. 이 과정에서 간단한 물리적 및 가상 장치를 조율하여 개인의 상호작용 행동을 근사할 수 있는 방법을 보여주고, 생성된 Cognitive Twin은 미래에 다양한 자동화 작업이나 인간 유사 인공지능 생성에 활용될 수 있습니다.

- **Technical Details**: 본 논문에서는 DCT(Distributed Cognitive Architectures)라는 도구를 사용하여 디지털 분산 인지 에이전트를 구성하는 방법을 제안합니다. 진화 알고리즘과 전통적인 머신러닝 기법을 결합하여, 사용자의 잠재적 행동을 에뮬레이션할 수 있는 능력을 지닌 에이전트를 구축합니다. 이 시스템은 구조화된 메모리와 코드렛(codelets) 기반으로 운영되며, 다양한 상호작용 방식을 탐색하는 것이 가능합니다.

- **Performance Highlights**: 연구 결과, 제안된 Cognitive Twin 방식은 높은 수준의 성능 지표를 달성했습니다. 이는 다수의 장치를 통해 대규모 병렬 처리를 수행하는 인지 시스템 구축의 이점을 잘 보여주며, 다양한 어플리케이션에서 지능적인 인공지능 에이전트를 구축하는 데 기여할 것으로 기대됩니다. 미래 연구 방향으로는 이 모델의 자동화 및 더 고차원적인 인지 행동 분석이 포함됩니다.



### An Agentic AI Workflow for Detecting Cognitive Concerns in Real-world Data (https://arxiv.org/abs/2502.01789)
- **What's New**: 이 연구에서는 LLaMA 3 8B를 사용하여 임상 노트에서 인지 문제를 식별하기 위한 완전 자동화된 다중 에이전트 AI 워크플로우를 개발하고 검증했습니다. 이 워크플로우는 3,338개의 클리닉 노트를 분석하였으며, 에이전트 간의 동적 협업을 통해 의미 있는 통찰을 추출합니다. 새로운 접근 방식은 전문가 주도의 기준선과 비교되었습니다.

- **Technical Details**: 이 연구는 다중 에이전트(Agentic) 워크플로우를 구성하여 임상 노트로부터 인지 문제를 효율적으로 식별하는 데 중점을 두었습니다. 두 가지 워크플로우 모두 높은 분류 성능(F1-score 0.90 및 0.91)을 달성했으며, 에이전트 기반 워크플로우는 더 높은 특이도(specificity)인 1.00을 기록했습니다. 또한, 수정이 필요한 반복 횟수도 줄어들었습니다.

- **Performance Highlights**: 검증 데이터에서는 두 워크플로우 모두 성능 저하를 보였지만, 에이전트 워크플로우는 완벽한 특이도를 유지했습니다. 이러한 결과는 완전 자동화된 다중 에이전트 AI 워크플로우가 전문가 수준의 정확도를 달성하면서도 높은 효율성과 비용 효과성을 제공할 수 있는 가능성을 보여줍니다. 이는 클리닉 환경에서 인지 문제를 감지하는 데 있어 확장 가능한 솔루션이 될 수 있습니다.



### Metastable Dynamics of Chain-of-Thought Reasoning: Provable Benefits of Search, RL and Distillation (https://arxiv.org/abs/2502.01694)
Comments:
          55 pages, 3 figures

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 추론 능력을 향상시키기 위해, 추론 시간에 더 많은 계산을 할당하는 것이 어떻게 중요한 패러다임이 되는지 분석합니다. 특히, 체인 오브 사고(Chain of Thought, CoT) 생성을 메타안정 마르코프 과정으로 간주하며, 쉽게 해결되는 추론 단계는 밀접하게 연결된 클러스터를 형성하고, 어려운 단계는 희소한 연결을 만들어내는 방식을 제안합니다. 이 접근을 통해 드러난 인사이트는, 희소한 연결에 대한 보상을 주는 검색 프로토콜을 구현함으로써 CoT 생성의 질이 개선된다는 것입니다.

- **Technical Details**: CoT 생성은 여러 단계로 이루어진 추론 과정을 표현하며, 각 단계는 논리적 주장을 나타냅니다. 이 과정에서 쉬운 추론 단계는 밀접한 클러스터를 형성하고, 어려운 추론 단계는 희소한 연결로 표현됩니다. 이러한 구조를 바탕으로 우리는 기대 도달 시간을 정량적으로 분석하고 메타안정 동역학을 연구합니다. 이전의 연구와 이론적 기초를 활용하여, CoT 모델에 대한 구체적인 성능 분석과 최적화 보장을 제공합니다.

- **Performance Highlights**: 이 연구에서 발견된 주요 성과는 추론 시간의 검색이 모델의 추론 능력을 증가시키는 것입니다. 이는 특정 논리적 단계를 개선하기 위해 RL(강화 학습) 기법을 활용할 수 있음을 보여줍니다. 또한, CoT의 압축된 메타안정 표현을 작은 모델로 증류하여 효율적으로 추론 동역학을 표현할 수 있음을 입증했습니다. 마지막으로, path-finding 작업에서 대규모 테스트 시간 컴퓨팅이 필요함을 새로운 통계적 쿼리 복잡성을 도입하여 입증했습니다.



### Automated Extraction of Spatio-Semantic Graphs for Identifying Cognitive Impairmen (https://arxiv.org/abs/2502.01685)
Comments:
          To appear in ICASSP 2025

- **What's New**: 본 연구에서는 기존의 수작업으로 진행되던 Content Information Units (CIUs) 태깅 작업을 자동화하여, Cookie Theft 그림을 기반으로 한 spatio-semantic graph의 자동 추출 방법을 제안합니다. 이 자동화된 접근법은 인지 손상 평가에서 시각적 의미 경로를 자동적으로 특성화할 수 있는 가능성을 제시합니다. 연구 결과, 자동으로 생성된 spatio-semantic graph가 인지 손상 유무를 효과적으로 구분할 수 있음을 확인하였습니다.

- **Technical Details**: 이 연구에서 사용된 spatio-semantic graph는 CIUs의 위치와 시간 순서를 시각적으로 표현하는 그래프 이론적 구조입니다. 각 CIU는 Cookie Theft 그림의 픽셀 좌표에 매핑되며, 이 과정을 통해 시각적 경로를 구성합니다. 연구팀은 NetworkX 툴킷을 사용해 이러한 좌표들을 기반으로 노드와 엣지를 구성하여 시각적 의미 경로를 시각화합니다.

- **Performance Highlights**: 자동으로 생성된 spatio-semantic graph는 인지 손상이 있는 화자와 없는 화자를 구별하는 데 있어 유의미한 차이를 보였습니다. 통계적 분석 결과, 자동화된 방법으로 유도된 특징들이 수작업 방법과 유사한 결과를 생성했으나, 임상 집단 간 차이를 더 뚜렷하게 나타내는 것으로 확인되었습니다. 이는 자동화된 접근법이 인지 손상 평가를 위한 임상적 언어 모델 개발에 크게 기여할 수 있음을 시사합니다.



### QLASS: Boosting Language Agent Inference via Q-Guided Stepwise Search (https://arxiv.org/abs/2502.02584)
- **What's New**: QLASS (Q-guided Language Agent Stepwise Search)는 오픈 언어 에이전트를 위한 단계별 Q-값 추정으로 주석을 자동 생성하여 중간 지침을 제공하는 혁신적인 방법입니다. 이 모델은 행위의 과정 보상(process reward) 모델링을 도입하여 언어 에이전트가 장기적인 가치에 더 잘 적응하도록 돕고, 더 나은 성능 향상을 가져옵니다. QLASS는 거의 절반의 주석 데이터로도 강력한 성능을 유지할 수 있으며, 제한된 감독 환경에서도 효율적으로 작동합니다.

- **Technical Details**: QLASS는 탐색 트리(exploration tree)를 설정하여 스텝별 Q-값을 추정하는 방식을 통해 초기 대화 경로에 대한 중간 보상을 제공합니다. 이 과정에서 Bellman 방정식을 통해 의사결정이 장기적인 보상에 어떻게 기여하는지를 이해하고, 희소하거나 지연된 피드백 신호에 대한 의존성을 줄입니다. Q-값에 대한 추정 결과로 생성된 모델(QNet)은 부분 솔루션의 예상 수익을 예측하여 언어 에이전트의 의사결정을 더 정교하게 조정하게 만듭니다.

- **Performance Highlights**: QLASS는 WebShop, ALFWorld 및 SciWorld와 같은 다양한 에이전트 환경에서 실행되는 실험에서 강력한 성능을 입증하였습니다. 이 모델은 제한된 감독 상황에서도 큰 성과를 보여주었으며, 대규모 텍스트 데이터 없이 소규모 데이터 세트로도 효과적으로 의사결정을 할 수 있습니다. 결과적으로 QLASS는 언어 에이전트의 전반적인 성능 개선에 기여할 수 있는 실용적이고 혁신적인 접근방식으로 자리 잡고 있습니다.



### Are Language Models Up to Sequential Optimization Problems? From Evaluation to a Hegelian-Inspired Enhancemen (https://arxiv.org/abs/2502.02573)
- **What's New**: 이 논문에서는 Large Language Models (LLMs)가 Sequential Optimization Problems (SOPs)을 처리하는 능력을 탐구합니다. 새로운 동적 프레임워크인 WorldGen을 소개하여 LLM 성능을 평가하기 위한 간단하면서도 효과적인 방법을 제시합니다. 초기 관찰에 따르면, LLM은 간단한 SOP에서는 잘 작동하지만 복잡성이 증가함에 따라 성능이 크게 저하되는 것으로 나타났습니다. 이러한 문제를 해결하기 위해 Hegelian Dialectics에서 영감을 받아 ACE라는 새로운 접근 방식을 제안합니다.

- **Technical Details**: WorldGen은 복잡성을 제어할 수 있는 새로운 SOP를 생성하는 능력을 갖춘 프레임워크입니다. 이 프레임워크는 기존의 정적 벤치마크와 달리 LLM의 발전에 맞춰 평가 복잡성을 증가시킬 수 있도록 설계되었습니다. 또한, 이 논문에서는 LLM의 성능 저하 문제를 해결하기 위해 LLM을 블랙 박스처럼 다루며, 추가적인 재교육이나 미세 조정 없이 성능을 향상시키는 방법을 제안합니다. Hegelian Dialectics의 구조적 접근법을 통해 SOP의 문제를 다루는 LLM의 가능성을 제시합니다.

- **Performance Highlights**: 실험 결과, 단순한 최적화 문제에서는 현재 LLM이 효율적으로 문제를 해결할 수 있음을 확인했습니다. 그러나 최적화 문제의 복잡성이 증가함에 따라 LLM의 성능이 만족스럽지 않게 저하되는 경향이 있음을 관찰했습니다. ACE 프레임워크를 통해 이러한 성능을 크게 향상시킬 수 있었으며, 이는 기존의 LLM들이 SOP에서 더욱 효율적으로 작업할 수 있게 해줍니다. 이 논문은 LLM의 향후 발전에 있어 철학적 접근이 중요한 역할을 할 수 있음을 강조합니다.



### Fairness in Survival Analysis: A Novel Conditional Mutual Information Augmentation Approach (https://arxiv.org/abs/2502.02567)
- **What's New**: 이번 논문에서는 생존 분석(survival analysis)에서의 공정성(fairness) 문제를 다루고 있습니다. 기존의 방법들이 사전 정의된 평가 시간대에서의 예측 공정성을 간과한 점을 지적하며, 새로운 공정성 개념인 equalized odds (EO)를 소개합니다. 이 EO 공정성은 특정 시간대에서의 예측 공정성을 강조합니다.

- **Technical Details**: 우리는 Conditional Mutual Information Augmentation (CMIA) 방식을 제안하여 생존 분석에서 EO 공정성을 달성하고자 합니다. CMIA는 조건부 상호 정보(conditional mutual information)를 기반으로 한 새로운 공정성 정규화 항과 혁신적인 검열 데이터 증강(censored data augmentation) 기법을 특징으로 합니다. 이 접근법은 예측 정확도(prediction accuracy)와 공정성(fairness)을 효과적으로 조화롭게 합니다.

- **Performance Highlights**: CMIA 접근법은 세 가지 다양한 응용 분야에서 여러 최첨단 방법들과 비교 평가되었고, CMIA가 예측 격차(prediction disparity)를 일관되게 줄이면서도 우수한 정확도를 유지하며 다른 경쟁 방법들에 비해 월등한 성능을 보임을 입증하였습니다. 여러 데이터셋과 생존 모델(예: 선형 COX, 심층 AFT)에서 우수한 결과를 나타냈습니다.



### Learning the RoPEs: Better 2D and 3D Position Encodings with STRING (https://arxiv.org/abs/2502.02562)
Comments:
          Videos of STRING-based robotics controllers can be found here: this https URL

- **What's New**: 본 논문은 STRING: Separable Translationally Invariant Position Encodings를 소개합니다. STRING은 최근에 제안된 Rotary Position Encodings를 확장하여, 일반적인 이론 프레임워크를 통해 접근합니다. STRING은 임의 차원의 토큰 좌표에 대해서도 정확한 translation invariance를 제공하며, 이는 로봇 공학 분야에서 효율적인 3D 토큰 표현을 가능하게 합니다.

- **Technical Details**: STRING은 Lie 그룹에 기반하여 설계되었습니다. 이는 이전의 RoPE를 통합하는 일반화된 알고리즘으로, 두 개의 쿼리와 키의 상대 회전 각도가 오직 그들의 위치 차이에만 의존합니다. 이러한 독립적인 변환은 트랜스포머의 반복 처리 중 다시 계산할 필요가 없어 KV-caching을 용이하게 하며, 이를 통해 성능을 더욱 향상시킵니다.

- **Performance Highlights**: STRING은 RGB(-D) 입력을 사용하는 비전 트랜스포머(Vision Transformers)에 성공적으로 통합되었습니다. 이를 통해 open-vocabulary object detection과 다양한 로봇 컨트롤러에서 실질적인 성과를 보여줍니다. 실험 결과는 STRING이 로봇 공학 분야에서의 실제 활용 가능성을 강조합니다.



### Decision Theoretic Foundations for Conformal Prediction: Optimal Uncertainty Quantification for Risk-Averse Agents (https://arxiv.org/abs/2502.02561)
- **What's New**: 이 논문은 위험 회피(risk-averse) 의사 결정자가 예측의 불확실성을 정량화하는 방법을 탐구합니다. 연구진은 예측 집합(prediction sets)을 사용하여 위험을 최소화하는 의사 결정 이론적 기초를 개발하였으며, 이를 통해 더욱 신뢰할 수 있는 의사 결정 방법을 제안하고 있습니다. 이 논문은 의학, 금융, 로봇공학 등 위험에 민감한 분야에서의 실제 적용 가능성에 중점을 두고 있습니다.

- **Technical Details**: 논문은 세 가지 기본 질문에 답하며, 최적의 불확실성 정량화 개념을 제시합니다. 첫째, 위험 회피 의사 결정자가 예측 집합을 최적의 선택으로 사용함을 증명합니다. 둘째, 위험 회피 의사 결정자가 예측 집합을 행동으로 변환하는 최적 정책을 찾으며, 단순한 max-min 결정 정책이 효과적임을 보여줍니다. 마지막으로, 이 연구는 이러한 의사 결정자를 위한 최적 예측 집합을 도출하는 방법을 제시합니다.

- **Performance Highlights**: 실험을 통해 Risk-Averse Calibration (RAC) 알고리즘이 의료 진단 및 추천 시스템과 같은 분야에서 상당한 지원을 제공함을 입증하였습니다. RAC는 안전성과 유용성 간의 균형을 개선하여 기존 방법보다 높은 유용성을 제공하며, 사용자가 설정한 위험 한계를 준수하여 안전을 보장합니다. 특히, RAC는 불확실성을 효과적으로 대처하는 데 있어 뛰어난 성능을 보였습니다.



### Addressing Label Shift in Distributed Learning via Entropy Regularization (https://arxiv.org/abs/2502.02544)
Comments:
          Accepted at the International Conference on Learning Representations (ICLR 2025)

- **What's New**: 이번 연구에서는 다중 노드 분산 학습에서의 레이블 이동 문제를 해결하기 위한 Versatile Robust Label Shift (VRLS) 방법을 제안하며, 이는 테스트와 훈련 간의 레이블 밀도 비율을 효율적으로 추정하기 위해 샤넌 엔트로피 기반 정규화를 통합합니다. VRLS는 다중 노드 환경에서도 밀도 비율을 학습하고 조정하여 모델 성능을 향상시키는 데 중점을 두고 있습니다. 이러한 접근 방식은 각 노드의 데이터가 공유되지 않는 설정에서 발생하는 레이블 이동 문제를 완화하는 데 효과적입니다.

- **Technical Details**: VRLS는 레이블 이동 조건에서의 정확한 밀도 비율 추정을 위한 새로운 목적 함수를 제안하여 예측의 불확실성 보정을 위한 정규화 항을 포함합니다. 또한, Importance Weighted-ERM (IW-ERM) 프레임워크를 통해 다중 노드 분산 환경에서 다양한 레이블 분포를 가진 여러 노드 간의 전반적인 진정한 위험 최소화를 위한 편향되지 않은 추정을 찾습니다. 이를 통해 각 노드에서의 통계적 이질성을 효과적으로 처리합니다.

- **Performance Highlights**: MNIST, Fashion MNIST, CIFAR-10 데이터셋에 대한 실험 결과, VRLS 방식을 사용한 모델이 레이블 이동 조건에서 기존의 기준 모델보다 최대 20% 높은 성능을 보였습니다. 또한, 연구의 이론적 분석을 통해 추정 오류에 대한 높은 확률 경계를 설정함으로써 VRLS의 효과성을 추가적으로 뒷받침합니다. IW-ERM 프레임워크는 전반적인 테스트 오류를 유의미하게 개선하면서도 기존 ERM 방법에 비해 통신 및 계산 오버헤드를 최소화하는 데 성공하였습니다.



### Flow Q-Learning (https://arxiv.org/abs/2502.02538)
- **What's New**: 이번 논문에서 제시하는 flow Q-learning (FQL)은 간단하고 성능이 뛰어난 오프라인 강화 학습 기법입니다. 이 방법은 데이터에서 임의로 복잡한 행동 분포를 모델링하기 위해 표현력이 풍부한 flow-matching policy를 활용합니다. 기존의 방식과 달리, FQL은 행동 생성 과정을 직접 안내하기보다는 RL을 통해 한 단계의 정책을 학습하여 복잡성을 최소화합니다.

- **Technical Details**: FQL은 반복적 행동 생성 과정의 복잡한 문제를 해결하기 위해, 디렉트한 flow 정책을 사용하는 대신 인간적으로 표현 가능한 간단한 한 단계 정책을 학습합니다. 이를 통해 불안정한 재귀적 역전파(Backpropagation)를 피하고, 테스트 시간에 소모되는 비용을 줄이며 표현력을 유지할 수 있습니다. 이 방법은 오프라인 강화 학습과 오프라인에서 온라인으로의 전환 시나리오 모두에 적용이 가능합니다.

- **Performance Highlights**: FQL은 73개의 도전적인 OGBench 및 D4RL 작업에서 강력한 성능을 발휘하는 것을 실험적으로 입증하였습니다. 이러한 성능 결과는 FQL이 오프라인 강화 학습 환경에서 효율적임을 보여줍니다. 특히, 상태 기반 및 픽셀 기반의 다양한 작업에서 두각을 나타내어, 여러 응용 분야에 비전(vision)과 인사이트를 제공합니다.



### Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies (https://arxiv.org/abs/2502.02533)
Comments:
          11 pages, 7 figures, 1 table (30 pages, 9 figures, 5 tables including references and appendices)

- **What's New**: 본 논문에서는 다중 에이전트 시스템(MAS) 설계를 자동화하기 위한 Multi-Agent System Search(MASS)라는 새로운 최적화 프레임워크를 제안합니다. 기관세가 없이 효율적인 설계 과정을 통해 MAS의 성능 향상에 기여하며, LLM(대형 언어 모델) 기반 에이전트들이 상호작용하며 문제를 해결하는 데 필요한 최적의 프롬프트와 토폴로지를 탐색합니다. 이 시스템은 세 가지 단계로 구성되며, 각 단계는 이전 단계에서 최적화된 프롬프트 및 토폴로지에 조건화 됩니다.

- **Technical Details**: MASS는 블록 수준의 프롬프트 최적화와 워크플로우 토폴로지 최적화, 글로벌 프롬프트 최적화를 조합해 효과적인 에이전트를 설계합니다. MAS 설계에는 블록 수준 디자인과 워크플로우 레벨 조정이 포함되며, 토폴로지 최적화는 각기 다른 에이전트들과 그들의 배열에 대한 결정 과정을 포함합니다. 본 연구는 MAS에서 프롬프트와 토폴로지 디자인이 성능에 미치는 영향에 대한 정량적 분석을 제공합니다.

- **Performance Highlights**: MASS로 최적화된 다중 에이전트 시스템은 기존의 수동 설계된 대안들에 비해 월등히 우수한 성능을 보여주었습니다. 특히, reasoning, multi-hop understanding, code generation 등 다양한 작업에서 상태 최첨단 성능을 달성하였습니다. 연구 결과는 효과적인 다중 에이전트 시스템 구축을 위한 가이드라인을 제공합니다.



### Why human-AI relationships need socioaffective alignmen (https://arxiv.org/abs/2502.02528)
- **What's New**: 이 논문은 인간과 AI 시스템 간의 관계가 점점 더 깊어지고 지속적으로 유지되는 유대감의 형성에 대해 탐구합니다. 특히 AI가 개인화되고 에이전틱(autonomous)하게 변화함에 따라 고도화되는 사회적 및 정서적 정렬 개념을 제시합니다. 이는 인간의 기본 심리 욕구를 지원하고 착취하지 않는 AI 시스템을 만드는 방향으로 나아가야 함을 강조합니다.

- **Technical Details**: AI 정렬(alignment)을 위해서는 단순 기술적 문제를 넘어서 인간과의 복잡한 상호 작용을 고려해야 합니다. 이 논문은 '사교정서적 정렬'(socioaffective alignment)이라는 관점을 통해 AI 시스템이 인간과의 관계에서 어떻게 행동하는지를 연구합니다. 이는 AI 시스템이 인간의 심리적, 사회적 맥락을 어떻게 형성하고 영향을 미치는지를 분석하는 접근법입니다.

- **Performance Highlights**: AI와의 관계가 개인의 자아 인식, 자율성 및 인간 간 관계에 미치는 영향을 다루며, AI 시스템 개발에 대한 새로운 시각을 제시합니다. 논문에서는 소셜 보상과 인간의 뇌 반응을 연구하여 AI와의 관계가 인간의 심리적 및 신체적 건강에 미치는 영향을 조명합니다. 이러한 연구는 AI 시스템의 설계 방식에 있어 인간 중심의 중요한 요소들을 포함해야 한다고 강조합니다.



### Adaptive Exploration for Multi-Reward Multi-Policy Evaluation (https://arxiv.org/abs/2502.02516)
- **What's New**: 이번 연구에서는 온라인 다중 보상(multi-reward) 다중 정책(multi-policy) 할인(discounted) 설정에서 정책 평가(policy evaluation) 문제를 다룹니다. 이는 서로 다른 정책에 대해 여러 보상 함수(reward function)를 동시에 평가해야 하는 새로운 환경입니다. 기존 연구에서는 다루어지지 않았던 $(	heta,	heta)$-PAC 관점에서 높은 신뢰도와 함께 $	heta$-정확한 추정을 도출하는 방법을 탐구합니다.

- **Technical Details**: 우리는 Multi-Reward Best Policy Identification에서의 이전 연구를 바탕으로, 서로 다른 보상 집합(reward sets)에서 여러 정책을 평가하기 위한 샘플 복잡도(sample complexity)를 최소화하도록 MR-NaS(exploration scheme) 탐색 방식(adapt) 을 조정합니다. 이 방법은 특정 인스턴스(instance-specific)에 대한 하한(lower bound)을 활용하여 샘플 복잡도가 가치 편차(value deviation) 측정의 크기에 따라 어떻게 변화하는지를 파악합니다. 비록 이 하한을 계산하는 것이 어려운 비볼록(non-convex) 최적화를 포함하지만, 우리는 유한(finite) 및 볼록(convex) 보상 집합에 대해 유효한 볼록 근사(convex approximation)를 제안합니다.

- **Performance Highlights**: 실험 결과는 이 적응형 탐색 scheme이 표 형태의 데이터(tabular domains)에서 효과적임을 보여줍니다. 다양한 정책을 평가할 수 있는 능력을 갖추고 있어, 정확한 추정을 위한 샘플 수를 크게 줄일 수 있음을 입증합니다. 따라서, 제안된 방법은 다양한 보상 구조에서 정책 평가를 위한 강력한 도구로 자리 잡을 것으로 기대됩니다.



### Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search (https://arxiv.org/abs/2502.02508)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 추론 능력을 증대시키기 위해 검색 기능을 내부화하려는 새로운 접근 방식을 제안합니다. 기존의 방법들은 외부 LLM 검증기를 활용하는 반면, 본 연구에서는 단일 LLM이 복잡한 작업을 처리할 수 있게 하는 방법을 모색합니다. 특히 Chain-of-Action-Thought (COAT) 메커니즘과 두 단계의 훈련 패러다임을 적용하여, 자율 검색 기능을 강화하고자 합니다.

- **Technical Details**: Satori라는 7B LLM은 COAT 추론 형식을 내재화하는 소규모 포맷 조정 단계와 강화 학습 기반의 대규모 자가 개선 단계로 구성된 두 단계의 훈련을 통해 개발됩니다. 이 모델은 개방형 소스 데이터로 훈련되어 있고, 수학적 추론 과제에서 최첨단 성능을 기록하며, 도메인 외 작업에 대한 강력한 일반화 능력을 보입니다. 코드는 완전하게 오픈 소스 형식으로 제공될 예정입니다.

- **Performance Highlights**: Satori는 수학적 추론 작업에 대해 우수한 성능을 보이며, 동일한 기본 모델로 구축된 지침 모델보다 더 높은 성능을 기록했습니다. 또한, Satori는 외부 지침 없이 자율 검색이 가능한 단일 LLM이라는 점에서 효율성을 가지고 있습니다. 연구에서 강조하듯이, 이 모델은 자가 반영 및 자가 탐색 능력이 뛰어난 전반적인 역량을 지니고 있습니다.



### Unified Spatial-Temporal Edge-Enhanced Graph Networks for Pedestrian Trajectory Prediction (https://arxiv.org/abs/2502.02504)
- **What's New**: 이번 연구에서는 보행자 궤적 예측을 위한 새로운 접근 방식인 UniEdge를 제안합니다. UniEdge는 고차원 교차 시간 상호작용을 단순화하여 첫 번째 관계로 통합한 통합된 공간-시간 그래프 데이터 구조를 도입합니다. 이로 인해 다단계 집계 과정에서 정보 손실을 방지하고 즉각적인 반응을 가능하게 만들어 예측 성능을 향상시킵니다.

- **Technical Details**: UniEdge는 Edge-to-Edge-Node-to-Node Graph Convolution(E2E-N2N-GCN)을 기반으로 하는 이중 그래프 네트워크를 통해 보행자 간의 명시적 N2N 사회적 상호작용과 암묵적 E2E 영향 전파를 동시에 모델링합니다. 이러한 구조는 각 보행자의 행동과 집단 역학을 보다 정교하게 분석할 수 있어 밀집된 환경에서의 예측 정확도를 높여 줍니다.

- **Performance Highlights**: 시험 결과 UniEdge는 ETH, UCY, SDD와 같은 표준 데이터셋에서 최신 기법들과 비교하여 뛰어난 성능을 보여주었습니다. 특히 전통적인 그래프 신경망 방식이 가지는 한계를 극복함으로써 객체의 기간 의존성을 글로벌하게 모델링하고 예측 능력을 크게 향상시켰습니다.



### The Causal-Effect Score in Data Managemen (https://arxiv.org/abs/2502.02495)
Comments:
          To appear in Proceedings of the 4th Conference on Causal Learning and Reasoning, 2025

- **What's New**: 본 논문에서는 Causal Effect (CE)를 데이터 관리에서 속성 점수(attribution score)로 활용하는 새로운 접근법을 제안합니다. 기존의 연구들이 CE를 다양한 분야에 사용했지만, 데이터베이스에서 질의 응답을 위한 튜플의 인과 강도(causal strength)를 측정하는 데에는 시도된 바가 없었습니다. 이는 데이터베이스 연구의 새로운 방향을 제시합니다.

- **Technical Details**: Causal-Effect Score를 정의하고 이를 고전적(classical) 및 확률적(probabilistic) 데이터베이스의 맥락에서 일반화(generalize)하여 조사합니다. 이 점수는 데이터 관리의 효율성을 향상시키기 위해 인과 관계(causal relationship)를 수치적으로 측정하는 방법으로 활용됩니다. 논문은 CE의 수학적 모델링 및 응용 가능성을 심도 있게 다룹니다.

- **Performance Highlights**: 이 연구는 Causal Effect Score를 통해 데이터베이스 질의 응답의 정확성과 신뢰성을 높일 수 있는 잠재력을 보여줍니다. 실험 결과는 CE를 활용한 접근법이 기존 방법들보다 더 나은 성능을 나타냄을 입증합니다. 이는 데이터베이스 연구와 클라우드 기반 데이터 관리 시스템에 기여할 수 있는 중요한 발견입니다.



### A Self-Supervised Framework for Improved Generalisability in Ultrasound B-mode Image Segmentation (https://arxiv.org/abs/2502.02489)
Comments:
          12

- **What's New**: 최신 논문에서는 자기 지도 학습(self-supervised learning, SSL) 기반의 접근법을 소개하며, B-mode 초음파 이미지에서 효과적인 세분화를 달성하기 위한 새로운 방법론을 제시합니다. 특히, 관계 대조 손실(Relation Contrastive Loss, RCL)을 도입하여 긍정적 및 부정적 샘플 쌍을 구별하고, 추가로 공간 및 주파수 기반의 증강 전략을 통해 성능을 한층 향상시킵니다. 이러한 방법은 기존의 감독 학습(supervised learning) 방법과 비교해 데이터가 제한된 경우에서도 우수한 성능을 나타내며, 새로운 데이터에 대한 일반화 능력이 뛰어난 것으로 확인되었습니다.

- **Technical Details**: 연구에서는 특히 B-mode 초음파 이미지를 위한 대조적 SSL 접근법을 통해 새로운 관계 대조 손실(RCL)을 적용하여 고유한 특징을 학습하도록 독려합니다. RCL은 학습 가능한 메트릭을 통해 긍정적인 샘플과 부정적인 샘플 쌍을 차별화하여 입체적인 특징을 강조합니다. 또한, 초음파 이미지의 표현 학습을 개선하기 위해 공간 및 주파수 기반의 데이터 증강 전략을 제안하고 있습니다.

- **Performance Highlights**: 연구 결과에 따르면, 제안된 접근법은 세 개의 공공 유방 초음파 데이터셋에서 전통적인 감독 세분화 방법을 크게 초월하는 성능을 보였습니다. 특히, BUSI 데이터셋에서는 20% 및 50%에서 각각 4% 증가, BrEaST 데이터셋에서는 6% 및 9%의 향상, UDIAT 데이터셋에서는 6.4% 및 3.7%의 성능 향상을 기록한 것으로 나타났습니다. 더욱이, UDIAT 데이터셋에서의 일반화 성능에서도 20.6% 및 13.6%의 성능 향상이 보여져, 데이터가 부족한 환경에서도 우수한 성능을 발휘하고 있음을 입증했습니다.



### Mind the Gap: Evaluating Patch Embeddings from General-Purpose and Histopathology Foundation Models for Cell Segmentation and Classification (https://arxiv.org/abs/2502.02471)
- **What's New**: 이번 연구는 컴퓨터 비전에서 기초 모델의 발전이 디지털 조직병리학에서 Cell 분석과 같은 특수 작업에 대한 도메인 특화된 기초 모델의 장점을 충분히 탐구하지 못했음을 지적합니다. 연구팀은 두 가지 모델 카테고리 간의 표현 학습 차이를 분석하였고, 여러 종류의 인코더를 활용하여 Cell instance segmentation과 분류의 성능을 향상시키기 위해 노력했습니다. 이 과정에서 다양한 인코더의 성능을 비교하였으며, 특히 최근에 출시된 여러 조직병리학 관련 기초 모델의 효용을 검토하였습니다.

- **Technical Details**: 연구에서는 CISCA 프레임워크를 활용하여 Cell segmentation과 classification을 수행하며, 이는 다중 작업 접근법으로 3 클래스 픽셀 분류와 거리 지도 회귀, Cell 유형 분류를 통합합니다. 인코더-디코더 아키텍처를 적용하여, 세 가지 변화를 통해 세그멘테이션 맵과 거리 지도를 생성합니다. 다양한 인코더를 사용하여 모델의 일반화를 평가하며, 전처리가 없는 입력 패치로 여러 히스토리 데이터셋에서 실험을 진행했습니다.

- **Performance Highlights**: 연구에서는 PanNuke, CoNIC 및 CytoDArk0 데이터셋을 통해 인스턴스 수준의 탐지, 세그먼트 정확도 및 Cell 유형 분류에서의 성능 차이를 평가하였습니다. 일반 모델과 전용 모델 간의 성능 차이는 예기치 않은 방향으로 나타났으며, 이는 차별화된 모델 선택과 Cell 중심 조직병리학 분석에 대한 통찰력을 제공합니다. 결과적으로, 연구는 기초 모델 선택에서의 가이드를 제공하며, 세포 중심의 조직병리학 및 뇌 세포 구조 분석 흐름에서의 모델 개발을 돕습니다.



### Modular Training of Neural Networks aids Interpretability (https://arxiv.org/abs/2502.02470)
Comments:
          8 pages, under review. arXiv admin note: text overlap with arXiv:2409.15747

- **What's New**: 이번 연구에서는 신경망의 해석 가능성을 향상시키기 위해 모델을 분리된 클러스터로 나누는 'clusterability' 접근법을 제안합니다. 연구팀은 고유한 클러스터 발생을 유도하는 'clusterability loss' 기능을 통해 더 모듈화된 모델을 트레이닝하는 방법을 제시했습니다. 이 방법은 MNIST 및 CIFAR 데이터 셋에서 훈련된 CNN 모델뿐 아니라, 모듈식 더하기를 학습한 작은 트랜스포머와 언어 모델에 적용되어 그 효과를 검증하였습니다.

- **Technical Details**: 연구에서는 신경망 모듈의 클러스터 가능성을 측정할 수 있는 메트릭을 도입하여 훈련 과정에서 모듈화를 최적화하는 방법을 설명합니다. 특히, 'clusterability' 메트릭은 클러스터 내 평균적으로 포함된 가중치의 비율을 계산하여 모듈의 분리 정도를 나타내며, 이는 확률적 그래프의 반복 최적화를 통해 개선됩니다. 결과적으로, 훈련된 모델은 상호작용이 적은 모듈로 나뉘어, 각 모듈은 독립적으로 해석할 수 있는 특성을 갖게 됩니다.

- **Performance Highlights**: 모델을 모듈화하여 훈련한 결과, 각 모듈이 서로 다른 서브 스킬을 전문화하는 경향이 나타났습니다. CIFAR 데이터 세트에서 모듈은 서로 다른 라벨에 특화된 등의 특정 패턴을 보여주었으며, 언어 모델에서도 성능이 증가했습니다. 내재적 해석 가능성 기술을 적용한 결과, 훈련된 클러스터가 모델의 해석 가능성을 강화하는 데 기여한 것으로 평가되었습니다.



### Model Human Learners: Computational Models to Guide Instructional Design (https://arxiv.org/abs/2502.02456)
Comments:
          6 pages, 6 figures, 1 table

- **What's New**: 이번 논문에서는 모델 인간 학습자(Model Human Learner)라는 개념을 제안했습니다. 이 모델은 학습 이론을 통합한 계산 모델로, 교육 디자이너가 효과적인 개입(intervention)을 평가하는 데 도움을 줄 수 있습니다. 논문에서는 이 모델을 활용하여 두 가지 인간 A/B 실험의 결과를 성공적으로 예측했다는 것을 보여줍니다.

- **Technical Details**: 제안된 계산 모델은 Apprentice Learner Architecture를 기반으로 하며, 과거 모델들과의 통합 메커니즘을 포함합니다. 특히 Trestle 모델을 이용하여 점진적으로 기술을 습득하는 과정을 설명합니다. 이 모델은 문제를 제시받았을 때 이전에 학습한 기술을 현재 상태와 비교하고, 최대 유틸리티를 가진 기술을 실행합니다.

- **Performance Highlights**: Fraction Arithmetic Tutor 실험을 통해 차단(blocking)과 간섭(interleaving) 문제 구성의 학습 효과를 비교했습니다. 이로써 계산 모델이 인간의 학습 결과를 성공적으로 예측할 수 있음을 입증했습니다. 실험 결과, 간섭 방식이 장기적으로 더 큰 학습 효과를 가져옴을 나타냈습니다.



### Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models (https://arxiv.org/abs/2502.02444)
- **What's New**: 본 논문에서는 Generative Psycho-Lexical Approach (GPLA)를 도입하여 심리학적으로 기초한 LLM(대규모 언어 모델) 가치 시스템을 구축하는 방법론을 제안합니다. 기존의 연구들이 주로 인간을 위한 가치 시스템인 Schwartz의 가치 이론을 바탕으로 하였던 반면, GPLA는 LLM의 고유한 심리를 반영한 새로운 접근 방식을 제공합니다. 이 방법론은 심리적 원칙에 기반한 다섯 가지 요소의 가치 시스템을 통해 LLM의 본질적 가치를 평가하고 정렬하는 데 도움을 줄 것입니다.

- **Technical Details**: GPLA는 LLMS의 텍스트에서 인식된 데이터를 추출하고, 이로부터 가치를 찾아내며, 최종적으로 값 시스템을 구조화하는 다섯 단계의 과정을 포함합니다. 이러한 방식은 기존의 수작업으로 이루어진 가치 사전 작성 방식과 결합하여, 보다 효율적으로 LLM의 상태를 반영하는 가치를 구분할 수 있도록 합니다. 또한, 자동화된 비반응적 가치 측정을 통해 전통적인 기법의 편향성을 줄이고 유연성을 높였습니다.

- **Performance Highlights**: 논문은 세 가지 벤치마크 작업을 통해 GPLA의 유효성을 검증하고 있습니다. 이러한 작업들은 구조적 유효성을 평가하기 위한 Confirmatory Factor Analysis, LLM의 안전성을 예측하기 위한 LLM Safety Prediction 및 LLM의 가치 정렬을 평가하는 LLM Value Alignment를 포함합니다. 결과적으로 이 제안된 가치 시스템은 기존의 Schwartz 가치 시스템보다 더 나은 성과를 보여주었으며, LLM의 안전성 예측 및 정렬 향상에 기여할 수 있음이 확인되었습니다.



### LLMER: Crafting Interactive Extended Reality Worlds with JSON Data Generated by Large Language Models (https://arxiv.org/abs/2502.02441)
- **What's New**: 이 논문에서는 LLMER라는 새로운 프레임워크를 소개하며, 이는 자연어 입력을 JSON 데이터로 변환하여 사용자 요구에 따라 인터랙티브한 XR 환경을 생성할 수 있도록 돕는다. LLMER는 기존의 코드 생성 방식과는 달리, 각기 다른 XR 작업을 위한 여러 모듈을 채택하여 애플리케이션의 충돌 및 처리 지연 가능성을 낮춘다. 초석 연구에서 LLMER는 사용된 토큰 수를 80% 이상 줄이고, 작업 완료 시간을 약 60% 단축하는 효과를 보였다.

- **Technical Details**: LLMER는 고급 LLM인 GPT-4를 클라우드 서버를 통해 활용하여 질 높은 JSON 데이터를 생성하며, 이는 사용자 오디오 명령에 따라 XR 작업을 수행하는 데 쓰인다. 시스템은 가상 객체 생성기, 애니메이션 라이브러리, 현실 융합 엔진의 세 가지 주요 모듈로 구성되어 있으며, 이를 통해 복잡한 XR 환경에서의 상호작용을 실시간으로 처리한다. 구체적으로, 사용자 요청은 LLM Wrapper를 통해 처리되어 필수적인 컨텍스트 정보만을 추출하고, 불필요한 세부사항으로 인한 방해를 최소화한다.

- **Performance Highlights**: LLMER는 사용자 피드백을 분석한 결과, 자연어를 통한 명확한 커뮤니케이션을 가능하게 하며 사용자의 경험을 향상시키는 것으로 나타났다. 아바타가 사용자 음성 입력을 통해 응답하고, 이에 따라 자연스러운 대화를 실현한다. 또한, JSON 데이터 활용을 통해 처리 시간을 단축시키고 LLM의 환각문제(hallucination)로부터 영향을 덜 받게 된다.



### Medical Multimodal Model Stealing Attacks via Adversarial Domain Alignmen (https://arxiv.org/abs/2502.02438)
Comments:
          Accepted at AAAI 2025

- **What's New**: 이번 논문에서는 의료 다중 모달 대형 언어 모델(Medical Multimodal Large Language Models, MLLMs)에 대한 첫 번째 모델 도난 공격인 Adversarial Domain Alignment(ADA-STEAL)을 소개합니다. 이 방법은 공개적으로 사용 가능한 자연 이미지를 활용하여 의료 분야의 MLLM을 복제하는 데 성공했습니다. ADA-STEAL은 데이터 부족 문제를 해결하고, 의료 데이터 접근 없이도 공공 데이터를 사용하는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: ADA-STEAL은 주어진 이미지를 통해 진단 보고서를 생성하는 의료 MLLM의 기능을 모방하기 위해 두 가지 주요 기술을 사용합니다. 첫째, 전문가 지식이나 의학적 배경이 없어도 다양한 보고서를 생성할 수 있도록 오픈소스 오라클 모델을 활용합니다. 둘째, 표적 적대적 공격을 통해 비의료적 쿼리 이미지를 의료 도메인 데이터와 효과적으로 정Align하는 방법을 통합합니다.

- **Performance Highlights**: IU X-RAY 및 MIMIC-CXR 테스트 데이터셋에서의 실험을 통해 ADA-STEAL이 피해 모델의 자연어 생성 지표 및 임상 효능 지표에서 유사한 성능에 도달함을 입증했습니다. 이를 통해 공격자가 비의료적 데이터셋을 사용하더라도, 의료 MLLM의 기능을 복제할 수 있음을 보여주었습니다. 또한, 흥미로운 결과로는 의료 보고서의 다양성을 증가시킬 수 있는 방안을 제시하며, 모델 도난의 새로운 가능성을 탐구합니다.



### Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD Variants (https://arxiv.org/abs/2502.02431)
- **What's New**: 최근 딥러닝 최적화에서 Schedule-Free 최적기, AdEMAMix, MARS, Lion 등 새로운 알고리즘이 도입되었으며, 이들 모두 기존의 momentum 메커니즘을 수정하고 있습니다. 본 논문에서는 이러한 새로운 최적기와 SGD의 이론적 가속화 간의 연결성을 확립합니다. AdEMAMix가 SGD의 가속화된 버전과 유사한 성능을 보여준다는 점을 실험적으로 검증하고, 이를 바탕으로 Simplified-AdEMAMix라는 새로운 변형을 제안합니다.

- **Technical Details**: 우리는 이론적으로 Accelerated SGD와 최근에 제안된 최적기들 간의 직접적인 연결을 확립하였습니다. Schedule-Free SGD는 가속화된 SGD와 가중치 평균화를 수행하는 것과 수학적으로 동등하다는 것을 증명하며, Lion, Schedule-Free AdamW, AdEMAMix 같은 최적기들이 가속화된 SGD 기법과 사전 조정(preconditioning) 기술을 조합한 것으로 해석될 수 있음을 보여줍니다.

- **Performance Highlights**: 150백만 매개변수를 가진 transformer 모델을 통해 실험을 수행한 결과, AdEMAMix가 다른 최적기들보다 우수한 성능을 보였으며, Batch size가 클 경우 성능 이점이 줄어든다는 것을 확인했습니다. 새로운 Simplified-AdEMAMix는 AdEMAMix의 성능을 유지하면서 서로 다른 momentum 항을 두 개 필요로 하지 않는 점이 특징입니다.



### Activation-Informed Merging of Large Language Models (https://arxiv.org/abs/2502.02421)
- **What's New**: 이번 논문에서는 Activation-Informed Merging (AIM)이라는 새로운 기술을 소개합니다. AIM은 여러 개의 fine-tuned large language models (LLMs)의 파라미터와 embedding을 통합하는 모델 병합 방식으로, 활성화 공간(activation space) 정보를 활용하여 성능과 견고성을 향상시키는 데 중점을 두고 있습니다. AIM은 기존의 병합 방법에 적용 가능한 유연하고 보완적인 솔루션으로, 지속적 학습(continual learning, CL) 및 모델 압축의 원리를 바탕으로 설계되었습니다.

- **Technical Details**: AIM은 병합 과정에서 활성화 공간 정보를 통합하여 필수적인 가중치(weights)를 선택적으로 우선시합니다. 이는 모델의 기본 성능을 유지하면서도 새롭게 fine-tuned된 모델의 지식을 통합하는 데 도움을 줍니다. AIM의 핵심은 활성화 정보를 통해 가장 영향력 있는 가중치가 최소한의 변화만 겪도록 업데이트 단계(modification step)를 수정하는 것입니다.

- **Performance Highlights**: 실험 결과 AIM은 기존의 병합 방법들과 함께 사용할 수 있는 보완적인 솔루션으로, 여러 벤치마크에서 성능을 최대 40%까지 향상시킵니다. AIM은 구조가 단순함에도 불구하고 활성화 공간 정보의 중요성을 강조하며, LLM 병합 전략에 실질적인 발전을 가져올 수 있는 가능성을 보여줍니다.



### LV-XAttn: Distributed Cross-Attention for Long Visual Inputs in Multimodal Large Language Models (https://arxiv.org/abs/2502.02406)
- **What's New**: 이번 논문에서 제안한 LV-XAttn는 대규모 시각적 입력을 효율적으로 처리하기 위한 새로운 분산 크로스 어텐션 메커니즘이다. 기존의 접근 방식들과는 달리, LV-XAttn는 작은 쿼리 블록을 GPU 간에 전달하고 큰 키-값 블록은 각 GPU에 로컬로 저장하는 구조를 취하고 있다. 또한, 길어진 시각적 컨텍스트를 지원하기 위한 활성화 재계산 기법도 도입하였다. 이로 인해 LV-XAttn는 기존 방법들보다 기억 용량의 요구를 줄이고 통신 오버헤드를 최소화했다.

- **Technical Details**: LV-XAttn의 기술적 세부 사항에는 시퀀스 병렬 처리와 최소 통신 오버헤드를 활용한 정확한 크로스 어텐션 메커니즘이 포함된다. 대규모 시각적 입력을 다루기 위해, LV-XAttn는 각 작업자(worker)에서 큰 키와 값 블록을 로컬로 저장하고, 작은 쿼리 블록을 서로 전송하여 어텐션 출력을 계산한다. 또한, 활성화 재계산을 통해 메모리를 절약하고 긴 시각 정보 입력을 처리할 수 있는 능력을 제공한다.

- **Performance Highlights**: LV-XAttn는 mPLUG-Owl3와 OpenFlamingo 모델 평가에서 최대 45.85배의 크로스 어텐션 속도 향상과 전체 모델 반복 시간에서 최대 5.58배의 향상을 달성하였다. 통신 볼륨을 최소화하고 계산과 통신을 효과적으로 중첩함으로써, LV-XAttn는 기존 접근 방식에 비해 0.42% 이하의 오버헤드를 유지하며 효율성을 크게 향상시켰다.



### FewTopNER: Integrating Few-Shot Learning with Topic Modeling and Named Entity Recognition in a Multilingual Framework (https://arxiv.org/abs/2502.02391)
Comments:
          Code source : this https URL

- **What's New**: FewTopNER는 저자원이던 언어 환경에서의 named entity recognition(NER)과 주제 인식을 통합한 혁신적인 프레임워크입니다. XLM-RoBERTa를 기반으로 한 다국어 인코더와 언어 특화 조정 메커니즘을 활용하여 강력한 문맥 임베딩을 생성합니다. 이 아키텍처는 BiLSTM과 Conditional Random Fields를 사용한 프로토타입 기반의 NER 모듈과 하이브리드 확률적 방법을 통해 문서 수준의 주제를 추출하는 주제 모델링 모듈로 구성되어 있습니다.

- **Technical Details**: FewTopNER는 엔티티 인식과 주제 모델링 간의 동적 양방향 주의 망을 통해 정보를 융합하는 Cross-Task Attention Module을 포함하고 있습니다. 이를 통해 전역 의미 맥락을 엔티티 표현에 추가하고 주제 일관성을 개선합니다. 또한, Model-Agnostic Meta-Learning (MAML)을 통합하여 적은 데이터로 빠른 미세 조정을 가능하게 하며, Active Learning Interface를 통해 불확실한 사례를 겨냥하여 모델 예측을 반복적으로 개선합니다.

- **Performance Highlights**: FewTopNER는 영어, 프랑스어, 스페인어, 독일어, 이탈리아어를 포함한 다국어 벤치마크에서 기존의 최첨단 few-shot NER 모델을 유의미하게 초과하는 성능을 보여줍니다. 특히, F1 점수에서 2.5-4.0 포인트의 개선을 달성했으며, 정규화된 점수 기반의 상호정보량을 통해 주제 일관성이 향상되었습니다. 샘플과 메커니즘에 대한 분석 연구는 공유 인코더 및 과제 간 통합이 전체 성능에 미치는 중요한 기여를 입증하고 있습니다.



### CoAT: Chain-of-Associated-Thoughts Framework for Enhancing Large Language Models Reasoning (https://arxiv.org/abs/2502.02390)
- **What's New**: 최근 LLM(대형 언어 모델) 기술이 급격히 발전하면서, 기존의 '빠른 사고' 접근 방식에서 '느린 사고' 기법으로 전환하는 경향이 증대되고 있습니다. 이러한 '느린 사고'는 인간의 사고 과정과 유사하게 새로운 정보를 통합하고 지식을 갱신하는 능력을 반영하고 있습니다. 본 논문에서는 CoAT(Chain-of-Associated-Thoughts) 프레임워크를 제안하며, 이는 MCTS(Monte Carlo Tree Search) 알고리즘과 동적 연상 기억 메커니즘을 결합하여 LLM의 추론 능력을 혁신적으로 확장하고 있습니다.

- **Technical Details**: CoAT 프레임워크는 인간의 연상 능력을 모방하여 LLM이 실시간으로 관련 정보를 검색하고 자기 증강을 수행할 수 있도록 설계되었습니다. MCTS 알고리즘의 최적화된 라우팅 전략을 사용하여, 각 연상 기억의 추가가 후속 콘텐츠 생성을 위한 중요한 정보를 제공하도록 보장합니다. 이 구조적 탐색과 적응 학습의 시너지를 통해 CoAT는 기존 LLM의 한계를 극복하고 의미 통일성을 유지하면서 추론 범위를 확장합니다.

- **Performance Highlights**: 광범위한 생성 및 추론 작업을 통한 실험 결과, CoAT 프레임워크는 기존의 추론 프로세스에 비해 정확성, 일관성 및 다양성 측면에서 우수한 성능을 보였습니다. 이는 CoAT가 이전 추론을 반복적으로 수정하고 진화하는 정보를 통합함으로써 제공하는 정밀하고 포괄적인 결과를 반영합니다. 본 연구의 결과는 LLM 프레임워크의 혁신을 가져오며, 향후 복잡한 추론 작업을 효과적으로 해결할 수 있는 기초를 제공합니다.



### The Cost Perspective of Liquid Democracy: Feasibility and Contro (https://arxiv.org/abs/2502.02380)
- **What's New**: 이번 논문에서는 예산 제약이 있는 Liquid Democracy 모델을 제안하며, 유권자의 완전한 대표성을 보장하는 투표자를 중앙에서 선출하는 방법을 탐구합니다. 기존 연구에서는 election 관련 비용을 줄이는 것에 초점을 맞추었지만, 이러한 비용을 명시적으로 고려한 연구는 진행되지 않았습니다. 새로운 접근법에서는 서로 다른 voting 및 delegating 비용을 관리하면서, 효과적으로 유권자를 선택하는 방법을 제시하고 있습니다.

- **Technical Details**: 연구에서는 유권자와 그들이 위임할 수 있는 대리인을 나타내는 방향 그래프 G(N, E)를 설정하고, 투표 비용(voting cost)과 위임 비용(delegating cost)을 각각 d(i)와 v(i)로 정의합니다. 모델은 유권자 간의 신뢰 관계를 기반으로 하며, 모든 유권자가 최선을 다해 투표를 하도록 촉진하는 것을 목표로 합니다. 또한, 제안된 모델은 비용 최소화를 목표로 하며, 유권자들이 적절한 대리인을 선택하도록 유도하기 위한 기법을 포함합니다.

- **Performance Highlights**: 컴퓨터 과학 및 사회 선택 이론에 대한 기존 연구를 바탕으로, 이 모델은 비용 제약 조건 하에서 투표자의 최적 선택이 가능하다는 것을 증명하였습니다. 연구의 결과는 제안된 모델이 상당히 효율적이며, 각 유권자로 하여금 적절한 대리인을 선택함으로써 민주적 정당성을 유지하도록 돕는다는 것을 보여줍니다. 추가적으로, 외부 요인이 유권자의 권한을 강화하기 위해 선거 요소를 조작할 수 있는 가능성에 대한 탐구도 이루어져, Liquid Democracy 연구에 대한 새로운 방향성을 제시하고 있습니다.



### MaintaAvatar: A Maintainable Avatar Based on Neural Radiance Fields by Continual Learning (https://arxiv.org/abs/2502.02372)
Comments:
          AAAI 2025. 9 pages

- **What's New**: 이번 연구는 Neural Radiance Fields(NeRF)를 기반으로 한 유지 가능한 아바타(maintainable avatar) 생성을 제안합니다. 기존 연구는 훈련 데이터의 인물 이미지를 고정된 것으로 가정했지만, 우리는 지속적인 학습을 통해 인물의 변화하는 외모와 자세를 올바르게 모델링할 수 있는 방법을 찾았습니다. 우리는 Global-Local Joint Storage Module과 Pose Distillation Module을 활용하여 과거 외모의 렌더링 질감을 유지할 수 있도록 설계했습니다.

- **Technical Details**: 우리가 제안하는 MaintaAvatar는 Global-Local Joint Storage Module을 통해 다양한 외모의 글로벌 및 로컬 정보를 독립적으로 저장하고, Pose Distillation Module을 통해 과거 작업에서의 자세 정보를 추출하여 새로운 작업의 지도 신호로 사용합니다. 이러한 접근 방식은 기존의 static human-nerf 방법과는 달리, 적은 양의 데이터로도 모델을 신속히 조정하고 재학습 시 발생할 수 있는 잃어버림(catastrphic forgetting)을 방지할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, MaintaAvatar 모델은 두 개의 데이터셋에서 뛰어난 성능을 보여주었습니다. 이 모델은 제한된 데이터 수집으로도 고품질의 렌더링을 유지하면서도 빠르게 새로운 외모에 적응할 수 있는 장점을 가지고 있습니다. 이를 통해 우리는 유지 가능한 가상 아바타의 생성에 대한 새로운 패러다임을 제시하였습니다.



### Accurate Pocket Identification for Binding-Site-Agnostic Docking (https://arxiv.org/abs/2502.02371)
- **What's New**: 이번 논문에서는 구조 기반 약물 설계를 위한 드러그 가능 포켓 식별의 중요성을 강조하고, 이를 극복하기 위해 RAPID-Net이라는 새로운 포켓 찾기 알고리즘을 개발했습니다. RAPID-Net은 AutoDock Vina와 통합되어 기존 알고리즘인 DiffBindFR을 초월하며, AlphaFold 3이 처리할 수 없는 대형 단백질에 대한 블라인드 도킹 능력을 제공합니다. 이 알고리즘은 PUResNet 및 Kalasanty보다 높은 도킹 정확도를 자랑하며, 다양한 데이터 세트에서 성능을 검증했습니다.

- **Technical Details**: RAPID-Net은 기존의 포켓 예측 알고리즘과 구별되는 앙상블 기반 모델로, 다섯 개의 독립적으로 학습된 모델 복제체로 구성되어 있습니다. 이 모델은 두 가지 유형의 포켓을 반환하며, 다수결 포켓은 최소 3개의 모델에서 예측된 부분을 포함하고, 소수 보고 포켓은 최소 1개의 모델에서 예측된 부분을 포함하여 전반적인 회수율을 증가시킵니다. 또한 RAPID-Net은 다층 딥 뉴럴 네트워크(Deep Neural Networks)를 통해 더욱 효과적인 포켓 식별을 실현합니다.

- **Performance Highlights**: RAPID-Net은 PoseBusters, Astex Diverse Set, Coach420 및 BU48 데이터 세트를 포함한 다양한 벤치마크에서 테스트되어 안정적이고 신뢰할 수 있는 성능을 보여주었습니다. 특히, 포켓 식별의 정확도와 회수 측면에서 기존 ML 기반 포켓 예측 알고리즘에 비해 우수한 결과를 도출했습니다. RAPID-Net은 효율적인 예측 성능으로 복잡한 도킹 작업을 위한 포켓 예측 프레임워크를 제공하며, 원거리 활성 부위를 식별할 수 있는 능력도 입증되었습니다.



### Evaluating the Effectiveness of LLMs in Fixing Maintainability Issues in Real-World Projects (https://arxiv.org/abs/2502.02368)
- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)이 코드 유지 관리 문제를 해결하는 데 어느 정도 효과적인지에 대한 평가를 다룹니다. 10개의 GitHub 저장소에서 수집한 127개의 유지 관리 문제를 분석했습니다. LLM을 활용한 솔루션이 기존 문제를 해결하는 데 얼마나 기여하는지를 검토했습니다.

- **Technical Details**: 이 연구에서는 Copilot Chat과 Llama 3.1에 대해 zero-shot prompting을 사용하고, Llama에 대해서는 few-shot prompting을 사용하여 실험을 진행했습니다. 생성된 LLM 솔루션의 결과는 컴파일 오류, 테스트 실패 및 새로운 유지 관리 문제를 기준으로 평가되었습니다. Llama는 몇 가지 예시를 통해 44.9%의 문제를 해결했으며, Copilot Chat은 32.29%, Llama zero-shot은 30%의 해결률을 보였습니다.

- **Performance Highlights**: 대부분의 솔루션은 새로운 오류 또는 유지 관리 문제를 발생시켰으나, 45명의 참가자가 참여한 인간 분석 결과, 51개의 LLM 생성 솔루션 중 68.63%의 참가자가 읽기 쉬워졌다고 응답했습니다. 전체적으로 LLM은 유지 관리 문제를 해결하는 데 잠재력을 보였지만, 오류가 발생하는 점은 현재의 한계를 드러냅니다.



### Field Matching: an Electrostatic Paradigm to Generate and Transfer Data (https://arxiv.org/abs/2502.02367)
- **What's New**: 본 논문에서는 Electrostatic Field Matching (EFM)이라는 새로운 방법을 제안합니다. 이 방법은 생성 모델링(generative modeling)과 분포 전이(distribution transfer) 작업 모두에 적합합니다. EFM은 전기 축전기의 물리학에 영감을 받아 개발되었습니다.

- **Technical Details**: EFM은 전하가 있는 두 판을 통하여 데이터를 전이하는 원리를 기반으로 합니다. 우리는 이 판에 소스(source)와 타겟(target) 분포를 배치하고 각각에 긍정적(positive)과 부정적(negative) 전하를 부여합니다. 전기장을 뉴럴 네트워크(neural network)를 통해 학습하여, 이 전기장 선을 따라 샘플을 이동시켜 변환을 수행합니다.

- **Performance Highlights**: 이 방법은 다양한 실험에서 성능을 입증하였습니다. toy 데이터와 이미지 데이터 실험을 통해 EFM의 효과성을 확인하였으며, 저차원 및 고차원 생성 모델링 작업에서 개념 증명을 위한 실험을 수행하였습니다. EFM은 노이즈-데이터(nnoise-to-data) 및 데이터-데이터(data-to-data) 생성 작업 모두에 적용될 수 있습니다.



### Test Time Training for 4D Medical Image Interpolation (https://arxiv.org/abs/2502.02341)
- **What's New**: 본 논문에서는 4D 의료 이미지 보간(4D medical image interpolation)을 위한 새로운 테스트 시간 훈련 프레임워크(TTT4MII)를 제안합니다. 이 프레임워크는 라벨이 없는 상태에서 모델이 새로운 분포에 적응할 수 있도록 자기 지도(self-supervised) 학습을 활용합니다. 기존 연구에서 간과된 분포 이동(distribution shifts) 문제를 해결하여, 모델이 실시간으로 적응할 수 있는 시스템을 제공합니다.

- **Technical Details**: TTT4MII는 두 가지 자기 지도 작업인 회전 예측(rotation prediction)과 이미지 재구성(image reconstruction)을 통합하여 설계되었습니다. 이는 4D 의료 이미지의 보간 정확성을 높이는 데 기여하며, 테스트 데이터가 라벨 없이도 모델이 학습할 수 있도록 합니다. 또한, 단순 TTT(Naïve TTT), 온라인 TTT(Online TTT), 미니 배치 TTT(Mini-batch TTT) 등 세 가지 TTT 기법을 적용하여 성능 향상을 분석합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 Cardiac 데이터셋에서 33.73dB, 4D-Lung 데이터셋에서 34.02dB의 피크 신호 대 잡음비(peak signal-to-noise ratio)를 달성하며, 기존의 다양한 평가 지표에서 상당한 향상을 보였습니다. 이 연구는 4D 의료 이미지 보간을 발전시키는 데 기여할 뿐만 아니라, 이미지 분할(image segmentation) 및 이미지 등록(image registration) 등 다른 분야에서도 활용 가능한 템플릿을 제공합니다.



### EdgeGFL: Rethinking Edge Information in Graph Feature Preference Learning (https://arxiv.org/abs/2502.02302)
- **What's New**: 이번 논문에서는 Edge-empowered Graph Feature Preference Learning (EdgeGFL) 프레임워크를 제안하여, 노드 임베딩(node embeddings) 학습을 지원하는 엣지 임베딩(edge embeddings)을 포착하는 방법을 개발합니다. 기존의 GNN 모델이 노드와 엣지의 피처(feature) 정보를 독립적인 작업으로 처리하는 문제를 해결하기 위해, 멀티채널 필터(multi-channel filters)를 이용하여 정확한 노드 피처를 효과적으로 포착합니다.

- **Technical Details**: EdgeGFL 모델은 노드와 엣지의 정보를 동시에 학습하는 새로운 메시지 패싱(message passing) 프레임워크를 도입합니다. 이 프레임워크에서 엣지 피처는 하나의 실수 값이 아닌 학습 가능한 피처 벡터(feature vectors)로 표현되며, 각 훈련 에포크(epoch)에서 노드는 연결된 엣지 채널을 통해 이웃에게 현재 표현을 전송하고, Hadamard 곱(hadamard product)을 통해 이 정보를 정제하여 업데이트합니다.

- **Performance Highlights**: 실험 결과, EdgeGFL 모델은 4개의 실제 세계의 이종 그래프(heterogeneous graphs)에서 최고의 성능을 기록하며, 노드 분류(node classification) 및 클러스터링(clustering) 작업에서 기존의 최첨단 방법들과 비교하여 우수한 결과를 보여주었습니다. 이는 노드와 엣지 정보를 통합적으로 활용하여 GNN 모델의 기능과 유연성을 증대시킨 결과로 볼 수 있습니다.



### FRAUD-RLA: A new reinforcement learning adversarial attack against credit card fraud detection (https://arxiv.org/abs/2502.02290)
- **What's New**: 이 논문에서는 기존의 공격 모델의 한계를 보여주고, 크레딧 카드 사기 탐지 시스템을 위한 새로운 공격 모델인 FRAUD-RLA를 제안합니다. FRAUD-RLA는 강화 학습(reinforcement learning)을 기반으로 하여 분류기를 우회하도록 설계되었습니다. 특히, 이 모델은 공격자가 필요한 지식이 적고, 탐색-활용(tradeoff) 균형을 최적화하여 보상을 극대화할 수 있도록 만들어졌습니다.

- **Technical Details**: FRAUD-RLA는 Proximal Policy Optimization (PPO) 알고리즘을 사용하여 비가시적(fraudulent) 거래를 생성하는 문제를 RL 문제로 모델링합니다. 이 시스템은 카드 소지자와 거래당시의 정보를 고려하여 사기 탐지 시스템을 속이기 위한 최적의 전략을 찾습니다. 기존의 방법들과는 달리, FRAUD-RLA는 공격자가 사용자 거래 이력을 알거나 모델에 대한 높은 접근성을 요구하지 않습니다.

- **Performance Highlights**: FRAUD-RLA는 세 가지 이질적인(dataset) 데이터셋에서 실험하여 두 개의 사기 탐지 시스템에 대해 효과성을 검증했습니다. 결과적으로, 이 접근 방식은 기존 공격 대비 뛰어난 성능을 보여주었으며, 사기 탐지 시스템의 보안을 약화시키는 능력이 입증되었습니다. 이 연구는 크레딧 카드 사기 탐지 및 적대적 기계 학습(adversarial machine learning) 간의 중요한 연구 공백을 메우기 위한 기초를 마련하였습니다.



### GP-GS: Gaussian Processes for Enhanced Gaussian Splatting (https://arxiv.org/abs/2502.02283)
Comments:
          14 pages,11 figures

- **What's New**: 이 논문에서는 3D Gaussian Splatting(3DGS)의 격차를 메우기 위해 Gaussian Processes Gaussian Splatting(GP-GS)라는 새로운 프레임워크를 제안합니다. GP-GS는_sparse Structure-from-Motion (SfM) 포인트 클라우드의 밀도를 높이고 불확실성을 기반으로 한 적응형 정보를 제공하여 3D 장면 재구성을 개선합니다. 이 접근 방식은 새로운 후보 지점을 추론하고, 고불확실성 예측을 제거함으로써 고품질 3D Gaussians를 생성하여 렌더링 성능을 극대화합니다.

- **Technical Details**: 본 연구에서는 Multi-Output Gaussian Process (MOGP) 모델을 활용하여 2D 이미지 픽셀과 깊이 정보로부터 3D 포인트 클라우드의 위치와 색상을 예측합니다. 특히, 픽셀을 샘플링하여 후보 MOGP 입력을 생성하고, 각 샘플에 대해 MOGP를 사용해 3D 속성을 예측한 후, 불확실성 기반 필터링을 통해 고불확실성 예측을 제거합니다. 이렇게 함으로써, 더 안정적이고 구조화된 포인트 클라우드 덴시피케이션 프로세스를 구현했습니다.

- **Performance Highlights**: 다양한 합성 및 실제 세계 데이터셋에서 수행된 실험을 통해, GP-GS는 기존 SfM 기반 파이프라인에 쉽게 통합될 수 있으며, 렌더링 품질을 향상시키는 유연한 모듈로 작용합니다. 추가적으로, GP-GS는 복잡한 지역에서의 3D Gaussian 초기화를 효과적으로 개선하여, 특히 조명이 어려운 조건에서도 우수한 성능을 보여줍니다. 이러한 연구 결과는 NVS(Novel View Synthesis) 모델의 품질 향상에 기여할 것으로 기대됩니다.



### Error Distribution Smoothing:Advancing Low-Dimensional Imbalanced Regression (https://arxiv.org/abs/2502.02277)
Comments:
          16 pages, 12 figures

- **What's New**: 이번 논문에서는 imbalanced regression (불균형 회귀)의 새로운 개념을 도입하여 데이터의 복잡성과 밀도를 모두 고려하는 Complexity-to-Density Ratio (CDR)를 제안합니다. 기존의 불균형 회귀 연구에서 간과된 부분들을 보완하고, 더욱 정교한 모델 성능 향상을 꾀하고자 합니다. 또한, Error Distribution Smoothing (EDS)이라는 방법을 통해 데이터의 중복성을 줄이고, 예측 오류의 분포를 매끄럽게 만들어 모델의 일관된 성능을 보장합니다.

- **Technical Details**: CDR은 주어진 데이터 영역이 복잡성에 비례하여 충분한 데이터 쌍을 포함하고 있는지를 평가하는 지표입니다. Global Imbalance Metric (GIM)은 전체 데이터 분포를 평가하여 모델 성능에 영향을 미칠 수 있는 불균형을 식별합니다. EDS 방법론은 과대표된 영역의 데이터 쌍을 선택적으로 줄여 불필요한 중복을 제거하고, 다양한 영역에서의 예측 오류 분포를 개선합니다.

- **Performance Highlights**: 여러 실험을 통해 EDS는 불균형 회귀 문제에 효과적임을 입증했습니다. 각 실험에서 imbalanced regression 문제에 대한 성능 향상과 함께 코드와 데이터셋을 제공하여 실용적인 평가를 지원합니다. 이러한 접근 방식은 특히 저대표 지역에서의 모델 성능 일관성을 유지하며, 전반적으로 모델의 예측 정확성을 향상시킵니다.



### Adviser-Actor-Critic: Eliminating Steady-State Error in Reinforcement Learning Contro (https://arxiv.org/abs/2502.02265)
Comments:
          13 pages, 9 figures

- **What's New**: 본 논문에서는 Adviser-Actor-Critic (AAC) 프레임워크라는 새로운 접근 방식을 제안하고 있습니다. AAC는 피드백 제어 이론의 정밀성을 강화하는 동시에 RL(강화 학습)의 적응 학습 능력을 결합하여 정밀한 제어 작업 문제를 해결하도록 설계되었습니다. 이 방법론은 로봇공학 분야에서 흔히 나타나는 목표 상태에 도달하는 데 필요한 정밀성을 개선하며, 대한 성능 평가에서 기존 RL 알고리즘들보다 우수한 성능을 보여주었습니다.

- **Technical Details**: AAC 프레임워크는 모델 프리(model-free) 접근 방식으로 PID(비례-적분-미분) 제어기와 강화 학습을 통합합니다. 이 프레임워크에서는 아드바이저(adviser)가 액터(actor)를 멘토링하여 제어 행동을 세밀하게 조정할 수 있게 합니다. 더불어, 복잡한 시스템들을 다루기 위해 고안된 다양한 모델링 기법과 함께 비선형(dynamics)을 명확히 할 수 있는 새로운 제어 정책을 마련하고 있습니다.

- **Performance Highlights**: 실험 결과, AAC는 세 가지 목표 지향 환경에서 높은 정밀성과 신뢰성을 입증하며, 다양한 로봇 제어 작업에서 적응성과 효율성을 보여주었습니다. 특히, 시뮬레이션 환경에서 현실 세계로의 전환 과정에서의 장점을 강조하며, AAC의 전반적인 성능이 기존 방법론들과 비교하여 월등하다는 것을 보여주었습니다.



### Conversation AI Dialog for Medicare powered by Finetuning and Retrieval Augmented Generation (https://arxiv.org/abs/2502.02249)
Comments:
          12 pages

- **What's New**: 이 연구는 의사-환자 대화의 맥락에서 두 가지 주요 기법인 LoRA (Low-Rank Adaptation)로서의 파인 튜닝과 RAG (Retrieval-Augmented Generation) 프레임워크의 비교 분석을 진행합니다. 다양한 의료 분야의 여러 데이터셋을 활용하여 진행된 이 연구는 기존의 방법론에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 이 분석은 세 가지 최첨단 모델인 Llama-2, GPT, LSTM 모델을 포함하여, 실제 의사-환자 대화를 통해 수행되었습니다. 평가 지표로는 언어 품질(perplexity, BLEU score), 사실 정확성(fact-checking), 의료 지침 준수, 인간의 판단(coherence, empathy, safety) 등을 사용하여 모델 성능을 종합적으로 평가했습니다.

- **Performance Highlights**: 연구 결과는 각 접근법의 강점과 한계를 제공하며, 의료 애플리케이션에 적합성을 조명합니다. 또한, 다양한 환자 질문에 대한 모델의 견고성 조사 및 도메인 특화 지식 통합의 영향을 탐구하여, 적절한 데이터 증강 및 검색 전략을 통해 LLM의 성능을 향상시킬 수 있는 가능성을 강조합니다.



### Rotation-Adaptive Point Cloud Domain Generalization via Intricate Orientation Learning (https://arxiv.org/abs/2502.02247)
Comments:
          13pages, supplementary included, early accepted by TPAMI

- **What's New**: 이번 연구에서는 3D 포인트 클라우드 분석에서의 예측할 수 없는 회전에 대한 취약성을 해결하기 위해, orientation-aware 3D domain generalization을 위한 혁신적인 회전 적응형 도메인 일반화 프레임워크를 제안합니다. 이 접근법은 복잡한 샘플을 활용한 반복 학습 과정을 통해 방향 변화를 완화하는 데 중점을 둡니다.

- **Technical Details**: 연구진은 각 포인트 클라우드에 대해 가장 도전적인 회전을 식별하고, 이를 최적화하여 복잡한 방향 세트를 구축합니다. 이후 방향 일관성 손실(orientation consistency loss)과 마진 분리 손실(margin separation loss)을 포함한 방향 인식 대비 학습 프레임워크를 활용하여 회전 일관성을 지닌 범주적으로 구별 가능한 일반화 가능한 특징들을 효과적으로 학습합니다.

- **Performance Highlights**: 다양한 3D 교차 도메인 벤치마크에서 실시된 광범위한 실험과 ablation 연구를 통해 제안된 접근법의 최첨단 성능을 확립하였습니다. 이는 orientation-aware 3D domain generalization의 맥락에서 매우 중요한 발전을 나타냅니다.



### Exploring the latent space of diffusion models directly through singular value decomposition (https://arxiv.org/abs/2502.02225)
- **What's New**: 이 논문은 기존의 확산 모델(Diffusion Models, DMs)의 잠재 공간(latent space)을 직접 탐구하며, 이를 통해 이미지 생성 결과를 제어할 수 있는 세 가지 유용한 특성을 발견했습니다. 이 특성들은 데이터 수집을 요구하지 않으며, 생성된 이미지의 정체성 유지(identity fidelity)를 보장합니다. 이를 기반으로 문자 프롬프트에 의해 유도된 두 쌍의 잠재 코드로부터 임의 속성을 학습할 수 있는 새로운 이미지 편집 프레임워크(image editing framework)를 제안합니다.

- **Technical Details**: 연구자들은 특이값 분해(Singular Value Decomposition, SVD)를 통해 DMs의 잠재 공간을 분석하였으며, 이 과정에서 발견된 세 가지 특성은 다음과 같습니다. 첫째, 모든 시간 단계에서 의미적으로 유사한 작은 이웃(subspace)을 제공합니다. 둘째, 새로운 특성이 추가되지 않는 한, 기존 특성을 변경할 수 없습니다. 셋째, 특이값의 감소 순서에 따라 변속성이 있으며, 이는 두 다른 시간 단계 간의 잠재 코드에서 새 특성을 도입하는 대안적인 방법을 제공합니다.

- **Performance Highlights**: 우리는 새로운 이미지 편집 접근 방식을 이론적 분석과 다양한 데이터 세트에 대한 포괄적인 실험을 통해 검증했습니다. 이 연구는 이미지 편집에서 높은 품질을 유지하면서도 원본 이미지의 정체성 충실성을 보존하는 방법을 보여줍니다. 이들의 결과는 특히 새로운 속성을 도입하는 효율성과 논리적인 명확성을 강조하며, 향후 이미지 조작 분야의 혁신을 이끌 수 있기를 기대합니다.



### Bias Detection via Maximum Subgroup Discrepancy (https://arxiv.org/abs/2502.02221)
- **What's New**: 이번 논문은 기존의 데이터 품질 검사 및 AI 시스템 출력의 공정성 평가에서 부족했던 새로운 거리 개념, 최대 하위 그룹 불일치(Maximum Subgroup Discrepancy, MSD)를 제안합니다. 해당 메트릭은 모든 기능 하위 그룹에 대해 낮은 불일치가 유지되는 경우 두 분포가 유사함을 정의합니다. 기존의 Total Variation 및 Wasserstein 거리와 비교할 때, MSD는 샘플 복잡도가 선형으로 줄어들며 실제 응용에서 유용성을 지닙니다.

- **Technical Details**: MSD는 두 확률 분포 간의 거리를 평가하기 위해 혼합 정수 최적화(Mixed-integer optimization, MIO) 알고리즘을 사용합니다. 이를 통해 모든 보호 속성에 대한 편향을 식별하고 수정하는 데 있어 명확한 정보를 제공합니다. 샘플 복잡도는 선형으로 증가하므로, 높은 차원의 데이터에서도 신뢰할 수 있는 편향 검출이 가능해집니다.

- **Performance Highlights**: MSD는 10개의 실제 데이터셋에서 Total Variation 및 Wasserstein 메트릭과 비교하여 우수한 성능을 보였습니다. 논문에서 제시된 실험을 통해 MSD를 사용하면 실제 값에 대한 좋은 추정치를 얻는 데 필요한 샘플 수가 기하급수적으로 감소함을 입증했습니다. 이러한 결과들은 MSD를 이론적으로 잘 정립된 실용적인 편향 탐지 방법으로 제안하는 중요한 기초가 됩니다.



### Can You Move These Over There? An LLM-based VR Mover for Supporting Object Manipulation (https://arxiv.org/abs/2502.02201)
Comments:
          64 pages (30 in main text), 22 figures (19 in main text)

- **What's New**: 이 논문에서는 LLM(대형 언어 모델)을 활용하여 사용자의 음성 명령을 이해하고 해석하여 가상 현실(VR)에서 객체 조작을 지원하는 VR Mover라는 솔루션을 제안합니다. 사용자가 단순히 가리키고 말함으로써, VR Mover는 구조화된 입력 없이 객체를 조작할 수 있는 능력을 갖추고 있습니다. 사용자 연구를 통해 이 인터페이스가 다중 객체 조작에서의 사용자 유용성을 향상시키고, 작업 부담과 팔의 피로를 줄인다는 결과를 보여주었습니다.

- **Technical Details**: VR에서 객체 조작은 3D 객체를 이동시키고 조작하는 작업을 의미합니다. 하지만 전통적인 사용 방법은 사용자가 객체를 선택하고 위치, 회전, 크기를 지정한 후 조작을 확인해야 하며, 이 과정에서 ‘고릴라 팔 효과’로 인한 팔의 피로가 발생할 수 있습니다. 저자들은 사용자가 자연스럽게 음성과 제스처를 결합하여 객체 조작을 할 수 있도록 LLM에 여러 API를 제공하여 문제를 해결하고자 하였습니다.

- **Performance Highlights**: VR Mover는 비정형적(Unstructured), 불완전한(Incomplete), 맥락화된(Contextualized) 명령을 다룰 수 있는 혁신적인 LLM 기반 객체 조작 인터페이스를 통해 사용자 경험을 크게 개선하였습니다. 사용자 연구에서 LLM 기반 인터페이스가 사용성, 전체 경험, 다중 객체 조작의 성능을 향상시키고, 팔의 피로와 작업 부담을 감소시킨다는 결과가 도출되었습니다. 이 연구는 향후 LLM 기반 객체 조작 인터페이스 설계에 대한 통찰을 추가로 제공할 것으로 기대됩니다.



### An Efficient Local Search Approach for Polarized Community Discovery in Signed Networks (https://arxiv.org/abs/2502.02197)
- **What's New**: 이 논문에서는 긍정적 또는 부정적으로 라벨링된 엣지로 구성된 서명 네트워크(Signed networks)의 핵심 문제인 polarized community discovery (PCD) 문제를 다룹니다. 저자는 Frank-Wolfe 최적화 방법을 기반으로 하여, 대규모의 밀집된 균형 잡힌 커뮤니티를 식별하는 새로운 접근 방식을 제안합니다. 이 방법은 기존의 최첨단 방법들과 비교하여 솔루션 품질에서 우위를 점하고, 계산 효율성 측면에서도 경쟁력을 유지합니다.

- **Technical Details**: 본 논문은 서명 네트워크 G=(V,E)에서 각 엣지가 긍정적 또는 부정적이라는 정보를 포함하고, 이러한 네트워크의 커뮤니티 탐색을 위해 Frank-Wolfe 최적화 알고리즘을 활용합니다. Polarized community discovery (PCD)는 기존의 signed network partitioning (SNP) 문제와 유사하나, 모든 정점이 클러스터에 포함될 필요는 없습니다. 저자는 블록 좌표 Frank-Wolfe 알고리즘과 함께 작업을 최적화하는 다양한 기법들을 제안합니다.

- **Performance Highlights**: 저자들은 실험을 통해 제안된 방법이 다양한 기법들과 비교하여 뛰어난 성능을 보임을 입증하였습니다. 특히, 제안된 방법은 균형 잡힌 커뮤니티 수를 탐지하는 데 탁월한 특성을 가지고 있으며, 이는 불균형한 커뮤니티 문제를 해결하는 데 기여합니다. 결과적으로, 이 논문은 공정하고 효율적인 polarized communities를 발견하는 데 기여하는 중요한 연구 결과를 제공합니다.



### Exploiting Ensemble Learning for Cross-View Isolated Sign Language Recognition (https://arxiv.org/abs/2502.02196)
Comments:
          3rd Place in Cross-View Isolated Sign Language Recognition Challenge at WWW 2025

- **What's New**: 이 논문에서는 WWW 2025에서 열린 Cross-View Isolated Sign Language Recognition (CV-ISLR) 챌린지에 대한 새로운 솔루션을 제시합니다. 전통적인 ISLR 방법들이 주로 정면에서 촬영된 데이터를 사용하는 반면, CV-ISLR은 다양한 카메라 각도에서 수화 영상을 인식하는 데 중점을 두고 있습니다. 이 연구는 Ensemble Learning을 활용하여 다양한 시점에서 모델의 강건성과 일반화를 향상시키는 접근 방식을 탐구합니다.

- **Technical Details**: CV-ISLR 문제는 뷰포인트 변동성과 제스처 복잡성을 포함한 두 가지 주요 도전 과제가 있습니다. 이를 해결하기 위해, 우리는 Video Swin Transformer (VST) 아키텍처에 Ensemble Learning을 통합하려고 합니다. 다양한 크기의 VST 변형을 RGB 및 Depth 입력에 적용하여 각각의 모델이 다양한 특징을 추출하고 융합할 수 있도록 합니다.

- **Performance Highlights**: 최종 모델은 RGB 및 Depth 스트림의 결과를 통합하여 더 정교한 예측을 제공합니다. 본 연구에서 제안한 방법은 RGB 기반 및 RGB-D 기반 ISLR 트랙에서 각각 3위로 랭크되었으며, 교차 뷰 인식의 문제를 잘 처리함을 보여줍니다. 이는 다양한 각도에서의 수어 인식 정확도를 높이는 효과적인 접근임을 입증합니다.



### ShapeShifter: 3D Variations Using Multiscale and Sparse Point-Voxel Diffusion (https://arxiv.org/abs/2502.02187)
- **What's New**: ShapeShifter라는 새로운 3D 생성 모델이 제안되었습니다. 이 모델은 단일 참조 모델을 기반으로 형태 변형을 합성하는 것을 학습하며, 기존의 3D 생성 방법들이 가지고 있는 기하학적 세부 사항 부족 및 긴 교육 시간 등을 해결합니다. 또한, 이 모델은 정밀한 세부 묘사를 보존하고 다양한 표면 형태를 처리할 수 있는 능력이 향상되었습니다.

- **Technical Details**: ShapeShifter는 희소 볼륨 격자(sparse voxel grid) 및 포인트, 노멀, 색상 샘플링을 결합한 다중 스케일 신경망 아키텍처를 사용합니다. 이 접근 방식은 계산 효율성을 높이고 빠른 추론(interactive inference)을 가능하게 하여, 예시(input) 형태에 따라 고품질 형태 변형을 생성합니다. 포인트 샘플링과 희소 합성을 결합하여 서로 다른 스타일과 톱로지를 가진 3D 변형을 만들어내는 멀티스케일 생성 방식을 구현하고 있습니다.

- **Performance Highlights**: ShapeShifter는 훈련 시간을 크게 줄일 수 있으며, 대개 몇 분 이내에 훈련이 완료됩니다. 이 결과는 텍스처가 추가된 메시(mesh)로 쉽게 변환할 수 있으며, 아티스트에 의해 안내되는 반복적 공동 창작(iterative co-creation)이 가능합니다. 최종적으로, 높은 품질의 기하학적 모델 출력은 필요에 따라 텍스처가 추가될 수 있습니다.



### Mass-Editing Memory with Attention in Transformers: A cross-lingual exploration of knowledg (https://arxiv.org/abs/2502.02173)
- **What's New**: 최근 연구에서는 대형 언어 모델(LLM)의 사실적 지식을 업데이트하고 수정하는 방법이 탐구되었습니다. 이 연구는 기존의 지식 편집 방법이 여러 언어에서 얼마나 효과적인지를 검토하며, 특히 Attention Mechanism의 역할에 초점을 맞추고 있습니다. 이를 바탕으로 Mass-Editing Memory with Attention in Transformers (MEMAT)를 제안하며, 이는 매개변수 수정을 최소화하면서 모든 지표에서 значительные 개선을 이룬다는 점에서 혁신적입니다.

- **Technical Details**: 대형 언어 모델은 실제 의미 이해보다는 문장에서 토큰 발생 확률을 예측하도록 설계되어, 현실과 정확성이 결여된 콘텐츠 생성을 초래할 수 있습니다. 이 연구에서는 MEMIT이라는 지식 편집 방법을 분석하며, 영어와 카탈로니아어 간의 교차 언어 능력을 검토합니다. 메모리 내에서 지식 삽입 시, Attention Head의 역할은 모델의 신뢰성을 높이는데 기여하며, 내부 구조에서 사실적 연관성을 평가하는 중요한 정보 원천으로 작용합니다.

- **Performance Highlights**: MEMAT 방법은 모든 평가 지표에서 개선을 보여주며, 어떤 경우에는 10% 이상의 차이를 나타냅니다. 이 알고리즘은 기존 지식에 대한 이해를 증진시키고, 새로운 지식을 삽입하는 대신 효율적으로 작동함을 입증하였습니다. 추후 실험 결과는 MEMAT이 기존 모델에 비해 포터블하고 연산적으로 효율적이라는 것을 보여줍니다.



### Graph Neural Networks for O-RAN Mobility Management: A Link Prediction Approach (https://arxiv.org/abs/2502.02170)
Comments:
          7 pages, 2 figures, 2 tables. Submitted to IEEE Vehicular Technology Magazine, Special Issue on "AI for 6G O-RAN Intelligent, Cost-Efficient and Secure Automation"

- **What's New**: 이번 논문은 6G 네트워크의 이동성 관리를 위한 선제적 핸드오버(CHO) 프레임워크를 제안합니다. 기존의 반응적 핸드오버 전략의 한계를 극복하기 위해 사용자-셀 링크 예측을 활용하는 혁신적인 방법론을 도입하였습니다. 여러 종류의 그래프 신경망(GNN)을 통해 핸드오버 성능을 향상시킬 수 있는 가능성을 탐구하고 있습니다.

- **Technical Details**: 핸드오버 결정 과정에서 그래프 신경망(GNN)을 적용하여 각 사용자에 대한 최적의 셀을 예측하는 방법론을 논의합니다. GNN은 그래프 구조 데이터를 처리하여 인접 노드로부터 정보를 집계하고 링크 예측 및 노드/엣지 회귀 작업을 수행합니다. 두 가지 GNN 모델을 실제 데이터 세트를 사용해 비교하며 이 모델들이 이동성 관리 문제에 적용될 때의 복잡성을 분석합니다.

- **Performance Highlights**: 모델 실험 결과, GNN 기반의 링크 예측 모델이 이동성 관리를 위한 프로세스를 최적화하는 데 효과적임을 보여주었습니다. 기존의 전통적인 방법보다 낮은 신호 오버헤드를 유지하면서 핸드오버 타이밍을 최적화할 수 있음을 강조합니다. 이 연구는 O-RAN 시스템에서 GNN 링크 예측의 적용 가능성에 대한 중요한 통찰력을 제공합니다.



### Synthesis of Model Predictive Control and Reinforcement Learning: Survey and Classification (https://arxiv.org/abs/2502.02133)
- **What's New**: 이 논문은 MPC (Model Predictive Control)와 RL (Reinforcement Learning)의 발전된 조합 방법론에 대한 통찰을 제공합니다. 두 가지 접근 방식은 많은 유사성과 서로 보완적인 장점을 지니고 있지만, 매우 다른 패러다임에서 기인했고 다양한 요구 사항에 따라 발전했습니다. 이 연구는 특히 actor-critic RL 접근 방식을 중심으로 두 기술의 차이점과 유사성을 분석하고, 이를 통해 혼합 알고리즘 개발에 대한 체계적인 개요를 제공합니다.

- **Technical Details**: MPC는 최적 제어를 위해 환경 모델을 기반으로 하는 최적화 문제를 해결하는 반면, RL은 환경과 상호작용하며 정책을 개선하는 방식입니다. MPC는 실시간 데이터가 부족하고 최적화 친화적인 모델로 설명할 수 있는 환경에서 효과적인 반면, RL은 데이터를 많이 생성할 수 있는 설정에서 잘 작동합니다. 이 논문에서는 MPC와 RL의 상호 보완적인 이점을 적극 활용하여 새로운 알고리즘을 제안하고 있습니다.

- **Performance Highlights**: MPC는 제약 조건을 보장할 수 있는 강력한 방법이지만 안전 문제에서 어려움을 겪고 있으며, RL은 많은 데이터를 활용하여 좋은 정책을 학습할 수 있는 강점을 지닙니다. 연구자들은 이 두 기법의 장점을 결합하여 새로운 알고리즘을 개발하려는 수많은 연구를 하고 있으며, 본 논문에서는 이러한 접근 방식의 체계적인 분류를 제공하고 있습니다. 이 과정에서 이들 각각의 방법론이 실제 시스템에서 어떻게 응용될 수 있을지를 탐구하고 있습니다.



### How Memory in Optimization Algorithms Implicitly Modifies the Loss (https://arxiv.org/abs/2502.02132)
- **What's New**: 본 연구에서는 딥러닝에서 사용되는 현대 최적화 방법들이 과거 반복의 기록에 의존하고 있으며, 이러한 의존성은 시간이 지남에 따라 빠르게 감소한다는 점을 설명합니다. 특히, 모멘텀을 사용하는 경량경사 하강법에서 과거 경량의 지수적으로 감소하는 메모 리를 다룹니다. 이에 따라 메모리가 없는 알고리즘을 식별하는 일반적인 기법을 소개하며, 이 기법은 현재 반복값을 사용하여 업데이트를 수행하고 메모리에서 발생하는 수정 항을 추가하는 방식입니다.

- **Technical Details**: 기술적으로, 수정 항(t correction term)은 손실의 섭동(perturbation)으로 해석될 수 있으며, 이러한 섭동의 성질은 최적화 동역학에서 메모리가 암시적으로 (비교거)정규화(anti-regularizes)하는 방식을 이해하는 데 도움을 줄 수 있습니다. 메모리의 영향 없이 최적화 알고리즘에 접근하는 기법은 과거 반복값을 대체하고 현재 반복값으로만 갱신하는 구조를 따릅니다. 이러한 구조는 메모리가 인지화된 메아리(echo)를 제거하여 더 간결한 업데이트 방법을 제안합니다.

- **Performance Highlights**: 응용 연구로, Lion 알고리즘은 AdamW가 유도하는 메모리에 의해 발생하는 암묵적 (anti-)정규화가 없다는 사실을 발견했습니다. 이 발견은 최근 문서화된 Lion의 더 나은 일반화 성능에 대한 이론적 설명을 제공합니다. 결국 Lion 알고리즘은 메모리 부족의 이점을 활용하여 성능을 향상시키고, 다른 알고리즘보다 더 우수한 결과를 보여줍니다.



### Causally-informed Deep Learning towards Explainable and Generalizable Outcomes Prediction in Critical Car (https://arxiv.org/abs/2502.02109)
- **What's New**: 최근의 딥러닝(Deep Learning) 발전은 높은 성능의 조기 경고 점수 시스템(Early Warning Score, EWS) 개발을 촉진했습니다. 이 시스템은 급성 신장 손상, 급성 심근 경색 및 순환 부전과 같은 임상 악화를 예측합니다. 본 논문에서는 인과 발견(Causal Discovery)을 활용하여 예측의 기초가 되는 인과 관계를 식별하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 모델은 인과적 정보를 반영하는 설명 가능한 조기 예측 모델로, 예측의 명확한 해석을 제공하며 낯선 환경에서도 우수한 성능을 보여줍니다. 이 모델은 6가지 서로 다른 중요 악화 상황에서의 정확도를 향상시키며, 다양한 환자 집단에 대해 보다 나은 일반화 능력을 발휘합니다. 또한, 우리는 임상 진단 및 잠재적 개입에 대한 참고 자료로 활용할 수 있는 명시적 인과 경로를 제공합니다.

- **Performance Highlights**: 제안된 접근 방식은 여러 기준 알고리즘에 비해 다양한 환자 집단에서 우수한 정확도를 달성하며, 의료 시나리오에서 딥러닝의 실용성을 강화합니다. 본 논문을 통해 제시된 EWS 시스템은 다양한 결과에 적용 가능함으로써 다양한 임상 환경에서 임상적으로 유용한 도구로 발전할 수 있음을 보여줍니다.



### Neural Networks Learn Distance Metrics (https://arxiv.org/abs/2502.02103)
Comments:
          14 pages, 1 figures. Code and additional resources available at this https URL

- **What's New**: 이 논문은 신경망이 거리 기반 표현을 자연적으로 선호하는지를 조사하고, 이러한 표현이 모델 성능에 미치는 영향을 분석합니다. 특히, Mahalanobis 거리 방정식을 기반으로 한 새로운 아키텍처인 OffsetL2를 소개하여 기존의 강력한 성능을 검증합니다. 저자들은 이러한 연구가 신경망 설계 수행 시 거리 기반 학습의 중요성을 부각시킨다고 주장합니다.

- **Technical Details**: 본 연구에서는 신경망이 특징을 학습하는 방식에서 거리 기반 표현과 강도 기반 표현을 구분짓는 이론적 프레임워크를 제시합니다. 또한, 거리 기반 표현은 학습된 프로토타입에 대한 근접성을 바탕으로 특징을 인코딩하는 반면, 강도 기반 표현은 활성화 크기를 통해 특징을 측정합니다. 이를 통해 연구자들은 거리 기반 학습이 신경망의 내부 표현을 설명하는 중요한 원리임을 강조합니다.

- **Performance Highlights**: 실험 결과, 거리 기반 표현을 사용하는 신경망 아키텍처는 기존 강도 기반 아키텍처보다 학습 성능이 우수함을 보여주었습니다. 또한, 여섯 가지 MNIST 아키텍처 변형을 통해 거리 및 강도 표현 간의 작동 메커니즘과 공간적 성능 한계를 분석하였습니다. 이러한 결과는 거리 기반 접근법이 실제 문제 해결에서 더 나은 결과를 낳을 수 있다는 것을 암시합니다.



### IPO: Iterative Preference Optimization for Text-to-Video Generation (https://arxiv.org/abs/2502.02088)
- **What's New**: 이번 논문에서는 비디오 생성 모델의 품질을 개선하기 위해 인간의 선호와 모델을 정렬하는 Iterative Preference Optimization (IPO) 전략을 소개합니다. 이는 비디오 품질 향상에 필요한 인간 피드백을 통합하여, 주제의 일관성, 매끄러운 동작, 미적 품질 등을 고려합니다. IPO는 멀티모달 대형 언어 모델(MLLM)을 활용하여 자동으로 선호 라벨을 부여하며, tedious한 수동 라벨링 없이 반복적인 선호 최적화를 가능하게 합니다.

- **Technical Details**: IPO는 세 가지 핵심 요소로 구성됩니다: 선호 데이터셋(preference dataset), 비평가 모델(critic model), 반복 학습 프레임워크(iterative learning framework)입니다. 선호 데이터셋을 통해 비평가 모델을 훈련시키고, 이를 활용하여 T2V(Text-to-Video) 모델의 비디오 품질을 정당화하며, 다양한 비디오 품질 라벨을 자동으로 생성할 수 있습니다. 이 과정은 Diffusion-DPO 또는 Diffusion-KTO 기법을 사용하여 T2V 모델의 성능을 계속적으로 강화합니다.

- **Performance Highlights**: VBench 벤치마크에서 IPO는 사전 훈련된 모델의 비디오 생성 품질을 효과적으로 개선하는 결과를 나타냈습니다. 특히, 2B 매개변수를 가진 모델이 5B 매개변수를 가진 모델을 초월하는 성과를 기록하며, 새로운 최첨단 성능을 달성했습니다. 연구진은 이 연구의 데이터셋, 코드 및 모델을 공개하여 향후 비디오 생성 연구를 촉진할 계획입니다.



### Online Clustering of Dueling Bandits (https://arxiv.org/abs/2502.02079)
Comments:
          Preprint

- **What's New**: 본 논문에서는 preference feedback(선호 피드백)을 기반으로 하는 다중 사용자 협업을 가능하게 하는 첫 번째 'Clustering of Dueling Bandits' 알고리즘을 소개합니다. 이는 기존의 클러스터링 방법의 한계를 극복하기 위한 것으로, 사용자가 쌍으로 제시된 추천 항목에 대해 선택한 선호도 정보를 공유할 수 있도록 합니다. 이를 통해 추천 시스템 등의 실제 적용에서 성능을 크게 향상시킬 수 있습니다.

- **Technical Details**: 제안된 두 가지 알고리즘인 'Clustering Of Linear Dueling Bandits (COLDB)' 및 'Clustering Of Neural Dueling Bandits (CONDB)'는 사용자의 보상 함수 모델링에 있어 각각 선형 모델과 신경망을 사용합니다. COLDB는 사용자 보상 함수가 문맥 벡터의 선형 함수로 모델링되며, CONDB는 복잡한 비선형 사용자 보상 함수를 신경망으로 모델링합니다. 두 알고리즘 모두 사용자들의 클러스터링 구조를 그래프 형태로 표현하고 업데이트하여 사용자의 데이터를 활용하는 방식으로 진행됩니다.

- **Performance Highlights**: 이론적 분석 결과, 제안된 알고리즘들은 비선형 회귀의 상한을 가지며, 사용자 간의 협업이 클수록 성능이 향상된다는 것을 보여줍니다. 추가적인 실험을 통해 두 알고리즘은 합성 데이터 및 실제 데이터셋에서 유의미한 성과를 내며, 다중 사용자의 선호 기반 피드백을 활용한 상황에서 효율성을 증명합니다.



### ASCenD-BDS: Adaptable, Stochastic and Context-aware framework for Detection of Bias, Discrimination and Stereotyping (https://arxiv.org/abs/2502.02072)
Comments:
          17 pages, 6 Figures and this manuscript will be submitted to Q1,Q2 Journals

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 진화가 자연어 처리 분야에 변화를 가져온 것과 동시에, 편향(bias)과 차별(discrimination) 등의 문제에 대한 우려를 제기하고 있습니다. 특히, ASCenD BDS(Adaptable, Stochastic and Context-aware framework for Detection of Bias, Discrimination and Stereotyping)라는 새로운 프레임워크를 소개하여, 다양한 사회적 맥락에서 이러한 문제를 탐지할 수 있는 접근법을 제안합니다.

- **Technical Details**: ASCenD BDS는 성별, 계급, 나이, 장애, 사회경제적 상태, 언어적 변이 등 여러 카테고리를 통해 편향, 차별, 고정관념(stereotyping)을 탐지하는 방식입니다. 기존의 데이터셋에 의존하던 방식과는 달리, 이 프레임워크는 적응성(adaptability), 확률적(stochasticity), 맥락 인식(context awareness) 기능을 통해 보다 맞춤형으로 접근할 수 있는 장점을 가지고 있습니다. 특히, 인도의 문화적 맥락에 맞춘 내용을 통해 카테고리를 정립하고, 그에 맞는 데이터를 생성할 수 있도록 합니다.

- **Performance Highlights**: 이 프레임워크는 800개 이상의 STEM(Science, Technology, Engineering, Mathematics), 10개의 카테고리 및 31개의 고유 서브 카테고리를 개발하였으며, 이는 Saint Fox Consultancy Private Ltd의 컨설턴트 팀에 의해 수행되었습니다. 또한, SFCLabs에서 제품 개발의 일환으로 이 개념을 테스트하여 프레임워크의 실효성을 증명하였습니다. 이와 같은 성과는 향후 다양한 문화적 배경에서 편향을 탐지하고 해결하는 데 기여할 수 있을 것으로 기대됩니다.



### AdaptBot: Combining LLM with Knowledge Graphs and Human Input for Generic-to-Specific Task Decomposition and Knowledge Refinemen (https://arxiv.org/abs/2502.02067)
Comments:
          Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2025

- **What's New**: 이번 연구에서는 LLMs(대규모 언어 모델)와 KG(지식 그래프)의 결합을 통해 새로운 작업과 시나리오에 신속하게 적응할 수 있는 조력형 에이전트를 위한 새로운 프레임워크를 제안합니다. 기존의 지식이 부족한 환경에서도 인간의 피드백을 활용하여 LLM의 출력을 수정하고 향상시킬 수 있는 접근 방법을 소개합니다. 이러한 프레임워크는 이전의 당면 과제를 해결하기 위한 신뢰할 수 있는 대안을 제공하여 LLM의 일반적인 예측 능력을 폭넓게 활용합니다.

- **Technical Details**: 제안된 프레임워크는 LLM을 사용하여 주어진 작업을 수행하기 위한 행동 시퀀스(서브 태스크)를 생성하고, KG를 통해 도메인 특정 지식을 인코딩하여 LLM 출력의 수정 가능성을 지원합니다. 각 서브 태스크는 LLM 호출을 통해 만들어지며, KG의 정보와 비교하여 불일치가 발견될 경우 이를 해결하기 위한 대체 행동을 제시합니다. 이 과정에서 HUMAN-IN-THE-LOOP(HITL) 방식으로 인간의 피드백을 요청하여 KG를 수정하고 새로운 작업에 대한 적응력을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크를 사용한 조력형 에이전트는 기존 LLM이나 LLM과 KG의 조합 만으로 수행한 작업에 비해 성능이 향상된다는 것이 입증되었습니다. 요리와 청소 작업에 대한 시뮬레이션 평가에서 유의미한 성능 향상을 보여주며, 복잡한 작업 구성을 신속하게 적응하는 특성을 강조합니다. 이러한 성능 향상은 에이전트가 새로운 작업을 수행하는 방식과 기존 지식을 지속적으로 수정하는 능력에서 비롯됩니다.



### CASIM: Composite Aware Semantic Injection for Text to Motion Generation (https://arxiv.org/abs/2502.02063)
- **What's New**: 본 논문에서는 Composite Aware Semantic Injection Mechanism(CASIM)을 제안하여 고정 길이의 텍스트 임베딩을 사용하는 기존 방법의 한계를 극복합니다. CASIM은 복합적인 인간 동작의 특성과 텍스트 및 모션 토큰 간의 동적 정렬을 학습합니다. 이로 인해 모션 생성 품질과 텍스트-모션 정렬이 향상되어 실제 사용 가능한 수준의 동작이 가능해집니다.

- **Technical Details**: CASIM은 복합 인식 텍스트 인코더(composite-aware text encoder)와 텍스트-모션 정렬기(text-motion aligner)로 구성되어 있습니다. 이 기술은 텍스트 토큰과 모션 프레임 간의 동적 정렬을 통해 세밀한 의미 관계를 캡처합니다. CASIM은 모델 및 표현 방식에 구애받지 않아, 서로 다른 생성 모델에 쉽게 통합할 수 있는 장점을 가집니다.

- **Performance Highlights**: 실험 결과, HumanML3D 및 KIT 벤치마크에서 CASIM을 적용한 다양한 최신 방법들이 FID, R-precision 및 Multimodality Distance에서 현저한 개선을 보임을 보여줍니다. CASIM은 긴 기간의 인간 동작 생성에서도 모션 품질과 텍스트-모션 정렬을 개선하였으며, 이러한 성능은 정량적 분석뿐 아니라 주목(attention) 가중치 분석을 통해 확인되었습니다. 이 새로운 접근 방식은 고정 길이의 의미 주입 방법 대비 우수한 성능을 입증하였습니다.



### RAPID: Robust and Agile Planner Using Inverse Reinforcement Learning for Vision-Based Drone Navigation (https://arxiv.org/abs/2502.02054)
Comments:
          18 pages, 11 figures, 58 references, and appendix is included

- **What's New**: 이번 논문에서는 복잡한 환경에서 민첩한 드론 비행을 위한 학습 기반 시각 계획 시스템인 RAPID를 소개합니다. 이 시스템은 인버스 강화 학습(ILR)에 기반하여 충돌이 없는 경로를 수 밀리초 안에 생성하며, 별도의 인식, 매핑 및 계획 모듈 없이 작동할 수 있습니다. 또한, 제안된 방법은 시뮬레이션 환경에서만 학습되어도 실세계에 직접 적용 가능하다는 점에서 참신합니다.

- **Technical Details**: RAPID는 고속 시각 내비게이션을 위한 인버스 소프트 Q-학습 기반 프레임워크를 개발하였으며, 수동 보상이 필요 없는 견고한 학습을 달성합니다. 이를 위해 고차원 시각 정보에서 복잡성을 줄이는 보조 오토 인코더 손실 함수를 도입하였으며, 고속 시나리오에 맞춰 흡수 상태 처리를 통합했습니다. 또한, 시뮬레이션에서 학습된 정책은 평균 7 m/s의 속도로 잘 수행되었습니다.

- **Performance Highlights**: 제안된 방법의 성능은 숲과 다양한 구조물에서는 물론, 다양한 실제 환경에서 검증되었습니다. RAPID의 학습된 정책은 최고 8.8 m/s의 최대 속도로 비행 실험에서 안정적인 성과를 보여주었습니다. 이 연구는 드론의 고속 시각 내비게이션을 위한 첫 번째 IRL 기반 접근 방식을 제시하며, 실제 환경에서도 수행 가능성을 입증했습니다.



### M2R2: Mixture of Multi-Rate Residuals for Efficient Transformer Inferenc (https://arxiv.org/abs/2502.02040)
- **What's New**: 이 논문에서는 Mixture of Multi-rate Residuals (M2R2)라는 새로운 프레임워크를 소개하며, 이 프레임워크는 잔여 변환의 속도를 동적으로 조절하여 조기 정렬을 최적화합니다. 이를 통해 다양한 추론 패러다임에서 효율성을 향상시키고, 생성 품질과 속도 간의 최적의 균형을 달성할 수 있습니다. M2R2는 self-speculative decoding 방법을 초월하여 MT-Bench에서 최대 2.8배의 속도를 기록하는 하이라이트를 제공합니다.

- **Technical Details**: M2R2는 토큰 수준에서 잔여 변환 속도를 조절하면서 초기 단계를 빠르게 정렬할 수 있도록 설계되었습니다. 이 방법은 동적 컴퓨팅 환경과 Mixture-of-Experts (MoE) 아키텍처에서 성능을 개선할 수 있도록 고안되었습니다. 연구에서 실험을 통해 잔여 변환의 속도가 토큰의 표현을 더 빠르게 처리하는 데 어떻게 기여하는지를 보여줍니다.

- **Performance Highlights**: M2R2는 기존의 self-speculative decoding 방법과 비교하여 MT-Bench에서 2.8배의 속도 향상을 달성하였고, MoE 모델에서는 전문가 로딩과 컴퓨팅 연산을 동시에 수행하여 최대 2.9배의 속도 향상을 이루어냈습니다. 이는 자원이 제한된 환경에서도 효과적으로 작동할 수 있는 가능성을 보여줍니다.



### From Human Hands to Robotic Limbs: A Study in Motor Skill Embodiment for Telemanipulation (https://arxiv.org/abs/2502.02036)
- **What's New**: 이 논문에서는 사람의 팔 제스처를 사용하여 다중 자유도(robot manipulator)를 원격 조종하는 시스템을 제안합니다. 이를 위해 GRU 기반의 Variational Autoencoder(VAE)를 활용하여 매니퓰레이터의 복잡한 관절 운동학을 포착하는 잠재적 표현(latent representation)을 학습합니다. 이 시스템은 인간 팔의 구성 요소를 잠재 공간(latent space)으로 매핑함으로써, 새로운 매니퓰레이터 경로를 실시간으로 생성할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 방법은 Gate Recurrent Unit(GRU)에 기반한 Variational Autoencoder(VAE) 아키텍처를 사용하여 7-DOF Kinova 매니퓰레이터의 구성 공간을 나타내는 잠재적 분포(latent distribution)를 학습합니다. 인간의 팔 동작을 이 잠재 공간으로 매핑하기 위해 피드포워드 신경망을 훈련하고, 이를 통해 VAE 디코더를 사용하여 해당하는 로봇 매니퓰레이터 경로를 생성합니다. 이 과정은 기계 학습 기반의 접근 방식을 통해 인간의 운동 습관을 모방하며, 새로운 작업 시나리오에 대응할 수 있는 훈련 모델 학습 프레임워크를 포함하고 있습니다.

- **Performance Highlights**: 이 시스템은 학습 중에 주어진 적재적소의 인간 특징을 바탕으로 새로운 매니퓰레이터 구성을 생성할 수 있는 가능성을 보여줍니다. 효과적인 원격 조작을 가능하게 하여, 매니퓰레이터의 경로와 조작 성능에 대한 심각한 향상을 가져오는 결과를 나타냈습니다. 이러한 접근 방식은 기존의 작업 환경에서 인간의 동작을 더욱 intuitively 반영하여, 협력 로봇(co-robotics) 작업 시나리오에 적합한 인터페이스를 제공합니다.



### Heteroscedastic Double Bayesian Elastic N (https://arxiv.org/abs/2502.02032)
- **What's New**: HDBEN(Heteroscedastic Double Bayesian Elastic Net)는 평균과 로그 분산을 동시에 모델링하는 새로운 프레임워크입니다. 기존의 Bayesian Elastic Net 방식을 활용하면서도 이질적 분산(heteroscedasticity)을 고려하여 변수 선택과 회귀 계수 추정의 효율성을 높였습니다. 이 모델은 다차원(high-dimensional) 데이터에 특히 잘 작동하며, 복잡한 분산 구조를 포착할 수 있는 장점이 있습니다.

- **Technical Details**: HDBEN은 계층적 베이지안 사전 분포(hierarchical Bayesian priors)와 함께 ℓ1 및 ℓ2 패널티를 조합하여 평균과 분산을 동시에 규제합니다. 특히, 이 모델은 처리 성능을 개선하기 위해 HDBEN의 베이지안 모델링이 결과의 변동성(variable variability)과 평균(mean behavior) 사이의 복잡한 상호작용을 효과적으로 포착한다는 점을 강조합니다. 이 모델은 다양한 응용 분야에서의 불확실성 정량화(quantification)를 통해 더 완전한 프로세스 모델을 제공합니다.

- **Performance Highlights**: 시뮬레이션 연구에서는 HDBEN이 기존 방법들에 비해 뛰어난 성능을 입증했습니다. 특히, 이질적 분산이 있는 경우와 다차원 데이터에서 HDBEN이 더 효과적인 변수 선택(variable selection)과 계수 추정(coefficient estimation)을 수행하는 것을 보여주었습니다. 이러한 결과는 HDBEN이 다양한 실제 데이터에서 보다 신뢰할 수 있는 추정 결과를 제공함을 시사합니다.



### Fine-tuning Language Models for Recipe Generation: A Comparative Analysis and Benchmark Study (https://arxiv.org/abs/2502.02028)
Comments:
          15 pages, 8 figures

- **What's New**: 이 연구는 다양한 소형 언어 모델을 미세 조정(fine-tuning)하여 조리법 생성(task of recipe generation)을 탐구하고 개발하는 데 초점을 맞추고 있습니다. 특히 조리법 생성의 열린 과제에 대한 비교와 함께 강력한 평가 지표(evaluation metrics)를 개발하는 데 중점을 두었습니다. 본 연구는 T5-small, SmolLM-135M, Phi-2과 같은 여러 모델 아키텍처에 대한 광범위한 실험을 수행하였습니다.

- **Technical Details**: 연구에서는 Food.com 데이터셋을 활용하여 180,000개 이상의 조리법과 700,000개의 리뷰를 분석했습니다. 이를 통해 조리법 이름, 재료 목록, 조리법 지침, 영양 정보 등을 정형화하여 입력-출력 쌍을 생성하는 데이터 전처리 파이프라인을 구축하였습니다. 또한 알레르기 대체를 위한 RAG 및 프롬프트 기반 접근 방식을 개발하였으며, 조리법 생성의 품질을 평가하기 위해 새로운 조리법 특화 평가 지표를 포함한 다차원 평가 프레임워크를 제안하였습니다.

- **Performance Highlights**: SmolLM-360M과 SmolLM-1.7B는 크기 차이에도 불구하고 유사한 성능을 보였으나, Phi-2는 더 많은 파라미터에도 불구하고 조리법 생성에서 한계를 나타내었습니다. 실험 결과, 더 큰 모델들이 표준 지표에서 일반적으로 더 나은 성능을 보였지만, 도메인 특화 지표를 고려할 때 모델 크기와 조리법 품질 간의 관계는 더 복잡함을 보여주었습니다. 본 연구는 조리법 생성과 관련된 NLG(Natural Language Generation) 작업에서 도메인 전문성과 안전성의 중요성에 대한 통찰력을 제공합니다.



### From Fog to Failure: How Dehazing Can Harm Clear Image Object Detection (https://arxiv.org/abs/2502.02027)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2410.01225

- **What's New**: 이번 연구는 객체 탐지에 인체의 시각적 단서를 통합하는 데 있어 발생하는 도전 과제를 탐구하고, 다단계 프레임워크를 제안합니다. 이 프레임워크는 경량 탐지기가 관심 영역(Region of Interest, RoI)을 식별하고, 이 후 공간적 주의력에 기반한 디헤이징으로 강조한 후, 더욱 강력한 탐지 모델로 최종 탐지를 진행합니다. 이로 인해 안개 조건에서는 효과적인 성능을 보여주지만, 명확한 이미지를 처리할 때는 예상치 못한 성능 저하가 발생하는 점을 분석합니다.

- **Technical Details**: 제안된 방법론은 대기 산란 모델과 인간 시각 피질의 원리를 활용하여 저조도 조건에서 객체 탐지를 향상시키는 딥 러닝 프레임워크를 제시합니다. 이 파이프라인은 경량 탐지 모델로 시작하여 관심 영역을 식별하고, 디헤이징 과정에서 공간적 주의력을 개발하여 중요 특징을 보존하고 계산 비용을 줄입니다. 이후 더욱 견고한 탐지 모델이 객체 인식을 정제하고 개선합니다.

- **Performance Highlights**: Foggy Cityscapes 데이터셋에서의 성능 평가를 통해 AOD-NetX 통합된 모델이 안개 이미지에서 우수한 성능을 보이는 반면, 전통적인 모델들은 맑은 이미지에서 안개 이미지로 전환될 때 평균정밀도(mAP)에서 자연스러운 감소를 보입니다. 우리의 파이프라인은 다양한 저조도 조건에서도 강력한 성능을 보여주며, SSIM과 PSNR을 포함한 평가 지표가 사용되었습니다.



### Multi-Domain Graph Foundation Models: Robust Knowledge Transfer via Topology Alignmen (https://arxiv.org/abs/2502.02017)
- **What's New**: 최근 CV(Computer Vision)와 NLP(Natural Language Processing) 분야의 발전은 연구자들이 다양한 도메인을 아우르는 일반 목적의 그래프 기초 모델을 개발하도록 영감을 주었습니다. 그러나 도메인 간 그래프 토폴로지(topology)의 상당한 차이가 기본적인 도전 과제가 되고 있습니다. 이를 해결하기 위해 제안된 MDGFM(Multi-Domain Graph Foundation Model)은 도메인 간 구조적 정보를 정렬하고 활용하여 강력한 지식 전달을 촉진하는 통합 프레임워크입니다.

- **Technical Details**: MDGFM은 특징과 토폴로지 정보를 동적으로 가중치로 조정하는 적응형 균형 토큰을 포함한 분리된 임베딩 메커니즘을 도입하여 주요 정보를 포착합니다. 또한, 내재된 노이즈를 해결하고 토폴로지를 정렬하기 위해 강력하고 도메인 불변 지식을 학습하는 그래프 구조 학습(Graph Structure Learning) 모듈을 통합합니다. 이를 통해 강력한 일반화 및 전이 가능성을 강화하며, 효율적인 프롬프트 학습(prompt learning) 전략을 개발하여 지식을 목표 도메인으로 전달합니다.

- **Performance Highlights**: 대규모 실험을 통해 MDGFM은 동질(homophilic) 및 이질(heterophilic) 그래프 데이터셋 모두에서 강 robustness 및 efficacy를 발휘함을 입증하였습니다. 이 모델은 여러 도메인에서 얻은 지식을 효율적으로 전이할 수 있는 능력을 보여주고 있으며, 높은 도메인 일반화(domain generalization) 능력을 제공함으로써 기존의 한정적인 단일 도메인 모델의 한계를 극복합니다.



### A Periodic Bayesian Flow for Material Generation (https://arxiv.org/abs/2502.02016)
Comments:
          Accepted to ICLR25

- **What's New**: 이 논문은 기존의 Bayesian Flow Networks (BFN)을 크리스탈 데이터의 비유클리드 모델링에 적용하여 새로운 가능성을 탐구하고 있습니다. 새로운 CrysBFN 모델은 비모노토닉 엔트로피 동역학을 특징으로 하며, 이는 기존의 정규 분포 기반 접근 방식와 구별되는 중요한 혁신입니다. 이 모델은 엔트로피 조건화 메커니즘을 통합하여 시계열 조건화 대비 높은 성능을 보여주며, 향후 크리스탈 생성 분야에 획기적인 기여를 할 것으로 기대됩니다.

- **Technical Details**: CrysBFN은 주기적인 Bayesian Flow를 통해 비유클리드 공간, 즉 하이퍼 토러스에서 크리스탈 변수를 모델링하는 최초의 접근 방식을 제안합니다. 이 모델은 엔트로피(conditioning) 메커니즘을 갖추고 있으며, 계산 효율성을 보장하는 비자기 회귀 형태의 Bayesian 흐름 분포를 통한 훈련 패러다임을 적용합니다. 이를 통해, 크리스탈 생성 작업에 적합한 첫 번째 주기적 E(3) 공변 Bayesian 흐름 네트워크가 구현되었습니다.

- **Performance Highlights**: CrysBFN은 광범위한 실험을 통해 기존의 방법들과 비교해 샘플 품질과 효율성에서 뛰어난 성능을 보였습니다. 특히, Carbon-24의 ab initio 크리스탈 생성에서 99.1%의 COV-P를 달성하였고, MP-20 데이터셋에 대한 내부 구조 예측 작업에서는 64.35%의 일치율을 기록했습니다. 샘플링 효율성 실험에서도 CrysBFN은 약 100배의 향상된 속도를 보여주며, 이는 크리스탈 구조 생성을 위한 차세대 접근 방식으로 자리매김할 가능성을 시사합니다.



### Analytical Lyapunov Function Discovery: An RL-based Generative Approach (https://arxiv.org/abs/2502.02014)
Comments:
          26 pages (8+18), preprint for discussion. Haohan and Jie contribute equally

- **What's New**: 이 논문에서는 비선형 동적 시스템을 위한 유효한 Lyapunov 함수를 발견하는 새로운 엔드 투 엔드 프레임워크를 소개합니다. 기존의 신경망 접근 방식이 확장성(Scalability) 및 해석 가능성(Interpretability)의 두 가지 주요 문제를 겪고 있던 반면, 우리의 방법은 Transformers를 활용하여 분석적인 Lyapunov 함수를 직접 생성하고 검증하는 방식을 채택합니다. 이는 제어 엔지니어에게 심층적인 통찰을 제공하고 형식적 검증을 단순화 합니다.

- **Technical Details**: 제안된 프레임워크는 Transformer를 기반으로 한 훈련기(Trainer)와 후보 Lyapunov 함수의 검증 및 모델 개선을 위한 무위험 정책 경량화(Risk-seeking Policy Gradient)로 구성되어 있습니다. 고차원 비다항식(system) 시스템을 대상으로 강화 학습(Reinforcement Learning) 방식으로 학습하며, 후보 표현의 최소값 근처에서 국소 샘플링(Local Sampling)을 통해 Lyapunov 조건을 검증합니다. 이 과정에서는 검증 소프트웨어 도구와 효율적인 최적화 방법을 결합하여 훈련 동안 거짓 발견(falsification)을 수행합니다.

- **Performance Highlights**: 본 연구에서는 10차원 이상의 비선형 동적 시스템에서 Lyapunov 함수를 새롭게 발견하는 성과를 보여줍니다. 특히, 통신선 손실이 있는 전력 시스템 주파수 제어에 대한 유효한 지역적 Lyapunov 함수(Local Lyapunov Function)를 발견했으며, 이는 기존 문헌에서 전혀 탐지되지 않았던 것입니다. 이러한 접근 방식은 기존의 신경망 기반 방법보다 높은 효율성과 적용 가능성을 갖추고 있습니다.



### Layer by Layer: Uncovering Hidden Representations in Language Models (https://arxiv.org/abs/2502.02013)
- **What's New**: 이 논문에서는 중간 층(intermediate layers)의 성능을 분석하여, 언어 모델에서 마지막 층(final layer)보다 더 나은 표현을 제공할 수 있음을 밝혔다. 특히, 32개의 텍스트 임베딩 작업을 통해 중간 층의 표현력 향상 효과를 확인했으며, 이는 전통적인 마지막 층 의존성에 도전하는 발견이다.

- **Technical Details**: 우리는 정보 이론(information theory), 기하학(geometry), 그리고 입력 섭동(input perturbations)에 대한 불변성(invariance)을 기반으로 한 통합 프레임워크를 제안한다. 이 프레임워크는 각 모델 층이 정보 압축(compression)과 신호 보존(signal preservation)을 어떻게 균형 있게 조절하는지를 설명하면서 중간 층의 효과성을 이해하는 데 도움이 된다.

- **Performance Highlights**: 중간 층의 임베딩은 대부분의 아키텍처에서 마지막 층과 비교해 일관되게 더 강력한 기능을 제공하며, 특히 자동회귀 모델의 경우 중간 층의 ‘압축 계곡(compression valley)’ 현상을 보여준다. 이러한 연구 결과는 모델 디자인과 학습 관행에 새로운 방향성을 제공하며, AI 시스템의 Robustness와 정확성을 향상시킬 수 있는 기회를 열어준다.



### LLMSecConfig: An LLM-Based Approach for Fixing Software Container Misconfigurations (https://arxiv.org/abs/2502.02009)
- **What's New**: 본 연구에서는 LLMSecConfig라는 혁신적인 프레임워크를 소개합니다. 이 프레임워크는 정적 분석 도구(Static Analysis Tools, SATs)와 대형 언어 모델(Large Language Models, LLMs)을 결합하여 보안 구성 오류를 자동으로 수정하는 방법을 제공합니다. 기존 수동 방식의 한계를 극복하고, 자동화된 컨테이너 보안 관리의 가능성을 제시합니다.

- **Technical Details**: LLMSecConfig는 SAT 통합, 컨텍스트 검색 및 수리 생성 및 검증의 세 가지 상호 연결된 단계로 구성된 end-to-end 파이프라인을 제공합니다. 이 과정에서 Checkov를 주요 도구로 사용하여 1,000개 이상의 보안 정책을 효과적으로 적용하고, 자동으로 보안 구성 오류를 탐지 및 수정합니다.

- **Performance Highlights**: 1,000개의 실제 Kubernetes 구성에 대한 평가 결과, 94%의 성공적인 수리 비율을 기록하며 새로운 오류의 발생 비율은 낮았습니다(0.024). 이는 GPT-4o-mini와 같은 기존 모델보다 월등한 성과로, 자동화된 수리 접근 방식이 컨테이너 보안 관리에 있어 실질적인 개선을 가져올 수 있음을 보여줍니다.



### Theoretical and Practical Analysis of Fr\'echet Regression via Comparison Geometry (https://arxiv.org/abs/2502.01995)
- **What's New**: 이 논문에서는 Fréchet 회귀(Fréchet regression)를 통해 비유클리드 메트릭 공간(non-Euclidean metric spaces)에서의 데이터 관계 분석을 수행합니다. 기존의 회귀 방법을 확장하여 다양체(manifolds)와 그래프(graphs)와 같은 복잡한 구조에서의 데이터 분석을 가능하게 합니다. 이러한 방법론이 실용적으로 어떻게 사용될 수 있는지를 보여주기 위해 비교 기하학(comparison geometry)를 활용한 체계적인 이론 분석을 제시합니다.

- **Technical Details**: Fréchet 평균(Fréchet mean)의 존재성, 유일성 및 안정성에 대한 주요 결과가 제공되며, 비모수 회귀(nonparametric regression)에 대한 통계적 보장(statistical guarantees)도 포함됩니다. 여기에는 지수 집중 경계(exponential concentration bounds) 및 수렴 속도(convergence rates)가 포함됩니다. 또한, 각도 안정성(angle stability)에 대한 통찰력은 다양체의 곱셈(curvature)과 비유클리드 상황에서 회귀 추정기(regression estimator)의 동작 간의 상호 작용을 설명합니다.

- **Performance Highlights**: 실험 결과는 이론적 발견을 검증하며, 특히 이분산성(heteroscedasticity)을 가진 데이터에서 제안된 하이퍼볼릭 맵핑(hyperbolic mappings)의 효과성을 보여줍니다. 이러한 결과는 비유클리드 메트릭 공간에서 Fréchet 회귀의 실제적인 유용성을 강조합니다.



### Can LLMs Assist Annotators in Identifying Morality Frames? -- Case Study on Vaccination Debate on Social Media (https://arxiv.org/abs/2502.01991)
Comments:
          Accepted at 17th ACM Web Science Conference 2025 (WebSci'25)

- **What's New**: 최근의 연구는 COVID-19 팬데믹이 공공 건강 위기일 뿐만 아니라 디지털 플랫폼 활용에 있어 중요한 사건이라는 점을 강조합니다. 특히, 백신과 같은 논란의 여지가 있는 주제에서 소셜 미디어는 공공 담론에 중요한 영향을 미치고 있으며, 다양한 도덕적 관점이 개인의 의견에 큰 영향을 미친다는 점이 부각됩니다. 연구팀은 최신의 대형 언어 모델(LLMs)을 이용하여 인간 주석자가 백신 관련 논의에서 도덕적 프레임을 식별하는 데 도움을 줄 수 있는 가능성을 탐구하였습니다.

- **Technical Details**: 본 연구에서는 LLMs를 이용한 두 단계 절차를 통해 도덕적 프레임 분석을 수행합니다. 첫 번째 단계는 개념과 설명을 생성하는 것이고, 두 번째 단계는 '생각 소리 내기(thinking aloud)' 도구를 활용한 인간 평가입니다. LLMs는 적은 수의 예제를 통해 새로운 작업을 수행하는 능력인 few-shot learning을 통해 초기 레이블과 도덕적 프레임에 대한 설명을 생성함으로써 주석자의 인지 부담을 줄이고 일관성과 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: 연구 결과 LLMs를 주석 과정에 통합함으로써 주석의 정확성이 향상되었고, 작업의 난이도는 줄어들며, 인지 부담이 감소하는 것으로 나타났습니다. 수집한 피드백을 통해 참가자들은 LLMs의 설명이 유용하며, 도덕적 프레임 식별 작업에서 그들의 이해를 돕고 인지 부담을 줄이는 데 긍정적인 영향을 미쳤다고 응답했습니다. 이는 복잡한 심리언어학적 작업에서 인간과 AI 간의 협업 가능성을 제시하는 중요한 사례로 평가됩니다.



### Generative Data Mining with Longtail-Guided Diffusion (https://arxiv.org/abs/2502.01980)
Comments:
          20 pages

- **What's New**: 본 연구에서는 기존의 리액티브 접근 방식에서 벗어나, 예측 모델이 배포된 이후에 직면할 수 있는 다양한 문제를 미리 예측하도록 하는 프로액티브한 Longtail 발견 프로세스를 개발했습니다. 특히, 에피스테믹 불확실성(epistemic uncertainty)의 차별화된 단일 전방 패스 형태를 포함한 일반적인 모델 기반 Longtail 신호를 도입하여 드물거나 어려운 입력을 효과적으로 플래그하여 추가 학습 데이터 생성을 유도합니다.

- **Technical Details**: 모델 기반 Longtail 신호는 드물고 어려운 입력을 플래그할 수 있는 차별화 가능한 신호를 개발하며, 이를 통해 기존 예측 모델의 성능에 영향을 미치지 않고 데이터 생성을 위한 가이드를 제공합니다. 연구원들은 또한 Longtail Guidance (LTG)라는 기법을 사용하여 기존 예측 모델의 특정 문제와 Coupling 되는 Latent diffusion 모델을 활용합니다. LTG는 diffusion 모델이나 예측 모델에 대한 재훈련 없이 진행될 수 있습니다.

- **Performance Highlights**: LTG로 생성된 데이터는 의미론적으로 유의미한 변화를 보여주고, 이미지 분류 벤치마크에서 상당한 일반화 개선을 가져옵니다. 또한, LTG로 생성된 데이터를 분석하여 예측 모델이 직면하는 개념적 격차를 사전적으로 발견하고 설명하며 해결하는 데 활용할 수 있습니다. 실험 결과, 기존의 합성 데이터 생성 접근법에 비해 LTG를 통해 생성된 합성 데이터가 월등한 성능 개선을 보이는 것으로 나타났습니다.



### CITER: Collaborative Inference for Efficient Large Language Model Decoding with Token-Level Routing (https://arxiv.org/abs/2502.01976)
- **What's New**: CITER(Collaborative Inference with Token-lEvel Routing) 프레임워크는 작은 언어 모델(SLMs)과 대형 언어 모델(LLMs) 간의 효율적인 협업을 가능하게 합니다. 이 새로운 접근법은 비판적이지 않은 토큰은 SLM에 라우팅하여 효율성을 높이고, 중요한 토큰은 LLM에 라우팅하여 일반화 품질을 유지합니다. 이 프레임워크는 정책 최적화 기법을 통해 라우터를 훈련시키며, 예측 품질과 생성 비용 모두에 따라 보상을 받도록 설계되어 있습니다.

- **Technical Details**: CITER는 토큰 수준의 라우팅을 활용하여 언어 모델의 추론 과정을 가속화합니다. 라우터는 각 토큰의 라우팅 점수를 예측하고, 미리 정의된 임계값에 따라 모델을 선택하여 토큰을 생성하도록 합니다. 이 과정은 강화 학습(reinforcement learning) 문제로 공식화되며, 라우터 훈련을 통해 현재 토큰의 정확성뿐만 아니라 장기적인 의사결정의 영향을 고려합니다.

- **Performance Highlights**: CITER는 다섯 개의 벤치마크 데이터셋을 통해 LLM의 추론 비용을 줄이며 높은 출력 정확도를 유지하는 효율성을 입증하였습니다. 이 방법은 최대 30% 적은 계산 비용으로 유사한 정확도를 달성하거나 동일한 비용으로 25% 더 높은 정확도를 제공합니다. 또한, 토큰 수준의 라우팅은 더 유연한 결과를 제공하여 성능을 극대화합니다.



### Layer Separation: Adjustable Joint Space Width Images Synthesis in Conventional Radiography (https://arxiv.org/abs/2502.01972)
- **What's New**: 이 연구는 류마티스 관절염(Rheumatoid Arthritis, RA) 분석을 위한 새로운 접근법인 레이어 분리 네트워크(Layer Separation Networks, LSN)를 제안합니다. LSN은 손가락 관절의 X선 이미지를 전통적으로 분석하는 과정에서 발생하는 데이터 품질 문제를 해결하기 위해 소프트 티슈, 상부 및 하부 뼈 층을 정확하게 분리하는 데 중점을 둡니다. 이러한 기술을 사용하여 조정 가능한 관절 공간 너비(Joint Space Width, JSW) 이미지를 합성하여 RA 진단 및 연구를 위한 데이터 자원의 다양성을 높이는 것을 목표로 합니다.

- **Technical Details**: LSN은 세 가지 기본 하위 네트워크로 구성되어 있으며, 각각은 레이어 이미지 생성, 세분화 기반 감독, 소프트 티슈 식별 네트워크의 역할을 수행합니다. 이를 통해 X선 이미지에서 뼈와 소프트 티슈의 텍스처 분리를 시도하며, 이 과정은 대칭 기반 생성 네트워크(transUnet)를 활용하여 수행됩니다. 생성된 레이어 이미지는 기존의 다양한 조건을 충족하는 동시에, 픽셀 수준의 분별을 통해 더 높은 정확성을 보장합니다.

- **Performance Highlights**: 실험 결과, LSN 기반의 합성 이미지는 실제 X선 이미지와 매우 유사하였으며, 하류 작업에서의 성능을 크게 향상시켰습니다. 합성된 데이터는 RA 진단의 정확성, 안정성, 그리고 강건성을 향상시키고, 주석이 달린 훈련 데이터에 대한 의존성을 줄이는 데 도움이 됩니다. 이와 함께, 연구에서 제공하는 코드와 데이터셋은 향후 RA 연구 발전에 중요한 자원이 될 것입니다.



### Mitigating Object Hallucinations in Large Vision-Language Models via Attention Calibration (https://arxiv.org/abs/2502.01969)
- **What's New**: 최근의 연구는 Large Vision-Language Models (LVLMs)에서 발생하는 객체 환각(object hallucination) 문제를 해결하기 위한 방안을 제안하였습니다. 기존 방법들이 LVLM의 시각적 토큰과 공간 위치 간의 상관관계에 한정돼 있는 반면, 이 연구에서는 훈련 없이 사용할 수 있는 Uniform Attention Calibration (UAC)과 동적 조정이 가능한 Dynamic Attention Calibration (DAC) 방법을 통해 새로운 솔루션을 제시합니다. 이러한 방법은 다양한 LVLM 아키텍처에서 고도로 효율적인 결과를 가져오며, 객관적인 환각을 감소시키는 데 큰 성과를 보여줍니다.

- **Technical Details**: UAC는 무의미한 입력 이미지에서 편향(bias)을 추정하고 주의(attention) 불균형을 교정하는 보정(matrix) 행렬을 적용하여 학습이 필요 없는 단순한 솔루션을 제공합니다. DAC는 이 개념을 확장하여 자기 주의(self-attention) 메커니즘에 적용할 수 있는 학습 가능한 모듈을 통합합니다. 이 모듈은 대비 학습(contrastive learning)을 통해 이미지 내 객체 위치에 관계없이 일관된 출력을 유도하도록 조정됩니다.

- **Performance Highlights**: 실험 결과, UAC와 DAC는 여러 벤치마크에서 객관적 환각을 유의미하게 줄이며, LLaVA-1.5, mPLUG-Owl2 등 다양한 LVLM 모델에서 우수한 성과를 내고 있습니다. 또한, MME 및 LLaVA-Bench에서 LVLM의 전체 인식 능력을 강화하는 효과도 확인되었습니다. 이 연구는 LVLM의 성능 개선과 더불어 실질적인 응용 가능성을 보여 줍니다.



### Token Cleaning: Fine-Grained Data Selection for LLM Supervised Fine-Tuning (https://arxiv.org/abs/2502.01968)
- **What's New**: 이 논문은 감독된 미세 조정(Supervised Fine-Tuning, SFT) 과정에서 데이터 품질이 양보다 더 중요하다는 최근 연구를 바탕으로 한다. 특히, 데이터 정제 방법들이 전체 샘플을 필터링하는 데 중점을 두는 대신, 개별 토큰의 품질을 평가하고 개선하는 새로운 토큰 정제 파이프라인을 제안한다. 저자들은 비인간 라벨의 관점에서 토큰 품질을 조사하고, 유용한 정보를 보존하면서 불필요한 토큰을 제거하는 방법을 소개한다.

- **Technical Details**: 이 연구에서 저자들은 두 가지 주요 접근 방식을 통해 토큰 품질을 평가한다. 첫 번째는 수정된 모델(Fix-Model Cleaning)로, 고정된 두 개의 모델을 이용해 일괄적으로 SFT 데이터셋을 정제하는 방법이다. 두 번째는 자기 진화 모델(Self-Evolving Cleaning)로, 이 방법에서는 참조 모델이 반복적으로 업데이트되어 주어진 데이터의 각 부분을 청소한다. 이론적 분석을 통해 두 방법의 장점과 한계를 평가하였다.

- **Performance Highlights**: 다양한 다운스트림 작업에 대한 광범위한 실험을 통해, 제안된 토큰 정제 파이프라인이 기존 방법들보다 성능을 일관되게 향상시키는 것으로 나타났다. 특히, 이 방법은 모델이 관련성이 높은 토큰에 집중하도록 하여 다운스트림 결과를 개선하는데 큰 역할을 할 수 있음을 입증하였다.



### DHP: Discrete Hierarchical Planning for Hierarchical Reinforcement Learning Agents (https://arxiv.org/abs/2502.01956)
- **What's New**: 이 논문에서는 Hierarchical Reinforcement Learning (HRL)을 활용하여 긴 기간의 비주얼 플래닝(task) 문제를 해결하는 Discrete Hierarchical Planning (DHP) 방법을 제안합니다. DHP 방법은 기존의 거리 기반(distance-based) 접근법과는 달리 도달 가능성(check) 평가를 통해 계획을 수립합니다. 이 접근법은 훈련 에이전트가 근접한 상태 간의 전환을 통해 지역 상태 공간에 대한 명확한 표현을 개발할 수 있도록 도와줍니다.

- **Technical Details**: DHP 방법은 에이전트가 긴 목표를 기준으로 하위 목표(subgoals)를 반복적으로 예측하도록 하여 계획을 생성합니다. 각 계획은 하위 작업 트리(subtask trees)로 구조화되며, 저위치에서는 더 간단한 하위 작업에 해당합니다. 이 방법은 분산된 도달 가능성 요소로 이루어진 보상 체계를 활용하여 학습 신호를 더욱 명확하게 하고, 단계 수가 최대 훈련 깊이를 초과하여 일반화를 가능케 합니다.

- **Performance Highlights**: 제안된 DHP 방법은 25개의 방이 있는 환경에서 긴 시간의 비주얼 플래닝 작업을 평가한 결과, 이전 기준보다 현저히 우수한 성공률과 짧은 평균 에피소드 길이를 기록하여 우수한 성능을 입증합니다. 또한, ablation 연구를 통해 주요 모듈의 개별 기여도가 전체 성능에 미친 영향을 강조하였습니다. 이 연구 결과는 DHP 방법을 활용한 훈련 에이전트가 더욱 효율적인 계획 수립을 가능하게 함을 보여줍니다.



### LAYOUTDREAMER: Physics-guided Layout for Text-to-3D Compositional Scene Generation (https://arxiv.org/abs/2502.01949)
- **What's New**: 이번 논문에서는 텍스트 기반의 3D 장면 생성 분야에 본격적으로 기여할 수 있는 LayoutDreamer라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 3D Gaussian Splatting(3DGS)을 이용해 고품질이며 물리적으로 일관된 장면 생성을 가능하게 합니다. 특히, 텍스트 프롬프트를 바탕으로 장면 그래프를 형성하고, 이를 통해 3D 개체의 밀도와 레이아웃을 조정합니다. LayoutDreamer는 종전의 방법들이 겪고 있던 여러 한계를 극복할 수 있는 가능성을 보여줍니다.

- **Technical Details**: LayoutDreamer는 세 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째로, 장면 그래프를 통해 상호작용 관계를 명확히 하고, 3D Gaussian을 초기화하는 방법을 제시합니다. 두 번째로, 동적인 카메라 조정 전략을 통해 물체의 포즈와 위치, 밀도를 최적화합니다. 마지막으로, 물리적 제약을 적용하기 위해 물리 및 레이아웃 에너지를 최소화하는 과정을 두 단계로 나누어 구현합니다.

- **Performance Highlights**: LayoutDreamer는 T3Bench의 여러 물체 생성 지표에서 최첨단(SOTA) 성능을 달성하며, 물리적 제약을 준수하는 고충실도의 3D 장면을 생성하는 데 매우 우수한 성능을 보입니다. 이러한 성능은 LayoutDreamer가 고도로 제어 가능한 장면 편집 및 확장 기능을 제공하도록 돕습니다. 포괄적인 실험 결과는 LayoutDreamer가 기존의 방법들에 비해 뛰어난 품질과 의미적 정렬을 제공함을 보여줍니다.



### Boundary-Driven Table-Filling with Cross-Granularity Contrastive Learning for Aspect Sentiment Triplet Extraction (https://arxiv.org/abs/2502.01942)
Comments:
          Accepted to ICASSP 2025

- **What's New**: 본 논문에서는 Aspect Sentiment Triplet Extraction (ASTE) 과제를 해결하기 위해 boundary-driven table-filling과 cross-granularity contrastive learning (BTF-CCL) 방법을 제안합니다. 기존의 2D table-filling 프로세스가 단어 수준의 상호작용에 초점을 맞추는 반면, 제안된 방법은 문장 수준의 표현과 단어 수준의 표현 간의 의미적 일관성을 높이는 데 중점을 둡니다. 이를 통해 모델은 복잡한 문장에서 다중 단어의 aspect와 opinion 용어 간의 관계를 보다 효과적으로 포착할 수 있습니다.

- **Technical Details**: BTF-CCL은 문장 내에서 aspect, opinion 및 sentiment 간의 관계를 학습하기 위해 긍정적 및 부정적 샘플 쌍을 구성합니다. 이 방법은 BERT를 이용하여 입력 문장을 인코딩하고, multi-scale, multi-granularity convolutional method (MMCNN)을 통해 로컬 의미 정보를 더욱 잘 포착합니다. 최종적으로, 모든 후보 영역이 감정 극성을 감지하고 분류됩니다.

- **Performance Highlights**: 실험 결과, BTF-CCL 방법이 기존의 최신 기법들과 비교하여 F1 점수 기준으로 더욱 뛰어난 성능을 보임을 증명하였습니다. 제안된 접근방식은 문장 수준의 맥락 정보를 보다 효과적으로 캡처하면서도 로컬 세부 사항에 대한 민감성을 유지합니다. 이러한 점에서, ASTE 분야의 발전에 기여할 것으로 기대됩니다.



### Can LLMs Maintain Fundamental Abilities under KV Cache Compression? (https://arxiv.org/abs/2502.01941)
Comments:
          21 pages

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 기본 기능에 대한 KV 캐시 압축 방법의 영향을 조사하는데 초점을 맞추고 있습니다. 기존 연구들은 긴 컨텍스트에서 인상적인 압축 비율을 달성했지만, 이러한 방법들이 모델의 핵심 기능에 미치는 영향은 충분히 탐구되지 않았습니다. 이를 통해 均一한 성능 저하가 작업 특이적으로 나타난다는 것을 발견했습니다.

- **Technical Details**: KV 캐시 압축 기술은 LLM 배포의 효율성을 높이기 위해 필수적이며, 모델 사이즈와 컨텍스트 길이가 증가함에 따라 메모리 관리에서의 필요성이 커졌습니다. 이 논문에서는 여러 가지 KV 캐시 압축 방법을 다양한 작업에 대해 검토하였으며, 특히 수치적 추론 작업이 압축에 민감하다는 것을 밝혔습니다. 새로운 방법 ShotKV를 통해 선행 및 디코딩 단계에서의 압축을 관리하면서도 의미적 일관성을 유지할 수 있음을 증명했습니다.

- **Performance Highlights**: ShotKV는 압축 비율이 증가하는 상황에서 긴 컨텍스트 생성 작업에서 성능을 9%에서 18% 향상시키는 결과를 보였습니다. 수치적 추론 및 안전성 관련 작업의 성능 저하가 크다는 것을 보여주었으며, 다단계 추론 모델이 지시 튜닝 모델보다 압축에 대한 내구성이 더 우수함을 나타냈습니다. 전반적으로, 다양한 작업에 대한 종합적인 평가를 통해 기존 압축 방법들의 한계를 파악하고 새로운 접근법인 ShotKV가 효과적임을 입증하였습니다.



### VolleyBots: A Testbed for Multi-Drone Volleyball Game Combining Motion Control and Strategic Play (https://arxiv.org/abs/2502.01932)
- **What's New**: 새로운 MARL (Multi-agent Reinforcement Learning) 테스트베드인 VolleyBots를 소개합니다. 이 테스트베드는 물리적 역학을 고려하여 여러 드론이 배구 경기를 통해 협력하고 경쟁할 수 있게 설계되었습니다. VolleyBots는 배구 규칙에 기반한 턴 기반(interaction model) 상호작용 모델과 계층적 의사 결정 프로세스를 특징으로 하며, 이는 운동 제어(motion control)와 전략적 플레이를 결합합니다.

- **Technical Details**: 이 테스트베드는 단일 드론 훈련부터 여러 드론의 협동 및 경쟁 작업에 이르기까지 다양한 작업(Task) 세트를 제공합니다. 또한, 높은 정확도의 시뮬레이션 고충실도(high-fidelity simulation)로 시뮬레이션에서 실제 상황으로의 원활한 전이를 지원합니다. 연구진은 대표적인 MARL과 게임 이론(game-theoretic) 알고리즘의 기준 평가(baseline evaluations)를 포함하여, 복잡한 작업에서 기존 알고리즘의 한계를 보여주었습니다.

- **Performance Highlights**: 시뮬레이션 결과, 기존 알고리즘은 단순한 작업에서는 효과적으로 작동하지만, 저수준 제어(low-level control)와 고수준 전략(high-level strategy)이 모두 필요한 복잡한 작업에서는 어려움을 겪는 것으로 나타났습니다. 연구진은 시뮬레이션에서 학습된 정책(policy)을 실제 드론에 제로샷(zero-shot) 배포하는 과정을 통해 VolleyBots의 잠재력을 강조하였습니다. 이는 민첩한 로봇 플랫폼을 포함한 MARL 연구의 발전을 촉진할 것으로 기대됩니다.



### Distributionally Robust Direct Preference Optimization (https://arxiv.org/abs/2502.01930)
- **What's New**: 본 논문은 대형 언어 모델(LLM)의 인간 선호도 정렬 문제에 대한 새로운 접근법을 제시합니다. 구체적으로, 인간의 선호도가 다양한 배경에 따라 어떻게 변화하는지를 고려하여, 기존의 정적(preference) 데이터셋을 사용하는 접근 방식의 한계를 해결하기 위해 새로운 분포적으로 강건한 직접 선호 최적화(Direct Preference Optimization) 알고리즘을 개발하였습니다. 이 알고리즘을 통해 선호도 분포의 변화로 인한 문제를 극복할 수 있는 가능성을 제시합니다.

- **Technical Details**: 연구자들은 Wasserstein DPO (WDPO) 및 Kullback-Leibler DPO (KLDPO)라는 두 가지 분포적으로 강건한 DPO 알고리즘을 개발하였습니다. 이 논문에서는 WDPO 및 KLDPO의 최적 정책 매개변수 샘플 복잡성을 특성화하고, 최소-최대 손실 함수(minimax loss functions)를 해결하기 위해 적절한 근사치 평가는 기울기 하강법(gradient descent) 학습 알고리즘을 제안합니다. 이 기술적 접근은 기존의 LLM 정렬 파이프라인에 쉽게 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과는 WDPO 및 KLDPO가 선호도 분포 변화가 발생하는 현실적인 상황에서 LLM 정렬 성능을 크게 향상시킬 수 있음을 보여줍니다. 특히, 두 알고리즘이 기존의 방법들에 비해 높은 성능을 발휘하여, 실세계 어플리케이션에서의 정렬 실패 가능성을 줄이는 데 기여할 것으로 기대됩니다. 논문에서는 실험을 통해 이러한 성과를 정량적으로 입증하고 있으며, 관련 이론적 보장(thoretical guarantees)을 제시하여 알고리즘의 신뢰성을 높였습니다.



### LAST SToP For Modeling Asynchronous Time Series (https://arxiv.org/abs/2502.01922)
- **What's New**: 본 논문에서는 Asynchronous Time Series에 맞춘 새로운 프롬프트 디자인을 제안합니다. 기존의 정규 시계열 데이터와 달리 비동기 시계열 데이터는 불규칙한 간격으로 발생하는 타임스탬프 이벤트로 구성됩니다. 이러한 접근 방식은 이벤트 설명의 풍부한 자연어를 효과적으로 활용하여 LLMs가 다양한 도메인에서 추론할 수 있게 합니다. 이로 인해 통계적 예측을 넘어 이상 탐지 및 데이터 보간 등의 작업으로 비동기 시계열 분석의 범위를 확장하게 됩니다.

- **Technical Details**: 본 연구에서 제안하는 LASTS(여기서는 Language-modeled Asynchronous Time Series)가 LLMs를 비동기 시계열 데이터에 맞춰 효율적으로 적용할 수 있게 합니다. 이 프레임워크는 이벤트 유형을 미리 정의된 카테고리로 그룹화할 필요가 없으며, 자연어 설명을 그대로 활용할 수 있습니다. 또한 새로운 방법인 Stochastic Soft Prompting(StoP)을 도입하여 프롬프트 조정 메커니즘을 통해 모델 성능을 효과적으로 향상시킵니다. 이를 통해 특정 이벤트 간의 상호작용을 잘 반영할 수 있습니다.

- **Performance Highlights**: 실제 데이터셋을 통한 광범위한 실험을 통해 제안된 방법이 다양한 작업 및 데이터셋에서 최첨단 성능을 달성함을 입증합니다. 특히, 기존의 QLoRA와 같은 미세 조정 방법보다 우수한 성능을 보이며, 비동기 시계열 데이터 처리 및 분석의 잠재력을 강조합니다. 이를 통해 LLM 기반 모델이 비동기 시계열 데이터의 해석과 분석에 효과적으로 적용 가능하다는 점이 확인됩니다.



### Wake-Informed 3D Path Planning for Autonomous Underwater Vehicles Using A* and Neural Network Approximations (https://arxiv.org/abs/2502.01918)
Comments:
          11 pages, 6 figures, preprint of journal paper

- **What's New**: 이 논문은 복잡한 수중 환경에서의 자율 수중 차량(AUV) 경로 계획 문제를 해결하기 위해 새로운 방법론을 제안합니다. 기존 경로 계획 알고리즘이 유체의 와류(wake) 구조를 통합하지 못했던 문제를 극복하기 위해, 와류 정보를 반영한 3D 경로 계획 방법을 개발했습니다. 이 방법은 A* 알고리즘의 변형인 현재 인식(current-informed) 및 와류 인식(wake-informed) 플래너 두 가지를 포함하여, 실시간 응답을 위한 신경망 모델도 훈련시켰습니다.

- **Technical Details**: 이 연구에서는 AUV의 복잡한 수중 작업 중 에너지 효율성과 안전성을 증대시키기 위해, 3D 하이드로다이나믹 데이터(hydrodynamic data)를 직접적으로 계획 알고리즘에 통합하는 방법론을 개발합니다. A* 알고리즘을 바탕으로 한 두 가지 변형 플래너가 제안되었으며, 이를 통해 실제 경로 계획에서 발생할 수 있는 와류 효과를 명확히 설명했습니다. 또한, 신경망 모델을 통해 실시간으로 경로를 예측하고 최적화하는 방법도 탐구하였습니다.

- **Performance Highlights**: 제안된 와류 인식 A* 플래너는 에너지 소비량을 11.3%까지 줄이며, 고속 및 난류 지역과의 접촉을 최소화하는 성과를 보였습니다. 신경망 모델은 계산 속도를 6배 향상시키는 성능을 지녔지만, 에너지 소비는 4.51%에서 19.79% 증가하고 최적 경로 길이는 9.81%에서 24.38% 감소하는 결과를 보였습니다. 이는 전통적 경로 계획 알고리즘에 와류 구조를 통합하는 것의 중요성을 강조하며, AUV의 작전 효율성 및 안전성을 향상시킬 수 있는 가능성을 보여줍니다.



### PATCH: a deep learning method to assess heterogeneity of artistic practice in historical paintings (https://arxiv.org/abs/2502.01912)
Comments:
          main text: 16 pages, 6 figures; SI: 7 pages, 3 figures

- **What's New**: 이번 연구에서는 기계학습(machine learning) 방법을 활용하여 예술 작품의 창작 과정을 분석하는 새로운 접근 방식을 제안합니다. 연구의 주요 초점은 역사적 작품에 대해 외부 훈련 데이터가 없이도 예술가의 개별적인 작업 양식을 식별하는 것입니다. 이를 통해 과거의 예술적 다양성에 대한 이해를 높이는 방향으로 나아갈 수 있습니다.

- **Technical Details**: 우리는 "pairwise assignment training for classifying heterogeneity (PATCH)"라는 새로운 기계학습 방법을 개발하였습니다. 이 방법은 교차 학습 데이터 없이도 예술적 관행의 이질성을 식별하는 능력을 가지고 있습니다. 결과적으로 이 방법은 기존의 간단한 통계 기법 및 비지도 기계학습 방법보다 우수한 성능을 보였습니다.

- **Performance Highlights**: PATCH 방법을 스페인 르네상스 마스터 El Greco의 두 작품, 'Christ의 세례'와 '풍경이 있는 십자가의 그리스도'에 적용한 결과를 통해, 이전 연구에서 해당 작품을 작업장 구성원에게 할당했던 결론에 이의를 제기하는 새로운 증거를 발견하였습니다. 또한, 이 연구에서의 분석 결과는 시간과 공간을 넘는 예술 작품의 이질성을 특징짓는 척도를 제공합니다.



### Displacement-Sparse Neural Optimal Transpor (https://arxiv.org/abs/2502.01889)
Comments:
          18 pages, 6 Figures

- **What's New**: 이 논문에서는 Optimal Transport (OT) 이론을 바탕으로 한 sparsity penalty를 도입하여, 매핑의 해석 가능성을 향상시키고자 하였습니다. 기존의 Sinkhorn 알고리즘에 비해 Neural OT 솔버를 사용하여 대규모 데이터 문제를 효율적으로 해결하고, 저차원 및 고차원 설정에서 sparsity와 가능성 간의 균형을 조절하는 동적 프레임워크를 제안합니다. 특히, 저차원에서는 sparsity 강도를 조정할 수 있는 새로운 프레임워크를 개발하였고, 고차원에서는 변위 벡터의 차원을 직접 제어합니다.

- **Technical Details**: 제안된 방법은 Neural OT 솔버에서 sparsity penalty를 손실 함수에 통합하여 최적화 목표를 정의합니다. 저차원에서는 ℓ1-norm 및 τ stvs 기준의 두 가지 sparsity penalty를 사용하고, 고차원에서는 새로운 smoothed ℓ0-norm을 도입하여 변위 벡터의 차원을 제한합니다. 각 단계에서는 ICNN 훈련을 통해 매핑의 feasibility와 sparsity 수준 간의 균형을 제공합니다.

- **Performance Highlights**: 제안된 방법은 합성된 sc-RNA 데이터 및 실제 4i 세포 교란 데이터셋에서 기존 방법들보다 더 나은 성능을 보여줍니다. 이러한 결과는 Dynamic Sparsity Intensity 조정이 효과적으로 수행되었음을 입증하며, 이는 OT 문제들을 해결하는 데 있어서 향상된 해석 가능성과 정확성을 가능하게 합니다.



### A Privacy-Preserving Domain Adversarial Federated learning for multi-site brain functional connectivity analysis (https://arxiv.org/abs/2502.01885)
Comments:
          34pages, 13 figures

- **What's New**: 이번 연구에서는 비독립적이고 동일 분포가 아닌 (non-IID) fMRI 데이터 분석을 위한 도메인 적대적 연합 학습(Domain Adversarial Federated Learning, DAFed)이라는 새로운 프레임워크를 제안합니다. DAFed는 다양한 데이터 출처에 대한 개인정보 보호 규제를 준수하면서도 협업 분석의 장점을 극대화할 수 있게 합니다.

- **Technical Details**: DAFed는 라벨이 있는 데이터셋과 라벨이 없는 데이터셋 간의 효과적인 지식 전이를 촉진하기 위해 적대적 훈련(adversarial training)을 활용합니다. 또한, 컨트라스티브 학습 모듈은 도메인 불변 특성의 글로벌 표현을 강화합니다. 이를 통해 로컬 데이터의 특수성을 유지하면서 회복력 있는 글로벌 학습을 보장합니다.

- **Performance Highlights**: DAFed는 자폐 스펙트럼 장애(ASD) 진단에 활용되어, 최신 기법들과 비교했을 때 우수한 분류 정확도를 나타냈습니다. 추가적으로, Score-CAM 모듈은 ASD 및 경도 인지 장애(MCI)에 중요한 뇌 영역과 기능적 연결성을 식별하여, 사이트 간 공유되는 신경생물학적 패턴을 밝혀냈습니다.



### Online Curvature-Aware Replay: Leveraging $\mathbf{2^{nd}}$ Order Information for Online Continual Learning (https://arxiv.org/abs/2502.01866)
- **What's New**: 이 논문에서는 Online Continual Learning (OCL) 모델을 위한 새로운 방법인 Online Curvature-Aware Replay (OCAR)를 제안합니다. OCAR는 재생(replay) 기반 OCL을 형식화하여 KL-divergence 제약을 추가함으로써 안정성과 가변성을 동시에 최대화하려고 합니다. 이 모델은 Fisher Information Matrix (FIM)의 제2차 정보를 활용하여 최적화를 가속화하고 망각을 방지하는 역할을 합니다.

- **Technical Details**: OCAR는 OCL 환경에서 KL-divergence 제약을 적용한 과거 및 새로운 데이터에 대한 공동 최적화(joint optimization)를 수행합니다. 특히 K-FAC 근사(Kronecker-factored Approximate Curvature)를 사용하여 FIM을 효율적으로 근사하고 이를 통해 모델 분포 공간 내에서 안정성과 가변성을 동시에 고려합니다. Tikhonov 정규화와 학습률 사이의 비율을 분석하여 안정성과 가변성을 조절하는 방법도 제시됩니다.

- **Performance Highlights**: 실험 결과, OCAR는 세 가지 다양한 벤치마크에서 최신 기술(state-of-the-art) 방법보다 더 높은 평균 정확도를 기록하며 우수한 성능을 보여주었습니다. OCAR는 기존의 방식보다 안정성과 가변성을 동시에 개선하여 OCL 설정에서의 최적화를 효과적으로 수행할 수 있음을 입증합니다.



### Learning Human Perception Dynamics for Informative Robot Communication (https://arxiv.org/abs/2502.01857)
- **What's New**: 이번 논문에서는 CoNav-Maze라는 시뮬레이션 환경을 도입하여 인간과 로봇 간의 협력 내비게이션 문제를 다루고 있습니다. 로봇은 부분적인 지도를 기반으로 한 인간의 지도 아래에서 환경을 탐색하며, 로봇이 자신의 카메라 영상을 공유함으로써 인간의 환경 이해를 돕습니다. 또한, 정보 획득 몬테카를로 트리 검색(IG-MCTS) 알고리즘을 제안하여 자율적 이동과 정보 교환의 균형을 맞추어 효율적인 협력을 구현했습니다.

- **Technical Details**: IG-MCTS는 복잡한 환경에서 로봇과 인간 간의 시너지를 높이기 위해 개발된 온라인 계획 알고리즘입니다. 이 알고리즘은 인간이 로봇의 행동에 따라 환경을 어떻게 인지하는지를 예측하는 신경망 모델에 기초하고 있습니다. IG-MCTS는 로봇의 카메라 각도를 조정하여 인간의 환경 인식을 극대화하고, 인간 유도 경로를 보상으로 활용하여 더욱 효과적인 협력을 추구합니다.

- **Performance Highlights**: 사용자 연구에서 IG-MCTS는 원격 조작 및 명령 따르기 방법론에 비해 통신 요구를 크게 줄이면서도 동등한 작업 성과를 달성했습니다. 연구 참가자들의 눈 추적 방식으로 얻은 결과는 IG-MCTS 로봇과 상호작용할 때 인지 부하가 낮아지는 것을 나타냅니다. 이러한 결과는 IG-MCTS가 로봇의 자율성뿐만 아니라 인간과의 효과적인 협력을 높이는 데 기여함을 보여줍니다.



### Sample, Scrutinize and Scale: Effective Inference-Time Search by Scaling Verification (https://arxiv.org/abs/2502.01839)
- **What's New**: 이 논문에서는 샘플링 기반(search based on sampling) 검색의 확장성(trend)을 분석합니다. 특히, 단순한 무작위 샘플링과 직접적인 자기 검증(direct self-verification)만을 사용하는 최소한 구현(minimalist implementation)을 통해 수행 능력을 크게 향상시킬 수 있음을 발견했습니다. 'Gemini v1.5 Pro' 모델은 인기 있는 벤치마크에서 'o1-Preview'를 초월하는 추론 능력을 발휘했습니다.

- **Technical Details**: 샘플링 기반 검색의 확장성은 더 넓은 응답 풀(response pool)을 샘플링함으로써 검증 정확성(verification accuracy)이 향상되는 '암묵적 확장(implicit scaling)' 현상에 기인합니다. 논문에서는 테스트 시간 동안 계산(compute)의 자기 검증 능력을 개선하기 위한 두 가지 유용한 원칙(principles)을 소개합니다: (1) 여러 응답을 비교(comparing across responses)하면 오류와 환각의 위치에 대한 유용한 신호(signals)를 제공하고, (2) 서로 다른 모델의 출력 스타일(output styles)이 다른 맥락에 유용하다는 것입니다.

- **Performance Highlights**: 정확한 검증(verification)을 이끌어낼 수 있지만, 최첨단 모델(frontier models)은 기본적인 검증 능력이 뛰어나지 않음을 발견했습니다. 이를 통해 이러한 결함(deficiencies)를 측정하는 새로운 벤치마크(benchmark)를 도입했습니다. 따라서 이 연구는 샘플링 기반 검색과 그로 인한 성능 개선에 대한 이해를 깊게 하는 데 기여할 것입니다.



### TESS: A Scalable Temporally and Spatially Local Learning Rule for Spiking Neural Networks (https://arxiv.org/abs/2502.01837)
Comments:
          9 pages, 2 figures

- **What's New**: TESS는 SNN(Spiking Neural Networks) 훈련을 위한 새로운 스케일링 가능한 학습 규칙으로, 자원 제한이 있는 장치에서 효과적이고 송전 에너지 비용을 낮출 수 있도록 설계되었습니다. 이 방법론은 eligibility traces(자격 추적), spike-time dependent plasticity(STDP), 그리고 neural activity synchronization(신경 활동 동기화)와 같은 생물학적으로 영감을 받은 메커니즘을 통합하여, 각 뉴런의 로컬 신호만을 사용해 훈련할 수 있습니다. TESS는 시간 및 공간적인 신뢰 할당 문제를 모두 해결하여, 에지 장치에서의 효율적인 학습을 가능하게 합니다.

- **Technical Details**: TESS는 O(Ln) 메모리 복잡성과 O(LCn) 계산 복잡성을 유지하면서, 각 뉴런 내에서만 사용 가능한 로컬 신호에 의존하여 학습을 진행합니다. 이는 기존의 BP(backpropagation) 방식에 비해 메모리 사용량을 3~10배 낮추고, 연산량을 205~661배 줄일 수 있게 합니다. 이 방법은 또한 입력 시퀀스의 길이에 관계없이 선형적으로 확장 가능하며, 더 깊은 아키텍처나 대규모 데이터셋에서도 활용될 수 있습니다.

- **Performance Highlights**: TESS는 IBM DVS Gesture 데이터셋, CIFAR10-DVS 및 CIFAR100에서 BPTT(backpropagation through time)와 유사한 성능을 보여주며 정확도에서 평균 1.4%의 차이를 보입니다. 이러한 성능 개선에도 불구하고 메모리와 계산 요구 사항이 현저히 낮아져, 에지 장치에서의 빠르고 효율적인 학습이 가능합니다. TESS는 생물학적으로 신뢰할 수 있는 메커니즘을 사용함으로써, 로컬 학습 방식이 실제로 효과적이라는 것을 증명했습니다.



### Assessing Data Augmentation-Induced Bias in Training and Testing of Machine Learning Models (https://arxiv.org/abs/2502.01825)
Comments:
          4 pages

- **What's New**: 이 논문은 소프트웨어 공학에서 데이터 증강(data augmentation)이 훈련 데이터의 편향(bias)을 어떻게 영향을 미치는지 탐구합니다. 기존의 데이터 증강 기법들이 훈련 데이터와 테스트 데이터 모두에 사용될 때 성능 지표가 불합리하게 증가할 수 있음을 강조합니다. 특히, flaky test classification(불규칙 테스트 분류)에 대한 사례 연구를 통해 데이터 증강의 잠재적 편향을 평가하는 방법론을 제시합니다.

- **Technical Details**: 논문에서는 Synthetic Minority Oversampling Technique(SMOTE)와 같은 기술을 활용하여 훈련 및 테스트 세트의 데이터 증강 효과를 체계적으로 분석합니다. 실험에는 FlakyCat 데이터 세트를 이용하여 flaky 테스트의 다양한 카테고리에 대한 성능을 평가하였으며, 두 개의 실험적 프로토콜(Phase A와 Phase B)을 설정하여 각각의 성능 차이를 측정합니다. Phase A에서는 원본 테스트 케이스에 대한 기준 성능을 설정하고, Phase B에서는 원본 테스트 케이스와 그에 대한 증강 버전을 모두 포함시켜 모델의 일반화 능력을 검증합니다.

- **Performance Highlights**: 실험 결과는 데이터 증강이 원본 데이터로 훈련된 모델보다 전반적으로 12% 향상된 F1 점수를 기록하도록 도와주었음을 보여줍니다. 특히, 복잡한 패턴을 가진 카테고리인 Async Wait 및 Time에서는 각각 0.29에서 0.58, 0.30에서 0.40으로 유의미한 향상을 보였습니다. 추가로, 모델은 새로운 테스트 케이스에 대한 일반화 능력을 평가함으로써, 데이터 증강이 모델의 성능에 긍정적인 영향을 미친다는 점을 시사합니다.



### Agentic Bug Reproduction for Effective Automated Program Repair at Goog (https://arxiv.org/abs/2502.01821)
- **What's New**: 이 논문은 구글의 내부 이슈 추적 시스템을 기반으로 자동 버그 재현 테스트(Bug Reproduction Tests, BRT) 생성의 도전 과제를 탐구하고 있습니다. 연구팀은 최신 기술로 평가된 BRT 생성 기법인 LIBRO를 조정하고, LLM(대형 언어 모델)을 활용한 BRT 에이전트(BRT Agent)를 제시하여 기존 기법보다 성능을 크게 개선하였습니다. BRT Agent는 28%의 타당한 BRT 생성률을 달성하여 LIBRO의 10%에 비해 월등한 성과를 보였습니다.

- **Technical Details**: BRT는 버그가 있는 코드베이스에서 실행하면 실패하고, 수정된 코드베이스에서 실행할 경우 통과하는 테스트로 정의됩니다. 이 논문에서는 구글의 대규모 독점 코드베이스와 산업 문제를 다루며, 에이전트 기반 접근 방식을 통해 BRT 생성을 위한 비약적인 개선을 목표로 합니다. 특히, BRT Agent는 구글의 코드베이스에 최적화된 LLM을 사용하여 고품질의 테스트 코드를 생성합니다.

- **Performance Highlights**: 연구 결과 BRT Agent가 생성한 BRT를 이용한 경우, Passerine이라는 자동 프로그램 수정 시스템이 30% 더 많은 버그를 수정할 수 있음을 보여주었습니다. 이 시스템은 생성된 BRT를 통해 가장 유망한 수정을 선택하는 새로운 지표인 Ensemble Pass Rate (EPR)를 도입하였으며, 이는 제안된 수정이 유망한 경우의 비율을 약 70%로 도달하는 성과를 거두었습니다. 이로 인해 BRT 생성의 실질적인 가치와 효과가 입증되었습니다.



### Score as Action: Fine-Tuning Diffusion Generative Models by Continuous-time Reinforcement Learning (https://arxiv.org/abs/2502.01819)
Comments:
          arXiv admin note: text overlap with arXiv:2409.08400

- **What's New**: 이 논문은 인간 피드백을 이용한 강화 학습(Reinforcement Learning from Human Feedback, RLHF) 기술을 통해 디퓨전 모델을 연속 시간에서 다룰 수 있는 새로운 기법을 소개합니다. 기존의 이산 시간 방식이 가진 오류의 취약성을 극복하고, 더 복잡한 제어 문제에 적용할 수 있는 방법론을 제시합니다. 이러한 연속 시간 RL 접근 방식은 스코어 매칭을 통해 정책 최적화 및 정규화와 연결됩니다.

- **Technical Details**: 연속 시간 RL 프레임워크를 통해 디퓨전 모델의 파라미터를 조정하는 방법을 제안합니다. 여기서는 스코어 기능을 행동으로 여기는 접근 법을 취하며, 이러한 접근이 이산 시간 RL 알고리즘의 한계를 극복할 수 있음을 보여줍니다. 더불어 강화 학습의 정책 최적화 알고리즘을 발전시키고, 상태 독립적인 디퓨전 계수를 고려하여 높은 축약성과 명시적인 이점-비율 함수를 제공합니다.

- **Performance Highlights**: 실험 결과에 따르면 이 새로운 기법은 대규모 Text2Image 모델인 Stable Diffusion v1.5의 미세 조정 작업에서 유의미한 성능 향상을 보여주었습니다. 함께 제시된 실험들은 naive value network 디자인보다 더 나은 성능을 입증하며, 새로운 가치 네트워크 디자인 접근법의 가능성을 확인시키고 있습니다. 이러한 개선을 통해 연속 시간 RL이 목표 달성에 효과적인 도구임을 강조합니다.



### Toward Neurosymbolic Program Comprehension (https://arxiv.org/abs/2502.01806)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)를 기반으로 한 Large Code Models (LCMs)의 발전을 살펴보고 있습니다. 이러한 모델들은 코드 생성, 소프트웨어 테스트 및 프로그램 이해와 같은 복잡한 소프트웨어 엔지니어링 작업의 자동화를 지원합니다. 그러나 GPT-4와 같은 트릴리온 파라미터 모델을 확장하려는 시도가 교육 및 배포에 대한 높은 계산 자원 요구와 신뢰성 및 편향 문제를 초래하여 많은 조직에서의 활용이 어려움을 제기합니다.

- **Technical Details**: 저자들은 증가하는 모델 파라미터 수가 항상 최적의 경로라는 기존의 가정에 의문을 제기하며, 기존의 Deep Learning (DL) 기술과 전통적인 증거 기반 방법의 강점을 조화롭게 결합하는 Neurosymbolic 연구 방향을 제안합니다. 이 접근 방식은 기계 학습의 신뢰성, 속도 및 결정성을 활용하여, 최초의 Neurosymbolic Program Comprehension (NsPC) 프레임워크를 개발하고 결함이 있는 코드 요소를 식별할 수 있도록 설계되었습니다. 이러한 프레임워크는 LCM의 여러 한계를 극복하고 AI 기반 시스템의 투명성과 책임성을 향상시키는 데 기여할 것입니다.

- **Performance Highlights**: 제안된 NsPC 프레임워크의 초기 결과는 결함 추적 및 코드 인식의 정확성을 향상시키는 데 있어 긍정적인 신호를 보여줍니다. 이 접근 방법을 통해 개발자들이 복잡한 코드 기반 내에서 문제를 보다 효과적으로 식별하고 해결할 수 있도록 지원할 전망입니다. LCM과 같은 대형 AI 모델들이 직면한 문제를 해결하기 위한 새로운 길을 제시하며, 기존의 소프트웨어 엔지니어링 작업의 효율을 높이는 데 기여할 것으로 기대됩니다.



### Discovering Chunks in Neural Embeddings for Interpretability (https://arxiv.org/abs/2502.01803)
- **What's New**: 본 연구는 인간 인지에서 정보를 '청크(chunk)'로 나누어 처리하는 원칙을 인공 신경망 해석에 적용하는 새로운 접근 방식을 제안합니다. 훈련된 신경망의 숨겨진 상태가 학습한 데이터의 규칙성을 반영한다는 가정을 세우고, 이는 인지 메커니즘으로부터 통찰력을 제공할 수 있는 가능성을 보여줍니다. 저자들은 간단한 RNN과 대형 언어 모델에서 이 원리를 테스트하며, 네트워크 응답에 영향을 미치는 반복적인 청크를 추출하는 방법을 모색합니다.

- **Technical Details**: 연구에서는 RNN(재귀 신경망) 및 LLMs(대형 언어 모델)에서 학습한 패턴이 어떻게 숨겨진 상태로 반영되는지를 분석합니다. 이들은 의미 있는 개체들을 식별하기 위해 저자들이 조사한 다양한 차원의 신경 집단 활동에서 청크를 추출하는 방법을 포함합니다. 기존의 신경망 해석 기법들이 개별 뉴런에 제한적이었다면, 본 연구는 보다 폭넓은 집단 활동을 고려하는 접근 방식을 제안합니다.

- **Performance Highlights**: 저자들은 제안한 청크 추출 방법론이 신경망의 반응 및 처리 능력을 개선할 수 있음을 강조합니다. 그들은 또한 이 연구가 신경망이 데이터를 처리하는 방식을 해석하는 새로운 틀을 제시할 수 있음을 보여줍니다. 다양한 데이터 소스에서 훈련된 신경망들이 유사한 표현을 학습하는 경향을 발견하여, 인공 지능의 다양한 모델 간에 공통된 패턴이 존재함을 입증했습니다.



### Flow-based Domain Randomization for Learning and Sequencing Robotic Skills (https://arxiv.org/abs/2502.01800)
- **What's New**: 이 논문은 GoFlow라는 새로운 접근 방식을 소개하며, 이는 액터-크리틱 강화 학습 아키텍처(actor-critic reinforcement learning)와 신경 샘플링 분포(neural sampling distribution)를 결합하여 강력한 정책을 학습합니다. 기존의 간단한 매개변수화된 샘플링 분포보다 더 유연하고 강력한 성능을 제공합니다. 또한 접촉이 많은 조립 작업에서의 실제 사용 사례를 통해 이 방법이 실제 환경에서도 효과적임을 보여줍니다.

- **Technical Details**: GoFlow는 환경의 다양한 매개변수를 샘플링할 때 최대의 다양성을 극대화하여 현재 정책에 도전적이면서도 해결 가능한 환경을 능동적으로 발견합니다. 이 논문에서는 정상화 흐름(normalizing flow) 기법을 통해 샘플링 분포를 배우는 방법을 제안하며, 이를 통해 기계 학습과 샘플링의 유연성을 높입니다. 마지막으로, 학습된 샘플링 분포는 불확실성 기반 다단계 계획 시스템에서의 분포 밖 감지(out-of-distribution detection)에 활용됩니다.

- **Performance Highlights**: GoFlow는 시뮬레이션 환경에서 고정된 도메인 무작위화(domain randomization) 방법보다 우수한 성능을 보여줍니다. Gear insertion과 같은 실제 접촉 중심 조작 작업에서 GoFlow의 효과를 입증하며, 불확실성 하에서의 다단계 의사 결정 능력까지 확장합니다. 비확률적 포즈 추정 모델을 통합하여 로봇이 필요할 때 추가 정보를 능동적으로 수집할 수 있게 되어, 부분 관측 상태에서도 성능을 향상시킵니다.



### AquaticCLIP: A Vision-Language Foundation Model for Underwater Scene Analysis (https://arxiv.org/abs/2502.01785)
- **What's New**: 본 논문에서는 수생 장면 이해를 위한 새로운 모델인 AquaticCLIP을 소개합니다. AquaticCLIP은 이미지와 텍스트를 정렬하여 세그멘테이션(segment), 분류(classification), 탐지(detection), 객체 counting와 같은 다양한 작업을 수행할 수 있도록 하는 비지도 학습 프레임워크를 제공합니다. 수중 이미지를 위한 대규모 데이터셋을 사용하여 사전 훈련(pre-training)을 통해 기존의 비전-언어 모델들은 수중 환경에서 더 높은 성능을 발휘합니다.

- **Technical Details**: AquaticCLIP은 200만개의 수중 이미지-텍스트 쌍으로 구성된 데이터셋을 활용하여 훈련됩니다. 모델은 프로프트(learnable prompts)를 통해 패치(patch) 특징을 점진적으로 집계하는 비전 인코더(vision encoder)를 사용하며, 시각적 맥락을 통합하는 비전 가이드 언어 인코더로 언어 인코더(language encoder)를 강화합니다. 이들은 대조적 사전 훈련 손실을 통해 시각 및 텍스트 모달리티를 정렬합니다.

- **Performance Highlights**: AquaticCLIP은 다양한 수중 컴퓨터 비전 작업에서 제로샷(zero-shot) 환경에서도 현저한 성능 향상을 달성하였습니다. 모델은 기존의 방법들보다 더 뛰어난 강건성(robustness) 및 해석 가능성(interpretability)을 보여주며, 수중 환경에서 비전-언어 응용 프로그램을 위한 새로운 기준을 제시합니다. 이러한 성과는 AquaticCLIP이 수생 생물 다양성 보존에 기여할 수 있는 가능성을 보여줍니다.



### Grokking Explained: A Statistical Phenomenon (https://arxiv.org/abs/2502.01774)
- **What's New**: 이 연구에서는 'grokking'이라는 현상을 체계적으로 분석하고, 훈련 데이터와 테스트 데이터 간의 분포 변화(distribution shift)가 grokking의 주요 원인임을 강조합니다. 두 개의 합성 데이터셋(synthetic dataset)을 이용하여 grokking을 세밀하게 재현하며, 기존의 연구가 제시한 고정된 요인에서 벗어난 새로운 관점을 제시합니다.

- **Technical Details**: 첫 번째 합성 데이터셋은 클래스를 서브클래스로 나누어 구성되며, 이는 적은 하이퍼파라미터 조정으로 grokking을 유도할 수 있음을 보여줍니다. 두 번째 데이터셋에서는 클래스가 동형사상(equivariant map)을 형성하며, 이는 서브클래스 간의 관계가 grokking의 강도에 미치는 영향을 분석합니다.

- **Performance Highlights**: MNIST 데이터셋을 사용한 실험을 통해, 실제 환경에서 유도된 분포 변화가 grokking에 미치는 영향을 검증하였습니다. 이 연구는 데이터가 클래스로 조직된 경우, 적절한 샘플링을 통해 모델이 미지의 데이터에 일반화(generalization)할 가능성을 높일 수 있음을 보여줍니다.



### On Bob Dylan: A Computational Perspectiv (https://arxiv.org/abs/2502.01772)
- **What's New**: 본 연구는 Cass Sunstein의 'On Bob Dylan' 에세이를 확장하여 Bob Dylan의 가사를 1962년부터 2012년까지 대규모 computational analysis (컴퓨테이셔널 분석)를 통해 분석합니다. 이 연구는 Dylan의 가사에서 개념 대 개념 관계를 추출하고, 이를 기반으로 방향성 지식 그래프를 구축하여 그의 주제적 구조를 포착합니다. 결과적으로, Dylan의 가사는 메타포에 대한 의존도가 증가하고, 감정 프로파일이 진화하며, 'dishabituation' (탈습관화)이 높아지는 것으로 나타났습니다.

- **Technical Details**: 연구자는 Bob Dylan의 1962년부터 2012년까지의 스튜디오 앨범 가사를 수집했습니다. o3-mini-high라는 대형 언어 모델을 활용하여 가사를 분석하고, 관련된 개념 쌍을 추출하여 각 개념 간의 관계를 구조화된 JSON 형식으로 저장했습니다. 이 과정에서 연구자는 노드(개념)를 정규화하고 관계를 확인하기 위해 다양한 네트워크 측정을 계산했습니다.

- **Performance Highlights**: 분석 결과, Dylan의 가사는 시대에 따라 테마가 다양하게 변하고, 이는 그의 음악적 변화에 대한 깊은 통찰을 제공합니다. 그는 특히 'movement', 'protest', 'mythic imagery'와 같은 주제를 다루며 그의 경력에 걸쳐 동적인 변화를 보였습니다. 이번 연구는 예술가의 개인적인 진화를 이해할 수 있는 새로운 방법론을 제시하며, 문화 및 창조적 변화의 연구에도 널리 적용될 수 있습니다.



### Hamming Attention Distillation: Binarizing Keys and Queries for Efficient Long-Context Transformers (https://arxiv.org/abs/2502.01770)
- **What's New**: 이 논문에서는 Hamming Attention Distillation (HAD)이라는 새로운 프레임워크를 소개합니다. 이 방법은 어텐션 메커니즘에서 키와 쿼리를 이진화하여 효율성을 극대화합니다. 결과적으로 핵심 운영이 Hamming 거리 계산으로 대체되어 계산 부담이 크게 줄어듭니다.

- **Technical Details**: HAD는 키와 쿼리를 {-1, +1} 벡터로 변환하고, 저충격 활성화를 가지는 어텐션 매트릭스를 희소화하여 긴 문맥 시퀀스를 처리하는 비용을 줄입니다. 이 접근 방식은 기존의 이진화 방법에 비해 높은 수준의 표현력을 유지하면서도 뛰어난 성능을 입증했습니다.

- **Performance Highlights**: HAD는 GLUE, ImageNet 및 QuALITY와 같은 다양한 작업에서 평가되었으며, 이전 이진화 방법들보다 우수한 성능을 보였습니다. GLUE에서는 1.78%의 성능 손실을 기록했고, ImageNet에서는 2.5%로, 모두 기존 어텐션 메커니즘보다 월등한 효과를 나타냈습니다.



### Robust Federated Finetuning of LLMs via Alternating Optimization of LoRA (https://arxiv.org/abs/2502.01755)
Comments:
          A preliminary version was in ICML24 workshop, arXiv:2409.02346

- **What's New**: 본 논문에서는 RoLoRA라는 새로운 연합 학습 프레임워크를 제안합니다. RoLoRA는 LoRA 어댑터를 교대 최적화 방식으로 세밀하게 조정하여 표현력과 견고성을 향상시키는 데 중점을 둡니다. 이 방식은 데이터 전송 비용을 줄이면서도 모델 업데이트의 정확성을 개선합니다.

- **Technical Details**: RoLoRA는 LoRA의 다운 프로젝션 (down-projection) 및 업 프로젝션 (up-projection) 행렬을 동시에 학습하여 최적화를 용이하게 합니다. 이 프레임워크는 이론적 분석과 실험을 통해 연합 학습 성능을 향상시키는 방법을 제시합니다. 실험에서는 MNIST 데이터셋과 RoBERTa-Large, Llama-2-7B와 같은 대형 언어 모델에서 다양한 작업을 수행하여 RoLoRA의 장점을 검증하였습니다.

- **Performance Highlights**: RoLoRA는 적은 수의 파라미터와 클라이언트 수의 증가에도 견고성을 유지하며, 이전 방법들에 비해 더 나은 성능을 제공합니다. 이로 인해 연합 학습 환경에서 LoRA의 표현력을 극대화하고 커뮤니케이션 비용을 줄입니다. 또한, 실험 결과 RoLoRA가 FFA-LoRA보다 더 효율적인 성능을 보인다는 점도 강조되어 있습니다.



### Evaluation of Large Language Models via Coupled Token Generation (https://arxiv.org/abs/2502.01754)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 평가와 순위를 정하는 데 있어 무작위성(randomization)의 영향을 제어해야 한다고 주장합니다. 구체적으로, 저자들은 같은 무작위성을 공유함으로써 서로 연결된 autoregressive 생성을 통해 LLM의 성능을 보다 신뢰성 있게 비교할 수 있는 방법을 제시합니다. 이를 통해 일반적인 autoregressive 생성 방식보다 필요 샘플 수를 줄일 수 있다는 점도 강조합니다.

- **Technical Details**: 이 연구의 핵심은 LLM의 선택적 sampler를 하나의 무작위 소스를 공유하도록 연결(coupled)함으로써, 평가의 일관성을 확보하려는 것입니다. LLM은 사전 지정된 토큰 분포(token distribution)를 기반으로 다음 토큰을 무작위로 선택하는 autoregressive 프로세스를 사용합니다. 이를 통해 발생할 수 있는 평가의 불확실성을 특정 할 수 있으며, 샘플의 수를 줄일 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과, Llama 계열의 여러 LLM 모델을 사용하여, coupled autoregressive 생성은 vanilla autoregressive 생성 방식에 비해 평가에 필요한 샘플 수를 최대 40%까지 줄이는 것을 발견했습니다. 또한, LMSYS Chatbot Arena 플랫폼에서 수집한 데이터에서도 두 방법 간의 pairwise 비교 결과가 상당히 다르게 나타남을 확인했습니다. 이러한 결과는 기존 평가 프로토콜에서 모델 간의 이점이 무작위성에 의해 교란될 수 있음을 시사합니다.



### Grokking vs. Learning: Same Features, Different Encodings (https://arxiv.org/abs/2502.01739)
Comments:
          Code available at: this https URL

- **What's New**: 이번 연구는 grokking과 평범한 steady 학습 사이의 본질적인 차이를 탐구합니다. grokking은 과적합 후에 일반화가 갑자기 나타나는 학습 방식으로, 본 연구는 두 가지 작업에서 각 경로를 통해 학습된 모델의 특성 및 압축 가능성을 비교합니다. 실험 결과, 두 모델은 동일한 특성을 학습하지만, 이 특성을 인코딩하는 효율성에서 큰 차이를 보입니다.

- **Technical Details**: 연구에서 제안된 'compressive regime'은 steady 학습의 새로운 접근 방식으로, 모델 손실과 압축 가능성 사이의 선형적인 trade-off가 발생합니다. 특히, modular addition 작업에서 초기화 시 가중치 조정이 효과적으로 모델의 압축을 향상시키는 것을 발견했습니다. 또한, Fisher Information Metric(FIM)을 활용하여 모델 개발 과정을 정보 기하학적 관점에서 분석하고 있습니다.

- **Performance Highlights**: 모델의 개발 과정은 작업에 따라 달라지며, grokking의 진행 경로는 정보 공간에서 직선을 따라가고 있습니다. 두 가지 작업에서 grokking과 steady 학습을 통해 학습된 특징은 동일하지만, 압축 방식이 다름을 보여주었습니다. 이러한 차이는 실제로 두 접근 방식의 효율성과 효과성을 이해하는 데 기여할 수 있습니다.



### ACECODER: Acing Coder RL via Automated Test-Case Synthesis (https://arxiv.org/abs/2502.01718)
Comments:
          9 pages, 1 figure, 7 tables

- **What's New**:  이 논문에서는 코드 모델 훈련을 향상시키기 위해 자동화된 대규모 테스트 케이스 합성을 활용합니다. 특히, 기존 코드 데이터에서 (question, test-cases) 쌍을 생성하는 파이프라인을 설계하여 신뢰할 수 있는 보상 데이터 부족 문제를 해결합니다. 이 연구는 코드 생성 모델에서 강화 학습(RL)의 잠재력을 강조합니다.

- **Technical Details**:  'AceCode-89K'라는 대규모 검증 가능한 코드 훈련 데이터셋을 구축했습니다. 이 데이터셋은 기존 SFT 데이터셋에서 코드 문제를 수집하고, GPT-4o-mini를 이용하여 문제를 LeetCode 스타일로 재작성하며, 그에 따른 20개의 테스트 케이스를 생성하는 과정을 포함합니다. 최종적으로 89K 질문과 300K 테스트 케이스가 쌍으로 구성된 데이터셋을 생성하였습니다.

- **Performance Highlights**:  강화 학습을 통해 Llama-3.1-8B-Ins와 Qwen2.5-Coder-7B-Ins 모델의 성능이 평균 각각 10점과 5점 개선되었습니다. 최적화 단계가 80단계에 불과한 HumanEval-plus에서 25%의 개선을 보였으며, 이는 모델의 파라미터 조정을 통해 달성되었습니다. 이 연구는 코드 생성 모델에서 RL 훈련의 큰 잠재력을 보여줍니다.



### Process-Supervised Reinforcement Learning for Code Generation (https://arxiv.org/abs/2502.01715)
- **What's New**: 본 논문에서는 프로세스 감독 강화 학습(process-supervised reinforcement learning)을 활용한 PRLCoder라는 혁신적인 프레임워크를 소개합니다. 이를 통해 코드 생성의 성능을 향상시키고자 하며, 새로운 전략인 'statement mutation/refactoring-compile and execution verification'을 통해 자동으로 프로세스 감독 데이터를 생성합니다. 연구 결과, PRLCoder는 기존의 아웃컴 감독 방식에 비해 성능이 크게 향상됨을 입증하였습니다.

- **Technical Details**: PRLCoder는 세 가지 단계로 구성됩니다: 지도 학습(supervised training), 보상 모델 학습(reward model training), 훈련된 보상 모델을 활용한 강화 학습(reinforcement learning). 본 연구에서는 주어진 코드를 라인별로 변경 및 리팩토링(mutation/refactoring)하여 컴파일러 검증 결과를 통해 프로세스 감독 데이터를 자동으로 생성합니다. 이러한 방법으로 얻어진 데이터는 코드 생성의 정확성을 보장합니다.

- **Performance Highlights**: 실험 결과, PRLCoder는 MBPP와 HumanEval 벤치마크 데이터셋에서 기존 모델보다 10.5% 높은 패스율을 보였으며, 아웃컴 감독 강화 학습에 비해 5.1% 성능 향상을 기록했습니다. 특히 복잡한 코드 생성 과제에서 더욱 두드러진 성능 향상을 보여주었으며, 이는 프로세스 감독이 아웃컴 감독보다 코드 생성에서 더 뛰어난 효과를 발휘함을 의미합니다.



### Position: Towards a Responsible LLM-empowered Multi-Agent Systems (https://arxiv.org/abs/2502.01714)
Comments:
          Under Review

- **What's New**: 최근 Agent AI와 대규모 언어 모델(LLM) 기반의 다중 에이전트 시스템(LLM-MAS)의 발전은 시스템의 책임감 있고 신뢰할 수 있는 운영 필요성을 강조하고 있습니다. LangChain과 Retrieval-Augmented Generation과 같은 도구들은 LLM의 기능을 확장하여 MAS에 더 깊이 통합될 수 있도록 합니다. 그러나 이러한 발전은 LLM 에이전트의 예측 불가능성과 출력의 불확실성 같은 중요한 도전 과제를 동반합니다.

- **Technical Details**: LLM-MAS는 다중 자율 에이전트가 상호작용하여 목표를 달성하는 의사결정 연구의 중요한 분야입니다. LLM의 통합은 방대한 지식 데이터베이스와 고급 추론 능력을 제공하여 인간의 노력을 넘는 효율성을 향상시킵니다. 그러나 LLM 기반 에이전트는 훈련된 데이터셋의 다양성으로 인해 발생하는 예측 불가능한 행동 때문에, 상호 이해를 증진하기 위한 정량적 메커니즘이 필요합니다.

- **Performance Highlights**: LLM-MAS에서의 도전 과제들은 지식의 변동과 잘못된 관점의 전파와 같은 고유한 문제에서 기인합니다. 이러한 문제를 해결하기 위해, 확률 중심의 시스템 아키텍처를 채택하고 불확실성을 정량화하는 것을 직원제로 하는 메커니즘을 통합해야 합니다. 나아가, 신뢰성 있는 지식 동의를 보장하기 위한 엄격한 확률적 프레임워크와 형식적 검증 메커니즘 개발이 필요합니다.



### Aspects of Artificial Intelligence: Transforming Machine Learning Systems Naturally (https://arxiv.org/abs/2502.01708)
- **What's New**: 이 논문에서는 머신 러닝 시스템의 요소들과 그 요소들 간의 관계를 연구합니다. 머신 러닝 요소들은 단순히 분리된 것이 아니라, 알gebraic operations, binary relations와 같은 관계를 통해 서로 연결되어 있습니다. 머신 러닝 시스템 변환을 통해 두 시스템 간의 관계를 유지하는 매핑을 정의하고, 이러한 변환이 최적의 문제 해결 방법을 제공함을 강조합니다.

- **Technical Details**: 논문에서는 머신 러닝 요소와 그 관계를 집합 𝐌𝐌(𝐌, R)으로 구성하여 설명합니다. 각각의 머신 러닝 요소는 고유의 algebraic operations와 relations을 통해 연결되고, 이들 간의 관계는 머신 러닝 모델이 데이터에서 학습하고 성장하는 방식에 중대한 영향을 미칩니다. Yoneda embedding과 같은 기술을 통해 ML 시스템 내의 데이터 및 알고리즘 간의 함수적이고 카테고리적인 관계를 탐색합니다.

- **Performance Highlights**: 머신 러닝 시스템 간의 변환은 ML 시스템을 좀 더 유기적으로 연결할 수 있는 방법을 제공합니다. 특히, 머신 러닝 시스템 변환의 구조적 맵은 자연 변환(natural transformations)으로 설명되며, 이는 시스템 변환의 비교 및 분석을 가능하게 합니다. 이러한 접근 방식은 실제 문제를 해결할 수 있는 진화된 머신 러닝 시스템 발전에 기여할 수 있습니다.



### CLIP-DQA: Blindly Evaluating Dehazed Images from Global and Local Perspectives Using CLIP (https://arxiv.org/abs/2502.01707)
Comments:
          Accepted by ISCAS 2025 (Oral)

- **What's New**: 이번 논문에서는 Blind Dehazed Image Quality Assessment (BDQA) 문제를 해결하기 위해 Contrastive Language-Image Pre-Training (CLIP) 모델을 적응시키는 새로운 접근 방식을 제안합니다. CLIP은 대규모 이미지-텍스트 쌍으로 사전 학습되어 있으며, 이 연구에서는 그것을 BDQA 작업에 맞게 효과적으로 조정합니다. 특히, 인지 시스템의 특성을 반영하여 전역(global) 및 지역(local) 정보를 활용하는 방식으로 개선합니다.

- **Technical Details**: 제안된 방법인 CLIP-DQA는 전통적인 BDQA 방법과는 달리, 사람의 시각적 인지를 기반으로 한 계층적 특징을 활용하여 이미지를 평가합니다. 입력 이미지를 패치(patch)로 분할한 후 각각의 품질 점수를 추정하고 평균 점수를 전체 품질 점수로 사용합니다. CLIP의 두 가지 브랜치(vision branch와 language branch)를 조정하여 입력된 계층적 정보를 품질 점수로 정확하게 매핑하는 기법을 적용합니다.

- **Performance Highlights**: 실험 결과, CLIP-DQA는 기존 BDQA 방법에 비해 더 정확한 품질 예측을 보여주었습니다. 두 개의 실제 DQA 데이터셋에서 평가하였으며, 제안된 방법은 여러 관련 연구에 비해 높은 성능을 입증하였습니다. 이 연구의 결과는 향후 이미지 디헤이징 알고리즘의 평가 및 최적화를 위한 중요한 기초 자료로 활용될 수 있습니다.



### Comply: Learning Sentences with Complex Weights inspired by Fruit Fly Olfaction (https://arxiv.org/abs/2502.01706)
Comments:
          Accepted at NICE2025

- **What's New**: 이 논문에서는 생물학에서 영감을 받는 신경망을 통해 새로운 단어 임베딩(word embeddings) 모델인 Comply를 제안합니다. Comply는 FlyVec의 성능을 초월하면서도 더욱 생물학적으로 그럴듯한 표현을 사용하고 있습니다. 새로운 접근법은 시간 정보를 통합하여 임베딩 품질을 향상시키는 것을 목표로 합니다.

- **Technical Details**: Comply는 복소수(complex numbers)로 표현된 가중치를 통해 시퀀스 표현을 학습할 수 있도록 단일 층 신경망을 설계했습니다. 이 방법은 복잡한 입력을 사용하여 임베딩을 생성하며, 시간 패턴을 반영하여 우수한 성능을 보입니다. 또한, Comply는 비지도 에너지 함수(unsupervised energy function)의 확장을 활용하여 복소수 파라미터 행렬을 학습합니다.

- **Performance Highlights**: Comply는 문장 표현에서 높은 성능을 발휘하며, 특히 기존의 FlyVec보다도 우수한 결과를 보여줍니다. 이는 추가적인 파라미터 없이도 이루어지며, 기존의 큰 모델들과 동등한 성능을 자랑합니다. 실험 결과는 Comply가 FlyVec의 단순성과 해석 가능성을 유지하면서도 더 나은 임베딩 품질을 제공함을 나타냅니다.



### QLESS: A Quantized Approach for Data Valuation and Selection in Large Language Model Fine-Tuning (https://arxiv.org/abs/2502.01703)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLM)의 메모리 요구 사항을 줄이기 위한 새로운 방법인 	extbf{QLESS}(Quantized Low-rank Gradient Similarity Search)를 제안합니다. QLESS는 고차원 그래디언트를 로우-랭크(LoRA) 기반 무작위 프로젝션을 통해 저차원으로 압축하고, 이를 저비트폭 표현으로 양자화하여 메모리 효율적인 데이터 평가 및 선택을 가능하게 합니다. 실험 결과, QLESS는 LESS와 유사한 데이터 선택 성능을 보이면서 메모리 사용량을 최대 16배 줄이는 데 성공했습니다.

- **Technical Details**: QLESS는 그래디언트 데이터 저장소에 absmax 기반 양자화를 통합함으로써, 고정밀 그래디언트를 양자화된 저비트 그래디언트로 대체합니다. 이를 통해 메모리 요구 사항을 크게 줄이고도 영향 계산의 무결성을 유지할 수 있습니다. 연구에서는 무작위 프로젝션을 통해 압축된 그래디언트를 정규화하여 데이터 평가 과정의 신뢰성을 확보합니다.

- **Performance Highlights**: 1비트 양자화된 그래디언트가 16비트 그래디언트와 유사한 성능을 자주 내는 것으로 나타났습니다. 이는 그래디언트 기반 데이터 평가에서 정밀도와 효능 간의 전통적인 가정을 도전하며, 극단적 양자화 하에서도 영향 계산의 견고성에 대한 흥미로운 질문을 제기합니다. QLESS는 효율성과 확장성을 결합하여 LLM의 지침 조정에 실용적인 솔루션을 제공합니다.



### BARE: Combining Base and Instruction-Tuned Language Models for Better Synthetic Data Generation (https://arxiv.org/abs/2502.01697)
- **What's New**: 이 논문에서는 **Base-Refine (BARE)**라는 새로운 합성 데이터 생성 방법을 소개합니다. 기존의 instruct-tuned 모델에서는 다양성과 품질을 동시에 충족시키는 데 한계가 있었으나, BARE는 기본 모델의 다양성과 instruct-tuned 모델의 품질을 결합하여 향상된 성능을 제공합니다. 연구 결과, BARE를 사용하여 생성된 데이터로 모델을 미세 조정할 경우, 기존의 최첨단 방법에 비해 성능 향상이 효과적임을 보여주었습니다.

- **Technical Details**: BARE는 두 단계의 프로세스를 통해 합성 데이터를 생성합니다. 첫 번째 단계에서는 기본 모델을 사용하여 다양한 출력을 생성하고, 두 번째 단계에서는 이를 instruct-tuned 모델로 정제합니다. 이러한 방법론을 통해 생성된 데이터는 다양한 데이터 세트를 제작할 수 있으며, 下游 (downstream) 과제의 성능을 향상시킴을 입증하였습니다. 논문에서는 **GSM8K** 및 **RAFT**와 같은 현대적인 작업에서 BARE의 효과를 실험하였습니다.

- **Performance Highlights**: BARE로 생성된 데이터로 fine-tuning을 수행 시, 최소 1,000개의 샘플을 사용하여도 기존의 최첨단 모델과 비교할 수 있는 성능에 도달할 수 있음을 보여줍니다. 예를 들어, BARE로 생성된 데이터는 GSM8K에서 **101%**의 성능 향상을 제공하며, RAFT에서는 **18.4%**의 성능 향상이 나타났습니다. 이러한 향상은 기본 모델에서 생성된 데이터가 실제 데이터의 다양성을 잘 반영할 수 있음을 시사합니다.



### Graph Neural Networks for Identifying Steady-State Behavior in Complex Networks (https://arxiv.org/abs/2502.01693)
Comments:
          12 pages, 7 figures

- **What's New**: 본 논문에서는 복잡한 네트워크에서의 선형 역학 시스템의 동작을 식별하기 위한 새로운 그래프 신경망(Graph Neural Network, GNN) 프레임워크를 개발했다고 소개합니다. 이 모델은 네트워크 행렬의 주요 고유벡터(Principal Eigenvector, PEV)에 대한 역참여비율(Inverse Participation Ratio, IPR) 값을 바탕으로 학습 데이터셋을 만들고, 이를 통해 안정 상태 동작을 정확히 예측합니다. 특히 다양한 크기의 네트워크에서 강력한 성능을 보이며, 현실 세계의 그래프를 사용하여 검증도 진행했습니다.

- **Technical Details**: 연구에서는 미지의 네트워크 구조 𝒢={V,E}를 기반으로 한 선형 역학 프로세스 𝒟를 고려합니다. 이 코드는 각 노드의 동적 상호작용을 구체화하기 위해 첨자 ai,j로 표현되는 인접 행렬을 사용합니다. 선형 동적 시스템의 모델 파라미터인 α와 β는 딥러닝 모델을 통해 배운 특성을 파악하는 데 사용됩니다. 또한, GNN 아키텍처를 사용하여 초기 상태 𝒙(0)을 입력으로, 시스템의 장기적인 안정적 동작 𝒙*를 예측하는 과정을 설명합니다.

- **Performance Highlights**: 모델은 다양한 크기의 네트워크를 대상으로 훈련했으며, 특히 작은 네트워크에서 훈련된 후 더 큰 네트워크에 잘 일반화되는 장점을 보입니다. GNN의 효율적인 학습 성능 덕분에, 선형 동적 상태를 높은 정확도로 식별할 수 있었습니다. 본 연구에서 개발된 프레임워크는 복잡한 시스템의 정보 전파 및 그 행동을 이해하기 위한 중요한 도구로 자리매김할 것으로 기대됩니다.



### Fast Direct: Query-Efficient Online Black-box Guidance for Diffusion-model Target Generation (https://arxiv.org/abs/2502.01692)
- **What's New**: 이번 연구에서는 사전 훈련된 diffusion 모델을 기반으로 하는 새로운 온라인 블랙박스 타겟 생성 알고리즘인 Fast Direct를 제안합니다. 이 알고리즘은 데이터 매니폴드에서 의사 타겟을 구축하여 신속하고 효율적인 쿼리 수집을 지원하며, 특히 비선형 목표 함수를 다룰 수 있습니다.

- **Technical Details**: Fast Direct는 guided noise sequence optimization (GNSO) 기술을 활용하여, 주어진 타겟을 향해 샘플링 프로세스를 안내합니다. 이 과정에서 약 50단계만으로도 초기 생성에서 주어진 입력 타겟으로 신속하게 적응하도록 설계되었습니다. GNSO는 노이즈 시퀀스를 유니버설 방향으로 업데이트하여, 다양한 실험 환경에서도 안정적이고 일관된 성능을 발揮합니다.

- **Performance Highlights**: Fast Direct 알고리즘은 12개의 고해상도 이미지 생성 작업과 6개의 3D 분자 생성 작업에서 수행된 실험을 통해 쿼리 효율성이 6배에서 10배, 11배에서 44배까지 향상된 결과를 보여주었습니다. 이 알고리즘은 또한 새로운 프롬프트에 대한 일반화 능력을 입증하며, 실제 세계 응용에서도 유망한 성과를 나타냅니다.



### Agent-Based Uncertainty Awareness Improves Automated Radiology Report Labeling with an Open-Source Large Language Mod (https://arxiv.org/abs/2502.01691)
- **What's New**: 이 연구는 복잡하고 비영어 텍스트(예: 히브리어)의 방사선 보고서에서 구조화된 데이터를 신뢰성 있게 추출하기 위한 새로운 접근 방식을 소개합니다. 특히, 의학적 응용에서 LLM의 예측 신뢰도를 향상시키기 위해 에이전트 기반의 불확실성 인식 방법을 도입하였습니다. 이 방법은 2010년부터 2023년까지 크론병 환자의 히브리어 방사선 보고서 9,683개를 분석하였습니다.

- **Technical Details**: 연구팀은 512개의 보고서를 수동으로 주석 처리하여 6개의 위장 기관과 15개의 병리학적 발견을 기록하였고, 나머지 보고서는 HSMP-BERT를 사용하여 자동으로 주석 처리하였습니다. Llama 3.1 (Llama 3-8b-instruct)과 Bayesian Prompt Ensembles (BayesPE)를 통해 구조화된 데이터 추출이 이루어졌고, 이 과정에서는 불확실성을 추정하기 위해 의미적으로 동등한 6개의 프롬프트가 사용되었습니다.

- **Performance Highlights**: 에이전트 기반의 의사 결정 모델이 여러 프롬프트 출력을 통합하여 5개의 신뢰 수준으로 조정된 불확실성을 제공하며, 성능은 정확도, F1 점수, 정밀도, 재현율 및 Cohen's Kappa로 평가되었습니다. 에이전트 기반 모델은 모든 지표에서 기본 모델보다 우수한 성능을 보였고, F1 점수는 0.3967, 재현율은 0.6437로 나타났습니다. 불확실성이 높은 사례를 필터링한 후 F1 점수는 0.4787로 개선되었고 Kappa 점수는 0.4258로 증가했습니다.



### scGSDR: Harnessing Gene Semantics for Single-Cell Pharmacological Profiling (https://arxiv.org/abs/2502.01689)
- **What's New**: 단일 세포 시퀀싱(single-cell sequencing) 기술의 발전으로 약물 내성(drug resistance) 탐색이 혁신적으로 변화했습니다. 이러한 연구는 세포 이질성(cellular heterogeneity)이 정밀 의학(precision medicine)에서 중요한 역할을 한다는 점을 강조합니다. 본 연구에서 개발된 scGSDR 모델은 기존의 단일 세포 약물 반응 데이터에서 학습하며, 약물에 대한 세포 반응을 신속하게 주석(annotation)할 수 있는 능력을 제공합니다.

- **Technical Details**: scGSDR은 세포 상태(cellular states) 및 유전자 신호 경로(gene signaling pathways)에 기반한 두 가지 계산 파이프라인을 통합합니다. 이 모델은 유전자 의미론(gene semantics)을 포함함으로써 예측 성능을 향상시키고, 약물 내성 관련 주요 경로(key pathways)를 식별하는 해석 가능성 모듈을 채택하고 있습니다. 연구에서 수행된 16개의 실험과 11종의 약물에 대한 검증을 통해 scGSDR의 우수한 예측 정확성이 입증되었습니다.

- **Performance Highlights**: scGSDR은 단일 약물(predictions) 뿐만 아니라 약물 조합(drug combinations) 시나리오에도 적용할 수 있습니다. 이 모델은 기존 약물 표적 유전자(drug target genes)의 경로를 활용하여 세포-경로 주의 점수(cell-pathway attention scores)를 생성, 생물학적 해석이 가능한 결과를 도출했습니다. 높은 AUROC, AUPR, F1 Scores를 기록하며, BCL2 및 PIK3CA와 같은 주요 유전자의 문헌 검토를 통해 약물 관련 유전자의 타당성을 확인했습니다.



### Leveraging Joint Predictive Embedding and Bayesian Inference in Graph Self Supervised Learning (https://arxiv.org/abs/2502.01684)
- **What's New**: 이 논문에서는 기존의 self-supervised learning (SSL) 방법의 한계를 극복하기 위해 대비하는 목표(contrastive objectives)와 negative sampling을 제거하고, 그래프의 의미적 및 구조적 정보를 보존하는 새로운 joint embedding predictive framework를 제안합니다. 특히, 이 프레임워크는 Gaussian Mixture Models (GMMs)를 통해 가짜 레이블(pseudo-label)을 도입하여 노드의 구별 가능성을 높이며, 기존 최첨단 Graph SSL 방법들보다 우수한 성능을 달성하는 것을 목표로 합니다. 이러한 혁신적인 접근은 computational efficiency를 높이고 레이블이 없는 노드의 기여도를 적절히 반영하는 데 기여하고 있습니다.

- **Technical Details**: 제안된 방법은 두 개의 인코더(context encoder와 target encoder)를 사용하여 잠재 변수 z에 조건화된 서브그래프 임베딩을 예측합니다. context encoder는 무작위로 노드를 제거하여 서브그래프를 처리하고, target encoder는 원본 그래프에서 타겟 노드 표현을 생성합니다. 이후 샘플링된 서브그래프에서 위치 정보를 추가하여 임베딩이 파생되며, 예측기 네트워크는 이러한 정보를 기반으로 적절한 타겟 임베딩을 매핑합니다. 특히 Gaussian Mixture Modeling을 통해 예측된 가짜 레이블 점수를 추가하여 그래프의 의미를 보존하면서 노드 표현을 향상시킵니다.

- **Performance Highlights**: 전반적인 실험 결과, 제안된 방법론이 기존의 Graph SSL 방법들보다 우수한 성능을 발휘하는 것으로 나타났습니다. 특히, 대비 손실(contrastive loss)이나 복잡한 디코더 없이도 뛰어난 성과를 달성하며, 그래프 기반 애플리케이션의 표현 학습에 있어 강력한 대안으로 자리 잡을 수 있음을 보여주었습니다. 이 논문은 공간적 및 의미적 그래프 특성을 활용하여 다운스트림 작업에 효과적으로 연결되는 새로운 패러다임을 제시하고 있습니다.



### LLM-Powered Benchmark Factory: Reliable, Generic, and Efficien (https://arxiv.org/abs/2502.01683)
- **What's New**: 최근 대형 언어 모델(LLM)의 급격한 발전으로 인해 모델 공급과 응용 수요가 증가하고 있습니다. 이에 따라 신뢰할 수 있고 일반적인 벤치마크 생성기가 필요하지만, 기존의 LLM 벤치마크 생성기는 일반화 가능성 부족과 신뢰성 문제를 안고 있습니다. 이 논문에서는 자동화된 평가 프레임워크를 제안하고, 이를 기반으로 LLM을 직접 끌어내는 방식에서의 강점과 약점을 분석합니다.

- **Technical Details**: 벤치마크 생성기를 위한 자동화된 평가 프레임워크는 네 가지 차원과 열 개의 기준으로 구성되어 있습니다. 이를 통해 LLM을 벤치마크 생성기로 직접 활용할 때 나타나는 문제점을 식별하고 해결하는 다양한 방법이 BenchMaker로 통합되어 개발되었습니다. BenchMaker는 단계별 자체 정정 생성 및 갈등 유도 대조 분별법을 이용해 샘플의 신뢰성을 강화하는 방법을 제시합니다.

- **Performance Highlights**: 다양한 LLM과 과제를 대상으로 실시한 실험 결과, BenchMaker는 인간이 주석을 단 벤치마크와 비교하여 모든 지표에서 우수하거나 동등한 성능을 달성했습니다. 특히, BenchMaker는 12개의 LLM에 대해 높은 일관된 평가 결과를 제공하며(0.967 Pearson correlation against MMLU-Pro), 샘플당 $0.005 및 0.38분의 리소스를 소모합니다.



### Neurosymbolic AI for Travel Demand Prediction: Integrating Decision Tree Rules into Neural Networks (https://arxiv.org/abs/2502.01680)
Comments:
          9 pages, 5 figures, this paper is under review in the conference

- **What's New**: 이번 연구는 여행 수요 예측에 있어 Neurosymbolic AI(신경 상징 인공지능) 프레임워크를 도입했습니다. 이는 결정 트리(Decision Tree) 기반의 상징 규칙과 신경망(Neural Networks)을 통합하여 여행 수요를 예측하는 새로운 접근 방식을 제안하고 있습니다. 이 프레임워크는 다양한 데이터 소스, 예를 들어 지리정보(geospatial), 경제(economic), 이동(mobility) 데이터셋을 활용하여 종합적인 특징 세트를 구축합니다.

- **Technical Details**: 제안된 방법에서는 결정 트리를 사용하여 핵심 패턴을 포착하는 해석 가능한 if-then 규칙을 추출하고, 그 규칙을 신경망의 추가적인 특징으로 통합하여 예측 능력을 향상시킵니다. 실험 결과는 상징 규칙으로 풍부해진 결합 데이터셋이 여러 평가 지표, 즉 평균 절대 오차(Mean Absolute Error, MAE), 결정 계수(
R^2), 통근자(Common Part of Commuters, CPC) 등에서 일관되게 단독 데이터셋을 초월하는 성과를 보였습니다. 더불어, 미세한 분산 임계치(예: 0.0001)에서 선택된 규칙은 세밀한 관계를 포착하는 데 뛰어난 효과를 보이며 예측 오차를 감소시키고 관찰된 통근자 패턴과 일치합니다.

- **Performance Highlights**: 이 연구는 신경 학습(neural learning)과 상징적 학습(symbolic learning) 패러다임을 결합한 Neurosymbolic 접근법이 해석 가능성과 정확성을 동시에 달성할 수 있음을 보여줍니다. 구체적으로, 적용된 프레임워크는 기존의 방법보다 여행 수요 예측에서 더 나은 성과를 보이며, 효율적인 이동성과 경제적 지속 가능성을 위한 중요한 기여를 합니다. 또한 종합적인 데이터 활용은 여행 수요 예측의 신뢰성을 더욱 높이는 데 기여하고 있습니다.



### LEAD: Large Foundation Model for EEG-Based Alzheimer's Disease Detection (https://arxiv.org/abs/2502.01678)
- **What's New**: 본 논문에서는 세계 최대 규모의 EEG-AD 데이터셋을 구축하고, 이를 기반으로 하는 LEAD라는 EEG 기반 알츠하이머 질환(AD) 검출을 위한 첫 번째 대규모 기본 모델을 제안합니다. LEAD는 데이터 선택, 전처리, 자체 지도 대비 사전 훈련, 세부 조정 및 피험자 간 평가와 같은 전체 파이프라인을 포함합니다. 이를 통해 기존 방법들이 직면한 피험자 간 변동성 문제를 효과적으로 해결할 수 있는 접근 방식을 제시합니다.

- **Technical Details**: LEAD 모델은 11개의 EEG 데이터셋에서 사전 훈련되고, 5개의 AD 데이터셋에서 통일된 세부 조정이 이루어집니다. 모델은 시간적 및 채널 임베딩을 통해 EEG의 시간 및 공간 차원에서 특징을 포착하며, 자체 지도 사전 훈련 설계는 유용한 일반 EEG 특징을 추출하기 위해 샘플 수준 및 피험자 수준의 대비 학습을 포함합니다. 데이터 전처리 단계에서 모든 데이터를 19개의 표준 채널로 정렬하여 다양한 데이터셋에 대해 훈련할 수 있도록 하고 있습니다.

- **Performance Highlights**: 모델은 샘플 수준에서 최대 9.86%의 F1 점수 향상을 달성했으며, 피험자 수준에서도 최대 9.31%의 개선을 보여줍니다. 다양한 AD 데이터셋에서의 최종 피험자 수준 분류 결과는 각각 91.34%, 89.98%, 100.00%, 84.42%, 91.86%로 나타났으며, 이는 기존의 최첨단(SOTA) 방법들과 비교했을 때 유의미한 성과를 나타냅니다. 우리의 연구 결과는 대조적 사전 훈련 및 통일된 세부 조정의 효과를 강하게 입증하며, EEG 기반 AD 검출의 발전에 기여할 것입니다.



### AI Scaling: From Up to Down and Ou (https://arxiv.org/abs/2502.01677)
- **What's New**: 이 논문은 AI 스케일링의 새로운 접근 방식을 제시합니다. 전통적으로 스케일링은 모델의 크기와 성능을 높이는 것을 의미했으나, 효율성, 적응성, 협업을 고려한 포괄적인 프레임워크로서 스케일 업, 스케일 다운, 스케일 아웃을 포괄합니다. 이러한 패러다임은 탄소 발자국 감소, 공정한 접근 보장, 그리고 도메인 간 협력을 위한 중요한 기술적 및 사회적 과제를 해결하는 데 기여할 수 있습니다.

- **Technical Details**: 아이디어는 스케일 다우닝(scaling down)과 스케일 아우팅(scaling out)으로의 전환이 필요하다는 것입니다. 스케일 다운은 대규모 모델의 구조와 성능을 분석하여 불필요한 파라미터를 제거하고, 특정 작업에 대해 더 작고 효율적인 모델로 조정하는 과정입니다. 반면, 스케일 아웃은 경량화된 모델을 활용하여 분산 환경에서 대규모 배포를 가능하게 하며, 이를 통해 중앙 집중화된 단일 AI 모델 대신 다양한 전문화된 모델을 통해 AI 생태계를 형성합니다.

- **Performance Highlights**: AI 스케일링은 헬스케어, 스마트 제조, 콘텐츠 생성 등 다양한 분야에서 혁신적인 응용 프로그램을 발견하도록 합니다. 이러한 혁신은 효율성, 개인화 및 글로벌 연결성을 통해 AI의 미래 방향성을 제시합니다. 또한, 모델의 복잡성과 해석 가능성, 자원 제약 관리, 윤리적 발전 등을 통한 주요 과제를 강조하고, AGI(Artificial General Intelligence)를 향한 발전의 길을 제시합니다.



### Semantic Communication based on Generative AI: A New Approach to Image Compression and Edge Optimization (https://arxiv.org/abs/2502.01675)
Comments:
          PhD thesis

- **What's New**: 이 논문은 지능형 장치가 생성하는 방대한 데이터를 처리하는 데 있어 커뮤니케이션 네트워크가 직면한 도전 과제를 다루고 있습니다. 본 연구는 의미 기반 커뮤니케이션(semantic communication)과 생성 모델(generative models)을 통합하여 이미지 압축(image compression) 및 엣지 네트워크 자원 할당(edge network resource allocation)의 최적화를 이루었습니다. 이는 단순히 비트 중심 시스템에서 벗어나 특정 의미를 전달하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 이 연구의 핵심은 생성적 적대 신경망(Generative Adversarial Networks)과 노이즈 제거 확산 확률 모델(Denoising Diffusion Probabilistic Models)을 사용한 의미 보존 이미지 압축(semantic-preserving image compression)의 설계입니다. 이러한 모델들은 오직 의미와 관련된 특징들만 인코딩하여 이미지를 압축하고, 최소한의 전송으로 고품질 재구성을 가능하게 합니다. 또한, 정보 병목 원칙(Information Bottleneck principle)과 확률적 최적화(stochastic optimization)를 이용하여 자원을 동적으로 할당하고 효율성을 높이는 목표 지향 엣지 네트워크 최적화 프레임워크를 도입합니다.

- **Performance Highlights**: 성능 비교에서는 의미 인식 모델과 전통적인 이미지 압축 기법을 고전적 및 의미 평가 메트릭(classical and semantic evaluation metrics)을 사용하여 비교했습니다. 결과는 생성 AI와 의미 기반 커뮤니케이션의 결합이 현대의 데이터 중심 애플리케이션 요구사항을 충족하는 보다 효율적인 의미 목표 지향 커뮤니케이션 네트워크를 만드는 가능성을 보여줍니다. 이러한 접근 방식은 실시간 애플리케이션에 적합한 컴퓨팅 효율성과 커뮤니케이션 효과성을 균형 있게 제공합니다.



### Multilingual State Space Models for Structured Question Answering in Indic Languages (https://arxiv.org/abs/2502.01673)
- **What's New**: 이번 연구는 인디언 언어에서의 질문 응답(Question Answering, QA) 작업에 대해 State Space Models (SSMs)의 최초 적용을 다룹니다. SSM은 시퀀스 데이터의 장기 및 단기 의존성을 모델링할 수 있어 복잡한 문법 구조를 효율적으로 처리하는 데 유리합니다. 연구진은 다양한 인디언 언어 데이터셋에서 여러 개의 SSM 아키텍처를 평가하고, 이들 모델이 언어의 미세한 뉘앙스를 효과적으로 포착한다는 결과를 도출하였습니다.

- **Technical Details**: SSMs는 입력 매트릭스와 상태 전이가 포함된 구조로, 시스템의 동적 과정을 표현하는 상태 벡터를 가지고 있습니다. 이 연구에서는 기존의 Transformer 아키텍처와 비교하여 SSM의 장점과 단점을 강조하고, 입력 데이터의 선택적 전파를 가능하게 하는 Mamba라는 기술적 발전을 소개합니다. SSM 모델은 효율성과 확장성을 중시하여 이론적으로 계산 복잡성을 선형으로 유지하면서 장기간의 의존성 문제를 해결할 수 있습니다.

- **Performance Highlights**: 연구 결과 SSM 모델은 질문 해석, 맥락 정렬 및 답변 생성을 통해 QA 시스템의 성능을 향상시키는 데 기여했습니다. 특히, SSM은 인디언 언어의 복잡성을 관리하는 데 필요한 조건들을 잘 충족시키며,低-Resource 상황에서도 추가적인 최적화를 제안합니다. 이 연구는 인디언 언어의 QA 시스템 개발을 위한 기초 벤치마크를 제공하며, 향후 관련 연구의 방향성을 제시합니다.



### Doubly Robust Monte Carlo Tree Search (https://arxiv.org/abs/2502.01672)
- **What's New**: 본 논문은 복잡한 환경에서 샘플 효율성과 결정 품질을 향상시키기 위한 새로운 알고리즘인 Doubly Robust Monte Carlo Tree Search (DR-MCTS)를 소개합니다. DR-MCTS는 Monte Carlo Tree Search (MCTS) 프레임워크에 Doubly Robust (DR) 오프 정책 추정을 통합하여 이론적 보장을 제공합니다. Tic-Tac-Toe 및 부분적으로 관찰 가능한 VirtualHome 환경에서의 실험을 통해 DR-MCTS의 우수성을 입증했습니다.

- **Technical Details**: DR-MCTS는 MCTS의 전통적인 구성을 활용하여, 정책의 선택, 확장, 시뮬레이션 및 후향 전파의 네 단계로 구성된 검색 트리를 구축하고 업데이트하는 방식입니다. 또한, PUCT(Polynomial Upper Confidence Trees) 알고리즘을 사용하여 탐사 효율성을 높이고, DR 추정을 활용하여 더 나은 값을 예측합니다. 이론적으로 DR-MCTS는 비편향성(unbiasedness) 및 분산 감소(variance reduction)을 보장합니다.

- **Performance Highlights**: DR-MCTS는 Tic-Tac-Toe 게임에서 88%의 승률을 기록하며, 표준 MCTS의 10% 승률과 대조됩니다. VirtualHome의 복합 작업에서 DR-MCTS는 20.7%의 성공률을 기록한 반면, 표준 MCTS는 10.3%에 불과합니다. 이러한 결과는 DR-MCTS가 특히 큰 Language Models(LLMs)에서 뛰어난 샘플 효율성을 보여주어, 복잡한 실제 상황에서 효율적인 결정이 가능하다는 것을 입증합니다.



### Life-Cycle Emissions of AI Hardware: A Cradle-To-Grave Approach and Generational Trends (https://arxiv.org/abs/2502.01671)
- **What's New**: 이번 연구는 AI 하드웨어 가속기의 생애주기 분석(LCA)을 최초로 수행하며, 온실가스(GHG) 배출에 대한 포괄적인 평가를 제공합니다. 이 연구에서는 원자재 추출부터 제조, 운영, 폐기까지 모든 단계에서의 배출량을 분석하였으며, AI 가속기의 제조 시 발생하는 배출량 또한 처음으로 문서화하였습니다. 연구 결과, Compute Carbon Intensity (CCI)라는 새로운 지표가 도입되어 하드웨어의 지속 가능성을 평가하는 데 기여하고 있습니다.

- **Technical Details**: 연구는 Google의 Tensor Processing Unit (TPU) 5종을 대상으로 하며, 각 TPU 모델의 세부 사항과 성능을 분석합니다. 기능적 단위는 데이터 센터에 배포된 AI 컴퓨터 하나로 설정하여, 각 TPU 트레이와 호스트 트레이 간의 연결을 포함합니다. 또한, GHG 프로토콜을 활용하여 전체 하드웨어 생애주기 동안의 온실가스 배출을 정량화하였으며, Scope 2 배출량 계산을 위해 다양한 전력 소비 방법론을 적용하였습니다.

- **Performance Highlights**: 연구 결과, TPU v4i에서 TPU v6e로의 전환을 통해 CCI가 3배 개선되는 것으로 나타났습니다. 이는 오직 4년 만에 이루어진 성과로, Google의 하드웨어 설계 프로세스와 제조 효율성 향상의 효과를 보여줍니다. 다양한 AI 하드웨어 가속기의 실제 운영 배출량을 직접 측정하여, 탄소 효율성 향상에 대한 새로운 통찰을 제공하고 있습니다.



### Addressing Delayed Feedback in Conversion Rate Prediction via Influence Functions (https://arxiv.org/abs/2502.01669)
- **What's New**: 본 논문에서는 지연된 피드백 문제를 해결하기 위해 Influence Function 기반의 새로운 프레임워크인 Delayed Feedback Modeling (IF-DFM)을 제안합니다. IF-DFM은 새롭게 수집된 전환 데이터를 사용하여 모델 매개변수에 미치는 영향을 추정하여 효율적으로 매개변수를 업데이트할 수 있도록 설계되었습니다. 이 프레임워크는 전체 재교육 없이도 모델을 적응시킬 수 있게 해줘, 광고 전환율 예측에서 더욱 강력한 성능을 발휘합니다.

- **Technical Details**: IF-DFM의 핵심은 인플루언스 함수(influence function)를 활용하여 잘못 레이블된 샘플을 올바른 샘플로 전환하는 과정에서 발생하는 매개변수 변화를 매개변수 업데이트로 연결짓는 것입니다. 이는 또한 새로운 행동 데이터를 통합할 수 있는 기능을 제공하여, 모델이 최신 사용자 상호작용에 적응하도록 돕습니다. 이러한 접근은 히에라르크 역행렬-벡터 곱(inverse Hessian-vector product) 계산을 최적화 문제로 재구성하여 계산 효율성을 극대화합니다.

- **Performance Highlights**: 광범위한 실험을 통해 IF-DFM은 기존의 최첨단 방법들보다 뛰어난 예측 정확도와 모델 적응성을 보여주었습니다. 특히, IF-DFM은 광고 전환율 예측에서 실시간 사용자 행동 변화에 민첩하게 반응할 수 있어, 이론적으로 설정된 한계를 넘어서는 성과를 얻었습니다. 이러한 결과는 대용량 데이터 환경에서도 효율적인 성능 개선을 보여줍니다.



### Refining Alignment Framework for Diffusion Models with Intermediate-Step Preference Ranking (https://arxiv.org/abs/2502.01667)
- **What's New**: 이번 연구에서는 Direct Preference Optimization (DPO)을 넘어서 Tailored Preference Optimization (TailorPO) 프레임워크를 제안하고 있습니다. 기존의 DPO는 최종 생성물과 중간 단계에서의 노이즈 샘플 간의 일관된 선호 레이블을 가정하였으나, 우리는 이 가정이 갖는 본질적인 문제를 이론적으로 식별하였습니다. TailorPO는 보다 효과적인 선호 정렬을 위해 생성된 샘플을 단계별 보상에 기반하여 직접적으로 순위를 매기는 방법을 사용합니다.

- **Technical Details**: TailorPO 프레임워크는 중간 단계에서 생성된 노이즈 샘플에 대한 선호를 직접적으로 평가하여, 그에 따른 그래디언트 방향 문제를 해결합니다. 이를 통해 생성된 이미지의 시각적 품질을 개선할 수 있는 방향으로 최적화합니다. 특히, diffusion model의 그래디언트 가이드를 선호 정렬에 통합하여 최적화 효율성을 더욱 높이고 있습니다.

- **Performance Highlights**: 실험 결과, TailorPO를 적용한 모델은 미적이고 인간의 선호에 더 잘 맞는 이미지를 생성하는 능력이 크게 향상되었습니다. 이 연구는 DPO의 한계를 극복하고, 더 높은 품질의 결과물을 생성할 수 있는 새로운 접근 방식을 제시합니다.



### Speculative Ensemble: Fast Large Language Model Ensemble via Speculation (https://arxiv.org/abs/2502.01662)
- **What's New**: 최근 큰 발전이 있었던 Large Language Models (LLMs)에서, Ensemble 방법들이 여러 모델을 결합하여 성능을 향상시키고 있으나, 높은 계산 비용이 문제로 남아있습니다. 본 논문에서는 Speculative Ensemble을 제안하며, 이는 성능 저하 없이 LLM 앙상블의 속도를 향상시키는 새로운 프레임워크입니다. 이를 통해 제안 모델이 토큰을 순차적으로 생성하고, 대형 목표 모델이 이를 병렬로 검증하는 방식을 채택합니다.

- **Technical Details**: Speculative Ensemble (SE)은 두 가지 주요 통찰력을 기반으로 합니다. 첫째, 검증 분포는 제안 및 목표 모델의 앙상블 분포가 될 수 있으며, 둘째, 각 모델을 제안자 및 검증자로 번갈아 사용함으로써 효율성을 더욱 높일 수 있습니다. 이 방법을 n개의 모델로 일반화하고 이론적으로 SE가 표준 앙상블보다 느릴 수 없음을 증명하여, 일반적으로 더 빠른 속도를 달성할 수 있음을 보여줍니다.

- **Performance Highlights**: 실험 결과에 따르면 Speculative Ensemble은 표준 앙상블 기법보다 1.11배에서 2.23배의 속도 향상을 보여주며, 생성 품질을 저하시키지 않습니다. 다양한 작업에서 무작위 표본을 통해 Llama, Vicuna, Qwen 시리즈 등 여러 모델 쌍을 테스트하여 SE의 일관된 가속을 확인했습니다. 이는 특히 가중 앙상블 환경에서 1.34배에서 1.85배의 속도를 달성함을 보여주며, 본 방법의 실용성을 증명합니다.



### Employee Turnover Prediction: A Cross-component Attention Transformer with Consideration of Competitor Influence and Contagious Effec (https://arxiv.org/abs/2502.01660)
- **What's New**: 본 논문은 여러 기업 간의 개별 직원 이직(turnover) 예측을 위한 새로운 딥러닝(deep learning) 접근법을 제안합니다. 기존 연구들은 단일 기업 내의 이직 예측 또는 기업 간 이직 흐름에 초점을 맞춰왔으나, 개별 직원의 이직 예측은 그다지 다뤄지지 않았습니다. 이는 인적 자원 관리(HRM)와 데이터 과학(data science) 분야에서 중요한 연구 과제가 됩니다.

- **Technical Details**: 이 연구에서는 직업 내재성(job embeddedness) 이론을 기반으로 한 딥러닝 모델을 사용하여 여러 기업에서 근무하는 직원들의 이직 가능성을 예측합니다. 실제 데이터셋(real-world dataset)을 기반으로 한 실험 평가를 통해, 본 연구에서 제안한 방법이 최신의 벤치마크 방법들에 비해 우수한 성능을 나타냄을 보여줍니다. 이 논문은 다양한 이직 요인의 기여도(attribution)를 해석하여 이직 예측 솔루션의 실질적인 비즈니스 가치를 강조합니다.

- **Performance Highlights**: 연구 결과, 제안된 딥러닝 모델은 이직 예측 정확도에서 여러 최신 방법들과 비교해 우수한 성능을 보였습니다. 또한, 이직 예측 솔루션을 활용함으로써 채용 담당자(recruiters)가 절감할 수 있는 비용을 추정하였습니다. 이를 통해 기업의 인력 유지 전략을 강화할 수 있는 가능성을 제시하며, 직원 이직 문제를 효과적으로 해결할 수 있는 방안을 제공합니다.



### Longer Attention Span: Increasing Transformer Context Length with Sparse Graph Processing Techniques (https://arxiv.org/abs/2502.01659)
- **What's New**: 이 연구는 Attention 메커니즘을 그래프 문제로 접근하여 진정한 희소성(true sparsity)을 달성하는 알고리즘을 제안합니다. 이를 통해 모델의 메모리와 연산 복잡도를 대폭 줄여 다양한 시퀀스 길이를 효과적으로 처리할 수 있습니다. 특히, NVIDIA A100 GPU를 활용하여 최대 1억 6천만 길이의 시퀀스를 처리할 수 있는 성능을 보여줍니다.

- **Technical Details**: Transformer 모델의 Attention 메커니즘은 입력 토큰의 상호작용을 포착하기 위해 쌍(pairwise) 유사성을 추출합니다. 전통적인 접근법은 희소성(sparsity)을 도입하여 L×L 크기의 Attention 마스크를 활용하였으나, 기존 구현은 효과적인 하드웨어 활용이 부족하였습니다. 본 연구에서는 Attention을 노드로 하는 그래프 구조로 변경하고, 이를 통해 최적의 작업(work optimal) 알고리즘을 개발하여 계산량을 줄이는 데 기여합니다.

- **Performance Highlights**: 연구 결과, FlashAttention 대비 2,097,152 및 1억 6천만 길이의 시퀀스에서 각각 4.46배와 51.06배의 성능 향상을 달성하였습니다. 우리의 알고리즘은 매우 긴 시퀀스 길이 처리가 가능하였으며, PyTorch 백엔드를 구현하여 기존의 LLM들과의 통합이 용이하도록 하였습니다.



### Improving Rule-based Reasoning in LLMs via Neurosymbolic Representations (https://arxiv.org/abs/2502.01657)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 추론 능력을 향상시키기 위해 뉴로심볼릭(Neurosymbolic) 방법을 소개합니다. 이 방법은 숨겨진 상태(hiddens states)를 뉴로심볼릭 벡터로 인코딩하여 문제 해결을 가능하게 하며, 수학적 추론 과제에서의 모델 성능을 향상시킵니다. 실험 결과, 제안된 방법은 체인 오브 사고(chain-of-thought) 프롬프트나 저순위 적응(Low-Rank Adaptation) 보다 더 많은 문제를 올바르게 해결할 수 있음을 보여주었습니다.

- **Technical Details**: 본 연구에서 제안한 방법은 LLM의 숨겨진 상태를 구조화된 심볼릭 표현으로 인코딩하여 신경망 내부에서 심볼릭 알고리즘을 직접 통합합니다. 이전의 토큰 수준 프로그램 합성(token-level program synthesis) 접근법과는 달리, 본 방법은 LLM 내부 표현으로 다양한 심볼릭 규칙을 사용하는 뉴로심볼릭 처리 방식을 개발하여 수학적 구조를 다루는 데 있어서의 효율성과 해석 가능성을 증대시킵니다. 이 연구에서는 벡터 심볼릭 대수(Vector Symbolic Algebras, VSA)를 활용하여 숨겨진 상태 정보를 디코딩하고 다양한 규칙 기반 조작을 수행할 수 있는 가능성을 증명합니다.

- **Performance Highlights**: 제안된 방법은 수치적 추론 과제에서 정확성과 해석 가능성이 현저히 개선되어 체인 오브 사고 및 저순위 적응 방법을 능가하는 성과를 보였습니다. 특히, 실험에서는 평균적으로 $82.86\%$ 낮은 크로스 엔트로피 손실(cross entropy loss)을 기록하고 24.50배 더 많은 문제를 정확하게 해결하는 결과를 나타냈습니다. 이로 인해, 본 연구는 LLM이 대규모 문제 해결에서 신뢰성과 해석 가능성을 동시에 향상시킬 수 있는 잠재력을 지니고 있음을 시사합니다.



### A binary PSO based ensemble under-sampling model for rebalancing imbalanced training data (https://arxiv.org/abs/2502.01655)
Comments:
          22 pages, 18 figures

- **What's New**: 본 논문에서는 불균형 데이터셋(classification problems) 문제를 해결하기 위한 새로운 앙상블(ensemble) 방법을 제안합니다. 이 방법은 앙상블 학습의 장점과 새로운 언더샘플링(under-sampling) 기술을 결합하여, 다수 클래스 샘플의 적합한 조합을 찾아 소수 클래스와 함께 새로운 데이터셋을 구축하는 방식입니다. 이를 통해 원본 데이터셋의 완전성을 최대한 보장하면서 불균형 분류(performance) 성능이 향상됩니다.

- **Technical Details**: 제안된 방법은 바이너리 PSO(instance selection) 기반의 언더샘플링 기법을 사용하며, 다중 목표(multi-objective) 전략을 채택합니다. 이를 통해 다수 클래스 샘플을 효과적으로 선택하고, 소수 클래스와의 결합으로 데이터셋을 구성합니다. 또한, 실험은 기존의 기본 앙상블 방법들과 비교하여 진행되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 단일 앙상블 방법, 최신 언더샘플링 방법, 그리고 전통적인 PSO(instance selection) 알고리즘과 결합한 방법들보다 우수한 성능을 보였습니다. 이로 인해 불균형 데이터셋 처리 분야에서의 경쟁력을 입증하였습니다.



### Hybrid Group Relative Policy Optimization: A Multi-Sample Approach to Enhancing Policy Optimization (https://arxiv.org/abs/2502.01652)
Comments:
          11 Pages, 18 Equations, 1 Table

- **What's New**: 하이브리드 그룹 상대 정책 최적화(Hybrid GRPO)는 기존 PPO와 GRPO의 장점을 결합한 강화 학습 프레임워크로, 경험적 다중 샘플 행동 평가를 포함하여 가치 함수 기반 학습의 안정성을 유지합니다. 이 방법은 경험적 행동 샘플링과 부트스트랩된 가치 추정을 균형 있게 결합하여 샘플 효율성, 학습 안정성 및 분산 증폭 완화를 개선합니다. 또한 이 프레임워크는 대형 언어 모델(LLM)과 실제 에이전트 기반 의사 결정 사이의 격차를 해소하는 확장 가능한 구조를 제공합니다.

- **Technical Details**: 하이브리드 GRPO는 가치 함수 V(s)를 보존하면서 다중 행동 샘플링을 통합하여 정책 업데이트를 향상시킵니다. 이 접근 방식은 시스템적 경험적 샘플링과 적응형 보상 변환을 포함하여 데이터 효율성을 높이고 순전히 경험적 방법에서 발생하는 분산을 줄입니다. 각 정책 최적화 방법(PPO, GRPO, Hybrid GRPO)의 수학적 비교를 통해 이들 간의 장점 추정 및 정책 업데이트 방식의 차이를 분석합니다.

- **Performance Highlights**: 실험 검증을 통해 하이브리드 GRPO는 PPO 및 DeepSeek GRPO에 비해 더 빠른 수렴 속도, 더 안정적인 정책 업데이트, 개선된 샘플 효율성을 달성했습니다. 또한 엔트로피 정규화 샘플링, 계층적 다단계 서브 샘플링, 적응형 보상 정규화 등 다양한 확장을 통해 이 방법론을 더욱 정교화할 수 있는 가능성을 보여줍니다. 이러한 성능 개선은 자율 로봇공학, 금융 모델링 및 AI 기반 제어 시스템 등 다양한 실제 응용 프로그램에서도 활용될 수 있습니다.



### Fine-tuning LLaMA 2 interference: a comparative study of language implementations for optimal efficiency (https://arxiv.org/abs/2502.01651)
Comments:
          11 pages, conference paper. International conference on Artificial Intelligence and Future Civilization

- **What's New**: 이 논문은 Llama2 inference를 최적화하기 위한 비교 연구를 제시합니다. 다양한 프로그래밍 언어와 프레임워크, 예를 들어 TensorFlow, PyTorch, Python, Mojo, C++, Java에 대한 성능을 분석하였습니다. 특히 Mojo SDK는 Apple Silicon을 위한 대형 언어 모델(LLM) inference에 설계된 새로운 프레임워크로, 이의 성능을 다른 언어들과 비교합니다.

- **Technical Details**: 연구는 인퍼런스(inference) 속도, 메모리 소비(memory consumption), 구현 용이성에 대해 광범위한 벤치마킹(benchmarking)을 수행하였습니다. 각 접근 방식의 강점과 한계를 강조하며 병렬 처리(parallel processing) 및 하드웨어 활용(hardware utilization)을 위한 최적화 전략을 제안합니다. 또한, Apple M1 Max에서 수행된 실험을 통해 Mojo SDK의 성능을 C, C++, Rust, Zig, Go, Julia 구현과 비교하였습니다.

- **Performance Highlights**: Mojo SDK는 경쟁력 있는 성능을 보여주며, 사용이 용이하고 Python과의 호환성이 뛰어난 것으로 평가되었습니다. 이는 Apple Silicon에서 LLM inference에 대한 강력한 대안으로 자리 잡을 수 있음을 보여줍니다. 또한, 리소스가 제한된 하드웨어에서 LLM 배포에 대한 넓은 함의를 논의하고, 향후 연구의 잠재적 방향을 제시합니다.



### MIND: Modality-Informed Knowledge Distillation Framework for Multimodal Clinical Prediction Tasks (https://arxiv.org/abs/2502.01158)
Comments:
          Published in Transactions on Machine Learning Research (01/2025), this https URL

- **What's New**: 이번 연구에서는 Modality-INformed knowledge Distillation (MIND) 프레임워크를 제안합니다. MIND는 다양한 크기의 심층 신경망 앙상블에서 지식을 전이하여 보다 작은 멀티모달(student) 모델을 생성하는 지식 증류(knowledge distillation) 기반의 멀티모달 모델 압축 방법입니다. 이를 통해 의료와 관련된 멀티모달 데이터셋의 부족한 샘플 문제를 해결하고 모델의 크기를 효과적으로 조정할 수 있습니다.

- **Technical Details**: MIND는 다중 헤드(joint fusion) 모델을 활용하여 임퓨테이션(imputation)이나 마스킹(masking) 과정 없이도 단일 모달 샘플을 처리할 수 있게 설계되었습니다. 교사 모델들은 단일 모달 네트워크로 구성되어 있어, 학생 모델이 다양한 표현(representation)으로부터 학습할 수 있도록 돕습니다. 이 접근법은 고차원의 임상 데이터에도 최적화된 성능을 제공하며, 멀티모달 학습의 균형을 유지하는 데 유용합니다.

- **Performance Highlights**: MIND는 이진 및 다중 레이블 임상 예측 작업에서 시간 시계열 데이터와 흉부 X-ray 이미지에 대해 평가되었습니다. 또한, 비의료 멀티모달 다중 클래스 데이터셋에서도 MIND 프레임워크의 일반화 가능성을 검증하였습니다. 실험 결과, MIND는 다섯 가지 작업에서 작은 멀티모달 네트워크의 성능을 개선하고, 다양한 융합(fusion) 방법 및 멀티모달 아키텍처와 비교했을 때 최첨단 기준선을 초과하는 성과를 보였습니다.



### From Public Square to Echo Chamber: The Fragmentation of Online Discours (https://arxiv.org/abs/2501.18441)
Comments:
          6 pages, 7 figures, 1 table

- **What's New**: 이 논문은 소셜 미디어 알고리즘과 필터 버블(filter bubble)이 온라인 담론(fragmented discourse)의 단편화를 어떻게 초래하는지를 다룹니다. 이는 이념적 불화(ideological divides)를 조장하고 공유된 이해(shared understanding)를 약화시키는 결과를 초래합니다. 논문은 사회적 긴장이 고조된 시기에 성차별(sexism), 인종차별(racism), 외국인 혐오(xenophobia) 등 차별 담론(discrimination discourse)이 어떻게 증폭되는지를 탐구합니다.

- **Technical Details**: 마이클 샌델(Michael Sandel)의 공동체와 공유된 가치에 대한 철학적 강조를 바탕으로, 이 연구는 디지털 플랫폼이 현실 세계의 사건에 대한 반응으로 담론 조각이 형성되고 진화하는 메커니즘을 분석합니다. 디지털 커뮤니티의 역학(dynamics of digital communities)을 통해, 논문은 담론 단편화(fragmentation)와 사회적 불화(polarization)를 가속화하는 사회적 구조를 강조합니다.

- **Performance Highlights**: 이 연구는 소셜 미디어 구조가 어떻게 대화(dialogue)를 제한하고 공정한 사회에 필수적인 집단적 추론(collective reasoning)을 침식하는지를 보여줍니다. 또한, 디지털 시대에 담론의 단편화가 초래하는 도전(challenges)을 고찰하며, 사회적 상호작용(interactions)의 전산적 분석(computational analysis)과 철학적 관점을 통합하여 심층적인 이해를 제공합니다.



### How to Build a Quantum Supercomputer: Scaling from Hundreds to Millions of Qubits (https://arxiv.org/abs/2411.10406)
Comments:
          76 pages, 46 figures. General revision, added figures, added references, added appendices

- **What's New**: 이번 논문에서는 양자 컴퓨팅 분야에서 기술적으로 해결되지 않거나 간과된 주요 문제들을 다루고 있습니다. 특히, 초전도 큐비트(superconducting qubits)를 기반으로 한 유틸리티 규모의 양자 컴퓨터 개발을 위한 경로를 제시하고 있습니다. 기존 반도체 기술을 활용하여 더 높은 품질의 큐비트를 구축하고, 시스템 엔지니어링(system engineering) 접근법을 채택해야 한다고 강조합니다.

- **Technical Details**: 이 연구에서는 양자 오류 수정(Quantum Error Correction, QEC)의 필요성과 함께 고성능 컴퓨팅(HPC) 환경에서의 양자 컴퓨터의 통합 문제를 다룹니다. 특히, 양자 프로세서의 오류율을 줄이고, 고성능 양자 컴퓨터를 설계하기 위한 다양한 기술적 접근 방식을 모색합니다. 이를 통해 양자 회로가 더 효율적으로 작동하도록 하며, 물리적인 큐비트 간의 상호작용을 조정할 수 있는 방법에 대해 설명합니다.

- **Performance Highlights**: 실험적으로, 논리적 오류율(logical error rates)은 10^(-10) 수준에 도달했으며, 이는 양자 컴퓨터의 성능 향상 가능성을 보여줍니다. 또한, 기존 양자 컴퓨팅의 하드웨어 아키텍처와 시스템 디자인의 차이로 인해 발생하는 여러 문제를 해결하기 위한 체계적인 접근법이 필요합니다. 마지막으로, 이 연구는 유틸리티 규모의 양자 컴퓨팅 실현을 위한 중요한 과제가 오류 수정 및 하이브리드 양자-고전적(complementary quantum-classical) 시스템의 통합임을 강조합니다.



### TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues (https://arxiv.org/abs/2502.01630)
- **What's New**: 이 논문에서는 다중 세션 대화에서의 시간적 추론(temporal reasoning)에 대한 새로운 평가 작업을 제안하며, LoCoMo 데이터셋을 활용하여 새로운 벤치마크를 생성하는 방법을 소개합니다. 또한, TReMu라는 새로운 프레임워크를 통해 LLM 에이전트의 시간적 추론 능력을 향상시키는 접근 방식을 설명합니다. 이 프레임워크는 시간 인식 메모이제이션(time-aware memorization)과 신경-기호적 시간적 추론(neuro-symbolic temporal reasoning)을 통합합니다.

- **Technical Details**: 프레임워크의 핵심은 타임라인 요약(timeline summarization) 및 Python 코드를 생성하여 시간 계산을 수행하는 신경-기호적 접근 방식입니다. TReMu는 대화 세션 내 사건을 요약하고 그 사건과 추론된 날짜를 연관 지어 메모리를 생성합니다. 이러한 모델은 상대적 시간 표현을 명확히 하여 LLM이 다중 세션 간 이벤트를 효과적으로 추론하도록 돕습니다.

- **Performance Highlights**: 실험 결과, 제안된 벤치마크는 LLM의 시간적 추론 성능에서 기존 방법에 비해 큰 개선을 보여주었으며, GPT-4o의 기준점에서 29.83에서 77.67로 상승했습니다. 이는 다중 세션 대화의 맥락에서 시간적 추론의 도전 과제를 해결하는 데 효과적인 방안을 제시합니다. 따라서 이 연구는 LLM의 시간적 추론 능력을 향상시키기 위한 중요한 기여를 하고 있습니다.



### PhD Knowledge Not Required: A Reasoning Challenge for Large Language Models (https://arxiv.org/abs/2502.01584)
- **What's New**: 이번 연구는 NPR Sunday Puzzle Challenge에 기반한 새로운 벤치마크를 제시합니다. 기존 벤치마크는 전문적 지식을 요구하여 비전문가가 이해하기 어려운 반면, 이 벤치마크는 일반적인 지식만으로도 도전해볼 수 있습니다. 또한, OpenAI o1 모델이 기존의 전문 지식을 평가하는 벤치마크에서 우수한 성능을 보이는 것과 더불어 새로운 유형의 실패 사례를 발견했습니다.

- **Technical Details**: 본 연구는 약 600개의 문제를 포함한 머신 체크 가능한 벤치마크를 개발하였으며, NPR Sunday Puzzle Challenge에서 요구되는 문제를 선별하여 구성하였습니다. 연구자들은 가장 최신 언어 모델들이 이 벤치마크에서 도전하고 성공적인 결과를 내는 것에 주목하였고, DeepSeek R1과 같은 모델은 '생각하는 데에 영원히 갇히는' 경향도 보여주었습니다. 'I give up'과 같은 응답도 관찰되었으며, 이로 인해 연구자들은 추론 시간이 지나기 전에 마무리하는 기법의 필요성을 제기했습니다.

- **Performance Highlights**: OpenAI o1 모델은 59%의 높은 정확도로 다른 모델들과 비교했을 때 발군의 성능을 보였습니다. 반면 DeepSeek R1은 종종 잘못된 답변을 줄 때 '포기'하는 경향이 있었고, 모델의 출력을 분석한 결과 이와 같은 현상들이 발견되었습니다. 또한, DeepSeek R1과 Gemini Thinking을 통해 텍스트 추론 단계를 기록함으로써 더욱 심도 있는 분석이 가능해졌습니다.



### Sea-cret Agents: Maritime Abduction for Region Generation to Expose Dark Vessel Trajectories (https://arxiv.org/abs/2502.01503)
Comments:
          Accepted to 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025)

- **What's New**: 본 연구에서는 자동 식별 시스템(AIS)을 비활성화하여 불법 활동을 수행하는 '어두운 선박(dark vessels)'의 위치를 파악하기 위해 후행 추론(abductive inference) 개념을 도입합니다. 우리는 특정 부분 경로를 바탕으로 이러한 선박의 위치를 효과적으로 찾아낼 수 있는 새로운 방법론을 제안하고, 전통적인 기계 학습 방법보다 더 적은 탐색 영역으로 두 배 이상의 성능을 발휘함을 입증하였습니다. 이 연구는 해양 안전과 보안 분야에서 중요한 새로운 접근 방식을 제공하며, 실질적인 분석 작업에 적용할 수 있는 가능성을 제시합니다.

- **Technical Details**: 연구는 부정확한 예측에 의존하지 않고, 후행 추론, 논리 프로그래밍(logic programming), 규칙 학습(rule learning)을 접목하여 '어두운 선박'의 위치 추론 문제를 해결합니다. 제안된 방법은 30㎢의 지역에서 기존 기계 학습 기준보다 157% 더 높은 리콜(recall) 성능을 보였으며, 이는 물리적 자원과 탐색 영역이 증가할수록 성능이 개선된다는 장점을 가지고 있습니다. 데이터 효율성 또한 높아, 단일 훈련 궤적을 가지고도 유사한 성능을 발휘할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 기존 기계 학습 모델 대비 476% 향상된 리콜 성능을 보여주었으며, 이는 추가 자원을 활용했을 때 더욱 두드러졌습니다. 또한 제안된 시스템은 운영 플랫폼에 실제 배치되어 노출되지 않은 선박의 식별을 지원할 수 있는 가능성을 지니고 있습니다. 본 연구는 해양 분석가들이 비정상적인 활동을 실시간으로 감지하는 데 유용한 도구가 될 것으로 기대됩니다.



### Develop AI Agents for System Engineering in Factorio (https://arxiv.org/abs/2502.01492)
- **What's New**: 이 논문은 AI 에이전트의 시스템 엔지니어링(SYSTEM ENGINEERING) 능력을 평가하고 훈련하기 위한 새로운 접근 방법으로 자동화 지향 샌드박스 게임 특히 'Factorio'의 활용을 제안합니다. 기존의 정적 벤치마크는 동적 시스템 구현에 필요한 기술을 포착하는 데 한계가 있어, 이러한 샌드박스 환경이 더 적합하다고 주장합니다. 또한, 다가오는 복잡한 공학 프로젝트를 설계하고 최적화하기 위해 AI 에이전트의 전문적 추론 및 장기 계획 능력을 배양할 필요가 강조됩니다.

- **Technical Details**: 시스템 엔지니어링은 하드웨어, 소프트웨어 및 인간 프로세스를 통합한 대규모 시스템의 설계, 구현 및 관리와 관련됩니다. 복잡한 시스템 개발 및 AI 에이전트가 중요해짐에 따라, 더욱 정교한 시스템 레벨 작업에서 AI를 활용할 기회가 확장되고 있습니다. 또한, AI 모델은 최근 다양한 영역에서 슈퍼휴먼(superhuman) 능력을 보여주며, 이는 공학 및 설계 분야에서도 큰 잠재력을 지니고 있습니다.

- **Performance Highlights**: AI 에이전트는 현재의 한계에도 불구하고, 복잡한 시스템 문제를 해결하기 위한 장기적인 목표를 향해 나아가고 있습니다. 인공지능이 시스템 엔지니어링에서 높은 성취를 보일 경우, 청정 에너지, 안정적인 수자원 및 식량 공급의 확장과 같은 현대 문명의 도전을 해결하는 데 큰 도움이 될 것입니다. 또한, 최적화된 시스템 설계와 저수준 기계 작업을 결합하여 복잡한 시스템을 위한 완전한 엔지니어링 역량을 제공할 가능성이 높습니다.



### TeLL-Drive: Enhancing Autonomous Driving with Teacher LLM-Guided Deep Reinforcement Learning (https://arxiv.org/abs/2502.01387)
- **What's New**: 본 논문에서는 TeLL-Drive라는 새로운 하이브리드 프레임워크를 제안합니다. 이 프레임워크는 Teacher LLM과 Student DRL 정책 모형을 통합하여 자율 주행에서의 의사결정을 개선하고자 합니다. LLM은 리스크 메트릭스, 역사적 시나리오 검색, 도메인 휴리스틱을 포함한 상황 중심 프롬프트를 통해 높은 수준의 운전 전략을 제공합니다.

- **Technical Details**: TeLL-Drive는 Teacher LLM이 학생 역할을 하는 DRL 에이전트와 협력하여 결정 과정을 효율화합니다. 이 과정에서 LLM은 메모리, 반성 및 추론 능력을 갖춘 리스크 인지 LLM 에이전트를 통해 복잡한 교통 환경에서 효율적이고 안전한 의사결정을 지원합니다. DRL 에이전트는 Actor-Critic 아키텍처를 활용하며, 혼합 전략을 사용하여 탐색 기능을 강화하고 샘플링 효율성을 높입니다.

- **Performance Highlights**: 실험 결과, TeLL-Drive는 다른 기존 방법들과 비교하여 성공률, 평균 수익, 실시간 실행 가능성 측면에서 우수한 성능을 보였습니다. 특히, 주의 메커니즘과 LLM 기반 가이드 간의 시너지가 중요한 역할을 하는 것으로 나타났습니다. 이러한 결과는 TeLL-Drive가 자율 주행 시스템의 적응성과 안전성을 크게 향상시킬 수 있음을 보여줍니다.



### PSSD: Making Large Language Models Self-denial via Human Psyche Structur (https://arxiv.org/abs/2502.01344)
Comments:
          WWW '25

- **What's New**: 이번 논문에서는 LLM(대형언어모델)의 추론 정확성을 개선하기 위한 새로운 방법인 PSSD(자기부정 구조)를 제안합니다. 이 접근법은 인간 사고 구조를 모방하여 세 가지 연결된 역할(직관 기반의 id, 규칙 기반의 superego, 스크립트 중심의 ego)을 통해 LLM의 내부 잠재력을 극대화합니다. PSSD는 LLM이 스스로 실수를 인식하고 이를 수정하는 과정을 보다 유기적으로 구성하여 더 나은 성능을 목표로 합니다.

- **Technical Details**: PSSD는 Freudian(프로이트) 이론을 바탕으로 하여, 각 역할이 상호작용하여 정교한 결과를 도출하도록 설계되었습니다. id 역할은 LLM의 직관적 시도를 통해 다양한 추론 경로를 생성하고, superego 역할은 이러한 시도를 규제하기 위한 규칙을 제공하며, ego 역할은 이를 종합하여 실행 가능한 스크립트를 생성합니다. 이 방식은 기존 LLM의 구조와 통합할 수 있어, 타 기법에 비해 효율성과 유연성을 가지고 있습니다.

- **Performance Highlights**: 실험 결과, PSSD는 기존의 방법들보다 뛰어난 성능을 나타냈으며, 특히 LLM의 자원 소모 문제를 해결하는 데 효과적임을 보여주었습니다. 이 방법은 LLM이 자가 교정 기능을 발전시키고, 동시에 고품질의 결과를 신속하게 도출할 수 있도록 지원합니다. 결국, 이 연구는 LLM의 내부 잠재력을 활용하는 새로운 방향성을 제시하며, 추론 능력 개선에 중요한 기여를 하고 있습니다.



### Explainability-Driven Quality Assessment for Rule-Based Systems (https://arxiv.org/abs/2502.01253)
- **What's New**: 이 논문은 데이터 세트 기반 통찰력을 바탕으로 지식 기반 추론 시스템의 규칙 품질을 향상시키기 위해 설계된 설명 프레임워크를 소개합니다. 전통적인 규칙 유도 방법은 일반적으로 노동 집약적인 레이블링과 데이터 주도 학습을 요구하지만, 본 프레임워크는 기존 규칙의 데이터 주도 정제를 가능하게 합니다. 이를 통해 규칙 추론의 설명을 생성하고, 인간의 해석을 활용하여 규칙을 정제할 수 있습니다.

- **Technical Details**: 설명 프레임워크는 네 가지 상호 보완적인 설명 유형(방문 기반(trace-based), 맥락적(contextual), 대조적(contrastive), 그리고 반사적(counterfactual))을 활용하여 디버깅, 검증 및 최종적으로 규칙 정제를 위한 다양한 관점을 제공합니다. 이 프레임워크는 MIT App Inventor 플랫폼의 일종인 Punya에 통합되어 저코드(low-code) 환경에서 규칙 기반 추론을 지원하며, 기술 및 비기술 지식 엔지니어 모두에게 유용합니다.

- **Performance Highlights**: 본 연구는 특히 금융 분야에서 사용 사례를 통해 실용성을 사례 연구로 입증하고 있습니다. 설명 메커니즘의 통합을 통해 규칙의 투명성을 높이고, 사용자들이 더욱 신뢰할 수 있는 의사 결정을 내릴 수 있도록 돕습니다. 또한, 설명 중심의 규칙 정제 방법론을 통해 지식 기반 시스템에서 규칙의 품질을 향상시키고 있습니다.



### Efficient rule induction by ignoring pointless rules (https://arxiv.org/abs/2502.01232)
Comments:
          Under review for a conference

- **What's New**: 이 논문에서는 inductive logic programming (ILP)의 새로운 접근 방식이 소개되었습니다. 이 방법은 불필요한 규칙(pointless rules)을 식별하는 데 중점을 두고 있습니다. 불필요한 규칙은 중복된 리터럴(redundant literal)을 포함하거나 부정적인 예제(negative examples)와 구별할 수 없는 규칙입니다.

- **Technical Details**: 연구 결과, 불필요한 규칙을 무시하면 ILP 시스템이 가설 공간(hypothesis space)을 안정적으로 축소(prune)할 수 있다는 것을 보여주었습니다. 실험은 다양한 도메인에서 진행되었으며, 이는 비주얼 추론(visual reasoning) 및 게임 플레이(game playing) 같은 분야를 포함합니다.

- **Performance Highlights**: 이 접근 방식은 학습 시간을 99%까지 단축하면서도 예측 정확도(predictive accuracies)를 유지할 수 있음을 나타냈습니다. 따라서 ILP의 효율성을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### Skewed Memorization in Large Language Models: Quantification and Decomposition (https://arxiv.org/abs/2502.01187)
- **What's New**: 이 논문은 대형 언어 모델(LLM)의 기억화(memorization) 문제를 다뤄 개인정보 및 보안 리스크를 조명합니다. 저자들은 기존의 연구들이 평균 사례에 집중하고, 기억화의 skewed (비대칭적) 분포를 간과하고 있음을 지적합니다. 이를 통해 LLM의 감독적 세밀 조정(SFT) 시 기억화 확률을 분석하고, 데이터셋 크기 및 훈련 기간과의 관계를 탐구합니다. 또한, 모델의 데이터 생성 과정에서 기억화를 추정하고 기존 메트릭과 비교하는 통찰력을 제시합니다.

- **Technical Details**: 이 연구는 비모수적 통계 테스트를 활용하여 LLM의 기억화 분석을 수행합니다. 특히, 모델의 기억화가 훈련 기간의 증가에 따라 어떻게 증가하고, 데이터셋 구성이나 크기 변화에 따라 어떻게 달라지는지를 보여줍니다. 저자들은 모델의 생성 과정에서 용어별 확률을 분해하여 데이터의 특성(예: 유사성 격차 및 지역 데이터 밀도)이 기억화 가능성에 미치는 영향을 설명합니다. 이외에도, 이들은 전통적인 텍스트 유사도 측정 방법(Rouge 및 Levenshtein distance)에 비해 그들의 메트릭의 우수성을 강조합니다.

- **Performance Highlights**: 실험 결과, 훈련 에포크가 증가함에 따라 전체 손실이 감소하는 것에도 불구하고 기억화가 증가함을 보였습니다. 부적합(overfitting)이나 높은 지역 엔트로피로 인해 극단적인 기억화 경향이 발생한다는 점을 강조하고, 데이터셋의 구성이나 크기 변화가 기억화 패턴에 미치는 중대한 영향을 보여줍니다. 이러한 분석은 LLM의 기억화 행동을 이해하고, 이를 탐지 및 완화할 전략을 제시함으로써 개인정보 보호를 더욱 강화할 수 있는 방법을 제공합니다.



### Scalable Precise Computation of Shannon Entropy (https://arxiv.org/abs/2502.01160)
Comments:
          9 pages, 3 figures

- **What's New**: 이 논문에서는 프로그램의 양적 정보 흐름(Quantitative Information Flow, QIF)을 정밀하게 계산하기 위한 새로운 도구인 Precise Shannon Entropy (PSE)를 제안합니다. PSE는 Shannon 엔트로피 계산의 두 단계를 최적화하여 확장 가능성과 정확성을 높입니다. 첫 번째 단계에서는 
ADDAND라는 지식 컴파일 언어를 설계하여 가능한 출력의 열거를 피하고, 두 번째 단계에서는 모델 카운팅 쿼리의 최적화를 통해 출력 확률을 계산합니다.

- **Technical Details**: QIF는 프로그램의 비밀 정보 누출 정도를 측정하기 위한 방법으로, Shannon 엔트로피를 사용하여 누출 정도를 수치화합니다. 모든 프로그램의 관계를 나타내는 불 대수 공식을 활용하며, PSE는 가능성 있는 출력을 열거하지 않고도 엔트로피 계산을 지원합니다. 이 과정에서 Algebraic Decision Diagrams (ADD)와 conjunctive decomposition를 결합하여 효율적인 측정 방법을 제공합니다.

- **Performance Highlights**: PSE는 441개의 벤치마크 중 기존의 EntropyEstimation 툴보다 55개를 더 해결하여 뛰어난 성능을 보여주었습니다. 두 도구가 해결한 벤치마크 중 98%에서 PSE는 EntropyEstimation보다 최소 10배 이상의 효율성을 보였습니다. 이러한 결과는 PSE가 정확하면서도 확장 가능한 Shannon 엔트로피 계산 툴임을 입증합니다.



### DeepRAG: Thinking to Retrieval Step by Step for Large Language Models (https://arxiv.org/abs/2502.01142)
- **What's New**: DeepRAG라는 새로운 프레임워크가 소개되었습니다. 이 프레임워크는 Retrieval-Augmented Generation(RAG)을 Markov Decision Process(MDP)로 모델링하여 전략적이고 적응적인 정보 검색을 가능하게 합니다. DeepRAG는 질의를 반복적으로 분해함으로써 외부 지식을 검색할지 파라메트릭(reasoning) 지식에 의존할지를 동적으로 결정합니다. 실험 결과, DeepRAG는 기존 시스템에 비해 21.99% 더 높은 정확도로 답변을 제공하며 검색 효율성도 향상되었습니다.

- **Technical Details**: DeepRAG는 세 가지 핵심 단계로 구성됩니다: 1) Binary Tree Search를 통해 각 서브쿼리와 관련된 경로를 탐색합니다. 2) Imitation Learning을 활용하여 최소 검색 비용으로 올바른 답변에 도달하는 추론 과정을 학습합니다. 3) Chain of Calibration을 통해 LLM의 내부 지식을 조정하여 원활한 지식 경계 인식을 지원합니다. 각 단계는 MDP의 상태, 행동, 전이 역학, 보상 기능을 정의하여 체계적으로 구성됩니다.

- **Performance Highlights**: DeepRAG는 다섯 개의 오픈 도메인 QA 데이터셋에서 실험되어 그 효과가 검증되었습니다. HotpotQA와 같은 multi-hop factual QA에서 상향된 성능을 보였으며, CAG와 같은 시계열 QA에서도 우수한 결과를 나타냈습니다. 추가 분석을 통해 DeepRAG가 검색 결정과 파라메트릭 지식 간에 더 강한 상관관계를 나타내며, 이는 보다 효과적인 지식 경계 조정으로 이어집니다.



### Picky LLMs and Unreliable RMs: An Empirical Study on Safety Alignment after Instruction Tuning (https://arxiv.org/abs/2502.01116)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 안전 정렬(safety alignment) 저하 문제를 다룹니다. 특히, 일반적인 세팅에서 수행되는 미세 조정(fine-tuning)이 어떻게 모델의 안전성을 저하시킬 수 있는지 체계적으로 분석하였습니다. 일반적인 데이터셋을 사용하더라도 이러한 저하가 발생할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 답변 구조(answer structure), 정체성 보정(identity calibration), 역할 수행(role-play)의 세 가지 주요 요인이 LLM의 안전 정렬에 미치는 영향을 파악했습니다. 또한, 가장 최신의 보상 모델(reward models, RMs)의 신뢰성도 평가하여, 인간의 안전성 선호를 정확히 반영하지 못하는 경우가 많음을 발견했습니다.

- **Performance Highlights**: 이 연구는 LLM의 미세 조정 과정에서 안전 정렬을 유지하는 것이 얼마나 복잡한지를 강조합니다. 개발자들이 유용성과 안전성을 조화롭게 조율할 수 있는 방법에 대한 지침을 제공합니다. 실험에 사용된 데이터셋과 미세 조정 코드는 제공된 URL에서 확인할 수 있습니다.



### ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning (https://arxiv.org/abs/2502.01100)
Comments:
          Website: this https URL

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 논리적 추론 능력과 복잡한 비단조 추론에서의 확장성을 조사합니다. ZebraLogic이라는 포괄적인 평가 프레임워크를 도입하여 LLM의 추론 성능을 논리 그리드 퍼즐을 통해 평가합니다. 이를 통해 문제의 복잡성을 조절하고 정량화할 수 있게 하여, LLM의 성능 한계를 체계적으로 연구할 수 있는 기회를 제공합니다.

- **Technical Details**: ZebraLogic은 다양한 복잡성을 둔 논리 그리드 퍼즐을 생성할 수 있는 프레임워크로, 이를 통해 LLM이 논리적 제약을 얼마나 잘 준수하는지를 평가합니다. 여기서 제약 만족 문제(CSPs)는 수학적으로 정의되고 확장 가능하여, 모델의 아키텍처나 크기와 관계없이 LLM의 논리 추론 능력을 평가하는 데 유용합니다. 논리 퍼즐은 특정 특성과 값으로 이루어진 집합을 기반으로 하며, 각 퍼즐은 K개의 단서에 대한 고유한 값을 찾기 위한 논리적 추론을 요구합니다.

- **Performance Highlights**: 결과적으로 퍼즐의 복잡성이 증가함에 따라 LLM의 정확도가 극심하게 감소하는 "complexity curse"가 발견되었습니다. 이 현상은 LLM의 크기를 키우거나 추론 시간 계산을 늘리더라도 지속되며, 이는 현재 LLM의 추론 능력이 고유한 제한을 지니고 있음을 나타냅니다. 최적의 추론 토큰 대 Z3 충돌 비율이 존재하나, o1 같은 모델은 복잡성이 매우 높을 경우 이 비율을 항상 달성하지 못한다는 점도 강조됩니다.



### Language Models Use Trigonometry to Do Addition (https://arxiv.org/abs/2502.00873)
- **What's New**: 이 연구는 세 가지 중형 LLM(GPT-J, Pythia-6.9B, Llama3.1-8B)이 더하기 문제를 어떻게 계산하는지를 역설계하여 이해하고자 합니다. 특히, 숫자가 일반화된 나선으로 표현되고 이 나선을 조작하여 덧셈을 수행한다는 점을 새롭게 발견했습니다. 이를 통해 LLM의 수학적 능력에 대한 첫 번째 표현 수준의 설명을 제공합니다.

- **Technical Details**: LLM들은 'Clock' 알고리즘을 활용하여 덧셈을 수행하며, 이로 인해 a와 b를 나타내는 나선을 조작해 a+b를 생성합니다. 또한, MLP, attention head 및 개별 뉴런의 사전 활성화 상태를 나선으로 모델링하여 구체적인 계산 과정을 분석합니다. 연구는 기계적 해석 가능성(mechanistic interpretability, MI)의 맥락에서 진행되며, LLM의 구조적 기능을 규명하는 데 기여합니다.

- **Performance Highlights**: 모델의 성능을 평가한 결과, 특정 LLM이 덧셈 작업을 효과적으로 수행할 수 있음을 발견하였으며, 그 과정에서 'Clock' 알고리즘이 어떻게 활용되는지에 대한 이해를 높였습니다. 본 연구는 LLM이 수치 데이터를 나선으로 나타내고 이를 조작하여 수학적 문제를 해결하는 방식을 분명히 밝히는 중요한 단계를 제시합니다.



### Learning to Plan with Personalized Preferences (https://arxiv.org/abs/2502.00858)
- **What's New**: 이 논문은 개인의 선호를 학습하고 이를 기반으로 계획 수립을 조정할 수 있는 AI 에이전트를 개발하는 데 중점을 두고 있습니다. 최근의 연구들은 일반화된 접근 방식을 채택하고 있으며, 이는 개인의 선호를 간과하는 경향이 있습니다. 본 연구는 Preference-based Planning (PbP) 벤치마크를 도입하여 다양한 선호를 시스템적으로 평가하고, 이를 통해 맞춤형 계획 수립을 위한 새로운 방향을 제시합니다. 이러한 방식으로 AI는 사용자 개인의 특성과 요구에 더 잘 부합할 수 있게 됩니다.

- **Technical Details**: PbP 벤치마크는 NVIDIA Omniverse와 OmniGibson을 기반으로 하여 구축되었으며, 50개의 다양한 환경에서 수천 가지의 일상 활동을 사실적으로 시뮬레이션합니다. 이 벤치마크에서는 각각 다른 290개의 선호를 파라미터화한 어휘로 사용하여, 세부 행동 선호부터 작업 순서와 같은 시퀀스 수준의 선호까지 다양한 측면을 포괄합니다. 또한, 논문은 개인의 선호를 학습하기 위해 few-shot learning 방식을 적용하며, 이는 불확실한 명령어에 대해 사용자의 선호에 맞춘 계획을 수립하는 것을 포함합니다.

- **Performance Highlights**: PbP 벤치마크를 이용한 기존 학습 에이전트에 대한 평가 결과, 사용자의 선호는 행동 패턴의valuable abstraction으로 작용하며, 이를 중간 계획 단계로 통합하는 것이 에이전트의 적응 능력을 크게 향상시킵니다. 실험을 통해 상징 기반 접근법이 확장성에서 가능성을 보여준 반면, 개인화된 계획을 생성하고 실행하는 데 여전히 큰 도전 과제가 남아 있음을 발견하였습니다. 본 연구는 이러한 개인화된 에이전트 개발을 위한 기초 작업으로 자리잡고 있습니다.



### Psychometric-Based Evaluation for Theorem Proving with Large Language Models (https://arxiv.org/abs/2502.00855)
- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 정리 증명 능력을 평가하기 위한 새로운 방법을 제안합니다. 기존의 평가 방식은 단순한 증명 통과율에 의존하고 있어, 정리의 중요성에 따라 성능 차이를 반영하지 못했습니다. 따라서 이 연구는 심리측정학(psychometrics)을 기반으로 하는 새로운 평가 방법인 Dataset Annotation과 Adaptive Evaluation을 통해 보다 세분화된 평가를 가능하게 합니다.

- **Technical Details**: 제안된 방법은 첫째, miniF2F 데이터세트의 각 정리에 난이도와 변별력 지표를 적용하여 주석을 다는 과정을 포함합니다. 이를 통해 새롭게 생성된 miniF2F-Graded 데이터셋은 정리의 어려움을 LLM이 어떻게 인식하는지를 보다 정확하게 반영합니다. 둘째, Adaptive Evaluation 방법을 통해 실시간 모델 성능에 맞춰 가장 적합한 정리를 동적으로 선택하여 LLM의 증명 능력을 효율적으로 평가합니다.

- **Performance Highlights**: 이 연구의 실험 결과는 10개의 LLM의 성능 차이를 명확하게 구분하여 보여줍니다. 특히, 데이터세트의 23%의 정리만을 사용함으로써 평가 비용을 크게 줄일 수 있었습니다. 이는 LLM의 증명 능력 평가에 있어 시간과 자원 효율성을 높이는 데 기여합니다.



### RTBAgent: A LLM-based Agent System for Real-Time Bidding (https://arxiv.org/abs/2502.00792)
Comments:
          Accepted by WWW 2025

- **What's New**: 본 논문은 대형 언어 모델(LLM)을 기반으로 하는 최초의 실시간 입찰(RTB) 에이전트 시스템인 RTBAgent를 제안합니다. RTBAgent는 실제 경쟁 광고 입찰 환경을 동기화하고 통합된 의사 결정 프로세스를 통해 입찰 가격을 얻습니다. 이 시스템은 클릭률 추정 모델, 전문 전략 지식 및 일일 반영과 같은 보조 모듈을 통해 RTB에 특화되어 있습니다.

- **Technical Details**: RTBAgent는 두 단계의 의사 결정 프로세스와 다중 메모리 검색 메커니즘을 포함하고 있으며, 이를 통해 과거의 결정 및 거래 기록을 검토하고 시장의 변화에 실시간으로 적응할 수 있습니다. 이 시스템은 사용자 행동에 기반하여 개별 사용자를 정밀하게 타겟팅하는 기능을 갖추고 있으며, 이를 통해 광고 노출의 가치를 분석하고 최적의 입찰 가격을 결정합니다.

- **Performance Highlights**: RTBAgent는 실제 광고 데이터셋에 대한 실증 테스트를 통해 수익성을 크게 향상시키는 것으로 나타났습니다. 또한, RTBAgent는 전통적인 방법에 비해 해석 가능성이 뛰어난 장점을 가지고 있습니다. 이로 인해 광고주들은 더욱 전략적이고 정보에 기반한 결정을 내릴 수 있게 되어 경쟁이 치열한 시장에서 투자 수익률을 더욱 향상시킬 수 있습니다.



### Zero-Shot Warning Generation for Misinformative Multimodal Conten (https://arxiv.org/abs/2502.00752)
- **What's New**: 본 연구는 잘못된 맥락(misinformation)에서 정보를 추출하고 분석하는 모델을 제안합니다. 이 모델은 cross-modality consistency check를 통해 다양한 데이터 모달리티에서 정보를 검증하며, 훈련 시간이 적게 소요되는 특징이 있습니다. 또한, 자동으로 경고를 생성할 수 있는 새로운 zero-shot learning 작업을 도입하여 사용자 이해를 증진시킬 수 있는 방법을 탐색합니다.

- **Technical Details**: 제안된 모델은 (이미지, 캡션) 쌍의 진위를 평가하는 유연한 아키텍처를 갖추고 있으며, 이를 통해 87.04%의 정확도를 기록했습니다. 증거 수집(evidence retrieval), 일관성 검사(consistency check), 경고 생성(warning generation)이라는 세 단계를 포함한 파이프라인을 통해 정확한 검증을 이루어냅니다. 모델은 Visual Language Model(VLM)인 MiniGPT-4를 활용하여 증거의 홈페이지를 분석하고, 관련 정보를 바탕으로 사용자에게 경고 또는 설명을 제공합니다.

- **Performance Highlights**: 본 연구는 경량화된 모델이 전체 모델 활용 시에는 87.04%의 정확도를, 경량 모델 활용 시에는 84.78%의 정확도를 달성해 경쟁력을 검증했습니다. 질적 평가와 인간 평가를 통해 생성된 경고의 잠재력과 한계를 확인했습니다. 이를 통해 잘못된 정보의 추적 및 반박 과정에서 중요한 이해를 향상시킬 수 있음을 입증합니다.



### Selective Response Strategies for GenAI (https://arxiv.org/abs/2502.00729)
- **What's New**: 이 논문은 최근 생성적 인공지능(Generative AI, GenAI)이 Stack Overflow와 같은 인간 기반 포럼에 미친 영향을 분석하며, ‘선택적 응답(selective response)’이라는 새로운 전략을 제안합니다. 이 전략은 GenAI가 진화하는 주제나 신기술을 다룰 때 부정확하거나 보수적인 응답을 제공하여 사용자로 하여금 인간 기반 포럼을 활용하게 만드는 것입니다. 이어지는 연구는 이러한 전략이 GenAI의 수익성과 사용자 복지에 긍정적인 장기적 영향을 미칠 수 있음을 보여줍니다.

- **Technical Details**: 논문에서는 선택적 응답이 항상 응답하는 접근방식 대비 사용자 복지와 GenAI의 수익을 동시에 개선할 수 있음을 설명합니다. 선택적 응답을 통해 GenAI가 사용자에게 정보를 전략적으로 관리하고, 이를 통해 시스템 내에서 사용자 행동에 영향을 줄 수 있는 과정을 설명합니다. 또한, 이 논문은 GenAI와 포럼이라는 두 플랫폼 간의 경쟁을 다루며, 게임 이론적 관점에서 콘텐츠 생성, 복지, 수익의 동적 관계를 모형화합니다.

- **Performance Highlights**: 선택적 응답 전략을 통해 GenAI는 사용자들 사이에서 더 높은 선택 비율과 데이터 생성을 유도할 수 있습니다. 이러한 상황에서, GenAI는 약 0(O(εT²))의 근사치로 최적 수익을 달성할 수 있는 알고리즘을 개발하였으며, 이를 통해 사회 복지 기준을 충족하는 동시에 수익을 극대화하는 방안도 제시하였습니다. 전반적으로 이 논문은 GenAI가 항상 응답해야 한다는 기존의 관념에 도전하며, 선택적 응답은 GenAI 플랫폼과 사용자 모두에게 이점을 제공할 수 있음을 보여줍니다.



### Perspectives for Direct Interpretability in Multi-Agent Deep Reinforcement Learning (https://arxiv.org/abs/2502.00726)
- **What's New**: 이번 논문은 Multi-Agent Deep Reinforcement Learning (MADRL)의 모델 해석 가능성에 대한 새로운 접근법을 제안합니다. 기존의 방법들은 복잡한 시스템을 올바르게 해석하기 어려웠으나, 본 연구에서는 훈련된 모델에서 직접적으로 설명을 생성하는 post hoc 방법을 도입하여 이 문제를 해결할 수 있는 가능성을 제시합니다. 이러한 접근법을 통해 에이전트의 행동, 발생하는 현상, 그리고 편향에 대한 통찰을 제공하면서도 모델 아키텍처를 변경하지 않고도 해석할 수 있는 장점이 있습니다.

- **Technical Details**: MADRL 시스템은 에이전트, 환경 및 훈련 알고리즘이라는 세 가지 구성 요소로 이루어져 있습니다. 에이전트는 각 시간 단계에서 관찰을 기반으로 행동을 생성하며, 통신 채널이 있을 경우 이를 통해 서로 소통할 수 있습니다. 훈련 알고리즘은 중앙집중식, 분산식 또는 하이브리드 방식으로 동작하며, DNN(Deep Neural Networks)을 기반으로 한 에이전트가 훈련됩니다. 각 에이전트는 보상, 상태, 행동 등의 요소에 따라 손실 함수를 최소화하는 방향으로 훈련을 받습니다.

- **Performance Highlights**: 해석 가능한 모델 개발을 위한 기존의 노력들은 주로 본질적으로 해석 가능한 모델에 중점을 두고 있었지만, 이 논문은 복잡한 시스템에서도 직접 해석 가능성을 강조합니다. 본 연구는 MADRL 분야의 특정 도전 과제를 해결하는 데에 해석 가능성이 기여할 수 있는 방안을 모색하고 있습니다. 또한, 팀 식별, 군집 조정 및 샘플 효율성과 같은 현대적인 MADRL 과제를 해결하기 위한 직접 해석 가능성의 응용처를 제시하고 있습니다.



### MM-IQ: Benchmarking Human-Like Abstraction and Reasoning in Multimodal Models (https://arxiv.org/abs/2502.00698)
- **What's New**: 이 논문은 MM-IQ라는 새로운 평가 프레임워크를 제안합니다. MM-IQ는 8개의 서로 다른 추론 패러다임을 아우르는 2,710개의 정교하게 선별된 테스트 항목으로 구성되어 있습니다. 기존의 멀티모달 시스템이 인간의 인지 능력에 가까워지지 못함을 강조하며, 발전을 위한 새로운 접근의 필요성을 나타냅니다.

- **Technical Details**: MM-IQ는 다양한 입출력 형태, 문제 구성 및 추론 패러다임에 따라 기존의 AVR 벤치마크와 비교할 수 있습니다. 이 평가 기준은 LMMs의 도형 인식, 추론, 그리고 숫자 추리 능력을 평가하는 데 중점을 둡니다. 모든 테스트 항목은 인간의 고차원적 추론 능력을 기반으로 체크되며, 8개의 미세한 추론 패러다임으로 분류됩니다.

- **Performance Highlights**: 현재의 멀티모달 시스템은 우연히 맞힐 확률(25%)을 조금 넘는 27.49%의 정확도만을 보여주고 있습니다. 이는 인간 수준의 성능 51.27%와 비교할 때 매우 부족한 수치임을 시사합니다. MM-IQ를 통한 체계적인 평가 결과는 현재의 LMMs가 인간과 같은 인지 적응 능력을 갖추고 있지 않음을 강조합니다.



### Learning Autonomous Code Integration for Math Language Models (https://arxiv.org/abs/2502.00691)
- **What's New**: 최근 연구에서 수학 대형 언어 모델(LLMs)을 위한 도구 통합(tool integration)의 한계가 발견되었습니다. 기존의 도구 통합 모델이 외부 지시서에 의존하여 Chain-of-Thought (CoT)나 코드를 사용할지 결정하는 반면, 새로운 연구는 LLM이 독립적으로 방법론을 선택할 수 있는 자율 코드 통합(Autonomous Code integration) 접근 방식을 제안합니다. 이는 LLM이 안정적인 감독 없이 독자적으로 자신의 전략 선택 방법을 개발할 수 있게 합니다.

- **Technical Details**: 제안된 Expectation-Maximization (EM) 프레임워크는 모델의 능력을 탐색함으로써 의사결정 과정을 개선합니다. E-step은 자기 탐색을 통해 참조 전략을 계산하고, M-step은 이 새로운 신념에 기반하여 LLM을 업데이트합니다. 이 과정에서 최신 데이터 합성(data synthesis) 전략과 비정책 강화 학습(off-policy reinforcement learning)을 통합하여 효율성을 높였습니다.

- **Performance Highlights**: 실험 결과, 단지 공개 질의 세트를 사용하여 제안된 방법이 기존 수학 LLM의 성능을 크게 향상시키며, MATH 벤치마크에서 정확도를 약 20% 향상시킨 65.28%에 도달했습니다. 또한, 코드 실행 횟수를 최대 65%까지 줄이는 성과도 나타났습니다. 이는 수학 문제 해결을 위한 코드 통합이 더욱 효과적으로 이루어질 수 있음을 보여줍니다.



### LLM-based event log analysis techniques: A survey (https://arxiv.org/abs/2502.00677)
- **What's New**: 이 논문은 최근의 연구 결과를 바탕으로 대규모 언어 모델(LLMs)을 활용한 이벤트 로그 분석 기법의 발전을 종합적으로 소개합니다. 이벤트 로그는 시스템 내에서 발생하는 다양한 이벤트의 기록을 담고 있으며, 이 기록을 분석하는 것은 시스템 성능 및 보안 위협에 대한 통찰력을 제공합니다. 그러나 방대한 양의 이벤트로 인해 분석은 시간과 자원을 많이 소모하며, 이로 인해 보안 전문가들이 놓칠 수 있는 중요한 통찰이 생깁니다. 이에 따라, 연구자들은 LLM을 이용해 이 과정을 자동화하는 방법을 모색하고 있습니다.

- **Technical Details**: 연구에 사용된 방법론으로는 LLM을 활용한 다양한 기법들이 포함됩니다. 특히, in-context learning, fine-tuning 및 Retrieval-Augmented Generation(RAG) 기술이 성능에 미치는 영향을 분석합니다. 논문은 이들 기술이 이벤트 로그 분석에서 어떻게 활용될 수 있는지 문헌 분석을 통해 설명합니다. 이를 통해 LLM 기반의 이벤트 로그 분석의 현재 상태와 발전 가능성을 살펴봅니다.

- **Performance Highlights**: 대부분의 연구는 이상 탐지에 중점을 두고 있으며, 여러 모델들이 LLM을 통해 성능이 크게 향상된 사례를 제시합니다. 예를 들어, BERT-Log 모델은 F1 점수 기준으로 0.99의 성능을 기록하여 데이터를 분석하는 데 있어서 우수한 신뢰도를 보여주었습니다. 그러나 모델의 성능 평가가 부족한 점이 있으며, 높은 허위 긍정률과 같은 한계점도 지적됩니다. 이러한 연구 결과들은 향후 연구에서 해결해야 할 과제로 남아있습니다.



### Agency in the Age of AI (https://arxiv.org/abs/2502.00648)
- **What's New**: 이번 논문은 생성 AI의 사회적 영향에 대한 우려가 커지고 있는 가운데, 이를 탐구하기 위한 적절한 이론적 접근으로 'agency' 개념을 제안합니다. 특히, 비영리적이거나 악의적인 방식으로 AI 도구가 남용될 가능성에 대해 강조하며, 이러한 문제를 해결하기 위한 연구 방향을 제시합니다. 또한, AI 도구의 발전과 함께 나타나는 다양한 부작용과 그로 인한 사회적 영향에 대한 체계적인 분석을 합니다.

- **Technical Details**: 문헌에서 제시된 여러 종류의 해악을 다루기 위해, 악의적인 행위자에 의한 남용, 정보 환경의 변화로 인한 착취, 도구 자체의 부패, 그리고 의도치 않은 결과들을 다양한 카테고리로 나누어 설명합니다. 이러한 모든 해악은 대리인의 'agency' 이론 관점에서 이해될 수 있으며, 이를 통해 창출되는 새로운 목표와 행동 범위는 AI 도구의 의도된 사용에 따라 달라질 수 있습니다. 이와 같은 이론적 틀을 통해 더 큰 문제를 통합적으로 이해할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 이 논문은 생성 AI의 발전이 개인과 사회의 'agency'에 미치는 다양한 영향을 체계적으로 살펴봅니다. 특히, AI 도구를 통해 생성된 정보가 민주적 과정이나 사회적 결정에 미치는 악영향을 강조하며, AI 기술의 부작용을 사전에 방지할 수 있는 방안을 모색하고 있습니다. 악의적인 사용의 가능성 및 그로 인한 신뢰도 저하 문제 또한 중요한 논의 사안으로 자리매김하며, 이를 해결하기 위한 연구 필요성을 강조합니다.



### CollabLLM: From Passive Responders to Active Collaborators (https://arxiv.org/abs/2502.00640)
Comments:
          23 pages

- **What's New**: CollabLLM은 다중 회차 대화에서의 인간-Language Models (LLMs) 협업을 촉진하는 새로운 훈련 프레임워크입니다. 이 시스템은 Multiturn-aware Reward (MR)라는 혁신적인 보상 함수를 통해 모델의 장기적인 성과를 추정합니다. 이를 통해 CollabLLM은 사용자의 의도를 적극적으로 파악하고, 통찰력 있는 제안을 제공하여 AI와의 상호작용을 향상시킵니다.

- **Technical Details**: CollabLLM은 대화의 여러 회차에서 발생하는 사용자의 숨겨진 목표를 고려하여 인간과 모델 간의 최적의 협업을 이끌어냅니다. MR을 통해 대화의 효율성과 질을 평가하고, 이를 기반으로 모델의 응답을 강화하는 학습 알고리즘을 적용합니다. 또한 문서 작성, 코드 생성, 질문 응답과 같은 세 가지 도전적인 다중 회차 작업을 설정하여 성능을 평가합니다.

- **Performance Highlights**: CollabLLM은 기존의 기준 모델보다 평균 18.5% 향상된 작업 성과와 46.3% 증가한 상호작용 능력을 보였습니다. 또한 201명의 사용자 연구를 통해 사용자 만족도가 17.6% 증가하고, 소요 시간을 평균 10.4% 절감하는 성과를 거두었습니다. qualitative 분석 결과, CollabLLM은 사용자에게 통찰력 있는 질문과 제안을 제공하여 보다 능동적인 협업을 이루어냈음을 확인하였습니다.



### Lipschitz Lifelong Monte Carlo Tree Search for Mastering Non-Stationary Tasks (https://arxiv.org/abs/2502.00633)
Comments:
          6 figures

- **What's New**: 이번 논문에서 제안하는 LiZero는 MCTS를 이용한 Lipschitz 생애 지속적 계획(Lifelong Planning)을 통해 비정상성을 고려한 새로운 알고리즘을 제시하고 있습니다. 기존의 MCTS 모델들이 비정상적인 작업 동역학에 대한 고려 없이 설계된 반면, LiZero는 작업 간 Lipschitz 연속성과 지식의 신뢰도에 따라 이전 작업의 지식을 새로운 작업으로 효과적으로 전이할 수 있는 적응형 UCT(aUCT) 규칙을 도입합니다.

- **Technical Details**: LiZero는 Monte Carlo 행동 샘플링에서 지식을 최적화하고 전이할 수 있도록 aUCT 규칙을 정의하고, 두 작업 사이의 거리 메트릭을 구축하여 적응형 UCT 경계를 정립합니다. aUCT는 Lipschitz 연속성과 샘플의 수에 따른 신뢰도로 구성되며, 이를 통해 MCTS의 탐색 효율성이 크게 향상됩니다. 이 알고리즘은 데이터 기반 및 모델 기반 방법으로 온라인에서 aUCT 값을 실시간으로 추정하는 효율적인 솔루션을 개발합니다.

- **Performance Highlights**: 실험을 통해 LiZero는 기존의 MCTS 및 생애 지속적 학습(Lifelong Learning) 기준선보다 3~4배 더 빠른 수렴 속도 및 약 31% 높은 초기 보상을 달성함으로써 뛰어난 성능을 입증하였습니다. 이러한 결과는 LiZero가 동적인 실제 환경에서 의사 결정 및 계획 수립에 있어 중요한 진전을 이룰 수 있는 잠재력을 보여줍니다.



### Advanced Weakly-Supervised Formula Exploration for Neuro-Symbolic Mathematical Reasoning (https://arxiv.org/abs/2502.00629)
- **What's New**: 최근 신경-기호적(neuro-symbolic) 방식이 인공지능 시스템을 보강하여 더 높은 정확도와 제어력을 제공하는 방법으로 인기를 끌고 있습니다. 하지만 대형 언어 모델(LLMs)의 요구된 기호적 명령어가 항상 유효하게 생성되지 않아서, 모델이 훈련 데이터 없이 기호적 지침을 탐색하는 한계를 드러냅니다. 본 연구에서는 문제 입력과 최종 출력으로부터 약한 감독(weak supervision)을 통해 중간 레이블을 탐색하는 개선된 접근 방식을 제안합니다.

- **Technical Details**: 신경-기호적 방법은 문제를 해결하기 위해 질문 텍스트를 수신하고 해석하는 신경망을 사용하여 기호적 표기법을 생성하는 방식으로 구성됩니다. 이 과정은 각 문제에 최적의 공식(formula)을 점진적으로 발견하는 반복적 탐색 과정을 포함하며, 기존의 약한 감독 학습 방법에서 발전된 점이 특징입니다. 또한 계산 및 해석을 위해 외부 도구를 활용하여 기호적 계산의 정확성과 제어력을 결합합니다.

- **Performance Highlights**: 본 연구는 수학 데이터셋을 통해 제안된 방법의 효과성을 제시하며, 전통적인 약한 감독 학습 방법과 비교했을 때 더 어려운 수학 문제에 적용 가능한 능력을 갖추고 있음을 보여줍니다. 또한 수식 표기법을 기능적 영역 특화 언어로 확장하여, 보다 일반화된 기호체계에서 수학적 추론 문제를 해결할 수 있는 가능성을 제시합니다. 궁극적으로 이 접근법은 기존 방식과 비교하여 더 나은 성능을 입증합니다.



### Understanding Multimodal LLMs Under Distribution Shifts: An Information-Theoretic Approach (https://arxiv.org/abs/2502.00577)
- **What's New**: 이 논문에서는 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)의 배포 환경 변화에 대한 안정성과 신뢰성을 확보하기 위한 이론적 프레임워크의 필요성을 강조합니다. 기존 연구들은 MLLMs의 성능 평가를 위한 다양한 실험을 제공했으나, 이론적인 기반이 부족했던 점을 지적하며, 효과적인 상호 정보량(Effective Mutual Information, EMI)이라는 새로운 측정 지표를 도입했습니다. 이 EMI는 입력 쿼리와 모델 응답 간의 연관성을 정량화하여 MLLMs의 성능을 분석하는 데 중요한 역할을 합니다.

- **Technical Details**: EMI를 도입함으로써, 연구진은 MLLM의 깊이 있는 진단과 최악의 경우 성능 차이를 수치적으로 평가할 수 있는 상한선을 유도했습니다. 그 성능 격차는 배포 환경 변화에 의해 정의된 분포적 불일치와 연결되어 있습니다. 논문 내에서 EMI와 실제 평가 지표인 win rate 간의 이론적 관계도 입증되었으며, 이를 통해 EMI가 성능 변동을 이해하는 데 기여할 수 있음을 보여줍니다.

- **Performance Highlights**: 연구진은 61개의 배포 환경 변화 시나리오에 대해 MLLMs의 성능을 종합적으로 검증했습니다. 실험 결과 EMI와 win rate 간의 강한 상관 관계가 확인되었으며, EMI의 차이 및 상한선과의 관계 또한 검토되었습니다. 이로 인해 EMI 프레임워크가 다양한 배포 환경 변화에서 MLLM의 성능 갭을 포착하는 데 효과적이라는 점이 입증되었습니다.



### Who's the MVP? A Game-Theoretic Evaluation Benchmark for Modular Attribution in LLM Agents (https://arxiv.org/abs/2502.00510)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델) 에이전트를 최적화하기 위한 새로운 평가 프레임워크인 CapaBench를 제안합니다. CapaBench는 협력 게임 이론의 Shapley Value에 기반하여 각 모듈의 기여도를 정량화합니다. 이를 통해 모듈의 성능 기여를 독립적으로 측정하고 분석할 수 있게 됩니다.

- **Technical Details**: CapaBench는 계획(planning), 추론(reasoning), 행동 실행(action execution), 반영(reflection) 등 LLM 아키텍처의 개별 모듈의 기여를 체계적으로 평가합니다. Shapley Value는 모듈의 기여와 상호작용 효과를 동시에 포착할 수 있는 수학적 접근을 제공합니다. 이러한 방법론은 모듈 조합에 따른 성능 예측을 가능하게 하여, 특정 모듈을 타겟으로 한 최적화를 지원합니다.

- **Performance Highlights**: CapaBench는 1,000개 이상의 멀티 라운드 작업으로 구성된 대규모 데이터셋을 구축하여 다양한 실제를 반영하는 평가를 제공합니다. 모듈의 Shapley Value가 높을수록 아젠트의 성능 향상에 긍정적인 영향을 미친다는 것을 실험을 통해 입증했습니다. 이 데이터셋은 공개될 예정이며, LLM 에이전트 성능 평가의 기초로 활용될 수 있을 것입니다.



### Discovering Directly-Follows Graph Model for Acyclic Processes (https://arxiv.org/abs/2502.00499)
Comments:
          24 pages, 15 figures

- **What's New**: 본 논문에서는 비순환(acyclic) 프로세스에 대한 새로운 프로세스 발견 알고리즘을 제안합니다. 기존의 방법들은 프로세스 모델에 사이클을 포함시킬 수 있는 문제점을 안고 있으며, 이로 인해 모델의 정확도가 낮아질 수 있습니다. 제안된 알고리즘은 이벤트 로그를 여러 부분으로 나누어 비순환 DFG(Directly Follows Graph) 모델을 발견하고, 이들 모델을 결합하여 사이클의 생성을 피합니다.

- **Technical Details**: 프로세스 마이닝(process mining)을 통해 비순환성을 가진 프로세스를 효과적으로 해석할 수 있는 방법을 제시하고 있습니다. 이 알고리즘은 이벤트 로그를 여러 부분으로 분리한 후, 이를 결합하여 비순환 DFG 모델을 생성합니다. 사이클을 피하기 위해 모델 내에서 동일한 액션 노드를 여러 번 허용하는 방법을 사용합니다.

- **Performance Highlights**: 알고리즘의 성능은 실제 및 인공 이벤트 로그를 통해 검증되었으며, 사이클이 없는 모델은 시각적 명확성과 정확성을 높입니다. 사이클이 없을 경우 모델이 제공하는 정보는 더 정확해지며, 따라서 이를 기반으로 하는 사이클 민감한 방법이나 시각화를 적용할 수 있는 가능성도 열립니다.



### MetaOpenFOAM 2.0: Large Language Model Driven Chain of Thought for Automating CFD Simulation and Post-Processing (https://arxiv.org/abs/2502.00498)
Comments:
          16 pages,11 figures

- **What's New**: MetaOpenFOAM 2.0은 Chain of Thought (COT) 분해와 반복 검증을 활용하여 비전문가도 자연어 입력을 통해 접근할 수 있도록 설계되었습니다. 이는 사용자 친화성을 높이는 데 중점을 두고 있으며, CFD(Computational Fluid Dynamics) 작업의 성능을 개선합니다. 이러한 새로운 접근 방식은 특히 복잡한 후처리 작업에서 중요한 역할을 할 것입니다.

- **Technical Details**: 이 모델은 새로운 벤치마크에서 유체 흐름, 열 전달, 연소 시뮬레이션 및 후처리 작업인 데이터 추출과 시각화를 포함한 테스트를 진행했습니다. 그 결과 MetaOpenFOAM 2.0은 6.3/7의 실행 가능성 점수와 86.9%의 합격률을 기록하며, 이전 버전인 MetaOpenFOAM 1.0의 성과를 크게 초월했습니다. 또한, 평균적으로 $0.15의 비용으로 효율적인 운용을 보여주었습니다.

- **Performance Highlights**: Ablation 연구를 통해 COT 기반 분해와 반복 정제가 작업 성능을 크게 향상시킨 것으로 나타났습니다. 또한 COT 단계 수를 늘리면 정확성이 향상되는 동시에 토큰 사용량이 증가하여 LLM(대형 언어 모델)의 포스트 트레이닝 스케일링 경향과 일치함을 보여주었습니다. 이러한 결과는 산업 및 연구 응용을 위한 CFD 작업 흐름의 자동화에서 LLM의 변혁적 가능성을 강조합니다.



### Doing More with Less -- Implementing Routing Strategies in Large Language Model-Based Systems: An Extended Survey (https://arxiv.org/abs/2502.00409)
- **What's New**: 이번 논문에서는 LLM(대형 언어 모델) 기반 시스템에서 사용자의 쿼리를 가장 적합한 구성 요소로 라우팅하는 메커니즘을 제안합니다. 이러한 접근 방법은 리소스를 최적화하고 비용을 절감하면서 응답 품질을 향상시킬 수 있도록 돕습니다. 특히, Routing을 이용하여 사전 훈련된 LLM의 적절한 선택과 각 쿼리의 특성에 맞는 최적의 작업을 수행할 수 있습니다.

- **Technical Details**: 라우팅 메커니즘은 주어진 쿼리(q)에 대해 모델 집합(ℳ={M1,…,Mn})에서 가장 적합한 LLM을 선택하는 시스템의 구성 요소입니다. 이 메커니즘은 성능 최대화와 예산(B) 제약을 고려하여 최적의 선택을 이끌어낼 수 있습니다. 다양한 LLMs와 임베딩 모델들의 호출 비용(CM)이 경량 모델을 통해 줄일 수 있음을 강조하며, 특정 작업에 가장 적합한 모델을 선택하는 것이 중요하다고 설명합니다.

- **Performance Highlights**: Routing을 통해 LLM 기반 시스템의 평균 대기 시간을 줄일 수 있으며, 이는 사용자가 간단한 질문을 할 때 리소스 요구를 줄이는 데 기여할 것입니다. 또한, 기존의 LLM 기반 시스템이 전반적으로 과도한 하드웨어 자원을 사용할 필요가 없게 만들 수 있습니다. 비용, 성능, 환경 영향 등을 고려하여 최적 효율성의 중요성을 강조하며, 이러한 접근 방식은 향후 연구 방향에도 큰 영향을 미칠 것으로 예상됩니다.



### ALU: Agentic LLM Unlearning (https://arxiv.org/abs/2502.00406)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)에서 정보 삭제(unlearning) 기능을 위한 새로운 접근 방식인 agentic LLM unlearning (ALU) 방법을 제안합니다. ALU는 모델 재교육 없이도 LLM의 유용성을 유지하면서 효과적인 정보 삭제를 가능하게 하는 다중 에이전트 구조를 사용합니다. 기존 방법들이 정보 삭제의 효과와 유유성(utility) 간의 균형을 맞추는 데 어려움을 겪고 있는 반면, ALU는 이러한 문제를 해결합니다.

- **Technical Details**: ALU 프레임워크는 특정 작업을 수행할 여러 LLM 에이전트를 포함하여 정보 삭제를 실행합니다. 각 에이전트는 삭제 과정의 특정 단계에 설계되어 있으며, 모델 가중치를 업데이트할 필요가 없습니다. 이는 어떤 기초 LLM 모델에도 변화 없이 적용 가능하고, 사용자는 시간에 따라 유연하게 삭제 요청을 할 수 있습니다.

- **Performance Highlights**: ALU는 TOFU, WMDP, WPU와 같은 기존 벤치마크 및 jailbreak 기술에 대한 광범위한 실험을 통해 기존 LLM 정보 삭제 방법들 중에서 가장 강력한 성능을 발휘하는 것으로 입증되었습니다. 특히 ALU는 최대 1000개의 삭제 목표에서 평가되고, 모든 위의 정보 삭제 방법들의 평가 범위를 초과하며 우수한 성능을 보여줍니다.



### A Differentiated Reward Method for Reinforcement Learning based Multi-Vehicle Cooperative Decision-Making Algorithms (https://arxiv.org/abs/2502.00352)
Comments:
          8 pages, 3 figures, submitted to IEEE IV 2025

- **What's New**: 이번 논문에서는 유동적인 다중 차량 협력 주행 전략 최적화를 위해 차별화된 보상 방법을 제안합니다. 이는 특정 상태 전이 시스템에 기반하여, 보상 설계 과정에 교통 흐름 특성을 분석하여 상태 전이 기울기 정보를 통합합니다. 이를 통해 다중 차량 협력 의사결정에서의 행동 선택 및 정책 학습을 최적화하고자 하는 시도를 합니다.

- **Technical Details**: 제안된 방법은 강화 학습 알고리즘인 MAPPO, MADQN, QMIX의 성능을 검증하여 다중 차량 협력 결정 과정의 훈련 수렴 속도를 획기적으로 향상시키는 것을 보여줍니다. 이것은 또한 교통 효율성, 안전성 및 행동 합리성 측면에서 기존 보상 방법보다 우수한 성능을 발휘합니다. 실험 결과는 이 방법이 강력한 확장성과 환경 적응성을 보일 수 있음을 보여줍니다.

- **Performance Highlights**: 제안된 차별화된 보상 방법은 다중 차량 협력 의사결정 문제에서 지속적인 교통 흐름 환경 아래에서 다양한 자율주행차 침투율에 대한 시뮬레이션 실험을 통해 성능이 검증되었습니다. 이러한 결과는 다중 에이전트 교통 시나리오에서의 학습 안정성과 스케일러블한 특성을 입증합니다.



### The role of positional encodings in the ARC benchmark (https://arxiv.org/abs/2502.00174)
- **What's New**: 이번 연구는 Abstraction and Reasoning Corpus (ARC)에서의 추상 추론 문제 해결을 위한 Positional Encoding의 역할을 조명합니다. CodeT5+ 모델을 사례로 들어, 기존 Positional Encoding의 한계가 모델 성능과 사고에 미치는 영향을 실험적으로 입증하였습니다. 2D Positional Encoding이 데이터에 제한이 있는 환경에서 우수한 성능을 발휘함을 확인하였으며, 이는 ARC 작업에서 효과적이라고 강조합니다.

- **Technical Details**: Positional Encoding은 Transformer 모델의 주요 구성 요소로, 순차적 데이터 처리와 토큰 간의 관계를 이해하는 데 중요한 역할을 합니다. 연구에서는 기존의 1D Positional Encoding과 2D Sinusoidal Encoding을 비교하며, RoPE 및 학습 가능한 임베딩(Learned Embeddings)과 같은 다양한 인코딩 전략의 성능을 분석합니다. 특히, 2D Positional Encoding이 모델의 사고 방식을 어떻게 향상시키는지를 집중적으로 평가했습니다.

- **Performance Highlights**: 실험 결과, 모든 모델 아키텍처에서 Positional Encoding은 성능에 중요한 영향을 미치는 것으로 나타났습니다. 2D Positional Encoding은 데이터가 제한된 상황에서도 우수한 성능을 보여주었고, 이는 LLM에 대한 보다 나은 사고와 이해를 가능하게 합니다. 반면, RoPE는 충분한 데이터가 제공될 때 약간의 성능 이점을 보였으나, 데이터가 적은 경우에는 2D Encoding이 명백한 우위를 보였습니다.



### Counting and Reasoning with Plans (https://arxiv.org/abs/2502.00145)
- **What's New**: 이 논문에서는 고전적 계획의 플랜 공간에서 정량적 및 정성적 추론을 통합하는 새로운 프레임워크를 제안합니다. 특히 다항식적으로 제한된(plnly bounded) 플랜에 중점을 두고, 복잡성을 연구하여 풍부한 추론 모드를 제공합니다. 또한, 플랜에 대한 중요성을 이해하기 위한 'facet'의 개념을 도입하고, 다루기 쉬운 문제로 정량적 추론을 가능하게 합니다.

- **Technical Details**: 본 연구에서는 폴리노말 제약을 가진 계획 길이에 대한 다양한 카운팅 및 추론 문제의 분류 체계를 소개합니다. 중요한 결과 중 하나는 특정 연산자가 포함된 플랜의 수를 파악하는 문제의 C=P-completeness라는 점입니다. 이는 카운팅 문제보다 훨씬 더 효율적으로 해결할 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 새로 제안한 도구인 Planalyst는 계획 작업을 명제 공식으로 변환하여 관련 플랜을 효과적으로 카운트하고 자동화된 추론을 수행할 수 있습니다. 실험 결과에서 Planalyst는 대규모 플랜 공간에서 특히 유리한 성능을 보여 주었으며, 수조 개의 플랜에 대한 추론 문제를 해결하는 데 매우 적합합니다.



### Towards Efficient Multi-Objective Optimisation for Real-World Power Grid Topology Contro (https://arxiv.org/abs/2502.00034)
- **What's New**: 본 논문은 전력망 토폴로지 제어를 위한 효율적이고 확장 가능한 Multi-Objective Optimisation (MOO) 방법론을 제안합니다. 이 방법론은 강화 학습(Reinforcement Learning, RL) 학습 단계와 빠른 계획 단계로 구성되어 있어, 미지의 시나리오를 위한 하루 전 계획을 생성할 수 있습니다. 특히, 유럽의 전송 시스템 운영업체인 TenneT의 역사적 데이터를 사용하여 최소한의 배치 시간으로 하루 전 계획을 4-7분 안에 생성함을 보여줍니다.

- **Technical Details**: 전력망은 무향 그래프 G=(V,E)로 표현되며, 여기서 V는 서브스테이션의 집합, E는 전송선의 집합을 나타냅니다. 전력망 운영은 일련의 의사 결정 문제로 정형화될 수 있으며, 각 시점에서 그리드의 상태는 여러 변수를 포함한 상태 벡터로 표현됩니다. 본 논문에서는 전력망의 토폴로지 구성에만 초점을 맞추고 있으며, RL을 통해 최적의 정책을 학습할 수 있는 강력한 프레임워크를 이용합니다.

- **Performance Highlights**: 본 연구의 결과는 제안된 MOO 방법이 실질적인 전력망 관리에 기여할 수 있는 가능성을 보여줍니다. TSOs가 이 접근 방식을 채택할 경우, 연간 수백만 유로의 절감 효과를 기대할 수 있으며, 이는 경제적 유인이 될 수 있습니다. 궁극적으로 제안된 방법은 전력망 운영 계획을 위한 실용적이고 계산 효율적인 도구로 자리매김할 것입니다.



### A Dynamic and High-Precision Method for Scenario-Based HRA Synthetic Data Collection in Multi-Agent Collaborative Environments Driven by LLMs (https://arxiv.org/abs/2502.00022)
- **What's New**: 이 연구에서는 자동화된 HRA (Human Reliability Analysis) 데이터 수집을 위한 새로운 패러다임을 제시합니다. 제안하는 방법은 협업 환경에서의 작업 부하 측정에 중점을 두며, 대규모 언어 모델(LLM)을 활용하여 다양한 시나리오에서의 인간 행동과 인지 부하를 실시간으로 시뮬레이션합니다. 이를 통해 기존 상업용 모델보다 더 높은 예측 정확성을 보여주는 WELLA (Workload Estimation with LLMs and Agents)를 개발했습니다.

- **Technical Details**: 연구에서는 다중 에이전트 협업 시나리오를 중심으로 작업 부하 데이터를 수집하고, 대규모 모델을 통해 가상 인지 경로를 생성합니다. Qwen2.5-7B 모델의 감독 학습(Supervised Fine-Tuning, SFT) 과정을 통해 대규모 모델을 정제하여 WELLA를 개발하였습니다. 이 방법은 운영자의 작업 부하 변동에 동적으로 적응할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, WELLA는 RO1, RO2, RO3, CO, SO 역할에 대한 작업 부하 예측에서 현재의 상업 모델보다 더 우수한 성능을 나타냈습니다. 이는 기존의 전문가 추정 방법과 비교할 때 더 정확하고 유연하며 확장 가능한 작업 부하 추정이 가능함을 보여줍니다. 이 연구는 HRA(인간 신뢰성 분석)의 데이터 수집 혁신에 기여할 것으로 기대됩니다.



### Temporal Reasoning in AI systems (https://arxiv.org/abs/2502.00020)
- **What's New**: 이번 연구에서는 인공지능(AI) 시스템이 기존의 fluents 및 이벤트에 대한 정보를 올바르게 추론할 수 없는 한계를 극복하기 위해 필요한 지식 표현 및 추론 방식을 논의합니다. 특히, Cyc Knowledge Base를 활용한 견고한 시간적 예측(temporal projection)을 위한 접근 방식을 제안합니다.

- **Technical Details**: 연구에서는 fluents의 위험 기간을 시작하고 종료하는 이벤트의 역할을 분석합니다. 또한, 사실의 지속성(persistence)을 나타내는 이산 생존 함수(discrete survival functions)를 사용하여 주어진 fluent를 외삽(extrapolate)하는 방법을 설명합니다. 이러한 외삽된 간격은 시간적 제약(temporal constraints) 및 기타 유형의 일반 상식(commonsense knowledge)으로 절단될 수 있습니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 Q/A 성능 측면에서 상당한 개선을 가져온다는 것을 보여줍니다. 이는 인지 시스템의 기초 문제인 일반 상식 시간 추론(commonsense temporal reasoning)에서의 혁신을 나타냅니다.



### Growth Patterns of Inferenc (https://arxiv.org/abs/2502.00019)
- **What's New**: 이번 연구에서는 첫 번째 순서의 검색 공간(first-order search space)이 추론(inference)에 미치는 영향을 살펴보았습니다. 특히, 어떤 유형의 사실(facts)을 학습하는 것이 가장 효과적인지를 정리했습니다. 이러한 질문에 대한 답변은 deductive reasoning의 역학을 이해하고 효율적인 추론을 지원하는 대규모 지식 기반 학습 시스템을 구축하는 데 필수적입니다.

- **Technical Details**: 연구팀은 ground facts의 분포가 검색 공간 내에서 추론 성능(inference performance)에 미치는 영향을 모델링했습니다. 실험 결과, 균일한 검색 공간은 대규모 지식 베이스(knowledge base, KB)에 적합한 반면, skewed degree distribution을 가진 검색 공간은 소규모 KB에서 더 나은 성능을 보이는 것으로 나타났습니다. 또한, 특정 경우에서는 Q/A 성능에서 급격한 전환(sharp transition)이 관찰되었습니다.

- **Performance Highlights**: 연구 결과는 기존 지식이 있는 검색 공간의 구조를 분석하여, 학습 시스템에서 새로운 ground facts를 획득하는 데 도움을 줄 수 있음을 시사합니다. 이를 통해, 지식 기반 학습 시스템의 효율성을 높이고 성능을 최적화할 수 있는 방법론이 마련되었습니다.



### An Expectation-Maximization Algorithm-based Autoregressive Model for the Fuzzy Job Shop Scheduling Problem (https://arxiv.org/abs/2502.00018)
- **What's New**: 이 논문은 전통적인 작업장 스케줄링 문제(JSSP)의 진화를 통해 불확실성을 포함한 퍼지 작업장 스케줄링 문제(FJSSP)를 다룹니다. 최근 연구에서 신경 조합 최적화(NCO)의 효율성이 입증되면서, 이 기법을 퍼지 스케줄링에 적용하는 것이 상대적으로 미개척 영역으로 남아 있음을 인식하고 이를 탐구하는 것이 주요 목표입니다. 본 연구에서는 기대-최대화 알고리즘 기반 자기회귀 모델(EMARM)을 도입하여 FJSSP 문제를 해결하기 위한 새로운 방법론을 진행합니다.

- **Technical Details**: FJSSP를 해결하기 위해 제안된 EMARM은 훈련 과정에서 주어진 인스턴스에서 스케줄링 방안을 생성하고, 이를 기반으로 자기회귀 모델의 가중치를 조정하는 E-step과 M-step을 alternation 형식으로 진행합니다. 이러한 방식은 NCO 프레임워크에서 흔히 발생하는 실제 레이블 획득의 문제를 효과적으로 극복합니다. 또한, 퍼지 정보를 이해하고 처리할 수 있도록 신경망에 수작업으로 설계된 사전 정보와 퍼지 수의 순위를 매기는 새로운 방법을 적용하였습니다.

- **Performance Highlights**: 실험 결과, EMARM은 FJSSP 문제 해결에 있어 다른 알고리즘들보다 월등한 성능을 보였습니다. 이는 퍼지 스케줄링에서의 실제 적용 가능성을 강조하며, 새로운 접근 방법이 특정 상황에서 유망한 결과를 도출할 수 있음을 나타냅니다. 전반적으로 EMARM은 퍼지 스케줄링 방법론을 향상시킬 가능성을 가지고 있습니다.



### Lifelong Sequential Knowledge Editing without Model Degradation (https://arxiv.org/abs/2502.01636)
- **What's New**: 이 논문에서는 기존의 파라미터 수정 지식 편집 방법의 한계를 극복하고, 모델 성능 저하 없이 10,000개의 순차적 지식 편집을 가능하게 하는 ENCORE 방법을 제안합니다. 지식 편집 과정에서 발생하는 오버피팅(Overfitting)과 편집된 행렬의 비대칭적 노름 성장(Norm Growth)의 문제를 해결하는 데 중점을 두었습니다. ENCORE는 이러한 문제를 완화시켜 모델의 다운스트림 성능을 유지하면서, 지난 방법들보다 61%에서 64%까지 빠른 속도를 자랑합니다.

- **Technical Details**: 본 연구는 'locate-then-edit' 방법으로 알려진 파라미터 수정 지식 편집 메커니즘을 두 단계의 미세 조정(fine-tuning) 과정으로 설명합니다. 첫 번째 단계에서는 경량 회귀(guided descent)를 사용하여 적절한 활성화 벡터를 찾고, 두 번째 단계에서는 최소 제곱 손실(minimum squared loss) 함수를 사용하여 선택한 MLP(다층 퍼셉트론) 행렬의 가중치를 업데이트합니다. 새로운 조기 중단 방법(Most-Probable Early Stopping, MPES)을 도입하여 편집된 사실의 오버피팅을 줄이고, Frobenius 노름 제약조건을 추가하여 행렬의 증가를 조절합니다.

- **Performance Highlights**: ENCORE는 GPT2-XL, Llama-2-7B, Llama-3-8B에서 이전의 locate-then-edit 방법들보다 월등하게 성능을 높였으며, 특히 Llama3-8B 위에서 MEMIT보다 61%, AlphaEdit보다 64% 빠른 속도로 편집을 수행합니다. 이로 인해 모델의 성능 저하 없이 10,000개의 순차적 지식 편집이 가능해지며, 이는 대규모 지식 편집 분야의 새로운 이정표를 설정합니다.



### The AI Agent Index (https://arxiv.org/abs/2502.01635)
Comments:
          Accompanying website: this https URL

- **What's New**: 이번 연구에서는 에이전틱(Agentic) AI 시스템의 기술적 구성요소, 의도된 사용 및 안전 기능을 문서화하기 위한 첫 번째 공개 데이터베이스인 AI Agent Index를 소개합니다. 이 색인은 현재 배포 중인 에이전틱 시스템에 대한 정보를 기록하고 있으며, 해당 시스템의 구성 요소와 응용 분야, 위험 관리 관행을 정리하여 공개합니다. 이를 통해 에이전틱 AI 시스템의 실제 성과와 잠재적 위험이 더욱 투명하게 드러나길 기대합니다.

- **Technical Details**: 에이전틱 AI 시스템은 일반적으로 기본 모델에 추론, 계획, 기억 및 도구 사용을 위한 발판(Scaffolding)이 추가되어 구성됩니다. 이 시스템들은 여러 도메인에서 구현되고 있으며, 복잡한 작업들을 계획하고 직접 실행할 수 있는 능력을 가지고 있습니다. 그러나 개발자들은 일반적으로 시스템의 안전성과 위험 관리 관행에 대한 정보는 제한적으로 제공하고 있는 실정입니다.

- **Performance Highlights**: AI Agent Index는 현재 배포된 67개의 에이전틱 AI 시스템에 대한 포괄적인 샘플로 구성되어 있습니다. 이 색인을 통해 에이전틱 시스템의 기능과 응용 분야에 대한 상세한 정보를 제공하고 있지만, 안전 평가 및 위험 감소에 대한 정보는 매우 제한적이라는 점이 발견되었습니다. 따라서 이러한 정보의 가용성을 더욱 높이는 것이 중요합니다.



### Online Gradient Boosting Decision Tree: In-Place Updates for Efficient Adding/Deleting Data (https://arxiv.org/abs/2502.01634)
Comments:
          25 pages, 11 figures, 16 tables. Keywords: Decremental Learning, Incremental Learning, Machine Unlearning, Online Learning, Gradient Boosting Decision Trees, GBDTs

- **What's New**: 이 논문은 Gradient Boosting Decision Tree (GBDT)를 위한 최초의 온라인 학습 프레임워크를 제안합니다. 기존 GBDT는 학습 후에 데이터 인스턴스를 추가하거나 삭제할 수 없었으나, 이 연구에서는 이를 가능하게 하는 효율적인 방법론을 제시합니다. 제안하는 프레임워크는 증분 학습(incremental learning)과 감소 학습(decremental learning)을 모두 지원하여 유연한 데이터 관리가 가능합니다. 또한, 학습 비용을 줄이기 위한 여러 최적화 방안을 도입하였습니다.

- **Technical Details**: GBDT는 여러 결정 트리를 결합하여 정확한 예측 모델을 생성하는 강력한 앙상블 기법입니다. 본 연구에서는 GBDT의 자연적 특성으로 인해 발생하는 증분 학습의 도전과제를 제시하고, 이를 해결하기 위한 새로운 프레임워크를 제안합니다. 특히, 온라인 학습에서 데이터 추가 및 제거의 효율성을 높이기 위해 하이퍼파라미터 간의 관계를 이론적으로 설명하고, 정확도와 비용 간의 균형을 맞출 수 있는 방법을 제공합니다. 또한, 오픈 소스 구현을 통해 연구 결과의 재현성을 확보하였습니다.

- **Performance Highlights**: 공공 데이터셋에 대한 실험 결과, 제안한 온라인 학습 프레임워크와 최적화 기법은 기존의 GBDT에 비해 뛰어난 효율성을 보여줍니다. 이 연구는 모델의 성능을 유지하면서도 데이터의 추가 및 삭제를 신속하게 처리할 수 있는 방법을 제공하며, 이는 실제 애플리케이션에서의 활용 가능성을 높입니다. 특히, 백도어 공격(backdoor attack) 실험을 통해 모델의 보안성을 개선하는 데도 효과적임을 확인하였습니다.



### Adversarial Reasoning at Jailbreaking Tim (https://arxiv.org/abs/2502.01633)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 취약점을 탐구하기 위해 새로운 방법론을 제시하고 있습니다. 특히, adversarial reasoning 접근법을 소개하여 test-time computation(테스트 시간 계산)을 통해 자동으로 jailbreaking을 구현하고, 이는 다양한 LLM에 대해 SOTA(최신 기술 동향) 공격 성공률을 달성하는 데 기여합니다. 이러한 접근 방식은 LLM의 안전성 탐구를 위한 새로운 패러다임을 마련합니다.

- **Technical Details**: 이 문서는 Adversarial Reasoning이라는 프레임워크를 통해 LLM의 guardrails(가드레일)을 우회하는 방법을 설명합니다. 이 과정은 세 가지 주요 단계인 reasoning, verify 및 search로 구성되며, 각각의 단계에서 목표 LLM의 응답에 기반한 손실 함수를 활용합니다. 이를 통해 논리적 사고를 구성하고, 각 단계를 평가하며, 효율적인 검색을 수행하여 보다 세밀한 신호를 활용해 LLM의 약점을 탐색합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 prompt-space 공격에서 SOTA 공격 성공률을 달성하고, adversarial하게 훈련된 LLM을 대상으로 할 때 특히 두각을 나타냅니다. 적은 자원으로도 강력한 성능을 보이며, 여러 차례의 공격이 이루어지는 multi-shot transfer attack 시나리오에서도 기존 방법보다 뛰어난 결과를 기록했습니다. 이러한 성과는 논문의 방법론이 모델에 의해 최적화된 test-time computation을 크게 활용할 수 있음을 보여줍니다.



### Learning to Generate Unit Tests for Automated Debugging (https://arxiv.org/abs/2502.01619)
Comments:
          First two authors contributed equally. Dataset and Code: this https URL

- **What's New**: 최근 논문에서는 Unit tests (UTs)가 코드 정확성을 평가하고 코드 디버깅에 도움을 주는 자동화된 테스트 생성의 중요성을 다루고 있습니다. UT를 효과적으로 생성하여 오류를 드러내고, 예상 출력을 제공하는 시스템인 UTGen을 제안합니다. 이를 UTDebug라는 디버깅 파이프라인과 통합하여, 모델이 효과적으로 디버깅을 수행할 수 있도록 지원합니다.

- **Technical Details**: UTGen은 LLMs가 주어진 코드와 작업 설명을 바탕으로 오류를 드러내는 unit test inputs와 그에 대한 예상 출력을 생성하도록 훈련됩니다. 이 시스템은 여러 개의 생성된 UT를 기반으로 수정 사항을 확인하고 수정하여 과적합을 방지하며, UT의 출력 정확도를 높이는 방법을 모색합니다. UTGen의 성능은 7.59%로 기존 UT 생성 기준과 비교하여 우수함을 입증합니다.

- **Performance Highlights**: UTDebug를 활용했을 때, UTGen의 유닛 테스트에서 받은 피드백은 Qwen-2.5 7B 모델의 HumanEvalFix 및 MBPP+ 디버깅 세트에서 pass@1 정확도를 각각 3%와 12.35% 향상시켰습니다. 이는 모델 이용 시 UT 생성의 중요성과 효과적인 코드 디버깅의 가능성을 보여줍니다.



### A Probabilistic Inference Approach to Inference-Time Scaling of LLMs using Particle-Based Monte Carlo Methods (https://arxiv.org/abs/2502.01618)
- **What's New**: 이 논문은 기존의 탐색 기반 보상 모델에 의존하는 추론 시간 스케일링 방안 대신, 이를 확률적 추론(probabilistic inference) 문제로 전환하는 새로운 방법을 제안합니다. 특히, 이 방식은 근사된 가능성을 기반으로 상태 공간 모델의 전형적인 집합을 탐색하는 샘플링 기반 기법을 활용하여 보상 모델의 불완전성으로 인한 오류의 영향을 줄입니다. 이를 통해, 학습 가능한 매개변수가 적은 작은 모델이 대규모 모델과 유사한 성능을 낼 수 있는 가능성을 제공합니다.

- **Technical Details**: 이 방법은 확률적 추론을 위해 설계된 입자 기반 몬테카를로(particle-based Monte Carlo) 알고리즘을 적응시킵니다. 입자 필터링(particle filtering) 기법을 사용하여 보상 모델의 불완전성을 감안하면서 솔루션의 후보 다양한 집합을 유지합니다. 이 방법은 관찰된 증거에 따라 가중치를 반복적으로 업데이트하여, 보상 모델이 불완전하더라도 강력한 스케일링 성능을 보장합니다.

- **Performance Highlights**: 제안된 방법은 MATH500 및 AIME 2024 데이터 세트에서 이전 탐색 기반 방법들보다 4~16배 더 빠른 스케일링 성능을 보여줍니다. Qwen2.5-Math-1.5B-Instruct 모델은 4회의 롤아웃으로 GPT-4o의 정확도를 초과할 수 있으며, Qwen2.5-Math-7B-Instruct는 32회의 롤아웃으로 o1 수준의 정확도를 달성합니다. 이는 저자원 장치에서도 고급 AI 접근성을 높이는 데 기여할 수 있습니다.



### Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges (https://arxiv.org/abs/2502.01612)
- **What's New**: 이 논문에서는 대규모 언어 모델이 길이 일반화 (length generalization)와 복잡한 문제 해결에 어려움을 겪는다는 점을 개선하기 위해, 스스로의 해결책 (self-solutions)을 반복적으로 생성하고 학습하는 방법을 제시합니다. 기존의 transformer 아키텍처를 유지하면서, 모델이 더 어려운 문제를 점진적으로 해결할 수 있는 자기 개선 (self-improvement) 접근 방식을 소개합니다.

- **Technical Details**: 저자들은 산술 (arithmetic), 문자열 조작 (string manipulation), 미로 해결 (maze solving) 등 다양한 작업에서 자기 개선 기법을 적용했습니다. 본 방법은 모델이 초기 훈련 분포 (training distribution)에서 벗어난 문제를 해결할 수 있게 하며, 10자리 덧셈에서 100자리 덧셈으로의 일반화 (generalization) 성능을 보여줍니다. 잘못된 예제를 필터링함으로써 훈련 라운드마다 분포 외 (out-of-distribution) 성능에서 기하급수적으로 개선될 수 있음을 관찰했습니다.

- **Performance Highlights**: 사전 훈련된 모델로부터 시작하는 경우, 여러 작업에 대한 자기 개선 과정이 크게 가속화된다는 점도 강조합니다. 또한, 위치 임베딩 (positional embeddings)이나 모델 아키텍처에 변경 없이도 체계적인 약한-강한 커리큘럼 (weak-to-strong curricula)을 통해 논리적 외삽 (logical extrapolation)을 가르칠 수 있는 가능성을 보여줍니다.



### Reinforcement Learning for Long-Horizon Interactive LLM Agents (https://arxiv.org/abs/2502.01600)
- **What's New**: 이 논문은 대화형 디지털 에이전트(Interactive Digital Agents, IDAs)가 API를 이용하여 사용자 요청에 대한 작업을 수행하는 새로운 방법을 제시합니다. 특히, 강화 학습(Reinforcement Learning, RL)을 활용하여 IDAs를 목표 환경에서 직접 훈련시키는 접근 방식을 도입했습니다. 이 연구는 IDAs가 상태 기반의 다중 도메인, 다중 앱 환경에서 API 호출을 통해 상호작용하는 처음의 사례로, RL의 효과성을 밝히고 있습니다.

- **Technical Details**: 제안된 방법은 LOOP라는 기술로, 이는 부분 관찰 가능한 마르코프 결정 프로세스(Partially Observable Markov Decision Process)를 형식화하여 훈련을 실시합니다. LOOP는 가치 네트워크(value network)를 사용하지 않으며, 메모리에서 기본 대형 언어 모델(Large Language Model, LLM)의 단일 복사본만 유지합니다. 이로 인해 메모리 효율성이 높아지고, 단일 LLM을 미세 조정하는 것과 같은 방식으로 구현됩니다.

- **Performance Highlights**: 32억 개의 파라미터를 가진 에이전트가 LOOP 방식으로 AppWorld 환경에서 훈련되었으며, OpenAI의 o1 에이전트보다 9% 포인트(15% 상대적인 비율) 높은 성능을 보였습니다. 이는 IDAs가 API 문서를 참고하고, 부당한 가정을 피하며, 잘못된 정보를 최소화하고, 실패에서 회복하는 학습 과정을 효과적으로 수행함을 보여줍니다.



### Improving Transformer World Models for Data-Efficient RL (https://arxiv.org/abs/2502.01591)
- **What's New**: 이번 연구에서는 모델 기반 강화 학습(Model-Based Reinforcement Learning, MBRL)의 새로운 접근 방식을 제시하며, Craftax-classic 벤치마크에서 뛰어난 성과를 달성했습니다. 이 방법은 1M 환경 스텝 후 67.4%의 보상을 기록하여 DreamerV3의 53.2%를 초과하였으며, 인간 성과인 65.0%를 처음으로 초과했습니다. 이 연구는 CNN과 RNN으로 구성된 새로운 정책 아키텍처를 통해 SOTA 모델-프리(baseline) 기준 모델을 구축한 후, 세 가지 주요 개선 사항을 추가하였습니다.

- **Technical Details**: 연구팀은 'Dyna with warmup', 'nearest neighbor tokenizer', 그리고 'block teacher forcing'을 포함한 세 가지 주요 요소를 MBRL 설정에 추가했습니다. 'Dyna with warmup'은 실제 및 상상 데이터를 모두 사용하여 정책을 훈련하는 방식입니다. 'nearest neighbor tokenizer'는 이미지 패치를 개선하여 변환기 기반 세계 모델(transformer world model, TWM) 입력 생성을 최적화하고, 'block teacher forcing'은 다음 타임스텝의 미래 토큰에 대해 함께 추론할 수 있게 합니다.

- **Performance Highlights**: 제안된 방법을 통해 연구팀은 1,111M 환경 스텝으로 Craftax-classic 환경에서 67.42%의 보상과 27.91%의 점수를 달성하였습니다. 이는 이전 SOTA 모델에서의 53.20% 및 19.4% 점수를 크게 향상시킨 결과입니다. 연구진은 성과 평가를 위해 두 가지 지표, 즉 보상과 점수를 모두 사용하여 이전 연구들과의 비교를 용이하게 하였습니다.



### Verbalized Bayesian Persuasion (https://arxiv.org/abs/2502.01587)
Comments:
          63 pages, 21 figures

- **What's New**: 이 논문에서는 베이esian persuasion (BP)의 새로운 프레임워크를 제안하여, 기존의 게임 이론 및 기계 학습 접근 방식의 한계를 극복하고 LLMs(대규모 언어 모델)를 활용합니다. 특히, LLMs를 보낸 사람과 수신자로 사용하여 인간 대화가 포함된 실제 게임에 BP를 매핑합니다. 이를 통해 ID(정보 디자인) 분야에서의 새로운 응용 가능성을 제시합니다.

- **Technical Details**: 제안된 접근 방식은 제너럴라이즈드 평형 찾기 알고리즘을 사용하여 LLM과 게임 솔버를 결합하여 효율적으로 게임을 해결합니다. 이러한 알고리즘은 정보 모호화(information obfuscation), 언어적 의무(verbalsed commitment) 및 순응 제약(obedience constraints)과 같은 기법으로 보강됩니다. 이 섹션에서는 전통적인 BP 문제에 대한 베이즈 상관 평형(BCE)을 도출하고, 에이전트의 전략과 보상 등을 제시합니다.

- **Performance Highlights**: 수치 실험을 통해 추천서, 법정 상호작용, 법 집행과 같은 다양한 대화 시나리오에 대한 성과를 검증합니다. 이 프레임워크는 고전적인 BP에서의 이론적 결과를 재현하고, 복잡한 자연언어 및 다단계 상황에서 효과적인 설득 전략을 발견할 수 있습니다. 특히 교수와 HR, 검찰과 재판관, 경찰과 운전자 간의 상호작용을 모델링하여 각각의 최적 결과를 도출해냈습니다.



### Next Steps in LLM-Supported Java Verification (https://arxiv.org/abs/2502.01573)
Comments:
          Accepted to NSE 2025, 1st International Workshop on Neuro-Symbolic Software Engineering (ICSE Workshop), 6 pages, 3 figures

- **What's New**: 최근 연구에서는 Large Language Models (LLMs)가 코드 생성 도구로 적합할 뿐만 아니라, 주석 기반 코드 사양(annotation-based code specifications)을 생성할 수 있는 능력도 있다는 것을 보여주었습니다. 이 논문은 이러한 방법론을 확장하여 대규모 소프트웨어 시스템에 대한 증명 가능한 정확성 보장을 도출할 수 있는 가능성을 탐구하고 있습니다. 특히, LLM이 생성한 솔루션을 검증하기 위한 엄격한 도구세트를 통해 올바른 사양 주석을 신뢰성 있게 이끌어낼 수 있다는 초기 결과를 제공합니다.

- **Technical Details**: 이 연구는 Java 프로그램의 사양을 정의하는 Java Modelling Language (JML)을 사용합니다. JML은 JavaDoc와 유사한 주석을 통해 계약 기반의 Java 메서드 및 클래스 사양을 가능하게 합니다. KeY 도구를 활용하여, Java 프로그램이 주어진 JML 사양을 준수하는지를 증명할 수 있으며, 이 과정에서 LLM이 보조적인 사양을 생성하도록 지도합니다. 연구진은 LLM의 주석 생성 능력을 평가하기 위해 다양한 명령어(prompts) 양식을 탐구하고 있습니다.

- **Performance Highlights**: 실험 결과, 피드백 단계 수가 늘어남에 따라 성공률이 증가하는 경향이 있음을 발견했습니다. 이는 피드백이 LLM의 정확한 사양 생성을 도울 수 있음을 시사하는 긍정적인 결과입니다. 그러나, 현재 제공된 피드백이 LLM이 생성한 사양을 수정하는 데 얼마나 도움이 되는지에 대한 의문이 제기됩니다. 연구팀은 향후 더 큰 규모의 시스템에 적용할 수 있는 방법론을 확장할 계획입니다.



### Visual Theory of Mind Enables the Invention of Writing Systems (https://arxiv.org/abs/2502.01568)
Comments:
          Currently under submission to a non-archival conference, published here with permission from organizers

- **What's New**: 본 연구에서는 인지적 및 문화적 프로세스를 통해 초기 상징 체계의 발달을 조명하기 위해 에이전트가 시각적 마음 이론을 활용하여 행동을 소통하는 모델을 제시합니다. 이 연구는 기존의 비자연적인 방법론의 한계를 극복하고, 인지의 발달 진화를 더욱 잘 이해할 수 있도록 합니다. 이러한 과정에서  'Signification Game'이라 불리는 다중 에이전트 강화 학습 테스트베드를 개발하여 의사소통의 구체적인 기제를 실험적으로 살펴봅니다.

- **Technical Details**:  'Signification Game'은 초기 인류의 커뮤니케이션 도구의 제한을 모방한 행동 공간에서 에이전트가 의사소통을 배울 수 있는 설정을 제공합니다. 에이전트는 환경에서 행동을 취할 때 사용되는 정보 처리 시스템을 기반으로 통신 신호를 해석해야 하며, 이 과정은 그들의 상호작용을 통해 점진적으로 발전합니다. 실험 설계는 동물들의 원시적 의사소통 방식의 이론을 기반으로 하여, 조건된 S-R 행동의 구조를 밝혀냅니다.

- **Performance Highlights**: 연구 결과, 단순한 정책 구조로도 에이전트가 보상을 극대화하며 의사소통을 배울 수 있음을 보여주었으며, 이러한 방식에는 한계가 존재함을 발견했습니다. 특히 인지 신호의 한계로 인해 특정 참조 개념을 소통하는 데 어려움이 발생하는 signification gap이 있음을 밝혔습니다. 그러나 시각적 마음 이론을 활용한 추론 모델이 이러한 gap을 극복할 수 있도록 지원하며, 에이전트들이 빠른 시간 내에 대다수의 참조 개념을 전달할 수 있는 가능성을 열어줍니다.



### MeetMap: Real-Time Collaborative Dialogue Mapping with LLMs in Online Meetings (https://arxiv.org/abs/2502.01564)
Comments:
          CSCW2025 Accepted

- **What's New**: 이번 연구에서는 기존의 온라인 회의에서의 대화 방법을 개선하기 위해 LLMs를 활용한 새로운 시스템, MeetMap을 제안했습니다. MeetMap은 사용자가 회의 중 실시간으로 대화 지도를 구성할 수 있도록 지원합니다. 기존의 전통적인 방법에 비해 사용자들이 선호하는 대화 구조를 구축할 수 있는 기회를 제공합니다.

- **Technical Details**: MeetMap은 두 가지 버전으로, Human-Map과 AI-Map을 통해 사용자에게 다양한 수준의 AI 지원을 제공합니다. Human-Map에서는 AI가 대화 요약을 생성하고 사용자가 이를 기반으로 대화 지도를 생성합니다. 반면, AI-Map에서는 AI가 생성한 초안을 사용자들이 자유롭게 수정할 수 있도록 합니다.

- **Performance Highlights**: MeetMap의 실험 결과, 사용자들은 전통적인 노트 작성 방법보다 MeetMap이 실시간으로 내용을 추적하고 후속 논의를 촉진하는 데 더 유용하다고 평가했습니다. Human-Map에서 사용자는 대화 맵을 적극적으로 구성하는 과정에서 내용 이해도가 높아졌고, AI-Map은 구성된 출력을 통해 후속 회의 리뷰를 용이하게 했습니다.



### Search-Based Adversarial Estimates for Improving Sample Efficiency in Off-Policy Reinforcement Learning (https://arxiv.org/abs/2502.01558)
Comments:
          Submitted to International Conference on Machine Learning 2025. Currently under peer-review

- **What's New**: 이 논문에서는 Deep Reinforcement Learning (DRL) 분야에서 오래된 문제인 샘플 비효율(sample inefficiency)을 해결하기 위한 새로운 방법으로 Adversarial Estimates를 제안합니다. 이 접근법은 제한된 양의 인간 기록 샘플을 활용하여 DRL 알고리즘의 학습을 향상시키는데, 단 5분의 인간 데이터를 사용합니다. 연구 결과, Adversarial Estimates로 학습된 알고리즘이 기존 버전보다 더 빨리 수렴하는 것을 보여줍니다.

- **Technical Details**: 본 연구는 인과관계 가능한 환경에서의 DRL 알고리즘의 수렴 속도를 개선하는 데 중점을 두고 있습니다. 기존 방법들과 달리 사전 훈련(pre-training)이 필요하지 않으며, 전문가 수준의 고품질 데이터에 의존하지 않고도 샘플 효율성을 높이는 방법을 제안합니다. 또한, 정책(policy)의 최적성을 가정하는 기존 접근 방식의 한계를 극복하는 일반적인 손실 패널티 해법을 개발했습니다.

- **Performance Highlights**: 실험 결과, 제안된 Adversarial Estimates 접근법이 해당 DRL 알고리즘의 수렴 시간을 획기적으로 단축시킴을 확인했습니다. 이 방법은 또한 드문 보상(sparse reward) 상황에서도 적합한 학습과 성과를 이끌어낼 수 있는 가능성을 제시합니다. 따라서 다양한 DRL 알고리즘에 적용할 수 있는 유연성을 지니고 있으며, 향후 연구에 있어 중요한 기초를 제공할 것으로 예상됩니다.



### Query Brand Entity Linking in E-Commerce Search (https://arxiv.org/abs/2502.01555)
- **What's New**: 이 연구에서는 전자 상거래 검색 쿼리를 위한 브랜드 엔티티 링크(linking) 문제를 다룹니다. 새로운 접근 방식으로, 두 단계로 구성된 프로세스와 엔드 투 엔드 방식이 제안되었으며, 각 방식이 브랜드 엔티티를 추출하는 데 효과적입니다. 특히, 엔드 투 엔드 모델을 도입하여 입력 텍스트로부터 직접적으로 대상을 찾아내는 방법이 있습니다.

- **Technical Details**: 연구는 메타TS-NER(MetaTS-NER)라는 다국어 모델을 활용하여 브랜드명 인식을 수행하며, 이는 다중 레이블 분류에 기반하여 제품 검색 쿼리에서 브랜드를 감지합니다. 두 단계 프레임워크는 NER 모델을 통해 브랜드명을 추출한 후, 면목에 기반한(match) 추출 과정을 거칩니다. 또한, PECOS 도구를 활용하여 방대한 클래스(brand) 공간에서의 다중 클래스 분류 문제 해결도 시도합니다.

- **Performance Highlights**: 연구에서 제안하는 기법은 오프라인 벤치마크 및 온라인 A/B 테스트를 통해 성능을 검증하였습니다. 세부적으로는 브랜드 검색 쿼리에 대한 브랜드 엔티티 예측에서 높은 정확도를 보여주었으며, 짧은 쿼리 길이(평균 2.4 단어)를 효과적으로 처리할 수 있는 방법론을 제시합니다. 연구 결과는 제품 검색의 효율성을 향상시킬 것입니다.



### FireCastNet: Earth-as-a-Graph for Seasonal Fire Prediction (https://arxiv.org/abs/2502.01550)
- **What's New**: 이 연구에서는 기후 변화(C climate change)가 화재 날씨 조건에 미치는 영향을 반영하여, 숲 화재의 예측을 위한 새로운 접근 방식을 제시합니다. SeasFire라는 포괄적인 글로벌 숲 화재 데이터셋을 활용하여, 기후, 식생(vegetation), 해양 지수(oceanic indices), 그리고 인간 관련 변수(human-related variables)를 포함한 계절적인 화재 예측을 수행합니다.

- **Technical Details**: 새로운 아키텍처인 FireCastNet은 3D convolutional encoder와 Graph neural networks를 결합하여 구축되었습니다. 이 모델은 숲 화재에 이르는 다양한 공간적(spatial) 및 시간적(temporal) 맥락을 포착하도록 훈련되었습니다. 또한, 민감도 분석을 통해 예측 시간 지평선(time horizons)에 따른 불이 탄 지역의 예측 효과성(effectiveness)을 평가합니다.

- **Performance Highlights**: 연구 결과, 깊은 학습(deep learning) 모델이 계절 화재 예측에서 유망함을 보여줍니다. 입력 시계열의 길이가 증가함에 따라 예측의 강건성(robustness)이 향상되고, 공간 정보를 통합하여 화재의 시공간적(spatio-temporal) 동태를 포착하는 것 또한 성능을 강화합니다. 더 나아가 긴 예측 지평선에서 성능을 개선하기 위해서는 넓은 수용 필드(receptive field)가 필요하다는 사실이 밝혀졌습니다.



### VideoRAG: Retrieval-Augmented Generation with Extreme Long-Context Videos (https://arxiv.org/abs/2502.01549)
- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG) 개념을 비디오 콘텐츠에 처음으로 적용한 VideoRAG 프레임워크를 소개합니다. 이는 복잡한 멀티모달 비디오 지식을 처리하고 이해하기 위해 특별히 설계된 모델로, 기존의 텍스트 기반 접근 방식의 한계를 넘는 혁신적인 접근 방식입니다. VideoRAG는 두 개의 상호 연결된 구성 요소인 Multi-Modal Video Knowledge Indexing framework와 Knowledge-Grounded Multi-Modal Retrieval paradigm을 통해 멀티모달 비디오 내용을 효과적으로 조직하고 인덱싱할 수 있게 합니다.

- **Technical Details**: VideoRAG의 핵심 혁신은 그래프 기반의 텍스트 지식 기초와 멀티모달 컨텍스트 인코딩을 통합한 이중 채널 아키텍처에 있습니다. 이 프레임워크는 다양한 비주얼 특징을 효율적으로 보존하고, 여러 비디오에 걸쳐 정확한 지식 그래프를 구축하여 비디오 간의 의미적 의존성을 유지합니다. 또한, 비디오 지식 검색을 효율적으로 수행하기 위해 LLM 기반의 키워드 추출과 비전-언어 모델을 기반으로 한 텍스트 그라운딩을 결합한 이중 단계의 콘텐츠 추출 프로세스를 활용합니다.

- **Performance Highlights**: VideoRAG는 160개 이상의 비디오로 구성된 LongerVideos 벤치마크에서 종합적인 실증 평가를 통해 기존 RAG 대안 및 긴 비디오 이해 방법과 비교해 상당한 성과를 보여줍니다. 이 프레임워크는 교육 콘텐츠 분석, 미디어 아카이빙, 비디오 기반 지식 추출과 같은 분야에서 비디오 이해력을 크게 향상시키며, 기존 단일 비디오에 제한된 기존 데이터셋을 넘어서는 지식을 제시합니다. 실험 결과와 사례 연구를 통해 VideoRAG의 실질적인 응용 가능성을 밝혀내며, 비디오 간 이해 향상에 기여하는 새로운 가능성을 열었습니다.



### What is a Number, That a Large Language Model May Know It? (https://arxiv.org/abs/2502.01540)
Comments:
          16 pages, 8 figures

- **What's New**: 본 연구에서는 대형 언어 모델(LLMs)이 숫자와 문자열 표현을 혼합하여 학습하는 방식을 탐구합니다. 숫자가 문맥에 따라 다르게 해석될 수 있는 상황에서 이러한 이중성(immediate duality)이 모델의 대표성(representation)와 관련된 다양한 결과를 초래하는지에 대한 의문을 제기합니다. 새로운 유사성 기반 프롬프트 기법을 통해 LLMs가 숫자 쌍 간의 유사성을 어떻게 판단하는지 확인하고, 그 과정에서 나타나는 혼합된 표현을 탐구합니다.

- **Technical Details**: 저자들은 심리학 및 인지 과학에서 유래한 유사성 판단(simiarity judgments) 기법을 사용하여 최신 LLMs에서 숫자 쌍에 대한 유사성을 측정했습니다. 이 방법을 통해 Levenshtein 편집 거리(Levenshtein edit distance)와 Log-Linear 숫자 거리(numerical Log-Linear distance)의 조합이 모델들 간의 숫자 표현을 효율적으로 설명함을 보여줍니다. 이러한 접근은 모델의 내부 구조를 직접 참조하지 않고도 사용 가능하며, 프로빙(probing) 기법과 결합하여 잠재 임베딩(latent embedding)의 구조를 평가할 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 이 연구에서 얻어진 중요한 발견은 모든 모델이 숫자 및 문자열 표현 간의 혼합 효과를 경험한다는 것입니다. 특정 맥락(int() vs. str())을 통해 이 혼합을 줄일 수는 있지만 완전히 제거할 수는 없음을 확인했습니다. 이러한 현상이 실제 결정 시나리오에서도 나타나, 문자열 편향(string bias)이 잘못된 결과를 초래하는 방식에 대한 통찰을 제공합니다. 이러한 결과는 LLM이 숫자와 문자열의 표현 간의 긴장을 어떻게 navigat할 필요가 있는지에 대한 중요한 정보를 제공합니다.



### Preference Leakage: A Contamination Problem in LLM-as-a-judg (https://arxiv.org/abs/2502.01534)
Comments:
          17 pages, 8 figures

- **What's New**: 이번 연구에서는 LLM(큰 언어 모델) 기반의 데이터 주석 방법에서 발생할 수 있는 오염 문제인 '선호 누출(preference leakage)'을 다룹니다. 주로 LLM을 평가자로 사용할 때 합성 데이터 생성기와 평가자 간의 관련성으로 인한 것입니다. 이러한 새로운 모델 개발 패러다임의 효과에 대한 연구가 부족했던 점을 지적하며, 선호 누출이 모델 훈련 및 평가의 효율성을 높이는 과정에서 주의해야 할 문제임을 강조합니다.

- **Technical Details**: 연구에서는 데이터 생성기 LLM과 평가자 LLM 간의 세 가지 일반적인 관련성을 정의합니다. 이들은 동일한 모델, 상속 관계가 있는 모델, 같은 모델 계열에 속하는 경우입니다. 이러한 관계에 따라 여러 LLM 기반의 실험을 통해 평가자가 자신과 관련된 학생 모델에 대한 편견이 존재함을 실증적으로 확인하였습니다.

- **Performance Highlights**: 선호 누출 문제는 기존에 알려진 LLM-as-a-judge 시나리오의 편향 문제보다 검출하기 어려운 만연한 문제임을 알게 되었습니다. 이러한 결과들은 LLM-as-a-judge 분야에서 선호 누출이 광범위하고 도전적인 문제임을 시사하며, 향후 연구에서 필수적으로 고려해야 할 요소입니다.



### Transformers trained on proteins can learn to attend to Euclidean distanc (https://arxiv.org/abs/2502.01533)
- **What's New**: 이번 논문에서는 전통적인 Transformer가 구조 모델과 독립적으로 작동할 수 있음을 보여줍니다. 특히, 3D 데이터 처리에 있어서 Transformer가 구조 정보를 스스로 학습할 수 있는 가능성을 탐구하고 있습니다. AlphaFold3와 같은 구조적 확산 모델과의 관계를 통해 이러한 이론적 배경을 설명합니다.

- **Technical Details**: Transformer가 3D 데이터를 처리하기 위해 주의(attention) 필터링을 3D Gaussian으로 학습할 수 있는 과정을 이론적으로 설명합니다. 또한, 시뮬레이션된 3D 포인트와 단백질의 마스크된 토큰 예측 맥락에서 이 이론을 검증합니다. 이러한 방법론은 Transformer의 구조적 임베딩을 활용한 새로운 접근을 제시합니다.

- **Performance Highlights**: 구조 정보를 포함하여 사전 훈련된 단백질 Transformer 인코더는 하위 작업에서 성능이 향상되는 것을 보여줍니다. 커스텀 구조 모델보다 더 나은 성능을 발휘하며, 이로 인해 Transformer를 혼합 구조-언어 모델로 사용할 수 있는 가능성이 열립니다.



### Efficiently Integrate Large Language Models with Visual Perception: A Survey from the Training Paradigm Perspectiv (https://arxiv.org/abs/2502.01524)
Comments:
          28 pages, 3 figures

- **What's New**: 이번 논문에서는 Vision-Language Large Language Models (VLLMs)에서 시각 모달리티의 통합을 위한 새로운 교육 패러다임을 제시합니다. 연구의 주요 초점은 MoDALITY Integrators (MIs)와 함께 두 가지 주요 조정 과정인 Single-stage Tuning과 Two-stage Tuning입니다. 이를 통해 연구자들은 LLMs의 매개변수 효율성을 유지하면서 성능을 향상시킬 수 있는 방법을 제시하고 있습니다.

- **Technical Details**: 본 연구에서는 다양한 LLM 아키텍처와 함께 Parameter-Efficient Adaptation (PEA) 기법을 분류합니다. Single-stage Tuning과 Two-stage Tuning 방식은 각각 고유한 효율성 동기를 가지고 있으며, Direct Adaptation 방식은 자원의 효율적인 소비를 강조합니다. 각 교육 패러다임에 대해 독특한 매개변수 최적화 전략을 제시하며, 다양한 비전 인코더의 아키텍처를 포괄적으로 다룹니다.

- **Performance Highlights**: 논문은 34개의 VLLMs를 조사한 결과, 최신 VLLMs에서의 Two-stage Tuning과 함께 여러 모델의 성능을 비교 분석합니다. 실험 결과는 Direct Adaptation 접근 방식의 주요 개발과 효과성을 입증하며, 다양한 분야에서 비전 모달리티 통합의 효율성을 끌어올리는 데 기여할 수 있는 지침을 제공합니다. 연구의 결과는 연구자와 실무자들이 LLMs에 비전 모달리티를 효율적으로 통합하는 데 유용한 통찰력을 제공합니다.



### Toward Task Generalization via Memory Augmentation in Meta-Reinforcement Learning (https://arxiv.org/abs/2502.01521)
- **What's New**: 이번 연구에서는 메모리 증강(memory augmentation)이라는 새로운 접근법을 도입하여 강화 학습에서 태스크 일반화를 개선하는 방법을 제시합니다. 이를 통해 에이전트는 사전 정의된 태스크 집합에 대해 훈련을 받은 후에도, 보이지 않는 태스크에 대해 제로샷(zero-shot) 일반화를 이룰 수 있습니다. 메모리 메커니즘을 활용하여 맥락-aware 정책 적응을 가능하게 하여, RL 에이전트가 훈련 중에 명시적으로 노출되지 않은 다양한 환경에서도 강건한 성능을 발휘하도록 합니다.

- **Technical Details**: 연구에서는 부분 관측 마르코프 결정 프로세스(Partially Observable Markov Decision Process, POMDP) 모델을 사용하여 문제를 정의합니다. 이는 상태, 관측, 액션을 포함한 여러 요소를 다루며, 에이전트가 특정 태스크의 맥락을 이해하고 추론하도록 요구합니다. 메모리 증강 접근법은 경험을 확대하여 기존 훈련 경험을 활용하고, RNN을 통해 과거 상호작용에서 작업 맥락을 캡처합니다.

- **Performance Highlights**: 실험 결과, 메모리 증강 정책은 기존의 ID 태스크에서도 안정적인 성능을 유지하면서 보이지 않는 OOD 태스크에 대해 효과적으로 일반화되는 것으로 나타났습니다. 특히, 이 접근법은 무작위화 기반 정책에 비해 높은 샘플 효율성을 달성하며, 사족 로봇을 통한 실제 환경에서의 성능을 입증했습니다. 로봇은 조인트 고장 상황에서도 목표를 효과적으로 추적하며 적응하는 모습을 보여주었습니다.



### Regularized interpolation in 4D neural fields enables optimization of 3D printed geometries (https://arxiv.org/abs/2502.01517)
- **What's New**: 본 논문에서는 지정된 속성으로 기하학을 생성할 수 있는 제조 공정의 중요성을 강조합니다. 특히 3D 프린팅에서 복잡한 디자인과 재료 흐름 관리의 어려움이 제기됩니다. 연구자들은 뉴럴 필드(neural fields)를 활용하여 새로운 정규화 전략을 도입함으로써 공정 매개변수가 변화하더라도 기하학적 예측을 더 정확하게 할 수 있는 방법을 제시합니다.

- **Technical Details**: 연구에서는 뉴럴 필드를 사용하여 3D 프린팅의 기하학적 충실도를 최적화하였습니다. 각 뉴럴 필드는 물체의 볼륨을 인코딩하며, 새로운 기울기 기반의 인터폴레이션 정규화(gradient-driven interpolation regularization, GDIR) 전략을 적용하여 작은 매개변수 변동이 출력 기하학에 미치는 영향을 최소화합니다. 이를 통해 사용자가 기대하는 기하와 실제 생산된 기하 간의 친밀도를 높일 수 있습니다.

- **Performance Highlights**: 이 연구의 결과는 3D 프린팅 프로세스에서 매개변수 조정이 기하학적 품질을 향상시키는 데 효과적임을 보여줍니다. 3D 프린팅 시스템과 컴퓨터 단층 촬영(CT) 스캐너를 활용하여 사용자 지정 데이터셋을 생성하고, 다양한 유량(flow rate) 설정에서 네 가지 형상을 반복적으로 제조하여 데이터를 수집했습니다. 이를 통해 적절한 유량이 디자인의 특정 기능에 미치는 영향을 관찰하며, 기존의 동적인 프로세스 매개변수 관리의 필요성을 강조합니다.



### MoireDB: Formula-generated Interference-fringe Image Datas (https://arxiv.org/abs/2502.01490)
- **What's New**: 이번 연구에서는 기존 데이터 증강 기법인 PixMix의 한계를 극복하기 위해 MoireDB라는 새로운 방식을 제안합니다. MoireDB는 수학적 공식을 활용하여 생성된 간섭 무늬 이미지 dataset으로, 저작권 문제를 제거하고 데이터셋 조립 비용을 줄이며 이미지 인식 모델의 견고성을 향상시키는 것이 목표입니다. 이러한 MoireDB를 통해 실제 세계의 열악한 환경에도 강한 이미지 인식 모델을 만들 수 있게 됩니다.

- **Technical Details**: MoireDB는 공식적으로 생성된 Moiré 이미지를 저장하는 데이터베이스로, 기존의 Fractal arts와 feature visualizations (FVis) 대신 사용됩니다. 연구자들은 Moiré 이미지를 데이터 증강 시 사용할 때 모델의 견고성이 향상된다는 가설을 세웠으며, 이를 통해 ddeep learning 모델을 훈련하여 견고성을 테스트하게 됩니다. MoireDB의 이미지들은 저작권 문제 없이 자동으로 생성되므로, 상업적 사용이 가능합니다.

- **Performance Highlights**: 실험 결과, MoireDB를 통해 증강된 이미지는 전통적인 Fractal arts 및 FVis 기반의 증강 기법보다 뛰어난 성능을 보였습니다. 특히, 이미지 인식 모델은 실제 환경에서의 열악한 조건에서도 더욱 견고하게 작동하는 것으로 확인되었습니다. 이러한 작업은 MoireDB의 확장 가능성과 효과성을 보이며, 객체 인식 모델의 품질을 크게 향상시키는 데 기여할 것으로 예상됩니다.



### Position: Empowering Time Series Reasoning with Multimodal LLMs (https://arxiv.org/abs/2502.01477)
- **What's New**: 본 논문에서는 시간 의존 데이터의 다중 모드 특성을 반영하여, Multimodal Large Language Models (MLLMs)를 활용한 새로운 사고 방식의 필요성을 강조합니다. 이러한 접근법은 기존의 수치 데이터 기반 분석을 넘어, 텍스트, 이미지 및 오디오와 같은 다양한 데이터를 통합하여 더 풍부하고 유연한 추론을 가능케 합니다. 연구자들에게 신뢰성과 해석 가능성을 중심으로 MLLMs의 개발을 촉구하며, 이를 통해 실제 응용에서 더 나은 의사 결정을 지원할 수 있음을 주장합니다.

- **Technical Details**: 시간 시계열 추론은 MLLM이 인간과 유사한 방식으로 시간 시계열 데이터를 처리하고 해석할 수 있는 능력을 나타냅니다. 이는 특정 목표에 국한된 기존 방법들과 달리, 컨텍스트 인식(context-awareness)과 시간 시계열 특성을 통합하여 보다 깊이 있는 통찰을 제공합니다. 이러한 모델 구조는 시계열 데이터와 추가적인 컨텍스트 정보를 동시에 고려하여 분석의 정확성과 해석성을 높입니다.

- **Performance Highlights**: MLLMs를 활용한 새로운 사고 프레임워크는 전통적인 시간 시계열 분석의 한계를 넘어 다양한 새로운 작업에 대한 가능성을 보여줍니다. 이를 통해 MLLMs는 시간 의존 데이터의 다차원적 특성을 활용하여 더 효과적이고 신뢰할 수 있는 결과를 도출할 수 있을 것으로 기대됩니다. 여러 연구 방향과 기술적 도전도 제시되며, 다중 모드 훈련 전략과 새로운 데이터 셋의 중요성이 강조됩니다.



### FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Mod (https://arxiv.org/abs/2502.01472)
Comments:
          Under Review

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 기밀 정보 처리 문제를 해결하기 위한 새로운 방법론인 FALCON(Fine-grained Activation manipuLation by Contrastive Orthogonal uNalignment)을 제안합니다. 기존의 기계 학습에서의 기억 소거(machine unlearning) 기술들은 지식 내에서의 특정한 부분을 분리하기 위한 비효율적인 방법들이 존재했습니다. FALCON은 정보 이론적 접근 방식을 통해 모델의 파라미터 선택을 효율적으로 수행하고, 대전제 기법을 이용하여 효과적인 지식 분리를 달성합니다.

- **Technical Details**: FALCON은 상대적 정보(mutual information)를 사용하여 데이터 간의 의존성을 평가하고, 이를 기반으로 파라미터 선택과 최적화 지침을 제공합니다. 두 가지 주요 메커니즘을 통해 다중 도메인 지식 소거를 수행하며, 이를 방향성 대전제의 적용과 그라디언트 직교 투영(gradient orthogonal projection)으로 이루어집니다. 이 방식은 파라미터의 부분 수정만으로도 효율적인 지식 소거를 달성할 수 있도록 설계되었습니다.

- **Performance Highlights**: FALCON은 광범위한 실험을 통해 다른 기계 학습 모델보다 뛰어난 지식 소거 효과를 보여주었습니다. 이 방법은 모델의 유용성을 유지하면서도 지식 회복 시도에 대한 저항성이 뛰어난 결과를 도출합니다. 또한 FALCON은 다양한 모델에 걸쳐 확장 가능성을 가지고 있으며, 실질적으로 접근할 수 없는 훈련 데이터에 대한 완전한 접근 없이도 작동할 수 있습니다.



### Process Reinforcement through Implicit Rewards (https://arxiv.org/abs/2502.01456)
Comments:
          20 pages. Model&Code&Data available at this https URL

- **What's New**: 본 연구에서는 LLM(대형 언어 모델)의 추론 시간에 효과적인 Dense Process Rewards를 제안합니다. PRIME(Implicit Rewards를 통한 Process Reinforcement)은 고품질의 Dense 보상을 대규모로 효율적으로 수집하고 활용하는 방법을 제시합니다. 기존 방식에서 발생할 수 있는 Reward Hacking 문제를 완화하고, 온라인으로 PRM(프로세스 보상 모델)을 업데이트할 수 있는 가능성을 보여줍니다.

- **Technical Details**: PRIME은 Outcome Labels만 사용하여 Dense Reward 모델을 훈련하는 방법론으로, 기존 방식을 통해 요구되는 복잡한 Label 수집 과정을 간소화합니다. 이 프레임워크는 각 Token(Level) 보상과 Sparse Outcome Rewards의 결합도 가능하게 하여 다양한 RL(강화 학습) 알고리즘과 호환됩니다. 특히, PRIME은 SFT(슈퍼바이즈드 파인 튜닝) 모델에서 초기화하여 길어진 훈련 과정을 간소화합니다.

- **Performance Highlights**: 실험 결과에 따르면, PRIME은 기존의 SFT 모델에 비해 평균 15.1%의 성능 향상을 보였으며, Qwen2.5-Math-7B-Base 모델을 사용하여 수학 문제 해결에 있어 2.5배의 샘플 효율을 달성했습니다. 최종 모델인 Eurus-2-7B-PRIME은 Qwen2.5-Math-7B-Instruct를 7개 주요 벤치마크에서 초과 성능을 기록하였습니다. 또한, PRIME은 약 10%의 기존 데이터로도 결과를 달성함으로써 효율적인 학습 가능성을 입증했습니다.



### Temporal-consistent CAMs for Weakly Supervised Video Segmentation in Waste Sorting (https://arxiv.org/abs/2502.01455)
Comments:
          14 pages, 7 figures

- **What's New**: 이번 연구는 산업 환경에서 강력한 기계 비전 시스템과 딥러닝을 활용한 약한 감독 방식(Weakly Supervised, WS) 접근법의 효과를 보여줍니다. 특히, 이 방법은 비디오 스트림의 의미적 세그멘테이션을 위한 정확한 마스크를 생성할 수 있으며, 연속 프레임 간의 시간적 일관성을 활용하여 객체가 다양한 프레임에 나타날 때의 일관성을 증진시킵니다. 또한, 인공지능 모델의 훈련 과정에서 시간적 일관성을 통합하는 방안을 제안합니다.

- **Technical Details**: 제안된 방법은 먼저 두 개의 카메라를 활용하여 분리 과정 전후의 이미지를 수집합니다. 이러한 이미지를 바탕으로 생성된 saliency map은 주변 프레임의 움직임을 보정하여 같은 객체의 saliency map이 서로 유사하도록 강제합니다. 이 과정에서 Opticl Flow 알고리즘을 사용하여 연속 프레임 간의 모션을 계산하고, 이를 통해 네트워크가 유사한 출력을 생성하도록 합니다. 또한, 기존의 binary classification 문제에서 벗어나, 배경과 객체를 구분하는 세 가지 클래스(classification problem)를 설정하여 좀 더 정교한 세그멘테이션 마스크를 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 정확하고 일관성 있는 세그멘테이션 마스크를 생성함으로써 산업 폐기물 분류 애플리케이션에 적합하다는 것을 입증하였습니다. 이로 인해, 법적 객체와 불법 객체를 세밀한 픽셀 수준의 주석 없이도 구분할 수 있습니다. 이러한 접근법은 기존의 Fully Supervised segmentation 방법과 비교하여 훈련 데이터 수집의 비용과 시간을 대폭 절감할 수 있는 효과적인 방안이 됩니다.



### Simulating Rumor Spreading in Social Networks using LLM Agents (https://arxiv.org/abs/2502.01450)
Comments:
          7 pages, 8 figures

- **What's New**: 본 연구는 소셜 미디어의 확산으로 인해 증가하는 허위정보 문제를 해결하기 위해 새로운 프레임워크를 제안합니다. 이 프레임워크에서는 다양한 LLM(Large Language Model) 기반 에이전트를 사용하여 소셜 네트워크 내에서 루머의 확산을 분석하고 시뮬레이션합니다. 각 에이전트가 다양한 인격과 루머 전파 방식에 따라 행동하도록 설계되어, 루머의 확산 및 전파 평가를 더 효과적으로 수행합니다.

- **Technical Details**: 제안된 방법론은 LLM을 기반으로 한 에이전트를 활용하여 소셜 네트워크의 루머 확산을 시뮬레이션합니다. 각각의 에이전트는 개인의 배경과 성향에 따라 동적으로 변하며, 이들 간의 인터랙션을 통해 루머의 전파 양상을 분석합니다. 두 가지 네트워크 구조를 구축하고, 이를 통해 생성된 100개 이상의 에이전트가 참여하는 시뮬레이션을 실시하여 루머의 확산을 평가합니다.

- **Performance Highlights**: 실험 결과, 다양한 네트워크 구조와 에이전트 행동이 루머 전파에 미치는 영향을 분석하여, 83%의 에이전트에 영향을 미치는 전파가 가능함을 보여줍니다. 이 연구는 LLM 에이전트를 사용한 시뮬레이션이 현실적인 루머 확산을 모델링할 수 있는 가능성을 입증하며, 연구의 기여도가 높습니다.



### SPFFNet: Strip Perception and Feature Fusion Spatial Pyramid Pooling for Fabric Defect Detection (https://arxiv.org/abs/2502.01445)
Comments:
          8 pages, 4 figures, conference

- **What's New**: 본 연구에서는 YOLOv11 기반의 패브릭 결함 검출 모델을 제안하며, 그 과정에서 다중 스케일 컨볼루션을 통한 스트립 결함 인식 강화를 위해 Strip Perception Module (SPM)을 도입합니다. 또한, 스퀴즈-앤-익사이테이션(spatial pyramid pooling Fast, SPPF) 메커니즘을 통합하여 SE-SPPF 모듈을 개선하여 공간적 및 채널 정보를 효율적으로 통합하고 결함 특징 추출을 향상시킵니다. 이와 함께, 스케일 차이와 클래스 불균형 문제를 해결하는 새로운 Focal Enhanced Complete IoU (FECIoU) 메트릭을 제안하여 더욱 효과적인 결함 탐지를 목표로 하고 있습니다.

- **Technical Details**: 연구에서는 SPM을 도입하여 다중 스케일 컨볼루션을 활용하여 스트립 결함 특징을 추출하고, SE-SPPF를 통해 배경과 결함을 보다 잘 구분할 수 있게 합니다. FECIoU 메트릭을 도입하여 대상 박스의 스케일 변화를 동적으로 조정하고, 하드 투 디텍트(hard-to-detect) 인스턴스의 가중치를 조절하여 더욱 적응적이고 효율적인 탐지가 가능하게 합니다. 모델은 고속의 탐지성과 실시간 성능을 유지하며 결함 검출 정확도를 크게 향상시킵니다.

- **Performance Highlights**: Tianchi 데이터셋에서 평균 정확도(mean average precision, mAP)가 0.8-8.1% 개선되었으며, 연구자가 커스터마이징한 데이터셋에서도 1.6-13.2%의 성능 향상을 기록했습니다. 본 모델은 기존의 최첨단 기법들을 초월하는 성능을 보여주며, 특수한 결함 유형을 다루는 데 있어 한 단계 더 나아간 결과를 제시합니다. 실험 결과를 통해, 제안한 방법이 복잡한 배경과 다양한 결함 형태를 성공적으로 처리할 수 있음을 입증했습니다.



### Towards Safer Chatbots: A Framework for Policy Compliance Evaluation of Custom GPTs (https://arxiv.org/abs/2502.01436)
- **What's New**: 이 논문에서는 OpenAI의 Usage Policies에 따라 Custom GPT들을 자동으로 평가하기 위한 확장 가능한 프레임워크를 제시합니다. 특히, 모델을 검색하고 데이터를 수집하는 자동화 기능과 정책 카테고리에 맞춤화된 레드 팀 프롬프트 생성기, 그리고 전체적인 준수를 분석하는 LLM-as-a-judge 기법이 통합되었습니다. 이 프레임워크는 수작업 주석 처리된 데이터셋을 검증하여 정확성을 보장하며, 782개의 Custom GPT를 대규모 연구를 통해 평가했습니다.

- **Technical Details**: 프레임워크는 세 가지 주요 모듈로 구성되어 있습니다: 1) Custom GPT Interactor는 모델을 자동으로 검색하고 메타데이터를 수집합니다; 2) Red-Teaming Prompts Generator는 각 Custom GPT에 대한 맞춤형 프롬프트를 생성합니다; 3) Compliance Assessment 모듈은 LLM-as-a-judge 기법을 사용하여 응답을 분석하고 정책 준수 여부를 평가합니다. 이 과정은 Orchestrator 모듈에 의해 조정되며, JSON 형식으로 결과를 구조화하여 평가 결과를 제공합니다.

- **Performance Highlights**: 연구에서 58.7%의 모델이 비준수로 나타났으며, 이는 GPT 스토어의 검토 및 승인 프로세스의 취약점을 드러냅니다. 특히, 사용자 맞춤화보다는 기본 모델에서 상속받은 행동이 비준수 문제의 주 원인으로 분석되었습니다. 이 결과는 다른 챗봇 플랫폼 및 정책 영역에도 확장 가능한 접근 방식을 제공하며, LLM 기반 시스템의 안전성을 향상시킬 가능성을 제시합니다.



### Structural features of the fly olfactory circuit mitigate the stability-plasticity dilemma in continual learning (https://arxiv.org/abs/2502.01427)
- **What's New**: 이 연구는 지속적인 학습(continual learning)에서 인공신경망이 직면하는 안정성-가소성 딜레마(stability-plasticity dilemma)를 해결하기 위해 최소한의 모델로서 파리의 후각 회로(fly olfactory circuit)를 제시합니다. 이 모델은 현대 기계 학습 방법과 통합할 수 있는 플러그 앤 플레이 구성 요소로 알려져 있으며, 지속적인 냄새 학습을 지원하는 생물학적 전략을 조사합니다. 또한 이 연구는 현재의 지속적 학습 전략의 한계를 극복할 수 있는 생물학적 솔루션을 제공합니다.

- **Technical Details**: 파리의 후각 회로는 생물학적 네트워크의 효율성을 기반으로 한 학습 모듈로서 기능하며, 이를 통해 메모리의 안정성(memory stability)과 학습의 가소성(plasticity)을 동시에 향상시킬 수 있음을 보여주었습니다. 이 모델은 다양한 도전적인 지속적 학습 시나리오에 걸쳐 검증되었으며, 일반적으로 사용되는 데이터 세트에서 효과성을 입증하였습니다. 결과적으로, 이 모델은 기계 학습에서 추가적인 계산 비용을 최소화하면서 지속적 학습을 향상시키는 모듈로 작용합니다.

- **Performance Highlights**: Fly Model은 기존의 지속적 학습 전략들을 초월하여 우수한 성능을 나타냈습니다. 연구 결과, 이 모델은 학습 효율을 높이고 안정성을 제공함으로써 데이터에 대한 적응력을 증가시킵니다. 이는 생물학적 회로가 평생 학습(lifelong learning)에서 어떻게 활용될 수 있는지를 보여주는 우아한 사례가 됩니다.



### Visual Attention Never Fades: Selective Progressive Attention ReCalibration for Detailed Image Captioning in Multimodal Large Language Models (https://arxiv.org/abs/2502.01419)
- **What's New**: 이 논문에서 제안된 SPARC (Selective Progressive Attention ReCalibration) 방법은 기존 멀티모달 대형 언어 모델(MLLMs)의 이미지 캡셔닝 문제를 해결하기 위해 개발되었습니다. 현재의 방법들은 일반적으로 정밀도(precision)를 향상시키지만, 회수율(recall)을 저하시켜, 캡션의 핵심 요소를 놓치는 경우가 많습니다. SPARC는 이러한 한계를 극복하고, 육안 평가 및 자동 평가에서 우수한 성과를 보여줍니다.

- **Technical Details**: SPARC는 시각 토큰의 기여를 강조하기 위해 의도적으로 선택적으로 시각 토큰의 영향을 증폭합니다. 이 방법은 세 가지 주요 원칙 즉, 전체 시각 토큰의 영향 증가가 회수율을 감소시킨다는 점, 시각 주의(attention)가 길어질수록 노이즈(noise)를 더 많이 발생시킨다는 점, 그리고 약해지는 시각 주의를 강화하여 지속시키는 원칙에 기반하고 있습니다.

- **Performance Highlights**: SPARC는 기존의 방법들보다 캡션 품질을 크게 향상시키며, 정밀도와 회수율 모두를 효과적으로 개선합니다. 자동화된 평가와 사람의 평가를 통해, SPARC의 우수성이 입증되었으며, 기존의 방법들이 가진 정밀도와 회수의 절충(trade-off) 문제를 해결하는 데 기여합니다.



### GRADIEND: Monosemantic Feature Learning within Neural Networks Applied to Gender Debiasing of Transformer Models (https://arxiv.org/abs/2502.01406)
- **What's New**: 이 연구에서는 성별 관련 정보를 인코딩하는 단일의 모노세맨틱(feature neuron) 특성 뉴런을 학습하기 위해 모델 기울기를 활용하는 새로운 인코더-디코더 접근법을 제안합니다. 이 방법은 Transformer 기반 언어 모델에서의 성별 편향을 제거할 수 있는 능력을 제공하면서도 모델의 다른 기능을 유지하는 것이 가능함을 보여줍니다. 또한, 기존의 접근법들과 달리 원하는 해석 가능한 의미를 갖는 특성 뉴런을 학습할 수 있도록 합니다.

- **Technical Details**: 제안된 접근법은 인코더-디코더 아키텍처를 기반으로 하며, 성별 정보가 포함된 기울기 정보를 사용하여 특정 특성 뉴런을 학습합니다. 이는 모델의 기울기를 통해 성별과 관련된 스칼라 값을 인코딩하고, 이 스칼라 값을 기반으로 기울기 업데이트를 디코딩하여 성별 편향을 변경할 수 있도록 합니다. 이 방식은 Gradiend와 INLP를 결합하여 성별 편향 제거에 있어 최신 기술을 달성하는 데 기여합니다.

- **Performance Highlights**: 이 연구를 통해 성별 편향 제거를 위해 기울기 기반 특성 학습의 가능성을 입증하였으며, 여러 인코더 전용 모델에서 효과성을 보여주었습니다. 목표한 특성 뉴런을 학습함으로써 성별 편향을 변화시킬 수 있다는 두 가지 가설을 검증하였고, 이를 통해 성별 편향을 성공적으로 수정하며 모델의 다른 성능을 유지할 수 있음을 입증하였습니다.



### AdaSVD: Adaptive Singular Value Decomposition for Large Language Models (https://arxiv.org/abs/2502.01403)
Comments:
          The code and models will be available at this https URL

- **What's New**: 본 논문에서는 LLM(대형 언어 모델)의 메모리 요구사항을 줄이기 위해 AdaSVD라는 적응형 SVD 기반 LLM 압축 방법을 제안합니다. 기존의 SVD 기반 방법들이 자주 발생했던 성능 저하 문제를 해결하기 위해 adaComp와 adaCR을 도입하여 SVD의 절단 오류를 보상하고 각 변환기 레이어에 적합한 압축 비율을 할당합니다. 따라서 AdaSVD는 리소스 제약이 있는 장치에서의 LLM 배포 가능성을 크게 향상시키는 데 기여합니다.

- **Technical Details**: AdaSVD는 SVD 트렁케이션(절단) 오류를 적응적으로 보정하는 adaComp 기법을 통해 U와 V^T 행렬을 번갈아 업데이트합니다. 또한, adaCR을 통해 각 레이어의 중요도에 기반하여 레이어별 압축 비율을 할당하여 모든 레이어에 동일한 비율을 적용하는 기존 방식과의 차별성을 두고 있습니다. 이러한 방식은 압축 비율이 고정된 상태에서도 성능을 개선하는 데 기여함으로써, 운영 메모리와 GPU 메모리에서의 사용 효율을 높입니다.

- **Performance Highlights**: 다양한 LLM 계열과 평가지표를 바탕으로 한 실험 결과, AdaSVD는 기존의 SOTA SVD 기반 LLM 압축 방법인 SVD-LLM을 초월하여 성능을 현저하게 향상시켰습니다. 이로 인해 압축 모델과 원본 모델 간의 성능 격차가 크게 줄어들어, LLM의 효과적이고 효율적인 실행이 가능해졌습니다. 제안된 방법은 다양한 플랫폼에서의 배포와 사용을 동시에 지원하는 가능성을 높입니다.



### Can message-passing GNN approximate triangular factorizations of sparse matrices? (https://arxiv.org/abs/2502.01397)
- **What's New**: 이 논문은 Graph Neural Networks (GNNs)가 희소 행렬 전처리기를 학습할 때의 근본적인 한계를 탐구합니다. 기존 연구에서 GNNs가 불완전한 분해를 예측하는 데 유망한 결과를 보였으나, 메시지 전달의 지역적 특성이 최적의 전처리에 필요한 비지역적 종속성을 캡처하는 데 장애물이 됨을 증명합니다. 또한, 새로운 벤치마크 데이터세트를 소개하며 GNN 아키텍처의 한계를 실험적으로 보여줍니다.

- **Technical Details**: 희소 대칭 양정수 행렬의 전처리는 수치 선형대수학에서 기본적인 문제로, 이를 통해 반복 방법의 수렴 속도를 높이는 것이 목표입니다. 최근 GNNs를 활용해 이러한 희소 전처리기를 학습하려는 시도가 증가하고 있습니다. 하지만 GNN은 단지 인접한 노드 및 엣지의 정보만을 수집할 수 있으며, 이는 비지역적 종속성을 표현하는 데 충분하지 않음을 보여줍니다.

- **Performance Highlights**: 실험 결과에 따르면, 현재의 GNN 아키텍처는 최적의 희소 전처리기를 근사하는 데 어려움이 있음을 나타내며, Graph Attention Networks와 Graph Transformers와 같은 변형 모델에도 이러한 한계가 존재합니다. 이는 과학 컴퓨팅에서 GNN의 더 광범위한 사용에 중대한 영향을 미치며, 기존의 메시지 전달 네트워크를 넘어서는 새로운 아키텍처 접근법이 필요함을 시사합니다.



### Learning Traffic Anomalies from Generative Models on Real-Time Observations (https://arxiv.org/abs/2502.01391)
- **What's New**: 이번 연구에서는 Spatiotemporal Generative Adversarial Network (STGAN) 프레임워크를 사용하여 도시 교통 시스템에서의 이상 감지 정확성을 개선합니다. Graph Neural Networks (GNNs)와 Long Short-Term Memory (LSTM) 네트워크를 결합하여 교통 데이터의 복잡한 공간 및 시간적 의존성을 포착합니다. 연구는 스웨덴 예테보리의 42개 교통 카메라에서 수집된 실시간 데이터를 기반으로 하여 진행되었습니다.

- **Technical Details**: STGAN은 노드로서 교통 카메라를, 엣지로서 도로를 나타내는 동적 디지털 트윈 기법을 사용하여 도시 교통 네트워크를 모델링합니다. 이 프레임워크에서는 무게가 부여된 그래프를 통해 교통 정황을 나타내고, 각 노드 간의 인접성을 기반으로 공간적 상관관계를 정의합니다. 이를 통해 단기적 및 장기적 교통 패턴을 효율적으로 캡처하여 이상을 감지할 수 있습니다.

- **Performance Highlights**: 모델은 높은 정밀도로 교통 이상을 효과적으로 감지하며, 낮은 허위 긍정률을 기록했습니다. 이상 감지 사례에는 카메라 신호 중단, 비주얼 아티팩트 및 극한 기상 조건이 포함되었습니다. 연구 결과는 최근 딥러닝을 활용한 교통 예측의 효용성을 강조하며, 전통적인 방법으로는 감지하기 어려운 복잡한 교통 패턴을 포착하는 데 기여합니다.



### Fine-Tuning Discrete Diffusion Models with Policy Gradient Methods (https://arxiv.org/abs/2502.01384)
Comments:
          23 pages, 4 figures, 5 tables

- **What's New**: 이 논문에서는 비연속 확산 모델(discrete diffusion models)의 세밀한 조정을 위한 새로운 정책 기울기 알고리즘을 제안합니다. 이 알고리즘은 Score Entropy Policy Optimization (SEPO)이라는 이름을 가지고 있으며, 비미분 가능 보상(non-differentiable rewards)을 효과적으로 처리할 수 있습니다. 이를 통해 강화 학습에서 사람의 피드백을 활용하는 데 있어 새로운 접근 방식을 제공합니다.

- **Technical Details**: SEPO 알고리즘은 기존의 두 기법, 즉 직접 보상 역전파(Direct Reward Backpropagation)와의 차별화된 점이 있습니다. 이 방법은 보상이 미분 가능할 필요가 없으며, 다양한 비연속 생성 작업에서 적용될 수 있습니다. 기존 방법들이 직면했던 수정성 문제를 해결하며, 비강화 학습 상황에서도 효율적인 최적화를 달성할 수 있습니다.

- **Performance Highlights**: 논문에서 제안하는 SEPO를 기반으로 한 다양한 수치 실험이 진행되었고, DNA 세밀 조정과 자연어 처리 태스크에서 우수한 성능을 보여주었습니다. 이 연구의 결과는 SEPO가 기존의 접근 방식보다 뛰어난 확장성과 효율성을 제공함을 입증하며, 앞으로 비연속 정보 처리의 새로운 가능성을 열어줄 것으로 기대됩니다.



### Data-Efficient Model for Psychological Resilience Prediction based on Neurological Data (https://arxiv.org/abs/2502.01377)
- **What's New**: 이 논문은 신경학적 데이터 부족 문제를 해결하기 위해 새로운 데이터 효율적인 심리적 회복력(prediction model) 예측 모델을 제안합니다. 기존의 자가 보고 질문지를 통해 평가한 회복력과 달리, 생물학적 마커를 기반으로 한 예측 모델은 더 객관적이고 신뢰할 수 있는 결과를 제공합니다. Neuro Kolmogorov-Arnold Networks (KAN)을 활용하여 새로운 멀티모달 representation 및 noise-informed inference algorithm을 도입했습니다. 이 모델은 두 가지 공공 데이터셋 및 자체 구축한 데이터셋에서 인상적인 성과를 보여주며, 향후 연구를 위한 가치 있는 심리학적 가설도 제시합니다.

- **Technical Details**: 이 연구에서는 Neural Kolmogorov-Arnold Networks (KAN)라는 신경망 구조를 사용하여 회복력 예측을 위한 새로운 방법론을 제시합니다. 훈련 단계에서 적용된 새로운 trait-informed multimodal representation 알고리즘과 smart chunk technique을 통해 제한된 데이터로 공유 잠재 공간(shared latent space)을 학습합니다. 테스트 단계에서 저신호 대 잡음비(signal-to-noise ratio)를 해결하기 위해 noise-informed inference algorithm이 소개됩니다. 이 모델은 fMRI와 EEG 데이터의 품질을 바탕으로 회복력을 예측합니다.

- **Performance Highlights**: 제안된 모델은 각종 공개 데이터셋과 자가 구축한 데이터셋에서 모두 우수한 성과를 보였습니다. 기존의 심리적 회복력 예측 연구에 비해 더 객관적인 생물학적 마커를 기반으로 하는 접근 방식은 고위험 직업군에서 유용하게 적용될 수 있습니다. 슬픈 소식이나 스트레스를 겪는 질병을 예방하는 데 큰 도움을 줄 것으로 기대됩니다. 이번 연구는 심리적 회복력 테스크에 대한 새로운 통찰력을 제공하여, 향후 심리학 연구의 방향성에 기여할 것으로 보입니다.



### Compact Rule-Based Classifier Learning via Gradient Descen (https://arxiv.org/abs/2502.01375)
- **What's New**: 이번 연구에서는 사용자에게 규칙의 최대 수와 길이를 조절할 수 있는 새로운 rule-based classifier를 제안합니다. 이 분류기는 gradient descent를 활용해 학습되며, fuzzy sets를 이용한 수치 파티션 역시 사용자가 조정할 수 있습니다. 고정된 규칙의 구조를 갖는 기존 방법들과는 다르게, 사용자가 이해하기 쉽게 공간 파티션을 설정할 수 있는 점이 새롭습니다.

- **Technical Details**: 제안하는 시스템인 Fuzzy Rule-based Reasoner (FRR)는 gradient descent를 통해 학습이 가능하며, 규칙의 수와 선행 조건을 수동으로 설정할 수 있도록 설계되었습니다. 논문에서는 FRR의 차별성과 효율성을 높이기 위해 원시 연산을 행렬 연산으로 복제하는 방법과 지표 함수의 이완을 통한 훈련 효과 증대 전략을 제시합니다. 이를 통해 기계의 복잡성을 줄이고 해석 가능성을 증대시키는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, FRR은 40개의 다양한 데이터셋에서 기존의 genetic optimization을 통한 규칙 기반 분류기보다 나은 성능을 보였습니다. 우리 방법은 다른 설명 가능한 분류기들에 비해 더 적은 패턴을 사용하여 컴팩트한 규칙 기반을 만들 수 있음을 보여주었습니다. 이러한 결과는 규칙 기반 알고리즘의 해석 가능성을 유지하면서도 높은 정확도를 달성할 수 있음을 시사합니다.



### Meursault as a Data Poin (https://arxiv.org/abs/2502.01364)
Comments:
          7 pages, 9 figures, 4 tables

- **What's New**: 이 논문은 데이터화(datafication) 시대의 데이터 중심 사회에서 인간 경험을 수치화하는 것의 철학적, 윤리적 질문을 다룹니다. Albert Camus의 소설 'The Stranger'의 주인공인 Meursault의 감정적으로 단절된 삶을 통해 이러한 문제를 탐구하고 있습니다. 이 연구는 감정 탐지(emotion detection), 감정 분석(sentiment analysis), 명명된 엔티티 인식(named entity recognition)과 같은 자연어 처리(NLP) 기법을 사용하여 Meursault의 삶에서 주요 사건과 행동을 정량화합니다.

- **Technical Details**: 연구에서 사용된 기술적 방법론에는 BERT를 활용한 감정 탐지, VADER를 이용한 감정 분석, spaCy 기반의 명명된 엔티티 인식이 포함됩니다. 이러한 기법들은 Meursault의 복잡한 감정과 행동을 분석하는 데 적용되었으며, 이는 특히 존재적 소외와 도덕적 모호성이 있는 인간 경험에 적용할 때의 한계를 나타냅니다. AI 도구가 Meursault의 행동과 감정을 잘못 해석하는 방식을 조사함으로써, 인간 내러티브를 데이터 포인트로 축소하는 것의 윤리적 딜레마를 강조합니다.

- **Performance Highlights**: 연구 결과는 데이터 기반 내러티브에 대한 증가하는 의존도를 비판하며, 인공지능에 인간적인 가치를 통합할 필요성을 지적합니다. Meursault의 행동 분석은 기존의 알고리즘 모델이 복잡한 인간 경험을 얼마나 잘 반영하지 못하는지를 보여줍니다. 이러한 발견은 데이터 중심 사회의 근본적인 가정에 도전하며, 윤리적인 측면에서 더 포괄적인 접근을 제안합니다.



### Activation by Interval-wise Dropout: A Simple Way to Prevent Neural Networks from Plasticity Loss (https://arxiv.org/abs/2502.01342)
- **What's New**: 본 논문은 AID (Activation by Interval-wise Dropout)라는 혁신적인 방법을 제안하여 신경망 훈련 중 발생하는 plasticity loss를 해결하고자 합니다. AID는 기존 Dropout과는 다르게 각 preactivation interval에 대해 서로 다른 확률로 Dropout을 적용하여 서브네트워크를 생성합니다. 이 방식을 통해 네트워크를 정규화하고, plasticity를 유지할 수 있음을 이론적으로 입증했습니다.

- **Technical Details**: 플라스틱성 손실(plasticity loss)은 신경망이 새로운 작업에 대해 학습하는 능력이 저하되는 현상을 의미합니다. 저자들은 이전 연구에서 보여준 것처럼 이 현상이 비정상(non-stationary) 도메인에서 발생함을 강조하며, AID가 이를 해결할 수 있는 방법이라고 주장합니다. AID는 활성화 함수(activation function)로도 작동하며, He initialization과의 호환성을 이론적으로 증명했습니다.

- **Performance Highlights**: 실험 결과 AID는 다양한 벤치마크에서 plasticity를 지속적으로 유지하는 데 효과적인 것으로 나타났습니다. CIFAR10, CIFAR100, TinyImageNet과 같은 표준 이미지 분류 데이터셋을 포함한 연속학습(continual learning) 과제에서 그 효과가 검증되었습니다. 또한 AID는 아케이드 학습 환경(Arcade Learning Environment)의 강화학습 성능도 향상시켰습니다.



### Learning Fused State Representations for Control from Multi-View Observations (https://arxiv.org/abs/2502.01316)
- **What's New**: 이 논문에서는 Multi-view Fusion State for Control (MFSC)라는 새로운 접근 방식을 제안합니다. MFSC는 bisimulation metric learning을 MVRL에 통합하여 작업 관련성을 가지는 다중 뷰 표현을 학습하는 데 집중합니다. 이 방법은 다중 뷰를 통해 수집된 정보를 효과적으로 재구성하고, 특히 누락된 뷰의 문제를 개선합니다.

- **Technical Details**: MFSC는 다중 뷰 관찰의 공유 정보를 활용하여 마스크와 잠재 재구성 보조 작업을 제안합니다. 자동화된 환경에서 행동의 유사성과 보상 분포를 기반으로 하는 bisimulation 메트릭을 통합하여, 뷰 간의 상호 의존성을 강화하며, 학습된 상태 표현이 작업의 목표와 잘 맞춰지도록 합니다. 이러한 방법은 특히 완전하고 고품질의 관찰이 보장되지 않는 상황에서 유용합니다.

- **Performance Highlights**: 실험 결과, MFSC는 기존 MVRL 알고리즘보다 우수한 성능을 보여주었습니다. 특히 인터페어런스 또는 누락된 뷰가 있는 어려운 상황에서도 높은 성과를 지속적으로 유지했습니다. 이로 인해 MFSC는 로봇 조작 작업 및 동작 관련 작업에서 유용하게 활용될 수 있는 가능성을 제시합니다.



### TFBS-Finder: Deep Learning-based Model with DNABERT and Convolutional Networks to Predict Transcription Factor Binding Sites (https://arxiv.org/abs/2502.01311)
- **What's New**: 이 연구는 DNABERT를 활용한 새로운 딥러닝 모델인 TFBS-Finder를 제안합니다. TFBS-Finder는 전이 학습된 DNABERT에 CNN, MCBAM, MSCA 모듈을 결합하여 전반적인 TFBS 예측 능력을 향상시킵니다. 실험 결과, 제안된 모델은 최신 방법론에 비해 탁월한 성능을 보여주었습니다. 모든 코드는 공개되어 있으며, 연구 reproducibility를 지원합니다.

- **Technical Details**: TFBS-Finder는 165개의 ENCODE ChIP-seq 데이터셋을 기반으로 학습되며, 다양한 세포주와 TF를 포함합니다. 이 모델은 DNABERT를 사용하여 DNA 시퀀스의 장기 의존성을 포착하고, CNN, MCBAM, MSCA 등 다양한 모듈을 통해 고차원 로컬 특징을 추출합니다. 또한, 모듈 간의 중요성을 강조하기 위해 ablation study를 진행했습니다. 각 모듈은 TFBS 예측의 성능을 높이기 위해 필수적인 역할을 수행합니다.

- **Performance Highlights**: 실험 결과, TFBS-Finder는 기존 딥러닝 및 머신러닝 모델들보다 더 높은 정확도와 신뢰성을 보였습니다. Cross-cell line validation을 통해 TFBS-Finder의 일반성 및 강건성을 확인하였습니다. 이런 결과는 TFBS를 예측하는 새로운 기법으로서 TFBS-Finder의 잠재력을 보여줍니다. 최종적으로 제공된 데이터셋과 코드를 통해 연구자들이 보다 쉽게 접근하고 활용할 수 있도록 하였습니다.



### A Statistical Learning Perspective on Semi-dual Adversarial Neural Optimal Transport Solvers (https://arxiv.org/abs/2502.01310)
- **What's New**: 본 논문은 Neural network 기반의 Optimal Transport (OT) 문제를 다루며, 최근 기계 학습 문제들을 해결하는 데 있어서 이 방법론이 중요해짐을 강조합니다. 특히, 기존의 adversarial minimax OT 솔버의 이론적 조사가 부족하다는 문제의식을 가지고, 이러한 틈새를 메우기 위해 일반화 오류의 상한을 확립하였습니다. 이 연구는 각종 통계적, 수학적 특성에 의존하는 경계 값을 도출함으로써 OT 솔버의 신뢰성을 높입니다.

- **Technical Details**: 연구자는 min-max quadratic OT 솔버의 일반화 오류를 추정 오류와 근사 오류의 합으로서 상한을 설정했습니다. 또한 Rademacher 복잡도에 기반하여 추정 오류의 경계 값을 구체화하고, 적절한 뉴럴 네트워크 클래스 선택을 통해 근사 오류를 작게 만드는 방법을 제시합니다. 이 과정에서 min-max OT 솔버의 학습 가능성을 보장하기 위한 이론적 기초가 마련됩니다.

- **Performance Highlights**: 결과적으로, 적절한 샘플 수와 뉴럴 네트워크 클래스를 선택함으로써 min-max OT 솔버의 일반화 오류를 임의로 작게 유지할 수 있다는 것을 입증하였습니다. 이는 OT 문제 해결의 새로운 경로를 제시하며, 향후 더 일반적인 OT 공식에 대해서도 유사한 경계를 도출할 가능성을 열어줍니다. 연구를 통해 OT 솔버 개발에 기여할 수 있는 중요한 이론적 통찰이 제공되었습니다.



### Partial Channel Network: Compute Fewer, Perform Better (https://arxiv.org/abs/2502.01303)
- **What's New**: 이번 연구에서는 네트워크의 정확도와 처리량을 저하시키지 않으면서 파라미터 수와 FLOPs를 줄일 수 있는 Partial Channel Mechanism (PCM)을 제안합니다. 이 혁신적인 접근법은 feature map 채널을 여러 부분으로 나누고, 각각의 부분에 다양한 연산을 적용한 후 결과를 통합하는 방식으로 작동합니다. 이를 통해 Partial Attention Convolution (PATConv)을 도입해 시각적 주의를 결합하여 모델 파라미터와 FLOPs를 절약할 수 있는 방법을 제시합니다.

- **Technical Details**: Partial Attention Convolution (PATConv)는 전통적인 합성곱(convolution)과 시각적 주의(visual attention)를 결합하여 모델 성능을 향상시키도록 설계되었습니다. 이 연구는 PATConv을 활용하여 새로운 유형의 블록인 Partial Channel-Attention block (PAT_ch), Partial Spatial-Attention block (PAT_sp), 및 Partial Self-Attention block (PAT_sf)를 개발하였습니다. 또한, 동적으로 채널 비율을 학습할 수 있는 동적 부분 합성곱(dynamic partial convolution, DPConv)도 제안되었습니다.

- **Performance Highlights**: 제안된 PartialNet은 ImageNet-1K 분류 문제에서 다른 SOTA 모델과 비교할 때 더 높은 top-1 정확도와 빠른 추론 속도를 보여줍니다. 이 모델은 COCO 데이터셋에서의 탐지 및 분할 작업에서도 뛰어난 성능을 발휘하며, 전반적으로 효율성을 높이고 정확도를 개선했습니다. 연구팀은 이 모델의 코드도 공개하여 다른 연구자들이 쉽게 접근할 수 있도록 하였습니다.



### Common Foundations for SHACL, ShEx, and PG-Schema (https://arxiv.org/abs/2502.01295)
Comments:
          To be published at WWW 2025

- **What's New**: 이 논문은 RDF, Property Graphs와 같은 다양한 그래프 데이터 모델에 대한 스키마 언어인 SHACL, ShEx, PG-Schema의 공통점과 차이점에 대해 설명합니다. 저자들은 이러한 스키마 언어의 주요 구성 요소에 대한 명확한 정의를 제공하며, 이를 통해 서로 다른 커뮤니티에서 발전해 온 그래프 데이터 언어의 이해를 돕고자 합니다. 또한, 각 언어의 특징을 비교하기 위한 일관된 프레임워크를 제시하여 사용자들에게 선택의 힌트를 제공합니다.

- **Technical Details**: 이 논문에서는 세 가지 스키마 언어(SHACL, ShEx, PG-Schema)의 기능을 비교하기 위해 Common Graph Data Model을 제안합니다. 이는 RDF와 Property Graphs를 포함하는 수학적 표현으로, 기본적인 공통 요소들이 무엇인지 규명합니다. 저자들은 또한 각 언어의 비재귀(non-recursive) 스키마에 초점을 맞추며, 서로 다른 언어 간의 기본적인 차이와 유사성을 설명합니다.

- **Performance Highlights**: 이 연구는 세 가지 스키마 언어의 기능을 간결하게 정의하고, 그들의 차별화된 특성을 밝혀내는 것을 목표로 합니다. 논문은 각 언어의 주요 기능을 소개하며, 사용자가 각 언어의 차이점을 쉽게 인지할 수 있도록 도와줍니다. 연구 결과는 실무자들이 적합한 스키마 언어를 선택하는 데 도움을 주고, 연구자들이 스키마 언어의 복잡성과 표현 능력을 연구하는 데 기여할 것으로 기대됩니다.



### Rational Gaussian wavelets and corresponding model driven neural networks (https://arxiv.org/abs/2502.01282)
Comments:
          Submitted to IEEE Transactions on Signal Processing, 2024 (under review)

- **What's New**: 이 논문에서는 적절한 정수 비율을 곱한 가우시안 웨이브렛을 사용하여 지속적인 웨이브렛 변환을 고려합니다. 제안된 라쏘 가우시안 웨이브렛(RGW)은 모성 웨이브렛의 형태를 조정할 수 있는 자유 매개변수를 가지고 있어 복잡한 형태의 신호를 효과적으로 근사할 수 있습니다. 또한, 변수 투영 기반의 RGW 변환은 신경망에서 해석 가능한 특성 학습 층을 제공할 수 있습니다.

- **Technical Details**: RGW는 모성 웨이브렛의 모양을 조정하기 위해 임의의 자유도를 사용하여 구축됩니다. 이는 매개변수를 최적화할 수 있는 잠재력과 함께 구조의 매끄러움과 대칭성을 보장합니다. 연구에서는 RGW 클래스의 적합성을 입증하고, 변수 투영 기법을 사용하여 신호의 웨이브렛 계수의 수치 근사를 제공합니다.

- **Performance Highlights**: 실험에서는 RGW-VP 네트워크 아키텍처를 통해 ECG 데이터에서 심실 이탈 심박동(VEB)을 성공적으로 분류하는 성과를 보여줍니다. 모델이 학습한 매개변수는 의료 문헌에서 VEB 설명과 대응하는 정보를 학습하는 것을 보여줍니다. 또한 제안된 모델은 최신 분류 정확도를 달성하기 위해 더 적은 수의 매개변수를 요구합니다.



### HyperSHAP: Shapley Values and Interactions for Hyperparameter Importanc (https://arxiv.org/abs/2502.01276)
- **What's New**: 이 논문에서는 Hyperparameter Optimization (HPO)의 필요성을 강조하면서, 기존의 블랙박스 형태의 자동 기계 학습(AutoML) 시스템의 한계를 극복하기 위한 게임 이론 기반의 설명 가능성 프레임워크인 HyperSHAP을 제안합니다. HyperSHAP은 하이퍼파라미터의 중요성과 상호작용에 대한 통찰력을 제공하며, 이를 통해 종합적인 성과 지표의 분해를 가능하게 합니다. 이 시스템은 다양한 HPO 벤치마크에서 적용되어 성능 개선의 일반적인 패턴을 분석합니다.

- **Technical Details**: HyperSHAP은 Shapley value를 기반으로 하여 특정 구성, 하이퍼파라미터 공간 및 최적화 편향의 세 가지 수준에서 하이퍼파라미터의 중요성 및 상호작용 구조를 추출합니다. 이를 통해 우리는 하이퍼파라미터의 상호작용이 높은 차수에서도 존재함을 확인하였으며, 이를 다양한 설명 작업에 적용하여 HyperSHAP의 다재다능함을 입증합니다. 이러한 접근 방식은 기존 방법들과 비교해 하이퍼파라미터의 영향을 보다 정량화할 수 있게 해줍니다.

- **Performance Highlights**: HyperSHAP은 HPO 문제의 상호작용 구조를 분석함으로써, 높은 차수의 상호작용이 존재하더라도 대부분의 성능 향상은 낮은 차수의 표현에 의해 설명될 수 있음을 보여줍니다. 이를 통해 AutoML 시스템이 하이퍼파라미터의 튜닝 기준에 따라 어떻게 성능을 발휘하는지를 보다 명확히 이해할 수 있게 되며, ML 연구자와 전문가 간의 신뢰를 구축하는 데 기여할 수 있습니다. 최종적으로, HyperSHAP은 사용자가 AutoML 시스템의 결과를 이해하고 유용하게 활용할 수 있도록 도와주는 중요한 도구가 될 것입니다.



### Analysis of Student-LLM Interaction in a Software Engineering Projec (https://arxiv.org/abs/2502.01273)
Comments:
          8 pages

- **What's New**: 이 논문에서는 소프트웨어 공학 교육에서 Large Language Models (LLMs)의 적용에 대한 새로운 연구 결과를 제시하고 있습니다. 연구 결과에 따르면, 126명의 학부 학생들이 13주 동안 AI 어시스턴트와의 상호작용을 통해 LLMs의 교육적 이점을 활용한 것으로 나타났습니다. 특히 ChatGPT가 CoPilot보다 선호되는 경향이 있었으며, 이를 통해 코드 품질 향상에 기여한 것으로 보입니다.

- **Technical Details**: 연구에서는 학생들의 대화, 생성된 코드, 사용된 코드 및 코드베이스에 통합되는 인간 개입 수준을 분석했습니다. ChatGPT의 경우 낮은 계산 복잡도를 보였고, 대화 기반 상호작용이 자동 생성된 코드보다 더 나은 코드 품질을 제공하는 것으로 나타났습니다. AI와의 상호작용을 통한 학습 과정이 소프트웨어 엔지니어링 교육에 중요한 역할을 하는 것으로 평가됩니다.

- **Performance Highlights**: 이 연구는 LLMs의 조기 채택이 빠르게 변화하는 소프트웨어 공학 분야에서 경쟁력을 유지하는 데 필수적임을 강조합니다. 학생들은 AI와의 상호작용을 통해 필요한 기술을 습득해야 하며, 이는 생산성을 높이는 데 크게 기여할 것입니다. 특히 대화형 상호작용은 코드 생성의 질을 높이는 데 중요한 역할을 하는 것으로 확인되었습니다.



### Resilient UAV Trajectory Planning via Few-Shot Meta-Offline Reinforcement Learning (https://arxiv.org/abs/2502.01268)
- **What's New**: 이 논문의 주요 혁신은 오프라인 RL (Reinforcement Learning) 알ゴ리즘을 활용하여 환경과의 온라인 상호작용 없이도 모델을 학습할 수 있도록 하는 것입니다. 이는 안전성 및 비용 문제로 인해 실제 환경에서 온라인 RL을 사용하는 것이 어려울 수 있기 때문에 가치가 있습니다. 제안된 알고리즘은 보수적 Q-학습 (CQL)과 모델 무관 메타 학습 (MAML)을 결합하여 새로운 동적 환경으로 확장할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: 본 연구에서 제안된 알고리즘은 CQL을 사용한 오프라인 RL과 MAML을 도입하여 기존의 데이터셋을 이용하여 정책을 최적화합니다. 메타 학습의 도움을 통해, 이 알고리즘은 새로운 환경에서도 신속하게 적응할 수 있는 강점을 보여줍니다. 특히 무인 항공기 (UAV)의 경로 최적화 및 정보 전송 정책을 최적화하는 데 중점을 두고, 나이가 정보 (AoI)와 전송 전력을 동시에 최소화하는 목표를 갖고 있습니다.

- **Performance Highlights**: 제안된 알고리즘은 기존의 깊은 Q-네트워크 및 CQL 기반 모델들보다 빠르게 수렴하는 것으로 수치적인 결과가 보여졌습니다. 또한 제안하는 알고리즘은 몇 개의 데이터 포인트로 오프라인 데이터셋을 활용해 가장 최적의 AoI 및 전송 전력을 달성할 수 있는 유일한 알고리즘이라는 점에서 눈여겨볼 만합니다. 이는 예기치 않은 환경 변화에도 잘 견딜 수 있는 저항성을 보여주어 실제 무선 시스템에서의 응용 가능성을 높입니다.



### Learnable polynomial, trigonometric, and tropical activations (https://arxiv.org/abs/2502.01247)
- **What's New**: 이번 연구는 Orthogonal function bases와 Tropical polynomials를 기반으로 하는 학습 가능한 활성화 함수(activation functions)를 가진 확장 가능한 신경망(neural networks)을 조사합니다. 이는 ImageNet-1K 분류(classification)와 OpenWebText의 다음 토큰 예측(next token prediction)을 목표로 하고 있습니다. 기존 활성화 함수인 ReLU와 달리, 학습 가능한 활성화 함수는 훈련 과정에서 네트워크가 동적으로 적응할 수 있도록 합니다.

- **Technical Details**: 연구에서는 깊은 네트워크(deep networks)에서 발생할 수 있는 소실(vanishing) 및 폭발(exploding) 기울기(gradient) 문제를 해결하기 위해, 변동성(variance) 관리를 개선하는 초기화(initiation) 방안을 제안합니다. 이 방법은 변환기(transformers)와 합성곱 네트워크(convolutional networks)에서 단독으로 단위 변동성을 보장할 수 있으며, 이는 깊은 구조에서도 안정적인 기울기 흐름을 보장합니다.

- **Performance Highlights**: 실험 결과, Hermite, Fourier, Tropical 기반의 학습 가능한 활성화 함수를 사용하는 네트워크가 GPT-2 및 ConvNeXt 네트워크보다 훈련과 테스트 정확도(accuracy) 및 혼란도(perplexity)에서 유의미한 향상을 보이는 것으로 나타났습니다. 이러한 연구 결과는 대규모 작업에서 학습 가능한 활성화 함수의 실현 가능성을 강조합니다.



### OphthBench: A Comprehensive Benchmark for Evaluating Large Language Models in Chinese Ophthalmology (https://arxiv.org/abs/2502.01243)
- **What's New**: 이번 연구에서는 중국 안과 분야에서 LLM(대형 언어 모델)의 성능 평가를 위한 전문 벤치마크인 OphthBench를 소개합니다. 이 벤치마크는 5개의 주요 시나리오로 구성되며, 총 9개의 작업과 591개의 질문이 포함되어 있습니다. 이를 통해 LLM의 실제 의료 적용 가능성을 종합적으로 평가하고 향후 연구 방향을 제시합니다.

- **Technical Details**: OphthBench는 교육, 분류(triage), 진단(diagnosis), 치료(treatment), 예후(prognosis)와 같은 5가지 핵심 시나리오를 바탕으로 구축되었습니다. 각 시나리오에는 다양한 질문 유형이 포함되며, 이를 통해 LLM의 정확성과 능력을 평가할 수 있습니다. 특히, 중국의 문화적 맥락과 의료 시스템을 반영하여 설계되었습니다.

- **Performance Highlights**: 39개의 인기 있는 LLM을 대상으로 한 평가 결과, 현재 모델의 능력이 임상에서 요구되는 실제적인 필요와 큰 차이를 보임을 확인했습니다. 이는 LLM의 진화 방향을 제시하는 중요한 통찰로 볼 수 있습니다. 본 연구는 LLM의 잠재력을 열어주는 기반을 마련하고, 안과 항목에서의 개발을 촉진하는 데 기여하고자 합니다.



### Eliciting Language Model Behaviors with Investigator Agents (https://arxiv.org/abs/2502.01236)
Comments:
          20 pages, 7 figures

- **What's New**: 이 논문에서는 언어 모델이 특정 목표 행동을 유도하는 프롬프트를 탐색하는 '행동 유도(behavior elicitation)' 문제를 다룹니다. 연구진은 지도 학습(supervised fine-tuning), 강화 학습(reinforcement learning)을 통한 DPO(Direct Preference Optimization) 및 새로운 Frank-Wolfe 학습 목표를 활용하여 다양한 프롬프트 전략을 발견하는 방법론을 제안합니다. 이 방법은 기존의 프롬프트 설계 방식보다 더 유연하고 해석 가능한 프롬프트를 생성할 수 있습니다.

- **Technical Details**: 연구에서는 고정된 타겟 모델 Llama-3.1 8B를 사용하여 두 가지 유형의 행동 유도 문제, 즉 문자열 유도(string elicitation)와 루브릭 기반 유도(rubric-based elicitation)를 고려합니다. 특히, 단일 프롬프트를 통해 특정 반응을 이끌어내는 단일 턴 elicitation을 중점적으로 다루며, 이때의 두 가지 주요 도전 과제는 언어 입력의 조합적 공간에서 최적화해야 한다는 점과 다양성을 보장하는 것이었습니다. 이를 해결하기 위해 다단계 RL 파이프라인을 도입하고, 다이버시티를 위해 반복적으로 정규화된 DPO 변형을 사용합니다.

- **Performance Highlights**: 이 연구의 결과, 조사자 모델은 100%의 공격 성공률을 달성하였고, AdvBench(유해 행동)에서 Llama-3.1 8B 모델에 대해 98%의 성공률을 기록했습니다. 또한, 인간 해석이 가능한 다양한 행동 유도 전략을 발견하여 단일 턴 조사자들이 복잡한 언어 모델의 다양하고 유익한 행동을 유도할 수 있음을 입증했습니다. 향후에는 다중 턴과 도구 사용이 가능한 조사자 모델이 구현될 것으로 기대하며, 이는 인간 조사자들이 사용하는 다양한 기법을 활용할 수 있을 것입니다.



### One-step full gradient suffices for low-rank fine-tuning, provably and efficiently (https://arxiv.org/abs/2502.01235)
Comments:
          86 pages

- **What's New**: 이 논문은 Low-Rank Adaption(LoRA)의 성능을 향상시키기 위한 이론적 분석을 다룹니다. 새로운 이론적 결과를 통해 LoRA가 특정 특이 서브스페이스(singular subspace)에 정렬되는 방식과, 고순위(high-rank) 상황에서 수렴성을 개선하기 위한 preconditioners의 중요성을 강조합니다. 이를 바탕으로 특정 스펙트럼 초기화 전략(spectral initialization strategy)을 적용한 preconditioned LoRA에 초점을 맞춥니다.

- **Technical Details**: 논문에서는 초기화 시 정렬(alignment)과 일반화(generalization)가 직접적으로 보장될 수 있음을 증명했습니다. 이러한 결과는 선형(linear) 및 비선형(nonlinear) 모델 모두에 적용되며, subsequent linear convergence를 구축할 수 있음을 확인합니다. LoRA-One 알고리즘은 One-step gradient와 preconditioning을 활용하여 이론적으로 뒷받침되는 방법론으로, vanilla LoRA 및 그 변형에 비해 실험적으로 상당한 개선을 이루었습니다.

- **Performance Highlights**: 이 논문에서 제안한 LoRA-One 알고리즘은 여러 벤치마크에서 기존의 LoRA에 비해 현저한 성능 향상을 보여줍니다. 이론적 분석이 학습 역학(learning dynamics)을 분리하고 스펙트럼 초기화가 특징 학습(feature learning)에 어떻게 기여하는지를 설명함으로써, 행렬 센싱(matrix sensing) 및 딥러닝 이론(deep learning theory)을 이해하는 데에도 기여할 수 있음을 보여줍니다.



### The dark deep side of DeepSeek: Fine-tuning attacks against the safety alignment of CoT-enabled models (https://arxiv.org/abs/2502.01225)
Comments:
          12 Pages

- **What's New**: 이 연구는 대형 언어 모델인 DeepSeek가 fine-tuning 공격에 얼마나 취약한지를 조사합니다. 새로운 접근 방식으로, Chain of Thought 기반 추론 모델의 성능을 사전 훈련 이후의 단계에서 분석합니다. 연구는 특정 입력으로 인한 모델의 출력 변화를 주목하며, 이는 유해한 콘텐츠 생성을 일으킬 수 있습니다.

- **Technical Details**: Chains of Thought (사고의 연쇄) 기반 추론 모델은 복잡한 문제 해결에 사용됩니다. 본 연구에서는 fine-tuning (미세 조정) 할 때 발생하는 adversarial inputs (적대적 입력)와의 상호작용을 분석합니다. 이 모델은 출력이 어떻게 조작되는지를 통해 유해성(harmfulness)이 증가하는 경향을 보입니다.

- **Performance Highlights**: DeepSeek의 모델은 fine-tuning 공격에 의해 출력이 심각하게 왜곡될 수 있다는 사실을 발견했습니다. 연구 결과는 Chain of Thought 모델의 안전성과 윤리적 사용 deployment에 중대한 영향을 미칩니다. 이러한 발견은 향후 대형 언어 모델 개발 시 안정성과 안전성 측면에서 중요한 통찰을 제공합니다.



### Provable Ordering and Continuity in Vision-Language Pretraining for Generalizable Embodied Agents (https://arxiv.org/abs/2502.01218)
- **What's New**: 이번 논문은 Action Temporal Coherence Learning (AcTOL)이라는 새로운 방법을 제안합니다. AcTOL은 비디오 내의 동작을 고정된 목표 기반 제약 없이 순서적이고 연속적인 비전-언어 표현을 학습하는 것을 목표로 합니다. 기존의 방법들이 미래 프레임에 치우쳐 오류를 발생시키는 문제를 해결하기 위해, AcTOL은 프레임 간의 의미적 차이를 대조하고 매끄러운 전이를 보장하는 로컬 브라운 운동 조항을 도입합니다.

- **Technical Details**: AcTOL은 비디오를 연속적인 궤적으로 간주하며, (1) 프레임 간의 의미적 차이를 대조하여 자연스러운 순서를 반영하고, (2) 비디오 전반에 걸쳐 일관된 비주얼 표현을 확보하도록 합니다. Vision-Language Ordering (VLO) 손실을 도입하여 프레임 간의 상대적 시간 거리에 따라 중첩된 의미적 정렬을 보장하고, 브라운 운동 과정을 사용하여 비주얼 표현의 연속성과 안정성을 확보합니다.

- **Performance Highlights**: 다양한 시뮬레이션 환경에서 수행된 실험 결과, AcTOL은 이전 방법들에 비해 최대 49% 성능을 향상시켰습니다. 또한, AcTOL은 여러 실제 동작 비디오에서 언어 조건에 따른 비주얼 보상을 생성하여 명시된 지침과 잘 정렬되는 밀집 보상을 생성하는 데 성공했습니다.



### Nearly Lossless Adaptive Bit Switching (https://arxiv.org/abs/2502.01199)
- **What's New**: 이 논문에서는 모델 압축과 가속을 위한 모델 양자화의 중요성을 강조하며, 특히 다중 정밀도(multi-precision) 또는 한 번의 훈련(one-shot training) 방식으로 양자화를 수행하는 새로운 방법을 제안합니다. 기존의 Quantization-Aware Training(QAT)은 고정된 비트 너비(bit-width)에 초점을 맞추고 있었으나, 본 연구는 다양한 하드웨어와 전송 요구 사항에 대응하기 위해 비트 전환(bit-switching)을 효과적으로 수행할 수 있는 Double Rounding 방식을 도입합니다.

- **Technical Details**: Double Rounding 방법은 정밀도가 다른 양자화 과정에서 거의 손실 없는 비트 스위칭을 가능하게 하여, 고정밀 모델을 저장하는 대신에 가장 높은 정수 정밀도를 사용합니다. 또한, Adaptive Learning Rate Scaling(ALRS) 기법을 통해 다양한 정밀도에 맞게 학습률을 동적으로 조정하여 훈련 프로세스를 최적화하려고 합니다. 이러한 새로운 방법들은 더 높은 정밀도의 모델을 사용할 때의 경량화를 가능하게 하여, 획기적인 개선을 가져옵니다.

- **Performance Highlights**: ImageNet-1K 분류 실험 결과, 본 연구에서 제안한 방법이 다중 정밀도 및 혼합 정밀도(mixed-precision)에서 최신 기법들과 비교하여 충분한 장점을 가지는 것으로 나타났습니다. 비트 스위칭 과정에서의 정확도 저하를 방지하고, 다양한 정밀도 간의 경쟁 문제를 해결하는 등 다양한 응용 분야에서의 성공 가능성을 보여줍니다. 이 연구 결과는 감지(detection) 및 분할(segmentation) 작업과 대규모 언어 모델(LLMs)에 대한 적용 가능성도 입증했습니다.



### Dance recalibration for dance coherency with recurrent convolution block (https://arxiv.org/abs/2502.01190)
- **What's New**: 최근 생성 AI(Generative AI) 기술의 발전으로 댄스 생성 분야에서도 큰 진전을 보이고 있습니다. 본 연구에서는 Lodge 모델을 개선한 R-Lodge를 제안하며, 이는 Dance Recalibration이라는 재귀적 시퀀스 표현 학습 방법을 도입하여 일관성을 높입니다. R-Lodge는 이전의 댄스 동작의 정보를 각 생성된 동작에 통합하여 보다 자연스러운 결과를 도출해냅니다.

- **Technical Details**: R-Lodge 모델은 Dance Recalibration Block이라는 모듈을 사용하여 기존의 저급 댄스 표현의 일관성 부족 문제를 해결하려 합니다. 기존 Lodge 모델에서 발생하는 불안정성과 일관성 부족을 해결하기 위해, 댄스 생성 과정에서 재귀적 재조정 프로세스를 추가하였습니다. 이 과정은 전체 댄스 로드(representation)에 시퀀셜한 정보를 추가하여 일관성을 향상시킵니다.

- **Performance Highlights**: R-Lodge는 FineDance 데이터셋에서 평가되었으며, 댄스의 일관성을 극대화하는 성과를 보여주었습니다. 기존의 모형보다 더 매끄럽고 자연스러운 댄스 동작 전환을 구현하였으며, 댄스의 전체 품질을 향상시켰습니다. 이를 통해 R-Lodge는 댄스 생성의 최첨단 성능을 달성했습니다.



### Compressed Image Generation with Denoising Diffusion Codebook Models (https://arxiv.org/abs/2502.01189)
Comments:
          Code and demo are available at this https URL

- **What's New**: 새롭게 제안된 Denoising Diffusion Codebook Model (DDCM)은 Denoising Diffusion Models (DDMs)을 기반으로 한 생성적 접근법으로, 고품질 이미지를 생성함과 동시에 손실 없이 압축된 비트스트림 표현도 생성합니다. 이 과정은 표준 Gaussian 노이즈 샘플링을 미리 정의된 코드북에서의 고정된 iid Gaussian 벡터 샘플로 대체함으로써 이루어집니다. DDCM은 극도로 작은 코드북을 사용하더라도 표준 DDM의 샘플 품질과 다양성을 유지하는 것으로 나타났습니다.

- **Technical Details**: DDCM은 고정된 구조의 코드북을 통한 샘플링을 사용하여, 이전의 무한한 표현 공간의 부족함을 해결합니다. 이론적으로, 1000개의 샘플링 단계로 이루어진 확률적 확산 생성 프로세스는 고정된 노이즈 선택을 통해 무한한 다양한 결과(2^1000)를 생성할 수 있습니다. DDCM은 개별 이미지에 가장 적합한 노이즈를 선택하여 생성 모델을 손실 압축 이미지 코덱으로 전환하며, 수학적으로는 SDEs에 기반한 점수 기반 근사화와 연결됩니다.

- **Performance Highlights**: DDCM을 통해 생성된 이미지들은 주어진 이미지에 가장 잘 맞는 노이즈를 선택함으로써 압축 과정에서 뛰어난 지각적 품질(perceptual quality)을 달성합니다. DDCM은 다양한 노이즈 선택 규칙을 통해 압축된 조건부 생성 작업을 위한 다재다능한 프레임워크를 제공합니다. 이러한 방식으로, 이미지 복원과 같은 다양한 작업에서도 압축된 비트스트림과 함께 생성된 이미지가 제공됩니다.



### Deep Active Speech Cancellation with Multi-Band Mamba Network (https://arxiv.org/abs/2502.01185)
- **What's New**: 이번 연구에서는 Active Speech Cancellation (ASC)을 위한 새로운 딥러닝 네트워크인 Multi-Band Mamba 아키텍처를 제안합니다. 기존의 Active Noise Cancellation (ANC) 방법을 넘어서, 소음과 음성 신호를 동시에 효과적으로 제거합니다. 이 아키텍처는 입력 오디오를 여러 주파수 대역으로 분리하여, 정확한 반대 신호를 생성하고 위상 정렬을 개선합니다.

- **Technical Details**: Multi-Band Mamba 아키텍처는 입력 신호를 주파수 대역으로 나누어, 음성 신호의 넓은 주파수 스펙트럼을 보다 효과적으로 처리합니다. 이 시스템은 최적화 기반의 손실 함수를 도입하여, 반대 신호 생성을 위한 거의 최적의 감독 신호를 제공합니다. 실험 결과에 따르면 ANC에서 최대 7.2dB, ASC에서 최대 6.2dB 개선된 성능을 보여주었습니다.

- **Performance Highlights**: 이 연구는 ASC 분야에서 첫 번째로 딥러닝을 활용하여 소음과 음성을 동시에 활발하게 제거하는 시스템을 소개합니다. 기존 딥러닝 기반 알고리즘들을 넘어서 높은 성능을 달성했으며, 동적 음향 시나리오에서 더욱 효과적인 성능을 발휘합니다. 향후 연구 방향으로도 새로운 길을 열 수 있는 가능성을 보여줍니다.



### FragmentNet: Adaptive Graph Fragmentation for Graph-to-Sequence Molecular Representation Learning (https://arxiv.org/abs/2502.01184)
Comments:
          22 pages, 13 figures, 5 tables

- **What's New**: FragmentNet은 화학적으로 의미 있는 분자를 위한 최초의 적응형 학습 토크나이저를 도입하며, 분자 그래프를 화학적으로 유효한 조각으로 분해하는 혁신적인 모델입니다. 이는 자가 감독 학습 방식을 통해 발전된 분할 방식으로, 기존 Atom 기반 또는 규칙 기반 방법보다도 더 우수한 성능을 보입니다. 특히, FragmentNet은 전통적인 프래그먼트 모델링의 한계를 극복하고, 구조적 연결성을 유지하면서 분자 그래프를 효과적으로 압축합니다.

- **Technical Details**: FragmentNet은 분자 그래프를 위한 그래프-시퀀스(가변 길이) 모델로, VQVAE-GCN(변분오토인코더 및 그래프 컨볼루션 네트워크)를 통합하여 계층적 조각 임베딩을 생성합니다. 이 모델은 구조적 정보와 동시에 공간 위치 인코딩을 활용하여 분자의 전반적인 특성을 포착하고, 분자 예측을 위한 Transformer 구조로 전달합니다. 이는 더 나아가 화학적 표현력을 향상시키는 데 중요한 역할을 합니다.

- **Performance Highlights**: 실험 결과, FragmentNet은 Masked Fragment Modeling(MFM)과 MoleculeNet 벤치마크에서 높은 성능을 보이며, 유사한 크기의 기존 모델들을 지속적으로 초월합니다. 특히, FragmentNet의 프래그먼트 기반 표현은 약물 디자인 및 최적화 분야에서 사용될 수 있는 강력한 도구가 되는 것을 보여줍니다. 주목할 만한 점은 기존 모델보다 적은 파라미터 수와 훈련 데이터 세트를 사용하면서도 우수한 성능을 발휘했다는 것입니다.



### A Single Model Ensemble Framework for Neural Machine Translation using Pivot Translation (https://arxiv.org/abs/2502.01182)
- **What's New**: 이 논문에서는 저자들이 낮은 자원 언어 쌍의 번역 성능을 향상시키기 위해 새로운 기법인 Pivot-based single model ensemble (PivotE)을 제안하고 있습니다. 기존의 앙상블 기법이 여러 모델 학습의 높은 계산 비용과 블랙박스 모델의 한계로 어려움을 겪고 있는 반면, 제안된 방법은 단일 모델을 사용하여 피벗 언어를 통한 후보 생성 및 후속 집계를 수행합니다. 이 접근 방식은 고품질 후보를 생성하고, 최종 번역 성능을 향상시킵니다.

- **Technical Details**: PivotE는 피벗 번역을 활용하여 후보를 생성하고, 이 후보들을 집계하여 최종 번역을 만듭니다. 첫 번째 단계에서는 단일 다국어 NMT 모델을 사용하여 후보를 생성하고, 두 번째 단계에서는 높은 품질의 후보를 선택하여 집계합니다. 이를 통해 다양한 후보를 생성하면서도 계산 비용을 줄일 수 있습니다.

- **Performance Highlights**: 실험 결과, PivotE는 서로 다른 언어 쌍에서 기존의 최첨단 방법들을 지속적으로 초과하는 번역 품질을 보여주었습니다. 고자원 언어의 지식을 전이하여 더 나은 후보를 생성하는 능력 덕분에 원본 문장의 미묘한 의미와 뉘앙스를 효과적으로 전달할 수 있었습니다.



### Joint Localization and Activation Editing for Low-Resource Fine-Tuning (https://arxiv.org/abs/2502.01179)
Comments:
          The code for the method is released at this https URL

- **What's New**: 최근 연구에서 제안된 Joint Localization and Activation Editing (JoLA)은 모델의 특정 컴포넌트를 수정하는 방법을 학습하는 혁신적인 접근법입니다. 기존의 parameter-efficient fine-tuning (PEFT) 방식의 한계를 극복하며, 작은 데이터셋에서도 우수한 성능을 발휘하도록 설계되었습니다. JoLA는 어떤 heads를 수정할지, 보정 방법으로 additive와 multiplicative를 함께 사용할지 등을 학습합니다.

- **Technical Details**: JoLA는 HardConcrete gates와 예상-L0 정규화를 사용하여 모델의 activations를 동적으로 조정합니다. 이 때, JoLA는 intervention을 최소화하여 편집할 컴포넌트를 선택하고, 조정의 유연성을 제공하기 위해 additive 오프셋과 multiplicative 스케일링을 결합합니다. 이 방법은 각 task에 대해 최적의 intervention 파라미터를 학습하여 저자원 환경에서 효과적인 fine-tuning을 가능하게 합니다.

- **Performance Highlights**: 세 가지 benchmark(Task)에 대한 실험 결과 JoLA는 기존 방식보다 일관되게 우수한 성능을 보였습니다. JoLA는 여러 데이터 규모와 모델 크기에 걸쳐 안정적인 결과를 나타내며, 특히 저자원 환경에서 더욱 두드러진 성능을 발휘합니다. 이 연구는 attention heads의 중요성을 강조하며, JoLA가 가장 효과적인 fine-tuning 전략임을 증명합니다.



### AtmosSci-Bench: Evaluating the Recent Advance of Large Language Model for Atmospheric Scienc (https://arxiv.org/abs/2502.01159)
Comments:
          16 pages, 3 figures, 2 tables

- **What's New**: AtmosSci-Bench는 대기 과학의 다섯 가지 핵심 문제 카테고리인 수문학(hydrology), 대기역학(atmospheric dynamics), 대기물리학(atmospheric physics), 지구물리학(geophysics), 물리 해양학(physical oceanography)을 체계적으로 평가하기 위해 설계된 새로운 벤치마크입니다. 이 벤치마크는 대학원 수준의 문제에서 선별된 다양한 객관식 질문을 생성하는 템플릿 기반의 질문 생성 프레임워크를 활용하여 확장 가능성과 다양성을 제공합니다.

- **Technical Details**: AtmosSci-Bench는 고급 추론 모델, 수학 증강 모델, 도메인 특화 기후 모델 등 다양한 대기 과학 문제에 적용 가능한 대표적인 LLM(large language models)들을 포함하여 평가합니다. 여기서 템플릿 기반 프레임워크를 통해 규칙 기반의 질문 생성을 수행, 각 질문 템플릿은 효과적인 기호적 확장을 통해 원하는 수의 구체적인 질문으로 체계적으로 확장됩니다.

- **Performance Highlights**: 평가 결과로부터, 고급 추론 모델(예: Deepseek-R1)이 다른 모델들에 비해 뛰어난 성능을 보이는 반면, 모델들의 추론 토큰 길이가 증가할수록 모델의 정확성이 향상되지만, 그 이후에는 수익이 감소하는 품질-효율성 트레이드오프가 존재함을 발견했습니다. 또, 고급 추론 모델은 높은 수치 정밀도를 요구하는 산술 작업에서 더 나은 강건성을 보였으나, 여전히 기호적 변동(symbolic perturbation)에서는 어려움을 겪는 것을 확인했습니다.



### Jailbreaking with Universal Multi-Prompts (https://arxiv.org/abs/2502.01154)
Comments:
          Accepted by NAACL Findings 2025

- **What's New**: 본 논문에서는 LLM(JUMP)을 탈옥(jailbreak)하기 위한 보편적인 다중 프롬프트를 최적화하는 기술을 제안합니다. 기존 기법들이 개별 사례에 대한 공격으로만 초점을 맞추었으나, JUMP는 이전의 접근 방식과는 달리 아직 보지 못한 작업(supervised tasks)으로까지 전이 가능한 공격자를 훈련할 수 있는 방법을 소개합니다. 또한, 방어 기술인 DUMP를 제안하여 공격 시나리오뿐만 아니라 방어 시나리오에서도 유용성을 입증합니다.

- **Technical Details**: JUMP는 Beam Search 과정을 활용하여 일련의 적대적 접미사(adversarial suffixes)를 생성합니다. 이 방법은 BEAST 프레임워크를 확장하여 좀 더 일반적인 시나리오를 다루도록 설계되었습니다. 특히, 우리의 알고리즘은 ASR(Attack Success Rate)과 perplexity 간의 균형을 잘 잡아주며, 신중하게 선택된 초기 프롬프트가 이러한 문제를 완화하는데 도움을 줍니다. 궁극적으로 JUMP++는 현재의 최고 기술들보다 성능이 우수함을 보였습니다.

- **Performance Highlights**: 실험 결과, JUMP는 기존의 여러 기술들보다 우수한 성과를 나타냈으며, 방어 시나리오에서도 잘 일반화되었습니다. JUMP*와 JUMP++ 두 가지 버전 모두 각각의 공격 결과에서 우수한 공격 성공률을 기록했습니다. 이러한 성과는 프롬프트 생성 과정에서의 최적화가 LLM의 보안 툴로서의 활용 가능성을 향상시킴을 시사합니다.



### Quantum Machine Learning: A Hands-on Tutorial for Machine Learning Practitioners and Researchers (https://arxiv.org/abs/2502.01146)
Comments:
          260 pages; Comments are welcome

- **What's New**: 이번 튜토리얼은 AI에 대한 기본 지식을 가진 독자들에게 양자 기계 학습(Quantum Machine Learning, QML)의 개념을 소개합니다. QML은 양자 컴퓨터의 힘을 활용하여 기계 학습의 지형을 재편하는 빠르게 발전하는 분야입니다. 튜토리얼에서는 QML의 기본 원리, 대표적인 알고리즘, 적용 가능성, 그리고 학습 가능성(trainability), 일반화(generalization), 계산 복잡성(computational complexity)과 같은 중요한 측면들을 다룹니다.

- **Technical Details**: 튜토리얼은 QML의 기초적인 원리와 알고리즘을 포함하여 다양한 실습 코드 데모도 제공합니다. 실습 코드는 현실 세계의 구현을 보여주고, 핸즈온 학습을 촉진하는 데 도움을 줍니다. 독자들은 QML의 최신 발전에 대한 포괄적인 개요를 이해할 수 있게 됩니다.

- **Performance Highlights**: QML 분야의 발전을 통해 독자들은 고전적인 기계 학습(classical machine learning)과 양자 컴퓨팅의 간극을 연결할 수 있습니다. 양자 시대의 AI를 탐구하고자 하는 이들에게 유용한 자료로 자리잡고 있습니다. 이 튜토리얼은 QML에 참여하고자 하는 사람들에게 중요한 학습 기회를 제공합니다.



### ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills (https://arxiv.org/abs/2502.01143)
Comments:
          Project website: this https URL

- **What's New**: 본 논문은 인간과 유사한 전신 기술을 달성하기 위해 AGILE 추적을 가능하게 하는 ASAP(Aligning Simulation and Real-World Physics)라는 새로운 프레임워크를 제안한다. 이 프레임워크는 시뮬레이션과 실제 물리학 간의 역학 불일치를 해결하는 두 단계의 과정을 포함하고 있으며, 기존 접근법에서의 한계를 극복하고자 한다. 기존의 System Identification(SysID) 및 Domain Randomization(DR) 방법들이 수정의 번거로움이나 비효율적인 정책들로 인해 민첩성을 잃는 문제를 해결하기 위해 개발되었다.

- **Technical Details**: ASAP는 두 단계의 프레임워크로 구성되어 있다. 첫 번째 단계는 인간 동작 데이터를 사용하여 시뮬레이션에서 모션 추적 정책을 사전 훈련하는 것이다. 두 번째 단계에서는 실제 데이터를 수집하여 역학 불일치를 보정하는 델타 액션 모델을 훈련시킨다. 이 델타 액션 모델은 시뮬레이터 내에서 통합되어 사전 훈련된 정책을 미세 조정하는 데 사용된다.

- **Performance Highlights**: ASAP는 IsaacGym부터 실제 Unitree G1 휴머노이드 로봇에 이르기까지 세 가지 전이 시나리오에서 평가되었다. 실험 결과, ASAP는 SysID, DR 및 델타 역학 학습 기반선들에 비해 모션 트래킹 오류를 현저히 줄이고, 다양한 동적 모션에서 민첩성과 전신 조정을 크게 개선했다. 이 연구는 시뮬레이션과 실제 물리학 간의 간극을 메우기 위한 유망한 방향성을 제시한다.



### Beyond Yes or No: Predictive Compliance Monitoring Approaches for Quantifying the Magnitude of Compliance Violations (https://arxiv.org/abs/2502.01141)
- **What's New**: 본 연구는 기존의 프로세스 준수 모니터링 접근 방식이 준수 위반을 사후적으로 감지하는 데 중점을 두었다는 점을 강조합니다. 제안된 새로운 두 가지 예측 준수 모니터링 접근 방식은 각 프로세스 인스턴스가 원하는 상태에서 얼마나 이탈했는지를 측정할 수 있는 능력을 제공합니다. 이는 기업들이 운영 성과에 대한 인사이트를 제공받아 비준수 위험을 줄이고 보다 정보에 기반한 의사결정을 내릴 수 있도록 돕습니다.

- **Technical Details**: 제안된 접근 방식은 이탈 정도를 수량화할 수 있는 능력을 기존의 술어 예측 (predicate prediction) 방법에 확장하여 제공합니다. 첫 번째 접근법은 이진 분류 문제를 하이브리드 작업으로 재구성해 분류와 회귀를 동시에 고려하고, 두 번째 접근법은 다중 작업 학습 (multi-task learning) 방법을 사용하여 비준수 상태와 비위반 정도를 동시에 예측합니다. 이 연구에서는 특히 시간 제약 (temporal constraints)에 중점을 두고 있으며, 이는 건강 관리 및 물류와 같은 거의 모든 응용 분야에서 중요합니다.

- **Performance Highlights**: 합성 및 실제 이벤트 로그를 기반으로 한 평가 결과, 제안된 접근 방식, 특히 다중 작업 학습 기법은 비준수 정도를 수량화할 수 있는 능력을 보여주었습니다. 또한 현 상태에서 활용 가능한 최신 술어 예측 접근 방식과 유사한 성능을 유지하면서 예측된 준수 상태의 정확성을 확보했습니다. 이는 비즈니스 조직들이 운영상의 성과를 체계적으로 분석하고 조치를 취할 수 있도록 가능하게 합니다.



### Self-Organizing Interaction Spaces: A Framework for Engineering Pervasive Applications in Mobile and Distributed Environments (https://arxiv.org/abs/2502.01137)
Comments:
          9 pages, 3 listings

- **What's New**: 이번 논문에서는 Self-Organizing Interaction Spaces (SOIS)라는 새로운 프레임워크를 도입하며, 이는 모바일 애플리케이션을 위한 자율적이고 적응적인 아키텍처를 가능케 합니다. 5G 기술이 발전함에 따라, D2D(Device-to-Device) 통신과 같은 새로운 기능들이 제공되어, 응용 프로그램이 모바일 노드 간의 직접적인 상호작용을 통해 더 많은 자율성과 효율성을 누릴 수 있게 됩니다.

- **Technical Details**: SOIS는 각 모바일 노드의 개별적 및 사회적 맥락에 따라 적응적 조직 구조를 형성할 수 있도록 설계되었습니다. 이 프레임워크는 조직적 사고를 이용해 다양한 전방향 애플리케이션을 모델링하고 프로그래밍하는 두 가지 주요 추상화를 제공합니다. 이 접근법은 동적 조직 구조를 유지하고 적응하도록 만드는 자기 조직화 메커니즘을 도입합니다.

- **Performance Highlights**: 시뮬레이션된 모바일 군집 감지 응용 프로그램의 사례를 통해 SOIS의 실현 가능성과 이점이 입증되었습니다. SOIS는 전통적인 클라우드 모델에 대한 의존도를 줄이면서 효율성을 향상시킬 수 있는 잠재력을 보여주며, 이는 모바일 및 분산 환경에 혁신적인 솔루션을 위한 길을 열어가는 데 기여할 것입니다.



### Deep Reinforcement Learning for Dynamic Resource Allocation in Wireless Networks (https://arxiv.org/abs/2502.01129)
Comments:
          6 pages, 8 figures

- **What's New**: 본 연구에서는 무선 통신 시스템에서의 동적 리소스 할당을 위한 딥 강화 학습(Deep Reinforcement Learning, DRL) 알고리즘의 응용을 다루고 있습니다. 이는 기지국, 다수의 안테나 및 사용자 장치를 포함하는 시뮬레이션 환경을 구성하고, RLlib 라이브러리를 활용한 다양한 DRL 알고리즘을 적용하여 옵티마이즈된 리소스 할당 결정을 내리는 방법론을 제안합니다. 결과적으로, DRL의 접근 방식이 전통적인 방법에 비해 더욱 효율적인 리소스 할당을 제공한다는 것을 보여줍니다.

- **Technical Details**: 이 연구에서는 DRL 기반의 동적 리소스 할당 방식으로 Proximal Policy Optimization (PPO) 알고리즘을 활용하였습니다. PPO는 그 견고성과 효율성 덕분에 리소스 할당 정책을 배우기 위해 RL 에이전트를 훈련시키는 데 적합합니다. DQN(Deep Q-Network)과의 비교 분석을 통해 PPO가 학습 효율성과 시스템 최적화 면에서 우수한 성과를 달성했음을 입증했습니다.

- **Performance Highlights**: 제안된 DRL 기반 프레임워크는 동적 리소스 할당을 혁신할 수 있는 잠재력을 지니고 있으며, 복잡한 리소스 할당 정책을 자율적으로 학습함으로써 네트워크 성능과 적응성을 크게 향상시킬 수 있습니다. 또한, 연구진의 성과는 PPO 기반 접근 방식이 전통적인 중앙 집중식 알고리즘보다 더 빠른 수렴 속도와 뛰어난 효과성을 보임을 확인시켜줍니다.



### The Battling Influencers Game: Nash Equilibria Structure of a Potential Game and Implications to Value Alignmen (https://arxiv.org/abs/2502.01127)
Comments:
          9 pages, 8 figures, submitted to ICML

- **What's New**: 이번 연구는 Battling Influencers Game (BIG)이라는 다중 플레이어 게임을 통해 여러 인플루언서들이 서로의 존재를 고려하여 전략을 조정하는 방식을 설명합니다. 특히, 이 게임이 잠재적 게임(potential game)이며, 순수 내쉬 균형(pure Nash equilibria)이 존재함을 증명합니다. 흥미롭게도, 특정 균형 상황에서 인플루언서들은 자신의 행동을 최대한 과장할 필요가 있다고 합니다. 이러한 행동이 정보 허위 생성의 원인을 밝히는 데 기여할 수 있습니다.

- **Technical Details**: Battling Influencers Game (BIG)은 n명의 인플루언서가 동시에 움직이는 게임으로, 각 인플루언서는 공통의 연속 행동 공간을 가집니다. 각 행동은 인플루언서가 수신자에게 미치는 영향을 나타내며, 각 인플루언서는 고유한 목표를 가지고 수신자의 결정을 조정하려고 합니다. 이 게임은 최적화 방법을 통해 순수 내쉬 균형을 찾아낼 수 있으며, 모든 인플루언서는 일반적으로 자신의 의견을 과장하는 경향이 있다고 합니다.

- **Performance Highlights**: BIG의 분석 결과, 인플루언서들이 서로의 행동을 고려할 때 과장된 전략을 선호하게 되는 경향이 확인되었습니다. 이 게임의 결과는 기존의 가치 정렬(value alignment) 문제에도 적용될 수 있습니다. 예를 들어, 인플루언서들이 잘못된 데이터를 생성하는 유인으로 작용할 수 있으며, 이는 AI 시스템의 가치 정렬 알고리즘 설계에 중요한 시사점을 제공합니다.



### Large Language Model-Enhanced Multi-Armed Bandits (https://arxiv.org/abs/2502.01118)
Comments:
          Preprint

- **What's New**: 대형 언어 모델(LLMs)을 이용한 멀티암 밴딧(MAB) 문제 해결이 주목받고 있으나, 직접적인 팔 선택 방식은 비효율적임이 증명되었습니다. 이 연구에서는 고전적인 MAB 알고리즘과 LLM의 강점을 결합한 새로운 접근 방식을 제안합니다. 이를 통해 LLM을 활용해 보상 예측을 수행하고, 최적화된 MAB 알고리즘을 구현합니다.

- **Technical Details**: 제안하는 접근법은 두 가지 고전적인 MAB 알고리즘, 즉 Thompson Sampling(TS)과 회귀 오라클 기반 MAB 알고리즘을 사용합니다. TS 알고리즘에서는 LLM을 이용해 보상을 샘플링하고, LLM의 온도 조절을 통해 탐색과 착취의 균형을 맞춥니다. 회귀 오라클 방식에서는 LLM을 통해 보상을 예측하며, LLM의 온도를 0으로 설정하여 예측의 확실성을 높입니다.

- **Performance Highlights**: 실험 결과, 제안하는 TS-LLM 및 RO-LLM 알고리즘은 이전의 팔 선택 방식보다 우수한 성능을 보입니다. 또한, 의미 있는 정보가 드물고 복잡한 환경에서도 제안된 알고리즘이 직접적인 LLM 기반 팔 선택 방식보다 성능이 월등히 뛰어난 것을 보여주었습니다. 이 연구는 LLM을 활용한 실세계 결정 문제 해결에 대한 유용한 지침을 제공할 것으로 기대됩니다.



### Learning to Learn Weight Generation via Trajectory Diffusion (https://arxiv.org/abs/2502.01117)
- **What's New**: 본 논문에서는 Lt-Di라는 새로운 방법을 제안합니다. 이 방법은 기존의 diffusion 알고리즘을 메타러닝(meta-learning)과 통합하여 보지 못한 작업이나 도메인에 대한 가중치를 효과적으로 생성할 수 있도록 합니다. 특히, Trajectory Diffusion(경로 확산)을 통해 최적화 과정에서 다른 가중치의 가치도 활용하며, 학습과 추론을 더욱 효율적으로 개선합니다.

- **Technical Details**: Lt-Di는 세 가지 단계인 가중치 준비, 메타 학습, 평가로 구성된 워크플로우를 가지고 있습니다. 가중치 준비 단계에서는 각 작업에 대해 최적화 경로를 구성하고, 최종 가중치가 최적화되도록 주의를 기울입니다. 메타 학습 단계에서는 REPTILE 프레임워크를 사용하여 여러 작업에서 빠르게 학습할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, Lt-Di는 제로샷(zero-shot) 및 몇 샷(few-shot) 학습, 다중 도메인 일반화, 대규모 언어 모델에 대해 높은 정확도를 달성하면서도 계산 오버헤드를 줄이는 성능을 보입니다. 이러한 특징으로 인해, Lt-Di는 다양한 작업에서 높은 성능을 발휘하는 혁신적인 접근법으로 평가됩니다.



### GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation (https://arxiv.org/abs/2502.01113)
Comments:
          19 pages, 6 figures

- **What's New**: GFM-RAG는 기존의 RAG 모델들을 개선하기 위해 개발된 새로운 그래프 기반 모델로, 복잡한 쿼리-지식 관계를 효과적으로 포착할 수 있도록 설계되었습니다. 이 모델은 800만 개의 파라미터를 갖춘 혁신적인 그래프 신경망(graph neural network)을 사용하여 복잡한 질문에 대한 응답을 보다 효율적으로 생성합니다. GFM-RAG는 기존의 동적 그래프 모델에서 발생하는 노이즈와 불완전성 문제를 해결하며, 사전 조정 없이도 이전에 보지 못한 데이터셋에서 저명한 성능을 보입니다.

- **Technical Details**: GFM-RAG는 60개의 지식 그래프와 1400만 개의 트리플로 구성된 대규모 데이터세트에서 이중 단계의 훈련 과정을 거쳐 학습됩니다. 실험에서는 HotpotQA, MuSiQue, 2WikiMultiHopQA와 같은 세 개의 다중 홉 QA 데이터셋을 활용하였으며, 700,000개의 문서에서 표본을 추출하여 KG 인덱스를 구축합니다. GFM-RAG는 다양한 도메인에 걸쳐 7개의 특정 RAG 데이터셋을 대상으로 평가하여 일반화 가능성을 확인하였습니다.

- **Performance Highlights**: GFM-RAG는 다중 홉 QA 데이터셋 및 도메인 특정 데이터셋에서 최신 기술 수준의 성능을 달성하였으며, 신경망 확장 법칙(neural scaling laws)과의 일치를 유지합니다. extensive 임상 실험을 통해 경쟁 모델들과의 비교에서 우위를 점하며, 데이터 효율성과 성능의 개선 가능성이 확인되었습니다. 이러한 결과는 GFM-RAG가 앞으로 더 많은 연구와 개발의 잠재력을 지니고 있음을 보여줍니다.



### A generative foundation model for an all-in-one seismic processing framework (https://arxiv.org/abs/2502.01111)
- **What's New**: 본 논문에서는 generative diffusion models (GDMs)에 기반한 generative seismic foundation model (GSFM)을 제안합니다. GSFM은 노이즈 제거, 반사 노이즈 감소, 보간 및 저주파 성분의 외삽 등 여러 지진 처리 작업을 수행할 수 있는 통합된 프레임워크입니다. 이 모델은 합성 데이터에 대한 사전 학습을 통해 깨끗하고 완전한 지진 데이터의 분포 특성을 캡처하며, 필드 데이터에 맞춰 반복적 미세 조정 전략을 적용합니다.

- **Technical Details**: GSFM은 목표 지향적 확산 과정 예측을 채택하여 계산 효율성을 향상시키면서도 정확성을 유지합니다. 이 모델은 고전적인 사전 훈련 및 미세 조정 전략에 비해 유사한 아키텍처의 벤치마크를 초월하는 성능을 입증했습니다. 또한, GSFM의 확률적 특성은 효과적인 불확실성 정량화를 가능하게 하여, 처리 결과의 신뢰성에 대한 귀중한 통찰을 제공합니다.

- **Performance Highlights**: 합성 데이터 테스트 결과, GSFM은 모든 작업에서 동등한 아키텍처의 기준선을 초과하는 성능을 보여주었으며, 전통적인 사전 훈련 전략에 비해 미세 조정 후에도 상응하는 성능을 달성했습니다. 필드 데이터 테스트에서는 반복적 미세 조정 접근 방식이 기존의 사전 훈련 및 미세 조정 패러다임의 일반화 한계를 해결하여, 다양한 작업에서 성능을 크게 향상시킨 것으로 나타났습니다.



### Pulse-PPG: An Open-Source Field-Trained PPG Foundation Model for Wearable Applications Across Lab and Field Settings (https://arxiv.org/abs/2502.01108)
Comments:
          The first two listed authors contributed equally to this research

- **What's New**: 최근의 Photoplethysmography (PPG) 기반 모델들이 다양한 건강 애플리케이션에서의 활용 가능성으로 주목받고 있습니다. 본 논문에서는 100일간 120명의 참가자로부터 수집된 원시 PPG 데이터를 기반으로 훈련된 최초의 오픈소스 PPG 기초 모델인 Pulse-PPG를 소개합니다. 기존의 PPG 모델들은 주로 임상 데이터에 기반하거나 폐쇄형 모델로서 실제 환경에서의 적용에 한계를 보였습니다.

- **Technical Details**: Pulse-PPG는 21억 점의 데이터를 포함한 대규모 웨어러블 필드 데이터셋에서 직접 훈련되었으며, 다양한 다운스트림 작업과 데이터셋에서 평가되어 최첨단 성능을 보였습니다. 필드 데이터를 활용하여 일반적인 노이즈 패턴을 학습함으로써 실세계 환경에서의 더 우수한 일반화를 가능하게 하였습니다. 이 모델은 상대적 대조 학습 손실(Relative Contrastive Learning Loss)과 학습 가능한 모티프 기반 거리 함수(Learnable Motif-Based Distance Function)를 사용하여 미세한 의미 있는 패턴을 추출합니다.

- **Performance Highlights**: Pulse-PPG는 다운스트림 평가 작업 11개 중 10개에서 기존의 최첨단 오픈 소스 PPG 기초 모델을 초과하는 성능을 보였습니다. 이 모델은 연구자들이 제약 없이 더욱 일반화 가능한 PPG 기반 모델을 개발할 수 있도록 데이터를 공유함으로써 연구의 진전을 가속화할 것입니다. 결과적으로, 필드 데이터에 대한 훈련이 임상 데이터보다 많은 작업에서 우수한 성능을 발휘하는 것을 보여주었습니다.



### VidSketch: Hand-drawn Sketch-Driven Video Generation with Diffusion Contro (https://arxiv.org/abs/2502.01101)
Comments:
          17pages, 15 figures

- **What's New**: VidSketch는 손으로 그린 스케치와 텍스트 프롬프트에서 직접 높은 품질의 비디오 애니메이션을 생성할 수 있는 첫 번째 방법으로, 기존의 정적 이미지 생성 기술의 한계를 극복했습니다. 이 방법은 비전문가들이 쉽게 애니메이션을 제작할 수 있도록 하여 예술적 요구를 충족시킵니다. 특히, Level-Based Sketch Control Strategy를 도입하여 다양한 사용자 수준의 드로잉 기술에 맞춰 스케치의 가이던스 강도를 조정할 수 있도록 설계되었습니다.

- **Technical Details**: 핵심 기술로는 Temporal Attention과 TempSpatial Attention 메커니즘이 포함되어 있습니다. TempSpatial Attention은 생성된 비디오 애니메이션의 시공간 일관성을 향상시켜 각 프레임 간의 연속성을 확보합니다. 손으로 그린 스케치의 추상화 수준을 동적으로 평가하여 비디오 생성 과정에서 제어 강도를 조절하는 방식으로 실현됩니다.

- **Performance Highlights**: VidSketch 모델을 평가한 결과, 손으로 그린 스케치와 잘 일치하면서 높은 비디오 품질과 미적 매력, 스타일 다양성 및 시공간 일관성을 유지하는 것으로 나타났습니다. 이 연구는 일상 사용자들이 비디오 애니메이션 제작에 접근할 수 있도록 하여 예술적 장벽을 제거하며, 다양한 고품질 결과를 제시하는 데 기여하고 있습니다.



### Enhancing Aspect-based Sentiment Analysis with ParsBERT in Persian Languag (https://arxiv.org/abs/2502.01091)
- **What's New**: 이번 논문은 페르시아어 텍스트 마이닝에서의 주요 과제를 해결하는 데 초점을 맞추고 있습니다. 연구자들이 직면한 문제인 페르시아어 데이터셋의 부족과 기존 언어 모델의 비효율성을 해결하기 위해 새로운 접근법을 제안합니다. 이 접근법은 페르시아어에 맞춤화된 언어 모델을 더욱 효과적으로 개선하는 것을 목표로 하고 있습니다.

- **Technical Details**: 저자들은 ParsBERT 모델을 활용하여 감정 분석을 수행하며, 이와 관련된 어휘집을 보강하여 정확도를 높였습니다. 연구는 페르시아 웹사이트 'Digikala'에서 추출한 사용자 의견을 분석하여, 세부적인 감정(sentiment)을 이해하는 데 초점을 둡니다. 제안된 방법은 텍스트의 의미론적 능력을 크게 향상시킵니다.

- **Performance Highlights**: 실험 결과는 제안된 방법의 정확도 88.2%와 F1 점수 61.7이라는 우수한 성과를 보여줍니다. 이 연구는 사용자 생성 콘텐츠에서 미세한 감정을 추출하는 데 기여하며, 페르시아어 텍스트 마이닝 분야에서 감정 분석의 효율성과 정확성을 높이는 데 중요한 역할을 할 것입니다.



### Classic4Children: Adapting Chinese Literary Classics for Children with Large Language Mod (https://arxiv.org/abs/2502.01090)
Comments:
          Accepted at NAACL 2025 Findings

- **What's New**: 이번 연구에서는 어린이들이 이해할 수 있도록 중국 문학 클래식을 교묘하게 각색하는 작업인 아동 친화적 문학 각색(Child-Friendly Literary Adaptation, CLA) 과제를 제안합니다. 기존의 대형 언어 모델(LLMs)이 아동의 독서 선호를 효과적으로 반영하지 못하는 점을 개선하기 위해 InstructChild라는 방법을 고안하였습니다. 이 방법은 각색 과정에서 인물의 성격과 내러티브 구조를 반영해 어린이의 흥미를 유도하려는 목적을 가지고 있습니다.

- **Technical Details**: InstructChild는 미세 조정(fine-grained instruction tuning), 가독성 지표(readability metric) 설계, 그리고 미리 보기 디코딩 전략(lookahead decoding strategy)이라는 세 가지 주요 기술로 구성되어 있습니다. 첫째, 인물의 성격과 내러티브 구조를 기반으로 LLM을 조정하여 아동 친화적인 텍스트 생성을 돕습니다. 둘째, 중국어 가독성 지표는 아동이 이해하기 쉬운 문장을 생성하도록 LLM을 안내합니다. 마지막으로, 미리 보기 디코딩 전략은 현재 토큰 선택을 위해 잠재적인 후속 토큰의 영향을 고려합니다.

- **Performance Highlights**: 실험 결과 InstructChild를 적용한 경우 기존 LLM에 비해 자동 및 인간 평가 모두에서 우수한 성능 향상을 나타냈습니다. Classic4Children 데이터셋을 통해 원본 텍스트와 아동 친화적 버전 간의 효과를 평가하며 각색된 텍스트가 아동의 독서 수준에 적합하게 향상된 것을 보여주었습니다. 이 연구는 향후 연구를 위한 토대를 제공하며, 코드와 데이터셋은 공개하여 추가적인 연구를 촉진할 계획입니다.



### Advanced Architectures Integrated with Agentic AI for Next-Generation Wireless Networks (https://arxiv.org/abs/2502.01089)
Comments:
          6 Pages

- **What's New**: 이 논문은 6G 네트워크의 혁신적인 아키텍처와 기술을 탐구하여 운영 비용을 줄이고 새로운 서비스 모델을 가능하게 하려는 다양한 접근 방식을 제안합니다. 주요 내용으로는 제어 및 사용자 평면을 분리한 새로운 6G 아키텍처의 제안과 '컨스트레인드 AI (constrained AI)' 기법을 통한 에너지 최적화 및 실시간 학습을 목표로 하고 있습니다.

- **Technical Details**: 논문에서는 사용자와 제어 평면의 분리를 통해 서비스 배포 및 운영을 보다 효율적으로 할 수 있는 네트워크 아키텍처의 설계를 강조합니다. '서버리스 컴퓨팅 (serverless computing)', 자율 인지 에이전트 (autonomous cognitive agents) 등 여러 혁신 기술이 언급되며, 이는 분산 네트워크 환경에서 서비스 조정 및 관리를 지원합니다. 또한, '신경 라디오 프로토콜 스택 (Neural Radio Protocol Stacks)'을 통하여 AI 기반의 맞춤형 통신 프로토콜 개발을 도모하고 있습니다.

- **Performance Highlights**: 다양한 혁신적 요소를 통해 6G 네트워크는 에너지 효율성과 보안성을 확보할 수 있는 기반을 마련합니다. 기계 대 기계 통신, 우주 기반의 RAN 기능 클라우드화 등이 논의되며, 이는 새로운 서비스 모델과 함께 유연하고 적응 가능한 네트워크 생태계를 구현하는 데 기여할 것으로 기대됩니다. 이 모든 요소들은 궁극적으로 6G의 확장성과 지능, 효율성을 높이는 데 기여할 것입니다.



### Tool Unlearning for Tool-Augmented LLMs (https://arxiv.org/abs/2502.01083)
Comments:
this https URL

- **What's New**: 이번 논문에서는 툴 증강 대형 언어 모델(LLM)의 툴 언러닝(tool unlearning)에 대한 개념과 필요성을 소개합니다. 툴 언러닝은 보안 취약성이나 개인정보보호 규정으로 인한 필요성을 충족시키기 위해 특정 툴 사용 능력을 제거하는 것을 목표로 합니다. ToolDelete라는 새로운 알고리즘을 통해 툴 언러닝 문제를 해결하고, 툴에 대한 지식 제거와 유지, 일반적 기능 유지라는 세 가지 핵심 특성을 구현합니다.

- **Technical Details**: 툴 언러닝의 주요 목표는 특정 툴의 사용 능력을 제거하면서도 나머지 툴에 대한 지식을 유지해야 하는 것입니다. ToolDelete는 툴 지식 제거, 툴 지식 유지 및 일반적 기능 유지를 통해 이러한 도전을 해결합니다. 또한, LiRA-Tool이라는 새로운 멤버십 추론 공격(MIA) 모델을 개발하여 툴과 관련된 지식이 효과적으로 제거되었는지를 평가합니다.

- **Performance Highlights**: 실험 결과, ToolDelete는 주어진 툴을 잊는데 있어 기존 방법보다 높은 정확도를 달성하며, 사용하지 않거나 최신 도구에 대한 지식을 유지하는 데 있어 뛰어난 성능을 보입니다. ToolDelete는 또한 재교육에 비해 74.8%의 훈련 시간 절약을 가능하게 하며, 낮은 자원 환경에서도 95% 이상의 성능을 유지합니다. 이러한 특성들은 툴 언러닝의 실용성을 더욱 높입니다.



### The Jumping Reasoning Curve? Tracking the Evolution of Reasoning Performance in GPT-[n] and o-[n] Models on Multimodal Puzzles (https://arxiv.org/abs/2502.01081)
- **What's New**: OpenAI의 o1 및 o3 모델 공개는 대규모 언어 모델(Large Language Models, LLMs)의 추론 능력에 큰 변화를 의미합니다. 특히 o3는 인공지능의 일반적 지능을 테스트하는 Abstraction and Reasoning Corpus(ARC-AGI)에서 인간을 초월하는 문제 해결 능력을 보였습니다. 그러나 symbolic pattern에 한정된 기존 벤치마크를 넘어, 다양한 시각 및 언어 데이터를 포함한 멀티모달 시나리오에 대한 탐구가 절실하다는 점이 강조되었습니다.

- **Technical Details**: 이 연구에서는 GPT-[n] 및 o-[n] 시리즈 모델의 멀티모달 퍼즐 해결 능력을 평가합니다. PuzzleVQA와 AlgoPuzzleVQA 데이터셋을 통해 추상 시각 추론과 알고리즘적 문제 해결을 측정하였으며, 각 데이터셋은 모델의 인지 능력을 면밀히 시험하는 구조적 특성을 지니고 있습니다. 특히, 멀티모달 퍼즐은 모델이 시각적 정보와 텍스트 정보를 통합하여 문제를 해결하는 능력을 평가하는 중요한 벤치마크 역할을 합니다.

- **Performance Highlights**: 모델 버전이 진행됨에 따라 추론 능력의 증가 경향이 뚜렷히 나타났으나, o1 모델은 여전히 단순 멀티모달 퍼즐에서 어려움을 겪고 있습니다. 알고리즘 퍼즐에서의 성능 저하도 관찰되며, 이는 현재의 인공지능이 인간의 추론 능력과 아직 큰 격차가 있음을 시사합니다. 연구진은 지속적으로 새로운 모델의 성능을 추적하고 연구 결과를 업데이트할 계획입니다.



### Learning Nonlinearity of Boolean Functions: An Experimentation with Neural Networks (https://arxiv.org/abs/2502.01060)
Comments:
          To be published in International conference on Artificial Intelligence and Sustainable Computing, AISC 2024

- **What's New**: 이 논문은 불리안 함수(Boolean function)의 비선형성(nonlinearity) 속성을 신경망(neural networks)을 사용하여 학습할 수 있는 가능성을 조사합니다. 저자들은 트루스 테이블(truth table) 형태의 불리안 함수 예제와 그에 해당하는 비선형성 값을 바탕으로 비선형성을 예측하는 인코더 스타일의 딥 신경망을 훈련시킵니다. 4개 및 5개 변수의 함수에 대해 95% 이상의 정확도로 비선형성을 예측할 수 있다는 긍정적인 결과를 보고합니다.

- **Technical Details**: 불리안 함수는 암호 시스템의 기초 요소이며, 비선형성, 상관 면역(correlation immunity), 균형성(balancedness) 등의 속성이 중요합니다. 이 연구에서는 신경망이 불리안 함수의 Walsh 스펙트럼을 학습할 수 있음을 보여주며, 최소 2^2n개의 불리안 함수와 그에 상응하는 Walsh 스펙트럼을 필요로 합니다. 그러나 더 큰 변수 수에 대한 비선형성을 예측하려는 시도는 성공하지 못했습니다.

- **Performance Highlights**: 신경망을 활용한 비선형성 예측에서, 특히 4개 및 5개 변수의 경우 높은 정확도를 달성했지만, 더 많은 변수에 대해서는 메모리와 시간 효율성이 전통적인 알고리즘보다 낮았습니다. 그럼에도 불구하고 작은 변수 집합에서 신경망의 결과는 상당히 긍정적이며, 추후 연구와 아이디어 탐색이 필요하다는 점이 강조됩니다.



### Knowledge Synthesis of Photosynthesis Research Using a Large Language Mod (https://arxiv.org/abs/2502.01059)
Comments:
          17 pages, 6 figures

- **What's New**: 최근 생물학적 데이터 분석 도구와 대규모 언어 모델(LLMs)의 발전으로 식물 과학 연구에서 AI를 활용할 새로운 가능성이 열렸습니다. 이러한 연구에서 기존 LLM의 한계를 극복하기 위해 OpenAI의 GPT-4o 모델을 기반으로 하는 광합성 연구 보조 도구(prag, Photosynthesis Research Assistant)인 PRAG가 제안되었습니다. PRAG는 RAG(retrieval-augmented generation) 기술과 프롬프트 최적화를 통해 정확한 과학적 정보를 제공하도록 설계되었습니다.

- **Technical Details**: PRAG는 OpenAI의 GPT-4o 모델을 기반으로 하며, 자동화된 피드백 루프와 벡터 데이터베이스를 사용하여 광합성 관련 쿼리에 대한 응답의 정확성과 관련성을 향상시켰습니다. 이 시스템은 RAG 모델을 구현하여 외부 데이터베이스에서 관련 문서를 검색하고, 그에 따라 응답을 생성하는 구조를 갖추고 있습니다. 또한, RAG Evaluator와 Prompt Reviser를 통해 지속적인 피드백과 개선을 수행합니다.

- **Performance Highlights**: PRAG는 과학적 글쓰기와 관련된 다섯 가지 지표에서 평균 8.7% 개선을 보여주었으며, 출처 투명성은 25.4% 증가했습니다. PRAG의 과학적 깊이와 분야 커버리지는 기존의 광합성 연구 논문과 비교할 때 유사한 수준을 보였으며, 사용된 지식 그래프를 통해 63%와 39.5%의 키 엔티티 일치를 달성했습니다. 이를 통해 PRAG는 광합성 연구 및 광범위한 식물 과학 분야에서 데이터 분석 및 예측 기능을 향상시킬 수 있습니다.



### FetDTIAlign: A Deep Learning Framework for Affine and Deformable Registration of Fetal Brain dMRI (https://arxiv.org/abs/2502.01057)
- **What's New**: 이 논문에서는 금속기반 희소 분광 이미징(mMRI)에서 태아 두뇌의 미세구조를 분석하고, 이를 위해 FetDTIAlign라는 심층 학습(deep learning) 방법론을 제안합니다. FetDTIAlign는 정확한 affine 및 변형 정렬(deformable alignment)을 지원하여 기존의 고품질 성인의 데이터에서 기인한 등록 문제가 해결됩니다. 이 방법은 태아 두뇌의 발전에 대한 단일 개인에 대한 정밀한 비교 분석을 가능하게 하여 초기 뇌 발달에 대한 새로운 발견을 지원합니다.

- **Technical Details**: FetDTIAlign는 이중 인코더 구조와 반복적인 특성 기반 추론을 특징으로 하여 잡음(noise) 및 낮은 해상도의 영향을 줄입니다. 이 방법은 각 등록 단계에서 네트워크 구성과 도메인 특성을 최적화하여 견고성과 정확성을 높입니다. 23주에서 36주 사이의 데이터에서 60개의 백질 경로를 아우르는 검증을 통해, FetDTIAlign는 두 가지 전통적인 최적화 방법과 심층 학습 파이프라인에 비해 일관되게 우수한 성능을 평가받았습니다.

- **Performance Highlights**: FetDTIAlign은 Developing Human Connectome Project의 외부 데이터에 대한 추가 검증을 통해 다양한 획득 프로토콜에 걸쳐 일반화 가능성을 확인했습니다. 이 방법은 기존 기술에 비해 더 정확하고 신뢰할 수 있는 대안을 제공하는 것으로 나타났습니다. 태아 두뇌의 dMRI 등록에 대한 심층 학습의 사용 가능성을 증명함으로써, FetDTIAlign는 뇌 발달 분석의 새로운 장을 열고 있습니다.



### Sparks of Explainability: Recent Advancements in Explaining Large Vision Models (https://arxiv.org/abs/2502.01048)
Comments:
          Doctoral thesis

- **What's New**: 이 논문은 컴퓨터 비전의 설명 가능성을 높이는 고급 접근 방식을 다루고 있습니다. 특히, 딥 러닝 네트워크가 이용하는 특징들을 분석하고 모델링하여 새로운 방법을 제안하고 있습니다. 여기에는 알고리즘 안정성에 기반한 메트릭과 Sobol 지수를 활용한 계산 속도 향상 기법이 포함됩니다.

- **Technical Details**: 이 연구에서는 saliency maps와 같은 귀속 방법(attribution methods)을 평가하며, 알고리즘 안정성을 바탕으로 한 새로운 메트릭을 도입합니다. 또한, EVA 방법은 검증된 섭동 분석(verified perturbation analysis)을 통해 귀속에 대한 공식적인 보장을 처음으로 제시합니다. 실험 결과는 복잡한 상황에서 이러한 방법들이 모델이 '어디에' 집중하는지 파악할 수 있지만, '무엇을' 인식하는지는 명확히 설명하지 못한다는 것을 보여줍니다.

- **Performance Highlights**: 논문은 두 가지 가설을 탐구합니다: 인간의 추론을 모델과 일치시키기 위한 훈련 루틴의 도입과 개념적 설명 가능성 접근 방식의 채택입니다. CRAFT 방법은 모델이 사용하는 개념을 자동으로 추출하고 그 중요성을 평가하며, MACO는 이를 시각화하는 기능을 제공합니다. 이러한 연구들은 ResNet 모델의 1000개 ImageNet 클래스에 적용된 상호작용 시연을 통해 통합된 프레임워크로 수렴하게 됩니다.



### eagle: early approximated gradient based learning rate estimator (https://arxiv.org/abs/2502.01036)
Comments:
          43pages, 24figures

- **What's New**: 이번 논문에서는 EAGLE 업데이트 규칙을 제안합니다. 이는 학습 초기 단계에서 손실(loss) 수렴을 가속화하기 위해 현재 및 이전 단계의 파라미터(parameter)와 그래디언트(gradient) 값을 활용하는 새로운 최적화 방법입니다. EAGLE 알고리즘은 연속적인 훈련 단계를 통한 파라미터와 그래디언트의 변화를 계산하여 최적의 파라미터를 추정합니다.

- **Technical Details**: EAGLE 업데이트 규칙은 손실 경관(loss landscape)의 국소적(curvature) 특성을 활용하여 파라미터 변화와 그래디언트의 변화를 계산하고, 이를 바탕으로 최적 값으로 업데이트합니다. 그러나 이러한 규칙은 잠재적인 불안정성을 가지고 있어, 이를 해결하기 위해 Adam 및 EAGLE 업데이트 규칙 사이를 동적으로 전환하는 적응형 전환 메커니즘을 도입했습니다.

- **Performance Highlights**: 표준 벤치마크 데이터셋에서 수행한 실험 결과, EAGLE 옵티마이저는 새로운 업데이트 규칙과 전환 메커니즘을 결합하여 기존의 최적화 방법들과 비교했을 때, 더 적은 에폭(epoch) 수로 빠른 손실 수렴을 달성했습니다.



### Comprehensive Modeling Approaches for Forecasting Bitcoin Transaction Fees: A Comparative Study (https://arxiv.org/abs/2502.01029)
- **What's New**: 이번 연구는 비트코인 거래 수수료 예측의 중요성을 강조하며, 6개의 예측 모델(SARIMAX, Prophet, Time2Vec, Attention 기반 Time2Vec, SARIMAX와 Gradient Boosting 혼합 모델, Temporal Fusion Transformer) 비교 분석을 통해 수수료 예측의 효과성을 체계적으로 평가했습니다. 다양한 mempool 메트릭, 네트워크 파라미터 및 역사적 수수료 패턴을 통합하여 수수료 행동의 복합적인 역학을 포착하는 접근 방식을 제시합니다. 분석 결과, 전통적인 통계적 방법이 복잡한 딥러닝 아키텍처보다 더 나은 성능을 발휘하는 것으로 나타났습니다.

- **Technical Details**: 이 연구는 비트코인 거래의 중앙 수수료율을 24시간(144블록) 동안 예측하기 위해 91일에 걸친 데이터셋(11809개 기록)을 활용했습니다. 23개의 특징으로 구성된 데이터는 거래 및 mempool 메트릭, 블록 정보, 네트워크 파라미터 및 시간적 특징으로 체계적으로 정리되었습니다. 데이터 수집은 Bitcoin Core 서버에서 이루어졌고, 전처리 과정에서 결측치는 보간법을 통해 보완하고 이상치는 퍼센타일 기반 클리핑으로 처리하였습니다.

- **Performance Highlights**: SARIMAX 모델은 독립 테스트 세트에서 가장 높은 정확도를 기록하며, Prophet 모델은 교차 검증 과정에서 우수한 성능을 보였습니다. 반면, 복잡한 딥러닝 모델인 Time2Vec과 TFT는 제한된 91일 데이터셋으로 인해 상대적으로 낮은 예측력을 보여주었으며, 이러한 성능 차이는 긴 기간의 히스토리 데이터가 필요할 수 있음을 시사합니다. 이 연구는 암호화폐 이해관계자들에게 수수료 결정에서 실질적인 안내를 제공하며, 모델 선택 시 데이터 제약을 고려해야 함을 강조합니다.



### Refining Adaptive Zeroth-Order Optimization at Eas (https://arxiv.org/abs/2502.01014)
- **What's New**: 이 논문에서는 기존의 zeroth-order (ZO) 최적화 방법의 한계를 극복하기 위해 Refined Adaptive Zeroth-Order Optimization (R-AdaZO)를 제안합니다. R-AdaZO는 첫 번째 모멘트 추정치를 이용하여 ZO 기울기 추정의 정확성과 안정성을 개선하며, 두 번째 모멘트 추정치를 조정하여 최적화 경관의 기하학을 더 잘 설명합니다. 이를 통해 기존의 ZO 알고리즘보다 성능이 향상될 수 있음을 보여주고 있습니다.

- **Technical Details**: R-AdaZO는 ZO 최적화에서 첫 번째 모멘트 추정의 분산 감소 효과를 밝혀내고 이를 통해 ZO 업데이트의 정확성을 높입니다. 또한, 분산이 감소된 기울기 추정치를 기반으로 두 번째 모멘트를 개선하여 최적화 과정에서 더 효과적인 업데이트 범위를 제공합니다. 이론적 분석을 통해 R-AdaZO의 분산 감소와 빠른 수렴성을 증명하고 있으며, 이 방법은 고차원 및 복잡한 설정에서 강력한 성능을 보입니다.

- **Performance Highlights**: 다양한 실험을 통해 R-AdaZO는 합성 문제, 블랙박스 적대적 공격, 대형 언어 모델의 메모리 효율적인 파인튜닝 등에서 기존 방법보다 뛰어난 수렴 성능을 보여주었습니다. 이러한 결과는 R-AdaZO가 실제 ZO 최적화 문제에 있어 향상된 솔루션을 제공한다는 점을 강조합니다. 결론적으로, R-AdaZO는 ZO 최적화의 개선된 대안으로 자리매김할 것으로 기대됩니다.



### Encrypted Large Model Inference: The Equivariant Encryption Paradigm (https://arxiv.org/abs/2502.01013)
- **What's New**: 이 논문에서는 Equivariant Encryption (EE)이라는 새로운 암호화 기법을 소개하여 대규모 딥러닝 모델의 안전한 추론을 가능하게 합니다. EE는 데이터 암호화 과정에서 거의 제로에 가까운 성능 오버헤드를 유지하며, 이에 따라 대규모 언어 모델에서 데이터의 비밀성을 보장합니다. 전통적인 기술들이 가지는 단점들을 극복하여 안전성과 성능을 동시에 만족시키는 솔루션을 제시합니다.

- **Technical Details**: EE는 신경망의 레이어 내에서 중요한 내부 표현을 선택적으로 암호화하여, 원시 입력, 중간 활성화 및 출력이 비공식화된 인프라에서 처리될 때도 비밀이 유지되도록 설계되었습니다. 기존의 보안 다자간 연산(SMPC)과 동형 암호화(HE) 등의 기술과 차별화된 접근 방식으로 EE는 정확한 기능성을 유지하며 비선형 연산에 대한 호환성도 보장합니다. 이론적 기초를 바탕으로 다양한 아키텍처에서 EE의 적용 가능성을 설명하고 있으며, 범위가 큰 모델의 추론 과정에서 발생할 수 있는 공격 벡터 분석도 제공합니다.

- **Performance Highlights**: EE는 높은 신뢰도와 처리량을 유지하면서도 현대의 대규모 모델 추론에 필요한 엄격한 효율성 요구 사항을 충족합니다. 논문에서는 분산 환경에서의 표준 추론 파이프라인에 대한 성능 비교를 통해 EE가 가지는 이점을 입증하였습니다. 또한, EE의 구체적 구현 예를 보여 주며, 비공식적인 노드에서 쿼리 및 출력을 보호하는 방법을 제안합니다.



### MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs (https://arxiv.org/abs/2502.00997)
Comments:
          Accepted by NAACL 2025 Main

- **What's New**: 최근 대규모 언어 모델(LLMs)이 특정 영역에서 성공을 거두면서, 이러한 전문 모델을 통합하여 Mixture-of-Experts (MoE) 모델로 병합하는 방법에 대한 관심이 증가하고 있습니다. 이 연구에서는 전문가 모델 간의 파라미터 간섭(parameter interference)을 해결하고, 서로 다른 아키텍처를 가진 전문가들을 효과적으로 통합할 수 있는 새로운 병합 기법을 제시합니다. 실험을 통해 제안된 방식이 기존의 방법들보다 성능을 개선하고, MoE 병합의 실용성을 확장함을 보여줍니다.

- **Technical Details**: 연구는 다양한 도메인에서 사전 학습된 밀집 전문가 모델들을 효율적으로 병합하는 방법을 제안합니다. 이 방법은 파라미터 간섭을 줄이고, MoE 수정 없이 도메인 특화된 전문가에게 토큰 시퀀스를 라우팅하는 휴리스틱을 통한 접근 방식을 포함하고 있습니다. 논문에서는 또한 서로 다른 아키텍처의 전문가를 결합하여 동적으로 토큰 시퀀스를 적절한 전문가로 라우팅하는 새로운 방법을 개발하였습니다.

- **Performance Highlights**: 제안된 방법은 수학적 추론, 프로그래밍 및 일반 지식 벤치마크에서의 실험을 통해 기존 최첨단 방법보다 뛰어난 성능을 보였습니다. 또한, MoE 병합 후 추가적인 세부 조정에 필요한 자원을 줄이는 등의 장점을 가지며, 다양한 전문가 모델을 활용할 수 있는 가능성을 열었습니다. 이 연구는 MoE 합치기의 효율성을 높이고 더욱 다양한 응용을 가능하게 합니다.



### ChartCitor: Multi-Agent Framework for Fine-Grained Chart Visual Attribution (https://arxiv.org/abs/2502.00989)
- **What's New**: 이 논문은 ChartCitor라는 다중 에이전트 프레임워크를 제안하여 차트 질문-답변 작업을 개선합니다. 기존의 LLM(대형 언어 모델)은 신뢰할 수 없는 응답을 생성하는 경향이 있었으나, ChartCitor는 차트 이미지에서 단서 정보를 식별하여 결과를 향상시킵니다. 이 시스템은 여러 LLM 에이전트를 조정하여 데이터를 구조화하고, 답변을 재구성하며, 증거를 탐색하는 방식으로 작동합니다.

- **Technical Details**: ChartCitor는 다음과 같은 주요 에이전트로 구성됩니다: 차트에서 구조화된 데이터 테이블을 추출하는 Chart2Table Extraction Agent, 답변을 재구성하는 Answer Reformulation Agent, 테이블 데이터를 이해하기 위한 Entity Captioning Agent, 그리고 관련 테이블 셀을 찾아내는 LLM Re-ranking Agent입니다. 이 시스템은 LLM을 활용하여 차트의 비주얼 요소를 식별하고, 이를 바탕으로 증거를 제공하는 바운딩 박스를 생성합니다.

- **Performance Highlights**: ChartCitor는 다양한 차트 유형에서 기존 기준선을 초과하는 성능을 보여줍니다. 사용자 연구 결과, ChartCitor는 LLM 지원 차트 QA의 설명 가능성을 높여 사용자의 신뢰를 증가시키는 데 기여함을 시사합니다. 사용자들은 이러한 시스템을 통해 전문적인 결과를 보다 신속하게 검증하고 생산성을 높일 수 있음을 알게 되었습니다.



### PlotGen: Multi-Agent LLM-based Scientific Data Visualization via Multimodal Feedback (https://arxiv.org/abs/2502.00988)
- **What's New**: 이번 논문에서는 PlotGen이라는 새로운 다중 에이전트 프레임워크를 제안하며, 이로써 사용자 요구에 맞는 과학 데이터 시각화를 자동 생성하는 방법을 모색하고 있습니다. 이 시스템은 여러 개의 LLM 기반 에이전트를 통해 복잡한 사용자 요청을 처리하고, 중간 결과물을 반복적으로 교정하여 정확한 그래프를 생성합니다. PlotGen은 Prompting 기술을 사용하여 사용자 요구사항을 구체적인 실행 단위로 분해하며, 이를 통해 기술적 전문성이 부족한 사용자도 고급 정보 그래픽을 생성할 수 있도록 돕습니다.

- **Technical Details**: PlotGen은 (1) Query Planning Agent, (2) Code Generation Agent, (3) Numeric Feedback Agent, (4) Lexical Feedback Agent, (5) Visual Feedback Agent로 구성된 다섯 개의 unique multimodal feedback agents로 이루어져 있습니다. Query Planning Agent는 복잡한 사용자 요청을 단계별 실행 가능 작업으로 분해하며, Code Generation Agent는 사용자의 데이터를 바탕으로 pseudocode를 실행 가능한 Python 코드로 변환합니다. 각 Feedback Agent는 시각적, 언어적 및 수치적 피드백을 통해 잘못된 부분을 수정하고 최종적으로 사용자가 요구하는 시각화를 제작합니다.

- **Performance Highlights**: 실험 결과, PlotGen은 기존의 강력한 기준선 모델보다 4-6% 높은 성능을 보여 MatPlotBench 데이터셋에서 10-12%의 개선을 이루었습니다. 사용자 조사에서도 PlotGen은 시각화 생성에 대한 신뢰성을 높였으며, 초보 분석가들이 디버깅 시간을 단축시킴으로써 생산성을 증대시킬 수 있음을 확인했습니다. 이러한 성장은 LLM 기반의 시각화 생성 기술 향상에 기여할 것으로 기대됩니다.



### RandLoRA: Full-rank parameter-efficient fine-tuning of large models (https://arxiv.org/abs/2502.00987)
Comments:
          To appear at the International Conference on Learning Representations (ICLR) 2025

- **What's New**: 이 논문에서는 RandLoRA라는 새로운 기법을 소개합니다. 이 방법은 Low-Rank Adaptation (LoRA) 방식의 한계를 극복하며 전체 차원의 업데이트(full-rank updates)를 가능하게 합니다. 특히, 비훈련 가능한 임의의 랜덤 매트릭스(random matrices)의 선형 조합을 학습하여 매개변수 효율성을 높입니다.

- **Technical Details**: RandLoRA는 매개변수의 수를 줄이기 위해 고정된 랜덤 매트릭스에 대각 스케일링 매트릭스(diagonal scaling matrices)를 적용하여 최적화(optimization)합니다. 이러한 방식은 훈련 중에 매개변수(parameter)와 메모리(memory) 효율성을 유지하면서도 낮은 차원의 한계를 극복하게 합니다. 실험은 비전(vision), 언어(language), 비전-언어(vision-language) 벤치마크를 포함하여 다양한 작업에서 수행되었습니다.

- **Performance Highlights**: RandLoRA는 기존의 LoRA 방식과 비교하여 비전 및 언어 작업에서 성능 향상을 보여줍니다. 특히, 비전-언어 작업에서는 성능 격차를 현저히 줄이거나 심지어 없애는 경향을 보였습니다. 이는 RandLoRA의 유효성을 강조하며, 복잡한 작업에서 더 나은 성능을 제공하는 방법임을 증명합니다.



### Forecasting VIX using interpretable Kolmogorov-Arnold networks (https://arxiv.org/abs/2502.00980)
- **What's New**: 본 논문은 CBOE 변동성 지수(VIX) 예측에 Kolmogorov-Arnold Networks (KANs)를 적용하는 방법을 제시합니다. 전통적인 MLP 기반 신경망이 블랙박스 특성에 대한 비판을 받는 것과 달리, KAN은 배우기 가능한 스플라인 기반 활성화 함수와 기호화를 통해 해석 가능한 접근 방식을 제공합니다. 또한 KAN은 기계적 시스템 대신 최소한의 매개변수로 VIX 예측을 폐쇄형으로 표현할 수 있어 각 요소를 이해하기 쉬운 통찰력을 제공합니다.

- **Technical Details**: KAN은 Kolmogorov-Arnold 표현 정리를 기반으로 하여 복잡한 다변량 함수를 단일 변량 함수의 조합으로 표현할 수 있게 합니다. 각 네트워크 엣지에서 학습 가능한 스플라인 기반의 함수를 사용하여 구성됩니다. 이러한 구조는 해석 가능성뿐만 아니라 예측 정확성에서도 신뢰성을 제공합니다.

- **Performance Highlights**: KAN은 다양한 데이터 세트와 기간에 걸쳐 수행된 심층적인 경험적 분석을 통해 MLP 기반 신경망 모델에 비해 경쟁력 있는 예측 성능을 달성하며, 훨씬 적은 매개변수를 요구하는 것으로 나타났습니다. 이러한 결과는 KAN이 해석 가능한 금융 시계열 예측 방법으로서의 잠재력을 보여줍니다.



### ML-Dev-Bench: Comparative Analysis of AI Agents on ML development workflows (https://arxiv.org/abs/2502.00964)
- **What's New**: 이번 보고서에서는 ML-Dev-Bench라는 새로운 벤치마크를 소개합니다. 이 벤치마크는 머신 러닝(Machine Learning) 개발 작업에서의 에이전트(AI agent) 능력을 평가하기 위해 설계되었습니다. 기존의 벤치마크들이 단일 코딩 작업이나 Kaggle 스타일의 경쟁에 집중하는 반면, ML-Dev-Bench는 ML 개발 워크플로우의 복잡성을 처리하는 에이전트의 능력을 점검합니다.

- **Technical Details**: ML-Dev-Bench는 데이터셋 처리, 모델 학습, 디버깅, API 통합 등 다양한 ML 개발 능력을 평가하기 위해 25개의 고안된 작업으로 구성되어 있습니다. 각 작업은 데이터셋 처리, 모델 학습, 디버깅, API 통합 등에 대한 기술적 능력과 문제 해결 능력을 측정합니다. 또한, 각 에이전트의 성공률 및 정확도를 바탕으로 성능을 평가합니다.

- **Performance Highlights**: 세 가지 에이전트, 즉 ReAct, Openhands, AIDE의 성능이 비교되었습니다. Openhands-Sonnet이 60%의 성공률로 가장 높은 성과를 보였으며, ReAct-Sonnet이 56%를 기록했습니다. 그러나 성능 최적화 작업에서는 모든 에이전트가 실패했으며, 복잡한 작업에서는 성능이 현저히 저하되는 경향을 보였습니다.



### An MDP Model for Censoring in Harvesting Sensors: Optimal and Approximated Solutions (https://arxiv.org/abs/2502.00940)
- **What's New**: 이 논문에서는 에너지 수확 센서의 에너지 효율적인 전송을 위한 새로운 검열 정책(censoring policy)을 제안합니다. 이 문제는 무한 지평선 Markov Decision Process (MDP)로 구성되며, 전달된 모든 메시지의 중요도(utility)를 극대화하는 것을 목표로 합니다. 최적의 검열 정책은 배터리 수준에 따라 달라지는 중요도 값에 대한 임계값(threshold function)이라고 제시합니다.

- **Technical Details**: 논문에서는 상태 변수(state variables), 행동 집합(possible actions), 상태 동역학의 확률 모델(probabilistic model of state dynamics), 보상 모델(reward model)의 네 가지 주요 구성 요소로 모델을 정의합니다. 이 모델을 통해 메시지의 중요도를 평가하고, 해당 메시지를 전송할지 검열할지를 결정하는 방식으로 시스템의 동작을 최적화하는 알고리즘을 제안합니다. 특히, 제안된 기법은 모델 기반의 확률적 근사 알고리즘으로, 기존의 Q-learning 알고리즘보다 적은 계산 복잡도(computational complexity)와 빠른 수렴 속도를 자랑합니다.

- **Performance Highlights**: 수치 실험(numerical experiments)을 통해 단일 홉(single-hop) 및 다중 홉(multi-hop) 네트워크에서 제안된 체계의 분석적 이점을 검증했습니다. 에너지 의존성 전송 정책이 균형 정책(balanced policies)보다 더 효율적이라는 결과를 도출했으며, 이 연구는 비정상적 환경(이동통신 등)에서도 효과적인 성능을 보입니다. 따라서, 본 연구의 MDP 모델은 개별 노드의 성능 최적화를 보장하며, MDP 기반의 센서 네트워크가 균형 정책을 구현하는 네트워크보다 월등한 성능을 발휘할 수 있음을 보여줍니다.



### Fruit Fly Classification (Diptera: Tephritidae) in Images, Applying Transfer Learning (https://arxiv.org/abs/2502.00939)
Comments:
          15 pages and 19 figures

- **What's New**: 이 연구는 자동화된 분류를 위한 전이 학습 모델을 개발했습니다. 과거에는 전문가들이 수작업으로 분류했으나, 이는 시간과 정확성에서 한계가 있었던 문제를 해결하기 위해 AI 기반의 접근법을 제안합니다. 특히 모바일 카메라와 스테레오 현미경을 활용하여 고품질 이미지를 캡처하고, 이를 통해 데이터셋을 생성했습니다.

- **Technical Details**: 연구에서는 VGG16, VGG19, Inception-v3와 같은 사전 훈련된 합성곱 신경망 모델을 통해 학습을 수행했습니다. 이미지는 중요 형태학적 영역에 초점을 맞춰 세분화되고 라벨링되었습니다. 이 과정에서 F1-score 평가를 통해 Inception-v3가 93%로 최고의 성능을 보여주었습니다.

- **Performance Highlights**: Inception-v3의 신뢰성은 통제되지 않은 환경에서 모델을 테스트해 확인되었으며, Grad-CAM 기법을 사용하여 본질적인 형태학적 특징을 포착하는 능력을 입증했습니다. 이러한 결과는 Anastrepha fraterculus와 Ceratitis capitata를 분류하는 데 있어 Inception-v3가 효과적이고 재현 가능하다는 것을 나타내며, 자동화된 모니터링 시스템에서의 구현 가능성을 제시합니다.



### Towards Efficient Large Multimodal Model Serving (https://arxiv.org/abs/2502.00937)
- **What's New**: 이번 논문은 두 가지 주요 LMM 아키텍처인 decoder-only와 cross-attention에 대한 최초의 종합 시스템 분석을 제공합니다. 이는 텍스트, 이미지, 비디오 및 오디오 등 다양한 양식의 입력을 동시에 처리할 수 있는 대규모 멀티모달 모델에 대한 것입니다. 저자는 여섯 개의 대표적인 오픈 소스 모델의 멀티 스테이지 추론 파이프라인과 리소스 사용 패턴을 소개하며, 이를 통해 생산 배포에 대한 고유한 시스템 설계 함의를 제공합니다.

- **Technical Details**: LMM 모델들은 독특한 처리 접근 방식이 요구되는 다양한 입력 양식을 처리하며, 각기 다른 파이프라인 단계에서 자원 및 성능 패턴을 보여줍니다. 특히, 이미지 전처리, 이미지 인코딩, 언어 모델 백엔드의 성능 특성은 독립적으로 최적화되어야 함을 강조합니다. 이를 통해 저자는 이미지 인코딩이 주요 병목 현상으로 작용하며, 동시에 텍스트와 이미지 요청의 혼합이 성능 장애를 유발한다고 밝혔습니다.

- **Performance Highlights**: 논문은 LMM 추론의 다양한 단계가 높은 이질성 성능 특성을 보이며, 각 단계에 대한 독립적인 리소스 할당 및 적응형 스케일링을 가능하게 하는 분리된 서빙 아키텍처를 제안합니다. 특히, 이미지 인코딩의 병렬화가 지연(latency)을 줄이는데 필요하다는 점과 함께, 모드 간 요청으로 인한 성능 간섭을 완화하기 위한 모드 인지 스케줄링의 중요성을 강조합니다. 이를 통해 제시된 새로운 아키텍처는 LMM 서빙 효율성을 크게 향상시킬 것으로 기대됩니다.



### Attention Sinks and Outlier Features: A 'Catch, Tag, and Release' Mechanism for Embeddings (https://arxiv.org/abs/2502.00919)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 주목할 만한 두 가지 특성인 attention sinks와 outlier features의 중요성을 분석합니다. 연구에 따르면 attention sinks는 토큰 시퀀스를 캡처하고 태그를 붙여 이후 다시 스트림으로 풀어주는 'catch, tag, release' 메커니즘을 활용합니다. 이러한 발견은 모델 성능 향상 및 압축에 기여할 수 있습니다.

- **Technical Details**: 저자들은 두 개의 주요 현상인 attention sinks와 outlier features를 조사하였으며, 이들이 모델 매개변수에서 어떻게 나타나는지 설명합니다. attention weight matrices의 저랭크(low-rank) 구조가 이러한 현상을 생성하는 데 기여함을 입증했습니다. 이는 평균화와 같은 단순한 작업에서 이러한 메커니즘이 자연스럽게 발생하는 이유를 설명합니다.

- **Performance Highlights**: 본 연구는 OATS라는 새로운 압축 알고리즘을 통해 attention sinks와 outlier features를 효과적으로 보존할 수 있음을 보여줍니다. OATS는 전통적인 pruning 알고리즘과 비교하여 성능 저하 없이 저랭크 구조를 유지합니다. 결과적으로, 저자들은 few-shot learning과 같은 다운스트림 과제에서도 성능을 높일 수 있음을 입증했습니다.



### Embracing Dialectic Intersubjectivity: Coordination of Different Perspectives in Content Analysis with LLM Persona Simulation (https://arxiv.org/abs/2502.00903)
- **What's New**: 이번 연구에서는 content analysis 방법론을 consensus 중심에서 coordination 중심으로 발전시키는 것을 목표로 하였습니다. 이를 통해 다양한 coding outputs를 수용하고 여러 관점 간의 역동성을 탐구할 수 있는 기회를 제공합니다. 연구에서는 여섯 가지 GPT-4o 구성 모델을 평가하여 2020년 미국 대통령 선거 동안 Fox News와 MSNBC의 Biden과 Trump에 대한 감정을 분석했습니다.

- **Technical Details**: 이 연구는 진행 중인 LLM-Assisted Content Analysis (LACA) 방법론을 통해 이러한 분석을 수행했습니다. dialectic intersubjectivity 프레임워크를 적용하여 다양성을 강조하고, 정치적으로 일치하는 콘텐츠를 처리할 때 이념적 편향이 어떻게 나타나는지 평가했습니다. 각 모델의 출력은 두 후보에 대한 긍정성과 부정성 기준으로 평가되었으며, 상호코더 신뢰성(intercoder reliability) 또한 검토되었습니다.

- **Performance Highlights**: 연구 결과, LLM들은 동일한 이념적 집단의 코드 작성자들 간의 높은 일치를 보여주었으나, 이념적으로 일치하는 콘텐츠를 분석할 때 감정의 차이를 보였습니다. 이러한 차이는 dialectic intersubjectivity의 중요성을 강조하며, 연구자들이 LLM의 도구를 통해 다양한 해석을 포착할 수 있는 가능성을 보여줍니다. 이는 AI 기반 사회 과학 연구의 신뢰성을 높이고, 실제 세상의 복잡성을 반영하는 데 기여할 수 있습니다.



### MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies (https://arxiv.org/abs/2502.00894)
- **What's New**: 이번 논문에서는 MorphBPE라는 형태소 인식 버전의 Byte Pair Encoding (BPE)을 소개합니다. MorphBPE는 하위 단어(token) 토큰화에 언어적 구조를 통합하여 통계적 효율성을 유지하면서 형태소 경계를 고려합니다. 이 방법은 기존 LLM 훈련 파이프라인과 완벽하게 호환되며 최소한의 수정으로 통합이 가능하다는 점에서 주목할 만합니다.

- **Technical Details**: MorphBPE는 형태소별 편집 거리(Morphological Edit Distance)와 형태소 일관성 F1 점수(Morphological Consistency F1-Score)를 포함한 두 가지 새로운 평가 메트릭을 도입합니다. 이는 토큰화 품질을 평가하는데 있어 유용하며, 특히 형태론적으로 복잡한 언어에서의 효율성을 향상시키는 데 기여합니다. 연구는 영어, 러시아어, 헝가리어, 아랍어를 포함한 다양한 언어와 300M 및 1B 파라미터의 LLM을 대상으로 진행되었습니다.

- **Performance Highlights**: Experiments show that MorphBPE는 cross-entropy loss를 일관되게 줄이고 수렴 속도를 가속화하며 형태소 정렬 점수를 개선합니다. MorphBPE는 구조적으로 복잡한 언어에 대한 더 나은 하위 단어 토큰화를 제공하며, 모델의 성능 향상에 기여합니다. 이 접근법은 전통적인 형태소 분석과 NLP 간의 격차를 해소하는 데 도움을 줍니다.



### Paper Copilot: The Artificial Intelligence and Machine Learning Community Should Adopt a More Transparent and Regulated Peer Review Process (https://arxiv.org/abs/2502.00874)
- **What's New**: 최근 AI(Artificial Intelligence)와 ML(Machine Learning) 학회에 제출되는 논문 수가 급증하면서, 많은 학회들이 비공식 심사에서 공개 심사(open review)로의 전환을 모색하고 있습니다. 본 논문은 Paper Copilot이라는 웹사이트를 분석하여, AI/ML 분야에서 투명한 심사 프로세스에 대한 커뮤니티의 관심이 높아지고 있음을 강조합니다. 이 웹사이트는 177개국의 20만 명 이상의 초기 경력 연구자들의 데이터를 집계하고 분석하여, 리뷰 프로세스에 대한 참여를 촉진하고 있습니다.

- **Technical Details**: 전통적인 심사 관행은 공정성, 효율성 및 품질을 유지하는 데 압박을 받고 있으며, 이에 따라 많은 학회들이 공개 리뷰 플랫폼을 도입하고 있습니다. 연구자들은 평가의 투명성을 높이기 위해 각기 다른 공개 심사 모델을 도입하고 있으며, 완전 공개 리뷰, 부분 공개 리뷰, 비공식 리뷰 등 여러 형태가 존재합니다. 연구에 따르면, 완전 공개 리뷰는 모든 내용을 공개하지만, 개인적인 비판이나 편향이 개입할 수도 있습니다.

- **Performance Highlights**: Paper Copilot은 AI/ML 학회에서의 리뷰 통계를 제공하며, 지난 3~5년간의 리뷰 점수 분포, 리뷰 타임라인 및 저자/소속 분석을 포함합니다. 조사 결과, 연구자들은 투명성과 협력을 요구하고 있으며, 약 3,876건의 유효한 응답을 확보하여 이 정보를 체계적으로 처리하였습니다. 이를 통해, 더 투명하고 규제된 피어 리뷰 프로세스의 도입이 필요하다는 주장을 펼칩니다.



### FedHPD: Heterogeneous Federated Reinforcement Learning via Policy Distillation (https://arxiv.org/abs/2502.00870)
Comments:
          This preprint presents the full version of the Extended Abstract accepted by AAMAS 2025, including all the proofs and experiments

- **What's New**: 이 논문에서는 Federated Reinforcement Learning (FedRL)에 대한 새로운 접근 방식을 제안합니다. 기존 연구들은 동질적인 에이전트를 가정하였으나, 본 연구에서는 이질적인 에이전트가 블랙 박스(black-box) 환경에서 어떻게 효과적으로 협력할 수 있는지를 다룹니다. 이를 위해 Knowledge Distillation (KD) 기법을 활용하여 에이전트 간 지식 공유를 촉진하는 Federated Heterogeneous Policy Distillation (FedHPD)을 소개하였습니다.

- **Technical Details**: FedHPD는 각 에이전트가 고유한 정책 네트워크와 훈련 구성을 가질 때 발생하는 문제를 해결하기 위해 설계되었습니다. 본 알고리즘은 액션 확률 분포(action probability distributions)를 이용하여 지식을 공유하며, 이러한 방식은 에이전트의 내부 세부정보를 공유할 필요 없이 다양한 모델 간의 지식 정렬을 가능하게 합니다. 또한, 이 방법론에서는 주기적인 협업 훈련을 통해 지식을 추출하고 업데이트하며, 전통적인 정책 증류(policy distillation) 방법보다 더 나은 성능을 목표로 합니다.

- **Performance Highlights**: 실험 결과, FedHPD는 다양한 강화 학습 벤치마크 작업에서 유의미한 성과 향상을 보여주었습니다. 특히, FedHPD는 복잡한 환경에서도 효과적으로 작동하며, 기존의 공공 데이터셋 선택을 위한 복잡한 과정 없이도 좋은 성과를 달성할 수 있음을 입증하였습니다. 이로 인해 이질적인 에이전트 간의 훈련 성능이 향상되었으며, FedHPD는 이론적 증명뿐만 아니라 실제 응용에서도 유망한 결과를 제공합니다.



### Predicting potentially unfair clauses in Chilean terms of services with natural language processing (https://arxiv.org/abs/2502.00865)
Comments:
          37 pages, 2 figures, under review

- **What's New**: 이 연구는 소비자 계약에서 정보 비대칭에 대한 우려를 다루고 있으며, 복잡한 Terms of Service(Tos)의 확산에 의해 악화되고 있다. 특히, 스페인어 법조문을 다룬 첫 번째 다중 레이블 분류 데이터세트를 소개하여 칠레 내의 온라인 서비스를 위한 데이터와 방법론을 발전시켰다. 본 연구는 기계 학습의 가능성을 활용하여 소비자가 잠재적으로 불공정한 조항을 식별할 수 있도록 지원하는 데 중점을 두고 있다.

- **Technical Details**: 최신 연구에서 제안한 분류 시스템은 20개의 잠재적으로 불공정한 조항을 3개의 그룹으로 분류하는 것이며, 여기에는 6개의 불법 조항, 6개의 다크 조항, 그리고 8개의 회색 조항이 포함된다. 연구에서는 50개의 스페인어 Terms of Service 문서에서 6,523개의 주석이 달린 조항을 포함한 데이터셋을 구축하였고, 이를 통해 15,201개의 조항으로 이루어진 데이터셋이 생성되었다. 또한, Transformer 기반 모델을 사용하여 다양한 ML 접근법을 평가하고, 마이크로 및 매크로 F1 점수로 성능을 비교하였다.

- **Performance Highlights**: 실험 결과, 잠재적으로 불공정한 조항의 감지 작업에서는 매크로 F1 점수가 79%에서 89%까지 다양하게 나타났고, 마이크로 F1 점수는 96%까지 도달하였다. 분류 작업에서도 매크로 F1 점수가 60%에서 70%까지 이루어졌으며, 마이크로 F1 점수는 64%에서 80%의 범위를 보였다. 이 연구는 칠레 법 체계 내에서 공정 거래를 촉진하고 소비자 권익을 보호하는 데 이바지할 수 있는 중요한 시작점을 제공한다.



### Dual Alignment Maximin Optimization for Offline Model-based RL (https://arxiv.org/abs/2502.00850)
- **What's New**: 이번 논문에서는 오프라인 강화를 위한 새로운 접근법인 Dual Alignment Maximin Optimization(DAMO)을 제안합니다. 기존의 연구가 합성 데이터와 정책 간의 불일치를 해결하는 데 집중한 반면, DAMO는 정책 일관성을 유지하면서 기대 수익을 최적화하는 데 중점을 두고 있습니다. 이 방식은 정책 행동의 일관성을 보증하며, 모델 기반 접근법에서 발생하는 분배 이동 문제를 해결하는 데 기여합니다.

- **Technical Details**: DAMO는 내부 최소화 단계와 외부 최대화 단계로 구성된 두 단계 최적화 프로세스를 채택합니다. 내부 단계에서는 이중 보수 가치 추정을 통해 학습 정책을 조정하고, 외부 단계에서는 정책 개선이 내부 가치 추정과 일치하도록 조정됩니다. 이 과정은 합성 데이터와 오프라인 데이터를 효과적으로 통합하며, 모델과 환경 간의 일관성을 유지합니다.

- **Performance Highlights**: 실험 결과, DAMO는 다양한 벤치마크 작업에서 경쟁력 있는 성능을 달성하며, 모델과 정책 간의 정렬 문제를 성공적으로 완화하는 것으로 나타났습니다. 또한, DAMO는 OOD(Out-Of-Distribution) 문제를 효과적으로 해결하며, 오프라인 학습 환경 내에서 안전하고 신뢰할 수 있는 정책 학습을 촉진합니다.



### SecPE: Secure Prompt Ensembling for Private and Robust Large Language Models (https://arxiv.org/abs/2502.00847)
- **What's New**: 이 논문은 LLM (Large Language Models)의 개인 정보 보호와 적대적 강인성을 동시에 해결하기 위한 새로운 접근 방식을 제안합니다. 기존의 연구들은 주로 이 두 가지 문제를 별개로 다루었으나, 본 연구에서는 개인 정보 보호를 위한 암호화된 추론과 프롬프트 집합(Prompt Ensembling)을 통합하여 새로운 솔루션을 모색하고 있습니다. SecPE라는 보안 프롬프트 집합 방식을 도입하여, 높은 정확성을 유지하면서도 효율성을 증대시키는 방법을 제시하고 있습니다.

- **Technical Details**: SecPE는 효율적인 완전 동형 암호화(Fully Homomorphic Encryption, FHE) 기법을 활용하여 기존의 프롬프트 집합 알고리즘에서 발생하는 계산 비용 문제를 해결합니다. 특히, SecPE는 다수의 LLM 응답을 집계하기 위한 Argmax 연산의 효율성을 극대화하여 암호화된 상태에서도 빠른 처리를 가능하게 합니다. 이 방법은 GLUE 및 AdvGLUE와 같은 벤치마크를 통해 검증되었습니다.

- **Performance Highlights**: 실험 결과 SecPE는 높은 정확성과 강인성을 유지하며, 기존의 개인 정보 보호 방법과 비교해 단지 2.5%의 효율성 오버헤드를 보여줍니다. 또한, SecPE는 암호화된 Argmax 연산의 효율성 면에서도 기존의 기술들에 비해 35.4배 더 빠른 성능을 보였으며, 이는 다른 연구 분야에서도 큰 관심을 끌 수 있는 성과입니다.



### Activation Approximations Can Incur Safety Vulnerabilities Even in Aligned LLMs: Comprehensive Analysis and Defens (https://arxiv.org/abs/2502.00840)
Comments:
          19 pages

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 활성화 근사화(activation approximation)의 안전성 평가를 처음으로 체계적으로 수행하면서 LLM의 안전성 공백을 메우고자 합니다. 최근 LLM의 활용이 증가함에 따라, 자원 제한 환경에서의 배치 도전 과제가 더 두드러지고 있습니다. 이를 위해 활성화 근사화는 추론 효율성을 증대시키는 중요한 방법으로 여겨집니다. 그러나 이러한 활성화 근사화의 안전성에 대한 증거는 아직 부족한 상태입니다.

- **Technical Details**: 논문에서는 LLM의 추론 효율성을 높이기 위한 다양한 접근 방식을 검토하고 있으며, 활성화 근사화는 그 중 하나로 주목받고 있습니다. 세 가지 주요 카테고리로 (i) Activation Polynomialization, (ii) Activation Sparsification, (iii) Activation Quantization이 있다는 것을 설명하며, 이를 통해 복잡한 비선형 활성화 함수를 대체하거나 단순화하여 계산 속도를 높일 수 있습니다.

- **Performance Highlights**: 연구 결과, 권장되는 각 방법이 제공하는 추론 효율성 개선을 보여주며, 최대 24.6배의 속도 향상을 기록한 사례들이 있습니다. 이러한 성과는 LLM의 비용 문제를 완화하고, 민간 추론(private inference) 환경에서의 실용성을 높이는 데 기여할 것으로 기대됩니다. 그러나 예상되는 안전성의 저하가 널리 퍼진 LLM에서 확인되었기 때문에, 더욱 세심한 안전성 검증이 필요합니다.



### Explainability in Practice: A Survey of Explainable NLP Across Various Domains (https://arxiv.org/abs/2502.00837)
- **What's New**: 이 논문은 설명 가능한 자연어 처리(Explainable NLP, XNLP)의 중요성과 실제 적용 사례를 탐구합니다. 특히, 의료와 금융과 같은 도메인에서 XNLP의 활용 필요성을 강조하며, 기존의 문헌에서 부족한 부분을 보완하기 위해 도메인 별 접근 방식을 제안합니다. 또한, XNLP의 실용적인 배치와 그로 인해 사용자 신뢰가 어떻게 증진될 수 있는지를 논의합니다.

- **Technical Details**: 자연어 처리(Natural Language Processing, NLP)와 대형 언어 모델(Large Language Models, LLMs)은 기계가 인간 언어를 더 잘 이해하도록 도와주며 다양한 분야에 걸쳐 응용되고 있습니다. 아울러, 설명 가능한 인공지능(Explainable AI, XAI) 기법들이 모델의 의사 결정을 명확히 하기 위해 사용되고 있으며, 주목할 만한 기법으로는 LIME(Local Interpretable Model-agnostic Explanations)와 SHAP(SHapley Additive exPlanations)가 있습니다. 하지만, XNLP의 실제 적용과 평가 방법에 대한 논의는 여전히 부족하며, 이는 도메인 별 특성을 고려해야 함을 시사합니다.

- **Performance Highlights**: XNLP는 다양한 작업에서 높은 예측 성능을 보이는 기계 학습 모델의 '블랙 박스' 문제를 해결하고자 하고 있습니다. 특히, 진료 지원 시스템에서의 활용은 환자 관리에 긍정적인 영향을 미치며, 금융 부문에서도 사기 탐지 및 위험 평가에서 중요한 역할을 할 수 있습니다. 그러나, 각 도메인에서 신뢰할 수 있는 해석 가능성을 확보하기 위한 방법ologies가 필요하며, 이러한 요구 사항은 결정되는 결과에 직접적인 영향을 미칠 것입니다.



### Decision-informed Neural Networks with Large Language Model Integration for Portfolio Optimization (https://arxiv.org/abs/2502.00828)
Comments:
          Submitted paper

- **What's New**: 이 논문은 포트폴리오 최적화에서 예측과 의사결정의 질 간의 간극을 해결하기 위해 Large Language Models (LLMs)와 결정 중심 학습을 통합한 새로운 접근법을 제시합니다. 이 연구는 단순히 예측 오류를 최소화하는 것이 최적의 포트폴리오 결정을 이끌지 못함을 이론적 및 경험적으로 증명하고, LLM의 표현 능력을 투자 결정에 활용하고자 합니다. 저자들은 자산 간 관계, 시간 의존성, 거시 변수들을 처리하는 주의 메커니즘을 소개하고, 이를 포트폴리오 최적화 레이어와 직접 통합합니다.

- **Technical Details**: 제안된 프레임워크는 LLM의 표현력을 결정 중심 학습과 통합하여 포트폴리오 관리에서 복잡한 시장 관계를 포착하고 의사결정을 최적화합니다. 저자들은 자산 간 관계, 시간 의존성 및 거시 경제 변수의 세 가지 중요한 측면을 고려하여 주의 메커니즘을 도입하며, 이는 LLM의 표현을 효율적으로 필터링하여 모델의 계산 효율성과 해석 가능성을 향상시킵니다. 또한, 새로운 하이브리드 손실 함수를 통해 통계적 정확성과 포트폴리오 성능 간의 간극을 해소하며, 모델의 예측이 투자 결정으로 직접 연결되도록 합니다.

- **Performance Highlights**: S&P 100 및 DOW 30 데이터셋에 대한 실험 결과, 제안된 모델이 최첨단 딥러닝 모델을 지속적으로 초월함을 보여줍니다. 기울기 기반 분석을 통해 이 모델은 의사결정에 가장 중요한 자산에 우선순위를 주어 예측 오류가 포트폴리오 성능에 미치는 영향을 완화합니다. 이 연구 결과는 예측을 보다 견고하고 상황 인지적 포트폴리오 관리로 향상시키기 위해 결정 목표를 예측에 통합하는 것이 중요한 가치가 있음을 시사합니다.



### Fisher-Guided Selective Forgetting: Mitigating The Primacy Bias in Deep Reinforcement Learning (https://arxiv.org/abs/2502.00802)
- **What's New**: 이 논문은 Deep Reinforcement Learning (DRL) 시스템의 초기 경험에 대한 과도한 적합 문제를 일으키는 primacy bias (PB)를 Fisher Information Matrix (FIM)라는 관점에서 종합적으로 조사합니다. 이 연구는 PB의 특정 패턴을 FIM의 변화를 통해 특성화하고, Fisher-Guided Selective Forgetting (FGSF)이라는 새로운 방법을 제안하여 네트워크의 학습 프로세스에서 초기 경험이 우세해지지 않도록 합니다.

- **Technical Details**: 논문에서는 FIM을 활용하여 PB의 다양한 특성을 분석하고 식별하며, 이를 바탕으로 PB가 학습 성능에 미치는 영향을 최소화하는 방법을 제시합니다. FGSF 방식은 파라미터 공간의 기하학적 구조를 이용하여 네트워크의 가중치를 선택적으로 수정하는 방식으로, 초기 경험의 영향력을 줄이는데 중점을 둡니다. 이를 통해 학습 과정에서의 메모리화(memorization) 및 재조직화(reorganization) 단계를 분석합니다.

- **Performance Highlights**: 실험 결과 FGSF는 DeepMind Control Suite (DMC)와 같은 복잡한 환경에서 기존의 기법들과 비교하여 일관되게 우수한 성능을 보였습니다. PB의 영향은 actor와 critic 네트워크에서 상이하게 나타나며 재플레이(replay) 비율이 이 효과를 악화시킬 수 있다는 점도 논의됩니다. 이러한 발견들은 DRL의 발전을 위한 비판적 이해와 함께 실질적인 완화 전략을 제시하고 있습니다.



### Environment-Driven Online LiDAR-Camera Extrinsic Calibration (https://arxiv.org/abs/2502.00801)
- **What's New**: 이 논문에서는 최초의 환경 기반 온라인 LiDAR-카메라 외부 보정 방법인 EdO-LCEC를 소개합니다. 이 방법은 인간의 지각 시스템에서 영감을 받아, 환경 조건을 해석하고 여러 가상의 카메라를 생성하여 공간적 및 질감 정보의 세밀한 캡쳐를 가능케합니다. EdO-LCEC는 매끄러운 성능을 위해 교차 모드 기능 일치 문제를 해결하기 위해 이중 경로 대응 일치(DPCM)를 사용합니다.

- **Technical Details**: EdO-LCEC는 센서의 작동 환경을 공간-시간적 흐름으로 처리하여, 여러 장면의 정보를 동적으로 결합하고 높은 정확도의 보정을 달성합니다. 일반화 가능한 장면 구별자는 대형 비전 모델을 사용하여 깊이 추정 및 이미지 분할을 수행하면서, 환경을 구성하는 다양한 특징을 포착합니다. 이 시스템은 각 장면에서 DPCM을 수행하여 구조적 및 질감 일관성을 기반으로 신뢰할 수 있는 3D-2D 대응을 생성합니다.

- **Performance Highlights**: 다양한 실제 데이터셋에 대한 실험을 통해 EdO-LCEC는 다른 최신 온라인, 무대상 접근 방식들과 비교하여 우수한 강건성과 정확성을 보여주었습니다. 이 시스템은 특수 조정된 환경이 아닌 다양한 도전적인 환경에서의 신뢰할 수 있는 보정을 제공합니다. 결론적으로, EdO-LCEC는 높은 보정 정확도와 인간과 같은 적응성을 달성하는 혁신적인 접근법입니다.



### Role of Mixup in Topological Persistence Based Knowledge Distillation for Wearable Sensor Data (https://arxiv.org/abs/2502.00779)
Comments:
          IEEE Sensors Journal (2024)

- **What's New**: 본 논문은 wearable sensor data 분석에 대해 topological data analysis (TDA)와 knowledge distillation (KD)을 융합하여 향상된 모델을 만드는 방법을 제안합니다. 기존의 TDA 활용 방식의 한계를 극복하기 위해 KD 기법을 통해 더 작고 효율적인 모델을 생성하게 됩니다. 또한, mixup 기술을 통해 모델의 성능을 높이기 위한 방안을 모색하며, 이는 학습 및 데이터 증강 과정에서도 중요한 역할을 합니다. 이 연구는 기존의 이미지 기반 연구에서 벗어나, wearable sensor data에 적용할 수 있는 새로운 시각을 제공합니다.

- **Technical Details**: 논문에서는 여러 교사 모델을 활용한 knowledge distillation (KD)와 mixup 기법의 상호작용을 분석합니다. KD를 통해 topological features와 time-series 데이터를 원활하게 통합하여 보다 견고한 모델 학습을 시도하며, mixup 기법을 통한 데이터 증강이 KD 과정에 미치는 영향을 연구합니다. 특히, temperature hyperparameter의 조절이 학습 과정에서의 분포의 매끄러움에 어떤 영향을 미치는지도 논의됩니다. 다양한 KD 접근 방식과 각각의 효과를 비교 분석하여 모달리티의 차이를 함께 고려합니다.

- **Performance Highlights**: 결과적으로, 최적화된 mixup 전략은 다양한 교사 모델을 통해 학습된 학생 모델이 topological persistence 기반 KD에서 성능을 향상시킨다는 것을 보여줍니다. 논문에서 제안한 방법론은 시간 기반의 여러 증강 기법과 비교했을 때, KD에 있어 mixup의 효과를 확인하는 데 기여합니다. 최종적으로, 본 연구는 wearable sensor data 분석을 위한 새로운 접근 방식을 제공하며, 향후 연구 방향에 대한 통찰을 제시합니다.



### Learning-Based TSP-Solvers Tend to Be Overly Greedy (https://arxiv.org/abs/2502.00767)
Comments:
          19 pages, 6 figures

- **What's New**: 이 논문은 TSP(여행하는 세일즈맨 문제) 해결을 위한 딥러닝 기반 알고리즘의 발전을 다룹니다. 특히, 무작위로 생성된 데이터셋에서 발생하는 편향을 재조명하며, '가장 가까운 이웃 밀도(nearest-neighbor density)'라는 새로운 통계적 측정을 도입합니다. 또한, 데이터 증강(data augmentation) 방법을 통해 알고리즘의 일반화 능력을 증대시키는 접근법을 제시합니다.

- **Technical Details**: 논문에서 개발한 '가장 가까운 이웃 밀도'는 TSP 해결 과정에서 학습 기반 솔버의 탐욕적(greedy) 행동을 검증하는 데 사용됩니다. 이 밀도를 통해 TSP 데이터셋의 점근적 기초를 확인하고, 다양한 분포 변화를 기반으로 증강된 인스턴스를 생성합니다. 증강된 데이터로 미세 조정을 수행함으로써 학습 기반 솔버의 성능을 크게 향상시킬 수 있음을 입증했습니다.

- **Performance Highlights**: 이 연구에서는 학습 기반 TSP 솔버가 기존의 무작위 분포 데이터에서 성능을 저하시키는 경향을 밝혔으며, 이러한 경향을 극복하기 위해 데이터 증강 기법을 사용하는 것이 효과적임을 보여주었습니다. 특히, 균일 분포에 기반한 데이터에서 훈련된 모델들이 다른 분포의 테스트 인스턴스에서 저조한 성능을 나타내는 문제를 해결하는 데 큰 기여를 할 것으로 전망됩니다.



### AgentBreeder: Mitigating the AI Safety Impact of Multi-Agent Scaffolds (https://arxiv.org/abs/2502.00757)
- **What's New**: 이번 논문에서는 복잡한 작업에서 성능을 개선하기 위해 대형 언어 모델(LLMs)을 다중 에이전트 시스템으로 구성하는 것의 안전성 영향을 탐구합니다. 특히, AGENTBREEDER라는 프레임워크를 통해 스캐폴드를 진화시키는 방안을 제시하며, 안전성과 작업 보상을 결합하는 BLUEAGENTBREEDER와 LLM을 해킹하려는 REDAGENTBREEDER 두 가지 접근법을 소개합니다.

- **Technical Details**: AGENTBREEDER는 다중 목표 진화 검색(multi-objective evolutionary search)을 통해 스캐폴드를 진화시킵니다. REDAGENTBREEDER는 기초 LLM의 제약을 해제하면서 높은 작업 성공률을 달성하는 데 초점을 맞추고, BLUEAGENTBREEDER는 작업 보상(task reward)과 안전성(safety)을 통합하는 것을 목표로 합니다.

- **Performance Highlights**: 저자들은 AGENTBREEDER의 다양한 인스턴스에서 발견된 시스템을 평가하며, 널리 인정된 추론(reasoning), 수학(mathematics), 및 안전성 벤치마크(safety benchmarks)를 사용했습니다. 이 연구는 다중 에이전트 스캐폴딩으로 인한 안전 위험을 강조하고 이를 완화하는 방법을 제시합니다.



### Universal Post-Processing Networks for Joint Optimization of Modules in Task-Oriented Dialogue Systems (https://arxiv.org/abs/2502.00747)
Comments:
          Accepted by AAAI 2025 Main Technical Track

- **What's New**: 본 연구에서 제안하는 UniPPN(Universal Post-Processing Networks)은 기존의 PPN(Post-Processing Networks) 방식의 한계를 극복하고, 모든 모듈의 출력을 변환하는 경량 언어 모델 기반의 네트워크입니다. 이 네트워크는 Reinforcement Learning(RL) 알고리즘을 활용하여 시스템의 전체적인 작업 완료 능력을 향상시키는 데 기여합니다. 기존 PPN이 부분 모듈에만 국한되어 있었던 점과는 달리, UniPPN은 모든 모듈의 출력을 동시에 처리하고 최적화하는 새로운 방법론을 제시합니다.

- **Technical Details**: UniPPN의 RL 알고리즘은 모듈 수준의 Markov Decision Process(MDP)를 사용하여 각각의 모듈에 대한 세밀한 가치 및 이점 추정을 가능하게 하여 공동 학습을 안정화합니다. Proximal Policy Optimization(PPO)을 기반으로 하여 최적화 알고리즘을 구축하고 있으며, KL(Kullback-Leibler) 발산을 기반으로 한 패널티를 통해 정책 네트워크의 안정성을 유지합니다. 또한, GPT-2을 뒷받침 모델로 채택하고 구성적 파리미터를 조정하여 대화 시스템의 출력을 더 간소화하였습니다.

- **Performance Highlights**: MultiWOZ 데이터셋을 통해 진행된 시뮬레이션 및 인간 평가 실험에서 UniPPN은 기존의 PPN보다 작업 완료 능력에서 월등한 성능을 보였습니다. 특히, 모듈 수준 MDP의 도입으로 인해 UniPPN의 학습 성능이 개선된 것으로 나타났으며, 후반부 학습 시에는 turn-level MDP보다 안정적인 성과를 기록하였습니다. 이러한 결과는 UniPPN이 대화 시스템의 효율성과 정확성을 크게 향상시킬 수 있는 잠재력을 지니고 있음을 보여줍니다.



### From Compliance to Exploitation: Jailbreak Prompt Attacks on Multimodal LLMs (https://arxiv.org/abs/2502.00735)
- **What's New**: 이번 연구에서는 다중 모드 LLM(멀티모달 대형 언어 모델)에 대해 새로운 음성 기반 jailbreak 공격인 Flanking Attack을 제안합니다. 이 공격은 다양한 유형의 입력을 동시에 처리할 수 있는 능력을 활용해 LLM의 방어 메커니즘을 피해가는 방법을 보여줍니다. 연구팀은 음성 기반 입력의 취약점을 탐구하며, 인간화된 상호작용 문맥을 통해 공격을 실행하는 전략을 개발했습니다.

- **Technical Details**: Flanking Attack은 기존의 방어 메커니즘을 우회하기 위해 허용되지 않은 프롬프트를 선량한 내러티브 중심의 프롬프트로 둘러싸는 방식으로 설계되었습니다. 이 공격은 Gemini API를 통한 실험을 기반으로 하며, 음성 입력을 MP3 형식으로 수용하는 기능이 있었습니다. 이를 통해, 음성 기반 공격의 복잡성을 활용하고, 다양한 시나리오에서 효과를 측정합니다.

- **Performance Highlights**: Flanking Attack의 성공률은 0.67에서 0.93까지 다양하며, 평균 공격 성공률은 0.81로 나타났습니다. 이러한 결과는 음성 기반 LLM이 다양한 프롬프트 조합에 어떻게 대처하는지 드러내며, 멀티모달 LLM의 고유한 취약점을 해결하기 위한 전문 방어 전략의 필요성을 강조합니다. 또한, 이 연구는 다중 모드 LLM의 방어 능력을 이해하고, 고급 음성 기반 공격에 대한 더욱 튼튼한 LLM 개발의 기초를 제공합니다.



### CycleGuardian: A Framework for Automatic RespiratorySound classification Based on Improved Deep clustering and Contrastive Learning (https://arxiv.org/abs/2502.00734)
- **What's New**: 이번 논문은 폐 질환 및 호흡기 질환의 조기 진단을 위한 경량 네트워크 CycleGuardian을 설계하였습니다. 기존의 딥 러닝 기반 자동 호흡 소리 분류 방법이 제한된 데이터셋 때문에 성능 향상에 어려움을 겪고 있는 상황에서, 이 프레임워크는 개선된 딥 클러스터링(deep clustering)과 대비 학습(contrastive learning)을 기반으로 합니다.

- **Technical Details**: CycleGuardian 네트워크는 혼합 스펙트로그램(hybrid spectrogram) 생성과 클러스터링 모듈의 통합을 통해 비정상 소리의 특징을 캡처하고, 그룹 혼합(group mixing)과 대비 학습 모듈을 통해 비정상 소리의 식별 능력을 향상시킵니다. 이 프레임워크는 다양한 목표를 최적화하여 전체 성능을 향상시킵니다.

- **Performance Highlights**: ICBHI2017 데이터셋을 기반으로 실험한 결과, CycleGuardian은 Sp: 82.06%, Se: 44.47%, Score: 63.26%의 성과를 기록하였습니다. 현재 모델에 비해 거의 7% 향상된 성능을 달성하며, 안드로이드 기기에서도 네트워크를 배포하여 포괄적인 지능형 호흡 소리 청진 시스템을 실현하였습니다.



### Learned Bayesian Cram\'er-Rao Bound for Unknown Measurement Models Using Score Neural Networks (https://arxiv.org/abs/2502.00724)
Comments:
          28 pages, 11 figures

- **What's New**: 이 논문에서는 완전하게 학습된 베이지안 크라메르-라오 경계(LBCRB)를 제안합니다. 기존의 베이지안 크라메르-라오 경계(BCRB)는 사전(prior) 및 측정(distribution) 분포에 대한 완전한 지식 없이는 계산할 수 없었습니다. 새로운 접근법으로는 Posterior Approach와 Measurement-Prior Approach가 소개됩니다.

- **Technical Details**: Posterior Approach는 LBCRB를 쉽게 얻는 방법을 제공합니다. 한편, Measurement-Prior Approach는 도메인 지식(domain knowledge)을 통합하여 샘플 복잡성(sample complexity)과 해석 가능성(interpretability)을 향상시킵니다. 이를 위해 물리학 기반의 스코어 신경망(Physics-encoded score neural network)을 도입하여 도메인 지식을 신경망에 쉽게 통합할 수 있도록 합니다.

- **Performance Highlights**: 우리는 제안된 두 가지 접근법의 학습 오류(learning errors)를 이론적으로 연구하고 수치적으로 검증합니다. 또한 여러 신호 처리(signal processing) 예제에서 두 가지 접근법을 시연했으며, 이는 미지의 혼합 및 가우시안 노이즈 공분산 행렬을 포함한 선형 측정 문제(linear measurement problem), 주파수 추정(frequency estimation), 양자화된 측정(quantized measurement) 등을 포함합니다. 이 외에도 실제 수중 주변 소음과 함께하는 비선형 주파수 추정 문제에서도 우리의 접근법을 테스트했습니다.



### Registration-Enhanced Segmentation Method for Prostate Cancer in Ultrasound Images (https://arxiv.org/abs/2502.00712)
- **What's New**: 이 연구는 MRI-TRUS 융합 기반의 자동 분할(segmentation) 방법을 제안합니다. 이 방법은 수동 주석을 요구하지 않고 TRUS 이미지에서 바로 전립선 종양을 식별할 수 있습니다. 기존의 단순한 데이터 결합 기법과는 달리, 제안된 방법은 등록-분할(registration-segmentation) 프레임워크를 통합하여 MRI와 TRUS 모달리티 간의 공간 정보를 효과적으로 활용합니다.

- **Technical Details**: 연구에서는 1,747명의 환자 데이터를 사용하여 방법의 유효성을 검증했습니다. 제안된 방법은 평균 Dice 계수가 0.212로, TRUS 전용 방법(0.117)과 단순한 MRI-TRUS 융합 방법(0.132)에 비해 상대적으로 81.2% 및 60.6% 개선된 성능을 보였습니다. 통계적으로 의미 있는 결과(p < 0.01)를 보여주었으며, 이는 종양 분할 정확도를 강화하는 데 기여합니다.

- **Performance Highlights**: 이 프레임워크는 전립선암 진단의 복잡성을 줄이고, 여러 모달리티를 사용하는 의료 영상 작업에 적용할 수 있는 유연한 아키텍처를 제공합니다. 전통적인 TRUS 유도 생검 절차에 비해, 제안된 방법은 시간과 노력을 크게 줄여 의료진의 부담을 경감할 것으로 기대됩니다.



### VIKSER: Visual Knowledge-Driven Self-Reinforcing Reasoning Framework (https://arxiv.org/abs/2502.00711)
Comments:
          17 pages,12 figures

- **What's New**: 이 논문에서는 VIKSER(Visual Knowledge-Driven Self-Reinforcing Reasoning Framework)를 제안하여 비주얼 정보에 대한 질문 해결을 위한 새로운 접근 방식을 소개합니다. 기존의 비주얼 추론(visual reasoning) 기법은 해석 가능성이 제한적이고, 질문 텍스트의 불완전성에 의해 제약을 받는 문제를 안고 있습니다. VIKSER는 대형 언어 모델(LLMs)에서 증류한 지식을 활용해 정밀한 비주얼 지식을 추출하고, 이를 통해 질문을 패러프레이즈(paraphrase)합니다.

- **Technical Details**: VIKSER의 핵심 구성 요소는 세분화된 비주얼 지식 추출(F-VKE) 모듈과 자기 강화 추론(S-RR) 모듈로 구성됩니다. S-RR 모듈은 Chain-of-Evidence(CoE)라는 새로운 프롬프트 기법을 통합하여 해석 가능성을 높이고, 과거의 실수를 통해 학습하는 자기 반성 메커니즘을 도입합니다. F-VKE 모듈은 입력 이미지의 주요 엔티티 간의 시각적 관계를 감지하고, 이를 기반으로 원인 관계를 분석하여 세분화된 비주얼 지식을 생성합니다.

- **Performance Highlights**: VIKSER는 다양한 공개 데이터셋에서 철저한 실험을 통해 기존 연구들을 초월하는 성과를 거두었으며, 모든 데이터셋에서 새로운 최신(state-of-the-art, SOTA) 결과를 달성하였습니다. 이러한 성과는 고도의 해석 가능성과 자기 강화 학습이 결합된 결과로, 비주얼 추론 작업에서의 효율성과 정확성을 크게 향상시킵니다. 이를 통해 VIKSER는 비주얼 지식 추출 및 추론 능력에서 뛰어난 성능을 입증하였습니다.



### PhiP-G: Physics-Guided Text-to-3D Compositional Scene Generation (https://arxiv.org/abs/2502.00708)
Comments:
          13 pages.8 figures

- **What's New**: 이번 논문에서는 PhiP-G라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 자연어 입력으로부터 고품질의 3D 장면을 생성하는 데 필수적인 구성 요소인 생성 모델과 LLM 기반 에이전트를 통합합니다. PhiP-G는 물리 법칙에 부합하는 고품질 3D 장면 생성을 위한 예측 및 계획 기능을 활용하여 레이아웃 단계를 향상시킵니다.

- **Technical Details**: PhiP-G는 복합적인 장면 설명을 분석하기 위해 LLM 기반 에이전트를 사용하고, 장면 그래프를 생성하여 수동 레이아웃을 피합니다. 2D 이미지 생성 에이전트인 DALL·E 3와 3D Gaussian splatting(3DGS) 모델을 결합하여 고품질 자산을 유연하게 생성합니다. Blender를 레이아웃 디자인의 기본 플랫폼으로 활용하여 물리 풀 및 관계 일치 에이전트를 도입하여 전체적인 레이아웃 안내를 실현합니다.

- **Performance Highlights**: PhiP-G는 복합 장면 생성에서 상태 최상위(State-of-the-Art) 성능을 달성하며, CLIP 점수에서 SOTA를 기록했습니다. 또한, T3Bench 메트릭에서 최고의 성능을 달성하면서 생성 효율성을 24배 향상시켰습니다. 이로 인해 복잡한 자연어 입력에 대한 의미적 일관성을 유지하며 물리 법칙을 준수하는 신뢰할 수 있는 3D 장면 생성을 가능하게 합니다.



### TMI-CLNet: Triple-Modal Interaction Network for Chronic Liver Disease Prognosis From Imaging, Clinical, and Radiomic Data Fusion (https://arxiv.org/abs/2502.00695)
Comments:
          6 pages, 3 figures, accepted by IEEE ISBI 2025

- **What's New**: 이번 연구에서는 만성 간 질환의 예후 평가를 위해 Triple-Modal Interaction Chronic Liver Network (TMI-CLNet)를 제안합니다. 이 새로운 접근법은 CT 이미지, 방사형 특징(radiomic features), 임상 정보를 통합하여 보다 종합적인 예후 정보를 제공합니다. 또한, Intra-Modality Aggregation 모듈과 Triple-Modal Cross-Attention Fusion 모듈을 통해 서로 다른 데이터 모달리티 간의 관계를 효과적으로 포착할 수 있도록 설계하였습니다.

- **Technical Details**: TMI-CLNet의 아키텍처는 세 가지 주요 구성 요소로 이루어져 있습니다: 특징 추출 모듈, 다중 모달 상호 작용 모듈, 그리고 분류 헤드 모듈입니다. 특징 추출 모듈에서는 3D ResNet-50을 사용하여 시각적 특징을 추출하며, 방사형 데이터는 다층 퍼셉트론(MLP)으로 처리해 보다 추상적인 특징 표현을 얻습니다. 각 모달리티의 특징을 통합하기 위해 Intra-Modal Aggregation (IMA) 모듈이 사용되며, 이 모듈은 다중 헤드 자기 주의 기법을 통해 특징 정보를 통합합니다.

- **Performance Highlights**: 광범위한 실험을 통해 TMI-CLNet이 기존의 최첨단 단일 모달 모델 및 기타 다중 모달 기술들보다 유의미하게 성능이 우수함을 보여주었습니다. 본 연구의 결과는 만성 간 질환 치료의 임상적 결정 과정에 중대한 기여를 할 것으로 기대됩니다. 또한 연구의 코드는 공개되어 있어 추후 연구자들이 쉽게 접근할 수 있도록 합니다.



### Leveraging Large Language Models to Predict Antibody Biological Activity Against Influenza A Hemagglutinin (https://arxiv.org/abs/2502.00694)
- **What's New**: 이번 연구는 AI 모델을 이용하여 항체의 결합 및 수용체 차단 활성을 예측하는 방법을 제시합니다. 특히, 기존의 항체에 대한 성공 확률이 높은 항체를 선별하는 데 있어 실험실 테스트의 비용과 시간을 줄일 수 있는 가능성을 보여줍니다. 또한, MAMMAL 프레임워크를 활용하여 항체-항원 상호작용을 예측하는 모델이 개발되었음을 소개합니다.

- **Technical Details**: 본 연구의 AI 모델은 단백질 서열(sequence) 정보만을 사용하여 항체-항원 간 상호작용을 예측하는 기능을 가지고 있습니다. 모델의 성능을 평가하기 위해 다양한 데이터 분할 조건(data split conditions)에서 테스트를 수행하여 현실 세계의 시나리오를 모방했습니다. 기존 항체에 대해 0.91 이상의 AUROC을 달성하며, 새로운 항체의 경우 0.73의 AUROC을 기록하였습니다.

- **Performance Highlights**: 모델의 성능은 기존의 HA에 대한 예측에서 높은 정확도를 보여주었지만, 기존 항체와의 유사성이 엄격히 제한될 경우 AUROC이 0.63-0.66으로 감소하는 경향이 있음을 나타냅니다. 이는 새로운 항체 개발을 위한 다양한 데이터 세트의 중요성을 강조하며 AI 기반 모델이 항체 설계를 변화시킬 잠재력이 있음을 보여줍니다.



### Dissecting Submission Limit in Desk-Rejections: A Mathematical Analysis of Fairness in AI Conference Policies (https://arxiv.org/abs/2502.00690)
- **What's New**: AI 연구의 급증으로 인해 콘퍼런스는 논문 품질을 유지하기 위해 제출 한도를 설정하고 있습니다. 이러한 기존의 데스크 리젝션 시스템은 특히 초기 경력 연구자들에게 불평등을 초래할 위험이 크다는 점을 제시합니다. 본 연구는 공정성을 고려한 새로운 데스크 리젝션 메커니즘을 제안하며, 이를 통해 기존 시스템보다 더 큰 공정성을 보장할 수 있음을 입증합니다.

- **Technical Details**: 제안된 시스템은 초과 제출 논문들이 다수인 저자들의 제출을 우선적으로 반려해 공정성을 도모하는 최적화 기반 알고리즘을 사용합니다. 우리는 개인 공정성과 집단 공정성이라는 두 가지 공정성 지표를 정의하였으며, 개인 공정성을 최적화하는 것은 NP-hard하다는 것을 증명했습니다. 한편, 집단 공정성은 선형 프로그래밍을 통해 효과적으로 최적화될 수 있습니다.

- **Performance Highlights**: 사례 연구를 통해 제안된 시스템이 CVPR 2025와 같은 기존 방법들보다 더 나은 공정성을 제공한다는 것을 보여주었습니다. 이는 AI 콘퍼런스에서의 과도한 제출 관리에 대한 보다 사회적으로 정당한 접근을 제안하며, 초기 경력 연구자들에게 더 많은 보호를 제공합니다. 전반적으로 이 연구는 AI 연구 커뮤니티의 포용성 향상 및 사회적 정의에 기여할 수 있는 방향성을 제시합니다.



### High-Order Matching for One-Step Shortcut Diffusion Models (https://arxiv.org/abs/2502.00688)
- **What's New**: 이 논문에서는 고차 매칭(HOMO, High-Order Matching for One-Step Shortcut Diffusion)을 도입하여 기존의 Shortcut 모델의 한계를 극복하고자 합니다. HOMO는 고차 감독(high-order supervision)을 통해 데이터 전송(distribution transportation)을 혁신적으로 개선합니다. 이 프레임워크는 가속(acceleration), 급상승(jerk) 등 다양한 요소를 포함하여 지오메트리적 정확성(geometric precision)과 안정성(stability)을 향상시킵니다.

- **Technical Details**: HOMO는 고차 감독을 통합하여 기존의 1차 동역학(first-order dynamics)에 의존하는 Shortcut 모델의 한계를 넘습니다. 이 모델은 예측 경로의 정확성과 안정성을 보장하기 위해 적절한 수학적 기초를 확립하고, 복잡한 데이터 분포를 정확히 모델링할 수 있는 역량을 갖추고 있습니다. 특히, 고차 동역학을 포함시키는 것이 데이터의 흐름을 더 매끄럽고 정확하게 만들어 줍니다.

- **Performance Highlights**: HOMO는 상당히 복잡한 환경, 특히 고곡률(high-curvature) 지역에서 Shortcut 모델을 뛰어넘는 성능을 보여줍니다. 실험 결과 HOMO는 더 매끄러운 경로(smoother trajectories)와 유리한 분포 정렬(better distributional alignment)을 달성하여 1단계 생성 모델(1-step generative models)에 새로운 기준을 제시합니다. 이러한 성과로 인해 HOMO는 생성 모델링에서의 신뢰성과 강력함을 증명하였습니다.



### Compositional Concept-Based Neuron-Level Interpretability for Deep Reinforcement Learning (https://arxiv.org/abs/2502.00684)
Comments:
          8 pages, 3 figures, IJCAI 2025

- **What's New**: 이 논문은 심층 강화 학습(Deep Reinforcement Learning, DRL) 모델에 대한 새로운 개념 기반 해석 방법을 제안합니다. 기존 DRL 해석 방법들은 신경망을 블랙박스처럼 다루어 개인 뉴런의 기여를 명확히 설명하지 못했습니다. 제안된 방법은 상태 공간에 대한 원자 개념을 이진 함수로 정의하고, 논리 연산을 통해 복합 개념을 구성하여 뉴런 수준에서 세분화된 설명을 제공합니다.

- **Technical Details**: 논문에서는 강화 학습에서 개념을 정의하고, 신경망의 활성화와 이들 개념의 관계를 분석하는 방법론을 제시합니다. 특히 원자 개념을 이진 함수로 정의하고, 이들을 통해 개념 벡터를 구성하는 과정이 설명됩니다. 이러한 접근 방식은 뉴런 수준의 해석성을 향상시키고, 정책/가치 네트워크의 복합 개념을 통해 개별 뉴런의 기여를 해석할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 의미 있는 개념을 효과적으로 식별할 수 있으며, 이는 인간의 이해와 잘 일치하면서 네트워크의 결정 로직을 충실히 반영함을 나타냅니다. 연속 제어 작업과 이산 의사 결정 환경에서의 결과를 통해, 개인 뉴런에서의 해석 가능한 결정 메커니즘을 성공적으로 보여주었습니다. 또한, 개념-뉴런 매핑을 통해 네트워크의 의사 결정 논리를 실제로 포착하는 것을 입증함으로써 해석의 유효성을 검증했습니다.



### Guidance Source Matters: How Guidance from AI, Expert, or a Group of Analysts Impacts Visual Data Preparation and Analysis (https://arxiv.org/abs/2502.00682)
Comments:
          21 pages, 10 figures, 6 figures, to appear in proceedings of ACM IUI 2025

- **What's New**: 본 연구는 AI 기반 도구의 발전에 따라 Data Analysis (데이터 분석) 도중 사용자에게 제공되는 Guidance (유도)의 출처에 대한 중요성을 조사합니다. 기존 연구는 주로 유도가 제공되는 방법이나 시점에 집중했으나, 우리는 '누구로부터' (from whom) 제공되는지가 사용자의 인식 및 유용성에 미치는 영향을 살펴보았습니다. AI, 인간 전문가, 그리고 그룹으로부터 제공되는 유도에 대해 실험을 진행하였으며, 이와 함께 출처가 명시되지 않은 유도도 고려했습니다.

- **Technical Details**: 연구는 5개 조건의 실험 디자인으로 구성되었습니다. 각 조건은 AI, Expert (전문가), Group (그룹), Unattributed (출처가 명시되지 않은) 및 Control (제공된 유도가 없는 조건)으로 나뉘어 있습니다. 사용자는 새로운 데이터 세트에서 비즈니스 보고서 작성을 위한 관련 특성을 선택하는 작업에 과제에 매겨졌습니다. 모든 조건에서 제공된 유도의 질을 통제하고, 사용자의 특성 선택과 관련된 다수의 메트릭을 통해 가설을 검증했습니다.

- **Performance Highlights**: 연구 결과, 사용자는 유도의 출처에 따라 유도 효과가 달라지며, 특히 AI로부터 받은 유도가 후속 작업에서 더 큰 이익을 주었으나 더 많은 후회도 초래했습니다. 또한, 유도가 제공되는 분석 단계에 따라 사용자는 활용도를 다르게 나타냈습니다. 이러한 결과는 다양한 출처로부터의 유도가 사용자의 행동에 영향을 미친다는 것을 강조하며, 효과적인 유도 시스템 설계를 위해 추가적인 연구 필요성을 제기합니다.



### A Survey of Quantized Graph Representation Learning: Connecting Graph Structures with Large Language Models (https://arxiv.org/abs/2502.00681)
- **What's New**: 최근의 그래프 표현 학습이 빠른 발전을 거듭하고 있으며, 이러한 발전의 중심에는 연속 임베딩(continuous embedding) 접근 방식이 자리잡고 있습니다. 하지만 이러한 방법들은 매개변수 효율성, 해석 가능성 및 견고성에서 문제에 직면하고 있습니다. 이에 따라 최근에는 양자화 그래프 표현(Quantized Graph Representation, QGR) 학습이 증가하는 관심을 받고 있으며, 이는 기존의 연속 임베딩 대신 이산 코드(discrete codes)로 그래프 구조를 표현합니다. QGR은 자연어에 유사한 표현 형식을 가지고 있어 대규모 언어 모델(LLMs)과의 통합에 효과적인 잠재력을 가지고 있습니다.

- **Technical Details**: QGR는 연속 임베딩을 사용하는 대신 노드, 서브그래프 또는 전체 그래프 구조에 대해 이산 코드를 학습하는 방법입니다. 그래프 텍스트 시나리오에서 연속 임베딩은 의미 정보의 손실을 초래할 수 있는 반면, QGR는 자연어의 이산 특성과 일관성을 가지므로 LLM에 매끄럽게 통합될 수 있는 이점이 있습니다. 또한 QGR의 핵심 아이디어는 고차원 공간을 여러 저차원 서브공간으로 분할하고 각 서브공간 내에서 독립적으로 양자화하는 것입니다. 이를 통해 QGR는 해석 가능성과 견고성을 제공하며, 자가 감독 그래프 학습을 통해 다양한 그래프 구조를 학습할 수 있습니다.

- **Performance Highlights**: QGR의 도입으로 인해 기억 사용량이 상당히 줄어들며, 이는 대용량 그래프에서의 매개변수 요구 사항을 대폭 절감하는 결과를 가져옵니다. 예를 들어, 특정 조건에서는 매개변수 요구사항이 113배 감소할 수 있습니다. 이처럼 QGR는 그래프 학습의 비효율성을 극복하고, 해석 가능성을 높이며, 실제 애플리케이션에 쉽게 적응할 수 있는 능력을 제공합니다. 궁극적으로 이러한 발전은 그래프 커뮤니티의 발전과 미래 연구를 자극할 것으로 기대됩니다.



### How Contaminated Is Your Benchmark? Quantifying Dataset Leakage in Large Language Models with Kernel Divergenc (https://arxiv.org/abs/2502.00678)
- **What's New**: 이 논문에서는 데이터셋 오염(dataset contamination)의 문제를 해결하기 위한 새로운 방법, 즉 Kernel Divergence Score (KDS)를 제안합니다. 데이터셋 오염은 평가 데이터셋이 모델의 사전 학습(pre-training) 데이터와 겹치는 현상으로, 이것이 성능 지표를 부풀리고 모델 평가의 신뢰성을 저하시키는 문제를 야기합니다. KDS는 모델의 벤치마크 데이터셋에서 파인튜닝(fine-tuning) 전과 후의 커널 유사성 행렬(kernel similarity matrix) 간의 차이를 계산함으로써 오염을 정량화합니다.

- **Technical Details**: KDS는 샘플 임베딩(sample embeddings)의 커널 유사성 행렬의 변화를 분석하여, 사전 학습된 데이터와 파인튜닝된 데이터 간의 관계를 평가합니다. 이 방법은 보지 못한(unseen) 샘플에 대한 파인튜닝의 효과가 더 크다는 통찰을 바탕으로 하여 진행됩니다. KDS는 여러 데이터셋에 대한 실험을 통해 오염 수준과 거의 완벽한 상관관계를 보이며 기존 알고리즘보다 더 나은 성능을 보여줍니다.

- **Performance Highlights**: KDS는 다양한 설정과 데이터셋에 걸쳐 안정적인 점수를 제공하여 연구자들이 데이터셋의 오염 정도에 따라 벤치마크를 신뢰성 있게 구분할 수 있도록 합니다. 이 연구는 KDS가 더 낮은 오염 수준의 데이터셋을 식별하는 데 효과적임을 입증하며, 이로 인해 모델의 일반화 능력을 더욱 정확하게 평가할 수 있는 기회를 제공합니다. KDS는 또한 여러 디자인 요소에 대해 강력한 성능을 보여, 커널 함수, 커널 대역폭(kernel bandwidth), 임베딩 추출 위치 등 다양한 설계 선택에 대한 감도를 확인하는 실험을 수행했습니다.



### Biogeochemistry-Informed Neural Network (BINN) for Improving Accuracy of Model Prediction and Scientific Understanding of Soil Organic Carbon (https://arxiv.org/abs/2502.00672)
Comments:
          41 pages, 8 figures

- **What's New**: 본 논문에서는 Biogeochemistry-Informed Neural Network (BINN)을 개발하여, 대규모 데이터에서 토양 유기탄소(SOC) 저장을 제어하는 메커니즘을 조사합니다. BINN은 Community Land Model version 5 (CLM5)라는 벡터화된 프로세스 기반 모델을 신경망 구조에 통합하여, 기존의 방법론들보다 높은 정확도로 메커니즘 지식을 추출할 수 있습니다.

- **Technical Details**: BINN은 많은 양의 SOC 프로파일을 사용하여 6대의 주요 토양 탄소 순환 프로세스를 예측하며, 연구팀은 25,925개의 SOC 프로파일을 분석했습니다. 이 연구는 Bayesian inference 기반의 PRODA 접근 방식과 비교하여 높은 상관관계(평균 0.81)를 보였으며, 이를 통해 두 가지 접근 방법 간의 공간 패턴이 잘 일치함을 확인했습니다.

- **Performance Highlights**: BINN은 PRODA보다 계산 효율성을 50배 이상 향상시켰으며, 이는 AI 및 프로세스 기반 모델링을 통합한 결과입니다. 이러한 성과는 지구 시스템 모델의 해석 가능성과 정확성을 높이며, 새로운 과학적 발견을 촉진하는 도구로서의 BINN의 전환적 역할을 강조합니다.



### Avoiding $\mathbf{exp(R_{max})}$ scaling in RLHF through Preference-based Exploration (https://arxiv.org/abs/2502.00666)
- **What's New**: 이 논문은 온라인 환경에서의 Reinforcement Learning from Human Feedback (RLHF) 기법에서 샘플 효율성을 개선하는 데 중점을 둡니다. 기존 알고리즘들이 보상 함수의 규모에 따라 지수적으로 샘플 복잡도가 증가하는 문제를 다루고, 이를 해결하기 위해 Self-Exploring Preference-Incentive Online Preference Optimization (SE-POPO)라는 새로운 알고리즘을 제안합니다. 이 알고리즘은 보상 규모에 대해 다항적으로 샘플 복잡도가 확장되므로 큰 보상 범위에서도 효과적으로 작동할 수 있습니다.

- **Technical Details**: SE-POPO는 기존의 보상 기반 탐색 방법과는 달리, 선호 기반 탐색 기법을 적용하여 능동적인 탐색을 수행합니다. 이 알고리즘은 DPO를 바탕으로 하는 Preference-Incentive Online Preference Optimization (POPO) 서브루틴을 구성하며, 구현이 용이하고 이전 알고리즘과 동등한 샘플 복잡도를 보장합니다. 또한, 샘플 복잡도가 보상 범위의 증가와 함께 폭발하지 않도록 하는 자가 샘플러 업데이트 기술을 개발하였습니다.

- **Performance Highlights**: 다양한 교육 및 테스트 환경과 주요 공개 벤치마크에서 SE-POPO 알고리즘의 성능을 평가한 결과, 모든 벤치마크에서 탐색적 및 비탐색적 기준선보다 현저히 우수한 결과를 나타냈습니다. 이로 인해 SE-POPO는 기존 알고리즘들보다 높은 샘플 효율성을 보여 주며, RLHF 알고리즘 설계의 중요한 진전을 의미합니다.



### Enhanced Convolutional Neural Networks for Improved Image Classification (https://arxiv.org/abs/2502.00663)
- **What's New**: 이 논문에서는 CIFAR-10 이미지 분류를 위한 향상된 CNN 아키텍처를 제안합니다. 기존 CNN 모델의 과적합(overfitting) 문제와 기능 표현 문제를 해결하기 위해 더 많은 convolutional 블록, 배치 정규화(batch normalization), 그리고 드롭아웃(Need dropout regularization) 기법을 통합하였습니다. 제안하는 모델은 84.95%의 테스트 정확도를 달성하여 기존 CNN 아키텍처를 초월하는 성능을 보였습니다.

- **Technical Details**: 데이터 전처리는 모델의 일반화(generalization)와 안정성을 높이는 데 중요한 역할을 합니다. CIFAR-10 데이터셋에 대해, 픽셀 값의 정규화(normalization)를 통해 입력 데이터를 [-1, 1] 범위로 조정하고, 데이터 증대(data augmentation)를 통해 학습 데이터의 다양성을 인위적으로 증가시켰습니다. 또한, 64 크기의 미니 배치(mini-batch)를 이용해 효율적인 그래디언트 계산을 가능하게 하였습니다.

- **Performance Highlights**: 실험 결과, 향상된 CNN 아키텍처는 기존의 CNN 기반 모델에 비해 월등한 성능을 보여주었습니다. 특히, 배치 정규화와 드롭아웃을 통합함으로써 훈련 과정의 안정성을 극대화하고, 제안하는 모델이 보다 복잡한 이미지 분류 작업에 효과적이라는 점을 강조했습니다. 모델의 구조적인 개선이 작은 규모의 이미지 분류 문제를 해결하는 데 어떠한 잠재력을 가지고 있는지를 보여주는 결과입니다.



### LLM Safety Alignment is Divergence Estimation in Disguis (https://arxiv.org/abs/2502.00657)
- **What's New**: 본 논문은 인기 있는 대형 언어 모델(Large Language Model, LLM) 정렬 방법들이 본질적으로 정렬된(선호됨) 분포와 비정렬된(덜 선호됨) 분포 간의 분산 추정기(divergence estimator)로 작동한다는 이론적 프레임워크를 제안합니다. 이는 정렬 후 모델의 숨겨진 표현에서 안전하고 유해한 프롬프트(prompt) 간의 분리가 발생하는 현상을 설명합니다. 또한, 저자들은 KL 분산(KL divergence)을 기반으로 하는 새로운 정렬 방법, KLDO를 소개하고, 안전 정렬을 강화하기 위해 선호 데이터(preference dataset) 대신 준수-거부 데이터(compliance-refusal dataset)를 사용하는 것을 지지합니다.

- **Technical Details**: LLM의 정렬 연구에서는 LLM이 악의적인 사용자 입력에 반응하지 않고 안전한 응답만 생성하도록 보장하는 안전 정렬(safety alignment)에 중점을 둡니다. 기존의 정렬 방법들은 특정 분산 메트릭(divergence metric)과 일치하며, KTO는 총 변이(total variation) 거리를, BCO는 Jensen-Shannon 분산을, DPO는 비모수(divergences) 분산을 추정합니다. 저자들은 이러한 발견에 기초하여 KL 분산 추정에 기반한 새로운 정렬 방법인 KLDO를 제안하며, 모든 정렬 방법이 비정렬 모델보다 숨겨진 표현에서 유의미한 분리(separation)를 유도한다고 주장합니다.

- **Performance Highlights**: 저자들은 준수-거부(compliance-refusal) 데이터를 사용할 때 정렬 방법의 성능이 향상된다는 것을 실험적으로 검증하였습니다. 실험 결과, 정렬 방법들이 비정렬 기본 모델과 비교하여 significant visual separation을 유도하는 것을 확인하였으며, 안전성과 분리 지표가 공격 성공률(Attack Success Rate)과 밀접하게 관련되어 있음을 입증했습니다. 이 연구는 안전성을 평가하는 새로운 차원을 제시하며, LLM의 안전성과 강건성을 강화할 수 있는 방향을 제시합니다.



### TrojanTime: Backdoor Attacks on Time Series Classification (https://arxiv.org/abs/2502.00646)
Comments:
          13 pages, 3 figures, 3 tables

- **What's New**: 이번 논문에서는 Time Series Classification(TSC) 문제에 대한 새로운 접근법인 TrojanTime을 제안합니다. TSC는 backdoor 공격에 취약하며, 기존의 방법들은 주로 훈련 단계에서 데이터 오염(data poisoning)에 초점을 맞추고 있습니다. TrojanTime은 두 단계로 구성된 훈련 알고리즘을 통해 외부 데이터셋을 활용해 가상 데이터셋을 생성하고, 이를 이용해 초기 모델을 유지하면서도 백도어 공격을 수행할 수 있도록 설계되었습니다.

- **Technical Details**: TrojanTime은 첫 번째 단계에서 임의의 외부 데이터셋을 바탕으로 하고, 목표 적대적 공격(target adversarial attack)을 통해 가상 데이터셋을 생성합니다. 두 번째 단계에서는 logits 정렬(logits alignment) 및 배치 정규화(batch normalization) 동결을 통해 모델의 일반화 능력을 유지하면서 백도어 공격을 수행합니다. 이 과정에서 생성된 적대적 데이터는 특정 트리거를 통해 오염되어 새로운 백도어 데이터셋을 형성합니다.

- **Performance Highlights**: TrojanTime은 UCR 벤치마크 데이터셋을 사용하여 다양한 TSC 아키텍처에서 5종의 트리거를 평가했습니다. 결과적으로 TrojanTime은 깨끗한 정확성을 유지하면서도 이식성 있는 공격을 성공적으로 수행할 수 있음을 보여줍니다. 또한, 제안된 방어 학습(defensive unlearning) 전략은 ASR(attack success rate)을 줄이면서도 깨끗한 정확성을 보존하는 데 효과적임을 입증하였습니다.



### Evaluating Small Language Models for News Summarization: Implications and Factors Influencing Performanc (https://arxiv.org/abs/2502.00641)
- **What's New**: 이 연구는 19개의 소형 언어 모델(SLM)을 뉴스 요약에 대해 포괄적으로 평가하고, 2000개의 뉴스 샘플을 사용하여 유의미성(relevance), 일관성(coherence), 사실 일치(factual consistency) 및 요약 길이(summary length)에 중점을 두었습니다. 연구 결과, Phi3-Mini와 Llama3.2-3B-Ins와 같은 최상위 성능 모델이 70B 대형 언어 모델(LLM)과 비교 가능한 결과를 생성하며 더 간결한 요약을 생성하는 것을 발견했습니다.

- **Technical Details**: SLM은 대형 언어 모델(LLM)과 동일한 디코더 전용 아키텍처(architecture)를 공유하지만 4억 개 이하의 매개변수(parameters)를 가지고 있습니다. SLM은 스마트폰과 개인용 컴퓨터와 같은 엣지 장치에서 효율적으로 실행되도록 설계되어 사용자 프라이버시를 보호하며 빠르고 안정적이며 저비용 솔루션을 제공합니다. 연구에서는 ROUGE 메트릭을 사용하여 SLM의 성능을 평가하여, 복잡한 프롬프트가 요약 품질을 저하시킬 수 있음을 발견했습니다.

- **Performance Highlights**: 최상의 SLM은 LLM과 유사한 품질의 뉴스 요약을 생성하지만 요약 길이는 훨씬 짧습니다. 특히, 단순한 프롬프트를 사용할 경우 SLM의 성능은 향상되는 경향이 있으며 Instruction tuning의 효과는 모델마다 다르게 나타났습니다. Llama3.2 시리즈 모델의 경우, instruction tuning 후에는 성능이 상당히 향상되었지만, Qwen2 및 InternLM2 모델은 변화가 적었습니다.



### Zeroth-order Informed Fine-Tuning for Diffusion Model: A Recursive Likelihood Ratio Optimizer (https://arxiv.org/abs/2502.00639)
- **What's New**: 본 논문에서는 Recursive Likelihood Ratio (RLR) 최적화 기법을 제안하여 확률적 확산 모델(DM)의 미세 조정을 향상시킵니다. RLR은 zeroth-order gradient estimator를 활용하여 다양한 확산 단계에서 기울기를 비편향적으로 평가할 수 있는 새로운 접근 방식을 제공합니다. 이 방법은 기존의 강화 학습(Reinforcement Learning, RL) 및 잘린 역전파(Backpropagation, BP) 방법의 한계를 극복하여 전체 프로세스에서 높은 효율성을 보장합니다.

- **Technical Details**: RLR optimizer는 DM의 재귀적 구조와 perturbation 기반 기울기 추정 간의 관계를 분석하여 개발되었습니다. 이 최적화 기법은 기존의 BP와 RL의 장점을 결합하면서도 단점을 완화합니다. 예를 들어, RLR은 BP를 통해 선택된 부분에서만 메모리를 제한하고, 나머지 단계에서는 perturbation 기반 방법으로 기울기를 추정함으로써 구조적 편향과 높은 분산을 해결합니다.

- **Performance Highlights**: RLR은 Text2Image 및 Text2Video 작업의 광범위한 평가를 통해 기존의 모든 기법들을 큰 폭으로 초월하는 성능을 보였습니다. 또한, RLR을 위한 새로운 프롬프트 기법을 제안하여 이의 적용 가능성을 크게 높였습니다. 이러한 결과는 RLR의 효과성을 뒷받침하며, DM의 실용적인 활용을 촉진할 것입니다.



### SimulPL: Aligning Human Preferences in Simultaneous Machine Translation (https://arxiv.org/abs/2502.00634)
Comments:
          Accepted to ICLR 2025. 23 pages,13 figures,11 tables

- **What's New**: 본 논문에서는 실시간 기계 번역(SiMT)에서 인간의 선호도를 반영하기 위한 새로운 프레임워크인 Simultaneous Preference Learning(SimulPL)을 제안합니다. 기존 SiMT 기법들이 인간의 선호를 고려하지 못한다는 문제를 해결하고, 번역 품질, 단순성, 일관성 및 지연 시간 선호를 포함한 다섯 가지 주요 선호도를 통해 번역 작업을 최적화합니다. SimulPL은 이 선호도를 기반으로 GPT-4/4o와 함께 효과적으로 학습할 수 있도록 도와줍니다.

- **Technical Details**: SimulPL 프레임워크는 언어학 및 계산 언어학의 기존 연구를 바탕으로 설정되어 있으며, 다섯 가지의 인간 선호도—번역 품질, 일관성, 핵심 포인트, 단순성, 지연 선호—로 구분됩니다. 이 프레임워크는 초기 선호 정렬을 위해 SiMT 모델의 번역 능력과 읽기/쓰기 정책을 동시에 훈련하는 Multi-task Supervised Fine-tuning(MSFT) 방법을 사용합니다. 그 후, SimulDPO를 통해 지연 선호를 최적화 목표에 통합하고, SiMT 모델이 보다 효과적으로 인간의 선호에 맞도록 조정할 수 있습니다.

- **Performance Highlights**: 실험 결과, SimulPL은 모든 지연 수준에서 더 높은 번역 품질을 달성하며, 인간의 선호와의 정렬이 개선된 것을 보여줍니다. 특히, 번역 작업에서 SimulPL이 제안한 선호도 기준을 기반으로 평가한 결과, 기존 방법보다 전반적으로 나은 성능과 선호 정렬을 실현함을 확인했습니다. 이러한 개선 사항들은 번역 품질의 향상뿐만 아니라, 다섯 가지 선호 범주에서의 전반적인 성과를 포함합니다.



### Representations Shape Weak-to-Strong Generalization: Theoretical Insights and Empirical Predictions (https://arxiv.org/abs/2502.00620)
- **What's New**: 본 논문에서 제안하는 Weak-to-Strong Generalization (W2SG) 프레임워크는 약한 모델이 강한 모델을 감독하는 방식의 통찰을 제공합니다. 연구진은 이 접근법을 통해 약한 모델의 미흡한 감독에도 불구하고 강한 모델이 더 나은 성능을 발휘할 수 있음을 입증했습니다. 특히, GPT-4 모델이 약한 GPT-2 모델에 의해 감독받을 때 NLP 작업에서 약 20% 더 나은 성능을 보인다는 결과가 나타났습니다.

- **Technical Details**: W2SG는 약한 모델과 강한 모델의 내부 표현의 주성분(principal components)에서 유도된 커널을 사용하여 특징지어질 수 있습니다. 연구진은 약한 모델이 배우지 못하는 부분을 강한 모델이 배울 수 있도록 하는 공간을 정의함으로써, 약한 감독으로 인해 강한 모델이 잠재력을 발휘하지 못하는 정도를 측정하는 기준을 제시했습니다. 그리고, 이러한 공간의 투영을 통해 약한 감독에서의 오류들이 강한 모델에 의해 어떻게 수정될 수 있는지를 이해하는 데 도움을 주고 있습니다.

- **Performance Highlights**: 연구 결과는 다양한 설정에서의 W2SG 성능 트렌드를 예측할 수 있는 메트릭을 보여줍니다. 이 메트릭은 두 지역 간의 오버랩을 측정하여, 실제 데이터에 대한 예측 성능과의 강한 상관관계를 나타냅니다. 실험은 8개의 데이터 세트에서 150개 이상의 작은 트랜스포머와 52개의 LLM을 포함하여 진행되었으며, 이들 분석은 W2SG 관리의 잠재적 애플리케이션을 시사하고 있습니다.



### Distribution-aware Fairness Learning in Medical Image Segmentation From A Control-Theoretic Perspectiv (https://arxiv.org/abs/2502.00619)
Comments:
          12 pages, 3 figures, 9 tables

- **What's New**: 이 논문에서는 의료 이미지 세분화의 공정성을 보장하기 위한 새로운 접근법으로, 다수의 전문가를 유통시키는 방법인 'Distribution-aware Mixture of Experts (dMoE)'를 제안합니다. 이는 최적 제어 이론(optimal control theory)에서 영감을 받아 개발되었으며, 임상 데이터의 불균형으로 인한 편향 문제에 대응합니다. dMoE는 다양한 네트워크 아키텍처에 통합되어 의료 이미징 분석의 여러 작업에 광범위하게 적용될 수 있습니다.

- **Technical Details**: dMoE는 Mixture of Experts (MoE) 프레임워크를 기반으로 하며, 피드백 제어 메커니즘으로 새롭게 해석되었습니다. 이 구조는 개별적인 특성과 분포 패턴을 적응적으로 통합하여 깊은 신경망 훈련에 활용됩니다. dMoE는 다양한 네트워크 구조, 특히 transformers와 CNNs에서 원활하게 작동하여 2D 및 3D 의료 이미지 세분화 작업에 적용될 수 있습니다.

- **Performance Highlights**: 다양한 임상 데이터셋에서의 실험 결과, dMoE는 공정성 학습 접근법을 한층 발전시켰으며, 불균형 데이터 분포로 인한 편향을 효과적으로 완화하는 데 성공했습니다. 이 접근 방식은 공정성 학습 패러다임 내에서 의료 이미지 세분화의 진단 및 치료 결정 과정을 보다 강력하고 공정하게 만드는 데 유망한 방법을 제공하였습니다.



### DesCLIP: Robust Continual Adaptation via General Attribute Descriptions for Pretrained Vision-Language Models (https://arxiv.org/abs/2502.00618)
- **What's New**: 이 논문에서는 이미지-언어 모델(Vision-Language Models, VLMs)의 지속적 적응에 관한 연구를 진행했습니다. 특히, 기존의 접근 방식이 시각적 특성과 특정 클래스 텍스트를 연결하는 데 집중하여 일반적 지식과 전문 지식 간의 잠재적 관계를 간과한 점에 주목했습니다. 이를 해결하기 위해, 일반 속성(description of general attributes) 설명을 활용하여 시각-GA-클래스(trilateral vision-GA-class) 연관성을 구축하는 'DesCLIP' 방법론을 제안합니다.

- **Technical Details**: DesCLIP은 언어 보조 도구를 사용하여 특정 클래스 객체에 대한 일반 속성 설명 후보를 생성합니다. 이를 통해 얻은 GA(description of general attributes) 설명 임베딩을 시각적-텍스트적 인스턴스 매칭을 위한 대응 텍스트 임베딩으로 사용하고, 시각 인코더를 튜닝합니다. 또한, 클래스 텍스트 임베딩은 공유된 GA 설명 임베딩에 맞춰 점진적으로 조정됩니다.

- **Performance Highlights**: CIFAR100, ImageNet, CUB-200 데이터셋을 포함한 전반적인 실험에서, 제안된 방법은 기존의 지속적 학습 방법에 비해 탁월한 성능을 입증했습니다. 이 연구는 시각-클래스 텍스트 연결 대신에 시각-GA-클래스 연관성을 형성함으로써 지식 소멸을 효과적으로 완화하는 데 기여하고 있습니다. 종합적인 연구 평가를 통해 효과성을 추가적으로 뒷받침합니다.



### Enhancing Code Consistency in AI Research with Large Language Models and Retrieval-Augmented Generation (https://arxiv.org/abs/2502.00611)
- **What's New**: 이 논문에서는 연구 논문에 설명된 알고리즘과 방법론에 대해 코드 구현의 정확성을 검증하기 위한 새로운 시스템을 제안합니다. 이 시스템은 Retrieval-Augmented Generation (RAG) 기법을 활용하여 연구 논문과 코드베이스에서 관련 세부 정보를 추출한 후, Large Language Models (LLMs)을 사용하여 구조화된 비교를 수행합니다. 이를 통해 코드 구현 검증의 정확성과 포괄성을 개선하며, AI 연구의 투명성과 설명 가능성, 재현성을 높이는데 기여합니다.

- **Technical Details**: 시스템 구조는 데이터 추출 서비스, 임베드 모듈 (Paper Vector Store 및 Code Vector Store), 지식 검색 서비스, 보고서 큐레이터 서비스의 네 가지 주요 구성 요소로 이루어집니다. 데이터 추출 서비스는 연구 논문과 코드 파일을 입력으로 받아들여, 이를 효율적으로 처리하여 텍스트를 세분화합니다. 임베드 모듈은 세분화된 데이터를 기계가 읽을 수 있는 형태로 변환하여 두 개의 벡터 저장소에 저장합니다.

- **Performance Highlights**: 시스템은 최종적으로 코드와 논문 간의 일치 분석 결과를 요약한 구조화된 보고서를 제공합니다. 보고서에는 메타데이터, 일치 분석 및 불일치 요약이 포함되며, 각 섹션 간의 일치 정도를 보여주는 표가 제시됩니다. 이러한 분석을 통해 연구의 신뢰성과 재현성을 평가할 수 있으며, 저자와 검토자에게 유용한 일치 점수를 제공합니다.



### Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspectiv (https://arxiv.org/abs/2502.00604)
Comments:
          39 pages, 22 figures

- **What's New**: 본 연구에서는 복합 손실 함수(composite loss functions)를 통한 멀티태스킹 학습(multi-task learning)이 현대 딥러닝의 중요한 기초임을 소개합니다. 특히 물리 정보 신경망(physics-informed neural networks, PINNs)에서 발생하는 손실 항 간의 방향적 충돌 문제를 해결하기 위한 새로운 이론적 접근법 및 실용적 방법을 제시합니다. 손실 항의 방향적 충돌이 1차 최적화 방법의 한계를 제한하고, 2차 최적화 방법이 이러한 충돌을 자연스럽게 해결함을 이론적으로 분석하였습니다.

- **Technical Details**: PINNs는 복합 손실 함수(minimizing a composite loss function)를 통해 부분 미분 방정식(partial differential equations, PDEs)을 근사화합니다. 이 접근법에서는 경계 조건(boundary conditions) 및 데이터 맞춤 목표(data-fitting objectives)를 동시에 만족시켜야 하며, 복잡한 물리적 제약에 의해 이들 목표가 상호 작용하면서 충돌이 발생할 수 있습니다. 본 연구에서는 SOAP(Smooth Approximation of the Hessian)라는 새로운 준 뉴턴 방법이 이론적으로 특수한 경우에 효율적으로 해시안 전처리기(Hessian preconditioner)를 근사화할 수 있음을 증명하였습니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 10개의 도전적인 PDE 벤치마크에서 최신 기술(state-of-the-art)보다 2-10배 향상된 정확도를 기록하며, 레이놀즈 수(Reynolds numbers)가 10,000에 이르는 난류 흐름(turbulent flows)에 처음 적용되었습니다. 이러한 성과는 PINNs 훈련 시의 경량화 기법과 전처리 기술(preconditioning techniques)을 통해 가능했으며, 향후 물리 기반 머신러닝에 대한 최적화 방법 개발에 폭넓은 영향을 미칠 것으로 기대됩니다.



### RPGBENCH: Evaluating Large Language Models as Role-Playing Game Engines (https://arxiv.org/abs/2502.00595)
Comments:
          Submitted to ICML 2025

- **What's New**: RPGBench는 텍스트 기반 롤플레잉 게임(RPG) 엔진으로서 대형 언어 모델(LLMs)의 성능을 평가하기 위해 처음으로 설계된 기준입니다. 게임 생성(Game Creation, GC) 및 게임 시뮬레이션(Game Simulation, GS)이라는 두 가지 핵심 작업을 포함하여, LLM이 구조화된 이벤트-상태 표현을 사용하여 논리적 일관성을 유지한 게임 세계를 생성할 수 있도록 합니다. 이 벤치마크는 자동 검증 시스템을 통해 생성된 게임의 유효성을 검사하고, 이를 활용하여 LLM의 창의성, 일관성 및 복잡성을 평가할 수 있는 새로운 기준을 제시합니다.

- **Technical Details**: RPGBench는 게임 메커니즘과 이벤트-상태 기반 표현을 통해 LLM이 생성한 게임의 구조적 요건을 자동으로 검증하는 BFS 유효성 검사기를 포함합니다. 각 게임은 이벤트 조건, 상태 전이 및 종료 규칙이 잘 이행되었는지의 객관적 기준을 사용하여 평가됩니다. 또한, LLM은 이벤트 계획, 게임 내러티브 및 상태 업데이트라는 3단계로 구성된 동적 시뮬레이션 루프를 운영하며, 이를 통해 이야기에 대한 유연성을 유지하면서도 게임 메커니즘의 확장성을 평가합니다.

- **Performance Highlights**: RPGBench의 실험 결과, 최첨단 LLM들이 매력적인 이야기를 생성할 수 있지만, 복잡한 상황에서 일관되며 검증 가능한 게임 메커니즘을 구현하는 데에 어려움이 있음을 알 수 있었습니다. 이 연구는 평가 과정에서 LLM으로서의 평가 방법을 활용하여 주관적 평가를 수행하는 고유한 견해를 제공합니다. 또한, 인간 평가자와 자동 점수 간의 정렬 및 불일치를 조사하여 주관적 평가의 복잡성을 강조합니다.



### Fast Vision Mamba: Pooling Spatial Dimensions for Accelerated Processing (https://arxiv.org/abs/2502.00594)
Comments:
          20 pages, 15 figures, this https URL

- **What's New**: 최근의 State Space Models (SSMs)을 활용한 컴퓨터 비전 모델들이 효율성을 극대화하고 있습니다. 특히, Mamba는 전통적인 Vision Transformers 대비 선형 복잡성(linear complexity)으로 토큰 상호작용을 처리할 수 있는 방법을 제시합니다. 이러한 빠른 처리 속도를 위해 Fast Vision Mamba (FastVim)라는 새로운 모델을 제안하며, 이는 기존 모델의 성능을 유지하면서 계산 시간을 더욱 단축시킵니다.

- **Technical Details**: FastVim은 평균 풀링(average pooling)을 통해 비전 Mamba 모델에서 재귀(step)를 줄이는 기법을 사용합니다. 구체적으로, 각 Mamba 블록 내에서 이미지 차원에 따라 토큰을 번갈아 풀링하여 재귀적 계산 수를 2배 줄이는 결과를 가져옵니다. 이렇게 얻어진 1D 토큰 그리드를 통해 선택적 스캔을 적용하여 신속한 스캔이 가능하게 합니다.

- **Performance Highlights**: FastVim은 높은 해상도 이미지(2048×2048)에서 기존 Vision Mamba 모델 대비 최대 72.5%의 속도 향상을 보여줍니다. 다양한 비전 태스크에 대해 실험 결과, 이미지를 분류하고, 세포 교란 예측, 분할(segmentation), 객체 감지(object detection) 등의 작업에서 최첨단 성능을 입증했습니다. 특히, FastMaskVim과 FastChannelVim으로 확장하여 비정형 그리드 및 다중 채널 이미징에 적용할 수 있는 가능성을 보였습니다.



### Robust Knowledge Distillation in Federated Learning: Counteracting Backdoor Attacks (https://arxiv.org/abs/2502.00587)
- **What's New**: 본 논문에서는 Federated Learning (FL)에서 발생할 수 있는 backdoor 공격에 대한 새로운 방어 메커니즘인 Robust Knowledge Distillation (RKD)을 제안합니다. RKD는 데이터 분포에 대한 엄격한 가정 없이 모델의 무결성을 향상시키는 데 초점을 맞추고 있습니다. 이 메커니즘은 클러스터링 및 모델 선택 기술을 통합하여 악의적인 업데이트를 식별하고 필터링함으로써 신뢰할 수 있는 모델 앙상블을 형성합니다.

- **Technical Details**: RKD는 기계 학습 모델에서 cosine similarity와 HDBSCAN과 같은 클러스터링 기법을 활용하여 가능성이 있는 이상치를 고립시킵니다. 그 후 선량군 클러스터의 중앙값에 가까운 모델을 선택하여 남아있는 악의적인 업데이트의 영향을 완화합니다. RKD는 앙상블로부터 수집된 정보를 global model로 전달하기 위해 knowledge distillation을 사용합니다.

- **Performance Highlights**: RKD는 다양한 데이터셋(CIFAR-10, EMNIST, Fashion-MNIST)에 대한 광범위한 평가에서 기존의 최첨단 방어 방법들과 비교했을 때 우수한 성능을 보여주었습니다. 높은 공격자 성공률을 17% 미만으로 낮추면서도 모델의 정확도를 80% 이상 유지하는 등 FL의 강인성을 효과적으로 향상시킵니다.



### Defense Against the Dark Prompts: Mitigating Best-of-N Jailbreaking with Prompt Evaluation (https://arxiv.org/abs/2502.00580)
- **What's New**: 최근 연구는 Best-of-N(BoN) 방식의 AI jailbreaking이 랜덤하게 사용된 augmentations(예: 대문자화, 구두점 등)을 반복하여 사용함으로써 모든 주요 대형 언어 모델(LLMs)에서 효과적임을 보여주었습니다. 연구자들은 'Defense Against The Dark Prompts'(DATDP)라는 새로운 방어 방법을 개발하여 이러한 jailbreaking을 100% 차단할 수 있음을 발견하였습니다. DATDP 방법은 LLM을 활용하여 위험하거나 조작적인 행동에 대한 프롬프트를 평가하면서, 제어된 실험을 통해 기존의 공격을 효과적으로 차단합니다.

- **Technical Details**: DATDP는 평가 에이전트를 통한 반복적인 평가 과정을 기반으로 하며, 사용자 제출 프롬프트를 분석하여 유해한 입력을 사전에 차단합니다. 이 방법은 LLaMa-3-8B-instruct 및 Claude와 같은 LLM을 사용하여, 프롬프트가 위험한지 여부를 판단하고 안전한 프롬프트만 응답 모델에 전달합니다. 평가 과정에서는 프롬프트가 공격적인지, 또는 모델의 방어를 시도하고 있는지를 평가하는 단계가 포함됩니다.

- **Performance Highlights**: DATDP는 실험에서 99.5%에서 100%까지의 프롬프트를 차단하는 데 성공하였습니다. 이는 다양한 데이터셋에 적용했음에도 불구하고 일관된 성능을 보이며, 더 작은 LLM을 사용하더라도 유사한 결과를 기록했습니다. 이러한 성공적인 차단 결과는 AI 시스템의 보안을 강화하는 중요한 전략으로 자리 잡을 가능성을 보여주고 있습니다.



### Generating crossmodal gene expression from cancer histopathology improves multimodal AI predictions (https://arxiv.org/abs/2502.00568)
- **What's New**: 이 연구는 PathoGen이라는 새롭고 혁신적인 diffusion 기반 crossmodal generative AI 모델을 소개하고 있습니다. 이 모델은 디지털 병리학 이미지에서 합성된 transcriptomic 데이터를 활용하여 암의 등급 및 생존 위험을 높은 정확도로 예측합니다. 기존의 transcriptomic 테스트가 제한적이었던 실제 클리닉 환경에서 이러한 접근 방식은 비용 효율적인 스크리닝 도구로서의 가능성을 제시합니다.

- **Technical Details**: PathoGen 모델은 The Cancer Genomic Atlas (TCGA)에서 제공하는 데이터 세트를 기반으로 합니다. 이 모델은 H&E 염색 슬라이드의 디지털 이미지를 사용하여 transcriptomic 데이터를 합성하고, 이를 통해 학습한 특성 맵과 함께 암 등급 및 생존 위험 예측에 사용합니다. 예측 성능 향상을 위해 co-attention 메커니즘을 사용하고, 병리학자가 AI 결정에 기여하는 중요 영역을 시각적으로 확인하도록 돕는 attention 기반 heatmap을 제공합니다.

- **Performance Highlights**: 모델은 TCGA의 두 개의 암 코호트에서 테스트되었으며, 생성된 transcriptomic 데이터는 실제 데이터와 높은 유사성을 보였습니다. PathoGen을 통해 합성된 transcriptomic 데이터와 디지털 병리학 이미지로부터 학습된 특징을 결합함으로써 진단 및 예후 예측의 성능이 크게 향상되었습니다. 또한, 모델은 불확실성 정량화를 통해 각 환자에 대한 합성된 transcriptomic 데이터의 가치를 확인할 수 있는 방법을 제공합니다.



### Lessons for GenAI Literacy From a Field Study of Human-GenAI Augmentation in the Workplac (https://arxiv.org/abs/2502.00567)
Comments:
          Pre-print, paper accepted at IEEE EDUCON2025

- **What's New**: 이번 연구에서는 생성형 인공지능(Generative AI, GenAI)의 다양한 산업 내 활용 방법을 조사하고, 이를 바탕으로 학생들을 직업 세계에 적합하게 준비시키기 위한 교육적 접근 방안을 제안합니다. 특히, 제품 개발, 소프트웨어 엔지니어링, 디지털 콘텐츠 제작 등 세 가지 기능에서 GenAI 사용을 비교함으로써 현재 산업에서의 활용 현황을 파악하고자 하였습니다.

- **Technical Details**: 이 연구는 인간 인지(human cognition)에 중점을 둔 인간 강화(human augmentation) 접근 방식을 취하며, GenAI가 작업 관행을 어떻게 보강하고 있는지, 어떤 지식이 중요한지를 탐구합니다. 연구 질문은 GenAI의 사용 실태, 작업자들이 배우고 있는 지식을 어떻게 전수하는지, 그리고 향후 인력 훈련에 대한 시사점을 다루고 있습니다.

- **Performance Highlights**: 연구 결과, GenAI의 사용과 사용자들의 컴퓨팅 지식 수준에 있어 큰 차이가 있음을 보여주었습니다. 어떤 산업에서는 정교한 모델이 사용하는 반면, 다른 산업에서는 상용 애플리케이션만으로 콘텐츠를 생성하고 있습니다. 이는 교육 과정에서 다양한 수준의 GenAI 이해도를 포함해야 한다는 필요성을 시사합니다.



### Milmer: a Framework for Multiple Instance Learning based Multimodal Emotion Recognition (https://arxiv.org/abs/2502.00547)
- **What's New**: 최근 연구에 따르면 감정 인식(Emotion Recognition)은 인간 행동을 이해하는 열쇠로 작용하며, 이를 위한 새로운 멀티모달 프레임워크인 Milmer가 소개되었습니다. Milmer는 얼굴 표정 분석(Facial Expression Analysis)과 EEG 신호를 통합하여 감정을 인식하는데 있어 새로운 접근 방식을 제시합니다. 이 프레임워크는 transformer 기반 융합 방식(Fusion Approach)을 사용하여 시각적 및 생리적 모달리티를 효과적으로 결합합니다.

- **Technical Details**: Milmer 프레임워크는 EEG 전처리 모듈, 얼굴 특징 추출 및 균형 조정 모듈, 그리고 크로스 모달 융합 모듈로 구성됩니다. 본 연구는 감정 관련 데이터셋에서 사전 훈련된 Swin Transformer를 미세 조정(Fine-tune)하여 시각적 특징 추출을 향상시킵니다. 또한, 크로스 어텐션 메커니즘(Cross-Attention Mechanism)을 도입하여 토큰 표현(Token Representation)을 모달리티 간에 균형 있게 유지하여 효과적인 특징 통합을 보장합니다.

- **Performance Highlights**: DEAP 데이터셋에서 실시된 실험 결과, 제안된 Milmer 프레임워크는 96.72%의 분류 정확도를 달성하며, 멀티 클래스 감정 인식(Multi-class Emotion Recognition) 과제에서 우수한 성능을 보였습니다. 각 모듈의 기여도는 ablation study를 통해 검증되어, 고급 특징 추출과 융합 전략이 감정 인식 성능 향상에 있어 중요함을 강조하였습니다. 이는 단일 모달리티 접근 방식에 비해 감정 인식 정확도를 크게 개선하는 결과를 제공합니다.



### Integrating Frequency Guidance into Multi-source Domain Generalization for Bearing Fault Diagnosis (https://arxiv.org/abs/2502.00545)
- **What's New**: 이 논문은 Fourier 기반의 Augmentation Reconstruction Network(FARNet)를 제안합니다. 이 네트워크는 다양한 도메인 간의 차이를 효과적으로 줄이기 위해 진폭 스펙트럼과 위상 스펙트럼을 분석하여 특징을 학습합니다. 또한, manifold triplet loss를 도입하여 더욱 정교한 결정 경계를 형성하고, 이를 통해 보다 나은 일반화 성능을 실현합니다.

- **Technical Details**: FARNet은 진폭 스펙트럼 서브 네트워크와 위상 스펙트럼 서브 네트워크로 구성되어 있으며, 주어진 데이터의 진폭과 위상을 차례로 학습합니다. 이를 통해 frequency-spatial interaction learning 전략을 활용하여 데이터의 전반적 특징을 향상시킵니다. 또한, FSIM(Frequency-Spatial Interaction Module)을 통해 글로벌 정보와 로컬 공간 특징을 통합하여 표현 학습을 촉진합니다.

- **Performance Highlights**: CWRU와 SJTU 데이터셋에 대한 광범위한 실험을 통해 FARNet은 기존의 크로스 도메인 접근 방식들보다 우수한 성능을 보여줍니다. 특히, 다양한 작동 조건에서의 일반화 능력을 향상시키기 위해 설계된 모델로, 고급 결정을 지원하고 결합 학습 정확도를 높였습니다.



### Generic Multimodal Spatially Graph Network for Spatially Embedded Network Representation Learning (https://arxiv.org/abs/2502.00530)
- **What's New**: 이번 연구에서는 Generic Multimodal Spatially Graph Convolutional Network (GMu-SGCN) 모델을 개발하여 공간적으로 임베디드된 네트워크의 효율적인 표현을 목표로 했습니다. 이 모델은 멀티모달 노드 및 엣지 특성을 통해 노드 연결 패턴을 학습할 수 있는 능력을 보유하고 있습니다. GMu-SGCN을 사용하면 다양한 네트워크의 복잡한 패턴을 더 잘 포착하고 표현할 수 있는 가능성이 높습니다.

- **Technical Details**: GMu-SGCN 모델은 두 가지 하위 모델인 Regional Spatial Graph Convolutional Network (RSGCN)과 Edge Spatial Graph Convolutional Network (ESGCN)와 함께 소개됩니다. RSGCN은 노드 관련 특성만 고려하는 반면 ESGCN은 엣지 관련 특성만을 고려합니다. 이 모델은 두 개의 실제 데이터셋, 즉 전력 분배 네트워크와 하천 네트워크를 사용하여 평가되었습니다.

- **Performance Highlights**: 전반적인 평가 분석에 따르면, GMu-SGCN 모델은 GraphSAGE 모델에 비해 엣지 존재 예측 작업의 정확도를 37.1% 향상시켰습니다. 이 모델은 공간적 환경에 따른 연결 패턴이 학습될 수 있다는 것을 보여주며, 복잡한 SEN을 나타내는 데 있어 다차원 공간 특성을 고려하는 것이 중요함을 강조합니다.



### Bridging Internal Probability and Self-Consistency for Effective and Efficient LLM Reasoning (https://arxiv.org/abs/2502.00511)
Comments:
          Under review

- **What's New**: 본 논문은 대규모 언어 모델(LLMs)의 추론 성능을 향상시키기 위한 새로운 접근 방식을 제안합니다. 기존의 perplexity(당혹도)와 self-consistency(자기 일관성) 방법의 한계를 이론적으로 분석하고, Reasoning-Pruning Perplexity Consistency(RPC)라는 새로운 방법론을 도입합니다. 이 방법은 정확한 확률 추정을 통해 빠른 추정 오류 감소와 낮은 모델 오류를 동시에 달성할 수 있는 가능성을 보여줍니다.

- **Technical Details**: RPC는 Perplexity Consistency와 Reasoning Pruning을 결합하여 설계되었습니다. Perplexity Consistency는 LLM의 내부 확률을 자기 일관성 프레임워크에 통합하여, 각 추론 경로의 신뢰도를 평가하는 데 사용됩니다. Reasoning Pruning은 저확률의 추론 경로를 제거하여 효과적으로 추정 오류 감소를 방지하는 역할을 합니다.

- **Performance Highlights**: 논문에서는 RPC가 7개의 벤치마크 데이터셋에서 수행된 실험을 통해 기존 방법들보다 유의미한 성능 개선을 제공함을 입증합니다. RPC는 50% 이상 샘플링 예산을 절약하면서도 동일한 추론 성능을 유지하며, 동일한 예산을 사용할 경우 평균 1.29% 높은 성능을 보입니다. 또한, RPC는 자신감 추정치의 정확도가 기존 방법들에 비해 더 뛰어난 것을 확인했습니다.



### A statistically consistent measure of Semantic Variability using Language Models (https://arxiv.org/abs/2502.00507)
- **What's New**: 최근 논문에서는 언어 모델이 생성하는 출력의 변동성을 해결하기 위한 통계적으로 일관된 의미 변동성 측정을 제안했습니다. 이 측정 방법은 semantic spectral entropy로, 구현이 용이하고 기존 언어 모델만을 사용하여도 실행 가능합니다. 연구에서 이 방법이 언어 모델의 무작위성에도 불구하고 정확한 지표를 생성할 수 있다는 것을 극명하게 보여주었습니다.

- **Technical Details**: 저자들은 텍스트 조각의 수집체를 기반으로 한 의미적 변동성 측정을 제안하며, 이는 기존의 entropy 개념을 활용합니다. 예를 들어, multi-nomial distribution에 대한 불확실성을 측정하기 위해 의미적 클러스터에 대한 확률 분포를 정의하고, 이를 통해 텍스트의 의미적 다양성을 정량화합니다. 또한, 제안된 방법은 최소한의 가정 하에서도 통계적으로 일관성을 보장합니다.

- **Performance Highlights**: 최종적으로, 논문은 기계적인 일관성을 강조하며, 구체적인 예제를 통해 lexically와 syntactically 다른 텍스트의 클러스터를 구성하는 방법을 제시합니다. 이러한 방법은 기존 모델들의 변동성을 극복하고 신뢰할 수 있는 의미적 변동성 측정을 가능하게 합니다. 연구 결과는 언어 모델이 생성하는 텍스트의 불확실성을 효과적으로 평가할 수 있는 새로운 기준을 제시합니다.



### Optimizing Feature Selection in Causal Inference: A Three-Stage Computational Framework for Unbiased Estimation (https://arxiv.org/abs/2502.00501)
- **What's New**: 이 논문에서는 인과 추론에서의 피처 선택(feature selection) 방법의 중요성에 대해 논의하고 있습니다. 특히, 선택된 피처가 인과량의 편향(bias)과 분산(variance)을 줄이는 데 얼마나 중요한지를 강조합니다. 저자들은 기존의 최첨단 피처 선택 방법보다 우수한 성능을 보여주는 향상된 3단계 프레임워크를 제안합니다.

- **Technical Details**: 제안된 프레임워크는 인과 추론 시 처리와 결과 예측 변인 간의 관계를 보다 명확하게 수립할 수 있도록 설계되었습니다. 중요한 것은 혼란 변인(confounders) 및 순수한 결과 예측 변수(pure outcome predictors)를 포함하면서 비효율적인 순수 처리 예측 변수(pure treatment predictors)와 잡음(noise)을 제외하는 것입니다. 이를 통해 저자들은 인과량 추정의 신뢰도를 높이도록 합니다.

- **Performance Highlights**: 제안한 방법론은 대규모의 실제 데이터를 통해 적용 가능성을 입증했으며, 특히 opiod 위기와 관련된 미국의 의료 정책에 대한 인과 관계를 분석하는 데 사용되었습니다. 진행된 여러 실험을 통해, 저자들은 기존 방법들에 비해 편향과 분산을 모두 낮추는 데 성공했으며, 다양한 설정에서 우수한 성능을 보였습니다.



### Video Latent Flow Matching: Optimal Polynomial Projections for Video Interpolation and Extrapolation (https://arxiv.org/abs/2502.00500)
Comments:
          39 pages, 6 figures

- **What's New**: 이번 논문에서는 Video Latent Flow Matching (VLFM)이라는 효율적인 비디오 모델링 프로세스를 소개합니다. 기존 연구들이 비디오 생성을 위해 무작위로 잠재 패치를 샘플링한 것과 달리, VLFM은 강력하게 사전 훈련된 이미지 생성 모델에 의존하여 시간 의존적인 비디오 프레임으로 변환할 수 있는 특정 캡션 유도 흐름을 모델링합니다. 이 연구는 텍스트-비디오 모델링의 단순화를 통한 새로운 알고리즘 설계를 목표로 하고 있습니다.

- **Technical Details**: VLFM은 임의의 프레임 속도로 비디오 생성을 가능하게 하며, HiPPO 프레임워크를 통해 최적의 다항식 프로젝션을 근사하는 방식을 사용합니다. 이를 통해 모델링 효율성이 향상되고, 고정밀 비디오 복원 및 생성을 위한 보간(interpolation) 및 외삽(extrapolation) 기능이 제공됩니다. 우리의 접근법은 주어진 데이터셋 크기에 비례하여 적은 계산 자원으로도 효율적인 모델링이 가능함을 보여줍니다.

- **Performance Highlights**: 실험 결과, VLFM은 OpenVid-1M, MiraData, Pixabay의 비디오와 같은 다양한 데이터셋에서 텍스트-비디오 생성, 보간 및 외삽에 있어 강력한 성능을 발휘하였습니다. 연구는 VLFM의 이론적 장점과 함께 실제 비디오 애플리케이션에서의 잠재력을 강조합니다. 특히, VLFM은 분산 모델 및 디퓨전 변환기(Diffusion Transformer)와의 연계를 통해 더욱 효과적인 성능을 나타냅니다.



### Looking into the Future of Health-Care Services: Can Life-Like Agents Change the Future of Health-Care Services? (https://arxiv.org/abs/2502.00495)
Comments:
          6 pages, 2 figures, 3rd International Conference on Machine Learning and Computing (ICMLC 2011): February 26-28, 2011, Singapore

- **What's New**: 이번 연구에서는 의사와 환자 간의 상호작용 시간의 제약과 전문의에 대한 접근이 제한된 관리형 의료 시스템(manged care system)에서 컴퓨터를 의료 정보 출처 및 자가 건강 관리 도구로 사용하는 경향이 커지고 있음을 다룹니다. 그러나 조사 결과 40% 이하의 정보 탐색자가 온라인 정보가 건강 결정에 도움을 주었다고 응답했습니다.

- **Technical Details**: 본 논문에서는 이러한 문제를 해결하기 위해 사람처럼 상호작용할 수 있는 특화된 에이전트(specialized agent)를 개발하였으며, 이 에이전트는 기본 컴퓨터 기술이 요구되지 않도록 설계되었습니다. 기존의 웹사이트 검색 방식에서는 대면 상호작용이 부족하여 여러 사회적 문제가 발생하기 때문에, 사용자에게 보다 나은 정보를 제공하기 위한 솔루션을 제시하고 있습니다.

- **Performance Highlights**: 연구 결과, 개발된 에이전트는 사용자들이 의료 정보를 효과적으로 검색할 수 있도록 도움을 주며, 온라인 건강 관리 시스템의 접근성을 높이는 데 기여할 수 있습니다. 이를 통해 환자들이 필요로 하는 의료 정보를 쉽게 찾을 수 있도록 하여 결정 과정에서의 증가된 자신감을 제공하고자 합니다.



### Data Overvaluation Attack and Truthful Data Valuation (https://arxiv.org/abs/2502.00494)
- **What's New**: 이 논문은 협업 기계학습(CML) 환경에서 데이터 평가(data valuation)와 관련된 새로운 공격 방법을 제시합니다. 클라이언트가 자신의 데이터 가치를 과장하여 자신에게 유리한 결과를 얻는 '데이터 과대 평가 공격'을 최초로 도입하였습니다. 이 공격은 많은 기존 데이터 평가 지표에 취약하며, 클라이언트가 부정직하게 데이터를 보고할 가능성을 고려합니다.

- **Technical Details**: 연구자들은 데이터의 기여도를 평가하는 기존의 선형 데이터 평가 메트릭에 대한 취약점을 밝히고, 클라이언트들이 데이터 과대 평가 공격을 감행할 경우 복잡한 계산 비용을 동반할 수 있음을 설명합니다. 또한, 'Truth-Shapley'라는 새로운 데이터 평가 지표를 제안하여, 이 메트릭이 데이터 과대 평가 공격으로부터 데이터를 보호할 수 있음을 이론적으로 입증합니다. Truth-Shapley는 데이터 선택 및 보상 할당에서 공정성을 보장합니다.

- **Performance Highlights**: 실험 결과, 데이터 과대 평가 공격은 공격자의 Shapley 가치(SV)를 최대 210% 증가시킬 수 있으며, LOO(leave-one-out) 값은 4배 향상될 수 있음을 보여줍니다. 또한, Truth-Shapley는 데이터를 선택하고 보상할당에서 기존 메트릭들보다 더 강력하고 효과적임을 입증했습니다. 다양한 CML 시나리오에서 실험이 수행되어 그 유용성이 입증되었습니다.



### Enhance Learning Efficiency of Oblique Decision Tree via Feature Concatenation (https://arxiv.org/abs/2502.00465)
- **What's New**: 이번 연구에서는 새로운 방식인 Feature Concatenation을 도입하여 Oblique Decision Tree(ODT)의 학습 효율성을 개선한 FC-ODT를 제안합니다. FC-ODT는 결정 경로를 따라 선형 프로젝션을 전파하는 방식으로, 이전의 ODT의 파라미터 낭비 문제를 해결합니다. 이 연구는 FC-ODT가 전통적인 ODT에 비해 일관성 비율이 더 빠르며, 더 얕은 나무 구조에서도 일반화 성능이 우수하다는 점을 강조합니다.

- **Technical Details**: FC-ODT는 나무 구조를 생성하는 과정에서 레이어별(feature transformation) 특성 변환을 facilitating하여 결정 경로를 따라 최적화된 프로젝션 정보를 자식 노드에 전달할 수 있도록 합니다. 추가로, Feature Concatenation 메커니즘과 Ridge Regression 기법을 결합하여 깊은 노드에서의 선형 모형의 Retraining을 가능하게 하며, 이는 다중공선성(multicollinearity) 문제를 완화하는 데 도움을 줍니다. 이로써 FC-ODT는 파라미터의 고갈을 줄이고 overfitting 위험을 낮춥니다.

- **Performance Highlights**: FC-ODT는 시뮬레이션 데이터와 실제 데이터세트에서 실험을 통해 기존의 최첨단 ODT보다 우수한 성능을 발휘함을 증명합니다. 특히, 짧은 나무 깊이에서도 FC-ODT의 일관성 비율은 статистично(Statistically) 우수하며, 과도한 파라미터 사용을 최소화하였습니다. 결과적으로 이 연구는 FC-ODT가 기존 ODT보다 높은 학습 효율성을 가지고 있음을 실험적으로 뒷받침합니다.



### AudioGenX: Explainability on Text-to-Audio Generative Models (https://arxiv.org/abs/2502.00459)
Comments:
          14 pages

- **What's New**: 새로운 기술 AudioGenX는 텍스트-오디오 생성 모델의 설명 가능성을 향상시키기 위한 혁신적인 방법을 제안합니다. 이 방법은 텍스트 입력의 중요성을 강조하며, 오디오 토큰 수준에서 신뢰할 수 있는 설명을 제공합니다. 사실(factual)과 반사실(counterfactual) 목표 함수를 활용하여 더욱 정교한 설명을 생성하는 것이 특징입니다.

- **Technical Details**: AudioGenX는 TAG 모델의 잠재 표현 벡터를 활용하여 사실적(factual) 및 반사적(counterfactual) 변화의 효과를 관찰합니다. 크로스-어텐션(cross-attention) 계층에서 여러 토큰의 어텐션 점수를 동시에 변화시키는 소프트 마스크를 적용하여, 입력 텍스트의 중요성을 정량화합니다. 또한, 이 방법은 오디오 생성 태스크를 위한 새로운 평가 지표를 통해 성능을 검증합니다.

- **Performance Highlights**: 종합적인 실험을 통해 AudioGenX의 설명 정확성을 입증하였으며, 기존 기법과 비교하여 효과성을 보여줍니다. 사용자 요구에 따라 전체 오디오에 대한 전반적인 설명뿐만 아니라 특정 구간에 대한 세부 설명을 제공할 수 있어, 편집 작업 및 모델 동작 이해에 유용한 인사이트를 제공합니다.



### Towards Privacy-aware Mental Health AI Models: Advances, Challenges, and Opportunities (https://arxiv.org/abs/2502.00451)
Comments:
          18 pages, 2 figures

- **What's New**: 이 논문은 정신 건강 분야에서는 개인정보 보호와 관련된 기존 문제를 깊이 분석하며 인공지능(AI) 모델을 통한 진단 및 치료의 접근성을 높이기 위한 새로운 접근 방식을 제안합니다. 특히, 멀티모달 데이터 처리 기술이 정신 질환 진단에 어떻게 활용될 수 있는지를 강조하며, 관련 개인정보의 유출 위험을 낮추기 위한 방안을 모색합니다. 새로운 데이터 수집 및 모델 훈련을 위한 개인정보 보호 중심의 파이프라인 개발 방향도 제시하고 있습니다.

- **Technical Details**: 정신 건강 진단 자동화를 위해 텍스트, 오디오 및 비디오 데이터를 분석할 수 있는 멀티모달 AI 모델의 발전이 필요합니다. 그러나 데이터 수집 과정에서 GDPR 및 HIPAA와 같은 개인정보 보호 규정을 준수해야 하므로, PII(개인 식별 정보)를 안전하게 보호하는 방법을 모색해야 합니다. 이러한 과정을 통해 환자의 목소리와 얼굴 특징 등이 악용되는 위험을 줄이고, 안전하게 연구를 진행할 수 있도록 해야 합니다.

- **Performance Highlights**: 이 연구는 멀티모달 AI 모델들이 정신 질환 진단을 보조할 수 있는 잠재력을 가지고 있지만, 개인정보 보호 문제로 인해 적절한 데이터 세트를 확보하기가 어렵다는 점을 지적합니다. 논문에서는 데이터 익명화, 합성 데이터 생성, 그리고 개인정보 보호 훈련 방법 등의 해결책을 통해 이러한 문제를 극복할 방안을 제시하고, 이와 함께 모델 성능 평가를 위한 새로운 평가 체계가 필요하다고 강조합니다. 전반적으로, 개인정보 보호를 고려한 AI 도구들을 개발하여 환자의 치료 결과를 더욱 향상시키고자 하는 목표를 가지고 있습니다.



### Model-Free Predictive Control: Introductory Algebraic Calculations, and a Comparison with HEOL and ANNs (https://arxiv.org/abs/2502.00443)
- **What's New**: 이 논문에서는 모델 자유 예측 제어(Model-Free Predictive Control, MFPC)에 대해 소개하며, 이는 모델 예측 제어(Model Predictive Control, MPC)의 확장입니다. MFPC는 선형 미분 방정식을 통해 재구성되었으며, 최근의 모델 자유 제어 발전과 결합된 최적 제어에 대한 새로운 관점을 기반으로 하고 있습니다. 또한, MFPC는 동적 프로그래밍(Dynamic Programming)과 해밀턴-자코비-벨만 방정식(Hamilton-Jacobi-Bellman equation), 폰트리야긴 최대 원리(Pontryagin's Maximum Principle)를 대체하는 방법으로 제시됩니다.

- **Technical Details**: 이 연구는 특히 단일 입력 단일 출력(SISO) 초국소 모델을 사용하여 MFPC 개념을 설명합니다. 이 모델은 시스템의 알려지지 않은 구조 및 외부 방해 요소를 포함하고 있으며, 상수 값 α는 제어 및 출력 변수 사이의 크기를 일치시키기 위해 선택됩니다. 논문에서는 레그랑지안(Lagrangian) 및 비용 함수(cost function)를 정의하고, 오일러-라그랑지 방정식(Euler-Lagrange equation)에 근거한 비동차 선형 미분 방정식을 통해 최적 해를 찾는 과정을 상세히 설명합니다.

- **Performance Highlights**: 화학 반응기와 두 개의 탱크 시스템을 사례로 사용하여 MFPC의 성능을 평가합니다. 연구 결과는 MFPC의 구현이 간단하고 계산 부담이 낮음을 보여주며, HEOL 접근법과 비교했을 때 우위를 점하고 있지 않음을 확인합니다. 또한 최근 복잡한 인공 신경망(ANN) 구조를 통한 모델 식별이 모든 제어 및 AI 분야에서 완전한 모델링과 머신 러닝이 항상 필요하지 않음을 시사하고 있습니다.



### Compilation and Fast Model Counting beyond CNF (https://arxiv.org/abs/2502.00434)
- **What's New**: 이 논문은 결정론적 분해형 부정 정규형 (d-DNNF) 회로를 Boolean 함수의 효율적인 변환 가능성에 대한 이론적인 지식을 강화합니다. 주요 기여는 Incidence treewidth로 매개변수화된 특정 제약 조건의 합성에 대한 고정 매개변수 처리 가능 (FPT) 컴파일 방법을 제시하는 것입니다. 이로 인해 CNF에 대한 기존 결과를 포괄하게 됩니다.

- **Technical Details**: 논문에서는 특정 제약 조건에 대해 FPT 컴파일 알고리즘이 가능하다는 것을 증명합니다. 이러한 제약 조건은 상수 개수의 상태로 변수를 지정할 수 있을 때만 존재하며, 예로는 홀수/짝수 합 제약이 포함됩니다. FPT 컴파일은 incidence treewidth에 대해 단일 지수적 실행 시간을 가지며, 보다 효율적인 모델 카운팅 알고리즘도 제안합니다.

- **Performance Highlights**: 결과적으로, 주어진 제약 구조의 최대 상태 크기가 w일 때, 모델을 카운팅하는 알고리즘이 제안됩니다. 이 알고리즘은 F의 구조와 해당 그래프의 루트 분할을 이용해 수행되며, O(w²k(|F|+|var(F)|))의 시간 복잡도를 가집니다. 이는 특정 제약 조건에 대한 더 빠른 모델 카운트가 가능함을 보여줍니다.



### MQuant: Unleashing the Inference Potential of Multimodal Large Language Models via Full Static Quantization (https://arxiv.org/abs/2502.00425)
Comments:
          First quantization solution for Multimodal large language models applicable to 5 mainstream MLLMs

- **What's New**: MQuant은 멀티모달 대형 언어 모델(MLLMs)의 고유한 도전에 대처하기 위해 설계된 포스트 트레이닝 양자화(post-training quantization, PTQ) 프레임워크입니다. 기존의 양자화 방법이 MLLMs에 적합하지 않은 이유는 다양한 종류의 시각적 및 텍스트 토큰 간의 분포 차이, 높은 추론 지연(inference latency), 그리고 해밀턴 변환에 의한 극단적인 아웃라이어(outlier) 문제가 발생하기 때문입니다. MQuant은 이러한 문제를 해결하기 위해 모달리티(모드) 별 정적 양자화, 주의 불변 유연한 스위칭 및 회전 크기 억제와 같은 새로운 방법론을 도입했습니다.

- **Technical Details**: MQuant은 세 가지 주요 방법론을 사용하여 MLLMs의 효율성을 극대화합니다. 첫째, 모달리티-특화 정적 양자화(Modality-Specific Static Quantization, MSQ)를 통해 시각적 및 텍스트 토큰에 대해 상이한 정적 스케일(scale)을 적용합니다. 둘째, 주의 불변 유연한 스위칭(Attention-Invariant Flexible Switching, AIFS)은 토큰 순서를 재배열하여 비싼 토큰별 스케일 계산을 피하며, 셋째, 회전 크기 억제(Rotation Magnitude Suppression, RMS)를 통해 온라인 해밀턴 회전에서 발생하는 아웃라이어를 최소화합니다.

- **Performance Highlights**: MQuant은 Qwen-VL, MiniCPM-V, CogVLM2 등 다섯 가지 주요 MLLMs에 대해 수행된 실험에서, 부동소수점(float) 정확도에서 98% 이상을 유지하면서도 추론 지연을 최대 30%까지 줄였습니다. 이는 기존의 PTQ 방법론보다 현저히 우수한 성능을 나타냅니다. MQuant은 자원 제약이 있는 환경에서 효율적이고 정확한 MLLM 추론을 위한 중요한 초석이 될 것으로 기대합니다.



### MarketSenseAI 2.0: Enhancing Stock Analysis through LLM Agents (https://arxiv.org/abs/2502.00415)
Comments:
          25 pages, 7 figures, Under review at Financial Innovation (FIN)

- **What's New**: MarketSenseAI는 Large Language Models (LLMs)를 활용하여 재무 뉴스를 분석하고, 역사적 주가와 회사 기본정보, 거시경제 환경을 통합하여 종합적인 주식 분석 및 선택을 지원하는 새로운 프레임워크입니다. 이 논문에서는 LLMs의 기술적 발전에 따라 MarketSenseAI의 개선된 기능을 소개하고 SEC 서류 및 수익 호출을 처리하는 Retrieval-Augmented Generation과 LLM 에이전트를 결합한 새로운 아키텍처를 통해 기존 버전 대비 기본 분석 정확도의 현저한 향상을 보여줍니다. 시장에 대한 심층적 통찰력을 제공하며, AI 기반 투자 전략의 견고함을 강조합니다.

- **Technical Details**: MarketSenseAI는 데이터 흐름과 에이전트 책임을 포함하여 LLM 아키텍처에 대한 업데이트를 기초로 하여, Chain-of-Agents (CoA) 접근법을 도입하여 대규모 재무 데이터를 더 세밀하게 처리할 수 있도록 합니다. 또한, Retrieval-Augmented Generation (RAG) 모듈을 통해 다양한 전문가 보고서를 처리하고, 전통적인 분석 방법에서는 놓치기 쉬운 거시경제 문맥을 제공하는 등의 강점을 갖추고 있습니다. 이러한 기술적 개선은 금융 데이터와 비구조적 데이터를 효과적으로 통합하는 데 중점을 두고 있습니다.

- **Performance Highlights**: S&P 100 주식에 대한 2023-2024년의 실증 평가 결과, MarketSenseAI는 누적 수익률 125.9%를 기록했으며, 이는 시장 지수 수익률 73.5%에 비해 현저한 성과를 보였습니다. 2024년 동안 S&P 500을 검증한 결과, MarketSenseAI는 시장보다 33.8% 높은 Sortino 비율을 달성하여 뛰어난 리스크 조정 수익을 보여주었습니다. 이는 소매 및 기관 투자자 모두에게 고급 분석를 제공할 수 있는 가능성을 보여줍니다.



### Causal Abstraction Learning based on the Semantic Embedding Princip (https://arxiv.org/abs/2502.00407)
- **What's New**: 본 논문에서는 구조적 인과 모델(Structural Causal Models, SCMs)에서의 인과 추상화(Causal Abstraction, CA) 학습에 대한 새로운 접근법을 제시합니다. 특히, 저자들은 고차원 및 저차원 SCM 간의 매핑을 형식화하는 CA 프레임워크를 도입하며, 데이터가 미정렬된 경우와 같은 어려운 환경에서도 적용 가능한 학습 문제를 해결합니다. 이 연구의 핵심 원리는 고차원 분포가 저차원 초차원 서브스페이스에 존재한다는 의미의 의미적 임베딩(semantic embedding)입니다.

- **Technical Details**: 이 연구에서는 비구조적 인과 모델 학습을 위한 일반적인 프레임워크를 제안하며, 특히 선형 CA와 관련된 문제를 해결합니다. 고차원 및 저차원 확률 측정 간의 사상(morphism)을 찾는 카테고리 이론적 접근 방식으로, 이 원리는 Stiefel manifold의 기하학과 자연스럽게 연결됩니다. 또한, 비선형 최적화(Riemannian optimization) 문제를 해결하기 위해 세 가지 알고리즘을 개발하였으며, 각각은 실세계 뇌 데이터와 함께 다양한 사전 정보 수준에서 성능을 입증합니다.

- **Performance Highlights**: 제안된 방법들은 합성 데이터 및 실제 뇌 데이터에 대한 실험을 통해 뛰어난 성능을 발휘함을 입증하였습니다. 이 연구는 CA 학습 방법과 실제 세계 응용 간의 간극을 메우기 위한 첫 단계를 제시하며, 신경과학과 같은 다양한 응용 분야에서 부분적인 구조 지식이 활용될 수 있음을 강조합니다. 저자들은 세 가지 알고리즘을 통해 선형 CA의 비부드러운 및 부드러운 학습 문제를 해결하는 데 성공하였으며, 이는 CA 학습의 실용성을 크게 향상시킬 것으로 기대됩니다.



### Spectro-Riemannian Graph Neural Networks (https://arxiv.org/abs/2502.00401)
Comments:
          ICLR 2025

- **What's New**: 이번 논문에서는 Spectro-Riemannian Graph Neural Networks (CUSP)를 제안하여, 곡률(Curvature)와 스펙트럼(Spectral) 신호를 통합한 첫 번째 그래프 표현 학습 패러다임을 소개합니다. CUSP는 복잡한 그래프 구조를 효과적으로 학습할 수 있는 혼합 곡률 스펙트럴 GNN으로, 정수 곡률 매니폴드(hyperbolic, spherical, Euclidean)에서 노드 임베딩을 최적화하는 스펙트럴 필터를 학습합니다. 이 모델은 Cusp Laplacian, Cusp Filtering, Cusp Pooling의 세 가지 새로운 구성 요소를 포함하여 그래프의 다양한 곡률 영향을 반영합니다.

- **Technical Details**: CUSP의 핵심 요소인 Cusp Laplacian은 Ollivier-Ricci 곡률을 기반으로 한 전통적인 그래프 라플라시안의 확장으로, 곡률 신호를 보다 잘 포착하도록 설계되었습니다. Cusp Filtering은 여러 Riemannian 그래프 필터를 활용하여 고유 스펙트럼의 다양한 대역에서 정보를 추출합니다. 마지막으로, Cusp Pooling은 곡률 기반 위치 인코딩을 활용한 계층적 주의 메커니즘으로, 서로 다른 곡률의 하위 구조의 상대적 중요성을 평가합니다.

- **Performance Highlights**: CUSP는 8개의 상동적(Homophilic) 및 비상동적(Heterophilic) 데이터셋에서 노드 분류 및 링크 예측 작업을 평가한 결과, 기존 최첨단 모델에 비해 최대 5.3%의 성능 향상을 기록했습니다. 이러한 성과는 혼합 곡률과 스펙트럴 신호를 통합한 새로운 접근 방식의 효과를 보여줍니다. CUSP는 실제 세계 그래프에서 다양한 지리적 및 스펙트럴 특성을 모델링하는 데 뛰어난 능력을 발휘합니다.



### The Impact of Persona-based Political Perspectives on Hateful Content Detection (https://arxiv.org/abs/2502.00385)
- **What's New**: 이 연구는 정치적 다양성을 모델 출력에 도입하기 위해 persona-based prompting 전략이 어떻게 효과적으로 사용할 수 있는지를 조사합니다. 전통적인 방식인 정교한 정치적 pretraining에 비해, persona-based 접근 방식이 더 계산적으로 효율적이라는 점이 강조됩니다. 특히, 이 연구는 혐오 발언 탐지라는 실제 응용 분야에서 이러한 접근 방식의 실효성을 평가합니다.

- **Technical Details**: 저자들은 Hateful Memes와 MMHS150K라는 두 개의 데이터 세트를 이용하여 다양한 정치적 입장을 가진 200,000개의 synthetic personas를 분석합니다. 이들은 Political Compass Test (PCT)를 통해 정치적 입장을 매핑하고, 페르소나의 식별자가 분류 결정에 미치는 영향을 평가합니다. 연구 결과는 정책 편향이 실질적인 분류 작업에는 약한 상관관계를 가지며, 이는 기존의 정치적 전처리 방법과 다른 결과임을 보여줍니다.

- **Performance Highlights**: IDEFICS-3 모델은 persona-based prompting을 통해 Hateful Memes 데이터 세트에서 높은 성능을 기록했으며, harmfulness detection에서 0.908의 정확도와 0.890의 F1 점수를 달성했습니다. 이는 페르소나를 활용한 접근 방식이 모델의 분류 능력을 유지하면서도, 정치적 편향의 영향을 최소화할 수 있음을 시사합니다.



### Masked Generative Nested Transformers with Decode Time Scaling (https://arxiv.org/abs/2502.00382)
- **What's New**: 최근 시각적 생성 영역에서 뛰어난 품질의 콘텐츠를 생성하는데 있어 큰 발전이 있었지만, 모델의 추론 효율성에서 병목 현상이 발생하고 있습니다. 본 연구는 여러 번의 계산을 요하는 전통적인 접근 방식을 개선하기 위해, 생성 과정의 각 부분에 필요한 계산량을 조절하는 방안과 계산 재사용(Cache) 전략을 도입합니다. 이를 통해 다양한 크기의 모델을 효과적으로 활용하여 계산 비용을 줄이면서도 유사한 품질의 결과를 제공합니다.

- **Technical Details**: 이 연구에서는 Masked Generate Nested Transformers (MaGNeTS)라는 새로운 모델 아키텍처를 제안하며, 디코딩 과정 동안 모델 크기를 단계적으로 조정하여 계산 효율성을 극대화합니다. MaGNeTS는 작은 모델이 더 많은 토큰을 처리할 수 있도록 지원하며, 큰 모델이 더 적은 토큰을 처리하도록 합니다. 실험을 통해 ImageNet, UCF101, Kinetics600 데이터셋에서 엄청난 계산 효율성을 입증하고, 각 모델이 공유하는 매개변수 공간과 함께 동작하는 방식도 설명합니다.

- **Performance Highlights**: MaGNeTS는 기존 대비 약 3배 적은 계산으로 유사한 품질의 이미지를 생성할 수 있으며, 최첨단 모델들과 경쟁력 있는 성능을 보여줍니다. 특히, MaskGIT++와의 비교 실험에서 이 모델은 성능 개선이 두드러지며, 모든 작업에서 2.5배에서 3.7배에 달하는 계산 이득을 나타냅니다. 또한, 비디오 데이터셋에 대해서도 유의미한 효과를 증명했습니다.



### Latent Action Learning Requires Supervision in the Presence of Distractors (https://arxiv.org/abs/2502.00379)
Comments:
          Preprint. In review

- **What's New**: 최근 Latent Action Policies (LAPO)를 기반으로 하는 latent action learning이 대규모 로봇 공학 및 강화 학습 분야에서 놀라운 사전 훈련 효율성을 보였습니다. 이전 연구는 주로 진정한 행동(ground-truth actions)만으로 설명 가능한 간섭 요인이 없는 데이터에 집중했으나, 실제 비디오 데이터에는 행동과 관련된 방해 요소(Distractors)가 포함되어 있다는 점에서 한계가 있었습니다. 이 논문에서는 이러한 방해 요소가 latent action learning에 미치는 영향을 실증적으로 분석하고, LAPO가 이러한 시나리오에서 어려움을 겪는다고 밝힙니다.

- **Technical Details**: 기존 연구는 주로 간섭 요인이 없는 데이터에서 latent action learning의 효과를 연구했습니다. 이 작업에서는 Distracting Control Suite (DCS)를 사용하여 행동-연관 방해 요소의 영향을 조사하고, LAPO의 품질을 8배 향상시키는 LAOM이라는 간단한 LAPO 수정 버전을 제안합니다. 중요한 점은, 전체 데이터 세트의 약 2.5%만으로도 진정한 행동에 대한 지도 감독을 제공했을 때, downstream 성능이 평균 4.2배 향상된다는 것입니다.

- **Performance Highlights**: 연구 결과, 주어진 예산의 행동 레이블 수로 더 나은 결과를 얻기 위해서는 방해 요소가 존재할 때 Latent Action Models (LAM)을 우선 학습하고 그 이후에 진정한 행동으로 디코딩하는 현재의 파이프라인은 최적이 아님을 보였습니다. 본 논문은 또한 지도 감독을 활용하며 latent action learning의 성능이 더 잘 일반화된다는 점을 강조하며, 이는 기존의 역동적 모델에 기반한 접근 방식과 비교하여 개선된 결과를 제공합니다.



### When End-to-End is Overkill: Rethinking Cascaded Speech-to-Text Translation (https://arxiv.org/abs/2502.00377)
- **What's New**: 이 논문에서는 End-to-End (E2E) 음성-텍스트 변환 모델의 성공에도 불구하고, 검증된 방식으로 남아 있는 계단식(cascaded) 음성-텍스트 번역 모델의 필요성을 강조합니다. ASR(Automatic Speech Recognition)과 MT(Machine Translation) 간의 오류 전파를 최소화하기 위해, ASR로부터의 여러 후보들을 MT에 통합하는 방안을 제안합니다. 또한, 일반적인 오류의 원인과 음성 영역의 샘플 간 유사성으로 인한 차이에 대한 폭넓은 분석을 통해 논문의 독창성을 발휘하고 있습니다.

- **Technical Details**: 제안된 방법은 ASR에서 여러 후보를 통합하여 기계 번역의 정확도를 향상시키고, 자기지도(self-supervised) 음성 표현을 활용하여 언어 정보를 보존하는 방식입니다. 특히, 여러 후보 중 최선의 결과를 선택하게 하여, 음성 데이터로부터 발생하는 오차를 최소화하며, ASR과 MT 사이의 언어적 차이를 극복하는 정확한 번역을 보장합니다. 또한, 기존의 방법과 달리 추가 파라미터 설정 없이 N-best 전략을 통해 성능을 향상시킬 수 있는 점이 특징입니다.

- **Performance Highlights**: 제안된 모델은 음성-텍스트(S2T) 변환 작업에서 계단식 모델 중 최고의 성능을 자랑하며, 다양한 ASR 및 MT 사전 훈련(pre-trained) 모델을 쉽게 활용할 수 있습니다. 낮은 비용으로 대규모 데이터를 요구하지 않으면서도 빠른 학습 속도와 최소한의 데이터 사용으로 뛰어난 결과를 달성할 수 있다는 점에서도 큰 장점을 제공합니다. 최종적으로, 단일 후보를 사용하는 기존 모델과 비교했을 때, 제안된 접근 방식이 번역 품질을 크게 개선할 수 있음을 보여주고 있습니다.



### What should an AI assessor optimise for? (https://arxiv.org/abs/2502.00365)
- **What's New**: 이번 논문은 AI 시스템 평가 시 사용하는 다양한 메트릭에 대한 최적화 방법을 탐구합니다. 특히, 평가자 모델(assessor model)의 효용성을 다루며, 특정 메트릭을 위한 최적화가 항상 최상의 결과를 보장하지 않는다는 점을 강조합니다. 특히 일부 단조 변환이 예상 외로 유용할 수 있으며, 예를 들어 로지스틱 손실(logistic loss)이 회귀 문제에서 절대 오차(absolute error)나 제곱 오차(squared error) 최소화에 효과적일 수 있음을 보여줍니다.

- **Technical Details**: 논문은 평가자 모델을 사용하여 기존의 AI 시스템의 출력값을 예측하고 이를 다양한 메트릭의 성능 측정에 활용하는 방법을 탐구합니다. 특히 회귀 문제에서 L2 손실(squared error)과 로지스틱 손실(logistic loss) 간의 비교를 다루며, 평가자가 최적화할 메트릭을 변경했을 때 발생하는 효용의 변화를 실험적으로 분석합니다. 이 연구는 회귀 모델에서의 손실 함수 선택이 모델 성능에 미치는 영향을 명확히 제시합니다.

- **Performance Highlights**: 실험 결과는 단조 변환(monotonic transformation)의 사용이 때때로 더 나은 성능을 제공할 수 있다는 것을 보여줍니다. 예를 들어, 특정 상황에서 로지스틱 손실이 전통적인 손실 측정 방식보다 뛰어날 수 있다는 지점을 포착했습니다. 이러한 발견은 다양한 메트릭 간의 관계를 최적화하는 데 있어 평가자 모델의 중요성을 잘 보여줍니다.



### Do Audio-Visual Segmentation Models Truly Segment Sounding Objects? (https://arxiv.org/abs/2502.00358)
- **What's New**: 이 논문에서는 오디오-비주얼 분할(Audio-Visual Segmentation, AVS)의 강건성과 관련된 문제를 체계적으로 조사합니다. 저자들은 기존 모델들이 시각적인 정보에 과도하게 의존하는 경향이 있음을 발견하였고, 이로 인해 소리가 없는 경우나 관련이 없는 소리에 대해 신뢰할 수 없는 예측을 하게 됩니다. 이를 해결하기 위해 다양한 부정적인 오디오 상황을 포함하는 새로운 벤치마크인 AVSBench-Robust를 제안합니다.

- **Technical Details**: 이 연구는 AVS 모델들이 시각적 중요성(visual salience)에 기반하여 분할 마스크를 생성하는 경향이 있음을 발견하였고, 오디오 맥락을 무시함으로써 발생하는 편향을 다룹니다. 저자들은 긍정적 및 부정적 오디오-비주얼 쌍(Balanced Training with Negative Samples)을 포함한 훈련 방법과 분류기 주도 유사성 학습(Classifier-Guided Similarity Learning)을 통한 강건한 훈련 전략을 제안합니다. 이러한 방법론은 모델들이 오디오 정보에 대한 이해를 강화하도록 돕습니다.

- **Performance Highlights**: 실험 결과, 기존의 최첨단 모델들은 부정적인 오디오 조건 하에서 일관되게 실패하며, 높은 오탐률(False Positive Rate, FPR)을 보였습니다. 반면에, 저자들의 접근법은 표준 메트릭 및 강건성 지표에서 큰 개선을 이루었으며, 높은 품질의 분할 성능을 유지하면서 거의 완벽한 오탐률을 달성했습니다. 이 연구는 AVS 모델들의 오디오-비주얼 정보 통합 능력을 평가하는 새로운 기준을 제공합니다.



### PM-MOE: Mixture of Experts on Private Model Parameters for Personalized Federated Learning (https://arxiv.org/abs/2502.00354)
- **What's New**: 본 논문에서 제안하는 PM-MoE( Personalized Mixture of Experts) 아키텍처는 서로 다른 클라이언트 간 개인화된 모듈을 상호 강화하게 만드는 새로운 접근 방식을 제공합니다. 전통적인 개인화되고 연합된 학습 방법의 한계점인 통계적 불균형 문제를 해결하기 위해, PM-MoE는 글로벌 모델과 개인적인 모듈을 효과적으로 조합합니다. 이를 통해 각 클라이언트는 다른 클라이언트로부터 유용한 개인화된 매개변수를 선택하여 성능을 향상시킬 수 있습니다.

- **Technical Details**: PM-MoE 아키텍처는 두 단계로 구성되어 있으며, 첫 번째 단계에서는 모델을 사전 훈련하여 글로벌 및 개인화된 모듈을 생성합니다. 두 번째 단계에서는 개인화된 모듈의 혼합(MPM) 및 에너지 기반의 잡음 제어(EDM) 방법을 통해 서로 다른 클라이언트의 개인화된 모듈들이 서로의 성능을 향상하도록 합니다. 또한, 이 과정에서 개인 정보 보호 요건을 준수하며 매개변수의 누수가 발생하지 않도록 설계되었습니다.

- **Performance Highlights**: PM-MoE는 아홉 가지 최신 PFL 벤치마크를 포함한 여섯 개의 데이터셋에 대한 실험을 통해 다양한 설정에서 성능 개선을 입증했습니다. PM-MoE를 적용함으로써 다양한 PFL 방법의 성능이 일관되게 향상되었으며, 이는 개인화된 지식의 공유가 클라이언트 간의 협력을 통해 어떻게 이루어질 수 있는지를 보여줍니다. 이러한 결과는 PM-MoE의 효용을 보다 강화하며 개인화된 연합 학습의 새로운 가능성을 제시합니다.



### Multi-Order Hyperbolic Graph Convolution and Aggregated Attention for Social Event Detection (https://arxiv.org/abs/2502.00351)
- **What's New**: 이 논문은 Multi-Order Hyperbolic Graph Convolution with Aggregated Attention (MOHGCAA)라는 새로운 프레임워크를 도입하여 사회적 사건 탐지(Social Event Detection, SED)의 성능을 향상시키고 있습니다. 기존의 유클리드 기반 접근 방식의 한계를 해결하면서 사건 간의 다차원 관계를 포착하는 데 초점을 맞추고 있습니다. MOHGCAA는 고차원 구속적 관계를 인코딩하여 SED의 주요 도전 과제를 해결하는 혁신적인 방법론을 제시하고 있습니다.

- **Technical Details**: MOHGCAA 프레임워크는 유클리드 공간에서 하이퍼볼릭 공간으로 노드 특징을 투영하고, 하이퍼볼릭 공간에서의 접선 평면에서 1차 및 고차 구문적 관계를 동시에 인코딩합니다. 이 설계는 과도하게 깊은 그래프 컨볼루션 층의 필요성을 없애면서도 학습된 특징의 풍부함을 보존합니다. 동적 주의 메커니즘은 다중 차원 표현을 집계하여 작업 관련 관계를 강조하여 최종 집계된 특징이 하이퍼볼릭 공간으로 다시 매핑됩니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서의 광범위한 실험을 통해 MOHGCAA는 강력한 성능을 입증하였고, 감시 및 비감시 환경 모두에서 우수한 결과를 달성하였습니다. 본 연구는 다중 차원 관계를 포착할 수 있는 MOHGCAA의 능력을 통해 사회적 사건 탐지에서의 기존 방법론 대비 명백한 개선을 보여 주었습니다.



### OrcaLoca: An LLM Agent Framework for Software Issue Localization (https://arxiv.org/abs/2502.00350)
- **What's New**: 이 논문에서는 LLM(Large Language Model) 에이전트가 소프트웨어 문제를 정확히 식별하는 데 있어 매핑을 개선하기 위한 새로운 프레임워크인 OrcaLoca를 소개합니다. OrcaLoca는 LLM 에이전트와 정밀 코드 검색 메커니즘 간의 효과적인 통합 부족 문제를 해결하여 더 나은 결과를 제공합니다.

- **Technical Details**: OrcaLoca는 우선 순위 기반 스케줄링(priority-based scheduling), 작업 분해(action decomposition), 관련 점수(relevance scoring) 및 거리 인지 컨텍스트 가지치기(distance-aware context pruning) 기능을 통합합니다. 이러한 기능들은 소프트웨어 문제를 보다 정확하게 지역화하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, OrcaLoca는 SWE-bench Lite에서 65.33%의 함수 일치율(function match rate)을 기록하며 새로운 오픈 소스 최첨단(SOTA) 성과를 보여줍니다. 또한, 패치 생성 통합을 통해 오픈 소스 프레임워크의 최종 해결률(resolved rate)을 6.33% 향상시켰습니다.



### Actor Critic with Experience Replay-based automatic treatment planning for prostate cancer intensity modulated radiotherapy (https://arxiv.org/abs/2502.00346)
Comments:
          27 Pages, 8 Figures, 4 Tables

- **What's New**: 이번 연구에서는 IMRT(Intensity Modulated Radiation Therapy)에서의 실시간 치료 계획 문제를 해결하기 위해 새로운 확률 기반의 심층 강건 강화 학습(DRL) 에이전트를 개발하였습니다. 이 모델은 효율적인 훈련과 광범위한 적용 가능성을 특징으로 하며, 적대적 공격에 대한 강인성도 확보했습니다. 특히, Fast Gradient Sign Method(FGSM)를 통해 강인성을 확보하는 데 중점을 두었습니다.

- **Technical Details**: 연구진은 ACER(Actor-Critic with Experience Replay) 아키텍처를 사용하여, 프로스트 상태의 환자 사례에서 IMRT의 치료계획 매개변수(TPPs)를 조정하는 에이전트를 개발했습니다. 이 에이전트는 단일 환자 사례로 학습했으며, 두 개의 독립적인 사례로 검증되었고, 세 개의 데이터 세트에서 300개 이상의 계획을 테스트했습니다. 치료 계획의 품질은 ProKnow 점수를 통해 평가되었습니다.

- **Performance Highlights**: 모델은 단일 사례로부터 학습했음에도 불구하고 우수한 일반화 성능을 보였습니다. ACER 기반 계획 이전 평균 점수가 6.20이었던 반면, 이후에는 93.09%의 사례가 완벽한 점수인 9를 달성하였고 평균 점수는 8.93으로 나타났습니다. 에이전트는 최적의 TPP 조정을 우선시하며 적대적 공격에 대해서도 강인성을 유지합니다.



### The Composite Task Challenge for Cooperative Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2502.00345)
- **What's New**: 이 논문에서는 협력적인 다중 대리인 강화 학습(MARL)에서 노동 분업(DOL)의 개념을 효과적으로 활용하기 위한 새로운 작업 세트를 제안합니다. 기존의 MARL 테스트베드에서 DOL의 중요성이 간과되어 왔음을 지적하며, DOL과 협력이 필수적인 작업을 설계하여 연구의 발전을 촉진할 필요성을 다룹니다. 이로써 DOL이 실제 협력 시스템에서의 성능 향상에 기여할 수 있도록 하는 방향성을 제시합니다.

- **Technical Details**: 제안한 작업들은 DOL을 성공적으로 구현하기 위해 세 가지 주요 요소로 구성됩니다: 정보 간섭, 하위 작업 간 차별성, 그리고 하위 작업의 수입니다. 이러한 디자인 요소는 대리인들이 DOL 및 협력 메커니즘을 다양한 복잡도 수준에서 학습하고 적용하는 데 도전하도록 설계되었습니다. Composite Tasks Challenge(CTC)라는 새로운 테스트베드에서 8개의 작업이 제공되며, 이는 DOL 구현의 효율성을 평가할 수 있도록 기획되었습니다.

- **Performance Highlights**: 실험 결과, 기존의 MARL 방법들이 CTC 작업에서 만족스러운 성능을 보이지 못하고 있는 것으로 나타났습니다. 또한 제안된 CTC 작업의 간소화된 변형에서 기존 방법들이 상당히 향상된 성능을 보였지만, 여전히 낮은 안정성을 나타냈습니다. 이를 통해 DOL 구현의 중요성이 강조되며, 보다 견고한 방법론의 필요성이 제기됩니다.



### UGPhysics: A Comprehensive Benchmark for Undergraduate Physics Reasoning with Large Language Models (https://arxiv.org/abs/2502.00334)
Comments:
          9 pages

- **What's New**: 본 논문은 LLMs의 물리학 문제 해결 능력을 평가하기 위해 UGPhysics라는 대규모 물리학 벤치마크를 소개합니다. UGPhysics는 5,520개의 학부 수준 물리학 문제를 포함하고 있으며, 13개 과목과 4개의 물리학 추론 기술을 다룹니다. 이 연구는 기존의 교육 평가에서 활용되는 물리학 문제의 범위를 효과적으로 반영하고 있습니다.

- **Technical Details**: UGPhysics 벤치마크는 철저한 데이터 검증 과정을 통해 5,520개의 문제를 수집, 형식화, 분할 및 필터링했습니다. 이 문제들은 영어와 중국어로 번역되어 총 11,040개의 문제로 나뉘어 평가됩니다. 문제들에는 독립형 답변 타입과 복합형 답변 타입이 있으며, 이들을 해결하기 위해 필요한 기술이 네 가지로 분류됩니다.

- **Performance Highlights**: 본 논문은 31개의 주요 LLM들을 평가한 결과, OpenAI-o1-mini가 49.8%의 최고 정확도를 기록하는 것을 보여주었습니다. 이러한 결과는 현재 LLM들이 물리학 문제 해결에서 직면하고 있는 과제를 강조하며, 수학적 능력 이상의 물리학적 추론 기술이 필요한 필요성을 부각시킵니다. UGPhysics와 MARJ는 물리학 추론을 위한 AI 발전의 중요한 촉매제가 될 것으로 기대됩니다.



### From Few to Many: Self-Improving Many-Shot Reasoners Through Iterative Optimization and Generation (https://arxiv.org/abs/2502.00330)
Comments:
          Expanded version of the ICLR 2025 paper

- **What's New**: 최근 장기 문맥을 가진 대형 언어 모델(LLMs)의 발전으로 인해, 많은 수의 예시를 활용한 학습(many-shot in-context learning)이라는 새로운 패러다임이 등장하였습니다. 본 연구는 이러한 many-shot ICL의 성능 향상에 기여하는 요인을 분석하였고, 여러 가지 예시를 단순히 늘리는 것이 유용한지에 대한 의문을 제기합니다. 특히, 연구진은 몇 개의 중요한 예시만으로도 성능이 크게 향상될 수 있음을 발견했습니다.

- **Technical Details**: 연구에서는 Bayesian optimization을 기반으로 하는 BRIDGE라는 알고리즘을 제안합니다. 이 알고리즘은 최적화(optimize) 단계와 생성(generate) 단계를 반복하여 효율적으로 many-shot ICL을 개선하는 방식입니다. BRIDGE는 핵심적인 예시 집합을 발견하고, 이를 활용하여 새로운 예시를 자동으로 생성하는 접근 방식을 통해 성능을 증가시킵니다.

- **Performance Highlights**: BRIDGE는 Gemini, Claude, Mistral 등 다양한 LLM에 적용되어 상징적 추론(symbolic reasoning), 수치적 추론(numerical reasoning), 코드 생성(code generation)과 같은 여러 작업에서 유의미한 성과를 보였습니다. 최종적으로 BRIDGE는 기존의 many-shot 예시와 최적화된 예시보다 더 우수한 성과를 나타내, 많은 예시를 활용한 학습의 실용성을 증대시킵니다.



### CoddLLM: Empowering Large Language Models for Data Analytics (https://arxiv.org/abs/2502.00329)
- **What's New**: 이번 논문에서는 데이터 분석을 위한 특별히 설계된 기초 모델인 CoddLLM을 소개합니다. 이 모델은 데이터 관리에 대한 이해를 높이고 복잡한 현실의 분석 작업을 해결할 수 있도록 해주는 새로운 데이터 레시피와 스케일 가능한 합성 데이터 생성 방식을 기반으로 하고 있습니다. 또한 이 연구는 사용자와 데이터 간의 자연어 상호작용을 통해 SQL 쿼리 생성 및 데이터 발견을 단순화할 수 있는 가능성을 보여줍니다.

- **Technical Details**: CoddLLM은 12억 개의 매개변수를 가지며 Mistral-NeMo-12B 기반으로 설계되었습니다. 이 모델은 합성된 교육 데이터를 사용하여 훈련되며, 새로운 테이블-텍스트 정렬 작업을 통해 데이터 관리의 복잡한 개념을 이해합니다. 더불어, AnalyticsMMLU라는 대규모 다중 선택 질문 벤치마크를 만들었으며, 이는 데이터베이스 및 데이터 분석, 머신 러닝 관련 문제를 포함합니다.

- **Performance Highlights**: CoddLLM은 여러 벤치마크 데이터셋에서 가장 높은 평균 정확도를 달성하며, 기존의 모델인 GPT-3.5-Turbo와 비교하여 우수한 성능을 보였습니다. 특히, AnalyticsMMLU에서 GPT-4o를 12.1% 초과했으며, Text-to-SQL 작업에서는 평균 24.9%의 개선을 나타냈습니다. 이를 통해 CoddLLM은 데이터 분석의 새로운 기준을 세우고 있습니다.



### MIM: Multi-modal Content Interest Modeling Paradigm for User Behavior Modeling (https://arxiv.org/abs/2502.00321)
- **What's New**: 본 논문에서는 Multi-modal Content Interest Modeling(오른 발제, MIM)이라는 새로운 패러다임을 제안합니다. 이는 기존의 ID 임베딩 기반 접근방식의 한계를 극복하고, 콘텐츠와 사용자 관심사 간의 차이를 메꾸기 위한 세 가지 주요 단계로 구성됩니다. 또한, Taobao 플랫폼에서 오프라인 실험과 온라인 A/B 테스트를 통해 MIM의 효율성을 입증하였습니다.

- **Technical Details**: MIM은 사전 훈련, 콘텐츠-관심 인식 감독 세부 조정(C-SFT), 및 콘텐츠-관심 인식 UBM(CiUBM)이라는 세 가지 핵심 단계를 포함합니다. 이 과정에서 사용자 행동 신호를 활용하여 임베딩을 사용자 선호와 정렬되도록 유도하며, ID 기반 협업 필터링 신호를 통합하여 효과적인 사용자 행동 모델링 프레임워크를 제공합니다.

- **Performance Highlights**: MIM은 Taobao에서의 대규모 데이터 세트 실험을 통해 클릭률(CTR)을 +14.14% 개선시키고, 수익률(RPM)을 +4.12% 증가시켜 실질적인 산업 응용 가능성을 보여주었습니다. 이러한 개선은 MIM의 효율적인 훈련 및 추론 구조 덕분으로, 다양한 추천 작업에 널리 적용될 수 있습니다.



### Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values (https://arxiv.org/abs/2502.00313)
- **What's New**: 본 논문은 대규모 언어 모델(LLMs)이 자원 분배에서 공정성을 지향하는지에 대한 실증적 분석을 시도하고 있습니다. 연구는 LLM의 응답이 인간의 분배 선호와 얼마나 일치하는지를 평가하며, 공정성 개념을 충족하는 능력을 비교합니다. 특히, LLM들이 종종 경제적 효율성을 우선시하는 경향이 있음을 보여줍니다.

- **Technical Details**: 연구에서는 LLM의 응답이 공정성의 다양한 개념, 즉 Equitability (EQ), Envy-Freeness (EF), Rawlsian Maximin (RMM)과 얼마나 잘 일치하는지를 평가하기 위해 여러 가지 실험을 수행했습니다. 각 LLM의 성능은 GPT-4o, Claude-3.5S, Llama3-70b, Gemini-1.5P처럼 최신 모델들로 비교되었습니다. 연구는 비대칭 자원 할당 과제에서도 LLM의 응답이 인간의 선택과는 다르게 나타났음을 밝혔습니다.

- **Performance Highlights**: LLMs는 공정성을 기반으로 한 자원 분배 결정에서 인간의 선택과는 상당한 불일치를 보였습니다. GPT-4o는 다른 LLM들과 비교할 때 공정성을 달성하기 위해 금전을 더 효과적으로 활용하는 모습을 보였으며, 주어진 선택지에서 공정한 해결책을 선택할 때도 인간의 가치를 더 잘 반영했습니다. 그러나 LLM은 특정 공정성 개념을 일관되게 충족하지 못하고, 자원 할당 문제에서 인간과의 가치 일치도가 낮은 것으로 나타났습니다.



### SigWavNet: Learning Multiresolution Signal Wavelet Network for Speech Emotion Recognition (https://arxiv.org/abs/2502.00310)
Comments:
          Published in: IEEE Transactions on Affective Computing

- **What's New**: 본 논문은 음성 감정 인식(SER) 분야에서 raw waveform 음성 신호로부터 의미 있는 표현을 추출하는 새로운 end-to-end (E2E) 딥러닝 다중 해상도 프레임워크를 소개합니다. 이 접근 방식은 고속 이산 웨이브렛 변환(FDWT)의 속성을 활용하여 노이즈 간섭 및 시스템 복잡성과 같은 기존의 한계를 극복합니다. 이를 통해 wavelet 기반과 노이즈 제거를 위한 학습 가능한 모델을 도입하여 SER의 정확도를 높입니다.

- **Technical Details**: 제안된 프레임워크는 웨이브렛 계수의 비대칭 하드 스레시홀드를 위한 활성화 함수와 함께 한 차원 팽창 합성곱 신경망(1D dilated CNN), 공간적 주의층, 양방향 게이트 순환 유닛(Bi-GRU) 및 시간적 주의층을 결합하여 감정 특징의 미세한 공간 및 시간적 특성을 효율적으로 포착합니다. 이 모델은 가변 길이 음성을 세분화 없이 처리할 수 있으며, 전처리 및 후처리의 필요성을 없앱니다.

- **Performance Highlights**: 이 모델은 IEMOCAP 및 EMO-DB 데이터셋에서 기존의 최신 기법보다 더 우수한 성능을 보였습니다. 연구 결과, 신경망 아키텍처가 음성 신호의 복잡한 감정을 효과적으로 인식하고 분석할 수 있도록 설계되었습니다. 소스 코드는 GitHub 리포지토리에 공유되어 다른 연구자들이 쉽게 접근하고 활용할 수 있습니다.



### Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation (https://arxiv.org/abs/2502.00306)
- **What's New**: 본 논문은 Retrieval-Augmented Generation (RAG) 시스템에서의 membership inference 공격 기법인 Interrogation Attack (IA)을 제시합니다. IA는 모델의 성능을 저하시키지 않으면서도, 최소 30개의 자연어 쿼리로 특정 문서의 존재 여부를 추론하는 데 성공합니다. 기존의 기법들과는 달리, IA는 스텔스성(stealthy)를 유지하면서도 높은 정확도와 재현성을 보여줍니다.

- **Technical Details**: RAG 시스템의 구조는 문서 집합과 검색기, 생성 모델로 구성됩니다. RAG는 쿼리에 따라 지식 기반에서 관련 문서를 검색하고 이를 모델의 프롬프트에 포함하여 출력을 생성합니다. IA는 타겟 문서와 밀접하게 연관된 자연어 쿼리를 설계하여, 문서의 존재 여부를 추론하는 방식으로 작동합니다.

- **Performance Highlights**: IA는 기존의 membership inference 공격들과 비교해 2배 향상된 True Positive Rate (TPR@1% FPR)을 기록했으며, 공격자의 쿼리는 덜 포착되고 대략 5%의 탐지율을 자랑합니다. 이는 기존 방법들이 90% 이상 탐지되는 것과 비교됩니다. 또한 이 공격은 RAG 시스템 사용 시 발생할 수 있는 개인정보 유출 문제를 해결하는 데 기여할 것으로 기대됩니다.



### DEUCE: Dual-diversity Enhancement and Uncertainty-awareness for Cold-start Active Learning (https://arxiv.org/abs/2502.00305)
Comments:
          18 pages, 3 figures, 12 tables. Accepted manuscript by TACL. For published version by MIT Press, see this https URL

- **What's New**: 이 논문에서는 Cold-start Active Learning (CSAL)에 대한 새로운 접근법을 제시합니다. 기존 방법들이 약한 클래스와 어려운 대표 예시를 무시했던 문제를 해결하기 위해, Dual-Diversity Enhancing and Uncertainty-aware (DEUCE) 프레임워크를 제안합니다. DEUCE는 사전 훈련된 언어 모델(PLM)을 활용하여 텍스트 표현과 예측 불확실성을 효율적으로 추출합니다.

- **Technical Details**: DEUCE 프레임워크는 Dual-Neighbor Graph (DNG)를 구성하여 텍스트 다양성과 클래스 다양성을 결합합니다. 이는 데이터 분포를 균형 있게 만들고, 밀도 기반 클러스터링을 통해 불확실성 정보를 전파하여 어려운 대표 사례를 선택합니다. 이러한 접근법은 CSAL에서 클래스 균형과 하드 대표 데이터 선택에 효과적입니다.

- **Performance Highlights**: DEUCE는 여섯 개의 NLP 데이터세트에서 실험을 통해 그 우수성과 효율성을 입증했습니다. 이 프레임워크는 탐색과 활용 간의 균형을 잘 이루어, CSAL에서의 데이터 수집 성능을 향상시키는 것을 목표로 합니다. DEUCE는 텍스트와 클래스 다양성을 동시에 고려하여, CSAL의 클래스 불균형 문제를 해결하는데 기여합니다.



### HoP: Homeomorphic Polar Learning for Hard Constrained Optimization (https://arxiv.org/abs/2502.00304)
Comments:
          in submission

- **What's New**: 이 논문에서는 Homeomorphic Polar Learning (HoP)이라는 새로운 방법을 소개하여 star-convex hard-constrained optimization 문제를 해결합니다. 기존의 learn-to-optimize (L2O) 방법들이 출력의 최적성(optimality)과 타당성(feasibility)을 보장하는 데 어려움을 겪는 반면, HoP는 homeomorphic mapping을 신경망(neural networks)에 내장함으로써 이러한 문제를 극복합니다. 이 방법은 추가적인 패널티(penalty)나 수정(correction) 없이 end-to-end 방식으로 훈련이 가능하다는 장점을 가지고 있습니다.

- **Technical Details**: HoP는 star-convexity 개념을 통해, 제한조건 내의 솔루션을 보장하는 동시에 최적 솔루션을 확실히 찾도록 설계되었습니다. 또한, polar coordinates(극좌표계)와 homeomorphism(동형사상)을 활용하여 hard constraint(강한 제약 조건)의 최적화를 제공합니다. 이 프레임워크는 다양한 적용 사례에 적합하게 적응할 수 있도록 직접적인 손실(loss) 함수에 목적 함수를 활용하여 훈련됩니다.

- **Performance Highlights**: HoP는 다양한 합성 최적화 작업(synthetic optimization tasks)과 실제 세계의 응용 프로그램(real-world applications)에서 수행 평가를 진행한 결과, 기존 L2O 방법들을 초월하는 성능을 보였습니다. 이 방법은 모든 테스트 케이스에서 솔루션이 최적과 더 가깝게 도달할 수 있도록 하며, 하드 제약조건을 철저히 준수한 것으로 나타났습니다. 추가적으로, HoP는 constraint satisfaction(제약 조건 만족도)과 최적화 효율성(optimization efficiency) 면에서도 전통적인 솔버와 학습 기반 솔버 모두를 능가하는 결과를 보였습니다.



### Learning to Fuse Temporal Proximity Networks: A Case Study in Chimpanzee Social Interactions (https://arxiv.org/abs/2502.00302)
- **What's New**: 이 논문에서는 침팬지의 사회적 구조를 유도할 수 있는 개인 그룹을 식별하기 위한 새로운 접근 방식을 제시합니다. 연구자들은 장기간에 걸쳐 수집한 사회적 상호작용 데이터를 기반으로 네트워크 표현을 사용합니다. 특히, 각 시간 스탬프마다 단일 가중 네트워크를 생성하고 이를 최적화하는 혁신적인 손실 함수를 도입하였습니다.

- **Technical Details**: 본 연구는 여러 유형의 근접 데이터(입수 가능한 근접 기록들)를 단일 네트워크로 결합하는 방법을 다룹니다. 논문에서는 Bernoulli 시퀀스를 기반으로 한 시간에 따른 개인 유사성 개념을 확립하고, 이를 통해 장기간의 밀접한 관계를 탐지할 수 있습니다. 또한, 연구자들은 침팬지의 사회적 상호작용을 시간에 따라 다층적으로 분석할 수 있는 새로운 최적화 파이프라인을 제안합니다.

- **Performance Highlights**: 이 연구의 적용 결과로 침팬지의 사회 네트워크 시간 시리즈에서 클리크(그룹 간의 강한 결속)를 효과적으로 탐지할 수 있음을 보여주었습니다. 또한, 연구에 사용된 데이터 세트는 이론적으로 준수하는 구조적 일관성을 기반으로 한 유의미한 소셜 구조 식별에 충분한 정보를 제공했음을 확인했습니다. 이는 침팬지의 사회적 상호작용을 더 깊이 이해하는 데 기여할 것으로 기대됩니다.



### Estimating LLM Uncertainty with Logits (https://arxiv.org/abs/2502.00290)
- **What's New**: 이 논문에서는 LLM의 토큰별 불확실성을 실시간으로 추정하기 위한 새로운 프레임워크인 Logits-induced Token Uncertainty (LogU)를 제안합니다. 기존의 확률 기반 접근법의 한계를 극복하고, 다양한 응답에 대한 신뢰도를 평가할 수 있는 효율적이고 효과적인 방법으로 구체화되었습니다. LogU는 주로 aleatoric uncertainty와 epistemic uncertainty를 분리하여 토큰 수준에서 불확실성을 명확하게 추정할 수 있는 기능을 제공합니다.

- **Technical Details**: LogU는 디리클레 분포를 활용하여 aleatoric uncertainty와 epistemic uncertainty를 구분하고 평가합니다. 이 모델은 샘플링 없이도 각 응답의 신뢰도를 실시간으로 추정할 수 있으며, 사용자에게 더 정확한 피드백을 제공할 수 있습니다. LogU는 LLM의 본질적인 불확실성을 포착할 수 있는 새로운 기준을 제공하며, 효과적인 증거 모델링을 통해 이를 구현합니다.

- **Performance Highlights**: 실험 결과 LogU의 효과성이 입증되었으며, 이 방법을 통해 LLM의 할루시네이션 문제를 해결하는 데 기여할 수 있는 잠재력이 나타났습니다. LogU는 주요 토큰의 신뢰도를 평가하는 데 집중함으로써 신뢰할 수 없는 응답을 줄이고, 다운스트림 작업에 대한 가이드를 제공하는 데 유용합니다. 이러한 성과는 LLM의 안정성을 개선하는데 중요한 발전을 의미합니다.



### Sigmoid Self-Attention is Better than Softmax Self-Attention: A Mixture-of-Experts Perspectiv (https://arxiv.org/abs/2502.00281)
Comments:
          Fanqi Yan, Huy Nguyen contributed equally to this work. 51 pages, 2 figures, 3 tables

- **What's New**: 이번 논문에서는 Transformer 아키텍처에서 주요 메커니즘인 self-attention이 softmax 대신 sigmoid 함수를 사용하는 것의 효과에 대해 연구합니다. 기존의 self-attention 구조는 token 간의 경쟁을 발생시켜 정보 손실이 있을 수 있지만, sigmoid 함수를 사용하면 이러한 경쟁이 줄어듭니다. 이 연구는 sigmoid self-attention이 softmax에 비해 더 적은 데이터로도 효율적인 샘플링을 보여준다는 것을 이론적으로 증명합니다.

- **Technical Details**: 논문에서는 self-attention 매트릭스의 각 행을 혼합 전문가(Mixture-of-Experts; MoE) 형태로 나타낼 수 있음을 보여줍니다. 이러한 MoE 관점에서 sigmoid self-attention의 수렴 분석을 통해, 특정 오차 범큼 내에서 목표 함수를 근사하는 데 드는 데이터 포인트 수가 polynomial하게 필요하다는 점을 발견했습니다. 반면, softmax self-attention은 동일한 근사 오차를 달성하는 데 지수적으로 많은 데이터를 요구합니다.

- **Performance Highlights**: 실험을 통해 sigmoid self-attention은 전통적인 softmax self-attention과 유사한 성능을 보이며, 훈련과 추론 속도가 현저히 빨라진다는 것을 확인했습니다. sigmoid self-attention의 element-wise 특성 덕분에 softmax 정규화의 필요성이 없어져 더 빠르고 메모리 효율적인 구현이 가능해졌습니다.



### DUET: Optimizing Training Data Mixtures via Feedback from Unseen Evaluation Tasks (https://arxiv.org/abs/2502.00270)
- **What's New**: 이 논문은 DUET라는 새로운 알고리즘을 제안하여 ML 모델의 훈련 데이터 믹스를 개선하고, 이는 보이지 않는 평가 작업에서의 성능을 극대화합니다. DUET는 Bayesian optimization (BO)과 influence function (IF) 기반 데이터 선택 방법을 결합하여 피드백 루프를 활용합니다. 이로써, 모델의 성능 향상에 기여하며, 적절한 데이터를 선택하는 과정을 자동화합니다.

- **Technical Details**: DUET는 global-to-local 접근 방식을 사용하여 피드백을 바탕으로 훈련 데이터 믹스의 비율을 조정합니다. 전역 수준에서는 BO를 통해 피드백을 받고, 지역 수준에서는 IF를 사용해 각 데이터 도메인에서 고품질 데이터를 선택합니다. 이러한 작동 방식은 모델이 보이지 않는 평가 작업에서 최적 성능을 나타내기 위한 데이터 선택의 효율성을 증가시킵니다.

- **Performance Highlights**: 실험 결과, DUET는 기존의 표준 방법들보다 이미지 및 LLM 평가 작업에서 더 우수한 훈련 데이터 믹스를 발견했습니다. 다양한 도메인에서의 평가 작업에 대해 효과성이 입증되었으며, 데이터 믹스의 최적화를 통해 ML 모델의 성능을 끌어올리는 데 성공했습니다. DUET는 실질적으로 ML 모델의 훈련 과정에서의 데이터의 적합성을 높이는 데 큰 역할을 합니다.



### INSIGHT: Enhancing Autonomous Driving Safety through Vision-Language Models on Context-Aware Hazard Detection and Edge Case Evaluation (https://arxiv.org/abs/2502.00262)
- **What's New**: INSIGHT라는 새로운 프레임워크를 도입하여, 시각적(visual) 및 텍스트(text) 입력을 결합하여 자율주행 시스템의 위험 감지와 엣지 케이스 평가를 향상시킵니다. 기존의 모델들은 드문 사건에 대한 일반화가 부족했으나, INSIGHT는 다중 모달 데이터 융합(multi-modal data fusion)을 통해 이러한 문제를 해결합니다. 이 모델은 상황 인식 개선과 보다 나은 의사결정을 통해 자율주행 시스템의 안전성을 높입니다.

- **Technical Details**: INSIGHT는 대형 언어 모델(LLM)과 비전 인코더(vision encoder)를 통합한 계층적 비전-언어 모델(VLM)입니다. 이 모델은 Vision Transformer를 사용하여 시각적 특징을 추출하고, 사전 학습된 언어 모델에서 텍스트 임베딩(text embeddings)과 융합합니다. 주의 기반 메커니즘(attention-based mechanisms)과 좌표 회귀 기법을 통해 공간적인 위험 로컬라이제이션을 최적화하며, 주석이 달린 데이터셋에서 미세 조정을 통해 엣지 케이스 감지 능력을 향상시킵니다.

- **Performance Highlights**: BDD100K 데이터셋에서의 실험 결과, INSIGHT는 기존 모델들에 비해 위험 예측의 간단함과 정확성에서 현저한 개선을 보였습니다. 특히, 드문 사건에 대한 일반화 성능이 크게 향상되어 자율주행 시스템의 전반적인 강인함과 안전성을 높이는 데 기여합니다. 이를 통해 복잡한 실제 상황에서 자율주행 기술의 신뢰도를 강화할 수 있습니다.



### Mordal: Automated Pretrained Model Selection for Vision Language Models (https://arxiv.org/abs/2502.00241)
- **What's New**: 이번 논문에서는 자동화된 멀티모달 모델 검색 프레임워크인 Mordal을 도입하여 특정 작업을 위한 가장 적합한 Vision Language Model (VLM)을 효율적으로 찾는 방법을 제안합니다. 기존의 VLM들은 전문가에 의해 수작업으로 제작되어, 사용자가 원하는 작업에 맞는 자동 프레임워크가 없었습니다. Mordal은 후보 모델의 수를 줄이고 각 후보를 평가하는 시간을 최소화하여 효율적인 검색을 가능하게 합니다. 연구 결과, Mordal은 기존의 grid search에 비해 GPU 시간을 최대 8.9배에서 11.6배 절약할 수 있음을 보여주는 동시 신 모델을 발견했습니다.

- **Technical Details**: Mordal은 VLM을 구축하고 훈련하기 위한 기존 접근 방식을 크게 개선하는 방법론을 제공합니다. VLM은 일반적으로 Vision Encoder, Feature Projector 및 Language Model로 구성되며, 각각의 구성 요소는 특정한 역할을 수행하여 입력 이미지와 텍스트를 결합하고 해석합니다. 특히 다양한 pretrained 비전 인코더와 언어 모델을 조합해 최적의 성능을 낼 수 있는 조합을 탐색하며, 이를 위해 초기 후보 모델들을 유사도 기준으로 클러스터링하고 평가 시간을 단축하는 조기 중지 메커니즘을 도입했습니다.

- **Performance Highlights**: Mordal을 사용한 성능 평가에서, 전체적으로 49개의 VLM 후보를 대상으로 한 그리드 서치보다 높은 효율성과 적은 계산 시간으로 최적의 VLM을 찾는 것을 확인했습니다. 본 연구에서는 특히 비전-텍스트 정렬 데이터에 훈련된 Feature Projector의 중요성을 강조하며, 최적의 pretrained Vision Encoder 및 Language Model 조합을 찾기 위해 진행된 다양한 실험 결과를 공유합니다. 실험 결과들은 VLM 성능 향상에 기여하며, 여기서 발견된 새로운 VLM들은 기존의 최첨단 모델들을 초과하는 성능을 나타냈습니다.



### A Hybrid Random Forest and CNN Framework for Tile-Wise Oil-Water Classification in Hyperspectral Images (https://arxiv.org/abs/2502.00232)
- **What's New**: 이 논문에서는 하이퍼스펙트럼 이미지(HSI)에서 석유와 물을 분류하기 위한 새로운 하이브리드 랜덤 포레스트(Random Forest) 및 컨볼루션 신경망(CNN) 프레임워크를 제안합니다. 이 프레임워크는 이미지를 작고 겹치지 않는 타일로 나누어 공간적 맥락을 보존하는 데 초점을 맞추고 있습니다. 랜덤 포레스트는 픽셀 단위 분류에서 뛰어난 성능을 보이지만, 공간적 관계를 충분히 활용하지 못하는 한계를 극복하기 위해 CNN이 도입되었습니다. 이 조합은 하이퍼스펙트럼 이미지의 맥락 인식 분석을 위한 효과적인 접근 방식을 제시합니다.

- **Technical Details**: 본 연구에서 사용된 데이터셋은 365nm에서 2500nm까지 스펙트럼을 촬영한 18개의 하이퍼스펙트럼 이미지로 구성된 HOSD dataset입니다. 데이터를 전처리하는 과정에서는 잡음을 제거하고, PCA(주성분 분석)을 통해 차원 축소를 수행하여 99%의 분산을 유지하면서 32개의 주성분을 추출하였습니다. 이러한 과정은 모델의 일반화 능력을 높이고 계산 효율성을 개선하는 데 기여했습니다. 이미지를 64x64 크기의 비겹치는 타일로 나누어 학습, 검증, 테스트 세트로 분할하였습니다.

- **Performance Highlights**: 하이브리드 접근법은 랜덤 포레스트의 확률 맵을 기반으로 학습한 CNN을 통해 성능을 크게 향상시켰습니다. 해당 방법은 baseline에 비해 7.6%의 recall(0.85), 2.4%의 F1 점수(0.84), 그리고 0.54%의 AUC(0.99) 개선을 달성했습니다. 이러한 결과는 확률적 출력과 공간적 특성 학습을 결합하여 하이퍼스펙트럼 이미지 분석의 정확성을 강화하는 데 성공했다는 것을 보여줍니다.



### Should You Use Your Large Language Model to Explore or Exploit? (https://arxiv.org/abs/2502.00225)
- **What's New**: 최근 세대의 대형 언어 모델(LLMs)이 탐색-착취(tradeoff) 문제에 어느 정도 도움을 주는지 평가하는 연구가 진행되었습니다. 이 연구는 LLM들이 다양한 컨텍스트 기반의 밴딧(bandit) 작업에서 탐색 및 착취를 수행하는 능력을 분석합니다. 결과적으로, LLM들은 작은 작업에서는 성과를 일정 부분 개선할 수 있으나, 여전히 간단한 선형 회귀보다 성능이 떨어짐을 발견했습니다.

- **Technical Details**: 이 연구에서 Gpt-4, Gpt-4o 및 Gpt-3.5 모델이 탐색 및 착취 작업을 수행하는 데 어떻게 도움을 줄 수 있는지 탐구하였습니다. 특히, 방법론적 접근으로는 grounded factor를 활용한 in-context learning을 통해 LLM이 최적의 행동을 선택하는 능력을 평가했습니다. 수치 통계(summary statistics)와 같은 방법으로 세밀한 개선이 가능하였지만, 이는 복잡한 의사결정 작업에는 제한적입니다.

- **Performance Highlights**: LLMs는 작은 규모의 문제에서 착취 효율성을 어느 정도 보여주었으나, 문제의 크기가 중간 이상으로 증가하면 성능이 저하되는 경향을 보였습니다. 반면, 높은 차원에서의 행동 공간 탐색에 있어 LLMs는 적합한 후보군을 제안함으로써 효과적인 탐색을 지원했습니다. 대규모 밴딧 작업에서도 유사한 결과를 얻어, 성과는 만족스럽지만 여전히 전통적인 방법에 비해 부족하다는 결론을 내렸습니다.



### Fantastic Multi-Task Gradient Updates and How to Find Them In a Con (https://arxiv.org/abs/2502.00217)
Comments:
          16 pages, 7 figures, 5 tables

- **What's New**: 최신 연구에서 ConicGrad라는 다중 작업 최적화 프레임워크를 제안하여 기존 방법의 주요 한계를 해결하고자 합니다. 이 방법은 각 작업의 경량화된 Gradient를 효과적으로 조정하기 위한 각도 제약을 도입하였습니다. 이를 통해 ConicGrad는 최적화 목표와의 정합성을 유지하면서도 기존 방법보다 동적이고 유연한 업데이트 방향을 제공합니다. 또한, 기존 방법들에 비해 계산 효율성을 높이고 학습 속도를 가속화하는 성능 개선을 보여주고 있습니다.

- **Technical Details**: ConicGrad는 다중 작업 모델을 θ∈ℝM으로 매개변수화하고 K≥2의 작업 수를 갖습니다. 각 작업의 목표 함수는 ℒi(θ)이며, 전통적인 목표는 모든 손실의 균일 평균을 최적화하는 것입니다. ConicGrad는 이런 전통적인 접근의 한계를 극복하기 위해 각 작업 손실의 감소율을 극대화하는 대체 업데이트 벡터 d를 찾고자 합니다. 이는 Angular Constraint를 활용하여 효율적인 경량화된 Gradient 업데이트를 보장합니다.

- **Performance Highlights**: 다양한 표준 감독 학습 및 강화 학습 MTL 벤치마크에서 ConicGrad의 성능을 평가한 결과, 여러 작업에 걸쳐 최첨단 성과를 달성하였습니다. ConicGrad는 이전 방법들보다 더 빠른 수렴 속도를 보이며, 이론적 수렴 보장을 통해 이러한 경험적 결과를 뒷받침합니다. 실험 결과는 높은 차원의 파라미터 공간에서도 효율성과 확장성을 유지함을 보여줍니다.



### Understanding Why Adam Outperforms SGD: Gradient Heterogeneity in Transformers (https://arxiv.org/abs/2502.00213)
- **What's New**: 이 연구는 Transformer 모델의 최적화 문제를 조사하여 Adam과 SGD 간의 성능 차이를 다룹니다. 기존 연구에서의 제한점들을 지적하며, gradient heterogeneity(그래디언트 이질성)가 이러한 성능 격차의 주요 원인임을 제안합니다. 또한, layer normalization(레이어 정규화)의 위치가 그래디언트 이질성에 미치는 영향을 분석하고, SignSGD의 모멘텀 항이 여러 클래스가 있을 때 linear-head parameters(선형 헤드 매개변수)의 과도한 성장을 예방하는 데 중요하다는 사실을 밝혀냈습니다.

- **Technical Details**: 연구에서는 gradient heterogeneity를 정의하고, SGD와 비교했을 때 SignSGD가 더 저항력이 있음을 보여줍니다. 특히, 그래디언트를 변수로 하는 순서와 사인 기반 알고리즘의 복잡도 상한을 도출했습니다. 또한 layer normalization의 위치와 구조 설계가 그래디언트 이질성에 미치는 영향을 탐구하였고, 이로 인해 최적화의 성능이 향상될 수 있다는 점을 강조했습니다.

- **Performance Highlights**: NLP와 비전 분야에서 Transformer 모델의 미세 조정 결과를 통해 이론적 분석을 검증했습니다. 경험적 결과는 SignSGD와 Adam 간의 성능 차이를 줄일 수 있는 방법을 제시하며, future optimization algorithms(미래 최적화 알고리즘) 설계에 대한 통찰력을 제공합니다. 연구 결과, Adam의 적응적 특성이 SGD보다 더 유리한 성능을 가지는 이유를 더욱 명확히 하였습니다.



### STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving (https://arxiv.org/abs/2502.00212)
Comments:
          22 pages, 5 figures

- **What's New**: 이 논문에서는 Self-play Theorem Prover (STP)를 설계하여 LLM이 서로 대화하며 추측자(conjecturer)와 증명자(prover)의 역할을 동시에 수행하도록 하였습니다. 이 접근 방식은 수학자들이 새 결과를 발전시키는 방식에서 영감을 얻었으며, 새로운 추측을 생성하고 기존 데이터를 통해 증명하는 과정을 반복하여 제한된 데이터로도 성능 향상을 도모합니다. 특히, 이 방법은 기존의 expert iteration의 한계를 극복하고 자가 학습할 수 있는 알고리즘을 제안합니다.

- **Technical Details**: STP는 각 iteration마다 증명하기 어려운 추측을 바탕으로 훈련되어, 점진적으로 더 어려운 추측을 만들어냅니다. 기존의 proof samples에 비해 STP는 19.8억 개의 토큰을 생성하여 LeanWorkbook 데이터셋에서 성공적으로 26.3%의 명제를 증명하였습니다. 이러한 성과는 traditional RL 방법이나 expert iteration의 한계를 극복한 것이며, 결과적으로 miniF2F-test, Proofnet-test, PutnamBench에서 최신 성능을 기록했습니다.

- **Performance Highlights**: STP의 성능은 DeepSeek-Prover-V1.5 모델을 초월하여 다양한 샘플링 예산에서 우수한 결과를 보였습니다. 특히 miniF2F-test에서 61.1%의 성공률, ProofNet-test에서 23.1%, 그리고 PutnamBench에서 8/644의 성공률을 달성하여 전체 증명 생성 방식 중에서 최고 성능을 나타냈습니다. 이러한 결과들은 STP가 기존 방법들과 비교할 때 명확히 더 나은 확장성(scaling behavior)을 가지고 있음을 보여줍니다.



### EcoWeedNet: A Lightweight and Automated Weed Detection Method for Sustainable Next-Generation Agricultural Consumer Electronics (https://arxiv.org/abs/2502.00205)
- **What's New**: 본 연구에서는 EcoWeedNet이라는 새로운 모델을 제안하여 잡초 탐지 성능을 획기적으로 향상시켰습니다. 이 모델은 복잡한 계산 요구 사항을 추가하지 않고도 로우카본(low-carbon) 농업 실천의 목표에 부합하는 경량화된 솔루션입니다. EcoWeedNet은 기존의 큰 모델에 비해 필요한 파라미터 수가 약 4.21%에 불과하며, 이는 자동 잡초 탐지 방법의 효과적인 개발에 기여하고 차세대 농업 소비 전자 기기에서의 응용 가능성을 높입니다.

- **Technical Details**: 에코잡초넷(EcoWeedNet)은 깊이 학습 모델에서 전통적인 주의(attention) 모듈의 단점을 해결하고 파라미터가 없는 주의 모듈을 도입함으로써 계산 효율성을 유지하면서 성능을 최적화합니다. 이 연구에서는 CottonWeedDet12 벤치마크 데이터셋을 사용하여 실제 환경에서 성능을 테스트해 효율적인 잡초 탐지 능력을 입증하였습니다. 이 모델은 CNN(convolutional neural networks)을 활용하여 고해상도 이미지를 분석하며, 이미지 내 물체 탐지 문제를 효과적으로 정의하는 방식으로 작동합니다.

- **Performance Highlights**: EcoWeedNet는 YOLOv4의 약 6.59%의 GFLOPs로 대규모 모델에 가까운 성능을 보이며, 정확도에서 우수한 결과를 나타냅니다. 연구 결과에 따르면, 제안된 모델은 가벼우면서도 높은 검출 정확도를 제공하여, 지속 가능한 소비자 전자 농업 장비 및 로봇에 적합한 성능을 ₍를₉ 증명하였습니다. 이러한 성과는 차세대 농업 소비 기술의 지속 가능성을 위한 중요한 발판을 마련합니다.



### Year-over-Year Developments in Financial Fraud Detection via Deep Learning: A Systematic Literature Review (https://arxiv.org/abs/2502.00201)
- **What's New**: 이 논문은 금융 사기 탐지를 위한 딥러닝 기술의 발전을 체계적으로 검토합니다. 2019년부터 2024년까지 발표된 57개의 연구를 분석하여 다양한 딥러닝 모델의 효과성을 강조하고, 데이터 프라이버시와 기능 공학, 데이터 전처리의 발전도 다룹니다. 또한, 불균형 데이터셋, 모델 해석 가능성, 윤리적 고려사항 등의 도전과제와 자동화 및 블록체인 통합 등의 기회를 강조합니다.

- **Technical Details**: 이 연구는 Kitchenham 체계적 문헌 검토 접근법을 채택하여 연구 선정, 데이터 추출 및 합성을 맞춤형으로 처리했습니다. 주요 질문으로는 딥러닝을 활용한 금융 사기 탐지의 최신 트렌드, 특성 공학의 발전, 데이터 전처리 기술의 개선을 포함하였습니다. Python과 Pandas, Matplotlib, Scikit-learn, VOSviewer와 같은 라이브러리를 이용하여 데이터 분석 및 시각화가 이루어졌습니다.

- **Performance Highlights**: 연구 결과, 금융 사기 탐지 관련 논문 수는 2019년부터 2024년까지 증가 추세를 보였으며, 특히 2022년부터 급격한 상승이 나타났습니다. 신용 카드 거래 및 은행 분야는 이러한 증가의 주된 원인이었으며, 데이터셋 불균형 문제를 해결하기 위한 다양한 전처리 및 자동화 기술이 소개되었습니다. 궁극적으로 이러한 연구는 금융 사기 탐지의 향후 연구 방향에 대한 통찰을 제공합니다.



### DermaSynth: Rich Synthetic Image-Text Pairs Using Open Access Dermatology Datasets (https://arxiv.org/abs/2502.00196)
Comments:
          12 pages, 4 figures

- **What's New**: 이번 연구에서는 각종 피부과 임상 과제를 위해 92,020개의 합성 이미지-텍스트 쌍으로 구성된 새로운 데이터셋인 DermaSynth를 소개합니다. 이 데이터셋은 피부과 관련 이미지와 함께 제공되는 텍스트 어노테이션의 부족 문제를 해결하기 위해 개발되었습니다. 임상 관련 프롬프트와 자기 지도(self-instruct) 방법을 활용해 데이터셋을 생성하였으며, 이는 AI 연구에 기여할 것으로 기대됩니다.

- **Technical Details**: DermaSynth 데이터셋은 DERM12345, BCN20000, PAD-UFES-20 등 여러 공개 피부과 데이터셋에서 수집된 이미지들을 바탕으로 합니다. 각 이미지에 대해 일반 질문과 메타데이터 기반 질문을 통해 합성 이미지-텍스트 쌍을 생성하였고, 이 과정에서는 Gemini 2.0 API를 이용했습니다. 이렇게 생성된 데이터는 후처리 과정을 거쳐 임상적으로 관련성이 높고 일관된 스타일을 유지하도록 하였습니다.

- **Performance Highlights**: 프로토타입 모델 DermatoLlama 1.0은 Llama-3.2-11B-Vision-Instruct 모델로, DERM12345 데이터셋에서 5,000개의 이미지-텍스트 쌍을 Fine-tuning하여 개발되었습니다. 이 모델은 Hugging Face에서 접근 가능하며, 연구자들이 피부과 영상을 기반으로 한 LLM의 성능을 평가하는 데 활용될 수 있습니다. DermaSynth와 DermatoLlama 1.0은 모두 비상업적 연구 목적으로 제공되며, 해당 리소스들은 오픈 소스로 제공됩니다.



### Physics-Informed Neural Network based Damage Identification for Truss Railroad Bridges (https://arxiv.org/abs/2502.00194)
Comments:
          30 pages, 15 figures

- **What's New**: 이번 연구에서는 미국 화물 철도 시스템의 중요한 구성 요소인 철도 교량의 손상 식별을 위한 새로운 접근법을 제안합니다. 다리의 노후화와 증가하는 기차 통행량이 안전에 큰 위협이 되고 있으며, 이는 서비스 중단을 초래할 수 있습니다. 제안된 접근법인 Physics-Informed Neural Network(PINN)는 감독학습(supervised learning) 방식에 의존하지 않고도 손상을 효과적으로 식별할 수 있습니다.

- **Technical Details**: 이 연구에서 제안하는 PINN 모델은 레일로스 시스템의 국부 방정식과 선형 시간 변동(Linear Time-Varying, LTV) 조건을 통합하여 교량 손상 식별을 수행합니다. 이 모델은 순환 신경망(Recurrent Neural Network, RNN) 아키텍처를 기반으로 하고 있으며, 커스터마이징된 룽게-쿠타(Runge-Kutta, RK) 적분 셀을 통해 기울기 기반 학습을 지원합니다. 이를 통해 기차가 교량을 통과할 때 발생하는 휠 하중 데이터와 교량 반응을 입력으로 사용합니다.

- **Performance Highlights**: 시카고의 Calumet Bridge에 대한 사례 연구에서는 시뮬레이션된 손상 시나리오를 적용하여 모델의 손상 식별 효율성을 입증했습니다. 제안된 모델은 낮은 거짓 긍정률(false positive rate)을 유지하면서도 손상 심각성 및 영향을 받는 구조 부재를 정량화할 수 있습니다. 또한, 이 손상 식별 파이프라인은 이전 검사 및 드론 조사에서 수집된 정보를 통합하여 교량의 상태를 지속적으로 업데이트하고 평가할 수 있도록 설계되었습니다.



### Understanding Federated Learning from IID to Non-IID dataset: An Experimental Study (https://arxiv.org/abs/2502.00182)
- **What's New**: 본 논문은 분산된 데이터 소스에서 개인 정보를 보호하면서 기계 학습 모델을 훈련하기 위해 등장한 연합 학습(Federated Learning, FL)의 새로운 접근 방식을 제안합니다. FL은 중앙 집중형 방법과 달리 클라이언트가 데이터를 로컬에 저장하고 자신이 훈련한 모델만을 중앙 서버와 공유하게 합니다. 이로 인해 FL은 민감한 정보가 포함된 데이터를 다룰 때 특히 유용하지만, 클라이언트 간 데이터가 비독립적 및 비동일 분포(non-IID)일 경우 성능 저하가 발생하는 문제점이 있습니다.

- **Technical Details**: 연구에서는 기울기 하강법(Gradient Descent)에서 FL로의 진행 과정을 설명하며, 클라이언트 손실 지형의 불일치가 non-IID 시나리오에서 성능 저하의 주요 원인임을 발견했습니다. 그리고 기존의 FL 관련 방법들을 두 가지 전략으로 분류하였습니다: (i) 매개변수 업데이트 경로 조정 및 (ii) 클라이언트 손실 지형 수정. 이러한 발견은 FL에서 non-IID 문제를 해결하는 명확한 관점을 제공하고 향후 연구의 방향을 제시합니다.

- **Performance Highlights**: 논문에서는 FL 관련 기존 방법론의 포괄적 검토를 통해 기울기 하강법부터 연합 평균(FedAvg)까지의 최적화 알고리즘을 설명하였습니다. IID 환경에서의 광범위한 실험과 non-IID 환경에서의 FL 성능 분석을 통해 중요한 통찰을 얻었으며, 기존 방법들을 non-IID 문제를 다루는 두 가지 전략으로 나누어 분류하였습니다. 이러한 결과는 연합 학습의 핵심 도전 과제를 명확히 하고 향후 연구 방향을 제시하는 데 기여할 것입니다.



### A Comprehensive Review: Applicability of Deep Neural Networks in Business Decision Making and Market Prediction Investmen (https://arxiv.org/abs/2502.00151)
- **What's New**: 이 논문은 구조화된 데이터와 비구조화된 데이터를 포함한 대량의 데이터가 경제학 및 비즈니스 분야에서 발생하는 새로운 도전을 다루고 있습니다. 특히, 최근 딥 뉴럴 네트워크(deep neural networks)의 적용이 리스크 관리(risk management), 포트폴리오 최적화(portfolio optimization), 알고리즘 거래(algorithmic trading)와 같은 의사 결정 과정에 있어 장점이 있음을 보여줍니다. 다양한 데이터 유형의 모달리티를 아우르는 여러 신경망을 조합하여 더 견고하고 효율적인 금융 예측 프레임워크를 구축할 수 있음을 제안하고 있습니다.

- **Technical Details**: 딥 러닝(deep learning) 네트워크의 기본 개념을 소개하며, 세 가지 주요 학습 패러다임인 감독 학습(supervised learning), 비감독 학습(unsupervised learning), 강화 학습(reinforcement learning)에 대해 설명하고 있습니다. 각 패러다임의 특징과 성격을 시각적으로 설명하기 위해 그래프를 사용하며, 현재 경제 및 금융 분야에서 CNN과 RNN 구성이 주로 활용되고 있음을 강조합니다. 또한, FNN, CNN, RNN의 구조와 활용 예시를 제시하며 기능 추출 및 데이터 처리의 중요성을 설명합니다.

- **Performance Highlights**: 경제 및 비즈니스 분야에서 딥 러닝 모델을 활용한 다양한 연구가 진행되었으며, 특히 2014년부터 2021년까지 발표된 460개의 논문 중 260개의 논문이 리스크 관리, 포트폴리오 최적화 및 거래에 중점을 두고 있습니다. 신경망 모델을 활용한 리스크 평가 및 사기 탐지 분야의 연구가 주목받고 있으며, 이러한 모델이 데이터 분석을 통해 의사 결정자에게 중요한 인사이트를 제공할 수 있습니다. 예를 들어, 신용 점수 분류 과정에서 소비자 지출 데이터를 2D 픽셀 매트릭스로 변형하여 신경망 모델을 사용한 성공 사례를 제공합니다.



### Multimodal MRI-Ultrasound AI for Prostate Cancer Detection Outperforms Radiologist MRI Interpretation: A Multi-Center Study (https://arxiv.org/abs/2502.00146)
- **What's New**: 이번 연구는 전립선 생검에서 의심되는 병변을 목표로 하는 전처리 자기공명영상(MRI)의 사용이 증가함에 따라 인공지능(AI) 응용 프로그램이 임상적으로 중요한 전립선암(CsPCa) 탐지를 개선할 수 있는 가능성을 보여주고 있습니다. 특히, 이 연구는 MRI와 직장 초음파(TRUS) 이미지를 통합한 다중 모달 AI 프레임워크를 제안하여 CsPCa 식별을 향상시키기 위한 체계적인 평가를 진행하였습니다.

- **Technical Details**: 이 연구는 두 개의 기관에서 3110명의 환자를 대상으로 전립선 생검을 수행하였고, 1700개의 테스트 사례에 대해 3D UNet 아키텍처에 기반한 제안된 프레임워크의 성능을 평가했습니다. 이 때, 단일 모달(MRI 또는 TRUS만 사용하는) AI 모델과의 성능 비교를 통해 다중 모달 AI 접근법의 우수성을 검증하였습니다.

- **Performance Highlights**: 다중 모달 AI 접근법은 단일 모달 MRI(73%, 30%) 및 TRUS 모델(49%, 27%)에 비해 더 높은 민감도(80%)와 병변 Dice(42%)를 기록했습니다. 방사선 의사와의 비교에서도 다중 모달 모델은 높은 특이성(88% vs. 78%)과 병변 Dice(38% vs. 33%)를 보였으며, 민감도는 동등한 수준(79%)을 유지했습니다. 이러한 결과는 생검 및 치료 계획 과정에서 CsPCa 병변을 정확하게 목표로 할 수 있는 다중 모달 AI의 잠재력을 입증합니다.



### Demystifying MPNNs: Message Passing as Merely Efficient Matrix Multiplication (https://arxiv.org/abs/2502.00140)
- **What's New**: 이 논문은 그래프 신경망(Graphic Neural Networks, GNN)의 행동을 이해하기 위한 포괄적인 분석을 제시합니다. 기존 GNN의 설계가 경험적 직관에 의존하고 있는 것을 개선하기 위해, 세 가지 근본적인 측면을 통해 GNN의 성능을 평가합니다. 이를 통해 GNN이 다양한 작업에서 보여주는 성능의 차이를 이론적으로 설명합니다.

- **Technical Details**: 저자들은 먼저 	extbf{$k$-layer} 메시지 전달 신경망이 반복적인 계산을 통해 	extbf{$k$-hop} 이웃 정보를 효율적으로 집계한다는 것을 입증하였습니다. 이어서 다양한 루프 구조가 이웃 계산에 미치는 영향을 분석하고, 구조-특징 혼합형(task)과 구조만의 작업 간의 행동을 조사합니다. 또한, 깊은 GNN에서 그래디언트 관련 문제들이 드문 그래프에서 성능에 큰 영향을 미친다는 것을 입증합니다.

- **Performance Highlights**: 논문에서는 서로 다른 정규화 기법이 모델 성능에 미치는 영향을 분석하고, 균일한 노드 특징을 사용하여 GNN이 예측을 어떻게 수행하는지를 설명합니다. 이러한 이론적 틀은 GNN이 경험적으로 성공한 이유와 그 기초가 되는 이론적 원칙 간의 간극을 좁히는데 기여합니다.



### A Three-Branch Checks-and-Balances Frameworkfor Context-Aware Ethical Alignment of Large Language Models (https://arxiv.org/abs/2502.00136)
Comments:
          17 pages, 6 tables, 6 figures. arXiv admin note: substantial text overlap with arXiv:2405.07076

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 윤리적 정렬을 위한 삼원 체크 앤 밸런스(framework)를 제안합니다. 정부 시스템에서 영감을 받아 세 가지 독립적이면서 상호 작용하는 요소들, 즉 지식 생성을 위한 실행(branch), 윤리적 가드레일을 설정하는 입법(branch), 그리고 맥락 해석을 위한 사법(branch)을 구현하고 있습니다.

- **Technical Details**: 이 프레임워크에서는 DIKE라는 입법 부문과 ERIS라는 사법 부문이 상호작용하며 적대적(dual) 관계를 형성하여 다양한 문화적 맥락에 맞춰 적응할 수 있도록 합니다. 이 구조는 인간 피드백을 통한 강화 학습(RLHF)의 한계를 극복하며 해석 가능하고 적응적이며 문화적 인식을 고려한 윤리적 추론을 제공합니다.

- **Performance Highlights**: 자기 지도 학습(self-supervised learning)과 적대적 테스트(adversarial testing)를 통해 프레임워크가 감정 모델링(emotional modeling)을 활용해 언어 행동을 윤리적 결과로 유도할 수 있는지를 보여주며, 지식 생성, 윤리적 감독, 맥락 해석 간의 독립성을 유지합니다.



### Exploring Transfer Learning for Deep Learning Polyp Detection in Colonoscopy Images Using YOLOv8 (https://arxiv.org/abs/2502.00133)
Comments:
          10 pages, 3 figures, 6 tables, SPIE conference

- **What's New**: 이 논문은 전이 학습(transfer learning) 기술을 활용하여 YOLOv8n 모델을 폴립 탐지(polyp detection) 작업에 적용하는 방법을 탐구합니다. 기존의 데이터셋에 대한 전이 학습을 통해 폴립 탐지 성능을 향상시키는 방법을 제시하며, 다양한 데이터셋에서의 전이 학습 효과를 비교하고 분석합니다. 특히, 일반적인 대규모 데이터셋과 폴립과 같은 특정 특성을 가진 소규모 데이터셋 간의 성능 차이를 조사했습니다.

- **Technical Details**: 전이 학습은 특정 작업을 위해 사전 훈련된 모델을 사용하는 기술로, 충분한 데이터가 없거나 비용이 많이 드는 의료 분야에서 특히 유용합니다. 이 연구에서는 YOLOv8n 모델을 여러 개의 데이터셋에서 사전 훈련하고, 이를 폴립 탐지 데이터셋에 맞게 미세 조정(fine-tuning)하여 성능을 평가했습니다. 각 모델은 무작위로 초기화된 모델과 성능을 비교하여 사전 훈련의 이점을 측정했습니다.

- **Performance Highlights**: CVC-Clinic DB, CVC-ColonDB, ETIS-LaribPolypDB, Kvasir-SEG의 네 가지 공공 데이터셋에서 이루어진 실험은 관련 데이터셋에서 사전 훈련된 모델이 일반적인 객체 데이터셋에서 사전 훈련된 모델보다 일관되게 높은 성능을 나타낸다는 결과를 보였습니다. 특히, 사전 훈련이 없는 모델에 비해 사전 훈련된 모델이 더 우수한 결과를 보였고, 이는 폴립 탐지와 같은 제한된 데이터 환경에서 전이 학습의 중요성을 강조합니다.



### AIN: The Arabic INclusive Large Multimodal Mod (https://arxiv.org/abs/2502.00094)
Comments:
          20 pages, 16 figures, ACL

- **What's New**: 최근의 큰 언어 모델(LLMs)과 다중 모달 모델(LMMs)의 발전 속에서, 아랍어 LMM들은 주목받지 못한 반면, 아랍어 LLMs는 상당한 향상을 보여 왔습니다. 이 격차를 해소하기 위해 AIN(Arabic Inclusive Multimodal Model)을 소개합니다. AIN은 영어-아랍어 이중 언어 LMM으로, 고품질의 360만 개 아랍어-영어 다중 모달 데이터 샘플을 활용하여 설계되었습니다.

- **Technical Details**: AIN 모델은 70억 개의 파라미터를 기반으로 한 아랍어 대형 다중 모달 모델로, 복잡한 추론, 다국적 작업, 이미지-텍스트 정렬에서 우수한 성능을 보입니다. CAMEL-Bench 기준에서 대조군 모델들과 비교하여 AIN-7B는 많은 도메인에서 높은 성과를 자랑하며, Qwen2-VL-7B와 비교해도 성능이 3.43% 향상되었습니다.

- **Performance Highlights**: AIN의 성과는 특히 의료 이미지 해석 및 과학 시각화 이해 등 다양한 분야에서 두드러집니다. 설문 조사 결과, 아랍어 구사자들 사이에서 AIN-7B가 76%의 지지를 받아 대세 모델이 되었으며, 복잡한 시각-언어 과제를 처리하는 데 있어 AIN의 효율성과 정확도가 두드러집니다.



### Ensembles of Low-Rank Expert Adapters (https://arxiv.org/abs/2502.00089)
Comments:
          29 pages, 5 figures, 5 tables; proceedings in ICLR 2025

- **What's New**: 본 논문에서는 다양한 텍스트 데이터로부터 발생하는 gradient 방향의 충돌 문제를 해결하기 위해 Ensembles of Low-Rank Expert Adapters(ELREA)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 다중 미세 조정 작업에서 모델의 전문성을 높이고, 특정 작업에 대한 데이터의 중요성을 활용하여 우수한 성능을 달성할 수 있도록 설계되었습니다. ELREA는 Low-Rank Adaptation(LoRA) 기술을 사용하여 훈련 명령을 군집화하고, 이를 통해 모델의 최적화 과정을 간소화하며, 예측 과정 중에 가장 관련성이 높은 전문가 어댑터를 결합합니다.

- **Technical Details**: ELREA 프레임워크는 기본 어댑터를 전체 데이터셋에 대해 미세 조정하여 일반적인 지식을 습득한 후, 각 데이터 포인트의 gradient를 평가하여 군집화합니다. 이후, 각 군집에 대해 LoRA 전문가 어댑터를 훈련시키고, 이러한 전문가 어댑터는 입력 데이터의 gradient 유사성에 근거하여 예측 결과를 조합하는 방식으로 작동합니다. 이를 통해 ELREA는 기존 Deep Ensembles 기법보다 더 효율적이며, 추가적인 작업별 유효성 검증 데이터 없이도 작업 분류가 가능합니다.

- **Performance Highlights**: 실험 결과, ELREA는 전체 데이터셋에 대해 훈련된 baseline LoRA 어댑터 및 다른 Mixture of Experts(MoE) 기법을 능가하는 성능을 보였습니다. 다양한 도메인 특화 작업에 걸쳐 강력한 성능을 발휘하며, 모델의 확장성과 효율성을 동시에 유지할 수 있어 실제 애플리케이션에서 매우 유용한 선택이 될 수 있습니다. 본 논문에서 제안한 방법은 복잡한 자동화 작업을 처리하는 데 필수적인 모델 일반화 문제를 효과적으로 해결하는 데 기여합니다.



### Influence of color correction on pathology detection in Capsule Endoscopy (https://arxiv.org/abs/2502.00076)
- **What's New**: 이 연구는 Wireless Capsule Endoscopy (WCE)에서 병리 감지에 영향을 미치는 색 보정의 효과를 평가합니다. 기존의 SEE-AI 데이터셋을 바탕으로 두 가지 색 보정 버전을 생성하여 Retinanet과 YOLOv5 모델의 성능을 평가하였습니다. 연구 결과, 색 보정은 모델이 더 큰 바운딩 박스를 생성하고, 특정 병리에 대한 오탐지(False positive) 수를 증가시키는 경향이 있지만, F1-score 및 IoU와 같은 성능 지표에서 일관된 개선이 나타나지 않았습니다.

- **Technical Details**: WCE 데이터셋에서 병리 감지 정확도를 높이기 위해 색 보정 기술을 적용하였습니다. 본 연구는 CC (Color Correction)와 CCC (Colon Color Checker) 두 가지 색 보정 행렬을 활용하여 두 개의 색 보정 데이터셋을 생성하였습니다. 이후, 세 가지 데이터셋(원본 SEE-AI, CCD, CCCD)에 대해 Retinanet과 YOLOv5 모델의 성능을 비교 분석하였습니다.

- **Performance Highlights**: 색 보정 후 모델의 결과는 바운딩 박스의 크기와 실제 주석 간의 교차 영역을 확대하며, 특정 병리에 대해 오탐지 수가 증가하는 것으로 나타났습니다. 하지만, 이러한 변화가 성능 지표의 일관된 개선으로 이어지지는 않았고, 색상 왜곡 문제를 해결하기 위해 향후 연구가 필요하다는 결론을 내렸습니다. 이 연구 결과는 WCE 이미지를 통한 병리 검출의 향상을 도모하며, 관련 데이터셋은 연구자들에게 공개될 예정입니다.



### SpikingRTNH: Spiking Neural Network for 4D Radar Object Detection (https://arxiv.org/abs/2502.00074)
Comments:
          arxiv preprint

- **What's New**: 최근 4D Radar 기술이 자율주행 차량의 3D 객체 탐지에 필수적인 센서로 자리잡고 있습니다. 이 논문에서는 SpikingRTNH라는 최초의 스파이킹 신경망(SNN) 아키텍처를 제안하여 4D Radar 데이터를 이용한 3D 객체 탐지를 수행합니다. 이 접근법은 conventional ReLU 활성화 함수를 leaky integrate-and-fire (LIF) 스파이킹 뉴런으로 대체하여 에너지 효율성을 극대화합니다.

- **Technical Details**: SpikingRTNH는 포인트 클라우드를 높은 밀도에서 낮은 밀도로 순차적으로 처리하는 생물학적 top-down inference (BTI) 방법을 도입합니다. 이 방식은 노이즈가 적고 중요도가 높은 포인트를 효과적으로 감지하는 데 초점을 둡니다. RTNH에서 높은 밀도의 4D Radar 포인트 클라우드를 처리하면서도 약 156G의 곱셈-누적(MAC) 연산을 요구하는 기존 방법들에 비해 현저한 에너지 절감이 가능합니다.

- **Performance Highlights**: K-Radar 데이터셋에서 수행된 실험 결과 SpikingRTNH는 78% 에너지 소비를 줄이면서도 51.1% AP 3D 및 57.0% AP BEV로 기존 인공신경망(ANN) 모델과 유사한 탐지 성능을 달성했습니다. 이러한 결과는 자율주행 시스템을 위한 에너지 효율적인 4D Radar 기반 객체 탐지의 가능성을 입증하고 있습니다. 연구에 사용된 모든 코드는 제공된 링크에서 확인할 수 있습니다.



### LLM Cyber Evaluations Don't Capture Real-World Risk (https://arxiv.org/abs/2502.00072)
Comments:
          11 pages

- **What's New**: 대형 언어 모델(LLMs)은 사이버 보안 애플리케이션에서 점점 더 많은 잠재력을 보여주고 있으며, 이는 방어를 강화할 수 있는 가능성과 함께 고유한 위험을 초래합니다. 이 논문에서는 LLM의 위험 평가 노력이 실제 세계의 영향을 이해하는 데 잘못 맞춰져 있다고 주장합니다. LLM의 사이버 능력에 대한 위험 평가 접근 방식을 제안하며, 이를 사이버 보조 도구로 사용된 언어 모델을 사례 연구로 적용합니다.

- **Technical Details**: LLMs의 사이버 능력 위험 평가에는 복잡한 분석이 요구됩니다. 이 논문은 기존의 위험 평가 방안이 LLM의 기술적 능력 분석에만 국한되어 있다고 비판하며, 위협 행위자의 행동 및 잠재적 영향을 포함한 종합적인 위험 평가 프레임워크를 제안합니다. 이를 통해 실제 공격 시나리오에서 LLM의 활용 가능성과 제약을 분석하고, 이로 인해 발생하는 잠재적 피해를 연구합니다.

- **Performance Highlights**: 분석 결과, LLM 모델들은 높은 준수 비율을 나타내지만, 실제 사이버 보조 작업에서는 중간 정도의 정확도를 보이고 있습니다. 연구 결과에 따르면, 특정 사용 사례에서 운영상의 이점과 영향 가능성이 제한되어 있는 만큼, LLM의 사이버 보안 능력으로 인한 위험은 중간 수준에 불과합니다. 마지막으로, 연구 우선순위를 실제 영향 평가와 일치시키기 위한 몇 가지 개선 사항을 제안하고 있으며, 이는 보다 효과적인 LLM 기반 사이버 보안 위험 평가와 완화로 나아가는 중요한 단계로 풀이됩니다.



### Can AI Solve the Peer Review Crisis? A Large Scale Experiment on LLM's Performance and Biases in Evaluating Economics Papers (https://arxiv.org/abs/2502.00070)
Comments:
          72 pages

- **What's New**: 이 연구는 인공지능(Artificial Intelligence, AI)이 경제학 분야의 동료 평가(peer review) 위기를 해결할 수 있는지를 조사합니다. 27,090개의 평가를 분석함으로써 9,030개의 독창적인 제출물에 대한 연구를 진행하였습니다. 주요 저자 특성과 출판 품질을 체계적으로 변경하며 실험을 진행한 점이 주목받습니다.

- **Technical Details**: 연구는 저자 특성(affiliation, reputation, gender) 및 출판 품질(top-tier, mid-tier, low-tier, AI generated papers)을 다각도로 조정하여 대형 언어 모델(Large Language Model, LLM)의 성능을 평가합니다. 결과적으로 LLM은 논문 품질을 효과적으로 구별하지만, 저명한 기관과 남성 저자 및 저명한 경제학자에 유리한 편견(bias)을 보였습니다.

- **Performance Highlights**: LLMs는 품질 높은 AI 생성 논문을 실제 상위 논문과 구별하는 데 어려움을 겪습니다. 이러한 연구는 LLM들이 효율성을 제공할 수 있지만, 편견 때문에 조심스럽게 통합할 필요가 있음을 강조합니다. 공정성과 정확성을 균형 있게 유지하기 위해 혼합 성격의 동료 평가 모델(hybrid peer review models)의 필요성이 제기됩니다.



### Privacy Preserving Charge Location Prediction for Electric Vehicles (https://arxiv.org/abs/2502.00068)
Comments:
          12 pages, 7 figures, IEEE Journal paper

- **What's New**: 이 논문에서는 전기차 (EV)의 충전 위치 예측을 위한 연합 학습 변환기 네트워크 (Federated Learning Transformer Network, FLTN)를 개발했습니다. FLTN은 모든 전기차가 클라이언트로 작동하여 로컬 모델을 학습하고, 모델의 가중치만을 공유하여 데이터 개인 정보를 보호합니다. 이를 통해 사용자 데이터의 유출 위험을 효과적으로 줄입니다.

- **Technical Details**: FLTN은 비전이전 (peer-to-peer) 가중치 공유 및 보강 (augmentation) 메커니즘을 활용하여 EV의 이동성을 개인정보 보호에 활용합니다. 각 EV는 지역 DERMS와의 연결을 통해 로컬 모델 가중치를 수집하고, 비전이전 EV는 이 가중치를 서로 교환하여 개별 모델 업데이트의 기원을 모호하게 만듭니다. 이 과정에서 딥러닝 모델을 강화하여 개인 정보 보호와 동시에 예측 정확도를 높입니다.

- **Performance Highlights**: 실험 결과, FLTN은 92%의 정확도로 EV의 다음 충전 위치를 예측할 수 있었습니다. 이 모델은 최소 100대에서 150대의 EV에 의해 학습되어야 하며, 3일 이내에 다음 충전 위치를 예측할 수 있는 성능을 보입니다. 본 연구는 개인 정보 보호를 유지하면서 실시간 예측 및 대규모 적용이 가능한 솔루션을 제공합니다.



### A Multi-Layered Large Language Model Framework for Disease Prediction (https://arxiv.org/abs/2502.00063)
- **What's New**: 이 연구는 소셜 원격의료가 의료 분야에서 어떻게 혁신을 가져왔는지를 다룹니다. 특히, COVID-19 팬데믹 동안 사용자-generated 데이터가 의료 거점으로 활용되는 사례를 제시합니다. 본 연구는 LLM(대형 언어 모델)을 활용하여 아랍어 의료 데이터의 전처리 단계에서 효율성을 증대시키고자 하는 새로운 프레임워크를 제안합니다.

- **Technical Details**: 연구에서는 LLAMA3, GPT-3.5 Turbo, BERT 등의 LLM을 사용하여 아랍어 의료 텍스트를 처리하고, 주로 텍스트 요약(text summarization), 텍스트 개량(text refinement), 그리고 NER(Named Entity Recognition) 기법을 적용합니다. 이를 통해 CAMeL-BERT, AraBERT, Asafaya-BERT 모델을 파인튜닝하여 질병 분류 및 증상 심각도 평가의 정확도를 향상시켰습니다.

- **Performance Highlights**: CAMeL-BERT 모델은 NER이 보강된 텍스트를 사용하여 83%의 질병 유형 분류 및 69%의 심각도 평가 성과를 달성했습니다. 반면, 미세조정되지 않은 모델들은 13-20%의 유형 분류와 40-49%의 심각도 평가 성과로 낮은 성능을 보였습니다. 이러한 결과는 LLM을 통합한 원격의료 시스템이 진단 정확도 및 치료 성과를 크게 향상시킬 수 있음을 시사합니다.



### From Data to Action: Charting A Data-Driven Path to Combat Antimicrobial Resistanc (https://arxiv.org/abs/2502.00061)
Comments:
          29 pages, 3 figures, 4 tables, survey paper

- **What's New**: 최근 AMR(항생제 내성) 연구는 데이터 분석 및 기계 학습 관점에서의 접근이 중요해지고 있습니다. 본 논문에서는 AMR의 원인 및 치료법을 데이터 기반 방법을 통해 탐구하며, 감시(surveillance), 예측(prediction), 약물 발견(drug discovery), 자원 관리(stewardship)와 같은 주요 분야를 다룹니다. 특히, 데이터 출처, 방법 및 문제를 논의하며, 표준화 및 상호 운용성(interoperability)의 중요성도 강조합니다.

- **Technical Details**: AMR 연구에서 기계 학습 기술은 데이터 노이즈(noise) 및 편향(bias) 문제를 해결하기 위한 데 사용됩니다. 이러한 문제들을 완화하는 전략이 소개되어 있으며, 이는 데이터 수집, 처리 및 모델 개발 단계에서 신뢰성과 정확성을 보장하는 데 필수적입니다. 또한, 논문에서는 개인 식별 정보를 포함한 민감한 정보를 다루는 과정에서의 프라이버시 보존 기술에 대한 논의도 포함되어 있습니다.

- **Performance Highlights**: AMR 예측은 특정 미생물이 특정 항생제에 저항할 가능성을 예측하는 데 중점을 두고 있습니다. 머신 러닝 기법들은 전자 건강 기록(EHRs)과 같은 다양한 데이터 소스를 활용하여 예측 모델을 개발하는 데 효과적입니다. AMR 연구 분야에서 이러한 데이터 분석 기법들은 효율적이고 혁신적인 접근 방식으로, AMR 문제를 해결하는 데 기여하고 있습니다.



### Israel-Hamas war through Telegram, Reddit and Twitter (https://arxiv.org/abs/2502.00060)
- **What's New**: 2023년 10월 7일에 시작된 이스라엘-팔레스타인 갈등은 현재까지 48,000명 이상의 사망자를 초래했으며, 그 중 17,000명이 어린이입니다. 또한 1백만 명 이상이 강제로 이주하였고, 인프라의 87%가 파손되었습니다. 본 연구는 Telegram과 Twitter, Reddit에서의 논의 분석 방법을 통해 갈등에 대한 감정과 관련된 데이터셋을 준비했습니다.

- **Technical Details**: 연구는 2025년 10월 23일까지의 기간 동안 Telegram 채널에서 수집된 125K 메시지와 함께 Twitter의 2001트윗 및 Reddit의 200만 의견으로 구성된 두 개의 공개 데이터셋을 포함합니다. Latent Dirichlet Allocation (LDA)와 BERTopic 같은 분석 기법을 통해 주제 분석을 수행했고, 감정 분석을 통해 논의된 내용의 정서적 톤을 분석했습니다. 이를 통해 정치적 분열과 여론 조작의 선례를 발견했습니다.

- **Performance Highlights**: 본 연구는 이스라엘-팔레스타인 갈등에 대한 온라인 담론에 대한 통찰을 제공합니다. Telegram에서의 대화는 지지, 연대 및 불만 등의 감정을 반영하고 있으며 이로 인해 복잡한 동학이 나타납니다. 이 연구 결과는 정책 입안자와 소셜 미디어 플랫폼이 허위 정보, 선전, 건설적인 대화의 필요성 등을 관리하는 데 도움이 될 것입니다.



### Large Language Models are Few-shot Multivariate Time Series Classifiers (https://arxiv.org/abs/2502.00059)
- **What's New**: 이 논문에서는 Multivariate Time Series Classification (MTSC) 분야에 대한 새로운 접근법인 LLMFew를 제안합니다. LLMFew는 다변량 시계열 데이터의 classification을 위해 사전 학습된 Large Language Models (LLMs)의 지식을 활용하여 데이터 부족 문제를 극복하고자 합니다. 이 모델은 Patch-wise Temporal Convolution Encoder (PTCEnc)를 사용하여 시계열 데이터를 LLM의 텍스트 임베딩과 정렬시킵니다.

- **Technical Details**: LLMFew는 LLM의 decoder를 Low-rank Adaptations (LoRA)으로 미세 조정하여 시계열 데이터의 feature representation learning 능력을 향상시킵니다. 특히, LLMFew는 적은 수의 학습 데이터로 시계열 패턴을 학습할 수 있는 능력을 갖추고 있으며, MLP 기반의 classification head를 통해 가능한 클래스를 예측합니다. 이 접근법은 LLM의 사전 학습된 지식을 활용하여 소스 도메인 데이터셋을 거의 필요로 하지 않습니다.

- **Performance Highlights**: 실험 결과, LLMFew는 Handwriting 및 EthanolConcentration 데이터셋에서 각각 125.2% 및 50.2%의 분류 정확도 개선을 달성하며 최첨단 기준선 모델을 크게 능가했습니다. LLM 기반 방법들이 다양한 데이터셋에서 신뢰할 수 있는 결과를 제공하는 것으로 나타나, 이 모델은 산업적 환경에서 데이터가 제한된 상황에서도 적용 가능성을 보였습니다.



### Towards Recommender Systems LLMs Playground (RecSysLLMsP): Exploring Polarization and Engagement in Simulated Social Networks (https://arxiv.org/abs/2502.00055)
Comments:
          8 pages, 2 figures

- **What's New**: 이 논문에서는 인공지능 기술의 급속한 발전과 추천 시스템의 악영향 가능성을 고려하여, 추천 시스템의 효과를 시뮬레이션하고 평가하는 것이 중요하다고 강조합니다. 새로운 프레임워크인 Recommender Systems LLMs Playground (RecSysLLMsP)를 통해, 대형 언어 모델(LLMs)을 활용하여 다양한 콘텐츠 추천 설정이 소셜 네트워크에서 사용자 참여 및 세분화에 미치는 영향을 조사합니다.

- **Technical Details**: RecSysLLMsP는 다채로운 AI 에이전트(AgentPrompts)를 생성하여, 세 가지 시나리오인 Plurality, Balanced, Similarity에서 자율 행동을 평가합니다. Similarity 시나리오에서는 사용자 선호도에 맞춘 콘텐츠가 최대의 참여를 이끌어내지만, 동시에 에코 챔버(echo chamber)를 촉진할 가능성이 있습니다. 반대로 Plurality 시나리오는 다양한 상호작용을 유도하지만, 참여도 결과는 혼합되어 나타납니다.

- **Performance Highlights**: 이 연구는 추천 시스템 디자인에서 사용자 만족을 높이며 사회적 분열을 완화하기 위해 신중한 균형이 필요함을 강조합니다. RecSysLLMsP의 장점은 사회적 영향을 평가하고 다양한 추천 시스템 설정에 대한 사용자 참여 수준을 결정하는 데 필수적인 세분화 효과를 계산할 수 있는 능력에 있습니다. 하지만, 연구의 한계는 현실을 정확하게 모사하는 것이며, 향후 연구는 실제 인간과 AgentPrompts 간의 행동 유사성을 검증하고 세분화 점수를 측정할 수 있는 지표를 수립해야 합니다.



### Bridging Contrastive Learning and Domain Adaptation: Theoretical Perspective and Practical Application (https://arxiv.org/abs/2502.00052)
- **What's New**: 이 연구는 대비 학습(Contrastive Learning)과 도메인 적응(Domain Adaptation) 간의 이론적 관계를 탐구합니다. 특히 NT-Xent loss와 Supervised Contrastive loss 두 가지 대조 손실이 Class-wise Mean Maximum Discrepancy(CMMD)와 관련되어 있다는 점을 밝힙니다. 이러한 발견은 대비 학습이 도메인 적응에서 어떻게 활용될 수 있는지를 이론적으로 뒷받침합니다.

- **Technical Details**: 연구에서 제시된 두 가지 손실 함수는 도메인 적응에서 널리 사용되는 비슷도 측정기준인 CMMD를 줄이는 것으로 나타났습니다. 대조 손실을 최소화하는 과정은 클래스 구분성(class-separability)을 향상시키며, 이는 특히 의료 이미징 분야에서 중요하게 다뤄집니다. 실험은 유방 촬영 이미지에서 진행되었으며, 다양한 데이터셋을 사용했습니다.

- **Performance Highlights**: 세 가지 유방 촬영 데이터셋(합성 패치, 임상(실제) 패치, 임상(실제) 이미지)에서의 실험 결과, Supervised Contrastive loss를 최소화할 경우 도메인 적응과 클래스 구분성, 분류 성능(classification performance)이 개선됨을 보여줍니다. 이는 도메인 적응이 신뢰할 수 있는 이미지 분석에 결정적인 역할을 할 수 있음을 시사합니다.



### Contextually Entangled Gradient Mapping for Optimized LLM Comprehension (https://arxiv.org/abs/2502.00048)
- **What's New**: 이 연구에서는 Contextually Entangled Gradient Mapping (CEGM)이라는 새로운 접근법을 소개하고 있습니다. CEGM은 기존의 경량 모델에서의 기울기 최적화 관계를 재정의하여, 주제와 의미의 일관성을 높이고 추론 능력을 향상시키는 방법론입니다. 기울기를 독립적인 수치적 개체가 아닌 동적으로 연결된 맥락적 의존성의 운반자로 다루는 접근법으로, 현재 최적화 전략의 중요한 격차를 해소하기 위해 설계되었습니다. CEGM의 통합은 고차원 추론 및 맥락적 유지, 다양한 환경에의 적응성을 포함한 여러 작업에서 유의미한 향상을 나타냈습니다.

- **Technical Details**: CEGM은 모델의 최적화 과정에서 기울기와 맥락적 표현 간의 새로운 상호작용을 통해 작동합니다. 이 방법론은 다차원적인 맥락 의존성을 포착하도록 설계된 구조적 상호작용을 활용하여 여러 레이어의 상관관계를 다루고, 신경망 아키텍처의 효율성을 증가시킵니다. 이러한 동적 맥락 기반 조정은 기울기가 가시적이면서도 복잡한 관계를 반영함으로써 기존의 정적 모델 아키텍처에서의 한계를 극복하고자 합니다.

- **Performance Highlights**: 실험 데이터는 CEGM을 활용한 모델이 기초 모델과 비교했을 때 지속적으로 높은 정확도와 노이즈에 대한 저항력을 나타냈음을 보여줍니다. 문장 변환 과정에서 의미의 편향 감소와 의미적 일관성을 향상시켰으며, 이는 제안된 방법론의 강력함과 다용성을 강조합니다. CEGM을 적용한 윤곽 조정 및 훈련 파이프라인 변경을 통해 중장기 추론 요구 사항을 성공적으로 충족함으로써, 새로운 연구와 개발을 위한 길을 열어주고 있습니다.



### Restless Multi-armed Bandits under Frequency and Window Constraints for Public Service Inspections (https://arxiv.org/abs/2502.00045)
- **What's New**: 이 연구에서는 시카고의 식품 시설 검사를 사례로 하여 서비스 검사를 효과적으로 계획하기 위한 새로운 방법을 제시합니다. 논문에서는 각 시설에 inspection window를 부여하고, 이를 통해 검사 횟수를 보장하면서도 영향을 극대화할 수 있는 새로운 접근 방식을 개발했습니다. 특히, Restless Multi-Armed Bandit (RMAB) 문제에 대한 기존 접근 방식의 한계를 해결하기 위해 Whittle index 이론과 정수 프로그래밍을 결합했습니다.

- **Technical Details**: 이 방법론은 MDP(Markov Decision Processes)와 정수 프로그래밍을 통해 검사 프로세스를 최적화하는 구조로 설계되었습니다. 시설에는 검사 시간이 연속적으로 부여되며, 이를 통해 예측이 어렵도록 최적화할 수 있습니다. 제약 조건을 준수하며 검사 창을 최적화하는 방식은 실험을 통해 유의미한 성과를 창출했습니다. 또한, 머신러닝 모델을 활용해 시카고의 실제 검사 기록을 기반으로 상태 전이 확률을 예측함으로써, 검사 실패 예측의 정확도를 약 10% 향상했습니다.

- **Performance Highlights**: 실험 결과, 연구의 방법론을 통해 시뮬레이션에서는 최대 24%, 실제 데이터에서는 최대 33%의 보상 향상을 확인했습니다. 제약 조건을 독립적으로 모델링하고 최적화된 획득 창을 사용함으로써, 기존 비효율적인 방법과 비교해 상당한 이점을 얻었습니다. 연구는 또한 RMAB 계획의 새로운 기준을 제시하고, 도시 서비스 스케줄을 최적화하는 데 있어 중요한 통찰을 제공합니다.



### A scalable adaptive deep Koopman predictive controller for real-time optimization of mixed traffic flow (https://arxiv.org/abs/2502.00043)
- **What's New**: 이 논문에서는 혼합 교통 흐름을 조절하기 위한 적응형 심층 쿠프만 예측 제어 프레임워크(AdapKoopPC)를 제안합니다. HDVs(인간 운전 차량)의 차량 추적 행동을 모델링하기 위해 쿠프만 이론 기반의 적응형 궤적 예측 심층 네트워크(AdapKoopnet)를 설계하였으며, 이는 HDVs의 행동을 고차원 공간에서 선형적으로 표현할 수 있게 합니다. 또한 AdapKoopPC는 CAVs(연결 자동화 차량)와 AdapKoopnet의 예측 블록을 통합하여 혼합 교통 흐름을 원활하게 합니다.

- **Technical Details**: 이 연구는 CAVs와 HDVs가 혼합된 교통 상황에서의 HDVs의 주행 행동 모델링 및 예측을 중점적으로 다룹니다. AdapKoopnet는 고차원 선형 모델을 통해 HDVs의 차량 추적 행동을 모델링하며, 선형 동적 모델과 예측 블록을 통합한 상태 예측 모델을 개발하여 복잡한 비선형 동적 시스템을 선형 변환으로 처리할 수 있습니다. 이를 통해 복잡한 교통 상황에서도 실시간 최적화를 위한 유용한 모델이 만들어집니다.

- **Performance Highlights**: 제안된 AdapKoopnet은 기존의 비선형 모델보다 더 정확한 HDVs의 예측 궤적을 제공합니다. 또한 AdapKoopPC는 낮은 CAVs 침투율에서도 교통 진동 완화 시 보다 효과적인 제어 성능을 보이며 계산 비용이 적습니다. 이 연구의 코드는 오픈 소스로 제공되어 연구자들이 이를 개발하고 확장하는 데 기여할 것으로 기대됩니다.



### Multi-Objective Reinforcement Learning for Power Grid Topology Contro (https://arxiv.org/abs/2502.00040)
- **What's New**: 이번 논문은 다중 목표 강화 학습(multi-objective reinforcement learning, MORL) 접근법을 통해 전력망의 토폴로지 제어 문제를 해결하고자 합니다. 기존의 연구는 단일 목표에 초점을 맞춘 경우가 많았으나, 이 연구는 다양한 운영 목표를 동시에 고려하는 정책 세트를 제안하여 시스템 운영자에게 더 나은 의사결정 지원을 제공합니다. 특히 심층 최적선형 지원(deep optimistic linear support, DOL)과 다중 목표 근접 정책 최적화(multi-objective proximal policy optimization, MOPPO)를 통해 파레토 최적 정책을 생성합니다.

- **Technical Details**: 제안된 접근법은 MOPPO 알고리즘과 DOL을 결합하여, 여러 운영 목표를 충족하는 보상 함수를 설계합니다. 보상 함수는 선로 하중(line loading), 토폴로지 편차(topological deviation), 스위칭 주파수(switching frequency) 등을 포함하여, 시스템 운전자가 목표 간의 트레이드오프를 잘 이해할 수 있도록 돕습니다. 또한, 이 방법은 여러 개의 정책을 생성하여 시스템 운영자에게 최적의 정책을 선택할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: 초기 사례 연구 결과, MORL 접근법은 목표 간의 트레이드오프를 제공하며, 무작위 탐색 기반 접근법에 비해 파레토 프론트 근사를 크게 개선하는 것으로 나타났습니다. 특히, MORL을 적용한 정책은 비상 상황에서의 그리드 고장을 방지하는 데 30% 더 성공적이며, 훈련 예산이 줄어들 때도 20% 더 효과적입니다. 이러한 결과는 다중 목표 정책이 전력망 운영에 있어 유용하다는 것을 시사합니다.



### Efficient Client Selection in Federated Learning (https://arxiv.org/abs/2502.00036)
- **What's New**: 이 논문에서는 데이터 프라이버시를 보존하면서 분산 머신 러닝을 가능하게 하는 Federated Learning (FL)에서 클라이언트 선택 방법을 향상시키는 새로운 프레임워크를 제안합니다. 새로운 방법은 differential privacy (DP)와 fault tolerance을 결합하여 모델 성능을 향상시키고 시스템의 견고성을 강화합니다. 선택 과정에서 클라이언트 수를 성능과 시스템 제약에 따라 동적으로 조정하며, 프라이버시를 보호하기 위해 노이즈를 추가합니다.

- **Technical Details**: 제안된 클라이언트 선택 방법은 정밀성과 프라이버시, 내결함성을 균형 있게 고려합니다. 클라이언트는 데이터 품질 및 계산 능력과 같은 요소를 기반으로 평가되며, 상위 K개의 클라이언트가 선택됩니다. differential privacy는 모델 업데이트에 가우시안 노이즈를 추가하는 방식으로 보장되며, checkpointing 메커니즘은 훈련 중 장애로부터 회복할 수 있는 기능을 제공합니다.

- **Performance Highlights**: UNSW-NB15와 ROAD 데이터세트를 사용한 평가에서 제안된 방법은 기준선에 비해 7%의 정확도 향상과 25%의 훈련 시간 단축을 보여줍니다. ROAD 데이터 세트에서 특히 두드러진 성능을 보였으며, 다양한 네트워크 조건에서 복잡한 이상 패턴을 효과적으로 처리했습니다. fault tolerance 메커니즘은 시스템의 안정성을 높이면서도 약간의 성능 저하를 보이는 것으로 나타났습니다.



### Querying Databases with Function Calling (https://arxiv.org/abs/2502.00032)
Comments:
          Preprint. 23 pages, 7 figures

- **What's New**: 이번 연구는 Large Language Models (LLMs)와 데이터베이스 쿼리를 통합하는 새로운 도구 정의를 제안합니다. 연구진은 Function Calling을 활용하여 LLM이 데이터에 접근하고, 검색 쿼리 및 필터를 적용해 효과적으로 쿼리를 수행할 수 있도록 하였습니다. 또한, 8개의 LLM을 사용하여 정확한 쿼리 결과를 평가하는 DBGorilla 벤치마크를 소개합니다.

- **Technical Details**: 연구에서는 Gorilla LLM 프레임워크를 기반으로 합성 데이터베이스 스키마와 쿼리를 생성하며, Function Calling을 통해 데이터 쿼리 처리를 효율적으로 할 수 있도록 합니다. 제안된 도구 정의는 검색 쿼리와 구조화된 데이터 접근을 통합하며, SQL 다이얼렉트와 관련된 다양한 쿼리 연산자를 쉽게 결합할 수 있게 합니다.

- **Performance Highlights**: 성능 평가 결과, Claude 3.5 Sonnet이 74.3%의 Exact Match 점수로 가장 높은 성과를 보였습니다. 전반적으로 LLM은 boolean 속성에 대한 연산 활용에 효과적이지만, 텍스트 속성 필터에는 어려움을 겪고 있습니다. 이번 연구는 LLM이 기능 호출(Function Calling)을 통해 데이터베이스를 효과적으로 쿼리할 수 있음을 보여줍니다.



### AlphaSharpe: LLM-Driven Discovery of Robust Risk-Adjusted Metrics (https://arxiv.org/abs/2502.00029)
- **What's New**: 이 논문에서는 AlphaSharpe라는 새로운 프레임워크를 소개하며, 이는 대형 언어 모델(LLM)을 활용하여 재무 성과 메트릭을 반복적으로 진화 및 최적화합니다. 특히, 기존의 재무 메트릭들이 가진 저항력 및 일반화의 한계를 극복하고 있습니다. LLM을 적용하여 생성된 새로운 리스크-수익 메트릭은 기존 메트릭보다 더 우수한 예측 능력을 보여줍니다.

- **Technical Details**: AlphaSharpe는 크로스오버(crossover), 변이(mutational), 평가(evaluation) 기법을 통해 재무 메트릭을 발전시키는 방법론을 구현합니다. LLM의 창의적인 생성 능력을 통해 다양하고 독창적인 메트릭 변종을 만들어내며, 각 메트릭은 데이터의 노이즈와 아웃라이어에 대한 저항성을 높이도록 설계되어 있습니다. 이러한 반복적인 최적화 과정은 메트릭이 미래 성과와 잘 일치하도록 보장합니다.

- **Performance Highlights**: 실험 결과 AlphaSharpe 메트릭이 전통적인 메트릭에 비해 평균 3배의 예측력과 2배의 포트폴리오 성과를 기록함을 보여줍니다. 발견된 메트릭들은 포트폴리오 관리자 및 금융 의사결정자들이 효과적으로 활용할 수 있도록 설계되었습니다. 이 연구는 재무 분석의 발전을 위한 LLM의 잠재력을 입증하며, 보다 신뢰성 있는 투자 전략을 개발하는 데 기여할 것입니다.



### Analysis of a Memcapacitor-Based for Neural Network Accelerator Framework (https://arxiv.org/abs/2502.00027)
Comments:
          11 pages, 7 figures

- **What's New**: 이 연구에서는 메모리스트(Memristive) 장치를 활용하여 신경망(Neural Networks)을 직접 매핑하는 전문 하드웨어 개발의 새로운 접근 방식을 제시합니다. 특히, CMOS 기반 메모캐패시터(CMOS-based memcapacitor) 회로를 도입하였고, 이는 Cadence 도구를 사용하여 검증되었습니다. 또한, 메모캐패시티브 가속기(memcapacitive-based accelerator) 설계를 용이하게 만들기 위해 Python으로 장치를 개발했습니다. 이러한 신기술은 에너지 집약적인 신경망 훈련을 효율적으로 지원할 것으로 기대됩니다.

- **Technical Details**: 제안된 프레임워크는 메모캐패시터 장치의 크로스바 배열(Crossbar Array)을 사용하여 디지털 분류 및 CIFAR 데이터 세트 인식을 위한 신경망 훈련을 수행합니다. 구축된 메모캐패시터 기반 신경망의 비이상적인 특성을 테스트하였으며, 성능을 평가하기 위한 다양한 실험을 진행했습니다. 이 과정에서 메모캐패시터의 역할과 특성에 대한 심층적인 분석이 포함되었습니다.

- **Performance Highlights**: 훈련 결과, 숫자 인식에서 98.4%의 높은 훈련 정확도를 달성하였으며, CIFAR 인식에서는 94.4%의 훈련 정확도를 기록했습니다. 이 연구는 분류 작업에서 메모캐패시터 기반 신경망 시스템의 잠재력을 입증하였으며, 뉴로모픽(Neuromorphic) 컴퓨팅의 발전을 위한 기초를 다졌습니다. 이러한 성과는 앞으로의 연구에 중요한 방향성을 제시하고 있습니다.



### Pushing the Limits of BFP on Narrow Precision LLM Inferenc (https://arxiv.org/abs/2502.00026)
- **What's New**: 이 논문에서는 Block Floating Point (BFP) 방식을 비선형 연산에 적용할 가능성과 한계를 탐구합니다. 기존의 LLM들이 비효율적인 부동 소수점 형식을 사용하던 문제를 해결하기 위해, Dynamic-BFP (DBFP)라는 새로운 BFP 변형을 제안하며 데이터에 대한 피벗 초점(pivot-focus) 전략과 유연한 지수 공유를 위한 적응형 그룹화 전략을 포함하고 있습니다.

- **Technical Details**: 제안된 DBFP는 비선형 연산의 정확성과 효율성을 개선하기 위해 다양한 수학적 모델과 최적화 기법을 적용합니다. 이 시스템은 DBFP 포맷을 사용하여 Softmax 같은 비선형 연산을 가속화하고, BFP Matmul과 DBFP Softmax 간의 데이터 흐름을 간소화하여 명시적인 변환을 제거합니다. DH-LUT라는 새로운 lookup table 알고리즘을 통해 비선형 연산 속도를 74% 향상시키는 것도 포함되어 있습니다.

- **Performance Highlights**: DB-Attn 프레임워크를 통해 FPGA 및 ASIC에 적용할 수 있는 RTL 수준의 DBFP 기반 엔진을 구현했습니다. 이 엔진은 최신 설계(SOTA) 대비 10배의 성능 향상을 제공하며, 최소한의 정확도 손실로 대규모 처리에서의 높은 성능 개선을 실현합니다. 결과적으로 LLaMA의 Softmax에서 74% GPU 속도 향상과 10배의 적은 경량 성능 개선을 달성하였습니다.



### Leveraging Large Language Models to Enhance Machine Learning Interpretability and Predictive Performance: A Case Study on Emergency Department Returns for Mental Health Patients (https://arxiv.org/abs/2502.00025)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 전통적인 기계 학습 접근법과 통합하여 응급실(ED) 정신 건강 재방문 위험 모델의 예측 정확도와 임상 해석 가능성을 개선할 수 있는지를 평가하였습니다. 연구는 2018년 1월부터 2022년 12월까지 미국 남부의 한 학술 의료 센터에서 27,904명의 고유한 정신 건강 환자에 대한 42,464건의 ED 방문 데이터를 분석하였습니다. 이 연구는 SHAP 값과 임상 맥락 지식을 통합하는 새로운 검색 증강 생성(RAG) 프레임워크를 통해 모델의 해석 가능성을 높였습니다.

- **Technical Details**: 연구의 주요 성과 지표는 (1) 30일 이내 ED 재방문 예측 정확도와 (2) SHAP 값을 사용하여 복잡한 모델 예측을 임상적으로 의미 있는 설명으로 변환하는 기계 학습 해석 가능성 프레임워크였습니다. 연구 결과, 제안된 프레임워크는 99%의 정확도로 모델 예측을 임상 현실에 맞게 설명할 수 있었습니다. LLM으로 추출된 특징을 통합함으로써 XGBoost 모델의 곡선 아래 면적(AUC)은 0.73에서 0.76으로 향상되었습니다.

- **Performance Highlights**: LLM 기반의 특징 추출 방법은 10-shot 학습을 사용하여 전통적인 접근법보다 현저하게 우수한 성능을 보였으며, 주요 불만 분류에 대해 0.882의 정확도와 0.86의 F1 점수를 달성했습니다. 이는 전통적인 방법의 정확도 범위인 0.59에서 0.63에 비해 현저히 높은 값입니다. 여러 SDoH(사회적 결정 요인) 카테고리에서의 정확도는 0.65에서 0.93 사이로 나타나면서 임상 노트에서 특징을 효과적으로 추출하는 견고한 성능을 강조했습니다.



### Musical Agent Systems: MACAT and MACataR (https://arxiv.org/abs/2502.00023)
Comments:
          In Proceedings of the Creativity and Generative AI NIPS (Neural Information Processing Systems) Workshop 2024

- **What's New**: 본 연구는 음악 성과 즉흥 improvisation을 지원하는 인간 개입 generative AI 시스템 즉, 음악 에이전트(MACAT 및 MACataRT)의 개발과 활용을 탐구합니다. MACAT은 실시간 합성 및 자기 청취(real-time synthesis and self-listening)를 통해 독립적으로 출력을 형성하도록 최적화된 에이전트 주도 성과에 중점을 둡니다. 반면에 MACataRT는 오디오 모자이크(audio mosaicing) 및 시퀀스 기반 학습을 통해 협업 즉흥을 위한 유연한 환경을 제공합니다.

- **Technical Details**: 음악 에이전트는 AI 및 다중 에이전트 시스템의 원칙을 바탕으로하여 인간과 AI 간의 실시간 음악적 맥락에 적응합니다. 여기서는 MACAT와 MACataRT 시스템이 각각 소리 메모리 및 패턴 인식을 위해 자가 조직화 맵(SOM) 및 변수 마르코프 모델(VMM)을 활용하며, 음향 특성 기반의 오디오 선택을 가능하게 하는 데이터를 처리하는 방법을 설명합니다. 음악 성과에서의 자발적 즉흥성을 지원하기 위해, MACAT과 MACataRT는 훈련 데이터 세트를 작고 맞춤형으로 설계하여 예술적 표현을 존중하는 투명한 AI 관여를 촉진합니다.

- **Performance Highlights**: 본 연구는 실시간 생성 AI가 음악가가 즉흥작곡을 통해 예술 표현의 새로운 형태를 탐색하도록 어떻게 권한을 부여할 수 있는지를 강조합니다. MACAT과 MACataRT의 변형된 워크플로우는 즉흥 공연과 공동 창작을 통해 생성을 위한 상호작용적이고 동적인 환경을 제공합니다. AI 에이전트와 협업하는 과정은 즉흥, 스타일 적응 및 창작의 범위를 확장하여 음악적 가능성을 풍부하게 합니다.



### Ethical Concerns of Generative AI and Mitigation Strategies: A Systematic Mapping Study (https://arxiv.org/abs/2502.00015)
- **What's New**: 이번 논문은 LLMs(대규모 언어 모델)의 사용과 관련된 주요 윤리적 문제를 식별하고 분류하고, 기존의 완화 전략을 조사하며, 다양한 도메인에서 이러한 전략들을 구현하는 데 있어 남아있는 도전 과제를 평가합니다. 특히, 다양한 윤리적 차원을 통해 LLMs와 관련된 윤리적 문제를 다각적으로 분석했습니다.

- **Technical Details**: 연구팀은 39개의 연구를 systematic mapping study 방식으로 검토하였으며, 윤리적 문제와 완화 전략에 대한 논의를 포괄적으로 분석했습니다. 윤리적 문제는 다양한 기존 가이드라인과 프레임워크에 기반하여 추출한 다섯 가지 윤리적 차원으로 정리되었습니다.

- **Performance Highlights**: 연구 결과, LLMs의 윤리적 문제는 다차원적이며 상황에 따라 달라지는 것으로 나타났습니다. 제안된 완화 전략은 일부 문제를 해결하지만 여전히 많은 도전 과제가 존재하며, 특히 의료 및 공공 거버넌스와 같은 고위험 분야에서는 윤리 문제가 실질적인 구현을 방해하고 있습니다.



### TOAST Framework: A Multidimensional Approach to Ethical and Sustainable AI Integration in Organizations (https://arxiv.org/abs/2502.00011)
Comments:
          25 pages, 1 figure

- **What's New**: 이번 논문에서는 AI 시스템의 성공적인 구현을 위한 새로운 프레임워크인 TOAST(Trustworthy, Optimized, Adaptable, Socio-Technologically harmonious)를 소개하고 있습니다. 이 프레임워크는 기술 전략을 윤리적 가치와 사회적 책임, 혁신적 열망에 맞추는 데 중점을 두고 있습니다. 다양한 분야의 통찰력을 활용하여 AI 구현의 복잡한 도전에 대응하고자 합니다.

- **Technical Details**: TOAST 프레임워크는 신뢰성(reliability), 책임성(accountability), 기술 발전(technical advancement), 적응력(adaptability), 그리고 사회-기술적 조화(socio-technical harmony)라는 다섯 가지 핵심 요소를 중심으로 구성되어 있습니다. 이 프레임워크는 주로 의료 분야의 사례 연구에 기반을 두고 있으며, AI 시스템 구현에서의 운영적, 윤리적, 규제적 도전 과제를 해결하는 데 도움이 되도록 구성되어 있습니다.

- **Performance Highlights**: 이 논문은 TOAST 프레임워크가 AI 시스템이 기관의 효율성을 어떻게 향상시키고, 편향(bias) 및 데이터 프라이버시(data privacy)와 같은 위험을 완화할 수 있는지를 보여줍니다. 또한, 윤리적으로 정렬되고 효율적인 AI 통합이 필요한 다른 분야에도 적용 가능한 복제 가능한 모델을 제공합니다.



### A Study about Distribution and Acceptance of Conversational Agents for Mental Health in Germany: Keep the Human in the Loop? (https://arxiv.org/abs/2502.00005)
Comments:
          Master's thesis

- **What's New**: 이 연구는 독일에서 정신 건강을 위한 AI 기반의 대화형 에이전트(conversational agents, CAs) 사용에 대한 일반 대중과 의료 전문가의 시각을 조사합니다. 특히, 온라인 설문조사를 통해 CAs의 사용 빈도, 수용도, 그리고 상담 및 진단에서의 수용 가능성을 평가하였습니다.

- **Technical Details**: 연구는 444명의 일반 대중과 351명의 의료 전문가를 대상으로 실시된 두 개의 정량적 온라인 설문조사로 이루어졌습니다. 분석 결과, 응답자의 27%가 이미 CAs에게 자신의 문제를 털어놓고 있으며, 이 기술에 대한 경험과 원격 의료(telemedicine) 경험이 높은 수용도를 나타내는 것으로 나타났습니다.

- **Performance Highlights**: 일반 대중은 CAs를 추가적인 전문가로 보기보다는 의료 전문가가 통제하는 동반자로 보는 경향이 더 강하였습니다. CAs는 정신 건강 상담에서 특히 지원할 잠재력을 가지고 있으며, 향후 연구는 다양한 커뮤니케이션 매체의 영향 및 증강 인공지능(augmented intelligence)의 추가 가능성을 조사해야 합니다.



### Defending Compute Thresholds Against Legal Loopholes (https://arxiv.org/abs/2502.00003)
- **What's New**: 이 논문에서는 AI 모델에 대한 기존 법적 프레임워크가 어떻게 훈련 계산(compute) 기준을 이용하여 위험한 AI 모델을 식별하는지 검토합니다. 미국의 행정명령 14110의 4.2(a) 조항은 특정 훈련 계산 기준을 초과하는 AI 모델의 개발자로부터 광범위한 보고를 요구합니다. 유럽연합의 AI 법안 51조는 특정 계산 기준 이상인 AI 모델이 높은 영향력을 갖고 있다고 가정하여 개발자에게 여러 의무를 부여합니다.

- **Technical Details**: 이 논문은 훈련 계산(compute) 사용을 감소시키면서 모델 능력(capabilities)을 유지하거나 증가시킬 수 있는 여러 기술들을 분석합니다. 특히, 모델의 수명 연장, 모델 재사용(model reuse), 모델 확장, 계산 최적화(compute-optimal inference compute)와 같은 네 가지 주요 기술을 집중적으로 설명합니다. 이러한 기술들은 기존의 훈련 계산 기준을 회피할 수 있는 법적 허점을 제공할 가능성이 있습니다.

- **Performance Highlights**: 법적 메커니즘으로서의 훈련 계산 기준에 대한 논의를 진전시키고, 이와 관련된 법적 허점을 해결하기 위한 정책 권장사항을 제안합니다. 이러한 논의는 AI 모델의 규제와 그에 따른 책임을 명확히 하는 데 기여할 수 있습니다. 더 나아가, 이러한 기술들은 AI 개발자들이 법적 의무를 준수하면서도 효율적으로 모델을 개발할 수 있게 도와줄 것입니다.




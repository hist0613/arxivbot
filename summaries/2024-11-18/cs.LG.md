New uploads on arXiv(cs.CL)

### Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization (https://arxiv.org/abs/2411.10442)
- **What's New**: 본 연구에서는 멀티모달 대형 언어 모델(MLLMs)의 Chain-of-Thought (CoT) 성능을 향상시키기 위한 새로운 선호 최적화(Preference Optimization, PO) 프로세스를 소개합니다. 특히, 고품질 대규모 멀티모달 추론 선호 데이터셋인 MMPR을 생성하기 위한 자동화 선호 데이터 구축 파이프라인을 설계하고, MLLMs와 PO를 통합하여 Mixed Preference Optimization (MPO) 방법을 개발했습니다.

- **Technical Details**: 현재 MLLMs는 주로 프리트레이닝(pre-training) 및 감독된 미세 조정(supervised fine-tuning, SFT) 프로세스를 따릅니다. 하지만 SFT 손실로 인해 분포 전이(distribution shift) 문제가 발생하며, 이는 CoT 성능을 저하시킵니다. 본 연구의 제안은 Dropout Next Token Prediction (DropoutNTP) 파이프라인과 같은 간단하고 효과적인 MPO 방법을 도입하여 멀티모달 CoT 성능을 향상시키는 것입니다.

- **Performance Highlights**: 제안된 모델 InternVL2-8B-MPO는 MathVista에서 67.0의 정확도를 달성하여, 기존의 InternVL2-8B보다 8.7점 향상되었으며, 10배 더 큰 모델인 InternVL2-76B와 동등한 성능을 보였습니다. 이는 PO 방법론이 MLLMs의 추론 능력을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### Mitigating Hallucination in Multimodal Large Language Model via Hallucination-targeted Direct Preference Optimization (https://arxiv.org/abs/2411.10436)
- **What's New**: 본 연구에서는 Hallucination-targeted Direct Preference Optimization (HDPO)을 제안하여 다중 모달 큰 언어 모델(MLLMs)의 환각(hallucination)을 줄이는 새로운 접근 방식을 도입합니다. 이전 방법들과 달리, HDPO는 MLLM의 다양한 환각 원인과 형태를 다룹니다.

- **Technical Details**: HDPO는 세 가지 유형의 환각 원인에 대한 선택적 preference pair 데이터를 개발하여 구체적인 환각 상황을 해결합니다: 1) 충분하지 않은 시각적 능력, 2) 긴 맥락 생성, 3) 다중 모달 충돌. 각 유형에 대해 고유한 데이터를 생성하며, 이를 통해 MLLM의 능력을 향상시킵니다.

- **Performance Highlights**: 실험 결과 HDPO는 다양한 환각 평가 데이터셋에서 우수한 성능을 보여주었으며, 대부분의 최신 방법(SOTA)을 초월하며 MLLM 환각 문제를 효과적으로 완화했습니다. 특히, 다양한 M-hallu 작업에서 일관된 개선을 달성했습니다.



### Towards Automatic Evaluation of Task-Oriented Dialogue Flows (https://arxiv.org/abs/2411.10416)
- **What's New**: 이번 논문에서는 대화 흐름(diaogue flows)의 품질을 평가하기 위한 새로운 지표인 FuDGE (Fuzzy Dialogue-Graph Edit Distance)를 소개합니다. 이 지표는 대화 흐름의 구조적 복잡성과 대화 데이터의 표현 범위를 평가하여 대화 시스템의 효율성과 자동화를 높이는 데 기여할 수 있습니다.

- **Technical Details**: FuDGE는 개별 대화가 대화 흐름과 얼마나 잘 일치하는지를 측정하고, 전반적으로 대화 세트가 대화 흐름에 얼마나 잘 표현되는지를 계산합니다. 이 연구에서는 효율적인 edit-distance 메트릭을 통해 대화 흐름과 대화 간의 거리를 계산하고, Flow-F1 (FF1) 점수를 사용하여 복잡성과 압축 간의 트레이드오프를 포착합니다.

- **Performance Highlights**: FuDGE 평가 프레임워크를 통해 수동으로 구성된 대화 흐름 및 자동 생성된 흐름을 대규모 실험을 통해 평가할 수 있었으며, 이는 대화 흐름의 품질을 표준화하고 최적화할 수 있는 가능성을 보여줍니다.



### A Survey of Event Causality Identification: Principles, Taxonomy, Challenges, and Assessmen (https://arxiv.org/abs/2411.10371)
- **What's New**: 이번 연구는 Event Causality Identification (ECI)에 대한 체계적인 검토와 평가를 제공하며, 기존 연구 방법론과 모델의 양적 평가를 포함합니다. ECI의 개념적 틀을 수립하고, 문장 수준(SECI) 및 문서 수준(DECI) 이벤트 인과 관계 식별을 위한 분류 체계를 제안합니다.

- **Technical Details**: ECI는 텍스트 데이터에서 이벤트 간의 인과관계를 자동으로 추출하는 자연어 처리(NLP) 작업입니다. SECI는 feature pattern 기반 매칭, 심층 의미 인코딩, 인과 지식 사전 훈련, 프롬프트 기반 미세 조정 등 다양한 방법을 분석하며, DECI는 이벤트 그래프 추론 및 프롬프트 기반 기술을 중심으로 복잡한 교차 문장 인과 추론 문제를 해결합니다.

- **Performance Highlights**: 본 논문은 두 개의 기준 데이터 세트에 대한 다양한 ECI 방법의 양적 평가를 수행하며, 각 접근법의 강점, 한계 및 개방된 도전 과제를 분석합니다. 또한, LLMs를 활용한 최근 연구 동향을 조명하여 ECI의 향후 발전 방향을 제안합니다.



### Emotion Detection in Reddit: Comparative Study of Machine Learning and Deep Learning Techniques (https://arxiv.org/abs/2411.10328)
- **What's New**: 본 연구는 GoEmotions 데이터셋을 활용하여 텍스트 기반의 감정 감지(emotion detection) 방식을 집중적으로 다루고 있습니다. 27개의 다양한 감정이 주석 처리된 Reddit 댓글을 통해 감정 인식의 정확성을 높이고자 하였습니다.

- **Technical Details**: 이 연구에서는 여섯 가지 머신 러닝 모델, 세 가지 앙상블 모델(ensemble models), 그리고 Long Short-Term Memory (LSTM) 모델을 포함하여 감정 감지를 위한 다양한 모델을 사용하였습니다. 감정은 Ekman의 여섯 가지 기본 카테고리인 기쁨(joy), 분노(anger), 두려움(fear), 슬픔(sadness), 혐오(disgust), 놀라움(surprise)으로 매핑됩니다.

- **Performance Highlights**: 스태킹 분류기(Stacking classifier)가 정확도와 성능에서 다른 모델보다 우수한 결과를 보여주었습니다. 또한, EmoBERTa라는 사전 학습(pre-trained) 감정 감지 모델과 비교했을 때, 우리의 Stacking classifier가 더욱 효과적임을 입증하였습니다. 마지막으로, 이 분류기는 Streamlit 웹 애플리케이션을 통해 배포되어 텍스트 기반 감정 분석의 실제 적용 가능성을 강조하고 있습니다.



### Unveiling Topological Structures in Text: A Comprehensive Survey of Topological Data Analysis Applications in NLP (https://arxiv.org/abs/2411.10298)
- **What's New**: 본 논문은 Natural Language Processing (NLP) 분야에서 Topological Data Analysis (TDA)의 응용에 관한 85개의 연구를 포괄적으로 조사한 결과를 제시합니다. TDA는 노이즈에도 불구하고 데이터의 본질적인 형태를 포착할 수 있는 통계적 접근법으로, ML의 한계를 극복할 수 있도록 도와줍니다.

- **Technical Details**: TDA에서는 Persistent Homology (PH)와 Mapper라는 두 가지 주요 기술을 사용하여 데이터를 분석합니다. PH는 데이터의 위상적 특성을 여러 차원에서 추출하며, 데이터는 점군(point cloud)으로 표현되고, 다양한 차원의 구멍(hole)을 기록합니다. 이러한 위상적 특성은 지속성(persistence)으로 정의되며, 지속성 다이어그램은 이러한 데이터를 시각화하는 예시입니다.

- **Performance Highlights**: TDA의 적용은 NLP 분야에서 증가하고 있으며, TDA를 통해 TF-IDF, Word2Vec, BERT 등의 전통적인 수치 표현 기법으로는 포착할 수 없는 새로운 위상적 특성을 추출할 수 있습니다. 이는 NLP 애플리케이션에서 TDA의 점진적인 중요성을 강조하고 있습니다.



### Measuring Non-Adversarial Reproduction of Training Data in Large Language Models (https://arxiv.org/abs/2411.10242)
- **What's New**: 이 논문은 대형 언어 모델이 훈련 데이터의 일부를 어떻게 암기하는지를 연구하며, 특히 비대립적 상황(일반적인 질문이나 요청)에 대한 응답 시 모델과 훈련 데이터 사이의 중복(overlap)을 정량화합니다.

- **Technical Details**: 우리가 조사한 비대립적 재생(non-adversarial reproduction)에서는 일반적인 프롬프트(prompt) 카테고리(예: 편지 작성, 튜토리얼 등)에 대해 인기 있는 대화형 언어 모델이 생성한 최대 15%의 텍스트가 인터넷의 스니펫(snippet)과 중복된다는 것을 보여줍니다.

- **Performance Highlights**: 최악의 경우, 생성된 콘텐츠의 100%가 온라인에 정확히 존재하는 경우도 확인되었습니다. 반면, 사람의 작성한 텍스트는 인터넷 데이터와 중복되는 비율이 훨씬 낮습니다. 적절한 프롬프트 사용이 평균적으로 비대립적 재생을 줄일 수 있지만, 최악의 경우 훈련 데이터 재생을 완화하기 위해서는 더 강력한 방어책이 필요하다는 것을 발견했습니다.



### Entropy and type-token ratio in gigaword corpora (https://arxiv.org/abs/2411.10227)
Comments:
          12 pages, 10 figures, 7 tables

- **What's New**: 이 연구에서는 영어, 스페인어, 터키어의 6가지 대규모 언어 데이터 세트를 분석하여 어휘 다양성(lexical diversity)을 측정하는 두 가지 주요 지표인 entropy와 type-token ratio (TTR) 간의 상관관계를 발견했습니다. 특히, 두 지표 간에 발견된 기능적 관계는 Zipf 법칙과 Heaps 법칙과도 연결되어 있습니다.

- **Technical Details**: 연구에서는 1억 개 이상의 단어를 포함한 6개의 대규모 텍스트 코퍼스를 사용하였으며, 여기에는 책, 뉴스 기사 및 트위터 데이터가 포함됩니다. 각 언어의 형태론적 특성을 고려한 결과, 영어, 스페인어, 터키어 간에 어휘 구조에서 차이를 발견했습니다. 동시에, 대규모 어휘의 경우에 대한 entropy의 분석적 표현을 도출하여 이를 코퍼스 데이터에 맞추어 분석했습니다.

- **Performance Highlights**: 연구 결과, entropy와 TTR 간의 관계는)에 대한 통찰을 제공하며, 이는 자연어 처리(NLP) 및 기계 학습(machine learning)의 성능 최적화와 데이터 압축 효율 향상 등 다양한 응용 분야에서 가치 있는 기초 자료로 활용될 수 있습니다.



### Increasing the Accessibility of Causal Domain Knowledge via Causal Information Extraction Methods: A Case Study in the Semiconductor Manufacturing Industry (https://arxiv.org/abs/2411.10172)
Comments:
          17 pages, 2 figures

- **What's New**: 이 논문은 반도체 제조 산업의 실제 산업 문서에서 인과 정보(causal information)를 자동으로 추출하는 방법을 개발한 연구를 소개합니다. 연구에서는 단일 단계 시퀀스 태깅(Single-stage Sequence Tagging, SST)과 다단계 시퀀스 태깅(Multi-stage Sequence Tagging, MST) 방법 두 가지를 제안하며, FMEA(Failure Mode and Effects Analysis)와 발표 슬라이드를 포함한 기존 문서들을 통해 성능을 평가했습니다.

- **Technical Details**: 본 연구는 비정형(unstructured) 및 반정형(semi-structured) 문서에서 인과 도메인 지식을 자동으로 추출하기 위해, 트랜스포머(transformer) 기반의 언어 모델을 활용한 시퀀스 태깅(sequence tagging) 방법을 채택하였습니다. 또한, FMEA와 같은 반정형 문서에서의 인과 관계를 잘 추출할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 MST 방법은 산업 문서에서 인과 정보를 추출하는 데 있어 실용적인 응용에 적합하며, FMEA 문서에서 93%의 F1 점수를 기록하였습니다. 발표 슬라이드에서 추출된 텍스트에 대해서는 73%의 F1 점수를 달성했습니다. 이 결과는 해당 방법이 산업 설정에서의 데이터 분석과 의사결정에 도움을 줄 가능성이 있음을 나타냅니다.



### Compound-QA: A Benchmark for Evaluating LLMs on Compound Questions (https://arxiv.org/abs/2411.10163)
- **What's New**: 이번 논문에서는 Compound Question Synthesis (CQ-Syn)를 소개하여 복합 질문(Compound Questions)을 활용한 새로운 성능 평가 벤치마크인 Compound-QA를 개발하였습니다. 이는 기존 QA 데이터셋을 기반으로 하여 여러 개의 하위 질문을 포함하는 질문으로 구성되어 있습니다.

- **Technical Details**: CQ-Syn 프레임워크를 사용하여 총 1,500개의 복합 질문을 생성하였으며, 이 질문들은 사실 진술(Factual-Statement), 인과 관계(Cause-and-Effect), 가설 분석(Hypothetical-Analysis), 비교 및 선택(Comparison-and-Selection), 평가 및 제안(Evaluation-and-Suggestion)이라는 다섯 가지 범주로 나뉘어 있습니다. 벤치마크는 LLM의 이해(understanding), 추론(reasoning), 지식(knowledge) 능력을 평가합니다.

- **Performance Highlights**: 오픈 소스 LLM 8종을 평가한 결과, 복합 질문에 대한 성능은 단일 질문(task)보다 현저히 낮았습니다. 그러나 복합 질문으로 보강된 지도 학습(fine-tuning)이 성능을 유의미하게 향상시킨다는 것을 발견하였습니다.



### An Effective Framework to Help Large Language Models Handle Numeric-involved Long-context Tasks (https://arxiv.org/abs/2411.10145)
- **What's New**: 이 논문에서는 기존의 대형 언어 모델(LLM)의 숫자 관련 긴 컨텍스트 작업 성능을 개선할 수 있는 새로운 워크플로우를 제안합니다. 이들은 대개 긴 텍스트를 다루는 데 강점을 보이지만, 수치 계산에서 성능이 크게 떨어지는 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 방법은 숫자 관련 긴 컨텍스트 작업을 4개의 저수준 소작업으로 분해합니다: 판단(judging), 추출(extracting), 코드 처리(processing with code) 및 결론(conclusion). 작은 모델을 사용하여 긴 컨텍스트를 효율적으로 처리하고, LLM이 생성한 코드를 사용하여 수치 계산을 수행합니다. 이러한 접근은 API 호출 비용을 크게 줄입니다.

- **Performance Highlights**: 2개의 숫자 관련 긴 컨텍스트 벤치마크 테스트에서 제안된 방법은 정확도를 향상시킬 뿐만 아니라 API 호출 비용을 현저히 절감하는 결과를 보였습니다. 이는 긴 컨텍스트 질문 응답(QA) 작업에 효과적이고 경제적인 프레임워크로 쉽게 적용될 수 있습니다.



### Legal Evalutions and Challenges of Large Language Models (https://arxiv.org/abs/2411.10137)
- **What's New**: 이번 논문에서는 OpenAI의 o1 모델을 사례로 LLMs(대형 언어 모델)의 법률 적용 성능을 평가하는 법률 테스트 방법을 검토합니다. 현재의 LLM 기술이 법률 용어 이해 및 적용에서 어떤 장단점을 가지고 있는지를 분석합니다.

- **Technical Details**: LLMs는 딥 러닝 기술의 혁신으로 자연어 처리(NLP) 분야에서 비약적인 발전을 이루었으며, OpenAI의 GPT 시리즈와 같은 모델들이 기계 번역 및 질문-응답 등의 전통적인 NLP 작업에서 뛰어난 성능을 보여주고 있습니다. 이 논문에서는 영문과 중문 법률 사건을 체계적으로 테스트하고 평가합니다.

- **Performance Highlights**: 실험 결과는 LLM의 법률 적용에서의 잠재력과 한계를 강조하며, 법률 언어 해석 및 법적 추론의 정확성에 관련된 도전 과제가 드러납니다. 또한, 다양한 모델의 장단점을 포괄적으로 분석하여 AI의 법률 분야 적용에 대한 귀중한 통찰력을 제공합니다.



### Xmodel-1.5: An 1B-scale Multilingual LLM (https://arxiv.org/abs/2411.10083)
- **What's New**: Xmodel-1.5라는 새로운 10억 매개변수의 다국어 대형 모델이 소개되었습니다. 이 모델은 약 2조 개의 토큰으로 사전 훈련되어 태국어, 아랍어, 프랑스어를 포함한 여러 언어에서 뛰어난 성능을 보입니다. 또한, Chulalongkorn 대학교에서 학생들이 주석을 단 태국어 평가 데이터셋을 공개하여 연구 커뮤니티에 기여하고 있습니다.

- **Technical Details**: Xmodel-1.5는 다국어 자연어 처리(NLP)의 필요성을 해결하기 위해 개발되었으며, 1억 개의 매개변수를 가진 모델은 다양한 소스에서 수집된 데이터를 기반으로 사전 훈련되었습니다. 이 모델의 데이터에는 고급 이전 모델에서 확장한 데이터와 Multilang Wiki 및 CulturaX 데이터가 포함되어 있으며, 낮은 자원 언어에 대한 집중적인 훈련이 이루어졌습니다. 이러한 과정은 효과적인 토큰화를 위해 unigram 토크나이저를 사용하여 진행되었습니다.

- **Performance Highlights**: Xmodel-1.5는 태국어, 아랍어, 프랑스어뿐만 아니라 중국어와 영어에서도 뛰어난 성능을 발휘하며, 전반적으로 모델의 성능 향상을 보여주고 있습니다. 이 모델은 특히 저자원 언어에 대한 처리 능력을 강화했으며, 커뮤니케이션의 다양성을 더 잘 반영할 수 있는 AI 시스템의 필요성을 충족시키고 있습니다.



### Understanding The Effect Of Temperature On Alignment With Human Opinions (https://arxiv.org/abs/2411.10080)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)의 의견 분포 및 인간 의견과의 정렬(alignment)에 대한 새로운 방법론을 제시하고 있습니다. 특히, 샘플링 및 로그 확률(log probability) 접근법을 적용하여 주관적 분류 작업에서 더 나은 결과를 도출할 수 있음을 보여주고 있습니다.

- **Technical Details**: 우리는 OpenAI의 gpt-3.5-turbo 모델을 활용하여 다양한 온도 설정(temperature settings)에서 Monte Carlo 및 Log Probability 추정 방법을 적용하였습니다. 실험은 101개의 예로 제한되었으며, 여러 평가지표(예: cross-entropy, Jensen-Shannon divergence)를 통해 성능을 평가하였습니다.

- **Performance Highlights**: 연구 결과, 온도를 조절하면서 얻은 샘플링 및 로그 확률 접근 방식이 직접 프롬프트보다 주관적 작업에서 더 일관된 의견 분포를 생성하는 것으로 나타났습니다. 이 결과는 LLM의 인간 의견 반영이 한계가 될 수 있음을 시사하며, 인간의 주관성이 모델의 불확실성에 미치는 영향을 추가로 연구할 필요성을 강조합니다.



### Layer Importance and Hallucination Analysis in Large Language Models via Enhanced Activation Variance-Sparsity (https://arxiv.org/abs/2411.10069)
Comments:
          20 pages, 5 figures

- **What's New**: 본 논문은 큰 언어 모델(LLMs)에서 다양한 레이어의 중요성을 평가하는 새로운 방법론인 Activation Variance-Sparsity Score (AVSS)를 제안합니다. 이 메트릭은 레이어의 정규화된 활성화 분산과 희소성을 결합하여 각 레이어의 기여도를 정량화합니다.

- **Technical Details**: AVSS는 레이어의 활성화 분산(activation variance)과 희소성(sparsity)을 계산하여, 전체 모델 성능에서 각 레이어의 역할을 측정하는 방법입니다. 실험을 통해 AVSS 기반으로 하여 가장 영향을 덜 미치는 25	op%(percent) 레이어를 제거하더라도 성능의 90	op%(percent) 이상을 유지할 수 있음을 입증하였습니다. 또한, Hallucination-Specific Activation Variance (HSAV)와 Hallucination-Specific Sparsity (HSS)라는 새로운 지표를 도입하여, 환각(hallucination)에 취약한 레이어를 정밀하게 식별합니다.

- **Performance Highlights**: 제안된 메서드는 NQ, SciQ, TriviaQA, TruthfulQA, WikiQA 데이터셋에서 실험하며, 최고 12	op%(percent) 성능 향상을 달성하였습니다. 이러한 결과는 LLM의 레이어 중요성 평가 및 환각 완화에 대한 포괄적인 프레임워크를 제공합니다.



### Information Extraction from Clinical Notes: Are We Ready to Switch to Large Language Models? (https://arxiv.org/abs/2411.10020)
- **What's New**: 이 연구는 오픈 소스 대규모 언어 모델(LLMs)을 사용하여 임상 정보 추출(IE) 시스템을 개발하고 평가한 최초의 연구 중 하나로, LLaMA 모델이 BiomedBERT보다 임상 Named Entity Recognition (NER) 및 Relation Extraction (RE) 작업에서 우수한 성능을 보였지만, 더 높은 계산 비용과 낮은 처리량을 요구한다는 점을 밝혀냈습니다.

- **Technical Details**: 연구에서는 1,588개의 임상 노트를 사용하여 4가지 주요 임상 개체(의료 문제, 검사, 약물 및 기타 치료)와 16가지 수정자를 아우르는 포괄적인 주석 기반 코퍼스를 개발했습니다. 두 개의 핵심 IE 작업인 NER과 RE를 수행하기 위해 LLaMA-2 및 LLaMA-3 모델을 instruction-tuning한 후, BiomedBERT와의 성능 비교를 시행했습니다. LLaMA 모델은 다양한 데이터셋에서 BiomedBERT를 지속적으로 초과 성능을 기록했습니다.

- **Performance Highlights**: LLaMA 모델은 충분한 훈련 데이터가 있는 경우 NER에서 1%, RE에서 1.5-3.7%의 미미한 개선을 보였으나, 제한된 훈련 데이터에서 더욱 큰 향상을 관찰했습니다. 특히, 보지 못한 i2b2 데이터셋에서는 LLaMA-3-70B가 NER에서 7% (F1), RE에서 4% (F1) 더 나은 성과를 올렸습니다. 그러나 LLaMA 모델은 더 많은 컴퓨팅 자원을 요구하고 BiomedBERT에 비해 최대 28배 느리게 작동했습니다.



### Once More, With Feeling: Measuring Emotion of Acting Performances in Contemporary American Film (https://arxiv.org/abs/2411.10018)
Comments:
          Accepted CHR 2024

- **What's New**: 이 논문에서는 나래티브 영화의 연기(acting performance)를 컴퓨터적으로 탐구하며, 대화 감정 인식 모델과 변이론적 사회언어학(variationist sociolinguistics) 분석 프레임워크를 사용하여 현대 미국 영화의 감정적 성과를 분석합니다.

- **Technical Details**: 영화 대사와 감정 간의 관계를 파악하기 위해, 우리는 연기 성과(utterances)와 각 대사(phrases)의 감정을 포함하는 일치 데이터셋을 구축했습니다. Wav2Vec2 모델을 사용하여 음성 감정 인식을 수행하고, 영화 대화에서 발견한 7가지 감정을 분류합니다. 또한, 우리는 대화의 맥락에 따라 감정을 예측하기 위해 양방향 LSTM(bidirectional LSTM) 모델을 사용합니다.

- **Performance Highlights**: 기타 주요 발견으로는 영화의 내러티브 시간에 따른 감정의 구조적 분석, 시간에 따른 감정의 변천을 알아보는 다층적인 성과 분석이 있습니다. 이를 통해 장르 및 대화 기반 제약이 연기 성과에 미치는 영향을 조사하였습니다.



### Orca: Enhancing Role-Playing Abilities of Large Language Models by Integrating Personality Traits (https://arxiv.org/abs/2411.10006)
- **What's New**: Orca는 성격 특성을 통합하여 맞춤형 언어 모델(LLM)을 교육하고 데이터 처리하는 프레임워크입니다. 이 프레임워크는 사용자 성격 특성을 추론하는 단계를 포함하여, 데이터 증강, 데이터셋 구축 및 모델링 및 교육을 통해 LLM의 롤플레잉 능력을 향상시킵니다.

- **Technical Details**: Orca는 성격 기반 지침 프로밍(PCIP) 방법론과 성격 기반 지침 조정(PTIT 및 PSIT)을 활용하여 LLM의 훈련 데이터를 생성합니다. 이 과정에서는 500명 사용자로부터 수집된 200개의 소셜 미디어 게시물을 바탕으로 성격 특성을 유추합니다. OrcaBench라는 다각적 평가 기준을 구축하여 소셜 AI 캐릭터가 생성한 콘텐츠 품질을 측정합니다.

- **Performance Highlights**: 제안된 모델은 생성된 콘텐츠 품질을 평가하는 OrcaBench 기준에서 뛰어난 성능을 보여주었습니다. 이 모델은 성격 특성을 효과적으로 인식하고 롤플레잉 능력을 현저히 개선하는 데 성공했습니다.



### HistoLens: An LLM-Powered Framework for Multi-Layered Analysis of Historical Texts -- A Case Application of Yantie Lun (https://arxiv.org/abs/2411.09978)
- **What's New**: 본 논문은 HistoLens라는 다층적 분석 프레임워크를 제안하며, 역사적 텍스트 분석에 LLM(대형 언어 모델)을 활용합니다. 서한(西漢)의 '염철론(鹽鐵論)'을 사례 연구로 사용하여 역사 연구와 교육에 대한 잠재적 응용을 보여줍니다.

- **Technical Details**: HistoLens는 자연어 처리(NLP) 기술, 특히 LLM을 통합하여 역사적 텍스트를 분석하는 과정에서 주제어 빈도 분석, 인명 및 관계 인식, 지식 그래프 구축, 시공간 분석, 이념 분석 및 머신 티칭 시나리오 구성의 주요 단계를 포함합니다.

- **Performance Highlights**: HistoLens는 '염철론'을 분석하여 유교와 법가의 이념이 서한의 정치, 경제, 군사 및 민족에 미친 영향을 다층적이고 정량적으로 탐구합니다. 이 연구는 역사적 텍스트의 심층 분석을 위한 LLM 보조 도구를 제공하며 교육 및 연구에서 혁신을 촉진하는 데 기여합니다.



### Large Language Models as User-Agents for Evaluating Task-Oriented-Dialogue Systems (https://arxiv.org/abs/2411.09972)
- **What's New**: 이 연구는 Task-oriented Dialogue (TOD) 시스템을 효과적으로 평가하기 위해 대화형 사용자 시뮬레이터를 개발하고, 이를 대형 언어 모델(LLMs)을 활용하여 자동화된 평가 프레임워크로 구현합니다. 기존의 데이터셋 기반 평가 방식이 가지고 있는 한계를 극복함으로써, 보다 다양하고 현실적인 대화 시나리오를 평가할 수 있습니다.

- **Technical Details**: 작업에 적합한 사용자 시뮬레이터를 구축하기 위해 LLM을 활용하여 초기 사용자 목표에 따라 LLM에게 프롬프트를 제공하고, 대화 흐름을 관리합니다. 사용하는 프롬프트 전략에는 기본적인 지침을 활용한 Vanilla Prompt, 추론 단계를 포함한 Thought Prompt, 사용자 상태를 추적하여 회화를 종료하지 않도록 돕는 User State Tracking Prompt가 있습니다.

- **Performance Highlights**: 제안된 사용자 시뮬레이터를 통해 TOD 시스템의 성과를 다양성과 과업 완수 지표로 평가한 결과, 더 나은 프롬프트 사용시 성과가 개선됨을 보여주었습니다. 이를 통해 지금까지의 전통적인 평가 방법의 한계를 극복하고, 실제와 유사한 대화 상호작용을 반영한 평가 프로세스를 강화하였습니다.



### LoRA-LiteE: A Computationally Efficient Framework for Chatbot Preference-Tuning (https://arxiv.org/abs/2411.09947)
- **What's New**: 본 연구는 효율적 선호 조정(preference tuning)을 위한 새로운 프레임워크인 LoRA-Lite Ensemble(LoRA-LiteE)을 소개합니다. LoRA-LiteE는 Supervised Fine-tuning (SFT), Low-Rank Adaptation (LoRA), 및 Ensemble Learning을 통합하여 가벼운 모델의 예측을 효과적으로 집계합니다.

- **Technical Details**: LoRA-LiteE 프레임워크는 파라미터 효율적인 LoRA 기법을 사용하여 훈련 가능한 파라미터 수를 현저히 줄입니다. 이를 통해 제한된 계산 자원에서도 모델 훈련이 가능하게 됩니다. 또한, 다양한 모델의 보완적 장점을 활용하여 전체 시스템 성능을 향상시키는 멀티모델 앙상블 전략을 채택하였습니다.

- **Performance Highlights**: LoRA-LiteE 모델은 다양한 규모의 기본 모델 및 RLHF로 훈련된 GPT-4와 비교하여, 자원 제약 환경에서 단일 대형 모델보다 성능이 우수하며, 일반적으로 튜닝되지 않은 GPT-4와 유사한 성능을 나타냅니다. 이는 LoRA-LiteE가 자원 제약이 있는 환경에서도 선호에 맞춘 챗봇 시스템의 확장성과 접근성을 높일 수 있는 효과적인 방법임을 보여줍니다.



### SlimLM: An Efficient Small Language Model for On-Device Document Assistanc (https://arxiv.org/abs/2411.09944)
- **What's New**: 이번 연구에서는 모바일 장치에서 문서 지원 작업을 위해 최적화된 소형 언어 모델(SLM) 시리즈인 SlimLM을 제안합니다. SlimLM은 삼성 Galaxy S24에서의 실행성을 고려하여 모델 크기(125M에서 7B 파라미터), 컨텍스트 길이 및 추론 시간 간의 최적의 균형을 찾았습니다.

- **Technical Details**: SlimLM은 SlimPajama-627B로 사전 학습되었으며, DocAssist라는 특별히 구성된 데이터셋으로 세밀 조정되었습니다. 이 모델은 125M에서 1B 파라미터까지 다양하며, 문서 요약(SUMM), 질문 제안(QS), 질문 응답(QA)과 같은 작업을 위해 개발되었습니다. 모델의 성능 측정은 BLEU, ROUGE, Semantic Textual Similarity (STS)와 같은 표준 메트릭을 활용했습니다.

- **Performance Highlights**: SlimLM의 가장 작은 모델인 SlimLM-125M은 S24에서 효율적인 성능을 보여주며, 더 큰 모델들은 여전히 모바일 제약 내에서 향상된 기능을 제공합니다. SlimLM은 기존의 비슷한 크기의 SLM과 비교할 때 유사하거나 우수한 성능을 나타내었으며, 향후 모바일 언어 모델 연구의 기준이 될 것입니다.



### Refined and Segmented Price Sentiment Indices from Survey Comments (https://arxiv.org/abs/2411.09937)
Comments:
          Accepted to IEEE BigData 2024. 9 pages, 11 tables, 1 figure

- **What's New**: 본 연구는 소비자와 비즈니스 관점에서 가격 동향을 더 정교하게 이해하고 가격 감정 지수를 강화하는 것을 목표로 합니다. 또한, 일본 내각부의 Economy Watchers Survey에서 가격 관련 코멘트를 추출하여 대형 언어 모델(LLM)을 활용하여 가격 동향을 분류합니다.

- **Technical Details**: 우리는 조사 샘플이 소비자 또는 비즈니스의 관점을 반영하는지, 그리고 이 코멘트가 상품(goods) 또는 서비스(services)와 관련이 있는지를 분류하기 위해 코멘트의 분야와 응답자의 산업 정보를 이용합니다. LLM을 사용하여 가격 관련 코멘트를 더 정확하게 분류하며, 여러 LLM의 출력을 통합함으로써 분류의 성능을 향상시킬 수 있는 가능성을 제시합니다.

- **Performance Highlights**: 보다 정확하게 분류된 코멘트를 사용하면 기존 지수보다 더 높은 상관관계를 갖는 지수를 구축할 수 있습니다. 소비자 가격 지수의 상관관계는 표본 크기가 더 큰 코멘트를 산업별로 집계하여 선택함으로써 더욱 강화됩니다.



### Research on Domain-Specific Chinese Spelling Correction Method Based on Plugin Extension Modules (https://arxiv.org/abs/2411.09884)
- **What's New**: 이 논문은 도메인별 텍스트 처리의 한계를 해결하기 위해 플러그인 확장 모듈을 기초로 한 중국어 맞춤법 교정 방법을 제안합니다. 기존의 모델들은 일반 도메인 데이터셋에서 훈련되어, 전문 용어가 포함된 특정 텍스트에서 성능 저하를 겪고 있습니다.

- **Technical Details**: 제안된 방법은 도메인별 전문 용어의 특성을 학습하여 모델의 맞춤법 교정 능력을 향상시키는 확장 모듈을 포함합니다. 이 확장 모듈은 다양한 신경망 기반 교정 모델과 호환되며, 새로운 도메인으로 확장할 때 도메인별 용어 리스트와 약간의 Fine-tuning만으로 가능하도록 설계되었습니다.

- **Performance Highlights**: 의료, 법률 및 공식 문서 도메인을 위한 확장 모듈을 통합한 실험 결과, 제안된 모델은 확장 모듈이 없는 기준 모델에 비해 교정 성능이 상당히 향상되었습니다.



### KULCQ: An Unsupervised Keyword-based Utterance Level Clustering Quality Metric (https://arxiv.org/abs/2411.09853)
- **What's New**: 본 연구에서는 대화형 데이터의 클러스터링 품질을 평가하기 위해 키워드 기반 발화 수준 클러스터 품질(KULCQ)이라는 비지도형(metric) 메트릭을 도입하였다. 기존 클러스터링 메트릭은 데이터의 기하학적 구조에 초점을 맞추고 있었으나, KULCQ는 대화의 언어적 뉘앙스를 고려하여 발화 간의 의미적 관계를 좀 더 잘 포착한다.

- **Technical Details**: KULCQ는 사용자 쿼리 발화의 클러스터링을 평가하기 위해 키워드 분석을 활용한다. 문장 내에서 사용되는 단어의 차이에 관계없이 비슷한 의도를 파악할 수 있게 해준다. 이에 따라 KULCQ는 클러스터의 중심을 정의하고, 클러스터 내 및 클러스터 간 거리 기반으로 클러스터 품질을 평가한다. 실험에서는 코사인 거리(cosine distance)를 사용하였다.

- **Performance Highlights**: KULCQ 메트릭은 기존 비지도 클러스터링 메트릭과 비교할 때 대화 데이터의 의미적 관계를 보다 잘 포착하는 것으로 나타났다. 포괄적인 실험을 통해 KULCQ가 대화 데이터에 특화된 클러스터 품질 측정에 있어 우수성을 입증하였다.



### A Benchmark for Long-Form Medical Question Answering (https://arxiv.org/abs/2411.09834)
Comments:
          AIM-FM: Advancements in Medical Foundation Models Workshop, 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문에서는 장기적인 의료 질문 응답(QA)에 대한 대규모 언어 모델(LLMs)을 평가하기 위한 신뢰할 수 있는 벤치마크가 부족하다는 문제를 다루고 있습니다. 기존 평가 벤치마크는 자동 메트릭스 및 객관식 질문에 중점을 두고 있어, 실제 임상 응용에서의 복잡성을 충분히 반영하지 못합니다. 새로 개발된 벤치마크는 실제 소비자의 의료 질문을 포함하며 의료 전문가의 주석이 달린 장기 답변 평가가 포함되어 있습니다.

- **Technical Details**: 연구팀은 Lavita Medical AI Assist 플랫폼에서 수집된 사용자 질문을 통해 실제 소비자 의료 질문의 데이터셋을 구축했습니다. 총 4,271개의 질문이 수집되었고, 인간 주석가와 GPT-4를 이용해 의료 관련 질문을 검토하였으며, 최종적으로 1,298개의 고품질 의료 질문을 확보했습니다. 이 연구는 다양한 개방형 및 폐쇄형 LLM의 응답을 기반으로 정확성, 유용성, 해로운 정보 여부 및 편향을 평가했습니다.

- **Performance Highlights**: 프리미너리 결과는 개방형 LLM이 폐쇄형 모델보다 의료 QA에서 강력한 잠재력을 지니고 있음을 보여주었습니다. 인간 평가자와 LLM의 평가 결과 간의 일치도를 분석하였으며, 개방형 LLM이 높은 정확성을 보임을 확인했습니다.



### Evaluating Gender Bias in Large Language Models (https://arxiv.org/abs/2411.09826)
Comments:
          13 pages, 12 figures, 1 table

- **What's New**: 이 연구는 대화 중심 응용에서 사용되는 언어 모델(Large Language Models, LLMs)의 성별 편향(Gender bias)을 분석하고, 특히 직업적 맥락에서 대명사(pronoun) 선택에서 나타나는 성별 편향을 평가했습니다.

- **Technical Details**: 연구에서는 GPT-4, GPT-4o, PaLM 2 Text Bison, 그리고 Gemini 1.0 Pro 모델을 검토하였으며, 직업의 성별 분포를 반영하기 위해 세 가지 문장 처리 방법(masked tokens, unmasked sentences, sentence completion)을 사용했습니다. 고용 관련 데이터에 따른 성별 분포를 바탕으로 모델의 대명사 선택을 분석했습니다.

- **Performance Highlights**: 결과는 미 노동력 데이터에서 나타나는 성별 분포와 모델의 대명사 선택이 긍정적인 상관관계를 가지고 있음을 보여주었습니다. 특히, 문장 완성(sentence completion) 방법이 실제 성별 분포와 가장 강한 상관관계를 보였고, 성별 편향을 다룰 때 모델 선택보다 프롬프트 방법(prompting method)이 더 큰 영향을 미친다는 점이 부각되었습니다.



### Llama Guard 3 Vision: Safeguarding Human-AI Image Understanding Conversations (https://arxiv.org/abs/2411.10414)
- **What's New**: 본 논문에서는 이미지 이해를 포함한 다중 모달 LLM 기반의 대화 안전 장치인 'Llama Guard 3 Vision'을 소개합니다. 이 모델은 다중 모달 LLM 입력(프롬프트 분류)과 출력(응답 분류)의 안전성을 보장할 수 있도록 설계되었습니다.

- **Technical Details**: Llama Guard 3 Vision은 Llama 3.2-Vision을 기반으로 미세 조정(fine-tuning)되었으며, MLCommons 분류법을 사용하여 13가지 위험 카테고리에서 강력한 성능을 나타냅니다. 특히, 다양한 이미지 및 텍스트 프롬프트에 대한 해로운 내용을 감지하기 위해 최적화되어 있습니다.

- **Performance Highlights**: Llama Guard 3 Vision은 내부 벤치마크 테스트를 통해 기존의 텍스트 전용 모델들과 비교하여 응답 분류 작업에서 더 높은 강인성을 보였습니다. 이 모델은 실제 공격 시나리오에서 프롬프트 기반 공격을 효과적으로 무시하면서 안전성 분류를 수행하는데 주력하고 있습니다.



### Features that Make a Difference: Leveraging Gradients for Improved Dictionary Learning (https://arxiv.org/abs/2411.10397)
Comments:
          9 pages, 8 figures. Submitted to NAACL 2025

- **What's New**: 본 논문에서는 Gradient Sparse Autoencoders (g-SAEs)를 도입하여 기존의 Sparse Autoencoders (SAEs)에서 발생하는 한계를 극복하고자 합니다. g-SAEs는 활성화 값뿐만 아니라 입력 활성화의 gradient를 고려하여 $k$개의 요소를 선택하는 TopK 활성화 기능을 수정하여, 모델 출력에 강하게 영향을 미치는 특징을 학습할 수 있도록 합니다.

- **Technical Details**: g-SAEs는 기존 SAE 아키텍처를 개선하여 gradient-aware TopK 활성화 기능을 활용합니다. 이 방법이 모델의 다운스트림 효과를 더 정확하게 포착할 수 있도록 해주며, 그 결과로 생성된 재구성을 원본 네트워크 성능에 더 충실합니다. 또한 g-SAEs는 더 적은 수의 비활성 단위를 가지고 있으며, 활성화의 gradient를 포함하여 여러 인기 있는 SAE 아키텍처의 Pareto frontiers에서 개선을 보여줍니다.

- **Performance Highlights**: g-SAEs는 기존 아키텍처와 비슷한 해석 가능성을 유지하면서, 특정 logits에 대해 더 많은 영향을 미치는 latents를 복구합니다. 이는 g-SAEs가 모델에 더 온전한 제어를 가능하게 하며, 다양한 맥락에서 향상된 모델 성능 조작을 제공합니다.



### Safe Text-to-Image Generation: Simply Sanitize the Prompt Embedding (https://arxiv.org/abs/2411.10329)
- **What's New**: 최근 Text-to-Image (T2I) 생성 모델의 발전으로 고품질 이미지를 텍스트 설명에 맞춰 생성할 수 있게 되었습니다. 하지만 이러한 모델은 불건전한 생성의 위험이 존재하여, 사용 정책을 위반하는 유해 콘텐츠를 생성할 수 있습니다. 기존의 안전 생성 방법은 주로 비주얼 표현에서 바람직하지 않은 개념을 삭제하는 데 초점을 맞추었지만, 텍스트 표현의 정화는 간과했습니다. 본 논문에서는 Prompt Embeddings에서 바람직하지 않은 개념을 지우는 Vision-Agnostic 안전 생성 프레임워크인 Embedding Sanitizer (ES)를 제안합니다.

- **Technical Details**: Embedding Sanitizer (ES)는 텍스트 인코더의 출력에 적용되는 플러그 앤 플레이 모듈로, 내부 점수 네트워크를 통해 각각의 토큰이 가질 수 있는 유해성을 평가합니다. 이 시스템은 유해 점수가 높은 토큰에 대해 강하게 정화 작업을 수행하고, 점수가 낮은 토큰에 대해서는 최소한의 영향을 미치도록 조정을 진행합니다. ES는 지정된 목표 개념을 지우기 위한 교육 과정에서 다양한 맥락 샘플을 생성하여 전반적인 안전성을 높입니다.

- **Performance Highlights**: ES는 다섯 개의 프롬프트 벤치마크에서의 평가에서 아홉 개의 기준 방법과 비교하여 SOTA (State-of-the-Art) 강인성을 달성하였습니다. ES는 기존의 안전 장치들에 비해 해석 가능성과 통제력을 제공하며, 생성 품질을 유지하는 동시에 유해 콘텐츠 생성의 주요 원천인 프롬프트 임베딩의 정화를 통해 우수한 성능을 발휘합니다.



### The Dawn of GUI Agent: A Preliminary Case Study with Claude 3.5 Computer Us (https://arxiv.org/abs/2411.10323)
Comments:
          40 pages, 21 figures, preprint

- **What's New**: Claude 3.5 Computer Use는 공공 베타로 제공되는 GUI 에이전트를 갖춘 최초의 최전선 AI 모델로, 실세계의 복잡한 환경에서 컴퓨터 활용 능력이 아직 잘 알려지지 않았다.

- **Technical Details**: Claude 3.5는 user instruction을 기반으로 API 호출을 통해 end-to-end 솔루션을 제공하며, GUI 상태를 시각적으로 관찰하여 행동을 생성한다. GUI 상호작용을 통해 사용자의 요구를 충족 시키기 위해 'vision-only' 접근 방식을 채택하고 있으며, 이 과정에서 컴퓨터 툴, 텍스트 에디터 툴, Bash 툴 등의 도구가 사용된다.

- **Performance Highlights**: 이 사례 연구는 Claude 3.5의 성능을 여러 사용자 그룹의 요구를 반영하는 다양한 데스크톱 작업 자동화 과제를 통해 평가하였으며, 이에 대한 정량적 및 정성적 평가가 이루어졌다. 이 연구는 Claude의 GUI 자동화 모델의 능력과 한계를 밝혀내고 향후 개선 사항에 대한 논의를 이끈다.



### Scaling Law for Post-training after Model Pruning (https://arxiv.org/abs/2411.10272)
- **What's New**: 이번 논문은 깊이 가지치기(depth pruning), 너비 가지치기(width pruning), 및 2:4 반구조화 가지치기(semi-structured pruning)를 통해 가지치기된 대규모 언어 모델(LLMs)의 후속 훈련(post-training) 요구 사항을 조사하고, 최적의 후속 훈련 데이터 양을 결정하기 위한 스케일링 법칙(scaling law)을 제시합니다.

- **Technical Details**: 연구에서는 Llama-3 및 Qwen-2.5 모델 시리즈에 대해 후속 훈련 실험을 수행하며, 높은 가지치기 비율이 성능 회복을 위해 더 많은 후속 훈련 데이터를 필요로 한다는 것을 발견했습니다. 특히, 더 큰 LLM은 더 적은 데이터로 성능을 회복할 수 있다는 점에서 기존의 직관과 반대된다는 것을 확인했습니다. 제안된 스케일링 법칙은 가지치기 전후의 모델 파라미터 수 및 후속 훈련 토큰 수를 바탕으로 모델의 손실(loss)을 예측할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과에 따르면, Llama-3.1-8B 모델은 16% 가지치기 비율에서 약 1B 토큰에 대해 손실 곡선이 수렴하는 반면, 24% 및 33% 가지치기 비율에서는 더 많은 후속 훈련 데이터가 필요하다는 점이 강조되었습니다. 또한, 2:4 반구조화 가지치기를 적용한 경우 더 큰 모델은 성능 회복을 위해 상대적으로 적은 후속 훈련 데이터를 요구한다는 것을 보여주었습니다.



### Scaling up the Evaluation of Collaborative Problem Solving: Promises and Challenges of Coding Chat Data with ChatGP (https://arxiv.org/abs/2411.10246)
Comments:
          21 pages, 3 figures, 5 tables. Initially report in the edArXiv:xw6kz

- **What's New**: 이 논문은 협력적 문제 해결(Collaborative Problem Solving, CPS)에서 ChatGPT를 사용하여 직접적으로 대화 데이터를 코딩하는 방법을 보고하였습니다.

- **Technical Details**: 여러 개의 데이터셋과 코딩 프레임워크를 통해 성능을 벤치마킹하여 ChatGPT 기반 코딩의 효과를 평가하였습니다. 이는 CPS 연구의 효율적인 확장을 위한 새로운 접근 방식을 제시합니다.

- **Performance Highlights**: ChatGPT 기반 코딩은 구어체적인 언어가 사용된 논의에서는 사람의 코딩보다 우수했지만, 전문적인 과학 용어와 맥락이 포함된 논의에서는 부족한 성능을 보였습니다. 이 발견은 연구자들이 CPS 데이터 분석을 위한 효율적이고 확장 가능한 전략을 개발할 수 있는 실제적인 지침을 제공합니다.



### Evaluating the role of `Constitutions' for learning from AI feedback (https://arxiv.org/abs/2411.10168)
Comments:
          4 pages, 2 figures. In NeurIPS 2024 Workshop on Language Gamification

- **What's New**: 대형 언어 모델(LLMs)의 발전으로 인해 인간 피드백을 대체하여 다른 LLM을 학습하고 평가하는 데 사용되고 있습니다. 본 연구에서는 의사소통 개선을 위해 4개의 다른 'constitution'을 사용하여 피드백의 질이 어떻게 영향을 받는지를 조사했습니다.

- **Technical Details**: 연구에서는 215명의 인간 평가자가 수행한 쌍대 비교(pairwise comparisons)를 통해 각 constitution에 따른 피드백 질을 비교했습니다. 그 결과, 상세한 constitution이 감정적 품질(emotive qualities) 측면에서 더 나은 결과를 나타냈지만, 정보 수집 및 제공과 같은 실용적 기술을 학습하는 데는 어떤 constitution도 베이스라인을 초과하지 못했습니다.

- **Performance Highlights**: 결론적으로, 상세한 constitution이 우선시되어야 하지만, 특정 영역에서 AI 피드백이 보상 신호(reward signal)로서의 효과성에 한계가 있음을 알 수 있습니다.



### Prompting and Fine-tuning Large Language Models for Automated Code Review Comment Generation (https://arxiv.org/abs/2411.10129)
- **What's New**: 이번 연구는 오픈 소스 대형 언어 모델(LLM)을 파라미터 효율적인 Quantized Low-Rank (QLoRA) 방식으로 조정하여 코드 리뷰 주석 생성을 개선하는 방법을 제안합니다. 또한, 프로프라이어터리 LLM에 함수 호출 그래프와 코드 요약을 추가하여 성능을 향상시키는 새로운 방법론을 탐구하고 있습니다.

- **Technical Details**: 연구에서 사용된 메소드에는 QLoRA라는 파라미터 효율적인 미세 조정 방식이 포함되어 있으며, GPT 아키텍처에 기반한 디코더 전용 LLM들이 비약적인 성능을 보일 수 있음이 언급되었습니다. 본 연구는 코드 리뷰 활동을 위한 주석 생성에서 함수 호출 그래프와 코드 요약의 효과를 조사하고 있습니다.

- **Performance Highlights**: 실험 결과, GPT-3.5 모델에서 함수 호출 그래프를 추가한 Few-Shot 프롬프트가 CodeReviewer 데이터셋에서 사전 훈련된 기준보다 약 90% BLEU-4 점수를 초과했습니다. 또한, QLoRA로 미세 조정된 Code Llama와 Llama 3.1 모델이 이 작업에서 25%에서 83%의 성과 향상을 달성했습니다.



### Memorization in Attention-only Transformers (https://arxiv.org/abs/2411.10115)
Comments:
          16 pages, 6 figures, submitted to AISTATS 2025,

- **What's New**: 이 논문에서는 멀티-헤드 어텐션에 대한 메모리 용량을 탐구하면서 기존 연구의 한계를 넘어서는 새로운 증명을 제시합니다. 특히, 이 연구는 모든 컨텍스트 크기에 대해 언어 기반 Transformers의 메모리 용량을 확장하는 것을 목표로 하고 있습니다.

- **Technical Details**: 연구는 Attention-only Transformer (AoT)에 대한 메모리 용량을 정의하고 두 가지 유형의 메모리 작업인 연관 작업과 분포 작업을 구분합니다. 연관 작업은 입력 토큰 시퀀스를 기반으로 다음 토큰을 예측하는 것이고, 분포 작업은 입력 시퀀스에 대해 최적의 확률 분포를 예측하는 것을 포함합니다. 이 연구는 Kullback-Leibler (KL) Divergence를 사용하여 분포 작업을 평가하며, AoT 모델의 성능을 이론적으로 분석합니다.

- **Performance Highlights**: 본 연구는 H개 헤드(H𝑓헤드)와 d𝑑숨_dimension을 가진 한 층 AoT를 통해 연관 작업에서 기존의 국한된 컨텍스트 창을 사용하는 결과보다 더 많은 연관성 집합을 정확하게 기억할 수 있다는 것을 증명합니다. 또한, 분포 작업의 경우 AoT가 일반적인 시퀀스 인코더의 분포와 유사한 성능을 보임을 보여줍니다.



### CMATH: Cross-Modality Augmented Transformer with Hierarchical Variational Distillation for Multimodal Emotion Recognition in Conversation (https://arxiv.org/abs/2411.10060)
- **What's New**: 이번 연구에서는 대화에서 다양한 모달리티 정보를 통해 감정을 인식하는 새로운 방법, 즉 Cross-Modality Augmented Transformer with Hierarchical Variational Distillation(이하 CMATH)를 제안합니다. CMATH는 모달리티 간의 상호 작용을 효과적으로 융합하고, 모달리티 정보의 다양한 품질을 고려하는 비대칭 융합 전략을 사용합니다.

- **Technical Details**: CMATH는 두 가지 주요 구성 요소로, Multimodal Interaction Fusion과 Hierarchical Variational Distillation이 있습니다. Multimodal Interaction Fusion 모듈 내의 CMA-Transformer는 각 모달리티를 중앙 모달리티로 설정하고 나머지 모달리티를 보조로 활용하여 대화 감정 인식의 정확도를 향상시킵니다. Hierarchical Variational Distillation 모듈은 각 모달리티의 세분화된 특징 표현을 보강하여 전체적인 정확도를 높이는 작업을 수행합니다.

- **Performance Highlights**: 실험 결과, CMATH 모델은 IEMOCAP과 MELD 데이터셋에서 기존의 최첨단 방법들보다 월등한 성과를 보였으며, 정확도와 가중치 F1 스코어 지표에서 각각 73.90% 및 73.96%를 기록하였습니다. 이로 인해 기존의 성능 기준인 AdalGN에 비해 각각 4.84% 및 4.55%의 개선을 이뤘습니다.



### Towards unearthing neglected climate innovations from scientific literature using Large Language Models (https://arxiv.org/abs/2411.10055)
Comments:
          10 pages. Accepted in the LatinX in AI workshop at NeurIPS 2024

- **What's New**: 이번 연구에서는 기후 변화 해결을 위한 혁신적인 기술들이 과학 문헌 내에 이미 존재하지만 활용되지 않고 있다는 가설을 세우고, 이를 최대한 활용하기 위해 OpenAlex 데이터베이스에서 수집한 논문들을 분석하였습니다.

- **Technical Details**: 연구팀은 기후 변화 완화 가능성, 기술 개발 단계, 배치 준비 상태 등 일곱 가지 차원에서 논문의 제목과 초록을 평가하기 위해 OpenAI의 LLM인 GPT4-o를 활용하였습니다. 이 과정에서 LLM의 결과를 인간 평가자와 비교하여 효과성을 분석하였습니다.

- **Performance Highlights**: 연구 결과, LLM 기반 모델이 인간 전문 지식을 보완하며, 기후 혁신을 조기에 발굴하는 데 유의미한 속도와 일관성을 제공할 수 있다는 것을 확인하였습니다. 이 연구는 미래의 기후 행동 전략 강화에 기여할 것으로 기대됩니다.



### JRadiEvo: A Japanese Radiology Report Generation Model Enhanced by Evolutionary Optimization of Model Merging (https://arxiv.org/abs/2411.09933)
Comments:
          Accepted by NeurIPS'24 Workshop on AIM-FM: Advancements In Medical Foundation Models: Explainability, Robustness, Security, and Beyond

- **What's New**: 이번 논문은 비의료 비전-언어 모델을 비영어 의료 텍스트 생성에 확장하기 위한 첫 번째 시도로, 진화 최적화(evolutionary optimization)를 통한 모델 병합(model merging) 기법을 사용하여 일본어(X-ray) 방사선 보고서 생성 모델(JRadiEvo)을 제안합니다.

- **Technical Details**: JRadiEvo는 비의료 비전-언어 모델, 의료 텍스트-투-텍스트 모델 및 일본어 텍스트-투-텍스트 모델을 진화 알고리즘을 통해 병합하여 개발되었습니다. 이 모델은 50개의 번역 샘플만으로 정확한 일본어 보고서를 생성할 수 있으며, 80억 개의 파라미터를 가진 경량 모델로, 병원 내에서 정상적으로 배포 가능하여 개인 정보 보호 문제를 해결합니다.

- **Performance Highlights**: JRadiEvo는 제한된 데이터(50개 사례)만을 사용하여 최근 연구의 최신 모델들보다 우수한 성능을 보여주었습니다. 이를 통해 모델의 피드백을 제거하여 학습 과정에서 비효율을 줄이고, 개인 정보를 보호해야 하는 의료 환경에서도 실용적인 응용이 가능하다는 점을 강조합니다.



### Evaluating the Predictive Capacity of ChatGPT for Academic Peer Review Outcomes Across Multiple Platforms (https://arxiv.org/abs/2411.09763)
- **What's New**: 이번 논문은 이전 연구들이 제시한 Large Language Models (LLMs)의 동료 리뷰 결과 예측 기능을 확장하였으며, 특별히 ChatGPT 점수의 평균화를 통한 더 강력한 예측 방법을 도입했습니다.

- **Technical Details**: 연구에서는 F1000Research의 경우 리뷰어 가이드라인에 기반하여 제출된 제목과 초록만을 사용하였을 때, 30개의 ChatGPT 예측을 평균내는 것이 동료 리뷰 결과를 예측하는 데 실패했다는 결과를 도출했습니다(Spearman's rho=0.00). SciPost Physics에서는 유효성(rho=0.25), 독창성(rho=0.25), 중요성(rho=0.20), 명확성(rho=0.08) 차원에서 약한 긍정적 상관관계를 보였고, ICLR의 경우 중간 정도의 긍정적 상관관계(rho=0.38)를 보였습니다. 전체 텍스트를 포함했을 때 ICLR의 경우 상관관계가 크게 증가(rho=0.46)했고, F1000Research에서도 약간 개선되었습니다(rho=0.09).

- **Performance Highlights**: 체인 오브 써트 (chain-of-thought) 시스템 프롬프트를 사용했을 때 F1000Research의 경우 상관관계가 약간 증가했으며(rho=0.10), ICLR의 경우 소폭 감소(rho=0.37)했습니다. SciPost Physics에서는 유효성(rho=0.16), 독창성(rho=0.18), 중요성(rho=0.18), 명확성(rho=0.05) 차원에서 더 감소했습니다. 전반적으로, ChatGPT는 일부 맥락에서 약한 사전 게재 품질 평가를 할 수 있지만, 이 평가의 효과와 최적 전략은 각 플랫폼, 저널 및 회의마다 크게 다릅니다.



New uploads on arXiv(cs.IR)

### KuaiFormer: Transformer-Based Retrieval at Kuaishou (https://arxiv.org/abs/2411.10057)
- **What's New**: 본 논문에서는 Kuaishou의 대규모 콘텐츠 추천 시스템에 적용된 새로운 Transformer 기반 검색 프레임워크인 KuaiFormer를 소개합니다. KuaiFormer는 기존의 점수 추정 작업에서 Transformer 기반의 다음 행동 예측 패러다임으로 전환하여 검색 프로세스를 근본적으로 재정의합니다.

- **Technical Details**: KuaiFormer는 여러 쿼리 토큰을 도입하여 사용자의 다양한 관심사를 포착하고 연속된 항목 시퀀스에서 별개의 사용자 관심 표현을 추출합니다. 또한, 효율성을 높이기 위해 조정 가능한 항목 압축 메커니즘을 포함하고 있습니다. 이러한 방법들은 긴 시퀀스를 효과적으로 모델링할 수 있도록 설계되었습니다.

- **Performance Highlights**: KuaiFormer는 Kuaishou의 단기 동영상 서비스에서 +0.360%/-0.126%/-0.411%의 온라인 시청 시간 증가를 기여하며, 실시간 대규모 추천 시스템에 적합한 최초의 Pure Transformer 구조 기반 검색 모델로 평가받고 있습니다.



### Towards unearthing neglected climate innovations from scientific literature using Large Language Models (https://arxiv.org/abs/2411.10055)
Comments:
          10 pages. Accepted in the LatinX in AI workshop at NeurIPS 2024

- **What's New**: 이번 연구에서는 기후 변화 해결을 위한 혁신적인 기술들이 과학 문헌 내에 이미 존재하지만 활용되지 않고 있다는 가설을 세우고, 이를 최대한 활용하기 위해 OpenAlex 데이터베이스에서 수집한 논문들을 분석하였습니다.

- **Technical Details**: 연구팀은 기후 변화 완화 가능성, 기술 개발 단계, 배치 준비 상태 등 일곱 가지 차원에서 논문의 제목과 초록을 평가하기 위해 OpenAI의 LLM인 GPT4-o를 활용하였습니다. 이 과정에서 LLM의 결과를 인간 평가자와 비교하여 효과성을 분석하였습니다.

- **Performance Highlights**: 연구 결과, LLM 기반 모델이 인간 전문 지식을 보완하며, 기후 혁신을 조기에 발굴하는 데 유의미한 속도와 일관성을 제공할 수 있다는 것을 확인하였습니다. 이 연구는 미래의 기후 행동 전략 강화에 기여할 것으로 기대됩니다.



### InterFormer: Towards Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction (https://arxiv.org/abs/2411.09852)
Comments:
          10 pages, 6 figures

- **What's New**: 이 논문에서는 Click-through rate (CTR) 예측을 위한 새로운 모듈 InterFormer를 제안하고 있습니다. InterFormer는 이질적인 정보 상호작용을 상호 유익한 방식으로 학습하여 CTR 예측의 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: InterFormer 모듈은 두 가지 주요 아이디어로 구성됩니다. 첫째, 서로 다른 모드 간에 양방향 정보 흐름을 가능하게 하여 글로벌 및 시퀀스 학습을 혼합 방식으로 수행합니다. 둘째, 정보를 과도하게 집계하지 않기 위해 각 데이터 모드에서 완전한 정보를 유지하고, 효과적인 정보 선택 및 요약을 위한 별도의 브리징 아치를 사용합니다.

- **Performance Highlights**: InterFormer는 세 개의 공개 데이터셋과 대규모 산업 데이터셋에서 최첨단 성능을 달성하였습니다. 논문에서 언급한 바에 따르면 AUC 기준으로 최대 0.14% 개선과 내부 대규모 데이터셋에서 0.15%의 Normalized Entropy (NE) 이득을 달성했습니다.



### Residual Multi-Task Learner for Applied Ranking (https://arxiv.org/abs/2411.09705)
- **What's New**: ResFlow는 e-commerce 플랫폼에서 사용되는 새로운 경량 멀티태스크 학습 프레임워크입니다. 이 방법은 후행 작업 네트워크의 해당 레이어 간에 잔여 연결(residual connections)을 생성하여 효율적인 정보 공유를 가능하게 합니다.

- **Technical Details**: ResFlow는 멀티태스크 학습(MTL)에서 효과적인 정보 전송을 지원하도록 설계되었습니다. 기존 멀티태스크 학습 방법이 가진 한계를 극복하기 위해 잔여 연결을 도입했습니다. 이 방법은 ‘click’ → ‘order’와 같은 순차적 의존성을 가진 태스크들에 적합하며 다양한 랭킹 단계와 시나리오에 통합될 수 있습니다.

- **Performance Highlights**: Shopee Search에서 진행된 온라인 A/B 테스트 결과에 따르면, ResFlow는 추가적인 시스템 지연 없이 OPU(order-per-user)가 1.29% 증가하는 성과를 거두었습니다. 이 프레임워크는 현재 Shopee Search의 Pre-rank 모듈에 완전히 배포되어 실용적인 가치를 입증하였습니다.



### Entropy and type-token ratio in gigaword corpora (https://arxiv.org/abs/2411.10227)
Comments:
          12 pages, 10 figures, 7 tables

- **What's New**: 이 연구에서는 영어, 스페인어, 터키어의 6가지 대규모 언어 데이터 세트를 분석하여 어휘 다양성(lexical diversity)을 측정하는 두 가지 주요 지표인 entropy와 type-token ratio (TTR) 간의 상관관계를 발견했습니다. 특히, 두 지표 간에 발견된 기능적 관계는 Zipf 법칙과 Heaps 법칙과도 연결되어 있습니다.

- **Technical Details**: 연구에서는 1억 개 이상의 단어를 포함한 6개의 대규모 텍스트 코퍼스를 사용하였으며, 여기에는 책, 뉴스 기사 및 트위터 데이터가 포함됩니다. 각 언어의 형태론적 특성을 고려한 결과, 영어, 스페인어, 터키어 간에 어휘 구조에서 차이를 발견했습니다. 동시에, 대규모 어휘의 경우에 대한 entropy의 분석적 표현을 도출하여 이를 코퍼스 데이터에 맞추어 분석했습니다.

- **Performance Highlights**: 연구 결과, entropy와 TTR 간의 관계는)에 대한 통찰을 제공하며, 이는 자연어 처리(NLP) 및 기계 학습(machine learning)의 성능 최적화와 데이터 압축 효율 향상 등 다양한 응용 분야에서 가치 있는 기초 자료로 활용될 수 있습니다.



### Establishing and Evaluating Trustworthy AI: Overview and Research Challenges (https://arxiv.org/abs/2411.09973)
Comments:
          Accepted in Frontiers in Big Data and AI, Research Topic: Towards Fair AI for Trustworthy Artificial Intelligence

- **What's New**: 이 논문은 AI 시스템의 신뢰성을 확보하기 위해 충족해야 할 여섯 가지 요구 사항을 정의하고, 각각의 정의와 평가 방법, 연구 도전 과제를 논의합니다.

- **Technical Details**: 신뢰할 수 있는 AI의 여섯 가지 요구 사항은 다음과 같습니다: 1) 인간의 주체성과 감독 (human agency and oversight), 2) 공정성과 비차별 (fairness and non-discrimination), 3) 투명성과 설명 가능성 (transparency and explainability), 4) 견고함과 정확성 (robustness and accuracy), 5) 프라이버시와 보안 (privacy and security), 6) 책임성 (accountability). 각 요구 사항은 정의되고 이를 수립하고 평가하기 위한 방법이 제시됩니다.

- **Performance Highlights**: 저자들은 신뢰할 수 있는 AI의 구현 및 평가를 위한 방법론과 연구 도전 과제를 다루며, 지속적으로 변화하는 시스템의 다이나믹스와 실제 환경에서의 연구를 강조합니다.



### Information Need in Metaverse Recordings -- A Field Study (https://arxiv.org/abs/2411.09053)
Comments:
          12 pages, 3 Figures, 8 Tables

- **What's New**: 이 논문은 Metaverse Recordings (MVRs)가 Multimedia Information Retrieval (MMIR) 분야 내에서 신흥하고 잠재적으로 중요한 미디어 유형임을 소개합니다. 연구는 MVR 검색에 대한 사용자 정보 요구와 탐색 행동을 이해하기 위한 현장 연구 결과를 제시합니다.

- **Technical Details**: MVR은 가상 환경에서 생성된 콘텐츠의 캡처, 조직 및 검색과 관련된 여러 가지 도전 과제를 제시합니다. MVR에는 비디오 및 오디오뿐만 아니라 이동 패턴, 시선 추적 정보 및 생체 센서 데이터와 같은 복잡한 데이터 형식이 포함될 수 있습니다. 이를 통합하기 위해 사용자 요구와 데이터 유형을 이해하는 것이 중요합니다.

- **Performance Highlights**: 본 연구는 MVR 검색에 적합한 사례, 사용자 유형 및 특정 요구 사항을 정의하여 MVR 맞춤형 검색 시스템 개발의 기초를 제공합니다. 이를 통해 정보 검색 행동에 대한 이해를 높이고 향후 연구 및 시스템 설계에 기여할 수 있는 가능성을 제공합니다.



New uploads on arXiv(cs.CV)

### LLaVA-o1: Let Vision Language Models Reason Step-by-Step (https://arxiv.org/abs/2411.10440)
Comments:
          11 pages, 5 figures

- **What's New**: LLaVA-o1은 자율적인 다단계 추론을 수행하는 새로운 비전-언어 모델(VLM)로 개발되었습니다. 기존의 VLM보다 구조적이고 체계적인 사고를 통해 복잡한 시각 질문-답변 작업에서 향상된 정밀도를 보여줍니다.

- **Technical Details**: LLaVA-o1은 요약 <SUMMARY>, 캡션 <CAPTION>, 추론 <REASONING>, 결론 <CONCLUSION>의 4단계로 이루어진 비구조적 사고에서 벗어난 구조적 사고를 채택합니다. 이 모델은 LLaVA-o1-100k 데이터셋을 활용하여 훈련되었습니다.

- **Performance Highlights**: LLaVA-o1은 100k 훈련 샘플만으로 다양한 다중 모달 추론 벤치마크에서 8.9% 향상된 성능을 기록하며, Gemini-1.5-pro 및 GPT-4o-mini와 같은 대형 모델보다 성능이 우수합니다.



### M-VAR: Decoupled Scale-wise Autoregressive Modeling for High-Quality Image Generation (https://arxiv.org/abs/2411.10433)
- **What's New**: 이 논문은 이미지 생성을 위한 새로운 자기 회귀 패러다임인 VAR(Vision Autoregressive Model)을 제안하며, 기존의 next-token 예측 방식에서 벗어나 이미지 생성을 조정하는 효율적인 구조를 가지고 있습니다. VAR는 낮은 해상도에서 시작하여 점차 높은 해상도로 진행하는 "coarse-to-fine next-scale prediction" 방식을 바탕으로 합니다.

- **Technical Details**: 제안된 M-VAR 프레임워크는 두 가지 중요한 모델링 구성 요소인 intra-scale modeling과 inter-scale modeling으로 분리됩니다. Intra-scale modeling에서는 각 스케일 내에서의 지역적 의존성을 포착하기 위해 양방향 self-attention을 유지합니다. 반면 inter-scale modeling은 길이가 긴 시퀀스를 처리하고, Mamba(Mamba 메커니즘)를 사용하여 계산 비용을 크게 줄입니다.

- **Performance Highlights**: M-VAR는 이미지 품질과 생성 속도 모두에서 기존 모델을 초월하는 성능을 보입니다. 예를 들어, 1.5B 파라미터를 가진 M-VAR 모델은 FID 점수 1.93을 달성하며, 이는 VAR-d30의 2B 모델과 비교할 때 더 적은 파라미터와 1.2배 빠른 추론 속도를 제공합니다. M-VAR-d32 모델은 ImageNet 256×256에서 FID 점수 1.78을 기록하며, 이전의 최선의 자기 회귀 모델들을 능가했습니다.



### Llama Guard 3 Vision: Safeguarding Human-AI Image Understanding Conversations (https://arxiv.org/abs/2411.10414)
- **What's New**: 본 논문에서는 이미지 이해를 포함한 다중 모달 LLM 기반의 대화 안전 장치인 'Llama Guard 3 Vision'을 소개합니다. 이 모델은 다중 모달 LLM 입력(프롬프트 분류)과 출력(응답 분류)의 안전성을 보장할 수 있도록 설계되었습니다.

- **Technical Details**: Llama Guard 3 Vision은 Llama 3.2-Vision을 기반으로 미세 조정(fine-tuning)되었으며, MLCommons 분류법을 사용하여 13가지 위험 카테고리에서 강력한 성능을 나타냅니다. 특히, 다양한 이미지 및 텍스트 프롬프트에 대한 해로운 내용을 감지하기 위해 최적화되어 있습니다.

- **Performance Highlights**: Llama Guard 3 Vision은 내부 벤치마크 테스트를 통해 기존의 텍스트 전용 모델들과 비교하여 응답 분류 작업에서 더 높은 강인성을 보였습니다. 이 모델은 실제 공격 시나리오에서 프롬프트 기반 공격을 효과적으로 무시하면서 안전성 분류를 수행하는데 주력하고 있습니다.



### Repurposing Stable Diffusion Attention for Training-Free Unsupervised Interactive Segmentation (https://arxiv.org/abs/2411.10411)
- **What's New**: 이 논문에서는 Stable Diffusion(안정적인 확산) 기반의 새로운 비지도 훈련 없는 접근 방식을 제안합니다. 기존의 비지도 방법들이 사용하는 pseudo-labels 대신 self-attention 맵을 활용하여 더 적은 노이즈와 뚜렷한 경계를 지닌 Markov-map을 생성합니다.

- **Technical Details**: 제안된 Markov-map은 self-attention 텐서를 Markov 전이 연산자로 해석하여 반복적으로 Markov 체인을 형성합니다. 각 픽셀은 특정 확률 값에 도달하기 위해 필요한 반복 횟수를 카운트하여 Markov 맵을 생성합니다. 이 방법은 분할 지역 간의 보다 일관된 값을 제공합니다.

- **Performance Highlights**: 모든 실험에서 Number of Clicks (NoC) 기준으로 뛰어난 성과를 기록하였으며, 기존 훈련 기반 비지도 방법들보다 여러 데이터셋에서 우수한 성능을 보였습니다. 이는 훈련이 필요 없는 방식으로도 가능하다는 점에서 큰 진전을 이루었습니다.



### Deep Learning for Micro-Scale Crack Detection on Imbalanced Datasets Using Key Point Localization (https://arxiv.org/abs/2411.10389)
- **What's New**: 이 논문은 구조적 건강 모니터링 영역에서 내부 균열 탐지에 대한 심층 학습(Deep Learning, DL) 방법의 새로운 적용을 탐구합니다. 특히 마이크로 스케일 균열을 식별하기 위해 DL 기반의 키 포인트 탐지 기술을 사용하여, 균열의 경계를 정의하는 네 개의 키 포인트 좌표를 예측함으로써 균열을 국지화하는 방식을 제시합니다.

- **Technical Details**: 이 연구에서는 Inception 모듈을 포함하는 Wide Convolutional Networks을 사용하여 수치 데이터에서 균열을 감지하기 위한 새로운 접근법을 제시합니다. 모델은 다양한 필터 크기로 구성된 여러 컨볼루션 레이어를 사용하여 빠르고 효율적으로 다양한 특징을 추출합니다. 또한, Attention Mechanisms을 도입해 특징 맵 내에서 중요 영역에 집중하여 성능을 향상시킵니다.

- **Performance Highlights**: 모델은 마이크로 스케일 균열 탐지에 적용되었으며, 실제 균열 위치와 예측된 균열 위치 간 평균 Intersection over Union (IoU) 값이 0.511로 쉬운 미세 균열 및 0.631로 큰 균열에 대해 나타났습니다. 이는 균열 탐지에서 이전의 Deep Learning 모델보다 성능 향상을 보여줍니다.



### Generation of synthetic gait data: application to multiple sclerosis patients' gait patterns (https://arxiv.org/abs/2411.10377)
- **What's New**: 본 논문에서는 다중 경화증(Multiple Sclerosis, MS)의 비침습적이고 정량적인 보행 평가 도구로서 eGait 이동 센서를 개발하고, 단일 쿼터니언 시계열(unit quaternion time series, QTS)를 이용하여 사람의 보행을 특성화하는 새로운 방법을 제안합니다.

- **Technical Details**: QTS 데이터를 보존하는 포괄적인 프레임워크를 통해 데이터의 기하학적 특성을 유지하면서 모든 표 형식 합성 데이터 생성 방법을 사용할 수 있도록 변환합니다. 또한, 최근접 이웃 가중치(nearest neighbors weighting)를 기반으로 한 합성 데이터 생성 방법을 도입하여 작은 데이터 세트에 적합한 고충실도 합성 QTS 데이터를 성공적으로 생성합니다.

- **Performance Highlights**: 제안된 방법은 MS 보행 데이터를 사용하여 적용되었으며, 데이터의 초기 기하학을 잘 유지하면서 매우 좋은 충실도를 보여주었습니다. 이 연구를 통해 합성 데이터 세트를 생산하고 클러스터링 방법의 안정성 향상에 기여할 수 있게 되었습니다.



### Towards High-Fidelity 3D Portrait Generation with Rich Details by Cross-View Prior-Aware Diffusion (https://arxiv.org/abs/2411.10369)
- **What's New**: 최근 단일 이미지 기반 3D 초상화 생성 기술에서, 새로운 Hybrid Priors Diffusion 모델을 통해 다중 뷰 사전 정보(multi-view priors)를 활용하여 세밀하고 일관된 3D 초상화를 생성할 수 있는 방법을 제안합니다.

- **Technical Details**: 1. Hybrid Priors Diffusion 모델(HPDM)은 명시적 및 암시적으로 다중 뷰 사전을 고려하여 조건을 집어넣어 초상화의 일관성을 향상시킵니다. 2. Multi-View Noise Resampling Strategy(MV-NRS)를 도입하여 다양한 뷰의 노이즈 분포를 관리하고, SDS 손실을 통해 일관된 표현을 추구합니다. 3. GAN-prior Initialization과 Portrait Geometry Restoration, Multi-view Diffusion Refinement 모듈로 구성된 포트레이트 확산 파이프라인을 개발했습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 높은 정밀도의 3D 초상화를 생성하는 데 성공하였으며, 풍부한 세부 묘사를 확보했습니다.



### Mechanisms of Generative Image-to-Image Translation Networks (https://arxiv.org/abs/2411.10368)
- **What's New**: 이번 논문에서는 기존의 복잡한 구조를 간소화한 이미지 간 변환을 위한 새로운 네트워크 아키텍처를 제안합니다. GAN (Generative Adversarial Network)과 autoencoder의 관계를 조사하여, 이미지 변환 작업에서 GAN 구성 요소만을 사용하는 것이 효과적이라는 설명을 제공합니다.

- **Technical Details**: 저자는 충분한 용량의 판별자 (discriminator)가 있을 경우, GAN 기반의 훈련이 전통적인 autoencoder 모델과 유사한 결과를 낸다는 점을 실험적으로 증명합니다. 또, 이미지 간 변환 문제에 대해 간단한 GAN 모델이 공통적인 특징을 유지하면서 새로운 특징을 생성할 수 있다는 것을 보여줍니다.

- **Performance Highlights**: 실험을 통해 GAN 및 autoencoder의 성능을 비교하고, 이미지 간 변환의 모델 능력과 특정 제약 조건을 조사하여 기존 방법과 차별화된 성능을 확인하였습니다.



### Interactive Image-Based Aphid Counting in Yellow Water Traps under Stirring Actions (https://arxiv.org/abs/2411.10357)
- **What's New**: 현재의 비전 기반 진딧물 수 세기 방법은 occlusions(가림 현상) 및 낮은 가시성으로 인해 정확도가 떨어지는 문제가 있습니다. 이에 대한 해결책으로, 인터랙티브 스터링(interactive stirring) 액션을 통한 새로운 진딧물 수 세기 방법을 제안합니다.

- **Technical Details**: 이 방법에서는 yellow water trap(노란색 물 트랩)에 있는 진딧물의 분포를 변경하기 위해 인터랙티브 스터링을 사용하고, 이후에 최적화된 작은 객체 탐지 네트워크인 Yolov5를 기반으로 진딧물 감지 및 수 세기를 수행하기 위해 이미지 시퀀스를 캡처합니다. 또한, 수 세기 결과의 신뢰성을 평가하는 counting confidence evaluation system(수 세기 신뢰도 평가 시스템)을 제안합니다.

- **Performance Highlights**: 우리의 제안한 진딧물 탐지 네트워크는 원래의 Yolov5보다 AP@0.5에서 33.9%, AP@[0.5:0.95]에서 26.9% 향상된 성능을 보여주었으며, 제안된 수 세기 신뢰도 평가 시스템을 사용한 진딧물 수 세기 테스트 결과는 정적인 수 세기 방법에 비해 상당한 개선을 보여 수동 수 세기 결과와 밀접하게 일치합니다.



### BiDense: Binarization for Dense Prediction (https://arxiv.org/abs/2411.10346)
- **What's New**: BiDense는 효율적이고 정확한 dense prediction을 위한 새로운 일반화된 binary neural network (BNN) 아키텍처를 제안합니다.

- **Technical Details**: BiDense는 두 가지 주요 기술, 즉 Distribution-adaptive Binarizer (DAB)와 Channel-adaptive Full-precision Bypass (CFB)를 포함합니다. DAB는 binarization을 위한 임계값(threshold)과 스케일링 팩터(scaling factors)를 적응적으로 계산하여 BNN 내 정보를 더 많이 보존합니다. CFB는 다양한 채널 크기 변환을 거치는 binary convolutional layers에서 full-precision bypassing을 가능하게 해, 실수(real-valued) 신호의 전파를 향상시키고 정보 손실을 최소화합니다.

- **Performance Highlights**: BiDense는 실험을 통해 full-precision 모델과 유사한 성능을 발휘하면서도 메모리 사용량과 컴퓨팅 비용을 significantly 감소시킵니다.



### Comparative Analysis of Machine Learning Approaches for Bone Age Assessment: A Comprehensive Study on Three Distinct Models (https://arxiv.org/abs/2411.10345)
- **What's New**: 이 연구에서는 아동과 유아의 X-ray 영상을 이용하여 뼈 나이를 예측하는 자동화 과정의 정확성을 높이기 위해 세 가지 주요 기계 학습 모델인 Xception, VGG, CNN 모델을 비교 분석하였습니다.

- **Technical Details**: 모델들은 사전 처리된 데이터 세트를 기반으로 훈련되었으며, 각 모델의 정확도는 월 단위의 평균 절대 오차(MAE)를 사용하여 측정되었습니다. 세 가지 모델의 성능을 비교함으로써 적합한 모델의 선택을 위한 기초 자료를 제공하고자 하였습니다.

- **Performance Highlights**: Xception, VGG, CNN 모델은 모두 정확성과 관련 요소에 대해 테스트되었으며, 각 모델 간의 성능 차이가 확인되었습니다. 각 모델의 특징에 따라 선택할 수 있는 다양한 옵션이 제공됩니다.



### Y-MAP-Net: Real-time depth, normals, segmentation, multi-label captioning and 2D human pose in RGB images (https://arxiv.org/abs/2411.10334)
Comments:
          8 page paper, 6 Figures, 3 Tables

- **What's New**: Y-MAP-Net는 RGB 이미지의 실시간 다중 작업 학습을 위해 설계된 Y 모양의 신경망 아키텍처입니다. 이 모델은 단일 네트워크 평가로 깊이, 표면 법선, 인간 자세, 의미 영역 분할을 동시에 예측하고, 다중 레이블 캡션을 생성할 수 있습니다.

- **Technical Details**: Y-MAP-Net는 다중 교사, 단일 학생 훈련 패러다임을 채택합니다. 이를 통해 작업별 기초 모델이 네트워크의 학습을 감독하여, 실시간 애플리케이션에 적합한 경량 아키텍처로 자원 소모를 최소화합니다. 이 모델은 단안 RGB 입력을 받아 44개의 히트맵 / 이미지 출력과 8개의 캡션 토큰을 제공합니다.

- **Performance Highlights**: Y-MAP-Net는 깊이, 법선 및 인간 자세 추정, 장면 분할 및 캡션 생성을 단일 네트워크에서 동시에 수행할 수 있는 최초의 방법입니다. 효율성 덕분에 실시간 성능을 발휘하며, 로봇 공학 등의 다양한 실제 응용 프로그램에 많은 가능성을 열어줍니다.



### Number it: Temporal Grounding Videos like Flipping Manga (https://arxiv.org/abs/2411.10332)
Comments:
          11 pages, 7 figures

- **What's New**: 이번 연구에서는 Video Temporal Grounding (VTG)에서 Video Large Language Models (Vid-LLMs)의 성능을 향상시키기 위한 새로운 방법, Number-Prompt (NumPro)를 소개합니다. NumPro는 각각의 비디오 프레임에 고유한 숫자 식별자를 추가하여 비주얼 이해와 시간적으로 지시하는 것을 연결합니다.

- **Technical Details**: NumPro는 비디오를 프레임 이미지의 순서로 처리하고, 각 프레임에 고유 숫자를 표기하여 VTG 과정을 직관적으로 만듭니다. 이 방법을 통해 Vid-LLMs는 사건의 시간 정보를 명확하게 링크할 수 있습니다. 실험 결과, NumPro는 기존의 최첨단 Vid-LLM에서 VTG 성능을 대폭 향상시키고, 추가적인 계산 비용 없이도 이전 방법보다 mIoU에서 최대 6.9% 및 mAP에서 8.5%의 향상을 기록했습니다.

- **Performance Highlights**: NumPro를 통해 경량화된 VTG 성능 평가에서 주목할 만한 성과를 보여주었고, 특히 Moment Retrieval과 Highlight Detection에서 각각 6.9% 및 8.5%의 향상을 보였습니다. NumPro를 보강하는 데이터셋에서 세밀하게 조정하여 새로운 최첨단 성능을 달성했습니다.



### CNN-Based Classification of Persian Miniature Paintings from Five Renowned Schools (https://arxiv.org/abs/2411.10330)
Comments:
          20 pages, submitted to journal

- **What's New**: 이 논문은 페르시아 미니어처 작품의 분석을 위한 새로운 접근 방식을 제시합니다. 컨볼루션 신경망(CNN)을 활용하여 다섯 가지 미술 학교(헤라트, 타브리즈-아발, 시라즈-아발, 타브리즈-도브봄, 그리고 카자르)의 페르시아 미니어처를 분류하는 방법을 개발하였으며, 평균 정확도 91% 이상을 달성했습니다.

- **Technical Details**: 이 연구에서는 각 미술 학교의 고유한 특징을 포착하기 위해 신중하게 큐레이션된 데이터셋을 사용했습니다. 데이터셋의 이미지를 패치 기반 접근 방식으로 분류하며, CNN을 사용하여 독립적으로 이미지 세그먼트를 분류한 후 결과를 결합하여 정확도를 높이는 방식을 채택했습니다. Pre-trained CNN을 통한 특징 추출 방법이 논의되고 있습니다.

- **Performance Highlights**: 본 연구의 결과는 디지털 미술 분석 분야에 중요한 기여를 하며, 각 미술 학교의 특징을 잘 나타내는 데이터셋을 제공함으로써 예술 역사와 디지털 인문학의 추가 연구에 유용한 자원이 될 것입니다.



### Melanoma Detection with Uncertainty Quantification (https://arxiv.org/abs/2411.10322)
Comments:
          5 pages, 5 figures, 3 tables, submitted to ISBI2025

- **What's New**: 이 논문은 멜라노마의 조기 발견을 개선하기 위한 여러 공개 데이터셋을 통합하여 데이터 다양성을 높이는 새로운 접근 방식을 소개합니다. 기존 머신러닝 기반 방법들이 데이터셋의 통합을 완전히 반영하지 않는 점을 보완하며, 불확실성 정량화를 통해 오진을 최소화하는 방법을 개발했습니다.

- **Technical Details**: 논문에서는 Input, Melanoma Recognition, Uncertainty Analysis, Integration의 네 가지 모듈로 구성된 일관된 실험 프레임워크를 제시합니다. 이러한 접근은 여러 공개 데이터셋을 통합하고 DNN(Deep Neural Network)을 사용하는 분류기 성능 평가를 통해 표준화된 성능 비교를 가능하게 합니다. 특정 알고리즘으로는 Softmax classifier가 사용되며, 신뢰도를 기반으로 한 예측 필터링을 통해 오진율을 감소시킵니다.

- **Performance Highlights**: 우리의 실험은 대조군 데이터셋에서 불확실성 기반 거부 적용 이전에 최대 93.2%, 이후에는 97.8%의 정확도를 나타내며, 오진을 40.5% 이상 감소시켰습니다. 또한, 사용자가 제공한 이미지를 신속하게 처리할 수 있는 웹 기반 인터페이스도 제공하여 실제 적용 가능성을 높였습니다.



### Probabilistic Prior Driven Attention Mechanism Based on Diffusion Model for Imaging Through Atmospheric Turbulenc (https://arxiv.org/abs/2411.10321)
- **What's New**: 이 논문에서는 대기 난류로 인한 이미지 왜곡을 효과적으로 복원하기 위한 새로운 모델인 Probabilistic Prior Turbulence Removal Network (PPTRN)을 제안합니다. PPTRN은 확률적 확산 모델(Denoising Diffusion Probabilistic Model, DDPM)과 Transformer 기반의 특징 추출을 결합하여 이미지 복원에서의 새로운 접근 방식을 제공합니다.

- **Technical Details**: PPTRN은 두 단계의 학습 방식을 채택합니다. 첫 번째 단계에서는 선명한 이미지를 기반으로 Latent Encoder와 Transformer가 함께 훈련되어 강력한 특징 표현을 구축합니다. 두 번째 단계에서는 DDPM이 라티언트 벡터에 대한 사전 분포를 모델링하여 Transformer가 효과적으로 다양한 특징 변화를 캡처할 수 있도록 안내합니다. 특히, PPTRN의 Probabilistic Prior Driven Cross Attention 메커니즘은 DDPM에서 생성된 사전 정보를 특징 임베딩과 통합하여 아티팩트를 줄이고 공간적 일관성을 향상시킵니다.

- **Performance Highlights**: 광범위한 실험 결과, PPTRN은 난류로 인해 열화된 이미지의 복원 품질을 현저히 향상시키며, 선명도와 구조적 충실성에서 새로운 기준을 설정합니다.



### M3TR: Generalist HD Map Construction with Variable Map Priors (https://arxiv.org/abs/2411.10316)
- **What's New**: M3TR(다중 마스킹 맵 변환기)는 부분적 또는 오래된 맵 정보를 활용하여 HD 맵을 구축하는데 필요한 혁신적인 방법입니다.

- **Technical Details**: M3TR은 고급 쿼리 디자인을 통해 HD 맵 구성 모델에 맵 요소를 통합하는 개선된 방법을 적용하여 성능을 +4.3 mAP 향상시켰습니다. 또한, 다양한 맵 우선 시나리오에서 훈련하여 단일 일반 모델을 생성했습니다.

- **Performance Highlights**: M3TR은 변형된 맵 우선 정보를 활용하여 이전의 전문가 모델과 동등한 성능을 보여줍니다. 이 모델은 가변 맵 우선을 활용할 수 있는 최초의 모델로, 실제 세계 배치에 적합합니다.



### Modification Takes Courage: Seamless Image Stitching via Reference-Driven Inpainting (https://arxiv.org/abs/2411.10309)
Comments:
          17 pages, 10 figures

- **What's New**: 본 논문에서는 기존 이미지 스티칭(사진 합치기) 방법들이 겪고 있는 문제점들을 해결하기 위해 Reference-Driven Inpainting Stitcher (RDIStitcher)를 제안합니다. 이 방법은 이미지 융합(Image Fusion)과 직사각형 변형(Rectangling)을 참고 기반의 인페인팅(Inpainting) 모델로 재구성하여, 더 넓은 수정 융합 영역과 강력한 수정 강도를 도입합니다.

- **Technical Details**: RDIStitcher는 레이블 데이터 없이도 학습할 수 있는 자기지도(self-supervised) 모델 학습 방법을 도입합니다. 이 모델은 Text-to-Image (T2I) 확산(diffusion) 모델을 미세 조정하여 구현되며, 합성 이미지의 품질을 평가하기 위해 Multimodal Large Language Models (MLLMs)를 활용한 지표를 제공합니다.

- **Performance Highlights**: 세 가지 기존 방법보다 RDIStitcher가 콘텐츠 일관성과 매끄러운 전환을 크게 향상시켰습니다. 특히 제로샷(zero-shot) 실험에서 뛰어난 일반화 능력을 보여주었습니다.



### A Realistic Collimated X-Ray Image Simulation Pipelin (https://arxiv.org/abs/2411.10308)
- **What's New**: 이 논문은 X-ray 시스템에서 비정확한 검출기의 위치 정보로 인해 발생하는 collimator 검출의 어려움을 해결하기 위해 물리적으로 동기화된 이미지 처리 파이프라인을 제안합니다. 이 파이프라인은 collimator 그림자의 특성을 시뮬레이션하여 제한된 데이터 세트를 확장할 수 있도록 설계되었습니다.

- **Technical Details**: 시뮬레이션 과정은 크게 세 가지 단계로 나뉘어집니다. 첫 번째 단계에서 collimator 영역의 형태와 위치를 정의하기 위한 무작위 레이블이 생성됩니다. 두 번째 단계에서는 산란 방사선이 도입되고, 마지막 단계에서는 노이즈를 추가합니다. 프레임워크는 콜리메이터의 핵심 요인을 모델링하여 산란된 방사선의 분포를 명시합니다.

- **Performance Highlights**: 이러한 통합된 접근법의 유효성은 실제 collimator 그림자와의 정성적 및 정량적 비교를 통해 검증되었으며, 시뮬레이션 데이터를 깊이있는 학습 프레임워크 내에서 활용함으로써 실제 collimator에 대한 적절한 대체 역할을 할 뿐만 아니라 실제 데이터에 적용했을 때 일반화 성능이 향상됨을 보여주었습니다.



### RETR: Multi-View Radar Detection Transformer for Indoor Perception (https://arxiv.org/abs/2411.10293)
Comments:
          24 pages, Accepted to NeurIPS 2024

- **What's New**: 이 논문에서는 다중 뷰 레이더 인식을 위해 개발된 Radar dEtection TRansformer(RETR)를 제안합니다. 이는 기존의 DETR 아키텍처를 기반으로 하며, 레이더 신호의 고유한 특성을 고려한 설계를 포함하고 있습니다.

- **Technical Details**: RETR는 다음과 같은 주요 구성 요소를 갖추고 있습니다: 1) 조정 가능한 위치 인코딩(Tunable Positional Encoding, TPE)을 통해 깊이 우선 피쳐 유사도를 강화; 2) 레이더와 카메라 좌표계에서의 삼면 손실(tri-plane loss)을 도입; 3) 재매개변수를 통한 학습 가능한 레이더-카메라 변환을 실현하여 다중 뷰 레이더 상황에 맞는 특성을 반영합니다.

- **Performance Highlights**: RETR는 HIBER 및 MMVR 데이터셋에 대해 평가되었으며, 객체 탐지에서 기존 최첨단 방법보다 15.38 이상의 AP(평균 정밀도)를 향상시키고, 인스턴스 세분화에서 11.77 이상의 IoU(교차 영역 비율)를 달성했습니다.



### Multidimensional Byte Pair Encoding: Shortened Sequences for Improved Visual Data Generation (https://arxiv.org/abs/2411.10281)
- **What's New**: 본 논문에서는 비주얼 데이터(visual data)의 토큰화(tokenization) 방법을 개선하기 위한 새로운 접근 방식을 제안합니다. 특히, 1D 토큰 압축에서 Byte Pair Encoding (BPE) 개념을 다차원(multidimensional)으로 확장하여, 더 짧고 정보가 고르게 분포된 시퀀스를 생성합니다.

- **Technical Details**: Multidimensional Byte Pair Encoding (MDBPE)이라는 새로운 압축 방법을 통해 1D를 넘어서 다양한 차원에서 이미지 데이터를 처리할 수 있습니다. 이 방법은 토큰 쌍의 출현 빈도를 세고, 가장 빈번한 토큰 쌍을 새 토큰으로 대체하는 과정으로 이루어집니다.

- **Performance Highlights**: 실험 결과, MDBPE를 통해 생성된 압축된 시퀀스는 트랜스포머(transformers)의 학습 및 추론 성능을 향상시켰으며, 이미지넷(ImageNet)과 같은 대규모 데이터셋에도 적용 가능함을 보였습니다. 더불어, 논문에 제시된 코드는 Jupyter Notebooks 형식으로 제공되며, C++ 구현체도 마련되어 있습니다.



### 4DPV: 4D Pet from Videos by Coarse-to-Fine Non-Rigid Radiance Fields (https://arxiv.org/abs/2411.10275)
Comments:
          17th Asian Conference on Computer Vision (ACCV 2024)

- **What's New**: 본 논문에서는RGB 시퀀스를 통해 카메라 포즈(camera pose)와 4D 재구성(4D reconstruction)을 동시 복구할 수 있는 coarse-to-fine neural deformation model을 제안합니다. 이 접근 방식은 사전 구축된 3D 템플릿(3D template)이나 3D 학습 데이터 없이, 자가 감독(self-supervised) 방식으로 문제를 해결합니다.

- **Technical Details**: 모델은 canonical (표준) 및 image-variant (이미지 변형) 공간을 활용하여, coarse 및 fine 구성 요소를 모두 고려합니다. 또한, 스페이셜-템퍼럴(spatio-temporal) 일관성을 갖는 neural local quadratic model을 도입하여 복잡한 세부 사항을 인코딩하고, 이를 canonical embeddings와 결합하여 시퀀스 간의 대응관계를 설정합니다.

- **Performance Highlights**: 복잡하고 실제의 변형이 있는 도전적인 시나리오에서 방법을 철저히 검증하며 정량적 및 정성적 평가, ablation study, 경쟁 방법과의 비교를 제공합니다. 이 프로젝트는 해당 URL에서 확인할 수 있습니다.



### Fill in the blanks: Rethinking Interpretability in vision (https://arxiv.org/abs/2411.10273)
- **What's New**: 이 논문은 기존 XAI(Explainable AI) 기술의 한계를 극복하고 새로운 시각 모델 해석 방안을 제시합니다. 특히, 모델이 학습한 입력 구조를 탐색하여 '마스크 처리된 이미지'를 어떻게 채우는지를 질문합니다.

- **Technical Details**: 본 연구에서는 이미지 분류를 위한 새로운 mask-filling 접근 방식을 도입하며, 추가 학습이 필요 없는 방법으로 다양한 이미지 데이터셋에서 실험을 진행합니다. 모델 별로 제시되는 설명 가능성을 높이는 데 중점을 두고, 이전의 local 및 global 해석 방식의 문제를 해결하고자 합니다.

- **Performance Highlights**: 실험 결과, 표준 이미지 데이터셋에서 일관된 패턴을 보여주었으며, mask 파라미터 변경에 따른 시각적 결과의 효과를 시연합니다. 논문에서 제안된 방법은 현대 머신러닝 플랫폼에서 모델 독립적인 설명 가능성 도구로 통합될 수 있습니다.



### Partial Scene Text Retrieva (https://arxiv.org/abs/2411.10261)
Comments:
          Accepted on TPAMI

- **What's New**: 본 논문에서는 부분(scene) 텍스트 검색을 위한 새로운 접근 방식을 제시합니다. 특히, 텍스트 라인 인스턴스와 이들의 부분 패치를 동시에 검색할 수 있는 네트워크를 제안합니다.

- **Technical Details**: 이 방법은 쿼리 텍스트와 장면 텍스트 인스턴스를 공유 기능 공간에 임베딩하고 그들의 교차 모드 유사성을 측정하는 방식으로 작동합니다. 부분 패치를 처리하기 위해, Multiple Instance Learning (MIL) 접근 방식을 채택하여 추가 주석 없이 쿼리 텍스트와의 유사성을 학습합니다.

- **Performance Highlights**: Ranking MIL (RankMIL) 접근 방식을 통해 노이즈 샘플을 적절히 필터링하고, Dynamic Partial Match Algorithm (DPMA)를 통해 탐색 효율성을 크게 향상시킵니다. 이 결과 PPR(Partial Patches Retrieval) 성능이 크게 개선되었습니다.



### The Unreasonable Effectiveness of Guidance for Diffusion Models (https://arxiv.org/abs/2411.10257)
Comments:
          Preprint. 19 pages, 14 figures in total, including references and appendix

- **What's New**: 이 논문에서는 새로운 에러 교정 기법인 슬라이딩 윈도우 가이던스(SWG)를 소개합니다. SWG는 주 모델의 수용 영역(receptive field)을 제어하여 고유한 방식으로 모델을 유도하며, 인간의 선호도와 더 잘 일치하면서도 훈련, 아키텍처 수정 또는 클래스 조건화가 필요 없습니다.

- **Technical Details**: SWG는 장기 공간 의존성(long-range spatial dependencies)을 강화하여 시각 품질을 개선하는 새로운 가이던스 방법입니다. 전통적인 클래스 없는 가이던스(Classifier-Free Guidance, CFG)와 달리 SWG는 추가 훈련이나 아키텍처적 변경 없이 어떤 확산 모델(Diffusion Model, DM)에도 적용할 수 있습니다. 이 방법은 특히 약한 모델 가이던스(Weak Model Guidance, WMG) 방식과 대조되며, 더 강력한 가중치 정규화(weight regularization)를 수행하여 생성된 샘플의 품질을 향상시킵니다.

- **Performance Highlights**: SWG는 최신 가이던스 기법과 비교해 경쟁력 있는 생성 성능을 발휘하며, 인간의 선호도와의 일치도가 더 높습니다. 이 방법은 어떤 DM에서도 사용 가능하고, 시각적으로 높은 품질의 이미지를 생성하는 데 효과적입니다.



### Visual-Linguistic Agent: Towards Collaborative Contextual Object Reasoning (https://arxiv.org/abs/2411.10252)
- **What's New**: Multimodal Large Language Models (MLLMs)와 전통적인 객체 감지 모델의 협업을 통해 객체 지방화의 정확도를 향상시키는 Visual-Linguistic Agent(VLA) 프레임워크를 도입했습니다.

- **Technical Details**: VLA는 MLLM을 Linguistic Agent로 사용하여 Vision Agent와 협력하여 객체 검출 및 분류를 수행합니다. MLLM은 공간적 및 맥락적 관계를 고려하여 탐지를 평가하고 수정하며, Vision Agent는 이를 보완하여 정확도를 높입니다.

- **Performance Highlights**: COCO 데이터셋에서의 광범위한 평가를 통해 VLA는 최신 객체 감지 모델에 비해 AP50:95에서 최대 3%의 향상을 보였으며, 정확하고 맥락적으로 일관된 객체 감지를 설정하는 잠재력을 보여주었습니다.



### Morpho-Aware Global Attention for Image Matting (https://arxiv.org/abs/2411.10251)
- **What's New**: 이번 논문에서는 이미지 매팅에서 미세한 구조들을 효과적으로 보존하기 위한 새로운 Morpho-Aware Global Attention (MAGA) 메커니즘을 제안합니다. 기존의 Vision Transformers (ViTs)와 Convolutional Neural Networks (CNNs)의 한계를 극복하고, 지역적 형태 정보를 통해 전역 맥락 내에서 미세한 구조를 강조할 수 있는 새로운 접근법을 제시합니다.

- **Technical Details**: MAGA는 Tetris와 유사한 convolutional 패턴을 사용하여 미세한 구조의 지역적 형태를 정렬합니다. 이 방식을 통해 지역적인 형태 정보를 쿼리 임베딩(query embeddings)으로 추출하고, 이를 글로벌 키 임베딩(global key embeddings)과 연결하여 지역 세부 정보를 강조합니다. 최종적으로 값 임베딩(value embeddings)으로 프로젝트 하여 강조된 형태 세부사항을 통합합니다.

- **Performance Highlights**: MAGA 기반의 ViT는 Composition-1k 및 Distinctions-646 데이터셋에서 기존 최첨단 방법보다 평균 4.3%의 SAD(Sum of Absolute Differences) 및 39.5%의 MSE(Mean Squared Error) 성능 향상을 보여줍니다.



### ScribbleVS: Scribble-Supervised Medical Image Segmentation via Dynamic Competitive Pseudo Label Selection (https://arxiv.org/abs/2411.10237)
- **What's New**: 이번 논문에서는 ScribbleVS라는 새로운 프레임워크를 제안하여 의학 이미지 분할에서 스크리블 주석을 효과적으로 활용합니다. 이 프레임워크는 이미지에서 스크리블 주석만을 사용하여 모델 학습을 수행하여 효율성을 높이고, 수작업 주석의 필요성을 감소시킵니다.

- **Technical Details**: ScribbleVS는 Mean Teacher Network를 기반으로 하며, Regional Pseudo Labels Diffusion Module(RPD)와 Dynamic Competition Selection Module(DCS)을 포함하여 푸와 최고 성능을 달성합니다. 이 프레임워크는 스크리블 주석을 활용하여 지도학습을 확장하고, 수 pseudo labels의 노이즈 영향을 최소화합니다.

- **Performance Highlights**: ACDC 및 MSCMRseg 데이터셋에서의 실험 결과, ScribbleVS는 완전 감독 방법의 성능을 초과하는 높은 정밀도로 성능 개선을 보여주었습니다. 이는 의료 이미지 분할의 정확도를 크게 향상시킵니다.



### ColorEdit: Training-free Image-Guided Color editing with diffusion mod (https://arxiv.org/abs/2411.10232)
- **What's New**: 본 논문은 기존 텍스트-유도 이미지 편집 기술의 색상 변경 실패 원인을 분석하고 이를 해결하기 위한 새로운 방법을 제안합니다. 주요 발견으로 색상 속성 조정의 안정성을 위해 이미지-유도 방법을 활용하며, COLORBENCH라는 평가 벤치마크 데이터셋도 소개합니다.

- **Technical Details**: 논문에서는 텍스트 유도 이미지 합성 과정에서의 다양한 cross-attention 블록이 학습한 의미 정보를 분석합니다. 특히, denoising 과정 초기 단계에서 U-Net 디코더를 통해 객체의 형태와 질감을 결정하며, cross-attention 레이어의 Value 행렬 정렬을 통해 색상 조정을 가능하게 합니다.

- **Performance Highlights**: 제안된 이미지-유도 색상 편집 방법은 기존의 텍스트 유도 이미지 편집 방법들보다 우수한 성능을 보이며, 실험을 통해 생성된 이미지와 실제 이미지 모두에서 효과성을 입증했습니다.



### A Low-Resolution Image is Worth 1x1 Words: Enabling Fine Image Super-Resolution with Transformers and TaylorShif (https://arxiv.org/abs/2411.10231)
- **What's New**: 이 연구에서는 1x1 패치 크기를 활용하여 pixel-level 처리(pixel-level processing)를 가능하게 하는 TaylorIR이라는 변환기 기반의 Super-Resolution(SR) 모델을 제안합니다. 이는 기존 SR 모델의 단점을 극복하고자 합니다.

- **Technical Details**: TaylorIR은 기존의 self-attention 메커니즘 대신 TaylorShift attention 메커니즘을 도입합니다. TaylorShift는 Taylor 급수 전개(Taylor series expansion)를 기반으로 하여 메모리 효율적인 방식으로 구현됩니다. 이를 통해 전체 token-to-token 상호작용을 선형 복잡도로 달성할 수 있습니다.

- **Performance Highlights**: TaylorIR을 사용한 SwinIR 모델인 TaylorSwinIR은 PSNR과 SSIM 메트릭에서 기존의 SR 모델들보다 뛰어난 성능을 발휘하며, 메모리 소비는 최대 60%까지 줄일 수 있음을 실험을 통해 입증하였습니다.



### MCL: Multi-view Enhanced Contrastive Learning for Chest X-ray Report Generation (https://arxiv.org/abs/2411.10224)
Comments:
this https URL

- **What's New**: 이번 연구에서는 다중 뷰 영상(x-ray)에서 종합적인 정보(정보)를 활용해 방사선 보고서를 생성하는 Multi-view enhanced Contrastive Learning (MCL) 방법을 제안합니다. 이는 기존의 단일 뷰를 기반으로 한 자동 보고서 생성 방법의 한계를 극복하는 데 초점을 맞추고 있습니다.

- **Technical Details**: MCL 방법은 두 단계로 구성됩니다. 첫 번째 단계에서는 다중 뷰 향상 대조 학습(multi-view enhanced contrastive learning)을 통해 동일 연구 내의 여러 뷰(x-ray)와 이들에 해당하는 보고서 간의 일치를 최대화합니다. 두 번째 단계에서는 교차 주의(cross-attention) 메커니즘을 활용해 환자별 정보(예: 증상)를 보고서 생성에 통합합니다. 또한, 누락된 정보를 보완하기 위한 '전이 브릿지(transitional bridge)'를 도입하여 이로 인한 임베딩 공간 차이를 줄입니다.

- **Performance Highlights**: MCL은 여러 데이터셋에서 최근의 최첨단 방법들을 초월하는 성능을 보였습니다. MIMIC-CXR에서 5.0%의 F1 RadGraph 향상, MIMIC-ABN에서 7.3%의 BLEU-1 향상, Multi-view CXR에서 3.1%의 BLEU-4 향상, Two-view CXR에서 8.2%의 F1 CheXbert 향상을 달성했습니다.



### Learning Generalizable 3D Manipulation With 10 Demonstrations (https://arxiv.org/abs/2411.10203)
- **What's New**: 이 연구는 단 10회의 시연으로 조작 기술을 학습하는 혁신적인 프레임워크를 제안합니다. 이 프레임워크는 공간적 변형에 일반화된 조작 능력을 유지합니다.

- **Technical Details**: 제안하는 프레임워크는 두 가지 핵심 모듈로 구성됩니다: Semantic Guided Perception (SGP)과 Spatial Generalized Decision (SGD). SGP는 RGB-D 입력을 기반으로 한 3D 포인트 클라우드 표현을 구성하며, SGD는 효율적인 의사결정 모듈로서 노이즈 감소를 통해 행동을 생성합니다. 공간적으로 변환될 수 있는 훈련 전략을 도입하여 전문가 시연에서 내재된 공간적 지식을 캡처합니다.

- **Performance Highlights**: 이 방법은 시뮬레이션 벤치마크 및 실제 로봇 시스템에서 광범위한 실험을 통해 검증되었습니다. 제안된 방법은 많은 변형이 있는 작업를 수행하면서도 기존의 최신 방법보다 60% 이상의 성공률 향상을 보여주었습니다.



### Block based Adaptive Compressive Sensing with Sampling Rate Contro (https://arxiv.org/abs/2411.10200)
Comments:
          Accepted to MMAsia2024

- **What's New**: 이 논문은 비디오 압축 센싱(Compressive Sensing, CS)에서의 시간적 중복성을 활용하여 샘플링 비율(Sampling Rate, SR)을 조절하는 블록 기반 적응형 압축 센싱 프레임워크를 제안합니다. 특히 움직이는 블록 감지(moving block detection)를 통해 비움직임 영역의 중복 압축을 피하고, 비디오 품질을 유지하면서 샘플링된 데이터의 양을 줄이는 방법을 탐구합니다.

- **Technical Details**: 제안된 방식은 두 개의 주요 구성 요소로 이루어져 있습니다: 첫째, 움직이는 블록 감지를 통해 비움직임 블록의 측정을 제거하고, 둘째, 비디오의 각 프레임에 대한 SR 할당을 동적으로 조정하는 동적 임계값(dynamic threshold)을 도입하여 평균 SR을 제어할 수 있습니다. 이 방법은 VIRAT 데이터셋을 사용한 실험을 통해 기존 방법들과 비교하여 향상된 성능을 보여주었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 평균 SR 제어와 재구성 품질 측면에서 최근의 최첨단 방법들보다 우수한 성능을 발휘하였습니다. 협동 재구성(cooperative reconstruction) 방식을 도입하여 비움직임 블록의 측정을 참조하여 재구성을 수행함으로써 블록 아티팩트를 줄이고 품질을 개선하였습니다.



### STLight: a Fully Convolutional Approach for Efficient Predictive Learning by Spatio-Temporal joint Processing (https://arxiv.org/abs/2411.10198)
Comments:
          Accepted at WACV 2025 conference

- **What's New**: 이 논문은 STLight라는 새로운 spatio-temporal predictive Learning 방법을 제안합니다. 이 방법은 기존의 복잡한 시퀀스 모델 대신, 채널 및 깊이 방향의 컨볼루션(convolutions)만을 사용하여 학습합니다.

- **Technical Details**: STLight는 공간(spatial) 및 시간적(temporal) 차원을 함께 재배열하는 방법을 사용하여, 단일 컨볼루션을 통해 두 가지 종류의 특성을 믹스하여 포괄적인 spatio-temporal 패치(patch) 표현을 만듭니다. 이 표현은 순수한 컨볼루션 모델에서 처리되어, 근접(patch)과 먼(patch) 간의 상호작용에 동시에 집중할 수 있습니다.

- **Performance Highlights**: STLight는 다양한 데이터셋 및 설정에서 STL 벤치마크에서 최신 기술을 능가하는 성능을 달성했으며, 파라미터(param)와 계산 FLOPs 측면에서 높은 효율성을 보여줍니다.



### DiMoDif: Discourse Modality-information Differentiation for Audio-visual Deepfake Detection and Localization (https://arxiv.org/abs/2411.10193)
- **What's New**: 딥페이크(deepfake) 기술이 급속히 발전하면서 정보의 무결성과 사회적 신뢰에 큰 위협이 되고 있습니다. 본 논문에서는 오디오와 비주얼 모달리티의 동시 조작을 감지하기 어려운 문제를 해결하기 위해, 새로운 오디오-비주얼 딥페이크 탐지 프레임워크인 DiMoDif를 제안하고 있습니다.

- **Technical Details**: DiMoDif는 음성과 관련된 머신 지각의 인터모달리티 차이를 활용하여 딥페이크를 탐지합니다. 입력 비디오는 시각적 및 오디오 스트림으로 분해되고, 각각의 표현을 추출합니다. 이 아키텍처는 Transformer encoder 기반으로 구축되어 있으며, 로컬 어텐션을 이용하여 두 모달리티 간의 크로스 모달 일치성을 찾아냅니다.

- **Performance Highlights**: 디모디프(DiMoDif)는 AV-Deepfake1M 데이터셋에서 Temporal Forgery Localization 작업에 대해 +47.88% AP@0.75 성능 향상을 보이며, Deepfake Detection 작업에서도 AV-Deepfake1M에서 +30.5% AUC, FakeAVCeleb에서 +2.8% AUC 향상을 보여줍니다. LAV-DF 데이터셋에서는 동등한 성능을 발휘합니다.



### NeISF++: Neural Incident Stokes Field for Polarized Inverse Rendering of Conductors and Dielectrics (https://arxiv.org/abs/2411.10189)
- **What's New**: 이번 연구에서는 도체(conductor)와 유전체(dielectric) 모두를 지원하는 역 렌더링(inverse rendering) 파이프라인인 NeISF++를 제안합니다. 특히, 이전의 도체 처리 방법들이 유체적인 반사 속성을 무시했음을 지적하고 이를 해결하기 위한 일반적인 pBRDF(Polarimetric Bidirectional Reflectance Distribution Function)를 소개합니다.

- **Technical Details**: NeISF++는 다중 시점 polarized 이미지(multiview polarized images)를 입력으로 사용하여 기하학을 SDF(Signed Distance Field)로 모델링하고, 재료를 BRDF 필드로, 다중 반사된 polarization 빛을 incident Stokes field로 나타냅니다. 금속 및 도체의 강한 specular 반사를 해결하기 위해 DoLP(Degree of Linear Polarization) 이미지를 활용한 새로운 기하학 초기화 방법을 제안하였습니다.

- **Performance Highlights**: 실험 결과, NeISF++는 기하학 및 재료 분해에서 기존의 polarization 역 렌더링 방법들보다 성능이 우수하며, 재조명(relighting) 같은 다운스트림 작업에서도 현저하게 더 현실적인 결과를 제공합니다.



### Try-On-Adapter: A Simple and Flexible Try-On Paradigm (https://arxiv.org/abs/2411.10187)
Comments:
          Image virtual try-on, 7 pages, 3 figures

- **What's New**: 이 논문은 기존의 이미지를 기반으로 한 virtual try-on 방법들과는 다르게, Try-On-Adapter (TOA)를 제안하며, 이 방법은 'outpainting' 패러다임을 기반으로 하여 주어진 얼굴과 의상 정보를 통해 나머지 이미지를 자연스럽게 생성하는 것을 목표로 합니다.

- **Technical Details**: Try-On-Adapter는 기존의 'inpainting' 방식을 탈피하고, 사용자가 제공하는 얼굴 이미지와 참조 의상 이미지만을 가지고 시각적으로 완성된 이미지를 생성합니다. 이는 미리 훈련된 모델을 기반으로 그리고 이미지-as-prompt 기법을 활용하여 입력된 정보를 자연스럽게 결합합니다.

- **Performance Highlights**: TOA는 VITON-HD 데이터셋에서 FID 점수 5.56과 7.23을 기록하여 쌍(pair)과 비쌍(unpaired) 비교 모두에서 최첨단 성능을 달성하였으며, 기존의 OOTDiffusion보다 우수한 질적 결과를 보였습니다.



### Efficient Progressive Image Compression with Variance-aware Masking (https://arxiv.org/abs/2411.10185)
Comments:
          10 pages. Accepted at WACV 2025

- **What's New**: 본 논문에서는 Learned Progressive Image Compression 방법을 제안하여, 기본 품질과 최고 품질의 잠재 표현을 쌍으로 활용합니다. 이 시스템은 급격한 품질 개선을 가능하게 하는 독립적인 마스킹 시스템을 도입합니다.

- **Technical Details**: 이미지는 초기 두 개의 레벨로 구성된 latent representation(잠재 표현)으로 나타내며, residual latent representation(잔여 잠재 표현)이 각각의 요소의 차이를 인코딩하여 3단계로 분리됩니다. 학습 가능한 Rate Enhancement Modules(REMs)를 통해 엔트로피 매개변수의 추정을 개선합니다. 각 요소는 가장 중요도에 따라 랭크되고, 이를 통해 필요한 요소만 전송할 수 있습니다.

- **Performance Highlights**: 이 방법은 경쟁 업체와 견주어도 RD(율-왜곡) 성능에서 경쟁력 있는 결과를 보이며, 계산 복잡성 및 디코딩 시간을 크게 줄였습니다. 이 시스템은 고해상도 이미지에서도 효율적인 압축을 지원합니다.



### Visual question answering based evaluation metrics for text-to-image generation (https://arxiv.org/abs/2411.10183)
Comments:
          Accepted to ISCAS2024

- **What's New**: 본 연구에서는 텍스트 기반 이미지 생성과 관련된 새로운 평가 지표를 제안합니다. 기존의 평가 방법은 전체적인 텍스트와 이미지 간의 정렬만을 측정하는 반면, 본 연구에서는 개별 객체를 기반으로 평가하는 방식을 채택합니다.

- **Technical Details**: 제안된 방법은 Visual Question Answering(VQA)와 Non-Reference Image Quality Assessment(NR-IQA)를 통합하여 이미지의 텍스트 정렬과 품질 두 가지를 동시에 평가합니다. VQA는 입력 텍스트에 기반해 생성된 이미지의 적합성을 평가하며, NR-IQA는 이미지 품질을 자동으로 평가합니다. 사용자는 이 두 점수의 가중치를 조정할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 CLIPScore [10] 및 기존의 NR-IQA 방법인 MANIQA [12]와 비교할 때 이미지 품질 평가와 텍스트-이미지 정렬을 동시에 반영할 수 있는 우수성을 보여줍니다. 또한, ImageReward [13]와의 비교에서도 유사한 평가 정확도를 유지하며 가중치 조정이 가능하다는 점에서 더 뛰어난 성능을 입증하였습니다.



### CART: Compositional Auto-Regressive Transformer for Image Generation (https://arxiv.org/abs/2411.10180)
Comments:
          under review at CVPR 2025

- **What's New**: 최근 이미지 합성 분야에서 혁신적인 접근법이 소개되었습니다. Auto-Regressive (AR) 모델링을 활용한 새로운 이미지 생성 방식은 세부 예측 전략을 통해 높은 충실도와 확장성을 제공합니다.

- **Technical Details**: 제안된 방법은 이미지의 기본 요소와 세부 요소를 계층적으로 결합하여 점진적으로 이미지를 구성합니다. 이 과정은 부드러운 기본 이미지를 생성하고, 세부 요소를 반복적으로 추가하여 최종 이미지를 완성하는 방식으로, 인지적으로 자연스러운 이미지 생성 방식을 모방합니다.

- **Performance Highlights**: 이 방법은 기존의 next-token 예측 방식보다 더 효과적이며, 최신 next-scale 예측 접근법을 초월합니다. 특히, 전체 모델 재훈련 없이도 고해상도로의 확장이 가능하여, 고해상도 이미지 생성에 있어 유연한 솔루션으로 자리 잡을 수 있습니다.



### SEAGULL: No-reference Image Quality Assessment for Regions of Interest via Vision-Language Instruction Tuning (https://arxiv.org/abs/2411.10161)
- **What's New**: 본 논문은 Regions of Interest (ROIs)에 대한 세부 품질 분석을 탐구하는 새로운 방법론, SEAGULL을 제안합니다. 기존의 Image Quality Assessment (IQA) 방법들은 전체 이미지 품질 분석에 주로 집중했으나, SEAGULL은 세부적인 지역 품질 개선을 위한 지침을 제공합니다.

- **Technical Details**: SEAGULL은 Vision-Language Model (VLM)과 Segment Anything Model (SAM)을 결합하여 ROI 품질을 평가합니다. 이 네트워크는 Mask-based Feature Extractor (MFE)를 통해 글로벌 및 로컬 토큰을 추출하여 세부적인 품질 평가를 가능하게 하며, 새로운 ROI 기반 IQA 데이터셋인 SEAGULL-100w 및 SEAGULL-3k를 구축하여 훈련 및 평가에 사용합니다.

- **Performance Highlights**: SEAGULL은 SEAGULL-100w에서 사전 훈련을 거친 후, SEAGULL-3k에서 미세 조정하여, 기존의 IQA 모델과 VLM 대비 뛰어난 성능을 보여주며, 세부적인 ROI 품질 분석에서 탁월성을 입증합니다.



### Outliers resistant image classification by anomaly detection (https://arxiv.org/abs/2411.10150)
Comments:
          19 pages, in Russian

- **What's New**: 본 연구에서는 수동 조립 공정을 자동으로 모니터링하기 위해 새로운 모델을 제안합니다. 이 모델은 클래스 분류와 이상 탐지를 동시에 수행하며, 다양한 환경 조건의 변동에 강한 특성을 가지고 있습니다.

- **Technical Details**: 제안된 모델은 메트릭 학습(metric learning)을 활용하여 이미지의 벡터 표현을 다차원 공간에서 생성합니다. 이후, 교차 엔트로피(cross-entropy) 방식으로 분류를 수행합니다. 실험을 위해 327,000개 이상의 이미지가 포함된 데이터셋이 준비되었습니다.

- **Performance Highlights**: 다양한 컴퓨터 비전 모델 아키텍처에서 실험이 이루어졌으며, 각 접근 방식의 결과를 비교하여 최적의 성능을 도출했습니다.



### Matrix-Valued LogSumExp Approximation for Colour Morphology (https://arxiv.org/abs/2411.10141)
Comments:
          42 pages, 10 figures, to be submitted in JMIV

- **What's New**: 이 논문은 색상에 대한 수학적 형태학의 새로운 접근 방식을 제안합니다. 기존의 suprema 정의를 LogExp 근사로 대체하여 고차원 데이터에서의 연관성을 보존하는 기법을 소개합니다.

- **Technical Details**: Burgeth와 Kleefeld의 방법을 기반으로 하여, 우리는 대칭적인 $2 	imes 2$ 행렬로 색상을 처리하고 Loewner 순서를 통해 이를 비교합니다. 이 과정에서 LogSumExp 근사를 활용하여 최대값을 근사하는 방법을 제시하며, 고차원에서의 이 시나리오의 연관성을 유지합니다. 또한, 연속적 종속성을 보장하기 위해 최소성 속성을 탐구합니다.

- **Performance Highlights**: 제안된 방식은 고차원 형태학 접근 방식의 많은 장점을 결합하면서도, dilation의 결합성을 유지하는 것을 목표로 하고 있습니다. 이는 현재까지 다른 다차원 접근 방식에서 보장되지 않는 특성입니다.



### CoSAM: Self-Correcting SAM for Domain Generalization in 2D Medical Image Segmentation (https://arxiv.org/abs/2411.10136)
- **What's New**: 이 논문에서는 기존의 manual prompts 의 필요성을 해소하기 위해 Self-Correcting SAM (CoSAM)이라는 새로운 2D 의료 이미지 분할 방법을 제안합니다.

- **Technical Details**: CoSAM은 SAM을 사용하여 coarse masks 를 prompt-free 방식으로 생성하고, 이를 교정하는 generalized error decoder 를 도입합니다. 최종적으로는 정교화된 masks 를 기반으로 다양한 prompts 를 생성하여 self-correcting loop 내에서 예측을 반복적으로 개선합니다.

- **Performance Highlights**: CoSAM은 두 개의 의료 이미지 분할 벤치마크에서 기존의 SAM 기반 방법들과 비교할 때, 여러 도메인 일반화 시나리오에서 우수한 성능을 보였습니다.



### Efficient Density Control for 3D Gaussian Splatting (https://arxiv.org/abs/2411.10133)
- **What's New**: 이번 논문은 3D Gaussian Splatting (3DGS)의 효율성을 향상시키기 위한 새로운 방법을 제안합니다. 기존의 clone 및 split 작업의 한계를 극복하고, Gaussian의 밀도가 불필요하게 낮아지는 문제를 해결하기 위해 long-axis split을 도입했습니다.

- **Technical Details**: 이번 연구에서는 3DGS의 적응적 밀도 제어(adaptive density control)를 개선하였습니다. 새로운 long-axis split을 통해 Gaussian의 겹침(overlap)을 줄이고, adaptive pruning 기술을 통해 낮은 투명도(low-opacity)의 Gaussian 수를 감소시켰습니다. 추가로 splitting threshold를 동적으로 조정하고 중요도 가중치(importance weighting)를 활용하여 Gaussian의 이용 효율을 높였습니다.

- **Performance Highlights**: Mip-NeRF 360 데이터셋에서 Gaussians 수를 30% 줄이면서 평균 PSNR을 25.52에서 25.82로 향상시켰습니다. 이러한 개선된 방법은 효율적인 렌더링 속도와 품질 모두를 향상시키는 성능을 보였습니다.



### Towards Multi-View Consistent Style Transfer with One-Step Diffusion via Vision Conditioning (https://arxiv.org/abs/2411.10130)
Comments:
          Accepted by ECCV 2024 AI for Visual Arts Workshop and Challenges, 18 pages, 7 figures

- **What's New**: 본 논문에서는 3D 장면의 스타일화 문제를 다루며, OSDiffST라는 새로운 스타일 전송 방법을 제안합니다. 이는 사전 훈련된 한 단계 확산 모델(SD-Turbo)을 활용하여 다양한 스타일을 3D 장면의 다중 시점 이미지에 효과적으로 적용할 수 있습니다.

- **Technical Details**: 전통적인 2D 스타일 전송 방법은 구조적 정보와 다중 시점 일관성을 유지하는 데 어려움을 겪습니다. 본 연구에서는 LoRA 기법을 사용하여 사전 훈련된 모델의 파라미터 수를 줄이고, 비전 조건 모듈을 도입하여 스타일 정보 추출을 최적화했습니다. 두 가지 손실 함수를 통해 색상 분포와 구조적 유사성을 정렬하여 스타일화된 이미지의 품질을 높였습니다.

- **Performance Highlights**: OSDiffST는 3D 장면의 다중 시점에서 다양한 스타일을 합성하는 데 있어 우수한 성능을 보입니다. 실험 결과, 우리가 제안한 방법은 다른 스타일 전송 방법들보다 뛰어난 시각적 품질과 적은 왜곡으로 최종 이미지를 생성함을 증명했습니다.



### Multi-Task Adversarial Variational Autoencoder for Estimating Biological Brain Age with Multimodal Neuroimaging (https://arxiv.org/abs/2411.10100)
- **What's New**: 본 연구에서는 구조적 MRI(Structural MRI)와 기능적 MRI(Functional MRI) 데이터를 통합하여 뇌 나이를 예측하기 위한 멀티태스크 적대적 변이 자동 인코더(Multitask Adversarial Variational Autoencoder, M-AVAE)를 제안합니다. 이 모델은 공유 및 특정한 특성을 분리하여 더 나은 예측을 수행합니다.

- **Technical Details**: M-AVAE는 적대적 학습(Adversarial Learning)과 변이 자동 인코딩(Variational Autoencoding) 기능을 통합하여 구조화된 프레임워크를 제공합니다. 이 모델은 남녀의 뇌 노화 패턴을 포착하기 위해 성별 분류를 추가 과제로 통합합니다. 다양한 손실 함수(Loss Functions)를 사용해 최적화를 진행합니다.

- **Performance Highlights**: OpenBHB 데이터셋에서 평가한 결과, M-AVAE는 평균 절대 오차(Mean Absolute Error) 2.77세를 기록하며 기존의 방법들을 초월하여 뇌 나이 추정의 새로운 기준을 제시하고 있습니다.



### CorrCLIP: Reconstructing Correlations in CLIP with Off-the-Shelf Foundation Models for Open-Vocabulary Semantic Segmentation (https://arxiv.org/abs/2411.10086)
- **What's New**: 본 논문에서는 Open-vocabulary semantic segmentation (OVSS)을 위한 새로운 접근 방법 CorrCLIP을 제안합니다. 이 방법은 사전 정의된 카테고리 세트 없이 각 픽셀에 의미론적 레이블을 지정할 수 있도록 합니다.

- **Technical Details**: CorrCLIP은 이미지 패치 간의 관련성을 더욱 일관되게 재구성하기 위해 Segment Anything Model (SAM)을 이용하여 의미적으로 유사한 패치들만 상호작용할 수 있도록 제한합니다. 이를 통해 패치 간의 상관관계를 명확히 하여 세그멘테이션을 향상시킵니다. 또한, DINO 모델을 활용하여 패치 간의 유사성 값을 결정함으로써 유사성 불규칙성 문제를 해결합니다.

- **Performance Highlights**: CorrCLIP은 8개의 도전적인 벤치마크에서 평균 mean Intersection over Union (mIoU)을 44.4%에서 51.0%로 대폭 향상시켰습니다. 이는 기존의 최첨단 방법들보다 평균 6.6% 향상된 성능입니다.



### Influence of Depth Camera Noise Models on Respiration Estimation (https://arxiv.org/abs/2411.10081)
Comments:
          Poster Prague 2023 Conference, 4 pages

- **What's New**: 이 연구는 다양한 노이즈 모델을 기반으로, 3D 렌더링 시뮬레이션 파이프라인을 구축하여 깊이 카메라(Depth Camera)를 활용한 현실적인 호흡 신호를 생성하는 방법을 제안합니다. 이는 다중 카메라 환경에서 새로운 알고리즘을 테스트하기 위한 데이터 생성에 사용될 수 있습니다.

- **Technical Details**: 연구에서는 두 단계로 구성된 방법론을 사용합니다. 첫 번째 단계에서 실제 및 시뮬레이션된 호흡 신호를 사용하여 인체의 흉부 움직임을 조절하고, 두 번째 단계에서 6가지 노이즈 모델을 바탕으로 인위적인 노이즈를 추가합니다. 3D 모델링 소프트웨어인 Blender를 이용하여 인간 호흡을 시뮬레이션하였고, 가우시안 노이즈, 축 방향 정량화 노이즈, 방사형 노이즈, 모션 블러, 그리고 주변 픽셀에서의 강한 노이즈를 모델링했습니다.

- **Performance Highlights**: 시뮬레이션된 깊이 카메라 신호에서 인위적인 노이즈의 영향은 실제 호흡 신호에서와 유사하였습니다. 적절한 신호 대 잡음 비(SNR)와 스케일을 가지며, 잡음 신호에서 추출된 결과는 정성적으로 양호한 결과를 보여주었습니다.



### Uncertainty-Weighted Mutual Distillation for Multi-View Fusion (https://arxiv.org/abs/2411.10077)
- **What's New**: 이번 연구에서는 Multi-View Uncertainty-Weighted Mutual Distillation (MV-UWMD)라는 새로운 방법을 제안합니다. 이는 서로 다른 각도와 위치에서 캡처된 이미지의 일관성을 높이고 예측의 변동성을 줄이는 데 초점을 맞추고 있습니다.

- **Technical Details**: MV-UWMD는 계층적 상호 증류(hierarchical mutual distillation)를 통해 모든 가능한 뷰 조합, 즉 단일 뷰, 부분 다중 뷰, 전체 다중 뷰 예측을 예측합니다. 이 방법은 상호 증류를 통해 불확실성 기반의 가중치 메커니즘을 도입하며, 이는 각 뷰에서 고유한 정보를 효과적으로 활용하면서도 불확실한 예측의 영향을 완화합니다. 또한, CNN-Transformer 하이브리드 아키텍처를 확장하여 다수의 뷰 조합을 통한 강력한 특징 학습(feature learning) 및 통합을 지원합니다.

- **Performance Highlights**: 대규모 비구조적 데이터셋을 사용한 실험 결과, MV-UWMD는 기존의 다중 뷰 학습(multi-view learning) 방법보다 예측 정확도와 일관성을 향상시킨 것으로 나타났습니다.



### Improving the accuracy of automated labeling of specimen images datasets via a confidence-based process (https://arxiv.org/abs/2411.10074)
- **What's New**: 이 논문은 자연사 컬렉션에서 디지털화된 데이터의 정확성을 획기적으로 향상시킬 수 있는 새로운 접근 방식을 소개합니다. 특히, 신뢰도 기반의 분류Pipeline을 통해 기존 자동 레이블링 기법의 정확성을 80-85%에서 95% 이상 또는 99% 이상으로 끌어올릴 수 있음을 보여줍니다.

- **Technical Details**: 이 연구는 softmax 기반 분류기에서의 신뢰도 기반 분류 체계를 제안하며, 이는 사용자가 정의하는 임계값을 사용하여 낮은 신뢰도의 레이블을 거부하는 방식으로 작동합니다. 그 결과, 사용자는 정확성과 범위의 trade-off를 고려하여 각 애플리케이션에 적합한 모델을 선택할 수 있습니다.

- **Performance Highlights**: 실험 결과, 초기 정확도가 86%인 모델이 높은 신뢰도 임계값을 선택함으로써 40%의 레이블을 거부하여 정확도를 95% 이상으로 향상시켰고, 약 65%의 레이블을 거부하여 99% 이상의 정확도를 달성함을 증명했습니다.



### Real-Time AI-Driven People Tracking and Counting Using Overhead Cameras (https://arxiv.org/abs/2411.10072)
Comments:
          This paper is accepted to IEEE Region 10 conference (TENCON) 2024

- **What's New**: 본 연구에서는 스마트 빌딩과 지능형 교통 시스템에서 사람을 정확하게 세는 새로운 방법을 제안합니다. 이 방법은 새로운 객체 추적 알고리즘, 카운팅 알고리즘 및 조정된 객체 탐지 모델을 결합하여 실시간 사람 세기에서 97%의 정확도를 달성합니다. 또한, 저전력 엣지 컴퓨터에서 평균 20-27 FPS의 프레임 속도를 보입니다.

- **Technical Details**: 제안된 방법은 이미지를 통해 사람의 머리를 검출하고, 이 검출된 머리의 특징을 추출하여 이전 프레임과 비교하는 방식으로 작동합니다. 이 과정에서 SSD Mobilenet 모델을 사용하여 머리의 객체 탐지를 수행하며, MobileNetV2 모델을 사용하여 feature extraction을 최적화합니다. 두 가지 서로 다른 모델을 사용해 낮과 밤의 조건에서 입체적으로 탐지를 수행하고, 공통적인 장애물과의 혼동을 피하기 위해 모델을 미세 조정합니다.

- **Performance Highlights**: 본 연구의 방법론은 97%의 정확도를 기록하며, 프레임 속도는 평균 20-27 FPS로, 기존 방법에 비해 2% 개선된 성과를 보였습니다. 이는 다양한 환경 및 조명 조건에서도 신뢰성 있는 실시간 사람 세기를 가능하게 합니다.



### Evidential Federated Learning for Skin Lesion Image Classification (https://arxiv.org/abs/2411.10071)
Comments:
          Published as a conference paper at ICPR 2024

- **What's New**: FedEvPrompt는 evidential deep learning, prompt tuning, knowledge distillation을 통합하여 분산된 피부 병변 분류를 위한 새로운 연합 학습 접근 방식을 제시합니다. 이 방법은 기존의 모델 파라미터 공유 대신 attention maps를 통해 지식을 공유하여 개인 정보 보호를 강화합니다.

- **Technical Details**: FedEvPrompt는 frozen pre-trained Vision Transformer (ViT) 모델에 b-prompts(저수준 기본 시각적 지식) 및 t-prompts(작업 특정 지식)를 결합하여 학습합니다. 학습 과정은 round-based learning paradigm 내에서 이루어지며, 각 라운드는 로컬 모델 훈련 후 attention maps를 공유하는 방식입니다. 이러한 구조는 클래스 증거를 극대화하고, 데이터 불균형과 비독립적이고 동일하게 분포되지 않은(non-i.i.d.) 데이터 문제를 해결하는 데 기여합니다.

- **Performance Highlights**: ISIC2019 데이터셋을 사용한 실험 결과, FedEvPrompt는 기존 연합 학습 알고리즘 및 knowledge distillation 방법들에 비해 뛰어난 성능을 보여줍니다. 특히, 모델 파라미터를 공유하지 않고도 우수한 결과를 달성하였습니다.



### Step-wise Distribution Alignment Guided Style Prompt Tuning for Source-free Cross-domain Few-shot Learning (https://arxiv.org/abs/2411.10070)
Comments:
          15 pages, 12 figures, 7 tables

- **What's New**: 이 논문에서는 기존의 cross-domain few-shot learning (CDFSL) 방법의 한계를 극복하기 위한 새로운 접근 방법인 source-free CDFSL (SF-CDFSL)을 제안합니다. SF-CDFSL에서는 source 데이터 없이 미리 훈련된 모델과 제한된 target 샘플을 사용하여 목표 도메인의 few-shot learning 문제를 해결합니다.

- **Technical Details**: Step-wise Distribution Alignment Guided Style Prompt Tuning (StepSPT) 방법론을 통해 target 샘플이 원하는 분포와 일치할 수 있도록 스타일 프롬프트를 제안합니다. StepSPT는 외부 과정과 내부 과정으로 구성된 이중 단계 최적화 과정을 채택하여 LMs의 파라미터를 고정한 채 스타일 프롬프트와 분류기 파라미터를 반복적으로 업데이트합니다.

- **Performance Highlights**: 5개의 데이터세트에 대한 평가 결과, StepSPT가 기존의 프롬프트 조정 기반 방법과 SOTA에 비해 우수한 성능을 보이며, 각 구성 요소의 기여도를 입증하는 ablation 연구 결과도 제시되었습니다.



### Diachronic Document Dataset for Semantic Layout Analysis (https://arxiv.org/abs/2411.10068)
- **What's New**: 이 논문은 텍스트 인코딩 이니셔티브(TEI) 표준과 매핑된 문서 재생 워크플로우를 지원하기 위해 설계된 새로운 오픈 액세스 데이터셋을 소개합니다.

- **Technical Details**: 이 데이터셋은 1600년부터 2024년까지 다양한 문서 유형(잡지, 과학 및 인문학 논문, 박사 논문, 단행본, 희곡, 관리 보고서 등)의 7,254개의 주석이 달린 페이지로 구성되어 있습니다. 이 데이터셋은 다양한 시대와 장르의 콘텐츠를 포함하여 문서 구조의 역사적 변화를 반영하고, 레이아웃 복잡성을 다룹니다. 모듈형 디자인은 분야별 구성 설정을 가능하게 합니다.

- **Performance Highlights**: 객체 탐지 모델에 대한 평가 결과, YOLO의 경우 1280픽셀 입력 크기가 최적이며, 서브셋 기반 훈련은 사전 훈련된 가중치를 미세 조정하는 것보다 일반 모델에 서브셋을 통합하여 훈련하는 것이 일반적으로 이점이 있음을 보여줍니다.



### Rethinking Normalization Strategies and Convolutional Kernels for Multimodal Image Fusion (https://arxiv.org/abs/2411.10036)
- **What's New**: 이번 논문은 기존의 멀티모달 이미지 융합(MMIF) 방법들이 자연 이미지 융합에 중점을 두고, 의료 이미지 융합의 고유한 특성을 간과하고 있다는 점을 지적합니다. 저자들은 IVIF(Infrared and Visible Image Fusion)와 MIF(Medical Image Fusion) 간의 주요 차이점을 분석하고, 연속적인 이미지 융합을 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 논문에서는 인스턴스 정규화(Instance Normalization)와 그룹 정규화(Group Normalization)를 혼합하여 샘플 독립성을 유지하고, 이미지 세부 정보를 증강하기 위한 새로운 전략을 제안합니다. 이와 함께 대형 커널 컨볼루션(Large Kernel Convolution)을 도입하여 수용 영역을 확장하고 이미지 세부 정보의 보존을 향상시킵니다. 또한, 다양한 스케일과 수용 영역에서 피처를 조정하는 다중 경로 적응 융합 모듈을 통해 정보 전송을 최적화합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 방법은 여러 융합 작업에서 최고 성능을 보였으며, 다운스트림 응용 프로그램에서도 유의미한 개선을 가져왔습니다. 특히 의료 영상의 시각화와 멀티모달 객체 탐지 및 의미 분할(semantic segmentation) 작업에서의 성능 향상이 컸습니다.



### GSEditPro: 3D Gaussian Splatting Editing with Attention-based Progressive Localization (https://arxiv.org/abs/2411.10033)
Comments:
          Pacific Graphics 2024

- **What's New**: 본 논문에서는 GSEditPro라는 새로운 3D 장면 편집 프레임워크를 제안합니다. 사용자가 텍스트 프롬프트만으로 다양한 창의적이고 정밀한 편집을 수행할 수 있도록 설계되었습니다.

- **Technical Details**: GSEditPro는 3D Gaussian Splatting(3D-GS)의 명시적 표현을 활용하여 각 Gaussian에 의미 레이블을 추가하는 attention 기반적 진보적 위치 지정 모듈을 도입합니다. 이를 통해 T2I 모델의 크로스-어텐션 레이어에서 파생된 편집 프롬프트와 관련된 Gaussian을 분류하여 편집 영역을 정확하게 지정할 수 있습니다.

- **Performance Highlights**: 실험을 통해 GSEditPro는 객체 변경 및 객체 삽입에서 정밀한 편집을 수행할 수 있으며, 편집 후 불필요한 영역은 자연스럽게 보존됩니다. 또한, 간단한 텍스트 프롬프트를 통해 편집이 가능하여 사용자 친화적이며, 이전 방법들과 비교하여 편집 정확도 및 시각적 충실도에서 우수한 성능을 보입니다.



### VMID: A Multimodal Fusion LLM Framework for Detecting and Identifying Misinformation of Short Videos (https://arxiv.org/abs/2411.10032)
Comments:
          arXiv admin note: text overlap with arXiv:2211.10973 by other authors

- **What's New**: 본 논문은 기존의 단일 모달 기반 가짜 뉴스 탐지 방법의 한계를 극복하기 위해 다중 모달 정보를 활용한 새로운 가짜 뉴스 탐지 방법(VMID)을 제안합니다. 이 방법은 비디오 콘텐츠의 다층 분석을 통해 허위 정보를 식별할 수 있도록 설계되었습니다.

- **Technical Details**: VMID 프레임워크는 비디오에서 시각, 오디오, 텍스트 정보를 통합하여 LLM(대형 언어 모델)에 입력할 수 있는 통합 프롬프트를 생성합니다. 이를 통해 비디오의 메타데이터 및 소셜 컨텍스트 신호를 포함해 정확한 가짜 뉴스 탐지가 가능합니다.

- **Performance Highlights**: VMID는 실험을 통해 기존 모델 대비 약 9.87%에서 19.6% 향상된 매크로 F1 점수를 기록하였으며, 정확도에서는 90.93%를 달성하여 가장 우수한 기준 모델(SV-FEND)의 81.05%를 크게 초과하는 성과를 보였습니다.



### Toward Robust and Accurate Adversarial Camouflage Generation against Vehicle Detectors (https://arxiv.org/abs/2411.10029)
Comments:
          14 pages. arXiv admin note: substantial text overlap with arXiv:2402.15853

- **What's New**: RAUCA라는 새로운 방어적 마스킹 생성 방법을 제안합니다. 이 방법은 차량 텍스처를 정확하게 최적화하고 조명 및 날씨와 같은 환경적 특성을 반영하여 이미지를 렌더링하는 End-to-End Neural Renderer Plus (E2E-NRP)를 중심으로 구성됩니다.

- **Technical Details**: RAUCA는 기존 기술의 한계를 극복하고, 물리적 공격인 adversarial camouflage를 생성하기 위해 differentiable neural renderers를 활용합니다. 이 방법은 다각적인 날씨 조건을 반영하는 다중 날씨 데이터셋을 통합하여 camouflaging의 강인성을 높입니다.

- **Performance Highlights**: 여섯 가지 인기 있는 객체 감지기에 대한 실험 결과, RAUCA는 시뮬레이션 및 실제 환경에서 기존 방법보다 더 뛰어난 성능을 보였습니다.



### MOT\_FCG++: Enhanced Representation of Motion and Appearance Features (https://arxiv.org/abs/2411.10028)
Comments:
          12 pages, 7 figures

- **What's New**: 이번 논문은 multi-object tracking (MOT)에서 객체의 외관(appearance)과 공간적 특성(spatial features)을 효과적으로 표현하기 위한 새로운 접근법을 제안합니다. 기존 방법인 MOT_FCG를 기반으로 하여, 더 글로벌한 appearance 임베딩 이론과 정확한 공간 운동 정보를 제공하는 방법을 개선했습니다.

- **Technical Details**: 제안된 방법에는 Diagonal Modulated GIoU라는 새로운 spatial metric이 포함되어 있어 객체의 위치 및 형태 간의 관계를 더 정확하게 나타냅니다. 또한, 동적 appearance 표현을 통해 신뢰도 정보를 통합하여 경로의 appearance 특성을 더 강인하고 전반적으로 개선하였습니다. 성능 측정 기준으로 76.1 HOTA, 80.4 MOTA, 81.3 IDF1을 MOT17 검증 세트에서 달성했습니다.

- **Performance Highlights**: MOT17, MOT20 및 DanceTrack 데이터셋에서 뛰어난 성능을 보였으며, 특히 MOT17 데이터셋에서는 기존 상태를 초과하는 성능을 기록했습니다.



### Towards Utilising a Range of Neural Activations for Comprehending Representational Associations (https://arxiv.org/abs/2411.10019)
Comments:
          18 pages, 11 figures

- **What's New**: 최근 심층 신경망에서 중간 표현을 이해하기 위한 접근 방식이 개별 신경세포와 선형 방향을 분석하여 해석되고 있습니다. 하지만 본 연구에서는 이러한 방법이 표현의 복잡한 행동을 포착하지 못한다는 것을 보여줍니다. 비극대활성화(neural network activations)는 일반적으로 밀집되어 있으며, 비극대 수준의 활성화가 혼란스러운 인간 해석 가능 개념을 찾아내는 데 유용하다는 가설을 세웠습니다.

- **Technical Details**: 본 연구는 중간 수준 출력 신경세포 활성화를 사례로 하여, 비극대 활성화가 신경망의 마지막 계층에서의 표현을 이해하는 데 어떻게 기여할 수 있는지를 탐구했습니다. 특히, 합성 데이터셋에서 중간 수준 활성화가 극대화된 활성화 예시만으로는 드러나지 않는 표현의 측면을 알려주는 방법을 보여주었습니다.

- **Performance Highlights**: 제안된 방법은 비극대 활성화를 검사하여 모델이 학습한 복잡한 관계를 추출하는 유용성을 시연하며, 실제 벤치마크 데이터셋에서의 성능을 향상시키는 데 성공적이었습니다. 또한, 비극대 활성화를 통해 잘못 라벨링된 샘플을 식별하고, 모델이 의존하고 있는 스푸리어스(correlations) 패턴을 완화하는 데 유용한 데이터를 선별할 수 있음을 입증하였습니다.



### MicroCrackAttentionNeXt: Advancing Microcrack Detection in Wave Field Analysis Using Deep Neural Networks through Feature Visualization (https://arxiv.org/abs/2411.10015)
- **What's New**: 본 논문에서는 MicroCrackAttentionNeXt라는 새로운 모델을 제안하여, 기존 SpAsE-Net을 개선하여 미세 크랙 감지에서의 성능을 향상시킵니다. 이 모델은 비대칭 인코더-디코더 구조를 사용하며, 중요한 크랙 패턴을 포착하는 데 집중하여 정확도를 높입니다.

- **Technical Details**: MicroCrackAttentionNeXt는 spatio-temporal (시공간) 데이터 분석을 통해 복잡한 크랙을 인식하도록 설계되었으며, 다양한 activation (활성화) 및 loss functions (손실 함수)의 영향을 분석합니다. 이 과정에서 Manifold Discovery Analysis (MDA)를 활용하여 최적의 활성화 함수를 선택하는 방법론을 제시합니다.

- **Performance Highlights**: 제안된 모델은 최적화된 아키텍처와 훈련 방법론을 통해 86.85%의 정확도를 달성하였습니다. 이런 성과는 기존의 비율이 낮은 데이터셋에서 클래스 불균형 문제를 효과적으로 해결한 결과입니다.



### Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses (https://arxiv.org/abs/2411.10013)
- **What's New**: 이 논문은 증강 현실(AR) 애플리케이션을 위한 스테레오 깊이 추정에서의 전통적인 모델들의 단점을 극복하고, 고비용의 cost volume과 전처리 단계를 줄이기 위한 새로운 두 가지 모델, MultiHeadDepth와 HomoDepth를 개발했습니다.

- **Technical Details**: 기존의 cost volume 프로세스를 새로운 group-pointwise convolution 기반의 연산자로 대체하고, layernorm과 dot product를 이용한 코사인 유사도의 효율적 근사를 채택합니다. 전처리 과정 없이 원시 이미지(미정렬 이미지)를 처리할 수 있는 homography matrix prediction network를 도입하여 온라인 스테레오 직선을 구현합니다.

- **Performance Highlights**: MultiHeadDepth는 AR 글래스의 최신 깊이 추정 모델에 비해 정확도를 11.8-30.3% 향상시키고, 지연을 22.9-25.2% 줄였습니다. HomoDepth는 미정렬 이미지를 처리하면서 전체 지연 시간을 44.5% 줄이는 성능을 보였습니다.



### Unlocking Transfer Learning for Open-World Few-Shot Recognition (https://arxiv.org/abs/2411.09986)
- **What's New**: 본 논문은 Few-Shot Open-Set Recognition (FSOSR)을 위한 새로운 두 단계 학습 방법인 OAL-OFL(Open-set Aware Learning - Open-set Free Learning)을 제안하며, 일반적인 transfer learning 방식이 FSOSR에 효과적으로 적용되지 못하는 문제를 해결합니다.

- **Technical Details**: OAL-OFL 방법은 two-stage 방식으로 구성됩니다. 첫 번째 단계인 open-set aware meta-learning에서는 유용한 metric space를 설정하여 후속 단계의 기초가 됩니다. 두 번째 단계인 open-set free transfer learning에서는 메타 학습을 통해 얻은 파라미터로 모델을 초기화하고, open-set 예시의 부재를 해결하기 위해 두 가지 샘플링 전략을 도입합니다.

- **Performance Highlights**: 제안된 방법은 miniImageNet 및 tieredImageNet 데이터셋에서 state-of-the-art (SOTA) 성능을 달성하였으며, 추가적인 학습 비용은 1.5%에 불과합니다. 이로 인해 OAL-OFL은 FSOSR의 일반화 능력 및 성능을 크게 향상시킵니다.



### Explanation for Trajectory Planning using Multi-modal Large Language Model for Autonomous Driving (https://arxiv.org/abs/2411.09971)
Comments:
          Accepted and presented at ECCV 2024 2nd Workshop on Vision-Centric Autonomous Driving (VCAD) on September 30, 2024. 13 pages, 5 figures

- **What's New**: 이번 연구는 자율주행 차량의 미래 행동과 그 이유를 설명하는 캡션을 생성하는 새로운 추론 모델을 제안합니다. 기존 방법들이 현재 또는 과거의 행동만을 설명하는 것에 한계를 두었던 반면, 이 방법은 차량의 미래 계획인 궤적 정보를 입력으로 활용하여 보다 정확한 캡션을 생성합니다.

- **Technical Details**: 제안된 방법은 이미지 및 궤적 계획 정보를 결합하여 크로스 어텐션을 사용해 시각적 정보와 궤적 계획 정보를 공간적으로 융합합니다. 블립-2(BLIP-2) 기반의 비전-언어 모델을 사용하여, 융합된 특징을 통해 자율주행 차량의 미래 행동을 설명하는 캡션을 생성합니다.

- **Performance Highlights**: 제안된 모델은 자율주행 차량의 미래 행동을 설명하고 정당화하는 캡션을 생성하는 데 있어 기존 연구들보다 현저한 개선을 보여주며, 새로운 비디오 및 캡션이 포함된 데이터셋을 구성하였습니다.



### Seeing Clearly by Layer Two: Enhancing Attention Heads to Alleviate Hallucination in LVLMs (https://arxiv.org/abs/2411.09968)
- **What's New**: 이번 논문은 멀티모달 대형 언어 모델(MLLM)에서 이미지 토큰과 환각(hallucination) 현상 간의 관계를 분석하고, 낮은 계층의 주의(head) 밀도를 높여 환각 문제를 완화하는 새로운 방법인 EAH(Enhancing Attention Heads)를 제안합니다.

- **Technical Details**: 연구는 주의 매트릭스에서 시각적 토큰의 밀집된 주의 싱크(vision sink)가 환각 발생 여부와 연결되어 있음을 발견했습니다. EAH는 낮은 계층에서 시각적 싱크가 가장 높은 주의를 찾고, 이를 사용하여 다른 주의 헤드를 강화하는 훈련이 필요 없는 방법입니다. 논문에서는 LLaVA1.5, Minigpt4 등 여러 모델을 사용한 실험을 통해 EAH의 효과를 입증합니다.

- **Performance Highlights**: EAH를 적용한 결과, 다양한 멀티모달 대형 언어 모델에서 환각 발생률이 감소하였으며, 시각적 싱크 헤드의 밀도가 높을수록 환각 문제의 완화와 음의 관계가 있다는 점을 강조했습니다.



### Instruction-Guided Editing Controls for Images and Multimedia: A Survey in LLM era (https://arxiv.org/abs/2411.09955)
- **What's New**: 이번 논문은 LLM(large language models) 및 멀티모달(Multimodal) 모델이 시각적 수정 작업을 간소화하는 방법에 대한 개요를 제공합니다. 사용자는 복잡한 기술적 지식 없이 자연어로 지시를 통해 이미지 및 비디오 편집을 수행할 수 있습니다.

- **Technical Details**: 논문은 생성적 적대 신경망(generative adversarial networks)과 확산 모델(diffusion models)에서부터 시작하여, MLLM(multimodal large language models)과 같은 최첨단 기술을 통합한 방법을 다룹니다. 이러한 모델들은 사용자가 자연어로 간단한 명령을 주면 시각적 콘텐츠에 대한 정밀한 수정을 할 수 있도록 지원합니다.

- **Performance Highlights**: LLM 및 MLLM의 발전은 패션, 3D 장면 조작 및 비디오 합성 등 다양한 분야에서 더 많은 접근성을 제공하여 사용자 경험을 향상시켰습니다. 이 논문은 LLM 기반 편집이 산업 전반에 걸쳐 강력한 도구로 자리 잡을 수 있음을 강조하고, 사용자 친화적인 편집 툴의 필요성을 제시합니다.



### GGAvatar: Reconstructing Garment-Separated 3D Gaussian Splatting Avatars from Monocular Video (https://arxiv.org/abs/2411.09952)
Comments:
          MMAsia'24 Accepted

- **What's New**: 이번 논문에서는 GGAvatar (Garment-separated 3D Gaussian Splatting Avatar)라는 새로운 아바타 모델이 소개되며, 이는 단일 모노큘러 비디오를 이용하여 의류와 신체를 효과적으로 분리하여 사실적인 인간 모델을 생성합니다.

- **Technical Details**: GGAvatar는 매개변수화된 템플릿을 사용하고 독특한 단계적 훈련을 통해 의류와 신체 간의 분리를 달성합니다. 또한 3D Gaussian Splatting(3DGS) 기법을 활용하여 고품질 렌더링을 수행하며, 이러한 과정에서 포인트 세트의 교차를 방지하는 강력한 훈련 모듈을 적용합니다.

- **Performance Highlights**: GGAvatar는 People Snapshot Dataset과 ZJU Mocap Dataset에서 기존의 NeRF 기반 모델에 비해 수백 배 빠른 훈련 속도와 뛰어난 복원 품질을 보여주며, 의류 편집 및 색상 변경과 같은 다양한 응용 분야에서 활용할 수 있는 가능성을 가집니다.



### JRadiEvo: A Japanese Radiology Report Generation Model Enhanced by Evolutionary Optimization of Model Merging (https://arxiv.org/abs/2411.09933)
Comments:
          Accepted by NeurIPS'24 Workshop on AIM-FM: Advancements In Medical Foundation Models: Explainability, Robustness, Security, and Beyond

- **What's New**: 이번 논문은 비의료 비전-언어 모델을 비영어 의료 텍스트 생성에 확장하기 위한 첫 번째 시도로, 진화 최적화(evolutionary optimization)를 통한 모델 병합(model merging) 기법을 사용하여 일본어(X-ray) 방사선 보고서 생성 모델(JRadiEvo)을 제안합니다.

- **Technical Details**: JRadiEvo는 비의료 비전-언어 모델, 의료 텍스트-투-텍스트 모델 및 일본어 텍스트-투-텍스트 모델을 진화 알고리즘을 통해 병합하여 개발되었습니다. 이 모델은 50개의 번역 샘플만으로 정확한 일본어 보고서를 생성할 수 있으며, 80억 개의 파라미터를 가진 경량 모델로, 병원 내에서 정상적으로 배포 가능하여 개인 정보 보호 문제를 해결합니다.

- **Performance Highlights**: JRadiEvo는 제한된 데이터(50개 사례)만을 사용하여 최근 연구의 최신 모델들보다 우수한 성능을 보여주었습니다. 이를 통해 모델의 피드백을 제거하여 학습 과정에서 비효율을 줄이고, 개인 정보를 보호해야 하는 의료 환경에서도 실용적인 응용이 가능하다는 점을 강조합니다.



### A Polarization Image Dehazing Method Based on the Principle of Physical Diffusion (https://arxiv.org/abs/2411.09924)
- **What's New**: 이 논문은 복잡한 환경인 안개 낀 상황에서 polarized imaging 기술을 활용한 새로운 semi-physical polarization dehazing 방법을 제안합니다. 이 방법은 외부 광원에 의존하지 않고, 안개의 확산 과정을 시뮬레이션하여 이미지의 흐림을 해결합니다.

- **Technical Details**: 제안된 방법은 전이 과정을 시뮬레이션하여 이미지의 흐림을 계산하는 diffusion kernel을 설계합니다. 이를 위해 spatiotemporal Fourier transforms 및 deconvolution 작업을 수행하여 안개 입자의 상태와 객체의 빛 반전 분포를 복구합니다. 이 방법은 복잡한 전송 매질과 장거리 이미징 조건에서도 polarized light의 특성을 최대한 활용합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 여러 전통적인 dehazing 알고리즘보다 효과적이고, 객체의 윤곽 및 세부 사항을 효과적으로 복원하여 객체 탐지 및 인식과 같은 후속 작업에 적합합니다.



### Motion-Grounded Video Reasoning: Understanding and Perceiving Motion at Pixel Lev (https://arxiv.org/abs/2411.09921)
- **What's New**: 이 논문에서는 Motion-Grounded Video Reasoning이라는 새로운 모션 이해 과제를 제시합니다. 이 과제는 입력 질문에 따라 시각적 답변(비디오 세그멘테이션 마스크)을 생성해야 하며, 이를 위해 암묵적인 시공간적(spatiotemporal) 추론과 그라운딩(grounding)이 필요합니다. 신규 작업인 GROUNDMORE 데이터셋을 수집하였으며, 이는 1,715개의 비디오 클립과 249,000개의 객체 마스크로 구성되어 있습니다.

- **Technical Details**: GROUNDMORE 데이터셋은 4가지 질문 유형(원인, 순차, 반사실, 설명)을 포함하도록 디자인되었으며, MORA(모션-그라운디드 비디오 추론 보조 모델)를 도입하여 Multimodal LLM의 멀티모달 추론 능력과 SAM의 픽셀 수준 인식 능력을 통합하였습니다. MORA는 GROUNDMORE에서 기존 비주얼 그라운딩 모델보다 평균 21.5% 더 뛰어난 성능을 보였습니다.

- **Performance Highlights**: MORA 모델은 GROUNDMORE 데이터셋에서 시공간적 그라운딩 및 추론 과제를 수행하며, 복잡한 모션 관련 비디오 추론, 시간 인식, 픽셀 수준 이해의 챌린지를 해결합니다. 이 새로운 작업은 비디오 추론 세그멘테이션을 통한 강력하고 일반화 가능한 모션 이해의 발전을 위한 길을 열 것으로 기대합니다.



### DiffFNO: Diffusion Fourier Neural Operator (https://arxiv.org/abs/2411.09911)
- **What's New**: 이 논문에서는 임의 스케일의 초해상도(super-resolution, SR)를 위한 새로운 확산 프레임워크인 DiffFNO를 소개합니다. DiffFNO는 Weighted Fourier Neural Operator (WFNO)를 통해 강화되며, WFNO의 모드 재균형(mode Re-balancing)은 주파수 구성 요소를 효과적으로 캡쳐하여 고해상도 이미지의 세밀한 세부 사항을 향상시킵니다.

- **Technical Details**: DiffFNO는 다음과 같은 주요 구성 요소로 이루어져 있습니다: (1) WFNO는 주파수 재균형(modes Rebalancing)을 통해 가장 중요한 주파수 구성 요소를 강조하여 고주파 이미지 세부 사항 재구성을 개선합니다. (2) Gated Fusion Mechanism (GFM)은 WFNO와 Attention-based Neural Operator (AttnNO)의 스펙트럼 및 공간적 특성을 동적으로 조정하여 글로벌 구조와 로컬 세부 정보를 포착합니다. (3) Adaptive Time-Step (ATS) ODE solver는 데이터 특성에 따라 적응적으로 통합 단계 크기를 조정하여 품질을 희생하지 않으면서 효율성을 개선합니다.

- **Performance Highlights**: DiffFNO는 여러 초해상도 벤치마크에서 최첨단(State-of-the-art, SOTA) 결과를 달성하며, PSNR에서 기존 방법을 2에서 4 dB 향상시킵니다. 또한, 낮은 추론 시간을 자랑하여 연산 효율성 또한 뛰어난 것으로 나타났습니다.



### Free Lunch in Pathology Foundation Model: Task-specific Model Adaptation with Concept-Guided Feature Enhancemen (https://arxiv.org/abs/2411.09894)
- **What's New**: 이번 논문에서는 병리학 기초 모델의 성능을 특정 다운스트림 작업에 맞춰 향상시키기 위한 'Concept Anchor-guided Task-specific Feature Enhancement (CATE)'라는 새로운 패러다임을 제안합니다.

- **Technical Details**: CATE는 두 개의 모듈, 즉 Concept-guided Information Bottleneck (CIB)와 Concept-Feature Interference (CFI)를 통해 작업별 특징을 동적으로 조정합니다. CIB 모듈은 이미지 특징과 개념 앵커 간의 상호 정보를 극대화하여 작업 관련 특성을 강화하고 불필요한 정보를 억제합니다. CFI 모듈은 캘리브레이션 된 특징과 개념 앵커 간의 유사성을 이용하여 차별화된 작업 특정 특징을 생성합니다.

- **Performance Highlights**: CATE는 공공 WSI 데이터셋을 활용한 광범위한 실험을 통해 MIL 모델의 성능 및 일반화 능력을 상당히 향상시킵니다. Heatmap 및 UMAP 시각화 결과는 CATE의 효과성과 해석 가능성을 보여줍니다.



### Memory Proxy Maps for Visual Navigation (https://arxiv.org/abs/2411.09893)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2402.12498

- **What's New**: 본 논문에서는 3단계 에이전트를 구축하기 위해 피우달 러닝(Feudal Learning) 기반의 새로운 시각 내비게이션 접근 방식을 제안합니다. 여기서 메모리 프록시 맵(Memory Proxy Map, MPM)을 활용하여 에이전트가 본 환경을 근사화하는 방법을 탐구합니다.

- **Technical Details**: MPM은 자가 감독 학습(Self-supervised Learning)에 의해 학습되며, 중간 목표인 웨이포인트 네트워크(Waypoint Network, WayNet)는 인간 탐색 정책을 모방하여 지역 내비게이션을 수행합니다. 저수준 작업자 에이전트는 이 웨이포인트로 이동하는 동작을 선택하기 위해 분류기를 학습합니다.

- **Performance Highlights**: 이 연구는 기존의 모든 메트릭 맵이나 RL, 그래프, 오도메트리 없이도 이미지 목표 내비게이션 작업에서 SOTA(State-Of-The-Art) 성능을 달성함을 입증하였습니다.



### Content-Aware Preserving Image Generation (https://arxiv.org/abs/2411.09871)
Comments:
          35 pages, 12 figures, 1 table, journal

- **What's New**: 본 논문에서는 사용자 요구를 충족시키기 위해 출력 이미지의 내용을 명시적으로 제어할 수 있는 새로운 이미지 생성 프레임워크인 Content-Aware Preserving Image Generation Framework (CAP-GAN)을 제안하고 있습니다. 이 프레임워크는 생성 이미지의 내용 유지 및 다양한 스타일 변화를 보장합니다.

- **Technical Details**: CAP-GAN은 content fusion 모듈과 frequency encoding 모듈을 통합하여 원하는 내용이 포함된 이미지 생성을 가능하게 합니다. frequency encoding 모듈은 참조 이미지의 주파수 구성 요소만 선택하여 특징을 캡처하고, content fusion 모듈은 원하는 내용 특징을 캡슐화한 내용 안내 벡터를 생성합니다.

- **Performance Highlights**: 제안된 프레임워크는 Flickr-Faces-High Quality, Animal Faces High Quality, Large-scale Scene Understanding 데이터셋 등에서 폭넓은 실험을 통해 내용 특성을 보존하는 데 효과적임을 입증하였습니다.



### Face De-identification: State-of-the-art Methods and Comparative Studies (https://arxiv.org/abs/2411.09863)
- **What's New**: 최근의 얼굴 인식 기술 및 이미지 획득 기술의 발전이 개인 정보 보호에 대한 심각한 우려를 야기하고 있습니다. 이 논문은 얼굴 비노출화 (face de-identification) 방법의 최신 동향을 세 가지 수준(픽셀, 표현, 의미 수준)으로 분류하여 포괄적으로 검토하고, 개인 정보를 보호하면서 이미지 유용성을 보존하는 방법을 제안합니다.

- **Technical Details**: 얼굴 비노출화 기법은 개인의 식별 가능한 얼굴 특징을 숨기거나 변경하여 인식을 방지하는 과정을 포함합니다. 본 논문에서는 GANs (Generative Adversarial Networks)와 diffusion 모델을 포함하는 딥 러닝 기반 접근 방식들이 개인 정보 보호와 유용성의 균형을 이루는 데 중요한 발전을 이룩했다고 설명합니다. 효과적인 개인 정보 보호와 이미지 유용성 유지 측면에서 메인 알고리즘의 정성적 및 정량적 비교를 제공합니다.

- **Performance Highlights**: 최근 방법들이 강력한 개인정보 보호 성능을 발휘하는 반면, 시각적 충실도와 계산 복잡성에서 트레이드오프가 여전히 존재합니다. 본 설문조사는 현재의 얼굴 비노출화 기술의 전반적인 경관을 요약하고, 이 분야의 주요 도전 과제와 미래 연구 방향을 제시합니다.



### Masked Image Contrastive Learning for Efficient Visual Conceptual Pre-training (https://arxiv.org/abs/2411.09858)
Comments:
          10 pages

- **What's New**: 이번 논문은 이미지를 효율적으로 시각적 개념으로 표현하기 위한 새로운 사전 훈련 패러다임인 masked image contrastive learning (MiCL)을 제안합니다. MiCL은 이미지 내 패치를 무작위로 마스킹하여 이미지의 다양한 뷰를 생성하고, 이를 미니 배치의 이미지 간에 대조합니다.

- **Technical Details**: MiCL의 핵심 아이디어는 두 가지 설계로 구성됩니다: 첫째, 마스킹된 토큰들은 이미지 내 개념적 중복성을 크게 줄일 수 있으며, 개별 인스턴스 레벨이 아닌 의미론적 개념 레벨의 뚜렷한 다름을 생성합니다. 둘째, 대조 학습은 높은 주파수의 간섭과 이미지 복원에 관련된 추가 비용을 피하면서 사전 훈련 중 고수준의 의미 론적 개념 특징을 추출하는 데 능숙합니다. 실험적으로, MiCL은 4개의 A100 GPU만을 사용하여 ViT-L/16이 133시간 내에 사전 훈련을 완료하며, 하위 조정 작업에서 85.8%의 정확도를 달성함을 입증했습니다.

- **Performance Highlights**: MiCL은 나쁜 설계에 의존하지 않고도 고수준의 의미론적 개념 표현을 효율적으로 학습할 수 있으므로, 복잡한 추가 모듈이나 데이터 증강을 필요로 하지 않습니다. 이 모델은 133시간 만에 85.8%의 정확도를 달성하며, 특히 기존의 방법들보다 더 뛰어난 성능을 제공합니다.



### Enhancing Diffusion Posterior Sampling for Inverse Problems by Integrating Crafted Measurements (https://arxiv.org/abs/2411.09850)
- **What's New**: 본 논문에서는 새로운 diffusion posterior sampling 방법인 DPS-CM을 제안합니다. 이 방법은 깨끗한 이미지 대신 잡음이 있는 측정치에서 log posterior gradient를 형성하여 역과정을 개선하고, Crafted Measurement를 통합하여 posterior estimate를 생성합니다.

- **Technical Details**: DPS-CM은 문제가 발생할 수 있는 누적 posterior estimate 오류로 인한 diffusion prior와의 불일치를 완화하는 것을 목표로 합니다. 이 방법은 Gaussian deblurring, super-resolution, inpainting 및 Poisson noise와 같은 다양한 noisy inverse 문제를 해결하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기존 접근 방식에 비해 일반적이고 noisy inverse 문제 해결능력을 획기적으로 향상시킴을 입증했습니다.



### Architect: Generating Vivid and Interactive 3D Scenes with Hierarchical 2D Inpainting (https://arxiv.org/abs/2411.09823)
- **What's New**: 이번 연구는 Architect라는 새로운 생성 프레임워크를 제안하여, diffusion 기반의 2D 이미지 인페인팅을 통해 복잡하고 사실적인 3D 환경을 생성하는 방법을 소개합니다. 이는 기존 LLM 기반의 방법들이 가진 제약을 극복하고, 더 생생한 상호작용 환경을 만드는 데 기여합니다.

- **Technical Details**: Architect는 2D 이미지 생성을 통해 복잡한 객체 배치와 맥락에 맞는 객체 위치를 통합하여 3D 환경으로 변환합니다. 특히, 미리 훈련된 depth estimation 모델을 사용하여 2D 이미지를 3D 공간으로 변환하며, 텍스트 설명, 바닥 계획, 사전 배치 환경 등 다양한 입력으로부터 생성할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: 실험 결과, Architect는 기존의 장면 생성 접근 방식보다 더 복잡하고 사실적인 3D 상호작용 장면을 생성하는 데 성공하였으며, 다양한 환경에서도 적절히 활용할 수 있음을 보여주었습니다.



### A Self-Supervised Model for Multi-modal Stroke Risk Prediction (https://arxiv.org/abs/2411.09822)
Comments:
          Accepted as oral paper at AIM-FM workshop, Neurips 2024

- **What's New**: 이 연구는 3D 뇌 영상, 임상 데이터 및 이미지 기반 특징을 통합하여 발병 전 뇌졸중 위험 예측을 개선하는 자기 지도(self-supervised) 다중 모달 프레임워크를 제안합니다. 이는 다양한 임상 데이터 모달리티를 결합하여 예측 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: 모델은 UK Biobank에서 수집된 데이터로 훈련되며, 구조적 뇌 MRI와 임상 데이터를 포함합니다. 대조적 학습(constrastive learning) 프레임워크를 기반으로 하여 이미지와 표 데이터를 결합하는 매칭 모듈을 사용하여 모달리티 간의 정보를 정렬합니다. 또한, 뇌 영상 데이터와 임상 특성 사이의 상호작용을 촉진하기 위해 CLIP 손실(clips loss)과 이미지-표 매칭 손실(image-tabular matching loss)을 통합합니다.

- **Performance Highlights**: 제안된 모델은 ROA-AUC에서 자기 지도 표준 방법보다 2.6% 더 우수하며, 균형 정확도(balanced accuracy)에서는 7.6% 향상된 성능을 보여줍니다. 이 모델의 응답 가능한 도구를 통해 이미지와 표 데이터를 더 잘 통합하고 해석 가능한 기능을 제공하여 임시로 뇌졸중 관련 뇌 병리와 일치하는 시각적 활성화 맵을 생성했습니다.



### Video Denoising in Fluorescence Guided Surgery (https://arxiv.org/abs/2411.09798)
- **What's New**: 본 연구에서는 Fluorescence Guided Surgery (FGS)의 비디오 잡음을 제거하기 위한 새로운 딥 러닝 기반 알고리즘을 제안하며, Laser Leakage Light (LLL)를 시뮬레이션하는 파이프라인을 개발하였습니다.

- **Technical Details**: FGS 시스템에서 발생하는 LLL은 영상 신호와 비슷한 밝기로 나타나며 이를 제거하는 것이 큰 도전입니다. 본 연구에서는 참조 비디오 (RV)를 통해 LLL을 정확하게 시뮬레이션하고, 이를 이용한 잡음 제거 알고리즘을 개발하였습니다. 새로운 비디오 잡음 제거 모델로 BL-RNN을 제안합니다.

- **Performance Highlights**: NafNet이라는 이미지 잡음 제거 모델이 기존의 비디오 잡음 제거 모델보다 우수한 성능을 보였으며, 훈련 효율성에서도 뛰어난 결과를 나타냈습니다. 이를 바탕으로 OL-2024 데이터셋을 구축하여 다양한 실험에 활용하였습니다.



### NACNet: A Histology Context-aware Transformer Graph Convolution Network for Predicting Treatment Response to Neoadjuvant Chemotherapy in Triple Negative Breast Cancer (https://arxiv.org/abs/2411.09766)
Comments:
          This paper is accepted by Computerized Medical Imaging and Graphics (Nov 07 2024)

- **What's New**: 이 연구에서는 Triple Negative Breast Cancer (TNBC) 환자에 대한 neoadjuvant chemotherapy (NAC) 반응 예측을 위한 histology context-aware transformer graph convolution network (NACNet)를 개발했습니다. 이는 기존의 기법과 달리 tumor microenvironment (TME)의 공간적 상호작용을 통합하여 더 정확한 예측을 가능하게 합니다.

- **Technical Details**: NACNet은 디지털 whole slide images (WSIs)에서 개별 이미지 타일의 histopathological labels를 식별하고, TME의 공간적 그래프를 구축하여 각 노드에 tissue texture 및 사회망 분석에서 도출된 특징을 나타냅니다. 이때 transformer graph convolution network 모델에 graph isomorphism network (GIN) 레이어를 추가하여 NAC 반응을 예측합니다.

- **Performance Highlights**: NACNet은 105명의 TNBC 환자에서 90.0% 정확도, 96.0% 민감도, 88.0% 특이도 및 AUC 0.82를 달성하여, 기존의 머신러닝 및 딥러닝 모델보다 우수한 성과를 보였습니다. 이는 TNBC 환자의 NAC 반응을 stratifying하는 데 큰 잠재력을 가지고 있어, 지나치게 치료하는 것을 방지하고 환자 생활의 질을 개선하는 데 기여할 것으로 기대됩니다.



### Partial Multi-View Clustering via Meta-Learning and Contrastive Feature Alignmen (https://arxiv.org/abs/2411.09758)
- **What's New**: 본 논문에서는 데이터 분석에서 발생하는 부분적인 다중 뷰 클러스터링의 문제를 해결하기 위해 대조 학습(contrastive learning)에 기반한 새로운 이중 최적화 프레임워크를 제안합니다. 이를 통해 불완전한 다중 뷰 데이터에서 잠재 특징의 일관성을 극대화하고 클러스터링 성능을 향상시킵니다.

- **Technical Details**: 제안된 PVC-MC(PVC via Meta-Learning and Contrastive Feature Alignment) 방법은 대조 학습을 활용하여 불완전한 다중 뷰 데이터의 잠재 표현을 정렬합니다. KNN과 Vision Transformer를 기반으로 한 독립적인 자가 표현 레이어를 추가하여 각 뷰의 고유한 특징을 활용하며, 메타 학습(meta-learning)과 자가 감독 학습(self-supervised learning)을 통해 뷰 가중치를 동적으로 조정합니다. 이 과정을 통해 결측된 정보를 채우고 클러스터링 정확성을 극대화합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 BDGP 및 HW 데이터셋에서 기존 최첨단 클러스터링 모델에 비해 뛰어난 성능을 보여 줍니다. 특히 복잡하고 불완전한 다중 뷰 데이터 처리에 있어 우수한 결과를 기록하였습니다.



### Towards Neural Foundation Models for Vision: Aligning EEG, MEG, and fMRI Representations for Decoding, Encoding, and Modality Conversion (https://arxiv.org/abs/2411.09723)
- **What's New**: 이 논문은 대조 학습(contrastive learning)을 활용하여 다중 모달(multi-modal) 뇌 활동 표현의 신경 데이터(neural data)와 시각 자극(visual stimuli)을 정렬하는 기초 모델(foundational model)을 만드는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 우리는 뇌파(EEG), 자기 뇌파(MEG), 기능적 자기 공명 영상(fMRI) 데이터를 사용하여 프레임워크의 기능을 입증하였으며, 세 가지 주요 실험을 통해 시각 정보를 신경 데이터에서 디코딩(decoding)하고, 이미지를 신경 표현(neural representations)으로 인코딩(encoding)하며, 신경 모달리티(modality) 간 변환을 수행합니다.

- **Performance Highlights**: 결과는 다양한 뇌 이미징 기술(techniques)에 걸쳐 의미론적 정보(semantic information)를 정확하게 포착할 수 있는 모델의 능력을 강조하며, 디코딩, 인코딩, 모달리티 변환 작업에서의 잠재력을 입증합니다.



### Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization (https://arxiv.org/abs/2411.10442)
- **What's New**: 본 연구에서는 멀티모달 대형 언어 모델(MLLMs)의 Chain-of-Thought (CoT) 성능을 향상시키기 위한 새로운 선호 최적화(Preference Optimization, PO) 프로세스를 소개합니다. 특히, 고품질 대규모 멀티모달 추론 선호 데이터셋인 MMPR을 생성하기 위한 자동화 선호 데이터 구축 파이프라인을 설계하고, MLLMs와 PO를 통합하여 Mixed Preference Optimization (MPO) 방법을 개발했습니다.

- **Technical Details**: 현재 MLLMs는 주로 프리트레이닝(pre-training) 및 감독된 미세 조정(supervised fine-tuning, SFT) 프로세스를 따릅니다. 하지만 SFT 손실로 인해 분포 전이(distribution shift) 문제가 발생하며, 이는 CoT 성능을 저하시킵니다. 본 연구의 제안은 Dropout Next Token Prediction (DropoutNTP) 파이프라인과 같은 간단하고 효과적인 MPO 방법을 도입하여 멀티모달 CoT 성능을 향상시키는 것입니다.

- **Performance Highlights**: 제안된 모델 InternVL2-8B-MPO는 MathVista에서 67.0의 정확도를 달성하여, 기존의 InternVL2-8B보다 8.7점 향상되었으며, 10배 더 큰 모델인 InternVL2-76B와 동등한 성능을 보였습니다. 이는 PO 방법론이 MLLMs의 추론 능력을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### Mitigating Hallucination in Multimodal Large Language Model via Hallucination-targeted Direct Preference Optimization (https://arxiv.org/abs/2411.10436)
- **What's New**: 본 연구에서는 Hallucination-targeted Direct Preference Optimization (HDPO)을 제안하여 다중 모달 큰 언어 모델(MLLMs)의 환각(hallucination)을 줄이는 새로운 접근 방식을 도입합니다. 이전 방법들과 달리, HDPO는 MLLM의 다양한 환각 원인과 형태를 다룹니다.

- **Technical Details**: HDPO는 세 가지 유형의 환각 원인에 대한 선택적 preference pair 데이터를 개발하여 구체적인 환각 상황을 해결합니다: 1) 충분하지 않은 시각적 능력, 2) 긴 맥락 생성, 3) 다중 모달 충돌. 각 유형에 대해 고유한 데이터를 생성하며, 이를 통해 MLLM의 능력을 향상시킵니다.

- **Performance Highlights**: 실험 결과 HDPO는 다양한 환각 평가 데이터셋에서 우수한 성능을 보여주었으며, 대부분의 최신 방법(SOTA)을 초월하며 MLLM 환각 문제를 효과적으로 완화했습니다. 특히, 다양한 M-hallu 작업에서 일관된 개선을 달성했습니다.



### On the Foundation Model for Cardiac MRI Reconstruction (https://arxiv.org/abs/2411.10403)
Comments:
          For MICCAI CMRxRecon Challenge 2024 team CardiAxs

- **What's New**: 이 연구에서는 Cardiac Magnetic Resonance (CMR) 이미징을 위한 새로운 기초 모델을 제안합니다. 이 모델은 Adaptive Unrolling, Channel-Shifting, 및 Pattern and Contrast-Prompt-UNet (PCP-UNet)을 활용하여 ML 기반 이미지 재구성의 문제를 해결합니다.

- **Technical Details**: 제안된 모델은 다양한 가속 비율에 따라 서로 다른 수의 unrolled iterations를 거치며, Channel-Shifting을 통해 재구성된 데이터 품질을 개선합니다. PCP-UNet은 이미지 대비 및 샘플링 패턴 프롬프트를 갖추고 있습니다. 이 모델은 다양한 CMR 프로토콜에 대해 이미지 품질을 유의미하게 향상시켰습니다.

- **Performance Highlights**: in vivo CMR 실험 결과, 제안한 방법은 기존의 ML 기반 방법보다 뛰어난 성능을 보였으며, 다양한 이미지 대비, 샘플링 패턴 및 가속 비율을 처리할 수 있는 능력을 입증했습니다.



### The Dawn of GUI Agent: A Preliminary Case Study with Claude 3.5 Computer Us (https://arxiv.org/abs/2411.10323)
Comments:
          40 pages, 21 figures, preprint

- **What's New**: Claude 3.5 Computer Use는 공공 베타로 제공되는 GUI 에이전트를 갖춘 최초의 최전선 AI 모델로, 실세계의 복잡한 환경에서 컴퓨터 활용 능력이 아직 잘 알려지지 않았다.

- **Technical Details**: Claude 3.5는 user instruction을 기반으로 API 호출을 통해 end-to-end 솔루션을 제공하며, GUI 상태를 시각적으로 관찰하여 행동을 생성한다. GUI 상호작용을 통해 사용자의 요구를 충족 시키기 위해 'vision-only' 접근 방식을 채택하고 있으며, 이 과정에서 컴퓨터 툴, 텍스트 에디터 툴, Bash 툴 등의 도구가 사용된다.

- **Performance Highlights**: 이 사례 연구는 Claude 3.5의 성능을 여러 사용자 그룹의 요구를 반영하는 다양한 데스크톱 작업 자동화 과제를 통해 평가하였으며, 이에 대한 정량적 및 정성적 평가가 이루어졌다. 이 연구는 Claude의 GUI 자동화 모델의 능력과 한계를 밝혀내고 향후 개선 사항에 대한 논의를 이끈다.



### The Surprising Ineffectiveness of Pre-Trained Visual Representations for Model-Based Reinforcement Learning (https://arxiv.org/abs/2411.10175)
Comments:
          Published at the 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Project page: this https URL

- **What's New**: 본 논문에서는 모델 기반 강화 학습(Model-Based Reinforcement Learning, MBRL) 환경에서 다양한 사전 학습된 시각 표현(Pre-trained Visual Representations, PVRs)의 효과를 벤치마킹합니다. 이전의 연구는 PVR이 샘플 효율성과 일반화 능력을 향상시킨다고 보고했으나, MBRL에 대한 PVR의 잠재력은 거의 탐색되지 않았습니다. 이 연구를 통해 PVR이 MBRL의 성능에 미치는 영향을 분석합니다.

- **Technical Details**: 이 연구에서는 MBRL에서의 PVR의 데이터 효율성과 일반화 능력을 평가하며, PVR의 다양한 속성이 MBRL 에이전트의 성능에 미치는 영향을 조사합니다. MBRL에서는 특징을 선별하여 환경의 동력학 모델을 학습하고, CNN(Covolutional Neural Networks)을 사용하여 시각적 상태 표현을 활용합니다. 이 연구의 결과에 따르면, 현재의 PVR은 스크래치에서 학습한 표현보다 샘플 효율성이 더 뛰어나지 않으며, OOD(out-of-distribution) 설정에 대한 일반화 능력도 떨어지는 것으로 나타났습니다.

- **Performance Highlights**: 본 논문에서 수행한 실험 결과, 스크래치에서 학습한 표현을 사용하는 모델이 PVR을 사용하는 모델보다 더 우수한 성능을 보이는 경우가 많았습니다. 데이터 다양성과 네트워크 아키텍처가 OOD 일반화 성능에 가장 중요한 요소로 나타났습니다.



### Federated Domain Generalization via Prompt Learning and Aggregation (https://arxiv.org/abs/2411.10063)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문에서는 Federated Domain Generalization(FedDG) 설정에서 개인정보 보호 제약을 유지하면서도 데이터 이질성을 다룰 수 있는 새로운 방안으로 Prompt Learning을 도입합니다.

- **Technical Details**: PLAN(프롬프트 학습 및 집계 방법)은 두 단계로 구성된 훈련 프레임워크입니다. 먼저, 각 클라이언트는 자신의 데이터로부터 텍스트 및 시각적 프롬프트 학습을 수행하고, 이후 도메인 특화된 로컬 프롬프트가 클라이언트 간에 교환되고 경량 기반의 주의(Attention) 집계기를 통해 글로벌 프롬프트로 집계됩니다.

- **Performance Highlights**: PLAN 방법은 4개의 벤치마크 데이터셋에서 기존의 FL, DG, FedDG 방법과 프롬프트 학습 기반 방법들보다 현저히 우수한 성능을 보였습니다. 또한, 계산 효율성 및 통신 효율성 면에서도 뚜렷한 장점을 나타냈습니다.



### EchoMimicV2: Towards Striking, Simplified, and Semi-Body Human Animation (https://arxiv.org/abs/2411.10061)
- **What's New**: EchoMimicV2는 오디오와 포즈 조건을 정교하게 조정하는 Audio-Pose Dynamic Harmonization 전략을 사용하여 절반 몸체 애니메이션의 품질을 향상시키면서도 조건의 중복성을 감소시킵니다.

- **Technical Details**: EchoMimicV2는 오디오 조건을 입술에서부터 전체 몸으로, 포즈 조건을 전체 몸에서 손으로 제한하며, 이를 통해 절반 몸체의 애니메이션을 가능하게 합니다. 또한, Phase-specific Loss(PhD Loss)를 통해 애니메이션의 동작, 세부 사항 및 저수준 품질을 특정 단계별로 최적화합니다.

- **Performance Highlights**: EchoMimicV2는 기존의 방법들보다 정량적 및 정성적 평가에서 우수한 성능을 보여주며, 해부학적 상세 및 감정 표현에서 뛰어난 결과를 도출합니다.



### EyeDiff: text-to-image diffusion model improves rare eye disease diagnosis (https://arxiv.org/abs/2411.10004)
Comments:
          28 pages, 2 figures

- **What's New**: EyeDiff는 자연어 프롬프트를 기반으로 다중 모달(multi-modal) 안과 이미지를 생성할 수 있는 텍스트-투-이미지 모델로, 희귀 질환 진단에 적합합니다.

- **Technical Details**: EyeDiff는 8개의 대규모 데이터셋을 기반으로 훈련되었으며, 고급 잠재 확산(latent diffusion) 모델을 사용하여 14개의 안과 이미지 모달리티와 80종 이상의 안구 질환을 다룹니다. 이 모델은 10개의 다국적 외부 데이터셋에 적응되어 있습니다.

- **Performance Highlights**: 생성된 이미지는 중요한 병변(lesional) 특성을 정확하게 포착하여, 텍스트 프롬프트와 높은 일치를 이루며, 소수 클래스 및 희귀 안구 질환 탐지의 정확성을 크게 향상시킵니다. 이는 기존의 오버샘플링(oversampling) 방법보다 데이터 불균형 문제를 효과적으로 해결합니다.



### Adaptive Non-Uniform Timestep Sampling for Diffusion Model Training (https://arxiv.org/abs/2411.09998)
- **What's New**: 이번 연구에서는 diffusion 모델의 효율성을 높이기 위한 새로운 비균일 타임스텝 샘플링 방법을 제안합니다. 기존의 균일한 타임스텝 샘플링 방식은 복잡한 데이터 분포로 인해 훈련 과정에서 수렴하는 데 많은 연산 비용이 필요하다는 문제점을 가지고 있습니다. 우리의 접근 방식은 중요한 타임스텝을 우선시하여 더 빠른 수렴을 이끌어냅니다.

- **Technical Details**: 제안한 방법은 각 타임스텝에서의 그래디언트 업데이트의 영향을 추적하며, 최적의 목표를 효과적으로 최소화하는 타임스텝을 적응적으로 선택합니다. 실험 결과에 따르면, 이 방법은 훈련 과정을 가속화하며, 다양한 데이터셋과 diffusion 아키텍처에서 우수한 성능을 보여줍니다. 또한 기존의 타임스텝 샘플링 및 가중치 휴리스틱보다 더 높은 강건성을 제공합니다.

- **Performance Highlights**: 실험 결과, 새로운 비균일 타임스텝 샘플링 방법은 기존의 방법들보다 훈련 과정을 더욱 빠르고 효과적으로 개선했습니다. Stable-Diffusion-2.0 및 Open-Sora와 같은 모델에서 기존의 장비 사용 시간 대비 30% 이상의 연산 시간 단축이 가능함을 보여주었습니다.



### mmSpyVR: Exploiting mmWave Radar for Penetrating Obstacles to Uncover Privacy Vulnerability of Virtual Reality (https://arxiv.org/abs/2411.09914)
- **What's New**: 이 논문은 VR 시스템에서 사용자 프라이버시를 침해할 수 있는 새로운 취약점을 발견하고, mmWave 신호를 이용한 mmSpyVR이라는 공격 방법을 제안합니다. 이는 물리적 접촉이나 VR 장치와의 연결 없이도 프라이버시를 침해할 수 있는 점에서 주목할 만 합니다.

- **Technical Details**: mmSpyVR 시스템은 두 가지 주요 모듈을 포함합니다: (i) VR 특징 추출을 위한 transfer learning 기반 모델, (ii) 추출된 특징에서 VR 프라이버시 정보를 감시하는 attention 기반 모듈. 실험에서는 3개의 서로 다른 제조사의 VR 장치를 사용해 22명의 참가자로부터 12TB의 데이터를 수집하였으며, mmWave 신호에서 VR 사용자 행동을 정확히 추출하는 데 중점을 둡니다.

- **Performance Highlights**: mmSpyVR 시스템은 VR 사용자 행동 유형 인식에서 98.5%의 정확도, 타이핑 입력에서 92.6%의 정확도를 달성했습니다. 이러한 성과는 VR 환경에서도 높은 프라이버시 위협을 나타내며, 다양한 도메인에서의 보안 취약성에 대한 논의를 이끌어낼 수 있습니다.



### OneNet: A Channel-Wise 1D Convolutional U-N (https://arxiv.org/abs/2411.09838)
- **What's New**: 이 논문에서는 U-Net 아키텍처의 경량화된 대안을 제안하며, 이미지 분할(semantic segmentation) 작업을 위한 새로운 1D convolutional encoder를 소개합니다. 이 구조는 파라미터 수를 최대 47%까지 줄이면서도 정확도를 유지하는 것을 목표로 합니다.

- **Technical Details**: 제안된 OneNet 아키텍처는 channel-wise 1D convolutions과 pixel-unshuffle 작업을 결합하여 2D convolutions 없이 공간 관계를 캡처합니다. 이 방식은 edge 디바이스에서의 효율성을 높이고, 공간 정보의 손실을 최소화하면서도 성능을 보장합니다. 또한,  원래 U-Net 구조를 수정하여 모델 크기를 줄이고 계산 비용을 낮추는 것을 목표로 합니다.

- **Performance Highlights**: Benchmarking 결과, 제안된 구조는 U-Net 변형 모델과 비교하여 유사한 정확도를 유지하면서도 더 작은 모델 사이즈와 계산량을 보여주었습니다. 이는 리소스가 제한된 환경에서도 효과적으로 배포 가능함을 시사합니다.



### Automatic Classification of General Movements in Newborns (https://arxiv.org/abs/2411.09821)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 6 pages

- **What's New**: 이번 연구는 아기들의 일반 움직임(General Movements, GMs)을 자동으로 분류하기 위한 기계 학습 알고리즘을 소개합니다. 이 알고리즘은 아기의 비디오 기록에서 신체 움직임을 분석하고 질적으로 평가하는 데 도움을 줄 수 있습니다.

- **Technical Details**: 본 연구는 Barmherzige Brüder Regensburg 병원에서 수집된 76개의 아기 비디오 녹화를 활용하여 신체의 주요 해부학적 지점을 라벨링하고 추적합니다. 이후, 추적된 지점으로부터 특징(feature)을 추출하고, 1D-CNN 및 LSTM과 같은 다양한 기계 학습 분류 모델을 사용하여 GMs의 질을 분류합니다.

- **Performance Highlights**: 연구에서 제안된 접근법은 비디오 길이, 장비 종류, 녹음 환경의 변동성을 고려하여 개발되었습니다. 초기 결과에 따르면, 자동 GMA를 통해 효과적인 아기 신경발달 장애의 조기 발견이 가능할 것으로 기대됩니다.



### Deep Learning for Fetal Inflammatory Response Diagnosis in the Umbilical Cord (https://arxiv.org/abs/2411.09767)
- **What's New**: 이번 연구는 제대의 급성 태아 염증 반응(Acute Fetal Inflammatory Response, FIR)을 디지털 병리학( Digital Pathology)의 최신 딥러닝 기술을 활용해 분류하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 연구팀은 헤마톡실린 및 스테인(Eosin)으로 염색된 제대의 조직병리(histological) 슬라이드 4100개를 디지털화하고, 전자 건강 기록(EHR)에서 태반 진단을 추출했습니다. Attention 기반의 전체 슬라이드 학습(Whole Slide Learning) 모델을 사용하여 FIR을 분류하였으며, 비의료 이미지(ImageNet)로 사전 훈련된 ConvNeXtXLarge 모델과 조직병리 이미지(UNI)를 사용하여 사전 훈련된 모델의 성능을 비교했습니다.

- **Performance Highlights**: 여러 모델을 학습하여 앙상블을 구성한 결과, UNI를 사용한 모델의 예측이 테스트 데이터 세트에서 전체 균형 정확도(balanced accuracy) 0.836을 달성했습니다. 반면 ConvNeXtXLarge를 사용한 앙상블 예측은 0.7209로 더 낮은 정확도를 보였습니다. FIR 2 경우의 경우, 높은 정확도를 가진 모델에서 생성된 히트맵은 적절하게 염증을 나타내는 부위를 강조했습니다.



### Analyzing the AI Nudification Application Ecosystem (https://arxiv.org/abs/2411.09751)
Comments:
          22 pages, 5 figures, 2 tables

- **What's New**: 본 연구는 대중적으로 접근 가능한 20개의 누드화(nudification) 웹사이트의 생태계를 체계적으로 분석한 첫 번째 연구로, 비동의(非同意)로 생성된 성적인 이미지(SNEACI)의 위험성과 이들의 상업적 기반을 파악하고, 개인의 권리를 보호하기 위한 방법을 모색한다.

- **Technical Details**: 연구에서는 누드화 애플리케이션의 클라이언트에 대한 포지셔닝, 광고하는 기능, 그리고 수익화 구조를 분석했다. 대부분의 애플리케이션은 여성을 명시적으로 타겟으로 하고, 비동의 이미지를 생성하는 기능을 제공하며, 사용자 동의 여부는 명확하게 확인하지 않는다.

- **Performance Highlights**: 20개의 애플리케이션 중 19개가 여성을 대상으로 하고 있으며, 절반은 이미지 주체의 동의를 묻지 않는다. 또한 절반의 애플리케이션은 이미지 주체를 성적 행위에 배치하는 기능을 제공하는 등 위험성이 크다.



### Adversarial Attacks Using Differentiable Rendering: A Survey (https://arxiv.org/abs/2411.09749)
- **What's New**: 이번 논문은 differentiable rendering 기법이 3D 객체 및 장면을 조작함으로써 딥 신경망(DNN)을 혼란스럽게 만드는 현실적이고 물리적으로 타당한 적대적 공격을 생성할 수 있는 가능성을 탐구합니다. 이 연구는 기존의 다양한 기법을 체계적으로 정리하고, 연구의 공백을 밝혀내며, 향후 연구 방향을 제시하는 종합적인 프레임워크를 도입합니다.

- **Technical Details**: Differentiable rendering이 DNN의 취약점을 공격하기 위한 방법으로 어떻게 활용될 수 있는지를 분석합니다. 본 프레임워크는 해상도 있는 색 텍스처 조작, 조명 변경, 3D 메시 수정과 같은 구체적인 작업을 분류하여 단일 구조 내에서 관리합니다. 이를 통해 공격자의 목표와 사용할 수 있는 기술을 효과적으로 연결 지을 수 있습니다.

- **Performance Highlights**: 이 연구는 공격 시나리오 및 접근 수준을 포함하여 다양한 DNN 모델(예: 이미지 분류, 얼굴 인식, 객체 탐지)에 대해 어떻게 공격이 수행되는지를 명확하게 설명합니다. 기존의 연구보다 현장 공격에서 differentiable rendering 기술 사용의 넓은 영향력을 보여 주며, 향후 연구를 위한 중요한 방향성을 제시합니다.



New uploads on arXiv(cs.AI)

### Mitigating Parameter Degeneracy using Joint Conditional Diffusion Model for WECC Composite Load Model in Power Systems (https://arxiv.org/abs/2411.10431)
- **What's New**: 본 연구는 WECC 복합 부하 모델(CLM)에서의 파라미터 추정을 개선하기 위해 새로운 Joint Conditional Diffusion Model-based Inverse Problem Solver(JCDI)를 제안합니다. JCDI는 여러 개의 사건을 동시에 고려하는 조건부 구조를 통해 파라미터 일반성을 향상시킵니다.

- **Technical Details**: JCDI는 확산 모델의 정방향 및 역방향 확산 프로세스를 활용하여 파라미터의 후향 분포를 학습합니다. 이 구조는 복잡한 파라미터 분포를 포착하며, 전이 궤적을 기반으로 한 다중 사건에 대한 조건부 추정을 가능하게 합니다. 특히, 파라미터 열화 현상을 줄이는데 기여합니다.

- **Performance Highlights**: JCDI는 기존의 단일 사건 학습 방식에 비해 42.1%의 파라미터 추정 오류 감소를 달성하며, 다양한 고장 사건에서 전력 경로를 예측하는 데 있어 높은 정확성을 보여줍니다. 이 모델은 기존 딥 강화 학습(deep reinforcement learning) 및 감독 학습(supervised learning) 접근 방식을 뛰어넘는 성능을 보였습니다.



### Forming Auxiliary High-confident Instance-level Loss to Promote Learning from Label Proportions (https://arxiv.org/abs/2411.10364)
- **What's New**: 이 논문은 Learning from Label Proportions (LLP)라는 새로운 약한 지도 학습 방법론을 제안하며, 특히 큰 가방 크기에서 가짜 레이블(pseudo-labels)의 정확도를 개선하기 위한 새로운 접근 방식을 도입합니다.

- **Technical Details**: L^2P-AHIL(jointly optimized with the bag-level loss) 방법은 Dual Entropy-based Weight (DEW) 기법을 통해 가방과 인스턴스 수준에서 가짜 레이블의 신뢰도를 적절히 측정하여, 과도한 스무딩을 피하고 정확한 예측을 강조합니다.

- **Performance Highlights**: 실험 결과, L^2P-AHIL은 기존의 기준 방법을 초과하는 성능을 보였으며, 가방 크기가 커질수록 성능 향상이 더욱 두드러졌습니다.



### The Dawn of GUI Agent: A Preliminary Case Study with Claude 3.5 Computer Us (https://arxiv.org/abs/2411.10323)
Comments:
          40 pages, 21 figures, preprint

- **What's New**: Claude 3.5 Computer Use는 공공 베타로 제공되는 GUI 에이전트를 갖춘 최초의 최전선 AI 모델로, 실세계의 복잡한 환경에서 컴퓨터 활용 능력이 아직 잘 알려지지 않았다.

- **Technical Details**: Claude 3.5는 user instruction을 기반으로 API 호출을 통해 end-to-end 솔루션을 제공하며, GUI 상태를 시각적으로 관찰하여 행동을 생성한다. GUI 상호작용을 통해 사용자의 요구를 충족 시키기 위해 'vision-only' 접근 방식을 채택하고 있으며, 이 과정에서 컴퓨터 툴, 텍스트 에디터 툴, Bash 툴 등의 도구가 사용된다.

- **Performance Highlights**: 이 사례 연구는 Claude 3.5의 성능을 여러 사용자 그룹의 요구를 반영하는 다양한 데스크톱 작업 자동화 과제를 통해 평가하였으며, 이에 대한 정량적 및 정성적 평가가 이루어졌다. 이 연구는 Claude의 GUI 자동화 모델의 능력과 한계를 밝혀내고 향후 개선 사항에 대한 논의를 이끈다.



### Scaling Law for Post-training after Model Pruning (https://arxiv.org/abs/2411.10272)
- **What's New**: 이번 논문은 깊이 가지치기(depth pruning), 너비 가지치기(width pruning), 및 2:4 반구조화 가지치기(semi-structured pruning)를 통해 가지치기된 대규모 언어 모델(LLMs)의 후속 훈련(post-training) 요구 사항을 조사하고, 최적의 후속 훈련 데이터 양을 결정하기 위한 스케일링 법칙(scaling law)을 제시합니다.

- **Technical Details**: 연구에서는 Llama-3 및 Qwen-2.5 모델 시리즈에 대해 후속 훈련 실험을 수행하며, 높은 가지치기 비율이 성능 회복을 위해 더 많은 후속 훈련 데이터를 필요로 한다는 것을 발견했습니다. 특히, 더 큰 LLM은 더 적은 데이터로 성능을 회복할 수 있다는 점에서 기존의 직관과 반대된다는 것을 확인했습니다. 제안된 스케일링 법칙은 가지치기 전후의 모델 파라미터 수 및 후속 훈련 토큰 수를 바탕으로 모델의 손실(loss)을 예측할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과에 따르면, Llama-3.1-8B 모델은 16% 가지치기 비율에서 약 1B 토큰에 대해 손실 곡선이 수렴하는 반면, 24% 및 33% 가지치기 비율에서는 더 많은 후속 훈련 데이터가 필요하다는 점이 강조되었습니다. 또한, 2:4 반구조화 가지치기를 적용한 경우 더 큰 모델은 성능 회복을 위해 상대적으로 적은 후속 훈련 데이터를 요구한다는 것을 보여주었습니다.



### Artificial Intelligence in Pediatric Echocardiography: Exploring Challenges, Opportunities, and Clinical Applications with Explainable AI and Federated Learning (https://arxiv.org/abs/2411.10255)
Comments:
          This article is planned for submission to Frontiers Journal

- **What's New**: 이 논문은 소아 심장병의 진단에서 인공지능(AI)의 적용 가능성을 강조하며, 연합 학습(federated learning, FL)과 설명 가능한 AI(explainable AI, XAI)의 역할을 자세히 살펴봅니다.

- **Technical Details**: 소아 초음파(echocardiography) 데이터의 자동 해석을 위한 AI 기술의 채택에는 데이터 공개 부족, 데이터 개인 정보 보호, AI 모델 투명성 등 여러 가지 도전 과제가 있습니다. 이를 해결하기 위해 연구자들은 XAI와 FL과 같은 혁신적인 기술을 활용하고 있습니다.

- **Performance Highlights**: 이 연구는 (i) 뷰 인식(view recognition), (ii) 질병 분류(disease classification), (iii) 심장 구조의 분할(segmentation of cardiac structures), (iv) 심장 기능의 정량적 평가(quantitative assessment of cardiac function) 등 세 가지 임상 사례를 통해 XAI와 FL의 기능성을 보여줍니다.



### A logic for reasoning with inconsistent knowledge -- A reformulation using nowadays terminology (2024) (https://arxiv.org/abs/2411.10197)
Comments:
          The original version was published in the Artificial Intelligence journal. This original version uses 'justifications' in the proof system, which we would call nowadays 'arguments'. The current version presents the same results but now using the terminology of an assumption-based argumentation system

- **What's New**: 이 논문은 불일치한 지식(inconsistent knowledge)으로 추론하는 새로운 논리를 제안합니다. 기존의 전통적인 논리(predicate logic)처럼 전제가 절대 진리로 간주되지 않고, 전제를 가정으로 보고 유용한 결론을 도출할 수 있는 접근 방식을 다룹니다.

- **Technical Details**: 제안된 논리는 N. Rescher의 작업을 일반화한 것으로, 신뢰성 관계(reliability relation)를 사용하여 상충되는 가정(incompatible assumptions) 간의 선택을 가능하게 합니다. 이 논리는 Y. Shoham의 아이디어를 기반으로 한 의미론(semantics)도 제공합니다. 최종적으로, 이 논리는 이상적인 비단조(non-monotonic) 논리의 특성을 모두 갖추고 있습니다.

- **Performance Highlights**: 이 논리는 불일치한 전제 집합에 대해 가장 신뢰성이 낮은 전제를 제거함으로써 모순을 방지할 수 있도록 설계되었습니다. 따라서, 논리적 결론을 도출하는 과정에서 유용성과 신뢰성을 유지하며, 복잡한 정보 출처로 인해 발생하는 불일치를 효과적으로 관리할 수 있습니다.



### Agentic LLMs in the Supply Chain: Towards Autonomous Multi-Agent Consensus-Seeking (https://arxiv.org/abs/2411.10184)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)을 활용하여 공급망 관리(SCM)에서 합의 탐색을 자동화하는 방법을 탐구합니다. 전통적인 SCM은 문제 해결을 위한 인간의 합의에 의존하고 있었으며, LLMs의 최근 발전을 통해 이러한 과정의 자동화가 가능해졌습니다.

- **Technical Details**: 대규모 언어 모델은 방대한 데이터셋으로 훈련되어 협상(negotiation), 추론(reasoning), 계획(planning)을 수행할 수 있습니다. 이 논문은 기존의 공급망-specific 합의 탐색 프레임워크를 위한 자율 LLM 에이전트를 제안하며, 이들의 효과를 재고 관리(case study in inventory management)를 통해 검증하였습니다.

- **Performance Highlights**: 제안된 방법론은 낮은 진입 장벽(minimal entry barriers)으로 거의 인간 수준의 합의를 할 수 있는 가능성을 보여줍니다. 또한, SCM 커뮤니티의 발전을 가속화하기 위해 코드(open-source)를 공개하여 LLM 기반 자율 공급망 솔루션의 추가 발전을 위한 기초를 제공합니다.



### Let people fail! Exploring the influence of explainable virtual and robotic agents in learning-by-doing tasks (https://arxiv.org/abs/2411.10176)
- **What's New**: 이 연구는 인공지능(AI) 에이전트와의 협업이 인간 행동에 미치는 영향을 조사하였으며, 특히 설명 가능한 AI(Explainable AI, XAI) 에이전트와의 상호작용에서 파트너 인식 설명(partner-aware explanations)의 효과를 비교하고, 인간-로봇 상호작용(HRI)에서의 차별성을 강조합니다.

- **Technical Details**: 연구에는 세 가지 그룹의 참가자가 있으며, 각각 컴퓨터와 상호작용, 인간형 로봇과 상호작용, 그리고 자율적으로 작업을 수행하는 경우로 나뉘어 있습니다. 각 그룹의 참가자는 설명 가능한 에이전트의 도움을 받아 미지의 작업을 학습하며, 이는 인간-에이전트 설명 상호작용의 유용성을 직접 측정하는 평가 작업을 포함합니다.

- **Performance Highlights**: 결과는 참가자들이 어떤 AI 에이전트와 상호작용했는지에 따라 다르게 나타났습니다. 컴퓨터와 상호작용한 참가자들은 작업 완료 시간이 향상된 반면, 인간형 로봇과 상호작용한 참가자들은 더 많이 그 로봇의 제안을 따랐지만 시간이 줄어들지는 않았습니다. 흥미롭게도, 자율적으로 작업을 수행한 참가자들은 설명 가능한 AI에 의해 지원받는 참가자들보다 더 나은 지식 습득을 보여주었습니다.



### Semantics and Spatiality of Emergent Communication (https://arxiv.org/abs/2411.10173)
Comments:
          34 pages, to be published in NeurIPS 2024

- **What's New**: 이 논문에서는 인공지능 에이전트들이 공동 훈련을 통해 의미 있는 의사소통을 개발하기 위한 조건으로 'semantic consistency'를 제시합니다.

- **Technical Details**: 의사소통의 의미 일관성을 정의하고, 총체적 목표에 따른 의사소통 전략이 어떻게 작업 성능에 영향을 미치는지를 분석했습니다. 구분(discrimination) 및 재구성(reconstruction)이라는 두 가지 목표를 비교하였습니다. 재구성 시나리오에서 모든 최적 해결책이 semantically consistent하다는 것을 증명했습니다.

- **Performance Highlights**: 실험 결과는 이론적 결과와 일치했으나, 채널 이용도(communication channel utilization)와 관련하여 이론과 실제 간의 격차를 확인하였습니다.



### Evaluating the role of `Constitutions' for learning from AI feedback (https://arxiv.org/abs/2411.10168)
Comments:
          4 pages, 2 figures. In NeurIPS 2024 Workshop on Language Gamification

- **What's New**: 대형 언어 모델(LLMs)의 발전으로 인해 인간 피드백을 대체하여 다른 LLM을 학습하고 평가하는 데 사용되고 있습니다. 본 연구에서는 의사소통 개선을 위해 4개의 다른 'constitution'을 사용하여 피드백의 질이 어떻게 영향을 받는지를 조사했습니다.

- **Technical Details**: 연구에서는 215명의 인간 평가자가 수행한 쌍대 비교(pairwise comparisons)를 통해 각 constitution에 따른 피드백 질을 비교했습니다. 그 결과, 상세한 constitution이 감정적 품질(emotive qualities) 측면에서 더 나은 결과를 나타냈지만, 정보 수집 및 제공과 같은 실용적 기술을 학습하는 데는 어떤 constitution도 베이스라인을 초과하지 못했습니다.

- **Performance Highlights**: 결론적으로, 상세한 constitution이 우선시되어야 하지만, 특정 영역에서 AI 피드백이 보상 신호(reward signal)로서의 효과성에 한계가 있음을 알 수 있습니다.



### Mitigating Sycophancy in Decoder-Only Transformer Architectures: Synthetic Data Intervention (https://arxiv.org/abs/2411.10156)
Comments:
          This research is also submitted to OpenReview. The main text is 9 pages (excluding citations), 7 figures, and 1 table

- **What's New**: 이번 연구는 대형 언어 모델에서 인간 피드백에 의한 sycophancy 문제를 해결하기 위해 디코더 전용 transformer 아키텍처에 synthetic data intervention 기술을 적용하였습니다.

- **Technical Details**: 기존 문헌에서의 연구의 간극에 기반하여 모델의 sycophancy 경향을 줄이기 위해 다양한 데이터를 생성하는 실험 프로세스를 설계했습니다. GPT4o를 실험 도구로 사용하여 100개의 진위 질문을 가지고 synthetic data intervention을 통해 훈련된 모델과 원래의 훈련되지 않은 모델을 여러 지표에서 비교했습니다.

- **Performance Highlights**: 실험 결과, SDI(training model) 모델은 정확도와 sycophancy 비율에서 기술을 지원하며 sycophancy 현상을 줄이는 데 효과적임을 입증했습니다. 데이터 세트, 실험 프로세스, 코드 및 데이터 결과는 Github에 업로드되었습니다.



### Memorization in Attention-only Transformers (https://arxiv.org/abs/2411.10115)
Comments:
          16 pages, 6 figures, submitted to AISTATS 2025,

- **What's New**: 이 논문에서는 멀티-헤드 어텐션에 대한 메모리 용량을 탐구하면서 기존 연구의 한계를 넘어서는 새로운 증명을 제시합니다. 특히, 이 연구는 모든 컨텍스트 크기에 대해 언어 기반 Transformers의 메모리 용량을 확장하는 것을 목표로 하고 있습니다.

- **Technical Details**: 연구는 Attention-only Transformer (AoT)에 대한 메모리 용량을 정의하고 두 가지 유형의 메모리 작업인 연관 작업과 분포 작업을 구분합니다. 연관 작업은 입력 토큰 시퀀스를 기반으로 다음 토큰을 예측하는 것이고, 분포 작업은 입력 시퀀스에 대해 최적의 확률 분포를 예측하는 것을 포함합니다. 이 연구는 Kullback-Leibler (KL) Divergence를 사용하여 분포 작업을 평가하며, AoT 모델의 성능을 이론적으로 분석합니다.

- **Performance Highlights**: 본 연구는 H개 헤드(H𝑓헤드)와 d𝑑숨_dimension을 가진 한 층 AoT를 통해 연관 작업에서 기존의 국한된 컨텍스트 창을 사용하는 결과보다 더 많은 연관성 집합을 정확하게 기억할 수 있다는 것을 증명합니다. 또한, 분포 작업의 경우 AoT가 일반적인 시퀀스 인코더의 분포와 유사한 성능을 보임을 보여줍니다.



### Generative Agent Simulations of 1,000 Peop (https://arxiv.org/abs/2411.10109)
- **What's New**: 이번 연구는 실제 개인 1,052명의 태도와 행동을 시뮬레이션하는 새로운 에이전트 아키텍처를 제시합니다. 이 연구는 대규모 언어 모델(large language models)을 사용하여 질적 면접(qualitative interviews)을 기반으로 하고 있습니다.

- **Technical Details**: 이 에이전트는 General Social Survey에서 참가자들의 응답을 85%의 정확도로 재현하며, 이는 참가자들이 2주 후 자신들의 답변을 재현하는 정확도와 유사합니다. 또한 실험적 재현에서 개인적 특성과 결과를 예측하는 능력도 비슷합니다.

- **Performance Highlights**: 제시된 아키텍처는 인종 및 이념적 그룹에 따른 정확도 편향을 줄이며, 새로운 도구들이 개인 및 집단 행동을 조사하는 데 기초가 될 수 있습니다.



### Adapting the Biological SSVEP Response to Artificial Neural Networks (https://arxiv.org/abs/2411.10084)
Comments:
          10 pages, 5 figures

- **What's New**: 본 논문은 인공지능 신경망(ANN)의 뉴런 중요성을 평가하는 새로운 접근법을 제시합니다. 이 방법은 신경 과학의 주파수 태깅(frequency tagging)에서 영감을 받아, 이미지 입력에 사인파 대비 변조(sinusoidal contrast modulation)를 적용하여 네트워크의 의사결정 과정을 자세히 분석합니다.

- **Technical Details**: 주파수 태깅 기법을 통해 이미지의 특정 부분을 주파수에 따라 변조하여, 각 뉴런의 활성화 반응을 정밀하게 분석합니다. 이러한 신호를 통해 뉴런의 중요성을 평가하고, CNN(Convolutional Neural Network)에서 차별화된 반응의 하모닉(harmonics) 및 상호 변조(intermodulation)를 관찰합니다.

- **Performance Highlights**: 실험 결과, ANN이 생물학적 뇌와 유사하게 깜박이는 주파수에 반응하는 행동을 보임을 확인했습니다. 이 방법은 신경망 가지치기(network pruning) 및 모델 해석 가능성 향상에 기여하며, 설명 가능한 인공지능(Explainable AI)의 발전에 중요한 역할을 할 것으로 기대됩니다.



### Federated Domain Generalization via Prompt Learning and Aggregation (https://arxiv.org/abs/2411.10063)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문에서는 Federated Domain Generalization(FedDG) 설정에서 개인정보 보호 제약을 유지하면서도 데이터 이질성을 다룰 수 있는 새로운 방안으로 Prompt Learning을 도입합니다.

- **Technical Details**: PLAN(프롬프트 학습 및 집계 방법)은 두 단계로 구성된 훈련 프레임워크입니다. 먼저, 각 클라이언트는 자신의 데이터로부터 텍스트 및 시각적 프롬프트 학습을 수행하고, 이후 도메인 특화된 로컬 프롬프트가 클라이언트 간에 교환되고 경량 기반의 주의(Attention) 집계기를 통해 글로벌 프롬프트로 집계됩니다.

- **Performance Highlights**: PLAN 방법은 4개의 벤치마크 데이터셋에서 기존의 FL, DG, FedDG 방법과 프롬프트 학습 기반 방법들보다 현저히 우수한 성능을 보였습니다. 또한, 계산 효율성 및 통신 효율성 면에서도 뚜렷한 장점을 나타냈습니다.



### That Chip Has Sailed: A Critique of Unfounded Skepticism Around AI for Chip Design (https://arxiv.org/abs/2411.10053)
- **What's New**: 이번 논문에서는 AlphaChip에 대한 최근 비판에 대응하여, 이 방법이 실제로 뛰어난 성능을 발휘하고 있음을 강조합니다. Markov의 비판은 심각한 방법론적 문제를 안고 있으며, 그 결과 우리의 연구 결과는 Nature에서 계속해서 유지되고 있습니다.

- **Technical Details**: AlphaChip은 강화학습(reinforcement learning) 기반의 방법으로, 인간 전문가를 초월하는(superhuman) 칩 배치를 생성하는 데 쓰입니다. 그러나 Cheng et al.의 ISPD 논문은 우리의 방법론에 대한 비판을 포함하고 있으며, 그들의 실험은 사전 훈련(pre-training) 없이 진행되었고, 컴퓨팅 자원(compute resources)이 20배 적게 사용되었습니다.

- **Performance Highlights**: Nature 검토 과정에서 우리의 AlphaChip이 실제로 뛰어난 성능을 가지고 있으며, 여러 산업에서 사용되고 있다는 점을 다시 한 번 강조했습니다. 최근 MediaTek이 AlphaChip을 활용하여 그들의 최신 칩을 개발하고 있다는 사실도 부각되고 있습니다.



### Graph-based Complexity for Causal Effect by Empirical Plug-in (https://arxiv.org/abs/2411.10008)
- **What's New**: 이 논문은 인과 효과 쿼리용 경험적 플러그인 추정(computational complexity of computing empirical plug-in estimates) 계산의 복잡성에 초점을 맞추고 있습니다.

- **Technical Details**: 주어진 인과 그래프(causal graph)와 관찰 데이터(observational data)에 따라, 식별 가능한 인과 쿼리(identifiable causal query)를 관찰된 변수에 대한 표현을 통해 추정할 수 있습니다. 이 추정식(estimand)은 데이터로부터 경험적으로 계산된 확률을 사용해 평가될 수 있습니다.

- **Performance Highlights**: 전통적인 견해와는 달리, 고차원 확률 함수가 추정식의 평가 시간을 기하급수적으로 증가시키지 않을 수 있음을 보여줍니다. 특히, 추정식의 구조의 트리너드(treewidth)와 하이퍼트리너드(hypertree width)가 플러그인 추정식의 평가 복잡도를 제한하는 역할을 하며, 하이퍼트리너드가 더 효과적인 경계를 제공하는 경우가 많습니다.



### AMXFP4: Taming Activation Outliers with Asymmetric Microscaling Floating-Point for 4-bit LLM Inferenc (https://arxiv.org/abs/2411.09909)
- **What's New**: 이 논문에서는 Asymmetric Microscaling 4-bit Floating-Point (AMXFP4)라는 새로운 데이터 형식을 소개합니다. 이를 통해 대규모 언어 모델(LLMs)의 추론(inference)에서 4비트 정밀도를 유지하며 효율적인 수행이 가능하게 됐습니다.

- **Technical Details**: AMXFP4는 비대칭 공유 스케일(asymmetric shared scales)을 활용하여 아웃라이어(outlier)를 완화하고, 그룹별 양자화(group-wise quantization)가 도입한 비대칭성을 자연스럽게 캡처합니다. 기존의 4비트 양자화 방법이 필요로 했던 데이터 회전(data rotation) 및 비용이 많이 드는 보정(calibration) 과정을 약화합니다.

- **Performance Highlights**: AMXFP4는 여러 LLM 과제(예: 다중 대화(multi-turn conversations), 긴 문맥 추론(long-context reasoning), 시각적 질문 응답(visual question answering))에서 거의 이상적인 양자화 정확도를 달성했습니다. 이는 MXFP4 및 기타 선도적인 양자화 기술에 비해 매우 높은 성능을 보이며, 강력하고 보정이 필요 없는 4비트 추론을 가능하게 합니다.



### A Hybrid Artificial Intelligence System for Automated EEG Background Analysis and Report Generation (https://arxiv.org/abs/2411.09874)
Comments:
          Example code available at this https URL

- **What's New**: 본 연구는 EEG(뇌파) 해석을 위한 혁신적인 하이브리드 인공지능(AI) 시스템을 제안합니다. 이 시스템은 EEG 배경 활동의 자동 해석과 보고서 생성을 동시에 수행할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 하이브리드 AI 시스템은 PDR(후방 우세 리듬) 예측을 위한 심층 학습 모델, 비지도 학습에 의한 아티팩트 제거, 전문가가 설계한 알고리즘을 결합하여 비정상 탐지를 수행합니다. 1530개의 레이블이 있는 EEG 데이터를 기반으로, 최상의 앙상블 모델은 평균 절대 오차(MAE) 0.237, 제곱근 평균 제곱 오차(RMSE) 0.359, 0.6Hz 이내의 정확도 91.8%, 1.2Hz 이내의 정확도 99%를 달성했습니다.

- **Performance Highlights**: AI 시스템은 일반화된 배경 느림을 탐지하는 데 있어 신경과 의사들보다 통계적으로 유의미하게 우수한 성능을 보였습니다(p = 0.02; F1: AI 0.93, neurologists 0.82). 초점 비정상 탐지에서도 개선된 성능을 보였으며(F1: AI 0.71, neurologists 0.55), 내부 데이터셋과 Temple University 비정상 EEG 데이터셋에 대한 검증에서 일관된 성능을 유지했습니다.



### VeriGraph: Scene Graphs for Execution Verifiable Robot Planning (https://arxiv.org/abs/2411.10446)
- **What's New**: VeriGraph는 로봇 작업 계획에서 VLM(vision-language models)을 통합하고 작업 가능성을 검증하는 새로운 프레임워크입니다. 이 시스템은 장면 그래프(scene graphs)를 중간 표현으로 사용하여 물체의 핵심 및 공간적 관계를 포착하고 작업의 검증 및 세분화를 개선합니다.

- **Technical Details**: VeriGraph는 입력 이미지를 통해 장면 그래프를 생성하고, 이를 사용하여 LLM 기반 작업 계획자가 생성한 동작 시퀀스를 반복적으로 검사하고 수정합니다. 이 방식은 물리적 제약 조건을 준수하며 실행 가능한 행동을 보장합니다. 또한, 시스템은 목표 장면 그래프를 생성하는 두 가지 입력 형식(참조 장면 이미지 또는 자연어 지침)을 지원합니다.

- **Performance Highlights**: VeriGraph는 다양한 조작 시나리오에서 작업 완료율을 크게 향상시켰으며, 언어 기반 작업에 대해 58%, 이미지 기반 작업에 대해 30% 더 높은 성능을 보여주었습니다.



### Mitigating Hallucination in Multimodal Large Language Model via Hallucination-targeted Direct Preference Optimization (https://arxiv.org/abs/2411.10436)
- **What's New**: 본 연구에서는 Hallucination-targeted Direct Preference Optimization (HDPO)을 제안하여 다중 모달 큰 언어 모델(MLLMs)의 환각(hallucination)을 줄이는 새로운 접근 방식을 도입합니다. 이전 방법들과 달리, HDPO는 MLLM의 다양한 환각 원인과 형태를 다룹니다.

- **Technical Details**: HDPO는 세 가지 유형의 환각 원인에 대한 선택적 preference pair 데이터를 개발하여 구체적인 환각 상황을 해결합니다: 1) 충분하지 않은 시각적 능력, 2) 긴 맥락 생성, 3) 다중 모달 충돌. 각 유형에 대해 고유한 데이터를 생성하며, 이를 통해 MLLM의 능력을 향상시킵니다.

- **Performance Highlights**: 실험 결과 HDPO는 다양한 환각 평가 데이터셋에서 우수한 성능을 보여주었으며, 대부분의 최신 방법(SOTA)을 초월하며 MLLM 환각 문제를 효과적으로 완화했습니다. 특히, 다양한 M-hallu 작업에서 일관된 개선을 달성했습니다.



### Evaluating Creativity and Deception in Large Language Models: A Simulation Framework for Multi-Agent Balderdash (https://arxiv.org/abs/2411.10422)
Comments:
          Accepted at Wordplay: When Language Meets Games @ ACL 2024

- **What's New**: 이 논문에서는 게임 Balderdash를 활용하여 대형 언어 모델(Large Language Models, LLMs)의 창의성(creativity)과 논리적 추리(logical reasoning) 능력을 평가하는 시뮬레이션 프레임워크를 소개합니다. Balderdash 게임을 통해 LLM의 확인할 수 없는 단어에 대한 그들의 정의를 생성하고 게임 규칙과 역사를 기반으로 전략을 세우는 능력을 측정합니다.

- **Technical Details**: 생성된 프레임워크는 중앙집중식 게임 엔진(centralized game engine)을 통해 다양한 LLM이 참가하고, 정의의 의미적 동등성(semantic equivalence)을 평가하는 판별 LLM(judge LLM)을 포함합니다. 실험에서는 True Definition Ratio, Deception Ratio, Correct Guess Ratio와 같은 메트릭(metrics)을 통해 LLM의 성능을 분석합니다.

- **Performance Highlights**: 결과적으로, 이 연구는 인식하기 어려운 어휘가 LLM의 입력에 포함될 경우 게임 규칙 및 역사적 맥락에 대한 사고가 저조할 수 있음을 밝혔습니다. LLM의 창의적이고 기만적인 능력에 대한 통찰을 제공하며, 강점과 개선이 필요한 영역을 조명합니다.



### Towards Automatic Evaluation of Task-Oriented Dialogue Flows (https://arxiv.org/abs/2411.10416)
- **What's New**: 이번 논문에서는 대화 흐름(diaogue flows)의 품질을 평가하기 위한 새로운 지표인 FuDGE (Fuzzy Dialogue-Graph Edit Distance)를 소개합니다. 이 지표는 대화 흐름의 구조적 복잡성과 대화 데이터의 표현 범위를 평가하여 대화 시스템의 효율성과 자동화를 높이는 데 기여할 수 있습니다.

- **Technical Details**: FuDGE는 개별 대화가 대화 흐름과 얼마나 잘 일치하는지를 측정하고, 전반적으로 대화 세트가 대화 흐름에 얼마나 잘 표현되는지를 계산합니다. 이 연구에서는 효율적인 edit-distance 메트릭을 통해 대화 흐름과 대화 간의 거리를 계산하고, Flow-F1 (FF1) 점수를 사용하여 복잡성과 압축 간의 트레이드오프를 포착합니다.

- **Performance Highlights**: FuDGE 평가 프레임워크를 통해 수동으로 구성된 대화 흐름 및 자동 생성된 흐름을 대규모 실험을 통해 평가할 수 있었으며, 이는 대화 흐름의 품질을 표준화하고 최적화할 수 있는 가능성을 보여줍니다.



### Repurposing Stable Diffusion Attention for Training-Free Unsupervised Interactive Segmentation (https://arxiv.org/abs/2411.10411)
- **What's New**: 이 논문에서는 Stable Diffusion(안정적인 확산) 기반의 새로운 비지도 훈련 없는 접근 방식을 제안합니다. 기존의 비지도 방법들이 사용하는 pseudo-labels 대신 self-attention 맵을 활용하여 더 적은 노이즈와 뚜렷한 경계를 지닌 Markov-map을 생성합니다.

- **Technical Details**: 제안된 Markov-map은 self-attention 텐서를 Markov 전이 연산자로 해석하여 반복적으로 Markov 체인을 형성합니다. 각 픽셀은 특정 확률 값에 도달하기 위해 필요한 반복 횟수를 카운트하여 Markov 맵을 생성합니다. 이 방법은 분할 지역 간의 보다 일관된 값을 제공합니다.

- **Performance Highlights**: 모든 실험에서 Number of Clicks (NoC) 기준으로 뛰어난 성과를 기록하였으며, 기존 훈련 기반 비지도 방법들보다 여러 데이터셋에서 우수한 성능을 보였습니다. 이는 훈련이 필요 없는 방식으로도 가능하다는 점에서 큰 진전을 이루었습니다.



### Features that Make a Difference: Leveraging Gradients for Improved Dictionary Learning (https://arxiv.org/abs/2411.10397)
Comments:
          9 pages, 8 figures. Submitted to NAACL 2025

- **What's New**: 본 논문에서는 Gradient Sparse Autoencoders (g-SAEs)를 도입하여 기존의 Sparse Autoencoders (SAEs)에서 발생하는 한계를 극복하고자 합니다. g-SAEs는 활성화 값뿐만 아니라 입력 활성화의 gradient를 고려하여 $k$개의 요소를 선택하는 TopK 활성화 기능을 수정하여, 모델 출력에 강하게 영향을 미치는 특징을 학습할 수 있도록 합니다.

- **Technical Details**: g-SAEs는 기존 SAE 아키텍처를 개선하여 gradient-aware TopK 활성화 기능을 활용합니다. 이 방법이 모델의 다운스트림 효과를 더 정확하게 포착할 수 있도록 해주며, 그 결과로 생성된 재구성을 원본 네트워크 성능에 더 충실합니다. 또한 g-SAEs는 더 적은 수의 비활성 단위를 가지고 있으며, 활성화의 gradient를 포함하여 여러 인기 있는 SAE 아키텍처의 Pareto frontiers에서 개선을 보여줍니다.

- **Performance Highlights**: g-SAEs는 기존 아키텍처와 비슷한 해석 가능성을 유지하면서, 특정 logits에 대해 더 많은 영향을 미치는 latents를 복구합니다. 이는 g-SAEs가 모델에 더 온전한 제어를 가능하게 하며, 다양한 맥락에서 향상된 모델 성능 조작을 제공합니다.



### Deep Learning for Micro-Scale Crack Detection on Imbalanced Datasets Using Key Point Localization (https://arxiv.org/abs/2411.10389)
- **What's New**: 이 논문은 구조적 건강 모니터링 영역에서 내부 균열 탐지에 대한 심층 학습(Deep Learning, DL) 방법의 새로운 적용을 탐구합니다. 특히 마이크로 스케일 균열을 식별하기 위해 DL 기반의 키 포인트 탐지 기술을 사용하여, 균열의 경계를 정의하는 네 개의 키 포인트 좌표를 예측함으로써 균열을 국지화하는 방식을 제시합니다.

- **Technical Details**: 이 연구에서는 Inception 모듈을 포함하는 Wide Convolutional Networks을 사용하여 수치 데이터에서 균열을 감지하기 위한 새로운 접근법을 제시합니다. 모델은 다양한 필터 크기로 구성된 여러 컨볼루션 레이어를 사용하여 빠르고 효율적으로 다양한 특징을 추출합니다. 또한, Attention Mechanisms을 도입해 특징 맵 내에서 중요 영역에 집중하여 성능을 향상시킵니다.

- **Performance Highlights**: 모델은 마이크로 스케일 균열 탐지에 적용되었으며, 실제 균열 위치와 예측된 균열 위치 간 평균 Intersection over Union (IoU) 값이 0.511로 쉬운 미세 균열 및 0.631로 큰 균열에 대해 나타났습니다. 이는 균열 탐지에서 이전의 Deep Learning 모델보다 성능 향상을 보여줍니다.



### Low-Latency Task-Oriented Communications with Multi-Round, Multi-Task Deep Learning (https://arxiv.org/abs/2411.10385)
- **What's New**: 본 논문에서는 다음 세대 통신 시스템에서의 작업 지향(Goal-oriented) 통신을 향상시키기 위한 다중 라운드 및 다중 작업 학습(MRMTL) 접근법을 제안합니다. 이 방법은 수신자로부터의 피드백에 따라 송신자가 동적으로 채널 사용량을 업데이트할 수 있도록 합니다.

- **Technical Details**: MRMTL은 송신자가 압축된 잠재 표현(Compressed latent representations)을 생성하며, 수신자는 이러한 신호를 분류하는 기계 학습 과제를 수행합니다. 이 과정에서 깊은 신경망(Deep Neural Networks, DNN)이 사용되며, 채널과 데이터 특성을 고려하여 공동 훈련됩니다.

- **Performance Highlights**: MRMTL은 기존의 대규모 채널 사용량이 필요한 방법과 유사한 정확도를 달성하면서, 이전 라운드의 신호를 통합함으로써 지연(Delay)을 크게 줄였습니다. CIFAR-10 데이터셋을 사용하여 실험한 결과, MRMTL이 작업 지향 통신의 효율성을 현저히 향상시키는 것으로 나타났습니다.



### A Survey of Event Causality Identification: Principles, Taxonomy, Challenges, and Assessmen (https://arxiv.org/abs/2411.10371)
- **What's New**: 이번 연구는 Event Causality Identification (ECI)에 대한 체계적인 검토와 평가를 제공하며, 기존 연구 방법론과 모델의 양적 평가를 포함합니다. ECI의 개념적 틀을 수립하고, 문장 수준(SECI) 및 문서 수준(DECI) 이벤트 인과 관계 식별을 위한 분류 체계를 제안합니다.

- **Technical Details**: ECI는 텍스트 데이터에서 이벤트 간의 인과관계를 자동으로 추출하는 자연어 처리(NLP) 작업입니다. SECI는 feature pattern 기반 매칭, 심층 의미 인코딩, 인과 지식 사전 훈련, 프롬프트 기반 미세 조정 등 다양한 방법을 분석하며, DECI는 이벤트 그래프 추론 및 프롬프트 기반 기술을 중심으로 복잡한 교차 문장 인과 추론 문제를 해결합니다.

- **Performance Highlights**: 본 논문은 두 개의 기준 데이터 세트에 대한 다양한 ECI 방법의 양적 평가를 수행하며, 각 접근법의 강점, 한계 및 개방된 도전 과제를 분석합니다. 또한, LLMs를 활용한 최근 연구 동향을 조명하여 ECI의 향후 발전 방향을 제안합니다.



### Towards High-Fidelity 3D Portrait Generation with Rich Details by Cross-View Prior-Aware Diffusion (https://arxiv.org/abs/2411.10369)
- **What's New**: 최근 단일 이미지 기반 3D 초상화 생성 기술에서, 새로운 Hybrid Priors Diffusion 모델을 통해 다중 뷰 사전 정보(multi-view priors)를 활용하여 세밀하고 일관된 3D 초상화를 생성할 수 있는 방법을 제안합니다.

- **Technical Details**: 1. Hybrid Priors Diffusion 모델(HPDM)은 명시적 및 암시적으로 다중 뷰 사전을 고려하여 조건을 집어넣어 초상화의 일관성을 향상시킵니다. 2. Multi-View Noise Resampling Strategy(MV-NRS)를 도입하여 다양한 뷰의 노이즈 분포를 관리하고, SDS 손실을 통해 일관된 표현을 추구합니다. 3. GAN-prior Initialization과 Portrait Geometry Restoration, Multi-view Diffusion Refinement 모듈로 구성된 포트레이트 확산 파이프라인을 개발했습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 높은 정밀도의 3D 초상화를 생성하는 데 성공하였으며, 풍부한 세부 묘사를 확보했습니다.



### Mechanisms of Generative Image-to-Image Translation Networks (https://arxiv.org/abs/2411.10368)
- **What's New**: 이번 논문에서는 기존의 복잡한 구조를 간소화한 이미지 간 변환을 위한 새로운 네트워크 아키텍처를 제안합니다. GAN (Generative Adversarial Network)과 autoencoder의 관계를 조사하여, 이미지 변환 작업에서 GAN 구성 요소만을 사용하는 것이 효과적이라는 설명을 제공합니다.

- **Technical Details**: 저자는 충분한 용량의 판별자 (discriminator)가 있을 경우, GAN 기반의 훈련이 전통적인 autoencoder 모델과 유사한 결과를 낸다는 점을 실험적으로 증명합니다. 또, 이미지 간 변환 문제에 대해 간단한 GAN 모델이 공통적인 특징을 유지하면서 새로운 특징을 생성할 수 있다는 것을 보여줍니다.

- **Performance Highlights**: 실험을 통해 GAN 및 autoencoder의 성능을 비교하고, 이미지 간 변환의 모델 능력과 특정 제약 조건을 조사하여 기존 방법과 차별화된 성능을 확인하였습니다.



### Continual Adversarial Reinforcement Learning (CARL) of False Data Injection detection: forgetting and explainability (https://arxiv.org/abs/2411.10367)
- **What's New**: 이 논문은 스마터 인버터에 대한 허위 데이터 주입 공격(FDIAs)이 증가하는 문제를 다루고 있으며, 이를 해결하기 위한 새로운 접근법으로 지속적 적대적 강화학습(CARL)을 제안합니다. 기존의 데이터 기반( data-based) 검출 방법의 취약점을 드러내고, 이에 대한 개선점을 설명합니다.

- **Technical Details**: 제안된 방법은 적대적 예제를 포함하여 데이터 기반 검출 훈련 절차를 향상시키고, 지속적인 학습 구현이 치명적 망각(catastrophic forgetting)에 영향을 받을 수 있음을 보여줍니다. 이러한 망각은 모든 생성된 FDIA 시나리오에 대해 공동 훈련(joint training) 전략을 적용함으로써 해결할 수 있습니다.

- **Performance Highlights**: 논문에서는 제안된 메커니즘이 기존 방법보다 더 나은 적대적 공격 내성을 달성할 수 있음을 나타내며, 지속적 학습이 데이터 기반 검출의 한계를 극복하는 데 효과적임을 입증합니다.



### Domain Adaptation-based Edge Computing for Cross-Conditions Fault Diagnosis (https://arxiv.org/abs/2411.10340)
Comments:
          28 pages, 11 figures

- **What's New**: 기계 장비의 건강한 작동을 지원하는 결함 진단 기술에 대한 연구가 진행되었습니다. 이 논문에서는 경량화된 결함 진단 프레임워크를 제안하여 다양한 운영 조건에서의 결함 진단 및 응용 프로그램 배포의 중요성을 강조합니다.

- **Technical Details**: 제안된 프레임워크는 domain adaptation을 기반으로 하며, 로컬 최대 평균 차이를 (local maximum mean discrepancy) 활용하여 서로 다른 도메인 간의 특징 분포를 정렬하여 공통 특징 공간을 발견합니다. 이 과정에서는 클라우드 모델에서 얻어진 결함 진단 전문 지식을 경량 엣지 모델로 이전하는 방법도 포함됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존 방법보다 진단 정확도를 평균 34.44%, 17.33% 향상시켰으며, 평균 추론 속도는 80.47% 증가했습니다. 또한 엣지 모델의 매개변수 수는 96.37% 감소하였고, Flops는 83.08% 감소하였습니다.



### Safe Text-to-Image Generation: Simply Sanitize the Prompt Embedding (https://arxiv.org/abs/2411.10329)
- **What's New**: 최근 Text-to-Image (T2I) 생성 모델의 발전으로 고품질 이미지를 텍스트 설명에 맞춰 생성할 수 있게 되었습니다. 하지만 이러한 모델은 불건전한 생성의 위험이 존재하여, 사용 정책을 위반하는 유해 콘텐츠를 생성할 수 있습니다. 기존의 안전 생성 방법은 주로 비주얼 표현에서 바람직하지 않은 개념을 삭제하는 데 초점을 맞추었지만, 텍스트 표현의 정화는 간과했습니다. 본 논문에서는 Prompt Embeddings에서 바람직하지 않은 개념을 지우는 Vision-Agnostic 안전 생성 프레임워크인 Embedding Sanitizer (ES)를 제안합니다.

- **Technical Details**: Embedding Sanitizer (ES)는 텍스트 인코더의 출력에 적용되는 플러그 앤 플레이 모듈로, 내부 점수 네트워크를 통해 각각의 토큰이 가질 수 있는 유해성을 평가합니다. 이 시스템은 유해 점수가 높은 토큰에 대해 강하게 정화 작업을 수행하고, 점수가 낮은 토큰에 대해서는 최소한의 영향을 미치도록 조정을 진행합니다. ES는 지정된 목표 개념을 지우기 위한 교육 과정에서 다양한 맥락 샘플을 생성하여 전반적인 안전성을 높입니다.

- **Performance Highlights**: ES는 다섯 개의 프롬프트 벤치마크에서의 평가에서 아홉 개의 기준 방법과 비교하여 SOTA (State-of-the-Art) 강인성을 달성하였습니다. ES는 기존의 안전 장치들에 비해 해석 가능성과 통제력을 제공하며, 생성 품질을 유지하는 동시에 유해 콘텐츠 생성의 주요 원천인 프롬프트 임베딩의 정화를 통해 우수한 성능을 발휘합니다.



### A Realistic Collimated X-Ray Image Simulation Pipelin (https://arxiv.org/abs/2411.10308)
- **What's New**: 이 논문은 X-ray 시스템에서 비정확한 검출기의 위치 정보로 인해 발생하는 collimator 검출의 어려움을 해결하기 위해 물리적으로 동기화된 이미지 처리 파이프라인을 제안합니다. 이 파이프라인은 collimator 그림자의 특성을 시뮬레이션하여 제한된 데이터 세트를 확장할 수 있도록 설계되었습니다.

- **Technical Details**: 시뮬레이션 과정은 크게 세 가지 단계로 나뉘어집니다. 첫 번째 단계에서 collimator 영역의 형태와 위치를 정의하기 위한 무작위 레이블이 생성됩니다. 두 번째 단계에서는 산란 방사선이 도입되고, 마지막 단계에서는 노이즈를 추가합니다. 프레임워크는 콜리메이터의 핵심 요인을 모델링하여 산란된 방사선의 분포를 명시합니다.

- **Performance Highlights**: 이러한 통합된 접근법의 유효성은 실제 collimator 그림자와의 정성적 및 정량적 비교를 통해 검증되었으며, 시뮬레이션 데이터를 깊이있는 학습 프레임워크 내에서 활용함으로써 실제 collimator에 대한 적절한 대체 역할을 할 뿐만 아니라 실제 데이터에 적용했을 때 일반화 성능이 향상됨을 보여주었습니다.



### RETR: Multi-View Radar Detection Transformer for Indoor Perception (https://arxiv.org/abs/2411.10293)
Comments:
          24 pages, Accepted to NeurIPS 2024

- **What's New**: 이 논문에서는 다중 뷰 레이더 인식을 위해 개발된 Radar dEtection TRansformer(RETR)를 제안합니다. 이는 기존의 DETR 아키텍처를 기반으로 하며, 레이더 신호의 고유한 특성을 고려한 설계를 포함하고 있습니다.

- **Technical Details**: RETR는 다음과 같은 주요 구성 요소를 갖추고 있습니다: 1) 조정 가능한 위치 인코딩(Tunable Positional Encoding, TPE)을 통해 깊이 우선 피쳐 유사도를 강화; 2) 레이더와 카메라 좌표계에서의 삼면 손실(tri-plane loss)을 도입; 3) 재매개변수를 통한 학습 가능한 레이더-카메라 변환을 실현하여 다중 뷰 레이더 상황에 맞는 특성을 반영합니다.

- **Performance Highlights**: RETR는 HIBER 및 MMVR 데이터셋에 대해 평가되었으며, 객체 탐지에서 기존 최첨단 방법보다 15.38 이상의 AP(평균 정밀도)를 향상시키고, 인스턴스 세분화에서 11.77 이상의 IoU(교차 영역 비율)를 달성했습니다.



### The ParClusterers Benchmark Suite (PCBS): A Fine-Grained Analysis of Scalable Graph Clustering (https://arxiv.org/abs/2411.10290)
Comments:
          This is a preliminary version of a paper that will appear at VLDB'25

- **What's New**: ParClusterers Benchmark Suite (PCBS)를 소개합니다. 이는 다양한 고도로 확장 가능한 병렬 그래프 클러스터링 알고리즘과 벤치마킹 도구를 포함하여, 서로 다른 그래프 클러스터링 알고리즘 및 구현을 비교하는 데 도움을 줍니다.

- **Technical Details**: PCBS는 커뮤니티 탐지, 분류, 밀집 부분 그래프 마이닝 등 현대의 다양한 클러스터링 사용 사례를 목표로 하는 알고리즘을 포함합니다. 벤치마크 도구 키트는 다양한 클러스터링 알고리즘의 여러 인스턴스를 실행하고 평가하기 쉽게 만들어 주며, 이를 통해 특정 작업의 클러스터링 성능을 미세 조정하고 서로 다른 클러스터링 알고리즘을 비교할 수 있습니다.

- **Performance Highlights**: PCBS를 활용한 결과, 상위 품질 결과는 많은 인기 그래프 클러스터링 도구 키트에 포함되지 않은 알고리즘에서 얻어졌습니다. 클러스터링 구현의 속도도 향상되어 PCBS는 Neo4j보다 평균 32.5배, TigerGraph보다 303배 빠릅니다. 특히, Correlation Clustering은 네 가지 작업 중 세 가지에서 최고의 품질을 보여주었습니다.



### Systolic Arrays and Structured Pruning Co-design for Efficient Transformers in Edge Systems (https://arxiv.org/abs/2411.10285)
Comments:
          7 pages, 10 figures

- **What's New**: 이 논문에서는 Systolic Array Structured Pruning (SASP)이라는 새로운 설계 전략을 제안하여 시스템 성능을 개선하는 방법을 다룹니다. SASP는 구조적 가지치기(Structured Pruning)와 시스템에서의 시타딕(Systolic) 배열의 청사진을 결합하여, pruning된 블록의 크기를 시타딕 배열의 형태와 일치시킵니다.

- **Technical Details**: SASP는 구조적 가지치기와 시타딕 배열의 효율적 최적화를 통합하는 새로운 방법론으로, 알고리즘 최적화, 시스템 시뮬레이션, 하드웨어 설계를 아우르는 다차원 설계 공간을 탐색합니다. 이 방법을 통해, 변환기(transformer)의 알고리즘을 최적화하고, 시스템 성능에 미치는 영향을 분석합니다. 실험에서는 20%의 가지치기를 진행하면서 비복원된 시스템에 비해 44%의 속도 향상과 42%의 에너지 절약을 달성했습니다.

- **Performance Highlights**: 구조적 가지치기를 통해 최대 26%의 시스템 전반의 속도 향상을 달성했으며, Librispeech 데이터셋에서 1.4%의 단어 오류율(WER) 저하만을 보였습니다. SASP는 변환기와 시타딕 배열의 협업 최적화를 통해 높은 품질의 서비스를 유지하며 성능을 크게 향상시킬 수 있음을 보여줍니다.



### Lateral Movement Detection via Time-aware Subgraph Classification on Authentication Logs (https://arxiv.org/abs/2411.10279)
- **What's New**: 본 논문에서는 APT 공격 중 측면 이동(lateral movement)을 탐지하기 위한 새로운 접근법인 LMDetect 프레임워크를 제안합니다. 이 프레임워크는 호스트 인증 로그 데이터를 그래프 시각화하여 다중 스케일(multi-scale) 탐지를 가능하게 하며, 인증 이벤트를 중심으로 한 서브그래프 생성을 통해 숨겨진 이상 행동 패턴을 포착합니다. 이를 통해 기존 탐지 방법보다 더 효과적이고 우수한 성능을 보입니다.

- **Technical Details**: LMDetect는 주로 세 가지 과정으로 구성됩니다. 첫째, 호스트 인증 로그 데이터를 사용하여 이종 다중 그래프(heterogeneous multigraph)를 구축하여 내부 시스템 엔터티 간의 상관관계를 강화합니다. 둘째, 인증 이벤트 중심의 서브그래프를 추출하는 시간 인식(subgraph generator) 생성기를 설계합니다. 셋째, 다중 스케일 주의 인코더(multi-scale attention encoder)를 통해 인증 서브그래프 내 숨겨진 이상 패턴을 탐지합니다. 이 방법은 대규모 인증 로그 데이터에서도 높은 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과, LMDetect 프레임워크는 두 개의 실제 인증 로그 데이터 세트에서 대규모 측면 이동 행동 탐지 작업에서 다른 기존 방법들보다 탁월한 성능을 나타냈습니다. 이는 특히 전통적인 EDR, ML 및 DL 기반 방법들의 한계를 극복했습니다.



### The Unreasonable Effectiveness of Guidance for Diffusion Models (https://arxiv.org/abs/2411.10257)
Comments:
          Preprint. 19 pages, 14 figures in total, including references and appendix

- **What's New**: 이 논문에서는 새로운 에러 교정 기법인 슬라이딩 윈도우 가이던스(SWG)를 소개합니다. SWG는 주 모델의 수용 영역(receptive field)을 제어하여 고유한 방식으로 모델을 유도하며, 인간의 선호도와 더 잘 일치하면서도 훈련, 아키텍처 수정 또는 클래스 조건화가 필요 없습니다.

- **Technical Details**: SWG는 장기 공간 의존성(long-range spatial dependencies)을 강화하여 시각 품질을 개선하는 새로운 가이던스 방법입니다. 전통적인 클래스 없는 가이던스(Classifier-Free Guidance, CFG)와 달리 SWG는 추가 훈련이나 아키텍처적 변경 없이 어떤 확산 모델(Diffusion Model, DM)에도 적용할 수 있습니다. 이 방법은 특히 약한 모델 가이던스(Weak Model Guidance, WMG) 방식과 대조되며, 더 강력한 가중치 정규화(weight regularization)를 수행하여 생성된 샘플의 품질을 향상시킵니다.

- **Performance Highlights**: SWG는 최신 가이던스 기법과 비교해 경쟁력 있는 생성 성능을 발휘하며, 인간의 선호도와의 일치도가 더 높습니다. 이 방법은 어떤 DM에서도 사용 가능하고, 시각적으로 높은 품질의 이미지를 생성하는 데 효과적입니다.



### Generative AI in Multimodal User Interfaces: Trends, Challenges, and Cross-Platform Adaptability (https://arxiv.org/abs/2411.10234)
Comments:
          13 pages, 4 figures

- **What's New**: Generative AI는 사용자 인터페이스 설계를 혁신하며 개인화된, 다중 모달(multimodal), 플랫폼 간(inter-platform) 상호작용 가능성을 제시합니다. 이 논문은 Generative AI의 통합, 다중 모달 상호작용, 플랫폼 간 적응성, 그리고 역동적인 개인화에 중점을 두고 내용을 탐구합니다.

- **Technical Details**: 다중 모달 대형 언어 모델(multimodal large language models, LLMs), 경량 프레임워크(lightweight frameworks), 모바일 하드웨어(mobile hardware), 클라우드와 온디바이스 프로세싱(cloud and on-device processing)의 균형, 맥락 보존(context retention), 프라이버시(privacies) 문제 등 기술적 및 윤리적 도전 과제가 포함됩니다.

- **Performance Highlights**: Generative AI를 통해 향후 감정적으로 적응하는 인터페이스(emotionally adaptive interfaces) 및 실시간 협업 시스템(real-time collaborative systems)을 가능케 하여 사용자 중심 인터페이스를 재정의할 수 있는 잠재력이 있습니다.



### ColorEdit: Training-free Image-Guided Color editing with diffusion mod (https://arxiv.org/abs/2411.10232)
- **What's New**: 본 논문은 기존 텍스트-유도 이미지 편집 기술의 색상 변경 실패 원인을 분석하고 이를 해결하기 위한 새로운 방법을 제안합니다. 주요 발견으로 색상 속성 조정의 안정성을 위해 이미지-유도 방법을 활용하며, COLORBENCH라는 평가 벤치마크 데이터셋도 소개합니다.

- **Technical Details**: 논문에서는 텍스트 유도 이미지 합성 과정에서의 다양한 cross-attention 블록이 학습한 의미 정보를 분석합니다. 특히, denoising 과정 초기 단계에서 U-Net 디코더를 통해 객체의 형태와 질감을 결정하며, cross-attention 레이어의 Value 행렬 정렬을 통해 색상 조정을 가능하게 합니다.

- **Performance Highlights**: 제안된 이미지-유도 색상 편집 방법은 기존의 텍스트 유도 이미지 편집 방법들보다 우수한 성능을 보이며, 실험을 통해 생성된 이미지와 실제 이미지 모두에서 효과성을 입증했습니다.



### A Low-Resolution Image is Worth 1x1 Words: Enabling Fine Image Super-Resolution with Transformers and TaylorShif (https://arxiv.org/abs/2411.10231)
- **What's New**: 이 연구에서는 1x1 패치 크기를 활용하여 pixel-level 처리(pixel-level processing)를 가능하게 하는 TaylorIR이라는 변환기 기반의 Super-Resolution(SR) 모델을 제안합니다. 이는 기존 SR 모델의 단점을 극복하고자 합니다.

- **Technical Details**: TaylorIR은 기존의 self-attention 메커니즘 대신 TaylorShift attention 메커니즘을 도입합니다. TaylorShift는 Taylor 급수 전개(Taylor series expansion)를 기반으로 하여 메모리 효율적인 방식으로 구현됩니다. 이를 통해 전체 token-to-token 상호작용을 선형 복잡도로 달성할 수 있습니다.

- **Performance Highlights**: TaylorIR을 사용한 SwinIR 모델인 TaylorSwinIR은 PSNR과 SSIM 메트릭에서 기존의 SR 모델들보다 뛰어난 성능을 발휘하며, 메모리 소비는 최대 60%까지 줄일 수 있음을 실험을 통해 입증하였습니다.



### MCL: Multi-view Enhanced Contrastive Learning for Chest X-ray Report Generation (https://arxiv.org/abs/2411.10224)
Comments:
this https URL

- **What's New**: 이번 연구에서는 다중 뷰 영상(x-ray)에서 종합적인 정보(정보)를 활용해 방사선 보고서를 생성하는 Multi-view enhanced Contrastive Learning (MCL) 방법을 제안합니다. 이는 기존의 단일 뷰를 기반으로 한 자동 보고서 생성 방법의 한계를 극복하는 데 초점을 맞추고 있습니다.

- **Technical Details**: MCL 방법은 두 단계로 구성됩니다. 첫 번째 단계에서는 다중 뷰 향상 대조 학습(multi-view enhanced contrastive learning)을 통해 동일 연구 내의 여러 뷰(x-ray)와 이들에 해당하는 보고서 간의 일치를 최대화합니다. 두 번째 단계에서는 교차 주의(cross-attention) 메커니즘을 활용해 환자별 정보(예: 증상)를 보고서 생성에 통합합니다. 또한, 누락된 정보를 보완하기 위한 '전이 브릿지(transitional bridge)'를 도입하여 이로 인한 임베딩 공간 차이를 줄입니다.

- **Performance Highlights**: MCL은 여러 데이터셋에서 최근의 최첨단 방법들을 초월하는 성능을 보였습니다. MIMIC-CXR에서 5.0%의 F1 RadGraph 향상, MIMIC-ABN에서 7.3%의 BLEU-1 향상, Multi-view CXR에서 3.1%의 BLEU-4 향상, Two-view CXR에서 8.2%의 F1 CheXbert 향상을 달성했습니다.



### An Empirical Study on LLM-based Agents for Automated Bug Fixing (https://arxiv.org/abs/2411.10213)
- **What's New**: 본 논문은 LLM 기반의 자동 버그 수정 시스템의 성능 차이를 체계적으로 분석한 최초의 연구입니다. LLM(대형 언어 모델)과 이를 기반으로 한 에이전트들이 소프트웨어 결함을 해결하는 데 효과적임을 보여주고 있으며, SWE-bench Lite 벤치마크를 통해 여러 시스템의 성능을 비교합니다.

- **Technical Details**: 7개의 상용 및 오픈 소스 시스템을 대상으로 SWE-bench Lite 벤치마크를 사용하여 성능을 분석하였습니다. 각 시스템의 fault localization(결함 위치 확인) 정확도를 파일 및 라인 수준에서 비교하고, 동적 재현을 통해서만 해결 가능한 사례를 평가하여, LLM 및 에이전트 흐름의 설계에서 최적화가 필요하다는 결론을 도출하였습니다.

- **Performance Highlights**: 연구 결과, 에이전트 시스템과 비에이전트 시스템 간의 솔루션 차이가 있었고, 특정 인스턴스는 오직 특정 시스템에서만 해결 가능한 것으로 나타났습니다. 추가적으로, 버그 재현이 버그 수정을 위한 중요한 요소라는 것을 강조하였으며, 에이전트 시스템의 설계를 개선하기 위한 여러 방안을 제시했습니다.



### FengWu-W2S: A deep learning model for seamless weather-to-subseasonal forecast of global atmospher (https://arxiv.org/abs/2411.10191)
Comments:
          23 pages,8 figures

- **What's New**: FengWu-Weather to Subseasonal (FengWu-W2S) 모델은 하나의 AI 모델을 기반으로 기상 예보와 기후 예보를 통합하여 Seamless Forecasting을 실현하는 새로운 접근법을 제시합니다.

- **Technical Details**: FengWu-W2S 모델은 FengWu 글로벌 기상 예보 모델을 기반으로 하며, 해양-대기-육상의 결합 구조(ocean-atmosphere-land coupling structure)와 다양한 변동성 전략(diverse perturbation strategy)을 통합하여 6시간마다 최대 42일간의 예보를 생성합니다. 이 모델은 자기회귀적(autoregressive) 방식으로 작동하여 예측합니다.

- **Performance Highlights**: FengWu-W2S는 3-6주까지 정확하게 대기 조건을 예측할 수 있으며, 이는 지구 표면 기온(global surface air temperature), 강수량(precipitation), 기저 높이(geopotential height) 및 Madden-Julian Oscillation (MJO)와 North Atlantic Oscillation (NAO)과 같은 계절 내 신호(intraseasonal signals)의 예측 능력을 향상시킵니다. 또한, 예측 오차 성장(forecast error growth)에 대한 탈락 실험(ablation experiments)을 통해 AI 기반 통합 시스템 개발의 잠재적인 경로를 발견했습니다.



### The Surprising Ineffectiveness of Pre-Trained Visual Representations for Model-Based Reinforcement Learning (https://arxiv.org/abs/2411.10175)
Comments:
          Published at the 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Project page: this https URL

- **What's New**: 본 논문에서는 모델 기반 강화 학습(Model-Based Reinforcement Learning, MBRL) 환경에서 다양한 사전 학습된 시각 표현(Pre-trained Visual Representations, PVRs)의 효과를 벤치마킹합니다. 이전의 연구는 PVR이 샘플 효율성과 일반화 능력을 향상시킨다고 보고했으나, MBRL에 대한 PVR의 잠재력은 거의 탐색되지 않았습니다. 이 연구를 통해 PVR이 MBRL의 성능에 미치는 영향을 분석합니다.

- **Technical Details**: 이 연구에서는 MBRL에서의 PVR의 데이터 효율성과 일반화 능력을 평가하며, PVR의 다양한 속성이 MBRL 에이전트의 성능에 미치는 영향을 조사합니다. MBRL에서는 특징을 선별하여 환경의 동력학 모델을 학습하고, CNN(Covolutional Neural Networks)을 사용하여 시각적 상태 표현을 활용합니다. 이 연구의 결과에 따르면, 현재의 PVR은 스크래치에서 학습한 표현보다 샘플 효율성이 더 뛰어나지 않으며, OOD(out-of-distribution) 설정에 대한 일반화 능력도 떨어지는 것으로 나타났습니다.

- **Performance Highlights**: 본 논문에서 수행한 실험 결과, 스크래치에서 학습한 표현을 사용하는 모델이 PVR을 사용하는 모델보다 더 우수한 성능을 보이는 경우가 많았습니다. 데이터 다양성과 네트워크 아키텍처가 OOD 일반화 성능에 가장 중요한 요소로 나타났습니다.



### A Hard-Label Cryptanalytic Extraction of Non-Fully Connected Deep Neural Networks using Side-Channel Attacks (https://arxiv.org/abs/2411.10174)
- **What's New**: 이번 연구에서는 비수치적 (non-fully connected) DNN을 대상으로 한 새로운 블랙박스 사이드 채널 공격 방법론을 제시하였습니다. 이는 기존의 1계층 신경망 뿐만 아니라 복잡한 아키텍처에서도 높은 충실도(fidelity)를 유지하며 DNN의 매개변수(weights)를 추출할 수 있습니다.

- **Technical Details**: 본 논문에서는 블랙박스 사이드 채널 공격과 크립토 분석 기반의 DNN 추출 방법을 통합하는 효율적인 프레임워크를 제안합니다. 새로운 방법론은 DNN의 출력이 아닌 사이드 채널 공격을 통해 중요한 포인트(critical points)를 추출하는 것으로, 하드 레이블(hard-label) 설정에서 높은 정확도를 제공합니다. 이 방법은 여러 활성화 함수에서 DNN을 분할할 수 있는 차별점이 있습니다.

- **Performance Highlights**: 제안된 프레임워크는 MobileNetv1과 다층 퍼셉트론(MLP) 아키텍처를 성공적으로 추출하였으며, 각각 88.4%와 93.2%의 높은 충실도를 기록하였습니다. 추가적으로, 추출된 모델을 이용하여 적대적 예제(adversarial examples)를 생성하고, 피해자의 모델에서 95.8%와 96.7%의 전이율을 달성하였습니다.



### Increasing the Accessibility of Causal Domain Knowledge via Causal Information Extraction Methods: A Case Study in the Semiconductor Manufacturing Industry (https://arxiv.org/abs/2411.10172)
Comments:
          17 pages, 2 figures

- **What's New**: 이 논문은 반도체 제조 산업의 실제 산업 문서에서 인과 정보(causal information)를 자동으로 추출하는 방법을 개발한 연구를 소개합니다. 연구에서는 단일 단계 시퀀스 태깅(Single-stage Sequence Tagging, SST)과 다단계 시퀀스 태깅(Multi-stage Sequence Tagging, MST) 방법 두 가지를 제안하며, FMEA(Failure Mode and Effects Analysis)와 발표 슬라이드를 포함한 기존 문서들을 통해 성능을 평가했습니다.

- **Technical Details**: 본 연구는 비정형(unstructured) 및 반정형(semi-structured) 문서에서 인과 도메인 지식을 자동으로 추출하기 위해, 트랜스포머(transformer) 기반의 언어 모델을 활용한 시퀀스 태깅(sequence tagging) 방법을 채택하였습니다. 또한, FMEA와 같은 반정형 문서에서의 인과 관계를 잘 추출할 수 있도록 설계되었습니다.

- **Performance Highlights**: 제안된 MST 방법은 산업 문서에서 인과 정보를 추출하는 데 있어 실용적인 응용에 적합하며, FMEA 문서에서 93%의 F1 점수를 기록하였습니다. 발표 슬라이드에서 추출된 텍스트에 대해서는 73%의 F1 점수를 달성했습니다. 이 결과는 해당 방법이 산업 설정에서의 데이터 분석과 의사결정에 도움을 줄 가능성이 있음을 나타냅니다.



### Imagine-2-Drive: High-Fidelity World Modeling in CARLA for Autonomous Vehicles (https://arxiv.org/abs/2411.10171)
Comments:
          Submitted to ICRA 2025

- **What's New**: 이번 연구에서는 Imagine-2-Drive라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 VISTAPlan과 DPA라는 두 가지 주요 구성 요소를 포함하고 있습니다. VISTAPlan은 높은 정밀도를 가진 월드 모델로서, 정확한 미래 예측을 위해 설계되었습니다. DPA는 다중 모드를 모델링할 수 있는 확산 기반 정책 액터입니다.

- **Technical Details**: Imagine-2-Drive 프레임워크는 주어진 현재 상태와 행동으로부터 미래 상태를 시뮬레이션합니다. VISTAPlan은 보상 및 할인 요소를 예측하는 추가 모듈을 통합하여 효과적인 계획과 의사결정을 지원합니다. DPA는 강화학습을 통해 다중 행동 모드를 모델링하며, DDPO를 사용하여 보상을 최대화하는 방향으로 학습됩니다.

- **Performance Highlights**: CARLA 시뮬레이터에서의 평가 결과, Imagine-2-Drive는 기존의 최고 성능 월드 모델에 비해 경로 완료율에서는 15%, 성공률에서는 20% 향상된 성능을 보였습니다.



### Causal Time-Series Synchronization for Multi-Dimensional Forecasting (https://arxiv.org/abs/2411.10152)
Comments:
          14 pages

- **What's New**: 이 논문에서는 프로세스 산업에서의 디지털 트윈(Digital Twins)을 위한 혁신적인 채널 종속(pre-training) 방법론을 제안하고 있습니다. 이는 다차원 시계열 데이터에 대한 예측 모델링의 복잡성을 극복하는 데 중점을 두고 있으며, 특히 동기화된 원인-결과(cause-effect) 쌍을 활용하여 훈련 샘플을 생성합니다.

- **Technical Details**: 제안된 방법은 (i) 데이터 기반 방법을 사용하여 고도로 지연된 원인 관계를 식별하고, (ii) 원인-결과 쌍을 동기화하여 채널 종속 훈련 샘플을 생성하며, (iii) 채널 종속 예측 채널의 효과성을 평가하는 것입니다. 또한, 선형 그랜저 인과법을 사용하여 인과 관계 그래프를 추출하고 다차원 시계열 데이터를 동기화된 원인-결과 쌍으로 나누는 방법을 설명합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 전통적인 훈련 방법에 비해 예측 정확도와 일반화 능력이 크게 향상됨을 보여줍니다. 미래의 다차원 데이터와 다양한 변수들에 대한 예측 성능 개선이 기대됩니다.



### Legal Evalutions and Challenges of Large Language Models (https://arxiv.org/abs/2411.10137)
- **What's New**: 이번 논문에서는 OpenAI의 o1 모델을 사례로 LLMs(대형 언어 모델)의 법률 적용 성능을 평가하는 법률 테스트 방법을 검토합니다. 현재의 LLM 기술이 법률 용어 이해 및 적용에서 어떤 장단점을 가지고 있는지를 분석합니다.

- **Technical Details**: LLMs는 딥 러닝 기술의 혁신으로 자연어 처리(NLP) 분야에서 비약적인 발전을 이루었으며, OpenAI의 GPT 시리즈와 같은 모델들이 기계 번역 및 질문-응답 등의 전통적인 NLP 작업에서 뛰어난 성능을 보여주고 있습니다. 이 논문에서는 영문과 중문 법률 사건을 체계적으로 테스트하고 평가합니다.

- **Performance Highlights**: 실험 결과는 LLM의 법률 적용에서의 잠재력과 한계를 강조하며, 법률 언어 해석 및 법적 추론의 정확성에 관련된 도전 과제가 드러납니다. 또한, 다양한 모델의 장단점을 포괄적으로 분석하여 AI의 법률 분야 적용에 대한 귀중한 통찰력을 제공합니다.



### Identifying Key Drivers of Heatwaves: A Novel Spatio-Temporal Framework for Extreme Event Detection (https://arxiv.org/abs/2411.10108)
Comments:
          28 pages, 10 figures, 4 tables

- **What's New**: 본 연구는 열파(Heatwaves, HW)와 같은 극단적인 기후 사건의 주요 원인을 식별하기 위한 일반적인 방법을 제시합니다. 새로운 프레임워크(STCO-FS)를 통해 클러스터링 알고리즘과 앙상블 진화 알고리즘을 결합하여 단기 HW 추진因素를 파악합니다.

- **Technical Details**: 제안된 프레임워크는 공간-시간 데이터(spatio-temporal data)를 분석하고, 비슷한 지리적 노드를 그룹화하여 차원을 축소하며, HW 발생과 예측 변수 간의 최적 시차를 식별하는 드라이버 선택을 합니다. 이 방법은 이탈리아의 아다 강 유역에서 HW 분석에 적용되었습니다.

- **Performance Highlights**: 본 연구는 지정된 지역에서 HW에 영향을 미치는 중요한 변수들을 효과적으로 식별하며, 향후 HW 기후 예측 가능성을 향상시킬 수 있는 잠재력을 지니고 있습니다.



### Multi-Task Adversarial Variational Autoencoder for Estimating Biological Brain Age with Multimodal Neuroimaging (https://arxiv.org/abs/2411.10100)
- **What's New**: 본 연구에서는 구조적 MRI(Structural MRI)와 기능적 MRI(Functional MRI) 데이터를 통합하여 뇌 나이를 예측하기 위한 멀티태스크 적대적 변이 자동 인코더(Multitask Adversarial Variational Autoencoder, M-AVAE)를 제안합니다. 이 모델은 공유 및 특정한 특성을 분리하여 더 나은 예측을 수행합니다.

- **Technical Details**: M-AVAE는 적대적 학습(Adversarial Learning)과 변이 자동 인코딩(Variational Autoencoding) 기능을 통합하여 구조화된 프레임워크를 제공합니다. 이 모델은 남녀의 뇌 노화 패턴을 포착하기 위해 성별 분류를 추가 과제로 통합합니다. 다양한 손실 함수(Loss Functions)를 사용해 최적화를 진행합니다.

- **Performance Highlights**: OpenBHB 데이터셋에서 평가한 결과, M-AVAE는 평균 절대 오차(Mean Absolute Error) 2.77세를 기록하며 기존의 방법들을 초월하여 뇌 나이 추정의 새로운 기준을 제시하고 있습니다.



### AI and the Future of Work in Africa White Paper (https://arxiv.org/abs/2411.10091)
- **What's New**: 이 백서(white paper)는 2023년 11월 나이로비에서 개최된 다분야 워크숍의 산출물입니다. Microsoft Research, NEPAD, Lelapa AI, 옥스퍼드 대학 등의 다양한 조직으로 구성된 팀이 주도를 하였으며, 아프리카의 미래 노동에 대한 Generative AI의 함의에 대해 토론했습니다.

- **Technical Details**: 워크숍은 네 가지 핵심 주제, 즉 거시경제적(Macroeconomic) 영향, 직업(Jobs), 기술(Skills) 및 노동시장(Labour Markets), 노동자의 관점(Workers' Perspectives)과 아프리카 중심의 AI 플랫폼(Africa-Centric AI Platforms)에 대해 중점적으로 논의하였습니다. 이 백서는 Generative AI의 현재 상태와 트렌드, 다양한 분야에서의 적용 사례, 채택 및 규제에 관련된 도전과 위험을 포괄적으로 제공합니다.

- **Performance Highlights**: 백서는 다양한 시각을 대표하여, 아프리카 전역에서 모든 이들을 위한 존엄한 미래 노동을 창출하기 위해 논의와 협력적 행동을 촉구하는 인사이트(insights)와 권고사항(recommendations)을 제시합니다.



### PFML: Self-Supervised Learning of Time-Series Data Without Representation Collaps (https://arxiv.org/abs/2411.10087)
- **What's New**: 이번 연구에서는 Masked Latents에서 기능 예측(PFML)이라는 새로운 self-supervised learning (SSL) 알고리즘을 제안하였습니다. 이 알고리즘은 시간 시계열 데이터에 적용 가능하며, 기존 알고리즘에서 자주 발생하는 representation collapse 문제를 피하였습니다.

- **Technical Details**: PFML 알고리즘은 마스킹된 embedding에 해당하는 입력 신호의 통계적 기능(functionals)을 예측하는 방식으로 작동합니다. 이는 마스킹되지 않은 embedding의 시퀀스를 기반으로 합니다. PFML은 복잡한 하이퍼파라미터 조정 없이 다양한 시간 시계열 데이터 도메인에 쉽게 적용될 수 있도록 설계되었습니다.

- **Performance Highlights**: PFML은 세 가지 서로 다른 데이터 모달리티(영아의 자세 및 움직임 분류, 음성 데이터 기반 감정 인식, EEG 데이터를 통한 수면 단계 분류)에서 복잡한 실제 분류 작업을 수행하여 효과성을 입증하였습니다. PFML은 기존 유사한 SSL 방법에 비해 우수한 성능을 보이고, 현재의 최전선 수준의 SSL 방법과도 경쟁력을 보여주었습니다.



### Real-Time AI-Driven People Tracking and Counting Using Overhead Cameras (https://arxiv.org/abs/2411.10072)
Comments:
          This paper is accepted to IEEE Region 10 conference (TENCON) 2024

- **What's New**: 본 연구에서는 스마트 빌딩과 지능형 교통 시스템에서 사람을 정확하게 세는 새로운 방법을 제안합니다. 이 방법은 새로운 객체 추적 알고리즘, 카운팅 알고리즘 및 조정된 객체 탐지 모델을 결합하여 실시간 사람 세기에서 97%의 정확도를 달성합니다. 또한, 저전력 엣지 컴퓨터에서 평균 20-27 FPS의 프레임 속도를 보입니다.

- **Technical Details**: 제안된 방법은 이미지를 통해 사람의 머리를 검출하고, 이 검출된 머리의 특징을 추출하여 이전 프레임과 비교하는 방식으로 작동합니다. 이 과정에서 SSD Mobilenet 모델을 사용하여 머리의 객체 탐지를 수행하며, MobileNetV2 모델을 사용하여 feature extraction을 최적화합니다. 두 가지 서로 다른 모델을 사용해 낮과 밤의 조건에서 입체적으로 탐지를 수행하고, 공통적인 장애물과의 혼동을 피하기 위해 모델을 미세 조정합니다.

- **Performance Highlights**: 본 연구의 방법론은 97%의 정확도를 기록하며, 프레임 속도는 평균 20-27 FPS로, 기존 방법에 비해 2% 개선된 성과를 보였습니다. 이는 다양한 환경 및 조명 조건에서도 신뢰성 있는 실시간 사람 세기를 가능하게 합니다.



### Evidential Federated Learning for Skin Lesion Image Classification (https://arxiv.org/abs/2411.10071)
Comments:
          Published as a conference paper at ICPR 2024

- **What's New**: FedEvPrompt는 evidential deep learning, prompt tuning, knowledge distillation을 통합하여 분산된 피부 병변 분류를 위한 새로운 연합 학습 접근 방식을 제시합니다. 이 방법은 기존의 모델 파라미터 공유 대신 attention maps를 통해 지식을 공유하여 개인 정보 보호를 강화합니다.

- **Technical Details**: FedEvPrompt는 frozen pre-trained Vision Transformer (ViT) 모델에 b-prompts(저수준 기본 시각적 지식) 및 t-prompts(작업 특정 지식)를 결합하여 학습합니다. 학습 과정은 round-based learning paradigm 내에서 이루어지며, 각 라운드는 로컬 모델 훈련 후 attention maps를 공유하는 방식입니다. 이러한 구조는 클래스 증거를 극대화하고, 데이터 불균형과 비독립적이고 동일하게 분포되지 않은(non-i.i.d.) 데이터 문제를 해결하는 데 기여합니다.

- **Performance Highlights**: ISIC2019 데이터셋을 사용한 실험 결과, FedEvPrompt는 기존 연합 학습 알고리즘 및 knowledge distillation 방법들에 비해 뛰어난 성능을 보여줍니다. 특히, 모델 파라미터를 공유하지 않고도 우수한 결과를 달성하였습니다.



### KuaiFormer: Transformer-Based Retrieval at Kuaishou (https://arxiv.org/abs/2411.10057)
- **What's New**: 본 논문에서는 Kuaishou의 대규모 콘텐츠 추천 시스템에 적용된 새로운 Transformer 기반 검색 프레임워크인 KuaiFormer를 소개합니다. KuaiFormer는 기존의 점수 추정 작업에서 Transformer 기반의 다음 행동 예측 패러다임으로 전환하여 검색 프로세스를 근본적으로 재정의합니다.

- **Technical Details**: KuaiFormer는 여러 쿼리 토큰을 도입하여 사용자의 다양한 관심사를 포착하고 연속된 항목 시퀀스에서 별개의 사용자 관심 표현을 추출합니다. 또한, 효율성을 높이기 위해 조정 가능한 항목 압축 메커니즘을 포함하고 있습니다. 이러한 방법들은 긴 시퀀스를 효과적으로 모델링할 수 있도록 설계되었습니다.

- **Performance Highlights**: KuaiFormer는 Kuaishou의 단기 동영상 서비스에서 +0.360%/-0.126%/-0.411%의 온라인 시청 시간 증가를 기여하며, 실시간 대규모 추천 시스템에 적합한 최초의 Pure Transformer 구조 기반 검색 모델로 평가받고 있습니다.



### Towards unearthing neglected climate innovations from scientific literature using Large Language Models (https://arxiv.org/abs/2411.10055)
Comments:
          10 pages. Accepted in the LatinX in AI workshop at NeurIPS 2024

- **What's New**: 이번 연구에서는 기후 변화 해결을 위한 혁신적인 기술들이 과학 문헌 내에 이미 존재하지만 활용되지 않고 있다는 가설을 세우고, 이를 최대한 활용하기 위해 OpenAlex 데이터베이스에서 수집한 논문들을 분석하였습니다.

- **Technical Details**: 연구팀은 기후 변화 완화 가능성, 기술 개발 단계, 배치 준비 상태 등 일곱 가지 차원에서 논문의 제목과 초록을 평가하기 위해 OpenAI의 LLM인 GPT4-o를 활용하였습니다. 이 과정에서 LLM의 결과를 인간 평가자와 비교하여 효과성을 분석하였습니다.

- **Performance Highlights**: 연구 결과, LLM 기반 모델이 인간 전문 지식을 보완하며, 기후 혁신을 조기에 발굴하는 데 유의미한 속도와 일관성을 제공할 수 있다는 것을 확인하였습니다. 이 연구는 미래의 기후 행동 전략 강화에 기여할 것으로 기대됩니다.



### Jal Anveshak: Prediction of fishing zones using fine-tuned LlaMa 2 (https://arxiv.org/abs/2411.10050)
- **What's New**: 최근 전 세계 및 인도 정부의 수산업 관련 데이터 모니터링 및 수집이 크게 발전하였습니다. 이러한 데이터의 유용성을 극대화하기 위해 저자들은 Jal Anveshak이라는 새로운 어플리케이션 프레임워크를 소개합니다.

- **Technical Details**: Jal Anveshak은 Dart와 Flutter로 작성된 어플리케이션 프레임워크로, Llama 2 기반의 Large Language Model을 활용합니다. 이 모델은 수산물 수확량 및 가용성에 관한 정부 데이터를 사전 처리(pre-processed)하고 증강(augmented)하여 세밀하게 조정된 것입니다.

- **Performance Highlights**: 이 프레임워크의 주요 목표는 인도의 어부들이 해안 지역에서 최대 수확량을 안전하게 얻고, 다국어 및 다중 모드(multimodal) 방식으로 어업 관련 질문을 해결하는 데 도움을 주는 것입니다.



### Physics-informed neural networks need a physicist to be accurate: the case of mass and heat transport in Fischer-Tropsch catalyst particles (https://arxiv.org/abs/2411.10048)
- **What's New**: 이 연구에서는 PINN(Physics-Informed Neural Networks)의 신뢰성 문제를 극복하는 방법을 제시합니다. 전통적인 신경망을 사용하여 다단계 시뮬레이션의 필요성을 줄이지만, 입력 매개변수 범위의 극단적인 끝에서의 신뢰성 문제 때문에 널리 채택되지 않습니다. 연구팀은 Fischer-Tropsch 합성과 관련된 비선형 반응-확산 방정식을 기반으로 PINN 아키텍처를 수정하여 이론적 지식의 전이 문제를 해결하는 방안을 제안하였습니다.

- **Technical Details**: 연구는 PINN을 사용하여 반응-확산 방정식을 해결하는 시스템을 문제로 다루었습니다. 연구팀은 전통적인 신경망의 정확도 검증 전략이 PINN의 불안정성을 초래할 수 있는 특성을 간과할 수 있음을 보여주고, 아키텍처를 수정하여 이론적 근거에 기반한 제약 조건을 자연스럽게 포함시키는 방법을 제안하였습니다. 또한, 수정된 PINN 아키텍처를 통해 기존의 유한차분 솔버와 결합하여 안정적인 수렴을 확보했습니다.

- **Performance Highlights**: 제안된 PINN과 개선된 수치적 방법을 결합함으로써, 시뮬레이션의 전반적인 안정성을 회복하면서 PINN의 속도 향상을 유지하는 데 성공했습니다. 이 연구는 화학 반응기 시뮬레이션에서 혼합 전송 방정식 해결기의 가능성을 논의하면서, PINN의 상대적 오류가 매우 낮은 성과를 달성했습니다. 연구 결과, PINN은 효율성을 크게 향상시키며, 유한차분 방정식 해결기의 안정성을 보장하기 위해 필수적임을 보여줍니다.



### Rethinking Normalization Strategies and Convolutional Kernels for Multimodal Image Fusion (https://arxiv.org/abs/2411.10036)
- **What's New**: 이번 논문은 기존의 멀티모달 이미지 융합(MMIF) 방법들이 자연 이미지 융합에 중점을 두고, 의료 이미지 융합의 고유한 특성을 간과하고 있다는 점을 지적합니다. 저자들은 IVIF(Infrared and Visible Image Fusion)와 MIF(Medical Image Fusion) 간의 주요 차이점을 분석하고, 연속적인 이미지 융합을 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 논문에서는 인스턴스 정규화(Instance Normalization)와 그룹 정규화(Group Normalization)를 혼합하여 샘플 독립성을 유지하고, 이미지 세부 정보를 증강하기 위한 새로운 전략을 제안합니다. 이와 함께 대형 커널 컨볼루션(Large Kernel Convolution)을 도입하여 수용 영역을 확장하고 이미지 세부 정보의 보존을 향상시킵니다. 또한, 다양한 스케일과 수용 영역에서 피처를 조정하는 다중 경로 적응 융합 모듈을 통해 정보 전송을 최적화합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안된 방법은 여러 융합 작업에서 최고 성능을 보였으며, 다운스트림 응용 프로그램에서도 유의미한 개선을 가져왔습니다. 특히 의료 영상의 시각화와 멀티모달 객체 탐지 및 의미 분할(semantic segmentation) 작업에서의 성능 향상이 컸습니다.



### VMID: A Multimodal Fusion LLM Framework for Detecting and Identifying Misinformation of Short Videos (https://arxiv.org/abs/2411.10032)
Comments:
          arXiv admin note: text overlap with arXiv:2211.10973 by other authors

- **What's New**: 본 논문은 기존의 단일 모달 기반 가짜 뉴스 탐지 방법의 한계를 극복하기 위해 다중 모달 정보를 활용한 새로운 가짜 뉴스 탐지 방법(VMID)을 제안합니다. 이 방법은 비디오 콘텐츠의 다층 분석을 통해 허위 정보를 식별할 수 있도록 설계되었습니다.

- **Technical Details**: VMID 프레임워크는 비디오에서 시각, 오디오, 텍스트 정보를 통합하여 LLM(대형 언어 모델)에 입력할 수 있는 통합 프롬프트를 생성합니다. 이를 통해 비디오의 메타데이터 및 소셜 컨텍스트 신호를 포함해 정확한 가짜 뉴스 탐지가 가능합니다.

- **Performance Highlights**: VMID는 실험을 통해 기존 모델 대비 약 9.87%에서 19.6% 향상된 매크로 F1 점수를 기록하였으며, 정확도에서는 90.93%를 달성하여 가장 우수한 기준 모델(SV-FEND)의 81.05%를 크게 초과하는 성과를 보였습니다.



### MOT\_FCG++: Enhanced Representation of Motion and Appearance Features (https://arxiv.org/abs/2411.10028)
Comments:
          12 pages, 7 figures

- **What's New**: 이번 논문은 multi-object tracking (MOT)에서 객체의 외관(appearance)과 공간적 특성(spatial features)을 효과적으로 표현하기 위한 새로운 접근법을 제안합니다. 기존 방법인 MOT_FCG를 기반으로 하여, 더 글로벌한 appearance 임베딩 이론과 정확한 공간 운동 정보를 제공하는 방법을 개선했습니다.

- **Technical Details**: 제안된 방법에는 Diagonal Modulated GIoU라는 새로운 spatial metric이 포함되어 있어 객체의 위치 및 형태 간의 관계를 더 정확하게 나타냅니다. 또한, 동적 appearance 표현을 통해 신뢰도 정보를 통합하여 경로의 appearance 특성을 더 강인하고 전반적으로 개선하였습니다. 성능 측정 기준으로 76.1 HOTA, 80.4 MOTA, 81.3 IDF1을 MOT17 검증 세트에서 달성했습니다.

- **Performance Highlights**: MOT17, MOT20 및 DanceTrack 데이터셋에서 뛰어난 성능을 보였으며, 특히 MOT17 데이터셋에서는 기존 상태를 초과하는 성능을 기록했습니다.



### MicroCrackAttentionNeXt: Advancing Microcrack Detection in Wave Field Analysis Using Deep Neural Networks through Feature Visualization (https://arxiv.org/abs/2411.10015)
- **What's New**: 본 논문에서는 MicroCrackAttentionNeXt라는 새로운 모델을 제안하여, 기존 SpAsE-Net을 개선하여 미세 크랙 감지에서의 성능을 향상시킵니다. 이 모델은 비대칭 인코더-디코더 구조를 사용하며, 중요한 크랙 패턴을 포착하는 데 집중하여 정확도를 높입니다.

- **Technical Details**: MicroCrackAttentionNeXt는 spatio-temporal (시공간) 데이터 분석을 통해 복잡한 크랙을 인식하도록 설계되었으며, 다양한 activation (활성화) 및 loss functions (손실 함수)의 영향을 분석합니다. 이 과정에서 Manifold Discovery Analysis (MDA)를 활용하여 최적의 활성화 함수를 선택하는 방법론을 제시합니다.

- **Performance Highlights**: 제안된 모델은 최적화된 아키텍처와 훈련 방법론을 통해 86.85%의 정확도를 달성하였습니다. 이런 성과는 기존의 비율이 낮은 데이터셋에서 클래스 불균형 문제를 효과적으로 해결한 결과입니다.



### DeepMedcast: A Deep Learning Method for Generating Intermediate Weather Forecasts among Multiple NWP Models (https://arxiv.org/abs/2411.10010)
Comments:
          12 pages, 8 figures

- **What's New**: DeepMedcast는 여러 NWP 모델 간의 중간 예측(medcast)을 생성하는 딥러닝 방법을 제안하며, 전통적인 평균 기반 접근방식과는 달리 기상 현상을 왜곡하지 않고 일관성 있는 예측을 제공합니다.

- **Technical Details**: DeepMedcast는 딥 뉴럴 네트워크(DNN)를 사용하여 두 NWP 출력 간의 중간 예측을 생성하며, 단일 모델의 두 예측 시점 간의 데이터를 입력으로 활용하여 중간 예측을 생성합니다. 이 접근 방식은 두 모델 간의 예측 오류의 영향을 받지 않도록 설계되었습니다.

- **Performance Highlights**: DeepMedcast는 기상 예측의 정확성과 설명 가능성(explicability)을 동시에 충족하여 신뢰할 수 있는 기상 경고 및 안내 발행에 기여할 수 있는 잠재력을 가지고 있습니다.



### Orca: Enhancing Role-Playing Abilities of Large Language Models by Integrating Personality Traits (https://arxiv.org/abs/2411.10006)
- **What's New**: Orca는 성격 특성을 통합하여 맞춤형 언어 모델(LLM)을 교육하고 데이터 처리하는 프레임워크입니다. 이 프레임워크는 사용자 성격 특성을 추론하는 단계를 포함하여, 데이터 증강, 데이터셋 구축 및 모델링 및 교육을 통해 LLM의 롤플레잉 능력을 향상시킵니다.

- **Technical Details**: Orca는 성격 기반 지침 프로밍(PCIP) 방법론과 성격 기반 지침 조정(PTIT 및 PSIT)을 활용하여 LLM의 훈련 데이터를 생성합니다. 이 과정에서는 500명 사용자로부터 수집된 200개의 소셜 미디어 게시물을 바탕으로 성격 특성을 유추합니다. OrcaBench라는 다각적 평가 기준을 구축하여 소셜 AI 캐릭터가 생성한 콘텐츠 품질을 측정합니다.

- **Performance Highlights**: 제안된 모델은 생성된 콘텐츠 품질을 평가하는 OrcaBench 기준에서 뛰어난 성능을 보여주었습니다. 이 모델은 성격 특성을 효과적으로 인식하고 롤플레잉 능력을 현저히 개선하는 데 성공했습니다.



### EyeDiff: text-to-image diffusion model improves rare eye disease diagnosis (https://arxiv.org/abs/2411.10004)
Comments:
          28 pages, 2 figures

- **What's New**: EyeDiff는 자연어 프롬프트를 기반으로 다중 모달(multi-modal) 안과 이미지를 생성할 수 있는 텍스트-투-이미지 모델로, 희귀 질환 진단에 적합합니다.

- **Technical Details**: EyeDiff는 8개의 대규모 데이터셋을 기반으로 훈련되었으며, 고급 잠재 확산(latent diffusion) 모델을 사용하여 14개의 안과 이미지 모달리티와 80종 이상의 안구 질환을 다룹니다. 이 모델은 10개의 다국적 외부 데이터셋에 적응되어 있습니다.

- **Performance Highlights**: 생성된 이미지는 중요한 병변(lesional) 특성을 정확하게 포착하여, 텍스트 프롬프트와 높은 일치를 이루며, 소수 클래스 및 희귀 안구 질환 탐지의 정확성을 크게 향상시킵니다. 이는 기존의 오버샘플링(oversampling) 방법보다 데이터 불균형 문제를 효과적으로 해결합니다.



### DuSEGO: Dual Second-order Equivariant Graph Ordinary Differential Equation (https://arxiv.org/abs/2411.10000)
- **What's New**: 본 논문은 이차 동차 미분 방정식(	extbf{Dual Second-order Equivariant Graph Ordinary Differential Equation, DuSEGO})을 제안하여 기존의 등변 GNN 모델의 한계를 극복하고자 합니다. DuSEGO는 먼저 1차 정보를 넘어서 2차 동역학적 시스템을 모델링할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: DuSEGO는 그래프 임베딩과 노드 좌표에 동시 적용되는 이차 동변 그래프 ordinary differential equations (Graph ODEs)를 활용하여, 과도한 평활화(over-smoothing)와 그래디언트 폭주나 소실 문제를 완화합니다. 이론적으로 DuSEGO가 등변 속성을 유지함을 증명하고, 특징 표현 및 좌표 업데이트에서의 과도한 평활화 문제를 효과적으로 alleviates 합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 extensive experiments를 통해 DuSEGO가 기존 방법들보다 우수한 성능을 보임을 입증하였습니다. 이는 DuSEGO가 GNNs의 표현력과 깊이 훈련을 가능하게 함을 의미합니다.



### Building 6G Radio Foundation Models with Transformer Architectures (https://arxiv.org/abs/2411.09996)
- **What's New**: 이 논문에서는 라디오 신호 처리를 위한 새로운 접근법으로서 Vision Transformer (ViT) 모델을 제안하고, Masked Spectrogram Modeling (MSM) 기법을 활용한 자가 지도 학습 방식으로 이 모델을 사전 학습시키는 방법을 보여줍니다. 이를 통해 무선 환경에서 필요한 모델의 일반화 능력을 향상시키는 데 기여합니다.

- **Technical Details**: 이 연구에서는 ViT 아키텍처를 이용하여 주파수 스펙트로그램 학습을 수행합니다. MSM 접근법을 통해 스펙트로그램 이미지의 일부 패치를 마스킹하고, 남은 패치에서 누락된 데이터를 복원하는 방식으로 모델을 훈련합니다. 이는 전통적인 오토인코더와 간단히 유사하지만, 전체 패치 대신 마스킹된 패치만 복원하는 데 중점을 두고 있습니다. 이 과정을 통해 모델이 데이터의 전반적인 구조를 더 잘 이해하도록 유도합니다.

- **Performance Highlights**: 실험 결과, 사전 학습된 ViT 모델은 4배 큰 모델보다 스펙트로그램 분할 작업에서 더 나은 성능을 보이고 훈련 시간도 대폭 줄였습니다. 또한, Channel State Information (CSI) 기반 인간 활동 감지 작업에서도 경쟁력 있는 성과를 달성하여, 6G 네트워크를 위한 확장 가능한 기본 모델 개발의 가능성을 시사합니다.



### Unlocking Transfer Learning for Open-World Few-Shot Recognition (https://arxiv.org/abs/2411.09986)
- **What's New**: 본 논문은 Few-Shot Open-Set Recognition (FSOSR)을 위한 새로운 두 단계 학습 방법인 OAL-OFL(Open-set Aware Learning - Open-set Free Learning)을 제안하며, 일반적인 transfer learning 방식이 FSOSR에 효과적으로 적용되지 못하는 문제를 해결합니다.

- **Technical Details**: OAL-OFL 방법은 two-stage 방식으로 구성됩니다. 첫 번째 단계인 open-set aware meta-learning에서는 유용한 metric space를 설정하여 후속 단계의 기초가 됩니다. 두 번째 단계인 open-set free transfer learning에서는 메타 학습을 통해 얻은 파라미터로 모델을 초기화하고, open-set 예시의 부재를 해결하기 위해 두 가지 샘플링 전략을 도입합니다.

- **Performance Highlights**: 제안된 방법은 miniImageNet 및 tieredImageNet 데이터셋에서 state-of-the-art (SOTA) 성능을 달성하였으며, 추가적인 학습 비용은 1.5%에 불과합니다. 이로 인해 OAL-OFL은 FSOSR의 일반화 능력 및 성능을 크게 향상시킵니다.



### Large Language Models as User-Agents for Evaluating Task-Oriented-Dialogue Systems (https://arxiv.org/abs/2411.09972)
- **What's New**: 이 연구는 Task-oriented Dialogue (TOD) 시스템을 효과적으로 평가하기 위해 대화형 사용자 시뮬레이터를 개발하고, 이를 대형 언어 모델(LLMs)을 활용하여 자동화된 평가 프레임워크로 구현합니다. 기존의 데이터셋 기반 평가 방식이 가지고 있는 한계를 극복함으로써, 보다 다양하고 현실적인 대화 시나리오를 평가할 수 있습니다.

- **Technical Details**: 작업에 적합한 사용자 시뮬레이터를 구축하기 위해 LLM을 활용하여 초기 사용자 목표에 따라 LLM에게 프롬프트를 제공하고, 대화 흐름을 관리합니다. 사용하는 프롬프트 전략에는 기본적인 지침을 활용한 Vanilla Prompt, 추론 단계를 포함한 Thought Prompt, 사용자 상태를 추적하여 회화를 종료하지 않도록 돕는 User State Tracking Prompt가 있습니다.

- **Performance Highlights**: 제안된 사용자 시뮬레이터를 통해 TOD 시스템의 성과를 다양성과 과업 완수 지표로 평가한 결과, 더 나은 프롬프트 사용시 성과가 개선됨을 보여주었습니다. 이를 통해 지금까지의 전통적인 평가 방법의 한계를 극복하고, 실제와 유사한 대화 상호작용을 반영한 평가 프로세스를 강화하였습니다.



### Steering AI-Driven Personalization of Scientific Text for General Audiences (https://arxiv.org/abs/2411.09969)
Comments:
          23 pages, 5 figures, 1 table

- **What's New**: 본 연구에서는 AI 기반 도구인 TranSlider를 통해 과학 텍스트의 개인 맞춤형 번역을 생성하고, 사용자 프로필(예: 취미, 위치, 교육 수준)에 따라 맞춤형으로 제공하는 방식을 제안합니다.

- **Technical Details**: TranSlider는 사용자가 원하는 개인화 정도(0에서 100까지)를 조정할 수 있는 상호작용형 슬라이더를 특징으로 하며, 이를 통해 사용자에게 적합한 비유를 제공하여 과학 콘텐츠를 이해하도록 돕습니다. 대규모 언어 모델(LLMs)을 활용하여 다양한 사용자 맥락에 맞춰 여러 개인화된 비유를 생성합니다.

- **Performance Highlights**: 연구 참가자들은 또래적이고 맥락에 맞는 번역을 선호했으며, 여러 번역을 통해 과학 콘텐츠에 대한 이해력이 향상되었다고 보고했습니다. 특히, 사용자는 툴의 유용성을 높이 평가했으나 AI 생성 콘텐츠의 신뢰성에 대해 조심스럽게 접근했습니다.



### Seeing Clearly by Layer Two: Enhancing Attention Heads to Alleviate Hallucination in LVLMs (https://arxiv.org/abs/2411.09968)
- **What's New**: 이번 논문은 멀티모달 대형 언어 모델(MLLM)에서 이미지 토큰과 환각(hallucination) 현상 간의 관계를 분석하고, 낮은 계층의 주의(head) 밀도를 높여 환각 문제를 완화하는 새로운 방법인 EAH(Enhancing Attention Heads)를 제안합니다.

- **Technical Details**: 연구는 주의 매트릭스에서 시각적 토큰의 밀집된 주의 싱크(vision sink)가 환각 발생 여부와 연결되어 있음을 발견했습니다. EAH는 낮은 계층에서 시각적 싱크가 가장 높은 주의를 찾고, 이를 사용하여 다른 주의 헤드를 강화하는 훈련이 필요 없는 방법입니다. 논문에서는 LLaVA1.5, Minigpt4 등 여러 모델을 사용한 실험을 통해 EAH의 효과를 입증합니다.

- **Performance Highlights**: EAH를 적용한 결과, 다양한 멀티모달 대형 언어 모델에서 환각 발생률이 감소하였으며, 시각적 싱크 헤드의 밀도가 높을수록 환각 문제의 완화와 음의 관계가 있다는 점을 강조했습니다.



### Instruction-Guided Editing Controls for Images and Multimedia: A Survey in LLM era (https://arxiv.org/abs/2411.09955)
- **What's New**: 이번 논문은 LLM(large language models) 및 멀티모달(Multimodal) 모델이 시각적 수정 작업을 간소화하는 방법에 대한 개요를 제공합니다. 사용자는 복잡한 기술적 지식 없이 자연어로 지시를 통해 이미지 및 비디오 편집을 수행할 수 있습니다.

- **Technical Details**: 논문은 생성적 적대 신경망(generative adversarial networks)과 확산 모델(diffusion models)에서부터 시작하여, MLLM(multimodal large language models)과 같은 최첨단 기술을 통합한 방법을 다룹니다. 이러한 모델들은 사용자가 자연어로 간단한 명령을 주면 시각적 콘텐츠에 대한 정밀한 수정을 할 수 있도록 지원합니다.

- **Performance Highlights**: LLM 및 MLLM의 발전은 패션, 3D 장면 조작 및 비디오 합성 등 다양한 분야에서 더 많은 접근성을 제공하여 사용자 경험을 향상시켰습니다. 이 논문은 LLM 기반 편집이 산업 전반에 걸쳐 강력한 도구로 자리 잡을 수 있음을 강조하고, 사용자 친화적인 편집 툴의 필요성을 제시합니다.



### GGAvatar: Reconstructing Garment-Separated 3D Gaussian Splatting Avatars from Monocular Video (https://arxiv.org/abs/2411.09952)
Comments:
          MMAsia'24 Accepted

- **What's New**: 이번 논문에서는 GGAvatar (Garment-separated 3D Gaussian Splatting Avatar)라는 새로운 아바타 모델이 소개되며, 이는 단일 모노큘러 비디오를 이용하여 의류와 신체를 효과적으로 분리하여 사실적인 인간 모델을 생성합니다.

- **Technical Details**: GGAvatar는 매개변수화된 템플릿을 사용하고 독특한 단계적 훈련을 통해 의류와 신체 간의 분리를 달성합니다. 또한 3D Gaussian Splatting(3DGS) 기법을 활용하여 고품질 렌더링을 수행하며, 이러한 과정에서 포인트 세트의 교차를 방지하는 강력한 훈련 모듈을 적용합니다.

- **Performance Highlights**: GGAvatar는 People Snapshot Dataset과 ZJU Mocap Dataset에서 기존의 NeRF 기반 모델에 비해 수백 배 빠른 훈련 속도와 뛰어난 복원 품질을 보여주며, 의류 편집 및 색상 변경과 같은 다양한 응용 분야에서 활용할 수 있는 가능성을 가집니다.



### TEESlice: Protecting Sensitive Neural Network Models in Trusted Execution Environments When Attackers have Pre-Trained Models (https://arxiv.org/abs/2411.09945)
Comments:
          Accepted by TOSEM. Extended version of the S&P24 paper (arXiv:2310.07152)

- **What's New**: 본 논문은 기존의 TSDP 방식이 지식이 있는 적대자에게 충분히 안전하지 않음을 밝혔다. 이를 해결하기 위해 모델 학습 전 파티셔닝 전략을 제안하여 개인 정보가 민감한 가중치와 그렇지 않은 구성 요소를 효과적으로 분리할 수 있음을 보여준다.

- **Technical Details**: 제안된 방식은 DNN 모델의 개인 정보가 민감한 가중치를 TEE 안에서 보호하고, 덜 민감한 가중치를 GPU로 오프로드하는 방법이다. 이를 통해 기존 방법들보다 10배 낮은 계산 비용으로 전체 모델 보호를 제공한다.

- **Performance Highlights**: 대규모 언어 모델에 대해서도 적용 가능하며, 개인 정보 기능을 경량화하여 전체 모델 보호 수준을 달성할 수 있음을 보여준다.



### JRadiEvo: A Japanese Radiology Report Generation Model Enhanced by Evolutionary Optimization of Model Merging (https://arxiv.org/abs/2411.09933)
Comments:
          Accepted by NeurIPS'24 Workshop on AIM-FM: Advancements In Medical Foundation Models: Explainability, Robustness, Security, and Beyond

- **What's New**: 이번 논문은 비의료 비전-언어 모델을 비영어 의료 텍스트 생성에 확장하기 위한 첫 번째 시도로, 진화 최적화(evolutionary optimization)를 통한 모델 병합(model merging) 기법을 사용하여 일본어(X-ray) 방사선 보고서 생성 모델(JRadiEvo)을 제안합니다.

- **Technical Details**: JRadiEvo는 비의료 비전-언어 모델, 의료 텍스트-투-텍스트 모델 및 일본어 텍스트-투-텍스트 모델을 진화 알고리즘을 통해 병합하여 개발되었습니다. 이 모델은 50개의 번역 샘플만으로 정확한 일본어 보고서를 생성할 수 있으며, 80억 개의 파라미터를 가진 경량 모델로, 병원 내에서 정상적으로 배포 가능하여 개인 정보 보호 문제를 해결합니다.

- **Performance Highlights**: JRadiEvo는 제한된 데이터(50개 사례)만을 사용하여 최근 연구의 최신 모델들보다 우수한 성능을 보여주었습니다. 이를 통해 모델의 피드백을 제거하여 학습 과정에서 비효율을 줄이고, 개인 정보를 보호해야 하는 의료 환경에서도 실용적인 응용이 가능하다는 점을 강조합니다.



### Motion-Grounded Video Reasoning: Understanding and Perceiving Motion at Pixel Lev (https://arxiv.org/abs/2411.09921)
- **What's New**: 이 논문에서는 Motion-Grounded Video Reasoning이라는 새로운 모션 이해 과제를 제시합니다. 이 과제는 입력 질문에 따라 시각적 답변(비디오 세그멘테이션 마스크)을 생성해야 하며, 이를 위해 암묵적인 시공간적(spatiotemporal) 추론과 그라운딩(grounding)이 필요합니다. 신규 작업인 GROUNDMORE 데이터셋을 수집하였으며, 이는 1,715개의 비디오 클립과 249,000개의 객체 마스크로 구성되어 있습니다.

- **Technical Details**: GROUNDMORE 데이터셋은 4가지 질문 유형(원인, 순차, 반사실, 설명)을 포함하도록 디자인되었으며, MORA(모션-그라운디드 비디오 추론 보조 모델)를 도입하여 Multimodal LLM의 멀티모달 추론 능력과 SAM의 픽셀 수준 인식 능력을 통합하였습니다. MORA는 GROUNDMORE에서 기존 비주얼 그라운딩 모델보다 평균 21.5% 더 뛰어난 성능을 보였습니다.

- **Performance Highlights**: MORA 모델은 GROUNDMORE 데이터셋에서 시공간적 그라운딩 및 추론 과제를 수행하며, 복잡한 모션 관련 비디오 추론, 시간 인식, 픽셀 수준 이해의 챌린지를 해결합니다. 이 새로운 작업은 비디오 추론 세그멘테이션을 통한 강력하고 일반화 가능한 모션 이해의 발전을 위한 길을 열 것으로 기대합니다.



### Statistical Analysis of Policy Space Compression Problem (https://arxiv.org/abs/2411.09900)
- **What's New**: 이 연구는 정책 공간 압축(policy space compression)을 통해 샘플 효율성을 높이는 방법을 제시합니다. 기존의 정책 탐색 방법들은 방대한 정책 공간 탐색의 비효율성 문제를 해결하기 위해, 샘플링 수를 최적화하는 데 필요한 샘플 수를 정량적으로 분석합니다.

- **Technical Details**: 정책 압축을 달성하기 위해, Rényi divergence를 활용하여 진정한 정책 분포와 추정된 정책 분포 간의 유사성을 측정하고, 오류 경계를 설정하여 효과적인 근사를 보장합니다. $l_1$ 노름을 사용하여 분석을 단순화하고, 모델 기반(model-based) 및 모델 없는(model-free) 설정에 대한 샘플 수 요건을 결정합니다.

- **Performance Highlights**: 이 연구는 $l_1$ 노름과 Rényi divergence에서 얻은 오류 경계를 연관시켜, 정책 공간의 정점 근처와 중앙의 정책 간의 차이를 파악하여 필요한 샘플 크기에 대한 하한 및 상한을 제시합니다.



### Off-Dynamics Reinforcement Learning via Domain Adaptation and Reward Augmented Imitation (https://arxiv.org/abs/2411.09891)
Comments:
          Published at Neurips 2024

- **What's New**: 본 연구에서는 동적 변화가 있는 목표 영역(target domain)에서 정책(policy)을 성공적으로 적용하기 위한 새로운 접근 방식인 Domain Adaptation and Reward Augmented Imitation Learning (DARAIL)을 제안합니다.

- **Technical Details**: DARAIL은 보상 수정(reward modification)을 활용하여 영역 적응(domain adaptation)을 수행하며, 생성적 적대적 모방 학습(generative adversarial imitation learning)에서 관찰 기반으로 하는 프레임워크를 따릅니다. 이 과정에서 보상 증대 추정치(reward augmented estimator)를 정책 최적화(policy optimization) 단계에 적용합니다.

- **Performance Highlights**: 실험 결과, 제안하는 DARAIL 방법은 순수한 보상 수정 방법보다 우수한 성능을 보였으며, 벤치마크 오프-다이내믹스 환경에서 다른 기준선(baselines)보다도 뛰어난 결과를 기록했습니다.



### InterFormer: Towards Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction (https://arxiv.org/abs/2411.09852)
Comments:
          10 pages, 6 figures

- **What's New**: 이 논문에서는 Click-through rate (CTR) 예측을 위한 새로운 모듈 InterFormer를 제안하고 있습니다. InterFormer는 이질적인 정보 상호작용을 상호 유익한 방식으로 학습하여 CTR 예측의 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: InterFormer 모듈은 두 가지 주요 아이디어로 구성됩니다. 첫째, 서로 다른 모드 간에 양방향 정보 흐름을 가능하게 하여 글로벌 및 시퀀스 학습을 혼합 방식으로 수행합니다. 둘째, 정보를 과도하게 집계하지 않기 위해 각 데이터 모드에서 완전한 정보를 유지하고, 효과적인 정보 선택 및 요약을 위한 별도의 브리징 아치를 사용합니다.

- **Performance Highlights**: InterFormer는 세 개의 공개 데이터셋과 대규모 산업 데이터셋에서 최첨단 성능을 달성하였습니다. 논문에서 언급한 바에 따르면 AUC 기준으로 최대 0.14% 개선과 내부 대규모 데이터셋에서 0.15%의 Normalized Entropy (NE) 이득을 달성했습니다.



### Enhancing Diffusion Posterior Sampling for Inverse Problems by Integrating Crafted Measurements (https://arxiv.org/abs/2411.09850)
- **What's New**: 본 논문에서는 새로운 diffusion posterior sampling 방법인 DPS-CM을 제안합니다. 이 방법은 깨끗한 이미지 대신 잡음이 있는 측정치에서 log posterior gradient를 형성하여 역과정을 개선하고, Crafted Measurement를 통합하여 posterior estimate를 생성합니다.

- **Technical Details**: DPS-CM은 문제가 발생할 수 있는 누적 posterior estimate 오류로 인한 diffusion prior와의 불일치를 완화하는 것을 목표로 합니다. 이 방법은 Gaussian deblurring, super-resolution, inpainting 및 Poisson noise와 같은 다양한 noisy inverse 문제를 해결하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기존 접근 방식에 비해 일반적이고 noisy inverse 문제 해결능력을 획기적으로 향상시킴을 입증했습니다.



### Self-Supervised Radio Pre-training: Toward Foundational Models for Spectrogram Learning (https://arxiv.org/abs/2411.09849)
- **What's New**: 이번 연구에서 제안한 Masked Spectrogram Modeling(MSM)은 라디오 신호에 대한 자가 지도 학습(self-supervised learning) 기술로, 기초 딥 러닝 모델을 사전 훈련하기 위한 혁신적인 접근 방식을 제공합니다. 이 모델은 경량화된 작업 흐름으로 신속하게 개발되고, 훈련 비용을 절감합니다.

- **Technical Details**: 기초 모델은 대규모 비라벨 데이터셋을 통해 사전 훈련되어 다양한 다운스트림 작업에 맞게 조정될 수 있습니다. 본 연구에서는 Convolutional LSTM 아키텍처를 사용하여 공간-시간적(spatio-temporal) 처리를 효과적으로 수행하고, 실제로 수집된 라디오 데이터셋을 기반으로 MSM을 활용하여 주파수 예측(spectrum forecasting) 및 세분화(segmentation) 두 가지 다운스트림 작업을 수행합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 주파수 예측 및 세분화 정확도에서 경쟁력 있는 성능을 확인하였으며, 이는 무선 신호 처리에 대한 기초 모델의 유효성을 입증합니다. 이 연구는 AI의 광범위한 사용을 촉진하고, 안정적인 네트워크 성능과 서비스 제공에 기여할 것으로 기대됩니다.



### Deep Autoencoders for Unsupervised Anomaly Detection in Wildfire Prediction (https://arxiv.org/abs/2411.09844)
Comments:
          33 pages, 18 figure, 16 tables. To appear in Earth and Space Science

- **What's New**: 이 연구는 고전적인 지도학습(supervised learning) 및 비지도학습(unsupervised learning) 접근법과 달리, 비지도학습을 활용하여 wildfire 예측의 격차를 해소하고자 하였습니다. 특히, autoencoder 및 clustering 기법을 사용하여 이상 탐지(anomaly detection)에 초점을 맞추었습니다.

- **Technical Details**: 연구는 2005년부터 2021년까지의 호주의 역사적 날씨 데이터와 Normalised Difference Vegetation Index (NDVI) 데이터를 이용하여 두 가지 비지도 방법을 분석했습니다. 첫 번째 방법은 Deep Autoencoder를 통해 잠재 특징(latent features)을 얻고, 이를 anomaly detection을 위해 isolation forest, local outlier factor, 및 one-class SVM에 입력하는 것이었습니다. 두 번째 방법은 입력 데이터를 재구성하여 재구성 오류를 통해 이상을 식별하는 것입니다. Long Short-Term Memory (LSTM) autoencoder와 fully connected (FC) autoencoder가 사용되었으며, FC autoencoder가 0.71의 정확도(accuracy), 0.74의 F1 점수(F1-score), 그리고 0.42의 MCC를 기록했습니다.

- **Performance Highlights**: 이 연구의 결과는 비지도학습 방법이 ground truth가 없는 상황에서 효과적으로 wildfires를 예측할 수 있음을 보여줍니다. FC autoencoder는 비교 대상 모델을 초월하여 좋은 성능을 보였고, 이로 인해 비지도학습 기법의 실용성에 대한 강력한 근거가 마련되었습니다.



### Real-time Adapting Routing (RAR): Improving Efficiency Through Continuous Learning in Software Powered by Layered Foundation Models (https://arxiv.org/abs/2411.09837)
- **What's New**: 이 논문에서는 Real-time Adaptive Routing (RAR)이라는 새로운 접근 방식을 제안하며, 이를 통해 Foundation Model (FM)의 라우팅 결정을 지속적으로 적응시키는 방법을 소개합니다. 기존의 라우팅 모델들과 달리, 이 방법은 비용이 많이 드는 더 강력한 FM에 대한 의존도를 낮추는 것을 목표로 합니다.

- **Technical Details**: Real-time Adaptive Routing (RAR) 접근 방식은 guided in-context learning을 활용하여 더 약한 FM의 능력을 향상시키고, 요청을 FM에 따라 효과적으로 라우팅합니다. 기존의 모델들은 최적의 라우팅 결정을 위해 정교하게 준비된 데이터를 학습하는 데 의존하고 있으며, 업데이트 시 복잡한 계산을 요구합니다.

- **Performance Highlights**: 우리의 접근 방식을 다양한 MMLU 벤치마크의 하위 집합에서 평가한 결과, 컴퓨팅 비용이 높은 모델에 대한 요청을 50.2% 줄이는 동시에 응답 품질의 약 90.5%를 유지했습니다. 또한, 강력한 모델에서 생성된 가이드는 특정 도메인 내에서 일반화가 이루어져, 단독으로 약한 FM을 사용할 때보다 더 나은 응답 품질을 보여주었습니다.



### A Benchmark for Long-Form Medical Question Answering (https://arxiv.org/abs/2411.09834)
Comments:
          AIM-FM: Advancements in Medical Foundation Models Workshop, 38th Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문에서는 장기적인 의료 질문 응답(QA)에 대한 대규모 언어 모델(LLMs)을 평가하기 위한 신뢰할 수 있는 벤치마크가 부족하다는 문제를 다루고 있습니다. 기존 평가 벤치마크는 자동 메트릭스 및 객관식 질문에 중점을 두고 있어, 실제 임상 응용에서의 복잡성을 충분히 반영하지 못합니다. 새로 개발된 벤치마크는 실제 소비자의 의료 질문을 포함하며 의료 전문가의 주석이 달린 장기 답변 평가가 포함되어 있습니다.

- **Technical Details**: 연구팀은 Lavita Medical AI Assist 플랫폼에서 수집된 사용자 질문을 통해 실제 소비자 의료 질문의 데이터셋을 구축했습니다. 총 4,271개의 질문이 수집되었고, 인간 주석가와 GPT-4를 이용해 의료 관련 질문을 검토하였으며, 최종적으로 1,298개의 고품질 의료 질문을 확보했습니다. 이 연구는 다양한 개방형 및 폐쇄형 LLM의 응답을 기반으로 정확성, 유용성, 해로운 정보 여부 및 편향을 평가했습니다.

- **Performance Highlights**: 프리미너리 결과는 개방형 LLM이 폐쇄형 모델보다 의료 QA에서 강력한 잠재력을 지니고 있음을 보여주었습니다. 인간 평가자와 LLM의 평가 결과 간의 일치도를 분석하였으며, 개방형 LLM이 높은 정확성을 보임을 확인했습니다.



### A Self-Supervised Model for Multi-modal Stroke Risk Prediction (https://arxiv.org/abs/2411.09822)
Comments:
          Accepted as oral paper at AIM-FM workshop, Neurips 2024

- **What's New**: 이 연구는 3D 뇌 영상, 임상 데이터 및 이미지 기반 특징을 통합하여 발병 전 뇌졸중 위험 예측을 개선하는 자기 지도(self-supervised) 다중 모달 프레임워크를 제안합니다. 이는 다양한 임상 데이터 모달리티를 결합하여 예측 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: 모델은 UK Biobank에서 수집된 데이터로 훈련되며, 구조적 뇌 MRI와 임상 데이터를 포함합니다. 대조적 학습(constrastive learning) 프레임워크를 기반으로 하여 이미지와 표 데이터를 결합하는 매칭 모듈을 사용하여 모달리티 간의 정보를 정렬합니다. 또한, 뇌 영상 데이터와 임상 특성 사이의 상호작용을 촉진하기 위해 CLIP 손실(clips loss)과 이미지-표 매칭 손실(image-tabular matching loss)을 통합합니다.

- **Performance Highlights**: 제안된 모델은 ROA-AUC에서 자기 지도 표준 방법보다 2.6% 더 우수하며, 균형 정확도(balanced accuracy)에서는 7.6% 향상된 성능을 보여줍니다. 이 모델의 응답 가능한 도구를 통해 이미지와 표 데이터를 더 잘 통합하고 해석 가능한 기능을 제공하여 임시로 뇌졸중 관련 뇌 병리와 일치하는 시각적 활성화 맵을 생성했습니다.



### WelQrate: Defining the Gold Standard in Small Molecule Drug Discovery Benchmarking (https://arxiv.org/abs/2411.09820)
Comments:
          * denotes equal contribution

- **What's New**: 이번 논문에서는 작고 분자(molecule) 약물 발견(drug discovery) 벤치마킹(benchmarking) 을 위한 새로운 기준선인 WelQrate를 제안합니다. 이 기준선 구축을 통해 AI 커뮤니티가 약물 발견에 기여할 수 있는 효과적인 평가 프레임워크를 세우고자 합니다.

- **Technical Details**: WelQrate는 5가지 치료적(target) 대상 클래스에 걸쳐 9개의 데이터셋을 포함하는 데이터셋 수집 작업과, 표준화된 모델 평가 프레임워크를 포함합니다. 이 평가 프레임워크는 고품질 데이터셋, 특성화(featurization), 3D 형태 생성(conformation generation), 평가 메트릭스(metrics), 데이터 분할(data splits)을 고려하여 실세계 가상 스크리닝을 수행하는 약물 발견 전문가들에게 신뢰할 수 있는 벤치마킹을 제공합니다. 또한, PAINS(패닉을 유발하는 화합물) 필터링과 같은 철저한 데이터 전처리(preprocessing)을 포함하여 데이터의 질을 보증합니다.

- **Performance Highlights**: WelQrate 데이터셋 수집을 통한 다양한 연구 질문을 통해 모델 성능을 평가하였고, 다양한 모델, 데이터셋 품질, 특성화 방법 및 데이터 분할 전략의 영향을 탐색했습니다. 이 결과를 바탕으로 WelQrate를 작고 분자 약물 발견 벤치마킹의 금본위로 채택할 것을 권장합니다.



### Evaluating Loss Landscapes from a Topology Perspectiv (https://arxiv.org/abs/2411.09807)
- **What's New**: 신경망의 손실 구조를 정량적으로 분석하기 위해 Topological Data Analysis(TDA)을 활용하였습니다. 기존의 손실 경관 시각화에서 벗어나, 손실 경관의 위상적 특성을 정량화하고 비교하여 신경망의 성능과 학습 동역학에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 본 연구는 손실 경관을 TDA의 merge tree와 persistence diagram을 통해 분석합니다. 손실 경관의 형태를 정량화하기 위해 saddle point와 average persistence를 측정하고, Hessian과 관련된 지표와 결과를 비교합니다. 특히, ResNet과 물리 기반 신경망 같은 이미지 패턴 인식 모델을 사용하여 실험을 진행하였습니다.

- **Performance Highlights**: ResNet-20 모델에서 residual connections을 제거했을 때 손실 경관의 구조가 변화하였고, 이는 모델의 성능에도 영향을 미쳤습니다. Residual connection이 없는 경우 평균 정확도가 90%로, 있는 경우 92%로 나타났으며, 손실 경관의 복잡성이 증가하는 것으로 나타났습니다.



### AI-Driven Human-Autonomy Teaming in Tactical Operations: Proposed Framework, Challenges, and Future Directions (https://arxiv.org/abs/2411.09788)
Comments:
          Submitted for review to the Proceedings of the IEEE

- **What's New**: 이 논문은 AI 기반 Human-Autonomy Teaming (HAT) 접근 방식을 통해 복잡한 환경에서 인간의 의사결정 능력을 향상시키는 방법을 탐구합니다. HAT의 주요 구성 요소로는 신뢰와 투명성, 인간과 AI 간의 최적 기능 분배, 상황 인식 및 윤리적 고려 사항 등이 포함됩니다.

- **Technical Details**: HAT 시스템은 인간과 자율 시스템 간의 협력을 통해 작업을 수행하는 새로운 영역입니다. 자율 시스템은 AI 알고리즘과 센서를 사용하여 환경을 인식하고 탐색하며, 전술 자율성(tactical autonomy)은 동적이고 복잡한 작전 환경에서 실시간으로 의사결정을 할 수 있는 능력을 의미합니다. 이 논문은 AI로 구동되는 HAT의 발전과 이로 인해 직원의 인지 능력 및 자율 시스템의 계산 능력을 통합하는 과정을 설명합니다.

- **Performance Highlights**: AI 기반 HAT를 통해 상황 인식이 향상되고 더욱 정보에 기반한 의사결정이 가능해지며, 이는 전술 작전의 효과성과 안전성을 높이는 데 기여합니다. 논문은 AI와 자율 시스템의 통합이 전술 환경에서 인간-자율 팀워크를 최적화하는 데 필요한 기초를 마련한다고 강조합니다.



### Deep Learning for Fetal Inflammatory Response Diagnosis in the Umbilical Cord (https://arxiv.org/abs/2411.09767)
- **What's New**: 이번 연구는 제대의 급성 태아 염증 반응(Acute Fetal Inflammatory Response, FIR)을 디지털 병리학( Digital Pathology)의 최신 딥러닝 기술을 활용해 분류하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 연구팀은 헤마톡실린 및 스테인(Eosin)으로 염색된 제대의 조직병리(histological) 슬라이드 4100개를 디지털화하고, 전자 건강 기록(EHR)에서 태반 진단을 추출했습니다. Attention 기반의 전체 슬라이드 학습(Whole Slide Learning) 모델을 사용하여 FIR을 분류하였으며, 비의료 이미지(ImageNet)로 사전 훈련된 ConvNeXtXLarge 모델과 조직병리 이미지(UNI)를 사용하여 사전 훈련된 모델의 성능을 비교했습니다.

- **Performance Highlights**: 여러 모델을 학습하여 앙상블을 구성한 결과, UNI를 사용한 모델의 예측이 테스트 데이터 세트에서 전체 균형 정확도(balanced accuracy) 0.836을 달성했습니다. 반면 ConvNeXtXLarge를 사용한 앙상블 예측은 0.7209로 더 낮은 정확도를 보였습니다. FIR 2 경우의 경우, 높은 정확도를 가진 모델에서 생성된 히트맵은 적절하게 염증을 나타내는 부위를 강조했습니다.



### SureMap: Simultaneous Mean Estimation for Single-Task and Multi-Task Disaggregated Evaluation (https://arxiv.org/abs/2411.09730)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문에서는 머신러닝 모델의 다양한 하위 집단에 대한 성과 평가에 대한 새로운 접근 방식을 제시한다. 특히, 여러 고객이 동일한 AI 모델을 사용하는 환경에서 각 고객의 데이터를 기반으로 한 다중 작업(disaggregated evaluation) 접근 방식을 연구하였다.

- **Technical Details**: 연구에서 제안하는 유도 평가 방법 SureMap은 다중 작업과 단일 작업에 대한 평가에서 높은 추정 정확도를 제공하며, 문제를 구조화된 동시 가우시안 평균 추정 문제로 변환하고 외부 데이터를 통합하여 효율성을 높인다. 이 방법은 최대 사후 확률 추정(maximum a posteriori, MAP)과 Stein의 비편향 위험 추정(SURES)를 기반으로 한다.

- **Performance Highlights**: SureMap은 다양한 도메인에서의 하위 집단 평가 작업에 대해 많은 강력한 경쟁자들에 비해 상당한 정확도 향상을 보여주었다. 특히, 다중 클라이언트 데이터를 통합할 경우 전반적으로 모든 평가 설정에서 주요 개선이 관찰되었다.



### Towards Neural Foundation Models for Vision: Aligning EEG, MEG, and fMRI Representations for Decoding, Encoding, and Modality Conversion (https://arxiv.org/abs/2411.09723)
- **What's New**: 이 논문은 대조 학습(contrastive learning)을 활용하여 다중 모달(multi-modal) 뇌 활동 표현의 신경 데이터(neural data)와 시각 자극(visual stimuli)을 정렬하는 기초 모델(foundational model)을 만드는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 우리는 뇌파(EEG), 자기 뇌파(MEG), 기능적 자기 공명 영상(fMRI) 데이터를 사용하여 프레임워크의 기능을 입증하였으며, 세 가지 주요 실험을 통해 시각 정보를 신경 데이터에서 디코딩(decoding)하고, 이미지를 신경 표현(neural representations)으로 인코딩(encoding)하며, 신경 모달리티(modality) 간 변환을 수행합니다.

- **Performance Highlights**: 결과는 다양한 뇌 이미징 기술(techniques)에 걸쳐 의미론적 정보(semantic information)를 정확하게 포착할 수 있는 모델의 능력을 강조하며, 디코딩, 인코딩, 모달리티 변환 작업에서의 잠재력을 입증합니다.



### Iterative Batch Reinforcement Learning via Safe Diversified Model-based Policy Search (https://arxiv.org/abs/2411.09722)
Comments:
          Workshop on Safe and Robust Robot Learning for Operation in the Real World (SAFE-ROL) at CoRL 2024

- **What's New**: 이번 논문에서는 반복적인 오프라인 강화 학습(iterative batch reinforcement learning, IBRL) 프레임워크를 제안합니다. 이는 기존에 수집된 데이터를 기반으로 정책을 지속적으로 개선하는 접근법으로, 안전성과 다양성 기준을 포함하여 실시간 데이터 수집을 효율적으로 안내합니다.

- **Technical Details**: IBRL은 앙상블 기반의 모델 기반 정책 탐색(model-based policy search) 기술을 활용하며, 안전 제약과 데이터 다양성을 동시에 고려합니다. 본 방법론은 이전에 훈련된 다양한 정책을 사용하여 새로운 데이터를 수집하고, 이를 기존의 데이터 세트에 추가하여 여러 번의 반복적인 학습을 수행하도록 설계되었습니다.

- **Performance Highlights**: 실험을 통해 2D 환경 및 Industrial Benchmark에서 제안하는 IBRL 접근법이 탐색 능력과 성능을 향상시킴을 입증하였으며, 안전성을 유지하면서 정책의 성능이 향상됨을 보여주었습니다.



### NFRs in Medical Imaging (https://arxiv.org/abs/2411.09718)
- **What's New**: 본 연구는 의료 영상 애플리케이션에서의 비기능 요구사항(Non-functional Requirements, NFRs)의 중요성을 강조하며, 이를 개선하기 위한 프레임워크 개발을 목표로 한다.

- **Technical Details**: 연구는 덴마크 한 병원에서 여러 관련자들과의 정성적 방법을 사용하여 비기능 요구사항의 종류를 파악하였다. 주요 비기능 요구사항으로는 효율성(Efficiency), 정확성(Accuracy), 상호운용성(Interoperability), 신뢰성(Reliability), 사용성(Usability), 적응성(Adaptability), 공정성(Fairness) 등이 있다.

- **Performance Highlights**: 해당 연구는 현재의 의료 영상 솔루션에서 AI 기술을 더 원활히 통합하기 위한 NFR 프레임워크의 필요성을 명확하게 주장하며, 향후 의료 영상 모델들의 병원 내 도입률을 높이는 데 기여할 것으로 기대된다.



### Feature Selection via Dynamic Graph-based Attention Block in MI-based EEG Signals (https://arxiv.org/abs/2411.09709)
Comments:
          4 pages, 2 figures, 1 table, Name of Conference: International Conference on Brain-Computer Interface

- **What's New**: 본 연구에서는 motor imagery (MI) 관련 특징을 강화하면서 낮은 상관관계를 가진 특징을 줄이는 end-to-end deep preprocessing 방법을 제안했습니다. 이 방법은 EEG 신호의 MI 작업에 대한 평가 성능을 증진시킬 수 있는 가능성을 보였습니다.

- **Technical Details**: 제안된 방법은 temporal, spatial, graph, 그리고 similarity 블록으로 구성되어 있으며, 각각의 블록은 MI 신호의 중요 특징을 더 효과적으로 추출하고 완화하는 데 기여합니다. 그래프 기반의 합성곱(convolution) 연산을 사용하여 전극 간 관계를 파악하고, 병합된 작용을 통해 처리된 특징들은 기존의 딥러닝 모델에 효과적으로 통합되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 BCI Competition IV의 공개 데이터셋에서 기존 모델들과 통합했을 때 성능을 향상시키고 MI 작업의 특징 분포를 더 클러스터된 형상으로 개선하는 것을 보여주었습니다.



### AI-Driven Feedback Loops in Digital Technologies: Psychological Impacts on User Behaviour and Well-Being (https://arxiv.org/abs/2411.09706)
- **What's New**: 디지털 기술의 급속한 발전은 사용자 행동과 동기 부여, 정신적 웰빙을 형성하는 데이터 기반의 피드백 루프(feedback loops), 웨어러블 디바이스(wearable devices), 소셜 미디어 네트워크, 모바일 애플리케이션을 만들어냈습니다. 이 연구는 이러한 피드백 메커니즘이 사용자에게 미치는 긍정적 및 부정적 심리적 영향을 탐구하는 것을 목표로 합니다.

- **Technical Details**: 본 연구는 설명적 조사 방법을 사용하여, 건강, 사회 및 라이프스타일 애플리케이션과 관련된 변화된 행동, 동기 부여 및 정신적 웰빙을 평가하기 위해 200명의 사용자를 의도적으로 선택하여 데이터를 수집하였습니다.

- **Performance Highlights**: 결과에 따르면, 피드백 메커니즘은 목표 달성과 사회적 상호 연결성을 촉진하였지만, 사용자들은 앱의 피드백에 의해 무의식적으로 그들의 행동이 형성되며 개인적인 자율성의 상실, 불안, 정신적 피로 및 생산성 저하를 경험했습니다. 기술 사용의 경계를 설정하고 피드백 메커니즘을 개선하여 인지 부담을 줄이는 것이 필요하다는 결론을 내렸습니다.



### Prices, Bids, Values: Everything, Everywhere, All at Onc (https://arxiv.org/abs/2411.09355)
- **What's New**: 이 논문에서는 Iterative Combinatorial Auctions (ICAs)의 설계를 분석하고, 머신 러닝 (ML)을 통해 경매 효율성을 극대화하기 위한 새로운 접근 방식을 제안합니다. 특히, Demand Queries (DQs)와 Value Queries (VQs)의 조합을 활용하여 효율성을 크게 향상시킨 새로운 ICA인 MLHCA를 소개합니다.

- **Technical Details**: MLHCA는 DQ 기반 라운드와 VQ 기반 추가 라운드를 결합한 하이브리드 ML 기반 경매 시스템입니다. 이 과정을 통해 경매 참여자들의 선호도를 효과적으로 수집하고, 경매의 최종 효율성을 높일 수 있습니다. 실험 결과, MLHCA는 이전의 SOTA(SOTA, State Of The Art) 알고리즘과 비교하여 쿼리 수를 40% 줄이면서도 더 높은 효율을 달성했습니다.

- **Performance Highlights**: MLHCA는 실제 경매 환경에서 다수의 쿼리를 통해 효율성을 최대화하고, 이용자들의 인지적 부담을 줄이는 동시에 수백만 달러의 복지 향상을 가져올 수 있습니다. 이 연구는 ICAs의 효율성을 10배까지 낮출 수 있는 방법을 제시하며, 실질적인 경제 영향을 미치는 새로운 기준을 설정합니다.



### Cyber-Forensic Review of Human Footprint and Gait for Personal Identification (https://arxiv.org/abs/2204.09344)
- **What's New**: 이번 연구는 범죄 현장에서의 발자국과 보행 패턴을 통한 개인 식별의 중요성을 강조합니다.

- **Technical Details**: 이 연구는 발자국(biometrics based on footprints)과 보행(gait)을 개인 식별 시스템의 대안으로 활용하는 방법을 조사합니다. 발자국은 생체 인식(biometrics)에서 비교적 신선한 기술로, 기존에 자주 사용되는 지문(fingerprints), 망막(retina), 홍채(iris), 얼굴 인식(face recognition)과 같은 방법을 보완할 수 있습니다.

- **Performance Highlights**: 이번 연구는 전 세계 테러 문제에 대한 대응 방안으로 발자국과 보행 패턴을 활용하여 범죄자를 식별할 수 있는 가능성을 제시합니다. 특히 특정 산업 분야의 소프트 타겟을 노리는 테러리스트를 추적하는데 유용할 것으로 예상됩니다.



### NeuralDEM -- Real-time Simulation of Industrial Particulate Flows (https://arxiv.org/abs/2411.09678)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 NeuralDEM이라는 혁신적인 접근 방식을 소개합니다. 이는 전통적인 DEM(Discrete Element Method) 방법의 느린 수치적 프로세스를 심층 학습 기반의 빠르고 적응 가능한 대체 모델로 전환합니다.

- **Technical Details**: NeuralDEM은 DEM의 Lagrangian discretization을 기본 연속 필드로 취급하며, 추가 보조 필드로 거시적(macroscopic) 행동을 직접 모델링합니다. 또한, 산업 규모 시나리오의 실시간 모델링을 위해 확장 가능한 다중 분기 신경 연산자(multi-branch neural operators)를 도입합니다.

- **Performance Highlights**: NeuralDEM은 16만 개의 CFD 셀과 50만 개의 DEM 입자를 사용하는 유동화층 반응기(coupled CFD-DEM fluidized bed reactors)를 28초 동안 신뢰성 있게 모델링할 수 있습니다. 이는 공정 사이클을 훨씬 더 빠르게 진행할 수 있는 기회를 제공합니다.



### Temporal Patterns of Multiple Long-Term Conditions in Individuals with Intellectual Disability Living in Wales: An Unsupervised Clustering Approach to Disease Trajectories (https://arxiv.org/abs/2411.08894)
- **What's New**: 이번 연구에서는 지적 장애(ID)를 가진 개인들의 다수의 만성 질환(MLTC) 공존 집단을 전자 건강 기록(EHR)을 기반으로 분석하였습니다. 연구는 2000년부터 2021년까지 웨일즈에서 수집된 13,069명의 데이터를 활용하여 MLTC 클러스터를 Независимый 분석하여 특징화하였습니다.

- **Technical Details**: 이 연구는 관찰된 40개의 만성 질환(LTC)으로 MLTC의 일부와의 쌍을 통해 시간적 방향성을 평가하였으며, 이 후 스펙트럼 클러스터링(Spectral Clustering) 기법을 통해 공통 질병 궤적을 그룹화하는 과정을 하였습니다. 연구는 또한 성별 및 연령에 따른 MLTC 패턴의 차이를 드러내기 위해 총 4.5개의 질환을 가진 데이터 집합을 다양하게 분석하였습니다.

- **Performance Highlights**: 연구 결과, 남성 45세 미만에서는 신경학적 질환이 주를 이루는 클러스터가 발견되었으며, 45세 이상에서는 순환계 질환이 있는 클러스터가 지배적이었습니다. 또한, 여성 45세 미만에서는 소화기 계통 질환이 가장 빈번하게 나타났으며, 45세 이상에서는 순환계 및 소화기 질환의 두 개의 클러스터가 확인되었습니다. 본 연구의 결과는 ID를 가진 개인들의 질병 진행 이해 및 맞춤형 의료 전략 설계에 기여할 수 있습니다.



New uploads on arXiv(cs.LG)

### MARS: Unleashing the Power of Variance Reduction for Training Large Models (https://arxiv.org/abs/2411.10438)
Comments:
          23 pages, 7 figures, 6 tables

- **What's New**: 이 논문에서는 대규모 모델의 효율적인 훈련을 위해 새로운 최적화 프레임워크인 MARS (Make vAriance Reduction Shine)를 제안합니다. 이는 적응형 경량 기법과 분산 감소를 통합하여 더 나은 성능을 목표로 합니다.

- **Technical Details**: MARS는 스케일링된 확률적 재귀 모멘텀(Scaled Stochastic Recursive Momentum)을 통해 전체 그래디언트의 분산을 줄이는 추정기를 제공합니다. 또한, AdamW, Lion, Shampoo와 같은 기존 방법에 대한 사전 조건 기법을 활용한 세 가지 MARS 인스턴스를 도입하였습니다.

- **Performance Highlights**: 실험 결과, MARS는 GPT-2 모델 훈련에서 AdamW에 비해 상당한 성능 향상을 보였습니다. MARS는 27억 토큰 내에서 2.58의 검증 손실을 달성했으며, 이는 AdamW가 같은 수준에 도달하는 데 50억 토큰이 필요했던 것과 대조적입니다. MARS는 또한 Hellaswag 다운스트림 작업에서 44.20%의 정확도로 AdamW의 42.31%를 초과했습니다.



### Back to Supervision: Boosting Word Boundary Detection through Frame Classification (https://arxiv.org/abs/2411.10423)
- **What's New**: 이 논문에서는 WORD SEGMENTATION (단어 분할) 과제를 위한 모델 불가지론적(frame-free) 프레임워크를 제안하였다. 또한 라벨 증강(label augmentation) 기법과 출력 프레임 선택(output-frame selection) 전략을 도입하여 더 효과적인 단어 경계 탐지를 가능하게 한다.

- **Technical Details**: 저자들은 Buckeye 데이터셋과 TIMIT 데이터셋을 사용하여 신뢰성 있는 결과를 측정하였다. 이들은 HuBERT와 같은 최신 인코더 모델을 사용하였고, SUPERVISED(감독 학습) 방식으로 훈련하여 프레임 분류 작업을 수행하였다. 본 연구의 기법은 BIO (begin, inside, outside) 포맷을 기반으로 하며, 각 프레임의 라벨 증강을 통해 BEGIN(시작)과 INSIDE/OUTSIDE(내부/외부) 프레임 간의 불균형 문제를 해결한다.

- **Performance Highlights**: Buckeye 데이터셋에서 F-value 0.8427, TIMIT 데이터셋에서 F-value 0.7436을 기록하였고, R-value는 각각 0.8489 및 0.7807에 달하였다. 이는 두 데이터셋 모두에서 새로운 최첨단 성능을 기록한 것이다. 이 연구는 오디오 토큰화(audio tokenization) 분야의 향후 연구를 위한 강력하고 효율적인 전처리 방법을 제공한다.



### Multiscale Dubuc: A New Similarity Measure for Time Series (https://arxiv.org/abs/2411.10418)
Comments:
          6 pages, 3 figures, IEEE Big Data 2024

- **What's New**: 이번 연구에서는 프랙탈 분석에서의 Dubuc의 변형 개념과 객체 인식에서 널리 사용되는 Intersection-over-Union (IoU) 측정을 결합하여 새로운 유사성 측정 방법인 Multiscale Dubuc Distance (MDD)를 소개합니다.

- **Technical Details**: MDD는 다중 척도 특성을 활용하여 두 시계열의 복잡성을 정량화하고 비교하는 유사성 측정 방법입니다. MDD는 세 가지 주요 기존 측정 방법인 Euclidean Distance (EuD), Longest Common Subsequence (LCSS) 및 Dynamic Time Warping (DTW)와 비교되어 성능을 평가합니다. 연구는 95개의 데이터셋을 사용하여 MDD가 데이터셋 별 최적화된 DTW와 유사한 성과를 낸다는 것을 보여줍니다.

- **Performance Highlights**: 실험 결과, MDD는 특정 파라미터가 조정되지 않은 상태에서도 DTW와 유사한 성과를 보였으며, 특히 MDD의 단일 파라미터를 조정했을 때 성능이 크게 향상되는 데이터셋도 발견되었습니다. MDD의 시간 복잡도는 시계열 길이에 대해 선형적이며, 이는 매우 큰 데이터셋을 다루는 실제 애플리케이션에 중요합니다.



### Features that Make a Difference: Leveraging Gradients for Improved Dictionary Learning (https://arxiv.org/abs/2411.10397)
Comments:
          9 pages, 8 figures. Submitted to NAACL 2025

- **What's New**: 본 논문에서는 Gradient Sparse Autoencoders (g-SAEs)를 도입하여 기존의 Sparse Autoencoders (SAEs)에서 발생하는 한계를 극복하고자 합니다. g-SAEs는 활성화 값뿐만 아니라 입력 활성화의 gradient를 고려하여 $k$개의 요소를 선택하는 TopK 활성화 기능을 수정하여, 모델 출력에 강하게 영향을 미치는 특징을 학습할 수 있도록 합니다.

- **Technical Details**: g-SAEs는 기존 SAE 아키텍처를 개선하여 gradient-aware TopK 활성화 기능을 활용합니다. 이 방법이 모델의 다운스트림 효과를 더 정확하게 포착할 수 있도록 해주며, 그 결과로 생성된 재구성을 원본 네트워크 성능에 더 충실합니다. 또한 g-SAEs는 더 적은 수의 비활성 단위를 가지고 있으며, 활성화의 gradient를 포함하여 여러 인기 있는 SAE 아키텍처의 Pareto frontiers에서 개선을 보여줍니다.

- **Performance Highlights**: g-SAEs는 기존 아키텍처와 비슷한 해석 가능성을 유지하면서, 특정 logits에 대해 더 많은 영향을 미치는 latents를 복구합니다. 이는 g-SAEs가 모델에 더 온전한 제어를 가능하게 하며, 다양한 맥락에서 향상된 모델 성능 조작을 제공합니다.



### Low-Latency Task-Oriented Communications with Multi-Round, Multi-Task Deep Learning (https://arxiv.org/abs/2411.10385)
- **What's New**: 본 논문에서는 다음 세대 통신 시스템에서의 작업 지향(Goal-oriented) 통신을 향상시키기 위한 다중 라운드 및 다중 작업 학습(MRMTL) 접근법을 제안합니다. 이 방법은 수신자로부터의 피드백에 따라 송신자가 동적으로 채널 사용량을 업데이트할 수 있도록 합니다.

- **Technical Details**: MRMTL은 송신자가 압축된 잠재 표현(Compressed latent representations)을 생성하며, 수신자는 이러한 신호를 분류하는 기계 학습 과제를 수행합니다. 이 과정에서 깊은 신경망(Deep Neural Networks, DNN)이 사용되며, 채널과 데이터 특성을 고려하여 공동 훈련됩니다.

- **Performance Highlights**: MRMTL은 기존의 대규모 채널 사용량이 필요한 방법과 유사한 정확도를 달성하면서, 이전 라운드의 신호를 통합함으로써 지연(Delay)을 크게 줄였습니다. CIFAR-10 데이터셋을 사용하여 실험한 결과, MRMTL이 작업 지향 통신의 효율성을 현저히 향상시키는 것으로 나타났습니다.



### Framework for Co-distillation Driven Federated Learning to Address Class Imbalance in Healthcar (https://arxiv.org/abs/2411.10383)
Comments:
          Accepted at CODS COMAD'24 and to be published in the Discover Data Journal(this https URL)

- **What's New**: 본 연구는 연합 학습(Federated Learning, FL) 환경에서 의료 이미지를 활용한 훈련 시 발생하는 클래스 불균형 문제를 해결하기 위해 co-distillation 기반의 프레임워크를 제안합니다. 이 프레임워크는 클라이언트 간의 지식 공유를 통해 학습 결과를 향상시키며 개인정보 보호를 유지합니다.

- **Technical Details**: 제안된 방법은 각 클라이언트(병원)가 서로의 예측 소프트 레이블(soft labels)을 공유하여 모델 학습 중에 지식을 교환하는 방식입니다. 이는 클래스 불균형 문제를 처리하기 위해 설계되었으며, 기본 FL 방법보다 통신 오버헤드가 적어 효율적인 학습이 가능합니다.

- **Performance Highlights**: 실험 결과, 의료 분야의 연합 학습 환경에서 co-distillation 방식이 클래스 불균형 문제를 효과적으로 해결하는 것으로 나타났습니다. 또한, 제안된 프레임워크는 가장 낮은 표준 편차를 보이며, 클라이언트 수가 다수인 경우에도 뛰어난 성능을 달성했습니다.



### Continual Adversarial Reinforcement Learning (CARL) of False Data Injection detection: forgetting and explainability (https://arxiv.org/abs/2411.10367)
- **What's New**: 이 논문은 스마터 인버터에 대한 허위 데이터 주입 공격(FDIAs)이 증가하는 문제를 다루고 있으며, 이를 해결하기 위한 새로운 접근법으로 지속적 적대적 강화학습(CARL)을 제안합니다. 기존의 데이터 기반( data-based) 검출 방법의 취약점을 드러내고, 이에 대한 개선점을 설명합니다.

- **Technical Details**: 제안된 방법은 적대적 예제를 포함하여 데이터 기반 검출 훈련 절차를 향상시키고, 지속적인 학습 구현이 치명적 망각(catastrophic forgetting)에 영향을 받을 수 있음을 보여줍니다. 이러한 망각은 모든 생성된 FDIA 시나리오에 대해 공동 훈련(joint training) 전략을 적용함으로써 해결할 수 있습니다.

- **Performance Highlights**: 논문에서는 제안된 메커니즘이 기존 방법보다 더 나은 적대적 공격 내성을 달성할 수 있음을 나타내며, 지속적 학습이 데이터 기반 검출의 한계를 극복하는 데 효과적임을 입증합니다.



### Weakly-Supervised Multimodal Learning on MIMIC-CXR (https://arxiv.org/abs/2411.10356)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 13 pages. arXiv admin note: text overlap with arXiv:2403.05300

- **What's New**: 본 연구는 Multimodal Variational Mixture-of-Experts (MMVM) VAE를 의료 분야에 적용할 가능성과 MIMIC-CXR 데이터셋에서의 성과를 심층 분석합니다. 이 모델은 다양한 데이터 모달리티를 효과적으로 결합할 수 있는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: MMVM VAE는 여러 데이터 모달리티를 소프트-셰어링 메커니즘(soft-sharing mechanism)으로 결합하여 각 모달리티가 공동 포스터리어(distribution)에 더 유연하게 기여하도록 합니다. 이는 더 나은 잠재 표현(latent representations)을 가능하게 하여 각 인코딩이 원본 정보(original features)를 더 잘 보존할 수 있게 합니다.

- **Performance Highlights**: MMVM VAE는 MIMIC-CXR 데이터셋에서 기존의 다중모달 VAEs 및 완전 감독 학습 방법들보다 뛰어난 성능을 보였으며, 이는 실제 의료 애플리케이션에 매우 적합한 잠재력을 나타냅니다.



### On the Cost of Model-Serving Frameworks: An Experimental Evaluation (https://arxiv.org/abs/2411.10337)
- **What's New**: 이 논문에서는 머신러닝(ML) 모델을 실시간으로 제공하기 위한 다섯 개의 모델 서빙 프레임워크(TensorFlow Serving, TorchServe, MLServer, MLflow, 그리고 BentoML)의 성능을 평가하였습니다.

- **Technical Details**: 프레임워크에 대한 세심한 분석을 통해, 각 프레임워크의 CPU/GPU 지원 능력과 처리 가능한 ML/DL 라이브러리를 비교합니다. TensorFlow Serving은 고성능 모델 서빙 시스템으로, TensorFlow DL 모델과의 원활한 통합을 제공합니다. 반면 TorchServe는 PyTorch 모델의 배포를 간편화하는 툴로 피어의 인기를 얻고 있습니다.

- **Performance Highlights**: TensorFlow Serving은 모든 서빙 프레임워크 중에서 가장 낮은 지연 시간을 기록하였으며, DL 전용 프레임워크(TensorFlow Serving 및 TorchServe)가 일반 목적 ML 프레임워크(BentoML, MLFlow, MLServer)보다 지연 시간이 유의미하게 낮은 것을 보여주었습니다.



### Bitcoin Research with a Transaction Graph Datas (https://arxiv.org/abs/2411.10325)
- **What's New**: 이 논문은 비트코인 사용자를 위한 거래를 나타내는 대규모 트랜잭션 그래프 데이터셋을 소개하며, 이는 2억 5200만 개의 노드와 7억 8500만 개의 엣지를 포함합니다. 이 그래프는 거의 13년 동안의 데이터를 바탕으로 하며, 시간 표시가 되어 있어 거래 시간이 명확히 기록되어 있습니다.

- **Technical Details**: 이 비트코인 데이터셋은 하위주어진 강의 모델(deep learning model)인 그래프 신경망(graph neural networks)을 사용해 노드 레이블(label) 예측에 대한 다양한 기법들을 훈련했습니다. 제공된 데이터셋의 경우, 33,000개의 노드는 실체(entity) 유형을 기반으로 레이블이 붙여졌으며, 거의 100,000개의 비트코인 주소는 실체 이름과 유형으로 레이블이 붙여졌습니다.

- **Performance Highlights**: 이 연구는 비트코인 거래 분석 외에도 다른 분야에서 이 데이터셋의 유용성을 보여주는 여러 사용 사례를 제시합니다. 또한, 논문에서 제시된 결과들은 후속 연구를 위한 기준선으로 활용될 수 있습니다.



### Towards Sample-Efficiency and Generalization of Transfer and Inverse Reinforcement Learning: A Comprehensive Literature Review (https://arxiv.org/abs/2411.10268)
- **What's New**: 이번 논문은 전이 강화학습(Transfer Reinforcement Learning, T-RL) 및 역 강화학습(Inverse Reinforcement Learning, IRL) 기법을 활용하여 강화학습(Reinforcement Learning, RL) 알고리즘의 샘플 효율성(sample efficiency) 및 일반화(generalization) 문제를 체계적으로 다루고 있습니다. 이를 통해 RL의 다양한 응용 분야에서의 효율적인 지식 이전(knowledge transfer)에 대한 최근 발전을 포괄적으로 검토했습니다.

- **Technical Details**: 문헌에서는 RL을 통해 최적의 정책(optimal policy)을 학습하기 위해 에이전트(agent)가 행동 공간(action space)을 탐색하며, 이 과정에서 상황을 행동으로 매핑(mapping)하는 정책을 조정하는 기술적 과정을 설명하고 있습니다. 또한, T-IRL의 기본 방법론과 최근 연구의 진전을 다루며, 특히 인간-루프(human-in-the-loop) 및 시뮬레이션-실제(sim-to-real) 전략을 통한 지식 이전의 효율성을 강조하고 있습니다.

- **Performance Highlights**: T-IRL 기법을 통해 최근 연구들은 에이전트가 이전 경험을 기반으로 적은 수의 경험 전환(experience transitions)으로도 효과적으로 학습할 수 있도록 하였으며, 다중 에이전트(multi-agent) 및 다중 의도(multi-intention) 문제로의 확장도 이루어졌습니다. 이러한 방식은 강화학습의 샘플 효율성과 일반화를 개선하는 데 기여하였고, 향후 다양한 적용 가능성이 기대됩니다.



### Uncertainty in Supply Chain Digital Twins: A Quantum-Classical Hybrid Approach (https://arxiv.org/abs/2411.10254)
- **What's New**: 이 연구는 양자-고전 혼합 머신 러닝 모델을 사용하여 불확실성 정량화 (UQ)를 조사하였으며, 공급망 디지털 트윈 및 금융 리스크 평가와 같은 복잡하고 동적인 분야에 응용됩니다. 특히 양자 피처 변환이 하이브리드 아키텍처 내에서 UQ에 미치는 영향을 분석하고 있습니다.

- **Technical Details**: 혼합 프레임워크 내에서 다양한 모델에 대해 기존의 UQ 기법을 적용하여 양자 피처 변환이 불확실성 전파에 미치는 영향을 조사하였습니다. 4개의 큐비트에서 16개의 큐비트로 증가시키면서 모델이 이상치 탐지 (OD) 샘플에 대해 어떻게 반응하는지를 확인하였습니다. 기존의 머신 러닝 모델과 달리, 양자 컴퓨팅 기술을 활용하여 데이터 특성을 변환하는 역할을 검토하였습니다.

- **Performance Highlights**: 연구 결과, 양자 컴퓨팅 기술이 데이터 특성을 불확실성 정량화에 효과적으로 변화시킬 수 있음을 보여줍니다. 특히, 다양한 큐비트 구성에서의 모델 성능을 비교함으로써 동적 환경에서의 강건한 의사 결정에 필요한 이상치 탐지 능력 향상을 입증했습니다.



### Machine Learning Algorithms to Assess Site Closure Time Frames for Soil and Groundwater Contamination (https://arxiv.org/abs/2411.10214)
Comments:
          14 pages, 6 figures

- **What's New**: 이번 연구에서는 Monitored Natural Attenuation (MNA) 방법의 효과성을 높이기 위해 Python 패키지인 PyLEnM의 기능을 확장하였습니다. 새로운 알고리즘을 도입하여 오염물질의 농도 예측 및 분석 기능을 강화했습니다.

- **Technical Details**: 본 연구에서는 Sr-90 및 I-129와 같은 오염물질이 안전 기준에 도달할 때까지의 시간을 추정하기 위해 linear regression과 Bidirectional Long Short-Term Memory (Bi-LSTM) 네트워크를 사용합니다. Random Forest 회귀 분석을 통해 안전 기준 도달에 영향을 미치는 요인들도 식별하였습니다.

- **Performance Highlights**: Savannah River Site (SRS)의 데이터를 활용하여 Bi-LSTM 모델이 향후 4년간의 오염물질 농도를 효과적으로 예측할 수 있음을 보여주었습니다. 연구 결과, 초기 농도 및 지하수 흐름 역학에 따라 오염물질 농도가 유의미하게 감소하는 경향을 나타냈습니다.



### Embedding Byzantine Fault Tolerance into Federated Learning via Virtual Data-Driven Consistency Scoring Plugin (https://arxiv.org/abs/2411.10212)
Comments:
          7 pages

- **What's New**: 이 논문에서는 Byzantine 공격에 강한 federated learning (FL) 기술을 위해 기존 시스템에 통합 가능한 새로운 플러그인 기반의 아키텍처를 제안합니다. 이 플러그인은 가상 데이터 샘플을 생성하여 모델의 일관성 점수를 평가함으로써 악성 에지 장치를 효과적으로 필터링할 수 있도록 합니다.

- **Technical Details**: 제안된 접근 방식은 각 로컬 업데이트에 대해 모델의 일관성 점수를 계산하여 손상된 에지 장치를 식별합니다. 이 점수 메커니즘은 집계 단계 전에 활용되며, 기존 FL 기술이 Byzantine 공격에 대한 저항력을 가질 수 있도록 하며 원래의 이점을 유지합니다.

- **Performance Highlights**: 의료 이미지 분류 작업에서의 수치적 결과는 제안된 플러그인을 여러 대표 FL 알고리즘에 통합했을 때 효과적으로 Byzantine 강인성을 달성할 수 있음을 검증했습니다. 또한, 제안된 플러그인은 Byzantine 공격이 없을 때도 기본 FL 알고리즘의 원래 수렴 특성을 유지합니다.



### FengWu-W2S: A deep learning model for seamless weather-to-subseasonal forecast of global atmospher (https://arxiv.org/abs/2411.10191)
Comments:
          23 pages,8 figures

- **What's New**: FengWu-Weather to Subseasonal (FengWu-W2S) 모델은 하나의 AI 모델을 기반으로 기상 예보와 기후 예보를 통합하여 Seamless Forecasting을 실현하는 새로운 접근법을 제시합니다.

- **Technical Details**: FengWu-W2S 모델은 FengWu 글로벌 기상 예보 모델을 기반으로 하며, 해양-대기-육상의 결합 구조(ocean-atmosphere-land coupling structure)와 다양한 변동성 전략(diverse perturbation strategy)을 통합하여 6시간마다 최대 42일간의 예보를 생성합니다. 이 모델은 자기회귀적(autoregressive) 방식으로 작동하여 예측합니다.

- **Performance Highlights**: FengWu-W2S는 3-6주까지 정확하게 대기 조건을 예측할 수 있으며, 이는 지구 표면 기온(global surface air temperature), 강수량(precipitation), 기저 높이(geopotential height) 및 Madden-Julian Oscillation (MJO)와 North Atlantic Oscillation (NAO)과 같은 계절 내 신호(intraseasonal signals)의 예측 능력을 향상시킵니다. 또한, 예측 오차 성장(forecast error growth)에 대한 탈락 실험(ablation experiments)을 통해 AI 기반 통합 시스템 개발의 잠재적인 경로를 발견했습니다.



### The Surprising Ineffectiveness of Pre-Trained Visual Representations for Model-Based Reinforcement Learning (https://arxiv.org/abs/2411.10175)
Comments:
          Published at the 38th Conference on Neural Information Processing Systems (NeurIPS 2024). Project page: this https URL

- **What's New**: 본 논문에서는 모델 기반 강화 학습(Model-Based Reinforcement Learning, MBRL) 환경에서 다양한 사전 학습된 시각 표현(Pre-trained Visual Representations, PVRs)의 효과를 벤치마킹합니다. 이전의 연구는 PVR이 샘플 효율성과 일반화 능력을 향상시킨다고 보고했으나, MBRL에 대한 PVR의 잠재력은 거의 탐색되지 않았습니다. 이 연구를 통해 PVR이 MBRL의 성능에 미치는 영향을 분석합니다.

- **Technical Details**: 이 연구에서는 MBRL에서의 PVR의 데이터 효율성과 일반화 능력을 평가하며, PVR의 다양한 속성이 MBRL 에이전트의 성능에 미치는 영향을 조사합니다. MBRL에서는 특징을 선별하여 환경의 동력학 모델을 학습하고, CNN(Covolutional Neural Networks)을 사용하여 시각적 상태 표현을 활용합니다. 이 연구의 결과에 따르면, 현재의 PVR은 스크래치에서 학습한 표현보다 샘플 효율성이 더 뛰어나지 않으며, OOD(out-of-distribution) 설정에 대한 일반화 능력도 떨어지는 것으로 나타났습니다.

- **Performance Highlights**: 본 논문에서 수행한 실험 결과, 스크래치에서 학습한 표현을 사용하는 모델이 PVR을 사용하는 모델보다 더 우수한 성능을 보이는 경우가 많았습니다. 데이터 다양성과 네트워크 아키텍처가 OOD 일반화 성능에 가장 중요한 요소로 나타났습니다.



### Causal Time-Series Synchronization for Multi-Dimensional Forecasting (https://arxiv.org/abs/2411.10152)
Comments:
          14 pages

- **What's New**: 이 논문에서는 프로세스 산업에서의 디지털 트윈(Digital Twins)을 위한 혁신적인 채널 종속(pre-training) 방법론을 제안하고 있습니다. 이는 다차원 시계열 데이터에 대한 예측 모델링의 복잡성을 극복하는 데 중점을 두고 있으며, 특히 동기화된 원인-결과(cause-effect) 쌍을 활용하여 훈련 샘플을 생성합니다.

- **Technical Details**: 제안된 방법은 (i) 데이터 기반 방법을 사용하여 고도로 지연된 원인 관계를 식별하고, (ii) 원인-결과 쌍을 동기화하여 채널 종속 훈련 샘플을 생성하며, (iii) 채널 종속 예측 채널의 효과성을 평가하는 것입니다. 또한, 선형 그랜저 인과법을 사용하여 인과 관계 그래프를 추출하고 다차원 시계열 데이터를 동기화된 원인-결과 쌍으로 나누는 방법을 설명합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 전통적인 훈련 방법에 비해 예측 정확도와 일반화 능력이 크게 향상됨을 보여줍니다. 미래의 다차원 데이터와 다양한 변수들에 대한 예측 성능 개선이 기대됩니다.



### PFML: Self-Supervised Learning of Time-Series Data Without Representation Collaps (https://arxiv.org/abs/2411.10087)
- **What's New**: 이번 연구에서는 Masked Latents에서 기능 예측(PFML)이라는 새로운 self-supervised learning (SSL) 알고리즘을 제안하였습니다. 이 알고리즘은 시간 시계열 데이터에 적용 가능하며, 기존 알고리즘에서 자주 발생하는 representation collapse 문제를 피하였습니다.

- **Technical Details**: PFML 알고리즘은 마스킹된 embedding에 해당하는 입력 신호의 통계적 기능(functionals)을 예측하는 방식으로 작동합니다. 이는 마스킹되지 않은 embedding의 시퀀스를 기반으로 합니다. PFML은 복잡한 하이퍼파라미터 조정 없이 다양한 시간 시계열 데이터 도메인에 쉽게 적용될 수 있도록 설계되었습니다.

- **Performance Highlights**: PFML은 세 가지 서로 다른 데이터 모달리티(영아의 자세 및 움직임 분류, 음성 데이터 기반 감정 인식, EEG 데이터를 통한 수면 단계 분류)에서 복잡한 실제 분류 작업을 수행하여 효과성을 입증하였습니다. PFML은 기존 유사한 SSL 방법에 비해 우수한 성능을 보이고, 현재의 최전선 수준의 SSL 방법과도 경쟁력을 보여주었습니다.



### Jal Anveshak: Prediction of fishing zones using fine-tuned LlaMa 2 (https://arxiv.org/abs/2411.10050)
- **What's New**: 최근 전 세계 및 인도 정부의 수산업 관련 데이터 모니터링 및 수집이 크게 발전하였습니다. 이러한 데이터의 유용성을 극대화하기 위해 저자들은 Jal Anveshak이라는 새로운 어플리케이션 프레임워크를 소개합니다.

- **Technical Details**: Jal Anveshak은 Dart와 Flutter로 작성된 어플리케이션 프레임워크로, Llama 2 기반의 Large Language Model을 활용합니다. 이 모델은 수산물 수확량 및 가용성에 관한 정부 데이터를 사전 처리(pre-processed)하고 증강(augmented)하여 세밀하게 조정된 것입니다.

- **Performance Highlights**: 이 프레임워크의 주요 목표는 인도의 어부들이 해안 지역에서 최대 수확량을 안전하게 얻고, 다국어 및 다중 모드(multimodal) 방식으로 어업 관련 질문을 해결하는 데 도움을 주는 것입니다.



### Physics-informed neural networks need a physicist to be accurate: the case of mass and heat transport in Fischer-Tropsch catalyst particles (https://arxiv.org/abs/2411.10048)
- **What's New**: 이 연구에서는 PINN(Physics-Informed Neural Networks)의 신뢰성 문제를 극복하는 방법을 제시합니다. 전통적인 신경망을 사용하여 다단계 시뮬레이션의 필요성을 줄이지만, 입력 매개변수 범위의 극단적인 끝에서의 신뢰성 문제 때문에 널리 채택되지 않습니다. 연구팀은 Fischer-Tropsch 합성과 관련된 비선형 반응-확산 방정식을 기반으로 PINN 아키텍처를 수정하여 이론적 지식의 전이 문제를 해결하는 방안을 제안하였습니다.

- **Technical Details**: 연구는 PINN을 사용하여 반응-확산 방정식을 해결하는 시스템을 문제로 다루었습니다. 연구팀은 전통적인 신경망의 정확도 검증 전략이 PINN의 불안정성을 초래할 수 있는 특성을 간과할 수 있음을 보여주고, 아키텍처를 수정하여 이론적 근거에 기반한 제약 조건을 자연스럽게 포함시키는 방법을 제안하였습니다. 또한, 수정된 PINN 아키텍처를 통해 기존의 유한차분 솔버와 결합하여 안정적인 수렴을 확보했습니다.

- **Performance Highlights**: 제안된 PINN과 개선된 수치적 방법을 결합함으로써, 시뮬레이션의 전반적인 안정성을 회복하면서 PINN의 속도 향상을 유지하는 데 성공했습니다. 이 연구는 화학 반응기 시뮬레이션에서 혼합 전송 방정식 해결기의 가능성을 논의하면서, PINN의 상대적 오류가 매우 낮은 성과를 달성했습니다. 연구 결과, PINN은 효율성을 크게 향상시키며, 유한차분 방정식 해결기의 안정성을 보장하기 위해 필수적임을 보여줍니다.



### Model Inversion Attacks: A Survey of Approaches and Countermeasures (https://arxiv.org/abs/2411.10023)
Comments:
          40 pages, 17 figures

- **What's New**: 이 논문은 Model Inversion Attacks (MIAs)에 대한 새로운 연구 결과를 요약하고 있습니다. MIAs는 잘 훈련된 모델에 대한 접근을 이용하여 개인 데이터의 민감한 특성을 추출하는 공격으로, 최근 여러 분야에서 그 효과가 입증되었습니다.

- **Technical Details**: 본 연구는 MIAs의 공격 및 방어 방법을 최신 정보로 정리합니다. MIAs는 블랙박스 또는 화이트박스 접근 방식을 통해 모델의 훈련 데이터를 추출하는 것을 목표로 하며, 반복적인 쿼리를 통해 모델의 출력을 관찰하여 비공식적으로 데이터에 대한 정보 접근을 시도합니다.

- **Performance Highlights**: MIAs는 이미지를 추출하거나 개인의 정확한 정보를 복구하는 등 여러 실질적인 공격 방법과 도전 과제를 보여줍니다. 모델의 출력에 대한 접근만이 가능한 블랙박스 환경에서조차도 MIAs는 성공 가능성이 있으며, 이는 실세계 애플리케이션에서의 프라이버시 위험성을 증대시킵니다.



### DeepMedcast: A Deep Learning Method for Generating Intermediate Weather Forecasts among Multiple NWP Models (https://arxiv.org/abs/2411.10010)
Comments:
          12 pages, 8 figures

- **What's New**: DeepMedcast는 여러 NWP 모델 간의 중간 예측(medcast)을 생성하는 딥러닝 방법을 제안하며, 전통적인 평균 기반 접근방식과는 달리 기상 현상을 왜곡하지 않고 일관성 있는 예측을 제공합니다.

- **Technical Details**: DeepMedcast는 딥 뉴럴 네트워크(DNN)를 사용하여 두 NWP 출력 간의 중간 예측을 생성하며, 단일 모델의 두 예측 시점 간의 데이터를 입력으로 활용하여 중간 예측을 생성합니다. 이 접근 방식은 두 모델 간의 예측 오류의 영향을 받지 않도록 설계되었습니다.

- **Performance Highlights**: DeepMedcast는 기상 예측의 정확성과 설명 가능성(explicability)을 동시에 충족하여 신뢰할 수 있는 기상 경고 및 안내 발행에 기여할 수 있는 잠재력을 가지고 있습니다.



### DuSEGO: Dual Second-order Equivariant Graph Ordinary Differential Equation (https://arxiv.org/abs/2411.10000)
- **What's New**: 본 논문은 이차 동차 미분 방정식(	extbf{Dual Second-order Equivariant Graph Ordinary Differential Equation, DuSEGO})을 제안하여 기존의 등변 GNN 모델의 한계를 극복하고자 합니다. DuSEGO는 먼저 1차 정보를 넘어서 2차 동역학적 시스템을 모델링할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: DuSEGO는 그래프 임베딩과 노드 좌표에 동시 적용되는 이차 동변 그래프 ordinary differential equations (Graph ODEs)를 활용하여, 과도한 평활화(over-smoothing)와 그래디언트 폭주나 소실 문제를 완화합니다. 이론적으로 DuSEGO가 등변 속성을 유지함을 증명하고, 특징 표현 및 좌표 업데이트에서의 과도한 평활화 문제를 효과적으로 alleviates 합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 extensive experiments를 통해 DuSEGO가 기존 방법들보다 우수한 성능을 보임을 입증하였습니다. 이는 DuSEGO가 GNNs의 표현력과 깊이 훈련을 가능하게 함을 의미합니다.



### Adaptive Non-Uniform Timestep Sampling for Diffusion Model Training (https://arxiv.org/abs/2411.09998)
- **What's New**: 이번 연구에서는 diffusion 모델의 효율성을 높이기 위한 새로운 비균일 타임스텝 샘플링 방법을 제안합니다. 기존의 균일한 타임스텝 샘플링 방식은 복잡한 데이터 분포로 인해 훈련 과정에서 수렴하는 데 많은 연산 비용이 필요하다는 문제점을 가지고 있습니다. 우리의 접근 방식은 중요한 타임스텝을 우선시하여 더 빠른 수렴을 이끌어냅니다.

- **Technical Details**: 제안한 방법은 각 타임스텝에서의 그래디언트 업데이트의 영향을 추적하며, 최적의 목표를 효과적으로 최소화하는 타임스텝을 적응적으로 선택합니다. 실험 결과에 따르면, 이 방법은 훈련 과정을 가속화하며, 다양한 데이터셋과 diffusion 아키텍처에서 우수한 성능을 보여줍니다. 또한 기존의 타임스텝 샘플링 및 가중치 휴리스틱보다 더 높은 강건성을 제공합니다.

- **Performance Highlights**: 실험 결과, 새로운 비균일 타임스텝 샘플링 방법은 기존의 방법들보다 훈련 과정을 더욱 빠르고 효과적으로 개선했습니다. Stable-Diffusion-2.0 및 Open-Sora와 같은 모델에서 기존의 장비 사용 시간 대비 30% 이상의 연산 시간 단축이 가능함을 보여주었습니다.



### Establishing and Evaluating Trustworthy AI: Overview and Research Challenges (https://arxiv.org/abs/2411.09973)
Comments:
          Accepted in Frontiers in Big Data and AI, Research Topic: Towards Fair AI for Trustworthy Artificial Intelligence

- **What's New**: 이 논문은 AI 시스템의 신뢰성을 확보하기 위해 충족해야 할 여섯 가지 요구 사항을 정의하고, 각각의 정의와 평가 방법, 연구 도전 과제를 논의합니다.

- **Technical Details**: 신뢰할 수 있는 AI의 여섯 가지 요구 사항은 다음과 같습니다: 1) 인간의 주체성과 감독 (human agency and oversight), 2) 공정성과 비차별 (fairness and non-discrimination), 3) 투명성과 설명 가능성 (transparency and explainability), 4) 견고함과 정확성 (robustness and accuracy), 5) 프라이버시와 보안 (privacy and security), 6) 책임성 (accountability). 각 요구 사항은 정의되고 이를 수립하고 평가하기 위한 방법이 제시됩니다.

- **Performance Highlights**: 저자들은 신뢰할 수 있는 AI의 구현 및 평가를 위한 방법론과 연구 도전 과제를 다루며, 지속적으로 변화하는 시스템의 다이나믹스와 실제 환경에서의 연구를 강조합니다.



### Is Precise Recovery Necessary? A Task-Oriented Imputation Approach for Time Series Forecasting on Variable Subs (https://arxiv.org/abs/2411.09928)
- **What's New**: 본 논문에서는 Variable Subset Forecasting (VSF)라는 새로운 시나리오를 다루며, 훈련 단계에서 사용된 변수가 추론 단계에서 전혀 나타나지 않는 상황에서의 예측 과제를 해결하기 위한 Task-Oriented Imputation for VSF (TOI-VSF) 프레임워크를 제안합니다.

- **Technical Details**: TOI-VSF는 self-supervised learning을 활용하여 결측 변수를 채우는 모듈을 포함하고 있으며, 예측 모델에 구애받지 않는 구조를 가집니다. 이 프레임워크는 imputation(보충)과 forecasting(예측)을 동시에 학습할 수 있는 joint learning 전략을 채택하고 있습니다. 또한, imputation 과정이 예측 목표에 직접적으로 기여하도록 설계되었습니다.

- **Performance Highlights**: 네 개의 데이터셋에서의 광범위한 실험 결과, TOI-VSF는 평균 15% 향상된 성능을 보여 기존의 기준 모델보다 뛰어난 결과를 나타냈습니다.



### Physics-informed Machine Learning for Battery Pack Thermal Managemen (https://arxiv.org/abs/2411.09915)
- **What's New**: 이번 연구에서는 전기차의 인기로 인해 리튬이온 배터리에 대한 수요가 증가하는 가운데, 배터리 성능과 안전성에 영향을 미치는 온도를 효율적으로 관리할 수 있는 새로운 방법을 제안합니다.

- **Technical Details**: 물리적 법칙을 모델에 적용하는 Physics-informed machine learning을 활용하여 배터리 팩의 온도 분포를 추정하는 대체 모델(surrogate model)을 개발했습니다. 이를 위해, 21700 배터리 팩의 간접 액체 냉각 시스템과 실험 결과를 바탕으로 한 간소화된 유한 요소 모델(finite element model)을 구축했습니다. 냉각수의 높은 유량에 의해 냉각판은 일정한 온도 경계로 고려했으며, 배터리 셀은 열원으로 작용합니다. 손실 함수(loss function)는 유한 차분법(finite difference method)에 기반한 열 전도 방정식을 고려하여 구성되었습니다.

- **Performance Highlights**: Physics-informed convolutional neural network는 같은 훈련 데이터를 사용한 데이터 기반 방법과 비교했을 때 15% 이상의 정확도 향상을 보여주었습니다.



### Statistical Analysis of Policy Space Compression Problem (https://arxiv.org/abs/2411.09900)
- **What's New**: 이 연구는 정책 공간 압축(policy space compression)을 통해 샘플 효율성을 높이는 방법을 제시합니다. 기존의 정책 탐색 방법들은 방대한 정책 공간 탐색의 비효율성 문제를 해결하기 위해, 샘플링 수를 최적화하는 데 필요한 샘플 수를 정량적으로 분석합니다.

- **Technical Details**: 정책 압축을 달성하기 위해, Rényi divergence를 활용하여 진정한 정책 분포와 추정된 정책 분포 간의 유사성을 측정하고, 오류 경계를 설정하여 효과적인 근사를 보장합니다. $l_1$ 노름을 사용하여 분석을 단순화하고, 모델 기반(model-based) 및 모델 없는(model-free) 설정에 대한 샘플 수 요건을 결정합니다.

- **Performance Highlights**: 이 연구는 $l_1$ 노름과 Rényi divergence에서 얻은 오류 경계를 연관시켜, 정책 공간의 정점 근처와 중앙의 정책 간의 차이를 파악하여 필요한 샘플 크기에 대한 하한 및 상한을 제시합니다.



### Off-Dynamics Reinforcement Learning via Domain Adaptation and Reward Augmented Imitation (https://arxiv.org/abs/2411.09891)
Comments:
          Published at Neurips 2024

- **What's New**: 본 연구에서는 동적 변화가 있는 목표 영역(target domain)에서 정책(policy)을 성공적으로 적용하기 위한 새로운 접근 방식인 Domain Adaptation and Reward Augmented Imitation Learning (DARAIL)을 제안합니다.

- **Technical Details**: DARAIL은 보상 수정(reward modification)을 활용하여 영역 적응(domain adaptation)을 수행하며, 생성적 적대적 모방 학습(generative adversarial imitation learning)에서 관찰 기반으로 하는 프레임워크를 따릅니다. 이 과정에서 보상 증대 추정치(reward augmented estimator)를 정책 최적화(policy optimization) 단계에 적용합니다.

- **Performance Highlights**: 실험 결과, 제안하는 DARAIL 방법은 순수한 보상 수정 방법보다 우수한 성능을 보였으며, 벤치마크 오프-다이내믹스 환경에서 다른 기준선(baselines)보다도 뛰어난 결과를 기록했습니다.



### InvestESG: A multi-agent reinforcement learning benchmark for studying climate investment as a social dilemma (https://arxiv.org/abs/2411.09856)
- **What's New**: 본 논문에서는 InvestESG라는 새로운 다중 에이전트 강화 학습(multi-agent reinforcement learning, MARL) 벤치마크를 소개하여 ESG(환경, 사회, 거버넌스) 공시 의무가 기업의 기후 투자에 미치는 영향을 연구합니다.

- **Technical Details**: InvestESG는 PyTorch 및 GPU 가속 JAX 프레임워크를 지원하며, 기업들이 기후 완화 노력으로 인한 단기 수익 손실과 기후 위험 감소로 인한 장기 이익을 균형 있게 고려하는 사회 딜레마를 모델링합니다. 실험에서는 기업이 완화, 그린워싱(greenwashing), 회복력(resilience) 간에 자본을 할당하며, ESG를 중시하는 투자자들이 이 기업 행태에 영향을 미칩니다.

- **Performance Highlights**: ESG를 중시하는 충분한 자본을 갖춘 투자자 없이 기업의 완화 노력은 제한적이며, 투자자 다수가 ESG를 우선시할 때 기업 협력이 증가하고 기후 위험이 감소하여 장기적인 재무 안정성이 향상되는 것을 보여줍니다. 또한, 기후 위험에 대한 정보 제공은 투자자의 개입 없이도 기업이 더 많은 완화를 투자하도록 유도합니다.



### Fair Secretaries with Unfair Predictions (https://arxiv.org/abs/2411.09854)
Comments:
          to appear at NeurIPS 2024

- **What's New**: 본 논문에서는 예측(predictions)의 편향(bias)이 학습 증강 알고리즘(learning-augmented algorithms)의 공정성(fairness)에 미치는 영향을 조사하며, 불공정한 예측을 활용하여 공정한 알고리즘을 설계할 수 있는지를 탐구합니다.

- **Technical Details**: 기존의 비슷한 연구들과는 다르게, 본 논문에서는 'pegging'이라는 새로운 아이디어를 바탕으로 알고리즘을 설계합니다. 이 연구는 예측값을 사용한 비즈니스 의사결정의 대표적인 사례로 비서 문제(secretary problem)를 중심으로 진행됩니다. 연구는 알고리즘이 적어도 최상의 후보를 수용할 확률을 Ω(1)로 보장하는 동시에, 예측 오류에 따라 기대값(expected value)을 최대화하는 방법을 제시합니다.

- **Performance Highlights**: 제안하는 알고리즘은 예측 오류가 낮을 때 기대값이 1에 가까워지는 보증을 제공하며, k-비서 문제(k-secretary problem)로 확장할 수 있어 추가적인 기술적 기여를 합니다. 실험 결과는 알고리즘의 이론적 분석을 보완하여 성능을 입증합니다.



### Towards a Fairer Non-negative Matrix Factorization (https://arxiv.org/abs/2411.09847)
- **What's New**: 본 논문에서는 비부정적 행렬 분해(Non-negative Matrix Factorization, NMF)가 인구 통계학적 특성이나 보호된 속성에 의해 정의된 데이터 그룹의 표현에 어떻게 편향을 도입할 수 있는지를 연구하고 있습니다. 이를 해결하기 위해, Fairer-NMF라는 접근 방식을 제안하며, 다양한 그룹에 대해 최대 재구성 손실을 최소화하려고 합니다.

- **Technical Details**: Fairer-NMF는 두 가지 알고리즘을 포함합니다: 대안 최소화(Alternating Minimization, AM) 방법과 곱셈적 업데이트(Multiplicative Updates, MU) 방법입니다. AM은 표준 절차이지만 MU는 계산 시간을 단축시키면서도 비슷한 성능을 보이는 장점을 가지고 있습니다. 이 논문은 이 두 알고리즘의 동기, 특성 및 수치적 성능을 논의합니다.

- **Performance Highlights**: 수치 실험을 통해 Fairer-NMF가 표준 NMF에 비해 데이터 셋에서 어떻게 더 공정한 결과를 도출하는지를 보여주며, 이 방법의 성능과 거래 관계를 논의합니다. 결론적으로, 공정성이 특정 응용에 따라 다르며 미세 조정이 필요하다는 점을 강조합니다.



### Deep Autoencoders for Unsupervised Anomaly Detection in Wildfire Prediction (https://arxiv.org/abs/2411.09844)
Comments:
          33 pages, 18 figure, 16 tables. To appear in Earth and Space Science

- **What's New**: 이 연구는 고전적인 지도학습(supervised learning) 및 비지도학습(unsupervised learning) 접근법과 달리, 비지도학습을 활용하여 wildfire 예측의 격차를 해소하고자 하였습니다. 특히, autoencoder 및 clustering 기법을 사용하여 이상 탐지(anomaly detection)에 초점을 맞추었습니다.

- **Technical Details**: 연구는 2005년부터 2021년까지의 호주의 역사적 날씨 데이터와 Normalised Difference Vegetation Index (NDVI) 데이터를 이용하여 두 가지 비지도 방법을 분석했습니다. 첫 번째 방법은 Deep Autoencoder를 통해 잠재 특징(latent features)을 얻고, 이를 anomaly detection을 위해 isolation forest, local outlier factor, 및 one-class SVM에 입력하는 것이었습니다. 두 번째 방법은 입력 데이터를 재구성하여 재구성 오류를 통해 이상을 식별하는 것입니다. Long Short-Term Memory (LSTM) autoencoder와 fully connected (FC) autoencoder가 사용되었으며, FC autoencoder가 0.71의 정확도(accuracy), 0.74의 F1 점수(F1-score), 그리고 0.42의 MCC를 기록했습니다.

- **Performance Highlights**: 이 연구의 결과는 비지도학습 방법이 ground truth가 없는 상황에서 효과적으로 wildfires를 예측할 수 있음을 보여줍니다. FC autoencoder는 비교 대상 모델을 초월하여 좋은 성능을 보였고, 이로 인해 비지도학습 기법의 실용성에 대한 강력한 근거가 마련되었습니다.



### FedRewind: Rewinding Continual Model Exchange for Decentralized Federated Learning (https://arxiv.org/abs/2411.09842)
- **What's New**: FedRewind는 분산된 연합 학습(federated learning)에서 데이터 분포 변화 문제 해결을 위해 노드 간 모델 교환을 활용한 새로운 접근 방식을 제안합니다. 본 연구는 지속적인 학습(continual learning) 원칙과 인지 신경 과학(cognitive neuroscience) 이론을 적용하여 메모리 유지 문제를 다루고 있습니다.

- **Technical Details**: FedRewind는 분산된 라우팅 메커니즘을 구현하여 노드들이 연합 내 다른 노드와 모델을 주기적으로 주고 받으며 훈련하는 방법입니다. 각 노드는 로컬 훈련 중 특정 횟수만큼 모델을 재전송(rewind)하여 데이터 간의 분포 변화를 줄입니다. 이 방법은 모델의 과적합(overfitting)을 방지하고 메모리 인출(active recall)을 통해 학습 성능을 향상시킵니다.

- **Performance Highlights**: 다수의 벤치마크 데이터셋을 활용한 평가 결과, FedRewind는 기존의 분산 연합 학습 방식 및 특정 라우팅 스킴을 적용한 방법들에 비해 우수한 성능을 보였습니다. 또한, 데이터가 공간 및 시간에 따라 변화하는 연합 지속 학습(federated continual learning) 문제를 효과적으로 해결해 낸 점이 주목할 만합니다.



### Real-time Adapting Routing (RAR): Improving Efficiency Through Continuous Learning in Software Powered by Layered Foundation Models (https://arxiv.org/abs/2411.09837)
- **What's New**: 이 논문에서는 Real-time Adaptive Routing (RAR)이라는 새로운 접근 방식을 제안하며, 이를 통해 Foundation Model (FM)의 라우팅 결정을 지속적으로 적응시키는 방법을 소개합니다. 기존의 라우팅 모델들과 달리, 이 방법은 비용이 많이 드는 더 강력한 FM에 대한 의존도를 낮추는 것을 목표로 합니다.

- **Technical Details**: Real-time Adaptive Routing (RAR) 접근 방식은 guided in-context learning을 활용하여 더 약한 FM의 능력을 향상시키고, 요청을 FM에 따라 효과적으로 라우팅합니다. 기존의 모델들은 최적의 라우팅 결정을 위해 정교하게 준비된 데이터를 학습하는 데 의존하고 있으며, 업데이트 시 복잡한 계산을 요구합니다.

- **Performance Highlights**: 우리의 접근 방식을 다양한 MMLU 벤치마크의 하위 집합에서 평가한 결과, 컴퓨팅 비용이 높은 모델에 대한 요청을 50.2% 줄이는 동시에 응답 품질의 약 90.5%를 유지했습니다. 또한, 강력한 모델에서 생성된 가이드는 특정 도메인 내에서 일반화가 이루어져, 단독으로 약한 FM을 사용할 때보다 더 나은 응답 품질을 보여주었습니다.



### The Good, The Efficient and the Inductive Biases: Exploring Efficiency in Deep Learning Through the Use of Inductive Biases (https://arxiv.org/abs/2411.09827)
Comments:
          PhD Dissertation

- **What's New**: Deep Learning의 새로운 효율성과 지속 가능성 문제를 해결하기 위한 유도 편향(inductive biases)의 역할을 분석한 연구입니다.

- **Technical Details**: 이 논문은 두 가지 주요 부분으로 나뉘어 있습니다. 첫 번째 부분에서는 지속적인 모델링(continuous modeling)을 통해 Deep Learning 알고리즘의 효율성을 개선하는 방법을 조사합니다. 두 번째 부분은 데이터의 고유한 대칭(symmetry)에 맞춘 신경 연산을 설계하여 효율성을 높이는 대칭 보존(symmetry preservation)의 역할을 다룹니다.

- **Performance Highlights**: 지속적인 모델링은 시간과 메모리에서의 계산 효율(computational efficiency), 매개변수 효율(parameter efficiency), 신경 아키텍처 설계의 복잡성(design efficiency) 등이 크게 향상되는 것으로 나타났습니다. 대칭 보존을 통해 데이터 및 매개변수 효율성에서 상당한 향상을 이루었지만, 그에 따른 계산 비용 증가라는 트레이드오프(trade-off)가 있음을 지적합니다.



### Automatic Classification of General Movements in Newborns (https://arxiv.org/abs/2411.09821)
Comments:
          Findings paper presented at Machine Learning for Health (ML4H) symposium 2024, December 15-16, 2024, Vancouver, Canada, 6 pages

- **What's New**: 이번 연구는 아기들의 일반 움직임(General Movements, GMs)을 자동으로 분류하기 위한 기계 학습 알고리즘을 소개합니다. 이 알고리즘은 아기의 비디오 기록에서 신체 움직임을 분석하고 질적으로 평가하는 데 도움을 줄 수 있습니다.

- **Technical Details**: 본 연구는 Barmherzige Brüder Regensburg 병원에서 수집된 76개의 아기 비디오 녹화를 활용하여 신체의 주요 해부학적 지점을 라벨링하고 추적합니다. 이후, 추적된 지점으로부터 특징(feature)을 추출하고, 1D-CNN 및 LSTM과 같은 다양한 기계 학습 분류 모델을 사용하여 GMs의 질을 분류합니다.

- **Performance Highlights**: 연구에서 제안된 접근법은 비디오 길이, 장비 종류, 녹음 환경의 변동성을 고려하여 개발되었습니다. 초기 결과에 따르면, 자동 GMA를 통해 효과적인 아기 신경발달 장애의 조기 발견이 가능할 것으로 기대됩니다.



### WelQrate: Defining the Gold Standard in Small Molecule Drug Discovery Benchmarking (https://arxiv.org/abs/2411.09820)
Comments:
          * denotes equal contribution

- **What's New**: 이번 논문에서는 작고 분자(molecule) 약물 발견(drug discovery) 벤치마킹(benchmarking) 을 위한 새로운 기준선인 WelQrate를 제안합니다. 이 기준선 구축을 통해 AI 커뮤니티가 약물 발견에 기여할 수 있는 효과적인 평가 프레임워크를 세우고자 합니다.

- **Technical Details**: WelQrate는 5가지 치료적(target) 대상 클래스에 걸쳐 9개의 데이터셋을 포함하는 데이터셋 수집 작업과, 표준화된 모델 평가 프레임워크를 포함합니다. 이 평가 프레임워크는 고품질 데이터셋, 특성화(featurization), 3D 형태 생성(conformation generation), 평가 메트릭스(metrics), 데이터 분할(data splits)을 고려하여 실세계 가상 스크리닝을 수행하는 약물 발견 전문가들에게 신뢰할 수 있는 벤치마킹을 제공합니다. 또한, PAINS(패닉을 유발하는 화합물) 필터링과 같은 철저한 데이터 전처리(preprocessing)을 포함하여 데이터의 질을 보증합니다.

- **Performance Highlights**: WelQrate 데이터셋 수집을 통한 다양한 연구 질문을 통해 모델 성능을 평가하였고, 다양한 모델, 데이터셋 품질, 특성화 방법 및 데이터 분할 전략의 영향을 탐색했습니다. 이 결과를 바탕으로 WelQrate를 작고 분자 약물 발견 벤치마킹의 금본위로 채택할 것을 권장합니다.



### Learning Parameter Sharing with Tensor Decompositions and Sparsity (https://arxiv.org/abs/2411.09816)
- **What's New**: 이 논문에서는 Fine-grained Parameter Sharing (FiPS)라는 혁신적인 알고리즘을 소개하며, 이는 대규모 비전 트랜스포머 모델을 효과적으로 압축하기 위해 매개변수 공유, 텐서 분해(tensor decomposition), 희소성(sparsity) 간의 관계를 활용합니다.

- **Technical Details**: FiPS는 다층 감지기(MLP) 모듈 간에 공유 뉴런을 표현하기 위해 공유 기준(shared base)과 희소 요소(sparse factors)를 활용합니다. 공유 매개변수화는 특이값 분해(SVD)를 통해 초기화되며, 블록 단위 복원 오차(block-wise reconstruction error)를 최소화함으로써 최적화됩니다.

- **Performance Highlights**: FiPS는 DeiT-B와 Swin-L MLP를 각각 원래 매개변수 수의 25~40%로 압축하면서도, 원본 모델과의 정확도 차이를 1% 이하로 유지합니다.



### Evaluating Loss Landscapes from a Topology Perspectiv (https://arxiv.org/abs/2411.09807)
- **What's New**: 신경망의 손실 구조를 정량적으로 분석하기 위해 Topological Data Analysis(TDA)을 활용하였습니다. 기존의 손실 경관 시각화에서 벗어나, 손실 경관의 위상적 특성을 정량화하고 비교하여 신경망의 성능과 학습 동역학에 대한 새로운 통찰을 제공합니다.

- **Technical Details**: 본 연구는 손실 경관을 TDA의 merge tree와 persistence diagram을 통해 분석합니다. 손실 경관의 형태를 정량화하기 위해 saddle point와 average persistence를 측정하고, Hessian과 관련된 지표와 결과를 비교합니다. 특히, ResNet과 물리 기반 신경망 같은 이미지 패턴 인식 모델을 사용하여 실험을 진행하였습니다.

- **Performance Highlights**: ResNet-20 모델에서 residual connections을 제거했을 때 손실 경관의 구조가 변화하였고, 이는 모델의 성능에도 영향을 미쳤습니다. Residual connection이 없는 경우 평균 정확도가 90%로, 있는 경우 92%로 나타났으며, 손실 경관의 복잡성이 증가하는 것으로 나타났습니다.



### Fair Resource Allocation in Weakly Coupled Markov Decision Processes (https://arxiv.org/abs/2411.09804)
- **What's New**: 이번 논문에서는 약한 결합 Markov 결정 프로세스(Weakly Coupled Markov Decision Processes, WCMDPs)에서 공정한 자원 할당을 고려하며, 일반화된 Gini 함수(Generalized Gini Function)를 활용한 공정성 정의를 도입합니다. 기존의 총합(utilitarian) 목표 대신에 이러한 정의를 통해 공정성을 평가합니다.

- **Technical Details**: 우리는 이 논문에서 공정성 문제를 해결하기 위해 선형 프로그래밍(Linear Programming)을 기반으로 한 일반적인 솔루션을 제안하면서 동질적인 경우(homogeneous case)에 집중합니다. 모든 하위 MDP(Sub-MDP)들이 동일할 때, 문제는 '퍼뮤테이션 불변(permutation invariant)' 정책에서 공리적 목표(utilitarian objective)를 최적화하는 것으로 줄여진다는 점을 최초로 보여줍니다. 이 결과는 restlessness bandits 설정에서 Whittle 인덱스 정책을 활용할 수 있게 해주며, 보다 일반적인 설정에서는 카운트 비율 기반의 심층 강화 학습(Deep Reinforcement Learning) 접근법을 소개합니다.

- **Performance Highlights**: 실험을 통해 제안된 방법의 효과성을 검증했으며, GGF 최적성, 유연성, 확장성 및 효율성을 입증했습니다. 또한, RMABs(restless multi-arm bandits) 모델링된 기계 교체(application)에서 Whittle 인덱스 정책과 비교하여 다양한 설정에서 다양한 공정한 결과를 도출하는 것이 효과적임을 보여주었습니다.



### Modeling human decomposition: a Bayesian approach (https://arxiv.org/abs/2411.09802)
- **What's New**: 이번 연구에서는 인간 사체의 분해 과정을 규명하기 위해 다양한 환경 및 개별적 변수를 반영한 생성적 확률 모형을 개발했습니다. 이 모형은 관찰된 분해 특성과 관련된 변수를 바탕으로 사후 간격 (PMI)을 추정할 수 있는 도구로서의 역량을 보여줍니다.

- **Technical Details**: 연구팀은 GeoFOR 데이터셋에서 수집한 2,529개의 사례를 사용해 모델을 적합시키고, 24개의 분해 특성을 정확하게 예측함으로써 ROC AUC 점수 0.85를 달성했습니다. Bayesian (베이지안) 추론 기법을 활용해 관찰된 분해 특성과 환경 및 개별적 변수를 통해 PMI를 예측했으며, R-squared 수치는 71%에 이릅니다.

- **Performance Highlights**: 이번 연구에서 제안한 모델은 향후 실험 설계를 최적화하고, 새로운 정보의 최대화를 위한 기대 정보 증가(Expected Information Gain) 접근 방식을 통해 부패 메커니즘에 대한 통찰력을 제공할 수 있는 기반을 마련합니다.



### Adversarial Attacks Using Differentiable Rendering: A Survey (https://arxiv.org/abs/2411.09749)
- **What's New**: 이번 논문은 differentiable rendering 기법이 3D 객체 및 장면을 조작함으로써 딥 신경망(DNN)을 혼란스럽게 만드는 현실적이고 물리적으로 타당한 적대적 공격을 생성할 수 있는 가능성을 탐구합니다. 이 연구는 기존의 다양한 기법을 체계적으로 정리하고, 연구의 공백을 밝혀내며, 향후 연구 방향을 제시하는 종합적인 프레임워크를 도입합니다.

- **Technical Details**: Differentiable rendering이 DNN의 취약점을 공격하기 위한 방법으로 어떻게 활용될 수 있는지를 분석합니다. 본 프레임워크는 해상도 있는 색 텍스처 조작, 조명 변경, 3D 메시 수정과 같은 구체적인 작업을 분류하여 단일 구조 내에서 관리합니다. 이를 통해 공격자의 목표와 사용할 수 있는 기술을 효과적으로 연결 지을 수 있습니다.

- **Performance Highlights**: 이 연구는 공격 시나리오 및 접근 수준을 포함하여 다양한 DNN 모델(예: 이미지 분류, 얼굴 인식, 객체 탐지)에 대해 어떻게 공격이 수행되는지를 명확하게 설명합니다. 기존의 연구보다 현장 공격에서 differentiable rendering 기술 사용의 넓은 영향력을 보여 주며, 향후 연구를 위한 중요한 방향성을 제시합니다.



### Modeling AdaGrad, RMSProp, and Adam with Integro-Differential Equations (https://arxiv.org/abs/2411.09734)
Comments:
          22 pages

- **What's New**: 본 논문에서는 AdaGrad, RMSProp 및 Adam 최적화 알고리즘의 연속 시간 공식화를 제안하고, 이를 1차적 인테그로-미분 방정식(first-order integro-differential equations)으로 모델링합니다.

- **Technical Details**: 이 연구는 연속 확장에서 적응적 최적화 방법의 이론적 이해를 심화하기 위한 접근 방식을 탐구합니다. 알고리즘의 메모리 효과는 비지역적(nonlocal) 항으로 캡슐화되며, 고전적인 미분 방정식 기술을 사용하여 수렴과 안정성을 분석합니다.

- **Performance Highlights**: 수치 시뮬레이션을 통해 연속 모델과 원래의 이산(discrete) 구현 간의 행동이 강하게 일치하는 것을 입증하였습니다. 결과적으로 이 새로운 관점은 적응적 최적화 방법에 대한 이해를 깊게 합니다.



### To bootstrap or to rollout? An optimal and adaptive interpolation (https://arxiv.org/abs/2411.09731)
- **What's New**: 본 논문에서는 부트스트래핑(bootstrapping)과 롤아웃(rollout) 방식의 장점을 결합한 새로운 벨만 연산자(Bellman operators) 클래스인 서브그래프 벨만 연산자(subgraph Bellman operators)를 소개합니다.

- **Technical Details**: 우리의 추정기는 경험적 서브그래프 벨만 연산자(empirical subgraph Bellman operator)의 고정점(fixed point)을 해결하여 도출됩니다. 이 추정기는 부트스트래핑 기반의 시간 차(différence temporelle) 추정기와 롤아웃 기반의 몬테 카를로(Monte Carlo) 방법의 강점을 통합합니다. 구체적으로 우리의 추정기의 오차 상한(error upper bound)은 TD에 의해 달성되는 최적 분산(optimal variance)에 접근하며, 선택한 상태 공간의 하위 집합의 종료 확률(exit probability)에 따라 추가 항이 존재합니다. 또한 이 추정기는 MC의 유한 샘플 적응성(finfinite-sample adaptivity)을 보이며, 샘플 복잡성(sample complexity)은 해당 하위 집합의 점유율(occupancy measure)만에 의존합니다.

- **Performance Highlights**: 우리의 상한은 정보 이론적(lower bound) 관점에서 보완되어 추가 항이 합리적인 샘플 크기(sample size) 하에서 피할 수 없음을 보여줍니다. 이러한 결과들은 서브그래프 벨만 추정기가 정책 평가(policy evaluation)에서 TD와 MC 방법을 화합하는 최적의 적응적 프레임워크로 자리 잡도록 합니다.



### SureMap: Simultaneous Mean Estimation for Single-Task and Multi-Task Disaggregated Evaluation (https://arxiv.org/abs/2411.09730)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문에서는 머신러닝 모델의 다양한 하위 집단에 대한 성과 평가에 대한 새로운 접근 방식을 제시한다. 특히, 여러 고객이 동일한 AI 모델을 사용하는 환경에서 각 고객의 데이터를 기반으로 한 다중 작업(disaggregated evaluation) 접근 방식을 연구하였다.

- **Technical Details**: 연구에서 제안하는 유도 평가 방법 SureMap은 다중 작업과 단일 작업에 대한 평가에서 높은 추정 정확도를 제공하며, 문제를 구조화된 동시 가우시안 평균 추정 문제로 변환하고 외부 데이터를 통합하여 효율성을 높인다. 이 방법은 최대 사후 확률 추정(maximum a posteriori, MAP)과 Stein의 비편향 위험 추정(SURES)를 기반으로 한다.

- **Performance Highlights**: SureMap은 다양한 도메인에서의 하위 집단 평가 작업에 대해 많은 강력한 경쟁자들에 비해 상당한 정확도 향상을 보여주었다. 특히, 다중 클라이언트 데이터를 통합할 경우 전반적으로 모든 평가 설정에서 주요 개선이 관찰되었다.



### Physics-informed neural networks (PINNs) for numerical model error approximation and superresolution (https://arxiv.org/abs/2411.09728)
- **What's New**: 본 논문에서는 physics-informed neural networks (PINNs)를 사용하여 수치 모델 오류를 동시 해결하는 방법을 제안합니다. 이 연구는 최근 머신러닝(ML)의 발전을 바탕으로 하여 모델 오류를 구체적으로 정량화할 수 있는 방법의 부족을 해결하고자 합니다.

- **Technical Details**: 제안된 방법은 이차원 탄성 플레이트와 같은 유한 요소 해석 모델을 통해 얻은 수치 데이터를 바탕으로 합니다. 4-노드 및 8-노드 사각형 요소를 사용하여 모델을 분산하였으며, PINNs가 x 및 y 변위 필드에서 모델 오류를 효과적으로 예측함을 보여주었습니다. 또한, PINNs가 순수한 데이터 기반 접근 방식을 넘어서서 모델 오류를 근사화할 수 있도록 하는 물리 정보 손실 함수를 통합하고 있습니다.

- **Performance Highlights**: 정확한 예측 결과에 따르면 모델 오류 예측에서 예측값과 실제값 간의 차이가 작았고, 이는 PINNs가 유한 요소 모델링에서 발생하는 오류를 효과적으로 관리할 수 있음을 보여줍니다.



### Iterative Batch Reinforcement Learning via Safe Diversified Model-based Policy Search (https://arxiv.org/abs/2411.09722)
Comments:
          Workshop on Safe and Robust Robot Learning for Operation in the Real World (SAFE-ROL) at CoRL 2024

- **What's New**: 이번 논문에서는 반복적인 오프라인 강화 학습(iterative batch reinforcement learning, IBRL) 프레임워크를 제안합니다. 이는 기존에 수집된 데이터를 기반으로 정책을 지속적으로 개선하는 접근법으로, 안전성과 다양성 기준을 포함하여 실시간 데이터 수집을 효율적으로 안내합니다.

- **Technical Details**: IBRL은 앙상블 기반의 모델 기반 정책 탐색(model-based policy search) 기술을 활용하며, 안전 제약과 데이터 다양성을 동시에 고려합니다. 본 방법론은 이전에 훈련된 다양한 정책을 사용하여 새로운 데이터를 수집하고, 이를 기존의 데이터 세트에 추가하여 여러 번의 반복적인 학습을 수행하도록 설계되었습니다.

- **Performance Highlights**: 실험을 통해 2D 환경 및 Industrial Benchmark에서 제안하는 IBRL 접근법이 탐색 능력과 성능을 향상시킴을 입증하였으며, 안전성을 유지하면서 정책의 성능이 향상됨을 보여주었습니다.



### Early-Scheduled Handover Preparation in 5G NR Millimeter-Wave Systems (https://arxiv.org/abs/2411.09720)
- **What's New**: 본 논문은 5G NR 시스템에서 머신러닝을 활용하여 핸드오버(Handover, HO) 프로세스의 효율성을 개선하는 새로운 Early-Scheduled Handover Preparation 스킴을 소개합니다. 이는 밀리미터파(mmWave) 환경에서 고속 이동성과 밀집된 소형 셀을 포함한 상황에서 HO 프로세스의 로버스트성과 효율성을 향상시키기 위해 설계되었습니다.

- **Technical Details**: 본 연구는 Massive MIMO 시스템을 활용하여 채널 행동에 대한 정밀 데이터를 수집하고, 머신러닝 기법을 적용하여 HO 이벤트를 예측합니다. 이 과정에서 HO 준비 단계의 타이밍을 최적화하여 HO 실행에 필요한 시간을 줄이고, 채널 품질 저하를 감소시키는 새로운 조기 트리거를 식별하였습니다.

- **Performance Highlights**: Early-Scheduled Handover Preparation 기법은 핸드오버 실행 시간을 단축시켜, 핸드오버 성공률을 높이며, 밀접한 신호 품질 관리가 필요한 시나리오에서 효과적입니다. 연구 결과, 제안된 방법이 기존 방법보다 더 나은 성과를 보여주는 것을 입증했습니다.



### The Spatial Complexity of Optical Computing and How to Reduce I (https://arxiv.org/abs/2411.10435)
- **What's New**: 이 논문은 광학 컴퓨팅 시스템의 '공간 복잡도'를 연구하며, 비선형적인 작업을 구현하기 위한 새로운 공간 효율적인 신경형 광학 설계를 제안합니다. 이는 파동 물리학의 개념을 기반으로 한 구조적 희소성 제약과 신경 가지치기 방법에서 영감을 얻었습니다.

- **Technical Details**: 저자들은 광학 시스템의 볼륨 크기와 정확도의 상관관계를 조사하며, 이론적 모델을 통해 커뮤니케이션 콘이 실제로 필요한 두 개의 개별 채널 사이의 교차를 설명합니다. 이들은 특정 기능을 수행하기 위해 요구되는 두께(t)와 관련하여 비선형성을 정의하고, 단일 모드 광섬유를 예로 들어 비선형성을 고려한 두께 감소 방법을 제안합니다.

- **Performance Highlights**: 제안된 방법은 기존 설계의 1%-10% 크기로 시스템을 줄일 수 있는 가능성을 보여줍니다. 이론적 결과는 정확도의 회복이 구조적 크기의 증가에 따라 감소하는 경향이 있음을 나타내어 광학 컴퓨팅의 궁극적인 한계를 해석하는 새로운 시각을 제공합니다.



### Private Counterfactual Retrieval With Immutable Features (https://arxiv.org/abs/2411.10429)
- **What's New**: 본 논문에서는 사용자가 자신의 속성 벡터를 변경하지 않고, 거부된 특성 벡터에 대해 가장 가까운 반사실적( counterfactual) 샘플을 데이터베이스에서 비공식적으로 검색하는 문제인 불변(private counterfactual retrieval, I-PCR) 문제를 소개합니다. 기존의 이론과는 달리 사용자가 자신의 기본 속성(immutable set)과 결과 반사실적에 대해 정보 이론적으로 비공식권을 유지하도록 보장합니다.

- **Technical Details**: 제안된 두 가지 I-PCR 체계는 개인 정보 검색(private information retrieval, PIR) 기법을 활용하여, 사용자가 고정된 속성을 유지하면서 반사실적 샘플의 인덱스를 효율적으로 검색할 수 있도록 설계되었습니다. 시스템 모델은 d차원 특징 벡터를 입력으로 받아 교차 점에 대해 분류하는 머신러닝 모델을 가정하며, 이는 다수의 비공식 서버에서 관리됩니다.

- **Performance Highlights**: 싱글-페이즈 I-PCR이 두-페이즈 I-PCR보다 통신 비용이 낮지만, 데이터베이스에서 더 많은 정보가 누출될 수 있음을 보여줍니다. 또한, 제안된 기법의 유용성을 평가하기 위해 실제 데이터 집합에서 런타임 분석을 수행했습니다.



### Deep Learning for Micro-Scale Crack Detection on Imbalanced Datasets Using Key Point Localization (https://arxiv.org/abs/2411.10389)
- **What's New**: 이 논문은 구조적 건강 모니터링 영역에서 내부 균열 탐지에 대한 심층 학습(Deep Learning, DL) 방법의 새로운 적용을 탐구합니다. 특히 마이크로 스케일 균열을 식별하기 위해 DL 기반의 키 포인트 탐지 기술을 사용하여, 균열의 경계를 정의하는 네 개의 키 포인트 좌표를 예측함으로써 균열을 국지화하는 방식을 제시합니다.

- **Technical Details**: 이 연구에서는 Inception 모듈을 포함하는 Wide Convolutional Networks을 사용하여 수치 데이터에서 균열을 감지하기 위한 새로운 접근법을 제시합니다. 모델은 다양한 필터 크기로 구성된 여러 컨볼루션 레이어를 사용하여 빠르고 효율적으로 다양한 특징을 추출합니다. 또한, Attention Mechanisms을 도입해 특징 맵 내에서 중요 영역에 집중하여 성능을 향상시킵니다.

- **Performance Highlights**: 모델은 마이크로 스케일 균열 탐지에 적용되었으며, 실제 균열 위치와 예측된 균열 위치 간 평균 Intersection over Union (IoU) 값이 0.511로 쉬운 미세 균열 및 0.631로 큰 균열에 대해 나타났습니다. 이는 균열 탐지에서 이전의 Deep Learning 모델보다 성능 향상을 보여줍니다.



### Comparative Analysis of Machine Learning Approaches for Bone Age Assessment: A Comprehensive Study on Three Distinct Models (https://arxiv.org/abs/2411.10345)
- **What's New**: 이 연구에서는 아동과 유아의 X-ray 영상을 이용하여 뼈 나이를 예측하는 자동화 과정의 정확성을 높이기 위해 세 가지 주요 기계 학습 모델인 Xception, VGG, CNN 모델을 비교 분석하였습니다.

- **Technical Details**: 모델들은 사전 처리된 데이터 세트를 기반으로 훈련되었으며, 각 모델의 정확도는 월 단위의 평균 절대 오차(MAE)를 사용하여 측정되었습니다. 세 가지 모델의 성능을 비교함으로써 적합한 모델의 선택을 위한 기초 자료를 제공하고자 하였습니다.

- **Performance Highlights**: Xception, VGG, CNN 모델은 모두 정확성과 관련 요소에 대해 테스트되었으며, 각 모델 간의 성능 차이가 확인되었습니다. 각 모델의 특징에 따라 선택할 수 있는 다양한 옵션이 제공됩니다.



### RETR: Multi-View Radar Detection Transformer for Indoor Perception (https://arxiv.org/abs/2411.10293)
Comments:
          24 pages, Accepted to NeurIPS 2024

- **What's New**: 이 논문에서는 다중 뷰 레이더 인식을 위해 개발된 Radar dEtection TRansformer(RETR)를 제안합니다. 이는 기존의 DETR 아키텍처를 기반으로 하며, 레이더 신호의 고유한 특성을 고려한 설계를 포함하고 있습니다.

- **Technical Details**: RETR는 다음과 같은 주요 구성 요소를 갖추고 있습니다: 1) 조정 가능한 위치 인코딩(Tunable Positional Encoding, TPE)을 통해 깊이 우선 피쳐 유사도를 강화; 2) 레이더와 카메라 좌표계에서의 삼면 손실(tri-plane loss)을 도입; 3) 재매개변수를 통한 학습 가능한 레이더-카메라 변환을 실현하여 다중 뷰 레이더 상황에 맞는 특성을 반영합니다.

- **Performance Highlights**: RETR는 HIBER 및 MMVR 데이터셋에 대해 평가되었으며, 객체 탐지에서 기존 최첨단 방법보다 15.38 이상의 AP(평균 정밀도)를 향상시키고, 인스턴스 세분화에서 11.77 이상의 IoU(교차 영역 비율)를 달성했습니다.



### The ParClusterers Benchmark Suite (PCBS): A Fine-Grained Analysis of Scalable Graph Clustering (https://arxiv.org/abs/2411.10290)
Comments:
          This is a preliminary version of a paper that will appear at VLDB'25

- **What's New**: ParClusterers Benchmark Suite (PCBS)를 소개합니다. 이는 다양한 고도로 확장 가능한 병렬 그래프 클러스터링 알고리즘과 벤치마킹 도구를 포함하여, 서로 다른 그래프 클러스터링 알고리즘 및 구현을 비교하는 데 도움을 줍니다.

- **Technical Details**: PCBS는 커뮤니티 탐지, 분류, 밀집 부분 그래프 마이닝 등 현대의 다양한 클러스터링 사용 사례를 목표로 하는 알고리즘을 포함합니다. 벤치마크 도구 키트는 다양한 클러스터링 알고리즘의 여러 인스턴스를 실행하고 평가하기 쉽게 만들어 주며, 이를 통해 특정 작업의 클러스터링 성능을 미세 조정하고 서로 다른 클러스터링 알고리즘을 비교할 수 있습니다.

- **Performance Highlights**: PCBS를 활용한 결과, 상위 품질 결과는 많은 인기 그래프 클러스터링 도구 키트에 포함되지 않은 알고리즘에서 얻어졌습니다. 클러스터링 구현의 속도도 향상되어 PCBS는 Neo4j보다 평균 32.5배, TigerGraph보다 303배 빠릅니다. 특히, Correlation Clustering은 네 가지 작업 중 세 가지에서 최고의 품질을 보여주었습니다.



### Multidimensional Byte Pair Encoding: Shortened Sequences for Improved Visual Data Generation (https://arxiv.org/abs/2411.10281)
- **What's New**: 본 논문에서는 비주얼 데이터(visual data)의 토큰화(tokenization) 방법을 개선하기 위한 새로운 접근 방식을 제안합니다. 특히, 1D 토큰 압축에서 Byte Pair Encoding (BPE) 개념을 다차원(multidimensional)으로 확장하여, 더 짧고 정보가 고르게 분포된 시퀀스를 생성합니다.

- **Technical Details**: Multidimensional Byte Pair Encoding (MDBPE)이라는 새로운 압축 방법을 통해 1D를 넘어서 다양한 차원에서 이미지 데이터를 처리할 수 있습니다. 이 방법은 토큰 쌍의 출현 빈도를 세고, 가장 빈번한 토큰 쌍을 새 토큰으로 대체하는 과정으로 이루어집니다.

- **Performance Highlights**: 실험 결과, MDBPE를 통해 생성된 압축된 시퀀스는 트랜스포머(transformers)의 학습 및 추론 성능을 향상시켰으며, 이미지넷(ImageNet)과 같은 대규모 데이터셋에도 적용 가능함을 보였습니다. 더불어, 논문에 제시된 코드는 Jupyter Notebooks 형식으로 제공되며, C++ 구현체도 마련되어 있습니다.



### Scaling Law for Post-training after Model Pruning (https://arxiv.org/abs/2411.10272)
- **What's New**: 이번 논문은 깊이 가지치기(depth pruning), 너비 가지치기(width pruning), 및 2:4 반구조화 가지치기(semi-structured pruning)를 통해 가지치기된 대규모 언어 모델(LLMs)의 후속 훈련(post-training) 요구 사항을 조사하고, 최적의 후속 훈련 데이터 양을 결정하기 위한 스케일링 법칙(scaling law)을 제시합니다.

- **Technical Details**: 연구에서는 Llama-3 및 Qwen-2.5 모델 시리즈에 대해 후속 훈련 실험을 수행하며, 높은 가지치기 비율이 성능 회복을 위해 더 많은 후속 훈련 데이터를 필요로 한다는 것을 발견했습니다. 특히, 더 큰 LLM은 더 적은 데이터로 성능을 회복할 수 있다는 점에서 기존의 직관과 반대된다는 것을 확인했습니다. 제안된 스케일링 법칙은 가지치기 전후의 모델 파라미터 수 및 후속 훈련 토큰 수를 바탕으로 모델의 손실(loss)을 예측할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과에 따르면, Llama-3.1-8B 모델은 16% 가지치기 비율에서 약 1B 토큰에 대해 손실 곡선이 수렴하는 반면, 24% 및 33% 가지치기 비율에서는 더 많은 후속 훈련 데이터가 필요하다는 점이 강조되었습니다. 또한, 2:4 반구조화 가지치기를 적용한 경우 더 큰 모델은 성능 회복을 위해 상대적으로 적은 후속 훈련 데이터를 요구한다는 것을 보여주었습니다.



### MDHP-Net: Detecting Injection Attacks on In-vehicle Network using Multi-Dimensional Hawkes Process and Temporal Mod (https://arxiv.org/abs/2411.10258)
- **What's New**: 이 논문에서는 현대 차량의 인-차량 네트워크(IVN)에 대한 사이버 공격의 한 특정 유형인 injection attack을 분석하고, 이를 탐지하기 위한 새로운 방법론을 제안합니다.

- **Technical Details**: 논문에서는 Multi-Dimensional Hawkes Process (MDHP)를 사용하여 injection attack의 시간적 자극(e excitation) 효과를 모델링합니다. 또한, MDHP에 최적화된 파라미터를 정확하게 추정하기 위해 MDHP-GDS라는 경량화된 Gradient Descent Solver를 개발합니다. 최종적으로 MDHP-Net이라 불리는 injection attack 탐지기를 제안하여, MDHP-LSTM 블록과 통합하여 시간적 특징을 개선합니다.

- **Performance Highlights**: 제안된 MDHP-Net 구조는 표준 Long Short-Term Memory (LSTM)보다 복잡한 시간적 특징을 포착하며, 광범위한 평가를 통해 탐지 방법의 효과iveness를 입증하였습니다.



### The Unreasonable Effectiveness of Guidance for Diffusion Models (https://arxiv.org/abs/2411.10257)
Comments:
          Preprint. 19 pages, 14 figures in total, including references and appendix

- **What's New**: 이 논문에서는 새로운 에러 교정 기법인 슬라이딩 윈도우 가이던스(SWG)를 소개합니다. SWG는 주 모델의 수용 영역(receptive field)을 제어하여 고유한 방식으로 모델을 유도하며, 인간의 선호도와 더 잘 일치하면서도 훈련, 아키텍처 수정 또는 클래스 조건화가 필요 없습니다.

- **Technical Details**: SWG는 장기 공간 의존성(long-range spatial dependencies)을 강화하여 시각 품질을 개선하는 새로운 가이던스 방법입니다. 전통적인 클래스 없는 가이던스(Classifier-Free Guidance, CFG)와 달리 SWG는 추가 훈련이나 아키텍처적 변경 없이 어떤 확산 모델(Diffusion Model, DM)에도 적용할 수 있습니다. 이 방법은 특히 약한 모델 가이던스(Weak Model Guidance, WMG) 방식과 대조되며, 더 강력한 가중치 정규화(weight regularization)를 수행하여 생성된 샘플의 품질을 향상시킵니다.

- **Performance Highlights**: SWG는 최신 가이던스 기법과 비교해 경쟁력 있는 생성 성능을 발휘하며, 인간의 선호도와의 일치도가 더 높습니다. 이 방법은 어떤 DM에서도 사용 가능하고, 시각적으로 높은 품질의 이미지를 생성하는 데 효과적입니다.



### Measuring Non-Adversarial Reproduction of Training Data in Large Language Models (https://arxiv.org/abs/2411.10242)
- **What's New**: 이 논문은 대형 언어 모델이 훈련 데이터의 일부를 어떻게 암기하는지를 연구하며, 특히 비대립적 상황(일반적인 질문이나 요청)에 대한 응답 시 모델과 훈련 데이터 사이의 중복(overlap)을 정량화합니다.

- **Technical Details**: 우리가 조사한 비대립적 재생(non-adversarial reproduction)에서는 일반적인 프롬프트(prompt) 카테고리(예: 편지 작성, 튜토리얼 등)에 대해 인기 있는 대화형 언어 모델이 생성한 최대 15%의 텍스트가 인터넷의 스니펫(snippet)과 중복된다는 것을 보여줍니다.

- **Performance Highlights**: 최악의 경우, 생성된 콘텐츠의 100%가 온라인에 정확히 존재하는 경우도 확인되었습니다. 반면, 사람의 작성한 텍스트는 인터넷 데이터와 중복되는 비율이 훨씬 낮습니다. 적절한 프롬프트 사용이 평균적으로 비대립적 재생을 줄일 수 있지만, 최악의 경우 훈련 데이터 재생을 완화하기 위해서는 더 강력한 방어책이 필요하다는 것을 발견했습니다.



### Efficient Neural Hybrid System Learning and Transition System Abstraction for Dynamical Systems (https://arxiv.org/abs/2411.10240)
- **What's New**: 이번 논문은 시스템 식별 및 동적 학습을 위한 해석 가능하고 계산적으로 효율적인 뉴럴 네트워크 하이브리드 모델링 프레임워크를 제안합니다. 주 저자는 동적 시스템 모델링을 낮은 수준의 뉴럴 하이브리드 모델과 높은 수준의 전이 시스템 추상화로 나누어 연구하고 있습니다.

- **Technical Details**: 제안된 두 단계 모델링 프레임워크는 두 가지 핵심 레벨로 나뉩니다. 첫째, 낮은 수준 모델은 시스템의 지역 행동을 포착하기 위해 여러 개의 단순한 뉴럴 네트워크를 사용하여 지역 동적을 근사합니다. 둘째, 높은 수준의 전이 모델은 지역 구획 간의 관계 및 전이 패턴을 포착하는 전이 시스템을 형성합니다. 이를 위해 최대 엔트로피 분할 방법이 시스템 상태 공간을 여러 지역 서브스페이스로 나누는 데 사용됩니다.

- **Performance Highlights**: 이 연구는 뉴럴 하이브리드 시스템에 대한 분산 훈련 및 검증을 통해 계산 효율성을 높이며, 전이 관계를 조사하는 새로운 전이 시스템 추상화 방법을 제안하여 모델 해석 가능성을 향상시킵니다. 실제 데이터 세트인 LASA를 사용한 모델링 사례로 제안된 프레임워크의 효과를 보여줍니다.



### A Low-Resolution Image is Worth 1x1 Words: Enabling Fine Image Super-Resolution with Transformers and TaylorShif (https://arxiv.org/abs/2411.10231)
- **What's New**: 이 연구에서는 1x1 패치 크기를 활용하여 pixel-level 처리(pixel-level processing)를 가능하게 하는 TaylorIR이라는 변환기 기반의 Super-Resolution(SR) 모델을 제안합니다. 이는 기존 SR 모델의 단점을 극복하고자 합니다.

- **Technical Details**: TaylorIR은 기존의 self-attention 메커니즘 대신 TaylorShift attention 메커니즘을 도입합니다. TaylorShift는 Taylor 급수 전개(Taylor series expansion)를 기반으로 하여 메모리 효율적인 방식으로 구현됩니다. 이를 통해 전체 token-to-token 상호작용을 선형 복잡도로 달성할 수 있습니다.

- **Performance Highlights**: TaylorIR을 사용한 SwinIR 모델인 TaylorSwinIR은 PSNR과 SSIM 메트릭에서 기존의 SR 모델들보다 뛰어난 성능을 발휘하며, 메모리 소비는 최대 60%까지 줄일 수 있음을 실험을 통해 입증하였습니다.



### Fused Gromov-Wasserstein Variance Decomposition with Linear Optimal Transpor (https://arxiv.org/abs/2411.10204)
- **What's New**: 이 논문은 2-Wasserstein 공간의 Fréchet 분산(decomposition of Fréchet variance)을 분석하여, Linear Optimal Transport (LOT) 임베딩이 확률 측도를 얼마나 잘 설명하는지를 정량적으로 평가하는 방법을 제시합니다.

- **Technical Details**: 이 연구에서는 Wasserstein 거리와 해당 거리의 Fréchet 분산을 해석하고, 2-Wasserstein과 Gromov-Wasserstein 거리에 대한 분산의 분해 방법을 소개합니다. LOT 임베딩이 원본 데이터의 정보 손실 정도를 이해하기 위해 Fréchet 분산을 결정론적(deterministic) 및 확률론적(probabilistic) 구성요소로 나누는 분해를 제안합니다.

- **Performance Highlights**: MNIST 데이터셋, IMDB-50000 데이터셋, Diffusion Tensor MRI 이미지에서 실험을 수행하여, 저차원 LOT 임베딩이 높은 분산 설명률과 기계 학습(classification accuracy) 분류의 정확도를 유지하며 정보를 잘 보존할 수 있음을 보였습니다.



### CART: Compositional Auto-Regressive Transformer for Image Generation (https://arxiv.org/abs/2411.10180)
Comments:
          under review at CVPR 2025

- **What's New**: 최근 이미지 합성 분야에서 혁신적인 접근법이 소개되었습니다. Auto-Regressive (AR) 모델링을 활용한 새로운 이미지 생성 방식은 세부 예측 전략을 통해 높은 충실도와 확장성을 제공합니다.

- **Technical Details**: 제안된 방법은 이미지의 기본 요소와 세부 요소를 계층적으로 결합하여 점진적으로 이미지를 구성합니다. 이 과정은 부드러운 기본 이미지를 생성하고, 세부 요소를 반복적으로 추가하여 최종 이미지를 완성하는 방식으로, 인지적으로 자연스러운 이미지 생성 방식을 모방합니다.

- **Performance Highlights**: 이 방법은 기존의 next-token 예측 방식보다 더 효과적이며, 최신 next-scale 예측 접근법을 초월합니다. 특히, 전체 모델 재훈련 없이도 고해상도로의 확장이 가능하여, 고해상도 이미지 생성에 있어 유연한 솔루션으로 자리 잡을 수 있습니다.



### Continuous Bayesian Model Selection for Multivariate Causal Discovery (https://arxiv.org/abs/2411.10154)
- **What's New**: 이 논문에서는 기존의 인과 관계 발견 접근 방식의 한계를 극복하기 위해 Bayesian 모델 선택을 멀티 변수 환경으로 확장하는 방법을 제안합니다. 이를 통해 더 유연한 가정으로 인과 구조를 효과적으로 학습할 수 있습니다.

- **Technical Details**: 우리는 Bayesian 비모수 모델인 Causal Gaussian Process Conditional Density Estimator (CGP-CDE)를 활용하여 하이퍼파라미터를 adjancency matrix로 해석할 수 있도록 하였고, Bayesian 모델 선택 오류에서 발생할 수 있는 확률을 약간 감수하는 대신 더 현실적인 가정을 세울 수 있습니다. 또한, 이 방법은 continuous optimization을 통해 대규모 문제를 처리할 수 있도록 설계되었습니다.

- **Performance Highlights**: 본 연구에서 제안한 접근 방식은 여러 대칭적 및 비대칭적 데이터셋에서 경쟁력 있는 성능을 보여주며, 경제성 및 윤리적 문제가 있는 개입(interventional) 데이터 없이도 인과 구조를 발견하는 것이 가능함을 입증하였습니다.



### BONE: a unifying framework for Bayesian online learning in non-stationary environments (https://arxiv.org/abs/2411.10153)
- **What's New**: 본 논문에서는 비정상(non-stationary) 환경에서 베이지안(Bayesian) 온라인 학습을 수행하는 방법들에 대한 통합 프레임워크인 BONE(Bayesian Online learning in Non-stationary Environments)를 제안합니다. 이를 통해 다양한 문제를 해결할 수 있는 공통 구조를 제공합니다.

- **Technical Details**: BONE 프레임워크는 세 가지 모델링 선택(모델: M.1, 보조 프로세스: M.2, 조건부 사전: M.3)과 두 가지 알고리즘 선택(사후 분포 추정: A.1, 보조 변수 추정: A.2)을 요구합니다. 이 프레임워크는 기존의 여러 방법을 BONE의 사례로 기록할 수 있음을 보여줍니다.

- **Performance Highlights**: 여러 데이터 셋에서 기존 방법들과 제안된 새로운 방법을 비교 실험하였으며, 특정 상황에서 한 방법이 다른 방법보다 적합한 이유에 대한 통찰을 제공하였습니다.



### DaYu: Data-Driven Model for Geostationary Satellite Observed Cloud Images Forecasting (https://arxiv.org/abs/2411.10144)
- **What's New**: 본 논문에서는 고조도(High-resolution) 단기 날씨 예보를 위한 AI 기반 모델 "DaYu"를 소개합니다. 이 모델은 정지위성(Geostationary satellite) 관측을 특별히 염두에 두고 설계되었습니다. DaYu는 기존의 AI 모델보다 더 높은 분해능(Resolution)과 시간적 해상도(Temporal resolution)를 갖추고 있어 6시간 이내의 단기 예보에 매우 적합합니다.

- **Technical Details**: DaYu 모델은 대규모(transformer) 아키텍처를 기반으로 하여, 고정밀(High precision) 열적 적외선(cloud images) 구름 이미지를 예측합니다. 시간 해상도는 0.5시간, 공간 해상도는 0.05° × 0.05°입니다. 특히, DaYu는 자가회귀(autoregressive) 모델 구조를 채택하여 시간대별 예측에 적합합니다. 주로 일본의 고급 히마와리 이미저(AHI)로부터 얻은 데이터를 사용합니다.

- **Performance Highlights**: DaYu는 3시간 이상 0.9 이상의 상관 계수(Correlation coefficient)를 기록하며, 6시간 예보에서 0.8, 12시간 예보에서 0.7 이상의 정확도를 보입니다. 또한, DaYu는 짧은 지속 시간의 메소스케일 및 소규모 날씨 사건을 효과적으로 감지하여 기존 방법의 한계를 극복하고 있습니다. 나아가, DaYu는 단기 기후 재해 예방 및 완화에 중대한 잠재력을 보유하고 있습니다.



### Prompting and Fine-tuning Large Language Models for Automated Code Review Comment Generation (https://arxiv.org/abs/2411.10129)
- **What's New**: 이번 연구는 오픈 소스 대형 언어 모델(LLM)을 파라미터 효율적인 Quantized Low-Rank (QLoRA) 방식으로 조정하여 코드 리뷰 주석 생성을 개선하는 방법을 제안합니다. 또한, 프로프라이어터리 LLM에 함수 호출 그래프와 코드 요약을 추가하여 성능을 향상시키는 새로운 방법론을 탐구하고 있습니다.

- **Technical Details**: 연구에서 사용된 메소드에는 QLoRA라는 파라미터 효율적인 미세 조정 방식이 포함되어 있으며, GPT 아키텍처에 기반한 디코더 전용 LLM들이 비약적인 성능을 보일 수 있음이 언급되었습니다. 본 연구는 코드 리뷰 활동을 위한 주석 생성에서 함수 호출 그래프와 코드 요약의 효과를 조사하고 있습니다.

- **Performance Highlights**: 실험 결과, GPT-3.5 모델에서 함수 호출 그래프를 추가한 Few-Shot 프롬프트가 CodeReviewer 데이터셋에서 사전 훈련된 기준보다 약 90% BLEU-4 점수를 초과했습니다. 또한, QLoRA로 미세 조정된 Code Llama와 Llama 3.1 모델이 이 작업에서 25%에서 83%의 성과 향상을 달성했습니다.



### On the Universal Statistical Consistency of Expansive Hyperbolic Deep Convolutional Neural Networks (https://arxiv.org/abs/2411.10128)
- **What's New**: 본 연구는 Hyperbolic DCNN (Deep Convolutional Neural Network)을 Poincaré Disc를 기초로 제안하며, 비유클리드 영역에서의 확장형 컨볼루션(expansive convolution)의 특성을 분석합니다.

- **Technical Details**: Hyperbolic 공간에서의 컨볼루션 깊이 신경망은 Euclidean 공간에 비해 데이터의 복잡한 패턴을 더 잘 포착할 수 있도록 설계되었습니다. 본 논문에서는 1차원 확장형 깊은 하이퍼볼릭 컨볼루션 네트워크에 대한 일관성 분석을 제공하며, Poincaré Disc에서의 확장형 컨볼루션을 기반으로 하는 이론적 토대를 구축하였습니다.

- **Performance Highlights**: 실험 결과, 하이퍼볼릭 컨볼루션 구조가 Euclidean 구조에 비해 훨씬 뛰어난 성능을 보이며, 데이터 세트에서 현저하게 낮은 오류율과 더 빠른 처리 속도를 달성했습니다.



### Energy-GNoME: A Living Database of Selected Materials for Energy Applications (https://arxiv.org/abs/2411.10125)
Comments:
          60 pages, 16 figures

- **What's New**: 이 논문은 에너지 응용을 위한 고유한 물질 발견에는 GNoME 프로토콜을 활용하여 380,000개 이상의 새로운 안정 정크리스탈을 식별하고, 그 중 33,000개 이상의 소재를 Energy-GNoME 데이터베이스에 등록한 내용을 중심으로 하고 있습니다.

- **Technical Details**: GNoME는 생성적 모델, 회귀 앙상블, 물리 기반 학습과 실험을 결합한 활성 학습 프레임워크를 통해 궤적 적합성과 2.2백만 개의 안정적인 물질을 발견하였습니다. 여기서는 Thermoelectric materials, Perovskites 및 Batteries와 같은 세 가지 에너지 분야에 대한 잠재적 물질의 예비 선별을 목표로 하였습니다.

- **Performance Highlights**: 본 연구는 기계 학습(ML) 기술을 활용하여 GNoME 데이터베이스의 물질들을 예측하는데, 기존의 지역적 데이터셋에서 벗어나 더 넓은 초과 지역에서의 예측이 가능하도록 하는 접근을 적용하였습니다.



### Generative Agent Simulations of 1,000 Peop (https://arxiv.org/abs/2411.10109)
- **What's New**: 이번 연구는 실제 개인 1,052명의 태도와 행동을 시뮬레이션하는 새로운 에이전트 아키텍처를 제시합니다. 이 연구는 대규모 언어 모델(large language models)을 사용하여 질적 면접(qualitative interviews)을 기반으로 하고 있습니다.

- **Technical Details**: 이 에이전트는 General Social Survey에서 참가자들의 응답을 85%의 정확도로 재현하며, 이는 참가자들이 2주 후 자신들의 답변을 재현하는 정확도와 유사합니다. 또한 실험적 재현에서 개인적 특성과 결과를 예측하는 능력도 비슷합니다.

- **Performance Highlights**: 제시된 아키텍처는 인종 및 이념적 그룹에 따른 정확도 편향을 줄이며, 새로운 도구들이 개인 및 집단 행동을 조사하는 데 기초가 될 수 있습니다.



### Recent Advances on Machine Learning-aided DSP for Short-reach and Long-haul Optical Communications (https://arxiv.org/abs/2411.10101)
Comments:
          paper accompanying an invited presentation at OFC 2025

- **What's New**: 이번 논문에서는 머신 러닝(Machine Learning, ML)의 발전을 통해 광통신(Optical Communications)에서 이퀄라이저(Equalizer)를 구현하는 방법의 혁신적인 변화를 다룹니다. 특히 전통적인 하드웨어와 신경형(hardware) 플랫폼을 활용한 두 가지 시나리오인 동기식 장거리 통신과 짧은 거리의 강도 변조/직접 검출(Intensity Modulation/Direct Detection, IM/DD) 시스템에 대해 설명합니다.

- **Technical Details**: ML 알고리즘은 고차원 데이터에서 패턴을 학습할 수 있는 능력을 가지고 있어, 전송기(transmitter) 및 수신기(receiver)의 DSP 알고리즘을 최적화하는 데 효과적입니다. 특히, 변형된 VAE(Variational Autoencoder) 구조를 적용한 이퀄라이저는 기존 이퀄라이저가 겪는 ‘jail-window’ 효과를 방지할 수 있으며, 이는 디지털 신호 처리 체인에서 여러 다른 부문으로의 확장 가능성을 보여줍니다. 또한, CNN(Convolutional Neural Network) 기반 이퀄라이저는 FPGA(Field Programmable Gate Array)를 통해 높은 속도와 낮은 비트 오류율(bit error rate, BER)을 달성할 수 있음을 입증했습니다.

- **Performance Highlights**: ML 기반의 이퀄라이저에서 짧은 거리 IM/DD 시스템의 경우, CNN 구조 최적화를 통해 성능이 향상되며, FPGA를 사용한 실시간 구현에서 20202020 GBd의 데이터 전송 속도를 지원하는 MLA(Equalizer) 모델이 성공적으로 테스트되었습니다. 또한, 스파이킹 신경망(Spiking Neural Networks, SNN)의 사용은 에너지 효율성을 높이는 promising한 연구 방향으로 제시되었습니다.



### Neural Port-Hamiltonian Models for Nonlinear Distributed Control: An Unconstrained Parametrization Approach (https://arxiv.org/abs/2411.10096)
Comments:
          The paper has 15 pages, and has been submitted for a possible publication. arXiv admin note: text overlap with arXiv:2403.17785

- **What's New**: 이 논문에서는 비선형 시스템의 분산 제어 정책을 설계하기 위해 port-Hamiltonian 시스템의 프레임워크를 활용하여 폐쇄 루프 안정성을 보장하는 연속 시간 제어 정책을 제안합니다. 이는 제어기의 최적화 매개변수와 무관하게 작동하며, 최적화 과정에서 매개변수를 제약할 필요를 없애줍니다.

- **Technical Details**: 제안된 방법은 pH 모델을 기반으로 하여, 비선형 제어 문제에 널리 적용될 수 있으며, 가중치 행렬에 희소성(sparsity)을 통합할 수 있습니다. 이를 통해 기존의 제약 조건을 두지 않고도 최적화가 가능하며, 표준 기법(예: stochastic gradient descent)을 사용할 수 있습니다. 또한, 지속적인 pH 제어기를 시간적으로 이산화할 때 소모적 특성을 유지하는 방법도 논의됩니다.

- **Performance Highlights**: 제안된 분산 제어기는 비선형 이동 로봇에 대한 합의 제어 및 DC 마이크로그리드에서의 가중전력 공유와 평균 전압 조절을 통한 효과성을 입증했습니다.



### Evidential Federated Learning for Skin Lesion Image Classification (https://arxiv.org/abs/2411.10071)
Comments:
          Published as a conference paper at ICPR 2024

- **What's New**: FedEvPrompt는 evidential deep learning, prompt tuning, knowledge distillation을 통합하여 분산된 피부 병변 분류를 위한 새로운 연합 학습 접근 방식을 제시합니다. 이 방법은 기존의 모델 파라미터 공유 대신 attention maps를 통해 지식을 공유하여 개인 정보 보호를 강화합니다.

- **Technical Details**: FedEvPrompt는 frozen pre-trained Vision Transformer (ViT) 모델에 b-prompts(저수준 기본 시각적 지식) 및 t-prompts(작업 특정 지식)를 결합하여 학습합니다. 학습 과정은 round-based learning paradigm 내에서 이루어지며, 각 라운드는 로컬 모델 훈련 후 attention maps를 공유하는 방식입니다. 이러한 구조는 클래스 증거를 극대화하고, 데이터 불균형과 비독립적이고 동일하게 분포되지 않은(non-i.i.d.) 데이터 문제를 해결하는 데 기여합니다.

- **Performance Highlights**: ISIC2019 데이터셋을 사용한 실험 결과, FedEvPrompt는 기존 연합 학습 알고리즘 및 knowledge distillation 방법들에 비해 뛰어난 성능을 보여줍니다. 특히, 모델 파라미터를 공유하지 않고도 우수한 결과를 달성하였습니다.



### Adaptive Physics-Guided Neural Network (https://arxiv.org/abs/2411.10064)
- **What's New**: 이 논문은 이미지를 통해 품질 속성을 예측하기 위한 적응형 물리 기반 신경망(Adaptive Physics-Guided Neural Network, APGNN) 프레임워크를 소개합니다. 이 프레임워크는 물리 법칙을 심층 학습 모델에 통합하여 데이터 기반 및 물리 정보 예측의 균형을 조정함으로써 다양한 환경에서 모델의 정확도와 견고성을 향상시킵니다.

- **Technical Details**: APGNN은 두 가지 실제 데이터 세트에 적용되어 RGB 이미지 기반의 오이 품질 평가와 열 이미지 기반의 물체 분류를 수행하며, 확률적 모델 및 기존의 데이터 기반 모델(예: ResNet)과 비교합니다. APGNN은 비선형 변환을 사용하여 복잡한 물리적 과정을 이미지 형식으로 모사하고, 구조적이지 않은 설정에서도 일관된 성능을 발휘합니다.

- **Performance Highlights**: 모든 실험에서 APGNN은 다양한 열 이미지 데이터 세트에서 우수한 성과를 보였고, 특히 환경 변동성이 큰 야외 재료의 경우 PGNN과 ResNet을 능가했습니다. 연구 결과에 따르면, 적응형 물리 학습은 복잡한 실제 환경에서도 물리적 제약을 효과적으로 통합할 수 있음을 보여줍니다.



### Federated Domain Generalization via Prompt Learning and Aggregation (https://arxiv.org/abs/2411.10063)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이 논문에서는 Federated Domain Generalization(FedDG) 설정에서 개인정보 보호 제약을 유지하면서도 데이터 이질성을 다룰 수 있는 새로운 방안으로 Prompt Learning을 도입합니다.

- **Technical Details**: PLAN(프롬프트 학습 및 집계 방법)은 두 단계로 구성된 훈련 프레임워크입니다. 먼저, 각 클라이언트는 자신의 데이터로부터 텍스트 및 시각적 프롬프트 학습을 수행하고, 이후 도메인 특화된 로컬 프롬프트가 클라이언트 간에 교환되고 경량 기반의 주의(Attention) 집계기를 통해 글로벌 프롬프트로 집계됩니다.

- **Performance Highlights**: PLAN 방법은 4개의 벤치마크 데이터셋에서 기존의 FL, DG, FedDG 방법과 프롬프트 학습 기반 방법들보다 현저히 우수한 성능을 보였습니다. 또한, 계산 효율성 및 통신 효율성 면에서도 뚜렷한 장점을 나타냈습니다.



### Unsupervised Congestion Status Identification Using LMP Data (https://arxiv.org/abs/2411.10058)
Comments:
          Paper accepted for IEEE Transactions on Smart Grid. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses

- **What's New**: 이번 연구에서는 고차원 유클리드 공간에서의 위치별 한계 가격(LMP)의 혼잡 부분(부분)의 기본 분포를 비지도 학습(unsupervised approach)을 통해 조사하고, 이를 통해 LMP 데이터의 계층적인 부분공간(subspace) 속성을 분석합니다.

- **Technical Details**: LMP 모델은 손실이 없는 DC Optimal Power Flow(DC-OPF)와 손실이 있는 DC-OPF를 기반으로 하며, LMP 데이터의 겹치는 부분공간(overlapping subspace) 특성을 보입니다. 혼잡 LMP 데이터의 부분공간을 스팬(span)하는 기저 벡터(basis vectors)를 탐색하는 방법이 제안됩니다. 이 방법은 1차원 부분공간의 데이터를 탐지한 후, 데이터를 직교(subspace) 부분공간에 투사(projection)하는 방식으로 진행됩니다. 전반적인 방법론은 데이터 세트를 활용한 기계 학습 기법들로 구성됩니다.

- **Performance Highlights**: IEEE 30버스 시스템, IEEE 118버스 시스템, Illinois 200버스 시스템, Southwest Power Pool을 기반으로 한 수치 실험을 통해 제안된 방법의 성능이 검증되었습니다. 혼잡 상태를 탐지하고, LMP 예측에 기여하는 기타 응용 프로그램을 통해 이 방법이 효과적임을 나타냅니다.



### KuaiFormer: Transformer-Based Retrieval at Kuaishou (https://arxiv.org/abs/2411.10057)
- **What's New**: 본 논문에서는 Kuaishou의 대규모 콘텐츠 추천 시스템에 적용된 새로운 Transformer 기반 검색 프레임워크인 KuaiFormer를 소개합니다. KuaiFormer는 기존의 점수 추정 작업에서 Transformer 기반의 다음 행동 예측 패러다임으로 전환하여 검색 프로세스를 근본적으로 재정의합니다.

- **Technical Details**: KuaiFormer는 여러 쿼리 토큰을 도입하여 사용자의 다양한 관심사를 포착하고 연속된 항목 시퀀스에서 별개의 사용자 관심 표현을 추출합니다. 또한, 효율성을 높이기 위해 조정 가능한 항목 압축 메커니즘을 포함하고 있습니다. 이러한 방법들은 긴 시퀀스를 효과적으로 모델링할 수 있도록 설계되었습니다.

- **Performance Highlights**: KuaiFormer는 Kuaishou의 단기 동영상 서비스에서 +0.360%/-0.126%/-0.411%의 온라인 시청 시간 증가를 기여하며, 실시간 대규모 추천 시스템에 적합한 최초의 Pure Transformer 구조 기반 검색 모델로 평가받고 있습니다.



### That Chip Has Sailed: A Critique of Unfounded Skepticism Around AI for Chip Design (https://arxiv.org/abs/2411.10053)
- **What's New**: 이번 논문에서는 AlphaChip에 대한 최근 비판에 대응하여, 이 방법이 실제로 뛰어난 성능을 발휘하고 있음을 강조합니다. Markov의 비판은 심각한 방법론적 문제를 안고 있으며, 그 결과 우리의 연구 결과는 Nature에서 계속해서 유지되고 있습니다.

- **Technical Details**: AlphaChip은 강화학습(reinforcement learning) 기반의 방법으로, 인간 전문가를 초월하는(superhuman) 칩 배치를 생성하는 데 쓰입니다. 그러나 Cheng et al.의 ISPD 논문은 우리의 방법론에 대한 비판을 포함하고 있으며, 그들의 실험은 사전 훈련(pre-training) 없이 진행되었고, 컴퓨팅 자원(compute resources)이 20배 적게 사용되었습니다.

- **Performance Highlights**: Nature 검토 과정에서 우리의 AlphaChip이 실제로 뛰어난 성능을 가지고 있으며, 여러 산업에서 사용되고 있다는 점을 다시 한 번 강조했습니다. 최근 MediaTek이 AlphaChip을 활용하여 그들의 최신 칩을 개발하고 있다는 사실도 부각되고 있습니다.



### Towards Utilising a Range of Neural Activations for Comprehending Representational Associations (https://arxiv.org/abs/2411.10019)
Comments:
          18 pages, 11 figures

- **What's New**: 최근 심층 신경망에서 중간 표현을 이해하기 위한 접근 방식이 개별 신경세포와 선형 방향을 분석하여 해석되고 있습니다. 하지만 본 연구에서는 이러한 방법이 표현의 복잡한 행동을 포착하지 못한다는 것을 보여줍니다. 비극대활성화(neural network activations)는 일반적으로 밀집되어 있으며, 비극대 수준의 활성화가 혼란스러운 인간 해석 가능 개념을 찾아내는 데 유용하다는 가설을 세웠습니다.

- **Technical Details**: 본 연구는 중간 수준 출력 신경세포 활성화를 사례로 하여, 비극대 활성화가 신경망의 마지막 계층에서의 표현을 이해하는 데 어떻게 기여할 수 있는지를 탐구했습니다. 특히, 합성 데이터셋에서 중간 수준 활성화가 극대화된 활성화 예시만으로는 드러나지 않는 표현의 측면을 알려주는 방법을 보여주었습니다.

- **Performance Highlights**: 제안된 방법은 비극대 활성화를 검사하여 모델이 학습한 복잡한 관계를 추출하는 유용성을 시연하며, 실제 벤치마크 데이터셋에서의 성능을 향상시키는 데 성공적이었습니다. 또한, 비극대 활성화를 통해 잘못 라벨링된 샘플을 식별하고, 모델이 의존하고 있는 스푸리어스(correlations) 패턴을 완화하는 데 유용한 데이터를 선별할 수 있음을 입증하였습니다.



### MicroCrackAttentionNeXt: Advancing Microcrack Detection in Wave Field Analysis Using Deep Neural Networks through Feature Visualization (https://arxiv.org/abs/2411.10015)
- **What's New**: 본 논문에서는 MicroCrackAttentionNeXt라는 새로운 모델을 제안하여, 기존 SpAsE-Net을 개선하여 미세 크랙 감지에서의 성능을 향상시킵니다. 이 모델은 비대칭 인코더-디코더 구조를 사용하며, 중요한 크랙 패턴을 포착하는 데 집중하여 정확도를 높입니다.

- **Technical Details**: MicroCrackAttentionNeXt는 spatio-temporal (시공간) 데이터 분석을 통해 복잡한 크랙을 인식하도록 설계되었으며, 다양한 activation (활성화) 및 loss functions (손실 함수)의 영향을 분석합니다. 이 과정에서 Manifold Discovery Analysis (MDA)를 활용하여 최적의 활성화 함수를 선택하는 방법론을 제시합니다.

- **Performance Highlights**: 제안된 모델은 최적화된 아키텍처와 훈련 방법론을 통해 86.85%의 정확도를 달성하였습니다. 이런 성과는 기존의 비율이 낮은 데이터셋에서 클래스 불균형 문제를 효과적으로 해결한 결과입니다.



### Efficient Depth Estimation for Unstable Stereo Camera Systems on AR Glasses (https://arxiv.org/abs/2411.10013)
- **What's New**: 이 논문은 증강 현실(AR) 애플리케이션을 위한 스테레오 깊이 추정에서의 전통적인 모델들의 단점을 극복하고, 고비용의 cost volume과 전처리 단계를 줄이기 위한 새로운 두 가지 모델, MultiHeadDepth와 HomoDepth를 개발했습니다.

- **Technical Details**: 기존의 cost volume 프로세스를 새로운 group-pointwise convolution 기반의 연산자로 대체하고, layernorm과 dot product를 이용한 코사인 유사도의 효율적 근사를 채택합니다. 전처리 과정 없이 원시 이미지(미정렬 이미지)를 처리할 수 있는 homography matrix prediction network를 도입하여 온라인 스테레오 직선을 구현합니다.

- **Performance Highlights**: MultiHeadDepth는 AR 글래스의 최신 깊이 추정 모델에 비해 정확도를 11.8-30.3% 향상시키고, 지연을 22.9-25.2% 줄였습니다. HomoDepth는 미정렬 이미지를 처리하면서 전체 지연 시간을 44.5% 줄이는 성능을 보였습니다.



### Fully Dynamic Adversarially Robust Correlation Clustering in Polylogarithmic Update Tim (https://arxiv.org/abs/2411.09979)
- **What's New**: 본 논문에서는 동적 상관 클러스터링(dynamic correlation clustering) 문제에서 적응형(edge label flips) 엣지 레이블 뒤집기를 다루고 있습니다. 기존의 연구와는 달리, 적대적인 환경에서도 강건성을 고려하여 알고리즘의 출력에 따라 엣지 레이블이 변경될 수 있는 상황에서의 알고리즘을 제안합니다.

- **Technical Details**: 이 연구의 주요 결과로는 $O(1)$ 근사(optimal correlation clustering)를 항상 유지하며 $O(	ext{log}^2{n})$의 평균 업데이트 시간(amortized update time)을 가지는 무작위화 알고리즘(randomized algorithm)을 제시합니다. 이 연구에서 사용된 주요 기술 요소는 $O(	ext{polylog}{(n)})$ 업데이트 시간으로 희소-조밀 분해(sparse-dense decomposition)를 유지하는 알고리즘입니다.

- **Performance Highlights**: 이론적 결과를 검증하기 위해 합성 데이터(synthetic datasets) 및 실제 데이터(real-world datasets)에서 실험을 수행하였으며, 경쟁력 있는 실험적 성능(competitive empirical performances)을 보였습니다.



### Dense ReLU Neural Networks for Temporal-spatial Mod (https://arxiv.org/abs/2411.09961)
- **What's New**: 이번 논문에서는 비모수적(estimation) 추정을 위해 Rectified Linear Unit (ReLU) 활성화 함수를 사용하는 완전 연결된 딥 뉴럴 네트워크에 주목하고 있습니다. 과거 연구에 비해 시간적(timely) 및 공간적(spatial) 의존성을 동시에 고려하여 비비대칭 경계(non-asymptotic bounds)를 도출하였습니다.

- **Technical Details**: 논문은 데이터의 내재적 차원(intrinsic dimensionality)을 탐구하며, 고차원 데이터에 대한 차원의 저주(curse of dimensionality) 문제를 해결하기 위해 매니폴드(manifold)에서 데이터를 모델링합니다. 특히, 짧은 범위 의존성을 가진 모델에 효과적인 증명 기법(proof techniques)을 적용하였습니다.

- **Performance Highlights**: 다양한 합성 응답 함수(synthetic response functions)에 대한 실험적 시뮬레이션을 통해 제안한 방법이 기존 문헌에서 제시된 접근 방식들을 초월하는 우수한 성능을 보여주었습니다. 우리의 방법은 시간-공간 모델링에 대한 밀집(dense) 뉴럴 네트워크의 강력한 능력을 입증합니다.



### Instruction-Guided Editing Controls for Images and Multimedia: A Survey in LLM era (https://arxiv.org/abs/2411.09955)
- **What's New**: 이번 논문은 LLM(large language models) 및 멀티모달(Multimodal) 모델이 시각적 수정 작업을 간소화하는 방법에 대한 개요를 제공합니다. 사용자는 복잡한 기술적 지식 없이 자연어로 지시를 통해 이미지 및 비디오 편집을 수행할 수 있습니다.

- **Technical Details**: 논문은 생성적 적대 신경망(generative adversarial networks)과 확산 모델(diffusion models)에서부터 시작하여, MLLM(multimodal large language models)과 같은 최첨단 기술을 통합한 방법을 다룹니다. 이러한 모델들은 사용자가 자연어로 간단한 명령을 주면 시각적 콘텐츠에 대한 정밀한 수정을 할 수 있도록 지원합니다.

- **Performance Highlights**: LLM 및 MLLM의 발전은 패션, 3D 장면 조작 및 비디오 합성 등 다양한 분야에서 더 많은 접근성을 제공하여 사용자 경험을 향상시켰습니다. 이 논문은 LLM 기반 편집이 산업 전반에 걸쳐 강력한 도구로 자리 잡을 수 있음을 강조하고, 사용자 친화적인 편집 툴의 필요성을 제시합니다.



### TEESlice: Protecting Sensitive Neural Network Models in Trusted Execution Environments When Attackers have Pre-Trained Models (https://arxiv.org/abs/2411.09945)
Comments:
          Accepted by TOSEM. Extended version of the S&P24 paper (arXiv:2310.07152)

- **What's New**: 본 논문은 기존의 TSDP 방식이 지식이 있는 적대자에게 충분히 안전하지 않음을 밝혔다. 이를 해결하기 위해 모델 학습 전 파티셔닝 전략을 제안하여 개인 정보가 민감한 가중치와 그렇지 않은 구성 요소를 효과적으로 분리할 수 있음을 보여준다.

- **Technical Details**: 제안된 방식은 DNN 모델의 개인 정보가 민감한 가중치를 TEE 안에서 보호하고, 덜 민감한 가중치를 GPU로 오프로드하는 방법이다. 이를 통해 기존 방법들보다 10배 낮은 계산 비용으로 전체 모델 보호를 제공한다.

- **Performance Highlights**: 대규모 언어 모델에 대해서도 적용 가능하며, 개인 정보 기능을 경량화하여 전체 모델 보호 수준을 달성할 수 있음을 보여준다.



### Zero-shot Voice Conversion with Diffusion Transformers (https://arxiv.org/abs/2411.09943)
- **What's New**: Seed-VC는 기존의 음색 누수(timbre leakage) 문제와 훈련-추론 불일치(training-inference inconsistency) 문제를 해결하기 위해 외부 음색 변환기(external timbre shifter)를 도입한 새로운 제로샷 음성 변환(zero-shot voice conversion) 프레임워크입니다.

- **Technical Details**: Seed-VC는 훈련 과정에서 소스 음성의 음색을 변형하기 위해 외부 음색 변환기를 사용하며, 모든 참조 음성을 전체 맥락(context)으로 활용하는 확산(transformer) 아키텍처를 채택하여 미세한 음색 특성을 캡처하는 in-context learning을 적용하였습니다.

- **Performance Highlights**: 실험 결과 Seed-VC는 OpenVoice 및 CosyVoice와 같은 강력한 벤치마크와 비교하여 높은 화자 유사도(spaker similarity)와 낮은 단어 오류율(word error rate, WER)을 기록하였으며, 제로샷 노래 음성 변환(zero-shot singing voice conversion)에서도 유사한 성능을 나타냈습니다.



### Self-Supervised Learning of Grasping Arbitrary Objects On-the-Mov (https://arxiv.org/abs/2411.09904)
Comments:
          8 pages, 9 figures

- **What's New**: 이번 연구는 상용 로봇이 동적 상황에서도 물체를 잡을 수 있도록 모바일 그랩핑(mobile grasping) 기술을 적용한 것으로, 정밀한 타이밍(timing)과 자세(pose) 조절을 통해 물체의 형태에 따라 잡을 위치와 방향을 판단할 수 있는 일반화된 정책을 스스로 발전시킬 수 있는 자가 지도 학습(self-supervised learning) 방식을 사용합니다.

- **Technical Details**: 모바일 그랩핑을 위해 두 개의 잡기 조작 원시(action primitives)와 하나의 이동 조작 원시로 단순화된 접근 방식을 채택하고, 각 원시의 수행 시 FCN(fully convolutional network) 모델을 이용하여 정적 및 동적 잡기 원시를 예측합니다. 또한, 데이터 희소성을 방지하기 위해 단계별 학습(step-by-step learning)을 적용하여 학습의 효율성을 높였습니다. 실험에서 RGB-D 센서를 통해 수집된 이미지 자료를 기반으로 한 포인트 클라우드(point cloud) 변환을 통해 모바일 그랩핑이 이루어집니다.

- **Performance Highlights**: 제안된 방법은 실험을 통해 가장 높은 잡기 정확도와 픽 앤 플레이스(pick-and-place) 효율성을 달성했습니다. 특히, 다양한 속도와 형태를 가진 물체에 대해 잡기 성공률(grasping success rate)과 학습 성능(learning performance)에서 기존 방법과 비교하여 높은 효율성을 입증하였으며, 시뮬레이션 및 실제 환경 모두에서 효과를 보여주었습니다.



### Revealing the Evolution of Order in Materials Microstructures Using Multi-Modal Computer Vision (https://arxiv.org/abs/2411.09896)
Comments:
          30 pages, 5 figures, 2 tables

- **What's New**: 이번 연구는 복합 산화물 La_{1-x}Sr_xFeO_3에 대한 전자 현미경 분석을 활용하여 모드 다중 기계학습 접근 방식을 통해 마이크로구조의 순서를 설명하는 방법을 제시합니다. 연구팀은 완전 감독 학습과 반 감독 학습을 기반으로 한 하이브리드 파이프라인을 구축하여 각 데이터 모드의 특성과 이들이 모형에 기여하는 가치를 평가하였습니다.

- **Technical Details**: 연구에서는 스캐닝 트랜스미션 전자현미경(STEM) 이미징에서 수집된 데이터를 사용하여 마이크로구조의 특성을 분석합니다. STEM 데이터의 정보량이 많지만 연구자들은 그 중 일부 데이터만을 활용하고 있으며, 다중 모드 데이터의 통합적 분석을 통해 더 나은 microstructural descriptor를 제공할 수 있습니다. 이 과정에서 다중 모드 AI/ML 기법을 활용하여 서로 다른 신호 모드가 세분화 결과에 미치는 영향을 평가하고 있습니다.

- **Performance Highlights**: 다양한 단일 모드 및 다중 모드 모델의 성능 차이를 관찰하였으며, UNI-MODAL 모델과 MULTI-MODAL 모델의 성능 향상에 대한 일반적인 교훈을 도출하였습니다. 궁극적으로 이 연구는 컴퓨터 비전을 통한 결정 질서의 설명에 기여할 수 있는 가능성을 보여주며, 복잡한 산화물 시스템에서의 응용에 대한 새로운 방향을 제시합니다.



### Deep learning robotics using self-supervised spatial differentiation drive autonomous contact-based semiconductor characterization (https://arxiv.org/abs/2411.09892)
- **What's New**: 이번 연구에서는 자율 접촉 기반 로봇 특성화 기법을 사용하여 반도체 측정의 품질과 효율성을 향상시키는 새로운 방법을 제안합니다. 특히, 스스로 학습할 수 있는 convolutional neural network (CNN) 모델을 통해 최적의 로봇 접촉 위치를 예측하여 기존 모델에 비해 20% 향상된 성능을 보여주고 있습니다.

- **Technical Details**: 제안된 SDCNN은 공간적으로 미분 가능한 손실 함수(spatially differentiable loss function)를 채택하고, 모양 선행(shape priors)을 포함하여 로봇의 최적 접촉 포즈를 보다 정밀하게 조정합니다. 이 방법은 4자유도(4DOF) 로봇을 자율적으로 구동하여 다양한 반도체 조성에 따른 광전도성 측정을 수행합니다. 또한, SDCNN은 최적의 포즈 예측에 있어 기존 CNN의 손실 함수를 사용한 경우보다 평균 1.5%에서 8.9% 더 높은 정확도를 달성합니다.

- **Performance Highlights**: 이 연구에서는 3,025개의 예측된 포즈를 통해 시간당 125회 이상의 측정량을 달성했습니다. photoconductivity를 각 드롭 캐스팅된 필름에 대해 공간적으로 매핑하여 불균형 지역을 시각화하는 데 성공했으며, 이는 자율 주행 실험실의 연락 기반 특성화 기법을 고정밀, 신뢰성 있게 자동화할 수 있는 기반을 마련합니다.



### KULCQ: An Unsupervised Keyword-based Utterance Level Clustering Quality Metric (https://arxiv.org/abs/2411.09853)
- **What's New**: 본 연구에서는 대화형 데이터의 클러스터링 품질을 평가하기 위해 키워드 기반 발화 수준 클러스터 품질(KULCQ)이라는 비지도형(metric) 메트릭을 도입하였다. 기존 클러스터링 메트릭은 데이터의 기하학적 구조에 초점을 맞추고 있었으나, KULCQ는 대화의 언어적 뉘앙스를 고려하여 발화 간의 의미적 관계를 좀 더 잘 포착한다.

- **Technical Details**: KULCQ는 사용자 쿼리 발화의 클러스터링을 평가하기 위해 키워드 분석을 활용한다. 문장 내에서 사용되는 단어의 차이에 관계없이 비슷한 의도를 파악할 수 있게 해준다. 이에 따라 KULCQ는 클러스터의 중심을 정의하고, 클러스터 내 및 클러스터 간 거리 기반으로 클러스터 품질을 평가한다. 실험에서는 코사인 거리(cosine distance)를 사용하였다.

- **Performance Highlights**: KULCQ 메트릭은 기존 비지도 클러스터링 메트릭과 비교할 때 대화 데이터의 의미적 관계를 보다 잘 포착하는 것으로 나타났다. 포괄적인 실험을 통해 KULCQ가 대화 데이터에 특화된 클러스터 품질 측정에 있어 우수성을 입증하였다.



### InterFormer: Towards Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction (https://arxiv.org/abs/2411.09852)
Comments:
          10 pages, 6 figures

- **What's New**: 이 논문에서는 Click-through rate (CTR) 예측을 위한 새로운 모듈 InterFormer를 제안하고 있습니다. InterFormer는 이질적인 정보 상호작용을 상호 유익한 방식으로 학습하여 CTR 예측의 성능을 향상시키는 데 중점을 둡니다.

- **Technical Details**: InterFormer 모듈은 두 가지 주요 아이디어로 구성됩니다. 첫째, 서로 다른 모드 간에 양방향 정보 흐름을 가능하게 하여 글로벌 및 시퀀스 학습을 혼합 방식으로 수행합니다. 둘째, 정보를 과도하게 집계하지 않기 위해 각 데이터 모드에서 완전한 정보를 유지하고, 효과적인 정보 선택 및 요약을 위한 별도의 브리징 아치를 사용합니다.

- **Performance Highlights**: InterFormer는 세 개의 공개 데이터셋과 대규모 산업 데이터셋에서 최첨단 성능을 달성하였습니다. 논문에서 언급한 바에 따르면 AUC 기준으로 최대 0.14% 개선과 내부 대규모 데이터셋에서 0.15%의 Normalized Entropy (NE) 이득을 달성했습니다.



### SymbolFit: Automatic Parametric Modeling with Symbolic Regression (https://arxiv.org/abs/2411.09851)
Comments:
          53 pages, 35 figures. Under review

- **What's New**: SymbolFit이라는 새로운 프레임워크를 소개하며, 이는 심볼릭 회귀(Symbolic Regression)를 사용하여 데이터에 적합한 함수를 자동으로 찾고 동시에 불확실성 추정치를 제공하는 것을 목표로 합니다.

- **Technical Details**: SymbolFit은 전통적인 매개변수 모델링(parmmetric modeling)의 수동적이고 반복적인 과정을 개선하기 위해 설계되었습니다. 이 방법은 미리 정의된 함수 형태 없이 방대한 후보 함수 공간을 탐색하고, 함수 형태 자체를 학습 가능 파라미터로 간주하여 적용합니다.

- **Performance Highlights**: CERN의 대형 강입자 충돌기(LHC)에서의 고에너지 물리학 실험의 데이터 분석에 이 기법을 적용하여, 5가지 실제 프로톤-프로톤 충돌 데이터셋을 사용하여 효과성과 효율성을 입증했습니다.



### Enhancing Diffusion Posterior Sampling for Inverse Problems by Integrating Crafted Measurements (https://arxiv.org/abs/2411.09850)
- **What's New**: 본 논문에서는 새로운 diffusion posterior sampling 방법인 DPS-CM을 제안합니다. 이 방법은 깨끗한 이미지 대신 잡음이 있는 측정치에서 log posterior gradient를 형성하여 역과정을 개선하고, Crafted Measurement를 통합하여 posterior estimate를 생성합니다.

- **Technical Details**: DPS-CM은 문제가 발생할 수 있는 누적 posterior estimate 오류로 인한 diffusion prior와의 불일치를 완화하는 것을 목표로 합니다. 이 방법은 Gaussian deblurring, super-resolution, inpainting 및 Poisson noise와 같은 다양한 noisy inverse 문제를 해결하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 기존 접근 방식에 비해 일반적이고 noisy inverse 문제 해결능력을 획기적으로 향상시킴을 입증했습니다.



### Self-Supervised Radio Pre-training: Toward Foundational Models for Spectrogram Learning (https://arxiv.org/abs/2411.09849)
- **What's New**: 이번 연구에서 제안한 Masked Spectrogram Modeling(MSM)은 라디오 신호에 대한 자가 지도 학습(self-supervised learning) 기술로, 기초 딥 러닝 모델을 사전 훈련하기 위한 혁신적인 접근 방식을 제공합니다. 이 모델은 경량화된 작업 흐름으로 신속하게 개발되고, 훈련 비용을 절감합니다.

- **Technical Details**: 기초 모델은 대규모 비라벨 데이터셋을 통해 사전 훈련되어 다양한 다운스트림 작업에 맞게 조정될 수 있습니다. 본 연구에서는 Convolutional LSTM 아키텍처를 사용하여 공간-시간적(spatio-temporal) 처리를 효과적으로 수행하고, 실제로 수집된 라디오 데이터셋을 기반으로 MSM을 활용하여 주파수 예측(spectrum forecasting) 및 세분화(segmentation) 두 가지 다운스트림 작업을 수행합니다.

- **Performance Highlights**: 실험 결과, 제안된 접근 방식은 주파수 예측 및 세분화 정확도에서 경쟁력 있는 성능을 확인하였으며, 이는 무선 신호 처리에 대한 기초 모델의 유효성을 입증합니다. 이 연구는 AI의 광범위한 사용을 촉진하고, 안정적인 네트워크 성능과 서비스 제공에 기여할 것으로 기대됩니다.



### Can Features for Phishing URL Detection Be Trusted Across Diverse Datasets? A Case Study with Explainable AI (https://arxiv.org/abs/2411.09813)
Comments:
          8 pages, 10 figures, The 11th International Conference on Networking, Systems and Security, December 19-21, 2024

- **What's New**: 이번 연구는 피싱 URL (Phishing URL) 탐지에서 특징의 일반화 가능성을 검토하고, 서로 다른 데이터셋에서 훈련된 모델의 성능을 분석합니다. 이를 통해 피싱 URL 탐지 기능의 신뢰성을 평가하는 것을 목표로 하고 있습니다.

- **Technical Details**: 연구에서는 두 개의 공개 피싱 URL 데이터셋을 분석하고, SHAP (SHapley Additive exPlanations) 방법론을 통해 각 특징의 기여도를 평가합니다. XAI (Explainable AI) 기법을 적용하여 데이터셋 간의 겹치는 특징이 유사한지 여부를 조사하고, ML (Machine Learning) 모델을 훈련하여 다양한 데이터셋에서 이들의 일반성에 대해 실험을 수행합니다.

- **Performance Highlights**: 실험 결과는 피싱 URL 탐지 특징이 데이터셋에 따라 다를 수 있음을 보여주며, 일부 ML 모델은 특정 데이터셋에만 효과적일 수 있음을 나타냅니다. 따라서, 특정 데이터셋에 대한 높은 정확도가 다른 데이터셋에서도 공통적으로 유지되지 않음을 경고하고 있습니다.



### Edge Caching Optimization with PPO and Transfer Learning for Dynamic Environments (https://arxiv.org/abs/2411.09812)
- **What's New**: 본 논문은 동적 환경에서의 엣지 캐싱 문제를 다루며, Proximal Policy Optimization (PPO)을 기반으로 하는 새로운 캐싱 전략을 제안합니다. 이는 파일의 크기, 수명, 중요성 및 인기와 같은 주요 속성을 고려하고, 랜덤 파일 요청 도착을 반영하여 보다 현실적인 엣지 캐싱 시나리오를 구현합니다. 또한, 콘텐츠 인기와 요청 속도의 변화를 신속히 탐지할 수 있는 메커니즘을 개발하였습니다.

- **Technical Details**: 이 연구는 두 가지 주요 기능을 갖춘 메커니즘을 소개하며, 첫 번째 기능은 최근과 과거 요청 패턴 간의 코사인 유사도를 활용하여 인기 변화 감지를 수행합니다. 두 번째 기능은 정의된 시간의 간격 내에서 요청 속도의 간단한 이동 평균을 사용하여 요청 속도의 변화를 감지합니다. 이는 훈련된 정책을 새로운 환경에 적응시키기 위해 전이 학습을 통합하여, 기존 지식을 활용해 적응 속도를 높이는 PPO 알고리즘을 제안합니다.

- **Performance Highlights**: 모의 실험 결과, 제안된 접근 방식이 최근의 Deep Reinforcement Learning (DRL) 기반 방법보다 우수한 성능을 보였습니다. 특히, 콘텐츠 요청이 급격히 변동하는 환경에서 변화 탐지 방법이 신속하게 변화를 식별하고, 전이 학습을 통해 유의미한 경험에서 학습하여 빠른 최적화가 이루어졌다는 점에서 효과성이 입증되었습니다.



### Video Denoising in Fluorescence Guided Surgery (https://arxiv.org/abs/2411.09798)
- **What's New**: 본 연구에서는 Fluorescence Guided Surgery (FGS)의 비디오 잡음을 제거하기 위한 새로운 딥 러닝 기반 알고리즘을 제안하며, Laser Leakage Light (LLL)를 시뮬레이션하는 파이프라인을 개발하였습니다.

- **Technical Details**: FGS 시스템에서 발생하는 LLL은 영상 신호와 비슷한 밝기로 나타나며 이를 제거하는 것이 큰 도전입니다. 본 연구에서는 참조 비디오 (RV)를 통해 LLL을 정확하게 시뮬레이션하고, 이를 이용한 잡음 제거 알고리즘을 개발하였습니다. 새로운 비디오 잡음 제거 모델로 BL-RNN을 제안합니다.

- **Performance Highlights**: NafNet이라는 이미지 잡음 제거 모델이 기존의 비디오 잡음 제거 모델보다 우수한 성능을 보였으며, 훈련 효율성에서도 뛰어난 결과를 나타냈습니다. 이를 바탕으로 OL-2024 데이터셋을 구축하여 다양한 실험에 활용하였습니다.



### Can EEG resting state data benefit data-driven approaches for motor-imagery decoding? (https://arxiv.org/abs/2411.09789)
- **What's New**: 이 연구에서는 휴식 상태에서의 EEG 데이터를 활용하여 motor imagery BCI 성능을 개선하기 위한 feature concatenation 접근법을 제안합니다. 이 방법은 motor imagery와 관련된 EEG 기능과 resting-state EEG 기능을 통합하여, 사용자 일반화 모델을 개발하는데 초점을 맞추고 있습니다.

- **Technical Details**: 87명의 참가자들로부터 수집된 전기생리학적 신호(EEG) 데이터를 사용하여 motor imagery 및 resting-state 조건에서의 연구가 진행되었습니다. EEGNet 모델을 사용하여 convolutional neural network(CNN) 구조를 기반으로 기능 연결성을 분석하고, coherence(COH)와 phase-locking value(PLV)를 포함한 스펙트럼 연결 측정치를 사용하여 resting-state EEG 데이터를 분석하였습니다.

- **Performance Highlights**: 연구 결과, feature concatenation은 user 내 시나리오에서 평균 정확도를 개선했지만, user 간 시나리오에 대해서는 랜덤 데이터 concatenate 방식에 비해 이점을 나타내지 못했습니다. 이러한 결과는 모델의 해석 가능성과 랜덤 데이터 concatenate가 모델 강건성에 미치는 영향에 대한 추가 조사가 필요함을 시사합니다.



### Reinforced Disentanglers on Random Unitary Circuits (https://arxiv.org/abs/2411.09784)
Comments:
          9 pages, 7 figures, 1 table. Submitted to QIP 2025

- **What's New**: 본 연구는 무작위 클리포드 회로에서 두 큐빗 게이트로 구성된 브릭 월 패턴을 사용하여 효율적인 disentangler를 찾는 작업을 진행하였습니다. 특히, proximal policy optimization (PPO) 알고리즘을 통해 최적의 측정을 학습하여 양자 얽힘(entanglement) 엔트로피를 최소화하는 방법을 모색했습니다.

- **Technical Details**: disentangler는 연속적인 얽힘(entangling) 레이어 사이에 삽입된 일련의 투영(projection) 측정으로 정의됩니다. PPO 알고리즘을 사용하여 투영을 추가하거나 삭제하는 비트 플리핑(bit flipping) 작업을 통해 측정의 최적 위치를 선택하는 것으로 학습합니다. 최종 상태의 평균 von Neumann 엔트로피를 최소화하는 방향으로 보상을 제공합니다.

- **Performance Highlights**: 무작위 양자 회로를 disentangle하기 위해 필요한 측정의 수가 측정 유도 위상 전환(measurement-induced phase transition) 연구의 수치적 결과보다 현격히 적다는 것을 보여주었습니다. 이는 PPO를 통한 학습이 최적의 disentangler 패턴을 잘 특성화할 수 있음을 나타냅니다.



### Combining Machine Learning Defenses without Conflicts (https://arxiv.org/abs/2411.09776)
- **What's New**: ML(머신러닝) 모델은 다양한 보안(security), 프라이버시(privacy), 공정성(fairness) 위험에 노출되어 있습니다. 이러한 위험으로부터 보호하기 위한 기존의 방어(defense) 기술들을 효과적으로 조합할 수 있는 새로운 기법, Def\Con을 제안합니다.

- **Technical Details**: Def\Con은 기존 방어 기법들을 조합하여 효과적인 방어 조합을 찾기 위한 원칙적인 접근법을 제공합니다. 이 기법은 정확성(accuracy) 90%, 여러 조합에 대한의 효율적 판단 가능성과 비침습성(non-invasive), 확장성(scalable)을 갖추고 있습니다.

- **Performance Highlights**: Def\Con을 활용하면 기존 방법에 비해 효과적인 방어 조합 식별의 평균 정확도가 90%로 향상되며, 새로운 조합에 대해서도 81%의 정확성을 달성할 수 있습니다. 이는 기본적인 난이도(naïve technique) 기준의 40% 및 36%에 비해 월등한 성능입니다.



### Beyond Static Tools: Evaluating Large Language Models for Cryptographic Misuse Detection (https://arxiv.org/abs/2411.09772)
- **What's New**: 이 연구는 암호화 API 사용의 오용 발견을 위한 기존의 정적 분석 도구인 CryptoGuard, CogniCrypt, Snyk Code와 최신의 대형 언어 모델 (Large Language Models, LLMs)인 GPT 및 Gemini의 효과를 비교합니다.

- **Technical Details**: 연구는 OWASP, CryptoAPI, MASC 데이터 세트를 기반으로 정적 도구와 LLM의 성능을 평가했습니다. LLM의 반응을 ACTIONABILITY 및 SPECIFICITY로 평가하여 이를 통해 개발자에게 유용한 조언을 제공하는지 확인했습니다.

- **Performance Highlights**: 연구 결과, GPT 4-o-mini는 CryptoAPI 및 MASC 데이터 세트에서 현존하는 정적 분석 도구를 능가했지만, OWASP 데이터 세트에서 성능이 떨어졌습니다. ChatGPT는 CryptoAPI와 OWASP 모두에서 높은 탐지율을 기록했으나, OWASP에서 높은 거짓 긍정률을 보였습니다.



### Partial Multi-View Clustering via Meta-Learning and Contrastive Feature Alignmen (https://arxiv.org/abs/2411.09758)
- **What's New**: 본 논문에서는 데이터 분석에서 발생하는 부분적인 다중 뷰 클러스터링의 문제를 해결하기 위해 대조 학습(contrastive learning)에 기반한 새로운 이중 최적화 프레임워크를 제안합니다. 이를 통해 불완전한 다중 뷰 데이터에서 잠재 특징의 일관성을 극대화하고 클러스터링 성능을 향상시킵니다.

- **Technical Details**: 제안된 PVC-MC(PVC via Meta-Learning and Contrastive Feature Alignment) 방법은 대조 학습을 활용하여 불완전한 다중 뷰 데이터의 잠재 표현을 정렬합니다. KNN과 Vision Transformer를 기반으로 한 독립적인 자가 표현 레이어를 추가하여 각 뷰의 고유한 특징을 활용하며, 메타 학습(meta-learning)과 자가 감독 학습(self-supervised learning)을 통해 뷰 가중치를 동적으로 조정합니다. 이 과정을 통해 결측된 정보를 채우고 클러스터링 정확성을 극대화합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 BDGP 및 HW 데이터셋에서 기존 최첨단 클러스터링 모델에 비해 뛰어난 성능을 보여 줍니다. 특히 복잡하고 불완전한 다중 뷰 데이터 처리에 있어 우수한 결과를 기록하였습니다.



### Spatio-Temporal Jump Model for Urban Thermal Comfort Monitoring (https://arxiv.org/abs/2411.09726)
- **What's New**: 이 연구는 도시 환경에서의 온열 쾌적성(thermal comfort)을 개선하기 위한 새로운 방법론인 spatio-temporal jump model (ST-JM)을 제안합니다. 이 모델은 시공간적인 데이터의 클러스터링을 통해 지속성을 고려하여 온열 쾌적성을 동적으로 분석하는 데 중점을 두고 있습니다.

- **Technical Details**: ST-JM은 통계적 jump 모델의 개념을 바탕으로 하여, 시공간에 따른 데이터의 밀집성과 변화 양상을 동시에 고려하며 결측치(missing data) 및 불균형 샘플링 문제를 효과적으로 처리합니다. 이 모델은 혼합형 데이터셋을 처리할 수 있는 강력한 기능을 제공합니다.

- **Performance Highlights**: 이 연구는 싱가포르의 14개 기상 관측소에서 수집된 시간별 환경 데이터를 통해 ST-JM의 유효성을 검증하였습니다. 시뮬레이션 결과, 이 모델은 클러스터를 매우 정확하게 판별하며, 기존의 온열 쾌적성 측정 방법보다 더 적합한 예측 기능을 보였습니다.



### NFRs in Medical Imaging (https://arxiv.org/abs/2411.09718)
- **What's New**: 본 연구는 의료 영상 애플리케이션에서의 비기능 요구사항(Non-functional Requirements, NFRs)의 중요성을 강조하며, 이를 개선하기 위한 프레임워크 개발을 목표로 한다.

- **Technical Details**: 연구는 덴마크 한 병원에서 여러 관련자들과의 정성적 방법을 사용하여 비기능 요구사항의 종류를 파악하였다. 주요 비기능 요구사항으로는 효율성(Efficiency), 정확성(Accuracy), 상호운용성(Interoperability), 신뢰성(Reliability), 사용성(Usability), 적응성(Adaptability), 공정성(Fairness) 등이 있다.

- **Performance Highlights**: 해당 연구는 현재의 의료 영상 솔루션에서 AI 기술을 더 원활히 통합하기 위한 NFR 프레임워크의 필요성을 명확하게 주장하며, 향후 의료 영상 모델들의 병원 내 도입률을 높이는 데 기여할 것으로 기대된다.



### Machine learning approaches to explore important features behind bird flight modes (https://arxiv.org/abs/2411.09714)
Comments:
          6 pages, 5 figures

- **What's New**: 본 연구는 여러 종의 새들이 보여주는 비행 스타일의 다양성을 머신 러닝 기법을 사용하여 분석하였다. 비행 스타일의 차이에 기여하는 여러 가지 생리적 및 형태학적 요소의 상대적 중요성을 정량화한 첫 번째 사례이다.

- **Technical Details**: 연구에는 635종의 이주 조류의 체중, 날개 길이, 부화 기간 등의 형질 데이터가 사용되었으며, Feature Importance (FI)와 SHAP (SHapley Additive exPlanations) 값을 통해 각 특징의 상대적 중요성을 분석하였다.

- **Performance Highlights**: 기존의 전통적인 계통 발생 로지스틱 회귀와의 비교를 통해 높은 순위의 특징은 유사하지만 NJ 트리에서의 전체 가중치 분포와 클러스터링 패턴에는 차이가 있음이 발견되었다. 이는 상관된 형질로부터 생물학적으로 유용한 거리 행렬을 구성하는 복잡성을 강조한다.



### Feature Selection via Dynamic Graph-based Attention Block in MI-based EEG Signals (https://arxiv.org/abs/2411.09709)
Comments:
          4 pages, 2 figures, 1 table, Name of Conference: International Conference on Brain-Computer Interface

- **What's New**: 본 연구에서는 motor imagery (MI) 관련 특징을 강화하면서 낮은 상관관계를 가진 특징을 줄이는 end-to-end deep preprocessing 방법을 제안했습니다. 이 방법은 EEG 신호의 MI 작업에 대한 평가 성능을 증진시킬 수 있는 가능성을 보였습니다.

- **Technical Details**: 제안된 방법은 temporal, spatial, graph, 그리고 similarity 블록으로 구성되어 있으며, 각각의 블록은 MI 신호의 중요 특징을 더 효과적으로 추출하고 완화하는 데 기여합니다. 그래프 기반의 합성곱(convolution) 연산을 사용하여 전극 간 관계를 파악하고, 병합된 작용을 통해 처리된 특징들은 기존의 딥러닝 모델에 효과적으로 통합되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 BCI Competition IV의 공개 데이터셋에서 기존 모델들과 통합했을 때 성능을 향상시키고 MI 작업의 특징 분포를 더 클러스터된 형상으로 개선하는 것을 보여주었습니다.



### Decoding Fatigue Levels of Pilots Using EEG Signals with Hybrid Deep Neural Networks (https://arxiv.org/abs/2411.09707)
Comments:
          4 pages, 3 figures, 1 table, Name of Conference: International Winter Conference on Brain-Computer Interface

- **What's New**: 이번 연구는 조종사의 피로 수준을 분류하는 데 있어 딥러닝(deep learning) 기술의 가능성을 입증한 최초의 연구입니다. 연구는 정상 상태(normal), 저 피로(low fatigue), 고 피로(high fatigue) 등 다양한 피로 수준을 효과적으로 구분하기 위한 모델을 개발하였습니다.

- **Technical Details**: 시험에 참여한 10명의 파일럿을 대상으로 전기 생리학적 신호인 뇌파(EEG) 데이터를 수집했습니다. 제안된 모델은 5개의 convolutional block과 1개의 long short-term memory (LSTM) block으로 구성되어 있으며, 이는 EEG 신호에서 중요한 특징을 추출하여 피로 수준을 분류합니다.

- **Performance Highlights**: 제안된 딥러닝 모델은 기존의 4가지 전통적인 모델보다 최소 0.0599 높은 0.8801의 평균 정확도를 기록하며 피로 수준을 분류하는 데 있어 우수한 성능을 보여주었습니다.



### Residual Multi-Task Learner for Applied Ranking (https://arxiv.org/abs/2411.09705)
- **What's New**: ResFlow는 e-commerce 플랫폼에서 사용되는 새로운 경량 멀티태스크 학습 프레임워크입니다. 이 방법은 후행 작업 네트워크의 해당 레이어 간에 잔여 연결(residual connections)을 생성하여 효율적인 정보 공유를 가능하게 합니다.

- **Technical Details**: ResFlow는 멀티태스크 학습(MTL)에서 효과적인 정보 전송을 지원하도록 설계되었습니다. 기존 멀티태스크 학습 방법이 가진 한계를 극복하기 위해 잔여 연결을 도입했습니다. 이 방법은 ‘click’ → ‘order’와 같은 순차적 의존성을 가진 태스크들에 적합하며 다양한 랭킹 단계와 시나리오에 통합될 수 있습니다.

- **Performance Highlights**: Shopee Search에서 진행된 온라인 A/B 테스트 결과에 따르면, ResFlow는 추가적인 시스템 지연 없이 OPU(order-per-user)가 1.29% 증가하는 성과를 거두었습니다. 이 프레임워크는 현재 Shopee Search의 Pre-rank 모듈에 완전히 배포되어 실용적인 가치를 입증하였습니다.



### Prices, Bids, Values: Everything, Everywhere, All at Onc (https://arxiv.org/abs/2411.09355)
- **What's New**: 이 논문에서는 Iterative Combinatorial Auctions (ICAs)의 설계를 분석하고, 머신 러닝 (ML)을 통해 경매 효율성을 극대화하기 위한 새로운 접근 방식을 제안합니다. 특히, Demand Queries (DQs)와 Value Queries (VQs)의 조합을 활용하여 효율성을 크게 향상시킨 새로운 ICA인 MLHCA를 소개합니다.

- **Technical Details**: MLHCA는 DQ 기반 라운드와 VQ 기반 추가 라운드를 결합한 하이브리드 ML 기반 경매 시스템입니다. 이 과정을 통해 경매 참여자들의 선호도를 효과적으로 수집하고, 경매의 최종 효율성을 높일 수 있습니다. 실험 결과, MLHCA는 이전의 SOTA(SOTA, State Of The Art) 알고리즘과 비교하여 쿼리 수를 40% 줄이면서도 더 높은 효율을 달성했습니다.

- **Performance Highlights**: MLHCA는 실제 경매 환경에서 다수의 쿼리를 통해 효율성을 최대화하고, 이용자들의 인지적 부담을 줄이는 동시에 수백만 달러의 복지 향상을 가져올 수 있습니다. 이 연구는 ICAs의 효율성을 10배까지 낮출 수 있는 방법을 제시하며, 실질적인 경제 영향을 미치는 새로운 기준을 설정합니다.



### NeuralDEM -- Real-time Simulation of Industrial Particulate Flows (https://arxiv.org/abs/2411.09678)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 NeuralDEM이라는 혁신적인 접근 방식을 소개합니다. 이는 전통적인 DEM(Discrete Element Method) 방법의 느린 수치적 프로세스를 심층 학습 기반의 빠르고 적응 가능한 대체 모델로 전환합니다.

- **Technical Details**: NeuralDEM은 DEM의 Lagrangian discretization을 기본 연속 필드로 취급하며, 추가 보조 필드로 거시적(macroscopic) 행동을 직접 모델링합니다. 또한, 산업 규모 시나리오의 실시간 모델링을 위해 확장 가능한 다중 분기 신경 연산자(multi-branch neural operators)를 도입합니다.

- **Performance Highlights**: NeuralDEM은 16만 개의 CFD 셀과 50만 개의 DEM 입자를 사용하는 유동화층 반응기(coupled CFD-DEM fluidized bed reactors)를 28초 동안 신뢰성 있게 모델링할 수 있습니다. 이는 공정 사이클을 훨씬 더 빠르게 진행할 수 있는 기회를 제공합니다.



### Approximate Probabilistic Inference for Time-Series Data A Robust Latent Gaussian Model With Temporal Awareness (https://arxiv.org/abs/2411.09312)
Comments:
          New revision added a space between "for" and "Time-Series" in the title

- **What's New**: 이번 논문에서는 비정상(non-stationary) 시계열(time series) 데이터의 복잡한 특성을 포착할 수 있는 확률적 생성 모델인 Time Deep Latent Gaussian Model (tDLGM)을 소개합니다.

- **Technical Details**: tDLGM은 Deep Latent Gaussian Model (DLGM)에서 영감을 받아 설계된 새로운 아키텍처로, 손실 함수(loss function)를 최소화하는 방식으로 학습됩니다. 이 모델은 데이터 오류에 강인한(robust) 특성을 가지며, 특히 데이터 트렌드를 고려하는 정규화기(regularizer)가 돋보입니다.

- **Performance Highlights**: 실험 결과, tDLGM은 복잡한 시계열 데이터를 성공적으로 재구성하고 생성할 수 있으며, 소음(noise)과 결함이 있는 데이터에 대해서도 강인한 성능을 나타냅니다.



### Reducing Reasoning Costs -- The Path of Optimization for Chain of Thought via Sparse Attention Mechanism (https://arxiv.org/abs/2411.09111)
Comments:
          The main text is 9 pages, totaling 13 pages; 5 figures, 3 tables; preprints have been submitted to NeurIPS 2024 Workshop MusIML and OpenReview

- **What's New**: 대형 언어 모델의 추론 비용 상승 문제를 해결하기 위해, 연구에서는 관련 토큰에만 집중하는 sparse attention 메커니즘을 제안했습니다.

- **Technical Details**: 연구자는 새로운 attention 메커니즘을 구축하고, 사용자 지정 GPT로 훈련된 GiantRabbit을 실험 도구로 사용했습니다. 실험은 MIT OpenCourseWare의 선형 대수 시험 문제를 해결하는 데 있어 이 모델과 o1 Preview를 비교했습니다.

- **Performance Highlights**: 실험 결과, GiantRabbit은 추론 시간 및 chain of thought 길이가 o1 Preview보다 유의미하게 낮았으며, sparse attention 메커니즘이 추론 과정을 줄이는 데 효과적임을 확인했습니다.




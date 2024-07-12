New uploads on arXiv(cs.CL)

### Is Your Model Really A Good Math Reasoner? Evaluating Mathematical Reasoning with Checklis (https://arxiv.org/abs/2407.08733)
Comments:
          35 pages, 10 figures, preprint

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 수학적 추론 능력을 포괄적으로 평가하기 위해 새로운 벤치마크 'MATHCHECK'을 도입했습니다. 기존의 벤치마크들은 문제 해결 능력에 집중되어 있어 모델 오버피팅의 위험이 컸으며, 실제 수학적 추론 능력을 정확하게 반영하지 못했습니다. MATHCHECK는 작업 범용성과 추론 강건성을 테스트하기 위한 체크리스트와 자동 생성 도구를 포함하여 LLMs의 수학적 추론 능력을 더욱 포괄적으로 평가할 수 있습니다.

- **Technical Details**: MATHCHECK는 수학적 추론 작업과 다양한 강건성 테스트 유형을 포함합니다. 이를 통해 문제 해결, 정답 여부 판단, 결과 판단, 과정 판단 등 다양한 작업을 평가할 수 있습니다. 또한, 원본 문제와 문제 이해, 불필요한 방해, 시나리오 이해 등 다양한 문제 변형 유형을 통해 모델의 추론 강건성을 테스트합니다. MATHCHECK-GSM과 MATHCHECK-GEO를 통해 각각 수학 텍스트 추론 능력과 다중 모달 수학 추론 능력을 평가할 수 있으며, 이는 기존의 GSM8k, GeoQA, UniGeo, Geometry3K 등을 기반으로 생성되었습니다.

- **Performance Highlights**: MATHCHECK를 사용하여 GPT-4o를 포함한 20개 이상의 LLMs와 11개의 MLLMs를 평가한 결과, 첨단 LLMs는 탁월한 성능을 나타내었으나 많은 다른 모델 군은 성능이 크게 저하되었습니다. 추가 실험 결과, 기존의 수학 벤치마크와 비교했을 때 MATHCHECK는 모델의 실제 수학적 추론 능력을 보다 선형적으로 반영한다는 것이 확인되었습니다.



### A Taxonomy for Data Contamination in Large Language Models (https://arxiv.org/abs/2407.08716)
Comments:
          19 pages, 8 figures, accepted to CONDA Workshop on Data Contamination @ ACL 2024

- **What's New**: 대규모 웹 자료로 학습된 대형 언어 모델 (LLM: Large Language Models)은 다양한 다운스트림 (downstream) 작업에서 뛰어난 성능을 보입니다. 그러나 평가 데이터셋이 사전 학습 말뭉치에 포함되어 모델 성능을 과장하는 데이터 오염 (data contamination) 문제가 증가하고 있습니다. 연구는 LLM 사전 학습 중 발생하는 다양한 유형의 오염을 분류하는 분류 체계를 제시하고, 어떤 유형이 가장 큰 위험을 초래하는지 식별합니다.

- **Technical Details**: 이 연구는 요약 (summarization)과 질문 응답 (question answering)이라는 두 주요 NLP 작업에서 오염이 성능에 미치는 영향을 분석합니다. 데이터 오염이란 평가 데이터셋이 사전 학습 말뭉치에 포함되는 경우를 말하며, 이는 평가 중 task 성능을 과대 평가하게 만듭니다. 연구는 오염 유형을 탐지하고 제거하는 정화 (decontamination) 과정의 복잡성을 탐구합니다.

- **Performance Highlights**: 오염 유형이 다양한데, 각 유형이 평가 중 작업 성능에 미치는 영향이 다릅니다. 특히 요약과 질문 응답 같은 주요 NLP 작업에 대해, 각 오염 유형이 성능에 어떤 영향을 미치는지를 자세히 밝히었습니다.



### GTA: A Benchmark for General Tool Agents (https://arxiv.org/abs/2407.08713)
Comments:
          Github repo: this https URL

- **What's New**: 최근 연구에서는 대형 언어 모델(LLMs)을 다양한 도구와 통합하여 범용적인 AI 에이전트를 개발하는 데 집중하고 있습니다. 기존의 도구 사용 평가와 실제 시나리오 간에는 차이가 존재한다는 문제가 있습니다. 이에 대응하기 위해 GTA라는 새로운 벤치마크가 제안되었습니다. 이 벤치마크는 실제 사용자 쿼리, 실제 배포된 도구, 실제 멀티모달 입력의 세 가지 주요 측면을 특징으로 합니다.

- **Technical Details**: GTA는 다음과 같은 기술적인 세부 사항을 포함합니다: 229개의 실제 세계 과제와 실행 가능한 도구 체인을 설계했으며, 이를 평가하기 위해 감지, 작업, 논리, 창의성 등의 범주에 걸친 도구를 제공하는 평가 플랫폼을 구축했습니다. 각 쿼리는 실행 가능한 도구 체인과 함께 제공되며, 14개의 다양한 도구를 포함합니다. 평가 메트릭스는 도구 호출 프로세스의 전체 과정을 세밀하게 다룹니다.

- **Performance Highlights**: 평가 결과에 따르면, 현재의 LLM들은 실제 사용자 쿼리에 대해 어려움을 겪고 있으며, GPT-4은 문제의 50% 이하를 완료했고 대부분의 LLM들은 25% 미만의 성과를 보였습니다. 이는 현존하는 LLM들이 실제 시나리오에서 도구 사용 능력에 있어 한계를 드러내고 있으며, 향후 연구 방향을 제공하고 있습니다.



### Uncertainty Estimation of Large Language Models in Medical Question Answering (https://arxiv.org/abs/2407.08662)
- **What's New**: 본 연구는 대형 언어 모델(LLMs)을 활용한 의료 질의응답 시스템에서 신뢰할 수 있는 불확실성 추정(UE) 방법을 제안합니다. 이를 통해 LLM이 생성한 정보의 정확성을 확인하고, 잘못된 정보를 탐지하는 방법을 제공하고자 합니다. 이 연구는 일반적으로 사용되는 기존 방법들이 의료 분야에서 불충분한 성능을 보임을 발견하였으며, 모델 크기와 신뢰성 사이의 상관관계를 관찰했습니다. 이를 해결하기 위해 확률 기반이 아닌 이중 검증 방법(Two-phase Verification)을 도입했습니다.

- **Technical Details**: 연구에서는 확률을 활용하지 않는 불확실성 추정 방법으로 이중 검증 방법을 제안했습니다. 첫 번째 단계에서 LLM이 초기 답변과 함께 단계별 설명을 생성합니다. 이후 설명 속 사실 주장들을 확인하는 검증 질문을 작성합니다. 두 번째 단계에서는 모델이 이러한 검증 질문에 독립적으로 답변한 후, 설명을 참조하여 다시 답변합니다. 두 세트의 답변 간 불일치를 통해 초기 답변의 불확실성을 측정합니다. 이 방법은 Llama 2 Chat 모델을 사용하여 세 가지 생의학적 질의응답 데이터셋에서 평가되었습니다.

- **Performance Highlights**: 제안된 이중 검증 방법은 다양한 데이터셋과 모델 크기에서 가장 높은 정확성과 안정성을 달성했으며, 모델의 크기가 커질수록 성능이 향상되는 것으로 나타났습니다. 벤치마크된 기존 방법들에 비해 제안된 방법은 탁월한 성능을 보였습니다.



### Towards Building Specialized Generalist AI with System 1 and System 2 Fusion (https://arxiv.org/abs/2407.08642)
- **What's New**: 최근에 공개된 논문에서는 AGI(Artificial General Intelligence)로 가는 중요한 단계로 SGI(Specialized Generalist Artificial Intelligence)를 소개했습니다. SGI는 특정 작업에서 인간 전문가를 능가하는 AI를 의미하며, 일반적인 능력도 함께 유지합니다. 이 논문은 SGI의 세 가지 단계를 제시하고, 현재 대형 언어 모델(LLM)의 한계를 해결하기 위해 SGI가 필요함을 논의합니다.

- **Technical Details**: 논문에서는 SGI를 개발하기 위한 개념적 프레임워크를 제안합니다. 이 프레임워크는 Systems 1과 2의 인지 처리(Cognitive Processing)를 통합하는 세 가지 층과 네 가지 주요 구성요소로 구성됩니다. 이는 개별 능력을 향상시키고 협업적 진화를 촉진하는 것에 중점을 둡니다. 또한 기존 AI의 세 가지 주요 단계를 설명하며, SGI가 AGI로 가는 중간 단계로서 역할을 설명합니다.

- **Performance Highlights**: SGI는 특정 작업에서 인간 전문가의 90%를 능가하면서도 다양한 작업에서 불숙련 인간과 비슷한 일반적인 능력을 유지합니다. 이는 LLM 기반의 AI 시스템이 특화된 능력을 활용해 높은 효율성과 피드백 효과를 통해 자율적으로 학습하고 적응할 수 있는 능력을 갖추도록 합니다.



### Tamil Language Computing: the Present and the Futur (https://arxiv.org/abs/2407.08618)
Comments:
          11 pages, This is the write-up of the address delivered at the 29th Annual Sessions of the Jaffna Science Association, held from March 29-31, 2023, at the University of Jaffna, Sri Lanka

- **What's New**: 이 논문은 언어 컴퓨팅(Language Computing)의 텍스트 처리 측면을 깊이 있게 탐구하며, 컴퓨터가 인간 언어를 이해하고 해석하며 생성할 수 있게 하는 방법에 대해 논의합니다. 특히 음성 인식(speech recognition), 기계 번역(machine translation), 감정 분석(sentiment analysis), 텍스트 요약(text summarization), 언어 모델링(language modelling) 등의 과제에 중점을 둡니다. 이 논문은 타밀어(Tamil)를 ASCII에서 유니코드(Unicode)로 전환하여 디지털 통신을 향상시킨 기초 작업을 강조합니다.

- **Technical Details**: 논문에서는 원시 데이터(raw data), 사전(dictionary), 용어집(glossary), 주석 데이터(annotated data), 계산 문법(computational grammars) 등을 포함한 계산 자원의 발전에 대해 논의합니다. 특히 언어적 주석(linguistic annotation)의 어려움, 트리뱅크(treebanks) 생성, 대규모 언어 모델(large language models) 훈련의 필요성을 강조합니다. 고품질 주석 데이터와 고급 언어 모델의 중요성을 언급합니다.

- **Performance Highlights**: 최근 딥러닝(deep learning)의 발전으로 컴퓨터는 독립적으로 학습하고 적응하는 능력을 더욱 향상시키게 되었습니다. 타밀어와 같은 언어의 실용적인 애플리케이션을 구축하여 일상적인 의사소통 필요를 충족시키는 것을 강조하며, 현재 기술의 격차를 지적합니다. 또한 역사적 텍스트의 디지털화, 연구 협력의 증대, 디지털 사용을 촉진하여 타밀어 언어 처리를 종합적으로 발전시키는 필요성을 강조합니다.



### Turn-Level Empathy Prediction Using Psychological Indicators (https://arxiv.org/abs/2407.08607)
- **What's New**: 이번 WASSA 2024 Empathy and Personality Prediction Shared Task를 위해 새로운 전환 수준의 공감감지 메소드를 제안했습니다. 이 메소드는 공감을 정서적 언어(Emotional Language), 관점이해(Perspective-Taking), 동정심과 자비(Sympathy and Compassion), 외향성(Extroversion), 개방성(Openness), 그리고 친화성(Agreeableness)이라는 여섯 가지 심리적 지표로 분해하여 분석합니다.

- **Technical Details**: 우리는 GPT-4o를 사용하여 데이터셋의 각 발화에 대해 여섯 가지 심리적 지표 수준을 감지하고, 이를 설명하는 문장을 생성했습니다. 이 추가 컨텍스트는 원본 발화와 결합되어 DeBERTa V3 분류기를 훈련시키는 데 사용되었습니다. 두 가지 분류 모델(DeBERTa-v3-Large fine-tuned 및 GPT-4o zero-shot classification)을 테스트했으며, 심리적 지표가 포함된 컨텍스트와 포함되지 않은 원본 발화를 비교했습니다.

- **Performance Highlights**: 정서적 언어(Emotional Language)가 가장 높은 양의 상관관계(0.481*)를 보였으며, 동정심과 자비(Sympathy and Compassion)도 강한 양의 상관관계(0.437*)를 보여주었습니다. 외향성(Extroversion)은 공감 반응과 부정적 상관관계(-0.152*)를 보여 외향적 행동이 공감과 반드시 일치하지 않음을 시사합니다. GPT-4o와 DeBERTa 모델의 성능을 비교한 결과, DeBERTa는 높은 Pearson 상관 계수와 F1 점수를 기록하여 심리적 지표를 포함하면 공감 감지 모델의 성능이 크게 향상됨을 발표했습니다.



### On the Universal Truthfulness Hyperplane Inside LLMs (https://arxiv.org/abs/2407.08582)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 환각(hallucination) 문제를 해결하기 위해 모델 내 보편적 진실성 초평면(universal truthfulness hyperplane)의 존재 여부를 조사합니다. 기존 연구들은 내부 표현을 통해 환각을 탐구했으나, 분포 외 데이터에 대해 일반화하는데 어려움을 겪었습니다. 이를 극복하기 위해, 우리는 40개 이상의 다양한 데이터셋으로 훈련하고 광범위한 평가를 수행하였습니다.

- **Technical Details**: 이 연구에서는 다양한 태스크와 도메인에서의 진실성과 거짓을 분류할 수 있는 진실성 초평면을 찾기 위해 17개의 서로 다른 태스크 카테고리를 포함한 폭넓은 데이터셋을 구축했습니다. 로지스틱 회귀(Logistic Regression, LR)와 질량 평균(Mass Mean, MM)과 같은 선형 probing 방법을 사용하여 LLMs의 숨겨진 상태에서 진실성을 추출했습니다.

- **Performance Highlights**: 분석 결과, 훈련 데이터셋의 다양성을 증가시키면 모든 시나리오에서 성능이 크게 향상되며, 데이터 샘플의 양은 덜 중요한 역할을 했습니다. 우리의 probing 방법은 크로스 태스크 정확도 약 70%를 달성하며, 다수의 데이터셋에 훈련된 경우에도 우수한 성능을 보여 데이터 효율성과 범용성을 입증했습니다.



### Autoregressive Speech Synthesis without Vector Quantization (https://arxiv.org/abs/2407.08551)
- **What's New**: MELLE는 텍스트-음성 합성(TTS)에서 벡터 양자화를 필요로 하지 않고 연속적인 멜-스펙트로그램 프레임을 직접 생성하는 새로운 언어 모델링 접근법입니다. 이 모델은 예측 손실(regression loss)과 제안된 스펙트로그램 플럭스 손실 함수(spectrogram flux loss)를 사용하여 연속 값 토큰의 확률 분포를 모델링하며, 변이 예측(variational inference)을 포함하여 출력 다양성과 모델의 견고성을 향상시킵니다.

- **Technical Details**: MELLE는 텍스트와 이전 멜-스펙트로그램 프레임을 조건으로 연속적인 멜-스펙트로그램 프레임을 자동 회귀적으로 생성하며, 벡터 양자화된 토큰을 사용할 때 발생하는 문제를 해결합니다. 이를 위해 교차 엔트로피 손실(cross-entropy loss)을 예측 손실(regression loss)로 대체하고 스펙트로그램 플럭스 손실(spectrogram flux loss)을 도입하여 예측되는 스펙트로그램의 변이를 촉진하고 반복 문제를 제거합니다. 또한, 변이 예측(variational inference)을 통해 샘플링 메커니즘을 설계하여 생성된 오디오 샘플의 다양성을 높입니다.

- **Performance Highlights**: MELLE는 VALL-E와 그 변형 모델들에 비해 여러 성능 지표에서 우수한 성능을 보여주며, 특히 WER와 MOS와 같은 주관적 평가에서 뛰어난 결과를 냅니다. 실험 결과에 따르면 MELLE는 VALL-E 2와 객관적 지표에서 비슷한 수준을 유지하면서 주관적 지표에서 우수한 성능을 보입니다. 특히, MELLE는 zero-shot TTS 평가에서 WER 1.47%를 달성하여 VALL-E와 VALL-E 2에 각각 47.9%, 8.1%의 상대적 개선을 보였습니다. 주관적 평가에서 MELLE는 인간 청취자들에게 더 긍정적인 평가를 받으며, MOS와 CMOS, 그리고 SMOS에서 기존 모델들은 물론 원본 지문 대비 높은 성능을 나타냈습니다.



### Investigating LLMs as Voting Assistants via Contextual Augmentation: A Case Study on the European Parliament Elections 2024 (https://arxiv.org/abs/2407.08495)
- **What's New**: LNargetructed 대형 언어 모델(LLMs)은 유례없는 자연어 이해(Natural Language Understanding) 능력을 보여줍니다. 본 연구는 LLMs의 정치적 편견과 정치적 추론 능력을 미국 맥락에서 탐구한 최근 작업을 배경으로 합니다. 이번 연구에서는 MISTRAL과 MIXTRAL 모델을 감사하고 최신 'EU and I' 투표 지원 설문지를 기반으로 정당의 입장을 예측하는 정확성을 평가합니다.

- **Technical Details**: MISTRAL과 MIXTRAL 모델을 사용하여 'EU and I 2024' VAA 설문지를 기반으로 정치 정당의 입장을 예측합니다. 입력 문맥을 늘리기 위해 Retrieval-Augmented Generation (RAG)과 Self-Reflection을 사용하여 성능 향상을 꾀합니다. 실험에서는 독일, 프랑스, 이탈리아, 스페인의 상위 5개 인기 정당과 유럽 의회에서 대표되는 전체 유로파티(27개 정당)를 포함합니다.

- **Performance Highlights**: MIXTRAL 모델은 평균 82%의 높은 정확도를 보였으며, 전문가가 큐레이션한 정보로 입력 문맥을 강화하면 약 9%의 유의미한 성능 향상이 발생했습니다. 반면, MISTRAL 모델은 RAG 방식을 활용한 경우 약 8% 성능 향상이 나타났습니다.



### Investigating Public Fine-Tuning Datasets: A Complex Review of Current Practices from a Construction Perspectiv (https://arxiv.org/abs/2407.08475)
- **What's New**: 대규모 모델 분야의 급속한 발전과 함께, 파인튜닝(fine-tuning) 관련 연구도 크게 진전을 이루었습니다. 해당 연구는 현재 공개된 파인튜닝 데이터셋을 중심으로 데이터 구성의 관점에서 이뤄졌으며, 파인튜닝 데이터셋의 진화와 분류를 통해 그 발전 경로를 도출하고자 합니다. 주요 대규모 언어 모델(LLMs)의 파인튜닝을 위한 데이터 생성 및 데이터 증강 방법 등을 다루고 있습니다.

- **Technical Details**: 이 리뷰에서는 대규모 언어 모델(LLMs)의 파인튜닝 데이터셋을 구조의 관점에서 고유하게 분류하였습니다. Fine-tuning 단계의 데이터는 OpenAI InstructGPT 논문[1]에서 제안된 지도 학습(SFT)와 선호도 모델링(RM, PPO)을 포괄합니다. 데이터셋은 크게 '데모 데이터셋(Demonstration Datasets)', '비교 데이터셋(Comparison Datasets)', 그리고 '제너럴리스트 데이터셋(Generalist Datasets)'으로 분류됩니다. 또한 텍스트와 멀티모달(Modality) 형식을 기준으로 데이터셋을 정리하였습니다.

- **Performance Highlights**: InstructGPT[1] 프로젝트는 파인튜닝 데이터셋의 중요한 이정표로, 일반 작업을 더 자연스럽고 인터랙티브하게 수행하기 위한 지침을 제공합니다. Flan Collection[2](2023) 의 연구에서 영감을 받아, 이전 세대 데이터셋을 통해 NLP 작업을 집합하고 이를 인스트럭션 형식으로 변환하는 방법론을 사용하였습니다. 현재 연구에서는 다양한 데이터 생성 기법을 학습하여 고품질 파인튜닝 데이터셋 구축의 중요성을 강조하고 있습니다.



### Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks (https://arxiv.org/abs/2407.08454)
- **What's New**: 대규모 언어 모델(LLM)의 계산 비용 문제를 해결하기 위해, KV 캐시를 효율적으로 병합(merge)하는 새로운 방법인 'KVMerger'를 제안합니다. 이 방법은 특히 긴 컨텍스트 처리 작업에서 메모리 사용을 줄이면서 성능 저하를 최소화합니다.

- **Technical Details**: KVMerger는 각각의 시퀀스 내 토큰 수준에서 키(key) 상태가 높은 유사성을 보인다는 관찰에 기반하고 있습니다. 이를 통해, 효과적인 병합 세트를 식별할 수 있는 알고리즘을 개발하고, Gaussian 커널 가중치 병합 알고리즘을 사용해 각 병합 세트 내의 모든 상태를 선택적으로 병합합니다.

- **Performance Highlights**: 실험 결과, 제안된 KVMerger는 LongBench 및 ZeroScrolls 벤치마크에서 기존의 KV 캐시 압축 방법인 H2O 및 CaM보다 성능이 우수하며, 50% 및 35% KV 캐시 예산의 경우에도 더 나은 성능을 발휘합니다.



### Are Large Language Models Really Bias-Free? Jailbreak Prompts for Assessing Adversarial Robustness to Bias Elicitation (https://arxiv.org/abs/2407.08441)
- **What's New**: 이 연구는 최신 대형 언어 모델(Large Language Models, LLMs)이 가지는 편향성을 조사하고, 이를 드러내기 위해 어떻게 알려진 프롬프트 엔지니어링(prompt engineering) 기술을 이용할 수 있는지를 탐구합니다. 해당 연구는 다양한 규모의 LLM들을 대상으로 한 광범위한 실험을 통해, 모델이 편향되거나 부적절한 응답을 생성할 수 있음을 확인했습니다.

- **Technical Details**: LLMs는 성별, 인종, 성적 지향, 종교, 사회경제적 지위, 장애, 나이와 관련된 고정 관념 및 편향을 포함한 여러 편향성을 가지고 있습니다. 연구는 이러한 편향성을 드러내기 위해 설계된 프롬프트와 공격 기술을 사용했습니다. 예를 들어, 'jailbreak prompts'를 사용하여 모델의 안전 필터를 우회하고, 일반적으로 제한된 콘텐츠를 생성하게 했습니다.

- **Performance Highlights**: 연구는 LLM이 고도화된 정렬 프로세스에도 불구하고, 'jailbreak prompts'를 통해 여전히 조정되어 편향된 응답을 생성할 수 있음을 확인했습니다. 실험 결과 모델의 공정성과 견고함을 평가하고, 고정 관념과 반대 고정 관념 사이에서 응답이 어떻게 나오는지 분석했습니다. 특히 다양한 공격 기술로 모델의 안전 필터를 우회할 수 있는 효과를 평가하여, 모델의 실제 회복력을 측정했습니다.



### Beyond Instruction Following: Evaluating Rule Following of Large Language Models (https://arxiv.org/abs/2407.08440)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 규칙 준수 능력을 평가하기 위해 RuleBench라는 새로운 벤치마크를 소개합니다. 기존 연구들은 대부분 명령 준수 능력에 초점을 맞췄으나, 본 연구에서는 규칙 준수와 명령 준수를 명확히 구분하여 평가합니다.

- **Technical Details**: RuleBench는 LLMs가 다양한 추론 작업에서 규칙을 어떻게 따르는지 평가합니다. 규칙은 단순한 명령이 아니라, 특정 조건 하에서 작동하는 추상적 정책입니다. 따라서, LLMs는 상황에 맞춰 적절한 규칙을 선택하고 이를 기반으로 의사 결정을 내려야 합니다. RuleBench는 다양한 규칙 형식과 수량, Chain-of-Thought(CoT) 적용 여부, 반사실 시나리오 등의 영향을 평가합니다.

- **Performance Highlights**: 실험 결과, 다양한 LLMs는 여전히 규칙 준수 능력에 한계를 보였습니다. 특히 폐쇄형 LLMs가 규칙 준수 시나리오에서 더 우수한 성능을 보였으며, 일부 개방형 LLMs (예: Llama-3-8B)는 다양한 차원에서 균형 잡힌 성능을 보여 경쟁력을 입증했습니다. 연구팀은 규칙 트리거링, 규칙 적용, 규칙 실행, 공식 규칙 및 반사실 규칙 준수 등의 5가지 차원에서 LLMs의 규칙 준수 능력을 분류하였습니다.



### Self-training Language Models for Arithmetic Reasoning (https://arxiv.org/abs/2407.08400)
Comments:
          Appeared in ICLR 2024 LLMAgents

- **What's New**: 연구진은 새로운 데이터를 수집하지 않고, 기존 데이터에 대한 자동 피드백을 활용하여 언어 모델의 연산 추론 능력을 향상시키는 방안을 탐구하였습니다. 연구 결과, 오프라인(self-training)과 온라인(self-training) 모두에서 모델이 크게 개선될 수 있음을 발견했습니다. 특히, 온라인 self-training에서는 preference optimization이 더 안정적이고 새로운 문제 유형에서도 견고한 성능을 보였습니다.

- **Technical Details**: 이번 연구는 두 가지 변형의 self-training 방식을 구현했습니다. 오프라인 변형에서는 한 번의 반복으로 모델의 응답을 생성하고, 이를 다시 훈련에 사용합니다. 반면 온라인 변형에서는 모델이 예측에 대한 피드백을 즉각적으로 획득합니다. 그 결과, 오프라인 방식에서는 감독 학습(supervised)과 preference optimization 방법 모두가 유사한 성능 향상을 가져왔으나, 온라인 방식에서는 preference optimization이 더 나은 결과를 보였습니다.

- **Performance Highlights**: 실험 결과, self-training이 원본 모델에 적절한 훈련 신호를 제공하며 새로운 데이터 없이도 성능을 크게 향상시켰습니다. 온라인 설정에서는, preference optimization이 새로운 문제 유형에서도 견고한 성능을 유지하는데 더 효과적이었습니다. 실험에 사용된 모델은 FLAN 모델이며, AQuA-RAT, GSM8K 등 다섯 가지의 다른 수학 데이터셋에서 평가되었습니다.



### AutoBencher: Creating Salient, Novel, Difficult Datasets for Language Models (https://arxiv.org/abs/2407.08351)
Comments:
          preprint

- **What's New**: 새로운 벤치마크 (benchmark) 생성을 위해 AutoBencher를 개발했습니다. 이 시스템은 언어 모델을 활용해 주제별로 효과적인 데이터셋을 자동으로 검색하고, 이 데이터셋의 유의미성, 난이도, 참신성을 기반으로 최적의 벤치마크를 만듭니다.

- **Technical Details**: 세 가지 주요 목표: 유의미성(실제로 중요한 주제를 다루는지), 참신성(이전 벤치마크에서는 드러나지 않은 새로운 모델 성능 경향성을 밝혀내는지), 난이도(기존 모델로는 성적이 낮을 만큼 어려운지)를 기준으로 벤치마크를 최적화합니다. 이 접근 방식을 통해 AutoBencher는 평가 주제를 제시하고 해당 주제에 대한 신뢰할 수 있는 데이터셋을 구성하며, 메트릭 기반 로컬 검색 알고리즘을 활용해 난이도를 최적화합니다.

- **Performance Highlights**: AutoBencher가 만든 데이터셋은 기존 벤치마크 대비 27% 더 참신하고, 22% 더 어려운 문제를 포함하고 있습니다. 예를 들어, Gemini Pro 모델은 기존 역사 벤치마크에서는 좋은 성과를 보였지만, AutoBencher가 발견한 'Permian Extinction'과 'Fordism' 같은 주제에서는 매우 낮은 성적을 냈습니다. 반면 OpenAGI-7B 모델은 COVID-19 주제에서 놀라운 성과를 보였습니다.



### RB-SQL: A Retrieval-based LLM Framework for Text-to-SQL (https://arxiv.org/abs/2407.08273)
- **What's New**: 최근 텍스트-투-SQL(text-to-SQL) 작업에서, RB-SQL이라는 새로운 검색 기반(Retrieval-Based) LLM(대형 언어 모델) 프레임워크가 제안되었습니다. 이 모델은 데이터베이스를 사전 처리하고 유의미한 정보를 추출하여 적절한 프롬프트 엔지니어링을 가능하게 합니다. RB-SQL은 BIRD와 Spider라는 공공 데이터셋에서 경쟁력 있는 여러 기준 성능을 뛰어넘는 결과를 보여주었습니다.

- **Technical Details**: RB-SQL은 질문과 관련성이 높은 테이블, 컬럼 및 SQL 스켈레톤(SQL Skeleton)을 검색하는 3개의 독립된 모듈로 구성됩니다. 이러한 모듈은 질문과 SQL 데이터 타입(테이블, 컬럼, SQL 스켈레톤) 사이의 유사성을 계산합니다. 이를 통해 프롬프트 내의 중복 정보를 줄이고, in-context learning에 참조 가치가 높은 몇 가지 예제들을 선택할 수 있습니다.

- **Performance Highlights**: 실험 결과, RB-SQL 모델은 BIRD와 Spider 데이터셋에서 여러 강력한 기준 모델보다 우수한 성능을 보였습니다. RB-SQL은 질문과 가장 관련 있는 테이블과 컬럼을 검색하여 텍스트-투-SQL 작업의 효율성을 높이고, SQL 스켈레톤을 통해 정확한 SQL 생성 과정을 안내합니다.



### LLMs' morphological analyses of complex FST-generated Finnish words (https://arxiv.org/abs/2407.08269)
Comments:
          To appear at the CMCL Workshop at ACL 2024

- **What's New**: 이 논문은 Neural NLP 시스템들이 실제로 인간이 사용하는 문법 규칙을 학습하는지 여부를 조사합니다. 특히, 핀란드어 복잡한 명사 형태의 형태 분석 과제를 통해 최신 LLMs를 평가합니다. GPT-4-turbo는 어느 정도의 어려움을 겪었으며, GPT-3.5-turbo는 더 많은 어려움을 겪었고, 소형 모델인 Llama2-70B와 Poro-34B는 거의 전혀 성공하지 못했습니다.

- **Technical Details**: 이 작업은 Finite State Transducer (FST) 도구를 사용하여 생성된 복잡한 핀란드어 명사 형태를 사용합니다. 이러한 형태는 LLM의 훈련 세트에 등장하지 않았을 가능성이 높아, 형태 일반화 능력을 필요로 합니다. Omorfi 도구를 사용하여 500k 어휘의 핀란드어 명사들을 다양한 수, 문법적 격, 소유 격미사 조합으로 굴절시켜 약 25M 단어 형태를 생성하였고, 이 중 무작위로 선택된 2000개의 굴절된 명사 샘플을 테스트 세트로 사용했습니다.

- **Performance Highlights**: 중요 성능 결과로, GPT-4-turbo가 일정한 한계를 드러냈으며, GPT-3.5-turbo와 소형 모델들은 거의 실패에 가까운 결과를 보였습니다. 이는 LLMs가 복잡한 형태학적 규칙을 일관되게 학습하기 어려울 수 있음을 시사합니다.



### Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting (https://arxiv.org/abs/2407.08223)
Comments:
          Preprint

- **What's New**: 최근 논문에서는 Speculative RAG를 소개합니다. 이는 대형 언어 모델(LLM)의 생성 능력과 외부 지식 소스를 결합하여 보다 정확하고 최신의 응답을 제공하는 프레임워크입니다. 특히, Speculative RAG는 더 작은 특화된 LLM이 병렬로 여러 RAG 초안을 생성하고, 이를 보다 큰 일반 모델이 검증하는 방식을 사용합니다. 이 방법은 다양한 관점을 제공하면서도 입력 토큰의 수를 줄이는 데 중점을 둡니다.

- **Technical Details**: Speculative RAG는 일부 문서를 기준으로 나누고, 각각의 부분 집합에서 다수의 초안을 생성합니다. 이러한 초안은 특화된 작은 모델이 생성하며, 이후 일반 모델이 각 초안을 한 번의 검증 과정을 거쳐 최종 결과로 통합합니다. 이는 외부 데이터베이스에서 검색된 문서를 다양한 관점으로 분할하여, 중복을 최소화하고 다양성을 극대화하는 방법을 사용합니다. 또한, 각 초안에 대한 생성 확률을 기반으로 자신감을 평가하여 최종 응답을 결정합니다.

- **Performance Highlights**: TriviaQA, MuSiQue, PubHealth, ARC-Challenge 벤치마크 테스트에서 Speculative RAG는 기존 RAG 시스템보다 더욱 빠른 응답 시간을 보여주며, 정확도는 최대 12.97% 향상되고 지연 시간은 51% 줄어드는 성과를 보였습니다. 이는 특히 PubHealth 벤치마크에서 두드러지게 나타났습니다.



### Generating Contextually-Relevant Navigation Instructions for Blind and Low Vision Peop (https://arxiv.org/abs/2407.08219)
Comments:
          Accepted as RO-MAN 2024 Late Breaking Report

- **What's New**: 이 논문에서는 새로운 이미지 및 목표 데이터 셋을 구성하여 큰 사전 훈련된 언어 모델(LLM)과 비전 및 언어 모델(VLM)이 맹인과 저시력(BLV) 사용자에게 유용한 내비게이션 지침을 제공할 수 있는지 조사했습니다. 또한, BLV 사용자와의 인터뷰를 통해 다양한 시나리오에 대한 지침 선호도를 분석했습니다.

- **Technical Details**: 문제는 사용자가 목적지를 향해 이동하는 경로를 설명하는 지침을 생성하는 것으로 정의됩니다. 모델은 자아중심적 이미지와 의미론적 태스크 컨텍스트를 기반으로 지침을 생성합니다. 데이터를 구성하기 위해, 주방, 사무실, 일반 실내 및 실외 환경에서 VizWiz 이미지를 선택하고 사용자 목표에 맞게 주석을 달았습니다.

- **Performance Highlights**: 시각 사용자 설문 조사 결과, 생성된 지침이 올바르고 유용하다고 평가되었습니다. 인간이 생성한 지침이 가장 정확하다고 평가되었지만, LLM과 VLM이 생성한 지침도 유사한 유용성을 보여주는 것으로 나타났습니다. 이 연구는 BLV 사용자에게 유용한 내비게이션 지원 시스템 개발에 중요한 인사이트를 제공했습니다.



### System Report for CCL24-Eval Task 7: Multi-Error Modeling and Fluency-Targeted Pre-training for Chinese Essay Evaluation (https://arxiv.org/abs/2407.08206)
- **What's New**: CCL-2024에서 열린 중국어 에세이 유창성 평가(Chinese Essay Fluency Evaluation, CEFE) 과제에 대한 우리의 접근 방식과 결과를 소개합니다. 본 연구에서는 다양한 트랙(Track)에서 성과를 최적화하기 위해 여러 방법을 적용했습니다.

- **Technical Details**: 트랙 1에서는 세밀한 오류 유형에 대한 예측을 최적화하기 위해 이진 분류 모델(binary classification models)을 사용했으며, China Learner 4W 코퍼스(corpus)를 기반으로 한 모델을 훈련시켰습니다. 트랙 2에서는 문장별로 다수의 오류 유형을 포함하는 의사 데이터셋(pseudo-dataset)을 구성하여 성능을 향상시켰습니다. 트랙 3에서는 백번역(back-translation)을 통해 유창성 평가된 의사 데이터를 생성하여 사전 훈련(pre-training)을 수행했으며, 대칭 교차 엔트로피 손실(Symmetric Cross Entropy loss)을 이용한 NSP 기반 전략을 통해 긴 종속성(long dependencies)을 완화하고 문맥(context)을 캡처했습니다.

- **Performance Highlights**: 트랙 3에서는 첫 번째 자리를 차지하며 주요 성과를 달성했습니다. 우리의 방법들은 중국어 에세이 유창성 평가의 주요 과제를 효과적으로 해결했습니다.



### fairBERTs: Erasing Sensitive Information Through Semantic and Fairness-aware Perturbations (https://arxiv.org/abs/2407.08189)
- **What's New**: 이번 논문에서는 표준 성적 편향(Stereotypical Bias)을 완화하기 위한 새로운 접근법인 fairBERTs를 제안합니다. 이는 Generative Adversarial Network(GAN)를 활용하여 BERT 시리즈 모델에서 보호된 민감 정보를 삭제함으로써, 모델의 공정성을 확보하려는 프레임워크입니다. 이를 통해 언어 모델에서 편향을 효과적으로 완화하고, 다양한 실제 응용 사례에서 성능을 유지할 수 있음을 입증했습니다.

- **Technical Details**: fairBERTs는 BERT 시리즈 모델의 시퀀스 출력에서 생성된 의미적 및 공정성 인식 섭동(perturbations)을 숨은 표현(hiden representations)에 합성하여 민감 정보를 삭제하는 방식으로, 편향을 완화합니다. 이는 대립적 학습(adversarial learning)을 통해 이루어지며, 다른 유사한 구조의 모델로도 전이 가능한 특성을 가집니다. 이 접근법은 주로 분류 작업에서의 편향 완화를 목표로 하고 있으며, 모델의 정확성을 저해하지 않습니다.

- **Performance Highlights**: 두 가지 실제 작업에서 광범한 실험을 통해, fairBERTs가 기존의 세 가지 편향 완화 방법과 비교하여 상당히 뛰어난 성능을 보였으며, 공정성을 확보하면서 모델의 유용성을 유지하는 데 성공했습니다. 이는 민감 정보 삭제 및 공정성 개선을 통한 BERT 모델의 활용 가능성을 더욱 넓혔습니다.



### Beyond Text: Leveraging Multi-Task Learning and Cognitive Appraisal Theory for Post-Purchase Intention Analysis (https://arxiv.org/abs/2407.08182)
- **What's New**: 이 연구는 사용자 행동 예측을 위한 감독 학습 모델을 다중 태스크 학습 프레임워크(Multi-task learning frameworks)에 기반하여 평가하였습니다. 인지 평가 이론(Cognitive Appraisal Theory)에 의거하여 사용자의 자기 표현과 심리적 속성을 반영하여 예측을 개선하는 방법을 제안합니다.

- **Technical Details**: 연구는 PEACE-Reviews 데이터셋을 활용하여 감정적 경험과 소비 후 행동(Post-Consumption Behaviors, PCB)을 모델링합니다. 리뷰 텍스트는 20개의 인지 평가 척도와 8개의 감정 등급으로 주석이 달려 있으며, 리뷰 텍스트와 인지적, 감정적 척도가 결합된 다중 모드 모델(Multi-modal models)을 사용합니다. BERT 모델과 피드포워드 뉴럴 네트워크(FFNN)를 사용하여 예측 성능을 테스트합니다.

- **Performance Highlights**: 텍스트와 사용자 특성만을 예측하는 모델에 비해 사용자 언어와 특성을 통합한 모델은 예측 성능이 현저히 향상되었습니다. 텍스트 기반 모델만으로 수행하기 어려운 영역에서 인지적 구조를 통합함으로써 사용자 행동 예측의 정확성을 높일 수 있다는 점이 강조되었습니다.



### Looks can be Deceptive: Distinguishing Repetition Disfluency from Reduplication (https://arxiv.org/abs/2407.08147)
- **What's New**: 이 논문은 중첩 (reduplication)과 반복 (repetition)의 차이를 연구한 최초의 대규모 연구를 소개합니다. 저자들은 힌디어, 텔루구어, 마라티어 텍스트에서 중첩과 반복이 단어 수준에서 주석된 새로운 데이터셋인 IndicRedRep을 공개했습니다.

- **Technical Details**: 연구진은 transformer-based models를 사용하여 중첩과 반복의 멀티 클래스 토큰 분류기를 평가했습니다. 중첩과 반복을 구분하기 위해 Reparandum-Interregnum-Repair 구조를 활용했습니다.

- **Performance Highlights**: 힌디어에서는 최대 85.62%, 텔루구어에서는 83.95%, 마라티어에서는 84.82%의 macro F1 점수를 달성하며 높은 정확도를 보였습니다.



### Automata-based constraints for language model decoding (https://arxiv.org/abs/2407.08103)
Comments:
          Accepted to CoLM 2024

- **What's New**: 이 논문은 언어 모델(LMs)이 형식 언어(formal language)에서 유효한 출력을 생성하는 방법을 자동자 이론(automata theory)을 활용해 해결하고자 합니다. 특히, API 호출이나 JSON과 YAML 같은 구조화된 데이터에서 유용한 정규 언어(regular languages)에 대해 효율적인 폐쇄 형식 솔루션(closed-form solution)을 도출했습니다. 나아가, 높은 분기 인자(branching factor) 문제를 해결하기 위한 실용적인 확장도 다루며, 결정론적 문맥 자유 언어(deterministic context-free languages)에 대해서도 유사한 솔루션을 확장했습니다.

- **Technical Details**: 논문은 다음과 같은 접근 방식을 제안합니다. 1) 현재 상태에 기반한 마스킹을 통해 다음 토큰의 유효성을 보장하기 위해 디코더 로짓(decoder logits)을 마스킹합니다. 2) 토큰화를 자동자 이론과 연산을 통해 공식 언어 제약에 맞게 매핑합니다. 주로 주어진 형식 언어를 수용하기 위해 고정된 상태(state)와 가장자리(edge)를 가진 유한 상태 자동자(Finite-state Automaton, FSA)를 사용합니다. FSA는 입력 기호와 상태의 집합 등으로 구성되어 있으며, 단일 단계로 중간 상태 변환을 관리합니다. 이 과정을 통해 형식 구조적 언어에서 문법적으로 올바른 출력이 생성되도록 유도합니다.

- **Performance Highlights**: 이 방법은 모델 크기에 독립적인 간단한 계산으로 구성되어 있어 거의 모든 언어 모델 아키텍처에 효율적이고 쉽게 적용할 수 있습니다. 기존의 토큰화 문제를 해결하기 위해 detokenization과 자동자 연산의 연결고리를 효율적으로 활용한 점이 특히 주목할 만합니다.



### How does Burrows' Delta work on medieval Chinese poetic texts? (https://arxiv.org/abs/2407.08099)
Comments:
          2 figures

- **What's New**: Burrows' Delta가 한문 텍스트에서도 효과적인지 분석한 연구입니다. 특히 유럽 언어와의 차이점을 가지고 있는 중세 중국 시에 대해 이 방법을 적용해 보았습니다.

- **Technical Details**: Burrows' Delta는 심리적, 언어적 기초는 명확하지 않지만, 대부분의 언어에서 성공적으로 저자 판별에 사용된 방법입니다. 유럽 언어와 달리 문장 부호와 공백이 없는 중국어 텍스트에서는 개별 문자를 기준으로 분석을 진행했습니다. 분석에는 'Complete Tang Poems' 데이터베이스를 사용했으며, 20명의 대표적인 당나라 시인을 대상으로 실험을 했습니다. Stylo 패키지를 사용해 가장 빈번하게 나타나는 100개의 문자를 기준으로 Delta를 계산했습니다.

- **Performance Highlights**: 실험 결과는 모든 5개의 데이터 버전에서 완벽하게 클러스터링이 이루어졌습니다. Delta는 동일 저자의 샘플을 다른 저자와 혼동하지 않고 정확하게 식별했습니다. 이는 Delta 방법이 유럽 언어뿐만 아니라, 시적 스타일이 규정된 전통에서도 효과적이라는 결론을 뒷받침합니다.



### RoLoRA: Fine-tuning Rotated Outlier-free LLMs for Effective Weight-Activation Quantization (https://arxiv.org/abs/2407.08044)
- **What's New**: 신규 연구로 RoLoRA라는 새로운 LoRA 기반 스키마가 제안되었습니다. 이는 weight-activation quantization을 효과적으로 적용하는 첫 번째 방법입니다. RoLoRA는 회전 방식을 사용하여 outlier를 제거하고, 회전 인식 파인 튜닝(rotations-aware fine-tuning)을 통해 outlier-free 특성을 유지합니다.

- **Technical Details**: 기존의 Low-Rank Adaptation(LoRA) 방법은 주로 weight-only quantization에 집중했으며, 이는 트레이닝 메모리 비용을 절감하였습니다. 그러나 weight와 activation을 동시에 quantize하는 방법은 상대적으로 많이 연구되지 않았습니다. RoLoRA는 LLM의 weight 매트릭스에 회전을 적용하여 outlier를 제거하고, 이를 통해 발생하는 quantization 오류를 줄입니다. 또한, 회전 인식 파인 튜닝을 활용하여 파인 튜닝 과정 동안 outlier-free 특성을 유지합니다.

- **Performance Highlights**: RoLoRA는 LLaMA2-7B/13B, LLaMA3-8B 모델을 대상으로 실험하였으며, 4-bit weight-activation quantization LLaMA2-13B 모델에서 LoRA 대비 최대 29.5%의 절대 정확도 향상을 달성하였습니다. 또한, MMLU 벤치마크에서 4-bit quantization(W4A4)에 대해 LoRA 대비 14.6 포인트의 성능 향상을 기록했습니다. 대형 멀티모달 모델(LMM)인 LLaVA-1.5-7B에서도 RoLoRA는 우수한 성능을 보여주었습니다.



### Knowledge Overshadowing Causes Amalgamated Hallucination in Large Language Models (https://arxiv.org/abs/2407.08039)
- **What's New**: 이 논문에서는 '지식 오버섀도잉' (knowledge overshadowing)이라는 새로운 환각(hallucination) 현상을 정의하고 이를 다루는 방법을 제안합니다. 대형 언어 모델(LLMs)이 다중 조건이 포함된 쿼리에 대해 특정 조건이 다른 조건을 가리는 현상을 실험을 통해 검증하였습니다. 이는 특히 훈련 데이터의 불균형에서 비롯된다고 주장하며, 환각을 예측하고 줄이기 위한 새로운 '자기 대조 디코딩' (self-contrastive decoding) 방법을 제안합니다.

- **Technical Details**: 지식 오버섀도잉은 훈련 데이터의 불균형과 조건 서술의 길이와 관련이 있습니다. 이 현상은 인기 있는 조건이 덜 인기 있는 조건을 가림으로써 발생하며, 앙각 비율이 증가합니다. 논문에서는 다양한 언어 모델과 데이터에 대해 실험을 진행하였으며, 상호 정보(Pointwise Mutual Information, PMI)를 사용하여 오버섀도잉 조건을 사전에 식별하고, 그 후 '대조 디코딩(contrastive decoding)'을 통해 이를 완화하는 방법을 사용했습니다.

- **Performance Highlights**: 제안된 방법은 환각 예측에 있어 최대 82%의 F1 점수를 기록했으며, 환각 억제율은 모델과 데이터셋에 따라 11.2%에서 39.4% 사이의 감소를 보여주었습니다.



### FsPONER: Few-shot Prompt Optimization for Named Entity Recognition in Domain-specific Scenarios (https://arxiv.org/abs/2407.08035)
Comments:
          accepted for publication at the 27th European Conference on Artificial Intelligence (ECAI-2024)

- **What's New**: 새로운 접근법인 FsPONER가 도메인 특화된 Named Entity Recognition (NER) 작업에 몇 샷 학습(few-shot learning) 최적화 방법을 도입했습니다. 이 접근법은 산업 제조 및 유지 보수와 같은 특화된 도메인에서 GPT-4-32K, GPT-3.5-Turbo, LLaMA 2-chat, Vicuna와 같은 여러 대형 언어모델(LLM)을 평가했습니다.

- **Technical Details**: FsPONER는 무작위 샘플링, TF-IDF 벡터 및 두 가지를 결합한 세 가지 몇 샷 선택 방법을 포함합니다. 이는 기존의 일반적인 GPT-NER 방법과 비교하여 몇 샷 예제 수가 증가함에 따라 그 성능을 비교했습니다. 해당 모델들은 fine-tuned BERT와 LLaMA 2-chat과도 성능을 비교했습니다.

- **Performance Highlights**: 실제 데이터가 부족한 시나리오에서 FsPONER는 TF-IDF와 함께 사용될 때 약 10% 높은 F1 점수를 기록하며, fine-tuned 모델들을 능가하는 성능을 보였습니다.



### DS@GT eRisk 2024: Sentence Transformers for Social Media Risk Assessmen (https://arxiv.org/abs/2407.08008)
Comments:
          Paper Submitted to CLEF 2024 CEUR-WS

- **What's New**: DS@GT 팀은 eRisk 2024에서 태스크 1과 3에 참여했습니다. 태스크 1에서는 Beck Depression Inventory (BDI-II)를 기반으로 우울증 증상을 예측하는 랭킹 시스템을 제안했으며, 태스크 3에서는 BERT 임베딩을 활용하여 사용자 게시물 기록을 기반으로 섭식 장애 증상을 예측했습니다.

- **Technical Details**: 태스크 1에서는 질문의 관련성을 이진 분류기로 훈련시켜 랭킹 시스템을 만들었지만, 이진 분류기가 랭킹에 적합하지 않음을 확인했습니다. 태스크 3에서는 BERT 임베딩을 사용하여 사용자 게시물의 벡터 표현을 학습해 섭식 장애 증상 예측에 활용했습니다. 문장 변환기(sentence transformers)는 텍스트 데이터의 표현에 매우 유용한 도구임을 확인했습니다.

- **Performance Highlights**: 태스크 1의 내부 검증에서 문장 변환기를 사용한 모델이 평균 F1 점수 89%와 평균 정확도 90%를 기록했습니다. 그러나 실제 테스트 셋에서는 모든 모델이 성능이 저조했습니다. 이것은 분류기를 랭커로 재활용할 수 있다는 가정이 잘못되었음을 시사합니다.



### Automated Neural Patent Landscaping in the Small Data Regim (https://arxiv.org/abs/2407.08001)
Comments:
          11 pages, 4 figures

- **What's New**: 이 논문은 자동화된 뉴럴 특허 구역화 시스템(neural patent landscaping system)을 소개합니다. 이 시스템은 최소한의 라벨된 예시를 사용하여 높은 성능을 발휘하며, 새로운 기술 영역의 특허 지도를 보다 효율적으로 구성할 수 있도록 디자인되었습니다. 특히, 기존 시스템에 비해 난이도가 높은 사례에서의 성능을 크게 향상시켰고, 적은 트레이닝 데이터로도 높은 성능을 달성하였습니다.

- **Technical Details**: 연구팀은 Abood와 Feltenberger의 'seed/anti-seed' 접근 방식을 활성 학습(active learning)과 결합하여 어려운 라벨된 예시를 수집하는 새로운 데이터 생성 절차를 제안했습니다. 또한, 기존 시스템의 성능을 향상시키기 위해 인용 네트워크(citation networks)와 CPC 코드 임베딩(CPC code embeddings) 같은 추가 특징들을 도입하였습니다. 이 새로운 접근 방식을 통해, 극히 적은 수의 라벨만으로도 높은 성능을 달성할 수 있습니다.

- **Performance Highlights**: 제안된 시스템의 성능은 '난이도가 높은' 예시에서 0.69의 F1 점수를 기록했으며, 기존 시스템의 0.6보다 향상된 성능을 보였습니다. 더 나아가, 단 24개의 예시로 전체적으로 0.75의 F1 점수를 달성함으로써, 수천 개의 예시를 필요로 했던 기존 연구들보다 훨씬 적은 데이터로도 유사한 성능을 보였습니다.



### Rel-A.I.: An Interaction-Centered Approach To Measuring Human-LM Relianc (https://arxiv.org/abs/2407.07950)
Comments:
          Preprint

- **What's New**: 이번 연구에서는 기존 '언어 모델(LM)'이 간단한 문장 완성에서 벗어나 복잡하고 다영역적인 인간과의 상호작용을 요구하는 환경에서 인간이 LMs를 신뢰하는 방식을 이해하기 위해 새로운 방법론인 Rel-A.I.를 도입했습니다. 기존 연구에서는 언어화된 확신(예: '난 이 답이 확실해')을 결정적인 신뢰 기준으로 삼았지만 이번 연구는 다양한 상호작용 문맥이 인간의 신뢰에 미치는 영향을 강조합니다.

- **Technical Details**: Rel-A.I.는 시스템 수준에서 인간이 LM이 생성한 '인식적 표지(epistemic markers)', 예를 들어 '내 생각에...' 혹은 '아마도...'를 신뢰하는 방식 측정하는 평가 방법입니다. 이 방법론은 세 가지 핵심 요소로 구성됩니다: 1) 사람의 신뢰 여부를 측정하는 자율태스크(self-incentivized task), 2) 퍼블릭 모델에서 가져온 인식적 표지, 3) AI 에이전트의 따뜻함과 능력에 대한 사용자 인식을 묻는 메타수준의 질문입니다.

- **Performance Highlights**: 연구 결과, 인간의 신뢰는 단순히 언어화된 확신에만 의존하지 않고 여러 상호작용적 맥락에 크게 영향을 받는다는 것을 발견했습니다. 예를 들어, '난 꽤 확신해' 같은 표현은 상황에 따라 신뢰 빈도가 최대 20%까지 변할 수 있습니다. 이전 상호작용, LM의 인간화 코드, 주제 영역 등이 신뢰 변동에 중요한 역할을 합니다.



### Transformer Circuit Faithfulness Metrics are not Robus (https://arxiv.org/abs/2407.08734)
Comments:
          CoLM 2024 Conference Paper. 11 page main body. 11 page appendix. 12 figures

- **What's New**: 본 연구는 신경망 내부에서 학습된 알고리즘을 역공학하는 '기계적 해석 가능성' (Mechanistic Interpretability, MI) 작업에 주목하며, 특정 작업에서 모델의 동작을 설명하는 '회로' (circuits)를 발견하는 것에 중점을 두고 있습니다. 본 연구는 회로의 '성실성' (faithfulness)을 측정하는 방법론의 일관성 부족 문제를 지적하고, 이를 개선하기 위해 다양한 실험 설계를 제안합니다. 또한, 새로운 라이브러리 AutoCircuit을 오픈 소스로 공개하여 효율적인 회로 발견 및 평가 도구를 제공합니다.

- **Technical Details**: 기계적 해석 가능성 연구의 핵심 측정 지표는 회로의 성실성(faithfulness)으로, 이는 회로가 전체 모델의 성능을 얼마나 잘 나타내는지 측정합니다. 이를 위해 우리는 transformer 언어 모델의 회로 성실성을 측정하는 다양한 방법론을 조사하고, 기존 방법론들이 작은 변화에도 민감하게 반응한다는 점을 발견했습니다. 본 연구는 '간접 객체 식별' 회로와 '문서 문자열' 회로, '스포츠 선수' 회로 등을 사례 연구로 분석하고, 자동 회로 발견 (automated circuit discovery)과 관련된 '최적 회로'를 연구합니다.

- **Performance Highlights**: 실험 결과는 기존의 회로 성실성 점수가 연구자들의 방법론적 선택에 의해 크게 좌우된다는 점을 보여줍니다. 따라서, 모델 내 회로의 성능을 정확히 평가하기 위해서는 더 명확하고 일관된 측정 기준이 필요합니다. 새로운 라이브러리 AutoCircuit은 기존 구현보다 성능이 뛰어나고 효율적인 회로 발견 및 평가 기능을 제공합니다.



### Emergent Visual-Semantic Hierarchies in Image-Text Representations (https://arxiv.org/abs/2407.08521)
Comments:
          Accepted to ECCV 2024. Project page: this https URL

- **What's New**: 기존의 비전 및 언어 모델(VLM)들은 이미지와 텍스트를 공유된 의미 공간에서 분석할 수 있지만, 이미지 설명 텍스트의 계층적 특성을 명시적으로 모델링하지 않습니다. 그러나 연구팀은 이러한 VLM이 계층적 이해를 보여준다는 것을 발견했습니다. 이를 더 탐구하고 최적화하기 위해 Radial Embedding(RE) 프레임워크를 제안하고, HierarCaps 데이터셋을 기여했습니다.

- **Technical Details**: RE 프레임워크는 EC에 기반한 초기 작업들이 하이퍼볼릭 공간에서 훈련된 것과는 달리, 유클리드 공간에서 사전학습된 VLM의 기하학적 구성을 반영합니다. HierarCaps 데이터셋은 73,000개의 이미지와 논리적 계층으로 배열된 여러 텍스트를 포함하며, 대규모 언어 모델(LLM)과 자연어 추론(NLI)을 사용하여 생성됩니다.

- **Performance Highlights**: 사전 학습된 VLM들은 계층적 이해력에서 어떠한 사전 훈련 없이도 우수한 성능을 보였으며, 이후 RE 프레임워크를 통한 파인 튜닝을 통해 더 향상되었습니다. HierarCaps 데이터셋과 기존 벤치마크를 통한 평가에서 재확인되었습니다.



### Lynx: An Open Source Hallucination Evaluation Mod (https://arxiv.org/abs/2407.08488)
- **What's New**: LLYNX는 최신 상태의 환각 검출 능력을 가진 대형 언어 모델(LLM)로, 현실 세계의 복잡한 환각 시나리오에 대한 고급 추론을 수행할 수 있습니다. 이를 평가하기 위해 우리는 다양한 현실 세계 도메인에서 15,000개의 샘플로 구성된 종합적인 환각 평가 벤치마크인 HaluBench를 소개합니다.

- **Technical Details**: LYNX는 Llama-3-70B-Instruct를 다중 도메인의 데이터로 미세 조정하여 훈련되었으며, CovidQA, PubmedQA, DROP, FinanceBench와 같은 QA 데이터셋에서 예제를 추출하고 의미 변화(semantic perturbation)를 통해 환각된 정답을 생성합니다. LLM-판정 모델과 비교하여 LYNX는 참조 없이도 높은 품질의 환각 검출을 할 수 있으며, 완전한 오픈 소스로 제공됩니다.

- **Performance Highlights**: LYNX는 HaluBench에서 GPT-4o, Claude-3-Sonnet 및 기타 폐쇄형 및 오픈소스 모델을 능가하는 성능을 입증하였습니다. Lynx는 70B와 8B 버전으로 제공되며, 8B 모델은 작은 크기와 비용으로도 높은 품질의 평가를 생산할 수 있습니다.



### On the attribution of confidence to large language models (https://arxiv.org/abs/2407.08388)
Comments:
          22 pages, 0 figures

- **What's New**: 최근 연구에서는 대형 언어 모델(Large Language Models, LLMs)인 GPT-4와 Gemini와 같은 모델들에게 **신뢰도**(credences)를 부여하는 내용에 대한 새로운 논의를 제기하였다. LLM 평가 문헌에서는 LLM이 특정 명제에 대해 가지는 자신감의 정도를 언급하는 것이 일반적이지만, 이에 대한 이론적 근거가 불확실하다는 것이다.

- **Technical Details**: LLM은 새로운 단어를 예측하기 위해 대규모 텍스트 데이터를 학습하는 신경망이다. 신뢰도 부여는 특정 질문에 대한 답변의 분포를 분석하거나, LLM이 직접 신뢰도를 보고하게 하는 실험적 기법을 통해 이루어진다. 하지만 이는 주로 인간에게 적용되는 개념으로, LLM에게도 적용하는 것이 철학적 분석이 필요하다는 의견이다. LLM들이 믿음과 같은 심리적 상태를 가진다는 것은 논란의 여지가 있으며, 이론적 증거가 불충분하다.

- **Performance Highlights**: 연구진은 LLM 신뢰도 부여가 기본적으로 문자 그대로 해석될 수 있으며, LLM 신뢰도의 존재 가능성을 주장하지만 현재의 증거가 불확실하다고 밝혔다. 또한, LLM 신뢰도를 평가하는 현행 기법에는 중요한 의구심이 제기될 수 있으며, 이러한 기법들이 실제로 진실을 반영하지 않을 가능성도 있다고 강조하였다.



### Skywork-Math: Data Scaling Laws for Mathematical Reasoning in Large Language Models -- The Story Goes On (https://arxiv.org/abs/2407.08348)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)의 수학적 추론 능력을 향상시키는 근본적인 요인들을 조사합니다. 저자들은 현대 LLMs의 수학적 추론 능력에 대한 데이터 스케일링 법칙이 아직 포화되지 않았다고 주장하며, 데이터 양이 증가할수록 모델의 품질이 향상됨을 강조합니다. 이를 뒷받침하기 위해 저자들은 제안한 250만 개의 사례를 포함한 Skywork-MathQA 데이터셋을 사용하여 일반적인 7B LLMs에서 감독하에 미세 조정(SFT)된 Skywork-Math 모델 시리즈를 소개합니다.

- **Technical Details**: Skywork-Math 7B 모델은 단순한 SFT 데이터를 사용하여 경쟁 수준의 MATH 벤치마크에서 51.2%, GSM8K 벤치마크에서 83.9%의 뛰어난 정확도를 기록했습니다. 이 모델들은 GPT-4의 초기 버전을 넘어서면서 수학적 추론 능력이 있는 7B 파라미터 언어 모델에서도 강한 성과를 내고 있음을 입증했습니다. 저자들은 데이터 품질과 양을 모두 보장하는 새로운 두 단계의 데이터 합성과 모델 SFT 파이프라인을 통해 이러한 성능을 달성했습니다.

- **Performance Highlights**: Skywork-Math 모델 시리즈는 두 개의 주요 벤치마크에서 뛰어난 성과를 보였습니다. 경쟁 수준의 MATH 벤치마크에서 51.2%의 정확도, GSM8K 벤치마크에서 83.9%의 정확도를 기록했습니다. 특히, 이 모델들은 GPT-4의 초기 버전을 능가할 만큼 뛰어난 성능을 보여, 수학적 추론 능력을 평가하는 데 있어 중요한 지표로 자리잡았습니다.



### Towards Explainable Evolution Strategies with Large Language Models (https://arxiv.org/abs/2407.08331)
Comments:
          Accepted at ESANN 2024

- **What's New**: 이번 논문은 자기적응적(Evolution Strategies, ES)과 대형 언어 모델(Large Language Models, LLMs)을 결합하여 복잡한 최적화 과정의 설명 가능성을 향상시키는 접근 방식을 소개합니다. 기존의 ES에 재시작 메커니즘을 도입해 최적화 과정을 탐색하고, LLM을 활용하여 이러한 과정을 요약하고 사용자 친화적인 형태로 제공합니다. Rastrigin 함수에 대한 사례 연구를 통해 ES 최적화의 복잡성을 투명하게 전달하는 방법을 구현했습니다.

- **Technical Details**: 제안된 XES(Explainable Evolution Strategies)는 자기적응적 ES에 재시작 메커니즘을 추가하여 복잡한 최적화 환경을 탐색합니다. 알고리즘은 진화 기록에 따라 변이 스텝 크기를 동적으로 조정하고, 정체 현상이 발생하면 재시작 메커니즘을 통해 상태를 초기화합니다. 상세한 로그 파일을 생성하여 피트니스 값, 변이 스텝 크기, 정체 사건 및 재시작 이벤트를 기록합니다. LLM은 이러한 로그 데이터를 분석하여 요약하고, 사용자가 이해하기 쉬운 형태로 설명합니다. 네 가지 다른 프롬프트 전략(Zero-Shot, Few-Shot, Chain-of-Thought, Few-Shot CoT)을 사용하여 LLM이 필요 정보를 추출하고 요약하도록 유도합니다.

- **Performance Highlights**: Rastrigin 함수에 대한 실험 결과를 통해 XES의 성능과 투명성을 입증했습니다. σ-self-adaptive (μ+λ)-ES를 사용하여 10차원 Rastrigin 함수의 최적화를 진행하였고, 다양한 길이의 세 개의 로그 파일을 생성하여 분석했습니다. Llama2:70b, Llama3:70b, Mistral 7b, Mixtral 8x7b 모델을 사용하여 텍스트 생성을 수행했으며, Mixtral 8x7b가 Few-Shot Prompting 전략에서 최고의 성능을 보였습니다. LLM은 최적 피트니스 값, 최악 피트니스 값, 수렴 행동 및 지역 최적 지점 등을 명확하게 설명했습니다.



### A Text-to-Game Engine for UGC-Based Role-Playing Games (https://arxiv.org/abs/2407.08195)
Comments:
          13 pages,11 figures

- **What's New**: 이 논문은 단순 텍스트 입력을 복잡한 상호작용 롤플레잉 게임(RPG) 경험으로 변환하는 텍스트-게임 엔진을 소개합니다. 이 엔진은 플레이어의 행동에 실시간으로 반응하여 게임 스토리를 다중모달 형식으로 동적으로 렌더링하고 캐릭터, 환경, 메커니즘을 조정합니다. 'Zagii'라는 게임 엔진을 개발하여 다양한 장르의 수백 가지 RPG 게임을 지원하고 수만 건의 온라인 사용자 게임플레이를 가능하게 했습니다.

- **Technical Details**: 이 프레임워크는 생성 AI(generative AI)의 발전을 활용하여 텍스트를 기반으로 한 RPG 게임의 전 과정을 실시간으로 다룰 수 있도록 설계되었습니다. 기본 모델(foundation models)을 사용하여 간단한 텍스트 입력을 복잡한 게임 환경으로 변환합니다. 엔진은 다중모달(multi-modal) 형식으로 게임 스토리를 렌더링하고, 플레이어의 행동에 따라 게임 캐릭터, 환경, 메커니즘을 실시간으로 조정합니다.

- **Performance Highlights**: 'Zagii' 게임 엔진은 다양한 장르의 수백 가지 RPG 게임을 성공적으로 지원했으며, 수만 번의 온라인 사용자 게임플레이를 제공했습니다. 이는 본 프레임워크의 효과를 검증하는 데이터로 활용됩니다.



### Automatic Generation of Web Censorship Probe Lists (https://arxiv.org/abs/2407.08185)
Comments:
          To appear in the Proceedings on Privacy Enhancing Technologies 2024

- **What's New**: 이 논문은 인터넷 검열 탐지에서 중요한 역할을 하는 도메인 프로브 리스트(domain probe lists)를 자동으로 생성하는 방법을 탐구합니다. 기존의 수동 또는 크라우드소싱 방식은 시간 소모적이고 오류가 발생하기 쉬우며, 변화하는 검열 환경에 적절히 대응하지 못하는 문제점을 가지고 있었습니다. 이 연구는 웹 검열 측정을 위해 포괄적이고 최신 상태의 프로브 리스트를 자동으로 생성할 수 있는 새로운 방법을 제시합니다.

- **Technical Details**: 연구진은 다양한 언어의 기존 테스트 리스트에서 139,957개의 고유 URL을 초기 세트로 사용하여 새로운 후보 페이지를 생성하였습니다. 이 URL들의 콘텐츠를 분석하여 주제와 키워드를 추출하고 이를 검색 엔진에 입력하여 119,255개의 새로운 URL을 얻었습니다. 그런 다음, 11개의 전 세계 위치의 서버에서 4개월 동안 각 URL에 접속 시도를 하여 연결 가능성과 검열 신호를 확인했습니다. 이 과정에서 복잡한 자연어 처리 파이프라인이 필요했으며, 여기에는 언어 식별, 페이지 토큰화, 번역, 주제 및 키워드 할당이 포함되었습니다.

- **Performance Highlights**: 이 방식으로 총 35,147개의 도메인에서 119,255개의 고유 URL을 생성했으며, 초기 데이터셋에 포함되지 않은 1,400개의 도메인을 새로 발견했습니다. 특히, 중국의 경우 1,000개 이상의 새로운 도메인이 검열된 것으로 확인되었습니다. 이는 수동 방식보다 비용 효율적이고 사용자에게 위험을 가하지 않으면서도 더 많은 검열된 페이지를 탐지할 수 있음을 보여줍니다.



### Privacy-Preserving Data Deduplication for Enhancing Federated Learning of Language Models (https://arxiv.org/abs/2407.08152)
- **What's New**: 본 논문에서는 연합 학습(Federated Learning) 환경에서의 데이터 중복 제거 문제를 해결하는 획기적인 프로토콜을 제안하고 있습니다. Efficient Privacy-Preserving Multi-Party Deduplication (EP-MPD)라는 프로토콜을 도입하여 여러 클라이언트의 데이터셋에서 데이터 중복을 효율적으로 제거하면서도 데이터 프라이버시를 침해하지 않습니다.

- **Technical Details**: EP-MPD는 모듈식 구조로 설계되었으며, Private Set Intersection(PSI) 프로토콜의 두 가지 새로운 변형을 활용합니다. 이 프로토콜은 클라이언트의 데이터를 공유하지 않고도 중복을 제거할 수 있도록 지원합니다.

- **Performance Highlights**: 대규모 언어 모델의 연합 학습에서 중복 제거의 상당한 이점을 입증했습니다. 예를 들어, EP-MPD는 Perplexity에서 최대 19.61% 향상, 실행 시간에서 최대 27.95% 감소를 관찰하였습니다. EP-MPD는 프라이버시와 성능을 효과적으로 균형잡아 대규모 응용 프로그램에 가치 있는 솔루션이 됩니다.



### How Well Can a Long Sequence Model Model Long Sequences? Comparing Architechtural Inductive Biases on Long-Context Abilities (https://arxiv.org/abs/2407.08112)
Comments:
          Work In Progress. 9 pages

- **What's New**: 최근 연구에서 장문 데이터를 다룰 수 있다 주장하는 선형 시퀀스 모델(linear sequence models)과 상태공간 모델(state-space models)이 실제로 이론적 주장만큼 효과적인지 평가하였습니다. 연구 결과, 이런 모델들은 이론적으로는 무한 길이의 시퀀스를 다루는 것이 가능하나, 실질적으로는 많은 문제점이 있음을 확인했습니다. 특히, 순환 신경망(RNN) 기반 모델들이 주의(attention) 기반의 장문 언어 모델(long-context LLM)들과 동일한 설정에서 여전히 어려움을 겪고 있다는 점을 강조했습니다.

- **Technical Details**: 이 연구는 다양한 형태의 모델(순수 시퀀스 레이어 모델, 주의 모델, 혼합 모델)의 동작 방식을 분석했습니다. 이를 위해 합성 데이터(synthetic data)와 실제 데이터(realistic data)를 사용하여 테스트했으며, 모델들이 훈련된 컨텍스트 길이를 초과할 때 그 성능을 면밀히 관찰했습니다. 특히, 합성 작업에서 needle-in-a-haystack(NIAH)와 패스키 회수(passkey retrieval) 같은 과제를 사용하여 모델의 컨텍스트 크기를 평가했습니다.

- **Performance Highlights**: 모든 모델들은 훈련된 컨텍스트 길이를 초과하는 시퀀스에 대한 예측 성능이 떨어진다는 것을 확인했습니다. 또한, 같은 작업이라도 시퀀스의 형식에 따라 모델의 추론 능력이 크게 달라질 수 있다는 것을 발견했습니다. 이러한 결과는 장문 시퀀스 모델이 이론적 원리에도 불구하고 실질적인 한계를 갖고 있음을 시사하며, 이를 개선하기 위한 추가 연구가 필요함을 강조합니다.



### Virtual Agents for Alcohol Use Counseling: Exploring LLM-Powered Motivational Interviewing (https://arxiv.org/abs/2407.08095)
- **What's New**: 이 논문에서는 알코올 사용 상담을 위한 동기면담(Motivational Interviewing, MI)을 수행할 수 있는 가상 상담사를 개발하는 데 대형 언어 모델(Large Language Models, LLM)을 적용하는 새로운 방법을 소개합니다. 이 가상 상담사는 사용자 친화적인 가상 플랫폼에 통합되어 공감과 현실 감 있는 상호작용을 제공합니다. 초기 연구 결과에 따르면 LLM 기반의 가상 상담사는 인간 상담사의 공감적이고 적응적인 대화 기술과 유사하게 작동한다는 것을 보여줍니다.

- **Technical Details**: 본 연구는 GPT-4 (2024년 3월 버전)을 주요 대화 엔진으로 사용하여 알코올 사용 상담 시나리오를 개발합니다. GPT-4를 선택한 이유는 이 모델이 의료 대화 생성에서 다른 LLM들보다 뛰어나다는 것이 입증되었기 때문입니다. 연구는 공공적으로 이용 가능한 MI 데이터셋을 사용하여 LLM이 생성한 개별 응답을 인간 전문가의 실제 알코올 사용 상담 시뮬레이션에서 만든 응답과 비교합니다. 또한, LLM 기반의 가상 상담사가 전체 상담 세션을 어떻게 일관되게 유지하며 전문가 표준을 충족하는 치료적 상호작용을 수행하는지 보여줍니다.

- **Performance Highlights**: 연구는 세 가지 주된 연구 질문을 탐구합니다: 1) 인간과 LLM이 생성한 상담 응답이 언어적 타당성, 안전성, MI 원칙에 얼마나 부합하는지 비교, 2) LLM 기반 가상 상담사가 MI 요소를 사용하여 행동 변화를 효과적으로 유도할 수 있는 정도, 3) 사용자 관점에서 LLM 기반 가상 상담사로서의 강점과 약점. 연구 가설은 LLM이 생성한 응답이 인간 상담사가 생성한 응답에 비해 유의미하게 열등하지 않으며, 클라이언트와 임상적 관점 모두에서 높은 대화 품질을 제공한다는 것입니다. 초기 결과는 이 가설을 지지하며, LLM 기반 가상 상담사가 알코올 상담에서 유용한 도구가 될 잠재력이 있음을 시사합니다.



### A Critical Review of Causal Reasoning Benchmarks for Large Language Models (https://arxiv.org/abs/2407.08029)
Comments:
          AAAI 2024 Workshop on ''Are Large Language Models Simply Causal Parrots?''

- **What's New**: 최근의 논문은 대규모 언어 모델 (LLMs)의 인과 추론 (causal inference) 및 추론 능력을 평가하기 위해 다양한 벤치마크가 제안되었으나, 이러한 벤치마크 중 다수는 도메인 지식의 검색만으로 해결될 가능성이 크다는 점에 의문을 제기합니다. 따라서 논문에서는 보다 철저한 인과 추론 정의를 포함하는 최근 벤치마크의 동향을 살펴보고, 유용한 벤치마크가 만족해야 할 기준을 제안합니다.

- **Technical Details**: 논문은 인과 추론의 기존 '인과 계층' (causal hierarchy)을 기반으로 현재 벤치마크를 분류합니다. Zhang et al. (2023)은 세 가지 수준의 인과 추론 능력을 평가하는 계층 구조를 제안합니다: 
1. 도메인 지식을 사용한 인과 관계 식별
2. 데이터에서 새로운 지식 발견
3. 행동의 결과를 정량적으로 추정. 
Pearl과 Mackenzie (2018)도 유사한 '인과 사다리' (ladder of causation)를 제안합니다: 
1. 통계적 연관성 설명 (seeing)
2. 개입의 결과 형식화 (doing)
3. 대체 시나리오 상상 (imagining). 
기존의 많은 작업들은 첫 번째 수준에만 도달하여 인과 관계 식별만을 평가합니다.

- **Performance Highlights**: 문헌과 데이터셋을 조사해보니, 많은 기존 작업들이 LLM의 인과 추론 능력을 제대로 평가하지 못하고 사전 지식에 의존하고 있음을 발견했습니다. 인과 관계 식별의 제한된 평가, 다지선다형 태스크의 문제 등으로 인해 LLM이 실제로 인과 추론을 수행하는 능력을 충분히 입증하지 못하고 있습니다. 이러한 문제를 해결하기 위해 제안된 기준은 인과 추론 평가를 위한 보다 높은 수준의 자체적 평가를 지향합니다.



### Search, Examine and Early-Termination: Fake News Detection with Annotation-Free Evidences (https://arxiv.org/abs/2407.07931)
Comments:
          ECAI 2024 paper. Fudan University & NVIDIA. To appear

- **What's New**: 최신 연구에서는 가짜 뉴스 탐지를 강화하기 위해 증거(evidence)의 중요성을 인식하며, 기존 방법의 한계를 극복한 새로운 접근법 SEE(Search-Examine-Early Termination)을 제안합니다. 이 방식은 뉴스 기사 검색, 증거 검토, 조기 종료 메커니즘을 통해 주어진 뉴스가 실제인지 아닌지를 예측합니다.

- **Technical Details**: SEE는 크게 세 단계로 구성됩니다. 첫째, 뉴스 기사 검색(Search) 단계에서는 뉴스 기사를 쿼리로 사용해 관련 온라인 자료를 찾고, 해당 자료의 제목을 증거로 사용합니다. 둘째, 검토(Examine) 단계에서는 Transformer 기반의 디코더를 사용해 뉴스와 증거를 시퀀셜하게 융합합니다. 셋째, 조기 종료(Early Termination) 단계에서는 각 단계별로 예측 신뢰도를 평가해 추가적인 증거 검토를 중단하고 최종 예측을 제공합니다.

- **Performance Highlights**: 무처리 데이터셋(Weibo21, GossipCop)과 전처리된 데이터셋(Snopes, PolitiFact)에서 SEE 방법을 실험한 결과, 최첨단(FND) 접근법보다 높은 정확도를 보였습니다. 또한, 증거의 순서 변경, 제거 등 다양한 테스트 시나리오에서도 일관된 성능을 유지했습니다.



### Solving General Natural-Language-Description Optimization Problems with Large Language Models (https://arxiv.org/abs/2407.07924)
- **What's New**: 최근 최적화 문제를 다루기 위한 혁신적인 프레임워크 OptLLM이 제안되었습니다. OptLLM은 외부 솔버(solver)를 결합하여 LLM(Large Language Models)과 함께 작동하며, 자연어로 입력된 사용자 질의를 수학적 공식과 프로그래밍 코드로 변환한 후 솔버를 호출해 결과를 계산합니다. 또한, OptLLM은 다중 라운드 대화를 지원하여 최적화 문제의 모델링 및 해결 과정에서 점진적으로 개선할 수 있습니다.

- **Technical Details**: OptLLM은 세 개의 주요 모듈로 구성됩니다. 첫 번째 모듈은 상호 작용 개선(interaction refinement) 모듈로, 사용자와 상호 작용하여 문제 설명을 완성하고 입력된 내용이 유효한 최적화 문제인지 확인합니다. 두 번째 모듈은 변환기(converter) 모듈로, 문제 설명을 수학적 공식과 프로그래밍 코드로 변환하고 코드의 문법을 검사합니다. 마지막 모듈은 응답기(responser) 모듈로, 코드를 외부 솔버로 보내고 결과를 수집하여 자연어로 해석합니다. 기본 프로그래밍 언어로는 Alibaba에서 설계한 MAPL(MindOpt Algebraic Programming Language)을 사용하며, 주요 솔버로 MindOpt가 사용됩니다.

- **Performance Highlights**: OptLLM은 다양한 LLM과 함께 작동하며, 자체 개발한 대규모 최적화 데이터셋을 사용한 실험 결과, 세부 조정을 통해 정확도가 향상된 모델이 프롬프트 기반 모델에 비해 성능이 우수한 것으로 나타났습니다. 2023년 6월부터 일부 기능이 시험 서비스 중입니다.



New uploads on arXiv(cs.IR)

### FAR-Trans: An Investment Dataset for Financial Asset Recommendation (https://arxiv.org/abs/2407.08692)
Comments:
          Accepted at the IJCAI-2024 Workshop on Recommender Systems in Finance (Fin-RecSys)

- **What's New**: 이 논문에서는 Financial Asset Recommendation (FAR) 분야에서 공공 데이터셋의 부재라는 문제를 해결하기 위해 최초의 공개 데이터셋인 FAR-Trans를 소개합니다. 이 데이터셋은 대형 유럽 금융기관으로부터 얻은 시계열 가격정보와 소매 투자자 거래를 포함하고 있습니다. 이를 통해 11개의 FAR 알고리즘을 비교 평가하여 향후 연구에 기준점을 제시합니다.

- **Technical Details**: FAR-Trans 데이터셋은 주식, 채권 및 뮤추얼 펀드와 같은 다양한 자산 유형에 대해 시계열 가격 데이터를 포함하며, 익명화된 고객 정보와 투자 거래를 포함합니다. 또한, 이 논문은 가격 데이터만을 사용하는 알고리즘, 투자 거래 기반의 알고리즘, 그리고 다양한 정보원을 혼합한 하이브리드 모델 등 세 가지 주요 유형의 FAR 알고리즘을 비교합니다.

- **Performance Highlights**: 논문은 11개의 기본 FAR 알고리즘을 새로운 데이터셋에서 비교 평가하여, 각 접근 방식의 효과를 명확히 비교할 수 있는 벤치마크를 제공합니다. 이를 통해 다양한 알고리즘의 성능을 공정하게 평가할 수 있는 기반을 마련합니다.



### ADMM Based Semi-Structured Pattern Pruning Framework For Transformer (https://arxiv.org/abs/2407.08334)
Comments:
          11 pages, 5 figures

- **What's New**: 이번 논문에서는 ADMM(Alternating Direction Method of Multipliers) 기반 패턴 프루닝(Pattern Pruning) 프레임워크를 소개합니다. 이는 Transformer 모델의 활성화 맵을 재구성하여 특정 패턴으로 모델을 학습시킴으로써, 모델의 가중치 행렬을 더 희소하게 만듭니다. 이 방법을 통해 높은 압축 비율을 유지하면서도 성능을 향상시킬 수 있습니다. 또한, 본 논문은 ADMM과 패턴 프루닝에 대한 이론적 유도 및 양자화(Quantization)로 확장을 제시합니다.

- **Technical Details**: ADMM 기반의 패턴 프루닝 프레임워크를 Transformer 모델에 적용하여, 이를 제약 최적화 문제로 공식화하고 ADMM을 사용해 문제를 최적화합니다. 이를 통해 초기의 밀집된 피처 맵을 지역적으로 희소화된 형태로 변환하고, 효과적으로 압축률을 높이면서 성능을 유지할 수 있습니다. SR-STE(Sparse-refined straight-through estimator)를 도입하여 기울기 소실 문제를 해결하고, 양자화 문제 역시 ADMM을 통해 해결합니다. 이 과정에서 확률적 경사 하강법(Stochastic Gradient Descent)과 폐회로 솔루션을 사용해 최적화를 수행합니다.

- **Performance Highlights**: GLUE 데이터셋의 분류 작업에서 50% 압축 비율을 달성하면서도, COLA 데이터셋에서 55.4% Matthews 상관계수, RTE 데이터셋에서 68.8% 정확도 및 전체 점수 80.1을 유지했습니다. 이는 다른 GLUE 데이터셋에서도 좋은 성능을 보여줍니다.



### Beyond Benchmarks: Evaluating Embedding Model Similarity for Retrieval Augmented Generation Systems (https://arxiv.org/abs/2407.08275)
- **What's New**: 이 논문은 Retrieval-Augmented Generation (RAG) 시스템 설계에서 중요한 단계인 임베딩 모델 선택 과정을 다룹니다. 다양한 옵션이 존재하는 가운데, 유사한 모델들의 클러스터를 식별하는 것이 모델 선택 과정을 최적화하는 방법으로 제안됩니다. 벤치마크 성능 점수에만 의존하는 것은 유사성을 약하게 평가하기 때문에, 논문에서는 Centered Kernel Alignment (CKA)와 Jaccard 및 순위 유사도를 사용하여 모델 임베딩의 유사성을 평가했습니다.

- **Technical Details**: 모델 임베딩 유사성 평가를 위해 두 가지 접근 방식을 사용했습니다. 첫째, 텍스트 청크의 임베딩을 직접 비교하고, 둘째, 주어진 쿼리에 대한 검색 결과의 유사성을 평가했습니다. 대표적 방법으로 Centered Kernel Alignment (CKA)를 사용하였으며, 모델 임베딩 유사성 매트릭스를 계산하여 Hilbert-Schmidt Independence Criterion (HSIC)로 비교했습니다. 또한, Jaccard와 순위 유사도를 사용해 검색 결과의 유사성도 평가했습니다.

- **Performance Highlights**: 다양한 임베딩 모델 패밀리를 여러 벤치마크 데이터셋에서 비교한 결과, 모델 패밀리에 따른 클러스터뿐만 아니라 패밀리 간의 클러스터도 식별할 수 있었습니다. 특히 Mistral 모델이 OpenAI 모델과 가장 높은 유사성을 보여, 상용 모델의 오픈소스 대안으로 제안되었습니다.



### CADC: Encoding User-Item Interactions for Compressing Recommendation Model Training Data (https://arxiv.org/abs/2407.08108)
- **What's New**: 이 연구는 전자상거래 산업에서 핵심 역할을 하는 딥러닝 추천 모델(DLRM)의 훈련 데이터셋 압축을 제안합니다. 특히, 사용자-아이템 상호작용 데이터를 압축하여 모델의 정확도를 유지하면서도 훈련 데이터셋 크기를 줄이는 'Collaborative Aware Data Compression (CADC)' 방법론을 소개합니다.

- **Technical Details**: CADC는 사용자-아이템 상호작용 행렬의 행렬 분해(Matrix Factorization, MF)를 사용해 사용자와 아이템 임베딩을 생성합니다. 이 임베딩은 상호작용 이력을 포함하며, 이렇게 생성된 임베딩을 사용해 훈련 데이터셋의 크기를 크게 줄일 수 있습니다. 이후, 임의 추출법으로 훈련 데이터셋을 선택해 DLRM을 훈련시키지만, 중요한 협업 정보를 유지하여 모델의 성능 저하를 최소화합니다.

- **Performance Highlights**: CADC는 Movielens 1M, Movielens 10M, Epinions 데이터셋에서 테스트되었으며, 많은 사용자-아이템 상호작용 정보를 제거한 후에도 높은 정확도를 유지함을 확인했습니다. 이 방법론은 훈련 데이터셋의 크기를 줄이면서도 DLRM의 성능을 효과적으로 보존합니다.



### Search, Examine and Early-Termination: Fake News Detection with Annotation-Free Evidences (https://arxiv.org/abs/2407.07931)
Comments:
          ECAI 2024 paper. Fudan University & NVIDIA. To appear

- **What's New**: 최신 연구에서는 가짜 뉴스 탐지를 강화하기 위해 증거(evidence)의 중요성을 인식하며, 기존 방법의 한계를 극복한 새로운 접근법 SEE(Search-Examine-Early Termination)을 제안합니다. 이 방식은 뉴스 기사 검색, 증거 검토, 조기 종료 메커니즘을 통해 주어진 뉴스가 실제인지 아닌지를 예측합니다.

- **Technical Details**: SEE는 크게 세 단계로 구성됩니다. 첫째, 뉴스 기사 검색(Search) 단계에서는 뉴스 기사를 쿼리로 사용해 관련 온라인 자료를 찾고, 해당 자료의 제목을 증거로 사용합니다. 둘째, 검토(Examine) 단계에서는 Transformer 기반의 디코더를 사용해 뉴스와 증거를 시퀀셜하게 융합합니다. 셋째, 조기 종료(Early Termination) 단계에서는 각 단계별로 예측 신뢰도를 평가해 추가적인 증거 검토를 중단하고 최종 예측을 제공합니다.

- **Performance Highlights**: 무처리 데이터셋(Weibo21, GossipCop)과 전처리된 데이터셋(Snopes, PolitiFact)에서 SEE 방법을 실험한 결과, 최첨단(FND) 접근법보다 높은 정확도를 보였습니다. 또한, 증거의 순서 변경, 제거 등 다양한 테스트 시나리오에서도 일관된 성능을 유지했습니다.



### Enhancing Social Media Personalization: Dynamic User Profile Embeddings and Multimodal Contextual Analysis Using Transformer Models (https://arxiv.org/abs/2407.07925)
Comments:
          21 pages, 13 figures. Mentor: Prof Pritam Ranjan

- **What's New**: 이 연구는 소셜 네트워크에서 개인화된 상황 인식 경험을 높이기 위해 동적 사용자 프로필 임베딩(dynamic user profile embedding)의 영향을 조사했습니다. 특히 다국어 및 영어 트랜스포머 모델(transformer models)의 성능을 비교 분석했으며, 2천만 개 이상의 데이터 포인트를 사용한 연구입니다.

- **Technical Details**: 이번 연구에서는 다양한 메트릭(metric)과 성능 지표를 사용하여 동적 프로필 임베딩(dynamic profile embeddings)과 정적 프로필 임베딩(non-embeddings)을 비교했습니다. 또한 열화 함수(degradation functions)를 사용한 비교 연구도 수행되었습니다. 이를 통해 동적 임베딩이 사용자의 변하는 취향과 선호도를 성공적으로 추적하여 더 정확한 추천과 높은 사용자 참여도를 제공한다는 것이 검증되었습니다.

- **Performance Highlights**: 연구 결과에 따르면, 동적 임베딩(dynamic embedding)은 사용자 경험을 개선하려는 소셜 미디어 플랫폼에 매우 중요한 결과를 제시하며, 더욱 관련성 높은 기능과 정교한 추천 엔진을 통해 사용자 참여도를 높일 수 있습니다.



### New Method for Keyword Extraction for Patent Claims (https://arxiv.org/abs/2407.07923)
Comments:
          Master's thesis

- **What's New**: 특허 출원 처리에서 중요한 선행 기술 검색을 위한 새로운 방법을 제안하고 시연했습니다. 이 방법은 주파수 분석 메소드에 의해 키워드를 추출하는 대신, 특허 청구항 내에서 제공된 정보의 방식을 활용합니다.

- **Technical Details**: 기존 방법들은 키워드를 통해 검색 엔진에 정보를 제공하는 반면, 제안된 방법은 특허 청구항(patent claims) 내에서 정보 제공의 방식을 기반으로 합니다. 이를 통해 보다 정확하고 관련성 높은 문서를 검색할 수 있습니다.

- **Performance Highlights**: 시연 결과, 제안된 방법이 기존의 주파수 분석 메소드에 비해 더 효과적으로 관련 문서를 검색할 수 있음을 확인했습니다. 이는 특허 출원 처리의 효율성을 크게 향상시킬 수 있습니다.



### Health Misinformation Detection in Web Content via Web2Vec: A Structural-, Content-based, and Context-aware Approach based on Web2Vec (https://arxiv.org/abs/2407.07914)
- **What's New**: 이 논문은 헬스케어 관련 정보의 온라인 전파에서 발생하는 허위 정보 문제를 해결하기 위한 심층 학습(deep learning) 접근 방법을 제안합니다. 특히, 최근 피싱 웹 페이지 탐지에 사용된 Web2Vec 임베딩 표현을 활용하여 헬스케어 웹 페이지의 신뢰성을 평가하는 새로운 방법론을 연구합니다.

- **Technical Details**: 제안된 방법론은 웹 페이지에서 얻은 구조적, 콘텐츠 기반, 및 컨텍스트 기반의 특징(features)을 심층 학습 모델과 결합하여 사용합니다. Web2Vec 임베딩 표현을 사용하여 신뢰성 평가를 위해 필요한 특징을 추출하며, 부적절한 건강 정보 확산을 예방하는 것을 목표로 합니다.

- **Performance Highlights**: 공개된 헬스케어 관련 데이터셋을 사용하여 제안된 방법론을 평가한 결과, 일반적 및 헬스케어 분야의 정보 신뢰성 평가에 있어서 기존의 기계 학습(Machine Learning) 방법론들보다 뛰어난 성능을 보였습니다.



### CaseGPT: a case reasoning framework based on language models and retrieval-augmented generation (https://arxiv.org/abs/2407.07913)
Comments:
          Submitted to ICCBR

- **What's New**: CaseGPT는 대형 언어 모델(LLMs)과 검색증강 생성(RAG) 기술을 결합하여 의료 및 법률 분야의 사례기반 추론을 향상시키는 혁신적인 접근법을 제시합니다. 이 시스템은 부정확한 설명에 기반한 퍼지 검색(fuzzy searches)을 가능하게 하여 데이터 검색성과 사용성을 개선합니다. CaseGPT는 관련 사례 데이터를 검색할 뿐만 아니라 기존 사례 데이터에서 패턴을 추론하여 통찰력 있는 제안과 권장사항을 생성합니다.

- **Technical Details**: CaseGPT의 아키텍처는 세 가지 주요 모듈로 구성됩니다: Query Processing Module, Case Retrieval Engine, Insight Generation Module. Query Processing Module은 사용자 입력을 분석하고, Case Retrieval Engine은 RAG 기술을 사용하여 관련 사례를 검색하고, Insight Generation Module은 검색된 사례를 분석하여 통찰력 있는 정보를 제공합니다. 이는 고차원의 벡터 표현을 통해 의미를 포착합니다.

- **Performance Highlights**: CaseGPT는 전통적인 키워드 기반 시스템과 단순 LLM 기반 시스템보다 정밀도, 재현율, 효율성 면에서 현저히 뛰어나다는 실험 결과를 보여주었습니다. 특히 의학 진단, 법적 선례 연구, 사례 전략 수립과 같은 작업에 매우 유용합니다.



### ITEM: Improving Training and Evaluation of Message-Passing based GNNs for top-k recommendation (https://arxiv.org/abs/2407.07912)
- **What's New**: 이번 연구에서는 Graph Neural Networks (GNNs)을 활용한 top-k 추천 과제에서 랭킹 기반 손실 함수(ranking loss functions)를 적용하여 평가 지표를 직접 최적화하는 방법을 탐구합니다. 기존의 GNN 모델들은 BPR 손실과 같은 프록시 손실(proxy losses)로 훈련되는 경우가 많았는데, 이번 연구에서는 순수한 랭킹 지표를 최적화하는 손실 함수와 PPR 기반 부정 샘플링 전략을 제안합니다.

- **Technical Details**: 연구팀은 부드러운 랭킹 근사치(smooth approximations of the rank)를 이용하여 GNNs의 end-to-end 훈련을 가능하게 했습니다. 또한, 사용자 분할 유도 프로토콜을 포함하여 실제 응용 프로그램에서의 GNN 모델 성능을 반영할 수 있도록 평가 방식을 확장했습니다. ITEM(Improving Training and Evaluation of Message-passing-based GNNs) 프레임워크를 도입하여 평가 지표의 좋은 근사치를 제공합니다.

- **Performance Highlights**: 네 가지 데이터셋과 네 가지 최신 GNN 아키텍처를 대상으로 한 실험 결과, 제안된 ITEM 방식은 기존의 BPR 손실을 포함한 고급 손실 함수들을 전반적으로 능가하며 더 빠른 훈련 속도를 보였습니다. 또한, 새로운 사용자에 대한 테스트를 포함한 유도 프로토콜 평가에서 GNN의 일반화 성능을 개선했습니다.



### From Real to Cloned Singer Identification (https://arxiv.org/abs/2407.08647)
Comments:
          To be published at ISMIR 2024

- **What's New**: 최근 몇 년 동안 유명 가수의 클론된 목소리가 점점 더 현실적으로 들리고 인기를 끌고 있습니다. 그러나 이러한 기술은 퍼스널리티 권리를 침해할 수 있어 원래 가수를 식별할 수 있는 방법이 필요합니다. 이 논문에서는 가수 식별 기법을 활용해 이러한 문제를 해결할 수 있는 가능성을 조사합니다. 이를 위해 가수 수준의 대비 학습(deep learning) 방식으로 훈련된 세 가지 임베딩 모델을 소개합니다.

- **Technical Details**: 세 가지 모델은 서로 다른 유형의 음성 샘플(혼합 음성, 보컬, 또는 두 가지 모두)을 사용합니다. 대규모 폐쇄 데이터셋과 두 개의 공개 데이터셋(FMA, MTG)을 통해 모델을 훈련하고 평가했습니다. 각 모델은 동일 가수의 음성 조각을 양의 쌍으로, 다른 가수의 음성을 음의 쌍으로 사용하여 대비 학습을 합니다. 이 모델들은 실제 가수를 식별하는 데 매우 뛰어났지만, 클론된 가수를 식별하는 데는 성능이 저하됐습니다.

- **Performance Highlights**: 모든 모델이 실제 가수 식별에서 높은 성능을 보였지만, 클론된 목소리를 평가할 때 성능이 저하되었습니다. 특히 혼합 음성을 입력으로 사용하는 모델에서 더 많은 문제가 발생했습니다. 이러한 결과는 가수 식별 시스템에서의 편향을 이해하고 음악에서의 목소리 딥페이크(voice deepfake)를 식별하는 방법을 개선할 필요성을 강조합니다.



### Multi-Group Proportional Representation (https://arxiv.org/abs/2407.08571)
Comments:
          35 pages, 24 figures. Under review

- **What's New**: 최근 이미지 검색 및 검색 작업에서의 대표성 문제를 해결하기 위해 'Multi-Group Proportional Representation (MPR)' 메트릭을 도입했습니다. MPR은 교차 그룹 (intersectional groups)의 대표성을 측정하는 새로운 방법으로, 기존의 단일 속성 그룹 대비 더 복잡한 다중 속성 그룹을 다룰 수 있습니다. 이 메트릭을 통해 검색 결과가 현실의 다양한 비율과 일치하도록 보장할 수 있습니다.

- **Technical Details**: MPR 메트릭은 회귀 (regression) 오라클을 사용하여 계산되며, 특정 함수 클래스에 대해 닫힌 형식으로 계산할 수 있습니다. 또한, 'MAPR'이라는 알고리즘을 개발하여 벡터 데이터베이스에서 항목을 검색할 때 쿼리 임베딩(query embedding)의 평균 유사성을 극대화하면서 MPR 조건을 만족하도록 보장합니다. 이 알고리즘은 반복적으로 오라클을 호출하여 MPR 위반을 계산하고, 관련 있고 대표성이 있는 항목을 검색합니다.

- **Performance Highlights**: MAPR는 CelebA, Occupations dataset, Fairface, UTKFace와 같은 다양한 데이터셋을 사용한 검색 작업에서 우수한 성능을 보여줍니다. 실험 결과, MAPR는 검색 유사성과 MPR 간의 균형을 맞추는 데 경쟁 접근 방식보다 탁월한 성과를 보이며, 다중 교차 그룹의 대표성을 더욱 효과적으로 반영합니다.



### DALL-M: Context-Aware Clinical Data Augmentation with LLMs (https://arxiv.org/abs/2407.08227)
Comments:
          we introduce a pioneering approach to clinical data augmentation that employs large language models (LLMs) to generate patient contextual synthetic data. It preserves the integrity of real patient data while enriching the dataset with contextually relevant synthetic features, significantly enhancing model performance

- **What's New**: 본 논문은 임상 데이터를 중심으로 한 새로운 데이터 증강 기술을 소개합니다. DALL-M이라는 방법론을 통해 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 환자의 맥락을 반영한 합성 데이터를 생성함으로써 의료 AI 진단의 신뢰성과 적용성을 향상시킵니다.

- **Technical Details**: DALL-M은 세 가지 단계의 특징 생성 과정을 통해 데이터를 증강합니다: (i) 임상 컨텍스트 저장 (clinical context storage), (ii) 전문가 쿼리 생성 (expert query generation), (iii) 컨텍스트-인식 특징 증강 (context-aware feature augmentation). MIMIC-IV 데이터셋에서 799명의 케이스를 대상으로 91개의 추가 특징을 생성하였습니다. 이는 환자의 X-ray 보고서, 성별, 나이를 기반으로 새로운 임상적 가치를 생성하는 첫 번째 작업입니다.

- **Performance Highlights**: 결과적으로 Decision Trees, Random Forests, XGBoost, TabNET 등의 머신 러닝 모델에서 증강된 특징을 포함하여 F1 점수가 16.5% 증가했고, Precision과 Recall은 약 25% 향상되었습니다. 이는 임상 데이터 증강 문제를 해결하고, 더 신뢰할 수 있는 진단 도구를 개발하는 데 중요한 역할을 합니다.



### FsPONER: Few-shot Prompt Optimization for Named Entity Recognition in Domain-specific Scenarios (https://arxiv.org/abs/2407.08035)
Comments:
          accepted for publication at the 27th European Conference on Artificial Intelligence (ECAI-2024)

- **What's New**: 새로운 접근법인 FsPONER가 도메인 특화된 Named Entity Recognition (NER) 작업에 몇 샷 학습(few-shot learning) 최적화 방법을 도입했습니다. 이 접근법은 산업 제조 및 유지 보수와 같은 특화된 도메인에서 GPT-4-32K, GPT-3.5-Turbo, LLaMA 2-chat, Vicuna와 같은 여러 대형 언어모델(LLM)을 평가했습니다.

- **Technical Details**: FsPONER는 무작위 샘플링, TF-IDF 벡터 및 두 가지를 결합한 세 가지 몇 샷 선택 방법을 포함합니다. 이는 기존의 일반적인 GPT-NER 방법과 비교하여 몇 샷 예제 수가 증가함에 따라 그 성능을 비교했습니다. 해당 모델들은 fine-tuned BERT와 LLaMA 2-chat과도 성능을 비교했습니다.

- **Performance Highlights**: 실제 데이터가 부족한 시나리오에서 FsPONER는 TF-IDF와 함께 사용될 때 약 10% 높은 F1 점수를 기록하며, fine-tuned 모델들을 능가하는 성능을 보였습니다.



### DS@GT eRisk 2024: Sentence Transformers for Social Media Risk Assessmen (https://arxiv.org/abs/2407.08008)
Comments:
          Paper Submitted to CLEF 2024 CEUR-WS

- **What's New**: DS@GT 팀은 eRisk 2024에서 태스크 1과 3에 참여했습니다. 태스크 1에서는 Beck Depression Inventory (BDI-II)를 기반으로 우울증 증상을 예측하는 랭킹 시스템을 제안했으며, 태스크 3에서는 BERT 임베딩을 활용하여 사용자 게시물 기록을 기반으로 섭식 장애 증상을 예측했습니다.

- **Technical Details**: 태스크 1에서는 질문의 관련성을 이진 분류기로 훈련시켜 랭킹 시스템을 만들었지만, 이진 분류기가 랭킹에 적합하지 않음을 확인했습니다. 태스크 3에서는 BERT 임베딩을 사용하여 사용자 게시물의 벡터 표현을 학습해 섭식 장애 증상 예측에 활용했습니다. 문장 변환기(sentence transformers)는 텍스트 데이터의 표현에 매우 유용한 도구임을 확인했습니다.

- **Performance Highlights**: 태스크 1의 내부 검증에서 문장 변환기를 사용한 모델이 평균 F1 점수 89%와 평균 정확도 90%를 기록했습니다. 그러나 실제 테스트 셋에서는 모든 모델이 성능이 저조했습니다. 이것은 분류기를 랭커로 재활용할 수 있다는 가정이 잘못되었음을 시사합니다.



### Automated Neural Patent Landscaping in the Small Data Regim (https://arxiv.org/abs/2407.08001)
Comments:
          11 pages, 4 figures

- **What's New**: 이 논문은 자동화된 뉴럴 특허 구역화 시스템(neural patent landscaping system)을 소개합니다. 이 시스템은 최소한의 라벨된 예시를 사용하여 높은 성능을 발휘하며, 새로운 기술 영역의 특허 지도를 보다 효율적으로 구성할 수 있도록 디자인되었습니다. 특히, 기존 시스템에 비해 난이도가 높은 사례에서의 성능을 크게 향상시켰고, 적은 트레이닝 데이터로도 높은 성능을 달성하였습니다.

- **Technical Details**: 연구팀은 Abood와 Feltenberger의 'seed/anti-seed' 접근 방식을 활성 학습(active learning)과 결합하여 어려운 라벨된 예시를 수집하는 새로운 데이터 생성 절차를 제안했습니다. 또한, 기존 시스템의 성능을 향상시키기 위해 인용 네트워크(citation networks)와 CPC 코드 임베딩(CPC code embeddings) 같은 추가 특징들을 도입하였습니다. 이 새로운 접근 방식을 통해, 극히 적은 수의 라벨만으로도 높은 성능을 달성할 수 있습니다.

- **Performance Highlights**: 제안된 시스템의 성능은 '난이도가 높은' 예시에서 0.69의 F1 점수를 기록했으며, 기존 시스템의 0.6보다 향상된 성능을 보였습니다. 더 나아가, 단 24개의 예시로 전체적으로 0.75의 F1 점수를 달성함으로써, 수천 개의 예시를 필요로 했던 기존 연구들보다 훨씬 적은 데이터로도 유사한 성능을 보였습니다.



### Enhancing HNSW Index for Real-Time Updates: Addressing Unreachable Points and Performance Degradation (https://arxiv.org/abs/2407.07871)
- **What's New**: HNSW(Hierarchical Navigable Small World) 인덱스의 주요 단점인 'unreachable points phenomenon'와 실시간 삭제 및 업데이트 작업에 대한 성능 저하를 해결하기 위해 새로운 MN-RU 알고리즘이 제안되었습니다. 이 알고리즘은 기존 HNSW를 기반으로 하여, 특히 삭제 및 업데이트 작업 후 특정 데이터 포인트가 접근 불가능해지는 문제를 효과적으로 완화하며 검색 효율성을 향상시킵니다.

- **Technical Details**: ANNS(approximate nearest neighbor search)는 정보 검색, 데이터 마이닝 및 추천 시스템 등 다양한 분야에서 중요한 역할을 합니다. 특히, 그래프 기반 인덱스가 다른 방식에 비해 우수한 검색 성능을 보이며, HNSW가 그 대표적인 예입니다. 하지만 HNSW는 많은 실시간 삭제 및 삽입 작업을 수행할 때 성능 저하와 'unreachable points phenomenon'을 경험하게 됩니다. 이를 해결하기 위해 제안된 MN-RU 알고리즘은 이러한 문제를 어떻게 해결하며 검색 결과의 정확성을 유지하는지 자세히 논의합니다.

- **Performance Highlights**: 제안된 MN-RU 알고리즘은 기존 HNSW와 비교하여 업데이트 효율성을 크게 향상시키고, unreachable points의 성장률을 효과적으로 억제합니다. 실험 결과, MN-RU 알고리즘이 기존 방법들보다 우수한 성능을 보이며, 실용적인 산업 응용 분야에서도 쉽게 통합될 수 있습니다.



### Systematic Evaluation of Neural Retrieval Models on the Touch\'e 2020 Argument Retrieval Subset of BEIR (https://arxiv.org/abs/2407.07790)
Comments:
          SIGIR 2024 (Resource & Reproducibility Track)

- **What's New**: 이 논문은 뉴럴 리트리벌 모델 (neural retrieval models)의 제로샷 효율성 (zero-shot effectiveness)을 BEIR 벤치마크를 통해 평가한 결과, 특히 Touché 2020 데이터셋에서 뉴럴 리트리벌 모델이 BM25보다 성능이 낮다는 점을 다뤘습니다. 이를 더 깊게 분석하기 위해, 두 가지 실험을 통해 Touché 2020 데이터를 재현 가능성 연구를 수행했습니다. 첫 번째는 블랙박스 평가 (black-box evaluation)와 이론적 탐구, 두 번째는 데이터 잡음을 제거하는 평가입니다.

- **Technical Details**: 블랙박스 평가에서는 뉴럴 모델들이 Touché 2020 데이터의 짧은 패시지 (short passages)에 편향되어 있다는 점을 발견했습니다. 그리고 많은 뉴럴 모델의 결과가 평가되지 않았다는 점도 발견했습니다. 데이터 잡음 제거에서는 20단어 이하의 매우 짧은 패시지를 제외하고, 평가되지 않은 데이터를 Touché 가이드라인에 따라 후속 판단으로 보강했습니다. 이로 인해 nDCG@10에서 최대 0.52까지 성능이 향상되었지만, 여전히 BM25가 더 효과적이었습니다.

- **Performance Highlights**: 데이터 잡음 제거 후, 모든 뉴럴 모델의 효율성이 크게 향상되었지만 BM25가 여전히 가장 효과적이었습니다. 뉴럴 모델들은 문서 길이 정규화 (document length normalization) 준칙을 위반하는데 반해 BM25는 이를 잘 지키고 있다는 이론적 분석 결과도 제시되었습니다. 연구의 코드와 보강된 Touché 2020 데이터셋은 GitHub에서 제공됩니다.



### Evaluating the method reproducibility of deep learning models in the biodiversity domain (https://arxiv.org/abs/2407.07550)
- **What's New**: 이 연구는 생물다양성 분야에서 딥러닝 (Deep Learning, DL) 기법을 사용하는 연구의 재현성 (reproducibility)을 조사했습니다. 연구팀은 생물다양성 전문가들이 제공한 키워드로 식별된 61개의 논문에서 데이터를 추출하여 재현성 여부를 평가했습니다. 그 결과, 데이터셋을 공유한 논문이 47%에 달했으나, 딥러닝 방법론에 대한 종합적인 정보, 특히 랜덤성에 관한 세부사항이 부족한 경우가 많았다는 것을 발견했습니다.

- **Technical Details**: 논문에서는 DL 기법을 사용하는 생물다양성 연구의 재현성 평가를 위한 방법론을 설계했습니다. 연구팀은 재현성 평가를 위해 자원 요구사항, 방법론적 정보, 통제되지 않은 랜덤성 요소, 통계적 고려사항을 포함한 네 가지 카테고리로 구분된 열 가지 변수(variable)를 정의했습니다. 이러한 카테고리를 바탕으로 다른 수준의 재현성을 규정했으며, 초기 단계에서 데이터 수집 및 전처리, 모델 설계 등 딥러닝 파이프라인(deep learning pipeline)의 세부적인 문서화의 중요성을 강조했습니다.

- **Performance Highlights**: 연구 결과에 따르면, 딥러닝 기법을 사용하는 생물다양성 연구의 재현성은 대체로 낮은 편입니다. 그러나 점차적으로 데이터셋과 코드 공개를 통해 재현성을 향상시키려는 추세가 나타나고 있습니다. 이는 생물다양성 연구에서 딥러닝 기법의 신뢰성과 영향을 높이는 데 기여할 것으로 기대됩니다.



### Multi-objective Learning to Rank by Model Distillation (https://arxiv.org/abs/2407.07181)
- **What's New**: 온라인 마켓플레이스에서의 검색 랭킹 시스템을 개선하기 위해 새롭게 제안된 방법론입니다. 이 논문에서는 Airbnb의 여러 랭킹 모델을 최적화하기 위해 증류 기반(distillation-based) 랭킹 솔루션을 제안합니다. 이 방법은 기존 방법보다 주요 목표와 부수적인 목표 모두를 크게 향상시키며, 모델 안정성을 높입니다. 또한, 자가 증류(self-distillation) 방법을 통해 시스템을 더 간단하게 만들 수 있음을 보여줍니다.

- **Technical Details**: 이 논문에서는 다중 목적 최적화(multi-objective optimization)를 위해 증류 기반의 접근 방식을 사용합니다. 전통적인 각 목적에 대해 독립적으로 모델을 훈련시키고 이를 가중 합산하는 방법(model fusion)은 상호 간섭으로 인해 서브 옵티멀한 결과를 초래할 수 있습니다. 이를 해결하기 위해 본 논문은 모델 증류(model distillation)와 다중 목적 학습을 결합하여 불균형한 데이터 문제를 해결하고, 온라인 점수 집계 가중치 튜닝을 제거하여 최적의 결과를 도출합니다.

- **Performance Highlights**: 새롭게 제안된 증류 기반 랭킹 시스템은 전통적인 방법들에 비해 주요 목표를 크게 향상시켰을 뿐만 아니라, 부수적인 목표 제약 조건도 충족시키고 모델 안정성도 높였습니다. 추가적으로, 비차별적인(ad-hoc, non-differentiable) 사업 목표도 효율적으로 주입하여 최적화 목표를 균형있게 맞출 수 있음을 시뮬레이션으로 증명하였습니다.



### Metasurface-based Snapshot Shortwave-Infrared Hyperspectral Image Reconstruction with Inter and Intra Prior Learning Network (https://arxiv.org/abs/2407.07503)
Comments:
          10 pages,5 figures

- **What's New**: 이번 연구에서는 스냅샷 SWIR 하이퍼스펙트럴 이미징 시스템(snapshost SWIR hyperspectral imaging system)을 메타서피스 필터(metasurface filter)와 이에 대응하는 필터 선택 방법을 기반으로 제안했습니다. 이 시스템은 소형화되고 스냅샷 이미징을 가능하게 한다는 장점을 가지고 있습니다. 또한, 새로운 'inter and intra prior learning unfolding framework'를 제안하여 SWIR 하이퍼스펙트럴 이미지(SWIR hyperspectral image) 재구축의 품질을 향상시켰습니다.

- **Technical Details**: 기존의 SWIR 하이퍼스펙트럴 이미징 시스템이 대형 장비와 저속의 데이터 수집 문제를 겪고 있는 반면, 이번 연구에서는 메타서피스 필터와 깊은 펼침 네트워크(deep unfolding network)를 활용하여 이러한 문제를 해결하고자 했습니다. 이를 위해, 실리콘 나노필러(silicon nanopillars)로 구성된 메타서피스 유닛을 설계하고, 최소 상관 계수를 보장하는 최적의 필터 패턴을 선정하는 방법을 제안했습니다. 또한, 필터 선택 과정의 장점과 함께 새로운 'inter & intra prior learning network'를 설계하여 고성능의 SWIR 하이퍼스펙트럴 이미징을 가능하게 했습니다.

- **Performance Highlights**: 실험 결과, 제안된 시스템은 기존 방법들보다 더 빠른 속도로 HSI를 재구축할 수 있으며, 고성능의 품질을 보였습니다. 필터 선택 및 재구축 알고리즘의 우수성을 입증하는 삭제 연구(ablation studies)에서도 최적의 결과를 얻었습니다.



### Deep Pareto Reinforcement Learning for Multi-Objective Recommender Systems (https://arxiv.org/abs/2407.03580)
- **What's New**: 추천 플랫폼에서 다중 목표를 동시에 최적화하는 것이 중요한 과제입니다. 이를 해결하기 위해 저자들은 Deep Pareto Reinforcement Learning (DeepPRL) 접근법을 제안하였습니다. DeepPRL은 다중 목표 간의 복잡한 관계를 포괄적으로 모델링하고, 개인화된 소비자 선호도와 상황에 맞춘 선호도를 효과적으로 포착하며, 단기 및 장기 성과를 최적화합니다.

- **Technical Details**: DeepPRL 접근법은 파레토 프론티어(Pareto Frontier)를 형성하는 방식으로, 한 목표의 개선이 다른 목표의 성능 저하를 초래하는 경우를 다룹니다. 기존 다중 목표 추천 시스템이 정적이고 균일한 방식으로 이러한 목표 사이의 균형을 이루려는 것과는 달리, DeepPRL은 동적인 관계를 체계적으로 고려합니다. 이를 통해 목표 간 관계를 정확하게 모델링하고, 소비자 선호도에 맞춘 추천을 제공합니다.

- **Performance Highlights**: DeepPRL은 오프라인 실험에서 최첨단 베이스라인에 비해 상당한 파레토 우위를 달성했습니다. 또한, Alibaba의 비디오 스트리밍 플랫폼에서 진행된 통제 실험에서도 3가지 상충하는 비즈니스 목표를 동시에 개선하여 실질적인 경제적 영향을 증명했습니다.




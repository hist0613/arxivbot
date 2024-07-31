New uploads on arXiv(cs.CL)

### Latent Causal Probing: A Formal Perspective on Probing with Causal Models of Data (https://arxiv.org/abs/2407.13765)
Comments:
          COLM 2024

- **What's New**: 이번 논문에서는 Language Models (LMs)의 내부 작동 방식을 이해하기 위해 사용하는 probing 기법을 구조적 인과 모델 (SCM)로 분석하는 새로운 접근법을 제안합니다. 이론적으로는 LMs가 텍스트 생성에 사용된 잠재적 인과 변수를 얼마나 잘 학습했는지를 검사할 수 있도록 하는 프레임워크를 개발했습니다. 또한, 근시성 프로빙 실험을 통해 LMs가 실제로 잠재적 인과 개념을 학습할 수 있는지에 대한 실험적 증거를 제공합니다.

- **Technical Details**: 구조적 인과 모델 (Structural Causal Models, SCM)은 데이터 생성 과정에서 인과 관계를 나타내는 그래픽 모델입니다. 이 논문에서는 SCM을 사용하여 LMs가 학습 중에 인과 변수를 나타낼 수 있는지 여부를 분석합니다. 이를 위해 텍스트 생성 과정과 LM 및 프로브의 학습을 포괄하는 SCM을 확장하였으며, 인과 매개 분석을 통해 프로브의 영향을 배제하고 LM에 의한 인과 경로만을 분리하는 기술을 제안합니다.

- **Performance Highlights**: 실험적으로는 Jin & Rinard (2023)의 연구를 확장하여 그리드 월드 네비게이션 과제에서 LMs가 잠재적 인과 개념을 학습할 수 있는지를 검증했습니다. 결과적으로 (1) LM이 잠재적 변수를 실제로 나타내는 것을 학습했으며, (2) LM의 표현이 새로운 행동 시퀀스에 일반화할 수 있는 귀납적 편향을 가지고 있음을 발견했습니다. 이는 LMs가 언어를 이해하는 방식에 대한 중요한 통찰을 제공합니다.



### Black-Box Opinion Manipulation Attacks to Retrieval-Augmented Generation of Large Language Models (https://arxiv.org/abs/2407.13757)
Comments:
          10 pages, 3 figures, under review

- **What's New**: 이 논문에서는 블랙박스 공격 환경에서 RAG 모델의 의견 조작 취약성을 탐구합니다. 이러한 공격은 사용자 인지와 의사결정에 미칠 잠재적 영향에 대해 새로운 통찰을 제공합니다.

- **Technical Details**: RAG 모델에서 리트리발(회수) 모듈의 순위 결과를 조작하고, 이를 통해 대리 모델(surrogate model)을 훈련합니다. 이 대리 모델을 이용한 적대적 회수 공격(adversarial retrieval attack)을 통해 실제 RAG 모델에 블랙박스 전이 공격을 수행합니다.

- **Performance Highlights**: 여러 주제에 걸친 의견 데이터셋에서 실험한 결과, 제안된 공격 전략이 RAG가 생성하는 콘텐츠의 의견 극성을 상당히 바꿀 수 있음을 보여주었습니다. 이는 모델의 취약성을 입증하고, 사용자에게 잘못된 정보나 편향된 정보를 수용하게 할 가능성을 나타냅니다.



### LLMs as Function Approximators: Terminology, Taxonomy, and Questions for Evaluation (https://arxiv.org/abs/2407.13744)
- **What’s New**: 최근 자연어 처리(NLP) 모델은 특정 작업에 특화된 모델로부터 범용 사전 훈련 모델을 미세 조정하는 방식으로 빠르게 발전해왔습니다. 이제는 본질적으로 범용 모델처럼 보이는 모델들이 등장하고 있습니다. 이 논문은 이러한 모델들이 무엇을 모델링하는지 명확하지 않아 평가 시에 도움이 되지 않는 '인공지능'과 같은 비유가 등장하게 된다고 주장합니다. 대신, 이 모델들을 전문적 기능을 근사할 수 있는 능력으로 바라보는 접근을 제안합니다.

- **Technical Details**: 기술적으로, 대형 언어 모델(LLM)은 토큰 시퀀스를 입력받아 토큰 어휘에 대한 분포를 출력하는 함수로 설명됩니다. 최근에 개발된 다양한 기술들(예: 질문/답변 쌍으로 작업 프레이밍, 명령 조정, 응답 선호도 조정 등)과 데이터 및 모델 크기의 확장이 이러한 모델들을 자극과 반응의 관계로 이해할 수 있게 만들었습니다. 이는 단순한 텍스트 연속성과는 다른 성질입니다. 예를 들어, 'CPU에 대한 라임릭 작성'이나 'LLMs에 대한 라임릭 작성'이라는 입력에 대해 적절한 라임릭이 생성되는 것입니다.

- **Performance Highlights**: 이 논문은 대형 언어 모델을 함수 근사기(function approximator)로서 바라보아, 이를 통해 발견성, 안정성, 보호 가능성 등의 질문이 어떻게 떠오를 수 있는지를 탐구합니다. 또한, 이 접근 방식이 실용적이고 이론적인 평가 측면에서 다양한 요소들을 하나의 개념적 틀 안에 모을 수 있음을 보여줍니다. 이를 통해 LLM의 다양한 기능들을 더 잘 이해하고 평가할 수 있는 방안이 제시됩니다.



### Baba Is AI: Break the Rules to Beat the Benchmark (https://arxiv.org/abs/2407.13729)
Comments:
          8 pages, 8 figures

- **What's New**: 이번 연구는 문제 해결을 위한 기존 규칙과 절차를 따르는 인간의 능력뿐만 아니라 창의력을 활용하여 이러한 규칙과 목표를 재정의하는 능력을 살펴봅니다. 이를 위해 새롭게 개발된 벤치마크는 퍼즐 게임 'Baba Is You'를 기반으로 하고 있습니다. 이 게임에서는 에이전트가 환경 내의 객체와 규칙을 조작하여 특정 목표에 도달합니다.

- **Technical Details**: 이 연구는 세 가지 최첨단 멀티모달 대형 언어 모델(LLMs)인 OpenAI GPT-4o, Google Gemini-1.5-Pro, Gemini-1.5-Flash을 테스트하여 게임의 규칙을 조작하고 결합해야 하는 상황에서 이들이 얼마나 잘 일반화하는지를 조사했습니다. 연구는 Textual rule blocks와 시각적 인풋을 활용하였으며, 에이전트가 규칙을 깨거나 활성화하는 등의 고수준 텍스트 계획을 생성하도록 했습니다.

- **Performance Highlights**: 결과는 GPT-4o가 주어진 몇몇 환경에서 완벽한 정확도를 보였음을 나타내며, Gemini-1.5-Flash는 Gemini-1.5-Pro를 앞섰습니다. 단, 시각적 인풋에 대한 높은 분산 로드가 있는 경우에는 세 모델 모두 정확도가 떨어졌습니다.



### Understanding Reference Policies in Direct Preference Optimization (https://arxiv.org/abs/2407.13709)
- **What's New**: 이 논문은 Direct Preference Optimization (DPO)의 미진한 부분인 참조 모델(혹은 정책)에 대한 의존성에 대해 탐구합니다. DPO는 대형 언어 모델(LLM)의 지시 미세 튜닝을 위한 훈련 방법입니다. 이 논문에서는 DPO의 효과에 한계를 줄 수 있는 참조 정책의 강도를 최적화하는 것, DPO와 관련 학습 목표를 이론적 및 실험적으로 비교하는 것, 그리고 더 강력한 참조 정책이 DPO의 성능에 미치는 영향을 조사합니다.

- **Technical Details**: 첫 번째 연구 질문(RQ1)은 DPO의 KL-divergence 제약의 최적 강도 탐구입니다. KL-divergence는 참조 정책에서 벗어나지 않도록 페널티를 부여하는 메커니즘입니다. 두 번째 질문은 지시 미세 튜닝에서 참조 정책이 필수적인지에 관한 것입니다. 마지막으로, 강력한 참조 정책이 DPO의 성능을 향상시킬 수 있는지 조사합니다. 이 연구는 알파카 애벌 평가(AlpacaEval) 벤치마크에서 Tulu 2와 Mistral이라는 두 개의 오픈 소스 사전 훈련된 LLM을 사용하여 모두 수행되었습니다.

- **Performance Highlights**: DPO는 참조 정책의 강도에 민감하게 반응하며, 약한 제약이 더 나은 성능을 보이지만 제약이 너무 약해지면 성능이 저하됩니다. 논문에서는 시퀀스 레벨과 토큰 레벨 분석을 통해 심층 분석을 수행하였습니다. 예를 들어, DPO 튜닝 후 시퀀스 종료 토큰의 예측 확률이 10,000배 이상 감소하는 등의 현상이 관찰되었습니다.



### ANHALTEN: Cross-Lingual Transfer for German Token-Level Reference-Free Hallucination Detection (https://arxiv.org/abs/2407.13702)
Comments:
          ACL 2024 Student Research Workshop

- **What's New**: 최근 연구는 영어에 국한된 토큰 수준의 참조 없는 헛소리 탐지(token-level reference-free hallucination detection)에 주목했습니다. 영어 외 언어에서 강력한 데이터셋이 부족하여 이런 탐지 기법의 다중언어 전이(cross-lingual transfer) 효과를 체계적으로 조사하기 어려웠습니다. 이 문제를 해결하기 위해 ANHALTEN이라는 새로운 평가 데이터셋을 소개합니다. ANHALTEN은 독일어로 확장된 첫 번째 헛소리 탐지 데이터셋으로, 독일어와 영어 간의 다중언어 모델과 전이 접근 방식을 직접 비교할 수 있도록 합니다.

- **Technical Details**: ANHALTEN은 영어 HaDes 데이터셋을 독일어로 완전히 번역하고 헛소리 범주와 라벨을 포함한 평행(parallel, 즉 동일한 텍스트와 라벨을 독일어로 번역) 데이터셋을 제공합니다. 실험에는 자동 번역 및 수동 후편집 작업을 병행하였으며, 토큰 수준에서의 헛소리 탐지를 위해 다양한 다중언어 전이approaches(XLT)를 평가했습니다. 세 가지 주요 XLT 방법을 사용했습니다: (1) Zero-Shot Transfer, (2) Few-Shot Transfer, (3) Translate-Train. 이 방법들은 각각 다른 양의 독일어 라벨된 데이터를 이용하며, 효율적인 전이를 위해 어댑터 기반 접근 방식을 제안했습니다.

- **Performance Highlights**: 대규모 맥락 길이는 독일어에서 헛소리 탐지를 개선하는 데 도움이 됨을 발견했으며, 이는 실시간 텍스트 생성 동안 헛소리를 예방하는 데 효과적입니다. 또한, 소량의 라벨된 데이터로 전이하는 Few-Shot Transfer 방법이 실제 응용에서 가장 높은 성능을 발휘했습니다.



### Benchmark Agreement Testing Done Right: A Guide for LLM Benchmark Evaluation (https://arxiv.org/abs/2407.13696)
Comments:
          Under Review

- **What's New**: 새로운 논문에서는 언어 모델(Language Models, LMs) 평가를 위한 벤치마크의 유효성을 검증하는 방법론, 즉 벤치마크 합의 테스트(Benchmark Agreement Testing, BAT)를 체계화하려는 시도를 다룹니다. BAT는 새로운 벤치마크가 기존의 확립된 벤치마크와 얼마나 일치하는지를 측정하는데 사용되며, 현재까지 표준화된 절차가 부족했습니다. 이러한 부족함은 부정확한 결론을 초래하여 벤치마크에 대한 신뢰성을 저하시킬 수 있습니다.

- **Technical Details**: 이 논문은 40개 이상의 대표적인 벤치마크와 이를 평가한 200개 이상의 언어 모델을 분석하여, BAT에서 간과될 수 있는 여러 방법론적인 결정들이 결과에 얼마나 큰 영향을 미치는지를 보여주었습니다. 주요한 요소는 참조 벤치마크(reference benchmark)의 선택, 테스트에 포함된 모델의 선정, 그리고 상관 계수(correlation metrics) 해석 방법 등이 있습니다. 이를 해결하기 위해 논문은 BAT의 모범 사례(best practices)를 제안하고, 이를 바탕으로 한 Python 패키지 'BenchBench'와 'BenchBench-리더보드'를 소개합니다.

- **Performance Highlights**: 이 연구는 BAT의 일관성과 유효성을 향상시키기 위한 구체적인 지침을 제공하며, 이를 통해 벤치마크 평가의 신뢰성을 높이는 것을 목표로 합니다. 또한, BenchBench 패키지는 사용자가 여러 벤치마크 데이터를 손쉽게 비교할 수 있는 프레임워크를 제공하며, 지속적으로 새로운 벤치마크를 추가할 수 있도록 설계되었습니다.



### Prover-Verifier Games improve legibility of LLM outputs (https://arxiv.org/abs/2407.13692)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)의 출력 신뢰성을 높이기 위한 방법으로 '가독성(Legibility)'이라는 개념을 도입하였다. 우리는 초등학교 수학 문제를 해결하는 맥락에서 가독성을 연구했으며, 단순히 정답률을 최적화하는 것이 가독성을 저해할 수 있음을 발견했다. 이를 해결하기 위해 Prover-Verifier 게임에서 영감을 받은 새로운 훈련 알고리즘을 제안했다. 이 알고리즘은 검증기(verifier)와 도움이 되는 증명자(Helpful prover), 그리고 속임수를 쓰는 증명자(Sneaky prover)를 반복적으로 훈련시킨다.

- **Technical Details**: 제안된 알고리즘은 Prover-Verifier 게임을 기반으로 한다. 검증기는 솔루션의 정확성을 예측하는 역할을 하며, 도움이 되는 증명자는 검증기가 수용하는 올바른 솔루션을 생성한다. 반대로 속임수를 쓰는 증명자는 검증기를 속이는 잘못된 솔루션을 생성한다. 검증기와 증명자는 각 훈련 라운드마다 번갈아 가며 훈련되며, 검증기는 이전 라운드의 증명자 솔루션을 기반으로 학습되며, 증명자는 현재 라운드 검증기의 승인율을 기준으로 최적화된다.

- **Performance Highlights**: 훈련 결과, 도움이 되는 증명자의 정확성과 검증기의 강건성이 증가하는 것을 확인했다. 검증기의 거짓 양성율이 낮아지고, 속임수를 쓰는 증명자가 점점 더 미묘한 결함을 생성함에 따라, 유용한 증명자의 테스트 통과율도 상승하였다. 또한, 시간 제약이 있는 사람들이 도움을 받는 증명자의 솔루션을 검증할 때 정확도가 증가했으며, 속임수를 쓰는 증명자의 솔루션을 검증할 때 정확도가 감소했다.



### DART-Math: Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving (https://arxiv.org/abs/2407.13690)
Comments:
          Preprint. Data and model checkpoints are available at this https URL

- **What's New**: 논문에서는 대규모 언어 모델(Large Language Models, LLMs)에게 수학 문제 해결 능력을 향상시키기 위한 새로운 방법인 DART(Difficulty-Aware Rejection Tuning) 방식을 제안합니다. 이는 어려운 질문에 대해 더 많은 샘플링 시도를 할당하여 훈련 데이터를 생성하는 방식을 개선한 것입니다.

- **Technical Details**: 기존 데이터 증강 방법들은 쉬운 질문에만 편향되어 어려운 질문에 대한 응답이 거의 생성되지 않는다는 문제점이 있었습니다. DART는 이러한 문제를 해결하기 위해 Uniform와 Prop2Diff 두 가지 전략을 사용하여 어려운 질문에 대해 더 많은 샘플을 생성합니다. 이 과정에서는 7B 크기의 오픈 소스 모델을 이용했으며, 이를 통해 데이터셋을 생성하고 다양한 베이스 모델을 미세 조정했습니다.

- **Performance Highlights**: DART로 생성된 데이터셋으로 훈련된 모델들은 기존의 거절 조정(vanilla rejection tuning) 또는 상용 모델들보다 뛰어난 성능을 보였습니다. 예를 들어, MATH 벤치마크에서 Llama3-8B 모델이 21.2%에서 46.6%로, GSM8K에서 51.0%에서 82.5%로 성능이 향상되었습니다. 이는 DART-MATH 데이터셋이 수학 문제 해결을 위한 매우 효율적이고 비용 효과적인 공개 리소스임을 입증합니다.



### FuLG: 150B Romanian Corpus for Language Model Pretraining (https://arxiv.org/abs/2407.13657)
- **What's New**: 이번 보고서에서는 FuLG라는 CommonCrawl에서 추출한 1천500억 토큰 규모의 루마니아어 코퍼스를 소개합니다. FuLG는 기존의 루마니아어 코퍼스보다 세 배 더 큰 규모로, 앞으로의 대규모 언어 모델(LLMs)의 사전 훈련과 미세 조정을 위해 공개된 데이터셋입니다.

- **Technical Details**: CommonCrawl에서 루마니아어 데이터를 추출하기 위해 다양한 필터링 기법을 사용하였습니다. 데이터 중복 제거와 품질 필터링을 위해 FastText와 RedPajama의 코드를 활용하였으며, HTML 텍스트 추출물과 논란의 소지가 있는 콘텐츠는 정규 표현식을 사용해 필터링하였습니다. 또한 개인정보(PII)를 포함한 항목은 특정 토큰으로 대체하였습니다.

- **Performance Highlights**: 데이터 중복 제거 후 데이터셋 크기가 37% 감소했으며, 퍼지 중복 제거(fuzzy deduplication)로 추가 50%가 감소하여 최종적으로 1560억 토큰(589GB)이 확보되었습니다. 다른 토크나이저를 사용할 경우, 예를 들어 Llama 3의 토크나이저를 사용하면, 약 2200억 토큰에 이릅니다.



### Weak-to-Strong Reasoning (https://arxiv.org/abs/2407.13647)
- **What's New**: 새로운 논문에서는 약한(weak) 모델을 이용해 강한(strong) 모델의 잠재적 능력을 발굴하고 자율적으로 훈련 데이터를 정제하는 '점진적 학습 프레임워크(progressive learning framework)'를 소개합니다. 이는 고급 모델 또는 인간 주석 데이터 없이 강한 모델이 스스로 훈련 데이터를 개선할 수 있게 합니다.

- **Technical Details**: 점진적 학습(framework)은 먼저 소규모의 고품질 데이터셋으로 감독 된(fine-tuning) 미세조정을 수행한 후, 강한 모델이 스스로 식별한 대조 샘플(contrastive samples)을 사용하여 선호도 최적화(preference optimization)를 진행합니다. 적용한 실험에서는 Llama2-70b 모델을 사용하며, 약한 모델로는 Llama2-7b, Gemma-2b, 그리고 Mistral-7b를 테스트했습니다.

- **Performance Highlights**: GSM8K와 MATH 데이터셋에서 실험 결과, 제안된 방법이 약 모델로만 감독된 경우에도 풀 웨이크 미세조정(full weak fine-tuning) 대비 성능을 크게 향상시키는 것으로 나타났습니다. 특히, GSM8K에서는 첫 번째 단계에서 26.99 포인트, 선호도 최적화 단계에서는 추가로 8.49 포인트가 향상되었습니다. 이 방법은 또한 OlympicArena 같은 복잡한 시나리오에서도 정확한 답변이 없는 상황에서 성능을 더 높였습니다.



### A Comparative Study on Automatic Coding of Medical Letters with Explainability (https://arxiv.org/abs/2407.13638)
Comments:
          working paper

- **What's New**: 이 연구는 자연어 처리(NLP) 및 머신러닝(ML) 기술을 사용하여 의료 서신의 코딩을 자동화하려는 시도를 탐구합니다. 현재 의료 현장에서 코딩은 수작업으로 수행되며, 이는 환자의 서류(예: SNOMED CT 코드 56265001 심장병)에 각 조건, 절차, 약물을 할당하는 것을 포함합니다. 이 연구는 지역 컴퓨터 설정에서 이러한 자동화 가능성을 탐구하고, AI 모델의 투명성을 위한 설명 가능성을 조사합니다.

- **Technical Details**: 연구에서는 ICD 코드 예측을 위해 공개적으로 사용 가능한 MIMIC-III 데이터베이스와 HAN/HLAN 네트워크 모델을 사용했습니다. ICD와 SNOMED CT 지식 베이스 간의 매핑(mapping) 실험도 실시했습니다. HAN 및 HLAN 모델은 텍스트의 계층적 속성을 활용하여 보다 정교한 예측을 가능하게 합니다. 특히, 이 모델들은 임상 코딩의 복잡성을 다룰 수 있는 다중 라벨 텍스트 분류 문제를 효율적으로 해결합니다.

- **Performance Highlights**: 실험 결과, 모델들은 97.98%의 코드에 대해 유용한 정보를 제공했습니다. 이는 자동 임상 코딩이 실제 병원 환경에서 구현될 가능성을 시사합니다. 연구 결과는 로컬 컴퓨터 환경에서 구동 가능한 경량 모델을 통해 실무에서도 사용 가능한 잠재력을 갖고 있음을 보여줍니다.



### Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies (https://arxiv.org/abs/2407.13623)
Comments:
          11 pages

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)의 확장 법칙에 있어 어휘 크기의 역할에 주목했습니다. 기존 연구들은 주로 모델 매개변수와 훈련 데이터 크기에 초점을 맞췄으나, 어휘 크기에 대한 고려가 부족했습니다. 저자들은 33M에서 3B 매개변수 범위의 모델을 500B 문자의 다양한 어휘 크기 구성으로 훈련하여, 계산 최적화 어휘 크기를 예측하는 세 가지 접근법을 제안합니다.

- **Technical Details**: 이 연구는 IsoFLOPs 분석, 도함수 추정 및 손실 함수의 파라메트릭 맞춤을 통해 최적 어휘 크기를 예측합니다. 모델의 비어휘 매개변수와 계산 자원을 고려하여 최적의 어휘 크기를 결정해야 함을 강조하고, 기존 다수의 LLM이 너무 작은 어휘 크기를 사용하고 있음을 지적합니다.

- **Performance Highlights**: 제안된 최적 어휘 크기를 채택한 모델은 기존의 일반적인 어휘 크기를 채택한 모델보다 성능이 향상되었습니다. 예를 들어, 어휘 크기를 32K에서 43K로 증가시키면서 동일한 2.3e21 FLOPs에서 ARC-Challenge 성능이 29.1에서 32.0으로 개선되었습니다.



### dzNLP at NADI 2024 Shared Task: Multi-Classifier Ensemble with Weighted Voting and TF-IDF Features (https://arxiv.org/abs/2407.13608)
Comments:
          Accepted for publication in the conference proceedings of ArabicNLP 2024

- **What's New**: 이번 논문에서는 dzNLP 팀이 NADI 2024 공유 과제, 특히 Subtask 1 - 다중 레이블 국가 수준 방언 식별(MLDID) (폐쇄 트랙)에서 수행한 기여를 소개합니다. 다양한 실험 구성을 통해 문제를 해결하려고 시도했습니다. 실험 1에서는 다양한 n-gram 값을 사용하는 n-gram 분석기를 결합하였고, 실험 2에서는 여러 가중치를 사용하여 Term Frequency-Inverse Document Frequency (TF-IDF) 특징의 가중된 결합을 시도했습니다. 실험 3에서는 세 가지 분류기(Linear Support Vector Classifier (LSVC), Random Forest (RF), K-Nearest Neighbors (KNN))를 사용한 가중 된 주요 투표 방식을 구현했습니다. 우리의 접근 방식은 단순함에도 불구하고 F1-score와 precision에서 경쟁력 있는 성과를 보여주었습니다.

- **Technical Details**: 우리의 제안 시스템은 세 가지 실험을 기반으로 했습니다. 실험 1에서는 세 가지 분석기(word, char, char_wb)를 사용한 TF-IDF 벡터화기를 사용하였고, 실험 2에서는 서로 다른 가중치를 가진 TF-IDF 특징을 결합하여 LSVC로 훈련했습니다. 실험 3에서는 LSVC, RF, KNN 분류기를 사용한 가중된 주요 투표 방식을 적용했습니다. N-GRAM의 범위는 unigram부터 5-gram까지 다양하게 실험했으며, TFIDF 변환기 가중치는 0.1에서 1, 특징의 최대 수는 300에서 1000까지 조정했습니다. LSVC의 정규화 매개변수 C는 1에서 5까지, 클래스 가중치는 'balanced'로 설정했습니다. RF 분류기는 기본 설정을 사용하였고, KNN에서는 이웃의 수를 3으로 설정했습니다. 주요 투표 기술에서는 가중치를 0.1에서 0.6까지 실험했습니다.

- **Performance Highlights**: 기본 실험에서 1-gram 특징과 선형 커널 SVC 분류기를 사용하여 낮은 F1-score 19.43%를 기록했습니다. 실험 1에서는 word와 char 수준의 n-gram을 결합하여 F1-score 20.64%를 달성했습니다. 실험 2에서는 다양한 n-gram 조합과 TF-IDF 특징 유형에 할당된 가중치의 변화를 탐색하여 F1-score 20.51%에서 22.51%까지 성과를 보였습니다. 실험 3에서 하드 투표 및 가중 하드 투표처럼 앙상블 방법을 도입했지만, 단일 LSVC 구성보다 낮은 F1-score를 기록했습니다. 따라서 앙상블 방법이 세부 조정된 단일 SVC 구성보다 효과적이지 않았음을 시사합니다.



### dzStance at StanceEval2024: Arabic Stance Detection based on Sentence Transformers (https://arxiv.org/abs/2407.13603)
Comments:
          Accepted for publication in the conference proceedings of ArabicNLP 2024

- **What's New**: 이번 연구에서는 Term Frequency-Inverse Document Frequency (TF-IDF) 특징과 Sentence Transformers를 비교하여 세 가지 주요 주제(코로나19 백신, 디지털 전환, 여성 권한 강화)에 대한 작가의 입장을 탐지합니다. 실험 결과, Sentence Transformers가 다양한 설정에서 TF-IDF 특징을 능가하는 것으로 나타났습니다.

- **Technical Details**: 입장 탐지는 특정 주제나 엔터티에 대해 텍스트에서 표현된 입장이나 태도를 판별하는 중요한 작업입니다. 기존의 TF-IDF 기반 방법은 텍스트를 숫자 형태로 변환하여 머신 러닝 모델이 이를 처리하도록 합니다. 그러나 이들은 텍스트 내의 깊이 있는 의미 관계와 컨텍스트적인 미묘함을 포착하는 데 제한이 있습니다. 최근 연구는 이러한 한계를 극복하기 위해 심층 학습 기술, 특히 Long Short-Term Memory (LSTM) 네트워크와 변환기(Transformers)를 활발히 활용하고 있습니다.

- **Performance Highlights**: Mawqif 2022 공유 작업에서 dzStance 팀은 여성 권한 강화에서 13위 (74.91%), 코로나19 백신에서 10위 (73.43%), 디지털 전환에서 12위 (66.97 %)로 전체 순위에서 13위 (71.77%)를 기록했습니다. 이 접근법은 다양한 주제에 대한 작가의 입장을 파악하는 데 뛰어난 성능을 발휘하며, 특히 F1-score에서 유망한 결과를 보여줍니다.



### PLANTS: A Novel Problem and Dataset for Summarization of Planning-Like (PL) Tasks (https://arxiv.org/abs/2407.13597)
- **What's New**: 이번 논문에서는 기존의 텍스트 요약 문제에서 벗어나 계획적(Planning-like, PL) 작업 요약이라는 새로운 문제를 제시합니다. PL 작업은 특정 목표를 달성하기 위해 연속적인 액션 시퀀스를 생성해야 하는 작업을 포함하는데, 예시로는 워크플로우, 요리법, 대화 및 여행 계획이 있습니다. 저자들은 새로운 PL 요약 문제를 도입하고, 데이터셋을 생성하며, 기본 방법을 제공하였습니다.

- **Technical Details**: 논문에서는 PL 요약 문제를 정의하고, 이 문제를 해결하기 위한 새로운 데이터셋 'PLANTS'를 소개합니다. PLANTS 데이터셋은 자동 계획, 요리법, 여행 계획과 같은 다양한 도메인을 포괄합니다. 또한, 기본 PL 요약 방법을 제안하고, 대규모 언어 모델(LLM)과의 성능 비교를 수행했습니다. PL 요약은 텍스트 요약과 달리 액션의 실행 가능성과 논리적 흐름을 고려해야 합니다.

- **Performance Highlights**: 제안된 방법과 LLM을 사용한 요약 결과를 정량적 메트릭과 정성적 사용자 연구를 통해 평가하였습니다. 이를 통해 기본 요약 방법과 큰 언어 모델의 성능을 비교분석한 결과, PL 요약 문제와 데이터셋이 연구 커뮤니티에 새로운 활력을 불러일으킬 수 있음을 확인했습니다. 연구의 주요 기여는 PL 작업 요약의 정의, PL 작업을 위한 데이터셋 생성, 요약 생성의 기본 방법 개발 및 사용자 인식 평가입니다.



### Towards Zero-Shot Multimodal Machine Translation (https://arxiv.org/abs/2407.13579)
Comments:
          Preprint. Under review

- **What's New**: 최근 멀티모달 기계 번역(Multimodal Machine Translation, MMT) 시스템은 완전한 감독 학습 데이터를 필요로 했습니다. 그러나 ZeroMMT라는 이번 연구는 영어 모노모달 데이터만을 사용하여 MMT 시스템을 학습시킴으로써 이 문제를 해결했습니다.

- **Technical Details**: ZeroMMT는 강력한 텍스트 전용 번역 모델을 기반으로 두 가지 목표로 학습됩니다: 시각적으로 조건화된 mask된 언어 모델링(VMLM)과 Kullback-Leibler 발산(KL divergence)을 통한 번역 성능 유지. 이는 이미지를 이용한 불명확한 텍스트 해석을 목표로 합니다.

- **Performance Highlights**: ZeroMMT는 표준 MMT 벤치마크뿐만 아니라 최신 CoMMuTE 벤치마크에서도 높은 성능을 보여줍니다. 특히, 새로운 언어 (아랍어, 러시아어, 중국어)에 대해 검증된 결과에서는 완전 감독 학습 데이터를 사용한 기존 모델에 필적하는 성능을 나타냈습니다. 추가 데이터 없이도 구분능력과 번역 충실도 간의 균형을 조절할 수 있습니다.



### Large Language Models as Reliable Knowledge Bases? (https://arxiv.org/abs/2407.13578)
- **What's New**: 최근 자연어 처리(NLP) 커뮤니티에서는 대형 언어 모델(LLM)을 지식 베이스(KB)로 활용하려는 관심이 증가하고 있습니다. 그러나 현재의 LLM이 얼마나 신뢰할 만한 KB로 기능할 수 있는지에 대한 연구는 부족합니다. 본 연구는 LLM-as-KB 모델의 신뢰성 평가 기준을 정의하고, 이러한 기준에 기반한 여러 메트릭스를 개발하여 26개의 인기 있는 LLM을 평가했습니다.

- **Technical Details**: 연구팀은 LLM이 KB로 기능할 때 중요한 두 가지 차원인 사실성(factuality)과 일관성(consistency)에 중점을 두었습니다. 사실성은 모델이 사실에 기반한 응답을 제공하는 능력을 의미하며, 일관성은 동일한 지식을 포함하는 질문에 대해 일관된 응답을 제공하는 능력을 의미합니다. 연구에서는 새롭게 설계된 UnseenQA 데이터셋을 활용하여 모델이 보지 못한 지식에 대한 평가를 수행했습니다.

- **Performance Highlights**: 평가 결과, GPT-3.5-turbo와 같은 고성능 모델조차도 사실성이나 일관성에서 신뢰할 만한 성과를 보이지 못했습니다. 또한, In-Context Learning (ICL)과 파인 튜닝(fine-tuning)과 같은 기법들도 LLM을 더 나은 KB로 개선하는데 성공적이지 않았습니다. LLM이 보지 못한 지식에 대한 응답에서는 높은 'uninformative rate'을 보여야 하며, 훈련중에 본 지식에 대한 평가에서는 높은 'correct rate'과 낮은 'wrong rate'이 필요합니다.



### dzFinNlp at AraFinNLP: Improving Intent Detection in Financial Conversational Agents (https://arxiv.org/abs/2407.13565)
Comments:
          Accepted for publication in the conference proceedings of ArabicNLP 2024

- **What's New**: dzFinNlp 팀은 금융 대화 에이전트의 의도 탐지를 위한 여러 모델과 특징 구성을 실험했습니다. 이 연구는 AraFinNLP 공유 작업의 일부로 진행되었으며, LinearSVC와 TF-IDF 같은 전통적인 머신러닝 방법론과 LSTM 등의 딥러닝 모델도 탐구했습니다. 또한, 우리는 Transformer 기반의 모델을 실험했습니다. 우리의 최고 모델은 ArBanking77 데이터셋의 개발 및 테스트 세트에서 각각 93.02%와 67.21%의 미세 F1 점수를 기록했습니다.

- **Technical Details**: 이 논문에서는 주로 전통적인 머신러닝 방법론과 딥러닝 기술을 결합하여 금융 대화 에이전트의 의도 탐지를 향상시키는 방법을 탐구합니다. 전통적인 머신러닝에서는 LinearSVC와 TF-IDF 벡터화를 사용하였으며, 딥러닝 측면에서는 단어 임베딩을 사용하는 LSTM 모델을 구현했습니다. 추가적으로, XLM-RoBERTa와 같은 Transformer 기반 아키텍처를 활용하여 사전 학습된 언어 표현으로부터 문맥 정보를 추출했습니다. 실험은 주로 scikit-learn 라이브러리를 이용하여 구현되었습니다.

- **Performance Highlights**: 우리의 연구에서 최고 모델은 ArBanking77 데이터셋의 개발 세트에서 93.02%의 미세 F1 점수를 기록했으며, 테스트 세트에서는 67.21%를 달성했습니다. 이 결과는 우리의 접근 방법이 금융 도메인의 대화 에이전트에서 의도 탐지 작업에 있어서 상당히 유망함을 보여줍니다. 특히, 우리는 팔레스타인 아랍어(PAL) 데이터셋을 중심으로 모델을 훈련시켜 높은 성능을 얻었습니다.



### Research on Tibetan Tourism Viewpoints information generation system based on LLM (https://arxiv.org/abs/2407.13561)
- **What's New**: 이번 연구는 티베트의 관광 서비스 인프라를 향상시키기 위해 혁신적인 AI 시스템인 DualGen Bridge AI를 소개합니다. 이 시스템은 감독 방식으로 미세 조정(Supervised Fine-Tuning)을 통해 모델 기능성을 강화하고 최적화 과정을 개선합니다. 뿐만 아니라, 다중 구조 생성 결과 평가 프레임워크를 도입하여 모델의 성능을 평가합니다.

- **Technical Details**: DualGen Bridge AI 시스템은 LLM(대규모 언어 모델, Large Language Model) 기반으로, SFT(Supervised Fine-Tuning)와 LoRA(Low-Rank Adaptation) 기법을 활용해 효율을 개선합니다. LLM 위치 정보 키워드 추출 모델과 관광 시점 정보 생성 모델, 그리고 이를 중계하는 Bridge 모델로 구성됩니다. 이는 세 가지 주요 처리 과정으로 나뉘며, 사용자 프롬프트(Prompt)에서 장소 정보를 추출하고, 위도와 경도 정보를 사용해 가까운 관광 시점을 계산한 뒤, 관련 정보를 생성합니다.

- **Performance Highlights**: 실험을 통해 DualGen Bridge AI 시스템이 티베트 관광 정보 생성에 있어 기존 LLM보다 우수한 성능을 보였음을 확인했습니다. 다중 소스로부터의 데이터 수집과 정제로 정확도를 높였으며, SFT+LoRA와 ORPO+LoRA 기법을 통해 생성된 정보의 품질과 일관성을 크게 향상시켰습니다.



### Can Open-Source LLMs Compete with Commercial Models? Exploring the Few-Shot Performance of Current GPT Models in Biomedical Tasks (https://arxiv.org/abs/2407.13511)
Comments:
          Version as accepted at the BioASQ Lab at CLEF 2024

- **What's New**: 이 논문에서는 최신의 상업용 대형 언어 모델(LLM)들인 OpenAI의 GPT-4와 Anthropic의 Claude 3 Opus와의 비교 분석 결과를 공개하였습니다. 특히, 상업용 모델들에 비해 경쟁력이 있는 오픈소스 모델들인 Mixtral 8x7B와 Llama 3에 대해 연구했습니다. BioASQ 챌린지에 참가하여 다양한 in-context learning (zero-shot, few-shot) 기법과 QLoRa fine-tuning을 사용하여 모델 성능을 비교하였습니다.

- **Technical Details**: 연구 팀은 BioASQ 챌린지 Task B 및 Synergy에 참여하였으며, PubMed에서 바이오 메디컬 관련 논문을 검색하고 질문에 대한 짧은 단락 스타일의 답변을 생성하는 RAG(Retrieval Augmented Generation) 환경을 활용하였습니다. 이 연구에서는 추가적으로 Wikipedia의 관련 지식을 LLM의 context-window에 추가해 성능 향상을 시도하였으나, 실질적인 성능 향상은 관찰되지 않았습니다.

- **Performance Highlights**: Mixtral 8x7B는 few-shot 설정에서 경쟁력 있는 성능을 보였지만, zero-shot 설정에서는 유용한 결과를 도출하는 데 실패했습니다. QLoRa fine-tuning과 Wikipedia 컨텍스트 추가는 측정 가능한 성능 향상을 가져오지 못했습니다. 상업용 모델과 오픈소스 모델 간의 성능 격차는 주로 zero-shot 설정에서 나타났으며, 몇 개의 샘플 예제를 수집함으로써 좁혀질 수 있음을 발견했습니다.



### Enhancing Biomedical Knowledge Discovery for Diseases: An End-To-End Open-Source Framework (https://arxiv.org/abs/2407.13492)
Comments:
          Under Review

- **What's New**: 의학 논문들이 엄청나게 늘어나고 있는 상황에서 특정 질병에 대한 지식을 직접 원문 텍스트로부터 구축하는 개방형 엔드투엔드 프레임워크가 소개되었습니다. 이 프레임워크는 Rett 증후군과 알츠하이머병 중심의 두 개의 주석 데이터셋을 통해 생물의료 엔티티 간의 의미적 관계를 식별할 수 있도록 도와줍니다.

- **Technical Details**: 데이터 파이프라인은 PubMed에서 관련 초록을 수집하고, 엔티티 탐지 및 UMLS와의 연계를 포함한 언급 추출을 통해 텍스트를 처리합니다. 추출된 텍스트와 공존 그래프를 활용해 수작업 주석을 위한 문장을 샘플링하고, Streamlit을 사용하여 구축된 주석 포탈을 통해 전문가 주석을 진행합니다. MetaMapLite 툴을 사용하여 다양한 생의학 개념을 커버하는 엔티티를 추출하고, 겹치거나 연속되는 엔티티를 통합합니다.

- **Performance Highlights**: Rett 증후군 (ReDReS) 데이터셋에서는 601개의 문장과 5,259개의 인스턴스, 알츠하이머병 (ReDAD) 데이터셋에서는 641개의 문장과 8,565개의 인스턴스가 구축되었습니다. 전문가들이 주석한 결합 결과로 각각 플리스 카파 점수가 0.6143과 0.6403을 기록하며 큰 일치도를 보였습니다.



### Combining Constraint Programming Reasoning with Large Language Model Predictions (https://arxiv.org/abs/2407.13490)
Comments:
          To appear at The 30th International Conference on Principles and Practice of Constraint Programming (CP 2024)

- **What's New**: 본 논문은 제약 프로그래밍(Constraint Programming, CP)과 머신 러닝(Machine Learning, ML)의 결합을 통해 텍스트 생성의 어려움을 해결하고자 합니다. 제약 프로그래밍은 구조적 제약을 관리하고, 대형 언어 모델(Large Language Model, LLM)은 단어 생성과 의미를 담당하는 방식입니다. 이 접근 방식은 GenCP라는, LLM이 생성한 도메인을 사용하는 OTFS(On-the-fly Constraint Programming Search)의 개선 버전에 기반합니다.

- **Technical Details**: 제안된 방안은 LLM이 변수 도메인을 관리하고, CP가 구조적 제약과 변수의 수를 관리하게 합니다. 이를 통해 생성된 문장이 의미를 가질 확률이 높고, 모든 제약을 충족시킬 수 있습니다. LLM은 텍스트 생성 중에도 사용되며, 문제 정의의 명시적 부분으로 사용됩니다. 본 논문은 GenCP라는 OTFS의 새로운 버전을 제안하며, 변수를 검색하는 동안 CP 변수를 LLM으로 계산하도록 수정된 생성 함수가 포함됩니다.

- **Performance Highlights**: 텍스트 생성 작업에서 GenCP 방식은 표준 NLP 방법인 Beam Search(BS)보다 빠르고 더 나은 결과를 제공합니다. 두 방법을 비교한 결과, 제안된 접근 방식은 모든 제약을 만족시키면서 향상된 성능을 보여줍니다.



### Attention Overflow: Language Model Input Blur during Long-Context Missing Items Recommendation (https://arxiv.org/abs/2407.13481)
Comments:
          Dataset URL: this https URL

- **What's New**: 최신 연구는 대형 언어 모델(LLMs)이 제공된 항목 목록에서 누락된 요소를 제안할 수 있지만, 항목이 너무 많아지면 성능이 저하된다는 것을 발견했습니다. 이 현상은 입력 목록에 이미 포함된 항목을 다시 제안하기 시작하는 약 100개의 항목 기준으로 발생합니다. 이를 주목하여 'attention overflow' 문제로 명명했습니다.

- **Technical Details**: 이 문제는 주어진 섞인 정수 범위에서 누락된 숫자를 찾는 합성 문제(synthetic problems)와 현실적인 영화 추천 시나리오에서 평가되었습니다. 'attention overflow' 문제는 모든 항목에 동시에 주의를 기울이는 것을 필요로 합니다. 반복 루프(iterative loops)를 사용하면 이 문제를 완화할 수 있지만, 반복률이 증가하는 경우 비용이 늘어나고 긴 입력에서 새로움을 도출하는 언어 모델의 능력에 영향을 줍니다.

- **Performance Highlights**: 2024년 중반 플래그십 대형 언어 모델에서는 약 100개의 항목에서 'attention overflow' 문제가 발생하기 시작하며, 반복 작업(iterative loops)을 통해 문제를 일부 완화할 수 있으나 비용 측면에서 비효율적일 수 있습니다.



### Fixed and Adaptive Simultaneous Machine Translation Strategies Using Adapters (https://arxiv.org/abs/2407.13469)
Comments:
          Accepted at IWSLT 2024

- **What's New**: 이 논문은 동시 기계번역(SiMT, Simultaneous Machine Translation)에서 여러 지연 수준을 지원할 수 있는 하나의 모델을 개발하는 방법을 제시합니다. 구체적으로, 디코더에 가벼운 어댑터 모듈(adapter modules)을 도입하여 wait-$k$ 정책을 적용하는데, 이는 입력 단어 $k$개를 처리한 후 번역을 시작하는 방식입니다. 어댑터는 다양한 wait-$k$ 값을 처리하도록 훈련되어, 여러 모델을 학습해야 하는 번거로움을 줄입니다.

- **Technical Details**: 어댑터 모듈은 디코더에 삽입되어 특정 wait-$k$ 값에 맞춰 워드의 읽기(READ) 및 쓰기(WRITE) 동작을 조절합니다. 어댑터 모듈은 경량화되어, 파라미터 공유와 간섭을 최소화하면서도 유연성을 제공합니다. 이를 통해 여러 지연 수준을 다룰 수 있는 단일 모델을 구축할 수 있습니다. 또한, 어댑터를 적응형 wait-$k$(Adaptive Wait-$k$) 전략과 결합하여 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 두 가지 언어 방향에서 대부분의 지연 수준에서 기존 강력한 기준(baselines)과 경쟁하거나 뛰어남을 보여주었습니다. 특히, wait-$k$ 어댑터와 적응형 전략을 결합한 결과, 지연 수준별로 더 나은 성능을 보였습니다.



### End-To-End Clinical Trial Matching with Large Language Models (https://arxiv.org/abs/2407.13463)
Comments:
          149 pages, including Supplements. 3 Main Figures

- **What's New**: 임상 시험 매칭을 위한 새로운 엔드-투-엔드(End-to-End) 파이프라인이 소개되었습니다. 이 시스템은 수동 작업 없이도 105,600개의 암 관련 임상 시험 중 관련 시험을 자동으로 찾아내고, 각 환자의 전자 건강 기록(EHRs)과 시도 기준별로 매칭이 가능합니다. 이는 Large Language Model(LLM)인 GPT-4o를 사용하여 가능해졌습니다.

- **Technical Details**: GPT-4o와 51개의 합성 EHRs를 사용하여 임상 시험 후보를 식별하고 기준 수준에서 환자 정보를 매칭합니다. 초기 인간 전문가 기준과 비교하여 93.3%의 사례에서 관련 후보 시험을 식별했으며, 기준 수준에서의 매칭 정확도는 88.0%였습니다. LLM 피드백을 사용하여 기준을 재평가함으로써 최종 모델 정확도는 92.7%로 증가했습니다.

- **Performance Highlights**: 이 접근 방식은 고도로 정밀한 시험 스크리닝과 개별 환자와의 매칭을 활용하여 의사보다 성능이 더 우수한 것으로 나타났습니다. 완전 자동화된 상태로 또는 인간 감독 하에 작동할 수 있는 이 파이프라인은 실제 환경에서도 암 환자 뿐만 아니라 다른 질병에 대해서도 확장 가능한 솔루션을 제공합니다.



### Enhancing Out-of-Vocabulary Performance of Indian TTS Systems for Practical Applications through Low-Effort Data Strategies (https://arxiv.org/abs/2407.13435)
Comments:
          Accepted at INTERSPEECH 2024

- **What's New**: 이번 연구는 힌디와 타밀어처럼 훈련 데이터가 적은 저자원 언어(low-resource languages)를 위한 새로운 접근법을 제안합니다. 구체적으로는 자원봉사자들이 이전의 훈련 데이터에 없는 문자 바이그램을 포함한 단어를 녹음함으로써 저비용으로 추가 훈련 데이터를 확보하는 방법을 통해 TTS 시스템 성능을 향상시키고자 합니다.

- **Technical Details**: 본 연구에서는 다양한 실세계 애플리케이션을 염두에 두고 OOV(out-of-vocabulary) 단어를 포함하는 벤치마크 데이터를 생성하였습니다. 이를 통해 기존 힌디와 타밀어 TTS 시스템의 OOV 단어 처리 성능이 낮음을 확인했습니다. 해결책으로는 고품질의 음성 아티스트 대신 자원봉사자의 음성을 사용하여 훈련 데이터를 확장하고, 이는 비용 효율적임과 동시에 OOV 단어 처리 성능을 향상시킨다는 점을 보였습니다.

- **Performance Highlights**: 새로운 방법론을 적용해 TTS 시스템의 OOV 단어 처리 성능이 개선되었습니다. 'IndicOOV'라는 벤치마크를 통해 다양한 카테고리의 OOV 단어에 대한 성능을 평가했을 때, 저비용의 데이터 사용에도 불구하고 음성 품질과 인도어(IV) 단어에 대한 성능이 유지됨을 확인했습니다.



### From Words to Worlds: Compositionality for Cognitive Architectures (https://arxiv.org/abs/2407.13419)
Comments:
          Accepted to ICML 2024 Workshop on LLMs & Cognition

- **What's New**: 최근 연구는 대규모 언어 모델(LLMs)의 구성적 능력 향상에 관한 새로운 통찰을 제공합니다. 이 연구는 LLMs가 규모가 커짐에 따라 구성적 능력이 향상되지만, 지시 조정(instruction tuning)은 종종 그 반대 효과를 가져올 수 있음을 발견했습니다.

- **Technical Details**: 이 연구는 네 가지 LLM 계열(Falcon, LLama, Codellama, Mistral)의 열두 모델을 대상으로 세 가지 작업 유형을 포함한 실증 분석을 수행했습니다. 주요 작업 유형에는 형용사-명사 조합(Adj-N) 구성 특성을 평가하는 세 가지 과제가 포함됩니다: 대체성(Substitutivity), 체계성 및 전체주의(Systematicity & Globalism), 과일반화(Over-generalization)입니다.

- **Performance Highlights**: 모든 모델 계열에서 Larger Model이 Base Model보다 항상 더 나은 성능을 보였습니다. 그러나 지시 조정의 영향은 일관되지 않았으며, 일부 모델에서는 성능이 감소하거나 증가하거나 변함이 없었습니다. 특히, Codellama 모델의 경우, 지시 조정 후에도 성능 변동이 없었습니다.



### Linear-Complexity Self-Supervised Learning for Speech Processing (https://arxiv.org/abs/2407.13377)
Comments:
          Interspeech 2024

- **What's New**: 자기 지도 학습(Self-supervised learning, SSL) 모델의 높은 사전 학습 비용 문제를 해결하기 위해, 이 논문에서는 MHSA(Multi-Headed Self-Attention) 대신 선형 복잡도(linear-complexity)의 컨텍스트 인코더를 탐구합니다. 이 논문에서 사용된 SummaryMixing 모델은 wav2vec 2.0 모델에 통합되었으며, 사전 학습 시간을 18%, VRAM 사용량을 23% 감소시켰습니다. 이는 4개의 Tesla A100 GPU로 한 주 만에 155M wav2vec 2.0 모델을 사전 학습할 수 있게 했습니다.

- **Technical Details**: SummaryMixing은 MHSA 대신에 선형 복잡도의 모델로, 지역적 정보는 point-wise feed-forward network로 포착하고, 전역적 정보는 입력 프레임의 평균 벡터를 활용하여 포착합니다. 두 가지 정보를 결합하여 입력의 숨겨진 표현을 형성합니다. 본 연구는 이 구조가 SSL에서도 효과가 있는지를 확인하기 위해 wav2vec 2.0 모델에 적용했습니다. 결과적으로, 다양한 다운스트림 과제에서 기존 MHSA 기반 모델과 비슷하거나 더 나은 성능을 보여주었습니다.

- **Performance Highlights**: SummaryMixing을 사용한 wav2vec 2.0 모델은 MP3S 벤치마크의 다운스트림 과제(자동 음성 인식, 의도 분류, 감정 인식, 자동 화자 검증)에서 기존 MHSA Conformer 기반 모델과 동일하거나 더 나은 성능을 보였습니다. 효율성 측면에서, 이 모델은 사전 학습 시간과 VRAM 사용량을 각각 18%, 23% 줄였습니다.



### Capturing Style in Author and Document Representation (https://arxiv.org/abs/2407.13358)
- **What's New**: 이 논문에서는 기존의 NLP 모델이 글의 스타일을 명시적으로 캡처하지 못하는 한계를 극복하기 위해 새로운 아키텍처를 제안합니다. 제안된 모델은 Variational Information Bottleneck(VIB) 프레임워크를 기반으로, 작가와 문서의 임베딩을 동시에 학습하며 스타일 제약을 추가합니다. 이를 통해 작가 식별 및 분류와 같은 NLP 태스크뿐만 아니라 추천 시스템에도 활용될 수 있습니다.

- **Technical Details**: 제안된 모델은 미리 훈련된 문서 인코더(document encoder)를 미세 조정(fine-tune)하며, 스타일 지표와 관련된 사전 정의된 특성들을 입력 특징으로 추가하여 스타일 감지를 촉진합니다. 중요한 목적 함수 항목을 추가함으로써 스타일 수집(representations)을 보장합니다. 이 모델은 작가 식별 및 스타일 특성 예측 태스크에서 검증되었습니다.

- **Performance Highlights**: 이 모델은 Project Gutenberg, Blog Authorship Corpus, 그리고 IMDb62 데이터셋에서 기존의 강력하고 최근의 기준선을 초과하거나 일치하는 성능을 보여 주었으며, 특히 작가의 스타일적 특징을 훨씬 더 정확하게 캡처하는 데 뛰어난 성능을 보였습니다. 이를 통해 학교, 문학연구자, 그리고 대중에게 해석할 수 있는 표현 공간을 구축할 수 있었습니다.



### Learning-From-Mistakes Prompting for Indigenous Language Translation (https://arxiv.org/abs/2407.13343)
- **What's New**: 이번 논문에서는 매우 저자원 언어 인디언 언어 번역을 향상시키기 위한 기법들을 제시합니다. 이 연구는 (1) 제한된 평행 번역 예제들로 이루어진 데이터저장소(datastore) 사용, (2) GPT-3.5와 같은 대형 언어 모델(LLMs)의 고유 기능, (3) 단어 수준 번역 사전(word-level translation dictionary)의 활용을 기반으로 하고 있습니다. 특히, 인디언 언어 번역을 위하여 LLMs와 컨텍스트 학습(in-context learning) 기법들을 어떻게 적용할 수 있는지 다룹니다. 세 가지 주요 기법으로 KNN-Prompting with Retrieved Prompting Context, Chain-of-Thought Prompting, 그리고 Learning-from-Mistakes Prompting을 도입하여 낮은 자원에서도 효과적인 번역을 이루게 합니다.

- **Technical Details**: 이 연구는 LLMs가 저자원 언어에 대한 번역을 어떻게 수행할 수 있는지에 초점을 맞추고 있습니다. 특히, LLMs가 새로운 언어 쌍의 구문 구조를 내재화 할 수 있다는 가정 하에, KNN-Prompting with Retrieved Prompting Context(RPC), Chain-of-Thought(CoT) Prompting, Learning-from-Mistakes(LFM) Prompting 등의 기법을 도입하여 LLMs를 언어 컴파일러로 활용합니다. 각 기법은 계층적으로 누적되며, RPC는 문맥적 유사성을 활용, CoT은 교육적 잠재력을 활용, LFM은 과거 오류를 보완합니다.

- **Performance Highlights**: 평가 결과는 제한된 말뭉치(corpus)에서도 적절한 프롬프팅 방법을 사용하면 LLMs가 매우 저자원 언어도 효과적으로 번역할 수 있음을 시사합니다. 특히, 제한된 평행 말뭉치를 사용하여 번역의 정밀도를 높이는 데 성공했습니다. 이 연구는 사전 훈련된 데이터에 포함되지 않은 언어를 대상으로 한 번역 메커니즘을 연구의 주요 혁신점으로 제시합니다.



### Why do you cite? An investigation on citation intents and decision-making classification processes (https://arxiv.org/abs/2407.13329)
Comments:
          42 pages, 14 figures, 1 table, submitted to Scientometrics Journal

- **What's New**: 이 연구는 저자가 다른 작업을 인용하는 이유를 분석하는 새로운 접근 방식을 제시합니다. 고급 앙상블 전략(Ensemble Strategies)과 언어 모델(Language Models)을 사용하여 인용 의도를 보다 신뢰할 수 있게 분류하고, 설명 가능한 AI(XAI) 기법을 활용하여 모델의 해석 가능성과 신뢰성을 높였습니다. 특히, 연구 결과는 섹션 타이틀(section titles)의 포함이 분류 성능을 크게 향상시킨다는 점을 강조합니다.

- **Technical Details**: 이 연구는 두 가지 앙상블 분류기(ensemble classifiers)를 사용합니다. 여기서 기초 모델(Baseline)로 SciBERT와 XLNet 언어 모델(LMs)을 미세 조정(fine-tuning)하여 사용했습니다. 또한, 결정 프로세스를 해석할 수 있도록 설명 가능한 AI(XAI)를 통합하여, 개별 단어들이 레벨-0(level-0) 분류에 미치는 영향과 개별 모델들이 메타분류(metaclassification)에 미치는 영향을 시각화했습니다. 마지막으로, CIC 작업의 성능을 향상시키기 위해 Flask 프레임워크를 사용하여 웹 애플리케이션을 개발했습니다.

- **Performance Highlights**: 우리의 모델 중 하나는 SciCite 벤치마크에서 89.46%의 매크로 F1 점수(Macro-F1 score)를 달성하여, 새로운 최첨단 성능(state-of-the-art)을 세웠습니다. 이는 섹션 타이틀(section titles)을 포함함으로써 모델 성능이 크게 향상되었음을 보여줍니다. 이러한 발견은 더 강력한 데이터세트와 방법론을 개발하는 데 유용한 통찰력을 제공합니다.



### CoD, Towards an Interpretable Medical Agent using Chain of Diagnosis (https://arxiv.org/abs/2407.13301)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)을 기반으로 하는 의학 진단의 해석 가능성을 향상시키기 위해 Chain-of-Diagnosis(CoD)를 도입했습니다. CoD는 진단 과정을 의료 전문가의 사고 과정을 반영하는 진단 체인으로 변환하여 투명한 설명 경로를 제공합니다. 또한, 질병 신뢰도 분포를 출력하여 의사 결정을 투명하게 만듭니다. 이 모델로 DiagnosisGPT를 개발하여 9604개의 질병을 진단할 수 있게 했으며, 기존 모델보다 뛰어난 성능을 보였습니다.

- **Technical Details**: CoD는 진단 과정을 다섯 단계로 구분하여 해석 가능성을 높입니다. 첫 번째 단계는 증상 요약이며, 두 번째 단계는 질병 식별, 세 번째 단계는 분석, 네 번째 단계는 결정, 마지막 단계는 신뢰도 분포를 출력하는 것입니다. 이 과정을 통해 진단이 이루어지며, 각 단계는 분명한 기능을 가지고 있어 진단 과정의 해석 가능성을 높입니다. 또한 CoD는 실세계 데이터를 대신하여 질병 백과사전 데이터로부터 합성 사례를 생성하여 훈련 데이터를 만듭니다.

- **Performance Highlights**: DiagnosisGPT는 9604개의 질병을 자동으로 진단할 수 있으며, 다양한 진단 데이터셋과 새롭게 만든 DxBench에서 다른 대형 언어 모델보다 뛰어난 성능을 보였습니다. 모든 데이터셋에서 0.55의 진단 기준으로 90% 이상의 정확도를 달성하였으며, 이는 신뢰도의 높은 수준을 나타냅니다. 또한, CoD의 신뢰도 기반 의사 결정은 진단 정확도를 높이는데 도움을 줍니다.



### Robust ASR Error Correction with Conservative Data Filtering (https://arxiv.org/abs/2407.13300)
- **What's New**: 최근 음성 인식 시스템(ASR)의 성능 향상을 위해 대형 언어 모델 기반의 오류 수정(EC)이 떠오르고 있습니다. 이 연구에서는 EC 모델 훈련 중 발생할 수 있는 데이터 노이즈를 줄이고 과도한 수정(overcorrection)을 방지하는 '보수적 데이터 필터링' 방법을 제안합니다. 일본어 ASR 및 일본어 LLMs를 대상으로 실험을 진행했으며, 다양한 도메인에서 성능을 비교 평가했습니다.

- **Technical Details**: 일반적으로 EC 훈련 데이터는 ASR 가설(source)과 금 표준 참조(target)를 자동으로 매칭하여 수집됩니다. 그러나 이러한 페어의 품질이 보장되지 않아 훈련 데이터에 노이즈가 섞일 수 있습니다. 본 연구에서는 EC 훈련 데이터가 두 가지 기준, 즉 (1) 언어적 수용성을 향상시키고 (2) 주어진 문맥에서 추론 가능해야 한다는 점을 강조합니다. 이를 통해 저품질의 데이터 페어를 식별해 모델이 잘못된 수정을 피할 수 있도록 훈련합니다. 실험은 일본어 ASR에 대해 Conformer-CTC 모델을 기본으로 하고, Swallow-Mistral 7B 및 Sarashina-2 7B LLMs를 미세 조정하여 수행되었습니다.

- **Performance Highlights**: 21개의 내부 벤치마크를 통해 실험한 결과, 본 방법이 과도한 수정을 크게 줄이고, 특히 도메인 외(out-of-domain) 환경에서 ASR의 정확도와 품질을 향상시키는 것이 확인되었습니다. 이는 기존의 단순한 데이터 필터링 방법에 비해 훨씬 효과적임이 증명되었습니다.



### SpeciaLex: A Benchmark for In-Context Specialized Lexicon Learning (https://arxiv.org/abs/2407.13297)
- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)이 특정한 어휘와 관련된 제약을 얼마나 잘 구현할 수 있는지 평가하기 위한 새로운 기준인 'SpeciaLex'를 소개합니다. 이 기준은 18개의 다양한 하위 과제와 1,285개의 테스트 인스턴스를 포함해 LLM의 성능을 종합적으로 평가할 수 있도록 한다는 점에서 유의미합니다.

- **Technical Details**: SpeciaLex는 LLM의 어휘 기반 제약을 평가하는 벤치마크로, Checking, Identification, Rewriting, Open Generation 네 가지 핵심 과제를 포함합니다. 이 연구에서는 총 15개의 최신 LLM을 대상으로 성능을 평가하였으며, 모델 크기, 개방성, 설정 및 최신성 등 성능에 영향을 미치는 요인을 분석하였습니다.

- **Performance Highlights**: 평가 결과, 모델의 크기와 최신성, 개방성 등이 성능에 큰 영향을 미치는 것으로 나타났습니다. 특히, 다양한 상용 및 오픈소스 LLM에 대한 비교를 통해 특정 제약 조건을 준수하는 텍스트 생성 능력에서의 차이를 발견하였습니다. SpeciaLex는 한정된 컴퓨팅 예산을 가진 연구자들이 필요에 따라 모델을 선택할 수 있는 참고 자료로 유용하게 활용될 수 있을 것입니다.



### Are Large Language Models Capable of Generating Human-Level Narratives? (https://arxiv.org/abs/2407.13248)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)의 이야기 만들어내는 능력을 탐구하며, 서사 전개와 플롯 진행에 중점을 둡니다. 우리는 새로운 계산 프레임워크를 도입하여 서사를 세 가지 담론 수준에서 분석합니다: 이야기 아크(story arcs), 전환점(turning points), 그리고 정서적 차원(affective dimensions)인 각성(arousal)과 기질(valence)을 포함합니다. 전문가와 자동 주석을 활용하여, LLM이 생성한 이야기와 인간이 작성한 이야기 사이의 중요한 불일치를 발견했습니다. 인간이 작성한 이야기는 긴장감, 다양성, 정서적 충격을 잘 포함하는 반면, LLM이 생성한 이야기는 일관되게 긍정적이며 긴장이 부족합니다.

- **Technical Details**: 이 논문에서는 서사 분석을 위해 세 가지 주요 담론 수준을 측정합니다: 1) 이야기 아크(거시 수준), 2) 전환점(중간 수준) 그리고 3) 각성과 기질(미시 수준)입니다. 이를 위해 영화 시놉시스 데이터셋을 수집하고, 각 수준에 대해 인간과 자동 주석을 실시했습니다. LLM이 생성한 이야기는 중요한 전환점을 제대로 전개하지 못하고, 플롯의 진행에서 긴장감과 다양성이 부족하다는 점을 발견했습니다. 이에 비해, 이야기 아크 및 전환점의 정보를 이용한 명시적 통합은 이야기 생성 능력을 크게 향상시킬 수 있었습니다.

- **Performance Highlights**: 우리는 담론 특성을 통합하는 생성 과정이 LLM의 서사 구성 능력을 향상시킬 수 있음을 보여주었습니다. 두 개의 평행 실험에서, 스토리 아크에 대한 인식을 통합하면 모델의 다양성이 45% 향상되는 반면, 전환점 정보를 통합하면 서사의 긴장감과 몰입도가 40% 향상되었습니다. 이는 인간 수준의 이야기 생성에는 여전히 부족하지만, LLM의 서사구조 이해와 생성 능력을 향상시킬 수 있는 가능성을 시사합니다.



### PM-LLM-Benchmark: Evaluating Large Language Models on Process Mining Tasks (https://arxiv.org/abs/2407.13244)
- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)이 프로세스 마이닝(Process Mining, PM) 작업을 반자동화할 가능성을 연구합니다. 특히, 다양한 상업용 모델들이 이미 많은 분석 작업에 적합하다는 점을 고려할 때, 오픈 소스 LLM의 PM 작업 수행 능력을 평가하기 위한 PM-LLM-Benchmark을 제안합니다. 이는 프로세스 마이닝 작업을 수행할 수 있는 첫 종합 벤치마크로, 도메인 지식(프로세스 마이닝 도메인 및 구체적인 도메인 지식)을 포함하여 다양한 구현 전략에 중점을 둡니다.

- **Technical Details**: PM-LLM-Benchmark는 인사이트 직접 제공, 코드 생성 등의 두 가지 구현 패러다임을 기반으로 LLM의 프로세스 마이닝 작업 수행 능력을 평가합니다. 여러 카테고리에 '정적' 프롬프트(TXT 파일에 저장된)를 포함하여 PM 도메인 및 구체적인 도메인 지식이 요구됩니다. 또한, 텍스트 및 코드 응답의 품질을 평가하는 확장 가능한 평가 전략을 제시하며, LLM이 제공한 답변의 평가 편향을 극복하고 더 철저한 LLM 순위를 실행할 필요가 있음을 언급합니다.

- **Performance Highlights**: 가장 주목할 만한 실험 결과는 대부분의 LLM이 만족스러운 수준의 프로세스 마이닝 작업을 수행할 수 있지만, 엣지 디바이스에서 실행될 수 있는 작은 모델은 여전히 부적절하다는 것입니다. 프로세스 마이닝 작업에 적합한 LLM을 식별하는 데에는 유용하지만, 고도로 성능이 뛰어난 LLM의 점수를 비교할 때는 주의가 필요합니다.



### Evaluating Large Language Models for Anxiety and Depression Classification using Counseling and Psychotherapy Transcripts (https://arxiv.org/abs/2407.13228)
- **What's New**: 이번 연구에서는 전통적 기계 학습 방법과 대규모 언어 모델(LLMs)이 장문의 대화 로그(transcripts)에서 불안과 우울증을 분류하는 데 어느 정도 효과적인지를 평가하였습니다. 특히 최신 모델들이 기존의 기계 학습 방법들에 비해 성능 개선을 이루지 못했다는 점이 주목됩니다.

- **Technical Details**: 이번 연구에서 미세 조정(fine-tuning)된 변환기(Transformers) 모델로는 BERT, RoBERTa, Longformer를 사용했으며, 최신 대형 모델로는 Mistral-7B를 사용했습니다. 또한, 특징 공학(feature engineering)을 통해 Support Vector Machine(SVM)을 훈련하고, GPT 모델은 프롬팅(prompting)을 통해 평가했습니다.

- **Performance Highlights**: 최신의 최첨단(state-of-the-art) 모델들이 분류 결과를 기존의 전통적인 기계 학습 방법에 비해 향상시키지 못했다는 결론에 도달했습니다. 즉, 최신 모델들의 성능이 기대에 미치지 못했다는 것을 의미합니다.



### Transformer-based Single-Cell Language Model: A Survey (https://arxiv.org/abs/2407.13205)
- **What's New**: 최신 리뷰 논문에서는 트랜스포머(Transformers) 모델이 단일 세포 데이터(single-cell data)를 분석하는 데 어떻게 활용되고 있는지를 체계적으로 다룹니다.

- **Technical Details**: 트랜스포머 모델은 뛰어난 병렬 처리 능력과 유연한 주의 메커니즘(attention mechanism)을 기본으로 하며, 기계 학습 초기에는 n-그램(n-gram)과 은닉 마르코프 모델(HMM)을 사용했으나, 현재는 리커런트 신경망(RNN)과 합성곱 신경망(CNN)같은 깊은 학습 모델이 주목받고 있습니다. 특히 구글이 개발한 트랜스포머는 문장 전체를 한 번에 처리하고, 위치 인코딩을 통해 장기 의존성을 학습하는 뛰어난 능력을 보이고 있습니다.

- **Performance Highlights**: Transformer-XL은 고유의 재귀 메커니즘과 위치 인코딩을 통해 긴 장기 의존성을 학습합니다. Reformer는 메모리 효율을 높이기 위해 가역 잔차 층(reversible residual layers)을 사용하며, BERT는 양방향 트랜스포머와 마스크 메커니즘을 도입하여 문맥 정보를 잘 반영합니다. XLNet과 RoBERTa 같은 모델은 각각 Transformer-XL과 BERT를 기반으로 개선된 성능을 보여줍니다.



### Retrieval-Augmented Generation for Natural Language Processing: A Survey (https://arxiv.org/abs/2407.13193)
- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 한계를 보완하기 위해 외부 지식 데이터베이스를 활용하는 Retrieval-Augmented Generation(RAG)의 주요 기술들을 리뷰합니다. 특히 검색기(retriever)와 검색 통합(retrieval fusion)의 기술에 중점을 두고 있으며, 실제 구현을 위한 튜토리얼 코드도 제공합니다.

- **Technical Details**: RAG는 크게 세 가지 모듈로 구성됩니다: 검색기(retriever), 생성기(generator), 그리고 검색 통합(retrieval fusion). 검색기 모듈은 인코더(encoder), 효율적인 인덱싱(indexing), 그리고 외부 지식을 저장하는 데이터스토어(datastore)로 구성됩니다. 검색 통합 기술은 입력 데이터에 검색된 정보를 통합하여 생성합니다. 이 논문에서는 쿼리 기반 통합(query-based fusion), 잠재적 통합(latent fusion), 로짓 기반 통합(logits-based fusion) 등 다양한 통합 기법을 다룹니다.

- **Performance Highlights**: RAG를 사용하여 대형 언어 모델의 주요 문제인 환각 문제(허상 생성), 지식 업데이트 문제, 그리고 도메인별 전문 지식의 부족을 해결할 수 있습니다. RAG는 외부 지식 데이터베이스로부터 최신 정보를 제공함으로써 더 정확하고 신뢰성 있는 결과를 생성할 수 있게 합니다.



### Translate-and-Revise: Boosting Large Language Models for Constrained Translation (https://arxiv.org/abs/2407.13164)
Comments:
          16 pages

- **What's New**: 이 논문에서는 기계 번역 시스템에서 사전 설정된 제약 조건을 따르는 번역을 수행하는 새로운 방법을 제안합니다. 특히, 대형 언어 모델(LLMs)을 사용하여 제약 조건 기반 번역의 정확성을 높이는 'Translate-and-Revise(TAR)' 전략을 소개합니다. 이 접근 방식은 기존의 번역 결과를 재검토하고 부족한 제약 조건을 강화하여 번역 품질을 향상시킵니다.

- **Technical Details**: TAR 접근 방식은 소스 언어 문장과 해당 이중언어 제약 조건을 입력으로 받아 처음에는 LLM을 사용해 초기 번역을 수행합니다. 이후 모델이 제약 조건을 완전히 따르지 못한 경우, 재검토 및 수정 과정을 거쳐 남은 제약 조건을 만족시키도록 유도합니다. 이를 위해 다중 반복 검토 과정을 도입하여 각 번역이 최대한 제약 조건을 충족할 때까지 수정됩니다.

- **Performance Highlights**: TAR 접근 방식을 사용한 결과, 제안된 방법은 네 가지 다양한 제약 번역 데이터셋에서 기존의 NMT(state-of-the-art) 방법을 능가하는 성과를 보였습니다. 특히, 제약 기반 번역 정확도에서 15% 이상의 향상을 달성하였으며, 여러 도메인에서 일관되게 SoTA(state-of-the-art) 결과를 기록하였습니다.



### Preset-Voice Matching for Privacy Regulated Speech-to-Speech Translation Systems (https://arxiv.org/abs/2407.13153)
Comments:
          Accepted to the ACL PrivateNLP 2024 Workshop, 7 pages, 2 figures

- **What's New**: 이 논문은 Preset-Voice Matching(PVM)이라는 새로운 규제된 음성-음성 번역(S2ST) 프레임워크를 제안합니다. PVM은 음성 복제를 직접적으로 수행하는 대신, 입력 음성과 유사한 사전 동의된 목표 언어 음성을 매칭함으로써 개인의 권리를 보호하고 오남용 위험을 줄입니다.

- **Technical Details**: PVM 프레임워크는 유사도 특징 추출 모듈(Similarity Feature Extraction), 프리셋-음성 라이브러리(Preset-Voice Library), 그리고 텍스트-음성 합성(TTS) 모듈로 구성되어 있습니다. GEMO-Match 알고리즘은 성별-감정 기반의 계층적 감정 분류기 아키텍처를 사용하여 PVM을 구현합니다.

- **Performance Highlights**: PVM은 다중 화자 자동 더빙 상황에서 시스템 실행 시간을 크게 줄이고, S2ST 합성 음성의 자연스러움을 향상시키는 것으로 나타났습니다. 또한, PVM은 특허 대비 안전하고, 다중 언어 환경에 보다 적합한 것으로 입증되었습니다.



### A light-weight and efficient punctuation and word casing prediction model for on-device streaming ASR (https://arxiv.org/abs/2407.13142)
- **What's New**: ASR(Automatic Speech Recognition)을 위한 실시간 도구로, 점과 대문자 예측을 함께 수행하는 경량 모델을 제안했습니다. 이 모델은 특히 디바이스에 직접 내장해 사용할 수 있도록 설계되었습니다.

- **Technical Details**: 새로 제안된 모델은 Convolutional Neural Network(CNN)와 Bidirectional Long Short-Term Memory(BiLSTM)를 기반으로 합니다. 이를 통해 실시간으로 점과 대문자 예측을 효율적으로 수행할 수 있습니다. Transformer 기반 모델들이 주로 사용되나 너무 크기 때문에 이보다 가벼우면서 효율적인 대안을 제시하려 합니다.

- **Performance Highlights**: 실험 결과는 IWSLT2011 테스트 세트에서 제안된 모델이 전반적인 F1-score에서 비Transformer(non-Transformer) 모델들보다 9%의 상대적 향상을 보여줬습니다. 또한, Transformer 기반 모델들과 성능이 거의 유사하면서도 모델 크기는 1/40에 불과하며 추론 시간도 2.5배 빠릅니다. 이러한 특성으로 인해 디바이스 내에서의 실시간 ASR 시스템에 매우 적합합니다.



### Retrieve, Summarize, Plan: Advancing Multi-hop Question Answering with an Iterative Approach (https://arxiv.org/abs/2407.13101)
- **What's New**: 본 연구에서는 다중 홉 질문 응답(multi-hop question answering) 문제를 해결하기 위해 ReSP(Reprieval, Summarize, Plan)라는 새로운 반복적 RAG(Retrieval-Augmented Generation) 방법을 제안합니다. 기존 방법들이 반복적인 정보 검색 시 과도한 컨텍스트와 계획 중복 문제를 겪는 것과 달리, ReSP는 이중 기능 요약기를 도입해 검색한 문서에서 얻은 정보를 요약하고 이를 전반적인 질문과 현재 하위 질문에 동시에 적용합니다.

- **Technical Details**: ReSP 프레임워크는 Reasoner, Retriever, Summarizer, Generator 총 4가지 구성 요소로 이루어져 있습니다. 특히 요약기는 검색된 문서의 정보를 요약하여 global evidence memory와 local pathway memory라는 두 가지 개별 메모리 큐(queue)에 저장합니다. 전반적인 질문에 대한 정보를 global evidence memory에 저장해 자원의 중복 사용방지, 충분한 정보가 수집되었을 경우 반복 과정을 막습니다. 현재 하위 질문에 대한 정보는 local pathway memory에 저장해 진행 상태를 명확히 하고 중복된 문자열 계획을 방지합니다.

- **Performance Highlights**: 제안된 ReSP 방법은 HotpotQA와 2WikiMultihopQA 데이터셋에서 기존 최첨단 방법보다 각각 4.1, 5.9 F1 점수 향상을 보였습니다. 특히 ReSP는 다양한 컨텍스트 길이에 대해 뛰어난 견고성을 나타내며, 실험 결과 반복적인 RAG 방법 중에서도 최고의 성능을 기록했습니다.



### AlcLaM: Arabic Dialectal Language Mod (https://arxiv.org/abs/2407.13097)
Comments:
          Accepted by ArabicNLP 2024, presented in ACL 2024

- **What's New**: 이 논문에서는 AlcLaM이라는 새로운 아랍어 방언(language specific PLM) 모델을 소개합니다. AlcLaM은 3.4백만 개의 문장으로 구성된 아랍어 방언 코퍼스를 사용하여 개발되었습니다. 이 코퍼스는 소셜 미디어 플랫폼(YouTube, Facebook)에서 수집되었습니다. 이 모델은 기존 아랍어 PLM들(CAMeL, MARBERT, ArBERT)보다 훨씬 적은 양의 데이터(13GB)로 학습되었음에도 불구하고 뛰어난 성능을 보입니다.

- **Technical Details**: AlcLaM은 BERT 기반의 모델로, 특히 아랍어 방언에 최적화되었습니다. 학습을 위해 3,372,744개의 아랍어 방언 문장으로 구성된 코퍼스를 사용하였으며, 아랍어에서 흔히 사용되지 않는 Modern Standard Arabic(MSA) 토큰을 필터링하기 위해 이진 분류기(MSA-Dialect)를 훈련했습니다. 해당 모델은 CAMeL 모델을 미세 조정하여 98%의 정확도를 달성했습니다.

- **Performance Highlights**: AlcLaM은 세 가지 아랍어 자연어 처리(NLP) 다운스트림 작업에서 탁월한 성능을 입증했습니다: (i) 아랍어 방언 식별(DID), (ii) 감정 분석(SA), (iii) 혐오 발언 및 공격적인 언어 탐지. 적은 양의 학습 데이터에도 불구하고, AlcLaM은 대부분의 데이터셋에서 state-of-the-art 성능을 보이며, 기존의 다중 언어 및 단일 언어 접근 방식을 능가했습니다.



### Dynamic Sentiment Analysis with Local Large Language Models using Majority Voting: A Study on Factors Affecting Restaurant Evaluation (https://arxiv.org/abs/2407.13069)
Comments:
          This manuscript is under peer review

- **What's New**: 이번 연구는 중형 규모의 로컬 LLMs(large language models)를 사용하여 온라인 리뷰 데이터의 감정 분석을 수행할 때, 다수결 투표 메커니즘을 도입해 더 견고한 결과를 얻을 수 있음을 입증합니다. 이는 기존 연구가 대형 모델과 단일 시도를 통해 수행한 것보다 다수결 투표와 복수 시도를 통해 더 많은 변동과 재현성을 확보할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 다수결 투표 메커니즘을 로컬에서 구동되는 LLMs에 적용하여 감정 분석 모델을 구축했습니다. 해당 모델은 사용자 리뷰 데이터를 통해 다양한 측면을 동적으로 분석할 수 있는 Aspect-based Sentiment Analysis (AbSA)을 목표로 하며, 임의의 측면(aspects) 설정이 가능합니다. 특히, 탐사적인 분석을 통해 하이퍼 파라미터와 정확도 사이의 관계를 조사했습니다.

- **Performance Highlights**: 중형 모델을 사용한 반복 접근법이 대형 모델을 단일 시도로 사용하는 방식보다 더 높은 변동성과 신뢰도(robustness)를 보였습니다. 이는 실제 인간 주석자가 다수결 투표를 통해 의견 불일치를 해결하는 방법과 유사한 방식입니다.



### Establishing Knowledge Preference in Language Models (https://arxiv.org/abs/2407.13048)
Comments:
          27 pages, 8 figures, 23 tables, working in progress

- **What's New**: 이 연구는 언어 모델이 다양한 지식 원천들을 효과적으로 통합하는 방법을 체계적으로 연구하고 정의합니다. 연구진은 언어 모델이 내재된 지식(parametric knowledge), 맥락적 지식(contextual knowledge), 사용자 지시 지식(instruction knowledge) 사이의 우선 순위를 설정하는 세 가지 지식 선호 계층 구조를 제안했습니다. 또한, 이러한 지식 선호를 효과적으로 평가하기 위해 기존 데이터셋인 IfQA, MQuAKE 및 MRQA 데이터를 활용하여 벤치마크를 구축하고, 사용자 가정을 포함한 다양한 질문-답변 쌍을 구성하는 데이터셋 합성 방법을 제안했습니다.

- **Technical Details**: 연구에서는 언어 모델이 내재지식 (parametric knowledge), 맥락적 지식(contextual knowledge), 사용자 지시 지식(instruction knowledge) 사이의 우선 순위를 설정하는 문제를 '지식 선호' 문제로 통합하였습니다. 또한, IfQA, MQuAKE, MRQA와 같은 기존 데이터셋을 활용하여 여러 설정(예: 사용자 명세의 유무, 문서의 유무)을 전반적으로 평가하는 벤치마크를 만들었습니다. 데이터셋 합성 방법론을 통해 다양한 질문-답변 쌍을 자동으로 생성하고 이를 활용하여 언어 모델을 미세 조정하였습니다. 이를 통해 언어 모델이 사용자가 지정한 지식 선호 계층 구조를 따르도록 학습시켰습니다.

- **Performance Highlights**: 제안된 방법론에 따라 7B 모델을 미세 조정한 결과, 모든 평가 벤치마크에서 18% 이상의 성능 향상을 달성했습니다. 특히, 오픈 소스 모델이 맥락적 지식이나 사용자 지시를 더 잘 따를 수 있게 되어 복잡한 멀티홉 질문에서도 우수한 성능을 보여 주었습니다. 예를 들어, 반사실 지식 편집(counterfactual knowledge editing) 작업에서 F1 점수가 28.48%에서 89.36%로 상승하였습니다.



### Turkish Delights: a Dataset on Turkish Euphemisms (https://arxiv.org/abs/2407.13040)
Comments:
          In Proceedings of The First SIGTURK workshop co-located with ACL 2024: this https URL

- **What's New**: 이 연구는 자연어 처리(NLP)에서 상대적으로 적게 연구된 터키어 완곡어구(euphemisms)에 대한 새로운 터키어 PET(잠재적 완곡어구) 데이터셋을 소개합니다. 이는 터키어에서 완곡어구를 탐지하기 위한 최초의 자동화된 시도로서, 터키어 완곡어구의 예시 문맥을 수집하고 주석을 달아 PET의 완곡적 및 비완곡적 예시를 제공합니다.

- **Technical Details**: 연구팀은 BERTurk, Electra, XLM-RoBERTa, mBERT 등 다양한 트랜스포머(Transformer) 기반 모델을 터키어 완곡어구 탐지 작업에 맞춰 학습시켰습니다. 평가 메트릭으로는 F1, 정확도(accuracy), 정밀도(precision)를 사용하여 모델의 성능을 비교했습니다. 터키어는 어간에 여러 접사가 붙어 단어가 형성되는 교착어로, 자유로운 어순과 문맥 의존적 의미 변화를 가지며 이는 자동 완곡어구 탐지 작업의 큰 도전 과제로 작용합니다.

- **Performance Highlights**: 다양한 트랜스포머 기반 모델을 평가한 결과, XLM-RoBERTa와 mBERT와 같은 대형 다국어 모델들이 강력한 언어 이해 능력을 제공하였으며, bert-base-turkish-cased와 electra-base-turkish-cased-discriminator는 터키어 표현과 언어 패턴을 더 정교하게 캡처하는데 우수한 성능을 보였습니다. 이는 완곡어구 탐지 작업에도 유효하게 작용하였습니다.



### A Survey of Prompt Engineering Methods in Large Language Models for Different NLP Tasks (https://arxiv.org/abs/2407.12994)
- **What's New**: 최근 AI 연구의 한 분야로 떠오른 프롬프트 엔지니어링(prompt engineering)에 관한 방대한 서베이 논문이 발표되었습니다. 프롬프트 엔지니어링은 대형 언어 모델(LLMs)에서 추가적인 파라미터 재훈련이나 미세 조정 없이 내재된 지식을 끌어내는 방법론을 제시합니다. 논문은 다양한 NLP 작업에 사용된 39가지 프롬프트 기법을 44개의 연구 논문을 통해 분석하였습니다. 기존 연구가 주로 프롬프트 기법의 넓은 카테고리에 집중한 반면, 본 논문은 각각의 NLP 작업에 대한 세부적인 분류를 시도했습니다.

- **Technical Details**: 프롬프트 엔지니어링에서 핵심 기법으로 제시된 몇 가지 방법론을 살펴보면, 1) Zero-shot과 Few-shot 세팅에서의 프롬프트 사용, 2) 기본 프롬프트(Basic prompting), 3) Chain of Thought(CoT) 및 Self-Consistency, 4) 증명 기반 프롬프트(Explainable Reasoning, ER), 5) 자동화 프롬프트(Auto-CoT), 그리고 6) 고난이도 프롬프트(Complex CoT) 등이 포함됩니다. 각 기법들은 NLP 작업 기반으로 분류되어 그 성능을 상세히 평가했습니다.

- **Performance Highlights**: 주요 성능 결과는 다음과 같습니다: Chain of Thought(CoT)는 수학 문제 해결에서는 약 39%, 상식 추론에서는 약 26% 성능 향상을 보였습니다. Self-Consistency 기법은 CoT에 비해 수학 문제 해결에서 평균 11%, 상식 추론에서 3%, 멀티 홉 추론에서 6% 성능 향상을 나타냈습니다. 자동화 프롬프트(Auto-CoT)는 수작업 프롬프트에 비해 동등하거나 더 나은 성능을 보이며, 고난이도 프롬프트(Complex CoT)는 수학 문제 해결, 상식 추론, 테이블 기반 수학 문제 해결, 멀티 홉 추론에서 평균 5.3%, 최대 18% 성능 향상을 보여주었습니다.



### Halu-J: Critique-Based Hallucination Judg (https://arxiv.org/abs/2407.12943)
- **What's New**: 이번 연구에서는 허위 생성(hallucination)을 탐지하는 고도화된 비판 기반 시스템인 Halu-J를 소개합니다. 이 모델은 기존 상당수의 탐지 방법론이 가지고 있던 문제점, 즉 부적절한 설명 부족, 검색 시스템의 결함, 미비한 다중 증거 분석 등을 개선하고자 설계되었습니다. Halu-J는 70억 매개변수(parameter)를 가지며, 비판(Critique)을 통해 허위를 탐지하고 상세한 설명을 제공합니다.

- **Technical Details**: Halu-J의 주요 기술적 진보로는 다중 증거 허위 탐지 데이터셋인 ME-FEVER의 도입, 우선순위 기반 학습 방법(preference-based learning)을 통한 관련 증거 식별 및 비판 생성 품질의 강화 등이 포함됩니다. ME-FEVER 데이터셋은 기존 FEVER 데이터셋에 기반하며, 완전히 무관한 증거, 부분적으로 무관한 증거, 고도로 관련된 증거 등 세 가지 유형으로 구성된 3901개의 인스턴스를 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, Halu-J는 다중 증거 기반 허위 탐지에서 GPT-4o보다 우수한 성능을 보였으며, 단일 증거 기반 탐지에서도 유사한 성능을 나타냈습니다. 또한, Halu-J가 생성한 비판의 품질은 GPT-4o와 비슷한 수준을 유지하며, 관련 증거 일치율에서 가장 높은 점수를 기록했습니다.



### Explainable Biomedical Hypothesis Generation via Retrieval Augmented Generation enabled Large Language Models (https://arxiv.org/abs/2407.12888)
- **What's New**: 최근 발표된 논문에서는 생의학적 정보 통합과 가설 생성을 지원하는 새로운 워크플로우 RUGGED(Retrieval Under Graph-Guided Explainable disease Distinction)을 소개합니다. 이 워크플로우는 대규모 언어 모델(LLMs)의 환각(hallucination) 문제를 해결하기 위해 Retrieval Augmented Generation (RAG)을 사용합니다.

- **Technical Details**: RUGGED는 텍스트 마이닝 연관 분석과 질병 노드에 대한 설명 가능한 그래프 예측 모델을 통해 출판물과 지식 베이스의 관련 생의학 정보를 검토, 통합 및 추출합니다. 이러한 분석과 생의학 텍스트를 RAG 기반 LLM 환경에 통합하여 사용자가 메커니즘을 명확히 설명하고 가설 탐색을 용이하게 합니다.

- **Performance Highlights**: RUGGED의 임상 사용 사례로 Arrhythmogenic Cardiomyopathy (ACM)와 Dilated Cardiomyopathy (DCM)의 치료법 평가 및 추천이 가능함을 보여주었습니다. 특히, 분자 상호작용과 미탐색 활용 방안을 분석하여 LLM의 환각을 최소화하고 새로운 치료제 탐색을 향상시키는 실질적인 인사이트를 제공합니다.



### Whitening Not Recommended for Classification Tasks in LLMs (https://arxiv.org/abs/2407.12886)
- **What's New**: 이 논문은 문장 임베딩(sentence embedding) 품질을 개선하기 위한 '휘팅(whitening)' 기법의 효과가 모델과 과제에 따라 다르다는 점을 발견했습니다. 특히, 분류 작업에서는 오히려 임베딩 품질이 저하됨을 확인했습니다. 또한, SentEval+라는 임베딩 평가 플랫폼을 개발하였습니다.

- **Technical Details**: 휘팅 변환 기법 중 PCA(Principal Component Analysis), ZCA(Zero-Phase Component Analysis), Cholesky 휘팅 등을 실험했습니다. 휘팅은 통계적으로 상관된 비등방성 특징을 상관이 없는 등방성으로 변환하는 과정입니다. 다양한 변종의 휘팅 기법을 실험한 결과, 전반적인 결론은 변하지 않았습니다.

- **Performance Highlights**: 분류 작업에서는 모든 평가 모델과 데이터셋에서 일관되게 휘팅이 임베딩 품질을 저하했습니다. 반면, STS(Semantic Text Similarity) 작업에서는 일부 모델에서 휘팅이 효과적이었습니다. OpenAI와 LLaMA 모델에서 임베딩 성능은 대체로 비슷한 수준을 보였습니다.



### BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieva (https://arxiv.org/abs/2407.12883)
Comments:
          50 pages

- **What's New**: 기존 검색 벤치마크(benchmarks)는 주로 정보 탐색을 위한 쿼리(queries)로 구성되어 있으며, 키워드 또는 의미 기반 검색이 주로 사용됩니다. 그러나 복잡한 실제 쿼리의 경우 표면적 일치(surfaced form matching)를 넘어서는 심층적 추론이 필요합니다. 이를 위해 BRIGHT라는 새로운 텍스트 검색 벤치마크를 도입했습니다. 이 벤치마크는 복잡한 추론이 필요한 쿼리를 기반으로 만들어졌습니다.

- **Technical Details**: BRIGHT는 경제학, 심리학, 로봇공학, 소프트웨어 공학, 지구 과학 등 다양한 분야에서 수집된 1,398개의 실제 쿼리를 기반으로 구성되었습니다. 이를 통해 실제 상황에서 복잡한 검색 요구를 더 잘 반영할 수 있게 했습니다. BRIGHT는 자연스럽게 발생하거나 신중하게 선별된 인간 데이터를 사용하여 구축되었습니다.

- **Performance Highlights**: 최신 검색 모델조차도 BRIGHT에서 성능이 좋지 않았으며, MTEB 리더보드에서 최고 모델이 59.0 nDCG@10의 점수를 기록한 반면, BRIGHT에서는 nDCG@10 18.0을 기록했습니다. 그러나 대형 언어 모델(LLMs)의 Chain-of-Thought 추론을 통해 쿼리를 증강하면 성능이 최대 12.2 포인트 향상될 수 있음을 입증했습니다. 또한, 사전 훈련 중 벤치마크 데이터 누출에 대해 강력한 내성을 가지고 있음을 확인했습니다.



### InstructAV: Instruction Fine-tuning Large Language Models for Authorship Verification (https://arxiv.org/abs/2407.12882)
- **What's New**: InstructAV라는 새로운 접근법이 소개되었습니다. 이는 저자 확인(Authorship Verification, AV) 문제를 다루기 위해 대형 언어 모델(LLMs)과 파라미터 효율화 미세 조정(Parameter-Efficient Fine-Tuning, PEFT) 방법을 결합하여 정확성과 설명 가능성을 동시에 향상시킵니다. 특히, InstructAV는 분류 결정을 투명하고 이해 가능한 설명으로 정렬할 수 있어, AV 분야에서의 큰 진전을 나타냅니다.

- **Technical Details**: InstructAV는 LoRA(Low-Rank Adaptation) 어댑터와 같은 PEFT 방법을 사용하여 LLMs의 파라미터를 효율적으로 미세 조정합니다. 이 방법은 기존 AV 데이터셋에서 분류 레이블과 관련 설명을 수집하고, 이를 결합하여 명령-튜닝 데이터를 생성한 후, LLMs를 정확하고 일관성 있는 설명을 제공할 수 있도록 조율합니다. InstructAV는 IMDB, Twitter, Yelp 리뷰 데이터셋을 사용해 설명의 품질을 엄격히 검사하고, LLMs를 미세 조정하였습니다.

- **Performance Highlights**: 다양한 AV 데이터셋에서 실험을 통해 InstructAV는 높은 분류 정확도와 향상된 설명 신뢰성을 입증했습니다. 이는 정확한 예측과 유용한 설명을 동시에 제공하여 AV 분야에서 모델의 투명성과 설명 가능한 인공지능의 발전에 기여합니다. 자동화된 평가와 인간 평가 결과 모두 InstructAV의 효과를 입증하며, 이 접근법은 향후 연구를 위한 귀중한 자원으로 사용될 수 있습니다.



### BinaryAlign: Word Alignment as Binary Sequence Labeling (https://arxiv.org/abs/2407.12881)
Comments:
          Accepted to ACL 2024

- **What's New**: BinaryAlign이라는 새로운 단어 정렬(Word Alignment) 기법을 제안합니다. 기존 접근법보다 고자원(high-resource) 및 저자원(low-resource) 언어 모두에서 뛰어난 성능을 보여줍니다. 이를 통해 단어 정렬 작업에 단일화된 접근 방식을 제공하며, 비영어 언어 쌍에서도 성능을 분석했습니다. 소스 코드는 공개되어 있습니다.

- **Technical Details**: BinaryAlign 모델은 바이너리 시퀀스 라벨링(Binary Sequence Labeling)을 기반으로 합니다. 이 접근법은 단어 정렬을 각 가능한 단어 쌍에 대해 개별적인 정렬 예측을 하는 바이너리 분류로 재구성합니다. 이는 멀티링구얼 프리트레인드 랭귀지 모델(multilingual pre-trained language model, mPLM)을 사용하며, 서브워드 토큰화(preprocessing)를 적용하여 소스 문장과 타겟 문장을 교차 인코딩(cross-encoding)합니다.

- **Performance Highlights**: BinaryAlign은 제로-샷(zero-shot), 퓨-샷(few-shot), 그리고 완전 지도학습(fully-supervised) 환경에서 모두 기존 접근법보다 우수한 성능을 입증했습니다. 다섯 가지 다른 언어 쌍에 대해 다양한 수준의 감독 하에 테스트되었으며, 모든 경우에서 뛰어난 성과를 보였습니다.



### Large Visual-Language Models Are Also Good Classifiers: A Study of In-Context Multimodal Fake News Detection (https://arxiv.org/abs/2407.12879)
- **What's New**: 최근 연구는 대형 언어 모델(Large Language Models, LLMs)이 가짜 뉴스 검출(Fake News Detection, FND) 작업에서 잘 훈련된 작은 모델(BERT 등)보다 성능이 낮다는 점을 밝혔습니다. 이 논문에서는 두 가지 주요 대형 시각-언어 모델(large visual-language models, LVLMs), CogVLM과 GPT4V, 그리고 더 작은 CLIP 모델을 비교해 LVLMs의 성능을 분석합니다. 특히 in-context learning(ICL)을 통해 LVLMs의 FND 성능을 향상시키는 In-context Multimodal Fake News Detection(IMFND) 프레임워크를 소개합니다.

- **Technical Details**: LVLMs의 가짜 뉴스 검출 성능을 분석하기 위해, 제로샷(Zero-shot) 콘텍스트에서 CogVLM과 GPT4V를 CLIP 모델과 비교했습니다. 다음으로, ICL을 사용하여 LVLMs의 성능을 평가했으며, IMFND 프레임워크를 통해 더 작은 모델(CLIP)에서 예측된 결과와 확률을 활용하여 FND 성능을 더욱 강화했습니다. 이 프레임워크는 특히 높은 확률의 뉴스 세그먼트에 집중하게 하여 분석 정확성을 높입니다.

- **Performance Highlights**: 실험 결과, IMFND 프레임워크는 세 가지 공개된 FND 데이터셋에서 LVLMs의 FND 효율성을 크게 향상시킴을 보여줍니다. 특히 표준 ICL 접근 방식보다 더 높은 정확도를 달성했으며, 이 향상된 성능은 다양한 few-shot 설정에서도 일관되었습니다.



### Do LLMs have Consistent Values? (https://arxiv.org/abs/2407.12878)
Comments:
          10 pages, 5 figures, and there are more in the appendix

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 인간과 유사한 가치 구조를 나타낼 수 있는지 조사하였습니다. 이는 심리학의 가치 구조 연구를 기반으로 하여, LLMs가 생성한 텍스트에서도 인간처럼 가치 순위와 가치 간의 상관관계가 나타나는지 분석했습니다. 특히 'Value Anchoring'이라는 프롬프트 전략을 사용하여 이런 가능성을 증명했습니다.

- **Technical Details**: 연구는 초기 GPT-4와 Gemini Pro를 대상으로 진행되었습니다. 연구진은 LLMs에게 알려진 '초상 가치 설문지'(PVQ-RR)를 통해 한 세션 내에서 다양한 질문에 답하게 하여, 이 답변들 간의 상관관계를 분석했습니다. 기본적인 프롬프트를 사용할 경우 인간과 유사한 가치 시스템을 보여주지 못하는 반면, 'Value Anchor' 프롬프트를 사용하면 인간 데이터와 크게 일치하는 결과를 나타냈습니다.

- **Performance Highlights**: 가장 중요한 발견은 LLMs는 일관된 인간과 같은 가치 시스템을 가진 것으로 간주되지 않지만, 적절한 프롬프트를 사용하면 인간 집단과 유사한 여러 가치 페르소나를 나타낼 수 있다는 점입니다. 또한, Schwartz의 가치 상관 관계 모델과 일치하는 상관 관계를 보여 주었습니다. 본 연구는 GPT-4와 Gemini Pro로 생성된 300개의 페르소나 데이터셋도 제공했습니다.



### Review-Feedback-Reason (ReFeR): A Novel Framework for NLG Evaluation and Reasoning (https://arxiv.org/abs/2407.12877)
Comments:
          Paper Under Review

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 NLG(Natural Language Generation) 출력 품질 평가를 위한 신규 프레임워크인 ReFeR(Review-Feedback-Reason)을 제안합니다. 이 프레임워크는 기존 평가 벤치마크를 약 20% 상회하는 정확성 향상뿐만 아니라, 건설적인 피드백 생성과 집단적 추론 능력 개선을 통해 NLG 평가를 혁신적으로 개선합니다. ReFeR를 사용하여 만든 학습 데이터셋은 Mistral-7B와 같은 작은 모델들을 GPT-3.5와 비슷한 성능을 지닌 평론가로 향상시킵니다.

- **Technical Details**: ReFeR는 학술 논문 심사 과정을 모방한 세 가지 모듈(Review, Feedback, Reason)로 구성되어 있습니다. Peer Review Body 모듈에서 세 명의 LLM 에이전트가 각각 NLG 출력을 독립적으로 평가하고, Critic Module(선택 사항)에서 또 다른 LLM 에이전트가 이 평가들을 재평가하며, 마지막으로 Area Chair 모듈에서 최고 LLM 에이전트가 최종 평가를 합니다. 평가 지침 모듈과 비판 코멘트 모듈을 통해 평가 정확성을 높이고 있습니다.

- **Performance Highlights**: ReFeR는 세 가지 추론 벤치마크에서 GPT-3.5 Turbo 모델 대비 11.67%, GPT-4 모델 대비 1% 높은 성능을 보였습니다. 또한, SummEval과 TopicalChat 벤치마크에서 기존 방식보다 높은 평가 정확도를 기록하였습니다.



### SELF-GUIDE: Better Task-Specific Instruction Following via Self-Synthetic Finetuning (https://arxiv.org/abs/2407.12874)
Comments:
          Accepted by COLM 2024

- **What's New**: SELF-GUIDE는 대형 언어 모델(LLM)이 특정 작업을 잘 수행할 수 있도록 자체적으로 데이터 생성 및 세분화하는 새로운 다단계 메커니즘입니다. 이제 더 강력한 '교사' LLM에 의존하지 않고도 학생 LLM을 자체 학습 데이터로 미세 조정할 수 있습니다.

- **Technical Details**: SELF-GUIDE는 주어진 작업에 대한 입력-출력 쌍을 LLM이 자체적으로 생성한 후, 동일한 모델을 이러한 '자체 생성' 데이터로 미세 조정합니다. 입력 생성은 제공된 예제와 작업 지시를 결합하여 새로운 입력을 생성하며, 여기서 생성된 입력은 품질 여과 과정을 거쳐 선별됩니다. 출력 생성 단계에서는 기존의 인컨텍스트 학습(in-context learning) 기법을 사용하여 모델이 이전 단계에서 생성된 모든 입력에 맞는 출력을 주석으로 달게 합니다. 생성된 데이터 품질 향상을 위해 하이퍼파라미터 튜닝 및 노이즈 필터링 등의 기술을 사용합니다.

- **Performance Highlights**: Natural Instructions V2 벤치마크에서 SELF-GUIDE를 적용한 결과, 생성 작업에서 17.9 포인트, 분류 작업에서 14.6 포인트의 절대 성능 향상을 보고하였습니다. 이는 기존에 동일한 모델을 프롬프트 방식으로 사용할 때와 비교하여 현저한 성능 향상입니다.



### Evaluation of RAG Metrics for Question Answering in the Telecom Domain (https://arxiv.org/abs/2407.12873)
Comments:
          Accepted for publication in ICML 2024 Workshop on Foundation Models in the Wild

- **What's New**: 이번 연구에서는 특수한 도메인에서의 질문 응답(QA) 작업을 수행하는 데 사용되는 Retrieval Augmented Generation (RAG)의 평가를 개선하려는 노력이 소개되었습니다. 특히, RAG 평가를 위한 기존의 RAGAS 라이브러리를 개조하여 중간 출력 값을 제공함으로써 평가 메트릭의 투명성을 높였습니다. 이 연구는 이러한 개선된 RAGAS를 통해 평가된 텔레콤 도메인의 응답들을 분석하고, 도메인 적응 및 파인 튜닝된 LLM이 RAG 메트릭에 미친 영향을 조사했습니다.

- **Technical Details**: RAG 시스템은 다수의 메트릭을 사용하여 생성된 응답의 정확성과 관련성을 평가합니다. 이 논문에서는 믿음성(Faithfulness), 문맥 관련성(Context Relevance), 응답 관련성(Answer Relevance), 응답 유사성(Answer Similarity), 사실 정확성(Factual Correctness), 응답 정확성(Answer Correctness) 등의 메트릭을 개선된 RAGAS 패키지를 통해 평가했습니다. 실험은 텔레콤 도메인의 질의 응답 데이터셋 (TeleQuAD)을 기반으로 수행되었으며, 다양한 LLM 모델(Mistral-7b, GPT-3.5)을 사용했습니다. 문서 정보는 3GPP Release 15 문서에서 추출되었으며, 앙상블 모델과 코사인 유사성을 이용해 최적의 문맥을 선택했습니다.

- **Performance Highlights**: RQ1 분석을 통해, 특정 메트릭이 텔레콤 도메인의 응답 평가에서 특히 유용함을 발견했습니다. 믿음성과 사실 정확성 메트릭은 전문가 평가와 높은 상관관계를 보여 종합적인 응답 정확성 평가에 효과적이었습니다. RQ2에서는 텔레콤 데이터에 도메인 적응된 LLM들이 더 나은 성능을 발휘함을 확인했습니다. RQ3에서는, 도메인 적응된 생성기 LLM이 문맥 오류에 대한 응답 생성 시 믿음성(Faithfulness) 점수가 낮아지는 경향을 보여줬으며, 이는 잘못된 문맥에서 벗어난 정보로부터 응답을 생성할 때 나타나는 현상임을 시사합니다.



### Evaluating Large Language Models with fmeva (https://arxiv.org/abs/2407.12872)
- **What's New**: fmeval은 대규모 언어 모델(LLMs)을 다양한 작업에서 평가할 수 있는 오픈 소스 라이브러리입니다. 이 라이브러리는 사용자들이 모델의 작업 성능뿐만 아니라 책임 있는 AI(Responsible AI) 차원에서도 평가할 수 있도록 도와줍니다. fmeval의 주요 설계 원칙은 단순성, 포괄성, 확장성, 그리고 성능입니다. 이 논문은 fmeval의 설계 원칙을 구현한 과학적 및 공학적 선택 사항들을 강조하며, 사례 연구를 통해 모델 선택과 평가 워크플로우에서의 활용을 보여줍니다. fmeval 라이브러리는 Amazon Bedrock 및 Amazon SageMaker JumpStart에 네이티브 통합되어 있습니다.

- **Technical Details**: fmeval의 설계 원칙은 다음과 같습니다: 

1. **단순성 (Simplicity)** - 사용자들이 책임 있는 AI에 대한 전문가가 되지 않아도 사용 가능하도록, 내장 평가를 통해 쉽게 이해할 수 있는 지표를 제공합니다. 

2. **포괄성 (Coverage)** - 다양한 LLMs에 대한 품질과 책임성을 평가하며, Amazon SageMaker JumpStart 및 Amazon Bedrock 모델과 HuggingFace 등 서드파티 모델을 지원합니다. 

3. **확장성 (Extensibility)** - 사용자 정의 데이터셋 및 평가를 지원함으로써 도메인 특화된 사용 사례를 다룰 수 있습니다. 

4. **성능 (Performance)** - 평가 작업의 속도와 확장성을 보장합니다.

- **Performance Highlights**: fmeval은 다양한 LLM 평가 프레임워크와 차별화됩니다. 예를 들어, HuggingFace Evaluate는 확장성과 단순성의 제한이 있는 반면, OpenAI Evals는 OpenAI 모델의 평가에 주력하여 포괄성이 제한됩니다. fmeval은 넓은 범위의 평가를 제공하며 사용자 정의 기능을 통해 추가 평가 요구를 충족시킵니다. 또한, 자동으로 생성되는 해석 가능한 보고서를 통해 결과를 제공합니다. Ray 프레임워크를 사용하여 분산 및 병렬 처리 기능을 개선하였습니다.



### MetaTool: Facilitating Large Language Models to Master Tools with Meta-task Augmentation (https://arxiv.org/abs/2407.12871)
Comments:
          8 pages, 4 figures

- **What's New**: 새로운 연구는 LLMs(Large Language Models)를 통해 복잡한 도구를 활용하는 방법론을 제안합니다. 기존의 소수 샷 프롬프팅(few-shot prompting)이나 전문가 경로의 미세 조정으로는 복잡한 도구의 사용법을 충분히 학습하기 어렵기 때문에, 본 연구에서는 메타 도구(MetaTool)라는 새로운 학습 방법론을 소개하였습니다. 이를 통해 LLM이 다양한 도구를 이해하고, 작업을 더 효과적으로 수행할 수 있게 합니다.

- **Technical Details**: MetaTool은 자가 지도 데이터 증강 기법(self-supervised data augmentation technique)을 사용하여 LLM이 도구 세트를 종합적으로 이해하도록 합니다. 이 방법은 6개의 메타 태스크(meta-tasks)를 포함하며, 이는 도구 실행의 원인과 기능성을 예측하는 과제를 포함합니다. 이를 통해 고품질의 질문-응답(QA) 데이터를 자동으로 생성할 수 있습니다. 메타 테스크 데이터를 학습 지침 튜닝 과정에 통합하여, MetaTool 모델은 다양한 도구 지향 과제에서 뛰어난 성능을 보입니다.

- **Performance Highlights**: MetaTool은 세 가지 도구 지향 태스크에서 성능을 평가한 결과, 개방형 소스 LLM(e.g., LLaMA-3)보다 성공률이 22.7% 향상되었으며, GPT-4/GPT-3.5-turbo와 비교할 만한 성과를 보였습니다. 이를 통해 MetaTool은 기존의 지침 튜닝 방법들보다 도구 이해도를 상당히 향상시켜 주었음을 확인할 수 있습니다.



### Bilingual Adaptation of Monolingual Foundation Models (https://arxiv.org/abs/2407.12869)
- **What's New**: 본 논문에서는 모노링구얼(단일 언어) 대형 언어 모델(Large Language Model, LLM)을 다른 언어로 적응시키는 효율적인 방법을 제안합니다. 특히 Llama 2 모델을 아랍어로 적응시키는 예를 들어 설명합니다. 이 방법은 임베딩 행렬을 훈련시키고, 이중언어 코퍼스(Bilingual Corpus)를 사용한 지속적인 전이 학습을 통해 모델을 확장합니다. 이를 통해 영어 능력 저하 없이 아랍어 능력을 획득할 수 있다는 것을 보여줍니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성됩니다. 첫 번째 단계는 어휘(vocabulary)를 확장하고 임베딩 매트릭스(embeddings matrix)만을 훈련시키는 것입니다. 두 번째 단계에서는 이중언어 코퍼스를 사용한 지속적인 전이 학습(continual pretraining)을 진행합니다. 이 과정에서 어휘 초기화 기술, 데이터 믹스 비율, 학습률 등에 대한 광범위한 제거 실험을 수행하여 최적의 설정을 도출합니다.

- **Performance Highlights**: 제안된 방법은 아랍어에서의 성능을 크게 향상시켰으며 영어에서도 약간의 향상을 보여주었습니다. 이러한 결과는 비용 효율적인 교차 언어 전이(Cross-lingual transfer)의 가능성을 입증합니다. Llama 2 모델을 기반으로 한 추가 실험 결과, 13B 및 70B 모델에서도 유사한 성능 향상이 확인되었습니다.



### Beyond KV Caching: Shared Attention for Efficient LLMs (https://arxiv.org/abs/2407.12866)
- **What's New**: 새로운 연구에서는 NLP 모델의 주요 도전 과제인 계산 및 메모리 효율성을 개선하기 위해 혁신적인 Shared Attention(SA) 메커니즘을 도입했습니다. 이 방법은 전통적인 주의 메커니즘이 아닌, 사전에 계산된 주의 가중치를 여러 계층에 걸쳐 직접 공유함으로써 자원을 절약합니다.

- **Technical Details**: Shared Attention(SA) 메커니즘은 주의 가중치가 사전 훈련 후 여러 계층에 걸쳐 거의 동일하다는 점에 착안해, 이를 여러 계층에서 직접 공유하도록 설계되었습니다. 이 접근방법은 기존의 KV-캐시를 계층 사이에서 공유하는 방식과는 다르게 동작하며, 주의 가중치의 분포가 일정함을 이용합니다. 이를 통해, softmax 연산을 반복할 필요 없이 단일 주의 행렬을 유지함으로써 계산 복잡성을 줄입니다.

- **Performance Highlights**: 실제 적용 실험에서, 다양한 LLM(Large Language Model)에 Shared Attention을 사용한 결과, 표준 벤치마크에서 정확도 손실이 거의 없는 것이 확인되었습니다. 또한 이러한 접근법은 모델이 자원 제한 환경에서 더 효율적으로 작동할 수 있도록 도와줍니다.



### GRAD-SUM: Leveraging Gradient Summarization for Optimal Prompt Engineering (https://arxiv.org/abs/2407.12865)
Comments:
          15 pages, 2 figures

- **What's New**: LLM(대형 언어 모델)을 위한 프롬프트 엔지니어링(prompt engineering)은 고품질 출력을 보장하기 위해 반복적으로 프롬프트(prompt)를 생성, 평가 및 수정하는 수동으로 시간 소모적인 과정입니다. 이 논문에서는 이러한 프롬프트 엔지니어링을 자동화하기 위한 새로운 방법인 GRAD-SUM을 소개합니다. GRAD-SUM은 사용자 정의 작업 설명 및 평가 기준을 통합하며, 대부분의 기존 방법보다 뛰어난 성능을 보입니다.

- **Technical Details**: GRAD-SUM은 사용자 정의 작업 설명과 평가 기준을 활용하여 프롬프트를 자동으로 최적화하는 방법으로, gradient-based optimization(기울기 기반 최적화) 기술을 사용합니다. 주요 구성 요소로는 생성(generation), 평가(evaluation), 기울기 생성(gradient generation), 기울기 요약(gradient summarization) 및 프롬프트 편집(promotion editing) 모듈이 포함됩니다. GRAD-SUM은 자연어 오류 설명과 시스템 응답 개선을 위한 제안을 바탕으로 프롬프트를 수정하는 데 중점을 둡니다.

- **Performance Highlights**: GRAD-SUM은 다양한 벤치마크에서 기존 방법을 꾸준히 능가하며 자동 프롬프트 최적화에 있어 높은 유연성과 효과를 보여줍니다. 특히, GRAD-SUM은 대규모 데이터셋에서도 효과적인 일반화가 가능하도록 합니다. 기존의 Monte Carlo search와 같은 방법에 비해 비용 효율성이 높으며, 사용자가 정의한 평가 기준을 기반으로 즉각적으로 피드백을 받아들여 프롬프트를 개선합니다.



### Token-Supervised Value Models for Enhancing Mathematical Reasoning Capabilities of Large Language Models (https://arxiv.org/abs/2407.12863)
- **What's New**: 새로운 연구에서는 수학 문제 해결을 위한 Large Language Models(LLMs)의 문제를 해결하기 위해 Token-Supervised Value Model(TVM)을 제안했습니다. TVM은 기존의 방법과 달리 각 토큰(token) 단위로 기대되는 축적 보상(expected cumulative reward)를 통해 검증을 실시합니다. 이를 통해 중간 추론 경로가 올바른 최종 정답에 도달할 가능성을 평가할 수 있게 합니다.

- **Technical Details**: TVM은 토큰 수준의 감독(supervision)과 기대되는 축적 보상(value)을 적용한 새로운 훈련 방식을 사용합니다. 기존의 검증 모델(ORMs와 PRMs)은 전체 추론 경로나 단계의 정확성을 평가하는데 중점을 두었지만, TVM은 추론 경로 내에서 각 토큰의 세부적인 변별력을 캡처할 수 있습니다. 또한, 각 토큰의 값이 올바른 최종 답변에 도달할 확률과 같다는 이론적 인사이트를 제공하며, 이를 통해 샘플링한 추론 경로를 따라 실제 값을 추정하여 각 토큰에 라벨을 할당합니다.

- **Performance Highlights**: 실험 결과에 따르면, TVM은 GSM8K와 MATH 벤치마크에서 Mistral과 Llama를 사용한 기존의 step-by-step verifiers보다 뛰어난 성능을 보여줍니다. 특히 10억 개 이하 파라미터(under 10B parameters)를 가진 LLMs에서도 성능 향상을 확인할 수 있었습니다.



### Analyzing Large language models chatbots: An experimental approach using a probability tes (https://arxiv.org/abs/2407.12862)
Comments:
          17 pages, 3 figures, Submitted to ACM Transactions on Intelligent systems and Technology

- **What's New**: 이 연구는 두 가지 서로 다른 대규모 언어 모델(Large Language Models, LLMs) 챗봇인 ChatGPT와 Gemini를 사용한 탐색 테스트를 통해 시행된 질적 경험적 연구입니다. 이 테스트는 인지 심리학에서 널리 알려진 'Linda 문제'와 이 실험을 위해 새로 개발된 'Mary 문제'를 기초로 하여 설계되었습니다. 연구의 목적은 각 챗봇이 제공한 출력을 분석하여 확률 이론에 맞는 논리적 추론을 주로 사용하는지 아니면 프롬프트의 고정 관념적 텍스트 설명에 더 자주 영향을 받는지를 검증하는 것입니다.

- **Technical Details**: 방법론적 절차는 확률 질문으로 설계된 프롬프트를 기반으로 한 탐색 테스트로 구성되었습니다. 분석 대상은 각 챗봇 상호작용에서 제공한 출력 데이터셋입니다. 챗봇의 출력에 대해 논리적 추론과 텍스트 구성 방식을 다루는 접근 방식을 조사했습니다.

- **Performance Highlights**: 연구 결과는 분석된 챗봇들이 잘 알려진 확률 문제에서는 만족스러운 성능을 보였지만, 확률적 논리의 직접적인 적용을 요구하는 새로운 테스트에서는 상당히 낮은 성능을 보이는 것으로 나타났습니다.



### CiteME: Can Language Models Accurately Cite Scientific Claims? (https://arxiv.org/abs/2407.12861)
- **What's New**: 매달 수천 건의 새로운 논문이 발표되면서 정보 과부하로 인해 최신 연구 동향을 따라잡고 아이디어 출처를 정확히 확인하는 것이 어려워졌습니다. 이에 대응해, 연구 텍스트에서 논문을 정확히 식별할 수 있는 언어모델(Language Model, LM)을 연구 보조 도구로 사용할 수 있는지에 대한 질문을 제기했습니다. 이를 위해 'CiteME'라는 새로운 벤치마크를 구축했으며, 이는 최근 기계학습 논문에서 특정 논문을 참조하는 텍스트 조각으로 구성됩니다.

- **Technical Details**: CiteME는 수동으로 큐레이션된 최초의 인용 어트리뷰션(citation attribution) 벤치마크로, 이러한 문제를 해결하기 위해 애매하지 않은 텍스트 조각만 사용합니다. 이 벤치마크는 인용 어트리뷰션 테스트를 통해 언어 모델의 성능을 평가하며, 인간 평가자는 69.7%의 정확도를 기록한 반면, 현재 최신 언어 모델은 4.2-18.5%의 정확도를 보였습니다.

- **Performance Highlights**: 새로운 자율 시스템인 'CiteAgent'를 도입하여 GPT-4o 언어 모델(GPT-4o LM)을 기반으로 논문을 검색하고 읽을 수 있게 했습니다. 이 시스템은 CiteME 벤치마크에서 35.3%의 정확도를 기록하며, 실제 연구 보조 도구로서의 가능성을 보여줍니다. 이는 기존 모델보다 크게 향상된 성능이며, 향후 연구에서는 이러한 시스템이 모든 언어 모델의 주장을 자동으로 검증할 수 있는 미래를 위한 기초를 마련할 수 있을 것입니다.



### STAGE: Simplified Text-Attributed Graph Embeddings Using Pre-trained LLMs (https://arxiv.org/abs/2407.12860)
- **What's New**: 이번에 소개할 논문에서는 Simplified Text-Attributed Graph Embeddings (STAGE)라는 새로운 기법을 제안합니다. 이는 Text-Attributed Graphs(TAGs)에서의 노드 특징을 향상시키기 위해 만들어진 방법으로, 대형 언어 모델(Large-Language Models, LLMs)을 활용하여 텍스트 속성을 임베딩합니다. STAGE는 현재의 최신 기법(state-of-the-art, SoTA)들에 비해 구현이 간단하면서도 경쟁력 있는 성능을 자랑합니다. 또한, 여러 학술 벤치마크에서 우수한 성과를 보이면서 여러 데이터 준비 및 트레이닝 단계를 단순화하였습니다.

- **Technical Details**: STAGE는 단일 사전 훈련된 LLM을 사용하여 노드 임베딩 모델로서 작동합니다. 데이터 추가 생성을 위한 별도의 프롬프트 없이 모델을 그대로 사용합니다. 또한, diffusion-pattern GNN을 구현하여 더 큰 그래프에도 적용할 수 있도록 했습니다. 이를 통해 학습 시간 및 데이터 준비의 복잡성을 크게 줄였습니다. 주요 구성 요소로는 단일 트레이닝 단계, LLM 프롬프트 없이 텍스트 특징 사용, off-the-shelf LLM을 텍스트 임베딩 모델로 사용, 그리고 확장 가능한 diffusion-pattern GNN 구현이 있습니다.

- **Performance Highlights**: STAGE는 학술 벤치마크에서 다양한 노드 분류 작업에서 경쟁력 있는 성능을 보여주었습니다. 특히, 별도의 데이터 증강 또는 프롬프트 단계 없이 단일 LLM을 사용하여 성능을 달성한 점이 특징입니다. 또한, 다양한 GNN 아키텍처에서 앙상블 트레이닝을 통해 정확도를 높였습니다. 이러한 접근법은 기존의 복잡하고 자원이 많이 드는 방법들보다 훨씬 효율적이고 간단합니다. 코드는 https://github.com/aaronzo/STAGE에서 확인할 수 있습니다.



### Automated Question Generation on Tabular Data for Conversational Data Exploration (https://arxiv.org/abs/2407.12859)
- **What's New**: 이번 연구에서는 비기술 사용자(non-technical user)가 데이터셋을 쉽게 탐색할 수 있도록 돕는 새로운 대화형 데이터 탐색 시스템(conversational data exploration system)을 제안합니다. 이 시스템은 데이터셋의 관련 슬라이스(slices)에 기반하여 자연어로 흥미로운 질문을 추천합니다.

- **Technical Details**: 구체적으로, 데이터셋이 주어졌을 때, 흥미로운 칼럼(columns)을 선택하고, 몇 가지 흥미로움 측정(interestingness measures)에 기반하여 그러한 칼럼 및 칼럼 조합의 흥미로운 슬라이스를 식별합니다. 이 시스템은 미세 조정된(pre-trained) 언어 모델(T5)을 변형하여 특정 방식으로 자연어 질문을 생성하고, 생성된 질문에 값을 채워서 추천 순위를 매긴 후 사용자에게 제공합니다.

- **Performance Highlights**: 이 시스템의 유용성을 다양한 실제(real) 데이터셋을 사용하여 대화형 설정(conversational setting)에서 시연하였습니다.



### Grounding and Evaluation for Large Language Models: Practical Challenges and Lessons Learned (Survey) (https://arxiv.org/abs/2407.12858)
Comments:
          Survey Article for the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2024) Tutorial

- **What's New**: KDD 2024 튜토리얼을 동반한 서베이 기사에서 인공지능(AI) 기술의 급격한 채택과 더불어 이러한 시스템의 신뢰성, 안전성, 관찰 가능성을 보장하는 것이 중요함을 강조합니다. 특히, 대형 언어 모델(LLMs)과 기타 생성적 AI 모델이 직면하는 문제점을 논의합니다.

- **Technical Details**: 이 튜토리얼에서는 LLMs와 생성적 AI 시스템의 잠재적 위험을 식별, 측정, 완화하는 프레임워크에 따라 다루며, 모델, 안전 시스템, 애플리케이션, 배치 레벨에서의 네 가지 완화 레이어를 포함합니다. 또한 구체적 해결책으로는 retrieval augmented generation, constrained decoding, guardrails 등이 있습니다. 

- **Performance Highlights**: LLMs의 평가에는 진실성, 안전성과 정렬, 편향과 공정성, 견고성과 보안, 프라이버시 보호, 저작권 침해, 캘리브레이션과 자신감, 투명성과 인과 개입 등을 포함한 다양한 측정 항목이 존재합니다.



### Automated Peer Reviewing in Paper SEA: Standardization, Evaluation, and Analysis (https://arxiv.org/abs/2407.12857)
- **What's New**: 최근 과학 논문의 급증으로 전통적인 리뷰 메커니즘이 과부하되어 출판물의 품질이 상이해지는 문제가 발생했습니다. 이를 해결하기 위해, 자동화된 논문 리뷰 프레임워크인 SEA를 소개합니다. SEA는 표준화, 평가, 분석이라는 세 가지 모듈로 구성되어 있으며, 각각 SEA-S, SEA-E, SEA-A 모델로 나타냅니다.

- **Technical Details**: SEA는 처음에 GPT-4을 사용해 여러 리뷰를 하나의 통일된 형식과 기준으로 통합하여 표준화합니다(S 모듈). 이후, 표준화된 데이터를 이용해 모델을 미세 조정하여 더 높은 품질의 리뷰를 생성합니다(E 모듈). 마지막으로, 논문의 내용과 리뷰 간의 일관성을 평가하기 위해 mismatch score라는 새로운 평가 메트릭을 도입하고, 자동 수정 전략을 활용해 일관성을 향상시킵니다(A 모듈).

- **Performance Highlights**: 8개의 다양한 데이터셋에서 수행한 실험 결과, SEA 프레임워크가 기존 방법들보다 논문의 품질, 포괄성, 일관성 면에서 뛰어난 리뷰를 생성하는 것으로 나타났습니다. 이 프레임워크는 저자들이 논문의 품질을 개선할 수 있는 유용한 인사이트를 제공합니다.



### AI AI Bias: Large Language Models Favor Their Own Generated Conten (https://arxiv.org/abs/2407.12856)
Comments:
          8 pages, 1 figure

- **What's New**: 최근 연구는 대형 언어 모델(LLM)이 인간이 작성한 텍스트보다 LLM이 생성한 텍스트를 선호하는 경향이 있는지를 조사했습니다. 연구 결과, LLM이 일관되게 LLM 생성 콘텐츠를 선호하는 것으로 나타나 인간에 대한 암묵적인 차별 가능성을 시사합니다.

- **Technical Details**: 이번 연구는 GPT-3.5 터보와 GPT-4 터보를 포함하여 광범위하게 사용되는 LLM을 대상으로 이진 선택 실험을 수행했습니다. 두 개의 실험이 있었는데, 하나는 상품 선택, 다른 하나는 학술 논문 선택입니다. 실험은 동일한 조건 하에 인간과 LLM이 작성한 설명을 비교했습니다. 각 LLM 모델은 각각 쌍으로 구성된 텍스트를 평가하고 선호도를 결정하였습니다.

- **Performance Highlights**: 실험 결과, LLM은 일관되게 LLM 생성 텍스트를 선호하였습니다. 이는 AI 시스템이 인간을 암묵적으로 차별할 가능성을 시사하며, 특히 강력한 AI 통합을 가진 시장에서 인간 근로자를 불공정하게 주변화할 수 있습니다.



### Large Language Models can impersonate politicians and other public figures (https://arxiv.org/abs/2407.12855)
Comments:
          Under review

- **What's New**: 현대 AI 기술, 특히 대형 언어 모델(LLM)이 만들어낸 콘텐츠는 공공 정보 영역을 오염시킬 가능성이 있으며, 이는 사회의 결속력에 큰 위협이 될 수 있습니다. 연구에 따르면 LLM은 인상적인 품질의 텍스트 생성이 가능하여 설득력 있는 정치적 연설, 정해진 스타일의 텍스트, 특정 역할에 맞춘 콘텐츠 등을 만들어낼 수 있습니다. 하지만 LLM이 정치 및 사회적 대표자를 흉내 내는 능력과 일반 대중이 이들의 진정성, 적절성 및 일관성을 어떻게 평가하는지에 대한 대규모 체계적인 연구는 부족합니다. 이 연구는 영국 사회를 대상으로 LLM이 방송 정치 토론 프로그램의 질문에 대한 응답을 생성하고, 이들 응답이 실제 인물이 한 응답보다 더 진정성 있고 적절하게 평가된다는 것을 보여줍니다. 이는 LLM이 공공 정치 토론에 의미 있게 기여할 수 있으며, 이에 따른 사회적 위해 가능성을 대중에게 알릴 필요가 있음을 보여줍니다.

- **Technical Details**: 연구는 영국 사회를 대표하는 단면을 기반으로 하여, LLM이 실제 인물을 흉내 내어 영국 정치 토론 프로그램에서 나오는 질문에 대한 응답을 생성하도록 했습니다. 연구 참가자들(n=948)은 인물의 실제 응답과 LLM이 흉내 낸 응답을 진정성(authenticity), 일관성(coherence), 적절성(relevance) 측면에서 평가했습니다. 연구 질문과 응답은 2020년부터 2022년까지 BBC의 'Question Time'에서 가져온 것으로, 정치인, 사업가, 언론인, 의료 전문가, 작가 등을 포함한 6개 범주에 속한 공공 인물들의 데이터를 사용했습니다.

- **Performance Highlights**: 연구 결과, LLM이 생성한 흉내낸 콘텐츠는 실제 응답보다 더 진정성 있고, 일관적이며 적절하다고 평가되었습니다. 진정성(d=0.66), 일관성(d=1.25), 적절성(d=1.23) 측면에서 중간에서 큰 효과 크기를 나타냈습니다. 참가자들이 실제 응답과 흉내낸 응답을 직접 비교했을 때도 적절성(d=0.84)과 일관성(d=1.04)에서 큰 효과 크기를 보였으며, 진정성(d=0.22)에서도 차이가 있었지만 줄어들었습니다. 이는 LLM이 생성한 흉내낸 응답이 실제 인물의 응답보다 더 진정성 있게 받아들여질 수 있음을 시사합니다.



### Scaling Retrieval-Based Language Models with a Trillion-Token Datastor (https://arxiv.org/abs/2407.12854)
- **What's New**: 이 논문은 사전 학습된 언어 모델(LM)의 효율성과 성능 간의 트레이드오프를 예측하기 위해 학습 데이터와 파라미터 수에 대한 스케일링 법칙을 적용합니다. 이 연구는 추론 시점에서 사용되는 데이터 저장소(datastore)의 크기를 확장하는 것이 언어 모델링과 여러 다운스트림 작업에서 성능을 개선할 수 있음을 발견했습니다. MassiveDS라고 불리는 1.4조 토큰 규모의 데이터 저장소를 구축하고 이를 활용하여 소형 모델이 대형 모델을 능가할 수 있는지를 분석했습니다.

- **Technical Details**: MassiveDS는 일반 웹 데이터와 도메인 특화 데이터를 포함한 데이터 저장소로, 데이터 필터링, 서브샘플링(subsampling), 인덱싱, 문서 검색 등 다양한 데이터 처리 작업을 통해 구축되었습니다. 효율성을 높이기 위해 기존 파이프라인보다 컴퓨팅 비용을 10배 줄이는 효율적인 데이터 저장소 구축 파이프라인을 설계했습니다. 논문에서는 다른 저장소, 모델, 사전 학습 데이터 크기를 다양하게 조합하여 데이터 저장소 확장이 모델 성능에 미치는 영향을 체계적으로 평가합니다.

- **Performance Highlights**: 데이터 저장소 크기를 증가시키면 언어 모델링과 일부 다운스트림 작업이 일관되게 개선됩니다. 특히 지식 집약적인 작업에서 작은 리트리벌 기반 LM(retrieval-based LM)은 대형 LM-only 모델보다 우수한 성능을 보입니다. 또한, 동일한 학습 비용으로 대형 데이터 저장소를 활용한 리트리벌 기반 LMs는 더 나은 컴퓨팅 최적화를 달성할 수 있습니다. 전반적으로 데이터 저장소의 크기는 LM 효율성과 성능 트레이드오프의 필수 요소로 고려되어야 합니다.



### Automated Justification Production for Claim Veracity in Fact Checking: A Survey on Architectures and Approaches (https://arxiv.org/abs/2407.12853)
Comments:
          Accepted to ACL 2024 Main Conference

- **What's New**: 최근 자동 사실 확인(AFC) 분야의 연구들은 주장의 진실성을 예측하는 것 외에도 그 결과를 설명하는 자동화된 정당화 생성에 중점을 두고 있습니다. 이 논문은 이러한 최신 방법론들을 조사하고, 포괄적인 분류 체계를 제안하여 연구의 발전을 제시합니다.

- **Technical Details**: AFC는 여러 단계로 구성된 파이프라인 프로세스를 가지며, 그 중 하나가 정당화 생성 단계입니다. 이 단계에서는 신뢰할 수 있는 출처에서 관련 데이터를 가져와 주장의 진실성을 예측하고, 그 결과를 설명하는 정당화를 제공합니다. 주장의 진실성 예측은 증거와의 일치 정도에 따라 결정됩니다.

- **Performance Highlights**: 정당화 생성 방법론은 주로 다양한 기술 접근 방식을 사용하여 평가됩니다. 여기에는 주목 기반(attention-based), 지식 그래프 기반(knowledge graph-based), 요약 기반(summarization-based), 다중 단계 기반(multi-hop-based), 대형 언어 모델(LLM) 기반 접근 방식이 포함됩니다. 특히, Transformer 기반 아키텍처와 LLM의 발전 덕분에 최근 큰 진전이 있었습니다.



### Historical Ink: Semantic Shift Detection for 19th Century Spanish (https://arxiv.org/abs/2407.12852)
- **What's New**: 이번 논문은 19세기 스페인어 텍스트의 단어 의미 변화에 대한 연구로, 특히 라틴 아메리카 스페인어에 중점을 두고 있습니다. 이 연구는 단어 의미 변화 탐지(Semantic Shift Detection, SSD) 작업을 통해 역사적 맥락에서 언어 진화를 이해하는 데 도움을 줍니다. 연구진은 이를 위해 19세기 스페인어 코퍼스를 구축하고, 다양한 상황에서 SSD 작업을 수행할 수 있는 커스터마이징 가능한 파이프라인을 개발했습니다.

- **Technical Details**: 연구에서는 미세 조정된 BERT(Bidirectional Encoder Representations from Transformers) 모델을 사용해 두 개의 코퍼스 간 단어 의미 변화를 측정합니다. 연구진은 1800년부터 1914년까지의 스페인어 텍스트 자료를 모아 코퍼스를 만들고, 'EUBookShop' 코퍼스를 현대 스페인어 자료로 사용하여 비교합니다.

- **Performance Highlights**: 이 연구는 코퍼스를 통해 19세기와 현대 스페인어 간의 의미 변화를 성공적으로 측정하였으며, 이는 언어 변화가 문화적, 사회적 변화와 어떻게 연관되는지를 이해하는 데 중요한 통찰을 제공합니다. 또한, 스페인어에 특화된 SSD 작업을 통해 더 정확한 다이어크로닉(diachronic) 분석을 수행할 수 있음을 입증합니다.



### ISPO: An Integrated Ontology of Symptom Phenotypes for Semantic Integration of Traditional Chinese Medical Data (https://arxiv.org/abs/2407.12851)
Comments:
          39 pages, 6 figures, 6 tables

- **What's New**: 이 연구에서는 'Integrated Ontology of Symptom Phenotypes (ISPO)'를 개발하여 중국 전통 의학(TCM) 분야의 전자 의료 기록(Electronic Medical Records, EMRs) 및 실제 데이터를 분석하기 쉽게 하였습니다. 이는 다양한 증상 용어의 통합된 온톨로지를 제공하여 의료 데이터 분석 및 지식 공유를 촉진하는 것을 목표로 하고 있습니다.

- **Technical Details**: ISPO를 구축하기 위해, 고전 TCM 교과서와 대규모 중국 전자 의료 기록(EMRs)을 수집하고 의료 텍스트 주석 시스템을 활용해 증상 용어를 수집하고 수동으로 주석을 달았습니다. 또한, 공용 생의학 용어집과의 매핑 작업을 통해 중국어 및 영어 용어 간의 의미적 상호 운용성을 확보하였습니다. 테스트를 통해 78,696개의 입원 사례, 5개의 생의학 용어집, 21권의 TCM 서적 및 사전을 통합하였습니다.

- **Performance Highlights**: ISPO는 3,147개의 개념, 23,475개의 용어, 55,552개의 정의 또는 문맥 텍스트를 제공하며, 12개의 상위 카테고리와 79개의 중간 수준 하위 카테고리를 포함합니다. 추가적으로 독립적인 임상 데이터 세트에서 0.5% 이상의 발생률을 지닌 증상 용어의 경우, ISPO는 95.35%, 98.53%, 92.66%의 커버리지율을 보이며 임상 용어를 온톨로지로 매핑하는 데 유의미한 가치를 입증했습니다.



### Limits to Predicting Online Speech Using Large Language Models (https://arxiv.org/abs/2407.12850)
- **What's New**: 본 연구는 소셜 미디어에서 사용자의 온라인 발화를 예측하는 문제를 고찰하며, 특히 사용자의 개인 게시물 외부 정보가 예측력을 향상시킬 수 있는지 여부를 탐구한다. 최근 연구에 따르면 사용자의 동료들이 작성한 게시물에 담긴 예측 정보가 사용자의 자체 게시물보다 뛰어날 수 있다고 제시된다. 이를 테스트하기 위해, 우리는 큰 언어 모델(large language models)의 성공에 영감을 받아 실증적으로 이 가설을 검증하였다.

- **Technical Details**: 연구에서는 모델의 불확실성을 부정적 로그 가능성(negative log-likelihood)으로 정의하여, 예측 불가능성을 측정하였다. 이를 위해 5,000명 이상의 X(이전의 Twitter) 사용자 및 그 동료로부터 625만 개의 게시물을 수집하였다. 세 가지 큰 언어 모델(GPT-2-XL-1.5B, Falcon-40B, Llama-2-70B)을 통해 예측 실험을 수행하였다.

- **Performance Highlights**: 실험 결과, 동료의 게시물로 사용자의 게시물을 예측하는 성능은 저조하였다. 사용자의 자체 게시물이 예측 가치가 가장 높았으며, 동료의 게시물보다 훨씬 우수했다. 전반적으로 소셜 미디어 게시물의 예측 가능성은 낮았으며, 이는 금융 뉴스 예측과 유사한 수준이었다. 특히, 해시태그와 @-멘션이 게시물의 주요 불확실성 원인으로 나타났으며, 이를 제거하면 예측 가능성이 향상되었다.



### Applicability of Large Language Models and Generative Models for Legal Case Judgement Summarization (https://arxiv.org/abs/2407.12848)
Comments:
          Accepted at Artificial Intelligence and Law, 2024

- **What's New**: 법적 사건 판결의 자동 요약이 최근 생성적 요약 모델과 대형 언어 모델(LLM, Large Language Models)을 통해 시도되고 있습니다. 이 연구에서는 영국 대법원과 인도 대법원의 판결문을 대상으로 이들 모델의 성능을 평가했습니다. 특히 LLM과 추출적 요약 모델과의 성능 비교 뿐만 아니라, 미국 정부 보고서와 같은 다른 유형의 법적 문서에서도 실험을 진행했습니다.

- **Technical Details**: 연구에서는 Legal-LED와 Legal-Pegasus와 같은 법률 도메인에 특화된 생성적 요약 모델과 ChatGPT, Davinci와 같은 일반 도메인의 LLM을 사용했습니다. 요약 품질 평가는 ROUGE, METEOR, BERTScore와 같은 전통적인 메트릭을 사용해 수행되었습니다. 또한, 요약문의 일관성을 평가하기 위해 SummaC, NEPrec, Numprec 등의 메트릭도 사용되었습니다.

- **Performance Highlights**: 일반적으로 생성적 요약 모델과 LLM이 추출적 요약 방법보다 더 나은 성과를 보였으나, 생성된 요약문에 일관성 문제와 허위 정보가 포함되는 경우가 많았습니다. 이를 해결하기 위해 도메인 별 미세조정, 적절한 프롬프트 시도, 의미적 유사성 기반 접근법 등을 사용해 일관성 문제를 줄이는 방안을 모색했습니다. 연구는 법률 학생들이 수행한 요약문의 인간 평가도 포함하고 있습니다.



### Aligning Model Evaluations with Human Preferences: Mitigating Token Count Bias in Language Model Assessments (https://arxiv.org/abs/2407.12847)
- **What's New**: 이 연구는 On-device 소형 언어 모델(SLM)이 API 기반 대형 언어 모델(LLM)인 OpenAI의 GPT-4와 유사한 성능을 제공할 수 있으며, 비용 효율적이라는 사실을 보여줍니다. 기존 연구에서 제기된 문제점인 인간 선호도와 자동 평가기 사이의 불일치를 해결하고자, 이 논문은 자동 평가기인 GPTScorer의 편향을 Bayesian 통계와 t-test를 활용해 재조정하는 방법을 탐구합니다.

- **Technical Details**: 자동 평가기의 편향을 정량화하기 위해 Bayesian 통계와 t-test를 사용했습니다. 사람의 평가 데이터와 비교하여 GPTScorer를 재조정함으로써, 인간 평가와의 일치를 높였습니다. 재조정 과정에서 여러 사용 사례에 대해 인간 평가와의 순위 상관점수가 개선되었습니다. 예를 들어, Recommendation 사용 사례에서 스피어만 순위 상관점수가 -27.27에서 44.55로 향상되었습니다.

- **Performance Highlights**: 재조정된 GPTScorer의 성능 향상은 여러 사용 사례에서 입증되었습니다. 예를 들어, Recommendation 사용 사례에서 순위 상관점수가 크게 향상되었고, 이는 보다 정확하고 공정한 모델 평가를 가능하게 합니다. 이러한 재조정 과정은 AI 모델들이 인간의 가치와 기대에 더 잘 부합하도록 하여 신뢰성을 높입니다.



### Identifying the Source of Generation for Large Language Models (https://arxiv.org/abs/2407.12846)
Comments:
          ICPRAI 2024

- **What's New**: 이번 논문은 토큰 수준 소스 식별(token-level source identification)을 디코딩 과정에서 도입하여, 생성된 콘텐츠의 출처 문서와 매핑하는 방법을 소개합니다. 이는 LLM이 생성한 콘텐츠의 신뢰성을 높이고, 사실성 및 개인정보 침해 문제를 해결하는 데 도움을 줄 수 있습니다.

- **Technical Details**: 토큰 수준 소스 식별을 위해 다층 퍼셉트론(MLP)을 제안하며, 연속된 두 개의 토큰 표현을 입력으로 받아 일반화를 향상시키는 바이그램(bi-gram) 소스 식별기를 사용합니다. 다양한 LLM(Pythia, OPT, Llama2)과 레이어 위치, 식별자 크기에서 실험을 수행하였습니다. 데이터셋은 Wikipedia와 PG19를 사용하였으며, 문서 라벨을 예측하기 위해 바이너리 크로스 엔트로피 손실을 사용합니다.

- **Performance Highlights**: 실험 결과, 토큰 수준 소스 식별자가 문서를 추적하는 데 가능성을 보여주었으며, LLM의 안전한 사용을 위해 유용하다는 것을 확인했습니다. 다양한 크기와 구조의 LLM을 활용한 결과, 일반화가 가능한 소스 태깅 역시 성능을 발휘했습니다.



### $\texttt{metabench}$ -- A Sparse Benchmark to Measure General Ability in Large Language Models (https://arxiv.org/abs/2407.12844)
Comments:
          LLMs, benchmarking, IRT, information, compression

- **What's New**: 최근 논문에서는 대규모 언어 모델(LLMs)의 능력을 효율적으로 평가하기 위한 새로운 벤치마크 시스템을 제안합니다. 이 논문에서 소개된 'metabench'는 ARC, GSM8K, HellaSwag, MMLU, TruthfulQA, WinoGrande의 6가지 주요 벤치마크를 분석하여, 이들이 실제로 측정하는 능력을 압축하여 3% 이하의 크기로 줄여 냈습니다. 이를 통해 원래의 개별 벤치마크 점수와 총 점수를 거의 정확하게 재구성하는 새로운 효율적인 평가 방법을 만들었습니다.

- **Technical Details**: 메타벤치(metabench)는 아이템 반응 이론(Item Response Theory, IRT)과 같은 심리측정학적 기법을 활용하여, 벤치마크 아이템들 중 가장 정보를 많이 담고 있는 아이템들을 선별하였습니다. 이러한 접근 방식을 통해, 원래의 벤치마크 크기의 3% 미만으로 축소된 상태에서도 LLMs의 능력을 정확히 평가할 수 있게 되었습니다. 데이터는 Hugging Face Datasets에서 공개적으로 제공된 6875개의 LLM 아이템 정확도 데이터에서 수집되었으며, 총 5055개의 LLM이 각 6개의 벤치마크에 대해 분석되었습니다.

- **Performance Highlights**: 새롭게 도출된 메타벤치(metabench)는 원래 벤치마크 점수를 재구성할 때 평균적으로 1.5%의 RMSE(root mean square error)를 나타내며, 총 점수를 재구성할 때는 0.8% RMSE를 보입니다. 또한 이 방법은 Spearman 상관계수(r = 0.93)로 측정된, LLMs의 총 점수와 매우 높은 상관관계를 나타내는 공통 요인을 밝혀냈습니다.



### NutriBench: A Dataset for Evaluating Large Language Models in Carbohydrate Estimation from Meal Descriptions (https://arxiv.org/abs/2407.12843)
- **What's New**: NutriBench는 식사 설명을 자연어로 기술한 최초의 공개 가능한 영양 벤치마크 데이터셋입니다. NutriBench는 5,000개의 사람이 검증한 식사 설명과 탄수화물, 단백질, 지방, 칼로리와 같은 매크로 영양소 레이블을 포함하고 있습니다. 이 데이터셋은 식사 구성 요소의 수, 서빙 크기 및 인기 정도에 따라 다양한 15개의 하위 집합으로 나뉩니다. NutriBench를 사용하여 GPT-3.5, Llama-3, MedAlpaca와 같은 최첨단 대형 언어 모델(LLM)의 영양 추정 성능을 평가했습니다.

- **Technical Details**: NutriBench 구성은 탄수화물 추정에 철저한 평가를 위해 설계되었습니다. GPT-3.5, Llama-3, MedAlpaca와 같은 모델이 표준, Chain-of-Thought(CoT) 및 Retrieval-Augmented Generation(RAG) 전략에서 테스트되었습니다. Chain-of-Thought는 모델이 단계별로 답변에 대해 추론할 수 있게 하며, RAG는 외부의 신뢰할 수 있는 데이터 소스를 검색해 모델에 추가 정보를 제공합니다. 데이터셋은 US Department of Agriculture(USDA)의 FoodData Central(FDC)에서 수집한 음식 항목과 영양소 레이블을 바탕으로 구축되었습니다.

- **Performance Highlights**: GPT-3.5는 Chain-of-Thought(CoT) 프롬프트를 사용하여 51.48%의 정확도를 기록하며, 답변 생성 비율은 89.80%였습니다. LLM들은 탄수화물 추정에서 전문가 및 비전문가 인간 참가자보다 더 정확하고 빠른 예측을 제공하는 것으로 나타났습니다. 이는 복잡한 식사 설명과 같은 다양한 쿼리에 대응할 수 있는 LLM의 잠재력을 보여줍니다.



### MS2SL: Multimodal Spoken Data-Driven Continuous Sign Language Production (https://arxiv.org/abs/2407.12842)
Comments:
          Accepted to ACL 2024 Findings; Project Page: this https URL

- **What's New**: 이 연구는 음성 또는 텍스트로부터 직접적으로 연속적인 수어(sign language) 시퀀스를 생성하는 통합 프레임워크를 제안합니다. 이 접근법은 수어 사용자와 비수어 사용자 간의 원활한 소통을 목표로 합니다.

- **Technical Details**: 제안된 프레임워크는 텍스트와 음성으로부터 추출된 임베딩을 활용하여 단계별로 수어 예측을 생성하는 시퀀스 디퓨전 모델(sequence diffusion model)을 사용합니다. 텍스트, 오디오 및 수어 간의 공동 임베딩 공간을 생성하여 모델 훈련 시 의미론적 일관성을 유지합니다. 이는 수어 트리플렛(triplets)에 의존하지 않고, 오디오 모달리티가 부족한 경우에도 모델의 지속적인 개선을 가능하게 합니다.

- **Performance Highlights**: How2Sign 및 PHOENIX14T 데이터셋 실험 결과, 제안된 모델이 수어 생산(SLP)에서 경쟁력 있는 성능을 달성했습니다.



### What to do if language models disagree? Black-box model ensembling for textual and visual question answering (https://arxiv.org/abs/2407.12841)
- **What's New**: 새로운 AI 모델로 InfoSel가 도입되었습니다. InfoSel은 블랙박스 모델들 사이에서 최적의 모델을 선택하여 예측하는 데이터 효율적 및 경량형 앙상블 방법입니다. 이 모델은 특히 텍스트 및 멀티모달 비주얼 질의응답(Visual Question Answering, VQA) 작업에서 기존의 대형 언어 모델(LLM) 및 VQA 모델의 성능 향상을 목표로 합니다.

- **Technical Details**: InfoSel는 전통적인 앙상블 기법과 달리 예측 확률 또는 확신도(confidence)를 사용하지 않습니다. 대신에 텍스트 변환기(Textual Transformer, TT, 110M 파라미터)와 멀티모달 변환기(Multimodal Transformer, MT, 115M 파라미터)를 백본으로 사용하여 텍스트 및 멀티모달 입력에 대한 상황 맥락적 표현을 생성합니다. 이는 각 입력에 대해 가장 정확한 기본 모델(LLM 또는 VQA 모델)을 동적으로 식별하는 메타 수준의 분류 작업으로 구현됩니다. 또, 제한된 수의 학습 샘플로 훈련 가능한 경량 모델입니다.

- **Performance Highlights**: InfoSel은 4개의 데이터셋에서 단독 LLM보다 최대 +5.27%의 F1-score 절대 향상을 달성했습니다. 특히, Multimodal 변환기를 포함한 InfoSel-MT 모델은 텍스트 전용 기존의 앙상블 기법보다 월등한 성능을 보였습니다. 추가로, InfoSel* 모델과 결합할 경우 Mini-Viz 데이터셋에서 성능이 +31.63%까지 향상되었습니다. 또한, InfoSel은 10개의 학습 샘플로도 주요 베이스 모델보다 뛰어난 성능을 보여줍니다.



### Historical Ink: 19th Century Latin American Spanish Newspaper Corpus with LLM OCR Correction (https://arxiv.org/abs/2407.12838)
- **What's New**: 이번 논문에서는 두 가지 주요 기여를 소개합니다. 첫째, 19세기 라틴 아메리카 신문 텍스트의 새로운 데이터셋을 공개합니다. 이 데이터셋은 이 지역의 역사적, 언어적 분석을 위한 전문 코퍼스의 부족 문제를 해결합니다. 둘째, 대형 언어 모델(Large Language Model, LLM)을 활용해 OCR 오류 수정 및 언어 표면 형태 감지를 위한 프레임워크를 소개합니다. 이 프레임워크는 다양한 상황에 적용할 수 있으며, 이번 논문에서는 특히 새로 생성된 데이터셋에 적용되었습니다.

- **Technical Details**: 19세기 라틴 아메리카 신문 텍스트로 구성된 데이터셋은 콜롬비아의 국립 도서관과 루이스 앤젤 아랑고 도서관의 디지털 카탈로그에서 수집되었습니다. 데이터셋은 4,032 페이지의 스캔된 이미지를 포함하며, Azure AI Vision 모델을 통해 OCR 처리가 이루어졌습니다. 그러나 많은 텍스트는 독해 불가능하거나 다수의 전사 오류를 포함하고 있어 텍스트의 읽기 능력과 NLP-LLM 모델의 입력 자료로 사용할 때 바이어스를 초래합니다. 이를 보완하고자 GPT-3.5를 이용하여 OCR 오류를 자동으로 감지하고 수정하는 프레임워크를 도입했습니다.

- **Performance Highlights**: OCR 오류 감지는 전통적인 규칙 기반 필터링과 GPT-3.5 모델을 활용한 교정을 결합하여 이루어졌습니다. 텍스트 전처리 과정에서 중복 제거, 노이즈 데이터 필터링, 비문자 비율이 높은 텍스트 제거 등 다양한 정화 작업을 수행했습니다. 그 결과, 텍스트의 정확성과 가독성이 크게 향상되었습니다. 구체적으로는, OCR 교정 성능이 크게 개선되었으며, 구체적인 역사적 텍스트에서 발생하는 특수 오류도 효과적으로 수정되었습니다.



### OSPC: Artificial VLM Features for Hateful Meme Detection (https://arxiv.org/abs/2407.12836)
- **What's New**: 이 논문은 디지털 혁명과 월드 와이드 웹의 등장으로 인간 소통에서 밈(meme)의 강력한 영향력을 조사합니다. 밈은 간단하면서도 인기 있는 표현 수단이지만 익명성과 사용의 용이성으로 인해 허위 정보 및 증오를 확산할 수 있습니다. 이에 대응하기 위해 팀 Baseline이 AI 싱가포르 온라인 안전 챌린지에서 개발한 솔루션을 소개합니다.

- **Technical Details**: 이 솔루션은 대규모 Vision-Language Models(VLMs)의 고유한 확률적 기능을 활용하여 텍스트에서 과제에 맞춘 특징 인코딩을 생성합니다. 또한 싱가포르의 특정 문화적 뉘앙스에 맞춘 증류화된 양자화(distilled quantization)을 적용하여 효율성을 높이고 정확도를 향상시킵니다. 이를 통해 대형 GPU 없이도 자원 제약이 있는 애플리케이션에서 확장 가능하며, 데이터가 거의 없거나 전혀 없는 경우에도 적용할 수 있습니다.

- **Performance Highlights**: 이 솔루션은 테스트 데이터셋에서 AUROC 0.76과 정확도 0.69를 달성하였습니다. OCR을 활용하여 중국어와 타밀어 텍스트를 효율적으로 필터링하는 트릭을 사용하여 정보를 처리하고, 시각적 처리를 통해 이미지 인식을 향상시켜 성능을 크게 높였습니다.



### Regurgitative Training: The Value of Real Data in Training Large Language Models (https://arxiv.org/abs/2407.12835)
- **What's New**: 본 논문에서는 대형 언어 모델(Large Language Models, LLMs)을 다른 LLM이 생성한 데이터로 학습할 때 발생하는 '회귀적 학습(regurgitative training)'이 LLM의 성능에 미치는 영향을 분석합니다. 최근 많은 온라인 콘텐츠가 인간이 아닌 LLM에 의해 생성되고 있으며, 이는 차세대 LLM의 학습 데이터에 들어갈 가능성이 높습니다. 논문은 이러한 회귀적 학습이 모델 성능을 명확하게 저해한다는 강력한 증거를 제시합니다.

- **Technical Details**: GPT-3.5를 기계 번역 작업에 대해 자체 생성 데이터 또는 다른 LLM이 생성한 데이터로 미세 조정(fine-tuning)하면서 회귀적 학습의 영향을 평가했습니다. 이를 통해, 회귀적 학습이 인간이 생성한 실제 데이터를 사용한 학습보다 성능이 떨어진다는 것을 확인했습니다. 특히, LLM이 생성한 데이터의 오류율이 높고 어휘 다양성이 낮다는 두 가지 메커니즘을 통해 성능 저하의 주된 원인을 규명했습니다.

- **Performance Highlights**: 다양한 생성적 작업과 모델 설정에서 일관되게 회귀적 학습을 거친 LLM이 실제 데이터를 사용한 LLM보다 성능이 낮았습니다. 더욱이 소량의 실제 데이터를 포함시키는 것만으로도 LLM 생성 데이터로만 학습한 것보다 우수한 성능을 냈습니다. 데이터 품질을 기준으로 한 우선 순위 학습, 다양한 LLM 소스의 데이터 결합, AI 탐지기를 이용한 인간 유사 데이터 우선 사용 등의 세 가지 전략을 제안하여 회귀적 학습의 성능 저하를 어느 정도 완화할 수 있음을 확인했습니다.



### ESQA: Event Sequences Question Answering (https://arxiv.org/abs/2407.12833)
Comments:
          25 pages, 3 figures

- **What's New**: 이 논문은 이벤트 시퀀스(Event Sequences, ESs)를 처리하기 위한 새로운 방법론을 제안합니다. 특히, 긴 시퀀스를 처리하고, 시간 및 숫자 특성 처리를 개선한 ESQA라는 새로운 신경망 아키텍처를 개발했습니다. 이 방법론은 기존의 대규모 언어 모델(Large Language Models, LLMs)을 활용하여, 적은 미세 조정(finetuning)으로도 여러 다운스트림 작업을 해결할 수 있습니다.

- **Technical Details**: 이벤트 시퀀스는 불규칙한 시간 간격으로 도착하는 이벤트와, 구조화된 주석을 가지는 데이터입니다. 제안된 방법은 LLM 백본(backbone)을 사용하여 이벤트 시퀀스 도메인에서 질문-응답 접근 방식을 발전시킵니다. 특히, LLM의 입력으로 구조화된 데이터를 효과적으로 인코딩하고, 긴 입력 시퀀스를 처리할 수 있는 능력을 갖추었으며, 시간 특성과 순서를 모델에 적절히 제공하는 것이 특징입니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, 제안된 ESQA 방법론은 이벤트 시퀀스 도메인에서 최신 성능(state-of-the-art)을 달성했습니다. 미세 조정이 필요하지 않은 상황에서도 여러 다운스트림 작업을 해결할 수 있는 능력을 보여주었습니다.



### Sentence-level Aggregation of Lexical Metrics Correlate Stronger with Human Judgements than Corpus-level Aggregation (https://arxiv.org/abs/2407.12832)
- **What's New**: 이 논문에서는 Machine Translation(MT) 시스템 평가에 있어 corpus-level aggregation이 lexical metrics의 정확성을 크게 저해한다는 점을 실험적으로 증명합니다. 개별 segment-level 점수의 평균을 사용하면 BLEU와 chrF 같은 지표가 인간 판정과 훨씬 더 강하게 상관관계를 가지며, COMET 및 BLEURT와 같은 신경망 지표(neural metrics)처럼 작동하게 됩니다. 이를 통해 저자는 corpus-level과 segment-level의 집계 방법이 수학적으로 차이가 존재하며, 이는 corpus-level 집계 방식이 통계적으로 덜 신뢰할 수 있음을 보여줍니다.

- **Technical Details**: CLA(Corpus-level Aggregation)와 SLA(segment-level Aggregation)의 주요 차이는 CLA는 모든 샘플에 대한 n-그램 매칭 통계를 첫 단계에서 계산하고 전체 테스트 셋의 글로벌 점수를 계산하는 반면, SLA는 각 샘플에 대해 통계와 점수를 개별적으로 계산하여 이러한 점수의 평균으로 테스트 셋을 평가합니다. 실험에서는 WMT23 metrics shared task의 492개 시스템 출력을 사용하였고, 각 방법의 성능을 다양한 기준으로 평가하였습니다. 결과는 CLA와 SLA의 점수 차이가 뚜렷하며, SLA가 더 통계적으로 신뢰할 수 있음을 보여주었습니다.

- **Performance Highlights**: CLA는 더 큰 테스트 셋에서의 점수 상관관계가 단일 샘플 테스트 셋에서의 상관관계와 크게 다르지 않다는 결과를 통해 통계적으로 신뢰할 수 없음을 보였습니다. 대신 SLA는 인간 판정과 더 강한 상관관계를 가지며 BERTScore와도 유사한 성능을 보여줍니다. 이는 SLA가 lexical metrics와 신경망 지표 사이의 성능 격차를 줄일 수 있음을 시사합니다. 이러한 결과는 특히 저자당 언어 자원이 충분하지 않은 저자 자원 언어(low-resource languages)에서 MT 시스템 평가의 신뢰성을 크게 높일 수 있습니다.



### Truth is Universal: Robust Detection of Lies in LLMs (https://arxiv.org/abs/2407.12831)
Comments:
          10 pages, 30 figures

- **What's New**: 새로운 연구는 대형 언어 모델(LLMs)이 거짓말을 하고 있음에도 이를 탐지할 수 있는 더 견고한 방법을 개발한다는 점에서 의미가 큽니다. 연구진은 진실과 거짓 진술의 내부 활성화 벡터를 선형적으로 분리할 수 있는 2차원적인 부분 공간(subspace)을 발견했습니다. 이는 다양한 LLMs, 예를 들어 Gemma-7B, LLaMA2-13B, LLaMA3-8B에서 보편적으로 유효하다고 합니다.

- **Technical Details**: 연구진은 LLM 거짓말 탐지를 위해 활성화 벡터 내에서 진실 방향(truth direction)인 'tG'와 극성 민감 진실 방향(polarity-sensitive truth direction)인 'tP'을 분리해냅니다. 단일 방향으로는 일반화가 어렵기 때문에, 진술의 긍정과 부정을 포함한 2차원 공간을 만들어 이를 해결했습니다. 또한 단순한 진술뿐 아니라 복잡한 현실적인 거짓말까지 탐지할 수 있는 성능을 입증했습니다.

- **Performance Highlights**: 제안된 분류기는 간단한 참/거짓 진술을 94%의 정확도로, 복잡한 현실 세계에서 거짓말을 95%의 정확도로 탐지하는 성능을 보였습니다. 이로써 기존 연구들의 일반화 실패를 극복하고 한 차원 높은 성능을 달성했습니다.



### Knowledge-based Consistency Testing of Large Language Models (https://arxiv.org/abs/2407.12830)
Comments:
          14 pages, 6 figures, 14 tables, Submitted to ACL ARR (June 2024)

- **What's New**: 이 연구에서는 대형 언어 모델(Large Language Models, LLMs)의 일관성 문제와 지식 격차를 체계적으로 드러내고 측정하는 새로운 자동화 테스트 프레임워크인 KONTEST를 제안합니다. KONTEST는 지식 그래프를 활용하여 테스트 케이스를 구축하고, LLM의 세계 지식에 대한 일관성을 측정합니다. 4개의 최신 LLM(Falcon, Gemini, GPT3.5, Llama2)을 사용하여 KONTEST가 19.2%의 오류 유도 입력을 생성하고(9983개의 테스트 입력 중 1917개의 오류) 16.5%의 지식 격차를 드러내었습니다.

- **Technical Details**: KONTEST는 지식 기반 일관성 테스트(knowledge-based consistency testing)를 위해 지식 그래프를 활용합니다. 지식 그래프로부터 엔티티와 엔티티 관계를 추출하여 일관성 테스트를 위한 테스트 케이스를 생성합니다. 메타모픽(metamorphic) 및 온톨로지적(ontological) 오라클을 사용하여 LLM의 일관성 오류를 측정합니다. 또한, KONTEST는 LLM 모델 앙상블(ensemble)을 통해 지식 격차를 완화시키며, 이에 대한 평균 지식 격차를 32.48% 감소시킵니다.

- **Performance Highlights**: KONTEST는 19.2%의 오류 유도 입력을 생성하였고, 테스트된 모든 LLM에서 평균 16.5%의 지식 격차를 드러냈습니다. KONTEST의 완화 방법은 LLM 지식 격차를 32.48% 감소시키는 것으로 나타났습니다. 또한, GPT3.5는 지식 기반 일관성 테스트에 적합하지 않으며, 이는 최대 68%의 정확도를 보이는 것으로 밝혀졌습니다.



### Why Does New Knowledge Create Messy Ripple Effects in LLMs? (https://arxiv.org/abs/2407.12828)
- **What's New**: 최근 언어 모델(Language Models, LMs)의 지식 편집(Knowledge Editing, KE) 방법에 대한 연구에서는 편집된 LMs가 논리적으로 관련된 지식을 정확히 처리하는 능력, 즉 파급 효과(Ripple Effects)를 어떻게 다루는가에 대한 관심이 높아지고 있습니다. 이번 연구에서는 대부분의 KE 방법이 여전히 혼란스러운 파급 효과를 생성하는 이유를 밝히고, GradSim이라는 새로운 지표를 통해 이를 효과적으로 설명하고자 합니다.

- **Technical Details**: GradSim은 원본 사실과 관련 지식의 기울기 간 코사인 유사도(Cosine Similarity)로 계산되며, LMs의 저장된 지식이 편집될 때 이러한 지식 간의 파급 효과를 예측하는 데 강력한 지표인 것으로 나타났습니다. 다양한 LMs, KE 방법 및 평가 메트릭에서 GradSim과 파급 효과 성능 간에 강한 상관관계를 발견했으며, 이는 지식 저장의 유사성이 파급 효과를 결정하는 데 중요한 역할을 한다는 가설을 뒷받침합니다.

- **Performance Highlights**: 실험 결과, GradSim이 높은 경우에는 높은 파급 효과 성능을 보였으며, 이러한 상관관계는 피어슨 상관 계수가 0.85에 이를 정도로 강하게 나타났습니다. 또한 반직관적인 실패 사례(Negation, Over-Ripple Errors, Cross-Lingual Transfer)를 분석한 결과, GradSim이 매우 낮을 때 이러한 실패가 자주 발생한다는 것을 확인했습니다.



### The Solution for The PST-KDD-2024 OAG-Challeng (https://arxiv.org/abs/2407.12827)
- **What's New**: KDD-2024 OAG-Challenge 논문 출처 추적 트랙에서 2위를 차지한 솔루션을 소개합니다. 우리의 솔루션은 주로 BERT와 GCN을 기반으로 하며, 최종 제출물에는 두 모델의 추론 결과를 결합하여 보완 성능을 달성했습니다.

- **Technical Details**: BERT 솔루션에서는 논문의 참고문헌 조각 처리에 중점을 두고, 여러 작업을 통해 불필요한 간섭을 줄여서 BERT가 수신하는 정보가 더욱 정제되었습니다. GCN 솔루션에서는 논문 조각, 초록, 제목 등을 임베딩 모델을 통해 고차원 의미 공간에 매핑하고, 제목, 초록, 조각 간의 경계를 구축하여 컨텍스트 관계를 통합했습니다.

- **Performance Highlights**: BERT와 GCN의 결합으로 인해, 우리의 솔루션은 총점 0.47691을 기록하여 대회에서 뛰어난 성과를 달성했습니다.



### Assessing the Effectiveness of GPT-4o in Climate Change Evidence Synthesis and Systematic Assessments: Preliminary Insights (https://arxiv.org/abs/2407.12826)
- **What's New**: 최근 연구는 GPT-4o라는 최신 대형 언어 모델(LLM)을 사용하여 증거 종합 및 체계적인 평가 작업을 수행하는 가능성을 조사했습니다. 전통적으로 이러한 작업은 대규모 도메인 전문가 그룹이 수동으로 문헌을 검토하고 종합하는 방법에 의존했지만, 과학 문헌의 기하급수적인 증가로 인해 새로운 도구의 필요성이 대두되고 있습니다. 이 연구는 GPT-4o의 효능을 Global Adaptation Mapping Initiative(GAMI) 데이터셋에서 기후 변화 적응 관련 특징 추출의 정확도를 세 가지 전문성 수준에서 평가했습니다.

- **Technical Details**: 연구에서는 GPT-4o가 지리적 위치 식별과 같은 낮은 전문성의 작업에서는 높은 정확도를 달성했지만, 이해관계자 식별 및 적응 응답의 깊이 평가와 같은 중간 및 높은 전문성 수준의 작업에서는 안정적이지 않은 성과를 보였습니다. 데이터셋은 IPCC 과학자들이 주도하는 GAMI에서 제공되었으며, 기후 변화 적응 전문가들이 1,682개의 피어 리뷰된 기사에서 25개의 특징을 라벨링한 내용이 포함되어 있습니다. 이번 연구에서는 음식 부문에 대한 적응 응답에 초점을 맞춰 샘플을 분석했습니다.

- **Performance Highlights**: GPT-4o는 지리적 위치를 식별하는 데 있어 높은 정확도를 보였으나, 중간 및 높은 전문성 수준의 작업에서는 성과가 떨어졌습니다. 특히 이해관계자 식별과 적응 응답의 깊이 평가 부분에서 모델의 성능이 낮게 평가되었습니다. 이는 LLM의 강점을 활용하면서도 이러한 작업에서 성능을 향상시키기 위한 평가 워크플로를 설계할 필요성을 제기합니다.



### A Depression Detection Method Based on Multi-Modal Feature Fusion Using Cross-Attention (https://arxiv.org/abs/2407.12825)
- **What's New**: 이 논문에서는 새로운 다중 모달(multi-modal) 특징 결합 방법을 소개합니다. Cross-Attention 메커니즘을 활용하여 우울증을 감지하는 모델을 제안합니다. 기존의 단순히 특징을 연결(concatenation)하는 방법과는 달리, 이 접근 방식은 교차 주의(cross-attention)를 사용해 보다 정확한 감지 기능을 제공합니다.

- **Technical Details**: 제안된 모델은 MacBERT을 사전 훈련(pre-training) 모델로 사용하여 텍스트의 어휘 특징을 추출합니다. 그리고 추가적인 Transformer 모듈을 포함하여 과제별 맥락 이해를 향상시킵니다. Cross-Attention 메커니즘을 통해 다중 모달 입력의 통합된 특징을 생성하게 됩니다. 이 모델은 다중 모달 입력(MFNN) 기반의 깊은 신경망 분류 네트워크로 우울증 감지에 특화되어 있습니다.

- **Performance Highlights**: 우울증 감지에서 이 모델은 테스트 데이터 세트에서 0.9495의 정확도를 달성하였습니다. 이는 기존 방식들보다 상당히 향상된 성과를 보여줍니다. 이 방법론은 다른 소셜 미디어 플랫폼이나 다중 모달 처리 관련 작업에도 유망한 접근법을 제시합니다.



### Whispering Experts: Neural Interventions for Toxicity Mitigation in Language Models (https://arxiv.org/abs/2407.12824)
Comments:
          ICML 2024, 8 pages + appendix

- **What's New**: LLM(대형 언어 모델)의 독성 언어 생성 문제를 해결하기 위해 AUROC 적응(AurA)이라는 새로운 방법을 제안했습니다. 이 방법은 사전 학습된 LLM의 독성 언어 생성을 억제하는 방법으로, 각 뉴런의 독성 문장을 구별하는 능력에 비례하여 독성 유발 뉴런의 활성화 수준을 줄입니다. 이는 모델에 의존하는 하이퍼파라미터 없이 적용할 수 있는 개입 방법을 제공합니다.

- **Technical Details**: AurA는 특정 개념을 인코딩하는 전문 뉴런을 조정함으로써 독성 언어 생성을 억제합니다. 각 뉴런의 능력에 비례하여 이들의 기여를 줄이는 소프트 개입 전략을 사용하며, 이는 완전히 무효화하는 것보다는 적은 영향을 미칩니다. 이 방법은 특정 모델에 의존하지 않으며 모든 사전 학습된 LLM에 적용할 수 있습니다.

- **Performance Highlights**: AurA는 Mistral-7B 모델에서 독성을 최대 2.2배까지 줄이고, 혼동도(perplexity)는 0.72점만 증가시킵니다. 또한, 다양한 크기의 모델(1.5B에서 40B 매개변수까지)에서 효과적이며, 상식 기반의 제로샷 능력을 유지하면서도 독성 언어를 효과적으로 억제할 수 있습니다. 악의적인 프롬프트에도 평균적으로 2배의 독성 저감을 달성하며, 사전 프롬프트 전략과 결합되면 독성 저감 능력이 더욱 향상됩니다.



### WTU-EVAL: A Whether-or-Not Tool Usage Evaluation Benchmark for Large Language Models (https://arxiv.org/abs/2407.12823)
- **What's New**: 최근 논문은 대형 언어 모델(LLMs)이 도구를 사용해야 하는 상황을 스스로 파악할 수 있는지 여부를 평가하는 WTU-Eval 벤치마크를 도입했습니다. 기존 연구들은 LLM이 무조건적으로 도구를 사용해야 한다는 가정을 하고 있었지만, 이 논문은 실제 환경에서는 도구 사용의 필요성이 항상 확실하지 않다는 점에 주목했습니다. WTU-Eval은 LLM이 도구 사용의 필요성을 인식하고 유연하게 도구를 사용할 수 있는지를 평가합니다.

- **Technical Details**: WTU-Eval 벤치마크는 총 11개의 데이터셋으로 구성되어 있습니다. 그 중 6개는 도구 사용 데이터셋이고, 5개는 일반 데이터셋입니다. LLM은 필요에 따라 도구를 사용하도록 유도됩니다. 또한, 도구 사용 결정을 개선하기 위한 파인튜닝(finetuning) 데이터셋도 개발되었습니다. 이를 통해 Llama2-7B 모델의 성능이 14% 향상되었고, 잘못된 도구 사용 비율이 16.8% 감소했습니다.

- **Performance Highlights**: WTU-Eval 벤치마크의 결과에 따르면, 대부분의 LLM은 일반 데이터셋에서 도구 사용 필요성을 잘 판단하지 못하며, 도구 사용 데이터셋에서는 ChatGPT와 유사한 성능을 보이는 경우 성능이 향상된다는 것을 발견했습니다. 또한, 잘못된 도구 사용은 LLM의 성능에 큰 영향을 미친다는 것을 확인했습니다. 파인튜닝을 거친 Llama2-7B 모델은 평균 성능 향상 14%를 기록하고, 잘못된 도구 사용 비율을 16.8% 줄이는 데 성공했습니다.



### Lightweight Large Language Model for Medication Enquiry: Med-Pa (https://arxiv.org/abs/2407.12822)
- **What's New**: Med-Pal이라는 새로운 약물 도메인 특정 LLM 챗봇이 개발되었습니다. 이 챗봇은 가벼운 오픈 소스 LLM 중 일부를 선별하여 미세 조정된 데이터셋으로 학습되었습니다. 특히 computational constraints(계산 제약)과 operational efficiency(운영 효율성)을 우선시 하였습니다.

- **Technical Details**: 다섯 개의 경량(파라미터 크기 7 billion 이하) 오픈 소스 LLM을 사용하여, 면밀히 조사된 데이터셋으로 미세 조정하고 임상 평가를 통해 Med-Pal을 개발하였습니다. 임상 평가 기준으로는 SCORE 기준(안전성, 정확성, 편향성, 재현성, 이해 용이성)을 사용하였습니다. 이후 Mistral-7b 모델이 가장 성능이 뛰어났음을 확인하여 Med-Pal의 기본 LLM으로 선정되었습니다.

- **Performance Highlights**: Mistral-7b는 평가에서 14점의 중간 점수와 71.9%의 고품질 응답률을 기록하여, Med-Pal의 백본으로 선택되었습니다. Med-Pal은 Biomistral과 Meerkat과 비교할 때 환자와의 소통에서 더 적합한 응답을 생성하며, 일반 LLM에서 흔히 발생하는 편향과 오류를 현저히 줄였습니다.



### AutoFlow: Automated Workflow Generation for Large Language Model Agents (https://arxiv.org/abs/2407.12821)
Comments:
          Open source code available at this https URL

- **What's New**: AutoFlow는 복잡한 작업을 해결하기 위해 AI 에이전트의 워크플로우를 자동으로 생성하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 자연어 프로그램을 워크플로우의 형식으로 사용하며, 작업의 품질을 반복적으로 최적화하는 워크플로우 최적화 절차를 포함합니다. 이를 통해 수동으로 워크플로우를 설계할 필요 없이, 높은 신뢰성을 가진 에이전트를 대규모로 개발 및 배포할 수 있습니다.

- **Technical Details**: AutoFlow는 워크플로우 생성 방식을 두 가지 방법으로 제안합니다: 파인 튜닝 기반(fine-tuning-based) 방법과 인 컨텍스트(in-context-based) 방법. 파인 튜닝 기반 방법은 특정 작업과 도메인에 맞춰 LLM의 파라미터를 조정하여 워크플로우 생성 과정을 맞춤화하는 방법입니다. 인 컨텍스트 기반 방법은 광범위한 파인 튜닝이 필요하지 않으며 LLM이 주어진 문맥 정보를 기반으로 워크플로우를 생성합니다. 이와 같은 방법은 오픈 소스 및 폐쇄 소스 LLM 모두에 적용 가능합니다. 이 외에도 강화 학습(RL)을 활용하여 작업의 보상을 기반으로 워크플로우 생성 LLM을 업데이트 함으로써 점차적으로 최적의 워크플로우를 생성할 수 있도록 합니다.

- **Performance Highlights**: AutoFlow 프레임워크는 실험 결과, 수동으로 설계된 워크플로우보다 우수한 성능을 보였습니다. 또한 생성된 워크플로우가 읽기 쉽고 사용하기 편리하며, 높은 신뢰성을 가지고 복잡한 작업을 수행할 수 있음을 입증했습니다. 이는 특히 LLM 기술의 빠른 발전과 맞물려 복잡한 문제를 해결하는데 유망한 새로운 패러다임을 제시합니다.



### PQCache: Product Quantization-based KVCache for Long Context LLM Inferenc (https://arxiv.org/abs/2407.12820)
- **What's New**: 최근 대형 언어 모델(LLMs)의 진화와 함께, 추론에서 사용하는 문맥 길이가 지속적으로 증가하고 있습니다. LLM 추론에서 중요한 요소인 키-값 캐시(KVCache)는 GPU 메모리 제한으로 인해 주요 메모리 병목 현상으로 작용하고 있습니다. 이를 해결하기 위해 PQCache라는 새로운 접근법을 제안합니다. PQCache는 데이터베이스 커뮤니티에서 사용하는 고급 임베딩 검색 기술을 도입하여 KVCache를 관리합니다. 특히, Product Quantization(PQ)을 사용하여 모델 품질을 유지하면서도 낮은 대기 시간을 제공합니다.

- **Technical Details**: PQCache는 Product Quantization(제품 양자화)를 활용하여 KVCache를 관리합니다. PQCache는 초기 단계에서 각각의 LLM 레이어와 헤드에 대해 토큰의 키에 PQ를 적용합니다. 자동 회귀 디코딩 단계에서는 새로 생성된 토큰에 대해 중요 토큰을 PQ 코드와 중심을 사용한 Maximum Inner-Product Search(MIPS)로 식별한 후, 해당 키-값 쌍을 자가 주의 계산을 위해 가져옵니다. PQCache는 겹침과 캐싱의 세심한 설계를 통해 각 단계에서 추가적인 계산 및 통신 오버헤드를 최소화합니다.

- **Performance Highlights**: 광범위한 실험을 통해 PQCache가 효과적이고 효율적임을 입증했습니다. 모델 품질을 유지하면서도 주의 계산에 단지 1/5의 토큰만을 포함시키며, 동시에 시스템 대기 시간도 수용 가능한 수준을 달성했습니다. 특히, LongBench 점수를 기존 방법에 비해 최대 6.21까지 개선하는 성과를 보였습니다.



### "I understand why I got this grade": Automatic Short Answer Grading with Feedback (https://arxiv.org/abs/2407.12818)
- **What's New**: Engineering Short Answer Feedback (EngSAF) 데이터셋은 자동 단답형 채점(Automatic Short Answer Grading, ASAG) 과제를 위해 5,800개의 학생 답변과 함께 참조 답변 및 질문을 포함하고 있습니다. 이 데이터셋은 다양한 엔지니어링 분야를 포괄하며, Label-Aware Synthetic Feedback Generation (LASFG) 전략을 이용해 피드백을 생성합니다. 이를 통해 자동 채점 시스템이 단순히 점수를 매기는 것 이상의 피드백을 제공할 수 있도록 합니다.

- **Technical Details**: EngSAF 데이터셋은 인도공과대학교 봄베이(IITB)에서 실제 기말고사에 사용되었으며, 여러 엔지니어링 과목에서 수집된 119개의 질문과 약 5,800개의 학생 답변을 포함하고 있습니다. 이 데이터셋은 필터링된 학습 데이터로 라벨링 된 '정답', '부분 정답', '오답'의 카테고리로 구분됩니다. 선행 연구에서는 주로 채점에 중점을 두었지만, 본 연구는 피드백 제공의 중요성을 강조하고 있습니다.

- **Performance Highlights**: EngSAF 데이터셋을 사용하여 다양한 대형 언어 모델(LLM) 기반의 zero-shot 및 미세 조정된 벤치마크 모델을 제시합니다. IITB에서의 실제 기말고사에도 성공적으로 배포되어 실질적인 효율성과 효과성을 입증했습니다.



### Error Correction by Paying Attention to Both Acoustic and Confidence References for Automatic Speech Recognition (https://arxiv.org/abs/2407.12817)
- **What's New**: 이번 연구에서는 비자계열(non-autoregressive) 음성 오류 수정 방법을 제안합니다. 이 방법은 N-best ASR 가설에 포함된 각 단어의 불확실성을 측정하는 Confidence Module을 사용하여 오류 위치를 찾아 수정합니다. 또한 ASR 인코더(acoustic feature)를 참조하여 올바른 발음 정보를 제공합니다. N-best 후보를 편집 경로에 맞춰 정렬하고 교차 주의 메커니즘(cross-attention mechanism)을 활용하여 오류 교정 참조 정보와 ASR 가설 간의 정보를 융합하여 수정 정확도를 높였습니다.

- **Technical Details**: 본 연구에서는 새로운 비자계열 음성 오류 수정 모델을 제안합니다. Confidence Module이 N-best ASR 가설의 각 단어의 불확실성을 측정하며, 이는 오류 위치를 찾는 데 활용됩니다. 또한 ASR 인코더의 발음 정보를 사용하여 올바른 발음 참조 자료를 제공합니다. N-best 후보는 편집 경로(edit path)에 따라 정렬되며, 교차 주의 메커니즘을 통해 오류 교정 참조와 ASR 가설 사이의 정보를 융합합니다. 제안된 모델은 신속한 추론 속도를 유지하면서 정확한 오류 교정을 목표로 합니다.

- **Performance Highlights**: 실험 결과, 제안된 시스템은 기존의 ASR 모델에 비해 오류율을 21% 감소시켰습니다. 이 모델은 음향과 신뢰도 참조가 모두 오류 교정에 도움을 주며, 빠른 추론 시간과 높은 오류 수정 정확도를 제공합니다.



### SMLT-MUGC: Small, Medium, and Large Texts -- Machine versus User-Generated Content Detection and Comparison (https://arxiv.org/abs/2407.12815)
- **What's New**: 이 논문에서는 최근 주목받고 있는 대형 언어 모델(LLMs)의 텍스트 생성 능력과 이들이 생성한 텍스트를 식별하는 방법에 대해 연구합니다. 다양한 텍스트 길이를 지닌 데이터셋을 분석하여, 기존의 머신러닝 알고리즘이 얼마나 효과적으로 LLMs가 생성한 텍스트를 식별할 수 있는지 비교합니다. 주로 SVM와 Voting Classifier(VC) 모델의 성능이 뛰어나고, Decision Tree(DT) 모델은 가장 낮은 성능을 보였습니다.

- **Technical Details**: 연구에서 사용한 데이터셋은 소형(선거, FIFA, 그리고 왕좌의 게임 트윗), 중형(Wikipedia 소개와 PubMed 초록), 대형(OpenAI 웹 텍스트 데이터셋)으로 분류되었습니다. GPT2 XL-1542 모델처럼 매개변수가 매우 큰 LLMs의 텍스트는 기존 머신러닝 방법으로 식별하기 어려웠으며(74% 정확도), 반면에 매개변수가 작은 LLMs (762 million 이하)의 경우 높은 정확도(96% 이상)로 식별되었습니다. 텍스트의 언어학적, 성격, 감정, 편향, 도덕성 등 여러 차원을 분석하였습니다.

- **Performance Highlights**: GPT-2 XL-1542 같은 큰 LLM의 경우 원본 텍스트 식별이 어렵고, 작은 LLM의 경우 높은 정확도로 식별할 수 있었습니다. SVM과 Voting Classifier(VC) 모델은 대부분의 데이터셋에서 높은 성능을 유지했지만, Decision Tree(DT) 모델은 가장 낮은 성능을 보였습니다. 재구성된 텍스트의 경우 성능이 떨어지며, 특히 트윗과 같은 짧은 텍스트에서 두드러졌습니다.



### Computational Politeness in Natural Language Processing: A Survey (https://arxiv.org/abs/2407.12814)
Comments:
          Manuscript accepted at the ACM Computing Surveys (DOI: this https URL)

- **What's New**: 이번 아카이브 논문은 자연어 처리(NLP)에서 예의(politeness)를 예측하고 생성하는 컴퓨팅 접근법에 대한 종합적인 연구 결과를 정리한 것입니다. 대화 분석 커뮤니티에서 예의의 중요성과 도전 과제를 인지하여, 기존 연구를 바탕으로 예의가 텍스트에서 어떻게 식별되고 유도될 수 있는지에 대한 네 가지 주요 단계를 제시했습니다.

- **Technical Details**: 이 논문은 감독 학습(supervised learning), 약한 감독 학습(weakly-supervised learning) 기능 추출, 목표 텍스트를 넘어서의 문맥(context) 통합, 다양한 사회적 요인에 따른 예의 연구, 그리고 예의와 사회언어학적 신호들 사이의 관계 연구라는 네 단계로 나누어 설명하고 있습니다. 이를 통해 데이터셋, 접근법, 트렌드, 그리고 컴퓨팅 예의 연구에서 나타나는 문제들을 다룹니다.

- **Performance Highlights**: 기존 연구에서 보고된 성능 값을 제시하면서, 컴퓨팅 예의 연구에서 사용된 다양한 차원(예: 기능 타입, 주석 기법, 사용된 데이터셋)들을 요약한 표를 포함하고 있습니다. 이는 연구자들이 현재의 연구 상태를 이해하는 데 유용한 자료를 제공합니다.



### Data Generation using Large Language Models for Text Classification: An Empirical Case Study (https://arxiv.org/abs/2407.12813)
Comments:
          Accepted by DMLR @ ICML 2024

- **What's New**: 최근 대형 언어 모델(LLMs)을 활용한 합성 데이터 생성을 통해 모델 학습을 개선하는 연구가 활발히 진행되고 있습니다. 본 연구에서는 텍스트 분류 작업에 합성 데이터를 활용하여 자연어 이해(NLU) 모델의 성능을 평가하였습니다. 다양한 데이터 생성 접근법의 영향을 실증적으로 분석하고, 보다 나은 데이터 생성 관행을 제안합니다.

- **Technical Details**: 연구는 LLM을 활용한 텍스트 분류용 합성 데이터 생성 방법을 탐구합니다. 주요 초점은 무중지(in-context) 학습 기법의 활용으로, 이러한 기법에는 제로샷(zero-shot), 원샷(one-shot), 그리고 몇몇 샘플을 제공하는 몇샷(few-shot) 기법이 포함됩니다. 실험은 6개의 일반적인 NLP 작업을 대상으로 수행되었으며, 각각의 작업은 SST-2, Twitter Emotion Classification, New York Times News Classification, Amazon Review Classification, RTE, BoolQ입니다. 합성 데이터를 평가하기 위해 GPT-3.5 turbo 모델을 사용하여 다양한 접근법으로 데이터를 생성하고 NLU 모델을 학습시켜 결과를 비교하였습니다.

- **Performance Highlights**: 연구 결과, 합성 데이터 생성 방식의 성능은 작업의 특성과 데이터의 양에 따라 다르게 나타났습니다. 특히 제한된 데이터가 있는 작업에서는 합성 데이터가 큰 도움이 되었으나, 이미 충분한 데이터가 있는 작업에서는 추가적인 합성 데이터가 성능 개선에 큰 영향을 주지 않았습니다. 제로샷 주제 기반 무중지 데이터 생성(Zero-shot topic in-context generation)은 특정 작업에서 유용한 결과를 보였습니다.



### Building Understandable Messaging for Policy and Evidence Review (BUMPER) with AI (https://arxiv.org/abs/2407.12812)
Comments:
          21 pages, 6 figures

- **What's New**: 이번 논문에서는 대규모 언어 모델(LLMs)을 활용하여 정책 및 증거 검토를 위한 이해 가능한 메시징을 구축하는 BUMPER 프레임워크를 소개합니다. 이 프레임워크는 과학적 증거를 정책과 행동으로 효과적으로 번역하여 글로벌 생활 수준을 향상시키는 것을 목표로 합니다. 하지만 LLMs의 접근성, 신뢰성, 책임성과 관련된 도전 과제도 존재합니다. BUMPER는 과학 지식 기반 위에 구축되며, 투명성, 범위 제한, 명시적 검토, 불확실성 측정 등을 통해 신뢰성을 높이는 솔루션을 제시합니다.

- **Technical Details**: BUMPER는 과학적 지식 기반(예: 문서, 코드, 설문 데이터)을 활용하며, 해당 지식 기반은 동일한 과학자(예: 개인 기여자, 연구소, 컨소시엄)에 의해 유지됩니다. 이 프레임워크는 사용자가 자연 언어로 질문을 하고 그에 대한 답변을 제공하는 반응형 대화 인터페이스를 사용합니다. 또한 코드와 사례 연구를 통해 과학자들이 이 공간을 탐색할 수 있도록 지원합니다. LLMs는 사용자에게 친숙한 언어로 질문을 할 수 있게 하여 과학적 지식 전수를 혁신할 가능성을 가지고 있지만, 신뢰성 문제 특히 '환각' 문제는 여전히 해결되지 않은 과제입니다. 이를 해결하기 위해 BUMPER는 투명성 있는 체크와 명확한 소유권을 통해 신뢰성을 증진시키고자 합니다.

- **Performance Highlights**: BUMPER 프레임워크의 성능을 평가하는 핵심 요소로는 컴플라이언스 점수(compliance score)의 공식화와 프레임워크의 안정성 특성을 평가하는 방법이 포함됩니다. 또한 프레임워크의 적용 사례로 럭비 팀 성과 모델과 홍역 시즌 관리 모델을 사용하여 실세계 응용 프로그램을 시연합니다.



### TourLLM: Enhancing LLMs with Tourism Knowledg (https://arxiv.org/abs/2407.12791)
- **What's New**: 최근 대규모 언어 모델(LLMs)은 여러 자연어 처리(NLP) 작업에서 큰 효과를 보였습니다. 그러나 관광 도메인 지식이 부족해 관광 명소 소개와 여행 계획에서의 성능이 제한됩니다. 이를 해결하기 위해 문화 및 관광 도메인을 위한 감독 학습 데이터셋 'Cultour'를 구축하였습니다. 이 데이터셋은 관광 지식 기반 QA 데이터, 여행기 데이터, 관광 다양성 QA 데이터로 구성됩니다. 또한, Cultour로 감독 학습된 Qwen 기반 모델인 TourLLM을 제안하여 명소 정보 제공과 여행 계획의 질을 개선했습니다.

- **Technical Details**: Cultour는 세 가지 주요 부분으로 구성됩니다. 첫째, 관광 명소와 관련된 자주 묻는 질문과 답변(QA) 데이터베이스를 구축했습니다. 둘째, 여행 계획과 관련된 데이터를 정리해 여행기를 수작업으로 작성했습니다. 셋째, 음식, 숙소, 이동, 관광, 쇼핑, 오락 측면에서 특색 있는 QA 데이터를 만들어 데이터셋의 다양성을 풍부하게 했습니다. TourLLM은 Qwen 모델을 기반으로 하며, Cultour로 감독 학습되었습니다. 모델의 성능을 평가하기 위해 자동 평가와 인간 평가를 모두 사용했으며, 일관성(Consistency), 가독성(Readability), 이용 가능성(Availability)을 고려한 인간 평가 기준 CRA를 제안했습니다.

- **Performance Highlights**: TourLLM의 성능 평가를 위해 자동 평가와 인간 평가를 병행하였으며, 실험 결과 TourLLM의 응답이 효과적임을 입증했습니다. 특히, CRA 기준에서 높은 점수를 얻었으며, 이는 TourLLM이 관광 도메인에서 유용한 정보를 제공함을 의미합니다.



### GPT Czech Poet: Generation of Czech Poetic Strophes with Language Models (https://arxiv.org/abs/2407.12790)
- **What's New**: 이 연구에서는 사전 학습된 대형 언어 모델(Large Language Model)을 활용하여 체코어 시 생성의 성능을 향상시키는 새로운 방법을 소개합니다. 구체적으로, 시 텍스트 내에서 연(시의 단락) 매개변수를 명시적으로 지정함으로써 생성 과정의 효과를 크게 개선할 수 있음을 시연합니다.

- **Technical Details**: 연구에서는 주로 음절이나 개별 문자 기반의 적절한 토크나이제이션(tokenization) 방법이 연의 생성에 있어 단어 하위 단위(subwords) 기반의 토크나이제이션 방법보다 우수함을 입증합니다. 그리고 추가적으로, 강제 생성(Forced generation)의 기법을 도입하여 기존에 생성된 텍스트를 기반으로 운율 및 구절 매개변수를 명시적으로 지정함으로써 결과를 더욱 개선합니다.

- **Performance Highlights**: 제안된 접근 방식은 생성된 시의 정형적 품질 측면에서 특히 운율(rhyming) 및 운율적인(metric) 요소에서 높은 정확도를 달성함을 보여줍니다.



### Scaling Granite Code Models to 128K Contex (https://arxiv.org/abs/2407.13739)
- **What's New**: 이번 논문에서 소개된 Granite 코드 모델은 최대 128K 토큰의 긴 컨텍스트를 효과적으로 지원합니다. Granite 3B/8B 모델의 컨텍스트 길이를 2K/4K에서 128K로 확장시키기 위해 지속적인 경량의 사전 학습이 적용되었으며, 이는 RoPE 베이스 주파수를 점진적으로 증가시키고 리포지토리 수준의 파일 패킹과 길이 업샘플링된 긴 컨텍스트 데이터를 사용하여 이루어졌습니다. 또한, 짧은 컨텍스트와 긴 컨텍스트 모두를 지원하는 인스트럭션 기반 모델도 공개되었습니다. 이 모델들은 짧은 컨텍스트 Granite 코드 모델과 비교했을 때 긴 컨텍스트 과제에서 상당한 성능 향상을 보였으며, 짧은 코드 완성 벤치마크에서는 성능 저하 없이 동등한 성능을 유지합니다. 모든 긴 컨텍스트 Granite 코드 모델은 연구와 상업적 용도로 Apache 2.0 라이선스 하에 공개됩니다.

- **Technical Details**: Granite 코드 모델의 컨텍스트 길이를 확장하기 위한 솔루션은 지속적인 사전 학습과 인스트럭션 튜닝 단계로 구성됩니다. 먼저 Granite 코드 3B/8B 기본 모델을 리포지토리 수준의 코드 데이터와 언어별 컨텍스트 길이 업샘플링으로 사전 학습합니다. RoPE 베이스 주파수를 점진적으로 증가시키며, 500단계 동안 배치 크기 32로 훈련합니다. 128K 토큰 길이를 지원하기 위해서는 Ring Attention과 같은 기술을 적용했습니다. 긴 컨텍스트 인스트럭션 데이터는 기존 Granite 코드 인스트럭션 모델을 기반으로 생성된 데이터와 짧은 컨텍스트 및 긴 컨텍스트 데이터의 혼합을 사용하여 모델을 추가로 미세 조정하여 준비되었습니다.

- **Performance Highlights**: 긴 컨텍스트 Granite 코드 모델은 HumanEvalPack, Long Code Completion, RepoBench-P, RepoQA, Key Retrieval과 같은 다양한 짧은 컨텍스트 및 긴 컨텍스트 과제에서 실험을 통해 평가되었습니다. 실험 결과, 긴 컨텍스트 과제에서 성능이 크게 향상되었으며, 짧은 컨텍스트 과제에서는 성능 저하 없이 유지되었습니다.



### New Capability to Look Up an ASL Sign from a Video Examp (https://arxiv.org/abs/2407.13571)
Comments:
          11 pages, 10 figures

- **What's New**: ASL(미국 수화) 사전을 사용하는 새로운 방식이 제안되었습니다. 이 시스템은 사용자들이 비디오를 제출하면 가장 가능성이 높은 다섯 가지 징후를 보여주고, 선택한 징후에 대한 상세 정보를 제공하는 'ASLLRP Sign Bank'로 이동할 수 있게 합니다. 더불어, 새로운 SignStream 소프트웨어의 버전에도 통합되어 ASL 비디오 데이터의 언어적 주석을 더 효율적으로 할 수 있습니다.

- **Technical Details**: 이 시스템은 사용자가 웹캠으로 녹화한 비디오나 지속적인 수화 비디오 클립을 제출하는 방식으로 작동합니다. 사용자가 제출한 비디오를 분석한 후, 가능성이 높은 다섯 가지 수화 징후를 보여줍니다. 사용자가 선택한 징후에 대해서는 ASLLRP Sign Bank의 상세 정보를 제공하며, SignStream 소프트웨어와의 통합을 통해 사용자들은 주석을 더 효율적으로 추가할 수 있습니다.

- **Performance Highlights**: 기존의 ASL 사전 검색 방식에 비해 이 새로운 시스템은 훨씬 직관적이고 효율적입니다. 비디오 기반 검색을 통해 사용자는 수화의 의미나 가능한 영어 번역을 알지 못해도 효과적으로 적절한 수화 징후를 찾을 수 있습니다. 이는 특히 언어적 주석 작업을 하는 사람들에게 큰 도움이 됩니다.



### Qalam : A Multimodal LLM for Arabic Optical Character and Handwriting Recognition (https://arxiv.org/abs/2407.13559)
- **What's New**: 이 연구는 아랍어 OCR (Optical Character Recognition) 및 HWR (Handwriting Recognition)을 위한 새로운 모델인 Qalam을 소개합니다. 이 모델은 SwinV2 인코더와 RoBERTa 디코더 아키텍처를 기반으로 구축되었으며, 기존 방법보다 월등하게 뛰어난 성능을 보입니다.

- **Technical Details**: Qalam은 450만 개 이상의 아랍어 원고 이미지와 60,000개의 이미지-텍스트 페어를 포함한 다양한 데이터셋에서 훈련되었습니다. 이 모델은 아랍어 대본의 중요한 특징인 짧은 소리 기호(diacritics)를 우수하게 처리할 수 있는 능력을 보여줍니다. 또한, 높은 해상도의 입력을 처리할 수 있어 현존하는 OCR 시스템의 일반적인 한계를 극복합니다.

- **Performance Highlights**: Qalam은 HWR 작업에서 WER (Word Error Rate) 0.80%, OCR 작업에서 1.18%를 달성하며 뛰어난 성능을 입증했습니다. 이는 아랍어 대본 인식에서 정확성과 효율성 면에서 상당한 도약을 나타냅니다.



### Spontaneous Style Text-to-Speech Synthesis with Controllable Spontaneous Behaviors Based on Language Models (https://arxiv.org/abs/2407.13509)
Comments:
          Accepted by INTERSPEECH 2024

- **What's New**: 이번 논문에서는 언어 모델(language model)을 기반으로 한 새로운 자발적 스타일 음성 합성(spontaneous style speech synthesis) 시스템을 제안합니다. 기존 음성 합성 모델들이 다양한 자발적 행동과 미세한 운율(prosody) 변화를 포착하는 데 어려움을 겪었던 부분을 개선하고자 합니다. 제안된 시스템은 자발적 행동을 체계적으로 분류하고 모델링하며, 세밀한 운율 모델링을 통해 더욱 자연스러운 자발적 스타일 음성을 생성할 수 있습니다.

- **Technical Details**: 제안된 모델은 VALL-E를 기반으로 하며, 텍스트 인코더, 주파수 임베딩(audio embedding), 자기회귀(transformer decoder, autoregressive), 비자기회귀(transformer decoder, non-autoregressive) 디코더 그리고 신경 오디오 코덱을 포함합니다. 자발적 행동을 모델링하기 위해 구문 인지적 자발적 행동 인코더(syntactic-aware spontaneous behavior encoder)를 도입했습니다. 또한 라벨 예측기(label predictor)와 자발적 운율 추출기(spontaneous prosody extractor)를 사용하여 세밀한 자발적 운율 표현을 예측합니다. 이 모델은 자발적 행동과 관련된 19개 행동을 표기하고 이를 통해 더 자연스러운 음성을 합성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기준(baseline) 방법들에 비해 운율 자연스러움(prosody naturalness)과 자발적 행동 자연스러움(spontaneous behavior naturalness) 측면에서 크게 뛰어나다는 것을 확인했습니다. 명시적 라벨 제어를 통해 모델이 자발적 행동을 더 잘 시뮬레이션 할 수 있으며, 자발적 운율 모델링이 음성 합성 품질을 향상시키는 데 중요한 역할을 합니다.



### BEAF: Observing BEfore-AFter Changes to Evaluate Hallucination in Vision-language Models (https://arxiv.org/abs/2407.13442)
Comments:
          Accepted at ECCV 2024. [Project Pages] this https URL

- **What's New**: 이번 연구에서는 VLMs(Vision Language Models)의 신뢰성을 높이고 환각(hallucination) 문제를 해결하기 위해 새로운 평가 데이터셋 BEAF(BEfore-AFter hallucination dataset)를 소개했습니다. 또한, True Understanding (TU), IGnorance (IG), StuBbornness (SB), InDecision (ID)이라는 새로운 메트릭스를 제안하여 장면 변화를 기반으로 모델의 정확한 이해도를 평가합니다.

- **Technical Details**: BEAF 데이터셋은 이미지 편집 모델을 활용해 시각적 장면 정보를 조작하고, 질문 및 답변 쌍을 생성합니다. 기존의 질문-답변(QnA) 형식뿐만 아니라 이미지 변경 전후 모델의 응답 변화를 통해 VLMs의 정확한 인지 능력을 평가합니다. 새로 제안된 메트릭스는 다음과 같습니다: True Understanding (TU)는 모델이 조작 전후 올바른 답변을 제공하는지를 측정하고, IGnorance (IG)는 모델이 이미지를 일관되게 인지하는지를 평가하며, StuBbornness (SB)는 모델이 잘못된 동일한 답변을 고수하는지를 평가하고, InDecision (ID)는 물체가 변경되지 않더라도 답변이 달라지는 비일관성을 평가합니다.

- **Performance Highlights**: BEAF 데이터셋과 변화 인지 메트릭스를 활용한 평가 결과, 기존의 텍스트 축 평가 방식에서 비환각으로 간주된 답변이 실제로는 환각일 수 있음을 확인했습니다. 이로 인해 VLMs의 환각 문제를 보다 세밀하게 분석하고 개선할 수 있는 가능성을 제시했습니다.



### Correcting the Mythos of KL-Regularization: Direct Alignment without Overparameterization via Chi-squared Preference Optimization (https://arxiv.org/abs/2407.13399)
- **What's New**: 이번 연구에서는 언어 모델 정렬(alignment)의 새로운 알고리즘인 $\\chi^2$-Preference Optimization ($\\chi$PO)를 소개합니다. 기존의 정렬 방법은 과최적화(overoptimization) 문제로 인해 성능이 저하되거나 정체되는 경우가 많았으나, $\\chi$PO는 이러한 문제를 해소하면서도 높은 샘플 효율성을 보장합니다.

- **Technical Details**: $\\chi$PO는 Direct Preference Optimization (DPO)의 로그 링크 함수(logarithmic link function)를 변경하는 간단한 수정만으로 이루어집니다. 이는 $\\chi^2$-발산($\\chi^2$-divergence)을 이용하여 불확실성에 대해 비관적인 접근을 취하며, 이를 통해 과최적화를 방지할 수 있습니다. 또한 $\\chi$PO는 단일 정책 집중(single-policy concentrability)에 기반한 샘플 복잡성(sample-complexity) 보장을 제공합니다.

- **Performance Highlights**: $\\chi$PO는 간단한 구현으로도 강력한 이론적 보장을 제공하며, 이는 기존의 오프라인 정렬 알고리즘 중 첫 번째로 과최적화에 대해 견고함을 입증한 일반적인 알고리즘입니다. 이로 인해 더 높은 성능과 효율성을 기대할 수 있습니다.



### Low-Resourced Speech Recognition for Iu Mien Language via Weakly-Supervised Phoneme-based Multilingual Pre-training (https://arxiv.org/abs/2407.13292)
- **What's New**: 이번 연구는 중국의 소수 민족인 야오족의 주요 언어인 '이우몐(Iu Mien)' 언어를 자동 음성 인식(ASR) 시스템에 적용하는 방법을 탐구합니다. 이우몐 언어는 자음 데이터가 10시간도 채 안 되는 저자원 언어로, 독립적인 연구로는 최초로 'Whistle' 모델을 사전 훈련된 백본 모델로 사용해 이우몐 언어의 음성 인식 성능을 비교했습니다.

- **Technical Details**: 이 연구는 CommonVoice 데이터셋의 10개 언어(CV-Lang10)를 기반으로 미리 훈련된 세 가지 백본 모델을 사용했습니다. 이 세 가지 접근법은 자가지도 학습(self-supervised pre-training), 하위 단어 기반의 지도 학습(subword-based supervised pre-training), 그리고 국제 음성 기호(IPA)를 활용한 음소 기반 지도 학습(phoneme-based supervised pre-training)입니다. 특히 Whistle 모델은 약한 지도 학습 방식(weakly-supervised phoneme-based multilingual pre-training)으로 이우몐 언어의 음성 인식에 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 실험 결과에 따르면, 음소 지도 학습 방식이 하위 단어 지도 학습과 자가지도 학습에 비해 데이터 효율성이 더 높은 것으로 나타났습니다. Whistle 모델이 이우몐 언어의 음성 인식에서 가장 경쟁력 있는 결과를 보였으며, 이는 한정된 이우몐 데이터로 우수한 성능을 도출하는 데 중요한 역할을 했습니다.



### SciCode: A Research Coding Benchmark Curated by Scientists (https://arxiv.org/abs/2407.13168)
Comments:
          25 pages, 9 figures, 7 tables

- **What's New**: 최근 언어 모델 (Language Models, LMs)의 성능이 상당히 발전하면서 이들을 평가하기 위한 어려운 고품질의 현실적인 평가 방법을 개발하는 것이 점점 더 어려워지고 있습니다. 이를 해결하기 위해 과학 연구 문제를 해결하기 위한 코드를 생성하는 LMs의 능력을 평가하는 'SciCode'라는 새로운 벤치마크를 소개합니다. SciCode는 수학, 물리학, 화학, 생물학, 재료 과학 등 16가지 다양한 자연 과학 하위 분야의 과학자와 AI 연구자의 의견을 반영하여 만들어졌으며, 총 80개의 주 문제에서 파생된 338개의 하위 문제를 포함합니다.

- **Technical Details**: SciCode는 주요 문제가 여러 하위 문제로 나뉘어져 있어 지식 회상, 추론, 코드 생성 등을 포함합니다. 각 문제는 필요한 과학적 배경 정보를 제공하며, 과학자들이 주석을 단 황금 표준의 솔루션과 테스트 케이스를 포함합니다. 평가 설정에 따라 과학적 배경 정보를 제공하거나 이전 서브 문제의 솔루션을 조건부로 제공하는 옵션도 제공합니다.

- **Performance Highlights**: 최고 성능을 보인 모델인 Claude3.5-Sonnet은 가장 현실적인 설정에서 단 4.6%의 문제를 해결할 수 있었습니다. 다른 강력한 모델인 Claude3-Opus와 GPT-4o는 1.5%를 해결했고, 최고의 오픈 소스 모델인 Deepseek-Coder-v2는 3.1%를 해결했습니다. 과학자들이 작성한 배경 지식을 이용해 모든 모델의 성능이 향상되었지만, 배경 지식을 제공하더라도 최선의 모델이 해결할 수 있는 문제는 12.3%에 불과했습니다.



### TrialEnroll: Predicting Clinical Trial Enrollment Success with Deep & Cross Network and Large Language Models (https://arxiv.org/abs/2407.13115)
- **What's New**: 이 논문은 임상 시험 등록 성공을 예측하는 새로운 딥 & 크로스 네트워크를 개발하였습니다. 특히, 대형 언어 모델(LLM)-보강 텍스트 특징을 통해 시험 적합 조건의 의미 정보를 학습하여 등록 성공 여부를 예측합니다. 이 방법은 텍스트의 어떤 문장/단어가 예측에 큰 기여를 하는지 이해함으로써 해석 가능성을 제공합니다.

- **Technical Details**: 논문에서는 Deep & Cross Network를 이용하여 대형 언어 모델-보강 특징을 학습합니다(Large Language Model-enhanced Text Feature). 특히, 텍스트 데이터의 의미 정보를 학습하기 위해 계층적 주의 메커니즘(Hierarchical Attention Mechanism)을 설계하여 단어 및 문장 수준의 중요도를 종단간(end-to-end) 방식으로 학습합니다. 이 모델은 31,094개의 이진 분류 레이블을 가진 임상 시험 데이터를 사용하여 훈련됩니다.

- **Performance Highlights**: 제안된 방법은 기존의 잘 확립된 머신 러닝 방법들보다 0.7002 PR-AUC 점수로 우수한 성능을 보여줍니다. 이는 최상의 기존 방법보다 0.0229의 개선을 이루었으며, AI 예측을 이해하는 데 도움이 되는 바람직한 해석 가능성을 제공합니다.



### MetaSumPerceiver: Multimodal Multi-Document Evidence Summarization for Fact-Checking (https://arxiv.org/abs/2407.13089)
Comments:
          16 pages, 7 figures, The 62nd Annual Meeting of the Association for Computational Linguistics

- **What's New**: 이 논문에서는 다중 문서와 다중 모드(multimodal) 데이터로부터 팩트 체크에 유용한 주장별 요약을 생성하는 요약 모델을 소개합니다. 이 모델은 문서, 이미지, 주장을 입력으로 받아, 팩트 체크 작업을 돕는 것을 목표로 합니다. 새로운 강화 학습 강화 기반의 entailment 목표를 통해, 주장별 증거를 제공하는 요약을 생성하는 시스템을 설계했습니다.

- **Technical Details**: 이 모델은 무작위 길이의 여러 모드 데이터를 처리할 수 있는 dynamic perceiver-based 모델을 도입합니다. 모델 훈련을 위해, 제안된 시스템은 요약 모델로 perceiver 모델을 초기 훈련시키고, proxy reward 메커니즘을 사용해 요약의 정확성과 관련성을 보장하며 업데이트합니다. 또한, 다중 문서 팩트 체크를 지원하기 위해, 다양한 문서로부터 증거를 추출한 새로운 데이터셋(Multi-News-Fact-Checking)을 제공합니다.

- **Performance Highlights**: MOCHEG 데이터셋에서 주장 검증 작업의 SOTA 방식을 4.6% 증가시켰으며, 새로운 Multi-News-Fact-Checking 데이터셋에서도 강력한 성능을 보였습니다.



### Analysing the Public Discourse around OpenAI's Text-To-Video Model 'Sora' using Topic Modeling (https://arxiv.org/abs/2407.13071)
- **What's New**: OpenAI가 출시한 텍스트-투-비디오 (text-to-video) 모델인 '소라'(Sora)에 대한 온라인 커뮤니티의 반응을 분석한 연구 결과가 발표되었습니다. 연구는 2024년 2월 소라 발표 이후 두 달간 Reddit에서 수집된 1,827개의 댓글을 대상으로 진행되었습니다.

- **Technical Details**: 연구팀은 r/OpenAI, r/technology, r/singularity, r/vfx, r/ChatGPT 등 5개의 관련 서브레딧에서 댓글을 수집하고, 데이터를 전처리한 후 LDA(Latent Dirichlet Allocation)를 활용하여 주요 주제를 도출했습니다. 도출된 주요 주제는 1) 소라 토론에서의 AI 영향과 트렌드, 2) 소라에 대한 대중의 의견과 우려, 3) 소라를 이용한 예술적 표현과 비디오 제작, 4) 미디어 및 엔터테인먼트에서의 소라의 활용입니다.

- **Performance Highlights**: 연구 결과, 소라가 산업과 고용에 미치는 잠재적 영향, 대중의 감정 및 윤리적 우려, 창의적 응용 및 미디어와 엔터테인먼트 분야에서의 활용 사례 등 다양한 서사가 두드러졌습니다. 단, 본 연구는 특정한 기간 동안의 Reddit 데이터에 한정되었으나, 이러한 분석을 통해 신생 생성 AI(Generative AI) 기술에 대한 공공 인식을 이해하는 틀을 제공합니다.



### E2Vec: Feature Embedding with Temporal Information for Analyzing Student Actions in E-Book Systems (https://arxiv.org/abs/2407.13053)
Comments:
          Published in proceedings of the 17th Educational Data Mining Conference (EDM 2024)

- **What's New**: 본 연구는 디지털 교과서 시스템의 이벤트 스트림(EventStream) 데이터를 기반으로 학생의 학습 활동을 나타내는 새로운 특징 표현 방법인 E2Vec을 제안합니다. 이 방법은 작업 로그와 시간 간격을 문자열 시퀀스로 간주하여 시간 정보를 포함한 학습 활동 특징의 학생 벡터를 생성합니다.

- **Technical Details**: E2Vec은 워드 임베딩 기반의 특징 표현 방법으로, 각 학생의 작업 로그와 시간 간격을 문자열로 변환하여 학습 활동을 표현합니다. fastText를 사용해 305명의 학생들의 임베딩 벡터를 생성하였으며, 이를 통해 학습 활동의 순서 및 시간 간격을 고려한 좀 더 정교한 표현을 제공합니다.

- **Performance Highlights**: E2Vec은 잠재적으로 일반화 가능하며 성능이 좋다고 평가되었습니다. 특히, 위험 학생 예측 과제(at-risk detection task)에서 기존의 통계 기반 특징보다 더 높은 효과를 보였습니다.



### Pre-Trained Foundation Model representations to uncover Breathing patterns in Speech (https://arxiv.org/abs/2407.13035)
Comments:
          8 pages, 6 figures, BioKDD workshop paper

- **What's New**: 해당 연구는 인간의 음성 데이터를 사용하여 호흡 속도(Respiratory Rate, RR)를 ML(Machine Learning) 알고리즘으로 추정하는 방법을 제안합니다. 기존에는 RR을 측정하기 위해 특별한 장비가 필요했으나, 이 연구에서는 close-talking 마이크 장치를 사용하여 수집한 음성 데이터만으로 RR을 추정할 수 있음을 보여주고 있습니다.

- **Technical Details**: 연구에서는 convolutional long-short-term memory (Conv-LSTM) 네트워크를 제안하여 음성 신호로부터 호흡 시계열 데이터를 추정합니다. 또한, 사전 훈련된 Wav2Vec2와 같은 foundation model의 사전 학습된 표현(pre-trained representations)을 사용했습니다. 이 모델은 표준 멜 필터뱅크(MFB) 기반의 음향 특징과 비교하여 더 나은 성능을 보여줄 것으로 기대됩니다.

- **Performance Highlights**: 연구 결과, 사전 학습된 모델을 사용한 음성 데이터는 호흡 시계열 신호를 추정하는 데 매우 높은 상관관계(coefficient of correlation)를 나타냈으며, 루트 평균 제곱 오차(root-mean-squared error)가 낮았습니다. 제안된 모델은 평균 절대 오차(mean absolute error, MAE)가 약 1.6 breaths/min으로 낮은 수준의 정확성을 보였습니다.



### Retrieval-Enhanced Machine Learning: Synthesis and Opportunities (https://arxiv.org/abs/2407.12982)
- **What's New**: 이 논문은 다양한 기계학습(ML) 분야들에 정보 검색(IR) 기법을 통합하여 모델의 성능을 향상시키는 새로운 패러다임인 Retrieval-Enhanced Machine Learning (REML)을 소개합니다. 기존 문헌에서 일관된 표기법이 부족한 점을 해결하며, IR 연구와의 통합을 통해 다양한 학문 분야에서 REML 모델의 프레임워크를 구성하는 것을 목표로 합니다.

- **Technical Details**: REML 모델은 정보 검색 기법을 활용하여 외부 지식을 모델의 예측 과정에 통합함으로써 모델 성능을 향상시킵니다. 이를 통해 모델의 용량을 늘리지 않고도 지식을 기반으로 예측할 수 있습니다. 특히, 중추적 정보 검색(IR)의 연구와 현대 REML 연구 사이의 격차를 해소하기 위해 각 구성 요소를 조사하였습니다. 해당 연구는 NLP 외에도 컴퓨터 비전, 시계열 예측, 계산생물학 등 여러 분야에 적용될 수 있는 일관된 수학적 표기법을 제공합니다.

- **Performance Highlights**: REML 패러다임은 NLP, 기계 번역, 질의 응답, 사실 검증, 오픈 도메인 다이얼로그 시스템, 시계열 예측, 단백질 구조 예측 등 다양한 응용 분야에서 높은 적응력과 성능 향상을 보여줍니다. 또한, LangChain, LlamaIndex, DSPy 등의 프레임워크를 통해 산업 및 오픈 소스 커뮤니티에서도 널리 채택되어 왔습니다.



### Cross-Modal Augmentation for Few-Shot Multimodal Fake News Detection (https://arxiv.org/abs/2407.12880)
- **What's New**: 이번 논문은 크로스-모달 보강(Cross-Modal Augmentation, CMA)를 통해 몇 개의 샘플만으로도 가짜 뉴스를 탐지할 수 있는 혁신적인 모델을 소개합니다. 이 방법은 기존의 대규모 파라미터 튜닝이나 복잡한 신경망 학습 대신, 간단한 선형 검사법(linear probing)을 사용하여 가짜 뉴스를 다중모달리티(multimodal) 특징을 통해 효과적으로 탐지합니다.

- **Technical Details**: CMA는 크로스-모달 보강을 통해 텍스트 및 이미지를 동시에 처리하는 CLIP 모델을 사용해 텍스트와 시각적 특징을 추출합니다. 클래스 레이블은 보완적인 훈련 샘플로 활용되어 n-shot 분류 방식을 (n×z)-shot 문제로 변환합니다. 여기서 z는 보완적 특징의 수를 의미합니다. 모델은 간단한 선형 검사법을 사용해 모달리티별 그리고 융합된 다중모달리티 특징을 분류합니다.

- **Performance Highlights**: 이 논문에서 제안된 CMA 방법은 세 개의 벤치마크 데이터셋에서 SOTA(state-of-the-art) 성능을 보여주었으며, 11개의 대조 모델을 능가했습니다. 특히, 이전 방법들에 비해 훈련 가능한 파라미터 수와 학습 기간 면에서 매우 가볍습니다.



### Exploring the Use of Abusive Generative AI Models on Civita (https://arxiv.org/abs/2407.12876)
Comments:
          Accepted to ACM Multimedia 2024

- **What's New**: AI 생성 컨텐츠(Artificial Intelligence Generated Content, AIGC)가 디지털 이미지와 온라인 창작 커뮤니티에 미치는 영향을 탐구한 첫 번째 포괄적인 실증 연구가 발표되었습니다. 이 연구는 Civitai라는 AIGC 소셜 플랫폼을 중심으로 진행되었으며, 이 플랫폼은 사용자가 생성한 AI 모델을 공유하고 피드백을 받을 수 있는 공간을 제공합니다.

- **Technical Details**: Civitai는 사용자가 자신의 생성 AI 모델과 이를 통해 만든 이미지를 공유할 수 있는 소셜 네트워크 형태의 플랫폼입니다. 이 연구에서는 87,042개의 모델과 2,740,149개의 AI 생성 이미지를 포함하는 데이터를 수집하였으며, 이를 통해 생성된 컨텐츠의 테마와 NSFW(Not-Safe-For-Work) 컨텐츠의 존재를 라벨링 하였습니다.

- **Performance Highlights**: 본 연구의 주요 결과로는, 전체 모델 중 16.97%, 전체 이미지 중 72.05%가 NSFW 관련 태그를 가지고 있으며, 심지어 딥페이크(deepfake) 이미지의 32.98%가 NSFW 컨텐츠와 관련이 있다고 나타났습니다. 또한 NSFW 모델이 비 NSFW 모델에 비해 더 많은 다운로드와 조회수를 기록했으며, 창작자들 사이에서도 더 높은 인기도를 보였습니다. NSFW 컨텐츠 생성자가 그렇지 않은 경우에 비해 더 높은 사회적 네트워크 중심성을 가지며, 이는 이들이 더 많은 팔로우 링크를 보유하고 영향력 있는 사용자들과의 교류가 활발함을 나타냅니다.



### Large language models are good medical coders, if provided with tools (https://arxiv.org/abs/2407.12849)
Comments:
          7 pages, 1 figure, 2 tables

- **What's New**: 이번 연구는 ICD-10-CM 의료 코딩을 자동화하기 위한 새로운 두 단계 Retrieve-Rank 시스템을 제안하고 그 성능을 Vanilla Large Language Model(LLM) 접근법과 비교했습니다. 100개의 단일 용어 의료 조건 데이터셋에 대해 평가한 결과, Retrieve-Rank 시스템이 100%의 정확도를 달성하며 Vanilla LLM(GPT-3.5-turbo)의 6% 성능을 크게 능가하는 것으로 나타났습니다.

- **Technical Details**: 연구는 Colbert-V2 retriever와 GPT-3.5-turbo를 사용하여 의료 조건에 대한 ICD-10-CM 코드를 예측하는 새로운 두 단계 Retrieve-Rank 시스템을 구축했습니다. 첫 번째 단계에서는 ColBERT-V2를 사용해 가장 관련성 높은 ICD-10-CM 코드를 검색한 뒤, 두 번째 단계에서 GPT-3.5-turbo로 다시 랭킹을 매겨 최적의 코드를 선택했습니다. 이 시스템은 CDC 웹사이트에서 다운로드한 ICD-10-CM 데이터를 기반으로 훈련되었습니다.

- **Performance Highlights**: Retrieve-Rank 시스템은 100개의 샘플에 대해 100% 정확도로 ICD-10-CM 코드를 예측했습니다. 반면, 컨트롤로 사용된 GPT-3.5-turbo LLM은 단지 6%의 정확도를 보였습니다. Retrieve-Rank 시스템이 다양한 의학 용어를 정확히 처리할 수 있는 능력을 입증했습니다. 특히 해부학적 세부 사항과 복잡한 의료 조건에 대해 높은 정확성을 보였습니다.



### A Look Into Training Large Language Models on Next Generation Datacenters (https://arxiv.org/abs/2407.12819)
- **What's New**: 이 논문은 Microsoft가 1000억 달러 규모의 대형 데이터 센터를 구축하려는 계획을 기반으로 새로운 연구 방향을 모색합니다. 특히, 대형 언어 모델(LLMs)을 훈련시키기 위한 이러한 데이터 센터의 기술적 과제와 가능성을 분석합니다.

- **Technical Details**: 논문은 냉각 및 전력 요구 사항 때문에 단일 위치에 데이터 센터를 구축하는 것이 비현실적임을 발견했습니다. 50T에서 100T의 모델을 훈련할 수 있으며 분산 훈련(distributed training)을 위한 네트워킹 요구 사항을 분석합니다. 이 데이터 센터는 최신 GPU, 특히 Nvidia의 Blackwell B200 시리즈를 활용하며, 각 랙은 약 1MW의 전력을 소모합니다. 5GW의 전력 소모가 예상되는 단일 위치 데이터 센터가 불가능하기 때문에, 여러 위치에 데이터 센터를 분할하는 방식이 고려됩니다.

- **Performance Highlights**: 데이터 센터는 총 167만 개의 GPU를 포함하여 최대 2.9GW의 전력을 소모할 수 있으며, 33.4e21 FP4 성능을 제공합니다. 모델 훈련은 기존의 정사각형 Transformer 모델을 기준으로 하며, FP4 정밀도에서 모든 작업을 수행해 GPU 텐서 코어와 메모리를 최대한 활용합니다.



### Spectra: A Comprehensive Study of Ternary, Quantized, and FP16 Language Models (https://arxiv.org/abs/2407.12327)
Comments:
          32 pages, 12 figures, and 10 tables

- **What's New**: 최근 발표된 논문에서는 LLM(Large Language Model)에 대한 메모리 관련 문제를 해결하기 위해 사전 학습 이후에 양자화(post-training quantization)를 사용하고 있지만, 4-bit 이하로 감소할 경우 성능 저하가 발생하는 문제에 대해 논의합니다. 이에 대한 대안으로, 바이너리 또는 터너리 모델과 같은 저비트(bitwidth) 상태에서 직접 압축 모델을 학습시키는 방법을 제안합니다. Spectra LLM suite는 이러한 저비트 모델에 대한 이해를 돕기 위해 총 54개의 언어 모델을 공개합니다. 이 모델들은 99M에서 3.9B 파라미터까지 다양하며, 300B 토큰으로 학습되었습니다.

- **Technical Details**: Spectra LLM suite는 FloatLMs, post-training 양자화된 QuantLMs (3, 4, 6, 8 bit), 그리고 ternary LLMs (TriLMs)으로 구성됩니다. 특히, TriLMs는 터너리 언어 모델링을 위한 개선된 아키텍처로, 기존의 터너리 모델보다 뛰어난 성능을 보입니다. 예를 들어, TriLM 3.9B 모델은 메모리 크기(bit-wise)가 830M 크기의 half-precision FloatLM보다 작지만, commonsense reasoning과 knowledge benchmarks에서는 half-precision FloatLM 3.9B와 비슷한 성능을 보입니다.

- **Performance Highlights**: TriLM 3.9B 모델은 성능 측면에서 half-precision FloatLM 3.9B와 거의 동등하지만, 유독성 및 고정 관념에 있어서는 FloatLM 3.9B와 동일한 수준을 보임으로써 약간의 우려가 있습니다. 또한, validation splits와 웹 기반 코퍼스에서의 perplexity에서는 FloatLM에 비해 약간 뒤쳐지지만, Lambada와 PennTreeBank와 같은 소음이 적은 데이터셋에서는 더 나은 성능을 제공합니다. 이러한 저비트 모델에 대한 이해를 돕기 위해, 논문 저자들은 Spectra suite의 500개 이상의 중간 체크포인트를 공개했습니다.



New uploads on arXiv(cs.IR)

### A Comprehensive Review of Recommender Systems: Transitioning from Theory to Practic (https://arxiv.org/abs/2407.13699)
Comments:
          we quarterly update of this literature

- **What's New**: 최근 추천 시스템(Recommender Systems, RS)의 발전에 대한 종합적인 리뷰가 발표되었습니다. 이 연구는 2017년부터 2024년까지의 이론적 진보와 실제 응용 사례를 다루고 있습니다. 주목할만한 점은 전통적인 콘텐츠 기반(Content-based) 및 협업 필터링(Collaborative Filtering) 기법에서 딥러닝, 그래프 기반 모델(Graph-based Models), 강화 학습(Reinforcement Learning), 대형 언어 모델(Large Language Models, LLMs)의 발전까지 포괄적으로 조사했다는 점입니다.

- **Technical Details**: 추천 시스템의 발전은 전통적인 기술에서 최신 딥러닝 및 그래프 기반 모델, 강화 학습, 그리고 대형 언어 모델(LLMs)을 포함한 고급 방법으로 확대되었습니다. 또한, 컨텍스트 인식(Context-aware), 리뷰 기반(Review-based), 공정성 인식(Fairness-aware) 추천 시스템과 같은 특수화된 시스템도 논의되었습니다. 이 연구는 이론적 진보와 실제 응용을 연결하는 데 중점을 두고 있으며, 다양한 부문에서 발생하는 도전 과제에 대해 논의했습니다.

- **Performance Highlights**: Amazon, Netflix, Spotify와 같은 주요 기업들은 추천 시스템을 통합하여 사용자 만족도 향상과 매출 증대를 달성했습니다. 예를 들어, Amazon은 추천 시스템을 통해 매출의 35%를 창출한다고 보고했으며, Netflix는 자사의 추천 시스템 덕분에 약 337억 달러의 매출과 고객 유지에 크게 기여했습니다. 앞으로 추천 엔진의 글로벌 시장은 2023년부터 2030년까지 상당한 성장을 보일 것으로 예측됩니다.



### The Language of Infographics: Toward Understanding Conceptual Metaphor Use in Scientific Storytelling (https://arxiv.org/abs/2407.13416)
Comments:
          11 pages, 8 figures, 1 table, accepted to IEEE VIS 2024 Conference

- **What's New**: 이 논문은 인지 언어학의 접근법을 시각화 영역에 적용하여 과학적 인포그래픽에서 자주 사용되는 시각적 개념 은유의 패턴을 다루고 있습니다. 이는 복잡한 개념을 설명하는 데 필수적인 역할을 하며, 종종 직관에 기반해 사용되는데, 저자들은 이를 공식화된 프로세스로 전환하려고 합니다.

- **Technical Details**: 저자들은 기존의 과학 인포그래픽에서 시각적 구성 요소를 분해하여 시각적 개념 매핑(classification of the visual conceptual mappings)을 분류했습니다. 이 매핑의 발전을 네 가지 영역(생의학, 기후, 우주, 인류학)에서 수집한 데이터의 세부 분석을 통해 시연했습니다. 이러한 데이터를 바탕으로 시각적 개념 은유의 사용 패턴을 식별하고, 특정 개념 은유가 사용되는 이유를 명확히 하며, 과학적 인포그래픽에서의 시각적 은유 사용에 대한 전반적인 이해도를 높였습니다.

- **Performance Highlights**: 분석 결과, 다수의 시각적 개념 은유 사용 패턴 중에서도 존재론적(ontological) 및 방향적(orientational) 개념 은유가 복잡한 과학적 개념을 번역하는 데 가장 널리 적용되었습니다. 이러한 결과를 뒷받침하기 위해, 개별 인포그래픽을 시공간적(spatio-temporal) 규모에 배치하고 시각적 개념 은유의 분해를 보여주는 시각 탐구 도구를 개발했습니다.



### DCNv3: Towards Next Generation Deep Cross Network for CTR Prediction (https://arxiv.org/abs/2407.13349)
- **What's New**: 최신 Deep & Cross Network 모델들이 클릭 스루율(Click-Through Rate, CTR) 예측에서 중요한 역할을 하고 있지만 몇 가지 한계를 지니고 있습니다. 이를 해결하기 위해 본 논문에서는 새로운 세대의 Deep Cross Network(DCNv3)와 Shallow & Deep Cross Network(SDCNv3)를 제안합니다. 이 모델들은 DNN(Deep Neural Network)을 사용하지 않고 고차원 특징 상호작용을 명시적으로 모델링하여 해석 가능성을 보장하고, Self-Mask 연산을 통해 노이즈를 필터링하며 파라미터 수를 절반으로 줄입니다.

- **Technical Details**: DCNv3와 SDCNv3 모델은 고차원 특징 상호작용을 명시적으로 모델링하기 위해 새로운 Deep Crossing 방법을 도입하였으며, Self-Mask 연산을 통해 노이즈를 필터링하고 Cross Network의 파라미터 수를 절반으로 줄였습니다. 또한, 퓨전(fusion) 레이어에서는 Tri-BCE라는 간단하지만 효과적인 손실 가중치 계산 방법을 사용해 각기 다른 상호작용 방법에 적절한 감독 신호를 제공합니다. 이 방법은 전통적 CTR 예측 모델들이 가지는 문제점들을 해결하고자 설계되었습니다.

- **Performance Highlights**: 여섯 가지 데이터셋에서 수행된 종합 실험을 통해 DCNv3와 SDCNv3 모델의 효율성, 효과성, 해석 가능성을 입증했습니다. 본 논문의 모델들은 여러 CTR 예측 벤치마크에서 1위를 차지하는 성과를 거두었습니다.



### Semantic-aware Representation Learning for Homography Estimation (https://arxiv.org/abs/2407.13284)
- **What's New**: 새로운 검출기 없는(feature detector-free) 특징 매칭 방식인 SRMatcher가 제안되었습니다. 이 모델은 의미적 정보(semantic information)를 바탕으로 한 특징 표현 학습 프레임워크를 통해, 매칭 정확도를 높이는 접근 방식입니다.

- **Technical Details**: SRMatcher는 최근에 인기를 끌고 있는 비전 기초 모델(vision foundation models, VFMs)을 활용하여 세밀하고 풍부한 의미적 특징을 학습합니다. 특히 Semantic-aware Fusion Block(SFB)을 통해 교차 이미지 간 의미적 특징을 통합하여 매칭 품질을 개선합니다.

- **Performance Highlights**: SRMatcher는 여러 실제 데이터셋에서 최첨단(SOTA) 성능을 달성하였으며, 특히 HPatches 데이터셋에서 이전 SOTA 기법인 GeoFormer보다 AUC를 약 11% 증가시켰습니다. 또한, LoFTR과 같은 다른 매칭 기법에 플러그 앤 플레이로 활용 가능한 프레임워크로, 상당한 정밀도 향상을 가져왔습니다.



### Aligning Explanations for Recommendation with Rating and Feature via Maximizing Mutual Information (https://arxiv.org/abs/2407.13274)
Comments:
          this paper has been accepted by cikm2024, and the camera-ready version will be updated soon

- **What's New**: MMI(Maximizing Mutual Information) 프레임워크를 도입하여 추천 시스템의 설명 생성과 예상 평점 및 추천 아이템의 중요한 특징 간의 정합성을 개선했습니다. 이 프레임워크는 mutual information(MI)을 주요 정렬 측정값으로 사용하며, 강화를 통해 설명 생성을 최적화합니다. 이는 사용자의 의사결정을 돕고 사용자 만족도와 신뢰성을 높이는 데 도움이 됩니다.

- **Technical Details**: MMI 프레임워크는 텍스트 기반 설명과 예상 평점/아이템 특징 간의 정합성을 측정하기 위해 neural MI 추정기(Neural MI Estimator)를 사용합니다. 또한, 기존의 MLE로 훈련된 설명 생성 모델을 백본으로 활용하고 MI로부터 출력된 보상을 통해 강화 학습(RL) 기반으로 미세 조정합니다. 이 과정에서 KL 및 Entropy 보상을 통합하여 균형을 유지합니다.

- **Performance Highlights**: 세 가지 데이터셋에 대해 실험을 수행한 결과, MMI 프레임워크는 다양한 백본 모델을 향상시켜 예상 평점 및 아이템 특징과의 정합성 측면에서 기존 기준을 초과하는 성능을 보였습니다. 사용자 연구를 통해 MI로 강화된 설명이 실제 사용자 결정을 촉진하며, 다른 기준 모델보다 선호된다는 것을 검증했습니다.



### ROLeR: Effective Reward Shaping in Offline Reinforcement Learning for Recommender Systems (https://arxiv.org/abs/2407.13163)
Comments:
          CIKM 2024

- **What's New**: 본 논문에서는 추천 시스템에서 더욱 정확한 보상 모델과 불확실성 추정을 위해 오프라인 강화학습 (Offline Reinforcement Learning, RL) 방식의 새로운 접근법인 ROLeR를 제안합니다. 이 방법은 보상 모델의 정밀도를 높이고, 오프라인 데이터와 실제 온라인 데이터 간의 불일치를 해결하기 위해 비모수적 (non-parametric) 보상 학습 방법과 유연한 불확실성 페널티를 도입합니다.

- **Technical Details**: 제안된 ROLeR는 두 가지 주요 구성 요소를 포함합니다. 첫째, 비모수적 보상 학습 방법을 통해 보상 모델의 정확성을 개선하고, 둘째, 클러스터링 기반의 불확실성 추정을 통해 보상 추정의 유틸리티를 평가합니다. 사용자의 지표 특징과 역사적 데이터를 활용하여 비슷한 사용자 간의 피드백을 상호 추론할 수 있도록 합니다.

- **Performance Highlights**: 제안된 ROLeR 방법은 KuaiRand, KuaiEnv, Coat, Yahoo와 같은 네 가지 벤치마크 데이터셋에서 기존의 최첨단 방법들과 비교하여 우수한 성능을 입증하였습니다. 이러한 결과는 ROLeR가 모델 기반 오프라인 RL 추천 시스템에서 더 높은 보상 모델의 정확성과 일반화 용량을 제공함을 보여줍니다.



### MLSA4Rec: Mamba Combined with Low-Rank Decomposed Self-Attention for Sequential Recommendation (https://arxiv.org/abs/2407.13135)
- **What's New**: 전통적인 자기 주의 기반(self-attention-based) 순차 추천 모델의 한계를 극복하기 위해 Mamba와 저차원 분해된 자기 주의(low-rank decomposed self-attention)를 결합한 새로운 하이브리드 추천 프레임워크, MLSA4Rec이 제안되었습니다. 이 프레임워크는 사용자의 역사적 상호작용(sequence)의 길이에 대해 선형 복잡도(linear complexity)를 가집니다.

- **Technical Details**: MLSA4Rec은 효율적인 Mamba-LSA 상호작용 모듈을 설계합니다. 이 모듈은 저차원 분해된 자기 주의(LSA) 모듈을 도입하여 선형 복잡도와 함께 구조적 바이어스(structural bias)를 Mamba를 통해 주입합니다. LSA 모듈은 다른 관점에서 사용자 선호도를 분석하고, 게이트형 정보 전송 메커니즘(gated information transmission mechanism)을 통해 Mamba가 사용자 상호작용의 중요한 정보를 동적으로 집중하도록 안내합니다. 마지막으로, MLSA4Rec은 Mamba와 LSA 모듈이 정제한 사용자 선호도 정보를 결합하여 사용자의 다음 가능한 상호작용을 정확하게 예측합니다.

- **Performance Highlights**: 실험 결과, MLSA4Rec은 세 가지 실제 데이터셋에서 기존의 자기 주의 및 Mamba 기반 순차 추천 모델보다 추천 정확도에서 뛰어난 성능을 보였습니다. 이는 Mamba와 자기 주의의 결합이 큰 잠재력을 가짐을 입증합니다.



### Large language models are good medical coders, if provided with tools (https://arxiv.org/abs/2407.12849)
Comments:
          7 pages, 1 figure, 2 tables

- **What's New**: 이번 연구는 ICD-10-CM 의료 코딩을 자동화하기 위한 새로운 두 단계 Retrieve-Rank 시스템을 제안하고 그 성능을 Vanilla Large Language Model(LLM) 접근법과 비교했습니다. 100개의 단일 용어 의료 조건 데이터셋에 대해 평가한 결과, Retrieve-Rank 시스템이 100%의 정확도를 달성하며 Vanilla LLM(GPT-3.5-turbo)의 6% 성능을 크게 능가하는 것으로 나타났습니다.

- **Technical Details**: 연구는 Colbert-V2 retriever와 GPT-3.5-turbo를 사용하여 의료 조건에 대한 ICD-10-CM 코드를 예측하는 새로운 두 단계 Retrieve-Rank 시스템을 구축했습니다. 첫 번째 단계에서는 ColBERT-V2를 사용해 가장 관련성 높은 ICD-10-CM 코드를 검색한 뒤, 두 번째 단계에서 GPT-3.5-turbo로 다시 랭킹을 매겨 최적의 코드를 선택했습니다. 이 시스템은 CDC 웹사이트에서 다운로드한 ICD-10-CM 데이터를 기반으로 훈련되었습니다.

- **Performance Highlights**: Retrieve-Rank 시스템은 100개의 샘플에 대해 100% 정확도로 ICD-10-CM 코드를 예측했습니다. 반면, 컨트롤로 사용된 GPT-3.5-turbo LLM은 단지 6%의 정확도를 보였습니다. Retrieve-Rank 시스템이 다양한 의학 용어를 정확히 처리할 수 있는 능력을 입증했습니다. 특히 해부학적 세부 사항과 복잡한 의료 조건에 대해 높은 정확성을 보였습니다.



### CellularLint: A Systematic Approach to Identify Inconsistent Behavior in Cellular Network Specifications (https://arxiv.org/abs/2407.13742)
Comments:
          Accepted at USENIX Security 24

- **What's New**: 이번 연구에서는 4G 및 5G 표준의 일관성 검사 시스템인 CellularLint를 소개하였습니다. 이 시스템은 자연어 처리(NLP) 기법을 활용하여 프로토콜 문서 내의 모순을 반자동적으로 탐지합니다. 이러한 방법을 통해 CellularLint는 대규모 프로토콜 사양의 자동 분석을 크게 향상시킵니다.

- **Technical Details**: CellularLint는 도메인에 적응된 대형 언어 모델을 활용하여 몇 개의 샷 학습(few-shot learning) 메커니즘을 사용합니다. 문서를 크기별로 적절히 나누어 맥락과 이벤트를 보존한 후, 문서 간 및 문서 내 테스트 케이스/세그먼트를 생성합니다. 이를 통해 효율적으로 검색 공간을 축소하며 중요한 유사성 측정 기반의 데이터셋을 만듭니다. 이 시스템은 도메인 인식 주석 및 앙상블 결정 메커니즘을 사용하여 소수의 레이블 된 예제를 활용합니다.

- **Performance Highlights**: CellularLint는 4G와 5G 네트워크의 NAS 및 보안 사양에서 157개의 불일치 사항을 82.67%의 정확도로 발견했습니다. 이를 상용 장치 및 오픈소스 구현에서 검증한 결과, 이러한 불일치가 디자인 결정에 상당한 영향을 미치며, 개인정보 보호, 무결성, 가용성 및 상호 운용성 문제를 유발할 수 있는 것으로 나타났습니다.



### Compressed models are NOT miniature versions of large models (https://arxiv.org/abs/2407.13174)
Comments:
          Accepted at the 33rd ACM International Conference on Information and Knowledge Management (CIKM 2024) for the Short Research Paper track, 5 pages

- **What's New**: 이번 연구는 BERT-large와 그 압축된 버전의 모델 특성을 비교하여, 압축된 모델의 사용에서 예상치 못한 부작용이 발생할 수 있음을 강조합니다. 연구는 BERT-large 모델과 그 5가지 압축 모델을 대상으로 예측 오류, 데이터 표현, 데이터 분포, 적대적 공격에 대한 취약성 등 네 가지 특성을 비교했습니다. 결과적으로, 압축된 모델들은 상당히 다른 특성을 보였으며, 이는 초기 가정과 달랐습니다.

- **Technical Details**: 이 작업에서는 BERT-large와 그 압축 버전들(BERT-base, Distil-BERT, BERT-medium, BERT-mini, Tiny-BERT)을 사용하여 다양한 실험을 수행했습니다. SQUAD2 데이터셋으로 모델들을 핀튜닝(finetuning)했으며 Extractive Question Answering 태스크를 수행했습니다. Out-of-Distribution(분포 외) 탐지를 위해 NewsQA를 사용하고, IMDB 리뷰 데이터셋을 사용하여 BERT-ATTACK을 수행했습니다. 압축된 모델들은 각기 다른 예측 오류, 데이터 표현, 데이터 분포를 나타내었으며, 적대적 공격에 대해서도 서로 다르게 반응했습니다.

- **Performance Highlights**: 압축된 모델들은 예측 성능 저하뿐만 아니라, 네 가지 모델 특성 모두에서 큰 차이를 보였습니다. 예측 오류 및 데이터 표현 측면에서 압축된 모델들은 BERT-large와 매우 다르게 동작했으며, Out-of-Distribution 데이터에 대한 반응도 본래 모델과 차이가 컸습니다. 적대적 공격에서도 압축된 모델들 간의 반응도 다양하게 나타났습니다.



### Using LLMs to Investigate Correlations of Conversational Follow-up Queries with User Satisfaction (https://arxiv.org/abs/2407.13166)
Comments:
          Accepted to LLM4Eval @ SIGIR 2024 - The First Workshop on Large Language Models (LLMs) for Evaluation in Information Retrieval

- **What's New**: 최근 연구에서 대형 언어 모델(LLMs)을 이용한 대화형 검색 엔진이 사용자가 웹에서 정보를 검색하는 방식을 바꾸고 있습니다. 이 연구는 Naver Cue:라는 상업용 대화형 검색 엔진을 통해 사용자의 후속 질문 패턴을 분석하고, 이를 바탕으로 새로운 분류 체계를 제안합니다. 사용자의 대화 패턴은 기존의 쿼리 수정(literature on query reformulations)과 비교해 새로운 동기와 행동을 포함하고 있습니다.

- **Technical Details**: 이 연구에서는 250개의 대화 턴(conversational turns)에 대한 질적 분석을 통해 사용자 후속 쿼리 패턴을 크게 두 축으로 분류했습니다: (1) 대화를 계속하는 사용자의 동기 (7개), (2) 후속 쿼리의 행동 (11개). 이 분류 체계를 대규모 데이터 분석에 사용하기 위해 LLM 기반 분류기(73% 정확도)를 개발하였습니다. 이 분류기를 사용하여 실제 Naver Cue: 사용 로그에서 2,061개의 대화 튜플(conversational tuples)을 분석하였습니다.

- **Performance Highlights**: 연구 초기 결과에 따르면, 일부 불만족 신호는 'Clarifying Queries', 'Excluding Condition', 'Substituting Condition' 과 같은 후속 쿼리에 의해 나타납니다. 이 접근 방식은 대화형 검색 경험의 자동 평가와 사용자 경험 시뮬레이션의 토대를 제공할 수 있습니다. 이는 궁극적으로 검색 엔진의 맞춤형 정보 탐색 경로 지원과 사용자 경험 이해를 돕는데 기여할 것입니다.



### On Causally Disentangled State Representation Learning for Reinforcement Learning based Recommender Systems (https://arxiv.org/abs/2407.13091)
- **What's New**: 이 논문은 강화 학습 기반 추천 시스템(RLRS)에서 상태 표현(state representation)의 복잡성과 고차원성을 해결하기 위해 새로운 인과 접근법(Causal Approach)을 도입했습니다. 저자들은 Causal-Indispensable State Representations (CIDS)이라는 기법을 통해 상태 변수를 분해하고 행동에 직접적으로 영향을 미치는 상태 변수(Directly Action-Influenced State Variables, DAIS)와 행동 영향을 미치는 조상(Action-Influence Ancestors, AIA)을 식별합니다.

- **Technical Details**: CIDS는 조건부 상호 정보(Conditional Mutual Information)를 활용하여 인과 관계를 식별하고 중요한 상태 변수를 고차원의 상태 표현에서 분리합니다. 이를 통해 RLRS 내의 인과 관계를 이해하고 결정적으로 영향을 미치는 상태 변수를 이론적으로 증명하며, 실험적으로 CIDS의 효과를 검증합니다. 상태 공간은 사용자 데이터, 과거 상호작용, 항목 속성 및 문맥적 요소 등으로 구성되며, 각 상태 변수의 상관 관계를 인과 그래픽 모델(Causal Graphical Models)을 사용해 밝혀냅니다. 중요한 상태 변수는 데이터 관찰을 통해 학습되고, 이는 조건부 상호 정보를 통해 탐구됩니다.

- **Performance Highlights**: 제안된 CIDS 기법은 온라인 시뮬레이터와 오프라인 데이터셋을 통해 테스트되었으며, 기존 최신 방법들보다 높은 효율성과 성능을 보였습니다. 인과 관계를 활용한 상태 변수 생성을 통해 추천 정책 학습의 효율성을 크게 향상시켰음을 입증하였습니다.



### Analysing the Public Discourse around OpenAI's Text-To-Video Model 'Sora' using Topic Modeling (https://arxiv.org/abs/2407.13071)
- **What's New**: OpenAI가 출시한 텍스트-투-비디오 (text-to-video) 모델인 '소라'(Sora)에 대한 온라인 커뮤니티의 반응을 분석한 연구 결과가 발표되었습니다. 연구는 2024년 2월 소라 발표 이후 두 달간 Reddit에서 수집된 1,827개의 댓글을 대상으로 진행되었습니다.

- **Technical Details**: 연구팀은 r/OpenAI, r/technology, r/singularity, r/vfx, r/ChatGPT 등 5개의 관련 서브레딧에서 댓글을 수집하고, 데이터를 전처리한 후 LDA(Latent Dirichlet Allocation)를 활용하여 주요 주제를 도출했습니다. 도출된 주요 주제는 1) 소라 토론에서의 AI 영향과 트렌드, 2) 소라에 대한 대중의 의견과 우려, 3) 소라를 이용한 예술적 표현과 비디오 제작, 4) 미디어 및 엔터테인먼트에서의 소라의 활용입니다.

- **Performance Highlights**: 연구 결과, 소라가 산업과 고용에 미치는 잠재적 영향, 대중의 감정 및 윤리적 우려, 창의적 응용 및 미디어와 엔터테인먼트 분야에서의 활용 사례 등 다양한 서사가 두드러졌습니다. 단, 본 연구는 특정한 기간 동안의 Reddit 데이터에 한정되었으나, 이러한 분석을 통해 신생 생성 AI(Generative AI) 기술에 대한 공공 인식을 이해하는 틀을 제공합니다.



### Dynamic Sentiment Analysis with Local Large Language Models using Majority Voting: A Study on Factors Affecting Restaurant Evaluation (https://arxiv.org/abs/2407.13069)
Comments:
          This manuscript is under peer review

- **What's New**: 이번 연구는 중형 규모의 로컬 LLMs(large language models)를 사용하여 온라인 리뷰 데이터의 감정 분석을 수행할 때, 다수결 투표 메커니즘을 도입해 더 견고한 결과를 얻을 수 있음을 입증합니다. 이는 기존 연구가 대형 모델과 단일 시도를 통해 수행한 것보다 다수결 투표와 복수 시도를 통해 더 많은 변동과 재현성을 확보할 수 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 다수결 투표 메커니즘을 로컬에서 구동되는 LLMs에 적용하여 감정 분석 모델을 구축했습니다. 해당 모델은 사용자 리뷰 데이터를 통해 다양한 측면을 동적으로 분석할 수 있는 Aspect-based Sentiment Analysis (AbSA)을 목표로 하며, 임의의 측면(aspects) 설정이 가능합니다. 특히, 탐사적인 분석을 통해 하이퍼 파라미터와 정확도 사이의 관계를 조사했습니다.

- **Performance Highlights**: 중형 모델을 사용한 반복 접근법이 대형 모델을 단일 시도로 사용하는 방식보다 더 높은 변동성과 신뢰도(robustness)를 보였습니다. 이는 실제 인간 주석자가 다수결 투표를 통해 의견 불일치를 해결하는 방법과 유사한 방식입니다.



### Retrieval-Enhanced Machine Learning: Synthesis and Opportunities (https://arxiv.org/abs/2407.12982)
- **What's New**: 이 논문은 다양한 기계학습(ML) 분야들에 정보 검색(IR) 기법을 통합하여 모델의 성능을 향상시키는 새로운 패러다임인 Retrieval-Enhanced Machine Learning (REML)을 소개합니다. 기존 문헌에서 일관된 표기법이 부족한 점을 해결하며, IR 연구와의 통합을 통해 다양한 학문 분야에서 REML 모델의 프레임워크를 구성하는 것을 목표로 합니다.

- **Technical Details**: REML 모델은 정보 검색 기법을 활용하여 외부 지식을 모델의 예측 과정에 통합함으로써 모델 성능을 향상시킵니다. 이를 통해 모델의 용량을 늘리지 않고도 지식을 기반으로 예측할 수 있습니다. 특히, 중추적 정보 검색(IR)의 연구와 현대 REML 연구 사이의 격차를 해소하기 위해 각 구성 요소를 조사하였습니다. 해당 연구는 NLP 외에도 컴퓨터 비전, 시계열 예측, 계산생물학 등 여러 분야에 적용될 수 있는 일관된 수학적 표기법을 제공합니다.

- **Performance Highlights**: REML 패러다임은 NLP, 기계 번역, 질의 응답, 사실 검증, 오픈 도메인 다이얼로그 시스템, 시계열 예측, 단백질 구조 예측 등 다양한 응용 분야에서 높은 적응력과 성능 향상을 보여줍니다. 또한, LangChain, LlamaIndex, DSPy 등의 프레임워크를 통해 산업 및 오픈 소스 커뮤니티에서도 널리 채택되어 왔습니다.



### BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieva (https://arxiv.org/abs/2407.12883)
Comments:
          50 pages

- **What's New**: 기존 검색 벤치마크(benchmarks)는 주로 정보 탐색을 위한 쿼리(queries)로 구성되어 있으며, 키워드 또는 의미 기반 검색이 주로 사용됩니다. 그러나 복잡한 실제 쿼리의 경우 표면적 일치(surfaced form matching)를 넘어서는 심층적 추론이 필요합니다. 이를 위해 BRIGHT라는 새로운 텍스트 검색 벤치마크를 도입했습니다. 이 벤치마크는 복잡한 추론이 필요한 쿼리를 기반으로 만들어졌습니다.

- **Technical Details**: BRIGHT는 경제학, 심리학, 로봇공학, 소프트웨어 공학, 지구 과학 등 다양한 분야에서 수집된 1,398개의 실제 쿼리를 기반으로 구성되었습니다. 이를 통해 실제 상황에서 복잡한 검색 요구를 더 잘 반영할 수 있게 했습니다. BRIGHT는 자연스럽게 발생하거나 신중하게 선별된 인간 데이터를 사용하여 구축되었습니다.

- **Performance Highlights**: 최신 검색 모델조차도 BRIGHT에서 성능이 좋지 않았으며, MTEB 리더보드에서 최고 모델이 59.0 nDCG@10의 점수를 기록한 반면, BRIGHT에서는 nDCG@10 18.0을 기록했습니다. 그러나 대형 언어 모델(LLMs)의 Chain-of-Thought 추론을 통해 쿼리를 증강하면 성능이 최대 12.2 포인트 향상될 수 있음을 입증했습니다. 또한, 사전 훈련 중 벤치마크 데이터 누출에 대해 강력한 내성을 가지고 있음을 확인했습니다.



### Evaluation of RAG Metrics for Question Answering in the Telecom Domain (https://arxiv.org/abs/2407.12873)
Comments:
          Accepted for publication in ICML 2024 Workshop on Foundation Models in the Wild

- **What's New**: 이번 연구에서는 특수한 도메인에서의 질문 응답(QA) 작업을 수행하는 데 사용되는 Retrieval Augmented Generation (RAG)의 평가를 개선하려는 노력이 소개되었습니다. 특히, RAG 평가를 위한 기존의 RAGAS 라이브러리를 개조하여 중간 출력 값을 제공함으로써 평가 메트릭의 투명성을 높였습니다. 이 연구는 이러한 개선된 RAGAS를 통해 평가된 텔레콤 도메인의 응답들을 분석하고, 도메인 적응 및 파인 튜닝된 LLM이 RAG 메트릭에 미친 영향을 조사했습니다.

- **Technical Details**: RAG 시스템은 다수의 메트릭을 사용하여 생성된 응답의 정확성과 관련성을 평가합니다. 이 논문에서는 믿음성(Faithfulness), 문맥 관련성(Context Relevance), 응답 관련성(Answer Relevance), 응답 유사성(Answer Similarity), 사실 정확성(Factual Correctness), 응답 정확성(Answer Correctness) 등의 메트릭을 개선된 RAGAS 패키지를 통해 평가했습니다. 실험은 텔레콤 도메인의 질의 응답 데이터셋 (TeleQuAD)을 기반으로 수행되었으며, 다양한 LLM 모델(Mistral-7b, GPT-3.5)을 사용했습니다. 문서 정보는 3GPP Release 15 문서에서 추출되었으며, 앙상블 모델과 코사인 유사성을 이용해 최적의 문맥을 선택했습니다.

- **Performance Highlights**: RQ1 분석을 통해, 특정 메트릭이 텔레콤 도메인의 응답 평가에서 특히 유용함을 발견했습니다. 믿음성과 사실 정확성 메트릭은 전문가 평가와 높은 상관관계를 보여 종합적인 응답 정확성 평가에 효과적이었습니다. RQ2에서는 텔레콤 데이터에 도메인 적응된 LLM들이 더 나은 성능을 발휘함을 확인했습니다. RQ3에서는, 도메인 적응된 생성기 LLM이 문맥 오류에 대한 응답 생성 시 믿음성(Faithfulness) 점수가 낮아지는 경향을 보여줬으며, 이는 잘못된 문맥에서 벗어난 정보로부터 응답을 생성할 때 나타나는 현상임을 시사합니다.



### Automated Peer Reviewing in Paper SEA: Standardization, Evaluation, and Analysis (https://arxiv.org/abs/2407.12857)
- **What's New**: 최근 과학 논문의 급증으로 전통적인 리뷰 메커니즘이 과부하되어 출판물의 품질이 상이해지는 문제가 발생했습니다. 이를 해결하기 위해, 자동화된 논문 리뷰 프레임워크인 SEA를 소개합니다. SEA는 표준화, 평가, 분석이라는 세 가지 모듈로 구성되어 있으며, 각각 SEA-S, SEA-E, SEA-A 모델로 나타냅니다.

- **Technical Details**: SEA는 처음에 GPT-4을 사용해 여러 리뷰를 하나의 통일된 형식과 기준으로 통합하여 표준화합니다(S 모듈). 이후, 표준화된 데이터를 이용해 모델을 미세 조정하여 더 높은 품질의 리뷰를 생성합니다(E 모듈). 마지막으로, 논문의 내용과 리뷰 간의 일관성을 평가하기 위해 mismatch score라는 새로운 평가 메트릭을 도입하고, 자동 수정 전략을 활용해 일관성을 향상시킵니다(A 모듈).

- **Performance Highlights**: 8개의 다양한 데이터셋에서 수행한 실험 결과, SEA 프레임워크가 기존 방법들보다 논문의 품질, 포괄성, 일관성 면에서 뛰어난 리뷰를 생성하는 것으로 나타났습니다. 이 프레임워크는 저자들이 논문의 품질을 개선할 수 있는 유용한 인사이트를 제공합니다.



### Scaling Retrieval-Based Language Models with a Trillion-Token Datastor (https://arxiv.org/abs/2407.12854)
- **What's New**: 이 논문은 사전 학습된 언어 모델(LM)의 효율성과 성능 간의 트레이드오프를 예측하기 위해 학습 데이터와 파라미터 수에 대한 스케일링 법칙을 적용합니다. 이 연구는 추론 시점에서 사용되는 데이터 저장소(datastore)의 크기를 확장하는 것이 언어 모델링과 여러 다운스트림 작업에서 성능을 개선할 수 있음을 발견했습니다. MassiveDS라고 불리는 1.4조 토큰 규모의 데이터 저장소를 구축하고 이를 활용하여 소형 모델이 대형 모델을 능가할 수 있는지를 분석했습니다.

- **Technical Details**: MassiveDS는 일반 웹 데이터와 도메인 특화 데이터를 포함한 데이터 저장소로, 데이터 필터링, 서브샘플링(subsampling), 인덱싱, 문서 검색 등 다양한 데이터 처리 작업을 통해 구축되었습니다. 효율성을 높이기 위해 기존 파이프라인보다 컴퓨팅 비용을 10배 줄이는 효율적인 데이터 저장소 구축 파이프라인을 설계했습니다. 논문에서는 다른 저장소, 모델, 사전 학습 데이터 크기를 다양하게 조합하여 데이터 저장소 확장이 모델 성능에 미치는 영향을 체계적으로 평가합니다.

- **Performance Highlights**: 데이터 저장소 크기를 증가시키면 언어 모델링과 일부 다운스트림 작업이 일관되게 개선됩니다. 특히 지식 집약적인 작업에서 작은 리트리벌 기반 LM(retrieval-based LM)은 대형 LM-only 모델보다 우수한 성능을 보입니다. 또한, 동일한 학습 비용으로 대형 데이터 저장소를 활용한 리트리벌 기반 LMs는 더 나은 컴퓨팅 최적화를 달성할 수 있습니다. 전반적으로 데이터 저장소의 크기는 LM 효율성과 성능 트레이드오프의 필수 요소로 고려되어야 합니다.



### Automated Justification Production for Claim Veracity in Fact Checking: A Survey on Architectures and Approaches (https://arxiv.org/abs/2407.12853)
Comments:
          Accepted to ACL 2024 Main Conference

- **What's New**: 최근 자동 사실 확인(AFC) 분야의 연구들은 주장의 진실성을 예측하는 것 외에도 그 결과를 설명하는 자동화된 정당화 생성에 중점을 두고 있습니다. 이 논문은 이러한 최신 방법론들을 조사하고, 포괄적인 분류 체계를 제안하여 연구의 발전을 제시합니다.

- **Technical Details**: AFC는 여러 단계로 구성된 파이프라인 프로세스를 가지며, 그 중 하나가 정당화 생성 단계입니다. 이 단계에서는 신뢰할 수 있는 출처에서 관련 데이터를 가져와 주장의 진실성을 예측하고, 그 결과를 설명하는 정당화를 제공합니다. 주장의 진실성 예측은 증거와의 일치 정도에 따라 결정됩니다.

- **Performance Highlights**: 정당화 생성 방법론은 주로 다양한 기술 접근 방식을 사용하여 평가됩니다. 여기에는 주목 기반(attention-based), 지식 그래프 기반(knowledge graph-based), 요약 기반(summarization-based), 다중 단계 기반(multi-hop-based), 대형 언어 모델(LLM) 기반 접근 방식이 포함됩니다. 특히, Transformer 기반 아키텍처와 LLM의 발전 덕분에 최근 큰 진전이 있었습니다.




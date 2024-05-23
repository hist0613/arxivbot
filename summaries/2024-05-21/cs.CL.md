### MathBench: Evaluating the Theory and Application Proficiency of LLMs with a Hierarchical Mathematics Benchmark (https://arxiv.org/abs/2405.12209)
Comments:
          Project: this https URL

- **What's New**: 최근 대형 언어 모델(LLMs)의 수학적 능력이 크게 향상되었습니다. 그러나 기존의 수학 벤치마크인 GSM8k와 같은 평가 기준은 일차원적인 관점을 제공하여 LLM의 수학적 능력을 포괄적으로 평가하는 데 한계가 있습니다. 이런 문제를 해결하기 위해, 우리는 MathBench라는 새로운 벤치마크를 소개합니다. MathBench는 LLM의 수학적 능력을 엄격하게 평가하도록 설계되었습니다.

- **Technical Details**: MathBench는 광범위한 수학 분야를 다루며, 이론적 이해와 실질적 문제 해결 능력을 세밀하게 평가합니다. 벤치마크는 기본 산수에서 대학 수학에 이르기까지 다섯 개의 뚜렷한 단계로 구성되어 있으며, 각 단계는 다양한 지식 깊이를 평가하도록 설계되었습니다. 각 단계는 이론적 질문과 응용 문제를 포함하여 모델의 수학적 능력과 개념을 실제 시나리오에 적용할 수 있는지를 측정합니다.

- **Performance Highlights**: MathBench는 LLM의 수학적 능력 평가를 향상시키고, 지식 이해 수준과 문제 해결 능력에 대한 세부적인 시각을 제공하는 것을 목표로 합니다. 프로젝트는 이중 언어 문맥에서 진행되며, 관련 URL에서 공개되어 있습니다.



### Modeling citation worthiness by using attention-based bidirectional long short-term memory networks and interpretable models (https://arxiv.org/abs/2405.12206)
- **What's New**: 이 논문에서는 인용이 필요한 문장을 자동으로 탐지하는 새로운 딥러닝 모델을 제안합니다. 기존 연구들과 달리, 우리는 대규모 공개 데이터셋을 활용한 BiLSTM 이 유의미한 성능 향상을 가져올 수 있다고 가정합니다. 이를 위해 PubMed Open Access 세트를 바탕으로 하는 새로운 대규모 데이터셋 (PMOA-CITE)을 생성하였고, 이를 기반으로 높은 성능을 달성했습니다.

- **Technical Details**: 제안된 모델은 Bidirectional Long Short-Term Memory (BiLSTM) 네트워크와 주의 메커니즘(attention mechanism)을 결합하여 문맥 정보를 활용해 인용의 필요성을 탐지합니다. 새로운 PMOA-CITE 데이터셋은 이전 데이터셋보다 수 백 배 더 크며, 이를 통해 모델의 성능을 극대화했습니다. 또한, 모델이 해석 가능한 기능을 제공하여 인용에 사용되는 특정 언어를 밝힐 수 있습니다.

- **Performance Highlights**: 우리의 실험에서 제안된 모델은 표준 ACL-ARC 데이터셋에서 $F_{1}=0.507$의 성능을 기록하며 최신의 성능을 달성했습니다. 새로운 PMOA-CITE 데이터셋에서는 $F_{1}=0.856$의 높은 성능을 나타냈습니다. 또한 우리는 이 모델이 두 데이터셋 간의 전이 학습(transfer learning)에서도 효과적임을 보여주었습니다.



### CT-Eval: Benchmarking Chinese Text-to-Table Performance in Large Language Models (https://arxiv.org/abs/2405.12174)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 중국어 텍스트를 표로 변환하는 새로운 데이터셋인 CT-Eval을 제안합니다. 이는 높은 다중 언어 성능을 보여준 대형 언어 모델(LLM)을 중국어 문서에서도 활용할 수 있도록 설계되었습니다.

- **Technical Details**: CT-Eval 데이터셋은 다양한 도메인의 정보를 담기 위해 Baidu Baike라는 온라인 백과사전을 데이터 소스로 선택했습니다. 데이터의 다양성을 극대화하고 환각(데이터 내 불필요한 정보)을 최소화하기 위해 LLM을 먼저 학습시켜 환각이 포함된 데이터를 필터링하고, 이후 사람 검수자를 통해 검증 및 테스트 세트의 환각 데이터를 정리했습니다.

- **Performance Highlights**: CT-Eval을 사용한 실험 결과, GPT-4와 같은 대형 언어 모델은 제로샷(Zero-shot) 학습에서도 인간 판단과 비교하여 여전히 성능 격차가 존재했습니다. 그러나 오픈 소스 LLM들은 학습 세트를 통해 파인 튜닝 후, GPT-4를 능가하는 성능을 보여주었습니다. 이러한 결과는 CT-Eval이 LLM의 텍스트-투-테이블 성능을 크게 향상시킬 수 있는 귀중한 자원임을 시사합니다.



### Fennec: Fine-grained Language Model Evaluation and Correction Extended through Branching and Bridging (https://arxiv.org/abs/2405.12163)
- **What's New**: 새로운 논문에서 큰 언어 모델(Large Language Models, LLMs)의 평가에 있어 인간 의도에 맞추는 작업이 복잡하여 인간 평가에 의존해야 한다는 문제를 해결하고자 합니다. 이를 위해 오픈소스 큰 언어 모델을 평가자로 사용하는 Fennec이라는 단계별 평가 프레임워크를 제안합니다.

- **Technical Details**: Fennec는 두 가지 주요 연산을 포함합니다: 브랜칭(branching)과 브리징(bridging). 브랜칭은 평가 작업을 여러 차원과 세분성으로 분해하여 평가의 어려움을 줄입니다. 반면 브리징은 다양한 훈련 데이터 세트를 융합하여 평가 작업의 다양성을 높입니다. 이를 통해 모델의 적응성을 증가시킵니다. 이 방식은 체인 오브 씽크(Chain-of-Thought)과 브랜치-솔브-머지(Branch-Solve-Merge) 방법론과 유사합니다.

- **Performance Highlights**: 연구의 결과, 7B 모델은 여러 일반적으로 사용되는 벤치마크에서 일관성과 합의(Agreement and Consistency) 측면에서 더 큰 스케일의 오픈 소스 평가 모델들을 꾸준히 능가했습니다. 또한, GPT-4와 비슷한 성능을 발휘했습니다. Fennec는 여러 모델 응답의 수정(correction) 능력을 통해 응답의 품질을 높이고, MT-Bench에서 1-2 포인트를 증가시켰습니다.

- **Reproducibility**: Fennec의 코드는 GitHub에 공개되어 있어 재현 가능성을 보장합니다. 새롭게 공개된 오픈소스 훈련 데이터 세트를 사용하여 모델을 훈련할 수 있습니다.



### MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning (https://arxiv.org/abs/2405.12130)
Comments:
          Work in Progress

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM) 파인 튜닝을 위한 파라미터 효율적 방법인 로우-랭크 어댑테이션(low-rank adaptation, LoRA)의 한계와 이를 개선한 새로운 방법인 모라(MoRA)를 소개합니다. MoRA는 기존 LoRA의 저랭크(low-rank) 업데이트 방식을 고랭크(high-rank) 업데이트로 변경하면서도 동일한 수의 훈련 가능 파라미터를 유지합니다. 이를 통해 LLM이 새로운 지식을 효과적으로 학습하고 기억할 수 있게 합니다.

- **Technical Details**: MoRA는 저랭크 행렬 대신 정방행렬(square matrix) M을 사용하여 고랭크 업데이트를 달성합니다. 이를 위해 입력 차원을 줄이고 출력 차원을 늘리는 비파라미터 연산자(non-parameter operators)를 도입했습니다. 이러한 방식으로 MoRA는 LoRA와 마찬가지로 LLM에 통합될 수 있습니다. 예를 들어, LoRA는 8 랭크에서 4096x8과 8x4096의 저랭크 행렬을 사용한 반면, MoRA는 동일한 파라미터 수에서 256x256의 정방행렬을 사용하여 256 랭크의 업데이트를 달성합니다.

- **Performance Highlights**: MoRA는 메모리 집중형 작업(memory-intensive tasks)에서 LoRA보다 뛰어난 성능을 보였으며, 다른 작업에서는 LoRA와 비교할 만한 성능을 달성했습니다. 5가지 작업(메모리, 지시 조정(instruction tuning), 수학적 추론(mathematical reasoning), 지속적 사전 학습(continual pretraining), 사전 학습(pretraining))에 대해 평가가 수행되었으며, 특히 메모리 집중형 작업에서 우수한 성과를 보였습니다.



### Linguistic Structure from a Bottleneck on Sequential Information Processing (https://arxiv.org/abs/2405.12109)
- **What's New**: 이 논문은 자연언어의 체계성이 어떤 정보 처리 제약 하에서 효율적인 커뮤니케이션 원칙으로부터 어떻게 발생하는지를 연구합니다. 연구자들은 과잉 엔트로피(excess entropy)를 최소화하는 것이 자연언어처럼 체계적인 언어를 생성한다는 것을 시뮬레이션과 대규모 언어 코퍼스 연구를 통해 보여주었습니다.

- **Technical Details**: 과잉 엔트로피는 미래의 시퀀스를 예측하기 위해 과거의 시퀀스에서 필수적으로 저장해야 하는 최소 정보 양을 나타내는 통계적 복잡성의 지표입니다. 저자들은 시뮬레이션에서 과잉 엔트로피를 최소화하는 코드가 근사적으로 독립적인 구성 요소로 소스 분포를 분해하고, 그 구성 요소를 체계적으로 그리고 로컬리하게 표현함을 보였습니다. 또한, 여러 언어에서 음운론, 형태론, 구문론, 의미론 수준에서 낮은 과잉 엔트로피를 갖고 있음을 확인했습니다.

- **Performance Highlights**: 인간 언어는 통계적 분포에 대한 독립 성분 분석을 순차적으로 일반화하는 형태를 취합니다. 이는 인지적 부담을 최소화하면서도 통신의 표현력을 극대화하려는 방향으로 자연언어가 진화했음을 시사합니다. 이 연구는 인간 언어의 통계적 구조와 대수적 구조 간의 연결을 확립합니다.



### DOP: Diagnostic-Oriented Prompting for Large Language Models in Mathematical Correction (https://arxiv.org/abs/2405.12100)
- **What's New**: 새로운 논문은 수학 문제 해결 과정에서 발생하는 논리적 오류를 수정하는 '수학 세계 문제 수정(MWPC)'이라는 새로운 과제를 제시합니다. 현재 대부분의 연구가 문제 해결의 정확성에 중점을 두고 있지만, 이 논문은 오류를 식별하고 수정하는 것이 교육적으로 더 중요하다고 강조합니다.

- **Technical Details**: 이 논문에서는 대규모 언어 모델(LLMs)을 활용하여 (1) 수학적 논리와 오류 수정을 구분하고, (2) 수학적 오류 수정 능력을 향상시키기 위한 전략을 탐구합니다. 이를 위해 '진단 지향 프롬프팅(DOP)'이라는 새로운 방법을 제안하여 LLMs가 오류 수정에서 뛰어난 성능을 발휘할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, DOP 방법이 뛰어난 성능을 보여주었으며, 이는 수학 교육에서 모델의 오류 수정 능력을 향상시키는 데 큰 영향을 미친다는 것을 나타냅니다. 연구자들은 수학 교육에서 뛰어난 정정자가 뛰어난 이유자보다 더 많은 수요가 있다고 주장합니다.



### Distributional Semantics, Holism, and the Instability of Meaning (https://arxiv.org/abs/2405.12084)
- **What's New**: 이 논문은 기존 언어 모델이 사용하는 '분포 의미론적 접근 방식(distributional semantic approach)'이 내포하는 '분포 가설(distributional hypothesis)'과 관련된 불안정성 문제를 재검토합니다. 특히 불안정성이 언어적 의사소통에 미치는 영향을 분석합니다.

- **Technical Details**: 논문에서는 두 가지 유형의 불안정성을 구분합니다: 절대적 불안정성(absolute instability)과 차별적 불안정성(differential instability). 차별적 불안정성은 공간 내 포인트 간 상대적인 거리 변화로 정의됩니다. 이를 설명하기 위해 두 개의 모델을 만듭니다: 두 소설 텍스트를 사용한 토이 모델과 Wikipedia와 SEP 기사를 사용한 Word2vec 알고리즘 기반의 더 정교한 모델.

- **Performance Highlights**: 이 모델들은 콜로베이트된 코퍼스(corpus)가 커질수록 어떻게 변화하는지를 통해 두 가지 형태의 불안정성을 시연합니다. 이러한 분석을 통해 차별적 불안정성이 의사소통에 실제로 어떤 영향을 미칠 수 있는지를 밝혀냅니다.



### Selective Annotation via Data Allocation: These Data Should Be Triaged to Experts for Annotation Rather Than the Mod (https://arxiv.org/abs/2405.12081)
Comments:
          18 pages, 4 figures

- **What's New**: 예산이 제한된 상황에서 고품질 주석을 얻기 위해 반자동 주석 방법(Semi-automatic annotation methods)을 많이 사용합니다. 기존 방법들은 전문가가 주석을 달 데이터 선택에 집중하지만, 나머지 데이터는 무분별하게 모델에 할당되어 비효율성이 발생할 수 있습니다. 이를 해결하기 위해 SANT(Selective Annotation)라는 새로운 프레임워크를 제안합니다. 이는 Error-Aware Triage와 Bi-weighting 메커니즘을 통해 데이터의 중요도에 따라 전문가와 모델 할당을 최적화합니다.

- **Technical Details**: SANT는 두 가지 주요 메커니즘으로 구성됩니다. 첫째, AL(Active Learning) 기반 메커니즘을 통해 모델 예측 능력을 높일 수 있는 데이터에 높은 가중치를 부여합니다. 둘째, Error-Aware Triage(EAT) 메커니즘으로 데이터의 '어려움(hardness)'을 평가하여, 어려운 데이터를 전문가에게 할당하고 쉬운 데이터는 모델에게 할당합니다. 이로 인해 모델의 예측 능력을 보장하면서도 어려운 데이터 주석을 전문가가 처리하도록 효율적으로 예산을 사용할 수 있습니다.

- **Performance Highlights**: 실험 결과, SANT는 다양한 주석 예산에서 일관되게 주석 품질을 향상시켰고, 기존 반자동 주석 방법들에 비해 평균 0.50%, 4.86%, 4.54% 더 높은 성능을 보였습니다. 또한, 최근 강력한 자동 주석 모델로 각광받는 ChatGPT와 비교해서도 SANT가 더 높은 주석 품질을 보여주었습니다.



### CLAMBER: A Benchmark of Identifying and Clarifying Ambiguous Information Needs in Large Language Models (https://arxiv.org/abs/2405.12063)
Comments:
          Accepted to ACL 2024

- **What's New**: Large Language Models (LLMs)을 평가하기 위한 새로운 벤치마크 CLAMBER가 소개되었습니다. 이 벤치마크는 사용자의 모호한 질문을 식별하고 명확히 하는 LLM의 능력을 종합적으로 평가합니다.

- **Technical Details**: CLAMBER는 체계적인 분류(taxonomy)를 기반으로 약 12,000개의 고품질 데이터를 구성하여 LLM의 강점, 약점 및 잠재적 리스크를 평가합니다. 또한, 체인 오브 소트(chain-of-thought) 및 few-shot 프롬프팅 같은 기법을 통해 LLM이 모호성을 식별하고 해결하려는 시도의 한계를 조사합니다.

- **Performance Highlights**: CLAMBER 벤치마크는 ChatGPT(OpenAI, 2022)가 다른 소규모 LLM보다 모호한 질문을 식별하고 명확히 하는 데 뛰어난 성능을 보인다는 것을 발견했습니다. 그러나 현재의 LLM은 여전히 고품질의 명확화 질문을 생성하는 데 어려움을 겪고 있으며, 내재된 지식을 정확하게 활용하지 못하는 등 해결할 점이 많습니다.



### STYLE: Improving Domain Transferability of Asking Clarification Questions in Large Language Model Powered Conversational Agents (https://arxiv.org/abs/2405.12059)
Comments:
          Accepted to Findings of ACL 2024

- **What's New**: 이번 연구는 대화형 검색 엔진에서 도메인 간 성능 전이를 개선하기 위한 새로운 방법, 'Style'을 소개합니다. 이 방법은 도메인 간 전략 전이(Domain Transferability)에 효과적이며, 네 가지 보지 못한 도메인에서 평균 약 10%의 검색 성능 향상을 달성했습니다.

- **Technical Details**: Style은 두 가지 주요 구성 요소로 구성됩니다. 하나는 도메인 불변 전략 플래너(Domain-Invariant Strategy Planner, DISP)로, 대화형 검색 엔진에서 도메인 불변 정보를 추출하여 도메인 간 표현의 불일치를 완화하고, 다른 하나는 다중 도메인 훈련 패러다임(Multi-Domain Training, MDT)으로, 다양한 도메인에서 훈련하여 DISP의 도메인 전이 가능성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, Style은 네 가지 데이터셋(전자상거래, 영화, 도서 등)에서 기존의 LLM 기반 방법들을 능가하며 평균 약 10%의 성능 향상을 보여주었습니다. 추가 분석에서는 Style이 도메인마다 맞춤형 전략을 다양하게 적용함으로써 효과적이라는 점이 드러났습니다.



### Unveiling factors influencing judgment variation in Sentiment Analysis with Natural Language Processing and Statistics (https://arxiv.org/abs/2405.12055)
Comments:
          Accepted manuscript to be published in PLoS One

- **What's New**: 이번 연구에서는 스페인어 TripAdvisor 리뷰 데이터를 중심으로 군중 소싱된 폴라리티 판단의 변동 원인을 탐구합니다. 특히 사전 의미(POS), 감성 단어(예: 'tasty'), 그리고 중립 단어(예: 'ok')가 폴라리티 판단에 미치는 영향을 조사합니다.

- **Technical Details**: 연구 방법론은 한 단어 제목을 사용하여 단어의 폴라리티 변동성을 연구합니다. 평균 평등에 대한 통계 테스트는 관심 있는 단어 그룹에 대해 수행됩니다. 결과적으로, 형용사가 한 단어 제목에서 다른 단어 유형이나 POS에 비해 낮은 판단 변동성을 보였으며, 감성 단어도 낮은 판단 변동에 기여함이 밝혀졌습니다. 반면 중립 단어는 예상대로 높은 판단 변동성과 연관이 있었습니다.

- **Performance Highlights**: 한 단어 제목을 사용할 때 감성 단어가 폴라리티 연구에 중요한 역할을 하며, 긴 제목에서는 이러한 효과가 항상 재현되지 않는다는 점을 발견했습니다. 이는 긴 제목이 폴라리티 변동성을 테스트하는 최선의 데이터 소스가 아닐 수 있음을 시사합니다.



### Can AI Relate: Testing Large Language Model Response for Mental Health Suppor (https://arxiv.org/abs/2405.12021)
Comments:
          Under review

- **What's New**: 대형 언어 모델(LLMs)이 임상 실험으로 병원 시스템에 도입되고 있으며, 정신 건강 치료에서 LLM 기반 챗봇을 사용하는 것이 제안되고 있습니다. 이 연구는 LLM의 응답이 정신 건강 치료 자동화를 위한 윤리적이고 실현 가능한 경로인지 평가하는 프레임워크를 개발했습니다. 이를 통해 환자의 하위 집단 간의 응답 격차를 조사하고, 공정한 치료 제공을 위한 안전 지침을 제안했습니다.

- **Technical Details**: 연구팀은 훈련된 임상심리학자와 심리학 연구에 기초한 자동 품질 관리 지표를 사용해, 동료 간 지원자와 최첨단 LLM(GPT-4)이 제공하는 응답을 비교했습니다. 연구 결과, GPT-4는 은연중에나 명시적으로 환자의 인구 통계를 추론할 수 있으며, 흑인 환자에게 제공된 응답이 다른 인구 통계 그룹보다 감정적 공감도가 낮다고 밝혔습니다.

- **Performance Highlights**: GPT-4 응답이 동료 지원자보다 공감적이고, 환자의 긍정적 행동 변화를 유도하는 데 있어 48% 더 우수했습니다. 그러나 흑인 및 아시아인 환자에게 제공된 응답의 감정적 공감도가 백인 환자에 비해 2%-17% 낮았습니다. 이는 LLM의 응답 생성 방식이 응답 품질에 크게 영향을 미칠 수 있음을 시사합니다.



### A review on the use of large language models as virtual tutors (https://arxiv.org/abs/2405.11983)
- **What's New**: 최신 Transformer 아키텍처가 NLP의 장기 의존성을 관리하는 데 기여하면서 LLMs의 혁신이 교육 분야에서도 큰 주목을 받고 있습니다. 이 리뷰는 교육 자료를 생성하고 평가하기 위해 설계된 LLMs 솔루션을 종합적으로 다루며, 학생과 교사가 설계나 실험 계획에 참여하는 경우에 초점을 맞추고 있습니다. 이는 교육용으로 설계된 LLMs의 첫 리뷰입니다.

- **Technical Details**: Transformer 아키텍처는 LLMs의 기초를 이루며, GPT-3 및 BERT와 같은 모델이 대표적입니다. 이러한 모델들은 대규모 텍스트 데이터로 사전 훈련(pre-training)되고 특정 작업에 맞춰 미세 조정(fine-tuning)되어 인간과 유사한 발화(human-like utterances)를 생성합니다. 특히 GPT-4는 보다 자연스러운 상호작용과 실시간 복잡한 질문에 답변할 수 있는 능력을 가지고 있습니다.

- **Performance Highlights**: LLMs는 자동화된 질문 생성과 같은 방식으로 가상 튜터 역할을 하여 학생 평가를 돕습니다. 또한 자동 답안 채점, 설명 생성, 문제 해결 등의 교육적 작업에서 뛰어난 성능을 발휘합니다. 특히 텍스트 요약 기능은 학생의 추상화 능력을 향상시키는 데 큰 역할을 합니다.



### Multiple-Choice Questions are Efficient and Robust LLM Evaluators (https://arxiv.org/abs/2405.11966)
Comments:
          data at this https URL

- **What's New**: 본 논문에서는 새로운 다지선다형(MC: Multiple-Choice) 데이터셋인 GSM-MC와 MATH-MC를 소개합니다. 이 데이터셋은 GSM8K와 MATH에서 도출된 정답 및 오답을 통해 생성되었습니다. 이를 통해 MC 형식의 평가가 원본 형식과 강하게 상관되어 있으며, 옵션 순서에 대한 강력한 견고성을 보여줍니다. 또, 평가 시간이 최대 30배까지 단축됩니다. 또한, 새로운 PythonIO 데이터셋도 소개됩니다. 이는 HumanEval과 MBPP에서 추출된 프로그램 출력 예측 데이터셋입니다.

- **Technical Details**: GSM-MC와 MATH-MC는 GSM8K와 MATH의 데이터를 활용하여 각각 다지선다형 형식으로 변환한 데이터셋입니다. 본 논문에서는 50개 이상의 오픈소스 모델로부터 오답을 수집하여 문제마다의 오답 선택지를 구성했고, 데이터와 코드가 공개되어 있습니다. 또한 PythonIO는 HumanEval과 MBPP에서 추출된 모델 코드 생성 결과를 예측하는 다지선다형 형식의 새로운 데이터셋입니다.

- **Performance Highlights**: LLM(대형 언어 모델)들은 GSM-MC와 같은 MC 형식의 데이터셋에서 원본 형식과 비교하여 높은 상관성을 보였고, 선택지와 옵션 순서에 대해 강력한 견고성을 나타냈습니다. MC 형식으로 변환된 데이터셋은 자원 소모를 크게 줄여주는 이점이 있으며, 예를 들어 Qwen1.5-32B 모델의 경우 GSM8K 원본 데이터셋 평가 시 7시간이 소요된 반면, MC 형식에서는 13분 만에 평가가 완료되었습니다.



### WisPerMed at BioLaySumm: Adapting Autoregressive Large Language Models for Lay Summarization of Scientific Articles (https://arxiv.org/abs/2405.11950)
Comments:
          4 pages, 6 figure, 3 tables, submitted to: BIONLP 2024 and Shared Tasks @ ACL 2024

- **What's New**: WisPerMed 팀은 BioLaySumm2024 Shared Task에서 자동 요약을 통한 생의학 도메인 내 과학 출판물의 접근성을 개선하기 위해 큰 언어 모델(Large Language Models, LLMs)인 BioMistral과 Llama3을 훈련시켜 복잡한 과학 텍스트로부터 요약을 생성했습니다. 다양한 접근법을 통해 성능을 최적화했으며, 특히 훈련을 통해 일반적으로 최고의 성능을 발휘했습니다.

- **Technical Details**: 이 연구는 지침 조정(Instruct Tuning), 몇-샷 학습(Few-shot Learning), 동적 전문가 선택(Dynamic Expert Selection) 등의 기술을 사용하여 LLM을 미세조정했습니다. BioMistral-7B-DARE 모델과 Llama3-70B-Instruct 모델을 한 에폭 동안 Quantized Low-Rank Adaptation (QLoRA) 방식을 활용해 eLife 및 PLOS 데이터셋에서 미세조정했습니다. 미세조정에서는 Mistral 및 Llama3 지침 템플릿을 사용하여 텍스트를 구조화했습니다.

- **Performance Highlights**: WisPerMed 팀은 54개의 참가 팀 중에서 4위를 차지했으며, 가독성, 사실성, 관련성에서 측정된 성능을 통해 약 1.5 퍼센트 포인트 차이로 1위 팀과 경쟁이 가능했습니다. BioM 모델은 검증 세트에 대한 몇-샷 설정에서 우수한 성능을 발휘했고, 모든 프롬프트 변형 실험에서 사용되었습니다. 또한, 초기 프롬프트를 통한 몇-샷 학습의 효과로 인해, 프롬프트 변형이 테스트 세트에서도 효과적이었습니다.



### FAME-MT Dataset: Formality Awareness Made Easy for Machine Translation Purposes (https://arxiv.org/abs/2405.11942)
Comments:
          Accepted at EAMT 2024

- **What's New**: 이번에 발표된 FAME-MT 데이터셋은 자동 번역에서 문체(Formality) 수준을 조절할 수 있도록 하는 대규모 데이터셋입니다. 총 112개의 유럽어 언어쌍을 대상으로 1,120만 개의 문장을 포함하고 있으며, 이는 문장 번역 시 정중함(존대어) 수준을 선택할 수 있도록 돕습니다.

- **Technical Details**: 이 데이터셋은 15개의 유럽어 소스 언어와 8개의 유럽어 타겟 언어로 이루어져 있으며, 각 번역 문장은 형식적(Formal) 또는 비형식적(Informal)으로 분류됩니다. 데이터셋 생성 과정에서는 광범위한 평행 코퍼스(Parallel Corpus)를 이용했고, 이를 통해 다양한 문장 포맷을 수집했습니다. 이 데이터셋은 Marian 등과 같은 기계 번역 모델을 미세 조정(Fine-Tuning)하는 데 사용될 수 있습니다.

- **Performance Highlights**: FAME-MT 데이터셋은 112개의 유럽어 언어쌍에 대해 100,000개의 형식성 주석이 달린 번역 예시를 제공합니다. 이는 기존 데이터셋보다 훨씬 더 광범위하고, 유럽 언어쌍 커버리지 측면에서 가장 크며, 특히 CoCoA-MT나 기존의 영어 기반 데이터셋들보다도 많은 내용을 포함합니다.



### Biomedical Entity Linking for Dutch: Fine-tuning a Self-alignment BERT Model on an Automatically Generated Wikipedia Corpus (https://arxiv.org/abs/2405.11941)
Comments:
          Published in the CL4Health workshop on Patient-oriented language processing @ LREC-COLING 2024

- **What's New**: 이 논문은 네덜란드어로 된 최초의 생물 의학 엔터티 링크(Biomedical Entity Linking) 모델을 평가하였습니다. UMLS와 네덜란드어 SNOMED에서 추출한 네덜란드어 생물 의학 온톨로지를 바탕으로 SapBERT 모델을 사용하여 링크 작업을 수행했습니다.

- **Technical Details**: 기본 모델로 SapBERT(Liu et al., 2021)를 사용하고, 네덜란드어로 된 청소된 UMLS 샘플에서 자가 정렬 프리트레이닝(self-alignment pretraining)을 수행하였습니다. 이후 Wikipedia에서 추출한 온톨로지 연결된 네덜란드어 생물 의학 엔터티를 포함한 데이터셋으로 세밀 조정을 진행하고, Mantra GSC 코퍼스의 네덜란드어 부분에서 평가를 수행하였습니다.

- **Performance Highlights**: 모델은 Mantra GSC 코퍼스의 네덜란드어 부분에서 54.7% 분류 정확도와 69.8% 1-거리 정확도를 기록했습니다. 또한, 환자 지원 포럼 데이터에 대한 사례 연구에서, 이전 엔터티 인식 단계의 품질이 한계가 있었지만, 추출된 엔터티 중 약 65%가 올바른 온톨로지 개념에 연결되었습니다.



### Chasing COMET: Leveraging Minimum Bayes Risk Decoding for Self-Improving Machine Translation (https://arxiv.org/abs/2405.11937)
Comments:
          EAMT 2024

- **What's New**: 이번 논문은 기계 번역(MT) 모델의 자기 개선을 위해 Minimum Bayes Risk (MBR) 디코딩을 탐구합니다. 특히 도메인 적응과 저자원 언어에 초점을 맞추고 있습니다. COMET를 MBR 유틸리티 메트릭으로 사용하여 번역의 재순위를 통해 인간의 선호와 더 잘 일치하도록 합니다. 이 접근 법의 반복적인 적용과 언어별 MBR 유틸리티 메트릭의 필요성을 논의합니다. 모든 조사된 언어 쌍에서 번역 품질이 크게 향상되었음을 보여주는 결과를 도출했습니다.

- **Technical Details**: MBR 디코딩은 모델의 예측을 활용해 후보 번역 중 가장 좋은 번역을 선택함으로써 전체 번역 품질을 개선합니다. COMET를 유틸리티 함수로 사용해 MT 모델이 생성한 후보 번역을 재순위합니다. 이 방법은 원본 언어의 단일 언어 데이터로부터 합성 병렬 데이터 세트를 생성하여 모델의 자기 개선을 돕습니다. 본 연구는 영어-독일어(고자원), 체코-우크라이나어(저자원), 영어-하우사어(저자원) 3개 언어 쌍에서의 MBR 디코딩 자기 개선의 유효성을 조사합니다.

- **Performance Highlights**: 모든 언어 쌍에서 번역 품질이 크게 개선되었습니다. 특히 영어-독일어 번역의 생물의학 분야 적용 및 체코-우크라이나어와 같은 저자원 언어 쌍에 대해서도 유효함을 보였습니다. 또한, COMET와 아프리카 언어에 특정한 AfriCOMET 메트릭을 비교함으로써 번역 품질에 대한 영향을 분석했습니다.



### ARAIDA: Analogical Reasoning-Augmented Interactive Data Annotation (https://arxiv.org/abs/2405.11912)
Comments:
          Accepted to ACL 2024

- **What's New**: Araida는 제한된 데이터 주석 환경에서 주석 정확도를 향상시키고 인간의 교정 노력을 줄이는 새로운 접근법입니다. 이 방식은 기존의 상호작용 데이터 주석 모델과 k-nearest neighbors (KNN) 모델을 동적으로 조정하는 오류 인식 통합 전략을 사용하여 제안한 것입니다.

- **Technical Details**: Araida는 KNN 및 주석 모델의 예측을 결합하는 오류 인식 통합 전략을 도입합니다. 주석 모델의 예측이 부정확하다고 판단될 때 KNN의 예측에 더 많은 비중을 두는 방식입니다. 이 방식은 모델이 예측을 향상시키기 위해 이전에 사람들이 주석을 달았던 예제를 검색하여 사용합니다. KNN 모듈은 법레벨과 문장레벨 주석 작업에서 시험되었으며, 결과적으로 기존 주석 모델보다 일관되게 성능을 개선했습니다.

- **Performance Highlights**: Araida는 다양한 주석 작업과 모델에서 인간 교정 노력을 평균 11.02% 감소시켰습니다. 이 개선은 KNN 모듈의 Few-shot 학습 기능과 오류 인식 통합 전략 덕분입니다. 이 전략은 상호 보완적인 주석을 효과적으로 조정하는 데 기여합니다. Araida는 다양한 주석 모델과 결합하여 유연성을 입증했습니다.



### A Constraint-Enforcing Reward for Adversarial Attacks on Text Classifiers (https://arxiv.org/abs/2405.11904)
- **What's New**: 최근의 연구는 텍스트 분류기에 대한 적대적 예시(adversarial examples)를 생성하는 새로운 접근법을 제안합니다. 기존에는 연속적인 조합 최적화 문제를 정의하고 해결하는 방식으로 적대적 예시를 찾았지만, 이 논문에서는 사전 학습된 언어 모델을 미세 조정하여 적대적 예시를 직접 생성하는 방법을 탐구합니다.

- **Technical Details**: 이 연구는 사전 학습된 인코더-디코더(encoder-decoder) 패러프레이즈 모델을 사용하여 다양한 적대적 예시를 생성하는 것을 목표로 합니다. 이를 위해 강화 학습 알고리즘(RL, Reinforcement Learning)인 REINFORCE를 기반으로 하는 정책 경사(Policy Gradient) 알고리즘을 사용합니다. 또한, 오리지널 예시와 비교하여 텍스트의 의미와 문법적 요소를 유지하면서도 타겟 모델의 예측을 잘못되게 만드는 것을 목표로 하는 제약을 강화할 수 있는 보상 함수(constraint-enforcing reward function)를 제안합니다.

- **Performance Highlights**: 제안된 접근법은 두 개의 텍스트 분류 데이터셋에서 기존의 패러프레이즈 모델과 비교하여 더 높은 성공률을 달성하였습니다. 실험 결과, 생성된 적대적 예시가 더욱 다양하고 효과적임을 확인할 수 있었습니다.



### CReMa: Crisis Response through Computational Identification and Matching of Cross-Lingual Requests and Offers Shared on Social Media (https://arxiv.org/abs/2405.11897)
Comments:
          This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible

- **What's New**: 이 논문은 비상 상황 동안 소셜 미디어 플랫폼에서 요청과 제안을 효율적으로 식별하고 매칭하는 문제를 다루고 있습니다. 저자들은 텍스트, 시간, 공간적 특징을 통합한 시스템 접근법인 CReMa (Crisis Response Matcher)를 제안했습니다. CReMa는 CrisisTransformers라는 비상 상황에 특화된 사전 학습된 모델들을 활용하며, 다국어 임베딩 공간(cross-lingual embedding space)을 사용하여 여러 언어에서의 요청-제안 매칭을 향상시킵니다. 저자들은 또한 새로운 다국어 데이터셋을 소개하고, 16개의 언어에 걸친 포괄적인 실험을 수행했습니다.

- **Technical Details**: CReMa는 CrisisTransformers라는 사전 학습된 모델들을 활용하여 소셜 미디어에서의 요청과 제안을 식별하고 매칭합니다. CrisisTransformers는 30개 이상의 비상 사건 관련 트윗에서 추출된 150억 개 이상의 토큰을 학습한 모델들입니다. 다국어 데이터 처리를 위해, 저자들은 cross-lingual embedding space를 사용했습니다. 또한 데이터셋은 호주에서 가장 많이 사용되는 16개의 언어로 시뮬레이션된 시나리오를 포함하고 있으며, 다양한 벡터 검색 전략과 정확도 사이의 트레이드오프를 검토했습니다.

- **Performance Highlights**: CReMa 시스템은 요청-제안 매칭 작업에서 기존의 강력한 베이스라인인 RoBERTa, MPNet, BERTweet를 능가했습니다. CrisisTransformers를 사용한 텍스트 분류 모델은 뛰어난 성능을 보였으며, 다국어 임베딩 공간의 사용은 매칭 작업에서도 효과적이었습니다. 저자들은 또한 COVID-19 팬데믹 동안 수백만 규모의 지오태그된 글로벌 데이터셋을 분석하여, 소셜 미디어에서의 요청과 제안의 분포를 탐구했습니다.



### Unveiling and Manipulating Prompt Influence in Large Language Models (https://arxiv.org/abs/2405.11891)
Comments:
          ICLR 2024

- **What's New**: 이 논문은 대형 언어 모델(LLM)의 입력 saliency(각 토큰의 중요성)를 더 잘 이해하고 조작하기 위해 Token Distribution Dynamics (TDD)라는 새로운 접근법을 제안합니다. TDD는 기존 방법들이 가진 한계를 극복하고, LLM에서 적절한 토큰 생성의 인과 관계를 명확히 밝혀냅니다.

- **Technical Details**: TDD는 언어 모델 헤드(LM head)의 해석 능력을 이용하여 입력 토큰을 임베딩 공간으로 투사하고, 이를 토대로 각 토큰의 중요성을 배포 동학(distribution dynamics)으로 평가합니다. TDD는 forward, backward, bidirectional 세 가지 변형을 제안하여 각각의 토큰 관련성에 대한 독특한 통찰을 제공합니다.

- **Performance Highlights**: 광범위한 실험을 통해 TDD는 기존의 최신 기법들을 크게 능가하며, LLM 출력과 프롬프트 간의 인과 관계를 더욱 명확히 설명합니다. 또한 TDD를 이용한 제로 샷 독성 언어 억제와 감정 조절 작업에서 강력한 성능을 입증하며, 프롬프트 내 독성 단서와 감정 단서를 효과적으로 식별하고, 이에 따라 생성된 텍스트의 독성이나 감정을 조절합니다.



### A Novel Cartography-Based Curriculum Learning Method Applied on RoNLI: The First Romanian Natural Language Inference Corpus (https://arxiv.org/abs/2405.11877)
Comments:
          Accepted at ACL 2024 (main)

- **What's New**: 이 논문에서는 루마니아어를 위한 첫 번째 NLI(Natural Language Inference) 코퍼스인 RoNLI를 소개합니다. RoNLI는 58K의 훈련 문장 쌍과, 수작업으로 정확한 레이블이 주어진 6K의 검증 및 테스트 문장 쌍으로 구성되어 있습니다. 이 데이터셋은 연구자들이 루마니아어와 같은 저자원 언어에 대한 NLI 모델을 개발하고 강화하는 데 중요한 자원이 될 것입니다.

- **Technical Details**: RoNLI 코퍼스는 루마니아어 위키백과에서 특정 연결 구문을 검색하여 문장 쌍을 수집한 것입니다. 훈련 문장 쌍은 자동 규칙 기반 주석 과정을 통해 생성된 반면, 검증 및 테스트 문장 쌍은 사람에 의해 수작업으로 주석이 추가되었습니다. 다양한 머신러닝 모델(Shallow 모델부터 Transformer 기반의 신경망까지)을 실험하여 경쟁력 있는 베이스라인을 마련했습니다. 또한 데이터 지도를 기반으로 새로운 Curriculum Learning 전략을 사용해 최고의 모델을 개선했습니다.

- **Performance Highlights**: 최고의 베이스라인 모델은 데이터 지도를 사용한 새로운 커리큘럼 학습 방식으로 향상되었습니다. 이로써 데이터 레이블링 과정에서 발생할 수 있는 노이즈를 효과적으로 줄일 수 있었습니다. RoNLI 데이터셋과 베이스라인을 재현할 수 있는 코드는 GitHub에 공개되었습니다.



### xFinder: Robust and Pinpoint Answer Extraction for Large Language Models (https://arxiv.org/abs/2405.11874)
Comments:
          34 Pages

- **What's New**: 최근 큰 관심을 받고 있는 대형 언어 모델(LLM)의 공정하고 신뢰할 수 있는 성능 평가 방법 개발의 중요성이 대두되고 있습니다. 이 논문은 이에 맞춰 차별화된 접근 방식을 제안합니다. 기존의 정규 표현식(Regular Expression, RegEx) 기반의 평가 방법이 가진 한계를 극복하기 위해 xFinder라는 새로운 평가 모델을 소개합니다. xFinder는 핵심 답변 추출 모듈의 최적화를 목표로 하며, 이를 통해 추출 정확도를 높이고 특정 형식에 대한 의존성을 줄여 LLM 평가의 신뢰성을 높입니다.

- **Technical Details**: xFinder는 Key Answer Finder(KAF) 데이터셋을 기반으로 훈련된 모델로, LLM 평가에서 핵심 답변을 추출하는 데 특화되어 있습니다. RegEx는 기존 평가 시스템에서 주로 쓰이는 방법이지만 추출 오류가 자주 발생하는 단점이 있었습니다. xFinder는 이 문제를 해결하기 위해 고정된 응답 형식을 요구하지 않으면서도 높은 정확도로 LLM의 답변을 추출하는 기능을 갖추고 있습니다. 이를 위해 다양한 LLM의 출력에 대한 종합적인 검증과 실험을 거쳤으며, KAF 데이터셋은 특히 이 모델의 훈련과 평가를 위해 설계되었습니다.

- **Performance Highlights**: 최소 크기의 xFinder 모델(약 5억 매개변수)은 93.42%의 평균 답변 추출 정확도를 보여주었으며, 이는 최고의 평가 프레임워크에서 RegEx의 74.38%와 비교해 큰 차이를 보입니다. 또한 실험 결과, xFinder는 현재의 평가 프레임워크보다 더 높은 안정성과 정확성을 가지고 있습니다. 다양한 LLM에 대해 일관된 평가 결과를 제공함으로써, 기존 평가 방법이 가진 문제를 효과적으로 해결하고 있습니다.



### Intuitive Fine-Tuning: Towards Unifying SFT and RLHF into a Single Process (https://arxiv.org/abs/2405.11870)
- **What's New**: 최신 연구에서는 Supervised Fine-Tuning (SFT)와 Reinforcement Learning from Human Feedback (RLHF)의 최적화 목표를 통합하는 새로운 접근법인 Intuitive Fine-Tuning (IFT)을 제안합니다. IFT는 RLHF의 강점을 흡수하면서 SFT의 데이터 및 계산 효율성을 유지합니다.

- **Technical Details**: IFT는 SFT와 RLHF를 하나의 프로세스로 통합하여 LMs(Language Models)의 직관적인 응답 생성을 촉진합니다. 이를 위해 Markov Decision Process (MDP) 프레임워크에서 token 레벨에서의 Preference Estimation과 Transition Optimization 두 하위 프로세스로 SFT와 RLHF를 모델링합니다. IFT는 temporal residual connection을 사용하여 LMs가 전체 응답에 대한 직관적인 감각을 포착합니다. 또한, SFT와 동일한 양의 비선호 레이블 데이터(non-preference-labeled data)를 사용하며, 단일 정책(policy) 모델만을 필요로 합니다.

- **Performance Highlights**: 실험 결과, IFT는 SFT와 RLHF의 순차적 결합보다 여러 과제에서 더 우수하거나 동일한 성능을 보였습니다. 특히, 생성, 추론 및 사실 준수 능력이 요구되는 작업에서 뛰어난 성능을 발휘했습니다. 설명 가능한 Frozen Lake 게임으로 IFT의 효율성을 추가적으로 검증했습니다.



### CoNLL#: Fine-grained Error Analysis and a Corrected Test Set for CoNLL-03 English (https://arxiv.org/abs/2405.11865)
Comments:
          Accepted to LREC-COLING 2024

- **What's New**: 새로운 논문에서는 현재 최고 성능을 자랑하는 Named Entity Recognition (NER) 모델들의 오류를 세밀히 분석하고, 이를 바탕으로 NER 연구의 향후 방향을 제시하기 위해 새로운 문서 수준의 주석을 도입했습니다. 특히 CoNLL-03 English 테스트 세트의 기존 오류를 수정한 CoNLL#라는 새로운 버전을 발표하였습니다.

- **Technical Details**: 논문은 CoNLL-03 English 데이터셋을 활용한 NER 모델들이 특정 문서 형식과 도메인에서 얼마나 성능을 발휘하는지 자세한 오류 분석을 통해 탐구했습니다. 이를 위해 원본 테스트 셋의 231개 문서에 도메인(domain)과 형식(format)별로 주석을 달았으며, 가장 우수한 3개의 NER 모델을 테스트했습니다. 또한, 원본 코퍼스의 주석 오류와 잘못된 문장 경계를 수정한 CoNLL#를 통해 보다 일관된 오류 분석을 가능하게 했습니다.

- **Performance Highlights**: CoNLL#는 기존 테스트 세트의 체계적이고 빈번한 오류를 수정한 개선된 버전으로, 이를 통해 모델들의 성능이 전반적으로 향상되었습니다. 새로운 분석에서는 NER 모델들이 여전히 겪고 있는 문제를 해석 가능하고 의미 있는 패턴으로 밝혀내어, 모델의 개선 방향을 제시하였습니다.



### Beyond MLE: Investigating SEARNN for Low-Resourced Neural Machine Translation (https://arxiv.org/abs/2405.11819)
Comments:
          In fulfillment of the 2024 practical coursework of IFT6132 course: this https URL

- **What's New**: 이번 연구는 SEARNN 알고리즘이 아프리카 저자원 언어(예: 영어-이보, 프랑스어-에웨, 프랑스어-그호말라 번역)에서 기계 번역 성능을 향상시킬 수 있는지를 탐구했습니다. SEARNN은 전통적인 최대 가능도 추정(Maximum Likelihood Estimation, MLE) 방법의 한계를 극복하기 위해 제안된 학습-탐색(learning to search, L2S) 프레임워크를 기반으로 하는 알고리즘입니다.

- **Technical Details**: SEARNN 알고리즘은 MLE의 노출 편향(exposure bias)과 훈련-테스트 지표의 불일치 문제를 해결하도록 설계되었습니다. SEARNN은 각 단계에서 가능한 모든 토큰에 대해 비용을 계산하고 이를 기반으로 비용 민감(classification) 손실을 통해 RNN을 훈련시킵니다. 롤인(roll-in) 및 롤아웃(roll-out) 정책을 도입해 더 다양한 시퀀스 구성을 탐색합니다.

- **Performance Highlights**: SEARNN 알고리즘을 기계 번역 문제(영어-이보, 프랑스어-에웨, 프랑스어-그호말라)에 적용한 결과, 평균 BLEU 점수가 5.4% 향상되었습니다. 이는 SEARNN이 저자원 언어의 기계 번역에 효과적일 수 있음을 시사합니다.



### (Perhaps) Beyond Human Translation: Harnessing Multi-Agent Collaboration for Translating Ultra-Long Literary Texts (https://arxiv.org/abs/2405.11804)
Comments:
          work in progress

- **What's New**: 문학 번역의 복잡한 요구를 해결하기 위해 TransAgents라는 새로운 다중 에이전트 프레임워크를 도입했습니다. 이는 대형 언어 모델(LLMs)을 기반으로 하며, 다양한 에이전트의 집단적인 역량을 활용해 독창적인 문학 번역을 제공합니다.

- **Technical Details**: TransAgents는 전통적인 번역 출판 절차를 반영하여 구성원의 다단계 협업을 통해 번역 과정을 진행합니다. 주요 단계에는 Senior Editor가 Junior Editor, Translator, Localization Specialist, Proofreader와 같은 팀을 구성하여 협력합니다. 또한, 평가 방법으로 Monolingual Human Preference(MHP)와 Bilingual LLM Preference(BLP)를 제안하여 단일 언어 독자 및 이중 언어 모델에서의 선호도를 측정합니다.

- **Performance Highlights**: 실험 결과, TransAgents의 번역은 d-BLEU 점수는 낮았지만 인간 평가자와 LLM 평가자 모두에게 인간 작성 번역보다 선호되었습니다. 특히 역사적 맥락이나 문화적 뉘앙스를 묘사하는 데 뛰어났습니다. 비용 측면에서도 프로페셔널 번역가 고용 대비 80배의 비용 절감 효과를 보였습니다.



### Exploring Ordinality in Text Classification: A Comparative Study of Explicit and Implicit Techniques (https://arxiv.org/abs/2405.11775)
Comments:
          Findings of ACL 2024

- **What's New**: 본 논문은 자연어 처리(NLP)에서 자주 다루는 문제인 서열 분류(Ordinal Classification, OC)에 대한 새로운 접근 방식을 제안합니다. 기존에는 레이블의 서열 특성을 명시적으로 고려한 손실 함수(loss function) 개선에 집중했다면, 최근에는 사전 학습된 언어 모델(Pretrained Language Models, PLMs)을 활용하여 레이블의 내재된 의미를 통해 서열성을 암묵적으로 다루는 방법이 등장했습니다. 이 논문은 이러한 접근 방식을 이론적 및 실험적으로 종합적으로 분석하고 효과적인 접근 방식을 제안합니다.

- **Technical Details**: 서열 분류(OC)는 감성 분석(Sentiment Analysis), 평점 예측(Rating Prediction), 연령대 분류(Age Group Classification) 등 다양한 분야에서 자연스러운 순서를 가지는 출력이 필요한 과제입니다. 기존 명시적 접근법은 레이블 간의 거리 개념을 바탕으로 손실 함수를 조정하여 접근해 왔습니다. 대표적으로 여러 연구에서 제안된 Cross Entropy(CE) 손실 함수의 변형을 통해 서열 분류에 적합한 형태로 사용되었습니다. 반면, 암묵적 접근법은 BERT나 GPT와 같은 사전 학습된 언어 모델을 활용하여, 레이블의 의미를 자연스럽게 반영하여 분류 작업에 활용하는 방식입니다. 이 논문은 이러한 두 가지 접근법을 비교 분석하고, 새로운 하이브리드 손실 함수를 제안합니다.

- **Performance Highlights**: 논문은 명시적 접근법에 사용되는 여러 손실 함수를 이론적 속성(적절한 채점 규칙, 볼록성, 단봉성, 서열성) 관점에서 분석하고, 하이브리드 손실 함수를 통해 명시적 접근법의 성능을 개선했습니다. 또한, 사전 학습된 언어 모델을 활용한 암묵적 접근법이 기존 방법보다 서열 분류에 더 자연스럽고 효과적인 결과를 가져올 수 있음을 실험적으로 증명했습니다. 마지막으로, 각 접근법의 성능을 다양한 시나리오에서 비교 분석하여, 특정 조건에 따라 적합한 접근법을 추천합니다.



### Token-wise Influential Training Data Retrieval for Large Language Models (https://arxiv.org/abs/2405.11724)
Comments:
          Accepted to ACL 2024

- **What's New**: 이 논문에서는 대형 언어 모델(LLM) 생성에 어떤 훈련 데이터가 영향을 미쳤는지 식별할 수 있는 'RapidIn'이라는 확장 가능한 프레임워크를 제안합니다. 이 프레임워크는 두 가지 단계, 캐싱(caching)과 검색(retrieval)을 통해 작동합니다. 우선, 기울기 벡터(gradient vectors)를 200,000배 이상 압축하여 디스크 또는 GPU/CPU 메모리에 캐시할 수 있습니다. 그런 다음, 주어진 생성물에 대해 캐시된 기울기를 효율적으로 탐색하여 몇 분 안에 그 영향을 추정할 수 있습니다.

- **Technical Details**: RapidIn은 두 가지 주요 단계로 구성됩니다. 첫 번째는 캐싱(caching) 단계로, 기울기 벡터를 200,000배로 압축하여 저장합니다. 두 번째는 검색(retrieval) 단계로, 주어진 생성물에 대해 캐시된 기울기를 효율적으로 탐색합니다. 이 과정은 기존 방법에 비해 6,326배의 속도 향상을 달성하며, 다중 GPU 병렬화를 지원하여 캐싱 및 검색 성능을 대폭 향상시킵니다.

- **Performance Highlights**: 실험 결과 RapidIn은 매우 효율적이고 효과적이라는 것이 확인되었습니다. 특히, 기울기 벡터를 압축하고 캐싱하는 과정에서 다중 GPU 병렬화를 통해 처리 속도를 크게 향상시켜, 캐싱 및 검색 과정에서 급격한 속도 향상을 보여주었습니다.



### Cyber Risks of Machine Translation Critical Errors : Arabic Mental Health Tweets as a Case Study (https://arxiv.org/abs/2405.11668)
- **What's New**: 최근 머신 번역(Neural Machine Translation, NMT) 시스템의 발전으로 인해 번역 품질이 크게 향상되었지만, NMT 시스템이 '기계 환각'에 취약하다는 문제가 제기되었습니다. 본 논문에서는 이러한 문제를 다루기 위해 정신 건강 관련 게시물의 번역 오류에 관한 데이터를 수집하고 분석했습니다. 특히 아랍어 트윗 데이터에서의 치명적인 번역 오류를 탐지하고, 연구 커뮤니티에 경각심을 심어 주는 것을 목표로 합니다.

- **Technical Details**: 연구진은 Google Translate를 사용하여 아랍어 트윗 데이터를 영어로 번역한 후, 번역 오류를 자동으로 탐지하는 분류기를 사용하여 잠재적인 오류를 추출했습니다. 번역 데이터는 AraDepSu 데이터셋을 활용했으며, 수집된 트윗은 다양한 아랍어 방언과 현대 표준 아랍어로 작성되었습니다. 분류기는 XML-Roberta-base 모델을 기반으로 훈련되었습니다.

- **Performance Highlights**: 연구진은 다양한 품질 지표를 사용하여 번역 오류 데이터를 테스트한 결과, 기존 품질 지표들이 치명적인 오류를 제대로 감지하지 못한다는 사실을 발견했습니다. 이는 특히 정신 건강과 관련된 고위험 설정에서 중요한 문제로, 연구 커뮤니티에 개선된 지표의 필요성을 촉구하는 바입니다.



### Zero-Shot Stance Detection using Contextual Data Generation with LLMs (https://arxiv.org/abs/2405.11637)
Comments:
          5 pages, AAAI-2024 Workshop on Public Sector LLMs

- **What's New**: 저자들은 새로운 Dynamic Model Adaptation with Contextual Data Generation (DyMoAdapt) 방법을 제안하였습니다. 이는 적은 양의 훈련 데이터로 모델의 성능을 향상시키기 위해 Few-Shot Learning과 대형 언어 모델(Large Language Models)을 결합한 접근법입니다. 또한, GPT-3를 사용하여 다중 주제로 확장된 Multi Generated Topic VAST (MGT-VAST) 데이터셋을 소개하였습니다.

- **Technical Details**: DyMoAdapt는 테스트 시점에 모델을 미세 조정하여 새로운 주제에 적응하게 하는 방법입니다. 이를 위해 GPT-3를 사용하여 특정 주제에 맞는 새로운 데이터를 생성합니다. 또한 VAST 데이터셋을 확장하여 여러 주제와 연결된 컨텍스트를 가진 MGT-VAST 데이터셋을 제안하였습니다.

- **Performance Highlights**: 실험 결과, BERT 및 GPT-3 모델이 TOAD보다 F1_macro 성능에서 우수한 결과를 보였습니다. 그러나 DyMoAdapt 방법은 기대만큼 성능 향상이 나타나지 않았습니다. MGT-VAST 데이터셋은 다양한 모델들이 좋은 성능을 발휘함을 보였으나, 기존 방법들에 비해 큰 성능 차이는 보이지 않았습니다.



### Continuous Predictive Modeling of Clinical Notes and ICD Codes in Patient Health Records (https://arxiv.org/abs/2405.11622)
- **What's New**: 이번 연구는 병원 퇴원 시점이 아니라 환자가 병원에 머무르는 여러 시점에서 적용 가능한 ICD 코드를 예측하는 시스템을 개발하는 것을 목표로 합니다. 이 시스템은 퇴원 요약 대신 입원 초기부터 다양한 시점에서 환자의 전체 체류 기간 동안 예측을 시도합니다.

- **Technical Details**: 저자들은 환자의 입원 기간 동안 다양한 시간점에서 ICD 코드를 예측하는 모델을 제안했습니다. 이 모델은 긴 임상 문서 시퀀스를 인코딩하고 주어진 시점에서 예측을 수행하는 구조로 설계되었습니다. 시퀀스를 작은 청크로 나누고, 각 청크를 사전 훈련된 언어 모델(PLM)을 사용하여 인코딩하며, 원시 토큰 임베딩을 추출합니다. 마지막으로, 인과적 주의 메커니즘(causal attention)을 사용하여 임베딩을 결합하여 시간에 따른 정보 조합을 수행합니다. 이 모델은 환자의 병원 체류 동안 이용 가능한 노트를 기반으로 ICD 코드 분포를 예측합니다.

- **Performance Highlights**: 모델 실험 결과, 입원 후 이틀 만에 최종 ICD 코드를 예측할 수 있으며, 초기 예측 작업에서 성능 향상을 이끌어냈습니다. 이 시스템은 초기 시점의 약한 상관 관계에도 불구하고 데이터를 증강하고 추론 중 컨텍스트를 확장하여 성능을 대폭 향상시켰습니다.



### Decoding by Contrasting Knowledge: Enhancing LLMs' Confidence on Edited Facts (https://arxiv.org/abs/2405.11613)
- **What's New**: 최근 큰 언어 모델(LLMs)이 저장한 지식은 실제 세계의 변화로 인해 빠르게 구식이 될 수 있습니다. 이를 해결하기 위해, 본 연구는 '디코딩 대비 지식(DeCK, Decoding by Contrasting Knowledge)'이라는 새로운 접근법을 제안하여 LLM의 고집스러운 지식을 효과적으로 편집하는 방법을 제시합니다. DeCK는 기존의 편집되지 않은 지식과 새로 추가된 지식을 대조하여 토큰별 분포를 분석하고, 이를 통해 LLM의 예측 정확성을 높입니다.

- **Technical Details**: DeCK의 작동원리는 다음 두 가지 구성 요소로 이루어져 있습니다: (1) 새 지식에 대한 주의를 높이도록 행동을 향상시키는 편집 향상 모듈, (2) 새로운 지식과 기존의 파라메트릭 지식을 대조한 후 다음 토큰을 예측하는 대조적 디코딩 전략. 이 두 요소를 결합하여, LLM의 기존 지식을 교정하는 과정을 보다 효과적으로 만듭니다.

- **Performance Highlights**: 본 연구의 실험 결과는 DeCK가 LLaMA3-8B-instruct 모델의 MQuAKE 벤치마크 성능을 최대 219% 향상시켰음을 보여줍니다. 이는 ICE 방법을 보강하여 LLM의 고집스러운 지식을 효과적으로 편집하도록 돕는 강력한 도구임을 입증합니다.



### Language Reconstruction with Brain Predictive Coding from fMRI Data (https://arxiv.org/abs/2405.11597)
- **What's New**: 최근 연구들은 뇌 신호에서 음성 인식을 디코딩하고 이를 연속적인 언어로 재구성할 수 있음을 보여주고 있습니다. 그러나, 뇌 신호 내에 포함된 의미 정보를 효과적으로 언어 재구성에 활용할 수 있는 신경학적 근거는 부족합니다. 이 논문에서는 언어 재구성 과정을 통해 뇌 예측 코딩(predictive coding) 이론을 탐구하는 새로운 모델 PredFT를 제안합니다. 이 모델은 신경 디코딩과 뇌 예측을 공동으로 모델링합니다.

- **Technical Details**: PredFT 모델은 언어 재구성을 위한 메인 디코딩 네트워크와 뇌 예측 코딩을 위한 사이드 네트워크로 구성되어 있습니다. 메인 네트워크는 공간-시간 특징 추출을 위해 3D CNN과 혈액 산소 수준 지연을 보정하는 FIR 모델을 사용합니다. 사이드 네트워크는 여러 관심 있는 뇌 영역에서 예측 코딩 표현을 얻어 크로스 어텐션(cross-attention) 모듈을 통해 메인 네트워크로 통합합니다. 실험은 Narratives라는 대규모 자연스러운 언어 이해 fMRI 데이터셋에서 수행되었습니다.

- **Performance Highlights**: PredFT는 현존하는 방법들에 비해 높은 디코딩 성능을 보여주었으며, 최대 BLEU-1 점수 27.8%를 기록했습니다. 또한, 뇌 예측 코딩 기능이 특정 뇌 영역에서 유래한다는 점을 확인했습니다. 이러한 공동 모델링은 fMRI-텍스트 디코딩 성능을 크게 향상시킵니다.



### Exploring the Capabilities of Prompted Large Language Models in Educational and Assessment Applications (https://arxiv.org/abs/2405.11579)
Comments:
          Accepted at EDM 2024

- **What's New**: 본 연구는 생성 AI 시대에서 대형 언어 모델(LLMs)의 잠재력을 교육과 평가 분야에서 탐구합니다. 본 연구는 학교 수준 교과서와 학부 수준의 기술 교재로부터 개방형 질문을 생성하고, 체인 오브 소트(chain-of-thought) 방식을 사용하여 언어 비종속적인 선택형 질문(MCQ)을 생성하는 가능성도 탐구합니다. 또한, 인도 저자원 언어인 벵골어 문법 오류 설명 사례 연구와 인적 자원 인터뷰 기록 평가 가능성도 조사합니다.

- **Technical Details**: 본 연구는 LLMs의 효능을 탐구하기 위해 여러 연구 질문을 설정했습니다. 프롬프트 기반 기법이 학교 및 학부 수준 교재에서 개방형 질문을 생성하는 데 얼마나 효과적인지, GPT 기반 모델을 사용한 체인 오브 소트 방식이 언어 비종속 MCQ 생성을 할 수 있는지, 사전 훈련된 LLMs가 벵골어 문법 오류를 설명하는 데 얼마나 효과적인지, HR 면접 기록을 평가하는 데 얼마나 준비가 되어 있는지를 조사합니다.

- **Performance Highlights**: 현재까지의 연구 결과, 프롬프트 기반 기법은 학교 수준 교과서에서 개방형 질문을 생성하는 데 유용할 수 있음을 발견했습니다. 이에 따라 새로운 데이터셋 'EduProbe'를 구축하여 학교 과목별로 풍부한 컨텍스트를 제공합니다. 이 데이터셋은 역사, 지리, 경제, 환경 학습, 과학 등 NCERT 교과서 콘텐츠를 사용하여 질문을 생성합니다. 개방형 질문 생성, 언어 비종속 MCQ 생성, 벵골어 문법 오류 설명, HR 면접 기록 평가에서 LLMs의 잠재력을 조사 중입니다.



### A Multi-Perspective Analysis of Memorization in Large Language Models (https://arxiv.org/abs/2405.11577)
- **What's New**: 대형 언어 모델(LLMs)의 '메모리제이션(mem.)' 현상을 다각도로 분석한 연구입니다. 이 연구에서는 모델의 크기, 문맥의 길이, 그리고 문장의 연속 크기와 메모리제이션 사이의 관계를 밝혀냈습니다. 또한, 메모리제이션 정도가 낮거나 전혀 기억하지 못하는 문장이 어떻게 메모리제이션 문장으로 전환되는지도 설명합니다.

- **Technical Details**: 메모리제이션을 측정하기 위해 K-extractability 방법을 사용하여 특정 문맥 길이에 대한 모델의 반응을 평가했습니다. 이어지는 토큰을 예측하고 이 예측이 실제 연속성과 어떻게 비교되는지 분석하였습니다. 또한, 임베딩 공간에서 메모리제이션 점수에 따른 문장의 분포와 디코딩 다이나믹스를 연구했습니다.

- **Performance Highlights**: [{'relation of memorization': '모델의 크기와 문맥의 길이가 메모리제이션 문장의 수에 어떤 영향을 미치는지 설명했습니다. 특히, 메모리제이션 문장의 수는 문맥의 길이에 따라 비선형적으로 증가하지만, 문장의 연속 크기에 따라 증가합니다.'}, {'transition of sentences': '메모리제이션 문장과 비메모리제이션 문장의 전환 과정을 분석했습니다. 큰 모델에서 메모리제이션 문장이 증가하는 경향이 있으며, 낮은 메모리제이션 문장도 점차 메모리제이션될 수 있음을 발견했습니다.'}, {'embedding analysis': '각기 다른 메모리제이션 점수를 갖는 문장들이 임베딩 공간에서 군집을 이루고 있으며, Cosine Similarity가 모델 크기에 따라 더 뚜렷해진다고 설명하였습니다.'}, {'n-gram and decoding entropy dynamics': 'n-gram과 디코딩 엔트로피 다이나믹스에서 메모리제이션 문장과 비메모리제이션 문장을 생성할 때 경계 현상이 발견되었습니다. 비메모리제이션 토큰을 생성할 때 빈도가 갑자기 감소하는 반면, 메모리제이션 토큰은 증가합니다.'}, {'prediction model': '문맥을 기반으로 메모리제이션 예측이 가능한 Transformer 모델을 훈련시킨 결과, 토큰 단위에서의 예측 정확도가 높았으며, 모델 크기가 증가함에 따라 예측 정확도가 향상된다고 설명했습니다.'}]



### SEEP: Training Dynamics Grounds Latent Representation Search for Mitigating Backdoor Poisoning Attacks (https://arxiv.org/abs/2405.11575)
Comments:
          accepted to TACL

- **What's New**: 해커가 NLP 모델에 백도어 공격(Backdoor Attack)을 심는 위협을 방어하기 위해 새로운 방법론 제시

- **Technical Details**: 이번 연구에서는 학습 중 다이내믹스를 활용하여 중독된 샘플을 정밀하게 식별한 후, 레이블 전파(label propagation) 단계를 거쳐 높은 재현율을 달성하는 방어 메커니즘을 제안하였습니다.

- **Performance Highlights**: 이 방법은 기존의 최신 방어법과 비교하여 백도어 공격의 성공률을 크게 줄이면서도, 깨끗한 테스트 세트에 대해 높은 분류 정확도를 유지합니다.



### DaVinci at SemEval-2024 Task 9: Few-shot prompting GPT-3.5 for Unconventional Reasoning (https://arxiv.org/abs/2405.11559)
- **What's New**: 이번 연구에서 소개된 BRAINTEASER 작업은 기존의 일반적인 논리적 사고를 넘어서는 문제 해결을 요구하는 새로운 종류의 질문을 도입했습니다. 이러한 문제는 전통적인 상식적 추론을 무시하고 비전통적인 관점에서 접근해야 하는 'Sentence Puzzles'와 'Word Puzzles' 두 가지 유형으로 나눠집니다.

- **Technical Details**: 이번 연구에서는 GPT-3.5 모델과 few-shot prompting 기법을 사용하여 BRAINTEASER 작업을 수행했습니다. GPT-3.5는 수백억 개의 매개변수로 학습된 Transformer 아키텍처 기반의 모델로, 자가 회귀적 특성과 대규모 데이터 학습을 통해 많은 자연어 처리 작업을 수행할 수 있습니다. Sentence Puzzles와 Word Puzzles는 각각 다른 특성을 가지므로, 각 유형별로 별도의 프롬프트 세트를 만들어 사용했습니다.

- **Performance Highlights**: Sentence Puzzle 작업에서 26위, Word Puzzle 작업에서 15위를 기록하였습니다. 이는 few-shot prompting 방식을 통해 LLM이 해당 작업을 성공적으로 수행할 수 있음을 보여줍니다.



### Simple-Sampling and Hard-Mixup with Prototypes to Rebalance Contrastive Learning for Text Classification (https://arxiv.org/abs/2405.11524)
Comments:
          12 pages, 9 figures

- **What's New**: 최근의 감독된 대조 학습 방식(supervised contrastive learning)이 놀라운 성능과 견고성으로 주목받고 있는 가운데, 텍스트 분류(text classification)를 위한 새로운 모델 SharpReCL이 도입되었습니다. 이 모델은 불균형 데이터세트에서도 우수한 성능을 발휘하도록 설계되었습니다.

- **Technical Details**: SharpReCL 모델은 두 가지 주요 개선점을 도입했습니다. 첫째, 균형 잡힌 분류 브랜치에서 각 클래스의 프로토타입 벡터(prototype vector)를 생성하고, 이를 통해 각 클래스에 대해 동일한 크기의 타겟 샘플 세트를 구성하여 감독된 대조 학습(supervised contrastive learning)을 수행합니다. 둘째, 분류와 대조 학습 브랜치를 명시적으로 연계하여 상호 간섭을 최소화합니다.

- **Performance Highlights**: 실증 결과, SharpReCL 모델은 여러 데이터세트에서 인기 있는 대규모 언어 모델(large language models)을 능가하는 성능을 보여주었습니다.



### MSNER: A Multilingual Speech Dataset for Named Entity Recognition (https://arxiv.org/abs/2405.11519)
- **What's New**: 본 논문에서는 음성 기반 언어 이해에서 그동안 주목받지 못했던 개체명 인식(NER)에 대해 다룹니다. 기존 자원이 영어에 한정된 단일 데이터셋에 집중되어 있는 한계를 보완하기 위해, 네덜란드어, 프랑스어, 독일어, 스페인어 등 4개 언어로 구성된 멀티링궐 음성 코퍼스 MSNER을 소개합니다. 이 데이터셋은 VoxPopuli 데이터셋을 기반으로 하여, 훈련 및 검증을 위한 590 및 15시간의 기계 주석된 음성과 17시간의 수동 주석된 평가 세트를 제공합니다.

- **Technical Details**: MSNER 데이터셋은 인기 있는 VoxPopuli 데이터셋의 테스트 세트를 네덜란드어, 프랑스어, 독일어 및 스페인어로 수동으로 주석 처리했습니다. 훈련 및 검증 세트에는 자동 주석 도구를 이용한 'silver' 주석을 제공하며, 평가 세트는 수동으로 주석된 'gold' 주석 데이터를 포함합니다. 데이터셋은 OntoNotes의 18개 클래스에 따라 주석이 달려 있으며, 이를 통해 cross-lingual 연구를 용이하게 하고 Spoken NER 모델의 포괄적인 평가 자원을 제공합니다.

- **Performance Highlights**: 기계 주석과 수동 주석의 비교 분석을 통해, 제시된 데이터셋이 Spoken NER 모델의 성능 평가와 연구에 유용한 도구가 될 수 있음을 확인했습니다. 또한, 본 데이터셋을 이용한 벤치마크 실험 결과, 다양한 언어에서의 Spoken NER 성능을 균형 있게 평가할 수 있음을 보여줍니다.



### Effective In-Context Example Selection through Data Compression (https://arxiv.org/abs/2405.11465)
Comments:
          Accepted by ACL 2024 finding

- **What's New**: 이번 연구에서는 대형언어모델(Large Language Models, LLMs)에서 인컨텍스트 러닝(In-Context Learning, ICL)의 예제 선택 전략에 대한 새로운 접근법을 제안합니다. ICL에서 예제 선택 메커니즘과 전략에 체계적인 연구가 부족하다는 점을 해결하기 위해, 데이터 압축 기법을 이용해 더 효율적으로 관련 예제를 선택하는 방법을 소개합니다. 이 방법은 여러 언어 모델에서 평균 5.90%의 성능 향상을 달성했습니다.

- **Technical Details**: 본 연구는 두 단계로 이루어진 예제 선택 방법을 제안합니다. 첫 번째 단계에서는 쿼리 입력과 관련된 예제를 추출하여 예제와 쿼리 소스 간의 연관성을 보장합니다. 두 번째 단계에서는 메타-그래디언트 기반 영향 함수(meta-gradient-based influence function)를 사용해 각 예제의 영향 점수를 계산하고, 이를 바탕으로 인컨텍스트 예제를 선택합니다. 이 프레임워크는 훈련 세트의 중요한 정보를 인컨텍스트 예제로 압축해 ICL의 성능을 향상시킵니다. 또한, 데이터에 독립적이며 소수의 모델 파라미터만을 사용하고 부가적인 모델 훈련이 필요 없습니다.

- **Performance Highlights**: 다양한 실험 결과, 제안한 방법은 5개의 실제 데이터셋과 4개의 언어모델(LM)을 사용한 실험에서 평균 5.90%의 성능 향상을 보였습니다.



### Efficient Prompt Tuning by Multi-Space Projection and Prompt Fusion (https://arxiv.org/abs/2405.11464)
- **What's New**: 새로운 Efficient Prompt Tuning (EPT) 방법이 제안되었습니다. 이 방법은 두 가지 핵심 모듈, 즉 '프롬프트 융합'(prompt fusion)과 '다중 공간 투영'(multi-space projection)을 활용하여 기존 프롬프트 튜닝(prompt tuning)의 한계를 극복합니다. EPT는 프롬프트를 짧고 두 개의 저랭크 행렬(low-rank matrices)로 분해하여 파라미터 수와 훈련 시간을 크게 줄이면서도 성능을 향상시킵니다.

- **Technical Details**: EPT는 먼저 소프트 프롬프트(soft prompt)를 짧은 프롬프트와 두 개의 저랭크 행렬로 분해합니다. 이 과정에서 짧은 프롬프트만 입력에 첨부하여 훈련 시간을 감소시킵니다. 저랭크 행렬은 고정된 입력 텍스트 임베딩을 업데이트하는데 사용됩니다. 이후, 프롬프트 융합 모듈은 저랭크 행렬과 짧은 프롬프트 사이의 지식 차이를 캡처하여 짧은 프롬프트에 추가적인 의미적 풍부함을 제공합니다. 다중 공간 투영 모듈은 단일 소프트 프롬프트를 여러 하위 공간으로 투영하고, 게이트 네트워크(gating network)를 통해 이들 하위 공간에서 프롬프트를 재가중치하여 다양한 다운스트림 작업에 적응할 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, EPT 방법은 13개의 자연어 처리 다운스트림 작업에서, 기존의 11개의 비교 방법을 크게 능가했으며, 최대 28.8%의 상대적 성능 향상과 14%의 훈련 시간 감소를 보였습니다. 특히 GLUE와 SuperGLUE 벤치마크에서 다른 파라미터 효율적 튜닝(PEFT) 방법인 LoRA 및 멀티태스크 학습 기반 PT 변형보다 우수한 성능을 발휘했습니다.



### MAML-en-LLM: Model Agnostic Meta-Training of LLMs for Improved In-Context Learning (https://arxiv.org/abs/2405.11446)
Comments:
          KDD 2024, 11 pages(9 main, 2 ref, 1 App) Openreview this https URL

- **What's New**: 이 연구는 대형 언어 모델(LLM)을 새로운 작업에 적응시키기 위한 새로운 메타 학습 방법인 MAML-en-LLM을 제안합니다. 기존의 메타 학습 방식인 MetaICL와 MetaICT는 다양한 작업에서 LLM을 미세 조정(fine-tuning) 하는 방식으로 메타 학습을 수행하였지만, MAML-en-LLM은 단계별 최적화를 통해 더 일반화된 파라미터를 학습합니다.

- **Technical Details**: MAML-en-LLM은 Model-Agnostic Meta-Learning (MAML)에서 영감을 받아 설계되었습니다. 이 방식은 내적 루프(inner loop)와 외적 루프(outer loop) 두 단계로 나누어 최적화를 수행합니다. 내적 루프에서는 모델 파라미터를 각 작업에 적응시키고, 외적 루프에서는 적응된 파라미터를 기반으로 모델 파라미터를 업데이트합니다. 이 과정에서 2차 기울기(second-order gradient) 업데이트를 사용하여 메타 업데이트 방향을 안내합니다.

- **Performance Highlights**: MAML-en-LLM은 보지 못한 도메인(unseen domains)에서 평균 2% 성능 향상과 적응 성능에서 4% 향상을 보였습니다. 또한, 데이터가 제한적일 때와 풍부할 때 모두 MAML-en-LLM이 기존 메타 학습 방법보다 평균 2% 더 높은 성능을 보였습니다. 7개의 작업 설정에서 수행된 종합적인 실험 결과 MAML-en-LLM이 최신(meta-training) 방법들을 능가함을 증명했습니다.



### MHPP: Exploring the Capabilities and Limitations of Language Models Beyond Basic Code Generation (https://arxiv.org/abs/2405.11430)
Comments:
          39 pages, dataset and code are available at this https URL

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전으로 함수 수준 코드 생성이 크게 개선되었습니다. 예를 들어, GPT-4는 HumanEval에서 88.4%의 통과율을 기록했습니다. 하지만 기존의 벤치마크가 함수 수준 코드 생성 능력을 충분히 평가하는지에 대한 의문이 제기되었습니다. 본 연구에서는 두 가지 일반적인 벤치마크, HumanEval과 MBPP를 분석한 결과, 품질, 난이도 및 세밀성 측면에서 LLMs의 코드 생성 능력을 충분히 평가하지 못할 수 있음을 발견했습니다. 이를 해결하기 위해 우리는 140개의 고유한 인간 큐레이션 문제로 구성된 Mostly Hard Python Problems (MHPP) 데이터를 소개합니다.

- **Technical Details**: MHPP는 자연어와 코드 추론의 결합에 중점을 두어, LLMs가 사양과 제약 조건을 이해하고, 여러 단계를 거치는 추론을 수행하며, 코딩 지식을 효과적으로 적용할 수 있는지 평가합니다. 22개의 LLMs를 MHPP를 사용하여 초기 평가한 결과 HumanEval에서 높은 성과를 보인 많은 모델들이 MHPP에서는 유사한 성공을 이끌어내지 못했습니다. 이는 MHPP가 다양한 LLMs의 이전에 발견되지 않았던 여러 제한 사항을 강조함을 보여줍니다.

- **Performance Highlights**: MHPP에 대한 초기 평가 결과, HumanEval에서 높은 성과를 보였던 여러 모델들이 MHPP에서는 유사한 성공을 거두지 못했습니다. 이는 기존 벤치마크가 충분하지 않음을 보여주며, MHPP가 LLMs의 능력과 한계를 보다 더 잘 이해할 수 있는 길을 열어줄 수 있음을 시사합니다. 데이터셋과 코드는 해당 URL을 통해 제공됩니다.



### Large Language Models are Biased Reinforcement Learners (https://arxiv.org/abs/2405.11422)
- **What's New**: 본 연구는 LLMs(대규모 언어 모델)가 강화 학습(RL) 작업에서 보상 극대화 선택을 수행하면서 나타내는 가치 편향을 조사합니다. 특히, 비교를 통한 상대적 가치 인코딩이 LLMs에 적용되는지를 다룹니다.

- **Technical Details**: 연구는 여러 밴딧 작업과 모델을 통해 LLMs가 상대적 가치 편향을 나타냄을 보여줍니다. 프롬프트에 명시적인 결과 비교를 추가하면 훈련된 선택 세트에서의 성능 향상과 새로운 선택 세트로의 일반화 성능 저하라는 반대 효과가 나타났습니다. 컴퓨팅 인지 모델링은 LLM의 행동이 단순한 RL 알고리즘으로 잘 설명됨을 나타냅니다.

- **Performance Highlights**: 모든 작업과 모델에서 보상 극대화 선택은 기회보다 높은 결과를 보였습니다. 세부적으로 네 개의 작업에서 상대적 가치 편향의 행동 신호가 발견되었으며, 특히 비교 프롬프트 조건에서 두드러졌습니다. 이러한 발견은 LLMs의 의사결정 응용에 중요한 시사점을 제공합니다.



### Can Public LLMs be used for Self-Diagnosis of Medical Conditions ? (https://arxiv.org/abs/2405.11407)
Comments:
          11 Pages, 4 figures, Submitted to ACM Transactions on Computing for Healthcare

- **What's New**: 대형 언어 모델(Large Language Models, LLMs)의 새로운 응용으로서 의료 자가 진단(self-diagnosis)에 대한 연구가 진행되었습니다. 이 논문에서는 Google의 Gemini와 Bing의 GPT-4.0이 자가 진단 작업에 어떻게 활용되는지를 조사합니다.

- **Technical Details**: 연구팀은 프롬프트 엔지니어링(prompt engineering)을 통해 10,000개의 샘플 데이터를 준비했습니다. 이 샘플을 통해 자가 진단 작업에서 Gemini와 GPT-4.0의 성능을 테스트하고 비교했습니다.

- **Performance Highlights**: 자가 진단 작업에서 GPT-4.0의 정확도는 63.07%인 반면, Gemini는 6.01%에 불과했습니다. 이는 두 모델의 성능 차이를 명확히 보여줍니다. 또한, Retrieval Augmented Generation을 사용한 성능 향상 가능성도 논의되었습니다.



### MapCoder: Multi-Agent Code Generation for Competitive Problem Solving (https://arxiv.org/abs/2405.11403)
- **What's New**: 이번 논문에서는 코드 생성 작업을 위한 새로운 접근법 'MapCoder'를 소개합니다. 이는 다중 에이전트 프롬프팅(multi-agent prompting)을 활용하여, 개발자들이 사용하는 전체 프로그램 합성 주기를 모방합니다. MapCoder는 네 개의 대형 언어 모델(LLM) 에이전트로 구성되어 있으며, 각각 예제 회상, 계획 수립, 코드 생성, 디버깅 단계를 수행합니다. 이 새로운 접근법을 통해 HumanEval, MBPP, APPS, CodeContests, xCodeEval과 같은 다양한 벤치마크에서 새로운 state-of-the-art 결과를 달성했습니다.

- **Technical Details**: MapCoder는 사람 개발자들의 문제 해결 주기를 모방한 다중 에이전트 시스템을 활용합니다. 각 LLM 에이전트는 문제 해결의 특정 단계를 담당하며, 이 에이전트들은 순차적으로 작동하여 각각 예제 회상, 계획 수립, 코드 작성, 디버깅을 수행합니다. 기존의 LLM이 매개변수나 데이터 셋 규모를 확장하면서도 복잡한 문제 해결에 어려움을 겪는 점을 보완하기 위해, MapCoder는 사용자가 문제를 최대한 이해하고 적절한 코드를 생성하도록 도와주는 구조를 갖추고 있습니다. 이를 통해 각 단계에서의 보완을 가능하게 하고, 유연한 반복 프로토콜을 적용하여 생성 절차를 개선합니다.

- **Performance Highlights**: MapCoder는 여러 벤치마크에서 인상적인 성능을 입증했습니다. HumanEval 벤치마크에서는 93.9%, MBPP에서는 83.1%, APPS에서는 22.0%, CodeContests에서는 28.5%, xCodeEval에서는 45.3%의 pass@1 점수를 기록하며 새로운 기록을 수립했습니다. 또한 다양한 프로그래밍 언어와 문제 난이도에서 일관된 성능 향상을 보이며, HumanEval, MBPP, APPS, CodeContests, xCodeEval과 같은 알려진 벤치마크에서 상당한 성능 향상을 달성했습니다.



### Large Language Models Lack Understanding of Character Composition of Words (https://arxiv.org/abs/2405.11357)
- **What's New**: 언어 모델(LLMs)은 다양한 자연어 작업에서 뛰어난 성과를 보여왔으나, 문자(character) 수준의 이해는 여전히 한계가 존재한다는 연구결과가 발표되었습니다. 본 논문에서는 LLMs가 단어의 문자 구성을 이해하는 능력을 평가하며, 그들이 매우 간단한 문자 수준의 작업에서도 신뢰할 수 없음을 보여줍니다. 이를 통해 문자 수준의 이해를 향상시키기 위한 잠재적 연구 방향을 논의합니다.

- **Technical Details**: LLMs 대부분이 토큰(token) 레벨에서 훈련되기 때문에 문자 구성의 미묘한 차이를 이해하기 어려운 문제가 있습니다. 본 논문에서는 GPT-4, Claude, Gemini 1.5, Mistral 7B와 같은 공개된 LLM을 대상으로, 문자 포함 단어 찾기, 문자 삽입/삭제/교체, 문자 재배열, 문자 개수 세기 등의 단순 작업에 대한 실험을 진행했습니다. 또한 인간 조사자와의 성능 비교를 통해 LLM의 문자 수준 이해도를 분석했습니다.

- **Performance Highlights**: 결과적으로 모든 대상 LLM들은 토큰 레벨 작업에 비해 문자 레벨 작업에서 성능이 현저히 저하됨을 확인했습니다. 인간은 모든 작업에서 거의 완벽한 성과를 보였지만, LLM들은 중대한 결함을 드러냈습니다. 이는 LLM의 훈련 방식 및 언어 인식에 근본적인 결함이 있음을 시사합니다. 예를 들어, 문자 포함 단어 찾기 작업에서는 GPT-4, Claude, Gemini 1.5, Mistral 7B 모두 인간 대비 낮은 정확도와 재현율을 기록했습니다.



### Enhancing Fine-Grained Image Classifications via Cascaded Vision Language Models (https://arxiv.org/abs/2405.11301)
- **What's New**: CascadeVLM은 고해상도 이미지 분류에서 특히 힘든 zero/few-shot 상황에서 기존의 CLIP 기반 모델을 능가하는 새로운 프레임워크입니다. 이 모델은 대규모 비전-언어 모델(LVLM)의 세밀한 지식을 효과적으로 활용하여 성능 향상이 가능합니다. 특히, Stanford Cars 데이터셋에서 CascadeVLM은 85.6%의 zero-shot 정확도를 기록하였습니다.

- **Technical Details**: CascadeVLM은 크게 두 단계로 나누어집니다: (1) CLIP 모델을 이용한 후보군 선택 (2) LVLMs를 활용한 zero-shot 또는 few-shot 예측. CLIP 모델은 이미지와 텍스트 표현을 통합된 임베딩 공간에 맞추어 후보군을 선정하며, 이후 LVLM이 이 후보군을 바탕으로 최종 분류를 수행합니다.

- **Performance Highlights**: CascadeVLM은 여러 장세분화 이미지 데이터셋에서 실험을 통해 기존 모델을 능가하는 성능을 보였습니다. 예를 들어 Stanford Cars 데이터셋에서는 85.6%의 zero-shot 정확도를 달성하였고, iNaturalist와 SUN397 데이터셋에서도 뛰어난 성능을 보여주었습니다.



### Unveiling Key Aspects of Fine-Tuning in Sentence Embeddings: A Representation Rank Analysis (https://arxiv.org/abs/2405.11297)
- **What's New**: 최신 비지도 학습 기반 문장 임베딩(sentence embeddings) 방법론에서 크게 두드러진 발전은 대조 학습(contrastive learning, CL) 기반의 미세 조정(fine-tuning)이다. 본 연구에서는 표현 순위(representation rank)를 주요 분석 도구로 사용하여 최신 문장 임베딩 기법을 분석했다. 이를 통해 우리가 알아낸 점은 미세 조정 과정에서 표현 순위가 정점에 달할 때를 기준으로 Phase 1과 Phase 2를 정의하고, 이 두 단계를 통해 정렬 및 균일성, 언어 능력, 성능과 순위 간의 상관관계를 철저히 분석한 것이다.

- **Technical Details**: 비지도 SimCSE 모델을 시작으로 여러 CL 기반 모델이 문장 임베딩에 사용되며 성능을 향상시키고 있다. 우리는 분석을 위해 MixCSE, ESimCSE, InfoCSE, PromptBERT와 같은 최신 모델들을 선택했다. 표현 순위는 주어진 배치에서 가장 큰 특정 특이 값(singular values)의 에너지를 기반으로 측정된다. 하지만 이 방법이 불연속적(discrete) 특성을 가지므로 정규화 목적에는 적합하지 않다. 이를 해결하기 위해 우리는 효과적 순위(effective rank)라는 근사치를 적용했다.

- **Performance Highlights**: 우리의 실험 결과, Rank Reduction(RR) 전략을 통해 CL 기반 모델의 성능과 안정성을 높임을 입증했다. MixCSE 및 PromptBERT와 같은 모델에서는 불안정성 문제가 제기되었지만, RR을 통해 이러한 문제를 완화하고 문장 임베딩의 성능 향상을 도모할 수 있었다. 효과적인 순위 감소와 상대적으로 높은 성능의 상관관계를 관찰했다.



### MBIAS: Mitigating Bias in Large Language Models While Retaining Contex (https://arxiv.org/abs/2405.11290)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)의 안전성을 개선하기 위해 MBIAS라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 맞춤형 데이터 세트에서 안전성 개선을 목표로 하는 'instruction fine-tuning' 방식으로 훈련되었습니다. 기존 모델들이 안전성을 강조하는 과정에서 맥락적 정확성을 잃는 문제를 해결하고자 하였습니다.

- **Technical Details**: MBIAS는 Mistral2-7B-instruct 모델을 기반으로 개발되었으며, 편향(bias)과 독성을 줄이기 위해 특수하게 구성된 데이터 세트를 사용하여 훈련되었습니다. 이 데이터 세트는 잠재적으로 위험한 요소(예: 편견을 포함한 텍스트)와 이에 상응하는 안전한 버전을 포함한 쌍으로 이루어져 있습니다. 학습 과정에는 ‘instruction fine-tuning’ 방법이 사용되었으며, 이는 모델이 편향을 인식하고 이를 피하면서 원래 입력의 의미를 유지하도록 유도합니다.

- **Performance Highlights**: 실험 결과, MBIAS는 편향과 독성을 30% 이상 줄이면서도 중요한 정보를 성공적으로 유지하는 것으로 나타났습니다. 추가적으로, 다양한 인구 통계에서의 성능을 평가한 결과, 편향과 독성이 90% 이상 감소한다는 점에서 우리의 접근 방법이 매우 견고함을 확인했습니다. 우리는 본 연구의 유용성을 높이기 위해 데이터 세트와 모델을 연구 커뮤니티에 공개하고자 합니다.



### Estimating the Level of Dialectness Predicts Interannotator Agreement in Multi-dialect Arabic Datasets (https://arxiv.org/abs/2405.11282)
Comments:
          Accepted to ACL 2024 (Main)

- **What's New**: 최신 연구는 다중 방언 아랍어 데이터셋 주석 작업에서 방언 샘플을 각 샘플의 해당 방언을 모국어로 사용하는 주석자에게 우선적으로 배정하는 것이 데이터셋의 품질을 높일 수 있다고 제안합니다. 이를 위해 각 문장이 표준 아랍어(MSA)로부터 얼마나 벗어나는지를 측정하는 'Arabic Level of Dialectness (ALDi)'를 사용하여 샘플을 분류합니다.

- **Technical Details**: 이 연구에서는 15개의 공개된 아랍어 데이터셋을 사용하여 ALDi 점수와 주석자 간의 합의 정도의 관계를 분석하였습니다. 문장별로 ALDi 점수를 계산하고, ALDi 점수와 주석자가 동일한 레이블을 할당한 샘플의 비율(% full agree) 간의 상관관계를 Pearson 상관 계수로 분석하였습니다. 또한, 로지스틱 회귀 모델을 사용해 Full Agreement와 ALDi 간의 이진 결과 분석도 진행했습니다.

- **Performance Highlights**: 연구 결과, 대부분의 데이터셋에서 높은 ALDi 점수를 가진 샘플일수록 주석자 간의 합의가 낮아지는 경향이 나타났습니다. 특히 감정 분석, 풍자 감지, 혐오 발언 감지, 입장 감지 등의 비-방언 식별 과제에서 이 경향이 두드러졌습니다. 반면 방언 식별 과제에서는 높은 ALDi 점수를 가진 샘플에서 더 높은 합의가 이루어졌습니다. 이는 자동 방언 식별 시스템이 이러한 샘플에서 더 높은 정확도를 가질 가능성을 나타냅니다.



### Action Controlled Paraphrasing (https://arxiv.org/abs/2405.11277)
- **What's New**: 이번 연구는 사용자가 설정한 의도를 action tokens으로 표현하는 새로운 방식의 제어된 패러프레이징(paraphrasing)을 제안합니다. 이는 사용자 중심의 디자인을 목표로 하며, 기존 방식들이 요구하던 상세한 구문 트리(parse trees)나 문법 예시(syntactic exemplars) 없이도 제어할 수 있음을 보여줍니다. 또한, 제어 명세가 학습 시에는 제공되지만 실제 추론 시에는 제공되지 않는 문제를 해결하기 위해 새로운 설정을 소개했습니다.

- **Technical Details**: 사용자의 의도를 action tokens으로 표현하여, 이들을 텍스트 임베딩(text embeddings)과 함께 결합해 self-attention 인코더로 전달합니다. 이 과정에서 선택적 action token을 도입하여 제어 명세가 접근 불가능할 때 모델이 적절한 액션을 스스로 결정하도록 합니다. 또한, action embeddings와 단어/위치 임베딩을 함께 사용하여 표현을 융합합니다. 이 접근법은 외부 도구나 수작업 없이 소스 텍스트와 목표 패러프레이즈 간의 차이로부터 자동으로 파생될 수 있는 추가적인 감독 신호로 작동합니다.

- **Performance Highlights**: 실험 결과, 새로운 방식의 제어된 패러프레이징이 특정 액션 제어를 성공적으로 수행하며, 상황에 따라 제어가 없는 경우에도 기존 방법과 동일하거나 더 나은 성능을 보임을 확인하였습니다. 특정 액션 제어는 사용자가 제약을 가하여 다양한 패러프레이징을 촉진시키고, 선택적 액션 제어는 성능과 제약 사이의 절충을 잘 맞춥니다.



### EnviroExam: Benchmarking Environmental Science Knowledge of Large Language Models (https://arxiv.org/abs/2405.11265)
- **What's New**: 최근 환경 과학 분야에서 대규모 언어 모델(Large Language Models)을 평가하기 위한 종합적인 방법인 EnviroExam이 제안되었습니다. EnviroExam은 전 세계 주요 대학의 커리큘럼을 기반으로 하여 학사, 석사, 박사 과정의 42개 핵심 과목에서 936개의 질문을 포함하고 있습니다. 이 평가 방법을 통해 31개의 오픈소스 대규모 언어 모델의 성능을 0-shot 및 5-shot 테스트로 평가하였습니다.

- **Technical Details**: EnviroExam은 하얼빈 공과대학교의 환경 과학 커리큘럼을 기반으로 하여 42개의 핵심 과목을 선정하고, GPT-4와 Claude를 이용한 초기 질문 초안을 생성한 후, 최종적으로 936개의 유효한 질문을 얻었습니다. 31개의 오픈소스 대규모 언어 모델을 대상으로 0-shot 및 5-shot 테스트 방법을 통해 평가하였으며, 테스트 동안 max_out_len=100, max_seq_len=4096, temperature=0.7, top_p=0.95와 같은 매개변수를 사용했습니다.

- **Performance Highlights**: 5-shot 테스트에서는 61.3%의 모델이 합격했으며, 0-shot 테스트에서는 48.39%의 모델이 합격하였습니다. 특히, DeepSeek-67B-Chat이 5-shot 테스트에서 최고 점수인 84.94점을, Llama-3-70B-Instruct가 0-shot 테스트에서 최고 점수인 80.47점을 기록했습니다. Chain-of-Thought (COT) 프롬프팅이 대부분의 모델에서 긍정적인 효과를 보였으나 일부 모델에서는 오류율이 증가하는 부정적인 영향을 미쳤습니다.



### Cross-Language Assessment of Mathematical Capability of ChatGP (https://arxiv.org/abs/2405.11264)
- **What's New**: 이번 논문에서는 ChatGPT가 다양한 언어, 특히 힌디어, 구자라트어, 마라티어 같은 인도 지역 언어에서 수학 문제를 해결하는 능력을 평가합니다. GPT-3.5에 기반한 ChatGPT는 자연어 이해 및 생성 능력으로 주목받고 있지만, 다양한 자연어로 수학 문제를 해결하는 능력은 비교적 미탐구 영역입니다.

- **Technical Details**: 이 연구에서는 chain-of-thought prompting을 사용하여, 이것이 영어에서와 같이 정확도를 얼마나 높이는지 그리고 현재의 한계점을 분석합니다.

- **Performance Highlights**: 논문은 ChatGPT가 영어뿐만 아니라 다양한 인도 지역 언어에서도 수학 문제를 해결할 때 어느 정도의 정확도를 유지하는지에 대한 통찰을 제공합니다.



### WisPerMed at "Discharge Me!": Advancing Text Generation in Healthcare with Large Language Models, Dynamic Expert Selection, and Priming Techniques on MIMIC-IV (https://arxiv.org/abs/2405.11255)
Comments:
          8 pages, 6 tables, 8 figures, submitted to: BioNLP 2024 and Shared Tasks @ ACL 2024

- **What's New**: 혁신적인 언어 모델을 활용해 MIMIC-IV 데이터셋의 '간략 입원 경과'와 '퇴원 지침' 섹션을 자동으로 생성하여 임상의의 행정 업무를 줄이는 연구가 발표되었습니다. 이 연구는 BioNLP @ ACL 2024의 Shared Task Discharge Me!에서 실시되었습니다.

- **Technical Details**: Few-shot learning, instruction tuning, Dynamic Expert Selection (DES) 등의 다양한 전략을 사용하여 필요한 텍스트 섹션을 생성할 수 있는 모델을 개발했습니다. 특히, 추가적인 임상 분야 데이터셋을 활용한 임상 언어 처리가 성과를 보였으며, 여러 가지 예측에서 최적의 텍스트 출력을 선택하는 DES 방식이 매우 효과적이었습니다.

- **Performance Highlights**: DES 방식은 경쟁에서 0.332의 최고 점수를 기록하여 단일 모델 출력을 능가했습니다. 그 결과, 전자 건강 기록 문서화 일부를 자동화하는 데 있어 심도 깊은 학습 방법과 DES의 조합이 효과적이라는 결론을 도출하였습니다.



### Transformer based neural networks for emotion recognition in conversations (https://arxiv.org/abs/2405.11222)
- **What's New**: 이 논문은 SemEval 2024 Task 10: 대화 내 감정 발견 및 감정 전환 분석 (Emotion Discovery and Reasoning its Flip in Conversation, EDiReF)에서 ISDS-NLP 팀의 접근 방식을 설명합니다. 주요 내용으로는 다양한 입력 길이, 분류기 구조 및 미세 조정 방법을 실험하여 감정을 예측하는 Masked Language Modeling (MLM)과 Causal Language Modeling (CLM)을 조사했습니다. 특히, Mistral 7B Instruct V0.2 모델을 사용하여 제로샷 및 소량샷 프롬팅 기술을 적용했습니다.

- **Technical Details**: 다양한 예측 모델을 이용한 실험에서 MLM 기반의 사전 학습된 BERT-like 모델을 다국어 환경에서 미세 조정하여 감정을 예측했습니다. 또한 최신 모델인 Mistral 7B Instruct V0.2를 이용하여 제로샷과 소량샷 프롬팅 기술을 적용했습니다. 제로샷 학습에서는 단일 예제를 제공하고 예측하도록 했으며, 소량샷 학습에서는 각 감정에 대한 여러 예제와 해당 레이블을 제공하여 모델의 성능을 향상시켰습니다. 최적의 시퀀스 길이는 55 토큰으로 설정되었으며, fully connected layer가 가장 우수한 성능을 보였습니다.

- **Performance Highlights**: Subtask 1에서 가중치 F1 스코어 0.43을 기록하며 리더보드에서 12위를 차지했습니다. 다른 서브태스크에서는 14위를 기록했습니다. 제로샷 및 소량샷 학습과 달리 MLM 기반 모델이 문장 수준의 감정 분류에서 더 나은 성능을 보였습니다.



### Identifying and Aligning Medical Claims Made on Social Media with Medical Evidenc (https://arxiv.org/abs/2405.11219)
- **What's New**: 이번 연구는 소셜 미디어에서 의료 주장을 식별하고, 해당 주장과 관련된 의학적 증거를 검색하기 위한 새로운 시스템을 제안합니다. 이 시스템은 PICO (Population, Intervention, Comparator, Outcome) 프레임워크를 사용하여 의료 주장을 분석하며, 새로운 데이터셋인 Expansive Medical Claim Corpus (EMCC)를 소개합니다. 이 연구는 소셜 미디어의 의료 정보를 보다 신뢰할 수 있게 평가할 수 있도록 도와줍니다.

- **Technical Details**: 연구는 세 가지 핵심 과제를 다룹니다: 1) 소셜 미디어 텍스트에서 의료 주장 식별, 2) 해당 주장에서 의학적 용어 추출, 3) 식별된 주장과 관련된 증거 검색입니다. 연구진은 생성적 언어 모델을 사용해 이 과제들을 지원하는 합성 데이터를 생성할 수 있는 시스템을 제안합니다. PICO 요소를 포함한 합성 의료 주장 데이터셋을 생성하여, 이를 통해 의료 주장의 분류 성능을 개선합니다.

- **Performance Highlights**: 제안된 시스템은 PICO 요소를 사용하는 방법론을 기반으로 하여, 소셜 미디어에서 의료 주장을 식별하고 관련된 의학적 증거를 검색하는 데 있어 기존 방법들보다 더 유연하고 종합적인 접근 방식을 제공합니다. 연구 결과는 이 합성 데이터셋을 사용할 때 모든 비교 가능한 메트릭에서 개선된 결과를 보여줍니다. 새로운 데이터셋 EMCC는 공개되어 연구 커뮤니티에 기여할 것입니다.



### MemeMQA: Multimodal Question Answering for Memes via Rationale-Based Inferencing (https://arxiv.org/abs/2405.11215)
Comments:
          The paper has been accepted in ACL'24 (Findings)

- **What's New**: 이 논문은 MemeMQA라는 멀티모달 질의응답 프레임워크를 도입하여 구조화된 질문에 대한 정확한 답변을 제공하고 일관된 설명을 부여하는 것을 목표로 하고 있습니다. 이와 관련하여 MemeMQACorpus라는 새로운 데이터셋을 큐레이션하여, 1,122개의 밈에 관련된 1,880개의 질문과 이에 대한 답변-설명 쌍을 포함하였습니다.

- **Technical Details**: 제안된 프레임워크는 'ARSENAL'이라는 새로운 두 단계 멀티모달 프레임워크로, LLMs(대형 언어 모델)의 추론 능력을 활용하여 MemeMQA를 해결합니다. 이 논문은 다양한 균형평형(baselines)을 사용하여 MemeMQA를 벤치마킹하고 최상의 baseline 대비 ~18% 향상된 답변 예측 정확성과 언어적 및 의미적 정렬을 측정하는 다양한 지표에 걸쳐 뛰어난 텍스트 생성 능력을 입증합니다. 또한, 질문 세트의 다양화, 혼란 기반 평가(confounder-based evaluation)를 통한 MemeMQA의 일반화 가능성, 그리고 모달리티별 평가 방식을 통해 ARSENAL의 견고성을 분석합니다.

- **Performance Highlights**: 이 연구는 MemeMQA 프레임워크의 뛰어난 성능을 뒷받침하며, 기존 최상의 baseline 대비 약 18% 향상된 답변 예측 정확성과 더욱 뛰어난 텍스트 생성 능력을 달성했습니다. 이로써 다양한 관점에서 밈 해석을 향상시키고, 멀티모달 커뮤니케이션 환경에서 밈이 가지는 잠재적 위험을 탐구하는 데 기여할 수 있습니다.



### Automated Text Identification Using CNN and Training Dynamics (https://arxiv.org/abs/2405.11212)
- **What's New**: 이 연구에서는 데이터 맵(Data Maps)을 사용하여 AuTexTification 데이터셋을 모델링하고 특성화했습니다. 이를 통해 개별 샘플의 학습 동태(training dynamics)를 이해하고, 학습이 쉬운 예제, 애매한 예제, 어려운 예제로 분류했습니다. 기존의 CNN 아키텍처를 사용하여 애매한 예제만을 학습함으로써 모델의 분포 외 일반화(out-of-distribution generalization)가 개선됨을 확인했습니다.

- **Technical Details**: AuTexTification 챌린지는 사람과 AI가 생성한 텍스트를 구별하는 복잡한 작업을 해결하는 것을 목표로 합니다. 학습 데이터는 세 가지 도메인에서 가져왔고 테스트 데이터는 두 가지 다른 도메인에서 가져왔습니다. 데이터 맵을 사용하여 각 샘플의 신뢰도, 다양성, 정확성의 세 가지 차원에서 특성화했습니다. CNN 모델은 5개의 Conv1D 레이어, 배치 정규화(BatchNormalization) 레이어, 드롭아웃(Dropout) 레이어와 3개의 완전 연결 레이어로 구성되었습니다.

- **Performance Highlights**: 전체 데이터를 학습한 후 CNN 모델의 F1 점수는 62였고, 애매한 예제만 학습했을 때는 F1 점수 64를 기록했습니다. 또, 애매한 예제의 45%만 학습했을 때 최적의 F1 점수 66.1을 얻었습니다. 이는 전체 데이터의 28%만 사용했을 때의 성과로, 모델이 일부 예제에서 학습을 더 잘한다는 것을 보여줍니다.



### LexGen: Domain-aware Multilingual Lexicon Generation (https://arxiv.org/abs/2405.11200)
- **What's New**: 본 논문에서는 6개의 인도 언어에 대해 다중 도메인 설정에서 사전 단어를 생성하는 새로운 모델 LexGen을 제안합니다. LexGen 모델은 도메인 특정 계층과 도메인 일반 계층으로 구성되어 있으며, 학습 가능한 라우팅 기술을 통해 이러한 계층을 호출합니다. 또한 인도 언어 간의 관련성을 명시적으로 활용하여 일관된 번역을 가능하게 합니다. 이를 통해 8개의 다양한 도메인을 아우르는 새로운 벤치마크 데이터셋을 공개하여 도메인 특정 사전 유도 연구를 촉진합니다.

- **Technical Details**: LexGen은 도메인 특정 네트워크와 공유 네트워크 간의 정보 흐름을 선택적으로 허용하는 토큰 수준 활성화 게이팅 메커니즘을 도입합니다. 다중 계층의 Transformer 기반 모델을 사용하며, 도메인 감도 있는 아키텍처를 통해 향상된 번역 품질을 제공합니다. 또한 예상 번역을 가이드하기 위해 영어 원문 구문에 표준 산스크리트어 번역을 추가합니다.

- **Performance Highlights**: 제안된 모델은 인도 언어와 다양한 도메인 전반에서 기존 베이스라인에 비해 평균 2-5%가 향상된 결과를 보여줍니다. 특히 제안된 도메인 라우팅 (Domain Routing, DR) 층과 인도 언어 간 공통 어간을 활용함으로써 이러한 성과를 이끌어 내었습니다.



### Designing NLP Systems That Adapt to Diverse Worldviews (https://arxiv.org/abs/2405.11197)
- **What's New**: 이 연구에서는 자연어 추론(NLI) 모델의 한계를 지적하며, 이러한 문제들이 의미의 주관적인 성격을 무시한 결과라고 주장합니다. 기존의 NLP 데이터셋은 다양한 주관적 견해를 포함하지 않고, 레이블을 집계하거나 불일치를 걸러내는 방식으로 작동해왔습니다. 저자들은 주관주의적 접근법(perspectivist approach)을 제안하며, 주석자의 인구통계, 가치관, 레이블에 대한 정당성을 포함하는 데이터셋 구축을 제안합니다.

- **Technical Details**: 본 연구는 SBIC 데이터셋의 일부를 사용하여 제한된 주석자 메타데이터만으로도 모델 성능이 향상될 수 있음을 초기 실험에서 입증했습니다. 주석자의 세계관(Weltanschauung)을 반영하는 데이터셋을 구축함으로써 모델의 다양성을 높이고, 다중 주석(multilabel categorical classification), 소프트 레이블 분류(soft label classification), 급진적 주관주의(radical perspectivist)와 같은 방법론을 제시합니다.

- **Performance Highlights**: 기존의 약한 주관주의적 접근법에서는 다중주석을 통해 다양한 관점을 포착하는 시도를 했으나, 여전히 불일치가 높은 샘플을 제거하거나 집계하여 인간 커뮤니케이션의 다채로운 해석을 희미하게 만들었습니다. 저자들은 이를 극복하기 위해 주석자의 세계관을 직접 모델링할 필요성을 강조하며, 초기 실험에서 주석자 메타데이터를 포함한 접근법이 모델 성능을 개선하는 것을 확인했습니다.



### BrainStorm @ iREL at SMM4H 2024: Leveraging Translation and Topical Embeddings for Annotation Detection in Tweets (https://arxiv.org/abs/2405.11192)
Comments:
          Submitted to SMM4H, colocated at ACL 2024

- **What's New**: 이 논문에서는 COVID-19 증상 탐지를 위한 트윗에서 대규모 언어 모델(LLMs)과 도메인 전문가가 작성한 주석(annotations)을 구분하는 새로운 접근법을 제안합니다. 특히, BrainStorm @ iREL 팀이 제안한 방법은 트윗 내 주제별 정보를 활용하여 주석의 신뢰성을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: 이 접근법은 데이터셋 준비에서 시작하여, 원본과 번역된 트윗 텍스트를 사용하여 다양한 모델의 성능을 평가합니다. 스페인어 트윗을 영어로 번역한 후, BERT(bert-base-spanish-wwm-cased)와 BERTweet(bertweet-covid19-base-cased)을 사용하여 모델 성능을 비교합니다. 또한, 토픽 모델링을 위해 BERTopic을 사용하여 각 트윗에 주제 레이블을 부여하고 이를 토큰화된 표현에 추가하여 분류기의 성능을 향상시킵니다.

- **Performance Highlights**: 결과적으로, 스페인어와 영어 버전의 토픽 임베딩 삽입은 원래 스페인어 트윗의 경우 점수 향상이 없었으며, 번역된 영어 트윗에서는 약간의 향상(0.51)을 보였습니다. 최종적으로, 이 방법이 주석 구분에 있어서 약간의 향상을 보였지만, 상당한 성능 향상을 이루기 위해서는 추가적인 고도화된 기술 또는 추가적인 기능이 필요함을 시사합니다.



### Automating PTSD Diagnostics in Clinical Interviews: Leveraging Large Language Models for Trauma Assessments (https://arxiv.org/abs/2405.11178)
- **What's New**: 새로운 연구는 임상 인력이 부족한 정신 건강 진료의 문제를 해결하기 위해 맞춤형 대규모 언어 모델(LLM)을 통합하는 방법을 제안합니다. 특히, 외상 후 스트레스 장애(PTSD) 진단을 위한 자동화 시스템을 개발하는 데 중점을 두고 있습니다. 이를 위해 411개의 임상 인터뷰 데이터를 수집하고, 주요 기여로는 PTSD 평가를 자동화할 수 있는 포괄적인 프레임워크 구축을 포함합니다. 이는 환자와 임상가 간의 실질적인 인터뷰를 해석할 수 있는 최초의 AI 시스템입니다.

- **Technical Details**: 본 연구에서는 최신의 두 개 LLM, GPT-4 및 Llama-2를 이용하여 PTSD 진단 평가를 자동화하는 시스템을 구축합니다. 이를 위해 700시간 이상의 인터뷰 데이터를 포함하는 새로운 데이터를 생성하고, 인터뷰 내용을 기반으로 하는 정보 추출 및 텍스트 요약을 통해 진단 질문에 답변할 수 있는 모형을 개발했습니다. 또한, 해당 시스템은 향후 다양한 임상 진단 인터뷰에 쉽게 적용될 수 있도록 설계되었습니다.

- **Performance Highlights**: 연구 결과는 통계 분석 결과 GPT-4와 Llama-2가 PTSD 진단에 있어 높은 정확성을 보여주었음을 나타냅니다. 특히, 이 시스템은 실질적인 임상 인터뷰 데이터를 사용함으로써 임상 전문가들이 진단 검증을 수행하는 데 큰 도움을 줄 것으로 기대됩니다. 이는 정신 건강 진단을 자동화하는 데 있어서 혁신적인 한 걸음을 내딛는 연구로 평가됩니다.



### LG AI Research & KAIST at EHRSQL 2024: Self-Training Large Language Models with Pseudo-Labeled Unanswerable Questions for a Reliable Text-to-SQL System on EHRs (https://arxiv.org/abs/2405.11162)
Comments:
          NAACL 2024 Clinical NLP Workshop

- **What's New**: 전자 건강 기록(EHR) 접근성을 텍스트-투-SQL(text-to-SQL) 모델을 통해 개선함으로써 SQL 지식이 없는 의료 전문가들이 복잡한 질문을 SQL 쿼리로 변환하여 기록을 조회할 수 있게 하는 연구입니다. 우리는 EHRs를 위한 텍스트-투-SQL 모델의 신뢰성을 높이기 위해 Pseudo-Labeled Unanswerable Questions(가상 라벨링된 비답변 질문)을 활용한 셀프 트레이닝(self-training) 전략을 제안합니다. 이 접근법은 토큰 엔트로피와 쿼리 실행에 기반한 필터링 절차를 포함한 이중 단계 트레이닝 프로세스를 포함합니다.

- **Technical Details**: 제안된 방법은 셀프 트레이닝을 활용하여 모델의 예측 성능을 향상시키고 비답변 질문을 더 잘 식별하도록 합니다. 첫 번째 단계에서는 주어진 훈련 데이터셋을 사용하여 모델을 미세 조정합니다. 다음 단계에서는 미세 조정된 모델이 식별한 비답변 질문을 포함하도록 훈련 데이터셋을 보강합니다. 이를 통해 모델을 다시 한 번 미세 조정합니다. 이중 단계 트레이닝 프로세스 후, 최대 엔트로피 값과 쿼리 실행 결과를 기반으로 필터링 과정을 거쳐 불확실한 예측을 제거합니다.

- **Performance Highlights**: 제안된 PLUQ 방법론은 EHRSQL 2024 공동 과제에서 최고의 성과를 기록하여, 텍스트-투-SQL 시스템의 신뢰성을 높여 더 나은 의료의사결정을 지원하는 잠재력을 입증했습니다.



### A Reproducibility Study on Quantifying Language Similarity: The Impact of Missing Values in the URIEL Knowledge Bas (https://arxiv.org/abs/2405.11125)
Comments:
          NAACL 2024 SRW

- **What's New**: 이번 연구는 다국어 자연 언어 처리(NLP)을 지원하기 위한 언어 특성을 특징짓는 도구들이 중요한 역할을 한다는 점에서 시작되었습니다. 주로 사용되는 전형적 지식 기반인 URIEL을 집중 분석하여 언어 유사성을 수량화하는 접근 방식의 타당성과 재현 가능성을 조사했습니다. 특히, URIEL의 언어 거리 계산과 누락 값 처리의 모호성이 드러났고, URIEL이 나타내는 언어 중 31%는 전형적 특징에 대한 정보를 제공하지 않는다는 점을 발견했습니다.

- **Technical Details**: URIEL은 4,005개의 언어에 대한 언어 정보를 다양한 데이터 소스에서 수집하여 숫자 벡터로 집계하는 지식 기반입니다. URIEL은 데이터를 단일 벡터로 통합하여 언어의 전형적 특징을 측정합니다. 하지만 언어 거리 계산 방법과 누락 값 처리 방법에 대한 명확한 문서화가 부족했습니다. URIEL은 문법적(syntax), 음운적(phonology), 어휘적(inventory) 특성을 포함한 여러 타입의 특징을 다룹니다. 그러나 현재 URIEL은 전체 언어 중 31%에 대해 전형적 특징 정보를 제공하지 않습니다.

- **Performance Highlights**: URIEL이 제공하는 언어 거리의 사전 계산 값을 재현하려는 시도에서, 특히 kNN 집계 벡터에서 완전히 일치하는 값을 얻지 못했습니다. 문서화에는 사용된 거리 계산 방법과 누락 값 처리 방법에 대한 모호함이 있으며, 이는 연구에서 발견되었듯이, 누락 값이 있을 경우 벡터를 모두 1의 값으로 채우거나 일부 값만 누락된 경우 0으로 대체하는 방법을 사용해야 가장 근접한 결과를 얻을 수 있었습니다.



### Dynamic Embeddings with Task-Oriented prompting (https://arxiv.org/abs/2405.11117)
- **What's New**: 이 논문은 Dynamic Embeddings with Task-Oriented prompting (DETOT)을 소개합니다. 이는 머신 러닝 모델의 적응성과 효율성을 개선하기 위한 새로운 접근 방식입니다. DETOT는 전통적인 정적 임베딩(static embeddings)과는 다르게, 태스크(task)별 요구사항과 성능 피드백에 따라 임베딩을 동적으로 조정합니다.

- **Technical Details**: DETOT는 개별 태스크의 입력 데이터 표현을 최적화하기 위해 임베딩 레이어를 유연하게 조정합니다. 이 방법은 태스크 특정의 적응성, 지속적인 피드백 루프, 오버피팅 방지 메커니즘을 강조합니다. 기술적으로는 구조를 통해 평가 성능과 계산 성능 모두를 향상시킵니다.

- **Performance Highlights**: 실증 평가를 통해 기존 방법들보다 우수한 성능을 보였으며, DETOT는 정확성과 계산 효율성 측면에서 현존하는 방법들을 능가합니다.



### Multilingual Substitution-based Word Sense Induction (https://arxiv.org/abs/2405.11086)
- **What's New**: 새로운 다국어 치환 기반의 WSI(Word Sense Induction) 방법이 제안되었습니다. 이는 기본 다국어 언어 모델(multilingual language model)을 통해 100개 언어를 지원하며, 최소한의 조정으로 높은 성능을 유지합니다. 특히, 영어 WSI 데이터셋에서 단일언어 접근법과 비교해도 동등한 성능을 보여줍니다. 이 방법은 어휘 자원이 부족한 언어에 가장 유용할 것입니다.

- **Technical Details**: 제안된 다국어 WSI 방법은 XLM-R(Conneau et al., 2020)과 같은 다국어 언어 모델을 사용하여 치환 모델을 구축하고, 이를 통해 각 인스턴스에 대한 치환 벡터를 생성한 후 클러스터링합니다. BERT LSDP와 +embs와 같은 기존 단일언어 기반의 치환 접근법을 다국어 환경에 적용했습니다. XLM-R을 통해 BERT의 역할을 대체하며 성능을 유지할 수 있습니다.

- **Performance Highlights**: 논문에서는 11개 언어의 데이터셋을 사용해 엄격한 평가와 분석을 수행했습니다. 다국어 Masked Language Model을 통해 언어 간의 어휘적 치환을 생성하는 데 큰 능력을 보였습니다. 이는 기존 단일언어 접근법과 동등한 성능을 입증했습니다.



### Prompt Exploration with Prompt Regression (https://arxiv.org/abs/2405.11083)
- **What's New**: 최근 대형 언어 모델(LLMs)의 대중화와 더불어, LLM 프롬프트(prompt) 생성 및 선택 과정의 체계화를 향한 요구가 증가하고 있습니다. 기존의 연구들은 주로 프롬프트 변형 간의 관계를 고려하지 않은 채 프롬프트 공간을 탐색하는 데 초점을 맞추었으나, 본 논문에서는 프롬프트 회귀(Prompt Regression)와 프롬프트 선택(Selection)을 결합한 프레임워크인 PEPR(Prompt Exploration with Prompt Regression)을 제안합니다. 이를 통해 개별 프롬프트 요소의 결과를 바탕으로 프롬프트 조합의 효과를 예측하고, 주어진 사용 사례에 적합한 효과적인 프롬프트를 선택할 수 있는 간단한 방법을 제공합니다.

- **Technical Details**: PEPR은 세 개의 주요 단계를 포함합니다. 먼저, 주어진 작업에 대해 프롬프트 라이브러리를 구축합니다. 그런 다음, PEPR의 첫 번째 부분인 프롬프트 회귀 과정에서 각 프롬프트 라이브러리 요소가 LLM 출력에 미치는 영향을 기반으로 매개 변수 가중치를 도출합니다. 이 가중치를 사용하여 PEPR의 두 번째 부분인 프롬프트 선택 과정에서 원하는 동작에 따른 프롬프트 요소를 선택합니다. 이 프롬프트 선택 단계 이후 전체 프롬프트가 복원됩니다. 프롬프트 회귀 및 선택 부분은 참조 텍스트 생성 또는 인간 라벨 선호 정보를 활용하여 제공된 데이터에 맞는 프롬프트를 도출할 수 있습니다. PEPR의 기초 이론과 가정은 프롬프트 회귀 섹션에서 자세히 설명되며, LLM과 주어진 프롬프트 라이브러리의 조합이 모델의 행동에 미치는 영향을 쉽게 예측할 수 있습니다.

- **Performance Highlights**: PEPR의 효과를 검증하기 위해 여러 오픈 소스 LLM과 다양한 데이터셋 및 과제를 사용해 평가를 진행했습니다. 그 결과, PEPR은 기존의 반복적 시도-오류 방법보다 효율적이고 효과적으로 최적의 프롬프트를 도출할 수 있음을 확인했습니다. 특히, 브라운(Brown) 등의 연구(2020)와 바이(Bai) 등의 연구(2022)에서처럼, PEPR의 프롬프트 회귀 및 최적화 과정은 in-context learning 예제 또는 LLM 원칙으로 구성된 프롬프트 라이브러리와 함께 작동할 수 있다는 점에서 탁월한 성능을 보였습니다.



### Leveraging discourse structure for the creation of meeting extracts (https://arxiv.org/abs/2405.11055)
- **What's New**: 회의 요약을 위한 새로운 추출적 요약 시스템을 소개합니다. 이 시스템은 복잡한 다자간 대화에서 중요한 정보를 더 잘 식별하기 위해 담화 구조를 활용합니다. 담화 그래프(discourse graphs)를 사용해 회의 발화의 내용 간 의미 관계를 나타내며, GNN(Graph Neural Networks) 기반 노드 분류 모델을 훈련시켜 가장 중요한 발화를 선택합니다. 선택된 발화는 결합되어 추출적 요약을 생성합니다. AMI와 ICSI 데이터셋에 대한 실험 결과, 기존의 텍스트 기반 및 그래프 기반 추출적 요약 시스템을 능가하는 성과를 보였습니다.

- **Technical Details**: 본 연구는 담화 구조 정보를 활용하여 추출적 요약(extractive summarization)을 개선합니다. 그래프 신경망(GNN)과 개별 발화의 내용을 나타내는 그래프의 결합을 통해 각 발화 노드의 중요성을 판단하고, 중요한 것으로 판단된 노드의 내용을 최종 요약본으로 결정합니다. 이는 전체 그래프 수준의 생성과 달리 발화의 중요성과 전반적인 담화에서의 역할 간의 상호작용을 세밀하게 분석할 수 있게 합니다. 또한, GNN은 주어진 노드의 이웃에만 집중하기 때문에 Transformers 모델의 맥락 길이 제한을 받지 않습니다.

- **Performance Highlights**: 우리의 접근 방식은 AMI와 ICSI 데이터셋에서 F1 점수, ROUGE, BERTScore와 같은 평가 지표에서 기존 방법보다 뛰어난 성과를 보였습니다. 다양한 그래프 구성 전략의 요약 품질에 미치는 영향을 분석하여 담화 구조가 콘텐츠 선택에 미치는 메커니즘에 대한 통찰을 제공했습니다. 이러한 연구는 자동 회의 요약 분야에 대한 새로운 방법론적 기여를 제공하며, 효과적인 요약을 구성하는 기본 담화 과정을 조명합니다.



### From Generalist to Specialist: Improving Large Language Models for Medical Physics Using ARCo (https://arxiv.org/abs/2405.11040)
Comments:
          8 pages, 3 figures, 1 table

- **What's New**: 대형 언어 모델(LLMs)은 일반적인 분야에서 뛰어난 성과를 보였으나, 의료 물리학과 같은 전문 분야에서의 적용은 제한적입니다. 이 연구는 ARCoT(Adaptable Retrieval-based Chain of Thought)이라는 프레임워크를 도입하여 LLMs의 전문 분야 정확성을 개선하려 합니다. 이 프레임워크는 검색 메커니즘을 통합해 관련 정보를 접근하고, step-back 및 chain-of-thought 프롬프트 기법을 사용해 LLM의 추론 과정을 안내합니다.

- **Technical Details**: ARCoT는 RAG(검색 기반 생성)와 CoT(chain-of-thought) 프롬프트 기법을 결합해 모델 성능을 향상시킵니다. Step-Back(SB) 프롬프트 전략을 통해 검색된 문서의 관련성을 최적화하고, 재랭킹 변환기를 통해 입력 쿼리에 가장 관련성 높은 내용을 우선적으로 선택합니다. 또한, 약 60여 개의 문서를 텍스트 파일로 변환해 벡터 데이터베이스를 구축했습니다. 각 청크는 OpenAI의 임베딩 모델(text-embedding-ada-002)을 사용해 임베딩 되었으며, 코사인 유사도를 통해 가장 유사한 벡터 임베딩을 검색합니다.

- **Performance Highlights**: ARCoT는 의료 물리학 다지선다형 시험에서 기존의 LLMs를 능가하며 평균 인간 점수와 비교해 최대 68% 개선을 보였으며, 최고 점수 90%를 달성했습니다. 이 프레임워크는 환각을 줄이고, 전문 분야의 성능을 증가시킵니다.



### CC-GPX: Extracting High-Quality Annotated Geospatial Data from Common Craw (https://arxiv.org/abs/2405.11039)
- **What's New**: Common Crawl (CC) 코퍼스에서 지리적 데이터(geospatial data)를 주석 처리하여 사용자 생성 트랙을 추출하는 효율적인 파이프라인이 도입되었습니다. 이로 인해 사람들의 야외 활동 패턴 및 활동 경로에 대한 설명을 연구하거나 경로 생성 모델 개발에 사용할 수 있는 새로운 다중 모드 데이터셋이 생성되었습니다.

- **Technical Details**: 새로운 파이프라인은 2008년 이후의 Common Crawl 데이터에서 사용자 생성 GPX 파일(GPX Exchange Format)을 식별, 다운로드 및 정리합니다. 이 과정에서 6개의 최신 CC 릴리스(CC-MAIN-2023-*에서 CC-MAIN-2024-10) 데이터를 사용하여 1,416개의 다중 언어로 된 활동 경로(features)와 텍스트 설명을 포함하는 데이터셋을 생성했습니다. DuckDB를 사용하여 GPX 파일의 MIME 유형을 검색하고, 파이썬의 requests 라이브러리와 Range 요청 헤더를 사용하여 각각의 GPX 파일을 빠르게 다운로드하였습니다. 또한 GPX 파일은 gpxpy 라이브러리를 사용하여 구문 분석되었습니다.

- **Performance Highlights**: 이 고유 데이터셋은 1,400개 이상의 주석 처리된 MultiLineString 특징(pairing)을 포함하고 있으며, 이 데이터셋을 통해 사용자의 야외 경험에 대한 설명을 연구하거나 야외 활동 궤적 연구를 수행할 수 있습니다. 데이터셋은 6개의 최신 CC 릴리스에서 추출된 파일 중 약 94,170개의 고유한 GPX 파일을 포함합니다.



### The Unappreciated Role of Intent in Algorithmic Moderation of Social Media Conten (https://arxiv.org/abs/2405.11030)
- **What's New**: 이 논문은 온라인 학대(content moderation) 탐지 시스템에서 '의도(intent)'의 역할을 탐구합니다. 기존 플랫폼 정책과 탐지 모델 간의 격차를 메꾸기 위해 의도를 포함하는 전략적 변경을 제안하는 것이 주목할 만합니다.

- **Technical Details**: 논문은 온라인 학대, 특히 증오발언(hate speech)과 사이버 불링(cyberbullying)을 탐지하는 최신 모델과 벤치마크 데이터셋을 검토합니다. 기존 탐지 모델들은 의도(intent)를 제대로 포착하지 못하여 발생하는 문제들을 강조하며, 의도를 포함한 새로운 탐지 및 중재 파이프라인을 제안합니다. 의도는 주로 텍스트에서 직접적으로 보이지 않으며, 정보를 추상화하여 판별하는 것이 어렵습니다.

- **Performance Highlights**: 기존 탐지 모델들은 문맥을 이해하지 못하므로, 의도 파악에서 많은 한계를 드러냅니다. 특히 언급된 ToxicBert 모델은 'I’m going to kill you if you leave the dishes for me again' 문장을 위협적이라고 잘못 판단하는 등 직접적이고 형식적이지 않은 언어를 분별하는 데 어려움을 겪습니다.



### The Arabic Noun System Generation (https://arxiv.org/abs/2405.11014)
Comments:
          In Proceedings of The International Conference on Arabic Processing, Lamanouba University, April 2002, Tunisia

- **What's New**: 이 논문에서는 깨진 복수 패턴(broken plural pattern)을 가진 명사들에 대한 다중-어간 접근법이 형태학적 체계에서 더 큰 일반화를 가능하게 한다는 것을 보여줍니다. 이러한 접근법은 고도의 동형성(allomorphic)을 가진 깨진 복수 시스템을 설명하기 위해 필요한 복잡한 규칙들을 생략합니다.

- **Technical Details**: 복수로 변화된 명사들의 생성을 위해서는 사전에 남성형 복수인 uwna와 여성형 복수인 aAt로 나타나는 접미사의 사전 규정이 필요합니다. 첫 번째 섹션에서는 아랍어 깨진 복수에 대한 이전의 분석들을 평가합니다. 두 번째 섹션에서는 어군기반 형태학(Lexeme-based Morphology) 프레임워크 내에서 아랍어 명사 복수 체계를 위한 다중-어간 접근법을 제안합니다. 세 번째 섹션에서는 아랍어 명사 시스템을 MORPHE에 구현한 내용을 제공합니다.

- **Performance Highlights**: 논문은 단수나 어근에서 깨진 복수를 유도하는 것에 대한 언어적 및 통계적 증거를 제시하지 않습니다. 언어적 분석 섹션에서 논의된 일반화가 Morphe의 동등화 노드(equivalencing nodes)를 통해 어떻게 포착되는지를 보여줍니다.



### Eliciting Problem Specifications via Large Language Models (https://arxiv.org/abs/2405.12147)
Comments:
          18 pages, Appendix. Submitted to Advances in Cognitive Systems 2024

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 활용하여 자연어로 정의된 문제의 클래스를 반형식적 사양으로 변환하고, 이를 기존의 추론 및 학습 시스템에서 활용하여 문제를 해결하는 방법을 소개합니다. 특히 LLM 에이전트를 구현하여 자연어로 명시된 작업의 문제 공간 정의를 자동으로 생성하는 시스템을 제안합니다. 이러한 접근법은 문제의 정의를 신속하게 수행하면서도 강력한 추론 및 온라인 학습 기능을 유지할 수 있음을 시사합니다.

- **Technical Details**: 제안된 시스템은 'How to Solve It'과 같은 일반적인 문제 해결 전략에서 도출된 LLM 프롬프트를 사용하여 문제 공간을 정의합니다. 문제 공간은 문제 클래스에 속하는 여러 문제를 접근할 수 있는 방식(상태와 작업자)을 정의하며, 이를 통해 개별 문제 인스턴스를 표현하고 해결할 수 있습니다. 이를 위해 CTA(Cognitive Task Analysis) 기법을 사용하여 LLM 에이전트가 문제 분석을 수행하도록 설계되었습니다.

- **Performance Highlights**: 초기 결과는 인간이 문제를 정의하는 과정을 단순화하면서도 강력한 추론 및 학습 기능을 유지할 수 있음을 시사합니다. 이는 인지 시스템 연구의 속도를 높이는 잠재력을 가지며, 문제 정의 단계에서 사람의 개입을 줄이는 데 큰 기여를 할 수 있습니다.



### Reindex-Then-Adapt: Improving Large Language Models for Conversational Recommendation (https://arxiv.org/abs/2405.12119)
- **What's New**: 대형 언어 모델(LLMs)을 사용하는 대화형 추천 시스템에서 추천 아이템의 분포 제어 문제를 해결하기 위해 Reindex-Then-Adapt(RTA) 프레임워크를 제안합니다. 이 프레임워크는 멀티토큰 아이템 제목을 싱글 토큰으로 변환하고, 그런 다음 이러한 싱글 토큰 아이템 제목의 확률 분포를 조정합니다. 이는 복잡한 질문을 이해하는 LLMs의 장점과 추천 아이템 분포를 효율적으로 제어하는 전통적인 추천 시스템(RecSys)의 장점을 결합합니다.

- **Technical Details**: RTA 프레임워크는 먼저 멀티토큰 아이템 제목을 싱글 토큰으로 변환하는 '재인덱싱(reindex)' 단계와 변환된 싱글 토큰 아이템 제목의 분포를 조정하는 '적응(adapt)' 단계를 포함합니다. 이를 통해 모델은 더 효율적으로 추천 확률 분포를 제어할 수 있습니다. 네 가지 재인덱싱 모듈과 두 가지 적응 전략을 세 가지 대화형 추천 데이터셋에서 시험해서, 시스템 성능을 개선했습니다.

- **Performance Highlights**: 우리의 프레임워크는 세 가지 대화형 추천 데이터셋과 두 가지 적응 설정에서 정확도 지표를 향상시켰습니다. 예를 들어, 원본 Llama2-7b 모델의 Top-10 Hit Rate를 59.37% 향상시키며 모든 오픈 소스 기준 모델을 능가했습니다.



### Imp: Highly Capable Large Multimodal Models for Mobile Devices (https://arxiv.org/abs/2405.12107)
Comments:
          19 pages, 6 figures

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 능력을 활용하여, 여러 모달리티를 포함하는 대형 멀티모달 모델(LMMs)의 경량화에 대한 체계적인 연구를 제시합니다. 특히, 2B에서 4B 규모의 Imp 모델 패밀리를 개발하였으며, 이 중 Imp-3B 모델은 기존의 유사한 크기의 경량 LMM들을 능가하고, 심지어 13B 규모의 최첨단 LMM보다도 뛰어난 성능을 발휘한다고 보고했습니다.

- **Technical Details**: 이 연구에서는 모델 아키텍처, 훈련 전략, 훈련 데이터 세 가지 측면에서 경량 LMMs의 설계 선택을 체계적으로 조사했습니다. LLaVA-1.5 모델을 기반으로 오픈소스 경량 LLMs(Phi-2, Phi-3 등)을 활용하여 Imp-2B, Imp-3B, Imp-4B 모델을 개발했습니다. 이들 모델은 낮은 비트의 양자화(low-bit quantization)와 이미지 해상도 저하 기술을 통해 Qualcomm Snapdragon 8Gen3 모바일 칩에서 초당 약 13개의 토큰을 처리할 수 있습니다.

- **Performance Highlights**: Imp-3B 모델은 유사한 크기의 모든 기존 경량 LMM을 능가하며, 13B 규모의 최첨단 LMM보다도 높은 성능을 보입니다. 또한, 본 연구에서는 독점적인 사전 학습 모델이나 비공개 훈련 데이터를 사용하지 않음으로써 재현 가능성을 보장했습니다. 최적화된 Imp-3B 모델은 모바일 디바이스에서도 높은 추론 속도를 보여줍니다. 코드와 사전 학습된 모델은 Github와 HuggingFace에서 공개되었습니다.



### KG-RAG: Bridging the Gap Between Knowledge and Creativity (https://arxiv.org/abs/2405.12035)
- **What's New**: 이 논문은 KG-RAG (Knowledge Graph-Retrieval Augmented Generation) 파이프라인이라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 구조화된 지식 그래프(KG)와 대형 언어 모델(LLM)의 기능을 통합하여, LLM의 잠재적인 지식 의존도를 크게 줄이고자 합니다. 이를 통해 정보의 환각 현상, 기억 상실, 긴 컨텍스트 처리의 한계를 해결하려는 시도입니다.

- **Technical Details**: KG-RAG 파이프라인은 비구조화된 텍스트에서 KG를 생성한 후, 새로 생성된 그래프에서 정보 검색을 수행하여 KGQA (Knowledge Graph Question Answering)를 수행합니다. 이 과정에서 새로운 알고리즘인 Chain of Explorations(CoE)를 사용하여 LLM의 추론 능력을 활용하여 KG 내의 노드와 관계를 순차적으로 탐색합니다. 이는 지식을 구조화된 방식으로 연결하고, 더 정확하고 신뢰할 수 있는 지식 보관소를 제공하여 정보 환각 문제를 줄입니다.

- **Performance Highlights**: ComplexWebQuestions 데이터셋에 대한 초기 실험 결과, KG-RAG 파이프라인이 환각된 콘텐츠를 줄이는 데에서 현저한 개선을 보이며, 지식 집약적인 작업을 처리하는 데 능숙한 지능형 시스템 개발의 유망한 길을 제시합니다.



### On Efficient and Statistical Quality Estimation for Data Annotation (https://arxiv.org/abs/2405.11919)
Comments:
          Accepted to ACL 2024

- **What's New**: 이 연구는 주석(annotated) 데이터셋의 품질 관리를 개선하기 위한 새로운 접근 방식을 제안합니다. 주석 데이터의 품질을 평가하기 위해 신뢰 구간(confidence intervals)을 사용하여 최소 샘플 크기(sample size)를 결정하는 방식을 상세히 설명하고, 수락 검사(acceptance sampling)를 통한 대안적인 오류율(error rate) 추정 방법을 제시합니다. 이를 통해 주석 품질 평가의 효율성을 높일 수 있습니다.

- **Technical Details**: 기존의 주석 데이터 품질 평가 방법은 많은 비용이 소요될 수 있는 모든 주석된 인스턴스를 수동으로 확인하는 것을 포함합니다. 보통은 샘플의 크기가 충분히 큰지 알 수 없기에, 작은 샘플로 인해 오류율 추정의 정확도가 떨어질 위험이 존재합니다. 본 연구에서는 신뢰 구간을 활용해 필요한 최소 샘플 크기를 결정하고, 이를 통해 오류율을 보다 정확히 추정할 수 있는 방법을 설명합니다. 또한, 수락 검사(acceptance sampling)를 이용하여 필요한 샘플 크기를 최대 50%까지 줄이는 방안을 제시하며, 여전히 동일한 통계적 보장(statistical guarantees)을 유지합니다.

- **Performance Highlights**: 수락 검사(acceptance sampling)를 적용한 결과, 기존 방법에 비해 샘플 크기를 최대 50%까지 줄일 수 있었으며, 이는 주석 데이터 품질 평가의 효율성과 비용 절감을 동시에 달성할 수 있음을 보여줍니다.



### Quantifying In-Context Reasoning Effects and Memorization Effects in LLMs (https://arxiv.org/abs/2405.11880)
- **What's New**: 이 연구에서는 대규모 언어 모델(LLM)이 언어 생성에 사용되는 정확한 기억(memorization) 및 맥락 내 추론(in-context reasoning) 효과를 정의하고 정량화하기 위한 공리 시스템(axiomatic system)을 제안합니다. 이 시스템은 LLM이 인코딩한 토큰/단어 간의 비선형 상호작용으로 이 효과를 공식화합니다.

- **Technical Details**: 공리 시스템은 기억 효과를 기본 기억 효과(foundation memorization effects)와 혼란 기억 효과(chaotic memorization effects)로, 맥락 내 추론 효과를 강화된 추론 패턴(enhanced inference patterns), 제거된 추론 패턴(eliminated inference patterns), 그리고 반전된 추론 패턴(reversed inference patterns)으로 분류합니다. 이 분해된 효과는 희소성(sparsity)과 보편적 매칭(universal matching) 속성을 만족하며, 이는 LLM의 신뢰도 점수를 기억 및 맥락 내 추론 효과로 충실히 분해할 수 있음을 수학적으로 보장합니다.

- **Performance Highlights**: 실험 결과, 기억 효과와 맥락 내 추론 효과의 명확한 분리는 LLM이 인코딩한 세부 추론 패턴을 명확하게 검토할 수 있게 함을 보여줍니다.



### Systematic Review on Healthcare Systems Engineering utilizing ChatGP (https://arxiv.org/abs/2405.11817)
- **What's New**: 이 논문은 Healthcare Systems Engineering 분야의 학술 리뷰를 위해 최신 언어 모델 중 하나인 ChatGPT를 활용한 분석 프레임워크를 제시합니다. 저자들은 학술 회의 발표에서 9,809개의 초록 단락을 사용하여 체계적으로 이 분야를 검토하였습니다.

- **Technical Details**: 이 프레임워크는 고유한 분석 프로세스로 구성되어 있으며, 각 프로세스는 맞춤형 프롬프트(prompt)와 ChatGPT API의 체계적인 사용을 포함합니다. 이를 통해 대상 분야를 11개의 주제 카테고리로 조직하고, 정량적 연간 트렌드와 세부 하위 카테고리를 포함하는 포괄적인 분석을 수행했습니다.

- **Performance Highlights**: ChatGPT의 활용을 통해 학술 리뷰에 필요한 시간과 노력을 줄이는 가능성을 탐구하였습니다. 또한, Healthcare Systems Engineering 연구의 역동적인 지형에 대한 귀중한 통찰력을 제공합니다.



### Inverse Design of Metal-Organic Frameworks Using Quantum Natural Language Processing (https://arxiv.org/abs/2405.11783)
Comments:
          45 pages, 7 figures, 6 supplementary figures, 1 table, 1 supplementary table

- **What's New**: 이번 연구에서는 양자 자연 언어 처리(Quantum Natural Language Processing, QNLP)를 활용하여 목표 속성을 가진 금속-유기 구조체(Metal-Organic Frameworks, MOFs)를 역설계하는 가능성을 탐구했습니다. 총 150개의 가상 MOF 구조를 분석하고, 이러한 구조를 기공 부피(pore volume)와 수소 흡착($H_{2}$ uptake) 값에 따라 네 가지 클래스(class)로 분류했습니다.

- **Technical Details**: 다양한 QNLP 모델, 즉 bag-of-words, DisCoCat(Distributional Compositional Categorical), 시퀀스 기반 모델(sequence-based models)을 비교하여 MOF 데이터셋을 처리하는 데 가장 효과적인 접근 방식을 식별했습니다. IBM Qiskit에서 제공하는 클래식 시뮬레이터를 사용하여, bag-of-words 모델이 최적임을 확인했습니다. 구체적으로, 이 모델은 기공 부피와 $H_{2}$ 흡착에 대한 이진 분류(binary classification) 작업에서 각각 85.7%와 86.7%의 검증 정확도를 달성했습니다. 또, 양자 회로의 확률적 특성에 맞춘 다중 클래스 분류 모델(multi-class classification models)을 개발하여, 기공 부피와 $H_{2}$ 흡착 데이터셋의 다양한 클래스에서 평균 테스트 정확도 88.4%와 80.7%를 기록했습니다.

- **Performance Highlights**: 목표 속성을 가진 MOF를 생성하는 성능은 기공 부피에서 93.5%, $H_{2}$ 흡착에서 89%의 정확도를 보였습니다. 비록 이번 연구가 방대한 MOF 탐색 공간의 일부만을 다루었지만, 이는 양자 컴퓨팅을 활용한 재료 설계(materials design)의 유망한 첫 걸음을 내디뎠음을 시사합니다.



### ColorFoil: Investigating Color Blindness in Large Vision and Language Models (https://arxiv.org/abs/2405.11685)
- **What's New**: 이번 연구에서는 새로운 Vision and Language (V&L) 벤치마크인 ColorFoil을 소개합니다. 이 벤치마크는 색깔 관련 foils(오답) 생성 접근법을 통해 V&L 모델의 색깔 인식 능력을 평가합니다. 이를 통해 모델들이 붉은색(red), 흰색(white), 녹색(green) 등의 색깔 인식을 얼마나 잘 수행하는지를 조사합니다.

- **Technical Details**: ColorFoil 벤치마크는 MS COCO와 Flickr30k 데이터셋에서 자동으로 유도됩니다. 각 이미지-텍스트 페어는 색깔 관련 단어를 포함하며, 원본 이미지와 텍스트의 나머지 부분은 그대로 두고 색깔 관련 단어만 수정합니다. Python의 webcolors 1.3 패키지를 사용하여 색깔 관련 단어를 식별하고, 가장 많이 사용되는 색상에 집중합니다. 평가된 모델은 CLIP, ViLT, GroupViT, BridgeTower 등입니다.

- **Performance Highlights**: 실험 결과, ViLT와 BridgeTower는 CLIP 및 그의 변형 모델 및 GroupViT보다 더 우수한 색 인식 능력을 보여주었습니다. 특히, CLIP-based 모델과 GroupViT는 일반적인 색 인지 능력을 가진 인간이 쉽게 구분할 수 있는 색깔을 구분하는 데 어려움을 겪었습니다.



### Inquire, Interact, and Integrate: A Proactive Agent Collaborative Framework for Zero-Shot Multimodal Medical Reasoning (https://arxiv.org/abs/2405.11640)
- **What's New**: 최근 아카이브된 논문에서는 의료 분야에서 대형 언어 모델(LLMs)의 활용에 대해 탐구하고 있습니다. 한계점으로는 도메인 특화된 지식과 의료 추론 능력의 부족, 그리고 텍스트만 처리할 수 있는 단일 모달 특성 때문입니다. 이를 해결하기 위해 'MultiMedRes'라는 다중 모달 의료 협업 추론 프레임워크를 제안하며, 도메인 특화 전문가 모델들로부터 필요한 정보를 능동적으로 얻어내어 의료 다중 모달 추론 문제를 해결합니다.

- **Technical Details**: MultiMedRes는 세 가지 단계로 구성됩니다: 1) Inquire (질문): 복잡한 의료 추론 문제를 다양한 도메인 특화 하위 문제로 분해하고, 2) Interact (상호작용): 전문가 모델과 '질문-답변' 과정을 반복하며 필요한 도메인 지식을 얻고, 3) Integrate (통합): 취득한 모든 지식을 통합하여 의료 추론 문제를 해결합니다. 이 접근법은 임상 의사들이 협력하는 방식을 모방하여 설계되었습니다.

- **Performance Highlights**: MIMIC-Diff-VQA 데이터셋에서의 실험 결과, MultiMedRes는 최첨단 성능을 기록했으며, 완전한 지도 학습 법보다도 뛰어난 결과를 보였습니다. 또한, MultiMedRes는 다양한 LLMs와 다중 모달 LLMs에 통합되어 성능을 크게 향상시킬 수 있습니다.



### SLAB: Efficient Transformers with Simplified Linear Attention and Progressive Re-parameterized Batch Normalization (https://arxiv.org/abs/2405.11582)
Comments:
          Accepted to ICML 2024

- **What's New**: 최근 논문에서는 Transformer의 계산 병목 모듈인 normalization 레이어와 attention 모듈을 효율적으로 개선하는 방법을 제안했습니다. LayerNorm은 통계 계산 때문에 비효율적인데 비해, 좀 더 효율적인 BatchNorm으로 대체하면 성능이 저하되고 학습이 불안정해질 수 있습니다. 이를 개선하기 위해 PRepBN이라는 점진적으로 LayerNorm을 BatchNorm으로 대체하는 방법을 도입했습니다. 더불어 간소화된 linear attention (SLA) 모듈도 제안되었습니다.

- **Technical Details**: PRepBN은 학습 초반에는 LayerNorm, 후반으로 갈수록 BatchNorm의 비율을 높여가는 방식으로, 이를 통해 학습 붕괴를 방지하고 성능을 안정화할 수 있습니다. SLA 모듈은 ReLU를 커널 함수로 사용하고 depth-wise convolution을 통합하여 지역적 피처 강화를 수행합니다. 이러한 방법들을 통해 Transformer의 계산 비용을 낮추면서도 성능을 유지할 수 있습니다.

- **Performance Highlights**: ImageNet-1K 데이터셋에서 SLAB-Swin 모델은 83.6%의 top-1 정확도를 기록했으며, 지연 시간은 16.2ms로 Flatten-Swin 모델보다 2.4ms 더 짧았습니다. 또한, 언어 모델링 태스크에서도 유사한 성능과 더 낮은 지연 시간을 보여줬습니다.



### DocReLM: Mastering Document Retrieval with Language Mod (https://arxiv.org/abs/2405.11461)
- **What's New**: 최근 발표된 DocReLM 시스템은 대형 언어 모델(LLMs)을 활용하여 학술 문서 검색을 효율적으로 개선하는 것을 목표로 합니다. 특히, 양자 물리학과 컴퓨터 비전 연구 분야에서 기존 시스템보다 월등한 성능을 보였습니다.

- **Technical Details**: DocReLM 시스템은 크게 세 가지 구성 요소로 나뉩니다. 첫 번째는 'retriever'로, 사용자의 쿼리에 맞는 후보 문서를 신속하게 검색하는 역할을 합니다. 두 번째는 'reranker'로 검색된 후보 문서를 더욱 정교하게 정렬합니다. 마지막으로 'reference extractor'는 상위 결과 문서의 내용을 분석하여 더 적합한 참고 문서를 추출합니다. 이 과정에서 대형 언어 모델이 사용되며, 각 구성 요소는 특정 데이터 세트로 훈련됩니다.

- **Performance Highlights**: DocReLM은 컴퓨터 비전 분야에서 Top 10 정확도가 44.12%로 Google Scholar의 15.69%를 크게 상회했으며, 양자 물리학 분야에서도 36.21%로 Google Scholar의 12.96%를 능가했습니다. 실험 결과, DocReLM은 Top 5 결과에서 컴퓨터 비전 38.73%와 양자 물리학 26.91%의 정확도를 기록하여, 학술 문서 검색의 혁신적 가능성을 보여주었습니다.



### Du-IN: Discrete units-guided mask modeling for decoding speech from Intracranial Neural signals (https://arxiv.org/abs/2405.11459)
- **What's New**: 공격적인 뇌-컴퓨터 인터페이스(BCI) 분야에서 침습적(stereoElectroEncephaloGraphy, sEEG) 뇌 신호 모델의 대표성을 높이기 위한 새로운 'Du-IN 모델'이 개발되었습니다. 특히, 다변량 표현을 통해 특정 뇌 영역에서의 신경 처리를 더 잘 포착할 수 있게 하려는 시도를 했으며, 체계적으로 잘 주석된 중국어 단어 읽기 sEEG 데이터셋을 사용하여 SOTA(State-of-the-Art) 성능을 달성했습니다.

- **Technical Details**: Du-IN 모델은 vSMC와 STG 영역의 채널을 융합하여 다변량 표현을 구축하고, 이산 코드북 유도 마스크 모델링(discrete codebook-guided mask modeling)을 통해 자기 감독(self-supervision)을 수행합니다. 이 모델은 빠른 시간적 변화와 특정 뇌 영역의 정확한 상태를 포착할 수 있는 의미 있는 토큰을 추출하여, 이후 다운스트림 작업에 사용하도록 설계되었습니다.

- **Performance Highlights**: 61개의 단어 분류 작업에서 Du-IN 모델은 최고 62.70%의 정확도를 달성하며, 모든 기존 베이스라인 모델을 초과하는 성능을 보여줍니다. 모델 비교 및 소거 분석(ablation analysis)을 통해 다변량 표현과 자기 감독 기법이 성능 향상에 크게 기여하는 것으로 나타났습니다.



### EmbSum: Leveraging the Summarization Capabilities of Large Language Models for Content-Based Recommendations (https://arxiv.org/abs/2405.11441)
Comments:
          Under review

- **What's New**: 디지털 세계에서 개인 맞춤형 콘텐츠를 제공하는 데 중요한 역할을 하는 콘텐츠 기반 추천 시스템이 이번 연구에서 EmbSum이라는 새로운 프레임워크를 소개합니다. EmbSum은 사용자와 후보 항목을 오프라인으로 사전 계산하여 사용자 참여 기록 내의 상호작용을 포착합니다. 사전 학습된 인코더-디코더 모델과 폴리-어텐션(poly-attention) 레이어를 활용하여 사용자 폴리-임베딩(User Poly-Embedding, UPE)과 콘텐츠 폴리-임베딩(Content Poly-Embedding, CPE)을 도출하고, 이를 통해 사용자와 후보 항목 간의 관련 점수를 계산합니다. EmbSum은 대형 언어 모델(LLM)의 감독하에 사용자 관심 요약을 생성하여 긴 사용자 참여 기록을 효과적으로 학습합니다.

- **Technical Details**: EmbSum은 사전 학습된 인코더-디코더 모델과 폴리-어텐션 레이어를 사용합니다. 이러한 구성 요소를 통해 사용자 폴리-임베딩(UPE)과 콘텐츠 폴리-임베딩(CPE)을 도출하며, 이는 사용자와 후보 항목 간의 관련성을 평가하는 데 사용됩니다. 이 시스템은 오프라인에서 사용자와 후보 항목에 대한 계산을 미리 수행함으로써 실시간 계산 부담을 줄입니다. 또한, 대형 언어 모델(LLM)의 감독을 받아 사용자 관심 요약을 생성함으로써, 긴 사용자 참여 기록을 효율적으로 학습할 수 있습니다.

- **Performance Highlights**: EmbSum은 다른 도메인에서 가져온 두 가지 데이터셋에서 최첨단(SoTA) 방법을 능가하는 성능을 검증받았습니다. 높은 정확성과 더 적은 수의 파라미터를 바탕으로 기존 방법보다 뛰어난 성과를 보였습니다. 또한, EmbSum의 사용자 관심 요약 생성 기능은 개인 맞춤형 콘텐츠 추천을 위한 유용한 부가 기능으로 작용합니다.



### Metric Dimension and Resolvability of Jaccard Spaces (https://arxiv.org/abs/2405.11424)
Comments:
          12 pages, 1 table

- **What's New**: 새로운 연구는 Jaccard 공간 (Jaccard spaces)의 해상 가능성 (resolvability)을 다루고 있습니다. 이 논문은 특히, 해상 집합이 최소 크기인 경우, Jaccard 거리로 정의된 메트릭 공간에서 해상 집합을 구성하는 방법을 탐구합니다.

- **Technical Details**: 메트릭 공간에서 해상 집합 (resolving set)은 공간의 각 점을 유일하게 식별할 수 있는 점들의 하위 집합을 의미합니다. 논문에서는 Jaccard 공간을 $(2^X, 	ext{Jac})$ 형태로 정의하며, 여기서 $2^X$는 유한 집합의 멱 집합 (power set)이고, $	ext{Jac}$는 두 하위 집합 사이의 Jaccard 거리입니다. Jaccard 거리는 두 집합의 대칭 차이 (symmetric difference)와 합집합의 크기로 정의됩니다. 저자들은 확률론적 (probabilistic) 및 선형대수학적 (linear algebra) 논증을 통해 거의 최적 (near-optimal)인 해상 집합을 구성합니다.

- **Performance Highlights**: 논문에서는 $(2^X, 	ext{Jac})$의 메트릭 차원 (metric dimension)이 $	heta(|X|/	ext{ln}|X|)$임을 보여줍니다. 이는 주어진 공간에서 해상 집합의 최소 크기가 $|X|/	ext{ln}|X|$ 정도라는 것을 의미합니다.



### Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts (https://arxiv.org/abs/2405.11273)
Comments:
          22 pages, 13 figures. Project Website: this https URL. Working in progress

- **What's New**: 최신 연구가 제시한 Uni-MoE는 Mixture of Experts (MoE) 아키텍처를 활용한 최초의 통합 멀티모달 대형 언어 모델(MLLM)입니다. 이 모델은 여러 모달리티를 처리할 수 있으며, 각 모달리티에 특화된 인코더와 커넥터를 사용해 통합된 멀티모달 표현을 형성합니다.

- **Technical Details**: Uni-MoE는 모달리티 특화 인코더와 커넥터를 통해 다양한 모달리티를 언어 표현 공간으로 매핑합니다. 이러한 방식은 sparse MoE 아키텍처를 활용하여 효율적인 학습과 추론을 가능하게 합니다. 트레이닝 단계에서는 크로스 모달리티 정렬, 모달리티 특화 전문가 학습, Low-Rank Adaptation(LoRA)을 사용하는 세 단계로 진행되며, 이를 통해 다중 전문가의 협업 및 일반화를 촉진합니다.

- **Performance Highlights**: 광범위한 벤치마크를 통한 실험 결과, Uni-MoE는 혼합 멀티모달 데이터셋을 처리할 때 성능 편향을 크게 줄이고, 다중 전문가 협업과 일반화를 개선하는 데 뛰어난 성과를 보여줍니다. 특히, 복잡한 도메인 외 작업에서도 두드러지는 성능을 발휘하며, 싱글 모달리티 작업에서도 각각의 성능이 향상되었습니다.



### BadActs: A Universal Backdoor Defense in the Activation Spac (https://arxiv.org/abs/2405.11227)
Comments:
          ACL2024 Findings

- **What's New**: 백도어 공격은 DNNs (Deep Neural Networks)의 개발 단계에서 점점 더 심각한 보안 위협이 되고 있습니다. 이러한 위협에 대응하기 위해, 백도어 샘플 정화(purification) 방법이 제안되었습니다. 이 논문에서는 활성화 공간(activation space)에서 백도어 샘플을 정화하는 보편적인 백도어 방어 방법을 소개합니다. 이 접근법은 비정상적인 활성화를 최소한의 깨끗한 활성화 분포 구간으로 유도함으로써, 다양한 트리거를 극복하며, 깨끗한 데이터의 완전성을 최대한 보존합니다.

- **Technical Details**: 이 방법은 활성화 공간에서 작동하여 표면 수준 정보(예: 단어)에서 높은 수준의 의미론적 개념(예: 구문)까지 캡처함으로써 다양한 트리거를 해결합니다. 또한, 활성화 공간이 미세하고 연속적이기 때문에 트리거를 제거하면서 깨끗한 콘텐츠를 더 정확하게 보존할 수 있습니다. 더불어, 비정상적인 활성화의 통계 정보를 기반으로 한 탐지 모듈을 도입하여 깨끗한 데이터와 방어 성능 간의 균형을 최적화합니다.

- **Performance Highlights**: 이 논문에서 제안된 방어 체계인 BaDActs는 네 가지 데이터셋과 네 가지 다른 공격 유형에서 방어 효율성과 깨끗한 데이터의 정확성 면에서 최첨단 성능을 달성합니다. 특히, 이전의 정화 방법이 실패한 피처(Feature)-공간 트리거에 대해 BaDActs는 효과적으로 방어할 수 있습니다. 또한, BaDActs는 활성화 수준의 정규화를 통한 적응형 공격에 대해 저항력이 있는 것으로 나타났습니다.



### Towards Knowledge-Infused Automated Disease Diagnosis Assistan (https://arxiv.org/abs/2405.11181)
- **What's New**: 이번 연구에서는 환자-의사 상호작용을 통해 질병을 식별하는 지식 주입(knowledge-infused) 및 담화 인식(discourse-aware) 진단 모델인 KI-DDI를 제안합니다. 이 모델은 환자와 의사의 대화를 Transformer 기반 인코더로 인코딩하고, 증상-질병 임베딩을 Graph Attention Network(GAT)로 생성합니다.

- **Technical Details**: KI-DDI 모델은 두 개의 채널로 구성됩니다. 첫 번째 채널은 환자-의사 대화를 Transformer 기반 인코더로 인코딩합니다. 두 번째 채널은 증상과 질병 간의 관계를 표현한 지식 그래프를 GAT로 임베딩합니다. 이후 이 두 임베딩을 합쳐 딥러닝 네트워크에 입력하여 질병을 식별합니다. 또한, Empathetic conversational medical corpus를 개발하여 환자와 의사 간의 대화를 의도(intent)와 증상 정보로 주석(annotation) 처리했습니다.

- **Performance Highlights**: 제안된 KI-DDI 모델은 기존의 최신 모델보다 성능이 크게 향상되었습니다. 이는 환자 자가보고(self-report)에서 추가적인 증상 추출의 중요성과 의학 지식을 통합하여 질병을 정확하게 식별하는 데 기여함을 보여줍니다.



### Towards Modular LLMs by Building and Reusing a Library of LoRAs (https://arxiv.org/abs/2405.11157)
- **What's New**: 이 논문에서는 다중 작업 데이터에서 학습된 어댑터(adapter)를 새 작업에 맞춰 재사용하기 위한 새로운 방법을 제안합니다. 주요 기여는 다중 작업 데이터에 대한 모델 기반 클러스터링(MBC) 방법과, 새로운 입력에 대해 적절한 어댑터를 동적으로 선택하는 화살표(Arrow)라는 제로샷 라우팅 메커니즘입니다.

- **Technical Details**: 기존의 어댑터 라이브러리를 구축하는 접근 방식을 벤치마킹하고, 어댑터 파라미터의 유사성을 기반으로 작업을 그룹화하는 MBC 방법을 도입합니다. 이 방법은 다중 작업 데이터셋 전반에 걸쳐 전이 성능 최적화를 간접적으로 돕습니다. 새로운 시나리오에서 마치 새로 학습하지 않은 것처럼 라이브러리를 재사용하기 위해, 각 LoRA를 동적으로 선택하는 Arrow라는 새로운 라우팅 메커니즘을 제안합니다. 이 메커니즘은 훈련 데이터에 접근하지 않고도 직렬적으로 선택을 수행할 수 있습니다.

- **Performance Highlights**: Phi-2와 Mistral을 포함한 여러 LLM과 다양한 홀드아웃 작업에 대해 실험한 결과, MBC 기반의 어댑터와 Arrow 라우팅은 새로운 작업에 대한 일반화 성능이 우수한 것으로 나타났습니다. Zero-shot 일반화 및 감독된 적응에서 기존의 합동 훈련 방법과 비교해 모듈식 LLM의 잠재력을 실증하였습니다.



### OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework (https://arxiv.org/abs/2405.11143)
- **What's New**: 대형 언어 모델(LLM)의 스케일링 법칙에 따라 강화 학습(=Reinforcement Learning)의 인간 피드백(RLHF)이 주목받고 있는 가운데, OpenRLHF라는 오픈소스 프레임워크를 소개합니다. 이 프레임워크는 70억 개 이상의 파라미터를 가진 모델을 효율적으로 교육하기 위해 설계되었으며, Ray, vLLM, DeepSpeed를 활용해 자원 활용도를 개선하고 다양한 학습 접근 방식을 지원합니다.

- **Technical Details**: 새로운 모델 스케줄링 접근 방식을 채택하여, 기존 RLHF 프레임워크와 달리 네 개의 모델을 동일 GPU에 배치하지 않고, 여러 GPU에 분산 배치합니다. 이를 통해 메모리 제약을 극복하고 모델 교육의 효율성을 높였습니다. OpenRLHF는 Hugging Face와의 완벽한 연동을 제공하며, Mixture of Experts, Jamba, QLoRA와 같은 최신 기술도 지원합니다. 또한 RLHF, DPO, 거부 샘플링 등의 정렬 기법을 구현하여, 사용자가 쉽게 접근할 수 있도록 설계되었습니다.

- **Performance Highlights**: OpenRLHF는 모델 교육의 효율성을 극대화하기 위해 다양한 최적화 기술을 적용했습니다. 예를 들어, Adam 옵티마이저 상태를 CPU로 오프로드 하여 GPU 메모리를 해방시키고, 플래시 어텐션 2를 사용하여 Transformer 모델의 교육을 가속화했습니다. 또한, 네 개의 모델을 ZeRO 단계 3을 사용해 샤딩하고, NVIDIA NCCL 및 vLLM 무게 로더를 활용해 무게를 동기화하여 빠르고 간편한 통합을 보장합니다. 이러한 기술적 개선을 통해 LLaMA2 모델 교육 시 DSChat보다 훨씬 더 효율적인 결과를 얻었습니다.



### Enhancing Watermarked Language Models to Identify Users (https://arxiv.org/abs/2405.11109)
Comments:
          37 pages

- **What's New**: 이번 논문에서는 AI 생성 텍스트를 특정 사용자나 그룹에 추적할 수 있는 다중 사용자 워터마킹(Multi-user watermarks)을 도입했습니다. 기존의 제로-비트 워터마킹(zero-bit watermarking)에서 파생된 이 기법은 텍스트가 기계에 의해 생성되었는지를 검출할 뿐만 아니라, 생성된 텍스트를 특정 사용자의 API 토큰과 연결지어 추적할 수 있게 합니다. 이를 통해 스팸이나 금전 사기 등 악용을 방지할 수 있습니다.

- **Technical Details**: 다중 사용자 워터마킹은 'AEB-강인성(AEB-robustness)'이라는 새로운 추상화를 도입하여, 워터마크가 편집된 텍스트에서도 검출 가능하도록 보장합니다. 기존의 워터마킹 기법들은 단일 프롬프트에 대한 보안만 보장했지만, 이번 연구는 사용자들이 인터랙티브하게 모델을 질의하고 전체 상호작용에서 텍스트를 도출할 때의 강인성을 증명합니다. 이러한 기법은 적응형 프롬프트(adaptive prompting)에도 대응하며, 여러 사용자가 협력해 작성한 텍스트에서도 특정 사용자를 추적할 수 있습니다.

- **Performance Highlights**: 단일 사용자가 아닐 경우 텍스트를 추적하는 시간은 O(log n)로 최적화되었으며, 다중 사용자가 협력한 경우에도 O(n log n) 시간 내에 추적이 가능합니다. 이 기법은 기존의 제로-비트 워터마킹 기법의 강인성을 유지하면서도, 텍스트의 정확성과 품질 저하가 없습니다. Google과 OpenAI 등 주요 기술 기업들은 이 기법을 도입하여 AI 생성 콘텐츠의 출처를 추적하고 악용을 방지하려는 계획을 세우고 있습니다.



### LLM-based Multi-Agent Reinforcement Learning: Current and Future Directions (https://arxiv.org/abs/2405.11106)
Comments:
          8 pages, 1 figure, 1 table, submitted to IEEE RA-L

- **What's New**: 최근 몇 년 동안 대규모 언어 모델(LLMs)은 질문 응답, 산수 문제 해결, 시 작성 등 다양한 작업에서 뛰어난 능력을 보여주었습니다. LLM 기반 강화 학습(RL)이 단일 에이전트 환경에서는 상당한 성과를 이루었지만, 다중 에이전트 시스템(MAS)으로의 확장은 여러 측면에서 쉽지 않습니다. 특히 에이전트 간의 협력 및 통신 등이 단일 에이전트 RL 프레임워크에서는 고려되지 않았기 때문입니다. 이번 논문에서는 기존의 LLM 기반 단일 에이전트 및 다중 에이전트 RL 프레임워크를 조사하고, 향후 연구 방향을 제시합니다.

- **Technical Details**: 이번 연구에서는 LLM 기반 RL의 다중 에이전트 시스템(MARL, Multi-Agent Reinforcement Learning) 확장을 중심으로 합니다. 우리는 공동 목표를 가진 여러 에이전트의 협력적 작업(cooperative tasks)과 그들 간의 통신(communication)을 중점적으로 다룹니다. 또한 언어 구성 요소를 활용하여 인간이 개입하는 시나리오(Human-in/on-the-loop scenarios)도 고려하고 있습니다.

- **Performance Highlights**: 현재까지의 연구에서는 LLM이 단일 에이전트 강화 학습(RL) 환경에서 성공적인 결과를 보였으나 다중 에이전트 시스템(MAS)에서는 충분히 탐구되지 않았습니다. 앞으로의 연구 방향으로는 LLM 기반 마르(MARL)의 효과적인 구현 및 에이전트 간의 협력과 통신 측면에서의 발전이 필요합니다.



### Are Large Language Models Moral Hypocrites? A Study Based on Moral Foundations (https://arxiv.org/abs/2405.11100)
Comments:
          13 pages, 4 figures, 2 tables

- **What's New**: 최근 대형 언어 모델(LLMs)인 GPT-4와 Claude 2.1이 인간의 중요한 가치와 얼마나 부합하는지를 평가하는 연구가 발표되었습니다. 이 연구는 LLM들이 도덕적 위선(moral hypocrisy)을 보이는지 여부를 조사합니다.

- **Technical Details**: 이 연구는 도덕적 기초 이론(Moral Foundations Theory)을 기반으로 하는 두 가지 도구를 사용합니다: (1) 도덕적 기초 설문지(Moral Foundations Questionnaire, MFQ), 추상적 도덕 판단에서 어떤 가치가 도덕적으로 중요한지를 조사하는 도구와 (2) 도덕적 기초 상황극(Moral Foundations Vignettes, MFVs), 각 도덕적 기초와 관련된 구체적인 시나리오에서의 도덕 인지를 평가하는 도구입니다.

- **Performance Highlights**: 연구 결과 GPT-4와 Claude 2.1은 각각의 도구 내에서는 인간과 비교해 합리적인 일관성을 보였으나, MFQ에서 나타난 추상적 가치와 MFV에서의 구체적 도덕 위반 평가를 비교할 때 모순적이고 위선적인 행동을 보였습니다. 이는 두 모델이 선언한 도덕적 가치와 구체적 행동 평가 간에 불일치가 있음을 시사합니다.



### AudioSetMix: Enhancing Audio-Language Datasets with LLM-Assisted Augmentations (https://arxiv.org/abs/2405.11093)
- **What's New**: 최근 오디오-언어 멀티모달 학습 분야에서 AudioSetMix라는 새로운 데이터셋을 발표하였습니다. 이 데이터셋은 기존의 AudioSet 클립을 강화하고 이에 따라 자연어 설명을 생성함으로써 만들어졌습니다. 이는 대규모 데이터셋의 부재와 품질 문제를 해결하는 데 기여합니다.

- **Technical Details**: AudioSetMix는 LLM(Large Language Model)을 이용해 오디오 클립에 맞춘 설명을 생성합니다. 데이터 전처리, 오디오 변환 및 LLM 기반 설명 생성 등 네 가지 단계로 구성된 파이프라인을 활용합니다. 오디오 변환에는 속도, 피치, 볼륨, 길이의 조정 및 여러 클립의 믹싱과 연결이 포함됩니다.

- **Performance Highlights**: 새로운 데이터셋을 사용하여 Language-Based Audio Retrieval에서 최첨단 성능을 달성하였고, 모델이 볼륨, 시간 등의 음성 이벤트 수식어를 더 잘 이해하도록 도왔습니다. 또한, hard negative mining 기술을 도입하여 모델 성능을 추가로 향상시켰습니다.



### Jill Watson: A Virtual Teaching Assistant powered by ChatGP (https://arxiv.org/abs/2405.11070)
- **What's New**: 이번 논문은 대화형 인공지능 에이전트 'Jill Watson'의 새로운 버전을 소개합니다. Jill Watson은 ChatGPT를 활용하여 별도의 훈련 없이 다양한 API와 통합할 수 있는 모듈식 설계를 특징으로 합니다. 이 새로운 Jill Watson은 교과서 관리와 같은 지능형 텍스트북 분야에서도 적합하며, 여러 대규모 문서를 처리하고 대화할 수 있습니다. 공개된 자원을 독점적으로 사용하여 재현성과 확장성을 높였습니다.

- **Technical Details**: Jill Watson의 아키텍처는 XiaoIce의 스킬 기반(대화 기능 기반) 설계에 영향을 받아 여러 API 서비스를 쉽게 통합할 수 있게 디자인되었습니다. 이는 과거의 메시지를 컨텍스트로 사용하여 사용자의 질문에 응답할 때 내용의 일관성을 유지할 수 있습니다. ChatGPT를 백엔드로 사용하며, Dense Passage Retrieval (DPR)을 통해 문서에서 적합한 구절을 검색하여 답변을 생성합니다. 안전성을 확보하기 위해 OpenAI Moderation API, 질문 관련성 분류기, 그리고 정중한 응답을 장려하는 프롬프트를 사용합니다.

- **Performance Highlights**: 비교 분석 결과, 새로 개발된 Jill Watson 시스템은 기존의 지식 기반 Jill Watson 및 OpenAI Assistants 서비스보다 뛰어난 성능을 보였습니다. 특히, 허위 생성(hallucination)과 독성 텍스트 생성을 줄이는 많은 안전 장치들이 적용되었습니다. 실제 교실 환경에서의 다양한 사례를 통해 효율성과 다양한 기능을 입증했습니다.



### Generative Artificial Intelligence: A Systematic Review and Applications (https://arxiv.org/abs/2405.11029)
- **What's New**: 이 논문은 최근 몇 년간의 생성 인공지능(Generative AI)의 발전을 체계적으로 검토하고 분석한 내용을 담고 있습니다. 논문은 특히 자연어 처리, 이미지 변환, 의료 진단 등에서의 생성 모델의 응용과 관련된 최신 기술적 진전을 중점적으로 다루고 있습니다.

- **Technical Details**: 생성 인공지능은 텍스트, 이미지, 기타 미디어를 생성할 수 있는 AI 시스템을 말하며, 이는 주로 Generative Adversarial Networks (GANs), Transformers, Variational Autoencoders (VAEs), Diffusion Models 등으로 구현됩니다. 각 기법의 구조와 특성뿐만 아니라, 최근의 응용 분야 및 기술적 도전 과제도 상세히 설명합니다.

- **Performance Highlights**: 논문은 생성 인공지능의 주요 응용 분야에서의 성능을 다룹니다. 주요 사례로는 대규모 언어 모델을 활용한 언어 생성, 이미지 변환, 텍스트와 이미지의 융합, 그리고 자연어 처리 등이 있습니다. 또한, 각 기술의 성과를 표준 데이터셋을 통해 평가하여 비교 분석합니다.

- **Responsible AI Principles**: 논문은 생성 인공지능의 지속 가능성과 성장을 위해 필요한 책임감 있는 AI 원칙과 윤리적 고려 사항을 마지막으로 논의합니다. 이는 생성 인공지능의 개발과 응용에 있어 필수적인 부분으로 강조됩니다.



### Petri nets in modelling glucose regulating processes in the liver (https://arxiv.org/abs/2405.11009)
Comments:
          submitted to International Workshop on Petri Nets and Software Engineering (PNSE 2024)

- **What's New**: 새로운 논문에서는 당뇨병의 생물학적 기초를 보다 잘 이해하기 위해 간 내에서 일어나는 포도당 분해(glycolysis)와 포도당 합성 과정을 다룬 페트리 네트(Petri net) 모델을 제시합니다. 이 모델은 의학 문헌을 바탕으로 작성되었으며, 페트리 네트의 표준 기술을 이용해 특성을 분석합니다.

- **Technical Details**: 모델링에는 덫(traps), 도달성 그래프(reachability graphs), 토큰 동역학(tokens dynamics), 교착 상태 분석(deadlocks analysis) 같은 페트리 네트 기법이 사용되었습니다. 이러한 분석을 통해 모델이 금식 시와 식사 시 발생하는 생물학적 과정과 일치하는 효소와 물질 간의 상호작용을 포착하고 있음을 확인했습니다.

- **Performance Highlights**: 이번 페트리 네트 모델은 당뇨병 환자와 건강한 사람의 체내 포도당 조절 과정을 포괄적으로 이해할 수 있게 합니다. 이는 새로운 치료 타겟 발굴과 약물 치료 설계에도 도움이 될 것으로 보입니다. 이 모델은 당뇨병 관리 시스템 개선에 기여할 가능성이 큽니다.



### Large Language Models for Tuning Evolution Strategies (https://arxiv.org/abs/2405.10999)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models, LLMs)을 활용한 피드백 루프 메커니즘을 제안하여 Evolution Strategies (ES) 매개변수를 효과적으로 튜닝하는 방법을 제시합니다. 이 접근법은 프로그래밍 명령을 제공하고 해당 코드를 실행한 후 철저히 분석하는 구조화된 프로세스를 포함합니다. LLaMA3 모델을 사용한 실험을 통해 이 방법의 실현 가능성을 입증합니다.

- **Technical Details**: LLM은 자연어 지시를 받으면 실행 가능한 코드로 번역하고, 이를 실행한 후 성능을 평가합니다. 이러한 프로세스는 반복적 주기(iterative cycle)를 통해 ES 매개변수를 지속적으로 개선하게 됩니다. 이 논문은 (1+1)-ES와 같은 ES 알고리즘의 매개변수 조정을 위해 LLM을 사용합니다. 주요 매개변수로는 변이 스케일링 요인 시그마(σ)와 적응 메커니즘의 변이 매개변수 타우(τ)가 있습니다.

- **Performance Highlights**: 실험 결과, LLaMA3 모델을 사용하여 Python 코드와 ES의 매개변수를 분석한 결과, 전형적인 실행에서 τ 값이 0.8에서 1.2 사이일 때 최적의 피트니스 값이 도출됨을 발견했습니다. 특히 τ = 0.95가 최적의 성능을 제공할 가능성이 높다고 결론지었습니다.



### Learnable Privacy Neurons Localization in Language Models (https://arxiv.org/abs/2405.10989)
Comments:
          ACL 2024 main conference

- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)이 개인 식별 정보(PII)를 기억하고 유출할 수 있는 문제를 해결하기 위해 PII-민감 뉴런(privacy neurons)을 찾아내는 새로운 방법을 제안했습니다. 이 방법은 학습 가능한 이진 가중치 마스크를 사용하여 뉴런을 정확하게 위치시키고, 적대적 학습(adversarial training)을 통해 PII를 기억하는 특정 뉴런을 찾습니다.

- **Technical Details**: 논문에서는 LLM 재교육 단계에서 하드 콘크리트 분포(hard concrete distribution)를 활용한 학습 가능한 뉴런 마스크를 사용해 뉴런을 로컬라이즈(Localize)하고, 특정 PII 토큰 시퀀스의 예측 정확도를 최소화하는 목표 함수를 설계합니다. 이를 통해 PII를 기억하는 뉴런들을 최소화된 하위 집합으로 제한합니다.

- **Performance Highlights**: 실험 결과, LLM의 모든 층에 걸쳐 소수의 뉴런이 PII를 기억한다는 사실을 확인했습니다. 특히, MLP 층에 많이 분포해 있으며, 특정 카테고리의 PII 지식에 대해 특이성을 보였습니다. 로컬라이즈된 뉴런을 비활성화하여 PII 유출을 방지할 수 있다는 것을 실험적으로 증명하였으며, 모델의 일반 성능에는 영향을 미치지 않음을 확인했습니다.



### Bottleneck-Minimal Indexing for Generative Document Retrieva (https://arxiv.org/abs/2405.10974)
Comments:
          Accepted for ICML 2024

- **What's New**: 이 논문에서는 정보 이론적 관점에서 생성적 문서 검색(Generic Document Retrieval, GDR)을 재고하려는 새로운 접근 방식을 소개합니다. GDR에서는 문서 x를 인덱스 t로 표현하며, 신경망 자율회귀 모델을 통해 쿼리 Q를 T로 매핑합니다. Shannon의 레이트-디스토션(rate-distortion) 이론을 적용하여 인덱스의 최적성을 상호 정보(mutual information)로 분석하고, 인덱스 T의 설계를 '병목 정보'(bottleneck) 관점에서 재정의합니다.

- **Technical Details**: GDR의 첫 번째 단계에서 문서 세트 𝒳에 대해 고유한 인덱스 문자열 𝒯를 얻는 문제를 해결하기 위해 레이트-디스토션 이론과 정보 병목 이론을 적용합니다. 이를 통해 문서 X와 쿼리 Q 간의 정보 전송 문제를 해결합니다. 논문에서는 NQ320K와 MARCO 데이터셋을 사용하여 새로운 병목-최소화 인덱싱 방법을 평가하고, 기존 인덱싱 방법들과 비교하여 우수함을 보였습니다.

- **Performance Highlights**: 제안된 인덱싱 방법은 NQ320K 데이터셋에서 Recall@1 점수가 1.26 포인트, MARCO 데이터셋에서는 3.72 포인트 향상되었습니다. 더 적은 파라미터를 가진 T5-mini 모델을 사용할 때, NQ320K 데이터셋에서 7.06 포인트, MARCO 데이터셋에서는 6.45 포인트의 향상을 보였습니다. 해당 코드는 https://github.com/kduxin/Bottleneck-Minimal-Indexing에서 제공됩니다.



### Unveiling Hallucination in Text, Image, Video, and Audio Foundation Models: A Comprehensive Survey (https://arxiv.org/abs/2405.09589)
- **What's New**: 최근 아카이브 논문은 언어, 이미지, 오디오, 비디오 등 다양한 도메인의 파운데이션 모델(FMs: foundation models)에서 발생하는 헬루시네이션(hallucination)을 식별하고 완화하려는 최근 발전 사항을 종합적으로 다룹니다. 헬루시네이션은 컨텍스트나 사실적 근거가 부족한 내용을 생성하는 현상을 의미하며, 이는 특히 현실적이고 신뢰도가 중요한 고위험 애플리케이션에서 큰 장애물이 되고 있습니다. 이 논문은 이러한 문제를 해결하기 위한 정의, 분류, 탐지 전략을 명확히 설정하고 있습니다.

- **Technical Details**: 파운데이션 모델의 헬루시네이션 현상은 다양한 형태로 나타납니다. 컨텍스트의 일관성이 없는 'Contextual disconnection', 입력의 의미가 왜곡되는 'Semantic distortion', 입력 데이터에 없는 요소를 생성하는 'Content hallucination', 그리고 잘못된 정보를 생성하는 'Factual inaccuracy'가 그 예입니다. 이러한 헬루시네이션의 원인은 훈련 데이터의 편향, 최신 정보의 제한, 그리고 모델 자체가 컨텍스트를 이해하고 생성하는 능력의 한계에서 비롯될 수 있습니다.

- **Performance Highlights**: 논문에서는 언어, 이미지, 비디오, 오디오 도메인에서 헬루시네이션을 탐지하고 완화하기 위한 여러 방법들을 집중적으로 다룹니다. 도메인별로 제안된 탐지 및 완화 전략은 모델의 훈련 데이터를 다양화하고, 도메인 특화 데이터를 활용하여 모델을 미세 조정하고, 개선된 평가 지표를 활용해 헬루시네이션 발생을 줄이는 데 효과적임을 보여줍니다. 이 논문은 헬루시네이션 문제를 포괄적이고 다중 모드로 이해하기 위한 중요한 자원을 제공하며, 더 견고한 AI 솔루션 개발을 돕기 위해 작성되었습니다.




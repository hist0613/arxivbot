New uploads on arXiv(cs.CL)

### Enhance Reasoning by Learning from Mistakes: Peer-Review Knowledge Distillation from Multiple Large Language Models (https://arxiv.org/abs/2410.03663)
Comments:
          14 pages, 5 figures

- **What's New**: 이번 논문에서는 Mistake-Aware Peer-Review Distillation (MAPD) 접근 방식을 소개하여, 학생 모델이 교사 LLM(대형 언어 모델)으로부터 오류에 대한 피드백을 받아 학습하도록 합니다. 이 방법은 기존의 고유한 레퍼런스를 사용하는 것과는 달리, 학생의 실수에 대한 설명과 반영을 통해 맞춤형 학습 데이터를 제공합니다.

- **Technical Details**: MAPD 접근 방식은 1) 학생의 실수를 파악하고 설명하도록 교사에게 요청하여 맞춤형 교육 데이터를 생성합니다. 2) 여러 LLM 간의 시뮬레이션된 동료 검토 프로세스를 설계하여, 기준치 이상으로 생성된 합리적 추론만을 선택하여 교육 데이터의 품질을 향상시킵니다.

- **Performance Highlights**: 수학, 상식, 논리적 추론 과제에 대한 종합 실험 결과, MAPD 방법이 기존보다 향상된 성능을 나타내며, 학생 모델의 전반적인 추론 능력을 개선하는 것으로 나타났습니다.



### RAFT: Realistic Attacks to Fool Text Detectors (https://arxiv.org/abs/2410.03658)
Comments:
          Accepted by EMNLP 2024

- **What's New**: 이 논문에서는 RAFT라는 새로운 공격 방법을 제안하여 기존의 LLM 감지기를 효과적으로 무력화할 수 있음을 입증합니다. 기존의 공격 방식과는 달리, RAFT는 원본 텍스트의 품질을 유지하면서 단어 수준에서 LLM 임베딩의 전달 가능성을 활용합니다.

- **Technical Details**: RAFT는 보조 LLM 임베딩을 사용하여 대체할 후보 단어를 선택하고, 블랙박스 LLM을 이용해 가장 효과적으로 탐지를 회피할 수 있는 대체 후보를 선정합니다. 이 방법은 텍스트의 품질을 유지하면서 탐지 성능을 99%까지 감소시키는 데 성공했습니다.

- **Performance Highlights**: 실험 결과 RAFT는 여러 도메인에서 감지기를 최대 99%까지 무력화할 수 있었고, 이로 인해 현재의 LLM 감지기가 적대적 공격에 강하지 않다는 것을 보여주었습니다. 또한 이 방법으로 생성된 예제들은 적대적 훈련을 통해 감지기의 강인함을 향상시키는 데 사용될 수 있습니다.



### Aligning LLMs with Individual Preferences via Interaction (https://arxiv.org/abs/2410.03642)
Comments:
          The code and dataset are made public at this https URL

- **What's New**: 이 논문은 대화형 대형 언어 모델(LLM)이 사용자 개인의 선호도를 반영하여 상호작용하는 능력을 훈련하고, 그들의 행동을 동적으로 일치시키도록 하는 새로운 접근법을 제안합니다. 이는 이전에 단순히 일반적인 원칙(예: 도움을 주고, 해를 끼치지 않고, 정직하게 대화하는 것)에 초점을 맞춘 연구와 차별화됩니다.

- **Technical Details**: 연구진은 사용자 페르소나(user persona)의 다양성을 반영하기 위해 3,310개의 고유한 페르소나를 생성하였으며, 이 페르소나 데이터를 바탕으로 3K 이상의 다중 대화 회차가 포함된 데이터셋을 구축합니다. 대화의 동적 변화를 통해 LLM이 사용자 선호를 추론하고 이에 맞게 응답을 조정할 수 있도록 하는 메타 기술을 발전시킵니다.

- **Performance Highlights**: 알려진 대형 언어 모델들은 개인화된 선호에 맞춰 동적으로 적응하는 데 어려움을 겪지만, 연구에서 제안한 방법은 이러한 능력을 평균 32.0% 향상시키며 LLM이 개인화된 경험을 제공하는 데 주요한 진전을 이루었음을 보여줍니다. ALOE라는 벤치마크를 통해 평가를 실시하였고, 이는 다양한 사용자 선호에 맞춘 대화의 일치 정도를 측정합니다.



### Efficiently Identifying Watermarked Segments in Mixed-Source Texts (https://arxiv.org/abs/2410.03600)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)에서 합성 텍스트를 탐지하기 위한 새로운 부분 워터마크 감지 방법을 제안합니다. 기존의 워터마크 감지 기술은 전체 문서를 워터마크가 있는지 여부로 분류하는 데 초점을 맞추고 있지만, 긴 혼합 소스 문서 내에서 개별 워터마크 세그먼트를 식별하는 농기 없이 다양한 어플리케이션에서의 수요가 증가하고 있습니다.

- **Technical Details**: 저자들은 두 가지 새로운 방법, 즉 기하학적 덮개 감지기(Geometric Cover Detector, GCD)와 적응형 온라인 로컬라이저(Adaptive Online Locator, AOL)를 개발하였습니다. GCD는 긴 텍스트에서 워터마크 세그먼트 여부를 판별하며, O(n log n)의 시간 복잡도로 효율적인 분류를 제공합니다. AOL은 온라인 상의 잡음 제거 작업으로 문제를 재정의하고, Alligator 알고리즘을 활용하여 높은 정확도로 세그먼트 위치를 특정합니다.

- **Performance Highlights**: 세 가지 인기 있는 워터마킹 기술(KGW-Watermark, Unigram-Watermark, Gumbel-Watermark)을 평가한 결과, 제안된 방법은 기준 방법들에 비해 높은 정확도를 기록하였으며, 클래스 분류 작업에서 더 높은 진짜 양성 비율을 달성했습니다. 또한 로컬라이제이션 작업에서는 평균 IoU(Intersection over Union) 점수가 0.55 이상으로, 기준 방법을 크게 초과했습니다.



### Explicit, Implicit, and Scattered: Revisiting Event Extraction to Capture Complex Arguments (https://arxiv.org/abs/2410.03594)
Comments:
          Accepted in EMNLP-2024 (Main). 21 pages, 8 figures, and 11 tables

- **What's New**: 이번 연구에서는 기존 이벤트 추출(Event Extraction, EE) 접근 방식을 다시 살펴보며, 이벤트 아규먼트 유형을 두 가지로 확장했습니다. 첫째, 임플리시트 아규먼트(Implicit Arguments)는 텍스트에 명시적으로 언급되지 않지만 맥락을 통해 유추할 수 있는 아규먼트입니다. 둘째, 스캐터드 아규먼트(Scattered Arguments)는 텍스트 전반에 흩어져 있는 정보로 구성된 아규먼트입니다. 이러한 두 가지 아규먼트 유형은 사건 모델링을 위한 전체적인 정보를 이끌어내는 데 중요합니다.

- **Technical Details**: 이 연구는 DiscourseEE라는 새로운 데이터셋을 개발하여, 7,464개의 아규먼트 주석을 포함하고 있습니다. 이 데이터셋은 온라인 건강 담론에서 수집되었으며, 51.2%의 아규먼트는 임플리시트이고 17.4%는 스캐터드입니다. EE 주석을 텍스트 생성 문제로 재구성하여 이질적 아규먼트 추출을 용이하게 했습니다.

- **Performance Highlights**: Generative Event Argument Extraction(EAE) 접근 방식을 통해 복잡한 아규먼트를 추출할 수 있는 가능성을 보여주며, 이를 통해 다양한 최신 EE 모델을 평가하였습니다. 또, 평가 결과 기존 모델들이 DiscourseEE에서 한계를 가지는 것을 관찰하였으며, 이러한 한계는 향후 EE 연구에 대한 동기를 제공합니다.



### Table Question Answering for Low-resourced Indic Languages (https://arxiv.org/abs/2410.03576)
Comments:
          Accepted at EMNLP,2024

- **What's New**: 논문은 낮은 자원을 가진 언어에서의 TableQA 문제를 집중적으로 다루고 있으며, 자동화된 대규모 데이터 생성 방법을 제안합니다. 이 방법은 베이너리(Bengali)와 힌디어(Hindi) 언어를 대상으로 하여 기존에 존재하지 않던 tableQA 데이터셋과 모델을 제공합니다.

- **Technical Details**: 이 연구에서 제안하는 방법은 자동으로 대량의 tableQA 데이터를 생성하고 품질을 평가할 수 있는 파이프라인을 설계하는 것입니다. 데이터 생성 과정은 두 가지 주요 언어인 베이너리와 힌디어에서 적용되었습니다. 이 작업은 테이블 내의 사실 기반 질문에 대해 수치적 이해(numeracy understanding) 및 테이블 추론(table reasoning) 능력이 필요합니다.

- **Performance Highlights**: 제안된 데이터셋과 모델로 훈련된 TableQA 시스템은 최신 LLMs(대형 언어 모델)보다 성능이 뛰어납니다. 또한, 이 연구는 다양한 수학적 추론 능력 및 제로샷 크로스링구얼 전이(zero-shot cross-lingual transfer)에 대한 연구도 포함하고 있습니다.



### Towards Linguistically-Aware and Language-Independent Tokenization for Large Language Models (LLMs) (https://arxiv.org/abs/2410.03568)
- **What's New**: 이 논문은 최신 대형 언어 모델(LLMs)의 토큰화(tokenization) 기술에 대한 종합적인 연구를 제시하며, 특히 자원이 부족한 언어에 대한 서비스의 비용과 가용성에 미치는 영향을 분석합니다.

- **Technical Details**: 이번 연구는 GPT-4(cl100k_base 임베딩 사용), GPT-3(p50k_base 임베딩 사용), DaVinci(r50k_base 임베딩 사용) 및 널리 사용되는 BERT base tokenizer와 같은 여러 LLM을 포함합니다. 각 모델 간의 토큰화 가변성을 평가하고 하위 단어 토큰화(subword tokenization)에서의 언어 표현의 문제점을 조사합니다.

- **Performance Highlights**: 이 연구는 특히 전자 건강 기록(EHR) 시스템의 맥락에서 토큰화 선택의 실제적인 영향을 강조하는 사례 연구를 포함하고 있습니다. AI 서비스 발전에 있어 포괄성(inclusivity)을 강조하며 전통적으로 AI 애플리케이션에서 부족한 언어를 지원하기 위한 일반화 가능한 국제화(I18N) 관행을 촉진하고자 합니다.



### BodyShapeGPT: SMPL Body Shape Manipulation with LLMs (https://arxiv.org/abs/2410.03556)
Comments:
          Accepted to ECCV 2024 Workshop on Foundation Models for 3D Humans. Code repository: this https URL

- **What's New**: 이 연구는 사전 훈련된 Large Language Models (LLMs)을 활용하여 사람의 물리적 특성을 이해하고, 이를 기반으로 SMPL-X 모델을 사용해 정확한 아바타를 생성하는 방법을 제안합니다. 특히 LLMs가 자연어를 통해 3D 인간 형태를 조작할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 본 연구는 LLMs의 미세 조정을 통해 SMPL-X 형태 매개변수를 정확하게 인식하고 생성하는 데 초점을 맞춥니다. 새로운 데이터셋을 생성하여 아바타의 물리적 속성과 관련된 언어적 설명을 포괄적으로 라벨링 했습니다. 모델 학습에 Low-Rank Adaptation (LoRA)와 quantization 기법을 적용하여 NVidia RTX-4090에서 효율적으로 훈련할 수 있도록 최적화했습니다.

- **Performance Highlights**: 본 연구의 결과는 LLMs의 조정된 아키텍처가 다양한 아바타 형상을 성공적으로 생성할 수 있으며, 이는 스토리텔링 및 가상 캐릭터 생성 분야에서의 활용 가능성을 보여줍니다. 최종 아바타는 기대되는 범위 내에서 생성되며, 이는 인간-기계 상호작용을 향상시키는 데 기여할 것입니다.



### Structure-Enhanced Protein Instruction Tuning: Towards General-Purpose Protein Understanding (https://arxiv.org/abs/2410.03553)
- **What's New**: 본 논문에서는 프로틴 언어 모델(protein language models, pLMs)과 대형 언어 모델(large language models, LLMs)의 결합을 통해 일반적인 프로틴 이해를 위한 구조 강화 프로틴 명령 조정(Structure-Enhanced Protein Instruction Tuning, SEPIT) 프레임워크를 소개합니다.

- **Technical Details**: SEPIT 프레임워크는 pLMs에 구조 인식 모듈(structure-aware module)을 통합하여, 구조적 지식을 활용하여 프로틴에 대한 이해도를 높입니다. 또한 두 단계의 명령 조정 파이프라인을 통해 기본 이해를 구축한 뒤 전문가 혼합(mixture of experts, MoEs)을 통해 더욱 복잡하고 다양한 특성과 기능을 학습합니다.

- **Performance Highlights**: SEPIT는 개방형 생성(open-ended generation) 및 폐쇄형 답변(closed-set answer) 작업에서 기존의 LLM 대비 우수한 성과를 나타냈으며, 현재까지 가장 포괄적인 프로틴 명령 데이터셋을 구성하여 일반적인 프로틴 이해 모델의 훈련 및 평가에 기여하고 있습니다.



### Enhancing Data Quality through Simple De-duplication: Navigating Responsible Computational Social Science Research (https://arxiv.org/abs/2410.03545)
Comments:
          Accepted at EMNLP 2024 Main

- **What's New**: 이 연구는 자연어 처리(NLP)와 계산 사회 과학(CSS)에서 사용되는 20개의 소셜 미디어 데이터셋을 깊이 있게 분석하여 데이터 품질 문제를 다룬 것을 특징으로 합니다.

- **Technical Details**: 연구에서는 데이터 중복으로 인해 모델의 신뢰성이 손상될 수 있는 문제를 논의하며, 소셜 미디어 콘텐츠의 빠른 주제 발전과 사회적 봇의 역할을 고려합니다. 다양한 CSS 작업(예: 공격적인 언어 감지, 허위 정보 탐지)에 대해 데이터 중복의 영향을 조사하였으며, 데이터 전처리 프로토콜을 제안합니다.

- **Performance Highlights**: 연구 결과, 대부분의 소셜 미디어 데이터셋이 중복 데이터(duplicate data)를 포함하고 있어 모델 성능을 과대 평가할 수 있다는 점이 드러났습니다. 이는 모델 예측의 신뢰성을 저하시킬 수 있습니다. 데이터셋 개발 및 활용에 대한 새로운 프로토콜이 제안되어, 보다 책임감 있고 효과적인 연구 방법론을 권장합니다.



### Re-examining Sexism and Misogyny Classification with Annotator Attitudes (https://arxiv.org/abs/2410.03543)
Comments:
          Accepted by EMNLP 2024

- **What's New**: 이번 연구에서는 성차별 폭력(Gender-Based Violence, GBV) 문제를 다루며, 주관적 의견을 반영하기 위한 데이터 레이블링과 자동 분류의 두 가지 주요 단계를 재조명합니다. 이를 통해 다양한 관점의 대표성을 확보하고, GBV 관련 데이터의 품질을 향상시키고자 하였습니다.

- **Technical Details**: 연구진은 두 가지 데이터셋(Explainable Detection of Sexism, EDOS와 Detection of Online Misogyny, DOM)을 재주석하고, 주석자들의 인구통계학적 정보와 태도를 수집하여 모델 훈련에 활용하였습니다. 대규모 언어 모델(Large Language Models)과 여러 프롬프트 전략을 통해 분류 실험을 수행하였고, 주석자의 태도가 분류기 성능에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 주석자의 태도를 입력으로 포함시킬 경우, 꽤 높은 성능 향상을 이루며, 주석자 설명을 잘 구조화하였을 때 가장 좋은 성과를 보였습니다. 실험 결과, 비례 투표 및 개별 주석 작업에서 ICL(Instruction-based Contextual Learning) 방식이 기존의 기준보다 각각 17%와 26% 향상된 결과를 보였으며, LLM의 미세 조정 시 31% 성능 향상이 나타났습니다.



### MARE: Multi-Aspect Rationale Extractor on Unsupervised Rationale Extraction (https://arxiv.org/abs/2410.03531)
Comments:
          Accepted in EMNLP2024(Main) conference

- **What's New**: 이 논문에서는 Multi-Aspect Rationale Extractor (MARE)를 제안하여 여러 측면을 동시에 예측하고 설명합니다. MARE는 기존의 Uni-Aspect 모델의 한계를 극복하고 다양한 측면 간의 내부 상관관계를 활용하여 성능을 향상시키고자 합니다.

- **Technical Details**: MARE는 하드 삭제 기반의 Multi-Aspect Multi-Head Attention (MAMHA) 메커니즘을 통해 입력 텍스트에 여러 특수 토큰을 추가하고, 이를 통해 여러 텍스트 조각을 동시에 인코딩합니다. 또한, 다중 작업 훈련(Multi-task training)을 적용하여 훈련 비용을 절감합니다.

- **Performance Highlights**: MARE는 BeerAdvocate와 Hotel Review 데이터셋의 두 개의 비지도 라지오 분석 벤치마크에서 기존의 최신 방법보다 월등한 성능을 보였으며, 토큰 수준의 F1 스코어에서 5.4% 향상을 기록했습니다.



### Steering Large Language Models between Code Execution and Textual Reasoning (https://arxiv.org/abs/2410.03524)
Comments:
          32 pages, 12 figures, 12 tables

- **What's New**: 본 논문에서는 코드 작성과 실행을 통해 복잡한 작업을 해결하는 능력을 지닌 LLM(대형 언어 모델)에 대해 논의합니다. 특히 기존의 텍스트 추론 방식의 한계를 극복하기 위해 코드 중심의 접근을 제안합니다.

- **Technical Details**: 실험은 7개의 인기 있는 방법을 이용하여 14개의 작업과 6종류의 LLM(새로운 O1-preview 포함)에 대해 단일 및 다중 턴 설정에서 수행되었습니다. 코드와 텍스트 추론의 사용을 모델의 복잡도 및 크기에 따라 분석하였으며, 예상치 못한 반비례 확장 법칙도 발견하였습니다.

- **Performance Highlights**: LLM이 작성한 코드의 성능이 항상 텍스트 추론 방식보다 우수하지 않다는 점을 발견하였고, 코드/텍스트 생성 방향을 보다 효과적으로 조정하기 위한 세 가지 방법을 제안하여 개선을 도모하였습니다.



### LCMDC: Large-scale Chinese Medical Dialogue Corpora for Automatic Triage and Medical Consultation (https://arxiv.org/abs/2410.03521)
- **What's New**: COVID-19 팬데믹으로 인해 전통적인 헬스케어 시스템의 한계가 드러나면서, 온라인 의료 서비스와 특히 의료 triage 및 상담의 발전이 가속화되었습니다. 이를 해결하기 위해 대규모 의료 대화 데이터셋 (Large-scale Chinese Medical Dialogue Corpora, LCMDC) 이 구축되었으며, 이는 Coarse-grained Triage, Fine-grained Diagnosis, Medical Consultation의 세 가지 데이터셋으로 구성되어 있습니다.

- **Technical Details**: LCMDC는 443,630개의 샘플로 구성된 Coarse-grained Triage 데이터셋, 199,600개의 샘플로 구성된 Fine-grained Diagnosis 데이터셋, 472,418개의 항목으로 구성된 Medical Consultation 데이터셋을 포함하고 있습니다. 또한 BERT 기반의 감독 학습과 prompt learning을 결합한 새로운 triage 시스템을 제안하고, 강화 학습을 이용한 GPT 기반의 의료 상담 모델을 개발했습니다. PLMs(Pre-trained Language Models)를 의료 지식으로 사전 학습하여 도메인 지식 습득을 강화했습니다.

- **Performance Highlights**: LCMDC에서의 실험 결과, 제안한 시스템이 기존 모델보다 탁월한 성능을 보였으며, 특히 희귀 질환에 대한 예측 정확도가 5% 향상되었습니다. 학습된 모델은 사용자 질문에 대한 정확한 의학적 답변을 제공하는 데 효과적이라는 것을 입증했습니다.



### CliMedBench: A Large-Scale Chinese Benchmark for Evaluating Medical Large Language Models in Clinical Scenarios (https://arxiv.org/abs/2410.03502)
Comments:
          accepted by ENMLP-2024

- **What's New**: CliMedBench는 14개의 전문 가이드 핵심 임상 시나리오를 포함한 포괄적 벤치마크로, LLM의 의료 능력을 평가하기 위해 설계되었습니다. 이는 실제 의료 보고서에서 유래한 33,735개의 질문으로 구성되어 있습니다.

- **Technical Details**: CliMedBench는 7개의 축 차원에서 LLM의 의료 능력을 평가하며, 전문 의료진의 지식과 실제 사례를 통합하여 만들어졌습니다. 이 벤치마크는 '의료 언어 능력'과 '인지 능력'을 측정하는 데 사용됩니다.

- **Performance Highlights**: 중국 의료 LLM은 이 벤치마크에서 저조한 성과를 보였으며, 특히 의료적 추론 및 사실적 일관성이 중요한 영역에서 그 필요성을 강조합니다. 그러나 일반 도메인 LLM들이 의료 진료에서 큰 잠재력을 보여주었고, 이는 많은 의료 LLM의 제한된 입력 용량이 실질적인 사용을 방해함을 나타냅니다.



### Towards Reproducible LLM Evaluation: Quantifying Uncertainty in LLM Benchmark Scores (https://arxiv.org/abs/2410.03492)
Comments:
          4 pages, 1 figure

- **What's New**: 이 연구는 대형 언어 모델(LLMs)의 불확실성을 정량화하기 위한 실험 반복의 영향을 탐구하며, 이를 통해 벤치마크 점수의 불확실성을 경제적으로 측정하는 방법을 제시합니다.

- **Technical Details**: 특정 방향에 대한 사고 능력을 평가하기 위해 설계된 벤치마크를 활용하여, LLM의 비결정적인 응답 문제를 다루고, 실험 반복 횟수에 따른 평균 점수 및 예측 구간을 분석합니다. 이 연구는 다섯 개의 LLM 모델을 사용하여 100개의 간단한 질문과 5760개의 템플릿 질문으로 구성된 benchmarks에서 실험을 수행하였습니다.

- **Performance Highlights**: 연구 결과, 반복 실험 수가 벤치마크 점수의 평균과 예측 구간에 미치는 영향을 명확히 보여줍니다. 이는 LLM의 평가 및 샘플링 방식을 개선하는 데 기여할 것으로 기대됩니다.



### Is Safer Better? The Impact of Guardrails on the Argumentative Strength of LLMs in Hate Speech Countering (https://arxiv.org/abs/2410.03466)
Comments:
          To appear in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (long paper)

- **What's New**: 이번 연구는 자동 생성된 반증 발언(counterspeech)에서의 설득력 있는 대응을 개선하기 위해 두 가지 주요 측면에 주목하고 있습니다. 첫째, LLMs의 안전 장치가 생성 품질에 미치는 영향을 조사합니다. 둘째, 특정 혐오 발언의 요소를 공격함으로써 더 효과적인 반증 전략을 평가합니다.

- **Technical Details**: 이 연구는 White Supremacy Forum 데이터셋을 사용하여 혐오 발언의 주장을 구조적으로 분석하고, 두 가지 연구 질문을 설정합니다. 연구 질문 1(RQ1)은 안전 장치가 생성된 반증의 품질에 영향을 미치는지를 검토하며, 연구 질문 2(RQ2)은 특정 혐오 발언의 요소에 집중하는 것이 전체 메시지를 일반적으로 공격하는 것보다 더 나은지를 비교합니다. 총 네 가지 공격 전략이 사용됩니다: 전체 혐오 발언, 암시된 진술, 혐오의 전제 및 결론, 그리고 가장 약한 전제를 겨냥합니다.

- **Performance Highlights**: 연구 결과 안전 장치가 반증 발언의 지원 주장의 양과 논리적 정확성에 부정적인 영향을 미치는 것으로 나타났습니다. 또한, 혐오 발언의 암시적 진술이나 혐오 요소를 겨냥하는 것이 전체 메시지를 공격하는 것보다 더 높은 품질의 반증 생성을 이끌어낸다고 밝혀졌습니다.



### Auto-GDA: Automatic Domain Adaptation for Efficient Grounding Verification in Retrieval Augmented Generation (https://arxiv.org/abs/2410.03461)
- **What's New**: 이 논문에서는 Retrieval Augmented Generation (RAG)과 자연어 추론(NLI) 모델의 결합을 통해 LLM의 허구성 문제를 해결하기 위한 새로운 접근 방식인 자동 생성 도메인 적응(Automatic Generative Domain Adaptation, Auto-GDA)을 제안합니다.

- **Technical Details**: Auto-GDA는 합성 데이터 생성을 활용하여 비지도 도메인 적응을 가능하게 합니다. 이 방법은 낮은 효율의 teacher 모델에서 약한 레이블을 사용하여 생성된 샘플의 품질을 지속적으로 개선하는 반복적인 프로세스를 차별화합니다. 또한, 기존의 NLI 모델에 비해 RAG 입력에 적합하게 모델을 조정하는 것을 목표로 합니다.

- **Performance Highlights**: Auto-GDA를 사용하여 합성 데이터로 미세 조정된 모델은 종종 teacher 모델을 초과하는 성능을 보이며, 인간 레이블 데이터로 미세 조정된 참조 모델과 유사한 성능을 달성합니다. 또한, LLM의 10배 적은 계산 비용으로도 유사한 성능을 나타냅니다.



### Multi-Dialect Vietnamese: Task, Dataset, Baseline Models and Challenges (https://arxiv.org/abs/2410.03458)
Comments:
          Main EMNLP 2024

- **What's New**: 본 연구에서는 베트남의 63개 지방 방언을 포괄적으로 담은 최초의 음성 데이터셋인 Vietnamese Multi-Dialect (ViMD) 데이터셋을 소개합니다. 이 데이터셋은 102.56시간의 오디오와 약 19,000개 발화를 포함하며, 120만 개 이상의 단어가 포함된 전사본을 제공합니다.

- **Technical Details**: ViMD 데이터셋은 63개 지방 방언을 포함하며, 각 지방의 고유한 발음 변화를 잘 반영하고 있습니다. 음성 데이터는 공공 소스에서 수집되었고, 전사 데이터는 반자동 라벨링 및 수작업 검증 과정을 통해 품질이 보장되었습니다. 또한 각 기록은 화자 식별 코드 및 성별과 같은 추가 속성을 포함하고 있어 다양한 음성 관련 작업을 지원합니다.

- **Performance Highlights**: 각 방언 식별 및 음성 인식이라는 두 가지 하위 작업에 대해 최신 사전 훈련된 모델을 미세 조정하여 실험을 수행했습니다. 실험 결과는 지리적 요소가 방언에 미치는 영향과 다중 방언 음성 데이터 처리 시의 현재 접근 방식의 한계를 제시합니다.



### CoCoLoFa: A Dataset of News Comments with Common Logical Fallacies Written by LLM-Assisted Crowds (https://arxiv.org/abs/2410.03457)
Comments:
          In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024)

- **What's New**: CoCoLoFa 데이터셋은 648개의 뉴스 기사에 대해 7,706개의 댓글을 포함하고 있으며, 각 댓글은 논리적 오류의 존재와 유형이 레이블링 되어 있습니다. 이는 기존의 다른 데이터셋보다 훨씬 더 크고 다양합니다.

- **Technical Details**: 이 연구에서는 143명의 크라우드 작업자를 모집하여 구체적인 오류 유형(예: slippery slope)에 대한 댓글을 작성하도록 하였습니다. 작성 지원을 위해 LLM(대형 언어 모델) 기반의 보조 장치를 인터페이스에 통합하여 작업자들이 댓글을 세밀하게 다듬을 수 있도록 도왔습니다. 데이터셋은 70% 훈련 세트, 20% 개발 세트, 10% 테스트 세트로 나뉘어 균형 잡힌 21개의 주제를 포함하고 있습니다.

- **Performance Highlights**: BERT 기반 모델은 CoCoLoFa를 활용하여 테스트 세트에서 논리적 오류 탐지(F1=0.86)와 분류(F1=0.87) 성능에서 최고 기록을 세웠습니다. 이는 현재 최첨단 LLM들을 초월하는 성과입니다.



### How Language Models Prioritize Contextual Grammatical Cues? (https://arxiv.org/abs/2410.03447)
Comments:
          Accepted to BlackboxNLP 2024

- **What's New**: 본 논문에서는 Transformer 기반 언어 모델이 여러 개의 성별 단서가 존재할 때 성별 일치를 처리하는 방법을 조사합니다.

- **Technical Details**: BERT(인코더 기반 모델)와 GPT-2(디코더 기반 모델)를 분석하며, 두 가지 보완적인 접근 방식을 사용합니다: context mixing analysis(맥락 혼합 분석)와 activation patching(활성화 패칭) 변형. 이를 통해 모델 내 정보 흐름을 추적하고 단서가 모델 예측에 미치는 영향을 분석합니다.

- **Performance Highlights**: BERT는 맥락 내 첫 번째 단서를 우선시하여 타겟 단어 표현 및 모델 예측을 형성하는 반면, GPT-2는 마지막 단서를 더 많이 의존하여 예측을 수행하는 경향을 보입니다. 이러한 연구 결과는 인코더 기반 모델과 디코더 기반 모델 간의 중요한 차이를 드러냅니다.



### Exploring the Benefit of Activation Sparsity in Pre-training (https://arxiv.org/abs/2410.03440)
Comments:
          ICML 2024

- **What's New**: 이 논문에서는 Pre-trained Transformers의 sparse activation에 대한 연구를 진행하며, 이를 활용한 새로운 학습 방식인 Switchable Sparse-Dense Learning (SSD)을 제안한다. SSD는 sparse training과 conventional dense training을 동적으로 전환하여 효율성을 높인다.

- **Technical Details**: 이 연구는 GPT, BERT 및 T5 모델을 사용하여 sparse activation의 변화 양상을 조사하였다. SSD는 초기에는 dense training을 수행하고, activation sparsity가 높아지면 sparse training으로 전환하여 Sparse-activated Mixture-of-Experts (SMoE) 모델로 대체한다. 이 딥러닝 프레임워크는 다단계 동적 학습을 통해 모델의 성능을 극대화한다.

- **Performance Highlights**: SSD는 동일한 모델 크기에서 기존 dense training 대비 유사한 성능을 유지하면서, 전이 훈련 비용을 감소시킨다. 예를 들어 FLOPs에서 최대 1.44배의 속도 향상과 함께, inference 속도는 최대 2배 빨라졌다. SSD에서 학습된 모델은 추가적인 훈련 없이도 SMoE 모델로 사용 가능하다.



### ToolGen: Unified Tool Retrieval and Calling via Generation (https://arxiv.org/abs/2410.03439)
- **What's New**: ToolGen은 기존의 도구 검색과 실행 방식을 통합하여 언어 모델의 매개변수 안에 도구 지식을 직접 통합하는 혁신적인 프레임워크입니다. 각 도구는 고유한 템플릿으로 표현되어 LLM의 다음 토큰 예측에 도구 호출 및 인자를 포함시킬 수 있습니다.

- **Technical Details**: ToolGen은 관리되는 세 단계의 훈련 과정을 통해 LLM이 대화 맥락 내에서 도구를 검색하고 호출할 수 있도록 합니다. 도구 암기 단계에서는 각 가상 도구 토큰을 해당 문서와 연관시키고, 검색 훈련 단계에서는 사용자 쿼리에 따라 적절한 도구 토큰을 생성하는 법을 학습하며, 최종적으로 에이전트 훈련 단계에서는 독립적인 에이전트로 기능할 수 있도록 훈련됩니다.

- **Performance Highlights**: ToolGen은 47,000개의 실제 도구 데이터셋을 사용하여 기존의 도구 검색 방법과 유사한 성능을 보이면서도 훨씬 더 낮은 비용과 향상된 효율성을 보여줍니다. 이는 LLM이 더욱 다재다능하고 효율적인 작업을 수행할 수 있게 함으로써 새로운 AI 에이전트 시대의 기초를 마련하게 됩니다.



### A General Framework for Producing Interpretable Semantic Text Embeddings (https://arxiv.org/abs/2410.03435)
Comments:
          19 pages, 5 figures, and 9 tables

- **What's New**: CQG-MBQA(Contrastive Question Generation - Multi-task Binary Question Answering)는 다양한 작업에 대한 해석 가능한 의미 텍스트 임베딩을 생성하기 위한 일반적인 프레임워크로, 고비용의 전문가 지식이나 정밀한 프롬프트 설계 없이도 높은 차별성을 지닌 질문을 체계적으로 생성한다.

- **Technical Details**: 이 프레임워크는 CQG 방법론을 활용하여 저 인지 부하의 Yes/No 질문을 생성하고, MBQA 모델을 통해 이를 효율적으로 답변함으로써 해석 가능한 임베딩을 비용 효율적으로 생성한다. CQG는 LLM을 활용하여 텍스트 간의 의미적 뉘앙스를 포착하는 이진 질문을 강조하여 임베딩 공간의 차원을 형성한다.

- **Performance Highlights**: CQG-MBQA는 블랙박스 모델과 유사한 품질의 임베딩을 제공하며, 다양한 다운스트림 작업에서 기존의 해석 가능한 텍스트 임베딩 방법들을 초월하는 성능을 보여준다.



### How Hard is this Test Set? NLI Characterization by Exploiting Training Dynamics (https://arxiv.org/abs/2410.03429)
Comments:
          Accepted at EMNLP 2024 Main Conference

- **What's New**: 본 연구에서는 자연어 추론(NLI) 데이터셋의 한계를 해결하기 위해 자동으로 챌린징한 테스트 세트를 구축하는 방법을 제안합니다. 기존의 인위적인 예제를 수동으로 생성하는 대신, 우리는 훈련 동태를 활용하여 테스트 세트를 세 가지 난이도 수준으로 분류합니다.

- **Technical Details**: 이 방법은 분류된 예제의 훈련 동태를 활용하여 가장 높은 난이도를 가진 예제를 특성화합니다. 이로 인해 데이터셋의 노이즈를 줄이고, 모델의 성능을 보다 정확하게 평가할 수 있습니다. 실험 결과, 전체 데이터셋의 일부만 사용하여도 유사한 성능을 달성할 수 있음을 보였습니다. 우리의 방법은 모델에 구애받지 않으며, 다른 데이터셋에도 쉽게 적용 가능합니다.

- **Performance Highlights**: 우리가 제안한 방법은 SNLI와 MultiNLI 데이터셋에서 모델 성능과 여러 스푸리어스 상관관계 간의 통계적으로 유의미한 상관관계를 보여줍니다. 특히, 기존의 데이터 셋 특성화 방법보다 더 높은 성능을 기록하였으며, 샘플 수를 33% 또는 59%로 줄여도 비슷한 테스트 성능을 달성할 수 있었습니다.



### One2set + Large Language Model: Best Partners for Keyphrase Generation (https://arxiv.org/abs/2410.03421)
Comments:
          Accepted by EMNLP 2024 Main Conference

- **What's New**: 이 논문에서는 기존의 keyphrase generation (KPG) 방식을 개선하기 위해 generate-then-select 프레임워크를 도입하였습니다. 이 프레임워크는 KPG를 두 단계로 분해하여 각각 generator와 selector가 수행합니다.

- **Technical Details**: 저자들은 one2set 기반 모델을 generator로 사용하여 후보 키프레이즈를 생성하고, LLM(large language model)을 selector로 사용하여 이들 후보 중에서 최종 키프레이즈를 선택합니다. 특히, Optimal Transport 기반의 할당 전략을 설계하여 supervision 신호의 부적절한 할당 문제를 해결하고, 키프레이즈 선택을 시퀀스 레이블링 태스크로 모델링하여 중복 선택 문제를 완화했습니다.

- **Performance Highlights**: 여러 벤치마크 데이터셋을 통한 실험 결과, 이 프레임워크는 특히 부재 키프레이즈 예측에서 기존 최첨단 모델들을 유의미하게 초월하는 성능을 보였습니다.



### Surgical, Cheap, and Flexible: Mitigating False Refusal in Language Models via Single Vector Ablation (https://arxiv.org/abs/2410.03415)
- **What's New**: 이 논문에서는 언어 모델의 잘못된 거부(false refusal) 문제를 해결하기 위해 단일 벡터 절삭(single vector ablation)이라는 새로운 접근 방식을 제안합니다. 이 방법은 안전한 요청을 무시하지 않도록 모델의 동작을 미세 조정하는 데 효과적이며, 훈련 없이 적용할 수 있습니다.

- **Technical Details**: 제안된 접근 방식에서는 해로운 쿼리와 유사해 보이지만 실제로는 안전한 쿼리를 사용하여 잘못된 거부 벡터를 추출합니다. 그런 다음 이 벡터를 제거하여 언어 모델의 잘못된 거부율을 감소시키는 데 초점을 맞추고 있으며, 이는 일반적인 모델 능력에 부정적인 영향을 미치지 않습니다. 추가 계산이 필요하지 않아 비용 효율적입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법을 사용했을 때 모델의 안전성 조정이 가능해졌고, 잘못된 거부율이 감소했으며, 모델의 전반적인 성능에도 해를 끼치지 않았습니다. 이는 외부 쿼리에 대한 모델의 반응성을 개선하는 데 중요한 기여를 합니다.



### Team MTS @ AutoMin 2021: An Overview of Existing Summarization Approaches and Comparison to Unsupervised Summarization Techniques (https://arxiv.org/abs/2410.03412)
Comments:
          First Shared Task on Automatic Minuting at Interspeech 2021

- **What's New**: 본 논문에서는 Automatic Minutes Challenge에 참여하는 동안 MTS 팀이 진행한 연구 결과를 제시합니다. 특히, 본 문서는 클러스터링 기반의 비지도 학습 요약 기법을 제안하고, 실제 녹음을 처리할 수 있는 적응된 Automatic Speech Recognition 블록을 포함하는 파이프라인을 제공합니다.

- **Technical Details**: 본 연구는 음성 요약(Automatic Speech Summarization) 및 자동 회의 요약(Automatic Meeting Minutes) 문제를 다룹니다. 기존의 수집된 데이터셋을 사용하여 수치적 결과를 도출하며, Extractive(추출형) 및 Abstractive(추상형) 요약 기법을 분석합니다. 연구팀은 USE 벡터화 및 Affinity Propagation 클러스터링 기법을 이용한 비지도 학습 요약 기법을 사용합니다.

- **Performance Highlights**: 비지도 학습 기법은 Automatic Minuting 작업에서 사전 훈련된 요약 모델들을 초월하여 dev 세트에서 Rouge 1, Rouge 2, Rouge L 값이 각각 0.21, 0.02 및 0.2를 기록하였고, test 세트에서 유사한 성과를 보여주었습니다.



### Killing Two Flies with One Stone: An Attempt to Break LLMs Using English->Icelandic Idioms and Proper Names (https://arxiv.org/abs/2410.03394)
Comments:
          WMT24 MT Test Suites subtask. 8 pages, 5 tables

- **What's New**: 이번 논문은 아이슬란드어 번역에서 관용구(idiomatic expressions) 및 고유명사(proper names)에 중점을 두고 Árni Magnússon Institute 팀의 WMT24 테스트 서브태스크(subtask) 제출 내용을 다루고 있습니다. 현대 번역 모델에 중요한 도전인 이 두 가지 주제를 규명하려고 하였습니다.

- **Technical Details**: 우리는 두 가지 테스트 세트를 제작하였습니다: 첫 번째는 일반적인 영어 관용구의 번역 능력을 평가하고, 두 번째는 역설적인 번역이 필요한 지명 및 성별에 따라 형태가 동일한 아이슬란드 이름을 다룹니다. 테스트는 정확한 번역의 빈도와 문맥에 따른 변환 능력을 측정하기 위해 엄격한 기준을 적용했습니다.

- **Performance Highlights**: 보고된 점수는 상대적으로 낮았으며, 특히 관용구와 지명 번역에서 큰 개선 여지가 있음을 나타냅니다. 이는 LLM 기반의 기계 번역 시스템이 아직 해결해야 할 중요하고 심각한 제한을 가지고 있음을 시사합니다.



### Cogs in a Machine, Doing What They're Meant to Do -- The AMI Submission to the WMT24 General Translation Task (https://arxiv.org/abs/2410.03381)
Comments:
          WMT24 General Translation Task System Description Paper, 10 pages, 1 figure, 6 tables

- **What's New**: 이 논문은 WMT24 일반 번역 작업에 제출한 Árni Magnusson Institute 팀의 결과를 제시합니다. 우리는 영어에서 아이슬란드어로의 번역 방식을 다루며, 모델의 질을 높이기 위해 데이터 세트를 신중하게 선정하고 필터링하였습니다.

- **Technical Details**: 우리 시스템은 네 개의 번역 모델과 문법 교정 모델로 구성되어 있습니다. 우리는 ALMA-R 모델(7B 및 13B 파라미터)을 활용하여 병렬 데이터 수집 및 다양한 데이터에서 생성된 추가적인 합성 데이터를 사용했습니다. Transformer 아키텍처를 기반으로 하여 학습하였으며, 최종 출력은 reranking 모델을 사용하여 가장 좋은 후보를 선정합니다.

- **Performance Highlights**: 우리의 번역 시스템은 WMT21 테스트 세트에서 아이슬란드어로부터 영어로의 번역 품질이 매우 경쟁력이 있음을 보여주었습니다. 필터링된 데이터 셋을 이용해 훈련한 결과, 고품질의 2,056,704 쌍의 영어-아이슬란드어 문장 쌍을 확보하였고, 이는 원본 데이터의 약 9.71%에 해당합니다.



### Should Cross-Lingual AMR Parsing go Meta? An Empirical Assessment of Meta-Learning and Joint Learning AMR Parsing (https://arxiv.org/abs/2410.03357)
Comments:
          to appear in Findings of EMNLP 2024

- **What's New**: 이 연구는 cross-lingual AMR parsing 분야에서 메타 학습(Meta-Learning)을 적용한 최초의 실증 연구로, 한국어와 크로아티아어에 대한 새로운 평가 데이터셋을 개발하고 공개하였습니다.

- **Technical Details**: AMR (Abstract Meaning Representation) 그래프는 텍스트의 의미를 표현하는 유향 비순환 그래프입니다. 본 연구에서는 MAML (Model-Agnostic Meta-Learning) 메타 학습 방법을 활용하여 AMR 파싱 모델을 훈련하고 평가합니다. 여러 언어에 대한 k-shot 학습(k𝑘k) 방식이 적용되었습니다. 실험에 사용된 언어로는 크로아티아어, 페르시아어, 한국어, 중국어, 프랑스어가 있습니다.

- **Performance Highlights**: 메타 학습 모델은 특정 언어에 대해 0-shot 평가에서 약간의 성능 향상을 보였으나, $k$가 0보다 클 경우 성능 향상은 미미하거나 없는 것으로 나타났습니다. 한국어 및 크로아티아어에 대한 신규 평가 데이터셋을 개발하여 평가하였습니다.



### Generating Equivalent Representations of Code By A Self-Reflection Approach (https://arxiv.org/abs/2410.03351)
- **What's New**: 이 논문에서는 코드의 Equivalent Representations (ERs)을 자동으로 생성하기 위한 자가 반영(self-reflection) 접근 방식을 제안합니다. 이 방식은 두 개의 Large Language Models (LLMs)가 상호 작용하여 코드를 이해하고, 자연어 주석, 의사 코드(pseudocode) 및 플로우차트 등 다양한 형식의 ERs를 생성하는 것을 가능하게 합니다.

- **Technical Details**: 제안된 접근 방식은 입력된 코드에 대해 두 개의 LLM이 서로 작업하여 ER을 생성하는 과정으로 구성되어 있습니다. 개방(open) 및 제약(constrained) 환경에서 각각 ER을 생성하는 실험이 수행되었으며, ER 생성에는 자연어 주석, 의사 코드 및 플로우차트와 같은 특정 형식이 포함됩니다. 또한, 피드백을 자연어로 변환하여 ER 생성을 최적화하는 자가 반영 알고리즘이 설계되었습니다.

- **Performance Highlights**: 1. 개방 환경에서 LLMs는 다양하고 구조화된 ER을 생성하는데, 상당수의 결과가 LLMs의 코드 이해 방식을 드러냅니다. 2. 제약 환경에서는 명확한 주석과 의사 코드, 플로우차트를 생성하여 소프트웨어 엔지니어링 작업을 지원하며, LLM들이 생성하는 환각을 효과적으로 줄입니다. 3. 향후 연구 방향으로는 코드 생성용 중간 언어 도출, LLM 친화적인 요구 사항 설명 탐색 등이 논의되었습니다.



### Zero-Shot Fact Verification via Natural Logic and Large Language Models (https://arxiv.org/abs/2410.03341)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 이번 연구에서는 자연 논리(natural logic)를 활용한 제로샷(zero-shot) 사실 검증(fact verification) 방법인 Zero-NatVer를 제안하여, 훈련 데이터에 대한 의존도를 줄이고 설명 가능성을 높였습니다. 이 방법은 대형 언어 모델(large language models)의 일반화 능력을 활용합니다.

- **Technical Details**: Zero-NatVer는 사실 검증을 위해 주장(claim)과 증거(evidence) 간의 관계를 자연 논리를 통해 평가합니다. 이 과정에서 추론(inference)은 일련의 단계로 나뉘며, 각각의 단계에서 주장과 증거를 정렬하고, 자연 논리 연산자(natural-logic operators)를 부여하며, 최종적으로 유한 상태 기계(finite state automaton)를 사용하여 결론을 도출합니다. 이 시스템은 단일 언어뿐만 아니라 다국어 데이터셋에서도 효과적으로 적용될 수 있습니다.

- **Performance Highlights**: 제로샷 설정에서 Zero-NatVer는 영어 데이터셋 전반에 걸쳐 기존 자연 논리 모델 대비 평균 8.96 포인트의 정확도 향상을 보여주었으며, 자연 논리 데이터에 기반한 모델보다 다양한 실세계 데이터셋에서 더 나은 일반화 성능을 나타냈습니다.



### Context and System Fusion in Post-ASR Emotion Recognition with Large Language Models (https://arxiv.org/abs/2410.03312)
- **What's New**: 이 연구에서는 최신 대형 언어 모델(LLM)을 활용하여 ASR(Automatic Speech Recognition) 후의 화자 감정 예측을 위한 최적 사용법을 모색합니다. 특히 GenSEC이라는 과제를 통해 LLM의 프롬팅 방법을 탐구하며, ASR 전사 데이터의 순위 선정, 변동적인 대화 맥락 및 시스템 출력 융합 등의 기술을 사용했습니다.

- **Technical Details**: 이 연구는 ASR 출력의 순위를 정하고, 다양한 메트릭을 사용해 입력으로 활용하며, 대화 이력과 여러 ASR 시스템의 출력을 융합하여 성과를 최적화하는 방법론에 중점을 두고 있습니다. ASR 시스템의 출력을 기반으로 감정 예측을 위한 적절한 프롬프트 전략을 탐색하며, 여러 문자열 기준을 이용한 ASR 출력 순위 정립을 포함합니다.

- **Performance Highlights**: 최종 제출물은 SER(스피치 감정 인식) 정확도에서 제공된 기준보다 20% 증가한 75.1%의 결과를 기록하며, 재훈련 없이 일반화가 가능한 접근법으로 기대됩니다.



### Comparing zero-shot self-explanations with human rationales in multilingual text classification (https://arxiv.org/abs/2410.03296)
Comments:
          preprint

- **What's New**: 이 논문은 Instruction-tuned LLM들이 생성하는 self-explanations의 품질을 평가하여 신뢰성과 투명성을 높이는 방법을 탐구하고 있습니다.

- **Technical Details**: 연구에서는 sentiment classification과 forced labour detection 두 가지 텍스트 분류 작업을 수행하고, 자기 생성 설명과 인간 주석을 비교하여 그 신뢰성 및 일관성을 평가하고 있습니다. 또한, layer-wise relevance propagation (LRP)과 같은 post-hoc feature attribution 기법을 적용하여 결과를 비교합니다.

- **Performance Highlights**: 결과적으로 self-explanations가 인간 주석과 더 밀접하게 일치하며, LRP보다 감정 표현에 대한 신뢰성 있는 설명 품질을 유지하는 것으로 나타났습니다.



### Five Years of COVID-19 Discourse on Instagram: A Labeled Instagram Dataset of Over Half a Million Posts for Multilingual Sentiment Analysis (https://arxiv.org/abs/2410.03293)
- **What's New**: 이 논문은 COVID-19와 관련된 인스타그램 포스트의 데이터셋과 다국적 감성 분석의 결과를 제시하면서 2020년부터 2024년까지의 감정 경향을 탐구합니다.

- **Technical Details**: 500,153개의 인스타그램 포스트로 구성된 다국어 데이터셋을 구축하였으며, 이는 161개의 언어와 535,021개의 해시태그를 포함합니다. 이 데이터셋은 감정 분석을 위한 속성을 포함하고 있으며, 매글로벌 언어마다 양의(positive), 음의(negative), 중립(neutral)으로 분류된 결과를 갖추고 있습니다.

- **Performance Highlights**: 연도별 감성 분석 결과, 긍정 감정 비율이 38.35%에서 28.69%로 감소한 반면, 중립 감정 비율은 44.19%에서 58.34%로 증가했습니다. 언어별 분석에서는 영어 포스트의 긍정 감정 비율이 49.68%인 반면, 힌디어 포스트는 4.40%에 불과해 언어에 따른 감정 분포에 뚜렷한 차이가 나타났습니다.



### What do Large Language Models Need for Machine Translation Evaluation? (https://arxiv.org/abs/2410.03278)
- **What's New**: 대규모 언어 모델(LLMs)을 활용한 기계 번역(MT) 품질 평가에 필요한 번역 정보, 프롬프트 기법과 이에 관한 종합적인 분석을 제시한 연구입니다.

- **Technical Details**: 본 연구에서는 LLM이 MT 품질을 평가하는 데 필요로 하는 번역 정보를 분석하며, 고-중-저 자원 언어의 8888쌍에 대한 zero-shot, Chain of Thought (CoT), few-shot 프롬프트 방법을 탐구하였습니다.

- **Performance Highlights**: LLMs의 MT 품질 평가에서의 성능은 확실히 참고 번역(reference translations)에 의존하며, 더 큰 모델은 CoT 프롬프트에 더 많은 이점을 누리는 경향이 있지만, 전체적으로 LLMs는 fine-tuning된 멀티링구얼 PTLMs보다는 여전히 성능이 뒤쳐짐을 발견했습니다.



### A Multi-task Learning Framework for Evaluating Machine Translation of Emotion-loaded User-generated Conten (https://arxiv.org/abs/2410.03277)
- **What's New**: 이 논문에서는 사용자 생성 콘텐츠(UGC)의 기계 번역(MT) 품질 평가를 위해 감정 관련 데이터 세트를 활용하고, 문장 및 단어 수준 평가 점수로 확장하여 다중 작업 설정(multi-task setting)에 적합한 데이터 세트를 제안합니다.

- **Technical Details**: 기존의 감정 관련 데이터 세트를 확장하여 감정 라벨과 MQM 기반의 인간 평가 번역 오류를 포함하며, 문장 수준 및 단어 수준 번역 평가와 감정 분류에 적합한 아키텍처를 제안합니다. 여기에는 서로 다른 손실 휴리스틱을 통합한 새로운 결합 손실 함수가 포함됩니다.

- **Performance Highlights**: 이 방법은 감정 관련 QE 데이터 세트와 표준 QE 데이터 세트에서 새로운 최첨단 성능을 달성했으며, 다중 QE 데이터 세트에 대한 평가에서 기존의 미세 조정(fine-tuning) 및 다중 작업 학습(MTL) 방법보다 성능이 향상되었습니다.



### Adaptive BPE Tokenization for Enhanced Vocabulary Adaptation in Finetuning Pretrained Language Models (https://arxiv.org/abs/2410.03258)
Comments:
          11 pages. Accepted at EMNLP Findings 2024 (The 2024 Conference on Empirical Methods in Natural Language Processing)

- **What's New**: 이번 연구에서는 사전 훈련된 언어 모델(PLM)을 전문가 도메인에 맞게 fine-tuning하는 데 사용하는 Byte-Pair Encoding (BPE) 토크나이제이션 방식의 근본적인 한계를 보여줍니다. 기존 방식은 PLM의 어휘 끝에 대상 도메인 특화 어휘를 단순히 추가하는 방식으로 진행돼, 이는 서브 옵티멀(tokenization) 토크나이제이션을 초래합니다. 이를 해결하기 위해, 우리는 'AdaptBPE'를 제안합니다.

- **Technical Details**: AdaptBPE는 BPE 토크나이저의 초기화 단계를 수정하여 추가된 어휘(VDOMAIN)에 대해 가장 긴 문자열 매칭을 먼저 수행하도록 합니다. 중앙 토크나이제이션 단계 대신 자음 문자 수준으로 분할하던 기존 방식을 변경하여 어휘 적응의 ill-tokenization 문제를 완전히 완화합니다. 이 연구에서는 AdaptBPE가 표준 BPE에 비해 3.57%의 정확도 및 1.87%의 Rouge-L 개선을 보여줍니다.

- **Performance Highlights**: AdaptBPE는 AVocaDo와 MEDVOC의 경우 각각 8개의 데이터 세트(4개의 분류 및 4개의 요약 작업)에서 표준 BPE 알고리즘에 비해 3.57% 및 1.87% 향상을 보였으며, MEDVOC의 어려운 생성 시나리오에서는 10.41% 및 3.30%의 적어도 향상을 달성하였습니다. 또한 의료 전문가에 의한 사람 평가 결과, AdaptBPE는 MEDVOC에 비해 더 관련성 높은 요약을 생성하는 것으로 나타났습니다.



### Are Expert-Level Language Models Expert-Level Annotators? (https://arxiv.org/abs/2410.03254)
Comments:
          Accepted to WiML @ NeurIPS 2024 (extended version)

- **What's New**: 이 연구는 LLMs(대형 언어 모델)를 전문가 수준의 데이터 주석자로서 체계적으로 평가한 최초의 연구로, 금융, 생물의학 및 법률의 세 가지 전문 분야를 다룹니다.

- **Technical Details**: 연구는 REFinD, FOMC, AP-Relation, CODA-19, CUAD 및 FoDS 데이터셋을 사용하여 LLMs의 주석 성능을 평가하며, Chain-of-Thought (CoT), 자기 일관성(self-consistency), 자기 수정(self-refine) 등의 다양한 접근 방식을 적용했습니다. 또한 다수의 모델(GPT-3.5-Turbo, GPT-4o 등)을 비교하였고, 표준 직접 답변 프롬프트에서부터 시작하여 복잡한 이유를 유도하는 방법도 사용했습니다.

- **Performance Highlights**: LLMs는 인간 전문가 주석자에 비해 평균 32.2~43.3% 부족한 성과를 보였으며, 최고의 성과를 기록한 GPT-4o도 여전히 약 20%의 차이를 보였습니다. 이러한 결과는 도메인 전문 지식이 필요한 작업에서 LLMs의 주석 품질을 개선하기 위한 추가적인 접근 방식이 필요함을 시사합니다.



### Beyond Film Subtitles: Is YouTube the Best Approximation of Spoken Vocabulary? (https://arxiv.org/abs/2410.03240)
Comments:
          Submitted for review to COLING 2025. 8 pages, 3 figures

- **What's New**: YouTube 자막에서 추출한 단어 빈도 수치를 사용하여 다양한 언어에 대한 새롭고 효율적인 빈도 기준을 구축하였다. 특히, 기존에 이용 가능한 자원보다 더욱 정확한 결과를 보여주며, 고품질 자막이나 말뭉치가 없는 언어에서도 유효하다.

- **Technical Details**: 연구에서는 YouTube 자막에서 빈도를 추출하여 중국어, 영어, 인도네시아어, 일본어, 스페인어의 빈도 기준을 구축하였다. 이 빈도들은 Lexical Decision Time (LDT), 단어 친숙도, 및 Lexical Complexity와 높은 상관관계를 보였으며, 영어와 일본어의 Lexical Complexity 예측 작업에서 새로운 최고 점수를 기록했다.

- **Performance Highlights**: YouTube 자막 기반 빈도는 기존의 영화 자막 빈도 및 LLM GPT-4보다 더 나은 예측 결과를 나타내며, 두 개의 심리 언어학적 변수와 강한 상관관계를 바탕으로 성능 향상을 보였다.



### ALR$^2$: A Retrieve-then-Reason Framework for Long-context Question Answering (https://arxiv.org/abs/2410.03227)
- **What's New**: 최근 LLM(대형 언어 모델)의 컨텍스트 윈도우가 크게 확장되었으나, 그에 따른 정확한 추론 능력은 감소하는 문제가 보고되었습니다. 이러한 문제를 해결하기 위해, 새로운 retrieve-then-reason 프레임워크와 ALR$^2(Alternative Long-context Reasoning)을 도입하여 LLM의 장기 컨텍스트 추론 능력을 증대시키는 방법론이 제안되었습니다.

- **Technical Details**: ALR$^2는 LLM이 정보 검색과 추론의 목표를 정렬하도록 돕는 두 단계 프로세스를 기반으로 하며, 먼저 관련 증거를 검색하고 그 후에 검색된 증거를 바탕으로 최종 답안을 이끌어내는 구조입니다. 실험에서는 HotpotQA 및 SQuAD와 같은 장기 질문 답변(QA) 벤치마크에서 ALR$^2가 다른 방법들과 비교해 우수한 성능을 나타냈습니다.

- **Performance Highlights**: ALR$^2는 HotpotQA와 SQuAD 데이터셋에서 각각 8.4 및 7.9 EM(exact match) 점수의 개선을 이루었으며, 이는 경쟁하는 기준선들에 비해 상당히 높은 성능 향위를 보여줍니다. 또한, ALR$^2는 LLM의 정보 검색 문제, 즉 hallucin(환각)과 낮은 정확도를 완화하는 데 효과적인 것으로 나타났습니다.



### Consultation on Industrial Machine Faults with Large language Models (https://arxiv.org/abs/2410.03223)
Comments:
          9 pages

- **What's New**: 본 논문에서는 대규모 언어 모델(LLMs)을 활용하여 산업 기계 고장 진단의 정확도를 향상시키는 새로운 접근법을 제안합니다. 특히 구조화된 다중 라운드 프롬프트 기법을 통해 다양한 데이터 소스에서 정보를 통합하고, 문맥을 이해하며 실행 가능한 권장사항을 제시할 수 있는 모델의 능력을 개선합니다.

- **Technical Details**: 제안된 프레임워크는 GPT-4와 프롬프트 엔지니어링(prompt engineering)을 결합하여 산업 기계 고장 상담을 향상시키는 것을 목표로 합니다. 프롬프트를 통해 LLM이 센서 데이터 트렌드를 요약하고, 역사적 고장 데이터와 비교하며, 기계 문제를 진단하도록 유도합니다. 이 방법은 세 단계로 나뉘어 있으며, 각 단계는 데이터 요약, 고장 분석, 그리고 실행 가능한 권장 사항의 제안을 포함합니다.

- **Performance Highlights**: 본 연구의 실험 결과는 제안한 접근법이 전통적인 진단 기법보다 우수한 성능을 보인다는 것을 보여줍니다. 다양한 고장 유형 진단에서 91%의 정확도를 기록하며, 이는 고장 진단의 속도와 정확도를 크게 개선한 것입니다. 이러한 결과는 LLM들이 산업 고장 상담 실무를 혁신할 가능성을 제시합니다.



### NLIP_Lab-IITH Low-Resource MT System for WMT24 Indic MT Shared Task (https://arxiv.org/abs/2410.03215)
Comments:
          WMT2024 INDICMT Shared Task

- **What's New**: 이번 연구에서는 WMT 2024 저자원 인도어 번역의 공유 작업을 위한 시스템을 소개합니다. 이 작업에서는 영어(eng)와 아삼어(as), 카시어(kha), 미조어(lus), 마니푸르어(mni) 등의 언어 쌍을 고려합니다. 저자원 언어 번역의 문제를 해결하기 위해 모델의 미세 조정(fine-tuning)과 다국어 훈련(multilingual training)을 탐구하였습니다.

- **Technical Details**: 연구의 핵심 시스템은 사전 학습(pre-trained) 모델에 대한 언어별 미세 조정으로 구성됩니다. 22개의 인도 언어에 대한 embedding을 정렬하는 목표로 사전 학습된 IndicRASP 모델을 기반으로 하였으며, 이 모델은 두 개의 사전 학습 모델인 IndicRASP와 IndicRASP Seed에 대한 미세 조정을 통해 다양한 실험을 수행했습니다. 실험 결과, eng→as, eng→kha, eng→lus, eng→mni 쌍에서 각각 chrF2 점수 50.6, 42.3, 54.9, 66.3을 달성하였습니다.

- **Performance Highlights**: BLEU 점수 또한 공개 테스트 세트에서 eng→as 20.1, eng→kha 19.1, eng→lus 30.0, eng→mni 35.6을 기록하며, 언어별 미세 조정이 번역 품질 향상에 기여하였음을 보여주었습니다. 연구 결과에 따르면, 사전 학습된 alignment-augmented 모델을 활용하는 것이 저자원 환경에서도 높은 번역 품질을 개선할 수 있는 잠재력을 지닌 것으로 나타났습니다.



### Learning Semantic Structure through First-Order-Logic Translation (https://arxiv.org/abs/2410.03203)
Comments:
          EMNLP 2024 Findings

- **What's New**: 본 연구는 transformer 기반의 언어 모델이 단순 문장에서 술어 인자 구조(predicate argument structure)를 추출할 수 있는지를 탐구합니다. 이 과정에서 언어 모델이 어떤 술어가 어떤 객체에 적용되는지를 혼동하는 경우가 있음을 보이고, 이를 해결하기 위한 두 가지 작업(질문 응답(Q/A) 및 1차 논리(FOL) 번역)을 제안합니다.

- **Technical Details**: 연구에서는 두 가지 접근 방식인 질문 응답(Q/A) 및 1차 논리(FOL) 형태로의 번역을 통해 LLM의 술어 인자 구조 학습 능력을 분석합니다. FOL 번역에서는 대규모 언어 모델을 합성 데이터셋에 대해 미세 조정(finetuning)하여 일반화 능력을 측정합니다. 질문 응답의 경우 BERT 및 RoBERTa와 같은 인코더 모델을 미세 조정하며 LLM에서는 프롬프팅(promting) 방식을 사용합니다.

- **Performance Highlights**: 연구 결과, LLM에서 FOL 번역이 Q/A 프롬프팅보다 훨씬 나은 성능을 보이며, LLM이 단순 술어 인자 구조를 더 복잡한 문장으로 일반화할 수 있음을 보여줍니다. 또한, 프롬프트 방식으로는 모델이 허구의 내용을 추가할 때를 파악할 수 없지만, FOL 번역 방법은 이러한 한계를 극복합니다.



### PersoBench: Benchmarking Personalized Response Generation in Large Language Models (https://arxiv.org/abs/2410.03198)
- **What's New**: 이번 연구에서는 LLM(대규모 언어 모델)의 개인화된 응답 생성 능력을 평가하기 위한 새로운 벤치마크인 PersoBench를 소개합니다. 이는 기존 벤치마크가 다루지 않았던 개인화 능력에 초점을 맞췄습니다.

- **Technical Details**: PersoBench는 개인 인식을 고려한 대화 생성에서 LLM의 개인화 응답 생성 능력을 평가하며, 이는 제로샷(Zero-shot) 설정에서 수행됩니다. 데이터셋은 세 가지 오픈 소스 및 폐쇄 소스 LLM을 사용하여, 유창성(fluency), 다양성(diversity), 일관성(coherence), 개인화(personalization)의 여러 차원을 평가합니다. 이 연구는 COT(Chain of Thought) 프로토콜의 영향을 조사합니다.

- **Performance Highlights**: 연구 결과, LLM은 유창하고 다양한 응답을 생성하는 데에는 뛰어나지만, 대화의 맥락과 주어진 개인을 고려할 때 개인화되고 일관된 응답을 생성하는 데에는 여전히 과제가 있다고 나타났습니다. 이는 LLM의 응답 개인화 능력을 향상시키기 위해 추가 연구가 필요함을 강조합니다.



### Cross-lingual Transfer for Automatic Question Generation by Learning Interrogative Structures in Target Languages (https://arxiv.org/abs/2410.03197)
Comments:
          EMNLP 2024

- **What's New**: 자동 질문 생성(QG) 논문에서는 높은 자원을 가진 언어 데이터셋에서 학습된 모델이 자원이 부족한 언어에서 질문을 생성할 수 있도록 하는 크로스-링구얼 전송(XLT-QG) 방법을 제안합니다. 이 기술은 타겟 언어의 모노링구얼, 병렬, 또는 레이블된 데이터 없이 소형 언어 모델을 활용하여 작동합니다.

- **Technical Details**: 제안된 방법인 QuIST는 다음의 두 단계로 구성됩니다: 1) 질문 유형 분류(QTC)와 2) 질문 생성을 위한 질문 예시 활용. QTC 모델은 주어진 맥락과 응답에 따라 생성될 질문의 유형을 결정하며, 그에 따라 선택된 질문 예시를 활용해 질문을 생성합니다. 이 모델은 오직 영어 QA 데이터로 학습되었으며 타겟 언어에 대한 추가 학습 없이 질문을 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과 QuIST는 여러 XLT-QG 기초선보다 높은 성능을 보였고, 여러 언어에서 GPT-3.5-turbo와 유사한 성능을 달성했습니다. 또한 QuIST가 생성한 합성 질문들은 GPT-3.5-turbo에서 생성된 질문들보다 멀티링구얼 QA 모델의 학습에 더 효과적임을 확인했습니다.



### Parallel Corpus Augmentation using Masked Language Models (https://arxiv.org/abs/2410.03194)
Comments:
          21 Pages, 3 Figures. arXiv admin note: text overlap with arXiv:2011.01536 by other authors

- **What's New**: 본 논문에서는 고품질의 평행 텍스트 코퍼스를 증강할 수 있는 새로운 방법을 제안합니다. 이 방법은 시작하는 시드 코퍼스(seed corpus)보다 훨씬 더 많은 양의 코퍼스를 생성할 수 있습니다. 추가적인 단일 언어 코퍼스(monolingual corpora)가 필요하지 않으며, Multi-Lingual Masked Language Model을 사용하여 문맥에서 대체 단어를 마스킹하고 예측합니다.

- **Technical Details**: 제안된 방법은 트랜스포머(Transformer) 기반 BERT 모델인 XLM-RoBERTa를 Fill-Mask 모드에서 사용하여 단어를 마스킹하고 예측하여 문장 쌍을 생성합니다. 생성된 문장은 Sentence Embeddings(예: LaBSE 모델)를 활용해 유사성을 검사하여 번역 쌍을 선택합니다. 번역 동등성을 확인하기 위해 생성된 문장 점수를 기준으로 필터링 합니다.

- **Performance Highlights**: 이 방법은 유사할 것으로 예상되는 문장 쌍을 생성하여 데이터 부족 문제를 크게 완화할 수 있으며, 특히 신뢰할 수 있는 단어 정렬 기술이 부족한 현재의 상황에서도 새로운 문장 쌍을 효과적으로 생성할 수 있는 가능성을 보여줍니다.



### Generating bilingual example sentences with large language models as lexicography assistants (https://arxiv.org/abs/2410.03182)
- **What's New**: 이번 연구에서는 고자원(high-resource) 언어인 프랑스어, 중자원(mid-resource) 언어인 인도네시아어, 저자원(low-resource) 언어인 테툰(Tetun)을 포함한 다양한 언어에서 LLM(대형 언어 모델)의 성능을 평가했습니다. LLM이 생성한 예문이 GDEX(Good Dictionary EXample) 기준을 충족하는지를 분석하여, 자원 수준에 따른 성능 차이를 확인했습니다.

- **Technical Details**: 연구에서는 두 개의 LLM, GPT-4o와 Llama 3.1 405b를 사용하여 예문을 생성했습니다. GDEX 기준에 따른 예문 생성을 위해 상위 10,000개의 빈도수가 높은 단어 리스트를 사용하였으며, 각 언어에 대해 50개의 단어 쌍을 선택하였습니다. 또한, 예문에 대한 질적 평가는 원어민이 수행하였습니다.

- **Performance Highlights**: 저자원 언어의 경우 LLM이 생성한 예문 품질이 낮아지는 경향을 보였으며, 인간 주석자 간의 일치율이 낮아 пример의 질에 대한 선호에 큰 변동성이 있음을 발견했습니다. 그러나, in-context learning 기법을 통해 LLM을 개인 주석자의 선호에 맞출 수 있음을 시연하였습니다. 자동 예문 평가는 높은 자원 수준 언어에서 문장 혼란도가 GDEX 기준의 typicality 및 intelligibility에 적합한 대리변수가 된다는 것을 조사했습니다.



### Kiss up, Kick down: Exploring Behavioral Changes in Multi-modal Large Language Models with Assigned Visual Personas (https://arxiv.org/abs/2410.03181)
Comments:
          EMNLP 2024

- **What's New**: 이번 연구는 다중 모달 (multi-modal) 대규모 언어 모델 (LLMs)이 시각적 페르소나 (visual personas)에 맞춰 행동을 조정할 수 있는지 여부를 최초로 탐구하였으며, 주로 텍스트 기반 페르소나에 국한된 문헌의 주요 공백을 다룹니다.

- **Technical Details**: 본 연구는 5,000개의 픽션 아바타 이미지로 구성된 신규 데이터셋을 개발하고, LLMs의 협상 행동을 분석하여 시각적 특성이 공격성이 어떻게 나타나는지를 중심으로 했습니다. 평가를 위해 GPT-4o 및 Claude 3 Haiku 모델을 사용하여 아바타 이미지의 공격성을 등급 매겼습니다.

- **Performance Highlights**: LLMs는 인간과 유사한 방식으로 이미지의 공격성을 평가하며, 공격적인 시각적 페르소나로 프롬프트를 받을 때 보다 공격적인 협상 행동을 보였습니다. 흥미롭게도, 자신의 이미지보다 상대방의 이미지가 덜 공격적으로 보일 때 LLM이 더 공격적인 협상 행동을 나타내고, 반대로 상대방의 이미지가 더 공격적으로 보일 때는 덜 공격적인 행동을 보였습니다.



### Autoregressive Large Language Models are Computationally Universa (https://arxiv.org/abs/2410.03170)
Comments:
          32 pages

- **What's New**: 이 논문에서는 transformer 기반의 언어 모델이 외부 개입이나 모델 가중치의 수정 없이도 범용 계산(universal computation)을 실현할 수 있다는 것을 보여줍니다. 특히, 자가 회귀 디코딩(autoregressive decoding)의 일반화된 형태를 제시하여, 긴 입력을 처리할 수 있는 구조를 제안합니다.

- **Technical Details**: 이 논문에서는 자가 회귀 디코딩의 일반적인 확장에 대해 논의하며, 긴 입력 문자열을 처리하기 위해 출력된 토큰을 시퀀스의 끝에 추가하는 방식으로 진행됩니다. 이 새로운 접근법은 고전적인 계산 모델인 Lag 시스템에 해당하며, 우리는 2027개의 프로덕션 규칙을 가진 Lag 시스템이 범용 튜링 기계(universal Turing machine)를 시뮬레이션할 수 있다는 것을 보여줍니다. 또한, gemini-1.5-pro-001 모델이 이러한 범용 Lag 시스템의 동작을 시뮬레이션할 수 있도록 하는 시스템 프롬프트를 개발하였습니다.

- **Performance Highlights**: gemini-1.5-pro-001은 확장된 자가 회귀 디코딩(greedy decoding) 이론을 통해, 특정 입력을 가지고 U15,2라는 범용 튜링 기계의 실행을 정확히 시뮬레이션할 수 있습니다. 따라서, gemini-1.5-pro-001은 범용 컴퓨터로 작동할 수 있는 가능성을 보여줍니다.



### Exploring Learnability in Memory-Augmented Recurrent Neural Networks: Precision, Stability, and Empirical Insights (https://arxiv.org/abs/2410.03154)
Comments:
          21 pages, 4 theorems, 5 tables

- **What's New**: 이 연구는 메모리가 없는 RNN과 메모리 증강 RNN의 학습 능력을 탐구하며, 이들이 Pushdown Automata와 이론적으로 동등하다는 점을 강조하고 있습니다. 또한, 메모리 컴포넌트를 동결(freeze)할 경우 성능이 크게 향상된다는 것을 보여주었습니다.

- **Technical Details**: 메모리 컴포넌트를 동결한 모델은 Penn Treebank 데이터셋에서 123.5에서 120.5로 perplexity가 감소하여 최첨단 성능을 달성했습니다. 메모리가 동결된 모델은 긴 시퀀스에 대해 초기 성능의 90%를 유지하는 반면, 일반 RNN 모델은 60%의 성능 하락을 겪었습니다.

- **Performance Highlights**: 이 연구의 주요 결과는 메모리 동결이 시간적 의존성을 안정화시켜 견고한 수렴을 이끌어낸다는 것이며, 이는 RNN의 학습 가능성 한계를 이해하기 위해 메모리 설계의 안정성과 긴 시퀀스 평가가 필요함을 강조합니다.



### Media Framing through the Lens of Event-Centric Narratives (https://arxiv.org/abs/2410.03151)
Comments:
          Accepted to the 6th Workshop on Narrative Understanding, co-located with EMNLP 2024

- **What's New**: 이 논문에서는 미디어 프레이밍(media framing) 분석을 위한 새로운 접근 방법을 제안합니다. 기존의 방법들이 고수준 토픽 마커를 사용하는데 비해, 본 연구는 서사(narrative)의 역할을 중심으로 두고, 사건(event)과 그 관계를 추출하여 높은 수준의 서사로 집단화하는 프레임워크를 소개합니다.

- **Technical Details**: 연구진은 사건 중심의 서사 표현(event-centric narrative representations)을 추출하고, 이를 통해 고수준 주제(high-level themes)로 그룹화하여 미디어 프레임을 설명하는 프레임워크를 구축했습니다. 이 과정에서 사건들 간의 시간적 관계(temporally related)와 인과 관계(causally related)를 예측합니다.

- **Performance Highlights**: 제안된 프레임워크는 미국의 이민(immigration)과 총기 규제(gun control) 두 가지 서로 다른 뉴스 도메인에서 활용 가능성을 입증하였고, 이민 분야에서는 고품질의 서사 클러스터를 생성하여 Boydstun et al. (2014) 정책 프레임 분류법에 대한 예측 및 설명에 중요한 신호를 제공하는 것으로 평가되었습니다.



### Analysis and Detection of Differences in Spoken User Behaviors between Autonomous and Wizard-of-Oz Systems (https://arxiv.org/abs/2410.03147)
Comments:
          Accepted and will be presented at the 27th conference of the Oriental COCOSDA (O-COCOSDA 2024)

- **What's New**: 이번 연구는 일본어 인간-로봇 상호작용의 대규모 코퍼스를 분석하여 원거리 조작 로봇과 자율 대화 시스템 간의 사용자의 행동 차이를 조사하였습니다. 주목할 점은 서로 다른 대화 시나리오에서 사용자 언어 행동의 메트릭이 어떻게 달라지는지를 비교했다는 것입니다.

- **Technical Details**: 본 연구는 대화형 로봇과 음성 대화 시스템(SDS)의 사용 성과를 측정하기 위해 대규모 인간-로봇 상호작용 데이터를 수집하고 분석했습니다. 실험에서는 반 자동화 시스템을 사용하여 사용자가 자율적으로 작동하는 로봇과 상호작용하도록 하여 언어 행동의 차이를 평가했습니다. 연구는 IPU(inter-pausal units)를 단위로 하여 발화 길이, 발화 속도, 채워넣기 표현(fillers), 백채널(backchannels), 비유창성(disfluencies), 웃음(laughter)과 같은 메트릭을 측정했습니다.

- **Performance Highlights**: 연구에서 생성한 예측 모델은 원거리 조작 시스템과 자율 시스템 상태를 구분하는 데 있어 기준 모델보다 더 높은 정확도와 정밀도를 보였으며, 여러 모델에서 F1 점수(F1 score)가 기준을 초과했습니다. 연구 결과는 수집한 데이터를 바탕으로 대화형 로봇의 사용자 행동 차이를 정확하게 분석하고 향후 시스템 개선의 가능성을 보여줍니다.



### Margin Matching Preference Optimization: Enhanced Model Alignment with Granular Feedback (https://arxiv.org/abs/2410.03145)
Comments:
          EMNLP 2024 Findings

- **What's New**: 본 논문에서는 Margin Matching Preference Optimization (MMPO)이라는 새로운 접근 방식을 소개하여 상대 품질 여백을 최적화에 통합함으로써 기존 방법의 한계를 극복하고자 합니다. 이는 모델 정책과 보상 모델을 개선하는 데 기여합니다.

- **Technical Details**: MMPO는 pairwise preference에서 주어진 품질 여백을 기반으로 소프트 타겟 확률을 설계하는 방식으로 진행됩니다. 이 확률은 Bradley-Terry 모델에 따라 결정되며, 이후 cross-entropy 목표를 사용하여 모델을 훈련하는 데 활용됩니다. MMPO는 사람의 피드백이나 AI 피드백 데이터 모두에서 사용됩니다.

- **Performance Highlights**: MMPO는 MT-bench 및 RewardBench와 같은 인기 있는 벤치마크에서 기존 방법들을 지속적으로 초과 성과를 보였으며, 특히 2024년 6월 기준으로 7B 모델이 RewardBench에서 최첨단 성능을 달성하였습니다. 또한 MMPO는 과적합에 대한 저항력이 더 높아 잘 보정된 모델을 생성하여 일반화 성능이 향상됩니다.



### SAG: Style-Aligned Article Generation via Model Collaboration (https://arxiv.org/abs/2410.03137)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)과 소형 언어 모델(SLMs)의 강점을 결합한 새로운 협력 훈련 프레임워크를 제안합니다. 이를 통해 스타일 기반 기사의 생성 성능을 향상시키고, 기존 모델의 한계를 초월하는 효과를 보여주고자 합니다.

- **Technical Details**: 제안된 접근 방식에서는 LLM을 동결시켜 해당 모델의 강력한 지침 따르기 능력을 활용하고, 이어서 스타일 특정 데이터에 대한 감독된 미세 조정(supvised fine-tuning, SFT)을 SLM에 적용합니다. SLM의 훈련 과정은 스타일 감독 미세 조정(style supervised fine-tuning, S-SFT)과 콘텐츠 직접 선호 최적화(content direct preference optimization, C-DPO)의 두 단계로 구성됩니다.

- **Performance Highlights**: 우리는 NoteBench라는 새로운 벤치마크를 도입하여 스타일 일치 생성 성능을 평가합니다. 우리의 방법론은 GPT-4 대비 ROUGE-L에서 0.78, BLEU-4에서 0.55의 향상을 보여주며, 사실성 및 충실도에 대한 낮은 환각(hallucination) 비율을 유지합니다.



### Deliberate Reasoning for LLMs as Structure-aware Planning with Accurate World Mod (https://arxiv.org/abs/2410.03136)
- **What's New**: 이번 논문에서 제안하는 구조적 계획 프레임워크(SWAP)는 기존의 Chain-of-Thought (CoT) 접근 방식보다 더 나은 복잡한 추론 능력을 위한 다중 단계 추론을 촉진합니다. SWAP는 정확한 세계 모델을 이용한 구조적 정보를 통합하며, 단계를 소프트 검증하는 메커니즘을 제공합니다.

- **Technical Details**: SWAP는 Generator-Discriminator 아키텍처를 도입하여 복잡한 추론 작업에서의 정확한 세계 상태 예측 문제를 해결합니다. Generator는 다음 상태를 예측하고, Discriminator는 문제 맥락에서 필요한 논리적 일관성을 보장합니다. 이를 통해 모델이 다음 단계의 상태를 예측하고, 오류를 줄일 수 있습니다. 또한 정책 모델이 다양한 행동을 탐색할 수 있도록 유도하여 조기 수렴을 방지합니다.

- **Performance Highlights**: SWAP는 수학적, 논리적 추론, 코딩 작업을 포함한 다양한 벤치마크에서 평가되었으며, 기존의 LLM과 비교하여 큰 성능 개선을 보여줍니다. 실험 결과 SWAP는 최근의 인기 있는 추론 및 계획 방법론에 비해 일관되게 우수한 성과를 나타냈으며, 일반화된 프레임워크로서의 가능성을 보여줍니다.



### On Unsupervised Prompt Learning for Classification with Black-box Language Models (https://arxiv.org/abs/2410.03124)
- **What's New**: 이번 연구에서는 라벨이 없는 데이터로 블랙박스 LLMs(large language models)를 파인튜닝(fine-tuning)하는 가능성을 탐구하는 언슈퍼바이즈드 프롬프트 학습(un,supervised prompt learning) 방식을 제안하였습니다. 특히, 프롬프트(prompt)와 가짜 라벨(pseudo labels)을 동시에 학습하여 분류 문제를 해결하며, 이렇게 생성된 가짜 라벨 데이터셋을 이용하여 추가적인 파인튜닝이 가능합니다.

- **Technical Details**: 저자들은 프롬프트를 의미 있는 암호화된 토큰 시퀀스로 모델링하며, 각 토큰마다 학습되어야 하는 범주형 분포(categorical distribution)를 최적화합니다. 이 과정에서 LLM의 인컨텍스트 학습(in-context learning; ICL) 기능을 활용하여 신뢰할 수 있는 가짜 라벨 데이터(pseudo-labeled data)를 식별하고, 이러한 데이터를 통해 다른 라벨이 없는 데이터에 대한 가짜 라벨을 생성합니다. 이를 통해 프롬프트 학습과 사용 단계에서의 일관성을 높입니다.

- **Performance Highlights**: 기준 데이터셋에서 수행한 실험 결과, 제안된 알고리즘의 효과가 입증되었습니다. 언슈퍼바이즈드 프롬프트 학습 후 생성된 가짜 라벨 데이터셋은 블랙박스 LLM 소유자에 의해 추가 파인튜닝을 위해 사용할 수 있습니다.



### RIPPLECOT: Amplifying Ripple Effect of Knowledge Editing in Language Models via Chain-of-Thought In-Context Learning (https://arxiv.org/abs/2410.03122)
Comments:
          EMNLP findings

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)에서 지식 편집의 도전 과제인 ripple effect를 해결하기 위한 새로운 접근법, RippleCOT를 제안합니다. 이 방법은 Chain-of-Thought (COT) 추론을 통합하여 다중 단계 질문을 처리할 수 있도록 돕습니다.

- **Technical Details**: RippleCOT는 'new fact, question, thought, answer' 구조로 시연을 구성하며, 새로운 사실을 기반으로 다중 단계 질문의 논리를 식별하고 분해합니다. 이 접근 방식은 첫 번째 단계에서 여러 관계를 식별하고 두 번째 단계에서 생성된 쌍 중에서 질문과 높은 코사인 유사성을 가지는 상위 k 후보를 선택합니다.

- **Performance Highlights**: RippleCOT는 기존의 방법보다 우수한 성능을 보이며, MQuAKE-cf 벤치마크에서 정확도를 7.8%에서 87.1%까지 향상시켰습니다.



### Precision, Stability, and Generalization: A Comprehensive Assessment of RNNs learnability capability for Classifying Counter and Dyck Languages (https://arxiv.org/abs/2410.03118)
Comments:
          21 pages, 5 figures, 5 tables

- **What's New**: 이번 연구는 순환 신경망(Recurrent Neural Networks, RNNs)의 학습 가능성을 정형 언어 분류에 대한 새로운 관점에서 조명합니다. 특히 카운터 언어와 Dyck 언어에 중점을 두어 연구했고, 기존의 LSTM과 O2RNN이 효과적이라는 기존 신념에 도전합니다. 연구 결과 RNN이 상태 기계로 작동하며 언어 처리 능력이 매핑의 정밀성과 부정 샘플링 전략에 크게 의존함을 보여주었습니다.

- **Technical Details**: 연구에서는 LSTM과 O2RNN의 성능을 비교하고, LSTM의 셀 상태가 동적 카운터를 모방할 충분한 표현력을 갖추고 있지만, 이 안정성이 반드시 보장되지 않는다는 점을 강조했습니다. 또한, 초기 가중치 분포는 카운터 동역학의 붕괴에 큰 영향을 미치지 않는다고 밝혔습니다.  진행한 실험에서는 40 길이의 문자열로 모델을 학습하고 41부터 500 길이의 문자열에서는 일반화 성능을 평가했습니다.

- **Performance Highlights**: 단순한 단층 분류기가 RNN 임베딩을 사용했을 때 무작위 결과보다 더 나은 성과를 보였고, O2RNN은 다양한 시나리오에서 더 높은 안정성을 제공했습니다. 연구는 RNN의 학습 가능성에 대한 기존의 신념을 수용하는 데 중요한 의미를 부여하며, 언어 분류 작업에서 데이터 구조와 샘플링 기법의 중요성을 강조합니다.



### X-ALMA: Plug & Play Modules and Adaptive Rejection for Quality Translation at Sca (https://arxiv.org/abs/2410.03115)
- **What's New**: 이 논문에서는 다국어(interpretation multilingual) 기계 번역(machinel translation)에 중점을 두고, 50개 다양한 언어에 대한 고품질 성능을 보장하는 X-ALMA 모델을 소개합니다.

- **Technical Details**: X-ALMA는 플러그 앤 플레이 언어 특화 모듈 아키텍처를 기반으로 하여 훈련 중 언어 충돌을 방지하는 기능을 가지고 있으며, 새로운 최적화 방법을 사용하여 번역 성능을 극대화합니다. 최종 훈련 단계에서는 Adaptive Rejection Preference Optimization (ARPO)를 통해 기존의 선호 최적화 방법을 초월합니다.

- **Performance Highlights**: X-ALMA는 FLORES와 WMT'23 테스트 데이터셋에서 Aya-101, Aya-23 등 기존 최신 오픈소스 다국어 LLM보다 모든 번역 방향에서 뛰어난 성능을 보입니다.



### CoCoHD: Congress Committee Hearing Datas (https://arxiv.org/abs/2410.03099)
Comments:
          Accepted at EMNLP 2024

- **What's New**: 이번 논문에서는 1997년부터 2024년까지의 미국 의회 청문회 자료를 포함하는 Congress Committee Hearing Dataset (CoCoHD)를 소개합니다. 이 데이터셋은 86개의 위원회에서 총 32,697개의 기록을 수집하여, 정책 언어를 연구하는 데 필요한 귀중한 자료를 제공합니다.

- **Technical Details**: CoCoHD는 1997년부터 2024년까지의 청문회 기록과 메타데이터를 포함하고 있으며, NLP(자연어 처리) 모델을 미세 조정하여 화석 연료 소비에 대한 청문회의 견해를 분석합니다. 데이터셋은 청문회 자료를 JSON 포맷으로 정리하고, 메타데이터를 통해 참여자 이름, 날짜, 토론 주제와 같은 중요한 정보를 제공합니다.

- **Performance Highlights**: CoCoHD를 활용한 분석 결과, 에너지 및 상업 위원회가 청문회에서 화석 연료 생산에 대해 긍정적, 부정적, 중립적인 견해를 가지는 경향을 정량화할 수 있음을 보여줍니다. 이 시장 분석은 CoCoHD가 에너지 분야의 트렌드를 예측하고 강조할 수 있는 기회를 제공한다는 것을 시사합니다.



### UNComp: Uncertainty-Aware Long-Context Compressor for Efficient Large Language Model Inferenc (https://arxiv.org/abs/2410.03090)
- **What's New**: 본 논문에서는 UNComp라는 새로운 압축 방안을 제안합니다. UNComp는 매트릭스 엔트로피(matrix entropy)를 활용하여 모델의 불확실성을 추정하고, 이를 통해 K-V 캐시(KV cache)와 은닉 상태(hidden states)를 적응적으로 압축할 수 있는 방법을 제공합니다.

- **Technical Details**: UNComp는 레이어(layer)와 헤드(head)를 기반으로 그룹화하여 압축 비율을 조정합니다. 이 방법은 KV 캐시를 4.74%까지 압축하고, 단일 배치에서 프리필링(prefilling) 단계에서 1.6배 속도 향상과 6.4배의 처리량 증가를 낳습니다. 또한 성능 저하는 1.41%에 불과합니다.

- **Performance Highlights**: UNComp는 특정 작업에서 9.38% 압축 비율에도 불구하고 전체 KV 캐시의 성능을 초과합니다. 이를 통해 기계 학습 모델이 효율성을 높이고, 긴 맥락 상황에서의 추론 속도를 대폭 향상시킬 수 있음을 보여줍니다.



### Scaling Parameter-Constrained Language Models with Quality Data (https://arxiv.org/abs/2410.03083)
Comments:
          Accepted to EMNLP 2024 Industry Track, 18 pages, 9 figures, 4 tables

- **What's New**: 이번 연구는 언어 모델(리터러리 모델)의 스케일링 법칙을 확장하여 데이터 품질의 중요성을 강조합니다. 특히, 매개변수 수에 의해 제한된 언어 모델에서 효과적인 훈련 토큰(effective training tokens)의 개념을 통합하여 데이터 품질이 모델 성능에 미치는 영향을 분석하였습니다.

- **Technical Details**: 본 논문에서는 모델 매개변수 수가 10억 이하인 경우 데이터의 질, 즉 효과적인 훈련 토큰을 모델 성능의 결정적 요소로 간주하고, 텍스트 다양성(text diversity)과 합성성(syntheticity)이라는 두 가지 지표를 활용하여 계산합니다. 텍스트 다양성은 압축 비율(compression ratio)로 측정하며, 합성성은 교사 모델(teacher model)을 활용한 Perplexity 지표로 평가합니다.

- **Performance Highlights**: 결과적으로, 학습된 200개 모델의 성능을 평가한 결과, 제안된 효과적인 훈련 토큰 개념을 통합한 스케일링 법칙이 기존 방식보다 +0.83 피어슨 상관관계(Pearson correlation)를 보이며, 데이터 샘플링 및 합성 등 널리 사용되는 데이터 기술에서 모델 성능을 크게 향상시켰습니다.



### CommonIT: Commonality-Aware Instruction Tuning for Large Language Models via Data Partitions (https://arxiv.org/abs/2410.03077)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 명령 수행 능력을 향상시키기 위해 새롭게 제안한 'CommonIT' 방법론을 소개합니다. CommonIT는 데이터 샘플링 관점에서 모델의 능력을 향상시키는 데 중점을 두며, 여러 데이터 세트를 군집화한 후 각 군집에 대해 고유한 특성을 고려하여 명령 튜닝을 적용합니다.

- **Technical Details**: CommonIT는 세 가지 기준(작업, 임베딩 및 길이)을 사용하여 명령 데이터 세트를 그룹으로 나누고 각 훈련 미니 배치가 오직 하나의 그룹의 데이터로만 구성되도록 합니다. 이렇게 함으로써 미니 배치 간 데이터의 무작위성을 보장하고, 배치 내 데이터 유사성을 높이게 됩니다. 연구에서는 다양한 LLaMa 모델을 통해 CommonIT의 효과를 검증했습니다.

- **Performance Highlights**: 실험 결과 CommonIT는 일반 도메인에서 평균 2.1% 향상, 특정 도메인에서 5.2% 향상, 그리고 특정 작업인 MMLU에서 3.8% 향상을 보였습니다. 이러한 결과는 모델이 명령을 더 잘 이해하고 응답에 있어 정확성을 높였음을 나타냅니다.



### Multilingual Topic Classification in X: Dataset and Analysis (https://arxiv.org/abs/2410.03075)
Comments:
          Accepted at EMNLP 2024

- **What's New**: 이번 연구에서는 X-Topic이라는 다국어 데이터셋을 소개합니다. 이 데이터셋은 영어, 스페인어, 일본어 및 그리스어로 작성된 트윗을 포함하며, 다국어 주제 분류를 위해 설계되었습니다.

- **Technical Details**: X-Topic 데이터셋은 19개의 주제에 대한 초기 주제 분류 체계를 기반으로 하며, 약 80만개의 트윗을 포함하고 있습니다. 데이터 수집에는 Twitter API를 사용하고, 게시된 트윗의 언어를 검증한 후 다양한 전처리 과정이 적용되었습니다.

- **Performance Highlights**: 연구진은 X-Topic 데이터셋을 기준으로 여러 모델 아키텍처와 크기를 탐색하였으며, 특히 LLMs(대형 언어 모델)와 감독 학습 방법을 사용하여 다국어 주제 분류 작업의 도전적인 성격을 강조했습니다.



### Enhancing Short-Text Topic Modeling with LLM-Driven Context Expansion and Prefix-Tuned VAEs (https://arxiv.org/abs/2410.03071)
Comments:
          EMNLP Findings 2024. arXiv admin note: substantial text overlap with arXiv:2310.15420

- **What's New**: 이번 연구에서는 짧은 텍스트에서 유의미한 주제를 추출하기 위해 대형 언어 모델(LLMs)을 활용하여 짧은 텍스트를 보다 상세한 시퀀스로 확장한 후 주제 모델링을 적용하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 Prefix-tuned Variational Topic Model (PVTM)을 도입하여 짧은 텍스트에서 주제를 추론하는 소형 언어 모델과 변분 오토인코더(VAE)를 결합합니다. 이때, 전체 언어 모델을 튜닝하는 대신 prefix tuning 방법을 사용하여 특정 도메인 기능을 효과적으로 포착합니다.

- **Performance Highlights**: 본 모델은 극심한 데이터 희소성 문제가 있는 실제 데이터셋에서 기존의 최첨단 주제 모델보다 우수한 성능을 보였습니다. 다양한 데이터셋과 여러 작업에 대한 포괄적인 실험을 통해 이 모델의 우수성을 입증하였습니다.



### Scalable Frame-based Construction of Sociocultural NormBases for Socially-Aware Dialogues (https://arxiv.org/abs/2410.03049)
Comments:
          17 pages

- **What's New**: 본 논문은 대화에서 사회적으로 인식된 행동을 지원하기 위해 대형 언어 모델(LLMs)을 활용한 사회문화적 규범(SCN) 제작을 제안합니다. 이를 통해 중국 문화에 특화된 첫 번째 SCN 데이터베이스인 ChineseNormBase를 구축했습니다. 이 데이터베이스는 사회적 맥락을 고려하여 생성된 자연어 규범 진술을 포함하고 있습니다.

- **Technical Details**: SCNs는 사회맥락적 프레임을 이용해 추출되며, 이 과정은 대화의 맥락을 이해하고 환각(hallucination)을 줄이는 데 도움이 됩니다. 실제 대화 데이터가 부족할 경우, 합성 데이터(synthetic data)를 사용하여 SCNs를 효과적으로 생성할 수 있습니다. 이와 더불어, RAG 기반 모델(Retrieval-Augmented Generation)을 통해 다양한 대화 작업에 대한 추론 능력을 시연했습니다.

- **Performance Highlights**: 실험 결과, 합성 데이터를 이용하여 추출한 SCNs의 품질이 금본(gold frames)으로 주석을 달아 만든 실제 대화에서 추출한 SCNs와 동등하다는 것을 확인했습니다. 또한, 은본(silver frames)이나 금본으로 주석을 단 실제 데이터에서 추출된 SCNs의 품질이 주석이 없는 데이터와 비교하여 우수함을 입증했습니다.



### Geometry is All You Need: A Unified Taxonomy of Matrix and Tensor Factorization for Compression of Generative Language Models (https://arxiv.org/abs/2410.03040)
- **What's New**: 이번 연구는 자연어 처리(NLP) 모델의 매트릭스(matrix)와 텐서(tensor) 지향 파라미터화(parametrization)를 통합하는 새로운 분류체계를 제안합니다. 이는 기존 연구의 수학적 접근에서 벗어나서 머신러닝(ML) 및 NLP 연구와의 연관성을 강화하려는 시도입니다.

- **Technical Details**: 연구는 선형대수학의 기본 개념인 서브스페이스(subspace)를 채택하여 매트릭스 및 텐서와 ML/NLP 개념을 재정형화했습니다. 이렇게 함으로써 일반적인 매트릭스 및 텐서 분해 알고리즘이 기하학적 변환으로 해석됩니다.

- **Performance Highlights**: 마지막으로, 본 연구는 최근 매트릭스 또는 텐서 기반 언어 모델 압축 문헌을 재조명하고, 이들의 핵심 아이디어를 비교 분석하여 현재 연구의 공백과 가능성 있는 해결책을 제시했습니다.



### Disentangling Textual and Acoustic Features of Neural Speech Representations (https://arxiv.org/abs/2410.03037)
- **What's New**: 이번 연구는 Neural Speech Models(NSMs)의 내부 표현을 텍스트 정보와 음향 특성으로 분리하는 새로운 Framework를 제안합니다. 정보 병목(Information Bottleneck) 원리를 기반으로 하여 복잡한 음성 표현을 내용(content)과 음향 특성(acoustic features)으로 분리합니다.

- **Technical Details**: 우리는 두 단계의 분리 프레임워크를 구축하였습니다. 첫 번째 단계에서는 Decoder를 훈련하여 내부 표현을 텍스트로 변환하고, 비관련 정보를 최소화하려 합니다. 두 번째 단계에서는 같은 음성 표현을 기반으로 하지만, 텍스트 정보를 피하면서 특정 작업에 유리한 음향 특성을 캡쳐하는 Decoder를 훈련합니다.

- **Performance Highlights**: 우리의 프레임워크는 감정 인식과 화자 식별 두 가지 다운스트림 작업에서 강력한 성능을 보였습니다. 텍스트 표현은 전통적인 음성 모델보다 효과적으로 텍스트를 예측할 수 있었으며, 음향 표현은 음향 특성을 더 잘 예측하는 데 성공했습니다.



### Characterizing Context Influence and Hallucination in Summarization (https://arxiv.org/abs/2410.03026)
- **What's New**: 이 연구에서는 Large Language Models (LLMs)의 맥락 정보가 요약 생성 시 어떻게 영향을 미치는지를 정의하고 분석합니다. 새로운 개념인 Context Influence Decoding (CID)을 도입하여 맥락의 영향을 증가시키는 방법을 제시합니다.

- **Technical Details**: 연구에서 제안된 CID는 Pointwise Cross-Mutual Information을 기반으로 한 맥락 영향력의 정의를 포함하고 있습니다. 이를 통해 기존의 모델에서는 고려되지 않았던 맥락 정보의 중요성과 LLM의 환각 문제를 동시에 분석합니다.

- **Performance Highlights**: 실험 결과, LLaMA 3 모델의 CNN-DM 데이터셋에서 ROGUE-L 점수를 10% 향상시킨 결과, 맥락의 영향력이 1.5배 증가하는 것을 확인했습니다. 이는 모델의 성능을 높이는 동시에 개인 정보 유출의 가능성을 분석하는 데 중요합니다.



### Is Your Paper Being Reviewed by an LLM? Investigating AI Text Detectability in Peer Review (https://arxiv.org/abs/2410.03019)
- **What's New**: 이번 연구는 현재의 AI 텍스트 탐지 알고리즘이 연구자의 피어 리뷰를 인간과 다양한 최신 대형 언어 모델(GPT-4o 등)로 구별하는 능력을 조사하였다. 기존 접근 방식들이 많은 GPT-4o 작성의 리뷰를 식별하는 데 실패하며 높은 허위 긍정 분류를 발생시키는 문제를 드러냈다.

- **Technical Details**: 본 연구에서는 피어 리뷰의 AI 생성 텍스트를 탐지하기 위해 여러 AI 텍스트 탐지 방법의 적합성을 분석하였다. GPT-4o 및 Llama-3.1 모델을 사용하여 AI 피어 리뷰를 생성하고, 기존의 오픈 소스 및 상용 텍스트 탐지 모델과 성능을 비교하였다. 제안된 새로운 방법은 참조 AI 생성 리뷰와 비교하는 방식으로 GPT-4o 작성 리뷰를 더 효과적으로 탐지하였다.

- **Performance Highlights**: 기존 AI 텍스트 탐지 방법들이 낮은 허위 긍정률을 유지하면서 AI 생성 리뷰를 식별하는 데 한계가 있었다. 우리의 방법은 GPT-4o 작성 리뷰를 97%, Llama-3.1 작성 리뷰를 87-90% 정확도로 탐지하는 성능을 보여주었다. 이는 피어 리뷰에서 AI 작성 텍스트를 탐지하는 데 큰 진전을 의미한다.



### Tutor CoPilot: A Human-AI Approach for Scaling Real-Time Expertis (https://arxiv.org/abs/2410.03017)
Comments:
          Our pre-registration for this randomized controlled trial can be found here: this https URL

- **What's New**: 이번 연구는 Tutor CoPilot이라는 새로운 Human-AI 시스템을 소개하며, 이는 교육 분야에서 전문가의 사고 모델을 활용하여 튜터들에게 전문가와 유사한 지침을 제공하는 혁신적인 접근 방식을 제시합니다.

- **Technical Details**: 이 연구는 900명의 튜터와 1,800명의 K-12 학생을 대상으로 한 무작위 대조 실험으로, 튜터가 Tutor CoPilot에 접근할 경우 학생들이 주제 마스터링 확률이 평균 4 퍼센트 포인트(p.p.) 증가함을 보여줍니다. 특히, 낮은 평가를 받은 튜터는 9 p.p.의 이익을 경험했습니다. 연구에서는 550,000건 이상의 메시지를 분석하여, Tutor CoPilot을 사용하는 튜터가 고품질의 교육 전략을 사용할 확률이 더 높음을 확인했습니다.

- **Performance Highlights**: Tutor CoPilot의 연간 비용은 튜터당 20달러로 매우 저렴하여 기존의 자원 집약적인 교육 프로그램에 비해 확장성과 비용 효율성을 제공합니다. 이 시스템은 저소득 학생들에게 고품질의 교육 경험을 제공하는 데 중요한 역할을 할 것으로 기대됩니다.



### Can Transformers Learn $n$-gram Language Models? (https://arxiv.org/abs/2410.03001)
- **What's New**: 본 연구는 transformers가 formal language를 학습할 수 있는 능력에 대한 새로운 실험을 실시합니다. 특히, n-gram language models(NLMs)와 관련하여 transformers의 성능을 비교 분석하고 있습니다.

- **Technical Details**: 연구에서는 transformers가 두 가지 유형의 random n-gram LMs를 학습하는 능력을 평가합니다: (1) 임의의 다음 기호 확률을 가진 LM과 (2) 공유 파라미터로 정의된 LM. 이 과정에서 classic estimation 기술인 add-λ smoothing이 transformers보다 우수하다는 것을 발견했습니다.

- **Performance Highlights**: transformers는 representation-based n-gram LMs에서 우수한 일반화 능력을 보여주며, 카운트 기반 기술과 핸드 크래프트된 피처를 사용한 모델보다 더 나은 성능을 보입니다. 그러나 arbitrary next-symbol probabilities를 가진 n-gram LMs에서는 카운트 기반 기술에 비해 성능이 떨어진다고 언급합니다.



### Coal Mining Question Answering with LLMs (https://arxiv.org/abs/2410.02959)
- **What's New**: 이번 논문은 맞춤형 프롬프트 엔지니어링 기법을 결합한 대형 언어 모델(LLMs)을 활용한 석탄 채굴 질문 응답(QA) 시스템의 혁신적인 접근 방식을 제시합니다. 이 방법은 복잡한 질의를 구조화된 구성 요소로 분해하여 LLM이 기술 정보를 더 효과적으로 처리할 수 있도록 돕습니다.

- **Technical Details**: 저자들은 GPT-4와 같은 LLM을 이용해 500개의 실제 석탄 채굴 시나리오에서 수집한 질문들로 구성된 데이터셋을 평가했습니다. 이 데이터셋은 안전, 운영, 환경적 측면에 초점을 맞춘 질문을 포함하고 있으며, 모델의 성능은 정확도(ACC)와 GPT-4 기반의 점수 매트릭스를 사용하여 측정되었습니다.

- **Performance Highlights**: 전통적인 QA 시스템과 비교할 때, 제안된 프롬프트-엔지니어링 접근 방식은 석탄 채굴 관련 질의에 대해 15-18%의 평균 정확도 향상을 보여주며, GPT-4의 점수도 유의미하게 증가했습니다. 이는 특정 산업 환경에서 LLM이 갖는 잠재력을 강조합니다.



### Unlocking Structured Thinking in Language Models with Cognitive prompting (https://arxiv.org/abs/2410.02953)
Comments:
          11 pages, submitted to ICLR 2025

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)의 문제 해결을 안내하기 위한 새로운 접근법인 인지 프롬프트(cognitive prompting)를 제안합니다. 인지 프롬프트는 목표 명확화, 분해, 필터링, 추상화 및 패턴 인식과 같은 구조화된 인지 작업(cognitive operations)을 통해 다단계 문제를 효과적으로 처리할 수 있도록 지원합니다.

- **Technical Details**: 인지 프롬프트는 문제 해결을 구조화된 인간 유사 인지 단계로 조직합니다. 이 프레임워크는 Chain of Thought (CoT) 방법과 달리 더욱 일반적인 다차원 작업 깊이를 제공하여, LLM이 다양한 문제를 더욱 체계적으로 접근할 수 있도록 돕습니다. 인지 프롬프트는 목표 설정, 문제 분해, 중요한 정보 선택, 데이터 재배열 등의 주요 단계를 포함합니다.

- **Performance Highlights**: Meta의 LLaMA 모델을 이용한 실험에서 인지 프롬프트를 적용했을 때, 성능이 유의미하게 향상되었습니다. 특히, 반영적 인지 프롬프트(reflective cognitive prompting)가 적용된 경우에는 더 나은 다단계 추론 능력을 보여주었습니다. 연구 결과에 따르면, 인지 프롬프트는 LLaMA3.1 70B와 같은 큰 모델에서 성능을 크게 개선하고, 복잡한 문제를 더욱 효과적으로 처리할 수 있도록 해줍니다.



### Visual Editing with LLM-based Tool Chaining: An Efficient Distillation Approach for Real-Time Applications (https://arxiv.org/abs/2410.02952)
Comments:
          EMNLP 2024

- **What's New**: 본 논문에서는 실시간 애플리케이션에서 도구를 호출하기 위해 LLMs(대형 언어 모델)를 미세 조정하는 실용적인 증류(distillation) 접근 방식을 제시합니다. 특히 영상 편집 작업에서 사용자의 스타일적 요청을 자연어로 해석하고 적절한 도구와 파라미터를 선택하여 원하는 시각적 효과를 얻는 방법에 대해 설명합니다.

- **Technical Details**: 우리는 (작은) 학생 LLM을 (큰) 교사 LLM의 가이드와 행동 신호를 기반으로 미세 조정하는 방법을 도입했습니다. 학생 모델의 성능 평가는 도구 및 파라미터 선택과 관련된 오프라인 메트릭스를 사용하여 수행합니다. 데이터 증가 기법을 통해 낮은 데이터 환경에서도 25% 성능 향상을 보여주었습니다.

- **Performance Highlights**: 실험 결과, 우리 학생 모델은 교사 모델(GPT-3.5-Turbo)의 성능을 성공적으로 맞추었으며, 비용과 지연 시간을 크게 줄이며 산업 애플리케이션에 적합한 해결책을 제시했습니다.



### Graph-tree Fusion Model with Bidirectional Information Propagation for Long Document Classification (https://arxiv.org/abs/2410.02930)
Comments:
          accepted to EMNLP findings 2024

- **What's New**: 이 논문에서는 긴 문서 분류를 위한 새로운 그래프-트리 구조를 제안합니다. 이 구조는 문장 인코딩에 대한 문법 트리와 문서 인코딩에 대한 문서 그래프를 통합하여, 국소적(local) 및 전역적(global) 종속성을 모두 캡처할 수 있도록 합니다.

- **Technical Details**: 제안된 모델은 문장의 문법 구조를 나타내는 구문 트리(syntax tree)와 문서의 위계적 관계를 보존하는 문서 그래프(document graph)를 사용합니다. 문장 인코딩을 포함하기 위해 Tree Transformers를 활용하고, intra- 및 inter-문장 종속성을 모델링하기 위해 Graph Attention Network(GAT)를 적용합니다. 또한 훈련 과정에서 단어-문장-문서 간의 양방향 정보 전파(bidirectional information propagation)를 구현하여 문서의 맥락 표현을 풍부하게 합니다.

- **Performance Highlights**: 제안된 방법은 긴 문서 분류 작업에 있어서 모든 유형의 분류 작업에서 효과성을 입증하였으며, 특히 이진 분류(binary classification), 다중 클래스(multi-class classification) 및 다중 레이블(multi-label classification) 문제에서 성능 향상을 보여줍니다.



### NNetscape Navigator: Complex Demonstrations for Web Agents Without a Demonstrator (https://arxiv.org/abs/2410.02907)
Comments:
          Preprint. Under Review

- **What's New**: 본 논문에서는 NNetnav(NNetscape Navigator)라는 방법을 소개합니다. 이 방법은 웹 에이전트를 합성 (synthetic) 시연을 통해 완전히 훈련시키는 방식으로, 브라우저와 상호작용하여 트레일 (trajectory)을 생성하고, 이를 언어 모델을 사용하여 지시사항으로 레이블링합니다. 이는 고비용의 인적 감독 없이 웹 에이전트를 훈련시키기 위한 것입니다.

- **Technical Details**: NNetnav는 탐색을 효율적으로 하기 위해 언어 지시사항의 계층적 구조를 활용합니다. 복잡한 지시사항은 간단한 하위 작업으로 분해될 수 있는데, 중간 트레일이 유의미한 하위 작업으로 주석을 달 수 없는 경우 탐색 에피소드를 자동으로 잘라냅니다. 이를 통해 지시사항을 개선하고 감소시켜 보다 의미 있는 상호작용을 생성합니다. 사용된 기반 언어 모델은 GPT-4o-mini이며, Llama-3-8B-Instruct를 통해 fine-tuning을 진행합니다.

- **Performance Highlights**: NNetnav를 통해 수집한 시연을 사용하여 MiniWoB++와 WebArena에서 소규모 언어 모델 정책의 성능을 개선했습니다. MiniWoB++에서는 28%에서 48%로, WebArena에서는 1%에서 7%로 성능이 개선되었습니다. 특히, 같은 언어 모델에서 유래한 NNetnav 시연으로 미세 조정했을 때 언어 모델 정책의 성능이 더욱 향상되는 것을 관찰했습니다. 최종적으로, 6천 개 이상의 NNetnav 시연 데이터셋을 WebArena에서 수집 및 공개했습니다.



### Better Instruction-Following Through Minimum Bayes Risk (https://arxiv.org/abs/2410.02902)
Comments:
          Under review at ICLR 2025

- **What's New**: 이 연구는人 수준의 평가를 제공할 수 있는 일반용 대형 언어 모델(LLM) 평가자의 사용을 탐구하며, Minimum Bayes Risk (MBR) 디코딩을 통해 LLM의 성능을 향상시킬 수 있는 가능성을 제시합니다.

- **Technical Details**: MBR 디코딩은 레퍼런스 기반 평가를 통해 후보 출력에서 고품질 출력을 선택하는 방법으로, 이 연구에서는 이를 LLM 판별자와 함께 응용하여 테스트 시간 성능 향상을 목적으로 합니다. 이 방법은 레퍼런스가 없는 경우에도 LLM 판별자가 다른 후보 출력을 이용하여 평가를 수행하도록 하여, 평균 점수가 가장 높은 후보 출력을 최종 출력으로 선택합니다.

- **Performance Highlights**: MBR 디코딩을 사용한 LLM 판별자 Prometheus-2-7B는 AlpacaEval에서 +3.6%의 성능 향상을 보였으며, Llama2-7b는 Llama2-13b보다 더 나은 결과를 보여주었습니다. 또한, 자체 학습을 통해 최종 모델이 MBR 디코딩으로 훈련된 기본 모델의 성능을 초과할 수 있음을 보여주었습니다.



### FactCheckmate: Preemptively Detecting and Mitigating Hallucinations in LMs (https://arxiv.org/abs/2410.02899)
- **What's New**: 이 연구는 언어 모델(LMs)의 내부 표현이 hallucination(환상)을 감지하고 완화하는 데 유용하다는 것을 보여주는 새로운 접근 방식을 제시합니다. FactCheckMate라는 시스템을 통해, LM의 hidden states(숨겨진 상태)를 기반으로 환상의 발생 여부를 예측하는 분류기를 학습합니다.

- **Technical Details**: FactCheckMate는 언어 모델의 hidden states를 분석하여 hallucinations를 조기에 감지하고, 감지된 환상을 완화하기 위해 LM의 hidden states를 조정하여 보다 사실적인 출력을 생성합니다. 이 방법은 lightweight하며, inference overhead(추론 오버헤드)가 적습니다.

- **Performance Highlights**: FactCheckMate는 총 70% 이상의 조기 감지 정확도를 달성하며, 개입이 이루어진 LM이 생성한 출력은 개입이 없는 경우보다 평균 34.4% 더 사실적입니다. FactCheckMate에 의해 추가되는 평균 추론 시간 오버헤드는 약 3.16초입니다.



### Computational Modeling of Artistic Inspiration: A Framework for Predicting Aesthetic Preferences in Lyrical Lines Using Linguistic and Stylistic Features (https://arxiv.org/abs/2410.02881)
- **What's New**: 본 연구는 예술적 영감을 모델링하는 새로운 프레임워크를 제안하여 다양한 개인의 예술적 선호도를 계산적으로 분석합니다. 특히, 	extit{EvocativeLines}라는 주석이 달린 가사 데이터셋을 소개하며, 이 데이터셋은 가사가 "영감을 주는" 것과 "영감을 주지 않는" 것으로 분류됩니다.

- **Technical Details**: 제안된 프레임워크는 AI가 생성한 시적 라인의 주요 언어적 및 시적 특성을 식별하고 형식화하여 예술가의 주관적 선호를 설명합니다. 이 모델은 예술가의 기술 수준에 관계없이 적용 가능하며, 말 그대로의 판별 기준이 없는 수준에서 작업을 수행할 수 있습니다. 프레임워크는 3025개의 시적 라인을 평가하여 신뢰성을 입증합니다.

- **Performance Highlights**: 본 연구에서 제안한 프레임워크는 최신 언어 모델인 LLaMA-3-70b보다 약 18점 높은 성능을 보이며, 예술 선호도 예측에 있어서의 우수성을 입증했습니다. 이로 인해 다양한 예술적 선호를 모델링하는 데 적합한 프레임워크로 자리잡을 수 있습니다.



### Position: LLM Unlearning Benchmarks are Weak Measures of Progress (https://arxiv.org/abs/2410.02879)
- **What's New**: 이 논문은 LLM(Large Language Models)에서의 unlearning(학습 데이터 삭제) 방법의 효과성을 평가하기 위한 기존 벤치마크가 과도하게 낙관적이며 오해의 소지가 있음을 지적합니다. 저자들은 여러 인기 있는 벤치마크에 간단한 수정을 가하여, 실제로 '잊혀진' 데이터가 여전히 접근 가능하거나, unlearning 과정이 지켜야 할 지식의 성능을 훨씬 더 악화시킬 수 있음을 드러냅니다.

- **Technical Details**: 이 연구는 LLM의 unlearning을 평가하기 위한 일반적인 평가 쿼리의 의존성 문제와 평가 쿼리 세트를 과도하게 적합시키는 경향을 식별합니다. 특히, 벤치마크 데이터에 대한 간단한 수정이 unlearning 성능에 극적인 영향을 미칠 수 있으며, 이러한 수정에도 불구하고 모델의 동작이 성공적으로 unlearning되었다는 주장과 상충할 수 있음을 보여줍니다.

- **Performance Highlights**: 연구 결과, 기존의 LLM unlearning 벤치마크는 진행상의 한계를 잘 드러내지 못하며, 특히 특정 쿼리와의 의존성을 갖게 되는 수정이 unlearning 알고리즘의 성능에 부정적인 영향을 미칠 수 있다는 점에서 수준 낮은 평가 기준으로 작용할 수 있습니다. 이에 따라 연구자들은 결과 해석 시 신중해야 하며, practical use cases에 보다 맞춤화된 정의와 지표를 설계해야 한다고 합니다.



### Ingest-And-Ground: Dispelling Hallucinations from Continually-Pretrained LLMs with RAG (https://arxiv.org/abs/2410.02825)
- **What's New**: 본 논문은 LLM(대형 언어 모델)와 RAG(검색 보강 생성)를 활용해 개인 정보 보호 프로세스를 효율적으로 개선할 수 있는 새로운 방법들을 제시합니다.

- **Technical Details**: 본 연구에서는 Llama-3.1 모델을 기반으로 지속적인 사전 학습을 통해 개인 정보 보호 지식 기반을 흡수하고, 지식 기반에서 관련 정보를 검색하여 LLM의 대답을 향상시키는 RAG층을 추가합니다.

- **Performance Highlights**: 평가 결과, 지속적인 사전 학습과 RAG 시스템을 통해 LLM의 응답 품질이 평균 40% 향상되었습니다.



### Unraveling Cross-Modality Knowledge Conflict in Large Vision-Language Models (https://arxiv.org/abs/2410.03659)
Comments:
          Website: this https URL

- **What's New**: 본 논문에서는 LVLMs(대형 비전-언어 모델)에서의 교차 모달 파라메트릭 지식 충돌(cross-modality parametric knowledge conflict) 문제를 정의하고, 이를 감지하고 해석하며 완화하기 위한 전반적인 접근 방법을 소개합니다.

- **Technical Details**: 교차 모달 파라메트릭 지식 충돌 문제를 체계적으로 분석하고, 여러 가지 모델에서 시각적인 답변과 문자적 답변 간의 충돌을 감지하는 파이프라인을 제안합니다. 저자들은 동적 대조 디코딩(dynamic contrastive decoding) 방법을 개발하여 신뢰도가 낮은 모달리티 구성 요소에서 추론한 불필요한 로그잇(logits)을 제거합니다.

- **Performance Highlights**: LLaVA-34B를 사용하여 제안된 동적 대조 디코딩 방법이 ViQuAE와 InfoSeek 데이터셋에서 평균 2.24% 향상된 정확도를 기록했습니다.



### What Matters for Model Merging at Scale? (https://arxiv.org/abs/2410.03617)
Comments:
          20 Pages, 7 Figures, 4 Tables

- **What's New**: 이번 논문에서는 모델 병합(model merging)의 확장성과 관련된 다양한 요소들의 영향을 체계적으로 평가하고, 큰 모델을 기반으로 한 병합의 효용성을 분석합니다.

- **Technical Details**: 1B부터 64B까지의 다양한 모델 크기를 사용하여 4가지 인기 있는 병합 방법인 Averaging, Task Arithmetic, Dare, TIES를 실험합니다. 미세 조정된 모델을 병합하고, 각 병합 모델을 익숙한 작업(held-in tasks)과 전혀 보지 못한 작업(zero-shot tasks)에서 평가합니다.

- **Performance Highlights**: 모델 병합은 강력한 기본 모델(base model)을 사용할 때 더욱 효과적이며, 더 큰 모델일수록 병합을 용이하게 하고 일반화 능력을 지속적으로 향상시키는 것으로 나타났습니다. 8개의 대형 전문가 모델을 병합했을 때, 멀티태스킹(multi-task) 훈련 모델보다 더 나은 일반화 성능을 보였습니다.



### TICKing All the Boxes: Generated Checklists Improve LLM Evaluation and Generation (https://arxiv.org/abs/2410.03608)
- **What's New**: TICK (Targeted Instruct-evaluation with ChecKlists)이라는 새로운 자동화된 평가 프로토콜을 제안하여 LLM의 instruction-following 능력을 해석 가능하고 유연하게 평가할 수 있게 되었습니다. 이 프로토콜은 LLM이 생성한 instruction-specific checklist를 통해 평가를 구조화합니다.

- **Technical Details**: TICK은 주어진 instruction에 대해 LLM이 YES/NO 질문으로 구성된 맞춤형 평가 체크리스트를 신뢰성 높게 생성할 수 있다는 것을 보여줍니다. 이 체크리스트는 각 후보 응답이 instruction의 특정 요구 사항을 충족하는지를 평가합니다. TICK을 사용하면 LLM의 판단과 인간의 선호 간의 정확한 일치 비율이 약 5.8% 증가합니다.

- **Performance Highlights**: 이 연구에서 제안된 STICK(Self-TICK)은 여러 벤치마크에서 자기 개선을 통해 세부 성능 향상을 이루었습니다. LiveBench에서 Command-R+는 3.8% 개선을 달성하고, WildBench에서 Best-of-N 선택 방식으로 5.3%의 성능 향상을 보여주었습니다. 또한 LLM이 생성한 체크리스트를 인간 평가자에게 제공하여 평가자 간의 일치도를 0.194에서 0.256으로 증가시켰습니다.



### Understanding Reasoning in Chain-of-Thought from the Hopfieldian View (https://arxiv.org/abs/2410.03595)
Comments:
          28 pages, a new version of "A Hopfieldian View-based Interpretation for Chain-of-Thought Reasoning"

- **What's New**: 이 연구에서는 Chain-of-Thought (CoT) 추론의 기초에 있는 요인을 설명하기 위한 새로운 관점을 제시합니다. Hopfieldian 인지 이론을 바탕으로 CoT 추론과 주요 인지 요소 간의 관계를 확립하여 CoT의 성공 요인을 이해하고자 합니다.

- **Technical Details**: CoT 추론을 이해하기 위해 신경 과학에서의 Hopfieldian 관점을 이용합니다. CoT는 자극, 행동, 신경 집단, 표현 공간과 같은 인지 요소와 맥락을 두고 연결되며, CoT를 통해 발생하는 추론 과정은 이러한 표현 공간 간의 이동으로 설명됩니다. 새로운 프레임워크 'Representation-of-Thought (RoT)'를 제안하여 CoT의 견고성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, RoT는 CoT 추론의 견고성과 해석 가능성을 개선하며, 추론 과정에 대한 세밀한 제어를 제공함을 보여주었습니다. 다양한 과제(산술, 상식, 기호적 추론)에 대한 포괄적인 실험이 실시되었고, CoT 추론의 오류를 추적하고 통제할 수 있는 직관적이고 해석 가능한 분석이 가능함을 발견했습니다.



### No Need to Talk: Asynchronous Mixture of Language Models (https://arxiv.org/abs/2410.03529)
Comments:
          23 pages

- **What's New**: SmallTalk LM은 비동기적(asynchronous) 언어 모델 혼합 훈련 방식으로, 모델 각각이 데이터 분포의 다른 부분에 전문화되어 있으며, 고대역폭 통신 없이도 훈련을 수행할 수 있습니다. 적은 수의 매개변수를 사용하여 추론(inference)에서 경량 경로 설정 라우터가 시퀀스의 적합한 전문가(expert)를 선택하는 방식입니다.

- **Technical Details**: SmallTalk LM은 각 모델이 독립적으로 훈련되고, 이들 모델 간의 통신 비용을 크게 줄입니다. 라우터는 각 전문가의 크기 대비 1.5% 미만의 크기를 가지며, 짧은 접두사를 기반으로 시퀀스의 가장 적합한 전문가를 선택합니다.

- **Performance Highlights**: 실험 결과, SmallTalk LM은 같은 훈련 FLOPs의 조밀한 모델(baseline)들에 비해 유의미하게 낮은 perplexity를 달성했으며, 같은 양의 훈련 데이터를 사용해도 대다수의 하부 작업(overall downstream tasks)에서 그 기준 모델보다 75%의 작업에서 더 나은 성능을 보였습니다.



### On Uncertainty In Natural Language Processing (https://arxiv.org/abs/2410.03446)
Comments:
          PhD thesis

- **What's New**: 이번 논문은 자연어 처리(Natural Language Processing)에서의 불확실성(uncertainty) 문제를 언어학적, 통계적(statistical) 및 신경망(neural) 관점에서 분석하고, 실험 프로세스의 설계를 통해 이를 어떻게 감소시키고 정량화할 수 있는지를 연구합니다.

- **Technical Details**: 연구에서는 텍스트 분류(text classification) 작업에서 유도 모델 편향(inductive model biases)의 영향을 이론적으로 및 실증적으로 분석하고, 덴마크어(Danish), 영어(English), 핀란드어(Finnish) 데이터를 포함한 세 가지 언어에 대한 실험을 실시합니다. 또한, 비교불가능한(conformal prediction) 방식에 기반하여 자연어 생성에서 보정된 샘플링(calibrated sampling) 방법을 제안하며 이는 실제 연속성의 더 나은 범위를 가진 더 타이트한 토큰 세트를 제공합니다.

- **Performance Highlights**: 대규모 블랙박스 언어 모델(large black-box language models)의 신뢰도를 양측 예측(auxiliary predictors)을 사용하여 정량화할 수 있는 접근법을 개발하며, 이는 target 모델의 입력과 생성된 출력 텍스트만으로 신뢰도를 예측합니다.



### Images Speak Volumes: User-Centric Assessment of Image Generation for Accessible Communication (https://arxiv.org/abs/2410.03430)
Comments:
          To be published at TSAR workshop 2024 (this https URL)

- **What's New**: 이번 연구는 Easy-to-Read (E2R) 텍스트에 최적화된 설명 이미지를 생성하기 위한 텍스트-이미지 생성 모델의 가능성을 조사했습니다. 연구진은 7개의 텍스트-이미지 모델을 벤치마킹하고, 사용자 그룹을 대상으로 한 연구를 통해 생성된 이미지가 E2R 텍스트에 적합한지를 평가했습니다.

- **Technical Details**: 연구진은 4개의 오픈소스 모델과 3개의 닫힌 소스 모델을 포함하여 총 7개의 텍스트-이미지 생성 모델을 평가했습니다. 생성된 이미지는 2,217개의 이미지 데이터셋으로 제공되며, 연구진은 560개의 이미지를 평가하여 정확성, 편향성, 목표 집단의 적합성을 기준으로 주석을 달았습니다.

- **Performance Highlights**: 일부 모델은 탁월한 성능을 보였으나, 인간의 감독 없이 대규모로 사용될 준비가 되어있지 않다는 것이 발견되었습니다. 이는 E2R 창작자들이 접근 가능한 정보를 생성하는 데에 중요한 이정표가 됩니다.



### Towards a Benchmark for Large Language Models for Business Process Management Tasks (https://arxiv.org/abs/2410.03255)
- **What's New**: 본 연구는 Business Process Management (BPM) 분야에 적합한 특정 LLM 성능을 평가하기 위한 새로운 벤치마크를 제안합니다. 기존의 LLM 벤치마크가 일반적 작업에 초점을 맞추고 있어 BPM과 같은 구체적인 도메인에서의 LLM 성능을 확인할 필요가 있었습니다.

- **Technical Details**: 연구에서는 네 가지 BPM 작업(활동 추천, RPA 후보 식별, 프로세스 질문 응답, 선언적 프로세스 모델 분석)에 대한 LLM 성능을 체계적으로 비교했습니다. 특히 오픈 소스 모델과 상업적 모델 간의 성능 차이 및 모델의 크기가 BPM 작업 성능에 미치는 영향을 평가했습니다.

- **Performance Highlights**: 이 연구는 다양한 LLM이 어떻게 BPM 작업을 수행하는지를 명확히 정의하고, 적합한 모델 선택에 있어 조직에게 실질적인 통찰을 제공합니다. 분석 결과, 오픈 소스 모델들이 특정 BPM 작업에서 상업적 모델과 유사하거나 뛰어난 성능을 나타낼 수 있다는 점이 밝혀졌습니다.



### How much can we forget about Data Contamination? (https://arxiv.org/abs/2410.03249)
- **What's New**: 이번 연구는 최신의 대형 언어 모델(LLMs)의 평가 시 벤치마크 데이터의 누출로 인한 문제를 다루고 있습니다. 연구자들은 벤치마크 데이터의 경미한 오염이 평가에 미치는 영향에 대해 실험적 증거와 이론적 추정을 제시하여 그러한 오염이 항상 부정적인 결과를 초래하지 않음을 밝혔다.

- **Technical Details**: 연구에서는 세 가지 차원에서 벤치마크 과적합(benchmark overfitting)의 크기를 정량화했습니다: 모델의 파라미터 수(1.6B까지), 예시가 본 데이터에서 보는 횟수(144회까지), 훈련 토큰 수(40B까지). Chinchilla 스케일링 법칙을 따랐을 경우, 오염이 적더라도 과적합을 초래할 수 있음을 보였습니다. 그러나 훈련 데이터를 다섯 배 확장할 경우, 과거의 데이터를 잊을 수 있는 수치적 경계를 제시합니다.

- **Performance Highlights**: 이 연구는 많은 대형 언어 모델들이 훈련 초기의 데이터를 잊고 있다는 것을 확인했습니다. 또한, 모델들이 데이터의 반복 노출에 대해 가장 강한 과적합을 보이는 현상을 통해, 새로운 데이터에 대한 노출이 중요하다는 것을 강조했습니다.



### Showing LLM-Generated Code Selectively Based on Confidence of LLMs (https://arxiv.org/abs/2410.03234)
- **What's New**: 이번 논문에서는 코드 생성에서 LLM(대형 언어 모델)의 신뢰도를 기반으로 선택적으로 생성된 코드를 개발자에게 보여주는 새로운 접근 방식인 HonestCoder를 제안합니다. 이 방법은 LLM이 생성한 코드의 정확성을 보다 효과적으로 측정하고 개발자의 에너지를 절약할 수 있도록 돕습니다.

- **Technical Details**: HonestCoder는 LLM이 생성한 프로그램의 신뢰도를 추정하기 위해 다중 모달 유사성(multi-modal similarity)을 측정하는 새로운 방법을 사용합니다. 이 시스템은 다국어 벤치마크인 TruthCodeBench를 활용하여 2,265개의 샘플을 제공하며, 파이썬과 자바 프로그래밍 언어를 다룹니다. HonestCoder는 기존의 코드 생성 방법들과 비교하여 개발자에게 더 많은 정확한 프로그램을 보여줄 수 있으며, 잘못된 프로그램의 수를 줄입니다.

- **Performance Highlights**: 실험 결과, HonestCoder는 AUROC에서 최신 기법보다 27.79% 향상되었고, AUCPR에서는 63.74%의 성능 개선을 보였습니다. HonestCoder는 약 0.4초의 경미한 시간 오버헤드를 추가하면서도 보다 효율적인 프로그래밍 협업을 가능하게 합니다.



### Frame-Voyager: Learning to Query Frames for Video Large Language Models (https://arxiv.org/abs/2410.03226)
Comments:
          19 pages, 10 figures

- **What's New**: 이번 논문에서는 비디오를 이해하는 작업에서 정보가 밀집된 프레임 조합을 쿼리하는 Frame-Voyager라는 새로운 접근법을 제안합니다. 이 방법은 텍스트 쿼리를 바탕으로 유용한 프레임 조합을 학습합니다.

- **Technical Details**: Frame-Voyager는 사전 학습된 Video-LLM을 사용하여 프레임 조합의 유용성을 기반으로 순위를 매깁니다. 이 방식을 통해 두 개의 주요 도전 과제를 해결하는데, 첫째, 높은 학습 복잡성을 효율적으로 관리하고, 둘째, 라벨링 데이터의 부족 문제를 극복합니다.

- **Performance Highlights**: 실험 결과, Frame-Voyager는 기존의 균일 샘플링 및 텍스트-프레임 검색 방법에 비해 유의미한 성능 향상을 보이며, 특히 복잡한 추론이 필요한 긴 비디오 작업에서 두각을 나타냅니다.



### Can Watermarked LLMs be Identified by Users via Crafted Prompts? (https://arxiv.org/abs/2410.03168)
Comments:
          25 pages, 5 figures, 8 tables

- **What's New**: 이번 연구는 Large Language Models (LLMs)의 워터마크(Watermark) 기술의 눈에 띄지 않음을 처음으로 탐구했습니다. 기존의 기술들은 고감도성과 텍스트 품질에 미치는 영향을 최소화하는 데 중점을 두었으나, 실제 서비스에서 워터마크의 존재를 밝히는 것이 사용자 경험에 미치는 영향을 간과한 점을 지적했습니다.

- **Technical Details**: 새롭게 설계된 알고리즘인 Water-Probe는 LLM에게 최적화된 프롬프트(prompts)를 사용하여 워터마크를 감지합니다. 이 연구에서는 동일한 워터마크 키에 대해 LLM들이 일관된 편향(bias)을 보이며, 이는 다양한 워터마크 키에 따라 유사한 차이를 만들어낸다고 주장합니다. 또한, 워터마크 키 선택의 무작위성을 증가시키는 것이 워터마크의 눈에 띄지 않음을 향상시키는 중요한 열쇠라고 제안합니다.

- **Performance Highlights**: 실험 결과, 기존의 워터마킹 알고리즘들은 잘 설계된 프롬프트에 의해 쉽게 식별되었으며, Water-Probe는 비워터마크 LLM에 대해 낮은 오탐지(false positive rate)를 나타냈습니다. 이 연구는 Water-Bag 전략을 통해 여러 워터마크 키를 결합함으로써 워터마크의 눈에 띄지 않음을 크게 향상시킬 수 있음을 보여줍니다.



### In-context Learning in Presence of Spurious Correlations (https://arxiv.org/abs/2410.03140)
- **What's New**: 대형 언어 모델이 spurious features가 포함된 분류 과제에서 in-context learning (ICL) 능력을 보여주는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 기존의 ICL 학습 방식이 spurious features에 취약하고 단일 작업의 경우 작업 기억화(task memorization)를 초래할 수 있음을 발견했습니다. 새로운 ICL 훈련 접근 방식은 입력 임베딩 차원을 무작위로 섞어주고, spurious features에 맞춰 훈련 데이터를 형성하는 방식을 포함합니다.

- **Performance Highlights**: 제안된 방법은 기존 알고리즘인 1-NN, ERM, GroupDRO보다 우수한 성과를 보이며, 다양한 binary classification 작업에서 unseen tasks에 대한 일반화를 이뤘습니다. 그러나 spurious features가 있는 unseen tasks에서는 일반화 능력이 떨어진다고 밝혔습니다.



### AIME: AI System Optimization via Multiple LLM Evaluators (https://arxiv.org/abs/2410.03131)
Comments:
          21 pages, 10 Figures, 4 Tables

- **What's New**: AI 시스템 최적화에서 단일 LLM 평가자 사용의 한계를 강조하고, 여러 LLM 평가자를 사용하는 AIME 프로토콜을 제안합니다. 이를 통해 복잡한 코드 생성 작업에서 높은 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: AIME (AI system optimization via Multiple Evaluators) 프로토콜은 서로 다른 기준에 대해 독립적으로 평가를 생성하는 여러 LLM을 사용하여 각 평가를 합치는 방식으로 작동합니다. 이 방법은 오류 감지율을 62%까지 향상시키고, 성공률을 16% 증가시킵니다.

- **Performance Highlights**: AIME는 LeetCodeHard와 HumanEval 벤치마크에서 단일 평가 프로토콜에 비해 최대 62% 높은 오류 감지율을 보이며, 테스트 케이스의 성공률은 최대 16% 더 높았습니다. 또한 평가자의 수와 기준 선택이 성공률에 영향을 미친다는 점을 강조합니다.



### ARB-LLM: Alternating Refined Binarizations for Large Language Models (https://arxiv.org/abs/2410.03129)
Comments:
          The code and models will be available at this https URL

- **What's New**: 본 논문은 ARB-LLM이라는 새로운 1-bit 포스트 트레이닝 양자화(PTQ) 기술을 제안합니다. 이는 대형 언어 모델(LLM)에 최적화되었으며, 이 모델의 메모리 및 계산 요구량을 크게 줄일 수 있습니다.

- **Technical Details**: ARB-LLM은 교차 정제 양자화(Alternating Refined Binarization, ARB) 알고리즘을 기반으로 하여, 양자화 오차를 줄이고, 컬럼 그룹 비트맵(Column-Group Bitmap, CGB) 전략을 개선하여 성능을 향상시킵니다. ARB-X 및 ARB-RC와 같은 확장 기술로, 교정 데이터를 통합하고 컬럼 방향의 편차를 최소화합니다.

- **Performance Highlights**: 실험 결과, ARB-LLM$_{RC}$는 현재의 SOTA 이진 PTQ 방법들보다 훨씬 높은 성능을 보이며, 동일한 크기의 FP16 모델을 초월하는 성과를 거두었습니다. 또한, 이 알고리즘은 대규모 LLM의 실용적인 배포에 필요한 메모리 자원을 최소화합니다.



### ProcBench: Benchmark for Multi-Step Reasoning and Following Procedur (https://arxiv.org/abs/2410.03117)
- **What's New**: 이 논문에서는 ProcBench라는 새로운 벤치마크를 제시하여 다단계 추론(multi-step inference)에 대한 직접 평가를 중점적으로 다루고 있습니다. 기존의 벤치마크와는 달리, ProcBench는 복잡한 지식 없이 제공된 지침을 따르는 능력을 평가합니다.

- **Technical Details**: ProcBench는 여러 가지 간단한 작업을 포함하며, 각 작업은 명확히 정의된 지침을 따르는 과정을 요구합니다. 연구에서 사용된 데이터셋은 명시적인 지침과 해당 질문의 쌍으로 구성되어 있으며, 각 단계에서 모델이 지침을 따르는 능력을 평가합니다.

- **Performance Highlights**: 최신 대형 언어 모델(LLMs)에 대한 평가 결과, 모델에 따라 성능 차이가 있었습니다. o1-preview와 o1-mini와 같은 일부 모델은 간단한 작업에서 높은 정확도를 보였지만, 복잡성이 증가함에 따라 성능이 크게 저하되는 한계를 드러냈습니다.



### LoRC: Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy (https://arxiv.org/abs/2410.03111)
Comments:
          15 pages, 4 figures

- **What's New**: 이 논문에서는 Key-Value (KV) 캐시 효율성을 높이기 위해 기존 Transformer 기반의 대규모 언어 모델 (LLMs)에 직접 적용할 수 있는 저랭크( low-rank) 근사의 새로운 접근 방식을 제안합니다. 이는 모델 재훈련 없이 사용할 수 있으며, KV 캐시의 메모리 소비를 효과적으로 줄이는 데 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 방법은 KV 가중치 행렬의 저랭크 근사를 통해 적용됩니다. 레이어별 민감성을 반영한 점진적인 압축 전략을 도입하여, 깊은 네트워크에서의 오류 전파를 이론적으로 분석하고, 각 레이어의 압축 오류 경계를 도출합니다. 이로 인해, 초기 레이어에서 발생한 오류가 심화된 레이어보다 더 크게 증가하는 경향이 있습니다.

- **Performance Highlights**: 8B, 13B, 70B 파라미터를 가진 LLaMA 모델에서 다양한 작업을 통해 실험했으며, 이 방법이 GPU 메모리 소모를 크게 줄이면서도 성능에는 미미한 영향을 미친다는 것을 입증했습니다.



### Mamba in Vision: A Comprehensive Survey of Techniques and Applications (https://arxiv.org/abs/2410.03105)
Comments:
          Under Review

- **What's New**: Mamba는 컴퓨터 비전 내에서 Convolutional Neural Networks (CNNs)와 Vision Transformers (ViTs)의 한계를 극복하기 위한 새로운 접근 방식으로 등장하였습니다. Mamba는 또는 선형 계산 복잡도를 기반으로 한 Selective Structured State Space Models를 활용하여 장거리 의존성을 효과적으로 캡처하는 데 중점을 두었습니다.

- **Technical Details**: Mamba 모델은 입력 데이터에 기반하여 동적으로 조정되는 선택적 상태 표현을 사용하여 computational overhead를 줄이고 효율성을 높입니다. 이는 구조적 상태 공간 모델(SSMs)의 발전을 기반으로 하여 이루어지며, Mamba는 GPU를 최적화한 스캔 기반 알고리즘을 활용하여 기존의 convolution 기반 SSMs의 비효율성을 피합니다.

- **Performance Highlights**: Mamba 모델은 비디오 처리, 원격 감지, 의료 영상 등 다양한 분야에서 특히 유리하며, CNNs와 ViTs는 높은 계산 요구로 인해 확장성 문제를 겪는 반면, Mamba 모델은 시퀀스 길이에 대한 선형 확장성을 제공하여 실시간 및 대규모 애플리케이션에 적합합니다.



### Horizon-Length Prediction: Advancing Fill-in-the-Middle Capabilities for Code Generation with Lookahead Planning (https://arxiv.org/abs/2410.03103)
- **What's New**: 이 논문은 코드 완성을 위한 Fill-in-the-Middle (FIM) 훈련의 새로운 접근법인 Horizon-Length Prediction (HLP)을 제안합니다. HLP는 모델이 중간 토큰 수를 예측할 수 있도록 가르쳐 코드 생성 문제의 성능을 향상시킵니다.

- **Technical Details**: HLP는 기존의 next-token prediction (NTP) 방식과 상호 보완적으로 작동하며, 각 훈련 단계에서 남은 중간 토큰 수를 예측하도록 모델을 훈련시켜 텍스트의 자연스러운 흐름을 유지합니다. 논문에서는 이 접근법이 기존의 rule-based post-processing 방법과는 달리 dataset-specific한 가정에 의존하지 않음을 강조합니다.

- **Performance Highlights**: HLP를 적용한 모델은 다양한 벤치마크에서 최대 24%의 상대적 성능 향상을 보여주었으며, 이는 파일 및 레포지토리 수준의 코드 추론에도 긍정적인 영향을 미쳤습니다. 또한 HLP는 훈련 시간에 거의 추가적인 비용을 들이지 않습니다.



### DocKD: Knowledge Distillation from LLMs for Open-World Document Understanding Models (https://arxiv.org/abs/2410.03061)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 이 논문에서는 Visual Document Understanding (VDU) 모델의 일반화 능력을 향상시키는 새로운 프레임워크인 DocKD를 제안합니다. 이 프레임워크는 LLM(large language models)에서의 지식을 증류합니다.

- **Technical Details**: DocKD는 LLM에 키-값 쌍(key-value pairs), 레이아웃(layouts), 설명(descriptions) 등의 다양한 문서 요소를 제공하여 열린 질문에 대한 답변을 유도합니다. 기존의 방법과 달리, 외부 문서 지식을 통합하는 데이터 생성 과정을 풍부하게 합니다.

- **Performance Highlights**: DocKD를 통해 생성된 문서 주석들은 고품질이며, 전통적인 지식 증류 접근법보다 우수한 성능을 보입니다. 특히, DocKD로 훈련된 학생 VDU 모델들은 인간 주석 데이터로 훈련된 모델과 비교할 때, 인도메인(task)에서는 유사한 성능을 유지하나, 아웃오브 도메인(out-of-domain task)에서는 훨씬 더 뛰어난 성능을 발휘합니다.



### MLP-KAN: Unifying Deep Representation and Function Learning (https://arxiv.org/abs/2410.03027)
- **What's New**: 최근 연구에서 MLP-KAN을 소개하며, 이를 통해 representation learning과 function learning의 통합을 다루고 있습니다. MLP-KAN은 Mixture-of-Experts(MoE) 아키텍처를 기반으로 하여 데이터를 효과적으로 처리할 수 있습니다.

- **Technical Details**: MLP-KAN은 Multi-Layer Perceptrons(MLP)과 Kolmogorov-Arnold Networks(KAN)을 혼합하여 사용합니다. MoE 메커니즘이 동적으로 입력을 적절한 전문가에게 라우팅하여 다양한 작업에서 성능을 극대화하였습니다. 이 모델은 transformer 기반 아키텍처에서 구성되어 있으며, 제시된 표준 데이터셋에서 우수한 결과를 보여주었습니다.

- **Performance Highlights**: MLP-KAN은 이미지 인식, 자연어 처리 등 여러 분야에서 기존 모델들과 비교해 우수한 성능을 보이며, 특히 representation learning에서 높은 정확도와 function learning에서 낮은 RMSE를 기록했습니다.



### FastAdaSP: Multitask-Adapted Efficient Inference for Large Speech Language Mod (https://arxiv.org/abs/2410.03007)
Comments:
          EMNLP 2024 Industry Track

- **What's New**: 본 연구에서는 Multitask Speech Language Model (SpeechLM)의 효율적인 추론을 위해 Token Reduction을 탐구합니다. FastAdaSP라는 새로운 프레임워크를 제안하여 음성 관련 다양한 작업에 대한 효율성과 성능 간의 균형을 개선합니다.

- **Technical Details**: FastAdaSP는 오디오 토큰 감소 방법을 포함하여 밀집(Dense) 및 희소(Sparse) 작업에 적합한 고속 추론을 가능하게 하는 통합된 프레임워크입니다. 본 연구는 오디오 특징의 효율적인 처리를 위한 레이어 선택 및 작업별 설계를 포함합니다.

- **Performance Highlights**: FastAdaSP는 WavLLM 및 Qwen-Audio 실험에서 7배의 메모리 효율성과 1.83배의 디코딩 처리량을 달성했으며, Emotion Recognition (ER) 및 Spoken Question Answering (SQA)와 같은 작업에서 성능 저하 없이 이러한 효율성을 이끌어냈습니다.



### Guided Stream of Search: Learning to Better Search with Language Models via Optimal Path Guidanc (https://arxiv.org/abs/2410.02992)
- **What's New**: 이번 연구에서 우리는 언어 모델의 검색(search) 및 계획(planning) 능력을 향상시키기 위해 최적의 솔루션(optimal solution)을 활용하는 방법을 탐구합니다. 이에 따라 우리는 guided stream of search (GSoS)라는 방법을 제안하며, 이는 최적의 솔루션을 탐색 과정에 점진적으로 통합하여 고품질의 검색 경로를 생성합니다.

- **Technical Details**: GSoS는 최적의 솔루션에서 각 중간 작업을 단계별로 통합하는 방식으로 작동합니다. 이 과정에서는 실패한 검색 경로를 맥락(context)으로 사용하며, 이러한 방식을 통해 사전 훈련(pre-trained)된 모델의 성능을 향상시킵니다. Countdown라는 수학적 추론 문제에서 이러한 접근 방식을 평가하였으며, 정확도가 13% 향상되었습니다. 강화 학습(RL) 미세 조정을 동시에 적용할 경우, 이 개선은 20%까지 증가합니다.

- **Performance Highlights**: 우리의 접근 방식은 이전의 감독 기반 미세 조정(supervised fine-tuning) 방법과 비교하여 더 높은 성능을 보여주며, 특히 GSoS를 사용한 RL 미세 조정이 효과적입니다. 이로 인해 모델의 검색 및 계획 능력이 크게 향상되었습니다.



### AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML (https://arxiv.org/abs/2410.02958)
Comments:
          47 pages, 5 figures

- **What's New**: 이번 논문에서는 AutoML-Agent라는 새로운 다중 에이전트 프레임워크를 제안하고 있습니다. 이 프레임워크는 데이터 검색부터 모델 배포까지 전체 AutoML 파이프라인을 지원합니다.

- **Technical Details**: AutoML-Agent는 사용자의 작업 설명을 기반으로 전문화된 LLM 에이전트 간의 협업을 촉진하며, 배포 준비가 완료된 모델을 제공합니다. 기존 연구와 달리 단일 계획을 세우는 대신 검색 증강 계획(retrieval-augmented planning) 전략을 도입하여 최적의 계획을 탐색합니다. 각 계획은 데이터 전처리(data preprocessing) 및 신경망 설계(neural network design)와 같은 하위 작업(sub-tasks)으로 분해되어, 병렬로 실행되는 전문화된 에이전트에 의해 해결됩니다.

- **Performance Highlights**: 14개의 데이터셋을 활용한 7개의 다운스트림(tasks) 실험에서 AutoML-Agent는 전체 AutoML 프로세스를 자동화하는 데 있어 더 높은 성공률을 보여주었으며, 다양한 도메인에서 좋은 성능을 발휘하는 시스템을 제공합니다.



### LLMCO2: Advancing Accurate Carbon Footprint Prediction for LLM Inferences (https://arxiv.org/abs/2410.02950)
Comments:
          9 pages, 11 figures

- **What's New**: 이 논문에서는 LLM(대규모 언어 모델) 추론의 탄소 발자국을 보다 정확하게 예측하기 위한 새로운 모델인 LLMCO2를 소개합니다. 기존의 예측 방법이 불완전함을 보완하여, 요청 구성 및 하드웨어 설정에 따른 탄소 영향을 신속하고 정확하게 추정할 수 있는 도구를 제공합니다.

- **Technical Details**: LLMCO2는 그래프 신경망(GNN) 기반 모델로서, 각 변환기 층의 커널을 그래프로 표현합니다. 각각의 노드는 커널을 나타내고, 엣지는 데이터 의존성을 캡처합니다. 모델은 전처리(prefill)와 디코딩(decode) 단계의 노드 특성을 별도로 인코딩하며, 각 노드의 Roofline 성능을 하드웨어 특성으로 통합합니다. 또한, 일반적인 요청 구성을 주요 변수로 삼아 데이터 샘플링 알고리즘을 개발하였습니다.

- **Performance Highlights**: LLMCO2는 다양한 추론 요청과 GPU 구성을 사용할 때 기존의 ML 기반 에너지 예측자들보다 탄소 발자국 예측 정확도를 51%-123% 개선하였습니다. 이는 LLM의 사용이 증가함에 따라 환경 영향 평가의 필요성을 더욱 부각시킵니다.



### Fine-Tuning Language Models with Differential Privacy through Adaptive Noise Allocation (https://arxiv.org/abs/2410.02912)
Comments:
          EMNLP 2024 findings

- **What's New**: 이 논문에서는 ANADP라는 새로운 알고리즘을 제안하여 언어 모델의 매개변수 중요도에 따라 추가 노이즈를 적응적으로 할당합니다. 이 접근법은 전통적인 Differential Privacy(DP) 방법의 한계를 극복하고 기계 학습 모델의 프라이버시를 강화하는 동시에 성능을 개선합니다.

- **Technical Details**: ANADP는 매개변수의 중요도에 기반하여 노이즈와 프라이버시 예산을 분배하는 방법입니다. 이는 매개변수의 감도(sensitivity)와 불확실성(uncertainty)을 고려하여 모델의 훈련 과정에서 안정적으로 적용됩니다. 기존의 DP 방법과 달리, ANADP는 각 매개변수의 기여도를 평가하여 균일하지 않은 방식으로 프라이버시 예산을 배분합니다.

- **Performance Highlights**: ANADP는 Glue benchmark에서 기존의 DP 방법보다 항상 우수한 성능을 보여주었으며, 전통적 DP와 비-DP 파인튜닝(수정 없이 원본을 유지한 파인튜닝) 간의 성능 격차를 줄이는 데 성공했습니다. 또한 ANADP는 기존의 DP 방법처럼 강력한 프라이버시 보호를 유지합니다.



### Cognitive Biases in Large Language Models for News Recommendation (https://arxiv.org/abs/2410.02897)
Comments:
          Accepted at the ROGEN '24 workshop, co-located with ACM RecSys '24

- **What's New**: 이 논문은 대형 언어 모델(LLM)이 뉴스 추천 시스템에 미치는 인지 편향(cognitive biases)의 영향을 다양한 측면에서 탐구하며, 이러한 편향이 시스템의 신뢰성에 어떤 악영향을 미칠 수 있는지를 분석합니다. 또한 데이터 증강(data augmentation), 프롬프트 엔지니어링(prompt engineering), 학습 알고리즘(learning algorithms) 등의 방법을 통해 이러한 편향을 완화하는 전략을 제시합니다.

- **Technical Details**: 이 논문에서는 인지 편향의 다양한 종류(앙커링 편향(anchoring bias), 프레이밍 편향(framing bias), 현 상태 유지 편향(status quo bias), 집단 귀인 편향(group attribution bias))가 LLM 기반 뉴스 추천 시스템에 미치는 영향을 분석합니다. 이를 통해, LLM이 트레이닝 데이터에서 상속받는 인간의 인지 편향이 결과에 어떻게 반영될 수 있는지를 보여줍니다.

- **Performance Highlights**: LLM 기반 뉴스 추천 시스템의 신뢰성을 높이기 위해 제시된 완화 전략은 다음과 같습니다: 합성 데이터 증강(synthetic data augmentation), 반복 수정(self-debiasing via iterative refinement), 인간 피드백을 통한 인지 편향 수정(cognitive debiasing through human feedback). 이러한 접근 방법들은 추천 시스템이 보다 객관적이고 공정한 출력을 생성하도록 도와줍니다.



### The Role of Deductive and Inductive Reasoning in Large Language Models (https://arxiv.org/abs/2410.02892)
Comments:
          4 figures

- **What's New**: 이 논문에서는 Deductive and InDuctive (DID) 방법을 제안하여 LLM(Large Language Models)의 추론 능력을 향상시키고, 동적으로 추론 경로를 조정할 수 있는 유연한 프레임워크를 제공합니다.

- **Technical Details**: DID 방법은 인지 과학의 원리에 기반하여 유도적(inductive)과 연역적(deductive) 추론 과정을 프롬프트 구성 과정에 통합하여 LLM의 추론 유연성과 적응성을 높입니다. 이 접근법은 다양한 데이터셋에서 검증되었으며, 모델의 성능을 크게 향상시킵니다.

- **Performance Highlights**: DID 방법을 사용한 결과, 기존 저명한 데이터셋인 AIW와 MR-GSM8K 및 자체 제작한 Holiday Puzzle에서 솔루션 정확도와 추론 품질 모두에서 유의미한 향상을 보여주었습니다. 이 모든 개선사항은 substantial computational overhead 없이 이루어졌습니다.



### LLaMA-Berry: Pairwise Optimization for O1-like Olympiad-Level Mathematical Reasoning (https://arxiv.org/abs/2410.02884)
- **What's New**: 본 논문은 LLaMA-Berry라는 고급 수학 문제 해결 프레임워크를 제안하여 대형 언어 모델(LLMs)의 수학적 추론 능력을 향상시키고자 합니다. Monte Carlo Tree Search (MCTS)와 반복적인 Self-Refine을 결합하여 추론 경로를 최적화하고, 쌍간 보상 모델을 활용하여 다양한 경로를 평가합니다.

- **Technical Details**: LLaMA-Berry 프레임워크는 Self-Refine을 Monte Carlo Tree Search에 적용한 SR-MCTS와 쌍간 선호 보상 모델(PPRM)이라는 두 가지 새로운 방법으로 구성됩니다. MCTS는 마르코프 결정 과정(MDP) 프레임워크에서 최적 솔루션 탐색을 진행하고, PPRM은 인간 피드백에서의 강화 학습(RLHF) 기법에서 영감을 받아 해결책 간의 선호 관계를 모델링하고 이를 글로벌 순위 점수로 집계합니다.

- **Performance Highlights**: 이 프레임워크는 GPQA, AIME24, AMC23 등 복잡한 올림피아드 레벨 벤치마크에서 ToT 및 rStar와 같은 기존 방법들보다 우수한 성능을 보여주었습니다. LLaMA-3.1-8B 모델이 추가 훈련 없이도 GPT-4 Turbo와 유사한 수준으로 성능을 향상시켰다는 결과는 LLaMA-Berry 방식이 소규모 데이터만으로도 LLM의 추론 능력을 효과적으로 개선할 수 있음을 시사합니다.



### PyRIT: A Framework for Security Risk Identification and Red Teaming in Generative AI System (https://arxiv.org/abs/2410.02828)
- **What's New**: Generative Artificial Intelligence (GenAI)의 사용이 확산되고 있는 가운데, 새로운 리스크 식별 프레임워크가 필요해졌습니다. 이를 해결하기 위해 Python Risk Identification Toolkit (PyRIT)이라는 오픈 소스 프레임워크가 소개되었습니다.

- **Technical Details**: PyRIT는 모델 및 플랫폼에 독립적인 툴로서, red teamers가 멀티모달 생성 AI 모델에서 새로운 위험과 jailbreaks를 탐색하고 식별하는 데 도움을 줍니다. 이 툴은 Python으로 작성되어 널리 접근 가능하며, 모듈형 구조를 통해 다양한 공격 조합을 쉽게 시도할 수 있도록 설계되었습니다.

- **Performance Highlights**: Microsoft AI Red Team(AIRT)은 PyRIT를 활용하여 100건 이상의 GenAI 모델에 대한 red teaming 작업을 성공적으로 수행했습니다. 이 연구에서는 PyRIT의 개념 증명(Proof-of-Concept) 실험과 실제 사례 연구를 통해 그 실용적인 응용 사례를 시연합니다.



### GPT's Judgements Under Uncertainty (https://arxiv.org/abs/2410.02820)
- **What's New**: 본 연구는 인간의 인지 편향이 GPT-4o의 확률적 판단 및 결정 형성에 어떻게 드러나는지를 1350번의 실험을 통해 조사하였습니다. 이를 통해 비슷한 확률적 표기에 대한 반응 방식에서 AI의 모순적 접근을 보여주었습니다.

- **Technical Details**: 연구에서는 인지 편향으로는 손실 회피, 형태 효과, 결합 오류 등 9개의 편향을 사용하여 1350개의 실험을 진행하였으며, 통계적 추론과 직관적 추론 간의 반응을 분석했습니다. 각 실험은 150번 반복하여 결과의 일관성을 높였습니다.

- **Performance Highlights**: 총 1350개의 실험 중, GPT-4o는 658개의 상세한(elaborate) 응답과 692개의 직관적(intuitive) 응답을 제공하였습니다. 특히 결합 오류에 대한 실험에서는 언제나 상세한 응답을 제공하며, 통계적으로 타당한 이유를 설명했습니다.



### SAC-KG: Exploiting Large Language Models as Skilled Automatic Constructors for Domain Knowledge Graphs (https://arxiv.org/abs/2410.02811)
Comments:
          ACL 2024 Main

- **What's New**: 본 논문에서는 SAC-KG라는 일반적인 지식 그래프(KG) 구축 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLM)을 전문 지식으로 활용하여 자동으로 특화된 다단계 KGs를 생성합니다.

- **Technical Details**: SAC-KG는 세 가지 구성 요소로 이루어져 있습니다: Generator, Verifier, Pruner. Generator는 원시 도메인 코퍼스에서 관련성을 가진 관계와 꼬리 엔티티를 생성하고, Verifier는 오류를 감지하여 수정하며, Pruner는 필요에 따라 다음 단계의 생성을 결정합니다.

- **Performance Highlights**: SAC-KG는 100만 개 이상의 노드 규모로 도메인 KG를 자동으로 구축하며, 89.32%의 정밀도를 기록했습니다. 기존 최첨단 방법들에 비해 20% 이상 향상된 정밀도를 달성했습니다.



### StateAct: State Tracking and Reasoning for Acting and Planning with Large Language Models (https://arxiv.org/abs/2410.02810)
Comments:
          9 pages, 5 pages appendix, 7 figures, 5 tables

- **What's New**: 새로운 연구에서는 대형 언어 모델(LLMs)을 사용하는 '실제' 작업 해결을 위한 계획 및 행동 기술에 대한 간단한 방법인 'StateAct'를 제안합니다. 이 방법은 상태 추적(state-tracking)을 통해 LLM의 'chain-of-thought'를 강화하여 더욱 장기적인 문제 해결이 가능합니다.

- **Technical Details**: StateAct는 few-shot in-context learning을 기반으로 하여 에이전트의 목표(예: 위치 및 인벤토리)를 지속적으로 추적합니다. 새 기술은 Alfworld에서 평가되며, 기존 방법보다 14% 성능 향상을 이루었습니다. 추가적인 데이터나 도구 없이도 성능이 동등한 수준을 유지합니다.

- **Performance Highlights**: StateAct는 여러 LLM에서 효율적으로 작동하며, 작업 해결에 필요한 단계 수를 줄이고 장기적인 문제 해결 능력을 향상시킵니다.  최첨단 성능에 도달한 결과는 LLM 분야에서 중요한 진전을 나타냅니다.



### TaCIE: Enhancing Instruction Comprehension in Large Language Models through Task-Centred Instruction Evolution (https://arxiv.org/abs/2410.02795)
- **What's New**: 연구는 새로운 접근법인 Task-Centered Instruction Evolution (TaCIE)을 소개하며, 이는 기존의 단순한 seed instruction의 진화 방식을 개선하였습니다. TaCIE는 복잡한 지침을 구성 요소로 나눈 다음, 이러한 요소를 조합하여 더욱 복잡하고 다양성이 높은 지침을 생성합니다.

- **Technical Details**: TaCIE는 지침을 배경정보, 목표 및 제약조건으로 분해하여, 각 요소를 세밀하게 수정하고 조합함으로써 진화된 지침을 생성합니다. 이러한 접근법은 LLMs의 difficulty scaling과 cross-domain 적용 가능성을 significantly 향상시킵니다.

- **Performance Highlights**: TaCIE로 fine-tuning된 LLM들이 기존 방법으로 조정된 모델들보다 다양한 벤치마크에서 성능이 현저히 향상되었습니다. 연구 결과는 TaCIE의 우수성을 입증하며, 모델 가중치와 코드가 공개되어 연구 협력을 촉진하고 있습니다.



### Navigation with VLM framework: Go to Any Languag (https://arxiv.org/abs/2410.02787)
Comments:
          under review

- **What's New**: 이번 논문은 Vision Large Language Models (VLMs)를 활용해 인간과 유사한 방식으로 탐색하며 어떤 언어 목표에도 도달할 수 있는 Navigation with VLM (NavVLM)이라는 프레임워크를 소개합니다. 이 프레임워크는 미리 훈련된 모델 없이도 에이전트가 환경 정보를 인식하고 길을 안내받아 목표를 향해 탐색할 수 있도록 합니다.

- **Technical Details**: NavVLM 프레임워크는 에이전트가 특정 또는 비특정 언어 목표를 지향해 탐색할 수 있도록 구성됩니다. 이 시스템은 cognitive core로서 VLM을 사용하여 환경을 인식하고, 목표와 가까워지면 탐색을 종료하고 VLM의 지침에 따라 탐색을 이어갑니다. 중요한 기술 요소로 SLAM (Simultaneous Localization and Mapping) 모듈과 경로 계획 모듈이 있으며, 이를 통해 에이전트는 탐색 중 장애물을 피하면서 실시간으로 업데이트되는 지도 기반으로 행동하게 됩니다.

- **Performance Highlights**: NavVLM은 기존의 특정 목표 설정에 대해 성공률(SR)과 경로 길이로 가중된 성공률(SPL) 모두에서 최첨단 성능을 달성하였습니다. 또한, 비특정 언어 목표에서도 뛰어난 탐색 능력을 보여줍니다. 다채로운 환경에서의 평가 결과, 환경에 대한 깊이 있는 이해와 탐색의 인간적 접근 방식을 성공적으로 구현했습니다.



### Learning variant product relationship and variation attributes from e-commerce website structures (https://arxiv.org/abs/2410.02779)
- **What's New**: 이 논문에서는 VARM(variant relationship matcher) 전략을 소개하여 전자상거래 카탈로그에서 변형된 제품 쌍을 식별하는 방법을 제안합니다. 기존의 엔티티 해상도는 제품 언급이 동일한 기본 제품을 참조하는지를 판단하는 데 중점을 두었으나, 이는 전자상거래 어플리케이션에서 중요한 제품 관계를 포착하지 못합니다.

- **Technical Details**: VARM은 두 가지 요구 사항을 만족시키기 위해 엔코딩(encoding) 및 생성적 AI(Generative AI) 모델의 강점을 활용하는 전략을 개발했습니다. 먼저, 웹페이지의 제품 링크 및 변형된 제품 관계를 포착하는 데이터셋을 구성하여, 주어진 제품 쌍에 대한 변형 매칭을 예측하기 위해 LLM을 훈련합니다. 두 번째로, RAG 기반 생성 LLM을 사용하여 변형된 그룹 간의 변동 및 공통 속성을 추출합니다.

- **Performance Highlights**: 세계적인 전자상거래 소매업체의 실제 데이터를 사용하여 모델 성능을 평가한 결과, 우리의 전략이 대안 솔루션보다 우수한 성능을 나타냈으며, 새로운 유형의 제품 관계를 활용하는 가능성을 제시합니다.



### Mind the Uncertainty in Human Disagreement: Evaluating Discrepancies between Model Predictions and Human Responses in VQA (https://arxiv.org/abs/2410.02773)
- **What's New**: 이번 연구는 Visual Question Answering (VQA) 작업에서 최첨단 비전-언어 모델이 인간 응답의 분포와 얼마나 잘 일치하는지 종합적으로 평가하고, 인간의 불확실성을 (Human Uncertainty in Disagreement, HUD) 고려하는 방법을 제안합니다.

- **Technical Details**: VQA 작업에서 HUD의 영향을 분석하기 위해, 샘플을 낮은, 중간, 높은 3가지 불확실성 수준으로 분류하였으며, 전통적인 정확도 외에도 총 변이 거리 (Total Variation Distance, TVD), Kullback-Leibler 발산 (KL), 인간 엔트로피 보정 오차 (Human Entropy Calibration Error, EntCE) 등의 새로운 지표를 사용하였습니다. 연구 결과, BEiT3와 같은 최신 모델도 다양한 인간 응답의 다중 레이블 분포를 포착하는 데 어려움을 겪고 있다는 것을 확인했습니다.

- **Performance Highlights**: 종합적으로 우리가 제안한 모델 보정 방법은 모델의 신뢰도를 인간의 불확실성과 더 잘 일치시킴으로써 VQA 성능을 향상시킬 수 있음을 보여주었습니다. 연구의 주요 기여는 HUD를 명시적으로 활용하여 VQA 모델의 인간 응답 분포와의 차이를 평가한 것입니다.



### AVG-LLaVA: A Large Multimodal Model with Adaptive Visual Granularity (https://arxiv.org/abs/2410.02745)
Comments:
          Preprint

- **What's New**: AVG-LLaVA는 이미지와 지시에 따라 적절한 시각적 세분화를 선택할 수 있는 LMM(대규모 다중모달 모델)입니다. 이 접근법은 시각적 토큰의 수를 줄이고 추론 속도를 증가시키며 모델 성능을 향상시킵니다.

- **Technical Details**: AVG-LLaVA는 (a) 여러 풀링 레이어를 포함하여 다양한 세분화의 시각적 토큰을 얻는 시각적 세분화 스케일러와 (b) Transformer 레이어, MLP 레이어, 투표자 레이어를 포함해 이미지와 지침에 기반하여 적절한 시각적 세분화를 선택하는 시각적 세분화 라우터를 도입합니다. RGLF(Ranking Granularity to align LMM Feedback)라는 새로운 훈련 패러다임을 제안하여 라우터가 시각적 세분화를 효과적으로 구별하도록 지원합니다.

- **Performance Highlights**: AVG-LLaVA는 11개의 벤치마크에서 우수한 성능을 달성하며, AI2D 벤치마크에서 시각적 토큰 수가 85.3% 감소하고 추론 속도가 2.53배 증가하는 등의 성과를 보였습니다.



New uploads on arXiv(cs.IR)

### Discovering Biases in Information Retrieval Models Using Relevance Thesaurus as Global Explanation (https://arxiv.org/abs/2410.03584)
- **What's New**: 이 논문은 신경 관련성 모델을 해석하는 새로운 방법론을 제안합니다. 이러한 방법론은 기존의 지역적 설명(local explanations)에서 벗어나 문서와 쿼리 간의 전역적(global) 설명을 제공합니다.

- **Technical Details**: 제안된 방법은 'relevance thesaurus'를 구축하여 의미상 관련된 쿼리와 문서 용어 쌍을 포함합니다. 이 슈어서(thesaurus)는 BM25와 같은 어휘 매칭 모델을 보강하여 신경 모델의 예측을 근사하는 데 사용됩니다. 또한 부분 쿼리 및 문서 세그먼트의 관련성을 평가하기 위해 신경 관련성 모델을 훈련합니다.

- **Performance Highlights**: 평가 결과, 슈어서의 설명은 순위 효과성과 타겟 신경 순위 모델에 대한 충실도의 기준을 만족합니다. 특히, 슈어서는 순위 모델에서 브랜드 이름 편bias이 존재함을 드러내며, 이는 설명 방법의 장점을 보여줍니다.



### Dreamming User Multimodal Representation for Micro-Video Recommendation (https://arxiv.org/abs/2410.03538)
- **What's New**: 이번 연구에서는 DreamUMM (Dreaming User Multi-Modal Representation)이라는 새로운 접근 방식을 제안하여 실시간 사용자 관심사를 다중 모드(multimodal) 공간에서 모델링합니다. 이는 사용자 역사 데이터를 활용하여 동적 관심사를 포착하는 데 중점을 두고 있습니다.

- **Technical Details**: DreamUMM은 사용자의 역사적 상호작용을 바탕으로 사용자의 비디오 선호도와 다중 모드 유사성을 연관짓는 폐쇄형 솔루션을 사용합니다. 또한, Candidate-DreamUMM을 통해 최근 사용자 행동 데이터가 부족한 상황에서도 후보 비디오에서 관심사를 추론할 수 있도록 설계되었습니다. 이 과정에서 대형 언어 모델(large language models)과 지식 증류(knowledge distillation)를 활용하여 비디오의 복잡한 시각적, 청각적, 텍스트 요소 간의 상호작용을 포착하는 고품질 다중 모드 임베딩을 생성합니다.

- **Performance Highlights**: 광범위한 온라인 A/B 테스트를 통해 사용자 참여 지표(활동 일수 및 재생 수 등)에서 눈에 띄는 개선이 나타났습니다. DreamUMM은 수억 명의 일간 활성 사용자(daily active users)를 가진 두 개의 마이크로 비디오 플랫폼에 성공적으로 배포되어 실용성과 확장성을 입증했습니다.



### EB-NeRD: A Large-Scale Dataset for News Recommendation (https://arxiv.org/abs/2410.03432)
Comments:
          11 pages, 8 tables, 2 figures, RecSys '24

- **What's New**: 에크스트라 블라뎃 뉴스 추천 데이터셋(EB-NeRD)이 도입되었습니다. 이 데이터셋은 100만 명 이상의 고유 사용자의 데이터와 3,700만 개 이상의 인상 로그를 포함하고 있으며, 125,000개 이상의 덴마크 뉴스 기사를 포함하고 있습니다. EB-NeRD는 RecSys '24 챌린지의 기준 데이터셋으로 활용되었습니다.

- **Technical Details**: EB-NeRD 데이터셋은 사용자 행동 로그로부터 수집되었으며, 다양한 기술적 문제를 해결하기 위한 연구를 지원합니다. 여기에는 뉴스 기사의 연속적인 출간, 신속한 소멸 문제, 사용자 피드백 기반의 모델링 기법이 포함됩니다. 또한, 텍스트 정보를 활용하는 방법도 강조됩니다.

- **Performance Highlights**: EB-NeRD 데이터셋은 고전적인 디지털 뉴스 게시자의 각기 다른 뉴스 소비 및 콘텐츠 프로필에 대한 일반화 가능성을 탐색할 수 있는 기회를 제공합니다. 데이터셋은 뉴스 추천 시스템의 설계에서 기술적 및 규범적 도전을 해결하는 데 매우 유용합니다.



### Multimodal Point-of-Interest Recommendation (https://arxiv.org/abs/2410.03265)
- **What's New**: 본 논문에서는 대규모 언어모델을 이용한 새로운 음식 추천 시스템을 제안합니다. 주 사용자 방문 이력을 기반으로 한 레스토랑 추천을 중심으로, 시각적 정보와 지리적 속성을 신중하게 결합하여 보다 효과적인 POI(Point of Interest) 추천을 목표로 하고 있습니다.

- **Technical Details**: 이 연구는 Foursquare와 FoodX-251 데이터셋을 결합하여 레스토랑 추천을 위한 반(半)다중 모달(different modalities) 데이터셋을 생성합니다. LLaVA(Visual Language Model)를 활용해 음식 이미지를 텍스트 설명으로 변환하고, Recformer라는 언어 기반 시퀀스 추천 프레임워크를 사용하여 학습합니다. 평가는 실제 사용자의 레스토랑 방문 기록을 기반으로 수행되며, 지리적 정보가 포함된 새로운 접근 방식을 통해 추천 모델의 성능을 향상시키려 합니다.

- **Performance Highlights**: 연구 결과, 음식 이미지 설명이 포함된 세미-다중 모달 모델이 기존의 단순 텍스트 모델을 초과하는 성능을 보여주었습니다. 이를 통해 사용자의 실제 행동을 반영하는 방향으로 추천 시스템을 개선할 수 있다는 가능성을 제시합니다.



### Data-Efficient Massive Tool Retrieval: A Reinforcement Learning Approach for Query-Tool Alignment with Language Models (https://arxiv.org/abs/2410.03212)
- **What's New**: 최근 대형 언어 모델(LLMs)과 외부 도구 및 API의 통합이 복잡한 작업을 효과적으로 처리하고 있습니다. 그러나 도구 검색의 대규모 처리에 어려움이 있는 가운데, 본 연구에서는 대규모 도구 검색(MTR) 문제를 새로운 방식으로 제안하고 이를 평가하기 위한 MTRB 벤치마크를 도입합니다.

- **Technical Details**: MTRB 벤치마크는 2,645개의 다양한 도구를 포함하며, 90개의 테스트 샘플과 10개의 훈련 샘플로 구성된 세 개의 하위 집합으로 나뉘어 있습니다. 이 연구는 쿼리와 도구 간의 정렬을 향상시키기 위한 새로운 QTA 프레임워크를 도입하여 LLMs를 활용해 사용자 쿼리를 재작성하고, 이를 통해 도구 검색 성능을 개선합니다.

- **Performance Highlights**: QTA 프레임워크는 MTRB 벤치마크에서 기존 최고의 모델들과 비교하여 top-5 및 top-10 검색 작업에서 일관되게 성능을 초월하며, 특히 Sufficiency@5 메트릭에서 93.28%의 개선을 달성했습니다. 또한, 단 하나의 주석 샘플만으로도 78.53%의 성능 향상을 이루어내어, 제한된 표본으로도 우수한 성능을 발휘함을 입증했습니다.



### Geometric Collaborative Filtering with Convergenc (https://arxiv.org/abs/2410.03064)
Comments:
          13 pages, 1 figure, 3 tables

- **What's New**: 이번 연구에서는 잠재 변수 기반의 협업 필터링(latent variable collaborative filtering) 방법의 수학적 특성을 처음으로 분석하고, 일반화(generalization) 갭을 정의하였습니다. 특히, 아이템 메타데이터(metadata)를 활용하여 추천 시스템의 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: 본 논문에서는 일반화 갭의 개념을 도입하여 기하학적 상한(geometric upper bound)을 제시합니다. 이를 바탕으로 새로운 추천 알고리즘 GeoCF를 개발하였으며, 아이템 간의 기하학적 관계를 고려한 손실 함수(loss function)를 정의하였습니다. 제안된 방법론은 통계 학습 이론(Statistical Learning Theory)을 기반으로 하여 과적합(overfitting)을 방지하는 방향으로 설계되었습니다.

- **Performance Highlights**: GeoCF 알고리즘은 Movielens20M 및 Netflix 데이터 세트와 두 개의 대규모 내부 데이터 세트에서 다른 기존 방법들보다 뛰어난 성능을 보여주었습니다. 이를 통해, GeoCF는 기존의 최첨단 방법들의 성능을 초월하는 것을 입증했습니다.



### Inductive Generative Recommendation via Retrieval-based Speculation (https://arxiv.org/abs/2410.02939)
- **What's New**: SpecGR는 Generative Recommendation (GR) 모델이 새로운 아이템을 추천할 수 있도록 하는 플러그 앤 플레이 프레임워크입니다. 이를 통해 GR 모델은 훈련 중에 본 적이 없는 아이템들을 추천할 수 있습니다.

- **Technical Details**: SpecGR는 드래프트(drafter) 모델과 검증자(verifier) 모델로 구성되어 있으며, 드래프트 모델은 새로운 아이템을 제안하고, GR 모델은 이를 검증하며 추천 순위를 매깁니다. guided re-drafting 기법을 활용하여 후보 아이템의 품질을 개선합니다.

- **Performance Highlights**: 세 가지 실제 데이터셋에서 실험을 수행한 결과, SpecGR는 GR 모델의 새로운 아이템 추천 능력을 크게 향상시키며 기존 방법들과 비교하여 최상의 전반적인 성능을 보여주었습니다.



### Streamlining Conformal Information Retrieval via Score Refinemen (https://arxiv.org/abs/2410.02914)
Comments:
          6 pages

- **What's New**: 본 논문에서는 정보 검색(Information Retrieval, IR) 시스템에서의 통계적 보장을 제공할 수 있는 새로운 스코어 리파인먼트(score refinement) 방법을 제안합니다. 이 방법은 기존의 큰 크기의 세트를 생성하는 문제를 해결하여 작은 크기의 세트를 유지하면서도 통계적 보장을 보장합니다.

- **Technical Details**: 우리가 제안하는 스코어 리파인먼트 방법은 단순한 단조 변환(monotone transformation)을 적용하여 IR 시스템의 점수를 조정합니다. 이러한 점수의 정제(refinement)를 통해, 표준적인 위신 예측(conformal prediction) 방법을 사용하여 컴팩트한 세트를 생성하고, 불필요한 계산 비용을 줄이며 응답 속도를 향상시킵니다.

- **Performance Highlights**: 실험 결과, BEIR 벤치마크 데이터셋을 통해 제안한 방법이 경쟁 방식들보다 더 효과적으로 관련 정보를 포함한 소형 세트를 생성함을 확인했습니다.



### Cognitive Biases in Large Language Models for News Recommendation (https://arxiv.org/abs/2410.02897)
Comments:
          Accepted at the ROGEN '24 workshop, co-located with ACM RecSys '24

- **What's New**: 이 논문은 대형 언어 모델(LLM)이 뉴스 추천 시스템에 미치는 인지 편향(cognitive biases)의 영향을 다양한 측면에서 탐구하며, 이러한 편향이 시스템의 신뢰성에 어떤 악영향을 미칠 수 있는지를 분석합니다. 또한 데이터 증강(data augmentation), 프롬프트 엔지니어링(prompt engineering), 학습 알고리즘(learning algorithms) 등의 방법을 통해 이러한 편향을 완화하는 전략을 제시합니다.

- **Technical Details**: 이 논문에서는 인지 편향의 다양한 종류(앙커링 편향(anchoring bias), 프레이밍 편향(framing bias), 현 상태 유지 편향(status quo bias), 집단 귀인 편향(group attribution bias))가 LLM 기반 뉴스 추천 시스템에 미치는 영향을 분석합니다. 이를 통해, LLM이 트레이닝 데이터에서 상속받는 인간의 인지 편향이 결과에 어떻게 반영될 수 있는지를 보여줍니다.

- **Performance Highlights**: LLM 기반 뉴스 추천 시스템의 신뢰성을 높이기 위해 제시된 완화 전략은 다음과 같습니다: 합성 데이터 증강(synthetic data augmentation), 반복 수정(self-debiasing via iterative refinement), 인간 피드백을 통한 인지 편향 수정(cognitive debiasing through human feedback). 이러한 접근 방법들은 추천 시스템이 보다 객관적이고 공정한 출력을 생성하도록 도와줍니다.



### DifFaiRec: Generative Fair Recommender with Conditional Diffusion Mod (https://arxiv.org/abs/2410.02791)
Comments:
          The paper was accepted by ICDM 2024

- **What's New**: 이 논문에서는 사용자 선호도 기반으로 공정한 추천을 제공하는 Diffusion-based Fair Recommender (DifFaiRec)라는 새로운 추천 알고리즘을 제안합니다. 이 알고리즘은 조건부 확산 모델(conditional diffusion model)을 기반으로 하여 사용자 선호도의 분포를 효과적으로 학습하고 다양한 추천을 생성할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: DifFaiRec는 두 개의 모듈로 구성됩니다. 첫 번째 모듈은 사용자를 서로 다른 그룹으로 매핑하여 그룹 공정성을 보장하는 역사실 모듈(counterfactual module)입니다. 두 번째 모듈은 이 역사실 모듈에 조건화된 조건부 확산 모델(conditional diffusion model)로, 관찰된 상호작용을 재구성하고 생성을 통해 알려지지 않은 상호작용을 예측합니다. 또한, 정확도(accuracy)와 공정성(fairness)의 두 목적을 하나로 압축하여 최적화 문제의 난이도를 줄입니다.

- **Performance Highlights**: 실험 결과, DifFaiRec는 두 개의 실제 데이터셋에서 기존의 여러 기준선(baselines)을 초월하고 정확성과 공정성을 동시에 유지하는 뛰어난 성능을 보였습니다.



### Learning variant product relationship and variation attributes from e-commerce website structures (https://arxiv.org/abs/2410.02779)
- **What's New**: 이 논문에서는 VARM(variant relationship matcher) 전략을 소개하여 전자상거래 카탈로그에서 변형된 제품 쌍을 식별하는 방법을 제안합니다. 기존의 엔티티 해상도는 제품 언급이 동일한 기본 제품을 참조하는지를 판단하는 데 중점을 두었으나, 이는 전자상거래 어플리케이션에서 중요한 제품 관계를 포착하지 못합니다.

- **Technical Details**: VARM은 두 가지 요구 사항을 만족시키기 위해 엔코딩(encoding) 및 생성적 AI(Generative AI) 모델의 강점을 활용하는 전략을 개발했습니다. 먼저, 웹페이지의 제품 링크 및 변형된 제품 관계를 포착하는 데이터셋을 구성하여, 주어진 제품 쌍에 대한 변형 매칭을 예측하기 위해 LLM을 훈련합니다. 두 번째로, RAG 기반 생성 LLM을 사용하여 변형된 그룹 간의 변동 및 공통 속성을 추출합니다.

- **Performance Highlights**: 세계적인 전자상거래 소매업체의 실제 데이터를 사용하여 모델 성능을 평가한 결과, 우리의 전략이 대안 솔루션보다 우수한 성능을 나타냈으며, 새로운 유형의 제품 관계를 활용하는 가능성을 제시합니다.



### Bypassing the Popularity Bias: Repurposing Models for Better Long-Tail Recommendation (https://arxiv.org/abs/2410.02776)
Comments:
          6 pages, 4 figures

- **What's New**: 이번 연구에서는 온라인 콘텐츠 플랫폼에서 추천 시스템의 공정성을 높이기 위해, 장기 콘텐츠(Long-tail content)를 생산하는 출판사에 대한 노출을 공정하게 분배하는 새로운 방법을 제안합니다. 기존의 추천 시스템 구성 요소를 재활용하여 추천 품질을 유지하면서도 저조한 출판사들에게 더 많은 노출을 제공하는 방법을 도모했습니다.

- **Technical Details**: 이 연구에서는 사용자를 위한 품목 추천의 분석을 통해, 각 출판사가 생산한 품목에 대해 가장 적합한 사용자를 찾는 역추천(Inverse Retrieval) 모델을 도입했습니다. 또한, 장기 콘텐츠를 추천하기 위한 최소 노출 기준을 설정하고, 사용자의 상관성을 높이기 위해 사용자-품목 임베딩을 활용했습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 출판사 간 노출의 공평한 분배를 이루는 데 성공했으며, 추천 시스템의 전반적인 성능 개선 또한 확인되었습니다. 장기적인 응용에서도 긍정적인 결과를 보이며, 추천 품질과 공정성 지표 모두에서 개선된 효과를 나타냈습니다.



### SoundSignature: What Type of Music Do You Like? (https://arxiv.org/abs/2410.03375)
Comments:
          10 pages, 1 figure, to be published in the 2024 International Symposium on the IEEE Internet of Sounds Proceedings

- **What's New**: SoundSignature는 사용자들이 좋아하는 음악을 분석하기 위해 OpenAI Assistant와 통합된 음악 애플리케이션입니다. 이 시스템은 최신 Music Information Retrieval (MIR) Python 패키지를 활용하여 추출된 음향/음악적 특성과 아티스트 및 밴드에 대한 어시스턴트의 광범위한 지식을 결합합니다.

- **Technical Details**: 음악 애플리케이션은 Semantic Audio와 Emerging Internet of Sounds (IoS) 원칙을 활용하여 사용자의 음악에 대한 개인화된 통찰력을 제공합니다. CREMA(Chord Recognition Algorithm), DEMUCS(Source Separation Algorithm), basic-pitch(Audio-to-MIDI Converter) 등의 오픈 소스 음악 도구도 통합되어 있습니다.

- **Performance Highlights**: 이 애플리케이션은 사용자들이 음악의 음향 특성에 대한 이해를 넓힐 수 있도록 학습과 상호작용을 촉진하며, 파일럿 사용자 연구의 결과를 통해 효과성과 사용성을 평가한 결과를 제시합니다.



### Enriching Music Descriptions with a Finetuned-LLM and Metadata for Text-to-Music Retrieva (https://arxiv.org/abs/2410.03264)
Comments:
          Accepted for publication at the IEEE ICASSP 2024

- **What's New**: 이 논문은 다양한 음악 데이터에 기반한 텍스트-음악 검색(Text-to-Music Retrieval) 시스템의 개선된 모델인 TTMR++를 제안합니다. 이 모델은 정교한 텍스트 설명과 메타데이터를 활용하여 사용자가 원하는 음악을 보다 효과적으로 검색할 수 있도록 합니다.

- **Technical Details**: TTMR++는 음악 오디오와 텍스트 샘플을 쌍으로 제공받아, 이를 공동 임베딩 공간에 매핑하는 모델입니다. 이 모델은 multi-modal InfoNCE 손실 함수를 사용하여 긍정 쌍의 유사성을 극대화하고 부정 쌍의 유사성을 최소화합니다. 오디오 인코더로는 수정된 ResNet-50을, 텍스트 인코더로는 RoBERTa를 사용하였습니다.

- **Performance Highlights**: TTMR++는 다양한 음악 텍스트 쿼리를 포함한 종합적인 평가를 통해 기존의 음악-텍스트 공동 임베딩 모델과 비교하여 우수한 성능을 보입니다. 특히, 사용자 요청에 맞춰 음악의 유사성을 이해하고 적절한 검색 결과를 제공하는 능력이 향상되었습니다.



### Enhancing Short-Text Topic Modeling with LLM-Driven Context Expansion and Prefix-Tuned VAEs (https://arxiv.org/abs/2410.03071)
Comments:
          EMNLP Findings 2024. arXiv admin note: substantial text overlap with arXiv:2310.15420

- **What's New**: 이번 연구에서는 짧은 텍스트에서 유의미한 주제를 추출하기 위해 대형 언어 모델(LLMs)을 활용하여 짧은 텍스트를 보다 상세한 시퀀스로 확장한 후 주제 모델링을 적용하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안된 방법은 Prefix-tuned Variational Topic Model (PVTM)을 도입하여 짧은 텍스트에서 주제를 추론하는 소형 언어 모델과 변분 오토인코더(VAE)를 결합합니다. 이때, 전체 언어 모델을 튜닝하는 대신 prefix tuning 방법을 사용하여 특정 도메인 기능을 효과적으로 포착합니다.

- **Performance Highlights**: 본 모델은 극심한 데이터 희소성 문제가 있는 실제 데이터셋에서 기존의 최첨단 주제 모델보다 우수한 성능을 보였습니다. 다양한 데이터셋과 여러 작업에 대한 포괄적인 실험을 통해 이 모델의 우수성을 입증하였습니다.



### Scalable Frame-based Construction of Sociocultural NormBases for Socially-Aware Dialogues (https://arxiv.org/abs/2410.03049)
Comments:
          17 pages

- **What's New**: 본 논문은 대화에서 사회적으로 인식된 행동을 지원하기 위해 대형 언어 모델(LLMs)을 활용한 사회문화적 규범(SCN) 제작을 제안합니다. 이를 통해 중국 문화에 특화된 첫 번째 SCN 데이터베이스인 ChineseNormBase를 구축했습니다. 이 데이터베이스는 사회적 맥락을 고려하여 생성된 자연어 규범 진술을 포함하고 있습니다.

- **Technical Details**: SCNs는 사회맥락적 프레임을 이용해 추출되며, 이 과정은 대화의 맥락을 이해하고 환각(hallucination)을 줄이는 데 도움이 됩니다. 실제 대화 데이터가 부족할 경우, 합성 데이터(synthetic data)를 사용하여 SCNs를 효과적으로 생성할 수 있습니다. 이와 더불어, RAG 기반 모델(Retrieval-Augmented Generation)을 통해 다양한 대화 작업에 대한 추론 능력을 시연했습니다.

- **Performance Highlights**: 실험 결과, 합성 데이터를 이용하여 추출한 SCNs의 품질이 금본(gold frames)으로 주석을 달아 만든 실제 대화에서 추출한 SCNs와 동등하다는 것을 확인했습니다. 또한, 은본(silver frames)이나 금본으로 주석을 단 실제 데이터에서 추출된 SCNs의 품질이 주석이 없는 데이터와 비교하여 우수함을 입증했습니다.



### YouTube Video Analytics for Patient Engagement: Evidence from Colonoscopy Preparation Videos (https://arxiv.org/abs/2410.02830)
Comments:
          The 30th WORKSHOP ON INFORMATION TECHNOLOGIES AND SYSTEMS. arXiv admin note: substantial text overlap with arXiv:2312.09425

- **What's New**: 이 연구는 환자 교육을 위한 YouTube 비디오에서의 의학적 정보 검색 및 분석을 위한 데이터 분석 파이프라인을 처음으로 선보입니다.

- **Technical Details**: YouTube Data API를 활용하여 선택한 검색 키워드에 대한 비디오 메타데이터를 수집하고, Google Video Intelligence API를 사용하여 텍스트, 프레임 및 객체 데이터를 분석합니다. 또한 Bidirectional Long Short-Term Memory (BiLSTM) 모델을 개발하여 비디오에서 의학 용어를 식별하고, 비디오의 의학적 정보 수준 및 이해 가능성에 따라 비디오를 분류하는 세 가지 클래시파이어를 구축합니다.

- **Performance Highlights**: 이 연구는 헬스케어 이해당사자들이 다양한 건강 상태 관리를 위한 새로운 교육 비디오 콘텐츠 생성을 위한 지침과 확장 가능한 방법론을 제공하여, 비디오의 의학적 정보와 이해 가능성을 향상시킵니다.



New uploads on arXiv(cs.CV)

### Estimating Body and Hand Motion in an Ego-sensed World (https://arxiv.org/abs/2410.03665)
Comments:
          Project page: this https URL

- **What's New**: EgoAllo는 헤드 마운트 장치에서 인간의 동작을 추정하는 시스템입니다. 이 시스템은 egocentric SLAM poses와 이미지를 사용하여 3D 몸 자세, 신장(height), 손 파라미터를 추정합니다.

- **Technical Details**: EgoAllo는 조건부 diffusion 모델을 활용하여 인간의 동작을 추정하며, 공간적 및 시간적 불변 특성(criteria)이 모델 성능을 향상시키는 데 기여합니다. 우리는 헤드 모션 조건화 매개변수를 제안하여 최대 18%의 추정 개선 효과를 보여줍니다.

- **Performance Highlights**: EgoAllo 시스템은 손 추정을 개선하여 시끄러운 단안 모노큘러 추정과 비교하여 40% 이상의 추정 오차 감소를 달성합니다.



### Unraveling Cross-Modality Knowledge Conflict in Large Vision-Language Models (https://arxiv.org/abs/2410.03659)
Comments:
          Website: this https URL

- **What's New**: 본 논문에서는 LVLMs(대형 비전-언어 모델)에서의 교차 모달 파라메트릭 지식 충돌(cross-modality parametric knowledge conflict) 문제를 정의하고, 이를 감지하고 해석하며 완화하기 위한 전반적인 접근 방법을 소개합니다.

- **Technical Details**: 교차 모달 파라메트릭 지식 충돌 문제를 체계적으로 분석하고, 여러 가지 모델에서 시각적인 답변과 문자적 답변 간의 충돌을 감지하는 파이프라인을 제안합니다. 저자들은 동적 대조 디코딩(dynamic contrastive decoding) 방법을 개발하여 신뢰도가 낮은 모달리티 구성 요소에서 추론한 불필요한 로그잇(logits)을 제거합니다.

- **Performance Highlights**: LLaVA-34B를 사용하여 제안된 동적 대조 디코딩 방법이 ViQuAE와 InfoSeek 데이터셋에서 평균 2.24% 향상된 정확도를 기록했습니다.



### Unlearnable 3D Point Clouds: Class-wise Transformation Is All You Need (https://arxiv.org/abs/2410.03644)
Comments:
          NeurIPS 2024

- **What's New**: 본 논문은 3D 포인트 클라우드 데이터를 위한 최초의 포괄적(unlearnable) 프레임워크를 제안하며, 이 프레임워크는 데이터 보호 및 복구 과정을 포함합니다. 이를 통해 무단 사용을 방지하면서도 허가된 사용자에게는 유용한 정보를 제공합니다.

- **Technical Details**: 제안된 구조는 두 가지 주요 과정으로 구성됩니다: (i) 카테고리 적응 할당 전략과 다중 변환(multi-transformations)을 통해 클래스별(unlearnable) 데이터 보호 체계를 설정하는 것, (ii) 클래스별 역행렬 변환(class-wise inverse matrix transformation)을 활용한 데이터 복구 체계입니다.

- **Performance Highlights**: 6개의 데이터셋과 16개의 모델을 활용한 실험적 결과는 제안한 구조가 기존 방식들보다 뛰어난 성능을 발휘함을 입증합니다. 특히, 제안한 데이터 복구 방식은 허가된 사용자들이 효과적으로 학습할 수 있도록 도와줍니다.



### Variational Bayes Gaussian Splatting (https://arxiv.org/abs/2410.03592)
- **What's New**: 최근, Variational Bayes Gaussian Splatting (VBGS)라는 새로운 접근법이 제안되어 Gaussian Splatting을 변분 추론으로 틀 지었습니다. 이 방법은 기억력 고갈 문제를 해결하는 동시에 외부 재생 버퍼 없이 효율적인 학습을 지원합니다.

- **Technical Details**: VBGS는 가우시안 혼합 모델의 매개변수에 대해 변분 추론을 프레임으로 하여 사용할 수 있는 폐쇄형 변분 업데이트 규칙을 도출합니다. 이 접근법은 연속적으로 입력되는 데이터의 부분 관측치를 통해 효율적인 업데이트를 수행할 수 있습니다. 또한, exponential family distributions의 공액적 성질을 활용합니다.

- **Performance Highlights**: VBGS는 TinyImageNet과 Blender 3D 모델 및 Habitat 장면의 데이터셋들을 사용하여 벤치마킹한 결과, 기존 최신 성능을 유지하며 2D 및 3D 데이터의 연속적인 학습에서 성능 향상을 보여주었습니다.



### Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models (https://arxiv.org/abs/2410.03577)
- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)에서 발생하는 'hallucination' 문제를 해결하기 위해 Memory-space Visual Retracing (MemVR)이라는 새로운 패러다임을 제안합니다. MemVR은 외부 지식 검색이나 추가적인 fine-tuning 없이 시각적 프롬프트를 MLLMs에 재주입하여 모델의 불확실성을 줄이는 접근법을 취합니다.

- **Technical Details**: MemVR은 Feed Forward Network (FFN)를 통해 시각적 증거를 key-value memory 형태로 재주입하는 방식을 사용합니다. 이는 시각적 메모리가 흐려질 때 중간 레이어에서 재탐색을 수행하여 보다 정확한 답변을 탐색할 수 있도록 만드는 방법론입니다.

- **Performance Highlights**: 광범위한 실험 결과를 통해 MemVR이 여러 MLLM들에서 hallucination 문제를 효과적으로 완화하며, 이전 연구들에 비해 효율성과 성능이 최적임을 보여주었습니다. MemVR은 실행 시간의 오버헤드를 추가하지 않고도 일반적인 기준에서 뛰어난 성능을 발휘하는 것을 강조합니다.



### Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features (https://arxiv.org/abs/2410.03558)
- **What's New**: 이번 논문에서는 diffusion 모델의 활성화(activations)를 활용한 특성 선택(feature selection)을 심층적으로 연구합니다. 과거 연구에서는 활성화의 평가 범위가 제한적이었으나, 많은 새로운 활성화가 등장함에 따라 우수한 선택 방법을 제안합니다.

- **Technical Details**: Diffusion 모델은 Gaussian 노이즈를 점진적으로 제거하여 이미지를 생성하는 강력한 generative 모델입니다. 이 연구에서는 U-Net 아키텍처 내에서 활성화를 추출하여 이들을 feature로 사용합니다. 이를 통해 활성화의 질이 성능에 미치는 영향을 분석하고, 효과적인 활성화 선택을 위한 세 가지 범주를 제시합니다: (i) 확산 노이즈의 매크로 수준, (ii) 해상도 내 세분화의 변화, (iii) 위치적 임베딩 없는 지역성.

- **Performance Highlights**: 여러 가지 discriminative 작업에서 실험을 통해 제안한 방법이 기존의 SOTA(superior of the art) 기법보다 우수한 성능을 발휘함을 입증하였습니다. 이로써 활성화 선택이 모델 성능 향상에 중요한 역할을 함을 보여줍니다.



### Constructive Apraxia: An Unexpected Limit of Instructible Vision-Language Models and Analog for Human Cognitive Disorders (https://arxiv.org/abs/2410.03551)
- **What's New**: 이번 연구는 instructible vision-language models (VLMs)와 인지 장애인 constructive apraxia 사이의 예상치 못한 유사점을 밝혀냈습니다.

- **Technical Details**: 25개의 최신 VLM 모델, 예를 들면 GPT-4 Vision, DALL-E 3, Midjourney v5를 테스트했습니다. 이들은 Ponzo illusion의 이미지를 생성하는 능력을 평가받았으며, 이는 기본적인 spatial reasoning을 요구합니다. 연구 결과, 25개 모델 중 24개가 수평선을 올바르게 렌더링하지 못했습니다. 이는 parietal lobe 손상이 있는 환자들에게서 보이는 결핍과 유사합니다.

- **Performance Highlights**: 모델들은 공간적 지침을 잘못 해석하여, 배경의 원근감에 맞춰 기울거나 정렬되지 않은 선을 생성했습니다. 이러한 행동은 apraxia 환자들이 간단한 도형을 복사하거나 구성하는 데 어려움을 겪는 방식과 유사합니다. 현재 VLM들은 다른 영역에서는 뛰어난 기능을 보이지만, 기본적인 spatial reasoning 능력에는 한계를 보이고 있습니다. 이러한 AI 시스템의 제약은 spatial cognition 결핍을 연구하기 위한 새로운 계산 모델을 제시하며, VLM의 구조 및 훈련 방법론에서 개선이 필요한 중요한 영역으로 부각됩니다.



### Classification-Denoising Networks (https://arxiv.org/abs/2410.03505)
Comments:
          18 pages, 5 figures

- **What's New**: 이번 논문에서는 이미지 분류(classification)와 노이즈 제거(denoising)를 통합하여 두 작업의 문제를 동시에 해결할 수 있는 새로운 프레임워크를 제시합니다. 이를 통해 노이즈가 있는 이미지와 클래스 레이블의 결합 확률을 모델링하며, 효율성과 견고성을 높일 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 모델은 양측 확률 밀도 log⁡p(𝐲,c)를 매개변수화하는 단일 네트워크로 구성되어 있습니다. 분류 및 조건부 노이즈 제거 작업이 하나의 네트워크에서 수행되며, 손실 함수는 교차 엔트로피(cross-entropy)와 노이즈 제거 점수(match score)를 결합한 형태로 설정됩니다. GradResNet 아키텍처를 통해 ResNet의 특정 수정을 최소한으로 하여 UNet의 유도 편향(inductive biases)을 통합합니다.

- **Performance Highlights**: CIFAR-10과 ImageNet 데이터셋에서 경쟁력 있는 이미지 분류 및 노이즈 제거 성능을 보여주며, 이전의 통합 모델에 비해 효율성이 현저하게 개선되었습니다. 또한, 제안된 모델은 표준 분류기보다 적대적 변형(adversarial perturbations)에 더욱 견고한 성능을 보여주며, 적대적 그래디언트를 새로운 해석으로 제시합니다.



### A Multimodal Framework for Deepfake Detection (https://arxiv.org/abs/2410.03487)
Comments:
          22 pages, 14 figures, Accepted in Journal of Electrical Systems

- **What's New**: 딥페이크(dipfake) 기술의 급속한 발전이 디지털 미디어의 무결성에 중대한 위협을 가하고 있습니다. 본 연구에서는 시각 및 청각 요소를 모두 포괄하는 혁신적인 다중 모달(multimodal) 접근 방식을 통해 딥페이크 문제를 해결하고자 했습니다.

- **Technical Details**: 우리의 모형은 고급 특성 추출 기법을 사용하여 비디오의 아홉 가지 개별적인 얼굴 특징을 추출하고, 다양한 머신러닝(machin learning) 및 딥러닝(deep learning) 모델을 적용했습니다. 오디오 분석을 위해 mel-spectrogram 분석을 활용하여 특징을 추출하고, 동일한 방식으로 머신러닝 및 딥러닝 기법을 적용했습니다. 우리 모형은 비디오 및 오디오 분류를 위해 인공신경망(Artificial Neural Network)과 VGG19를 사용하여 전체 샘플을 딥페이크로 분류합니다.

- **Performance Highlights**: 우리의 다중 모달 프레임워크는 시각 및 청각 분석을 결합하여 94%의 정확도를 달성했습니다.



### VEDIT: Latent Prediction Architecture For Procedural Video Representation Learning (https://arxiv.org/abs/2410.03478)
Comments:
          10 pages

- **What's New**: 본 연구는 비디오 클립 시퀀스를 학습하기 위해 대규모 사전 훈련을 필요로 하지 않는 새로운 접근 방식을 제안합니다. 특히, 동결된(pretrained) 비주얼 인코더와 잘 설계된 예측 모델을 사용하여 최첨단 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 이 방법은 공용으로 제공되는 비전 인코더의 잠재 임베딩(latent embedding) 공간을 활용하며, 관찰된 단계에서 추출한 동결된 클립 수준의 임베딩을 조건으로 사용하여 보이지 않는 단계의 행동을 예측합니다. 이를 통해 예측 모델은 반복적 디노이징(iterative denoising)을 통해 견고한 표현을 학습하게 됩니다. 본 연구는 Diffusion Transformers의 최근 발전을 활용했습니다.

- **Performance Highlights**: 총 4개의 데이터셋(NIV, CrossTask, COIN, Ego4D-v2)에서 5개의 절차적 학습 태스크에 대한 실험적으로, long-horizon action anticipation에서 강력한 기준선을 +2.6%(Verb ED@20) 및 +3.1%(Noun ED@20) 향상시키며, 단계 예측(step forecasting)에서 +5.0%, 작업 분류(task classification)에서 +3.8%, 절차 계획(procedure planning) 작업에서 최대 +2.28%(success rate), +3.39%(mAcc), +0.90%(mIoU) 향상된 결과를 보였습니다.



### Dynamic Diffusion Transformer (https://arxiv.org/abs/2410.03456)
- **What's New**: Dynamic Diffusion Transformer (DyDiT)를 제안하여 Diffusion Transformer (DiT)의 비효율성을 극복하기 위해 동적 컴퓨테이션을 도입했습니다.

- **Technical Details**: DyDiT는 Timestep-wise Dynamic Width (TDW)와 Spatial-wise Dynamic Token (SDT)으로 구성되어, 생성 프로세스 중에 timesteps와 spatial dimensions에 따라 동적으로 컴퓨터 자원을 할당합니다. 이로 인해 연산 비용을 줄이고 효율성을 높이는 것이 가능합니다.

- **Performance Highlights**: DyDiT는 DiT-XL 대비 51%의 FLOPs 감소와 1.73배의 속도 향상을 보여주며, ImageNet에서 2.07의 경쟁력 있는 FID 점수를 달성했습니다.



### CLoSD: Closing the Loop between Simulation and Diffusion for multi-task character contro (https://arxiv.org/abs/2410.03441)
- **What's New**: CLoSD는 텍스트 기반의 RL 물리 제어기를 사용하여 다양한 작업을 수행할 수 있는 새로운 방법을 제시합니다. 이 방법은 동작 디퓨전 모델과 RL 제어기의 장점을 결합하여 강력하게 작동합니다.

- **Technical Details**: CLoSD는 Diffusion Planner (DiP)와 추적 제어기 간의 폐쇄 루프 상호 작용을 유지합니다. DiP는 텍스트 프롬프트와 목표 위치에 의해 제어되는 자가 회귀적 디퓨전 모델로, 물리 기반의 motion tracking controller가 이 계획을 실행합니다.

- **Performance Highlights**: 실험 결과에 따르면, "+method"는 현재의 선도적인 텍스트 기반 모션 제어기와 다중 작업 제어기를 능가하며, 3,500 fps에서 40프레임 계획을 생성하여 실시간 응용 가능성을 입증했습니다.



### Dessie: Disentanglement for Articulated 3D Horse Shape and Pose Estimation from Images (https://arxiv.org/abs/2410.03438)
Comments:
          ACCV2024

- **What's New**: 이 논문은 말의 3D 형상과 자세를 추정하기 위해 합성 데이터 생성과 분리(disentanglement) 학습을 활용한 최초의 방법인 Dessie를 소개합니다.

- **Technical Details**: Dessie는 horses에 초점을 맞추며, 텍스트 기반의 텍스처 생성 및 합성 데이터 파이프라인을 사용하여 다양한 형상, 자세, 외관을 생성하고, 이를 통해 분리된 공간을 학습합니다. DessiePIPE라는 새로운 합성 파이프라인을 통해 리얼타임으로 사실적인 동물 이미지를 생성하며, DINO라는 비전 기반의 기초 모델을 활용하여 분리된 잠재 공간에서의 3D 자세 및 형상 추정을 수행합니다.

- **Performance Highlights**: Dessie는 제한된 실제 데이터에서도 기존 3D 동물 재구성 방법을 초월하여, 부정확한 정보의 영향을 최소화하고 벤치마크에서 SOTA 성능을 달성함으로써, 데이터 부족 문제에 대한 솔루션을 제공합니다.



### Images Speak Volumes: User-Centric Assessment of Image Generation for Accessible Communication (https://arxiv.org/abs/2410.03430)
Comments:
          To be published at TSAR workshop 2024 (this https URL)

- **What's New**: 이번 연구는 Easy-to-Read (E2R) 텍스트에 최적화된 설명 이미지를 생성하기 위한 텍스트-이미지 생성 모델의 가능성을 조사했습니다. 연구진은 7개의 텍스트-이미지 모델을 벤치마킹하고, 사용자 그룹을 대상으로 한 연구를 통해 생성된 이미지가 E2R 텍스트에 적합한지를 평가했습니다.

- **Technical Details**: 연구진은 4개의 오픈소스 모델과 3개의 닫힌 소스 모델을 포함하여 총 7개의 텍스트-이미지 생성 모델을 평가했습니다. 생성된 이미지는 2,217개의 이미지 데이터셋으로 제공되며, 연구진은 560개의 이미지를 평가하여 정확성, 편향성, 목표 집단의 적합성을 기준으로 주석을 달았습니다.

- **Performance Highlights**: 일부 모델은 탁월한 성능을 보였으나, 인간의 감독 없이 대규모로 사용될 준비가 되어있지 않다는 것이 발견되었습니다. 이는 E2R 창작자들이 접근 가능한 정보를 생성하는 데에 중요한 이정표가 됩니다.



### Img2CAD: Conditioned 3D CAD Model Generation from Single Image with Structured Visual Geometry (https://arxiv.org/abs/2410.03417)
- **What's New**: 이 논문에서는 2D 이미지 입력을 사용하여 편집 가능한 CAD 모델을 생성하는 최초의 접근법인 Img2CAD를 제안합니다. 이는 기존의 메쉬 기반 3D 모델 생성 방법과는 다른 접근으로, CAD 도구와의 통합을 용이하게 합니다.

- **Technical Details**: Img2CAD는 이미지를 인코딩하는 Transformer 기반 네트워크와 CAD 생성 명령 및 매개변수를 디코딩하는 Transformer를 사용합니다. 새로운 중간 표현인 Structured Visual Geometry (SVG)를 도입하여 물체의 벡터화된 와이어프레임을 추출하고, Holistic Attention Transformer (HAT) 필드를 사용하여 선 세그먼트를 인코딩합니다. Joint-Decoupled Line-of-Interest Aligning (JD LOIAlign) 모듈을 통해 잘못된 제안을 필터링하여 CAD 생성을 위한 명령과 매개변수를 만듭니다.

- **Performance Highlights**: 이 방법은 기존 3D AIGC 방법들과 비교하여 충실도, 표면 품질 및 추론 속도 면에서 최첨단(SOTA) 성능을 달성하였으며, 스케치 입력 및 실제 환경 이미지로부터 고충실도의 3D CAD 모델을 생성할 수 있습니다.



### Lightning UQ Box: A Comprehensive Framework for Uncertainty Quantification in Deep Learning (https://arxiv.org/abs/2410.03390)
Comments:
          10 pages, 8 figures

- **What's New**: Lightning UQ Box는 딥 뉴럴 네트워크(DNN)에 대한 불확실성 정량화(Uncertainty Quantification, UQ) 방법을 효과적으로 적용하고 평가할 수 있는 통합 인터페이스를 제공합니다. 이는 DNN의 결과에 신뢰도를 부여하는 데 필수적인 도구입니다.

- **Technical Details**: 이 툴박스는 PyTorch와 Lightning과 함께 작동하며, 다양한 UQ 방법론의 구현을 제공합니다. 특히, Bayes convolution layer와 같은 유연한 레이어 구성으로 비전 응용 프로그램에 최적화되어 있습니다.

- **Performance Highlights**: Lightning UQ Box는 위성 이미지를 기반으로 열대 저기압의 최대 지속 풍속을 추정하고, RGB 이미지를 이용하여 태양광 패널의 전력 출력을 예측하는 두 가지 도전적인 비전 작업에서 그 유용성을 입증합니다.



### LANTERN: Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding (https://arxiv.org/abs/2410.03355)
- **What's New**: 최근 Auto-Regressive (AR) 모델이 이미지 생성에 있어서 중요성을 얻고 있으며, diffusion 모델을 능가하는 성능을 보였습니다. 그러나 AR 모델의 시퀀스 처리 특성으로 인해 생성 속도가 저하되는 문제가 있습니다. 이 논문에서는 이러한 문제를 극복하기 위해 LANTERN 이라는 새로운 방법론을 제안합니다.

- **Technical Details**: 논문에서는 'token selection ambiguity'라는 문제를 다루고 있으며, 이는 시각적 AR 모델에서 토큰의 낮은 확률 분포로 인해 speculative decoding의 성능이 저하됨을 설명합니다. LANTERN은 이러한 수용 조건을 완화하고, 잠재 공간에서의 토큰 간의 상호 교환성을 이용하여 효과적으로 후보 토큰을 활용할 수 있도록 합니다. 이를 통해 이미지 품질과 의미론적 일관성을 크게 손상하지 않으면서 생성 속도를 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, LANTERN을 사용한 경우 LlamaGen 모델을 기준으로 greedy decoding에서 1.75배, random sampling에서 1.76배의 속도 향상이 나타났습니다. 이는 기존의 speculative decoding 방법보다 큰 성과입니다.



### An X-Ray Is Worth 15 Features: Sparse Autoencoders for Interpretable Radiology Report Generation (https://arxiv.org/abs/2410.03334)
- **What's New**: 이 논문은 기존의 비전-언어 모델에서 발생하는 hallucination 문제를 해결하기 위해 sparse autoencoders(SAEs)를 활용한 새로운 접근법인 SAE-Rad를 소개합니다. 이 모델은 기계적 해석 가능성(mechanistic interpretability) 기법을 적용하여, 방사선 이미지 인코더의 잠재 표현을 인간이 해석할 수 있는 특징으로 분해하는 방식입니다.

- **Technical Details**: SAE-Rad는 sparse autoencoders를 기반으로 한 하이브리드 아키텍처로, 상태-of-the-art 기술과 비교하여 유사한 밀도로 정확한 재구성을 달성합니다. 이 모델은 사전 훈련된 언어 모델을 사용하여 각 SAE 특징에 대한 실제 보고서를 방사선묘사로 증류(distil)한 후, 전체 보고서를 제작합니다. 이 방법은 대규모 모델의 세부 조정을 필요로 하지 않습니다.

- **Performance Highlights**: MIMIC-CXR 데이터셋에서 SAE-Rad는 경쟁력 있는 방사선 특화 지표를 달성하였으며, 훈련을 위한 계산 자원이 현저히 적게 소비되었습니다. SAE-Rad의 질적 분석 결과는 이 모델이 유의미한 시각 개념을 학습하고 전문가 해석과 유사한 보고서를 생성한다는 것을 보여줍니다. SAEs는 의료 분야에서 다중모드 추론을 향상시킬 수 있는 해석 가능한 대안을 제공합니다.



### Comparative Analysis and Ensemble Enhancement of Leading CNN Architectures for Breast Cancer Classification (https://arxiv.org/abs/2410.03333)
- **What's New**: 본 연구는 유방암의 진단을 위한 새로운 히스토 병리 이미지 분류 접근법을 소개합니다. 다양한 Convolutional Neural Network (CNN) 모델을 비교하고 최적의 하이퍼파라미터를 식별하여 분류 효율성을 기반으로 순위를 매깁니다.

- **Technical Details**: CNN 모델들에 대한 포괄적인 비교를 통해 초기 이미지를 시퀀스화하여 일관된 데이터 조건을 보장하고 훈련 기간을 단축시키는 원래의 개념들이 포함되어 있습니다. 연구에서는 2000개 이상의 훈련 조합을 탐구하였으며, BreakHis x40과 x200 데이터셋에서 각각 99.75%, Bach 데이터셋에서 95.18%의 높은 정확도를 기록했습니다.

- **Performance Highlights**: 본 연구의 앙상블 아키텍처는 세 개의 고성능 CNN 모델을 조합하여 분류 정확도를 높이는 데 성공하였으며, Bach Online 블라인드 챌린지에서는 89%의 성과를 나타냈습니다. 이 방법론은 유방암 히스토 병리 이미지 데이터셋 뿐만 아니라 다른 의학 이미지 데이터셋에도 적용 가능합니다.



### EmojiHeroVR: A Study on Facial Expression Recognition under Partial Occlusion from Head-Mounted Displays (https://arxiv.org/abs/2410.03331)
- **What's New**: 이 논문은 감정 인식을 위한 데이터베이스 EmoHeVRDB(EmojiHeroVR Database)를 소개하며, HMD(Head-Mounted Displays)로 인한 상반신 가림 현상에서의 Facial Expression Recognition (FER) 가능성을 탐구합니다. 이 데이터베이스는 1,778개의 재현된 감정으로부터 3,556개의 레이블된 얼굴 이미지를 포함하고 있으며, VR 환경에서 실제 감정 인식을 위한 기초 자료로 활용됩니다.

- **Technical Details**: 연구는 37명의 참가자가 novel affective VR 게임인 EmojiHeroVR을 플레이하며 수집된 얼굴 이미지 데이터로 진행되었습니다. EmoHeVRDB는 레이블된 이미지와 더불어 각 이미지 전후에 기록된 29개의 추가 프레임을 포함하여 동적 FER을 촉진합니다. EfficientNet-B0 아키텍처를 사용하여 정적 FER 분류 작업을 수행했으며, 기본 감정 및 중립 감정을 포함하여 모델은 최종적으로 69.84%의 정확도를 기록하였습니다.

- **Performance Highlights**: 정적 FER 시스템에서 HMD 가림 현상 아래에서도 FER이 가능하다는 것을 보여주었지만, 전통적인 FER보다 훨씬 더 복잡한 도전이라는 점도 강조되었습니다. 이 연구를 통해 VR 환경에서 감정을 인식하고 분석하는 새로운 기틀을 마련하고, 시스템의 상호작용성과 개인화 가능성을 더욱 강화할 수 있는 기반을 제공합니다.



### Does SpatioTemporal information benefit Two video summarization benchmarks? (https://arxiv.org/abs/2410.03323)
Comments:
          Accepted for presentation at AEQUITAS workshop, Co-located with ECAI 2024

- **What's New**: 이 논문은 비디오 요약에서 시공간(spatial-temporal) 관계가 실제로 필요한지에 대해 탐구합니다. 기존의 접근 방식이 상태-최고 성능을 달성하는지에 대한 의문을 제기합니다.

- **Technical Details**: 시공간 관계를 사용하지 않는 모델이 TVSum 및 SumMe 데이터셋에서 경쟁력 있는 상관 점수를 달성하는 것을 보였습니다. 또한, 시간 순서를 교란하여 이러한 모델에 미치는 영향을 조사했습니다.

- **Performance Highlights**: 교란 기술을 사용한 결과, 시공간 관계의 역할이 미미하다는 것을 발견하였고, 일부 교란 전략이 모델 성능을 향상시킬 수 있다는 사실을 보여주었습니다.



### Visual-O1: Understanding Ambiguous Instructions via Multi-modal Multi-turn Chain-of-thoughts Reasoning (https://arxiv.org/abs/2410.03321)
- **What's New**: 이 논문은 모호한 지시를 이해하기 위해 비주얼 컨텍스트(visual context)와 일반 상식(common sense)을 통합하는 새로운 멀티모달(multi-modal) 멀티턴(multi-turn) 체인 오브 쏘트(thought reasoning) 프레임워크인 Visual-O1을 제안합니다.

- **Technical Details**: Visual-O1 프레임워크는 인간의 멀티모달 멀티턴(reasoning 과정을 시뮬레이션하며, 고성능 모델(High intelligent models)와 일반 모델(Generally intelligent models)이 모호한 지시를 이해하는데 필요한 경험을 제공합니다. 이 프레임워크는 전통적인 방법들보다 계산 비용(computational overhead)을 대폭 증가시키지 않으며, 보다 일반적이고 효과적인 접근 방식을 제공합니다.

- **Performance Highlights**: 실험 결과, 다양한 지능 수준을 가진 모델들이 모호한 지시(task)에서 성능이 크게 향상되었고, 일반 데이터셋에서도 개선된 성능을 보여주었습니다. 이는 인공지능(AI)이 불확실성과 모호성이 존재하는 실제 상황에서도 인간처럼 작업할 수 있는 잠재력을 강조합니다.



### Quo Vadis, Motion Generation? From Large Language Models to Large Motion Models (https://arxiv.org/abs/2410.03311)
- **What's New**: 이 논문에서는 대규모 모션 모델 개발을 위한 첫 번째 백만 레벨의 모션 생성 벤치마크인 MotionBase를 소개합니다. 이는 이전의 가장 큰 데이터셋보다 15배 더 많은 데이터를 제공하고, 계층적으로 세분화된 텍스트 설명을 포함한 다중 모드 데이터로 구성되어 있습니다.

- **Technical Details**: MotionBase는 100만 개 이상의 모션 시퀀스를 포함하여 모션 데이터 수집의 필요성을 해결합니다. 이 연구는 모션 토큰화를 위한 새로운 2D 조회 없는 접근 방식을 도입하여 모션 정보를 보존하고 코드북 용량을 확장하여 대규모 모션 모델의 표현 능력을 향상시킵니다.

- **Performance Highlights**: MotionBase를 활용하여 대규모 모션 모델은 다양한 모션, 특히 보지 못한 모션에서도 높은 성능을 나타냅니다. 데이터 및 모델 크기 확장이 중요한 요인으로 밝혀졌고, 기존의 평가 메트릭은 도메인 외 텍스트 지침 처리에 한계를 보였음을 강조합니다.



### Action Selection Learning for Multi-label Multi-view Action Recognition (https://arxiv.org/abs/2410.03302)
Comments:
          ACM Multimedia Asia 2024

- **What's New**: 이번 연구는 멀티-레이블 멀티-뷰 액션 인식을 다루며, 여러 카메라에서 수집된 비정렬 비디오에서 동시 또는 순차적으로 발생하는 여러 행동을 인식하는 새로운 방법을 제안합니다. 기존의 연구들은 힘이 있는 라벨이 있는 좁은 영역에 한정되었으나, 본 연구는 약한 라벨만 존재하는 실제 시나리오를 다룹니다.

- **Technical Details**: 본 연구에서는 액션 선택 학습 (Action Selection Learning)을 활용하여 여러 시점에서 수집된 유용한 정보를 선택하고 강력한 라벨 없이 멀티-뷰 액션 인식을 향상시키는 MultiASL 방법을 제안합니다. Multi-view Spatial-Temporal Transformer 비디오 인코더를 도입하여 멀티-뷰 비디오에서 공간적(Spatial) 및 시간적(Temporal) 특징을 추출합니다. 또한, 비디오 수준의 약한 라벨에서 얻어진 의사 정답(pseudo ground-truth)을 사용하여 액션 식별에서 가장 관련 있는 프레임을 찾습니다.

- **Performance Highlights**: MM-Office 데이터셋을 사용한 실험 결과, 제안된 MultiASL 방법이 기존 방법들보다 우수한 성능을 보이는 것으로 나타났습니다. 특히, 다양한 멀티-뷰 융합 전략을 탐색한 결과, 최대 풀링(max pooling) 방법이 일관되게 가장 좋은 성능을 발휘한 것으로 확인되었습니다.



### Grounded-VideoLLM: Sharpening Fine-grained Temporal Grounding in Video Large Language Models (https://arxiv.org/abs/2410.03290)
- **What's New**: 본 논문에서는 특정 비디오 순간을 정밀하게 인식하고 추론할 수 있는 새로운 Video-LLM인 Grounded-VideoLLM을 제안합니다. 이 모델은 기존의 Video-LLM들이 가지는 세밀한 시간 기반 이해의 한계를 극복하기 위해 추가적인 시간 스트림과 특정 시간 지식이 포함된 이산적 시간 토큰을 도입하였습니다.

- **Technical Details**: Grounded-VideoLLM은 (1) 두 개의 스트림으로 구성된 인코딩 방식인 Two-Stream Encoding을 사용하여 각 비디오 단편의 공간적 및 시간적 요소를 분리하여 모델링합니다. (2) 시간적 토큰을 도입하여 시간 스탬프를 효율적으로 표현할 수 있도록 하고, 이 토큰들은 LLM의 임베딩 공간과 통합되어 직관적인 예측을 가능하게 합니다.

- **Performance Highlights**: 다양한 실험을 통해 Grounded-VideoLLM은 시간 문장 정량화, 밀집 비디오 캡션, Grounded VideoQA 등에서 높은 성능을 보이며 기존의 Video-LLM들보다 우수한 결과를 나타냈습니다. 이는 Grounded-VideoLLM이 일반 비디오 이해를 위한 다재다능한 비디오 어시스턴트로서 잠재력을 지니고 있음을 보여줍니다.



### Sm: enhanced localization in Multiple Instance Learning for medical imaging classification (https://arxiv.org/abs/2410.03276)
Comments:
          24 pages, 14 figures, 2024 Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문에서는 여러 인스턴스 학습(Multiple Instance Learning, MIL)에서 지역 종속성을 모델링하기 위한 새로운 방법을 제안합니다. 이 방법은 인스턴스의 이웃이 같은 라벨을 가질 가능성이 높다는 간단한 관찰에 기반합니다.

- **Technical Details**: 제안된 새로운 메커니즘은 지역 종속성을 모델링하는데 유연하고 원칙적인 접근을 사용하며, 기존의 글로벌 종속성 모델링 메커니즘(예: transformers)과 함께 사용될 수 있습니다. 이전의 MIL 방법들은 인스턴스를 독립적으로 취급하였으나, 최근 방법들은 인스턴스 간의 글로벌 및 로컬 종속성을 고려하고 있습니다.

- **Performance Highlights**: 제안된 모듈은 로컬라이제이션(localization)에서 최첨단 성능을 보이며, 분류(classification)에서도 경쟁력 있는 또는 우수한 성능을 보여줍니다.



### Frame-Voyager: Learning to Query Frames for Video Large Language Models (https://arxiv.org/abs/2410.03226)
Comments:
          19 pages, 10 figures

- **What's New**: 이번 논문에서는 비디오를 이해하는 작업에서 정보가 밀집된 프레임 조합을 쿼리하는 Frame-Voyager라는 새로운 접근법을 제안합니다. 이 방법은 텍스트 쿼리를 바탕으로 유용한 프레임 조합을 학습합니다.

- **Technical Details**: Frame-Voyager는 사전 학습된 Video-LLM을 사용하여 프레임 조합의 유용성을 기반으로 순위를 매깁니다. 이 방식을 통해 두 개의 주요 도전 과제를 해결하는데, 첫째, 높은 학습 복잡성을 효율적으로 관리하고, 둘째, 라벨링 데이터의 부족 문제를 극복합니다.

- **Performance Highlights**: 실험 결과, Frame-Voyager는 기존의 균일 샘플링 및 텍스트-프레임 검색 방법에 비해 유의미한 성능 향상을 보이며, 특히 복잡한 추론이 필요한 긴 비디오 작업에서 두각을 나타냅니다.



### Tuning Timestep-Distilled Diffusion Model Using Pairwise Sample Optimization (https://arxiv.org/abs/2410.03190)
- **What's New**: 본 논문에서는 'pairwise sample optimization (PSO)'라는 새로운 알고리즘을 제안합니다. PSO는 시간 단계가 추출된 확산 모형(timestep-distilled diffusion model)을 직접적으로 미세 조정(fine-tuning)할 수 있게 해줍니다.

- **Technical Details**: PSO는 현재 시간 단계에서 추출된 모델에서 샘플링한 추가 참조 이미지(reference images)를 도입하여 훈련 이미지(training images)와 참조 이미지 간의 상대적 가능성_MARGIN_을 증가시킵니다. 이를 통해 모델이 몇 단계 생성 능력을 유지하면서도 출력 분포를 미세 조정할 수 있습니다. PSO는 오프라인 샘플링과 온라인 샘플링 데이터 모두에 유연하게 확장할 수 있는 일반화된 공식화(generalized formulation)입니다.

- **Performance Highlights**: PSO는 선호 최적화(preference optimization) 및 기타 미세 조정 작업에서 평가되었으며, 스타일 전이(style transfer) 및 개념 맞춤(customization)에서도 효과적임을 보여주었습니다. PSO는 오프라인 및 온라인 생성된 쌍별 선호 이미지 데이터(pairwise preference image data)에 직접적으로 인간이 선호하는 생성에 적응할 수 있도록 도와줍니다.



### Generalizable Prompt Tuning for Vision-Language Models (https://arxiv.org/abs/2410.03189)
- **What's New**: 이 연구는 텍스트 프롬프트의 최적화를 통해 다운스트림 성능과 일반화 능력을 동시에 향상시키는 새로운 프롬프트 튜닝 방법을 제안합니다.

- **Technical Details**: 프롬프트 튜닝은 텍스트 프롬프트를 학습하여 이미지-텍스트 쌍을 생성하는 과정을 최적화합니다. 연구는 소프트 프롬프트와 수작업 프롬프트를 텍스트 모달리티의 이중 관점으로 처리하여 상호 정보를 최대화하는 방법에 중점을 둡니다. 또한 각 클래스에 대한 시각적 모달리티에서 증강을 도입하여 효과적인 프롬프트 생성을 가능하게 합니다.

- **Performance Highlights**: 제안된 방법은 다양한 벤치마크에서 다운스트림 성능과 일반 능력 모두에서 경쟁력 있는 결과를 보였습니다.



### Looking into Concept Explanation Methods for Diabetic Retinopathy Classification (https://arxiv.org/abs/2410.03188)
Comments:
          Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) this https URL

- **What's New**: 이 논문에서는 당뇨병성 망막병증(diabetic retinopathy, DR) 자동 진단을 위한 심층 신경망의 해석 가능성을 높이기 위해, 개념 기반 설명 방법 두 가지를 비교하고 평가했습니다. 특히, Concept Activation Vectors(TCAV)와 Concept Bottleneck Models(CBM) 방법을 조사하였으며, 두 방법은 각각 장점과 단점을 가지고 있습니다.

- **Technical Details**: 본 연구에서는 TCAV와 CBM 두 가지 개념 기반 설명 기술을 사용하여 DR 진단 인공지능 모델의 해석 가능성을 강조합니다. TCAV는 концепт의 방향으로 이미지 변경에 모델의 민감도를 측정하여 개념의 상대 중요성을 평가하며, CBM은 중간 개념 예측을 직접 수정하고 최종 예측이 어떻게 영향을 받는지 관찰하는 방법입니다.

- **Performance Highlights**: TCAV와 CBM 모두 심층 신경망의 예측을 설명하는 데 유용하지만, 선택한 방법은 사용 가능한 데이터와 최종 사용자의 선호도에 따라 달라져야 합니다. 정량적 분석 결과, 두 방법의 설명 능력과 해석의 직관성이 비교되었습니다.



### Autonomous Character-Scene Interaction Synthesis from Text Instruction (https://arxiv.org/abs/2410.03187)
- **What's New**: 본 논문에서는 단일 텍스트 지시문과 목표 위치로부터 다단계 장면 인식을 반영한 인간 움직임을 합성하는 새로운 프레임워크를 제안합니다. 현재의 모델들이 사용자 정의 경로와 단계 전환을 자동화하는 데 한계가 있는 점을 착안하여, 이 문제를 해결하고자 합니다.

- **Technical Details**: 자동회귀적 확산 모델(auto-regressive diffusion model)을 활용하여 다음 동작 세그먼트를 합성하며, 각 동작 단계에 대한 전환을 예측하는 자율 스케줄러(autonomous scheduler)를 도입합니다. 또한, 시작 위치와 목표 위치 모두에서의 국소적 인식을 반영하는 장면 표현(scene representation)을 제안하여 생성된 움직임이 환경에 잘 융합되도록 합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 환경과 텍스트 조건에 밀접하게 일치하는 고품질 다단계 모션을 생성하는 데 효과적임을 보여줍니다. 특히, 16시간의 모션 캡처 데이터셋을 기반으로 다양한 동작 유형을 포함한 데이터가 사용되었습니다.



### Investigating and Mitigating Object Hallucinations in Pretrained Vision-Language (CLIP) Models (https://arxiv.org/abs/2410.03176)
Comments:
          EMNLP 2024

- **What's New**: 이번 논문에서는 CLIP 모델에서 발생하는 객체 환각(object hallucination) 문제를 집중적으로 조사하고, 새로운 벤치마크인 OHD-Caps를 제안합니다. 이 모델은 시각-언어 시스템의 핵심 역할을 하며, 모델 내부의 환각이 어떤 부분에서 발생하는지를 분석하였습니다.

- **Technical Details**: LVLMs (Large Vision-Language Models)는 이미지 캡셔닝, 시각적 질문 응답과 같은 다양한 작업에서 우수한 성능을 보입니다. 본 논문에서는 CLIP 모델을 기반으로 하여, 'fine-grained object-level contrastive loss' 방법을 통해 CLIP 모델의 객체 환각을 완화하는 방법을 제안합니다. OHD-Caps 벤치마크를 사용하여 EVL 프레임워크의 성과를 평가하였습니다.

- **Performance Highlights**: CLIP 모델의 객체 환각 문제를 해결하기 위한 제안 방법은 기존 상태인 ‘CLIP ViT-B/32’에서 14.3%에서 82.5%로 향상되었습니다. 또한, LLaVA-1.5 모델에서도 환각 문제가 80.2%에서 83.2%로 개선되었습니다.



### HRVMamba: High-Resolution Visual State Space Model for Dense Prediction (https://arxiv.org/abs/2410.03174)
- **What's New**: 본 연구에서는 동적 비주얼 상태 공간(Dynamic Visual State Space, DVSS) 블록을 제안하여 Mamba 모델의 한계를 극복하고, 고해상도 비주얼 상태 공간 모델(High-Resolution Visual State Space Model, HRVMamba)을 통해 밀집 예측(dense prediction) 작업에서 주목할만한 성과를 보였습니다.

- **Technical Details**: DVSS 블록은 다양한 크기의 지역 특성을 추출하기 위해 다중 스케일(convolutional kernels) 합성곱을 사용하며, 변형 가능한 합성곱(deformable convolution)을 활용하여 긴 거리 정보를 잃어버리는 문제를 완화합니다. 이를 통해 고해상도 표현을 유지하며, HRNet에서 제안된 다중 해상도 평행 설계를 통해 HRVMamba를 구축하였습니다.

- **Performance Highlights**: HRVMamba는 이미지 분류, 인체 자세 추정(human pose estimation), 및 의미 분할(semantic segmentation) 작업에서 기존의 CNN, ViT 및 SSM 벤치마크 모델에 대해 경쟁력 있는 결과를 달성했습니다.



### Selective Transformer for Hyperspectral Image Classification (https://arxiv.org/abs/2410.03171)
- **What's New**: 이번 연구에서는 하이퍼스펙트럴 이미지(HSI) 분류를 위한 새로운 선택적 트랜스포머(SFormer)를 제안합니다. SFormer는 공간적 및 스펙트럴(contextual information) 맥락 정보를 동적으로 선택하고, 불필요한 데이터의 영향을 줄이며, 분류 정확성을 향상시킵니다.

- **Technical Details**: SFormer는 커널 선택형 트랜스포머 블록(KSTB)과 토큰 선택형 트랜스포머 블록(TSTB)을 포함합니다. KSTB는 최적의 수용 범위를 선택하여 효과적으로 공간-스펙트럴 특징을 추출합니다. TSTB는 주의(attention) 점수가 높은 토큰을 선택하여 관련 정보를 효과적으로 통합합니다.

- **Performance Highlights**: 네 가지 벤치마크 HSI 데이터셋(Pavia University, Houston, Indian Pines, WHU-HongHu)에서의 실험 결과, 제안된 SFormer가 기존 최첨단 HSI 분류 모델을 초월하는 성과를 보여주었습니다.



### Redefining Temporal Modeling in Video Diffusion: The Vectorized Timestep Approach (https://arxiv.org/abs/2410.03160)
Comments:
          Code at this https URL

- **What's New**: 이 논문은 기존의 비디오 확산 모델(VDM)의 한계를 해결하기 위해 새롭게 프레임 인식 비디오 확산 모델(FVDM)을 제안합니다. 이 모델은 각 프레임에 대해 독립적인 노이즈 스케줄을 허용하는 벡터화된 시간을 도입하여, 복잡한 시간적 의존성을 효과적으로 모델링합니다.

- **Technical Details**: FVDM은 기존 VDM의 스칼라 시점 변수를 대신해 벡터화된 시점 변수(VTV)를 사용합니다. 이를 통해 각 프레임은 고유의 시간 경로를 따라 발전할 수 있으며, 노이즈에서 복구되는 과정에서도 더 정교한 시간적 의존성을 캡처할 수 있습니다. 이 방법은 이전 모델에서 발생하는 catastrophic forgetting(재학습시의 대폭망각) 문제를 해결하고, 제로샷(zero-shot) 학습에서도 뛰어난 일반화를 보여줍니다.

- **Performance Highlights**: FVDM은 기존 최신 기술보다 비디오 생성 품질에서 뛰어난 성능을 발휘하며, 표준 비디오 생성뿐만 아니라 이미지-비디오 변환, 비디오 보간(video interpolation), 긴 비디오 생성과 같은 다양한 작업에서도 우수성을 나타냅니다. 이러한 성능 개선은 FVDM이 생성 모델링 및 멀티미디어 애플리케이션에 중요한 영향을 미친다는 것을 시사합니다.



### Bridging the Gap between Text, Audio, Image, and Any Sequence: A Novel Approach using Gloss-based Annotation (https://arxiv.org/abs/2410.03146)
- **What's New**: BGTAI(Bridging the Gap between Text, Audio, Image, and any Sequence) 프레임워크를 소개하며, 이는 Text, Audio, Image 간의 불균형을 해결하는 혁신적인 접근법입니다.

- **Technical Details**: 이 연구에서는 Gloss 기반의 주석(annotation)을 활용하여 텍스트와 오디오 입력을 이미지와 정확하게 정렬하는 과정을 개선하고자 합니다. 새로운 DS-Net(Data-Pair Selection Network), Result Filter 모듈, SP-Loss 함수를 제안하고, Langue2Gloss 모델을 UniBriVL에 통합하여 공동 훈련을 진행합니다.

- **Performance Highlights**: 제안된 방식은 기존의 다중모달 모델보다 우수한 성능을 보이며, 이미지와 언어의 적합성 및 다중모달 표현 개선에 기여함을 입증했습니다.



### ARB-LLM: Alternating Refined Binarizations for Large Language Models (https://arxiv.org/abs/2410.03129)
Comments:
          The code and models will be available at this https URL

- **What's New**: 본 논문은 ARB-LLM이라는 새로운 1-bit 포스트 트레이닝 양자화(PTQ) 기술을 제안합니다. 이는 대형 언어 모델(LLM)에 최적화되었으며, 이 모델의 메모리 및 계산 요구량을 크게 줄일 수 있습니다.

- **Technical Details**: ARB-LLM은 교차 정제 양자화(Alternating Refined Binarization, ARB) 알고리즘을 기반으로 하여, 양자화 오차를 줄이고, 컬럼 그룹 비트맵(Column-Group Bitmap, CGB) 전략을 개선하여 성능을 향상시킵니다. ARB-X 및 ARB-RC와 같은 확장 기술로, 교정 데이터를 통합하고 컬럼 방향의 편차를 최소화합니다.

- **Performance Highlights**: 실험 결과, ARB-LLM$_{RC}$는 현재의 SOTA 이진 PTQ 방법들보다 훨씬 높은 성능을 보이며, 동일한 크기의 FP16 모델을 초월하는 성과를 거두었습니다. 또한, 이 알고리즘은 대규모 LLM의 실용적인 배포에 필요한 메모리 자원을 최소화합니다.



### MBDS: A Multi-Body Dynamics Simulation Dataset for Graph Networks Simulators (https://arxiv.org/abs/2410.03107)
- **What's New**: 본 논문에서는 신경망 기반의 물리 시스템 모델링과 시뮬레이션의 한계를 극복하기 위한 새로운 데이터셋인 Multi-Body Dynamics Simulation (MBDS) 데이터셋을 제안합니다. MBDS 데이터셋은 실제 세계의 복잡한 기계 구조를 모델링하며, 기존 데이터셋보다 더 많은 모션 궤적과 더 높은 시간 단계 수를 포함하고 있습니다.

- **Technical Details**: MBDS 데이터셋은 1D, 2D 및 3D 장면을 포함하여 총 150,000개의 동작 궤적과 복합 링크 구조를 제공하여 보다 복잡한 다중 바디 역학 시나리오의 시뮬레이션을 가능하게 합니다. 여기에는 질량, 마찰과 같은 물리적 특성을 포함한 정밀한 다중 바디 동역학 모델링이 포함됩니다.

- **Performance Highlights**: 시뮬레이션의 temporal horizon이 늘어날수록 예측 오차가 증가하는 경향을 보이며, 이는 현재 모델링 프레임워크의 견고성이 부족하다는 것을 강조합니다. 또한 기존 모델들은 특정 시나리오에서만 뛰어난 성능을 보이며, 그 시나리오가 변경될 경우 정확도가 급격히 감소하는 경향이 있습니다.



### Mamba in Vision: A Comprehensive Survey of Techniques and Applications (https://arxiv.org/abs/2410.03105)
Comments:
          Under Review

- **What's New**: Mamba는 컴퓨터 비전 내에서 Convolutional Neural Networks (CNNs)와 Vision Transformers (ViTs)의 한계를 극복하기 위한 새로운 접근 방식으로 등장하였습니다. Mamba는 또는 선형 계산 복잡도를 기반으로 한 Selective Structured State Space Models를 활용하여 장거리 의존성을 효과적으로 캡처하는 데 중점을 두었습니다.

- **Technical Details**: Mamba 모델은 입력 데이터에 기반하여 동적으로 조정되는 선택적 상태 표현을 사용하여 computational overhead를 줄이고 효율성을 높입니다. 이는 구조적 상태 공간 모델(SSMs)의 발전을 기반으로 하여 이루어지며, Mamba는 GPU를 최적화한 스캔 기반 알고리즘을 활용하여 기존의 convolution 기반 SSMs의 비효율성을 피합니다.

- **Performance Highlights**: Mamba 모델은 비디오 처리, 원격 감지, 의료 영상 등 다양한 분야에서 특히 유리하며, CNNs와 ViTs는 높은 계산 요구로 인해 확장성 문제를 겪는 반면, Mamba 모델은 시퀀스 길이에 대한 선형 확장성을 제공하여 실시간 및 대규모 애플리케이션에 적합합니다.



### Combing Text-based and Drag-based Editing for Precise and Flexible Image Editing (https://arxiv.org/abs/2410.03097)
Comments:
          12 pages, 9 figures

- **What's New**: 이번 논문에서는 텍스트 기반 및 드래그 기반 이미지 편집 기술의 단점을 분석하고, 이 두 가지 방법을 결합하여 더욱 정확하고 유연한 이미지 편집을 가능한 CLIPDrag라는 새로운 방법을 제안합니다.

- **Technical Details**: CLIPDrag는 텍스트 신호를 글로벌 가이드로, 드래그 포인트를 로컬 정보로 활용하며, 글로벌-로컬 모션 감독(Global-Local Motion Supervision, GLMS) 방식을 도입하여 텍스트 신호를 드래그 기반 방법에 통합합니다. 또한 빠른 포인트 추적(Fast Point Tracking, FPT) 방법을 통해 성능을 향상시킵니다.

- **Performance Highlights**: CLIPDrag는 기존의 단일 드래그 기반 방법이나 텍스트 기반 방법보다 우수한 성능을 보이며, 양적 및 질적으로 현저한 개선을 기록했습니다.



### Generative Edge Detection with Stable Diffusion (https://arxiv.org/abs/2410.03080)
- **What's New**: 새로운 Generative Edge Detector (GED) 모델을 제안하며, 이는 사전 훈련된 stable diffusion 모델의 잠재력을 최대한 활용합니다. multi-step denoising 과정을 생략하고 효율적인 학습 및 추론을 가능하게 합니다.

- **Technical Details**: GED는 latent image feature maps를 입력으로 받고, denoising U-Net을 미세 조정(finetune)하여 잠재적인 edge maps를 직접 예측합니다. 또한 edge의 주관성과 모호성을 수용하기 위해 edgess의 품질(혹은 granularity)를 고려합니다.

- **Performance Highlights**: 여러 데이터셋에 대한 실험을 통해 BSDS 테스트 데이터셋에서 ODS 및 OIS 기준으로 각각 0.870 및 0.880의 경쟁력 있는 성과를 달성했습니다.



### DocKD: Knowledge Distillation from LLMs for Open-World Document Understanding Models (https://arxiv.org/abs/2410.03061)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 이 논문에서는 Visual Document Understanding (VDU) 모델의 일반화 능력을 향상시키는 새로운 프레임워크인 DocKD를 제안합니다. 이 프레임워크는 LLM(large language models)에서의 지식을 증류합니다.

- **Technical Details**: DocKD는 LLM에 키-값 쌍(key-value pairs), 레이아웃(layouts), 설명(descriptions) 등의 다양한 문서 요소를 제공하여 열린 질문에 대한 답변을 유도합니다. 기존의 방법과 달리, 외부 문서 지식을 통합하는 데이터 생성 과정을 풍부하게 합니다.

- **Performance Highlights**: DocKD를 통해 생성된 문서 주석들은 고품질이며, 전통적인 지식 증류 접근법보다 우수한 성능을 보입니다. 특히, DocKD로 훈련된 학생 VDU 모델들은 인간 주석 데이터로 훈련된 모델과 비교할 때, 인도메인(task)에서는 유사한 성능을 유지하나, 아웃오브 도메인(out-of-domain task)에서는 훨씬 더 뛰어난 성능을 발휘합니다.



### DiffKillR: Killing and Recreating Diffeomorphisms for Cell Annotation in Dense Microscopy Images (https://arxiv.org/abs/2410.03058)
- **What's New**: DiffKillR는 세포 주석을 아키타입(Archetype) 매칭과 이미지 등록(Image Registration) 작업의 조합으로 재구성하는 혁신적인 프레임워크입니다.

- **Technical Details**: DiffKillR는 두 개의 상호 보완적인 신경망을 사용합니다. 첫 번째 네트워크는 diffeomorphism-invariant 특징 공간을 학습하는 DiffeoInvariantNet이며, 두 번째 네트워크는 주석 매핑을 위한 정확한 왜곡 필드를 계산하는 DiffeoMappingNet입니다. 이 방법으로 작은 세트의 아키타입 주석을 대규모 미세 사진 이미지에 효율적으로 전파합니다.

- **Performance Highlights**: DiffKillR는 세포 집계, 세포 방향 예측 및 소수의 세포 분할 등의 미세 사진 작업에서 성능을 검증하였으며, 기존의 감독(supervised), 준 감독(semi-supervised), 비 감독(unsupervised) 방법에 비해 우수한 성능을 보입니다.



### CLIP-Clique: Graph-based Correspondence Matching Augmented by Vision Language Models for Object-based Global Localization (https://arxiv.org/abs/2410.03054)
Comments:
          IEEE Robotics and Automation Letters

- **What's New**: 이 논문에서는 의미 있는 객체 랜드마크를 이용한 글로벌 로컬라이제이션 방법을 제안합니다. 특히, Vision Language Models (VLMs)를 활용하여 랜드마크의 식별력을 향상시키고, 그래프 이론적 접근법을 통해 인라이어(inliers) 추출을 결정론적으로 수행하는 방법이 포함되어 있습니다. 이 방법은 이전의 RANSAC 기반의 접근이 가진 취약점을 보완합니다.

- **Technical Details**: 제안된 방법인 CLIP-Clique는 의미 그래프(semantic graph)와 VLM을 결합하여 객체 대응(matching) 강화를 목적으로 합니다. 랜드마크에 대한 딥러닝 기반의 임베딩(embedding) 벡터를 부여하여 관측 및 맵 객체 간의 유사성을 계산하며, 호환성 그래프를 구축하여 최대 클리크(maximal cliques)를 찾습니다. 이를 통해 유사성이 높은 랜드마크 세트를 찾고, 관찰 완전성을 고려한 가중치 최소 제곱(weighted least squares) 방법으로 카메라 포즈(pose)를 계산합니다.

- **Performance Highlights**: 실험 결과, ScanNet 및 TUM 데이터셋을 통해 제안하는 방안인 CLIP-Clique가 기존 방법보다 매칭(matching) 및 포즈 추정(pose estimation)의 정확성을 향상시킴을 확인하였습니다. 이 방법은 랜드마크와 관측 간의 유사성 및 관찰의 완전성을 동시에 고려하여 더 강력한 로컬라이제이션 성능을 보여줍니다.



### AuroraCap: Efficient, Performant Video Detailed Captioning and a New Benchmark (https://arxiv.org/abs/2410.03051)
Comments:
          Code, docs, weight, benchmark and training data are all avaliable at \href{this https URL}{website}

- **What's New**: 이 논문에서는 AuroraCap이라는 대규모 다중 모달 모델 기반의 비디오 캡셔닝 모델을 제안합니다. 이 모델은 시간적 모델링을 위한 추가 매개변수 없이 가장 간단한 아키텍처 디자인을 따르며, 긴 비디오 시퀀스에서 발생하는 오버헤드를 줄이기 위해 토큰 병합(token merging) 전략을 구현합니다.

- **Technical Details**: AuroraCap은 비디오 캡셔닝 작업을 성능 저하 없이 실행하는데 필요한 입력 시각적 토큰의 수를 줄이기 위해 bipartite soft matching 알고리즘을 활용하여 Transformer 계층에서 유사한 토큰을 점진적으로 결합합니다. 이 방법을 통해 모델은 원래 ViT에서 생성된 시각적 토큰의 10%에서 20%만을 사용하면서도 다양한 벤치마크에서 비슷한 성능을 보입니다. 또한, VDC(Video Detailed Captions)라는 새로운 비디오 상세 캡셔닝 벤치마크를 개발하여 1,000개 이상의 고품질 비디오-캡션 쌍을 제공합니다.

- **Performance Highlights**: AuroraCap은 다양한 비디오 및 이미지 캡셔닝 벤치마크에서 우수한 성능을 보였으며, 예를 들어 Flickr30k에서 CIDEr 88.9를 얻어 GPT-4V(55.3)와 Gemini-1.5 Pro(82.2)를 능가했습니다. 또한 VDC 벤치마크는 비디오 상세 캡셔닝의 품질을 개선하는 새로운 평가 지표인 VDCscore를 도입하여 인간의 판단과 더 나은 상관관계를 보였습니다.



### Revealing the Unseen: Guiding Personalized Diffusion Models to Expose Training Data (https://arxiv.org/abs/2410.03039)
Comments:
          Under review

- **What's New**: 최근 Diffusion Models (DMs)의 발전으로 이미지 생성 및 개인화된 스타일 학습이 가능해졌습니다. 그러나 이러한 모델의 미세 조정(checkpoint)을 공유할 경우 데이터 유출과 저작권 침해 우려가 있습니다. 본 논문에서는 FineXtract라는 새로운 프레임워크를 제안하여 온라인에서 공유된 DMs로부터 훈련 데이터를 추출할 수 있는 방법을 모색합니다.

- **Technical Details**: FineXtract 방법은 사전 학습된 모델에서 미세 조정된 모델로의 학습 분포 변화를 모델링합니다. 이 과정에서 사전 학습 모델과 미세 조정 모델의 스코어 함수를 외삽(extrapolate)하여, 고밀도(high-density) 지역으로의 생성 과정을 유도합니다. 클러스터링(Clustering) 알고리즘을 적용하여 생성된 이미지 중 최상위 확률 이미지를 추출합니다. 이 방법은 조건부 및 비조건부 DMs에 모두 적용 가능합니다.

- **Performance Highlights**: WikiArt, DreamBooth 및 실세계 체크포인트를 포함한 여러 데이터셋에서 실험을 통해, 본 방법이 대개 20%의 미세 조정 데이터를 정확히 추출할 수 있음을 입증하였습니다. 이는 기존 방법 대비 월등한 성능을 증명합니다.



### Dynamic Sparse Training versus Dense Training: The Unexpected Winner in Image Corruption Robustness (https://arxiv.org/abs/2410.03030)
- **What's New**: 본 연구는 Dense Training(밀집 훈련)이 모델의 강건성을 최대화하는 전형적인 접근법이라는 일반적인 인식을 questioning(질문)합니다. 특히, Dynamic Sparse Training(동적 희소 훈련) 방법이 Dense Training보다 강건성 측면에서 일관되게 뛰어난 성능을 발휘할 수 있다는 주장을 합니다.

- **Technical Details**: 저자들은 다양한 Deep Learning(딥 러닝) 아키텍처를 사용하여 이미지와 비디오 데이터에서 3종의 동적 희소 훈련 알고리즘을 실험했습니다. 데이터에 대해 다양한 이미지 왜곡을 적용하여 DST가 제공하는 강건성 향상 효과를 분석하고, Dynamic Sparsity Corruption Robustness(DSCR) Hypothesis(가설)를 제안했습니다.

- **Performance Highlights**: DST 방법(예: SET)은 19가지 이미지 왜곡 중 18개의 경우에서 밀집 훈련보다 더 나은 강건성을 보였습니다. 일반적인 왜곡에 대한 DST의 뛰어난 성능을 통해 현재의 강건성 연구의 지경을 넘어서 새로운 가능성을 제시합니다.



### PixelShuffler: A Simple Image Translation Through Pixel Rearrangemen (https://arxiv.org/abs/2410.03021)
- **What's New**: 본 논문에서는 복잡한 신경망 아키텍처 없이 이미지 스타일 변환(image style transfer) 문제를 해결하는 단순한 픽셀 셔플(pixel shuffle) 방법을 제안합니다. 기존의 최신 기술들이 가지는 높은 계산 비용과 복잡성을 줄이며, 스타일 이미지의 색상을 보존하면서 내용 이미지의 구조적 세부 정보를 유지하는 데 초점을 맞추었습니다.

- **Technical Details**: 제안된 픽셀 셔플 알고리즘은 스타일 이미지의 픽셀을 섞어 해당 이미지와 내용 이미지 간의 상호 정보(mutual information)를 최대화하는 방식으로 작동합니다. 이 방법은 스타일 이미지의 색상 팔레트를 보존하고, 내용 이미지의 구조적 내용을 일치하게 하여 스타일과 내용을 모두 만족시키는 효과를 발휘합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 Learned Perceptual Image Patch Similarity (LPIPS) 손실과 Fréchet Inception Distance (FID) 점수 측면에서 기존의 최신 기술들과 비교해도 경쟁력 있는 성능을 보였으며, 복잡성을 대폭 줄이면서 효율적인 이미지 스타일 전환을 위한 유망한 대안을 제공합니다.



### Fully Automated CTC Detection, Segmentation and Classification for Multi-Channel IF Imaging (https://arxiv.org/abs/2410.02988)
Comments:
          Published in MICCAI 2024 MOVI Workshop Conference Proceedings

- **What's New**: 이 논문에서는 유방암 전이(mBCa) 환자에서 순환종양세포(CTC)를 자동으로 탐지하고 분류하는 완전 자동화된 머신러닝 기반 파이프라인인 BRIA(BReast cancer Imaging Algorithm)를 제안합니다.

- **Technical Details**: BRIA 파이프라인은 다채널 면역형광(IF) 이미지를 처리하고, CTC 탐지, 세포 및 핵 분할, 특성 추출을 포함합니다. 이 시스템은 9,533개의 세포에서 99%의 민감도와 97%의 특이도를 달성하였으며, 15명의 mBCa 환자 샘플을 기반으로 하였습니다.

- **Performance Highlights**: BRIA를 사용하여 환자당 평균 14M개의 세포를 335개의 CTC 후보로 줄였습니다. 전체 세포 확인과 분류에 필요한 시간을 크게 단축시키며, 자동화된 프로세스를 통해 임상의의 부담을 줄이고 정확성을 높였습니다.



### RSA: Resolving Scale Ambiguities in Monocular Depth Estimators through Language Descriptions (https://arxiv.org/abs/2410.02924)
- **What's New**: 새로운 방법론 RSA(Resolving Scale Ambiguities)를 제안하여 단일 이미지에서 메트릭 스케일로 깊이 추정이 가능해졌습니다. 이 방법은 이미지에 대한 텍스트 설명을 입력으로 사용하여 상대 깊이를 메트릭 깊이로 변환하는 선형 변환의 매개변수를 산출합니다.

- **Technical Details**: 상대 깊이 예측을 메트릭 깊이로 변환하기 위해, RSA는 입력으로 제공된 텍스트 캡션을 기반으로 선형 변환의 매개변수를 예측하는 방법론입니다. 이 모델은 단일 이미지(모노큘러)에서 깊이 지도를 밀집으로 예측하며, NYUv2 및 KITTI와 같은 데이터셋에서 실험을 진행했습니다.

- **Performance Highlights**: RSA 방법은 상대 깊이에서 메트릭 깊이로의 변환에서 우수한 성능을 나타내며, 다양한 데이터셋에서 상반된 일반화 능력을 보여줍니다. RSA는 상대 깊이를 기반으로 하는 일반적 정렬 모듈로 작용하며, 관측된 결과는 해당 종류의 작업에서 사용할 수 있는 능력을 갖추고 있습니다.



### AirLetters: An Open Video Dataset of Characters Drawn in the Air (https://arxiv.org/abs/2410.02921)
Comments:
          ECCV'24, HANDS workshop

- **What's New**: AirLetters는 실제 환경에서 생성된 사람 손의 동작을 포함한 새로운 비디오 데이터셋으로, 공중에서 사람에 의해 그려진 글자 예측을 요구합니다. 기존 비디오 데이터셋과 달리 AirLetters는 동작 패턴을 알아내고, 시간에 따른 장기적 정보를 통합하는 데 중점을 둡니다.

- **Technical Details**: AirLetters는 161,652개의 라벨이 지정된 비디오로 구성되어 있으며, 이는 라틴 알파벳의 숫자와 글자에 해당하는 인간 손의 동작을 캡처합니다. 전반적으로 이 데이터셋은 기존의 손 제스처 데이터셋보다 더 도전적이며, 모델이 손을 정확하게 추적하고 장기적 종속성을 분석하도록 요구합니다. 모든 라벨은 동적이며 몇 개의 중요 프레임만으로 추론할 수 없습니다. 데이터셋은 조명 조건, 손 위치, 배경, 그리기 동작 및 기타 신체 동작의 상당한 변화를 포함하고 있습니다.

- **Performance Highlights**: 대규모 이미지 및 비디오 이해 모델에 대한 실험 결과는 AirLetters에서 이러한 방법들이 인간 기준선에 비해 아주 저조한 성능을 보였음을 보여줍니다. 이 연구는 비디오 이해에 대한 최근의 진전에도 불구하고, 복잡한 동작 표현을 정확하게 수행하는 것이 여전히 열려 있는 문제라는 것을 강조합니다.



### Task-Decoupled Image Inpainting Framework for Class-specific Object Remover (https://arxiv.org/abs/2410.02894)
- **What's New**: 이 논문에서는 객체 제거(object removal) 작업의 성능 개선을 위해 작업을 분리한 이미지 인페인팅 프레임워크를 제안합니다. 기존의 접근방식은 단일 이미지 인페인팅 모델이 객체 제거와 복원 작업을 동시에 처리하도록 학습시켜 성능 저하를 가져오는 문제점을 발견했습니다.

- **Technical Details**: 제안된 프레임워크는 두 개의 별도의 인페인팅 모델을 생성합니다. 하나는 객체 복원(object restoration)을 위한 객체 복원기(object restorer)이고, 다른 하나는 객체 제거(object removal)를 위한 객체 제거기(object remover)입니다. 객체 복원기는 제거 대상의 일부를 덮는 마스크를 사용하여 훈련되고, 객체 제거기는 복원기로부터 제공받은 가이드를 사용하여 훈련됩니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, 제안된 클래스별 객체 제거기는 기존의 이미지 인페인팅 네트워크 기반의 객체 제거기보다 목표 클래스 객체를 더 효과적으로 제거하는 성능을 보였습니다.



### Investigating the Impact of Randomness on Reproducibility in Computer Vision: A Study on Applications in Civil Engineering and Medicin (https://arxiv.org/abs/2410.02806)
- **What's New**: 본 논문은 CUDA에 의해 발생하는 무작위성이 기계 학습에서 재현성(Reproducibility) 문제에 미치는 영향을 분석합니다. CUDA는 GPU에서 알고리즘 실행을 가속화하지만, 여러 실행 간 비결정적(Non-deterministic) 행동이 발생할 수 있습니다. 이 연구는 이 문제의 중요성을 강조하며, 다양한 데이터셋에서 성능 점수에서 최대 4.77%의 차이를 초래할 수 있음을 보여줍니다.

- **Technical Details**: 우리는 CIFAR-10, SDNET2018, CBIS-DDSM 데이터셋을 사용하여 CUDA 무작위성의 영향을 분석했습니다. 다양한 하이퍼파라미터 구성에 대해 결정적(Deterministic) 및 비결정적 설정에서 실험을 수행했습니다. ADAM 및 SGD와 같은 두 가지 최적화 알고리즘을 채택하였고, CUDA로 인한 무작위성의 영향을 평가하기 위해 480개의 실험을 수행했습니다. 각 데이터셋에 대해 고정된 시드(Configuration)를 사용하여 무작위성을 통제했습니다.

- **Performance Highlights**: 실험 결과, CIFAR-10에서는 SGD가 ADAM보다 유사한 정확도로 수렴했습니다. SDNET2018에서의 성능 지표가 향상되었고, CBIS-DDSM 데이터셋에서는 AUC 점수가 주요 메트릭으로 보고되었습니다. 결정적 실행의 영향은 성능 변동성을 줄이는데 기여했으며, 최대 4.77%의 성능 차이를 나타내었습니다. ADAM 최적화기에서 F1-score의 표준 편차가 가장 크게 나타나는 등, 무작위성의 영향이 뚜렷하게 드러났습니다.



### Leveraging Retrieval Augment Approach for Multimodal Emotion Recognition Under Missing Modalities (https://arxiv.org/abs/2410.02804)
Comments:
          Under reviewing

- **What's New**: 이번 논문에서 제안하는 RAMER(Recovery Augmentation for Missing Modality Multimodal Emotion Recognition) 프레임워크는 누락된 모달리티(missing modalities) 상황에서 감정 인식을 개선하기 위해 유사한 다중 모달 감정 데이터를 활용하는 접근 방식을 도입합니다.

- **Technical Details**: RAMER는 다양한 누락 조건 하에서 감정 특징을 검색하여 감정 정보를 보충하는 데 중점을 둡니다. 이를 위해 감정 특징 데이터베이스를 구성하고 관련 다중 모달 감정 데이터를 불러들여 누락된 정보의 공백을 메우게 됩니다.

- **Performance Highlights**: 다양한 실험 결과에 따르면, RAMER 프레임워크는 기존의 최신 방법들에 비해 누락된 모달리티 MER 과제에서 월등한 성능을 보였습니다. 이러한 결과는 감정 인식의 견고함과 정확성을 향상시킬 수 있는 방법론적 기여를 보여줍니다.



### Estimating Body Volume and Height Using 3D Data (https://arxiv.org/abs/2410.02800)
Comments:
          6 pages

- **What's New**: 이 논문은 3D 이미징 기술을 활용하여 비침습적으로 체중을 추정하는 새로운 방법을 제시합니다. RealSense D415 카메라를 사용하여 환자의 고해상도 깊이 맵을 캡처하고, Convex Hull Algorithm을 통해 총 체적을 계산합니다.

- **Technical Details**: 제안된 방법은 3D 모델에서 신체의 주요 포인트 간의 거리를 식별하여 신장(Height)을 추출하고, 포인트 클라우드 데이터를 여러 섹션으로 분리해 개별 볼륨을 합산하여 총 체적을 계산합니다. 이러한 데이터 처리는 MobileNet 및 ResNet과 같은 신경망 아키텍처를 이용하여 신체 구성 요소를 분할 및 분석합니다.

- **Performance Highlights**: 이 비침습적인 체중 추정 방법은 긴급 상황에서 신뢰할 수 있는 체중 데이터를 제공하여 의료 개입의 정확성을 향상시킬 가능성을 보여줍니다. 이 연구는 전문 의료 환경에서 체중 추정을 위한 3D 카메라 기술의 필요성을 강조합니다.



### Logic-Free Building Automation: Learning the Control of Room Facilities with Wall Switches and Ceiling Camera (https://arxiv.org/abs/2410.02789)
Comments:
          5 pages, 3 figures, 2 tables

- **What's New**: 본 논문에서는 Reinforcement Learning(RL)에 의존하지 않고, 사용자의 선호도를 기반으로 하는 Logic-Free Building Automation(LFBA) 아키텍처를 제안합니다. 이 시스템은 기존의 건물 자동화에 비해 더 직관적이고 사용자 친화적인 방식을 제공합니다.

- **Technical Details**: LFBA는 ceiling camera와 wall switches를 활용하여 사용자의 제어 방식을 직접 모니터링하고 학습합니다. 이 연구에서는 VGG, ResNet, Vision Transformer(ViT) 같은 다양한 Deep Learning(DL) 모델을 사용하여 성능을 평가하였으며, VGG 모델이 가장 높은 제어 정확도를 기록했습니다.

- **Performance Highlights**: LFBA 시스템은 다양한 조건과 사용자 활동에 대한 테스트 결과, 93%-98%의 제어 정확도를 달성하였으며, 이는 기존의 DL 모델들보다 뛰어난 성능입니다. 또한 해당 시스템은 향후 더 다양한 스마트 빌딩 응용 프로그램으로 확장 가능성을 보여줍니다.



### RoMo: A Robust Solver for Full-body Unlabeled Optical Motion Captur (https://arxiv.org/abs/2410.02788)
Comments:
          Siggraph Asia 2024 Conference Paper

- **What's New**: 이 논문에서는 RoMo라는 새로운 학습 기반 프레임워크를 소개합니다. RoMo는 raw optical motion capture (MoCap) 데이터를 견고하게 라벨링하고 문제를 해결하기 위해 설계되었습니다. RoMo는 라벨링 단계에서 divide-and-conquer 전략을 도입하여 복잡한 전체 신체 라벨링 문제를 관리 가능한 하위 작업으로 분할합니다.

- **Technical Details**: RoMo는 K-partite graph 기반 클러스터링 알고리즘을 사용하여 마커 트랙렛을 생성하고, 동적인 포즈 추정에 필요한 정확한 joint positions를 유지합니다. 또한, hybrid inverse kinematics 솔버를 도입하여 kinematic chain을 따라 발생할 수 있는 오류 누적을 방지합니다. 이 시스템은 전체 바디와 손 마커의 데이터를 처리하는 데 있어 독특한 데이터 분포를 처리할 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, RoMo는 다양한 메트릭과 데이터셋에서 매우 높은 라벨링 및 해결 정확도를 달성했습니다. 예를 들어, 실제 데이터셋에서는 손 라벨링의 F1 점수를 0.94에서 0.98로 개선하고, 하체 동작 해결에서 관절 위치 오류를 25% 줄였습니다. 또한 RoMo는 상업적 시스템으로는 불가능한 시나리오에서도 적용 가능합니다.



### Navigation with VLM framework: Go to Any Languag (https://arxiv.org/abs/2410.02787)
Comments:
          under review

- **What's New**: 이번 논문은 Vision Large Language Models (VLMs)를 활용해 인간과 유사한 방식으로 탐색하며 어떤 언어 목표에도 도달할 수 있는 Navigation with VLM (NavVLM)이라는 프레임워크를 소개합니다. 이 프레임워크는 미리 훈련된 모델 없이도 에이전트가 환경 정보를 인식하고 길을 안내받아 목표를 향해 탐색할 수 있도록 합니다.

- **Technical Details**: NavVLM 프레임워크는 에이전트가 특정 또는 비특정 언어 목표를 지향해 탐색할 수 있도록 구성됩니다. 이 시스템은 cognitive core로서 VLM을 사용하여 환경을 인식하고, 목표와 가까워지면 탐색을 종료하고 VLM의 지침에 따라 탐색을 이어갑니다. 중요한 기술 요소로 SLAM (Simultaneous Localization and Mapping) 모듈과 경로 계획 모듈이 있으며, 이를 통해 에이전트는 탐색 중 장애물을 피하면서 실시간으로 업데이트되는 지도 기반으로 행동하게 됩니다.

- **Performance Highlights**: NavVLM은 기존의 특정 목표 설정에 대해 성공률(SR)과 경로 길이로 가중된 성공률(SPL) 모두에서 최첨단 성능을 달성하였습니다. 또한, 비특정 언어 목표에서도 뛰어난 탐색 능력을 보여줍니다. 다채로운 환경에서의 평가 결과, 환경에 대한 깊이 있는 이해와 탐색의 인간적 접근 방식을 성공적으로 구현했습니다.



### Robust Symmetry Detection via Riemannian Langevin Dynamics (https://arxiv.org/abs/2410.02786)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 기존의 대칭 탐지 방법론에 클래식 기법과 최신 생성 모델(generative modeling)의 발전을 결합한 새로운 대칭 탐지 방법을 제안합니다. 특히, Langevin dynamics를 활용하여 노이즈에 대한 강인성을 향상시킴으로써 대칭 탐지의 효용을 극대화하고자 했습니다.

- **Technical Details**: 이 새로운 방법은 재정의된 대칭 공간(symmetry space)에서 미분 방정식을 기반으로 한 Langevin dynamics를 적용하여, 노이즈 환경에서도 효율적으로 포괄적인(global) 및 부분(partial) 대칭을 탐지할 수 있습니다. 기존의 방법들과 달리 데이터로부터 자가 학습 없이도 향상된 노이즈 저항성을 보여줍니다.

- **Performance Highlights**: 제안된 대칭 탐지 알고리즘은 다양한 형태에 대한 임상 실험 결과를 통해 노이즈에 대한 강인성을 입증하며, 2D 및 3D 형태 모두에 대한 대칭성을 효과적으로 탐지할 수 있는 능력을 가집니다. 결과적으로, 이 알고리즘은 대칭 탐지의 강력한 가능성을 제시합니다.



### Guess What I Think: Streamlined EEG-to-Image Generation with Latent Diffusion Models (https://arxiv.org/abs/2410.02780)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 뇌파로부터 이미지를 생성하는 기술이 주목받고 있으며, 이는 brain-computer interface (BCI) 시스템을 발전시키는 데 기여할 수 있다. 기존 연구들은 주로 고해상도 이미지를 생성할 수 있는 fMRI에 의존했으나, 이 연구에서는 저렴하고 비침습적인 electroencephalography (EEG)를 활용하여 실시간 BCI 응용에 적합한 새로운 방법을 제안하고 있다.

- **Technical Details**: 본 논문에서는 ControlNet 어댑터를 기반으로 한 효율적인 latent diffusion model (LDM) 프레임워크를 제안한다. 이 방법은 EEG 신호를 조건화하여 이미지 생성 과정을 간소화하며, 복잡한 전처리나 다양한 손실 함수, 또는 캡셔닝 모델 없이도 이미지를 생성할 수 있다. 실험을 통해 최첨단 모델들과 비교하여 우수한 성능을 입증하였다.

- **Performance Highlights**: 제안된 GWIT 프레임워크는 minimal preprocessing과 효율적인 트레이닝을 통해 EEG로부터 이미지를 생성하는 데 성공하였다. 본 연구는 EEG 데이터를 사용하여 image generation을 위한 ControlNet의 최초 응용을 보여주며, 기존의 GAN 기반 방법들보다 뛰어난 성능을 보였다.



### Mind the Uncertainty in Human Disagreement: Evaluating Discrepancies between Model Predictions and Human Responses in VQA (https://arxiv.org/abs/2410.02773)
- **What's New**: 이번 연구는 Visual Question Answering (VQA) 작업에서 최첨단 비전-언어 모델이 인간 응답의 분포와 얼마나 잘 일치하는지 종합적으로 평가하고, 인간의 불확실성을 (Human Uncertainty in Disagreement, HUD) 고려하는 방법을 제안합니다.

- **Technical Details**: VQA 작업에서 HUD의 영향을 분석하기 위해, 샘플을 낮은, 중간, 높은 3가지 불확실성 수준으로 분류하였으며, 전통적인 정확도 외에도 총 변이 거리 (Total Variation Distance, TVD), Kullback-Leibler 발산 (KL), 인간 엔트로피 보정 오차 (Human Entropy Calibration Error, EntCE) 등의 새로운 지표를 사용하였습니다. 연구 결과, BEiT3와 같은 최신 모델도 다양한 인간 응답의 다중 레이블 분포를 포착하는 데 어려움을 겪고 있다는 것을 확인했습니다.

- **Performance Highlights**: 종합적으로 우리가 제안한 모델 보정 방법은 모델의 신뢰도를 인간의 불확실성과 더 잘 일치시킴으로써 VQA 성능을 향상시킬 수 있음을 보여주었습니다. 연구의 주요 기여는 HUD를 명시적으로 활용하여 VQA 모델의 인간 응답 분포와의 차이를 평가한 것입니다.



### Complex-valued convolutional neural network classification of hand gesture from radar images (https://arxiv.org/abs/2410.02771)
Comments:
          173 pages, 36 tables, 50 figures

- **What's New**: 이번 연구에서는 복소수 (Complex) CNN을 제안하여 손 제스처 인식 시스템의 성능을 향상시키고자 합니다. 기존의 실수 (Real) 기반 알고리즘의 한계를 극복하기 위해 모든 구성 요소가 복소수 도메인에서 작동하도록 설계되었습니다.

- **Technical Details**: 제안된 CV-CNN은 기존의 MLP, CNN, RNN과 같은 다양한 딥러닝 아키텍처의 빌딩 블록을 포함하고 forward 및 backward 연산과 미분(differentiation)을 복소수 도메인에서 수행합니다. 이를 통해 기존의 실수 기반 모델의 두 배의 차원을 요구하는 문제를 해결합니다.

- **Performance Highlights**: 복소수 CNN 모델은 두 세트의 복소수 손 제스처 레이더 이미지 데이터셋에서 실험하였으며, 해당 결과는 기존의 실수 모델과 비교하여 우수한 분류 성능을 입증하였습니다. 또한 복소수 forward residual network를 제안하여 이진 분류에서도 성능을 개선했습니다.



### BoViLA: Bootstrapping Video-Language Alignment via LLM-Based Self-Questioning and Answering (https://arxiv.org/abs/2410.02768)
- **What's New**: 이번 논문에서는 비디오-텍스트 쌍의 주석 작업의 비효율성을 해결하기 위해 LLM 기반의 셀프-질문 및 답변(self-questioning and answering) 프레임워크인 BoViLA를 제안합니다. 이 접근법은 비디오 정보와 LLM 내부 지식을 더 효과적으로 활용하여 모달리티 정렬(modality alignment)을 개선하는 데 기여합니다.

- **Technical Details**: BoViLA는 두 개의 역할을 포함하고 있습니다: 질문자(questioner)와 답변자(answerer)로서, 두 역할은 동일한 모델이 수행하며, 셀프 질문 및 답변을 통해 서로 개선합니다. 또한, 저자는 저품질의 질문을 필터링하기 위해 EDL(Evidential Deep Learning)을 활용하여 불확실성을 추정하고 질문 품질을 평가합니다.

- **Performance Highlights**: BoViLA는 STAR, How2QA, DramaQA, TVQA, VLEP 등 5개의 VideoQA 벤치마크에서 여러 최첨단 방법들보다 우수한 성과를 보이며, 단 4.5M의 훈련 가능한 파라미터로도 효과적임을 입증했습니다.



### GenSim2: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs (https://arxiv.org/abs/2410.03645)
Comments:
          CoRL 2024. Project website: this https URL

- **What's New**: 이번 연구에서는 GenSim2라는 확장 가능한 로봇 시뮬레이션 프레임워크를 제안합니다. 이 프레임워크는 복잡하고 현실적인 시뮬레이션 작업 생성을 위해 다중 모달 및 추론 능력을 갖춘 코딩 LLM을 활용합니다.

- **Technical Details**: GenSim2는 200개의 객체로 최대 100개의 관절 작업에 대한 데이터를 자동 생성할 수 있는 계획(solvers) 및 강화학습(RL) 솔버들을 사용하여 객체 카테고리 내에서 일반화합니다. 제안된 다중 작업 언어 조건화 정책 아키텍처인 proprioceptive point-cloud transformer (PPT)는 생성된 데모에서 학습하여 시뮬레이션에서 현실로의 제로샷 전이가 뛰어납니다.

- **Performance Highlights**: GenSim2는 생성된 데이터를 사용하여 제로샷 전이 또는 현실 세계에서 수집된 데이터와 공동 학습할 수 있는 가능성을 보여줍니다. 이는 제한된 실제 데이터에서만 훈련했을 때보다 정책 성능을 20% 향상시킬 수 있습니다.



### HyperCMR: Enhanced Multi-Contrast CMR Reconstruction with Eagle Loss (https://arxiv.org/abs/2410.03624)
Comments:
          MICCAI 2024 STACOM-CMRxRecon

- **What's New**: 이번 연구에서는 multi-contrast MRI 촬영을 위한 HyperCMR이라는 새로운 프레임워크가 제안되었습니다. 특히 Eagle Loss라는 고급 손실 함수를 도입하여 저표본 k-space에서 고주파 정보를 회복하는 데 중점을 두었습니다.

- **Technical Details**: HyperCMR는 기존의 PromptMR 모델을 기반으로 하며, 데이터 충실도 손실(Data Fidelity Loss), 구조적 유사성 손실(SSIM Loss), Eagle Loss, VGG 지각 손실(VGG Perceptual Loss), 정규화 손실(Regularization Loss) 등의 조합을 사용하여 CMR 이미지의 고주파 세부사항을 보존합니다. 이 프레임워크는 10개의 코일을 활용하여 입력 데이터 형식을 확장하고, SensNet을 통해 감도 맵을 추정합니다. 또한 가우시안 고역 통과 필터 대신 버터워스 고역 통과 필터를 사용하여 고주파 정보를 보존합니다.

- **Performance Highlights**: HyperCMR는 CMRxRecon2024 챌린지 데이터셋에서 여러 평가 지표에서 기준 모델을 지속적으로 초과하며, SSIM 및 PSNR 점수에서 뛰어난 성능을 기록했습니다.



### BodyShapeGPT: SMPL Body Shape Manipulation with LLMs (https://arxiv.org/abs/2410.03556)
Comments:
          Accepted to ECCV 2024 Workshop on Foundation Models for 3D Humans. Code repository: this https URL

- **What's New**: 이 연구는 사전 훈련된 Large Language Models (LLMs)을 활용하여 사람의 물리적 특성을 이해하고, 이를 기반으로 SMPL-X 모델을 사용해 정확한 아바타를 생성하는 방법을 제안합니다. 특히 LLMs가 자연어를 통해 3D 인간 형태를 조작할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 본 연구는 LLMs의 미세 조정을 통해 SMPL-X 형태 매개변수를 정확하게 인식하고 생성하는 데 초점을 맞춥니다. 새로운 데이터셋을 생성하여 아바타의 물리적 속성과 관련된 언어적 설명을 포괄적으로 라벨링 했습니다. 모델 학습에 Low-Rank Adaptation (LoRA)와 quantization 기법을 적용하여 NVidia RTX-4090에서 효율적으로 훈련할 수 있도록 최적화했습니다.

- **Performance Highlights**: 본 연구의 결과는 LLMs의 조정된 아키텍처가 다양한 아바타 형상을 성공적으로 생성할 수 있으며, 이는 스토리텔링 및 가상 캐릭터 생성 분야에서의 활용 가능성을 보여줍니다. 최종 아바타는 기대되는 범위 내에서 생성되며, 이는 인간-기계 상호작용을 향상시키는 데 기여할 것입니다.



### Enhancing Autonomous Navigation by Imaging Hidden Objects using Single-Photon LiDAR (https://arxiv.org/abs/2410.03555)
Comments:
          Project webpage: this https URL

- **What's New**: 본 논문은 비가시선(Non-Line-of-Sight, NLOS) 감지를 통해 자율 내비게이션(autonomous navigation)의 시야를 개선하는 새로운 접근 방식을 제안합니다. 단일 광자 LiDAR(single-photon LiDAR)를 활용하여 로봇이 "코너를 돌아볼" 수 있는 기능을 통해 인식 범위를 확장하고, 복잡한 환경에서의 안전성을 높입니다.

- **Technical Details**: 제안된 방법은 세 가지 모듈로 구성된 파이프라인을 포함합니다: 1) Sensing(감지) - SPAD 기반 LiDAR를 사용하여 다중 반사 히스토그램(multi-bounce histograms)을 캡처; 2) Perception(인지) - 이 히스토그램을 사용하여 숨겨진 영역의 점유 맵(occupancy maps)을 추정하는 컨볼루션 신경망(convolutional neural network); 3) Control(제어) - 추정된 점유 맵을 바탕으로 로봇이 안전한 경로를 따를 수 있도록 합니다.

- **Performance Highlights**: 제안된 NLOS 감지 방법은 L자형 복도에서 숨겨진 장애물을 피하며 navigation 실험에서 LOS(라인 오브 사이트) 전용 감지보다 50% 이상 이동 시간을 단축하고, 경로 길이를 33% 단축하여 성능을 크게 향상시켰습니다. 이는 자율 로봇이 복잡한 환경에서 보다 안전하고 효율적으로 작동할 수 있는 토대를 마련합니다.



### Dreamming User Multimodal Representation for Micro-Video Recommendation (https://arxiv.org/abs/2410.03538)
- **What's New**: 이번 연구에서는 DreamUMM (Dreaming User Multi-Modal Representation)이라는 새로운 접근 방식을 제안하여 실시간 사용자 관심사를 다중 모드(multimodal) 공간에서 모델링합니다. 이는 사용자 역사 데이터를 활용하여 동적 관심사를 포착하는 데 중점을 두고 있습니다.

- **Technical Details**: DreamUMM은 사용자의 역사적 상호작용을 바탕으로 사용자의 비디오 선호도와 다중 모드 유사성을 연관짓는 폐쇄형 솔루션을 사용합니다. 또한, Candidate-DreamUMM을 통해 최근 사용자 행동 데이터가 부족한 상황에서도 후보 비디오에서 관심사를 추론할 수 있도록 설계되었습니다. 이 과정에서 대형 언어 모델(large language models)과 지식 증류(knowledge distillation)를 활용하여 비디오의 복잡한 시각적, 청각적, 텍스트 요소 간의 상호작용을 포착하는 고품질 다중 모드 임베딩을 생성합니다.

- **Performance Highlights**: 광범위한 온라인 A/B 테스트를 통해 사용자 참여 지표(활동 일수 및 재생 수 등)에서 눈에 띄는 개선이 나타났습니다. DreamUMM은 수억 명의 일간 활성 사용자(daily active users)를 가진 두 개의 마이크로 비디오 플랫폼에 성공적으로 배포되어 실용성과 확장성을 입증했습니다.



### Computer Vision Intelligence Test Modeling and Generation: A Case Study on Smart OCR (https://arxiv.org/abs/2410.03536)
- **What's New**: AI 기반 시스템의 품질 평가가 중요해지면서, 본 논문에서는 AI 소프트웨어의 기능 테스트 모델을 제안합니다. 기존 문헌에 대한 포괄적인 리뷰를 제공하고, 이미지 기반 텍스트 추출 AI 기능을 평가하기 위한 3D 분류 모델을 소개합니다.

- **Technical Details**: AI 소프트웨어 테스트는 전통적인 테스트와 달리 대규모 비구조적 입력 데이터, 예측 불가능한 시나리오, 시스템 출력의 불확실성 등을 고려해야 합니다. 이를 해결하기 위해, Modified CER와 WER, METAMORPHIC Testing(메타모픽 테스트) 기법 및 다양한 테스트 프레임워크가 활용됩니다. 또한, OCR 성능 평가는 일반적으로 Ground Truth(GT)와의 편집 거리 비교로 이루어집니다.

- **Performance Highlights**: 제안된 테스트 모델은 모바일 Optical Character Recognition(OCR) 사례를 통해 검증되었으며, 다양한 평가 지표로 AI 기능 품질을 효과적으로 평가할 수 있음을 보여줍니다. 연구는 AI 소프트웨어 테스트에 대한 체계적인 접근 방식을 제공하며, 향후 질문과 연구 방향을 제시합니다.



### FedStein: Enhancing Multi-Domain Federated Learning Through James-Stein Estimator (https://arxiv.org/abs/2410.03499)
Comments:
          12 pages, 2 figures. Accepted at International Workshop on Federated Foundation Models In Conjunction with NeurIPS 2024 (FL@FM-NeurIPS'24)

- **What's New**: 이번 연구에서는 Multi-Domain Federated Learning(FL)에서의 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. FedStein이라는 방법을 통해 James-Stein Estimator를 활용하여 Batch Normalization 통계를 공유하고, 데이터의 다양성을 극복하여 성능 향상에 기여합니다.

- **Technical Details**: FedStein은 클라이언트 간에 Batch Normalization(BN) 통계의 James-Stein 추정값만 공유하므로, 기존의 방법들보다 데이터의 독립성과 정규성에 대한 문제 해결을 돕습니다. Non-BN 레이어의 매개변수는 일반적인 Federated Learning 기법을 이용하여 교환되며, 이 과정에서 모델의 성능을 유지합니다.

- **Performance Highlights**: FedStein은 3개의 데이터셋과 여러 모델에서 실험을 통해 기존의 FedAvg 및 FedBN 방법들과 비교하였을 때, 특정 도메인에서 14% 이상의 정확도 향상을 기록하였으며, 이러한 결과는 다양한 도메인 일반화에 긍정적인 영향을 미쳤습니다.



### Diffusion State-Guided Projected Gradient for Inverse Problems (https://arxiv.org/abs/2410.03463)
Comments:
          preprint. under review. RZ and BT have equal contributions

- **What's New**: Diffusion State-Guided Projected Gradient (DiffStateGrad)라는 새로운 접근 방식을 제안하여, 데이터 매니폴드(data manifold) 상에 남아있도록 하여 역문제(inverse problems)를 해결하는 데 있어 확산 모델의 성능 및 견고성을 향상시킵니다.

- **Technical Details**: DiffStateGrad는 측정 가이던스.gradient을 저차원 저랭크 서브스페이스(low-rank subspace)에 투영하는 과정을 포함하며, 이 서브스페이스는 확산 과정의 중간 상태(intermediate state)를 근사합니다. 이 과정은 singular value decomposition (SVD)를 통해 수행되며, 저차원 투영을 통해 지역 매니폴드 구조에 수직한 방향을 제거합니다.

- **Performance Highlights**: DiffStateGrad는 확산 모델의 시간 가이던스 스텝 크기(measurement guidance step size)와 노이즈에 대한 견고성을 개선하며, 조건부 화소 결합을 사용하여 이미지 복원 시의 성능을 상승시킵니다. 예를 들어, 큰 스텝 크기와 높은 측정 노이즈 상황에서 PSNR이 20 미만으로 떨어지는 실패율을 크게 줄였습니다.



### Towards Real-time Intrahepatic Vessel Identification in Intraoperative Ultrasound-Guided Liver Surgery (https://arxiv.org/abs/2410.03420)
Comments:
          MICCAI 2024, Oct 2024, Marrakech, Morocco

- **What's New**: 이번 연구에서는 복강경( laparoscopic ) 간절제술에서 간의 내부 구조를 효과적으로 식별하고, 환자 개인 맞춤형 접근 방식을 통해 3D 초음파( ultrasound )를 기반으로 한 AI 모델을 개발하였습니다.

- **Technical Details**: 사전 수술 3D 초음파 간 용적을 사용하여 심층 학습( deep learning ) 모델을 훈련시켜 실시간( real-time )으로 간문맥( portal tree )과 가지 구조( branch structures )를 식별하는 방법을 제안하였습니다.

- **Performance Highlights**: 개인 맞춤형 AI 모델은 ex vivo( 생체 외 ) 돼지 간을 대상으로 검증되었으며, 외과의사에 비해 정확도( precision )는 0.95, 재현율( recall )은 0.93으로 우수한 성능을 보였습니다. 이는 초음파 기반의 간 절제술에서 정밀한 혈관 식별을 위한 기초를 마련합니다.



### An Enhanced Harmonic Densely Connected Hybrid Transformer Network Architecture for Chronic Wound Segmentation Utilising Multi-Colour Space Tensor Merging (https://arxiv.org/abs/2410.03359)
- **What's New**: 이 논문은 만성 상처(segmentation) 처리에 대한 새로운 접근법을 제시하며, 초기 레이어에서 대비를 제거하는 구성요소를 통합한 HarDNet 아키텍처를 개선했습니다. 이 연구는 특히 어두운 피부색을 가진 사례를 대상으로 한 최초의 만성 상처 세분화 연구입니다.

- **Technical Details**: 논문에서는 HarDNet(segmentation architecture) 모델을 사용하여 다양한 색상 공간(tensor merging process)에서 특성을 최적화하고, 합성곱 블록의 조화를 조정하여 성능을 향상시킵니다. 훈련 데이터는 밝은 피부를 가진 환자들의 상처 이미지를 사용하고, 어두운 피부색이 있는 두 개의 테스트 세트에서 모델을 평가합니다.

- **Performance Highlights**: 어두운 피부톤 세트에서 Dice 유사도 계수(Dice similarity coefficient)가 +0.1221, 교차 비율(intersection over union)이 +0.1274 향상되었으며, 임상 전문가들로부터 받은 주관 평가에서도 제안된 모델이 기존 모델보다 3% 이상의 개선을 보였습니다.



### Audio-Agent: Leveraging LLMs For Audio Generation, Editing and Composition (https://arxiv.org/abs/2410.03335)
- **What's New**: 이번 논문에서는 Audio-Agent라는 멀티모달 프레임워크를 소개합니다. 이 프레임워크는 텍스트 또는 비디오 입력 기반으로 오디오 생성과 편집을 수행합니다. 기존의 텍스트-오디오(TTA) 생성 방식이 복잡한 텍스트 조건에 대해 성능이 제한된 반면, Audio-Agent는 입력을 세분화하여 고품질 오디오를 생성할 수 있도록 설계되었습니다.

- **Technical Details**: Audio-Agent는 GPT-4와 사전 훈련된 TTA 확산 네트워크를 결합하여 오디오 생성 과정을 다단계를 통해 수행합니다. 특히 비디오-오디오(VTA) 작업에 있어, Gemma2-2B-it라는 대형 언어 모델을 사용하여 비주얼 입력을 의미론적 토큰으로 변환하며, 이러한 정보를 사용해 두 가지 모달리티 간의 격차를 연결합니다. TTA와 VTA 모두에서 효율적으로 작동하며, 기존의 복잡한 훈련 과정 없이도 높은 성능을 발휘할 수 있도록 최적화되었습니다.

- **Performance Highlights**: Audio-Agent는 긴 복잡한 텍스트 입력에 대해 고품질 오디오를 생성할 수 있으며, 광범위한 평가에서 기존의 특정 작업 모델들과 유사한 성능을 보여주었습니다. 이는 특히 영화 더빙 및 음악 작곡과 같은 다양한 콘텐츠 생성 분야에서 큰 잠재력을 가지고 있습니다.



### Lost in Tracking: Uncertainty-guided Cardiac Cine MRI Segmentation at Right Ventricle Bas (https://arxiv.org/abs/2410.03320)
- **What's New**: 이번 연구는 심장 자기 공명 영상(CMR)에서 RV(오른쪽 심실) 기저부 세분화의 난제를 해결하려고 합니다. 우리는 ACDC 데이터셋에 대한 RV 기저부 주석을 재주석 처리하고, 새로운 이중 인코더 U-Net 아키텍처를 제안하여 시간 불일치를 활용하여 RV 세분화를 개선했습니다.

- **Technical Details**: 우리는 RV 기저부의 세분화 문제를 다루기 위해 다음 두 가지 주석 개선 및 기법을 도입했습니다. 첫 번째로, 전문가 심장 전문의의 지도 아래 RV 기저부의 정교한 형태를 제공하여 ACDC 데이터셋의 공개 자원을 보강했습니다. 두 번째로, Bayesian motion tracking framework을 사용하여 시간 동안의 모션 불확실성을 추정하고, 이를 이중 인코더 U-Net 아키텍처에 통합하여 RV 기저부 세분화 성능을 향상시켰습니다.

- **Performance Highlights**: 우리의 실험 결과는 제안한 방법이 RV 기저부 세분화를 크게 개선하며, 특히 시간 불일치를 고려한 결과를 보여줍니다. 또한 일관된 주석과 모션 추적 손실의 결합을 통해 RV 세분화의 재현성을 높일 수 있음을 확인했습니다.



### SELU: Self-Learning Embodied MLLMs in Unknown Environments (https://arxiv.org/abs/2410.03303)
- **What's New**: 최근 다중모달 대형 언어 모델(MLLM)들이 강력한 시각적 이해 및 의사결정 능력을 보여주며, 미지의 환경에서 자율적으로 개선할 수 있는 가능성이 열리고 있습니다. 하지만 외부 피드백(예: 인간이나 환경 피드백)이 항상 존재하지 않는다는 문제가 있습니다. 이 논문에서는 씽크-크리틱(self-learning) 패러다임인 SELU를 제안합니다.

- **Technical Details**: SELU는 강화 학습의 행동자-비평가(actor-critic) 패러다임에서 영감을 받아, MLLM이 미지의 환경에서 스스로 학습할 수 있는 새로운 방법론입니다. 비평가는 자기 질문(self-asking) 및 회상 레이블링(hindsight relabeling)을 통해 행동자가 수집한 상호작용 경로에서 지식을 추출하여 환경 이해를 높이고, 행동자는 비평가의 자기 피드백을 통해 의사결정을 개선합니다.

- **Performance Highlights**: AI2-THOR 및 VirtualHome 환경에서 SELU 방법의 평가 결과, 비평가는 약 28% 및 30%의 개선을 보였고, 행동자는 약 20% 및 24%의 성능 향상을 달성했습니다.



### Semantic Segmentation Based Quality Control of Histopathology Whole Slide Images (https://arxiv.org/abs/2410.03289)
Comments:
          14 pages, 8 figures

- **What's New**: 본 연구에서는 다양한 품질 관리(QC)를 위한 소프트웨어 파이프라인을 개발하였습니다. 이 파이프라인은 흐림(blur) 정도, 조직(tissue) 영역, 조직 주름(tissue folds), 펜 마크(pen marks) 등을 세분화하는 여러 경량화된 딥러닝 모델을 포함합니다. 이를 통해 높은 정확도와 속도 간의 균형을 맞췄습니다.

- **Technical Details**: 제안된 파이프라인은 11,000개 이상의 조직병리학적 이미지(histo-pathological images)를 포함하는 TCGA 데이터셋에서 평가되었습니다. 테크닉적으로, HistoROI 패치 분류 도구를 사용하여 라벨이 식별된 여러 WSIs로부터 패치를 모자이크하여 주석 이미지(annotation images)를 자동으로 준비하여 조직 및 흐림 세분화의 주석 노력을 최소화했습니다.

- **Performance Highlights**: 본 연구의 주요 기여는 세분화 모델을 위한 새로운 방법론을 제안하며, 이는 HistoROI로부터 생성된 도메인 지식을 활용합니다. TCGA 데이터 포털에서 11,000개 이상의 WSI에 대한 모델 예측 결과를 공개하며, 연구 커뮤니티가 이 파이프라인을 즉시 사용하거나 향후 새로운 데이터셋 및 응용에 맞게 추가로 맞춤화할 수 있도록 하였습니다.



### 3D Segmentation of Neuronal Nuclei and Cell-Type Identification using Multi-channel Information (https://arxiv.org/abs/2410.03248)
- **What's New**: 이 논문에서는 뇌에서 서로 다른 세포 유형의 수를 정확하게 추정하기 위한 자동화된 방법론을 제시합니다. 특히, 신경 세포의 뉴클레우스를 개선된 3D 재구성 방법을 통해 비신경 세포의 뉴클레우스를 제외하고 분할(segmentation)하는 방법을 소개합니다.

- **Technical Details**: 제안된 알고리즘은 쥐 신피질(neocortex)에서 촬영된 이미지 스택을 테스트하여, 불균일한 염색(uneven staining) 및 세 가지 다른 채널을 사용해 다양한 세포 마커를 시각화하는 복잡한 시나리오에서 신경 세포의 뉴클레우스를 효과적으로 식별하고 3D로 분할할 수 있었습니다.

- **Performance Highlights**: 다양한 자동화 도구들이 존재하지만, 동일한 뇌 영역에서도 세포 수 추정치가 상이하여 신경해부학자(neuroanatomist)들에 의해 부정확하거나 일관되지 않다고 보고된 결과들이 있습니다. 이 연구는 신경 세포, 신경교세포(glial cell), 주혈관 세포(perivascular cell)를 구분할 수 있는 자동 분할 도구의 필요성을 강조하며, 수작업을 줄이고 세포 수 카운팅을 체계화할 수 있는 가능성을 보여줍니다.



### ScriptViz: A Visualization Tool to Aid Scriptwriting based on a Large Movie Databas (https://arxiv.org/abs/2410.03224)
Comments:
          Accepted in the 37th Annual ACM Symposium on User Interface Software and Technology (UIST'24). Webpage: this https URL

- **What's New**: 이 논문에서는 ScriptViz라는 도구를 소개하여, 스크립트 작가들이 영화 데이터베이스에서 외부 비주얼을 제공받아 스크립트를 작성하는 과정에서 도움이 되도록 합니다. ScriptViz는 스크립트의 텍스트와 대화를 기반으로 적절한 시각 자료를 실시간으로 검색하여 제공합니다.

- **Technical Details**: ScriptViz는 두 가지 유형의 비주얼 요소 제어를 통해 작가들이 고정된 비주얼 요소를 명확하게 볼 수 있도록 하며, 불확실한 요소의 변화를 살펴볼 수 있게 합니다. 이 도구는 스크립트의 부분적 내용과 고정 및 변화 가능한 비주얼 속성을 지정하여 기존 영화와 일치하는 장면을 찾고, 각 장면에 대한 키프레임을 검색하여 제공합니다.

- **Performance Highlights**: 15명의 스크립트 작가들을 대상으로 한 사용자 평가에서, ScriptViz는 스크립트와 일치하는 비주얼 가능성을 일관되면서도 다양하게 제시하여 창작에 도움이 되는 것으로 나타났습니다. 이 도구는 기획 과정에서 스크립트 작가들에게 효과적인 지원을 제공하며, 지정된 비주얼 속성을 활용하여 더 나은 브레인스토밍과 기획을 가능하게 합니다.



### ECHOPulse: ECG controlled echocardio-grams video generation (https://arxiv.org/abs/2410.03143)
- **What's New**: ECHOPULSE는 최초로 ECG 신호를 기반으로 한 ECHO 비디오 생성 모델로, 복잡한 조건 프롬프트 없이 빠르고 효율적으로 ECHO 영상을 생성할 수 있는 가능성을 보여줍니다.

- **Technical Details**: ECHOPULSE는 비디오 토큰화 모델, 마스킹 생성 변환기, 그리고 비디오 생성 파이프라인으로 구성되어 있으며, VQ-VAE 토큰화 및 마스킹 비주얼 토큰 모델링을 활용하여 ECHO 비디오 생성을 가속화합니다. ECG 신호를 사용하여 비디오를 생성하는 방식은 시간적 상관관계를 포착하는 것이 핵심입니다.

- **Performance Highlights**: 세 가지 공공 및 개인 데이터 세트에서 ECHOPULSE는 정량적 및 정성적 측정 모두에서 SOTA 성과를 달성하며, 심장 MRI, fMRI, 3D CT와 같은 다른 모달리티 생성 작업에도 쉽게 일반화될 수 있습니다.



### Machine Learning for Asymptomatic Ratoon Stunting Disease Detection With Freely Available Satellite Based Multispectral Imaging (https://arxiv.org/abs/2410.03141)
Comments:
          13 pages, 1 figure and 2 tables (main text), 1 figure and 3 tables (appendices). Submitted to "Computers and Electronics in Agriculture"

- **What's New**: 본 연구에서는 자유롭게 이용할 수 있는 위성 기반 원격 감지 데이터를 활용하여 고당수염병(RSD)과 같은 무증상 감염병을 포함한 사탕수수 품종의 질병 탐지에 기계 학습(machines learning) 기술을 적용하였습니다.

- **Technical Details**: 연구는 세 가지 단계로 구성됩니다: 1단계에서는 RSD 데이터 수집, 2단계에서는 Sentinel-2 이미지를 사용한 데이터 전처리, 3단계에서는 기계 학습 개발을 진행합니다. 특히, SVM-RBF(Support Vector Machine with Radial Basis Function Kernel) 알고리즘이 가장 높은 분류 정확도(85.64%~96.55%)를 기록하였습니다.

- **Performance Highlights**: Gradient Boosting 및 Random Forest도 높은 성능을 보였고, 정확도는 83.33%~96.55%로 나타났습니다. RSD 탐지에 도움이 되는 사탕수수 품종과 식생 지수를 포함시키는 것이 중요하다는 점이 확인되었습니다.



### CPFD: Confidence-aware Privileged Feature Distillation for Short Video Classification (https://arxiv.org/abs/2410.03038)
Comments:
          Camera ready for CIKM 2024

- **What's New**: 이 연구는 최첨단 다중 모달(multi-modal) 비디오 분류에서 권장 밀집 특성(Privileged Dense Features)의 활용을 위한 새로운 접근 방식을 제안합니다. 전통적인 PFD(Privileged Feature Distillation) 방법의 한계를 극복하기 위해, 확신 점수(confidence scores)를 활용하여 학생 모델의 성능 변동을 줄이는 방법인 CPFD(Confidence-aware Privileged Feature Distillation)를 도입했습니다.

- **Technical Details**: CPFD는 학생 모델이 교사 모델의 출력을 학습하는 것뿐만 아니라 교사의 확신 수준에서 파생된 추가적인 통찰을 활용하도록 설계되었습니다. 이를 통해 밀집 특성의 이점과 전통적인 증류와 관련된 성능 변동 문제를 해결합니다. CPFD는 X-VLM 모델을 기반으로 하며, DF-X-VLM으로부터의 밀집 특성을 효과적으로 증류합니다.

- **Performance Highlights**: CPFD는 5개의 다양한 작업에서 비디오 분류 F1 점수를 평균 6.76% 향상시켰으며, 기존 PFD 대비 2.31%의 성능 개선을 보였습니다. 또한, 교사 모델인 DF-X-VLM과의 성능 갭을 84.6% 줄였고, 실제 운영 데이터에서의 효과성을 입증해 여러 모델에 배포되었습니다.



### MMP: Towards Robust Multi-Modal Learning with Masked Modality Projection (https://arxiv.org/abs/2410.03010)
- **What's New**: 이번 논문에서는 Masked Modality Projection (MMP)라는 새로운 방법을 제안하여, 어떤 형태의 모달리티가 결여되어도 견고하게 작동하는 단일 모델을 훈련할 수 있는 방법을 소개합니다. MMP는 훈련 중에 일부 모달리티를 무작위로 마스킹하여 남아있는 모달리티의 정보를 활용하여 결여된 모달리티의 토큰을 추정합니다.

- **Technical Details**: MMP 방법은 크게 두 가지 단계로 구성됩니다: (1) 모달리티 마스킹: 훈련 중에 무작위로 선택된 모달리티를 마스킹합니다; (2) 모달리티 프로젝션: 남아있는 모달리티를 사용하여 마스킹된 모달리티의 토큰을 예측합니다. 또한, 우리는 프로젝션된 토큰과 실제 토큰을 정렬하기 위해 정렬 손실 (alignment loss)를 사용합니다.

- **Performance Highlights**: MMP를 적용한 모델은 세 가지 작업과 다섯 가지 데이터셋을 포함한 광범위한 실험에서, 기존의 모달리티 결여에 대응하기 위한 방법들보다 성능이 더 우수한 결과를 보였습니다. 특히, MMP는 모든 모달리티가 활용 가능한 상태에서의 성능 유지뿐 아니라, 일부 모달리티가 결여된 상황에서도 상당한 성능 향상을 관찰할 수 있었습니다.



### GABIC: Graph-based Attention Block for Image Compression (https://arxiv.org/abs/2410.02981)
Comments:
          10 pages, 5 figures, accepted at ICIP 2024

- **What's New**: 이번 연구에서는 Graph-based Attention Block for Image Compression (GABIC)이라는 새로운 주의(attention) 메커니즘을 소개합니다. GABIC는 k-Nearest Neighbors (k-NN) 메커니즘을 활용하여 중복된 특징을 줄이는 방법을 제안하고 있습니다.

- **Technical Details**: GABIC는 지역 그래프를 기반으로 한 주의 메커니즘으로, 주의 과정에서 k-NN 기술을 사용하여 중복된 시각적 특징의 집합을 방지합니다. 실험을 통해 GABIC는 특히 높은 비트 전송률에서 기존의 유사한 방법들에 비해 압축 성능이 향상됨을 보여주었습니다.

- **Performance Highlights**: 실험 결과 GABIC는 비트 비율이 높은 상황에서도 압축 성능의 개선을 보여주었으며, 이는 최근에 보편화된 Labeled Image Compression (LIC) 모델에 비해 우수한 성능을 나타냅니다. 또한, 코드와 학습된 모델이 공개되어 연구자들이 활용할 수 있도록 되어 있습니다.



### SymmetricDiffusers: Learning Discrete Diffusion on Finite Symmetric Groups (https://arxiv.org/abs/2410.02942)
- **What's New**: 본 논문에서는 SymmetricDiffusers라는 새로운 이산(discete) 확산(diffusion) 모델을 소개하여, $S_n$에 대한 복잡한 분포 학습을 단순화합니다. 이 모델은 심층 신경망을 사용하여 역확산(reverse diffusion) 과정의 개별 전이(transition)를 학습하는 방식으로 문제를 분해합니다.

- **Technical Details**: 이 연구는 유한 대칭군($S_n$)에서의 확산 모델을 다루며, 리플 셔플(riffle shuffle)을 효과적인 전이로 식별하고, PL(Plackett-Luce) 분포의 일반화된 형태를 제안합니다. 또한 샘플링 및 학습 효율성을 향상시키기 위해 이론적으로 기반한 'Denoising Schedule'을 도입합니다.

- **Performance Highlights**: 모델은 4자리 MNIST 이미지 정렬, 노이즈가 있는 MNIST와 CIFAR-10 데이터셋의 직소 퍼즐 해결, 그리고 여행하는 세일즈맨 문제(TSP) 해결 등 다양한 작업에서 최첨단(state-of-the-art) 성능을 달성하거나 비교 가능한 성능을 보였습니다.



### Individuation of 3D perceptual units from neurogeometry of binocular cells (https://arxiv.org/abs/2410.02870)
Comments:
          30 pages, 13 figures

- **What's New**: 이 논문에서는 3차원 비전의 초기 단계를 모델링하기 위해 새로운 신경기하학적(sub-Riemannian) 모델을 도입하였습니다. 이 모델은 스테레오 대응(stereo correspondence) 문제를 해결하기 위해 신경 기반(neural-based) 알고리즘을 포함하여 장면 세분화(scene segmentation)을 효과적으로 수행합니다.

- **Technical Details**: 논문은 신경기하학 모델을 사용하여 3D 공간에서 시각 신호의 전파를 기술하는 적분-미분 방정식(integro-differential equation)을 제안하고, 이 방정식을 통해 cortical connectivity를 연구합니다. 이를 위해 스펙트럼 분석(spectral analysis)과 차원 축소(dimensionality reduction) 기법을 활용하여 3D 인식 단위(perceptual units)를 추출합니다.

- **Performance Highlights**: 제안된 방법을 통해 서브 리만 거리(sub-Riemannian distance)와 비교했을 때, 3D 장면의 세분화 및 스테레오 대응 문제를 성공적으로 해결할 수 있음을 보였습니다. 특히, 고유벡터(eigenvectors) 및 고유값(eigenvalues)을 이용한 해당 단위의 중요도를 평가하는 방식이 효과적임을 입증했습니다.



### YouTube Video Analytics for Patient Engagement: Evidence from Colonoscopy Preparation Videos (https://arxiv.org/abs/2410.02830)
Comments:
          The 30th WORKSHOP ON INFORMATION TECHNOLOGIES AND SYSTEMS. arXiv admin note: substantial text overlap with arXiv:2312.09425

- **What's New**: 이 연구는 환자 교육을 위한 YouTube 비디오에서의 의학적 정보 검색 및 분석을 위한 데이터 분석 파이프라인을 처음으로 선보입니다.

- **Technical Details**: YouTube Data API를 활용하여 선택한 검색 키워드에 대한 비디오 메타데이터를 수집하고, Google Video Intelligence API를 사용하여 텍스트, 프레임 및 객체 데이터를 분석합니다. 또한 Bidirectional Long Short-Term Memory (BiLSTM) 모델을 개발하여 비디오에서 의학 용어를 식별하고, 비디오의 의학적 정보 수준 및 이해 가능성에 따라 비디오를 분류하는 세 가지 클래시파이어를 구축합니다.

- **Performance Highlights**: 이 연구는 헬스케어 이해당사자들이 다양한 건강 상태 관리를 위한 새로운 교육 비디오 콘텐츠 생성을 위한 지침과 확장 가능한 방법론을 제공하여, 비디오의 의학적 정보와 이해 가능성을 향상시킵니다.



### KLDD: Kalman Filter based Linear Deformable Diffusion Model in Retinal Image Segmentation (https://arxiv.org/abs/2410.02808)
Comments:
          Accepted at BIBM 2024

- **What's New**: 이번 논문에서는 망막 혈관 분할을 위한 새로운 Kalman filter 기반의 Linear Deformable Diffusion (KLDD) 모델을 제안하였습니다. 이 모델은 기존의 U-Net 방식의 한계를 극복하고 소형 혈관과 모세혈관을 보다 효과적으로 분할할 수 있도록 설계되었습니다.

- **Technical Details**: KLDD 모델은 선형 변형 합성곱(linear deformable convolution)과 확산 모델(diffusion model)의 조합을 이용하여 망막 혈관 구조를 정확하게 캡처합니다. Kalman filter를 활용하여 변형 합성곱의 좌표 위치를 최적화하고, Cross-Attention Aggregation module (CAAM) 및 Channel-wise Soft Attention module (CSAM)을 활용하여 특징을 더욱 강화하였습니다.

- **Performance Highlights**: 실험 결과, 제안한 KLDD 모델이 DRIVE, CHASE_DB1, OCTA-500 데이터셋에서 기존의 방법들보다 우수한 성능을 보였으며, 특히 세밀한 혈관 구조를 포착하는 데 있어 높은 정확도를 나타내었습니다.



### AutoPETIII: The Tracer Frontier. What Frontier? (https://arxiv.org/abs/2410.02807)
- **What's New**: 2024년 AutoPET 대회에서는 FDG와 PSMA 두 가지의 트레이서 없이도 PET/CT 스캔에서 병변 분할(lesion segmentation)을 수행할 수 있는 완전 자동화된 알고리즘 개발이 주요 목표이다.

- **Technical Details**: nnUNetv2 프레임워크를 사용하여 6개 모델의 6-fold 앙상블을 훈련하여 PET/CT 병변 분할을 자동으로 수행한다. 각 이미지의 CT와 PET 볼륨을 조합된 4 채널 입력으로 사용하고, 윈도우링(windowing) 기법으로 노이즈를 줄인다.

- **Performance Highlights**: 모델의 정확도는 99.64%로, 전체 데이터셋에 대한 평균 추론(인퍼런스) 시간은 환자당 2.18초였으며, 이 작업의 반복 평가에서는 매우 낮은 잘못된 음성(False Negative) 및 잘못된 양성(False Positive) 비율을 기록하였다.



### Trust-informed Decision-Making Through An Uncertainty-Aware Stacked Neural Networks Framework: Case Study in COVID-19 Classification (https://arxiv.org/abs/2410.02805)
Comments:
          15 pages, 7 figures, 6 tables

- **What's New**: 본 연구는 방사선 이미지를 기반으로 COVID-19를 신뢰성 있게 분류하기 위한 불확실성 인식(stacked) 신경망 모델을 제안합니다. 이 모델은 자동화 시스템에 대한 신뢰를 증진할 수 있도록 불확실한 예측을 사용자에게 알리고, 자신 있게 올바른 예측을 정확히 식별하는 데 중점을 둡니다.

- **Technical Details**: 제안된 모델은 Monte Carlo dropout과 ensemble 기법을 포함한 불확실성 정량화 방법을 통합하여 진단 예측의 확실성을 평가합니다. 두 단계로 구성된 모델 프레임워크에서, 첫 번째 단계 모델은 초기 예측과 관련된 불확실성을 생성하며, 두 번째 단계 모델은 진단 결과와 함께 신뢰 지표(trust indicator)를 생성합니다.

- **Performance Highlights**: COVIDx CXR-4 데이터셋에 대한 광범위한 실험을 통해 신뢰할 수없는 사례와 확실하지 않은 사례를 식별하고 처리하는 데 있어 혁신적인 접근 방식이 입증되어, 임상 환경에서 자동화된 진단의 신뢰성을 증진시킵니다.



### AVG-LLaVA: A Large Multimodal Model with Adaptive Visual Granularity (https://arxiv.org/abs/2410.02745)
Comments:
          Preprint

- **What's New**: AVG-LLaVA는 이미지와 지시에 따라 적절한 시각적 세분화를 선택할 수 있는 LMM(대규모 다중모달 모델)입니다. 이 접근법은 시각적 토큰의 수를 줄이고 추론 속도를 증가시키며 모델 성능을 향상시킵니다.

- **Technical Details**: AVG-LLaVA는 (a) 여러 풀링 레이어를 포함하여 다양한 세분화의 시각적 토큰을 얻는 시각적 세분화 스케일러와 (b) Transformer 레이어, MLP 레이어, 투표자 레이어를 포함해 이미지와 지침에 기반하여 적절한 시각적 세분화를 선택하는 시각적 세분화 라우터를 도입합니다. RGLF(Ranking Granularity to align LMM Feedback)라는 새로운 훈련 패러다임을 제안하여 라우터가 시각적 세분화를 효과적으로 구별하도록 지원합니다.

- **Performance Highlights**: AVG-LLaVA는 11개의 벤치마크에서 우수한 성능을 달성하며, AI2D 벤치마크에서 시각적 토큰 수가 85.3% 감소하고 추론 속도가 2.53배 증가하는 등의 성과를 보였습니다.



New uploads on arXiv(cs.AI)

### System 2 reasoning capabilities are nigh (https://arxiv.org/abs/2410.03662)
- **What's New**: 본 연구는 머신러닝 모델의 인간과 유사한 추론 능력을 개발하기 위한 최신 문헌을 검토하며, System 2와 유사한 신경망 모델을 설계하기 위해 필요한 단계들을 설명합니다. 연구진은 지금까지의 발전에도 불구하고 현재의 모델들이 추론을 제대로 수행하지 못하고 있다고 주장하며, 목표 달성을 위한 남은 과제가 매우 적다고 강조합니다.

- **Technical Details**: 이 연구는 심리학에서 제안한 이원적 사고 이론을 바탕으로 하여, 인간의 사고 방식인 System 1과 System 2의 차이를 분석합니다. System 1은 빠르고 자동적이며 감정적이고, System 2는 느리고 의도적이며 논리적입니다. 머신러닝 모델이 System 2와 일치하기 위해서는 패턴 매칭을 넘어서는 증명 과정과 가설 검증이 필요합니다. 또한, 모델을 효과적으로 학습시키기 위해 Actor-Critic 알고리즘과 관련된 개념이 언급됩니다.

- **Performance Highlights**: 딥러닝을 통해 훈련된 복잡한 에이전트들이 실제 세계에 대해 논리적으로 추론할 수 있는 가능성이 가까운 미래에 실현될 것으로 기대됩니다. 이전 연구들은 체인-오브-투사(Chain-of-Thought) 기법을 통해 LLM이 문제 해결 과정을 단계별로 따르도록 유도함으로써 성능 개선 효과를 보여주었으며, 이는 머신러닝 모델의 발전에 중요한 기초 자료로 작용할 것입니다.



### TICKing All the Boxes: Generated Checklists Improve LLM Evaluation and Generation (https://arxiv.org/abs/2410.03608)
- **What's New**: TICK (Targeted Instruct-evaluation with ChecKlists)이라는 새로운 자동화된 평가 프로토콜을 제안하여 LLM의 instruction-following 능력을 해석 가능하고 유연하게 평가할 수 있게 되었습니다. 이 프로토콜은 LLM이 생성한 instruction-specific checklist를 통해 평가를 구조화합니다.

- **Technical Details**: TICK은 주어진 instruction에 대해 LLM이 YES/NO 질문으로 구성된 맞춤형 평가 체크리스트를 신뢰성 높게 생성할 수 있다는 것을 보여줍니다. 이 체크리스트는 각 후보 응답이 instruction의 특정 요구 사항을 충족하는지를 평가합니다. TICK을 사용하면 LLM의 판단과 인간의 선호 간의 정확한 일치 비율이 약 5.8% 증가합니다.

- **Performance Highlights**: 이 연구에서 제안된 STICK(Self-TICK)은 여러 벤치마크에서 자기 개선을 통해 세부 성능 향상을 이루었습니다. LiveBench에서 Command-R+는 3.8% 개선을 달성하고, WildBench에서 Best-of-N 선택 방식으로 5.3%의 성능 향상을 보여주었습니다. 또한 LLM이 생성한 체크리스트를 인간 평가자에게 제공하여 평가자 간의 일치도를 0.194에서 0.256으로 증가시켰습니다.



### SiMilarity-Enhanced Homophily for Multi-View Heterophilous Graph Clustering (https://arxiv.org/abs/2410.03596)
- **What's New**: 본 논문에서는 다중 뷰 이질 그래프 클러스터링을 위한 새로운 접근법인 SiMilarity-enhanced Homophily for Multi-view Heterophilous Graph Clustering (SMHGC)을 제안합니다. 이 방법은 이질 그래프 데이터에서 유사성을 증가시킴으로써 클러스터링 성능을 향상시키는데 중점을 둡니다.

- **Technical Details**: SMHGC는 이웃 패턴 유사성(neighbor pattern similarity), 노드 특성 유사성(node feature similarity), 다중 뷰 글로벌 유사성(multi-view global similarity)이라는 세 가지 유사성 개념을 도입합니다. 이들은 레이블 없는 방식(label-free)으로 이질 그래프에서 동질 정보(homophily)를 효율적으로 추출하고 융합하는 데 사용됩니다. 이를 통해 클러스터링 과정에서 더욱 개선된 메시지 패싱(message passing) 과정을 가능하게 합니다.

- **Performance Highlights**: SMHGC는 다양한 다중 뷰 이질 및 동질 데이터셋에서 최첨단 성능을 달성하였으며, 특히 이질 비율이 70%를 초과하는 반-합성 데이터셋에서도 클러스터링 성능에 감소가 없음을 보여주었습니다. 이는 기존 방법들에 비해 상당한 개선을 의미하며, 제안한 방법의 효능을 실험적으로 입증합니다.



### Understanding Reasoning in Chain-of-Thought from the Hopfieldian View (https://arxiv.org/abs/2410.03595)
Comments:
          28 pages, a new version of "A Hopfieldian View-based Interpretation for Chain-of-Thought Reasoning"

- **What's New**: 이 연구에서는 Chain-of-Thought (CoT) 추론의 기초에 있는 요인을 설명하기 위한 새로운 관점을 제시합니다. Hopfieldian 인지 이론을 바탕으로 CoT 추론과 주요 인지 요소 간의 관계를 확립하여 CoT의 성공 요인을 이해하고자 합니다.

- **Technical Details**: CoT 추론을 이해하기 위해 신경 과학에서의 Hopfieldian 관점을 이용합니다. CoT는 자극, 행동, 신경 집단, 표현 공간과 같은 인지 요소와 맥락을 두고 연결되며, CoT를 통해 발생하는 추론 과정은 이러한 표현 공간 간의 이동으로 설명됩니다. 새로운 프레임워크 'Representation-of-Thought (RoT)'를 제안하여 CoT의 견고성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, RoT는 CoT 추론의 견고성과 해석 가능성을 개선하며, 추론 과정에 대한 세밀한 제어를 제공함을 보여주었습니다. 다양한 과제(산술, 상식, 기호적 추론)에 대한 포괄적인 실험이 실시되었고, CoT 추론의 오류를 추적하고 통제할 수 있는 직관적이고 해석 가능한 분석이 가능함을 발견했습니다.



### f{\ae}rdXel: An Expert System for Danish Traffic Law (https://arxiv.org/abs/2410.03560)
- **What's New**: 새로운 도구인 færdXel은 덴마크 교통 법률 분야에서의 상징적 추론(symbolic reasoning)을 제공하며, 논리 프로그래밍(logic programming) 기법과 사용자 친화적인 인터페이스를 결합하여 사용자가 추론 과정을 탐색할 수 있습니다. 이 시스템은 덴마크 법률 분야의 전문가들을 지원하기 위한 기초가 될 가능성을 보여줍니다.

- **Technical Details**: færdXel은 Datalog의 강화된 버전을 이용하여 작성된 지식 기반을 가지고 있으며, SLD-resolution을 추론 시스템으로 활용합니다. 이 시스템은 법률 조항과 비슷한 사례를 고려하여 피고인이 덴마크 교통 법률의 특정 조항을 위반했음을 입증하는 주장을 찾는 데 중점을 둡니다.

- **Performance Highlights**:  초기 실증 평가에 따르면, færdXel은 매우 유망한 결과를 보이며 덴마크 법률 분야의 전문가들을 지원하는 데 큰 잠재력을 지니고 있습니다. 이 시스템은 결정을 내리는 대신, 전문가가 의사 결정을 내리는 데 필요한 의미 있는 정보와 방대한 입력을 제공합니다.



### On Uncertainty In Natural Language Processing (https://arxiv.org/abs/2410.03446)
Comments:
          PhD thesis

- **What's New**: 이번 논문은 자연어 처리(Natural Language Processing)에서의 불확실성(uncertainty) 문제를 언어학적, 통계적(statistical) 및 신경망(neural) 관점에서 분석하고, 실험 프로세스의 설계를 통해 이를 어떻게 감소시키고 정량화할 수 있는지를 연구합니다.

- **Technical Details**: 연구에서는 텍스트 분류(text classification) 작업에서 유도 모델 편향(inductive model biases)의 영향을 이론적으로 및 실증적으로 분석하고, 덴마크어(Danish), 영어(English), 핀란드어(Finnish) 데이터를 포함한 세 가지 언어에 대한 실험을 실시합니다. 또한, 비교불가능한(conformal prediction) 방식에 기반하여 자연어 생성에서 보정된 샘플링(calibrated sampling) 방법을 제안하며 이는 실제 연속성의 더 나은 범위를 가진 더 타이트한 토큰 세트를 제공합니다.

- **Performance Highlights**: 대규모 블랙박스 언어 모델(large black-box language models)의 신뢰도를 양측 예측(auxiliary predictors)을 사용하여 정량화할 수 있는 접근법을 개발하며, 이는 target 모델의 입력과 생성된 출력 텍스트만으로 신뢰도를 예측합니다.



### Towards a Benchmark for Large Language Models for Business Process Management Tasks (https://arxiv.org/abs/2410.03255)
- **What's New**: 본 연구는 Business Process Management (BPM) 분야에 적합한 특정 LLM 성능을 평가하기 위한 새로운 벤치마크를 제안합니다. 기존의 LLM 벤치마크가 일반적 작업에 초점을 맞추고 있어 BPM과 같은 구체적인 도메인에서의 LLM 성능을 확인할 필요가 있었습니다.

- **Technical Details**: 연구에서는 네 가지 BPM 작업(활동 추천, RPA 후보 식별, 프로세스 질문 응답, 선언적 프로세스 모델 분석)에 대한 LLM 성능을 체계적으로 비교했습니다. 특히 오픈 소스 모델과 상업적 모델 간의 성능 차이 및 모델의 크기가 BPM 작업 성능에 미치는 영향을 평가했습니다.

- **Performance Highlights**: 이 연구는 다양한 LLM이 어떻게 BPM 작업을 수행하는지를 명확히 정의하고, 적합한 모델 선택에 있어 조직에게 실질적인 통찰을 제공합니다. 분석 결과, 오픈 소스 모델들이 특정 BPM 작업에서 상업적 모델과 유사하거나 뛰어난 성능을 나타낼 수 있다는 점이 밝혀졌습니다.



### Enriching Ontologies with Disjointness Axioms using Large Language Models (https://arxiv.org/abs/2410.03235)
Comments:
          Accepted at KBC-LM'24 workshop at ISWC 2024

- **What's New**: 이 연구는 Knowledge Graphs에서 클래스 간의 명백한 disjointness 선언이 부족한 문제를 해결하기 위해 Large Language Models (LLMs)를 활용하는 방법을 탐구합니다. LLM의 암묵적인 지식을 활용하여 클래스의 disjointness를 식별하고 명시하는 접근 방식을 제안합니다.

- **Technical Details**: 연구에서는 DBpedia ontology를 대상으로 open-source LLM을 사용하여 disjoint class 관계를 신뢰성 있게 식별하기 위한 효과적인 prompt 전략을 적용합니다. 이는 ontological disjointness를 분류하기 위한 prompt engineering을 포함하며, 논리적 관계를 고려한 종합적인 disjointness enrichment 절차를 제안합니다.

- **Performance Highlights**: LLMs는 효과적인 prompt 사용 시 disjoint class 관계를 신뢰성 있게 식별할 수 있음을 보여주어, ontology 완성을 위한 수작업 프로세스를 간소화할 수 있습니다. 이 연구 결과는 LLM의 자동화된 ontology 강화 및 전략적 prompt 설계에 대한 통찰을 제공합니다.



### Adaptive Masking Enhances Visual Grounding (https://arxiv.org/abs/2410.03161)
Comments:
          Code will be available at this https URL

- **What's New**: 최근의 연구에서, 시각적 기초 학습(visual grounding)에서 제로샷(zero-shot) 및 핀샷(few-shot) 학습이 큰 주목을 받고 있습니다. 이 연구에서는 대규모 비전-언어 사전 학습을 통해 얻은 인사이트를 활용한 새로운 방법인 IMAGE를 제안합니다. IMAGE는 데이터셋 크기를 늘리지 않고도 저샷 학습 상황에서 단어 기초를 향상시킬 수 있도록 설계되었습니다.

- **Technical Details**: IMAGE(Interpretative MAsking with Gaussian radiation modEling)는 인간의 인지 과학에서 영감을 받아 생성된 적응형 마스킹(adaptive masking) 전략을 활용하여, 모델이 이미지의 주요 특징에 집중하도록 합니다. 중요도 우선 생성 블록과 적응형 마스크 생성 블록으로 구성되어 있습니다. 이들 블록은 이미지 패치의 중요도를 기반으로 적응형 마스크를 생성하고, 비전 백본의 특징 맵에서 주목할 만한 영역을 강조합니다.

- **Performance Highlights**: COCO 및 ODinW와 같은 벤치마크 데이터셋에서 IMAGE의 유효성을 평가한 결과, 기존 모델들을 뛰어넘는 성능을 확인했습니다. 제로샷 및 핀샷 작업에서 향상된 일반화와 성능을 얻었으며, 실험 결과는 IMAGE가 기초 모델들보다 월등히 우수하다는 것을 보여주었습니다. 이 연구는 데이터셋 크기를 확장하는 것에 의존하지 않고도 효과적인 저샷 학습을 위한 잠재력을 강조합니다.



### AIME: AI System Optimization via Multiple LLM Evaluators (https://arxiv.org/abs/2410.03131)
Comments:
          21 pages, 10 Figures, 4 Tables

- **What's New**: AI 시스템 최적화에서 단일 LLM 평가자 사용의 한계를 강조하고, 여러 LLM 평가자를 사용하는 AIME 프로토콜을 제안합니다. 이를 통해 복잡한 코드 생성 작업에서 높은 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: AIME (AI system optimization via Multiple Evaluators) 프로토콜은 서로 다른 기준에 대해 독립적으로 평가를 생성하는 여러 LLM을 사용하여 각 평가를 합치는 방식으로 작동합니다. 이 방법은 오류 감지율을 62%까지 향상시키고, 성공률을 16% 증가시킵니다.

- **Performance Highlights**: AIME는 LeetCodeHard와 HumanEval 벤치마크에서 단일 평가 프로토콜에 비해 최대 62% 높은 오류 감지율을 보이며, 테스트 케이스의 성공률은 최대 16% 더 높았습니다. 또한 평가자의 수와 기준 선택이 성공률에 영향을 미친다는 점을 강조합니다.



### ProcBench: Benchmark for Multi-Step Reasoning and Following Procedur (https://arxiv.org/abs/2410.03117)
- **What's New**: 이 논문에서는 ProcBench라는 새로운 벤치마크를 제시하여 다단계 추론(multi-step inference)에 대한 직접 평가를 중점적으로 다루고 있습니다. 기존의 벤치마크와는 달리, ProcBench는 복잡한 지식 없이 제공된 지침을 따르는 능력을 평가합니다.

- **Technical Details**: ProcBench는 여러 가지 간단한 작업을 포함하며, 각 작업은 명확히 정의된 지침을 따르는 과정을 요구합니다. 연구에서 사용된 데이터셋은 명시적인 지침과 해당 질문의 쌍으로 구성되어 있으며, 각 단계에서 모델이 지침을 따르는 능력을 평가합니다.

- **Performance Highlights**: 최신 대형 언어 모델(LLMs)에 대한 평가 결과, 모델에 따라 성능 차이가 있었습니다. o1-preview와 o1-mini와 같은 일부 모델은 간단한 작업에서 높은 정확도를 보였지만, 복잡성이 증가함에 따라 성능이 크게 저하되는 한계를 드러냈습니다.



### Image First or Text First? Optimising the Sequencing of Modalities in Large Language Model Prompting and Reasoning Tasks (https://arxiv.org/abs/2410.03062)
- **What's New**: 이 논문은 다중 모달 프롬프트(multi-modal prompts)에서 이미지와 텍스트의 순서가 대형 언어 모델(LLMs)의 추론 성능에 미치는 영향을 조사합니다.

- **Technical Details**: 세 가지 상용 LLM(GPT-4o, Gemini-1.5 Flash, Claude-3-Haiku)을 사용하여 다중 모달 LLM의 추론 성능에 대한 이미지와 텍스트 프롬프트의 시퀀싱(sequencing) 영향의 실증 평가를 수행했습니다. 결과에 따르면, 모달리티 시퀀싱은 특히 복잡한 추론 작업에서 성능에 긍정적인 영향을 미칩니다. 심지어, 명확한 답변 성능 향상을 위해 이미지와 텍스트 입력 모달리티의 특정 요소가 시퀀스에 민감하게 반응함을 발견했습니다.

- **Performance Highlights**: 이 연구는 모달리티 시퀀싱이 복잡한 추론 작업에서 성능에 상당한 영향을 미친다는 점을 강조합니다. 최적의 다중 모달 프롬프트 구성을 위한 실용적인 가이드를 제안하며, 교육, 의료 영상 및 교차 모달 학습 분야에서의 적용 가능성을 제시합니다.



### Guided Stream of Search: Learning to Better Search with Language Models via Optimal Path Guidanc (https://arxiv.org/abs/2410.02992)
- **What's New**: 이번 연구에서 우리는 언어 모델의 검색(search) 및 계획(planning) 능력을 향상시키기 위해 최적의 솔루션(optimal solution)을 활용하는 방법을 탐구합니다. 이에 따라 우리는 guided stream of search (GSoS)라는 방법을 제안하며, 이는 최적의 솔루션을 탐색 과정에 점진적으로 통합하여 고품질의 검색 경로를 생성합니다.

- **Technical Details**: GSoS는 최적의 솔루션에서 각 중간 작업을 단계별로 통합하는 방식으로 작동합니다. 이 과정에서는 실패한 검색 경로를 맥락(context)으로 사용하며, 이러한 방식을 통해 사전 훈련(pre-trained)된 모델의 성능을 향상시킵니다. Countdown라는 수학적 추론 문제에서 이러한 접근 방식을 평가하였으며, 정확도가 13% 향상되었습니다. 강화 학습(RL) 미세 조정을 동시에 적용할 경우, 이 개선은 20%까지 증가합니다.

- **Performance Highlights**: 우리의 접근 방식은 이전의 감독 기반 미세 조정(supervised fine-tuning) 방법과 비교하여 더 높은 성능을 보여주며, 특히 GSoS를 사용한 RL 미세 조정이 효과적입니다. 이로 인해 모델의 검색 및 계획 능력이 크게 향상되었습니다.



### AiBAT: Artificial Intelligence/Instructions for Build, Assembly, and Tes (https://arxiv.org/abs/2410.02955)
Comments:
          9 pages, 6 figures, 2 tables

- **What's New**: 본 논문에서는 IBAT(Instructions for Build, Assembly, and Test) 문서 생성을 지원하는 새로운 시스템인 AiBAT에 대해 소개합니다. 이 시스템은 기계 학습(machine learning)과 컴퓨터 비전(computer vision) 기술을 활용하여 IBAT 템플릿의 일부를 자동으로 채워주는 기능을 제공합니다.

- **Technical Details**: AiBAT 시스템은 먼저 조립 도면(assembly drawing)을 분석하고, 필요한 정보를 추출한 후, IBAT 템플릿을 해당 정보로 채우는 과정을 거칩니다. 이는 사용자로 하여금 엔지니어링 도면과 부품 리스트에서 정보를 수동으로 입력하는 시간을 절약할 수 있게 해줍니다.

- **Performance Highlights**: 예비 결과에 따르면, AiBAT 시스템은 시간 절약 및 비용 절감 잠재력을 갖추고 있으며, 사용자에게 보다 고급 기술 작업에 집중할 수 있는 여유를 제공합니다. 논문에서는 시스템의 초기 성과와 향후 연구 방향에 대한 논의도 포함되어 있습니다.



### Intrinsic Evaluation of RAG Systems for Deep-Logic Questions (https://arxiv.org/abs/2410.02932)
- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG) 메커니즘을 평가하기 위한 새로운 내재적 지표인 Overall Performance Index (OPI)를 소개합니다. OPI는 두 개의 주요 지표인 Logical-Relation Correctness Ratio와 지면 진답과 생성된 답변 간의 BERT 임베딩 유사도 점수의 평균을 조화 평균으로 계산합니다.

- **Technical Details**: RAG 시스템은 인덱싱(Indexing)과 검색(Retrieval) 두 가지 주요 구성 요소로 이루어져 있으며, OPI는 LLM이 생성한 답변의 정확성과 분류기가 생성한 논리 관계의 정확성을 측정하는 지표입니다. LangChain이라는 인기 있는 RAG 도구를 활용해 RAG-Dataset-12000을 평가하였으며, 여러 검색기(류)들 간의 OPI 점수를 비교하여 성능 차이를 분석했습니다.

- **Performance Highlights**: BERT 기반 임베딩을 사용하는 코사인 유사도 검색기가 다른 검색기들보다 뛰어난 성능을 보였으며, 유클리드 거리 기반 검색기는 가장 낮은 성능을 기록했습니다. 또한 여러 검색기를 조합하거나 검색된 문장을 병합하는 방식이 단일 검색기를 사용할 때보다 우수한 성능을 나타냈습니다.



### Fine-Tuning Language Models with Differential Privacy through Adaptive Noise Allocation (https://arxiv.org/abs/2410.02912)
Comments:
          EMNLP 2024 findings

- **What's New**: 이 논문에서는 ANADP라는 새로운 알고리즘을 제안하여 언어 모델의 매개변수 중요도에 따라 추가 노이즈를 적응적으로 할당합니다. 이 접근법은 전통적인 Differential Privacy(DP) 방법의 한계를 극복하고 기계 학습 모델의 프라이버시를 강화하는 동시에 성능을 개선합니다.

- **Technical Details**: ANADP는 매개변수의 중요도에 기반하여 노이즈와 프라이버시 예산을 분배하는 방법입니다. 이는 매개변수의 감도(sensitivity)와 불확실성(uncertainty)을 고려하여 모델의 훈련 과정에서 안정적으로 적용됩니다. 기존의 DP 방법과 달리, ANADP는 각 매개변수의 기여도를 평가하여 균일하지 않은 방식으로 프라이버시 예산을 배분합니다.

- **Performance Highlights**: ANADP는 Glue benchmark에서 기존의 DP 방법보다 항상 우수한 성능을 보여주었으며, 전통적 DP와 비-DP 파인튜닝(수정 없이 원본을 유지한 파인튜닝) 간의 성능 격차를 줄이는 데 성공했습니다. 또한 ANADP는 기존의 DP 방법처럼 강력한 프라이버시 보호를 유지합니다.



### The Role of Deductive and Inductive Reasoning in Large Language Models (https://arxiv.org/abs/2410.02892)
Comments:
          4 figures

- **What's New**: 이 논문에서는 Deductive and InDuctive (DID) 방법을 제안하여 LLM(Large Language Models)의 추론 능력을 향상시키고, 동적으로 추론 경로를 조정할 수 있는 유연한 프레임워크를 제공합니다.

- **Technical Details**: DID 방법은 인지 과학의 원리에 기반하여 유도적(inductive)과 연역적(deductive) 추론 과정을 프롬프트 구성 과정에 통합하여 LLM의 추론 유연성과 적응성을 높입니다. 이 접근법은 다양한 데이터셋에서 검증되었으며, 모델의 성능을 크게 향상시킵니다.

- **Performance Highlights**: DID 방법을 사용한 결과, 기존 저명한 데이터셋인 AIW와 MR-GSM8K 및 자체 제작한 Holiday Puzzle에서 솔루션 정확도와 추론 품질 모두에서 유의미한 향상을 보여주었습니다. 이 모든 개선사항은 substantial computational overhead 없이 이루어졌습니다.



### LLaMA-Berry: Pairwise Optimization for O1-like Olympiad-Level Mathematical Reasoning (https://arxiv.org/abs/2410.02884)
- **What's New**: 본 논문은 LLaMA-Berry라는 고급 수학 문제 해결 프레임워크를 제안하여 대형 언어 모델(LLMs)의 수학적 추론 능력을 향상시키고자 합니다. Monte Carlo Tree Search (MCTS)와 반복적인 Self-Refine을 결합하여 추론 경로를 최적화하고, 쌍간 보상 모델을 활용하여 다양한 경로를 평가합니다.

- **Technical Details**: LLaMA-Berry 프레임워크는 Self-Refine을 Monte Carlo Tree Search에 적용한 SR-MCTS와 쌍간 선호 보상 모델(PPRM)이라는 두 가지 새로운 방법으로 구성됩니다. MCTS는 마르코프 결정 과정(MDP) 프레임워크에서 최적 솔루션 탐색을 진행하고, PPRM은 인간 피드백에서의 강화 학습(RLHF) 기법에서 영감을 받아 해결책 간의 선호 관계를 모델링하고 이를 글로벌 순위 점수로 집계합니다.

- **Performance Highlights**: 이 프레임워크는 GPQA, AIME24, AMC23 등 복잡한 올림피아드 레벨 벤치마크에서 ToT 및 rStar와 같은 기존 방법들보다 우수한 성능을 보여주었습니다. LLaMA-3.1-8B 모델이 추가 훈련 없이도 GPT-4 Turbo와 유사한 수준으로 성능을 향상시켰다는 결과는 LLaMA-Berry 방식이 소규모 데이터만으로도 LLM의 추론 능력을 효과적으로 개선할 수 있음을 시사합니다.



### Skill Issues: An Analysis of CS:GO Skill Rating Systems (https://arxiv.org/abs/2410.02831)
- **What's New**: 이번 연구에서는 Counter-Strike: Global Offensive (CS:GO) 게임의 스킬 레이팅 시스템인 Elo, Glicko2, TrueSkill을 분석하였습니다. 각 시스템의 성능을 실제 데이터를 기반으로 비교하고, 매칭 알고리즘의 영향을 고찰하였습니다.

- **Technical Details**: 연구는 스킬 레이팅 시스템을 다양한 에뮬레이터(Emulator)로 구현하고, 매칭 알고리즘을 Acquisition Function으로 설정하여 통합적인 시뮬레이터(Simulator) 환경을 조성하였습니다. 이 환경에서는 실제 게임 데이터에 따라 팀 매칭을 선택하고, 각 시스템의 성능을 검증하였습니다.

- **Performance Highlights**: Elo, Glicko2, TrueSkill 각 시스템의 성능을 대규모 CS:GO 데이터셋을 통해 시험하였으며, TrueSkill이 CS:GO에서 62%의 정확성을 보이는 등 높은 성능을 나타냈습니다. 이를 통해 매칭 알고리즘과 스킬 레이팅 간의 원형 의존성(circular dependency)을 정량적으로 평가할 수 있었습니다.



### LLMs May Not Be Human-Level Players, But They Can Be Testers: Measuring Game Difficulty with LLM Agents (https://arxiv.org/abs/2410.02829)
- **What's New**: 최근의 Large Language Models (LLMs)의 발전이 게임 테스트 분야에 새로운 가능성을 제시하고 있습니다. 이 연구에서는 LLM을 활용하여 게임의 난이도를 측정할 수 있는지 탐구하였으며, Wordle과 Slay the Spire와 같은 전략 게임을 통해 그 유효성을 검증하였습니다.

- **Technical Details**: 연구진은 LLM 에이전트를 활용한 일반 게임 테스트 프레임워크를 제안하였습니다. LLM은 간단한 지침을 통해 인간 플레이어의 평가와 강한 상관관계를 보여주며, 이는 게임 내부의 다양한 도전 과제의 난이도를 측정하는 데 유용할 수 있음을 시사합니다. 이 연구는 LLM이 별도의 세부 조정 없이 여러 게임에서 플레이하고 난이도를 평가할 수 있는 일반적인 프레임워크를 목표로 하고 있습니다.

- **Performance Highlights**: LLMs는 평균 인간 플레이어와 비교할 때 성능이 다소 부족하지만, 특정 과제에서 LLM의 난이도 인식은 인간 플레이어의 인식과 유사한 경향을 보였습니다. 이는 게임 개발 과정에서 LLM이 효과적인 난이도 평가 도구로 사용될 수 있음을 나타냅니다. 이 연구의 결과는 LLM 기반 게임 테스트 환경 개발에 대한 통찰력을 제공할 수 있습니다.



### DANA: Domain-Aware Neurosymbolic Agents for Consistency and Accuracy (https://arxiv.org/abs/2410.02823)
- **What's New**: 이번 연구에서는 DANA(Domain-Aware Neurosymbolic Agent)라는 새로운 아키텍처를 소개합니다. 이 아키텍처는 도메인 특화 지식과 심볼릭(symbolic) 접근 방식을 통합하여 대형 언어 모델(LLM)의 확률적 특성으로 인한 불일치성과 부정확성을 해결합니다.

- **Technical Details**: DANA는 자연어와 심볼릭 형태로 도메인 전문 지식을 캡처하고 적용하여 더 결정적이며 신뢰할 수 있는 문제 해결 행동을 가능하게 합니다. 이 아키텍처는 OpenSSA 프레임워크에서 계층적 작업 계획(Hierarchical Task Plans, HTPs)을 사용하여 구현되었으며, 금융 분석 벤치마크인 FinanceBench에서 90% 이상의 정확도를 달성했습니다.

- **Performance Highlights**: DANA는 현재 LLM 기반 시스템보다 일관성과 정확성 면에서 우수한 성능을 보였으며, 반도체 제조 프로세스와 같은 물리적 산업에 적용 가능하다는 점을 강조하였습니다.



### GPT's Judgements Under Uncertainty (https://arxiv.org/abs/2410.02820)
- **What's New**: 본 연구는 인간의 인지 편향이 GPT-4o의 확률적 판단 및 결정 형성에 어떻게 드러나는지를 1350번의 실험을 통해 조사하였습니다. 이를 통해 비슷한 확률적 표기에 대한 반응 방식에서 AI의 모순적 접근을 보여주었습니다.

- **Technical Details**: 연구에서는 인지 편향으로는 손실 회피, 형태 효과, 결합 오류 등 9개의 편향을 사용하여 1350개의 실험을 진행하였으며, 통계적 추론과 직관적 추론 간의 반응을 분석했습니다. 각 실험은 150번 반복하여 결과의 일관성을 높였습니다.

- **Performance Highlights**: 총 1350개의 실험 중, GPT-4o는 658개의 상세한(elaborate) 응답과 692개의 직관적(intuitive) 응답을 제공하였습니다. 특히 결합 오류에 대한 실험에서는 언제나 상세한 응답을 제공하며, 통계적으로 타당한 이유를 설명했습니다.



### Bipolar fuzzy relation equations systems based on the product t-norm (https://arxiv.org/abs/2410.02816)
- **What's New**: 본 논문에서는 바이어럴 퍼지 관계 방정식(Bipolar fuzzy relation equations)의 해법 및 대수적 구조에 대한 연구가 진행되었습니다. 특히, max-product t-norm 조합을 기반으로 한 바이어럴 퍼지 관계 방정식 시스템의 해의 타당성과 구조를 살펴봅니다.

- **Technical Details**: 바이어럴 퍼지 관계 방정식은 알려지지 않은 변수와 그 논리적 부정을 동시에 다루는 새로운 유형의 방정식입니다. 논문은 max-product t-norm을 이용한 방정식의 해법과 독립항이 제로인 경우를 포함하여 다양한 대수적 구조를 연구합니다.

- **Performance Highlights**: 이 연구의 결과는 바이어럴 max-product FREs의 해의 존재 여부와 최댓값/최솟값 솔루션 또는 유한한 수의 최대/최소 솔루션이 존재할 수 있는 조건을 명확히 하며, 이로 인해 이는 최적화 문제와 같은 응용 분야에서 매우 유용하게 사용될 수 있습니다.



### SAC-KG: Exploiting Large Language Models as Skilled Automatic Constructors for Domain Knowledge Graphs (https://arxiv.org/abs/2410.02811)
Comments:
          ACL 2024 Main

- **What's New**: 본 논문에서는 SAC-KG라는 일반적인 지식 그래프(KG) 구축 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLM)을 전문 지식으로 활용하여 자동으로 특화된 다단계 KGs를 생성합니다.

- **Technical Details**: SAC-KG는 세 가지 구성 요소로 이루어져 있습니다: Generator, Verifier, Pruner. Generator는 원시 도메인 코퍼스에서 관련성을 가진 관계와 꼬리 엔티티를 생성하고, Verifier는 오류를 감지하여 수정하며, Pruner는 필요에 따라 다음 단계의 생성을 결정합니다.

- **Performance Highlights**: SAC-KG는 100만 개 이상의 노드 규모로 도메인 KG를 자동으로 구축하며, 89.32%의 정밀도를 기록했습니다. 기존 최첨단 방법들에 비해 20% 이상 향상된 정밀도를 달성했습니다.



### StateAct: State Tracking and Reasoning for Acting and Planning with Large Language Models (https://arxiv.org/abs/2410.02810)
Comments:
          9 pages, 5 pages appendix, 7 figures, 5 tables

- **What's New**: 새로운 연구에서는 대형 언어 모델(LLMs)을 사용하는 '실제' 작업 해결을 위한 계획 및 행동 기술에 대한 간단한 방법인 'StateAct'를 제안합니다. 이 방법은 상태 추적(state-tracking)을 통해 LLM의 'chain-of-thought'를 강화하여 더욱 장기적인 문제 해결이 가능합니다.

- **Technical Details**: StateAct는 few-shot in-context learning을 기반으로 하여 에이전트의 목표(예: 위치 및 인벤토리)를 지속적으로 추적합니다. 새 기술은 Alfworld에서 평가되며, 기존 방법보다 14% 성능 향상을 이루었습니다. 추가적인 데이터나 도구 없이도 성능이 동등한 수준을 유지합니다.

- **Performance Highlights**: StateAct는 여러 LLM에서 효율적으로 작동하며, 작업 해결에 필요한 단계 수를 줄이고 장기적인 문제 해결 능력을 향상시킵니다.  최첨단 성능에 도달한 결과는 LLM 분야에서 중요한 진전을 나타냅니다.



### Estimating Body and Hand Motion in an Ego-sensed World (https://arxiv.org/abs/2410.03665)
Comments:
          Project page: this https URL

- **What's New**: EgoAllo는 헤드 마운트 장치에서 인간의 동작을 추정하는 시스템입니다. 이 시스템은 egocentric SLAM poses와 이미지를 사용하여 3D 몸 자세, 신장(height), 손 파라미터를 추정합니다.

- **Technical Details**: EgoAllo는 조건부 diffusion 모델을 활용하여 인간의 동작을 추정하며, 공간적 및 시간적 불변 특성(criteria)이 모델 성능을 향상시키는 데 기여합니다. 우리는 헤드 모션 조건화 매개변수를 제안하여 최대 18%의 추정 개선 효과를 보여줍니다.

- **Performance Highlights**: EgoAllo 시스템은 손 추정을 개선하여 시끄러운 단안 모노큘러 추정과 비교하여 40% 이상의 추정 오차 감소를 달성합니다.



### Enhance Reasoning by Learning from Mistakes: Peer-Review Knowledge Distillation from Multiple Large Language Models (https://arxiv.org/abs/2410.03663)
Comments:
          14 pages, 5 figures

- **What's New**: 이번 논문에서는 Mistake-Aware Peer-Review Distillation (MAPD) 접근 방식을 소개하여, 학생 모델이 교사 LLM(대형 언어 모델)으로부터 오류에 대한 피드백을 받아 학습하도록 합니다. 이 방법은 기존의 고유한 레퍼런스를 사용하는 것과는 달리, 학생의 실수에 대한 설명과 반영을 통해 맞춤형 학습 데이터를 제공합니다.

- **Technical Details**: MAPD 접근 방식은 1) 학생의 실수를 파악하고 설명하도록 교사에게 요청하여 맞춤형 교육 데이터를 생성합니다. 2) 여러 LLM 간의 시뮬레이션된 동료 검토 프로세스를 설계하여, 기준치 이상으로 생성된 합리적 추론만을 선택하여 교육 데이터의 품질을 향상시킵니다.

- **Performance Highlights**: 수학, 상식, 논리적 추론 과제에 대한 종합 실험 결과, MAPD 방법이 기존보다 향상된 성능을 나타내며, 학생 모델의 전반적인 추론 능력을 개선하는 것으로 나타났습니다.



### Geometric Representation Condition Improves Equivariant Molecule Generation (https://arxiv.org/abs/2410.03655)
- **What's New**: 이번 연구에서는 GeoRCG라는 새로운 프레임워크를 도입하여, 기하학적 표현 조건(geometric representation conditions)을 통합함으로써 분자 생성 모델(molecular generative models)의 성능을 향상시킵니다. 이 방법은 분자 생성을 두 단계로 나누어 첫 번째 단계에서 정보성 기하학적 표현을 생성하고, 두 번째 단계에서 이 표현을 기반으로 분자를 생성하는 방식입니다.

- **Technical Details**: GeoRCG는 분자의 분포(q(ℳ))를 직접 학습하는 것 대신, 잘 훈련된 기하학적 인코더(e.g., Unimol, Frad)를 사용하여 보다 의미 있는 표현 분포로 변환하는 접근 방식을 사용합니다. 이를 통해 분자 구조 및 속성에 대한 정보가 담긴 기하학적 표현이 다음 단계에서 고품질 분자 생성을 유도합니다.

- **Performance Highlights**: GeoRCG는 QM9 및 GEOM-DRUG 데이터셋에서 무조건 분자 생성의 품질을 크게 향상시켰으며, 조건부 분자 생성(task)에서 평균 31%의 성능 향상을 가져왔습니다. 또한, 생성 품질을 유지하면서 확산 단계 수(diffusion steps)를 줄여 생성 과정을 가속화할 수 있음을 보였습니다.



### GenSim2: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs (https://arxiv.org/abs/2410.03645)
Comments:
          CoRL 2024. Project website: this https URL

- **What's New**: 이번 연구에서는 GenSim2라는 확장 가능한 로봇 시뮬레이션 프레임워크를 제안합니다. 이 프레임워크는 복잡하고 현실적인 시뮬레이션 작업 생성을 위해 다중 모달 및 추론 능력을 갖춘 코딩 LLM을 활용합니다.

- **Technical Details**: GenSim2는 200개의 객체로 최대 100개의 관절 작업에 대한 데이터를 자동 생성할 수 있는 계획(solvers) 및 강화학습(RL) 솔버들을 사용하여 객체 카테고리 내에서 일반화합니다. 제안된 다중 작업 언어 조건화 정책 아키텍처인 proprioceptive point-cloud transformer (PPT)는 생성된 데모에서 학습하여 시뮬레이션에서 현실로의 제로샷 전이가 뛰어납니다.

- **Performance Highlights**: GenSim2는 생성된 데이터를 사용하여 제로샷 전이 또는 현실 세계에서 수집된 데이터와 공동 학습할 수 있는 가능성을 보여줍니다. 이는 제한된 실제 데이터에서만 훈련했을 때보다 정책 성능을 20% 향상시킬 수 있습니다.



### Aligning LLMs with Individual Preferences via Interaction (https://arxiv.org/abs/2410.03642)
Comments:
          The code and dataset are made public at this https URL

- **What's New**: 이 논문은 대화형 대형 언어 모델(LLM)이 사용자 개인의 선호도를 반영하여 상호작용하는 능력을 훈련하고, 그들의 행동을 동적으로 일치시키도록 하는 새로운 접근법을 제안합니다. 이는 이전에 단순히 일반적인 원칙(예: 도움을 주고, 해를 끼치지 않고, 정직하게 대화하는 것)에 초점을 맞춘 연구와 차별화됩니다.

- **Technical Details**: 연구진은 사용자 페르소나(user persona)의 다양성을 반영하기 위해 3,310개의 고유한 페르소나를 생성하였으며, 이 페르소나 데이터를 바탕으로 3K 이상의 다중 대화 회차가 포함된 데이터셋을 구축합니다. 대화의 동적 변화를 통해 LLM이 사용자 선호를 추론하고 이에 맞게 응답을 조정할 수 있도록 하는 메타 기술을 발전시킵니다.

- **Performance Highlights**: 알려진 대형 언어 모델들은 개인화된 선호에 맞춰 동적으로 적응하는 데 어려움을 겪지만, 연구에서 제안한 방법은 이러한 능력을 평균 32.0% 향상시키며 LLM이 개인화된 경험을 제공하는 데 주요한 진전을 이루었음을 보여줍니다. ALOE라는 벤치마크를 통해 평가를 실시하였고, 이는 다양한 사용자 선호에 맞춘 대화의 일치 정도를 측정합니다.



### What Matters for Model Merging at Scale? (https://arxiv.org/abs/2410.03617)
Comments:
          20 Pages, 7 Figures, 4 Tables

- **What's New**: 이번 논문에서는 모델 병합(model merging)의 확장성과 관련된 다양한 요소들의 영향을 체계적으로 평가하고, 큰 모델을 기반으로 한 병합의 효용성을 분석합니다.

- **Technical Details**: 1B부터 64B까지의 다양한 모델 크기를 사용하여 4가지 인기 있는 병합 방법인 Averaging, Task Arithmetic, Dare, TIES를 실험합니다. 미세 조정된 모델을 병합하고, 각 병합 모델을 익숙한 작업(held-in tasks)과 전혀 보지 못한 작업(zero-shot tasks)에서 평가합니다.

- **Performance Highlights**: 모델 병합은 강력한 기본 모델(base model)을 사용할 때 더욱 효과적이며, 더 큰 모델일수록 병합을 용이하게 하고 일반화 능력을 지속적으로 향상시키는 것으로 나타났습니다. 8개의 대형 전문가 모델을 병합했을 때, 멀티태스킹(multi-task) 훈련 모델보다 더 나은 일반화 성능을 보였습니다.



### Variational Bayes Gaussian Splatting (https://arxiv.org/abs/2410.03592)
- **What's New**: 최근, Variational Bayes Gaussian Splatting (VBGS)라는 새로운 접근법이 제안되어 Gaussian Splatting을 변분 추론으로 틀 지었습니다. 이 방법은 기억력 고갈 문제를 해결하는 동시에 외부 재생 버퍼 없이 효율적인 학습을 지원합니다.

- **Technical Details**: VBGS는 가우시안 혼합 모델의 매개변수에 대해 변분 추론을 프레임으로 하여 사용할 수 있는 폐쇄형 변분 업데이트 규칙을 도출합니다. 이 접근법은 연속적으로 입력되는 데이터의 부분 관측치를 통해 효율적인 업데이트를 수행할 수 있습니다. 또한, exponential family distributions의 공액적 성질을 활용합니다.

- **Performance Highlights**: VBGS는 TinyImageNet과 Blender 3D 모델 및 Habitat 장면의 데이터셋들을 사용하여 벤치마킹한 결과, 기존 최신 성능을 유지하며 2D 및 3D 데이터의 연속적인 학습에서 성능 향상을 보여주었습니다.



### A Multi-model Approach for Video Data Retrieval in Autonomous Vehicle Developmen (https://arxiv.org/abs/2410.03580)
- **What's New**: 이 논문에서는 SQL 대신 자연어 설명을 사용하여 차량 로그에서 특정 시나리오를 검색할 수 있는 새로운 파이프라인을 제안합니다. 이 접근법은 소프트웨어 개발 워크플로우를 개선할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: Generative AI를 활용하여 시나리오 설명을 생성하고, 이를 벡터 데이터베이스에 저장하여 자연어 쿼리로 검색할 수 있도록 합니다. 이 연구는 다중 모달(multi-modal) 접근 방식을 통해 신호 데이터와 이미지 정보를 결합하여 시나리오 설명을 향상하는 방법에 대해 다룹니다.

- **Performance Highlights**: 엔지니어들이 평가한 결과, 생성된 설명은 평균 3.3점을 기록하였으며, 이는 제안된 접근법이 차량 로그 분석에 있어 유용할 수 있음을 보여줍니다.



### A Survey on Offensive AI Within Cybersecurity (https://arxiv.org/abs/2410.03566)
- **What's New**: AI의 발전과 넷상에서의 범죄적 사용의 결합이 점점 더 우려되고 있는 가운데, 본 논문은 AI 시스템에 대한 공격과 공격적 AI 사용의 여러 측면을 포괄적으로 조사하고자 한다. 특히 공격적 AI가 소비자, 기업, 공공 디지털 인프라 분야에 미치는 영향에 집중하고 있다.

- **Technical Details**: 이 연구는 공격적 AI 관련 주제를 다루며, 적대적 기계 학습(adversarial machine learning), AI 모델 공격, 인프라 및 인터페이스 공격, 그리고 정보 수집, 사회공학(social engineering), 무기화된 AI와 같은 공격 기법을 포함한다. MLOps(Machine Learning Operations) 생애주기도 다루며, ML 모델을 제작, 테스트, 배포, 모니터링하는 과정에서의 보안 고려사항을 논의한다.

- **Performance Highlights**: AI는 사이버 보안에서 공격 및 방어 양쪽 역할을 담당하고 있으며, 공격자들은 AI를 비즈니스, 건강 관리 및 금융 서비스 등 다양한 분야에서 무기로 사용할 수 있다. 본 논문은 공격적 AI의 다양한 형태와 그로 인한 실제 사례 및 기존 사례 연구를 통해 이 분야에서의 기회와 도전을 탐색한다.



### Training on more Reachable Tasks for Generalisation in Reinforcement Learning (https://arxiv.org/abs/2410.03565)
Comments:
          arXiv admin note: text overlap with arXiv:2406.08069

- **What's New**: 이 논문에서는 다중작업 강화학습(multi-task reinforcement learning)에서 탐색(exploration)과 일반화(generalization)의 관계를 탐구하고, '도달 가능성(reachability)' 개념을 도입하여 탐색의 역할을 명확히 합니다. 새로운 메소드인 Explore-Go를 제안하여 에피소드 초기에 탐색 단계를 도입함으로써 에이전트가 도달 가능한 작업의 수를 증가시킵니다.

- **Technical Details**: 도달 가능한 작업은 훈련 중에 만날 수 있는 상태와 보상을 포함하는 작업이며, 도달 불가능한 작업은 훈련 작업에 어떤 상태/보상도 공유하지 않는 작업으로 정의됩니다. Explore-Go는 기존의 on-policy나 off-policy 강화학습 알고리즘과 결합될 수 있으며, 에이전트의 경험 수집 방식을 수정하여 작동합니다.

- **Performance Highlights**: Explore-Go를 다양한 환경에서 인기 있는 알고리즘과 결합했을 때, 도달 가능 및 도달 불가능한 작업에 대한 일반화 성능이 향상되었음을 실험적으로 보여주었습니다. 이 연구는 에이전트가 탐색하는 시점과 최적의 도달 가능 작업 수가 일반화 성능과 더 밀접하게 관련되어 있음을 강조합니다.



### Optimizing food taste sensory evaluation through neural network-based taste electroencephalogram channel selection (https://arxiv.org/abs/2410.03559)
Comments:
          33 pages, 13 figures

- **What's New**: 본 논문은 taste 전기뇌파(electroencephalogram, EEG) 데이터에서 채널 선택을 위한 새로운 방법인 class activation mapping with attention (CAM-Attention)을 제안합니다.

- **Technical Details**: CAM-Attention 방법은 convolutional neural network과 channel 및 spatial attention 모델(CNN-CSA)과 gradient-weighted class activation mapping(Grad-CAM) 모델을 결합하여 EEG 데이터의 핵심 기능을 주목하는 방법론을 사용합니다.

- **Performance Highlights**: 이 방법은 taste EEG 인식의 컴퓨팅 부하를 줄이고, 네 가지 맛을 효과적으로 구별하여 뛰어난 인식 성능을 발휘하며, 맛 감각 평가를 위한 기술적 지원을 제공합니다.



### Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features (https://arxiv.org/abs/2410.03558)
- **What's New**: 이번 논문에서는 diffusion 모델의 활성화(activations)를 활용한 특성 선택(feature selection)을 심층적으로 연구합니다. 과거 연구에서는 활성화의 평가 범위가 제한적이었으나, 많은 새로운 활성화가 등장함에 따라 우수한 선택 방법을 제안합니다.

- **Technical Details**: Diffusion 모델은 Gaussian 노이즈를 점진적으로 제거하여 이미지를 생성하는 강력한 generative 모델입니다. 이 연구에서는 U-Net 아키텍처 내에서 활성화를 추출하여 이들을 feature로 사용합니다. 이를 통해 활성화의 질이 성능에 미치는 영향을 분석하고, 효과적인 활성화 선택을 위한 세 가지 범주를 제시합니다: (i) 확산 노이즈의 매크로 수준, (ii) 해상도 내 세분화의 변화, (iii) 위치적 임베딩 없는 지역성.

- **Performance Highlights**: 여러 가지 discriminative 작업에서 실험을 통해 제안한 방법이 기존의 SOTA(superior of the art) 기법보다 우수한 성능을 발휘함을 입증하였습니다. 이로써 활성화 선택이 모델 성능 향상에 중요한 역할을 함을 보여줍니다.



### Evaluating Investment Risks in LATAM AI Startups: Ranking of Investment Potential and Framework for Valuation (https://arxiv.org/abs/2410.03552)
Comments:
          21 pages, 7 figures, 8 tables, Accepted for publication to the International Association for Applied Business Research Journal (IAABR)

- **What's New**: 이번 연구는 라틴 아메리카의 온라인 음식 배달 산업을 중심으로 한 기술 스타트업의 가치 평가 모델을 제시하며, Total Addressable Market (TAM), Serviceable Available Market (SAM), Serviceable Obtainable Market (SOM) 메트릭스를 사용하여 새로운 접근 방식을 제공합니다.

- **Technical Details**: 라틴 아메리카의 테크 스타트업 생태계는 2010년에서 2020년 사이에 32배 성장하며, VC 투자는 콜롬비아, 멕시코 등의 국가에서 1억 달러 이상에 도달하는 등 두각을 나타내고 있습니다. DCF (Discounted Cash Flow) 방법을 통해 스타트업의 가치를 평가하는 다양한 방법론이 소개됩니다.

- **Performance Highlights**: AI 중심 스타트업이 예상되는 2030년까지 실질 GDP에 5.4% 기여할 것으로 보이며, 이러한 성장은 라틴 아메리카에서의 혁신과 경제 발전에 중요한 요소로 작용할 것입니다.



### Constructive Apraxia: An Unexpected Limit of Instructible Vision-Language Models and Analog for Human Cognitive Disorders (https://arxiv.org/abs/2410.03551)
- **What's New**: 이번 연구는 instructible vision-language models (VLMs)와 인지 장애인 constructive apraxia 사이의 예상치 못한 유사점을 밝혀냈습니다.

- **Technical Details**: 25개의 최신 VLM 모델, 예를 들면 GPT-4 Vision, DALL-E 3, Midjourney v5를 테스트했습니다. 이들은 Ponzo illusion의 이미지를 생성하는 능력을 평가받았으며, 이는 기본적인 spatial reasoning을 요구합니다. 연구 결과, 25개 모델 중 24개가 수평선을 올바르게 렌더링하지 못했습니다. 이는 parietal lobe 손상이 있는 환자들에게서 보이는 결핍과 유사합니다.

- **Performance Highlights**: 모델들은 공간적 지침을 잘못 해석하여, 배경의 원근감에 맞춰 기울거나 정렬되지 않은 선을 생성했습니다. 이러한 행동은 apraxia 환자들이 간단한 도형을 복사하거나 구성하는 데 어려움을 겪는 방식과 유사합니다. 현재 VLM들은 다른 영역에서는 뛰어난 기능을 보이지만, 기본적인 spatial reasoning 능력에는 한계를 보이고 있습니다. 이러한 AI 시스템의 제약은 spatial cognition 결핍을 연구하기 위한 새로운 계산 모델을 제시하며, VLM의 구조 및 훈련 방법론에서 개선이 필요한 중요한 영역으로 부각됩니다.



### Dreamming User Multimodal Representation for Micro-Video Recommendation (https://arxiv.org/abs/2410.03538)
- **What's New**: 이번 연구에서는 DreamUMM (Dreaming User Multi-Modal Representation)이라는 새로운 접근 방식을 제안하여 실시간 사용자 관심사를 다중 모드(multimodal) 공간에서 모델링합니다. 이는 사용자 역사 데이터를 활용하여 동적 관심사를 포착하는 데 중점을 두고 있습니다.

- **Technical Details**: DreamUMM은 사용자의 역사적 상호작용을 바탕으로 사용자의 비디오 선호도와 다중 모드 유사성을 연관짓는 폐쇄형 솔루션을 사용합니다. 또한, Candidate-DreamUMM을 통해 최근 사용자 행동 데이터가 부족한 상황에서도 후보 비디오에서 관심사를 추론할 수 있도록 설계되었습니다. 이 과정에서 대형 언어 모델(large language models)과 지식 증류(knowledge distillation)를 활용하여 비디오의 복잡한 시각적, 청각적, 텍스트 요소 간의 상호작용을 포착하는 고품질 다중 모드 임베딩을 생성합니다.

- **Performance Highlights**: 광범위한 온라인 A/B 테스트를 통해 사용자 참여 지표(활동 일수 및 재생 수 등)에서 눈에 띄는 개선이 나타났습니다. DreamUMM은 수억 명의 일간 활성 사용자(daily active users)를 가진 두 개의 마이크로 비디오 플랫폼에 성공적으로 배포되어 실용성과 확장성을 입증했습니다.



### Ward: Provable RAG Dataset Inference via LLM Watermarks (https://arxiv.org/abs/2410.03537)
- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG) 시스템에서의 비인가 사용을 탐지하는 방법을 제시합니다. 특히, 데이터 소유자가 자신의 콘텐츠가 어떻게 사용되는지에 대한 강력한 통계적 보장을 받을 수 있는 방법에 중점을 두었습니다.

- **Technical Details**: 연구는 문제를 (black-box) RAG Dataset Inference (RAG-DI)로 공식화하였고, 현실적인 조건에서 RAG-DI 방법들을 벤치마킹하기 위한 새로운 데이터셋을 도입하였습니다. 또한, LLM watermark를 활용한 새로운 RAG-DI 방법인 Ward를 소개합니다. 이 방법은 데이터 소유자가 데이터셋 사용에 대한 통계적 보장을 받을 수 있도록 합니다.

- **Performance Highlights**: Ward는 다양한 어려운 환경에서 모든 기준선(baseline) 방법들을 지속적으로 능가하며, 더 높은 정확도(accuracy), 우수한 쿼리 효율성(query efficiency) 및 강건성(robustness)을 달성합니다. 이 연구는 향후 RAG-DI 연구의 기초를 제공하며, LLM watermark를 효과적인 솔루션으로 강조합니다.



### Computer Vision Intelligence Test Modeling and Generation: A Case Study on Smart OCR (https://arxiv.org/abs/2410.03536)
- **What's New**: AI 기반 시스템의 품질 평가가 중요해지면서, 본 논문에서는 AI 소프트웨어의 기능 테스트 모델을 제안합니다. 기존 문헌에 대한 포괄적인 리뷰를 제공하고, 이미지 기반 텍스트 추출 AI 기능을 평가하기 위한 3D 분류 모델을 소개합니다.

- **Technical Details**: AI 소프트웨어 테스트는 전통적인 테스트와 달리 대규모 비구조적 입력 데이터, 예측 불가능한 시나리오, 시스템 출력의 불확실성 등을 고려해야 합니다. 이를 해결하기 위해, Modified CER와 WER, METAMORPHIC Testing(메타모픽 테스트) 기법 및 다양한 테스트 프레임워크가 활용됩니다. 또한, OCR 성능 평가는 일반적으로 Ground Truth(GT)와의 편집 거리 비교로 이루어집니다.

- **Performance Highlights**: 제안된 테스트 모델은 모바일 Optical Character Recognition(OCR) 사례를 통해 검증되었으며, 다양한 평가 지표로 AI 기능 품질을 효과적으로 평가할 수 있음을 보여줍니다. 연구는 AI 소프트웨어 테스트에 대한 체계적인 접근 방식을 제공하며, 향후 질문과 연구 방향을 제시합니다.



### Multiscale fusion enhanced spiking neural network for invasive BCI neural signal decoding (https://arxiv.org/abs/2410.03533)
- **What's New**: 이번 논문에서는 멀티스케일 융합(Multiscale Fusion) 향상 스파이킹 신경망(Spiking Neural Network, SNN)을 이용하여 뇌-컴퓨터 인터페이스(Brain-Computer Interface, BCI)의 신경 신호 디코딩을 향상시키는 새로운 방법론을 제안합니다. MFSNN은 인간의 시각 인식을 모방하여 실시간으로 효율적이고 에너지 절약적인 신경 신호 디코딩을 가능하게 합니다.

- **Technical Details**: MFSNN은 입력된 스파이크 신호를 여러 서브 신호로 나누어 병렬 처리하고, 이 구조는 채널 주의 메커니즘(channel attention mechanisms)과 시계열 컨볼루션 네트워크를 통해 데이터에서 시공간적(spatiotemporal) 특징을 추출합니다. 또한, 스킵 연결(skip connections)을 통해 디코딩 성능을 향상시킵니다. 이 방법은 미니 배치 감독 일반화 학습(mini-batch supervised generalization learning) 기법을 통해 모든 날짜에 걸쳐 신호 디코딩의 일반성과 강건성을 개선합니다.

- **Performance Highlights**: MFSNN은 단일 손 그립 및 터치, 중심-외각 도달 작업 등 두 가지 침습적 BCI 작업에서 전통적인 인공 신경망(Artificial Neural Network, ANN) 방식인 MLP 및 GRU를 능가하는 정확도 및 계산 효율성을 보여주었습니다. 또한, MFSNN의 멀티스케일 특징 융합 프레임워크는 뉴로모픽 칩(neuromorphic chips)에서의 적용이 가능하여 침습적 BCI 신호의 온라인 디코딩을 위한 에너지 효율적인 솔루션을 제공합니다.



### MARE: Multi-Aspect Rationale Extractor on Unsupervised Rationale Extraction (https://arxiv.org/abs/2410.03531)
Comments:
          Accepted in EMNLP2024(Main) conference

- **What's New**: 이 논문에서는 Multi-Aspect Rationale Extractor (MARE)를 제안하여 여러 측면을 동시에 예측하고 설명합니다. MARE는 기존의 Uni-Aspect 모델의 한계를 극복하고 다양한 측면 간의 내부 상관관계를 활용하여 성능을 향상시키고자 합니다.

- **Technical Details**: MARE는 하드 삭제 기반의 Multi-Aspect Multi-Head Attention (MAMHA) 메커니즘을 통해 입력 텍스트에 여러 특수 토큰을 추가하고, 이를 통해 여러 텍스트 조각을 동시에 인코딩합니다. 또한, 다중 작업 훈련(Multi-task training)을 적용하여 훈련 비용을 절감합니다.

- **Performance Highlights**: MARE는 BeerAdvocate와 Hotel Review 데이터셋의 두 개의 비지도 라지오 분석 벤치마크에서 기존의 최신 방법보다 월등한 성능을 보였으며, 토큰 수준의 F1 스코어에서 5.4% 향상을 기록했습니다.



### A Probabilistic Perspective on Unlearning and Alignment for Large Language Models (https://arxiv.org/abs/2410.03523)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)의 평가를 위한 첫 번째 공식적인 확률적 평가 프레임워크를 도입합니다. 기존의 결정론적 (deterministic) 평가 방식은 모델의 전체 출력 분포를 포착하지 못해 모델의 능력을 정확하게 추정하는 데 한계가 있음을 증명했습니다.

- **Technical Details**: 논문에서 제안한 새로운 지표는 모델의 출력 분포와 관련된 높은 확률 보장을 기반으로 합니다. 이 지표는 응용 프로그램에 의존하지 않으며, 사용자가 배포 전에 모델의 능력을 보다 신뢰성 있게 추정할 수 있게 해줍니다. 더불어, 엔트로피 최적화 (entropy optimization) 및 적응형 온도 조정 (adaptive temperature scaling)에 기초한 새로운 유학 손실도 제안합니다.

- **Performance Highlights**: 연구 결과, 기존의 결정론적 평가 방식에서는 성공적인 유학을 나타내는 반면, 확률적 평가 결과에 따르면 대부분, 아니면 모든 유학되지 않은 정보가 모델 내에 여전히 접근 가능하다는 것을 보여주었습니다. 이는 LLMs의 평가 방식에 있어 중요한 혁신을 제시합니다.



### LCMDC: Large-scale Chinese Medical Dialogue Corpora for Automatic Triage and Medical Consultation (https://arxiv.org/abs/2410.03521)
- **What's New**: COVID-19 팬데믹으로 인해 전통적인 헬스케어 시스템의 한계가 드러나면서, 온라인 의료 서비스와 특히 의료 triage 및 상담의 발전이 가속화되었습니다. 이를 해결하기 위해 대규모 의료 대화 데이터셋 (Large-scale Chinese Medical Dialogue Corpora, LCMDC) 이 구축되었으며, 이는 Coarse-grained Triage, Fine-grained Diagnosis, Medical Consultation의 세 가지 데이터셋으로 구성되어 있습니다.

- **Technical Details**: LCMDC는 443,630개의 샘플로 구성된 Coarse-grained Triage 데이터셋, 199,600개의 샘플로 구성된 Fine-grained Diagnosis 데이터셋, 472,418개의 항목으로 구성된 Medical Consultation 데이터셋을 포함하고 있습니다. 또한 BERT 기반의 감독 학습과 prompt learning을 결합한 새로운 triage 시스템을 제안하고, 강화 학습을 이용한 GPT 기반의 의료 상담 모델을 개발했습니다. PLMs(Pre-trained Language Models)를 의료 지식으로 사전 학습하여 도메인 지식 습득을 강화했습니다.

- **Performance Highlights**: LCMDC에서의 실험 결과, 제안한 시스템이 기존 모델보다 탁월한 성능을 보였으며, 특히 희귀 질환에 대한 예측 정확도가 5% 향상되었습니다. 학습된 모델은 사용자 질문에 대한 정확한 의학적 답변을 제공하는 데 효과적이라는 것을 입증했습니다.



### FedStein: Enhancing Multi-Domain Federated Learning Through James-Stein Estimator (https://arxiv.org/abs/2410.03499)
Comments:
          12 pages, 2 figures. Accepted at International Workshop on Federated Foundation Models In Conjunction with NeurIPS 2024 (FL@FM-NeurIPS'24)

- **What's New**: 이번 연구에서는 Multi-Domain Federated Learning(FL)에서의 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. FedStein이라는 방법을 통해 James-Stein Estimator를 활용하여 Batch Normalization 통계를 공유하고, 데이터의 다양성을 극복하여 성능 향상에 기여합니다.

- **Technical Details**: FedStein은 클라이언트 간에 Batch Normalization(BN) 통계의 James-Stein 추정값만 공유하므로, 기존의 방법들보다 데이터의 독립성과 정규성에 대한 문제 해결을 돕습니다. Non-BN 레이어의 매개변수는 일반적인 Federated Learning 기법을 이용하여 교환되며, 이 과정에서 모델의 성능을 유지합니다.

- **Performance Highlights**: FedStein은 3개의 데이터셋과 여러 모델에서 실험을 통해 기존의 FedAvg 및 FedBN 방법들과 비교하였을 때, 특정 도메인에서 14% 이상의 정확도 향상을 기록하였으며, 이러한 결과는 다양한 도메인 일반화에 긍정적인 영향을 미쳤습니다.



### Generative Artificial Intelligence for Navigating Synthesizable Chemical Spac (https://arxiv.org/abs/2410.03494)
- **What's New**: SynFormer는 합성 가능한 화학 공간을 효과적으로 탐색하는 생성 모델링 프레임워크로, 전통적인 분자 생성 방식과는 달리 분자의 합성 경로(synthetic pathways)를 생성하여 설계가 실제로 합성 가능한지를 보장합니다.

- **Technical Details**: 이 프레임워크는 확장 가능한 트랜스포머 아키텍처(transformer architecture)와 노이즈 제거 확산 모듈(denoising diffusion module)을 결합하여 분자 구축 블록을 선택합니다. SynFormer는 크게 두 가지 형태로 구현됩니다: (1) SynFormer-ED, 주어진 입력 분자에 대응하는 합성 경로를 생성하는 인코더-디코더 모델이며, (2) SynFormer-D, 특정 속성 목표를 위한 합성 경로를 생성하는 디코더 전용 모델입니다. 두 모델 모두 115개 반응 템플릿과 223,244개의 상업적으로 판매 가능한 구축 블록을 기반으로 훈련되었습니다.

- **Performance Highlights**: SynFormer는 (a) Enamine REAL과 ChEMBL 화학 공간 내의 분자 재구성, (b) 참조 분자를 기반으로 하는 지역 합성 가능한 화학 공간 탐색, (c) 블랙박스 속성 예측 모델에 의해 안내되는 전 세계 합성 가능한 화학 공간 탐색에서 성공을 거두었습니다. 이는 SynFormer의 다양한 제어 전략을 통해 합성 가능한 화학 공간을 탐색하고, 실제 분자 설계 사례에서의 적용 가능성을 높인다는 점에서 중요합니다.



### Gradient-based Jailbreak Images for Multimodal Fusion Models (https://arxiv.org/abs/2410.03489)
- **What's New**: 이 논문에서는 멀티모달 퓨전 모델(multimodal fusion models)의 공격을 위해 토크나이저 숏컷(tokenizer shortcut) 개념을 도입하여 연속 최적화(continuous optimization)를 가능하게 합니다. 이는 이미지 입력을 통해 더 효과적인 jailbreak 공격을 가능하게 하며, 텍스트 입력에 비해 많은 계산 자원을 절약할 수 있습니다.

- **Technical Details**: 리서치에서는 토크나이저 숏컷을 사용하여 다중 모드 퓨전 모델에 대한 최초의 엔드-투-엔드 그래디언트 이미지 공격을 구현하였습니다. 이 방법은 이미지 토크나이제이션(tokenization)을 연속 함수로 근사화하여 연속 최적화를 가능하게 하고, 원래 이미지와는 전혀 구별되는 이미지를 사용할 수 있게 합니다. 두 가지 숏컷 설계가 제안되었으며, 이를 통해 얻어진 이미지는 높은 성공률을 보였습니다.

- **Performance Highlights**: 이 연구에서 최적화된 jailbreak 이미지는 Chameleon 모델에서 72.5%의 프롬프트(prompts)에 대해 유해한 응답을 이끌어내며, 텍스트 기반의 공격보다 우수한 성능을 나타냈습니다. 또한 동일한 목표로 최적화된 텍스트 기반 공격보다 3배 적은 컴퓨팅 예산(compute budget)을 요구하며, 50배 많은 입력 토큰을 최적화할 수 있었습니다. 하지만, 모델 간 전이는 이루어지지 않았습니다.



### A Multimodal Framework for Deepfake Detection (https://arxiv.org/abs/2410.03487)
Comments:
          22 pages, 14 figures, Accepted in Journal of Electrical Systems

- **What's New**: 딥페이크(dipfake) 기술의 급속한 발전이 디지털 미디어의 무결성에 중대한 위협을 가하고 있습니다. 본 연구에서는 시각 및 청각 요소를 모두 포괄하는 혁신적인 다중 모달(multimodal) 접근 방식을 통해 딥페이크 문제를 해결하고자 했습니다.

- **Technical Details**: 우리의 모형은 고급 특성 추출 기법을 사용하여 비디오의 아홉 가지 개별적인 얼굴 특징을 추출하고, 다양한 머신러닝(machin learning) 및 딥러닝(deep learning) 모델을 적용했습니다. 오디오 분석을 위해 mel-spectrogram 분석을 활용하여 특징을 추출하고, 동일한 방식으로 머신러닝 및 딥러닝 기법을 적용했습니다. 우리 모형은 비디오 및 오디오 분류를 위해 인공신경망(Artificial Neural Network)과 VGG19를 사용하여 전체 샘플을 딥페이크로 분류합니다.

- **Performance Highlights**: 우리의 다중 모달 프레임워크는 시각 및 청각 분석을 결합하여 94%의 정확도를 달성했습니다.



### Group Fairness in Peer Review (https://arxiv.org/abs/2410.03474)
Comments:
          A preliminary version appeared at NeurIPS 2023

- **What's New**: 이번 연구에서는 대규모 AI 학술대회에서 다양한 연구 커뮤니티의 제출물이 서로 검토되어야 하는 문제를 해결하기 위해 'core'라는 그룹 공정성 개념을 소개합니다. 이 개념은 어떤 커뮤니티도 대회에서 독립적으로 이익을 취할 수 없도록 보장합니다.

- **Technical Details**: 대규모 학술대회는 자동화된 절차를 통해 제출된 논문에 대한 검토자를 배정합니다. 본 연구는 'core' 개념을 통해 각 연구자의 제출물에 대한 검토 권한을 적절히 배분하는 효율적인 알고리즘을 제안합니다.

- **Performance Highlights**: CVPR 및 ICLR에서 실제 데이터를 사용하여 제안된 알고리즘이 기존 검토 배정 방법들과 비교해 그룹 공정성을 유지하면서도 일정한 사회적 후생을 희생해야 함을 보여주었습니다.



### Vulnerability Detection via Topological Analysis of Attention Maps (https://arxiv.org/abs/2410.03470)
Comments:
          Accepted to ITaS2024. Contains 8 pages

- **What's New**: 최근 취약점 감지(vulnerability detection)에 대한 딥러닝(deep learning, DL) 접근 방식이 주목받고 있으며, 이 연구에서는 BERT 모델의 attention 행렬을 활용하여 새로운 방법을 탐구했습니다. 전통적인 머신러닝(machine learning, ML) 기법이 추출된 최상위 특징을 기반으로 경쟁력 있는 성능을 보였다는 점이 주요 발견입니다.

- **Technical Details**: 본 연구에서는 topological data analysis (TDA) 도구를 사용하여 BERT 모델의 attention 행렬에서 취약점 감지를 위한 특징을 추출하고, 이를 통해 생성된 attention 그래프의 지속적 동질성(persistent homology)을 계산하여 상징적인 정보를 포착합니다. TDA는 데이터를 분석할 때 특정 임계값을 설정하는 대신, 여러 점의 연결성(clusters) 및 주기적 구조(cycles)를 확인할 수 있게 해줍니다.

- **Performance Highlights**: Logistic Regression, Support Vector Machine(SVM), Gradient Boosting 분류기를 대상으로 한 실험에서는, 제안된 방법이 Devign 데이터셋에서 효과적인 성능을 발휘하여 CodeBERTa와 동등한 수준의 경쟁력이 있을 수 있음을 입증했습니다. 이는 전통적인 정적 코드 분석 도구보다 유의미한 장점으로 작용할 수 있습니다.



### Diffusion State-Guided Projected Gradient for Inverse Problems (https://arxiv.org/abs/2410.03463)
Comments:
          preprint. under review. RZ and BT have equal contributions

- **What's New**: Diffusion State-Guided Projected Gradient (DiffStateGrad)라는 새로운 접근 방식을 제안하여, 데이터 매니폴드(data manifold) 상에 남아있도록 하여 역문제(inverse problems)를 해결하는 데 있어 확산 모델의 성능 및 견고성을 향상시킵니다.

- **Technical Details**: DiffStateGrad는 측정 가이던스.gradient을 저차원 저랭크 서브스페이스(low-rank subspace)에 투영하는 과정을 포함하며, 이 서브스페이스는 확산 과정의 중간 상태(intermediate state)를 근사합니다. 이 과정은 singular value decomposition (SVD)를 통해 수행되며, 저차원 투영을 통해 지역 매니폴드 구조에 수직한 방향을 제거합니다.

- **Performance Highlights**: DiffStateGrad는 확산 모델의 시간 가이던스 스텝 크기(measurement guidance step size)와 노이즈에 대한 견고성을 개선하며, 조건부 화소 결합을 사용하여 이미지 복원 시의 성능을 상승시킵니다. 예를 들어, 큰 스텝 크기와 높은 측정 노이즈 상황에서 PSNR이 20 미만으로 떨어지는 실패율을 크게 줄였습니다.



### How Toxicity Classifiers and Large Language Models Respond to Ableism (https://arxiv.org/abs/2410.03448)
- **What's New**: 이 연구는 장애인을 대상으로 한 유해한 온라인 콘텐츠의 식별 및 설명에 있어 최신 기계 학습 모델, 즉 toxicity classifiers (TCs)와 large language models (LLMs)의 효과를 평가합니다.

- **Technical Details**: 연구에서는 100개의 소셜 미디어 댓글(시장성 85개, 비시장성 15개)에 대한 데이터셋을 구축하고, 160명의 참여자(장의 100명, 비장애인 60명)에게 댓글의 독성 및 장애 혐오도를 평가하도록 요청했습니다. 또한, TCs와 LLMs에게 댓글의 독성과 장애 혐오도를 평가하고 설명하도록 요청했습니다.

- **Performance Highlights**: TCs는 장애인보다 독성을 훨씬 낮게 평가했으며, LLMs는 장애인과 비슷한 수준으로 장애 혐오도를 평가했습니다. 그러나 LLMs의 장애 혐오에 대한 설명은 감정적 피해를 간과하고, 특정성 및 맥락 인정을 부족했습니다.



### Exploring the Benefit of Activation Sparsity in Pre-training (https://arxiv.org/abs/2410.03440)
Comments:
          ICML 2024

- **What's New**: 이 논문에서는 Pre-trained Transformers의 sparse activation에 대한 연구를 진행하며, 이를 활용한 새로운 학습 방식인 Switchable Sparse-Dense Learning (SSD)을 제안한다. SSD는 sparse training과 conventional dense training을 동적으로 전환하여 효율성을 높인다.

- **Technical Details**: 이 연구는 GPT, BERT 및 T5 모델을 사용하여 sparse activation의 변화 양상을 조사하였다. SSD는 초기에는 dense training을 수행하고, activation sparsity가 높아지면 sparse training으로 전환하여 Sparse-activated Mixture-of-Experts (SMoE) 모델로 대체한다. 이 딥러닝 프레임워크는 다단계 동적 학습을 통해 모델의 성능을 극대화한다.

- **Performance Highlights**: SSD는 동일한 모델 크기에서 기존 dense training 대비 유사한 성능을 유지하면서, 전이 훈련 비용을 감소시킨다. 예를 들어 FLOPs에서 최대 1.44배의 속도 향상과 함께, inference 속도는 최대 2배 빨라졌다. SSD에서 학습된 모델은 추가적인 훈련 없이도 SMoE 모델로 사용 가능하다.



### A General Framework for Producing Interpretable Semantic Text Embeddings (https://arxiv.org/abs/2410.03435)
Comments:
          19 pages, 5 figures, and 9 tables

- **What's New**: CQG-MBQA(Contrastive Question Generation - Multi-task Binary Question Answering)는 다양한 작업에 대한 해석 가능한 의미 텍스트 임베딩을 생성하기 위한 일반적인 프레임워크로, 고비용의 전문가 지식이나 정밀한 프롬프트 설계 없이도 높은 차별성을 지닌 질문을 체계적으로 생성한다.

- **Technical Details**: 이 프레임워크는 CQG 방법론을 활용하여 저 인지 부하의 Yes/No 질문을 생성하고, MBQA 모델을 통해 이를 효율적으로 답변함으로써 해석 가능한 임베딩을 비용 효율적으로 생성한다. CQG는 LLM을 활용하여 텍스트 간의 의미적 뉘앙스를 포착하는 이진 질문을 강조하여 임베딩 공간의 차원을 형성한다.

- **Performance Highlights**: CQG-MBQA는 블랙박스 모델과 유사한 품질의 임베딩을 제공하며, 다양한 다운스트림 작업에서 기존의 해석 가능한 텍스트 임베딩 방법들을 초월하는 성능을 보여준다.



### Self-supervised Spatio-Temporal Graph Mask-Passing Attention Network for Perceptual Importance Prediction of Multi-point Tactility (https://arxiv.org/abs/2410.03434)
Comments:
          Published as a conference paper at Eurohaptics 2024

- **What's New**: 이번 연구에서는 멀티포인트 촉각 인식 시나리오에서 진동 촉각 신호의 지각적 중요성을 예측하기 위한 Self-supervised Spatio-Temporal Graph Mask-Passing Attention Network(SSTGMPAN) 모델을 제안합니다.

- **Technical Details**: 제안하는 모델은 다차원 시간 시계열 촉각 신호를 입력으로 받아 각 상호작용 포인트의 중요도 지수를 출력합니다. 모델은 그래프 신경망(GNN) 기반으로, 특히 시공간 그래프 신경망(STGNN)을 활용하여 시간 및 공간 도메인에서의 마스킹 관계를 분석합니다.

- **Performance Highlights**: 초기 실험 결과, 제안한 모델이 멀티포인트 촉각 인식에서 서로 다른 포인트의 지각적 중요성을 효과적으로 예측할 수 있음을 보여주었습니다.



### EB-NeRD: A Large-Scale Dataset for News Recommendation (https://arxiv.org/abs/2410.03432)
Comments:
          11 pages, 8 tables, 2 figures, RecSys '24

- **What's New**: 에크스트라 블라뎃 뉴스 추천 데이터셋(EB-NeRD)이 도입되었습니다. 이 데이터셋은 100만 명 이상의 고유 사용자의 데이터와 3,700만 개 이상의 인상 로그를 포함하고 있으며, 125,000개 이상의 덴마크 뉴스 기사를 포함하고 있습니다. EB-NeRD는 RecSys '24 챌린지의 기준 데이터셋으로 활용되었습니다.

- **Technical Details**: EB-NeRD 데이터셋은 사용자 행동 로그로부터 수집되었으며, 다양한 기술적 문제를 해결하기 위한 연구를 지원합니다. 여기에는 뉴스 기사의 연속적인 출간, 신속한 소멸 문제, 사용자 피드백 기반의 모델링 기법이 포함됩니다. 또한, 텍스트 정보를 활용하는 방법도 강조됩니다.

- **Performance Highlights**: EB-NeRD 데이터셋은 고전적인 디지털 뉴스 게시자의 각기 다른 뉴스 소비 및 콘텐츠 프로필에 대한 일반화 가능성을 탐색할 수 있는 기회를 제공합니다. 데이터셋은 뉴스 추천 시스템의 설계에서 기술적 및 규범적 도전을 해결하는 데 매우 유용합니다.



### Cayley Graph Propagation (https://arxiv.org/abs/2410.03424)
Comments:
          20 pages, 6 figures

- **What's New**: 이 연구는 Graph Neural Networks (GNNs)의 정보 흐름 문제를 해결하기 위해 새로운 접근 방식인 CGP (Complete Cayley Graph Propagation)를 제안합니다. CGP는 과도한 정보 압축(over-squashing)을 방지하기 위해 완전한 Cayley 그래프 구조를 활용하여 정보 전파를 강화합니다.

- **Technical Details**: CGP는 Cayley graphs of the SL(2,ℤ_n) 특수 선형 그룹을 기반으로 하며, 노드 간의 정보 흐름이 원활해집니다. 이 연구는 GNN의 데이터 처리 구조를 변경하여, 정보가 병목 현상 없이 원활하게 전달되도록 합니다.

- **Performance Highlights**: 여러 실제 데이터 세트를 통한 실험 결과, CGP는 EGP (Expander Graph Propagation)와 비교하여 상당한 성능 향상을 보였으며, 또한 복잡한 그래프 재구성과 유사하거나 더 나은 성능을 나타냈습니다.



### One2set + Large Language Model: Best Partners for Keyphrase Generation (https://arxiv.org/abs/2410.03421)
Comments:
          Accepted by EMNLP 2024 Main Conference

- **What's New**: 이 논문에서는 기존의 keyphrase generation (KPG) 방식을 개선하기 위해 generate-then-select 프레임워크를 도입하였습니다. 이 프레임워크는 KPG를 두 단계로 분해하여 각각 generator와 selector가 수행합니다.

- **Technical Details**: 저자들은 one2set 기반 모델을 generator로 사용하여 후보 키프레이즈를 생성하고, LLM(large language model)을 selector로 사용하여 이들 후보 중에서 최종 키프레이즈를 선택합니다. 특히, Optimal Transport 기반의 할당 전략을 설계하여 supervision 신호의 부적절한 할당 문제를 해결하고, 키프레이즈 선택을 시퀀스 레이블링 태스크로 모델링하여 중복 선택 문제를 완화했습니다.

- **Performance Highlights**: 여러 벤치마크 데이터셋을 통한 실험 결과, 이 프레임워크는 특히 부재 키프레이즈 예측에서 기존 최첨단 모델들을 유의미하게 초월하는 성능을 보였습니다.



### Towards Real-time Intrahepatic Vessel Identification in Intraoperative Ultrasound-Guided Liver Surgery (https://arxiv.org/abs/2410.03420)
Comments:
          MICCAI 2024, Oct 2024, Marrakech, Morocco

- **What's New**: 이번 연구에서는 복강경( laparoscopic ) 간절제술에서 간의 내부 구조를 효과적으로 식별하고, 환자 개인 맞춤형 접근 방식을 통해 3D 초음파( ultrasound )를 기반으로 한 AI 모델을 개발하였습니다.

- **Technical Details**: 사전 수술 3D 초음파 간 용적을 사용하여 심층 학습( deep learning ) 모델을 훈련시켜 실시간( real-time )으로 간문맥( portal tree )과 가지 구조( branch structures )를 식별하는 방법을 제안하였습니다.

- **Performance Highlights**: 개인 맞춤형 AI 모델은 ex vivo( 생체 외 ) 돼지 간을 대상으로 검증되었으며, 외과의사에 비해 정확도( precision )는 0.95, 재현율( recall )은 0.93으로 우수한 성능을 보였습니다. 이는 초음파 기반의 간 절제술에서 정밀한 혈관 식별을 위한 기초를 마련합니다.



### Comparative study of regression vs pairwise models for surrogate-based heuristic optimisation (https://arxiv.org/abs/2410.03409)
- **What's New**: 이번 연구는 서브로게이트 모델(surrogate models)을 활용하여 비싼 피트니스 함수(fitness function)의 계산 비용을 완화하는 새로운 접근 방식을 제시합니다. 특히, 페어와이즈 서브로게이트 모델(pairwise surrogate models)을 통해 피트니스 값 대신 해결책 간의 상대적인 우열 관계를 평가하는 방식을 소개했습니다.

- **Technical Details**: 연구에서는 다양한 기계 학습 알고리즘(정규화 회귀(regularized regression), 신경망(neural networks), 결정 트리(decision trees), 부스팅 방법(boosting methods), 랜덤 포레스트(random forests))과 서브로게이트 전략(다양성을 권장하거나 예측 임계치를 완화하는 방법)하에서 서브로게이트 모델을 다차원적으로 분석하였습니다. DE(차별 진화, Differential Evolution) 알고리즘과 함께 활용하여 서브로게이트 모델이 최적화 과정에서 어떻게 적용되는지를 조사하였습니다.

- **Performance Highlights**: 서브로게이트 모델을 활용한 실험은 SOCO2011 대회 및 GECCO2021 산업 챌린지의 벤치마크 문제를 포함하여 진행되었으며, 온라인 머신러닝 기반 서브로게이트 모델의 성능은 예측 모델의 정확성뿐만 아니라 긍정적 또는 부정적 사례에 대한 편향의 종류, 그리고 최적화가 그 예측을 사용하는 방식에 따라 달라진다는 결론에 도달했습니다.



### EBES: Easy Benchmarking for Event Sequences (https://arxiv.org/abs/2410.03399)
- **What's New**: 이 논문에서는 이벤트 시퀀스(event sequences)에 대한 표준화된 벤치마크인 EBES를 도입합니다. EBES는 데이터셋, 모델, 실험 프로토콜에 대한 통합된 인터페이스를 제공하여 향후 연구를 촉진합니다.

- **Technical Details**: EBES는 회귀(regression)와 분류(classification) 문제를 중심으로 하며, 특정한 평가 시나리오를 포함합니다. 제안된 라이브러리는 새로운 합성 데이터셋과 일반적으로 사용되는 실제 데이터셋을 제공합니다.

- **Performance Highlights**: 이 분석을 통해 향후 연구를 위한 권장 사항들을 제시하고, 데이터셋 사용과 모델 평가와 관련된 잠재적인 위험 요소를 강조합니다.



### GraphCroc: Cross-Correlation Autoencoder for Graph Structural Reconstruction (https://arxiv.org/abs/2410.03396)
Comments:
          22 pages, 16 figures. Accepted in NeurIPS 2024

- **What's New**: 이 논문은 Graph Autoencoders (GAE) 모델의 한계를 극복하고 다중 그래프 상황에 적합한 새로운 Cross-Correlation 메커니즘을 도입하여 그래프 구조 재구성을 개선하는 GraphCroc 모델을 제안합니다.

- **Technical Details**: 종전의 GAE 모델들은 self-correlation을 사용하여 그래프 구조를 표현해왔으나, 이 연구에서는 Cross-Correlation을 활용하여 보다 정확한 노드 임베딩을 제공합니다. GraphCroc 모델은 U-Net과 유사한 인코딩-디코딩 절차를 사용하며, 비대칭 연결 문제를 해결하기 위해 손실 균형 전략을 적용합니다.

- **Performance Highlights**: 이론적 분석과 수치 평가 모두에서 GraphCroc은 기존 self-correlation 기반 GAE 모델 대비 그래프 구조 재구성에서 유의미한 성과를 내어, 특히 다중 그래프에서 우수한 성능을 발휘함이 입증되었습니다.



### Predicting perturbation targets with causal differential networks (https://arxiv.org/abs/2410.03380)
- **What's New**: 이 연구에서는 생물학적 데이터에서 개입 목표를 예측하기 위한 새로운 접근법인 Causal Differential Networks (Cdn)를 제안합니다. 이 방법은 관찰적 데이터와 개입 데이터를 별도로 처리하여 인과 그래프를 추론한 후, 이를 통해 노출된 변수들을 찾아냅니다.

- **Technical Details**: Cdn은 관찰적 데이터와 개입 데이터에서 각각 인과 그래프를 추론하기 위해 사전 훈련된 causal discovery 모듈을 활용하고, 그런 다음 큰 축 주의 기반 분류기를 사용하여 정답 개입 목표에 대해 지도 학습(un supervised learning) 방식으로 학습합니다.

- **Performance Highlights**: Cdn은 기존 방법들과 비교하여 일관되게 더 나은 성능을 보여주며, 특히 7개의 단일 세포 전사체 데이터 세트에서 perturbation modeling에서의 성능을 입증했습니다. 또한 다양한 합성 데이터 세트에서 개입 목표 예측에 있어서 6개의 기존 인과 발견 알고리즘보다 우수한 성능을 기록하였습니다.



### Mitigating Adversarial Perturbations for Deep Reinforcement Learning via Vector Quantization (https://arxiv.org/abs/2410.03376)
Comments:
          8 pages, IROS 2024 (Code: this https URL)

- **What's New**: 이 논문은 강화 학습 (Reinforcement Learning, RL) 에이전트의 적대적 공격에 대한 방어를 개선하기 위해 벡터 양자화 (Vector Quantization, VQ) 변환 기법을 제안합니다. 이는 기존의 훈련 기반 방법 대신 입력 변환 방어 기법을 통해 시행되며, RL 에이전트의 입력에 대한 공격의 범위를 줄이는 데 도움을 줍니다.

- **Technical Details**: 논문에서 제안하는 VQ 기반 방어 기법은 RL 에이전트의 입력 관측값을 변환하여 적대적 관측값의 영향을 최소화합니다. 이 방법은 컴퓨팅 효율이 높으며, 기존의 RL 알고리즘과 잘 통합될 수 있습니다. VQ를 활용하여 관측 공간을 이산화(discretization)하고, 그 변환된 공간 내에서 RL 에이전트를 훈련시키는 방식으로 작동합니다.

- **Performance Highlights**: 다양한 환경에서의 실험을 통해, 제안된 VQ 입력 변환은 공격에 대한 방어력을 높이는 데 효과적임을 입증하였습니다. 이 연구는 RL 에이전트가 적대적 공격에 대해 더 강건하도록 만들고, 기존의 강건화를 위한 훈련 방법과 상호 보완적으로 작용할 수 있는 가능성을 보여줍니다.



### SoundSignature: What Type of Music Do You Like? (https://arxiv.org/abs/2410.03375)
Comments:
          10 pages, 1 figure, to be published in the 2024 International Symposium on the IEEE Internet of Sounds Proceedings

- **What's New**: SoundSignature는 사용자들이 좋아하는 음악을 분석하기 위해 OpenAI Assistant와 통합된 음악 애플리케이션입니다. 이 시스템은 최신 Music Information Retrieval (MIR) Python 패키지를 활용하여 추출된 음향/음악적 특성과 아티스트 및 밴드에 대한 어시스턴트의 광범위한 지식을 결합합니다.

- **Technical Details**: 음악 애플리케이션은 Semantic Audio와 Emerging Internet of Sounds (IoS) 원칙을 활용하여 사용자의 음악에 대한 개인화된 통찰력을 제공합니다. CREMA(Chord Recognition Algorithm), DEMUCS(Source Separation Algorithm), basic-pitch(Audio-to-MIDI Converter) 등의 오픈 소스 음악 도구도 통합되어 있습니다.

- **Performance Highlights**: 이 애플리케이션은 사용자들이 음악의 음향 특성에 대한 이해를 넓힐 수 있도록 학습과 상호작용을 촉진하며, 파일럿 사용자 연구의 결과를 통해 효과성과 사용성을 평가한 결과를 제시합니다.



### Make Interval Bound Propagation great again (https://arxiv.org/abs/2410.03373)
- **What's New**: 본 논문은 Neural Network Certification(NNC) 분야에서의 기술적 진전을 다루고 있으며, Interval Bound Propagation(IBP) 방식의 최적성이 떨어짐을 보여줍니다. 또한, 두 가지 새로운 방법(Doubleton Arithmetic와 Affine Arithmetic)을 신경망의 안정성 보증을 위해 적용한 것이 주요한 새롭게 제안된 내용입니다.

- **Technical Details**: 네트워크의 안전성을 평가하기 위한 기존의 IBP 방법은 wrapping 효과(wrapping effect)의 영향을 받아 비효율적임을 설명합니다. 본 연구에서는 이를 극복하기 위하여 Doubleton Arithmetic(DA)와 Affine Arithmetic(AA)를 도입했습니다. 이 두 방법은 비선형 활성함수인 ReLU(ReLU)와 softmax를 사용하는 신경망의 층을 정확하게 평가할 수 있는 알고리즘을 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, Affine Arithmetic는 IBP 대비 월등히 좋은 성능을 보였으며, 신경망의 출력에 대한 경계를 제공하는 데 있어 가장 우수한 성능을 보였습니다. 특히, AA는 대규모 네트워크에서 DA보다 수십 배 더 빠른 계산 속도를 자랑합니다.



### An Enhanced Harmonic Densely Connected Hybrid Transformer Network Architecture for Chronic Wound Segmentation Utilising Multi-Colour Space Tensor Merging (https://arxiv.org/abs/2410.03359)
- **What's New**: 이 논문은 만성 상처(segmentation) 처리에 대한 새로운 접근법을 제시하며, 초기 레이어에서 대비를 제거하는 구성요소를 통합한 HarDNet 아키텍처를 개선했습니다. 이 연구는 특히 어두운 피부색을 가진 사례를 대상으로 한 최초의 만성 상처 세분화 연구입니다.

- **Technical Details**: 논문에서는 HarDNet(segmentation architecture) 모델을 사용하여 다양한 색상 공간(tensor merging process)에서 특성을 최적화하고, 합성곱 블록의 조화를 조정하여 성능을 향상시킵니다. 훈련 데이터는 밝은 피부를 가진 환자들의 상처 이미지를 사용하고, 어두운 피부색이 있는 두 개의 테스트 세트에서 모델을 평가합니다.

- **Performance Highlights**: 어두운 피부톤 세트에서 Dice 유사도 계수(Dice similarity coefficient)가 +0.1221, 교차 비율(intersection over union)이 +0.1274 향상되었으며, 임상 전문가들로부터 받은 주관 평가에서도 제안된 모델이 기존 모델보다 3% 이상의 개선을 보였습니다.



### LANTERN: Accelerating Visual Autoregressive Models with Relaxed Speculative Decoding (https://arxiv.org/abs/2410.03355)
- **What's New**: 최근 Auto-Regressive (AR) 모델이 이미지 생성에 있어서 중요성을 얻고 있으며, diffusion 모델을 능가하는 성능을 보였습니다. 그러나 AR 모델의 시퀀스 처리 특성으로 인해 생성 속도가 저하되는 문제가 있습니다. 이 논문에서는 이러한 문제를 극복하기 위해 LANTERN 이라는 새로운 방법론을 제안합니다.

- **Technical Details**: 논문에서는 'token selection ambiguity'라는 문제를 다루고 있으며, 이는 시각적 AR 모델에서 토큰의 낮은 확률 분포로 인해 speculative decoding의 성능이 저하됨을 설명합니다. LANTERN은 이러한 수용 조건을 완화하고, 잠재 공간에서의 토큰 간의 상호 교환성을 이용하여 효과적으로 후보 토큰을 활용할 수 있도록 합니다. 이를 통해 이미지 품질과 의미론적 일관성을 크게 손상하지 않으면서 생성 속도를 향상시키는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과, LANTERN을 사용한 경우 LlamaGen 모델을 기준으로 greedy decoding에서 1.75배, random sampling에서 1.76배의 속도 향상이 나타났습니다. 이는 기존의 speculative decoding 방법보다 큰 성과입니다.



### An X-Ray Is Worth 15 Features: Sparse Autoencoders for Interpretable Radiology Report Generation (https://arxiv.org/abs/2410.03334)
- **What's New**: 이 논문은 기존의 비전-언어 모델에서 발생하는 hallucination 문제를 해결하기 위해 sparse autoencoders(SAEs)를 활용한 새로운 접근법인 SAE-Rad를 소개합니다. 이 모델은 기계적 해석 가능성(mechanistic interpretability) 기법을 적용하여, 방사선 이미지 인코더의 잠재 표현을 인간이 해석할 수 있는 특징으로 분해하는 방식입니다.

- **Technical Details**: SAE-Rad는 sparse autoencoders를 기반으로 한 하이브리드 아키텍처로, 상태-of-the-art 기술과 비교하여 유사한 밀도로 정확한 재구성을 달성합니다. 이 모델은 사전 훈련된 언어 모델을 사용하여 각 SAE 특징에 대한 실제 보고서를 방사선묘사로 증류(distil)한 후, 전체 보고서를 제작합니다. 이 방법은 대규모 모델의 세부 조정을 필요로 하지 않습니다.

- **Performance Highlights**: MIMIC-CXR 데이터셋에서 SAE-Rad는 경쟁력 있는 방사선 특화 지표를 달성하였으며, 훈련을 위한 계산 자원이 현저히 적게 소비되었습니다. SAE-Rad의 질적 분석 결과는 이 모델이 유의미한 시각 개념을 학습하고 전문가 해석과 유사한 보고서를 생성한다는 것을 보여줍니다. SAEs는 의료 분야에서 다중모드 추론을 향상시킬 수 있는 해석 가능한 대안을 제공합니다.



### Comparative Analysis and Ensemble Enhancement of Leading CNN Architectures for Breast Cancer Classification (https://arxiv.org/abs/2410.03333)
- **What's New**: 본 연구는 유방암의 진단을 위한 새로운 히스토 병리 이미지 분류 접근법을 소개합니다. 다양한 Convolutional Neural Network (CNN) 모델을 비교하고 최적의 하이퍼파라미터를 식별하여 분류 효율성을 기반으로 순위를 매깁니다.

- **Technical Details**: CNN 모델들에 대한 포괄적인 비교를 통해 초기 이미지를 시퀀스화하여 일관된 데이터 조건을 보장하고 훈련 기간을 단축시키는 원래의 개념들이 포함되어 있습니다. 연구에서는 2000개 이상의 훈련 조합을 탐구하였으며, BreakHis x40과 x200 데이터셋에서 각각 99.75%, Bach 데이터셋에서 95.18%의 높은 정확도를 기록했습니다.

- **Performance Highlights**: 본 연구의 앙상블 아키텍처는 세 개의 고성능 CNN 모델을 조합하여 분류 정확도를 높이는 데 성공하였으며, Bach Online 블라인드 챌린지에서는 89%의 성과를 나타냈습니다. 이 방법론은 유방암 히스토 병리 이미지 데이터셋 뿐만 아니라 다른 의학 이미지 데이터셋에도 적용 가능합니다.



### Influence-oriented Personalized Federated Learning (https://arxiv.org/abs/2410.03315)
- **What's New**: 본 연구는 기존의 federated learning (FL) 방법의 한계를 극복하기 위해 클라이언트 수준(client-level) 및 클래스 수준(class-level) 영향력(influence)를 양적으로 측정하여 각 클라이언트의 파라미터 집합을 적응적으로 조정할 수 있는 새로운 프레임워크인 FedC^2I를 제안합니다. 이 접근법은 FL 시스템 내의 클라이언트 간 상호 영향을 명확하게 모델링합니다.

- **Technical Details**: FedC^2I 프레임워크는 영향 벡터(influence vector)와 영향 행렬(influence matrix)을 사용하여 클라이언트 및 클래스 간의 영향을 모델링합니다. 영향 벡터는 각 클라이언트의 영향력을 정량화하여 다른 클라이언트로부터 지식을 선택적으로 획득하도록 돕고, 영향 행렬은 보다 세밀한 방식으로 클래스 수준의 영향을 캡처하여 개별화된 분류기 집합을 이루도록 합니다.

- **Performance Highlights**: 비독립적-동일 분포(non-IID) 환경에서 FedC^2I의 성능을 기존의 FL 방법들과 비교하여 우수성을 증명하였으며, 이 프레임워크의 주요 구성 요소의 효과적 기여를 검증하였습니다.



### Comparing zero-shot self-explanations with human rationales in multilingual text classification (https://arxiv.org/abs/2410.03296)
Comments:
          preprint

- **What's New**: 이 논문은 Instruction-tuned LLM들이 생성하는 self-explanations의 품질을 평가하여 신뢰성과 투명성을 높이는 방법을 탐구하고 있습니다.

- **Technical Details**: 연구에서는 sentiment classification과 forced labour detection 두 가지 텍스트 분류 작업을 수행하고, 자기 생성 설명과 인간 주석을 비교하여 그 신뢰성 및 일관성을 평가하고 있습니다. 또한, layer-wise relevance propagation (LRP)과 같은 post-hoc feature attribution 기법을 적용하여 결과를 비교합니다.

- **Performance Highlights**: 결과적으로 self-explanations가 인간 주석과 더 밀접하게 일치하며, LRP보다 감정 표현에 대한 신뢰성 있는 설명 품질을 유지하는 것으로 나타났습니다.



### Five Years of COVID-19 Discourse on Instagram: A Labeled Instagram Dataset of Over Half a Million Posts for Multilingual Sentiment Analysis (https://arxiv.org/abs/2410.03293)
- **What's New**: 이 논문은 COVID-19와 관련된 인스타그램 포스트의 데이터셋과 다국적 감성 분석의 결과를 제시하면서 2020년부터 2024년까지의 감정 경향을 탐구합니다.

- **Technical Details**: 500,153개의 인스타그램 포스트로 구성된 다국어 데이터셋을 구축하였으며, 이는 161개의 언어와 535,021개의 해시태그를 포함합니다. 이 데이터셋은 감정 분석을 위한 속성을 포함하고 있으며, 매글로벌 언어마다 양의(positive), 음의(negative), 중립(neutral)으로 분류된 결과를 갖추고 있습니다.

- **Performance Highlights**: 연도별 감성 분석 결과, 긍정 감정 비율이 38.35%에서 28.69%로 감소한 반면, 중립 감정 비율은 44.19%에서 58.34%로 증가했습니다. 언어별 분석에서는 영어 포스트의 긍정 감정 비율이 49.68%인 반면, 힌디어 포스트는 4.40%에 불과해 언어에 따른 감정 분포에 뚜렷한 차이가 나타났습니다.



### Enhanced Transformer architecture for in-context learning of dynamical systems (https://arxiv.org/abs/2410.03291)
- **What's New**: 이 논문은 in-context identification 패러다임을 통해 다이나믹 시스템의 메타 모델을 오프라인에서 합성 데이터 기반으로 추정하는 방법을 제안합니다. 새로운 메타 모델링 프레임워크를 통해 확률론적 접근 방식과 비연속적인 컨텍스트 및 쿼리 윈도우를 관리하는 방법을 도입하고 있으며, RNN을 이용한 반복 패칭을 통해 긴 컨텍스트 시퀀스를 효과적으로 처리할 수 있도록 개선되었습니다.

- **Technical Details**: 메타 모델은 확률적 설정에서 학습 문제를 형성하며, 추정된 쿼리 출력의 조건부 분포를 사용하여 예측 불확실성에 대한 정보를 제공합니다. 비연속적인 컨텍스트와 쿼리 윈도우를 허용하여 초기 조건 정보도 메타 모델에 제공합니다. 이 메타 모델은 기본적으로 encoder-decoder Transformer 아키텍처로 구성되어 있으며, 여기서 RNN을 패칭 네트워크로 사용하여 메모리 제한을 해결합니다.

- **Performance Highlights**: 본 논문의 변경 사항은 Wiener-Hammerstein 시스템 클래스를 대상으로 한 수치 예제를 통해 모델의 성능 및 확장성을 향상시킨 것으로 입증되었습니다. 이러한 혁신적인 접근 방식은 메타 모델링 프레임워크의 기존 한계들을 극복하고 SYSID 문제에 적합한 구성을 제공합니다.



### Grounded-VideoLLM: Sharpening Fine-grained Temporal Grounding in Video Large Language Models (https://arxiv.org/abs/2410.03290)
- **What's New**: 본 논문에서는 특정 비디오 순간을 정밀하게 인식하고 추론할 수 있는 새로운 Video-LLM인 Grounded-VideoLLM을 제안합니다. 이 모델은 기존의 Video-LLM들이 가지는 세밀한 시간 기반 이해의 한계를 극복하기 위해 추가적인 시간 스트림과 특정 시간 지식이 포함된 이산적 시간 토큰을 도입하였습니다.

- **Technical Details**: Grounded-VideoLLM은 (1) 두 개의 스트림으로 구성된 인코딩 방식인 Two-Stream Encoding을 사용하여 각 비디오 단편의 공간적 및 시간적 요소를 분리하여 모델링합니다. (2) 시간적 토큰을 도입하여 시간 스탬프를 효율적으로 표현할 수 있도록 하고, 이 토큰들은 LLM의 임베딩 공간과 통합되어 직관적인 예측을 가능하게 합니다.

- **Performance Highlights**: 다양한 실험을 통해 Grounded-VideoLLM은 시간 문장 정량화, 밀집 비디오 캡션, Grounded VideoQA 등에서 높은 성능을 보이며 기존의 Video-LLM들보다 우수한 결과를 나타냈습니다. 이는 Grounded-VideoLLM이 일반 비디오 이해를 위한 다재다능한 비디오 어시스턴트로서 잠재력을 지니고 있음을 보여줍니다.



### Manikin-Recorded Cardiopulmonary Sounds Dataset Using Digital Stethoscop (https://arxiv.org/abs/2410.03280)
- **What's New**: 이번 논문은 심장 및 폐 소리를 모니터링하기 위한 첫 번째 데이터셋을 소개합니다. 개별 및 혼합 심폐(cardiorespiratory) 소리를 포함하는 데이터셋으로, 향상된 정밀도가 특징입니다.

- **Technical Details**: 디지털 청진기(digital stethoscope)를 사용하여 임상 인형(clinical manikin)에서 수집한 데이터로, 다양한 인체 위치에서의 청진 소리를 포함합니다. 이 데이터셋은 정상 소리와 여러 비정상 음향(murmur, atrial fibrillation, tachycardia 등)도 포함합니다.

- **Performance Highlights**: 인공 지능(artificial intelligence) 애플리케이션에 적합한 이 데이터셋은 자동화된 심폐질환 감지, 소리 분류, 비지도 분리 기술(unsupervised separation techniques), 딥러닝(deep learning) 알고리즘에 유용합니다.



### Test-time Adaptation for Regression by Subspace Alignmen (https://arxiv.org/abs/2410.03263)
- **What's New**: 이 논문은 회귀(regression) 모델의 테스트 시간 적응(Test-Time Adaptation, TTA)에 대해 조사합니다. 기존의 TTA 방법이 분류(classification)를 위한 설계에 국한되어 있다는 문제를 다룹니다. 기존 방법들은 클래스-카테고리(class-categorical) 예측을 전제로 하였으나, 회귀 모델은 단일 스칼라 값만을 출력합니다. 이를 해결하기 위해, 소스(source)와 타겟(target) 도메인 간의 특징 분포(feature distributions)를 정렬하는 접근법을 제안합니다.

- **Technical Details**: 제안된 방법은 Significant-subspace Alignment (SSA)로, 두 개의 구성 요소인 서브스페이스 탐지(subspace detection)와 차원 가중화(dimension weighting)로 이루어져 있습니다. 서브스페이스 탐지에서는 PCA(주성분 분석)를 사용하여 출력에 유의미한 특징 벡터가 집중된 서브스페이스(subspace)를 찾습니다. 차원 가중화는 출력에 더 큰 영향을 미치는 서브스페이스 차원의 중요성을 높입니다.

- **Performance Highlights**: 다양한 회귀 작업인 UTKFace, Biwi Kinect, California Housing에 대한 실험 결과, SSA가 기존 분류 기법에 맞춰 설계된 TTA 기반선 모델들보다 뛰어난 성능을 보임을 입증했습니다.



### How much can we forget about Data Contamination? (https://arxiv.org/abs/2410.03249)
- **What's New**: 이번 연구는 최신의 대형 언어 모델(LLMs)의 평가 시 벤치마크 데이터의 누출로 인한 문제를 다루고 있습니다. 연구자들은 벤치마크 데이터의 경미한 오염이 평가에 미치는 영향에 대해 실험적 증거와 이론적 추정을 제시하여 그러한 오염이 항상 부정적인 결과를 초래하지 않음을 밝혔다.

- **Technical Details**: 연구에서는 세 가지 차원에서 벤치마크 과적합(benchmark overfitting)의 크기를 정량화했습니다: 모델의 파라미터 수(1.6B까지), 예시가 본 데이터에서 보는 횟수(144회까지), 훈련 토큰 수(40B까지). Chinchilla 스케일링 법칙을 따랐을 경우, 오염이 적더라도 과적합을 초래할 수 있음을 보였습니다. 그러나 훈련 데이터를 다섯 배 확장할 경우, 과거의 데이터를 잊을 수 있는 수치적 경계를 제시합니다.

- **Performance Highlights**: 이 연구는 많은 대형 언어 모델들이 훈련 초기의 데이터를 잊고 있다는 것을 확인했습니다. 또한, 모델들이 데이터의 반복 노출에 대해 가장 강한 과적합을 보이는 현상을 통해, 새로운 데이터에 대한 노출이 중요하다는 것을 강조했습니다.



### Latent Action Priors From a Single Gait Cycle Demonstration for Online Imitation Learning (https://arxiv.org/abs/2410.03246)
Comments:
          Submitted to ICRA 2025

- **What's New**: 이 논문에서는 Deep Reinforcement Learning (DRL)에서 로봇 학습을 위한 새로운 유도 편향(inductive bias)인 잠재 행동(latent action)을 제안합니다. 이 잠재 행동은 전문가의 데모에서 학습되며, 단일 오픈루프(가동 중인 시스템의 예측과 실제 작동 중인 상태) 보행 주기를 통해 습득할 수 있습니다.

- **Technical Details**: 잠재 행동은 간단한 오토인코더(autoencoder)를 사용하여 학습되며, 이 과정을 통해 얻어진 정보는 DRL 알고리즘의 학습을 돕습니다. 이 방법은 행동 공간을 저차원의 다양체(low-dimensional manifold)로 압축하여, 드물게 관찰되는 전문 데이터의 범위를 넘어서자, 에이전트가 전문가 수준 이상의 성과를 달성하게 합니다.

- **Performance Highlights**: 잠재 행동을 활용한 DRL 모델은 기존의 전문가 데이터에 비해 학습 속도가 빠르고 높은 보상을 제공합니다. 더불어 전이 학습(transfer tasks)의 수행 성능을 현저히 향상시키며, 목표 속도가 높을 경우 보행 전환(gait transitions)도 가능함을 보여줍니다.



### AutoPenBench: Benchmarking Generative Agents for Penetration Testing (https://arxiv.org/abs/2410.03225)
Comments:
          Codes for the benchmark: this https URL Codes for the paper experiments: this https URL

- **What's New**: 이 논문에서는 자동 침투 테스트를 위한 새로운 벤치마크인 AutoPenBench를 소개합니다. 이 벤치마크는 생성형 AI 에이전트의 성능을 평가하기 위한 체계적이고 표준화된 프레임워크를 제공합니다.

- **Technical Details**: AutoPenBench는 다양한 난이도의 33개 작업을 포함하며, 각 작업은 취약한 시스템을 나타냅니다. 작업은 비강제(in-vitro)와 실제(real-world) 시나리오로 구분됩니다. 에이전트 성능 평가는 일반적 및 특정 이정표(milestones)를 기반으로 하여 수행됩니다.

- **Performance Highlights**: 완전 자율 에이전트는 21%의 성공률(Success Rate)에 그쳤으나, 인간과 협력하는 반 자율 에이전트는 64%로 상당한 개선을 보였습니다. AutoPenBench의 구조를 통해 다양한 LLM의 영향을 평가할 수 있습니다.



### ScriptViz: A Visualization Tool to Aid Scriptwriting based on a Large Movie Databas (https://arxiv.org/abs/2410.03224)
Comments:
          Accepted in the 37th Annual ACM Symposium on User Interface Software and Technology (UIST'24). Webpage: this https URL

- **What's New**: 이 논문에서는 ScriptViz라는 도구를 소개하여, 스크립트 작가들이 영화 데이터베이스에서 외부 비주얼을 제공받아 스크립트를 작성하는 과정에서 도움이 되도록 합니다. ScriptViz는 스크립트의 텍스트와 대화를 기반으로 적절한 시각 자료를 실시간으로 검색하여 제공합니다.

- **Technical Details**: ScriptViz는 두 가지 유형의 비주얼 요소 제어를 통해 작가들이 고정된 비주얼 요소를 명확하게 볼 수 있도록 하며, 불확실한 요소의 변화를 살펴볼 수 있게 합니다. 이 도구는 스크립트의 부분적 내용과 고정 및 변화 가능한 비주얼 속성을 지정하여 기존 영화와 일치하는 장면을 찾고, 각 장면에 대한 키프레임을 검색하여 제공합니다.

- **Performance Highlights**: 15명의 스크립트 작가들을 대상으로 한 사용자 평가에서, ScriptViz는 스크립트와 일치하는 비주얼 가능성을 일관되면서도 다양하게 제시하여 창작에 도움이 되는 것으로 나타났습니다. 이 도구는 기획 과정에서 스크립트 작가들에게 효과적인 지원을 제공하며, 지정된 비주얼 속성을 활용하여 더 나은 브레인스토밍과 기획을 가능하게 합니다.



### A Tutorial on the Design, Experimentation and Application of Metaheuristic Algorithms to Real-World Optimization Problems (https://arxiv.org/abs/2410.03205)
- **What's New**: 이 연구는 메타휴리스틱(metaheuristic) 알고리즘을 활용한 최적화 연구에서의 과학적 엄격성과 투명성을 제공하기 위한 좋은 관습을 제안합니다. 또한, 문제 모델링에서 실질적인 알고리즘 배포 및 운영까지의 연구 단계를 단계별로 안내하는 방법론을 소개합니다.

- **Technical Details**: 본 연구에서는 메타휴리스틱 방법들을 이용한 최적화 문제의 모델링, 알고리즘 설계 및 구현, 성능 평가, 재현성 검증, 실제 응용을 위한 배포 등 여러 단계를 구조화하여 방법론을 제안합니다. 각 단계에서는 문제의 구체화, 해결책 부호화(solution encoding), 탐색 연산자(operators)의 구현, 실험 설계 및 성과 평가 지표와 같은 중요한 요소들을 다루게 됩니다.

- **Performance Highlights**: 제안된 방법론은 메타휴리스틱 알고리즘의 재현성 및 실질적인 응용 측면에서 효율성을 높이는 데 기여하며,의도된 응용 환경에서 새로운 최적화 메타휴리스틱의 성공적인 배포 및 운영을 위한 중요한 고려 사항과 연구 방향도 제시합니다.



### MultiVerse: Efficient and Expressive Zero-Shot Multi-Task Text-to-Speech (https://arxiv.org/abs/2410.03192)
Comments:
          Accepted to EMNLP 2024 Findings

- **What's New**: 최근 제안된 MultiVerse는 전통적인 TTS 시스템보다 훨씬 적은 훈련 데이터를 요구하면서도 zero-shot 및 cross-lingual 조건에서 TTS 및 음성 스타일 전이 기능을 수행할 수 있는 다중 작업(text-to-speech) TTS 시스템입니다.

- **Technical Details**: MultiVerse는 source-filter theory를 기반으로 한 해체(disentanglement) 방식을 사용하여 음성을 필터 관련 및 소스 관련 표현으로 분해합니다. 또한, 프로조디(prosody) 모델링을 개선하기 위해 autoregressive(AR) 및 non-autoregressive 방법을 결합하여 프로조디를 모델링합니다. 이 시스템은 각 세부 요소를 명확하게 구분하여 학습할 수 있습니다.

- **Performance Highlights**: MultiVerse는 기존의 데이터 기반 TTS 시스템과 유사한 zero-shot TTS 성능을 보여주며, 동일한 소규모 데이터로 훈련된 타 제로샷 TTS 시스템들보다도 현저히 우수한 성능을 발휘합니다. 특히, 새로운 프로조디 모델링 기법이 강조되어, 주어진 프롬프트에 고유한 프로조디 유사성을 생성하는 능력이 크게 향상되었습니다.



### Looking into Concept Explanation Methods for Diabetic Retinopathy Classification (https://arxiv.org/abs/2410.03188)
Comments:
          Accepted for publication at the Journal of Machine Learning for Biomedical Imaging (MELBA) this https URL

- **What's New**: 이 논문에서는 당뇨병성 망막병증(diabetic retinopathy, DR) 자동 진단을 위한 심층 신경망의 해석 가능성을 높이기 위해, 개념 기반 설명 방법 두 가지를 비교하고 평가했습니다. 특히, Concept Activation Vectors(TCAV)와 Concept Bottleneck Models(CBM) 방법을 조사하였으며, 두 방법은 각각 장점과 단점을 가지고 있습니다.

- **Technical Details**: 본 연구에서는 TCAV와 CBM 두 가지 개념 기반 설명 기술을 사용하여 DR 진단 인공지능 모델의 해석 가능성을 강조합니다. TCAV는 концепт의 방향으로 이미지 변경에 모델의 민감도를 측정하여 개념의 상대 중요성을 평가하며, CBM은 중간 개념 예측을 직접 수정하고 최종 예측이 어떻게 영향을 받는지 관찰하는 방법입니다.

- **Performance Highlights**: TCAV와 CBM 모두 심층 신경망의 예측을 설명하는 데 유용하지만, 선택한 방법은 사용 가능한 데이터와 최종 사용자의 선호도에 따라 달라져야 합니다. 정량적 분석 결과, 두 방법의 설명 능력과 해석의 직관성이 비교되었습니다.



### EXAQ: Exponent Aware Quantization For LLMs Acceleration (https://arxiv.org/abs/2410.03185)
- **What's New**: 본 연구는 LLMs inference에서 주로 소프트맥스 레이어가 성능 저하의 병목 지점이 됨을 발견하였습니다. 소프트맥스 연산의 초기 단계인 지수 계산과 누적을 최적화하는 데 중점을 두고 있습니다.

- **Technical Details**: 소프트맥스 함수의 입력에 대한 최적의 클리핑 값(clipping value)을 결정하는 분석적 접근 방식을 제안하였습니다. 이를 통해 LLMs inference를 위한 4비트 이하의 양자화가 가능합니다.

- **Performance Highlights**: LLaMA1-30B 모델에 대한 검증 결과, 'Physical Interaction: Question Answering' (PIQA) 데이터셋에서 2비트 양자화로 기준 성능을 달성하였고, 누적 단계에서 약 4배의 속도 향상을 이루었습니다. 소프트맥스 연산에서 e^x와 ∑(e^x) 계산을 가속화하여 36.9%의 성능 향상을 달성했습니다.



### Investigating and Mitigating Object Hallucinations in Pretrained Vision-Language (CLIP) Models (https://arxiv.org/abs/2410.03176)
Comments:
          EMNLP 2024

- **What's New**: 이번 논문에서는 CLIP 모델에서 발생하는 객체 환각(object hallucination) 문제를 집중적으로 조사하고, 새로운 벤치마크인 OHD-Caps를 제안합니다. 이 모델은 시각-언어 시스템의 핵심 역할을 하며, 모델 내부의 환각이 어떤 부분에서 발생하는지를 분석하였습니다.

- **Technical Details**: LVLMs (Large Vision-Language Models)는 이미지 캡셔닝, 시각적 질문 응답과 같은 다양한 작업에서 우수한 성능을 보입니다. 본 논문에서는 CLIP 모델을 기반으로 하여, 'fine-grained object-level contrastive loss' 방법을 통해 CLIP 모델의 객체 환각을 완화하는 방법을 제안합니다. OHD-Caps 벤치마크를 사용하여 EVL 프레임워크의 성과를 평가하였습니다.

- **Performance Highlights**: CLIP 모델의 객체 환각 문제를 해결하기 위한 제안 방법은 기존 상태인 ‘CLIP ViT-B/32’에서 14.3%에서 82.5%로 향상되었습니다. 또한, LLaVA-1.5 모델에서도 환각 문제가 80.2%에서 83.2%로 개선되었습니다.



### Autoregressive Moving-average Attention Mechanism for Time Series Forecasting (https://arxiv.org/abs/2410.03159)
- **What's New**: 본 논문에서는 다양한 선형 attention 메커니즘에 적응할 수 있는 Autoregressive (AR) Moving-average (MA) attention 구조를 제안하여 시계열(time series)에서 장기 및 지역적 패턴을 포착하는 능력을 향상시킵니다.

- **Technical Details**: ARMA 구조를 기존의 autoregressive attention 메커니즘에 통합한 ARMA attention 메커니즘을 제안합니다. 이 방법은 계산 비용을 크게 증가시키지 않으면서 성능을 개선하며, MA 출력을 직접 계산하지 않고 간접적인 MA 가중치 생성을 통해 구현됩니다.

- **Performance Highlights**: ARMA 구조를 도입함으로써 다양한 AR attention의 시계열 예측 성능이 일관되게 향상되었습니다. 최첨단 성능을 기록하며, 토큰화(tokenization) 및 훈련 방법이 적절할 경우, 기본 AR Transformer만으로도 기존의 최첨단 결과에 필적하는 성과를 달성했습니다.



### Mathematical Formalism for Memory Compression in Selective State Space Models (https://arxiv.org/abs/2410.03158)
Comments:
          27 Pages

- **What's New**: 이 논문에서는 선택적 상태 공간 모델(selective state space models, SSMs)을 통해 시퀀스 모델링의 메모리 압축 기능을 rigorously(엄격하게) 분석합니다. 이 새로운 접근 방식은 전통적인 RNN과 CNN의 한계를 극복하고, 제어 이론(control theory) 및 동적 시스템(dynamical systems)의 원리를 활용하여 메모리를 효율적으로 관리합니다.

- **Technical Details**: 선택적 SSM은 입력의 관련성에 따라 동적으로 상태를 필터링하고 업데이트하는 selective gating mechanism을 도입하여 긴 시퀀스를 효과적으로 압축합니다. 이 논문에서는 rate-distortion theory 및 information bottleneck method를 활용하여 메모리 효율성과 정보 보존 간의 trade-off를 정량화합니다. Fano의 불평등(Fano's inequality)과 데이터 처리 불평등(data processing inequality)을 사용하여 메모리 압축에 대한 이론적 한계를 제공합니다.

- **Performance Highlights**: 실험적으로, 선택적 SSM은 시계열 예측(time-series forecasting)과 자연어 처리(natural language processing)와 같은 시퀀스 모델링 작업에서 state-of-the-art 성능을 달성하며, 기존 RNN 기반 모델에 비해 메모리와 계산 자원 사용을 줄이고도 향상된 속도 및 효율성을 보여주었습니다.



### MELODI: Exploring Memory Compression for Long Contexts (https://arxiv.org/abs/2410.03156)
- **What's New**: 이번 논문에서는 MELODI라고 불리는 새로운 메모리 아키텍처를 제안합니다. 이 아키텍처는 짧은 문맥(window)을 사용하여 효율적으로 긴 문서를 처리하기 위한 것입니다.

- **Technical Details**: MELODI는 네트워크 레이어와 문맥 윈도우를 통해 계층적 압축 구조를 사용하여 단기 메모리와 장기 메모리를 표현합니다. 단기 메모리는 여러 레이어에 걸쳐 문맥 윈도우를 순환적으로 압축하여 부드러운 전환을 보장합니다. 반면, 장기 메모리는 단일 중간 레이어 내에서 추가 압축을 수행하고 여러 문맥 윈도우에서 정보를 집계하여 정보를 통합합니다.

- **Performance Highlights**: MELODI는 Memorizing Transformer와 비교하여 8배의 메모리 사용량을 줄이면서도 다양한 긴 문맥 데이터셋에서 우수한 성능을 보여줍니다. 예를 들어, 13 레이어 트랜스포머 네트워크를 사용하여 PG-19와 arXiv Math에서 각각 10.44 및 2.11의 perplexity를 달성했습니다.



### Remaining Useful Life Prediction: A Study on Multidimensional Industrial Signal Processing and Efficient Transfer Learning Based on Large Language Models (https://arxiv.org/abs/2410.03134)
- **What's New**: 이 논문은 장비의 남은 유용 수명(Remaining Useful Life, RUL) 예측을 개선하기 위해 대규모 언어 모델(large language models, LLMs)을 활용한 혁신적인 회귀 프레임워크(regression framework)를 소개합니다.

- **Technical Details**: 제안된 모델은 코퍼스 데이터(corpus data)에서 사전 학습된 LLM의 모델링 능력을 활용하여 복잡한 시간 의존성(temporal dependencies)을 효과적으로 포착하고 예측 정확도를 향상시킵니다. 이 모델은 모든 부분 집합(subset)에 대해 동일한 슬라이딩 윈도우 길이(sliding window length)와 모든 센서 신호를 사용하여 강력한 일관성과 일반화를 보여줍니다.

- **Performance Highlights**: Turbofan 엔진의 RUL 예측 과제에서 제안된 모델은 FD002 및 FD004 부분 집합에서 최신 기술(state-of-the-art, SOTA) 방법을 초월하며, 다른 부분 집합에서는 거의 SOTA 성과를 달성했습니다. 또한 전이 학습(transfer learning) 실험에서 최소한의 타겟 도메인 데이터로 미세 조정(fine-tuning)했을 때, 전체 타겟 도메인 데이터를 기반으로 훈련된 SOTA 방법보다 더 나은 성능을 보였습니다.



### Autoregressive Action Sequence Learning for Robotic Manipulation (https://arxiv.org/abs/2410.03132)
- **What's New**: 이 논문에서는 로봇 조작 작업을 위한 새로운 자율 회귀(autoregessive) 아키텍처인 Chunking Causal Transformer (CCT)를 제안합니다. CCT는 인과(transformer) 모델의 다음 단일 토큰 예측을 다중 토큰 예측으로 확장하여, 효율적인 학습과 실행을 가능하게 합니다.

- **Technical Details**: CCT는 고유의 attention interleaving 전략을 채택하여 teacher-forcing으로 효율적으로 학습할 수 있게 하며, 이로 인해 다양한 로봇 조작 환경에서 근본적인 인과 관계(causal relations)를 활용할 수 있습니다. 논문에서는 주요 로봇 환경인 Push-T, ALOHA, RLBench에서 ARP(Autoregressive Policy) 모델의 성능을 평가합니다.

- **Performance Highlights**: ARP는 모든 테스트 환경에서 최첨단(State-of-the-Art) 방법들을 능가하는 성능을 보여주며, 계산 효율과 파라미터 크기에서 더 높은 효율성을 자랑합니다. 실제 로봇 실험에 대한 비디오 데모와 소스 코드는 논문에 포함되어 있습니다.



### ARB-LLM: Alternating Refined Binarizations for Large Language Models (https://arxiv.org/abs/2410.03129)
Comments:
          The code and models will be available at this https URL

- **What's New**: 본 논문은 ARB-LLM이라는 새로운 1-bit 포스트 트레이닝 양자화(PTQ) 기술을 제안합니다. 이는 대형 언어 모델(LLM)에 최적화되었으며, 이 모델의 메모리 및 계산 요구량을 크게 줄일 수 있습니다.

- **Technical Details**: ARB-LLM은 교차 정제 양자화(Alternating Refined Binarization, ARB) 알고리즘을 기반으로 하여, 양자화 오차를 줄이고, 컬럼 그룹 비트맵(Column-Group Bitmap, CGB) 전략을 개선하여 성능을 향상시킵니다. ARB-X 및 ARB-RC와 같은 확장 기술로, 교정 데이터를 통합하고 컬럼 방향의 편차를 최소화합니다.

- **Performance Highlights**: 실험 결과, ARB-LLM$_{RC}$는 현재의 SOTA 이진 PTQ 방법들보다 훨씬 높은 성능을 보이며, 동일한 크기의 FP16 모델을 초월하는 성과를 거두었습니다. 또한, 이 알고리즘은 대규모 LLM의 실용적인 배포에 필요한 메모리 자원을 최소화합니다.



### Understanding Decision Subjects' Engagement with and Perceived Fairness of AI Models When Opportunities of Qualification Improvement Exis (https://arxiv.org/abs/2410.03126)
- **What's New**: 이번 연구에서는 AI 모델의 결정 공정성(fairness)이 의사결정 대상자(decision subjects)의 참여도와 AI 모델에 대한 공정성 인식에 미치는 영향을 조사했습니다. 의사결정 대상자는 AI 모델의 결정에 반복적으로 전략적으로 반응할 수 있는 경우를 분석했습니다.

- **Technical Details**: 연구는 세 가지 인체 실험을 통해 진행되었으며, 대상자들은 AI 모델이 결정하는 대출 신청 과제를 수행했습니다. AI 모델의 공정성 속성이 의사결정 대상자의 참여도(자신을 개선하려는 의향 및 AI 모델의 결정에 따를 의향)와 공정성 인식에 미치는 영향을 분석했습니다.

- **Performance Highlights**: 연구 결과, 의사결정 대상자가 AI 모델에 전략적으로 반복적으로 반응할 수 있는 경우, AI 모델의 결정 공정성은 의사결정 대상자의 AI와의 상호작용의 지속성이나 자기 개선 의향에 영향을 미치지 않는 것으로 나타났습니다. 그러나 AI 모델이 특정 그룹에 대해 편향을 보일 경우, 의사결정 대상자는 AI 모델을 덜 공정하다고 인식했습니다.



### RIPPLECOT: Amplifying Ripple Effect of Knowledge Editing in Language Models via Chain-of-Thought In-Context Learning (https://arxiv.org/abs/2410.03122)
Comments:
          EMNLP findings

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)에서 지식 편집의 도전 과제인 ripple effect를 해결하기 위한 새로운 접근법, RippleCOT를 제안합니다. 이 방법은 Chain-of-Thought (COT) 추론을 통합하여 다중 단계 질문을 처리할 수 있도록 돕습니다.

- **Technical Details**: RippleCOT는 'new fact, question, thought, answer' 구조로 시연을 구성하며, 새로운 사실을 기반으로 다중 단계 질문의 논리를 식별하고 분해합니다. 이 접근 방식은 첫 번째 단계에서 여러 관계를 식별하고 두 번째 단계에서 생성된 쌍 중에서 질문과 높은 코사인 유사성을 가지는 상위 k 후보를 선택합니다.

- **Performance Highlights**: RippleCOT는 기존의 방법보다 우수한 성능을 보이며, MQuAKE-cf 벤치마크에서 정확도를 7.8%에서 87.1%까지 향상시켰습니다.



### LoRC: Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy (https://arxiv.org/abs/2410.03111)
Comments:
          15 pages, 4 figures

- **What's New**: 이 논문에서는 Key-Value (KV) 캐시 효율성을 높이기 위해 기존 Transformer 기반의 대규모 언어 모델 (LLMs)에 직접 적용할 수 있는 저랭크( low-rank) 근사의 새로운 접근 방식을 제안합니다. 이는 모델 재훈련 없이 사용할 수 있으며, KV 캐시의 메모리 소비를 효과적으로 줄이는 데 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 방법은 KV 가중치 행렬의 저랭크 근사를 통해 적용됩니다. 레이어별 민감성을 반영한 점진적인 압축 전략을 도입하여, 깊은 네트워크에서의 오류 전파를 이론적으로 분석하고, 각 레이어의 압축 오류 경계를 도출합니다. 이로 인해, 초기 레이어에서 발생한 오류가 심화된 레이어보다 더 크게 증가하는 경향이 있습니다.

- **Performance Highlights**: 8B, 13B, 70B 파라미터를 가진 LLaMA 모델에서 다양한 작업을 통해 실험했으며, 이 방법이 GPU 메모리 소모를 크게 줄이면서도 성능에는 미미한 영향을 미친다는 것을 입증했습니다.



### MBDS: A Multi-Body Dynamics Simulation Dataset for Graph Networks Simulators (https://arxiv.org/abs/2410.03107)
- **What's New**: 본 논문에서는 신경망 기반의 물리 시스템 모델링과 시뮬레이션의 한계를 극복하기 위한 새로운 데이터셋인 Multi-Body Dynamics Simulation (MBDS) 데이터셋을 제안합니다. MBDS 데이터셋은 실제 세계의 복잡한 기계 구조를 모델링하며, 기존 데이터셋보다 더 많은 모션 궤적과 더 높은 시간 단계 수를 포함하고 있습니다.

- **Technical Details**: MBDS 데이터셋은 1D, 2D 및 3D 장면을 포함하여 총 150,000개의 동작 궤적과 복합 링크 구조를 제공하여 보다 복잡한 다중 바디 역학 시나리오의 시뮬레이션을 가능하게 합니다. 여기에는 질량, 마찰과 같은 물리적 특성을 포함한 정밀한 다중 바디 동역학 모델링이 포함됩니다.

- **Performance Highlights**: 시뮬레이션의 temporal horizon이 늘어날수록 예측 오차가 증가하는 경향을 보이며, 이는 현재 모델링 프레임워크의 견고성이 부족하다는 것을 강조합니다. 또한 기존 모델들은 특정 시나리오에서만 뛰어난 성능을 보이며, 그 시나리오가 변경될 경우 정확도가 급격히 감소하는 경향이 있습니다.



### Mamba in Vision: A Comprehensive Survey of Techniques and Applications (https://arxiv.org/abs/2410.03105)
Comments:
          Under Review

- **What's New**: Mamba는 컴퓨터 비전 내에서 Convolutional Neural Networks (CNNs)와 Vision Transformers (ViTs)의 한계를 극복하기 위한 새로운 접근 방식으로 등장하였습니다. Mamba는 또는 선형 계산 복잡도를 기반으로 한 Selective Structured State Space Models를 활용하여 장거리 의존성을 효과적으로 캡처하는 데 중점을 두었습니다.

- **Technical Details**: Mamba 모델은 입력 데이터에 기반하여 동적으로 조정되는 선택적 상태 표현을 사용하여 computational overhead를 줄이고 효율성을 높입니다. 이는 구조적 상태 공간 모델(SSMs)의 발전을 기반으로 하여 이루어지며, Mamba는 GPU를 최적화한 스캔 기반 알고리즘을 활용하여 기존의 convolution 기반 SSMs의 비효율성을 피합니다.

- **Performance Highlights**: Mamba 모델은 비디오 처리, 원격 감지, 의료 영상 등 다양한 분야에서 특히 유리하며, CNNs와 ViTs는 높은 계산 요구로 인해 확장성 문제를 겪는 반면, Mamba 모델은 시퀀스 길이에 대한 선형 확장성을 제공하여 실시간 및 대규모 애플리케이션에 적합합니다.



### Combing Text-based and Drag-based Editing for Precise and Flexible Image Editing (https://arxiv.org/abs/2410.03097)
Comments:
          12 pages, 9 figures

- **What's New**: 이번 논문에서는 텍스트 기반 및 드래그 기반 이미지 편집 기술의 단점을 분석하고, 이 두 가지 방법을 결합하여 더욱 정확하고 유연한 이미지 편집을 가능한 CLIPDrag라는 새로운 방법을 제안합니다.

- **Technical Details**: CLIPDrag는 텍스트 신호를 글로벌 가이드로, 드래그 포인트를 로컬 정보로 활용하며, 글로벌-로컬 모션 감독(Global-Local Motion Supervision, GLMS) 방식을 도입하여 텍스트 신호를 드래그 기반 방법에 통합합니다. 또한 빠른 포인트 추적(Fast Point Tracking, FPT) 방법을 통해 성능을 향상시킵니다.

- **Performance Highlights**: CLIPDrag는 기존의 단일 드래그 기반 방법이나 텍스트 기반 방법보다 우수한 성능을 보이며, 양적 및 질적으로 현저한 개선을 기록했습니다.



### Strategic Insights from Simulation Gaming of AI Race Dynamics (https://arxiv.org/abs/2410.03092)
Comments:
          41 pages, includes executive summary. Under review for academic journal

- **What's New**: 이 논문은 'Intelligence Rising' 시나리오 탐색 연습에서 얻은 통찰을 제공합니다. AI의 미래에 대한 가능한 시나리오를 탐색한 결과입니다.

- **Technical Details**: 연구자들은 4년 동안 총 43회의 게임을 운영하며 관찰한 패턴과 전략, 의사결정 과정을 분석했습니다. 주요 전략적 고려사항으로는 AI 경주에 의한 불안정성이며, 국제 협력의 중요성, 기업과 국가의 이해관계 조정의 어려움 등이 있습니다.

- **Performance Highlights**: AI 거버넌스의 복잡성과 불확실성을 참가자에게 체험할 수 있도록 설계된 게임의 효과를 강조합니다. 주요 주제로는 국제 협정의 출현, 사이버 보안의 역할, 그리고 예기치 않은 위기가 AI 경로에 미치는 영향 등이 있습니다.



### Scaling Parameter-Constrained Language Models with Quality Data (https://arxiv.org/abs/2410.03083)
Comments:
          Accepted to EMNLP 2024 Industry Track, 18 pages, 9 figures, 4 tables

- **What's New**: 이번 연구는 언어 모델(리터러리 모델)의 스케일링 법칙을 확장하여 데이터 품질의 중요성을 강조합니다. 특히, 매개변수 수에 의해 제한된 언어 모델에서 효과적인 훈련 토큰(effective training tokens)의 개념을 통합하여 데이터 품질이 모델 성능에 미치는 영향을 분석하였습니다.

- **Technical Details**: 본 논문에서는 모델 매개변수 수가 10억 이하인 경우 데이터의 질, 즉 효과적인 훈련 토큰을 모델 성능의 결정적 요소로 간주하고, 텍스트 다양성(text diversity)과 합성성(syntheticity)이라는 두 가지 지표를 활용하여 계산합니다. 텍스트 다양성은 압축 비율(compression ratio)로 측정하며, 합성성은 교사 모델(teacher model)을 활용한 Perplexity 지표로 평가합니다.

- **Performance Highlights**: 결과적으로, 학습된 200개 모델의 성능을 평가한 결과, 제안된 효과적인 훈련 토큰 개념을 통합한 스케일링 법칙이 기존 방식보다 +0.83 피어슨 상관관계(Pearson correlation)를 보이며, 데이터 샘플링 및 합성 등 널리 사용되는 데이터 기술에서 모델 성능을 크게 향상시켰습니다.



### CommonIT: Commonality-Aware Instruction Tuning for Large Language Models via Data Partitions (https://arxiv.org/abs/2410.03077)
Comments:
          Accepted to EMNLP 2024

- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)의 명령 수행 능력을 향상시키기 위해 새롭게 제안한 'CommonIT' 방법론을 소개합니다. CommonIT는 데이터 샘플링 관점에서 모델의 능력을 향상시키는 데 중점을 두며, 여러 데이터 세트를 군집화한 후 각 군집에 대해 고유한 특성을 고려하여 명령 튜닝을 적용합니다.

- **Technical Details**: CommonIT는 세 가지 기준(작업, 임베딩 및 길이)을 사용하여 명령 데이터 세트를 그룹으로 나누고 각 훈련 미니 배치가 오직 하나의 그룹의 데이터로만 구성되도록 합니다. 이렇게 함으로써 미니 배치 간 데이터의 무작위성을 보장하고, 배치 내 데이터 유사성을 높이게 됩니다. 연구에서는 다양한 LLaMa 모델을 통해 CommonIT의 효과를 검증했습니다.

- **Performance Highlights**: 실험 결과 CommonIT는 일반 도메인에서 평균 2.1% 향상, 특정 도메인에서 5.2% 향상, 그리고 특정 작업인 MMLU에서 3.8% 향상을 보였습니다. 이러한 결과는 모델이 명령을 더 잘 이해하고 응답에 있어 정확성을 높였음을 나타냅니다.



### Multi-Robot Motion Planning with Diffusion Models (https://arxiv.org/abs/2410.03072)
Comments:
          The first three authors contributed equally to this work. Under review for ICLR 2025

- **What's New**: 이 논문에서는 단일 로봇 데이터만을 사용하여 다중 로봇의 충돌 없는 궤적을 생성하는 방법을 제안합니다. 이를 위해 Multi-robot Multi-model planning Diffusion (MMD)라는 알고리즘을 개발하였습니다. 이 알고리즘은 학습된 확산 모델(diffusion model)과 전통적인 탐색 기반 기법을 결합하여 데이터 기반의 움직임을 생성합니다.

- **Technical Details**: MMD는 제약 기반의 다중 에이전트 경로 탐색 문제(multi-agent path finding, MAPF)와 확산 모델을 결합하여 다중 로봇의 모션 플래닝(motion planning) 문제를 해결합니다. 주목할 점은 단일 로봇에 대한 확산 모델만 학습함으로써 다중 로봇 상호작용 데이터를 요구하지 않으며, 차원의 저주(the curse of dimensionality)를 극복한다는 것입니다. MMD는 특수한 공간-시간 가이드 함수(spatio-temporal guiding functions)를 통해 충돌을 피한 궤적을 생성합니다.

- **Performance Highlights**: 시뮬레이션된 물류 환경에서 수십 대의 로봇을 위한 계획에서 MMD 방법의 효과를 입증하였습니다. 다양한 모션 플래닝 문제에서 실험 결과, MMD는 대안 방법들과 비교하여 에이전트 수와 환경 크기에 따라 비율적으로 확장 가능한 성능을 보였습니다.



### Integrating Natural Language Prompting Tasks in Introductory Programming Courses (https://arxiv.org/abs/2410.03063)
Comments:
          7 pages, 6 figures. Accepted for publication at SIGCSE Virtual 2024

- **What's New**: 본 논문은 프로그래밍 입문 과정에서 두 가지 자연어 프롬프트 기반 활동의 통합을 탐구합니다. 이는 학생들이 문제 해결에 중점을 두고 자연어를 사용하여 프로그래밍을 배우도록 돕는 새로운 접근 방식입니다.

- **Technical Details**: 입문 프로그램에서 학생들은 LLM(거대 언어 모델)을 사용하여 코드 생성을 위한 자연어 프롬프트를 작성하고, 주어진 코드 조각에 대해 동등한 코드를 생성하는 프롬프트를 구성하는 두 가지 과제를 수행합니다. 이 연구는 학생들이 전통적인 프로그래밍 평가와 비교해 자연어 작업에서 얼마나 성공적인지를 평가합니다.

- **Performance Highlights**: 학생들이 프로그래밍 학습에서 느끼는 어려움과 전통적인 프로그래밍 과제에서의 성과 간의 관계가 예상과 매우 일치했습니다. 하지만 자연어 과제의 성과는 자가 보고된 어려움과의 관계가 덜 강해, 이 두 가지 과제가 서로 다른 기술을 목표로 하고 있음을 시사합니다.



### Towards an Improved Metric for Evaluating Disentangled Representations (https://arxiv.org/abs/2410.03056)
- **What's New**: 이 논문에서는 disentangled representation learning(분리된 표현 학습)이 담당하는 역할에 주목하고, 신뢰할 수 있는 정량적 disentanglement metric(분리 척도)의 개발에 대한 문제를 다룹니다. 새로운 metric인 EDI(Exclusivity Disentanglement Index)를 제안하여 기존 척도의 한계를 극복하고, 더 나은 안정성을 제공합니다.

- **Technical Details**: 기존의 인기 있는 정량적 disentanglement metric들을 분석하며, 이들 각각의 이론적 기반과 특성의 차이를 설명합니다. EDI는 exclusivity(배타성) 개념을 활용하여 factor-code relationship(인자-코드 관계)을 개선하여 ad-hoc(즉흥적) 결정을 최소화하는 새로운 프레임워크를 제시합니다. 이 metric은 Modularity(모듈화), Compactness(압축성), Explicitness(명확성)을 포함하는 다양한 특성을 측정합니다.

- **Performance Highlights**: EDI는 기존의 metrics들과 비교할 때, calibration(보정), non-linearity(비선형성) 및 noise(노이즈) 하에서의 강인성에 있어 우수한 성능을 보이며, 계산 효율성도 유지합니다. 본 연구는 고품질의 오픈 소스 코드베이스를 제공하여, 연구자들이 결과를 재현하고 추가 연구를 진행할 수 있도록 합니다.



### Permissive Information-Flow Analysis for Large Language Models (https://arxiv.org/abs/2410.03055)
Comments:
          16 pages, 11 figures

- **What's New**: 본 논문은 대형 언어 모델(LLMs)의 정보 흐름 라벨 전파를 보다 관대하게 수행하는 새로운 접근 방식을 제안합니다. 이는 데이터의 무결성과 비밀성을 유지하면서도 라벨 크리프(label creep) 현상을 완화하는 데 중점을 둡니다.

- **Technical Details**: 제안하는 접근 방식은 모델 출력 생성에 영향을 미친 샘플의 라벨만을 전파하고, 필요하지 않은 입력의 라벨은 제거하는 것입니다. 이 방법은 (i) prompt-based retrieval augmentation과 (ii) $k$-nearest-neighbors language model에 기반한 두 가지 변형으로 구현되었습니다.

- **Performance Highlights**: 제안된 접근 방식은 LLM 에이전트 환경에서 85% 이상의 경우에 라벨을 개선하는 성과를 보였습니다. 이는 특히 라벨 전파의 실용성을 강조합니다.



### Scalable Frame-based Construction of Sociocultural NormBases for Socially-Aware Dialogues (https://arxiv.org/abs/2410.03049)
Comments:
          17 pages

- **What's New**: 본 논문은 대화에서 사회적으로 인식된 행동을 지원하기 위해 대형 언어 모델(LLMs)을 활용한 사회문화적 규범(SCN) 제작을 제안합니다. 이를 통해 중국 문화에 특화된 첫 번째 SCN 데이터베이스인 ChineseNormBase를 구축했습니다. 이 데이터베이스는 사회적 맥락을 고려하여 생성된 자연어 규범 진술을 포함하고 있습니다.

- **Technical Details**: SCNs는 사회맥락적 프레임을 이용해 추출되며, 이 과정은 대화의 맥락을 이해하고 환각(hallucination)을 줄이는 데 도움이 됩니다. 실제 대화 데이터가 부족할 경우, 합성 데이터(synthetic data)를 사용하여 SCNs를 효과적으로 생성할 수 있습니다. 이와 더불어, RAG 기반 모델(Retrieval-Augmented Generation)을 통해 다양한 대화 작업에 대한 추론 능력을 시연했습니다.

- **Performance Highlights**: 실험 결과, 합성 데이터를 이용하여 추출한 SCNs의 품질이 금본(gold frames)으로 주석을 달아 만든 실제 대화에서 추출한 SCNs와 동등하다는 것을 확인했습니다. 또한, 은본(silver frames)이나 금본으로 주석을 단 실제 데이터에서 추출된 SCNs의 품질이 주석이 없는 데이터와 비교하여 우수함을 입증했습니다.



### Revealing the Unseen: Guiding Personalized Diffusion Models to Expose Training Data (https://arxiv.org/abs/2410.03039)
Comments:
          Under review

- **What's New**: 최근 Diffusion Models (DMs)의 발전으로 이미지 생성 및 개인화된 스타일 학습이 가능해졌습니다. 그러나 이러한 모델의 미세 조정(checkpoint)을 공유할 경우 데이터 유출과 저작권 침해 우려가 있습니다. 본 논문에서는 FineXtract라는 새로운 프레임워크를 제안하여 온라인에서 공유된 DMs로부터 훈련 데이터를 추출할 수 있는 방법을 모색합니다.

- **Technical Details**: FineXtract 방법은 사전 학습된 모델에서 미세 조정된 모델로의 학습 분포 변화를 모델링합니다. 이 과정에서 사전 학습 모델과 미세 조정 모델의 스코어 함수를 외삽(extrapolate)하여, 고밀도(high-density) 지역으로의 생성 과정을 유도합니다. 클러스터링(Clustering) 알고리즘을 적용하여 생성된 이미지 중 최상위 확률 이미지를 추출합니다. 이 방법은 조건부 및 비조건부 DMs에 모두 적용 가능합니다.

- **Performance Highlights**: WikiArt, DreamBooth 및 실세계 체크포인트를 포함한 여러 데이터셋에서 실험을 통해, 본 방법이 대개 20%의 미세 조정 데이터를 정확히 추출할 수 있음을 입증하였습니다. 이는 기존 방법 대비 월등한 성능을 증명합니다.



### SPINE: Online Semantic Planning for Missions with Incomplete Natural Language Specifications in Unstructured Environments (https://arxiv.org/abs/2410.03035)
- **What's New**: 이 논문에서는 SPINE(온라인 Semantic Planner for missions with Incomplete Natural language specifications in unstructured Environments)이라는 새로운 접근 방식을 소개합니다. 이는 로봇이 자연어로 설명된 미션의 불완전한 사양을 기반으로 하여 임무 수행을 위한 하위 작업을 추론하고 계획을 실시간으로 조정하는 것을 가능하게 합니다.

- **Technical Details**: SPINE은 Large Language Models (LLMs)를 활용하여 미션에 의해 암시된 하위 작업을 추론하고, 이를 Receding Horizon Framework 내에서 실현합니다. 이 방법은 새로운 관찰을 통해 미션의 안전성을 자동으로 검증하고 제거합니다.

- **Performance Highlights**: SPINE은 20,000m² 이상의 복잡한 실외 환경에서 다단계의 의미적 추론 및 탐색을 요구하는 평가 미션에서 경쟁 기준과 비교하여 평가되었습니다. 평가 결과는 SPINE의 효과성과 실용성을 입증하고 있습니다.



### CounterQuill: Investigating the Potential of Human-AI Collaboration in Online Counterspeech Writing (https://arxiv.org/abs/2410.03032)
- **What's New**: CounterQuill은 사용자들이 효과적이고 공감 있는 카운터스피치(counterspeech)를 작성하는 데 도움을 주는 AI 기반 시스템입니다. 이 시스템은 세 단계로 구성되어 있으며, 사용자들이 온라인 혐오 발언을 이해하고 대응 전략을 탐색하도록 돕는 과정을 제공합니다.

- **Technical Details**: CounterQuill은 사용자의 아이디어를 기반으로 카운터스피치를 공동으로 작성하는 기능을 제공합니다. 첫 번째 단계는 카운터스피치의 이해를 돕기 위한 학습 세션이며, 두 번째 단계는 핵심 요소를 식별하고 아이디어를 브레인스토밍하는 것입니다. 마지막 단계는 카운터스피치를 작성하고 수정하는 공동 글쓰기 세션입니다.

- **Performance Highlights**: CounterQuill을 사용한 사용자들은 자신이 참여한 카운터스피치에 대해 더 강한 소유감을 느꼈으며, 더 기꺼이 그 글을 온라인에 게시하고자 했습니다. ChatGPT를 사용할 때보다 카운터퀼의 지원 덕분에 공감 있는 카운터스피치를 작성하는 데 자신감을 느꼈습니다.



### Dynamic Sparse Training versus Dense Training: The Unexpected Winner in Image Corruption Robustness (https://arxiv.org/abs/2410.03030)
- **What's New**: 본 연구는 Dense Training(밀집 훈련)이 모델의 강건성을 최대화하는 전형적인 접근법이라는 일반적인 인식을 questioning(질문)합니다. 특히, Dynamic Sparse Training(동적 희소 훈련) 방법이 Dense Training보다 강건성 측면에서 일관되게 뛰어난 성능을 발휘할 수 있다는 주장을 합니다.

- **Technical Details**: 저자들은 다양한 Deep Learning(딥 러닝) 아키텍처를 사용하여 이미지와 비디오 데이터에서 3종의 동적 희소 훈련 알고리즘을 실험했습니다. 데이터에 대해 다양한 이미지 왜곡을 적용하여 DST가 제공하는 강건성 향상 효과를 분석하고, Dynamic Sparsity Corruption Robustness(DSCR) Hypothesis(가설)를 제안했습니다.

- **Performance Highlights**: DST 방법(예: SET)은 19가지 이미지 왜곡 중 18개의 경우에서 밀집 훈련보다 더 나은 강건성을 보였습니다. 일반적인 왜곡에 대한 DST의 뛰어난 성능을 통해 현재의 강건성 연구의 지경을 넘어서 새로운 가능성을 제시합니다.



### Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting (https://arxiv.org/abs/2410.03024)
- **What's New**: 최근 생성 모델의 발전, 특히 diffusion 모델이 시계열 모델링에 새로운 방향을 열고 있으며, 예측 및 합성에서 최첨단 성능을 달성하고 있습니다. 그러나 diffusion 기반 모델이 단순하고 고정된 prior에 의존하는 문제를 해결하기 위해 TSFlow라는 조건부 flow matching 모델을 개발했습니다.

- **Technical Details**: TSFlow는 Gaussian processes, 최적 수송 경로(optimal transport paths), 데이터 의존적 prior distributions를 결합하여 생성 문제를 단순화합니다. 이 모델은 conditionally Gaussian processes를 통합함으로써 prior 분포를 데이터의 시간적 구조와 더 밀접하게 정렬시킵니다.

- **Performance Highlights**: 8개의 실제 데이터셋에 대한 실험 평가 결과, TSFlow의 생성 능력이 입증되었으며, 조건부 및 비조건부 훈련 모델 모두 예측 벤치마크에서 경쟁력 있는 결과를 달성했습니다. 8개 데이터셋 중 6개에서 다른 방법을 초월하는 성과를 나타냈습니다.



### Is Your Paper Being Reviewed by an LLM? Investigating AI Text Detectability in Peer Review (https://arxiv.org/abs/2410.03019)
- **What's New**: 이번 연구는 현재의 AI 텍스트 탐지 알고리즘이 연구자의 피어 리뷰를 인간과 다양한 최신 대형 언어 모델(GPT-4o 등)로 구별하는 능력을 조사하였다. 기존 접근 방식들이 많은 GPT-4o 작성의 리뷰를 식별하는 데 실패하며 높은 허위 긍정 분류를 발생시키는 문제를 드러냈다.

- **Technical Details**: 본 연구에서는 피어 리뷰의 AI 생성 텍스트를 탐지하기 위해 여러 AI 텍스트 탐지 방법의 적합성을 분석하였다. GPT-4o 및 Llama-3.1 모델을 사용하여 AI 피어 리뷰를 생성하고, 기존의 오픈 소스 및 상용 텍스트 탐지 모델과 성능을 비교하였다. 제안된 새로운 방법은 참조 AI 생성 리뷰와 비교하는 방식으로 GPT-4o 작성 리뷰를 더 효과적으로 탐지하였다.

- **Performance Highlights**: 기존 AI 텍스트 탐지 방법들이 낮은 허위 긍정률을 유지하면서 AI 생성 리뷰를 식별하는 데 한계가 있었다. 우리의 방법은 GPT-4o 작성 리뷰를 97%, Llama-3.1 작성 리뷰를 87-90% 정확도로 탐지하는 성능을 보여주었다. 이는 피어 리뷰에서 AI 작성 텍스트를 탐지하는 데 큰 진전을 의미한다.



### Transforming Teachers' Roles and Agencies in the Era of Generative AI: Perceptions, Acceptance, Knowledge, and Practices (https://arxiv.org/abs/2410.03018)
- **What's New**: 이 논문은 생성적 인공지능(Generative Artificial Intelligence, GenAI)이 교육 분야에서 교사의 역할과 에이전시(Agency)에 미치는 변화를 탐구하며, 교사의 인식, 지식, 수용성 및 GenAI 활용 방식을 설명하는 포괄적인 프레임워크를 제시합니다.

- **Technical Details**: GenAI technologies는 교육 환경에서 점점 더 통합되고 있으며, 교수 및 학습 방식의 변화를 가져오고 있습니다. 특히, ChatGPT와 같은 고급 자연어 처리(natural language processing) 도구, 콘텐츠 생성(content creation) 및 상호작용 학습 보조기구(interactive learning assistants)의 도입이 중요합니다. 이 연구에서는 교사를 관찰자(Observer), 수용자(Adopter), 협력자(Collaborator), 혁신자(Innovator)의 네 가지 역할로 분류합니다.

- **Performance Highlights**: 이 연구는 GenAI가 교육적 잠재력을 최대한 발휘하기 위해 교사가 단순히 도구를 수용하는 것을 넘어서, GenAI 시스템과 함께 지식의 공동 창출(co-creation of knowledge)을 이루어야 함을 강조합니다. 또한 교사의 지속적인 전문성 개발과 제도적 지원이 필요하다는 점을 밝히며, 교사 교육 향상에 기여할 실질적인 시사점을 제공합니다.



### FastAdaSP: Multitask-Adapted Efficient Inference for Large Speech Language Mod (https://arxiv.org/abs/2410.03007)
Comments:
          EMNLP 2024 Industry Track

- **What's New**: 본 연구에서는 Multitask Speech Language Model (SpeechLM)의 효율적인 추론을 위해 Token Reduction을 탐구합니다. FastAdaSP라는 새로운 프레임워크를 제안하여 음성 관련 다양한 작업에 대한 효율성과 성능 간의 균형을 개선합니다.

- **Technical Details**: FastAdaSP는 오디오 토큰 감소 방법을 포함하여 밀집(Dense) 및 희소(Sparse) 작업에 적합한 고속 추론을 가능하게 하는 통합된 프레임워크입니다. 본 연구는 오디오 특징의 효율적인 처리를 위한 레이어 선택 및 작업별 설계를 포함합니다.

- **Performance Highlights**: FastAdaSP는 WavLLM 및 Qwen-Audio 실험에서 7배의 메모리 효율성과 1.83배의 디코딩 처리량을 달성했으며, Emotion Recognition (ER) 및 Spoken Question Answering (SQA)와 같은 작업에서 성능 저하 없이 이러한 효율성을 이끌어냈습니다.



### Task-unaware Lifelong Robot Learning with Retrieval-based Weighted Local Adaptation (https://arxiv.org/abs/2410.02995)
- **What's New**: 이 논문에서는 로봇이 명확한 작업 경계 없이 새로운 기술을 지속적으로 습득하고 이전 기술을 보존할 수 있도록 돕는 새로운 방법론을 제안합니다. 이는 로봇이 경험 재생(Experience Replay)과 테스트 중 선택적 가중치 기법을 이용해 이전 작업에서의 숙련도를 빠르게 회복할 수 있도록 하여 동적인 환경에서의 적응성을 높입니다.

- **Technical Details**: 이 방법론은 에피소딕 메모리(Episodic Memory)를 활용하여 교육 중 경험 재생을 통해 과거의 demonstrations(시연)을 재현하고, 테스트 중에는 로컬 미세 조정을 위해 이를 검색합니다. 선택적 가중치 메커니즘을 사용하여 가장 도전적인 segment(구간)을 강조하여 적응을 최적화합니다. 이를 통해 로봇은 작업 식별자 없이도 지속적인 학습이 가능해집니다.

- **Performance Highlights**: 이 프레임워크는 로봇의 성능 향상을 통해 동적 환경에서도 지속적으로 학습하고 적응할 수 있는 효과적인 솔루션을 제공합니다. 실험 결과, 기존 작업에서의 숙련도를 빠르게 회복할 수 있으며, 명확한 작업 경계 없이도 효과적으로 로봇이 새로운 환경에 적응할 수 있음을 증명합니다.



### Differentiation and Specialization of Attention Heads via the Refined Local Learning Coefficien (https://arxiv.org/abs/2410.02984)
- **What's New**: 본 논문에서는 싱귤러 학습 이론에 기반한 모델 복잡성 척도인 Local Learning Coefficient (LLC)의 개선된 변형인 정제된 LLC (rLLC)를 소개합니다. 이를 통해 트랜스포머 언어 모델의 교육 과정에서 내부 구조가 어떻게 발전하는지를 분석합니다.

- **Technical Details**: rLLC를 사용하여 두 개 층으로 구성된 어텐션 전용 트랜스포머의 개별 구성 요소를 연구하였으며, 어텐션 헤드의 차별화 및 전문화 과정을 분석합니다. 데이터의 구조가 neural network의 특성에 미치는 영향을 살펴보며, 이 행위에 기반한 rLLC의 패턴을 통해 어텐션 헤드들이 전문화되는 과정을 확인합니다.

- **Performance Highlights**: rLLC는 주목할 만한 성능 지표로, 어텐션 헤드들이 교육 과정 동안 어떻게 기능적 역할을 다르게 하는지를 정량적으로 분석할 수 있는 도구입니다. 또한, 새로운 멀티그램 회로를 식별하여 데이터 구조와 손실 경관의 기하학적 특성과 학습 동역학, 그리고 신경망의 출현하는 계산 구조 간의 상관관계를 establishe 하는 데 기여합니다.



### An explainable approach to detect case law on housing and eviction issues within the HUDOC databas (https://arxiv.org/abs/2410.02978)
- **What's New**: 이 논문은 유럽 인권 법원(ECtHR)의 사례 법을 분석하는 자동화 모델을 개발하는 데 초점을 맞추고 있으며, 특히 주택과 강제 퇴거와 관련된 사례를 탐색하기 위한 작업을 진행합니다. 연구는 40,000건 이상의 사례로 구성된 HUDOC 데이터베이스를 효율적으로 분석할 필요성을 강조합니다.

- **Technical Details**: 연구는 Adaptive Chordal Distance-based Subspace Learning Vector Quantization (AChorDS-LVQ) 모델을 사용하여 법적 문서 분류기를 구축했습니다. 이 모델은 텍스트 분석과 인용 패턴의 조합을 활용하여 주택 문제와 관련된 사건을 식별합니다. 또한, 이 모델은 결정에 대한 설명을 제공하는 해석 가능성을 강조합니다.

- **Performance Highlights**: 모델은 특히 주택 및 퇴거 문제와 관련된 새로운 사례를 식별하는 데 성공했으며, 다른 고급 접근 방식과 비교할 때 유사한 성능을 보여주었습니다. 연구에서 수집된 데이터셋은 주택 관련 문제와 관련된 사례를 효율적으로 분류하는 데 도움이 되는 머신러닝 접근 방식을 통해 신뢰할 수 있는 결과를 제공합니다.



### Harm Ratio: A Novel and Versatile Fairness Criterion (https://arxiv.org/abs/2410.02977)
Comments:
          To appear at EAAMO 2024

- **What's New**: 본 논문에서는 envy-freeness에 기반한 새로운 공정성 기준인 individual harm ratio(IHR)를 제안하며, 이는 광범위한 집단 의사결정 설정에 적용될 수 있다. 이 연구는 개인들이 다른 개인에 대해 느끼는 질투나 원한의 감정을 공정성의 중요한 요소로 간주함으로써, 기존 연구에서 간과된 측면을 보완한다.

- **Technical Details**: individual harm ratio(IHR)는 파트너 간의 비교를 하지 않는 비슷한 공정성 기준이며, ‘공공 결과(public outcomes)’ 모델에 적용될 수 있다. IHR은 additve utility를 가진 자원 배분에서 envy-freeness보다 논리적으로 강력한 성질을 갖는다. 또한, group harm ratio(GHR)와 equal-sized group harm ratio(EGHR)와 같은 그룹 확장 또한 정의한다.

- **Performance Highlights**: 실험을 통해 제안된 공정성 기준이 다양한 결정-making 알고리즘 간의 차별성을 제공함을 입증하였다. 이 기준은 투표(voting), 공정한 분배(fair division), 참여 예산(participatory budgeting) 및 동료 평가(peer review)와 같은 다양한 업무(Task)에서 효과적으로 작동한다.



### F-Fidelity: A Robust Framework for Faithfulness Evaluation of Explainable AI (https://arxiv.org/abs/2410.02970)
Comments:
          Preprint; 26 pages, 4 figures

- **What's New**: 최근 연구에서 설명 가능한 인공지능 (XAI) 기술이 발전되었습니다. 비록 딥 러닝 모델에서 의미 있는 통찰력을 추출하는 방법이 제시되었지만, 이러한 XAI 방법의 적절한 평가 방법이 여전히 해결되지 않은 문제입니다.

- **Technical Details**:  기존의 평가 방법은 입력 데이터에서 가장 중요한 특징을 제거하거나 변형시켜 출력 예측의 변화를 관찰하는 방식을 사용하였습니다. 그러나 이는 Out-of-Distribution (OOD) 문제를 초래하여, 변형된 샘플이 원래 데이터 분포를 따르지 않게 됩니다. 새로운 방법인 RemOve And Retrain (ROAR)은 이 문제를 해결하는데 도움을 주긴 하지만, 모델의 재훈련이 항상 수렴되지 않을 수 있습니다. 저자들은 Fine-tuned Fidelity (F-Fidelity)라는 새로운 평가 프레임워크를 제안하였고, 이 프레임워크는 i) 설명에 독립적인 미세 조정 전략을 사용하여 정보 유출 문제를 완화하고, ii) OOD 입력 생성을 방지하기 위한 랜덤 마스킹 작업을 포함합니다.

- **Performance Highlights**: F-Fidelity는 여러 데이터 구조(이미지, 시계열, 자연어)에 대해 Controlled Experiments를 수행하여 기존의 평가 메트릭보다 설명자의 정확한 순위를 복원하는 데 유의미한 개선을 보였습니다. 이 방법은 설명자 신뢰도에 대한 평가에서 뛰어난 성능을 입증하며, 게임 이론 및 경험적으로 설명의 크기와 F-Fidelity 메트릭 간의 관계를 분석할 수 있는 근거도 제공합니다.



### Label-Free Subjective Player Experience Modelling via Let's Play Videos (https://arxiv.org/abs/2410.02967)
Comments:
          9 pages, 3 figures, AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment

- **What's New**: 이번 연구는 Let’s Play 비디오를 활용하여 플레이어 경험 모델링(Player Experience Modelling, PEM)을 개발하는 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: PEM의 개발을 위해 수집된 Angry Birds 게임의 Let’s Play 비디오에서 비디오 프레임과 오디오 정보를 활용합니다. 컷트된 오디오 조각의 크기를 기반으로 한 음성의 크기(Amplitude)와 감정을 매핑하여 Convolutional Neural Network (CNN)를 훈련시킵니다. 오디오의 정상화된 진폭을 변환하여 감정 값을 예측하는 모델을 개발합니다.

- **Performance Highlights**: 인간 피험자 연구를 통해 수집된 생리적 신호 및 설문 조사 데이터와 CNN 출력을 비교함으로써 제안한 모델이 자기보고식 측정과 높은 상관관계를 보이는 것을 확인했습니다.



### AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML (https://arxiv.org/abs/2410.02958)
Comments:
          47 pages, 5 figures

- **What's New**: 이번 논문에서는 AutoML-Agent라는 새로운 다중 에이전트 프레임워크를 제안하고 있습니다. 이 프레임워크는 데이터 검색부터 모델 배포까지 전체 AutoML 파이프라인을 지원합니다.

- **Technical Details**: AutoML-Agent는 사용자의 작업 설명을 기반으로 전문화된 LLM 에이전트 간의 협업을 촉진하며, 배포 준비가 완료된 모델을 제공합니다. 기존 연구와 달리 단일 계획을 세우는 대신 검색 증강 계획(retrieval-augmented planning) 전략을 도입하여 최적의 계획을 탐색합니다. 각 계획은 데이터 전처리(data preprocessing) 및 신경망 설계(neural network design)와 같은 하위 작업(sub-tasks)으로 분해되어, 병렬로 실행되는 전문화된 에이전트에 의해 해결됩니다.

- **Performance Highlights**: 14개의 데이터셋을 활용한 7개의 다운스트림(tasks) 실험에서 AutoML-Agent는 전체 AutoML 프로세스를 자동화하는 데 있어 더 높은 성공률을 보여주었으며, 다양한 도메인에서 좋은 성능을 발휘하는 시스템을 제공합니다.



### Visual Editing with LLM-based Tool Chaining: An Efficient Distillation Approach for Real-Time Applications (https://arxiv.org/abs/2410.02952)
Comments:
          EMNLP 2024

- **What's New**: 본 논문에서는 실시간 애플리케이션에서 도구를 호출하기 위해 LLMs(대형 언어 모델)를 미세 조정하는 실용적인 증류(distillation) 접근 방식을 제시합니다. 특히 영상 편집 작업에서 사용자의 스타일적 요청을 자연어로 해석하고 적절한 도구와 파라미터를 선택하여 원하는 시각적 효과를 얻는 방법에 대해 설명합니다.

- **Technical Details**: 우리는 (작은) 학생 LLM을 (큰) 교사 LLM의 가이드와 행동 신호를 기반으로 미세 조정하는 방법을 도입했습니다. 학생 모델의 성능 평가는 도구 및 파라미터 선택과 관련된 오프라인 메트릭스를 사용하여 수행합니다. 데이터 증가 기법을 통해 낮은 데이터 환경에서도 25% 성능 향상을 보여주었습니다.

- **Performance Highlights**: 실험 결과, 우리 학생 모델은 교사 모델(GPT-3.5-Turbo)의 성능을 성공적으로 맞추었으며, 비용과 지연 시간을 크게 줄이며 산업 애플리케이션에 적합한 해결책을 제시했습니다.



### LLMCO2: Advancing Accurate Carbon Footprint Prediction for LLM Inferences (https://arxiv.org/abs/2410.02950)
Comments:
          9 pages, 11 figures

- **What's New**: 이 논문에서는 LLM(대규모 언어 모델) 추론의 탄소 발자국을 보다 정확하게 예측하기 위한 새로운 모델인 LLMCO2를 소개합니다. 기존의 예측 방법이 불완전함을 보완하여, 요청 구성 및 하드웨어 설정에 따른 탄소 영향을 신속하고 정확하게 추정할 수 있는 도구를 제공합니다.

- **Technical Details**: LLMCO2는 그래프 신경망(GNN) 기반 모델로서, 각 변환기 층의 커널을 그래프로 표현합니다. 각각의 노드는 커널을 나타내고, 엣지는 데이터 의존성을 캡처합니다. 모델은 전처리(prefill)와 디코딩(decode) 단계의 노드 특성을 별도로 인코딩하며, 각 노드의 Roofline 성능을 하드웨어 특성으로 통합합니다. 또한, 일반적인 요청 구성을 주요 변수로 삼아 데이터 샘플링 알고리즘을 개발하였습니다.

- **Performance Highlights**: LLMCO2는 다양한 추론 요청과 GPU 구성을 사용할 때 기존의 ML 기반 에너지 예측자들보다 탄소 발자국 예측 정확도를 51%-123% 개선하였습니다. 이는 LLM의 사용이 증가함에 따라 환경 영향 평가의 필요성을 더욱 부각시킵니다.



### SymmetricDiffusers: Learning Discrete Diffusion on Finite Symmetric Groups (https://arxiv.org/abs/2410.02942)
- **What's New**: 본 논문에서는 SymmetricDiffusers라는 새로운 이산(discete) 확산(diffusion) 모델을 소개하여, $S_n$에 대한 복잡한 분포 학습을 단순화합니다. 이 모델은 심층 신경망을 사용하여 역확산(reverse diffusion) 과정의 개별 전이(transition)를 학습하는 방식으로 문제를 분해합니다.

- **Technical Details**: 이 연구는 유한 대칭군($S_n$)에서의 확산 모델을 다루며, 리플 셔플(riffle shuffle)을 효과적인 전이로 식별하고, PL(Plackett-Luce) 분포의 일반화된 형태를 제안합니다. 또한 샘플링 및 학습 효율성을 향상시키기 위해 이론적으로 기반한 'Denoising Schedule'을 도입합니다.

- **Performance Highlights**: 모델은 4자리 MNIST 이미지 정렬, 노이즈가 있는 MNIST와 CIFAR-10 데이터셋의 직소 퍼즐 해결, 그리고 여행하는 세일즈맨 문제(TSP) 해결 등 다양한 작업에서 최첨단(state-of-the-art) 성능을 달성하거나 비교 가능한 성능을 보였습니다.



### Deep image-based Adaptive BRDF Measur (https://arxiv.org/abs/2410.02917)
Comments:
          9

- **What's New**: 이 논문에서는 gonio-reflectometer 설계를 적용하여 고품질 BRDF(비방향 반사 분포 함수) 측정을 위한 샘플 수를 최소화하는 새로운 방법을 제시합니다. 이 방법은 물질 샘플의 이미지를 입력으로 받아 경량 신경망을 통해 BRDF 모델의 파라미터와 샘플 위치 분포를 추정합니다.

- **Technical Details**: 본 연구에서는 CNN(Convolutional Neural Network) 인코더를 사용하여 단일 이미지에서 BRDF 파라미터를 추정하고, 중요 샘플링 기술을 사용하여 BRDF 측정을 위한 적응형 샘플링 패턴을 도출합니다. 이 과정에서 Ward BRDF 모델을 통해 매개변수를 학습하며, L1 손실을 이미지 및 매개변수에 적용하여 최적의 샘플 수를 결정합니다.

- **Performance Highlights**: 이 접근 방식은 측정 과정을 획기적으로 가속화하며, BRDF 표현의 정확도와 충실도를 유지합니다. MERL 데이터셋을 사용하여 방법의 정확성을 검증하였으며, 대부분의 물질에 대해 적응형 샘플링 전략을 통해 측정 시간을 최소화했습니다.



### Safeguard is a Double-edged Sword: Denial-of-service Attack on Large Language Models (https://arxiv.org/abs/2410.02916)
- **What's New**: 이번 연구는 LLM(대형 언어 모델)의 보안에 대한 새로운 위협을 제기합니다. 공격자가 프롬프트 템플릿에 악의적인 'adversarial prompt'(적대적 프롬프트)를 삽입하여, 안전한 사용자 요청을 잘못된 요청으로 분류하도록 만드는 'denial-of-service (DoS)' 공격 방법을 제안합니다.

- **Technical Details**: 공격자는 사용자의 클라이언트 소프트웨어의 설정 파일에 악성 프롬프트를 삽입할 수 있는 취약점을 악용하여, 짧고 무해한 것처럼 보이는 적대적 프롬프트를 생성합니다. 이 과정에서 gradient(그라디언트) 및 attention(어텐션) 정보를 활용한 최적화 프로세스를 설계합니다. 실험 결과, 이 공격은 Llama Guard 3에서 사용자의 요청을 97% 이상 차단할 수 있습니다.

- **Performance Highlights**: 실험을 통해 다양한 사용자 프롬프트를 토대로 30자의 적대적 프롬프트만으로도 97%의 사용자 요청을 차단할 수 있음을 증명했습니다. 이 연구는 LLM 안전성을 평가하는 새로운 차원을 제공하며, 기존의 jailbreak 공격과는 fundamentally (근본적으로) 다른 점을 강조합니다.



### Streamlining Conformal Information Retrieval via Score Refinemen (https://arxiv.org/abs/2410.02914)
Comments:
          6 pages

- **What's New**: 본 논문에서는 정보 검색(Information Retrieval, IR) 시스템에서의 통계적 보장을 제공할 수 있는 새로운 스코어 리파인먼트(score refinement) 방법을 제안합니다. 이 방법은 기존의 큰 크기의 세트를 생성하는 문제를 해결하여 작은 크기의 세트를 유지하면서도 통계적 보장을 보장합니다.

- **Technical Details**: 우리가 제안하는 스코어 리파인먼트 방법은 단순한 단조 변환(monotone transformation)을 적용하여 IR 시스템의 점수를 조정합니다. 이러한 점수의 정제(refinement)를 통해, 표준적인 위신 예측(conformal prediction) 방법을 사용하여 컴팩트한 세트를 생성하고, 불필요한 계산 비용을 줄이며 응답 속도를 향상시킵니다.

- **Performance Highlights**: 실험 결과, BEIR 벤치마크 데이터셋을 통해 제안한 방법이 경쟁 방식들보다 더 효과적으로 관련 정보를 포함한 소형 세트를 생성함을 확인했습니다.



### Better Instruction-Following Through Minimum Bayes Risk (https://arxiv.org/abs/2410.02902)
Comments:
          Under review at ICLR 2025

- **What's New**: 이 연구는人 수준의 평가를 제공할 수 있는 일반용 대형 언어 모델(LLM) 평가자의 사용을 탐구하며, Minimum Bayes Risk (MBR) 디코딩을 통해 LLM의 성능을 향상시킬 수 있는 가능성을 제시합니다.

- **Technical Details**: MBR 디코딩은 레퍼런스 기반 평가를 통해 후보 출력에서 고품질 출력을 선택하는 방법으로, 이 연구에서는 이를 LLM 판별자와 함께 응용하여 테스트 시간 성능 향상을 목적으로 합니다. 이 방법은 레퍼런스가 없는 경우에도 LLM 판별자가 다른 후보 출력을 이용하여 평가를 수행하도록 하여, 평균 점수가 가장 높은 후보 출력을 최종 출력으로 선택합니다.

- **Performance Highlights**: MBR 디코딩을 사용한 LLM 판별자 Prometheus-2-7B는 AlpacaEval에서 +3.6%의 성능 향상을 보였으며, Llama2-7b는 Llama2-13b보다 더 나은 결과를 보여주었습니다. 또한, 자체 학습을 통해 최종 모델이 MBR 디코딩으로 훈련된 기본 모델의 성능을 초과할 수 있음을 보여주었습니다.



### Cognitive Biases in Large Language Models for News Recommendation (https://arxiv.org/abs/2410.02897)
Comments:
          Accepted at the ROGEN '24 workshop, co-located with ACM RecSys '24

- **What's New**: 이 논문은 대형 언어 모델(LLM)이 뉴스 추천 시스템에 미치는 인지 편향(cognitive biases)의 영향을 다양한 측면에서 탐구하며, 이러한 편향이 시스템의 신뢰성에 어떤 악영향을 미칠 수 있는지를 분석합니다. 또한 데이터 증강(data augmentation), 프롬프트 엔지니어링(prompt engineering), 학습 알고리즘(learning algorithms) 등의 방법을 통해 이러한 편향을 완화하는 전략을 제시합니다.

- **Technical Details**: 이 논문에서는 인지 편향의 다양한 종류(앙커링 편향(anchoring bias), 프레이밍 편향(framing bias), 현 상태 유지 편향(status quo bias), 집단 귀인 편향(group attribution bias))가 LLM 기반 뉴스 추천 시스템에 미치는 영향을 분석합니다. 이를 통해, LLM이 트레이닝 데이터에서 상속받는 인간의 인지 편향이 결과에 어떻게 반영될 수 있는지를 보여줍니다.

- **Performance Highlights**: LLM 기반 뉴스 추천 시스템의 신뢰성을 높이기 위해 제시된 완화 전략은 다음과 같습니다: 합성 데이터 증강(synthetic data augmentation), 반복 수정(self-debiasing via iterative refinement), 인간 피드백을 통한 인지 편향 수정(cognitive debiasing through human feedback). 이러한 접근 방법들은 추천 시스템이 보다 객관적이고 공정한 출력을 생성하도록 도와줍니다.



### Real-World Cooking Robot System from Recipes Based on Food State Recognition Using Foundation Models and PDDL (https://arxiv.org/abs/2410.02874)
Comments:
          Accepted at Advanced Robotics

- **What's New**: 본 연구에서는 로봇이 새로운 레시피를 기반으로 실제 환경에서 요리 동작을 실행할 수 있도록 하는 로봇 시스템을 제안합니다. 이 시스템은 Large Language Model (LLM)과 전통적인 PDDL(planning domain description language) 계획 및 Vision-Language Model (VLM)을 통해 음식 재료의 상태 인식을 배운 요소들을 통합합니다.

- **Technical Details**: 제안된 시스템은 두 가지 주요 문제, 즉 "실제 환경에서 실행 가능한 동작 계획"과 "식자재 상태 변화 인식"을 해결합니다. LLM을 통해 자연어로 된 레시피 설명을 프로그램적으로 해석할 수 있는 요리 기능 시퀀스로 변환하고, PDDL을 적용하여 실행 가능한 행동 계획을 수립합니다. VLM을 사용하여 소량의 데이터를 기반으로 정말 요리 중 재료의 상태 변화를 실시간으로 인식합니다. 이 시스템은 PR2라는 듀얼 암 휠 로봇을 사용하여 새로운 레시피를 가지고 실제 요리를 수행할 수 있음을 입증하였습니다.

- **Performance Highlights**: PR2 로봇은 실제 환경에서 새로운 레시피인 버터가 첨가된 써니 사이드 업 요리와 볶은 브로콜리를 요리하는 실험에서 성공적으로 요리를 수행하였습니다. 이 연구를 통해 제안된 로봇 시스템은 요리 행동을 기반으로 한 레시피 설명에 따른 일련의 실제 요리 동작을 수행할 수 있는 가능성을 보여주었습니다.



### Deep Signature: Characterization of Large-Scale Molecular Dynamics (https://arxiv.org/abs/2410.02847)
Comments:
          17 page, 8 figures

- **What's New**: 이 논문에서는 Deep Signature라는 새로운 계산 프레임워크를 도입하여 복잡한 단백질 동역학 및 원자 간 상호작용을 효과적으로 분석하고 있다. 이는 기존의 접근법이 가진 한계를 극복하는 데 중점을 두고 있으며, 생물학적 과정의 동적 특성을 이해하는 데 기여할 것이다.

- **Technical Details**: Deep Signature는 소프트 스펙트럴 클러스터링(soft spectral clustering) 및 서명 변환(signature transform)을 포함하여 복잡한 동역학을 특성화한다. 이 방법은 시스템 크기를 줄이는 동시에 비부드러운 상호작용 동역학에 대한 전반적인 특성을 제공한다. 또한, 이 방법은 원자 좌표 순서의 치환에 대해 동등성(equivariance)을 유지하고, 시간 재파라미터화에 대해 불변성을 갖는 여러 유리한 특성을 보인다고 한다.

- **Performance Highlights**: Deep Signature 모델은 유전자 조절 동역학(gene regulatory dynamics), EGFR 돌연변이 동역학, GPCR 동역학(G protein-coupled receptors dynamics)이라는 세 가지 벤치마크에서 기존의 기준 방법보다 뛰어난 성능을 보였다. 실험 결과에 따르면, 이 프레임워크는 단백질의 기능적 특성을 예측하는 데 있어 효과적이며, 새로운 약물 치료 개발에 기여할 수 있을 것이다.



### Towards Layer-Wise Personalized Federated Learning: Adaptive Layer Disentanglement via Conflicting Gradients (https://arxiv.org/abs/2410.02845)
- **What's New**: 본 연구에서는 개인화된 연합 학습(personalized Federated Learning, pFL)에서 고유한 데이터 이질성(heterogeneity)으로 인해 발생하는 기울기 발산(gradient divergence) 문제를 해결하기 위해 새로운 접근법인 FedLAG(Federated Learning with Layer-wise Aggregation via Gradient Analysis)를 제안합니다.

- **Technical Details**: FedLAG는 층별(layer-wise) 기울기(gradient) 충돌(conflict) 개념을 활용하여 다양한 클라이언트(client)에서 기울기가 형성되는 각도에 따라 개인화된 학습 과정을 조정합니다. 특히, 기울기가 예각(acute angle)을 형성할 때는 서로 같은 방향으로 정렬되어 클라이언트 불변(feature) 식별에 중점을 두고, 둔각(obtuse angle)을 형성할 경우 클라이언트 특정(tasks) 작업에 주안점을 두게 됩니다. 기울기 충돌이 발생할 경우, 해당 층은 글로벌 집계(aggregation) 과정에서 제외됩니다.

- **Performance Highlights**: 실험 결과 FedLAG는 다양한 최신 방법(state-of-the-art methods)을 초월하는 성능을 보였으며, 다른 기존 방법들과 쉽게 통합되어 성능을 더욱 향상시킬 수 있게 설계되었습니다.



### CAnDOIT: Causal Discovery with Observational and Interventional Data from Time-Series (https://arxiv.org/abs/2410.02844)
Comments:
          Published in Advanced Intelligent Systems

- **What's New**: 본 논문에서는 CAnDOIT라는 새로운 인과 발견 방법을 제안하여 관찰 데이터와 개입 간섭(interventional data)을 활용하여 인과 모델을 재구성하는 데 중점을 두고 있습니다. 기존 연구들과는 달리, 시간적 시계열 데이터를 이용한 인과 발견을 수행하는 첫 번째 접근 방식으로, 실세계 문제 해결에 기여할 것으로 기대됩니다.

- **Technical Details**: CAnDOIT는 인과적 구조를 추론하기 위해 관찰 데이터와 실험적 개입 데이터를 통합하는 알고리즘입니다. 이 방법은 LPCMCI라는 기존의 최첨단 알고리즘을 확장하여 시계열 데이터에서 인과 모델을 정확하게 추출할 수 있도록 개선되었습니다. 주요 구성 요소로는 개입 간섭을 통해 수집된 데이터의 효과적인 분석이 포함되어 있으며, Python 구현도 제공되어 GitHub에서 공개되었습니다.

- **Performance Highlights**: 실험 결과는 CAnDOIT가 랜덤 생성된 합성 모델과 로봇 조작 환경에서 실행된 벤치마크에서 인과 모델을 정확하게 재구성할 수 있음을 보여줍니다. 본 알고리즘은 시계열 데이터의 복잡한 문제를 해결하는 데 있어 기존 방법들보다 우수한 성능을 보이며, 이는 다양한 분야에서의 응용 가능성을 시사합니다.



### Neural DDEs with Learnable Delays for Partially Observed Dynamical Systems (https://arxiv.org/abs/2410.02843)
- **What's New**: 이 논문에서는 부분적으로 관측된 동적 시스템을 모델링하기 위해 Constant Lag Neural Delay Differential Equations (NDDEs)를 도입합니다. 이러한 NDDE는 통계 물리학에서의 Mori-Zwanzig (MZ) 형식을 활용하여 효과적인 모델을 제공합니다.

- **Technical Details**: Constant Lag Neural Delay Differential Equations (NDDEs)는 현재 상태에서 과거의 정보 및 지연을 통합하여 동적 시스템을 모델링하는데 사용할 수 있습니다. NDDE는 일반적인 지연 미분 방정식(DDE)으로, 역사 함수와 지연 함수 등을 포함하여 구성됩니다.

- **Performance Highlights**: NDDE 모델은 합성 데이터 및 실험 데이터를 기반으로 한 평가에서 기존 방법들을 초월하는 성능을 보였습니다. 이러한 결과는 NDDE가 부분적으로 관측된 시스템에 적용할 때 효과적임을 보여줍니다.



### FlipAttack: Jailbreak LLMs via Flipping (https://arxiv.org/abs/2410.02832)
Comments:
          43 pages, 31 figures

- **What's New**: 이 논문은 FlipAttack이라는 간단하면서도 효과적인 jailbreak 공격 방법을 제안합니다. 이 방법은 black-box LLMs를 대상으로 하며, LLM이 텍스트를 왼쪽에서 오른쪽으로 이해하는 경향이 있음을 밝히고, 이를 활용하여 공격을 수행합니다.

- **Technical Details**: FlipAttack은 먼저 유해한 프롬프트를 숨기기 위해 자체적으로 생성된 왼쪽 노이즈를 추가하는 방식으로, 총 4가지 플리핑 모드(Flipping Word Order, Flipping Characters in Sentence, Flipping Characters in Word, Fool Model Mode)를 제안합니다. 이 공격은 단 하나의 쿼리로 black-box LLM을 jailbreak할 수 있는 간단한 방법을 제공합니다.

- **Performance Highlights**: FlipAttack은 8개의 LLM을 대상으로 한 실험에서 뛰어난 성능을 보였습니다. 특히 GPT-4 Turbo에서는 약 98.85%의 공격 성공률을 기록했으며, 평균 5개의 guardrail 모델에 대해 약 98%의 우회율을 달성했습니다.



### PyRIT: A Framework for Security Risk Identification and Red Teaming in Generative AI System (https://arxiv.org/abs/2410.02828)
- **What's New**: Generative Artificial Intelligence (GenAI)의 사용이 확산되고 있는 가운데, 새로운 리스크 식별 프레임워크가 필요해졌습니다. 이를 해결하기 위해 Python Risk Identification Toolkit (PyRIT)이라는 오픈 소스 프레임워크가 소개되었습니다.

- **Technical Details**: PyRIT는 모델 및 플랫폼에 독립적인 툴로서, red teamers가 멀티모달 생성 AI 모델에서 새로운 위험과 jailbreaks를 탐색하고 식별하는 데 도움을 줍니다. 이 툴은 Python으로 작성되어 널리 접근 가능하며, 모듈형 구조를 통해 다양한 공격 조합을 쉽게 시도할 수 있도록 설계되었습니다.

- **Performance Highlights**: Microsoft AI Red Team(AIRT)은 PyRIT를 활용하여 100건 이상의 GenAI 모델에 대한 red teaming 작업을 성공적으로 수행했습니다. 이 연구에서는 PyRIT의 개념 증명(Proof-of-Concept) 실험과 실제 사례 연구를 통해 그 실용적인 응용 사례를 시연합니다.



### Effective Intrusion Detection for UAV Communications using Autoencoder-based Feature Extraction and Machine Learning Approach (https://arxiv.org/abs/2410.02827)
Comments:
          4 pages

- **What's New**: 본 논문은 최근 실제 UAV 침입 데이터 세트를 활용하여 무인 항공기(UAV)에 대한 새로운 침입 탐지 방법을 제안합니다. 기존 연구는 주로 시뮬레이션 데이터 세트나 무관한 데이터 세트를 사용했으나, 본 연구는 실제 데이터 세트를 사용하여 autoencoder 기반의 머신 러닝 침입 탐지 방법을 처음으로 제안한 것입니다.

- **Technical Details**: 제안된 침입 탐지 시스템(IDS)은 '데이터 전처리', '특징 추출', '공격 분류'의 세 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째 단계로, raw data를 전처리하여 null 값 처리 및 데이터 정규화를 수행합니다. 두 번째로, autoencoder를 사용하여 특징을 추출하고 차원 축소를 수행합니다. 마지막으로, 추출된 데이터를 다양한 머신 러닝 모델(예: Random Forest, Support Vector Machine, K-Nearest Neighbors 등)에 입력하여 네트워크 트래픽을 정상 또는 비정상으로 분류하고 특정 공격 유형을 식별합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 이진 및 다중 클래스 분류 작업 모두에서 기존 방법론(특징 선택 기법)에 비해 우수한 성능을 보임을 나타냅니다. 이는 실제 UAV 침입 데이터 세트를 기반으로 한 고급 특징 추출 방법이 공격 탐지에 매우 효과적임을 시사합니다.



### LinkThief: Combining Generalized Structure Knowledge with Node Similarity for Link Stealing Attack against GNN (https://arxiv.org/abs/2410.02826)
- **What's New**: 이 논문에서는 LinkThief라는 새로운 link stealing attack을 제안합니다. 기존의 공격 방식은 비슷한 posterior를 가진 두 노드 간의 edge가 존재한다고 가정하는 데 반해, LinkThief는 이러한 가정에 얽매이지 않고 더 발전된 방법을 사용합니다.

- **Technical Details**: LinkThief는 attackers의 배경 지식으로부터 부분적으로 유출된 target graph와 shadow graph를 결합하여 edge 구조를 이해하기 위한 Shadow-Target Bridge Graph를 생성합니다. 이 과정을 통해 link structure에 대한 통찰을 제공하며, Edge Structure Feature Extractor를 설계하여 일반화된 구조 지식을 추출합니다.

- **Performance Highlights**: 실험 결과, LinkThief는 추가적인 가정을 두지 않고도 효과적으로 링크를 훔치는 것을 보여줍니다. 이론 분석과 실험을 통해 공격 모델의 정확성을 입증하였습니다.



### KLDD: Kalman Filter based Linear Deformable Diffusion Model in Retinal Image Segmentation (https://arxiv.org/abs/2410.02808)
Comments:
          Accepted at BIBM 2024

- **What's New**: 이번 논문에서는 망막 혈관 분할을 위한 새로운 Kalman filter 기반의 Linear Deformable Diffusion (KLDD) 모델을 제안하였습니다. 이 모델은 기존의 U-Net 방식의 한계를 극복하고 소형 혈관과 모세혈관을 보다 효과적으로 분할할 수 있도록 설계되었습니다.

- **Technical Details**: KLDD 모델은 선형 변형 합성곱(linear deformable convolution)과 확산 모델(diffusion model)의 조합을 이용하여 망막 혈관 구조를 정확하게 캡처합니다. Kalman filter를 활용하여 변형 합성곱의 좌표 위치를 최적화하고, Cross-Attention Aggregation module (CAAM) 및 Channel-wise Soft Attention module (CSAM)을 활용하여 특징을 더욱 강화하였습니다.

- **Performance Highlights**: 실험 결과, 제안한 KLDD 모델이 DRIVE, CHASE_DB1, OCTA-500 데이터셋에서 기존의 방법들보다 우수한 성능을 보였으며, 특히 세밀한 혈관 구조를 포착하는 데 있어 높은 정확도를 나타내었습니다.



### AutoPETIII: The Tracer Frontier. What Frontier? (https://arxiv.org/abs/2410.02807)
- **What's New**: 2024년 AutoPET 대회에서는 FDG와 PSMA 두 가지의 트레이서 없이도 PET/CT 스캔에서 병변 분할(lesion segmentation)을 수행할 수 있는 완전 자동화된 알고리즘 개발이 주요 목표이다.

- **Technical Details**: nnUNetv2 프레임워크를 사용하여 6개 모델의 6-fold 앙상블을 훈련하여 PET/CT 병변 분할을 자동으로 수행한다. 각 이미지의 CT와 PET 볼륨을 조합된 4 채널 입력으로 사용하고, 윈도우링(windowing) 기법으로 노이즈를 줄인다.

- **Performance Highlights**: 모델의 정확도는 99.64%로, 전체 데이터셋에 대한 평균 추론(인퍼런스) 시간은 환자당 2.18초였으며, 이 작업의 반복 평가에서는 매우 낮은 잘못된 음성(False Negative) 및 잘못된 양성(False Positive) 비율을 기록하였다.



### Investigating the Impact of Randomness on Reproducibility in Computer Vision: A Study on Applications in Civil Engineering and Medicin (https://arxiv.org/abs/2410.02806)
- **What's New**: 본 논문은 CUDA에 의해 발생하는 무작위성이 기계 학습에서 재현성(Reproducibility) 문제에 미치는 영향을 분석합니다. CUDA는 GPU에서 알고리즘 실행을 가속화하지만, 여러 실행 간 비결정적(Non-deterministic) 행동이 발생할 수 있습니다. 이 연구는 이 문제의 중요성을 강조하며, 다양한 데이터셋에서 성능 점수에서 최대 4.77%의 차이를 초래할 수 있음을 보여줍니다.

- **Technical Details**: 우리는 CIFAR-10, SDNET2018, CBIS-DDSM 데이터셋을 사용하여 CUDA 무작위성의 영향을 분석했습니다. 다양한 하이퍼파라미터 구성에 대해 결정적(Deterministic) 및 비결정적 설정에서 실험을 수행했습니다. ADAM 및 SGD와 같은 두 가지 최적화 알고리즘을 채택하였고, CUDA로 인한 무작위성의 영향을 평가하기 위해 480개의 실험을 수행했습니다. 각 데이터셋에 대해 고정된 시드(Configuration)를 사용하여 무작위성을 통제했습니다.

- **Performance Highlights**: 실험 결과, CIFAR-10에서는 SGD가 ADAM보다 유사한 정확도로 수렴했습니다. SDNET2018에서의 성능 지표가 향상되었고, CBIS-DDSM 데이터셋에서는 AUC 점수가 주요 메트릭으로 보고되었습니다. 결정적 실행의 영향은 성능 변동성을 줄이는데 기여했으며, 최대 4.77%의 성능 차이를 나타내었습니다. ADAM 최적화기에서 F1-score의 표준 편차가 가장 크게 나타나는 등, 무작위성의 영향이 뚜렷하게 드러났습니다.



### Trust-informed Decision-Making Through An Uncertainty-Aware Stacked Neural Networks Framework: Case Study in COVID-19 Classification (https://arxiv.org/abs/2410.02805)
Comments:
          15 pages, 7 figures, 6 tables

- **What's New**: 본 연구는 방사선 이미지를 기반으로 COVID-19를 신뢰성 있게 분류하기 위한 불확실성 인식(stacked) 신경망 모델을 제안합니다. 이 모델은 자동화 시스템에 대한 신뢰를 증진할 수 있도록 불확실한 예측을 사용자에게 알리고, 자신 있게 올바른 예측을 정확히 식별하는 데 중점을 둡니다.

- **Technical Details**: 제안된 모델은 Monte Carlo dropout과 ensemble 기법을 포함한 불확실성 정량화 방법을 통합하여 진단 예측의 확실성을 평가합니다. 두 단계로 구성된 모델 프레임워크에서, 첫 번째 단계 모델은 초기 예측과 관련된 불확실성을 생성하며, 두 번째 단계 모델은 진단 결과와 함께 신뢰 지표(trust indicator)를 생성합니다.

- **Performance Highlights**: COVIDx CXR-4 데이터셋에 대한 광범위한 실험을 통해 신뢰할 수없는 사례와 확실하지 않은 사례를 식별하고 처리하는 데 있어 혁신적인 접근 방식이 입증되어, 임상 환경에서 자동화된 진단의 신뢰성을 증진시킵니다.



### Leveraging Retrieval Augment Approach for Multimodal Emotion Recognition Under Missing Modalities (https://arxiv.org/abs/2410.02804)
Comments:
          Under reviewing

- **What's New**: 이번 논문에서 제안하는 RAMER(Recovery Augmentation for Missing Modality Multimodal Emotion Recognition) 프레임워크는 누락된 모달리티(missing modalities) 상황에서 감정 인식을 개선하기 위해 유사한 다중 모달 감정 데이터를 활용하는 접근 방식을 도입합니다.

- **Technical Details**: RAMER는 다양한 누락 조건 하에서 감정 특징을 검색하여 감정 정보를 보충하는 데 중점을 둡니다. 이를 위해 감정 특징 데이터베이스를 구성하고 관련 다중 모달 감정 데이터를 불러들여 누락된 정보의 공백을 메우게 됩니다.

- **Performance Highlights**: 다양한 실험 결과에 따르면, RAMER 프레임워크는 기존의 최신 방법들에 비해 누락된 모달리티 MER 과제에서 월등한 성능을 보였습니다. 이러한 결과는 감정 인식의 견고함과 정확성을 향상시킬 수 있는 방법론적 기여를 보여줍니다.



### Estimating Body Volume and Height Using 3D Data (https://arxiv.org/abs/2410.02800)
Comments:
          6 pages

- **What's New**: 이 논문은 3D 이미징 기술을 활용하여 비침습적으로 체중을 추정하는 새로운 방법을 제시합니다. RealSense D415 카메라를 사용하여 환자의 고해상도 깊이 맵을 캡처하고, Convex Hull Algorithm을 통해 총 체적을 계산합니다.

- **Technical Details**: 제안된 방법은 3D 모델에서 신체의 주요 포인트 간의 거리를 식별하여 신장(Height)을 추출하고, 포인트 클라우드 데이터를 여러 섹션으로 분리해 개별 볼륨을 합산하여 총 체적을 계산합니다. 이러한 데이터 처리는 MobileNet 및 ResNet과 같은 신경망 아키텍처를 이용하여 신체 구성 요소를 분할 및 분석합니다.

- **Performance Highlights**: 이 비침습적인 체중 추정 방법은 긴급 상황에서 신뢰할 수 있는 체중 데이터를 제공하여 의료 개입의 정확성을 향상시킬 가능성을 보여줍니다. 이 연구는 전문 의료 환경에서 체중 추정을 위한 3D 카메라 기술의 필요성을 강조합니다.



### TaCIE: Enhancing Instruction Comprehension in Large Language Models through Task-Centred Instruction Evolution (https://arxiv.org/abs/2410.02795)
- **What's New**: 연구는 새로운 접근법인 Task-Centered Instruction Evolution (TaCIE)을 소개하며, 이는 기존의 단순한 seed instruction의 진화 방식을 개선하였습니다. TaCIE는 복잡한 지침을 구성 요소로 나눈 다음, 이러한 요소를 조합하여 더욱 복잡하고 다양성이 높은 지침을 생성합니다.

- **Technical Details**: TaCIE는 지침을 배경정보, 목표 및 제약조건으로 분해하여, 각 요소를 세밀하게 수정하고 조합함으로써 진화된 지침을 생성합니다. 이러한 접근법은 LLMs의 difficulty scaling과 cross-domain 적용 가능성을 significantly 향상시킵니다.

- **Performance Highlights**: TaCIE로 fine-tuning된 LLM들이 기존 방법으로 조정된 모델들보다 다양한 벤치마크에서 성능이 현저히 향상되었습니다. 연구 결과는 TaCIE의 우수성을 입증하며, 모델 가중치와 코드가 공개되어 연구 협력을 촉진하고 있습니다.



### DifFaiRec: Generative Fair Recommender with Conditional Diffusion Mod (https://arxiv.org/abs/2410.02791)
Comments:
          The paper was accepted by ICDM 2024

- **What's New**: 이 논문에서는 사용자 선호도 기반으로 공정한 추천을 제공하는 Diffusion-based Fair Recommender (DifFaiRec)라는 새로운 추천 알고리즘을 제안합니다. 이 알고리즘은 조건부 확산 모델(conditional diffusion model)을 기반으로 하여 사용자 선호도의 분포를 효과적으로 학습하고 다양한 추천을 생성할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: DifFaiRec는 두 개의 모듈로 구성됩니다. 첫 번째 모듈은 사용자를 서로 다른 그룹으로 매핑하여 그룹 공정성을 보장하는 역사실 모듈(counterfactual module)입니다. 두 번째 모듈은 이 역사실 모듈에 조건화된 조건부 확산 모델(conditional diffusion model)로, 관찰된 상호작용을 재구성하고 생성을 통해 알려지지 않은 상호작용을 예측합니다. 또한, 정확도(accuracy)와 공정성(fairness)의 두 목적을 하나로 압축하여 최적화 문제의 난이도를 줄입니다.

- **Performance Highlights**: 실험 결과, DifFaiRec는 두 개의 실제 데이터셋에서 기존의 여러 기준선(baselines)을 초월하고 정확성과 공정성을 동시에 유지하는 뛰어난 성능을 보였습니다.



### Logic-Free Building Automation: Learning the Control of Room Facilities with Wall Switches and Ceiling Camera (https://arxiv.org/abs/2410.02789)
Comments:
          5 pages, 3 figures, 2 tables

- **What's New**: 본 논문에서는 Reinforcement Learning(RL)에 의존하지 않고, 사용자의 선호도를 기반으로 하는 Logic-Free Building Automation(LFBA) 아키텍처를 제안합니다. 이 시스템은 기존의 건물 자동화에 비해 더 직관적이고 사용자 친화적인 방식을 제공합니다.

- **Technical Details**: LFBA는 ceiling camera와 wall switches를 활용하여 사용자의 제어 방식을 직접 모니터링하고 학습합니다. 이 연구에서는 VGG, ResNet, Vision Transformer(ViT) 같은 다양한 Deep Learning(DL) 모델을 사용하여 성능을 평가하였으며, VGG 모델이 가장 높은 제어 정확도를 기록했습니다.

- **Performance Highlights**: LFBA 시스템은 다양한 조건과 사용자 활동에 대한 테스트 결과, 93%-98%의 제어 정확도를 달성하였으며, 이는 기존의 DL 모델들보다 뛰어난 성능입니다. 또한 해당 시스템은 향후 더 다양한 스마트 빌딩 응용 프로그램으로 확장 가능성을 보여줍니다.



### Navigation with VLM framework: Go to Any Languag (https://arxiv.org/abs/2410.02787)
Comments:
          under review

- **What's New**: 이번 논문은 Vision Large Language Models (VLMs)를 활용해 인간과 유사한 방식으로 탐색하며 어떤 언어 목표에도 도달할 수 있는 Navigation with VLM (NavVLM)이라는 프레임워크를 소개합니다. 이 프레임워크는 미리 훈련된 모델 없이도 에이전트가 환경 정보를 인식하고 길을 안내받아 목표를 향해 탐색할 수 있도록 합니다.

- **Technical Details**: NavVLM 프레임워크는 에이전트가 특정 또는 비특정 언어 목표를 지향해 탐색할 수 있도록 구성됩니다. 이 시스템은 cognitive core로서 VLM을 사용하여 환경을 인식하고, 목표와 가까워지면 탐색을 종료하고 VLM의 지침에 따라 탐색을 이어갑니다. 중요한 기술 요소로 SLAM (Simultaneous Localization and Mapping) 모듈과 경로 계획 모듈이 있으며, 이를 통해 에이전트는 탐색 중 장애물을 피하면서 실시간으로 업데이트되는 지도 기반으로 행동하게 됩니다.

- **Performance Highlights**: NavVLM은 기존의 특정 목표 설정에 대해 성공률(SR)과 경로 길이로 가중된 성공률(SPL) 모두에서 최첨단 성능을 달성하였습니다. 또한, 비특정 언어 목표에서도 뛰어난 탐색 능력을 보여줍니다. 다채로운 환경에서의 평가 결과, 환경에 대한 깊이 있는 이해와 탐색의 인간적 접근 방식을 성공적으로 구현했습니다.



### Robust Symmetry Detection via Riemannian Langevin Dynamics (https://arxiv.org/abs/2410.02786)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 기존의 대칭 탐지 방법론에 클래식 기법과 최신 생성 모델(generative modeling)의 발전을 결합한 새로운 대칭 탐지 방법을 제안합니다. 특히, Langevin dynamics를 활용하여 노이즈에 대한 강인성을 향상시킴으로써 대칭 탐지의 효용을 극대화하고자 했습니다.

- **Technical Details**: 이 새로운 방법은 재정의된 대칭 공간(symmetry space)에서 미분 방정식을 기반으로 한 Langevin dynamics를 적용하여, 노이즈 환경에서도 효율적으로 포괄적인(global) 및 부분(partial) 대칭을 탐지할 수 있습니다. 기존의 방법들과 달리 데이터로부터 자가 학습 없이도 향상된 노이즈 저항성을 보여줍니다.

- **Performance Highlights**: 제안된 대칭 탐지 알고리즘은 다양한 형태에 대한 임상 실험 결과를 통해 노이즈에 대한 강인성을 입증하며, 2D 및 3D 형태 모두에 대한 대칭성을 효과적으로 탐지할 수 있는 능력을 가집니다. 결과적으로, 이 알고리즘은 대칭 탐지의 강력한 가능성을 제시합니다.



### Enhancing Mental Health Support through Human-AI Collaboration: Toward Secure and Empathetic AI-enabled chatbots (https://arxiv.org/abs/2410.02783)
Comments:
          17 pages, 9 Figures

- **What's New**: 이 논문은 정신 건강 지원에 대한 접근이 제한된 주변 지역 사회에서 AI(chatbot)를 활용한 최신 기술, 특히 대형 언어 모델(LLMs)인 GPT-4, Mistral Large, LLama V3.1의 가능성을 탐구합니다.

- **Technical Details**: AI-enabled(chatbot)를 통해 감정적으로 공감하고, 의미 있는 응답을 제공하는 능력을 평가하며, 이를 위해 연합 학습(federated learning) 프레임워크를 제안하여 데이터 개인 정보 보호를 보장하고, 편향(bias)을 줄이고, 정신 건강 전문가와의 지속적인 검증을 통합합니다.

- **Performance Highlights**: AI 기반 chatbot은 구조화된 응답 생성을 통해 가능성을 보이지만, 인간 치료사와 같은 감정적 깊이와 적응력(reliability)에서는 부족하다는 한계가 존재하며, 신뢰성(trustworthiness), 편향, 개인 정보 보호 등의 도전 과제가 여전히 남아 있습니다.



### Guess What I Think: Streamlined EEG-to-Image Generation with Latent Diffusion Models (https://arxiv.org/abs/2410.02780)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 뇌파로부터 이미지를 생성하는 기술이 주목받고 있으며, 이는 brain-computer interface (BCI) 시스템을 발전시키는 데 기여할 수 있다. 기존 연구들은 주로 고해상도 이미지를 생성할 수 있는 fMRI에 의존했으나, 이 연구에서는 저렴하고 비침습적인 electroencephalography (EEG)를 활용하여 실시간 BCI 응용에 적합한 새로운 방법을 제안하고 있다.

- **Technical Details**: 본 논문에서는 ControlNet 어댑터를 기반으로 한 효율적인 latent diffusion model (LDM) 프레임워크를 제안한다. 이 방법은 EEG 신호를 조건화하여 이미지 생성 과정을 간소화하며, 복잡한 전처리나 다양한 손실 함수, 또는 캡셔닝 모델 없이도 이미지를 생성할 수 있다. 실험을 통해 최첨단 모델들과 비교하여 우수한 성능을 입증하였다.

- **Performance Highlights**: 제안된 GWIT 프레임워크는 minimal preprocessing과 효율적인 트레이닝을 통해 EEG로부터 이미지를 생성하는 데 성공하였다. 본 연구는 EEG 데이터를 사용하여 image generation을 위한 ControlNet의 최초 응용을 보여주며, 기존의 GAN 기반 방법들보다 뛰어난 성능을 보였다.



### Learning variant product relationship and variation attributes from e-commerce website structures (https://arxiv.org/abs/2410.02779)
- **What's New**: 이 논문에서는 VARM(variant relationship matcher) 전략을 소개하여 전자상거래 카탈로그에서 변형된 제품 쌍을 식별하는 방법을 제안합니다. 기존의 엔티티 해상도는 제품 언급이 동일한 기본 제품을 참조하는지를 판단하는 데 중점을 두었으나, 이는 전자상거래 어플리케이션에서 중요한 제품 관계를 포착하지 못합니다.

- **Technical Details**: VARM은 두 가지 요구 사항을 만족시키기 위해 엔코딩(encoding) 및 생성적 AI(Generative AI) 모델의 강점을 활용하는 전략을 개발했습니다. 먼저, 웹페이지의 제품 링크 및 변형된 제품 관계를 포착하는 데이터셋을 구성하여, 주어진 제품 쌍에 대한 변형 매칭을 예측하기 위해 LLM을 훈련합니다. 두 번째로, RAG 기반 생성 LLM을 사용하여 변형된 그룹 간의 변동 및 공통 속성을 추출합니다.

- **Performance Highlights**: 세계적인 전자상거래 소매업체의 실제 데이터를 사용하여 모델 성능을 평가한 결과, 우리의 전략이 대안 솔루션보다 우수한 성능을 나타냈으며, 새로운 유형의 제품 관계를 활용하는 가능성을 제시합니다.



### Mind the Uncertainty in Human Disagreement: Evaluating Discrepancies between Model Predictions and Human Responses in VQA (https://arxiv.org/abs/2410.02773)
- **What's New**: 이번 연구는 Visual Question Answering (VQA) 작업에서 최첨단 비전-언어 모델이 인간 응답의 분포와 얼마나 잘 일치하는지 종합적으로 평가하고, 인간의 불확실성을 (Human Uncertainty in Disagreement, HUD) 고려하는 방법을 제안합니다.

- **Technical Details**: VQA 작업에서 HUD의 영향을 분석하기 위해, 샘플을 낮은, 중간, 높은 3가지 불확실성 수준으로 분류하였으며, 전통적인 정확도 외에도 총 변이 거리 (Total Variation Distance, TVD), Kullback-Leibler 발산 (KL), 인간 엔트로피 보정 오차 (Human Entropy Calibration Error, EntCE) 등의 새로운 지표를 사용하였습니다. 연구 결과, BEiT3와 같은 최신 모델도 다양한 인간 응답의 다중 레이블 분포를 포착하는 데 어려움을 겪고 있다는 것을 확인했습니다.

- **Performance Highlights**: 종합적으로 우리가 제안한 모델 보정 방법은 모델의 신뢰도를 인간의 불확실성과 더 잘 일치시킴으로써 VQA 성능을 향상시킬 수 있음을 보여주었습니다. 연구의 주요 기여는 HUD를 명시적으로 활용하여 VQA 모델의 인간 응답 분포와의 차이를 평가한 것입니다.



### Complex-valued convolutional neural network classification of hand gesture from radar images (https://arxiv.org/abs/2410.02771)
Comments:
          173 pages, 36 tables, 50 figures

- **What's New**: 이번 연구에서는 복소수 (Complex) CNN을 제안하여 손 제스처 인식 시스템의 성능을 향상시키고자 합니다. 기존의 실수 (Real) 기반 알고리즘의 한계를 극복하기 위해 모든 구성 요소가 복소수 도메인에서 작동하도록 설계되었습니다.

- **Technical Details**: 제안된 CV-CNN은 기존의 MLP, CNN, RNN과 같은 다양한 딥러닝 아키텍처의 빌딩 블록을 포함하고 forward 및 backward 연산과 미분(differentiation)을 복소수 도메인에서 수행합니다. 이를 통해 기존의 실수 기반 모델의 두 배의 차원을 요구하는 문제를 해결합니다.

- **Performance Highlights**: 복소수 CNN 모델은 두 세트의 복소수 손 제스처 레이더 이미지 데이터셋에서 실험하였으며, 해당 결과는 기존의 실수 모델과 비교하여 우수한 분류 성능을 입증하였습니다. 또한 복소수 forward residual network를 제안하여 이진 분류에서도 성능을 개선했습니다.



### Fundamentals of legislation for autonomous artificial intelligence systems (https://arxiv.org/abs/2410.02769)
Comments:
          in Russian language

- **What's New**: 이 논문은 자율 기업 관리 시스템 개발 및 구현 과정에서 전용 운영 맥락을 형성하는 방법을 제안합니다. 특히 이 방법은 이사회용 자율 시스템을 예로 들며, 자율 인공지능 시스템에 대한 독특한 운영 맥락을 창출하는 데 중점을 두고 있습니다.

- **Technical Details**: 자율 기업 관리 시스템의 운영 맥락에서 중요한 부분은 기업이 운영되는 규제 및 법적 환경입니다. 자율 인공지능 시스템을 위한 특별한 운영 맥락을 생성하기 위해, 지역 규정 문서의 표현을 사람과 자율 시스템이 사용할 수 있는 두 가지 버전으로 동시에 제시할 수 있습니다. 이러한 접근 방식은 인공지능 시스템이 필요 기준 내에서 기능을 수행할 수 있도록 하는 잘 정의된 운영 맥락을 제공하는 데 기여합니다.

- **Performance Highlights**: 인간과 자율 인공지능 시스템의 공동 작업 특성을 반영하는 지역 규정은 자율 시스템 개발 및 구현에 관한 관련 법률의 기초를 형성할 수 있습니다.



### BoViLA: Bootstrapping Video-Language Alignment via LLM-Based Self-Questioning and Answering (https://arxiv.org/abs/2410.02768)
- **What's New**: 이번 논문에서는 비디오-텍스트 쌍의 주석 작업의 비효율성을 해결하기 위해 LLM 기반의 셀프-질문 및 답변(self-questioning and answering) 프레임워크인 BoViLA를 제안합니다. 이 접근법은 비디오 정보와 LLM 내부 지식을 더 효과적으로 활용하여 모달리티 정렬(modality alignment)을 개선하는 데 기여합니다.

- **Technical Details**: BoViLA는 두 개의 역할을 포함하고 있습니다: 질문자(questioner)와 답변자(answerer)로서, 두 역할은 동일한 모델이 수행하며, 셀프 질문 및 답변을 통해 서로 개선합니다. 또한, 저자는 저품질의 질문을 필터링하기 위해 EDL(Evidential Deep Learning)을 활용하여 불확실성을 추정하고 질문 품질을 평가합니다.

- **Performance Highlights**: BoViLA는 STAR, How2QA, DramaQA, TVQA, VLEP 등 5개의 VideoQA 벤치마크에서 여러 최첨단 방법들보다 우수한 성과를 보이며, 단 4.5M의 훈련 가능한 파라미터로도 효과적임을 입증했습니다.



### AVG-LLaVA: A Large Multimodal Model with Adaptive Visual Granularity (https://arxiv.org/abs/2410.02745)
Comments:
          Preprint

- **What's New**: AVG-LLaVA는 이미지와 지시에 따라 적절한 시각적 세분화를 선택할 수 있는 LMM(대규모 다중모달 모델)입니다. 이 접근법은 시각적 토큰의 수를 줄이고 추론 속도를 증가시키며 모델 성능을 향상시킵니다.

- **Technical Details**: AVG-LLaVA는 (a) 여러 풀링 레이어를 포함하여 다양한 세분화의 시각적 토큰을 얻는 시각적 세분화 스케일러와 (b) Transformer 레이어, MLP 레이어, 투표자 레이어를 포함해 이미지와 지침에 기반하여 적절한 시각적 세분화를 선택하는 시각적 세분화 라우터를 도입합니다. RGLF(Ranking Granularity to align LMM Feedback)라는 새로운 훈련 패러다임을 제안하여 라우터가 시각적 세분화를 효과적으로 구별하도록 지원합니다.

- **Performance Highlights**: AVG-LLaVA는 11개의 벤치마크에서 우수한 성능을 달성하며, AI2D 벤치마크에서 시각적 토큰 수가 85.3% 감소하고 추론 속도가 2.53배 증가하는 등의 성과를 보였습니다.



### Bayes-CATSI: A variational Bayesian deep learning framework for medical time series data imputation (https://arxiv.org/abs/2410.01847)
- **What's New**: 본 논문에서는 Bayesian Context-Aware Time Series Imputation (Bayes-CATSI) 프레임워크를 제안하여 기존 CATSI 모델에 비해 불확실성 정량화를 통합하여 성능을 향상시킵니다.

- **Technical Details**: Bayes-CATSI는 Electroencephalography (EEG), Electrooculography (EOG), Electromyography (EMG), Electrocardiology (EKG)로부터 파생된 시계열 데이터를 사용합니다. 이 모델은 변별적 추론(Variational Inference)을 통해 사후 분포(Posterior Distribution)의 형상을 가정하고 Kullback-Leibler (KL) 발산을 최소화하여 진정한 사후 분포에 가까운 변별 밀도를 찾아냅니다. 또한, Bayes-CATSI는 기존 CATSI 모델 아키텍처에 사용자 정의된 Bayesian 딥 러닝 계층을 통합하여, 모든 결정론적 딥 러닝 계층을 Bayesian 딥 러닝 계층으로 대체합니다.

- **Performance Highlights**: Bayes-CATSI는 CATSI 모델에 비해 9.57% 향상된 성능을 보여줍니다. 또한, 통합된 Bayesian 계층의 활용 덕분에 불확실성 정량화뿐만 아니라 데이터 임퓨테이션 성능 면에서도 뛰어난 결과를 나타냅니다.



New uploads on arXiv(cs.LG)

### Geometric Representation Condition Improves Equivariant Molecule Generation (https://arxiv.org/abs/2410.03655)
- **What's New**: 이번 연구에서는 GeoRCG라는 새로운 프레임워크를 도입하여, 기하학적 표현 조건(geometric representation conditions)을 통합함으로써 분자 생성 모델(molecular generative models)의 성능을 향상시킵니다. 이 방법은 분자 생성을 두 단계로 나누어 첫 번째 단계에서 정보성 기하학적 표현을 생성하고, 두 번째 단계에서 이 표현을 기반으로 분자를 생성하는 방식입니다.

- **Technical Details**: GeoRCG는 분자의 분포(q(ℳ))를 직접 학습하는 것 대신, 잘 훈련된 기하학적 인코더(e.g., Unimol, Frad)를 사용하여 보다 의미 있는 표현 분포로 변환하는 접근 방식을 사용합니다. 이를 통해 분자 구조 및 속성에 대한 정보가 담긴 기하학적 표현이 다음 단계에서 고품질 분자 생성을 유도합니다.

- **Performance Highlights**: GeoRCG는 QM9 및 GEOM-DRUG 데이터셋에서 무조건 분자 생성의 품질을 크게 향상시켰으며, 조건부 분자 생성(task)에서 평균 31%의 성능 향상을 가져왔습니다. 또한, 생성 품질을 유지하면서 확산 단계 수(diffusion steps)를 줄여 생성 과정을 가속화할 수 있음을 보였습니다.



### Real-World Benchmarks Make Membership Inference Attacks Fail on Diffusion Models (https://arxiv.org/abs/2410.03640)
- **What's New**: Diffusion 모델에 대한 Membership Inference Attacks (MIAs)의 평가에서 중대한 결함을 발견하고, 새로운 벤치마크인 CopyMark를 소개했습니다. 현재의 MIA가 사전 훈련된 diffusion 모델에서 비신뢰할 수 있는 방법이라는 경고를 전합니다.

- **Technical Details**: 이 논문에서는 CopyMark라는 새로운 MIA 벤치마크를 제시합니다. 이는 사전 훈련된 diffusion 모델과 편향되지 않은 데이터셋, 공정한 평가 파이프라인을 지원하여 기존 평가의 결함을 극복합니다. Extensive experiments를 통해 현재 MIA 방법들이 더욱 현실적인 조건에서 성능 저하를 겪는다는 것을 보여줍니다.

- **Performance Highlights**: 현재 MIA가 비현실적인 평가 설정에서만 성공적으로 보이며, 실세계 시나리오에서는 성능이 급격히 저하된다는 것을 발견했습니다. 이는 AI 저작권 소송에서 MIA를 증거로 기대하는 사람들에게 중요한 경고가 됩니다.



### Robust Offline Imitation Learning from Diverse Auxiliary Data (https://arxiv.org/abs/2410.03626)
- **What's New**: 이 논문에서는 다양한 보조 데이터에서 강건한 오프라인 모방 학습 방법(ROIDA)을 제안합니다. ROIDA는 전문가 데이터에 대한 추가 데이터 세트를 통합함으로써 데이터 품질에 대한 전제를 두지 않고도 고품질 전환을 식별하여 임무를 달성합니다.

- **Technical Details**: ROIDA는 두 가지 주요 단계로 구성됩니다: 1) 긍정-비레벨(PU) 학습을 사용하여 전문가와 비전문가 전환을 구별하는 판별기 학습, 2) 판별기가 제공하는 점수에 기반한 중요도 샘플링 비율을 통해 모든 보조 데이터의 상태-액션 쌍에 가중치 BC를 적용합니다. 이 방법은 비전문가 전환의 정보를 활용하여 장기적인 보상을 향상시킵니다.

- **Performance Highlights**: 다양한 비율의 전문가 데이터를 포함한 7개의 환경에서의 D4RL 벤치마크 실험 결과, ROIDA는 일관된 고성능을 달성했습니다. 기존 오프라인 모방 학습 방법은 데이터 구성에 대한 특정 가정에 따라 성능이 제한되는 반면, ROIDA는 데이터 품질에 대한 가정을 완화하여 전문가 정책에 대한 포괄적인 지식을 얻는 데 성공하였습니다.



### A Global Medical Data Security and Privacy Preserving Standards Identification Framework for Electronic Healthcare Consumers (https://arxiv.org/abs/2410.03621)
- **What's New**: 본 논문은 전 세계적으로 전자 건강 기록(EHR)의 보안 및 개인 정보 보호를 표준화하기 위한 종합적인 프레임워크인 GDSPS(Global Medical Data Security and Privacy Preserving Standard Identification for Electronic healthcare Consumers)를 제안합니다.

- **Technical Details**: 이 프레임워크는 K-means clustering을 사용하여 20개의 보안 및 개인 정보 보호와 관련된 개념을 5개의 주요 요소로 분류합니다. 또한, EHRs의 보안 및 개인 정보 보호를 위한 구현 우선 순위를 정하는 Ordinal Priority Approach를 적용합니다.

- **Performance Highlights**: 제안된 GDSPS 프레임워크는 다양한 의료 표준들보다 향상된 개인 정보 보호 및 보안을 제공하며, 의료 분야의 전문가 및 정책 담당자에게 유용한 정보를 제공합니다.



### Open-World Reinforcement Learning over Long Short-Term Imagination (https://arxiv.org/abs/2410.03618)
- **What's New**: 이 논문은 LS-Imagine이라는 새로운 시각적 강화 학습(visual reinforcement learning) 방법을 제시합니다. LS-Imagine은 에이전트가 가능한 장기적인 피드백을 탐색할 수 있도록 한정된 상태 전환 단계 내에서 상상을 확장합니다.

- **Technical Details**: LS-Imagine의 핵심 기술은 에이전트가 특정 행동의 장기적인 효과를 효율적으로 시뮬레이션 할 수 있도록 하는 것입니다. 이를 위해, 우리는 목표 조건부 스테이트 전환을 시뮬레이션하고, 특정 이미지 내에서 확대하여 해당 지역의 affordance maps를 계산합니다.

- **Performance Highlights**: 우리의 접근법은 MineDojo에서 기존의 시각적 강화 학습 방법에 비해 우수한 성능을 보여줍니다. LS-Imagine은 즉각적이고 점프한 상태 전환을 모두 포착하여 행동 학습의 탐색 효율성을 향상시킵니다.



### What Matters for Model Merging at Scale? (https://arxiv.org/abs/2410.03617)
Comments:
          20 Pages, 7 Figures, 4 Tables

- **What's New**: 이번 논문에서는 모델 병합(model merging)의 확장성과 관련된 다양한 요소들의 영향을 체계적으로 평가하고, 큰 모델을 기반으로 한 병합의 효용성을 분석합니다.

- **Technical Details**: 1B부터 64B까지의 다양한 모델 크기를 사용하여 4가지 인기 있는 병합 방법인 Averaging, Task Arithmetic, Dare, TIES를 실험합니다. 미세 조정된 모델을 병합하고, 각 병합 모델을 익숙한 작업(held-in tasks)과 전혀 보지 못한 작업(zero-shot tasks)에서 평가합니다.

- **Performance Highlights**: 모델 병합은 강력한 기본 모델(base model)을 사용할 때 더욱 효과적이며, 더 큰 모델일수록 병합을 용이하게 하고 일반화 능력을 지속적으로 향상시키는 것으로 나타났습니다. 8개의 대형 전문가 모델을 병합했을 때, 멀티태스킹(multi-task) 훈련 모델보다 더 나은 일반화 성능을 보였습니다.



### Large Language Model Performance Benchmarking on Mobile Platforms: A Thorough Evaluation (https://arxiv.org/abs/2410.03613)
- **What's New**: 이 논문에서는 모바일 장치에서의 경량 LLM(large language models)의 성능을 종합적으로 측정하고 분석하여 개발자들이 이해할 수 있는 데이터와 통찰을 제시합니다.

- **Technical Details**: 모바일 기기에서의 LLM 성능을 평가하기 위해 token throughput, latency, 배터리 소비와 같은 사용자 경험에 영향을 미치는 메트릭과 CPU/GPU 활용, DVFS 전략, 추론 엔진과 같은 개발자에게 중요한 요소들을 측정했습니다. 또한, Qualcomm, HiSilicon, MediaTek의 SoC(system-on-chip)의 성능 비교를 통해 LLM의 구현이 모바일 환경에서 어떻게 최적화될 수 있는지를 분석했습니다.

- **Performance Highlights**: 우리는 LLM의 성능이 모바일 GPU의 수학적 연산 유닛의 5%~20%만 활용하고 있다는 점을 발견했습니다. Adreno GPU가 Mali GPU에 비해 전반적인 성능에서 일관되게 우수함을 확인했으며, big.LITTLE 아키텍처에서 수치 코어의 수를 조절함으로써 최적의 성능을 얻을 수 있음을 보여주었습니다.



### How Discrete and Continuous Diffusion Meet: Comprehensive Analysis of Discrete Diffusion Models via a Stochastic Integral Framework (https://arxiv.org/abs/2410.03601)
- **What's New**: 이번 연구에서는 Lévy 유형의 확률적 적분(stochastic integral)을 기반으로 한 이산 확산 모델(discrete diffusion models)에 대한 포괄적인 오류 분석 프레임워크를 제안합니다. 이 프레임워크는 포아송 무작위 측정(Poisson random measure)을 시간 의존성과 상태 의존성을 갖춘 강도(intensity)로 일반화하여 명확한 오류 원인을 식별할 수 있게 하는 새로운 통찰을 제공합니다.

- **Technical Details**: 이 논문은 포아송 무작위 측정을 통해 이산 확산 모델을 확률적 적분으로 수식화하고, Itô 적분(Itô integral) 및 Girsanov 정리(Girsanov's theorem)에 유사한 측도 변화 정리를 수립합니다. 본 연구는 KL 발산(KL divergence)에서 처음으로 τ-leaping 스킴에 대한 오류 경계를 도출하며, 세 가지 오류를 분해하는 방법을 제시합니다: 잘림(truncation), 근사화(approximation), 그리고 이산화(discretization).

- **Performance Highlights**: 이 프레임워크는 τ-leaping 및 균일화(uniformization) 스킴을 비교하는 연구를 포함하고 있으며, 오류 분석을 통합하고 강화하여 이산 확산 모델에 대한 더 폭넓은 클래스의 분석을 가능하게 합니다. 또한, 새로운 통계적 기법을 통해 실제 응용에 적합한 효율적이고 정확한 알고리즘 설계에 유용한 통찰을 제공합니다.



### Training Over a Distribution of Hyperparameters for Enhanced Performance and Adaptability on Imbalanced Classification (https://arxiv.org/abs/2410.03588)
- **What's New**: 이 논문에서는 Loss Conditional Training (LCT)이라는 새로운 접근 방식을 제안하며, 여러 하이퍼파라미터 값을 활용하여 성능을 향상시키는 방법을 소개합니다. 이 방식은 클래스 불균형 문제에서 효율성을 높이고, 다양한 정밀도-재현율(Precision-Recall) 거래에서 최적의 성능을 발휘할 수 있습니다.

- **Technical Details**: Loss Conditional Training (LCT)은 단일 모델이 여러 하이퍼파라미터 값의 분포를 통해 훈련되도록 하여, 이로 인해 모델이 여러 모델의 성능을 근사할 수 있도록 합니다. 이 방법은 Focal loss 및 VS loss와 같은 기존의 손실 함수와 결합되어 사용되며, 모델의 AUC(Area Under the ROC Curve), F1 score, Brier score를 개선합니다.

- **Performance Highlights**: LCT는 CIFAR, SIIM-ISIC Melanoma, APTOS Diabetic Retinopathy 등 여러 심각한 불균형 데이터셋에서 훈련 효율성을 높이며, 훈련 후에도 하이퍼파라미터 조정이 가능하여 모델의 성능을 개선합니다. 본 연구 결과, 기존의 여러 방법들과 비교했을 때, LCT를 적용한 모델이 전반적으로 더 나은 성능을 보였습니다.



### HyResPINNs: Adaptive Hybrid Residual Networks for Learning Optimal Combinations of Neural and RBF Components for Physics-Informed Modeling (https://arxiv.org/abs/2410.03573)
Comments:
          14 pages, 6 figures

- **What's New**: 이번 논문에서는 기존의 Physics-informed neural networks (PINNs) 기법에 새로운 접근 방식인 HyResPINNs를 제안합니다. HyResPINNs는 표준 신경망과 Radial Basis Function (RBF) 네트워크를 결합한 적응형 하이브리드 잔차 블록을 통해 PINNs의 능력을 확장합니다.

- **Technical Details**: HyResPINNs는 적응형 조합 파라미터를 통해 각 잔차 블록에서 신경망과 RBF 네트워크의 출력을 동적으로 조정하며, 잔차 블록 간의 적응형 연결을 통해 정보 흐름을 유연하게 처리합니다. 이러한 구조는 PINNs의 전통적인 방식보다 더 큰 수치적 정확도를 제공합니다.

- **Performance Highlights**: HyResPINNs는 다양한 문제에서 기존의 PINNs 및 최신 방법들과 비교하여 월등한 성능을 보였으며, Allen-Cahn 방정식 및 Darcy-Flow 방정식과 같은 복잡한 PDE 문제에서도 견고한 결과를 보여주었습니다.



### Teaching Transformers Modular Arithmetic at Sca (https://arxiv.org/abs/2410.03569)
- **What's New**: 이번 논문에서는 모듈러 덧셈(modular addition)의 ML(Model Learning) 모델 훈련 파이프라인을 개선하기 위한 세 가지 변화를 제안합니다. 더 다양한 훈련 데이터, 각도 임베딩(angular embedding), 그리고 사용자 정의 손실 함수(custom loss function)가 포함됩니다.

- **Technical Details**: 이 연구는 $N = 256$ 및 $q = 3329$의 경우에 성공적인 접근 방식을 시연합니다. 이는 암호학적(application for cryptographic) 응용 프로그램에 유용하며, 이전 연구들보다 $N$과 $q$의 값이 크게 증가한 사례입니다. 제안된 기법들은 다른 모듈러 산술 문제(modular arithmetic problems)에도 일반화 가능합니다.

- **Performance Highlights**: 이 연구는 $N$과 $q$가 대규모로 증가한 케이스에서 성능이 향상되었음을 보여줍니다. 특히, 암호 분석(cryptanalysis)과 관련된 응용 프로그램에 대해 해결할 수 있는 새로운 가능성을 제공합니다.



### Training on more Reachable Tasks for Generalisation in Reinforcement Learning (https://arxiv.org/abs/2410.03565)
Comments:
          arXiv admin note: text overlap with arXiv:2406.08069

- **What's New**: 이 논문에서는 다중작업 강화학습(multi-task reinforcement learning)에서 탐색(exploration)과 일반화(generalization)의 관계를 탐구하고, '도달 가능성(reachability)' 개념을 도입하여 탐색의 역할을 명확히 합니다. 새로운 메소드인 Explore-Go를 제안하여 에피소드 초기에 탐색 단계를 도입함으로써 에이전트가 도달 가능한 작업의 수를 증가시킵니다.

- **Technical Details**: 도달 가능한 작업은 훈련 중에 만날 수 있는 상태와 보상을 포함하는 작업이며, 도달 불가능한 작업은 훈련 작업에 어떤 상태/보상도 공유하지 않는 작업으로 정의됩니다. Explore-Go는 기존의 on-policy나 off-policy 강화학습 알고리즘과 결합될 수 있으며, 에이전트의 경험 수집 방식을 수정하여 작동합니다.

- **Performance Highlights**: Explore-Go를 다양한 환경에서 인기 있는 알고리즘과 결합했을 때, 도달 가능 및 도달 불가능한 작업에 대한 일반화 성능이 향상되었음을 실험적으로 보여주었습니다. 이 연구는 에이전트가 탐색하는 시점과 최적의 도달 가능 작업 수가 일반화 성능과 더 밀접하게 관련되어 있음을 강조합니다.



### Artificial intelligence inspired freeform optics design: a review (https://arxiv.org/abs/2410.03554)
- **What's New**: 인공지능(AI) 기술을 자유형 광학 설계에 통합함으로써 설계 효율성이 크게 향상되었으며, 설계 공간이 확장되고 혁신적인 솔루션이 등장했습니다.

- **Technical Details**: 이 논문에서는 초기 설계 생성(initial design generation), 최적화(optimization), 성능 예측(performance prediction) 등에서의 AI 적용의 최신 발전을 다룹니다. AI의 이점으로는 개선된 정확성(accuracy)과 성능(performance)이 있으며, 데이터 요구사항(data requirements), 모델 해석 가능성(model interpretability), 계산 복잡성(computational complexity) 등의 도전 과제가 있습니다.

- **Performance Highlights**: 자유형 광학 설계에서 AI의 미래는 하이브리드 설계 방법(hybrid design methods), 해석 가능한 AI(interpretable AI), AI 기반 제조(AI-driven manufacturing), 특정 응용 프로그램을 위한 목표 연구(targeted research)에서의 잠재적 발전으로 매우 밝습니다. 연구원, 엔지니어 및 디자이너 간의 협력이 AI의 잠재력을 최대화하고 광학 혁신을 촉진하는 데 필수적입니다.



### Ward: Provable RAG Dataset Inference via LLM Watermarks (https://arxiv.org/abs/2410.03537)
- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG) 시스템에서의 비인가 사용을 탐지하는 방법을 제시합니다. 특히, 데이터 소유자가 자신의 콘텐츠가 어떻게 사용되는지에 대한 강력한 통계적 보장을 받을 수 있는 방법에 중점을 두었습니다.

- **Technical Details**: 연구는 문제를 (black-box) RAG Dataset Inference (RAG-DI)로 공식화하였고, 현실적인 조건에서 RAG-DI 방법들을 벤치마킹하기 위한 새로운 데이터셋을 도입하였습니다. 또한, LLM watermark를 활용한 새로운 RAG-DI 방법인 Ward를 소개합니다. 이 방법은 데이터 소유자가 데이터셋 사용에 대한 통계적 보장을 받을 수 있도록 합니다.

- **Performance Highlights**: Ward는 다양한 어려운 환경에서 모든 기준선(baseline) 방법들을 지속적으로 능가하며, 더 높은 정확도(accuracy), 우수한 쿼리 효율성(query efficiency) 및 강건성(robustness)을 달성합니다. 이 연구는 향후 RAG-DI 연구의 기초를 제공하며, LLM watermark를 효과적인 솔루션으로 강조합니다.



### NRGBoost: Energy-Based Generative Boosted Trees (https://arxiv.org/abs/2410.03535)
- **What's New**: 본 논문에서는 Random Forests(RF)와 Gradient Boosted Decision Trees(GBDT)와 같은 나무 기반 모델의 일반화된 확장을 탐구하고 있으며, 특히 데이터 밀도를 명시적으로 모델링하는 데 중점을 두고 있습니다. 이는 샘플링 외에도 다양한 응용 프로그램을 가능하게 합니다. 주된 기여는 XGBoost와 같은 인기 있는 패키지에서 구현된 2차 부스팅에 비유되는 에너지 기반 생성 부스팅 알고리즘인 NRGBoost를 제안하는 것입니다.

- **Technical Details**: NRGBoost는 likelihood의 로컬 2차 근사를 최대화하도록 학습되는 새로운 에너지 기반 생성 부스팅 모델입니다. 이 모델은 각 부스팅 라운드에서 에너지 기능을 개선하여 데이터 밀도를 모델링합니다. 또한, feature subsampling과 함께 Density Estimation Trees(DET)의 배깅 앙상블 사용을 탐구하며, 이는 Random Forests의 생성적 대안으로 작용합니다.

- **Performance Highlights**: 작은 데이터 세트에서 NRGBoost의 구현은 mid-range 소비자 CPU에서 몇 분 만에 학습되며, 표준 GBDT 모델과 유사한 분류 성능을 달성합니다. 또한, 샘플링을 위한 최고의 생성 모델과 비교해도 경쟁력을 유지합니다.



### No Need to Talk: Asynchronous Mixture of Language Models (https://arxiv.org/abs/2410.03529)
Comments:
          23 pages

- **What's New**: SmallTalk LM은 비동기적(asynchronous) 언어 모델 혼합 훈련 방식으로, 모델 각각이 데이터 분포의 다른 부분에 전문화되어 있으며, 고대역폭 통신 없이도 훈련을 수행할 수 있습니다. 적은 수의 매개변수를 사용하여 추론(inference)에서 경량 경로 설정 라우터가 시퀀스의 적합한 전문가(expert)를 선택하는 방식입니다.

- **Technical Details**: SmallTalk LM은 각 모델이 독립적으로 훈련되고, 이들 모델 간의 통신 비용을 크게 줄입니다. 라우터는 각 전문가의 크기 대비 1.5% 미만의 크기를 가지며, 짧은 접두사를 기반으로 시퀀스의 가장 적합한 전문가를 선택합니다.

- **Performance Highlights**: 실험 결과, SmallTalk LM은 같은 훈련 FLOPs의 조밀한 모델(baseline)들에 비해 유의미하게 낮은 perplexity를 달성했으며, 같은 양의 훈련 데이터를 사용해도 대다수의 하부 작업(overall downstream tasks)에서 그 기준 모델보다 75%의 작업에서 더 나은 성능을 보였습니다.



### A Probabilistic Perspective on Unlearning and Alignment for Large Language Models (https://arxiv.org/abs/2410.03523)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)의 평가를 위한 첫 번째 공식적인 확률적 평가 프레임워크를 도입합니다. 기존의 결정론적 (deterministic) 평가 방식은 모델의 전체 출력 분포를 포착하지 못해 모델의 능력을 정확하게 추정하는 데 한계가 있음을 증명했습니다.

- **Technical Details**: 논문에서 제안한 새로운 지표는 모델의 출력 분포와 관련된 높은 확률 보장을 기반으로 합니다. 이 지표는 응용 프로그램에 의존하지 않으며, 사용자가 배포 전에 모델의 능력을 보다 신뢰성 있게 추정할 수 있게 해줍니다. 더불어, 엔트로피 최적화 (entropy optimization) 및 적응형 온도 조정 (adaptive temperature scaling)에 기초한 새로운 유학 손실도 제안합니다.

- **Performance Highlights**: 연구 결과, 기존의 결정론적 평가 방식에서는 성공적인 유학을 나타내는 반면, 확률적 평가 결과에 따르면 대부분, 아니면 모든 유학되지 않은 정보가 모델 내에 여전히 접근 가능하다는 것을 보여주었습니다. 이는 LLMs의 평가 방식에 있어 중요한 혁신을 제시합니다.



### Improving Online Bagging for Complex Imbalanced Data Stream (https://arxiv.org/abs/2410.03519)
Comments:
          16 pages, 4 figures

- **What's New**: 이 논문은 비대칭적이고 개념 변화(concept drifting)가 있는 데이터 스트림에서 분류기를 학습하는 새로운 접근법을 제안합니다. 기존 연구들은 전반적인 불균형 비율(global imbalance ratio)에만 초점을 맞추었지만, 이 논문에서는 안전하지 않은 소수 클래스 예제(borderline, rare 등)와 같은 지역적(local) 어려움 요인들을 고려하여 네이버후드 언더샘플링(neighbourhood undersampling) 및 오버샘플링(oversampling) 온라인 배깅을 강화합니다.

- **Technical Details**: 제안된 알고리즘은 온라인 배깅(Online Bagging) 방식에 기반하여 Poisson 분포를 활용해 각 예제를 분류기에 전송하는 방식을 조정합니다. 네이버후드 분석(neighbourhood analysis)을 통해 소수 클래스 예제의 안전성 수준(unsafeness level)을 정의하고, 이를 통해 새로운 예제를 처리하는 방식에서 기존의 언더샘플링 및 오버샘플링 기법을 개선합니다.

- **Performance Highlights**: 실험 결과, 새로운 방법론이 기존의 온라인 배깅 리샘플링 앙상블(ensemble)보다 우수한 성능을 보임을 증명하였습니다. 특히, 안전하지 않은 예제의 비율이 증가할 때 기존 분류기들은 성능 저하를 겪는 반면 제안된 방법론은 이러한 상황에서 더 나은 성과를 나타냈습니다.



### Fine-Grained Expressive Power of Weisfeiler-Leman: A Homomorphism Counting Perspectiv (https://arxiv.org/abs/2410.03517)
- **What's New**: 그래프 신경망 (GNN) 분야에서 이식 동형 함수(counting homomorphisms)를 측정하는 새로운 방법에 대한 연구를 소개합니다. 이번 논문에서는 일반화된 포크로레 와이스파일러-레만 (GFWL) 알고리즘을 제안하며, 모든 GNN 클래스의 동형 함수 계산 능력을 결정할 수 있는 이론적 프레임워크를 제공합니다.

- **Technical Details**: GFWL 알고리즘을 통해 대부분의 기존 WL 알고리즘을 포괄하는 통합된 분석 도구를 제공합니다. 이를 통해 다양한 GNN의 대표성을 비교하고 특성을 정량적으로 분석할 수 있는 방법을 제시합니다. GFWL 알고리즘의 동형 함수 계산 능력을 결정하는 메타 알고리즘을 구현하여, 기존의 이론적 결과와 그 출력 결과를 비교하여 그 정확성을 검증하였습니다.

- **Performance Highlights**: GFWL 알고리즘은 동형 함수 계산의 자동화된 절차를 통해 GNN 모델 디자인의 효율성을 크게 향상시킬 수 있는 가능성을 제시합니다. 실험 결과, GFWL 알고리즘은 유명한 여러 WL 변형 알고리즘보다도 우수한 성능을 보여주었으며, 향후 GNN 모델 디자인과 최적화에 많은 활용될 것으로 기대됩니다.



### Stabilized Neural Prediction of Potential Outcomes in Continuous Tim (https://arxiv.org/abs/2410.03514)
- **What's New**: 이 연구에서는 환자 건강 기록을 기반으로 치료 결과를 예측하기 위해 새로운 신경망 방법인 stabilized continuous time inverse propensity network (SCIP-Net)를 제안합니다. 기존의 방법들이 고정된 시간 간격에서만 작동했다면, SCIP-Net은 불규칙한 시간 간격에서도 작동할 수 있도록 설계되었습니다.

- **Technical Details**: SCIP-Net은 시간의 연속성을 고려하여 inverse propensity weighting (IPW) 기법을 응용했습니다. 이를 통해 시간 가변 혼란 요인을 제대로 조정하는 것이 가능해졌습니다. 연구진은 안정화된 IPW를 도출하여 SCIP-Net에서 활용하였습니다. 이 방법은 기존의 방법들보다 우수한 결과를 도출합니다.

- **Performance Highlights**: 실험 결과, SCIP-Net은 기존의 신경망 방법들보다 예측 성능에서 우수함을 입증하였습니다. 특히, 시간 가변 혼란 요인을 적절히 조정할 수 있음에 따라 개인화된 치료 예측의 가능성을 높였습니다.



### FedStein: Enhancing Multi-Domain Federated Learning Through James-Stein Estimator (https://arxiv.org/abs/2410.03499)
Comments:
          12 pages, 2 figures. Accepted at International Workshop on Federated Foundation Models In Conjunction with NeurIPS 2024 (FL@FM-NeurIPS'24)

- **What's New**: 이번 연구에서는 Multi-Domain Federated Learning(FL)에서의 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. FedStein이라는 방법을 통해 James-Stein Estimator를 활용하여 Batch Normalization 통계를 공유하고, 데이터의 다양성을 극복하여 성능 향상에 기여합니다.

- **Technical Details**: FedStein은 클라이언트 간에 Batch Normalization(BN) 통계의 James-Stein 추정값만 공유하므로, 기존의 방법들보다 데이터의 독립성과 정규성에 대한 문제 해결을 돕습니다. Non-BN 레이어의 매개변수는 일반적인 Federated Learning 기법을 이용하여 교환되며, 이 과정에서 모델의 성능을 유지합니다.

- **Performance Highlights**: FedStein은 3개의 데이터셋과 여러 모델에서 실험을 통해 기존의 FedAvg 및 FedBN 방법들과 비교하였을 때, 특정 도메인에서 14% 이상의 정확도 향상을 기록하였으며, 이러한 결과는 다양한 도메인 일반화에 긍정적인 영향을 미쳤습니다.



### Collaborative and Efficient Personalization with Mixtures of Adaptors (https://arxiv.org/abs/2410.03497)
Comments:
          36 pages, 10 figures

- **What's New**: 이번 연구에서는 Federated Learning(FL)에서 데이터의 비독립적이고 비동질적인(non-iid) 특성과 관련하여, 개념 변화(concept shifts)에 기반한 이질성(heterogeneity)에 초점을 맞추었습니다. 특히, 클라이언트 별로 다른 작업(task) 적응을 위한 multi-task learning 환경에서, 각 클라이언트가 자신의 작업에 맞춰 파라미터 효율적인 어댑터(adaptor)를 혼합하여 학습하는 프레임워크(FLoRAL)를 제안하였습니다.

- **Technical Details**: FLoRAL은 Low-Rank Adaptors(LoRA)를 기반으로 하여 다양한 계층(layer)에 확장 가능한 모델 파라미터화(parameterization) 방법으로, 여러 FL 알고리즘 위에서 작동할 수 있습니다. 이는 각 클라이언트가 지역적으로 혼합 벡터(mixture vector)를 사용하여 어댑터를 조합함으로써, 개인화된 모델을 생성하는 데 있어 메모리 효율적이며 작은 상태 정보를 필요로 합니다.

- **Performance Highlights**: FLoRAL은 MNIST, CIFAR-10 및 CIFAR-100과 같은 실제 federated multi-task 문제에서 promising experimental results를 보여주었습니다. FLoRAL은 최적의 클러스터 할당을 가진 전체 모델의 앙상블(ensemble)을 초과하는 성능을 보였으며, 오버피팅을 방지하는 강인성(robustness) 또한 확인되었습니다.



### Fourier PINNs: From Strong Boundary Conditions to Adaptive Fourier Bases (https://arxiv.org/abs/2410.03496)
Comments:
          24 pages, 15 figures

- **What's New**: 이 논문은 Physics-Informed Neural Networks (PINNs)의 한계를 극복하기 위해 새롭게 고안된 Fourier PINNs 아키텍처를 소개합니다. 기존의 PINNs가 고주파 및 다중 스케일 솔루션을 학습하는 데 문제를 겪는 반면, Fourier PINNs는 고주파 구성 요소를 더 잘 학습할 수 있도록 설계되었습니다.

- **Technical Details**: Fourier PINNs는 표준 신경망(Neural Network)과 선형 조합을 취한 Fourier 기반을 결합하여 다양한 주파수를 효율적으로 샘플링합니다. 이 아키텍처는 주어진 훈련 중에 경량화된 Fourier 기저를 최적화하며 중요하지 않은 기저는 제거하는 적응 학습 및 기저 선택 알고리즘을 구현합니다. 강한 경계 조건(Strong Boundary Condition)을 만족하며, 기존의 PINNs와 달리 경계 조건, 문제 도메인에 대한 제약 없이 유연하게 적용 가능합니다.

- **Performance Highlights**: 여러 벤치마크 PDE에 대해 테스트한 결과, Fourier PINNs는 꾸준히 낮은 솔루션 오차(예: ∼10^-3 또는 ∼10^-4)를 달성했습니다. 반면, 표준 PINNs는 이러한 정확도를 달성하지 못했습니다. 랜덤 Fourier 특성(RFF-PINNs)을 사용한 경우도 다양한 가우시안 분산 및 스케일에서 실패하는 경향이 있었습니다.



### Generative Artificial Intelligence for Navigating Synthesizable Chemical Spac (https://arxiv.org/abs/2410.03494)
- **What's New**: SynFormer는 합성 가능한 화학 공간을 효과적으로 탐색하는 생성 모델링 프레임워크로, 전통적인 분자 생성 방식과는 달리 분자의 합성 경로(synthetic pathways)를 생성하여 설계가 실제로 합성 가능한지를 보장합니다.

- **Technical Details**: 이 프레임워크는 확장 가능한 트랜스포머 아키텍처(transformer architecture)와 노이즈 제거 확산 모듈(denoising diffusion module)을 결합하여 분자 구축 블록을 선택합니다. SynFormer는 크게 두 가지 형태로 구현됩니다: (1) SynFormer-ED, 주어진 입력 분자에 대응하는 합성 경로를 생성하는 인코더-디코더 모델이며, (2) SynFormer-D, 특정 속성 목표를 위한 합성 경로를 생성하는 디코더 전용 모델입니다. 두 모델 모두 115개 반응 템플릿과 223,244개의 상업적으로 판매 가능한 구축 블록을 기반으로 훈련되었습니다.

- **Performance Highlights**: SynFormer는 (a) Enamine REAL과 ChEMBL 화학 공간 내의 분자 재구성, (b) 참조 분자를 기반으로 하는 지역 합성 가능한 화학 공간 탐색, (c) 블랙박스 속성 예측 모델에 의해 안내되는 전 세계 합성 가능한 화학 공간 탐색에서 성공을 거두었습니다. 이는 SynFormer의 다양한 제어 전략을 통해 합성 가능한 화학 공간을 탐색하고, 실제 분자 설계 사례에서의 적용 가능성을 높인다는 점에서 중요합니다.



### On the Hardness of Learning One Hidden Layer Neural Networks (https://arxiv.org/abs/2410.03477)
Comments:
          18 pages

- **What's New**: 본 연구에서는 $	ext{ReLU}$(Rectified Linear Unit) 신경망의 한 개 숨겨진 층을 학습하는 문제를 다룹니다. 이 문제는 표준 암호학적 가정 하에서 어려운 문제임을 보였습니다.

- **Technical Details**: 우리는 입력 차원 $	ext{d}$에 대해 다항식으로 크기를 가진 신경망, 표준 가우시안 입력 분포 및 가우시안 노이즈가 $	ext{d}$에 대해 다항식으로 작을 때조차도 이 학습 문제의 어려움을 증명했습니다. 이는 연속 학습 오류(Continuous Learning with Errors, CLWE) 문제의 난이도를 기반으로 하며, 특정적으로는 최단 벡터 문제(shortest vector problem)를 근사적으로 해결하는 데 필요한 최악의 경우 난이도에 크게 의존합니다.

- **Performance Highlights**: 이 연구에서 도출된 결과는 신경망 학습 알고리즘의 안전성에 대한 새로운 통찰을 제공하며, 암호학 분야와 머신러닝의 교차점에서의 문제 해결에 기여할 수 있을 것으로 기대됩니다.



### Vulnerability Detection via Topological Analysis of Attention Maps (https://arxiv.org/abs/2410.03470)
Comments:
          Accepted to ITaS2024. Contains 8 pages

- **What's New**: 최근 취약점 감지(vulnerability detection)에 대한 딥러닝(deep learning, DL) 접근 방식이 주목받고 있으며, 이 연구에서는 BERT 모델의 attention 행렬을 활용하여 새로운 방법을 탐구했습니다. 전통적인 머신러닝(machine learning, ML) 기법이 추출된 최상위 특징을 기반으로 경쟁력 있는 성능을 보였다는 점이 주요 발견입니다.

- **Technical Details**: 본 연구에서는 topological data analysis (TDA) 도구를 사용하여 BERT 모델의 attention 행렬에서 취약점 감지를 위한 특징을 추출하고, 이를 통해 생성된 attention 그래프의 지속적 동질성(persistent homology)을 계산하여 상징적인 정보를 포착합니다. TDA는 데이터를 분석할 때 특정 임계값을 설정하는 대신, 여러 점의 연결성(clusters) 및 주기적 구조(cycles)를 확인할 수 있게 해줍니다.

- **Performance Highlights**: Logistic Regression, Support Vector Machine(SVM), Gradient Boosting 분류기를 대상으로 한 실험에서는, 제안된 방법이 Devign 데이터셋에서 효과적인 성능을 발휘하여 CodeBERTa와 동등한 수준의 경쟁력이 있을 수 있음을 입증했습니다. 이는 전통적인 정적 코드 분석 도구보다 유의미한 장점으로 작용할 수 있습니다.



### S7: Selective and Simplified State Space Layers for Sequence Modeling (https://arxiv.org/abs/2410.03464)
Comments:
          23 pages, 3 figures, 11 tables. Equal contribution by Taylan Soydan and Nikola Zubić

- **What's New**: S7라는 새로운 State Space Model (SSM)을 제안하며, 입력 의존성을 처리할 수 있도록 설계되었습니다. 이는 안정적인 재매개변수화(stable reparameterization) 및 특정 디자인 선택을 통해 입력 내용에 따라 동적으로 상태 전환을 조정함으로써 효율성과 성능을 유지합니다.

- **Technical Details**: S7는 입력 의존성이 있는 상태 전이(state transition)를 유지하는 간소화된 모델로, 긴 시퀀스 모델링에서 안정성을 보장하고 그래디언트 노드를 제어하여 효율적인 트레이닝을 가능하게 합니다. 이전 모델들과 달리 입력에 따라 동적으로 필터링할 수 있는 능력을 제공합니다.

- **Performance Highlights**: S7는 DVS-Gesture에서 99.2%, Spiking Heidelberg Digits에서 96.3%, Spiking Speech Commands에서 88.2%의 정확도로 이벤트 기반 비전 데이터셋에서 최첨단 결과를 달성했습니다. Genomics 분류에서는 EigenWorms 데이터셋에서 97.5%의 정확도를 기록하며 긴 의존성을 효과적으로 포착했습니다. Long Range Arena 벤치마크에서도 다양한 작업에서 우수한 성능을 보였습니다.



### Diffusion State-Guided Projected Gradient for Inverse Problems (https://arxiv.org/abs/2410.03463)
Comments:
          preprint. under review. RZ and BT have equal contributions

- **What's New**: Diffusion State-Guided Projected Gradient (DiffStateGrad)라는 새로운 접근 방식을 제안하여, 데이터 매니폴드(data manifold) 상에 남아있도록 하여 역문제(inverse problems)를 해결하는 데 있어 확산 모델의 성능 및 견고성을 향상시킵니다.

- **Technical Details**: DiffStateGrad는 측정 가이던스.gradient을 저차원 저랭크 서브스페이스(low-rank subspace)에 투영하는 과정을 포함하며, 이 서브스페이스는 확산 과정의 중간 상태(intermediate state)를 근사합니다. 이 과정은 singular value decomposition (SVD)를 통해 수행되며, 저차원 투영을 통해 지역 매니폴드 구조에 수직한 방향을 제거합니다.

- **Performance Highlights**: DiffStateGrad는 확산 모델의 시간 가이던스 스텝 크기(measurement guidance step size)와 노이즈에 대한 견고성을 개선하며, 조건부 화소 결합을 사용하여 이미지 복원 시의 성능을 상승시킵니다. 예를 들어, 큰 스텝 크기와 높은 측정 노이즈 상황에서 PSNR이 20 미만으로 떨어지는 실패율을 크게 줄였습니다.



### Linear Transformer Topological Masking with Graph Random Features (https://arxiv.org/abs/2410.03462)
- **What's New**: 이 논문에서는 그래프 구조 데이터를 처리하기 위한 transformer의 topological masking(위상 마스킹) 개념을 제안하며, 이를 통해 학습 가능한 함수로 위상 마스크를 파라미터화하는 새로운 접근 방식을 소개합니다.

- **Technical Details**: 우리는 weighted adjacency matrix(가중 인접 행렬)를 사용하여 transformer의 attention(어텐션) 메커니즘을 조정하는 방식으로 topological masking을 구현합니다. 이 과정에서 graph random features(그래프 랜덤 특징)를 사용해 성능을 최적화하며, 이를 통해 입력 토큰 수에 대해 $	ext{O}(N)$의 시간 및 공간 복잡도를 유지합니다. 이는 이전의 $	ext{O}(N 	ext{log} N)$ 시간 복잡도를 가진 방법과 비교할 때 훨씬 효율적입니다.

- **Performance Highlights**: 제안된 방법은 이미지 및 포인트 클라우드 데이터 분석에서 강력한 성능 향상을 보이며, 특히 30k 이상의 노드를 가진 대규모 포인트 클라우드에서 뛰어난 정확도를 입증했습니다.



### MLLM as Retriever: Interactively Learning Multimodal Retrieval for Embodied Agents (https://arxiv.org/abs/2410.03450)
- **What's New**: 본 연구에서는 MLLM(다중 모달 대형 언어 모델)을 사용한 새로운 경로 검색 방법인 MART(MLLM as ReTriever)를 제안합니다. 기존의 검색 방법들이 표면적인 유사성에만 집중하고 있는 반면, MART는 상호작용 데이터와 선호 학습(preference learning)을 활용하여 MLLM 검색기를 미세 조정(fine-tune)하는 방식으로 경로의 효과성을 충분히 고려합니다.

- **Technical Details**: MART는 전문 경로를 MLLM 프롬프트로 사용하고, 환경과 상호작용하여 다양한 성공률을 수집하여 선호 쌍(preference pairs)으로 구성하는 방법론을 갖추고 있습니다. 이를 바탕으로 MLLM(LLaVA)과 브래들리-테리 모델(Bradley-Terry model)을 결합하여 미세 조정 과정을 거칩니다. 또한, Trajectory Abstraction 메커니즘을 통해 경로를 요약하여 적은 수의 토큰(token)으로 표현하되 중요한 정보를 유지합니다.

- **Performance Highlights**: 다양한 환경에서의 실험 결과, MART는 기존 방법에 비해 10% 이상의 성능 개선을 보였습니다. 본 연구는 새로운 다중 모달 검색 패러다임을 제시하며, MLLM을 경로 효과성을 평가할 수 있는 검색기로 미세 조정하는 방법론을 통해 임의의 환경에서 에이전트의 작업 성공률을 높이는 데 기여합니다.



### Zebra: In-Context and Generative Pretraining for Solving Parametric PDEs (https://arxiv.org/abs/2410.03437)
- **What's New**: 이 논문에서는 Zebra라는 새로운 generative autoregressive transformer를 소개합니다. Zebra는 파라메트릭 편미분 방정식(PDE)을 해결하기 위해 설계되었으며, 추론 단계에서 gradient adaptation이 필요하지 않습니다.

- **Technical Details**: Zebra는 encode-generate-decode 프레임워크를 사용하여 작동합니다. 먼저, 물리적 상태를 디스크리트 토큰으로 압축하는 vector-quantized variational auto-encoder(VQ-VAE)를 학습합니다. 이후, 다음 토큰 목표로 pretrained된 generative autoregressive transformer가 사용됩니다. Zebra는 다양한 크기의 context를 처리할 수 있으며, 불확실성 정량화(uncertainty quantification)를 지원합니다.

- **Performance Highlights**: Zebra는 다양한 도전적인 PDE 시나리오에서 평가되었으며, 적응성, 견고성, 그리고 기존 접근 방식에 비해 뛰어난 성능을 입증했습니다. 특히, Zebra는 one-shot 설정과 제한된 과거 프레임에서의 데이터로부터 동작을 유도하는 상황에서도 경쟁력 있는 성능을 보여주었습니다.



### Cayley Graph Propagation (https://arxiv.org/abs/2410.03424)
Comments:
          20 pages, 6 figures

- **What's New**: 이 연구는 Graph Neural Networks (GNNs)의 정보 흐름 문제를 해결하기 위해 새로운 접근 방식인 CGP (Complete Cayley Graph Propagation)를 제안합니다. CGP는 과도한 정보 압축(over-squashing)을 방지하기 위해 완전한 Cayley 그래프 구조를 활용하여 정보 전파를 강화합니다.

- **Technical Details**: CGP는 Cayley graphs of the SL(2,ℤ_n) 특수 선형 그룹을 기반으로 하며, 노드 간의 정보 흐름이 원활해집니다. 이 연구는 GNN의 데이터 처리 구조를 변경하여, 정보가 병목 현상 없이 원활하게 전달되도록 합니다.

- **Performance Highlights**: 여러 실제 데이터 세트를 통한 실험 결과, CGP는 EGP (Expander Graph Propagation)와 비교하여 상당한 성능 향상을 보였으며, 또한 복잡한 그래프 재구성과 유사하거나 더 나은 성능을 나타냈습니다.



### Predictive Coding for Decision Transformer (https://arxiv.org/abs/2410.03408)
Comments:
          8 pages, IROS 2024 (Code: this https URL)

- **What's New**: 최근 오프라인 강화 학습(offline reinforcement learning, RL) 분야에서는 리턴을 조건으로 하는 슈퍼바이즈드 러닝(supervised learning) 방식으로 의사결정 문제를 효과적으로 표현하는 방법이 제안되었습니다. 특히, Decision Transformer(DT) 아키텍처가 여러 도메인에서 promising한 성과를 보여주었습니다. 하지만 DT는 목표 조건(goal-conditioned) 강화 학습에서 몇 가지 도전적인 데이터세트에서 성능 저하를 경험하였습니다. 이에 대한 해결책으로 Predictive Coding for Decision Transformer(PCDT) 프레임워크를 제안합니다.

- **Technical Details**: PCDT 프레임워크는 일반화된 미래 조건(generalized future conditioning)을 사용하여 DT 방법을 향상시키는 구조로 구성되어 있습니다. 본 연구는 공격 상태-행동쌍을 입력으로 받아 후속 행동을 예측하는 인과 변환기(causal transformer)를 활용하여 목표 달성에 필요한 행동을 할 수 있도록 합니다. 이러한 방식으로 PCDT는 수집이 용이한 보상 없는 데이터에서 효율성을 극대화하고, 비구조화된 및 최적이 아닌 데이터셋에서도 효과적으로 학습할 수 있습니다.

- **Performance Highlights**: PCDT는 AntMaze와 FrankaKitchen 환경에서 총 8개의 데이터 세트에서 extensive한 실험을 수행하여 기존의 인기 있는 가치 기반(value-based) 및 변환기 기반(transformer-based) 방법들과 동등하거나 우수한 성능을 달성하였습니다. 또한, 실제 로봇을 이용한 목표 달성(task) 실험에서도 PCDT의 유효성을 검증하였습니다.



### EBES: Easy Benchmarking for Event Sequences (https://arxiv.org/abs/2410.03399)
- **What's New**: 이 논문에서는 이벤트 시퀀스(event sequences)에 대한 표준화된 벤치마크인 EBES를 도입합니다. EBES는 데이터셋, 모델, 실험 프로토콜에 대한 통합된 인터페이스를 제공하여 향후 연구를 촉진합니다.

- **Technical Details**: EBES는 회귀(regression)와 분류(classification) 문제를 중심으로 하며, 특정한 평가 시나리오를 포함합니다. 제안된 라이브러리는 새로운 합성 데이터셋과 일반적으로 사용되는 실제 데이터셋을 제공합니다.

- **Performance Highlights**: 이 분석을 통해 향후 연구를 위한 권장 사항들을 제시하고, 데이터셋 사용과 모델 평가와 관련된 잠재적인 위험 요소를 강조합니다.



### GraphCroc: Cross-Correlation Autoencoder for Graph Structural Reconstruction (https://arxiv.org/abs/2410.03396)
Comments:
          22 pages, 16 figures. Accepted in NeurIPS 2024

- **What's New**: 이 논문은 Graph Autoencoders (GAE) 모델의 한계를 극복하고 다중 그래프 상황에 적합한 새로운 Cross-Correlation 메커니즘을 도입하여 그래프 구조 재구성을 개선하는 GraphCroc 모델을 제안합니다.

- **Technical Details**: 종전의 GAE 모델들은 self-correlation을 사용하여 그래프 구조를 표현해왔으나, 이 연구에서는 Cross-Correlation을 활용하여 보다 정확한 노드 임베딩을 제공합니다. GraphCroc 모델은 U-Net과 유사한 인코딩-디코딩 절차를 사용하며, 비대칭 연결 문제를 해결하기 위해 손실 균형 전략을 적용합니다.

- **Performance Highlights**: 이론적 분석과 수치 평가 모두에서 GraphCroc은 기존 self-correlation 기반 GAE 모델 대비 그래프 구조 재구성에서 유의미한 성과를 내어, 특히 다중 그래프에서 우수한 성능을 발휘함이 입증되었습니다.



### From Epilepsy Seizures Classification to Detection: A Deep Learning-based Approach for Raw EEG Signals (https://arxiv.org/abs/2410.03385)
Comments:
          25 pages, 7 tables, 4 figures

- **What's New**: 이번 연구에서는 무작위 간질 발작(Seizure) 감지를 위한 새로운 딥 러닝 모델 기반의 파이프라인을 소개했습니다. 이 파이프라인은 EEG(전기뇌파) 신호를 처리하는 새로운 방법을 포함하고 있습니다.

- **Technical Details**: 새로운 전처리 기술이 연속적인 EEG 신호를 구분 없이 세분화하며, 후처리 알고리즘이 EEG 세그먼트를 재조립해 발작의 시작 및 종료를 식별할 수 있도록 합니다. 또한, 예측된 레이블과 실제 레이블 간의 엄격한 비교에 기반한 새로운 평가 절차도 포함되어 있습니다. 데이터 분할 전략을 사용하여 모델 훈련을 수행하였습니다.

- **Performance Highlights**: 모델은 동물 EEG로 훈련되고 인간 EEG에서 테스트되며, 균형 잡힌 Bonn 데이터셋에서 93%의 F1-score를 기록했습니다. 이 연구는 발작 감지와 분류 작업 간의 기본적인 차이를 입증하였습니다.



### Predicting perturbation targets with causal differential networks (https://arxiv.org/abs/2410.03380)
- **What's New**: 이 연구에서는 생물학적 데이터에서 개입 목표를 예측하기 위한 새로운 접근법인 Causal Differential Networks (Cdn)를 제안합니다. 이 방법은 관찰적 데이터와 개입 데이터를 별도로 처리하여 인과 그래프를 추론한 후, 이를 통해 노출된 변수들을 찾아냅니다.

- **Technical Details**: Cdn은 관찰적 데이터와 개입 데이터에서 각각 인과 그래프를 추론하기 위해 사전 훈련된 causal discovery 모듈을 활용하고, 그런 다음 큰 축 주의 기반 분류기를 사용하여 정답 개입 목표에 대해 지도 학습(un supervised learning) 방식으로 학습합니다.

- **Performance Highlights**: Cdn은 기존 방법들과 비교하여 일관되게 더 나은 성능을 보여주며, 특히 7개의 단일 세포 전사체 데이터 세트에서 perturbation modeling에서의 성능을 입증했습니다. 또한 다양한 합성 데이터 세트에서 개입 목표 예측에 있어서 6개의 기존 인과 발견 알고리즘보다 우수한 성능을 기록하였습니다.



### Mitigating Adversarial Perturbations for Deep Reinforcement Learning via Vector Quantization (https://arxiv.org/abs/2410.03376)
Comments:
          8 pages, IROS 2024 (Code: this https URL)

- **What's New**: 이 논문은 강화 학습 (Reinforcement Learning, RL) 에이전트의 적대적 공격에 대한 방어를 개선하기 위해 벡터 양자화 (Vector Quantization, VQ) 변환 기법을 제안합니다. 이는 기존의 훈련 기반 방법 대신 입력 변환 방어 기법을 통해 시행되며, RL 에이전트의 입력에 대한 공격의 범위를 줄이는 데 도움을 줍니다.

- **Technical Details**: 논문에서 제안하는 VQ 기반 방어 기법은 RL 에이전트의 입력 관측값을 변환하여 적대적 관측값의 영향을 최소화합니다. 이 방법은 컴퓨팅 효율이 높으며, 기존의 RL 알고리즘과 잘 통합될 수 있습니다. VQ를 활용하여 관측 공간을 이산화(discretization)하고, 그 변환된 공간 내에서 RL 에이전트를 훈련시키는 방식으로 작동합니다.

- **Performance Highlights**: 다양한 환경에서의 실험을 통해, 제안된 VQ 입력 변환은 공격에 대한 방어력을 높이는 데 효과적임을 입증하였습니다. 이 연구는 RL 에이전트가 적대적 공격에 대해 더 강건하도록 만들고, 기존의 강건화를 위한 훈련 방법과 상호 보완적으로 작용할 수 있는 가능성을 보여줍니다.



### Make Interval Bound Propagation great again (https://arxiv.org/abs/2410.03373)
- **What's New**: 본 논문은 Neural Network Certification(NNC) 분야에서의 기술적 진전을 다루고 있으며, Interval Bound Propagation(IBP) 방식의 최적성이 떨어짐을 보여줍니다. 또한, 두 가지 새로운 방법(Doubleton Arithmetic와 Affine Arithmetic)을 신경망의 안정성 보증을 위해 적용한 것이 주요한 새롭게 제안된 내용입니다.

- **Technical Details**: 네트워크의 안전성을 평가하기 위한 기존의 IBP 방법은 wrapping 효과(wrapping effect)의 영향을 받아 비효율적임을 설명합니다. 본 연구에서는 이를 극복하기 위하여 Doubleton Arithmetic(DA)와 Affine Arithmetic(AA)를 도입했습니다. 이 두 방법은 비선형 활성함수인 ReLU(ReLU)와 softmax를 사용하는 신경망의 층을 정확하게 평가할 수 있는 알고리즘을 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, Affine Arithmetic는 IBP 대비 월등히 좋은 성능을 보였으며, 신경망의 출력에 대한 경계를 제공하는 데 있어 가장 우수한 성능을 보였습니다. 특히, AA는 대규모 네트워크에서 DA보다 수십 배 더 빠른 계산 속도를 자랑합니다.



### Latent Abstractions in Generative Diffusion Models (https://arxiv.org/abs/2410.03368)
- **What's New**: 이번 연구에서는 확산 기반의 생성 모델(difusion-based generative models)이 어떻게 고차원 데이터(고차원 데이터)인 이미지를 생성하는지를 다룹니다. 생성 과정에서 저차원 잠재 추상(latent abstractions)의 연결을 통해 진행되는 방법론을 제시합니다.

- **Technical Details**: 새로운 이론적 프레임워크를 통해 NLF(Nonlinear Filtering)를 확장하여 SDE(확률적 미분 방정식)에 기반한 생성 모델에 대한 독특한 관점을 제공합니다. 측정 과정에 대한 정보 이론적 영향을 고려하여 비가시적 잠재 추상이 가시적 측정 과정을 이끄는 비선형 필터로서의 SDE 시스템을 설명합니다.

- **Performance Highlights**: 실험 연구를 통해 생성 과정의 다양한 단계에서의 잠재 추상의 출현에 대한 이전의 경험적 결과를 검증했습니다. 특히, 확산 기반의 생성 모델이 잠재 추상화를 어떻게 구성하고 활용하는지를 명시적으로 조사하며, 측정 과정과의 밀접한 관계를 보였습니다.



### Dolphin: A Programmable Framework for Scalable Neurosymbolic Learning (https://arxiv.org/abs/2410.03348)
- **What's New**: 본 논문에서는 Dolphin이라는 새로운 neurosymbolic (신경 구조적) 학습 프레임워크를 제안합니다. 이 프레임워크는 효율적인 벡터화된 계산을 통해 symbolic reasoning (상징적 추론)을 심층 학습 모델에 통합할 수 있도록 합니다.

- **Technical Details**: Dolphin은 PyTorch와 통합되어 있어 사용자가 친숙한 Python에서 neurosymbolic 프로그램을 쉽게 작성할 수 있습니다. 또한, forward chaining (전방 체이닝)과 backward gradient propagation (역전파 기울기 전파)을 벡터화된 계산으로 매핑하여 스케일 문제를 해결합니다. 이를 위해 Dolphin은 symbolic gradients (상징적 기울기)을 효율적으로 계산할 수 있는 vectorized provenance semirings (벡터화된 출처 반링)을 개발했습니다.

- **Performance Highlights**: Dolphin은 13개 벤치마크에 걸쳐 5개의 neurosymbolic 작업을 평가하였으며, 기존 베이스라인 모델들과 비교했을 때 평균 2.77%의 시간에 학습이 완료되었습니다. Dolphin으로 작성된 모델은 최고 성능의 정확도를 달성하며, 복잡한 기준에도 잘 확장됩니다.



### Influence-oriented Personalized Federated Learning (https://arxiv.org/abs/2410.03315)
- **What's New**: 본 연구는 기존의 federated learning (FL) 방법의 한계를 극복하기 위해 클라이언트 수준(client-level) 및 클래스 수준(class-level) 영향력(influence)를 양적으로 측정하여 각 클라이언트의 파라미터 집합을 적응적으로 조정할 수 있는 새로운 프레임워크인 FedC^2I를 제안합니다. 이 접근법은 FL 시스템 내의 클라이언트 간 상호 영향을 명확하게 모델링합니다.

- **Technical Details**: FedC^2I 프레임워크는 영향 벡터(influence vector)와 영향 행렬(influence matrix)을 사용하여 클라이언트 및 클래스 간의 영향을 모델링합니다. 영향 벡터는 각 클라이언트의 영향력을 정량화하여 다른 클라이언트로부터 지식을 선택적으로 획득하도록 돕고, 영향 행렬은 보다 세밀한 방식으로 클래스 수준의 영향을 캡처하여 개별화된 분류기 집합을 이루도록 합니다.

- **Performance Highlights**: 비독립적-동일 분포(non-IID) 환경에서 FedC^2I의 성능을 기존의 FL 방법들과 비교하여 우수성을 증명하였으며, 이 프레임워크의 주요 구성 요소의 효과적 기여를 검증하였습니다.



### Selective Test-Time Adaptation for Unsupervised Anomaly Detection using Neural Implicit Representations (https://arxiv.org/abs/2410.03306)
Comments:
          Accepted at MICCAIw ADSMI

- **What's New**: 이 논문에서는 의료 이미징의 이상 탐지(anomaly detection) 분야에서 적용된 새로운 선택적 테스트 시간 적응(selective test-time adaptation)의 개념을 소개합니다. 이는 기존 모델을 적응시키는데 있어 도메인 변화(domain shifts)에서의 효율성을 향상시키는 방안으로, 병리적 변화가 포함된 데이터에서도 효과적으로 작동합니다.

- **Technical Details**: 우리는 경량의 다층 퍼셉트론(multi-layer perceptron)을 사용하여, 사전 학습된 딥 피쳐(deep pre-trained features)의 특성을 활용하여 테스트 이미지에 대해 제로샷(zero-shot) 방식으로 선택적으로 적응합니다. 이 방법은 다양한 복원 기반 이상 탐지(reconstruction-based AD) 방식의 출력을 조정할 수 있도록 하여, 원래 훈련된 모델에 변화를 주지 않고도 다양한 데이터 세트에서 즉각적인 적응을 가능하게 합니다.

- **Performance Highlights**: 브레인 AD에 대한 엄격한 검증 결과, 우리의 선택적 테스트 시간 적응 프레임워크(STA-AD)가 여러 조건과 다양한 목표 분포(target distributions)에서 탐지 정확도를 크게 향상시켰습니다. 특히, 발달한 뇌실(enlarged ventricles)의 탐지율은 최대 78%, 부종(edemas)은 24% 증가하였습니다.



### SELU: Self-Learning Embodied MLLMs in Unknown Environments (https://arxiv.org/abs/2410.03303)
- **What's New**: 최근 다중모달 대형 언어 모델(MLLM)들이 강력한 시각적 이해 및 의사결정 능력을 보여주며, 미지의 환경에서 자율적으로 개선할 수 있는 가능성이 열리고 있습니다. 하지만 외부 피드백(예: 인간이나 환경 피드백)이 항상 존재하지 않는다는 문제가 있습니다. 이 논문에서는 씽크-크리틱(self-learning) 패러다임인 SELU를 제안합니다.

- **Technical Details**: SELU는 강화 학습의 행동자-비평가(actor-critic) 패러다임에서 영감을 받아, MLLM이 미지의 환경에서 스스로 학습할 수 있는 새로운 방법론입니다. 비평가는 자기 질문(self-asking) 및 회상 레이블링(hindsight relabeling)을 통해 행동자가 수집한 상호작용 경로에서 지식을 추출하여 환경 이해를 높이고, 행동자는 비평가의 자기 피드백을 통해 의사결정을 개선합니다.

- **Performance Highlights**: AI2-THOR 및 VirtualHome 환경에서 SELU 방법의 평가 결과, 비평가는 약 28% 및 30%의 개선을 보였고, 행동자는 약 20% 및 24%의 성능 향상을 달성했습니다.



### Resource-aware Mixed-precision Quantization for Enhancing Deployability of Transformers for Time-series Forecasting on Embedded FPGAs (https://arxiv.org/abs/2410.03294)
Comments:
          Accepted by the 21st EAI International Conference on Mobile and Ubiquitous Systems: Computing, Networking and Services (MobiQuitous2024). 20 pages, 8 figures, 6 tables

- **What's New**: 이 연구는 자원이 제한된 임베디드 FPGA(Xilinx Spartan-7 XC7S15)에서 정수 기반의 양자화된 Transformers의 배포 과제를 다루고 있습니다. 모델 층 간 intermediate 결과를 저장하기 위한 선택 가능한 리소스 유형을 도입하여 VHDL 템플릿의 유연성을 높였습니다. 이로 인해 BRAM을 효율적으로 활용하여 배포 병목 현상을 극복했습니다.

- **Technical Details**: 우리는 mixed-precision quantization 접근 방식을 개발했으며, 자원 인지(resource-aware) 기능을 포함하여 하드웨어 수명과 관련된 다양한 양자화 전략을 탐색할 수 있습니다. 이 방법은 실제 배포 지표와 비교하여 3%의 정밀도 불일치를 유지하며, 각 모델 구성 요소의 자원 사용량을 예측할 수 있는 데이터베이스를 구축하였습니다.

- **Performance Highlights**: 우리의 접근 방식은 랜덤 검색을 통한 15개 후보에서 15개 배포 가능한 후보를 발견했으며, 이전의 Baseline 방법에서는 5개만 발견했습니다. 우리 후보들이 Baseline 2보다 RMSE에서 더 나은 성능을 보임으로써, 본 방법의 안정성과 확장성을 입증하였습니다.



### Demystifying the Token Dynamics of Deep Selective State Space Models (https://arxiv.org/abs/2410.03292)
- **What's New**: 본 논문은 사전 훈련된 Mamba 모델의 동적 속성을 조사하고, Mamba 모델의 연속 시간 제한을 governing하는 동적 시스템을 유도하여 그 해의 점근적 행동을 특성화합니다.

- **Technical Details**: 연구에서, Mamba 모델의 경우 특별한 선택적 상태 공간 층(S6)을 기반으로 하며, 이 층의 매개변수는 입력의 함수로 설정되어 있습니다. 이를 통해 모델은 내용 인식(content awareness) 기능을 갖추고 있습니다. 논문에서는 두 가지 시나리오: 모든 토큰이 0으로 수렴하거나 모든 토큰이 무한대(∞)로 발산하는 경우를 규명하고, 각 경우가 발생하는 모델 파라미터에 대한 기준을 제시합니다.

- **Performance Highlights**: 모델의 수렴 시나리오가 성능에 부정적 영향을 미친다는 것을 실험적으로 검증하였으며, 발산 시나리오에서는 다양한 토큰들이 서로 다른 비율로 무한대로 발산하여 모델 훈련중 불균형한 기여를 함을 증명했습니다. 이러한 조사 결과를 바탕으로, 모델의 효율성을 높이기 위해 수렴 시나리오 제외 및 중요도 점수에 따른 토큰 재정렬을 제안합니다.



### Enhanced Transformer architecture for in-context learning of dynamical systems (https://arxiv.org/abs/2410.03291)
- **What's New**: 이 논문은 in-context identification 패러다임을 통해 다이나믹 시스템의 메타 모델을 오프라인에서 합성 데이터 기반으로 추정하는 방법을 제안합니다. 새로운 메타 모델링 프레임워크를 통해 확률론적 접근 방식과 비연속적인 컨텍스트 및 쿼리 윈도우를 관리하는 방법을 도입하고 있으며, RNN을 이용한 반복 패칭을 통해 긴 컨텍스트 시퀀스를 효과적으로 처리할 수 있도록 개선되었습니다.

- **Technical Details**: 메타 모델은 확률적 설정에서 학습 문제를 형성하며, 추정된 쿼리 출력의 조건부 분포를 사용하여 예측 불확실성에 대한 정보를 제공합니다. 비연속적인 컨텍스트와 쿼리 윈도우를 허용하여 초기 조건 정보도 메타 모델에 제공합니다. 이 메타 모델은 기본적으로 encoder-decoder Transformer 아키텍처로 구성되어 있으며, 여기서 RNN을 패칭 네트워크로 사용하여 메모리 제한을 해결합니다.

- **Performance Highlights**: 본 논문의 변경 사항은 Wiener-Hammerstein 시스템 클래스를 대상으로 한 수치 예제를 통해 모델의 성능 및 확장성을 향상시킨 것으로 입증되었습니다. 이러한 혁신적인 접근 방식은 메타 모델링 프레임워크의 기존 한계들을 극복하고 SYSID 문제에 적합한 구성을 제공합니다.



### uniINF: Best-of-Both-Worlds Algorithm for Parameter-Free Heavy-Tailed MABs (https://arxiv.org/abs/2410.03284)
- **What's New**: 이번 논문에서는 Heavy-Tailed Multi-Armed Bandits (HTMAB) 문제를 위한 새로운 알고리즘 uniINF를 제안하며, 이는 확률적 (stochastic) 및 적대적 (adversarial) 환경 모두에서 강건성 (robustness)과 적응성 (adaptability)을 보여줍니다. uniINF는 Best-of-Both-Worlds (BoBW) 속성을 구현하여 환경의 정확한 유형을 알지 못하더라도 두 환경에서 최적 성능을 보입니다.

- **Technical Details**: uniINF는 파라미터가 사전 지식 없이도 작동하며, heavy-tail 파라미터 (σ, α)를 사전에 알 필요가 없습니다. 이는 확률적 및 적대적 환경 모두에서 거의 최적의 후회 (regret)를 보장하며, (σ, α)이 알려진 경우의 하한 (lower bound)과 일치합니다. 알고리즘 설계에는 refined log-barrier의 분석, 자동 균형 조정 (auto-balancing) 학습률 (learning rate) 스케줄링 기법, 및 adaptive skipping-clipping 손실 조정 기법이 포함됩니다.

- **Performance Highlights**: uniINF는 적대적 환경에서의 성능이 Bubeck et al. (2013)에서 제시된 보편적인 인스턴스 독립적 하한과 거의 일치하며, 확률적 환경에서 인스턴스 의존적 하한을 달성합니다. 이러한 특성 덕분에 uniINF는 다양한 예측 불가능하고 비이상적인 조건에서의 bandit 알고리즘의 강건성과 적용 가능성을 크게 향상시킵니다.



### Neural Sampling from Boltzmann Densities: Fisher-Rao Curves in the Wasserstein Geometry (https://arxiv.org/abs/2410.03282)
- **What's New**: 이 논문에서는 비정규화된 Boltzmann 밀도 $ho_D$에서 샘플링하는 작업을 다루며, 간단한 밀도 $ho_Z$에서 시작하여 에너지 $f_t$의 Boltzmann 곡선을 학습하는 방법을 제안합니다. 특히, Fisher-Rao 흐름과 Wasserstein 기하학에서의 절대 연속성 조건을 조사하며, 선형 보간(interpolation) 방식의 한계를 지적합니다.

- **Technical Details**: 우리는 비정규화된 Boltzmann 밀도 $ho_D=e^{-f_D}/Z_D$에서 샘플링하는 문제를 다룬다. 이를 위해 에너지 함수 $f_t$에 대한 선형 보간 방식이 가질 수 있는 문제점, 즉 '질량의 텔레포트'(teleportation-of-mass) 현상에 대해 설명합니다. Wasserstein 기하학의 도구를 활용하여, 변속도 필드 $v_t$의 폭발적 증가를 효과적으로 측정할 수 있는 해석적 예를 제공합니다.

- **Performance Highlights**: 숫자적 예제를 통해 제안된 모델이 잘 동작하는 흐름 필드를 제공하며, 실제로 위에서 언급한 샘플링 작업을 성공적으로 해결함을 보여줍니다. 또한, Máté 및 Fleuret의 방법에서 파생된 새로운 보간 방식이 기존 방법보다 더욱 바람직한 결과를 도출하는 점을 강조합니다.



### BN-SCAFFOLD: controlling the drift of Batch Normalization statistics in Federated Learning (https://arxiv.org/abs/2410.03281)
- **What's New**: 최근 발표된 BN-SCAFFOLD 알고리즘은 Federated Learning (FL) 환경에서 Batch Normalization (BN)의 성능 저하 문제를 해결하기 위해 개발되었습니다. 이 알고리즘은 SCAFFOLD의 클라이언트 드리프트 수정 기능을 BN 통계에 확장하여 효율적인 DNN 훈련을 가능하게 합니다.

- **Technical Details**: 이 연구에서는 BN-DNN 설정에서의 분산 감소(variance reduction) 알고리즘의 수렴(convergence)을 분석하기 위한 통일된 이론적 프레임워크를 소개합니다. SCAFFOLD 알고리즘이 BN에 의해 도입된 편향(bias)을 제거하지 못한다는 점을 발견하였고, 이를 해결하기 위해 BN-SCAFFOLD를 제안했습니다. 실험은 MNIST와 CIFAR-10 데이터셋을 사용하여 이론적 결과를 검증했습니다.

- **Performance Highlights**: BN-SCAFFOLD는 FedTAN과 동등한 성능을 보이면서도 높은 통신 비용을 요구하지 않으며, Federated Averaging (FedAvg), SCAFFOLD 및 BN 이질성을 완화하기 위해 설계된 다른 FL 알고리즘보다 더 우수한 성능을 나타냅니다.



### Test-time Adaptation for Regression by Subspace Alignmen (https://arxiv.org/abs/2410.03263)
- **What's New**: 이 논문은 회귀(regression) 모델의 테스트 시간 적응(Test-Time Adaptation, TTA)에 대해 조사합니다. 기존의 TTA 방법이 분류(classification)를 위한 설계에 국한되어 있다는 문제를 다룹니다. 기존 방법들은 클래스-카테고리(class-categorical) 예측을 전제로 하였으나, 회귀 모델은 단일 스칼라 값만을 출력합니다. 이를 해결하기 위해, 소스(source)와 타겟(target) 도메인 간의 특징 분포(feature distributions)를 정렬하는 접근법을 제안합니다.

- **Technical Details**: 제안된 방법은 Significant-subspace Alignment (SSA)로, 두 개의 구성 요소인 서브스페이스 탐지(subspace detection)와 차원 가중화(dimension weighting)로 이루어져 있습니다. 서브스페이스 탐지에서는 PCA(주성분 분석)를 사용하여 출력에 유의미한 특징 벡터가 집중된 서브스페이스(subspace)를 찾습니다. 차원 가중화는 출력에 더 큰 영향을 미치는 서브스페이스 차원의 중요성을 높입니다.

- **Performance Highlights**: 다양한 회귀 작업인 UTKFace, Biwi Kinect, California Housing에 대한 실험 결과, SSA가 기존 분류 기법에 맞춰 설계된 TTA 기반선 모델들보다 뛰어난 성능을 보임을 입증했습니다.



### How much can we forget about Data Contamination? (https://arxiv.org/abs/2410.03249)
- **What's New**: 이번 연구는 최신의 대형 언어 모델(LLMs)의 평가 시 벤치마크 데이터의 누출로 인한 문제를 다루고 있습니다. 연구자들은 벤치마크 데이터의 경미한 오염이 평가에 미치는 영향에 대해 실험적 증거와 이론적 추정을 제시하여 그러한 오염이 항상 부정적인 결과를 초래하지 않음을 밝혔다.

- **Technical Details**: 연구에서는 세 가지 차원에서 벤치마크 과적합(benchmark overfitting)의 크기를 정량화했습니다: 모델의 파라미터 수(1.6B까지), 예시가 본 데이터에서 보는 횟수(144회까지), 훈련 토큰 수(40B까지). Chinchilla 스케일링 법칙을 따랐을 경우, 오염이 적더라도 과적합을 초래할 수 있음을 보였습니다. 그러나 훈련 데이터를 다섯 배 확장할 경우, 과거의 데이터를 잊을 수 있는 수치적 경계를 제시합니다.

- **Performance Highlights**: 이 연구는 많은 대형 언어 모델들이 훈련 초기의 데이터를 잊고 있다는 것을 확인했습니다. 또한, 모델들이 데이터의 반복 노출에 대해 가장 강한 과적합을 보이는 현상을 통해, 새로운 데이터에 대한 노출이 중요하다는 것을 강조했습니다.



### CUDLE: Learning Under Label Scarcity to Detect Cannabis Use in Uncontrolled Environments (https://arxiv.org/abs/2410.03211)
Comments:
          8 pages, 5 figures, 1 table

- **What's New**: CUDLE(캐너비스 사용 탐지와 라벨 효율성)라는 새로운 프레임워크를 소개합니다. 이는 실제 환경에서의 웨어러블 센서 데이터를 활용하여 캐너비스 소비를 자동으로 감지하는 문제를 해결하기 위해 자가 지도 학습(self-supervised learning)을 이용합니다.

- **Technical Details**: CUDLE는 센서에서 발생한 데이터를 바탕으로 캐너비스 소비 순간을 식별하는 대조 학습(contrastive learning) 프레임워크를 활용합니다. 초기에는 데이터 증강(data augmentation)과 함께 자가 지도(pretext task)를 통해 강력한 표현(representation)을 학습한 후, 이후 작업에서는 얕은 분류기(shallow classifier)를 통해 미세 조정(fine-tuning)을 합니다.

- **Performance Highlights**: CUDLE는 20명의 캐너비스 사용자와의 임상 연구를 통해 500시간 이상의 웨어러블 센서 데이터를 수집했습니다. 수집된 데이터를 통해 CUDLE는 73.4%의 정확도를 달성했으며, 이는 감독 방식(supervised approach)의 71.1%보다 높은 결과입니다. 또한, CUDLE는 75% 더 적은 라벨을 사용하면서도 성능이 더 뛰어나고, 적은 피험자 수에서도 최적의 성과를 이루었습니다.



### Tadashi: Enabling AI-Based Automated Code Generation With Guaranteed Correctness (https://arxiv.org/abs/2410.03210)
Comments:
          Submitted to CGO

- **What's New**: 이번 논문에서는 Tadashi라고 불리는 Python 라이브러리를 소개합니다. 이 라이브러리는 ML(기계 학습)을 활용하여 코드 생성을 위한 데이터셋을 운영할 수 있도록 돕고, 변환된 코드의 적법성을 보장하는 기능을 제공합니다.

- **Technical Details**: Tadashi는 polyhedral model을 활용하여 코드 생성 과정에서 루프 변환의 적법성을 검증할 수 있는 기능을 갖추고 있으며, ML 엔지니어가 데이터셋을 효율적으로 관리하고 새로운 최적화 방법을 탐색할 수 있게 합니다. 이 라이브러리는 사용자가 프로그래밍적으로 적법한 변환 패턴을 지정할 수 있도록 설계되었습니다.

- **Performance Highlights**: Tadashi는 초당 32개 이상의 변환 및 적법성 체크를 수행할 수 있으며, 일부 벤치마크에서는 781까지 도달하는 성능을 보여주었습니다. 코드 파싱 및 생성에 드는 오버헤드는 생성된 코드를 컴파일하고 실행하는 것에 비해 미미합니다.



### SPHINX: Structural Prediction using Hypergraph Inference Network (https://arxiv.org/abs/2410.03208)
- **What's New**: 본 연구에서는 Hypergraph의 높은 차원 관계를 모델링하기 위한 새로운 접근법인 SPHINX(Structural Prediction using Hypergraph Inference Network)를 소개합니다. 이 모델은 주어진 노드 레벨 신호만을 사용하여 잠재적인 하이퍼그래프 구조를 비지도 학습 방식으로 유추합니다.

- **Technical Details**: SPHINX 모델은 하이퍼 엣지 발견을 클러스터링 문제로 모델링하며, 노드 집합에 대한 확률 분포를 예측하는 연속적인 소프트 클러스터링 방법을 사용합니다. 또한, k-subset 샘플링을 통해 명시적인 하이퍼그래프 구조를 생성하여 이전 연구에서 나타난 몇 가지 훈련 불안정성을 해결했습니다.

- **Performance Highlights**: 진행된 실험을 통해 SPHINX 모델은 지표 예측 작업에서 잠재적인 하이퍼그래프를 추론할 수 있으며, 이는 해석 가능하고, 실제 연결성과 높은 상관관계를 가지며, 성능을 향상하는 데 기여합니다.



### Learning test generators for cyber-physical systems (https://arxiv.org/abs/2410.03202)
Comments:
          34 pages, 4 figures, 7 tables

- **What's New**: 본 논문에서는 사이버 물리 시스템(Cyber-Physical System, CPS)의 런타임 검증을 위한 새로운 알고리즘인 WOGAN(Wasserstein Generative Adversarial Network)을 소개합니다. 이 알고리즘은 단일 요구 사항에 대해 여러 개의 다양한 반례를 생성할 수 있는 테스트 생성기를 자동으로 만들어줍니다.

- **Technical Details**: WOGAN 알고리즘은 Wasserstein 생성적 적대 신경망(WGAN)을 기반으로 하여 작동하며, 테스트 입력의 유니폼 분포를 목표로 하는 생성적 모델을 훈련합니다. 이 알고리즘은 이전 데이터나 모델이 필요 없으며, 실시간으로 필요한 훈련 데이터를 모두 생성합니다. 생성된 테스트는 각기 다른 입력 조건에서 시스템 실패를 노출시키고, 결함의 근본 원인 분석(root cause analysis)을 지원합니다.

- **Performance Highlights**: 실험 결과 WOGAN 알고리즘에 의해 생성된 테스트는 기존의 최첨단 요구 사항 위배 알고리즘과 같은 효율성을 보이며, 유니폼 랜덤 샘플링에서의 샘플만큼 다양성을 제공하는 것으로 나타났습니다. 즉, WOGAN 알고리즘은 CPS의 런타임 검증을 위한 테스트 생성기를 자동으로 생성할 수 있는 유효한 방법임을 입증하였습니다.



### EXAQ: Exponent Aware Quantization For LLMs Acceleration (https://arxiv.org/abs/2410.03185)
- **What's New**: 본 연구는 LLMs inference에서 주로 소프트맥스 레이어가 성능 저하의 병목 지점이 됨을 발견하였습니다. 소프트맥스 연산의 초기 단계인 지수 계산과 누적을 최적화하는 데 중점을 두고 있습니다.

- **Technical Details**: 소프트맥스 함수의 입력에 대한 최적의 클리핑 값(clipping value)을 결정하는 분석적 접근 방식을 제안하였습니다. 이를 통해 LLMs inference를 위한 4비트 이하의 양자화가 가능합니다.

- **Performance Highlights**: LLaMA1-30B 모델에 대한 검증 결과, 'Physical Interaction: Question Answering' (PIQA) 데이터셋에서 2비트 양자화로 기준 성능을 달성하였고, 누적 단계에서 약 4배의 속도 향상을 이루었습니다. 소프트맥스 연산에서 e^x와 ∑(e^x) 계산을 가속화하여 36.9%의 성능 향상을 달성했습니다.



### Rapid optimization in high dimensional space by deep kernel learning augmented genetic algorithms (https://arxiv.org/abs/2410.03173)
Comments:
          17 pages, 5 figures

- **What's New**: 본 연구는 유전 알고리즘(Genetic Algorithms, GAs)의 생성 능력과 심층 커널 학습(Deep Kernel Learning, DKL)의 효율성을 결합한 새로운 접근 방식을 제안합니다.

- **Technical Details**: 이 새로운 접근 방식은 GAs의 생성적 능력을 활용하여 새로운 후보 구조를 생성하고, DKL 기반 대리 모델(surrogate models)을 사용하여 후보 공간의 행동을 신속하게 평가합니다. 이를 통해 복잡한 고차원 공간을 탐색하는 데 필요한 계산적 부담을 줄일 수 있습니다.

- **Performance Highlights**: 우리는 FerroSIM 모델의 최적화를 통해 이 DKL-GA 프레임워크의 효과성을 입증하였으며, 분자 발견(molecular discovery) 및 배터리 충전 최적화와 같은 다양한 문제에 폭넓게 적용 가능함을 보여주었습니다.



### Autoregressive Moving-average Attention Mechanism for Time Series Forecasting (https://arxiv.org/abs/2410.03159)
- **What's New**: 본 논문에서는 다양한 선형 attention 메커니즘에 적응할 수 있는 Autoregressive (AR) Moving-average (MA) attention 구조를 제안하여 시계열(time series)에서 장기 및 지역적 패턴을 포착하는 능력을 향상시킵니다.

- **Technical Details**: ARMA 구조를 기존의 autoregressive attention 메커니즘에 통합한 ARMA attention 메커니즘을 제안합니다. 이 방법은 계산 비용을 크게 증가시키지 않으면서 성능을 개선하며, MA 출력을 직접 계산하지 않고 간접적인 MA 가중치 생성을 통해 구현됩니다.

- **Performance Highlights**: ARMA 구조를 도입함으로써 다양한 AR attention의 시계열 예측 성능이 일관되게 향상되었습니다. 최첨단 성능을 기록하며, 토큰화(tokenization) 및 훈련 방법이 적절할 경우, 기본 AR Transformer만으로도 기존의 최첨단 결과에 필적하는 성과를 달성했습니다.



### Mathematical Formalism for Memory Compression in Selective State Space Models (https://arxiv.org/abs/2410.03158)
Comments:
          27 Pages

- **What's New**: 이 논문에서는 선택적 상태 공간 모델(selective state space models, SSMs)을 통해 시퀀스 모델링의 메모리 압축 기능을 rigorously(엄격하게) 분석합니다. 이 새로운 접근 방식은 전통적인 RNN과 CNN의 한계를 극복하고, 제어 이론(control theory) 및 동적 시스템(dynamical systems)의 원리를 활용하여 메모리를 효율적으로 관리합니다.

- **Technical Details**: 선택적 SSM은 입력의 관련성에 따라 동적으로 상태를 필터링하고 업데이트하는 selective gating mechanism을 도입하여 긴 시퀀스를 효과적으로 압축합니다. 이 논문에서는 rate-distortion theory 및 information bottleneck method를 활용하여 메모리 효율성과 정보 보존 간의 trade-off를 정량화합니다. Fano의 불평등(Fano's inequality)과 데이터 처리 불평등(data processing inequality)을 사용하여 메모리 압축에 대한 이론적 한계를 제공합니다.

- **Performance Highlights**: 실험적으로, 선택적 SSM은 시계열 예측(time-series forecasting)과 자연어 처리(natural language processing)와 같은 시퀀스 모델링 작업에서 state-of-the-art 성능을 달성하며, 기존 RNN 기반 모델에 비해 메모리와 계산 자원 사용을 줄이고도 향상된 속도 및 효율성을 보여주었습니다.



### MELODI: Exploring Memory Compression for Long Contexts (https://arxiv.org/abs/2410.03156)
- **What's New**: 이번 논문에서는 MELODI라고 불리는 새로운 메모리 아키텍처를 제안합니다. 이 아키텍처는 짧은 문맥(window)을 사용하여 효율적으로 긴 문서를 처리하기 위한 것입니다.

- **Technical Details**: MELODI는 네트워크 레이어와 문맥 윈도우를 통해 계층적 압축 구조를 사용하여 단기 메모리와 장기 메모리를 표현합니다. 단기 메모리는 여러 레이어에 걸쳐 문맥 윈도우를 순환적으로 압축하여 부드러운 전환을 보장합니다. 반면, 장기 메모리는 단일 중간 레이어 내에서 추가 압축을 수행하고 여러 문맥 윈도우에서 정보를 집계하여 정보를 통합합니다.

- **Performance Highlights**: MELODI는 Memorizing Transformer와 비교하여 8배의 메모리 사용량을 줄이면서도 다양한 긴 문맥 데이터셋에서 우수한 성능을 보여줍니다. 예를 들어, 13 레이어 트랜스포머 네트워크를 사용하여 PG-19와 arXiv Math에서 각각 10.44 및 2.11의 perplexity를 달성했습니다.



### Machine Learning for Asymptomatic Ratoon Stunting Disease Detection With Freely Available Satellite Based Multispectral Imaging (https://arxiv.org/abs/2410.03141)
Comments:
          13 pages, 1 figure and 2 tables (main text), 1 figure and 3 tables (appendices). Submitted to "Computers and Electronics in Agriculture"

- **What's New**: 본 연구에서는 자유롭게 이용할 수 있는 위성 기반 원격 감지 데이터를 활용하여 고당수염병(RSD)과 같은 무증상 감염병을 포함한 사탕수수 품종의 질병 탐지에 기계 학습(machines learning) 기술을 적용하였습니다.

- **Technical Details**: 연구는 세 가지 단계로 구성됩니다: 1단계에서는 RSD 데이터 수집, 2단계에서는 Sentinel-2 이미지를 사용한 데이터 전처리, 3단계에서는 기계 학습 개발을 진행합니다. 특히, SVM-RBF(Support Vector Machine with Radial Basis Function Kernel) 알고리즘이 가장 높은 분류 정확도(85.64%~96.55%)를 기록하였습니다.

- **Performance Highlights**: Gradient Boosting 및 Random Forest도 높은 성능을 보였고, 정확도는 83.33%~96.55%로 나타났습니다. RSD 탐지에 도움이 되는 사탕수수 품종과 식생 지수를 포함시키는 것이 중요하다는 점이 확인되었습니다.



### In-context Learning in Presence of Spurious Correlations (https://arxiv.org/abs/2410.03140)
- **What's New**: 대형 언어 모델이 spurious features가 포함된 분류 과제에서 in-context learning (ICL) 능력을 보여주는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 기존의 ICL 학습 방식이 spurious features에 취약하고 단일 작업의 경우 작업 기억화(task memorization)를 초래할 수 있음을 발견했습니다. 새로운 ICL 훈련 접근 방식은 입력 임베딩 차원을 무작위로 섞어주고, spurious features에 맞춰 훈련 데이터를 형성하는 방식을 포함합니다.

- **Performance Highlights**: 제안된 방법은 기존 알고리즘인 1-NN, ERM, GroupDRO보다 우수한 성과를 보이며, 다양한 binary classification 작업에서 unseen tasks에 대한 일반화를 이뤘습니다. 그러나 spurious features가 있는 unseen tasks에서는 일반화 능력이 떨어진다고 밝혔습니다.



### Can LLMs Generate Diverse Molecules? Towards Alignment with Structural Diversity (https://arxiv.org/abs/2410.03138)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLMs)을 통해 약물 발견을 가속화하는 데 있어 중요한 구조적으로 다양한 분자의 생성을 위한 새로운 방법을 제안합니다.

- **Technical Details**: 본 연구는 두 단계의 방법론으로 구성되어 있습니다. 첫 번째 단계는 LLM을 감독하에 미세 조정(suprevised fine-tuning)하여 분자를 연속적으로 생성하도록 하고, 두 번째 단계는 생성된 분자의 구조적 다양성을 극대화하기 위해 강화 학습(reinforcement learning)을 적용합니다.

- **Performance Highlights**: 실험 결과, 제안한 미세 조정 방법이 기존의 디코딩 방식보다 구조적으로 다양한 분자를 더 잘 발견할 수 있게 하며, 다른 대표적인 LLMs와 비교하여 다채로운 분자를 생성하는 성능이 뛰어난 것으로 나타났습니다.



### Remaining Useful Life Prediction: A Study on Multidimensional Industrial Signal Processing and Efficient Transfer Learning Based on Large Language Models (https://arxiv.org/abs/2410.03134)
- **What's New**: 이 논문은 장비의 남은 유용 수명(Remaining Useful Life, RUL) 예측을 개선하기 위해 대규모 언어 모델(large language models, LLMs)을 활용한 혁신적인 회귀 프레임워크(regression framework)를 소개합니다.

- **Technical Details**: 제안된 모델은 코퍼스 데이터(corpus data)에서 사전 학습된 LLM의 모델링 능력을 활용하여 복잡한 시간 의존성(temporal dependencies)을 효과적으로 포착하고 예측 정확도를 향상시킵니다. 이 모델은 모든 부분 집합(subset)에 대해 동일한 슬라이딩 윈도우 길이(sliding window length)와 모든 센서 신호를 사용하여 강력한 일관성과 일반화를 보여줍니다.

- **Performance Highlights**: Turbofan 엔진의 RUL 예측 과제에서 제안된 모델은 FD002 및 FD004 부분 집합에서 최신 기술(state-of-the-art, SOTA) 방법을 초월하며, 다른 부분 집합에서는 거의 SOTA 성과를 달성했습니다. 또한 전이 학습(transfer learning) 실험에서 최소한의 타겟 도메인 데이터로 미세 조정(fine-tuning)했을 때, 전체 타겟 도메인 데이터를 기반으로 훈련된 SOTA 방법보다 더 나은 성능을 보였습니다.



### Spatial-aware decision-making with ring attractors in reinforcement learning systems (https://arxiv.org/abs/2410.03119)
- **What's New**: 이 논문은 강화학습(RL)에서의 행동 선택 과정에 뇌 회로 동역학에서 영감을 받은 수학적 모델인 ring attractors의 통합을 탐구합니다. 이를 통해 행동 공간을 명시적으로 인코딩하고, 신경 활동을 조직화하며, 딥 RL 맥락에서 신경망 전반에 걸쳐 공간 표현을 분산시킬 수 있는 생물학적으로 그럴듯한 메커니즘을 제공합니다.

- **Technical Details**: Ring attractors는 신경 회로 모델로서, 뉴런들이 원형으로 연결되어 저항성이 뛰어난 활성화 패턴을 유지합니다. 이들은 RL 에이전트가 공간 정보를 안정적으로 나타내도록 하여 복잡한 환경에서 보다 정확하고 효율적인 학습을 지원합니다. 이 연구에서는 ring attractors를 외부 모델로 구성하거나 딥 러닝 정책 알고리즘의 일부로 통합하는 방법을 사용했습니다.

- **Performance Highlights**: Atari 100k 벤치마크에서 최신 모델 대비 성능이 53% 향상된 결과를 보여주었으며, 이는 ring attractor 통합 접근법이 행동 선택 및 불확실성 인식 결정 과정에서 두드러진 개선을 가져왔음을 시사합니다.



### LoRC: Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy (https://arxiv.org/abs/2410.03111)
Comments:
          15 pages, 4 figures

- **What's New**: 이 논문에서는 Key-Value (KV) 캐시 효율성을 높이기 위해 기존 Transformer 기반의 대규모 언어 모델 (LLMs)에 직접 적용할 수 있는 저랭크( low-rank) 근사의 새로운 접근 방식을 제안합니다. 이는 모델 재훈련 없이 사용할 수 있으며, KV 캐시의 메모리 소비를 효과적으로 줄이는 데 초점을 맞추고 있습니다.

- **Technical Details**: 제안된 방법은 KV 가중치 행렬의 저랭크 근사를 통해 적용됩니다. 레이어별 민감성을 반영한 점진적인 압축 전략을 도입하여, 깊은 네트워크에서의 오류 전파를 이론적으로 분석하고, 각 레이어의 압축 오류 경계를 도출합니다. 이로 인해, 초기 레이어에서 발생한 오류가 심화된 레이어보다 더 크게 증가하는 경향이 있습니다.

- **Performance Highlights**: 8B, 13B, 70B 파라미터를 가진 LLaMA 모델에서 다양한 작업을 통해 실험했으며, 이 방법이 GPU 메모리 소모를 크게 줄이면서도 성능에는 미미한 영향을 미친다는 것을 입증했습니다.



### A Training-Free Conditional Diffusion Model for Learning Stochastic Dynamical Systems (https://arxiv.org/abs/2410.03108)
- **What's New**: 이 연구는 데이터를 사용하여 알려지지 않은 확률적 미분 방정식(SDEs)을 학습하기 위한 훈련이 필요 없는 조건부 확산 모델을 도입합니다. 이 접근 방식은 확산 모델을 이용하여 확률적 흐름 맵을 근사함으로써 SDE 모델링의 계산 효율성과 정확도 문제를 해결합니다.

- **Technical Details**: 제안된 방법은 조건부 확산 모델에 대한 정확한 스코어 함수(score function)의 분석적 닫힌 형태(closed-form)를 유도하며, 이를 통해 몬테 카를로 방법(Monte Carlo methods)을 사용하여 궤적 데이터(trajectory data)를 통해 효과적으로 추정할 수 있습니다. 이 방법은 신경망 훈련 없이도 스코어 함수를 학습할 수 있습니다.

- **Performance Highlights**: 다양한 SDE 타입에 대한 광범위한 수치 실험을 통해 제안된 방법의 다재다능성과 효과성이 입증되었습니다. 학습된 모델은 알려지지 않은 확률적 시스템의 단기 및 장기 행동 예측에서 상당한 개선을 보여주며, 드리프트(drift) 및 확산 계수(diffusion coefficient)를 추정하는 데 있어 기존 GANs 기반 방법을 초월하는 성능을 보였습니다.



### Horizon-Length Prediction: Advancing Fill-in-the-Middle Capabilities for Code Generation with Lookahead Planning (https://arxiv.org/abs/2410.03103)
- **What's New**: 이 논문은 코드 완성을 위한 Fill-in-the-Middle (FIM) 훈련의 새로운 접근법인 Horizon-Length Prediction (HLP)을 제안합니다. HLP는 모델이 중간 토큰 수를 예측할 수 있도록 가르쳐 코드 생성 문제의 성능을 향상시킵니다.

- **Technical Details**: HLP는 기존의 next-token prediction (NTP) 방식과 상호 보완적으로 작동하며, 각 훈련 단계에서 남은 중간 토큰 수를 예측하도록 모델을 훈련시켜 텍스트의 자연스러운 흐름을 유지합니다. 논문에서는 이 접근법이 기존의 rule-based post-processing 방법과는 달리 dataset-specific한 가정에 의존하지 않음을 강조합니다.

- **Performance Highlights**: HLP를 적용한 모델은 다양한 벤치마크에서 최대 24%의 상대적 성능 향상을 보여주었으며, 이는 파일 및 레포지토리 수준의 코드 추론에도 긍정적인 영향을 미쳤습니다. 또한 HLP는 훈련 시간에 거의 추가적인 비용을 들이지 않습니다.



### Optimization Proxies using Limited Labeled Data and Training Time -- A Semi-Supervised Bayesian Neural Network Approach (https://arxiv.org/abs/2410.03085)
- **What's New**: 본 연구는 제한된 라벨 데이터와 제한된 모델 훈련 시간 내에서 제약 조건 최적화 문제를 해결하기 위해 Bayesian Neural Networks (BNNs)를 활용하는 새로운 학습 방식을 소개합니다. 이 방식은 반지도 학습(semisupervised learning) 방법론을 통해, 레이블이 있는 데이터와 레이블이 없는 데이터를 사용하여 최적화를 진행합니다.

- **Technical Details**: 제안된 방법은 두 가지 단계로 구성됩니다: 감독 학습 단계(레벨이 있는 데이터를 활용하여 비용 최소화)와 비감독 학습 단계(레벨이 없는 데이터를 활용하여 제약 조건의 유효성을 보장). 이 과정에서 Stochastic Variational Inference를 사용하여 대략적인 Bayesian 추론을 수행합니다. 이러한 접근 방식은 에너지 네트워크 운영에서 주요 비볼록 제약 조건 최적화 문제를 해결하는 데 성능이 뛰어난 것으로 나타났습니다.

- **Performance Highlights**: 제안된 반지도 학습 방식은 기존의 BNN 및 딥 신경망(DNN) 아키텍처보다 더 나은 성과를 보여주었으며, 기대 최대 동등성 격차를 10배까지 줄이고 최적성 및 불평등(유효성) 격차를 절반으로 줄였습니다. 또한, 최소한의 계산 비용으로 후방 샘플을 제공할 수 있는 BNN의 장점을 활용하여, 후방 선택(SvP) 방식을 통해 동등성 격차를 10% 이상 추가로 줄일 수 있음을 확인하였습니다.



### MetaOOD: Automatic Selection of OOD Detection Models (https://arxiv.org/abs/2410.03074)
Comments:
          Best paper at 2024 KDD Workshop on Resource-Efficient Learning. Extended version

- **What's New**: 이번 연구에서는 다양한 기저 작업에 대해 자동으로 OOD (Out-of-Distribution) 탐지 모델을 선택하는 MetaOOD라는 첫 번째 제로샷 무감독 프레임워크를 소개합니다. 이 방법은 메타 학습(meta-learning)을 활용하여 레이블이 없는 새로운 데이터셋에 적합한 모델을 효과적으로 선택할 수 있도록 합니다.

- **Technical Details**: MetaOOD는 기존 OOD 탐지 방법의 역사적 성능 데이터를 활용합니다. 언어 모델 기반의 임베딩(embeddings)을 도입하여 데이터셋과 탐지 모델의 OOD 특성을 포착하고, OOD 탐지 모델들 간의 유사성을 정량화합니다. 우리의 방법은 11가지 OOD 탐지 모델 중 24개의 테스트 데이터셋 쌍에 대한 실험에서 검증되었습니다.

- **Performance Highlights**: MetaOOD는 기존 11개 기준선 모델 및 최첨단 무감독 선택 방법과 비교했을 때 성능이 유의미하게 뛰어났으며, 검증된 Wilcoxon 통계 테스트 결과에서 모든 기준선보다 평균 순위에서 우수한 성과를 보였습니다.



### FedMAC: Tackling Partial-Modality Missing in Federated Learning with Cross-Modal Aggregation and Contrastive Regularization (https://arxiv.org/abs/2410.03070)
Comments:
          The 22nd International Symposium on Network Computing and Applications (NCA 2024)

- **What's New**: 이번 연구에서는 Federated Learning (FL)에서 부분 모달리티(missing modalities) 누락 문제를 해결하기 위한 새로운 프레임워크인 FedMAC을 제안합니다. FedMAC은 다양한 클라이언트에서 발생할 수 있는 모달리티의 결여에 효과적으로 대응하며, 기존 방법 대비 성능이 최대 26% 향상되었습니다.

- **Technical Details**: FedMAC은 클라이언트와 서버 간의 정보를 동기화하는 모달리티 보완 임베딩(modality imputation embeddings)을 도입합니다. 또한, 다양한 입력 간의 상관관계를 포착하고 특정 모달리티에 대한 편향을 최소화하기 위한 교차 모달 집계(cross-modal aggregation) 방식을 구현합니다. 이 연구는 모달리티 간의 정보를 중심으로 하는 대조 학습(contrastive learning)을 활용하여 집계 과정에서 관련된 모달리티만을 포함하도록 정규화합니다.

- **Performance Highlights**: 실험 결과, FedMAC은 다양한 클라이언트 구성과 통계적 이질성(statistical heterogeneity)에서 효과성을 입증하였고, 심각한 누락 시나리오에서 기존 방법들과 비교해 최대 26% 성능 향상을 보여주었습니다.



### FedCert: Federated Accuracy Certification (https://arxiv.org/abs/2410.03067)
Comments:
          The 22nd International Symposium on Network Computing and Applications (NCA 2024)

- **What's New**: 이 연구는 Federated Learning (FL) 시스템의 견고성을 평가하기 위해 FedCert라는 방법을 제안합니다. 이 방법은 각 클라이언트의 인증된 정확도와 클래스 분포에 기반하여 글로벌 모델의 인증된 정확도를 추정하는 최초의 접근법입니다.

- **Technical Details**: FedCert는 FL 시스템의 견고성을 평가하기 위해 빈도 기반 가중 평균(VW) 방법의 한계를 극복하고, 각 클라이언트의 인증된 정확도와 클래스 분포를 고려하여 글로벌 모델의 인증된 정확도를 보다 정확하게 평가합니다. 또한, Non-IID 데이터의 영향을 고려하여 클라이언트 그룹화 알고리즘을 도입하여 평가의 신뢰성을 향상시킵니다.

- **Performance Highlights**: CIFAR-10 및 CIFAR-100 데이터셋에 대한 다양한 실험 결과에서, FedCert는 기존의 기준 방법들과 비교하여 지속적으로 추정 오차를 감소시키는 것으로 나타났습니다. 이는 FL 시스템의 견고성 및 신뢰성을 평가할 수 있는 가능성을 보여줍니다.



### Compute Or Load KV Cache? Why Not Both? (https://arxiv.org/abs/2410.03065)
- **What's New**: 본 논문은 KV 캐시 로더 시스템인 Cake를 제안하여 긴 컨텍스트의 LLM 추론에서 지연 시간을 최적화하는 방법을 다루고 있습니다. 이 시스템은 기존의 전통적인 프리픽스 캐싱과 비교하여 현저한 성능 향상을 보여줍니다.

- **Technical Details**: Cake는 양방향 병렬화된 KV 캐시 생성 전략을 사용하여 프리픽스 캐시 위치에서 KV 캐시를 동시적이고 동적으로 로드하고, 로컬 GPU에서 계산을 수행합니다. 이를 통해 계산 및 I/O 대역폭 리소스를 최대한 활용합니다. 이 시스템은 수동 매개변수 조정 없이도 다양한 시스템 상태에 자동으로 적응합니다.

- **Performance Highlights**: Cake는 여러 실험에서 TTFT(Time To First Token)를 평균 36.7% 감소시키며, I/O 전용 방법과 비교했을 때는 평균 60.55%의 감소를 보여줍니다. 또한 시스템에 대한 오버헤드는 최소화하여 긴 컨텍스트 LLM 추론 최적화의 효율적이고 실용적인 솔루션을 제공합니다.



### Towards an Improved Metric for Evaluating Disentangled Representations (https://arxiv.org/abs/2410.03056)
- **What's New**: 이 논문에서는 disentangled representation learning(분리된 표현 학습)이 담당하는 역할에 주목하고, 신뢰할 수 있는 정량적 disentanglement metric(분리 척도)의 개발에 대한 문제를 다룹니다. 새로운 metric인 EDI(Exclusivity Disentanglement Index)를 제안하여 기존 척도의 한계를 극복하고, 더 나은 안정성을 제공합니다.

- **Technical Details**: 기존의 인기 있는 정량적 disentanglement metric들을 분석하며, 이들 각각의 이론적 기반과 특성의 차이를 설명합니다. EDI는 exclusivity(배타성) 개념을 활용하여 factor-code relationship(인자-코드 관계)을 개선하여 ad-hoc(즉흥적) 결정을 최소화하는 새로운 프레임워크를 제시합니다. 이 metric은 Modularity(모듈화), Compactness(압축성), Explicitness(명확성)을 포함하는 다양한 특성을 측정합니다.

- **Performance Highlights**: EDI는 기존의 metrics들과 비교할 때, calibration(보정), non-linearity(비선형성) 및 noise(노이즈) 하에서의 강인성에 있어 우수한 성능을 보이며, 계산 효율성도 유지합니다. 본 연구는 고품질의 오픈 소스 코드베이스를 제공하여, 연구자들이 결과를 재현하고 추가 연구를 진행할 수 있도록 합니다.



### Permissive Information-Flow Analysis for Large Language Models (https://arxiv.org/abs/2410.03055)
Comments:
          16 pages, 11 figures

- **What's New**: 본 논문은 대형 언어 모델(LLMs)의 정보 흐름 라벨 전파를 보다 관대하게 수행하는 새로운 접근 방식을 제안합니다. 이는 데이터의 무결성과 비밀성을 유지하면서도 라벨 크리프(label creep) 현상을 완화하는 데 중점을 둡니다.

- **Technical Details**: 제안하는 접근 방식은 모델 출력 생성에 영향을 미친 샘플의 라벨만을 전파하고, 필요하지 않은 입력의 라벨은 제거하는 것입니다. 이 방법은 (i) prompt-based retrieval augmentation과 (ii) $k$-nearest-neighbors language model에 기반한 두 가지 변형으로 구현되었습니다.

- **Performance Highlights**: 제안된 접근 방식은 LLM 에이전트 환경에서 85% 이상의 경우에 라벨을 개선하는 성과를 보였습니다. 이는 특히 라벨 전파의 실용성을 강조합니다.



### Learning Structured Representations by Embedding Class Hierarchy with Fast Optimal Transpor (https://arxiv.org/abs/2410.03052)
- **What's New**: 이번 연구는 기존의 CPCC (Cophenetic Correlation Coefficient) 프레임워크를 기반으로 클래스를 평가하기 위한 더 효과적인 지표로 EMD (Earth Mover's Distance)를 도입하였습니다. 이로써 다중 모드 클래스 분포를 보다 정확히 반영할 수 있습니다.

- **Technical Details**: EMD는 클래스 간의 쌍별 거리 측정을 통해 계층적 표현을 효과적으로 학습하는 방법으로, 본론에서 제안된 OT-CPCC 알고리즘 군은 EMD와 그 변형들로 구성됩니다. 이 방법은 ℓ2-CPCC와 비교하여 보다 나은 성능을 제공하며, FastFT 알고리즘을 통해 계산 효율성을 더욱 개선하였습니다.

- **Performance Highlights**: OT-CPCC를 적용한 결과, 다양한 실세계 데이터셋 및 작업에서 ℓ2-CPCC에 비해 우수한 성능을 보여주었으며, OT-CPCC의 가장 효율적인 변형은 데이터셋 크기에 대해 선형 시간으로 작동합니다.



### Towards Understanding the Feasibility of Machine Unlearning (https://arxiv.org/abs/2410.03043)
- **What's New**: 본 논문은 머신 언러닝의 최신 동향과 개인 정보 보호 규정에 대응하기 위한 새로운 접근 방식인 Machine Unlearning (MU) 측정 기준을 소개합니다. 특히, 각 훈련 샘플의 언러닝 어려움을 정량화하기 위한 독창적인 메트릭스를 제안하며, 기존 연구들에서는 간과해온 접근입니다.

- **Technical Details**: 우리는 다양한 훈련 샘플에 대한 언러닝의 난이도를 평가하기 위한 여러 가지 휴리스틱(heuristics)과 조건을 제시합니다. 특히, Kernelized Stein Discrepancy (KSD)에 기반한 파라미터화된 커널 함수를 통해 각 모델 및 데이터셋에 최적화된 언러닝 난이도를 평가하는 방법을 제안합니다. 언러닝의 성공 조건을 분석하고 평가하기 위해 데이터 분포와 모델 특성을 함께 고려합니다.

- **Performance Highlights**: 여러 분류 작업과 기존의 머신 언러닝 알고리즘을 통해 제안한 방법의 실용성을 검증하였습니다. 제안한 평가 메트릭스를 통해 언러닝 작업이 불가능한 데이터 포인트를 사전에 파악하여 불필요한 언러닝 작업을 줄일 수 있음을 입증하였습니다.



### FedPeWS: Personalized Warmup via Subnetworks for Enhanced Heterogeneous Federated Learning (https://arxiv.org/abs/2410.03042)
- **What's New**: 이 논문에서는 극단적인 데이터 이질성(data heterogeneity) 문제를 해결하기 위한 새로운 방법인 개인화된 워밍업(Personalized Warmup) 전략인 FedPeWS(Federated Personalized Warmup via Subnetworks)를 제안합니다. 이 방법은 참가자들이 초기 단계에서 자신의 데이터에 맞춘 서브네트워크(subnetwork)를 학습하도록 하여 데이터 충돌을 줄이고, 그 후 일반적인 연합 최적화(federated optimization)로 돌아가는 방식입니다.

- **Technical Details**: FedPeWS 접근법은 각 참가자에게 데이터 분포에 맞춘 개인화된 바이너리 마스크를 사용하도록 하여 이를 통해 초기 데이터 집중 학습을 가능하게 합니다. 이 방법은 실제로 3개의 데이터셋(합성 데이터셋, MNIST와 CIFAR-10 조합, 세 가지 개별 의료 데이터셋)을 통해 실제 환경에서도 효과성을 입증하였습니다.

- **Performance Highlights**: FedPeWS는 기존 연합 최적화 방법과 비교할 때 정확도와 수렴 속도를 현저하게 향상시킴을 실증적으로 증명했습니다. 특히 극단적인 비독립적이고 동질적이지 않은(non-i.i.d.) 데이터 상황에서도 성능 개선이 이루어졌습니다.



### CPFD: Confidence-aware Privileged Feature Distillation for Short Video Classification (https://arxiv.org/abs/2410.03038)
Comments:
          Camera ready for CIKM 2024

- **What's New**: 이 연구는 최첨단 다중 모달(multi-modal) 비디오 분류에서 권장 밀집 특성(Privileged Dense Features)의 활용을 위한 새로운 접근 방식을 제안합니다. 전통적인 PFD(Privileged Feature Distillation) 방법의 한계를 극복하기 위해, 확신 점수(confidence scores)를 활용하여 학생 모델의 성능 변동을 줄이는 방법인 CPFD(Confidence-aware Privileged Feature Distillation)를 도입했습니다.

- **Technical Details**: CPFD는 학생 모델이 교사 모델의 출력을 학습하는 것뿐만 아니라 교사의 확신 수준에서 파생된 추가적인 통찰을 활용하도록 설계되었습니다. 이를 통해 밀집 특성의 이점과 전통적인 증류와 관련된 성능 변동 문제를 해결합니다. CPFD는 X-VLM 모델을 기반으로 하며, DF-X-VLM으로부터의 밀집 특성을 효과적으로 증류합니다.

- **Performance Highlights**: CPFD는 5개의 다양한 작업에서 비디오 분류 F1 점수를 평균 6.76% 향상시켰으며, 기존 PFD 대비 2.31%의 성능 개선을 보였습니다. 또한, 교사 모델인 DF-X-VLM과의 성능 갭을 84.6% 줄였고, 실제 운영 데이터에서의 효과성을 입증해 여러 모델에 배포되었습니다.



### MLP-KAN: Unifying Deep Representation and Function Learning (https://arxiv.org/abs/2410.03027)
- **What's New**: 최근 연구에서 MLP-KAN을 소개하며, 이를 통해 representation learning과 function learning의 통합을 다루고 있습니다. MLP-KAN은 Mixture-of-Experts(MoE) 아키텍처를 기반으로 하여 데이터를 효과적으로 처리할 수 있습니다.

- **Technical Details**: MLP-KAN은 Multi-Layer Perceptrons(MLP)과 Kolmogorov-Arnold Networks(KAN)을 혼합하여 사용합니다. MoE 메커니즘이 동적으로 입력을 적절한 전문가에게 라우팅하여 다양한 작업에서 성능을 극대화하였습니다. 이 모델은 transformer 기반 아키텍처에서 구성되어 있으며, 제시된 표준 데이터셋에서 우수한 결과를 보여주었습니다.

- **Performance Highlights**: MLP-KAN은 이미지 인식, 자연어 처리 등 여러 분야에서 기존 모델들과 비교해 우수한 성능을 보이며, 특히 representation learning에서 높은 정확도와 function learning에서 낮은 RMSE를 기록했습니다.



### Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting (https://arxiv.org/abs/2410.03024)
- **What's New**: 최근 생성 모델의 발전, 특히 diffusion 모델이 시계열 모델링에 새로운 방향을 열고 있으며, 예측 및 합성에서 최첨단 성능을 달성하고 있습니다. 그러나 diffusion 기반 모델이 단순하고 고정된 prior에 의존하는 문제를 해결하기 위해 TSFlow라는 조건부 flow matching 모델을 개발했습니다.

- **Technical Details**: TSFlow는 Gaussian processes, 최적 수송 경로(optimal transport paths), 데이터 의존적 prior distributions를 결합하여 생성 문제를 단순화합니다. 이 모델은 conditionally Gaussian processes를 통합함으로써 prior 분포를 데이터의 시간적 구조와 더 밀접하게 정렬시킵니다.

- **Performance Highlights**: 8개의 실제 데이터셋에 대한 실험 평가 결과, TSFlow의 생성 능력이 입증되었으며, 조건부 및 비조건부 훈련 모델 모두 예측 벤치마크에서 경쟁력 있는 결과를 달성했습니다. 8개 데이터셋 중 6개에서 다른 방법을 초월하는 성과를 나타냈습니다.



### On Logical Extrapolation for Mazes with Recurrent and Implicit Networks (https://arxiv.org/abs/2410.03020)
- **What's New**: 최근 연구에서 회귀 신경망 (RNN)과 암시적 신경망 (INN)과 같은 몇 가지 신경망 아키텍처가 논리적 외삽 (logical extrapolation) 능력을 가진 것으로 제안되었습니다. 하지만 본 논문에서는 이 능력이 덜 견고하다는 것을 보여주며, 문제의 난이도에 따라 전반적인 일반화 능력이 다름을 입증합니다.

- **Technical Details**: 여기서는 미로 해결 문제라는 단일 과제의 맥락에서 RNN과 INN에 대한 기존 결과를 재조명합니다. 특히, deadend_start 변수를 도입하여 난이도를 조절하며, INN은 항상 고정된 점으로 수렴하는 반면, RNN은 사이클과 같은 복잡한 제한 행동을 보이는 경향이 있음을 언급합니다. 이 연구는 PyTorch 기반의 TDA 도구를 제공하며, 이는 다양한 신경망 아키텍처에 대한 외삽을 연구하는 데 활용될 수 있습니다.

- **Performance Highlights**: 결과적으로, 네트워크의 외삽 능력은 난이도의 축에 따라 달라질 수 있으며, 신경망을 외삽에 사용할 때는 더 많은 주의가 필요하다는 것을 알립니다. 이 연구에서 소개된 도구들은 다른 딥러닝 맥락에서도 유용할 수 있다는 점을 논의합니다.



### Learning a Fast Mixing Exogenous Block MDP using a Single Trajectory (https://arxiv.org/abs/2410.03016)
- **What's New**: 본 논문에서는 새로운 목표나 보상 함수에 빠르게 적응할 수 있는 에이전트를 훈련하기 위한 효율적인 비지도 표현 학습을 다룹니다. 특히, Exogenous Block Markov Decision Process (Ex-BMDP) 프레임워크를 활용하여, 단일 경로에서 에이전트의 제어 가능한 동역학을 학습할 수 있는 STEEL 알고리즘을 제안합니다.

- **Technical Details**: STEEL은 비지도 환경에서의 제어 가능한 동역학을 학습하기 위한 최초의 수학적으로 검증된 샘플 효율적인 알고리즘입니다. 이 알고리즘의 샘플 복잡성은 제어 가능한 잠재 공간의 크기와 인코더 함수 클래스의 크기에만 의존하며, 외부 노이즈 요인의 혼합 시간에 따라 선형적으로 증가합니다. STEEL은 funcion approximation 설정에서도 적용됩니다.

- **Performance Highlights**: STEEL 알고리즘은 두 개의 toy 문제에서 실험적으로 평가되었으며, 이 알고리즘의 정확성과 샘플 효율성을 입증하였습니다. 코드 또한 공개되어 있어 연구자들이 쉽게 접근할 수 있습니다.



### MMP: Towards Robust Multi-Modal Learning with Masked Modality Projection (https://arxiv.org/abs/2410.03010)
- **What's New**: 이번 논문에서는 Masked Modality Projection (MMP)라는 새로운 방법을 제안하여, 어떤 형태의 모달리티가 결여되어도 견고하게 작동하는 단일 모델을 훈련할 수 있는 방법을 소개합니다. MMP는 훈련 중에 일부 모달리티를 무작위로 마스킹하여 남아있는 모달리티의 정보를 활용하여 결여된 모달리티의 토큰을 추정합니다.

- **Technical Details**: MMP 방법은 크게 두 가지 단계로 구성됩니다: (1) 모달리티 마스킹: 훈련 중에 무작위로 선택된 모달리티를 마스킹합니다; (2) 모달리티 프로젝션: 남아있는 모달리티를 사용하여 마스킹된 모달리티의 토큰을 예측합니다. 또한, 우리는 프로젝션된 토큰과 실제 토큰을 정렬하기 위해 정렬 손실 (alignment loss)를 사용합니다.

- **Performance Highlights**: MMP를 적용한 모델은 세 가지 작업과 다섯 가지 데이터셋을 포함한 광범위한 실험에서, 기존의 모달리티 결여에 대응하기 위한 방법들보다 성능이 더 우수한 결과를 보였습니다. 특히, MMP는 모든 모달리티가 활용 가능한 상태에서의 성능 유지뿐 아니라, 일부 모달리티가 결여된 상황에서도 상당한 성능 향상을 관찰할 수 있었습니다.



### Formation of Representations in Neural Networks (https://arxiv.org/abs/2410.03006)
Comments:
          preprint

- **What's New**: 이번 논문에서는 Canonical Representation Hypothesis (CRH)를 제안하여 신경망의 은닉층에서의 표현 형성을 설명하는 6개의 정렬 관계를 설정합니다. 이는 신경망이 학습 중에 나타내는 내부 구조를 이해하는 새로운 접근 방식을 제공하며, gradient noise와 regularization 간의 균형이 중요하다는 사실을 강조합니다.

- **Technical Details**: CRH에 따르면, 신경망의 latent representations (R), weights (W), neuron gradients (G)는 서로 정렬되어 훈련됩니다. 이러한 정렬은 신경망이 작업과 무관한 변환에 대해 불변성을 갖는 compact representations를 자연스레 학습하도록 이끕니다. CRH가 깨졌을 경우 R, W, G 간의 reciprocal power-law 관계가 나타나며, 이를 Polynomial Alignment Hypothesis (PAH)라고 부릅니다.

- **Performance Highlights**: CRH와 PAH는 신경망의 다양한 현상(예: neural collapse, neural feature ansatz)을 통합할 수 있는 가능성을 제시합니다. 이는 모델 설계에 있어 더 효율적이고 해석 가능한 접근 방식을 가능하게 하며, 이론적 이해와 실제 응용에 모두 기여할 수 있습니다.



### Towards Universal Certified Robustness with Multi-Norm Training (https://arxiv.org/abs/2410.03000)
- **What's New**: 본 연구에서는 CURE(Certified training for Union RobustnEss)라는 최초의 다중 노름(multi-norm) 인증 훈련 프레임워크를 제안합니다. 기존의 인증 훈련 방법들은 특정 타입의 변동성에만 견딜 수 있도록 모델을 훈련시킬 수 있었으나, CURE는 여러 가지 변동성에 대해 더 나은 유니온 로버스트니스(union robustness)를 달성하도록 설계되었습니다.

- **Technical Details**: CURE 프레임워크에는 새로운 l2 결정론적 인증 훈련 방어(defense) 방법과 여러 다중 노름 인증 훈련 방법이 포함되어 있습니다. 여기서는 IBP(Interval Bound Propagation) 손실을 사용하여 더 강력한 인증 로버스트니스(certified robustness)를 보장합니다. 또한, 자연 훈련(natural training)과의 연결을 통해 다중 노름에 대한 로버스트니스도 향상시킵니다.

- **Performance Highlights**: CURE는 MNIST에서 최대 22.8%, CIFAR-10에서 23.9%, TinyImagenet에서 8.0%의 유니온 로버스트니스 향상을 보여줍니다. 또한, CIFAR-10에서 다양한 보지 못한 기하형 변동성에 대한 일반화 성능이 6.8% 추가 개선되었습니다. 이 결과는 향후 '보편적인 인증 로버스트니스(universal certified robustness)'의 방향으로 나아갈 수 있는 기반을 제공합니다.



### Q-SCALE: Quantum computing-based Sensor Calibration for Advanced Learning and Efficiency (https://arxiv.org/abs/2410.02998)
Comments:
          Accepted at QCE24

- **What's New**: 이번 연구는 Quantum Computing (QC) 및 Machine Learning (ML)을 활용하여 저렴한 광학 미세먼지 센서를 보정하는 고급 방법론을 탐구하고 있습니다. 이러한 방법을 통해 스마트 시티 내 공기 질 모니터링 시스템의 정확성을 향상시키고자 합니다.

- **Technical Details**: 연구에서는 고전적인 Feed-Forward Neural Network (FFNN) 및 Long Short-Term Memory (LSTM) 모델과 Variational Quantum Regressors (VQR) 및 Quantum LSTM (QLSTM) 회로와 같은 양자 모델을 비교합니다. 하이퍼파라미터 최적화 및 교차 검증을 통한 체계적 테스트를 수행하여 양자 모델의 보정 성능을 평가합니다. 특히, Quantum Machine Learning (QML)의 이점도 강조하며, 다양한 데이터 조건에 적응할 수 있는 모델의 개발을 모색합니다.

- **Performance Highlights**: FFNN 모델은 VQR 모델에 비해 테스트 세트에서 더 나은 보정 정확도를 기록했습니다 (L1 손실 함수: 2.92 대 4.81). QLSTM은 LSTM 모델보다 약간의 우위를 보였으며 (테스트 세트 손실: 2.70 대 2.77), 훈련 가능한 가중치 수가 적었습니다 (66 대 482). 이러한 결과는 양자 모델이 공기 질 모니터링의 보정 성능을 향상할 수 있는 가능성을 보여줍니다.



### Finite-Sample Analysis of the Monte Carlo Exploring Starts Algorithm for Reinforcement Learning (https://arxiv.org/abs/2410.02994)
Comments:
          13 pages

- **What's New**: 이번 논문에서는 Monte Carlo Exploring Starts (MCES) 알고리즘의 수정된 버전에 대한 유한 샘플 경계를 개발하였습니다. 이 알고리즘은 확률적 최단 경로 문제를 해결하기 위해 최적 정책을 샘플 수익만으로 학습하는 것을 목표로 합니다.

- **Technical Details**: 우리는 정책 반복 알고리즘의 수렴 속도에 대한 새로운 결과를 증명하였고, 이를 통해 $	ilde{O}(SAK^{3}	ext{log}^{3}rac{1}{	ext{δ}})$개의 샘플링 에피소드 이후에 최적 정책을 반환할 확률이 최소한 $1-	ext{δ}$라는 것을 증명했습니다. 여기서 $S$는 상태의 수, $A$는 행동의 수, $K$는 에피소드 길이의 대리 변수입니다.

- **Performance Highlights**: 이 연구는 MCES 알고리즘을 사용하여 할인되지 않은 에피소드 MDP에서 정확한 최적 정책을 학습하는 유한 샘플 분석을 제공한 첫 번째 연구로, 이는 강화 학습( RL)에서 이론적 보장을 개선하는 데 기여합니다.



### Differentiation and Specialization of Attention Heads via the Refined Local Learning Coefficien (https://arxiv.org/abs/2410.02984)
- **What's New**: 본 논문에서는 싱귤러 학습 이론에 기반한 모델 복잡성 척도인 Local Learning Coefficient (LLC)의 개선된 변형인 정제된 LLC (rLLC)를 소개합니다. 이를 통해 트랜스포머 언어 모델의 교육 과정에서 내부 구조가 어떻게 발전하는지를 분석합니다.

- **Technical Details**: rLLC를 사용하여 두 개 층으로 구성된 어텐션 전용 트랜스포머의 개별 구성 요소를 연구하였으며, 어텐션 헤드의 차별화 및 전문화 과정을 분석합니다. 데이터의 구조가 neural network의 특성에 미치는 영향을 살펴보며, 이 행위에 기반한 rLLC의 패턴을 통해 어텐션 헤드들이 전문화되는 과정을 확인합니다.

- **Performance Highlights**: rLLC는 주목할 만한 성능 지표로, 어텐션 헤드들이 교육 과정 동안 어떻게 기능적 역할을 다르게 하는지를 정량적으로 분석할 수 있는 도구입니다. 또한, 새로운 멀티그램 회로를 식별하여 데이터 구조와 손실 경관의 기하학적 특성과 학습 동역학, 그리고 신경망의 출현하는 계산 구조 간의 상관관계를 establishe 하는 데 기여합니다.



### DecTrain: Deciding When to Train a DNN Onlin (https://arxiv.org/abs/2410.02980)
Comments:
          8 pages

- **What's New**: 본 연구에서는 DecTrain이라는 새로운 알고리즘을 제안하여, 낮은 오버헤드로 자가 지도(self-supervision)를 활용하여 모노큘러 깊이 딥 뉴럴 네트워크(DNN)를 온라인으로 훈련할 시점을 결정합니다.

- **Technical Details**: DecTrain은 각 타임스텝에서 훈련의 비용과 예상 정확도 상승을 비교하여 훈련 여부를 결정합니다. 이 방법은 평균 44%의 시간만 훈련을 수행하면서, 모든 타임스텝에서 온라인 훈련을 수행한 것과 동일한 정확도를 유지합니다.

- **Performance Highlights**: DecTrain을 통해 낮은 추론 비용의 DNN이 높은 추론 비용의 DNN에 비해 97%의 정확도 향상을 달성하며, 추론 비용을 56% 줄일 수 있었습니다. 또한, 더 작은 DNN을 사용했을 때도 89%의 정확도 회복을 달성했습니다.



### An explainable approach to detect case law on housing and eviction issues within the HUDOC databas (https://arxiv.org/abs/2410.02978)
- **What's New**: 이 논문은 유럽 인권 법원(ECtHR)의 사례 법을 분석하는 자동화 모델을 개발하는 데 초점을 맞추고 있으며, 특히 주택과 강제 퇴거와 관련된 사례를 탐색하기 위한 작업을 진행합니다. 연구는 40,000건 이상의 사례로 구성된 HUDOC 데이터베이스를 효율적으로 분석할 필요성을 강조합니다.

- **Technical Details**: 연구는 Adaptive Chordal Distance-based Subspace Learning Vector Quantization (AChorDS-LVQ) 모델을 사용하여 법적 문서 분류기를 구축했습니다. 이 모델은 텍스트 분석과 인용 패턴의 조합을 활용하여 주택 문제와 관련된 사건을 식별합니다. 또한, 이 모델은 결정에 대한 설명을 제공하는 해석 가능성을 강조합니다.

- **Performance Highlights**: 모델은 특히 주택 및 퇴거 문제와 관련된 새로운 사례를 식별하는 데 성공했으며, 다른 고급 접근 방식과 비교할 때 유사한 성능을 보여주었습니다. 연구에서 수집된 데이터셋은 주택 관련 문제와 관련된 사례를 효율적으로 분류하는 데 도움이 되는 머신러닝 접근 방식을 통해 신뢰할 수 있는 결과를 제공합니다.



### Learning Optimal Control and Dynamical Structure of Global Trajectory Search Problems with Diffusion Models (https://arxiv.org/abs/2410.02976)
Comments:
          This paper was presented at the AAS/AIAA Astrodynamics Specialist Conference

- **What's New**: 본 논문에서는 데이터 기반 방법으로 포착할 수 있는 특정 솔루션 구조를 이용하여 우주선 궤적 설계의 글로벌 검색 문제를 탐구합니다.

- **Technical Details**: 우주선 궤적 설계를 위한 hybridal cost function을 연구하며 최소 연료(minimum fuel)와 비행 시간(minimum time-of-flight)의 혼합 목표 함수를 사용하여 해결합니다. 문제는 CR3BP(Circular Restricted Three-Body Problem)에서 에너지 의존적인 안정적인 불변 매니폴드(invariant manifold)에 대한 전이(transfers)가 포함됩니다. 이 과정에서 Amortized Global Search (AmorGS) 프레임워크와 Conditional Variational Autoencoder (CVAE), 최신 Diffusion Models를 활용하며, 이 모델들은 full-dimensional solution distribution을 직접 모델링하는 데 유용합니다.

- **Performance Highlights**: AmorGS 예측 결과는 하이퍼플레인(hyperplane) 구조를 잘 캡처하며, 미지의 최대 허용 추력(maximum allowable thrust)에 대해 빠르게 수렴하고 최적화 솔버의 계산 효율성을 크게 향상시킵니다. Diffusion Models는 구조 식별을 위한 사람의 개입 없이 복잡한 구조를 자동으로 캡처하며, 이로 인해 이전 방법들에 비해 훨씬 빠르게 높은 품질의 초기 추정값을 생성합니다.



### F-Fidelity: A Robust Framework for Faithfulness Evaluation of Explainable AI (https://arxiv.org/abs/2410.02970)
Comments:
          Preprint; 26 pages, 4 figures

- **What's New**: 최근 연구에서 설명 가능한 인공지능 (XAI) 기술이 발전되었습니다. 비록 딥 러닝 모델에서 의미 있는 통찰력을 추출하는 방법이 제시되었지만, 이러한 XAI 방법의 적절한 평가 방법이 여전히 해결되지 않은 문제입니다.

- **Technical Details**:  기존의 평가 방법은 입력 데이터에서 가장 중요한 특징을 제거하거나 변형시켜 출력 예측의 변화를 관찰하는 방식을 사용하였습니다. 그러나 이는 Out-of-Distribution (OOD) 문제를 초래하여, 변형된 샘플이 원래 데이터 분포를 따르지 않게 됩니다. 새로운 방법인 RemOve And Retrain (ROAR)은 이 문제를 해결하는데 도움을 주긴 하지만, 모델의 재훈련이 항상 수렴되지 않을 수 있습니다. 저자들은 Fine-tuned Fidelity (F-Fidelity)라는 새로운 평가 프레임워크를 제안하였고, 이 프레임워크는 i) 설명에 독립적인 미세 조정 전략을 사용하여 정보 유출 문제를 완화하고, ii) OOD 입력 생성을 방지하기 위한 랜덤 마스킹 작업을 포함합니다.

- **Performance Highlights**: F-Fidelity는 여러 데이터 구조(이미지, 시계열, 자연어)에 대해 Controlled Experiments를 수행하여 기존의 평가 메트릭보다 설명자의 정확한 순위를 복원하는 데 유의미한 개선을 보였습니다. 이 방법은 설명자 신뢰도에 대한 평가에서 뛰어난 성능을 입증하며, 게임 이론 및 경험적으로 설명의 크기와 F-Fidelity 메트릭 간의 관계를 분석할 수 있는 근거도 제공합니다.



### AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML (https://arxiv.org/abs/2410.02958)
Comments:
          47 pages, 5 figures

- **What's New**: 이번 논문에서는 AutoML-Agent라는 새로운 다중 에이전트 프레임워크를 제안하고 있습니다. 이 프레임워크는 데이터 검색부터 모델 배포까지 전체 AutoML 파이프라인을 지원합니다.

- **Technical Details**: AutoML-Agent는 사용자의 작업 설명을 기반으로 전문화된 LLM 에이전트 간의 협업을 촉진하며, 배포 준비가 완료된 모델을 제공합니다. 기존 연구와 달리 단일 계획을 세우는 대신 검색 증강 계획(retrieval-augmented planning) 전략을 도입하여 최적의 계획을 탐색합니다. 각 계획은 데이터 전처리(data preprocessing) 및 신경망 설계(neural network design)와 같은 하위 작업(sub-tasks)으로 분해되어, 병렬로 실행되는 전문화된 에이전트에 의해 해결됩니다.

- **Performance Highlights**: 14개의 데이터셋을 활용한 7개의 다운스트림(tasks) 실험에서 AutoML-Agent는 전체 AutoML 프로세스를 자동화하는 데 있어 더 높은 성공률을 보여주었으며, 다양한 도메인에서 좋은 성능을 발휘하는 시스템을 제공합니다.



### LLMCO2: Advancing Accurate Carbon Footprint Prediction for LLM Inferences (https://arxiv.org/abs/2410.02950)
Comments:
          9 pages, 11 figures

- **What's New**: 이 논문에서는 LLM(대규모 언어 모델) 추론의 탄소 발자국을 보다 정확하게 예측하기 위한 새로운 모델인 LLMCO2를 소개합니다. 기존의 예측 방법이 불완전함을 보완하여, 요청 구성 및 하드웨어 설정에 따른 탄소 영향을 신속하고 정확하게 추정할 수 있는 도구를 제공합니다.

- **Technical Details**: LLMCO2는 그래프 신경망(GNN) 기반 모델로서, 각 변환기 층의 커널을 그래프로 표현합니다. 각각의 노드는 커널을 나타내고, 엣지는 데이터 의존성을 캡처합니다. 모델은 전처리(prefill)와 디코딩(decode) 단계의 노드 특성을 별도로 인코딩하며, 각 노드의 Roofline 성능을 하드웨어 특성으로 통합합니다. 또한, 일반적인 요청 구성을 주요 변수로 삼아 데이터 샘플링 알고리즘을 개발하였습니다.

- **Performance Highlights**: LLMCO2는 다양한 추론 요청과 GPU 구성을 사용할 때 기존의 ML 기반 에너지 예측자들보다 탄소 발자국 예측 정확도를 51%-123% 개선하였습니다. 이는 LLM의 사용이 증가함에 따라 환경 영향 평가의 필요성을 더욱 부각시킵니다.



### SymmetricDiffusers: Learning Discrete Diffusion on Finite Symmetric Groups (https://arxiv.org/abs/2410.02942)
- **What's New**: 본 논문에서는 SymmetricDiffusers라는 새로운 이산(discete) 확산(diffusion) 모델을 소개하여, $S_n$에 대한 복잡한 분포 학습을 단순화합니다. 이 모델은 심층 신경망을 사용하여 역확산(reverse diffusion) 과정의 개별 전이(transition)를 학습하는 방식으로 문제를 분해합니다.

- **Technical Details**: 이 연구는 유한 대칭군($S_n$)에서의 확산 모델을 다루며, 리플 셔플(riffle shuffle)을 효과적인 전이로 식별하고, PL(Plackett-Luce) 분포의 일반화된 형태를 제안합니다. 또한 샘플링 및 학습 효율성을 향상시키기 위해 이론적으로 기반한 'Denoising Schedule'을 도입합니다.

- **Performance Highlights**: 모델은 4자리 MNIST 이미지 정렬, 노이즈가 있는 MNIST와 CIFAR-10 데이터셋의 직소 퍼즐 해결, 그리고 여행하는 세일즈맨 문제(TSP) 해결 등 다양한 작업에서 최첨단(state-of-the-art) 성능을 달성하거나 비교 가능한 성능을 보였습니다.



### Comparison of Autoencoder Encodings for ECG Representation in Downstream Prediction Tasks (https://arxiv.org/abs/2410.02937)
- **What's New**: 이 연구에서는 심전도(ECG) 신호의 복잡성을 줄이기 위한 다양한 Variational Autoencoder (VAE) 변형들을 소개하며, 그들을 기존의 방법들과 비교하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: 세 가지 새로운 VAE 변형(Stochastic Autoencoder (SAE), Annealed beta-VAE (Abeta-VAE), Cyclical beta-VAE (Cbeta-VAE))을 활용하여 ECG 데이터를 낮은 차원으로 변환하고, 신호 재구성을 유지하면서 예측 작업의 성능을 향상시키기 위한 방법론을 제안합니다.

- **Performance Highlights**: Abeta-VAE는 신호 재구성에서 평균 절대 오차(MAE)를 15.7 ± 3.2 마이크로볼트로 줄였으며, SAE 인코딩을 ECG 요약 기능과 결합하여 Left Ventricular Ejection Fraction (LVEF) 감소 예측의 AUROC를 0.901로 향상시켜, 최신 CNN 모델의 AUROC 0.910과 거의 비슷한 성능을 보여 주었습니다.



### Towards Layer-Wise Personalized Federated Learning: Adaptive Layer Disentanglement via Conflicting Gradients (https://arxiv.org/abs/2410.02845)
- **What's New**: 본 연구에서는 개인화된 연합 학습(personalized Federated Learning, pFL)에서 고유한 데이터 이질성(heterogeneity)으로 인해 발생하는 기울기 발산(gradient divergence) 문제를 해결하기 위해 새로운 접근법인 FedLAG(Federated Learning with Layer-wise Aggregation via Gradient Analysis)를 제안합니다.

- **Technical Details**: FedLAG는 층별(layer-wise) 기울기(gradient) 충돌(conflict) 개념을 활용하여 다양한 클라이언트(client)에서 기울기가 형성되는 각도에 따라 개인화된 학습 과정을 조정합니다. 특히, 기울기가 예각(acute angle)을 형성할 때는 서로 같은 방향으로 정렬되어 클라이언트 불변(feature) 식별에 중점을 두고, 둔각(obtuse angle)을 형성할 경우 클라이언트 특정(tasks) 작업에 주안점을 두게 됩니다. 기울기 충돌이 발생할 경우, 해당 층은 글로벌 집계(aggregation) 과정에서 제외됩니다.

- **Performance Highlights**: 실험 결과 FedLAG는 다양한 최신 방법(state-of-the-art methods)을 초월하는 성능을 보였으며, 다른 기존 방법들과 쉽게 통합되어 성능을 더욱 향상시킬 수 있게 설계되었습니다.



### Neural DDEs with Learnable Delays for Partially Observed Dynamical Systems (https://arxiv.org/abs/2410.02843)
- **What's New**: 이 논문에서는 부분적으로 관측된 동적 시스템을 모델링하기 위해 Constant Lag Neural Delay Differential Equations (NDDEs)를 도입합니다. 이러한 NDDE는 통계 물리학에서의 Mori-Zwanzig (MZ) 형식을 활용하여 효과적인 모델을 제공합니다.

- **Technical Details**: Constant Lag Neural Delay Differential Equations (NDDEs)는 현재 상태에서 과거의 정보 및 지연을 통합하여 동적 시스템을 모델링하는데 사용할 수 있습니다. NDDE는 일반적인 지연 미분 방정식(DDE)으로, 역사 함수와 지연 함수 등을 포함하여 구성됩니다.

- **Performance Highlights**: NDDE 모델은 합성 데이터 및 실험 데이터를 기반으로 한 평가에서 기존 방법들을 초월하는 성능을 보였습니다. 이러한 결과는 NDDE가 부분적으로 관측된 시스템에 적용할 때 효과적임을 보여줍니다.



### Overcoming Representation Bias in Fairness-Aware data Repair using Optimal Transpor (https://arxiv.org/abs/2410.02840)
- **What's New**: 이번 연구에서는 데이터 분포의 불공정성을 해결하기 위해 Bayesian nonparametric stopping rule을 채택하여 각 특성 라벨 구성 요소를 학습하는 새로운 방법론을 제안합니다. 이 방법은 잘못 학습된 OT 연산자를 개선하고, 아카이브 데이터를 수리하는 데 사용할 수 있는 OT-optimal quantization operators를 도출할 수 있습니다.

- **Technical Details**: 본 연구는 표현 편향을 극복하기 위한 데이터 기반 방법론을 제시합니다. 이를 통해 불균형한 데이터에 대한 강건성을 높이고, 공정성 수리를 가능하게 합니다. 또한 새로운 공정한 분포 목표 정의 및 이를 통해 변환된 데이터의 손상과 공정성을 거래할 수 있는 정량기를 도입합니다.

- **Performance Highlights**: 적극적인 시뮬레이션 및 벤치마크 데이터셋에서 우리의 표현 편향 허용 방안이 우수한 성능을 보임을 입증합니다. 이는 일반화 성능을 높이는데 기여하며, 불균형 데이터의 공정성 수정 작업에서의 효과성을 보여줍니다.



### System 2 reasoning capabilities are nigh (https://arxiv.org/abs/2410.03662)
- **What's New**: 본 연구는 머신러닝 모델의 인간과 유사한 추론 능력을 개발하기 위한 최신 문헌을 검토하며, System 2와 유사한 신경망 모델을 설계하기 위해 필요한 단계들을 설명합니다. 연구진은 지금까지의 발전에도 불구하고 현재의 모델들이 추론을 제대로 수행하지 못하고 있다고 주장하며, 목표 달성을 위한 남은 과제가 매우 적다고 강조합니다.

- **Technical Details**: 이 연구는 심리학에서 제안한 이원적 사고 이론을 바탕으로 하여, 인간의 사고 방식인 System 1과 System 2의 차이를 분석합니다. System 1은 빠르고 자동적이며 감정적이고, System 2는 느리고 의도적이며 논리적입니다. 머신러닝 모델이 System 2와 일치하기 위해서는 패턴 매칭을 넘어서는 증명 과정과 가설 검증이 필요합니다. 또한, 모델을 효과적으로 학습시키기 위해 Actor-Critic 알고리즘과 관련된 개념이 언급됩니다.

- **Performance Highlights**: 딥러닝을 통해 훈련된 복잡한 에이전트들이 실제 세계에 대해 논리적으로 추론할 수 있는 가능성이 가까운 미래에 실현될 것으로 기대됩니다. 이전 연구들은 체인-오브-투사(Chain-of-Thought) 기법을 통해 LLM이 문제 해결 과정을 단계별로 따르도록 유도함으로써 성능 개선 효과를 보여주었으며, 이는 머신러닝 모델의 발전에 중요한 기초 자료로 작용할 것입니다.



### RAFT: Realistic Attacks to Fool Text Detectors (https://arxiv.org/abs/2410.03658)
Comments:
          Accepted by EMNLP 2024

- **What's New**: 이 논문에서는 RAFT라는 새로운 공격 방법을 제안하여 기존의 LLM 감지기를 효과적으로 무력화할 수 있음을 입증합니다. 기존의 공격 방식과는 달리, RAFT는 원본 텍스트의 품질을 유지하면서 단어 수준에서 LLM 임베딩의 전달 가능성을 활용합니다.

- **Technical Details**: RAFT는 보조 LLM 임베딩을 사용하여 대체할 후보 단어를 선택하고, 블랙박스 LLM을 이용해 가장 효과적으로 탐지를 회피할 수 있는 대체 후보를 선정합니다. 이 방법은 텍스트의 품질을 유지하면서 탐지 성능을 99%까지 감소시키는 데 성공했습니다.

- **Performance Highlights**: 실험 결과 RAFT는 여러 도메인에서 감지기를 최대 99%까지 무력화할 수 있었고, 이로 인해 현재의 LLM 감지기가 적대적 공격에 강하지 않다는 것을 보여주었습니다. 또한 이 방법으로 생성된 예제들은 적대적 훈련을 통해 감지기의 강인함을 향상시키는 데 사용될 수 있습니다.



### Learning Humanoid Locomotion over Challenging Terrain (https://arxiv.org/abs/2410.03654)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 다양한 도전적인 자연 및 인공 지형에서 이동할 수 있는 학습 기반의 유인 로봇 보행 제어 방법을 제안합니다. 기존의 방법이 주로 부드러운 지형에 초점을 맞춘 반면, 이 연구는 실제 로봇이 거친 지형을 성공적으로 통과할 수 있는 방법을 소개합니다.

- **Technical Details**: 모델은 프로프리오셉션 관찰 및 행동의 역사 기반으로 다음 행동을 예측하는 transformer 모델을 사용합니다. 학습은 먼저 평탄한 지면의 궤적 데이터셋에서 시퀀스 모델링을 통해 사전 학습(pre-training)된 후, 불규칙한 지형에서 강화 학습(reinforcement learning)을 통해 미세 조정(fine-tuning)됩니다. 이 모델은 Digit 유인 로봇에서 테스트되어 다양한 미지의 지형에서도 효율적으로 작동합니다.

- **Performance Highlights**: 실험에서 제안된 모델은 4마일 이상의 하이킹 경로를 성공적으로 이동하고, 샌프란시스코의 31% 경사도 이상의 매우 가파른 도로도 무사히 통과했습니다. 모델은 각기 다른 지형에서 강력한 성능과 적응력을 보여주었으며, 이러한 성과는 모델의 사전 학습과 미세 조정 덕분으로 해석됩니다.



### Minimax-optimal trust-aware multi-armed bandits (https://arxiv.org/abs/2410.03651)
- **What's New**: 이번 논문은 기존의 multi-armed bandit (MAB) 알고리즘에서 간과되었던 인간의 신뢰(trust)를 고려한 새로운 접근 방식을 제안합니다. 신뢰가 부족할 경우 추천된 정책을 따르지 않는 인간의 특성을 반영하여 신뢰를 동적으로 모델링하고, 이를 통해 추천 성과를 개선하려고 합니다.

- **Technical Details**: 논문에서는 신뢰 기반 다중 무장 도둑 문제(trust-aware MAB problem)를 다루며, 세 가지 주요 구성 요소를 포함합니다: (i) 불확실한 환경과 학습 가능한 인간 신뢰, (ii) 의사결정을 위한 MAB 알고리즘을 설계하는 정책 설정자(policy-maker), (iii) 신뢰에 따라 추천된 행동을 실제로 실행할 수 있는 수행자(implementer). 또한, 제안된 두 단계의 신뢰 인식 프로세스가 강화된 통계적 보증(statistical guarantees)을 갖도록 합니다.

- **Performance Highlights**: 시뮬레이션 연구를 통해 신뢰 문제를 다루는 데 있어 제안한 알고리즘이 기존의 vanilla MAB 알고리즘들, 특히 upper confidence bound (UCB) 알고리즘에 비해 성능 향상 효과가 있음을 입증합니다. 이로 인해 추천된 정책에 대한 신뢰도가 향상되고, 최적의 성과를 달성할 가능성이 커집니다.



### GenSim2: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs (https://arxiv.org/abs/2410.03645)
Comments:
          CoRL 2024. Project website: this https URL

- **What's New**: 이번 연구에서는 GenSim2라는 확장 가능한 로봇 시뮬레이션 프레임워크를 제안합니다. 이 프레임워크는 복잡하고 현실적인 시뮬레이션 작업 생성을 위해 다중 모달 및 추론 능력을 갖춘 코딩 LLM을 활용합니다.

- **Technical Details**: GenSim2는 200개의 객체로 최대 100개의 관절 작업에 대한 데이터를 자동 생성할 수 있는 계획(solvers) 및 강화학습(RL) 솔버들을 사용하여 객체 카테고리 내에서 일반화합니다. 제안된 다중 작업 언어 조건화 정책 아키텍처인 proprioceptive point-cloud transformer (PPT)는 생성된 데모에서 학습하여 시뮬레이션에서 현실로의 제로샷 전이가 뛰어납니다.

- **Performance Highlights**: GenSim2는 생성된 데이터를 사용하여 제로샷 전이 또는 현실 세계에서 수집된 데이터와 공동 학습할 수 있는 가능성을 보여줍니다. 이는 제한된 실제 데이터에서만 훈련했을 때보다 정책 성능을 20% 향상시킬 수 있습니다.



### Conditional Enzyme Generation Using Protein Language Models with Adapters (https://arxiv.org/abs/2410.03634)
- **What's New**: 이 연구에서는 ProCALM(Protein Conditionally Adapted Language Model)이라는 새로운 모델을 제안하여, 조건부로 특정 기능과 세부정보를 갖춘 단백질 생성을 가능하게 합니다. 기존의 단순 토큰 기반 조건화 방법의 한계를 극복하고, 보이는 기능과 세부정보에 국한되지 않는 일반화 능력을 본 모델이 갖춘 것을 돋보입니다.

- **Technical Details**: ProCALM은 프로틴 언어 모델을 위한 어댑터 기술을 활용하여 조건부 생성을 수행합니다. 이 모델은 ProGen2를 미세 조정(finetuning)하여 효소 기능 및 분류학적 정보의 조건 표현을 포함하도록 했습니다. ProCALM은 목표 효소 계열에서의 시퀀스 생성을 위한 기존 방법들과 동등한 성능을 보여주며, 드물고 새로운 효소 계열 및 분류학적 정보에 대한 일반화 능력을 갖추고 있습니다.

- **Performance Highlights**: ProCALM은 학습 비효율성과 높은 계산 비용 없이도 다양한 조건화에 대해 유연하게 반응합니다. 또한, 이는 공통 효소 클래스에서의 단백질 시퀀스 생성을 성공적으로 수행하며, 훈련 세트에 보이지 않는 조건에 대해서도 시퀀스를 생성할 수 있는 능력을 증명했습니다.



### TICKing All the Boxes: Generated Checklists Improve LLM Evaluation and Generation (https://arxiv.org/abs/2410.03608)
- **What's New**: TICK (Targeted Instruct-evaluation with ChecKlists)이라는 새로운 자동화된 평가 프로토콜을 제안하여 LLM의 instruction-following 능력을 해석 가능하고 유연하게 평가할 수 있게 되었습니다. 이 프로토콜은 LLM이 생성한 instruction-specific checklist를 통해 평가를 구조화합니다.

- **Technical Details**: TICK은 주어진 instruction에 대해 LLM이 YES/NO 질문으로 구성된 맞춤형 평가 체크리스트를 신뢰성 높게 생성할 수 있다는 것을 보여줍니다. 이 체크리스트는 각 후보 응답이 instruction의 특정 요구 사항을 충족하는지를 평가합니다. TICK을 사용하면 LLM의 판단과 인간의 선호 간의 정확한 일치 비율이 약 5.8% 증가합니다.

- **Performance Highlights**: 이 연구에서 제안된 STICK(Self-TICK)은 여러 벤치마크에서 자기 개선을 통해 세부 성능 향상을 이루었습니다. LiveBench에서 Command-R+는 3.8% 개선을 달성하고, WildBench에서 Best-of-N 선택 방식으로 5.3%의 성능 향상을 보여주었습니다. 또한 LLM이 생성한 체크리스트를 인간 평가자에게 제공하여 평가자 간의 일치도를 0.194에서 0.256으로 증가시켰습니다.



### Exploring gauge-fixing conditions with gradient-based optimization (https://arxiv.org/abs/2410.03602)
Comments:
          9 pages, 2 figures; Proceedings of the 41st International Symposium on Lattice Field Theory (Lattice 2024)

- **What's New**: 본 연구는 격자 이론( lattice field theory )에서 gauge-fixing( 게이지 고정) 방법을 시스템적으로 탐색하는 새로운 접근 방식을 소개합니다. 새로운 differentiable parameterization( 미분 가능 매개변수화 )이 기존의 Landau gauge, Coulomb gauge 및 최대 트리 게이지를 아우를 수 있도록 설계되었습니다.

- **Technical Details**: 이 연구는 gauge-fixing( 게이지 고정 )을 위한 미분 가능 매개변수를 도입하며, 임의의 loss function( 손실 함수) 최소화에 대한 gradient-based optimization( 경량 기반 최적화 ) 방법을 개발합니다. 이 방법은 gauge-fixed configuration( 게이지 고정 구성) 데이터로부터 손실 함수의 그래디언트를 계산할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 특정 손실 함수의 최소화를 통해 gauge-fixing 방법을 선택하는 데 성공적이며, 두 개의 회귀 문제를 해결하는 데 효과적임을 입증했습니다. 이를 통해 알고리즘의 효율성을 크게 향상시키는 결과를 나타냅니다.



### Understanding Reasoning in Chain-of-Thought from the Hopfieldian View (https://arxiv.org/abs/2410.03595)
Comments:
          28 pages, a new version of "A Hopfieldian View-based Interpretation for Chain-of-Thought Reasoning"

- **What's New**: 이 연구에서는 Chain-of-Thought (CoT) 추론의 기초에 있는 요인을 설명하기 위한 새로운 관점을 제시합니다. Hopfieldian 인지 이론을 바탕으로 CoT 추론과 주요 인지 요소 간의 관계를 확립하여 CoT의 성공 요인을 이해하고자 합니다.

- **Technical Details**: CoT 추론을 이해하기 위해 신경 과학에서의 Hopfieldian 관점을 이용합니다. CoT는 자극, 행동, 신경 집단, 표현 공간과 같은 인지 요소와 맥락을 두고 연결되며, CoT를 통해 발생하는 추론 과정은 이러한 표현 공간 간의 이동으로 설명됩니다. 새로운 프레임워크 'Representation-of-Thought (RoT)'를 제안하여 CoT의 견고성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, RoT는 CoT 추론의 견고성과 해석 가능성을 개선하며, 추론 과정에 대한 세밀한 제어를 제공함을 보여주었습니다. 다양한 과제(산술, 상식, 기호적 추론)에 대한 포괄적인 실험이 실시되었고, CoT 추론의 오류를 추적하고 통제할 수 있는 직관적이고 해석 가능한 분석이 가능함을 발견했습니다.



### Nonstationary Sparse Spectral Permanental Process (https://arxiv.org/abs/2410.03581)
- **What's New**: 기존의 permanental processes는 kernel types 또는 stationarity에 대한 제약을 부과하여 모델의 표현력을 제한했습니다. 본 연구에서는 nonstationary kernels의 sparse spectral representation을 활용하여 이러한 제약을 극복하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: 제안하는 모델인 Nonstationary Sparse Spectral Permanental Process (NSSPP)와 그 심층 변형인 DNSSPP는 해석적 솔루션을 보장하는 표준 커널의 요구사항을 완화하며, 커널의 비정상성을 허용합니다. 모델은 여러 spectral feature mappings을 계층적으로 쌓아 deep kernel을 구성하여 복잡한 데이터를 효과적으로 캡처할 수 있는 표현력을 증가시킵니다.

- **Performance Highlights**: 실험 결과, (D)NSSPP는 데이터가 비정상적일 때 기저 모델보다 우수한 성능을 보여줍니다. 또한, 다양한 하이퍼파라미터의 모델 성능에 대한 영향을 평가하기 위한 ablation studies도 수행하여 모델의 유연성을 입증합니다.



### Towards Linguistically-Aware and Language-Independent Tokenization for Large Language Models (LLMs) (https://arxiv.org/abs/2410.03568)
- **What's New**: 이 논문은 최신 대형 언어 모델(LLMs)의 토큰화(tokenization) 기술에 대한 종합적인 연구를 제시하며, 특히 자원이 부족한 언어에 대한 서비스의 비용과 가용성에 미치는 영향을 분석합니다.

- **Technical Details**: 이번 연구는 GPT-4(cl100k_base 임베딩 사용), GPT-3(p50k_base 임베딩 사용), DaVinci(r50k_base 임베딩 사용) 및 널리 사용되는 BERT base tokenizer와 같은 여러 LLM을 포함합니다. 각 모델 간의 토큰화 가변성을 평가하고 하위 단어 토큰화(subword tokenization)에서의 언어 표현의 문제점을 조사합니다.

- **Performance Highlights**: 이 연구는 특히 전자 건강 기록(EHR) 시스템의 맥락에서 토큰화 선택의 실제적인 영향을 강조하는 사례 연구를 포함하고 있습니다. AI 서비스 발전에 있어 포괄성(inclusivity)을 강조하며 전통적으로 AI 애플리케이션에서 부족한 언어를 지원하기 위한 일반화 가능한 국제화(I18N) 관행을 촉진하고자 합니다.



### Optimizing food taste sensory evaluation through neural network-based taste electroencephalogram channel selection (https://arxiv.org/abs/2410.03559)
Comments:
          33 pages, 13 figures

- **What's New**: 본 논문은 taste 전기뇌파(electroencephalogram, EEG) 데이터에서 채널 선택을 위한 새로운 방법인 class activation mapping with attention (CAM-Attention)을 제안합니다.

- **Technical Details**: CAM-Attention 방법은 convolutional neural network과 channel 및 spatial attention 모델(CNN-CSA)과 gradient-weighted class activation mapping(Grad-CAM) 모델을 결합하여 EEG 데이터의 핵심 기능을 주목하는 방법론을 사용합니다.

- **Performance Highlights**: 이 방법은 taste EEG 인식의 컴퓨팅 부하를 줄이고, 네 가지 맛을 효과적으로 구별하여 뛰어난 인식 성능을 발휘하며, 맛 감각 평가를 위한 기술적 지원을 제공합니다.



### BodyShapeGPT: SMPL Body Shape Manipulation with LLMs (https://arxiv.org/abs/2410.03556)
Comments:
          Accepted to ECCV 2024 Workshop on Foundation Models for 3D Humans. Code repository: this https URL

- **What's New**: 이 연구는 사전 훈련된 Large Language Models (LLMs)을 활용하여 사람의 물리적 특성을 이해하고, 이를 기반으로 SMPL-X 모델을 사용해 정확한 아바타를 생성하는 방법을 제안합니다. 특히 LLMs가 자연어를 통해 3D 인간 형태를 조작할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 본 연구는 LLMs의 미세 조정을 통해 SMPL-X 형태 매개변수를 정확하게 인식하고 생성하는 데 초점을 맞춥니다. 새로운 데이터셋을 생성하여 아바타의 물리적 속성과 관련된 언어적 설명을 포괄적으로 라벨링 했습니다. 모델 학습에 Low-Rank Adaptation (LoRA)와 quantization 기법을 적용하여 NVidia RTX-4090에서 효율적으로 훈련할 수 있도록 최적화했습니다.

- **Performance Highlights**: 본 연구의 결과는 LLMs의 조정된 아키텍처가 다양한 아바타 형상을 성공적으로 생성할 수 있으며, 이는 스토리텔링 및 가상 캐릭터 생성 분야에서의 활용 가능성을 보여줍니다. 최종 아바타는 기대되는 범위 내에서 생성되며, 이는 인간-기계 상호작용을 향상시키는 데 기여할 것입니다.



### Multi-modal Atmospheric Sensing to Augment Wearable IMU-Based Hand Washing Detection (https://arxiv.org/abs/2410.03549)
Comments:
          iWOAR2024

- **What's New**: 본 연구에서는 수용성이 있는 IMU 기반 센서를 포함하여 습도, 온도 및 기압 센서를 추가한 새로운 오픈 소스 프로토타입 장치를 제시합니다. 10명의 참여자와 43회의 손 씻기 이벤트로 구성된 벤치마크 데이터셋을 제공하고, 추가 센서의 유용성을 머신 러닝 모델과 주석 파이프라인에서 평가하였습니다.

- **Technical Details**: 수용성이 있는 IMU 센서와 함께 추가된 대기 센서(습도, 온도, 기압)를 사용하여 손 씻기를 감지하는 새로운 접근 방식을 제안합니다. Puck.js 장치를 기반으로 하며, 블루투스 저전력(BLE)을 통해 데이터를 실시간으로 전송하여 수집하였습니다.

- **Performance Highlights**: 기존 IMU 데이터의 단독 성능보다 향상된 손 씻기 감지를 위한 추가 센서의 기여를 분석했습니다. 특히 습도 센서가 손 씻기 활동 중 상대 습도의 강한 증가를 기록하였고, 머신 러닝 분석을 통해 간섭 패턴을 이용한 구체적인 특성의 식별이 필요함을 보여주었습니다.



### Multidimensional Human Activity Recognition With Large Language Model: A Conceptual Framework (https://arxiv.org/abs/2410.03546)
- **What's New**: 이 논문은 대형 언어 모델(LLM)과 인간 활동 인식(HAR) 시스템의 통합을 통해 긴급 대응 및 노인 돌봄과 같은 고 위험 환경에서의 위험 평가, 자원 배분, 긴급 대응의 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: 논문은 LLM을 HAR 시스템과 통합하는 개념적 프레임워크를 제안하며, 다양한 착용형 센서를 통해 수집된 데이터를 다차원적으로 학습하는 접근 방식을 사용합니다. 이 데이터는 복잡한 센서 입력을 실행 가능한 인사이트로 변환하는 데 활용됩니다. LLM의 적응형 학습 기능은 HAR 시스템의 필요에 맞추어 조정됩니다.

- **Performance Highlights**: 이 통합 접근 방식을 통해 긴급 대응자는 위험 평가를 보다 잘 수행하고, 실시간으로 건강 위험을 예측하며 효과적인 결정 내리기를 가능하게 합니다. 데이터 수집, 처리 및 학습의 네 가지 상호 의존적인 하위 모듈이 이 시스템 아키텍처를 구성하며, 이를 통해 응급 서비스의 반응성과 효과성이 향상됩니다.



### Authentication by Location Tracking in Underwater Acoustic Networks (https://arxiv.org/abs/2410.03511)
Comments:
          Article submitted to IEEE Transaction on Wireless Communications

- **What's New**: 본 논문에서는 수중 음향 네트워크(UWANs)에서 사용하는 물리 계층 메시지 인증 메커니즘을 개선하는 방법을 제안합니다. 제안된 방법은 두 단계로 구성되며, 첫 번째 단계에서는 수중 장치의 위치를 추정하고, 두 번째 단계에서는 이전에 추정한 위치를 바탕으로 미래 위치를 예측합니다.

- **Technical Details**: 위치 추정은 샘플 공분산 행렬(sample covariance matrix)을 입력으로 사용하는 convolutional neural network(CNN)를 통해 수행됩니다. 위치 예측은 Kalman filter 또는 recurrent neural network(RNN)를 사용하여 이루어집니다. 인증 검사는 예측된 위치와 추정된 위치 간의 제곱 오차(squared error)를 비교하는 방식으로 진행됩니다.

- **Performance Highlights**: Kalman filter를 기반으로 한 솔루션이 Gauss-Markov 이동 모델을 따르는 장치 이동에서 RNN 기반 솔루션보다 성능이 우수한 것을 보여줍니다.



### Classification-Denoising Networks (https://arxiv.org/abs/2410.03505)
Comments:
          18 pages, 5 figures

- **What's New**: 이번 논문에서는 이미지 분류(classification)와 노이즈 제거(denoising)를 통합하여 두 작업의 문제를 동시에 해결할 수 있는 새로운 프레임워크를 제시합니다. 이를 통해 노이즈가 있는 이미지와 클래스 레이블의 결합 확률을 모델링하며, 효율성과 견고성을 높일 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 모델은 양측 확률 밀도 log⁡p(𝐲,c)를 매개변수화하는 단일 네트워크로 구성되어 있습니다. 분류 및 조건부 노이즈 제거 작업이 하나의 네트워크에서 수행되며, 손실 함수는 교차 엔트로피(cross-entropy)와 노이즈 제거 점수(match score)를 결합한 형태로 설정됩니다. GradResNet 아키텍처를 통해 ResNet의 특정 수정을 최소한으로 하여 UNet의 유도 편향(inductive biases)을 통합합니다.

- **Performance Highlights**: CIFAR-10과 ImageNet 데이터셋에서 경쟁력 있는 이미지 분류 및 노이즈 제거 성능을 보여주며, 이전의 통합 모델에 비해 효율성이 현저하게 개선되었습니다. 또한, 제안된 모델은 표준 분류기보다 적대적 변형(adversarial perturbations)에 더욱 견고한 성능을 보여주며, 적대적 그래디언트를 새로운 해석으로 제시합니다.



### A Multimodal Framework for Deepfake Detection (https://arxiv.org/abs/2410.03487)
Comments:
          22 pages, 14 figures, Accepted in Journal of Electrical Systems

- **What's New**: 딥페이크(dipfake) 기술의 급속한 발전이 디지털 미디어의 무결성에 중대한 위협을 가하고 있습니다. 본 연구에서는 시각 및 청각 요소를 모두 포괄하는 혁신적인 다중 모달(multimodal) 접근 방식을 통해 딥페이크 문제를 해결하고자 했습니다.

- **Technical Details**: 우리의 모형은 고급 특성 추출 기법을 사용하여 비디오의 아홉 가지 개별적인 얼굴 특징을 추출하고, 다양한 머신러닝(machin learning) 및 딥러닝(deep learning) 모델을 적용했습니다. 오디오 분석을 위해 mel-spectrogram 분석을 활용하여 특징을 추출하고, 동일한 방식으로 머신러닝 및 딥러닝 기법을 적용했습니다. 우리 모형은 비디오 및 오디오 분류를 위해 인공신경망(Artificial Neural Network)과 VGG19를 사용하여 전체 샘플을 딥페이크로 분류합니다.

- **Performance Highlights**: 우리의 다중 모달 프레임워크는 시각 및 청각 분석을 결합하여 94%의 정확도를 달성했습니다.



### VEDIT: Latent Prediction Architecture For Procedural Video Representation Learning (https://arxiv.org/abs/2410.03478)
Comments:
          10 pages

- **What's New**: 본 연구는 비디오 클립 시퀀스를 학습하기 위해 대규모 사전 훈련을 필요로 하지 않는 새로운 접근 방식을 제안합니다. 특히, 동결된(pretrained) 비주얼 인코더와 잘 설계된 예측 모델을 사용하여 최첨단 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 이 방법은 공용으로 제공되는 비전 인코더의 잠재 임베딩(latent embedding) 공간을 활용하며, 관찰된 단계에서 추출한 동결된 클립 수준의 임베딩을 조건으로 사용하여 보이지 않는 단계의 행동을 예측합니다. 이를 통해 예측 모델은 반복적 디노이징(iterative denoising)을 통해 견고한 표현을 학습하게 됩니다. 본 연구는 Diffusion Transformers의 최근 발전을 활용했습니다.

- **Performance Highlights**: 총 4개의 데이터셋(NIV, CrossTask, COIN, Ego4D-v2)에서 5개의 절차적 학습 태스크에 대한 실험적으로, long-horizon action anticipation에서 강력한 기준선을 +2.6%(Verb ED@20) 및 +3.1%(Noun ED@20) 향상시키며, 단계 예측(step forecasting)에서 +5.0%, 작업 분류(task classification)에서 +3.8%, 절차 계획(procedure planning) 작업에서 최대 +2.28%(success rate), +3.39%(mAcc), +0.90%(mIoU) 향상된 결과를 보였습니다.



### Auto-GDA: Automatic Domain Adaptation for Efficient Grounding Verification in Retrieval Augmented Generation (https://arxiv.org/abs/2410.03461)
- **What's New**: 이 논문에서는 Retrieval Augmented Generation (RAG)과 자연어 추론(NLI) 모델의 결합을 통해 LLM의 허구성 문제를 해결하기 위한 새로운 접근 방식인 자동 생성 도메인 적응(Automatic Generative Domain Adaptation, Auto-GDA)을 제안합니다.

- **Technical Details**: Auto-GDA는 합성 데이터 생성을 활용하여 비지도 도메인 적응을 가능하게 합니다. 이 방법은 낮은 효율의 teacher 모델에서 약한 레이블을 사용하여 생성된 샘플의 품질을 지속적으로 개선하는 반복적인 프로세스를 차별화합니다. 또한, 기존의 NLI 모델에 비해 RAG 입력에 적합하게 모델을 조정하는 것을 목표로 합니다.

- **Performance Highlights**: Auto-GDA를 사용하여 합성 데이터로 미세 조정된 모델은 종종 teacher 모델을 초과하는 성능을 보이며, 인간 레이블 데이터로 미세 조정된 참조 모델과 유사한 성능을 달성합니다. 또한, LLM의 10배 적은 계산 비용으로도 유사한 성능을 나타냅니다.



### Generative Semantic Communication for Text-to-Speech Synthesis (https://arxiv.org/abs/2410.03459)
Comments:
          The paper has been accepted by IEEE Globecom Workshop

- **What's New**: 본 논문은 텍스트-음성 변환(TTS) 합성을 위한 새롭고 혁신적인 생성적 의미 통신(generative semantic communication) 프레임워크를 제안합니다. 이 프레임워크는 기존의 데이터 재구성 작업에 한정된 전통적 의미 통신 방식의 한계를 극복합니다.

- **Technical Details**: 제안된 프레임워크는 두 개의 의미 지식 기반(semantic knowledge bases, KBs)을 통해 구성됩니다. 송신기에는 WavLM이라는 대규모 음성 모델을 사용하여 효과적인 의미 추출을 위한 KB를 구축하고, 수신기에는 생생한 음성을 합성하기 위한 KB를 구축합니다. 이를 통해 transformer encoder와 diffusion model을 사용하여 효율적인 의미 코딩을 달성합니다.

- **Performance Highlights**: 제안하는 프레임워크는 네 가지 기준선 모델과 비교하여, 백색 가우시안 잡음(additive white Gaussian noise) 채널과 레일리 페이딩(channel) 조건 모두에서 생성된 음성의 충실도를 크게 향상시킵니다.



### A General Framework for Producing Interpretable Semantic Text Embeddings (https://arxiv.org/abs/2410.03435)
Comments:
          19 pages, 5 figures, and 9 tables

- **What's New**: CQG-MBQA(Contrastive Question Generation - Multi-task Binary Question Answering)는 다양한 작업에 대한 해석 가능한 의미 텍스트 임베딩을 생성하기 위한 일반적인 프레임워크로, 고비용의 전문가 지식이나 정밀한 프롬프트 설계 없이도 높은 차별성을 지닌 질문을 체계적으로 생성한다.

- **Technical Details**: 이 프레임워크는 CQG 방법론을 활용하여 저 인지 부하의 Yes/No 질문을 생성하고, MBQA 모델을 통해 이를 효율적으로 답변함으로써 해석 가능한 임베딩을 비용 효율적으로 생성한다. CQG는 LLM을 활용하여 텍스트 간의 의미적 뉘앙스를 포착하는 이진 질문을 강조하여 임베딩 공간의 차원을 형성한다.

- **Performance Highlights**: CQG-MBQA는 블랙박스 모델과 유사한 품질의 임베딩을 제공하며, 다양한 다운스트림 작업에서 기존의 해석 가능한 텍스트 임베딩 방법들을 초월하는 성능을 보여준다.



### EB-NeRD: A Large-Scale Dataset for News Recommendation (https://arxiv.org/abs/2410.03432)
Comments:
          11 pages, 8 tables, 2 figures, RecSys '24

- **What's New**: 에크스트라 블라뎃 뉴스 추천 데이터셋(EB-NeRD)이 도입되었습니다. 이 데이터셋은 100만 명 이상의 고유 사용자의 데이터와 3,700만 개 이상의 인상 로그를 포함하고 있으며, 125,000개 이상의 덴마크 뉴스 기사를 포함하고 있습니다. EB-NeRD는 RecSys '24 챌린지의 기준 데이터셋으로 활용되었습니다.

- **Technical Details**: EB-NeRD 데이터셋은 사용자 행동 로그로부터 수집되었으며, 다양한 기술적 문제를 해결하기 위한 연구를 지원합니다. 여기에는 뉴스 기사의 연속적인 출간, 신속한 소멸 문제, 사용자 피드백 기반의 모델링 기법이 포함됩니다. 또한, 텍스트 정보를 활용하는 방법도 강조됩니다.

- **Performance Highlights**: EB-NeRD 데이터셋은 고전적인 디지털 뉴스 게시자의 각기 다른 뉴스 소비 및 콘텐츠 프로필에 대한 일반화 가능성을 탐색할 수 있는 기회를 제공합니다. 데이터셋은 뉴스 추천 시스템의 설계에서 기술적 및 규범적 도전을 해결하는 데 매우 유용합니다.



### Aircraft Radar Altimeter Interference Mitigation Through a CNN-Layer Only Denoising Autoencoder Architectur (https://arxiv.org/abs/2410.03423)
Comments:
          To be presented at MILCOM 2024, Washington DC

- **What's New**: 본 연구에서는 통신 시스템에서 들리는 방해 신호를 제거하는 동시에 고도로 구조화된 FMCW 레이더 신호를 재구성하기 위해 CNN 전용 오토인코더 구조를 활용하는 방법을 제시합니다.

- **Technical Details**: 이 연구는 CNN (Convolutional Neural Network) 레이어만을 사용하는 오토인코더 아키텍처를 적용하여, 열악한 방해 환경에서도 레이더 높이 측정의 정확성을 개선하는 과정을 설명합니다. FMCW 레이더 신호의 IQ 샘플을 사용하여 신호 처리에 적합한 구조를 갖춘 간결한 아키텍처가 강조됩니다.

- **Performance Highlights**: 제안된 방법은 좁은 대역 톤 방해 신호 및 넓은 대역 QPSK 방해 신호의 존재에도 불구하고 레이저 altimeter의 RMS 범위 오류, 잘못된 고도 보고서 수, 그리고 결과적인 범위 프로파일의 피크 대 사이드 로브 비율에서 성능 개선을 이끌어냅니다.



### Benchmarking the Fidelity and Utility of Synthetic Relational Data (https://arxiv.org/abs/2410.03411)
- **What's New**: 관계형 데이터(relational data) 생성에 대한 연구가 증가하고 있으며, 본 연구는 기존 방법론에 대한 종합적인 평가를 통해 효율적인 평가 기법을 제시합니다.

- **Technical Details**: 관계형 데이터의 합성을 위한 새로운 평가 방법론을 개발하였으며, SDMetrics 패키지와 결합된 기능을 통해 6개의 분석 기법을 비교했습니다. 각각의 방법들은 실제 데이터와 상이한 점이 존재하며, 유용성 측면에서 실 데이터와 합성 데이터 간에 보통 중간 정도의 상관관계를 보입니다.

- **Performance Highlights**: 상업적 도구를 포함한 여러 방법을 평가한 결과, 어떤 방법도 원본 데이터와 구별되지 않는 합성 데이터 세트를 생성하지 못했습니다. 합성 데이터의 유용성은 일반적으로 모델 예측 성능(predictive performance) 및 속성 중요도(feature importance)에 있어서 실 데이터와 중간 정도의 상관관계를 보였습니다.



### Conformal confidence sets for biomedical image segmentation (https://arxiv.org/abs/2410.03406)
- **What's New**: 이번 연구에서는 이미지 분할을 위한 블랙박스 머신러닝 모델의 출력을 위한 공간적 불확실성 보장을 제공하는 신뢰 구간(confidence sets)을 개발하였습니다. 이 과정에서 우리는 이미지 설정에 맞게 조정된 conformal inference를 활용하여, 진짜 마스크 내외부의 변형된 logit 점수의 최대값 분포를 기반으로 한 보정 데이터셋에서 임계값(thresholds)을 얻습니다.

- **Technical Details**: 이 연구에서는 split-conformal inference 접근법을 사용하여 보정 데이터셋에서 분할기의 출력을 임계값으로 설정하는 방법을 배웁니다. 이 임계값은 모델이 제공한 변형된 logit 점수의 최대값 분포를 고려하여 산출합니다. 이를 통해 예측에 적용하면, 실제로 알려지지 않은 분할된 마스크를 특정 확률로 포함할 수 있는 내부 및 외부 신뢰 구간을 생성할 수 있습니다.

- **Performance Highlights**: 진단 영상의 조기 개입 기회를 놓치거나 잘못된 치료 결정을 피하기 위해, 적절한 점수 변환을 학습하는 것이 성능 최적화에 매우 중요함을 보여주고 있습니다. 특히, 병변 위치에 대한 엄격한 경계를 유지하면서도 잘못된 커버리지 비율(false coverage rate)을 제어하는 방안을 시연합니다.



### Distributed Networked Multi-task Learning (https://arxiv.org/abs/2410.03403)
- **What's New**: 이번 논문에서는 이질적(homogeneous)이고 상관관계가 있는 데이터 스트림을 고려한 분산 다중 작업 학습(DAMTL, Distributed and Asynchronous algorithm for Multi-task Learning) 기법을 제안합니다. 이 방식은 노드들이 그룹으로 분할되어 서로 다른 학습 작업을 수행하고, 비대칭적인 형태로 통신하며 동작합니다.

- **Technical Details**: 제안된 알고리즘은 비선형 모델 추정 작업을 다루며, 각각의 노드는 그룹 내 로컬 정규화와 그룹 간 글로벌 정규화 조건을 따릅니다. 논문에서는 인과 관계를 추정하기 위해 두 가지 단계의 최적화 문제를 제시하며, 외부 문제는 가우시안 작업 관계 모델의 공분산 행렬을 추정하고, 내부 문제는 선형 모델을 추정하는 것입니다. 비동기적 SGD(stochastic gradient descent) 업데이트를 통해 지속적으로 데이터 업데이트를 처리합니다.

- **Performance Highlights**: 한 실험으로 합성 온도 추정 문제와 다양한 학술 지구의 학생 성과 모델링에 대한 실제 데이터를 사용하여 알고리즘의 성능을 평가하였으며, 제안된 기법이 정밀 행렬 추정의 정확성 및 효율성을 향상시키는 데 기여한다고 보고합니다.



### Lightning UQ Box: A Comprehensive Framework for Uncertainty Quantification in Deep Learning (https://arxiv.org/abs/2410.03390)
Comments:
          10 pages, 8 figures

- **What's New**: Lightning UQ Box는 딥 뉴럴 네트워크(DNN)에 대한 불확실성 정량화(Uncertainty Quantification, UQ) 방법을 효과적으로 적용하고 평가할 수 있는 통합 인터페이스를 제공합니다. 이는 DNN의 결과에 신뢰도를 부여하는 데 필수적인 도구입니다.

- **Technical Details**: 이 툴박스는 PyTorch와 Lightning과 함께 작동하며, 다양한 UQ 방법론의 구현을 제공합니다. 특히, Bayes convolution layer와 같은 유연한 레이어 구성으로 비전 응용 프로그램에 최적화되어 있습니다.

- **Performance Highlights**: Lightning UQ Box는 위성 이미지를 기반으로 열대 저기압의 최대 지속 풍속을 추정하고, RGB 이미지를 이용하여 태양광 패널의 전력 출력을 예측하는 두 가지 도전적인 비전 작업에서 그 유용성을 입증합니다.



### Error Correction Code Transformer: From Non-Unified to Unified (https://arxiv.org/abs/2410.03364)
- **What's New**: 본 연구에서는 Polar, Low-Density Parity-Check (LDPC), Bose-Chaudhuri-Hocquenghem (BCH) 등 여러 선형 블록 코드에 대해 
 통합된 Transformer 기반 디코딩 아키텍처를 제안합니다. 이 아키텍처는 코드 유형에 독립적이며, 다양한 디코딩 알고리즘을 위한 고유 하드웨어 회로의 필요성을 없애 
 엔지니어링 노력과 하드웨어 자원 효율성을 줄입니다.

- **Technical Details**: 제안된 디코딩 아키텍처는 표준화된 유닛과 통합 주의 메커니즘을 통해 최대 길이의 코드워드에 맞춰 파라미터를 
 정렬합니다. 통합된 주의 메커니즘은 각 주의 헤드가 독립적으로 계산하지 않고 파라미터를 공유하여 코드워드의 
 구조적 정보를 압축합니다. 여기에 희소 마스크를 도입하여 정보 비트와 패리티 검사 비트 간의 내재적 제약을 포착하여 
 디코딩 성능을 크게 향상시킵니다.

- **Performance Highlights**: 제안된 Transformer 기반 디코딩 아키텍처는 전통적인 O(N²) 복잡도를 O(N)으로 감소시키며, 희소 마스크를 통해 
 계산 복잡도를 83% 줄입니다. 이러한 성능 개선 덕분에 제안된 아키텍처는 차세대 무선 통신 시스템에서 실시간 
 애플리케이션에 적합합니다.



### Audio-Agent: Leveraging LLMs For Audio Generation, Editing and Composition (https://arxiv.org/abs/2410.03335)
- **What's New**: 이번 논문에서는 Audio-Agent라는 멀티모달 프레임워크를 소개합니다. 이 프레임워크는 텍스트 또는 비디오 입력 기반으로 오디오 생성과 편집을 수행합니다. 기존의 텍스트-오디오(TTA) 생성 방식이 복잡한 텍스트 조건에 대해 성능이 제한된 반면, Audio-Agent는 입력을 세분화하여 고품질 오디오를 생성할 수 있도록 설계되었습니다.

- **Technical Details**: Audio-Agent는 GPT-4와 사전 훈련된 TTA 확산 네트워크를 결합하여 오디오 생성 과정을 다단계를 통해 수행합니다. 특히 비디오-오디오(VTA) 작업에 있어, Gemma2-2B-it라는 대형 언어 모델을 사용하여 비주얼 입력을 의미론적 토큰으로 변환하며, 이러한 정보를 사용해 두 가지 모달리티 간의 격차를 연결합니다. TTA와 VTA 모두에서 효율적으로 작동하며, 기존의 복잡한 훈련 과정 없이도 높은 성능을 발휘할 수 있도록 최적화되었습니다.

- **Performance Highlights**: Audio-Agent는 긴 복잡한 텍스트 입력에 대해 고품질 오디오를 생성할 수 있으며, 광범위한 평가에서 기존의 특정 작업 모델들과 유사한 성능을 보여주었습니다. 이는 특히 영화 더빙 및 음악 작곡과 같은 다양한 콘텐츠 생성 분야에서 큰 잠재력을 가지고 있습니다.



### Quo Vadis, Motion Generation? From Large Language Models to Large Motion Models (https://arxiv.org/abs/2410.03311)
- **What's New**: 이 논문에서는 대규모 모션 모델 개발을 위한 첫 번째 백만 레벨의 모션 생성 벤치마크인 MotionBase를 소개합니다. 이는 이전의 가장 큰 데이터셋보다 15배 더 많은 데이터를 제공하고, 계층적으로 세분화된 텍스트 설명을 포함한 다중 모드 데이터로 구성되어 있습니다.

- **Technical Details**: MotionBase는 100만 개 이상의 모션 시퀀스를 포함하여 모션 데이터 수집의 필요성을 해결합니다. 이 연구는 모션 토큰화를 위한 새로운 2D 조회 없는 접근 방식을 도입하여 모션 정보를 보존하고 코드북 용량을 확장하여 대규모 모션 모델의 표현 능력을 향상시킵니다.

- **Performance Highlights**: MotionBase를 활용하여 대규모 모션 모델은 다양한 모션, 특히 보지 못한 모션에서도 높은 성능을 나타냅니다. 데이터 및 모델 크기 확장이 중요한 요인으로 밝혀졌고, 기존의 평가 메트릭은 도메인 외 텍스트 지침 처리에 한계를 보였음을 강조합니다.



### Five Years of COVID-19 Discourse on Instagram: A Labeled Instagram Dataset of Over Half a Million Posts for Multilingual Sentiment Analysis (https://arxiv.org/abs/2410.03293)
- **What's New**: 이 논문은 COVID-19와 관련된 인스타그램 포스트의 데이터셋과 다국적 감성 분석의 결과를 제시하면서 2020년부터 2024년까지의 감정 경향을 탐구합니다.

- **Technical Details**: 500,153개의 인스타그램 포스트로 구성된 다국어 데이터셋을 구축하였으며, 이는 161개의 언어와 535,021개의 해시태그를 포함합니다. 이 데이터셋은 감정 분석을 위한 속성을 포함하고 있으며, 매글로벌 언어마다 양의(positive), 음의(negative), 중립(neutral)으로 분류된 결과를 갖추고 있습니다.

- **Performance Highlights**: 연도별 감성 분석 결과, 긍정 감정 비율이 38.35%에서 28.69%로 감소한 반면, 중립 감정 비율은 44.19%에서 58.34%로 증가했습니다. 언어별 분석에서는 영어 포스트의 긍정 감정 비율이 49.68%인 반면, 힌디어 포스트는 4.40%에 불과해 언어에 따른 감정 분포에 뚜렷한 차이가 나타났습니다.



### Manikin-Recorded Cardiopulmonary Sounds Dataset Using Digital Stethoscop (https://arxiv.org/abs/2410.03280)
- **What's New**: 이번 논문은 심장 및 폐 소리를 모니터링하기 위한 첫 번째 데이터셋을 소개합니다. 개별 및 혼합 심폐(cardiorespiratory) 소리를 포함하는 데이터셋으로, 향상된 정밀도가 특징입니다.

- **Technical Details**: 디지털 청진기(digital stethoscope)를 사용하여 임상 인형(clinical manikin)에서 수집한 데이터로, 다양한 인체 위치에서의 청진 소리를 포함합니다. 이 데이터셋은 정상 소리와 여러 비정상 음향(murmur, atrial fibrillation, tachycardia 등)도 포함합니다.

- **Performance Highlights**: 인공 지능(artificial intelligence) 애플리케이션에 적합한 이 데이터셋은 자동화된 심폐질환 감지, 소리 분류, 비지도 분리 기술(unsupervised separation techniques), 딥러닝(deep learning) 알고리즘에 유용합니다.



### Sm: enhanced localization in Multiple Instance Learning for medical imaging classification (https://arxiv.org/abs/2410.03276)
Comments:
          24 pages, 14 figures, 2024 Conference on Neural Information Processing Systems (NeurIPS 2024)

- **What's New**: 이 논문에서는 여러 인스턴스 학습(Multiple Instance Learning, MIL)에서 지역 종속성을 모델링하기 위한 새로운 방법을 제안합니다. 이 방법은 인스턴스의 이웃이 같은 라벨을 가질 가능성이 높다는 간단한 관찰에 기반합니다.

- **Technical Details**: 제안된 새로운 메커니즘은 지역 종속성을 모델링하는데 유연하고 원칙적인 접근을 사용하며, 기존의 글로벌 종속성 모델링 메커니즘(예: transformers)과 함께 사용될 수 있습니다. 이전의 MIL 방법들은 인스턴스를 독립적으로 취급하였으나, 최근 방법들은 인스턴스 간의 글로벌 및 로컬 종속성을 고려하고 있습니다.

- **Performance Highlights**: 제안된 모듈은 로컬라이제이션(localization)에서 최첨단 성능을 보이며, 분류(classification)에서도 경쟁력 있는 또는 우수한 성능을 보여줍니다.



### Optimal Transport for $\epsilon$-Contaminated Credal Sets (https://arxiv.org/abs/2410.03267)
- **What's New**: 이번 논문에서는 Monge 및 Kantorovich의 최적 수송 문제에 대한 하위 확률 버전을 제안합니다. 특히, 하위 확률이 $	heta$-오염된 집합의 하한봉일 때 우리의 Monge 버전과 제한된 Kantorovich 버전이 각각의 고전적 버전과 일치한다는 점을 입증하였습니다.

- **Technical Details**: 논문에서는 Monge 및 Kantorovich 문제의 최적 계획을 찾기 위한 충분한 조건을 디스크립션하고 있으며, 이 두 문제의 동등성을 보여줍니다. 또한, $	heta$-오염 상태에서는 하위 확률 버전들 사이에 일치하지 않을 수 있다는 것을 강조합니다.

- **Performance Highlights**: 연구 결과는 머신러닝(Machine Learning)과 인공지능(Artificial Intelligence) 분야에도 적용 가능성을 제시하며, 다양한 응용 프로그램에서의 확률적 접근 방법의 효과성을 강조합니다.



### Elucidating the Design Choice of Probability Paths in Flow Matching for Forecasting (https://arxiv.org/abs/2410.03229)
Comments:
          30 pages

- **What's New**: 이 연구에서는 spatio-temporal 데이터 예측에서 flow matching의 효과성을 높이기 위해 새로운 확률 경로 모델을 제안합니다. 이 모델은 예측 성능을 개선하는 데 중요한 역할을 하며, 훈련 시 빠른 수렴을 보여줍니다.

- **Technical Details**: 제안된 모델은 latent space에서의 flow matching을 적용하기 위한 이론적 틀과 효율적인 알고리즘을 제공합니다. 이 모델은 확률적 예측을 위한 시스템으로 시간 시계열 데이터의 복잡한 시간 의존성을 효과적으로 모델링합니다.

- **Performance Highlights**: 우리의 실험 결과, 새로운 확률 경로 모델은 기존 모델에 비해 훈련 동안 더 빠른 수렴성과 향상된 예측 성능을 보여주었습니다. 모델의 효율성 덕분에 실제 적용 사례에서도 유용하게 사용될 수 있습니다.



### Learning to steer with Brownian nois (https://arxiv.org/abs/2410.03221)
- **What's New**: 이번 논문은 주어진 시스템의 매개변수에 대한 완전한 지식이 없는 결정자가 제어를 수행하면서 동시에 매개변수를 학습해야 하는 제한 속도 추적기 문제의 ergodic 버전을 다룹니다. 이 논문은 이동 평균을 기반으로 한 알고리즘을 제안하고, 통계적 방법과 확률적 제어 이론을 통합하는 프레임워크를 개발하였습니다.

- **Technical Details**: 이 논문에서는 Brownian motion (
W)는 표준 Brownian motion을 사용하여 결정자가 드리프트 bₜ를 제어함으로써 목표 값 0에 가능한 한 가깝게 유지하는 문제를 설정합니다. 핵심 가정은 드리프트 속도 bₜ가 제한된 간격 [θ₀, θ₁]에서 선택된다는 것입니다. 제어 알고리즘은 탐색 우선 (explore-first) 전략과 Adaptive Position Averaging with Clipping을 통해 이루어지며, 후자의 알고리즘은 시간에 따른 매개변수 추정을 업데이트합니다.

- **Performance Highlights**: 제안된 알고리즘의 이론적 분석에 따르면, explore-first 알고리즘은 T의 제곱근에 비례하는 레그렛 비율을 달성하며, Adaptive Position Averaging 알고리즘은 Log(T)에 비례하는 급격한 수렴 속도를 나타냅니다. 이러한 결과는 기존의 모델 없는 접근 방식, 예를 들어 deep-Q learning에 비해 데이터 요구량이 현저히 적다는 것을 보여줍니다.



### Learning Semantic Structure through First-Order-Logic Translation (https://arxiv.org/abs/2410.03203)
Comments:
          EMNLP 2024 Findings

- **What's New**: 본 연구는 transformer 기반의 언어 모델이 단순 문장에서 술어 인자 구조(predicate argument structure)를 추출할 수 있는지를 탐구합니다. 이 과정에서 언어 모델이 어떤 술어가 어떤 객체에 적용되는지를 혼동하는 경우가 있음을 보이고, 이를 해결하기 위한 두 가지 작업(질문 응답(Q/A) 및 1차 논리(FOL) 번역)을 제안합니다.

- **Technical Details**: 연구에서는 두 가지 접근 방식인 질문 응답(Q/A) 및 1차 논리(FOL) 형태로의 번역을 통해 LLM의 술어 인자 구조 학습 능력을 분석합니다. FOL 번역에서는 대규모 언어 모델을 합성 데이터셋에 대해 미세 조정(finetuning)하여 일반화 능력을 측정합니다. 질문 응답의 경우 BERT 및 RoBERTa와 같은 인코더 모델을 미세 조정하며 LLM에서는 프롬프팅(promting) 방식을 사용합니다.

- **Performance Highlights**: 연구 결과, LLM에서 FOL 번역이 Q/A 프롬프팅보다 훨씬 나은 성능을 보이며, LLM이 단순 술어 인자 구조를 더 복잡한 문장으로 일반화할 수 있음을 보여줍니다. 또한, 프롬프트 방식으로는 모델이 허구의 내용을 추가할 때를 파악할 수 없지만, FOL 번역 방법은 이러한 한계를 극복합니다.



### Nested Deep Learning Model: A Foundation Model for Brain Signal Data (https://arxiv.org/abs/2410.03191)
Comments:
          31 pages; references added; 14 pages supplementary materials added

- **What's New**: 본 연구는 Epilepsy(간질) 환자를 위한 새로운 Nested Deep Learning (NDL) 프레임워크를 제안하여 EEG/MEG 기반 스파이크 탐지의 한계를 극복하고자 한다.

- **Technical Details**: NDL은 모든 채널의 신호를 가중치 조합하여 적용함으로써 다양한 채널 구성에 대한 적응성을 제공하며, 임상의가 중요한 채널을 보다 정확하게 식별할 수 있도록 돕는다.

- **Performance Highlights**: 실제 EEG/MEG 데이터셋을 통한 이론적 분석 및 실증적 검증을 통해 NDL은 기존 방법에 비해 스파이크 탐지 및 채널 로컬라이제이션에서 더 높은 정확도를 보여주었으며, 이는 다양한 신경생리학적 응용에 대해 세밀한 조정이 가능하다는 것을 나타낸다.



### Redefining Temporal Modeling in Video Diffusion: The Vectorized Timestep Approach (https://arxiv.org/abs/2410.03160)
Comments:
          Code at this https URL

- **What's New**: 이 논문은 기존의 비디오 확산 모델(VDM)의 한계를 해결하기 위해 새롭게 프레임 인식 비디오 확산 모델(FVDM)을 제안합니다. 이 모델은 각 프레임에 대해 독립적인 노이즈 스케줄을 허용하는 벡터화된 시간을 도입하여, 복잡한 시간적 의존성을 효과적으로 모델링합니다.

- **Technical Details**: FVDM은 기존 VDM의 스칼라 시점 변수를 대신해 벡터화된 시점 변수(VTV)를 사용합니다. 이를 통해 각 프레임은 고유의 시간 경로를 따라 발전할 수 있으며, 노이즈에서 복구되는 과정에서도 더 정교한 시간적 의존성을 캡처할 수 있습니다. 이 방법은 이전 모델에서 발생하는 catastrophic forgetting(재학습시의 대폭망각) 문제를 해결하고, 제로샷(zero-shot) 학습에서도 뛰어난 일반화를 보여줍니다.

- **Performance Highlights**: FVDM은 기존 최신 기술보다 비디오 생성 품질에서 뛰어난 성능을 발휘하며, 표준 비디오 생성뿐만 아니라 이미지-비디오 변환, 비디오 보간(video interpolation), 긴 비디오 생성과 같은 다양한 작업에서도 우수성을 나타냅니다. 이러한 성능 개선은 FVDM이 생성 모델링 및 멀티미디어 애플리케이션에 중요한 영향을 미친다는 것을 시사합니다.



### ECHOPulse: ECG controlled echocardio-grams video generation (https://arxiv.org/abs/2410.03143)
- **What's New**: ECHOPULSE는 최초로 ECG 신호를 기반으로 한 ECHO 비디오 생성 모델로, 복잡한 조건 프롬프트 없이 빠르고 효율적으로 ECHO 영상을 생성할 수 있는 가능성을 보여줍니다.

- **Technical Details**: ECHOPULSE는 비디오 토큰화 모델, 마스킹 생성 변환기, 그리고 비디오 생성 파이프라인으로 구성되어 있으며, VQ-VAE 토큰화 및 마스킹 비주얼 토큰 모델링을 활용하여 ECHO 비디오 생성을 가속화합니다. ECG 신호를 사용하여 비디오를 생성하는 방식은 시간적 상관관계를 포착하는 것이 핵심입니다.

- **Performance Highlights**: 세 가지 공공 및 개인 데이터 세트에서 ECHOPULSE는 정량적 및 정성적 측정 모두에서 SOTA 성과를 달성하며, 심장 MRI, fMRI, 3D CT와 같은 다른 모달리티 생성 작업에도 쉽게 일반화될 수 있습니다.



### Autoregressive Action Sequence Learning for Robotic Manipulation (https://arxiv.org/abs/2410.03132)
- **What's New**: 이 논문에서는 로봇 조작 작업을 위한 새로운 자율 회귀(autoregessive) 아키텍처인 Chunking Causal Transformer (CCT)를 제안합니다. CCT는 인과(transformer) 모델의 다음 단일 토큰 예측을 다중 토큰 예측으로 확장하여, 효율적인 학습과 실행을 가능하게 합니다.

- **Technical Details**: CCT는 고유의 attention interleaving 전략을 채택하여 teacher-forcing으로 효율적으로 학습할 수 있게 하며, 이로 인해 다양한 로봇 조작 환경에서 근본적인 인과 관계(causal relations)를 활용할 수 있습니다. 논문에서는 주요 로봇 환경인 Push-T, ALOHA, RLBench에서 ARP(Autoregressive Policy) 모델의 성능을 평가합니다.

- **Performance Highlights**: ARP는 모든 테스트 환경에서 최첨단(State-of-the-Art) 방법들을 능가하는 성능을 보여주며, 계산 효율과 파라미터 크기에서 더 높은 효율성을 자랑합니다. 실제 로봇 실험에 대한 비디오 데모와 소스 코드는 논문에 포함되어 있습니다.



### AIME: AI System Optimization via Multiple LLM Evaluators (https://arxiv.org/abs/2410.03131)
Comments:
          21 pages, 10 Figures, 4 Tables

- **What's New**: AI 시스템 최적화에서 단일 LLM 평가자 사용의 한계를 강조하고, 여러 LLM 평가자를 사용하는 AIME 프로토콜을 제안합니다. 이를 통해 복잡한 코드 생성 작업에서 높은 성능을 달성할 수 있음을 보여줍니다.

- **Technical Details**: AIME (AI system optimization via Multiple Evaluators) 프로토콜은 서로 다른 기준에 대해 독립적으로 평가를 생성하는 여러 LLM을 사용하여 각 평가를 합치는 방식으로 작동합니다. 이 방법은 오류 감지율을 62%까지 향상시키고, 성공률을 16% 증가시킵니다.

- **Performance Highlights**: AIME는 LeetCodeHard와 HumanEval 벤치마크에서 단일 평가 프로토콜에 비해 최대 62% 높은 오류 감지율을 보이며, 테스트 케이스의 성공률은 최대 16% 더 높았습니다. 또한 평가자의 수와 기준 선택이 성공률에 영향을 미친다는 점을 강조합니다.



### ARB-LLM: Alternating Refined Binarizations for Large Language Models (https://arxiv.org/abs/2410.03129)
Comments:
          The code and models will be available at this https URL

- **What's New**: 본 논문은 ARB-LLM이라는 새로운 1-bit 포스트 트레이닝 양자화(PTQ) 기술을 제안합니다. 이는 대형 언어 모델(LLM)에 최적화되었으며, 이 모델의 메모리 및 계산 요구량을 크게 줄일 수 있습니다.

- **Technical Details**: ARB-LLM은 교차 정제 양자화(Alternating Refined Binarization, ARB) 알고리즘을 기반으로 하여, 양자화 오차를 줄이고, 컬럼 그룹 비트맵(Column-Group Bitmap, CGB) 전략을 개선하여 성능을 향상시킵니다. ARB-X 및 ARB-RC와 같은 확장 기술로, 교정 데이터를 통합하고 컬럼 방향의 편차를 최소화합니다.

- **Performance Highlights**: 실험 결과, ARB-LLM$_{RC}$는 현재의 SOTA 이진 PTQ 방법들보다 훨씬 높은 성능을 보이며, 동일한 크기의 FP16 모델을 초월하는 성과를 거두었습니다. 또한, 이 알고리즘은 대규모 LLM의 실용적인 배포에 필요한 메모리 자원을 최소화합니다.



### On Unsupervised Prompt Learning for Classification with Black-box Language Models (https://arxiv.org/abs/2410.03124)
- **What's New**: 이번 연구에서는 라벨이 없는 데이터로 블랙박스 LLMs(large language models)를 파인튜닝(fine-tuning)하는 가능성을 탐구하는 언슈퍼바이즈드 프롬프트 학습(un,supervised prompt learning) 방식을 제안하였습니다. 특히, 프롬프트(prompt)와 가짜 라벨(pseudo labels)을 동시에 학습하여 분류 문제를 해결하며, 이렇게 생성된 가짜 라벨 데이터셋을 이용하여 추가적인 파인튜닝이 가능합니다.

- **Technical Details**: 저자들은 프롬프트를 의미 있는 암호화된 토큰 시퀀스로 모델링하며, 각 토큰마다 학습되어야 하는 범주형 분포(categorical distribution)를 최적화합니다. 이 과정에서 LLM의 인컨텍스트 학습(in-context learning; ICL) 기능을 활용하여 신뢰할 수 있는 가짜 라벨 데이터(pseudo-labeled data)를 식별하고, 이러한 데이터를 통해 다른 라벨이 없는 데이터에 대한 가짜 라벨을 생성합니다. 이를 통해 프롬프트 학습과 사용 단계에서의 일관성을 높입니다.

- **Performance Highlights**: 기준 데이터셋에서 수행한 실험 결과, 제안된 알고리즘의 효과가 입증되었습니다. 언슈퍼바이즈드 프롬프트 학습 후 생성된 가짜 라벨 데이터셋은 블랙박스 LLM 소유자에 의해 추가 파인튜닝을 위해 사용할 수 있습니다.



### Shrinking: Reconstruction of Parameterized Surfaces from Signed Distance Fields (https://arxiv.org/abs/2410.03123)
Comments:
          6 pages, 4 figures, accepted by ICMLA

- **What's New**: 본 논문에서는 반사거리장(SDF)으로부터 명시적으로 매개변수화된 표면을 재구축하는 새로운 방법을 제안합니다. 기존의 전통적인 재구축 방법들과는 달리, 초기 구를 반복적으로 축소하여 SDF 형태에 맞추는 접근 방식을 채택합니다.

- **Technical Details**: 우리의 방법은 매개변수화된 초기 구를 반복적으로 축소하여 목표 SDF 형태에 적합하도록 조정합니다. 이 과정에서 미분 가능성과 표면 매개변수를 유지하며, 각 단계에서 균일 분포를 유지하기 위해 리메싱(remeshing)을 Integration하여 표면 연속성과 부드러운 매개변수를 보장합니다.

- **Performance Highlights**: 우리의 방법은 대표적인 기하학적 형태와 ABC 데이터셋의 일부에서 평가되었으며, 전통적인 방법에 비해 경쟁력 있는 재구축 품질을 달성했습니다. 이러한 결과는 고급 컴퓨터 그래픽스 및 기하학적 심층 학습 애플리케이션에 필요한 부드러움과 미분 가능성을 유지한다는 점에서 중요합니다.



### RIPPLECOT: Amplifying Ripple Effect of Knowledge Editing in Language Models via Chain-of-Thought In-Context Learning (https://arxiv.org/abs/2410.03122)
Comments:
          EMNLP findings

- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)에서 지식 편집의 도전 과제인 ripple effect를 해결하기 위한 새로운 접근법, RippleCOT를 제안합니다. 이 방법은 Chain-of-Thought (COT) 추론을 통합하여 다중 단계 질문을 처리할 수 있도록 돕습니다.

- **Technical Details**: RippleCOT는 'new fact, question, thought, answer' 구조로 시연을 구성하며, 새로운 사실을 기반으로 다중 단계 질문의 논리를 식별하고 분해합니다. 이 접근 방식은 첫 번째 단계에서 여러 관계를 식별하고 두 번째 단계에서 생성된 쌍 중에서 질문과 높은 코사인 유사성을 가지는 상위 k 후보를 선택합니다.

- **Performance Highlights**: RippleCOT는 기존의 방법보다 우수한 성능을 보이며, MQuAKE-cf 벤치마크에서 정확도를 7.8%에서 87.1%까지 향상시켰습니다.



### ProcBench: Benchmark for Multi-Step Reasoning and Following Procedur (https://arxiv.org/abs/2410.03117)
- **What's New**: 이 논문에서는 ProcBench라는 새로운 벤치마크를 제시하여 다단계 추론(multi-step inference)에 대한 직접 평가를 중점적으로 다루고 있습니다. 기존의 벤치마크와는 달리, ProcBench는 복잡한 지식 없이 제공된 지침을 따르는 능력을 평가합니다.

- **Technical Details**: ProcBench는 여러 가지 간단한 작업을 포함하며, 각 작업은 명확히 정의된 지침을 따르는 과정을 요구합니다. 연구에서 사용된 데이터셋은 명시적인 지침과 해당 질문의 쌍으로 구성되어 있으며, 각 단계에서 모델이 지침을 따르는 능력을 평가합니다.

- **Performance Highlights**: 최신 대형 언어 모델(LLMs)에 대한 평가 결과, 모델에 따라 성능 차이가 있었습니다. o1-preview와 o1-mini와 같은 일부 모델은 간단한 작업에서 높은 정확도를 보였지만, 복잡성이 증가함에 따라 성능이 크게 저하되는 한계를 드러냈습니다.



### Mamba in Vision: A Comprehensive Survey of Techniques and Applications (https://arxiv.org/abs/2410.03105)
Comments:
          Under Review

- **What's New**: Mamba는 컴퓨터 비전 내에서 Convolutional Neural Networks (CNNs)와 Vision Transformers (ViTs)의 한계를 극복하기 위한 새로운 접근 방식으로 등장하였습니다. Mamba는 또는 선형 계산 복잡도를 기반으로 한 Selective Structured State Space Models를 활용하여 장거리 의존성을 효과적으로 캡처하는 데 중점을 두었습니다.

- **Technical Details**: Mamba 모델은 입력 데이터에 기반하여 동적으로 조정되는 선택적 상태 표현을 사용하여 computational overhead를 줄이고 효율성을 높입니다. 이는 구조적 상태 공간 모델(SSMs)의 발전을 기반으로 하여 이루어지며, Mamba는 GPU를 최적화한 스캔 기반 알고리즘을 활용하여 기존의 convolution 기반 SSMs의 비효율성을 피합니다.

- **Performance Highlights**: Mamba 모델은 비디오 처리, 원격 감지, 의료 영상 등 다양한 분야에서 특히 유리하며, CNNs와 ViTs는 높은 계산 요구로 인해 확장성 문제를 겪는 반면, Mamba 모델은 시퀀스 길이에 대한 선형 확장성을 제공하여 실시간 및 대규모 애플리케이션에 적합합니다.



### Forest Proximities for Time Series (https://arxiv.org/abs/2410.03098)
- **What's New**: PF-GAP는 기존의 RF-GAP를 확장하여 시계열 데이터 분류의 정확성과 효율성을 높이기 위해 제안된 새로운 접근 방식입니다.

- **Technical Details**: PF-GAP는 Multidimensional Scaling을 사용하여 단일 변수 시계열의 벡터 임베딩을 얻고, Local Outlier Factors와 결합하여 잘못 분류된 포인트와 이상치 간의 관계를 조사합니다. 또한, Random Forest의 RF-GAP 근접성을 시계열 데이터의 Proximity Forest로 확장하였습니다.

- **Performance Highlights**: PF-GAP를 사용하면 다른 일반적인 시계열 거리 측정 방법을 사용하는 방법들보다 시계열 시각화와 이상치 탐지 작업에서 더 우수한 성능을 보입니다.



### Entanglement-induced provable and robust quantum learning advantages (https://arxiv.org/abs/2410.03094)
Comments:
          7 pages, 2 figures + 13-page supplementary materials

- **What's New**: 본 논문은 양자 컴퓨팅(Quantum Computing)이 기계 학습(Machine Learning)에 미치는 이점을 명확히 입증하지 못한 상황에서, 노이즈에 강하고 무조건적인 양자 학습 우위를 확립했습니다.

- **Technical Details**: 정보 이론 기반으로 양자 얽힘(Quantum Entanglement)을 활용하여 비국소(non-local) 기계 학습 작업의 통신량을 줄이는 방법을 제시하고, 클래스 모델과의 비교를 통해 양자 모델의 교육 시간과 샘플 수를 문제 크기에 반비례하도록 설계했습니다.

- **Performance Highlights**: 양자 모델은 상수 시간과 샘플 수로 교육할 수 있으며, 전통적인 클래스 모델은 최소 선형 비율로 크기를 조정해야 대폭 향상된 정확도를 달성할 수 있습니다. 논문의 수치 시뮬레이션 및 IonQ Aria에서의 트랩 이온 실험을 통해 양자-클래식 학습 분리를 입증했습니다.



### UNComp: Uncertainty-Aware Long-Context Compressor for Efficient Large Language Model Inferenc (https://arxiv.org/abs/2410.03090)
- **What's New**: 본 논문에서는 UNComp라는 새로운 압축 방안을 제안합니다. UNComp는 매트릭스 엔트로피(matrix entropy)를 활용하여 모델의 불확실성을 추정하고, 이를 통해 K-V 캐시(KV cache)와 은닉 상태(hidden states)를 적응적으로 압축할 수 있는 방법을 제공합니다.

- **Technical Details**: UNComp는 레이어(layer)와 헤드(head)를 기반으로 그룹화하여 압축 비율을 조정합니다. 이 방법은 KV 캐시를 4.74%까지 압축하고, 단일 배치에서 프리필링(prefilling) 단계에서 1.6배 속도 향상과 6.4배의 처리량 증가를 낳습니다. 또한 성능 저하는 1.41%에 불과합니다.

- **Performance Highlights**: UNComp는 특정 작업에서 9.38% 압축 비율에도 불구하고 전체 KV 캐시의 성능을 초과합니다. 이를 통해 기계 학습 모델이 효율성을 높이고, 긴 맥락 상황에서의 추론 속도를 대폭 향상시킬 수 있음을 보여줍니다.



### Geometric Collaborative Filtering with Convergenc (https://arxiv.org/abs/2410.03064)
Comments:
          13 pages, 1 figure, 3 tables

- **What's New**: 이번 연구에서는 잠재 변수 기반의 협업 필터링(latent variable collaborative filtering) 방법의 수학적 특성을 처음으로 분석하고, 일반화(generalization) 갭을 정의하였습니다. 특히, 아이템 메타데이터(metadata)를 활용하여 추천 시스템의 성능을 향상시키는 방법을 제안합니다.

- **Technical Details**: 본 논문에서는 일반화 갭의 개념을 도입하여 기하학적 상한(geometric upper bound)을 제시합니다. 이를 바탕으로 새로운 추천 알고리즘 GeoCF를 개발하였으며, 아이템 간의 기하학적 관계를 고려한 손실 함수(loss function)를 정의하였습니다. 제안된 방법론은 통계 학습 이론(Statistical Learning Theory)을 기반으로 하여 과적합(overfitting)을 방지하는 방향으로 설계되었습니다.

- **Performance Highlights**: GeoCF 알고리즘은 Movielens20M 및 Netflix 데이터 세트와 두 개의 대규모 내부 데이터 세트에서 다른 기존 방법들보다 뛰어난 성능을 보여주었습니다. 이를 통해, GeoCF는 기존의 최첨단 방법들의 성능을 초월하는 것을 입증했습니다.



### Scalable Frame-based Construction of Sociocultural NormBases for Socially-Aware Dialogues (https://arxiv.org/abs/2410.03049)
Comments:
          17 pages

- **What's New**: 본 논문은 대화에서 사회적으로 인식된 행동을 지원하기 위해 대형 언어 모델(LLMs)을 활용한 사회문화적 규범(SCN) 제작을 제안합니다. 이를 통해 중국 문화에 특화된 첫 번째 SCN 데이터베이스인 ChineseNormBase를 구축했습니다. 이 데이터베이스는 사회적 맥락을 고려하여 생성된 자연어 규범 진술을 포함하고 있습니다.

- **Technical Details**: SCNs는 사회맥락적 프레임을 이용해 추출되며, 이 과정은 대화의 맥락을 이해하고 환각(hallucination)을 줄이는 데 도움이 됩니다. 실제 대화 데이터가 부족할 경우, 합성 데이터(synthetic data)를 사용하여 SCNs를 효과적으로 생성할 수 있습니다. 이와 더불어, RAG 기반 모델(Retrieval-Augmented Generation)을 통해 다양한 대화 작업에 대한 추론 능력을 시연했습니다.

- **Performance Highlights**: 실험 결과, 합성 데이터를 이용하여 추출한 SCNs의 품질이 금본(gold frames)으로 주석을 달아 만든 실제 대화에서 추출한 SCNs와 동등하다는 것을 확인했습니다. 또한, 은본(silver frames)이나 금본으로 주석을 단 실제 데이터에서 추출된 SCNs의 품질이 주석이 없는 데이터와 비교하여 우수함을 입증했습니다.



### Vehicle Suspension Recommendation System: Multi-Fidelity Neural Network-based Mechanism Design Optimization (https://arxiv.org/abs/2410.03045)
- **What's New**: 본 논문에서는 다중 진실성(multi-fidelity) 설계 프레임워크를 제안하여 차량 서스펜션 시스템의 최적 유형과 설계를 추천하는 접근 방식을 소개합니다. 이 방법은 저비용의 간단한 분석과 고비용의 정밀 분석을 통합하여 설계 공간 내에서 최적의 성능 지표를 찾는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 프레임워크는 저진실성(저급) 강체 동역학 분석과 고진실성(고급) 다중 탄성체 동역학 시뮬레이션을 통합하여, 깊이 있는 학습 기반 대리 모델(surrogate model)을 구축합니다. 이를 통해, 다양한 서스펜션 구조의 성능 메트릭을 다중 목표 최적화(multi-objective optimization) 문제로 설정하여 균형 잡힌 설계 솔루션을 도출합니다. 데이터 마이닝(data mining) 기술을 활용하여 파레토 솔루션(Pareto solutions)의 기본적인 설계 규칙을 추출합니다.

- **Performance Highlights**: 세 가지 서스펜션 유형(맥퍼슨 기둥, 더블 위시본 등)에 대한 사례研究를 통해 제안된 방법론의 효과성을 입증했습니다. 이 방법을 적용한 결과 설계 효율성이 향상되고, 비용이 절감되며, 전체 차량 성능이 개선된 것을 확인했습니다.



### Minmax Trend Filtering: A Locally Adaptive Nonparametric Regression Method via Pointwise Min Max Optimization (https://arxiv.org/abs/2410.03041)
- **What's New**: 이번 연구에서는 Fused Lasso와 Total Variation Denoising의 로컬 어댑티비티(local adaptivity)를 설명하는 새로운 점별(pointwise) 공식을 개발하였습니다. 이 공식은 로컬 평균(penalized local averages) 최적화의 min-max/max-min 관점에서 도출되어 Fused Lasso의 로컬 어댑티비티를 구체적으로 설명합니다.

- **Technical Details**: Fused Lasso 추정량에 대한 새로운 점별 공식은 min-max/max-min 최적화를 기반으로 하며, 이는 배타적으로 전통적인 선형 스무딩(linear smoothing) 방법과 구별되는 비모수 회귀(nonparametric regression) 접근 방식을 제공합니다. 이 논문에서는 추가로 Fused Lasso의 높은 차수 다항식 버전을 제안하였으며 이를 Minmax Trend Filtering이라고 명명하였습니다.

- **Performance Highlights**: Minmax Trend Filtering 방법은 각 지점에서 최적의(bias-variance tradeoff) 편향-분산 균형에 의해 추정 오차가 제한되며, 이는 기존 선형 스무딩 방법들의 로컬 어댑티비티 부족을 극복하는 혁신적인 접근법입니다.



### Geometry is All You Need: A Unified Taxonomy of Matrix and Tensor Factorization for Compression of Generative Language Models (https://arxiv.org/abs/2410.03040)
- **What's New**: 이번 연구는 자연어 처리(NLP) 모델의 매트릭스(matrix)와 텐서(tensor) 지향 파라미터화(parametrization)를 통합하는 새로운 분류체계를 제안합니다. 이는 기존 연구의 수학적 접근에서 벗어나서 머신러닝(ML) 및 NLP 연구와의 연관성을 강화하려는 시도입니다.

- **Technical Details**: 연구는 선형대수학의 기본 개념인 서브스페이스(subspace)를 채택하여 매트릭스 및 텐서와 ML/NLP 개념을 재정형화했습니다. 이렇게 함으로써 일반적인 매트릭스 및 텐서 분해 알고리즘이 기하학적 변환으로 해석됩니다.

- **Performance Highlights**: 마지막으로, 본 연구는 최근 매트릭스 또는 텐서 기반 언어 모델 압축 문헌을 재조명하고, 이들의 핵심 아이디어를 비교 분석하여 현재 연구의 공백과 가능성 있는 해결책을 제시했습니다.



### Revealing the Unseen: Guiding Personalized Diffusion Models to Expose Training Data (https://arxiv.org/abs/2410.03039)
Comments:
          Under review

- **What's New**: 최근 Diffusion Models (DMs)의 발전으로 이미지 생성 및 개인화된 스타일 학습이 가능해졌습니다. 그러나 이러한 모델의 미세 조정(checkpoint)을 공유할 경우 데이터 유출과 저작권 침해 우려가 있습니다. 본 논문에서는 FineXtract라는 새로운 프레임워크를 제안하여 온라인에서 공유된 DMs로부터 훈련 데이터를 추출할 수 있는 방법을 모색합니다.

- **Technical Details**: FineXtract 방법은 사전 학습된 모델에서 미세 조정된 모델로의 학습 분포 변화를 모델링합니다. 이 과정에서 사전 학습 모델과 미세 조정 모델의 스코어 함수를 외삽(extrapolate)하여, 고밀도(high-density) 지역으로의 생성 과정을 유도합니다. 클러스터링(Clustering) 알고리즘을 적용하여 생성된 이미지 중 최상위 확률 이미지를 추출합니다. 이 방법은 조건부 및 비조건부 DMs에 모두 적용 가능합니다.

- **Performance Highlights**: WikiArt, DreamBooth 및 실세계 체크포인트를 포함한 여러 데이터셋에서 실험을 통해, 본 방법이 대개 20%의 미세 조정 데이터를 정확히 추출할 수 있음을 입증하였습니다. 이는 기존 방법 대비 월등한 성능을 증명합니다.



### Disentangling Textual and Acoustic Features of Neural Speech Representations (https://arxiv.org/abs/2410.03037)
- **What's New**: 이번 연구는 Neural Speech Models(NSMs)의 내부 표현을 텍스트 정보와 음향 특성으로 분리하는 새로운 Framework를 제안합니다. 정보 병목(Information Bottleneck) 원리를 기반으로 하여 복잡한 음성 표현을 내용(content)과 음향 특성(acoustic features)으로 분리합니다.

- **Technical Details**: 우리는 두 단계의 분리 프레임워크를 구축하였습니다. 첫 번째 단계에서는 Decoder를 훈련하여 내부 표현을 텍스트로 변환하고, 비관련 정보를 최소화하려 합니다. 두 번째 단계에서는 같은 음성 표현을 기반으로 하지만, 텍스트 정보를 피하면서 특정 작업에 유리한 음향 특성을 캡쳐하는 Decoder를 훈련합니다.

- **Performance Highlights**: 우리의 프레임워크는 감정 인식과 화자 식별 두 가지 다운스트림 작업에서 강력한 성능을 보였습니다. 텍스트 표현은 전통적인 음성 모델보다 효과적으로 텍스트를 예측할 수 있었으며, 음향 표현은 음향 특성을 더 잘 예측하는 데 성공했습니다.



### Characterizing Context Influence and Hallucination in Summarization (https://arxiv.org/abs/2410.03026)
- **What's New**: 이 연구에서는 Large Language Models (LLMs)의 맥락 정보가 요약 생성 시 어떻게 영향을 미치는지를 정의하고 분석합니다. 새로운 개념인 Context Influence Decoding (CID)을 도입하여 맥락의 영향을 증가시키는 방법을 제시합니다.

- **Technical Details**: 연구에서 제안된 CID는 Pointwise Cross-Mutual Information을 기반으로 한 맥락 영향력의 정의를 포함하고 있습니다. 이를 통해 기존의 모델에서는 고려되지 않았던 맥락 정보의 중요성과 LLM의 환각 문제를 동시에 분석합니다.

- **Performance Highlights**: 실험 결과, LLaMA 3 모델의 CNN-DM 데이터셋에서 ROGUE-L 점수를 10% 향상시킨 결과, 맥락의 영향력이 1.5배 증가하는 것을 확인했습니다. 이는 모델의 성능을 높이는 동시에 개인 정보 유출의 가능성을 분석하는 데 중요합니다.



### Towards Understanding the Universality of Transformers for Next-Token Prediction (https://arxiv.org/abs/2410.03011)
Comments:
          Preprint, 22 pages

- **What's New**: Causal Transformers의 다음 토큰 예측 능력에 대한 이해를 깊이 있게 다룬 연구로, 기존 self-attention 메커니즘 뒤에 있는 기본적인 작동 원리를 명확히 하고자 했습니다.

- **Technical Details**: 이 논문에서는 특정 함수 $f$가 선형이거나 주기적일 때, Causal Transformers가 다음 토큰 $x_{t+1}$을 예측하는 능력을 분석합니다. 주어진 자기회귀 시퀀스 $(x_1, 
ewline 	ext{...}, x_t)$에 대해, causal kernel descent 방식을 통해 토큰 맵핑을 학습하는 Transformer를 구축하고, Kaczmarz 알고리즘과의 연관성을 설명합니다.

- **Performance Highlights**: 이론적인 발견을 검증하는 실험 결과를 제시하며, 일반적인 맵핑 $f$에도 적용 가능성을 시사합니다.



### GABIC: Graph-based Attention Block for Image Compression (https://arxiv.org/abs/2410.02981)
Comments:
          10 pages, 5 figures, accepted at ICIP 2024

- **What's New**: 이번 연구에서는 Graph-based Attention Block for Image Compression (GABIC)이라는 새로운 주의(attention) 메커니즘을 소개합니다. GABIC는 k-Nearest Neighbors (k-NN) 메커니즘을 활용하여 중복된 특징을 줄이는 방법을 제안하고 있습니다.

- **Technical Details**: GABIC는 지역 그래프를 기반으로 한 주의 메커니즘으로, 주의 과정에서 k-NN 기술을 사용하여 중복된 시각적 특징의 집합을 방지합니다. 실험을 통해 GABIC는 특히 높은 비트 전송률에서 기존의 유사한 방법들에 비해 압축 성능이 향상됨을 보여주었습니다.

- **Performance Highlights**: 실험 결과 GABIC는 비트 비율이 높은 상황에서도 압축 성능의 개선을 보여주었으며, 이는 최근에 보편화된 Labeled Image Compression (LIC) 모델에 비해 우수한 성능을 나타냅니다. 또한, 코드와 학습된 모델이 공개되어 연구자들이 활용할 수 있도록 되어 있습니다.



### From Optimization to Sampling via Lyapunov Potentials (https://arxiv.org/abs/2410.02979)
- **What's New**: 본 연구는 Langevin Dynamics를 이용하여 고차원 분포에서 샘플링하는 문제를 탐구하고 있다. 기존의 Gradient Descent 방법과의 유사성 덕분에, 최적화된 log-density를 바탕으로 수치적으로 적절한 온도 수준에서 샘플링할 수 있다는 점을 발견하였다.

- **Technical Details**: Langevin Dynamics는 각 단계에서 적절히 조정된 Gaussian noise를 추가하는 Gradient Descent의 변형으로, 고차원 분포 μβ에서 샘플링하기 위한 Stochastic Differential Equation(SDE)을 제시한다. 이 연구에서는 Poincaré Inequality와 Log-Sobolev Inequality 등 여러 기하학적 특성을 기반으로 μβ가 샘플링할 수 있는 조건을 구명하였다.

- **Performance Highlights**: 이 알고리즘을 통해 새로운 비-log-concave 밀도의 클래스에서 효율적으로 샘플링할 수 있는 가능성이 확장되었다. 특히, ζ-온도 범위에서의 성능이 강조되며, 비선형 에너지 함수를 최적화할 수 있는 샘플을 제공하는 데 성공하였다.



### Label-Free Subjective Player Experience Modelling via Let's Play Videos (https://arxiv.org/abs/2410.02967)
Comments:
          9 pages, 3 figures, AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment

- **What's New**: 이번 연구는 Let’s Play 비디오를 활용하여 플레이어 경험 모델링(Player Experience Modelling, PEM)을 개발하는 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: PEM의 개발을 위해 수집된 Angry Birds 게임의 Let’s Play 비디오에서 비디오 프레임과 오디오 정보를 활용합니다. 컷트된 오디오 조각의 크기를 기반으로 한 음성의 크기(Amplitude)와 감정을 매핑하여 Convolutional Neural Network (CNN)를 훈련시킵니다. 오디오의 정상화된 진폭을 변환하여 감정 값을 예측하는 모델을 개발합니다.

- **Performance Highlights**: 인간 피험자 연구를 통해 수집된 생리적 신호 및 설문 조사 데이터와 CNN 출력을 비교함으로써 제안한 모델이 자기보고식 측정과 높은 상관관계를 보이는 것을 확인했습니다.



### Bushfire Severity Modelling and Future Trend Prediction Across Australia: Integrating Remote Sensing and Machine Learning (https://arxiv.org/abs/2410.02963)
Comments:
          15 pages

- **What's New**: 이 연구는 호주의 지난 12년 동안의 bushfire(산불) 심각도를 철저히 분석한 결과를 제시하며, remote sensing(원거리 탐지) 데이터와 machine learning(기계 학습) 기술을 결합하여 미래 산불 경향을 예측합니다.

- **Technical Details**: Landsat imagery(랜드샛 이미지)를 이용하여 NDVI(Normalized Difference Vegetation Index), NBR(Normalized Burn Ratio), Burn Index(소각 지수)와 같은 스펙트럴 인덱스를 통합하고, 지형 및 기후 요인과 결합하여 XGBoost를 사용하여 강력한 예측 모델을 개발했습니다.

- **Performance Highlights**: 이 모델은 86.13%의 높은 정확도를 달성하였으며, 호주의 다양한 생태계에서의 산불 심각도를 예측하는 데 효과적임을 보여줍니다.



### On Expert Estimation in Hierarchical Mixture of Experts: Beyond Softmax Gating Functions (https://arxiv.org/abs/2410.02935)
Comments:
          58 pages

- **What's New**: Mixture of Experts (MoE) 아키텍처의 발전과 함께 Hierarchical Mixture of Experts (HMoE)라는 특수한 변형을 조사하며, 이는 복잡한 입력을 처리하고 특정 작업에서 성능을 향상시키는 데 뛰어납니다.

- **Technical Details**: HMoE 프레임워크 내에서 softmax gating을 넘어선 다양한 gating functions를 사용해야 하는 장점이 강조됩니다. 이 연구에서 맞춤형 gating functions를 각 전문 그룹에 적용함으로써 HMoE가 강력한 결과를 달성하는 이론적 근거를 제시합니다. 이는 최적의 gating functions가 선택된 계층 수준에서만 적용되더라도 가능합니다.

- **Performance Highlights**: 대규모 멀티모달 작업, 이미지 분류 및 잠재 도메인 발견과 예측 작업을 포함한 다양한 시나리오에서 우리의 수정된 HMoE 모델이 뛰어난 성능 향상을 보였습니다.



### Streamlining Conformal Information Retrieval via Score Refinemen (https://arxiv.org/abs/2410.02914)
Comments:
          6 pages

- **What's New**: 본 논문에서는 정보 검색(Information Retrieval, IR) 시스템에서의 통계적 보장을 제공할 수 있는 새로운 스코어 리파인먼트(score refinement) 방법을 제안합니다. 이 방법은 기존의 큰 크기의 세트를 생성하는 문제를 해결하여 작은 크기의 세트를 유지하면서도 통계적 보장을 보장합니다.

- **Technical Details**: 우리가 제안하는 스코어 리파인먼트 방법은 단순한 단조 변환(monotone transformation)을 적용하여 IR 시스템의 점수를 조정합니다. 이러한 점수의 정제(refinement)를 통해, 표준적인 위신 예측(conformal prediction) 방법을 사용하여 컴팩트한 세트를 생성하고, 불필요한 계산 비용을 줄이며 응답 속도를 향상시킵니다.

- **Performance Highlights**: 실험 결과, BEIR 벤치마크 데이터셋을 통해 제안한 방법이 경쟁 방식들보다 더 효과적으로 관련 정보를 포함한 소형 세트를 생성함을 확인했습니다.



### Fine-Tuning Language Models with Differential Privacy through Adaptive Noise Allocation (https://arxiv.org/abs/2410.02912)
Comments:
          EMNLP 2024 findings

- **What's New**: 이 논문에서는 ANADP라는 새로운 알고리즘을 제안하여 언어 모델의 매개변수 중요도에 따라 추가 노이즈를 적응적으로 할당합니다. 이 접근법은 전통적인 Differential Privacy(DP) 방법의 한계를 극복하고 기계 학습 모델의 프라이버시를 강화하는 동시에 성능을 개선합니다.

- **Technical Details**: ANADP는 매개변수의 중요도에 기반하여 노이즈와 프라이버시 예산을 분배하는 방법입니다. 이는 매개변수의 감도(sensitivity)와 불확실성(uncertainty)을 고려하여 모델의 훈련 과정에서 안정적으로 적용됩니다. 기존의 DP 방법과 달리, ANADP는 각 매개변수의 기여도를 평가하여 균일하지 않은 방식으로 프라이버시 예산을 배분합니다.

- **Performance Highlights**: ANADP는 Glue benchmark에서 기존의 DP 방법보다 항상 우수한 성능을 보여주었으며, 전통적 DP와 비-DP 파인튜닝(수정 없이 원본을 유지한 파인튜닝) 간의 성능 격차를 줄이는 데 성공했습니다. 또한 ANADP는 기존의 DP 방법처럼 강력한 프라이버시 보호를 유지합니다.



### Solving Reach-Avoid-Stay Problems Using Deep Deterministic Policy Gradients (https://arxiv.org/abs/2410.02898)
- **What's New**: 본 논문에서는 로봇과 에어 택시가 목표에 도달하고, 장애물을 피하며, 목표 근처에 머무는 Reach-Avoid-Stay (RAS) 최적 제어 문제를 해결하기 위해 두 단계의 deep deterministic policy gradient (DDPG) 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 두 가지 단계로 구성됩니다. 첫 번째 단계에서는 시스템이 안전하게 머물 수 있는 최대 견고한 제어 불변 집합(maximal robust control invariant set)을 교육하고, 두 번째 단계에서는 이 제어 불변 집합에 안전하게 도달할 수 있는 상태의 집합을 정의하는 함수를 교육합니다.

- **Performance Highlights**: 제안한 방법은 복잡한 환경에서도 RAS 문제를 해결할 수 있고, 고차원 시스템에 확장 가능하며, 이전 방법들에 비해 높은 성공률을 보임을 시뮬레이션과 두 가지 고차원 실험을 통해 검증하였습니다.



### The Role of Deductive and Inductive Reasoning in Large Language Models (https://arxiv.org/abs/2410.02892)
Comments:
          4 figures

- **What's New**: 이 논문에서는 Deductive and InDuctive (DID) 방법을 제안하여 LLM(Large Language Models)의 추론 능력을 향상시키고, 동적으로 추론 경로를 조정할 수 있는 유연한 프레임워크를 제공합니다.

- **Technical Details**: DID 방법은 인지 과학의 원리에 기반하여 유도적(inductive)과 연역적(deductive) 추론 과정을 프롬프트 구성 과정에 통합하여 LLM의 추론 유연성과 적응성을 높입니다. 이 접근법은 다양한 데이터셋에서 검증되었으며, 모델의 성능을 크게 향상시킵니다.

- **Performance Highlights**: DID 방법을 사용한 결과, 기존 저명한 데이터셋인 AIW와 MR-GSM8K 및 자체 제작한 Holiday Puzzle에서 솔루션 정확도와 추론 품질 모두에서 유의미한 향상을 보여주었습니다. 이 모든 개선사항은 substantial computational overhead 없이 이루어졌습니다.



### Universally Optimal Watermarking Schemes for LLMs: from Theory to Practic (https://arxiv.org/abs/2410.02890)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 워터마킹(watermarking) 문제를 독립성 테스트(independence test)로 공식화하여 워터마킹 스킴과 탐지기를 공동 최적화하는 새로운 이론적 프레임워크를 제안합니다. 이를 통해 최소한의 Type-II 오류를 달성하여 탐지 성능과 왜곡(distortion) 간의 fundamental trade-off를 이해합니다.

- **Technical Details**: 본 프레임워크는 detectability, distortion, robustness 간의 fundamental trade-off를 설명하며, 이를 위해 Type-I 오류의 최악 사례 성능을 통제하고 모든 텍스트 분포에서 효과적인 보편적 탐지기를 설계합니다. 제안하는 방법론은 surrogate 모델(surrogate model)과 Gumbel-max 트릭(Gumbel-max trick)을 활용하여 토큰 수준의 워터마킹 알고리즘을 구현하며, 이는 모델 비종속적(model-agnostic)이고 계산적으로 효율적입니다.

- **Performance Highlights**: Llama-13B 및 Mistral-8×7B 모델에서 실시한 실험 결과, 제안된 방법이 token replacement 공격에도 불구하고 효과적임을 입증하였습니다. 또한, semantic-invariant 공격에 대한 강건성을 통합하여 향후 최적의 semantically 기반 워터마킹 시스템을 설계하는 데 필요한 통찰력을 제공합니다.



### FAIR Universe HiggsML Uncertainty Challenge Competition (https://arxiv.org/abs/2410.02867)
Comments:
          Whitepaper for the FAIR Universe HiggsML Uncertainty Challenge Competition, available : this https URL

- **What's New**: FAIR Universe 프로젝트는 불완전한 시뮬레이터 때문에 발생하는 물리적 속성 측정의 불확실성을 다루는 새로운 기회를 제공합니다. 이 과제는 대규모 AI 플랫폼을 활용하여 데이터셋을 공유하고, 모델을 훈련시키며, 머신러닝 대회를 개최하는 것을 목표로 하고 있습니다.

- **Technical Details**: 이 대회는 고에너지 물리학(HEP) 분야에서의 불확실성 인식 머신러닝(ML) 기술에 중점을 두며, 특히 HEP의 시스템적 불확실성을 이해하고 교정하는 방법론을 발전시키는 데 기여합니다. 참가자들은 Higgs 보존 사건 수를 예측하고 이에 대해 68.27%의 신뢰구간(Confidence Interval, CI)을 생성해야 하며, 여기에는 aleatoric(랜덤) 및 epistemic(시스템적) 불확실성이 포함됩니다.

- **Performance Highlights**: 이 대회는 그동안의 경험을 바탕으로 고안된 것으로, 약 2억 8천만 개의 이벤트로 이루어진 데이터셋을 사용하며, 이는 기존 HiggsML 경연의 데이터보다 세 배 이상의 양입니다. 참가자들은 새로운 메트릭을 통해 성과를 평가받게 되며, 향후 불확실성 인식 AI/ML 기술 개발에 중요한 이정표가 될 것입니다.



### Reconstructing Galaxy Cluster Mass Maps using Score-based Generative Modeling (https://arxiv.org/abs/2410.02857)
Comments:
          15 pages, 9 figures, submitted to The Open Journal of Astrophysics

- **What's New**: 본 논문에서는 score 기반 생성 모델링(score-based generative modeling)을 활용하여 은하단의 가스 및 암흑 물질의 투사 밀도 맵을 재구성하는 새로운 접근법을 제시합니다. 이 모델은 인공 SZ(Sunyaev-Zel'dovich) 및 X-ray 이미지를 조건부 관찰로 사용하여, 학습된 데이터 포스터리어(data posterior)로부터 샘플링을 통해 대응하는 맵을 생성합니다.

- **Technical Details**: 이 diffusion 모델은 대수 및 그림자 과정을 통해 관찰된 데이터에서 높은 정확도로 가스 및 암흑 물질 맵을 추정하는 데 사용됩니다. 모델의 성능은 수치 해석을 통해 생성된 가짜 데이터(mock data)를 사용하여 훈련 및 검증하였으며, 결과적으로 모델은 방사형 밀도 프로파일의 평균과 분산을 5% 이내로 정확하게 재구성합니다. 이 모델은 커다란 우주적 구조를 탐지하고 분석하는 데 매우 유용합니다.

- **Performance Highlights**: 모델은 스펙트럼 도메인에서 편향(bias)과 교차 상관 계수(cross-correlation coefficient)에서 거의 1에 가까운 값을 달성하였으며, 이는 모델이 다양한 크기의 클러스터 구조를 정확하게 프로빙(probing)할 수 있음을 나타냅니다. 또한, 이 모델은 관측된 데이터를 기반으로 추가적인 관측 가능성(observables)을 통합하고, 실제 관측 데이터를 활용하여 은하단 밀도 분포를 예측할 수 있도록 조정 및 일반화할 수 있는 가능성을 보여주었습니다.



### A Spatio-Temporal Machine Learning Model for Mortgage Credit Risk: Default Probabilities and Loan Portfolios (https://arxiv.org/abs/2410.02846)
- **What's New**: 본 논문에서는 신용 위험 분석을 위해 tree-boosting과 latent spatio-temporal Gaussian process 모델을 결합한 새로운 기계 학습 모델을 소개합니다. 이 모델은 비선형성과 예측 변수 간의 상호 작용을 유연하게 모델링할 수 있으며, 관측 가능한 예측 변수로 설명할 수 없는 시공간적 변동성을 반영합니다.

- **Technical Details**: 제안된 모델은 tree-boosting을 활용하여 예측 변수 간의 상호작용 및 비선형 효과를 포착하며, 동시에 시공간적 frailty correlation을 고려합니다. 이를 통해 상황에 따라 변동하는 신용 위험을 더 정확하게 모델링할 수 있는 장점을 제공합니다.

- **Performance Highlights**: 미국의 대규모 주택 담보 대출 데이터 세트에 적용한 결과, 제안된 spatio-temporal 기계 학습 모델은 기존의 독립적인 선형 위험 모델 및 선형 시공간 모델에 비해 대출의 예측적 디폴트 확률이 더 정확하다는 것을 발견했습니다. 특히, 글로벌 금융 위기 당시의 예측적 손실 수준에서도 보다 현실적인 예측을 보여 주었습니다.



### CAnDOIT: Causal Discovery with Observational and Interventional Data from Time-Series (https://arxiv.org/abs/2410.02844)
Comments:
          Published in Advanced Intelligent Systems

- **What's New**: 본 논문에서는 CAnDOIT라는 새로운 인과 발견 방법을 제안하여 관찰 데이터와 개입 간섭(interventional data)을 활용하여 인과 모델을 재구성하는 데 중점을 두고 있습니다. 기존 연구들과는 달리, 시간적 시계열 데이터를 이용한 인과 발견을 수행하는 첫 번째 접근 방식으로, 실세계 문제 해결에 기여할 것으로 기대됩니다.

- **Technical Details**: CAnDOIT는 인과적 구조를 추론하기 위해 관찰 데이터와 실험적 개입 데이터를 통합하는 알고리즘입니다. 이 방법은 LPCMCI라는 기존의 최첨단 알고리즘을 확장하여 시계열 데이터에서 인과 모델을 정확하게 추출할 수 있도록 개선되었습니다. 주요 구성 요소로는 개입 간섭을 통해 수집된 데이터의 효과적인 분석이 포함되어 있으며, Python 구현도 제공되어 GitHub에서 공개되었습니다.

- **Performance Highlights**: 실험 결과는 CAnDOIT가 랜덤 생성된 합성 모델과 로봇 조작 환경에서 실행된 벤치마크에서 인과 모델을 정확하게 재구성할 수 있음을 보여줍니다. 본 알고리즘은 시계열 데이터의 복잡한 문제를 해결하는 데 있어 기존 방법들보다 우수한 성능을 보이며, 이는 다양한 분야에서의 응용 가능성을 시사합니다.



### Modelling the longevity of complex living systems (https://arxiv.org/abs/2410.02838)
- **What's New**: 이 확장된 초록은 리투아니아 빌뉴스에서 열린 ECML PKDD 2024의 Nectar Track에서 발표되었습니다. 본 내용은 최근 발표된 논문 'Laws of Macroevolutionary Expansion'을 보완합니다.

- **Technical Details**: 이 연구는 생태적 관계의 거대 진화 (Macroevolution)와 관련된 관계의 진화를 탐구하며, 과거의 유물에서 과거 키 (key)가 소멸되는지에 대한 의문을 제기합니다. 또한, 자연 및 사회에서의 진화적 과정 (evolutionary processes)을 비교하는 연구 프로젝트에 기여하고 있습니다.

- **Performance Highlights**: 이 연구는 핀란드 연구 위원회 (Research Council of Finland)와 Kone 재단의 지원을 받아 진화 이론 (evolutionary theory)과 관련하여 치아 해부학 (Dental Anatomy)의 Valio Armas Korvenkontio 유닛에서 기여되었습니다.



### The MLE is minimax optimal for LGC (https://arxiv.org/abs/2410.02835)
- **What's New**: 이번 연구에서는 최근 도입된 Local Glivenko-Cantelli 설정을 재조명하고, 최대 우도 추정기(Maximum Likelihood Estimator, MLE) 이외의 임의의 추정기를 허용한 일반화를 탐구합니다. 특히, 더 큰 클래스의 측정을 학습할 수 있는지, 그리고 더 나은 위험 감소 속도를 얻을 수 있는지에 대한 질문을 다룹니다.

- **Technical Details**: 이 논문은 binomial empirical process를 통해 위의 설정을 다루며, 특정 분포 μ에 대해 i.i.d. 샘플을 통해 평균을 추정하는 방법을 제시합니다. 또한, MLE에 의한 추정 과정과 그 결과의 수렴성에 대해 다루고 있습니다. 연구진은 무한 차원 패스(patological)로의 활용을 막을 시 학습 가능한 측정의 클래스는 줄어들며, 이러한 제약을 허용할 경우 더 큰 클래스가 학습 가능하다는 점을 밝혔습니다.

- **Performance Highlights**: 이 조사의 결과는 제한을 두지 않고 허용할 경우, 더 큰 학습 가능한 측정을 제공하더라도 위험 감소 속도를 개선하는 데는 한계가 있다는 것을 보여줍니다. 이는 실용적인 데이터 분석 및 통계적 추정 이론에 중요한 의미를 가집니다.



### Asymmetry of the Relative Entropy in the Regularization of Empirical Risk Minimization (https://arxiv.org/abs/2410.02833)
- **What's New**: 이 연구에서는 상대 엔트로피 비대칭성(relativen entropy asymmetry)이 경험적 위험 최소화(empirical risk minimization, ERM)와 상대 엔트로피 정규화(ERM-RER) 맥락에서 연구되었습니다.

- **Technical Details**: 두 가지 정규화 방식이 검토되었습니다: $(a)$ 최적화할 측정치에 대한 기준 측정치의 상대 엔트로피(Type-I ERM-RER)와 $(b)$ 기준 측정치에 대한 최적화할 측정치의 상대 엔트로피(Type-II ERM-RER). 주요 결과는 Type-II ERM-RER 문제의 해(solution)를 특성화하고 그 핵심 속성을 밝히는 것입니다.

- **Performance Highlights**: 분석 결과, 두 경우 모두 상대 엔트로피에 의한 정규화가 솔루션의 지지(support)를 기준 측정치의 지지로 수렴하게 하며, 이는 훈련 데이터가 제공하는 증거를 압도할 수 있는 강한 귀납적 편향(strong inductive bias)을 도입함을 보여주었습니다. 또한, Type-II 정규화는 경험적 위험 함수(empirical risk function)의 적절한 변환을 통해 Type-I 정규화와 동등하다는 것이 증명되었습니다.



### Skill Issues: An Analysis of CS:GO Skill Rating Systems (https://arxiv.org/abs/2410.02831)
- **What's New**: 이번 연구에서는 Counter-Strike: Global Offensive (CS:GO) 게임의 스킬 레이팅 시스템인 Elo, Glicko2, TrueSkill을 분석하였습니다. 각 시스템의 성능을 실제 데이터를 기반으로 비교하고, 매칭 알고리즘의 영향을 고찰하였습니다.

- **Technical Details**: 연구는 스킬 레이팅 시스템을 다양한 에뮬레이터(Emulator)로 구현하고, 매칭 알고리즘을 Acquisition Function으로 설정하여 통합적인 시뮬레이터(Simulator) 환경을 조성하였습니다. 이 환경에서는 실제 게임 데이터에 따라 팀 매칭을 선택하고, 각 시스템의 성능을 검증하였습니다.

- **Performance Highlights**: Elo, Glicko2, TrueSkill 각 시스템의 성능을 대규모 CS:GO 데이터셋을 통해 시험하였으며, TrueSkill이 CS:GO에서 62%의 정확성을 보이는 등 높은 성능을 나타냈습니다. 이를 통해 매칭 알고리즘과 스킬 레이팅 간의 원형 의존성(circular dependency)을 정량적으로 평가할 수 있었습니다.



### LLMs May Not Be Human-Level Players, But They Can Be Testers: Measuring Game Difficulty with LLM Agents (https://arxiv.org/abs/2410.02829)
- **What's New**: 최근의 Large Language Models (LLMs)의 발전이 게임 테스트 분야에 새로운 가능성을 제시하고 있습니다. 이 연구에서는 LLM을 활용하여 게임의 난이도를 측정할 수 있는지 탐구하였으며, Wordle과 Slay the Spire와 같은 전략 게임을 통해 그 유효성을 검증하였습니다.

- **Technical Details**: 연구진은 LLM 에이전트를 활용한 일반 게임 테스트 프레임워크를 제안하였습니다. LLM은 간단한 지침을 통해 인간 플레이어의 평가와 강한 상관관계를 보여주며, 이는 게임 내부의 다양한 도전 과제의 난이도를 측정하는 데 유용할 수 있음을 시사합니다. 이 연구는 LLM이 별도의 세부 조정 없이 여러 게임에서 플레이하고 난이도를 평가할 수 있는 일반적인 프레임워크를 목표로 하고 있습니다.

- **Performance Highlights**: LLMs는 평균 인간 플레이어와 비교할 때 성능이 다소 부족하지만, 특정 과제에서 LLM의 난이도 인식은 인간 플레이어의 인식과 유사한 경향을 보였습니다. 이는 게임 개발 과정에서 LLM이 효과적인 난이도 평가 도구로 사용될 수 있음을 나타냅니다. 이 연구의 결과는 LLM 기반 게임 테스트 환경 개발에 대한 통찰력을 제공할 수 있습니다.



### Effective Intrusion Detection for UAV Communications using Autoencoder-based Feature Extraction and Machine Learning Approach (https://arxiv.org/abs/2410.02827)
Comments:
          4 pages

- **What's New**: 본 논문은 최근 실제 UAV 침입 데이터 세트를 활용하여 무인 항공기(UAV)에 대한 새로운 침입 탐지 방법을 제안합니다. 기존 연구는 주로 시뮬레이션 데이터 세트나 무관한 데이터 세트를 사용했으나, 본 연구는 실제 데이터 세트를 사용하여 autoencoder 기반의 머신 러닝 침입 탐지 방법을 처음으로 제안한 것입니다.

- **Technical Details**: 제안된 침입 탐지 시스템(IDS)은 '데이터 전처리', '특징 추출', '공격 분류'의 세 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째 단계로, raw data를 전처리하여 null 값 처리 및 데이터 정규화를 수행합니다. 두 번째로, autoencoder를 사용하여 특징을 추출하고 차원 축소를 수행합니다. 마지막으로, 추출된 데이터를 다양한 머신 러닝 모델(예: Random Forest, Support Vector Machine, K-Nearest Neighbors 등)에 입력하여 네트워크 트래픽을 정상 또는 비정상으로 분류하고 특정 공격 유형을 식별합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 이진 및 다중 클래스 분류 작업 모두에서 기존 방법론(특징 선택 기법)에 비해 우수한 성능을 보임을 나타냅니다. 이는 실제 UAV 침입 데이터 세트를 기반으로 한 고급 특징 추출 방법이 공격 탐지에 매우 효과적임을 시사합니다.



### LinkThief: Combining Generalized Structure Knowledge with Node Similarity for Link Stealing Attack against GNN (https://arxiv.org/abs/2410.02826)
- **What's New**: 이 논문에서는 LinkThief라는 새로운 link stealing attack을 제안합니다. 기존의 공격 방식은 비슷한 posterior를 가진 두 노드 간의 edge가 존재한다고 가정하는 데 반해, LinkThief는 이러한 가정에 얽매이지 않고 더 발전된 방법을 사용합니다.

- **Technical Details**: LinkThief는 attackers의 배경 지식으로부터 부분적으로 유출된 target graph와 shadow graph를 결합하여 edge 구조를 이해하기 위한 Shadow-Target Bridge Graph를 생성합니다. 이 과정을 통해 link structure에 대한 통찰을 제공하며, Edge Structure Feature Extractor를 설계하여 일반화된 구조 지식을 추출합니다.

- **Performance Highlights**: 실험 결과, LinkThief는 추가적인 가정을 두지 않고도 효과적으로 링크를 훔치는 것을 보여줍니다. 이론 분석과 실험을 통해 공격 모델의 정확성을 입증하였습니다.



### Inverse Design of Copolymers Including Stoichiometry and Chain Architectur (https://arxiv.org/abs/2410.02824)
Comments:
          24 pages, 20 figures

- **What's New**: 본 논문에서는 고유한 구조를 가진 합성 폴리머의 신속한 발견을 위해 generative 디자인 방식을 탐구하며, 특히 반복 단위와 모노머 집합(mononer ensembles)을 생성할 수 있는 새로운 딥러닝 구조인 Variational Autoencoder (VAE)를 제시합니다.

- **Technical Details**: 전통적인 폴리머 설계 방법과 달리 이 연구에서는 semi-supervised VAE를 사용하여 레이블이 부분적으로 있는 데이터셋을 처리하고, continuous latent space (LS)를 활용하여 다양한 monomer 조성과 체인 아키텍처를 갖는 copolymer 구조를 생성하는 모델을 개발합니다.

- **Performance Highlights**: 이 모델은 수소 생산을 위한 새로운 conjugated copolymer photocatalysts의 in-silico 발견을 위한 사례 연구를 통해, 잠재적(electron affinity) 및 이온화 잠재력(ionization potential) 최적화를 기반으로 효과적으로 작동함을 입증하였습니다.



### DANA: Domain-Aware Neurosymbolic Agents for Consistency and Accuracy (https://arxiv.org/abs/2410.02823)
- **What's New**: 이번 연구에서는 DANA(Domain-Aware Neurosymbolic Agent)라는 새로운 아키텍처를 소개합니다. 이 아키텍처는 도메인 특화 지식과 심볼릭(symbolic) 접근 방식을 통합하여 대형 언어 모델(LLM)의 확률적 특성으로 인한 불일치성과 부정확성을 해결합니다.

- **Technical Details**: DANA는 자연어와 심볼릭 형태로 도메인 전문 지식을 캡처하고 적용하여 더 결정적이며 신뢰할 수 있는 문제 해결 행동을 가능하게 합니다. 이 아키텍처는 OpenSSA 프레임워크에서 계층적 작업 계획(Hierarchical Task Plans, HTPs)을 사용하여 구현되었으며, 금융 분석 벤치마크인 FinanceBench에서 90% 이상의 정확도를 달성했습니다.

- **Performance Highlights**: DANA는 현재 LLM 기반 시스템보다 일관성과 정확성 면에서 우수한 성능을 보였으며, 반도체 제조 프로세스와 같은 물리적 산업에 적용 가능하다는 점을 강조하였습니다.



### GPT's Judgements Under Uncertainty (https://arxiv.org/abs/2410.02820)
- **What's New**: 본 연구는 인간의 인지 편향이 GPT-4o의 확률적 판단 및 결정 형성에 어떻게 드러나는지를 1350번의 실험을 통해 조사하였습니다. 이를 통해 비슷한 확률적 표기에 대한 반응 방식에서 AI의 모순적 접근을 보여주었습니다.

- **Technical Details**: 연구에서는 인지 편향으로는 손실 회피, 형태 효과, 결합 오류 등 9개의 편향을 사용하여 1350개의 실험을 진행하였으며, 통계적 추론과 직관적 추론 간의 반응을 분석했습니다. 각 실험은 150번 반복하여 결과의 일관성을 높였습니다.

- **Performance Highlights**: 총 1350개의 실험 중, GPT-4o는 658개의 상세한(elaborate) 응답과 692개의 직관적(intuitive) 응답을 제공하였습니다. 특히 결합 오류에 대한 실험에서는 언제나 상세한 응답을 제공하며, 통계적으로 타당한 이유를 설명했습니다.



### Physics-Informed Graph-Mesh Networks for PDEs: A hybrid approach for complex problems (https://arxiv.org/abs/2410.02819)
- **What's New**: 최근 딥러닝의 발전으로 Physics-Informed Neural Networks(PINNs)를 활용한 부분 미분 방정식(PDEs) 문제 해결이 가능해졌습니다. 하지만 물리적 불변성의 부족과 복잡한 기하학 다루기에서의 한계로 인해 산업 환경에서는 고전적인 수치 해법에 비해 경쟁력이 떨어집니다. 이번 연구에서는 자동 미분(automatic differentiation)의 한계를 지적하고, 물리 정보 그래프 신경망(Physics-Informed Graph Neural Networks)과 유한 요소법(numerical kernels)의 결합한 하이브리드 접근 방식을 소개합니다.

- **Technical Details**: 물리 정보를 활용한 그래프 망 네트워크는 복잡한 기하학을 다루기 위해 고안되었습니다. 이 모델은 초록형 및 삼차원 데이터에 적용되어 자동 미분의 한계를 해결하고, 수치 해법을 통해 물리적 잔차를 계산합니다. 연구는 이론적 특성 및 모델의 일반화 능력을 평가하는 실험을 포함하고 있습니다.

- **Performance Highlights**: 제안된 접근 방식은 자동 미분의 한계를 극복하고, 복잡한 산업 문제에 대해 훌륭한 일반화 능력을 보여줍니다. 실험 결과는 이 하이브리드 모델의 성능 향상을 입증하며, 기존의 PINNs 프레임워크에 비해 실용적인 면에서 큰 장점을 제공합니다.



### Neural Coordination and Capacity Control for Inventory Managemen (https://arxiv.org/abs/2410.02817)
- **What's New**: 본 논문은 제한된 자원을 가진 여러 제품을 관리하는 소매업체의 주기적 재고 관리 문제를 다루고 있습니다. 특히, 용량 제어 메커니즘의 백테스트(backtest) 방법과 최근의 Deep Reinforcement Learning(DRL) 발전에 맞는 용량 제어 메커니즘 개발에 대한 질문에 혁신적인 접근법을 제시합니다.

- **Technical Details**: 본 논문은 Amazon의 과거 용량 제한의 단일 샘플 경로를 기반으로 하여 실제 시나리오를 포함하는 가능한 제약 경로 분포에서 샘플링하는 방법을 제안합니다. 또한, exo-IDP(Exogenous Decision Process) 프레임워크를 확장하여 유용한 역량 제어 문제를 정의하고, 'neural coordinator'를 소개하여 용량 가격의 예측을 생산하여 전통적인 모델 예측 제어기를 대체합니다. 이는 RL 기반의 구매 정책 학습을 위해 수정된 DirectBackprop 알고리즘을 적용합니다.

- **Performance Highlights**: 제안된 방법론은 대규모 백테스트를 통해 평가되었으며, neural coordinator를 사용한 RL 구매 정책이 전통적인 기준선보다 누적 할인 보상과 용량 준수 모두에서 최대 50% 향상된 성과를 보였습니다.



### SAC-KG: Exploiting Large Language Models as Skilled Automatic Constructors for Domain Knowledge Graphs (https://arxiv.org/abs/2410.02811)
Comments:
          ACL 2024 Main

- **What's New**: 본 논문에서는 SAC-KG라는 일반적인 지식 그래프(KG) 구축 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLM)을 전문 지식으로 활용하여 자동으로 특화된 다단계 KGs를 생성합니다.

- **Technical Details**: SAC-KG는 세 가지 구성 요소로 이루어져 있습니다: Generator, Verifier, Pruner. Generator는 원시 도메인 코퍼스에서 관련성을 가진 관계와 꼬리 엔티티를 생성하고, Verifier는 오류를 감지하여 수정하며, Pruner는 필요에 따라 다음 단계의 생성을 결정합니다.

- **Performance Highlights**: SAC-KG는 100만 개 이상의 노드 규모로 도메인 KG를 자동으로 구축하며, 89.32%의 정밀도를 기록했습니다. 기존 최첨단 방법들에 비해 20% 이상 향상된 정밀도를 달성했습니다.



### StateAct: State Tracking and Reasoning for Acting and Planning with Large Language Models (https://arxiv.org/abs/2410.02810)
Comments:
          9 pages, 5 pages appendix, 7 figures, 5 tables

- **What's New**: 새로운 연구에서는 대형 언어 모델(LLMs)을 사용하는 '실제' 작업 해결을 위한 계획 및 행동 기술에 대한 간단한 방법인 'StateAct'를 제안합니다. 이 방법은 상태 추적(state-tracking)을 통해 LLM의 'chain-of-thought'를 강화하여 더욱 장기적인 문제 해결이 가능합니다.

- **Technical Details**: StateAct는 few-shot in-context learning을 기반으로 하여 에이전트의 목표(예: 위치 및 인벤토리)를 지속적으로 추적합니다. 새 기술은 Alfworld에서 평가되며, 기존 방법보다 14% 성능 향상을 이루었습니다. 추가적인 데이터나 도구 없이도 성능이 동등한 수준을 유지합니다.

- **Performance Highlights**: StateAct는 여러 LLM에서 효율적으로 작동하며, 작업 해결에 필요한 단계 수를 줄이고 장기적인 문제 해결 능력을 향상시킵니다.  최첨단 성능에 도달한 결과는 LLM 분야에서 중요한 진전을 나타냅니다.



### A Data Envelopment Analysis Approach for Assessing Fairness in Resource Allocation: Application to Kidney Exchange Programs (https://arxiv.org/abs/2410.02799)
- **What's New**: 신장 이식 프로그램의 공정성을 평가하기 위해 Data Envelopment Analysis (DEA) 기반의 새로운 프레임워크를 제안하였습니다. 이 모델은 Priority, Access, Outcome 세 가지 공정성 기준을 단일 모델 내에서 동시에 평가합니다.

- **Technical Details**: 우리의 모델에서는 Priority Fairness를 대기 리스트 기간으로, Access Fairness를 Kidney Donor Profile Index 점수로, Outcome Fairness를 이식 생존 기간으로 측정합니다. DEA 모델에 conformal prediction 기법을 적용하여 불확실성을 정량화하고 그룹 조건부 예측 구간을 제공합니다.

- **Performance Highlights**: 분석 결과, 아시아 환자들은 더 긴 대기 리스트 기간을 가지지만 더 나은 이식 생존 결과를 나타내며, 반면 백인 환자들은 짧은 대기 시간에도 불구하고 높은 이식 거부 위험을 경험합니다. 다양한 인종 그룹 간의 신장 할당 효율성에서 현저한 차이를 보여주었습니다.



### DifFaiRec: Generative Fair Recommender with Conditional Diffusion Mod (https://arxiv.org/abs/2410.02791)
Comments:
          The paper was accepted by ICDM 2024

- **What's New**: 이 논문에서는 사용자 선호도 기반으로 공정한 추천을 제공하는 Diffusion-based Fair Recommender (DifFaiRec)라는 새로운 추천 알고리즘을 제안합니다. 이 알고리즘은 조건부 확산 모델(conditional diffusion model)을 기반으로 하여 사용자 선호도의 분포를 효과적으로 학습하고 다양한 추천을 생성할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: DifFaiRec는 두 개의 모듈로 구성됩니다. 첫 번째 모듈은 사용자를 서로 다른 그룹으로 매핑하여 그룹 공정성을 보장하는 역사실 모듈(counterfactual module)입니다. 두 번째 모듈은 이 역사실 모듈에 조건화된 조건부 확산 모델(conditional diffusion model)로, 관찰된 상호작용을 재구성하고 생성을 통해 알려지지 않은 상호작용을 예측합니다. 또한, 정확도(accuracy)와 공정성(fairness)의 두 목적을 하나로 압축하여 최적화 문제의 난이도를 줄입니다.

- **Performance Highlights**: 실험 결과, DifFaiRec는 두 개의 실제 데이터셋에서 기존의 여러 기준선(baselines)을 초월하고 정확성과 공정성을 동시에 유지하는 뛰어난 성능을 보였습니다.



### Raising the Bar(ometer): Identifying a User's Stair and Lift Usage Through Wearable Sensor Data Analysis (https://arxiv.org/abs/2410.02790)
Comments:
          submitted to iWOAR 2024

- **What's New**: 본 연구에서는 건강한 생활 방식 및 비만과 관련된 건강 문제를 감소시키기 위해 착용 가능한 기술( wearable technology)을 활용하여 계단 및 엘리베이터 사용 패턴과 행동을 조사하는 새로운 탐색적 데이터셋을 소개합니다. 이를 통해 사용자의 활동을 정밀하게 추적하고 동기를 부여하여 건강 통찰력을 제공합니다.

- **Technical Details**: 데이터셋은 20명의 참가자로부터 수집되었으며, 이들은 다양한 상황에서 계단을 오르고 내리며 엘리베이터를 활용하는 데이터를 확보하였습니다. 수집된 데이터는 Random Forest 머신 러닝 모델을 훈련 및 테스트하는 데 사용되었고, 8초 간의 시간 창( time windows)에서 계단과 엘리베이터 작업을 분류하는 정확도는 87.61%, 멀티 클래스 가중 평균 F1-score는 87.56%를 기록했습니다.

- **Performance Highlights**: 인르셜(inertial) 및 압력 센서(pressure sensors)의 조합이 실시간 활동 감지(activity detection)를 위한 효과적인 솔루션이라는 것을 확인했습니다. 이러한 결과는 착용 가능한 센서 데이터를 활용한 건강 모니터링의 가능성을 제시합니다.



### Guess What I Think: Streamlined EEG-to-Image Generation with Latent Diffusion Models (https://arxiv.org/abs/2410.02780)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 뇌파로부터 이미지를 생성하는 기술이 주목받고 있으며, 이는 brain-computer interface (BCI) 시스템을 발전시키는 데 기여할 수 있다. 기존 연구들은 주로 고해상도 이미지를 생성할 수 있는 fMRI에 의존했으나, 이 연구에서는 저렴하고 비침습적인 electroencephalography (EEG)를 활용하여 실시간 BCI 응용에 적합한 새로운 방법을 제안하고 있다.

- **Technical Details**: 본 논문에서는 ControlNet 어댑터를 기반으로 한 효율적인 latent diffusion model (LDM) 프레임워크를 제안한다. 이 방법은 EEG 신호를 조건화하여 이미지 생성 과정을 간소화하며, 복잡한 전처리나 다양한 손실 함수, 또는 캡셔닝 모델 없이도 이미지를 생성할 수 있다. 실험을 통해 최첨단 모델들과 비교하여 우수한 성능을 입증하였다.

- **Performance Highlights**: 제안된 GWIT 프레임워크는 minimal preprocessing과 효율적인 트레이닝을 통해 EEG로부터 이미지를 생성하는 데 성공하였다. 본 연구는 EEG 데이터를 사용하여 image generation을 위한 ControlNet의 최초 응용을 보여주며, 기존의 GAN 기반 방법들보다 뛰어난 성능을 보였다.



### Learning variant product relationship and variation attributes from e-commerce website structures (https://arxiv.org/abs/2410.02779)
- **What's New**: 이 논문에서는 VARM(variant relationship matcher) 전략을 소개하여 전자상거래 카탈로그에서 변형된 제품 쌍을 식별하는 방법을 제안합니다. 기존의 엔티티 해상도는 제품 언급이 동일한 기본 제품을 참조하는지를 판단하는 데 중점을 두었으나, 이는 전자상거래 어플리케이션에서 중요한 제품 관계를 포착하지 못합니다.

- **Technical Details**: VARM은 두 가지 요구 사항을 만족시키기 위해 엔코딩(encoding) 및 생성적 AI(Generative AI) 모델의 강점을 활용하는 전략을 개발했습니다. 먼저, 웹페이지의 제품 링크 및 변형된 제품 관계를 포착하는 데이터셋을 구성하여, 주어진 제품 쌍에 대한 변형 매칭을 예측하기 위해 LLM을 훈련합니다. 두 번째로, RAG 기반 생성 LLM을 사용하여 변형된 그룹 간의 변동 및 공통 속성을 추출합니다.

- **Performance Highlights**: 세계적인 전자상거래 소매업체의 실제 데이터를 사용하여 모델 성능을 평가한 결과, 우리의 전략이 대안 솔루션보다 우수한 성능을 나타냈으며, 새로운 유형의 제품 관계를 활용하는 가능성을 제시합니다.



### OATH: Efficient and Flexible Zero-Knowledge Proofs of End-to-End ML Fairness (https://arxiv.org/abs/2410.02777)
- **What's New**: 이 연구는 출시 가능한 Zero-Knowledge Proofs of Fairness (ZKPoF)의 프레임워크인 OATH를 소개합니다. OATH는 실제 세계의 배포에 적합한 첫 번째 ZKPoF 시스템으로, 다양한 인구 집단을 공정하게 서비스할 수 있음을 검증합니다.

- **Technical Details**: OATH는 (i) 클라이언트 인터페이스를 통해 효율적으로 통신할 수 있으며, ML as a Service 쿼리 응답과 유사한 대화식 대화 기능을 제공합니다. (ii) 점수 기반 분류기에 대한 모듈화 유연성을 가지고 있으며, 정확한 추론에 대한 zero-knowledge proof를 필요로 합니다. (iii) 훈련, 추론 및 감사 전반에 걸쳐 기밀성과 공정성을 보장하는 종단간 보안 모델을 지원합니다.

- **Performance Highlights**: OATH는 이전 ZKPoF 모델과 비교하여 1343배의 실행 시간 개선을 달성하며, 수천만 개의 매개변수를 가진 DNN(Deep Neural Networks)와 같은 대규모 모델에 대한 확장이 가능합니다.



### Bypassing the Popularity Bias: Repurposing Models for Better Long-Tail Recommendation (https://arxiv.org/abs/2410.02776)
Comments:
          6 pages, 4 figures

- **What's New**: 이번 연구에서는 온라인 콘텐츠 플랫폼에서 추천 시스템의 공정성을 높이기 위해, 장기 콘텐츠(Long-tail content)를 생산하는 출판사에 대한 노출을 공정하게 분배하는 새로운 방법을 제안합니다. 기존의 추천 시스템 구성 요소를 재활용하여 추천 품질을 유지하면서도 저조한 출판사들에게 더 많은 노출을 제공하는 방법을 도모했습니다.

- **Technical Details**: 이 연구에서는 사용자를 위한 품목 추천의 분석을 통해, 각 출판사가 생산한 품목에 대해 가장 적합한 사용자를 찾는 역추천(Inverse Retrieval) 모델을 도입했습니다. 또한, 장기 콘텐츠를 추천하기 위한 최소 노출 기준을 설정하고, 사용자의 상관성을 높이기 위해 사용자-품목 임베딩을 활용했습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 출판사 간 노출의 공평한 분배를 이루는 데 성공했으며, 추천 시스템의 전반적인 성능 개선 또한 확인되었습니다. 장기적인 응용에서도 긍정적인 결과를 보이며, 추천 품질과 공정성 지표 모두에서 개선된 효과를 나타냈습니다.



### A Deep Learning Approach for User-Centric Clustering in Cell-Free Massive MIMO Systems (https://arxiv.org/abs/2410.02775)
Comments:
          Accepted to 25th IEEE International Workshop on Signal Processing Advances in Wireless Communications (SPAWC), 2024

- **What's New**: 기존의 MIMO 시스템에서의 셀 간 간섭 문제를 해결하기 위한 Cell-free massive MIMO (CF-mMIMO) 시스템이 제안되었습니다. 이 시스템은 사용자와 다수의 엑세스 포인트(Access Points, APs) 간의 최적 연관성을 구하는 복잡한 조합 문제를 해결하도록 딥러닝 접근 방식을 도입했습니다.

- **Technical Details**: 제안된 방법은 Long Short-Term Memory (LSTM) 네트워크를 기반으로 하여 사용자 수의 증가와 관계없이 이상적인 AP-UE 연관성을 가능케 하며, 이는 성능을 향상시키기 위한 중요한 요소입니다. 시스템 모델은 K개의 단일 안테나 사용자 장비(Ω)와 L개의 AP로 구성되어 있으며, 대규모 MIMO 환경에서 작동합니다. 통신은 시간 분할 이중 홉(Time-Division Duplex, TDD) 프로토콜을 통해 이루어지며, 불확실한 채널 상태 정보를 다룰 수 있는 최적화 문제를 해결합니다.

- **Performance Highlights**: 제안된 딥러닝 접근법은 사용자 수의 증가에도 불구하고 재학습(retraining) 없이 효과적으로 작동하며, 실험 결과는 제안된 솔루션의 효과성을 보여주었습니다. 특히, 파일럿 오염(pilot contamination) 문제와 같은 불완전한 채널 상태 정보에서도 성능 저하 없이 시스템의 스펙트럼 효율성을 극대화할 수 있음을 입증하였습니다.



### Estimating the Unobservable Components of Electricity Demand Response with Inverse Optimization (https://arxiv.org/abs/2410.02774)
- **What's New**: 전통적인 기계 학습 및 시계열 분석이 정기적인 수요 패턴에만 적합했던 것을 넘어서, 이 논문은 전기 수요의 개별 소비자 반응을 가격에 기반하여 이해하고 예측하는 새로운 방법인 데이터 기반 역최적화(inverse optimization) 방법론을 제안합니다.

- **Technical Details**: 이 연구는 기계 학습 알고리즘과 시계열 분석이 단순히 관측 가능한 수요에 적용되어 온 반면, 그리드에 접속된 지점에서 측정된 순수요(net demand)를 분석하여 비관측 성분을 추정하는 역최적화 방식을 활용합니다. 이 방법은 소비자의 전기 소비 패턴을 비드레전트(behind-the-meter) 행동이나 장치 수준 데이터 관측 없이도 해석할 수 있게 합니다.

- **Performance Highlights**: 제안된 방법론은 미국 가정의 상세한 장치 수준 행동 데이터를 활용한 두 단계 실험 분석을 통해 우수한 성능을 입증했습니다. 일본의 TOU 가격 데이터에서 IO 방법론이 다른 기준 모델보다 우수한 예측 성능을 보였으며, 비관측 유연성 및 기초 부하 수요의 명확한 추정값을 제공했습니다.



### Efficient Numerical Calibration of Water Delivery Network Using Short-Burst Hydrant Trials (https://arxiv.org/abs/2410.02772)
Comments:
          16 pages, 6 figures, submitted to ASCE Journal of Water Resources Planning and Management

- **What's New**: 이 연구는 물 배급 네트워크(Water Distribution Network, WDN) 수압 모델의 보정(calibration) 방법론을 제안하며, 야간 hydrant 시험을 통해 높은 압력 기울기를 실현함으로써 불확실성을 줄이고 더 효과적인 보정 결과를 제공합니다.

- **Technical Details**: 제안된 방법은 짧은 시간에 걸쳐 야간 hydrant 시험을 실시하고, 이러한 데이터를 시간 단위 소비 패턴에 맞춰 재샘플링(resampled)하여 가시적인 변화 없이 고급 보정 알고리즘을 사용하는 것입니다. 이 연구는 두 가지 최신 보정 알고리즘을 활용하여 절대 오차(absolut error)를 최대 45% 줄일 수 있는 성과를 도출했습니다.

- **Performance Highlights**: 우리는 제안된 방법이 전통적인 주간 보정 방법에 비해 동등하거나 더 나은 수압 모델 보정 결과를 달성했음을 입증하였습니다. 구체적으로, 야간에서 몇 분 간의 짧은 흐름 유도 펄스(k), 시간 단위의 물 소비 데이터와 결합하여보정이 이루어졌으며, 이는 실험 지역의 복잡한 WDN에서 매우 효과적이었습니다.



### Insightful Railway Track Evaluation: Leveraging NARX Feature Interpretation (https://arxiv.org/abs/2410.02770)
Comments:
          In English. CBA 2024 - XXV Brazilian Congress of Automation (CBA - XXV Congresso Brasileiro de Automática)

- **What's New**: 이 논문은 NARX 방법론을 로지스틱 회귀(Logistic Regression)와 결합한 새로운 분류 알고리즘인 Logistic-NARX Multinomial을 소개합니다.

- **Technical Details**: Logistic-NARX Multinomial은 다중 클래스 분류(multiclass classification) 문제를 해결하기 위해 NARX 모델을 사용하며, 복잡한 프로세스를 이해하는 데 도움을 주는 파라메트릭 모델링(parametric modeling) 기법을 활용합니다.

- **Performance Highlights**: 철도(railway) 분야를 위한 혁신적인 방법론을 제시하며, 온보드 센서(onboard sensors)에서 파생된 다양한 특성(feature)의 해석을 통해 안전 및 유지보수에 대한 정보 기반 의사결정을 지원합니다.



### Bayes-CATSI: A variational Bayesian deep learning framework for medical time series data imputation (https://arxiv.org/abs/2410.01847)
- **What's New**: 본 논문에서는 Bayesian Context-Aware Time Series Imputation (Bayes-CATSI) 프레임워크를 제안하여 기존 CATSI 모델에 비해 불확실성 정량화를 통합하여 성능을 향상시킵니다.

- **Technical Details**: Bayes-CATSI는 Electroencephalography (EEG), Electrooculography (EOG), Electromyography (EMG), Electrocardiology (EKG)로부터 파생된 시계열 데이터를 사용합니다. 이 모델은 변별적 추론(Variational Inference)을 통해 사후 분포(Posterior Distribution)의 형상을 가정하고 Kullback-Leibler (KL) 발산을 최소화하여 진정한 사후 분포에 가까운 변별 밀도를 찾아냅니다. 또한, Bayes-CATSI는 기존 CATSI 모델 아키텍처에 사용자 정의된 Bayesian 딥 러닝 계층을 통합하여, 모든 결정론적 딥 러닝 계층을 Bayesian 딥 러닝 계층으로 대체합니다.

- **Performance Highlights**: Bayes-CATSI는 CATSI 모델에 비해 9.57% 향상된 성능을 보여줍니다. 또한, 통합된 Bayesian 계층의 활용 덕분에 불확실성 정량화뿐만 아니라 데이터 임퓨테이션 성능 면에서도 뛰어난 결과를 나타냅니다.




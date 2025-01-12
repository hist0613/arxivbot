New uploads on arXiv(cs.CL)

### EpiCoder: Encompassing Diversity and Complexity in Code Generation (https://arxiv.org/abs/2501.04694)
Comments:
          40 pages, 11 figures

- **What's New**: 이 연구는 코드 LLMs에 최적화를 위한 효과적인 instruction tuning의 중요성을 강조합니다. 기존 방법들이 코드 조각(code snippets)에 의존하여 특정 기능에 국한됨에 따라 데이터의 복잡성과 다양성이 제한되고 있다는 점에서 새로운 피쳐 트리 기반 합성 프레임워크를 제안합니다. 이 프레임워크는 Semantic Relationship을 모델링하여 더 정교하고 다양한 데이터를 생성할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: 제안된 방법의 주요 구성 요소는 피쳐 트리 추출, 진화, 및 코드 생성의 세 가지 단계로 나뉩니다. 피쳐 트리는 raw 데이터로부터 구성되며, 이들 간의 semantic 관계를 포착하기 위해 iterative clustering이 사용됩니다. 이를 통해 코드 생성 시 다양성과 복잡성을 조절할 수 있는 제어 가능성을 가지며, 다양한 수준의 작업을 포함합니다.

- **Performance Highlights**: EpiCoder 시리즈를 통해 다수의 벤치마크에서 기능 및 파일 수준에서 최첨단 성능을 보여주며, 특히 복잡한 저장소 수준의 데이터를 합성하는 능력이 두드러집니다. 433k의 instruction 데이터를 합성하고 EpiCoder-Qwen-7B 모델을 통해 맞춤형 학습을 실시하여 높아진 성능을 실현했습니다. 이를 통해 다양한 프로그래밍 문제를 해결할 수 있는 잠재력을 입증하였습니다.



### URSA: Understanding and Verifying Chain-of-thought Reasoning in Multimodal Mathematics (https://arxiv.org/abs/2501.04686)
Comments:
          27 pages, 10 tables, 17 figures. The training data has been released. The code and model are currently undergoing internal review. They will be made available soon. Project url: this https URL

- **What's New**: 본 논문에서는 Chain-of-thought (CoT) 추론을 다룬 새로운 접근법을 제안합니다. 특히, 다중 모드 수학적 추론에서 CoT 훈련 데이터의 부족 문제를 해결하기 위해 3개의 모듈 합성 전략을 도입하였습니다. 이를 통해 MMathCoT-1M이라는 고품질 CoT 추론 지침 데이터셋을 생성하고, URSA-7B 모델의 성능을 여러 벤치마크에서 검증했습니다.

- **Technical Details**: 제안된 시스템은 CoT 증류(Cot Distillation), 궤적 형식 재작성(Trajectory-format rewriting), 및 형식 통합(Format unification)을 포함한 세 가지 모듈로 구성됩니다. 이 과정을 통해 고품질 CoT 추론 데이터셋이 생성되며, URSA-7B 모델은 DualMath-1.1M이라는 데이터 합성 전략을 통해 향상된 성능을 나타냅니다. 모델은 특히 다중 모드 정보 처리 과정에서 발생할 수 있는 오류를 국지화할 수 있는 새로운 방법론을 지니고 있습니다.

- **Performance Highlights**: URSA-7B 모델은 MathVista, MathVerse, WE-MATH 등 여러 다중 모드 수학 벤치마크에서 SOTA 성능을 달성하였습니다. 또한, URSA-RM-7B 모델은 URSA-7B의 검증기로 작동하여 테스트 시간 동안 더욱 향상된 성능을 보여주고 있습니다. 궁극적으로, 이 연구는 다중 모드 수학적 추론에서 모델의 성능 한계를 효과적으로 높이는 데 기여하고 있습니다.



### Enhancing Financial VQA in Vision Language Models using Intermediate Structured Representations (https://arxiv.org/abs/2501.04675)
- **What's New**: 이 연구는 50,000개의 막대 차트에 대해 고유한 구조적 특성을 활용하여 차트 이미지를 선형화된 테이블로 변환하는 DEPLOT(모드 전환 모듈)의 미세 조정을 조사합니다. 미세 조정된 DEPLOT 모델은 카테고리별 매핑 정확도를 측정하는 Relative Mapping Similarity(RMS)와 수치적 해석 정확도를 평가하는 Relative Number Set Similarity(RNSS)를 통해 기본 모델과 비교 평가됩니다. 또한, 100개의 차트 이미지와 질문-응답 세트를 추가하여 대규모 언어 모델(LLMs)의 추론 능력을 탐구합니다.

- **Technical Details**: DEPLOT은 시각 차트 데이터를 구조화된 데이터 테이블로 매핑하기 위한 모드 전환 모듈로, 다양한 차트 유형에서 훈련되지만 도메인별 데이터 세트를 사용하여 미세 조정할 수 있습니다. 본 논문에서는 RNSS와 RMS 두 가지 주요 지표를 통해 모델의 정량적 및 범주적 해석 능력을 평가하며, 차트 구조를 추적할 수 있는 능력을 강조합니다. 이러한 새로운 접근법은 DEPLOT의 성능을 높이고, 보다 신뢰할 수 있는 데이터 시각화 모델 개발을 위한 기초를 제공합니다.

- **Performance Highlights**: 미세 조정된 DEPLOT 모델을 활용한 실험 결과, 높은 품질의 구조화된 데이터와 함께 제공된 경우 LLM의 추론 능력이 크게 향상됨을 보여줍니다. 특히 Qwen2-VL-7B와 같은 소형 모델이 고급 모델인 GPT-4o보다 더 나은 성능을 발휘하여 차트 데이터 해석의 정확성을 높였습니다. 이 연구는 자동 차트 해석 및 추론 향상을 위한 모드 전환 통합의 혁신적 잠재력을 강조합니다.



### On The Origin of Cultural Biases in Language Models: From Pre-training Data to Linguistic Phenomena (https://arxiv.org/abs/2501.04662)
- **What's New**: 이 논문에서는 멀티링구얼 언어 모델(Multilingual Language Models, LMs)이 비서구 언어에서 서구 문화와 관련된 개체에 편향된 행동을 보이는 현상을 분석했습니다. 특히, 아랍어와 영어간의 편향을 줄이기 위한 새로운 병렬 기준인 CAMeL-2를 소개합니다. CAMeL-2는 아랍 문화와 서구 문화에 관련된 58,086개의 개체와 367개의 자연 맥락을 포함하고 있습니다.

- **Technical Details**: 논문의 분석을 통해 LMs가 아랍어에서 자주 등장하는 개체를 인식하는 데 어려움을 겪고 있음을 발견했습니다. 이는 아랍어에서 다의어(Polysemy)로 나타나는 경우와 높은 빈도를 가진 개체와 관련이 있습니다. 또한, 아랍어 스크립트를 사용하는 다른 언어들과의 어휘 중복이 LM의 성능 저하를 유발하는 요인으로 나타났습니다. 더욱이, 대규모 아랍어 어휘에서 토큰화(Tokenization) 문제가 추가적인 어려움을 초래하는 것으로 확인되었습니다.

- **Performance Highlights**: CAMeL-2를 활용한 평가에서는 LMs가 영어로 테스트할 때 아랍어보다 문화간 성능 차이가 더 적다는 것을 보여주었습니다. 이는 아랍 문화와 관련된 개체를 인식하는 데 있어 LMs의 성능 차이를 선명하게 드러내는 결과를 나타냅니다. 이 연구는 언어 모델의 선입견을 이해하고 해결하기 위한 방향성을 제시하고 있으며, 다양한 문화적 맥락에서 AI의 응용 가능성을 향상시키는 데 기여할 것으로 기대됩니다.



### Assessing Language Comprehension in Large Language Models Using Construction Grammar (https://arxiv.org/abs/2501.04661)
- **What's New**: 이번 연구는 Construction Grammar (CxG)를 활용하여 대규모 언어 모델(LLMs)의 자연어 이해(NLU)를 체계적으로 평가하는 새로운 방법론을 제시합니다. 평가 작업은 CxG의 8가지 고유한 Cxn을 바탕으로 하여 LLM의 언어 이해를 인간의 이해와 비교합니다. 특히, 일반적으로 훈련 데이터에 나타나지 않는 문장 예제를 포함하여 LLM의 언어 이해의 한계를 강조합니다.

- **Technical Details**: CxG는 형태소, 단어, 관용구 및 언어의 도식적 구조가 형태-의미 쌍, 즉 construction (Cxn)으로 표현될 수 있다는 이론적 기초를 제공합니다. 본 연구는 이러한 Cxn의 의미를 이해해야하는 자연어 추론(NLI) 작업을 통해 LLM의 이해 능력을 평가하며, 이는 LLM의 진정한 언어 이해를 측정하는 데 필수적입니다. LLM의 성능은 일반적으로 훈련 데이터에서 등장하지 않는 예제에 대한 이해를 요구하는 도전 과제를 통해 평가됩니다.

- **Performance Highlights**: 실험 결과, 최신 모델인 GPT-4o를 포함한 LLM은 Cxn이 전달하는 추상적인 의미를 이해하는 데 어려움을 겪는 것으로 나타났습니다. LLM은 constructual 정보에 대한 일부 지식은 보유하고 있지만, 실제로는 통계적으로 예상되는 예제와는 다른 테스트 문장에서 어려움을 보여줍니다. 이러한 결과는 LLM의 의미적 한계를 강조하며, 진정한 언어 이해의 평가 측면에서 중요합니다.



### Multi-task retriever fine-tuning for domain-specific and efficient RAG (https://arxiv.org/abs/2501.04652)
Comments:
          9 pages, 2 figures. Submitted to NAACL 2025 Industry Track

- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG) 기술을 활용하여 Large Language Models (LLMs)의 성능을 향상시키기 위한 새로운 접근법을 제시합니다. 특히, 다양한 도메인 특정 작업에 대응할 수 있는 소형 retriever encoder를 instruction fine-tuning 방식으로 학습시켜, 여러 환경에서 활용 가능한 효과적인 솔루션을 제공합니다. 이를 통해 RAG 응용 프로그램의 가능한 확장성을 극대화하고, 비용과 처리 속도를 줄일 수 있습니다.

- **Technical Details**: 연구에서는 mGTE 모델을 fine-tune하여 생성된 데이터셋을 기반으로 소형 retriever를 다양한 작업에 맞춰 훈련합니다. retriever는 steps, table names, field names 등 다양한 구조화된 데이터를 데이터베이스에서 검색하여 LLM에 전달하는 방식으로, RAG 응용 프로그램의 결과물 품질을 높입니다. 본 연구는 OOD(Out-Of-Domain) 설정에서 retriever의 일반화 성능을 평가하며, 학습 데이터셋은 내부 데이터베이스와 Flow Generation training set에서 추출됩니다.

- **Performance Highlights**: 제안된 방법은 다양한 도메인과 관련된 과제를 해결하는 데 뛰어난 성능을 발휘합니다. 실험 결과, instruction fine-tuning을 통한 retriever 모델이 OOD 설정에서도 유의미한 성능 향상을 보여주며, 차별화된 data retrieval 과제를 해결할 수 있는 능력을 입증했습니다. 이를 통해, GenAI 응용 프로그램의 다양성과 효율성을 높이는 새로운 가능성을 열어줍니다.



### Quantum-inspired Embeddings Projection and Similarity Metrics for Representation Learning (https://arxiv.org/abs/2501.04591)
- **What's New**: 이번 연구는 양자 영감을 받은 프로젝션 헤드를 제안하며, 이는 기존의 내포된 정보 맵핑 방식을 개선하는 혁신적인 접근법입니다. 이 새로운 프로젝션 헤드는 고전적인 임베딩을 양자 상태로 매핑하고, 효율성 높은 양자 회로 기반 아키텍처를 사용하여 임베딩 차원을 축소합니다. 또한, 이 연구에서는 BERT 언어 모델을 통해 해당 접근법의 효용성을 검증하고, 정보 검색 작업에서 활용되는 성능을 비교합니다.

- **Technical Details**: 연구팀은 한쪽 눈의(Qubit) 단일 및 이중 게이트로 구성된 양자 회로 기반 프로젝션 헤드를 사용하여 BERT 임베딩을 더 낮은 차원으로 압축합니다. 이 과정에서 기존 고전적인 방법과 비교하여 32배 적은 매개변수로도 경쟁력을 유지할 수 있음을 보여줍니다. 특히, 소규모 데이터 세트에서의 성능이 뛰어나며, 이를 통해 고전적인 방법보다 우수한 임베딩 압축 성능을 입증합니다.

- **Performance Highlights**: 실험 결과는 양자 영감을 받은 프로젝션 헤드가 고전적인 접근법보다 우수한 성능을 보이며, 특히 데이터가 부족한 시나리오에서 효과적이라는 것을 보여줍니다. 이 연구는 또한 효율적인 저엉겅트 회로 시뮬레이션이 신경망 내에서 강력한 양자 영감을 준 기법으로 이동될 수 있음을 강조합니다. 따라서 양자 컴퓨팅의 최근 발전과 함께 본 접근법은 앞으로의 머신러닝 연구에 중요한 기여를 할 것으로 기대됩니다.



### OpenOmni: Large Language Models Pivot Zero-shot Omnimodal Alignment across Language with Real-time Self-Aware Emotional Speech Synthesis (https://arxiv.org/abs/2501.04561)
- **What's New**: 최근 오믹모달(omnimodal) 학습의 발전으로 이미지, 텍스트, 음성 간의 이해 및 생성이 가능해졌습니다. 하지만, 제한된 오믹모달 데이터셋과 실시간 감정 음성 생성의 어려움으로 인해 오픈소스 진전이 제약받고 있습니다. 이를 해결하기 위해 openomni라는 두 단계 훈련 방법을 제안하여 최첨단 오믹모달 대형 언어 모델을 개발합니다.

- **Technical Details**: openomni는 두 단계로 구성된 훈련 방식을 채택합니다. 첫 번째는 오믹모달 정렬(omnimodal alignment)으로, 사전 훈련된 음성 모델을 텍스트-이미지 작업에 추가로 훈련하여 비전에서 음성으로의 일반화를 달성합니다. 두 번째 단계는 실시간 감정 음성을 생성하기 위한 경량 디코더를 사용하여 음성 작업과 선호 학습에 대해 훈련을 수행합니다.

- **Performance Highlights**: openomni는 오믹모달, 비전-언어 및 음성-언어 평가에서 일관된 성능 향상을 보여줍니다. 기존의 대형 오픈소스 모델과 비교할 때, openomni는 훨씬 적은 모델 크기 및 훈련 데이터로 OmniBench 벤치마크에서 탁월한 성능을 기록하며, 실시간 음성 생성, 음성 이해 및 이미지-텍스트 질문 응답과 같은 다양한 이중 모달 작업에서도 경쟁력 있는 결과를 도출합니다.



### rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking (https://arxiv.org/abs/2501.04519)
- **What's New**: rStar-Math는 작은 언어 모델(SLMs)이 OpenAI의 o1이 아닌 자체적으로 수학 문제 해결 능력을 겨룰 수 있음은 물론 때로는 능가할 수 있음을 보여준다는 점이 혁신적입니다. 이 시스템은 Monte Carlo Tree Search (MCTS)를 통해 '깊은 사고'를 활용하여 수학 정책 SLM이 보상 모델에 의해 안내된 테스트 시간 탐색을 수행합니다. 주요 혁신으로는 코드 보강(CoT) 데이터 합성 방법, 프로세스 보상 모델 훈련 기법, 그리고 정책 SLM과 PPM의 자가 진화 전략이 포함됩니다.

- **Technical Details**: rStar-Math는 세 가지 주요 혁신을 통해 수학 추론 성능을 개선합니다. 첫째, MCTS를 사용하여 단계별 검증된 추론 경로를 생성하는 코드 보강(CoT) 데이터 합성 방법을 적용합니다. 둘째, 프로세스 선호 모델(PPM)을 훈련하여 수학 추론 단계의 보상 레이블을 예측하고, 정확한 단계별 보상을 제공하는 방법을 개발합니다. 마지막으로, 자가 진화 생태계를 통해 단계적으로 더욱 강력한 정책 SLM과 PPM을 구축하고, 반복적인 훈련 데이터 품질 개선을 달성합니다.

- **Performance Highlights**: rStar-Math는 MATH 벤치마크에서 Qwen2.5-Math-7B 모델의 성능을 58.8%에서 90.0%로 향상시켰으며, Phi3-mini-3.8B 모델 역시 41.4%에서 86.4%로 개선시켰습니다. AIME 대회에서는 평균적으로 문제의 53.3%(8/15)를 해결하여 상위 20%에 들어가는 성과를 보였습니다. 이는 기존 OpenAI 모델 등의 수학 문제 해결 능력을 초월하는 것을 나타내며, 전체적인 SLM 성능에 커다란 기여를 하고 있습니다.



### PolInterviews -- A Dataset of German Politician Public Broadcast Interviews (https://arxiv.org/abs/2501.04484)
- **What's New**: 이 논문은 독일 고위 정치인들의 공공 방송 인터뷰로 구성된 새로운 데이터셋을 소개합니다. 이 데이터셋은 YouTube에서 소스를 확보하고, 발화자 식별을 위한 전처리를 거쳐 깔끔하고 공개된 형식으로 저장되었습니다. 99개의 인터뷰와 33명의 정치인이 포함되어 있어 정치 커뮤니케이션의 다양한 측면에 대한 연구 기회를 제공합니다.

- **Technical Details**: 데이터셋은 2020년부터 2024년 사이의 유럽국가 독일의 골격을 이루는 정치 담론을 이해하기 위해 고위 정치인을 대상으로 한 99개의 인터뷰를 포함하고 있습니다. 궁극적으로 Whisper 전사 모델로 음성 파일을 전사하고, ECAPA-TDNN 모델을 활용해 발화자 다이어리제이션을 수행하여 인터뷰의 발화자를 식별했습니다. 데이터셋은 NLP 작업을 위한 품질 기준을 만족시키도록 수작업 검토를 거쳐 최종적으로 저장되었습니다.

- **Performance Highlights**: 이 데이터를 통해 정치적 의사소통의 다양한 요소를 분석할 수 있으며, 이를 통해 독일 정치의 주요 트렌드와 변화를 포착할 수 있습니다. 앞으로의 연구에서는 수집된 데이터를 바탕으로 감정 인식 및 의미 유사성 측정과 같은 컴퓨팅 기법을 활용할 수 있는 새로운 기회를 제공할 것입니다. 이 데이터셋은 정치 커뮤니케이션 분야에서의 중요한 연구 도구로 자리매김할 것입니다.



### When LLMs Struggle: Reference-less Translation Evaluation for Low-resource Languages (https://arxiv.org/abs/2501.04473)
- **What's New**: 이 논문에서는 저자원이 부족한 언어 쌍의 기계 번역 품질에 대한 reference-less (참조 없음) 평가인 Quality Estimation (QE)을 조사합니다. 세그먼트 단위의 QE는 기계 번역 결과물에 품질 점수(0-100)를 제공하는 도전적인 과제입니다. 본 연구는 제로 샷(zero-shot), 몇 샷(few-shot)/상황에서의 학습(in-context learning), 그리고 어댑터를 이용한 지시적 미세 조정(instruction fine-tuning)을 수행하며, 주목할 만한 성과를 도출했습니다.

- **Technical Details**: 저자들은 WMT QE 공동 작업에 있는 저자원 언어 쌍을 대상으로 하여, 고유한 주석 지침 기반 프롬프트(AG-prompt)를 활용해 세그먼트 수준 QE를 평가하였습니다. 실험은 독립적인 언어 쌍 훈련(ILT)과 다국적 훈련(UMT) 설정에서 진행하였으며, 다양한 LLMs 모델을 사용하여 품질 평가의 도전을 측정했습니다.

- **Performance Highlights**: 연구 결과, AG-prompt를 통해 제로 샷 성능이 향상됨을 보였으며, LLM의 품질 평가 시 발생하는 문제들을 분석했습니다. 특히, 저자원 언어에서의 LLM들 간의 성능 차이를 강조했으며, LLM의 사전 훈련을 개선할 필요성이 있음을 주장했습니다. 논문은 연구를 지속할 수 있도록 훈련된 데이터와 모델을 공개합니다.



### Hidden Entity Detection from GitHub Leveraging Large Language Models (https://arxiv.org/abs/2501.04455)
Comments:
          accepted by KDD2024 workshop DL4KG

- **What's New**: 본 논문은 비정형 데이터 소스에서 지식 베이스를 구축하는 과정에서 중요한 역할을 하는 Named Entity Recognition (NER) 작업에 관한 연구다. 기존 방법들이 대량의 훈련 데이터에 의존했으나, 최근 LLMs(대규모 언어 모델)의 발전으로 인해 Zero-shot Learning (ZSL) 및 Few-shot Learning (FSL)에 기초한 새로운 접근 방식이 가능해졌다. 저자들은 GitHub 리포지토리의 텍스트 콘텐츠에서 데이터 세트와 소프트웨어를 자동으로 탐지하는 방법을 제안하고, URL로 표현된 자원까지 포함하여 NER의 범위를 확장한다.

- **Technical Details**: 이 연구는 LLM을 활용하여 GitHub 리포지토리의 README 페이지에서 소프트웨어와 데이터 세트를 자동으로 추출하는 작업을 다룬다. 연구에서는 LLaMA 2와 Mistral 7B 같은 LLM 모델이 사용되며, 이를 통해 URL 추출 및 분류(Extraction and Classification, E+CL) 작업을 수행한다. 또한, 다양한 FSL 프롬프트 학습 접근 방식을 탐색하여 LLM의 데이터 및 소프트웨어 언급 인식 능력을 향상시키려고 한다.

- **Performance Highlights**: 결과적으로, 이 연구는 GitHub 리포지토리에서 811,811개의 URL과 그 맥락을 포함한 수작업 주석 데이터셋을 생성하였으며, LLMs의 유효성을 평가한다. LLaMA 2는 특정 벤치마크에서 GPT-3를 초월했으며, Mistral 7B는 보다 빠른 추론을 위한 다양한 기술을 사용한다. 최종적으로, 고급 언어 모델을 통한 자동화된 엔터티 탐지의 잠재력을 보여주며, 짧은 훈련 데이터 환경에서도 액세스 가능성을 높이는 계기를 마련한다.



### End-to-End Bangla AI for Solving Math Olympiad Problem Benchmark: Leveraging Large Language Model Using Integrated Approach (https://arxiv.org/abs/2501.04425)
- **What's New**: 이 연구는 방글라어 AI 수학 문제를 해결하기 위한 대형 언어 모델(LLMs)의 성능 향상을 위한 체계적인 접근 방식을 도입합니다. 다양한 LLM 구성과 특정 데이터 세트로 세부 조정(fine-tuning)을 통해 모델의 추론 정밀도를 향상시키고자 하며, Retrieval-Augmented Generation (RAG)을 적용하여 다국어 환경에서의 효율성을 높였습니다. 특별히 맞춤형 프롬프트, 데이터 세트 증대, 반복적 추론이 올림피아드 수준의 수학 문제 해결에 있어 매우 중요하다는 것을 발견했습니다.

- **Technical Details**: 이 연구에서는 방글라어로 수학 문제 해결을 최적화하기 위해 여러 개의 LLM을 평가하여 가장 적합한 모델을 선택했습니다. 사용한 데이터 세트에 대해서 효과적으로 작동한 모델들을 중심으로 실험을 진행했으며, 특히 Qwen2.5-7B-Instruct 모델이 방글라어 수학 문제를 이해하고 잘 해결함을 확인했습니다. 또한, 데이터 세트를 증대시키기 위해 OpenAI의 GPT-4.0를 활용하여 원래 문제의 바리에이션을 생성하였으며, Retrieval-Augmented Generation (RAG) 기법을 통해 문제 해결 능력을 개선했습니다.

- **Performance Highlights**: 각 모델에 대한 성능을 지표로 평가한 결과, Qwen2.5-32B-Instruct-AWQ가 가장 높은 정확도 77을 기록하며 최상위 성능을 보였습니다. 또한 MetaMath-7B 모델과 같은 새로운 모델들은 기존의 오픈 소스 LLM들보다 뛰어난 성능을 보여주며, GSM8K 및 MATH 벤치마크에서 눈에 띄는 결과를 나타냈습니다. 하지만 최근 평가에서는 현대 모델들이 평균 초등학교 3학년 학생들의 점수보다 크게 낮은 점수를 기록하며, 향후 모델 개선 필요성이 제기되었습니다.



### SEO: Stochastic Experience Optimization for Large Language Models (https://arxiv.org/abs/2501.04393)
- **What's New**: 이 논문에서는 Stochastic Experience Optimization (SEO)라는 새로운 자동 경험 최적화 프레임워크를 제안합니다. 이 접근 방식은 모델 파라미터를 수정하지 않고도 특정 LLM에 맞는 경험을 찾는 데 중점을 둡니다. SEO는 자연어를 통해 경험을 업데이트하여 의미 있는 최적화를 가능하게 합니다.

- **Technical Details**: SEO는 기계 학습 시스템의 학습 과정을 모델링하여 설계되었습니다. SEO 과정은 주로 생성 모델(Mgᵉⁿ)과 최적화 모델(Moᵖᵗ)의 두 가지 주요 구성 요소로 이루어져 있으며, 초기 경험을 바탕으로 반복적으로 업데이트합니다. 경험 업데이트는 기계 학습에서 사용되는 "stochastic gradient descent"(SGD) 접근 방식을 따릅니다.

- **Performance Highlights**: SEO를 통해 얻어진 경험은 세 가지 작업, 즉 multi-hop Question Answering (QA), Machine Translation (MT), 및 Text Classification에서 다양한 LLM의 성능을 일관되게 향상시킵니다. 추가 분석 결과, SEO로 최적화된 경험은 유사한 작업에서 out-of-distribution 데이터에 대해 일반화할 수 있는 잠재력을 보여 줍니다.



### Understanding Before Reasoning: Enhancing Chain-of-Thought with Iterative Summarization Pre-Prompting (https://arxiv.org/abs/2501.04341)
- **What's New**: 이 논문에서는 Iterative Summarization Pre-Prompting (ISP²)이라는 새로운 전단계 프롬프트 방법을 제안합니다. 이는 대규모 언어 모델(LLM)의 추론 성능을 보다 향상시키기 위해 설계되었습니다. ISP²는 키 정보가 명시적으로 제공되지 않을 때 LLM의 추론을 개선하는 데 도움을 주며, 복잡한 문제를 해결하는 데 효과적입니다. 실험 결과 기존 방법에 비해 7.1%의 성능 향상을 보여주었습니다.

- **Technical Details**: ISP²는 후보 정보의 적응적 추출, 정보 쌍의 신뢰도 평가 및 반복 요약이라는 세 가지 주요 단계를 포함합니다. 이 과정은 기존 문제 공간에 대한 처리 및 이해를 위한 정보를 요약하고 통합하며, 복잡한 문제를 해결하기 위한 전략을 수립합니다. ISP²는 CoT(Cycle of Thought)와 결합되어 LLM의 성능을 높이는 새로운 접근 방식을 제공하며, 다양한 추론 프레임워크에 유연하게 통합될 수 있습니다.

- **Performance Highlights**: ISP²는 GPT-3.5 Turbo와 같은 다양한 모델에서 뛰어난 성능을 보이며, CoT 및 복합 CoT 전환에 따른 성능 향상이 각각 7.1% 및 8.1%로 나타났습니다. ISP²와 CoT의 평균 성능 점수는 79.43으로, 다른 SOTA(최첨단) 방법들을 초월했습니다. 이러한 결과는 ISP²가 다양한 추론 환경에서 효과적임을 입증하며, 기존의 플러그 앤 플레이 방법들 중 최고 성능을 기록하였습니다.



### Who Does the Giant Number Pile Like Best: Analyzing Fairness in Hiring Contexts (https://arxiv.org/abs/2501.04316)
- **What's New**: 이번 연구는 대형 언어 모델(LLM)을 활용한 채용 시스템의 공정성을 평가하기 위해 이력서 요약 및 검색을 포함하는 두 가지 실제 작업을 조사합니다. 특히, 의미 있는 인구통계학적 변이가 있는 이력서에서 LLM의 출력이 어떻게 다른지를 분석함으로써, 인종과 성별에 따른 차별적인 결정이 발생할 수 있는지를 탐구합니다.

- **Technical Details**: 연구팀은 이력서와 직업 공고의 합성 데이터셋을 구성하고, 요약 결과와 검색에서의 공정성을 평가하기 위해 여러 가지 메트릭을 제안했습니다. 또한, LLM의 출력 차이를 자동으로 측정하기 위해 독서 용이성, 독서 시간, 주관성 등을 포함하는 매개변수를 활용하여 인간의 선호를 반영한 공정성 측정을 시행했습니다.

- **Performance Highlights**: 연구 결과, 인종에 기반한 차이가 생성된 요약의 약 10%에서 나타나는 반면, 성별에 기반한 차이는 단지 1%에 그쳤습니다. 또한, 검색 작업을 통해 모든 모델이 인구통계학적 그룹에 따라 비균일한 선택 패턴을 보이며, 특히 이름 및 인종 변동에 강한 민감성을 나타내는 것으로 관찰되었습니다.결과적으로 LLM 기반의 채용 시스템이 초기 검색 단계에서 상당한 편향을 나타내고 있음을 확인했습니다.



### LLM4SR: A Survey on Large Language Models for Scientific Research (https://arxiv.org/abs/2501.04306)
- **What's New**: 최근 대형 언어 모델(LLMs)의 급속한 발전은 과학 연구의 다양한 단계에서 전례 없는 지원을 제공하며, 연구 과정의 혁신을 가져오고 있습니다. 이 논문은 LLMs가 과학 연구 프로세스를 어떻게 변화시키고 있는지를 조사하기 위한 최초의 체계적인 조사 결과를 제시합니다. 우리는 LLMs의 독특한 역할을 가설 발견(hypothesis discovery), 실험 계획 및 실행(experiment planning and implementation), 과학적 글쓰기(scientific writing), 동료 평가(peer reviewing)의 네 가지 중요한 단계에 걸쳐 분석합니다.

- **Technical Details**: LLMs의 최근 발전으로 AI 및 자연어 처리(NLP) 분야에서 새로운 기준을 설정하는 데 성공하였습니다. GPT-4와 LLaMA와 같은 모델들은 방대한 데이터셋과 혁신적인 아키텍처를 통해 인간 언어와의 상호작용에서 놀라운 성능을 발휘합니다. 이 모델들은 단순한 NLP 작업을 넘어서 더 복잡하고 도메인 특화된 과제에까지 그 능력을 확장하고 있으며, 특히 방대한 데이터를 처리하고 인간과 유사한 텍스트를 생성할 수 있는 능력은 과학 연구 커뮤니티에서 큰 주목을 받고 있습니다.

- **Performance Highlights**: 이 논문은 LLMs가 과학 연구의 수행 방식, 문서화, 및 평가에 혁신을 가져올 수 있는 잠재력을 강조합니다. LLMs는 연구자들이 과학적 질문에 접근하는 방식을 변화시키고, 더 빠르고 효율적인 연구 프로세스를 위해 자동화된 도구를 제공함으로써 연구 생산성을 크게 향상시킬 것입니다. 논문은 현재의 도전과제를 식별하고, 미래 연구 방향을 제안함으로써 LLMs의 혁신적인 가능성을 조명합니다.



### Multimodal Graph Constrastive Learning and Prompt for ChartQA (https://arxiv.org/abs/2501.04303)
- **What's New**: 이 논문에서는 ChartQA라는 차트 이해 과제를 위한 새로운 방법론을 제안합니다. 다중 모달 (multimodal) 장면 그래프를 통해 차트 요소 간의 관계를 명확하게 표현하고, 이를 이용하여 차트 데이터 내의 암묵적인 패턴을 이해합니다. 또한, 다양한 모달리티 간의 통합 표현을 생성하기 위해 대조 학습 (contrastive learning) 방식을 활용하고, 학습된 그래프 표현을 소프트 프롬프트 (soft prompt)로 변환기에 적용합니다.

- **Technical Details**: 제안된 방법은 시각 그래프와 텍스트 그래프의 두 가지 구성 요소로 이루어져 있으며, 이는 차트 내 구조적 및 의미적 정보를 캡처하도록 설계되었습니다. 학습된 그래프 표현은 차트 질문 응답 과제에서 대조 학습 기법을 통해 개선되며, 최종적으로는 MLLMs에 대한 체인-오브-생각 (Chain-of-Thought, CoT) 프롬프트 디자인을 포함하여 비약적 성과를 얻습니다. 이러한 구조적 및 관계적 정보를 효과적으로 융합하기 위해 그래프 대조 학습 방법을 도입하여 다중 모달 환경에서의 새로운 기법을 구현했습니다.

- **Performance Highlights**: 공공 벤치마크인 ChartQA, OpenCQA, ChartX에서 두 가지 방법을 시험한 결과, 이전 방법보다 향상된 성능을 달성했습니다. 특히, CoT 프롬프트는 모델의 설명 가능성과 정확성을 동시에 향상시킴을 보여주었습니다. 전반적인 성능 개선이 미약할 수 있지만, 세부 사례 연구를 통해 보다 훌륭한 성과를 확인할 수 있었습니다.



### IOLBENCH: Benchmarking LLMs on Linguistic Reasoning (https://arxiv.org/abs/2501.04249)
- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 언어적 추론 능력을 평가하기 위해 새로운 벤치마크인 IOLBENCH를 소개합니다. 이 데이터셋은 국제 언어 올림피아드(IOL)의 문제에서 파생되었으며, 구문(syntax), 형태학(morphology), 음운론(phonology) 및 의미론(semantics)을 포함한 다양한 문제를 포함하고 있습니다. 기존의 모델들이 갖고 있는 제한적 능력을 극복하고, 인간과 유사한 변별력을 지닌 모델 개발을 위한 기반을 제공합니다.

- **Technical Details**: IOLBENCH는 2003년부터 매년 개최된 국제 언어 올림피아드의 문제를 기반으로 한 고품질 데이터셋입니다. 이 데이터셋은 언어에 대한 사전 지식 없이 최소한의 예시를 통해 언어적 원리를 추론하기 요구하며, 특히 자원 부족 언어와 같은 도전적인 문제를 포함하는 것이 특징입니다. 데이터베이스는 약 1,500개의 문제 인스턴스로 구성되어 있으며, 각 문제는 공식 해답과 연결되어 전문가들이 제공한 정답을 포함합니다.

- **Performance Highlights**: 실험을 통해 OpenAI의 GPT-4, Anthropic의 Claude 모델, Google의 Gemini 모델 등 여러 최신 LLM을 평가한 결과, 언어의 복잡성을 다루는 데 있어 모델들이 지속적으로 어려움을 겪고 있음을 발견했습니다. 특히, 순차적 추론 및 구조적 데이터처리 요구되는 구문과 형태학 분야에서의 성능 격차가 두드러졌습니다. IOLBENCH의 사용은 AI와 계산 언어학의 융합을 촉진하고, 인간의 언어적 문제 해결 과정을 모방하는 모델 개발의 새로운 방향성을 제시합니다.



### Multimodal Multihop Source Retrieval for Web Question Answering (https://arxiv.org/abs/2501.04173)
Comments:
          arXiv admin note: text overlap with arXiv:2010.03604 by other authors

- **What's New**: 이번 연구에서는 멀티모달 멀티홉 질문 응답(Multi-modal Multi-hop Question Answering) 문제를 해결하기 위해 그래프 사고 네트워크(graph reasoning network)를 제안하고 있습니다. 이 네트워크는 이미지와 텍스트 기반의 다양한 출처에서 지원 사실(supporting facts)을 찾기 위해 구문(syntactic) 및 의미 구조(semantic structure)를 활용합니다. 본 연구의 핵심은 그래프 구조의 중요성을 강조하며, 적절한 feature representations를 사용해 수행 성능을 높일 수 있음을 입증합니다.

- **Technical Details**: 이 연구에서는 멀티모달 멀티홉 질문 응답을 위한 그래프 합성곱 네트워크(Graph Convolutional Networks, GCN)를 사용하여 다양한 소스에서 정보 추출 및 관련 출처 선택을 해결합니다. GCN은 그래프의 다양한 노드 간 정보 공유를 통해 의사결정을 하는 데 최적화되어 있으며, 이를 통해 정보 집합(aggregation)과 멀티모달 검색(multi-modal retrieval) 작업에 적합한 것으로 나타났습니다. 세 가지 독립적인 접근 방식을 통해 GCN의 차별점과 성능을 보여줍니다.

- **Performance Highlights**: 본 연구에서는 GCN를 활용하여, 기존 transformer 기반의 baseline 모델을 4.6% Retrieval F1 score 향상시켜 성능 개선을 입증하였습니다. 실험 결과, 그래프 네트워크에서 메시지 전파(message propagation)를 통해 멀티모달 transformer를 대체할 수 있음 또한 밝혀졌습니다. 저자는 이 모델이 대규모 검색 환경에서도 적용 가능함을 보여주었습니다.



### Reasoning-Enhanced Self-Training for Long-Form Personalized Text Generation (https://arxiv.org/abs/2501.04167)
- **What's New**: 본 논문은 개인화된 텍스트 생성을 위해 Reasoning-Enhanced Self-Training for Personalized Text Generation (REST-PG) 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLM)이 사용자에 대한 과거의 선호도, 배경 지식 및 작성 스타일을 기반으로 reasoning(추론)을 수행하도록 훈련합니다. 이를 통해 REST-PG는 모델이 개인화된 맥락을 보다 효과적으로 활용하게 함으로써, 사용자 기대에 부합하는 응답을 생성할 수 있도록 합니다.

- **Technical Details**: REST-PG는 LLM이 개인화된 데이터를 기반으로 응답을 생성하는 동안 reasoning을 진행할 수 있도록 하는 다단계 프레임워크입니다. 이 시스템은 Expectation-Maximization Reinforced Self-Training을 활용하여, 높은 보상을 받는 응답을 생성하는 reasoning 경로를 반복적으로 학습시킵니다. 초기 단계에서는 LLM이 스스로 reasoning 경로를 생성하고, 이후에는 이러한 경로를 토대로 어떻게 보상을 극대화할 수 있는지를 학습합니다.

- **Performance Highlights**: 실험 결과, REST-PG는 LongLaMP 벤치마크에서 기존 최첨단 모델들보다 평균 14.5% 향상된 성능을 보였습니다. 특히, supervised fine-tuning(SFT)과 비교해도 14.5%의 성능 향상이 있었으며, reasoning enhancement 없이 self-training을 진행한 경우보다도 6.5% 개선된 결과를 나타냈습니다. 이러한 성과는 REST-PG가 개인화된 텍스트 생성 분야에서 효과적인 접근법임을 입증합니다.



### Multilingual Open QA on the MIA Shared Task (https://arxiv.org/abs/2501.04153)
- **What's New**: 이번 논문은 Cross-lingual Information Retrieval (CLIR) 모델을 개발하여 저자원 언어에서도 효과적으로 검색할 수 있도록 하는 새로운 접근 방식을 제안합니다. 특히 이 연구는 추가적인 감독이나 레이블된 데이터를 필요로 하지 않으며, 다양한 언어에서 검색한 결과를 재조정하는 간단하고 효과적인 방법을 제공합니다.

- **Technical Details**: 제안된 재점수 기법은 사전 학습된 다국어 질문 생성 모델을 사용하여 검색된 패세지를 재조정합니다. 이 모델은 입력 질문의 확률을 계산하기 위해 retrieval된 패세지에 조건부로 입력 질문의 확률을 평가합니다. 이 과정에서 BM-25와 같은 희소 검색 방법으로 얻은 결과를 재조정할 수 있으며, 따라서 고가의 필수 레이블된 코퍼스를 얻지 않고도 저자원 언어에 활용할 수 있습니다.

- **Performance Highlights**: 저자들은 모델의 성능을 완전한 제로샷 환경에서 평가하였으며, 이 방법은 별도의 훈련 없이도 신뢰할 수 있는 결과를 보여주었습니다. 특히, 한국어와 벵골어를 포함한 저자원 언어의 데이터 증강이 이루어졌으며, 이를 통해 저자원 언어에 대한 성능이 개선될 것으로 기대하고 있습니다.



### "Yeah Right!" -- Do LLMs Exhibit Multimodal Feature Transfer? (https://arxiv.org/abs/2501.04138)
- **What's New**: 이 논문은 인간 간의 의사소통에서에서 말의 감춰진 의미를 인식하는 능력을 테스트한 결과 멀티모달(Multimodal) 모델이 일모달(Unimodal) 모델에 비해 우위를 보인다는 것을 발견했습니다. 특히, 대화에 대한 훈련을 받은 모델이 이 과정에서 더 유리하게 작용합니다. 연구는 또한 인간이 처음으로 음성을 통해 소통의 기술을 배운 후, 이를 글쓰기에 전이하는 과정에도 주목합니다.

- **Technical Details**: 직접적인 언어 인식 대신, 이 논문은 음성으로 훈련된 모델이 의사소통에서의 미묘한 컨셉(Cognition)과 기법(Technique)을 더 잘 이해할 수 있음을 제시합니다. 연구에서는 Big-Bench 데이터셋을 이용하여 비꼬는(Sarcasm), 아이러니(Irony), 그리고 경멸(Condescension)과 같은 기법들을 탐지하는 능력을 평가했습니다. 다양한 프롬프트를 통해 모델의 분류 성능을 시뮬레이션하며, 음성이 결합된 훈련이 언어적 텍스트의 접근 방식을 개선할 수 있음을 확인합니다.

- **Performance Highlights**: 논문은 GPT-4o와 같은 멀티모달 모델이 단순 텍스트 모델인 GPT-4-Turbo보다 평균적으로 모든 데이터셋에서 향상된 성능을 보였음을 보여줍니다. 이러한 성능 차이는 특히 감춰진 비유의식을 인식하는 데 있어서 두드러지며, 대화에 특화된 훈련이 해당 모델의 성능을 더욱 강화할 수 있음을 제안합니다. 결과적으로, 다중모달 특징이 텍스트 인식 성능에 유익하다는 중요한 시사점을 제공합니다.



### A Survey on Large Language Models with some Insights on their Capabilities and Limitations (https://arxiv.org/abs/2501.04040)
Comments:
          174 pages, to be submitted to a journal in a shorter version. arXiv admin note: text overlap with arXiv:2303.18223, arXiv:2303.17564, arXiv:2301.00234, arXiv:2303.08774, arXiv:2402.02315, arXiv:2210.03493, arXiv:2402.01817, arXiv:2407.21783, arXiv:2208.05051 by other authors

- **What's New**: 최근 인공지능 분야는 Transformers 아키텍처에 기반한 대형 언어 모델(LLMs)의 발전에 의해 크게 변화하였습니다. LLMs는 텍스트 생성, 질문 답변, 번역 및 요약과 같은 다양한 언어 관련 작업에서 사람처럼 이해하는 능력을 선보이며, 이는 자연어 처리 방식에 혁신을 가져왔습니다. 특히, 이 모델들은 코딩 생성, 기본 상식 추론 및 산술 계산과 같은 본연의 기능을 넘어서는 능력을 보여주기로 주목받고 있습니다.

- **Technical Details**: 이 논문에서는 LLM의 기본 구성 요소와 스케일링 메커니즘, 건축 전략을 탐구하며, 특히 GPT와 LLaMA와 같은 모델들에 중점을 둡니다. 급증하는 데이터와 계산 능력이 LLM 성능에 미치는 영향을 분석하며, 스케일링과 관련된 트레이드오프에 대해서도 논의합니다. 또한 LLM의 다양한 적용 사례에 대해 살펴보고, 이것이 의료, 금융, 교육, 법률 등 각 분야에서의 문제 해결 능력을 어떻게 나타내는지 설명합니다.

- **Performance Highlights**: LLM은 향상된 언어 이해 덕분에 복잡한 언어적 도전 과제를 해결할 수 있는 잠재력을 지니고 있습니다. LLM들은 코그니티브한 작업을 수행하는 데 필요한 계획 및 추론 능력을 갖추고 있으며, Chain of Thought(CoT) 및 Plan of Thought(PoT)와 같은 새로운 접근 방식을 통해 그 성능을 더욱 향상시킬 수 있습니다. 본 논문은 LLM의 능력과 한계를 지속적으로 탐구하여, 이 분야에서의 책임 있는 개발 및 활용 방안을 모색할 것입니다.



### Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Though (https://arxiv.org/abs/2501.04682)
- **What's New**: 새로운 프레임워크인 Meta Chain-of-Thought (Meta-CoT)를 제안합니다. 이 프레임워크는 기존의 Chain-of-Thought (CoT)를 확장하여 특정 CoT에 도달하기 위한 기본적인 추론을 명시적으로 모델링합니다. 이 과정에서 최신 모델들이 맥락 내 검색(in-context search)과 일치하는 행동을 보이는 실증적 증거를 제시합니다.

- **Technical Details**: Meta-CoT의 생성 방법으로는 과정 감독(process supervision), 합성 데이터 생성(synthetic data generation), 검색 알고리즘(search algorithms) 등을 탐구하였습니다. 또한, 모델 훈련을 위한 구체적인 파이프라인(pipeline)을 제시하며, 이 파이프라인은 선형화된 검색 흔적(linearized search traces)과 강화 학습(reinforcement learning) 후 훈련을 포함합니다.

- **Performance Highlights**: 이 연구는 LLM(대형 언어 모델)에서 Meta-CoT를 가능하게 하는 이론적 및 실용적 로드맵을 제공합니다. 더 나아가, 스케일링 법칙(scaling laws), 검증자 역할(verifier roles), 새로운 추론 알고리즘 발견 가능성에 대한 여러 가지 열린 연구 질문을 논의합니다.



### FlairGPT: Repurposing LLMs for Interior Designs (https://arxiv.org/abs/2501.04648)
Comments:
          Accepted at EUROGRAPHICS 2025

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 내부 디자인에 어떻게 활용할 수 있는지를 탐구합니다. 기존의 데이터 기반 접근 방식과 달리, 이 연구는 LLMs가 디자인의 다양한 제약 조건을 효과적으로 생성할 수 있음을 보여주며, 이를 통해 내부 공간의 디자인 과정을 보다 구조화할 수 있는 방법을 제시합니다. 최종적으로, LLMs의 출력을 기반으로 한 레이아웃 제약 그래프를 통한 최적화를 통해 높은 품질의 디자인 레이아웃을 생성할 수 있습니다.

- **Technical Details**: 이 연구는 LLMs의 구조적 활용을 통해 사용자가 제시한 공간을 여러 영역으로 나누고, 각 영역에 배치할 개체 목록 및 제약 조건을 추출합니다. 이러한 정보를 바탕으로 레이아웃 제약 그래프를 구성한 후, 이를 해결하기 위해 외부 도구인 제약 최적화 설정을 사용하여 최종 디자인을 생성합니다. 권장된 방법에서는 LLM의 출력을 대수적 제약 조건으로 변환하여 개체 변수의 크기 및 배치를 기반으로 한 레이아웃을 도출합니다.

- **Performance Highlights**: 본 연구에서 제안한 방법은 기존의 LLM 기반 메서드 및 인간 디자인과 비교하여 다양한 디자인 설정에서 평가되었습니다. 사용자 연구를 통해, 사용자들은 생성된 레이아웃이 디자인 사양에 잘 맞춰져 있으며 기능적으로 유용하다고 평가하여 우리의 방법을 선호했습니다. 이러한 경험적 평가로, LLMs를 구조적으로 활용할 수 있을 때 고품질의 다양한 레이아웃 생성이 가능함을 입증했습니다.



### InfiGUIAgent: A Multimodal Generalist GUI Agent with Native Reasoning and Reflection (https://arxiv.org/abs/2501.04575)
Comments:
          14 pages, 7 figures, work in progress

- **What's New**: InfiGUIAgent는 다중 모드 대형 언어 모델(MLLM)을 활용한 새로운 GUI 에이전트이다. 이 에이전트는 2단계 감독 세밀 조정(supervised fine-tuning) 파이프라인을 통해 훈련되어, GUI 이해 및 기초 능력뿐만 아니라 고급 추론 능력을 개선한다. 이 연구는 복잡한 GUI 작업 수행을 위한 본격적인 추론 능력을 추가하여 에이전트의 성능을 향상시키는 데 중점을 두고 있다.

- **Technical Details**: InfiGUIAgent의 개발은 두 가지 주요 단계로 나뉘어 있다. 첫 번째 단계에서는 GUI 이해와 관련된 기초 능력을 강화하기 위해 시각-언어 데이터셋을 수집하고, 두 번째 단계에서는 계층적 추론(hierarchical reasoning)과 기대-반사 추론(expectation-reflection reasoning)과 같은 고급 추론 능력을 통합하는 데이터셋에서 훈련된다. 이를 통해 에이전트는 직관적인 사고를 바탕으로 복잡한 작업을 수행할 수 있다.

- **Performance Highlights**: InfiGUIAgent는 여러 GUI 벤치마크에서 경쟁력 있는 성능을 달성하였다. 기본 능력과 고급 추론 능력을 동시에 향상시키는 두 단계의 감독 세부 조정이 이 모델의 강점을 보여준다. 연구 결과는 GUI 상호작용을 효과적으로 향상시키고 자동화 작업의 범위를 확대할 수 있는 가능성을 시사한다.



### Supervision-free Vision-Language Alignmen (https://arxiv.org/abs/2501.04568)
Comments:
          Preprint

- **What's New**: 이 논문에서는 SVP(Supervision-free Visual Projection)라는 새로운 프레임워크를 소개합니다. VLMs(비전-언어 모델)의 성능 향상에 초점을 맞추며, 이는 비급식 데이터나 선호 주석 없이도 가능하다는 점에서 이전 연구들과 차별화됩니다. SVP는 자기 캡셔닝(self-captioning)과 사전 훈련된 그라운딩 모델(pre-trained grounding model)을 활용하여 VLM의 잠재 정보를 이끌어내는 피드백 메커니즘을 이용합니다.

- **Technical Details**: SVP는 크게 이미지-텍스트 쌍의 수집이 필요하지 않은 점이 특징이며, 이를 통해 비전-언어 정합성(vision-language alignment)을 개선합니다. 연구에서는 캡셔닝(captioning), 참조(referring), 비주얼 질문 응답(visual question answering), 멀티태스킹(multitasking), 할루시네이션 제어(hallucination control), 객체 회상(object recall) 등 여섯 가지 주요 영역에서 평가가 이루어졌습니다.

- **Performance Highlights**: SVP를 적용한 결과, 캡셔닝 작업에서 평균 14%의 성능 향상, 객체 회상에서 최대 12% 증가, 할루시네이션 비율 대폭 감소 등 주요 성과가 보고되었습니다. 특히, SVP를 활용한 작은 VLM이 원래 크기가 다섯 배 큰 모델과 비교할 만한 수준으로 할루시네이션을 줄인 점이 주목할 만합니다.



### Improving Image Captioning by Mimicking Human Reformulation Feedback at Inference-tim (https://arxiv.org/abs/2501.04513)
- **What's New**: 본 논문에서는 생성 모델 훈련에 인간 피드백을 자동으로 예측하여 반영하려는 최근의 관심을 바탕으로, 새로운 피드백 유형인 caption reformulations (캡션 개편)을 도입하고 이를 바탕으로 모델을 훈련합니다. 이 방법은 기존의 이미지 캡셔닝 모델에 추가적인 훈련 없이도 쉽게 적용할 수 있어 컴퓨팅 자원을 크게 절약합니다. 특히, 비영어 이미지 캡셔닝 영역에서도 개선된 성능을 보여 주목할 만합니다.

- **Technical Details**: 연구에서 도입한 reformulation 피드백은 기존의 비교 피드백과 달리 이미지 캡션의 오류를 교정하는 방식으로 구성됩니다. 연구팀은 몇 천 개의 샘플을 수집하여 이를 기반으로 모델을 훈련하고, 훈련된 모델을 기존 캡셔닝 모델의 추론 단계에 통합하여 활용합니다. 이 과정에서 'challenge domains'에서의 특별한 유용성을 강조하며, 독일어 이미지 캡셔닝에서도 뛰어난 성능을 발휘하는 것을 확인했습니다.

- **Performance Highlights**: 자동화된 캡션 개편 과정은 질이 낮은 모델에서 생성된 캡션의 품질을 크게 향상시키며, 특히 없는 정보를 추가함으로써 개선된 결과를 도출합니다. 스타일 전이 작업에서도 기존의 캡셔닝 모델과 비교하여 더 뛰어난 성능을 발휘하며, 메시지의 구조를 유지하면서 타겟 스타일로 수정할 수 있는 가능성을 보여줍니다. 주어진 데이터셋에서 이 방법은 최신 기술 수준을 달성하며, 인간 평가에서도 향상된 캡션 스타일을 증명했습니다.



### Developing a Modular Compiler for a Subset of a C-like Languag (https://arxiv.org/abs/2501.04503)
- **What's New**: 이 논문은 C유사 언어의 일부에 대한 모듈형 컴파일러(modular compiler) 개발을 소개합니다. 이 접근법은 고급 언어의 컴파일러를 구축하는 데 있어 발생하는 다양한 어려움을 해결합니다. 개발자는 필요에 따라 하위 집합(subset)을 추가하거나 제거하여 언어를 수정할 수 있어, 최소한의 메모리와 효율적인 컴파일러를 만들 수 있습니다.

- **Technical Details**: 개발 과정은 작고 점진적인 단계로 나누어져 있으며, 각 단계는 언어의 확장된 하위 집합을 위한 완벽하게 작동하는 컴파일러를 생성합니다. 논문은 이러한 반복적인 개발 단계(iterative developmental phase)를 설명하며, 능력과 기능의 점진적인 향상을 강조합니다. 모듈화 설계(modular design), 코드 재사용성(code reusability), 문서화(documentation)와 같은 산업 모범 사례(best practices)를 준수함으로써, 최종 컴파일러의 기능적 효율성(functional efficiency), 유지 관리성(maintainability), 확장성(extensibility)이 가능해졌습니다.

- **Performance Highlights**: 이 컴파일러는 언어 구조 관리뿐만 아니라 최적화된 코드 개발에서도 유용함을 입증하였습니다. 이의 실용성을 확인하기 위해 작은 메모리 부족(single-board computer) 시스템에서 컴파일러를 사용한 결과, 자원이 제한된 장치에서도 높은 효율성과 적합성을 보여주었습니다.



### NSA: Neuro-symbolic ARC Challeng (https://arxiv.org/abs/2501.04424)
- **What's New**: 이 논문은 아브스트랙션 및 추론 데이터셋(ARC)을 해결하기 위한 신경-상징적 접근 방식을 제안합니다. 제안된 방법은 transformer를 사용하여 유망한 검색 방향을 제안하고, 이를 도메인 특화 언어(DSL)와 결합하여 조합적 검색을 수행합니다. 연구 결과, 이 접근 방법은 ARC 평가 세트에서 기존 방법 대비 27% 높은 성능을 기록했습니다.

- **Technical Details**: 제안된 방식은 transformer 모델을 사전 훈련하기 위해 합성적으로 생성된 데이터를 사용하며, 테스트 시간 동안 특정 작업 데이터셋의 세부 과제를 생성하고 모델을 미세 조정합니다. DSL은 올바른 추상화 수준에서 추론하는 데 유용한 인덕티브 바이어스를 제공합니다. 이를 통해 조합적 검색이 제한된 시간 내에 실제 솔루션을 찾을 수 있도록 유도합니다.

- **Performance Highlights**: 기존 대안들과 비교했을 때, 제안된 방식은 ARC 훈련 및 평가 세트에서 효과성을 입증하였습니다. 특히, FETCH의 조합적 접근법은 최신 ML 접근 방식을 초월하는 성과를 보여주었으며, 기존의 DSL 기반 방법들과도 경쟁력을 가지고 있습니다. 이 연구 결과는 오프라인과 온라인 조건 모두에서 이 방법이 얼마나 효과적인지를 잘 나타냅니다.



### Decoding EEG Speech Perception with Transformers and VAE-based Data Augmentation (https://arxiv.org/abs/2501.04359)
Comments:
          19 pages, 15 figures, 2 tables

- **What's New**: 이번 연구는 비침습적인 뇌 신호인 EEG를 기반으로 한 음성 디코딩의 발전 가능성을 탐구합니다. 연구팀은 variational autoencoders (VAEs)와 최신의 시퀀스-투-시퀀스 (sequence-to-sequence) 딥 러닝 아키텍처를 활용하여 EEG 데이터의 품질을 향상시키고, 복잡한 음성 인식 과제에서의 성능 향상을 목표로 하고 있습니다. 이것은 조용한 커뮤니케이션이나 소음이 있는 환경에서의 보조 기술로서 큰 가능성을 지닙니다.

- **Technical Details**: EEG 신호에서의 음성을 디코딩하기 위해, 연구팀은 심층 학습 기법과 데이터 전처리에 중점을 두고 대규모 데이터셋을 활용합니다. 변형 오토인코더(VAE)를 사용하여 노이즈 저항성을 강화하고, EMG 기반의 SOTA transformer 모델을 EEG 신호에 적응하는 접근 방식도 연구하였습니다. 이러한 방법의 조합을 통해 EEG 기반 음성 인식의 최신 기술을 향상시킬 수 있는 잠재력을 갖추었습니다.

- **Performance Highlights**: 실험 결과는 VAE가 인공 EEG 데이터를 재구성하여 데이터 증강에 유용할 수 있음을 보여주었습니다. 또한, 시퀀스-투-시퀀스 모델은 분류 모델에 비해 문장 생성 성능이 더 유망한 것으로 나타났습니다. 이러한 결과는 EEG를 통한 음성 인식 디코딩의 미래 연구를 위한 기초를 마련하며, 조용한 음성이나 상상된 음성과 같은 음성 생산 과제로의 확장 가능성을 제시합니다.



### TimelineKGQA: A Comprehensive Question-Answer Pair Generator for Temporal Knowledge Graphs (https://arxiv.org/abs/2501.04343)
- **What's New**: 본 연구는 Temporal Knowledge Graphs(TKGs)를 기반으로 한 새로운 질문 응답 프레임워크인 TimelineKGQA를 제안하여, 시계열 데이터에 대한 정보 검색과 추론 간의 전환을 가능하게 합니다. 연구진은 또한 다양한 질문 유형을 포용하는 포괄적인 데이터 집합 생성을 위한 Python 패키지를 오픈 소스로 제공하여, 연구자들이 TKGQA 연구에서 활용할 수 있도록 지원합니다.

- **Technical Details**: Temporal Knowledge Graphs는 (e1,r,e2,tstart,tend) 형태의 데이터 쌍을 사용하여 시간적 관계를 통합합니다. TKGQA는 정보 검색(IR)과 시간적 추론 간의 연결 고리를 제공하며, 질문 복잡성에 대한 기준을 네 가지 차원(문맥 복잡성, 답변 초점, 시간적 관계, 시간적 능력)으로 구분합니다. 이 프레임워크는 사용자가 다양한 질문을 생성하고 분석할 수 있게 지원합니다.

- **Performance Highlights**: 현재 TKGQA 연구는 데이터셋의 한계로 인해 발전이 저해되고 있으며, 기존 데이터셋(CronQuestion 등)이 시간적 복잡성을 포괄적으로 다루지 못하고 있습니다. 연구진의 제안된 프레임워크를 통해 사용자 정의 QA 쌍 생성을 가능하게 하여, 시계열 데이터 연구의 진전을 도모할 수 있게 됩니다. 많은 경우, 모델은 0.9 이상의 성능을 달성했지만, 데이터셋의 다양성 부족으로 정보를 찾는 데 한계가 있습니다.



### Circuit Complexity Bounds for Visual Autoregressive Mod (https://arxiv.org/abs/2501.04299)
- **What's New**: 이 연구는 Visual AutoRegressive (VAR) 모델의 회로 복잡성(circuit complexity)에 대한 경계를 설정하고, VAR 모델이 $	ext{TC}^0$ 임계 회로(threshold circuit)의 시뮬레이션으로 동등함을 증명합니다. 이 회로는 은닉 차원(hidden dimension) $d 	ext{(d)}$가 $O(n)$ 이하이고, $	ext{poly}(n)$ 정밀도(precision)를 가지고 있습니다. VAR 모델은 이전 기술인 Diffusion Transformers를 초월하는 이미지 생성 능력을 보여주며, 본 연구는 이러한 모델의 표현력 표현의 한계를 철저히 분석한 첫 번째 연구입니다.

- **Technical Details**: 본 연구에서는 VAR 모델의 구조와 구성 요소(예: 업샘플링(interpolation) 레이어, 컨볼루션(convolution) 레이어, Transformer 블록 등)의 계산 복잡성(computational complexity)을 분석합니다. 회로 복잡성 이론을 통해 VAR 모델을 복잡성 회로(complexity circuits)로 표현함으로써, 이 모델이 수행할 수 있는 문제의 하한(bounds)을 정량적으로 평가할 수 있는 방법론을 제시합니다. 특히, $	ext{DLOGTIME}$-균일 $	ext{TC}^0$ 회로 패밀리가 VAR 모델을 O(1) 깊이(depth), $	ext{poly}(n)$ 크기(size), 및 $	ext{poly}(n)$ 정밀도로 시뮬레이션 가능함을 보여줍니다.

- **Performance Highlights**: VAR 모델은 기존의 이미지 생성 방법들과 비교했을 때, 더욱 사실적이고 다양한 이미지를 생성하는 능력을 보여줍니다. 특히, VAR 모델은 제로 샷 제너럴라이제이션(zero-shot generalization) 능력을 갖춰 이미지 인페인팅(image inpainting) 및 조작(manipulation) 작업 등 다양한 분야에서 뛰어난 성능을 발휘합니다. 본 연구는 VAR 모델의 표현력의 한계를 밝혀내며, 이로 인해 더 효율적이고 표현력 있는 아키텍처 개발에 기여할 가능성을 지니고 있습니다.



### Agent Laboratory: Using LLM Agents as Research Assistants (https://arxiv.org/abs/2501.04227)
- **What's New**: 본 논문에서는 Agent Laboratory라는 자율적인 LLM(언어 모델 기반) 프레임워크를 소개합니다. 이 프레임워크는 인간이 제공하는 연구 아이디어를 바탕으로 문헌 검토, 실험, 보고서 작성의 세 단계를 거쳐 종합적인 연구 결과물(코드 저장소 및 연구 보고서 포함)을 생성합니다. 사용자는 각 단계에서 피드백과 지침을 제공할 수 있어 연구 과정이 더욱 원활하게 진행됩니다.

- **Technical Details**: Agent Laboratory는 다양한 최첨단 LLM을 활용하여 연구 과정을 자동으로 수행합니다. 이 시스템은 연구의 모든 단계를 포괄하며, 연구자들에게 과정 전반에서 인간의 피드백을 받을 수 있는 경로를 제공합니다. 연구 결과는 자율 연구 방식의 전통적인 비용과 시간을 크게 절감합니다.

- **Performance Highlights**: 연구 결과에 따르면, o1-preview 기반의 Agent Laboratory가 가장 우수한 연구 결과물을 생성했습니다. 생성된 머신러닝 코드는 기존 방법들과 비교하여 최첨단 성능을 달성하였으며, 각 단계에서의 인간의 참여가 연구의 전반적인 품질을 크게 개선하는 것으로 나타났습니다. 최종적으로 Agent Laboratory는 연구 비용을 84%나 절감하는 효과를 보였습니다.



### MM-GEN: Enhancing Task Performance Through Targeted Multimodal Data Curation (https://arxiv.org/abs/2501.04155)
- **What's New**: 이번 논문에서는 MM-Gen이라는 새로운 방법론을 소개합니다. 이 방법은 고품질의 합성 텍스트(annotation)를 생성하여 특정 작업(task)에 맞는 시각-언어 모델(VLM)의 성능을 높입니다. MM-Gen은 세 가지 단계의 프로세스를 통해 필요한 데이터를 효과적으로 생성하며, 기존의 양질의 데이터가 부족한 문제를 해결합니다.

- **Technical Details**: MM-Gen은 강력한 VLM을 활용하여 이미지와 연관된 텍스트 주석을 생성합니다. 이 과정은 크게 세 가지로 나뉘며, (1) 데이터를 소그룹으로 분리하고, (2) 작업 설명에 기초하여 목적에 맞는 텍스트 생성, (3) 불필요한 데이터 필터링을 통해 이루어집니다. 이를 통해 VLM을 fine-tuning하면 다수의 작업에서 성능을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, MM-Gen을 통해 Llava-1.5(7B) 모델은 차트 이해, 공간 추론 및 도표 이해에서 각각 15%, 14%, 29%의 성능 향상을 기록했습니다. 원본 모델에 비해 MM-Gen 데이터는 최대 1.6배 높은 개선 효과를 보이며, 이를 통해 특정 작업에서 VLM 성능을 증가시킬 수 있음을 입증합니다.



### More is not always better? Enhancing Many-Shot In-Context Learning with Differentiated and Reweighting Objectives (https://arxiv.org/abs/2501.04070)
Comments:
          13 pages, 8 figures, 11 tables

- **What's New**: 이 논문은 DR-ICL이라는 새로운 최적화 방법을 도입하여 대형 언어 모델(LLMs)의 Many-shot In-context Learning(아래 ICL)의 성능을 향상시키는 접근 방식을 제안합니다. DR-ICL은 Differentiated Learning과 advantage-based Reweighting을 활용하여 모델이 발생하는 데이터 노이즈의 영향을 줄이면서 더 나은 성능을 발휘하도록 합니다. 또한, MICLB라는 대규모 벤치마크 데이터셋을 소개하여 다양한 NLP 작업에서 Many-shot ICL 전략을 평가할 수 있도록 합니다.

- **Technical Details**: DR-ICL은 NLL(negative log-likelihood) 최적화 목표를 개선하여 모델이 많은 시연을 가진 경우에도 제로 샷 성능을 초과하도록 합니다. 이 방법은 강화 학습에서 영감을 받은 advantage function을 활용하여 샘플들의 가중치를 동적으로 조정함으로써, 훈련 과정에서의 노이즈 데이터를 필터링합니다. 이러한 기법은 ICL에서 여러 샷 수를 효과적으로 처리할 수 있게 하며, 다양한 작업에서의 일반화를 개선합니다.

- **Performance Highlights**: 실험 결과, DR-ICL로 개선된 LLM은 다양한 작업에서의 Many-shot ICL 설정에서 유의미한 성능 향상을 달성했습니다. 특히, 문맥적 단서에 대한 더 깊은 이해를 통해 모델이 맥락 정보를 효과적으로 활용하도록 유도합니다. 이 연구의 결과는 LLM의 다중 작업 학습 및 평가 연구에 있어 중요한 기여를 하며, 오픈 소스 LLM에 대한 연구의 발전을 촉진할 것입니다.



### The Power of Negative Zero: Datatype Customization for Quantized Large Language Models (https://arxiv.org/abs/2501.04052)
Comments:
          under submission

- **What's New**: 이번 논문에서는 LLMs(대형 언어 모델)의 메모리 요구 사항을 줄이기 위해 FP(부동 소수점) 양자화를 활용한 RaZeR(중복 제로 재매핑) 기술을 제안합니다. RaZeR는 부동 소수점의 음의 영 표현을 미리 정의된 특별 값으로 재매핑하여 양자화 성능을 극대화합니다. 이를 통해 전통적인 INT 양자화 방법보다 더 나은 성능을 보여줍니다.

- **Technical Details**: RaZeR는 특수 값을 신중하게 선택하여 양자화에서 더 많은 숫자 분포를 잘 맞출 수 있도록 설계되었습니다. 이 기술은 가중치 및 KV-cache 양자화 알고리즘과 통합될 수 있으며, 클리핑 및 변환 등의 고급 방법과도 호환됩니다. 또한, 4비트 RaZeR 값을 FP16으로 변환하는 빠른 GEMV(행렬-벡터 곱) 커널을 구현하여 개선된 연산 효율을 제공합니다.

- **Performance Highlights**: 현대 GPU에서 RaZeR는 FP16 구현에 비해 GEMV 속도를 최대 7.56배 향상시키고 LLM 디코딩 처리량에서는 최대 2.72배 속도를 증가시킵니다. 이러한 성과는 RaZeR가 LLM의 성능과 처리 효율성을 크게 개선할 수 있음을 보여줍니다. 결과적으로, RaZeR는 최신 AI 모델의 상용화에 중요한 역할을 할 수 있을 것입니다.



### DPO Kernels: A Semantically-Aware, Kernel-Enhanced, and Divergence-Rich Paradigm for Direct Preference Optimization (https://arxiv.org/abs/2501.03271)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)과의 정렬(alignment) 문제를 해결하기 위한 새로운 접근법인 DPO-Kernels를 제안합니다. DPO-Kernels는 커널 방법(kernel methods)을 통합하여 고정된 편차(fixed divergences) 및 제한된 특성 변환(feature transformations)의 문제를 극복하고자 합니다. 이를 통해 다각적인 전이(transformations)와 손실(loss) 함수를 개선하였습니다.

- **Technical Details**: DPO-Kernels의 주요 기여는 네 가지입니다: (i) 다항식(polynomial), RBF, 마할라노비스(Mahalanobis), 스펙트럼(spectral) 커널을 활용한 커널화된 표현(kernelized representations)과 임베딩 기반(embedding-based) 및 확률 기반(probability-based) 목표를 결합한 하이브리드 손실(hybrid loss); (ii) 다양성 대안(divergence alternatives)으로 제센-샤논(Jensen-Shannon), 헬링거(Hellinger), 레니(Renyi), 바타차리야(Bhattacharyya), 워서스타인(Wasserstein), f-다양성(f-divergences)을 포함하여 더 큰 안정성(stability) 제공; (iii) 데이터 기반 선택(metrics)으로 최적의 커널-다양성 쌍을 자동으로 선택; (iv) 지역 정확성과(global modeling) 전역 모델링을 위한 계층적 혼합(hierarchical mixture) 커널입니다.

- **Performance Highlights**: 12개의 데이터셋에 대한 평가 결과, DPO-Kernels는 사실성(factuality), 안전성(safety), 추론(reasoning), 지침 따르기(instruction following)에서 최첨단 성능(state-of-the-art performance)을 보였습니다. 이 연구는 Heavy-Tailed Self-Regularization에 기초하여 LLM의 강건한 일반화(robust generalization)를 유지하며, 정렬 연구(alignment research)를 위한 포괄적인 자원(resources)을 제공합니다.



New uploads on arXiv(cs.IR)

### Knowledge Retrieval Based on Generative AI (https://arxiv.org/abs/2501.04635)
Comments:
          8 pages, 13 figures, 1 table

- **What's New**: 이번 연구에서는 중국어 위키백과와 Lawbank를 정보 검색(Source)으로 사용하여 Retrieval-Augmented Generation (RAG) 기반 질문-답변 시스템을 개발했습니다. BGE-M3를 활용한 밀집 벡터 검색을 통해 관련 검색 결과를 얻고, BGE-reranker로 이 결과를 쿼리 관련성에 따라 재정렬하여 성능을 향상시킵니다. 이를 통해 생성형 AI에 기반한 지식 검색 시스템을 구축하게 되었습니다.

- **Technical Details**: 연구에서는 양자된 쿼리를 입력받아 다국어, 다기능, 다중 세분화 지원이 가능한 BGE-M3 모델을 사용합니다. 이 모델은 100개 이상의 언어에 대한 검색 기능을 지원하며, 8,192개의 토큰 길이까지 처리할 수 있습니다. 또한 질문 답변에 필요한 정확한 결과를 위해 BGE-reranker를 도입하여 검색 결과의 관련성을 높였습니다.

- **Performance Highlights**: 자동 성능 평가에서는 모델의 자가 생성된 레이블과 실제 정답을 비교하여 정확도를 측정하고, 주관적 평가에서는 재정렬된 정보를 바탕으로 노지연 참여자들이 답변한 결과를 통해 시스템의 안정성을 검증했습니다. 연구 결과, 로컬 운영으로 데이터 프라이버시를 개선하고 상업적 서비스 의존도를 낮추면서 데이터 보안을 강화했습니다.



### Evaluating Interval-based Tokenization for Pitch Representation in Symbolic Music Analysis (https://arxiv.org/abs/2501.04630)
Comments:
          Accepted at Artificial Intelligence for Music Workshop at AAAI 2025 (this https URL)

- **What's New**: 본 연구에서는 음악 분석 작업을 위한 새로운 tokenization 전략인 interval-based tokenization을 소개합니다. 기존의 absolute pitch encoding 방식 대신, 피치 간의 상대적 거리인 피치 인터벌을 활용하여 더 직관적인 데이터 표현이 가능하다는 점을 강조합니다. 이를 통해 여러 음악 분석 과제를 통해 모델 성능 향상과 설명 가능성(ease of explainability)을 높일 수 있다는 것을 보여줍니다.

- **Technical Details**: 연구는 음악 정보 검색(Music Information Retrieval) 분야의 기존 tokenization 방법론에 대한 배경을 제시합니다. 대부분의 tokenization 전략은 MIDI 숫자인 absolute pitch encoding을 사용하지만, 이 연구에서는 melodic contour와 harmonic relations을 반영한 intervalization 프로세스를 정형화합니다. 이는 피치와 시간의 관계를 보다 시각적으로 명확히 하여, 각 음표의 상대적 간격을 표현합니다.

- **Performance Highlights**: 세 가지 음악 분석 작업을 통해 interval-based tokenization이 분석 모델의 성능을 향상시키는 것을 입증하였습니다. 각 작업에 적합한 최적의 tokenization 설정이 있으며, 이는 음악적으로 의미 있는 해석을 제공할 수 있습니다. 또한, intervalization을 통해 얻어진 결과는 음악적 스타일의 발견을 용이하게 합니다.



### A Closer Look on Gender Stereotypes in Movie Recommender Systems and Their Implications with Privacy (https://arxiv.org/abs/2501.04420)
Comments:
          19 pages, 2 figures

- **What's New**: 본 연구는 성별 고정관념(Gender Stereotypes)이 영화 추천 시스템에 미치는 영향을 분석하였습니다. 이 과정에서 성별이라는 개인 속성을 추출하기 위해 사용자의 피드백 데이터를 활용하는 특정 공격 시나리오를 설정하였습니다. 연구는 총 두 단계로 이루어져 있으며, 첫 번째 단계에서는 630명의 참가자를 대상으로 성별 고정관념이 영화 장르에 미치는 영향을 조사했습니다.

- **Technical Details**: 두 번째 단계에서는 첫 번째 단계에서 도출된 성별 고정관념을 사용자 피드백 데이터와 결합하여 성별을 추정하는 네 가지 추론 알고리즘을 적용했습니다. 이 알고리즘의 성능은 오직 피드백 데이터에만 의존했을 때보다 더 높았습니다. 주요 실험 데이터셋으로는 MovieLens 1M과 Yahoo!Movie가 사용되었으며, 관련된 자세한 실험 정보는 GitHub 리포지토리에 공개되어 있습니다.

- **Performance Highlights**: 이 연구는 영화 장르에 대한 성별의 선호가 비즈니스 관점에서 어떻게 이용될 수 있는지를 설명합니다. 또한, 성별 고정관념이 사용자 데이터의 사생활 침해와 편향 문제에 미치는 심각성을 강조하여, 추천 시스템에서의 성별 편향 문제를 풀어가는 데 기여할 수 있는 새로운 통찰력을 제공합니다. 연구 결과는 향후 추천 시스템의 투명성과 공정성을 위한 중요한 함의를 가집니다.



### An innovative data collection method to eliminate the preprocessing phase in web usage mining (https://arxiv.org/abs/2501.04364)
Comments:
          15 pages, 8 figures

- **What's New**: 이 연구에서는 웹 사용 데이터 수집 및 세션 관리를 위한 혁신적인 방법을 제안합니다. 기존의 서버 로그 대신, 애플리케이션 기반 API를 활용하여 로깅 데이터를 수집하고 처리하는 새로운 방식을 도입합니다. 이 방법은 웹 분석을 위한 데이터 소스로서의 활용성을 높이며, 더 신뢰할 수 있는 구조화된 데이터를 제공합니다.

- **Technical Details**: 제안된 방법은 기존 웹 서버 로그의 전처리 단계를 제거하여 데이터 수집 및 처리를 애플리케이션 수준에서 가능하게 합니다. 수집된 데이터는 관계형 데이터베이스에 구조화되어 저장되며, 이를 통해 높은 성능의 웹 사용 데이터 마이닝 활동과 실시간 웹 분석이 가능합니다. 이 과정에서 서버 부하를 줄이면서도 데이터의 정확성을 높이는 이점을 제공합니다.

- **Performance Highlights**: 제안된 방법은 사용자 및 세션 식별을 성공적으로 수행하고, 데이터 정확성을 증가시켜 결과적으로 웹 사용 데이터 마이닝(WUM) 프로세스를 가속화합니다. 이 데이터는 기계 학습 및 딥 러닝과 같은 인공지능 방법에 입력으로 사용될 수 있으며, 보다 유의미한 정보 제공을 가능하게 합니다. 또한, 이 과정에서 발생하는 데이터는 다목적 사용에 적합하게 구조화됩니다.



### Advancing Similarity Search with GenAI: A Retrieval Augmented Generation Approach (https://arxiv.org/abs/2501.04006)
- **What's New**: 이 논문에서는 혁신적인 Retrieval Augmented Generation 접근 방식을 사용하여 유사도 검색(simiarity search)을 다룹니다. 제안된 방법은 생성 모델(generative model)을 활용하여 미세한 의미 정보를 포착하고 고급 컨텍스트 이해를 기반으로 유사도 점수를 검색합니다. 이러한 접근 방식은 기존에 비해 향상된 성능을 보여줍니다.

- **Technical Details**: 연구는 생물 의학 분야에서 추출된 100쌍의 문장을 포함하는 BIOSSES 데이터셋에 초점을 맞추고 있습니다. 실험을 통해 모델의 민감도 분석을 수행하고, 0.5의 온도(temperture)와 20개의 예제(sample size)를 제공했을 때, 최고 유사도 검색 정확도를 달성하는 최적의 조건을 밝혔습니다. 이 조건에서 Pearson 상관 계수(Pearson correlation score)는 0.905에 이릅니다.

- **Performance Highlights**: 이 연구 결과는 생성 모델이 의미 정보 검색에 있어 강력한 잠재력을 가지고 있음을 다시 한 번 보여줍니다. 또한, 유사도 검색을 위한 유망한 연구 방향을 제시하고 있으며, 향후 더 나은 성능을 위한 기초를 제공합니다. 제안된 방법론은 생물 의학 분야에서의 다양성 있는 응용 가능성을 가지고 있습니다.



### Re-ranking the Context for Multimodal Retrieval Augmented Generation (https://arxiv.org/abs/2501.04695)
- **What's New**: 이번 논문은 Retrieval-augmented generation (RAG)을 활용하여 대형 언어 모델(LLMs)을 개선하기 위한 새로운 접근을 제안합니다. 특히, multi-modal RAG 시스템의 정보 검색(retrieval) 과정에서 관련성을 보다 정확하게 평가하기 위해 새롭게 개발한 relevancy score (RS) 모델을 도입했습니다. 기존 CLIP 기반의 방법의 한계를 극복하고 보다 정밀한 검색 결과를 얻기 위해 RS 모델을 사용하여 부정확한 정보를 제거하는 방법에 초점을 맞췄습니다.

- **Technical Details**: RS 모델은 VLM(vision-language models)을 활용하여 사용자 쿼리와 검색된 엔트리 간의 의미적 관계를 학습합니다. 훈련 데이터셋은 인간이 주석을 단 데이터와 ChatGPT로 생성된 합성 쿼리-컨텍스트 쌍을 포함하여 균형 잡힌 triplet 형태로 구성됩니다. RS 모델의 출력은 0에서 1까지의 스칼라 점수로, 높은 점수는 그만큼 쿼리와 높은 관련성을 나타냅니다.

- **Performance Highlights**: COCO 데이터셋을 사용한 평가 결과, RS 기반의 재정렬(re-ranking) 방법이 검색 이미지의 품질을 크게 향상시키고, 더 정확하고 사실 기반의 응답을 생성하는 것으로 나타났습니다. RS는 CLIP에 비해 의미적 유사성을 넘어 정확한 맥락적 관련성을 포착하는 데 뛰어난 성능을 보였으며, 검색 정밀성을 높여 hallucinations(비현실적 결과)를 줄일 수 있는 잠재력을 지니고 있음을 강조합니다.



### Multi-task retriever fine-tuning for domain-specific and efficient RAG (https://arxiv.org/abs/2501.04652)
Comments:
          9 pages, 2 figures. Submitted to NAACL 2025 Industry Track

- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG) 기술을 활용하여 Large Language Models (LLMs)의 성능을 향상시키기 위한 새로운 접근법을 제시합니다. 특히, 다양한 도메인 특정 작업에 대응할 수 있는 소형 retriever encoder를 instruction fine-tuning 방식으로 학습시켜, 여러 환경에서 활용 가능한 효과적인 솔루션을 제공합니다. 이를 통해 RAG 응용 프로그램의 가능한 확장성을 극대화하고, 비용과 처리 속도를 줄일 수 있습니다.

- **Technical Details**: 연구에서는 mGTE 모델을 fine-tune하여 생성된 데이터셋을 기반으로 소형 retriever를 다양한 작업에 맞춰 훈련합니다. retriever는 steps, table names, field names 등 다양한 구조화된 데이터를 데이터베이스에서 검색하여 LLM에 전달하는 방식으로, RAG 응용 프로그램의 결과물 품질을 높입니다. 본 연구는 OOD(Out-Of-Domain) 설정에서 retriever의 일반화 성능을 평가하며, 학습 데이터셋은 내부 데이터베이스와 Flow Generation training set에서 추출됩니다.

- **Performance Highlights**: 제안된 방법은 다양한 도메인과 관련된 과제를 해결하는 데 뛰어난 성능을 발휘합니다. 실험 결과, instruction fine-tuning을 통한 retriever 모델이 OOD 설정에서도 유의미한 성능 향상을 보여주며, 차별화된 data retrieval 과제를 해결할 수 있는 능력을 입증했습니다. 이를 통해, GenAI 응용 프로그램의 다양성과 효율성을 높이는 새로운 가능성을 열어줍니다.



### User Simulation in the Era of Generative AI: User Modeling, Synthetic Data Generation, and System Evaluation (https://arxiv.org/abs/2501.04410)
- **What's New**: 이번 논문에서는 Generative AI 시대에 등장한 사용자 시뮬레이션(user simulation)의 필요성과 그 응용 가능성에 대해 다루고 있습니다. 사용자가 AI 시스템과 상호작용하는 방식을 모방하는 지능형 에이전트를 생성함으로써, 연구자들은 사용자의 행동을 모델링하고 분석할 수 있으며, 이는 AI 시스템의 안전하고 책임감 있는 발전에 중요한 역할을 합니다. 또한, 사용자 시뮬레이션은 인공지능의 일반 지능(AGI) 개발에도 중대한 영향을 미칠 것으로 기대됩니다.

- **Technical Details**: 사용자 시뮬레이션은 사용자 행동을 기반으로 하여, 사용자의 의사결정 패턴을 모델링하는 과정입니다. 이를 위해 시스템의 특성, 사용자가 수행하는 작업의 종류, 사용자에 대한 정보와 같은 변수들이 고려되어야 합니다. 여러 유형의 사용자 행동을 효과적으로 시뮬레이션하기 위해서 Markov Decision Process(MDP)와 같은 계산적 모델을 활용할 수 있으며, 이로 인해 사용자 시뮬레이터가 다양한 사용자와 조건을 반영할 수 있도록 구성됩니다.

- **Performance Highlights**: 본 논문은 사용자 시뮬레이션이라는 주제를 심도 있게 탐구하며, 관련된 최신 연구 동향과 응용 분야를 정리합니다. 사용자 행동 모델링, 데이터 증대(data augmentation), 그리고 시스템 평가와 관련된 사례들(예: 대화 시스템의 현실적인 대화 생성 등)을 제시하여 사용자 시뮬레이션의 유용성을 강조합니다. 사용자 시뮬레이션은 실제 데이터 확보가 어려운 상황에서도 대량의 합성 데이터를 생성하고, AI 모델의 효율성을 개선하는 데 기여할 수 있는 잠재력을 지니고 있습니다.



### Reasoning-Enhanced Self-Training for Long-Form Personalized Text Generation (https://arxiv.org/abs/2501.04167)
- **What's New**: 본 논문은 개인화된 텍스트 생성을 위해 Reasoning-Enhanced Self-Training for Personalized Text Generation (REST-PG) 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLM)이 사용자에 대한 과거의 선호도, 배경 지식 및 작성 스타일을 기반으로 reasoning(추론)을 수행하도록 훈련합니다. 이를 통해 REST-PG는 모델이 개인화된 맥락을 보다 효과적으로 활용하게 함으로써, 사용자 기대에 부합하는 응답을 생성할 수 있도록 합니다.

- **Technical Details**: REST-PG는 LLM이 개인화된 데이터를 기반으로 응답을 생성하는 동안 reasoning을 진행할 수 있도록 하는 다단계 프레임워크입니다. 이 시스템은 Expectation-Maximization Reinforced Self-Training을 활용하여, 높은 보상을 받는 응답을 생성하는 reasoning 경로를 반복적으로 학습시킵니다. 초기 단계에서는 LLM이 스스로 reasoning 경로를 생성하고, 이후에는 이러한 경로를 토대로 어떻게 보상을 극대화할 수 있는지를 학습합니다.

- **Performance Highlights**: 실험 결과, REST-PG는 LongLaMP 벤치마크에서 기존 최첨단 모델들보다 평균 14.5% 향상된 성능을 보였습니다. 특히, supervised fine-tuning(SFT)과 비교해도 14.5%의 성능 향상이 있었으며, reasoning enhancement 없이 self-training을 진행한 경우보다도 6.5% 개선된 결과를 나타냈습니다. 이러한 성과는 REST-PG가 개인화된 텍스트 생성 분야에서 효과적인 접근법임을 입증합니다.



### KGIF: Optimizing Relation-Aware Recommendations with Knowledge Graph Information Fusion (https://arxiv.org/abs/2501.04161)
Comments:
          Published at IEEE Big Data 2024

- **What's New**: KGIF(지식 그래프 주의 네트워크)는 사용자-아이템 상호작용과 아이템-속성 관계를 명확하게 융합하는 특화된 프레임워크로, 추천 품질을 향상시키기 위해 맞춤형 self-attention 메커니즘을 활용합니다. 이 연구는 KGIF를 통해 지식 그래프 내의 복잡한 관계를 효과적으로 표현하고, 추론 과정을 시각적으로 해석할 수 있는 기능을 제공합니다. 또한, KGIF는 희소 지식 그래프에 대한 강인성을 개선하고, 설명 가능한 추천을 생성할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: KGIF는 초기 임베딩, 관계 정보 융합, 주의 임베딩 전파, 추천 생성의 네 가지 계층 구조로 구성됩니다. 초기 데이터는 사용자-아이템 이분 그래프와 아이템-엔티티 지식 그래프로 재구성되어 임베딩이 생성됩니다. 관계 특정 정보가 엔티티 임베딩에 명시적으로 융합되어 하위 작업의 표현이 풍부해지며, 주의 메커니즘을 통해 관련 정보가 그래프를 통해 전파됩니다. 이러한 과정을 통해 KGIF는 복잡하고 희소한 데이터를 보다 잘 처리할 수 있습니다.

- **Performance Highlights**: KGIF는 Amazon-book, Last-FM, Yelp2018와 같은 여러 벤치마크 데이터 세트에서 기존의 최첨단 기법(SOTA)을 혁신적으로 초월하는 성과를 보였습니다. 이러한 실험을 통해 KGIF의 임베딩 방식과 정보 융합에 대한 유효성을 확립하고, 추천 생성의 투명성을 높일 수 있는 시각적 해석을 제공했습니다. KGIF는 사용자와 아이템 모두에 대한 복잡한 관계를 효과적으로 포착하는 데 있어 큰 기여를 하고 있습니다.



### A Generative AI-driven Metadata Modelling Approach (https://arxiv.org/abs/2501.04008)
Comments:
          Accepted for publication @ Special Issue on "Generative AI and Libraries" - Library Trends Journal, Johns Hopkins University Press, Maryland, USA

- **What's New**: 이 논문은 최신 연구 동향 중 하나인 Generative AI와 연계하여 도서관 메타데이터 모델을 개선하는 방법을 제시합니다. 특히, 기존의 몇 가지 핵심 메타데이터 모델에 대한 의존성을 극복하고, 다양한 정보 서비스에서 필요로 하는 모델의 복잡성을 해결하는 접근 방식을 소개합니다. 이를 통해 도서관의 정보 서비스가 더 효과적이고 유연해질 수 있음을 강조합니다.

- **Technical Details**: 저자는 메타데이터 모델을 다섯 가지 기능적으로 연결된 표현 수준의 조합으로 구성된 온톨로지 기반 모델로 재구성합니다. 이 과정에서는 각 표현 수준의 내재적 표현 다면성(representational manifoldness)이 강조되며, 이를 통해 복잡한 정보 환경에서의 메타데이터 모델 재사용성을 증대시킬 수 있음을 설명합니다. 마지막으로, Generative AI와 Human-Large Language Model(LLM)의 협업을 바탕으로 한 메타데이터 모델링 방식을 제안합니다.

- **Performance Highlights**: 이 연구는 암 정보 처리에 유능한 대표적인 도서관의 사례를 들어, 제안된 메타데이터 모델이 어떻게 복잡성을 줄이고 명확한 정보 제공을 가능하게 하는지를 보여줍니다. 연구 결과는 도서관들이 Generative AI를 활용하여 사용자 맞춤형 정보 서비스를 제공할 수 있음을 시사하며, 새로운 메타데이터 모델이 정보 접근성을 크게 향상할 수 있는 가능성도 깔고 있습니다.



New uploads on arXiv(cs.CV)

### EditAR: Unified Conditional Generation with Autoregressive Models (https://arxiv.org/abs/2501.04699)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 EditAR을 통해 다양한 조건부 이미지 생성 작업을 통합 관리할 수 있는 새로운 자가회귀(autoregressive) 프레임워크를 제안합니다. 기존의 Diffusion 모델들이 특정 작업에 특화되어 있었다면, EditAR은 텍스트와 이미지를 모두 이용하여 수정된 이미지 토큰을 예측하는 방식으로 동작합니다. 이 모델은 단일의 기초 모델을 구축할 수 있는 가능성을 제시하며, 각기 다른 작업에서 검증된 성능을 보여줍니다.

- **Technical Details**: EditAR은 VQVAE와 자가회귀 변환기(autoregressive transformer)로 구성되어 있으며, 입력으로는 이미지와 텍스트 명령어를 받습니다. 이 모델은 두 가지 단계로 이루어져 있으며, 첫 번째로 비트맵 패치를 토큰 인덱스로 변환하고, 두 번째로 주어진 텍스트와 이미지를 기반으로 출력 토큰의 범주적 분포를 모델링합니다. 또한, DINOv2를 기반으로 시각적 일관성을 강화하는 보조 증류 손실이 포함되어 있습니다.

- **Performance Highlights**: EditAR은 다양한 이미지 조작 작업, 예를 들어 텍스처 조작, 객체 교체 및 제거에서 관찰된 강력한 성능을 보여줍니다. 연구 결과는 EditAR이 전통적인 조건부 모델들에서 해결하기 힘든 여러 작업을 해결할 수 있는 잠재력을 가지고 있음을 나타냅니다. 특히, EditAR은 대규모 벤치를 통해서도 다양성 있는 조건부 이미지 생성 작업에서 경쟁력 있는 성과를 입증했습니다.



### ConceptMaster: Multi-Concept Video Customization on Diffusion Transformer Models Without Test-Time Tuning (https://arxiv.org/abs/2501.04698)
Comments:
          Project Page: this https URL

- **What's New**: 이번 논문에서는 Multi-Concept Video Customization (MCVC)이라는 어려운 과제를 해결하기 위해 ConceptMaster라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 아이디어 디커플링(Identity Decoupling) 문제를 해결하고, 사용자 정의 개념의 신뢰성을 유지하면서도 여러 개념을 동시에 처리할 수 있는 능력을 갖추었습니다. 또한, 사용자 개념에 맞춘 고품질 비디오 생성을 위한 데이터 수집 파이프라인을 구축하여 130만 개 이상의 비디오-엔티티 쌍을 확보했습니다.

- **Technical Details**: ConceptMaster의 핵심 기능은 분리된 다중 개념 임베딩(multi-concept embeddings)을 학습하고 이를 디퓨전(transformer) 모델에 주입하는 것입니다. 이 과정에서는 CLIP 이미지 인코더를 사용하여 밀집 비주얼 토큰(dense visual tokens)을 추출하고, 고유 쿼리(transformer) 네트워크를 통해 각 개념과의 정합성을 높입니다. 또한, Decouple Attention Module (DAM)을 도입하여 각 개념의 시각적 임베딩과 텍스트 임베딩을 개별적으로 결합하여 개념의 고유성을 유지합니다.

- **Performance Highlights**: 종합적인 벤치마크 테스트를 통해 ConceptMaster는 기존 방법들보다 우수한 성능을 보여주었습니다. 본 연구의 결과는 개념 신뢰성, 아이디어 디커플링 능력, 비디오 생성 품질 등 6가지 다양한 개념 조합 시나리오에서 평가되었습니다. 실험 결과, ConceptMaster는 개인화되고 의미적으로 정확한 비디오를 생성하는 데 있어 이전의 최첨단 방법들을 초월하는 성과를 기록했습니다.



### Test-Time Optimization for Domain Adaptive Open Vocabulary Segmentation (https://arxiv.org/abs/2501.04696)
- **What's New**: Seg-TTO는 제로샷(Zero-Shot) 오픈 어휘(Open-Vocabulary) 의미 분할(Semantic Segmentation)을 위한 새로운 프레임워크로, 특히 전문 도메인 작업에서 뛰어난 성능을 발휘합니다. 기존의 오픈 어휘 접근법들은 일반적인 분할 벤치마크에서 인상적인 성능을 보이지만, 특화된 데이터셋에서는 감독(Supervised) 방식에 비해 부족한 성능을 보입니다. 이 연구는 이러한 격차를 해소하기 위해 세그멘테이션 특화 시험 시간 최적화(Test-Time Optimization) 전략에 집중하고 있습니다.

- **Technical Details**: Seg-TTO는 입력 이미지와 모델 매개변수를 정렬하기 위한 새로운 자기 지도(Self-Supervised) 목표를 제안합니다. 이 목표는 다중 개념을 이해하고 지역성(Locality) 및 공간 구조(Spatial Structure)를 보존하는 것을 요구합니다. 시각적 모달리티(Visual Modality)에서는 픽셀 수준 손실(Pixel-Level Losses)을 계산하고 임베딩 집합(Embedding Aggregation) 작업을 통해 지역 구조를 유지합니다. 또한, 각 범주에 대한 다중 임베딩을 학습하여 이미지 내 다양한 개념을 포착하는 방식을 활용합니다.

- **Performance Highlights**: Seg-TTO는 세 개의 최첨단 오픈 어휘 의미 분할 접근법과 통합되어 22개의 복잡한 OVSS 작업을 평가합니다. 그 결과 Seg-TTO는 여러 전문 분야에서 분명한 성능 향상을 보여주어 새로운 최첨단 성능을 설정했습니다. 이 연구는 Seg-TTO가 기존 OVSS 모델의 도메인 외 성능을 개선할 수 있는 플러그-인-플레이(Plug-and-Play) 접근법임을 강조합니다.



### SPAR3D: Stable Point-Aware Reconstruction of 3D Objects from Single Images (https://arxiv.org/abs/2501.04689)
- **What's New**: 최근 단일 이미지 3D 객체 재구축의 연구가 두 가지 방향으로 나누어졌습니다: 회귀 기반 모델링 및 생성 모델링. 그러나 기존 모델들은 각각의 한계를 가지고 있으며, 제안된 SPAR3D는 이러한 두 가지 접근을 결합한 새로운 두 단계 모델입니다. 첫 단계는 경량의 point diffusion 모델을 사용해 희소한 3D 포인트 클라우드를 생성하고, 두 번째 단계는 이를 바탕으로 고해상도의 메쉬를 만들어 냅니다.

- **Technical Details**: SPAR3D는 두 단계로 구성된 구조를 가지며, 첫 번째 단계에서 점 샘플링에 diffusion 모델을 활용하여 빠르고 효율적으로 희소 포인트 클라우드를 생성합니다. 이어지는 메싱 단계에서는 입력 이미지와 샘플링된 포인트 클라우드를 결합하여 고해상도의 메쉬를 생성합니다. 이 과정은 비지도 학습을 통해 조명 및 재질을 자동으로 학습하는 데 기여하고, 이는 텍스처 내에 내장된 조명 아티팩트 문제를 줄입니다.

- **Performance Highlights**: SPAR3D는 다양한 데이터셋에서 실시된 평가에서 이전의 최첨단 방법들에 비해 뛰어난 성능을 나타냈습니다. 추론 속도는 0.7초 이내로, 효율성과 사용자 편집 기능을 모두 갖춘 솔루션을 제공합니다. 또한, 실시간 이미지나 AI 생성 이미지에 대해서도 강력한 일반화 능력을 나타내며, 이는 고품질 3D 자산 생성에 중요한 의미를 갖습니다.



### DRIVINGVQA: Analyzing Visual Chain-of-Thought Reasoning of Vision Language Models in Real-World Scenarios with Driving Theory Tests (https://arxiv.org/abs/2501.04671)
- **What's New**: 최근의 대형 비전-언어 모델(LVLMs)은 언어 모델에 비주얼 이해를 통합하여 다중 모달(Modal) 추론을 가능하게 합니다. 그러나 텍스트와 비주얼 데이터 사이의 모달 간 격차로 인해, 이 모델들은 텍스트 우선 의존성(over-reliance on text priors), 환각(hallucinations) 및 복잡한 비주얼 추론에 대한 제한된 능력과 같은 도전에 직면합니다. 이를 해결하기 위해, 우리는 DrivingVQA라는 새로운 벤치마크를 제안하여 복잡한 실제 상황에서 비주얼 체인 오프 씽킹(visua chain-of-thought reasoning)을 평가합니다.

- **Technical Details**: DrivingVQA는 운전 이론 테스트에서 파생된 데이터셋으로, 총 3,931개의 전문가 제작 다중 선택 문제(multiple-choice problems)를 포함하고 있습니다. 각 문제는 추론 과정과 관련된 엔티티(entities)에 기반한 교차 설명(interleaved explanations)을 제공합니다. 우리는 이 데이터셋을 활용하여 LVLMs의 복잡한 비주얼 시나리오에 대한 추론 능력을 광범위하게 연구하였습니다. 실험 결과, 오픈 소스와 상용 LVLM들이 제로샷(zeroshot) 설정 하에서 비주얼 체인 오프 씽킹에 어려움을 겪고 있음을 발견했습니다.

- **Performance Highlights**: 특히, 이미지 토큰(image tokens)의 자르는 지역(cropped regions)에 기반한 엔티티를 활용할 때, 비주얼 추론이 최대 7% 향상되는 성과를 보였습니다. 이는 현재 모델들이 지역의 관심 영역(localization of regions of interests)에서 정보를 효과적으로 활용하지 못하고 있다는 것을 보여줍니다. 따라서, LVLM의 비주얼 추론 개선을 위한 관련 엔티티 활용 전략을 탐구하는 것이 중요합니다.



### Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs (https://arxiv.org/abs/2501.04670)
Comments:
          project page: this https URL

- **What's New**: 이 연구는 최근 Multimodal Large Language Models (MLLMs)의 시각적 일치(matching) 능력을 탐구하며, 기존 LLMs가 겪는 한계를 밝혀냈습니다. 연구팀은 이를 통해 30종 이상의 MLLM을 공정하게 평가할 수 있는 Multimodal Visual Matching (MMVM) 벤치마크를 구축했습니다. 이는 15개의 오픈소스 데이터셋과 인터넷 비디오로부터 수집된 샘플을 바탕으로 만듭니다.

- **Technical Details**: MMVM 벤치마크는 8개의 주요 요소를 기반으로 데이터 샘플을 카테고리화하여 MLLM의 시각적 일치 능력을 더욱 포괄적으로 평가합니다. 이 연구는 220K 시각적 일치 데이터와 추론 주석을 포함하는 자동 주석 파이프라인도 설계했습니다. CoLVA라는 새로운 MLLM 모델은 물체 수준 대조 학습(Object-level Contrastive Learning)과 지침 증대 전략(Instruction Augmentation Strategy)을 활용합니다.

- **Performance Highlights**: CoLVA는 MMVM 벤치마크에서 51.06%의 전반적 정확도(Overall Accuracy)를 달성했으며, 이는 GPT-4o보다 8.41% 향상된 수치입니다. 연구 결과는 MMVM SFT 데이터셋의 효과성을 입증하며, 새로운 기술 설계가 MLLM의 시각적 일치 능력을 크게 향상시킬 수 있음을 보여줍니다. 이 연구의 코드, 벤치마크, 데이터셋, 모델은 제공된 URL에서 확인할 수 있습니다.



### Enhancing Virtual Try-On with Synthetic Pairs and Error-Aware Noise Scheduling (https://arxiv.org/abs/2501.04666)
- **What's New**: 본 논문에서는 가상 착용(Virtual Try-On) 모델의 성능 향상을 위한 새로운 접근 방식을 소개합니다. 특히, (인간, 합성 의류) 쌍을 이용한 데이터 보강 방법과 발생하는 오류를 정교하게 수정할 수 있는 EARSB(오류 인식 기반 슈뢰딩거 브리지) 모델을 제안합니다. 이로써 기존 모델에서 나타나는 질감 왜곡 및 텍스트 흐림 문제를 해결하고자 합니다.

- **Technical Details**: EARSB는 주어진 이미지에서 발생하는 국소적 생성 오류를 식별하여 이를 보정하는 방법론입니다. 비지도 학습 방식으로 오류를 분류하는 클래스파이어를 통해 생성 이미지의 저품질 영역을 강조합니다. 추가적으로, 기존의 노이즈 스케줄에 오류 예측 정보를 통합하여 국소 오류에 집중하는 방법론을 개발하였습니다.

- **Performance Highlights**: 실험 결과에 따르면, VITON-HD와 DressCode-Upper 데이터셋에서 EARSB는 기존 모델에 비해 이미지 생성 품질이 향상되었습니다. 사용자 연구에서도, 사용자의 평균 59%가 우리 모델을 선호하는 결과를 보였습니다. 이러한 향상은 합성 데이터 보강과 오류 수정 기법을 통해 이루어졌습니다.



### Discrete Wavelet Transform-Based Capsule Network for Hyperspectral Image Classification (https://arxiv.org/abs/2501.04643)
Comments:
          28 Pages; 9 Figure

- **What's New**: 본 논문에서는 하이퍼스펙트럴 이미지(HSI) 분류를 위한 새로운 접근 방식인 DWT-CapsNet을 제안합니다. 이 방법은 CapsNet의 부분적이고 중요한 연결성을 파악하여 HSI 분류의 효과성과 효율성을 극대화하는 데 중점을 두고 있습니다. 기존의 복잡한 연결 구조를 개선하여 연산 요구 사항을 줄이는 동시에 성능을 유지하고 있습니다.

- **Technical Details**: DWT-CapsNet은 Discrete Wavelet Transform(DWT) 기반의 다운샘플링 계층과 맞춤형 주의 메커니즘을 통합하여 정보 손실 문제를 완화합니다. 또한, 다중 스케일 루팅 알고리즘을 도입하여 CapsNet의 연결성을 대폭 줄이고, 다층의 스펙트럼-공간 관계를 집계하는 캡슐 피라미드 융합 메커니즘을 설계하였습니다. 이러한 구조는 부분적이고 국소적으로 연결된 아키텍처에서 의미 있는 관계를 강조하는 자기 주의 메커니즘을 통해 강화됩니다.

- **Performance Highlights**: 실험 결과, DWT-CapsNet은 낮은 연산 요구 사항(실행 시간, flops, 파라미터 수)에도 불구하고 최상의 정확도를 달성했습니다. 이는 지구 모니터링 시스템의 실제 응용 분야에 매력적인 선택이 될 수 있음을 나타냅니다. DWT-CapsNet은 HSI 분류에서 전통적인 방법보다 더 나은 성능을 보여주는 혁신적인 접근 방식으로 부각되고 있습니다.



### Disentangled Clothed Avatar Generation with Layered Representation (https://arxiv.org/abs/2501.04631)
Comments:
          project page: this https URL

- **What's New**: 본 논문에서는 각 구성 요소가 분리된 의상 아바타를 생성할 수 있는 첫 번째 피드포워드 확산 기반 방법인 LayerAvatar를 제안합니다. 이전의 방법들은 의상, 머리카락 등이 결합된 형태로 아바타를 생성하는 데 한계를 보였으나, LayerAvatar는 이를 해결하여 시간 소모 없이 아바타를 생성할 수 있습니다. 또한, 이 방법은 고해상도와 실시간 렌더링을 지원하며 감정 및 제스처 조작을 가능하게 합니다.

- **Technical Details**: LayerAvatar는 각 구성 요소가 논리적으로 분리된 형태로 아바타를 생성하기 위해, 다양한 구성 요소를 Gaussian 기반의 UV 특성 평면에 분포시키는 새로운 표현 방식을 도입했습니다. 이 UV 평면은 여러 레이어로 나뉘어 있어, 각 아바타의 구성 요소를 독립적으로 캡처할 수 있습니다. 이 접근 방식은 여러 데이터 세트를 통해 검증되었으며, 각 구성 요소의 효과적인 애니메이션과 변환을 지원합니다.

- **Performance Highlights**: LayerAvatar는 여러 데이터 세트에서 뛰어난 성능을 보여주었으며, 이를 통해 고해상도의 디지털 아바타를 생성하는 동시에 각 구성 요소의 이동과 변환을 원활하게 처리할 수 있습니다. 생성된 아바타는 1024 해상도와 실시간 렌더링을 지원하며, 제스처와 표정의 조절 능력을 특징으로 합니다. 이 기술은 아바타의 의상과 머리카락 등 구성 요소를 효율적으로 전송할 수 있는 응용 가능성을 보여줍니다.



### FatesGS: Fast and Accurate Sparse-View Surface Reconstruction using Gaussian Splatting with Depth-Feature Consistency (https://arxiv.org/abs/2501.04628)
Comments:
          Accepted by AAAI 2025. Project page: this https URL

- **What's New**: 이번 논문에서는 최신의 sparse-view reconstruction 프레임워크를 제안하고 있습니다. 이 프레임워크는 Gaussian Splatting의 효율적인 파이프라인을 활용하여 깊이 분포의 일관성과 패치 내에서 깊이 순위를 관리하는 방식을 채택합니다. 이러한 접근은 노이즈와 불완전한 재구성을 방지하며, 신속하고 정교한 메쉬 재구성을 가능하게 합니다. 특히, 기존 방법에 비해 60배에서 200배 빠른 속도를 보이며, 고비용의 사전 학습이 필요 없습니다.

- **Technical Details**: 논문에서 제안하는 방식은 3D 엘립소이드 Gaussian을 2D 엘립스 Gaussian으로 변환함으로써 기하학적 표현의 정확성을 향상시킵니다. 또한, 깊이 분포의 연속성을 높이기 위해 이미지 패치를 세분화하고, monocular depth 정보로 순위 관계를 조정합니다. 이 과정에서, 갑작스러운 깊이 변화 문제를 해결하기 위해 smoothness loss를 도입하여, 특히 텍스처가 없는 영역에서 깊이 분포의 연속성을 확보합니다. 마지막으로, multi-view feature 일관성을 최적화하여 높은 품질의 표면 재구성을 이룹니다.

- **Performance Highlights**: 실험 결과에 따르면, DTU와 BlendedMVS 데이터셋에서 제안한 방법은 기존 최첨단 기법들을 초월한 성능을 보여주었습니다. 본 기술은 sparse views에서 표면 재구성의 정확도를 높이면서도 신속한 결과를 제공합니다. 이러한 성과는 복잡한 데이터 처리 과정 없이도 도출되었으며, 특히 시각적 콘텐츠 생성이나 로봇 비전 분야에 큰 영향을 미칠 것으로 예상됩니다.



### Enhancing Low-Cost Video Editing with Lightweight Adaptors and Temporal-Aware Inversion (https://arxiv.org/abs/2501.04606)
- **What's New**: 최근의 텍스트-비디오 (T2V) 생성 및 편집 기술은 텍스트 설명을 충실히 해석하여 동적인 영상을 생성하고 수정하는 데 중점을 두고 있습니다. 이 논문에서는 비용 효율적인 T2V 작업을 위해, General and Efficient Adapter (GE-Adapter)를 제안합니다. 이 프레임워크는 Frame-based Temporal Consistency Blocks (FTC Blocks), Channel-dependent Spatial Consistency Blocks (SCD Blocks), 및 Token-based Semantic Consistency Module (TSC Module)로 구성되어 있어, 더욱 향상된 영상 품질과 시간적 일관성을 제공합니다.

- **Technical Details**: 이 연구는 시간적-공간적 및 의미적 일관성을 유지하기 위해 three key components로 구성된 GE-Adapter를 제안합니다. FTC Blocks는 프레임별 특징을 캡처하고 매끄러운 전이 (transition)를 보장하기 위해 시간 인지 손실 함수를 사용합니다. SCD Blocks는 bilateral filters를 적용하여 공간적 일관성을 향상시키고, TSC Module은 공유된 프롬프트 토큰과 프레임 특화 토큰을 활용하여 의미적 정렬을 유지합니다.

- **Performance Highlights**: 논문에서 제안한 방법은 MSR-VTT 데이터셋에서 감지된 품질, 텍스트-이미지 정렬 및 시간적 일관성을 크게 향상시킵니다. 제안된 GE-Adapter는 기존의 T2V 모델에 비해 50% 이상의 효율성을 개선하고, 시간적 일관성, 의미적 정렬 및 영상 품질을 증대시키는 실용적인 솔루션을 제공합니다.



### Identity-Preserving Video Dubbing Using Motion Warping (https://arxiv.org/abs/2501.04586)
Comments:
          Under Review

- **What's New**: 이번 논문에서는 IPTalker라는 새로운 프레임워크를 제안하여 오디오 신호와 참조 영상 간의 정밀한 정렬을 통해 비디오 더빙의 신뢰성을 향상시킵니다. IPTalker는 기존 비디오 더빙 기술의 한계를 극복하고, 고유한 피부 질감 및 구조적 세부 정보를 보존하면서도 높은 품질의 얼굴 애니메이션을 생성합니다. 이 프레임워크는 오디오 피처와 참조 영상 간의 상관 관계를 동적으로 캡처하는 transformer 기반 정렬 메커니즘을 중심으로 합니다.

- **Technical Details**: IPTalker의 핵심은 Audio-Visual Alignment Unit (AVAU)라는 transformer 기반 구성품으로, 오디오 피처와 참조 이미지 간의 정Correspondence embedding을 학습합니다. 특히, Spatial Deformation Module을 통해 예측된 motion flow에 따라 참조 이미지를 왜곡하여 오디오 드리븐 구성에 맞게 변형합니다. 이 과정에서 Inpainting Module은 occlusion 아티팩트를 제거하고 세부 텍스처를 보존하여 최종 영상 품질을 개선합니다.

- **Performance Highlights**: IPTalker는 VFHQ 및 HDTF 데이터셋에서의 실험을 통해 기존 방법들에 비해 사실감, 시간 일관성, 정체성 보존 및 립 싱크 정확도면에서 우수한 성능을 보입니다. 특히, 저희 방법은 현실적인 비디오를 생성하면서도 참조 신원의 정보 보존에 있어 최첨단 성능을 달성했습니다. 이 방법은 AR/VR, 오락 및 커뮤니케이션 기술 등 다양한 응용 분야에서 신뢰할 수 있는 솔루션을 제공할 수 있을 것으로 기대됩니다.



### Boosting Salient Object Detection with Knowledge Distillated from Large Foundation Models (https://arxiv.org/abs/2501.04582)
- **What's New**: 이 논문에서는 Salient Object Detection(SOD)에 대한 접근 방식을 혁신적으로 개선하는 새로운 데이터셋인 BDS-TR을 소개합니다. BDS-TR은 기존의 DUTS-TR 데이터셋보다 더 다양하고 넓은 범위의 객체 및 장면을 포함하고 있으며, 약 260,000개의 이미지를 제공합니다. 또한 동적 업샘플링을 기반으로 한 경계 보존 디코더(DEDecoder)를 제안하여 이미지를 복원하는 동안 관리를 지원합니다.

- **Technical Details**: 이 연구는 약한 감독(weakly supervised) 접근 방식을 통해 대형 모델을 이용하여 정밀한 의사 라벨(pseudo-labels)을 생성합니다. 이 과정에서는 먼저 Blip 모델을 사용하여 이미지에 대한 텍스트 설명을 생성하고, 이 텍스트를 바탕으로 GroundingDINO를 통해 경계 상자(bounding box)를 생성하였습니다. 이후 SAM을 활용하여 최종적으로 SOD를 위한 의사 라벨을 생성합니다.

- **Performance Highlights**: 다섯 개의 벤치마크 데이터셋에서 수행된 광범위한 실험 결과, 본 기법은 기존의 최첨단 알고리즘을 초월하며, 몇몇 완전 감독(supervised) SOD 방법을 능가하는 성능을 보여주었습니다. 이 모델은 태스크에 대해 더 넓은 범위의 일반화 능력을 향상시킬 수 있는 잠재력을 지니고 있으며, 향후 연구에 유익한 기반을 제공합니다.



### Unified Coding for Both Human Perception and Generalized Machine Analytics with CLIP Supervision (https://arxiv.org/abs/2501.04579)
Comments:
          9 pages, 10 figures, publised to AAAI 2025

- **What's New**: 이 논문에서는 이미지 압축 모델의 일반성과 적응성을 개선하기 위해 새로운 접근 방식인 UG-ICM(Unified and Generalized Image Coding for Machine)을 제안합니다. 이 모델은 인간의 시각 인식과 기계 비전을 동시에 지원하기 위해 멀티모달 사전 학습 모델에서 얻은 감독 정보(supervision)를 활용합니다. 또한, 컨트롤 가능한 매개변수를 조정하여 단일 비트스트림에서 인간 또는 기계의 선호도에 맞춘 다양한 버전을 제공할 수 있는 조건부 디코딩 전략을 도입합니다.

- **Technical Details**: 제안된 UG-ICM의 주요 구성 요소는 Preference Conditional Decoding Module(PCDM)입니다. PCDM은 인간 또는 기계의 선호에 따른 바이어스 특징을 생성하고, 이를 이미지 특징과 결합하여 디코딩 과정을 안내합니다. 또한, Contrastive Language-Image Pre-training (CLIP) 모델을 이용하여 다양한 수준의 계층적 의미를 포착하는 다중 척도 CLIP 손실(MS-CLIP loss)을 적용하여 일반화를 향상시킵니다.

- **Performance Highlights**: 광범위한 실험을 통해 UG-ICM은 다양한 기계 분석 작업에서 주목할 만한 성과를 달성하며, 동시에 시각적으로 만족스러운 이미지를 제공합니다. 이 모델은 기계 분석 성능 향상과 함께 인식 품질을 개선하여, 기존 최첨단 ICM 작업을 능가합니다. 이러한 결과는 UG-ICM이 인간과 기계의 필요를 충족시키는 유망한 접근 방식임을 보여줍니다.



### Supervision-free Vision-Language Alignmen (https://arxiv.org/abs/2501.04568)
Comments:
          Preprint

- **What's New**: 이 논문에서는 SVP(Supervision-free Visual Projection)라는 새로운 프레임워크를 소개합니다. VLMs(비전-언어 모델)의 성능 향상에 초점을 맞추며, 이는 비급식 데이터나 선호 주석 없이도 가능하다는 점에서 이전 연구들과 차별화됩니다. SVP는 자기 캡셔닝(self-captioning)과 사전 훈련된 그라운딩 모델(pre-trained grounding model)을 활용하여 VLM의 잠재 정보를 이끌어내는 피드백 메커니즘을 이용합니다.

- **Technical Details**: SVP는 크게 이미지-텍스트 쌍의 수집이 필요하지 않은 점이 특징이며, 이를 통해 비전-언어 정합성(vision-language alignment)을 개선합니다. 연구에서는 캡셔닝(captioning), 참조(referring), 비주얼 질문 응답(visual question answering), 멀티태스킹(multitasking), 할루시네이션 제어(hallucination control), 객체 회상(object recall) 등 여섯 가지 주요 영역에서 평가가 이루어졌습니다.

- **Performance Highlights**: SVP를 적용한 결과, 캡셔닝 작업에서 평균 14%의 성능 향상, 객체 회상에서 최대 12% 증가, 할루시네이션 비율 대폭 감소 등 주요 성과가 보고되었습니다. 특히, SVP를 활용한 작은 VLM이 원래 크기가 다섯 배 큰 모델과 비교할 만한 수준으로 할루시네이션을 줄인 점이 주목할 만합니다.



### Learnable Scaled Gradient Descent for Guaranteed Robust Tensor PCA (https://arxiv.org/abs/2501.04565)
- **What's New**: 이 논문은 최초로 t-SVD 프레임워크 내에서 효율적인 Scaled Gradient Descent (SGD) 접근법을 탐구하며, 이를 기반으로 하는 RTPCA-SGD 방법을 제안합니다. 기존의 tensor nuclear norm (TNN)에 의존하지 않으면서, 저차원 텐서를 선형 수렴하는 방식으로 복구할 수 있는 이론적 보장을 수립합니다. 또한, RTPCA-SGD의 실용적 적용을 위해 배우기 가능한 self-supervised deep unfolding 모델인 RTPCA-LSGD를 추가로 제안하여 효과적인 파라미터 학습을 가능하게 합니다.

- **Technical Details**: RTPCA-SGD는 저차원 구성 요소를 두 개의 작은 요소로 분해하고, 이들 요소를 gradient descent 알고리즘으로 업데이트하는 방식으로 작동합니다. 스케일링 팩터를 도입하여 수렴 속도가 condition number에 종속되지 않도록 조정합니다. 또한 이 방법은 일반 비대칭 경우를 처리할 수 있도록 설계되었으며, 대칭 positive semi-definite (PSD) 텐서에 대해 일반적으로 적용될 수 있습니다.

- **Performance Highlights**: 수치 실험 결과, RTPCA-SGD는 복구 정확도에서 최신 기법을 능가하면서도 계산 효율성을 유지하며, 특히 RTPCA-TNN보다 더 적은 시간을 소모하는 것으로 나타났습니다. 추가적으로, RTPCA-LSGD는 배우기 가능한 파라미터 최적화를 통해 성능 향상을 제공했음을 증명하였습니다.



### Combining YOLO and Visual Rhythm for Vehicle Counting (https://arxiv.org/abs/2501.04534)
Comments:
          Accepted for presentation at the Conference on Graphics, Patterns and Images (SIBGRAPI) 2023

- **What's New**: 이 논문에서는 전통적인 이미지 기반 차량 검출 및 카운팅 방식의 두 가지 주요 단계인 초기 검출 및 이후 추적 단계를 제거한 새로운 방법을 제안합니다. 제안된 방법은 키 비디오 프레임에서만 차량을 검출하는 데 초점을 맞추어 효율성을 크게 향상시킵니다. 특히 YOLO와 Visual Rhythm을 결합하여 유용한 정보가 포함된 프레임에 초점을 맞춰 차량을 검출합니다.

- **Technical Details**: YOLO(You Only Look Once)는 객체 검출을 위한 실시간 딥러닝 모델로, 컴퓨터 비전에서 중요한 역할을 맡고 있습니다. 이 논문에서는 Visual Rhythm(VR) 기법을 적용하여 비디오 프레임의 시간-공간 이미지를 생성하고, 이 이미지를 통해 차량 검출 및 카운팅을 위한 주요 프레임을 선택하도록 설계되었습니다. VR 이미지는 정의된 카운팅 라인 근처의 픽셀만 포함하여, 비디오의 정보를 효율적으로 집약합니다.

- **Performance Highlights**: 실험 분석 결과, 제안된 방법은 여러 비디오 세트를 통해 평균 99.15%의 차량 카운팅 정확도를 달성하였으며, 이는 추적 기반 접근 방식에 비해 처리 속도가 3배 빠릅니다. 따라서 이 접근 방식은 단방향으로 이동하는 대상의 검출 및 식별이 필요한 다양한 응용 프로그램에서도 효과적으로 사용될 수 있습니다.



### Improving Image Captioning by Mimicking Human Reformulation Feedback at Inference-tim (https://arxiv.org/abs/2501.04513)
- **What's New**: 본 논문에서는 생성 모델 훈련에 인간 피드백을 자동으로 예측하여 반영하려는 최근의 관심을 바탕으로, 새로운 피드백 유형인 caption reformulations (캡션 개편)을 도입하고 이를 바탕으로 모델을 훈련합니다. 이 방법은 기존의 이미지 캡셔닝 모델에 추가적인 훈련 없이도 쉽게 적용할 수 있어 컴퓨팅 자원을 크게 절약합니다. 특히, 비영어 이미지 캡셔닝 영역에서도 개선된 성능을 보여 주목할 만합니다.

- **Technical Details**: 연구에서 도입한 reformulation 피드백은 기존의 비교 피드백과 달리 이미지 캡션의 오류를 교정하는 방식으로 구성됩니다. 연구팀은 몇 천 개의 샘플을 수집하여 이를 기반으로 모델을 훈련하고, 훈련된 모델을 기존 캡셔닝 모델의 추론 단계에 통합하여 활용합니다. 이 과정에서 'challenge domains'에서의 특별한 유용성을 강조하며, 독일어 이미지 캡셔닝에서도 뛰어난 성능을 발휘하는 것을 확인했습니다.

- **Performance Highlights**: 자동화된 캡션 개편 과정은 질이 낮은 모델에서 생성된 캡션의 품질을 크게 향상시키며, 특히 없는 정보를 추가함으로써 개선된 결과를 도출합니다. 스타일 전이 작업에서도 기존의 캡셔닝 모델과 비교하여 더 뛰어난 성능을 발휘하며, 메시지의 구조를 유지하면서 타겟 스타일로 수정할 수 있는 가능성을 보여줍니다. 주어진 데이터셋에서 이 방법은 최신 기술 수준을 달성하며, 인간 평가에서도 향상된 캡션 스타일을 증명했습니다.



### MB-TaylorFormer V2: Improved Multi-branch Linear Transformer Expanded by Taylor Formula for Image Restoration (https://arxiv.org/abs/2501.04486)
- **What's New**: 본 논문에서는 이미지 복원(image restoration) 분야에서 Transformer 네트워크의 효율성을 높이기 위한 새로운 변형 모델, 즉 MB-TaylorFormer V2를 제안합니다. 이 모델은 Taylor 확장을 활용하여 Softmax-attention을 근사화하고, 노름 보존 맵핑(norm-preserving mapping) 개념을 통해 첫 번째 차수 Taylor 확장의 나머지를 근사합니다. 여기서 중요한 점은 계산 복잡도를 선형으로 감소시키며, 다양한 수용 영역(receptive field) 및 다중 수준의 의미 정보(multi-level semantic information)를 활용한 멀티 브랜치 아키텍처가 도입되었다는 것입니다.

- **Technical Details**: MB-TaylorFormer V2는 T-MSA++라는 새로운 주의(attention) 메커니즘을 적용하여 이미지 복원에서의 장거리 픽셀 상호작용(long-distance pixel interactions)을 효과적으로 모델링할 수 있도록 설계되었습니다. 이 메커니즘은 Softmax-attention의 1차 Taylor 확장을 사용하여 비선형성을 나타내며, 픽셀 수준의 상호작용을 강조합니다. 또한, 멀티-브랜치 아키텍처는 서로 다른 크기 및 차원의 패치(patch)를 사용하여 유연하고 빠른 신경망 처리(performance)를 가능하게 합니다.

- **Performance Highlights**: 실험 결과 MB-TaylorFormer V2는 이미지 디헤이징(image dehazing), 디레인(demaraining), 디스노우(dessnowing), 모션 블러(motion deblurring), 노이즈 제거(denoising)와 같은 다양한 이미지 복원 과제에서 최첨단 성능(state-of-the-art performance)을 달성하였습니다. 기존 모델들에 비해 계산 오버헤드가 적고, 프로세스 처리 속도와 효율성에서 향상된 결과를 보여줍니다. 이러한 결과는 MB-TaylorFormer V2가 가진 우수한 성능과 신뢰성을 입증합니다.



### Rethinking High-speed Image Reconstruction Framework with Spike Camera (https://arxiv.org/abs/2501.04477)
Comments:
          Accepted by AAAI2025

- **What's New**: 본 논문에서는 새로운 스파이크 이미지 재구성 프레임워크인 SpikeCLIP을 소개합니다. 기존의 학습 기반 방법들은 인공 데이터셋에 의존하였으나, SpikeCLIP은 텍스트 설명과 비연결 고품질 데이터셋을 이용하여 네트워크 훈련을 시행합니다. 이를 통해 스파이크 카메라의 낮은 조명 환경에서도 생생한 텍스처와 밝기 균형을 가진 이미지를 복원할 수 있습니다.

- **Technical Details**: SpikeCLIP은 세 가지 단계로 구성됩니다: (1) Coarse Reconstruction에서는 경량 네트워크를 통해 스파이크 입력을 초기 재구성 출력으로 매핑합니다. (2) Prompt Learning 단계에서는 학습 가능한 프롬프트를 최적화하여 고품질 및 저품질 이미지의 분포를 캡처합니다. (3) Refinement 단계에서는 클래스 레이블의 특성과 고품질 이미지의 학습 프롬프트를 활용하여 네트워크 복원 성능을 더욱 향상시킵니다.

- **Performance Highlights**: U-CALTECH와 U-CIFAR 데이터셋을 사용한 실험 결과, SpikeCLIP은 저조도 환경에서 뛰어난 복원 성능을 보였습니다. 제안된 방법은 이전 방법들보다 고해상도 텍스처 특징 회복에 유리함을 입증하였으며, 다양한 하위 작업에 대한 더 튼튼하고 다재다능한 성능을 확보했습니다.



### A Histologic Dataset of Normal and Atypical Mitotic Figures on Human Breast Cancer (AMi-Br) (https://arxiv.org/abs/2501.04467)
- **What's New**: 이 연구에서는 유방암의 비정형 유사체(mitotic figures, MFs)와 정상 유사체를 분리할 수 있는 최초의 공공 데이터셋인 AMi-Br을 소개합니다. 이 데이터셋은 TUPAC과 MIDOG 2021의 두 개의 주요 MFs 데이터셋을 활용하여, 전문가 간 다수결 투표를 통해 만들어졌습니다. 총 3720개의 MFs가 포함되어 있으며, 이 중 832개(22.4%)가 비정형으로 분류되었습니다.

- **Technical Details**: AMi-Br 데이터셋은 223개 종양 사례에서 수집된 3720개의 MFs로 구성되어 있습니다. 데이터셋의 품질 검증을 위해 DenseNet-121과 EfficientNet V2 S를 이용한 기본 성능 분류 실험이 수행되었습니다. 클래스 불균형을 해결하기 위한 여러 전략들이 적용되었으며, 패치 수준과 환자 수준에서 데이터를 분할하여 실험을 진행했습니다.

- **Performance Highlights**: 패치 수준으로 나누어진 데이터 셋에서 평균 균형 정확도는 최대 0.806에 도달하고, ROC AUC는 최대 0.899에 도달했습니다. 그러나 환자 수준으로 나누었을 때 성능이 크게 떨어졌으며, 이는 모델이 특정 환자에 대해서 과적합(overfitting) 될 가능성을 시사합니다. 본 연구의 결과는 AMFs 분석을 위한 공개 데이터셋의 필요성을 강조합니다.



### A novel Facial Recognition technique with Focusing on Masked Faces (https://arxiv.org/abs/2501.04444)
- **What's New**: 이 연구는 마스크 착용 유무에 관계없이 동일한 얼굴을 인식하는 기능이 보안 및 공공 안전 분야에서 얼마나 중요한지를 강조합니다. 전통적인 얼굴 인식 시스템은 마스크로 얼굴이 가려진 경우에 심각한 정확도 문제를 겪으므로, 마스크 착용 상황에서도 신뢰할 수 있는 인식 방법을 개발하는 것이 필요합니다. 이를 위해 Masked-Unmasked Face Matching Model (MUFM)을 제안하며, 이는 새로운 접근 방식을 제공합니다.

- **Technical Details**: MUFM 모델은 Visual Geometry Group (VGG16) 모델을 활용하여 중요한 얼굴 특징을 추출하고, K-Nearest Neighbors (K-NN) 알고리즘을 이용하여 이러한 특징을 분류합니다. 또한 cosine similarity 메트릭을 사용하여 동일한 개인의 마스크 착용과 비착용 얼굴을 비교합니다. 이러한 기술적 접근은 마스크 착용 유무에 따른 동일 개인 인식이라는 과제를 해결하는데 기여합니다.

- **Performance Highlights**: 이 연구는 마스크에도 불구하고 개인을 효과적으로 식별할 수 있는 능력을 입증하며, 이는 기존 시스템의 주요 한계를 극복합니다. 연구에 사용된 이미지 데이터셋은 세 가지 서로 다른 출처에서 수집되었으며, 이는 연구의 신뢰성을 높이는 데 기여합니다. 이러한 데이터는 마스크 착용 및 비착용 상태의 동일한 얼굴을 포함하고 있어, 연구의 광범위한 가능성을 제시합니다.



### RSAR: Restricted State Angle Resolver and Rotated SAR Benchmark (https://arxiv.org/abs/2501.04440)
- **What's New**: 이 연구에서는 주로 Synthetic Aperture Radar (SAR) 분야에서 회전 물체 감지의 한계를 극복하기 위한 새로운 접근법을 제안합니다. 현재의 약한 감독(weakly supervised) 모델이 검출 객체의 각도를 정확히 예측하지 못하는 문제를 해결하기 위해 유니트 사이클 리졸버(Unit Cycle Resolver, UCR)를 개발하였습니다. UCR은 유니트 서클 제약 손실(unit circle constraint loss)을 도입하여 인코딩 상태가 유니트 서클 조건을 준수하도록 보장합니다.

- **Technical Details**: 기존의 각도 리졸버는 각도 인코딩 상태가 유니트 사이클 조건에 따라 달라지더라도 이를 무시하여 편향된 예측 결과를 초래합니다. UCR은 이러한 문제를 해결하기 위해 각도 예측을 개선하고, 결과적으로 대규모 회전 SAR 객체 감지 데이터 세트인 RSAR을 구축하여 제공하였습니다. RSAR 데이터 세트는 95,842개의 SAR 이미지와 183,534개의 주석된 인스턴스를 포함하고 있습니다.

- **Performance Highlights**: UCR을 적용한 모델은 RSAR과 DOTA-v1.0 데이터 세트에서 이전의 최첨단 방법들보다 향상된 성능을 보였습니다. 특히, UCR을 통해 기존 약한 감독 방법의 성능을 개선하고, 심지어 완전히 감독된 방법들과 동등한 성능을 달성하였습니다. 이 연구는 회전 물체 감지의 정확성을 크게 향상시키는 결과를 보여주며, 중요한 기여를 하고 있습니다.



### iFADIT: Invertible Face Anonymization via Disentangled Identity Transform (https://arxiv.org/abs/2501.04390)
- **What's New**: 이 논문에서는 얼굴 익명화(Face anonymization)에 대한 새로운 프레임워크인 iFADIT(Invertible Face Anonymization via Disentangled Identity Transform)를 제안합니다. 기존의 전통적인 접근법이 저하된 이미지 품질과 재구성 공격에 취약한 반면, 제안된 방법은 분리된 정체성 정보와 비정체적 속성을 결합하여 자연스러운 얼굴 세부 정보를 유지하는 동시에 안전한 방식으로 이를 익명화할 수 있습니다. 이 프레임워크는 주의 깊게 설계된 훈련 전략과 함께, 비밀키를 활용해 원본 얼굴 이미지의 복구도 가능하도록 합니다.

- **Technical Details**: iFADIT는 얼굴 이미지의 정체성과 비정체 속성을 분리하는 이중 구조(disentanglement architecture)와 사전 훈련된 생성 모델인 StyleGAN을 결합합니다. 이 과정을 통해 얼굴의 정체성이 안전하고 되돌릴 수 있는 방식으로 변환되어 익명화됩니다. 스타일GAN을 통해 생성되는 고품질의 얼굴 이미지는 인간의 얼굴에 대한 방대한 데이터에서 학습된 지식을 활용하여 구축됩니다. 또한, 양자 훈련 전략을 통해 얼굴 익명화와 복구의 특성이 최적화됩니다.

- **Performance Highlights**: 제안된 방법의 효과는 다양한 얼굴 이미지 데이터셋을 통해 정성적 및 정량적으로 평가되었습니다. 이를 통해 익명성(anonymity), 복원 가능성(reversibility), 보안(security), 다양성(diversity), 해석 가능성(interpretability) 측면에서 기존 방법 대비 우수함을 입증하였습니다. 특히, 원본 얼굴과 익명화된 이미지 간의 유사성을 높이고, 다양한 해석 가능성을 제공하여 실제 적용 가능성을 크게 향상시켰습니다.



### Exploring Unbiased Deepfake Detection via Token-Level Shuffling and Mixing (https://arxiv.org/abs/2501.04376)
- **What's New**: 이 논문에서는 기존의 딥페이크 탐지에서 일반화 문제가 특정 위조 방법 간의 차이에 의해 발생한다고 여겨져온 기존 연구와는 달리, 위조와 무관한 요인들이 변화할 때도 일반화 문제가 발생할 수 있음을 밝혔다. 저자들은 위치 편향(position bias)과 내용 편향(content bias)이라는 두 가지 주요 편향을 발견하였으며, 이는 탐지기가 특정 이미지 위치나 무관한 정보를 통해 잘못 학습되는 경향을 보임을 의미한다.

- **Technical Details**: 이를 해결하기 위해 저자들은 Transformers의 잠재 공간(latent space) 내에서 토큰을 셔플(shuffling)하고 혼합(mixing)하는 두 가지 가지(branch)를 구현하였다. 셔플 가지는 각 이미지의 토큰과 해당 위치 임베딩을 재배열하여 로컬 상관성을 유지하며, 혼합 가지는 미니 배치 내에서 동일한 레이블을 가지는 두 이미지 사이에서 랜덤으로 토큰을 선택하고 혼합한다. 이 과정에서 대조 손실(contrastive losses)과 발산 손실(divergence losses)을 적용하여 편향 없는 특징 표현을 얻고, 분류기를 개선하는 전략을 사용한다.

- **Performance Highlights**: 광범위한 평가 데이터세트에서 실험한 결과, 제안된 방법이 기존의 최첨단 방법들보다 더 우수한 성능을 보임을 입증하였다. 특히, 이전 데이터세트에서는 탐지기의 일반화 문제가 두드러지게 나타나는 반면, 이 새로운 접근 방식은 이러한 문제를 효과적으로 해결함으로써 더욱 견고한 딥페이크 탐지기를 구현하게 되었다.



### Instructive3D: Editing Large Reconstruction Models with Text Instructions (https://arxiv.org/abs/2501.04374)
Comments:
          Accepted at WACV 2025. First two authors contributed equally

- **What's New**: 최근 제안된 대규모 재구성 모델(Large Reconstruction Models, LRM)을 통해 단일 물체 이미지를 활용하여 고품질 3D 모델을 생성할 수 있는 능력이 제공됩니다. 그러나 기존 모델들은 표준 디자인 패턴 추가나 색상 및 반사율 변화를 포함한 세부 조정을 수행할 수 있는 제한이 있어, 이는 증강 현실 및 애니메이션 같은 분야에서 유용합니다. 본 논문에서는 이러한 제한을 극복하기 위해 Instructive3D라는 새로운 LRM 기반 모델을 제안합니다.

- **Technical Details**: Instructive3D는 사용자 텍스트 프롬프트를 통해 3D 객체의 생성 및 세밀한 수정을 통합하는 방식으로 작동합니다. 이를 위해 3D 객체 모델의 트리플레인(latent space) 표현에서 텍스트 프롬프트에 조건화된 확산 과정(diffusion process)을 수행하는 어댑터를 추가합니다. 이 과정에서 편집된 3D 객체를 생성할 필요 없이 기하학적으로 일관된 수정을 가능하게 하여 3D 객체의 유연성과 정밀성을 강화합니다.

- **Performance Highlights**: Instructive3D는 표준 LRM 모델을 사용하여 먼저 3D 객체 메시를 생성한 후, 주어진 이미지에 대한 텍스트 프롬프트를 통해 이 객체들을 수정하는 기존 기준 모델에 비해 질적으로 우수한 3D 객체를 생성합니다. 본 논문에서 제안하는 Instructive3D는 기존 모델들에 비해 더 높은 품질의 3D 객체 생성을 가능하게 하며, 다양한 객체 데이터셋에서 확인된 이러한 성능은 사용자 정의 텍스트 프롬프트에 의해 이루어진 편집 사항들을 반영할 수 있음을 보여줍니다.



### FGU3R: Fine-Grained Fusion via Unified 3D Representation for Multimodal 3D Object Detection (https://arxiv.org/abs/2501.04373)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 이번 논문에서는 자율주행에서의 다중 모드 3D 물체 탐지 문제를 다루며, FGU3R이라는 새로운 프레임워크를 제안합니다. 기존의 LiDAR와 2D 이미지 간의 비효율적인 융합 문제를 해결하기 위해, 문서에서는 통합된 3D 표현을 사용한 세분화된(fine-grained) 융합 방법을 도입합니다. 핵심 구성 요소로는 Pseudo-Raw Convolution (PRConv)과 Cross-Attention Adaptive Fusion (CAAF)이 있으며, 이들이 각각 모드 간의 상호작용을 극대화하고 3D RoI 특징을 적응적으로 융합하는 데 기여합니다.

- **Technical Details**: FGU3R의 주요 구성은 raw points와 이미지를 통해 생성된 pseudo points를 바탕으로 합니다. 이 과정에서 depth 보완(depth completion) 네트워크를 활용하여 보다 신뢰할 수 있는 3D pseudo point clouds를 생성합니다. 또한, PRConv는 다양한 모드의 피쳐를 추출하고 상호작용을 가능하게 하며, CAAF는 서로 다른 특성의 융합을 통해 최종 회귀 작업을 수행합니다.

- **Performance Highlights**: KITTI 및 nuScenes 데이터셋에서 수행된 실험 결과, 제안된 FGU3R 방법이 기존의 방법들보다 더 효율적으로 작동하며, 3D 물체 탐지 성능에서 현저한 개선을 보였음을 입증하였습니다. 실험은 gross-grained fusion 대신 fine-grained fusion이 탐지 성능을 향상시킬 수 있음을 보여줍니다. 즉, 제안된 방법은 자율주행 시스템에 있어 더 높은 정확도를 제공할 잠재력을 가지고 있습니다.



### DeFusion: An Effective Decoupling Fusion Network for Multi-Modal Pregnancy Prediction (https://arxiv.org/abs/2501.04353)
- **What's New**: 이번 연구에서는 체외 수정 배아 이식 (IVF-ET)에서 임신 예측을 위한 새로운 모델인 DeFusion을 제안합니다. 이 모델은 다양한 발생단계의 배아 이미지와 부모의 생식지표를 통합하여 예측 정확도를 향상시키는 데 초점을 맞추고 있습니다. 기존 연구와는 달리, DeFusion은 첫 3일 간의 배아 이미지와 피험자 데이터를 효율적으로 결합하여 최적의 결과를 도출합니다.

- **Technical Details**: DeFusion은 두 가지 주요 모듈로 구성됩니다: 첫째, 다양한 발생일수의 배아 이미지를 활용하기 위해 시간-공간 포지션 인코딩 (spatial-temporal position encoding)을 사용하며, 둘째, 테이블 트랜스포머 (table transformer)로 생식지표 데이터를 추출합니다. 이 두 가지 구성 요소는 각 개별 데이터를 관련 및 비관련 정보로 분리하여 더 정교한 융합 (fusion)을 가능하게 합니다.

- **Performance Highlights**: 4046개의 데이터를 포함하는 새로운 데이터셋을 사용하여 평가한 결과, DeFusion은 최신의 다른 방법들보다 뛰어난 성능을 보였습니다. 또한, 안과 질병 예측 데이터셋에서도 잘 일반화되는 성능을 입증하여, 본 모델의 폭넓은 적용 가능성을 나타냈습니다.



### Online Gaussian Test-Time Adaptation of Vision-Language Models (https://arxiv.org/abs/2501.04352)
- **What's New**: 최근 온라인 테스트 시 적응(OTTA) 기술이 시각-언어 모델(VLM) 분야에서 주목받고 있습니다. 기존 방법들은 데이터셋 특화된 하이퍼파라미터에 의존하여 새로운 작업에 대한 적응력이 제한됩니다. 이를 해결하기 위해 새로운 방법인 Online Gaussian Adaptation (OGA)을 제안하며, 이는 다변량 가우시안 분포를 활용해 시각적 특징의 가능성을 모델링하고, 고정된 하이퍼파라미터로 해석 가능한 최대 사후 확률(MAP) 추정 규칙을 제공합니다.

- **Technical Details**: OGA는 관찰된 시각 특징의 가능성을 모델링하기 위해 다변량 가우시안 분포를 사용하고 이를 제로샷(Zero-shot) 사전과 결합하여 고정된 하이퍼파라미터로 작동합니다. 이 방식은 예측 규칙을 단순하고 해석 가능하게 만들며, 기존 방식들보다 더 나은 성능을 제공합니다. 또한, OGA는 블랙박스(black-box) 프레임워크에 적합하며, 계산적으로도 효율적입니다.

- **Performance Highlights**: OGA는 다양한 데이터셋과 여러 실행에서 최신 기술보다 뛰어난 성능을 보여줍니다. 일반적인 OTTA 평가 프로토콜이 성능의 변동성을 간과하고 있다는 점을 언급하며, 더 많은 실행을 통해 평균 정확도를 측정할 것을 권장합니다. 또한 예상 꼬리 정확도(Expected Tail Accuracy, ETA)를 도입하여 최악의 경우 시나리오에서의 성능을 평가하는 방식을 제안합니다.



### Building a Mind Palace: Structuring Environment-Grounded Semantic Graphs for Effective Long Video Analysis with LLMs (https://arxiv.org/abs/2501.04336)
- **What's New**: 이번 연구에서는 비디오 이해를 위한 새로운 프레임워크인 VideoMindPalace를 소개합니다. 이는 'Mind Palace' 개념에서 영감을 받아, 비디오의 중요한 순간을 계층적으로 구조화된 의미 그래프로 조직합니다. 이 시스템은 사람-객체 추적, 클러스터링된 활동 영역, 환경 레이아웃 맵핑을 통해 정보를 구성하여, 자연어 이해를 위한 LLM의 활용을 가능하게 합니다.

- **Technical Details**: VideoMindPalace는 (i) 인간-객체 추적을 통해 인물과 객체 간의 상호작용을 포착하고, (ii) 다양한 시간적 배경에서의 특정 활동 영역을 클러스터링하며, (iii) 활동 영역의 위상적 배열을 반영하는 환경 레이아웃으로 구성됩니다. 이러한 구성 요소들은 인간 활동을 모델링하고 매핑하는 시각적 인식 모델을 활용하여, 비디오 분석에 필요한 정보를 포착합니다.

- **Performance Highlights**: 비디오 MindPalace는 VMB(Benchmark) 및 EgoSchema, NExT-QA, IntentQA와 같은 기존 비디오 QA 데이터셋에서 평가되었습니다. 이 시스템은 시공간(Spatial and Temporal) 일관성과 인간 중심(reasoning) 추론에서 현저한 향상을 보여주며, VLM에서의 장기 비디오 분석 능력을 발전시키고 있습니다.



### An Efficient Adaptive Compression Method for Human Perception and Machine Vision Tasks (https://arxiv.org/abs/2501.04329)
- **What's New**: 이번 연구에서는 Efficient Adaptive Compression (EAC) 방식을 소개하며, 이는 인간 시각과 머신 비전 작업을 모두 고려한 효율적인 압축 방법입니다. EAC는 두 가지 주요 모듈, 즉 적응형 압축 메커니즘과 작업별 어댑터(task-specific adapter)를 통해 구성됩니다. 이 방법은 다양한 머신 비전 작업에 대한 최적화를 균형 있게 수행할 수 있는 기능을 가지고 있습니다.

- **Technical Details**: EAC의 적응형 압축 모듈은 잠재 특징(latent features)에서 여러 하위 집합을 선택하여 다양한 머신 비전 작업을 처리하는 능력을 갖추고 있습니다. 또한, 작업별 어댑터는 parameter-efficient delta-tuning 전략을 통해 파라미터 수를 최소화하며, 특정 머신 비전 작업을 위한 하위 네트워크를 자극하는 역할을 합니다. 이 방식을 통해 비트레이트 비용을 최적화하고 머신 비전 성능을 향상시킬 수 있습니다.

- **Performance Highlights**: EAC는 다양한 벤치마크 데이터셋에서 광범위한 평가를 수행하였으며, 여러 머신 비전 작업에서 성능이 향상되었음이 입증되었습니다. 특히, 기존의 NIC 및 NVC 방법과 원활하게 통합될 수 있으며, 인간 시각에 대한 품질을 유지하면서도 여러 머신 비전 작업에 대해 유의미한 비트레이트 절감을 제공합니다. 결과적으로 EAC는 최신 표준 VTM보다 대부분의 비트레이트에서 뛰어난 성능을 보여주었습니다.



### Edit as You See: Image-guided Video Editing via Masked Motion Modeling (https://arxiv.org/abs/2501.04325)
- **What's New**: 본 논문에서는 기존의 텍스트 기반 비디오 편집 방법의 한계를 보완하기 위해 이미지 기반 비디오 편집을 위한 새로운 확산 모델인 IVEDiff를 제안합니다. IVEDiff는 사용자가 첫 번째 프레임에서 편집하고자 하는 객체를 지정하고 RGB 이미지를 참조 이미지로 제공함으로써 비디오를 편집할 수 있는 혁신적인 방법입니다. 이 모델은 기존의 이미지 편집 모델을 기반으로 하며, 시간 일관성을 유지할 수 있도록 학습 가능한 모션 모듈이 장착되어 있습니다.

- **Technical Details**: IVEDiff는 이미지-가이드 이미지 편집 확산 모델인 MimicBrush를 기반으로 하며, 각 층에 모션 레퍼런스 네트워크인 MotRefNet을 통합하여 이웃이 편집된 프레임들 간의 외관 일관성을 유지합니다. 또한, IVEDiff는 마스킹된 모션 모델링(Masked Motion Modeling, MMM) 파인튜닝 전략을 통해, 연속된 프레임에서의 모션 동적을 보다 잘 포착할 수 있도록 합니다. 이 방법은 불필요한 정보로 인한 잘못된 상관관계 모델링을 완화하여 일관된 비디오 결과를 생성하는 데 기여합니다.

- **Performance Highlights**: 종합적인 실험 결과, IVEDiff는 다양한 편집 객체에 대해 높은 품질을 유지하면서도 시간적으로 매끄러운 편집된 비디오를 생성하는 것으로 나타났습니다. 제한된 리소스 내에서 효율적으로 비디오 편집을 수행할 수 있으며, 성능 평가를 위한 벤치마크도 구성하여 다양한 방법들의 시각적 일관성과 품질을 철저하게 평가할 수 있게 하였습니다. 이러한 성과는 이미지 기반 비디오 편집 분야의 연구를 위한 중요한 기초 자료가 될 것입니다.



### Eve: Efficient Multimodal Vision Language Models with Elastic Visual Experts (https://arxiv.org/abs/2501.04322)
- **What's New**: 이 논문에서 제안하는 새로운 프레임워크인 Efficient Vision Language Models with Elastic Visual Experts (Eve)은 모델의 크기를 줄이면서도 언어 및 멀티모달(multi-modal) 능력을 향상시키기 위해 세 가지 훈련 단계를 활용합니다. Eve는 다양한 훈련 단계에서 적응 가능한 비주얼 전문가(visual experts)를 통합하여 언어 능력을 보존하고 멀티모달 성능을 향상시키는 균형 잡힌 접근방식을 보여줍니다. 이 접근법은 1.8B 파라미터의 모델에서도 뛰어난 성능을 유지하며, 3B 이하의 파라미터에서 언어 벤치마크에서 우위를 점합니다.

- **Technical Details**: Eve 프레임워크는 세 단계의 훈련 프로세스를 통해 비주얼 입력을 처리하는 elastic vision experts를 도입하고, 초기 두 단계에서는 잘 사전 훈련된 비전 인코더를 활용합니다. 세 번째 단계에서는 LLM 트랜스포머와 통합된 elastic vision feed forward network(FFN)를 도입하여, 각 전문가가 특정 비주얼 작업에 집중할 수 있도록 합니다. 이러한 구조적 통합은 언어 모델의 성능을 유지하면서 멀티모달 능력을 크게 향상시키는 데 기여합니다.

- **Performance Highlights**: Eve는 3B 파라미터 이하의 모델 크기에서도 VLM 및 언어 벤치마크에서 최상의 성능을 달성하며, 동시에 7B LLaVA-1.5 모델보다 높은 멀티모달 정확도를 기록합니다. 특히, Eve는 기존의 다른 멀티모달 모델에 비해 더욱 뛰어난 언어 능력을 보존하며, 구조와 훈련 방식에서 혁신적인 접근을 통해 빠른 훈련을 가능하게 했습니다. 이 모델은 다양한 비주얼 및 텍스트 입력을 처리하는 데 매우 유용할 것으로 기대됩니다.



### DGQ: Distribution-Aware Group Quantization for Text-to-Image Diffusion Models (https://arxiv.org/abs/2501.04304)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 텍스트-이미지 확산 모델의 양자화(quantization)에서 발생하는 어려움을 분석하고, 이미지 품질과 텍스트-이미지 정합성(text-image alignment)을 동시에 유지하는 새로운 접근법인 Distribution-aware Group Quantization (DGQ)을 제안합니다. 이 방법은 픽셀별 및 채널별 아웃라이어(outlier)를 식별하여 처리하는데 초점을 맞추어 이미지 품질을 보존합니다. 그뿐만 아니라, 프롬프트에 따라 로그 양자화(logarithmic quantization) 스케일을 적용하여 텍스트와 이미지 간의 정합성을 유지합니다.

- **Technical Details**: 제안된 DGQ 방법은 두 가지 주요 도전 과제를 다룹니다. 첫 번째는 아웃라이어가 포함된 활성화(activation)로, 이는 이미지 품질을 보존하는 데 중요한 역할을 합니다. 두 번째는 크로스-어텐션(cross-attention)에서의 독특한 패턴으로, 이는 텍스트-이미지 정합성에 큰 영향을 미칩니다. DGQ는 아웃라이어 보존 그룹 양자화를 통해 이러한 아웃라이어를 픽셀 및 채널 단위로 그룹화하고, 입력 프롬프트에 따라 각기 다른 양자화 스케일을 적용합니다.

- **Performance Highlights**: DGQ 방법은 MS-COCO 및 PartiPrompts와 같은 데이터셋에서 우수한 성능을 보였습니다. MS-COCO 데이터셋에서는 FID 점수를 13.15로 줄였으며, 이는 풀 정밀도(full precision)보다도 낮은 점수입니다. 또한, DGQ는 양자화 후 93.7%의 비트 연산 절약(비트 작업 수 694 TBOPs에서 43.4 TBOPs로)을 달성하였습니다. 본 연구는 추가 파인튜닝 없이 8비트 이하의 텍스트-이미지 확산 모델에서 저비트 양자화를 최초로 성공적으로 달성했습니다.



### H-MBA: Hierarchical MamBa Adaptation for Multi-Modal Video Understanding in Autonomous Driving (https://arxiv.org/abs/2501.04302)
Comments:
          7 pages, 4 figures

- **What's New**: 이 논문에서는 자율주행에 대한 새로운 기회와 도전을 제시합니다. 특히, 다중 모달 비디오 이해가 자율주행 과정에서 발생할 사건을 분석하는 데 필수적임을 강조합니다. 기존 Multimodal Large Language Models(MLLMs)의 일반화 능력을 제한하는 복잡한 시공간 동작을 고려하여, 복잡한 움직임 변화를 수용하기 위한 새로운 Hierarchical Mamba Adaptation(H-MBA) 프레임워크를 제안합니다.

- **Technical Details**: H-MBA는 Context Mamba(C-Mamba)와 Query Mamba(Q-Mamba)라는 두 가지 모듈로 구성되어 있습니다. C-Mamba는 다양한 구조 상태 공간 모델을 포함하여 서로 다른 시간 해상도를 위한 비디오 맥락을 효과적으로 캡처합니다. Q-Mamba는 현재 프레임을 학습 가능한 쿼리로 변환하고, 다중 해상도 비디오 맥락을 선택적으로 조합하여 비디오 이해를 향상시킵니다.

- **Performance Highlights**: 이 방법을 통해 자율주행에서의 다중 모달 비디오 작업을 위한 성능이 크게 향상됩니다. 예를 들어, 위험 객체 탐지에서 기존 성능 주도 방법(SOTA)에 비해 5.5% mIoU(improved Intersection over Union) 향상을 달성했습니다. 모델 추론 과정에서는 초당 약 0.2초의 처리 시간을 기록하며, 이는 실제 응용 프로그램에서의 효용성을 강조합니다.



### TADFormer : Task-Adaptive Dynamic Transformer for Efficient Multi-Task Learning (https://arxiv.org/abs/2501.04293)
- **What's New**: 본 논문은 Task-Adaptive Dynamic transFormer(TADFormer)라는 새로운 PEFT 프레임워크를 소개합니다. 이 프레임워크는 멀티 태스크 학습(Multi-Task Learning, MTL) 설정에서 태스크에 민감한 특징을 세밀하게 적응하도록 설계되었습니다. TADFormer는 입력 컨텍스트에 따라 동적으로 작동하는 Task Filter(Dynamic Task Filter, DTF)를 사용하여 작업별 정보를 캡쳐하는 방법을 제안합니다.

- **Technical Details**: TADFormer는 태스크 응답 맵(task attention maps)을 통해 각 태스크에 대한 세밀한 태스크 적응 특징을 생성하며, DTF는 동적 합성곱(dynamic convolution) 작업을 활용하여 입력 샘플의 컨텍스트 정보를 이용합니다. 이 모듈은 여러 태스크 간의 상호작용을 고려하여 특징을 추출하며, 이는 MTL 성능 향상에 필수적입니다.

- **Performance Highlights**: TADFormer는 PASCAL-Context 벤치마크에서 밀집 장면 이해(dense scene understanding) 작업들을 수행할 때, 전체 모델의 완전한 튜닝(full fine-tuning)에 비해 최대 8.4배 적은 학습 가능 매개변수(trainable parameters)로 더 높은 정확도를 달성했습니다. 또한, 최근의 PEFT 방법들과 비교해도 우수한 매개변수 효율성과 정확도를 보여줍니다.



### ContextMRI: Enhancing Compressed Sensing MRI through Metadata Conditioning (https://arxiv.org/abs/2501.04284)
Comments:
          29 pages, 9 figures

- **What's New**: 본 연구에서는 ContextMRI라는 텍스트 조건화 확산 모델을 제안하여 MRI 재구성 과정에 임상 메타데이터를 통합했습니다. 이 모델은 기존의 MRI 재구성 방식이 간과했던 환자 인구 통계, 영상 매개변수와 같은 중요한 정보를 활용하여 더 정확한 이미지를 생성할 수 있도록 설계되었습니다. 이러한 접근은 CS-MRI에서 재구성 성능을 획기적으로 향상시키는 것으로 나타났습니다.

- **Technical Details**: ContextMRI는 최소한의 처리된 복소수 MRI 이미지를 기반으로 훈련된 픽셀-공간 확산 모델입니다. 훈련 과정에서 CLIP 텍스트 임베딩을 활용하여 메타데이터를 구조화된 텍스트 프롬프트로 변환하고 모델에 공급합니다. 이 과정은 재구성에 필요한 사전 정보를 조건화함으로써 이루어집니다.

- **Performance Highlights**: 실험 결과, 다양한 데이터 세트 및 샘플링 패턴에 걸쳐 제안된 모델이 일관되게 높은 재구성 성능을 보였습니다. Meta데이터의 충실도가 증가할수록 재구성 품질이 체계적으로 향상되는 경향이 있는 것으로 나타났습니다. ContextMRI는 무조건적인 접근 방식과 비교할 때 모든 상황에서 두드러진 성능 향상을 보여주었습니다.



### Enhancing Scene Classification in Cloudy Image Scenarios: A Collaborative Transfer Method with Information Regulation Mechanism using Optical Cloud-Covered and SAR Remote Sensing Images (https://arxiv.org/abs/2501.04283)
- **What's New**: 이번 연구는 클라우드 오염으로 인한 광학 정보 손실에 대응하기 위해, 클라우드가 있는 광학 데이터와 Synthetic Aperture Radar (SAR) 데이터를 활용하여 장면 분류를 수행하는 새로운 방법을 제안합니다. 새로운 방식인 협동 이식 전략(collaborative transfer strategy)을 통해 서로 다른 데이터 간의 지식을 효과적으로 전달할 수 있으며, 정보 조절 메커니즘(information regulation mechanism)을 통해 전송 과정에서 발생할 수 있는 양식 불균형 문제(modality imbalance issue)를 해결하려고 합니다. 이 연구는 주어진 최첨단 방식이 클라우드 오염이 있는 상황에서 다른 솔루션들보다 뛰어난 성능을 보여줌을 확인했습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 요소로 구성됩니다: 첫째, 지식 증류(knowledge distillation)를 기반으로 한 협동 이식 전략은 이질적인 데이터 간에 효율적으로 사전 지식을 전달할 수 있게 합니다. 둘째, 정보 조절 메커니즘(IRM)은 전이 과정에서 발생할 수 있는 양식 불균형 문제를 해결하기 위해 보조 모델을 활용하여 각 양식의 기여 비율을 측정하고, 샘플 단위에서 양식 간 정보 활용의 균형을 자동으로 맞춥니다. 이를 통해 클라우드가 있는 광학 데이터와 SAR 데이터의 통합된 특성을 효과적으로 활용하여 장면 분류 작업을 진행합니다.

- **Performance Highlights**: 제안된 방법은 시뮬레이션된 클라우드 데이터셋과 실제 클라우드 데이터셋에서 전이 실험을 수행한 결과, 클라우드가 덮인 상황에서 다른 기존의 솔루션에 비해 우수한 성능을 입증하였습니다. 또한, IRM의 중요성과 한계를 검증하고 모델 전이 과정에서 발생하는 양식 불균형 문제를 논의하고 시각화했습니다. 이러한 결과는 클라우드 오염이 많은 지역에서 장면 분류 작업의 효율성을 높이는 데 기여할 것으로 기대됩니다.



### Open set label noise learning with robust sample selection and margin-guided modu (https://arxiv.org/abs/2501.04269)
- **What's New**: 최근 깊은 신경망(Deep Neural Network, DNN)의 성공은 대규모, 고품질의 라벨링된 데이터셋 덕분입니다. 그러나 실제 데이터셋에서 라벨 노이즈가 포함된 경우 과적합(overfitting)이 발생할 수 있습니다. 본 논문에서는 라벨 노이즈 문제를 해결하기 위해 Robust Sample Selection and Margin-Guided Module (RSS-MGM)이라는 새로운 방법론을 제안합니다. 이는 특히 개방 세트 오픈셋 라벨 노이즈를 처리하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 RSS-MGM 방법은 두 가지 주요 모듈로 구성됩니다. 첫째, 강인한 샘플 선택 모듈은 작고 손실이 적은 샘플이나 높은 신뢰성을 가진 샘플을 결합하여 더 많은 클린 샘플을 선택합니다. 둘째, 마진 가이디드 모듈은 ID(인-디스트리뷰션)와 OOD(아웃-오브-디스트리뷰션) 샘플을 구분하는 마진 함수를 도입하여 효과적으로 라벨 노이즈를 필터링합니다. 이를 통해 트레이닝 데이터셋을 클린 세트, 클로즈드 세트, 오픈 세트로 나누어 처리합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 수행된 실험 결과에 따르면, RSS-MGM 방법은 기존의 라벨 노이즈 학습 방법들보다 우수한 성능을 보였습니다. 특히 오픈 세트와 클로즈드 세트 샘플을 정확하게 분류하는 데 효과적임을 입증하였습니다. CIFAR-100N-C, CIFAR80N-O, WebFG-469, Food101N와 같은 실제 데이터셋에서도 뛰어난 성능을 보이면서 모델의 신뢰성과 일반화 능력을 향상시켰습니다.



### Continual Self-supervised Learning Considering Medical Domain Knowledge in Chest CT Images (https://arxiv.org/abs/2501.04217)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 이 연구에서는 흉부 CT 이미지 분석을 위한 새로운 지속적 자기 지도 학습 방법(CSSL)을 제안합니다. 이 방법은 다양한 학습 단계에서 이전에 학습한 지식과 새로운 정보를 효율적으로 연관시켜, 데이터 간의 간섭 문제를 해결하는 데 중점을 두고 있습니다. 특히, 강화된 Dark Experience Replay(DER)를 도입하여, 재허가 버퍼의 다양성과 대표성을 유지하면서, 모델의 특징 표현 학습을 극대화합니다.

- **Technical Details**: 제안된 방법은 세 단계로 구성된 CSSL 접근 방식을 활용하여, 두 개의 CT 이미지 영역 간의 데이터 간섭 현상을 줄이고, 재학습 시의 파국적 망각을 방지합니다. 첫 번째 단계에서는 한 영역 내에서 자기 지도 학습을 실행하여 특징 표현을 학습하며, 두 번째 단계에서는 이전 단계에서 선택된 이미지를 버퍼에 저장하여 다양성과 대표성을 유지합니다. 마지막 단계에서는 또 다른 도메인에 대한 지속적 자기 지도 학습을 적용하며, 이를 통해 더 풍부한 특징 표현을 학습합니다.

- **Performance Highlights**: 실험 결과, 제안된 CSSL 방법이 최신 기술을 초월하는 성능을 보여주었습니다. 두 가지 다른 이미징 조건에서 얻은 흉부 CT 이미지를 사용하여 사전 학습을 진행하고, 공개 CT 이미지 데이터셋에 대한 평가를 통해 높은 정확도를 기록했습니다. 이 방법은 다양한 도메인 간의 데이터 분포 차이를 고려하여 더욱 강력한 특징 표현을 학습할 수 있도록 돕습니다.



### UPAQ: A Framework for Real-Time and Energy-Efficient 3D Object Detection in Autonomous Vehicles (https://arxiv.org/abs/2501.04213)
- **What's New**: 이 논문은 자율주행 차량(AV)에서 3D 객체 감지기의 효율성을 높이는 새로운 프레임워크인 UPAQ를 소개합니다. UPAQ는 반구조적 패턴 가지치기(semi-structured pattern pruning)와 양자화(quantization)를 활용하여 자원 제약이 있는 내장형 AV 플랫폼에서 LiDAR 포인트 클라우드와 카메라 기반 3D 객체 검출기의 효율성을 향상시킵니다.

- **Technical Details**: UPAQ는 최신 모델 압축 프레임워크에 비해 Pointpillar와 SMOKE 모델에서 각각 최대 5.62배 및 5.13배의 모델 압축률을 달성합니다. 또한, 추론 속도가 최대 1.97배 및 1.86배 향상되며, 에너지 소비는 최대 2.07배 및 1.87배 감소합니다. 이러한 결과는 Jetson Orin Nano 내장 플랫폼에서 실시된 실험을 기반으로 합니다.

- **Performance Highlights**: UPAQ 프레임워크는 기존의 2D 객체 감지기보다 더 포괄적인 예측을 제공하면서도, 자원 제한이 있는 상황에서 높은 성능을 발휘합니다. 성능 측면에서 더 빠른 추론 시간과 낮은 에너지 소비를 결합하여, 자율주행 기술의 발전을 이끌고 있습니다.



### Recognition-Oriented Low-Light Image Enhancement based on Global and Pixelwise Optimization (https://arxiv.org/abs/2501.04210)
Comments:
          accepted to VISAPP2025

- **What's New**: 이 논문에서는 인식 모델의 성능 향상을 목표로 하는 새로운 저조도 이미지 향상(low-light image enhancement) 방법을 제안합니다. 최근 딥러닝의 발전에도 불구하고, 저조도 환경에서의 이미지 인식은 여전히 어려운 문제로 남아 있습니다. 기존의 저조도 이미지 향상 방법들은 인간 시각을 위한 이미지 가시성 개선에 초점을 두고 있지만, 인식 모델 성능 향상에는 특별히 중점을 두지 않았습니다.

- **Technical Details**: 제안된 저조도 이미지 향상 방법은 두 가지 주요 모듈로 구성됩니다: 전체 밝기 및 색상 균형을 조정하는 Global Enhance Module과, 픽셀 단위에서 이미지 특성을 정제하는 Pixelwise Adjustment Module입니다. 이 모듈들은 입력 이미지를 향상시켜 다운스트림 인식 모델 성능을 효과적으로 향상시키도록 학습됩니다. 특히, 제안된 방법은 다운스트림 인식 모델을 재훈련하지 않고도 저조도 인식 성능을 개선하기 위한 전처리 필터로 적용될 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 저조도 환경에서 사전 훈련된 인식 모델의 성능을 향상시키는 데 효과적인 것으로 나타났습니다. 이 방법은 기존 인식 모델들의 성능을 저조도 조건에서도 높일 수 있는 가능성을 보여주며, 실제 응용에서 강력한 효과를 발휘합니다.



### LipGen: Viseme-Guided Lip Video Generation for Enhancing Visual Speech Recognition (https://arxiv.org/abs/2501.04204)
Comments:
          This paper has been accepted for presentation at ICASSP 2025

- **What's New**: 이번 논문에서는 LipGen이라는 새로운 프레임워크를 제안하여 비주얼 스피치 레코그니션(Visual Speech Recognition, VSR) 모델의 강인성을 개선하고자 합니다. 이 방법은 음성 기반으로 생성된 합성 비주얼 데이터(synthetic visual data)를 활용하여 기존 데이터셋의 제한을 극복하고, 동시적으로 viseme 분류(auxiliary task)를 도입하여 모델의 성능을 높이는 데 초점을 맞추고 있습니다. 결과적으로, 이 방법은 현재 최첨단 기술보다 우수한 성능을 보여줍니다.

- **Technical Details**: LipGen 프레임워크는 두 단계로 구성된 음성 기반의 립 애니메이션 모델을 통해 훈련 데이터셋을 증강하는 방식을採用합니다. 이 모델은 AniPortrait를 사용하여 오디오에 따라 3D 얼굴 메쉬를 생성하고, 이를 통해 자연스러운 얼굴 애니메이션을 생성합니다. 또한, viseme 분류 보조 작업을 제안하여 여러 입 모양을 구분하는데 도움을 주며, 이를 통해 훈련 데이터의 다양성을 높이고 있습니다.

- **Performance Highlights**: LipGen의 방식을 통해 생성된 합성 데이터는 다양한 환경과 연출에서의 입 모양 변화를 효과적으로 캡처하여 모델의 일반화 능력을 향상시킵니다. 실험 결과, LipGen은 lip reading in the wild (LRW) 데이터셋에서 현재 최첨단 기술보다 더욱 뛰어난 성능을 보이며, 어려운 조건에서도 뚜렷한 장점을 발휘합니다. 이러한 성과는 인식 능력을 대폭 개선하며, 비주얼 스피치 레코그니션의 미래 방향성을 제시합니다.



### Generative Dataset Distillation Based on Self-knowledge Distillation (https://arxiv.org/abs/2501.04202)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 새로운 연구에서는 dataset distillation의 효율성을 향상시키기 위해 self-knowledge distillation을 도입한 새로운 생성 기반 데이터셋 디스틸레이션 방법을 제안했습니다. 이 방법은 원본 데이터와 합성 데이터 간의 예측 로짓(logits) 정렬의 정확성을 향상시키고, 보다 효과적인 데이터 구조와 관계를 포착합니다. 특히, 정규화를 통해 로짓의 일관성을 유지함으로써 보다 정교한 분포 매칭(distribution matching)을 도모하고 있습니다.

- **Technical Details**: 본 연구의 방법론은 두 단계로 구성됩니다. 첫 번째로, 조건부 생성적 적대 신경망(GAN)을 학습하여 합성 데이터셋 S를 생성합니다. 이후 모델 풀(pool)에서 무작위로 선택된 모델을 통해 original 데이터셋 O와의 정렬을 수행하며, self-knowledge distillation을 통합하여 원본 및 합성 데이터의 분포를 효과적으로 일치시킵니다. 또한, 로짓의 범위를 일관되게 유지하기 위한 정규화 단계를 도입하여 정렬의 정확성을 높였습니다.

- **Performance Highlights**: 제안된 방법은 여러 벤치마크 데이터셋을 통해 기존의 최신 방법들보다 우수한 성능을 보여주었습니다. 특히, 생성자가 보다 정확하고 대표성 있는 합성 데이터를 생성하도록 돕고, 결과적으로 더 높은 품질의 데이터 디스틸레이션을 가능하게 했습니다. 이러한 접근 방식은 다양한 모델 아키텍처에서 통칭성과 성능을 개선하는데 기여하고 있습니다.



### MedicalNarratives: Connecting Medical Vision and Language with Localized Narratives (https://arxiv.org/abs/2501.04184)
- **What's New**: 이번 논문에서는 MedicalNarratives라는 데이터셋을 제안합니다. 이 데이터셋은 의료 교육 비디오에서 수집된 것으로, Think-Aloud 연구에서 수집된 데이터와 유사합니다. MedicalNarratives는 이미지-텍스트 쌍을 통해 의료 분야의 다양한 작업을 위한 사전 훈련을 가능하게 하여, 의료 관련 작업을 적절한 크기의 데이터셋 없이도 효율적으로 수행할 수 있도록 합니다.

- **Technical Details**: MedicalNarratives 데이터셋은 총 4.7M개의 이미지-텍스트 쌍을 포함하고 있으며, 1M 표본에는 밀집 주석(dense annotation)이 포함되어 있습니다. 이 데이터셋은 다양한 의료 도메인에서 수집된 데이터로 구성되어 있으며, 875K개의 표본은 마스크나 바운딩 박스로 변환이 가능합니다. 이를 통해 GenMedClip이라는 비전-언어 모델을 훈련시키고, 성능을 평가했습니다.

- **Performance Highlights**: GenMedClip 모델은 11개의 의료 도메인에서 수집된 새로운 벤치마크 데이터셋을 통해 평균적으로 기존 SOTA 모델인 BiomedCLIP보다 분류 및 검색 작업에서 각각 3%와 14% 향상된 성능을 보였습니다. 이는 각각의 의료 이미징 모달리티에서 뛰어난 성능 향상을 보여줍니다. 향후 이 데이터셋을 활용하여 더 많은 접지된 생성 모델과 개방형 어휘(segmentation/detection) 모델을 훈련할 수 있기를 기대합니다.



### MM-GEN: Enhancing Task Performance Through Targeted Multimodal Data Curation (https://arxiv.org/abs/2501.04155)
- **What's New**: 이번 논문에서는 MM-Gen이라는 새로운 방법론을 소개합니다. 이 방법은 고품질의 합성 텍스트(annotation)를 생성하여 특정 작업(task)에 맞는 시각-언어 모델(VLM)의 성능을 높입니다. MM-Gen은 세 가지 단계의 프로세스를 통해 필요한 데이터를 효과적으로 생성하며, 기존의 양질의 데이터가 부족한 문제를 해결합니다.

- **Technical Details**: MM-Gen은 강력한 VLM을 활용하여 이미지와 연관된 텍스트 주석을 생성합니다. 이 과정은 크게 세 가지로 나뉘며, (1) 데이터를 소그룹으로 분리하고, (2) 작업 설명에 기초하여 목적에 맞는 텍스트 생성, (3) 불필요한 데이터 필터링을 통해 이루어집니다. 이를 통해 VLM을 fine-tuning하면 다수의 작업에서 성능을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, MM-Gen을 통해 Llava-1.5(7B) 모델은 차트 이해, 공간 추론 및 도표 이해에서 각각 15%, 14%, 29%의 성능 향상을 기록했습니다. 원본 모델에 비해 MM-Gen 데이터는 최대 1.6배 높은 개선 효과를 보이며, 이를 통해 특정 작업에서 VLM 성능을 증가시킬 수 있음을 입증합니다.



### Benchmarking Large and Small MLLMs (https://arxiv.org/abs/2501.04150)
- **What's New**: 이번 연구는 대형 다중 모달 언어 모델(MLLMs)인 GPT-4V와 GPT-4o와 소형 MLLMs인 LLava 시리즈 모델, Phi-3-Vision 간의 성능 경계를 체계적으로 평가했습니다. 특히 소형 모델이 진입장벽을 낮추고 빠른 추론 시간과 낮은 배포 비용을 제공하는 가능성을 강조합니다. 또한, 특정 시나리오에서 소형 모델이 대형 모델과 유사한 성능을 보여주지만 복잡한 작업에서 부족한 점을 발견했습니다.

- **Technical Details**: 연구팀은 물체 인식(object recognition), 시간적 추론(temporal reasoning), 다중 모달 이해(multimodal comprehension) 등 일반적인 능력을 포함한 다양한 평가를 수행했습니다. 이 과정에서 산업 및 자동차와 같은 실제 응용 분야에서의 성능도 분석하였습니다. 평가 결과, 소형 MLLMs는 특정 기준에서 대형 모델과 유사한 성능을 보이나, 심층적 추론이나 미묘한 이해가 필요한 복잡한 작업에서는 뒤처진 것으로 나타났습니다.

- **Performance Highlights**: 우리는 소형 및 대형 MLLMs 모두에서 공통적인 실패 사례를 식별했습니다. 이로 인해 최신 모델조차도 어려움을 겪는 도메인을 강조합니다. 연구 결과는 MLLMs의 품질 경계를 발전시키고 다양한 응용 분야에서의 유용성과 효과를 높이기 위한 연구 커뮤니티에 가이드를 제공할 것으로 기대합니다.



### Chirpy3D: Continuous Part Latents for Creative 3D Bird Generation (https://arxiv.org/abs/2501.04144)
Comments:
          20 pages

- **What's New**: 이 논문에서는 창의적인 3D 생성의 한계를 확장하여 기존의 객체를 단순 모방하지 않고도 완전히 새로운 3D 객체를 생성하는 방법을 제시합니다. 2D에서의 세밀한 이해를 3D로 전이하고, 부분 잠재 변수를 연속 분포로 모델링하여 새로운 파트를 생성하는 능력을 개발했습니다. 새로운 자아 지도 특징 일관성 손실(self-supervised feature consistency loss)을 도입하여 이 생성 과정의 안정성을 보장합니다.

- **Technical Details**: 주요 기술적 기여 중 하나는 다중 관점을 복원하는 MVDream 프레임워크를 통해 2D 세밀한 이미지를 3D 공간으로 전이하는 것입니다. 우리는 part latent space를 Gaussian 분포로 모델링하여 훈련 종족 간의 연속적 변화를 포착하면서 상호 보완적인 무작위 샘플링과 보간(interpolation)을 가능하게 합니다. 또한, 크로스 어텐션 기능 맵 유사성을 강조하는 새로운 자가 지도 손실을 도입하여 '보지 못한' 파트를 생성하는 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, 이 방법은 지금까지 존재했던 예시를 초월하여 새로운 3D 객체를 생성하는 초기 시스템을 보여줍니다. 주요 특징은 비대칭 개체의 세부를 유지하면서 완전히 새로운 3D 조류를 생성할 수 있다는 것입니다. 이러한 기술 혁신은 새를 넘어 모든 세밀한 생성 작업에 적용 가능성이 있는 도메인 비종속적(domain-agnostic)입니다.



### Graph-Based Multimodal and Multi-view Alignment for Keystep Recognition (https://arxiv.org/abs/2501.04121)
Comments:
          9 pages, 6 figures

- **What's New**: 이번 연구에서는 egocentric 비디오에서 keystep 인식을 위한 유연한 그래프 학습 프레임워크를 제안합니다. 기존 방법들보다 12포인트 이상 높은 정확도로 성능을 개선하며, egocentric 비디오의 long-term dependencies를 효과적으로 활용합니다. 다양한 멀티모달 기능(예: 내레이터, 깊이, 객체 클래스 레이블)을 활용하여 heterogenous 그래프에서의 기여도를 분석하고 있습니다.

- **Technical Details**: 우리는 MAGLEV(Multi-view Alignment with Graph for Long and Effective Video understanding)라는 그래프 기반 프레임워크를 제안합니다. 이 프레임워크는 egocentric 비디오의 각 클립을 노드로 구성하여 그래프를 형성하며, exocentric 비디오를 훈련 시 추가 노드로 사용합니다. 각 노드 간의 연결을 정의하는 여러 전략을 시험하고, keystep 인식을 노드 분류(task)로 설정하여 실험하고 있습니다.

- **Performance Highlights**: Ego-Exo4D 데이터셋에서 광범위한 실험을 통해 MAGLEV가 기존 egocentric 방법들보다 15% 이상 성능을 향상시킴을 입증했습니다. multi-view(egocentric 및 exocentric) 및 multi-modal 정렬 기능을 활용하여 성능이 더욱 개선되었으며, 이미 최첨단 방법으로 인정받고 있습니다. 이러한 효율적인 접근 방식은 메모리 및 컴퓨팅 자원에서도 효율적입니다.



### NeRFs are Mirror Detectors: Using Structural Similarity for Multi-View Mirror Scene Reconstruction with 3D Surface Primitives (https://arxiv.org/abs/2501.04074)
- **What's New**: 이번 논문에서는 NeRF-MD라는 새로운 방법을 제시합니다. 이 방법은 NeRF를 거울 탐지기로 활용하여, 사전 주석 없이 반사 표면이 포함된 장면의 neural radiance fields를 재구성할 수 있도록 합니다. 기존 방법들과 달리 사용자의 추가적인 주석 필요 없이 거울 탐지와 재구성을 동시에 가능하게 합니다.

- **Technical Details**: NeRF-MD의 핵심 아이디어는 복잡한 장면 기하학을 추정하는 과정에서, 반사 표면에 해당하는 장면의 일부에서 보이는 현저한 포토메트릭 불일치(photometric inconsistency)를 활용하는 것입니다. 이를 바탕으로 초기 학습 단계에서 geometrical primitives를 사용해 불일치 지역을 감지하고, 이후 두 번째 학습 단계에서 radiance field와 거울 기하학을 공동 최적화하여 품질을 향상시킵니다. 이 과정에서 depth reprojection loss를 사용하여 신뢰할 수 있는 초기 기하 추정을 제공합니다.

- **Performance Highlights**: 제안된 방법은 장면에서의 거울을 정확하게 탐지할 수 있는 능력을 보여줍니다. 또한, 전체적인 장면 재구성이 일관되게 이루어지며, 기존의 기준선(baseline) 및 거울 인식 접근 방식에 비해 우수성을 입증합니다. 이 연구의 결과는 다양한 분야에서 응용될 수 있는 잠재력을 지니고 있습니다.



### Planarian Neural Networks: Evolutionary Patterns from Basic Bilateria Shaping Modern Artificial Neural Network Architectures (https://arxiv.org/abs/2501.04700)
Comments:
          11 pages, 9 figures

- **What's New**: 이번 연구에서는 인공 신경망 (ANNs)의 이미지 분류 정확도를 증가시키기 위한 새로운 방안을 제시합니다. 생물 신경망의 진화 패턴을 모델로 삼아, 플라나리안 (planarians)의 신경 구조에서 영감을 받은 ANNs를 개발했습니다. 이를 통해 ANNs의 성능 향상이 가능하다는 점을 강조하며, ResNet을 기본 모델로 선택하여 연구를 진행했습니다.

- **Technical Details**: 연구는 플라나리안의 신경 구조가 포함된 새로운 신경망 아키텍처를 바탕으로 하여, CIFAR-10 및 CIFAR-100 데이터셋에서 평가되었습니다. 플라나리안의 두 개의 신경줄과 뇌를 포함하는 독특한 구조는 ANNs의 성능 개선에 중요한 통찰력을 제공합니다. 이 연구는 이러한 생물적 영감이 주는 가능성을 살펴보고자 하였습니다.

- **Performance Highlights**: 결과적으로, 제안된 방법은 기존의 기본 신경망 모델에 비해 이미지 분류 과제에서 더 높은 예측 정확도를 보였습니다. 이는 다양한 응용 분야에서 ANNs의 성능을 향상시킬 수 있는 생물학적으로 영감을 받은 신경망 아키텍처의 중요성을 보여줍니다.



### Grokking at the Edge of Numerical Stability (https://arxiv.org/abs/2501.04697)
- **What's New**: 이번 연구에서는 grokking 현상에 대한 새로운 관점을 제시합니다. 연구자들은 정규화 없이 모델이 학습 중 발생하는 Softmax Collapse (SC)와 같은 수치적 불안정을 방지하기 위한 StableMax와 ⟂Grad와 같은 새로운 기법을 도입했습니다. 이러한 기법들은 grokking을 촉진하고, 기존의 방법들이 왜 효과적인지를 설명하는 데 도움을 줍니다. 이를 통해 delayed generalization의 기저 원인에 대한 인사이트를 제공합니다.

- **Technical Details**: grokking은 과적합(Overfitting) 후 예상치 못한 일반화를 의미합니다. 연구팀은 SC라는 현상이 이러한 일반화가 발생하지 않는 이유 중 하나로 작용한다고 주장합니다. SC는 Softmax 함수의 부동 소수점 오류로 인해 발생하며, 이는 모델의 성능이 정체되거나 오히려 저하되는 결과를 초래합니다. Naïve Loss Minimization (NLM) 방향으로 그래디언트가 정렬되어 가며, 이는 최종적으로 loss를 감소시키지만 SC를 유발합니다.

- **Performance Highlights**: StableMax라는 새로운 활성화 함수는 SC를 방지하며, 정규화 없이도 grokking을 가능하게 합니다. ⟂Grad 옵티마이저는 NLM을 억제하여 grokking 작업에서 빠른 일반화를 촉진합니다. 본 연구의 결과는 정규화가 없는 환경에서도 grokking을 유도할 수 있는 새로운 방법론을 제공하며, 이는 딥러닝에서의 일반화 이해를 한층 더 발전시킵니다.



### Re-ranking the Context for Multimodal Retrieval Augmented Generation (https://arxiv.org/abs/2501.04695)
- **What's New**: 이번 논문은 Retrieval-augmented generation (RAG)을 활용하여 대형 언어 모델(LLMs)을 개선하기 위한 새로운 접근을 제안합니다. 특히, multi-modal RAG 시스템의 정보 검색(retrieval) 과정에서 관련성을 보다 정확하게 평가하기 위해 새롭게 개발한 relevancy score (RS) 모델을 도입했습니다. 기존 CLIP 기반의 방법의 한계를 극복하고 보다 정밀한 검색 결과를 얻기 위해 RS 모델을 사용하여 부정확한 정보를 제거하는 방법에 초점을 맞췄습니다.

- **Technical Details**: RS 모델은 VLM(vision-language models)을 활용하여 사용자 쿼리와 검색된 엔트리 간의 의미적 관계를 학습합니다. 훈련 데이터셋은 인간이 주석을 단 데이터와 ChatGPT로 생성된 합성 쿼리-컨텍스트 쌍을 포함하여 균형 잡힌 triplet 형태로 구성됩니다. RS 모델의 출력은 0에서 1까지의 스칼라 점수로, 높은 점수는 그만큼 쿼리와 높은 관련성을 나타냅니다.

- **Performance Highlights**: COCO 데이터셋을 사용한 평가 결과, RS 기반의 재정렬(re-ranking) 방법이 검색 이미지의 품질을 크게 향상시키고, 더 정확하고 사실 기반의 응답을 생성하는 것으로 나타났습니다. RS는 CLIP에 비해 의미적 유사성을 넘어 정확한 맥락적 관련성을 포착하는 데 뛰어난 성능을 보였으며, 검색 정밀성을 높여 hallucinations(비현실적 결과)를 줄일 수 있는 잠재력을 지니고 있음을 강조합니다.



### RadGPT: Constructing 3D Image-Text Tumor Datasets (https://arxiv.org/abs/2501.04678)
- **What's New**: 이번 연구에서는 RadGPT라는 Anatomy-Aware Vision-Language AI Agent를 소개합니다. 해당 모델은 CT 스캔에서 종양 관련 보고서를 생성하는 데 도움을 주며, 양성 낭종과 악성 종양을 포함한 종양을 분할하고, 구조적 및 서사적 보고서로 변환합니다. RadGPT는 종양의 크기, 모양, 위치 등을 정밀하게 분석할 수 있습니다.

- **Technical Details**: RadGPT는 CT 스캔과 구조적 보고서만을 사용하는 훈련 전략을 적용하여 모델을 학습시켰습니다. 모델의 성능 향상을 위해 최적의 하이퍼파라미터 검색과 훈련 알고리즘을 사용하였고, 이 과정에서 몇 가지 최소한의 조정을 통해 복부 CT 보고서 생성을 위한 독특한 도전 과제를 해결했습니다. 기존 높은 노이즈 발생 문제를 해결하기 위해 특수한 알고리즘을 도입하였습니다.

- **Performance Highlights**: RadGPT는 독립된 병원에서 평가를 거친 결과, 특히 작은 종양(<2 cm) 감지에서 높은 정확도를 보였습니다: 간 종양의 경우 80/73%, 신장 종양은 92/78%, 췌장 종양은 77/77%의 민감도/특이도가 나타났습니다. 이 연구는 17개의 공개 데이터셋을 기반으로 하며, 8,562개의 관찰에서 948개의 초기 단계 종양에 대한 보고서를 포함합니다.



### Enhancing Financial VQA in Vision Language Models using Intermediate Structured Representations (https://arxiv.org/abs/2501.04675)
- **What's New**: 이 연구는 50,000개의 막대 차트에 대해 고유한 구조적 특성을 활용하여 차트 이미지를 선형화된 테이블로 변환하는 DEPLOT(모드 전환 모듈)의 미세 조정을 조사합니다. 미세 조정된 DEPLOT 모델은 카테고리별 매핑 정확도를 측정하는 Relative Mapping Similarity(RMS)와 수치적 해석 정확도를 평가하는 Relative Number Set Similarity(RNSS)를 통해 기본 모델과 비교 평가됩니다. 또한, 100개의 차트 이미지와 질문-응답 세트를 추가하여 대규모 언어 모델(LLMs)의 추론 능력을 탐구합니다.

- **Technical Details**: DEPLOT은 시각 차트 데이터를 구조화된 데이터 테이블로 매핑하기 위한 모드 전환 모듈로, 다양한 차트 유형에서 훈련되지만 도메인별 데이터 세트를 사용하여 미세 조정할 수 있습니다. 본 논문에서는 RNSS와 RMS 두 가지 주요 지표를 통해 모델의 정량적 및 범주적 해석 능력을 평가하며, 차트 구조를 추적할 수 있는 능력을 강조합니다. 이러한 새로운 접근법은 DEPLOT의 성능을 높이고, 보다 신뢰할 수 있는 데이터 시각화 모델 개발을 위한 기초를 제공합니다.

- **Performance Highlights**: 미세 조정된 DEPLOT 모델을 활용한 실험 결과, 높은 품질의 구조화된 데이터와 함께 제공된 경우 LLM의 추론 능력이 크게 향상됨을 보여줍니다. 특히 Qwen2-VL-7B와 같은 소형 모델이 고급 모델인 GPT-4o보다 더 나은 성능을 발휘하여 차트 데이터 해석의 정확성을 높였습니다. 이 연구는 자동 차트 해석 및 추론 향상을 위한 모드 전환 통합의 혁신적 잠재력을 강조합니다.



### HyFusion: Enhanced Reception Field Transformer for Hyperspectral Image Fusion (https://arxiv.org/abs/2501.04665)
Comments:
          Submitted to IGARSS 2025

- **What's New**: 본 논문은 HyFusion이라는 새로운 프레임워크를 제안하여 고해상도 멀티스펙트럴 이미지(HR-MSI)와 저해상도 하이퍼스펙트럴 이미지(LR-HSI)를 융합하여 고해상도 하이퍼스펙트럴 이미지(HR-HSI)를 재구성하는 문제를 해결합니다. 기존의 방법들이 가지는 수신 범위의 제한과 특징 활용의 부족을 극복하기 위해, HyFusion은 효과적인 데이터 활용을 통해 재구성 품질을 극대화합니다. 특히, Enhanced Reception Field Block(ERFB)와 Dual-Coupled Network(DCN) 구조를 통해 정보 손실을 줄이고 데이터 효율성을 높입니다.

- **Technical Details**: HyFusion 프레임워크는 공간적 및 스펙트럴 정보를 효과적으로 결합하여 HR-HSI 재구성을 향상시키는 데 중점을 두고 설계되었습니다. Enhanced Reception Field Block(ERFB)은 이동 창 주의(attention)와 밀집 연결(dense connections)을 조합하여 수신 범위를 확대하며, Dual-Coupled Network(DCN)는 스펙트러와 공간 특징을 동적으로 추출하여 효율적인 크로스 도메인 융합을 지원합니다. 이러한 아키텍처는 HSI 응용 프로그램의 계산 및 하드웨어 제한을 고려하여 간단하고 효율적으로 설계되었습니다.

- **Performance Highlights**: 실험 결과, HyFusion은 HR-MSI/LR-HSI 융합 작업에서 최신 성능을 달성하였으며, 재구성 품질을 크게 향상시키면서 компакт한 모델 크기와 계산 효율성을 유지합니다. 향상된 수신 범위와 특징 맵 재사용을 통합함으로써 HyFusion은 자원-제한 환경에서 HSI 융합을 위한 실용적이고 효과적인 솔루션을 제공하며, 하이퍼스펙트럴 이미징의 새로운 기준을 설정합니다.



### FlairGPT: Repurposing LLMs for Interior Designs (https://arxiv.org/abs/2501.04648)
Comments:
          Accepted at EUROGRAPHICS 2025

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 내부 디자인에 어떻게 활용할 수 있는지를 탐구합니다. 기존의 데이터 기반 접근 방식과 달리, 이 연구는 LLMs가 디자인의 다양한 제약 조건을 효과적으로 생성할 수 있음을 보여주며, 이를 통해 내부 공간의 디자인 과정을 보다 구조화할 수 있는 방법을 제시합니다. 최종적으로, LLMs의 출력을 기반으로 한 레이아웃 제약 그래프를 통한 최적화를 통해 높은 품질의 디자인 레이아웃을 생성할 수 있습니다.

- **Technical Details**: 이 연구는 LLMs의 구조적 활용을 통해 사용자가 제시한 공간을 여러 영역으로 나누고, 각 영역에 배치할 개체 목록 및 제약 조건을 추출합니다. 이러한 정보를 바탕으로 레이아웃 제약 그래프를 구성한 후, 이를 해결하기 위해 외부 도구인 제약 최적화 설정을 사용하여 최종 디자인을 생성합니다. 권장된 방법에서는 LLM의 출력을 대수적 제약 조건으로 변환하여 개체 변수의 크기 및 배치를 기반으로 한 레이아웃을 도출합니다.

- **Performance Highlights**: 본 연구에서 제안한 방법은 기존의 LLM 기반 메서드 및 인간 디자인과 비교하여 다양한 디자인 설정에서 평가되었습니다. 사용자 연구를 통해, 사용자들은 생성된 레이아웃이 디자인 사양에 잘 맞춰져 있으며 기능적으로 유용하다고 평가하여 우리의 방법을 선호했습니다. 이러한 경험적 평가로, LLMs를 구조적으로 활용할 수 있을 때 고품질의 다양한 레이아웃 생성이 가능함을 입증했습니다.



### Comprehensive Examination of Unrolled Networks for Linear Inverse Problems (https://arxiv.org/abs/2501.04608)
Comments:
          27 pages, 10 figures. Project Page: this https URL

- **What's New**: 이번 논문에서는 unrolled networks의 개념을 통해 다양한 컴퓨터 비전과 이미지 처리 작업에서의 디자인 결정을 줄일 수 있는 방법을 제안합니다. 기존의 어려운 디자인 선택 상황을 해결하기 위해, 연구진은 유용한 방법론들과 아이디어들을 통합하는 것을 목표로 하며, unrolled networks의 설계에서 각 선택의 영향을 설명하는 포괄적인 ablation study를 제공합니다.

- **Technical Details**: 연구는 linear inverse problems에 적용 가능성을 가진 unrolled networks의 개발을 중점적으로 다룹니다. 특히, 측정 과정은 자료 수집과 데이터의 복원 과정을 수학적으로 모델링하며, 여러 측정 행렬과 그 속성을 고려하여 최적의 성능을 이끌어내는 방법을 논의합니다. 이러한 네트워크는 심층 학습을 필요로 하지만, 그 과정에서 발생하는 과도한 계산 부담을 줄이기 위한 방법들을 소개합니다.

- **Performance Highlights**: 이 논문은 unrolled networks의 효과를 높이고 디자인 결정의 복잡성을 줄이기 위한 실용적인 권고사항들을 제시합니다. 연구는 다양한 최적화 알고리즘과 neural network을 결합하여 성능을 향상시킬 수 있는 가능성을 탐구하며, 새로운 응용 분야에 대한 적응성을 강조합니다. 이를 통해 연구자들이 unrolled networks를 손쉽게 설계하고, 네트워크 내 문제를 효율적으로 진단할 수 있는 방안을 제시하고자 합니다.



### FrontierNet: Learning Visual Cues to Explor (https://arxiv.org/abs/2501.04597)
- **What's New**: 이번 연구는 자율 로봇의 미지의 환경 탐색을 위한 새로운 접근법을 제안합니다. 기존의 3D 맵 기반 방법의 한계를 보완하기 위해, 2D RGB 이미지에서 유용한 단서를 활용하여 목표 위치를 효율적으로 탐색합니다. FrontierNet이라는 학습 기반 모델을 통해, 이미지에서 직접 전선(frontier)을 감지하고 정보 이득을 예측하는 시스템을 개발했습니다.

- **Technical Details**: 제안된 시스템은 개별 RGB 이미지만을 입력으로 사용하며, 이를 통해 로봇의 탐색 결정을 3D 공간과 연결합니다. FrontierNet은 전선 감지와 정보 이득 예측을 함께 수행하며, 단안 깊이 정보를 활용해 탐색 효율성을 높입니다. 이 방식은 3D 맵에 의존하지 않으며, 시각적 단서로부터 경계 정보를 추출할 수 있습니다.

- **Performance Highlights**: 시뮬레이션과 실제 실험을 통해, 제안된 시스템은 초기 탐색 효율성을 기존 시스템보다 16% 향상시켰음을 입증했습니다. 탐색 과정에서 3D 맵의 품질에 대한 의존성을 줄여, 보다 효율적인 탐색 경로를 제공합니다. 이로써 인프라 모델링, 검색 및 구조, 작물 모니터링 등 다양한 응용 분야에서 활용 가능한 가능성을 보여줍니다.



### OpenOmni: Large Language Models Pivot Zero-shot Omnimodal Alignment across Language with Real-time Self-Aware Emotional Speech Synthesis (https://arxiv.org/abs/2501.04561)
- **What's New**: 최근 오믹모달(omnimodal) 학습의 발전으로 이미지, 텍스트, 음성 간의 이해 및 생성이 가능해졌습니다. 하지만, 제한된 오믹모달 데이터셋과 실시간 감정 음성 생성의 어려움으로 인해 오픈소스 진전이 제약받고 있습니다. 이를 해결하기 위해 openomni라는 두 단계 훈련 방법을 제안하여 최첨단 오믹모달 대형 언어 모델을 개발합니다.

- **Technical Details**: openomni는 두 단계로 구성된 훈련 방식을 채택합니다. 첫 번째는 오믹모달 정렬(omnimodal alignment)으로, 사전 훈련된 음성 모델을 텍스트-이미지 작업에 추가로 훈련하여 비전에서 음성으로의 일반화를 달성합니다. 두 번째 단계는 실시간 감정 음성을 생성하기 위한 경량 디코더를 사용하여 음성 작업과 선호 학습에 대해 훈련을 수행합니다.

- **Performance Highlights**: openomni는 오믹모달, 비전-언어 및 음성-언어 평가에서 일관된 성능 향상을 보여줍니다. 기존의 대형 오픈소스 모델과 비교할 때, openomni는 훨씬 적은 모델 크기 및 훈련 데이터로 OmniBench 벤치마크에서 탁월한 성능을 기록하며, 실시간 음성 생성, 음성 이해 및 이미지-텍스트 질문 응답과 같은 다양한 이중 모달 작업에서도 경쟁력 있는 결과를 도출합니다.



### Towards Fair Class-wise Robustness: Class Optimal Distribution Adversarial Training (https://arxiv.org/abs/2501.04527)
- **What's New**: 이 논문에서는 Class Optimal Distribution Adversarial Training (CODAT)라는 새로운 min-max 훈련 프레임워크를 제안하였습니다. CODAT는 distributionally robust optimization (DRO) 이론을 활용하여 클래스별 가중치 공간을 철저히 탐색하고, 이론적 보장을 통해 최적의 가중치를 식별할 수 있는 방법을 제공합니다. 또한 내부 극대화 문제에 대한 닫힌 형태의 최적 해를 도출하고, 이를 통해 가중치와 모델 매개변수의 일관된 최적화 방향을 보장하는 결정론적 동등 목표 함수를 얻습니다.

- **Technical Details**: CODAT는 min-max 최적화 문제로 형식화될 수 있으며, 내부 극대화 목표는 클래스별 기대 리스크를 극대화하는 데 목표를 두고 있습니다. CODAT는 worst-case class adversarial distribution을 학습하기 위해 클래스 적대적 분포 공간을 완전히 탐색할 수 있도록 설계되었습니다. 또한, 가중치와 모델의 최적화 방향을 일관되게 유지하기 위해 내부 극대화 문제에 대한 닫힌 형태의 최적 해를 파생하고 원래 모델 목표에 통합된 결정론적 동등 목표 함수를 제공합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 수행된 실험 결과, CODAT는 기존의 최첨단 방법들을 초과하는 성능을 보였으며, 모델의 robust fairness를 효과적으로 개선할 수 있음을 입증하였습니다. 이 연구에서 제안된 fairness elasticity coefficient는 알고리즘의 견고성과 공정성을 모두 평가하는 데 사용됩니다. CODAT는 복잡한 분야에서도 적용 가능한 새로운 접근법으로, 특히 보안이 중요한 분야에서 악의적인 공격으로부터의 저항력을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### SplineFormer: An Explainable Transformer-Based Approach for Autonomous Endovascular Navigation (https://arxiv.org/abs/2501.04515)
Comments:
          8 pages

- **What's New**: 이 논문에서는 SplineFormer라는 새로운 transformer 기반의 아키텍처를 소개합니다. 이 모델은 가이드와이어의 연속적이고 부드러운 형태를 예측할 수 있도록 설계되어, 복잡한 혈관 내에서의 정확한 내비게이션을 가능하게 합니다. SplineFormer는 기존의 세그멘테이션 방식의 한계를 극복하고, 설명 가능한 방식으로 가이드와이어 형태를 포착할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: SplineFormer는 B-spline transformer 모델로, 가이드와이어의 고유한 기하학적 구조를 효과적으로 활용합니다. 이 네트워크는 유의미하고 간결한 표현을 획득하고, 이 잠재 공간을 사용하여 해부학 구조를 성공적으로 내비게이션하기 위한 적절한 행동을 이끌어내도록 훈련됩니다. 실험에서는 로봇이 완전히 자율적으로 내혈관 내비게이션 작업을 수행할 수 있는 능력을 보여줍니다.

- **Performance Highlights**: 우리의 실험 결과, SplineFormer는 실제 로봇을 사용하여 팔목 동맥을 카뉼화하는 데 50%의 성공률을 기록했습니다. 이는 기존의 방법들과 비교해 혁신적인 접근 방식이며, 내혈관 내비게이션의 안전성과 효율성을 크게 향상시킬 수 있는 잠재력을 보여줍니다. 또한, SplineFormer의 설명 가능한 설계는 향후 임상 환경에서의 신뢰성을 높일 것으로 기대됩니다.



### The Role of Machine Learning in Congenital Heart Disease Diagnosis: Datasets, Algorithms, and Insights (https://arxiv.org/abs/2501.04493)
- **What's New**: 이 논문은 선천성 심장병(Congeital Heart Disease, CHD) 인식에 대한 머신러닝(Machine Learning) 기반의 체계적인 문헌 리뷰(Systematic Literature Review, SLR)를 제공합니다. 2018년부터 2024년까지의 주요 저널에 게재된 432개의 참고문헌을 메타 분석하고, 74개의 주요 연구 결과를 심층 분석하였습니다. 또한, 본 연구는 머신러닝 전문가들이 CHD 인식에 사용한 데이터셋을 정리하여 주목받는 데이터를 실제로 활용하는 방법을 제시합니다.

- **Technical Details**: 이 논문에서는 머신러닝 알고리즘 및 데이터베이스를 사용하는 CHD 인식에서의 최근 발전을 포괄적으로 다룹니다. SLR 방법론을 통해 연구 팀은 다양한 진단 방법, 사용된 데이터셋, 알고리즘 적용, 그리고 솔루션을 분석하였습니다. 주요 머신러닝(Machine Learning) 및 딥러닝(Deep Learning) 알고리즘에 대한 강점과 약점을 조명하며, 이들 기법의 실질적인 응용 방법을 제공합니다.

- **Performance Highlights**: CHD-ML 분야는 여전히 충분히 탐구되지 않은 영역으로, 연구자와 전문가들에게 중요한 참고자료로 사용될 수 있습니다. 이 리뷰는 CHD 인식을 위한 최신 머신러닝 및 딥러닝 접근 방식에 대한 포괄적인 분석을 제공하며, CHD-ML 시스템 개발을 위한 주요 요소를 확인합니다. 연구 결과는 의료 실무자와 연구자 간의 협력을 통해 이 중요한 분야의 발전을 촉진하는 데 기여할 것으로 예상됩니다.



### Rapid Automated Mapping of Clouds on Titan With Instance Segmentation (https://arxiv.org/abs/2501.04459)
- **What's New**: 이번 논문에서는 행성 과학에서 Mask R-CNN을 활용한 새로운 접근법을 제안합니다. 특히, 카시니 우주선이 촬영한 타이탄(Titan) 이미지에서 구름(instance segmentation) 분석을 수행하여 기존의 전통적인 방법에 비해 효율성을 높입니다. 이러한 방법은 데이터의 정량적 분석을 가능하게 하며, 행성 기후 연구에 기여할 것으로 기대됩니다.

- **Technical Details**: Mask R-CNN 모델은 전이 학습(transfer learning)을 통해 구름의 영역(area)과 중심점(centroids)을 자동으로 측정합니다. 본 연구는 타이탄의 복잡한 기후 데이터 분석을 진행하며, 자동화 기술이 기존의 수작업 맵핑에 비해 시간적으로 효율적임을 보여줍니다. 특히, 타이탄에서의 도전에 불구하고, 정확성은 지구와 다른 세계에서의 현대 구름 식별 연구와 유사한 수준으로 나타났습니다.

- **Performance Highlights**: 인간의 방식과 알고리즘 기반 접근 방식의 효율성을 비교한 결과, 전이 학습을 통해 데이터 조사 속도가 크게 향상되는 것을 확인했습니다. 앞으로의 행성 탐사 임무와 원거리 감지(remote sensing) 이니셔티브는 대량의 이미지 데이터를 생성할 것이며, 기계 학습 접근법이 이러한 분석에 큰 도움을 줄 것으로 기대됩니다.



### On Computational Limits and Provably Efficient Criteria of Visual Autoregressive Models: A Fine-Grained Complexity Analysis (https://arxiv.org/abs/2501.04377)
- **What's New**: 최근 Visual Autoregressive (VAR) 모델이 이미지 생성 분야에서 혁신적인 발전을 가져왔습니다. 이 모델은 coarse-to-fine "next-scale prediction" 접근 방식을 통해 확장 가능한 방식으로 이미지 생성을 가능하게 합니다. 그러나 VAR 모델의 현재 최첨단 알고리즘은 O(n^4)의 시간 복잡도를 가지며, 이는 계산 효율성이 낮습니다. 따라서 본 연구는 VAR 모델의 계산 한계와 효율성을 분석하여 개선 방안을 제시합니다.

- **Technical Details**: VAR 모델의 효율성을 갖추기 위해서는 입력 행렬의 노름이 특정 활용 범위 아래에 있어야 하며, 이 기준을 초과할 경우 진정한 서브-쿼틱(sub-quartic) 시간 알고리즘을 설계하는 것이 불가능한 것으로 확인했습니다. 본 연구에서는 Strong Exponential Time Hypothesis (SETH)를 바탕으로 VAR 모델의 계산 성능을 평가하는 새로운 기준을 설정하였습니다. 이를 통해 계산 비용을 절감할 수 있는 경량화된 구조도 제안합니다.

- **Performance Highlights**: 본 연구의 기여는 VAR 모델의 계산 시간을 O(n^4)보다 더 빠르게 수행할 수 있는 조건을 제시한 것입니다. 입력 행렬의 요소가 특정 임계값을 초과하지 않을 경우 효율적인 알고리즘을 통해 거의 저차 제곱 시간 복잡도 O(n^{2+o(1)})로 VAR 모델을 근사할 수 있음을 보여줍니다. 이러한 발견은 VAR 모델의 이론적 발전을 위한 기초가 되며, 향후 스케일러블한 이미지 생성기를 발전시킬 수 있는 길을 열어줄 것입니다.



### A Unified Framework for Foreground and Anonymization Area Segmentation in CT and MRI Data (https://arxiv.org/abs/2501.04361)
Comments:
          6 pages

- **What's New**: 이번 연구에서는 3D 의료 이미지를 위한 self-supervised learning (SSL) 데이터 전처리의 주요 과제를 해결하는 오픈소스 툴킷을 제안합니다. 이 툴킷은 데이터 샘플링 최적화를 위해 전경(foreground) 영역을 구분하는 세그멘테이션 네트워크와, 잘못된 감독을 방지하기 위해 익명화된 영역을 식별하는 네트워크 두 가지로 구성되어 있습니다. 실험 결과는 모든 익명화 방법에서 평균 Dice 점수가 98.5를 초과하며, 전경 세그멘테이션 작업에서는 99.5를 초과했습니다.

- **Technical Details**: 이 연구에서 구축한 데이터 세트는 3299개의 3D 이미지로 구성되어 있으며, 이 중 1899개는 CT 스캔, 1400개는 MRI 스캔입니다. 전경 세그멘테이션을 위한 nnU-Net 모델이 훈련되었고, CT와 MRI 이미지를 포함한 모든 훈련 데이터를 사용하여 동시 훈련이 이루어졌습니다. 익명화된 얼굴 부분의 세그멘테이션은 OASIS 3 데이터셋을 사용하여 훈련되었으며, AFNI 소프트웨어를 통해 세 가지 익명화 방법을 적용하여 3015개의 익명화된 이미지가 생성되었습니다.

- **Performance Highlights**: 모델은 관계된 훈련 데이터셋과 외부 데이터셋의 테스트에서 각각 평균 Dice 계수 99.56 및 98.57을 기록하며 뛰어난 성능을 보여주었습니다. 특히, 외부 테스트 데이터셋에서도 상당한 강인성을 발휘하였으며, 모델의 세그멘테이션 정확도는 매우 높습니다. 이러한 결과는 SSL 메서드에 매우 유용한 자원으로서 이 툴킷의 활용 가능성을 시사합니다.



### Robotic Programmer: Video Instructed Policy Code Generation for Robotic Manipulation (https://arxiv.org/abs/2501.04268)
- **What's New**: 이번 연구에서는 Robotic Programmer (RoboPro)라는 로봇 기초 모델을 제안하며, 이는 고급 작업 지침을 이해하고 시각 정보를 인식하여 정책 코드를 통해 로봇 조작을 수행할 수 있는 능력을 갖추고 있습니다. Video2Code라는 자동 데이터 커레이션 파이프라인을 통해 실제 비디오에서 실행 가능한 코드를 합성하여 로봇 작업을 위한 런타임 코드 데이터 수집의 비효율성과 높은 비용 문제를 해결하고자 하였습니다. 이를 통해 RoboPro는 시뮬레이터 및 실제 환경 모두에서 혁신적인 제로샷 제어 성능을 달성하였습니다.

- **Technical Details**: RoboPro는 자유형식 언어 지침(I)과 RGBD 데이터를 기반으로 로봇의 낮은 차원 상태(s)에서 동작 궤적(T)을 생성하는 시스템입니다. 이를 위해 정책 코드 생성 방법은 장기 지침을 다양한 원자 스킬 세트로 매핑하여 다양한 로봇 플랫폼에서의 빠른 적응을 가능하게 합니다. 또한 VLM(visual-language models)이 지능형 계획자로서 기능하여 작업 실행 프로세스를 생성된 프로그램으로 변환하는 역할을 합니다.

- **Performance Highlights**: RoboPro는 RLBench에서 제로샷 성공률이 11.6% 향상되어 최신 모델 GPT-4o를 초월하는 성과를 보였습니다. 이는 강력한 감독 훈련 방법인 PerAct와 비교 할 때도 매우 유사한 성능을 입증합니다. 또한 RoboPro는 API 형식 및 스킬 세트의 변화에 대한 강인성을 보여주어 다양한 환경에서의 활용 가능성을 확장합니다.



### GRAPHITE: Graph-Based Interpretable Tissue Examination for Enhanced Explainability in Breast Cancer Histopathology (https://arxiv.org/abs/2501.04206)
Comments:
          24 Pages, 9 Figures, 1 Tables

- **What's New**: 이번 연구에서는 유방암 조직 마이크로어레이(TMA) 분석을 위해 설계된 GRAPHITE(그래프 기반 해석 가능한 조직 검사)라는 새로운 포스트 혹 설명 가능성 프레임워크를 소개합니다. GRAPHITE는 다양한 배율에서 패치를 추출하고 계층적 그래프를 구축하는 멀티스케일 접근 방식을 활용합니다. 이는 깊은 학습 모델의 해석 가능성과 임상 신뢰성을 향상시키는 데 중요한 역할을 하여, 의료 분야에서의 AI 도구의 채택을 촉진할 것으로 기대됩니다.

- **Technical Details**: GRAPHITE는 140개의 종양 TMA 코어와 4개의 양성 전체 슬라이드 영상으로 학습되었으며, 53개의 병리학자 주석 TMA 샘플에서 성능을 평가했습니다. 그래프 주의 네트워크(GAT)와 스케일별 주의 네트워크(SAN)를 활용하여 스케일 의존적인 피쳐를 포착하며, 여러 배율에서의 패치 분석을 통해 계층적 의존성을 적절히 캡처합니다.

- **Performance Highlights**: GRAPHITE는 전통적인 XAI 방법에 비해 뛰어난 성능을 보였으며, 평균 평균 정밀도(mAP)가 0.56, 수신자 조작 특성 곡선 아래 면적(AUROC) 0.94, 그리고 임계값 강건성(ThR) 0.70을 달성했습니다. 임상 유용성에 있어서도 GRAPHITE는 4.17e+5의 결정 곡선 아래 면적(AUDC)를 기록하며 다양한 임계값에서 신뢰할 수 있는 결정 지원을 제공함을 입증하였습니다.



### Machine Learning for Identifying Grain Boundaries in Scanning Electron Microscopy (SEM) Images of Nanoparticle Superlattices (https://arxiv.org/abs/2501.04172)
- **What's New**: 이 연구에서는 나노입자 슈퍼라티스(nanoparticle superlattices)의 전자 현미경(SEM) 이미지를 자동으로 분석하는 머신러닝 작업 흐름을 제시합니다. 기존의 수동 분석 방법은 수고스럽고 오류에 취약하여, 미세구조(microstructure) 특성을 정량화하는 데 어려움이 있었습니다. 이 새로운 접근법은 수동 주석 데이터 없이도 나노입자의 배열을 식별하고 세분화할 수 있습니다.

- **Technical Details**: 작업 흐름은 Radon 변환(Radon transforms)과 비지도 학습 방법인 집합적 계층 클러스터링(agglomerative hierarchical clustering)을 통합하여 섬유 방향성(superlattice orientations)의 수치적 표현으로 원시 픽셀 데이터를 변환합니다. 이 과정을 통해 노이즈가 있는 이미지나 경계 사례(edge cases)에 대해 비교적 높은 강건성을 보여주며, 각각의 이미지에 대한 처리 속도는 분당 4장입니다. 이러한 효율성 덕분에 대규모 데이터셋에 쉽게 적용할 수 있으며, 재료 설계와 분석에 있어 데이터 기반 모델의 통합에 유용합니다.

- **Performance Highlights**: 벤치마크 결과는 이 작업 흐름이 다양한 처리 조건, 예를 들어 온도와 압력에 따른 입자 크기 분포를 정량화하고, 이를 통해 원하는 슈퍼라티스 방향성과 입자 크기를 얻기 위한 처리 조건 조정에 활용될 수 있음을 보여줍니다. 이 시스템은 나노입자 슈퍼라티스를 연구하는 데 있어 기존의 수동 방식보다 현저한 시간 절약과 정확도를 제공합니다.



### Deep Learning for Ophthalmology: The State-of-the-Art and Future Trends (https://arxiv.org/abs/2501.04073)
Comments:
          First version

- **What's New**: 인공지능(AI)와 특히 딥러닝(DL)의 발전이 안과 의학 분야에 큰 변화를 가져오고 있습니다. 이 리뷰는 당뇨병성 망막병증, 녹내장, 노인성 황반퇴행 등 다양한 안과 질환에서의 DL의 최첨단 응용 프로그램을 탐색합니다. AI는 진단 정확도를 높이고 치료 전략을 최적화하며 환자 관리를 개선하는 데 중요한 역할을 하고 있습니다.

- **Technical Details**: 딥러닝(DL) 기술은 이미지 분류, 객체 탐지 및 세분화에 주로 사용되는 CNN(convolutional neural networks), RNN(recurrent neural networks), 변환기(transformers)와 같은 여러 모델을 포함합니다. 이 모델들은 복잡한 의료 이미지를 신속하고 정확하게 분석할 수 있는 능력을 갖추고 있어 안과 질병의 진단 및 관리 향상에 기여합니다. 특히, 데이터 다변화, 알고리즘 투명성 개선, 다중 모드 데이터 활용의 중요성을 강조합니다.

- **Performance Highlights**: AI 기반의 예측 모델은 녹내장 진행 상황을 예측하고 개별 환자에 맞춘 치료 계획을 수립하는 데 유용합니다. 연구에 따르면, DL 알고리즘은 당뇨병성 망막병증과 같은 질환 간의 높은 진단 정확도를 보이며, 환자 치료의 개인화를 통해 치료 결과를 개선할 수 있습니다. 그러나 DL을 안과에 적용할 때 발생할 수 있는 신뢰 문제와 데이터 프라이버시 문제 등 여러 과제가 존재합니다.



New uploads on arXiv(cs.AI)

### Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Though (https://arxiv.org/abs/2501.04682)
- **What's New**: 새로운 프레임워크인 Meta Chain-of-Thought (Meta-CoT)를 제안합니다. 이 프레임워크는 기존의 Chain-of-Thought (CoT)를 확장하여 특정 CoT에 도달하기 위한 기본적인 추론을 명시적으로 모델링합니다. 이 과정에서 최신 모델들이 맥락 내 검색(in-context search)과 일치하는 행동을 보이는 실증적 증거를 제시합니다.

- **Technical Details**: Meta-CoT의 생성 방법으로는 과정 감독(process supervision), 합성 데이터 생성(synthetic data generation), 검색 알고리즘(search algorithms) 등을 탐구하였습니다. 또한, 모델 훈련을 위한 구체적인 파이프라인(pipeline)을 제시하며, 이 파이프라인은 선형화된 검색 흔적(linearized search traces)과 강화 학습(reinforcement learning) 후 훈련을 포함합니다.

- **Performance Highlights**: 이 연구는 LLM(대형 언어 모델)에서 Meta-CoT를 가능하게 하는 이론적 및 실용적 로드맵을 제공합니다. 더 나아가, 스케일링 법칙(scaling laws), 검증자 역할(verifier roles), 새로운 추론 알고리즘 발견 가능성에 대한 여러 가지 열린 연구 질문을 논의합니다.



### MedCoDi-M: A Multi-Prompt Foundation Model for Multimodal Medical Data Generation (https://arxiv.org/abs/2501.04614)
- **What's New**: 이번 연구는 MedCoDi-M이라는 새로운 6.77억 개의 파라미터로 이루어진 모델을 제안합니다. 이 모델은 다중모달(multi-modal) 의학 데이터 생성을 위해 설계되었으며, 대조 학습(contrastive learning)과 대량의 데이터를 활용하여 서로 다른 데이터 모달리티 간의 관계를 캡처하는 공유 잠재 공간(shared latent space)을 구축합니다. MedCoDi-M은 다양한 데이터 세트에서의 유용성을 평가하였습니다.

- **Technical Details**: MedCoDi-M은 기존의 GANs와 Diffusion Models(DMs)에서 발전된 기술을 사용하여 안정적이고 고품질의 합성 데이터를 생성할 수 있는 능력을 갖추고 있습니다. Multi-Prompt 학습 기법을 통해 서로 다른 모달리티의 정보를 융합하여 일관성 있는 데이터를 생성할 수 있도록 설계되었습니다. 저자들은 이를 통해 다양한 유형의 의학적 데이터 조합을 효과적으로 다룰 수 있는 모델의 필요성을 강조하고 있습니다.

- **Performance Highlights**: MedCoDi-M의 효과성을 검증하기 위해 MIMIC-CXR 데이터셋에 대해 다섯 가지 경쟁 모델과의 비교가 진행되었습니다. 이어 전문 방사선의들과 함께 실시한 비주얼 터링 테스트(Visual Turing Test)를 통해 생성된 데이터의 현실성과 임상적 관련성을 평가하였습니다. 결과적으로 MedCoDi-M은 데이터 익명화, 데이터 부족, 불균형 학습과 같은 의료 분야의 주요 과제들을 해결하는 데 도움을 줄 수 있는 가능성을 보였습니다.



### InfiGUIAgent: A Multimodal Generalist GUI Agent with Native Reasoning and Reflection (https://arxiv.org/abs/2501.04575)
Comments:
          14 pages, 7 figures, work in progress

- **What's New**: InfiGUIAgent는 다중 모드 대형 언어 모델(MLLM)을 활용한 새로운 GUI 에이전트이다. 이 에이전트는 2단계 감독 세밀 조정(supervised fine-tuning) 파이프라인을 통해 훈련되어, GUI 이해 및 기초 능력뿐만 아니라 고급 추론 능력을 개선한다. 이 연구는 복잡한 GUI 작업 수행을 위한 본격적인 추론 능력을 추가하여 에이전트의 성능을 향상시키는 데 중점을 두고 있다.

- **Technical Details**: InfiGUIAgent의 개발은 두 가지 주요 단계로 나뉘어 있다. 첫 번째 단계에서는 GUI 이해와 관련된 기초 능력을 강화하기 위해 시각-언어 데이터셋을 수집하고, 두 번째 단계에서는 계층적 추론(hierarchical reasoning)과 기대-반사 추론(expectation-reflection reasoning)과 같은 고급 추론 능력을 통합하는 데이터셋에서 훈련된다. 이를 통해 에이전트는 직관적인 사고를 바탕으로 복잡한 작업을 수행할 수 있다.

- **Performance Highlights**: InfiGUIAgent는 여러 GUI 벤치마크에서 경쟁력 있는 성능을 달성하였다. 기본 능력과 고급 추론 능력을 동시에 향상시키는 두 단계의 감독 세부 조정이 이 모델의 강점을 보여준다. 연구 결과는 GUI 상호작용을 효과적으로 향상시키고 자동화 작업의 범위를 확대할 수 있는 가능성을 시사한다.



### Research on environment perception and behavior prediction of intelligent UAV based on semantic communication (https://arxiv.org/abs/2501.04480)
- **What's New**: 이 논문은 드론 배달 시스템, 가상 세계(virtual worlds), 블록체인(blockchain)의 융합이 물류 및 공급망 관리에 미치는 혁신적인 영향을 다룬다. 이 연구는 드론이 자율적으로 새로운 가상 시나리오에 적응할 수 있도록 하는 강화 학습(reinforcement learning) 접근법을 도입하며, 메타유니버스 서비스(meta-universe services)를 위한 의미적 통신(framework for semantic communication) 구조를 제안한다. 또한 사용자 정보 보안을 위한 경량화된 인증(authentication) 및 키 합의(key agreement) 체계를 설계하였다.

- **Technical Details**: 연구에서는 드론이 빠른 학습 능력을 발휘하고 효율적으로 자원을 할당(resource allocation)하는 방법론을 제시한다. 메타유니버스의 의미 정보를 추출하여 통신 비용(communication cost)을 줄이는 의미적 통신 시스템을 개발하였다. 블록체인 기술을 활용하여 드론과 사용자 간의 안전한 정보 교환을 보장하는 체계를 수립하여 인증 및 키 관리 문제를 해결하였다.

- **Performance Highlights**: 실험 결과, 드론의 적응 성능이 약 35% 개선되었고, 기지국(base stations)의 수가 증가함에 따라 로컬 오프로드 비율(offloading rate)은 90%에 도달할 수 있었다. 의미적 통신 시스템의 성능은 Cross Entropy 기반 모델과 비교하였으며, 다양한 수의 드론이 있는 상황에서도 블록체인 기술을 통해 거래의 처리량(throughput)이 안정적으로 유지됨을 보여주었다.



### Hybrid Artificial Intelligence Strategies for Drone Navigation (https://arxiv.org/abs/2501.04472)
- **What's New**: 이 논문은 드론 내비게이션을 위한 하이브리드 인공지능 전략 개발에 대한 내용을 다루고 있습니다. 주목할 만한 점은 딥러닝 모델과 규칙 기반 엔진을 결합하여 에이전트 상태에 따라 내비게이션 모듈을 구성한 것입니다.

- **Technical Details**: 내비게이션 모듈은 강화 학습(reinforcement learning)으로 훈련된 딥러닝 모델과 전문가 지식을 활용한 규칙 기반 엔진을 포함하고 있습니다. 이 모델은 드론의 관찰 공간을 기반으로 결정을 설명하는 여러 전략을 통합하며, 내비게이션 과정에 인간의 결정을 포함시키기 위한 다양한 메커니즘을 구현합니다.

- **Performance Highlights**: 두 가지 주요 내비게이션 문제를 연구하였으며, 첫 번째 시나리오에서는 90%의 작업 완료율을 달성하고 규칙 기반 엔진 덕분에 충돌을 상당히 줄였습니다. 두 번째 시나리오에서는 강화 학습 모델을 사용하여 모든 목표를 찾는 데 필요한 시간을 20% 단축하는 데 성공했습니다.



### A Digital Shadow for Modeling, Studying and Preventing Urban Crim (https://arxiv.org/abs/2501.04435)
- **What's New**: 이 논문은 도시 범죄 모델링 및 시뮬레이션을 위한 디지털 섀도(Shadow) 플랫폼의 개발과 검증을 제시합니다. 이 플랫폼은 데이터 기반의 에이전트 기반 모델링(agent-based modeling) 및 시뮬레이션 기법을 사용하여 개인과 그들의 환경 간의 동적 상호작용을 캡처합니다. 특히, 말라가(Malaga) 도시의 300,000건 이상의 범죄 신고 데이터를 활용하여 최초로 대규모 도시 영역에 대해 보정(calibrated)된 디지털 섀도를 제공합니다.

- **Technical Details**: 디지털 섀도는 잘 알려진 범죄학 이론(criminological theories)과 법 집행 기관(LEA), 정책 결정자(policy makers) 및 기타 이해 관계자들의 전문 지식을 통합하여 이론적 모델로 변환하였습니다. 이 모델은 실시간 범죄 및 공간(cartographic) 정보, 사회경제적 데이터(socio-economic data)를 조합하여 시민들의 일상 행동을 특성화하는 도시 모델을 형성합니다.

- **Performance Highlights**: 모델의 성능 지표는 예측 경찰 업무(predictive policing)에서 일반적으로 사용되는 메트릭(metrics)에 따라 도시 범죄의 일반적인 패턴과 잘 일치하는 시뮬레이션된 범죄 생성을 제안합니다. 따라서 이 디지털 섀도 플랫폼은 도시 환경 내에서 범죄 행동을 모델링하고 예측하는 유용한 도구가 될 수 있으며, 정책 결정자, 범죄학자, 사회학자 및 법 집행 기관에게 도시 범죄를 연구하고 예방하는 데 유용합니다.



### NSA: Neuro-symbolic ARC Challeng (https://arxiv.org/abs/2501.04424)
- **What's New**: 이 논문은 아브스트랙션 및 추론 데이터셋(ARC)을 해결하기 위한 신경-상징적 접근 방식을 제안합니다. 제안된 방법은 transformer를 사용하여 유망한 검색 방향을 제안하고, 이를 도메인 특화 언어(DSL)와 결합하여 조합적 검색을 수행합니다. 연구 결과, 이 접근 방법은 ARC 평가 세트에서 기존 방법 대비 27% 높은 성능을 기록했습니다.

- **Technical Details**: 제안된 방식은 transformer 모델을 사전 훈련하기 위해 합성적으로 생성된 데이터를 사용하며, 테스트 시간 동안 특정 작업 데이터셋의 세부 과제를 생성하고 모델을 미세 조정합니다. DSL은 올바른 추상화 수준에서 추론하는 데 유용한 인덕티브 바이어스를 제공합니다. 이를 통해 조합적 검색이 제한된 시간 내에 실제 솔루션을 찾을 수 있도록 유도합니다.

- **Performance Highlights**: 기존 대안들과 비교했을 때, 제안된 방식은 ARC 훈련 및 평가 세트에서 효과성을 입증하였습니다. 특히, FETCH의 조합적 접근법은 최신 ML 접근 방식을 초월하는 성과를 보여주었으며, 기존의 DSL 기반 방법들과도 경쟁력을 가지고 있습니다. 이 연구 결과는 오프라인과 온라인 조건 모두에서 이 방법이 얼마나 효과적인지를 잘 나타냅니다.



### User Simulation in the Era of Generative AI: User Modeling, Synthetic Data Generation, and System Evaluation (https://arxiv.org/abs/2501.04410)
- **What's New**: 이번 논문에서는 Generative AI 시대에 등장한 사용자 시뮬레이션(user simulation)의 필요성과 그 응용 가능성에 대해 다루고 있습니다. 사용자가 AI 시스템과 상호작용하는 방식을 모방하는 지능형 에이전트를 생성함으로써, 연구자들은 사용자의 행동을 모델링하고 분석할 수 있으며, 이는 AI 시스템의 안전하고 책임감 있는 발전에 중요한 역할을 합니다. 또한, 사용자 시뮬레이션은 인공지능의 일반 지능(AGI) 개발에도 중대한 영향을 미칠 것으로 기대됩니다.

- **Technical Details**: 사용자 시뮬레이션은 사용자 행동을 기반으로 하여, 사용자의 의사결정 패턴을 모델링하는 과정입니다. 이를 위해 시스템의 특성, 사용자가 수행하는 작업의 종류, 사용자에 대한 정보와 같은 변수들이 고려되어야 합니다. 여러 유형의 사용자 행동을 효과적으로 시뮬레이션하기 위해서 Markov Decision Process(MDP)와 같은 계산적 모델을 활용할 수 있으며, 이로 인해 사용자 시뮬레이터가 다양한 사용자와 조건을 반영할 수 있도록 구성됩니다.

- **Performance Highlights**: 본 논문은 사용자 시뮬레이션이라는 주제를 심도 있게 탐구하며, 관련된 최신 연구 동향과 응용 분야를 정리합니다. 사용자 행동 모델링, 데이터 증대(data augmentation), 그리고 시스템 평가와 관련된 사례들(예: 대화 시스템의 현실적인 대화 생성 등)을 제시하여 사용자 시뮬레이션의 유용성을 강조합니다. 사용자 시뮬레이션은 실제 데이터 확보가 어려운 상황에서도 대량의 합성 데이터를 생성하고, AI 모델의 효율성을 개선하는 데 기여할 수 있는 잠재력을 지니고 있습니다.



### Implementing Systemic Thinking for Automatic Schema Matching: An Agent-Based Modeling Approach (https://arxiv.org/abs/2501.04136)
Comments:
          COGNITIVE 2018 : The Tenth International Conference on Advanced Cognitive Technologies and Applications

- **What's New**: 이번 논문에서는 Automatic Schema Matching (ASM) 문제를 해결하기 위한 여러 가지 접근 방식을 제안합니다. 특히, 복잡성과 불확실성을 다루는 과정에서 생기는 도전 과제를 해결하기 위해 생물에서 영감을 받은 새로운 패러다임을 탐구했습니다. 이를 통해 복합적 적응 시스템(Complex Adaptive System, CAS)으로서 ASM을 접근하고 모델링하는 방식을 설명합니다.

- **Technical Details**: 이 논문에서는 에이전트 기반 모델링 및 시뮬레이션(Agent-Based Modeling and Simulation, ABMS) 접근 방식을 사용하여 자동 스키마 매칭(Automatic Schema Matching)을 모델링했습니다. 이 과정을 통해 Reflex-SMAS라는 스키마 매칭 도구(프로토타입)가 개발되었습니다. 우리의 모델링은 스키마 매칭의 복잡한 과정을 체계적으로 이해하고 관리하는 데 기여합니다.

- **Performance Highlights**: 우리는 두 가지 주요 측면에서 우리의 접근 방식의 실행 가능성을 입증하는 실험을 수행했습니다. 첫째로, 효과성(effectiveness)은 발견된 매칭의 품질을 높이는 데 기여했으며, 둘째로, 효율성(efficiency)은 이 작업에 필요한 노력을 줄이는 것으로 나타났습니다. 이러한 접근 방식은 자동 스키마 매칭 분야에서 중요한 패러다임 전환을 나타냅니다.



### Planarian Neural Networks: Evolutionary Patterns from Basic Bilateria Shaping Modern Artificial Neural Network Architectures (https://arxiv.org/abs/2501.04700)
Comments:
          11 pages, 9 figures

- **What's New**: 이번 연구에서는 인공 신경망 (ANNs)의 이미지 분류 정확도를 증가시키기 위한 새로운 방안을 제시합니다. 생물 신경망의 진화 패턴을 모델로 삼아, 플라나리안 (planarians)의 신경 구조에서 영감을 받은 ANNs를 개발했습니다. 이를 통해 ANNs의 성능 향상이 가능하다는 점을 강조하며, ResNet을 기본 모델로 선택하여 연구를 진행했습니다.

- **Technical Details**: 연구는 플라나리안의 신경 구조가 포함된 새로운 신경망 아키텍처를 바탕으로 하여, CIFAR-10 및 CIFAR-100 데이터셋에서 평가되었습니다. 플라나리안의 두 개의 신경줄과 뇌를 포함하는 독특한 구조는 ANNs의 성능 개선에 중요한 통찰력을 제공합니다. 이 연구는 이러한 생물적 영감이 주는 가능성을 살펴보고자 하였습니다.

- **Performance Highlights**: 결과적으로, 제안된 방법은 기존의 기본 신경망 모델에 비해 이미지 분류 과제에서 더 높은 예측 정확도를 보였습니다. 이는 다양한 응용 분야에서 ANNs의 성능을 향상시킬 수 있는 생물학적으로 영감을 받은 신경망 아키텍처의 중요성을 보여줍니다.



### Grokking at the Edge of Numerical Stability (https://arxiv.org/abs/2501.04697)
- **What's New**: 이번 연구에서는 grokking 현상에 대한 새로운 관점을 제시합니다. 연구자들은 정규화 없이 모델이 학습 중 발생하는 Softmax Collapse (SC)와 같은 수치적 불안정을 방지하기 위한 StableMax와 ⟂Grad와 같은 새로운 기법을 도입했습니다. 이러한 기법들은 grokking을 촉진하고, 기존의 방법들이 왜 효과적인지를 설명하는 데 도움을 줍니다. 이를 통해 delayed generalization의 기저 원인에 대한 인사이트를 제공합니다.

- **Technical Details**: grokking은 과적합(Overfitting) 후 예상치 못한 일반화를 의미합니다. 연구팀은 SC라는 현상이 이러한 일반화가 발생하지 않는 이유 중 하나로 작용한다고 주장합니다. SC는 Softmax 함수의 부동 소수점 오류로 인해 발생하며, 이는 모델의 성능이 정체되거나 오히려 저하되는 결과를 초래합니다. Naïve Loss Minimization (NLM) 방향으로 그래디언트가 정렬되어 가며, 이는 최종적으로 loss를 감소시키지만 SC를 유발합니다.

- **Performance Highlights**: StableMax라는 새로운 활성화 함수는 SC를 방지하며, 정규화 없이도 grokking을 가능하게 합니다. ⟂Grad 옵티마이저는 NLM을 억제하여 grokking 작업에서 빠른 일반화를 촉진합니다. 본 연구의 결과는 정규화가 없는 환경에서도 grokking을 유도할 수 있는 새로운 방법론을 제공하며, 이는 딥러닝에서의 일반화 이해를 한층 더 발전시킵니다.



### EpiCoder: Encompassing Diversity and Complexity in Code Generation (https://arxiv.org/abs/2501.04694)
Comments:
          40 pages, 11 figures

- **What's New**: 이 연구는 코드 LLMs에 최적화를 위한 효과적인 instruction tuning의 중요성을 강조합니다. 기존 방법들이 코드 조각(code snippets)에 의존하여 특정 기능에 국한됨에 따라 데이터의 복잡성과 다양성이 제한되고 있다는 점에서 새로운 피쳐 트리 기반 합성 프레임워크를 제안합니다. 이 프레임워크는 Semantic Relationship을 모델링하여 더 정교하고 다양한 데이터를 생성할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: 제안된 방법의 주요 구성 요소는 피쳐 트리 추출, 진화, 및 코드 생성의 세 가지 단계로 나뉩니다. 피쳐 트리는 raw 데이터로부터 구성되며, 이들 간의 semantic 관계를 포착하기 위해 iterative clustering이 사용됩니다. 이를 통해 코드 생성 시 다양성과 복잡성을 조절할 수 있는 제어 가능성을 가지며, 다양한 수준의 작업을 포함합니다.

- **Performance Highlights**: EpiCoder 시리즈를 통해 다수의 벤치마크에서 기능 및 파일 수준에서 최첨단 성능을 보여주며, 특히 복잡한 저장소 수준의 데이터를 합성하는 능력이 두드러집니다. 433k의 instruction 데이터를 합성하고 EpiCoder-Qwen-7B 모델을 통해 맞춤형 학습을 실시하여 높아진 성능을 실현했습니다. 이를 통해 다양한 프로그래밍 문제를 해결할 수 있는 잠재력을 입증하였습니다.



### Beyond Sight: Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding (https://arxiv.org/abs/2501.04693)
- **What's New**: 본 연구에서는 FuSe라는 새로운 접근 방식을 제안하여 이질적인 센서 데이터를 기반으로 한 generalist 로봇 정책을 미세 조정할 수 있도록 합니다. FuSe는 자연어를 공통적인 크로스 모달 그라운딩으로 활용하여 데이터가 부족한 다양한 센서 모달리티를 결합하는 기법입니다. 이로 인해 시각과 촉각, 음향을 결합한 복잡한 작업을 제로 샷 설정에서 수행할 수 있는 역량이 향상됩니다.

- **Technical Details**: FuSe는 multimodal contrastive loss와 sensory-grounded language generation loss를 조합하여 고차원 시맨틱을 인코딩합니다. 이 미세 조정 과정에서 다양한 센서 모달리티를 통합하여 로봇의 작업 수행 능력을 향상시킵니다. Octo라는 변환 기반 정책과 PaliGemma라는 비전-언어-행동(VLA) 모델을 사용하여 실험을 진행하며, 이를 통해 수집된 27K 로봇 궤적 데이터셋을 활용했습니다.

- **Performance Highlights**: FuSe는 모든 기준선에 비해 성공률을 20% 이상 향상시키는 것으로 나타났습니다. 다양한 generalist 정책에 대해 동일한 레시피가 적용되며, 이는 기존의 시각 데이터로만 미세 조정된 정책보다 우수한 성능을 보입니다. 이 연구는 로봇 조작 작업에서 복합적이고 창의적인 작업 지시를 수행할 수 있는 가능성을 제시합니다.



### URSA: Understanding and Verifying Chain-of-thought Reasoning in Multimodal Mathematics (https://arxiv.org/abs/2501.04686)
Comments:
          27 pages, 10 tables, 17 figures. The training data has been released. The code and model are currently undergoing internal review. They will be made available soon. Project url: this https URL

- **What's New**: 본 논문에서는 Chain-of-thought (CoT) 추론을 다룬 새로운 접근법을 제안합니다. 특히, 다중 모드 수학적 추론에서 CoT 훈련 데이터의 부족 문제를 해결하기 위해 3개의 모듈 합성 전략을 도입하였습니다. 이를 통해 MMathCoT-1M이라는 고품질 CoT 추론 지침 데이터셋을 생성하고, URSA-7B 모델의 성능을 여러 벤치마크에서 검증했습니다.

- **Technical Details**: 제안된 시스템은 CoT 증류(Cot Distillation), 궤적 형식 재작성(Trajectory-format rewriting), 및 형식 통합(Format unification)을 포함한 세 가지 모듈로 구성됩니다. 이 과정을 통해 고품질 CoT 추론 데이터셋이 생성되며, URSA-7B 모델은 DualMath-1.1M이라는 데이터 합성 전략을 통해 향상된 성능을 나타냅니다. 모델은 특히 다중 모드 정보 처리 과정에서 발생할 수 있는 오류를 국지화할 수 있는 새로운 방법론을 지니고 있습니다.

- **Performance Highlights**: URSA-7B 모델은 MathVista, MathVerse, WE-MATH 등 여러 다중 모드 수학 벤치마크에서 SOTA 성능을 달성하였습니다. 또한, URSA-RM-7B 모델은 URSA-7B의 검증기로 작동하여 테스트 시간 동안 더욱 향상된 성능을 보여주고 있습니다. 궁극적으로, 이 연구는 다중 모드 수학적 추론에서 모델의 성능 한계를 효과적으로 높이는 데 기여하고 있습니다.



### Enhancing Financial VQA in Vision Language Models using Intermediate Structured Representations (https://arxiv.org/abs/2501.04675)
- **What's New**: 이 연구는 50,000개의 막대 차트에 대해 고유한 구조적 특성을 활용하여 차트 이미지를 선형화된 테이블로 변환하는 DEPLOT(모드 전환 모듈)의 미세 조정을 조사합니다. 미세 조정된 DEPLOT 모델은 카테고리별 매핑 정확도를 측정하는 Relative Mapping Similarity(RMS)와 수치적 해석 정확도를 평가하는 Relative Number Set Similarity(RNSS)를 통해 기본 모델과 비교 평가됩니다. 또한, 100개의 차트 이미지와 질문-응답 세트를 추가하여 대규모 언어 모델(LLMs)의 추론 능력을 탐구합니다.

- **Technical Details**: DEPLOT은 시각 차트 데이터를 구조화된 데이터 테이블로 매핑하기 위한 모드 전환 모듈로, 다양한 차트 유형에서 훈련되지만 도메인별 데이터 세트를 사용하여 미세 조정할 수 있습니다. 본 논문에서는 RNSS와 RMS 두 가지 주요 지표를 통해 모델의 정량적 및 범주적 해석 능력을 평가하며, 차트 구조를 추적할 수 있는 능력을 강조합니다. 이러한 새로운 접근법은 DEPLOT의 성능을 높이고, 보다 신뢰할 수 있는 데이터 시각화 모델 개발을 위한 기초를 제공합니다.

- **Performance Highlights**: 미세 조정된 DEPLOT 모델을 활용한 실험 결과, 높은 품질의 구조화된 데이터와 함께 제공된 경우 LLM의 추론 능력이 크게 향상됨을 보여줍니다. 특히 Qwen2-VL-7B와 같은 소형 모델이 고급 모델인 GPT-4o보다 더 나은 성능을 발휘하여 차트 데이터 해석의 정확성을 높였습니다. 이 연구는 자동 차트 해석 및 추론 향상을 위한 모드 전환 통합의 혁신적 잠재력을 강조합니다.



### DRIVINGVQA: Analyzing Visual Chain-of-Thought Reasoning of Vision Language Models in Real-World Scenarios with Driving Theory Tests (https://arxiv.org/abs/2501.04671)
- **What's New**: 최근의 대형 비전-언어 모델(LVLMs)은 언어 모델에 비주얼 이해를 통합하여 다중 모달(Modal) 추론을 가능하게 합니다. 그러나 텍스트와 비주얼 데이터 사이의 모달 간 격차로 인해, 이 모델들은 텍스트 우선 의존성(over-reliance on text priors), 환각(hallucinations) 및 복잡한 비주얼 추론에 대한 제한된 능력과 같은 도전에 직면합니다. 이를 해결하기 위해, 우리는 DrivingVQA라는 새로운 벤치마크를 제안하여 복잡한 실제 상황에서 비주얼 체인 오프 씽킹(visua chain-of-thought reasoning)을 평가합니다.

- **Technical Details**: DrivingVQA는 운전 이론 테스트에서 파생된 데이터셋으로, 총 3,931개의 전문가 제작 다중 선택 문제(multiple-choice problems)를 포함하고 있습니다. 각 문제는 추론 과정과 관련된 엔티티(entities)에 기반한 교차 설명(interleaved explanations)을 제공합니다. 우리는 이 데이터셋을 활용하여 LVLMs의 복잡한 비주얼 시나리오에 대한 추론 능력을 광범위하게 연구하였습니다. 실험 결과, 오픈 소스와 상용 LVLM들이 제로샷(zeroshot) 설정 하에서 비주얼 체인 오프 씽킹에 어려움을 겪고 있음을 발견했습니다.

- **Performance Highlights**: 특히, 이미지 토큰(image tokens)의 자르는 지역(cropped regions)에 기반한 엔티티를 활용할 때, 비주얼 추론이 최대 7% 향상되는 성과를 보였습니다. 이는 현재 모델들이 지역의 관심 영역(localization of regions of interests)에서 정보를 효과적으로 활용하지 못하고 있다는 것을 보여줍니다. 따라서, LVLM의 비주얼 추론 개선을 위한 관련 엔티티 활용 전략을 탐구하는 것이 중요합니다.



### Assessing Language Comprehension in Large Language Models Using Construction Grammar (https://arxiv.org/abs/2501.04661)
- **What's New**: 이번 연구는 Construction Grammar (CxG)를 활용하여 대규모 언어 모델(LLMs)의 자연어 이해(NLU)를 체계적으로 평가하는 새로운 방법론을 제시합니다. 평가 작업은 CxG의 8가지 고유한 Cxn을 바탕으로 하여 LLM의 언어 이해를 인간의 이해와 비교합니다. 특히, 일반적으로 훈련 데이터에 나타나지 않는 문장 예제를 포함하여 LLM의 언어 이해의 한계를 강조합니다.

- **Technical Details**: CxG는 형태소, 단어, 관용구 및 언어의 도식적 구조가 형태-의미 쌍, 즉 construction (Cxn)으로 표현될 수 있다는 이론적 기초를 제공합니다. 본 연구는 이러한 Cxn의 의미를 이해해야하는 자연어 추론(NLI) 작업을 통해 LLM의 이해 능력을 평가하며, 이는 LLM의 진정한 언어 이해를 측정하는 데 필수적입니다. LLM의 성능은 일반적으로 훈련 데이터에서 등장하지 않는 예제에 대한 이해를 요구하는 도전 과제를 통해 평가됩니다.

- **Performance Highlights**: 실험 결과, 최신 모델인 GPT-4o를 포함한 LLM은 Cxn이 전달하는 추상적인 의미를 이해하는 데 어려움을 겪는 것으로 나타났습니다. LLM은 constructual 정보에 대한 일부 지식은 보유하고 있지만, 실제로는 통계적으로 예상되는 예제와는 다른 테스트 문장에서 어려움을 보여줍니다. 이러한 결과는 LLM의 의미적 한계를 강조하며, 진정한 언어 이해의 평가 측면에서 중요합니다.



### Knowledge Retrieval Based on Generative AI (https://arxiv.org/abs/2501.04635)
Comments:
          8 pages, 13 figures, 1 table

- **What's New**: 이번 연구에서는 중국어 위키백과와 Lawbank를 정보 검색(Source)으로 사용하여 Retrieval-Augmented Generation (RAG) 기반 질문-답변 시스템을 개발했습니다. BGE-M3를 활용한 밀집 벡터 검색을 통해 관련 검색 결과를 얻고, BGE-reranker로 이 결과를 쿼리 관련성에 따라 재정렬하여 성능을 향상시킵니다. 이를 통해 생성형 AI에 기반한 지식 검색 시스템을 구축하게 되었습니다.

- **Technical Details**: 연구에서는 양자된 쿼리를 입력받아 다국어, 다기능, 다중 세분화 지원이 가능한 BGE-M3 모델을 사용합니다. 이 모델은 100개 이상의 언어에 대한 검색 기능을 지원하며, 8,192개의 토큰 길이까지 처리할 수 있습니다. 또한 질문 답변에 필요한 정확한 결과를 위해 BGE-reranker를 도입하여 검색 결과의 관련성을 높였습니다.

- **Performance Highlights**: 자동 성능 평가에서는 모델의 자가 생성된 레이블과 실제 정답을 비교하여 정확도를 측정하고, 주관적 평가에서는 재정렬된 정보를 바탕으로 노지연 참여자들이 답변한 결과를 통해 시스템의 안정성을 검증했습니다. 연구 결과, 로컬 운영으로 데이터 프라이버시를 개선하고 상업적 서비스 의존도를 낮추면서 데이터 보안을 강화했습니다.



### Federated-Continual Dynamic Segmentation of Histopathology guided by Barlow Continuity (https://arxiv.org/abs/2501.04588)
- **What's New**: 이 연구에서는 Federated Learning과 Continual Learning에서 발생하는 Client Drift와 Catastrophic Forgetting 문제를 동시에 해결할 수 있는 방법을 제안했습니다. 제안된 Dynamic Barlow Continuity(DynBC) 방법은 공공 참조 데이터셋에서 클라이언트 업데이트를 평가하여 훈련 과정을 조정합니다. 이를 통해 시공간 불변성을 가진 모델을 구축하고, histopathology 데이터셋에서 성능을 크게 향상시킬 수 있었습니다.

- **Technical Details**: DynBC는 Barlow Twins를 기반으로 한 평가 방법으로, 다양한 데이터를 기반으로 두 모델 상태의 변화를 비교하여 spatio-temporal continuity를 평가합니다. 이 방법은 기존 중앙 집중식 훈련에서 성공적으로 사용되었으며, 이제는 Federated Learning 및 Continual Learning 환경에서도 적용됩니다. 평가 프로세스에서 프라이버시를 유지하기 위해 별도의 공공 데이터셋에서 샘플링한 데이터를 사용하여 모델 반응을 측정합니다.

- **Performance Highlights**: BCSS와 Semicol 데이터셋에서 실험을 통해 DynBC 방법이 Client Drift를 15.8%에서 71.6%로, Catastrophic Forgetting을 42.5%에서 62.8%로 향상시키는 것을 입증했습니다. 이러한 결과는 제안한 방법이 AI 지원 histopathology 분야에서 모델 성능을 개선할 수 있음을 알립니다. 최종적으로, 이 방법은 향후 다양한 의료 AI 시스템에 적용될 가능성을 제시합니다.



### A 65 nm Bayesian Neural Network Accelerator with 360 fJ/Sample In-Word GRNG for AI Uncertainty Estimation (https://arxiv.org/abs/2501.04577)
Comments:
          7 pages, 12 figures

- **What's New**: 이번 논문에서는 Bayesian Neural Networks (BNNs)의 랜덤 넘버 생성(Generate Random Number, RNG) 오버헤드를 줄이고 메모리 내 연산(compute-in-memory, CIM)을 통해 BNN 성능을 향상시키기 위한 ASIC 칩을 제안합니다. 이 칩은 360 fJ/Sample의 가우시안 RNG를 SRAM 메모리 단어에 직접 통합하여 운영됩니다. 이러한 혁신적인 접근방법은 RNG 관련 오버헤드를 감소시키고, BNN의 동시 병렬 처리(compute-in-memory)를 가능하게 합니다. 이를 통해 AI 불확실성 추정이 엣지 컴퓨테이션(edge computation)에 적용될 수 있습니다.

- **Technical Details**: Bayesian Neural Networks (BNNs)는 확률적(weight의 posterior distribution으로 대체) 예측을 제공하는 데이터 기반 심층 학습 시스템을 지원합니다. 이 논문에서는 메모리 내 직접 가우시안 무작위 수 생성(Gaussian RNG) 통합을 통한 연산 최적화를 논의하며, BNN으로의 연산 효율을 높이는 방법을 설명합니다. 또한, 새로운 하드웨어 설계를 통해 기존 디지털 RNG의 한계를 극복하고, 메모리 작업을 줄이면서도 BNN의 추론 성능을 향상시키는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 프로토타입 칩은 5.12 GSa/s의 RNG 처리량과 102 GOp/s의 신경망 처리량을 달성하면서, 면적은 0.45 mm2에 불과합니다. 이러한 성능은 BNN을 엣지 장비(Edge devices)에서의 구현을 가능하게 해 주며, 에너지 효율성 또한 뛰어납니다. 기기 집적화 및 효율적인 메모리 접근 방식을 통해, BNN의 세부적인 인퍼런스 작업에서 발생할 수 있는 에너지 소모를 크게 줄일 수 있습니다.



### Supervision-free Vision-Language Alignmen (https://arxiv.org/abs/2501.04568)
Comments:
          Preprint

- **What's New**: 이 논문에서는 SVP(Supervision-free Visual Projection)라는 새로운 프레임워크를 소개합니다. VLMs(비전-언어 모델)의 성능 향상에 초점을 맞추며, 이는 비급식 데이터나 선호 주석 없이도 가능하다는 점에서 이전 연구들과 차별화됩니다. SVP는 자기 캡셔닝(self-captioning)과 사전 훈련된 그라운딩 모델(pre-trained grounding model)을 활용하여 VLM의 잠재 정보를 이끌어내는 피드백 메커니즘을 이용합니다.

- **Technical Details**: SVP는 크게 이미지-텍스트 쌍의 수집이 필요하지 않은 점이 특징이며, 이를 통해 비전-언어 정합성(vision-language alignment)을 개선합니다. 연구에서는 캡셔닝(captioning), 참조(referring), 비주얼 질문 응답(visual question answering), 멀티태스킹(multitasking), 할루시네이션 제어(hallucination control), 객체 회상(object recall) 등 여섯 가지 주요 영역에서 평가가 이루어졌습니다.

- **Performance Highlights**: SVP를 적용한 결과, 캡셔닝 작업에서 평균 14%의 성능 향상, 객체 회상에서 최대 12% 증가, 할루시네이션 비율 대폭 감소 등 주요 성과가 보고되었습니다. 특히, SVP를 활용한 작은 VLM이 원래 크기가 다섯 배 큰 모델과 비교할 만한 수준으로 할루시네이션을 줄인 점이 주목할 만합니다.



### Cyber-Physical Steganography in Robotic Motion Contro (https://arxiv.org/abs/2501.04541)
- **What's New**: 이 연구에서는 로봇의 움직임을 통해 비밀 메시지를 은닉하는 새로운 스테가노그래피(steganography) 방법론을 제안합니다. 로봇이 환경 변화에 민감하게 반응함을 활용하여 메시지를 그 움직임의 경로에 암호화하고, 이를 해독하는 메커니즘을 탐구합니다. 이는 기존의 시각, 청각, 언어적 미디어를 넘어 로봇 공학의 경계에서 비밀 통신 채널을 확장하는 시도로 볼 수 있습니다.

- **Technical Details**: 제안된 방법론은 로봇의 움직임을 감시하는 수신자와 비밀 메시지를 삽입하는 송신자 간의 상호작용을 기반으로 합니다. 로봇의 행동은 사전 훈련된 제어 모델에 의해 결정되며, 송신자는 로봇의 움직임 궤적을 약간 변화시킴으로써 메시지를 인코딩합니다. 이 연구는 로봇의 무결성(integrity)과 최소한의 움직임 편차를 보장하는 조건을 충족해야 하며, 이는 스테가노그래피의 근본 원칙입니다.

- **Performance Highlights**: 실험은 여러 조작 작업을 포함하는 시뮬레이션 환경에서 수행되었습니다. 제안된 방법론은 시스템의 수용 능력, 비밀성 및 효율성을 평가하는 데 중점을 둡니다. 결과는 로봇의 움직임이 잘 제어된 비밀 통신의 매개체 역할을 할 수 있음을 보여주며, 기존의 방법들과 비교해 더 높은 유연성을 제공합니다.



### Towards a Problem-Oriented Domain Adaptation Framework for Machine Learning (https://arxiv.org/abs/2501.04528)
- **What's New**: 이번 논문에서는 도메인 적응(Domain Adaptation) 문제를 해결하기 위해 문제 지향적인 프레임워크를 개발하였습니다. 이 프레임워크는 다양한 도메인 적응 시나리오를 구분하고, 각 시나리오에 대한 해결책을 제시하며, 문제의 특성을 파악하는 가이드를 제공합니다. 평가를 통해 프레임워크가 도메인 적응 문제를 효과적으로 설명할 수 있는 능력을 갖추고 있음을 입증하였습니다.

- **Technical Details**: 기계 학습(Machine Learning)에서 도메인 적응은 서로 다른 특성 분포를 가진 두 문제 간의 간극을 메우는 접근 방식을 의미합니다. 이를 통해 소스 도메인(source domain)에서 타겟 도메인(target domain)으로의 지식 이관이 가능합니다. 이 연구는 디자인 과학 연구(Design Science Research) 패러다임을 따르며, 이론적 엄밀성뿐만 아니라 실용적 함의도 강조합니다.

- **Performance Highlights**: 제시된 프레임워크는 인공지능 연구자 및 실무자가 도메인 적응 사용 시 방향성을 제공하며, 다양한 데이터 세트를 통해 평가되었습니다. 프레임워크의 개발 과정에서, 인공 데이터 및 실제 데이터셋에 대한 실험을 통해 성공적인 성과를 나타냈습니다. 이러한 평가 결과는 도메인 간 전이 효과(positive transfer)를 보장할 수 있는 접근법이 존재함을 설명합니다.



### CGP-Tuning: Structure-Aware Soft Prompt Tuning for Code Vulnerability Detection (https://arxiv.org/abs/2501.04510)
Comments:
          14 pages, 5 figures

- **What's New**: 이 논문에서는 소프트 프롬프트 튜닝의 효율성을 바탕으로 코드 기반의 구조 인식 소프트 프롬프트 튜닝 방법, CGP-Tuning을 제안합니다. 이 방법은 코드 그래프의 풍부한 의미 정보를 포착하기 위해 혁신적인 타입 인식 임베딩을 도입합니다. CGP-Tuning은 그래프-텍스트 상호 작용을 포함하면서도 선형 계산 비용을 달성할 수 있는 새로운 효율적인 크로스 모달 정렬 모듈을 사용하여 소프트 프롬프트 튜닝을 진행합니다.

- **Technical Details**: CGP-Tuning은 코드 속성 그래프를 사용하여 소스 코드의 그래프 기반 구조 정보를 표현하며, 두 개의 학습 가능한 타입 인식 임베딩을 도입하여 노드 간의 의미 정보를 캡처합니다. 이 방법론은 그래프-텍스트 상호 작용을 적절하고 효율적으로 처리하기 위해 특별히 설계된 다양한 모듈을 포함합니다. CGP-Tuning의 구조는 복잡한 코드의 구조와 의미를 효과적으로 분석할 수 있도록 돕고 있습니다.

- **Performance Highlights**: DiverseVul 데이터셋과 최신 오픈 소스 코드 LLM인 CodeLlama 및 CodeGemma에서 평가된 결과, CGP-Tuning은 평균적으로 최신의 최고 방법론 대비 3.5% 향상된 정확도를 보였습니다. 이는 장기 소스 코드의 취약성 탐지 능력도 간직하면서 이루어낸 성과입니다. CGP-Tuning은 기존 접근 방식의 한계를 극복함으로써 소프트웨어 보안 분야에 중요한 기여를 할 것으로 기대됩니다.



### The Role of Machine Learning in Congenital Heart Disease Diagnosis: Datasets, Algorithms, and Insights (https://arxiv.org/abs/2501.04493)
- **What's New**: 이 논문은 선천성 심장병(Congeital Heart Disease, CHD) 인식에 대한 머신러닝(Machine Learning) 기반의 체계적인 문헌 리뷰(Systematic Literature Review, SLR)를 제공합니다. 2018년부터 2024년까지의 주요 저널에 게재된 432개의 참고문헌을 메타 분석하고, 74개의 주요 연구 결과를 심층 분석하였습니다. 또한, 본 연구는 머신러닝 전문가들이 CHD 인식에 사용한 데이터셋을 정리하여 주목받는 데이터를 실제로 활용하는 방법을 제시합니다.

- **Technical Details**: 이 논문에서는 머신러닝 알고리즘 및 데이터베이스를 사용하는 CHD 인식에서의 최근 발전을 포괄적으로 다룹니다. SLR 방법론을 통해 연구 팀은 다양한 진단 방법, 사용된 데이터셋, 알고리즘 적용, 그리고 솔루션을 분석하였습니다. 주요 머신러닝(Machine Learning) 및 딥러닝(Deep Learning) 알고리즘에 대한 강점과 약점을 조명하며, 이들 기법의 실질적인 응용 방법을 제공합니다.

- **Performance Highlights**: CHD-ML 분야는 여전히 충분히 탐구되지 않은 영역으로, 연구자와 전문가들에게 중요한 참고자료로 사용될 수 있습니다. 이 리뷰는 CHD 인식을 위한 최신 머신러닝 및 딥러닝 접근 방식에 대한 포괄적인 분석을 제공하며, CHD-ML 시스템 개발을 위한 주요 요소를 확인합니다. 연구 결과는 의료 실무자와 연구자 간의 협력을 통해 이 중요한 분야의 발전을 촉진하는 데 기여할 것으로 예상됩니다.



### Integrating remote sensing data assimilation, deep learning and large language model for interactive wheat breeding yield prediction (https://arxiv.org/abs/2501.04487)
- **What's New**: 이 연구는 농작물 수확량 예측을 위한 하이브리드 방법과 도구를 도입합니다. 이 도구는 품종 육종가가 대화형으로 wheat yield (밀 수확량)를 예측할 수 있도록 설계되었습니다. 새로운 데이터 동화 알고리즘을 WOFOST 모델에 통합하여 처리하며, 사용자 친화적이고 지속적인 데이터 업데이트를 지원합니다.

- **Technical Details**: 연구에서는 먼저 leaf area index (엽면적지수)를 WOFOST 모델에 동화시키는 데이터 동화 알고리즘을 활용합니다. 그런 다음, 이 동화 과정에서 선택된 출력과 remote sensing (원거리 감지) 결과를 활용해 time-series temporal fusion transformer model (시간 시계열 융합 변환기 모델)을 구동하여 wheat yield를 예측합니다. 이 모든 과정은 대화형 웹 도구를 통해 이루어집니다.

- **Performance Highlights**: 개발된 도구는 다원 소스 데이터를 통합하여 육종 의사 결정을 지원하는 기능을 갖추고 있습니다. 이 혁신적인 접근 방식은 품종 육종 과정에서의 고수확량 자재를 더 빨리 식별할 수 있도록 하고, 육종 효율성을 향상시키며, 보다 과학적이고 스마트한 육종 결정을 가능하게 합니다.



### A novel Facial Recognition technique with Focusing on Masked Faces (https://arxiv.org/abs/2501.04444)
- **What's New**: 이 연구는 마스크 착용 유무에 관계없이 동일한 얼굴을 인식하는 기능이 보안 및 공공 안전 분야에서 얼마나 중요한지를 강조합니다. 전통적인 얼굴 인식 시스템은 마스크로 얼굴이 가려진 경우에 심각한 정확도 문제를 겪으므로, 마스크 착용 상황에서도 신뢰할 수 있는 인식 방법을 개발하는 것이 필요합니다. 이를 위해 Masked-Unmasked Face Matching Model (MUFM)을 제안하며, 이는 새로운 접근 방식을 제공합니다.

- **Technical Details**: MUFM 모델은 Visual Geometry Group (VGG16) 모델을 활용하여 중요한 얼굴 특징을 추출하고, K-Nearest Neighbors (K-NN) 알고리즘을 이용하여 이러한 특징을 분류합니다. 또한 cosine similarity 메트릭을 사용하여 동일한 개인의 마스크 착용과 비착용 얼굴을 비교합니다. 이러한 기술적 접근은 마스크 착용 유무에 따른 동일 개인 인식이라는 과제를 해결하는데 기여합니다.

- **Performance Highlights**: 이 연구는 마스크에도 불구하고 개인을 효과적으로 식별할 수 있는 능력을 입증하며, 이는 기존 시스템의 주요 한계를 극복합니다. 연구에 사용된 이미지 데이터셋은 세 가지 서로 다른 출처에서 수집되었으며, 이는 연구의 신뢰성을 높이는 데 기여합니다. 이러한 데이터는 마스크 착용 및 비착용 상태의 동일한 얼굴을 포함하고 있어, 연구의 광범위한 가능성을 제시합니다.



### Effect of Information Technology on Job Creation to Support Economic: Case Studies of Graduates in Universities (2023-2024) of the KRG of Iraq (https://arxiv.org/abs/2501.04438)
- **What's New**: 이번 연구는 쿠르디스탄 지역 대학교 졸업생들이 정보 기술(IT)을 활용하여 고용을 창출하고 경제 회복에 기여하는 방법을 탐구합니다. IT가 졸업생들의 취업에 미치는 영향을 분석하며, 이 결과는 경제와 고용 문제 해결에 중요한 기여를 할 수 있습니다.

- **Technical Details**: 연구는 기술적 변수들을 이해하기 위해 기술적 연구 방법론과 양적 접근방식을 사용했습니다. 샘플 크기는 총 314명으로, 판단 샘플링 절차를 통해 선정되었고, 데이터 수집을 위하여 설문지를 작성하였습니다. 수집된 데이터는 SPSS 통계 소프트웨어(version 22)와 Excel 2010을 통해 수정, 컴파일, 테이블화 하였습니다.

- **Performance Highlights**: 연구 결과, 정보 기술은 매우 혁신적이며 장래성이 밝고, 모든 사람의 삶을 쉽게 만드는 것으로 나타났습니다. 정보 기술의 이해는 졸업생들이 적합한 경로를 찾는 데 큰 도움이 되었고, IT에 대한 기술과 자격 증명은 취업을 원하는 이들에게 큰 이점을 제공합니다. 결과적으로, 정보 기술은 국가 경제를 적극적으로 발전시키는 데 기여하고 있습니다.



### Integrating LLMs with ITS: Recent Advances, Potentials, Challenges, and Future Directions (https://arxiv.org/abs/2501.04437)
Comments:
          Accepted for publication in IEEE Transactions on Intelligent Transportation Systems

- **What's New**: 이 논문은 Intelligent Transportation Systems (ITS)의 최적화에서 Large Language Models (LLMs)의 혁신적 잠재력을 종합적으로 검토합니다. 저자들은 ITS의 구성 요소와 운영 원리를 개괄하고, 다양한 LLM 기술(GPT, T5, CTRL, BERT)의 이론적 배경을 설명합니다. 또한, LLMs가 ITS 내에서의 여러 응용 프로그램(교통 예측, 자율 주행 등)에 미치는 영향을 분석하여, 이러한 첨단 모델들이 교통 관리 및 안전을 크게 향상시킬 수 있음을 강조합니다.

- **Technical Details**: ITS는 교통 네트워크의 효율성, 안전성, 지속 가능성을 향상시키기 위한 변혁적 접근 방식을 나타냅니다. 최근 딥러닝 기술의 도입은 예측 및 적응형 교통 관리를 개선합니다. LLMs는 사용자와의 상호작용을 향상시키고 실제 교통 데이터의 분석을 통해 더 나은 의사결정을 지원할 수 있는 능력을 갖추고 있으며, 이를 통해 교통 흐름 예측 및 비상 대응 시스템의 효율성을 높일 수 있습니다.

- **Performance Highlights**: LLMs는 ITS 분야에서 사용자 상호작용과 커뮤니케이션을 혁신하고 있습니다. 실시간 교통 데이터를 바탕으로 교통 혼잡을 예측하고, 최적의 교통 신호를 조정하여 안전성을 유지하면서 혼잡을 줄입니다. 또한, 대중교통의 예측 유지보수를 가능하게 하여 서비스 중단을 예방하고, 자율주행 차량 간의 데이터 공유를 통해 보다 안전하고 효율적인 교통 관리를 가능하게 합니다.



### Federated Fine-Tuning of LLMs: Framework Comparison and Research Directions (https://arxiv.org/abs/2501.04436)
- **What's New**: 이 논문은 분산된 개인 데이터셋을 사용하여 프리트레인된 대형 언어 모델(LLMs)의 파인튜닝을 위한 연합 학습(Federated Learning, FL)의 새로운 접근 방식을 제시합니다. 세 가지 고급 연합 LLM 프레임워크(FedLLM)인 FedLLMs, KD-FedLLMs, Split-FedLLMs를 비교하며, 각각의 모델 업데이트 방법과 지식 전달 메커니즘의 차별성을 강조합니다. 이러한 분석을 통해 FL 환경에서의 LLM 파인튜닝의 효율성을 극대화할 수 있는 기회를 제시하고 있습니다.

- **Technical Details**: 이 논문에서는 PEFT(Parameters Efficient Fine-Tuning) 기법을 적용하여 자원 제약이 있는 연합 시나리오에서 LLM을 파인튜닝하는 세 가지 프레임워크를 설명합니다. 첫 번째 프레임워크, FedLLMs는 클라이언트가 모델 업데이트를 서버에 직접 제출하는 방식을 사용합니다. 두 번째 프레임워크, KD-FedLLMs는 지식 증류(Knowledge Distillation, KD)를 활용하여 클라이언트와 서버 간의 지식 공유를 용이하게 합니다. 마지막으로 Split-FedLLMs는 모델을 클라이언트와 서버 간에 나누어 연산 부하를 조절하는 접근 방식을 채택하고 있습니다.

- **Performance Highlights**: 이 연구는 각 프레임워크의 모델 정확도, 통신 비용, 클라이언트 측 계산 부하와 같은 주요 성능 지표를 기준으로 평가하였습니다. 이를 통해 서로 다른 방식의 파인튜닝 방법 간의 무역 가지를 이해할 수 있으며, 실제 연합 학습 시나리오에서 각 프레임워크의 적합성을 파악하는 데 기여합니다. 또한, 실용적인 사용 사례를 통해 다양한 설정에서 이들 프레임워크의 성능을 비교하고 실제 적용 가능성을 시연하고 있습니다.



### Dual-Force: Enhanced Offline Diversity Maximization under Imitation Constraints (https://arxiv.org/abs/2501.04426)
- **What's New**: 이 논문에서는 오프라인 환경에서 다양성을 극대화하기 위한 새로운 알고리즘인 Dual-Force를 소개합니다. 이 알고리즘은 Van der Waals (VdW) 힘과 Successor features를 기반으로 하여 과거에 사용된 skill discriminator를 학습할 필요를 없앱니다. 또한, Functional Reward Encoding (FRE)을 활용하여 비국소성 보상을 효과적으로 처리할 수 있습니다.

- **Technical Details**: Dual-Force 알고리즘은 강화 학습 (Reinforcement Learning)에서의 Fenchel 이중성 이론을 적용하여 오프라인 데이터를 활용합니다. 이 방법은 VdW 힘을 사용하여 다양성을 확장하며 skill discriminator를 학습할 필요성을 제거합니다. 각 비국소성 보상에 대해 관련된 skill을 FRE 잠재 임베딩을 통해 쉽게 회상할 수 있습니다.

- **Performance Highlights**: 우리는 Dual-Force의 유효성을 12-DoF의 사족 보행 로봇 Solo12에서 수집된 두 가지 오프라인 데이터셋을 통해 입증합니다. 이 알고리즘은 다양한 행동을 효율적이고 견고하게 재현하며, 학습 데이터를 기반으로 목표 전문가 상태 점유를 모방합니다. 이로 인해 학습된 기술 세트가 크게 확장됩니다.



### On Computational Limits and Provably Efficient Criteria of Visual Autoregressive Models: A Fine-Grained Complexity Analysis (https://arxiv.org/abs/2501.04377)
- **What's New**: 최근 Visual Autoregressive (VAR) 모델이 이미지 생성 분야에서 혁신적인 발전을 가져왔습니다. 이 모델은 coarse-to-fine "next-scale prediction" 접근 방식을 통해 확장 가능한 방식으로 이미지 생성을 가능하게 합니다. 그러나 VAR 모델의 현재 최첨단 알고리즘은 O(n^4)의 시간 복잡도를 가지며, 이는 계산 효율성이 낮습니다. 따라서 본 연구는 VAR 모델의 계산 한계와 효율성을 분석하여 개선 방안을 제시합니다.

- **Technical Details**: VAR 모델의 효율성을 갖추기 위해서는 입력 행렬의 노름이 특정 활용 범위 아래에 있어야 하며, 이 기준을 초과할 경우 진정한 서브-쿼틱(sub-quartic) 시간 알고리즘을 설계하는 것이 불가능한 것으로 확인했습니다. 본 연구에서는 Strong Exponential Time Hypothesis (SETH)를 바탕으로 VAR 모델의 계산 성능을 평가하는 새로운 기준을 설정하였습니다. 이를 통해 계산 비용을 절감할 수 있는 경량화된 구조도 제안합니다.

- **Performance Highlights**: 본 연구의 기여는 VAR 모델의 계산 시간을 O(n^4)보다 더 빠르게 수행할 수 있는 조건을 제시한 것입니다. 입력 행렬의 요소가 특정 임계값을 초과하지 않을 경우 효율적인 알고리즘을 통해 거의 저차 제곱 시간 복잡도 O(n^{2+o(1)})로 VAR 모델을 근사할 수 있음을 보여줍니다. 이러한 발견은 VAR 모델의 이론적 발전을 위한 기초가 되며, 향후 스케일러블한 이미지 생성기를 발전시킬 수 있는 길을 열어줄 것입니다.



### DispFormer: Pretrained Transformer for Flexible Dispersion Curve Inversion from Global Synthesis to Regional Applications (https://arxiv.org/abs/2501.04366)
Comments:
          11 pages, 11 figures, related codes and data are available at this https URL

- **What's New**: 이 연구는 Rayleigh 파의 위상 및 그룹 분산 곡선에서 지하의 전단파 속도($v_s$) 프로필을 추정하기 위한 transformer 기반의 신경망 모델인 DispFormer를 제안합니다. 기존의 방법들이 데이터의 길이에 따라 조정해야 하는 복잡함을 갖고 있었던 반면, DispFormer는 각 주기(period)에서 분산 데이터를 독립적으로 처리하여 변화하는 길이에 대한 유연성을 제공합니다. Pref-training과 zero-shot 및 few-shot 테스트를 통해 다양한 데이터 세트에서 뛰어난 성능을 입증하였습니다.

- **Technical Details**: DispFormer는 선형 계층(linear layers)과 포지션 임베딩(position embeddings)을 사용하여 분산 데이터를 인코딩하고, 여러 transformer 블록을 통해 주기와 관련된 특징을 추출하여 최종적으로 1-D 속도 프로필로 변환합니다. 이 모델은 글로벌 합성 데이터 세트에서 사전 훈련(pre-training)을 수행하며, 이를 통해 미지의 데이터 세트에 대해서도 일반화(generalization)가 우수합니다. 데이터의 lengths가 다를 때도 효과적으로 작동하여 기존 모델들보다 더 뛰어난 결과를 보여줍니다.

- **Performance Highlights**: 실험 결과에 따르면 zero-shot DispFormer는 레이블이 없는 데이터에서도 신뢰할 수 있는 초기 모델을 생성할 수 있으며, 기존 방법들과 비교했을 때 매우 유사한 결과를 산출합니다. 또한, 적은 양의 레이블이 주어졌을 경우, fine-tuning을 통해 누적된 성능 향상이 이루어져 전통적인 방법들을 초월하는 성과를 보여주었습니다. 실세계 데이터 세트에서도 DispFormer는 데이터 잔여량을 줄이는 데 성공하여 실제 응용에도 가능성을 지니고 있습니다.



### TimelineKGQA: A Comprehensive Question-Answer Pair Generator for Temporal Knowledge Graphs (https://arxiv.org/abs/2501.04343)
- **What's New**: 본 연구는 Temporal Knowledge Graphs(TKGs)를 기반으로 한 새로운 질문 응답 프레임워크인 TimelineKGQA를 제안하여, 시계열 데이터에 대한 정보 검색과 추론 간의 전환을 가능하게 합니다. 연구진은 또한 다양한 질문 유형을 포용하는 포괄적인 데이터 집합 생성을 위한 Python 패키지를 오픈 소스로 제공하여, 연구자들이 TKGQA 연구에서 활용할 수 있도록 지원합니다.

- **Technical Details**: Temporal Knowledge Graphs는 (e1,r,e2,tstart,tend) 형태의 데이터 쌍을 사용하여 시간적 관계를 통합합니다. TKGQA는 정보 검색(IR)과 시간적 추론 간의 연결 고리를 제공하며, 질문 복잡성에 대한 기준을 네 가지 차원(문맥 복잡성, 답변 초점, 시간적 관계, 시간적 능력)으로 구분합니다. 이 프레임워크는 사용자가 다양한 질문을 생성하고 분석할 수 있게 지원합니다.

- **Performance Highlights**: 현재 TKGQA 연구는 데이터셋의 한계로 인해 발전이 저해되고 있으며, 기존 데이터셋(CronQuestion 등)이 시간적 복잡성을 포괄적으로 다루지 못하고 있습니다. 연구진의 제안된 프레임워크를 통해 사용자 정의 QA 쌍 생성을 가능하게 하여, 시계열 데이터 연구의 진전을 도모할 수 있게 됩니다. 많은 경우, 모델은 0.9 이상의 성능을 달성했지만, 데이터셋의 다양성 부족으로 정보를 찾는 데 한계가 있습니다.



### RoRA: Efficient Fine-Tuning of LLM with Reliability Optimization for Rank Adaptation (https://arxiv.org/abs/2501.04315)
Comments:
          ICASSP 2025

- **What's New**: 이 논문은 RoRA (Rank-adaptive Reliability Optimization)라는 새로운 기법을 제안하며, 이는 LoRA의 스케일링 인자를 최적화하여 성능을 향상시킨다. RoRA는 스케일링 인자에서 r의 제곱근을 사용함으로써, rank 크기를 증가시켰을 때에도 성능이 개선될 수 있도록 설계되었다. 이 방법은 특히 압축된 모델과 비압축 모델의 파인튜닝에 효과적이다.

- **Technical Details**: RoRA의 핵심은 스케일링 인자 $
\alpha/
\sqrt{r}$을 통해 rank에 대한 그래디언트 업데이트의 변동성을 줄이는 것이다. 이로운 점으로는, LoRA 및 DoRA와 비교하여 더 높은 평균 정확도를 달성하며, rank가 증가하더라도 안정성을 유지하는 점이 있다. RoRA는 LLaMA 모델에서 6.5% 및 2.9%의 정확도 향상 결과를 보여준다.

- **Performance Highlights**: RoRA 메소드는 LLaMA-7B/13B, LLaMA2-7B 및 LLaMA3-8B 모델에서 최첨단 성능을 초월하는 성과를 기록하였다. 특히, SHEARED-LLAMA-1.3 모델에서는 81.4%의 프루닝을 달성하면서 LoRA보다 평균 정확도가 5.7% 높았다. 이러한 결과는 RoRA가 파인튜닝 과정에서 매우 효과적이라는 것을 입증한다.



### H-MBA: Hierarchical MamBa Adaptation for Multi-Modal Video Understanding in Autonomous Driving (https://arxiv.org/abs/2501.04302)
Comments:
          7 pages, 4 figures

- **What's New**: 이 논문에서는 자율주행에 대한 새로운 기회와 도전을 제시합니다. 특히, 다중 모달 비디오 이해가 자율주행 과정에서 발생할 사건을 분석하는 데 필수적임을 강조합니다. 기존 Multimodal Large Language Models(MLLMs)의 일반화 능력을 제한하는 복잡한 시공간 동작을 고려하여, 복잡한 움직임 변화를 수용하기 위한 새로운 Hierarchical Mamba Adaptation(H-MBA) 프레임워크를 제안합니다.

- **Technical Details**: H-MBA는 Context Mamba(C-Mamba)와 Query Mamba(Q-Mamba)라는 두 가지 모듈로 구성되어 있습니다. C-Mamba는 다양한 구조 상태 공간 모델을 포함하여 서로 다른 시간 해상도를 위한 비디오 맥락을 효과적으로 캡처합니다. Q-Mamba는 현재 프레임을 학습 가능한 쿼리로 변환하고, 다중 해상도 비디오 맥락을 선택적으로 조합하여 비디오 이해를 향상시킵니다.

- **Performance Highlights**: 이 방법을 통해 자율주행에서의 다중 모달 비디오 작업을 위한 성능이 크게 향상됩니다. 예를 들어, 위험 객체 탐지에서 기존 성능 주도 방법(SOTA)에 비해 5.5% mIoU(improved Intersection over Union) 향상을 달성했습니다. 모델 추론 과정에서는 초당 약 0.2초의 처리 시간을 기록하며, 이는 실제 응용 프로그램에서의 효용성을 강조합니다.



### Circuit Complexity Bounds for Visual Autoregressive Mod (https://arxiv.org/abs/2501.04299)
- **What's New**: 이 연구는 Visual AutoRegressive (VAR) 모델의 회로 복잡성(circuit complexity)에 대한 경계를 설정하고, VAR 모델이 $	ext{TC}^0$ 임계 회로(threshold circuit)의 시뮬레이션으로 동등함을 증명합니다. 이 회로는 은닉 차원(hidden dimension) $d 	ext{(d)}$가 $O(n)$ 이하이고, $	ext{poly}(n)$ 정밀도(precision)를 가지고 있습니다. VAR 모델은 이전 기술인 Diffusion Transformers를 초월하는 이미지 생성 능력을 보여주며, 본 연구는 이러한 모델의 표현력 표현의 한계를 철저히 분석한 첫 번째 연구입니다.

- **Technical Details**: 본 연구에서는 VAR 모델의 구조와 구성 요소(예: 업샘플링(interpolation) 레이어, 컨볼루션(convolution) 레이어, Transformer 블록 등)의 계산 복잡성(computational complexity)을 분석합니다. 회로 복잡성 이론을 통해 VAR 모델을 복잡성 회로(complexity circuits)로 표현함으로써, 이 모델이 수행할 수 있는 문제의 하한(bounds)을 정량적으로 평가할 수 있는 방법론을 제시합니다. 특히, $	ext{DLOGTIME}$-균일 $	ext{TC}^0$ 회로 패밀리가 VAR 모델을 O(1) 깊이(depth), $	ext{poly}(n)$ 크기(size), 및 $	ext{poly}(n)$ 정밀도로 시뮬레이션 가능함을 보여줍니다.

- **Performance Highlights**: VAR 모델은 기존의 이미지 생성 방법들과 비교했을 때, 더욱 사실적이고 다양한 이미지를 생성하는 능력을 보여줍니다. 특히, VAR 모델은 제로 샷 제너럴라이제이션(zero-shot generalization) 능력을 갖춰 이미지 인페인팅(image inpainting) 및 조작(manipulation) 작업 등 다양한 분야에서 뛰어난 성능을 발휘합니다. 본 연구는 VAR 모델의 표현력의 한계를 밝혀내며, 이로 인해 더 효율적이고 표현력 있는 아키텍처 개발에 기여할 가능성을 지니고 있습니다.



### MAD-UV: The 1st INTERSPEECH Mice Autism Detection via Ultrasound Vocalization Challeng (https://arxiv.org/abs/2501.04292)
Comments:
          5 pages, 1 figure and 2 tables. For MAD-UV Challenge 2025

- **What's New**: MAD-UV (Mice Autism Detection via Ultrasound Vocalization) 챌린지가 발표되었습니다. 이 챌린지는 마우스의 음성 발화를 기반으로 자폐 스펙트럼 장애(ASD)를 감지하는 최초의 INTERSPEECH 챌린지입니다. 참가자들은 고속 샘플링 비율로 녹음된 데이터를 바탕으로 마우스를 ASD 모델 또는 정상형으로 자동 분류하는 모델을 개발해야 합니다. 연구 결과는 자폐 조기 감지의 가능성을 시사합니다.

- **Technical Details**: 이 연구에서는 약 7777시간의 초음파 음성을 포함하는 데이터셋을 사용하여 848개의 주제에서 마우스의 음성 발화를 분석하였습니다. 기본 시스템은 세 가지 서로 다른 스펙트로그램 특징을 사용하는 간단한 CNN 기반 분류기를 적용했습니다. 연구 결과, 과제를 위한 오디오 신호의 자동화된 분석이 가능한 것을 나타냈으며, 주제 수준 분류에서 0.625의 UAR을 기록했습니다. 초음파 발화는 마우스의 건강 및 행동의 지표로 활용될 수 있습니다.

- **Performance Highlights**: 초음파 음성을 활용한 ASD 모델 선정에서 높은 성능을 보였습니다. UAR 기준으로 세그먼트 수준에서 0.600 및 주제 수준에서 0.625의 결과를 기록하면서, 자폐 스펙트럼 장애 감지의 가능성과 그 활용 가능성을 증명했습니다. 이 챌린지는 음성 기술 및 생물 의학 연구의 진전을 위한 중요한 발판이 될 것입니다. 차후 참가자들은 인간 음성 분석에서 기존 모델과 기법을 마우스의 초음파 발화 감지에 적용하는 탐색을 권장받고 있습니다.



### Mapping the Edge of Chaos: Fractal-Like Boundaries in The Trainability of Decoder-Only Transformer Models (https://arxiv.org/abs/2501.04286)
Comments:
          15 pages

- **What's New**: 본 연구에서는 작은 신경망에서 관찰된 하이퍼파라미터 경계의 프랙탈(Fractal) 특성을 중간 크기의 디코더 전용 트랜스포머 모델에 적용하고 있습니다. 특히, 안정적인 수렴과 발산 경계를 구분하는 하이퍼파라미터의 복잡성을 조사하여, 이러한 경계가 자가 유사적(self-similar) 구조를 형성하고 있음을 발견했습니다. 이를 통해 하이퍼파라미터 변화가 트레이닝 결과에 미치는 민감성을 강조하고 있습니다.

- **Technical Details**: 연구에서는 주목(attention)과 완전 연결층(fully connected layers)에서 하이퍼파라미터 경관을 분석하고, 보다 일관된 수렴(convergence) 척도를 사용하여 학습 속도(learning rate)와 수렴 범위를 조사하였습니다. 이 과정에서는 아담(Adam) 등의 최적화 알고리즘을 사용하여, 파라미터 업데이트 과정이 어떻게 프랙탈과 유사한 구조를 이루는지를 밝히고 있습니다. 결과적으로 안정된 수렴 영역이 복잡한 혼돈(chaotic) 경계로 둘러싸인 구조임을 나타내고 있습니다.

- **Performance Highlights**: 실험 결과, 트레인하우는 단순한 임계값(threshold)이 아닌 여러 스케일(scale)에서 평균적으로 일관된 패턴이 나타나는 자가 유사한 구조임을 발견했습니다. 이는 중간 크기의 트랜스포머 모델에서 수렴 제어가 더 복잡하게 작용함을 보여줍니다. 연구의 통찰은 하이퍼파라미터 조정이 신경망 훈련의 성능에 얼마나 큰 영향을 미치는지 이해하는 데 중요한 기여를 할 수 있음을 강조합니다.



### Enhancing Scene Classification in Cloudy Image Scenarios: A Collaborative Transfer Method with Information Regulation Mechanism using Optical Cloud-Covered and SAR Remote Sensing Images (https://arxiv.org/abs/2501.04283)
- **What's New**: 이번 연구는 클라우드 오염으로 인한 광학 정보 손실에 대응하기 위해, 클라우드가 있는 광학 데이터와 Synthetic Aperture Radar (SAR) 데이터를 활용하여 장면 분류를 수행하는 새로운 방법을 제안합니다. 새로운 방식인 협동 이식 전략(collaborative transfer strategy)을 통해 서로 다른 데이터 간의 지식을 효과적으로 전달할 수 있으며, 정보 조절 메커니즘(information regulation mechanism)을 통해 전송 과정에서 발생할 수 있는 양식 불균형 문제(modality imbalance issue)를 해결하려고 합니다. 이 연구는 주어진 최첨단 방식이 클라우드 오염이 있는 상황에서 다른 솔루션들보다 뛰어난 성능을 보여줌을 확인했습니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 요소로 구성됩니다: 첫째, 지식 증류(knowledge distillation)를 기반으로 한 협동 이식 전략은 이질적인 데이터 간에 효율적으로 사전 지식을 전달할 수 있게 합니다. 둘째, 정보 조절 메커니즘(IRM)은 전이 과정에서 발생할 수 있는 양식 불균형 문제를 해결하기 위해 보조 모델을 활용하여 각 양식의 기여 비율을 측정하고, 샘플 단위에서 양식 간 정보 활용의 균형을 자동으로 맞춥니다. 이를 통해 클라우드가 있는 광학 데이터와 SAR 데이터의 통합된 특성을 효과적으로 활용하여 장면 분류 작업을 진행합니다.

- **Performance Highlights**: 제안된 방법은 시뮬레이션된 클라우드 데이터셋과 실제 클라우드 데이터셋에서 전이 실험을 수행한 결과, 클라우드가 덮인 상황에서 다른 기존의 솔루션에 비해 우수한 성능을 입증하였습니다. 또한, IRM의 중요성과 한계를 검증하고 모델 전이 과정에서 발생하는 양식 불균형 문제를 논의하고 시각화했습니다. 이러한 결과는 클라우드 오염이 많은 지역에서 장면 분류 작업의 효율성을 높이는 데 기여할 것으로 기대됩니다.



### Scaling Large Language Model Training on Frontier with Low-Bandwidth Partitioning (https://arxiv.org/abs/2501.04266)
- **What's New**: 본 연구에서는 Frontier 슈퍼컴퓨터 클러스터를 위해 ZeRO++의 통신 및 최적화 전략을 제안하며, 메모리 활용도를 개선하고 통신 비용을 줄이는 목표를 가지고 있습니다. 도입된 3단계 계층 구조 분할 전략과 함께, 다양한 통신 레이어의 대역폭을 활용하여 통신 오버헤드를 줄이고자 합니다. 이 연구는 Frontier의 AMD GPU에서 LLM 워크로드를 최적화하는 첫 번째 노력으로 알려져 있습니다.

- **Technical Details**: 연구자는 20B GPT 모델을 대상으로, ZeRO++와 비교해 GPU당 1.71배의 TFLOPS 증가를 관찰하였으며, 최대 384 GCDs에서 스케일링 효율성 0.94를 달성했습니다. ZeRO++는 양자화 기법을 적용하여 메시지 크기를 줄이고, 이차 파티셔닝을 통해 인터 노드 통신을 최소화하는 방안을 도입하였습니다. 통신 최적화를 위해 GCD와 MI250X, 노드 간의 대역폭 차이를 고려하여 설계를 진행했습니다.

- **Performance Highlights**: ZeRO++를 적용한 결과, 총 연산 성능과 메모리 효율성을 극대화할 수 있었습니다. 특히, Frontier의 아키텍처 분석을 바탕으로 설계된 통신 전략이 더 효율적으로 작용함을 확인했습니다. 이러한 최적화 접근 방식은 LLM 교육의 연산-통신 비율을 개선하고 전체적인 학습 효율을 높이는 데 기여하고 있습니다.



### KN-LIO: Geometric Kinematics and Neural Field Coupled LiDAR-Inertial Odometry (https://arxiv.org/abs/2501.04263)
- **What's New**: 이 논문에서는 기존 LiDAR-Inertial Odometry (LIO) 시스템의 한계를 극복하기 위한 새로운 솔루션을 제안합니다. 특히, 기하학적 운동학과 신경 필드를 밀접하게 결합하여 동시에 상태 추정과 밀집 맵핑 기능을 개선했습니다. 이를 통해 우리는 semi-coupled 및 tightly coupled Kinematic-Neural LIO (KN-LIO) 시스템을 도입하였으며, 다양한 고역동적 데이터셋에서 평가를 통해 기존 기술보다 우수한 성능을 입증하였습니다.

- **Technical Details**: KN-LIO 시스템은 온라인 Signed Distance Function (SDF) 디코딩과 반복误差 상태 칼만 필터링을 활용하여 레이저 및 관성 데이터의 융합을 구현합니다. 반 쌍결합(semi-coupled) 접근법에서는 IMU의 고주파 독서를 통해 시스템 상태를 예측하고 점 군을 수정한 후 맵에 등록하는 과정을 진행합니다. 완전 결합(tightly coupled) 접근에서는 현재 점 군과 신경 필드 정보를 기반으로 Kalman gain을 업데이트하여 더욱 정밀한 상태 추정이 가능합니다.

- **Performance Highlights**: 다양한 고역동적 플랫폼에서 수집된 대규모 데이터셋을 통해 우리의 KN-LIO 시스템이 포즈 추정 시 기존 최첨단 솔루션과 동등한 성능을 달성하고, 밀집 맵핑에서는 순수 LiDAR 기반 방법들보다 더 나은 정확도를 제공하였음을 확인했습니다. 이를 통해 KN-LIO는 높은 동적 변화에도 안정적이며, 향후 로봇공학 커뮤니티에 기여할 수 있는 잠재력을 보여줍니다.



### Integrated Offline and Online Learning to Solve a Large Class of Scheduling Problems (https://arxiv.org/abs/2501.04253)
- **What's New**: 이 연구에서는 단일 머신 스케줄링 문제를 다루기 위해 통합된 머신러닝(Machine Learning) 접근 방식을 개발했습니다. 이 방법은 비감소(min-sum) 목표 함수를 가지고 있으며, 릴리스 시간(release times)이 있을 수도 없을 수도 있습니다. 특히, 시간 인덱스(time-indexed) 공식을 사용하여 전체 클래스 문제를 통합적으로 구성했습니다.

- **Technical Details**: 기술적인 측면에서, 저자들은 깊은 신경망(Deep Neural Network) 모델을 사용하여 비용 매개변수(cost parameters)를 입력으로 활용하고, 연속적인 솔루션을 예측하여 이로부터 정수 해답(discrete solution)을 쉽게 구축할 수 있도록 했습니다. 또한, NP-hard 성질로 인해 최적 솔루션의 라벨을 생성하기 어려운 문제를 해결하기 위해 특별한 인스턴스를 생성해 모델을 오프라인으로 학습했습니다. 추가로, 온라인 단일 인스턴스 학습을 통해 주어진 인스턴스에 대해 DNN의 매개변수를 미세 조정하는 방법도 포함되었습니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법은 최대 1000개의 작업을 포함하는 다양한 단일 머신 스케줄링 min-sum 문제에 대해 효율적으로 고품질 솔루션을 생성할 수 있음을 보여주었습니다. 이러한 결과는 효율적인 스케줄링 해결책을 제공하는 데 있어 머신러닝의 가능성을 입증합니다.



### Constraints as Rewards: Reinforcement Learning for Robots without Reward Functions (https://arxiv.org/abs/2501.04228)
- **What's New**: 이 논문에서는 로봇 행동 생성에 대한 기존 보상 함수 조정 과정을 피하기 위해 'Constraints as Rewards (CaR)', 즉 제약 조건을 보상으로 사용하는 새로운 접근 방식을 제안합니다. CaR는 로봇의 작업 목표를 제약 조건 함수로만 구성하여 보상 함수를 제거하고, Lagrangian 방법을 사용하여 여러 목표 간의 가중치를 자동으로 조정할 수 있게 합니다. 이 접근 방식은 로봇이 의미 있는 행동을 더 효율적으로 학습할 수 있도록 돕습니다.

- **Technical Details**: 제안된 방법은 CaR을 통해 제약 조건을 사용하여 강화 학습 문제를 해결합니다. CaR는 QRSAC-Lagrangian 알고리즘으로 구현되며, 이는 기존 QRSAC 알고리즘의 확장입니다. 이 알고리즘은 Lagrange 승수를 사용하여 여러 목표를 자동으로 균형잡게 하며, 네 가지 특정 제약 함수 설계를 제안하여 각 작업 목표를 직관적으로 해석 가능하게 합니다.

- **Performance Highlights**: 실험에서는 제안된 기법이 여섯 바퀴의 텔레스코픽 다리 로봇인 Tachyon 3의 일어서는 동작 생성 작업에 효과적으로 적용되었습니다. 전통적인 수동 설계된 보상 함수를 활용하는 방식보다 더 빠르고 안정적인 학습을 달성하며, 실제 환경에서 학습한 정책이 시뮬레이션과 유사하게 성공적으로 작업을 수행함을 입증합니다.



### Agent Laboratory: Using LLM Agents as Research Assistants (https://arxiv.org/abs/2501.04227)
- **What's New**: 본 논문에서는 Agent Laboratory라는 자율적인 LLM(언어 모델 기반) 프레임워크를 소개합니다. 이 프레임워크는 인간이 제공하는 연구 아이디어를 바탕으로 문헌 검토, 실험, 보고서 작성의 세 단계를 거쳐 종합적인 연구 결과물(코드 저장소 및 연구 보고서 포함)을 생성합니다. 사용자는 각 단계에서 피드백과 지침을 제공할 수 있어 연구 과정이 더욱 원활하게 진행됩니다.

- **Technical Details**: Agent Laboratory는 다양한 최첨단 LLM을 활용하여 연구 과정을 자동으로 수행합니다. 이 시스템은 연구의 모든 단계를 포괄하며, 연구자들에게 과정 전반에서 인간의 피드백을 받을 수 있는 경로를 제공합니다. 연구 결과는 자율 연구 방식의 전통적인 비용과 시간을 크게 절감합니다.

- **Performance Highlights**: 연구 결과에 따르면, o1-preview 기반의 Agent Laboratory가 가장 우수한 연구 결과물을 생성했습니다. 생성된 머신러닝 코드는 기존 방법들과 비교하여 최첨단 성능을 달성하였으며, 각 단계에서의 인간의 참여가 연구의 전반적인 품질을 크게 개선하는 것으로 나타났습니다. 최종적으로 Agent Laboratory는 연구 비용을 84%나 절감하는 효과를 보였습니다.



### Continual Self-supervised Learning Considering Medical Domain Knowledge in Chest CT Images (https://arxiv.org/abs/2501.04217)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 이 연구에서는 흉부 CT 이미지 분석을 위한 새로운 지속적 자기 지도 학습 방법(CSSL)을 제안합니다. 이 방법은 다양한 학습 단계에서 이전에 학습한 지식과 새로운 정보를 효율적으로 연관시켜, 데이터 간의 간섭 문제를 해결하는 데 중점을 두고 있습니다. 특히, 강화된 Dark Experience Replay(DER)를 도입하여, 재허가 버퍼의 다양성과 대표성을 유지하면서, 모델의 특징 표현 학습을 극대화합니다.

- **Technical Details**: 제안된 방법은 세 단계로 구성된 CSSL 접근 방식을 활용하여, 두 개의 CT 이미지 영역 간의 데이터 간섭 현상을 줄이고, 재학습 시의 파국적 망각을 방지합니다. 첫 번째 단계에서는 한 영역 내에서 자기 지도 학습을 실행하여 특징 표현을 학습하며, 두 번째 단계에서는 이전 단계에서 선택된 이미지를 버퍼에 저장하여 다양성과 대표성을 유지합니다. 마지막 단계에서는 또 다른 도메인에 대한 지속적 자기 지도 학습을 적용하며, 이를 통해 더 풍부한 특징 표현을 학습합니다.

- **Performance Highlights**: 실험 결과, 제안된 CSSL 방법이 최신 기술을 초월하는 성능을 보여주었습니다. 두 가지 다른 이미징 조건에서 얻은 흉부 CT 이미지를 사용하여 사전 학습을 진행하고, 공개 CT 이미지 데이터셋에 대한 평가를 통해 높은 정확도를 기록했습니다. 이 방법은 다양한 도메인 간의 데이터 분포 차이를 고려하여 더욱 강력한 특징 표현을 학습할 수 있도록 돕습니다.



### UPAQ: A Framework for Real-Time and Energy-Efficient 3D Object Detection in Autonomous Vehicles (https://arxiv.org/abs/2501.04213)
- **What's New**: 이 논문은 자율주행 차량(AV)에서 3D 객체 감지기의 효율성을 높이는 새로운 프레임워크인 UPAQ를 소개합니다. UPAQ는 반구조적 패턴 가지치기(semi-structured pattern pruning)와 양자화(quantization)를 활용하여 자원 제약이 있는 내장형 AV 플랫폼에서 LiDAR 포인트 클라우드와 카메라 기반 3D 객체 검출기의 효율성을 향상시킵니다.

- **Technical Details**: UPAQ는 최신 모델 압축 프레임워크에 비해 Pointpillar와 SMOKE 모델에서 각각 최대 5.62배 및 5.13배의 모델 압축률을 달성합니다. 또한, 추론 속도가 최대 1.97배 및 1.86배 향상되며, 에너지 소비는 최대 2.07배 및 1.87배 감소합니다. 이러한 결과는 Jetson Orin Nano 내장 플랫폼에서 실시된 실험을 기반으로 합니다.

- **Performance Highlights**: UPAQ 프레임워크는 기존의 2D 객체 감지기보다 더 포괄적인 예측을 제공하면서도, 자원 제한이 있는 상황에서 높은 성능을 발휘합니다. 성능 측면에서 더 빠른 추론 시간과 낮은 에너지 소비를 결합하여, 자율주행 기술의 발전을 이끌고 있습니다.



### CURing Large Models: Compression via CUR Decomposition (https://arxiv.org/abs/2501.04211)
- **What's New**: CURing이라는 새로운 모델 압축 방법을 소개합니다. 이 방법은 CUR 행렬 분해(CUR matrix decomposition)에 기반하여, 체중 행렬을 선택된 열(C)과 행(R) 및 소형 연결 행렬(U)의 곱으로 근사합니다. 이 접근법은 크기와 활성화의 조합 영향에 따라 선택된 가중치에 적용됩니다.

- **Technical Details**: CURing은 중요한 행렬을 식별하고 유지하여 모델 크기를 크게 줄입니다. 이 과정에서 원본 네트워크의 입력/출력 구조를 유지하며, 중요한 특징인 비음수(non-negativity)를 보존합니다. 또한, 압축된 모델의 활성화 패턴은 원본과 일치하여 해석 가능성(interpretability)을 높입니다.

- **Performance Highlights**: CURing은 최소한의 성능 손실(minimal performance loss)로 모델 크기를 줄이는 데 크게 기여합니다. 이는 깊은 학습 모델의 계산 비용(computational cost) 및 메모리 사용(memory usage) 문제를 효과적으로 해결하는 방향으로 나아가고 있습니다.



### Generative Dataset Distillation Based on Self-knowledge Distillation (https://arxiv.org/abs/2501.04202)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 새로운 연구에서는 dataset distillation의 효율성을 향상시키기 위해 self-knowledge distillation을 도입한 새로운 생성 기반 데이터셋 디스틸레이션 방법을 제안했습니다. 이 방법은 원본 데이터와 합성 데이터 간의 예측 로짓(logits) 정렬의 정확성을 향상시키고, 보다 효과적인 데이터 구조와 관계를 포착합니다. 특히, 정규화를 통해 로짓의 일관성을 유지함으로써 보다 정교한 분포 매칭(distribution matching)을 도모하고 있습니다.

- **Technical Details**: 본 연구의 방법론은 두 단계로 구성됩니다. 첫 번째로, 조건부 생성적 적대 신경망(GAN)을 학습하여 합성 데이터셋 S를 생성합니다. 이후 모델 풀(pool)에서 무작위로 선택된 모델을 통해 original 데이터셋 O와의 정렬을 수행하며, self-knowledge distillation을 통합하여 원본 및 합성 데이터의 분포를 효과적으로 일치시킵니다. 또한, 로짓의 범위를 일관되게 유지하기 위한 정규화 단계를 도입하여 정렬의 정확성을 높였습니다.

- **Performance Highlights**: 제안된 방법은 여러 벤치마크 데이터셋을 통해 기존의 최신 방법들보다 우수한 성능을 보여주었습니다. 특히, 생성자가 보다 정확하고 대표성 있는 합성 데이터를 생성하도록 돕고, 결과적으로 더 높은 품질의 데이터 디스틸레이션을 가능하게 했습니다. 이러한 접근 방식은 다양한 모델 아키텍처에서 통칭성과 성능을 개선하는데 기여하고 있습니다.



### GNN-based Decentralized Perception in Multirobot Systems for Predicting Worker Actions (https://arxiv.org/abs/2501.04193)
Comments:
          Submitted to RA-L

- **What's New**: 본 논문은 동적인 산업 환경에서 인간 행동을 예측하기 위한 지각 프레임워크를 제안합니다. 이 프레임워크는 각 로봇이 주변 상황을 나타내는 공간 그래프를 구축하고 이를 다른 로봇과 공유함으로써 느슨한 협력을 가능하게 합니다. 로봇들은 공간 데이터와 시간 정보를 결합하여 인간 행동을 시간에 따라 추적하고, 모든 로봇이 인간 행동에 대한 통합 해석에 동의하도록 보장하는 결정을 내립니다.

- **Technical Details**: 기술적으로, 이 시스템은 Graph Neural Networks (GNNs)와 Recurrent Neural Networks (RNNs)를 사용하여 인간 의도를 예측합니다. 로봇들은 ROS 환경에서 구현된 이 파이프라인을 통해 여러 로봇의 데이터를 결합하여 인간과 주변 물체 간의 공간 관계를 모델링합니다. 또한, 다수의 로봇이 환경에 대한 정보를 공유하며, 이 과정에서 발생할 수 있는 센서 실패를 보완하여 산업 환경에서의 안전성을 높입니다.

- **Performance Highlights**: 실험 결과, 로봇의 수를 늘리고 더 긴 시간 시퀀스를 포함시키면 예측 정확도가 향상됨을 보여줍니다. 또한, 합의 메커니즘은 시스템의 복원력을 강화하여 동적인 산업 환경에서 다중 로봇 설정의 신뢰성을 높입니다. 이러한 연구는 협력적 로봇과 인간 간의 안전하고 효과적인 협업을 지원하는 중요한 기초가 됩니다.



### Fixed Points of Deep Neural Networks: Emergence, Stability, and Applications (https://arxiv.org/abs/2501.04182)
Comments:
          21 pages, 7 figures

- **What's New**: 이번 연구에서는 Deep Neural Networks (DNNs)의 고정점(fixed points) 형성과 안정성에 대한 수치적 및 분석적 결과를 제시하고 있습니다. 특히 입력과 출력 벡터의 차원이 동일할 때 형성되는 고정점의 특성과 이들을 활용한 다양한 학습 방식(지도학습, 반지도학습, 비지도학습)에 대한 사례를 보여줍니다. 연구 결과에 따르면, 학습되지 않은 DNN에서 무작위로 초기화된 가중치와 편향이 있을 경우 오직 하나의 고정점만 존재한다고 합니다.

- **Technical Details**: 연구에서는 DNN의 구조(층의 너비, DNN의 깊이, 활성화 함수 등)와 가중치 행렬의 확률 분포가 고정점의 존재 및 개수에 미치는 영향을 분석하였습니다. 고정점의 안정성과 매력의 분포에 대한 연구는 DNN의 훈련 과정 동안 파라미터 분포가 'heavy-tailed'로 변화하는 현상을 설명합니다. DNN의 고정점 수 Q(N, L)은 층의 수 L과 층의 너비 N에 의존하며, 이 함수는 비단조(non-monotone) 행동을 보입니다.

- **Performance Highlights**: 실험적으로, DNN의 매개변수가 'light-tailed' 분포로 초기화된 경우, 훈련 후에는 매개변수 분포가 'heavy-tailed'로 변하는 것을 관찰하였습니다. 고정된 층 너비(N = N0)에 대해 Q(N0, L) 함수는 초기에는 증가하다가 다시 감소하여 1로 수렴하는 양상을 보입니다. 이러한 비단조적 행동은 입력-출력 Jacobian의 경험적 스펙트럼 분포의 방정식을 유도하고 이를 수치적으로 해결함으로써 확인되었습니다.



### HIVEX: A High-Impact Environment Suite for Multi-Agent Research (extended version) (https://arxiv.org/abs/2501.04180)
- **What's New**: 이번 논문에서는 HIVEX라는 새로운 환경 스위트를 소개합니다. 이는 생태적 도전 과제를 중심으로 한 multi-agent 연구의 벤치마크를 목적으로 합니다. HIVEX는 실제 문제를 모방함으로써 연구자들이 즉각적인 생태적 문제에 접근하고 해결할 수 있도록 영감을 주고자 합니다.

- **Technical Details**: HIVEX는 다양한 환경을 포함하고 있습니다: 바람 농장 제어(Wind Farm Control), 산불 자원 관리(Wildfire Resource Management), 드론 기반 재조림(Drone-Based Reforestation), 해양 플라스틱 수거(Ocean Plastic Collection), 공중 산불 억제(Aerial Wildfire Suppression) 등입니다. 이 환경들은 복잡성을 증대시키며, 기계 간 및 인간-기계 협력을 요구하는 multi-agent 시나리오에 적합합니다.

- **Performance Highlights**: 제안된 환경 스위트에서는 훈련 예제와 메인 및 서브 작업을 위한 기준선(baseline)을 제공하며, 실험 결과로 생성된 모든 훈련 모델은 Hugging Face에 호스팅됩니다. 또한 커뮤니티가 환경 스위트에서 훈련된 모델을 제출하고 순위를 확인할 수 있도록 Hugging Face에서 리더보드를 제공하고 있습니다.



### Multimodal Multihop Source Retrieval for Web Question Answering (https://arxiv.org/abs/2501.04173)
Comments:
          arXiv admin note: text overlap with arXiv:2010.03604 by other authors

- **What's New**: 이번 연구에서는 멀티모달 멀티홉 질문 응답(Multi-modal Multi-hop Question Answering) 문제를 해결하기 위해 그래프 사고 네트워크(graph reasoning network)를 제안하고 있습니다. 이 네트워크는 이미지와 텍스트 기반의 다양한 출처에서 지원 사실(supporting facts)을 찾기 위해 구문(syntactic) 및 의미 구조(semantic structure)를 활용합니다. 본 연구의 핵심은 그래프 구조의 중요성을 강조하며, 적절한 feature representations를 사용해 수행 성능을 높일 수 있음을 입증합니다.

- **Technical Details**: 이 연구에서는 멀티모달 멀티홉 질문 응답을 위한 그래프 합성곱 네트워크(Graph Convolutional Networks, GCN)를 사용하여 다양한 소스에서 정보 추출 및 관련 출처 선택을 해결합니다. GCN은 그래프의 다양한 노드 간 정보 공유를 통해 의사결정을 하는 데 최적화되어 있으며, 이를 통해 정보 집합(aggregation)과 멀티모달 검색(multi-modal retrieval) 작업에 적합한 것으로 나타났습니다. 세 가지 독립적인 접근 방식을 통해 GCN의 차별점과 성능을 보여줍니다.

- **Performance Highlights**: 본 연구에서는 GCN를 활용하여, 기존 transformer 기반의 baseline 모델을 4.6% Retrieval F1 score 향상시켜 성능 개선을 입증하였습니다. 실험 결과, 그래프 네트워크에서 메시지 전파(message propagation)를 통해 멀티모달 transformer를 대체할 수 있음 또한 밝혀졌습니다. 저자는 이 모델이 대규모 검색 환경에서도 적용 가능함을 보여주었습니다.



### Learning to Transfer Human Hand Skills for Robot Manipulations (https://arxiv.org/abs/2501.04169)
Comments:
          Preprint. Under Review

- **What's New**: 이번 논문에서는 로봇이 인간의 손 동작을 통해 섬세한 조작 작업을 학습할 수 있는 새로운 방법을 제안합니다. 기존 방법들은 로봇과 객체 간의 상호작용의 타당성을 고려하지 않고 운동학(kineamatics) 정보에만 의존하였으나, 이 방법은 인간의 동작 시연에서 로봇의 조작 동작을 직접 추론합니다. 이로써 인간 손과 로봇 시스템 간의 신체성 차이를 해결하는 학습 기반 방법을 구사하여, 실제 실험에서 전통적인 리타게팅(retargeting) 기법보다 훨씬 높은 성능을 보여주고 있습니다.

- **Technical Details**: 제안된 방법은 인간 손 동작, 로봇의 손 동작, 그리고 객체의 3D 움직임 간의 공동 운동 매니폴드(joint motion manifold)를 학습하는 것을 목표로 합니다. 이를 통해 각 움직임 요소 간의 관계를 파악하고 인간 동작 시연에서 얻은 정보를 바탕으로 로봇의 조작 경로를 추론할 수 있습니다. 핵심 아이디어는 인간, 객체, 로봇의 움직임 경로를 조합하여 생성된 가상 감독 삼중쌍(pseudo-supervision triplets)을 활용하여 학습하는 것입니다.

- **Performance Highlights**: 실험 결과, 제안된 데이터 기반 리타게팅 방법은 복잡한 덱스터러스 조작 작업을 해결하는 데 있어 매우 효과적임을 입증했습니다. 이는 기존의 리타게팅 기법이 가지는 한계를 극복하면서도, 인간의 손 움직임을 로봇이 더욱 자연스럽고 물리적으로 타당하게 모사할 수 있도록 합니다. 특히, 이 방법은 다양한 손/객체 동작에 일반화 가능해, 향후 인간의 동작 데이터를 활용한 로봇 조작의 발전을 기대하게 합니다.



### Reasoning-Enhanced Self-Training for Long-Form Personalized Text Generation (https://arxiv.org/abs/2501.04167)
- **What's New**: 본 논문은 개인화된 텍스트 생성을 위해 Reasoning-Enhanced Self-Training for Personalized Text Generation (REST-PG) 프레임워크를 제안합니다. 이 프레임워크는 대형 언어 모델(LLM)이 사용자에 대한 과거의 선호도, 배경 지식 및 작성 스타일을 기반으로 reasoning(추론)을 수행하도록 훈련합니다. 이를 통해 REST-PG는 모델이 개인화된 맥락을 보다 효과적으로 활용하게 함으로써, 사용자 기대에 부합하는 응답을 생성할 수 있도록 합니다.

- **Technical Details**: REST-PG는 LLM이 개인화된 데이터를 기반으로 응답을 생성하는 동안 reasoning을 진행할 수 있도록 하는 다단계 프레임워크입니다. 이 시스템은 Expectation-Maximization Reinforced Self-Training을 활용하여, 높은 보상을 받는 응답을 생성하는 reasoning 경로를 반복적으로 학습시킵니다. 초기 단계에서는 LLM이 스스로 reasoning 경로를 생성하고, 이후에는 이러한 경로를 토대로 어떻게 보상을 극대화할 수 있는지를 학습합니다.

- **Performance Highlights**: 실험 결과, REST-PG는 LongLaMP 벤치마크에서 기존 최첨단 모델들보다 평균 14.5% 향상된 성능을 보였습니다. 특히, supervised fine-tuning(SFT)과 비교해도 14.5%의 성능 향상이 있었으며, reasoning enhancement 없이 self-training을 진행한 경우보다도 6.5% 개선된 결과를 나타냈습니다. 이러한 성과는 REST-PG가 개인화된 텍스트 생성 분야에서 효과적인 접근법임을 입증합니다.



### BiasGuard: Guardrailing Fairness in Machine Learning Production Systems (https://arxiv.org/abs/2501.04142)
- **What's New**: 이 논문에서는 최전선 머신러닝 시스템에서 공정성을 보장하기 위해 설계된 새로운 접근 방법인 'BiasGuard'를 소개합니다. BiasGuard는 Conditional Generative Adversarial Network(CTGAN)를 활용하여 역 보호 속성 값에 조건화된 데이터 샘플을 합성하여 다양한 그룹 간의 공정한 결과를 촉진합니다. 이 방법은 모델 재훈련 없이 배포된 시스템의 공정성 메트릭스를 향상시키는 것을 목표로 합니다.

- **Technical Details**: BiasGuard는 Test-Time Augmentation(TTA)을 사용하여 배포 후 생성된 테스트 샘플을 보강함으로써 공정한 예측을 촉진합니다. 이러한 접근법은 CTGAN이 생성한 합성 데이터를 사용하여 예측을 동적으로 재조정하고, 이는 공정한 기회를 모든 인구 통계학적 집단에 제공하는 데 기여합니다. BiasGuard는 기존의 검증되지 않은 기준선에 비해 공정성을 31% 향상시키는 동시에 정확도를 0.09%만 감소시킵니다.

- **Performance Highlights**: BiasGuard는 재훈련이 불가능한 생산 환경에서도 공정성을 보장할 수 있는 강력한 방법론으로 자리 잡고 있습니다. 기존의 포스트 프로세싱 방법들보다 공정성 향상에서 더 나은 성능을 보이며, 다양한 데이터 세트에서 공정성을 증대시키는 효과를 나타내고 있습니다. 이러한 강력한 결과는 BiasGuard가 머신러닝 시스템에서 편향으로부터 보호하기 위한 효과적인 도구임을示합니다.



### TrojanDec: Data-free Detection of Trojan Inputs in Self-supervised Learning (https://arxiv.org/abs/2501.04108)
Comments:
          To appear in AAAI 2025

- **What's New**: 이 논문은 TrojanDec를 제안하며, 이는 당임이 없는 방법으로 트로이카 이미지의 동작성(behavior)을 식별하고 복원하는 첫 번째 프레임워크입니다. 기존의 트로이카 방어 방법들이 일반적으로 데이터가 필요한 반면, TrojanDec는 데이터가 없거나 최소한의 정보로도 원활하게 작동할 수 있다는 점에서 혁신적입니다. 또한, 이 프레임워크는 많은 양의 비상태 데이터에도 불구하고 트로이카를 효과적으로 처리할 수 있는 가능성을 보여줍니다.

- **Technical Details**: TrojanDec는 세 가지 주요 구성 요소로 이루어져 있습니다: 1) 메타데이터 추출, 2) 트로이카 탐지, 3) 이미지 복원입니다. 메타데이터 추출 단계에서는 트로이카가 포함된 입력의 주요 메타데이터를 파악하여 이미지가 트로이카인지 여부를 감지합니다. 이어서 통계 분석을 통해 이미지를 평가한 후, 트로이카가 확인되면 확산 모델(diffusion model)을 사용하여 복원 과정을 진행합니다. 이러한 방식은 기존의 방어 솔루션과 비교 시 획기적인 접근 방식을 보여줍니다.

- **Performance Highlights**: TrojanDec는 여러 선진 트로이카 공격에 대해 광범위한 실험을 통해 효과성을 입증했습니다. 실험 결과, 기존의 방어 수단들과 비교할 때 TrojanDec는 항상 더 나은 성능을 나타냈습니다. 특히, 다양한 유형의 트로이카 공격에 대해 PassTrans를 포함한 여러 백도어 공격에 대한 방어 성능도 확인되었습니다.



### Enhancing Distribution and Label Consistency for Graph Out-of-Distribution Generalization (https://arxiv.org/abs/2501.04102)
Comments:
          Accepted by ICDM 2024

- **What's New**: 이 논문에서는 그래프 데이터에서의 분포 변화 (distribution shifts)를 처리하기 위한 새로운 방법론을 제안합니다. 기존의 그래프 OOD (Out-of-Distribution) 일반화 기술이 일관성 문제에 부딪히고 있으며, 본 연구에서는 이를 해결하기 위해 두 가지 일관성을 향상시키는 혁신적인 접근법을 도입합니다. 제안된 방법은 그래프를 동시에 증강하고 불변하게 만드는 modifier를 설계하여 데이터의 대체 관계를 보장합니다.

- **Technical Details**: 우리의 프레임워크는 두 가지 중요한 모듈을 갖추고 있습니다. 첫째, 분포 일관성 증대 모듈은 훈련 중에 증강된 그래프와 기존 그래프 간의 정보를 최대화하여, 그래프 간의 일관성을 유지하게 합니다. 둘째, 라벨 일관성 증대 모듈은 추출된 불변 하위 그래프가 원본 그래프와 최대한 많은 감독 정보를 공유하도록 보장하여 라벨의 유효성을 유지합니다.

- **Performance Highlights**: 실제 그래프 데이터셋에 대한 광범위한 실험을 통해, 우리의 프레임워크가 다른 최신 방법들보다 우월한 성능을 보임을 보여주었습니다. 우리의 프레임워크는 다양한 그래프 수준과 노드 수준의 OOD 일반화 데이터셋에서 효과성을 증명하였습니다. 이는 분포와 라벨의 일관성 문제를 해결하는데 실질적인 기여를 합니다.



### Multi-armed Bandit and Backbone boost Lin-Kernighan-Helsgaun Algorithm for the Traveling Salesman Problems (https://arxiv.org/abs/2501.04072)
- **What's New**: 본 논문에서는 Traveling Salesman Problem (TSP)을 해결하기 위해 Lin-Kernighan-Helsgaun (LKH) 휴리스틱 알고리즘을 개선하는 방법을 제안합니다. 기존의 $\\alpha$-value를 대체하여 검색 중의 이력 정보를 보다 효과적으로 활용하고 로컬 옵티마에서 탈피할 수 있도록 지원하는 새로운 접근 방식을 도입합니다. 이를 통해 알고리즘의 유연성을 높이고 성능을 크게 향상시킵니다.

- **Technical Details**: 저자들은 TSP의 로컬 검색 과정에서 backbone 정보를 동적으로 추출하여 사용합니다. 이와 함께 $\\alpha$-value, distance, backbone 정보를 조합하여 엣지 품질을 평가합니다. 또한, multi-armed bandit (MAB) 모델을 사용하여 적절한 평가 메트릭을 동적으로 선택함으로써 알고리즘은 다양한 유도 정보를 학습하고 적용할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 LKH 및 LKH-3 알고리즘에 적용되어, 대표적인 TSP 및 Vehicle Routing Problem (VRP) 변형 문제인 Colored TSP (CTSP)와 Capacitated VRP with Time Windows (CVRPTW)에서 우수한 성능을 보여줍니다. 광범위한 실험 결과를 통해 제안된 방법의 효과성과 일반화 능력이 입증되었습니다.



### More is not always better? Enhancing Many-Shot In-Context Learning with Differentiated and Reweighting Objectives (https://arxiv.org/abs/2501.04070)
Comments:
          13 pages, 8 figures, 11 tables

- **What's New**: 이 논문은 DR-ICL이라는 새로운 최적화 방법을 도입하여 대형 언어 모델(LLMs)의 Many-shot In-context Learning(아래 ICL)의 성능을 향상시키는 접근 방식을 제안합니다. DR-ICL은 Differentiated Learning과 advantage-based Reweighting을 활용하여 모델이 발생하는 데이터 노이즈의 영향을 줄이면서 더 나은 성능을 발휘하도록 합니다. 또한, MICLB라는 대규모 벤치마크 데이터셋을 소개하여 다양한 NLP 작업에서 Many-shot ICL 전략을 평가할 수 있도록 합니다.

- **Technical Details**: DR-ICL은 NLL(negative log-likelihood) 최적화 목표를 개선하여 모델이 많은 시연을 가진 경우에도 제로 샷 성능을 초과하도록 합니다. 이 방법은 강화 학습에서 영감을 받은 advantage function을 활용하여 샘플들의 가중치를 동적으로 조정함으로써, 훈련 과정에서의 노이즈 데이터를 필터링합니다. 이러한 기법은 ICL에서 여러 샷 수를 효과적으로 처리할 수 있게 하며, 다양한 작업에서의 일반화를 개선합니다.

- **Performance Highlights**: 실험 결과, DR-ICL로 개선된 LLM은 다양한 작업에서의 Many-shot ICL 설정에서 유의미한 성능 향상을 달성했습니다. 특히, 문맥적 단서에 대한 더 깊은 이해를 통해 모델이 맥락 정보를 효과적으로 활용하도록 유도합니다. 이 연구의 결과는 LLM의 다중 작업 학습 및 평가 연구에 있어 중요한 기여를 하며, 오픈 소스 LLM에 대한 연구의 발전을 촉진할 것입니다.



### Explainable Reinforcement Learning for Formula One Race Strategy (https://arxiv.org/abs/2501.04068)
Comments:
          9 pages, 6 figures. Copyright ACM 2025. This is the authors' version of the work. It is posted here for your personal use. Not for redistribution. The definitive Version of Record will be published in SAC 2025, this http URL

- **What's New**: 이번 연구에서는 F1 경주에서 주행 전략을 제어하기 위해 강화 학습 모델인 RSRL(경주 전략 강화 학습)을 소개합니다. 전통적인 하드코딩 및 몬테카를로 기반의 경주 전략에 비해 더 빠른 대안을 제공하며, 이를 통해 경주 당 최고 위치를 개선할 수 있는 가능성을 보여줍니다. 특히 2023 바레인 그랑프리 테스트에서 평균 완주 위치 P5.33을 기록하여 기존 모델을 초월하였습니다.

- **Technical Details**: 연구에서 사용하는 RL(강화 학습) 모델은 Q-learning, 딥 Q-네트워크(DQN), 심층 순환 Q-네트워크(DRQN) 등의 기법을 포함합니다. DQN은 Q-learning과 깊은 신경망을 결합한 모델로, TD 오류를 최소화하면서 현재 상태에서의 Q 값을 예측합니다. DRQN은 과거 상태의 정보를 활용하여 현재 상태를 예측하는 방향으로 DQN의 성과를 향상시키고 있습니다.

- **Performance Highlights**: 기술적으로 RSRL 모델은 2023 바레인 그랑프리에서 P5.33의 평균 완주 위치를 기록하여 기존 가장 좋은 기준 모델인 P5.63을 초과 달성했습니다. 또한 성장 가능성을 보여주는 일반화 연구에서 다양한 트랙에 대한 성능 우선 순위를 학습을 통해 정할 수 있음을 입증했습니다. 마지막으로, XAI(설명 가능한 인공지능) 기법을 통해 모델의 결정을 설명하여 사용자 신뢰를 증진시키고 있습니다.



### Explainable Time Series Prediction of Tyre Energy in Formula One Race Strategy (https://arxiv.org/abs/2501.04067)
Comments:
          9 pages, 9 figures. Copyright ACM 2025. This is the authors' version of the work. It is posted here for your personal use. Not for redistribution. The definitive Version of Record will be published in SAC 2025, this http URL

- **What's New**: 본 연구는 Formula One(Formula 1)에서 타이어 에너지 예측을 위한 심층 학습 모델을 훈련하고 최적화한 내용입니다. 특히, 경주 전략의 중요한 요소인 피트 스톱 및 타이어 컴파운드 선택 시 타이어 열화를 예측할 수 있게 되었습니다. 우리의 접근 방식은 Mercedes-AMG PETRONAS F1 팀의 역사적 데이터에서 훈련된 AI 모델을 통해 이루어졌으며, 이는 F1 팀들이 경주 전략을 최적화하는 데 도움을 줄 수 있는 자동화된 방법을 제공합니다.

- **Technical Details**: 이 연구에서는 여러 종류의 심층 학습 모델, 특히 Recurrent Neural Networks(RNN)와 최신 Transformer 기반 모델을 포함하여 총 네 가지의 모델이 훈련되었습니다. 이 모델들은 차량의 텔레메트리 데이터에서 얻은 다양한 변수를 입력받아 타이어 에너지를 예측하도록 설계되었습니다. 또한, Linear Regression과 XGBoost와 같은 기계 학습 알고리즘 비교를 통해 예측의 정확성을 검증하고 두 가지 설명 가능한 인공지능(XAI) 기법을 통합하여 예측의 기반이 되는 이유를 이해할 수 있도록 하였습니다.

- **Performance Highlights**: 모델의 성능을 평가한 결과, XGBoost가 테스트 세트에서 가장 정확한 예측을 보여주었으나 심층 학습 모델들의 결과도 유망하다고 평가되었습니다. 두 가지 XAI 방법이 통합됨으로써 제공되는 예측의 설명이 신뢰를 높이는 데 기여하고 있습니다. 이러한 연구는 F1에서 실시간 데이터 처리 및 타이어 에너지 예측을 위한 AI 모델 활용에 한 걸음 더 나아간 성과로 평가됩니다.



### ChronoLLM: A Framework for Customizing Large Language Model for Digital Twins generalization based on PyChrono (https://arxiv.org/abs/2501.04062)
- **What's New**: 최근 인공지능(AI)와 고급 시뮬레이션 기술의 통합은 과학 및 공학 연구에 혁신을 가져오고 있습니다. ChronoLlama는 특히 코드 생성에 맞게 오픈 소스 LLM을 커스터마이즈하고 다중 물리학 시뮬레이션을 위한 PyChrono와 결합한 새로운 프레임워크를 도입합니다. 이 통합은 시뮬레이션 스크립트의 생성 과정을 자동화하고 개선하면서 모델의 정확도와 효율성을 높이는 것을 목표로 하고 있습니다. 실험 결과는 시뮬레이션 설정 속도, 생성된 코드의 정확도 및 전반적인 계산 효율성에서 실질적인 향상을 보여줍니다.

- **Technical Details**: Project Chrono는 복잡한 시스템의 모델링, 시뮬레이션 및 분석을 지원하는 오픈 소스 물리 기반 시뮬레이션 프레임워크입니다. PyChrono는 Project Chrono의 기능을 사용자 친화적으로 이용할 수 있도록 해주는 Python 래퍼입니다. Chrono는 다중체 역학 및 비선형 유한 요소 분석을 위한 핵심 기능을 제공하며, CAD 도구와의 호환성을 통해 SolidWorks에서 정의된 기계 시스템을 Chrono로 가져오는 기능을 갖추고 있습니다. Chrono는 NASA의 VIPER 미션, 국방부 HPCMP의 CREATE-GV 프로젝트 등 다양한 분야에서 활용되었습니다.

- **Performance Highlights**: ChronoLlama는 다중체 시스템 개발 및 테스트의 속도를 가속화하며 복잡한 기계 시뮬레이션 관리를 위한 확장 가능하고 AI 향상된 접근법을 선도하고 있습니다. 최신 AI 기술의 통합은 엔지니어링 응용 프로그램의 설계 프로세스 자동화 및 최적화에서 중요한 진전을 나타내며, 법칙에 기반한 시뮬레이션의 신뢰성과 AI 주도의 코드 생성 속도를 결합하여 연구자와 엔지니어에게 강력한 도구를 제공합니다. 이 프로젝트는 과학 및 엔지니어링 전문 분야에서 문제 해결 및 설계 접근 방식을 변혁시키고 있습니다.



### Traits of a Leader: User Influence Level Prediction through Sociolinguistic Modeling (https://arxiv.org/abs/2501.04046)
- **What's New**: 이번 논문은 사용자의 영향력을 예측하는 새로운 접근 방식을 제시합니다. 기존의 연구들이 특정 도메인에 국한되어 있었던 반면, 이 연구는 사용자 댓글과 보상을 기반으로 커뮤니티의 지지를 이용하여 사용자 영향력 수준을 정의합니다. 또한, 인구 통계 및 성격 데이터를 활용하여 새로운 모델을 개발함으로써 다양한 도메인에서 성능을 향상시켰습니다.

- **Technical Details**: 이 논문에서는 사용자의 영향력 수준을 커뮤니티의 지지에 대한 함수로 정의하고, 최소 32개의 토큰을 포함하는 댓글을 바탕으로 사용자의 영향력 수준을 예측하는 방법을 개발합니다. k-index 점수를 사용하여 사용자의 영향력을 측정하며, 이 점수는 사용자 댓글의 karma 점수를 기반으로 합니다. 연구에서는 다중 작업 모델을 통한 성격 및 인구 통계 데이터 탐지와 같은 보조 과제가 도입되어 영향력 예측을 개선했습니다.

- **Performance Highlights**: 이 연구는 8개의 서로 다른 도메인에서 RankDCG 점수를 일관되게 개선했습니다. 특히, 사용자 중심의 정보를 활용하여 텍스트 기반 모델에서 우수한 성능을 보였으며, 그래프 신경망을 텍스트 분석에 응용할 가능성을 열었습니다. 연구 결과는 사용자 댓글을 통한 영향력 예측의 정확도를 높이는 데 기여할 것으로 기대됩니다.



### A Survey on Large Language Models with some Insights on their Capabilities and Limitations (https://arxiv.org/abs/2501.04040)
Comments:
          174 pages, to be submitted to a journal in a shorter version. arXiv admin note: text overlap with arXiv:2303.18223, arXiv:2303.17564, arXiv:2301.00234, arXiv:2303.08774, arXiv:2402.02315, arXiv:2210.03493, arXiv:2402.01817, arXiv:2407.21783, arXiv:2208.05051 by other authors

- **What's New**: 최근 인공지능 분야는 Transformers 아키텍처에 기반한 대형 언어 모델(LLMs)의 발전에 의해 크게 변화하였습니다. LLMs는 텍스트 생성, 질문 답변, 번역 및 요약과 같은 다양한 언어 관련 작업에서 사람처럼 이해하는 능력을 선보이며, 이는 자연어 처리 방식에 혁신을 가져왔습니다. 특히, 이 모델들은 코딩 생성, 기본 상식 추론 및 산술 계산과 같은 본연의 기능을 넘어서는 능력을 보여주기로 주목받고 있습니다.

- **Technical Details**: 이 논문에서는 LLM의 기본 구성 요소와 스케일링 메커니즘, 건축 전략을 탐구하며, 특히 GPT와 LLaMA와 같은 모델들에 중점을 둡니다. 급증하는 데이터와 계산 능력이 LLM 성능에 미치는 영향을 분석하며, 스케일링과 관련된 트레이드오프에 대해서도 논의합니다. 또한 LLM의 다양한 적용 사례에 대해 살펴보고, 이것이 의료, 금융, 교육, 법률 등 각 분야에서의 문제 해결 능력을 어떻게 나타내는지 설명합니다.

- **Performance Highlights**: LLM은 향상된 언어 이해 덕분에 복잡한 언어적 도전 과제를 해결할 수 있는 잠재력을 지니고 있습니다. LLM들은 코그니티브한 작업을 수행하는 데 필요한 계획 및 추론 능력을 갖추고 있으며, Chain of Thought(CoT) 및 Plan of Thought(PoT)와 같은 새로운 접근 방식을 통해 그 성능을 더욱 향상시킬 수 있습니다. 본 논문은 LLM의 능력과 한계를 지속적으로 탐구하여, 이 분야에서의 책임 있는 개발 및 활용 방안을 모색할 것입니다.



### Listening and Seeing Again: Generative Error Correction for Audio-Visual Speech Recognition (https://arxiv.org/abs/2501.04038)
- **What's New**: 이번 논문에서는 Audio-Visual Speech Recognition (AVSR) 시스템을 위한 새로운 Generative Error Correction (GER) 패러다임인 AVGER를 제안합니다. AVGER는 '다시 듣고 보고'라는 개념을 따르며, 오디오와 비주얼 정보를 동시에 처리하여 ASR 결과를 정정합니다. 기존의 방법들과 달리, AVGER는 멀티모달 정보의 이해를 개선하고 정확한 필사를 생성하는 효율적인 방법으로 자리잡고 있습니다.

- **Technical Details**: AVGER는 Q-Former 기반의 멀티모달 동기 부여 인코더를 사용하여 N-최선 가설(N-Best hypotheses)과 원본 오디오 및 비주얼 정보를 통합합니다. 이 방식은 Cross-modal Prompt를 통해 LLM을 안내하여 가장 정확한 필사를 생성하도록 합니다. 또한, logits-level, utterance-level 및 representations-level을 포함하는 다중 레벨 일관성 제약 훈련 기준을 제안합니다.

- **Performance Highlights**: 실험 결과 LRS3 데이터셋에서 AVGER 방법이 현재의 주류 AVSR 시스템을 초과하여 WER(워드 오류율)를 24% 줄이는 성과를 보였습니다. 이는 시각적 정보의 통합이 AVSR의 정정 정확성을 어떻게 향상시키는지를 보여주는 중요한 예시입니다. 본 연구의 접근법은 AVSR 시스템이 소음을 극복하고 보다 정확한 음성 인식을 수행하는 데 기여할 것입니다.



### AICat: An AI Cataloguing Approach to Support the EU AI Ac (https://arxiv.org/abs/2501.04014)
Comments:
          Presented at 37th International Conference on Legal Knowledge and Information Systems (JURIX) 2024

- **What's New**: 유럽연합의 인공지능 법안(AI Act)은 고위험 AI 애플리케이션의 제공자와 배포자가 시스템을 EU 데이터베이스에 등록해야 한다고 규정하고 있습니다. 여기서 제공된 정보는 쉽게 탐색 가능하고 기계가 읽을 수 있는 형태로 유지되어야 합니다. 본 논문은 AI 시스템을 카탈로그화할 수 있는 AICat을 소개하며, 이는 DCAT의 확장으로 고위험 AI 시스템에 대한 관리적 해결책을 제공합니다.

- **Technical Details**: AICat은 AI 시스템 및 구성요소를 나타내기 위한 기계 가독성, 검색 가능성 및 상호 운용성을 제공하는 카탈로깅 접근법입니다. 이 기술적 프레임워크는 DCAT(Application Profile) 및 기타 관련 메타데이터 표준을 기반으로 하며, AI 시스템의 데이터베이스와 그 자원 및 모델을 효과적으로 관리할 수 있도록 돕습니다. AI Act의 등록 요구사항을 살펴보고, AICat의 구현을 통해 이 공백을 메우는 방안을 제안합니다.

- **Performance Highlights**: AICat은 AI 시스템 및 모델의 등록 및 공유에 필요한 메타데이터의 표준화된 카탈로그 기능을 제공합니다. 적용 가능한 각 카탈로그는 일관된 메타데이터를 통해 투명성과 추적 가능성을 보장하며, 이는 EU 고위험 AI 시스템의 요구사항을 충족시키는 데 필수적입니다. 이 연구는 EU 내 AI 애플리케이션 시장에서의 책임성 및 공정성을 보장하기 위한 길잡이가 될 것입니다.



### A Generative AI-driven Metadata Modelling Approach (https://arxiv.org/abs/2501.04008)
Comments:
          Accepted for publication @ Special Issue on "Generative AI and Libraries" - Library Trends Journal, Johns Hopkins University Press, Maryland, USA

- **What's New**: 이 논문은 최신 연구 동향 중 하나인 Generative AI와 연계하여 도서관 메타데이터 모델을 개선하는 방법을 제시합니다. 특히, 기존의 몇 가지 핵심 메타데이터 모델에 대한 의존성을 극복하고, 다양한 정보 서비스에서 필요로 하는 모델의 복잡성을 해결하는 접근 방식을 소개합니다. 이를 통해 도서관의 정보 서비스가 더 효과적이고 유연해질 수 있음을 강조합니다.

- **Technical Details**: 저자는 메타데이터 모델을 다섯 가지 기능적으로 연결된 표현 수준의 조합으로 구성된 온톨로지 기반 모델로 재구성합니다. 이 과정에서는 각 표현 수준의 내재적 표현 다면성(representational manifoldness)이 강조되며, 이를 통해 복잡한 정보 환경에서의 메타데이터 모델 재사용성을 증대시킬 수 있음을 설명합니다. 마지막으로, Generative AI와 Human-Large Language Model(LLM)의 협업을 바탕으로 한 메타데이터 모델링 방식을 제안합니다.

- **Performance Highlights**: 이 연구는 암 정보 처리에 유능한 대표적인 도서관의 사례를 들어, 제안된 메타데이터 모델이 어떻게 복잡성을 줄이고 명확한 정보 제공을 가능하게 하는지를 보여줍니다. 연구 결과는 도서관들이 Generative AI를 활용하여 사용자 맞춤형 정보 서비스를 제공할 수 있음을 시사하며, 새로운 메타데이터 모델이 정보 접근성을 크게 향상할 수 있는 가능성도 깔고 있습니다.



### DPO Kernels: A Semantically-Aware, Kernel-Enhanced, and Divergence-Rich Paradigm for Direct Preference Optimization (https://arxiv.org/abs/2501.03271)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)과의 정렬(alignment) 문제를 해결하기 위한 새로운 접근법인 DPO-Kernels를 제안합니다. DPO-Kernels는 커널 방법(kernel methods)을 통합하여 고정된 편차(fixed divergences) 및 제한된 특성 변환(feature transformations)의 문제를 극복하고자 합니다. 이를 통해 다각적인 전이(transformations)와 손실(loss) 함수를 개선하였습니다.

- **Technical Details**: DPO-Kernels의 주요 기여는 네 가지입니다: (i) 다항식(polynomial), RBF, 마할라노비스(Mahalanobis), 스펙트럼(spectral) 커널을 활용한 커널화된 표현(kernelized representations)과 임베딩 기반(embedding-based) 및 확률 기반(probability-based) 목표를 결합한 하이브리드 손실(hybrid loss); (ii) 다양성 대안(divergence alternatives)으로 제센-샤논(Jensen-Shannon), 헬링거(Hellinger), 레니(Renyi), 바타차리야(Bhattacharyya), 워서스타인(Wasserstein), f-다양성(f-divergences)을 포함하여 더 큰 안정성(stability) 제공; (iii) 데이터 기반 선택(metrics)으로 최적의 커널-다양성 쌍을 자동으로 선택; (iv) 지역 정확성과(global modeling) 전역 모델링을 위한 계층적 혼합(hierarchical mixture) 커널입니다.

- **Performance Highlights**: 12개의 데이터셋에 대한 평가 결과, DPO-Kernels는 사실성(factuality), 안전성(safety), 추론(reasoning), 지침 따르기(instruction following)에서 최첨단 성능(state-of-the-art performance)을 보였습니다. 이 연구는 Heavy-Tailed Self-Regularization에 기초하여 LLM의 강건한 일반화(robust generalization)를 유지하며, 정렬 연구(alignment research)를 위한 포괄적인 자원(resources)을 제공합니다.



New uploads on arXiv(cs.LG)

### Grokking at the Edge of Numerical Stability (https://arxiv.org/abs/2501.04697)
- **What's New**: 이번 연구에서는 grokking 현상에 대한 새로운 관점을 제시합니다. 연구자들은 정규화 없이 모델이 학습 중 발생하는 Softmax Collapse (SC)와 같은 수치적 불안정을 방지하기 위한 StableMax와 ⟂Grad와 같은 새로운 기법을 도입했습니다. 이러한 기법들은 grokking을 촉진하고, 기존의 방법들이 왜 효과적인지를 설명하는 데 도움을 줍니다. 이를 통해 delayed generalization의 기저 원인에 대한 인사이트를 제공합니다.

- **Technical Details**: grokking은 과적합(Overfitting) 후 예상치 못한 일반화를 의미합니다. 연구팀은 SC라는 현상이 이러한 일반화가 발생하지 않는 이유 중 하나로 작용한다고 주장합니다. SC는 Softmax 함수의 부동 소수점 오류로 인해 발생하며, 이는 모델의 성능이 정체되거나 오히려 저하되는 결과를 초래합니다. Naïve Loss Minimization (NLM) 방향으로 그래디언트가 정렬되어 가며, 이는 최종적으로 loss를 감소시키지만 SC를 유발합니다.

- **Performance Highlights**: StableMax라는 새로운 활성화 함수는 SC를 방지하며, 정규화 없이도 grokking을 가능하게 합니다. ⟂Grad 옵티마이저는 NLM을 억제하여 grokking 작업에서 빠른 일반화를 촉진합니다. 본 연구의 결과는 정규화가 없는 환경에서도 grokking을 유도할 수 있는 새로운 방법론을 제공하며, 이는 딥러닝에서의 일반화 이해를 한층 더 발전시킵니다.



### Re-ranking the Context for Multimodal Retrieval Augmented Generation (https://arxiv.org/abs/2501.04695)
- **What's New**: 이번 논문은 Retrieval-augmented generation (RAG)을 활용하여 대형 언어 모델(LLMs)을 개선하기 위한 새로운 접근을 제안합니다. 특히, multi-modal RAG 시스템의 정보 검색(retrieval) 과정에서 관련성을 보다 정확하게 평가하기 위해 새롭게 개발한 relevancy score (RS) 모델을 도입했습니다. 기존 CLIP 기반의 방법의 한계를 극복하고 보다 정밀한 검색 결과를 얻기 위해 RS 모델을 사용하여 부정확한 정보를 제거하는 방법에 초점을 맞췄습니다.

- **Technical Details**: RS 모델은 VLM(vision-language models)을 활용하여 사용자 쿼리와 검색된 엔트리 간의 의미적 관계를 학습합니다. 훈련 데이터셋은 인간이 주석을 단 데이터와 ChatGPT로 생성된 합성 쿼리-컨텍스트 쌍을 포함하여 균형 잡힌 triplet 형태로 구성됩니다. RS 모델의 출력은 0에서 1까지의 스칼라 점수로, 높은 점수는 그만큼 쿼리와 높은 관련성을 나타냅니다.

- **Performance Highlights**: COCO 데이터셋을 사용한 평가 결과, RS 기반의 재정렬(re-ranking) 방법이 검색 이미지의 품질을 크게 향상시키고, 더 정확하고 사실 기반의 응답을 생성하는 것으로 나타났습니다. RS는 CLIP에 비해 의미적 유사성을 넘어 정확한 맥락적 관련성을 포착하는 데 뛰어난 성능을 보였으며, 검색 정밀성을 높여 hallucinations(비현실적 결과)를 줄일 수 있는 잠재력을 지니고 있음을 강조합니다.



### A Statistical Theory of Contrastive Pre-training and Multimodal Generative AI (https://arxiv.org/abs/2501.04641)
Comments:
          108 pages

- **What's New**: 이번 논문은 다중 모달 (multi-modal) 생성 AI 시스템의 효과를 보다 철저하게 이해하기 위해, 대조적 사전 훈련 (contrastive pre-training) 프레임워크에 대한 이론적 접근을 개발했습니다. 이 프레임워크는 이미지와 텍스트의 결합 분포를 설명하기 위한 Joint Generative Hierarchical Model을 제안하며, 이것이 대조적 사전 훈련의 성공을 순차적 작업들, 특히 제로샷 분류 (zero-shot classification)와 비전-언어 모델 (vision-language models)에서 어떻게 설명하는지를 밝힙니다.

- **Technical Details**: 논문은 고전적 충분 통계 (sufficient statistics)의 일반화인 근사 충분 통계 (approximate sufficient statistics) 개념을 도입하여, 대조적 사전 훈련 손실의 근접 최소값이 다양한 하위 작업에 적응 가능함을 보여줍니다. 또한, 변환기 (transformers)가 신념 전파 (belief propagation)를 통해 이 모델 내의 관련 함수를 효율적으로 근사할 수 있는 방법을 제시합니다. 이러한 이론적 기반을 바탕으로, 대조적 사전 훈련된 표현을 활용한 다중 모달 학습의 샘플 복잡도 보장을 도출합니다.

- **Performance Highlights**: 수치 시뮬레이션을 통해 이론적인 발견을 검증하였으며, 다양한 다중 모달 작업에서 대조적으로 사전 훈련된 변환기의 강력한 일반화 성능을 입증했습니다. 이러한 결과는 대조적 사전 훈련이 실제 애플리케이션에서도 유용함을 뒷받침합니다. 연구 결과는 학계와 산업계 모두에 중요한 기여를 할 것으로 기대됩니다.



### A Semantic Partitioning Method for Large-Scale Training of Knowledge Graph Embeddings (https://arxiv.org/abs/2501.04613)
Comments:
          Accepted at WWW '23 Companion: Companion Proceedings of the ACM Web Conference 2023

- **What's New**: 이 논문에서는 지식 그래프(knowledge graph)의 온톨로지 정보를 통합하고, 이를 기반으로 클래스에 따라 지식 그래프를 분할하여 더 많은 의미론적 정보를 포함하도록 새로운 알고리즘을 제안합니다. 이 방법은 대규모 지식 그래프 임베딩의 병렬 훈련(parallel training)을 가능하게 합니다. 초기 결과는 이 알고리즘이 여러 인기 있는 벤치마크에서 좋은 성과를 보여주고 있음을 나타냅니다.

- **Technical Details**: 지식 그래프 임베딩(Knowledge Graph Embeddings, KGE) 방법들은 주로 KG의 사실 트리플(fact triplets)에서 학습한 저차원 벡터 표현을 사용합니다. 본 연구는 지식 그래프의 온톨로지(ontology) 정보 외에도 다양한 정보를 포함하여 훈련된 임베딩이 더 많은 의미론적 정보를 담을 수 있도록 합니다. 이러한 방식은 링크 예측(link prediction), 엔터티 정렬(entity alignment) 등 여러 하위 작업(downstream tasks)에서의 성능을 향상시킵니다.

- **Performance Highlights**: 제안된 알고리즘은 기존의 KG 방법들과 비교하여 의미론적 정보의 풍부함을 증대시키며, 향후 하위 작업에서의 성능을 개선할 가능성을 보여줍니다. 특히, 이 기술은 대규모 데이터셋에서의 효율적인 처리와 더불어, 사실 트리플의 불완전성을 해결할 수 있는 방향으로 나아가고 있습니다. 초기 실험 결과는 알고리즘이 여러 평가 지표에서 우수한 성능을 보인다는 것을 확인했습니다.



### Resilient Peer-to-peer Learning based on Adaptive Aggregation (https://arxiv.org/abs/2501.04610)
Comments:
          11 pages

- **What's New**: 본 논문은 분산 피어-투-피어 학습(P2P Learning)에서의 복원력이 저하되는 문제와 비Convex(비볼록) 손실 함수 및 비독립 동일 분포(Non-IID) 데이터로 인해 발생하는 복잡한 도전 과제를 해결하고자 하는 새로운 접근법을 제시합니다. 저자들은 적대적인 작업자(adversarial workers)가 네트워크에 악의적인 정보를 주입하는 위협을 최소화하며, 이를 통해 피어의 학습 과정을 유사하게 유지하는 복원력 있는 집계 기법을 도입했습니다.

- **Technical Details**: 이 연구에서 제안하는 집계 방법은 각 작업자의 개인 데이터와 이웃의 모델을 기반으로 최적화된 손실 함수(Loss Function)를 계산하여 가중치를 결정합니다. 이를 통해 데이터 개인 정보 보호 우려를 해소하면서도 비볼록 손실 함수와 비독립 동일 분포 데이터를 처리할 수 있도록 설계되었습니다. 이론적 분석 결과는 제안된 방법이 비볼록 손실 함수 및 비IID 데이터 분포에서 매개변수가 수렴한다는 것을 입증합니다.

- **Performance Highlights**: 세 가지 서로 다른 머신러닝 작업에 대한 실험 평가 결과, 제안된 알고리즘이 기존 복원 알고리즘들과 비교해 보다 높은 정확도를 달성하는 것으로 나타났습니다. 다양한 공격 모델을 포함한 실험 결과는 제안된 접근 방식이 적대적인 이웃의 영향을 효과적으로 견디면서도 성능을 유지한다는 것을 보여줍니다.



### Federated-Continual Dynamic Segmentation of Histopathology guided by Barlow Continuity (https://arxiv.org/abs/2501.04588)
- **What's New**: 이 연구에서는 Federated Learning과 Continual Learning에서 발생하는 Client Drift와 Catastrophic Forgetting 문제를 동시에 해결할 수 있는 방법을 제안했습니다. 제안된 Dynamic Barlow Continuity(DynBC) 방법은 공공 참조 데이터셋에서 클라이언트 업데이트를 평가하여 훈련 과정을 조정합니다. 이를 통해 시공간 불변성을 가진 모델을 구축하고, histopathology 데이터셋에서 성능을 크게 향상시킬 수 있었습니다.

- **Technical Details**: DynBC는 Barlow Twins를 기반으로 한 평가 방법으로, 다양한 데이터를 기반으로 두 모델 상태의 변화를 비교하여 spatio-temporal continuity를 평가합니다. 이 방법은 기존 중앙 집중식 훈련에서 성공적으로 사용되었으며, 이제는 Federated Learning 및 Continual Learning 환경에서도 적용됩니다. 평가 프로세스에서 프라이버시를 유지하기 위해 별도의 공공 데이터셋에서 샘플링한 데이터를 사용하여 모델 반응을 측정합니다.

- **Performance Highlights**: BCSS와 Semicol 데이터셋에서 실험을 통해 DynBC 방법이 Client Drift를 15.8%에서 71.6%로, Catastrophic Forgetting을 42.5%에서 62.8%로 향상시키는 것을 입증했습니다. 이러한 결과는 제안한 방법이 AI 지원 histopathology 분야에서 모델 성능을 개선할 수 있음을 알립니다. 최종적으로, 이 방법은 향후 다양한 의료 AI 시스템에 적용될 가능성을 제시합니다.



### Large-Scale Spectral Graph Neural Networks via Laplacian Sparsification: Technical Repor (https://arxiv.org/abs/2501.04570)
- **What's New**: 이번 연구에서 제안하는 'Spectral Graph Neural Networks with Laplacian Sparsification (SGNN-LS)'는 스펙트럴 그래프 신경망의 확장성을 개선하는 새로운 방법론입니다. 이 방법은 라플라시안 희소화 기술을 활용하여 스펙트럴 GNN의 전파 패턴을 근사화하는 데 중점을 두고 있습니다. SGNN-LS는 고차원의 입력 특성을 처리할 수 있도록 선형 계층을 입력 노드 특징에 적용할 수 있게 합니다.

- **Technical Details**: SGNN-LS는 스펙트럴 GNN의 전파 행렬을 ε-sparsifiers 형태로 도출하는 전략을 제공합니다. 이는 그래프 전파의 Multi-hop 이웃들을 효과적으로 연결하면서도, 그래프 전파 과정의 통합을 유지합니다. 제안된 방법은 비직선적 단계를 포함하며, 고차원 특성에 대한 고유한 처리 능력을 갖추고 있습니다.

- **Performance Highlights**: 대규모 그래프 데이터셋에서의 실험을 통해 SGNN-LS의 효율성과 효과성이 입증되었습니다. 특히 Ogbn-papers100M(1.11억 노드, 16억 엣지) 및 MAG-scholar-C(280만 특징) 데이터셋에서 기존의 근사화된 베이스 모델들과 비교하여 우수한 성능을 나타냈습니다. 이 연구는 스펙트럴 GNN의 확장성 문제를 해결하는 중요한 기초 작업으로 평가받고 있습니다.



### Medical artificial intelligence toolbox (MAIT): an explainable machine learning framework for binary classification, survival modelling, and regression analyses (https://arxiv.org/abs/2501.04547)
Comments:
          14 pages, 2 figures, 1 table

- **What's New**: 이번 논문에서는 의료 분야의 기계 학습을 위한 새로운 도구인 Medical Artificial Intelligence Toolbox (MAIT)를 소개합니다. MAIT는 이진 분류(binary classification), 회귀(regression), 생존 모델(survival models)을 개발하고 평가하는 데 사용되는 설명 가능한(open-source) Python 파이프라인으로, 통합된 모델 개발 및 해석을 용이하게 하여 기계 학습 접근 방식을 통합할 수 있는 기반을 제공합니다.

- **Technical Details**: MAIT는 높은 차원(high dimensionality), 클래스 불균형(class imbalance), 혼합 변수 타입(mixed variable types) 및 결측치(missingness)와 같은 주요 과제를 해결합니다. 이 툴박스는 초보자를 위한 자동 설정(automated configurations)과 전문가를 위한 커스터마이즈 가능한 소스 코드(customizable source code)를 제공하여 통합된 점수 계산(unified scoring, 예: SHAP)을 통해 특징 중요성을 발견할 수 있도록 지원합니다. 또한, 이진 분류의 확률 임계값(threshold) 미세 조정(fine-tuning)과 같은 새로운 기술을 제안합니다.

- **Performance Highlights**: MAIT는 데이터 제약(data constraints) 및 연구 설계(study designs)에 적응할 수 있는 여러 기능들을 제공합니다. 모델 해석(model interpretation)을 위한 향상된 시각화(visualizations)와 반감독 학습(semi-supervised learning)을 통한 검열(censoring) 처리 등의 새로운 기법을 도입하여 의료 연구에서 기계 학습 모델의 구현 및 해석을 개선할 수 있게 합니다. GitHub에서는 네 개의 오픈 액세스 데이터 세트를 사용한 자세한 튜토리얼이 제공되어 MAIT의 사용 방법을 시범적으로 안내합니다.



### HypeRL: Parameter-Informed Reinforcement Learning for Parametric PDEs (https://arxiv.org/abs/2501.04538)
- **What's New**: 본 논문에서는 파라메트릭 부분 미분 방정식(PDE)의 최적 제어를 위한 새로운 일반 목적 강화 학습 전략인 HypeRL을 제안합니다. 기존의 전통적인 수치 기법과 비교하여 HypeRL은 고차원의 매개변수가 분포된 복잡한 문제에서도 효율적으로 동작할 수 있도록 설계되었습니다. 특히, HypeRL은 액터-크리틱(actor-critic) 방법론을 사용하여 매개변수에 따라 변화하는 피드백 제어 전략을 학습하고, 하이퍼네트워크(hypernetwork)를 통해 정책 정책(policy) 및 가치 함수(value function)를 최적화합니다.

- **Technical Details**: HypeRL은 기존의 강화 학습에서 나타나는 샘플 비효율(sample inefficiency)과 제한된 일반화(generalization) 문제를 해결하기 위해 개발된 프레임워크입니다. 논문에서는 PDE 파라미터를 분석하여 강화 학습 정책을 매개변수에 의존하도록 세분화하는 접근 방식을 채택하였으며, 하이퍼네트워크를 통해 정책의 가중치와 편향을 학습하는 과정을 포함합니다. 이는 데이터 양을 줄이고 unseen scenarios 에서의 적응력을 강화하는 데 기여합니다.

- **Performance Highlights**: 제안된 HypeRL은 1차원 Kuramoto-Sivashinsky 방정식과 2차원 Navier-Stokes 방정식이라는 두 개의 PDE 제약 최적 제어 벤치마크에서 성능 검증을 수행하였습니다. 결과적으로, 하이퍼네트워크를 활용한 매개변수 정보를 기반으로 하는 DRL이 매개변수에 무관한 접근보다 월등한 성능을 발휘함을 보여주었습니다. 이는 데이터 효율성과 제어 정책의 일반화 능력 개선에 중요한 요소로 작용합니다.



### A Plug-and-Play Bregman ADMM Module for Inferring Event Branches in Temporal Point Processes (https://arxiv.org/abs/2501.04529)
Comments:
          Accepted at AAAI 2025

- **What's New**: 이번 연구에서는 Bregman ADMM (BADMM) 알고리즘 기반의 새로운 플러그 앤 플레이 모듈을 설계하여, 시간적 포인트 프로세스(temporal point process, TPP)의 최대 우도 추정(maximum likelihood estimation) 프레임워크 내에서 사건(event) 시퀀스에 관련된 사건 분기(event branch)를 추론합니다. 이는 사건과 사건 간의 유발 관계(triggering relations)를 표현하는 구조화된 이벤트 분기 프로세스(hidden and structured event branching process)에 대한 통찰을 제공합니다.

- **Technical Details**: 사건 분기 추론을 희소성(sparse) 및 낮은 계급(low-rank) 제약 조건 하에 사건 전이 행렬(event transition matrix)을 최적화하는 문제로 공식화합니다. 이 최적화 문제는 서브스페이스 클러스터링(subspace clustering)과 희소 그룹 라소(sparse group-lasso)를 기반으로 구현될 수 있으며, Bregman ADMM 알고리즘을 통해 해결됩니다. 특히, 이는 기존 TPP 모델이나 학습 패러다임에 통합될 수 있습니다.

- **Performance Highlights**: 실험 결과, BADMM 모듈을 기존 TPP 모델에 추가하면 모델 성능이 향상되고 해석 가능한 구조화된 사건 분기를 제공할 수 있음을 보여주었습니다. 저자는 이를 통해 고립 사건(isolated events)과 다수의 후속 사건을 유발하는 주요 사건(key events)을 추론할 수 있는 학습된 사건 전이 행렬(learned event transition matrices)을 도출할 수 있음을 입증했습니다. 코드도 제공되어 공개된 링크를 통해 접근할 수 있습니다.



### Towards a Problem-Oriented Domain Adaptation Framework for Machine Learning (https://arxiv.org/abs/2501.04528)
- **What's New**: 이번 논문에서는 도메인 적응(Domain Adaptation) 문제를 해결하기 위해 문제 지향적인 프레임워크를 개발하였습니다. 이 프레임워크는 다양한 도메인 적응 시나리오를 구분하고, 각 시나리오에 대한 해결책을 제시하며, 문제의 특성을 파악하는 가이드를 제공합니다. 평가를 통해 프레임워크가 도메인 적응 문제를 효과적으로 설명할 수 있는 능력을 갖추고 있음을 입증하였습니다.

- **Technical Details**: 기계 학습(Machine Learning)에서 도메인 적응은 서로 다른 특성 분포를 가진 두 문제 간의 간극을 메우는 접근 방식을 의미합니다. 이를 통해 소스 도메인(source domain)에서 타겟 도메인(target domain)으로의 지식 이관이 가능합니다. 이 연구는 디자인 과학 연구(Design Science Research) 패러다임을 따르며, 이론적 엄밀성뿐만 아니라 실용적 함의도 강조합니다.

- **Performance Highlights**: 제시된 프레임워크는 인공지능 연구자 및 실무자가 도메인 적응 사용 시 방향성을 제공하며, 다양한 데이터 세트를 통해 평가되었습니다. 프레임워크의 개발 과정에서, 인공 데이터 및 실제 데이터셋에 대한 실험을 통해 성공적인 성과를 나타냈습니다. 이러한 평가 결과는 도메인 간 전이 효과(positive transfer)를 보장할 수 있는 접근법이 존재함을 설명합니다.



### Towards Fair Class-wise Robustness: Class Optimal Distribution Adversarial Training (https://arxiv.org/abs/2501.04527)
- **What's New**: 이 논문에서는 Class Optimal Distribution Adversarial Training (CODAT)라는 새로운 min-max 훈련 프레임워크를 제안하였습니다. CODAT는 distributionally robust optimization (DRO) 이론을 활용하여 클래스별 가중치 공간을 철저히 탐색하고, 이론적 보장을 통해 최적의 가중치를 식별할 수 있는 방법을 제공합니다. 또한 내부 극대화 문제에 대한 닫힌 형태의 최적 해를 도출하고, 이를 통해 가중치와 모델 매개변수의 일관된 최적화 방향을 보장하는 결정론적 동등 목표 함수를 얻습니다.

- **Technical Details**: CODAT는 min-max 최적화 문제로 형식화될 수 있으며, 내부 극대화 목표는 클래스별 기대 리스크를 극대화하는 데 목표를 두고 있습니다. CODAT는 worst-case class adversarial distribution을 학습하기 위해 클래스 적대적 분포 공간을 완전히 탐색할 수 있도록 설계되었습니다. 또한, 가중치와 모델의 최적화 방향을 일관되게 유지하기 위해 내부 극대화 문제에 대한 닫힌 형태의 최적 해를 파생하고 원래 모델 목표에 통합된 결정론적 동등 목표 함수를 제공합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 수행된 실험 결과, CODAT는 기존의 최첨단 방법들을 초과하는 성능을 보였으며, 모델의 robust fairness를 효과적으로 개선할 수 있음을 입증하였습니다. 이 연구에서 제안된 fairness elasticity coefficient는 알고리즘의 견고성과 공정성을 모두 평가하는 데 사용됩니다. CODAT는 복잡한 분야에서도 적용 가능한 새로운 접근법으로, 특히 보안이 중요한 분야에서 악의적인 공격으로부터의 저항력을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### Histogram-Equalized Quantization for logic-gated Residual Neural Networks (https://arxiv.org/abs/2501.04517)
Comments:
          Published at IEEE ISCAS 2022

- **What's New**: 이 논문에서는 Histogram-Equalized Quantization (HEQ)라는 새로운 방법을 제안합니다. HEQ는 양자화(threshold)를 데이터나 모델 손실(loss)에 따라 조정하여 높은 정확도를 제공합니다. HEQ는 고유한 단계 크기 최적화를 사용하여 양자화 임계값을 자동으로 조정하는 적응형 프레임워크입니다. 실험 결과 HEQ는 CIFAR-10에서 최첨단 성능을 달성했으며, STL-10 데이터셋에서도 높은 정확도로 저 하드웨어 복잡도로 논리 게이트 회귀 네트워크를 학습할 수 있음을 보여줍니다.

- **Technical Details**: 본 논문에서는 제한된 범위의 이산 값으로 선형 대칭 양자화(linear symmetric quantization)를 다루고 있습니다. 기존 연구에서는 양자화 값을 클리핑(clipping) 과정에서 보존하지만, HEQ는 모든 양자화 값을 [-1, +1] 범위로 수축시킵니다. 또한, HEQ는 양자화 진행 중 가중치의 히스토그램(histogram) 정보를 기반으로 단계 크기(s)가 자동으로 조정되도록 돕습니다. 이를 통해 데이터 표현 공간에서의 양자화 값의 균형 있는 활용을 최적화할 수 있습니다.

- **Performance Highlights**: HEQ는 다양한 신경망 토폴로지에서 기존 방법보다 더 나은 정확도를 기록했습니다. 특히, OR 및 MUX 논리 게이트를 포함한 양자화 모델의 훈련에서 효과적으로 작동하여 하드웨어 매핑을 단순화했습니다. 최적화된 양자화는 네트워크의 정확성을 개선하고 불필요한 하드웨어 비용을 줄이는 데 기여하며, HEQ는 저비용 하드웨어에서 추론을 최적화하는 차별화된 접근법을 제시합니다.



### Integrating remote sensing data assimilation, deep learning and large language model for interactive wheat breeding yield prediction (https://arxiv.org/abs/2501.04487)
- **What's New**: 이 연구는 농작물 수확량 예측을 위한 하이브리드 방법과 도구를 도입합니다. 이 도구는 품종 육종가가 대화형으로 wheat yield (밀 수확량)를 예측할 수 있도록 설계되었습니다. 새로운 데이터 동화 알고리즘을 WOFOST 모델에 통합하여 처리하며, 사용자 친화적이고 지속적인 데이터 업데이트를 지원합니다.

- **Technical Details**: 연구에서는 먼저 leaf area index (엽면적지수)를 WOFOST 모델에 동화시키는 데이터 동화 알고리즘을 활용합니다. 그런 다음, 이 동화 과정에서 선택된 출력과 remote sensing (원거리 감지) 결과를 활용해 time-series temporal fusion transformer model (시간 시계열 융합 변환기 모델)을 구동하여 wheat yield를 예측합니다. 이 모든 과정은 대화형 웹 도구를 통해 이루어집니다.

- **Performance Highlights**: 개발된 도구는 다원 소스 데이터를 통합하여 육종 의사 결정을 지원하는 기능을 갖추고 있습니다. 이 혁신적인 접근 방식은 품종 육종 과정에서의 고수확량 자재를 더 빨리 식별할 수 있도록 하고, 육종 효율성을 향상시키며, 보다 과학적이고 스마트한 육종 결정을 가능하게 합니다.



### Safe Reinforcement Learning with Minimal Supervision (https://arxiv.org/abs/2501.04481)
Comments:
          Initially submitted to ICML 2023

- **What's New**: 본 논문은 안전한 강화 학습(safe reinforcement learning, safe-RL)을 위한 새로운 접근법을 탐구합니다. 기존 안전 학습 방법은 주로 사람의 시연(demonstration)을 필요로 하였으나, 본 연구에서는 비지도 강화 학습(unsupervised RL)을 활용하여 데이터 수집 절차를 제안합니다. 이를 통해 복잡한 정책을 손쉬운 데이터 수집 없이 배울 수 있는 방법을 제공합니다.

- **Technical Details**: 연구에서는 에이전트가 안전하게 온라인 탐색을 할 수 있도록 안전 집합(safe-set)을 학습하는 것에 집중합니다. 특히, 시연이 거의 없거나 전혀 없는 작업을 다루며, 최적의 안전-RL 정책을 학습할 수 있는 기초 자료를 마련하는 것이 중요하다고 강조합니다. 이 과정에서 'optimistic forgetting'이라는 새로운 온라인 안전-RL 방법을 제안합니다.

- **Performance Highlights**: 방법론적 연구 결과는 비지도 학습의 데이터 수집 접근법이 에이전트에게 안전한 RL 정책을 보다 효과적으로 학습시키는 데 기여할 수 있음을 보여줍니다. 에이전트가 온라인에서 최적의 안전-RL 정책을 학습하기 위해서는 충분한 데이터의 양과 질을 균형 있게 제공해야 한다는 것도 발견되었습니다. 이러한 접근법을 통해 다양한 환경과 과제에서의 범용적인 안전-RL 정책 학습 가능성을 제시합니다.



### Regularising NARX models with multi-task learning (https://arxiv.org/abs/2501.04470)
- **What's New**: 본 연구에서는 Nonlinear Auto-Regressive with eXogenous inputs (NARX) 모델의 오버피팅(overfitting) 문제를 해결하기 위해 Multi-task Learning (MTL) 접근 방식을 도입하였습니다. 이러한 MTL 기법을 활용하면 리드 타임(lead time) 예측을 통해 현재 출력 예측을 정규화할 수 있으며, 이는 높은 노이즈 조건에서도 Normalised Mean Square Error (NMSE) 값을 낮추는 효과가 있습니다.

- **Technical Details**: NARX 신경망은 이전의 출력과 외부 입력 변수를 모두 고려하여 현재 출력을 예측하는 방식으로 작동합니다. MTL을 적용하여 NARX 신경망의 성능을 극대화하기 위해 비운동적 특징(non-operational features)을 추가하여 알고리즘의 일반화를 향상시키는 방법을 제안하고 있습니다. 연구에 사용된 사례 연구는 단일 자유도(SDOF) Duffing 진동기로, 비교적 논리적인 이동 방정식이 사용되었습니다.

- **Performance Highlights**: MT-NARX 모델을 통해 MTL 접근 방식을 적용했을 때, 기존의 단일학습자 모델에 비해 예측의 정확성을 높이고, 고잡음 환경에서도 모델의 성능을 검증할 수 있었습니다. 이로 인해 구조적 데이터셋에 적용할 수 있는 다양한 MTL 방식의 유용성이 입증되었습니다.



### Gradient Purification: Defense Against Poisoning Attack in Decentralized Federated Learning (https://arxiv.org/abs/2501.04453)
- **What's New**: 본 논문에서는 분산 학습의 일종인 분산 연합 학습(DFL)에서의 중독 공격(poisoning attack)에 대한 새로운 방어 기법인 GPD(Gradient Purification Defense)를 제안합니다. 기존의 방어 방법이 악성 기여자를 검출하여 그들의 기여를 전부 무효화하는 반면, GPD는 악성 기여자에게서 오는 해악을 줄이면서도 그들의 유용한 기여를 유지하도록 설계되었습니다. 이로 인해 모델의 정확도를 높이는 것을 목표로 합니다.

- **Technical Details**: GPD는 각 양성 클라이언트가 이웃 클라이언트로부터 수신한 과거의 기여도를 기록하는 변수를 유지하게 합니다. 이 기록 변수를 통해 양성 클라이언트는 악성 클라이언트를 정확하게 식별하고, 단일 반복(iteration) 내에서 모든 악성 기여를 신속히 완화할 수 있는 능력을 갖추게 됩니다. 그 후, GPD는 양성 기여자만의 기여를 기반으로 모델 가중치를 최적화하여 좀 더 정확한 모델을 생성합니다.

- **Performance Highlights**: GPD는 다양한 데이터셋에 대한 광범위한 실험을 통해 기존 최선의 방어 기법보다 평균 8% 더 높은 정확도를 기록했습니다. GPD는 iid 및 non-iid 데이터 분포 하에서도 중독 공격을 완화할 수 있는 능력을 보여주며, 모델의 정확도를 극대화하는 데 성공했습니다. 이는 GPD 방식이 기여자 간의 유용한 정보를 유지하면서도 악성 영향을 효과적으로 제거한다는 것을 의미합니다.



### Motif Discovery Framework for Psychiatric EEG Data Classification (https://arxiv.org/abs/2501.04441)
- **What's New**: 이 연구는 우울증 치료의 초기 반응을 EEG 신호를 바탕으로 예측하는 새로운 모티프 기반 프레임워크를 제안합니다. 치료 시작 7일째에 수집된 EEG 데이터를 이용해 우울증 반응자와 비반응자를 분류하는 접근법은 기존의 진단 방법과 차별화 됩니다. 이를 통해 환자의 치료 반응을 더 빠르게 파악할 수 있어, 감정적 및 경제적 부담을 줄일 가능성이 높습니다.

- **Technical Details**: 우리는 EEG 신호에서 모티프(motif)를 추출하여 우울증 치료 반응자를 분류하는 문제를 해결합니다. 모티프 발견(motif discovery) 기법을 통해 짧은 시간 시퀀스에서 발생하는 반복 패턴을 식별하고, 이를 특징으로 활용하여 분류기를 구축합니다. 이 과정에서 EEG의 알파, 베타, 세타 주파수 대역을 각각 분석하여 최종적으로 각 클래스 구분에서 높은 가능성을 가진 모티프를 선정합니다.

- **Performance Highlights**: 우리는 MDD, 조현병, 불응성 발작, 알츠하이머 및 치매 환자의 EEG 데이터를 포함한 여러 정신과 EEG 데이터 세트에서 높은 분류 정확도를 달성했습니다. 연구 결과는 EEG의 동적 특성이 의사들의 진단 및 치료 반응 예측에서 효율적으로 활용될 수 있음을 보여줍니다. 우리의 방법은 우울증 진단 및 치료 반응 예측에 있어 모티프를 처음으로 적용한 것으로, 향후 정신과 진료의 개선에 기여할 수 있을 것으로 보입니다.



### Federated Fine-Tuning of LLMs: Framework Comparison and Research Directions (https://arxiv.org/abs/2501.04436)
- **What's New**: 이 논문은 분산된 개인 데이터셋을 사용하여 프리트레인된 대형 언어 모델(LLMs)의 파인튜닝을 위한 연합 학습(Federated Learning, FL)의 새로운 접근 방식을 제시합니다. 세 가지 고급 연합 LLM 프레임워크(FedLLM)인 FedLLMs, KD-FedLLMs, Split-FedLLMs를 비교하며, 각각의 모델 업데이트 방법과 지식 전달 메커니즘의 차별성을 강조합니다. 이러한 분석을 통해 FL 환경에서의 LLM 파인튜닝의 효율성을 극대화할 수 있는 기회를 제시하고 있습니다.

- **Technical Details**: 이 논문에서는 PEFT(Parameters Efficient Fine-Tuning) 기법을 적용하여 자원 제약이 있는 연합 시나리오에서 LLM을 파인튜닝하는 세 가지 프레임워크를 설명합니다. 첫 번째 프레임워크, FedLLMs는 클라이언트가 모델 업데이트를 서버에 직접 제출하는 방식을 사용합니다. 두 번째 프레임워크, KD-FedLLMs는 지식 증류(Knowledge Distillation, KD)를 활용하여 클라이언트와 서버 간의 지식 공유를 용이하게 합니다. 마지막으로 Split-FedLLMs는 모델을 클라이언트와 서버 간에 나누어 연산 부하를 조절하는 접근 방식을 채택하고 있습니다.

- **Performance Highlights**: 이 연구는 각 프레임워크의 모델 정확도, 통신 비용, 클라이언트 측 계산 부하와 같은 주요 성능 지표를 기준으로 평가하였습니다. 이를 통해 서로 다른 방식의 파인튜닝 방법 간의 무역 가지를 이해할 수 있으며, 실제 연합 학습 시나리오에서 각 프레임워크의 적합성을 파악하는 데 기여합니다. 또한, 실용적인 사용 사례를 통해 다양한 설정에서 이들 프레임워크의 성능을 비교하고 실제 적용 가능성을 시연하고 있습니다.



### Dual-Force: Enhanced Offline Diversity Maximization under Imitation Constraints (https://arxiv.org/abs/2501.04426)
- **What's New**: 이 논문에서는 오프라인 환경에서 다양성을 극대화하기 위한 새로운 알고리즘인 Dual-Force를 소개합니다. 이 알고리즘은 Van der Waals (VdW) 힘과 Successor features를 기반으로 하여 과거에 사용된 skill discriminator를 학습할 필요를 없앱니다. 또한, Functional Reward Encoding (FRE)을 활용하여 비국소성 보상을 효과적으로 처리할 수 있습니다.

- **Technical Details**: Dual-Force 알고리즘은 강화 학습 (Reinforcement Learning)에서의 Fenchel 이중성 이론을 적용하여 오프라인 데이터를 활용합니다. 이 방법은 VdW 힘을 사용하여 다양성을 확장하며 skill discriminator를 학습할 필요성을 제거합니다. 각 비국소성 보상에 대해 관련된 skill을 FRE 잠재 임베딩을 통해 쉽게 회상할 수 있습니다.

- **Performance Highlights**: 우리는 Dual-Force의 유효성을 12-DoF의 사족 보행 로봇 Solo12에서 수집된 두 가지 오프라인 데이터셋을 통해 입증합니다. 이 알고리즘은 다양한 행동을 효율적이고 견고하게 재현하며, 학습 데이터를 기반으로 목표 전문가 상태 점유를 모방합니다. 이로 인해 학습된 기술 세트가 크게 확장됩니다.



### Risk-averse policies for natural gas futures trading using distributional reinforcement learning (https://arxiv.org/abs/2501.04421)
- **What's New**: 최근 금융 시장의 불안정성을 고려하여, 본 논문에서는 분포적 강화 학습 (Distributional Reinforcement Learning, RL) 알고리즘을 활용한 위험 회피 전략 개발의 가능성을 탐구한다. 이는 Categorical Deep Q-Network (C51), Quantile Regression Deep Q-Network (QR-DQN), Implicit Quantile Network (IQN) 세 가지 알고리즘의 성능을 가스 선물 거래에 적용하여 검증하는 연구이다. 특히, 이 알고리즘들은 거래 분야에서 처음으로 적용되며, 기존의 기계 학습 모델과 비교하여 성과를 분석한다.

- **Technical Details**: 김치 강화 학습 알고리즘은 누적 보상의 전체 분포를 모델링하며, 이로 인해 리스크를 고려한 정책을 개발할 수 있는 잠재력을 지닌다. 본 연구에서는 CVaR (Conditional Value-at-Risk)을 최대화하도록 C51 및 IQN을 훈련시켜 조정 가능한 위험 회피 정책을 생성하는 과정을 상세히 설명한다. 또한, 저신뢰도 CVaR에서는 위험 회피가 증가하고, 높은 신뢰도에서는 감소하는 경향을 보이는 것으로 나타났으며, QR-DQN은 예측 가능성이 낮은 행동을 보인다.

- **Performance Highlights**: 연구 결과, 분포적 RL 알고리즘은 고전적인 RL 방법에 비해 32% 이상의 성능 개선을 보여주었다. 이를 통해 불확실한 시장에서도 적응 가능하고 위험 회피적인 거래 전략 개발에 강력한 잠재력을 발휘할 수 있음을 밝혔다. 토대로, 이러한 분석은 가스 선물 거래와 같은 변동성이 큰 자산에서 분포적 RL 접근법의 강점을 강조한다.



### Lossless Privacy-Preserving Aggregation for Decentralized Federated Learning (https://arxiv.org/abs/2501.04409)
- **What's New**: 이번 연구에서는 손실 없는 프라이버시 보호 집계 규칙인 LPPA( Lossless Privacy-Preserving Aggregation)를 제안하여 분산 연합 학습(DFL)에서의 그래디언트 보호를 향상 시킵니다. LPPA는 전송된 그래디언트에 수신된 노이즈와 전송된 노이즈의 차이를 미세하게 주입하여 데이터 유출을 효과적으로 방지합니다. 이 방식은 모델의 예측 정확도를 손상시키지 않으면서 파트너의 난수를 통합하여 대칭성을 유지하는데 중점을 둡니다.

- **Technical Details**: LPPA는 그래픽 이론의 흐름 보존 이론에 영감을 받아 DFL 연결 토폴로지 내에서 발생하는 전반적인 노이즈 유입과 유출이 같다는 것을 활용합니다. 이를 통해 추가적인 노이즈 교환 라운드를 도입하여 DFL 클라이언트를 미리 혼란시킴으로써 모델 가중치와 그래디언트의 감수성을 줄였습니다. 또한, 각 클라이언트는 전송된 노이즈 차이를 사용하여 로컬 그래디언트에 주입함으로써 더 강한 무작위성을 촉진하고 전세계적인 노이즈 제거를 보장합니다.

- **Performance Highlights**: LPPA는 기존의 노이즈 추가 방법에 비해 이론적으로 22\sqrt{2}배 더 높은 프라이버시 보호 능력을 가지고 있으며, 노이즈 주입 없이 표준 DFL 집계와 비교할 때 손실 없는 모델 정확도를 보장합니다. 실험 결과 LPPA는 노이즈 추가 방식에 비해 평균 13%의 정확도 향상을 이루어냈으며, 원 데이터 보호와 손실 없는 모델 정확도를 보장하는 데 효과적임을 입증하였습니다.



### Rising Rested MAB with Linear Drif (https://arxiv.org/abs/2501.04403)
- **What's New**: 이번 논문에서는 비정상적인 다중팔 밴딧(non-stationary multi-arm bandit, MAB) 모델에서 기대 보상이 실행 횟수(수행 수)의 선형 함수로 나타나는 문제를 다룹니다. 연구의 주요 결과로는 $	ilde{	heta}(T^{4/5}K^{3/5})$의 밀접한 후회 경계(regret bound)를 제공하며, 이는 보상의 선형 드리프트(linear drift)에 대한 알려지지 않은 매개 변수화(unknown parametrization)에 의존하는 실전 종속 후회 경계(instance dependent regret bounds)를 유도합니다.

- **Technical Details**: 비정상적 MAB에서 중요한 문제는 팔을 고르는 과정에서 탐색(exploration)과 활용(exploitation) 간의 균형을 유지하는 것입니다. 이 연구는 상승한 휴식형 MAB(Rising Rested MAB)에서의 후회(regret)를 연구하며, 여기서 각 팔의 기대 보상 함수는 팔이 수행된 횟수에 대해 선형(linear)입니다. 알고리즘 R-ed-EE, R-ed-AE 및 HR-re-AE가 설계되어 각각 후회 경계를 제공합니다.

- **Performance Highlights**: 연구 결과, 상승한 휴식형 MAB에서 동적 후회(dynamic regret)와 정적 후회(static regret)가 동일하다는 것을 보여줍니다. 또한, R-ed-EE 알고리즘은 O(T^{4/5}(	ext{Φ}K)^{3/5}	ext{ln}(	ext{Φ}KT)^{1/5})의 후회 경계로 제시되며, HR-re-AE 알고리즘은 worse case의 O(T^{4/5}(	ext{Φ}K)^{3/5}	ext{ln}(	ext{Φ}KT^{2})^{1/5})를 추가적으로 제공합니다. 이러한 결과는 비정상적 보상이 발생하는 MAB의 새로운 이론적 이해를 제공합니다.



### Tracking UWB Devices Through Radio Frequency Fingerprinting Is Possib (https://arxiv.org/abs/2501.04401)
Comments:
          conference ICNC'25, 7 pages, 7 figures

- **What's New**: 이 논문은 Ultra-Wideband (UWB) 기술에 Radio Frequency Fingerprinting (RFF)를 적용할 가능성을 탐구합니다. 기존의 연구들은 Wi-Fi, 5G, Bluetooth 등 다양한 무선 도메인에서 효과적으로 분류된 장비에 대해 다루었으나, UWB 신호에 대한 RFF 연구는 부족했습니다. 이 연구는 신호 전송 방식에 따라 장치의 고유한 하드웨어 지문을 추출하는 데 연구의 초점을 맞추고 있습니다.

- **Technical Details**: UWB 기술은 높은 해상도와 짧은 범위의 로컬라이제이션을 지원하여 데이터를 높은 속도로 전송할 수 있으며, 데이터 수집 과정에서 Qorvo DWM3001CDK 보드를 사용했습니다. Controlled 실험을 통해 UWB 신호에서 지문 정보를 추출하기 위해 다양한 깊이의 딥러닝 파이프라인을 개발했습니다. 본 논문에서는 장치 위치 변화에 대한 RFF 감지의 견고성을 평가하고 통제된 변수를 사용하여 데이터 수집 캠페인을 수행했습니다.

- **Performance Highlights**: 실험 결과, 안정적인 조건에서 RFF는 99% 이상의 정확도를 달성하였고, 환경 변화에 따라 정확도가 감소하긴 했으나 훈련되지 않은 위치에서도 최대 76%의 정확도로 장치 식별이 가능했습니다. 이는 UWB가 채택된 스마트시티 응용 프로그램에서 보안과 프라이버시를 동시에 고려해야 함을 시사합니다. UWB 기술의 성과와 함께, RFF의 개선된 방법론을 통해 딥러닝 기반의 연구가 이제껏 발견되지 않은 가능성을 열게 되었습니다.



### On Computational Limits and Provably Efficient Criteria of Visual Autoregressive Models: A Fine-Grained Complexity Analysis (https://arxiv.org/abs/2501.04377)
- **What's New**: 최근 Visual Autoregressive (VAR) 모델이 이미지 생성 분야에서 혁신적인 발전을 가져왔습니다. 이 모델은 coarse-to-fine "next-scale prediction" 접근 방식을 통해 확장 가능한 방식으로 이미지 생성을 가능하게 합니다. 그러나 VAR 모델의 현재 최첨단 알고리즘은 O(n^4)의 시간 복잡도를 가지며, 이는 계산 효율성이 낮습니다. 따라서 본 연구는 VAR 모델의 계산 한계와 효율성을 분석하여 개선 방안을 제시합니다.

- **Technical Details**: VAR 모델의 효율성을 갖추기 위해서는 입력 행렬의 노름이 특정 활용 범위 아래에 있어야 하며, 이 기준을 초과할 경우 진정한 서브-쿼틱(sub-quartic) 시간 알고리즘을 설계하는 것이 불가능한 것으로 확인했습니다. 본 연구에서는 Strong Exponential Time Hypothesis (SETH)를 바탕으로 VAR 모델의 계산 성능을 평가하는 새로운 기준을 설정하였습니다. 이를 통해 계산 비용을 절감할 수 있는 경량화된 구조도 제안합니다.

- **Performance Highlights**: 본 연구의 기여는 VAR 모델의 계산 시간을 O(n^4)보다 더 빠르게 수행할 수 있는 조건을 제시한 것입니다. 입력 행렬의 요소가 특정 임계값을 초과하지 않을 경우 효율적인 알고리즘을 통해 거의 저차 제곱 시간 복잡도 O(n^{2+o(1)})로 VAR 모델을 근사할 수 있음을 보여줍니다. 이러한 발견은 VAR 모델의 이론적 발전을 위한 기초가 되며, 향후 스케일러블한 이미지 생성기를 발전시킬 수 있는 길을 열어줄 것입니다.



### Navigating the Designs of Privacy-Preserving Fine-tuning for Large Language Models (https://arxiv.org/abs/2501.04323)
Comments:
          4 pages, 2 figures

- **What's New**: 본 논문에서는 GuardedTuning이라는 새로운 접근 방식을 제안합니다. 이는 시스템 아키텍처, 개인 정보 보호 향상 방법 및 최신 컴퓨팅 기법의 혁신적인 조합으로 구성됩니다. 이러한 설계는 모델 유용성, 개인 정보 보호 보장 및 비용 간의 뚜렷한 trade-off를 반영합니다.

- **Technical Details**: GuardedTuning은 클라이언트와 서버 간의 중간 값이나 압축된 모델 구성 요소만을 교환하여 데이터 재구성 공격(DRA)을 방지하는 구조를 갖고 있습니다. 본 논문은 세 가지 범주의 메트릭을 통해 다양한 요구 사항을 충족하는 설계를 평가합니다: 유틸리티, 개인 정보 보호 및 튜닝 비용. 또한, 새로운 개인 정보 보호 증가 방법으로 거리 비상관 기법을 사용하여 데이터 위협을 감소시킵니다.

- **Performance Highlights**: 실험 결과, GuardedTuning은 데이터 재구성 공격의 효과를 50% 이하로 감소시키며, 커뮤니케이션 비용을 최대 73.7% 줄입니다. 또한, Tuning된 모델의 유용성 감소는 거의 모든 경우에서 1.5% 이하로 유지되었습니다. 이러한 성과는 클라이언트의 데이터와 서버 모델 가중치 모두의 개인 정보를 효과적으로 보호합니다.



### RoRA: Efficient Fine-Tuning of LLM with Reliability Optimization for Rank Adaptation (https://arxiv.org/abs/2501.04315)
Comments:
          ICASSP 2025

- **What's New**: 이 논문은 RoRA (Rank-adaptive Reliability Optimization)라는 새로운 기법을 제안하며, 이는 LoRA의 스케일링 인자를 최적화하여 성능을 향상시킨다. RoRA는 스케일링 인자에서 r의 제곱근을 사용함으로써, rank 크기를 증가시켰을 때에도 성능이 개선될 수 있도록 설계되었다. 이 방법은 특히 압축된 모델과 비압축 모델의 파인튜닝에 효과적이다.

- **Technical Details**: RoRA의 핵심은 스케일링 인자 $
\alpha/
\sqrt{r}$을 통해 rank에 대한 그래디언트 업데이트의 변동성을 줄이는 것이다. 이로운 점으로는, LoRA 및 DoRA와 비교하여 더 높은 평균 정확도를 달성하며, rank가 증가하더라도 안정성을 유지하는 점이 있다. RoRA는 LLaMA 모델에서 6.5% 및 2.9%의 정확도 향상 결과를 보여준다.

- **Performance Highlights**: RoRA 메소드는 LLaMA-7B/13B, LLaMA2-7B 및 LLaMA3-8B 모델에서 최첨단 성능을 초월하는 성과를 기록하였다. 특히, SHEARED-LLAMA-1.3 모델에서는 81.4%의 프루닝을 달성하면서 LoRA보다 평균 정확도가 5.7% 높았다. 이러한 결과는 RoRA가 파인튜닝 과정에서 매우 효과적이라는 것을 입증한다.



### Physics-Informed Super-Resolution Diffusion for 6D Phase Space Diagnostics (https://arxiv.org/abs/2501.04305)
- **What's New**: 이 논문은 비침습적인 가상 진단 방법을 위해 적응형 물리 정보 슈퍼 해상도 확산(adaptive physics-informed super-resolution diffusion) 기술을 개발했습니다. 이 방법은 6D 차원 공간에서의 전하 입자 빔의 밀도를 효율적으로 분석할 수 있게 해줍니다. 또한, 기계 학습 기법인 변량 오토 인코더(variational autoencoder)를 사용하여 초기 빔 조건 이미지를 저차원 잠재 공간으로 매핑합니다.

- **Technical Details**: 이 방법은 6D 텐서(tensor) 표현을 기반으로 하여 빔의 6D 위상 공간 밀도로부터 326개의 픽셀을 생성합니다. 물리 법칙에 기반한 슈퍼 해상도 확산 기법은 6D 밀도의 저해상도 이미지를 256x256 픽셀 고해상도 이미지로 변환합니다. 또한, 비지도 학습된 잠재 공간 조정(unsupervised adaptive latent space tuning)을 통해 시간에 따라 변하는 빔을 초기 조건에 대한 지식 없이 추적할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 HiRES UED에서의 실험 데이터와 다중 입자 시뮬레이션을 통해 입증되었습니다. 또한, 이 방법은 재훈련 없이 분포 이동(distribution shift)에 대해 강건함을 보였습니다. 이 접근 방식은 고차원 위상 공간에서 진화하는 복잡한 동적 시스템의 다양한 경우에 적용 가능합니다.



### Handling Incomplete Heterogeneous Data using a Data-Dependent Kern (https://arxiv.org/abs/2501.04300)
- **What's New**: 이 논문은 불완전한 데이터를 처리하는 새로운 접근법을 제시합니다. 기존 방법들이 주로 수치 데이터에만 초점을 맞추고, 범주형 데이터나 혼합 데이터셋에 대한 한계를 가지며, 데이터의 결측이 무작위라고 가정하는 것에 대한 문제점을 해결하고자 합니다.

- **Technical Details**: 제안된 방법은 Probability Mass Similarity Kernel (PMK)라는 데이터 의존형 커널을 사용합니다. 이 커널은 데이터 유형이나 결측 메커니즘에 대한 가정을 하지 않으며, 관측된 데이터의 분포를 활용하여 다양한 데이터 유형의 표현을 통합합니다.

- **Performance Highlights**: 논문에서는 10개 이상의 데이터셋에 대해 수치 데이터, 범주형 데이터, 혼합 기능을 갖는 다양한 경우에서 우리 방법을 평가했습니다. 분류와 클러스터링 작업 모두에서 기존 기술을 일관되게 초월하여, 불완전한 이질적 데이터 관리에서의 강력함과 효과성을 입증했습니다.



### An Analysis of Model Robustness across Concurrent Distribution Shifts (https://arxiv.org/abs/2501.04288)
Comments:
          Accepted to TMLR

- **What's New**: 이번 연구는 기존의 단일 분포 변화(UniDS) 평가 방식과 달리 복합 분포 변화(ConDS)를 고려한 모델 평가 프레임워크를 제안합니다. 이 프레임워크는 여러 속성을 가진 데이터셋을 활용하여 복잡한 변화의 동시 발생을 검토하고, 자주 발생하는 다양한 분포 변화 조합을 분석하는 데 중점을 두었습니다. 이를 통해 실제 환경에 가까운 조건에서 모델의 일반화를 체계적으로 평가할 수 있습니다.

- **Technical Details**: 연구팀은 7가지 유형의 분포 변화(DS)와 이를 포함한 33개의 독특한 사례를 탐구했습니다. 26가지 알고리즘을 168개의 데이터셋 쌍에 대해 평가하며, 특히 판단 능력의 향상을 확인했습니다. 또한, 기존 데이터셋의 속성 주석을 활용하여 다중 속성을 기반으로 한 제어 가능한 동시 변화(concurrent shifts)를 구현했습니다.

- **Performance Highlights**: 흥미로운 발견 중 하나는 복합 분포 변화가 단일 변화보다 일반적으로 성능 저하를 초래하지만, 한 모델이 특정 변화에 대해 일반화가 개선되면 다른 변화에도 효과적이라는 것입니다. 또한, 데이터 증강 기법이 전체적으로 최상의 성능을 내며, 복잡한 실제 데이터셋에서 비전-언어 기초 모델의 성능이 급격히 저하되는 모습도 확인되었습니다.



### ElasticZO: A Memory-Efficient On-Device Learning with Combined Zeroth- and First-Order Optimization (https://arxiv.org/abs/2501.04287)
- **What's New**: 이 논문에서는 제로스 오더(Zero-order, ZO) 최적화를 기반으로 한 온디바이스 학습 방법인 ElasticZO 및 ElasticZO-INT8을 제안합니다. 이 방법은 전체 모델 훈련을 수행하며, 메모리 사용량은 추론(inference)과 거의 동일합니다. 특히, ElasticZO-INT8은 8비트 양자화된 딥 뉴럴 네트워크(DNN) 훈련을 위한 새로운 접근법으로, 정수 교차 엔트로피 손실 값을 활용하여 ZO 그래디언트를 계산하는 방법을 도입합니다.

- **Technical Details**: ElasticZO는 대부분의 DNN 훈련에 ZO 방법을 사용하고 마지막 몇 레이어에서만 역전파(backpropagation, BP)를 활용합니다. 이 논문은 ZO와 BP를 결합한 하이브리드 접근법을 최초로 제시하며, ElasticZO-INT8은 표준 8비트 정수 산술만을 사용하여 훈련을 진행하는 ZO 기반 방법입니다. 이를 통해 메모리 효율성을 보다 높이고, 저비용 장치에서도 ZO 기반 훈련이 가능하게 합니다.

- **Performance Highlights**: 실험 결과, ElasticZO는 vanilla ZO에 비해 5.2-9.5% 더 높은 정확도를 달성하면서도 메모리 오버헤드는 0.072-1.7%에 불과합니다. 또한, ElasticZO-INT8은 메모리 사용량을 1.46-1.60배 줄이고 훈련 시간을 1.38-1.42배 단축시키면서도 정확도를 손상시키지 않습니다. 두 방법 모두 fine-tuning 및 초기화에서의 훈련에도 적용 가능성이 높습니다.



### Mapping the Edge of Chaos: Fractal-Like Boundaries in The Trainability of Decoder-Only Transformer Models (https://arxiv.org/abs/2501.04286)
Comments:
          15 pages

- **What's New**: 본 연구에서는 작은 신경망에서 관찰된 하이퍼파라미터 경계의 프랙탈(Fractal) 특성을 중간 크기의 디코더 전용 트랜스포머 모델에 적용하고 있습니다. 특히, 안정적인 수렴과 발산 경계를 구분하는 하이퍼파라미터의 복잡성을 조사하여, 이러한 경계가 자가 유사적(self-similar) 구조를 형성하고 있음을 발견했습니다. 이를 통해 하이퍼파라미터 변화가 트레이닝 결과에 미치는 민감성을 강조하고 있습니다.

- **Technical Details**: 연구에서는 주목(attention)과 완전 연결층(fully connected layers)에서 하이퍼파라미터 경관을 분석하고, 보다 일관된 수렴(convergence) 척도를 사용하여 학습 속도(learning rate)와 수렴 범위를 조사하였습니다. 이 과정에서는 아담(Adam) 등의 최적화 알고리즘을 사용하여, 파라미터 업데이트 과정이 어떻게 프랙탈과 유사한 구조를 이루는지를 밝히고 있습니다. 결과적으로 안정된 수렴 영역이 복잡한 혼돈(chaotic) 경계로 둘러싸인 구조임을 나타내고 있습니다.

- **Performance Highlights**: 실험 결과, 트레인하우는 단순한 임계값(threshold)이 아닌 여러 스케일(scale)에서 평균적으로 일관된 패턴이 나타나는 자가 유사한 구조임을 발견했습니다. 이는 중간 크기의 트랜스포머 모델에서 수렴 제어가 더 복잡하게 작용함을 보여줍니다. 연구의 통찰은 하이퍼파라미터 조정이 신경망 훈련의 성능에 얼마나 큰 영향을 미치는지 이해하는 데 중요한 기여를 할 수 있음을 강조합니다.



### Modeling All Response Surfaces in One for Conditional Search Spaces (https://arxiv.org/abs/2501.04260)
- **What's New**: 본 논문에서는 전통적인 Bayesian Optimization (BO)의 한계를 극복하기 위한 새로운 접근 방식을 제안합니다. 기존의 BO 방식은 서로 다른 subspace에서 하이퍼파라미터 간의 관계를 고려하지 못하고 하이퍼파라미터 간의 의존성을 무시한 채로 진행됩니다. 반면에 저자들은 self-attention 메커니즘을 사용하여 모든 subspace의 반응 표면을 통합하여 모델링하는 방법을 개발하였습니다.

- **Technical Details**: 제안된 방법은 구조 인식을 위한 하이퍼파라미터 임베딩을 설계하여 각 하이퍼파라미터의 구조적 정보를 유지합니다. 이를 통해 attention 기반의 깊은 특징 추출기가 서로 다른 구조를 가진 하이퍼파라미터 설정들을 통합된 특징 공간으로 변환할 수 있도록 합니다. 이 방법은 모든 관측치에 대해 공유되는 surrogate model의 매개변수를 사용하여 sample efficiency를 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 조건부 검색 공간에서 BO의 효율성과 효과성을 개선함을 보여줍니다. 다양한 실제 작업과 HPO-B 벤치마크에서의 결과는 제안된 접근 방식이 기존의 방법들보다 우수하다는 것을 증명합니다. 따라서 제안된 접근 방식은 AutoML 환경에서 하이퍼파라미터 최적화의 성능을 높일 수 있는 가능성을 제공합니다.



### Stable Derivative Free Gaussian Mixture Variational Inference for Bayesian Inverse Problems (https://arxiv.org/abs/2501.04259)
Comments:
          25 pages, 10 figures

- **What's New**: 이번 논문에서는 정규화 상수까지 알려진 확률 분포의 근사화에 대해 논의하고, 과학적 컴퓨팅에서 대규모 역문제에 대한 베이지안 추론(Bayesian inference)에 초점을 맞추고 있습니다. 특히, 복잡한 후방 분포를 근사하는 새로운 방법론인 Derivative Free Gaussian Mixture Variational Inference (DF-GMVI)를 개발하였습니다. 이 방법은 경량의 변분 추론 프레임워크를 제공함으로써 복잡한 분포를 안정적이고 효율적으로 근사할 수 있도록 합니다.

- **Technical Details**: DF-GMVI는 Fisher-Rao 자연 기울기와 특수 사각형 규칙(quadrature rules)을 결합하여 유도된 비파생 업데이트(derivative free updates)를 구현합니다. 이 방법은 가우시안 혼합 변분 가족(Gaussian mixture variational families)의 근사를 통해 높은 정확도와 계산 효율성을 달성하며, 공분산(Covariance) 양호성과 아핀 불변성(affine invariance)을 보장합니다. 이를 통해 대규모 모델 문제에서의 안정성을 확보하고, 복잡한 후방 분포의 다양한 모드들을 적절하게 포착할 수 있습니다.

- **Performance Highlights**: DF-GMVI는 여러 모드(multiple modes), 무한 모드(infinitely many modes), 그리고 곡선 모드(curved modes)에 대한 수치 실험을 통해 그 효과iveness가 입증되었습니다. 특히, Navier-Stokes 방정식의 초기 조건을 성공적으로 복원하는 대규모 응용 프로그램에서 성능을 확인하였습니다. 이러한 결과는 DF-GMVI가 고차원 문제를 처리하는 데 있어 매우 유용한 도구임을 나타냅니다.



### Dynamic Localisation of Spatial-Temporal Graph Neural Network (https://arxiv.org/abs/2501.04239)
Comments:
          This paper was accepted by KDD'25

- **What's New**: 이 논문에서는 ASTGNN(Adaptive Spatial-Temporal Graph Neural Networks)의 새로운 접근 방식인 DynAGS를 소개합니다. DynAGS는 동적 지역화(dynamic localisation)와 시간에 따라 변화하는 공간 그래프(time-evolving spatial graphs)를 결합하여 분산 배치에서 효율성과 정확성을 극대화합니다. 이를 통해 노드 간의 데이터 교환을 최소화하고 개인화된(localised) 정보를 활용하여 전체 시스템의 효율성을 높이는 데 기여합니다.

- **Technical Details**: DynAGS는 중앙 모듈인 Dynamic Graph Generator(DGG)를 기반으로 하며, 크로스 어텐션 메커니즘을 사용하여 시점에 따라 변화하는 데이터의 표현을 생성합니다. 이 프레임워크는 공간 그래프의 토폴로지와 엣지 가중치를 동적으로 조정하며, 각 노드가 자신의 자원을 고려하여 데이터 전송과 추론 정확성 사이의 균형을 개인화할 수 있도록 합니다. 이를 통해 모델의 유연성과 표현성을 극대화하고, 필요 없는 데이터 전송을 줄입니다.

- **Performance Highlights**: 실험 결과, DynAGS는 다양한 현업 데이터셋과 두 가지 주요 ASTGNN 아키텍처에서 기존 최고의 성능을 초월했습니다. 특히, 99.5%의 지역화 정도에서 DynAGS는 80%에서 현재의 선도적인 기준 모델과 동등하거나 더 우수한 결과를 보여주었으며, 데이터 교환 비용을 최소 30배 줄였습니다. 이러한 개선은 분산 환경에서의 모델의 효율성 및 유연성을 크게 향상시킵니다.



### CURing Large Models: Compression via CUR Decomposition (https://arxiv.org/abs/2501.04211)
- **What's New**: CURing이라는 새로운 모델 압축 방법을 소개합니다. 이 방법은 CUR 행렬 분해(CUR matrix decomposition)에 기반하여, 체중 행렬을 선택된 열(C)과 행(R) 및 소형 연결 행렬(U)의 곱으로 근사합니다. 이 접근법은 크기와 활성화의 조합 영향에 따라 선택된 가중치에 적용됩니다.

- **Technical Details**: CURing은 중요한 행렬을 식별하고 유지하여 모델 크기를 크게 줄입니다. 이 과정에서 원본 네트워크의 입력/출력 구조를 유지하며, 중요한 특징인 비음수(non-negativity)를 보존합니다. 또한, 압축된 모델의 활성화 패턴은 원본과 일치하여 해석 가능성(interpretability)을 높입니다.

- **Performance Highlights**: CURing은 최소한의 성능 손실(minimal performance loss)로 모델 크기를 줄이는 데 크게 기여합니다. 이는 깊은 학습 모델의 계산 비용(computational cost) 및 메모리 사용(memory usage) 문제를 효과적으로 해결하는 방향으로 나아가고 있습니다.



### Fixed Points of Deep Neural Networks: Emergence, Stability, and Applications (https://arxiv.org/abs/2501.04182)
Comments:
          21 pages, 7 figures

- **What's New**: 이번 연구에서는 Deep Neural Networks (DNNs)의 고정점(fixed points) 형성과 안정성에 대한 수치적 및 분석적 결과를 제시하고 있습니다. 특히 입력과 출력 벡터의 차원이 동일할 때 형성되는 고정점의 특성과 이들을 활용한 다양한 학습 방식(지도학습, 반지도학습, 비지도학습)에 대한 사례를 보여줍니다. 연구 결과에 따르면, 학습되지 않은 DNN에서 무작위로 초기화된 가중치와 편향이 있을 경우 오직 하나의 고정점만 존재한다고 합니다.

- **Technical Details**: 연구에서는 DNN의 구조(층의 너비, DNN의 깊이, 활성화 함수 등)와 가중치 행렬의 확률 분포가 고정점의 존재 및 개수에 미치는 영향을 분석하였습니다. 고정점의 안정성과 매력의 분포에 대한 연구는 DNN의 훈련 과정 동안 파라미터 분포가 'heavy-tailed'로 변화하는 현상을 설명합니다. DNN의 고정점 수 Q(N, L)은 층의 수 L과 층의 너비 N에 의존하며, 이 함수는 비단조(non-monotone) 행동을 보입니다.

- **Performance Highlights**: 실험적으로, DNN의 매개변수가 'light-tailed' 분포로 초기화된 경우, 훈련 후에는 매개변수 분포가 'heavy-tailed'로 변하는 것을 관찰하였습니다. 고정된 층 너비(N = N0)에 대해 Q(N0, L) 함수는 초기에는 증가하다가 다시 감소하여 1로 수렴하는 양상을 보입니다. 이러한 비단조적 행동은 입력-출력 Jacobian의 경험적 스펙트럼 분포의 방정식을 유도하고 이를 수치적으로 해결함으로써 확인되었습니다.



### KGIF: Optimizing Relation-Aware Recommendations with Knowledge Graph Information Fusion (https://arxiv.org/abs/2501.04161)
Comments:
          Published at IEEE Big Data 2024

- **What's New**: KGIF(지식 그래프 주의 네트워크)는 사용자-아이템 상호작용과 아이템-속성 관계를 명확하게 융합하는 특화된 프레임워크로, 추천 품질을 향상시키기 위해 맞춤형 self-attention 메커니즘을 활용합니다. 이 연구는 KGIF를 통해 지식 그래프 내의 복잡한 관계를 효과적으로 표현하고, 추론 과정을 시각적으로 해석할 수 있는 기능을 제공합니다. 또한, KGIF는 희소 지식 그래프에 대한 강인성을 개선하고, 설명 가능한 추천을 생성할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: KGIF는 초기 임베딩, 관계 정보 융합, 주의 임베딩 전파, 추천 생성의 네 가지 계층 구조로 구성됩니다. 초기 데이터는 사용자-아이템 이분 그래프와 아이템-엔티티 지식 그래프로 재구성되어 임베딩이 생성됩니다. 관계 특정 정보가 엔티티 임베딩에 명시적으로 융합되어 하위 작업의 표현이 풍부해지며, 주의 메커니즘을 통해 관련 정보가 그래프를 통해 전파됩니다. 이러한 과정을 통해 KGIF는 복잡하고 희소한 데이터를 보다 잘 처리할 수 있습니다.

- **Performance Highlights**: KGIF는 Amazon-book, Last-FM, Yelp2018와 같은 여러 벤치마크 데이터 세트에서 기존의 최첨단 기법(SOTA)을 혁신적으로 초월하는 성과를 보였습니다. 이러한 실험을 통해 KGIF의 임베딩 방식과 정보 융합에 대한 유효성을 확립하고, 추천 생성의 투명성을 높일 수 있는 시각적 해석을 제공했습니다. KGIF는 사용자와 아이템 모두에 대한 복잡한 관계를 효과적으로 포착하는 데 있어 큰 기여를 하고 있습니다.



### BiasGuard: Guardrailing Fairness in Machine Learning Production Systems (https://arxiv.org/abs/2501.04142)
- **What's New**: 이 논문에서는 최전선 머신러닝 시스템에서 공정성을 보장하기 위해 설계된 새로운 접근 방법인 'BiasGuard'를 소개합니다. BiasGuard는 Conditional Generative Adversarial Network(CTGAN)를 활용하여 역 보호 속성 값에 조건화된 데이터 샘플을 합성하여 다양한 그룹 간의 공정한 결과를 촉진합니다. 이 방법은 모델 재훈련 없이 배포된 시스템의 공정성 메트릭스를 향상시키는 것을 목표로 합니다.

- **Technical Details**: BiasGuard는 Test-Time Augmentation(TTA)을 사용하여 배포 후 생성된 테스트 샘플을 보강함으로써 공정한 예측을 촉진합니다. 이러한 접근법은 CTGAN이 생성한 합성 데이터를 사용하여 예측을 동적으로 재조정하고, 이는 공정한 기회를 모든 인구 통계학적 집단에 제공하는 데 기여합니다. BiasGuard는 기존의 검증되지 않은 기준선에 비해 공정성을 31% 향상시키는 동시에 정확도를 0.09%만 감소시킵니다.

- **Performance Highlights**: BiasGuard는 재훈련이 불가능한 생산 환경에서도 공정성을 보장할 수 있는 강력한 방법론으로 자리 잡고 있습니다. 기존의 포스트 프로세싱 방법들보다 공정성 향상에서 더 나은 성능을 보이며, 다양한 데이터 세트에서 공정성을 증대시키는 효과를 나타내고 있습니다. 이러한 강력한 결과는 BiasGuard가 머신러닝 시스템에서 편향으로부터 보호하기 위한 효과적인 도구임을示합니다.



### Stochastic Process Learning via Operator Flow Matching (https://arxiv.org/abs/2501.04126)
- **What's New**: 이번 논문에서는 임의의 도메인에서의 확률적 과정 학습(stochastic process learning)을 위한 새로운 프레임워크를 제안합니다. 특히, 함수 공간에서의 확률적 과정 사전(priors) 학습을 위해 operator flow matching (	extit{alg})를 개발하였습니다. 이 방법은 포인트 집합의 값에 대한 확률 밀도를 제공하여 새로운 포인트에서의 평균(mean) 및 밀도 추정(density estimation)에 기반한 수학적으로 처리 가능한 기능 회귀(functional regression)를 가능하게 합니다.

- **Technical Details**: 제안된 	extit{alg}는 확률적 과정 priors를 학습하기 위한 도구로서, 다양한 함수 공간에 적용할 수 있는 범용성을 가지고 있습니다. 이는 기존의 모델들과 달리 수학적으로 정교한 트리트먼트를 가능하게 하며, 특히 새로운 포인트에 대한 예측을 보다 확실하게 수행할 수 있게 합니다. 모델은 여러 새로운 도메인에 걸쳐 효과적인 학습을 수행할 수 있습니다.

- **Performance Highlights**: 제안하는 방법은 확률적 과정 학습, 기능 회귀 및 사전 학습(prior learning)에서 최첨단(state-of-the-art) 모델을 초월하는 성능을 보여줍니다. 실험 결과는 	extit{alg}가 기존 기법들보다 우수한 예측 능력을 발휘하며, 다양한 도메인에서 폭넓은 일반화 능력을 갖춘다는 것을 입증합니다.



### DeepVIVONet: Using deep neural operators to optimize sensor locations with application to vortex-induced vibrations (https://arxiv.org/abs/2501.04105)
- **What's New**: DeepVIVONet라는 새로운 프레임워크가 도입되어 해양 라이저의 소용돌이 유도 진동(VIV)의 최적 동적 재구성 및 예측을 가능케 합니다. 이 모델은 필드 데이터를 이용해 희소한 시공간(spatio-temporal) 측정치를 효과적으로 활용하여 해양 구조물의 동적인 움직임을 정확히 재구성합니다. 또한, 트랜스퍼 러닝(transfer learning)을 통해 다양한 유동 조건에서도 일반화된 성능을 보여줍니다.

- **Technical Details**: DeepVIVONet은 DeepONet 아키텍처를 기반으로 설계되어 희소한 공간 측정과 밀집 시간 측정(dense temporal measurements)을 활용하여 VIV의 동적 예측을 수행합니다. DeepONet은 함수 간의 매핑을 학습하기 위한 두 개의 하위 네트워크, 즉 브랜치 네트워크(branch net)와 트렁크 네트워크(trunk net)로 구성됩니다. 이러한 구성 요소들은 DeepVIVONet이 복잡한 비선형 연산자(nonlinear operator)를 효과적으로 모델링할 수 있도록 돕습니다.

- **Performance Highlights**: DeepVIVONet은 기존의 적절한 직교 분해(proper orthogonal decomposition, POD) 기반 센서 배치 방법과 비교했을 때, 더 정밀하고 비용 효율적인 구성을 생성합니다. 이를 통해 해양 라이저의 운영 효율성을 확보하고 예측의 정확도를 향상시킬 수 있습니다. 또한, DeepVIVONet의 적응 능력을 평가함으로써 다양한 운영 환경에 유연하게 대응 가능한 가능성을 보여줍니다.



### Enhancing Distribution and Label Consistency for Graph Out-of-Distribution Generalization (https://arxiv.org/abs/2501.04102)
Comments:
          Accepted by ICDM 2024

- **What's New**: 이 논문에서는 그래프 데이터에서의 분포 변화 (distribution shifts)를 처리하기 위한 새로운 방법론을 제안합니다. 기존의 그래프 OOD (Out-of-Distribution) 일반화 기술이 일관성 문제에 부딪히고 있으며, 본 연구에서는 이를 해결하기 위해 두 가지 일관성을 향상시키는 혁신적인 접근법을 도입합니다. 제안된 방법은 그래프를 동시에 증강하고 불변하게 만드는 modifier를 설계하여 데이터의 대체 관계를 보장합니다.

- **Technical Details**: 우리의 프레임워크는 두 가지 중요한 모듈을 갖추고 있습니다. 첫째, 분포 일관성 증대 모듈은 훈련 중에 증강된 그래프와 기존 그래프 간의 정보를 최대화하여, 그래프 간의 일관성을 유지하게 합니다. 둘째, 라벨 일관성 증대 모듈은 추출된 불변 하위 그래프가 원본 그래프와 최대한 많은 감독 정보를 공유하도록 보장하여 라벨의 유효성을 유지합니다.

- **Performance Highlights**: 실제 그래프 데이터셋에 대한 광범위한 실험을 통해, 우리의 프레임워크가 다른 최신 방법들보다 우월한 성능을 보임을 보여주었습니다. 우리의 프레임워크는 다양한 그래프 수준과 노드 수준의 OOD 일반화 데이터셋에서 효과성을 증명하였습니다. 이는 분포와 라벨의 일관성 문제를 해결하는데 실질적인 기여를 합니다.



### Neighbor displacement-based enhanced synthetic oversampling for multiclass imbalanced data (https://arxiv.org/abs/2501.04099)
- **What's New**: 본 논문에서는 Neighbor Displacement-based Enhanced Synthetic Oversampling (NDESO)라는 새로운 하이브리드 방법을 제안합니다. 이 방법은 노이즈가 있는 데이터 포인트의 위치를 조정하여 클래스 중심에 더 가깝게 이동시킨 후, 불균형 데이터셋의 오버샘플링을 수행합니다. 이는 데이터의 패턴을 보존하면서 소수 클래스를 균형 있게 만드는 데 효과적입니다.

- **Technical Details**: NDESO는 각 클래스의 노이즈 데이터 포인트를 찾아서 그 이웃들과의 평균 거리 분석을 통해 이들을 클래스의 중심에 가까운 위치로 이동시킵니다. 이 과정에서 데이터 포인트는 원래의 클래스 레이블을 유지하며, 이를 통해 클래스 간의 분리도를 향상시킵니다. 이후 소수 클래스에 대해 랜덤 오버샘플링을 수행하여 데이터 distributions의 균형을 맞추게 됩니다.

- **Performance Highlights**: 14가지 대안 방법들과의 비교 평가를 통해 NDESO는 평균 G-mean 점수에서 가장 뛰어난 성과를 보였으며, 가장 낮은 통계적 평균 순위를 기록했습니다. 이러한 결과는 NDESO의 효과성을 입증했으며, 실제 데이터 불균형 문제 해결에 적합하다는 점을 강조합니다.



### More is not always better? Enhancing Many-Shot In-Context Learning with Differentiated and Reweighting Objectives (https://arxiv.org/abs/2501.04070)
Comments:
          13 pages, 8 figures, 11 tables

- **What's New**: 이 논문은 DR-ICL이라는 새로운 최적화 방법을 도입하여 대형 언어 모델(LLMs)의 Many-shot In-context Learning(아래 ICL)의 성능을 향상시키는 접근 방식을 제안합니다. DR-ICL은 Differentiated Learning과 advantage-based Reweighting을 활용하여 모델이 발생하는 데이터 노이즈의 영향을 줄이면서 더 나은 성능을 발휘하도록 합니다. 또한, MICLB라는 대규모 벤치마크 데이터셋을 소개하여 다양한 NLP 작업에서 Many-shot ICL 전략을 평가할 수 있도록 합니다.

- **Technical Details**: DR-ICL은 NLL(negative log-likelihood) 최적화 목표를 개선하여 모델이 많은 시연을 가진 경우에도 제로 샷 성능을 초과하도록 합니다. 이 방법은 강화 학습에서 영감을 받은 advantage function을 활용하여 샘플들의 가중치를 동적으로 조정함으로써, 훈련 과정에서의 노이즈 데이터를 필터링합니다. 이러한 기법은 ICL에서 여러 샷 수를 효과적으로 처리할 수 있게 하며, 다양한 작업에서의 일반화를 개선합니다.

- **Performance Highlights**: 실험 결과, DR-ICL로 개선된 LLM은 다양한 작업에서의 Many-shot ICL 설정에서 유의미한 성능 향상을 달성했습니다. 특히, 문맥적 단서에 대한 더 깊은 이해를 통해 모델이 맥락 정보를 효과적으로 활용하도록 유도합니다. 이 연구의 결과는 LLM의 다중 작업 학습 및 평가 연구에 있어 중요한 기여를 하며, 오픈 소스 LLM에 대한 연구의 발전을 촉진할 것입니다.



### Explainable Reinforcement Learning for Formula One Race Strategy (https://arxiv.org/abs/2501.04068)
Comments:
          9 pages, 6 figures. Copyright ACM 2025. This is the authors' version of the work. It is posted here for your personal use. Not for redistribution. The definitive Version of Record will be published in SAC 2025, this http URL

- **What's New**: 이번 연구에서는 F1 경주에서 주행 전략을 제어하기 위해 강화 학습 모델인 RSRL(경주 전략 강화 학습)을 소개합니다. 전통적인 하드코딩 및 몬테카를로 기반의 경주 전략에 비해 더 빠른 대안을 제공하며, 이를 통해 경주 당 최고 위치를 개선할 수 있는 가능성을 보여줍니다. 특히 2023 바레인 그랑프리 테스트에서 평균 완주 위치 P5.33을 기록하여 기존 모델을 초월하였습니다.

- **Technical Details**: 연구에서 사용하는 RL(강화 학습) 모델은 Q-learning, 딥 Q-네트워크(DQN), 심층 순환 Q-네트워크(DRQN) 등의 기법을 포함합니다. DQN은 Q-learning과 깊은 신경망을 결합한 모델로, TD 오류를 최소화하면서 현재 상태에서의 Q 값을 예측합니다. DRQN은 과거 상태의 정보를 활용하여 현재 상태를 예측하는 방향으로 DQN의 성과를 향상시키고 있습니다.

- **Performance Highlights**: 기술적으로 RSRL 모델은 2023 바레인 그랑프리에서 P5.33의 평균 완주 위치를 기록하여 기존 가장 좋은 기준 모델인 P5.63을 초과 달성했습니다. 또한 성장 가능성을 보여주는 일반화 연구에서 다양한 트랙에 대한 성능 우선 순위를 학습을 통해 정할 수 있음을 입증했습니다. 마지막으로, XAI(설명 가능한 인공지능) 기법을 통해 모델의 결정을 설명하여 사용자 신뢰를 증진시키고 있습니다.



### Explainable Time Series Prediction of Tyre Energy in Formula One Race Strategy (https://arxiv.org/abs/2501.04067)
Comments:
          9 pages, 9 figures. Copyright ACM 2025. This is the authors' version of the work. It is posted here for your personal use. Not for redistribution. The definitive Version of Record will be published in SAC 2025, this http URL

- **What's New**: 본 연구는 Formula One(Formula 1)에서 타이어 에너지 예측을 위한 심층 학습 모델을 훈련하고 최적화한 내용입니다. 특히, 경주 전략의 중요한 요소인 피트 스톱 및 타이어 컴파운드 선택 시 타이어 열화를 예측할 수 있게 되었습니다. 우리의 접근 방식은 Mercedes-AMG PETRONAS F1 팀의 역사적 데이터에서 훈련된 AI 모델을 통해 이루어졌으며, 이는 F1 팀들이 경주 전략을 최적화하는 데 도움을 줄 수 있는 자동화된 방법을 제공합니다.

- **Technical Details**: 이 연구에서는 여러 종류의 심층 학습 모델, 특히 Recurrent Neural Networks(RNN)와 최신 Transformer 기반 모델을 포함하여 총 네 가지의 모델이 훈련되었습니다. 이 모델들은 차량의 텔레메트리 데이터에서 얻은 다양한 변수를 입력받아 타이어 에너지를 예측하도록 설계되었습니다. 또한, Linear Regression과 XGBoost와 같은 기계 학습 알고리즘 비교를 통해 예측의 정확성을 검증하고 두 가지 설명 가능한 인공지능(XAI) 기법을 통합하여 예측의 기반이 되는 이유를 이해할 수 있도록 하였습니다.

- **Performance Highlights**: 모델의 성능을 평가한 결과, XGBoost가 테스트 세트에서 가장 정확한 예측을 보여주었으나 심층 학습 모델들의 결과도 유망하다고 평가되었습니다. 두 가지 XAI 방법이 통합됨으로써 제공되는 예측의 설명이 신뢰를 높이는 데 기여하고 있습니다. 이러한 연구는 F1에서 실시간 데이터 처리 및 타이어 에너지 예측을 위한 AI 모델 활용에 한 걸음 더 나아간 성과로 평가됩니다.



### FedKD-hybrid: Federated Hybrid Knowledge Distillation for Lithography Hotspot Detection (https://arxiv.org/abs/2501.04066)
- **What's New**: 본 연구에서는 FedKD-hybrid라는 새로운 방법론을 제안하여 Federated Learning (FL) 환경에서의 lithography hotspot detection (LHD)의 효율성을 향상시키고자 합니다. 기존의 parameter 기반 또는 nonparameter 기반 방법론이 가진 정보 전송의 한계를 극복하기 위해, 동일한 레이어를 활용하고 공개 데이터셋으로 글로벌 합의를 달성하는 방식으로 연구를 진행했습니다. 이를 통해 FL 기반 LHD의 잠재력을 최대한 활용할 수 있는 기회를 제공합니다.

- **Technical Details**: 제안된 FedKD-hybrid는 각 고객이 동일한 레이어를 공유하고, 공개 데이터셋을 통해 모델을 훈련하고 평가함으로써 지식 이전을 극대화합니다. 학습 주기 동안, 각 클라이언트는 프라이빗 데이터에서 모델을 훈련한 후, 공개 데이터셋에서 생성된 logits를 서버에 업로드하고, 이 정보는 지역 모델을 업데이트하는 데 사용됩니다. 이 프로세스는 교차 클라이언트 협력과 정보 교환을 통해 FL의 잠재력을 극대화합니다.

- **Performance Highlights**: 실험적으로, FedKD-hybrid는 ICCAD-2012 및 FAB 데이터셋에서 여러 최첨단 FL 방법들과 비교하여 우수한 성능을 입증했습니다. FedKD-hybrid는 비하면적 통계 기반의 지식 전파 기법을 통해 다양한 시나리오에서 LHD 학습 작업의 능률을 높였습니다. 이러한 결과는 성공적인 지식 이전과 성능 향상을 통해 FL이 LHD 분야에서 어떻게 활용될 수 있는지를 보여줍니다.



### Fuzzy Information Entropy and Region Biased Matrix Factorization for Web Service QoS Prediction (https://arxiv.org/abs/2501.04063)
- **What's New**: 이번 논문에서는 QoS(Quality of Service) 예측을 위한 새로운 접근법인 퍼지 정보 엔트로피(fuzzy information entropy)와 지역 편향(region bias)을 기반으로 한 매트릭스 분해(Matrix Factorization) 방법을 제안합니다. 기존의 매트릭스 분해 알고리즘들이 사용자의 지역적 이웃 간 유사성을 간과하고 비대화(non-interactive) 효과를 고려하지 못했던 문제를 해결하기 위해 사용자의 정보 엔트로피를 활용하여 더 현실적인 QoS 예측이 가능해졌습니다. 실험 결과, 제안한 방법이 최신 기술들과 비교했을 때 우수한 성능을 보임을 입증했습니다.

- **Technical Details**: 제안된 방법은 사용자의 위치 정보를 바탕으로 사용자들을 클러스터링하고 서비스 별 편향을 통합하여 비대화 특성을 포착합니다. 퍼지 정보 엔트로피는 사용자의 평점 선호도와 그 불확실성을 측정하는 데 활용되며, 이로 인해 보다 대표성이 있는 이웃 사용자들이 발견됩니다. 마지막으로, 매트릭스 분해 모델에 선형적으로 편향을 결합하고 사용자 간의 이웃 정규화 항을 통합하여 모델의 정확도와 견고성을 높입니다.

- **Performance Highlights**: 실제 QoS 데이터셋을 기반으로 한 실험을 통해 제안된 기법이 5%에서 20%의 매트릭스 밀도를 가진 최신 방법들보다 우수한 성능을 발휘함을 보여주었습니다. 다양한 환경에서의 성능 평가 결과, 사용자와 서비스 간 비대화 특성을 잘 포착하여 실제 네트워크 상황에 적합하게 모델을 개선할 수 있었습니다.



### Causal Machine Learning Methods for Estimating Personalised Treatment Effects -- Insights on validity from two large trials (https://arxiv.org/abs/2501.04061)
Comments:
          15 pages 1 Main table 2 Figures

- **What's New**: 본 연구는 인과적 머신러닝( causal machine learning ) 방법론의 신뢰성에 대해 실증적 환경에서 평가한 결과를 제시합니다. 17개의 주요 인과적 이질성 ML( machine learning ) 방법, 즉 메타러너(metalearners), 트리 기반 방법(tree-based methods), 딥 러닝 방법(deep learning methods)을 국제적 연구와 중국의 대규모 무작위 대조 시험 데이터를 활용하여 검증하였습니다. 이 연구는 개인 맞춤형 치료 효과를 추정하는 인과적 ML의 응용 가능성에 대한 의문을 제기하고 있습니다.

- **Technical Details**: 연구의 중점은 두 개의 대규모 무작위 통제 시험, 즉 국제 뇌졸중 시험(International Stroke Trial)과 중국 급성 뇌졸중 시험(Chinese Acute Stroke Trial)에서 수집된 데이터를 기반으로 17개의 인과적 ML 방법의 내부 및 외부 타당성을 평가하는 것이었습니다. 실험 결과, 이러한 ML 방법들은 훈련 데이터와 테스트 데이터 간의 주요 지표에서 큰 불일치를 보여주었으며, 이는 전문가들이 기존 인과적 ML 모델이 실제의 의학적 상황에 적합하지 않을 수 있음을 알게 만듭니다.

- **Performance Highlights**: 결과적으로, 훈련 데이터를 통해 추정된 개인화된 치료 효과가 테스트 데이터로 일반화되는 데 실패했습니다. 이는 분포 변화(distribution shifts)가 없던 상황에서도 확인되었습니다. 이러한 결과는 정밀 의학(precision medicine) 분야에서 인과적 ML 모델의 현재 적용 가능성에 대한 우려를 불러일으키며, 일반화 가능성을 보장하기 위한 보다 강력한 검증 기술의 필요성을 강조하고 있습니다.



### SFADNet: Spatio-temporal Fused Graph based on Attention Decoupling Network for Traffic Prediction (https://arxiv.org/abs/2501.04060)
Comments:
          Accepted by 2025 lEEE International Conference on Acoustics, speech, and signal Processing (lCASSP2025)

- **What's New**: 본 연구는 SFADNet이라는 혁신적인 교통 흐름 예측 네트워크를 제안합니다. 이 네트워크는 교통 흐름을 시간적 및 공간적 특징 행렬을 통해 여러 교통 패턴으로 나누어 예측 정확성을 향상시킵니다. 또한, 교차 주의 메커니즘을 이용해 각 패턴에 대해 독립적인 적응형 시공간 융합 그래프를 구성합니다.

- **Technical Details**: SFADNet의 구조는 시간 및 공간 임베딩을 기반으로 하여 다양한 교통 패턴 흐름을 모델링합니다. 각 교통 패턴 스트림은 고유한 적응형 시공간 융합 그래프를 할당받아 잔여 그래프 컨볼루션을 수행합니다. 이 모델은 비선형 관계를 포착하기 위해 RNN을 활용하여 단기적인 시간 정보를 통합합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, SFADNet은 네 가지 대규모 데이터 세트를 통해 현재의 최첨단 방법보다 뛰어난 성능을 보였습니다. 이러한 결과는 동적 시공간 관계를 효과적으로 캡처할 수 있는 이 모델의 능력을 잘 보여줍니다.



### The Power of Negative Zero: Datatype Customization for Quantized Large Language Models (https://arxiv.org/abs/2501.04052)
Comments:
          under submission

- **What's New**: 이번 논문에서는 LLMs(대형 언어 모델)의 메모리 요구 사항을 줄이기 위해 FP(부동 소수점) 양자화를 활용한 RaZeR(중복 제로 재매핑) 기술을 제안합니다. RaZeR는 부동 소수점의 음의 영 표현을 미리 정의된 특별 값으로 재매핑하여 양자화 성능을 극대화합니다. 이를 통해 전통적인 INT 양자화 방법보다 더 나은 성능을 보여줍니다.

- **Technical Details**: RaZeR는 특수 값을 신중하게 선택하여 양자화에서 더 많은 숫자 분포를 잘 맞출 수 있도록 설계되었습니다. 이 기술은 가중치 및 KV-cache 양자화 알고리즘과 통합될 수 있으며, 클리핑 및 변환 등의 고급 방법과도 호환됩니다. 또한, 4비트 RaZeR 값을 FP16으로 변환하는 빠른 GEMV(행렬-벡터 곱) 커널을 구현하여 개선된 연산 효율을 제공합니다.

- **Performance Highlights**: 현대 GPU에서 RaZeR는 FP16 구현에 비해 GEMV 속도를 최대 7.56배 향상시키고 LLM 디코딩 처리량에서는 최대 2.72배 속도를 증가시킵니다. 이러한 성과는 RaZeR가 LLM의 성능과 처리 효율성을 크게 개선할 수 있음을 보여줍니다. 결과적으로, RaZeR는 최신 AI 모델의 상용화에 중요한 역할을 할 수 있을 것입니다.



### Planarian Neural Networks: Evolutionary Patterns from Basic Bilateria Shaping Modern Artificial Neural Network Architectures (https://arxiv.org/abs/2501.04700)
Comments:
          11 pages, 9 figures

- **What's New**: 이번 연구에서는 인공 신경망 (ANNs)의 이미지 분류 정확도를 증가시키기 위한 새로운 방안을 제시합니다. 생물 신경망의 진화 패턴을 모델로 삼아, 플라나리안 (planarians)의 신경 구조에서 영감을 받은 ANNs를 개발했습니다. 이를 통해 ANNs의 성능 향상이 가능하다는 점을 강조하며, ResNet을 기본 모델로 선택하여 연구를 진행했습니다.

- **Technical Details**: 연구는 플라나리안의 신경 구조가 포함된 새로운 신경망 아키텍처를 바탕으로 하여, CIFAR-10 및 CIFAR-100 데이터셋에서 평가되었습니다. 플라나리안의 두 개의 신경줄과 뇌를 포함하는 독특한 구조는 ANNs의 성능 개선에 중요한 통찰력을 제공합니다. 이 연구는 이러한 생물적 영감이 주는 가능성을 살펴보고자 하였습니다.

- **Performance Highlights**: 결과적으로, 제안된 방법은 기존의 기본 신경망 모델에 비해 이미지 분류 과제에서 더 높은 예측 정확도를 보였습니다. 이는 다양한 응용 분야에서 ANNs의 성능을 향상시킬 수 있는 생물학적으로 영감을 받은 신경망 아키텍처의 중요성을 보여줍니다.



### Comparative Analysis of Quantum and Classical Support Vector Classifiers for Software Bug Prediction: An Exploratory Study (https://arxiv.org/abs/2501.04690)
Comments:
          Accepted for publication in the Springer Journal: Quantum Machine Intelligence (this https URL)

- **What's New**: 이 논문은 Quantum Computing이 소프트웨어 버그 탐지 문제를 해결하는 데 어떻게 적용될 수 있는지를 탐구합니다. 특히, Quantum Support Vector Classifiers (QSVC)를 사용하여 실제 소스 코드 리포지토리에서 결함이 있는 소프트웨어 커밋을 탐지하는 새로운 접근 방식을 제안합니다. 이 연구는 14개의 오픈 소스 프로젝트에서 수집된 30,924개의 데이터 인스턴스를 분석하였으며, 이는 Quantum Machine Learning (QML) 분야에서 혁신적인 진전을 의미합니다.

- **Technical Details**: 연구에서는 QSVC와 PQSVC 알고리즘을 Classical Support Vector Classifier (SVC)와 비교합니다. 데이터셋의 크기가 커질 경우 QSVC의 성능 저하 문제를 해결하기 위해 데이터셋을 작은 부분집합으로 나누고, 이들로부터 예측 결과를 집계하여 전체 테스트 데이터셋의 버그 탐지 정확도를 높이는 방법을 사용합니다. 이를 통해 Quantum Feature Mapping의 어려움을 극복하기 위한 점진적 테스트 방법론도 제안합니다.

- **Performance Highlights**: QSVC와 PQSVC는 버그가 있는 소프트웨어 커밋을 탐지하는 데 효과적임을 입증하였습니다. 데이터 집합을 작게 나누어 예측 결과를 집계하는 기술이 전체 테스트 데이터셋에 대해 정확도를 높이는 데 성공했습니다. 결과적으로, 본 연구는 QML 알고리즘의 결함 예측 분야에서의 가능성을 열어주며, 향후 연구 방향에 대한 통찰을 제공합니다.



### URSA: Understanding and Verifying Chain-of-thought Reasoning in Multimodal Mathematics (https://arxiv.org/abs/2501.04686)
Comments:
          27 pages, 10 tables, 17 figures. The training data has been released. The code and model are currently undergoing internal review. They will be made available soon. Project url: this https URL

- **What's New**: 본 논문에서는 Chain-of-thought (CoT) 추론을 다룬 새로운 접근법을 제안합니다. 특히, 다중 모드 수학적 추론에서 CoT 훈련 데이터의 부족 문제를 해결하기 위해 3개의 모듈 합성 전략을 도입하였습니다. 이를 통해 MMathCoT-1M이라는 고품질 CoT 추론 지침 데이터셋을 생성하고, URSA-7B 모델의 성능을 여러 벤치마크에서 검증했습니다.

- **Technical Details**: 제안된 시스템은 CoT 증류(Cot Distillation), 궤적 형식 재작성(Trajectory-format rewriting), 및 형식 통합(Format unification)을 포함한 세 가지 모듈로 구성됩니다. 이 과정을 통해 고품질 CoT 추론 데이터셋이 생성되며, URSA-7B 모델은 DualMath-1.1M이라는 데이터 합성 전략을 통해 향상된 성능을 나타냅니다. 모델은 특히 다중 모드 정보 처리 과정에서 발생할 수 있는 오류를 국지화할 수 있는 새로운 방법론을 지니고 있습니다.

- **Performance Highlights**: URSA-7B 모델은 MathVista, MathVerse, WE-MATH 등 여러 다중 모드 수학 벤치마크에서 SOTA 성능을 달성하였습니다. 또한, URSA-RM-7B 모델은 URSA-7B의 검증기로 작동하여 테스트 시간 동안 더욱 향상된 성능을 보여주고 있습니다. 궁극적으로, 이 연구는 다중 모드 수학적 추론에서 모델의 성능 한계를 효과적으로 높이는 데 기여하고 있습니다.



### Toward Sufficient Statistical Power in Algorithmic Bias Assessment: A Test for ABROCA (https://arxiv.org/abs/2501.04683)
- **What's New**: 이 연구는 교육 데이터 마이닝(EDM)에서 알고리즘 편향(algorithmic bias)의 통계적 검증을 위한 새로운 방법론을 제안합니다. 특히, Area Between ROC Curves (ABROCA) 지표의 분포 특성과 그 유의성 검정 방법을 탐구함으로써, 알고리즘 공정성에 대한 평가의 신뢰성을 높이고자 합니다. 연구 결과, ABROCA는 표준 분포와 잘 맞지 않으며, 특히 클래스 불균형(class imbalance)이 존재하는 경우 더욱 그렇다는 사실이 밝혀졌습니다.

- **Technical Details**: 대부분의 알고리즘 편향 연구와 마찬가지로, 이 연구는 ABROCA를 통해 집단 간 성능 성과의 차이를 측정합니다. ABROCA의 데이터 분포 분석을 통해, 대규모 샘플 또는 상당한 효과 크기(effect size)가 필요하다는 것이 확인되었습니다. 또한, 비모수적 랜덤화 테스트(nonparametric randomization tests)를 통해 ABROCA의 유의성을 효과적으로 검정할 수 있는 방법을 제시하며, 지금까지의 알고리즘 편향 평가에서 통계적 파워(statistical power)의 역할이 부족했던 점을 지적합니다.

- **Performance Highlights**: 연구 결과에 따르면, 일반적인 EDM 샘플 크기에서 ABROCA 기반의 편향 평가가 낮은 통계적 파워를 가지고 있어, 모델의 공정성에 대한 결론의 신뢰성을 약화시킵니다. 연구진은 이런 문제를 해결하기 위해 오픈소스 코드를 제공하여 다양한 조건에서 ABROCA의 통계적 검증을 시뮬레이션 할 수 있는 방법을 제안합니다. 이 연구는 알고리즘 공정성 연구의 rigor를 높이고, 교육 모델링에서의 형평성과 재현성을 촉진하기 위한 기초 자료를 마련합니다.



### Enhancing Financial VQA in Vision Language Models using Intermediate Structured Representations (https://arxiv.org/abs/2501.04675)
- **What's New**: 이 연구는 50,000개의 막대 차트에 대해 고유한 구조적 특성을 활용하여 차트 이미지를 선형화된 테이블로 변환하는 DEPLOT(모드 전환 모듈)의 미세 조정을 조사합니다. 미세 조정된 DEPLOT 모델은 카테고리별 매핑 정확도를 측정하는 Relative Mapping Similarity(RMS)와 수치적 해석 정확도를 평가하는 Relative Number Set Similarity(RNSS)를 통해 기본 모델과 비교 평가됩니다. 또한, 100개의 차트 이미지와 질문-응답 세트를 추가하여 대규모 언어 모델(LLMs)의 추론 능력을 탐구합니다.

- **Technical Details**: DEPLOT은 시각 차트 데이터를 구조화된 데이터 테이블로 매핑하기 위한 모드 전환 모듈로, 다양한 차트 유형에서 훈련되지만 도메인별 데이터 세트를 사용하여 미세 조정할 수 있습니다. 본 논문에서는 RNSS와 RMS 두 가지 주요 지표를 통해 모델의 정량적 및 범주적 해석 능력을 평가하며, 차트 구조를 추적할 수 있는 능력을 강조합니다. 이러한 새로운 접근법은 DEPLOT의 성능을 높이고, 보다 신뢰할 수 있는 데이터 시각화 모델 개발을 위한 기초를 제공합니다.

- **Performance Highlights**: 미세 조정된 DEPLOT 모델을 활용한 실험 결과, 높은 품질의 구조화된 데이터와 함께 제공된 경우 LLM의 추론 능력이 크게 향상됨을 보여줍니다. 특히 Qwen2-VL-7B와 같은 소형 모델이 고급 모델인 GPT-4o보다 더 나은 성능을 발휘하여 차트 데이터 해석의 정확성을 높였습니다. 이 연구는 자동 차트 해석 및 추론 향상을 위한 모드 전환 통합의 혁신적 잠재력을 강조합니다.



### Natural Variational Annealing for Multimodal Optimization (https://arxiv.org/abs/2501.04667)
- **What's New**: 본 논문에서는 Natural Variational Annealing (NVA)이라는 새로운 다중 모드 최적화 접근 방식을 소개합니다. 이 접근법은 블랙박스 비볼록 목표의 여러 글로벌 및 로컬 모드를 동시에 검색하는 데 필요한 세 가지 기본 개념을 결합합니다. NVA는 변별 후방을 사용한 동시 검색, 탐색과 활용의 점진적인 거래를 위한 애닐링, 그리고 자연 경량 학습을 통한 변별 검색 분포의 학습을 포함합니다.

- **Technical Details**: NVA는 여러 가지 모드에 대한 동시 검색을 가능하게 하는 변별 근사를 활용합니다. 여기서 최적화하려는 목표는 혼합 분포를 통해 달성되며, 각각의 혼합 구성 요소는 검색 공간의 다른 지역에 초점을 맞추고 있습니다. 또한, 엔트로피 애닐링을 통해 탐색과 활용의 균형을 맞추고, 자연 경량 상승 방법을 사용하여 간단하고 효율적인 알고리즘을 구현합니다.

- **Performance Highlights**: NVA는 시뮬레이션을 통해 검색의 품질을 평가하고, 기존의 기울기 하강법 및 진화 전략과 비교했습니다. 실제 행성 과학의 역문제 문제에도 적용 가능성을 보여주며, 다양한 해결책을 마련할 수 있는 다중 모드 최적화의 잠재력을 강조합니다. 기존의 최적화 방법들이 가진 한계를 극복하는 데 기여할 수 있는 새로운 길을 제시합니다.



### Multi-task retriever fine-tuning for domain-specific and efficient RAG (https://arxiv.org/abs/2501.04652)
Comments:
          9 pages, 2 figures. Submitted to NAACL 2025 Industry Track

- **What's New**: 본 논문에서는 Retrieval-Augmented Generation (RAG) 기술을 활용하여 Large Language Models (LLMs)의 성능을 향상시키기 위한 새로운 접근법을 제시합니다. 특히, 다양한 도메인 특정 작업에 대응할 수 있는 소형 retriever encoder를 instruction fine-tuning 방식으로 학습시켜, 여러 환경에서 활용 가능한 효과적인 솔루션을 제공합니다. 이를 통해 RAG 응용 프로그램의 가능한 확장성을 극대화하고, 비용과 처리 속도를 줄일 수 있습니다.

- **Technical Details**: 연구에서는 mGTE 모델을 fine-tune하여 생성된 데이터셋을 기반으로 소형 retriever를 다양한 작업에 맞춰 훈련합니다. retriever는 steps, table names, field names 등 다양한 구조화된 데이터를 데이터베이스에서 검색하여 LLM에 전달하는 방식으로, RAG 응용 프로그램의 결과물 품질을 높입니다. 본 연구는 OOD(Out-Of-Domain) 설정에서 retriever의 일반화 성능을 평가하며, 학습 데이터셋은 내부 데이터베이스와 Flow Generation training set에서 추출됩니다.

- **Performance Highlights**: 제안된 방법은 다양한 도메인과 관련된 과제를 해결하는 데 뛰어난 성능을 발휘합니다. 실험 결과, instruction fine-tuning을 통한 retriever 모델이 OOD 설정에서도 유의미한 성능 향상을 보여주며, 차별화된 data retrieval 과제를 해결할 수 있는 능력을 입증했습니다. 이를 통해, GenAI 응용 프로그램의 다양성과 효율성을 높이는 새로운 가능성을 열어줍니다.



### MedCoDi-M: A Multi-Prompt Foundation Model for Multimodal Medical Data Generation (https://arxiv.org/abs/2501.04614)
- **What's New**: 이번 연구는 MedCoDi-M이라는 새로운 6.77억 개의 파라미터로 이루어진 모델을 제안합니다. 이 모델은 다중모달(multi-modal) 의학 데이터 생성을 위해 설계되었으며, 대조 학습(contrastive learning)과 대량의 데이터를 활용하여 서로 다른 데이터 모달리티 간의 관계를 캡처하는 공유 잠재 공간(shared latent space)을 구축합니다. MedCoDi-M은 다양한 데이터 세트에서의 유용성을 평가하였습니다.

- **Technical Details**: MedCoDi-M은 기존의 GANs와 Diffusion Models(DMs)에서 발전된 기술을 사용하여 안정적이고 고품질의 합성 데이터를 생성할 수 있는 능력을 갖추고 있습니다. Multi-Prompt 학습 기법을 통해 서로 다른 모달리티의 정보를 융합하여 일관성 있는 데이터를 생성할 수 있도록 설계되었습니다. 저자들은 이를 통해 다양한 유형의 의학적 데이터 조합을 효과적으로 다룰 수 있는 모델의 필요성을 강조하고 있습니다.

- **Performance Highlights**: MedCoDi-M의 효과성을 검증하기 위해 MIMIC-CXR 데이터셋에 대해 다섯 가지 경쟁 모델과의 비교가 진행되었습니다. 이어 전문 방사선의들과 함께 실시한 비주얼 터링 테스트(Visual Turing Test)를 통해 생성된 데이터의 현실성과 임상적 관련성을 평가하였습니다. 결과적으로 MedCoDi-M은 데이터 익명화, 데이터 부족, 불균형 학습과 같은 의료 분야의 주요 과제들을 해결하는 데 도움을 줄 수 있는 가능성을 보였습니다.



### Comprehensive Examination of Unrolled Networks for Linear Inverse Problems (https://arxiv.org/abs/2501.04608)
Comments:
          27 pages, 10 figures. Project Page: this https URL

- **What's New**: 이번 논문에서는 unrolled networks의 개념을 통해 다양한 컴퓨터 비전과 이미지 처리 작업에서의 디자인 결정을 줄일 수 있는 방법을 제안합니다. 기존의 어려운 디자인 선택 상황을 해결하기 위해, 연구진은 유용한 방법론들과 아이디어들을 통합하는 것을 목표로 하며, unrolled networks의 설계에서 각 선택의 영향을 설명하는 포괄적인 ablation study를 제공합니다.

- **Technical Details**: 연구는 linear inverse problems에 적용 가능성을 가진 unrolled networks의 개발을 중점적으로 다룹니다. 특히, 측정 과정은 자료 수집과 데이터의 복원 과정을 수학적으로 모델링하며, 여러 측정 행렬과 그 속성을 고려하여 최적의 성능을 이끌어내는 방법을 논의합니다. 이러한 네트워크는 심층 학습을 필요로 하지만, 그 과정에서 발생하는 과도한 계산 부담을 줄이기 위한 방법들을 소개합니다.

- **Performance Highlights**: 이 논문은 unrolled networks의 효과를 높이고 디자인 결정의 복잡성을 줄이기 위한 실용적인 권고사항들을 제시합니다. 연구는 다양한 최적화 알고리즘과 neural network을 결합하여 성능을 향상시킬 수 있는 가능성을 탐구하며, 새로운 응용 분야에 대한 적응성을 강조합니다. 이를 통해 연구자들이 unrolled networks를 손쉽게 설계하고, 네트워크 내 문제를 효율적으로 진단할 수 있는 방안을 제시하고자 합니다.



### A 65 nm Bayesian Neural Network Accelerator with 360 fJ/Sample In-Word GRNG for AI Uncertainty Estimation (https://arxiv.org/abs/2501.04577)
Comments:
          7 pages, 12 figures

- **What's New**: 이번 논문에서는 Bayesian Neural Networks (BNNs)의 랜덤 넘버 생성(Generate Random Number, RNG) 오버헤드를 줄이고 메모리 내 연산(compute-in-memory, CIM)을 통해 BNN 성능을 향상시키기 위한 ASIC 칩을 제안합니다. 이 칩은 360 fJ/Sample의 가우시안 RNG를 SRAM 메모리 단어에 직접 통합하여 운영됩니다. 이러한 혁신적인 접근방법은 RNG 관련 오버헤드를 감소시키고, BNN의 동시 병렬 처리(compute-in-memory)를 가능하게 합니다. 이를 통해 AI 불확실성 추정이 엣지 컴퓨테이션(edge computation)에 적용될 수 있습니다.

- **Technical Details**: Bayesian Neural Networks (BNNs)는 확률적(weight의 posterior distribution으로 대체) 예측을 제공하는 데이터 기반 심층 학습 시스템을 지원합니다. 이 논문에서는 메모리 내 직접 가우시안 무작위 수 생성(Gaussian RNG) 통합을 통한 연산 최적화를 논의하며, BNN으로의 연산 효율을 높이는 방법을 설명합니다. 또한, 새로운 하드웨어 설계를 통해 기존 디지털 RNG의 한계를 극복하고, 메모리 작업을 줄이면서도 BNN의 추론 성능을 향상시키는 것을 목표로 하고 있습니다.

- **Performance Highlights**: 프로토타입 칩은 5.12 GSa/s의 RNG 처리량과 102 GOp/s의 신경망 처리량을 달성하면서, 면적은 0.45 mm2에 불과합니다. 이러한 성능은 BNN을 엣지 장비(Edge devices)에서의 구현을 가능하게 해 주며, 에너지 효율성 또한 뛰어납니다. 기기 집적화 및 효율적인 메모리 접근 방식을 통해, BNN의 세부적인 인퍼런스 작업에서 발생할 수 있는 에너지 소모를 크게 줄일 수 있습니다.



### Regret Analysis: a control perspectiv (https://arxiv.org/abs/2501.04572)
Comments:
          10 pages no figures

- **What's New**: 이번 연구에서는 온라인 학습과 모델 기준 적응 제어(model reference adaptive control) 간의 유사점과 차이점을 심층적으로 논의합니다. 특히, 두 분야에서의 알고리즘 분석 방법과 '좋은' 알고리즘과 '나쁜' 알고리즘을 구분짓는 목표(metrics)에서의 차이를 강조합니다. 본 논문은 경량 함수에 대한 경량 하강(gradient descent)의 후회 분석을 통해 이러한 차이를 명확히 하고, 온라인 적응 제어(online adaptive control)의 새로운 패러다임에 대해 논의합니다.

- **Technical Details**: 적응 제어는 시스템의 모든 시간 변동 매개변수(state) 및 상태가 제한되어 있음을 증명하고, 적응 제어된 시스템과 참조 시스템(reference system) 간의 순간적 오차가 시간에 따라 수렴한다는 두 가지 주요 목표를 가지고 있습니다. 반면, 온라인 학습에서는 알고리즘 성능이 후회(regret)라는 개념을 통해 측정됩니다. 후회는 온라인 알고리즘의 누적 비용에서 단일 최적 고정 매개변수의 누적 비용을 차감하여 정의됩니다.

- **Performance Highlights**: 본 연구는 온라인 학습 및 적응 제어 각각의 성능 목표가 다르다는 점을 부각시킵니다. 적응 제어는 시스템 안정성(stability)이나 신뢰성(reliability)을 중시하는 반면, 온라인 학습은 최적화(optimization)나 통계적 학습 이론(statistical learning theory)에 중점을 두고 분석됩니다. 이 연구의 결과는 후회 최적 제어 전략이 실제로 구현 가능할지를 검토하는 데 중요한 기초 자료를 제공합니다.



### Supervision-free Vision-Language Alignmen (https://arxiv.org/abs/2501.04568)
Comments:
          Preprint

- **What's New**: 이 논문에서는 SVP(Supervision-free Visual Projection)라는 새로운 프레임워크를 소개합니다. VLMs(비전-언어 모델)의 성능 향상에 초점을 맞추며, 이는 비급식 데이터나 선호 주석 없이도 가능하다는 점에서 이전 연구들과 차별화됩니다. SVP는 자기 캡셔닝(self-captioning)과 사전 훈련된 그라운딩 모델(pre-trained grounding model)을 활용하여 VLM의 잠재 정보를 이끌어내는 피드백 메커니즘을 이용합니다.

- **Technical Details**: SVP는 크게 이미지-텍스트 쌍의 수집이 필요하지 않은 점이 특징이며, 이를 통해 비전-언어 정합성(vision-language alignment)을 개선합니다. 연구에서는 캡셔닝(captioning), 참조(referring), 비주얼 질문 응답(visual question answering), 멀티태스킹(multitasking), 할루시네이션 제어(hallucination control), 객체 회상(object recall) 등 여섯 가지 주요 영역에서 평가가 이루어졌습니다.

- **Performance Highlights**: SVP를 적용한 결과, 캡셔닝 작업에서 평균 14%의 성능 향상, 객체 회상에서 최대 12% 증가, 할루시네이션 비율 대폭 감소 등 주요 성과가 보고되었습니다. 특히, SVP를 활용한 작은 VLM이 원래 크기가 다섯 배 큰 모델과 비교할 만한 수준으로 할루시네이션을 줄인 점이 주목할 만합니다.



### Combining YOLO and Visual Rhythm for Vehicle Counting (https://arxiv.org/abs/2501.04534)
Comments:
          Accepted for presentation at the Conference on Graphics, Patterns and Images (SIBGRAPI) 2023

- **What's New**: 이 논문에서는 전통적인 이미지 기반 차량 검출 및 카운팅 방식의 두 가지 주요 단계인 초기 검출 및 이후 추적 단계를 제거한 새로운 방법을 제안합니다. 제안된 방법은 키 비디오 프레임에서만 차량을 검출하는 데 초점을 맞추어 효율성을 크게 향상시킵니다. 특히 YOLO와 Visual Rhythm을 결합하여 유용한 정보가 포함된 프레임에 초점을 맞춰 차량을 검출합니다.

- **Technical Details**: YOLO(You Only Look Once)는 객체 검출을 위한 실시간 딥러닝 모델로, 컴퓨터 비전에서 중요한 역할을 맡고 있습니다. 이 논문에서는 Visual Rhythm(VR) 기법을 적용하여 비디오 프레임의 시간-공간 이미지를 생성하고, 이 이미지를 통해 차량 검출 및 카운팅을 위한 주요 프레임을 선택하도록 설계되었습니다. VR 이미지는 정의된 카운팅 라인 근처의 픽셀만 포함하여, 비디오의 정보를 효율적으로 집약합니다.

- **Performance Highlights**: 실험 분석 결과, 제안된 방법은 여러 비디오 세트를 통해 평균 99.15%의 차량 카운팅 정확도를 달성하였으며, 이는 추적 기반 접근 방식에 비해 처리 속도가 3배 빠릅니다. 따라서 이 접근 방식은 단방향으로 이동하는 대상의 검출 및 식별이 필요한 다양한 응용 프로그램에서도 효과적으로 사용될 수 있습니다.



### Revisiting LocalSGD and SCAFFOLD: Improved Rates and Missing Analysis (https://arxiv.org/abs/2501.04443)
- **What's New**: 이 논문에서는 LocalSGD와 SCAFFOLD의 수렴 속성을 다양한 조건 하에서 분석합니다. 여기에는 gradient similarity, Hessian similarity, weak convexity, 그리고 Lipschitz continuous Hessian 등이 포함됩니다. 기존의 분석이 강한 가정이나 비현실적인 전제에 의존하는 것을 넘어서, 이 두 방법의 이론적 장점을 명확히 해주는 결과를 제시합니다.

- **Technical Details**: LocalSGD는 약한 볼록 함수에 대해 MbSGD보다 더 빠른 수렴 속도를 실현합니다. 또한, SCAFFOLD는 일반적인 비이차(non-quadratic) 함수 집합에서 MbSGD보다 더 빠르게 수렴함을 보여줍니다. 현재 연구에서 제시하는 주요 가정은 고차 조건이 LocalSGD의 성능 향상에 기여할 수 있음을 입증합니다.

- **Performance Highlights**: LocalSGD와 SCAFFOLD는 모두 기존의 MbSGD 방식보다 더 나은 성능을 보여주며, 특히 커뮤니케이션 효율성을 고려할 때 매우 중요한 방법들로 자리잡고 있습니다. 이 연구는 이론적으로 두 방법이 언제 MbSGD에 비해 우위를 점하는지를 분명히 규명하여 향후 알고리즘 비교와 발전 방향을 제시합니다.



### Machine Learning and statistical classification of CRISPR-Cas12a diagnostic assays (https://arxiv.org/abs/2501.04413)
Comments:
          25 pages, 5 figures, research paper. Nathan Khosla and Jake M. Lesinski contributed equally. Electronic supporting information is included as an appendix

- **What's New**: 이번 연구에서는 CRISPR 기반 진단법의 데이터 분석 기술을 개선하는 데 중점을 두었습니다. 기존의 slope-based classification 방법론과 비교하여 새로운 통계 테스트 방법을 도입하여 진단 속도와 정확성을 향상시킬 수 있음을 보여주었습니다. 특히 Kolmogorov-Smirnov와 Anderson-Darling 테스트가 가장 빠른 결과 도출과 높은 정확성을 기록했습니다.

- **Technical Details**: 연구자는 진단 정확도, 민감도(sensitivity) 및 특이도(specificity)와 같은 성능 벤치마크를 설정하였으며, 다양한 slope-based 방법론의 성과를 비교했습니다. 세 가지 다른 quadratic empirical distribution function 통계 테스트를 활용하여 Clinical data set에서 성능을 평가했고, 이는 기존 방법에 비해 현저하게 향상된 결과를 가져왔습니다. 또한, Long Short-Term Memory (LSTM) recurrent neural network를 사용하여 CRISPR 생체 감지 데이터 분류를 수행하여 100%의 특이도를 달성했습니다.

- **Performance Highlights**: 연구의 결과는 새로운 통계 방법이 CRISPR 진단의 속도와 정확성을 크게 개선할 수 있음을 입증하였습니다. 특히, Kolmogorov-Smirnov와 Anderson-Darling 테스트는 다른 방법들보다 우수한 성능을 보였으며, CRISPR 진단의 프레임워크 개선에 기여할 수 있는 가능성을 제시합니다. 이러한 발견은 향후 CRISPR 기반 진단 기술의 적용과 발전에 기여할 것으로 기대됩니다.



### User Simulation in the Era of Generative AI: User Modeling, Synthetic Data Generation, and System Evaluation (https://arxiv.org/abs/2501.04410)
- **What's New**: 이번 논문에서는 Generative AI 시대에 등장한 사용자 시뮬레이션(user simulation)의 필요성과 그 응용 가능성에 대해 다루고 있습니다. 사용자가 AI 시스템과 상호작용하는 방식을 모방하는 지능형 에이전트를 생성함으로써, 연구자들은 사용자의 행동을 모델링하고 분석할 수 있으며, 이는 AI 시스템의 안전하고 책임감 있는 발전에 중요한 역할을 합니다. 또한, 사용자 시뮬레이션은 인공지능의 일반 지능(AGI) 개발에도 중대한 영향을 미칠 것으로 기대됩니다.

- **Technical Details**: 사용자 시뮬레이션은 사용자 행동을 기반으로 하여, 사용자의 의사결정 패턴을 모델링하는 과정입니다. 이를 위해 시스템의 특성, 사용자가 수행하는 작업의 종류, 사용자에 대한 정보와 같은 변수들이 고려되어야 합니다. 여러 유형의 사용자 행동을 효과적으로 시뮬레이션하기 위해서 Markov Decision Process(MDP)와 같은 계산적 모델을 활용할 수 있으며, 이로 인해 사용자 시뮬레이터가 다양한 사용자와 조건을 반영할 수 있도록 구성됩니다.

- **Performance Highlights**: 본 논문은 사용자 시뮬레이션이라는 주제를 심도 있게 탐구하며, 관련된 최신 연구 동향과 응용 분야를 정리합니다. 사용자 행동 모델링, 데이터 증대(data augmentation), 그리고 시스템 평가와 관련된 사례들(예: 대화 시스템의 현실적인 대화 생성 등)을 제시하여 사용자 시뮬레이션의 유용성을 강조합니다. 사용자 시뮬레이션은 실제 데이터 확보가 어려운 상황에서도 대량의 합성 데이터를 생성하고, AI 모델의 효율성을 개선하는 데 기여할 수 있는 잠재력을 지니고 있습니다.



### The unbearable lightness of Restricted Boltzmann Machines: Theoretical Insights and Biological Applications (https://arxiv.org/abs/2501.04387)
Comments:
          7 pages, 3 figures. To be published in EPL as di Sarra et al 2025 EPL. Accepted manuscript available online at this https URL

- **What's New**: 본 논문에서는 Restricted Boltzmann Machines (RBM)의 활성화 함수가 모델의 기능성에 미치는 역할을 집중적으로 리뷰합니다. 비록 RBM이 간단하지만 강력한 신경망임에도 불구하고, 활성화 함수의 선택이 데이터 분석의 결과에 중요한 영향을 미친다는 점을 강조합니다. 최근의 이론적 결과를 바탕으로 다양한 활성화 함수의 장점과 한계를 논의합니다.

- **Technical Details**: RBM은 입력과 출력 간의 관계를 정의하는 단일 뉴런의 활성화 함수에 따라 기능적으로 다르게 작동할 수 있습니다. 특히, 바이너리 유닛을 사용하는 시그모이드 활성화 함수와 비바이너리 유닛을 사용하는 비시그모이드 활성화 함수에 대한 비교 분석이 이루어집니다. 이러한 활성화 함수들이 각기 다른 응용 분야에서 어떻게 적용되는지에 대한 통찰을 제공합니다.

- **Performance Highlights**: RBM은 생물학적 데이터 분석, 특히 신경 데이터 분석과 단백질 데이터 분석에 적용되어 중요한 인사이트를 제공합니다. 신경 데이터 분석에 있어 시그모이드 활성화 함수를 사용하는 경우 대부분의 RBM 유닛이 효과적으로 작용하는 것으로 나타났습니다. 그러나 면역학과 같은 다른 분야에서는 비시그모이드 활성화 함수가 데이터에 대한 중요한 통찰을 제공하는 것으로 보고됩니다.



### Decoding EEG Speech Perception with Transformers and VAE-based Data Augmentation (https://arxiv.org/abs/2501.04359)
Comments:
          19 pages, 15 figures, 2 tables

- **What's New**: 이번 연구는 비침습적인 뇌 신호인 EEG를 기반으로 한 음성 디코딩의 발전 가능성을 탐구합니다. 연구팀은 variational autoencoders (VAEs)와 최신의 시퀀스-투-시퀀스 (sequence-to-sequence) 딥 러닝 아키텍처를 활용하여 EEG 데이터의 품질을 향상시키고, 복잡한 음성 인식 과제에서의 성능 향상을 목표로 하고 있습니다. 이것은 조용한 커뮤니케이션이나 소음이 있는 환경에서의 보조 기술로서 큰 가능성을 지닙니다.

- **Technical Details**: EEG 신호에서의 음성을 디코딩하기 위해, 연구팀은 심층 학습 기법과 데이터 전처리에 중점을 두고 대규모 데이터셋을 활용합니다. 변형 오토인코더(VAE)를 사용하여 노이즈 저항성을 강화하고, EMG 기반의 SOTA transformer 모델을 EEG 신호에 적응하는 접근 방식도 연구하였습니다. 이러한 방법의 조합을 통해 EEG 기반 음성 인식의 최신 기술을 향상시킬 수 있는 잠재력을 갖추었습니다.

- **Performance Highlights**: 실험 결과는 VAE가 인공 EEG 데이터를 재구성하여 데이터 증강에 유용할 수 있음을 보여주었습니다. 또한, 시퀀스-투-시퀀스 모델은 분류 모델에 비해 문장 생성 성능이 더 유망한 것으로 나타났습니다. 이러한 결과는 EEG를 통한 음성 인식 디코딩의 미래 연구를 위한 기초를 마련하며, 조용한 음성이나 상상된 음성과 같은 음성 생산 과제로의 확장 가능성을 제시합니다.



### DeFusion: An Effective Decoupling Fusion Network for Multi-Modal Pregnancy Prediction (https://arxiv.org/abs/2501.04353)
- **What's New**: 이번 연구에서는 체외 수정 배아 이식 (IVF-ET)에서 임신 예측을 위한 새로운 모델인 DeFusion을 제안합니다. 이 모델은 다양한 발생단계의 배아 이미지와 부모의 생식지표를 통합하여 예측 정확도를 향상시키는 데 초점을 맞추고 있습니다. 기존 연구와는 달리, DeFusion은 첫 3일 간의 배아 이미지와 피험자 데이터를 효율적으로 결합하여 최적의 결과를 도출합니다.

- **Technical Details**: DeFusion은 두 가지 주요 모듈로 구성됩니다: 첫째, 다양한 발생일수의 배아 이미지를 활용하기 위해 시간-공간 포지션 인코딩 (spatial-temporal position encoding)을 사용하며, 둘째, 테이블 트랜스포머 (table transformer)로 생식지표 데이터를 추출합니다. 이 두 가지 구성 요소는 각 개별 데이터를 관련 및 비관련 정보로 분리하여 더 정교한 융합 (fusion)을 가능하게 합니다.

- **Performance Highlights**: 4046개의 데이터를 포함하는 새로운 데이터셋을 사용하여 평가한 결과, DeFusion은 최신의 다른 방법들보다 뛰어난 성능을 보였습니다. 또한, 안과 질병 예측 데이터셋에서도 잘 일반화되는 성능을 입증하여, 본 모델의 폭넓은 적용 가능성을 나타냈습니다.



### DCIts -- Deep Convolutional Interpreter for time series (https://arxiv.org/abs/2501.04339)
Comments:
          37 pages, 15 figures

- **What's New**: 이번 논문에서는 다변량 시계열 예측을 위한 해석 가능한 딥러닝 모델을 소개합니다. 이 모델은 예측 성능과 해석 가능성을 동시에 우선시하며, 복잡한 물리 현상을 이해하는 데 매우 중요한 도구가 될 것입니다. 기존의 해석 가능성 방법을 초과하는 성능을 보여주며, 정확도를 손상시키지 않고 예측의 근거를 제공합니다.

- **Technical Details**: 제안된 모델은 다변량 시계열(Multivariate Time Series)을 특별히 설계하여, 모든 필요한 상호작용을 최소한의 시간 프레임 내에서 포착하는 최적의 윈도우 크기를 확보할 수 있도록 제작되었습니다. 또한, 모델 차수를 최적화하여 복잡성과 추가적인 항을 통합할 때 균형을 맞춥니다. 모델은 시간의 지연(Lags) 및 가장 관련성 높은 시계열을 확인하여 예측을 위한 직관적이고 투명한 설명을 제공합니다.

- **Performance Highlights**: 밀도 추정이 필요 없는 데이터 구조와 잘 맞아떨어지는 이 모델은 방대한 실험을 통해 예측 성능을 검증하였습니다. 결정적으로, 이 모델은 기존 모델들과 비교할 때 해석 가능성을 유지하면서도 예측 정확도를 유지 또는 초과하는 성과를 보여줍니다. 결과적으로, 이 모델은 다이나믹 시스템을 모델링하고 이해하는 데 큰 의미를 가져올 수 있는 가치 있는 도구로 자리 잡게 됩니다.



### AutoDFL: A Scalable and Automated Reputation-Aware Decentralized Federated Learning (https://arxiv.org/abs/2501.04331)
Comments:
          Paper accepted at NOMS'2025 (pages 9, figures 5)

- **What's New**: 이 논문에서는 블록체인 연합 학습(Blockchained Federated Learning, BFL)의 효율성을 개선하기 위해 AutoDFL이라는 새로운 프레임워크를 제안합니다. AutoDFL은 분산된 연합 학습 환경에서 복잡한 평판 관리 과정을 자동화하고, 블록체인 기반의 보안성과 투명성을 유지할 수 있는 특징을 갖추고 있습니다.

- **Technical Details**: AutoDFL은 Layer-2 스케일링 솔루션인 zk-Rollups를 활용하여 성능을 증가시키면서도 블록체인 Layer-1의 보안을 유지합니다. 또한, 연합 학습 참여자들에게 동기를 부여할 수 있도록 설계된 자동화되고 공정한 평판 모델을 도입하여 프레임워크의 효율성을 높입니다.

- **Performance Highlights**: AutoDFL은 다양한 커스텀 워크로드를 테스트한 결과 평균 3000 TPS(Transactions Per Second)의 처리량을 달성하며, 가스 비용을 20배까지 줄였습니다. 이러한 성과는 BFL의 확장성과 비용 효율성을 크게 향상시킵니다.



### VerifBFL: Leveraging zk-SNARKs for A Verifiable Blockchained Federated Learning (https://arxiv.org/abs/2501.04319)
Comments:
          Paper accepted at NOMS'25 (9 pages, 6 Figures)

- **What's New**: 이번 논문에서는 분산형 머신러닝 패러다임인 Blockchain-based Federated Learning (BFL)에서 새로운 framework인 VerifBFL을 소개합니다. VerifBFL은 신뢰를 요구하지 않으며, 개인 정보 보호와 검증 가능성을 동시에 제공하는 시스템입니다. 이전의 BFL 프레임워크들이 다양한 공격에 취약했던 반면, VerifBFL은 블록체인 기술과 암호화 프로토콜을 통합하여 안전성을 높였습니다.

- **Technical Details**: VerifBFL은 zero-knowledge Succinct Non-Interactive Argument of Knowledge (zk-SNARKs)와 incrementally verifiable computation (IVC)을 활용하여 로컬 훈련 및 집계 과정의 검증 가능성을 확보합니다. 모든 교육 및 집계 증명은 블록체인상에서 검증되며, 이는 각 참여자의 기여에 대한 무결성과 감사 가능성을 보장합니다. 또한, VerifBFL은 데이터 추출 공격으로부터 훈련 데이터를 보호하기 위해 differential privacy를 적용합니다.

- **Performance Highlights**: 제안된 프로토콜의 효율성을 입증하기 위해, emerging tools를 사용하여 개념 증명을 구축했습니다. 실험 결과, VerifBFL 내에서 로컬 훈련 및 집계에 대한 증명 생성에는 각각 81초 이하와 2초 이하가 소요되며, on-chain에서의 검증은 0.6초 이하로 완료된다고 합니다.



### FSC-loss: A Frequency-domain Structure Consistency Learning Approach for Signal Data Recovery and Reconstruction (https://arxiv.org/abs/2501.04308)
Comments:
          11 pages,7 figures

- **What's New**: 이번 연구에서 제안된 새로운 접근 방식은 Frequency Structure Consistency (FSC) loss 함수와 데이터 구성 요소 임베딩 전략을 통해 신호 매트릭스(Signal Matrix, SM)의 고해상도 복원을 실현하는 것이다. 기존의 측정 기반 방식의 한계를 극복하고, 획득 시간은 단축시키면서 구조적 세부사항이 뚜렷한 고해상도 이미지를 복원할 수 있다. 제안된 방법은 기존 최첨단(SOTA) 방법보다 고주파 신호 회복에서 성능을 뛰어넘는 것으로 보인다.

- **Technical Details**: 연구에서는 SM 복원을 위해 스윈(Swin) 변환기 기반의 자체 적응 다중 스케일 이동 윈도우 변환기 구조를 채택하여, 신호 유사성과 구조를 감독 정보로 활용하여 신호 분포를 학습한다. 제안된 RIM-임베딩 방법은 실제, 허상 및 크기 정보를 3개 채널의 입력으로 인코딩하며, 신호의 진폭, 위상 및 주파수 특성에 대한 사전 정보를 토대로 신호 분포를 학습하도록 지원한다.

- **Performance Highlights**: 두 개의 시뮬레이션 데이터셋과 네 개의 공개 데이터셋에서 평가한 결과, 제안된 방법이 SOTA 방법들보다 우수한 성과를 보였다. 특히, 다운 샘플링 배율이 16일 때, 15초 미만에 고해상도 SM을 복원할 수 있으며, 이는 기존 측정 기반의 SM에 비해 60배 이상 빠른 속도다. 이로 인해 세 가지 내부 MPI 시스템에서도 성능을 향상시키는 결과를 도출하였다.



### DGQ: Distribution-Aware Group Quantization for Text-to-Image Diffusion Models (https://arxiv.org/abs/2501.04304)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 텍스트-이미지 확산 모델의 양자화(quantization)에서 발생하는 어려움을 분석하고, 이미지 품질과 텍스트-이미지 정합성(text-image alignment)을 동시에 유지하는 새로운 접근법인 Distribution-aware Group Quantization (DGQ)을 제안합니다. 이 방법은 픽셀별 및 채널별 아웃라이어(outlier)를 식별하여 처리하는데 초점을 맞추어 이미지 품질을 보존합니다. 그뿐만 아니라, 프롬프트에 따라 로그 양자화(logarithmic quantization) 스케일을 적용하여 텍스트와 이미지 간의 정합성을 유지합니다.

- **Technical Details**: 제안된 DGQ 방법은 두 가지 주요 도전 과제를 다룹니다. 첫 번째는 아웃라이어가 포함된 활성화(activation)로, 이는 이미지 품질을 보존하는 데 중요한 역할을 합니다. 두 번째는 크로스-어텐션(cross-attention)에서의 독특한 패턴으로, 이는 텍스트-이미지 정합성에 큰 영향을 미칩니다. DGQ는 아웃라이어 보존 그룹 양자화를 통해 이러한 아웃라이어를 픽셀 및 채널 단위로 그룹화하고, 입력 프롬프트에 따라 각기 다른 양자화 스케일을 적용합니다.

- **Performance Highlights**: DGQ 방법은 MS-COCO 및 PartiPrompts와 같은 데이터셋에서 우수한 성능을 보였습니다. MS-COCO 데이터셋에서는 FID 점수를 13.15로 줄였으며, 이는 풀 정밀도(full precision)보다도 낮은 점수입니다. 또한, DGQ는 양자화 후 93.7%의 비트 연산 절약(비트 작업 수 694 TBOPs에서 43.4 TBOPs로)을 달성하였습니다. 본 연구는 추가 파인튜닝 없이 8비트 이하의 텍스트-이미지 확산 모델에서 저비트 양자화를 최초로 성공적으로 달성했습니다.



### Circuit Complexity Bounds for Visual Autoregressive Mod (https://arxiv.org/abs/2501.04299)
- **What's New**: 이 연구는 Visual AutoRegressive (VAR) 모델의 회로 복잡성(circuit complexity)에 대한 경계를 설정하고, VAR 모델이 $	ext{TC}^0$ 임계 회로(threshold circuit)의 시뮬레이션으로 동등함을 증명합니다. 이 회로는 은닉 차원(hidden dimension) $d 	ext{(d)}$가 $O(n)$ 이하이고, $	ext{poly}(n)$ 정밀도(precision)를 가지고 있습니다. VAR 모델은 이전 기술인 Diffusion Transformers를 초월하는 이미지 생성 능력을 보여주며, 본 연구는 이러한 모델의 표현력 표현의 한계를 철저히 분석한 첫 번째 연구입니다.

- **Technical Details**: 본 연구에서는 VAR 모델의 구조와 구성 요소(예: 업샘플링(interpolation) 레이어, 컨볼루션(convolution) 레이어, Transformer 블록 등)의 계산 복잡성(computational complexity)을 분석합니다. 회로 복잡성 이론을 통해 VAR 모델을 복잡성 회로(complexity circuits)로 표현함으로써, 이 모델이 수행할 수 있는 문제의 하한(bounds)을 정량적으로 평가할 수 있는 방법론을 제시합니다. 특히, $	ext{DLOGTIME}$-균일 $	ext{TC}^0$ 회로 패밀리가 VAR 모델을 O(1) 깊이(depth), $	ext{poly}(n)$ 크기(size), 및 $	ext{poly}(n)$ 정밀도로 시뮬레이션 가능함을 보여줍니다.

- **Performance Highlights**: VAR 모델은 기존의 이미지 생성 방법들과 비교했을 때, 더욱 사실적이고 다양한 이미지를 생성하는 능력을 보여줍니다. 특히, VAR 모델은 제로 샷 제너럴라이제이션(zero-shot generalization) 능력을 갖춰 이미지 인페인팅(image inpainting) 및 조작(manipulation) 작업 등 다양한 분야에서 뛰어난 성능을 발휘합니다. 본 연구는 VAR 모델의 표현력의 한계를 밝혀내며, 이로 인해 더 효율적이고 표현력 있는 아키텍처 개발에 기여할 가능성을 지니고 있습니다.



### MAD-UV: The 1st INTERSPEECH Mice Autism Detection via Ultrasound Vocalization Challeng (https://arxiv.org/abs/2501.04292)
Comments:
          5 pages, 1 figure and 2 tables. For MAD-UV Challenge 2025

- **What's New**: MAD-UV (Mice Autism Detection via Ultrasound Vocalization) 챌린지가 발표되었습니다. 이 챌린지는 마우스의 음성 발화를 기반으로 자폐 스펙트럼 장애(ASD)를 감지하는 최초의 INTERSPEECH 챌린지입니다. 참가자들은 고속 샘플링 비율로 녹음된 데이터를 바탕으로 마우스를 ASD 모델 또는 정상형으로 자동 분류하는 모델을 개발해야 합니다. 연구 결과는 자폐 조기 감지의 가능성을 시사합니다.

- **Technical Details**: 이 연구에서는 약 7777시간의 초음파 음성을 포함하는 데이터셋을 사용하여 848개의 주제에서 마우스의 음성 발화를 분석하였습니다. 기본 시스템은 세 가지 서로 다른 스펙트로그램 특징을 사용하는 간단한 CNN 기반 분류기를 적용했습니다. 연구 결과, 과제를 위한 오디오 신호의 자동화된 분석이 가능한 것을 나타냈으며, 주제 수준 분류에서 0.625의 UAR을 기록했습니다. 초음파 발화는 마우스의 건강 및 행동의 지표로 활용될 수 있습니다.

- **Performance Highlights**: 초음파 음성을 활용한 ASD 모델 선정에서 높은 성능을 보였습니다. UAR 기준으로 세그먼트 수준에서 0.600 및 주제 수준에서 0.625의 결과를 기록하면서, 자폐 스펙트럼 장애 감지의 가능성과 그 활용 가능성을 증명했습니다. 이 챌린지는 음성 기술 및 생물 의학 연구의 진전을 위한 중요한 발판이 될 것입니다. 차후 참가자들은 인간 음성 분석에서 기존 모델과 기법을 마우스의 초음파 발화 감지에 적용하는 탐색을 권장받고 있습니다.



### ContextMRI: Enhancing Compressed Sensing MRI through Metadata Conditioning (https://arxiv.org/abs/2501.04284)
Comments:
          29 pages, 9 figures

- **What's New**: 본 연구에서는 ContextMRI라는 텍스트 조건화 확산 모델을 제안하여 MRI 재구성 과정에 임상 메타데이터를 통합했습니다. 이 모델은 기존의 MRI 재구성 방식이 간과했던 환자 인구 통계, 영상 매개변수와 같은 중요한 정보를 활용하여 더 정확한 이미지를 생성할 수 있도록 설계되었습니다. 이러한 접근은 CS-MRI에서 재구성 성능을 획기적으로 향상시키는 것으로 나타났습니다.

- **Technical Details**: ContextMRI는 최소한의 처리된 복소수 MRI 이미지를 기반으로 훈련된 픽셀-공간 확산 모델입니다. 훈련 과정에서 CLIP 텍스트 임베딩을 활용하여 메타데이터를 구조화된 텍스트 프롬프트로 변환하고 모델에 공급합니다. 이 과정은 재구성에 필요한 사전 정보를 조건화함으로써 이루어집니다.

- **Performance Highlights**: 실험 결과, 다양한 데이터 세트 및 샘플링 패턴에 걸쳐 제안된 모델이 일관되게 높은 재구성 성능을 보였습니다. Meta데이터의 충실도가 증가할수록 재구성 품질이 체계적으로 향상되는 경향이 있는 것으로 나타났습니다. ContextMRI는 무조건적인 접근 방식과 비교할 때 모든 상황에서 두드러진 성능 향상을 보여주었습니다.



### Cluster & Disperse: a general air conflict resolution heuristic using unsupervised learning (https://arxiv.org/abs/2501.04281)
- **What's New**: 본 연구에서는 공중 충돌 해결 문제에 대한 일반적이고 변형 가능한 휴리스틱을 제시합니다. 이 휴리스틱은 비지도 학습을 활용하여 충돌 지점을 클러스터링하고 다양한 비행 고도에서 이를 분산시키는 새로운 이웃 구조에 기반합니다. 이를 통해 'Cluster & Disperse' 알고리즘을 최초로 도입하여, 비행 경로의 효율성을 높이면서 충돌을 완화할 수 있습니다.

- **Technical Details**: 'Cluster & Disperse' 알고리즘은 각 클러스터에서 문제를 일으키는 비행을 다른 비행 고도로 할당하여 이들을 섞어 최적의 비행 구성을 만들어냅니다. 두 가지 주요 기술적 접근법을 사용하는데, 첫째는 단계별 클러스터의 형성과 이를 기반으로 하는 비행 고도 재배치이며, 둘째는 RF-legs를 사용하여 비행 경로를 곡선으로 만드는 방법론입니다. 이 과정은 그라디언트 하강법과 사회적 힘을 통해 충돌 지점을 공간적으로 클러스터링하고 분산시킵니다.

- **Performance Highlights**: 'Cluster & Disperse'의 각 반복은 대부분의 평면 충돌 해결 알고리즘의 실행 시간과 비교해 아주 짧은 시간 내에 수행됩니다(초기화에는 10초 미만, 반복 추적에는 0.01초 미만 소요). 이로 인해 이 프레임워크는 대부분의 실시간 응용 프로그램에 적합합니다. 특히 이 알고리즘은 다양한 제약 조건을 수용할 수 있는 유연성과 적응성을 제공합니다.



### Bridging Adaptivity and Safety: Learning Agile Collision-Free Locomotion Across Varied Physics (https://arxiv.org/abs/2501.04276)
Comments:
          11 Pages, 6 Figures

- **What's New**: 이번 논문에서는 BAS(Bridging Adaptivity and Safety)라는 새로운 방법론을 소개합니다. BAS는 기존의 Agile But Safe (ABS) 연구를 기반으로 하며, 동적인 환경에서 안전성을 높이도록 설계되었습니다. BAS는 신속한 장애물 회피를 위한 agile policy와 충돌 방지를 위한 recovery policy를 포함하며, 이러한 정책은 물리적 파라미터에 따라 조정됩니다.

- **Technical Details**: BAS는 agile policy와 RA (reach-avoid) 네트워크를 결합하여, 정책 전환을 제어하는 학습된 제어 이론적 구조를 가집니다. 또한, 물리적 파라미터 추정기는 agile policy와 동시에 훈련되어 정확도와 강인성을 높입니다. 분포 이동(distribution shift) 문제를 완화하기 위해, on-policy fine-tuning 단계를 추가하여 estimator의 성능을 개선합니다.

- **Performance Highlights**: 시뮬레이션 결과, BAS는 동적인 환경에서 안전성을 50% 향상시키는 동시에 평균 속도를 유지합니다. 실제 실험에서도 BAS는 슬라이퍼리 바닥과 같은 복잡한 환경에서 우수한 성능을 보여주며, ABS에 비해 19.8% 속도를 증가시키고 2.36배 낮은 충돌률을 기록했습니다.



### On weight and variance uncertainty in neural networks for regression tasks (https://arxiv.org/abs/2501.04272)
Comments:
          Submitted to journal

- **What's New**: 본 논문은 Blundell et al. (2015)의 연구를 바탕으로, 신경망(Neural Networks)에서의 가중치 불확실성(weight uncertainty) 문제를 다룹니다. 이 연구는 기존 모델에서 분산 불확실성(variance uncertainty)을 포함하는 것이 Bayesian Neural Networks (BNNs)의 예측 성능을 향상시킬 수 있음을 보여줍니다. 특히, 분산 불확실성이 모델의 일반화 능력을 어떻게 개선하는지를 살펴봅니다.

- **Technical Details**: Bayesian Neural Networks에서는 모델 파라미터의 사전 분포(prior distribution)를 고려하여 파라미터의 후방 분포(posterior distribution)를 추정하고자 합니다. MCMC(Markov Chain Monte Carlo)와 같은 방법을 사용하여 후방 밀도를 근사하는 과정에서 계산 효율성과 안정성을 높이기 위해 Variational Bayes (VB)를 적용합니다. 본 연구에서는 Variational Bayes 방법과 가중치의 분산 불확실성을 활용한 Bayes by Backprop 기법을 소개합니다.

- **Performance Highlights**: 실험 결과, 가변 분산을 고려한 Bayes by Backprop 기법이 고정된 분산을 사용한 모델에 비해 더 나은 성능을 보임을 확인하였습니다. 논문에서는 비선형 함수 근사 문제와 리보플라빈 유전자 데이터셋을 사용하여 제안된 모델의 성능을 평가하였습니다. 결과적으로, 제안된 방법은 MSPE 및 커버리지 확률 측면에서 우수한 성능을 나타냈습니다.



### Integrated Offline and Online Learning to Solve a Large Class of Scheduling Problems (https://arxiv.org/abs/2501.04253)
- **What's New**: 이 연구에서는 단일 머신 스케줄링 문제를 다루기 위해 통합된 머신러닝(Machine Learning) 접근 방식을 개발했습니다. 이 방법은 비감소(min-sum) 목표 함수를 가지고 있으며, 릴리스 시간(release times)이 있을 수도 없을 수도 있습니다. 특히, 시간 인덱스(time-indexed) 공식을 사용하여 전체 클래스 문제를 통합적으로 구성했습니다.

- **Technical Details**: 기술적인 측면에서, 저자들은 깊은 신경망(Deep Neural Network) 모델을 사용하여 비용 매개변수(cost parameters)를 입력으로 활용하고, 연속적인 솔루션을 예측하여 이로부터 정수 해답(discrete solution)을 쉽게 구축할 수 있도록 했습니다. 또한, NP-hard 성질로 인해 최적 솔루션의 라벨을 생성하기 어려운 문제를 해결하기 위해 특별한 인스턴스를 생성해 모델을 오프라인으로 학습했습니다. 추가로, 온라인 단일 인스턴스 학습을 통해 주어진 인스턴스에 대해 DNN의 매개변수를 미세 조정하는 방법도 포함되었습니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 방법은 최대 1000개의 작업을 포함하는 다양한 단일 머신 스케줄링 min-sum 문제에 대해 효율적으로 고품질 솔루션을 생성할 수 있음을 보여주었습니다. 이러한 결과는 효율적인 스케줄링 해결책을 제공하는 데 있어 머신러닝의 가능성을 입증합니다.



### Statistical Uncertainty Quantification for Aggregate Performance Metrics in Machine Learning Benchmarks (https://arxiv.org/abs/2501.04234)
Comments:
          LA-UR-24-25289; presented at the Workshop on Statistical Frontiers in LLMs and Foundation Models at NeurIPS 2024

- **What's New**: 이번 연구에서는 다중 작업에 걸친 성능 메트릭의 불확실성을 정량화하는 통계적 방법론의 활용을 보여줍니다. 기존의 평가 방식은 특정 모델의 성능을 단순한 포인트 추정에 의존하는 경향이 있지만, 이는 샘플링 변동성을 간과함으로써 잘못된 결론을 초래할 수 있습니다. 본 연구에서는 Visual Task Adaptation Benchmark (VTAB)를 통해 모델 성능의 보다 현실적인 이해를 돕기 위한 다양한 통계적 분석 방법을 적용합니다.

- **Technical Details**: 이 연구에서는 주로 부트스트래핑(bootstrapping), 베이지안 계층 모델링(Bayesian hierarchical modeling)과 같은 통계적 방법을 사용하여 성과 메트릭을 집계하는 방법을 설명합니다. 각 모델의 성과 메트릭을 다양한 작업(task)에 대해 집계하는 과정에서 이러한 통계적 파라미터의 유용성을 강조합니다. 특히, 여러 작업에 대한 성과 메트릭을 통합하는 데 있어 정량적 불확실성을 고려하는 것이 중요하다는 점을 부각시킵니다.

- **Performance Highlights**: VTAB을 활용한 연구 결과, 특정 작업에 대해 점수화된 모델들이 전반적으로는 낮은 성과를 보일 수 있지만, 특정 작업에서는 두드러지는 성능을 발휘할 수 있다는 점을 알 수 있었습니다. 이러한 결과는 모델 성능을 비교하는 데 있어 표준 오차(standard errors)에 기반한 다양한 시각화를 통해 보다 깊은 통찰력을 제공하며, 다른 작업 가중치(task weightings)의 영향을 고려할 수 있게 합니다. 연구에서는 이와 같은 분석이 실제로 프리트레인(pretrained)된 모델을 사용하는 모든 실무자에게 귀중한 통찰을 제공할 수 있음을 강조합니다.



### Constraints as Rewards: Reinforcement Learning for Robots without Reward Functions (https://arxiv.org/abs/2501.04228)
- **What's New**: 이 논문에서는 로봇 행동 생성에 대한 기존 보상 함수 조정 과정을 피하기 위해 'Constraints as Rewards (CaR)', 즉 제약 조건을 보상으로 사용하는 새로운 접근 방식을 제안합니다. CaR는 로봇의 작업 목표를 제약 조건 함수로만 구성하여 보상 함수를 제거하고, Lagrangian 방법을 사용하여 여러 목표 간의 가중치를 자동으로 조정할 수 있게 합니다. 이 접근 방식은 로봇이 의미 있는 행동을 더 효율적으로 학습할 수 있도록 돕습니다.

- **Technical Details**: 제안된 방법은 CaR을 통해 제약 조건을 사용하여 강화 학습 문제를 해결합니다. CaR는 QRSAC-Lagrangian 알고리즘으로 구현되며, 이는 기존 QRSAC 알고리즘의 확장입니다. 이 알고리즘은 Lagrange 승수를 사용하여 여러 목표를 자동으로 균형잡게 하며, 네 가지 특정 제약 함수 설계를 제안하여 각 작업 목표를 직관적으로 해석 가능하게 합니다.

- **Performance Highlights**: 실험에서는 제안된 기법이 여섯 바퀴의 텔레스코픽 다리 로봇인 Tachyon 3의 일어서는 동작 생성 작업에 효과적으로 적용되었습니다. 전통적인 수동 설계된 보상 함수를 활용하는 방식보다 더 빠르고 안정적인 학습을 달성하며, 실제 환경에서 학습한 정책이 시뮬레이션과 유사하게 성공적으로 작업을 수행함을 입증합니다.



### Agent Laboratory: Using LLM Agents as Research Assistants (https://arxiv.org/abs/2501.04227)
- **What's New**: 본 논문에서는 Agent Laboratory라는 자율적인 LLM(언어 모델 기반) 프레임워크를 소개합니다. 이 프레임워크는 인간이 제공하는 연구 아이디어를 바탕으로 문헌 검토, 실험, 보고서 작성의 세 단계를 거쳐 종합적인 연구 결과물(코드 저장소 및 연구 보고서 포함)을 생성합니다. 사용자는 각 단계에서 피드백과 지침을 제공할 수 있어 연구 과정이 더욱 원활하게 진행됩니다.

- **Technical Details**: Agent Laboratory는 다양한 최첨단 LLM을 활용하여 연구 과정을 자동으로 수행합니다. 이 시스템은 연구의 모든 단계를 포괄하며, 연구자들에게 과정 전반에서 인간의 피드백을 받을 수 있는 경로를 제공합니다. 연구 결과는 자율 연구 방식의 전통적인 비용과 시간을 크게 절감합니다.

- **Performance Highlights**: 연구 결과에 따르면, o1-preview 기반의 Agent Laboratory가 가장 우수한 연구 결과물을 생성했습니다. 생성된 머신러닝 코드는 기존 방법들과 비교하여 최첨단 성능을 달성하였으며, 각 단계에서의 인간의 참여가 연구의 전반적인 품질을 크게 개선하는 것으로 나타났습니다. 최종적으로 Agent Laboratory는 연구 비용을 84%나 절감하는 효과를 보였습니다.



### UPAQ: A Framework for Real-Time and Energy-Efficient 3D Object Detection in Autonomous Vehicles (https://arxiv.org/abs/2501.04213)
- **What's New**: 이 논문은 자율주행 차량(AV)에서 3D 객체 감지기의 효율성을 높이는 새로운 프레임워크인 UPAQ를 소개합니다. UPAQ는 반구조적 패턴 가지치기(semi-structured pattern pruning)와 양자화(quantization)를 활용하여 자원 제약이 있는 내장형 AV 플랫폼에서 LiDAR 포인트 클라우드와 카메라 기반 3D 객체 검출기의 효율성을 향상시킵니다.

- **Technical Details**: UPAQ는 최신 모델 압축 프레임워크에 비해 Pointpillar와 SMOKE 모델에서 각각 최대 5.62배 및 5.13배의 모델 압축률을 달성합니다. 또한, 추론 속도가 최대 1.97배 및 1.86배 향상되며, 에너지 소비는 최대 2.07배 및 1.87배 감소합니다. 이러한 결과는 Jetson Orin Nano 내장 플랫폼에서 실시된 실험을 기반으로 합니다.

- **Performance Highlights**: UPAQ 프레임워크는 기존의 2D 객체 감지기보다 더 포괄적인 예측을 제공하면서도, 자원 제한이 있는 상황에서 높은 성능을 발휘합니다. 성능 측면에서 더 빠른 추론 시간과 낮은 에너지 소비를 결합하여, 자율주행 기술의 발전을 이끌고 있습니다.



### Generative Dataset Distillation Based on Self-knowledge Distillation (https://arxiv.org/abs/2501.04202)
Comments:
          Accepted by ICASSP 2025

- **What's New**: 새로운 연구에서는 dataset distillation의 효율성을 향상시키기 위해 self-knowledge distillation을 도입한 새로운 생성 기반 데이터셋 디스틸레이션 방법을 제안했습니다. 이 방법은 원본 데이터와 합성 데이터 간의 예측 로짓(logits) 정렬의 정확성을 향상시키고, 보다 효과적인 데이터 구조와 관계를 포착합니다. 특히, 정규화를 통해 로짓의 일관성을 유지함으로써 보다 정교한 분포 매칭(distribution matching)을 도모하고 있습니다.

- **Technical Details**: 본 연구의 방법론은 두 단계로 구성됩니다. 첫 번째로, 조건부 생성적 적대 신경망(GAN)을 학습하여 합성 데이터셋 S를 생성합니다. 이후 모델 풀(pool)에서 무작위로 선택된 모델을 통해 original 데이터셋 O와의 정렬을 수행하며, self-knowledge distillation을 통합하여 원본 및 합성 데이터의 분포를 효과적으로 일치시킵니다. 또한, 로짓의 범위를 일관되게 유지하기 위한 정규화 단계를 도입하여 정렬의 정확성을 높였습니다.

- **Performance Highlights**: 제안된 방법은 여러 벤치마크 데이터셋을 통해 기존의 최신 방법들보다 우수한 성능을 보여주었습니다. 특히, 생성자가 보다 정확하고 대표성 있는 합성 데이터를 생성하도록 돕고, 결과적으로 더 높은 품질의 데이터 디스틸레이션을 가능하게 했습니다. 이러한 접근 방식은 다양한 모델 아키텍처에서 통칭성과 성능을 개선하는데 기여하고 있습니다.



### Comparison of Neural Models for X-ray Image Classification in COVID-19 Detection (https://arxiv.org/abs/2501.04196)
Comments:
          9 pages, 7 tables, 5 figures. XXXIX SIMPOSIO BRASILEIRO DE TELECOMUNICACOES E PROCESSAMENTO DE SINAIS - SBrT 2021

- **What's New**: 이번 연구에서는 COVID-19 감염을 감지하기 위한 방안으로 방사선 이미지 분석 방법들을 비교하였습니다. 공공 데이터셋에서 확보한 이미지를 '정상(normal)', '폐렴(pneumonia)', 'COVID'의 세 가지 클래스으로 분류하였습니다.

- **Technical Details**: 실험에는 SqueezeNet, DenseNet, ResNet, AlexNet, VGG, GoogleNet, ShuffleNet, MobileNet의 여덟 가지 사전 훈련(pre-trained) 네트워크가 사용되었습니다. DenseNet은 다중 클래스(multiclass) 접근 방식에서 ADAM 최적화 함수(optimization function)를 활용하여 97.64%의 정확도를 기록하였습니다.

- **Performance Highlights**: 이진 분류(binary classification) 접근법에서는 VGG, ResNet, MobileNet 네트워크가 99.98%의 최고 정밀도(precision)를 달성하였습니다. 또한, 열 지도를 활용한 비교 평가(heat maps)는 이 연구의 중요한 종합적 결론을 제공하였습니다.



### STLCG++: A Masking Approach for Differentiable Signal Temporal Logic Specification (https://arxiv.org/abs/2501.04194)
Comments:
          To be submitted to robotics journal for review

- **What's New**: 이번 논문에서는 STLCG++를 제안하여 기존의 STL 로버스트니스 평가 및 역전파 과정을 마스킹 기반으로 병렬화함으로써 1000배 이상의 빠른 계산 속도를 실현합니다. 이는 긴 시퀀스에서 STL 요구조건을 효율적으로 처리할 수 있는 새로운 가능성을 제시하며, 특히 로봇 제어 및 학습 응용 분야에 적합합니다. 더불어, 시간 간격의 경계에 대한 부드러운 함수(Smoothing function)를 도입하여 STL 로버스트니스 값을 시간 매개변수에 대해 미분할 수 있도록 하였습니다.

- **Technical Details**: STLCG++는 트랜스포머 아키텍처에서 영감을 받아 마스킹 기법을 활용하여 STL 로버스트니스의 효율적인 평가 및 역전파가 가능하도록 합니다. 기존의 RNN 기반 접근법 대신, 병렬 처리를 통해 실시간 데이터에서 적용할 수 있는 가능성을 늘립니다. 이 과정에서 JAX 및 PyTorch와 같은 최첨단 자동 미분(Automatic Differentiation) 라이브러리를 활용하여 효율적인 계산 그래프를 구축합니다.

- **Performance Highlights**: 우리는 STLCG++의 장점을 세 가지 로봇 사용 사례를 통해 입증하였으며, 제안된 방법이 기존의 STL 평가 기법에 비해 현저히 높은 효율성을 보여준다고 강조합니다. 개발된 오픈소스 라이브러리는 최신 로봇워크플로우에 원활하게 통합될 수 있도록 설계되었습니다. 이러한 접근법은 향후 스페이쇼-템포럴(Spatio-Temporal) 행동 생성 및 제어 합성과 같은 다양한 응용 분야로의 확장을 가능하게 합니다.



### Generation from Noisy Examples (https://arxiv.org/abs/2501.04179)
Comments:
          19 pages. arXiv admin note: text overlap with arXiv:2410.13714

- **What's New**: 이번 논문에서는 Kleinberg와 Mullainathan (2024), Li et al. (2024)의 기존 연구 결과를 noisy example streams(노이즈 예제 스트림)을 고려하여 확장하였습니다. 저자들은 generator(생성기)가 새로운, 보지 못한 positive examples(양성 예제)를 출력하는 목표를 달성하기 위해 필요한 조건을 제시합니다. 특히, 이 논문에서는 noisy setting(노이즈 환경)에서의 binary hypothesis class(이진 가설 클래스)의 generatability(생성 가능성) 조건을 탐구합니다.

- **Technical Details**: 논문에서 제안하는 'noisy uniform generatability'(노이즈 균일 생성 가능성), 'noisy non-uniform generatability'(노이즈 비균일 생성 가능성), 'noisy generatability in the limit'(한계에서의 노이즈 생성 가능성) 개념은 이전 연구인 Kleinberg와 Mullainathan (2024) 및 Li et al. (2024)의 개념을 확장합니다. 또한, Noisy Closure dimension(노이즈 클로저 차원)이라는 새로운 척도를 통해 노이즈가 있는 예제의 생성 가능성을 완전히 특징화하고 있습니다. 이 클래스가 noisily uniformly generatable(노이즈 균일 생성 가능)하기 위한 조건도 제시되었습니다.

- **Performance Highlights**: 노이즈 예제가 포함된 환경에서도 이진 가설 클래스의 생성 가능성은 크게 영향을 받지 않는다는 흥미로운 결과를 도출하였습니다. 특히, finite(유한) 및 countable(가산) 클래스의 경우, 생성 가능성은 제한된 수의 noisy examples(노이즈 예제)가 포함되어 있어도 크게 변하지 않는 것으로 나타났습니다. 이는 실질적 적용에 있어서 중요한 의미를 가지며, adversarial robustness(적대적 강인성) 등 다양한 문제에 대한 대응력을 높일 수 있는 기초를 제공합니다.



### Learning to Transfer Human Hand Skills for Robot Manipulations (https://arxiv.org/abs/2501.04169)
Comments:
          Preprint. Under Review

- **What's New**: 이번 논문에서는 로봇이 인간의 손 동작을 통해 섬세한 조작 작업을 학습할 수 있는 새로운 방법을 제안합니다. 기존 방법들은 로봇과 객체 간의 상호작용의 타당성을 고려하지 않고 운동학(kineamatics) 정보에만 의존하였으나, 이 방법은 인간의 동작 시연에서 로봇의 조작 동작을 직접 추론합니다. 이로써 인간 손과 로봇 시스템 간의 신체성 차이를 해결하는 학습 기반 방법을 구사하여, 실제 실험에서 전통적인 리타게팅(retargeting) 기법보다 훨씬 높은 성능을 보여주고 있습니다.

- **Technical Details**: 제안된 방법은 인간 손 동작, 로봇의 손 동작, 그리고 객체의 3D 움직임 간의 공동 운동 매니폴드(joint motion manifold)를 학습하는 것을 목표로 합니다. 이를 통해 각 움직임 요소 간의 관계를 파악하고 인간 동작 시연에서 얻은 정보를 바탕으로 로봇의 조작 경로를 추론할 수 있습니다. 핵심 아이디어는 인간, 객체, 로봇의 움직임 경로를 조합하여 생성된 가상 감독 삼중쌍(pseudo-supervision triplets)을 활용하여 학습하는 것입니다.

- **Performance Highlights**: 실험 결과, 제안된 데이터 기반 리타게팅 방법은 복잡한 덱스터러스 조작 작업을 해결하는 데 있어 매우 효과적임을 입증했습니다. 이는 기존의 리타게팅 기법이 가지는 한계를 극복하면서도, 인간의 손 움직임을 로봇이 더욱 자연스럽고 물리적으로 타당하게 모사할 수 있도록 합니다. 특히, 이 방법은 다양한 손/객체 동작에 일반화 가능해, 향후 인간의 동작 데이터를 활용한 로봇 조작의 발전을 기대하게 합니다.



### MM-GEN: Enhancing Task Performance Through Targeted Multimodal Data Curation (https://arxiv.org/abs/2501.04155)
- **What's New**: 이번 논문에서는 MM-Gen이라는 새로운 방법론을 소개합니다. 이 방법은 고품질의 합성 텍스트(annotation)를 생성하여 특정 작업(task)에 맞는 시각-언어 모델(VLM)의 성능을 높입니다. MM-Gen은 세 가지 단계의 프로세스를 통해 필요한 데이터를 효과적으로 생성하며, 기존의 양질의 데이터가 부족한 문제를 해결합니다.

- **Technical Details**: MM-Gen은 강력한 VLM을 활용하여 이미지와 연관된 텍스트 주석을 생성합니다. 이 과정은 크게 세 가지로 나뉘며, (1) 데이터를 소그룹으로 분리하고, (2) 작업 설명에 기초하여 목적에 맞는 텍스트 생성, (3) 불필요한 데이터 필터링을 통해 이루어집니다. 이를 통해 VLM을 fine-tuning하면 다수의 작업에서 성능을 크게 향상시킬 수 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, MM-Gen을 통해 Llava-1.5(7B) 모델은 차트 이해, 공간 추론 및 도표 이해에서 각각 15%, 14%, 29%의 성능 향상을 기록했습니다. 원본 모델에 비해 MM-Gen 데이터는 최대 1.6배 높은 개선 효과를 보이며, 이를 통해 특정 작업에서 VLM 성능을 증가시킬 수 있음을 입증합니다.



### Multilingual Open QA on the MIA Shared Task (https://arxiv.org/abs/2501.04153)
- **What's New**: 이번 논문은 Cross-lingual Information Retrieval (CLIR) 모델을 개발하여 저자원 언어에서도 효과적으로 검색할 수 있도록 하는 새로운 접근 방식을 제안합니다. 특히 이 연구는 추가적인 감독이나 레이블된 데이터를 필요로 하지 않으며, 다양한 언어에서 검색한 결과를 재조정하는 간단하고 효과적인 방법을 제공합니다.

- **Technical Details**: 제안된 재점수 기법은 사전 학습된 다국어 질문 생성 모델을 사용하여 검색된 패세지를 재조정합니다. 이 모델은 입력 질문의 확률을 계산하기 위해 retrieval된 패세지에 조건부로 입력 질문의 확률을 평가합니다. 이 과정에서 BM-25와 같은 희소 검색 방법으로 얻은 결과를 재조정할 수 있으며, 따라서 고가의 필수 레이블된 코퍼스를 얻지 않고도 저자원 언어에 활용할 수 있습니다.

- **Performance Highlights**: 저자들은 모델의 성능을 완전한 제로샷 환경에서 평가하였으며, 이 방법은 별도의 훈련 없이도 신뢰할 수 있는 결과를 보여주었습니다. 특히, 한국어와 벵골어를 포함한 저자원 언어의 데이터 증강이 이루어졌으며, 이를 통해 저자원 언어에 대한 성능이 개선될 것으로 기대하고 있습니다.



### Mixing Times and Privacy Analysis for the Projected Langevin Algorithm under a Modulus of Continuity (https://arxiv.org/abs/2501.04134)
Comments:
          40 pages, 2 figures

- **What's New**: 이번 연구에서는 projected Langevin algorithm (LA)의 mixing time과 noisy Stochastic Gradient Descent (SGD)의 privacy curve에 대한 새로운 결과를 제시합니다. 특히, LA의 mixing time에 대한 새롭고 중요한 차원 중립적인 경계와 poly-logarithmic accuracy에 대한 결과를 도출하였습니다. 이러한 결과는 기존의 smooth convex 케이스와 밀접하게 일치합니다.

- **Technical Details**: 연구에서는 Privacy Amplification by Iteration (PABI) 프레임워크를 확장하여 noisy iterations의 경계 문제를 다룹니다. 이를 위해 최적화 문제를 설계하여 Rényi divergence bound를 최대한으로 얻는 방법을 제공합니다. 연구의 주요 성과로는 비매끄러운 convex, 약한 smooth, 및 강한 소산 행위를 포함한 여러 경우에서 문제를 정확히 해결할 수 있음을 보였습니다.

- **Performance Highlights**: 연구에서 제시된 새로운 경계는 이전 연구와 비교하여 훨씬 날카롭고 새로운 결과를 제공합니다. 특히, 이 결과들은 convex한 손실 함수에 대한 분석을 강화하는 데 유용하며, 비매끄러운 케이스를 넘어 더 넓은 범위에서 적용될 수 있습니다. 또한, 각 적용 사례에 대한 Rényi divergence의 경계도 새로운 발견으로 간주됩니다.



### A Survey on Large Language Models with some Insights on their Capabilities and Limitations (https://arxiv.org/abs/2501.04040)
Comments:
          174 pages, to be submitted to a journal in a shorter version. arXiv admin note: text overlap with arXiv:2303.18223, arXiv:2303.17564, arXiv:2301.00234, arXiv:2303.08774, arXiv:2402.02315, arXiv:2210.03493, arXiv:2402.01817, arXiv:2407.21783, arXiv:2208.05051 by other authors

- **What's New**: 최근 인공지능 분야는 Transformers 아키텍처에 기반한 대형 언어 모델(LLMs)의 발전에 의해 크게 변화하였습니다. LLMs는 텍스트 생성, 질문 답변, 번역 및 요약과 같은 다양한 언어 관련 작업에서 사람처럼 이해하는 능력을 선보이며, 이는 자연어 처리 방식에 혁신을 가져왔습니다. 특히, 이 모델들은 코딩 생성, 기본 상식 추론 및 산술 계산과 같은 본연의 기능을 넘어서는 능력을 보여주기로 주목받고 있습니다.

- **Technical Details**: 이 논문에서는 LLM의 기본 구성 요소와 스케일링 메커니즘, 건축 전략을 탐구하며, 특히 GPT와 LLaMA와 같은 모델들에 중점을 둡니다. 급증하는 데이터와 계산 능력이 LLM 성능에 미치는 영향을 분석하며, 스케일링과 관련된 트레이드오프에 대해서도 논의합니다. 또한 LLM의 다양한 적용 사례에 대해 살펴보고, 이것이 의료, 금융, 교육, 법률 등 각 분야에서의 문제 해결 능력을 어떻게 나타내는지 설명합니다.

- **Performance Highlights**: LLM은 향상된 언어 이해 덕분에 복잡한 언어적 도전 과제를 해결할 수 있는 잠재력을 지니고 있습니다. LLM들은 코그니티브한 작업을 수행하는 데 필요한 계획 및 추론 능력을 갖추고 있으며, Chain of Thought(CoT) 및 Plan of Thought(PoT)와 같은 새로운 접근 방식을 통해 그 성능을 더욱 향상시킬 수 있습니다. 본 논문은 LLM의 능력과 한계를 지속적으로 탐구하여, 이 분야에서의 책임 있는 개발 및 활용 방안을 모색할 것입니다.



### Approximation Rates in Fr\'echet Metrics: Barron Spaces, Paley-Wiener Spaces, and Fourier Multipliers (https://arxiv.org/abs/2501.04023)
- **What's New**: 본 논문은 신경망을 이용하여 부분 미분 방정식(Partial Differential Equations, PDEs)의 해를 근사시키는 새로운 접근법인 Operator learning을 탐구합니다. 연산자(operator)의 행동을 학습하여, 무한 차원 공간에서의 매핑(mappings)을 시뮬레이션하는 능력을 개발하는 것이 이 방법의 핵심입니다. 특히, 우리는 고전적인 Hörmander-Symbols 구조와 유사한 식으로, 푸리에(Fourier) 도메인에서의 대칭(approximation) 능력을 연구합니다.

- **Technical Details**: 우리는 Semi-norms의 시퀀스를 통하여 유도된 위상(topology)에 대한 대칭 능력을 고려합니다. 이 과정에서 대칭 오차는 Fréchet metric을 통해 측정되며, 사전 정의된 대칭 오차를 달성하기 위한 충분 조건을 제시합니다. 또한, 기존의 대칭 결과를 기반으로, 구성 요소를 보다 간소화한 확장된 정리를 소개합니다.

- **Performance Highlights**: 신경 연산자 모델의 성능과 관련하여, 디지털 필터(digital filters)의 설계와 유사한 방식으로 설명된 상징(symbol)의 구체적 예시가 제시됩니다. 우리의 연구는 신경 연산자 모델이 무한 차원 문제를 잘 처리할 수 있는 유망한 대체 모델로 자리잡을 수 있음을 보여줍니다.



### MERCURY: A fast and versatile multi-resolution based global emulator of compound climate hazards (https://arxiv.org/abs/2501.04018)
- **What's New**: 이번 연구에서는 복합적 기후 위험 분석을 위한 다중 해상도 환경 모델(MERCURY)을 소개합니다. MERCURY는 기후 변수의 복합 에뮬레이션을 가능케 하는 경량 데이터 기반 대안으로, 공간-시간 프레임워크를 확장하여 다양한 기후 변수를 유연하게 모사합니다. 이 모델은 데이터 기반 이미지 압축 기법을 활용하여 메모리 효율적으로 에뮬레이션을 생성합니다.

- **Technical Details**: MERCURY는 연간 평균 온도(Global Mean Temperature, GMT)에 대한 기후 변수의 지역 월 평균 응답을 확률 회귀 기반 가법 모델을 사용하여 나타냅니다. 지역 평균 값을 그리드 셀로 역도로 분할하는 역 리프팅 스킴을 적용하여 다변수 에뮬레이션을 수행합니다. 이 과정에는 불규칙한 형태의 필드를 처리할 수 있는 이산 웨이브렛 변환 방법이 활용되며, 이를 통해 지역 간의 상관관계를 효율적으로 유지합니다.

- **Performance Highlights**: MERCURY는 습도와 온도를 기반으로 한 웻 벌브 글로브 온도(WBGT) 지표를 높은 정확도로 재현하며, 에뮬레이션된 WBGT의 공간적 상관관계는 지구 시스템 모델(ESM)과 잘 일치합니다. WBGT 분포의 95% 및 97.5% 분위수를 잘 포착하며 평균적으로 5% 이내의 오차를 보입니다. MERCURY의 설정을 통해 지역 특정의 에뮬레이션을 효과적으로 수행하며, 이는 다수의 변수를 그리드 셀 수준으로 "줌"인 하는 데 유용합니다.



### FlexCache: Flexible Approximate Cache System for Video Diffusion (https://arxiv.org/abs/2501.04012)
- **What's New**: 텍스트-비디오(Text-to-Video) 애플리케이션에 대한 관심이 높아지면서, 이번 논문에서는 Diffusion 모델을 활용한 새로운 접근 방식인 FlexCache를 소개합니다. FlexCache는 비디오 생성에서의 높은 연산 복잡성을 해결하기 위해 설계되었습니다. 특히, 기존의 텍스트-이미지 모델에서 사용되던 기술들은 비디오 생성에 적합하지 않다는 점을 지적하며, 새로운 캐시 시스템을 제안합니다.

- **Technical Details**: FlexCache는 두 가지 주요 디자인 요소로 구성되어 있습니다. 첫째, 캐시를 저장하기 전에 압축하는 전략을 통해 평균적으로 6.7배의 공간 절약을 달성합니다. 둘째, 물체와 배경을 분리하여 근사 캐시 시스템이 더 높은 히트율(hit rate)과 연산 절약을 이룰 수 있음을 발견했습니다. 이러한 기술을 지원하기 위한 맞춤형 캐시 교체 정책도 설계하였습니다.

- **Performance Highlights**: FlexCache는 기존의 최첨단 diffusion 근사 캐시 시스템에 비해 처리량(throughput)을 1.26배 향상시키고, 비용을 25% 절감하는 성과를 보였습니다. 이러한 결과는 비디오 생성에서의 효율성을 크게 향상시키며, 실용적인 텍스트-비디오 응용의 발전을 기대할 수 있습니다.



### Multi-SpaCE: Multi-Objective Subsequence-based Sparse Counterfactual Explanations for Multivariate Time Series Classification (https://arxiv.org/abs/2501.04009)
- **What's New**: 이 논문에서는 Multi-SpaCE라는 다목적 반사실적 설명 방법을 소개합니다. 이는 다변량 시계열 데이터를 지원하며, 효율적인 NSGA-II 유전자 알고리즘을 통해 근접성, 희소성, 실현 가능성 및 연속성의 균형을 맞춥니다. 기존 방법들과의 차별점으로는 완벽한 유효성을 보장하며, Pareto 프론트(Pareto front) 솔루션을 제공합니다.

- **Technical Details**: Multi-SpaCE는 다목적 최적화 프레임워크를 이용하여 반사실적 설명을 생성합니다. 이는 입력 데이터의 포인트를 수정하기 위한 최적화 작업을 포함하며, Nearest Unlike Neighbor (NUN)를 통해 특정 값을 대체합니다. 기존의 반사실적 설명 방법들이 단일 변수 데이터에 제한되는 반면, Multi-SpaCE는 다변량 데이터에 대한 적합성을 보장합니다.

- **Performance Highlights**: 실험을 통해 Multi-SpaCE는 다양한 데이터 세트에서 완벽한 유효성을 지속적으로 달성하며 기존 방법들에 비해 우수한 성능을 보여줍니다. 이는 복잡한 응용 분야에서 신뢰할 수 있는 반사실적 설명을 제공하는 데 기여하여 인공지능 시스템의 투명성과 신뢰성을 높입니다.



### DPO Kernels: A Semantically-Aware, Kernel-Enhanced, and Divergence-Rich Paradigm for Direct Preference Optimization (https://arxiv.org/abs/2501.03271)
- **What's New**: 이번 논문에서는 대형 언어 모델(LLM)과의 정렬(alignment) 문제를 해결하기 위한 새로운 접근법인 DPO-Kernels를 제안합니다. DPO-Kernels는 커널 방법(kernel methods)을 통합하여 고정된 편차(fixed divergences) 및 제한된 특성 변환(feature transformations)의 문제를 극복하고자 합니다. 이를 통해 다각적인 전이(transformations)와 손실(loss) 함수를 개선하였습니다.

- **Technical Details**: DPO-Kernels의 주요 기여는 네 가지입니다: (i) 다항식(polynomial), RBF, 마할라노비스(Mahalanobis), 스펙트럼(spectral) 커널을 활용한 커널화된 표현(kernelized representations)과 임베딩 기반(embedding-based) 및 확률 기반(probability-based) 목표를 결합한 하이브리드 손실(hybrid loss); (ii) 다양성 대안(divergence alternatives)으로 제센-샤논(Jensen-Shannon), 헬링거(Hellinger), 레니(Renyi), 바타차리야(Bhattacharyya), 워서스타인(Wasserstein), f-다양성(f-divergences)을 포함하여 더 큰 안정성(stability) 제공; (iii) 데이터 기반 선택(metrics)으로 최적의 커널-다양성 쌍을 자동으로 선택; (iv) 지역 정확성과(global modeling) 전역 모델링을 위한 계층적 혼합(hierarchical mixture) 커널입니다.

- **Performance Highlights**: 12개의 데이터셋에 대한 평가 결과, DPO-Kernels는 사실성(factuality), 안전성(safety), 추론(reasoning), 지침 따르기(instruction following)에서 최첨단 성능(state-of-the-art performance)을 보였습니다. 이 연구는 Heavy-Tailed Self-Regularization에 기초하여 LLM의 강건한 일반화(robust generalization)를 유지하며, 정렬 연구(alignment research)를 위한 포괄적인 자원(resources)을 제공합니다.




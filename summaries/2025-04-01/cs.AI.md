New uploads on arXiv(cs.CL)

### Harnessing the Reasoning Economy: A Survey of Efficient Reasoning for Large Language Models (https://arxiv.org/abs/2503.24377)
Comments:
          In Progress; Paper list Repo: this https URL

- **What's New**: 최근 대형 언어 모델(Large Language Models, LLMs)에서의 발전은 복잡한 추론(task reasoning) 작업 수행 능력을 크게 향상시켰습니다. 이 연구에서는 신속한 직관적 사고(System 1)에서 느리지만 깊은 사고(System 2)로의 전환이 이루어졌으며, 이는 작업의 정확성을 높이지만 계산 비용이 증가하는 단점을 동반합니다. 따라서 성능과 계산 비용 사이의 균형을 맞추는 추론 경제(reasoning economy)의 개념이 중요하다는 점을 강조하고 있습니다.

- **Technical Details**: 이 연구에서는 LLMs의 훈련 후 및 테스트 시간 추론 단계에서의 추론 경제를 분석합니다. 첫째, LLMs에서의 추론 비효율성의 원인을 파악하고, 둘째 다양한 추론 패턴의 행동을 분석하며, 셋째 추론 경제를 달성하기 위한 잠재적 해결책을 모색합니다. 최종적으로는 효율적인 LLMs를 달성하기 위한 도전 과제와 해결책을 명확하게 제시합니다.

- **Performance Highlights**: LLMs는 Chain-of-Thought prompting의 도입으로 다양한 언어 이해 및 생성 작업에서 뛰어난 성능을 발휘하고 있습니다. 그러나 모든 작업이 깊은 사고를 필요로 하지 않기 때문에, 각 작업의 복잡도에 맞게 계산 노력을 조정할 필요가 있습니다. 효율적인 리소스 사용을 위해 필요한 추론 단계를 강조하며, 불필요한 중복을 줄이고 동적으로 계산 노력을 조정하는 방법을 제안합니다.



### Query and Conquer: Execution-Guided SQL Generation (https://arxiv.org/abs/2503.24364)
- **What's New**: 본 논문은 SQL 텍스트 변환 작업에서 정확도를 크게 향상시키는 새로운 접근 방식을 제안합니다. 이 방법은 실행 결과를 활용하여 여러 후보 쿼리 중에서 가장 의미적으로 일관된 쿼리를 선택하여, 계산 집약적인 방법을 초월하는 작은 모델들이 최대 30배의 추론 비용을 줄이도록 합니다. 또한 기존 모델과 원활하게 통합될 수 있는 실용적이고 확장 가능한 경로를 제공합니다.

- **Technical Details**: 우리는 SQL 생성에 특화된 새로운 self-consistency 접근 방식을 제안합니다. 이 방법은 쿼리 출력에서 직접 의미적 동등성을 평가하기 위해 정확하고 근사적인 실행 기반 유사성 메트릭을 활용합니다. Minimum Bayes Risk (MBR) 디코딩 프레임워크를 통해 우리의 방법에 대한 이론적 정당성을 제공하며, 실행 행동으로 정의된 출력 공간에 self-consistency을 확장하는 방식을 소개합니다.

- **Performance Highlights**: 우리의 연구 결과에 따르면, 실행 기반 self-consistency를 적용함으로써 작은 모델들이 큰 모델들과 유사한 성능을 발휘할 수 있음을 보여줍니다. 특히, 7억 파라미터를 가진 Qwen 2.5 Coder는 O1 수준에 도달하면서 정확도를 거의 10% 개선하고 30배 낮은 추론 비용을 달성했습니다. 이러한 결과는 실제 SQL 생성 작업을 위한 강력한 후보로 우리 방법의 효율성과 확장성을 강조합니다.



### BEATS: Bias Evaluation and Assessment Test Suite for Large Language Models (https://arxiv.org/abs/2503.24310)
Comments:
          32 pages, 33 figures, preprint version

- **What's New**: 이 연구에서는 BEATS라는 새로운 프레임워크를 소개하여 대규모 언어 모델(LLMs)의 편향(Bias), 윤리(Ethics), 공정성(Fairness), 사실성(Factuality)을 평가하는 방법을 제안합니다. 이 프레임워크를 바탕으로 29개의 다양한 지표를 통해 모델의 성능을 측정하는 편향 벤치마크를 개발했습니다. 이러한 지표들은 사회적 편견의 지속 가능성을 정량적으로 평가할 수 있는 가능성을 제공합니다.

- **Technical Details**: BEATS 프레임워크는 LLM의 편향, 윤리, 공정성 및 사실성을 평가하기 위한 체계적이며 확장 가능한 절차를 제공합니다. 이 프레임워크의 핵심은 다양한 사고와 윤리적 기준을 탐구하기 위해 설계된 질문 데이터셋입니다. 연구자는 이러한 질문을 통해 LLM의 응답을 분석하고 그 결과를 구조적 데이터베이스에 저장하여 벤치마크 평가를 수행합니다.

- **Performance Highlights**: 실험 결과, 업계 선두 모델의 37.65% 출력에서 어떤 형태의 편향이 발견되었습니다. BEATS 프레임워크와 벤치마크는 LLM 평가의 일관성과 반응성을 높여야 할 필요성이 있다는 것을 강조합니다. 이 연구는 다양한 AI 모델의 공정성과 윤리적 기준에 대한 인식을 높이고 지속 가능한 AI 모델 개발을 촉진하는 데 목표를 두고 있습니다.



### A Systematic Evaluation of LLM Strategies for Mental Health Text Analysis: Fine-tuning vs. Prompt Engineering vs. RAG (https://arxiv.org/abs/2503.24307)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 활용한 정신 건강 텍스트 분석을 위한 세 가지 접근 방식을 체계적으로 비교하였습니다: prompt engineering, retrieval augmented generation (RAG), 그리고 fine-tuning입니다. 이 연구는 LLaMA 3를 사용하여 감정 분류 및 정신 건강 상태 감지 작업을 두 개의 데이터셋에서 평가하였습니다. 연구 결과, fine-tuning이 감정 분류에서 91%, 정신 건강 조건 분류에서 80%의 정확도를 달성하였으며, prompt engineering과 RAG는 보다 유연한 배포가 가능하지만 보통의 성능(40-68% 정확도)을 보여주었습니다.

- **Technical Details**: 정신 건강 텍스트 분석을 위한 LLM의 세 가지 접근 방식은 fine-tuning, prompt engineering, RAG입니다. 특히, fine-tuning은 높은 정확도를 요구하지만 많은 컴퓨팅 리소스와 대규모 훈련셋을 필요로 합니다. 반면, prompt engineering과 RAG는 상대적으로 적은 자원으로 보다 유연한 배포가 가능하게 하며, 다양한 설정에서 효과적으로 구현할 수 있다는 장점이 있습니다.

- **Performance Highlights**: 이 연구는 정신 건강 분야에서 LLaMA 3 기반 모델의 효과를 입증하였으며, 감정 분류 및 정신 건강 상태 분류에서 매우 높은 정확도를 기록했습니다. 이러한 결과는 임상 환경에서 LLM 기반 솔루션의 구현에 있어 중요한 통찰력을 제공합니다. 향후 정신 건강 평가 도구의 개발에 중요한 의미를 가지며, 높은 정확도의 fine-tuning 외에도 prompt engineering과 RAG 접근 방식이 자원과 배포 유연성 면에서 유효한 대안이 된다는 점을 강조하고 있습니다.



### Is analogy enough to draw novel adjective-noun inferences? (https://arxiv.org/abs/2503.24293)
Comments:
          8 pages (16 pages with appendix). Submitted to SCiL 2025

- **What's New**: 이 논문은 인간과 LLM(대형 언어 모델)이 새로운 형용사-명사 조합에 일반화할 수 있는 능력을 분석합니다. 기존 연구에서는 구성(composition) 메커니즘에 의존한다고 주장했지만, 본 연구는 유사성을 통한 유추(analogy)로도 이 작업을 수행할 수 있는지를 탐구합니다. 결과적으로, 두 접근 방법을 통해 유추가 일부 경우에 잘 작동하지만, 모든 경우를 설명할 수는 없다는 사실을 발견하였습니다.

- **Technical Details**: 연구는 두 가지 주요 방법을 사용하여 유추 추론을 탐구합니다. 첫 번째로, 데이터셋의 저주파와 제로 주파수 두 문구에 대해 유추를 통해 점수를 예측하는 계산 모델을 구축하였습니다. 두 번째로, 인간 참가자들에게 유사한 예제를 사용하여 유추 기반으로 추론하도록 요청하고, 이로부터 도출된 평가 분포를 비교 분석하였습니다.

- **Performance Highlights**: 연구 결과, 유추 모델은 많은 경우에 유사한 결과를 도출하지만 모든 추론 데이터를 완전히 설명할 수는 없다는 것을 밝혀냈습니다. 또한 유사성 모델은 새로운 문구에 대해 Ross et al. (2024)에서 제시한 최상의 LLM보다 성능이 떨어졌습니다. 이는 LLM이 단순히 유추를 사용하지 않으며, 경우에 따라서는 구성 메커니즘을 활용한다는 주장을 뒷받침합니다.



### Enhancing Large Language Models (LLMs) for Telecommunications using Knowledge Graphs and Retrieval-Augmented Generation (https://arxiv.org/abs/2503.24245)
Comments:
          This work has been accepted to ICC 2025 IEEE International Conference on Communications. copyright 2025 IEEE

- **What's New**: 본 논문은 통신 분야에서 대형 언어 모델(LLM)의 성능을 향상시키기 위해 지식 그래프(KG)와 검색 보강 생성(RAG) 기술을 결합한 새로운 프레임워크를 제안합니다. 이 프레임워크는 네트워크 프로토콜 및 표준과 같은 통신 관련 정보의 구조적 표현을 제공하여 LLM의 도메인 이해력을 높입니다. KG와 RAG의 통합을 통해 최신 정보를 동적으로 검색하여 LLM의 응답 생성을 지원합니다.

- **Technical Details**: 제안된 시스템 모델은 LLM, KG 및 RAG 프레임워크의 통합을 통해 통신 분야에서의 특정 언어 처리 성능을 향상시킵니다. KG는 핵심 엔터티와 그 관계를 캡슐화한 도메인 특화 KG를 구성하며, 사용자가 쿼리를 제출하면 관련 서브 그래프나 노드를 동적으로 검색합니다. 이러한 검색된 정보는 LLM에 통합되어 정확하고 맥락에 맞는 응답을 생성하는 데 사용됩니다.

- **Performance Highlights**: KG-RAG 모델은 통신 전용 데이터셋인 Tspec-LLM에서 질문 응답 작업에 대해 88%의 정확도를 달성했습니다. 이는 RAG 모델의 82% 및 LLM 전용 접근 방식의 48%와 비교할 때 더 높은 성능을 보입니다. 이러한 성과는 KG-RAG 모델이 기술적 쿼리를 정확하게 처리할 수 있다는 것을 보여줍니다.



### What, How, Where, and How Well? A Survey on Test-Time Scaling in Large Language Models (https://arxiv.org/abs/2503.24235)
- **What's New**: 최근 데이터와 파라미터를 확장하는 것에 대한 관심이 줄어들면서, 테스트 시간 확장(test-time scaling, TTS)이라는 새로운 연구 주제가 부각되었습니다. TTS는 대규모 언어 모델(large language models, LLMs)의 문제 해결 능력을 더욱 향상시킬 수 있음을 보여줍니다. 특히, 수학 및 코딩과 같은 전문적 추론 과제뿐 아니라 열려 있는 질문 응답(open-ended Q&A) 과제 등 일반 과제에서도 큰 breakthroughs를 가능하게 합니다.

- **Technical Details**: 본 연구는 TTS 연구의 네 가지 핵심 차원인 무엇을 확장할 것인지, 어떻게 확장할 것인지, 어디서 확장할 것인지, 얼마나 잘 확장할 것인지에 따라 구조화된 통합적이고 다차원적인 프레임워크를 제안합니다. 이러한 분류법을 기반으로 우리는 방법론, 적용 시나리오, 평가 측면을 포함한 광범위한 검토를 수행합니다. 각 기법의 기능적 역할을 강조하는 체계화된 분해를 통해 TTS의 주요 개발 경과를 정리합니다.

- **Performance Highlights**: TTS의 분석을 통해 우리는 실용적인 배치를 위한 가이드를 제공하고 몇 가지 오픈 챌린지를 식별합니다. 여기에는 추가 확장, 기법의 기능적 본질 명확화, 다양한 작업에 대한 일반화 및 기타 속성에 대한 통찰력이 포함됩니다. 이러한 방향은 향후 TTS 연구에서 중요한 기여를 할 수 있는 가능성을 제시합니다.



### BAR-Analytics: A Web-based Platform for Analyzing Information Spreading Barriers in News: Comparative Analysis Across Multiple Barriers and Events (https://arxiv.org/abs/2503.24220)
Comments:
          46 pages

- **What's New**: 이번 논문에서는 BAR-Analytics라는 웹 기반(open-source) 플랫폼을 소개합니다. 이 플랫폼은 지리적, 경제적, 정치적, 문화적 경계를 넘어서 뉴스 유포를 분석하는 데 설계되었습니다. 특히 러시아-우크라이나 전쟁과 이스라엘-팔레스타인 갈등을 사례로 하여 다양한 분석 방법을 통합하고 있습니다.

- **Technical Details**: BAR-Analytics는 네 가지 분석 방법인 propagation analysis, trend analysis, sentiment analysis, temporal topic modeling을 사용합니다. 350,000개 이상의 기사를 수집하고, 메타데이터(enrichment)를 활용하여 경제적 차이와 지리적 영향을 중심으로 분석하였습니다. 분석 결과는 coherence, sentiment polarity, topic frequency, trend shifts와 같은 주요 메트릭(metrics)을 기준으로 평가됩니다.

- **Performance Highlights**: 분석 결과 이스라엘-팔레스타인 갈등의 뉴스는 인권에 초점을 맞춘 부정적 감정을 가지는 반면, 러시아-우크라이나 갈등은 선거 간섭을 강조하며 더 긍정적인 경향을 보였습니다. 이러한 발견은 다양한 갈등에서 미디어 내러티브를 형성하는 정치적, 경제적, 지역적 요인의 영향을 강조합니다.



### Synthetic News Generation for Fake News Classification (https://arxiv.org/abs/2503.24206)
Comments:
          13 pages, 8 figures

- **What's New**: 이번 연구에서는 대형 언어 모델(Large Language Models, LLMs)을 사용하여 실제 기사를 바탕으로 사실을 조작하고 합성된 가짜 뉴스를 생성 및 평가하는 방법론을 제시합니다. 우리는 실제 기사에서 주요 사실을 추출하고 이를 수정하여 일관성을 유지하며 가짜 뉴스를 시뮬레이션하는 새로운 방법을 도입합니다. 또한, 생성된 콘텐츠의 품질을 평가하기 위한 평가 지표로 일관성(coherence), 비유사성(dissimilarity), 정확성(correctness)을 제안합니다.

- **Technical Details**: 이 연구에서는 합성 데이터(synthetic data)를 이용한 가짜 뉴스 분류의 적용을 조사하고, 전통적인 머신러닝 모델과 BERT와 같은 트랜스포머(transformer) 기반 모델을 비교합니다. 실험 결과, 트랜스포머 모델, 특히 BERT는 가짜 뉴스 탐지에 있어 합성 데이터를 효과적으로 활용하며, 적은 비율의 합성 데이터에서도 개선된 성과를 보입니다. 또한, 사실 검증 기능(fact verification features)은 사실적 불일치(factual inconsistencies)를 식별하는 데 중점을 두어 합성 가짜 뉴스를 구분하는 데 가장 유망한 결과를 제공합니다.

- **Performance Highlights**: 합성 데이터의 가능성을 통해 가짜 뉴스 탐지 시스템을 향상시킬 수 있다는 점을 강조합니다. 연구 결과는 향후 연구에 대한 귀중한 통찰을 제공하며, 합성 데이터 생성을 대상으로 하는 개선이 탐지 모델을 더욱 강화할 수 있음을 시사합니다. 전반적으로 본 연구는 합성 데이터의 효과적인 사용이 가짜 뉴스 탐지의 성능을 크게 향상시킬 수 있음을 보여줍니다.



### TwT: Thinking without Tokens by Habitual Reasoning Distillation with Multi-Teachers' Guidanc (https://arxiv.org/abs/2503.24198)
- **What's New**: 이 논문에서는 TwT (Thinking without Tokens)라는 새로운 접근법을 제안하여, 대형 언어 모델(LLMs)의 추론 시간 비용을 줄이는 동시에 성능을 유지합니다. Habitual Reasoning Distillation (HaRD) 방법을 활용하여 명시적 추론을 모델의 습관적 행동으로 내재화하는 방식을 소개합니다. 추가로, Dual-Criteria Rejection Sampling (DCRS) 기법을 통해 다양한 교사 모델의 도움을 받아 고품질의 증류 데이터셋을 생성합니다.

- **Technical Details**: TwT 방법은 첫째, DCRS를 이용해 다중 교사 LLM으로부터 생성된 pseudo-labels를 통해 비지도 학습 환경에 적합하게 합니다. 둘째, HaRD는 다단계 증류 프로세스를 통해 명시적 추론 능력을 학생 모델에 점진적으로 내재화합니다. 이 과정은 (a) 전체 추론 증류, (b) 추론 압축 증류, (c) 추론 없는 증류의 三단계로 이루어져 있으며, 이를 통해 훈련 시에만 추론 요구를 줄이는 효과를 가져옵니다.

- **Performance Highlights**: TwT 접근방식은 기존 증류 기법에 비해 최고 13.6%의 성능 향상을 이루는 동시에, 더 적은 수의 출력 토큰을 생성합니다. 실험 결과는 이 방법이 높은 효율로 비지도 학습 환경에 적합하게 적용 가능하다는 것을 보여주며, 추론 시간에 따른 계산 비용을 효과적으로 줄이는 방법을 제시합니다.



### Implicit In-Context Learning: Evidence from Artificial Language Experiments (https://arxiv.org/abs/2503.24190)
- **What's New**: 본 연구는 대형 언어 모델(LLMs)이 인간과 유사한 패턴 인식을 보이는지를 평가하기 위해 기존의 인공 언어 학습 실험을 재구성했습니다. 특히, gpt-4o 및 o3-mini 모델을 사용하여 형태론(morphology), 형태통사(morphosyntax), 구문(syntax)에서의 암묵적 학습(implicit learning) 능력을 분석했습니다. 연구 결과는 LLM과 인간 간의 언어적 도메인 특화(alignment)가 존재하며, o3-mini가 형태론에서 더 나은 일치를 보임을 시사합니다.

- **Technical Details**: 이 연구는 인지 과학(cognitive science)의 유형론을 기반으로 하여, LLM의 암묵적 학습 능력을 분석하기 위해 세 가지 고전적 인공 언어 학습 실험을 적응했습니다. 두 가지 최신의 GPT 모델, gpt-4o는 일반 언어 작업에 최적화된 반면 o3-mini는 명시적 추론(explicit reasoning)을 위해 미세 조정되었습니다. 이러한 모델들이 제시된 제한된 예시에서 구조화된 언어 지식을 신속하게 습득하는 방법을 연구하였으며, 예시 기반 일반화의 인지 및 계산 메커니즘에 대한 이해가 부족한 현주소를 보여주고 있습니다.

- **Performance Highlights**: 모델의 행동은 세 가지 암묵적 학습 실험 전반에 걸쳐 유의미한 변동성을 보였으나 명확한 패턴은 발견되지 않았습니다. 기존 연구들과 비교할 때, LLM의 성능은 특정 언어 도메인에서 인간의 훈련 방식과 연관된 차이점을 드러냈습니다. 앞으로의 연구는 이러한 인지 과정 간의 일치와 차이를 더 깊이 탐구할 수 있는 기반을 제공할 것으로 기대됩니다.



### Multi-Task Learning for Extracting Menstrual Characteristics from Clinical Notes (https://arxiv.org/abs/2503.24116)
- **What's New**: 이번 연구에서는 여성의 생리 건강을 위한 새로운 자연어 처리(Natural Language Processing, NLP) 파이프라인을 제안합니다. 이 파이프라인은 생리 주기의 주요 특성인 통증, 규칙성, 혈량 및 월경 사이 출혈을 추출하는 데 중점을 두고 있습니다. GatorTron 모델을 활용한 멀티태스크 프롬프트 기반 학습을 사용하여, 모두 100개 미만의 주석이 달린 임상 기록으로 훈련되었음에도 평균 F1-score가 90%를 달성하였습니다. 이 연구는 생리 건강 관련 정보를 추출하는 데 있어 다중 과제를 결합한 접근 방식의 효과를 보여줍니다.

- **Technical Details**: 연구에서는 Mount Sinai Data Warehouse의 전자 건강 기록(EHR)에서 200개의 임상 기록을 확보하였고, 이 중 생리 중인 환자에 대한 정보만을 선별하여 최종적으로 140개의 임상 기록을 훈련 세트와 테스트 세트로 나누어 사용하였습니다. 각 기록은 생리통의 유무, 생리통의 중증도, 생리 규칙성, 혈량 및 월경 사이 출혈의 존재 여부로 수동 주석이 달렸습니다. 연구진은 BM25와 MedEmbed-small-v0.1을 결합하여 하이브리드 검색을 통해 임상 기록에서 가장 관련성이 높은 텍스트 세그먼트를 식별하였고, 이를 통해 모델이 의사 결정에 필요한 정보를 보다 정확히 추출할 수 있도록 하였습니다.

- **Performance Highlights**: 제안된 방법은 기존의 기준 모델과 비교할 때 월경 특성을 효과적으로 추출하며 탁월한 성능을 보여주었습니다. GatorTron-Base 및 Meditron-3 모델을 활용하여 기존의 단일 작업 접근 방식보다 다중 작업 학습이 더 우수한 일반화 성능을 보여줍니다. 이 연구의 결과는 임상 기록에서 자동화된 정보 추출 기술을 향상시키고, 여성 건강 연구를 지원하는 데 기여할 것으로 기대됩니다.



### TeleAntiFraud-28k: A Audio-Text Slow-Thinking Dataset for Telecom Fraud Detection (https://arxiv.org/abs/2503.24115)
- **What's New**: 이번 연구에서는 자동화된 통신 사기 분석을 위해 특별히 설계된 첫 번째 오픈소스 오디오-텍스트 슬로우-씽킹 데이터셋인 TeleAntiFraud-28k를 소개합니다. 이 데이터셋은 자동 음성 인식(ASR) 기술을 통해 생성된 음성-텍스트 쌍을 포함하고 있으며, 이를 통해 높은 품질의 멀티모달 훈련 데이터를 제공합니다. 또한, 다중 에이전트 적대적 합성을 통해 새로운 사기 전술을 시뮬레이션하고, 정교한 주석을 제공합니다.

- **Technical Details**: TeleAntiFraud-28k는 다양한 방법론을 통해 구축되었습니다. 첫 번째는 ASR 기술을 이용해 익명 처리된 통화 기록을 텍스트로 변환한 후, 이를 기반으로 음성과 일치하도록 다시 생성하는 방식입니다. 두 번째로, 대형 언어 모델(LLM)을 활용해 의미적으로 풍부한 자가 지시 샘플링 전략을 적용하였으며, 마지막으로 다중 에이전트 프레임워크를 통해 통신 시나리오와 사기 유형을 모델링하였습니다. 이 데이터셋은 세 가지 주요 작업(시나리오 분류, 사기 탐지, 사기 유형 분류)을 포함합니다.

- **Performance Highlights**: 여러 최첨단 대형 오디오 언어 모델(LALM)에 대한 실험을 통해 TeleAntiFraud-28k 데이터셋의 효과를 확인했습니다. Qwen2Audio 모델을 TeleAntiFraud-28k 훈련 세트를 사용하여 미세 조정한 후, 성능에서 유의미한 개선을 보였습니다. 이러한 결과는 이 데이터셋이 오디오 기반의 사기 탐지 모델 개발에 실질적인 가치를 제공함을 입증합니다.



### Is LLM the Silver Bullet to Low-Resource Languages Machine Translation? (https://arxiv.org/abs/2503.24102)
- **What's New**: 이 논문은 Low-Resource Languages (LRLs)의 번역 문제들에 대해 새로운 접근법을 제시합니다. 기존의 LLMs와 Neural Machine Translation (NMT)의 성능 부족을 해결하기 위해 200개 언어를 대상으로 한 체계적인 평가를 수행하였습니다. 특히, 지식 증류(knowledge distillation) 기법을 사용하여 작은 모델에서도 LRL 번역 품질을 획기적으로 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: 현재의 LLM들은 LRL을 처리하는 데 있어서 여러 한계를 보입니다. 연구는 FLORES-200 데이터셋을 사용하여 200개의 언어에서 번역 성능을 정량적으로 평가하였고, 뉴스 기사와 이중언어 사전 같은 대체 데이터를 도입하여 LRM 모델의 세부 조정을 효과적으로 진행하였습니다. 이외에도, 적은 양의 데이터(총 데이터의 1%)로도 성능 격차를 줄일 수 있는 다양한 최신 세부 조정 방법론을 explored했습니다.

- **Performance Highlights**: 연구 결과, LRL 번역에서 큰 모델이 더 나은 성능을 발휘하는 경향이 있음을 발견했습니다. 또한, 경제적이고 인구가 많은 지역에서 사용되는 언어들이 번역 품질이 더 높다는 것을 발견했습니다. 하지만, LRL에 대한 집중적인 지원이 부족함에 따라 번역 품질 격차는 여전히 존재하며, 이는 고급 언어 자원과의 불균형에서 비롯된 문제입니다.



### Artificial Conversations, Real Results: Fostering Language Detection with Synthetic Data (https://arxiv.org/abs/2503.24062)
- **What's New**: 이 연구는 고품질 훈련 데이터를 수집하는 대신 LLMs를 사용하여 생성된 합성 데이터의 가능성을 탐구합니다. 특히, 이탈리아어 직업 광고에서 포괄적 언어 감지를 위한 작업에 초점을 맞추어, 데이터 부족 문제를 해결하기 위해 합성 데이터 생성을 위한 파이프라인을 제안하고 효과적인 프롬프트 전략과 텍스트 길이, 특수 과제에서의 목표 위치 같은 요소가 모델 성능에 미치는 영향을 분석합니다.

- **Technical Details**: 연구 방법론은 LLM 기반 합성 데이터 생성을 위한 프레임워크를 수립하고, 이탈리아어 직업 광고에서 비포괄적 언어를 탐지하는 데 활용됩니다. 이 방식은 실제 및 생성된 데이터를 결합한 합성 데이터셋 생성, 다양한 프롬프트 기법의 적용, 합성 데이터에 대한 모델의 파인 튜닝을 포함합니다. 데이터셋은 70-30 비율로 훈련 및 평가용으로 분할되어 모델의 일반화 능력을 검증합니다.

- **Performance Highlights**: 실험 결과, 합성 데이터로 훈련된 세밀화 모델이 실제 및 합성 테스트 데이터 세트에서 다른 모델보다 일관되게 우수한 성능을 보였습니다. 이는 LLMs의 합성 데이터 활용이 비용 효율적이고 확장 가능한 솔루션임을 보여줍니다. 본 연구는 이러한 합성 데이터 사용의 실질적 임팩트와 한계를 논의하면서, 포괄적 언어 탐지 작업에 대한 새로운 패러다임을 제시합니다.



### Crossing Boundaries: Leveraging Semantic Divergences to Explore Cultural Novelty in Cooking Recipes (https://arxiv.org/abs/2503.24027)
- **What's New**: 이번 연구는 문화적 새로움(cultural novelty)의 개념을 기반으로 한 새로운 방법론을 제안합니다. 이는 자연어 처리(NLP)와 인공지능(AI) 분야에서의 새로움 탐지 문제를 다룹니다. 연구진은 글로벌 퓨전(GlobalFusion)이라는 새로운 데이터셋을 활용하여 다양한 문화적 배경에서 요리 레시피의 텍스트 변화를 분석하였습니다. 데이터셋은 500개 요리와 약 10만 개의 요리 레시피를 포함하고 있으며, 150개국 이상의 문화적 적응을 담고 있습니다.

- **Technical Details**: 이 연구에서는 정보를 측정하는 데 사용되는 다섯 가지 정보 이론적 문화적 새로움 지표(cultural novelty metrics)를 제안했습니다. 이 지표는 텍스트의 분포 차이를 기반으로 하며, Jensen-Shannon Divergence 메트릭스를 포함합니다. 이를 통해 서로 다른 문화적 배경을 가진 커뮤니티에서 요리가 어떻게 수정되는지를 분석하고, 사회 과학 문헌에서 문화적 거리(cultural distance)와의 상관관계를 탐구합니다. 이 과정에서 인글하르트-웰젤 문화 맵(Inglehart–Welzel cultural map)과 같은 기존의 문화적 거리 측정을 활용하였습니다.

- **Performance Highlights**: 제안된 문화적 새로움 지표가 기존 문화적 거리 측정과 상당한 상관관계를 보임을 확인했습니다. 또한 연구를 통해 문화적 새로움이 어떻게 서로 다른 문화 간의 정보의 유사성 또는 차별성을 강조하는지를 보여주었습니다. 이 프레임워크는 다양한 태스크와 문화적 매개변수에 쉽게 적용 가능하여, 대규모 언어 모델(LLMs)의 문화적 표현 능력을 이해하고 향상시키는 데 기여할 수 있는 잠재력을 갖고 있습니다.



### You Cannot Feed Two Birds with One Score: the Accuracy-Naturalness Tradeoff in Translation (https://arxiv.org/abs/2503.24013)
- **What's New**: 이번 논문에서는 기계 번역(Machine Translation)의 품질 평가에 대한 새로운 접근 방식을 제시합니다. 저자들은 특정 기준(metric)을 최적화하는 것만으로 번역 시스템의 정확성과 자연스러움을 동시에 향상시키는 것이 불가능하다고 수학적으로 증명했습니다. 이를 통해 번역 성능을 정량적으로 평가하는 데 사용되는 전통적인 단일 점수를 넘어서, 정확성과 자연스러움을 동시에 고려해야 함을 강조합니다.

- **Technical Details**: 문헌에 따르면, 번역의 정확성은 원문 의미를 얼마나 잘 전달하느냐에 달려 있으며, 기계 번역 시스템을 평가하기 위한 다양한 참조 기준(reference metrics)들이 존재합니다. 최근에는 BLEU나 chrF와 같은 전통적인 기준에서 벗어나, MetricX 및 Comet과 같은 신경망 기반 메트릭(neural metrics)이 도입되었습니다. 그러나 이러한 메트릭이 인간 평정자와의 상관관계를 갖지 못하는 경우가 많아, 번역 평가의 필요성에 대한 재검토가 이루어졌습니다.

- **Performance Highlights**: 연구 결과, 정확성과 자연스러움 간에 트레이드오프가 존재한다는 것이 밝혀졌습니다. 예를 들어, 특정 정확성 기준(BLEU)을 최적화하는 경향이 오히려 번역의 자연스러움에 악영향을 미치는 경우가 발생합니다. 이는 번역 최적화가 정확성과 자연스러움의 균형을 탈선시킬 수 있음을 보여주며, 평가 방법론의 변화가 필요함을 지적합니다.



### Comparing representations of long clinical texts for the task of patient note-identification (https://arxiv.org/abs/2503.24006)
- **What's New**: 이 논문은 환자-노트 식별(patient-note identification) 과제를 다루며, 익명화된 임상 노트를 해당 환자와 정확하게 연결하는 방법을 제시합니다. 이를 통해 중복 기록 탐지 및 환자 유사성 분석과 같은 다양한 응용 프로그램에서 효과적인 환자 수준의 표현(patience-level representations)을 구축하는 것을 목표로 합니다. 또한, 다양한 임베딩 메소드와 단어 수준 임베딩을 환자 수준으로 집계하는 방법론(pooling strategies)을 평가합니다.

- **Technical Details**: 다양한 임베딩 기법으로는 계층적 주의 네트워크(Hierarchical Attention Networks, HAN), 세 가지 수준의 계층적 트랜스포머 네트워크(Hierarchical Transformer Networks, HTN), LongFormer 및 BERT 기반 모델들이 포함됩니다. 특히, BERT 기반 임베딩이 긴 임상 노트를 처리하는 데 있어 전통적인 모델들보다 우수한 성능을 발휘합니다. 평균, 최대 및 평균-최대(mean_max) 풀링 전략을 비교한 결과, 평균-최대 풀링이 임상 노트의 중요한 특징을 포착하는 데 가장 효과적임을 밝혔습니다.

- **Performance Highlights**: MIMIC 데이터셋 및 Necker 병원 데이터 웨어하우스에서 재현된 결과는 이 연구의 방법이 실제 응용에서도 일반화 가능함을 보여줍니다. BERT 기반 모델은 슬라이딩 윈도우(sliding window) 메커니즘과 평균 및 최대 풀링(combination of mean and max pooling)을 통해 가장 높은 정확도를 보였습니다. 본 연구는 환자-노트 식별 과제에 대한 명확한 정의와 함께 효과적인 환자 표현 방법에 대한 실증 비교를 제공합니다.



### BeMERC: Behavior-Aware MLLM-based Framework for Multimodal Emotion Recognition in Conversation (https://arxiv.org/abs/2503.23990)
- **What's New**: 이번 논문에서는 대화 중 감정 인식(multimodal emotion recognition in conversation, MERC)을 위한 새로운 행동 인식 기반 다중 모드 대형 언어 모델(MLLM) 프레임워크인 BeMERC를 제안합니다. 이는 기존의 대화 분석이 텍스트와 음성 특성에만 초점을 맞춘 것에서 벗어나 비디오에서 발생하는 행동 정보를 활용하여 감정 예측을 보다 정확하게 수행하는 것을 목표로 합니다. BeMERC는 미세한 얼굴 표정, 몸짓, 자세 등을 포착하여 감정 다이내믹스를 모델링합니다. 또한, 두 단계의 지시 튜닝(instruction tuning) 전략을 채택하여 대화 시나리오에 맞게 모델을 확장합니다.

- **Technical Details**: BeMERC는 비디오에서 유도된 미세한 행동 정보를 통해 감정 동태를 캡처하는 새로운 방법론으로, Qwen2-VL을 사용하여 비디오에서 발생하는 행동에 대한 자연 언어 설명을 생성합니다. 이 과정은 큰 언어 모델이 비디오-derived behaviors 와 대상 발화 감정(label)을 명시적으로 연결할 수 있도록 합니다. BeMERC의 교육 과정에서 특히, 작업별 튜닝을 위해 두 단계의 지시 튜닝이 사용되며 이는 감정의 세부적 변동을 동시에 학습하는 데 중점을 둡니다.

- **Performance Highlights**: 실험 결과, BeMERC는 널리 사용되는 두 개의 벤치마크 데이터세트인 IEMOCAP과 MELD에서 기존의 최첨단 방법(state-of-the-art, SOTA)과 비교하여 우수한 성능을 달성했습니다. 비디오에서 유도된 행동 정보의 중요성을 강조하며, BeMERC는 MERC 작업에서의 성능을 획기적으로 향상시키는 데 기여하게 됩니다. 이 연구는 향후 MERC 연구에 대한 견고한 기반을 마련할 것으로 기대됩니다.



### Model Hemorrhage and the Robustness Limits of Large Language Models (https://arxiv.org/abs/2503.23924)
Comments:
          33 pages, 18 figures

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 성능 저하 현상을 '모델 출혈(model hemorrhage)'로 정의하며, 이는 매개변수 조정 및 아키텍처 변화에 의한 성능 감소를 포함합니다. 특히 다양한 LLM 프레임워크를 분석하여 주의(attention) 메커니즘의 방해, 압축 기법에서의 정보 손실, 디코딩 조정 시 예측 편차 확대와 같은 주요 취약성 패턴을 식별했습니다. 이러한 통찰력을 바탕으로 성능 저하를 방지하기 위한 세 가지 완화 전략을 제안하며, 이는 모델 안정성을 평가하기 위한 기초적인 메트릭을 구축합니다.

- **Technical Details**: 모델 출혈의 주요 원인은 네트워크의 아키텍처 및 파라미터 수정으로 나타납니다. 본 연구에서는 그래디언트 인식 가지치기(gradient-aware pruning), 동적 양자화 스케일링(dynamic quantization scaling), 디코딩 보정(decoding calibration) 등의 방법을 통해 성능 유지 방안을 제시합니다. 이 과정에서 Transformer 아키텍처가 수정 방법에 따라 성능 저하의 강도를 결정짓는 내재적 강건성 임계값을 가진다는 점도 밝혀졌습니다.

- **Performance Highlights**: LLM의 크기가 증가하면서, 이전의 단일 모델 아키텍처는 성능 및 효율성 면에서 이중 도전에 직면합니다. 최근에 출시된 MoE 모델 DeepSeek-R1은 학습 및 추론 과정에서 병렬화와 비용 최적화의 새 경로를 제시하며, 이는 다양한 작업 간 효율적인 전환을 가능하게 합니다. 이러한 발전은 대규모 언어 모델의 실용적 활용에 기여하며, 목표 지향적인 미래 연구를 위한 새로운 아이디어를 제공합니다.



### Entropy-Based Adaptive Weighting for Self-Training (https://arxiv.org/abs/2503.23913)
- **What's New**: 대형 언어 모델(LLM)의 수학적 문제 해결 능력이 주요 연구 대상이 되고 있으며, 자가 생성된 추론 경로(self-generated reasoning paths)를 활용하여 이러한 모델을 개선할 수 있는 가능성이 증가하고 있습니다. 본 논문에서는 Entropy-Based Adaptive Weighting for Self-Training (EAST)을 제안하며, 이는 불확실한 데이터를 우선시하는 적응형 가중치 전략을 통해 모델 훈련의 성능을 향상시키는 방법입니다. EAST는 불확실성을 측정하기 위해 엔트로피(entropy)를 이용하여 정보를 보다 효과적으로 활용하는 방향으로 모델을 훈련시킵니다.

- **Technical Details**: EAST는 모델이 생성한 여러 샘플의 최종 답변에 따라 이를 클러스터링하고, 클러스터 기반 분포에 대한 엔트로피를 계산합니다. 그 후, 이 값을 특정 제약조건 하에 가중치로 변환하는 매핑 함수(mapping function)를 적용합니다. 이 함수는 가중치의 날카로움을 조절하는 파라미터를 포함하여, 불확실한 데이터에 대해 더 높은 가중치를 부여함으로써 모델이 더 정보량이 많고 도전적인 예제에 집중할 수 있도록 유도합니다.

- **Performance Highlights**: EAST는 GSM8K와 MATH 벤치마크에서 평가되어, 기본 모델에 비해 각각 5.6%와 약 1%의 성능 향상을 보여주었습니다. 또한, EAST를 적용한 방식이 기존의 일반적인 방법보다 훨씬 더 우수한 성과를 보였으며, 불확실한 데이터를 더 잘 활용하여 모델의 추론 능력을 강화하는 데 기여하는 것으로 나타났습니다. 전반적으로 EAST는 다양한 훈련 설정에 유연하게 통합될 수 있는 프레임워크이며, 자가 훈련(self-training)의 성능을 효과적으로 향상시키는 데 중요한 역할을 합니다.



### Rubrik's Cube: Testing a New Rubric for Evaluating Explanations on the CUBE datas (https://arxiv.org/abs/2503.23899)
Comments:
          9 main pages (21 appendix pages), 7 figures, submitted to ACL 2025

- **What's New**: 이번 논문에서는 Rubrik의 CUBE라는 새로운 평가 기준과 26,000개의 설명 데이터셋을 제시합니다. 이 기준은 인간과 여러 개방형 및 폐쇄형 Large-Language Models (LLMs)에 의해 작성된 설명을 품질 기준으로 평가하기 위해 설계되었습니다. CUBE는 추론과 언어 작업에 초점을 맞추어 다양한 평가를 가능하게 하며, LLM이 생성한 설명의 품질 저하 원인을 규명하고 이에 대한 통찰을 제공합니다.

- **Technical Details**: Rubrik의 CUBE는 설명의 품질을 평가하기 위해 세 가지 주요 요소를 포함합니다. 첫째, LLM에서 생성된 설명의 핵심 요소와 특징을 구분합니다. 둘째, 설명의 품질을 측정하기 위한 두 개의 사용자 정의 동의(agreement) 메트릭스를 포함하여 설명의 유형 및 수행된 작업과 인지된 난이도에 따라 변화를 관찰합니다.

- **Performance Highlights**: 연구 결과, LLM 생성 설명의 저품질은 주로 간결함의 부족에서 기인하고 있음이 밝혀졌습니다. Rubrik을 통해 다양한 설명에 대한 통찰이 제공되며, 설명 유형은 작업과 인지된 난이도에 따라 달라짐을 확인하였습니다. 이는 설명의 투명성을 높이고 사용자의 이해를 돕는 데 기여할 수 있습니다.



### Better wit than wealth: Dynamic Parametric Retrieval Augmented Generation for Test-time Knowledge Enhancemen (https://arxiv.org/abs/2503.23895)
Comments:
          preprint

- **What's New**: 본 논문에서는 Dynamic Parametric RAG (DyPRAG)이라는 새로운 프레임워크를 제안하여 외부 문서를 효율적으로 파라미터로 변환함으로써 LLMs의 지식을 동적으로 향상시킵니다. DyPRAG는 파라미터 전환 모델을 활용하여 기존의 방법에 비해 훈련 및 저장 비용을 대폭 줄입니다. 또한, 이 프레임워크는 RAG 환각 문제를 완화하고, 모델의 지식 충돌을 해결하는 방식으로 실시간 성능 향상을 가능하게 합니다.

- **Technical Details**: DyPRAG는 최신 파라미터 변환 모델을 기반으로 하여 외부 문서를 LLM의 파라미터로 직접 변환하는 방식을 채택합니다. 이 방식은 테스트 시간에 필요한 지식 형성을 즉각적으로 수행하며, 이를 통해 전통적인 RAG 방식에 비해 더 낮은 추론 비용과 훈련 비용을 자랑합니다. 이는 효율적인 지식 주입을 통해 LLM의 성능을 극대화하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, DyPRAG는 다양한 데이터셋에서 기존 RAG 방법들을 능가하는 성능을 보였습니다. 특히 테스트 시간에 생성된 파라미터 지식과 맥락 지식을 결합함으로써 지식 융합에서 우수한 성과를 발휘하며, RAG 환각 문제를 효과적으로 완화하는 데 성공했습니다. DyPRAG-Combine 방법을 통해 높은 일반화 능력을 보여주며 실제 응용에서 신뢰할 수 있는 RAG 시스템 구축 가능성을 확인했습니다.



### SpeechDialogueFactory: Generating High-Quality Speech Dialogue Data to Accelerate Your Speech-LLM Developmen (https://arxiv.org/abs/2503.23848)
- **What's New**: 본 논문에서는 Speech-LLM 개발을 위해 음성 대화 데이터셋의 고품질 생성에 필요한 새로운 프레임워크인 SpeechDialogueFactory를 소개합니다. 기존의 수집 방법들은 한계가 있으며, 특히 인체 기록 방식은 비용과 개인 정보 보호 문제를 유발합니다. SpeechDialogueFactory는 자연적인 대화를 효율적으로 생성할 수 있는 종합 파이프라인을 구축하여 기존 문제를 해결할 수 있습니다.

- **Technical Details**: SpeechDialogueFactory는 메타데이터 생성, 대화 스크립트 작성, 표현이 풍부한 발화 시뮬레이션, 목소리 클로닝을 이용한 자연 음성 합성을 포함하는 통합된 파이프라인을 구현합니다. 이 시스템은 사용자가 샘플을 상세히 검사할 수 있는 인터랙티브 UI와 고속 일괄 합성 모드를 제공합니다. 사전 정의된 메타데이터는 특정 대화 종류, 시간적 맥락, 문화적 배경 등을 포함하여 강화된 대화 생성 옵션을 제공합니다.

- **Performance Highlights**: 실험 결과, SpeechDialogueFactory는 생성된 대화가 전문가 기록과 유사한 품질을 달성하는 동시에, 생산 비용을 상당히 절감할 수 있음을 확인하였습니다. 이 시스템은 연구자와 개발자들이 특정 요구 사항에 맞춘 사용자 정의 음성 대화 데이터셋을 생성하는 unprecedented flexibility(전례 없는 유연성)를 제공합니다. 또한, 영어와 중국어로 공개될 대화 데이터셋을 포함하여 연속적인 연구 및 개발에 기여할 예정입니다.



### Expanding RL with Verifiable Rewards Across Diverse Domains (https://arxiv.org/abs/2503.23829)
- **What's New**: 이번 논문에서는 검증 가능한 보상(Reinforcement learning with Verifiable Rewards, RLVR)이 수학적 추론 및 코딩 과제와 같은 잘 구조화된 작업을 넘어 다양한 영역으로 확장되는 가능성을 탐구합니다. 특히, 의학, 화학, 심리학, 경제학 등의 분야에서 다수의 대형 언어 모델(LLMs)의 이진 판단의 높은 일치를 관찰했습니다. 이를 통해 특정 도메인 별 보상 모델 교육을 위한 대규모 주석 작업의 필요성이 질문받고 있습니다.

- **Technical Details**: 이 연구는 RLVR를 확장하여 비구조적 참조 답변이 있는 과제를 처리하며, 기존의 이진 보상 점수의 한계를 극복하기 위해 모델 기반의 소프트 스코어를 RLVR에 통합하였습니다. 연구에서는 7B 모델과 같은 상대적으로 작은 LLM을 사용하여도 다양한 도메인에서의 성능을 크게 향상시킬 수 있음을 입증했습니다. 또한, 다양한 RL 알고리즘을 통해 정책을 미세 조정하여 Qwen2.5-72B-Instruct와 같은 기존의 강력한 모델들을 초월하는 결과를 얻었습니다.

- **Performance Highlights**: 이 연구의 실험 결과는 초강력 오픈소스 LLM들을 상회하는 성능을 보여주며, 특히 정답이 비구조적일 때 모델 기반 보상이 효과적임을 입증하였습니다. 데이터 크기가 증가함에 따라 모델 기반 보상은 뛰어난 확장성을 보여주며, 기존의 규칙 기반 보상은 성과 저하를 일으키는 반면, 제안된 방법은 일반화 능력을 향상시켰습니다. 이는 RLVR의 강건성과 확장 가능성을 강조하며, 실제 환경에서의 적용 가능성을 제시하고 있습니다.



### Did ChatGPT or Copilot use alter the style of internet news headlines? A time series regression analysis (https://arxiv.org/abs/2503.23811)
- **What's New**: 이번 연구는 ChatGPT와 Copilot와 같은 최신 대형 언어 모델(Large Language Models, LLMs)의 출시가 전 세계 뉴스 웹사이트의 헤드라인과 링크 문체에 미친 영향을 조사했습니다. 451억 개의 헤드라인/링크 데이터셋에서 175개의 NLP(자연어 처리) 특징이 수집되었고, 이 특징들이 모델 출시 이후 통계적으로 유의미한 변화를 보였는지를 검토했습니다.

- **Technical Details**: 연구에서는 중단 시간 시계열 분석(interrupted time series analysis)을 사용해 출시 이전과 이후의 변화 여부를 평가했습니다. 분석 결과, ChatGPT와 Copilot의 출시 이후 44개의 특징에서는 유의미한 변화가 없었고, 91개의 다른 특징에서 유의미한 변화가 관찰되었으나, 이전 LLM 출시(예: GPT-1/2/3, Gopher)와의 비교에서 해당 변화를 제외했습니다.

- **Performance Highlights**: 이 초기 분석은 대형 언어 모델들이 뉴스 헤드라인/링크 문체에 미치는 영향이 제한적일 수 있으며, 특정 NLP 측정 기준에서만 변화가 있었음을 시사합니다. 즉, 최신 LLM의 출시가 뉴스 콘텐츠에 미친 전반적인 영향은 제한적이라는 결론을 도출했습니다.



### Adaptive Layer-skipping in Pre-trained LLMs (https://arxiv.org/abs/2503.23798)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)에서의 토큰 생성 시 계산 요구 사항이 어떻게 달라지는지를 탐구하며, FlexiDepth라는 새로운 방법을 제안합니다. FlexiDepth는 각 토큰에 대해 Transformer 레이어의 수를 동적으로 조절하여 계산 효율성을 높입니다. 기존의 방법들과 달리, FlexiDepth는 사전 훈련된 모델의 파라미터를 변경하지 않고 레이어 스킵을 구현할 수 있도록 설계되어 있습니다.

- **Technical Details**: FlexiDepth는 각 Transformer 레이어에 라우터(router)와 어댑터(adapter)를 추가하여 동적 레이어 스킵을 가능하게 합니다. 라우터는 입력된 상태를 바탕으로 레이어를 통과할지 스킵할지를 결정하며, 어댑터는 스킵된 상태의 표현을 조정하여 일관성을 유지합니다. 이 방식은 자동 회귀 생성(auto-regressive generation)과의 호환성을 보장하기 위해 모든 상태에 대해 KV 캐시를 계산합니다.

- **Performance Highlights**: 실험 결과에 따르면, FlexiDepth는 Llama-3-8B 모델에서 32개 레이어 중 8개를 스킵하면서도 100%의 성능을 유지하였고, 특히 연속 생성 작업에서 기존 레이어 스킵 방법들보다 뛰어난 성과를 보였습니다. FlexiDepth는 또한 다양한 토큰 유형 생성 시 계산 요구 사항이 어떻게 달라지는지를 보여주는 레이어 할당 패턴을 드러내며, 향후 연구를 위한 데이터셋을 공개했습니다.



### WinoWhat: A Parallel Corpus of Paraphrased WinoGrande Sentences with Common Sense Categorization (https://arxiv.org/abs/2503.23779)
- **What's New**: 이번 연구에서는 Winograd 스키마(Winograd Schema) 챌린지를 통해 대형 언어 모델(LLM)의 상식 추론(common sense reasoning) 능력을 평가하는 방법을 조명합니다. 새로운 데이터셋인 WinoWhat을 발표하여 WinoGrande의 검증 집합의 각 인스턴스를 패러프레이즈(paraphrase)했습니다. 이를 통해 LLM의 WinoGrande 성능이 패러프레이징에 강한지 여부를 테스트합니다.

- **Technical Details**: 연구는 Gemma 2, LlaMA 2, OPT 등의 오픈 소스 모델들을 WinoGrande에서 평가합니다. 또한, 데이터 누수(data leakage)에 대한 검증을 위해 LLM의 사전학습(pre-training) 데이터와 검증 집합 인스턴스의 매칭을 통해 두 개의 테스트 세트를 생성합니다. 모델 성능 평가를 위해 상식 지식 범주를 정의하여 각 범주별로 모델을 평가하며, 이는 상식 추론 과제에서 모델의 강약점을 이해하는 데 기여합니다.

- **Performance Highlights**: 모든 모델은 WinoWhat에서 상대적으로 낮은 성과를 보였으며, 이는 LLM의 WinoGrande에서의 추론 능력이 과대평가되었음을 암시합니다. 데이터 기억(memorization)이 모델 성능에 미치는 영향은 미미하다는 것을 확인하였으며, 기존의 WinoGrande 벤치마크 결과는 상식 획득을 나타내지 않음을 시사합니다. 이러한 결과는 상식 지식 평가를 위한 새로운 접근 방식을 요구합니다.



### CONGRAD:Conflicting Gradient Filtering for Multilingual Preference Alignmen (https://arxiv.org/abs/2503.23777)
- **What's New**: 이번 연구에서는 다국어 (multilingual) 선호 정렬 (preference alignment)에서 발생하는 부정적 간섭 (negative interference) 문제를 다룬 혁신적인 방법인 CONGRAD를 제안합니다. 다국어 모델 (language models) 훈련에서 상충하는 목표들이 전체 성능 저하를 초래하는 사례가 많지만, 다국어 선호 정렬 맥락에서는 이 문제의 연구가 부족했습니다. CONGRAD는 언어 간 최소한의 그래디언트 갈등 (gradient conflicts)을 유지하면서 고품질 선호 샘플을 선택할 수 있는 효과적이고 확장성 있는 필터링 방법입니다.

- **Technical Details**: CONGRAD 방법은 그래디언트 수술 (gradient surgery)을 활용하여 집계된 다국어 업데이트 방향과 정렬되는 샘플을 유지합니다. 또한, 그래디언트 축적 (gradient accumulation) 시 메모리 오버헤드를 줄이기 위한 서브리니어 (sublinear) 그래디언트 압축 전략을 통합합니다. 이를 통해 기존의 다국어 훈련 방식의 문제를 효과적으로 해결할 수 있습니다.

- **Performance Highlights**: LLaMA3-8B 및 Gemma2-2B 모델을 10개 언어에서 평가한 결과, CONGRAD는 보이는 언어 (seen languages)와 보이지 않는 언어 (unseen languages) 모두에서 강력한 기준선 (baselines)을 일관되게 초월하는 성능을 보였습니다. 게다가 최소한의 정렬 세금 (alignment tax)으로 성능 향상을 이루어냈습니다.



### Texture or Semantics? Vision-Language Models Get Lost in Font Recognition (https://arxiv.org/abs/2503.23768)
- **What's New**: 본 논문은 현대 비전-언어 모델(Visual-Language Models, VLMs)의 폰트 인식 능력을 평가하기 위해 Font Recognition Benchmark (FRB)을 제안합니다. 이 데이터셋은 15종의 일반적으로 사용되는 폰트로 구성되며, 쉬운 버전과 어려운 버전으로 나뉘어 폰트 인식의 어려움을 측정합니다. 현재 VLM들이 폰트 인식 작업에서 기대 이하의 성과를 보이고 있음을 밝혔습니다.

- **Technical Details**: Font Recognition Benchmark (FRB)는 두 가지 버전으로 나뉘며, 쉬운 버전은 다양한 폰트로 렌더링된 문장으로 구성되고, 어려운 버전은 폰트 이름을 잘못된 유형으로 표시하여 Stroop 효과를 도입합니다. 이 데이터셋은 Serif, Sans-Serif, Script & Decorative이라는 세 가지 주요 카테고리의 15개 폰트를 포함합니다. 이미지 생성에는 일관된 Python 스크립트를 사용하여 폰트 크기와 배경을 균일하게 유지합니다.

- **Performance Highlights**: 본 연구에서는 13개의 VLM 모델을 평가한 결과, 가장 높은 성과를 보인 모델조차 쉬운 버전에서 약 30%의 정확도만 달성했으며, 어려운 버전에서는 15%로 떨어졌습니다. Chain-of-Thought (CoT) 프롬프트는 성과 개선에 미미한 효과를 보였고, 적은 예시(Few-shot learning)조차도 VLM의 성능 향상에는 한계를 드러냈습니다. 이러한 결과는 VLM들이 폰트 인식에서 적절한 성능을 발휘하지 못하고 있음을 나타냅니다.



### LANID: LLM-assisted New Intent Discovery (https://arxiv.org/abs/2503.23740)
Comments:
          Published in LREC-COLING 2024

- **What's New**: 본 논문에서는 새로운 의도 인식(New Intent Discovery, NID)을 향상시키기 위해, 경량 NID 인코더를 대형 언어 모델(Large Language Models, LLM)의 세멘틱 리프레젠테이션을 통해 개선하는 LANID 프레임워크를 제안합니다. 기존 NID 방법의 한계를 극복하고, 효율적으로 작동하는 경량 모델링을 가능하게 합니다. 이 프레임워크는 LLM을 활용하여 유의미한 발화 쌍을 샘플링하고, 이를 기반으로 대비 학습(contrastive learning)을 적용하여 높은 성능을 보여줍니다.

- **Technical Details**: LANID는 K-최인접 알고리즘(K-nearest neighbors) 및 DBSCAN(Density-Based Spatial Clustering of Applications with Noise) 알고리즘을 사용하여 선택적 발화 쌍을 샘플링합니다. 이후 LLM에 이들 쌍 간의 관계를 질의하여 정보를 얻고, 이를 사용하여 대비 학습의 과제를 구성합니다. Small encoder를 대비 삼중 손실(contrastive triplet loss)로 훈련하여 효율성을 극대화하며, 데이터 샘플링과 교육 단계를 반복적으로 수행합니다.

- **Performance Highlights**: 실험 결과, LANID는 세 가지 NID 데이터셋에서 기존 강력한 기준선들을 초과하는 성과를 보여주었습니다. 언수퍼바이즈드(unsupervised) 및 세미-슈퍼바이즈드(semi-supervised) 설정 모두에서 뛰어난 효율성과 성능을 입증하였습니다. 최종적으로 경량화된 인코더도 기존과 같은 성능을 유지하며, 업계의 실질적인 응용에 적합한 솔루션을 제공합니다.



### AdaMMS: Model Merging for Heterogeneous Multimodal Large Language Models with Unsupervised Coefficient Optimization (https://arxiv.org/abs/2503.23733)
Comments:
          CVPR 2025

- **What's New**: 최근 모델 병합(model merging) 기법이 다수의 대형 언어 모델(LLMs)을 결합하여 효과적으로 다양한 작업을 수행할 수 있음이 입증되었습니다. 그러나 기존의 모델 병합 방법들은 주로 동일한 아키텍처의 이종 모델에 초점을 맞추어 왔습니다. 이 논문에서는 이질적인 다중 모달 대형 언어 모델(MLLMs)에 특화된 새로운 모델 병합 방법인 AdaMMS(AdaMMS: Adaptive Model Merging Strategy)를 제안합니다.

- **Technical Details**: AdaMMS는 세 가지 단계인 매핑(mapping), 병합(merging), 검색(searching)으로 구성됩니다. 첫 번째 단계인 매핑에서는 서로 다른 아키텍처의 모델들 간의 매핑 함수를 설계해 병합 작업을 수행할 수 있도록 합니다. 두 번째 단계에서는 모델 가중치에 선형 보간(linear interpolation)을 적용하여 이질적인 MLLMs의 비대칭성을 조정하며, 세 번째 단계에서는 비지도 학습 방식으로 하이퍼파라미터를 선택하는 방법을 제안합니다.

- **Performance Highlights**: 많은 실험을 통해 AdaMMS가 다양한 시각-언어(vision-language) 벤치마크에서 이전의 모델 병합 방법보다 우수한 성능을 보임을 입증했습니다. 특히, Qwen 및 LLaMA 아키텍처를 기반으로 한 이질적인 MLLM 쌍에서 강력한 성능을 보여주며, 이전의 방법들과 비교해 적은 양의 데이터로도 고성능을 유지할 수 있는 안정성을 확인했습니다.



### Building Instruction-Tuning Datasets from Human-Written Instructions with Open-Weight Large Language Models (https://arxiv.org/abs/2503.23714)
Comments:
          15 pages, 5 figures

- **What's New**: 이번 연구는 복잡한 기계 학습 과제를 해결하기 위해 LLM(대형 언어 모델)의 instruction tuning의 중요성을 보이며, 인간 작성의 지침과 LLM이 생성한 응답을 조화롭게 결합하여 최신 데이터를 구축했습니다. 이는 특히 인간의 기여가 여전히 중요하다는 것을 강조하며, 모델의 성능 향상에 기여합니다.

- **Technical Details**: 연구진은 LMSYS-Chat-1M이라는 실제 인간-봇 대화 데이터세트를 기반으로, 약 732,392개의 교육 지침을 수집했습니다. 이 지침에는 독성 대화가 제외되었으며, 이를 통해 최종적으로 453,889개의 유일한 지침을 얻었습니다. 각각의 지침은 Llama-3.1과 Gemma-2 등 오픈 가중치 LLM을 통해 대응하는 응답을 생성하는 방식으로 데이터 세트를 구성했습니다.

- **Performance Highlights**: 인간 작성의 지침을 기반으로 하는 데이터 세트에서 훈련된 LLM 모델은 기존의 데이터 세트인 Magpie에서 훈련된 모델들에 비해 일관되게 더 높은 성능을 보였습니다. 일본어에 대해서도 본 연구의 방법론을 적용하여 최첨단 성능을 도출했으며, 이는 다른 언어로의 확장 가능성을 확인해주며, 문화적 지식 부족을 나타내는 한계도 보여주었습니다.



### Mapping Geopolitical Bias in 11 Large Language Models: A Bilingual, Dual-Framing Analysis of U.S.-China Tensions (https://arxiv.org/abs/2503.23688)
Comments:
          Preliminary version,20 pages, 10 figures, 1 table

- **What's New**: 이 연구는 미국-중국 관계에서의 7개의 주요 주제에 대한 11개의 주요 대형 언어 모델(LLMs)의 지정학적 편향을 체계적으로 분석했습니다. 이중 언어(English, Chinese) 및 이중 프레이밍(affirmative, reverse) 방법론을 사용하여 이념 성향을 탐지하기 위한 19,712개의 프롬프트를 생성했습니다.

- **Technical Details**: 모델의 응답은 -2 (강력한 친중파)에서 +2 (강력한 친미파)까지의 정규화된 척도로 정량적으로 평가되었으며, 입장, 중립성 및 거부율에 따라 분류되었습니다. 연구는 모델의 지리적 출처에 따라 상당하고 일관된 이념적 정렬을 발견하였으며, 미국 기반 모델들은 주로 친미파 입장을 취하고, 중국 유래 모델들은 뚜렷한 친중파 편향을 보였습니다.

- **Performance Highlights**: 언어와 프롬프트의 프레이밍(Framing)은 모델 응답에 큰 영향을 미쳤고, 여러 LLM들이 프롬프트의 극성(polarity)이나 언어적 맥락에 따라 입장 반전을 보였습니다. 또한, 응답의 일관성을 평가하기 위한 포괄적인 메트릭(Metric)을 도입하여 모델의 행동에서의 변동성과 취약성을 확인했습니다.



### MKA: Leveraging Cross-Lingual Consensus for Model Abstention (https://arxiv.org/abs/2503.23687)
Comments:
          To appear in Building Trust Workshop at ICLR 2025

- **What's New**: 본 연구에서는 LLMs(Large Language Models)의 신뢰성을 보장하기 위한 새로운 접근 방식을 제안합니다. 다국어(multilingual) 지식을 활용하여 모델의 응답에 대한 신뢰도를 조정하고, 불확실한 경우에는 응답을 삼가하도록 하는 파이프라인을 개발했습니다. 이 방법은 다양한 언어 모델에 적용되며, LLMs의 객관성을 높이기 위한 중요한 발판이 될 것으로 기대됩니다.

- **Technical Details**: MKA(Multilingual Knowledge Abstention) 파이프라인은 모델의 질문에 대한 응답을 여러 언어로 번역하여, 각각의 언어에서 생성된 응답을 비교하고 그 신뢰도를 조정하는 과정으로 이루어집니다. 심지어 저자원 언어(low-resource language)에서도 LLM의 성능을 향상시킬 수 있는 방법을 제시합니다. 이러한 방법은 신뢰성을 보장하기 위해 예를 들어 cosine similarity와 같은 기법을 활용하여 응답 간의 유사성을 측정합니다.

- **Performance Highlights**: 다국어 파이프라인을 통해 벵골어(Bengali)에서는 71.2%의 정확도 향상을 기록했으며, 영어에서도 15.5% 향상이 있음을 발견했습니다. 이는 특정 언어에서의 LLM 신뢰도 개선이 가능하다는 점을 시사합니다. 이러한 결과는 앞으로 더 많은 언어 및 모델에 대해 적용될 수 있는 잠재력을 보여줍니다.



### Large Language Models Pass the Turing Tes (https://arxiv.org/abs/2503.23674)
- **What's New**: 이번 연구는 현대의 대형 언어 모델(LLMs)인 GPT-4.5와 LLaMa-3.1-405B가 인간과 구별될 수 있는지를 평가하기 위해 세 패널의 Turing 테스트를 실시했습니다. GPT-4.5는 인간으로 오인될 확률이 73%에 달하며, LLaMa-3.1은 56%로 평가되었습니다. 이 연구는 인공지능 시스템이 Turing 테스트를 통과한 최초의 실증적 증거를 제공합니다.

- **Technical Details**: 연구에서는 네 가지 AI 시스템(ELIZA, GPT-4o, LLaMa-3.1-405B, GPT-4.5)을 사용하여 5분 동안 진행된 대화에서 참가자들이 AI와 실제 인간의 구분을 시도했습니다. 각 AI 모델은 기본 프롬프트와 특정 페르소나를 채택하라는 프롬프트 하에 테스트되었습니다. ELIZA는 규칙 기반 챗봇으로, 'ELIZA 효과'가 관찰되었습니다.

- **Performance Highlights**: GPT-4.5는 73%의 비율로 인간으로 오인되었으며, 이는 실제 인간 참가자보다 유의미하게 높은 수치입니다. 반면, ELIZA와 GPT-4o는 각각 23%와 21%로 기초 모델로서 챗봇을 인식하는 데 있어 성과가 크게 저조했습니다. 이러한 결과는 대형 언어 모델의 지능과 사회적, 경제적 영향을 둘러싼 논의에 중요한 함의를 가집니다.



### WHERE and WHICH: Iterative Debate for Biomedical Synthetic Data Augmentation (https://arxiv.org/abs/2503.23673)
- **What's New**: 이 논문에서는 생물의학 자연어 처리(BioNLP) 과제에서 데이터 희소성 문제를 해결하기 위해 BioRDA라는 새로운 합성 데이터 증강 방법을 제안합니다. 기존의 단순한 유사성 기반 방법과 달리, BioRDA는 생물학적 관계(bio-relation)에 강력한 상관관계를 지닌 합성 인스턴스를 생성하여 증강 데이터를 의미적으로 적절하게 만듭니다. 또한, 다중 에이전트 반영 메커니즘을 통해 유사한 엔티티의 사용을 세밀하게 식별하며 잘못된 대체를 방지합니다.

- **Technical Details**: BioRDA 방법은 데이터 증강 과정에서 ‘WHERE’와 ‘WHICH’라는 두 가지 자기 질문을 통해 분할된 프로세스를 사용합니다. 'WHERE'는 생물학적 관계와의 강한 상관관계를 가진 단어 교체 위치를 찾아내고, 'WHICH'는 유사한 의미를 가진 가장 합리적인 단어를 선택하는 과정을 포함합니다. 이 방식은 문장의 전체 의미를 유지하면서 다양한 어휘를 사용할 수 있도록 합니다.

- **Performance Highlights**: BioRDA는 BLURB 및 BigBIO 기준을 사용하여 수행된 9개의 데이터 세트에서 평균 2.98%의 성능 향상을 보여줍니다. 실험 결과는 BioRDA가 생물의학 작업에서 데이터 희소성 문제를 해결하고 전반적인 모델 성능을 향상시키는 데 효과적임을 강조합니다. 추가 실험에서는 WHERE와 WHICH 정보가 고품질 생물의학 인스턴스를 생성하는 데 필수적임을 입증하였습니다.



### CrossFormer: Cross-Segment Semantic Fusion for Document Segmentation (https://arxiv.org/abs/2503.23671)
Comments:
          10 pages, 4 figures

- **What's New**: 이번 논문에서는 CrossFormer라는 새로운 transformer 기반 모델을 제안했습니다. 이 모델은 문서의 여러 세그먼트 간의 잠재적인 의미적 의존성을 동적으로 모델링하는 혁신적인 cross-segment fusion module (CSFM)을 포함합니다. 이를 통해 기존의 세그먼트 방법에서는 잃어버리기 쉬운 중요한 의미적 정보를 복구하며, 문서 세분화의 정확도를 크게 향상시킵니다.

- **Technical Details**: CrossFormer는 문서 세그먼트 간의 semantic coherence를 강화하여 텍스트 세멘틱 세그멘테이션에서 뛰어난 성능을 발휘합니다. 여기서는 CSFM을 통해 각 문서 세그먼트 간의 상호작용을 모델링하고, 이로써 문서의 전반적인 의미 구조를 더 잘 이해할 수 있도록 돕습니다. RAG 시스템 내에서 텍스트 덩어리를 효과적으로 나누는 데 사용되며, 이전의 규칙 기반(chunking) 방법들을 대체합니다.

- **Performance Highlights**: CrossFormer는 공공 텍스트 세멘틱 세그멘테이션 데이터셋에서 최첨단 성능을 입증했습니다. 평가를 통해 CrossFormer가 기존의 세분화 방법보다 더 일관되며 의미 있는 덩어리를 생성함을 확인할 수 있었습니다. 이는 RAG 벤치마크에서도 제목별로 유의미한 향상을 보여줍니다.



### The Impact of Code-switched Synthetic Data Quality is Task Dependent: Insights from MT and ASR (https://arxiv.org/abs/2503.23576)
Comments:
          Accepted to the Workshop on Computational Approaches to Linguistic Code-Switching (CALCS)

- **What's New**: 본 논문에서는 코드 스위칭(code-switching, CSW) 데이터 증강(data augmentation) 기술에 대한 체계적인 연구를 진행하고, 기존 연구의 한계를 극복하고자 합니다. 특히, 자동 음성 인식(automatic speech recognition, ASR)과 비순차 음성 번역(cascaded speech translation, ST)에 대한 새로운 결과를 제시합니다. 이를 통해 다양한 증강 기술의 효과를 비교하고 NLP 작업에서의 개선 사항과 품질 간의 관계를 이해하고자 합니다.

- **Technical Details**: 우리는 아랍어-영어 병렬 문장을 활용하여 코드 스위칭 아랍어-영어 문장을 생성하기 위해 다양한 데이터 증강 기법을 적용했습니다. 이 과정에서, 랜덤 단어 대체(random lexical replacements), 언어학 기반 접근법(linguistic-based approaches), 및 역번역(back-translation) 기법을 포함했습니다. 또한, mBERT 모델을 사용하여 CSW 예측 모델을 훈련하고, 아랍어-영어 문장에서 코드 스위치된 단어를 식별하는 방법을 개발했습니다.

- **Performance Highlights**: 실험 결과, 특정 증강 기법들은 다양한 과제에서 일관된 성능을 보였으나, 일부는 특정 과제에 의존하는 경향이 있음을 발견했습니다. 데이터 품질과 NLP 개선 간의 관계는 기계 번역(machine translation)에서는 확인되었으나 ASR에서는 그렇지 않은 것으로 나타났습니다. 이러한 결과를 바탕으로, 품질 외에도 데이터 다양성과 작업 복잡성과 같은 다른 요인들이 결과에 영향을 미칠 수 있음을 논의합니다.



### When LLM Therapists Become Salespeople: Evaluating Large Language Models for Ethical Motivational Interviewing (https://arxiv.org/abs/2503.23566)
- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 동기 면담(motivational interviewing, MI) 윤리를 이해하는 능력을 평가하는 새로운 관점을 제시합니다. 현재까지 연구들은 LLM이 MI 관행의 품질이나 효율성을 평가하는 데 중점을 두었으나, 윤리적 인식에 대한 연구는 부족했습니다. 이러한 윤리적 이해 부족은 잠재적으로 악의적인 행위자들이 LLM을 비윤리적인 목적으로 악용할 위험을 증가시킵니다. 연구진은 이러한 위험을 줄이기 위해 Chain-of-Ethic 프롬프트를 제안했습니다.

- **Technical Details**: 연구는 LLM의 동기 면담에 대한 일반적인 지식과 윤리 인식 수준을 평가하기 위한 다양한 실험을 통해 진행되었습니다. 이 과정에서 LLM의 MI 지식 테스트를 수행하고, 비윤리적 MI 요청에 대한 응답을 수집하여 그 응답이 윤리적인지 비윤리적인지를 주석 처리했습니다. LLM의 윤리적 이해 수준은 양호하나, 실제 동기 면담 원칙과는 일치하지 않는 결과를 보였습니다. Chain-of-Ethic 프롬프트는 윤리적 응답 생성을 개선하는 데 효과적이었습니다.

- **Performance Highlights**: 연구 결과, LLM은 MI에 대한 중간에서 강한 이해도를 보였지만, 비윤리적 응답을 탐지하는 능력은 부족했습니다. 이는 LLM이 동기 면담의 윤리적 기준을 잘 준수하지 않음을 나타냅니다. 최종적으로 제안된 Chain-of-Ethic 프롬프트는 비윤리적 요청에 대한 대응에서 윤리적 응답 생성을 개선하는 데 기여했습니다. 이 연구는 LLM 기반 심리 치료에서 윤리적 안전성 평가 및 지침의 필요성을 강조합니다.



### NRC VAD Lexicon v2: Norms for Valence, Arousal, and Dominance for over 55k English Terms (https://arxiv.org/abs/2503.23547)
- **What's New**: 이 논문은 NRC VAD Lexicon v2를 소개하며, 이는 55,000개가 넘는 영어 단어와 구문에 대한 Valence, Arousal, Dominance에 대한 인간 평가를 포함합니다. 특히, 기존 버전(v1.0)에 비해 약 25,000개의 추가 단어와 10,000개의 다단어 구문이 처음으로 포함되었습니다. 이 사전은 심리학, 자연어 처리(NLP), 공공 건강, 디지털 인문학, 사회 과학 등 다양한 연구를 가능하게 합니다.

- **Technical Details**: 사전의 주요 차원은 Valence(정서적 가치), Arousal(각성 수준), Dominance(지배력)을 포함하며, 이들은 개인의 사회적 능력, 감정 조절 및 직장 내 성공에 영향을 미칩니다. 해당 연구는 이 차원들이 신뢰할 수 있는 연관성을 가지고 있음을 잘 보여줍니다. 이는 감정 및 언어 인식의 연구에서 중요한 기초 자료로 활용될 수 있습니다.

- **Performance Highlights**: NRC VAD Lexicon v2는 매우 신뢰할 수 있는 결과를 도출하였으며, 인간 평가의 의미에서 긍정적이고 부정적인 정서를 잘 반영하고 있습니다. 연구자들은 이 데이터셋을 활용하여 감정 인식, 언어의 의미 분석 및 다양한 사회적 현상에 대한 연구를 더욱 심층적으로 진행할 수 있을 것입니다.



### Whisper-LM: Improving ASR Models with Language Models for Low-Resource Languages (https://arxiv.org/abs/2503.23542)
Comments:
          26 pages, 6 figures, includes supplementary materials. Will be submitted to IEEE/ACM Transactions on Audio, Speech, and Language Processing

- **What's New**: 이 연구는 Whisper 모델과 전통적인 언어 모델을 통합하여 저자원 언어에서의 성능을 개선하는 방법을 제안합니다. Whisper는 다양한 언어에서 음성을 인식하는 멀티태스킹 ASR 모델로, 이 모델의 잠재력을 극대화하기 위해 언어 모델을 도입했습니다. 통계적 언어 모델을 통해 말하기 오류율(Word Error Rate)을 최대 51%까지 개선했으며, 일반적으로는 34%까지 향상시켰습니다.

- **Technical Details**: 연구는 Whisper 모델에 언어 모델을 통합하여 저자원 언어에서 성능을 최적화했습니다. 이를 위해 문맥의 일관성을 강화하고 음성 인식에서의 구문적 및 의미적 구조를 관리하는 데 도움을 주는 언어 모델 사용을 강조했습니다. 모델의 성능을 OOD(Out-Of-Distribution) 데이터에 대한 강인성 향상 측면에서 개선하였으며, 새로운 지표인 ERER(Effective Robustness of Relative Error Reduction)을 통해 성능을 정량화했습니다.

- **Performance Highlights**: 이 연구는 Whisper 및 언어 모델 통합을 통해 저자원 언어에서의 정확도와 강인성을 현저하게 높였습니다. 특히 OOD 데이터에서의 성능 향상은 n-gram 및 대형 언어 모델을 사용하여 가능했습니다. 이러한 접근 방식은 다양한 언어 환경에서 강력하고 포괄적인 ASR 모델을 개발할 수 있는 길을 열어줍니다.



### Question-Aware Knowledge Graph Prompting for Enhancing Large Language Models (https://arxiv.org/abs/2503.23523)
- **What's New**: 본 연구에서는 질문-인지 지식 그래프 프롬프팅(Question-Aware Knowledge Graph Prompting, QAP)이라는 새로운 접근 방식을 제안합니다. QAP는 질문 임베딩을 GNN 집계 과정에 통합하여 KG(coherent Knowledge Graph) 관련성을 동적으로 평가함으로써, 이론적 한계를 극복하려고 합니다. 또한, QAP는 글로벌 어텐션을 사용하여 답변 옵션 간의 관계를 포착하여 소프트 프롬프트의 지식을 강화합니다.

- **Technical Details**: QAP는 세 가지 주요 단계로 구성됩니다: (i) 서브그래프 검색(Subgraph Retrieval), (ii) 질문-인지 이웃 집계(Question-Aware Neighborhood Aggregation, QNA), (iii) 글로벌 어텐션 기반 프롬프팅(Global Attention-Derived Prompting, GTP). QNA는 질문에 따라 KG 정보를 강조하고 적절한 출력을 생성하도록 설계되었습니다. GTP는 모든 옵션 간의 관계를 포착하여 글로벌 정보를 포함하는 소프트 프롬프트 토큰 임베딩을 생성합니다.

- **Performance Highlights**: 실험 결과, QAP는 다양한 데이터셋에서 기존의 최첨단 방법들을 초월하는 성능을 보였습니다. 이는 QAP가 도메인 특화 추론 작업에서 효과적이고 우수한 결과를 낼 수 있음을 확인시켜 줍니다. QAP의 도입으로 LLM의 추론 능력이 크게 향상될 것으로 기대됩니다.



### If an LLM Were a Character, Would It Know Its Own Story? Evaluating Lifelong Learning in LLMs (https://arxiv.org/abs/2503.23514)
- **What's New**: 이번 논문에서는 큰 언어 모델(LLMs)이 다중 대화 상호작용 중에 일관된 성격을 보이는 현상을 관찰하고, 이를 평가하기 위해 LIFESTATE-BENCH라는 새로운 벤치마크를 도입합니다. LLMs의 상태 진화(상태 변화를 측정하는 지표)를 평가하며, 기존 벤치마크의 한계를 극복하고자 합니다. 특히 에피소드 기반 데이터셋을 통해 현재와 과거의 대화 맥락을 연결하는 방식을 탐구합니다.

- **Technical Details**: LIFESTATE-BENCH는 Hamlet 및 합성 스크립트와 같은 두 가지 에피소드 데이터셋으로 구성되어 있습니다. 이 데이터셋은 LLMs의 자기 인식, 기억 회상, 관계 추적을 평가하기 위한 사실 기반 질문 차원을 설계했습니다. 또한, 비모수적(non-parametric) 방법과 모수적(parametric) 방법을 통해 LLM의 장기 기억 능력을 측정하는 다양한 접근 방식을 탐구했습니다.

- **Performance Highlights**: 실험 결과, 비모수적 방법이 모수적 방법보다 상태 기반 학습에 더 효과적임을 보여주었습니다. 그러나 모든 모델은 대화가 진행됨에 따라 기억 상실(catasrophic forgetting) 문제에 직면하며, 지속적인 개선이 필요하다는 점이 분명해졌습니다. 본 논문은 LMs의 평생 학습 능력을 평가하고 향후 연구 방향을 제시하는 데 중요한 기여를 하고 있습니다.



### RARE: Retrieval-Augmented Reasoning Modeling (https://arxiv.org/abs/2503.23513)
Comments:
          Work in progress

- **What's New**: RARE(Retrieval-Augmented Reasoning Modeling)라는 새로운 패러다임은 지식 저장소와 추론 최적화 과정을 분리하여 도메인 특화 지능을 효과적으로 개발할 수 있는 방법을 제시합니다. 특히 이 모델은 외부 지식 저장소에서 도메인 지식을 검색하여 훈련 중에 내부적으로 도메인 별 사고 패턴을 학습하도록 설계되었습니다. 연구 결과, RARE로 훈련된 경량 모델들이 기존의 대규모 모델(GPT-4 등)을 초월하는 성능을 보였습니다.

- **Technical Details**: RARE의 핵심 원리는 도메인 지식을 외부화하고 도메인 사고를 내부화하는 것입니다. 이를 통해 경량 모델들은 굳이 방대한 양의 지식을 기억해야 하는 부담 없이 도메인 특화 추론 능력에 집중할 수 있습니다. 훈련 중에 검색된 지식이 훈련 프롬프트에 주입되어, 지식을 기억하는 대신 맥락 적용을 강조하는 학습 목표로 전환됩니다.

- **Performance Highlights**: RARE로 훈련된 Qwen-2.5-7B 모델은 PubMedQA에서 78.63%, CoVERT에서 74.14%의 정확도로 기존의 GPT-4 모델을 초월함으로써 성능의 우수성을 입증했습니다. 이러한 성과는 경량 모델이 대규모 모델에 비해 최대 20%의 정확도 향상을 달성했음을 보여줍니다. RARE는 지속적인 지식 업데이트와 사실 정확성 문제를 해결할 수 있는 새로운 패러다임을 보여줍니다.



### SCORE: Story Coherence and Retrieval Enhancement for AI Narratives (https://arxiv.org/abs/2503.23512)
- **What's New**: 이번 논문에서는 SCORE(Story Coherence and Retrieval Enhancement)라는 프레임워크를 제안합니다. SCORE는 Dynamic State Tracking, Context-Aware Summarization, Hybrid Retrieval의 세 가지 구성 요소를 통합하여 스토리의 일관성과 감정적 깊이를 개선하는 데 기여합니다. 이 시스템은 Retrieval-Augmented Generation (RAG) 파이프라인을 통해 맥락 일관성을 검증합니다.

- **Technical Details**: SCORE의 핵심 구성 요소는 1) Dynamic State Tracking(상징 논리를 사용하여 캐릭터 및 객체의 상태를 모니터링하고 수정), 2) Context-Aware Summarization(플롯 발전과 감정적 진행을 포착하는 계층적 에피소드 요약 생성), 3) Hybrid Retrieval(TF-IDF 키워드 관련성과 코사인 유사성 기반 의미적 임베딩을 결합하여 일관성을 강화)입니다. 이 방법론은 인간의 내러티브 일관성 심리학 모델에 기반하여 내러티브 엔트로피를 감소시키고 일관성을 보장합니다.

- **Performance Highlights**: SCORE는 NCI-2.0 벤치마크에서 23.6% 더 높은 일관성, EASM 메트릭에서 89.7%의 감정적 일관성, 그리고 기준 GPT 모델에 비해 41.8% 적은 허위 정보가 나타나는 성과를 보여줍니다. 이를 통해 SCORE는 스토리 생성 시스템에서 요구되는 높은 수준의 장기적 일관성을 유지하는 데 효과적인 솔루션을 제공합니다.



### Evolutionary Prompt Optimization Discovers Emergent Multimodal Reasoning Strategies in Vision-Language Models (https://arxiv.org/abs/2503.23503)
Comments:
          Published at ICLR 2025 Workshop on Reasoning and Planning for LLMs

- **What's New**: 이번 논문에서는 비전-언어 모델(Vision-Language Models, VLMs)에서 프롬프트를 최적화하는 새로운 프레임워크를 제안합니다. 이 방법은 모델 재학습 없이 시각적 과제를 위해 프롬프트를 진화 알고리즘을 통해 업데이트하는 방식으로, 기존 프롬프트 업데이트 알고리즘의 한계를 극복합니다. 연구 결과, 이 접근 method는 언어 모델이 점진적인 문제 해결 능력을 독립적으로 발견할 수 있도록 돕는 것으로 나타났습니다.

- **Technical Details**: 우리는 비전-언어 태스크에 맞춰 진화 프롬프트 최적화 프레임워크를 설계하였습니다. 이 방법은 표준 VLMs에서 모델의 재학습 없이 동작하며, 시각 및 언어 처리를 조정하는 태스크별 프롬프트를 진화시킵니다. 우리의 실험은 공통 감각 물리학, 복잡한 카운팅 및 수학적 추리를 요구하는 다중 모달 벤치마크에서 수행되었습니다.

- **Performance Highlights**: 실험 결과, 이 방법을 통해 프롬프트 변이를 통해 재귀적 도구 사용, 계층적 이미지 분할 및 동적 프로그래밍과 유사한 카운팅 등의 복잡한 행동을 발견할 수 있었습니다. 우리의 방법은 상대적으로 약 50%의 성능 향상을 나타내며, 몇 개의 레이블이 부착된 예제만으로도 신뢰할 수 있는 일반화를 이룹니다.



### Order Independence With Finetuning (https://arxiv.org/abs/2503.23483)
Comments:
          Published as a Bi-Align workshop paper at ICLR 2025

- **What's New**: 이 논문은 세트 기반 프롬프트(Set-Based Prompting, SBP)를 활용하여 대형 언어 모델(LLMs)의 순서 의존성을 줄이는 새로운 미세 조정 방법을 제안합니다. 기존의 SBP 방법은 순서를 변경해도 동일한 의미를 가지는 답안 후보에 대해 모델의 예측이 일관되도록 하는 데 초점을 맞추었습니다. 그러나 본 연구에서는 SBP를 훈련 과정에 통합함으로써 성능 저하를 방지하고 모델의 일반적인 언어 모델링 능력을 유지할 수 있음을 보였습니다.

- **Technical Details**: 논문에서는 SBP를 LLM의 훈련 과정에 통합하는 미세 조정 전략을 소개하며, 이를 통해 SBP 형식의 프롬프트가 모델의 학습된 매니폴드(training manifold)에 더 가까워지도록 합니다. 특히, 마진 기반 대조 손실(margin-based contrastive loss)을 사용하여 정답과 오답 간의 구분을 명확히 하는 방법을 채택하였습니다. 이를 통해 SBP 형식의 입력을 훈련할 때 발생하는 분포 이동을 효과적으로 해결하였습니다.

- **Performance Highlights**: 실험 결과, SBP로 미세 조정된 모델은 다중 선택 질문에서 순서에 독립적인 정답률을 크게 향상시키며, CSQA 및 ARC Challenge 데이터셋에 대한 일반화 성능도 개선되었습니다. 또한, 모델이 WikiText-103의 perplexity를 유지함으로써, 보다 넓은 언어 모델링 능력을 저하시키지 않으면서도 안정성을 확보하는 데 성공하였습니다.



### Speculative End-Turn Detector for Efficient Speech Chatbot Assistan (https://arxiv.org/abs/2503.23439)
Comments:
          Preprint

- **What's New**: 이번 논문에서는 대화 시스템의 중요한 문제인 end-turn detection (ETD) 문제를 해결하기 위해 ETD Dataset을 출시했습니다. 이 데이터셋은 텍스트 대화 데이터를 기반으로 생성된 합성 음성 데이터와 웹 소스에서 수집된 실제 음성 데이터로 구성되어 있습니다. 또한, 자원 제한 환경에서 실시간 ETD를 개선하기 위한 새로운 협업 추론 프레임워크인 SpeculativeETD를 제안합니다.

- **Technical Details**: SpeculativeETD는 경량의 GRU 기반 모델과 고성능 Wav2vec 기반 모델을 조합하여 효율성과 정확성을 균형있게 유지합니다. 경량 모델은 로컬 디바이스에서 빠르게 비말 단위를 탐지하고, 침묵이 감지되면 고성능 모델에 질의하여 효과적으로 턴의 종료 여부를 판단합니다. 이 접근 방식은 고성능 모델이 실시간으로 작동할 필요가 없으므로 필요한 계산량을 대폭 줄일 수 있습니다.

- **Performance Highlights**: 실험 결과, SpeculativeETD는 ETD 정확도를 크게 향상시키면서도 계산 요구량을 최소화하는 것으로 나타났습니다. 120,000개 이상의 샘플과 300시간 이상의 대화 데이터가 포함된 ETD 데이터셋은 모델 트레이닝 및 평가에 유용하게 활용될 것입니다. 데이터셋과 코드는 리뷰 후 공개될 예정입니다.



### CoRanking: Collaborative Ranking with Small and Large Ranking Agents (https://arxiv.org/abs/2503.23427)
- **What's New**: 최근 논문에서는 효율적이고 효과적인 랭킹을 위해 소규모와 대규모 랭킹 모델을 결합한 새로운 협업 랭킹 프레임워크인 CoRanking을 제안했습니다. CoRanking은 작은 리랭커가 후보 단락을 사전 랭킹하여 최상위 부분에 관련 단락을 배치하고, 그 후 LLM(대형 언어 모델)에 의해 최상위 단락들만을 재랭킹함으로써 전반적인 효율성을 크게 향상시킵니다. 또한, 리인포스먼트 러닝을 통해 작은 리랭커에서 나오는 단락의 순서를 조정하는 새로운 구조인 POA를 도입하여 LLM의 선호와 맞추고 있습니다.

- **Technical Details**: CoRanking 프레임워크는 세 가지 주요 구성요소로 이루어져 있습니다. 첫 번째는 소규모 리스트 리랭커(SLR)로 초기 단락 재랭킹을 수행하고, 두 번째는 단락 순서 조정기(POA)로 SLR의 최상위 단락 순서를 LLM의 선호와 맞추며, 세 번째는 최종 랭킹을 위한 LLM 리스트 리랭커(LLR)입니다. 기존 랭킹 방법에서 제기된 한계를 극복하며, CoRanking은 성능 향상과 더불어 70%의 랭킹 지연 시간을 줄였습니다.

- **Performance Highlights**: CoRanking은 다수의 정보 검색(IR) 벤치마크에서 광범위한 실험을 통해 효율성과 효과성 면에서 우수한 결과를 보였습니다. 이 프레임워크는 순위 정확도를 높이는 동시에 랭킹 시간을 70% 단축하여 기존 LLM 리스트 리랭커보다 더욱 향상된 성능을 보여주었습니다. 이러한 성과는 작은 리랭커와 LLM 간의 협업을 통해 이루어진 결과입니다.



### An Analysis of Decoding Methods for LLM-based Agents for Faithful Multi-Hop Question Answering (https://arxiv.org/abs/2503.23415)
- **What's New**: 본 연구에서는 대형 언어 모델(LLM)의 응답의 사실 충실도(faithfulness)를 향상시키기 위한 다채로운 디코딩 전략과 ReAct 프레임워크의 결합이 효과적임을 제안합니다. 특히, ReAct와 검증된 외부 지식을 활용하여 부정확성을 감소시키고, Multi-Hop Question Answering (QA) 과제에서 성능 향상을 이끌어낼 수 있음을 보여주었습니다. 따라서 본 연구는 LLM의 응답에 대한 맥락 적합성 평가에 기여하며, 훈련이 필요 없는 디코딩 방법을 사용하여 더욱 접근 가능하도록 하고 있습니다.

- **Technical Details**: ReAct 프레임워크는 순차적으로 의사결정 및 행동을 수행하는 구조로 이루어져 있으며, LLM이 특정 키워드를 포함한 문서의 첫 문장을 검색하는 도구를 사용하여 정보를 획득합니다. 본 연구에서는 Context-Aware Decoding (CAD), Decoding by Contrasting Layers (DoLa), Decoding by Contrasting Retrieval Heads (DeCoRe)와 같은 디코딩 방법을 분석합니다. 이러한 방법들은 LLM의 차별화된 계층이나 주의 헤드를 활용하여 결과의 사실성을 향상시키는 데 기여합니다.

- **Performance Highlights**: 연구 결과, ReAct와 DoLa를 결합했을 때 HotpotQA에서 F1 점수가 19.5에서 32.6으로 증가하였으며, 답변 인식률(Answer F1)에서 최대 13.1의 향상을 보였습니다. 다양한 데이터셋에서 일관된 성능 향상을 확인했으며, 지원 문서에서 정답이 포함된 비율(Answer Support Recall) 역시 명확하게 개선되었습니다. 이를 통해, 복합적인 질문 응답 과제에서 LLM의 효율성이 극대화될 수 있음을 확인했습니다.



### ToRL: Scaling Tool-Integrated RL (https://arxiv.org/abs/2503.23383)
- **What's New**: 이 논문에서는 ToRL(Tool-Integrated Reinforcement Learning)이라는 프레임워크를 소개합니다. ToRL은 대규모 언어 모델(LLM)을 강화 학습을 통해 자율적으로 계산 도구를 사용할 수 있게 훈련합니다. 기존의 감독된 미세 조정(supervised fine-tuning)과는 달리, ToRL은 모델이 도구 사용을 위한 최적의 전략을 탐색하고 발견할 수 있도록 합니다.

- **Technical Details**: ToRL 프레임워크는 기존의 감독된 미세 조정 제약 없이 기본 모델부터 직접 강화 학습을 확장합니다. 이는 도구 통합 추론(Tool-Integrated Reasoning, TIR) 방법론과 결합되어 있으며, 모델이 코드 인터프리터를 통해 코드를 작성하고 실행하도록 합니다. 이에 따라 모델은 복잡한 문제 해결 시 피드백을 기반으로 동적으로 추론 경로를 조정하며, 각 단계에서의 코드는 이전 실행 결과에 의해 영향을 받습니다.

- **Performance Highlights**: Qwen2.5-Math 모델을 사용한 실험에서는 ToRL-7B가 AIME~24에서 43.3% 정확도를 기록하여 기존의 도구 통합 강화 학습을 통한 모델보다 14% 향상된 성과를 나타냈습니다. 모델은 코드 사용 감소, 비효율적인 코드 패턴 자가 규제 및 계산적, 분석적 추론 간의 동적 적응 같은 emergent behaviors를 보이는 것으로 분석되었습니다. 이를 통해 ToRL이 LLM의 추론 능력을 향상시키는 유망한 방향임을 제시하였습니다.



### FeRG-LLM : Feature Engineering by Reason Generation Large Language Models (https://arxiv.org/abs/2503.23371)
Comments:
          Accepted to NAACL 2025 Findings

- **What's New**: 이번 연구에서 제안하는 	extbf{FeRG-LLM}은 80억 개의 파라미터 규모로 자동으로 feature engineering을 수행할 수 있도록 설계된 대형 언어 모델입니다. 두 단계 대화(dialogue)를 통해 머신러닝 작업을 분석하고 새로운 feature를 발견하는 능력을 보유하고 있으며, Chain-of-Thought (CoT) 기능을 활용합니다. 이러한 자동화된 feature 생성 방법은 인적 자원의 부담을 줄이고 기업 환경에 적합한 솔루션을 제공합니다.

- **Technical Details**: FeRG-LLM은 두 단계의 대화 모델을 통해 Llama 3.1 8B 모델을 미세 조정하여 Direct Preference Optimization (DPO)를 통합함으로써 feature 생성을 위한 향상된 근거를 제공합니다. 이는 binary classification 작업을 포함한 여러 데이터셋에서 효과적으로 평가되었으며, 70B 모델과 동등하거나 더 나은 성능을 보였습니다. DPO는 피드백을 받아 모델의 성능을 개선할 수 있는 방법을 제공합니다.

- **Performance Highlights**: 실험 결과 FeRG-LLM은 자원 소모가 적고 추론 시간을 단축하면서도 대부분의 데이터셋에서 Llama 3.1 70B 모델과 유사하거나 더 나은 성능을 달성했습니다. 특히, 분류 작업에서 다른 연구보다 뛰어난 성능을 보였고 회귀 작업에서도 우수함을 입증했습니다. 리소스 제약이 있는 환경에서도 안정적인 feature 생성을 보장하여 머신러닝 성능을 높일 수 있습니다.



### Mixture of Routers (https://arxiv.org/abs/2503.23362)
Comments:
          10 pages,4 figures

- **What's New**: 본 논문에서는 파라미터 효율적인 미세조정(PET-Fine Tuning, PEFT)과 Mixture-of-Experts (MoE)를 결합한 새로운 미세조정 방법인 Mixture of Routers (MoR)를 제안합니다. MoR은 여러 서브 라우터(sub-router)를 사용하여 전문가 모델(expert model)을 선택하고, 메인 라우터(main router)가 이들의 선택을 결정하여 모델의 성능을 개선합니다. MoR은 다양한 복잡성을 가질 수 있는 작업에 유연하게 적용 가능하며, 기존 MoE 모델의 라우터 레이어를 대체할 수 있는 플러그 앤 플레이 솔루션입니다.

- **Technical Details**: MoR은 여러 개의 서브 라우터(sub-router)를 통해 공동 선택을 수행하며, 각 서브 라우터의 기여로 최종 결정을 내리도록 설계되었습니다. 여기서 메인 라우터(main router)가 서브 라우터의 점수를 기반으로 상위 서브 라우터를 선택하여, 오류를 최소화할 수 있도록 합니다. 이러한 방식은 각 전문가 선택을 위한 점수를 조정하여, 최종 추론 단계에서 가장 우수한 전문가를 선택합니다.

- **Performance Highlights**: 여섯 가지 벤치마크에서 실험을 수행한 결과, MoR은 대부분의 작업에서 기존 모델에 비해 평균 1%의 성능 향상을 보였습니다. MoR은 다양한 NLP 및 상식 추론(Common Sense Reasoning, CR) 작업에서 우수한 성능을 나타내며, 최적의 경량화 미세조정 솔루션으로 자리매김할 수 있음을 보여줍니다. 또한, Consistent Routing Weighting (CRW)이라는 변형을 통해 전이 학습에서의 불안정성을 개선하여 모델의 안정성과 일반화 능력을 효과적으로 향상시켰습니다.



### Discovering Knowledge Deficiencies of Language Models on Massive Knowledge Bas (https://arxiv.org/abs/2503.23361)
- **What's New**: 본 논문에서는 Stochastic Error Ascent (SEA)라는 확장 가능하고 효율적인 프레임워크를 제안하여, LLM(대형 언어 모델)의 지식 결함을 발견하는 방법을 모색하고 있습니다. 기존의 정적인 벤치마크 테스트는 방대한 지식 기반을 다루기에는 비효율적이며, SEA는 쿼리 예산 내에서 지식 결함을 자동으로 발견할 수 있도록 설계되었습니다. 이를 통해 LLM의 지식 부족을 보다 효과적으로 분석할 수 있는 새로운 접근 방식을 제공합니다.

- **Technical Details**: SEA는 고차원적인 오류 유도 과정을 통해, 이전의 실패와 의미적으로 유사한 새로운 후보를 반복적으로 선택함으로써 지식 결함을 발견하는 스토캐스틱 최적화 문제로 구성됩니다. 이 접근법은 문서와 문단 수준에서의 계층적 검색을 채택하고, 오류 전파를 모델링하기 위해 관계 방향 비순환 그래프(relation DAG)를 구축하여 체계적인 실패 모드를 이해합니다. SEA는 기존 방법론인 Automated Capability Discovery 및 AutoBencher와 비교하여, 오류 발견에서 상당한 성과를 보입니다.

- **Performance Highlights**: SEA는 기존의 두 기준선 방법보다 40.7배 더 많은 지식 오류를 발견하고, 오류당 비용을 각각 599배 및 9배 줄여 효율성을 크게 향상시켰습니다. 1,000개의 무작위 샘플 질문에 대한 인간 평가에서 모든 모델이 생성한 질문의 신뢰성이 100%로 평가되었으며, 모든 구성 요소가 일관성 있게 기여하고 있다는 것을 확인했습니다. 이러한 결과는 LLM 모델의 특정 취약점을 밝히고, 향후 데이터 수집 전략을 안내할 수 있는 기반을 마련합니다.



### Not All LoRA Parameters Are Essential: Insights on Inference Necessity (https://arxiv.org/abs/2503.23360)
- **What's New**: 이 논문에서는 LoRA(Low-Rank Adaptation)의 각 레이어가 모델의 예측 능력에 기여하는 바를 분석하고, 특정 LoRA 레이어가 모델의 추론 및 이해에 어떻게 결정적인 역할을 하는지 가설을 세웁니다. 저자들은 ''boundary layer''라는 개념을 도입하여, 중요한 LoRA 레이어를 식별하고, 추론 중에 이 경계를 넘어서는 모든 LoRA 레이어를 제거하는 방법을 제안합니다. 이를 통해 대가의 언어 모델의 성능을 향상시키는 효과적인 접근 방식을 보여줍니다.

- **Technical Details**: LoRA는 파라미터 효율적인 미세 조정(parameter-efficient fine-tuning) 방법으로, 각 LLM 레이어에 대해 학습 가능한 어댑터를 도입하여 나머지 파라미터를 고정하는 방식으로 작동합니다. 저자들은 작은 검증 샘플 세트를 분석하여 ''boundary layer''를 식별하고, 여러 강력한 기준 모델을 사용하여 제안된 방법의 효율성을 평가합니다. LoRA의 각 레이어에 대한 임팩트 연구를 통해, 각 레이어가 추출한 정보를 기반으로 상위 레이어가 효과적으로 응답을 생성할 수 있음을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 네 가지 널리 사용되는 텍스트 생성 데이터셋을 대상으로 세 가지 최첨단 베이스라인 모델을 통해 평가되었습니다. 결과는 중요한 LoRA 레이어를 선택적으로 유지할 경우, 일관되며 유의미한 성능 향상이 있음을 시사합니다. 이 연구는 효율성을 유지하면서도 성능 향상에 기여하는 매우 효과적인 접근 방식으로 자리매김하고 있습니다.



### Linguistic Loops and Geometric Invariants as a Way to Pre-Verbal Thought? (https://arxiv.org/abs/2503.23311)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 언어학적 변환(linguistic transformation), 언어학적 루프(linguistic loop), 그리고 의미 결핍(semantic deficit)이라는 개념을 소개합니다. Lie 그룹 이론(Lie group theory)과 기하학적 기법을 이용하여 언어학적 루프의 구조적 특성을 포착하는 불변량(invariants)을 정의했습니다. 이 결과는 언어 연구에서 Lie 이론과 고차원 기하학을 활용하는 새로운 연구 방향의 시작을 제안합니다.

- **Technical Details**: 언어학적 루프는 언어적 변환을 통해 원래의 의미(core)를 유지하며 변형된, 제어된 방법으로 변환하는 주장(map)에 기반합니다. 이 루프는 물리학자 K. G. Wilson이 1974년에 도입한 윌슨 루프(Wilson loop)와 유사한 구조를 가지며, 비선형적 양자 중력(nonperturbative quantum gravity)에서도 활용됩니다. 본 연구는 기존의 언어학 모델에서 벗어나, 메타 언어적 사고(meta-linguistic thought), 즉 언어로 표현되기 이전의 인지 구조를 수학적으로 규명하는 방향으로 진행됩니다.

- **Performance Highlights**: 고전적 및 양자 중력 이론의 수립에서 일반 상대성이론을 넘어서는 새로운 관점이 제시됩니다. 연구는 언어 변환을 특정한 불변량으로 규명하는 데 집중하고 있으며, LLM에서 발생하는 특정한 언어 학습 루프를 이용하여 기하학적 불변량을 찾아내는 방법론을 제안하고 있습니다. 이러한 작업은 주요하게 인간의 정신 표현과 관련된 연구에 기여할 것으로 기대됩니다.



### Focus Directions Make Your Language Models Pay More Attention to Relevant Contexts (https://arxiv.org/abs/2503.23306)
- **What's New**: 이번 논문에서는 장기 컨텍스트를 가진 대형 언어 모델(LLMs)이 비관련 컨텍스트에 주의를 분산하는 이유를 규명하고자 합니다. 저자들은 주목할만한 'contextual heads'라는 주의 헤드를 식별하고, 이 헤드들이 관련 컨텍스트에 충분한 주의를 기울이지 못할 때 산만해진다고 설명합니다. 이 연구를 통해 주의의 방향성을 조정하여 모델의 성능을 향상시킬 수 있는 방법을 제시합니다.

- **Technical Details**: 저자들은 'contextual scoring'이라는 방법론을 도입하여 LLM이 생성하는 과정에서 관련 컨텍스트에 대한 주의의 강도를 측정합니다. 이 방법을 통해 관련 컨텍스트에 가장 많은 주의를 기울이는 'contextual heads'를 확인하고, 이를 통해 주의의 강도를 조정하여 모델의 성능을 개선하는 방식으로 연구를 진행합니다. 또한, 'focus directions'라는 새로운 방향성을 소개하여, 모델이 명시적으로 관련 컨텍스트를 지정하지 않고도 더 많은 주의를 집중할 수 있도록 합니다.

- **Performance Highlights**: 연구 결과에 따르면, focus directions는 LLM의 긴 컨텍스트 작업에서 성능을 개선하는 데 기여합니다. 저자들은 이 방법이 LLM의 작업 정합성 문제를 완화할 수 있음을 발견했으며, 다양한 LLM 모델에 적용하여 효과를 평가하였습니다. 이러한 발견은 장기 컨텍스트 LLM 조정에 대한 추가 연구를 촉진할 수 있을 것으로 기대합니다.



### Using Source-Side Confidence Estimation for Reliable Translation into Unfamiliar Languages (https://arxiv.org/abs/2503.23305)
Comments:
          7 pages, 5 figures, 1 table. Submitted to ACL 2025 System Demonstrations

- **What's New**: 이번 연구에서는 상호작용형 기계 번역(Interactive Machine Translation, MT) 시스템을 소개합니다. 이 시스템은 사용자들이 목표 언어에 능숙하지 않을 때도 신뢰성과 설명력을 향상시키기 위해 설계되었습니다. 특히, 번역 오류를 수정할 수 있는 사용자 개입을 허용하기 위해, 불확실성이 높은 단어를 강조하고 이에 대한 수정 제안을 제공합니다.

- **Technical Details**: 이 시스템은 소스 측 신뢰도 추정을 통해 불확실성이 높은 단어를 강조합니다. 구체적으로, 소스 단어의 임베딩에 대한 출력 시퀀스의 민감도를 측정하여 불확실성 점수를 계산합니다. 이렇게 얻은 점수는 단어의 온도를 평가하며, 밀접한 단어의 변화를 감지함으로써 보다 정확한 번역을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 기존의 정렬 기반 방법보다 번역 오류 감지에서 우수한 성능을 보였습니다. 이 시스템은 특히 소스 언어에 능숙한 사용자들이 보다 신뢰할 수 있는 번역 결과를 받아볼 수 있게끔 하는 데 기여할 것입니다. 향후, 이러한 불확실성 점수는 사전 정의 검색과 같은 다른 애플리케이션에도 통합될 수 있습니다.



### Advancing Sentiment Analysis in Tamil-English Code-Mixed Texts: Challenges and Transformer-Based Solutions (https://arxiv.org/abs/2503.23295)
- **What's New**: 이번 연구에서는 Tamil-English 코드 혼합 텍스트의 감성 분석(sentiment analysis) 작업을 첨단 transformer 기반 모델을 활용하여 탐구했습니다. 문법적 불일치, 철자 변형(orthographic variations), 그리고 음성적 모호성(phonetic ambiguities) 등의 문제점을 다루었으며, 기존 데이터셋의 한계와 주석(annotation)에서의 격차를 분석했습니다. 이러한 문제들을 해결하기 위해 더 크고 다양한 코퍼스(corpora)의 필요성을 강조합니다.

- **Technical Details**: XLM-RoBERTa, mT5, IndicBERT, RemBERT 등 다양한 transformer 아키텍처가 저자원(low-resource) 코드 혼합 환경에서 평가되었습니다. 성능 지표(performance metrics)를 분석하며, 다국어 감성 분류(multilingual sentiment classification)에서 특정 모델들이 효과적으로 작용하는 것을 확인했습니다. 이 연구는 데이터 증강(data augmentation), 음성 정규화(phonetic normalization), 하이브리드 모델링(hybrid modeling) 접근 방식을 통해 정확성을 높이기 위한 추가적인 발전이 필요하다는 점을 제안합니다.

- **Performance Highlights**: 연구 결과, 특정 transformer 모델들이 코드 혼합 텍스트에 대한 감성 분석 수행에서 뛰어난 성능을 보였음을 밝혔다. 저자원 상황에서의 성능 분석을 통해 각 모델의 강점과 약점을 평가했습니다. 앞으로의 연구 방향으로 코드 혼합 텍스트에 대한 감성 분석을 개선하기 위한 다양한 접근 방식이 제안되었습니다.



### Cocktail: Chunk-Adaptive Mixed-Precision Quantization for Long-Context LLM Inferenc (https://arxiv.org/abs/2503.23294)
Comments:
          Accepted by the Design, Automation, and Test in Europe 2025 (DATE 2025)

- **What's New**: 최근 대형 언어 모델(LLMs)은 점점 더 긴 컨텍스트를 처리할 수 있게 되었습니다. 그러나 너무 긴 컨텍스트는 이견이 발생할 수 있는 추론 지연(inference latency)과 GPU 메모리 사용량의 증가를 초래합니다. 본 논문에서는 KV 캐시를 최적화하기 위해 단계별 적응형 혼합 정밀도 양자화(chunk-adaptive mixed-precision quantization)라는 새로운 접근 방식인 Cocktail을 소개합니다.

- **Technical Details**: Cocktail은 주로 두 개의 모듈로 구성됩니다: 청크 수준 양자화 검색(chunk-level quantization search) 및 청크 수준 KV 캐시 계산(chunk-level KV cache computation). 청크 수준 양자화 검색 모듈은 쿼리와 각 컨텍스트 청크 간의 유사도 점수를 계산하여 해당 청크에 대해 최적의 비트 너비(bitwidth)를 결정하며, 이를 통해 모델의 정확성을 유지합니다. 또한, 청크 수준 KV 캐시 계산 모듈은 혼합 정밀도 양자화로 인한 하드웨어 비효율성을 피하기 위해 양자화를 수행하기 전에 KV 캐시 청크를 재정렬합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, Cocktail은 다양한 모델 및 데이터셋에서 최신 KV 캐시 양자화 방법들보다 더 나은 모델 정확성과 더 낮은 지연, 그리고 메모리 사용량을 줄이는 성과를 보입니다. 이 연구는 긴 컨텍스트를 가진 LLM 추론의 성능을 크게 향상시키는 데 기여할 것입니다.



### Extracting Patient History from Clinical Text: A Comparative Study of Clinical Large Language Models (https://arxiv.org/abs/2503.23281)
- **What's New**: 이번 연구에서는 환자의 주요 불만(Chief Complaint, CC), 현재 병력(History of Present Illness, HPI), 과거 및 가족의 사회적 병력(Past, Family, and Social History, PFSH)과 관련된 의료 기록 엔터티(Medical History Entities, MHEs)를 추출하는 방식에 대해 발표하였습니다. 이는 비정형화된 임상 노트를 체계적인 전자 건강 기록(Electronic Health Records, EHRs)으로 변환하여 의료 제공의 연속성, 의료 코딩, 품질 지표 등의 downstream 작업을 효율화하는 데 기여합니다. 연구에서는 최신 임상 대형 언어 모델(Fine-tuned Clinical Large Language Models, cLLMs)을 활용하여 이러한 MHE를 인식하고, 노트 특성이 모델의 정확도에 미치는 영향을 분석하였습니다.

- **Technical Details**: 연구팀은 MTSamples 리포지토리에서 61개의 외래 환자 관련 임상 노트로부터 1,449개의 MHE를 주석 처리하고, 인지 모델을 인식하기 위해 7개의 최첨단 cLLMs를 파인튜닝(Fine-tuning) 하였습니다. 추가로, 문제, 테스트, 치료 및 기타 기본 의료 엔터티(Basic Medical Entities, BMEs)를 통합하여 모델 성능을 평가하였습니다. 실험은 zero-shot 설정에서 GPT-4o와 비교하여 이루어졌으며, 텍스트 특성이 모델의 정확도에 미치는 영향에 대한 오류 분석도 수행되었습니다.

- **Performance Highlights**: 연구 결과, cLLMs는 MHE 추출에 필요한 시간을 20% 이상 단축시킬 잠재력을 보여주었습니다. 그러나 다의성(polysomy)의 특성과 비의료 용어의 빈번한 사용으로 인해 MHE 탐지에서 여전히 어려움이 존재했습니다. 특히, GatorTron과 GatorTronS 두 가지 모델이 가장 높은 성능을 발휘했으며, 사전에 식별된 BME 정보 통합이 특정 엔터티에 대한 모델 성능 향상에 기여했습니다. 또한, 텍스트 길이, 엔터티 길이, 세분화와 같은 특성이 모델 성능에 미치는 영향에 대한 유의미한 결과가 도출되었습니다.



### PromptDistill: Query-based Selective Token Retention in Intermediate Layers for Efficient Large Language Model Inferenc (https://arxiv.org/abs/2503.23274)
- **What's New**: PromptDistill은 훈련 없이 추론 효율을 높이면서 생성 품질을 유지하는 새로운 방법론입니다. 이 방법은 초기 레이어에서 주의(attention) 상호작용을 활용하여 가장 정보가 풍부한 토큰을 식별하고 선택하여, 계산 부담을 줄이면서도 필요한 맥락 정보를 집중할 수 있게 합니다. 기존의 방법들과는 달리 PromptDistill은 전체 입력을 처리한 후에 압축하는 게 아니라, 동적으로 가장 관련성 높은 토큰에 리소스를 할당합니다.

- **Technical Details**: PromptDistill은 각 레이어의 쿼리와 키 간의 내적(dot product)을 활용하여 중요 토큰을 선택합니다. 선택된 토큰의 숨겨진 상태를 유지하면서 이후 레이어에서 계속해서 처리되도록 하여 맥락 정보를 보존합니다. 초기 레이어에서 더 많은 토큰을 선택할수록 효율성을 높이는 데 기여할 수 있으며, 캐시 잘림(cache truncation)을 도입하여 비선택 토큰의 키와 값을 정리하여 메모리 효율성을 개선합니다.

- **Performance Highlights**: PromptDistill은 LLaMA 3.1, Qwen2 등 다양한 LLM에 적용되어 실험을 통해 기존 방법들과 비교할 때 성능과 시간 효율성을 모두 향상시킵니다. 특히 GemFilter와 비교할 때 $1	ext{%}$에서 $5	ext{%}$의 성능 향상을 달성하며, 모든 토큰을 처리할 필요가 없어 계산 비용을 절감하는 장점이 있습니다. 멀티 스테이지 선택을 통해 추론 효율성을 추가로 높일 수 있음을 보여주며, 이는 전체적인 성능 저하 없이 수행됩니다.



### Evaluating how LLM annotations represent diverse views on contentious topics (https://arxiv.org/abs/2503.23243)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 데이터 라벨링에 활용하는 방법을 제시하고, 이 모델들이 기존 자연어 모델에 비해 어떻게 성능이 개선되었는지를 강조합니다. 많은 기존 문헌에서는 LLM이 정확도, 정밀도, 재현율 및 F1 점수와 같은 표준 지표에서 다른 모델보다 더 우수하다고 언급하고 있습니다. 그러나 LLM의 언어 모델에 내재된 편향 문제도 조명되고 있으며, 특히 논란이 많은 주제와 관련된 부분에서 이러한 편향이 문제가 될 수 있습니다.

- **Technical Details**: LLMs의 성능 평가를 위해 연구팀은 NLPositionality 데이터셋, POPQUORN의 공격성 및 공손성 데이터셋, 위키피디아 댓글 데이터셋을 사용하였습니다. 연구는 성별, 인종, 교육 수준과 같은 인구통계학적 그룹에 따른 LLM의 동의 여부를 분석합니다. 공정성을 보장하기 위해 세 가지 다른 프롬프트를 사용하여 각 데이터셋에 대해 레이블을 생성하였으며, LLM의 동의 수준은 개인 라벨러와의 일치도를 사용하여 측정했습니다.

- **Performance Highlights**: 결과적으로, LLM은 인구통계학적 기준에 따라 평가자와의 상당한 불일치를 보이지 않았습니다. 오히려 라벨링 작업의 난이도와 모델, 그리고 프롬프트 사용이 LLM 합의에 더 큰 영향을 미쳤습니다. 연구 결과, 특정 집단의 의견이 과소 대표되는 문제는 크지 않은 것으로 나타났으며, 이러한 동의 수준은 데이터셋에 따라 다르기 때문에 LLM의 편향 문제는 LLM 자체의 문제가 아닐 수 있음을 시사하고 있습니다.



### Beyond speculation: Measuring the growing presence of LLM-generated texts in multilingual disinformation (https://arxiv.org/abs/2503.23242)
- **What's New**: 본 연구는 대형 언어 모델(LLM)의 발전과 이로 인해 생성된 다국어 텍스트의 품질이 높아짐에 따라 발생하는 허위정보(disinformation) 사용 가능성에 대한 우려를 다룹니다. 연구는 LLM이 최근 허위정보 데이터셋에 존재한다는 최초의 실증적 증거를 제공하며, ChatGPT의 출시 이후 기계에 의해 생성된 콘텐츠의 증가를 문서화합니다.

- **Technical Details**: 연구는 LLM에 의해 생성된 콘텐츠의 비율, 다양한 언어와 플랫폼, 시간대에 따른 패턴을 분석합니다. 이러한 분석은 자연 생태계의 한계에도 불구하고 특정 '롱테일(longtail)' 컨텍스트가 간과된 위험에 직면해 있음을 지적합니다.

- **Performance Highlights**: 연구 결과를 통해, LLM이 생성하는 텍스트가 인간이 작성한 텍스트와 구별하기 어렵다는 것을 보여주며, 이는 허위정보의 확산에 중요한 영향을 미칠 수 있습니다. 또한, 기계 생성 콘텐츠의 증가가 특정 플랫폼과 언어에서 두드러지며, 연구자들에게 이와 관련된 주의를 촉구합니다.



### RECALL-MM: A Multimodal Dataset of Consumer Product Recalls for Risk Analysis using Computational Methods and Large Language Models (https://arxiv.org/abs/2503.23213)
- **What's New**: 이 연구에서는 미국 소비자 제품 안전 위원회(CPSC)의 리콜 데이터베이스를 기반으로 multimodal dataset인 RECALL-MM을 개발했습니다. 이 데이터셋은 과거 정보에 기반한 데이터 주도적 위험 평가를 지원하며, 생성적 방법(generative methods)을 통해 확장됩니다. 데이터셋의 패턴은 개선된 안전 조치를 통해 큰 영향을 미칠 수 있는 특정 영역을 강조합니다.

- **Technical Details**: 연구에서는 2000년부터 2024년까지의 6,874개의 리콜 데이터를 수집하였고, 이를 대형 언어 모델(LLM)로 확장하여 위험 평가를 위한 새로운 분류 및 시각적 설명을 추가했습니다. 데이터셋의 각 항목은 위험 분류, 제품 카테고리, 치료 유형 등의 주요 리콜 속성을 포함하고 있으며, GPT-4o를 활용해 내용을 구조화했습니다. 연구는 또한 LLM을 이용해 제품 이미지만으로도 잠재적인 위험을 예측하는 방법론을 소개합니다.

- **Performance Highlights**: 사례 연구를 통해 리콜 데이터의 유틸리티를 증명하고 제품 위험을 식별하는 데 어떻게 기여하는지를 보여줍니다. 첫 두 가지 사례 연구는 설계자들이 리콜된 제품 간의 패턴을 시각화하고 새로운 제품 아이디어를 전반적인 리콜 환경 속에 위치시킬 수 있음을 보여줍니다. 마지막 사례 연구에서는 LLM을 활용하여 제품 이미지에 기반한 위험 예측의 강점과 한계를 강조하며, 설계 과정 전반에 걸친 위험 인식의 중요성을 부각시킵니다.



### Enhancing Knowledge Graph Completion with Entity Neighborhood and Relation Contex (https://arxiv.org/abs/2503.23205)
- **What's New**: 이 논문에서는 지식 그래프 완성(KGC) 문제를 해결하기 위해 KGC-ERC라는 새로운 프레임워크를 제안합니다. KGC-ERC는 생성적 언어 모델에 엔티티 이웃(entity neighborhood)과 관계(context of relation) 정보를 통합하여 예측 성능을 향상시킵니다. 또한, 입력 토큰 제한 속에서 중요한 컨텍스트를 효율적으로 선택하기 위한 샘플링 전략을 소개하여 전체 환경 정보를 최적화합니다.

- **Technical Details**: KGC-ERC는 KGT5-context를 기반으로 하여 쿼리에 관계 컨텍스트를 추가하고 이를 통해 입력 시퀀스를 풍부하게 합니다. 이 프레임워크의 핵심 요소인 선택자(Selector) 모듈은 지식 그래프에서 유의미한 엔티티 및 관계 컨텍스트를 샘플링하여 필터링합니다. 이 과정에서 엔티티 이웃 샘플링과 관계 컨텍스트 샘플링을 사용하는데, 이를 통해 최대한 의미 있는 정보를 선택하여 생성적 언어 모델에 전달합니다.

- **Performance Highlights**: Wikidata5M, Wiki27K, FB15K-237-N 데이터셋에 대한 실험 결과, KGC-ERC는 기존의 최신 기법들에 비해 예측 성능과 확장성 모두에서 우수한 결과를 보여주었습니다. 이 모델은 MRR과 Hits@k와 같은 평가 지표에서 뛰어난 성과를 달성하였으며, 복잡한 관계를 효과적으로 처리할 수 있는 능력을 입증했습니다.



### The Challenge of Achieving Attributability in Multilingual Table-to-Text Generation with Question-Answer Blueprints (https://arxiv.org/abs/2503.23204)
- **What's New**: 이 논문에서는 저자들이 낮은 자원의 언어인 아프리카 언어로 구성된 TaTA 데이터셋에 대한 멀티링구얼(멀티언어) Table-to-Text 생성 작업에서 Question-Answer(QA) 청사진을 사용하여 결과의 신뢰성을 증대시키는 방법을 탐구하고 있습니다. 또한 이 작업은 첫 번째로 QA 청사진을 적용하여 신뢰성을 개선하기 위한 새로운 방법을 제안합니다. 저자들은 영어 예시에 대해서는 QA 청사진이 결과의 신뢰성을 높이는 데 효과적이라는 것을 발견했으나, 멀티링구얼 환경에서는 효과가 떨어진다고 보고했습니다.

- **Technical Details**: 저자들은 Seq2Seq 모델을 활용하여 입력된 표의 정보를 기반으로 유창하고 정확한 설명을 생성하는 작업을 수행했습니다. 이 논문에서는 QA 청사진이 포함된 TaTA 데이터셋에서 만큼 Seq2Seq 언어 모델을 미세조정(finetuning)했습니다. 그러나 멀티링구얼 환경에서는 영어에서 타겟 언어로 QA 청사진을 번역하는 과정에서 발생하는 부정확성으로 인해 제약이 있다고 합니다.

- **Performance Highlights**: QA 청사진을 사용한 결과, 모델이 영어 데이터에 대해 학습 및 평가되었을 때는 결과의 신뢰성이 향상된 것으로 나타났습니다. 그러나 멀티링구얼 환경에서는 기대하는 성과를 내지 못하였으며, 이는 영어에서 다른 언어로 번역되는 과정에서 발생하는 오류와 모델이 생성한 청사진을 제대로 활용하지 못하는 문제 때문이라고 분석되었습니다. 이 논문은 전반적인 성능 평가에 대한 깊이 있는 분석을 제공하여 향후 연구에 기초 자료를 제공하고 있습니다.



### The realization of tones in spontaneous spoken Taiwan Mandarin: a corpus-based survey and theory-driven computational modeling (https://arxiv.org/abs/2503.23163)
- **What's New**: 최근 연구에서 의미(semantics)가 미세한 음성적 세부사항(phonetic detail)을 공동 결정할 수 있다는 사실이 부각되고 있지만, 특히 음높이(realization of pitch)와의 복잡한 상호작용은 상대적으로 덜 연구되었습니다. 본 연구는 대만 만다린의 자발적(spontaneous)의 말에서 발견된 두 개의 음조 조합에 대한 연구로, 20가지의 가능한 조합을 중점적으로 다룹니다.

- **Technical Details**: 우리는 Generalized Additive Mixed Models (GAMs)를 활용하여 f0 (기본 주파수) 곡선을 성별(gender), 음조 맥락(tonal context), 음조 패턴(tone pattern), 말하기 속도(speech rate), 단어 위치(word position), 빅그램 확률(bigram probability), 화자(speaker) 및 단어(word)의 여러 요소를 기준으로 모델링했습니다. 분석 결과 단어(word)와 의미(sense)는 f0 곡선의 중요한 예측 변수로 나타났으며, 이러한 효과는 일반적인 음조 패턴보다 더 큰 것으로 나타났습니다.

- **Performance Highlights**: 발음(manding)하는 단어의 음높이 곡선은 특정 맥락에서의 의미를 근사화하는 단어의 임베딩(embedding)을 기반으로 예측할 수 있다는 결론을 도출했습니다. 연구 결과, 맥락 내 의미와 음성적 표현이 일반적인 언어 이론이 예측하는 것보다 훨씬 더 얽혀 있으며, 이는 언어 기호의 임의성(arbitrariness of the linguistic sign)이라는 주요 가설에 도전하는 결과를 초래했습니다.



### Memory-Aware and Uncertainty-Guided Retrieval for Multi-Hop Question Answering (https://arxiv.org/abs/2503.23095)
- **What's New**: MIND(Memory-Informed & INteractive Dynamic RAG)는 멀티 홉 질문 응답의 주요 한계를 극복하기 위해 고안된 새로운 프레임워크입니다. 이는 정보 검색 시 동적인 언급 추출, 토큰 수준 불확실성을 기반으로 한 정보 검색 트리거링 및 높은 신뢰도를 가진 사실을 기억하는 방법론이 통합되어 있습니다. MIND는 주변 맥락을 유지하면서 누락된 정보를 수집하는 서브 쿼리 생성을 지원하여 복잡한 질문에 대한 유연한 응답을 제공합니다.

- **Technical Details**: MIND는 실시간 불확실성 신호를 기반으로 검색 결정을 적응적으로 내리는 Retrieval-Integrated Neural Decision-making(RIND) 방법론을 채택합니다. 이를 통해 MIND는 토큰 수준의 엔트로피와 주의(attention) 패턴을 모니터링하여 추가 검색이 필요한 시점을 결정합니다. 또한, 메모리 저장소를 통해 이전 단계에서 검색된 엔티티가 여러 추론 단계 동안 액세스 가능하도록 보장하여 일관된 멀티 홉 생성을 지원합니다.

- **Performance Highlights**: MIND는 HotpotQA, 2WikiMultihopQA, StrategyQA, IIRC와 같은 다양한 멀티 홉 QA 데이터셋에서 평가되었습니다. 실험 결과, MIND는 불필요한 검색 호출을 크게 줄이면서도 F1 점수와 Exact Match(EM) 지표를 통해 답변 품질을 향상시켰습니다. 다양한 필터링 모드가 검색 효율성과 정확성에 미치는 영향을 분석함으로써 멀티 홉 추론의 효율성과 정확성을 균형 있게 조절할 수 있는 인사이트를 제공합니다.



### Parsing Through Boundaries in Chinese Word Segmentation (https://arxiv.org/abs/2503.23091)
Comments:
          Submitted to ACL2025 System Demonstration

- **What's New**: 이번 연구는 중국어 단어 분할(Chinese word segmentation)과 구문 분석(syntactic parsing) 간의 복잡한 관계를 조명하며, 다양한 단어 경계(word boundary) 방법이 중국어의 의존 구조(dependency structures)에 미치는 영향을 분석합니다. 또한, 연구진은 사용자들이 서로 다른 분할 방법에 따른 파싱 결과를 비교할 수 있도록 인터랙티브 웹 기반 시각화 도구를 개발했습니다. 이를 통해, 중국어 NLP 분야에서 구문 분석의 명확한 이해를 도모하고자 합니다.

- **Technical Details**: 중국어는 모폴로지(morphology)에서 고유하게 단어와 형태소(morpheme) 간의 구별이 중요합니다. 연구자들은 중국어 GSD 트리뱅크를 사용하여, 여러 단어 경계 세분화 방식이 의존 관계에 미치는 영향을 조사했습니다. 각 세분화 전략은 언어적 가정과 계산적 방법론에서 차이를 보이며, 이는 결과적으로 구문 분석의 유형과 이의 구조에 영향을 미칩니다.

- **Performance Highlights**: 연구 결과, 단어 분할 전략에 따라 의존 트리의 구조가 달라짐을 확인했습니다. 예를 들어, ‘upward’라는 단어를 하나의 부사적 단위로 보느냐, 아니면 구성 형태소로 나누느냐에 따라 의존 구조가 변화하게 됩니다. 이와 같은 분석은 향후 NLP 모델 디버깅 및 교육적 응용에 긍정적인 영향을 미칠 것으로 기대됩니다.



### UNITYAI-GUARD: Pioneering Toxicity Detection Across Low-Resource Indian Languages (https://arxiv.org/abs/2503.23088)
- **What's New**: UnityAI-Guard는 저자원이 운영되는 인도 언어를 대상으로 한 이진 독성 분류 프레임워크로, 독성이 있는 콘텐츠를 식별하는 혁신적인 모델을 제공합니다. 기존의 시스템은 주로 자원이 풍부한 언어에 맞춰져 있었으나, UnityAI-Guard는 이 중요한 격차를 해소하기 위해 개발되었습니다. 이 프레임워크는 다양한 Brahmic/Indic 스크립트를 지원하며 연평균 F1 점수 84.23%를 달성했습니다.

- **Technical Details**: UnityAI-Guard는 888,000개의 훈련 인스턴스와 35,000개의 수동 검증 테스트 인스턴스를 활용하여 불균형 문제를 해결하고, 여러 언어에 대한 독성 분류 모델을 개발하였습니다. 자동 자막 및 API를 통해 다양한 기능을 제공하며, 전반적인 아키텍처는 효율성과 사용자 친화성을 고려하여 설계되었습니다. 모집단의 다양성과 정합성을 고려하여 두 명의 원어민이 샘플을 검토하여 데이터의 신뢰성을 높였습니다.

- **Performance Highlights**: UnityAI-Guard는 세 가지 크기의 모델을 사용하여 실험을 수행하였으며, 가장 큰 모델인 aya-expanse-8B가 87.21%의 정확도를 기록함으로써 독성 감지에서 더 나은 성능을 보여주었습니다. 실험 결과, 모델 크기가 커짐에 따라 F1 점수가 일관되게 향상되었으며, 특히 다양한 인도 스크립트를 처리하는 데 있어 큰 모델들이 뛰어난 능력을 보였습니다.



### The Reasoning-Memorization Interplay in Language Models Is Mediated by a Single Direction (https://arxiv.org/abs/2503.23084)
- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)의 추론 및 기억 동학을 이해하기 위해 기계적 접근 방식을 제안합니다. 연구팀은 모델의 잔여 흐름(residual stream)에서 특정 선형 특성(Linear Features)을 찾아 이를 통해 모델의 추론과 기억 간의 균형을 조절할 수 있음을 보여주었습니다. 이러한 특성을 통해 LLMs의 추론 능력을 보다 효과적으로 활성화할 수 있는 방법이 제시되어, 더 견고하고 해석 가능한 생성 AI 시스템 개발의 기초가 될 것입니다.

- **Technical Details**: 연구자들은 LLM의 메모리 사용을 일반화능력(generalizability) 부족으로 정의하며, 이를 평가하기 위해 합성 추론 벤치마크를 설계했습니다. 조사 결과, 특정 선형 추론 특성(Linear Reasoning Features, LiReFs)이 모델의 활성화 공간 내에서 모델의 일반화 능력을 지배하며, 이를 통해 문제 해결을 위한 추론 능력을 향상시킬 수 있는 잠재력이 있음을 입증했습니다. 이 특성을 조작함으로써 모델이 강력한 일반화 기능을 발휘하도록 유도할 수 있습니다.

- **Performance Highlights**: 연구의 주요 결과는 네 가지 서로 다른 LLMs와 여섯 개 데이터셋에서 실험을 통해 확인되었습니다. LiReFs를 활용하면 모델의 추론 오류를 줄이고, 더 적절한 문제 해결 기능을 활성화하여 성능을 향상시킬 수 있습니다. 이 연구 결과는 LLMs의 추론 능력이 선형 특성에 의해 매개됨을 입증하며, 다양한 지식 영역과 언어에서의 모델 성능을 효과적으로 조절하는 새로운 기제를 제시합니다.



### EventWeave: A Dynamic Framework for Capturing Core and Supporting Events in Dialogue Systems (https://arxiv.org/abs/2503.23078)
- **What's New**: 기존의 대규모 언어 모델(LLMs)은 대화 시스템에서 주목할 만한 발전을 이루었으나, 다회전(interaction) 대화에서의 사건(event)의 역할을 간과하여 문맥 추적이 불완전한 문제가 발생하고 있습니다. 이 문제를 해결하기 위해 저자들은 핵심 사건(core event)과 보조 사건(supporting event)을 식별하고 업데이트하는 이벤트 중심 프레임워크인 EventWeave를 제안합니다. 이벤트 그래프를 통해 이들 사건의 상호작용을 표현하고 대화에서 발생하는 중요한 사건에 집중할 수 있도록 하여 명확하고 일관된 응답을 생성하는 것을 목표로 합니다.

- **Technical Details**: EventWeave는 동적 이벤트 그래프(dynamic event graph)를 구축하여 대화의 흐름에 따라 새로운 사건과 기존의 사건을 지속적으로 통합합니다. 각 대화 턴(turn)에서 사용자의 발화로부터 나오는 사건을 식별하고 의미적으로 연결된 노드로 클러스터(cluster)하여 중요한 세부 사항을 고려합니다. 이를 통해 대화의 초점을 맞추고 전체 대화 이력을 반복적으로 처리할 필요성을 줄이며 정보 과다(overload)나 충돌(conflict)을 피할 수 있습니다.

- **Performance Highlights**: 실험 결과, EventWeave는 Conversation Chronicle CC와 Multi-Session Conversations MSC의 두 가지 벤치마크 데이터셋에서 응답 품질(response quality)과 사건 중심 평가(measures) 모두에서 최신 모델들을 초월하는 성과를 보였습니다. EventWeave는 앞으로의 대화 모델에 매끄럽게 통합될 수 있는 간단하지만 강력한 메커니즘을 제공하여 대화 시스템에서 사건 중심의 응답(event-centered response) 생성에 기여할 것으로 기대됩니다.



### Efficient Inference for Large Reasoning Models: A Survey (https://arxiv.org/abs/2503.23077)
- **What's New**: 본 논문은 대규모 추론 모델(LRMs)을 위한 효율적인 추론 방법을 종합적으로 검토하며, 토큰 비효율성을 완화하는 데 집중하고 있습니다. 특히, 두 가지 주요 범주인 명시적 컴팩트 체인 오브 사고(Chain of Thought, CoT)와 암묵적 잠재적 CoT를 소개합니다. 이 연구는 최신 LRMs의 효율성을 높이는 데 필요한 기술적 인사이트를 제공합니다.

- **Technical Details**: LRMs는 복잡한 문제를 해결하기 위해 명시적인 중간 토큰을 사용하여 구조화된 사고 과정을 구현합니다. 그러나 LRMs는 많은 수의 중간 사고 토큰을 생성해야 하므로 자원 소모가 많고 효율성을 저하시키는 문제를 가지고 있습니다. 따라서 이 논문에서는 명시적 컴팩트 CoT와 암묵적 잠재적 CoT라는 두 가지 방법을 제안하며, 각 방법의 장단점을 분석합니다.

- **Performance Highlights**: 논문에서는 CoT 압축, 컴팩트 사고를 위한 파인튜닝, 보상 기반 인센티브와 같은 다양한 기술을 활용하여 정확성을 유지하면서 더 간결한 추론 경로를 만들기 위한 연구를 다룹니다. 또한, 유망한 기술 솔루션으로 모델 병합, 새로운 아키텍처 및 에이전트 라우터를 제안하고, LRMs의 효율적인 추론을 향상시키기 위한 기술적 통찰을 강조합니다.



### A Training-free LLM Framework with Interaction between Contextually Related Subtasks in Solving Complex Tasks (https://arxiv.org/abs/2503.23053)
- **What's New**: 본 논문에서는 대형 언어 모델(LLM)들이 복잡한 작업을 해결하는 데 뛰어난 능력을 보여준 것을 기반으로, 작업을 독립적인 하위 작업(subtask)으로 분해하는 방법을 제안합니다. 또한, 하위 작업 간의 정보 손실 문제를 해결하기 위해 상호작용(interaction) 메커니즘을 도입하고, 훈련 없이도 이를 구현할 수 있는 프레임워크를 제시합니다.

- **Technical Details**: 제안된 프레임워크는 상호작용 요청을 통해 하위 작업이 완료된 작업에서 특정 정보를 조회하거나 특정 작업을 트리거할 수 있도록 합니다. 이를 위해 하위 작업 궤적 메모리(subtask trajectory memory)를 도입하여 상호작용 요청을 받을 때 완료된 하위 작업을 재개할 수 있게 하였습니다. 또한, 실행 과정 중 발생하는 새로운 작업(action)을 통해 하위 작업의 실행 과정 및 결과에 대한 간결하고 정확한 설명을 생성합니다.

- **Performance Highlights**: 우리는 제안한 프레임워크를 인터랙티브 의사결정 작업(WebShop)과 다중 홉 질문 응답(HotpotQA)에서 평가했습니다. GPT-3.5와 GPT-4를 사용하여 비교한 결과, 우리 프레임워크가 기존의 최첨단 훈련 없는 기준 모델들보다 우수한 성능을 보였습니다.



### A Retrieval-Augmented Knowledge Mining Method with Deep Thinking LLMs for Biomedical Research and Clinical Suppor (https://arxiv.org/abs/2503.23029)
- **What's New**: 이 논문에서는 대규모 생의학 기사를 기반으로 생의학 지식 그래프(BioStrataKG)를 구축하고 다문서 질문-응답 데이터셋(BioCDQA)을 생성하는 파이프라인을 제안합니다. 제안된 통합 및 점진적 검색 증강 추론(IP-RAR) 프레임워크는 정보를 효과적으로 검색하고 지식을 향상시킵니다. IP-RAR은 문서 검색의 F1 점수를 20% 개선하고 답변 생성의 정확도를 25% 향상시키는 성과를 보여줍니다.

- **Technical Details**: 생의학 지식 그래프(BioStrataKG)는 LLMs과 결합된 두 가지 레이어의 문서-개체 이중 표현 융합 아키텍처에서 구축됩니다. 이 프로세스는 개체 관계 추출 및 기사의 연구 방법 및 분야와 같은 의미 정보의 구조적 표현을 포함하여 세분화된 지식 추출을 활용합니다. BioCDQA 데이터셋은 비구조적 텍스트 및 구조적 지식 그래프의 데이터를 통합하여 설계되었으며, 질문, 답변, 출처 논문 등을 포함한 총 1,183 개의 질문-답변 쌍을 포함합니다.

- **Performance Highlights**: IP-RAR 프레임워크는 깊은 사고 능력을 갖춘 LLMs를 활용하여 정밀한 추론과 지식 통합을 가능하게 합니다. 이 프레임워크는 약물 상호작용 분석, 정밀 의학, 지식 이전 등과 같은 다양한 생의학 연구 응용 프로그램에 유용한 기술적 지원을 제공합니다. 실험적으로, BioStrataKG는 문서 검색의 효율성을 크게 향상시키며 연구원들이 연구 전략 수립 및 의사 결정을 가속화하는 데 기여합니다.



### S2MoE: Robust Sparse Mixture of Experts via Stochastic Learning (https://arxiv.org/abs/2503.23007)
Comments:
          4 pages

- **What's New**: 이번 연구에서는 Robust Sparse Mixture of Experts via Stochastic Learning (S2MoE)이라는 새로운 접근 방식을 제안합니다. S2MoE는 결정론적(deterministic) 및 비결정론적(nondeterministic) 입력 모두에서 학습하도록 설계된 전문가 조합입니다. 기존의 SMoE의 한계를 해결하기 위해, 입력에 Gaussian noise를 추가하여 전문가 선택 이전의 특성 학습을 향상시킵니다.

- **Technical Details**: S2MoE는 기본적인 입력과 노이즈가 추가된 입력의 두 가지 부분으로 구성되어 있습니다. 이 방법은 전문가의 출력을 가중합(weighted sum)하여 최종 결과를 도출하는 Sparse Mixture of Experts (SMoE) 아키텍처를 활용합니다. 불확실성 손실(uncertainty loss)을 도입하여 노이즈 생성 과정의 품질을 조절합니다.

- **Performance Highlights**: 광범위한 NLP 작업에 대한 실험 결과, S2MoE는 기존의 라우팅 방법들과 비슷한 성능을 보이면서도 계산 추론 비용을 28% 감소시켰습니다. 이는 S2MoE가 입력에 대해 단일 전문가만 활성화하여 다른 방법들과 유사한 성능을 달성할 수 있음을 의미합니다. 이로 인해 실제 응용 프로그램에 LLM을 배포하는 효율성이 크게 향상됩니다.



### Sparse Mixture of Experts as Unified Competitive Learning (https://arxiv.org/abs/2503.22996)
Comments:
          18 pages

- **What's New**: 이 논문에서는 Sparse Mixture of Experts (SMoE)의 일반화 능력에 대한 질문을 탐구하며, 기존 두 가지 형태인 Token Choice와 Expert Choice의 한계를 지적합니다. 이를 통해 Token Choice가 관련 없는 전문가에 너무 집중하는 경향이 있고, Expert Choice는 중요한 토큰을 버리는 위험이 있음을 발견했습니다. 이러한 분석을 바탕으로, 변형된 SMoE인 Unified Competitive Learning SMoE (USMoE)를 제안합니다.

- **Technical Details**: USMoE는 두 가지 주요 구성 요소, 즉 Unified Competitive Score와 Unified Competitive Mechanism을 포함하여 기존 SMoE의 한계를 극복하기 위해 고안된 새로운 프레임워크입니다. Unified Competitive Score는 전문가_selection을 균형 있게 조정하며, Unified Competitive Mechanism은 효과적이고 효율적인 전문가 할당을 보장하는 구조화된 라우팅 전략을 제공합니다. 이는 경쟁 학습(Competitive Learning)의 관점을 통해 이루어지며, 각 전문가와 토큰 간의 최적 매칭을 지원합니다.

- **Performance Highlights**: 다양한 작업을 통해 USMoE는 전통적인 접근법에 비해 최대 10% 성능 개선을 이루거나, 계산 추론 비용을 14% 줄였습니다. 특히, 심화된 입력 이해가 요구되는 작업에서 두드러진 성과를 보여주며, 세부적인 분석을 통해 한국어 처리 및 분류 작업에서 강한 성능을 발휘하였습니다.



### FReM: A Flexible Reasoning Mechanism for Balancing Quick and Slow Thinking in Long-Context Question Answering (https://arxiv.org/abs/2503.22985)
- **What's New**: 최근 논문에서 소개된 FReM(Flexible Reasoning Mechanism)은 LCQA(롱 컨텍스트 질문 응답) 시스템의 비효율성을 개선하기 위한 혁신적인 접근 방식을 제안합니다. FReM은 각 질문의 복잡성에 따라 추론의 깊이를 조정하는 방식을 통해 빠르고 느린 사고 모드의 한계를 극복하려고 합니다. 이 방법론은 간단한 질문에 대해서는 효율적인 처리를, 복잡한 질문에 대해서는 심층적인 사고를 가능하게 합니다.

- **Technical Details**: FReM의 핵심은 질문을 구성 요소로 분해하고 그러한 요소에 맞는 유사 질문-답변(QA) 예제를 생성하여 명확한 사고 체인을 제공하는 것입니다. 초기 질문을 분해한 후, FReM은 여러 QA 참조 예제를 통합하여 각 질문의 복잡성에 맞는 적절한 사고 체인을 선택합니다. 이러한 메커니즘은 명시적인 사고 경로 생성을 통해 신속한 패턴 매칭에 의존하기보다는 더 깊이 있는 논리를 보장합니다.

- **Performance Highlights**: 실험 결과 FReM은 7개의 QA 데이터 셋에서도 향상된 추론 정확도를 보여주었으며, 특히 복잡한 멀티홉 질문에 대한 처리에 있어서 그 효과가 두드러졌습니다. 이로 인해 FReM은 LCQA 방법론의 발전에 기여할 잠재력을 가지고 있다는 점에서 주목받고 있습니다. FReM을 통해 시간 소모를 최소화하고 질문의 다양성에 맞춘 유연한 추론이 가능해졌습니다.



### XL-Instruct: Synthetic Data for Cross-Lingual Open-Ended Generation (https://arxiv.org/abs/2503.22973)
- **What's New**: XL-AlpacaEval은 대형 언어 모델(LLM)의 다국어 생성 성능을 평가하기 위한 새로운 벤치마크로 소개됩니다. 기존의 연구들에서 공통적으로 나타난 문제를 해결하기 위해 고품질의 합성 데이터를 생성하는 XL-Instruct 방법이 제안되었습니다. 이 방법을 활용한 세밀한 튜닝을 통해 모델의 성능이 크게 향상되었으며, 구체적으로 GPT-4o-Mini 대비 win rate가 7.4%에서 21.5%로 증가한 것으로 나타났습니다.

- **Technical Details**: 교차 언어 생성(cross-lingual generation)은 특정 언어로 된 질의를 이해하고 다른 언어로 응답을 생성하는 작업입니다. 연구자들은 기계 번역(MT)의 노이즈 문제와 정보 손실 문제를 지적하며, XL-Instruct라는 합성 데이터 생성 기술을 활용하여 고품질의 교차 언어 데이터를 대규모로 생성할 수 있음을 보여주었습니다. 이 방법이 적용된 모델은 영어나 다국어 생성 작업에서도 강력한 제로샷 전이(zero-shot transfer) 성능을 보였습니다.

- **Performance Highlights**: XL-Instruct의 활용으로 다양한 LLM의 교차 언어 성능이 일관되게 개선되었으며, 성능 평가에서 LLM의 'off-the-shelf' 성능이 낮음을 확인했습니다. 실험 결과, 성능 향상뿐만 아니라 다국어 후처리 단계에 XL-Instruct를 포함할 것을 강력히 추천합니다. 또한, XL-AlpacaEval 벤치마크와 XL-Instruct 데이터셋이 향후 교차 언어 LLM 연구에 기여할 것으로 기대됩니다.



### Can LLMs Support Medical Knowledge Imputation? An Evaluation-Based Perspectiv (https://arxiv.org/abs/2503.22954)
Comments:
          10 pages, 3 figures, AMIA

- **What's New**: 이번 연구는 Medical Knowledge Graphs (KGs)의 불완전성을 해결하기 위해 Large Language Models (LLMs)를 활용하는 새로운 접근법을 제시합니다. 특히 LLM이 생성한 치료 매핑을 체계적으로 평가하여 신뢰성을 검증했습니다. 또한, LLM 사용 시 발생할 수 있는 위험 요소와 기존 임상 지침과의 불일치 문제에 대해 심층적으로 분석했습니다.

- **Technical Details**: 연구에서는 LLM이 결여된 치료 관계를 보완하기 위해 ICD, Mondo, ATC와 같은 의료 코딩 시스템을 활용했습니다. LLM은 임상 문헌과 약물 정보를 바탕으로 질병과 치료 간의 관계를 생성하고, 기존의 KGs와 비교할 수 있는 평가 프레임워크를 개발했습니다. 이 프레임워크는 LLM이 생성한 출력과 임상적으로 승인된 기준 간의 정합성을 분석하여 LLM의 강점과 한계를 파악합니다.

- **Performance Highlights**: 연구 결과 LLM은 일부 유용한 치료 제안을 생성할 수 있는 반면, 일관성 결여 및 오류 생성 가능성도 드러났습니다. 이러한 발견은 의료 분야에서 LLM의 통합이 신중해야 함을 강조하며, LLM의 안전한 활용을 위해 하이브리드 접근법의 필요성을 규명합니다. 전반적으로, 이 연구는 의료 KG의 개선을 위한 LLM 활용에 대한 경고를 제시하고, 투명한 검증의 중요성을 강조합니다.



### SUV: Scalable Large Language Model Copyright Compliance with Regularized Selective Unlearning (https://arxiv.org/abs/2503.22948)
- **What's New**: 이번 연구에서는 SUV (Selective Unlearning for Verbatim data)라는 새로운 선택적 학습 해제 프레임워크를 소개합니다. 이 프레임워크는 LLM이 저작권이 있는 콘텐츠를 기억하는 것을 방지하면서도 모델의 전체 유틸리티를 유지하도록 설계되었습니다. 저작권 침해 사례를 포착한 데이터셋을 구축하고, 직접 선호 최적화(Direct Preference Optimization)을 사용하여 기억된 콘텐츠를 대체하는 방안을 제시합니다.

- **Technical Details**: SUV 프레임워크는 전통적인 방법과 달리 슬라이딩 윈도우(sliding-window) 메커니즘을 사용하여 기억된 구간을 세분화하여 식별합니다. 또한, DPO를 통해 플라거리(표절)된 내용을 제거하고 무작위로 생성된 텍스트로 대체합니다. 이 과정에서 그라디언트 프로젝션(gradient projection)과 피셔 정보 정규화(Fisher information regularization)를 통합하여 모델의 성능 저하를 최소화합니다.

- **Performance Highlights**: 500개의 저명한 책으로 구성된 대규모 데이터셋을 사용하여 SUV의 성능을 검증하였으며, 저작권이 있는 콘텐츠의 기억을 크게 줄이면서도 관련 없는 작업에서의 성능에 미치는 영향은 미미하다는 사실을 입증했습니다. 우리의 접근 방식은 공개 기준에서도 우수한 성과를 보여주며, 기존의 방법들에 비해 유용성을 유지하면서 저작권 위험 완화에 효과적임을 강조합니다.



### Resona: Improving Context Copying in Linear Recurrence Models with Retrieva (https://arxiv.org/abs/2503.22913)
- **What's New**: 이번 연구에서는 저자들이 Resona라는 새로운 프레임워크를 소개합니다. Resona는 Linear Recurrent Models (LRMs)을 향상시키기 위해 Retrieval 기능을 통합하여 다양한 작업 요구사항에 맞는 맞춤형 동작을 가능하게 합니다. 이를 통해 LRMs가 과거의 정보를 기억하며 학습할 수 있는 능력이 향상됩니다.

- **Technical Details**: Resona는 입력 컨텍스트에서 검색된 정보를 통합하는 메커니즘으로, 고정 크기의 상태에서 발생하는 정보 병목 현상을 극복합니다. 이 과정에서 특정 LRM 층에 Resona를 증강하는 방식으로, 컨텍스트를 조각으로 나누어 LRM 상태에 따라 검색합니다. 이후 검색된 컨텍스트는 LRM 출력에 혼합되어 정보 흐름을 개선합니다.

- **Performance Highlights**: 다양한 synthetic 및 real-world 자연어 작업에서 Resona는 LRM의 context-specific 정보 활용 능력을 상당히 향상시켰습니다. 결과적으로 Resona는 전반적인 성능 향상과 테스트 시 모델 커스터마이징을 가능하게 하여 효율성과 성능 간의 균형을 맞출 수 있도록 돕습니다.



### Understanding Inequality of LLM Fact-Checking over Geographic Regions with Agent and Retrieval models (https://arxiv.org/abs/2503.22877)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 사실 확인(fact-checking)에서의 성능을 다양한 지역과 시나리오를 통해 평가하였습니다. 특히, LLM의 성능이 지역에 따라 다르게 나타나는 점을 강조하여, 이를 통해 오정보의 확산을 방지하는 방법을 모색하였습니다.

- **Technical Details**: 연구는 600개의 사실 확인이 이루어진 진술(statement)을 포함한 데이터셋을 사용하여 세 가지 실험 세팅을 평가하였습니다: (1) 진술만 있을 때, (2) 위키피디아 접근이 가능한 LLM 기반 에이전트를 사용할 때, (3) 공식 사실 확인이 제공되는 Retrieval-Augmented Generation (RAG) 시스템을 사용할 때입니다. 이러한 설정을 통해 각 모델의 성능 차이를 분석하였습니다.

- **Performance Highlights**: 연구 결과, GPT-4, Claude Sonnet 및 LLaMA를 포함한 어떤 LLM을 사용하더라도, 선진국(Global North) 출처의 진술이 개발도상국(Global South) 출처의 진술보다 상당히 더 높은 성능을 나타냈습니다. 이러한 차이는 위키피디아 에이전트 기반 시스템을 사용한 경우에 더욱 확대되었습니다. 이는 지역적 특성을 반영할 수 있는 자료의 균형 잡기와 강력한 검색 전략의 필요성을 강조하고 있습니다.



### Generating Synthetic Oracle Datasets to Analyze Noise Impact: A Study on Building Function Classification Using Tweets (https://arxiv.org/abs/2503.22856)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델)을 사용하여 Building Function Classification(BFC)을 위한 합성 오라클 데이터셋을 생성하는 방법을 제안합니다. 이는 정답이 명확히 라벨링되고 해당 건물과 의미적으로 관련된 트윗만 포함된 데이터셋으로, 실제 데이터에서 격리하기 어려운 노이즈의 영향을 체계적으로 조사할 수 있게 합니다. 이러한 접근은 트위터에서 지리적 휴리스틱을 통해 수집된 트윗의 학습 성능 저하 문제를 해결하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 생성된 데이터셋은 건물의 기능, 위치 및 원하는 트윗의 언어 배포에 대한 메타데이터를 기반으로 하여 LLM에게 프롬프트를 전달하여 작성됩니다. 세 가지 단계로 구성된 파이프라인은 메타데이터를 수집하고 정리한 후, LLM을 사용하여 맥락적으로 관련된 트윗을 생성하는 과정을 포함합니다. 각 트윗은 OSM(OpenStreetMap)과 같은 외부 데이터베이스로부터 수집된 정보를 기준으로 하여 고유한 구조와 주제에 맞는 트윗을 생성하도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, 실제 트윗의 노이즈가 mBERT의 맥락 학습 능력을 상당히 저하시켰으며, 단순 키워드 기반 모델의 성능에 가까운 결과를 보였습니다. 반면, 합성 데이터셋을 사용한 mBERT 모델은 Naive Bayes 모델에 비해 현저히 우수한 성능을 보여주었습니다. 이러한 결과는 기능 노이즈를 처리하는 것이 모델의 복잡성을 증가시키는 것보다 더 중요하다는 점을 강조합니다.



### Learning to Reason for Long-Form Story Generation (https://arxiv.org/abs/2503.22828)
- **What's New**: 이 논문에서는 장편 소설 생성을 위한 새로운 과제인 Next-Chapter Prediction(NCP)을 제안하고, 비확인 보상(Verified Rewards)을 통해 확률 개선(Completion Likelihood Improvement) 방식의 보상 모형을 도입합니다. 이는 정형화된 데이터셋을 바탕으로 이야기를 생성하는 데 있어 합리적 추론을 가능하게 하여, 작가와 유사한 방식으로 작동하도록 돕습니다. 본 연구는 소설 생성 작업의 다양한 장기적 문제를 극복하기 위한 혁신적인 접근을 나타냅니다.

- **Technical Details**: Next-Chapter Prediction(NCP)은 주어진 이야기 정보의 요약에 기반하여 다음 장을 예측하는 방식으로 모델링됩니다. Verifiable Rewards via Completion Likelihood Improvement(VR-CLI)는 다음 장의 예측 확률을 향상시키는 보상 모형으로 설정되며, 이는 토큰 단위의 perplexity를 통해 자연스럽게 표현됩니다. 이 과정을 통해 저자 정보에 대한 추론을 학습하고, 장 생성에 필요한 세부 계획을 작성합니다.

- **Performance Highlights**: 실험 결과, 논문에서 제안한 모델은 기존의 근거 기반(non-reasoning) 및 감독 학습(supervised finetuning) 기준선보다 뛰어난 성능을 보였습니다. 특히, Sci-fi 및 Fantasy 장르에서 훈련된 모델이 생성한 장의 품질은 대부분의 평가 기준에서 선호되는 것으로 나타났습니다. 이러한 결과는 NCP와 VR-CLI의 효과성을 분명하게 보여줍니다.



### Boosting Large Language Models with Mask Fine-Tuning (https://arxiv.org/abs/2503.22764)
- **What's New**: 이 논문에서는 Mask Fine-Tuning (MFT)이라는 새로운 LLM 파인튜닝 패러다임을 소개합니다. MFT는 모델의 무결성을 의도적으로 파괴함으로써 놀라운 성능 향상을 이끌 수 있음을 보여줍니다. 이 연구는 LLM 파인튜닝의 기존 프로토콜을 통해 구조적 무결성이 반드시 필요하지 않음을 주장합니다.

- **Technical Details**: MFT는 사전 훈련된 LLM에 대해 작동되며, 이 LLM은 전체 파인튜닝(Fine-Tuning)을 통해 훈련됩니다. 여기서 MFT는 이 훈련된 LLM에 이진 마스크를 추가하여 특정 파라미터를 선택적으로 마스킹합니다. 최적화는 스트레이트-스루 그래디언트 추정기(gradient estimator)를 활용하여 이루어지며, 이는 배포형 학습(supervised learning)으로 안내됩니다.

- **Performance Highlights**: MFT는 다양한 도메인과 백본을 통해 일관된 성능 향상을 보였습니다. 예를 들어, LLaMA2-7B와 LLaMA3.1-8B 모델을 사용하여 각각 1.95% 및 1.88%의 평균 성능 향상을 기록했습니다. 이 연구는 기존의 파인튜닝 방식과 비교하여 성능을 크게 향상시킬 수 있는 새로운 관점을 제공합니다.



### Susceptibility of Large Language Models to User-Driven Factors in Medical Queries (https://arxiv.org/abs/2503.22746)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 의료 분야에서 어떻게 사용되고 있는지 살펴보았으며, 사용자 질문의 phrasing(문구)와 임상 정보의 완전성이 진단 정확도에 미치는 영향을 조사했습니다. 특히, 잘못된 정보의 framing(프레이밍), 출처의 권위, 모델의 persona(페르소나), 그리고 주요 임상 세부 정보의 생략이 LLM의 출력 신뢰성에 미치는 영향을 분석했습니다.

- **Technical Details**: 연구는 perturbation test와 ablation test라는 두 가지 실험을 통해 진행되었습니다. perturbation test에서는 다양한 주장 강도를 가진 잘못된 외부 의견을 도입하였고, ablation test에서는 특정 환자 정보 범주를 제거하며 모델의 성능을 평가했습니다. 이를 위해 MedQA와 Medbullets와 같은 공개 데이터셋을 사용해 GPT-4o, Claude 3.5 Sonnet 등과 같은 여러 상용 및 오픈 소스 모델을 비교했습니다.

- **Performance Highlights**: 모든 모델은 사용자 주도의 잘못된 정보에 취약했으며, 특히 상용 모델은 확정적이고 권위 있는 언어의 영향을 가장 많이 받았습니다. assertive tone(주장적인 어조)은 정확도에 가장 큰 부정적인 영향을 미쳤으며, ablation test에서는 신체 검사 결과와 실험실 결과의 생략이 성능 하락의 가장 큰 원인이 되었습니다. 결국, 연구 결과는 잘 구조화된 프롬프트와 완전한 임상 맥락의 필요성을 강조하며, 사용자들이 복잡한 사례에서 권위 있는 잘못된 정보 프레이밍을 피하고 전체 임상 정보를 제공해야 함을 시사합니다.



### A Large-Scale Vision-Language Dataset Derived from Open Scientific Literature to Advance Biomedical Generalist AI (https://arxiv.org/abs/2503.22727)
- **What's New**: 이번 논문에서는 Biomedica라는 오픈 소스 데이터셋을 소개합니다. Biomedica는 600만 개의 과학 기사와 2400만 개의 이미지-텍스트 쌍을 포함하고 있으며, 이를 통해 생물의학 인공지능(AI) 시스템의 성능을 향상시키는 것이 목표입니다. 웹 서버를 통해 제공되는 확장 가능한 스트리밍 및 검색 API는 AI 시스템과의 통합을 용이하게 합니다.

- **Technical Details**: Biomedica 데이터셋은 다양한 분야의 생물의학 연구 문헌을 포함하여, 다양한 카테고리의 이미지를 수집합니다. 각 인스턴스는 기사 수치와 이미지 수준의 메타데이터를 포함하고 있으며, 세밀한 주석이 제공됩니다. 이 데이터셋은 대규모로 저장 및 관리하기 어려울 수 있으나, Hugging Face에서 호스팅되어 필요할 때마다 스트리밍할 수 있습니다.

- **Performance Highlights**: Biomedica 데이터셋을 활용하여 구축된 다양한 AI 모델은 이전 시스템을 초과하는 성능을 보여주었습니다. 특히, BMC-CLIP과 BMC-SmolVLM 모델은 각각의 작업에서 이전의 모델들과 비교하여 성능이 크게 향상되었습니다. BIOMEDICA Index는 AI 에이전트 시스템이 의료 지침 기반 질문에 답변할 수 있도록 하여, 실질적인 임상적 응용에 기여할 수 있는 가능성을 보여줍니다.



### TRIDIS: A Comprehensive Medieval and Early Modern Corpus for HTR and NER (https://arxiv.org/abs/2503.22714)
Comments:
          6 pages, 3 figures, 2 tables

- **What's New**: 이 논문에서는 TRIDIS(Tria Digita Scribunt)라는 오픈 소스의 중세 및 초기 현대 원고 코퍼스를 소개합니다. TRIDIS는 다양한 레거시 컬렉션을 통합하여 메타데이터 설명을 포함하고 있으며, 이전의 연구에서 일부 하위 집합이 활용되었지만, 이번에는 전체 코퍼스를 통합적으로 설명합니다. 새로운 전반적인 코퍼스 구성은 문학적 전통을 연구하는 데 필요한 언어적 측면을 고려하여 설계되었습니다.

- **Technical Details**: TRIDIS는 여러 개의 오픈 소스 하위 컬렉션을 결합하여 구성되며, 반영된 데이터는 공통된 스키마를 따른 일관된 구조로 포장됩니다. 또한, 연구에서는 고유한 Outlier-driven partition 전략을 제안하여 훈련 데이터와 테스트 데이터 간의 도메인 중첩 문제를 해결하고, 복잡한 레이아웃과 드문 어휘를 가진 예제를 테스트 세트로 정의하여 HTR 모델의 일반화 능력을 평가합니다. 이러한 접근 방식은 TrOCR 및 MiniCPM2.5와 같은 사전 훈련 모델을 사용하여 검증되었습니다.

- **Performance Highlights**: TRIDIS 코퍼스는 다체로운 문서 유형을 다루며, 복잡한 레이아웃과 큰 필기 변동성을 갖춘 자료에 중점을 둡니다. 초기 실험 결과는 outlier-driven 테스트 분할을 사용할 때 HTR 모델의 성능이 크게 저하된다는 것을 보여주며, 엄격한 평가 방법론의 중요성을 강조합니다. 이를 통해 HTR 훈련 및 평가에서 이전에 간과된 난제를 드러내고, 전통적인 HTR뿐만 아니라 후속 NLP 작업을 지원하는 리소스를 제공합니다.



### Fragile Mastery: Are Domain-Specific Trade-Offs Undermining On-Device Language Models? (https://arxiv.org/abs/2503.22698)
Comments:
          14 Pages, 5 figures

- **What's New**: 이 논문은 자원 제한적인 엣지 디바이스에서의 온디바이스 언어 모델(ODLMs)의 적용을 다루고 있습니다. 새로운 아키텍처인 Generalized Edge Model(GEM)은 특정 도메인 최적화와 교차 도메인 강인성 간의 균형을 제공합니다. 실험 결과에 따르면, 기존 최적화 기법은 특정 작업에서 perplexity를 18-25% 감소시켰지만, 일반적인 작업 성능이 큰 폭으로 감소(12-29%)하는 경향이 있습니다.

- **Technical Details**: GEM은 Sparse Cross-Attention Router(SCAR)를 활용하여 동적으로 계산 리소스를 할당하며, Raspberry Pi 4, Pixel 6, iPhone 13 등의 장치에서 100ms 미만의 지연으로 0.89의 교차 도메인 F1 정확도를 보여줍니다. 또한, Domain Specialization Index(DSI), Generalization Gap(GG), Cross-Domain Transfer Ratio(CDTR) 등의 새로운 측정 도구를 제안하여 모델 압축 강도와 취약성 간의 상관관계를 나타냅니다.

- **Performance Highlights**: GEM은 GPT-4 Lite에 비해 일반 작업 수준을 7% 향상시켰으며, 도메인 특정 성능의 균형도 유지하고 있습니다. 이 모델은 진단 질문에 대해 F1 점수 0.95를 기록했으며, 일반적인 쿼리에 대해서는 F1 점수가 0.40으로 하락하는 등 성능 차이를 보였습니다. 이와 같은 성과들은 에너지 효율성과 계산 비용 분석을 통해 구체적으로 수치로 나타나고 있습니다.



### RIG: Synergizing Reasoning and Imagination in End-to-End Generalist Policy (https://arxiv.org/abs/2503.24388)
- **What's New**: 이번 논문은 복잡한 오픈 월드 환경에서 작동하는 에이전트에 필요한 상상력(imagination)과 추론(reasoning)을 통합한 최초의 정책인 RIG를 소개합니다. 이전 연구에서는 이러한 능력이 분리된 모델로 구현되었지만, RIG는 데이터 파이프라인을 통해 두 가지 능력을 효과적으로 결합하여 학습의 효율성과 일반화 능력을 향상시킵니다. 또한, RIG는 추론과 미래 이미지를 생성하는 과정을 결합하여 환경의 동역학을 명확하게 모델링합니다.

- **Technical Details**: RIG는 오토회귀 Transformer를 통해 텍스트 추론, 저수준의 행동 제어 및 이미지 생성을 학습합니다. 초기 단계에서는 기존의 데이터에서 수집한 궤적(trajectory)을 바탕으로, 텍스트 추론이 포함된 궤적을 생성하여 RIG-basic을 훈련시키고, 이후에는 상상력을 적용하여 실패한 궤적을 수정하는 RIG-lookahead를 학습합니다. 이러한 접근 방식은 궤적의 예측된 이미지를 환경 상태로 활용하여 가상 궤적을 생성하고 이를 기반으로 추론하여 행동을 예측하는 구조를 제공합니다.

- **Performance Highlights**: RIG는 마인크래프트 환경에서 광범위한 실험을 통해 현재의 최첨단 성능을 크게 향상시켰습니다. 결과적으로 111시간의 비디오로 훈련함으로써 전작들에 비해 17배 더 높은 샘플 효율성을 보여주며, 다양한 환경 상호작용과 추론 중 미리보기 단계를 조정하여 견고성과 일반화 능력이 지속적으로 향상됨을 입증하였습니다.



### Exploring the Effect of Reinforcement Learning on Video Understanding: Insights from SEED-Bench-R1 (https://arxiv.org/abs/2503.24376)
Comments:
          Technical Report (In Progress); Code released at: this https URL

- **What's New**: 본 논문은 멀티모달 대규모 언어 모델(MLLMs)의 비디오 이해를 평가하기 위한 새로운 벤치마크인 SEED-Bench-R1을 제안합니다. SEED-Bench-R1은 복잡한 일상적인 계획 작업을 여러 선택 질문 형태로 포함하여, 정교한 인식(perception)과 논리적 추론(logical reasoning)을 요구합니다. 또한, 이 벤치마크는 세 가지 수준의 일반화(generalization) 시나리오를 통해 MLLMs의 포스트 트레이닝(post-training) 방법을 체계적으로 평가합니다.

- **Technical Details**: SEED-Bench-R1은 현실적인 일상 활동을 기반으로 한 비디오를 사용하여, 모델이 목표를 이해하고 긴 시간 동안 시각적인 진행을 추적하며, 복잡한 환경 관찰을 인지하고, 세계 지식을 사용하여 다음 행동을 추론할 수 있도록 설계되었습니다. 이 벤치마크는 교육 데이터셋을 기반으로 하며, 명확하게 검증 가능한 정답을 제공하여 일반화 능력을 철저히 평가할 수 있는 구조로 되어 있습니다. Qwen2-VL-Instruct-7B를 사용하여 RL 및 감독된 파인튜닝(SFT) 방법을 비교하여 RL이 데이터 효율성과 성능 면에서 우수하다는 것을 보여주었습니다.

- **Performance Highlights**: 실험 결과, RL은 특히 OOD(Out-of-Distribution) 시나리오에서 SFT를 능가하며, 비디오 이해의 일반적인 벤치마크에서도 높은 성과를 보였습니다. RL은 시각적 인식을 향상시키고 COT(Chain of Thought) 토큰을 동적으로 쿼리하도록 모델을 교육하는 데 효과적이었습니다. 그러나 모델이 때때로 중요한 시각적 단서를 무시하는 등, 몇 가지 한계점도 드러났고, 이는 향후 연구와 개선 방향 설정에 중요한 요소가 될 것입니다.



### Effectively Controlling Reasoning Models through Thinking Intervention (https://arxiv.org/abs/2503.24370)
- **What's New**: 이번 논문에서는 Reasoning-enhanced 대규모 언어 모델(LLM)들이 최종 답변을 생성하기 전에 중간 사고 단계를 명확히 생성함으로써 복잡한 문제 해결에서 우수한 성능을 보인다는 점을 강조합니다. 저자들은 Thinking Intervention이라는 새로운 패러다임을 제안하여 모델의 내부 사고 과정을 명확하게 안내하고, 이로 인해 모델 행동을 조정할 수 있는 새로운 기회를 제공합니다.

- **Technical Details**: Thinking Intervention은 모델이 전통적인 프롬프트 엔지니어링을 넘어서서 사고 과정 중 특정 토큰 시퀀스를 삽입하거나 수정하여 더 세밀하게 제어할 수 있도록 합니다. 이 방식은 모델 훈련이 필요하지 않으며, 실제 환경에서 최소한의 엔지니어링 노력으로 배치할 수 있습니다. 또한 기존의 모델 제어 기법들과 호환되며, 올해 문맥 및 작업에 따라 적응적으로 사고 단계를 삽입하거나 수정할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: Thinking Intervention은 다양한 작업에서 성능을 크게 향상시킵니다. IFEval, SEP, XSTest 및 SORRY-Bench에서의 평가 결과, 이 접근법은 지침 따르기 작업에서 최대 6.7%의 정확도 향상과, 지침 계층 문제 해결에서 15.4% 개선, 그리고 안전 프롬프트에 대한 거부율을 40.0%까지 증가시켰습니다. 전반적으로, 저자들은 이 접근법이 LLM의 추론 프로세스에 대한 더 정밀하고 투명한 제어를 가능하게 한다고 주장합니다.



### SQuat: Subspace-orthogonal KV Cache Quantization (https://arxiv.org/abs/2503.24358)
- **What's New**: 이번 논문에서는 SQuat(Subspace-orthogonal KV cache quantization)이라는 새로운 접근법을 소개합니다. SQuat는 기존의 KV 캐시 양자화 방법과는 달리, 쿼리 텐서들로 구성된 서브스페이스를 활용하여 과거 토큰의 키 텐서를 양자화하는 과정에서 양자화 오류가 주의 메커니즘에 미치는 영향을 최소화합니다. 이 방법은 모델 재학습이나 추가적인 데이터 수집 없이도 적용될 수 있으며, 이론적 토대를 기반으로 개발되었습니다.

- **Technical Details**: SQuat은 주어진 사용자 프롬프트의 모든 토큰에서 쿼리 텐서를 통해 작업 관련 서브스페이스를 먼저 구성합니다. 그런 다음, 각 토큰의 키 텐서를 양자화하면서, 양자화된 키 텐서와 원래 키 텐서의 차이가 이 서브스페이스에 대해 직교하도록 유지합니다. 이를 통해 중요한 과업 정보에 대한 양자화 오류의 영향을 줄이고, 최적의 업데이트 규칙을 통한 효율적인 연산이 가능합니다.

- **Performance Highlights**: SQuat는 Llama-2-7B 모델을 기반으로 할 때, 피크 메모리 사용량을 2.17배에서 2.82배까지 줄일 수 있으며, 처리량은 2.45배에서 3.60배까지 향상됩니다. 또한, 이 방법은 기존의 다른 비튜닝(baseline) 방법들에 비해 더욱 우수한 성능을 발휘하며, 14개의 다양한 벤치마크 과제를 포함한 다양한 평가에서 그 효율성을 입증하였습니다.



### ORAL: Prompting Your Large-Scale LoRAs via Conditional Recurrent Diffusion (https://arxiv.org/abs/2503.24354)
- **What's New**: 본 논문에서는 기존의 방법들이 직면한 확장성과 조정 가능성의 한계를 해결하기 위해 새로운 조건부 반복 확산(conditional recurrent diffusion) 프레임워크인 ORAL을 소개합니다. ORAL은 고유한 조건화 메커니즘을 포함하여 모델 아키텍처와 작업 사양을 통합하여, 진화하는 기초 모델을 통해 효율적으로 전이 가능한 LoRA 파라미터를 생성할 수 있습니다. 이 접근법은 수십억 개의 파라미터를 가진 대형 언어 모델에서도 조정 가능성을 유지하면서 확장을 성공적으로 수행합니다.

- **Technical Details**: ORAL의 주요 기여는 LoRA 파라미터의 유연한 생성을 위한 새로운 조건화 메커니즘을 개발한 것입니다. 이 메커니즘은 모델 아키텍처 및 텍스트 기반 작업 사양을 입력으로 사용하여, 특정 다운스트림 작업에 맞춤화된 LoRA 파라미터를 생성할 수 있게 합니다. ORAL은 기존의 반복 확산 아키텍처를 기반으로 하여, 자원 집약적인 재교육 없이도 진화하는 기초 모델에 생성된 파라미터를 원활하게 전이할 수 있는 새로운 조건부 파라미터 생성 파이프라인을 제안합니다.

- **Performance Highlights**: 다양한 실험을 통해 ORAL은 7개의 언어 작업, 4개의 비전 작업, 3개의 다중 모달 작업을 수행하였으며, 5개의 사전 학습된 LLM을 사용하여 그 효율성을 입증했습니다. 연구 결과, ORAL은 7777억 개의 파라미터를 효과적으로 처리하면서도 전통적인 미세 조정 방법과 비슷하거나 우수한 성능을 보여줍니다. 이는 ORAL이 기존 방법과 비교할 때 scalability, controllability 및 portability를 모두 충족하는 새로운 기준을 정립하고 있음을 의미합니다.



### Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Mod (https://arxiv.org/abs/2503.24290)
- **What's New**: Open-Reasoner-Zero(ORZ)는 대규모 추론 지향 강화 학습(RL) 훈련의 첫 번째 오픈 소스 구현체입니다. 이 프로젝트는 단순성과 확장성에 중점을 두고 있으며, 기존의 DeepSeek-R1-Zero를 넘어서는 성능을 목표로 하고 있습니다. 연구자들이 접근할 수 있는 다양한 훈련 자원과 데이터셋을 공유하며, 오픈 소스 커뮤니티의 민주화에 중점을 두고 있습니다.

- **Technical Details**: ORZ는 기본 모델(Qwen-32B)에서 대규모 RL 훈련을 직접 수행하는 전략을 적용합니다. Proximal Policy Optimization(PPO) 알고리즘을 활용하며, GAE(Generalized Advantage Estimation)를 사용하여 간단한 규칙 기반 보상 함수와 결합하여 사용합니다. 훈련 데이터는 수천 개의 수학 및 추론 문제로 구성되어 있으며, 모델이 복잡한 문제를 해결할 수 있도록 설계되었습니다.

- **Performance Highlights**: ORZ는 AIME2024, MATH500, GPQA Diamond 벤치마크에서 기존 모델보다 월등한 성능을 보이며, DeepSeek-R1-Zero보다 훈련 단계 수가 10분의 1에 불과합니다. 모델 성능은 훈련 데이터의 양이 증가할수록 지속적으로 개선되며 포화 상태에 도달하지 않음을 보여줍니다. 우리의 접근 방식은 단순한 RL 알고리즘 설계를 통해 효율적인 훈련 과정을 스케일업하는 것에 중점을 두고 있습니다.



### Rec-R1: Bridging Generative Large Language Models and User-Centric Recommendation Systems via Reinforcement Learning (https://arxiv.org/abs/2503.24289)
- **What's New**: Rec-R1은 일반 강화 학습(framework) 구조로, 대형 언어 모델(LLMs)과 추천 시스템을 밀접하게 연계하는 방식으로 작동합니다. 이 방법은 기존의 유도(prompting)와 감독 세부 조정(SFT) 방식과는 달리, 고정된 블랙박스 추천 모델의 피드백을 직접 활용하여 LLM 생성을 최적화합니다. Rec-R1은 데이터 증류(data distillation)에 필요한 막대한 비용과 노력을 피할 수 있도록 설계되었습니다.

- **Technical Details**: Rec-R1은 사용자 질의(user query)나 행동 기록(behavioral history)과 같은 추천 관련 입력을 받아 LLM이 텍스트 출력을 생성하도록 합니다. 이 출력은 하위 추천 모델에 의해 사용되고 성능 기반 평가(performance-based evaluation)를 통해 품질이 평가됩니다. LLM은 지속적인 상호작용을 통해 추천 시스템의 목표에 더 잘 부합하는 입력을 생성하는 방법을 배우며, 이를 통해 추천 성능이 개선되도록 합니다.

- **Performance Highlights**: Rec-R1은 두 가지 대표적인 추천 시나리오인 제품 검색(product search)과 순차 추천(sequential recommendation)에서 효과성을 입증하였습니다. 실험 결과, Rec-R1은 기존의 방법들보다 일관되게 우수한 성능을 보여주었으며, 특히 사용자 프로필 정보가 없는 상황에서도 강력한 성능을 유지했습니다. 또한, Rec-R1은 대형 언어 모델의 일반적 능력을 유지하며, 추천 성능과 지시 따르기를 모두 개선하였습니다.



### MaintainCoder: Maintainable Code Generation Under Dynamic Requirements (https://arxiv.org/abs/2503.24260)
- **What's New**: 현대 코드 생성 기술은 기능적 정확성과 실행 효율성에서 큰 발전을 이루었지만, 유지보수성(maintainability)이라는 중요한 측면을 간과해왔다. 이 논문에서는 동적인 요구 사항에 적절히 대응하기 위해 MaintainCoder라는 혁신적인 솔루션을 제안하고, 이를 통해 코드의 응집력과 결합도를 낮추며 적응성을 향상시킬 수 있음을 보여준다.

- **Technical Details**: MaintainCoder는 폭포수 모델(Waterfall model)과 디자인 패턴(design patterns), 다중 에이전트 협업을 통합하여 유지보수성을 체계적으로 향상시키는 방법론이다. 또한 MaintainBench라는 벤치마크를 도입하여 요구 사항 변화에 따른 동적 메트릭을 통해 유지보수 노력을 평가하도록 설계되었다. 이 연구는 또한 기존 코드 생성 방식들이 변화하는 요구 사항에 대해 유지보수성 기준을 충족하기 어렵다는 실험적 증거를 제시한다.

- **Performance Highlights**: MaintainCoder는 초기 코드 생성에서의 유지보수성 메트릭을 14-30% 개선하며, 기능적 정확성(pass@k) 또한 더욱 높아지는 결과를 보여주었다. 실험 결과, 유지보수를 중시한 코드 생성이 평균 30.7% 향상된 초기 정확성을 나타내는 것으로 나타났다. 이는 소프트웨어 발전에 따라 프로그램 구조의 향상이 유지보수성을 높이는 데 중요한 역할을 함을 시사한다.



### PAARS: Persona Aligned Agentic Retail Shoppers (https://arxiv.org/abs/2503.24228)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델) 기반의 에이전트 프레임워크 PAARS를 제안하여 인공지능 쇼핑 에이전트의 행동을 인간 소비자와 일치시키고자 했습니다. 이 프레임워크는 과거 쇼핑 데이터를 기반으로 만들어진 페르소나(persona)를 이용하여 시뮬레이션한 쇼핑 세션을 생성합니다. 이 접근 방식은 개인의 행동하고 유사한 집단 레벨의 행동을 연구하여 사용자 행동에 대한 훨씬 더 신뢰할 수 있는 대안을 제공합니다.

- **Technical Details**: PAARS는 두 가지 주요 단계로 이루어진 페르소나 마이닝(persona mining) 방법론을 활용합니다. 첫 번째 단계에서는 고객의 쇼핑 이력을 기반으로 소비자 프로필을 생성하고, 두 번째 단계에서는 이 프로필을 통해 개인의 쇼핑 선호도를 추론합니다. 생성된 페르소나는 대형 언어 모델이 다양한 작업을 수행하는 데 필요한 필수 정보로 구성됩니다.

- **Performance Highlights**: 실험 결과, 페르소나를 사용한 쇼핑 에이전트는 기존 작업에 비해 높은 정렬 성능을 보였지만 여전히 인간 행동과의 격차가 존재함을 입증했습니다. 또한, PAARS의 초기 적용을 통한 자동화된 A/B 테스트의 가능성을 제시하며, 향후 에이전트 기반의 A/B 테스트 및 조사에서 유의미한 응용을 기대할 수 있습니다.



### MB-ORES: A Multi-Branch Object Reasoner for Visual Grounding in Remote Sensing (https://arxiv.org/abs/2503.24219)
- **What's New**: 이 논문에서는 원격 감지 이미지에 대해 객체 검출(Object Detection, OD)과 시각적 기초(Visual Grounding, VG)를 통합하는 통합 프레임워크를 제안하고 있습니다. 전통적인 OD와 VG 작업을 위한 직관적인 사전 지식을 수립하기 위해, 언급 표현 데이터를 사용하여 오픈 세트 객체 감지기를 세밀 조정하고, 부분적으로 감독된 OD 작업으로 설정합니다. 이러한 구조를 통하여 모든 객체를 탐지하면서 특정 객체의 위치를 정확하게 찾는 것을 목표로 합니다.

- **Technical Details**: 제안된 모델은 객체 질의, 클래스 임베딩, 및 제안 위치로 구성된 그래프 표현을 사용하여 각 이미지를 구성합니다. 멀티-브랜치 네트워크는 공간적, 시각적, 범주적 특성을 통합하여 작업 인식 제안을 생성하며, 객체 추론 네트워크는 제안들 사이의 확률을 할당합니다. 이 과정은 마지막으로 언급된 객체를 로컬라이즈하기 위한 부드러운 선택 메커니즘으로 이어집니다.

- **Performance Highlights**: 이 방법은 OPT-RSVG 및 DIOR-RSVG 데이터 세트에서 뛰어난 성능을 입증하였으며, 최신 방법들에 비해 상당한 성능 개선을 보여 주었습니다. 전통적인 OD 기능을 유지하면서도 보다 다양한 시나리오에서 OD의 적용 가능성을 확대하였습니다. 또한, 이 논문의 코드는 연구 결과를 재현하고 실험할 수 있도록 제공될 예정입니다.



### Grounding Agent Reasoning in Image Schemas: A Neurosymbolic Approach to Embodied Cognition (https://arxiv.org/abs/2503.24110)
- **What's New**: 이번 연구에서는 embodiment cognition 이론과 agent 시스템 간의 간극을 메우기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 sensorimotor 경험의 반복적인 패턴인 이미지 스키마(image schemas)를 포괄하는 공식적 특성을 활용합니다. LLM을 맞춤화하여 자연어 설명을 이러한 패턴 기반의 공식 표현으로 변환함으로써, 기본 개념 구조에서 에이전트의 이해를 뒷받침하는 neurosymbolic 시스템을 생성할 수 있는 가능성을 열게 됩니다.

- **Technical Details**: 연구는 신경망(neural networks)과 상징적 언어(symbolic languages)의 통합된 접근 방식을 제공합니다. 이미지 스키마는 Mandler와 Cánovas의 연구에 따라 개념적 원소로 분해될 수 있으며, 이는 구체적 물리적 상황을 다루는 데 적합한 요구 사항들을 충족해야 합니다. 이 연구는 이미지 스키마를 표현하기 위한 formalism의 주요 속성들에 대해 논의하며, 이미지 스키마가 구조화하는 무수한 물리적 구성을 표현할 수 있는 기초를 다지고 있습니다.

- **Performance Highlights**: 제안된 접근법은 인간과 에이전트 간의 직관적이고 설명 가능한 상호작용을 가능하게 하면서 reasoning 및 자연어 이해의 향상을 보여줄 것으로 예상됩니다. 본 연구는 이미지 스키마의 개념을 다루는 기존의 작업과 비교하여, 개념 구조의 완전한 공식적 특성과 기존의 상징적 해결책을 사용하는 강점을 강조합니다. 결론적으로, 이러한 새로운 프레임워크는 현대 AI 시스템의 다음 단계로 나아가는 데 필요한 기초를 제공할 것으로 보입니다.



### Get the Agents Drunk: Memory Perturbations in Autonomous Agent-based Recommender Systems (https://arxiv.org/abs/2503.23804)
- **What's New**: 이번 논문은 사용자 및 아이템 상호작용을 기반으로 한 메모리 메커니즘을 도입한 추천 시스템(Agent4RSs)의 안전 취약점을 공격하는 첫 번째 연구입니다. 특히, Agent4RSs의 메모리를 교란시켜 제한 사항을 발견하고 보안을 강화하는 방법을 제안합니다. 새로운 실용 공격 프레임워크인 DrunkAgent를 활용하여 이러한 공격이 성공적으로 이루어질 수 있도록 합니다.

- **Technical Details**: DrunkAgent는 생성 모듈, 전략 모듈, 대리 모듈로 구성되어 있습니다. 생성 모듈은 효과적인 적대적 텍스트 트리거를 생성하여 공격 목표를 달성하는 데 사용됩니다. 전략 모듈은 목표 에이전트를 혼란스럽게 하여 메모리 업데이트를 방해하며, 대리 모듈은 공격의 전이성과 인지 불가능성을 향상시키는 역할을 합니다.

- **Performance Highlights**: 다양한 실제 데이터셋을 대상으로 한 실험에서 DrunkAgent의 효과가 입증되었습니다. 이 연구는 추천 시스템의 안전하고 강건한 에이전트 개발을 위한 중요한 통찰력을 제공합니다. 전반적으로 Agent4RSs의 보안을 강화하고 더 안전한 AI 에이전트를 만드는 데 기여할 것으로 기대됩니다.



### Towards a cognitive architecture to enable natural language interaction in co-constructive task learning (https://arxiv.org/abs/2503.23760)
Comments:
          8 pages, 5 figures, submitted to: IEEE RO-MAN 2025

- **What's New**: 이번 연구에서는 Co-Constructive Task Learning (CCTL)을 활용하기 위해 인지 아키텍처(cognitive architecture)가 가져야 할 특성에 대해 다루고 있습니다. 먼저 Interactive Task Learning (ITL)의 맥락을 설명하며, 인간의 기억 시스템과 자연어 및 다중 모달리티(multi-modality)의 중요성에 대해 논의합니다. 이 논문은 인지 아키텍처의 현재 상태를 분석하고, 다양한 연구 분야에서 얻은 통찰을 통합하여 CCTL을 지원하는 통합 프레임워크를 개발합니다.

- **Technical Details**: CCTL과 관련된 핵심적인 기술적 요소로는 다중 모달리티의 중요성이 강조됩니다. 특정 행동을 성취하기 위해서는 기억 체계가 절대적으로 중요한데, 인간의 기억은 절차적(Procedural), 의미적(Semantic), 일화적(Episodic) 기억으로 나뉘며, 이러한 기억의 상호작용이 로봇 학습에 기여할 수 있습니다. 이 논문에서는 인간 상호작용 모형에서 도출된 개념들을 로봇에게 적용할 수 있는 방법을 제안합니다.

- **Performance Highlights**: 연구에서 제시된 CCTL를 성공적으로 수행하기 위해서는 교사가 로봇 학습자를 효과적으로 이끌 수 있는 몇 가지 핵심 능력들이 필요합니다. 이러한 능력으로는 주의집중을 유도하는 것, 피드백을 제공하는 백채널(back-channel), 자연어를 활용한 과제 설명, 그리고 과제를 시연하는 것이 포함됩니다. 이 연구는 로봇이 이러한 상호작용 장치를 해석하고 기존 지식 기반에 통합할 수 있도록 하는 데 필요한 아키텍처 특성을 규명하는 데 목적을 두고 있습니다.



### Short-video Propagation Influence Rating: A New Real-world Dataset and A New Large Graph Mod (https://arxiv.org/abs/2503.23746)
- **What's New**: 이 논문은 Short-video Propagation Influence Rating (SPIR) 작업을 제안하며, 단기적인 인기 예측을 넘어서서 긴 시간에 걸친 비디오의 전파 영향을 추정하려는 새로운 접근 방식을 소개합니다. 이는 사용자의 다양한 상호작용 정보를 고려하여 단일 지표에 의존하지 않고, 실질적인 전파의 영향을 평가하는 것을 목표로 합니다. 또한 최초의 크로스 플랫폼(short-video propagation) 데이터셋인 XS-Video를 도입하여, 5개 주요 플랫폼에서 수집한 비디오 데이터로 구성된 신뢰할 수 있는 근거를 제공합니다.

- **Technical Details**: XS-Video 데이터셋은 총 117,720 개의 비디오와 381,926 개의 샘플, 535 개의 주제를 포함하고 있으며, 비디오의 전파 영향력은 0에서 9까지의 등급으로 주어집니다. 논문은 또한 대규모 그래프 모델(NetGPT)을 제안하여, 서로 다른 형태의 그래프 구조 데이터를 처리하는 것과 동시에 대규모 언어 모델(LLM)의 추론 능력을 결합합니다. NetGPT는 이 새로운 세 단계 훈련 메커니즘을 기반으로 하여, 단기 비디오 전파 그래프를 이해하고 분석함으로써 전파 영향력을 예측할 수 있습니다.

- **Performance Highlights**: 실험 결과, NetGPT는 XS-Video 데이터셋에서 기존의 최첨단 방법들(GNNs, LLMs 및 멀티모달 LLMs)에 비해 월등한 성능을 보여주었습니다. 기존 방법들이 비디오 전파 분석에서 비효율적이라는 점을 보여주며, SPIR 작업에 대한 보다 정확한 접근이 필요하다는 점을 강조합니다. 이 모델은 특히 복잡한 그래프 구조와 비디오의 이종 특성을 포착하여 전파 영향력 수준을 예측하는 데 큰 강점을 보이며, 다양한 응용 분야에서 유용할 것입니다.



### KOFFVQA: An Objectively Evaluated Free-form VQA Benchmark for Large Vision-Language Models in the Korean Languag (https://arxiv.org/abs/2503.23730)
Comments:
          Accepted to CVPRW 2025, Workshop on Benchmarking and Expanding AI Multimodal Approaches

- **What's New**: 최근 대규모 비전-언어 모델(Visual-Language Models, VLMs)의 발전으로 다양한 평가 기준이 등장했습니다. 하지만 기존 평가 방법들은 주어진 응답 중에서 모델이 선택하도록 요구하거나, 주판 모델(judge model)을 사용하여 주관적이라는 문제점이 있었습니다. 본 연구에서는 한국어를 위한 새로운 평가 기준을 제공하는 KOFFVQA 벤치마크를 제안합니다.

- **Technical Details**: KOFFVQA는 275개의 주어진 이미지와 질문 쌍을 포함하며, 10가지 VLM 성능 측면을 평가하는 grading criteria(채점 기준)를 제공합니다. 각 응답은 미리 정의된 채점 기준을 기반으로 LLM(대형 언어 모델)으로 채점되며, 이는 평가의 신뢰성을 높이는 데 기여합니다. 이를 통해 작은 오픈 소스 모델이라도 신뢰할 수 있는 평가를 할 수 있습니다.

- **Performance Highlights**: KOFFVQA 벤치마크를 활용하여 47개의 VLM 모델을 평가한 결과, 한국어 언어에서의 성능은 영어 벤치마크에서의 성능과는 상이한 패턴을 보였습니다. 우리의 접근 방식은 기존 방법과 비교하여 평가의 일관성을 크게 향상시켰고, 이는 장기적인 응답을 평가할 때 발생할 수 있는 주관적 문제들을 줄여줍니다. LLM을 평가지로 사용한 방법은 특히 더 신뢰할 수 있는 결과를 제공했습니다.



### Benchmarking Systematic Relational Reasoning with Large Language and Reasoning Models (https://arxiv.org/abs/2503.23487)
Comments:
          Submitted to ACL 2025

- **What's New**: 이 논문은 대규모 언어 모델(LLM)과 대규모 추론 모델(LRM)을 사용한 체계적 추론(systematic reasoning)의 중요성을 강조합니다. 모델의 성능은 종종 규칙적인 추론보다는 지름길에 의존하는 경향이 있으며, 이는 분포 외(out-of-distribution) 예제에서 성능 저하로 이어집니다. 저자들은 공간적 및 시간적 추론에 대한 문제를 통해 이러한 모델들이 어떻게 일반화하는지를 탐구하며, 체계적 일반화(Systematic Generalization, SG) 메트릭을 기반으로 LLM과 LRM의 추론 능력을 평가하는 것이 중요하다고 주장합니다.

- **Technical Details**: 논문에서는 공간 시간 추론(Spatial Temporal Reasoning, STaR) 벤치마크를 활용하여 모델의 성능을 분석합니다. STaR는 복합적 구조를 가지며, 이를 통해 전례 없는 문제 사례를 생성할 수 있어 데이터 세트 오염 문제를 피할 수 있습니다. 이러한 문제는 계산적으로 해결 가능하며, LRM이 접근할 수 있는 문제로 설계되었습니다.

- **Performance Highlights**: 많은 유명한 LLM과 LRM이 STaR에서 어려움을 겪지만, 무작위 기회보다 나은 성과를 보입니다. 모델 규모, 파인튜닝(fine-tuning) 및 체인 오브 띵크(CoT) 테스트 시간이 성능에 미치는 영향을 파악하며, 논문에서 다루는 문제의 복잡도와 모델의 일반화 능력 간의 관계를 평가합니다.



### Codehacks: A Dataset of Adversarial Tests for Competitive Programming Problems Obtained from Codeforces (https://arxiv.org/abs/2503.23466)
Comments:
          Accepted for publication at the 18th IEEE International Conference on Software Testing, Verification and Validation (ICST 2025)

- **What's New**: 이 논문에서는 Codeforces 플랫폼에서 자동으로 수집한 실패 유도 테스트 케이스를 기반으로 하는 새로운 데이터셋인 ‘Codehacks’를 소개합니다. 이 데이터셋은 5,578개의 프로그래밍 문제에 대해 288,617개의 해킹 사례를 포함하고 있으며, 각 문제는 자연어 설명과 함께 제공됩니다. Codehacks는 LLM(대형 언어 모델)을 통해 생성된 코드의 품질을 평가하는 데 중요한 자원으로 활용될 수 있습니다.

- **Technical Details**: Codehacks 데이터셋에는 프로그래밍 문제뿐만 아니라, 이러한 문제에 대한 2,196개의 제출 솔루션의 소스 코드도 포함되어 있습니다. 이러한 해킹 기술은 사용자가 제출한 솔루션의 취약성을 발견하기 위한 것으로, 수동으로 만들기에는 비용이 많이 드는 경계 사례 테스트를 제공해 줍니다. 논문에서는 추가 테스트가 필요하다고 강조하며, Codeforces의 온라인 판별 플랫폼이 유용한 자원이라는 점을 지적합니다.

- **Performance Highlights**: Codehacks는 LLM을 사용한 프로그램 합성 기술의 검증과 평가에서 중요한 기여를 할 것으로 기대됩니다. 과거에는 테스트 검증 시 존재하는 허위 부정 결과를 찾는 데 필요한 추가 테스트를 작동시키는 과정이 비용과 시간이 많이 소요되었습니다. Codeforces에서의 해킹 사례를 활용하여, 이러한 부정확한 결과를 줄일 수 있는 효과적인 방법을 제시합니다.



### Semantic-Preserving Transformations as Mutation Operators: A Study on Their Effectiveness in Defect Detection (https://arxiv.org/abs/2503.23448)
Comments:
          Accepted for publication in Mutation 2025 at the 18th IEEE International Conference on Software Testing, Verification and Validation (ICST 2025)

- **What's New**: 이번 연구에서는 결함 탐지(defect detection) 도구의 성능을 개선하기 위해 의미 보존 변환(semantic-preserving transformations)을 사용할 수 있는지 분석했습니다. 기존의 연구들은 의미적으로 동일한 코드에서 모델의 강건성을 향상시키기 위해 훈련 데이터를 강화하는 데 집중했지만, 이러한 코드가 실제 도구 성능 개선에 어떻게 사용될 수 있는지는 잘 알려져 있지 않았습니다. 이를 통해 우리는 LLMs(대형 언어 모델)와 도구의 조합이 기존에 알려진 방식과는 다른 새로운 접근을 제시할 수 있음을 발견했습니다.

- **Technical Details**: 연구진은 28개의 논문에서 94개의 의미 보존 변환을 수집하였으며, 이 중 39개의 변환을 실제로 구현하였습니다. 그러나 수작업 검토 결과 39개 중 23개가 코드 의미를 변경하는 것으로 나타났습니다. 최종적으로 16개의 변환을 사용하여 LLMs를 통해 결함 탐지 도구의 성능을 향상시킬 수 있는지를 실험하였습니다. 연구 과정에서 세 가지 앙상블 기법을 적용하여 성과를 평가하였습니다.

- **Performance Highlights**: 본 연구의 결과, 선택된 16개의 올바른 변환과 세 가지 앙상블 기법을 사용했음에도 결함 탐지 모델의 정확도를 향상시키지 못했습니다. 연구진은 의미 보존 변환을 재사용하는 것이 어렵고, 일부 변환이 의도치 않게 의미를 변경할 수 있음을 발견했습니다. 따라서 향후 연구에서는 이러한 구현의 어려움을 극복하기 위한 방안이 필요하다는 인사이트를 제공합니다.



### What Makes an Evaluation Useful? Common Pitfalls and Best Practices (https://arxiv.org/abs/2503.23424)
- **What's New**: 최근 몇 년 사이 인공지능(AI)의 발전이 급격히 이루어짐에 따라 AI 커뮤니티에서는 잠재적인 안전 위험에 대한 우려가 커지고 있습니다. 본 논문에서는 AI 시스템의 안전한 사용과 개발을 위한 고품질 평가의 필요성을 강조하며, 이러한 평가를 위한 모범 사례를 제공하고 있습니다. 특히, 사이버 보안 사례를 통해 모델 평가의 모범 사례를 어떻게 정의하고 적용할 수 있는지를 설명합니다.

- **Technical Details**: AI 모델의 평가 설계는 위협 모델링(threat modeling)과 평가 설계를 잇는 초기 사고 과정 단계를 논의하는 것으로 시작됩니다. 또한 유용한 평가의 특성과 파라미터를 제시하고, 특정 평가 구축에서 전체적인 평가 스위트(suite) 구축으로 넘어갈 때 고려해야 할 사항들을 다룹니다. 이를 통해 AI 시스템의 안전 평가를 위한 체계적 접근 방식을 제안하고 있습니다.

- **Performance Highlights**: 이 연구의 주요 기여 중 하나는 결정 과정(decision making processes), 위협 모델링과 평가 설계(threat modeling and evaluation design) 간의 중요한 연결 고리를 수립한 점입니다. 모범 사례에 대한 명확한 원칙을 확립하고 이를 바탕으로 안전 평가를 위한 평가 스위트 구성을 위한 가이드를 제공합니다. 이 논문은 실험적 검증이 향후 중요한 작업이 될 것임을 지적하며, AI 기술의 보다 안전하고 책임 있는 발전에 기여할 것으로 기대됩니다.



### Large Language Models Are Better Logical Fallacy Reasoners with Counterargument, Explanation, and Goal-Aware Prompt Formulation (https://arxiv.org/abs/2503.23363)
Comments:
          Accepted to NAACL 2025 Findings

- **What's New**: 본 연구에서는 논리적 오류 탐지를 위한 새로운 프롬프트(formulation) 기법을 제안하며, 이는 감독 학습(supervised) 및 비감독 학습(unsupervised) 환경에 모두 적용이 가능하다. 이 방법은 입력 텍스트에 암묵적 맥락 정보(implicit contextual information)를 통합하여 오류의 유효성을 평가하는 쿼리를 생성하고, 이를 기반으로 결과를 분류한다. 또한, 다섯 개의 데이터 세트를 사용한 평가 결과 시간적 모델들에 비해 현저한 성능 향상을 확인했다.

- **Technical Details**: 제안된 접근법은 네 개의 주요 단계로 구성되며, 첫 단계에서는 LLM을 이용해 맥락 개선을 통해 앵커 쿼리(context-informed queries)를 생성한다. 이후 생성된 쿼리를 통해 논리적 오류를 분류하며, 마지막 단계에서는 각 쿼리에 대해 신뢰도 기반으로 순위를 매긴다. 특이하게도, 본 연구에서는 각 입력 텍스트를 증강하기 위해 세 가지 유형의 암묵적 정보(반론(counterargument), 설명(explanation), 목표(goal))를 활용한다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 제로샷(zero-shot) 환경에서 Macro-F1 점수가 최대 0.60, 파인튜닝(fine-tuned) 모델에서는 최대 0.45 향상된 성능을 보였다. 따라서 본 접근법이 최첨단 모델들보다 월등한 결과를 드러내었으며, 이는 프롬프트 순위 매기기 방법의 효과적 활용에 기인한 것으로 분석되었다. 단계별로 수행된 심층 분석을 통해 제안 방법의 장점과 개선점을 추가적으로 검토하였다.



### A Scalable Framework for Evaluating Health Language Models (https://arxiv.org/abs/2503.23339)
- **What's New**: 최근의 연구에 따르면, 대형 언어 모델(LLMs)은 개인화된 건강 정보를 제공할 때 유용한 응답을 생성하는 데 있어 잠재력을 가지고 있습니다. 이에 따라 건강 응용 프로그램에서 LLM의 채택이 증가하고 있으며, 정확성, 개인화 및 안전성을 포함한 여러 차원에서 응답 품질을 보장하기 위한 엄격하고 효율적인 평가 방법론이 필수적입니다.

- **Technical Details**: 이 연구에서는 Adaptive Precise Boolean rubrics라는 평가 프레임워크를 소개합니다. 이 방법은 모델 응답에서의 격차를 식별하기 위해 최소한의 목표 기반 질문 세트를 사용하여 개방형 질문의 인간 및 자동 평가를 간소화합니다. 이 접근법은 복잡한 평가 목표와 보다 구체적이고 정량화된 목표를 대조하는 최근의 연구를 기반으로 합니다.

- **Performance Highlights**: Adaptive Precise Boolean rubrics는 전문가와 비전문가 평가자 간의 높은 평가자 간 일치를 도출하며, 자동 평가에서도 전통적인 Likert 척도에 비해 더 높은 일치를 기록했습니다. 또한, 이 방법은 Likert 기반 방식의 평가 시간의 약 절반만을 요구해 효율성을 크게 향상시킵니다. 이는 건강 분야에서 LLM에 대한 더 광범위하고 비용 효율적인 평가를 가능하게 합니다.



### Beyond Unimodal Boundaries: Generative Recommendation with Multimodal Semantics (https://arxiv.org/abs/2503.23333)
- **What's New**: 이번 논문에서는 Multimodal Generative Recommendation (MGR)이라는 새로운 접근 방식을 제안합니다. 기존의 Generative Recommendation (GR) 방법들이 주로 unimodal 데이터를 사용하였던 한계를 극복하여, 다양한 모달리티를 통합하는 방법론을 다룹니다. 저자들은 특히 모달리티 선택의 중요성과 그것이 GR 모델의 성능에 미치는 영향을 강조하고 있습니다.

- **Technical Details**: MGR-LF++라는 새로운 Late Fusion 프레임워크를 도입하여, 서로 다른 모달리티 정보를 효과적으로 관리하는 방법을 제안합니다. 이 프레임워크는 contrastive modality alignment 훈련 기법과 각 모달리티를 구분하는 특별한 토큰을 사용하여, 서로 다른 semantic IDs의 일치를 도모합니다. 이를 통해, 다양한 모달리티의 정보를 손실 없이 통합할 수 있는 방법을 모색합니다.

- **Performance Highlights**: MGR-LF++는 기존의 unimodal 접근 방법 대비 20% 이상의 성능 향상을 달성하였습니다. 저자들은 6개의 기준선 모델을 사용하여 3개 데이터셋에서 실험을 실시하였으며, 그 결과 다중 모달리티 정보를 활용하는 것이 Generative Recommendation의 효과를 크게 향상시킬 수 있음을 입증했습니다.



### SPIO: Ensemble and Selective Strategies via LLM-Based Multi-Agent Planning in Automated Data Scienc (https://arxiv.org/abs/2503.23314)
Comments:
          Under Review

- **What's New**: 이번 논문에서는 SPIO(Sequential Plan Integration and Optimization)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 LLM(대형 언어 모델)을 이용한 의사결정 방식을 통해 샘플 전략을 생성하고 최적화합니다. 기존의 단일 경로 워크플로우와 달리 SPIO는 다단계 처리 프로세스를 적용하여 데이터 전처리, 특성 엔지니어링, 모델링, 하이퍼파라미터 조정까지 아우릅니다. 이를 통해 다양한 전략을 탐색할 수 있는 기회를 제공합니다.

- **Technical Details**: SPIO 프레임워크는 네 가지 주요 모듈인 데이터 전처리(data preprocessing), 특성 엔지니어링(feature engineering), 모델 선택(model selection), 하이퍼파라미터 조정(hyperparameter tuning)으로 구성됩니다. 각 모듈에서는 독립적으로 후보 전략을 생성하는 전용 계획 에이전트(planning agents)가 존재합니다. SPIO는 또한 두 가지 변형인 SPIO-S(단일 최적 계획 선택)와 SPIO-E(상위 k개 계획을 조합)로 나뉘어 각각의 활용도를 극대화할 수 있습니다.

- **Performance Highlights**: Kaggle과 OpenML 데이터 세트를 대상으로 한 광범위한 실험에서 SPIO는 최신 방법론보다 우수한 성능을 나타냈습니다. SPIO의 적응형 다경로(reasoning) 접근 방식은 다양한 통찰력을 통합할 수 있어 고정된 단일 경로 워크플로우의 한계를 효과적으로 극복합니다. 이로 인해 SPIO는 예측 정확도를 지속적으로 향상시키고, 다양한 데이터 시나리오에 적응하며, 실행 신뢰성을 높이는 데 있어 탁월한 성과를 보이고 있습니다.



### Beyond Contrastive Learning: Synthetic Data Enables List-wise Training with Multiple Levels of Relevanc (https://arxiv.org/abs/2503.23239)
Comments:
          Code: this https URL

- **What's New**: 이 논문에서는 정보 검색(Information Retrieval, IR)에서 큰 언어 모델(LLM)을 활용하여 기존의 훈련 방식을 개선하는 SyCL (Synthetic ranking Context for List-wise training) 방법을 제안합니다. SyCL은 실제 문서를 사용하지 않고도 여러 수준의 관련성을 가진 합성 문서를 생성하여 IR의 효율성을 극대화합니다. 이를 통해 기존의 이진 라벨로 한정된 훈련 방식을 뛰어넘고, 더 복잡한 문서 순위 매기기를 가능하게 합니다.

- **Technical Details**: 제안된 SyCL 방법은 오픈 소스 LLM을 활용하여 MS MARCO 데이터셋에 대한 질의에 따라 네 가지 다른 관련성 수준을 가진 전방위 대량 합성 문서를 생성합니다. 이 문서들은 Wasserstein Distance를 손실 함수로 사용하여 훈련 중 상대적 라벨 불일치를 반영하여 모델의 점수 선택을 다르게 패널티합니다. SyCL은 대규모 IR 데이터셋(~2M 샘플)을 생성하며, 이로 인해 복잡한 훈련 파이프라인을 피하고 데이터 품질 문제를 완화합니다.

- **Performance Highlights**: SyCL을 사용한 실험 결과, 제안된 방법은 InfoNCE 기반의 전통적인 훈련 방식에 비해 성능이 현저히 향상됨을 보여줍니다. BEIR 데이터셋의 제로샷 평가에서 SyCL은 36.8에서 43.2로 평균 nDCG@10 점수를 개선하여, 실제 라벨이 있는 문서로 훈련된 모델과 유사한 성능을 달성하였습니다. 이 결과는 실제 문서 없이도 강력한 순위 매기기 성능을 구현할 수 있음을 잘 보여줍니다.



### TRA: Better Length Generalisation with Threshold Relative Attention (https://arxiv.org/abs/2503.23174)
- **What's New**: 이번 연구에서는 트랜스포머 모델의 길이 일반화(length generalisation) 문제를 분석합니다. 이 문제는 주로 self-attention 메커니즘의 두 가지 주요 결함에 기인한다고 주장합니다. 첫째, 관련 없는 정보를 완전히 제거하지 못하는 것이고, 둘째, 위치와 관련된 학습된 편향이 관련 없는 키를 불필요하게 강조할 수 있다는 점입니다. 이를 해결하기 위한 새로운 기법으로 임계값 상대 주의(Threshold Relative Attention, TRA)를 제안하여 성능을 향상시키려 합니다.

- **Technical Details**: TRA 메커니즘은 주의(weighting) 메커니즘을 개선하여 일반적인 솔루션을 학습할 수 있도록 합니다. 이 방법은 우선 원시 주의 가중치에 기반하여 관련 없는 키를 마스킹(masking)하고, 그 다음에 남은 키 사이의 상대 거리를 계산합니다. 이 과정을 통해, TRA는 대규모 트랜스포머의 길이 일반화 성능을 개선하고, 고장률(perplexity)을 낮출 수 있음을 보여줍니다.

- **Performance Highlights**: TRA를 적용한 결과, 합성 벤치마크에서 길이 일반화가 크게 개선되었습니다. 또한, 배포 밖의 시퀀스 길이에서도 언어 모델링의 고장률이 우수하였으며, 최대 32배까지 성능이 향상되었습니다. 이러한 결과는 더욱 강력한 주의 메커니즘 개발에 기여할 것으로 기대됩니다.



### CodeARC: Benchmarking Reasoning Capabilities of LLM Agents for Inductive Program Synthesis (https://arxiv.org/abs/2503.23145)
- **What's New**: 새로운 평가 프레임워크인 CodeARC(코드 추상화 및 추론 챌린지)가 도입되었습니다. 이는 에이전트가 숨겨진 목표 함수를 질의하고 반복적으로 솔루션을 조정할 수 있는 상호작용 설정을 제공합니다. 기존의 정적 예제에 의존했던 평가 방식의 한계를 보완하며, 실제 상황을 반영할 수 있도록 설계되었습니다.

- **Technical Details**: CodeARC는 LLM 기반의 에이전트가 초기 입력-출력 예제 집합을 사용하여 작업을 시작하고, 새로운 입력으로 목표 함수를 질의하며, 미분 테스트 오라클을 통해 검증하고 디버깅할 수 있는 구조를 가집니다. 이 과정에서는 에이전트가 스스로 입력을 생성하고 피드백에 따라 솔루션을 수정해야 합니다.

- **Performance Highlights**: CodeARC를 사용한 실험 결과, 총 18개 모델 중 OpenAI의 o3-mini가 52.7%의 성공률로 가장 뛰어난 성과를 거두었습니다. 또한, LLaMA-3.1-8B-Instruct의 세부 조정(Fine-tuning)을 통해 최대 31%의 상대적 성능 향상을 달성했습니다.



### When 'YES' Meets 'BUT': Can Large Models Comprehend Contradictory Humor Through Comparative Reasoning? (https://arxiv.org/abs/2503.23137)
- **What's New**: 이 논문에서는 복잡하고 모순적인 내러티브를 통한 유머 이해의 어려움을 다루기 위해 다국어 및 다문화적 맥락에서 발췌한 1,262개의 만화 이미지로 구성된 새로운 벤치마크인 YesBut(V2)를 도입합니다. 이 데이터셋은 내러티브 이해의 다양한 측면을 포괄하는 종합적인 주석이 포함되어 있으며, 이를 통해 대조적인 요소들 사이의 비교적 추론을 평가합니다. 본 연구는 VLMs(vision-language models)가 유머를 포함하는 시각적 대비 구조를 해석하는 데 있어 갖는 한계를 드러내며, 향후 AI의 인지적 역할을 발전시키기 위한 기초를 제공합니다.

- **Technical Details**: 논문에서 제안한 YesBut 벤치마크에는 만화 내에서 표현된 시각적 및 텍스트 구성 요소에 대한 표면적 이해에서부터 깊이 있는 내러티브 추론에 이르기까지 네 가지 보완적 작업이 포함됩니다. 각 작업은 VLM이 만화의 복잡한 내러티브를 이해하는 성과를 평가하기 위해 설계되었습니다. VLMs 는 일반적으로 시각적 인식, 주요 요소 식별, 비교 분석, 환각 등의 분야에서 인간보다 크게 뒤떨어지는 성능을 보였습니다.

- **Performance Highlights**: 실험 결과, VLM들은 내러티브 유머를 이해하는 데 있어서 상당한 한계를 보였으며, 특히 비주얼 인식 및 주요 요소 분석에서 일반적인 오류가 발생했습니다. 모델의 성능 향상을 위한 텍스트 기반 훈련 전략 및 사회적 지식 강화 방법이 제안되었고, 이를 통해 향후 연구 방향이 명확해졌습니다. YesBut의 확장 이후 성공적인 평가를 위한 새로운 벤치마크 설정은 VLM의 심층적 세멘틱 추론 발전을 이끄는 데 중요한 기초를 형성합니다.



### Can DeepSeek-V3 Reason Like a Surgeon? An Empirical Evaluation for Vision-Language Understanding in Robotic-Assisted Surgery (https://arxiv.org/abs/2503.23130)
Comments:
          Technical Report

- **What's New**: DeepSeek-V3는 최근에 등장한 대규모 언어 모델(LLM)로, 일반적인 장면 이해, 질문 응답(QA), 텍스트 생성 작업에서 탁월한 성능을 보여줍니다. 이 연구에서는 DeepSeek-V3의 로봇 수술 시나리오에서의 대화 능력을 조사하며, 단일 구문 QA, 시각 QA 및 상세 설명과 같은 작업에 중점을 두고 있습니다. 평가 결과, DeepSeek-V3는 특정 프롬프트를 제공할 때 수술 도구 및 조직 인식 작업에서 좋은 성능을 보이나, 공간 위치 분석에서는 제한된 능력을 보입니다.

- **Technical Details**: DeepSeek-V3는 로봇 보조 수술(RAS) 환경에서의 복잡한 시각 환경 이해와 맥락 인식 도움을 중요하게 여기는 인공지능 모델입니다. 실험은 단일 구문 QA, 시각 QA, 그리고 상세 설명의 세 가지 패러다임으로 나뉘어 진행되며, EndoVis18과 CholecT50 데이터셋을 사용하여 성능을 평가합니다. 각 패러다임은 수술에 관련된 비주얼 질문 응답(VQA) 모델의 성능을 체계적으로 분석하기 위해 설계되었습니다.

- **Performance Highlights**: DeepSeek-V3는 EndoVis18 데이터셋에서 전체적으로 더 나은 성과를 보였으며, 단일 구문 QA에서는 모델이 질문 이해와 정보 처리에서 우수한 성과를 나타냈습니다. 그러나 수술 도구의 동작 및 공간 위치 분석에서는 여전히 한계를 보여주었으며, 깊이 있는 설명이 필요한 작업에서 낮은 성능을 보였습니다. CholecT50 데이터셋에서도 비슷한 성능을 발휘했지만, 수술 조직 분석은 두 모델 모두에게 가장 큰 도전 과제가 되고 있습니다.



### A large-scale image-text dataset benchmark for farmland segmentation (https://arxiv.org/abs/2503.23106)
- **What's New**: 이번 논문에서는 전통적인 딥러닝 패러다임의 한계를 극복하기 위해 언어 기반의 학습 패러다임을 제안합니다. 특히 농지의 시공간적(spatiotemporal) 특징을 명확하게 표현할 수 있는 언어의 역할을 강조하고 있습니다. 이를 통해 농지의 동적 시간적 진화와 공간 이질성을 효과적으로 모델링할 수 있습니다. 새로운 FarmSeg-VL 데이터셋을 창출함으로써 이 분야의 연구에 필요한 기본적인 벤치마크 데이터셋을 제공합니다.

- **Technical Details**: FarmSeg-VL은 농지에 대한 언어 기반의 설명을 포함한 최초의 상세 이미지-텍스트 데이터셋으로, 농지의 시공간적(spatiotemporal) 특징을 지원합니다. 또한, 이 논문에서는 각 이미지에 대한 설명(caption)을 정확하게 부여하는 반자동(annotation) 방법을 개발하여 데이터의 질과 의미적 풍부성을 보장합니다. 데이터셋은 8개의 전형적인 농업 지역을 포함하여 공간 차원에서의 다양성을 갖추고 있습니다.

- **Performance Highlights**: FarmSeg-VL로 훈련된 VLMs(비전-언어 모델)와 라벨에만 의존하는 딥러닝 모델의 성능 분석을 제시하였습니다. 이를 통해 FarmSeg-VL이 농지 분할을 위한 표준 벤치마크로서의 잠재력을 시연하는 결과를 보여줍니다. 또한 농지의 고유한 속성, 단계적(pheno-logical) 특성, 공간 분포 등의 풍부한 시공간적 특성을 포괄합니다.



### Beyond Standard MoE: Mixture of Latent Experts for Resource-Efficient Language Models (https://arxiv.org/abs/2503.23100)
- **What's New**: 본 논문은 Mixture of Experts (MoE) 아키텍처의 한계를 극복하기 위해 Mixture of Latent Experts (MoLE)라는 새로운 매개변수화 방법론을 제안합니다. MoLE는 전문가 모듈을 공유 잠재 공간으로 매핑함으로써 모델의 파라미터 수와 계산 요구사항을 현저히 줄입니다. 이를 통해 기존 MoE 아키텍처보다 더 효율적으로 대규모 언어 모델(LLMs)을 확장할 수 있는 방법을 제시합니다.

- **Technical Details**: MoLE 접근법은 각 전문가 작업을 두 가지 주요 구성 요소로 체계적으로 분해합니다: 먼저 압축된 잠재 공간으로의 공유 투영을 수행하고, 그 다음 전문가에 특화된 변환을 적용합니다. 이 과정에서 MoLE는 각 전문가의 가중치 행렬을 인수 분해하여 매개변수 수를 크게 줄이고 계산 복잡성을 감소시킵니다. 알고리즘 측면에서, 최적의 인수 분해 조건을 수학적으로 정의하고, 효율적인 두 단계 변환 알고리즘을 개발합니다.

- **Performance Highlights**: 광범위한 실험 평가를 통해 MoLE는 기존 MoE 아키텍처와 비교했을 때 경쟁력 있는 성능을 유지하면서도 자원 요구사항을 획기적으로 줄임을 입증하였습니다. 이러한 점은 특히 자원이 제한된 환경에서도 MoLE 모델을 실용적으로 적용할 수 있는 가능성을 보여줍니다. 또한, MoLE는 다양한 언어 처리 작업에서 모델 능력을 보존하거나 향상시키는 것으로 나타났습니다.



### Efficient Adaptation For Remote Sensing Visual Grounding (https://arxiv.org/abs/2503.23083)
- **What's New**: 이번 연구에서는 Parameter Efficient Fine Tuning (PEFT) 기법을 적용하여 원격 탐사(remote sensing, RS) 작업에 적합하도록 Grounding DINO와 OFA 모델을 조정했습니다. 연구 결과, LoRA 기법을 통해 DIOR-RSVG 데이터세트에서 최상위 성능을 달성했으며, Adapter와 BitFit 기술을 비교한 결과 Adapter가 고성능을 보였습니다. 이 연구는 PEFT 기법의 가능성을 강조하며, 전체 모델 학습의 비용 대비 효율적인 대안을 제시합니다.

- **Technical Details**: Parameter Efficient Fine Tuning (PEFT)은 모델의 최소한의 파라미터 집합만을 조정하여 계산 효율성을 제공합니다. Adapters는 경량 모듈로, 사전 훈련된 모델의 레이어 사이에 삽입되어 특정 작업 학습을 위한 추가 파라미터를 도입합니다. LoRA는 모델의 가중치 행렬에 저랭크 업데이트를 적용하여 학습 가능한 파라미터 수를 줄이고, BitFit은 모델 레이어의 편향 항만을 미세 조정하는 방법입니다.

- **Performance Highlights**: 실험 결과, Grounding DINO는 SOTA VG 모델로서, 텍스트 프롬프트와 이미지 내 특정 영역을 연결하는데 있어 뛰어난 성능을 발휘했습니다. Multi-scale deformable attention이 다양한 공간 해상도와 이미지 특징의 통합을 용이하게 하여 정확한 객체 탐지와 구문 로컬라이제이션을 가능하게 했습니다. 또한 모델의 계산 효율을 높이기 위해 고정 파라미터 비율을 측정하여 PEFT 기술을 통한 비용 절감 효과를 평가했습니다.



### Agentic Large Language Models, a survey (https://arxiv.org/abs/2503.23037)
- **What's New**: 최근 에이전틱 LLMs(agentic large language models)의 발전은 연구자들 사이에서 큰 관심을 받고 있습니다. 이 논문에서는 이러한 LLM들이 (1) 추론(reasoning), (2) 행동(action), (3) 상호작용(interaction)하는 능력을 갖춘다고 정의하며, 이에 대한 문헌을 체계적으로 정리하고 있습니다. 에이전틱 LLMs는 의학, 물류, 금융 등 다양한 분야에 활용되고 있으며, 자가 반성(self-reflection) 및 역할놀이(role-playing)는 새로운 연구의 가능성을 열어줍니다.

- **Technical Details**: 에이전틱 LLMs는 자연어 처리(natural language processing), 도구 통합(tool integration), 강화 학습(reinforcement learning) 등의 다양한 기술적 발전에 의존하고 있습니다. 논문은 세 가지 범주로 문헌을 나누어 에이전트가 어떻게 더 지능적으로 행동하고 상호작용할 수 있도록 발전해왔는지를 설명합니다. 이러한 기술적 발전은 에이전트들이 환경과 상호작용함으로써 새로운 훈련 데이터를 생성하고, 더 나아가 기존의 LLM 교육 방식을 보완하는 데 기여합니다.

- **Performance Highlights**: 이 논문은 에이전틱 LLM의 성능을 높이기 위한 연구 의제를 제시하며, LLM들이 의사 결정 및 협업 문제 해결에 어떻게 기여할 수 있는지를 강조합니다. 또한, LLM들이 자가 훈련(self-training) 수행을 통해 더 많은 훈련 데이터를 생성할 수 있는 기회를 제공함으로써 언어 모델이 계속해서 학습할 수 있는 방법리를 제시합니다. 그러나 LLM들이 현실 세계에서 행동할 경우 발생할 수 있는 위험 요소에 대해서도 경고하고 있습니다.



### FindTheFlaws: Annotated Errors for Detecting Flawed Reasoning and Scalable Oversight Research (https://arxiv.org/abs/2503.22989)
Comments:
          43 pages, 3 figures. for associated repository, see this https URL

- **What's New**: 이번 논문에서는 AI 모델의 감독이 점점 더 어려워지는 문제를 해결하기 위한 새로운 데이터셋을 소개합니다. 이 데이터셋은 의료, 수학, 과학, 코딩 및 Lojban 언어를 포함한 다섯 가지 다양한 분야에서 활용됩니다. 저자들은 FindTheFlaws라는 이름의 데이터셋을 통해 전문 검증된 정확한 솔루션과 특정 오류를 강조한 잘못된 솔루션을 포함하고 있습니다.

- **Technical Details**: FindTheFlaws 데이터셋은 전문가 주석이 달린 긴 형식의 질문과 솔루션으로 구성됩니다. 연구에서는 debate, critique, prover-verifier games와 같은 다양한 AI 감독 방식의 확장성을 평가하며, 각 모델의 비판적 능력을 평가합니다. 모델들의 성능을 통해 특정 데이터셋에서 잘못된 성능을 보이는 모델이 보다 능력 있는 모델의 판단자로 사용될 수 있음을 제안합니다.

- **Performance Highlights**: 평가 결과, 일부 태스크/데이터셋 조합에서 전문 기준이 최고 모델의 성능을 초과하는 경우가 발견되었습니다. 이는 전문 지식 기반이 더욱 확장 가능한 감독 실험에 유리할 수 있음을 나타냅니다. 이 연구는 AI 감독의 미래 방향에 대한 중요한 통찰을 제공합니다.



### HRET: A Self-Evolving LLM Evaluation Toolkit for Korean (https://arxiv.org/abs/2503.22968)
- **What's New**: 최근 한국 대형 언어 모델(LLM)의 발전에 따라 여러 벤치마크와 평가 방법론이 생겨났지만, 표준화된 평가 프레임워크의 부재로 인해 불일치한 결과와 비교의 한계가 발생했습니다. 이를 해결하기 위해 우리는 한국 LLM에 특화된 오픈 소스인 HRET(하래 평가 도구킷)를 소개합니다. HRET는 로짓 기반 점수 측정, 정확 일치, 언어 불일치 패널화 및 LLM-as-a-Judge 평가를 포함한 다양한 평가 방법을 통합합니다.

- **Technical Details**: HRET는 모듈화된 레지스트리 기반 아키텍처를 갖추고 있으며 주요 벤치마크(HAE-RAE Bench, KMMLU, KUDGE, HRM8K)와 여러 인퍼런스 백엔드(vLLM, HuggingFace, OpenAI와 호환되는 엔드포인트)를 통합합니다. 자가 진화하는 파이프라인을 통해 HRET는 지속적인 발전을 지원하며, 한국어 NLP 연구에서 재현 가능하고 공정한 평가의 기초를 제공합니다. 이 도구킷은 직관적인 API와 간소화된 커맨드라인 인터페이스를 통해 연구자와 실무자 모두에게 접근성을 높여줍니다.

- **Performance Highlights**: HRET를 통해 한국 LLM의 평가에서 나타나는 재현성 및 비교 용이성 문제를 해결할 수 있습니다. 평가 방법의 표준화를 통해 HRET는 한국어 모델의 성능을 보다 신뢰성 있게 평가할 수 있는 기반을 제공합니다. 이 도구킷은 Apache License 2.0 하에 공개될 예정으로, 널리 사용되고 커뮤니티의 기여를 장려합니다.



### Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models (https://arxiv.org/abs/2503.22879)
- **What's New**: 최근 State Space Models (SSMs)은 메모리 사용의 일관성과 높은 성능 덕분에 Transformers에 대한 매력적인 대안으로 떠오르고 있습니다. 그러나 클라우드 서비스나 자원 제한 장치에서 SSM을 확장하는 데 필요한 저장 요구량과 컴퓨팅 파워가 도전 과제가 되고 있습니다. 이를 해결하기 위해, Quamba2는 다양한 상황에 대한 효율성을 고려하여 W8A8, W4A8 및 W4A16 비트폭을 지원하는 포스트 트레이닝 양자화(PTQ) 프레임워크를 제공합니다.

- **Technical Details**: Quamba2는 SSM의 채널 순서 보존과 활성화 지속성을 기반으로 한 오프라인 양자화 방식을 제안합니다. 입력 데이터의 정렬 및 클러스터링을 통해 8비트 양자화를 처리하며, 상태 그룹별로 양자화를 적용하여 입력 의존 매개변수를 정밀하게 최적화합니다. 이러한 방식은 SSM의 속성을 활용하여 양자화 정확도를 높이고, 메모리 요구 사항을 줄이며, 성능 저하를 최소화합니다.

- **Performance Highlights**: Quamba2-8B는 여러 최신 SSM 양자화 방법들을 초월하여, 예비 채우기 및 생성 단계에서 각각 1.3배 및 3배 빠른 속도를 제공하며, 4배의 메모리 감소도 달성합니다. 평균 1.6%의 정확도 손실로 6개의 제로샷 작업에서 성능을 유지하는 동시에, MMLU 데이터셋에서의 평가를 통해 모델의 일반화 및 내구성을 입증하였습니다.



### L0-Reasoning Bench: Evaluating Procedural Correctness in Language Models via Simple Program Execution (https://arxiv.org/abs/2503.22832)
- **What's New**: L0-Bench는 복잡한 추론 작업의 핵심인 'level-0' 추론 능력을 평가하기 위해 개발된 새로운 벤치마크입니다. 이 벤치마크는 순차적이고 정확한 규칙 적용을 요구하며, 모델이 완벽하게 실행 추적(execution trace)을 생성할 수 있는지를 검증합니다. 기존의 성과 평가 방식과는 달리, 결과의 정당성을 넘어 과정의 정확성에 중점을 둡니다. L0-Bench는 다양한 차원에서 테스트 프로그램을 시스템적으로 생성할 수 있다는 점에서 혁신적인 접근을 제공합니다.

- **Technical Details**: L0-Bench는 합성 파이썬 프로그램(synthetic Python programs)을 사용하여 모델의 절차적 정확성(procedural correctness)을 평가합니다. 프로그래밍 언어의 명확한 정의된 프로세스를 통해 자연어의 모호성을 제거하고, 각 명령어에 대한 신뢰할 수 있는 실행을 분리합니다. 여기에 사용되는 생선 문법(generative grammar)을 통해 프로그램 생성의 효율성을 높이고, 여러 측면에서 테스트의 범위를 조정할 수 있습니다. L0-Bench는 20개 모델을 평가하여 프로시저를 따르는 능력을 정량적으로 분석합니다.

- **Performance Highlights**: 결과에 따르면 모든 모델은 목표 추적 단계의 수가 증가함에 따라 성능이 저하되는 경향을 보였습니다. 그러나 더 큰 모델과 'reasoning-enhanced' 모델이 여러 단계의 정확성을 더 잘 유지하는 것으로 나타났습니다. 또한 테스트 시 스케일링을 통해 기본적인 조건을 만족할 수 있는 성능 개선도 나타났으나, 높은 수치의 demonstratons와 majority voting에서 성능 한계가 드러났습니다. L0-Bench의 도입은 현재의 LLMs의 한계를 규명하고, 더 신뢰할 수 있는 추론 시스템에 대한 발전 가능성을 제시합니다.



### Adaptive Integrated Layered Attention (AILA) (https://arxiv.org/abs/2503.22742)
- **What's New**: 이번 연구에서는 Adaptive Integrated Layered Attention (AILA)라는 신경망 아키텍처를 제안합니다. AILA는 다양한 네트워크 층 간의 적응형(feature reuse) 기능을 위해 밀집 스킵 연결(dense skip connections)과 여러 메커니즘을 융합하여 구성되어 있습니다. AILA는 가격 예측, 이미지 인식, 감정 분석의 세 가지 도전 과제를 평가받았으며, 기존의 강력한 딥러닝 모델과 유사한 성능을 보이면서도 훈련 및 추론 시간을 크게 단축시켰습니다.

- **Technical Details**: AILA는 두 가지 아키텍처, 즉 AILA-Architecture 1과 AILA-Architecture 2로 나뉘어 있습니다. AILA-Architecture 1은 층 간의 연결 메커니즘으로 간단한 선형 층(linear layers)을 사용하고, AILA-Architecture 2는_attention_ 메커니즘을 구현하여 이전 층의 출력을 선택적으로 강조합니다. 이러한 아키텍처는 각기 다른 태스크에 대해 개별적으로 훈련되며, 다양한 네트워크 깊이에서 관련 기능을 유연하게 재사용함으로써, 강력한 성능 향상을 이루어냅니다.

- **Performance Highlights**: AILA는 세 가지 기준 벤치마크에서 강력한 성능 지표를 달성했습니다. 가격 예측, CIFAR-10 데이터셋에 대한 이미지 인식, IMDB 영화 리뷰 데이터셋의 감정 분석에서 AILA-Architecture 1 및 2 모두 LSTM, Transformer, CNN과 같은 기존의 강력한 기준 모델과 경쟁하며 이를 초월하는 성과를 보여주었습니다. 결과적으로 AILA는 일반적인 고정 연결 방식이 아닌, 적응형 정보 흐름을 통해 복잡한 태스크에서 성능을 향상시키는 새로운 길을 열었습니다.



### Training in translation tools and technologies: Findings of the EMT survey 2023 (https://arxiv.org/abs/2503.22735)
- **What's New**: 이번 논문은 대학원 번역 교육 프로그램에서 가르치는 컴퓨터화된 도구와 기술에 대한 세 번째 조사 결과를 보고합니다. EMT 네트워크의 지원 하에 진행된 이 조사는 50% 이상이 네트워크 외부에서 응답을 받았습니다. 조사 결과는 프로그램이 번역 기술의 혁신에 얼마나 민감하게 반응하고 있는지를 보여주며, 기계 번역(machine translation), 포스트 에디팅(post-editing), 품질 평가(quality evaluation)의 의무적 포함이 증가하고 생성을 돕는 도구의 출시에 신속히 대응하고 있음을 나타냅니다.

- **Technical Details**: Covid-19 팬데믹 동안 요구되었던 유연성이 프로그램에 지속적인 변화를 가져왔습니다. 교육받고 있는 도구의 범위는 계속 확장되고 있지만, 프로그램들은 클라우드 기반 소프트웨어(cloud-based software)와 무료 학술 접근이 가능한 도구 중심으로 핵심 프로그램을 통합하고 있는 것으로 보입니다. 번역 기술과 관련된 전문적 맥락 및 작업 흐름의 포함이 증가하였고, 일반적인 파일 관리(file management) 및 데이터 보안(data security) 기술의 중요성 인식이 증가했습니다.

- **Performance Highlights**: 번역 데이터와 관련된 법적 및 윤리적 문제들이 더욱 두드러지게 나타났습니다. 과정 전달 측면에서 EMT2017에서 확인된 전통적인 실험실(labs)로부터의 이탈이 뚜렷하게 가속화되었으며, 이는 팬데믹으로 인한 영향을 받았으리라 추측됩니다. 학생 개인 장치(personal devices)의 사용이 급격히 확대된 점도 주목할 만합니다.



### Reasoning Beyond Limits: Advances and Open Problems for LLMs (https://arxiv.org/abs/2503.22732)
Comments:
          41 pages

- **What's New**: 본 논문은 최근의 생성적 추론(Generative Reasoning) 혁신이 대형 언어 모델(LLMs)의 복잡한 문제 해결 방식을 변화시킨 내용을 다룹니다. 예를 들어, DeepSeek-R1, OpenAI의 o1 & o3, GPT-4o 모델과 같은 것을 포함하여, 2023-2025년 사이에 발표된 상위 27개의 LLM 모델을 종합적으로 분석하였습니다. 논문에서는 일반 훈련 접근법, 믹스처 오브 엑스퍼트(Mixture-of-Experts, MoE), 정보 검색 증강 생성(Retrieval-Augmented Generation, RAG) 등을 포함한 다양한 훈련 방법론을 소개합니다.

- **Technical Details**: 모델을 정제하고 성능을 향상시키기 위한 방법으로는 인퍼런스 타임 스케일링(Inference-time scaling), 강화 학습(Reinforcement Learning), 수퍼바이즈드 파인튜닝(Supervised Fine-tuning) 및 증류(Distillation) 등이 포함됩니다. 이 논문은 LLM의 훈련 방법론을 카테고리별로 나누고, 골수 디자인 혁신 및 테스트 타임 컴퓨팅 스케일링(Test-time Compute Scaling)과 같은 요소도 고려하고 있습니다. 이러한 방법론을 통해 복잡한 과제에서의 투명한  다단계 추론을 생성할 수 있는 방향을 모색합니다.

- **Performance Highlights**: 우리가 분석한 선정된 LLM 모델들은 고급 수학 및 코딩 문제들과 같은 복잡한 작업에서 성능이 향상되었습니다. 특히, OmegaPRM 기법을 사용하여 150만 개 이상의 고품질 과정 주석을 자동으로 수집하는 알고리즘이 주목받고 있습니다. 이러한 접근을 통해 MATH500 및 GSM8K와 같은 벤치마크에서 성능을 크게 향상시킬 수 있음을 보여줍니다.



### InfoBid: A Simulation Framework for Studying Information Disclosure in Auctions with Large Language Model-based Agents (https://arxiv.org/abs/2503.22726)
Comments:
          AAAI 2025 Workshop: Economics of Modern ML: Markets, Incentives, and Generative AI

- **What's New**: 이번 연구에서 소개하는 InfoBid 프레임워크는 대형 언어 모델(LLMs) 기반의 에이전트를 활용하여 온라인 광고 경매에서 정보 공개 전략의 효과를 탐구하는 새로운 접근 방식을 제공합니다. 이는 정보 비대칭 및 신호 전략의 연구에 있어 중요한 혁신입니다. 특히, 기본적인 가정에 의존하지 않고 인간과 유사한 추론 능력을 갖춘 에이전트를 통해 경매 메커니즘을 심층적으로 분석할 수 있는 가능성을 열어줍니다.

- **Technical Details**: InfoBid 프레임워크는 다양한 정보 공개 전략의 영향을 연구하는 데 필요한 다양한 기능을 갖추고 있으며, GPT-4o를 활용하여 각기 다른 정보 구조를 가진 2위 경매의 시뮬레이션을 구현했습니다. 연구는 에이전트가 공개된 정보와 자신의 지식을 기반으로 합리적인 의사결정을 내리는지를 고려하여 설계되었습니다. 이를 통해 입찰자들이 정보에 따라 조정된 입찰을 나타내는 다양한 신호 전략을 면밀히 분석합니다.

- **Performance Highlights**: 시뮬레이션 결과, 입찰자들은 다양한 신호 전략에 따라 진실한 입찰 방식에서 일치하거나 이탈하는 행동을 보였으며, 정보 공개가 경매 결과에 미치는 영향을 확인했습니다. 특히, 고급 정보를 타겟으로 하는 방법이 경매 수익을 최적화하는 데 가장 효과적이라는 발견이 있었습니다. 이러한 발견은 경매 설계와 경제 이론 간의 연관성을 강화하며, LLM 에이전트의 가능성을 실증적으로 입증합니다.



### CodeScientist: End-to-End Semi-Automated Scientific Discovery with Code-based Experimentation (https://arxiv.org/abs/2503.22708)
Comments:
          98 Pages (13 pages: main paper body; 85 pages: appendix)

- **What's New**: 이번 연구에서는 CodeScientist라는 새로운 자율 과학 발견(ASD) 시스템을 소개합니다. 이 시스템은 기존 코드베이스와 유사한 설계 공간을 탐색하는 한계를 극복하고, 아이디어 생성 및 실험 구성을 유전적 탐색(genetic search)의 형태로 재구성합니다. CodeScientist는 연구 논문과 코드 블록의 조합을 이용하여 자동화된 실험을 수행합니다.

- **Technical Details**: CodeScientist 시스템은 언어 모델을 부르는 것과 같은 도메인 내 일반 작업을 정의하는 코드 블록을 활용하여 연구 아이디어를 생성하는 자동화된 실험을 수백 건 수행합니다. 이 시스템은 기존 연구에서 수행한 평가 방식뿐만 아니라 외부 리뷰와 코드 리뷰, 복제 시도를 포함한 다각적 평가(multi-faceted evaluation)를 통해 발견된 결과들을 검증합니다.

- **Performance Highlights**: 이 시스템을 통해 19개의 발견이 이루어졌으며, 이 중 6개는 최소한의 신뢰성과 혁신성을 갖춘 것으로 평가되었습니다. 발견들은 새로운 작업, 에이전트, 메트릭 및 데이터를 포함하여, 기존의 벤치마크 최적화에서 더 넓은 발견으로의 질적 변화를 제안합니다.



### Enhancing nonnative speech perception and production through an AI-powered application (https://arxiv.org/abs/2503.22705)
- **What's New**: 이 연구는 인공지능(AI)을 활용한 모바일 애플리케이션이 외국어 발음에 미치는 영향을 조사합니다. 기존 연구들이 이해 가능성(comprehensibility)과 명료성(intelligibility)에 초점을 맞춘 반면, 개인 발음 소리 개선은 소홀히 했음을 지적합니다. 이렇듯 개인 발음 소리에 대한 연구의 공백을 메우고자 AI 기반 훈련이 비원어민의 발음 지각과 생산에 미치는 효과를 다룹니다.

- **Technical Details**: 참여자들은 'heed-hid' 대비를 구별할 수 있는 능력을 평가하기 위해 사전 테스트(pretest)를 완료했습니다. 이후 Speakometer 모바일 애플리케이션으로 훈련을 진행하였으며, 이 앱은 영어 모음 녹음 작업과 발음 피드백(pronunciation feedback) 및 연습을 포함하고 있습니다. 사후 테스트(posttest)는 사전 테스트와 유사하게 수행되어 성과의 변화를 측정했습니다.

- **Performance Highlights**: 연구 결과는 훈련 후 비원어민들이 발음을 구별하는 정확도와 목표 대비의 생산에서 유의미한 개선을 보였음을 보여줍니다. 그러나 참여자들은 원어민 수준의 발음 능력에는 도달하지 못했습니다. 이 발견은 AI 기반 애플리케이션이 발음 습득을 촉진하는 데 효과적이며, 교실을 넘어 개인화된 대화형 발음 훈련에 활용될 가능성을 지원합니다.



### Bridging Language Models and Financial Analysis (https://arxiv.org/abs/2503.22693)
Comments:
          28 pages

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 자연어 처리에서 혁신적인 가능성을 열었으며, 특히 금융 부문에서 큰 변화를 이끌고 있습니다. 전통적인 방법이 다루기 어려운 복잡한 텍스트, 수치표, 시각적 차트를 포함한 금융 데이터를 효과적으로 처리하고 분석하는 새로운 경로를 제공하고 있습니다. 그러나 LLM의 신기술이 금융 산업에서 실질적으로 사용되기까지는 여전히 큰 갭이 존재하며, 이에 대한 종합적인 검토가 필요합니다.

- **Technical Details**: LLM은 대규모 텍스트 데이터를 처리하고, 긴 문맥에서도 미세한 맥락을 이해하며, 복잡한 추론 작업을 수행하는 데 뛰어난 능력을 보이고 있습니다. 특히 금융 분야에서 LLM의 활용은 요구되는 전문 지식과 데이터 분석의 복잡성 덕분에 더욱 주목받고 있습니다. 이 설문 조사는 데이터 세트와 모델 두 가지 측면에 중점을 두어 LLM의 현재 활용 현황을 검토하고, 향후 금융 분야에서의 적용 가능성을 탐구합니다.

- **Performance Highlights**: LLM을 활용한 금융 데이터의 분석과 처리는 주요 작업으로는 텍스트 분류, 정보 추출, 텍스트 요약 및 질문 응답을 포함합니다. 이러한 작업들은 각각 금융 텍스트를 분류하고, 구조화된 정보를 추출하며, 긴 문서를 압축하고, 복잡한 질문에 응답하는 능력을 강화해 주기 때문에 중요한 의미를 갖습니다. 이 논문은 LLM이 금융 분야에서 갖춘 잠재력을 강조하며, 향후 연구 방향과 LLM의 혁신적인 응용 가능성을 제시합니다.



### Enhancing Aviation Communication Transcription: Fine-Tuning Distil-Whisper with LoRA (https://arxiv.org/abs/2503.22692)
Comments:
          14 pages, 4 Figures, 4 Tables, Under review by Journal of Aerospace Information Systems

- **What's New**: 본 논문은 항공 통신의 문서화(Transcription)에 있어 최신 인공지능 기술을 적용하여 정확성을 향상시키기 위한 연구입니다. 특히 OpenAI의 Whisper 모델을 항공 통신에 맞게 미세 조정(Fine-tuning)하는 방법을 다룹니다. 이를 통해 효율적으로 Whisper의 한 버전인 distil-Whisper를 미세 조정하는 Parameter-Efficient Fine-tuning 방법인 Low-Rank Adaptation을 활용했습니다.

- **Technical Details**: 이 연구에서는 약 70시간 분량의 항공 교통 통제 데이터세트(Air Traffic Control Corpus)를 사용하여 실험을 진행했습니다. 또한 LoRA(Low-Rank Adaptation)의 하이퍼파라미터를 설정하기 위해 그리드 서치(Grid Search) 및 5-겹 교차 검증(5-fold Cross-validation)을 적용했습니다. 이 과정에서 Alpha = 64 및 Rank = 32를 초기 하이퍼파라미터로 설정하고 최적의 조합을 찾아냈습니다.

- **Performance Highlights**: 미세 조정 과정 후, 모델의 평균 단어 오류율(Word Error Rate)은 3.86%로 측정되어 매우 우수한 성능을 보였습니다. 이 결과는 항공기의 조종실(Cockpit)에서의 적용 가능성을 입증해 주며, 논문이 제시하는 방법론이 향후 항공 통신의 효율성을 높일 수 있음을 시사합니다.



### ActionStudio: A Lightweight Framework for Data and Training of Large Action Models (https://arxiv.org/abs/2503.22673)
Comments:
          15 pages; large action models; xLAM

- **What's New**: 이번 논문에서는 Action models의 중요성을 강조하며, 이를 통한 자율 에이전트의 복잡한 작업 수행 가능성을 설명합니다. ActionStudio라는 새로운 데이터 및 훈련 프레임워크를 소개하였으며, 이는 대규모 Action models을 위한 경량화되고 확장 가능한 솔루션입니다. 기존 인프라의 한계를 극복하고 다양한 훈련 패러다임을 지원하는 기능을 갖추고 있습니다.

- **Technical Details**: ActionStudio는 다양한 에이전트의 궤적을 표준화된 포맷으로 통합하고, LoRA, 전체 파인튜닝(full fine-tuning), 분산(distributed) 환경을 포함한 다양한 훈련 패러다임을 지원합니다. 또한, 강력한 전처리(preprocessing) 및 검증(verification) 도구를 통합하여 데이터 관리의 효율성을 높입니다. 이를 통해 에이전트 특정(fine-tuning) 훈련을 보다 쉽게 수행할 수 있습니다.

- **Performance Highlights**: 우리는 공용 및 실제 산업 벤치마크에서 ActionStudio의 효과를 검증하였으며, 강력한 성능과 실용적인 확장성(practical scalability)을 입증하였습니다. 연구 커뮤니티를 지원하기 위해 코드 및 데이터는 오픈 소스로 제공하고 있습니다.



### The Mathematical Relationship Between Layer Normalization and Dynamic Activation Functions (https://arxiv.org/abs/2503.21708)
Comments:
          New title, renamed DyISRU, added missing parentheses in proof of theorem 3, minor language corrections

- **What's New**: 최근 연구에서 제안된 Dynamic Tanh (DyT)는 layer normalization (LN)을 대체할 수 있는 방법입니다. 이 접근법은 실제적으로 유용하지만, 이론적인 근거가 부족하였습니다. 본 논문에서는 LN과 동적 활성화 함수 간의 수학적 관계를 규명하고, DyT를 LN에서 유도하는 방법을 제시하고 있어 이론적 이해를 심화하고 있습니다.

- **Technical Details**: DyT 함수는 LN의 특정 수학적 유도를 통해 개발되며, 이 과정에는 미분 방정식을 해결하는 단계가 포함됩니다. 연구자들은 LN의 입력에 대한 미분을 계산하고, 이를 단순화하여 DyT 함수를 도출하였습니다. 이러한 과정에서 정밀한 근사가 필요하다는 것을 발견하였으며, 이를 제거함으로써 Dynamic Inverse Square Root Unit (DyISRU)라는 대체 기능을 제안했습니다.

- **Performance Highlights**: DyISRU는 layer normalization의 정확한 대응 개념으로, 수치적으로 DyT보다 LN에 더 정확히 유사하다는 것을 증명했습니다. 이 연구는 변동성(variance) 가정에서 벗어나 새로운 요소별 변환을 제공함으로써 layer normalization을 대체할 가능성을 제시하고 있습니다. DyT는 사전 조정이 필요할 수 있지만, DyISRU는 훨씬 더 안정적인 성능을 기대할 수 있는 장점이 있습니다.



New uploads on arXiv(cs.IR)

### Rec-R1: Bridging Generative Large Language Models and User-Centric Recommendation Systems via Reinforcement Learning (https://arxiv.org/abs/2503.24289)
- **What's New**: Rec-R1은 일반 강화 학습(framework) 구조로, 대형 언어 모델(LLMs)과 추천 시스템을 밀접하게 연계하는 방식으로 작동합니다. 이 방법은 기존의 유도(prompting)와 감독 세부 조정(SFT) 방식과는 달리, 고정된 블랙박스 추천 모델의 피드백을 직접 활용하여 LLM 생성을 최적화합니다. Rec-R1은 데이터 증류(data distillation)에 필요한 막대한 비용과 노력을 피할 수 있도록 설계되었습니다.

- **Technical Details**: Rec-R1은 사용자 질의(user query)나 행동 기록(behavioral history)과 같은 추천 관련 입력을 받아 LLM이 텍스트 출력을 생성하도록 합니다. 이 출력은 하위 추천 모델에 의해 사용되고 성능 기반 평가(performance-based evaluation)를 통해 품질이 평가됩니다. LLM은 지속적인 상호작용을 통해 추천 시스템의 목표에 더 잘 부합하는 입력을 생성하는 방법을 배우며, 이를 통해 추천 성능이 개선되도록 합니다.

- **Performance Highlights**: Rec-R1은 두 가지 대표적인 추천 시나리오인 제품 검색(product search)과 순차 추천(sequential recommendation)에서 효과성을 입증하였습니다. 실험 결과, Rec-R1은 기존의 방법들보다 일관되게 우수한 성능을 보여주었으며, 특히 사용자 프로필 정보가 없는 상황에서도 강력한 성능을 유지했습니다. 또한, Rec-R1은 대형 언어 모델의 일반적 능력을 유지하며, 추천 성능과 지시 따르기를 모두 개선하였습니다.



### Combining Query Performance Predictors: A Reproducibility Study (https://arxiv.org/abs/2503.24251)
- **What's New**: 이 논문은 Query Performance Prediction(QPP)에 대한 연구를 재조명하며, Hauff et al.(2009)의 연구를 바탕으로 새로운 예측 방법, 평가 메트릭 및 데이터 세트를 통한 재현 가능성을 평가합니다. 이전 연구를 확장하여, 감독된 신경 기법을 포함한 포스트 리트리벌(post-retrieval) 방법을 고려하고, sMARE평가 메트릭을 사용하며, Clueweb09B 및 TREC DL과 같은 추가 데이터 세트로 실험을 수행했습니다.

- **Technical Details**: QPP는 정보 검색(IR)의 중요한 분야로, 쿼리가 주어졌을 때 검색 엔진의 성능을 추정하는 데 초점을 맞춥니다. 다양한 QPP 방법이 제안되었으며, 이들은 일반적으로 프리 리트리벌(pre-retrieval) 및 포스트 리트리벌(post-retrieval) 기술로 분류됩니다. 새로운 연구에서는 sMARE를 사용하는 것 외에도 다양한 데이터 세트를 통해 (예: ClueWeb09B, MS MARCO) 더 포괄적인 평가를 제공합니다.

- **Performance Highlights**: 연구 결과는 이전의 주장들을 상당수 지지하며 동시에 흥미로운 발견도 제시합니다. 특히, QPP 방법 간의 상관관계를 보다 미세하게 살펴보아 서로 다른 정보가 포착되거나 중복된 요소에 의존하는지를 분석하였습니다. 이러한 발견은 QPP 방법의 조합이 Average Precision(AP)과의 상관관계를 일관되게 증대시키는지 여부를 재검토하는 기초를 제공합니다.



### Text2Tracks: Prompt-based Music Recommendation via Generative Retrieva (https://arxiv.org/abs/2503.24193)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전 덕분에 사용자들은 자연어를 통해 매우 구체적인 음악 추천 요청을 할 수 있게 되었습니다. 그러나 기존의 추천 방식에는 일반적인 토큰화 방식이 드러나는 여러 제한점이 있었습니다. 본 논문에서는 이러한 문제를 해결하기 위해 Text2Tracks라는 새로운 생성형 검색 모델을 제안하고, 사용자의 음악 추천 프롬프트에 직접적으로 연관된 트랙 ID를 매핑하는 방식을 도입하였습니다.

- **Technical Details**: Text2Tracks는 프롬프트 기반의 음악 추천 작업을 생성형 검색 문제로 규정합니다. 이 모델은 쿼리에서 연관된 트랙을 마치 텍스트를 생성하듯이 직접적으로 ID를 생성합니다. 실험을 통해, 의미론적 ID를 학습하는 전략이 아티스트 이름이나 트랙 제목을 사용하는 기존 방법보다 48% 더 효과적임을 보여주며, 디코딩 단계를 약 7.5배 줄이는 성과를 달성하였습니다.

- **Performance Highlights**: Text2Tracks는 프롬프트 기반 음악 추천 작업에서 기존의 희소(sparse) 및 밀집(dense) 검색 솔루션을 초월하는 성능을 보였습니다. 제안된 접근 방식은 사용자 언어 프롬프트와 관련성 있는 트랙을 정확하게 찾는 데 있어 획기적인 개선을 이루어냈으며, 모델 학습 방법론에서 새로운 가능성을 제시합니다.



### On the Reproducibility of Learned Sparse Retrieval Adaptations for Long Documents (https://arxiv.org/abs/2503.23824)
Comments:
          This is a preprint of our paper accepted at ECIR 2025

- **What's New**: 최근의 연구에서는 긴 문서에 대한 Learned Sparse Retrieval (LSR)의 적응 기전 을 재현하고 검토하는 데 초점을 맞추고 있습니다. 기존의 방법인 ExactSDM과 SoftSDM을 포함하여 다양한 세그먼트 집계 전략의 효과를 비교하고, LSR을 긴 문서에 적용하여 성과를 분석하는 것을 목적으로 합니다. 이를 통해 문서 검색의 성능을 향상시키기 위한 중요한 발견과 통찰을 제공합니다.

- **Technical Details**: LSR 기법은 쿼리와 문서 각각에 대해 스파스 벡터를 생성한 후, 이들 간의 내적(dot product)을 계산하여 스코어를 산출합니다. 쿼리 인코더 fQ와 문서 인코더 fD를 사용하여 쿼리 q와 문서 d의 표현 간의 점수를 산출하며, 이는 전통적인 sparse retrieval 기법과 유사한 고차원 스파스 벡터에 기반합니다. LSR은 통계적인 정보에 의존하기 보다는 학습을 통해 용어 가중치를 결정하는 접근 방식을 사용합니다.

- **Performance Highlights**: 본 연구는 긴 문서에 대한 LSR의 적응 가능성을 평가하며, 특히 Segment와 Term Frequency의 영향력을 분석합니다. 실험을 통해 문서 내의 특정 세그먼트가 쿼리-문서 관련성에 미치는 영향을 확인하고, Score-Max와 SDM의 집계 능력이 데이터셋 및 세그먼트 변화에 대해 견고함을 보임을 증명합니다. 이러한 결과는 LSR 기반의 문서 검색 모델의 효율성과 정확성을 높이는 데 기여할 것입니다.



### Finding Interest Needle in Popularity Haystack: Improving Retrieval by Modeling Item Exposur (https://arxiv.org/abs/2503.23630)
Comments:
          2 pages

- **What's New**: 본 논문에서는 추천 시스템의 지속적인 피드백 루프 속에서 발생하는 인기 편향(popularity bias) 문제를 해결하기 위한 새로운 접근법을 소개합니다. 기존 방법들과 달리, 이 연구는 실시간으로 노출 동역학(exposure dynamics)을 조절할 수 있는 점에서 차별성이 있습니다. 특히, 노출 확률(exposure probability)을 모델링하여 검색 단계에서의 순위를 조정하는 방법을 제안합니다.

- **Technical Details**: 우리가 제안하는 접근법은 노출 효과(exposure effects)와 참여 가능성(engagement likelihood)을 분리하여, 대규모 추천 플랫폼에서 공정성(fairness)과 참여도(engagement) 간의 조절 가능한 균형을 이룹니다. 실제 비디오 추천 시스템에서의 온라인 A/B 실험을 통한 검증을 실시하였습니다. 이 과정에서 노출 점수(retrieval scoring) 접근법이 추천 알고리즘의 성능을 어떻게 개선할 수 있는지를 보여주었습니다.

- **Performance Highlights**: 이 방법은 고유하게 검색된 항목(unique retrieval items)의 수를 25% 증가시키고, 과도한 인기 콘텐츠의 지배력을 40% 낮추는 결과를 얻었습니다. 또한, 사용자 참여 수준을 유지하면서 이러한 성과를 달성하여, 인기 편향을 완화하기 위한 확장 가능하고 배포 가능한 솔루션을 제시하게 되었습니다.



### Understanding Visual Saliency of Outlier Items in Product Search (https://arxiv.org/abs/2503.23596)
- **What's New**: 이번 연구는 사용자 주목에 따라 공급자의 수익이 결정되는 양면 시장에서 아이템의 노출(item exposure)이 필수적임을 강조합니다. 특히, 아웃라이어(Outlier) 아이템의 존재가 기타 아이템의 노출에 미치는 영향을 탐구합니다. 기존 연구에서 다루지 않았던 두 가지 질문에 집중하여, 시각적 특성과 인지적 요소가 아웃라이어에 대한 사용자 인식에 미치는 영향을 분석합니다.

- **Technical Details**: 연구는 시각적 주목도 모델(visual saliency models)을 사용하여 시각적 속성에 따른 아웃라이어 감지 능력을 평가합니다. 각각의 아웃라이어가 어떻게 사용자의 주목을 끌게 되는가를 밝히기 위해, 타겟(item)을 비타겟(non-outlier)과 구분하며 반응 시간(reaction time)과 정확도를 측정하는 실험을 수행했습니다. 특히, 아웃라이어 아이템이 일반 아이템 대비 사용자와 더 많은 상호작용을 유도하는지에 대한 탐색적 분석이 이루어졌습니다.

- **Performance Highlights**: 실험 결과, 시각적 주목도 모델이 색상 대비(color contrast) 및 모양 등 하향(bottom-up) 요인을 감지하는 데 효과적임을 확인했습니다. 아이템 목록에서 아웃라이어가 더 오랜 시간 동안 사용자의 주목을 끌며, 이는 판매에 긍정적인 영향을 미칠 수 있음을 시사합니다. 비주얼 트래킹 실험에서는 아웃라이어 아이템과 그 근처 품목이 더 빠르고 오랜 시간 동안 주목받음을 보여 상향(top-down) 요인의 중요성을 강조합니다.



### Filtering with Time-frequency Analysis: An Adaptive and Lightweight Model for Sequential Recommender Systems Based on Discrete Wavelet Transform (https://arxiv.org/abs/2503.23436)
- **What's New**: 이 논문은 Discrete Wavelet Transform (DWT)를 활용한 새로운 Sequential Recommender System (SRS) 모델인 DWTRec을 제안합니다. 기존 Transformer 기반의 모델이 자주 고주파 정보를 간과하는 문제를 해결하기 위해 DWT를 적용하여 사용자 관심사를 효과적으로 분석합니다. DWTRec은 시간과 주파수에서 사용자 관심사의 신호를 분리하여 높은 성능을 발휘하는 모델입니다.

- **Technical Details**: DWTRec은 고주파 및 저주파 정보를 모두 처리할 수 있는 적응형 시간-주파수 필터를 기반으로 설계되었습니다. 이 필터는 사용자 관심사의 다양한 신호를 자동으로 학습하며, 신호를 시간 및 주파수에서 재구성하여 노이즈와 유용한 정보를 구분합니다. DWT는 특히 지역적인 주파수 분해를 가능하게 하여 단기적인 급변 관심사를 잘 반영할 수 있는 장점을 가지고 있습니다.

- **Performance Highlights**: DWTRec은 다양한 도메인과 데이터셋에 대한 실험 결과, 기존의 최첨단 모델보다 우수한 성능을 보였습니다. 특히 시퀀스의 길이가 늘어날수록 성능이 더욱 향상되는 것을 확인할 수 있으며, 이는 단기적인 급변 관심사를 효과적으로 포착할 수 있음을 의미합니다. 실험 결과, DWTRec은 SOTA 모델들에 비해 시간 복잡도와 공간 복잡도 측면에서도 우수함을 입증하였습니다.



### LIRA: A Learning-based Query-aware Partition Framework for Large-scale ANN Search (https://arxiv.org/abs/2503.23409)
Comments:
          This paper is accepted by WWW 2025

- **What's New**: 본 논문에서는 인접 이웃 검색(Nearest Neighbor Search)의 비효율성을 개선하기 위한 새로운 프레임워크, LIRA를 제안합니다. LIRA는 LearnIng 기반의 queRy-aware pArtition 프레임워크로, 쿼리와 관련된 파티션만을 직접 탐색하여 불필요한 탐색을 줄이고 쿼리 인지 접근을 허용합니다. 기존의 파티션 기반 탐색 방식에서 발생하는 여러 한계를 해결하기 위해 설계되었습니다.

- **Technical Details**: LIRA의 핵심 기술은 쿼리의 kNN(가장 가까운 이웃)로 포함되는 파티션을 직접 탐색하는 모델입니다. 이 접근 방식은 nprobe(탐색 파티션의 수)를 개별적으로 설정하여 각 쿼리에 적합하게 최적화합니다. 또한, 쿼리 전후 파티션 탐색의 비효율성을 줄이기 위해 학습 기반의 중복성 전략을 통합하여 긴 꼬리 분포(이웃이 많이 흩어져 있는 분포)가 검색 효율성에 미치는 부정적 영향을 완화합니다.

- **Performance Highlights**: 실제 벡터 데이터셋에 대한 광범위한 실험 결과, LIRA는 정확도(accuracy), 지연(latency), 쿼리 팬 아웃(query fan-out) 간의 트레이드오프에서 기존 방법들에 비해 우수함을 입증하였습니다. LIRA는 특히 데이터셋의 차원이 높아질수록 탐색 효율성을 크게 개선하였습니다. 이와 같은 성과를 통해 LIRA는 대규모 구조가 없는 데이터 검색에 있어 효율적인 솔루션으로 자리매김할 것으로 기대됩니다.



### RuleAgent: Discovering Rules for Recommendation Denoising with Autonomous Language Agents (https://arxiv.org/abs/2503.23374)
Comments:
          11 pages, 4 figures

- **What's New**: 이번 논문에서는 사용자 피드백의 노이즈 문제를 해결하기 위한 새로운 접근법인 RuleAgent를 소개합니다. RuleAgent는 언어 에이전트 기반의 프레임워크로, 사용자 데이터를 분석하여 추천 시스템의 노이즈를 자동으로 제거할 수 있는 규칙을 발견합니다. 기존의 수동 규칙 작성 방식과의 차별점은, RuleAgent가 신속하게 규칙을 발견하고 다양한 상황에 적응할 수 있도록 설계되었다는 점입니다.

- **Technical Details**: RuleAgent는 프로파일 모듈, 메모리 모듈, 계획 모듈, 액션 모듈 등 네 가지 핵심 모듈로 구성되어 있습니다. 이러한 모듈은 규칙 발견을 지원하며, 반사 메커니즘을 통해 에이전트의 추론 능력을 향상시킵니다. 또한, LossEraser라는 전략을 도입하여 에이전트가 불량 샘플의 영향을 '지우는' 방식으로 리트레이닝 필요성을 줄입니다.

- **Performance Highlights**: 실험 결과, RuleAgent는 세 가지 데이터셋에서 기존의 최첨단 노이즈 제거 모델보다 더욱 뛰어난 추천 성능을 보여주었습니다. 이는 RuleAgent의 규칙 발견이 실제 데이터와 모델의 동적 상호작용 덕분임을 확인시킵니다. 이러한 성과는 데이터 청소 작업의 효율성을 크게 향상시키며, 규칙의 일반화 가능성을 제공합니다.



### Graph-Structured Driven Dual Adaptation for Mitigating Popularity Bias (https://arxiv.org/abs/2503.23358)
- **What's New**: 이 논문은 추천 시스템에서 인기 편향(popularity bias)의 문제를 해결하기 위해 그래프 구조적 이중 적응 프레임워크(Graph-Structured Dual Adaptation Framework, GSDA)를 제안하고 있습니다. GSDA는 그래프의 인접 행렬(adjacency matrix)에서 구조적 및 분포적 특성을 포착하여, 추천 품목의 인기와 비인기 사이의 불균형을 완화합니다. 이 approach는 기존의 감독적 정렬(supervised alignment) 방법들이 깊은 레이어에서의 임베딩 동질화(over-smoothing)를 고려하지 못하는 한계를 극복합니다.

- **Technical Details**: GSDA는 계층적 적응 정렬 메커니즘과 분포 인식(dynamic contrast weighting) 동적 가중치 전략을 통해 두 가지 주요 요소를 결합합니다. 계층적 메커니즘은 각 레이어의 가중치 감소를 조정하여 깊은 레이어에서도 조건부 엔트로피(conditional entropy) 감소 효과를 억제합니다. 분포 인식 전략은 실시간 지니 계수(Gini coefficient)에 따라 변경 가능하며, 고정된 하이퍼 파라미터 없이도 적응성을 확보합니다.

- **Performance Highlights**: GSDA는 3개의 벤치마크 데이터셋에서 실험을 통해 기존 최첨단 추천 방법들보다 추천 정확도에서 일관되게 우수한 성능을 보여주며 인기 편향 문제를 효과적으로 완화합니다. 이 논문은 GCN의 깊은 레이어에서의 감독적 정렬의 효과가 현저히 감소하며, 이를 극복하기 위한 새로운 접근 방식을 제시함으로써 추천 시스템의 성능 향상에 기여하고 있습니다.



### Beyond Unimodal Boundaries: Generative Recommendation with Multimodal Semantics (https://arxiv.org/abs/2503.23333)
- **What's New**: 이번 논문에서는 Multimodal Generative Recommendation (MGR)이라는 새로운 접근 방식을 제안합니다. 기존의 Generative Recommendation (GR) 방법들이 주로 unimodal 데이터를 사용하였던 한계를 극복하여, 다양한 모달리티를 통합하는 방법론을 다룹니다. 저자들은 특히 모달리티 선택의 중요성과 그것이 GR 모델의 성능에 미치는 영향을 강조하고 있습니다.

- **Technical Details**: MGR-LF++라는 새로운 Late Fusion 프레임워크를 도입하여, 서로 다른 모달리티 정보를 효과적으로 관리하는 방법을 제안합니다. 이 프레임워크는 contrastive modality alignment 훈련 기법과 각 모달리티를 구분하는 특별한 토큰을 사용하여, 서로 다른 semantic IDs의 일치를 도모합니다. 이를 통해, 다양한 모달리티의 정보를 손실 없이 통합할 수 있는 방법을 모색합니다.

- **Performance Highlights**: MGR-LF++는 기존의 unimodal 접근 방법 대비 20% 이상의 성능 향상을 달성하였습니다. 저자들은 6개의 기준선 모델을 사용하여 3개 데이터셋에서 실험을 실시하였으며, 그 결과 다중 모달리티 정보를 활용하는 것이 Generative Recommendation의 효과를 크게 향상시킬 수 있음을 입증했습니다.



### Beyond Contrastive Learning: Synthetic Data Enables List-wise Training with Multiple Levels of Relevanc (https://arxiv.org/abs/2503.23239)
Comments:
          Code: this https URL

- **What's New**: 이 논문에서는 정보 검색(Information Retrieval, IR)에서 큰 언어 모델(LLM)을 활용하여 기존의 훈련 방식을 개선하는 SyCL (Synthetic ranking Context for List-wise training) 방법을 제안합니다. SyCL은 실제 문서를 사용하지 않고도 여러 수준의 관련성을 가진 합성 문서를 생성하여 IR의 효율성을 극대화합니다. 이를 통해 기존의 이진 라벨로 한정된 훈련 방식을 뛰어넘고, 더 복잡한 문서 순위 매기기를 가능하게 합니다.

- **Technical Details**: 제안된 SyCL 방법은 오픈 소스 LLM을 활용하여 MS MARCO 데이터셋에 대한 질의에 따라 네 가지 다른 관련성 수준을 가진 전방위 대량 합성 문서를 생성합니다. 이 문서들은 Wasserstein Distance를 손실 함수로 사용하여 훈련 중 상대적 라벨 불일치를 반영하여 모델의 점수 선택을 다르게 패널티합니다. SyCL은 대규모 IR 데이터셋(~2M 샘플)을 생성하며, 이로 인해 복잡한 훈련 파이프라인을 피하고 데이터 품질 문제를 완화합니다.

- **Performance Highlights**: SyCL을 사용한 실험 결과, 제안된 방법은 InfoNCE 기반의 전통적인 훈련 방식에 비해 성능이 현저히 향상됨을 보여줍니다. BEIR 데이터셋의 제로샷 평가에서 SyCL은 36.8에서 43.2로 평균 nDCG@10 점수를 개선하여, 실제 라벨이 있는 문서로 훈련된 모델과 유사한 성능을 달성하였습니다. 이 결과는 실제 문서 없이도 강력한 순위 매기기 성능을 구현할 수 있음을 잘 보여줍니다.



### Reproducibility Companion Paper:In-processing User Constrained Dominant Sets for User-Oriented Fairness in Recommender Systems (https://arxiv.org/abs/2503.23040)
Comments:
          4 pages

- **What's New**: 이 논문은 이전 연구인 "In-processing User Constrained Dominant Sets for User-Oriented Fairness in Recommender Systems"의 실험 결과를 재현하여 방법의 효과성을 검증하고, 재현성이 있는 연구 수행에 대한 지침을 제공합니다. 논문에서는 전처리된 데이터셋, 소스 코드의 구조, 구성 파일 설정, 실험 환경, 재현된 실험 결과들에 대한 구체적인 설명을 제시합니다. 이러한 접근 방식은 추천 시스템의 공정성을 확보하는 데 기여하고 있습니다.

- **Technical Details**: 새로운 공정성 인식 훈련 프레임워크인 In-processing User Constrained Dominant Sets (In-UCDS)를 제안하여, 열악한 사용자(불리한 사용자)의 학습을 개선합니다. In-UCDS는 (1) UCDS 모델링 및 (2) 프로세싱 훈련의 두 가지 주요 단계로 구성됩니다. 이러한 방식은 추천 성능을 저해하지 않으면서도 공정성을 반영한 훈련을 가능하게 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 In-UCDS 프레임워크의 효과를 입증하였습니다. NeuMF, VAECF, PMF, NGCF를 기본 추천 모델로 사용하여 여러 방법과 성능을 비교하며, NDCG와 F1-score 같은 평가 지표를 활용하여 모델 성능을 평가했습니다. 결과적으로, Advantaged와 Disadvantaged 사용자 간의 추천 성능 격차를 측정하는 μO𝑓을 통해 더욱 공정한 추천 성능을 달성할 수 있음을 보여주었습니다.



### Imagine All The Relevance: Scenario-Profiled Indexing with Knowledge Expansion for Dense Retrieva (https://arxiv.org/abs/2503.23033)
Comments:
          9 pages

- **What's New**: SPIKE (Scenario-Profiled Indexing with Knowledge Expansion)은 문서를 시나리오 기반 검색 단위로 분해하여 암묵적 관련성을 명시적으로 인덱싱하는 새로운 밀집 검색 프레임워크입니다. 이 접근법은 문서와 가정적 정보 요구 간의 암묵적 관계를 밝히기 위해 필요한 추론 과정을 캡슐화한 시나리오를 통해 문서를 조직합니다. 이러한 방법은 사용자의 검색 경험을 향상시키고, 검색 보강 생성(РAG)에서 LLM에 유용한 맥락 정보를 제공합니다.

- **Technical Details**: SPIKE는 LLM(대형 언어 모델)을 사용하여 시나리오 보강 데이터셋을 구성하고, 이를 통해 효율적인 시나리오 생성기를 추출하여 최적화하는 과정으로 이루어집니다. 검색 과정에서 SPIKE는 문서 수준의 관련성과 함께 시나리오 수준의 관련성을 통합하여 암묵적 관련성을 탐지하는 데 중점을 둡니다. 각 문서에 대해 특정 정보 요구를 충족시킬 수 있는 시나리오를 다수 생성함으로써, 기존 밀집 검색 모델이 다루지 못하는 정보를 효과적으로 탐색할 수 있게 합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, SPIKE는 다양한 쿼리 유형과 밀집 검색기 전반에서 검색 성능을 일관되게 향상시킵니다. 또한 SPIKE는 사용자에게 의미 있는 설명을 제공하여 검색 결과를 이해하기 쉽게 만들고, RAG 환경에서 LLM이 더 정확한 답변을 생성할 수 있도록 도와줍니다. 이러한 기능들은 기존의 검색 시스템이 다루기 어려운 틈새를 보완하며, 검색의 효율성을 크게 높입니다.



### Reproducibility Companion Paper: Making Users Indistinguishable: Attribute-wise Unlearning in Recommender Systems (https://arxiv.org/abs/2503.23032)
- **What's New**: 이번 논문에서는 이전 연구 "Making Users Indistinguishable: Attribute-wise Unlearning in Recommender Systems"에서 제시한 실험 결과를 reproducible하게 재현하는 방법을 설명합니다. 본 연구의 목표는 기존 방법의 유효성을 검증하고 다른 연구자들이 결과를 재현할 수 있도록 돕는 것입니다. 데이터셋, 소스코드 구조, 설정 파일, 실험 환경 및 재현된 실험 결과를 상세히 설명합니다.

- **Technical Details**: 본 연구는 추천 시스템에서 사용자의 민감한 속성을 보호하기 위해 Attribute Unlearning (AU) 기술을 적용합니다. 특히 Post-Training Attribute Unlearning (PoT-AU) 설정을 엄격하게 검토하며, 이를 위해 추천 성능과 unlearning 성능의 균형을 위해 두 가지 구성 요소로 이루어진 손실 함수 (loss function)를 설계합니다. U2U(User-to-User)와 D2D(Distribution-to-Distribution) 측정을 통해 실험에서 제안된 방법의 효과를 검증합니다.

- **Performance Highlights**: 세 가지 실제 데이터셋을 사용하여 실험을 수행하였으며, 추천 성능과 unlearning 성능, 효율성을 평가합니다. 각 추천 모델의 효과성을 NMF와 LightGCN을 통해 분석하며, 다양한 평가 지표(Accuracy, Precision, Recall, AUC)를 사용하여 제안된 방법의 성능을 기존 방법들과 비교합니다. 사용자 임베딩 분포의 변화를 분석하여 제안된 방법의 메커니즘을 이해하도록 돕습니다.



### Federated Semantic Learning for Privacy-preserving Cross-domain Recommendation (https://arxiv.org/abs/2503.23026)
- **What's New**: 본 논문은 접근성과 효과를 가진 개인정보 보호(Cross-Domain Recommendation, CDR) 방법 개발을 목표로 하고 있습니다. 특히, 기존 방법들이 가지고 있는 한계를 극복하기 위해 연합 학습(federated learning)과 고급 의미학적 학습(federated semantic learning)을 통합한 새로운 접근 방식을 제시합니다. 본 연구의 주요 기여는 각 도메인에서 서로 다른 아이템에 대한 심층적 의미 표현을 활용하여 사각 이웃 관계를 해결하는 것입니다.

- **Technical Details**: 제안된 FFMSR 프레임워크는 다층 언어 모델(Pre-trained Language Model, PLM)과 전문가 혼합 기법(Mixture of Experts, MoE)을 사용하여 아이템의 원본 설명 텍스트에서 다층 의미 인코딩을 모델링합니다. 또한 아이디(ID) 모달리티를 텍스트 모달리티와 결합하여 아이템의 다양한 측면을 모델링합니다. FFT(Fast Fourier Transform) 기반의 필터를 적용하여 불필요한 의미 정보를 효과적으로 제거합니다.

- **Performance Highlights**: 제안된 FFMSR 방법은 두 개의 실제 데이터셋에서의 실험 결과를 통해 기존의 최첨단(SOTA) 방법들과 비교했을 때 우수한 성능을 보였습니다. 연구진의 결과는 제안된 방법이 개인정보 보호를 유지하면서도 의미 기반 정보를 효과적으로 활용할 수 있음을 보여줍니다. 이러한 접근은 사용자의 개인정보를 보호하면서도 추천 시스템의 성능을 크게 향상시킬 수 있는 가능성을 제시합니다.



### DAT: Dynamic Alpha Tuning for Hybrid Retrieval in Retrieval-Augmented Generation (https://arxiv.org/abs/2503.23013)
- **What's New**: 이번 연구에서는 Retrieval-Augmented Generation (RAG) 시스템에서 정보 검색을 개선하기 위해 DAT(Dynamic Alpha Tuning)라는 새로운 하이브리드 검색 프레임워크를 제안합니다. DAT는 각 쿼리에 대해 dense 검색과 BM25 간의 동적 균형을 맞추며, LLM(대형 언어 모델)을 활용해 두 검색 방법의 상위 결과를 평가하여 효과 점수를 부여합니다. 이를 통해 고정된 가중치 방식의 한계를 극복하고 보다 적응적인 검색 성능을 제공합니다.

- **Technical Details**: 기존의 하이브리드 검색 방식은 쿼리의 다양성을 반영하지 못하는 고정 가중치를 사용합니다. DAT는 쿼리의 특성에 따라 검색 방법의 유효성을 평가하고, 이를 바탕으로 동적으로 가중치를 조정합니다. 이 과정에서 불필요한 계산 부하를 줄이기 위해 각 검색 방법의 상위 결과만을 비교하여 효율적으로 최적의 가중치 α를 계산합니다.

- **Performance Highlights**: DAT는 다양한 평가 지표에서 고정 가중치 방식보다 우수한 성능을 기록했습니다. 특히 도전적인 쿼리에서는 기존 하이브리드 검색 방법이 가지는 한계를 넘어서 많은 이점을 보였으며, 개인 쿼리에 대한 성능 변동성을 줄이는 효과도 입증되었습니다. 이러한 결과는 DAT가 사용자 경험을 보다 일관되게 만들어 준다는 것을 보여줍니다.



### A Systematic Evaluation of LLM Strategies for Mental Health Text Analysis: Fine-tuning vs. Prompt Engineering vs. RAG (https://arxiv.org/abs/2503.24307)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 활용한 정신 건강 텍스트 분석을 위한 세 가지 접근 방식을 체계적으로 비교하였습니다: prompt engineering, retrieval augmented generation (RAG), 그리고 fine-tuning입니다. 이 연구는 LLaMA 3를 사용하여 감정 분류 및 정신 건강 상태 감지 작업을 두 개의 데이터셋에서 평가하였습니다. 연구 결과, fine-tuning이 감정 분류에서 91%, 정신 건강 조건 분류에서 80%의 정확도를 달성하였으며, prompt engineering과 RAG는 보다 유연한 배포가 가능하지만 보통의 성능(40-68% 정확도)을 보여주었습니다.

- **Technical Details**: 정신 건강 텍스트 분석을 위한 LLM의 세 가지 접근 방식은 fine-tuning, prompt engineering, RAG입니다. 특히, fine-tuning은 높은 정확도를 요구하지만 많은 컴퓨팅 리소스와 대규모 훈련셋을 필요로 합니다. 반면, prompt engineering과 RAG는 상대적으로 적은 자원으로 보다 유연한 배포가 가능하게 하며, 다양한 설정에서 효과적으로 구현할 수 있다는 장점이 있습니다.

- **Performance Highlights**: 이 연구는 정신 건강 분야에서 LLaMA 3 기반 모델의 효과를 입증하였으며, 감정 분류 및 정신 건강 상태 분류에서 매우 높은 정확도를 기록했습니다. 이러한 결과는 임상 환경에서 LLM 기반 솔루션의 구현에 있어 중요한 통찰력을 제공합니다. 향후 정신 건강 평가 도구의 개발에 중요한 의미를 가지며, 높은 정확도의 fine-tuning 외에도 prompt engineering과 RAG 접근 방식이 자원과 배포 유연성 면에서 유효한 대안이 된다는 점을 강조하고 있습니다.



### Music Information Retrieval on Representative Mexican Folk Vocal Melodies Through MIDI Feature Extraction (https://arxiv.org/abs/2503.24243)
Comments:
          10 pages, 5 figures, 2 tables

- **What's New**: 이번 연구는 MIDI 기능 추출을 통해 멕시코 민속 보컬 멜로디를 분석하며, ambitus, pitch-class entropy, interval distribution을 살펴봅니다. 연구 결과, ambitus의 값은 8에서 27 세미톤까지 다양하게 나타났으며, 이는 장르 내에서 다양한 작곡 스타일과 보컬 요구를 나타냅니다. 또한, pitch-class entropy 분석을 통해 Armando Manzanero의 'Somos Novios'가 가장 높은 엔트로피를 보이며 복잡한 멜로딕 구조를 제안하고 있습니다.

- **Technical Details**: 연구는 MATLAB과 MIDI Toolbox를 사용하여 데이터를 추출하고 통계 분석을 수행했습니다. MIDI 파일은 Pro Tools를 사용해 정규화한 후 MATLAB에 가져와 ambitus, pitch-class entropy, interval distribution과 같은 기능을 추출하였습니다. 이 과정에서 MATLAB의 시각화 도구와 통계 도구를 활용하여 추출된 기능들을 분석하였습니다. 이를 통해 멕시코 민속 음악의 특징을 포괄적으로 이해할 수 있는 기초를 마련했습니다.

- **Performance Highlights**: 연구에서는 ambitus나 entropy와 Spotify 재생 수 사이에 유의미한 상관관계를 발견하지 못했습니다. 그럼에도 불구하고, 멕시코 민속 음악의 멜로디는 서로 다른 음역과 복잡성을 가지고 있어 해당 장르를 깊이 이해하는 데 기여합니다. MIDI 기능 분석 결과, 멕시코 민속 음악은 접근성 높은 밀접한 간격을 선호하는 작곡 스타일이 특징적임을 나타내고 있습니다.



### Get the Agents Drunk: Memory Perturbations in Autonomous Agent-based Recommender Systems (https://arxiv.org/abs/2503.23804)
- **What's New**: 이번 논문은 사용자 및 아이템 상호작용을 기반으로 한 메모리 메커니즘을 도입한 추천 시스템(Agent4RSs)의 안전 취약점을 공격하는 첫 번째 연구입니다. 특히, Agent4RSs의 메모리를 교란시켜 제한 사항을 발견하고 보안을 강화하는 방법을 제안합니다. 새로운 실용 공격 프레임워크인 DrunkAgent를 활용하여 이러한 공격이 성공적으로 이루어질 수 있도록 합니다.

- **Technical Details**: DrunkAgent는 생성 모듈, 전략 모듈, 대리 모듈로 구성되어 있습니다. 생성 모듈은 효과적인 적대적 텍스트 트리거를 생성하여 공격 목표를 달성하는 데 사용됩니다. 전략 모듈은 목표 에이전트를 혼란스럽게 하여 메모리 업데이트를 방해하며, 대리 모듈은 공격의 전이성과 인지 불가능성을 향상시키는 역할을 합니다.

- **Performance Highlights**: 다양한 실제 데이터셋을 대상으로 한 실험에서 DrunkAgent의 효과가 입증되었습니다. 이 연구는 추천 시스템의 안전하고 강건한 에이전트 개발을 위한 중요한 통찰력을 제공합니다. 전반적으로 Agent4RSs의 보안을 강화하고 더 안전한 AI 에이전트를 만드는 데 기여할 것으로 기대됩니다.



### Design and Experimental Validation of an Autonomous USV for Sensor Fusion-Based Navigation in GNSS-Denied Environments (https://arxiv.org/abs/2503.23445)
Comments:
          submitted to IEEE OCEANS 2025 Brest

- **What's New**: 이 논문은 센서 융합 기반 항법 알고리즘의 필드 테스트를 위해 설계된 자율 무인 수상 차량(MARVEL)을 소개합니다. MARVEL은 비용 효율성, 휴대성과 해양 환경에서의 성능을 중시하여 개발되었으며, 통합된 여러 센서를 통해 고빈도 데이터 수집과 실험 학습을 지원합니다. 이 플랫폼은 실제 해양 환경에서의 검증과 세밀한 운영을 가능하게 하여 연구자들이 저비용으로 접근할 수 있는 도구를 제공합니다.

- **Technical Details**: MARVEL은 전자기 속도 로그, 도플러 속도 로그, 관성 센서 및 실시간 동적 GNSS 위치 지정을 통합하여 센서 융합 실험을 가능하게 합니다. 이러한 복잡한 시스템은 다양한 해양 조건에 적응하고 고유한 성능을 발휘하도록 설계되었습니다. 또한, MARVEL은 명확한 문서와 사용자 친화적 디자인을 통해 기술적 배경이 제한된 사용자도 쉽게 조작할 수 있도록 하였습니다.

- **Performance Highlights**: 필드 실험 결과, MARVEL은 다양한 해양 조건에서의 안정성, 조작 가능성 및 적응성을 입증하였습니다. 이는 높은 신뢰도를 보장하며, 비주얼 인식과 센서 융합 기술로 강화된 자율 항법을 지원합니다. 이 시스템은 저비용의 구성 요소와 오픈 소스 소프트웨어를 활용하여 유연하고 확장성 있는 연구 플랫폼으로 자리 매김하고 있습니다.



### CoRanking: Collaborative Ranking with Small and Large Ranking Agents (https://arxiv.org/abs/2503.23427)
- **What's New**: 최근 논문에서는 효율적이고 효과적인 랭킹을 위해 소규모와 대규모 랭킹 모델을 결합한 새로운 협업 랭킹 프레임워크인 CoRanking을 제안했습니다. CoRanking은 작은 리랭커가 후보 단락을 사전 랭킹하여 최상위 부분에 관련 단락을 배치하고, 그 후 LLM(대형 언어 모델)에 의해 최상위 단락들만을 재랭킹함으로써 전반적인 효율성을 크게 향상시킵니다. 또한, 리인포스먼트 러닝을 통해 작은 리랭커에서 나오는 단락의 순서를 조정하는 새로운 구조인 POA를 도입하여 LLM의 선호와 맞추고 있습니다.

- **Technical Details**: CoRanking 프레임워크는 세 가지 주요 구성요소로 이루어져 있습니다. 첫 번째는 소규모 리스트 리랭커(SLR)로 초기 단락 재랭킹을 수행하고, 두 번째는 단락 순서 조정기(POA)로 SLR의 최상위 단락 순서를 LLM의 선호와 맞추며, 세 번째는 최종 랭킹을 위한 LLM 리스트 리랭커(LLR)입니다. 기존 랭킹 방법에서 제기된 한계를 극복하며, CoRanking은 성능 향상과 더불어 70%의 랭킹 지연 시간을 줄였습니다.

- **Performance Highlights**: CoRanking은 다수의 정보 검색(IR) 벤치마크에서 광범위한 실험을 통해 효율성과 효과성 면에서 우수한 결과를 보였습니다. 이 프레임워크는 순위 정확도를 높이는 동시에 랭킹 시간을 70% 단축하여 기존 LLM 리스트 리랭커보다 더욱 향상된 성능을 보여주었습니다. 이러한 성과는 작은 리랭커와 LLM 간의 협업을 통해 이루어진 결과입니다.



### CAWAL: A novel unified analytics framework for enterprise web applications and multi-server environments (https://arxiv.org/abs/2503.23244)
Comments:
          This is a preprint version of a research article printed in journal. The manuscript includes 21 pages, 10 figures, and 3 tables

- **What's New**: 이번 논문에서는 웹 분석의 한계를 극복하기 위해 Combined Analytics and Web Application Log (CAWAL) 프레임워크를 제안합니다. CAWAL은 데이터 소유권(Data Ownership)과 프라이버시(Privacy) 문제를 해결하면서도 애플리케이션 로깅(Application Logging)과 통합하여 정밀한 데이터 수집이 가능합니다. 이 프레임워크는 기업 환경에서 데이터 정확성을 높이고 문제 해결을 용이하게 합니다.

- **Technical Details**: CAWAL은 기업-grade 웹 애플리케이션에 통합되어 있으며, Open Web Analytics (OWA) 및 Matomo와 비교해 24% 및 85% 더 낮은 응답 시간을 기록하였습니다. 이 프레임워크는 복잡한 웹 환경에서 세션 관리(Session Management)와 교차 도메인 사용자 추적(Cross-Domain User Tracking)을 가능하게 하여, 다양한 사용자 상호작용을 효과적으로 포착합니다. 또한 머신러닝(Machine Learning)과 인공지능(Artificial Intelligence)의 기능을 접목하여 예측 분석(Predictive Analytics)을 허용합니다.

- **Performance Highlights**: 자체적인 성과 평가 결과 CAWAL은 기존의 도구들보다 유의미한 성능 향상을 보여줍니다. CAWAL은 웹 분석을 위한 고급 데이터 인프라로서의 기능을 완벽히 갖추고 있으며, 데이터 관리와 분석을 위한 포괄적인 프레임워크로 자리 잡을 가능성이 큽니다. 이 연구는 웹 분석과 애플리케이션 로깅의 결합을 통해 향후 기술 발전을 위한 중요한 기초를 제공합니다.



### Understanding Inequality of LLM Fact-Checking over Geographic Regions with Agent and Retrieval models (https://arxiv.org/abs/2503.22877)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 사실 확인(fact-checking)에서의 성능을 다양한 지역과 시나리오를 통해 평가하였습니다. 특히, LLM의 성능이 지역에 따라 다르게 나타나는 점을 강조하여, 이를 통해 오정보의 확산을 방지하는 방법을 모색하였습니다.

- **Technical Details**: 연구는 600개의 사실 확인이 이루어진 진술(statement)을 포함한 데이터셋을 사용하여 세 가지 실험 세팅을 평가하였습니다: (1) 진술만 있을 때, (2) 위키피디아 접근이 가능한 LLM 기반 에이전트를 사용할 때, (3) 공식 사실 확인이 제공되는 Retrieval-Augmented Generation (RAG) 시스템을 사용할 때입니다. 이러한 설정을 통해 각 모델의 성능 차이를 분석하였습니다.

- **Performance Highlights**: 연구 결과, GPT-4, Claude Sonnet 및 LLaMA를 포함한 어떤 LLM을 사용하더라도, 선진국(Global North) 출처의 진술이 개발도상국(Global South) 출처의 진술보다 상당히 더 높은 성능을 나타냈습니다. 이러한 차이는 위키피디아 에이전트 기반 시스템을 사용한 경우에 더욱 확대되었습니다. 이는 지역적 특성을 반영할 수 있는 자료의 균형 잡기와 강력한 검색 전략의 필요성을 강조하고 있습니다.



### From Individual to Group: Developing a Context-Aware Multi-Criteria Group Recommender System (https://arxiv.org/abs/2503.22752)
Comments:
          The 16th International Conference on Management of Digital EcoSystems, Nov 2024, Naples, Italy

- **What's New**: 이 연구에서는 다양한 개인의 선호를 고려해야 하는 그룹 의사결정 상황에서의 문제를 해결하기 위해 Context-Aware Multi-Criteria Group Recommender System (CA-MCGRS)을 개발하였습니다. 기존의 추천 시스템은 개별화에 효과적이지만 그룹 의사결정에서는 갈등하는 선호와 다양한 평가 기준을 다루는 데 한계가 있습니다. CA-MCGRS는 이러한 맥락적 요소와 다중 기준을 통합하여 추천의 정확도를 높이는 데 초점을 맞추었습니다.

- **Technical Details**: CA-MCGRS는 Multi-Head Attention 메커니즘을 활용하여 서로 다른 특징의 중요성을 동적으로 조정합니다. 이 모델은 교육 데이터셋에서 다양한 평가와 맥락 변수를 고려하여 실험을 진행하였습니다. 이러한 접근 방식은 추천 정확도를 향상시키는데 기여하는 것으로 나타났습니다.

- **Performance Highlights**: 실험 결과, CA-MCGRS는 네 가지 시나리오 모두에서 다른 접근 방식들보다 지속적으로 우수한 성과를 보였습니다. 따라서 본 연구의 발견은 그룹 추천 시스템의 개발에 있어 맥락과 다중 기준 평가를 포함하는 것이 얼마나 중요한지를 강조합니다.



### Adaptive Integrated Layered Attention (AILA) (https://arxiv.org/abs/2503.22742)
- **What's New**: 이번 연구에서는 Adaptive Integrated Layered Attention (AILA)라는 신경망 아키텍처를 제안합니다. AILA는 다양한 네트워크 층 간의 적응형(feature reuse) 기능을 위해 밀집 스킵 연결(dense skip connections)과 여러 메커니즘을 융합하여 구성되어 있습니다. AILA는 가격 예측, 이미지 인식, 감정 분석의 세 가지 도전 과제를 평가받았으며, 기존의 강력한 딥러닝 모델과 유사한 성능을 보이면서도 훈련 및 추론 시간을 크게 단축시켰습니다.

- **Technical Details**: AILA는 두 가지 아키텍처, 즉 AILA-Architecture 1과 AILA-Architecture 2로 나뉘어 있습니다. AILA-Architecture 1은 층 간의 연결 메커니즘으로 간단한 선형 층(linear layers)을 사용하고, AILA-Architecture 2는_attention_ 메커니즘을 구현하여 이전 층의 출력을 선택적으로 강조합니다. 이러한 아키텍처는 각기 다른 태스크에 대해 개별적으로 훈련되며, 다양한 네트워크 깊이에서 관련 기능을 유연하게 재사용함으로써, 강력한 성능 향상을 이루어냅니다.

- **Performance Highlights**: AILA는 세 가지 기준 벤치마크에서 강력한 성능 지표를 달성했습니다. 가격 예측, CIFAR-10 데이터셋에 대한 이미지 인식, IMDB 영화 리뷰 데이터셋의 감정 분석에서 AILA-Architecture 1 및 2 모두 LSTM, Transformer, CNN과 같은 기존의 강력한 기준 모델과 경쟁하며 이를 초월하는 성과를 보여주었습니다. 결과적으로 AILA는 일반적인 고정 연결 방식이 아닌, 적응형 정보 흐름을 통해 복잡한 태스크에서 성능을 향상시키는 새로운 길을 열었습니다.



New uploads on arXiv(cs.CV)

### Easi3R: Estimating Disentangled Motion from DUSt3R Without Training (https://arxiv.org/abs/2503.24391)
Comments:
          Page: this https URL Code: this https URL

- **What's New**: 본 논문에서는 Easi3R이라는 새로운 방법을 소개하고 있습니다. 이 방법은 4D 재구성을 위한 훈련이 필요 없는 간단하고 효율적인 접근법으로, 기존 3D 모델에 대한 미세 조정 없이 동적 비디오에서 카메라 자세 추정을 수행할 수 있습니다. 이를 통해 복잡한 4D 모델을 다루는 데 필요한 대규모 데이터의 필요성을 줄일 수 있습니다.

- **Technical Details**: Easi3R는 DUSt3R의 주의(attention) 레이어를 활용하여, 동적 객체 세분화와 밀집 포인트 맵 재구성을 수행할 수 있습니다. 이 방법은 동적 비디오에서 카메라 모션 추정과 함께 장기적인 동적 객체 감지를 가능하게 하는 간단한 분해 전략을 제공합니다. Easi3R의 핵심은 훈련 과정 없이 두 이미지 피쳐를 입력으로 받아 출력으로 픽셀 정렬된 포인트 맵을 생성하는 것입니다.

- **Performance Highlights**: 실험 결과, Easi3R는 기존의 동적 데이터셋을 기반으로 훈련된 최신 방법들에 비해 월등한 성능을 보였습니다. 카메라 자세 추정, 동적 객체 세분화, 포인트 클라우드 재구성의 세 가지 작업에서 뛰어난 결과를 얻었습니다. 본 연구의 코드도 공개되어 있어 추가 연구와 응용이 가능합니다.



### SU-YOLO: Spiking Neural Network for Efficient Underwater Object Detection (https://arxiv.org/abs/2503.24389)
- **What's New**: 본 논문에서는 수중 객체 탐지를 위한 새로운 모델인 Spiking Underwater YOLO (SU-YOLO)를 제안합니다. SU-YOLO는 스파이킹 신경망(Spiking Neural Network, SNN) 기반으로, 이전에 적용된 일반 객체 탐지 모델을 개조한 것입니다. 특히 이 모델은 연산 효율성을 높이고 에너지 소모를 줄이는 동시에 수중 이미지의 잡음 제거를 위한 새로운 방법을 통합하여 성능을 극대화합니다.

- **Technical Details**: SU-YOLO는 YOLO 아키텍처와 Cross Stage Partial Network (CSPNet)를 통합하여 신경망 구조를 경량화하고 스파이크 경량화를 통해 약한 스파이크 문제를 완화합니다. 이 모델은 Separated Batch Normalization (SeBN)이라는 새로운 정규화 방법을 도입하여 여러 시간 단계에서 피처 맵을 독립적으로 정규화하고, 잔차 구조(residual structure)와 통합하여 SNN의 시간 동역학을 효과적으로 캡처합니다.

- **Performance Highlights**: URPC2019 수중 데이터셋 테스트에서 SU-YOLO는 78.8%의 mAP(Mean Average Precision)를 달성하며, 6.97M의 파라미터와 2.98 mJ의 에너지 소비를 기록했습니다. 이를 통해 기존의 SNN 모델은 물론 여러 ANN 모델보다 더 나은 탐지 정확도와 계산 효율성을 입증하며, SNN이 엔지니어링 응용 분야에서 가진 잠재력을 부각시킵니다.



### Consistent Subject Generation via Contrastive Instantiated Concepts (https://arxiv.org/abs/2503.24387)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 Contrastive Concept Instantiation (CoCoIns) 프레임워크를 소개하여, 텍스트-이미지 생성에서 주제 일관성을 효과적으로 유지할 수 있는 방법을 제안합니다. 기존의 방법과 달리, 시간 소모가 많은 튜닝이나 참조 이미지 준비 없이도 독립적인 생성물들 간의 일관된 주제를 생성할 수 있습니다. CoCoIns는 생성 모델과 매핑 네트워크로 구성되어 있으며, 이를 통해 사용자는 동일한 잠재 코드(latent codes)를 사용하여 일관된 주제를 생성할 수 있습니다.

- **Technical Details**: CoCoIns 프레임워크는 입력 잠재 코드(latent codes)를 특정 개념 인스턴스와 연관된 의사 단어(pseudo-words)로 변환하는 매핑 네트워크를 포함합니다. 또한, 대조적 학습(contrastive learning) 방식을 통해 네트워크가 프롬프트(prompts)와 잠재 코드의 조합을 구분하도록 학습합니다. 이 방식은 한정된 데이터에서의 직접적인 학습을 피하고 잠재 코드와 주제 인스턴스의 연관성을 확립하는 데 효과적입니다.

- **Performance Highlights**: CoCoIns 프레임워크는 인간 이미지에 대한 실험을 통해 우수한 주제 일관성을 달성하였고, 기존의 배치 생성 방식과 비교하여 더 높은 유연성을 유지합니다. 이 방법은 여러 주제와 다른 개체 카테고리로 확장할 수 있는 잠재력을 보여주며, 시간 소모가 많은 튜닝이나 배치 생성 접근 방식을 요구하지 않는 최초의 주제 일관성 생성 프레임워크로 자리잡을 수 있습니다.



### Free360: Layered Gaussian Splatting for Unbounded 360-Degree View Synthesis from Extremely Sparse and Unposed Views (https://arxiv.org/abs/2503.24382)
Comments:
          Accepted to CVPR 2025. Project Page: this https URL

- **What's New**: 이번 연구에서는 극도로 희박한 뷰(views)와 포즈(pose)가 없는 상태의 3D 재구성을 가능하게 하는 새로운 신경 렌더링 프레임워크를 제안합니다. 기존 방법들이 해결하지 못했던 문제인 360° 장면에서의 공간 모호성을 해결하기 위해 층(layer) 기반의 Gaussian 표현을 사용합니다. 또한, 생기된 관측값을 활용하여 재구성을 강화하는 반복적 융합 전략도 개발하였습니다.

- **Technical Details**: 이 프레임워크는 희박한 뷰로부터 초기 기하구조를 복구하기 위해 밀집 스테레오(stereo) 재구성 모델을 활용합니다. 그렇게 얻은 정보를 기반으로 공간을 층 구조로 나누고, 각 층에 대해 부트스트랩 최적화를 통해 재구성 오류를 정제합니다. 비디오 확산 모델을 적용하여 생성된 뷰를 통해 관측값을 추가 생성하고, 이로 인해 재구성의 정확성을 높입니다.

- **Performance Highlights**: 광범위한 실험을 통해 본 방법이 기존의 최신 기술들과 비교하여 렌더링 품질과 표면 재구성 정확도에서 우수한 성능을 보임을 확인했습니다. 특히, 단 3-4개의 뷰만 사용하여도 분명한 결과를 만들어 내어 신경 렌더링(NR)의 새로운 가능성을 제시하였습니다.



### UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving (https://arxiv.org/abs/2503.24381)
Comments:
          14 pages; Dataset: this https URL Code: this https URL

- **What's New**: UniOcc는 점유 예측(occupancy forecasting) 및 현재 프레임 점유 예측을 위한 포괄적인 벤치마크입니다. 다양한 실제 데이터셋(nuScenes, Waymo)과 고충실도 시뮬레이터(CARLA, OpenCOOD)의 데이터를 통합하여 2D/3D 점유 레이블과 각 복셀(voxel) 흐름(flow) 주석을 제공합니다. 새로운 평가 지표를 통합하여 기존의 잘못된 중간 진리(ground-truth)에 의존하지 않고 점유 품질을 평가할 수 있는 robust한 방안을 제시합니다.

- **Technical Details**: UniOcc는 단일 데이터셋에 의존하던 기존 방법들의 제약을 극복하고 크로스 데이터셋 학습을 지원합니다. CARLA 시뮬레이션을 활용하여 다양한 훈련 데이터를 제공하고, 각 복셀에 대한 전향(forward) 및 역방향(reverse) 흐름 주석을 통해 동적 장면 단서를 포착할 수 있도록 합니다. 이는 협력적 주행(cooperative driving) 시나리오를 지원하는 최초의 데이터셋이기도 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 대규모 다양한 훈련 데이터와 명시적인 흐름 정보가 점유 예측 및 예측 성능을 유의미하게 향상시킨다는 것을 입증하였습니다. 우리는 UniOcc이 점유 중심 연구의 촉매제로 작용하여 자율 주행에서의 혁신을 촉진할 것이라고 기대합니다. 또한 기존 방법들이 크로스 도메인 일반화에 어려움을 겪고 있음을 보여주어 향후 연구의 방향성을 제시합니다.



### Any2Caption:Interpreting Any Condition to Caption for Controllable Video Generation (https://arxiv.org/abs/2503.24379)
Comments:
          Project Page: this https URL

- **What's New**: 본 연구는 사용자의 의도를 정확히 해석하는 데 발생하는 병목 현상을 해결하기 위해 Any2Caption이라는 새로운 프레임워크를 제시합니다. 이 프레임워크는 비디오 생성 시 다양한 조건을 주석으로 구조화하여 비디오 생성기에게 더 나은 지침을 제공합니다. 추가로 Any2CapIns라는 대규모 데이터셋을 구축하여 다양한 조건을 활용한 주석 튜닝을 가능하게 합니다.

- **Technical Details**: Any2Caption은 텍스트, 이미지, 비디오 및 특수한 신호(예: 지역, 모션, 카메라 포즈)를 포함한 다양한 입력을 해석하여 밀집하고 구조화된 캡션을 생성하는 MLLM 기반의 조건 인터프리터입니다. 이를 통해 비디오 생성 모델의 제어력과 비디오 품질이 향상됩니다. 본 연구는 337K의 사례와 407K의 조건으로 구성된 Any2CapIns 데이터셋을 통해 성능을 평가합니다.

- **Performance Highlights**: Any2Caption은 여러 SoTA 비디오 생성 모델과의 통합에서 높은 품질의 비디오 생성을 가능하게 하며, 특히 복합적인 조건을 처리하는 데 탁월한 성능을 보입니다. 실험 결과, Any2Caption을 통해 생성된 비디오는 보다 풍부하고 의미 있는 주석으로 향상되며, 제어 가능성과 비디오 품질에서 기존 모델들을 뛰어넘는 결과를 나타냅니다.



### Exploring the Effect of Reinforcement Learning on Video Understanding: Insights from SEED-Bench-R1 (https://arxiv.org/abs/2503.24376)
Comments:
          Technical Report (In Progress); Code released at: this https URL

- **What's New**: 본 논문은 멀티모달 대규모 언어 모델(MLLMs)의 비디오 이해를 평가하기 위한 새로운 벤치마크인 SEED-Bench-R1을 제안합니다. SEED-Bench-R1은 복잡한 일상적인 계획 작업을 여러 선택 질문 형태로 포함하여, 정교한 인식(perception)과 논리적 추론(logical reasoning)을 요구합니다. 또한, 이 벤치마크는 세 가지 수준의 일반화(generalization) 시나리오를 통해 MLLMs의 포스트 트레이닝(post-training) 방법을 체계적으로 평가합니다.

- **Technical Details**: SEED-Bench-R1은 현실적인 일상 활동을 기반으로 한 비디오를 사용하여, 모델이 목표를 이해하고 긴 시간 동안 시각적인 진행을 추적하며, 복잡한 환경 관찰을 인지하고, 세계 지식을 사용하여 다음 행동을 추론할 수 있도록 설계되었습니다. 이 벤치마크는 교육 데이터셋을 기반으로 하며, 명확하게 검증 가능한 정답을 제공하여 일반화 능력을 철저히 평가할 수 있는 구조로 되어 있습니다. Qwen2-VL-Instruct-7B를 사용하여 RL 및 감독된 파인튜닝(SFT) 방법을 비교하여 RL이 데이터 효율성과 성능 면에서 우수하다는 것을 보여주었습니다.

- **Performance Highlights**: 실험 결과, RL은 특히 OOD(Out-of-Distribution) 시나리오에서 SFT를 능가하며, 비디오 이해의 일반적인 벤치마크에서도 높은 성과를 보였습니다. RL은 시각적 인식을 향상시키고 COT(Chain of Thought) 토큰을 동적으로 쿼리하도록 모델을 교육하는 데 효과적이었습니다. 그러나 모델이 때때로 중요한 시각적 단서를 무시하는 등, 몇 가지 한계점도 드러났고, 이는 향후 연구와 개선 방향 설정에 중요한 요소가 될 것입니다.



### ERUPT: Efficient Rendering with Unposed Patch Transformer (https://arxiv.org/abs/2503.24374)
Comments:
          Accepted to CVPR 2025

- **What's New**: 이번 연구에서는 RGB 이미지의 작은 집합을 이용하여 다양한 장면의 새로운 뷰 신합(Novel View Synthesis) 문제를 다룹니다. 우리는 ERUPT (Efficient Rendering with Unposed Patch Transformer)라는 최신의 장면 재구성 모델을 제안하며, 비명시적(Unposed) 이미지 사용으로 효율적인 장면 렌더링을 가능하게 합니다. 특히, 우리는 픽셀 기반 쿼리 대신 패치 기반 쿼리를 도입하여 타겟 뷰를 렌더링하는 데 필요한 연산량을 줄였습니다.

- **Technical Details**: ERUPT 모델은 학습된 잠재 카메라 포즈를 사용하여 비명시적 감지 할 수 있는 데이터를 활용할 수 있도록 설계되었습니다. 이를 통해 희소하거나 부정확한 카메라 포즈가 있는 데이터셋에서도 훈련이 가능합니다. 또한, 기존 모델보다 10배 이상의 계산 효율성을 높이기 위해 픽셀 기반 레이 대신 패치 기반 레이를 사용하는 혁신적인 디코딩 전략을 도입했습니다.

- **Performance Highlights**: ERUPT는 상용 하드웨어에서 600 fps로 새로운 뷰를 렌더링할 수 있을 만큼 효율적이며, 기존 최첨단(산업 표준) 방법보다 더 나은 렌더링 이미지 품질을 달성했습니다. 라벨이 지정된 데이터 요구 사항을 약 95% 줄여주어, 다양한 실제 세계 장면에 대한 효율적인 새로운 뷰 신합을 제공합니다. 우리는 또한 MSVS-1M이라는 새로운 벤치마크 데이터셋을 도입하여, 실제 세계 데이터를 통한 직접 비교를 가능하게 합니다.



### Adapting Vision Foundation Models for Real-time Ultrasound Image Segmentation (https://arxiv.org/abs/2503.24368)
- **What's New**: 이 논문에서는 실시간 초음파 이미지 분할을 위해 계층적 비전 기초 모델(Hierarchical Vision Foundation Models)을 적응시키는 새로운 접근 방식을 제안합니다. 기존의 초음파 분할 방법들은 새로운 작업에 대한 적응성이 떨어지고, 비싼 수동 주석에 의존하는 경우가 많습니다. 본 연구는 Hiera 모델을 이용하여 다중 스케일 특성을 추출하고, DINOv2 표현과 결합하여 시각적 표현을 강화합니다.

- **Technical Details**: Hiera의 계층 인코더를 적응시키고 DINOv2 피처를 통합하여 시각적 표현을 향상시키는 방법론을 제시합니다. 이 방법은 6개의 공개 데이터셋과 1개의 내부 데이터셋에서 평가되었으며, 심장 및 갑상선 초음파 분할을 포괄합니다. 최종적으로, 이 접근법은 다양한 데이터셋에서 기존 방법들보다 향상된 성능을 보이며, 한정된 감독 하에서도 우수한 일반화를 보여줍니다.

- **Performance Highlights**: 제안된 방법은 nnUNet보다 평균 20% 이상 성능이 향상되었고, CAMUS 및 TN3K 데이터셋에서는 최첨단 성능을 기록하였습니다. 실시간 임상 응용이 가능한 약 77 FPS의 추론 속도를 달성하여, 환자 진단과 치료에서의 적용 가능성을 높입니다.



### StochasticSplats: Stochastic Rasterization for Sorting-Free 3D Gaussian Splatting (https://arxiv.org/abs/2503.24366)
- **What's New**: 본 논문에서는 3D Gaussian splatting (3DGS) 기법을 개선하기 위해 확률적 래스터화(stochastic rasterization)와 결합한 새로운 방법을 제안합니다. 기존 알고리즘의 깊이 정렬(depth sorting) 및 래스터화(rasterization) 과정에서 발생하는 렌더링 아티팩트를 제거하고 정확한 3D 혼합을 가능하게 합니다. 이 접근법을 통해 컴퓨터 파라미터를 조정하여 처리 시간과 품질을 균형 있게 조절할 수 있습니다.

- **Technical Details**: 이 논문은 비편향 몬테 카를로 추정기(unbiased Monte Carlo estimator)를 활용하여 깊이 정렬을 생략하고, 색상 혼합(color blending)과 경량의 훈련(training)을 가능하게 합니다. 이 방법은 오픈GL(OpenGL) 셰이더를 사용하여 현대 GPU 하드웨어에서 효율적으로 구현됩니다. 각 픽셀에 대해 하나의 표본만 사용할 경우, 제안된 방법은 기존 CUDA 3DGS 구현보다 최대 4배 빠른 렌더링 성능을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 렌더링 품질을 유지하면서 이를 여러 번의 표본을 사용하여 개선할 수 있는 장점이 있습니다. 다운스트림 애플리케이션은 렌더링 시간 동안 샘플링을 동적으로 조정하여 레이턴시와 품질 간의 균형을 맞출 수 있습니다. 이러한 유연성은 로봇 공학 및 자율 주행과 같은 지연에 민감한 응용 프로그램에서 실용성을 더욱 높입니다.



### InstructRestore: Region-Customized Image Restoration with Human Instructions (https://arxiv.org/abs/2503.24357)
- **What's New**: 최근의 확산 기반 모델들이 이미지 복원에 있어 중요한 진전을 보였음에도 불구하고, 기존 방법들은 전체 이미지에 균일한 처리를 적용하여 사용자 지시에 따른 지역 맞춤 복원의 능력이 부족함. 본 연구에서는 'InstructRestore'라는 새로운 프레임워크를 제안하여 사용자 지침에 따라 조정 가능한 이미지 복원을 수행할 수 있는 시스템을 개발함. 이를 위해 고품질 이미지, 대상 영역 설명 및 해당 영역 마스크로 구성된 훈련 삼중체를 생성하는 데이터 생성 엔진을 구축하였음.

- **Technical Details**: InstructRestore는 조건부 요소를 조정하기 위해 ControlNet 아키텍처를 통합하여 저품질 이미지의 특징을 조정하는 방식을 적용. 사용자 지침에 맞춰 지역을 식별하고 각 지역에 다른 통합 스케일을 배정하여, 맞춤형 이미지 복원이 가능하도록 설계됨. 또한, 세분화된 데이터를 통해 훈련된 InstructRestore 모델은 지역별 복원을 효과적으로 수행할 수 있도록 구성됨.

- **Performance Highlights**: 실험 결과, InstructRestore 방법이 사용자 지시에 따라 효과적인 이미지 복원이 가능함을 보여줌. 예를 들어, 배경이 흐릿하게 처리된 초상화 이미지와 특정 지역 수정을 요구하는 상황에서 뛰어난 성능을 발휘함. 본 연구는 상호작용적인 이미지 복원 및 향상 기술에 대한 새로운 접근법을 제시함.



### PathOrchestra: A Comprehensive Foundation Model for Computational Pathology with Over 100 Diverse Clinical-Grade Tasks (https://arxiv.org/abs/2503.24345)
- **What's New**: 이 논문에서는 PathOrchestra라는 다목적 병리학 기초 모델을 제안합니다. self-supervised learning(자기 지도 학습)을 통해 20개 조직 및 장기 유형에서 30만 개의 병리 슬라이드로 훈련되었습니다. 이는 대규모 데이터셋, 상당한 저장 용량, 그리고 많은 컴퓨팅 자원을 요구하는 기존 모델의 한계를 극복한 새로운 접근 방식입니다.

- **Technical Details**: PathOrchestra는 61개의 프라이빗(private) 데이터셋과 51개의 퍼블릭(public) 데이터셋을 조합하여 112개의 임상 임무에서 철저히 평가되었습니다. 이 임무는 디지털 슬라이드 전처리, 전체 암(pan-cancer) 분류, 병변 식별, 다중 암 아형 분류, 바이오마커 평가, 유전자 발현 예측 및 구조화된 보고서 생성 등을 포함합니다. 이 모델은 27,755개 WSI(Whole Slide Images) 및 9,415,729개 ROI(Region of Interest)에서 뛰어난 성능을 보였습니다.

- **Performance Highlights**: PathOrchestra는 전체 47개 임무에서 0.950 이상의 정확도를 달성하였으며, 이는 다양한 장기에서의 전체 암 분류, 림프종 아형 진단 및 방광암 스크리닝 등을 포함합니다. 특히, 이 모델은 고빈도 대장암 및 진단이 복잡한 림프종에 대한 구조화된 보고서를 생성하는 최초의 모델입니다. PathOrchestra는 방대한 규모의 self-supervised 병리학 모델의 실행 가능성과 효율성을 입증하며, 임상 통합을 위한 가능성을 제공합니다.



### Self-Supervised Pretraining for Aerial Road Extraction (https://arxiv.org/abs/2503.24326)
- **What's New**: 이 논문의 주요 혁신은 라벨이 없는 데이터를 이용하여 항공 이미지 세분화 성능을 향상시키는 자기 감독형(pretraining) 방법을 제안하는 것입니다. 이를 위해 모델은 항공 이미지의 누락된 영역을 복원하는 inpainting 기반 전 훈련 기법을 사용하여, 도로 추출을 위한 미세 조정 단계에 들어갑니다. 이 방법은 라벨된 데이터에 대한 의존도를 줄이고, 일반화(generalization)를 개선하며 도메인 변화(domain shift)에 강인성을 높입니다.

- **Technical Details**: 제안된 방법은 세 가지 단계로 구성되어 있으며, 라벨이 없는 항공 이미지로 시작하여 자기 감독형 inpainting 단계를 통해 정보 구조를 학습합니다. 그런 다음, 도메인 갭을 해소하는 데 초점을 맞춘 두 번째 훈련 단계를 통해 도로 세분화(task)와 연계된 특성의 전이 가능성을 향상시키고자 합니다. 이러한 방식은 CNN 기반 모델 아키텍처와 데이터 세트에 관계없이 적용 가능하며, 최소한의 수정으로도 성능 향상을 꾀할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 전 훈련 방법이 세분화 정확도를 크게 향상시킴을 보여주었습니다. 특히 데이터가 적은 상황에서도 성능 상승을 이루어내며, 다양한 아키텍처에서 일관된 성능 개선을 보였습니다. 이를 통해 제안된 방법은 항공 이미지 분석의 확장 가능한 솔루션으로 자리매김할 수 있음을 나타냅니다.



### Can Test-Time Scaling Improve World Foundation Model? (https://arxiv.org/abs/2503.24320)
- **What's New**: 이 논문에서는 SWIFT라는 새로운 테스트 시간 확장(framework)을 소개합니다. 이 프레임워크는 세계 기초 모델(World Foundation Models, WFM)을 대상으로 하며, 기존의 모델 크기 확대나 재훈련 없이도 성능을 향상시키는 방법을 제시합니다. SWIFT는 빠른 토크나이징(fast tokenization), 확률 기반 Top-K 프루닝(probability-based Top-K pruning), 효율적인 빔 탐색(beam search) 등을 통합하여 효과적인 테스트 시간 검색 전략을 제공합니다.

- **Technical Details**: SWIFT는 프로세스 수준에서의 추론 전략을 통합하여 WFM의 효율적인 테스트 시간 검색을 가능하게 합니다. 이 시스템은 여러 비디오 입력을 처리할 때 필요한 대규모 계산 자원을 최소화하는 데 초점을 맞추고 있습니다. 논문에서는 COSMOS 모델을 기반으로 테스트 시간 확장 법칙(test-time scaling laws)의 존재를 보여주며, SWIFT가 추가 샘플 생성을 통해 성능을 더욱 향상시킬 수 있음을 보여줍니다.

- **Performance Highlights**: SWIFT를 통해 소규모 모델이 테스트 시간 확장을 적용 시, 상당히 큰 모델의 성능을 초과할 수 있음을 발견했습니다. 테스트 시간에서의 확장은 기존의 모델 크기 증가보다 더 효율적인 방법으로, 자원을 적게 사용하면서도 성능을 극대화할 수 있습니다. 이 프레임워크는 특정 데이터 집합에서 효율성 높은 방식으로 성능을 향상시킬 수 있는 새로운 경로를 제공합니다.



### Point Tracking in Surgery--The 2024 Surgical Tattoos in Infrared (STIR) Challeng (https://arxiv.org/abs/2503.24306)
- **What's New**: 2024 STIR Challenge는 수술에서의 추적 및 재구성 방법을 개선하기 위해 설계되었습니다. 본 연구에서는 조직의 움직임 및 위치 파악이 의료 컴퓨터 비전의 여러 작업을 가능하게 한다는 점을 강조합니다. 향상된 모션 추정의 정확성은 자동화된 조작 및 자율 스캔을 위한 중요한 요소입니다. 이 챌린지는 점 추적 메트릭을 사용하여 제출된 방법의 성능을 정량화하는 최초의 시도입니다.

- **Technical Details**: STIR Challenge 2024 데이터셋은 da Vinci Xi 시스템에서 수집된 스테레오 비디오 클립으로 구성됩니다. 각 클립은 시작 및 종료 IR 이미지, 형광 잉크의 분할 정보, 그리고 가시광 비디오로 이루어져 있습니다. 데이터셋은 총 60개의 시퀀스를 포함하며, 평균 길이는 8.9초입니다. 이 데이터는 포르신(subjects)에서 수집되었으며, ICG 잉크가 사용되어 실제 위치를 라벨링하는 방법이 설명됩니다.

- **Performance Highlights**: 챌린지는 정확도와 효율성 두 가지 주요 구성 요소로 평가됩니다. 정확도는 in vivo 및 ex vivo 시퀀스에서 알고리즘의 성능을 테스트하며, 효율성은 알고리즘 추론 지연 시간을 측정합니다. 2024 STIR Challenge는 MICCAI EndoVis 2024의 일환으로 진행되었으며, 총 8팀의 제출물이 있었습니다. 이러한 도전은 더 정확하고 효율적인 수술 공간 이해 알고리즘 개발을 촉진하는 데 기여할 것입니다.



### Order Matters: On Parameter-Efficient Image-to-Video Probing for Recognizing Nearly Symmetric Actions (https://arxiv.org/abs/2503.24298)
- **What's New**: 이번 연구에서는 Self-attentive Temporal Embedding Probing (STEP)이라는 새로운 방법을 소개하여 매력적인 영상 인식 과제를 해결합니다. 이 방법은 시간 민감도를 강화하기 위해 세 가지 주요 수정 사항을 도입하여, 이미지-비디오 전송 시 성능을 크게 향상시킵니다. STEP은 기존 영상 인식 기법에 비해 1/3의 학습 가능한 파라미터로도 성능을 3-15% 향상시켰습니다.

- **Technical Details**: STEP은 (1) 시간적 순서를 명확히 하여 강화된 학습 가능한 프레임별 위치 인코딩, (2) 시퀀스 일관성을 위한 단일 글로벌 CLS 토큰, (3) 파라미터 효율성을 높이기 위한 간소화된 주의 메커니즘을 포함합니다. 이러한 구성 요소들은 비디오의 시퀀스 본질을 강조하며, 시간적 순서의 이해도를 높입니다. STEP은 네 가지 데이터셋(NTU-RGB+D 120, IKEA-ASM, Drive&Act, SSv2)에서 테스트되어, 기존의 probing 방법들을 지속적으로 초월한 성과를 보였습니다.

- **Performance Highlights**: STEP은 IKEA-ASM 및 Drive&Act 데이터셋에서 최신 상태의 결과를 달성하여 인식 정확도를 각각 13.5%와 14.86% 향상시켰습니다. 이는 손 중심 및 시간 민감도가 높은 행동을 포함한 데이터셋에서 특히 두드러집니다. 또한, STEP은 NTU-120에서 대칭 행동에 대한 성능에서도 PEFT 방법들을 초월하며, 데이터 효율성을 입증했습니다.



### Style Quantization for Data-Efficient GAN Training (https://arxiv.org/abs/2503.24282)
- **What's New**: 새로운 접근법인 SQ-GAN을 제안합니다. 이는 스타일 공간 양자화(style space quantization) 기법을 도입해 생성 모델의 일관성 정규화(consistency regularization)를 개선합니다. 이 방법은 희소한 입력 잠재 공간(sparse input latent space)을 구조화된 이산 프록시 공간(compact discrete proxy space)으로 변환하여 각 요소가 실제 데이터 포인트에 대응할 수 있도록 합니다. 이를 통해 GAN의 성능 저하를 완화하고 생성 이미지를 더욱 다양화할 수 있습니다.

- **Technical Details**: SQ-GAN은 입력 잠재 변수( latent variables)를 낮은 얽힘의 스타일 공간으로 매핑하여 진행됩니다. 그런 다음 학습 가능한 코드북을 이용해 양자화를 적용하며, 양자화된 코드는 각각의 다양한 변화 요인을 제어할 수 있도록 합니다. 또한 최적 수송 거리(optimal transport distance)를 최적화해 코드북 코드와 훈련 데이터에서 추출된 특성을 정렬하며, 이를 통해 코드북에 외부 지식을 내재시킵니다. 이러한 과정은 잠재 공간을 더 구조화된 형태로 변환하여 더 일관된 이미지를 생성하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, SQ-GAN은 판별자(discriminator)의 강건성(discriminator robustness)과 생성 품질(generation quality)의 유의미한 개선을 보여줍니다. 특히 제한된 데이터 조건 하에서도 향상된 일관성 정규화 성능을 발휘하여, 생성된 이미지의 진정성(authenticity)을 효과적으로 높였습니다. 이러한 결과는 GAN 훈련 방식에 대한 새로운 통찰을 제공하며 향후 연구의 방향성을 제시합니다.



### Learning Velocity and Acceleration: Self-Supervised Motion Consistency for Pedestrian Trajectory Prediction (https://arxiv.org/abs/2503.24272)
- **What's New**: 본 논문은 보행자 궤적 예측에서 기존의 감독 학습(supervised learning) 대신 자가 감독 학습(self-supervised learning) 프레임워크를 제안합니다. 이 방법은 위치(position), 속도(velocity), 가속도(acceleration)를 명시적으로 모델링하여 궤적 예측의 정확도를 높입니다. 특히, 모션 일관성 평가(mechanical consistency evaluation) 전략을 도입하여 예측된 동작 경향과 역사적 동작을 비교하고, 이를 바탕으로 궤적 생성을 가이드합니다.

- **Technical Details**: 제안된 프레임워크는 세 개의 스트림 네트워크(three-stream network)를 기반으로 하며, 각 스트림은 Historical 데이터에서 위치, 속도, 가속도 정보를 처리합니다. 이 네트워크는 속도와 가속도 정보를 통해 위치 예측을 보강하며, 속도 스트림에는 가속도 기능이 추가되어 함께 최적화됩니다. 또한, 사회적 디코더(social decoders)를 활용하여 보행자 상호작용을 분석하고, 결과적으로 예측된 위치와 물리적 일관성을 유지합니다.

- **Performance Highlights**: ETH-UCY 및 Stanford Drone 데이터셋을 사용한 실험 결과, 제안된 방법은 기존의 방법들과 비교하여 최첨단 성능(state-of-the-art performance)을 달성했습니다. 특히, 장기 분포(long-tail distribution)로 인한 모델의 과적합 문제를 완화하고, 비정상적인 동작을 보다 정확하게 포착할 수 있는 능력을 보여주었습니다.



### Visual Acoustic Fields (https://arxiv.org/abs/2503.24270)
- **What's New**: 본 연구에서는 3D 공간 내에서 시각과 음향 신호의 교차 모델 관계를 링크하는 'Visual Acoustic Fields'라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 두 가지 주요 모듈, 즉 소리 생성(sound generation)과 소리 위치화(sound localization)를 포함합니다. 'Visual Acoustic Fields'는 3D Gaussian Splatting(3DGS)을 이용하여 시각 신호와 타격 소리를 연결하여, 현존하는 데이터셋과는 달리 3D 맥락에서 시각과 음향 신호를 연결합니다.

- **Technical Details**: 연구진은 다중 시점 이미지(multiview images)와 해당 타격 위치 및 연관된 소리로 이루어진 데이터셋을 수집하기 위한 파이프라인을 구현하였습니다. 구조-모션 추정(structure-from-motion) 기술을 사용하여 이미지의 카메라 포즈를 추정하고, 이 정보를 바탕으로 충격 소리와 그에 상응하는 시각 신호를 통합합니다. 'AudioCLIP' 기능을 사용하여 시각적 및 청각적 신호를 매칭하고, 타격점에서의 소리를 생성하기 위해 오디오 확산 모델을 활용합니다.

- **Performance Highlights**: 실험 결과, 수집된 시각-소리 쌍의 타격 위치를 3D 공간에서 정확하게 로컬라이징할 수 있음을 보여주었습니다. 예측된 타격 소리는 실제 타격 위치와 일치하며, 'Visual Acoustic Fields'를 통해 충격 지역이나 객체를 소리에 기반하여 정밀하게 검색할 수 있음을 증명하였습니다. 이 연구는 로봇 공학, 가상 현실 및 콘텐츠 생성 등 다양한 분야에서 응용될 가능성을 제시합니다.



### FakeScope: Large Multimodal Expert Model for Transparent AI-Generated Image Forensics (https://arxiv.org/abs/2503.24267)
- **What's New**: 이 논문에서는 AI 생성 이미지의 감별을 위한 새로운 모델인 FakeScope를 제안합니다. 이 모델은 정확하게 AI가 합성한 이미지를 찾아내는 것뿐만 아니라 해석 가능하고 쿼리 기반의 포렌식 인사이트를 제공합니다. 또한, FakeChain이라는 대규모 멀티모달 데이터셋을 구축하여 이미지 진위에 대한 언어적 근거를 제공합니다.

- **Technical Details**: FakeScope는 AI 생성 이미지 포렌식의 전문 멀티모달 모델로서, 탐지 정확도를 높이고, 사용자 질의에 대해 신뢰성 있는 답변을 제공합니다. 이 모델은 포렌식 지식을 활용하여 LMM의 포렌식 인식을 향상시키며, 데이터셋인 FakeInstruct는 200만 개의 시각적 지침을 포함하여 다양한 포렌식 기능을 강화합니다.

- **Performance Highlights**: FakeScope는 폐쇄형 및 개방형 포렌식 시나리오 모두에서 최첨단 성능을 달성합니다. 이 모델은 합성 이미지를 높은 정확도로 구별하는 동시에 일관된 설명과 세밀한 위조 속성에 대한 분석을 제공합니다. 특히, FakeScope는 정성적 하드 레이블만을 학습하였음에도 불구하고 제로 샷(Zero-shot) 능력을 발휘하여 탐지에서 뛰어난 성과를 보여줍니다.



### Beyond a Single Mode: GAN Ensembles for Diverse Medical Data Generation (https://arxiv.org/abs/2503.24258)
- **What's New**: 이 논문은 Generative Adversarial Networks (GANs)의 집합을 사용하여 의료 이미징에서 인공 데이터 생성을 위한 새로운 방법을 제안합니다. GANs는 높은 품질 샘플 생성, 다양한 모드 커버리지를 제공하지만, 여전히 여러 문제를 겪고 있습니다. 이를 해결하기 위해, 저자들은 다중 목표 최적화 문제를 해결하여 최적의 GAN 앙상블을 선택하는 새로운 방법을 제시합니다.

- **Technical Details**: 이 연구는 22개의 다양한 GAN 아키텍처를 포함하고 있으며, 각 모델의 훈련 단계에서 샘플링하여 총 110개의 고유 구성을 만들었습니다. 생성된 의료 이미지의 품질을 높이기 위해, GAN 앙상블의 각 모델은 고유한 기여를 하여 중복성을 최소화합니다. 다수의 아키텍처와 훈련 방법을 통합하여 리얼 데이터의 복잡성을 잘 반영하는 강력한 앙상블을 구축하는 것을 목표로 합니다.

- **Performance Highlights**: 저자들은 제안한 GAN 앙상블 방법이 세 가지 의료 데이터 세트를 통해 수행된 광범위한 평가에서 우수한 결과를 보여줌을 입증했습니다. 이 연구는 기본 연구 및 임상 훈련, 알고리즘 검증에서의 활용성을 높이는데 중요한 기여를 합니다. 궁극적으로, 이 방식은 진단 모델링과 같은 다운스트림 작업의 효율성을 개선하는 데 기여할 것입니다.



### Pre-training with 3D Synthetic Data: Learning 3D Point Cloud Instance Segmentation from 3D Synthetic Scenes (https://arxiv.org/abs/2503.24229)
- **What's New**: 최근 3D point cloud 데이터의 활용이 증가하고 있으며, 이는 로봇 및 차량의 기계적 제어와 같은 다양한 실제 응용분야에 적용되고 있습니다. 본 논문에서는 3D 포인트 클라우드 인스턴스 분할(instance segmentation) 개선을 목표로 하며, 이를 위한 새로운 접근법으로 generative model을 이용한 3D 합성 데이터(pre-training using 3D synthetic data)를 제안합니다. 이 과정은 수작업 데이터 주석의 필요성을 줄여 3D point cloud 데이터셋 생성을 보다 효율적으로 할 수 있게 합니다.

- **Technical Details**: 3D 포인트 클라우드 인스턴스 분할은 개별 물체 인스턴스에 대한 픽셀 특정 위치 정보(infer pixel-specific position information)를 추론할 수 있는 기술입니다. 본 연구는 Point-E라는 3D 생성 모델을 사용하여 자동으로 3D 포인트 클라우드 데이터를 생성하고, 이를 3D 씬에 삽입하여 데이터셋을 확장합니다. 이를 통해 수작업으로 주석을 달 필요 없이 대규모의 다양하고 세부적인 인스턴스/포인트 수준의 주석을 가진 데이터를 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과, 우리가 제안한 3D 합성 데이터로 사전 학습(pre-training)한 모델은 전통적인 수기 주석 데이터셋을 통해 얻은 성능을 초과하거나 비슷한 성능을 보였습니다. 이는 3D 포인트 클라우드 인스턴스 분할 솔루션의 개발 속도를 높이고, 실제 응용에서의 광범위한 채택을 촉진하는 데 큰 도움이 됩니다. 결과적으로, 본 연구는 3D 인식 분야의 데이터셋 준비 문제를 해결하기 위한 중요한 기반을 마련하고 있습니다.



### MB-ORES: A Multi-Branch Object Reasoner for Visual Grounding in Remote Sensing (https://arxiv.org/abs/2503.24219)
- **What's New**: 이 논문에서는 원격 감지 이미지에 대해 객체 검출(Object Detection, OD)과 시각적 기초(Visual Grounding, VG)를 통합하는 통합 프레임워크를 제안하고 있습니다. 전통적인 OD와 VG 작업을 위한 직관적인 사전 지식을 수립하기 위해, 언급 표현 데이터를 사용하여 오픈 세트 객체 감지기를 세밀 조정하고, 부분적으로 감독된 OD 작업으로 설정합니다. 이러한 구조를 통하여 모든 객체를 탐지하면서 특정 객체의 위치를 정확하게 찾는 것을 목표로 합니다.

- **Technical Details**: 제안된 모델은 객체 질의, 클래스 임베딩, 및 제안 위치로 구성된 그래프 표현을 사용하여 각 이미지를 구성합니다. 멀티-브랜치 네트워크는 공간적, 시각적, 범주적 특성을 통합하여 작업 인식 제안을 생성하며, 객체 추론 네트워크는 제안들 사이의 확률을 할당합니다. 이 과정은 마지막으로 언급된 객체를 로컬라이즈하기 위한 부드러운 선택 메커니즘으로 이어집니다.

- **Performance Highlights**: 이 방법은 OPT-RSVG 및 DIOR-RSVG 데이터 세트에서 뛰어난 성능을 입증하였으며, 최신 방법들에 비해 상당한 성능 개선을 보여 주었습니다. 전통적인 OD 기능을 유지하면서도 보다 다양한 시나리오에서 OD의 적용 가능성을 확대하였습니다. 또한, 이 논문의 코드는 연구 결과를 재현하고 실험할 수 있도록 제공될 예정입니다.



### DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting (https://arxiv.org/abs/2503.24210)
Comments:
          CVPR 2025. Project Page: this https URL

- **What's New**: 이번 논문에서는 DiET-GS라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 블러가 있는 다중 시점 이미지에서 선명한 3D 표현을 재구성할 수 있는 기술을 제공합니다. 여기서는 이 프레임워크를 통해 이벤트 스트림과 확산(회복) 모델을 결합하여 최적의 시각적 품질을 달성하는 방법을 설명합니다.

- **Technical Details**: DiET-GS는 두 단계의 훈련 전략을 사용하여 이벤트 스트림과 사전 훈련된 확산 모델의 지식을 효과적으로 결합합니다. 이를 통해 정밀한 색상 정보와 세부 정보를 재구성할 수 있는 새로운 수학적 프레임워크를 소개하며, Event Double Integral(EDI)을 사용하여 3DGAussian Splatting(3DGS)의 정교한 세부 정보를 회복합니다.

- **Performance Highlights**: 실험 결과, DiET-GS는 기존의 방법들과 비교하여 시각적 품질에서 크게 향상된 성능을 보여줍니다. 이는 합성 데이터와 실제 데이터 모두에서 검증되었으며, 최적의 색상과 세밀한 디테일을 복구하는 능력을 갖췄음을 입증합니다. 또한, 새로운 기술적 접근 방식이 기존의 이미지 복원 모델들보다 더 뛰어난 성능을 발휘합니다.



### CIBR: Cross-modal Information Bottleneck Regularization for Robust CLIP Generalization (https://arxiv.org/abs/2503.24182)
- **What's New**: 이번 연구는 Contrastive Language-Image Pretraining (CLIP)의 강력한 일반화 성능 뒤에 있는 이론적 기초를 탐구합니다. Cross-modal Information Bottleneck (CIB) 프레임워크를 제안하여 CLIP의 대조 학습 목표를 정보 병목 최적화로 해석합니다. CIB는 서로 다른 모달리티 간의 핵심 의미를 보존하면서도 모달리티 특이적 중복을 폐기하는 방식으로 교차 모달 정보 공유를 극대화합니다.

- **Technical Details**: CIBR (Cross-modal Information Bottleneck Regularization) 방법론은 학습 과정에서 정보 병목 원리를 명시적으로 적용해 이미지와 텍스트 특성 간의 의미 정렬을 강화합니다. 이 방법은 모달리티 특이적 중복을 억제하는 패널티 항을 도입하여 CLIP 모델이 최적의 교차 모달 표현 학습을 진행하도록 인도합니다. 연구에서 이론적 통찰을 바탕으로 한 모델의 유효성을 다양한 비전-언어 기준 데이터셋을 통해 검증하였습니다.

- **Performance Highlights**: CIBR은 7개 다양한 이미지 데이터셋에 대한 제로샷 분류와 MSCOCO 및 Flickr30K에서의 텍스트-이미지 검색에서 일관된 성능 향상을 보여주었습니다. 이 결과는 CLIP의 일반화에 대한 처음으로 제시하는 이론적 배경을 제공하며, 교차 모달 표현 학습의 향후 방향을 제시합니다. CIBR 방법론은 대조적 다중 모달 학습에 있어 이론에 기반한 접근의 실용적 가치를 증명합니다.



### Navi-plus: Managing Ambiguous GUI Navigation Tasks with Follow-up (https://arxiv.org/abs/2503.24180)
- **What's New**: 이번 연구에서는 Self-Correction GUI Navigation이라는 새로운 작업을 도입하여 GUI 자동화 에이전트가 사용자 의도에서 불확실한 부분을 해결할 수 있도록 합니다. 이 작업은 "ASK" 액션을 추가함으로써 에이전트가 사용자의 추가 질문을 통해 정보의 공백을 메우도록 지원합니다. 또한, Navi-plus 데이터 세트를 개발하여 GUI 후속 질문-응답 쌍을 포함시켰습니다.

- **Technical Details**: 이 연구는 사용자 작업을 설명할 때 중요한 정보가 누락될 수 있는 문제를 해결하기 위해 GUI 자동화 에이전트의 새로운 기능을 평가합니다. 에이전트는 사용자가 제공하는 정보를 보완하기 위해 ASK 액션을 사용하여 자연어로 소통하며, 이러한 상호작용이 에이전트의 내비게이션 진행에 기여하게 됩니다. Dual-Stream Trajectory Evaluation 방식이 도입되어, ASK 액션 전후의 성능을 보다 명확히 평가할 수 있습니다.

- **Performance Highlights**: 실험 결과, 누락된 정보를 처리하는 ASK 액션 덕분에 에이전트는 원래 성능의 99.4% 이상을 회복할 수 있음을 보여주었습니다. 또한, 현대 MLLM 기반 GUI 에이전트는 0.932 이상의 타이밍 정확도와 0.807 이상의 내용 정확도로 후속 질문을 효과적으로 제안할 수 있는 능력을 갖추었습니다. 추가적인 후속 질문을 통해 에이전트의 작업 성공률이 26.5% 상승했습니다.



### Foundation Models For Seismic Data Processing: An Extensive Review (https://arxiv.org/abs/2503.24166)
- **What's New**: 이 논문은 지진처리(Seismic Processing) 분야에서 깊은 학습(Deep Learning) 접근법이 아닌 기초 모델(Foundation Models)의 적용을 조사합니다. 기초 모델은 자연 이미지 처리(Natural Image Processing)에서의 성공을 바탕으로, 실제 육상 지진 데이터에서 학습한 후 특정 다운스트림 작업(Downstream Tasks)에 세밀하게 조정되어 사용될 수 있습니다. 본 연구는 여러 자연 이미지 기초 모델을 비판적으로 검토하고, 향후 탐색에 적합한 몇 가지 후보를 제안합니다.

- **Technical Details**: 저자들은 변환기(Transformer) 기반의 인코더-디코더 구조를 사용하여 기초 모델의 성능을 벤치마킹할 수 있는 프레임워크를 개발했습니다. 이 모델은 입력의 두 차원 공간에서 매핑하여 출력의 두 차원 공간으로 변환하는 과정을 정의하며, 각 인코더가 세 가지 차원 공간에서의 임베딩(Embedding)을 출력하도록 설계되었습니다. 이러한 정의를 통해 다양한 모델 특성들이 성능에 미치는 영향을 분석합니다.

- **Performance Highlights**: 논문에서 제시된 프레임워크는 demultiple, interpolation 및 denoising 작업에서 다양한 기초 모델의 성능을 평가합니다. 특히, 스코어코드를 오픈 소스 데이터셋과 결합하여 범위를 확장하고, 성능 평가를 정량적 및 정성적으로 진행합니다. 이러한 연구는 기초 모델을 기반으로 한 지진 처리의 발전 가능성을 제시하며, 자연 이미지 사전 훈련(Natural Image Pre-training)의 효과를 분석하는 데 중점을 둡니다.



### PixelCAM: Pixel Class Activation Mapping for Histology Image Classification and ROI Localization (https://arxiv.org/abs/2503.24135)
Comments:
          32 pages, 20 figures, Medical Imaging with Deep Learning (MIDL 2025)

- **What's New**: 본 논문에서는 약하게 감독된 객체 위치 지정 기법인 PixelCAM을 소개합니다. 이 방법은 픽셀-특징 공간에서 전방향 배경 분류기를 사용하여 이미지 분류와 객체 위치 지정을 동시에 수행합니다. PixelCAM은 기존의 방법들의 비동기 수렴 문제를 해결하고, OOD(Out-Of-Distribution) 데이터에 대한 강인성을 제공합니다.

- **Technical Details**: PixelCAM은 사전 학습된 WSOL 모델에서 수집한 픽셀 의사 라벨을 사용하여 훈련됩니다. 이 모델은 CNN(Convolutional Neural Network) 및 Transformer 기반 아키텍처에 쉽게 통합될 수 있으며, 픽셀-레벨과 이미지-레벨의 분류기가 동시에 훈련됩니다. 또한, 다중 작업 최적화를 통해 두 가지 작업(분류 및 위치 지정)을 동시에 수행하여 특징 학습을 향상시킵니다.

- **Performance Highlights**: 두 개의 공공 데이터 세트(GlaS 및 CAMELYON16)에서 수행된 실험 결과, PixelCAM은 기존의 WSOL 기법들보다 뛰어난 성능을 보여주었습니다. 특히, 다양한 암 유형 간의 큰 도메인 이동에서도 높은 정확도를 유지하며, OOD 시나리오에 적합한 선택으로 평가되었습니다.



### It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data (https://arxiv.org/abs/2503.24129)
Comments:
          Accepted to CVPR 2025, Project page: this https URL

- **What's New**: 이번 논문에서는 비전과 언어의 기본 모델들이 발전함에 따라 표현의 동질화(homogeneity)가 증가한다는 이론을 제시합니다. 특히, 서로 다른 모달리티(modality) 간의 거리(pairwise distance)가 더욱 유사해진다는 점에 주목하고, 이를 통해 전통적인 데이터 쌍이 없이도 '블라인드' 방식으로 비전과 언어 표현을 매칭하는 가능성을 검토합니다. 기존 연구에 대한 비판적 시각을 살펴보며, 우리가 제시하는 방법이 비전-언어 정렬의 새로운 가능성을 열어줄 것임을 강조하였습니다.

- **Technical Details**: 이 연구는 쌍관계 문제를 quadratic assignment problem (QAP)의 형태로 수학적으로 공식화합니다. 제안된 새로운 heuristic 기법은 기존의 알고리즘보다 더 효율적이며, 최적 매칭 문제에 대한 해결책을 제시합니다. 비전-언어 표현 간의 매칭을 평가하기 위해 33개의 비전 및 27개의 언어 모델을 사용해 대규모 연구를 수행하였으며, 이 과정에서 pairwise distances를 활용하여 유의미한 결과를 도출하였습니다.

- **Performance Highlights**: 연구 결과, 많은 비전-언어 과제에서 무감독 상태에서도 비전과 언어 표현을 유의미하게 매칭할 수 있음을 보여주었습니다. 이는 주석 없는 상태에서도 이미지의 의미를 분류할 수 있는 가능성을 여는 혁신적인 성과입니다. 제시된 무감독 분류기는 이미지-텍스트 대응이 전혀 없이도 분류 정확도를 달성할 수 있는 효능을 입증하였습니다.



### IMPACT: A Generic Semantic Loss for Multimodal Medical Image Registration (https://arxiv.org/abs/2503.24121)
Comments:
          Submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). This is a preprint version and has not been peer-reviewed

- **What's New**: 이번 연구에서는 의료 영상에서의 정합(Registration)을 위한 새로운 유사도 측정 기법인 IMPACT(Image Metric with Pretrained model-Agnostic Comparison for Transmodality registration)를 소개합니다. IMPACT는 다양한 이미지 등록 프레임워크(예: Elastix, Voxelmorph)에 통합될 수 있도록 설계된 일반적인 의미적 유사도 메트릭입니다. 이 메트릭은 특정 작업에 대한 훈련 없이 의료 영상에서 추출된 딥러닝 기반 피쳐를 비교함으로써 여러 가지 모달리티에서 폭넓게 적용 가능하게 합니다.

- **Technical Details**: IMPACT는 대규모 사전 훈련된 TotalSegmentator 모델의 피쳐와 Segment Anything Model(SAM) 및 기타 대규모 세분화 네트워크를 통합하여 이점이 있습니다. 이 방법은 강건하고 확장 가능하며 효율적인 멀티모달 이미지 등록 솔루션을 제공합니다. 연구팀은 IMPACT 손실을 흉부 CT/CBCT 및 골반 MR/CT 데이터셋을 포함한 다섯 개의 도전적인 등록 작업에 대해 평가했습니다.

- **Performance Highlights**: 수치 메트릭(예: Target Registration Error, Dice Similarity Coefficient)은 기존 방법 대비 해부학적 정렬에서 유의미한 개선을 보였습니다. 질적 분석에서도 노이즈, 아티팩트 및 모달리티 변동에 강한 Robustness를 확인했습니다. IMPACT는 임상 및 연구 응용에서 등록 성능을 향상시키는 데 기여할 수 있는 유용한 도구로, 멀티모달 의료 영상의 주요 도전 과제를 해결하고 있습니다.



### PolypSegTrack: Unified Foundation Model for Colonoscopy Video Analysis (https://arxiv.org/abs/2503.24108)
- **What's New**: 이 논문에서는 colonoscopic 비디오에서 폴립(polyp)의 감지, 분할(segmentation), 분류(classification) 및 비지도 추적(unsupervised tracking)을 동시에 수행할 수 있는 새로운 기초 모델인 PolypSegTrack을 제안합니다. 기존의 방법들은 각각의 작업(Task)별로 특정한 미세 조정(fine-tuning)이 필요하거나, 추적 기능이 부족하거나, 도메인 특화된 사전 훈련(pre-training)에 의존하고 있었습니다. 제안된 방법은 새로운 조건부 마스크 손실(conditional mask loss)을 활용하여 데이터셋 사이의 유연한 훈련을 가능하게 하며, 이로 인해 특정 작업에 대한 미세 조정을 생략할 수 있게 됩니다.

- **Technical Details**: 제안한 PolypSegTrack 모델은 두 단계로 구성되어 있으며, 첫 번째 단계에서는 각 프레임의 바운딩 박스(bounding boxes), 세그멘테이션 마스크(segmentation masks), 클래스 확률(class probabilities)을 생성합니다. 두 번째 단계에서는 연속된 두 프레임 간의 객체를 매칭하여 추적을 수행합니다. 이 과정에서 비지도 및 비휴리스틱(Non-heuristic) 추적 방법이 사용되며, 객체 쿼리(object queries)를 통해 폴립 인스턴스를 신뢰성 있게 연결합니다. 모델은 자연 이미지에서 비지도(pre-trained on natural images)로 사전 훈련되어 도메인 특화된 데이터에 대한 의존도를 줄이고 있습니다.

- **Performance Highlights**: PolypSegTrack 모델은 ETIS, CVC-ColonDB, CVC-300, Kvasir-SEG 및 CVC-Clinic-DB 데이터셋으로 수행한 다양한 테스트에서 기존의 최첨단 방법들보다 월등한 성과를 거두었습니다. 모델은 검출(detection), 분할(segmentation), 분류(classification), 그리고 추적(tracking) 작업에서 모두 뛰어난 결과를 보여주었습니다. 이 모델은 폴립 진단의 속도와 정확성 및 일관성을 획기적으로 향상시킬 수 있는 잠재력을 가지고 있습니다.



### DANTE-AD: Dual-Vision Attention Network for Long-Term Audio Description (https://arxiv.org/abs/2503.24096)
- **What's New**: 이 논문에서는 DANTE-AD라는 새로운 오디오 설명 모델을 소개합니다. 환경의 장기적인 시각적 서사 유지 문제를 해결하기 위해, 이 모델은 이중 비전 Transformer 기반 아키텍처를 활용하여 프레임 및 장면 수준의 임베딩을 순차적으로 융합합니다. DANTE-AD는 세밀한 오디오 설명 생성을 위한 문맥적 기초를 달성하기 위한 새로운 방식의 순차적 크로스 어텐션을 제안합니다.

- **Technical Details**: DANTE-AD는 두 개의 병렬 기능 추출 브랜치로 구성되어 있으며, 하나는 프레임 수준의 임베딩을, 다른 하나는 전역 장면 수준의 표현을 처리합니다. 모델은 별도의 공간 프레임 세부정보와 장기적인 시간적 컨텍스트 장면 정보를 통합하는 이중 비전 Transformer 네트워크를 사용합니다. 최종적으로, 융합된 정보는 언어 모델을 통해 자연어로 디코딩됩니다.

- **Performance Highlights**: DANTE-AD는 유명한 영화 클립의 다양한 주요 장면에 대한 평가에서 기존 방법들을 초월하여 전통적인 NLP 지표와 LLM 기반 평가에서 우수한 성능을 보였습니다. 이를 통해 모델의 장기적인 문맥 이해 능력이 개선되었음을 보여줍니다.



### 4D mmWave Radar in Adverse Environments for Autonomous Driving: A Survey (https://arxiv.org/abs/2503.24091)
Comments:
          8 pages

- **What's New**: 이번 연구는 자율주행 시스템에 대한 4D mmWave 레이더의 성능을 개괄적으로 정리한 첫 번째 연구로, 기존의 연구에서 부족했던 악조건에서의 4D mmWave 레이더 데이터셋과 방법론을 포함하고 있습니다. 활성화된 연구는 다양한 기상 및 조명 조건을 고려한 4D mmWave 레이더의 효용성을 강조하며, 이 기술이 자율주행에서 점차 중요해지고 있음을 시사합니다. 또한, 4D mmWave 레이더의 발전을 위한 다각적인 연구 방향과 도전 과제를 논의합니다.

- **Technical Details**: 4D mmWave 레이더는 기존의 3D mmWave 레이더에 높이 측정을 추가하여 3D 공간 인식을 향상시킵니다. 이 레이더는 특히 비, 눈, 안개와 같은 악조건에서도 뛰어난 성능을 발휘합니다. 연구에서는 다양한 기존 4D mmWave 레이더 데이터셋을 분석하고, 특정 기상 조건에 맞춘 기술적 접근 방식을 검토하여 자율주행에 필요한 인식 및 동시 장소 기록(SLAM) 작업에 공헌합니다.

- **Performance Highlights**: 4D mmWave 레이더는 다양한 기상 조건에서도 신뢰성을 갖춘 성능을 발휘하며, LiDAR 및 카메라에 비해 뛰어난 장점을 보여 줍니다. 연구에 따르면, 4D mmWave 레이더는 작은 공기 중 입자를 관통할 수 있어 어려운 환경에서도 일관된 성능을 유지합니다. 특히, 4D mmWave 레이더는 비와 안개와 같은 복잡한 조건에서도 높은 정확도를 유지하며 향후 자율주행 시스템에 더 큰 기여를 할 수 있는 가능성을 보여주고 있습니다.



### A Plasticity-Aware Method for Continual Self-Supervised Learning in Remote Sensing (https://arxiv.org/abs/2503.24088)
Comments:
          Accepted at IEEE International Geoscience and Remote Sensing Symposium 2025

- **What's New**: 이번 논문에서는 원거리 감지(Remote Sensing, RS) 분야에서 새로운 작업을 연속적으로 학습할 수 있는 지속적 자가 지도 학습(Continual Self-Supervised Learning, CSSL) 방법을 제안합니다. 기존의 CSSL 방법들은 재난적 망각(catastrophic forgetting)을 방지하는 데 주안점을 두었지만, 이로 인해 새로운 작업 데이터에 대한 적응력이 저하되는 문제가 발생합니다. 본 연구에서는 이러한 문제를 해결하기 위해 지식 증류(knowledge distillation) 전략과 결합된 새로운 분리 메커니즘을 활용한 방법을 제안합니다.

- **Technical Details**: 제안된 방법은 작업 공통 및 특정 부분으로 피쳐 차원을 분리하여 학습을 진행합니다. 작업 공통 특징은 메모리 안정성을 보장하기 위해 연관성을 유지하도록 강제되며, 작업 특정 특징은 새로운 피쳐 학습을 촉진하기 위해 분리되어야 합니다. 본 연구는 BarlowTwins 프레임워크를 활용하여 일반적인 특징 인코더를 구축하며, 이 과정에서 제안된 지식 증류 전략을 사용합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 기존의 CaSSLe CSSL 프레임워크에 비해 평균 정확도에서 최대 1.12%, 저항성(intransigence)에서는 최대 2.33% 향상된 성능을 보였습니다. 또한, 클래스 증가 상황에서 평균 정확도 1.24%, 저항성 2.01%의 개선 결과를 얻었습니다. 이는 제안된 방법이 지속적 학습에서의 세부적 성능 향상을 가능하게 함을 보여줍니다.



### From Colors to Classes: Emergence of Concepts in Vision Transformers (https://arxiv.org/abs/2503.24071)
Comments:
          Preprint. Accepted at The 3rd World Conference on eXplainable Artificial Intelligence

- **What's New**: 이번 연구에서는 Vision Transformers (ViTs)의 레이어별 정보 처리 과정을 분석하여, 각 레이어에서 인코딩되는 개념들의 복잡도를 조사합니다. 기존의 연구는 주로 Convolutional Neural Networks (CNNs)에 중점을 두었으나, ViTs에 대한 레이어-wise 분석은 부족했던 점을 보완합니다. 연구 결과, ViTs가 초기 레이어에서 기본적인 특징을 인코딩하고 후반 레이어에서 더 복잡한 개념을 점차적으로 학습한다는 사실을 확인했습니다.

- **Technical Details**: 본 연구에서는 CLIP-dissect 방법을 사용하여 ViTs의 레이어별 학습 프로세스를 분석합니다. 이 방법은 네트워크의 각 뉴런에 관련된 개념을 식별하기 위한 신뢰성 있는 neuron labeling 기법을 제공합니다. 초기 레이어는 주로 색상과 질감과 같은 기본 특징을 인코딩하며, 후반 레이어에서는 물체와 자연 요소와 같은 더 전문화된 개념을 인코딩합니다.

- **Performance Highlights**: 연구 결과, ViTs는 초기 레이어에서 더 보편적인 개념을, 후반 레이어에서는 더욱 다채롭고 전문화된 개념을 인코딩하여 레이어별 특징 추출의 계층 구조를 드러냅니다. 또한, 특정 하향 작업에 대한 미세 조정(finetuning)은 인코딩된 개념의 수를 줄이고, 보다 관련성 있는 개념으로 이동하게 하는 경향이 있습니다.



### COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation (https://arxiv.org/abs/2503.24065)
- **What's New**: 이번 논문에서는 Vision-and-Language Navigation (VLN) 문제를 다루고 있으며, COmbination of Selective MemOrization (COSMO)라는 새로운 아키텍처를 제안합니다. COSMO는 기존의 Transformer 아키텍처에 비해 높은 성능을 유지하면서도 낮은 계산 비용을 목표로 하고 있습니다. 이를 위해, Round Selective Scan (RSS)와 Cross-modal Selective State Space Module (CS3)이라는 두 개의 VLN 맞춤형 선택적 상태 공간 모듈을 통합하였습니다.

- **Technical Details**: COSMO 아키텍처는 상태 공간 모듈(state-space modules)과 Transformer 모듈을 통합하고, 선택적 SSM을 적용하여 VLN 작업에 특화된 컴포넌트를 더합니다. RSS는 단일 스캔 내에서 포괄적인 상호작용을 가능하게 하며, CS3 모듈은 이중 스트림 아키텍처로 적응하여 크로스 모달 상호작용을 강화합니다. 또한, 선택적 SSM과 Transformer의 조합을 통해 메모리 필터링 능력을 통합하여 효율성을 극대화합니다.

- **Performance Highlights**: REVERIE, R2R, R2R-CE와 같은 세 가지 VLN 벤치마크에서 실험한 결과, COSMO는 이전의 DUET 모델보다 우수한 내비게이션 성능을 보여주었습니다. 성능 개선은 SR과 SPL에서 각각 +3.83% 및 +2.2%의 절대적 향상을 보였으며, 총 파라미터 수는 15.5%에 불과하고 FLOPs는 9.3%로 대폭 줄어들었습니다. 이는 계산 비용을 현저히 줄이면서도 높은 성능을 달성했음을 나타냅니다.



### AMMSM: Adaptive Motion Magnification and Sparse Mamba for Micro-Expression Recognition (https://arxiv.org/abs/2503.24057)
Comments:
          Accepted by ICME 2025

- **What's New**: 이번 연구에서는 미세표정(micro-expressions)의 인식 문제를 해결하기 위해 Adaptive Motion Magnification and Sparse Mamba (AMMSM)이라는 다중 작업 학습(framework)을 제안합니다. 이 프레임워크는 자기 지도 학습(self-supervised learning)을 통해 미세한 동작을 확대하고, Sparse Mamba 구조를 통해 공간적으로 중요한 동작 영역을 모델링합니다. 특히, 진화적 탐색(evolutionary search)을 활용하여 확대 정도(magnification factor)와 선택의 희소성(sparsity ratios)을 최적화합니다.

- **Technical Details**: AMMSM은 두 가지 모듈로 구성됩니다: 동작 확대(module) 및 분류(module)입니다. 동작 확대 모듈은 확장 목표와 분류 작업을 통합하여 초점을 맞춘 동작만을 정확하게 확대할 수 있도록 훈련됩니다. 이 모델은 시간적 스트림(temporal stream)과 공간적 스트림(spatial stream)으로 나뉘며, 이 두 가지를 결합하여 더 의미 있는 표현(representations)을 학습합니다.

- **Performance Highlights**: 실험 결과, AMMSM은 CASME II 및 SAMM 데이터셋에서 최첨단 성과(SOTA)를 달성했습니다. 다중 벤치마크에서 검증을 통해 제안된 방안의 유효성이 입증되었으며, 기법의 효과성을 확인하기 위한 제안된 방법의 ablation studies도 수행되었습니다.



### BBoxCut: A Targeted Data Augmentation Technique for Enhancing Wheat Head Detection Under Occlusions (https://arxiv.org/abs/2503.24032)
- **What's New**: 이번 연구에서는 밀 수확량의 정확한 예측을 위해 새로운 데이터 증강 기법인 BBoxCut을 제안합니다. 이 기법은 잎이나 이웃 밀 헤드로 인한 장애물(occlusion)을 시뮬레이션하기 위해 무작위로 국소 마스킹을 적용합니다. 기존의 수작업 방법이 아닌 자동화된 시스템을 통해 시간과 효율성을 크게 개선할 수 있는 가능성을 보여줍니다. BBoxCut을 통해 Faster R-CNN, FCOS 및 DETR 모델에서 평균 정밀도(mean Average Precision)가 현저히 향상되었습니다.

- **Technical Details**: 밀 헤드의 정확한 탐지를 위해 BBoxCut은 국소적으로 경계 상자(bounding box)의 일부를 가리는 방식으로 작동합니다. 이는 예를 들어 잎이나 이웃 밀 헤드가 부분적으로 가려진 상황을 보다 현실적으로 모사합니다. 데이터 증강 파이프라인에 BBoxCut을 통합한 결과, 특히 장애물이 있는 시나리오에서 mAP의 유의미한 향상을 발견하였으며, 이는 농민의 수익성에 직접적인 영향을 미칠 수 있습니다.

- **Performance Highlights**: 제안된 BBoxCut 기법은 장애물이 있는 상황에서 밀 헤드를 감지하는 성능을 크게 향상시킵니다. 세 가지 최신 객체 탐지기(Faster R-CNN, FCOS, DETR)를 평가한 결과, 평균 2.76, 3.26, 1.9의 mAP 향상을 보였습니다. 이러한 개선의 효과는 특히 방해 요소가 있는 시나리오에서 뚜렷하게 나타났으며, 이는 필드 조건에서의 방법의 강건성을 잘 보여줍니다.



### HumanDreamer: Generating Controllable Human-Motion Videos via Decoupled Generation (https://arxiv.org/abs/2503.24026)
Comments:
          Project Page: this https URL

- **What's New**: 이번 연구에서는 HumanDreamer라는 새로운 프레임워크를 제안하여 인간-모션 비디오 생성의 flexibility를 향상시켰습니다. 이 프레임워크는 텍스트 프롬프트에서 다양하고 구조화된 모션 포즈를 생성한 후, 이를 기반으로 고품질 인간-모션 비디오를 제작합니다. HumanDreamer는 1.2M개의 텍스트-포즈 쌍을 포함한 MotionVid 데이터셋을 활용하여, 기존 방법들보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: HumanDreamer는 Text-to-Pose와 Pose-to-Video 두 단계를 거치는 분리된 구조를 가지고 있습니다. 새로운 MotionDiT 모델은 전 세계적인 주의 메커니즘(global attention mechanism)과 주변 포즈 포인트의 정보를 집계하는 로컬(feature aggregation) 메커니즘을 통합하여 포즈의 품질을 향상시킵니다. 또한, 새롭게 도입된 LAMA loss는 동작의 의미적 특징을 정렬하여 모델의 해석 가능성과 전반적인 성능을 개선합니다.

- **Performance Highlights**: 제안된 방법은 FID에서 62.4%의 향상을 보였으며, top1, top2, top3에서 각각 41.8%, 26.3%, 18.3%의 R-precision 개선이 확인되었습니다. 이는 인간-모션 비디오의 질적 향상을 의미하며, 생성된 포즈는 포즈 시퀀스 예측 및 2D-3D 모션 변환과 같은 다운스트림 작업을 지원할 수 있습니다.



### Crossmodal Knowledge Distillation with WordNet-Relaxed Text Embeddings for Robust Image Classification (https://arxiv.org/abs/2503.24017)
- **What's New**: 본 논문에서는 unimodal 학생 모델의 성능을 향상시키기 위한 multi-teacher crossmodal knowledge distillation (KD) 프레임워크를 제안합니다. 이 방법론은 CLIP 이미지 임베딩과 학습 가능한 WordNet-relaxed 텍스트 임베딩을 계층적 손실(hierarchical loss) 구조 하에 통합하여 라벨 유출(label leakage)을 완화하고, 보다 다양한 텍스트 신호를 제공합니다. 실험 결과, 이러한 전략이 기존 방식보다 학생 모델의 성능을 크게 향상시키는 데 기여함을 보여줍니다.

- **Technical Details**: 제안하는 방법에서는 WordNet이라는 의미론적 관계를 가진 단어 데이터베이스를 활용하여 CLIP 텍스트 임베딩의 의미적 풍부성을 높입니다. 학생 모델은 단순히 이미지에만 의존하는 경우 여전히 교사의 텍스트와 이미지의 상호작용을 통해 학습을 강화하는 데 필요한 다양한 신호를 수신할 수 있습니다. 계층적 손실과 코사인 정규화(cosine regularization)를 도입하여 정확한 클래스 이름 사용을 피하고, 일반적인 시각적 모달리티 특징을 강조함으로써 보다 강건한 학습이 가능합니다.

- **Performance Highlights**: 우리의 방법은 여섯 개의 공공 데이터셋에서 state-of-the-art(SOTA) 또는 최소한 두 번째로 우수한 성능을 기록하며, 이는 crossmodal KD 분야에서의 유의미한 발전을 보여줍니다. 또한, WordNet 기반의 정규화가 강력한 시각적 특징에 대한 의존도를 높이고, 텍스트 암기를 줄이며, 새로운 텍스트 신호를 효과적으로 활용함을 입증하였습니다. 이러한 결과들은 제안하는 접근 방식이 실질적인 성능 향상에 기여함을 나타냅니다.



### Optimization of Layer Skipping and Frequency Scaling for Convolutional Neural Networks under Latency Constrain (https://arxiv.org/abs/2503.24014)
Comments:
          12 pages, 6 figures, Accepted in Proc. Eur. Conf. Comput. Vis. (ECCV) Workshops. Milan, Italy: Springer, September 2024

- **What's New**: 이 논문에서는 자원 제한 환경의 CNN(Convolutional Neural Networks)에서 에너지 소비를 최소화하기 위한 새로운 방법인 비례 레이어 건너뛰기(Proportional Layer Skipping, PLS)와 주파수 조절(Frequency Scaling, FS)을 제안합니다. 이 방법은 네트워크 레이어를 선택적으로 우회하여 계산 복잡성을 줄이며, 프로세서의 주파수를 조절하여 에너지 사용을 최적화합니다. 실험을 통해 ResNet-152 아키텍처와 CIFAR-10 데이터셋에서 새로운 방법의 효과성을 입증하였습니다.

- **Technical Details**: CNN 아키텍처는 여러 레이어를 통해 시각 데이터를 분석하며, 레이어 스킵핑과 주파수 조정 기법을 결합하여 에너지 소비를 줄이는 최적화 프레임워크를 제안합니다. 이 연구에서는 ResNet-152의 각 레이어가 서로 다른 그룹으로 구성되어 특정 조건을 만족하며 실행된다고 설명합니다. 이러한 방법은 레이어의 선택적 우회와 주파수 조절을 통해 실시간 처리에서 에너지 효율성을 높이는 것을 목표로 합니다.

- **Performance Highlights**: PLS와 FS 기법을 통해 계산 수요와 에너지 소비를 크게 줄이고, CPU 및 GPU 아키텍처 모두에서 경쟁력 있는 정확도를 유지하는 데 성공했습니다. 실험 결과는 제한된 리소스 환경에서의 실시간 처리 효율성을 향상시킬 수 있는 실제 솔루션을 제공합니다. 이 연구는 에너지 효율성과 모델 정확도 간의 균형을 이해하는 데 중요한 통찰력을 제공합니다.



### H2VU-Benchmark: A Comprehensive Benchmark for Hierarchical Holistic Video Understanding (https://arxiv.org/abs/2503.24008)
- **What's New**: 이번 논문에서는 비디오 이해 능력을 평가하기 위한 새로운 H2VU 벤치마크를 제안합니다. 현재의 벤치마크가 지닌 한계를 극복하고자 짧은 비디오와 1.5시간에 걸친 긴 레코딩을 포함한 다양한 비디오 길이를 평가할 수 있습니다. 그뿐만 아니라, 새로운 획기적인 평가 모듈인 'Counterfactual Reasoning'과 'Trajectory State Tracking'을 통해 단순한 지식 기반을 넘어서는 모델의 깊은 이해 능력을 검증합니다.

- **Technical Details**: H2VU-Benchmark는 비디오 이해 모델 평가를 위해 총 10,183개의 평가 작업을 포함하는 3단계 계층적 역량 분류 시스템을 개발했습니다. 일반 오프라인 비디오와 온라인 스트리밍 비디오라는 두 가지 주요 영역을 통해 평가를 진행하며, 각기 다른 유형의 인식 및 추론 작업을 포함합니다. 또한, 이를 통해 비디오 이해 모델의 동적 장면 이해 능력을 심도 있게 평가할 수 있습니다.

- **Performance Highlights**: H2VU의 실험 결과, 기존의 멀티모달 대형 언어 모델(MLLMs)이 새로운 평가 작업에서 상당한 개선 가능성을 지니고 있다는 것을 보여줍니다. 특히, 속임수를 활용한 이해(task)와 상태 궤적 추적(task)에서 모델의 성능 차이가 두드러졌습니다. 이 결과들은 현재 모델들이 비디오 콘텐츠 기반의 응답을 효과적으로 생성하기 위해서는 더 많은 개선이 필요하다는 것을 시사하며, 실제 세계의 비디오 이해 적용에서 여전히 도전 과제가 남아 있음을 강조합니다.



### DenseFormer: Learning Dense Depth Map from Sparse Depth and Image via Conditional Diffusion Mod (https://arxiv.org/abs/2503.23993)
- **What's New**: 본 논문에서는 자율주행에서 필수적인 깊이 완성(depth completion) 작업을 위한 새로운 방법인 DenseFormer를 제안합니다. DenseFormer는 전통적인 방식의 공간 전파 네트워크(spatial propagation network) 대신에 전이(diffusion) 모델을 통합하여 복잡한 깊이 맵을 생성합니다. 특히, 이 방법은 초기 랜덤 깊이 분포를 여러 번의 반복을 통해 점진적으로 개선하여 밀도 높은 깊이 맵을 생성하는 데 초점을 맞추고 있습니다.

- **Technical Details**: DenseFormer는 특징 추출 모듈(feature extraction module)과 깊이 개선 모듈(depth refinement module)을 포함하고 있습니다. 특징 추출 모듈은 다층 변형(attention) 구조를 활용하여 희소 깊이 맵(sparse depth maps)과 RGB 이미지에서 효과적으로 특징을 추출하고 통합합니다. 또한, 깊이 개선 모듈은 전이 과정을 통해 생성된 깊이 결과에 대해 다양한 범위에서 다단계 반복 개선(multi-step iterative refinement)을 적용하여 더욱 향상된 정확성을 제공합니다.

- **Performance Highlights**: KITTI 야외 장면 데이터셋에서 진행된 포괄적인 실험을 통해 DenseFormer가 기존의 클래식 깊이 완성 방법들보다 우수한 성능을 보임을 입증하였습니다. 이 연구 결과는 자율주행 기술에 있어 깊이 정보를 정확하게 생성하는 데 기여할 것으로 기대됩니다.



### SALT: A Flexible Semi-Automatic Labeling Tool for General LiDAR Point Clouds with Cross-Scene Adaptability and 4D Consistency (https://arxiv.org/abs/2503.23980)
- **What's New**: SALT(Semi-Automatic Labeling Tool)는 라이다(LiDAR) 데이터에서 직접 작동할 수 있는 유연한 도구입니다. 최근의 접근 방식들은 카메라 캡쳐에 의존했지만, SALT는 원시 라이다 데이터를 활용하여 자동으로 프리 세그멘테이션 결과를 생성합니다. 이 도구는 새로운 zero-shot learning 패러다임을 사용하여 라이다 데이터를 의사 이미지로 변환하며, 4D 일관성을 유지하면서 높은 품질의 세그멘테이션을 보장합니다.

- **Technical Details**: SALT는 라이다 데이터를 VFM(vision foundation models)의 훈련 분포와 정렬하여 의사 이미지(pseudo-images)로 변환합니다. 이 과정에서 distance minimization을 통해 modal transformation을 최적화하며, 4D-consistent prompting 전략과 비최대 억제(non-maximum suppression) 모듈을 설계하여 일관된 세그멘테이션을 달성합니다. 이 접근법은 샘플 데이터셋에서의 미세한 조정을 통해 자동 세그멘테이션 성능을 크게 향상시킵니다.

- **Performance Highlights**: SALT는 최신 zero-shot 방법들보다 18.4% PQ(Panoptic Quality) 향상을 이루었으며, 자가 구축한 저해상도 라이다 데이터에서도 인간 주석자 성능의 40-50%에 도달했습니다. 이 도구는 다양한 라이다 타입의 조합에서 높은 주석 효율성을 제공하며, 공공 데이터셋(Table)에서의 우수한 성능을 입증했습니다. SALT의 오픈 소스화는 라이다 데이터셋의 확장을 촉진하고 향후 라이다 기반 모델 개발의 기초를 마련할 것으로 기대됩니다.



### Video-based Traffic Light Recognition by Rockchip RV1126 for Autonomous Driving (https://arxiv.org/abs/2503.23965)
Comments:
          Accepted by IEEE IV'25

- **What's New**: 이 논문에서 제안하는 ViTLR은 복수의 연속적인 프레임을 처리하여 교차로에서의 실시간 신호등 인식을 가능하게 하는 새로운 비디오 기반의 신경망 구조입니다. 기존의 한 프레임 분석 방법의 한계를 극복하고 occlusions와 악천후와 같은 복잡한 시나리오에서도 견고한 성능을 발휘합니다.

- **Technical Details**: ViTLR은 transformer-like 아키텍처를 토대로 하며, convolutional self-attention 모듈을 이용하여 Rockchip RV1126 SoC(시스템온칩)에서 최적화되어 배치됩니다. 이 신경망은 단일 입력 프레임 대신 여러 프레임을 입력받아 현재 프레임의 신호등 상태와 위치를 동시에 출력하며, 효율적인 실행을 위해 설계되었습니다. 

- **Performance Highlights**: ViTLR의 성능은 독일과 중국 본토에서 수집된 두 개의 실제 데이터 세트를 통해 광범위하게 평가되었습니다. 결과적으로, ViTLR은 기존의 단일 프레임 방법들보다 더 우수한 성능과 시간 안정성을 유지하면서, RV1126의 NPU에서 25 FPS 이상의 실시간 처리 능력을 보여주었습니다. 또한 이 시스템은 자율 주행 응용 프로그램에 사용될 수 있도록 HD 맵과 통합되어 신뢰성을 향상시킵니다.



### A Benchmark for Vision-Centric HD Mapping by V2I Systems (https://arxiv.org/abs/2503.23963)
Comments:
          Accepted by IEEE IV'25

- **What's New**: 이 논문에서는 차량-인프라 협력 자율주행(VICAD) 연구를 위해 실제 세계 데이터셋을 발표하고 있습니다. 이 데이터셋은 차량과 도로변 인프라의 협력적 카메라 프레임을 포함하며, HD 맵 요소에 대한 인간 주석이 포함되어 있습니다. 이는 도로 안전성을 확보하는데 기여할 수 있는 새로운 데이터 기반의 연구 기회를 제공합니다.

- **Technical Details**: 연구진은 V2I 시스템을 활용해 벡터화된 맵을 구축하기 위한 종단 간 신경망 프레임워크인 V2I-HD를 제안합니다. 이를 통해 V2I-HD는 방향적으로 분리된 자기 주의(attention) 메커니즘을 도입하여 계산 비용을 절감하고 자율주행차에 효과적으로 배포할 수 있습니다. 이 프레임워크는 영상을 중심으로 한 V2I 시스템을 활용하여 더욱 효율적인 맵 생성이 가능합니다.

- **Performance Highlights**: V2I-HD는 실제 세계 데이터셋을 통해 테스트된 결과로, 실시간 추론 속도에서 우수한 성능을 나타냈습니다. 또한, 다양한 복잡한 주행 장면에서도 안정적이고 견고한 맵 생성 품질을 유지하면서 낮은 비용으로 작동하는 결과를 보여줍니다. 이 연구는 향후 연구를 위한 기준으로 삼을 수 있는 소스 코드와 데이터셋을 OneDrive에 공개하였습니다.



### Local Information Matters: Inference Acceleration For Grounded Conversation Generation Models Through Adaptive Local-Aware Token Pruning (https://arxiv.org/abs/2503.23959)
Comments:
          Work in progress

- **What's New**: 이번 연구에서는 Grounded Conversation Generation (GCG)을 위한 새로운 접근법, Adaptive Local-Aware Token Pruning (ALTP)을 제안합니다. ALTP는 기존의 토큰 프루닝 방법들이 지역적 시각적 특징을 유지하는 데 실패하고, 이로 인해 GCG 작업에서 성능 저하가 발생하는 문제를 해결합니다. 특히, ALTP는 초픽셀 세분화와 동적 밀도 할당 전략을 결합하여 객체 중심 지역의 정보를 우선적으로 보존합니다.

- **Technical Details**: GCG는 특정 이미지 영역에 대한 분할 마스크와 함께 자연어 응답을 생성하는 비전-언어 작업입니다. ALTP는 Detail Density Capture (DDC)와 Dynamic Density Formation (DDF) 두 가지 주요 구성 요소로 구성되어 있습니다. DDC는 객체 중심 지역에서 토큰을 유지하여 세부 정보를 보존하고, DDF는 정보 밀도에 따라 동적으로 토큰을 할당하여 의미가 풍부한 영역에서의 보존 비율을 높입니다.

- **Performance Highlights**: ALTP는 GranDf 데이터세트에 대해 광범위한 실험을 통해 기존의 토큰 프루닝 방법들과 비교하여 GLaMM과 OMG-LLaVA 모델에서 성능을 크게 향상시켰습니다. GLaMM에 ALTP를 적용했을 때 90%의 시각적 토큰 감소와 함께 AP50에서 4.9%, Recall에서 5.0% 향상된 결과를 보여주었습니다. 이와 유사하게 OMG-LLaVA에서도 PDrop과 비교하여 90% 토큰 감소 시 AP가 2.1% 향상되었고, mIOU는 3.0% 증가했습니다.



### A Multi-Stage Auto-Context Deep Learning Framework for Tissue and Nuclei Segmentation and Classification in H&E-Stained Histological Images of Advanced Melanoma (https://arxiv.org/abs/2503.23958)
Comments:
          15 pages

- **What's New**: 이번 논문에서는 피부암의 가장 치명적인 형태인 멜라노마의 진단 및 치료를 위한 새로운 멀티 스테이지 딥 러닝 접근 방식을 제시하였습니다. 기존의 조직(tissue) 및 핵(nuclei) 분석 방법과는 달리, 우리가 제안한 방법은 두 가지 정보를 통합하여 하나의 통합된 프레임워크에서 분할(segmentation)과 분류(classification)를 수행합니다. 이를 통해 PUMA 챌린지에서 평균 마이크로 Dice 조직 점수 73.40%와 합산된 핵 F1 점수 63.48%를 달성하며 뛰어난 성과를 보였습니다.

- **Technical Details**: 제안된 방법은 auto-context 개념에 기반한 멀티 스테이지 파이프라인으로, 각 단계의 출력 결과를 서로의 입력으로 활용하여 조직 및 핵 분할과 분류를 개선합니다. 파이프라인은 네 단계로 구성되며, 첫 번째 단계에서 입력 이미지의 유형을 분류하고, 두 번째 단계에서 초기 조직 분할을 수행합니다. 세 번째 단계에서 HoVer-Next 모델을 사용해 핵 인스턴스 분할을 진행하며, 마지막 네 번째 단계에서는 세 번째 단계에서 생성된 핵 마스크를 사용하여 조직 분할을 정교화합니다.

- **Performance Highlights**: 우리의 연구는 PUMA 챌린지에서 Track 1에서 2위, Track 2에서 1위를 기록하였습니다. 종합적으로 310개의 멜라노마 이미지 패치를 사용하며, 206개 이미지는 훈련용, 10개 이미지는 검증용, 94개 이미지는 테스트용으로 나누어 평가하였습니다. 이 성과는 핵 및 조직의 동시 분할 및 분류에 대한 연구의 필요성을 강조하며 후속 연구에 대한 기초를 제공합니다.



### AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inferenc (https://arxiv.org/abs/2503.23956)
- **What's New**: 이 논문은 최근의 대규모 시각 언어 모델(LVLMs)의 발전과 이들이 제기하는 계산적 문제를 해결하기 위한 새로운 방법인 AirCache를 제안합니다. AirCache는 KV(cache) 압축 기법으로, 시각 토큰 간의 중복성을 줄여 모델의 성능을 유지하면서도 추론 속도를 향상시키는 데 중점을 두고 있습니다. 특히, 중요도가 높은 시각 구성 요소를 선택하기 위한 엘리트 관찰 창(elite observation window)을 도입하여 다양한 이점을 제공합니다.

- **Technical Details**: AirCache의 주요 구성 요소는 시각 토큰의 중요도 점수화와 레이어별 KV cache 예산 할당입니다. 이 기법은 스스로 어텐션 점수를 활용하여 시각 토큰의 중요도를 평가하고, 이를 기반으로 압축 예산을 레이어별로 차별화하여 최적의 성능을 도출합니다. 논문에서는 이 기술을 통해 단순히 토큰 수를 줄이는 것이 아니라, 중요한 정보가 포함된 10%의 시각 KV cache만 유지하여도 모델 성능에 미미한 영향을 미친다고 보고합니다.

- **Performance Highlights**: 종합적인 실험 결과, AirCache는 여러 LVLM 및 벤치마크 데이터 세트에서 기존 방법들에 비해 현저히 개선된 성능을 보여줍니다. 이 방법을 통해 KV cache를 10%만 유지하면서도 디코딩 대기 시간을 29%에서 66%까지 줄일 수 있었으며, 캐시 유지 비율이 감소할수록 기존 방법들보다 성능적으로 우위에 있음을 입증했습니다.



### JointTuner: Appearance-Motion Adaptive Joint Training for Customized Video Generation (https://arxiv.org/abs/2503.23951)
Comments:
          Project Page: this https URL

- **What's New**: 최근 텍스트-비디오 생성 기술은 텍스트 프롬프트를 기반으로 일관성 있는 비디오 합성 능력이 크게 발전하였습니다. 이 연구에서는 JointTuner라는 새로운 적응형 공동 훈련 프레임워크를 제안하여, 기존 방법들의 문제점인 개념 간섭(concept interference)과 외관 오염(appearance contamination)을 해결하고자 합니다. Adaptive LoRA와 Appearance-independent Temporal Loss라는 두 가지 주요 혁신을 통해 비디오의 외관과 동작을 동시에 최적화할 수 있는 방법을 고안했습니다.

- **Technical Details**: JointTuner는 Adaptive LoRA라는 맥락 인식 게이팅 메커니즘을 포함하여 모달 입력(이미지 또는 비디오)에 동적으로 적응합니다. 또한, Appearance-independent Temporal Loss는 참조 비디오에서 고유 외관을 분리하며, 노이즈 예측 작업을 통해 동작 패턴을 외관과 분리합니다. 이러한 혁신적인 접근 방식은 비디오의 프레임 특성과 시간 일관성을 유지하면서도 공간적 속성을 방해할 수 있도록 설계되었습니다.

- **Performance Highlights**: JointTuner는 총 90개의 맞춤형 외관-동작 조합을 포함하는 벤치마크와 10개의 자동 측정 지표를 기반으로 종합적인 평가 체계를 구성하였습니다. 광범위한 실험을 통해, 제안된 방법이 현재의 첨단 접근 방식들과 비교하여 우수한 성능을 보임을 입증했습니다. 이 연구는 사용자가 원하는 외관과 동작 모두를 동시에 조정할 수 있는 새로운 가능성을 열어주는 기초를 제공하고 있습니다.



### Spectral-Adaptive Modulation Networks for Visual Perception (https://arxiv.org/abs/2503.23947)
- **What's New**: 이 논문은 2D convolution과 self-attention의 스펙트럼 특성을 그래프 스펙트럴 분석을 통해 비교하고, 이를 기반으로 스펙트럴 적응형 조절(Spectral-Adaptive Modulation, SPAM) 믹서를 제안합니다. SPAM은 멀티 스케일 컨볼루션 커널과 스펙트럴 재조정 메커니즘을 사용하여 시각 특성을 처리, 향상시키며, 새로운 시각 백본(SPANetV2)을 개발합니다. 이러한 접근이 과거의 경험적 결과를 뒷받침하며, CNN과 Transformer의 통합을 통해 새로운 가능성을 제시합니다.

- **Technical Details**: 본 연구는 그래프 신호 처리(graph signal processing)를 통해 2D convolution과 self-attention의 이론적 분석을 수행합니다. 이 분석을 통해, 작은 커널 디자인이 고주파 필터링에 적합하고, 큰 커널 디자인이 self-attention과 유사한 저주파 필터링을 선호한다는 것을 증명합니다. SPAM은 이러한 이론을 바탕으로 하여, 여러 커널 크기를 활용해 이미지 패치를 스펙트럴 적응형 방식으로 인코딩하며, 주파수 영역 마스크 필터링을 적용하여 스펙트럴 특성을 조절합니다.

- **Performance Highlights**: SPANetV2는 ImageNet-1K 분류, COCO 객체 탐지, ADE20K 의미 분할 등 여러 비전 작업에서 최첨단 모델을 능가하는 성능을 입증했습니다. 본 논문에서 제안하는 SPANetV2는 CNN과 Transformer의 장점을 융합하여 다양한 비전 작업에서 경쟁력을 갖춘 아키텍처로 자리매김하고 있습니다. 이러한 실험 결과는 SPANetV2가 효과적으로 스펙트럼 성분을 정제하여 비전 모델의 성능을 크게 향상함을 보여줍니다.



### Exploring Reliable PPG Authentication on Smartwatches in Daily Scenarios (https://arxiv.org/abs/2503.23930)
- **What's New**: 이번 연구에서는 MTL-RAPID라는 새로운 PPG 인증 모델을 제안합니다. 이 모델은 신뢰할 수 있는 인증을 위해 다중 작업(MLT) 공동 훈련 전략을 사용하여 신호 품질을 평가하고 사용자 신원을 동시에 확인합니다. MTL-RAPID는 따로 훈련된 모델보다 적은 매개변수로도 더 높은 성능을 발휘합니다.

- **Technical Details**: MTL-RAPID는 80,000개의 매개변수를 사용하여 PPG 신호에서 글로벌 및 로컬 혈관 특성, 심장 박동 패턴, 사람의 움직임 형상을 추출하고 분석합니다. 이 모델은 품질 평가 작업의 통과 비율을 조정함으로써 다양한 시나리오에 맞춘 인증 강도를 조절할 수 있습니다. 결과적으로, MTL-RAPID는 일상적인 사용에 적합한 경량화된 모델입니다.

- **Performance Highlights**: MTL-RAPID 모델은 30명의 참가자를 대상으로 한 운동 아티팩트, 32명의 참가자를 대상으로 한 시간 변동성, 16명의 사용자 선호도를 평가한 연구에서 99.2%의 AUC와 3.5%의 EER을 달성하여 기존 방법을 능가하였습니다. 이러한 결과는 사용자들이 PIN 방식보다 MTL-RAPID 모델을 더 선호한다는 점에서도 드러났습니다.



### CoMatch: Dynamic Covisibility-Aware Transformer for Bilateral Subpixel-Level Semi-Dense Image Matching (https://arxiv.org/abs/2503.23925)
- **What's New**: 이번 연구는 CoMatch라는 새로운 세미-밀집 이미지 매칭 기법을 제안합니다. 이 기법은 동적 공시비( co-visible) 인식과 양면 서브픽셀 정확도를 갖춘 특징 변환 방식을 포함합니다. 최적화된 토큰 운영 방식을 통해 계산 효율성을 보장하면서도, 매칭 과정에서의 정확도를 크게 향상시킵니다.

- **Technical Details**: CoMatch는 공시비 기반 토큰 압축기(covisibility-guided token condenser)를 사용하여 각 토큰의 공시비 점수를 동적으로 예측하며, 이 정보를 바탕으로 토큰을 적응적으로 선택합니다. 더불어 비공시비 지역으로부터의 불필요한 정보의 방해를 최소화하기 위해 공시비 보조 주의 메커니즘(covisibility-assisted attention mechanism)을 도입했습니다. 마지막으로, 소스 및 타겟 뷰의 매칭 후보를 리파인하기 위한 새로운 모듈이 제안되어 양측에서 서브픽셀 레벨로 조정됩니다.

- **Performance Highlights**: 광범위한 실험을 통해 CoMatch는 여러 시각적 작업에서 최첨단 성능을 달성하며, 이전 연구와 비교해도 높은 효율성을 보입니다. 이로 인해 CoMatch는 실제 응용 프로그램에서의 사용 가능성이 높아졌으며, 기존 방법들의 한계를 극복하는 방식으로 주목받고 있습니다.



### FineCausal: A Causal-Based Framework for Interpretable Fine-Grained Action Quality Assessmen (https://arxiv.org/abs/2503.23911)
- **What's New**: 이번 연구에서는 FineCausal이라는 새로운 AQA(행동 품질 평가) 프레임워크를 제안합니다. FineCausal은 기존의 블랙박스 방식의 한계를 극복하고, 인과 기반(causal-based) 접근 방식을 통해 투명성과 해석 가능성을 높입니다. 이 프레임워크는 Graph Attention Network(GAT) 기반의 인과 개입 모듈을 활용하여 인간 중심의 전경 특징과 배경의 혼란 요소를 구분하고, 시간적 인과 주의 모듈을 통해 행동 단계 간의 미세한 시간적 의존성을 포착합니다.

- **Technical Details**: FineCausal 프레임워크는 네 가지 중심 변수를 포함하는 인과 그래프를 구축하여 원본 비디오 특징, 융합 비디오 특징, 단계 특징 및 최종 행동 점수 간의 관계를 명확히 모델링합니다. GAT 기반의 인과 개입 모듈을 통해 전경 정보와 가치 있는 배경 정보를 효과적으로 통합하고, 각 하위 행동이 전체 품질 점수에 기여하는 방식에 대한 세밀한 통찰을 제공합니다. 또한, FineCausal은 비디오 및 마스크 특징의 간단한 융합 기법을 피해 진정한 인과 관계를 명확하게 나타냅니다.

- **Performance Highlights**: FineCausal은 FineDiving-HM 데이터셋에서 최첨단 성능을 발휘하며, 높은 예측 정확도를 기록합니다. 이 시스템은 투명하고 해석 가능한 피드백을 제공함으로써 코치와 운동선수가 특정 개선 영역을 확인하는 데 도움이 됩니다. 그러나 FineCausal의 활용에는 전문가의 광범위한 지식과 고품질 주석이 필요하며, 이는 향후 연구 방향으로 제시됩니다.



### HumanAesExpert: Advancing a Multi-Modality Foundation Model for Human Image Aesthetic Assessmen (https://arxiv.org/abs/2503.23907)
- **What's New**: 이번 연구에서는 Human Image Aesthetic Assessment (HIAA)를 위한 혁신적인 구현 프레임워크를 제시하며, HIAA를 위해 처음으로 설계된 HumanBeauty 데이터셋을 도입했습니다. 이 데이터셋은 108,000개의 고퀄리티 인간 이미지를 포함하고 있으며 수작업으로 주석이 붙여져 있습니다. HIAA의 개선된 평가를 위해 Vision Language Model (VLM) 기반의 HumanAesExpert 모델을 제안하였습니다.

- **Technical Details**: HumanAesExpert는 12차원의 미적 기준을 활용하여 HIAA의 세부 측면을 평가하는 독창적인 Expert Head를 도입합니다. 이 모델은 Language Modeling (LM) Head와 Regression Head를 함께 사용하여 정확하고 세밀한 평가를 가능하게 합니다. 평가의 정밀도를 높이기 위해 세 가지 헤드의 점수를 집계하는 MetaVoter를 설계하였습니다.

- **Performance Highlights**: HumanAesExpert 모델은 HIAA 과제에서 기존 최첨단 모델들과 비교할 때 월등히 우수한 성능을 보여주었습니다. 연구진은 방대한 실험을 통해 HIAA 전반에 걸쳐 SOTA 성능을 달성함을 입증하였으며, 차세대 HIAA 커뮤니티의 발전을 위해 데이터셋, 모델 및 코드 공개를 계획하고 있습니다.



### Boosting MLLM Reasoning with Text-Debiased Hint-GRPO (https://arxiv.org/abs/2503.23905)
- **What's New**: MLLM(다중모달 LLM) 추론은 OpenAI의 o1 모델 출시 이후 뛰어난 문제 해결 능력으로 인해 많은 연구의 관심을 받고 있습니다. 기존 추론 방법은 PRM(프로세스 보상 방법)과 ORM(결과 보상 방법)으로 분류되며, DeepSeek-R1이 ORM 방법이 기존 PRM보다 우수하다는 것을 입증했습니다. 그러나 기존 MLLM의 GRPO(그룹 상대 정책 최적화) 알고리즘은 수학적 추론과 같은 복잡한 다중모달 문제를 해결하는 데 여전히 어려움을 겪고 있습니다.

- **Technical Details**: 이 연구에서는 GRPO의 성능을 저하하는 두 가지 문제를 파악했습니다: 낮은 데이터 활용과 텍스트 편향입니다. 낮은 데이터 활용 문제는 GRPO가 어려운 샘플에 대해 긍정적인 보상을 얻지 못하는 상황을 나타내며, 텍스트 편향은 MLLM이 이미지 조건을 무시하고 텍스트 조건에만 의존하여 생기는 현상입니다. 이 문제를 해결하기 위해 제안된 Hint-GRPO는 다양한 난이도의 샘플에 대한 힌트를 능동적으로 제공하며, 텍스트 편향 보정 방법은 이미지 조건을 기반으로 토큰 예측 로짓을 조정하여 성능을 향상시킵니다.

- **Performance Highlights**: 세 가지 기본 MLLM을 대상으로 한 11개 데이터셋에서 실험한 결과, 제안한 방법들이 원래 MLLM의 추론 능력을 크게 향상시켰으며, 기존 MLLM 추론 방법들보다 우수한 성능을 보였습니다. 힌트-GRPO는 높이 어려운 질문에 대한 추가 힌트를 제공하여 정확한 정답 생성을 유도합니다. 실험 결과는 제안한 방법들이 기존 방법들에 비해 상당한 성과를 달성했다는 것을 잘 보여줍니다.



### Training-Free Text-Guided Image Editing with Visual Autoregressive Mod (https://arxiv.org/abs/2503.23897)
- **What's New**: 이번 논문에서는 VAR(Visual AutoRegressive) 기반의 새로운 텍스트 유도 이미지 편집 프레임워크를 제안하고 있습니다. 이 프레임워크는 명시적 역전(inversion) 과정 없이도 정밀하고 통제된 수정이 가능한 방법론을 제공합니다. 특히, 원본 이미지의 토큰 인덱스와 확률 분포를 저장하는 캐싱 메커니즘을 도입하여 텍스트 프롬프트와 이미지 간의 관계를 포착합니다.

- **Technical Details**: VAR 기반의 모델은 visual tokenizer와 transformer를 포함하여 이미지를 인코딩하고 합성하는 과정을 수행합니다. 이 과정에서 원본 이미지를 연속적(feature map)으로 인코딩한 후, 다중 척도의 이산적인 잔여 맵(residual maps)으로 양자화합니다. 또한, 적응형 미세 마스킹 전략을 통해 적절한 지역의 수정을 식별하여 원하지 않는 변경을 방지합니다.

- **Performance Highlights**: 제안된 AREdit 프레임워크는 훈련 없이 고충실도의 편집을 가능하게 하며, 기존의 확산(diffusion) 및 정정 흐름(rectified flow) 기반의 접근법과 비교하여 유사하거나 그 이상의 성능을 보여줍니다. 실제 테스트 결과는 1K 해상도의 이미지를 단 1.2초 만에 처리할 수 있는 속도를 자랑하며, 다양한 정량적 지표와 시각적 품질 면에서도 우수한 결과를 보였습니다.



### MuseFace: Text-driven Face Editing via Diffusion-based Mask Generation Approach (https://arxiv.org/abs/2503.23888)
Comments:
          6 pages, 5 figures,IEEE International Conference on Multimedia & Expo 2025

- **What's New**: 이번 논문에서 제안된 MuseFace는 텍스트 프롬프트만을 이용하여 얼굴 편집을 가능하게 하는 텍스트 기반 얼굴 편집 프레임워크입니다. MuseFace는 텍스트에서 세분화된 마스크를 직접 생성하고 이를 기반으로 얼굴 이미지를 수정하는 두 가지 모델, 즉 Text-to-Mask diffusion 모델과 의미 인식(face editing) 모델을 통합하여 사용합니다. 이 프레임워크는 다양성(diversity)과 유연성(flexibility)뿐만 아니라 제어 가능성(controllability)을 동시에 충족하는 것을 목표로 하고 있습니다.

- **Technical Details**: MuseFace는 처음으로 텍스트 입력에 의해 다양한 위치-aware 세분화 마스크를 생성하는 방식을 도입합니다. 이를 통해 기존의 거칠고 제한적인 마스크 대신 세밀한 마스크를 생성함으로써 사용자가 편집 지점을 명확하게 제어할 수 있습니다. 이 시스템은 또한 두 가지 방식으로 활용될 수 있어, 사용자 제공 마스크와 함께 또는 마스크가 없는 모드에서 자율적으로 다양한 편집 제안을 생성할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: 광범위한 실험을 통해 MuseFace는 고충실도(high-fidelity) 성능을 발휘하며, 기존의 얼굴 편집 모델보다 월등한 성능을 입증하였습니다. 본 연구는 특화된 학습 데이터의 부족 문제를 해결하기 위해 점진적으로 진화하는 접근 방식을 제시하며, 기존의 방법과 비교했을 때 더욱 창의적인 생성 및 다양성을 확보할 수 있는 가능성을 보여줍니다.



### GLane3D : Detecting Lanes with Graph of 3D Keypoints (https://arxiv.org/abs/2503.23882)
Comments:
          Accepted to CVPR 2025

- **What's New**: GLane3D는 3D 차선 탐지를 위한 새로운 접근 방식을 제안합니다. 이는 주요 앵커를 통해 차선 연결을 예측하여 완전한 3D 차선을 구성하는 방식을 채택하고 있습니다. 이 방식은 차선 연속성을 유지하는 데 필수적인 키포인트를 식별하고, 각 키포인트에 대해 여러 제안들을 활용하여 탐지 회수(recall)를 향상시킵니다.

- **Technical Details**: 이 연구에서는 PointNMS를 사용하여 겹치는 제안 키포인트를 줄이고, 추정된 BEV(graph)가 대체로 간결해지고 계산 부담을 최소화합니다. GLane3D는 Inverse Perspective Mapping(IPM)을 기반으로 사용자 정의된 BEV 위치에서 동일하게 분포된 BEV 위치 대신 샘플링 포인트를 계산하여 차량 주변 지역의 밀도를 감소시키고 먼 지역의 포화 문제를 완화합니다. 이러한 설계를 통해 GLane3D는 뛰어난 일반화 능력을 보여줍니다.

- **Performance Highlights**: GLane3D는 OpenLane과 Apollo 데이터셋 모두에서 최첨단 방법들을 초월하는 성과를 기록했습니다. F1 점수에서 우수한 성능을 보였으며, OpenLane에서 훈련된 모델이 Apollo 데이터셋에서 강력한 일반화 성능을 나타내는 것이 확인되었습니다. 또한 GLane3D는 Camera+Lidar 융합을 통해 가장 높은 FPS를 달성하여 실용적인 응용에 매우 적합합니다.



### ExScene: Free-View 3D Scene Reconstruction with Gaussian Splatting from a Single Imag (https://arxiv.org/abs/2503.23881)
Comments:
          ICME 2025

- **What's New**: 이 논문은 단일 뷰 이미지를 사용하여 몰입감 있는 3D 장면을 재구성하기 위한 새로운 두 단계 프로세스인 ExScene을 제안합니다. 이는 특히 기존 방식들이 가지는 시각 요소의 제한을 극복하기 위한 혁신적인 접근법입니다. 새로운 멀티모달 확산 모델(multi-modal diffusion model)을 설계하여 높은 정확도의 파노라마 이미지를 생성하고, 이를 통해 3D Gaussian Splatting(3DGS) 모델을 초기화하는 방법을 도입합니다.

- **Technical Details**: ExScene은 첫 번째 단계에서 멀티모달 확산 모델을 사용해 파노라마 이미지를 생성한 후, 파노라마 깊이 추정 기법을 통해 기하학적 정보를 계산합니다. 이 기하학적 정보를 고품질의 파노라마 이미지와 결합하여 초기 3D Gaussian Splatting 모델을 훈련합니다. 마지막으로 2D 안정 비디오 확산 기반의 정제 기법을 도입하여 다시 모델의 품질을 개선하고, 이미지 시퀀스 전반에 걸쳐 색상 및 공간의 일관성을 확보합니다.

- **Performance Highlights**: ExScene은 다양한 단일 시점 데이터 세트에서 실험을 통해 기존의 최첨단 방법들을 질적으로 및 양적으로 초월하는 성능을 보였습니다. 우리의 멀티모달 파노라마 생성 모듈과 시각적 정제 기법의 효과는 탈락 연구를 통해 확인되었습니다. 이 연구는 단일 뷰 이미지를 기반으로 하는 고품질 3D 장면 재구성을 위한 최첨단 접근법을 제공하며, 혁신적인 기술적 기여를 포함합니다.



### Learned Image Compression and Restoration for Digital Pathology (https://arxiv.org/abs/2503.23862)
- **What's New**: 이 논문에서는 디지털 병리 이미지의 효율적 압축을 위한 새로운 딥러닝 기반 프레임워크인 CLERIC을 제안합니다. CLERIC은 전체 슬라이드 이미지(Whole Slide Images, WSIs)의 고해상도를 고려하여 설계되었으며, 병리학적 세부 사항을 유지하면서도 압축 효율성을 높이는 학습 가능한 리프팅 방식을 통합합니다. 이 프레임워크는 이미지 복원 과정에서 고품질의 조직 구조를 보장하기 위해 역 리프팅 변환을 적용합니다.

- **Technical Details**: CLERIC은 이미지 분석 단계에서 고주파 및 저주파 구성 요소로 분해하기 위해 리프팅 스킴 변환을 사용합니다. 이 과정에서 변형 잔여 블록(Deformable Residual Blocks, DRB)과 순환 잔여 블록(Recurrent Residual Blocks, R2B)을 포함한 병렬 인코더가 특징 추출 및 공간 적응성을 개선하여 복잡한 병리 이미지를 효과적으로 처리합니다. 이러한 기법은 이미지 세분화 및 초해상도 과제에서도 성공적으로 적용되었습니다.

- **Performance Highlights**: 실험 결과, CLERIC은 기존의 최첨단 학습 이미지 압축(Learned Image Compression, LIC) 모델에 비해 우수한 비율-왜곡(rate-distortion) 성능을 보여주었으며, 저장 요구량을 현저히 줄이면서도 진단 이미지의 품질을 유지함을 입증하였습니다. 또한, CLERIC은 다중 해상도 형식을 지원하여 표준 병리학적 시각화 소프트웨어와의 원활한 통합이 가능합니다.



### FlexiMo: A Flexible Remote Sensing Foundation Mod (https://arxiv.org/abs/2503.23844)
- **What's New**: 이번 논문에서는 다중 출처의 위성 이미지를 활용하는 원격 탐사 기초 모델인 FlexiMo를 제안합니다. FlexiMo는 고정된 공간 해상도와 패치 크기에서 벗어나 유연하게 적응할 수 있는 모델로, 입력 이미지의 해상도에 따라 패치 임베딩을 동적으로 재조정합니다. 이를 통해 필수 토큰 특성을 보존하면서도 멀티 스케일 기능을 충실히 유지할 수 있게 합니다.

- **Technical Details**: FlexiMo의 핵심은 공간 해상도 인식 모듈로, parameter-free alignment embedding 메커니즘을 사용하여 입력 이미지의 해상도와 차원을 고려하여 패치 임베딩을 조정합니다. 또한, 경량 채널 적응 모듈을 통해 센서의 이전 분광 정보를 활용하여 다양한 수의 채널을 갖는 이미지도 처리할 수 있습니다. 이러한 설계는 모델의 구조를 변경하지 않고도 효율적인 기능 추출을 가능하게 합니다.

- **Performance Highlights**: 다양한 다중모달 및 다중 해상도 데이터셋에서 FlexiMo가 뛰어난 성능을 보였음을 실험을 통해 검증하였습니다. 특히 장면 분류, 토지 피복 분류, 도시 건물 세분화 및 구름 검출과 같은 다운스트림 작업에서 우수한 성과를 기록했습니다. FlexiMo는 매개변수 효율적이고 물리적으로 일관된 적응을 가능하게 하여 실세계 원격 탐사 애플리케이션에 보다 적합한 모델로 자리잡을 수 있는 기반을 마련합니다.



### Bridge the Gap Between Visual and Linguistic Comprehension for Generalized Zero-shot Semantic Segmentation (https://arxiv.org/abs/2503.23806)
- **What's New**: 이번 연구에서는 일반화된 제로샷 의미 분할(GZS3)을 위해 DeVLMatch 프레임워크를 제안합니다. 이 프레임워크는 시각적 정보와 언어적 정보를 활용하여 클래스를 부분과 상태로 분리하여 새로운 방식으로 지식을 전이할 수 있도록 합니다. 이를 통해 사람의 인지 능력을 모방하여, 본 연구는 제로샷 학습의 성능을 크게 향상시킵니다.

- **Technical Details**: DeVLMatch는 두 가지 주요 모듈, 즉 공간 부분 매칭(SPMatch)과 채널 상태 매칭(CSMatch)으로 구성됩니다. SPMatch 모듈에서는 시각적 피쳐와 언어적 클래스 특성을 결합하여 그래프 매칭 작업을 수행합니다. CSMatch 모듈에서는 채널 정보를 시각적 공간에서 상태 정보와 연결하여 지식 전이를 촉진합니다.

- **Performance Highlights**: 제안된 DeVLMatch 프레임워크는 PASCAL VOC, COCO-Stuff, CATARACTS와 같은 표준 벤치마크에서 기존 GZS3 방법들을 초월하는 성능을 보여주었습니다. 특히, 클래스 설명을 분리하여 사용하는 것이 단일 의미 표현보다 월등한 성능을 발휘함을 입증했습니다.



### On-device Sora: Enabling Training-Free Diffusion-based Text-to-Video Generation for Mobile Devices (https://arxiv.org/abs/2503.23796)
- **What's New**: On-device Sora는 모바일 기기에서 텍스트로부터 비디오 생성하는 혁신적인 기술입니다. 이전의 접근 방식과는 달리 이 모델은 학습 과정을 필요로 하지 않으며, 스마트폰에서도 고품질 비디오를 직접 생성할 수 있도록 설계되었습니다. 제안된 기술은 Linear Proportional Leap, Temporal Dimension Token Merging, Concurrent Inference with Dynamic Loading으로, 제한된 리소스에서도 높은 효율성을 보여줍니다.

- **Technical Details**: On-device Sora는 세 가지 주요 기술을 통해 모바일 기기에서의 비디오 생성을 지원합니다. 첫째, Linear Proportional Leap (LPL)은 denoising 단계 수를 줄여주는 기능을 제공하여 생성 과정의 효율을 높입니다. 둘째, Temporal Dimension Token Merging (TDTM)은 주의 레이어에서 consecutive tokens를 결합하여 계산 복잡성을 줄입니다. 셋째, Concurrent Inference with Dynamic Loading (CI-DL)은 메모리 용량이 제한된 모바일 디바이스에서 동시 모델 추론을 가능하게 합니다.

- **Performance Highlights**: 실험 결과 On-device Sora는 iPhone 15 Pro에서 NVIDIA A6000 GPU와 유사한 비디오 품질을 유지하며 생성 속도를 효과적으로 개선했습니다. iPhone 15 Pro의 GPU 성능은 NVIDIA A6000에 비해 143배 낮음에도 불구하고, 제안된 방법들을 통해 비디오 생성의 효율성이 크게 향상되었습니다. 이 결과는 리소스가 제한된 모바일 디바이스에서 고품질 비디오 생성을 가능하게 함을 보여줍니다.



### Pan-LUT: Efficient Pan-sharpening via Learnable Look-Up Tables (https://arxiv.org/abs/2503.23793)
Comments:
          12 pages, 6 figures

- **What's New**: 최근 심층 학습 기반의 팬 샤프닝(pan-sharpening) 알고리즘이 전통적인 방법에 비해 주목할 만한 발전을 이루었습니다. 하지만 많은 심층 학습 기법들이 고해상도 이미지 처리시 상당한 연산 오버헤드를 초래하여 실제 환경에서의 적용을 제한하고 있습니다. 이를 해결하기 위해, 성능과 계산 효율성의 균형을 이루는 새로운 학습 가능한 look-up table(LUT) 프레임워크인 Pan-LUT를 제안합니다.

- **Technical Details**: 제안하는 Pan-LUT는 300K 미만의 파라미터로 구성되어 있으며, NVIDIA GeForce RTX 2080 Ti GPU를 이용해 1ms 이내에 8K 해상도 이미지를 처리할 수 있습니다. 이를 통해 세부적인 스펙트럼 변환을 정밀하게 제어하는 PAN-guided look-up table(PGLUT)과 세밀한 공간 세부사항을 효과적으로 포착하고 지역context를 학습하기 위한 spatial details look-up table(SDLUT) 및 adaptive aggregation look-up table(AALUT)를 도입했습니다. 다수의 위성 데이터셋을 통한 실험 결과, Pan-LUT는 기존의 전통적 방법보다 7 dB 높은 성능을 보였습니다.

- **Performance Highlights**: Pan-LUT는 경량화된 방식으로 대규모 원격 탐지 이미지를 효율적으로 처리하며, 실제 애플리케이션에 적합한 솔루션을 제공합니다. 또한 기존 방법들에 비해 뛰어난 속도와 효율성을 자랑하며, 실제 환경에서 전체 해상도 장면에서 SOTA(state-of-the-art) 방법들을 초월하는 성능을 보여주었습니다. 이로써 Pan-LUT는 고해상도 이미지에서 신뢰할 수 있는 팬 샤프닝 방법으로 자리잡을 것으로 기대됩니다.



### MGD-SAM2: Multi-view Guided Detail-enhanced Segment Anything Model 2 for High-Resolution Class-agnostic Segmentation (https://arxiv.org/abs/2503.23786)
- **What's New**: 이번 연구에서는 새로운 MGD-SAM2 모델을 제안합니다. MGD-SAM2는 고해상도 클래스 비의존 분할(HRCS) 작업을 위해 설계되었으며, Multi-view images와 SAM2의 일반적인 시각 정보의 상호작용을 결합하여 보다 정밀한 물체 분할을 달성합니다. 본 모델은 Multi-view Perception Adapter(MPAdapter), Multi-view Complementary Enhancement Module(MCEM), Hierarchical Multi-view Interaction Module(HMIM), Detail Refinement Module(DRM) 등 네 가지 새로운 모듈을 포함하고 있습니다.

- **Technical Details**: MGD-SAM2는 이미지의 전역 이미지와 지역 패치 간의 다중 뷰 기능 상호작용을 통해 세밀한 세분화를 달성합니다. MPAdapter는 SAM2의 인코더를 조정하여 HRCS 이미지에서 지역 세부 사항과 전역 의미를 강화합니다. MCEM와 HMIM은 다중 뷰 기능을 집계하여 지역 질감과 전역 문맥을 추가로 활용하며, DRM은 데이터 세트에서 세부 사항 손실을 보완하기 위해 서서히 복원된 고해상도 마스크 예측을 생성합니다.

- **Performance Highlights**: 실험 결과, MGD-SAM2는 여러 고해상도 및 일반 해상도 데이터 세트에서 뛰어난 성능을 나타냈습니다. 이전의 SOTA(State-of-the-art) 방법들을 초월하며 DIS5K, HRSOD, UHRSD, DAVIS-S와 같은 고해상도 데이터 세트와 DUTS, HKU-IS와 같은 일반 해상도 데이터 세트에서도 새로운 기록을 세웠습니다. 이는 MGD-SAM2의 효과성과 견고함을 입증하는 결과입니다.



### Evaluation of (Un-)Supervised Machine Learning Methods for GNSS Interference Classification with Real-World Data Discrepancies (https://arxiv.org/abs/2503.23775)
Comments:
          34 pages, 25 figures

- **What's New**: 이 논문은 자율주행차, 통행료 시스템 및 디지털 타코그래프와 같은 응용 프로그램에서 차량 위치 확인(vhicle localization)의 중요성을 강조합니다. 글로벌 내비게이션 위성 시스템(GNSS) 수신기를 사용하여 정확한 위치 결정을 시도하지만, 간섭 신호(interference signals)로 인해 이 과정이 방해받을 수 있음을 다룹니다. 최근의 머신러닝(ML) 기반 접근 방식이 이러한 간섭 모니터링에서 뛰어난 성능을 보였지만, 실제 환경에서의 적용 가능성은 아직 평가되지 않았습니다.

- **Technical Details**: 해당 연구에서는 독일의 두 고속도로 위치와 오스트리아의 지탈 알프스에서 수행된 대규모 측정 캠페인을 설명합니다. ML 기술의 효과적인 구현을 위해서는 현실적인 간섭 신호(noise)와 관련된 훈련 데이터셋이 필요하며, 이 데이터셋은 법적 제한으로 인해 생성하기 어려워하는 문제를 다룹니다. 또, 최신의 감독형 ML 방법을 평가하고 비감독 학습(unsupervised learning)을 위한 의사 레이블링(pseudo-labeling)의 적용 가능성을 제시합니다.

- **Performance Highlights**: 데이터 불일치로 인해 데이터셋 결합의 어려움이 크며, 이상 탐지(outlier detection), 도메인 적응(domain adaptation), 데이터 증강(data augmentation) 기법 등을 평가하여 모델의 적응 능력을 보여줍니다. 이 연구는 ML 기반 방법들이 실제 응용에서 어떻게 성능을 발휘하는지를 진단하고, 데이터 간 변화를 수용하는 모델링의 가능성을 제시합니다.



### XLRS-Bench: Could Your Multimodal LLMs Understand Extremely Large Ultra-High-Resolution Remote Sensing Imagery? (https://arxiv.org/abs/2503.23771)
Comments:
          It has been accepted by CVPR2025

- **What's New**: 최근의 연구에서 다중모드 대형 언어 모델(MLLMs)의 발전은 이를 평가하기 위한 새로운 벤치마크인 XLRS-Bench의 필요성을 전제로 하고 있습니다. 이 벤치마크는 초고해상도 원격 감지(ultra-high-resolution remote sensing) 이미지를 대상으로 MLLMs의 지각(perception) 및 추론(reasoning) 능력을 평가하는 데 중점을 둡니다. XLRS-Bench는 평균 이미지 크기가 8500×8500으로, 기존 벤치마크에 비해 월등히 큰 데이터셋으로 구성되어 있습니다.

- **Technical Details**: XLRS-Bench는 16개의 하위 작업을 정의하여 MLLM의 10가지 지각 능력과 6가지 추론 능력을 평가합니다. 이 벤치마크는 1,400개의 실제 초고해상도 이미지를 수집하였으며, 각각은 전문가에 의해 손수 주석(annotation) 처리되고 검증되었습니다. 아울러 GPT-4o를 활용한 반자동 주석화 파이프라인을 통해 데이터의 품질을 높였습니다.

- **Performance Highlights**: XLRS-Bench의 주요 장점은 초고해상도 이미지의 사용과 품질 높은 주석으로, 10,000 × 10,000 픽셀 크기의 이미지를 포함하는 등 기존 벤치마크에 비해 상당히 개선된 평가 기준을 제공합니다. 이 벤치마크를 통해 여러 MLLM의 성능을 평가한 결과, 실제 원격 감지 응용에 더 많은 노력이 필요하다는 것을 보여주었습니다.



### STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding? (https://arxiv.org/abs/2503.23765)
- **What's New**: 이번 연구는 MLLMs(Multimodal Large Language Models)가 Embodied AI 및 Autonomous Driving의 솔루션으로서의 잠재력을 평가하기 위해 Spatial-Temporal Intelligence Benchmark(STI-Bench)를 제안합니다. 이 벤치마크는 MLLMs의 공간-시간 이해 능력을 다양한 비디오 및 이미지 입력을 통해 평가합니다. 300개 비디오와 2,000개 이상의 Q&A 쌍을 포함하여, 실제 환경에서 MLLMs의 작동을 분석합니다.

- **Technical Details**: STI-Bench는 Desktop, Indoor, Outdoor 세 가지 주요 시나리오를 고려하여 설계되었으며, MLLMs의 공간-시간 이해도를 평가하기 위한 8개의 개별 작업을 포함하고 있습니다. 이 작업들은 정적 공간 측정 및 기초 작업, 그리고 속도와 같은 동적 추정 작업을 포함합니다. 실험 결과, 최신 MLLMs조차도 정밀 거리 추정과 운동 분석을 요구하는 과제에서 어려움을 겪고 있음을 보여줍니다.

- **Performance Highlights**: 이 연구의 결과는 MLLMs가 현실 세계의 공간-시간 정보를 정확히 이해하는 데 있어 다수의 한계를 가지고 있음을 강조하고 있습니다. 정량적 평가 및 오차 분석을 통해 공간 정량화의 부정확성, 시간 동역학의 이해 부족, 그리고 크로스 모달 기초와 통합의 약점을 밝혀냅니다. 이로 인해 STI-Bench는 더 나은 MLLMs 개발의 중추적인 기준으로 자리잡을 전망입니다.



### WaveFormer: A 3D Transformer with Wavelet-Driven Feature Representation for Efficient Medical Image Segmentation (https://arxiv.org/abs/2503.23764)
- **What's New**: WaveFormer는 3D-transformer 아키텍처로, 의료 이미지 분석의 새로운 장을 열고 있습니다. 이 모델은 주파수 도메인 속성을 활용하여 효과적으로 맥락을 표현하고, 인간의 시각 인식 시스템의 상위 하향 메커니즘을 통해 설계되었습니다. WaveFormer는 디스크리트 웨이블릿 변환(Discrete Wavelet Transform, DWT)을 여러 스케일에서 활용하여 전역적 맥락과 세부 정보를 동시에 유지하는 혁신적인 접근을 제안합니다.

- **Technical Details**: WaveFormer는 두 가지 주요 설계 원칙을 바탕으로 구성됩니다: 효과적인 전역 맥락 모델링과 세부 정보 보존입니다. DWT를 사용해 저주파 서브밴드를 추출함으로써, WaveFormer는 자가참조(self-attention)를 더 компакт한 표현에서 수행하며 필수적인 맥락 정보를 보존합니다. 또한, IDWT(inverse DWT) 메커니즘을 통해 고해상도 세그멘테이션 마스크를 점진적으로 재구성하여 세밀한 구조를 캡처합니다.

- **Performance Highlights**: WaveFormer는 BraTS2023, FLARE2021, KiTS2023과 같은 주요 3D 의료 벤치마크에서 평가되었습니다. 실험 결과 WaveFormer는 기존 최첨단 방법과 비교하여 경쟁력 있는 정확도를 달성했으며, 모델 복잡성과 추론시간을 현저히 줄였습니다. 이러한 성능 덕분에 WaveFormer는 자원이 제한된 임상 환경에서도 효율적으로 배포할 수 있는 가능성을 보여주고 있습니다.



### Decoupled Distillation to Erase: A General Unlearning Method for Any Class-centric Tasks (https://arxiv.org/abs/2503.23751)
Comments:
          CVPR2025, Equal contributions from first two authors

- **What's New**: 이번 연구에서는 DEcoupLEd Distillation To Erase (DELETE)라는 강력한 Unlearning (언러닝) 방법을 제안합니다. 이 방법은 특정 클래스에 초점을 맞춘 작업에서 기존 방법의 한계를 극복하기 위해 개발되었습니다. DELETE는 Unlearning 손실(loss)을 잊기(Forgetting)와 유지(Retention) 항목으로 분해하여 두 요소를 동시에 최적화합니다.

- **Technical Details**: 우리는 이론적 프레임워크를 통해 Unlearning 손실을 정의하고 조정합니다. 이는 '어두운 지식'(dark knowledge)을 활용하여, 각 클래스의 확률 분포를 분석하여 잊기와 유지의 손실을 구분하는 masking 전략을 사용합니다. DELETE는 기존 언러닝 방법보다 월등한 성능을 보이며, 다양한 데이터셋에서 진행된 실험을 통해 그 효과를 입증했습니다.

- **Performance Highlights**: DELETE는 안면 인식, 백도어 방어, 의미 분할 등 여러 실제 응용 프로그램에서 놀라운 성능을 발휘하며, 두 가지 클래스의 지식을 잊고 남은 클래스의 지식은 유지하는 데 성공했습니다. 우리의 연구는 기존의 클래스 중심 Unlearning 방법에 대한 포괄적이고 공정한 비교를 제시하며, 불완전한 데이터나 개입이 필요 없는 상황에서도 효과적으로 머신 언러닝을 수행할 수 있음을 보여줍니다.



### Consistency-aware Self-Training for Iterative-based Stereo Matching (https://arxiv.org/abs/2503.23747)
Comments:
          Accepted by CVPR 2025

- **What's New**: 이 논문은 스테레오 매칭 분야에서 첫 번째로, 일관성 인식(self-training) 자기 학습 프레임워크를 제안하여 비표시(비라벨)된 실제 데이터를 활용하는 방법을 제시합니다. 기존의 반복 기반 스테레오 매칭 방법들이 레이블된 데이터에 크게 의존하는 문제를 보완하기 위해 개발되었습니다. 본 방식은 모델 예측 시 발생하는 오류가 큰 영역에서 더 강한 진동 특성을 보인다는 통찰을 바탕으로 개발되었습니다.

- **Technical Details**: 제안된 방법은 일관성 인식 소프트 필터링 모듈과 일관성 인식 소프트 가중 손실을 포함합니다. 각각 다중 해상도 예측 일관성 필터와 반복 예측 일관성 필터로 구성되어 있습니다. 이 모듈은 신뢰할 수 없는 가짜 레이블을 소프트 필터링하여 오류 축적을 감소시키고 성능 저하 문제를 완화합니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 방법은 기존 반복 기반 스테레오 매칭 방법의 성능을 향상시키는 것을 입증했습니다. 특히 Middlebury, KITTI2015 및 ETH3D 데이터셋에서 최신 최첨단(SOTA) 방법들보다 더 나은 성능을 달성하였습니다. 이러한 성과는 다양한 도메인에서의 일반화 능력을 높여 주며, 훈련 중 보지 못한 실제 데이터셋에서도 뛰어난 성능을 보여줍니다.



### Short-video Propagation Influence Rating: A New Real-world Dataset and A New Large Graph Mod (https://arxiv.org/abs/2503.23746)
- **What's New**: 이 논문은 Short-video Propagation Influence Rating (SPIR) 작업을 제안하며, 단기적인 인기 예측을 넘어서서 긴 시간에 걸친 비디오의 전파 영향을 추정하려는 새로운 접근 방식을 소개합니다. 이는 사용자의 다양한 상호작용 정보를 고려하여 단일 지표에 의존하지 않고, 실질적인 전파의 영향을 평가하는 것을 목표로 합니다. 또한 최초의 크로스 플랫폼(short-video propagation) 데이터셋인 XS-Video를 도입하여, 5개 주요 플랫폼에서 수집한 비디오 데이터로 구성된 신뢰할 수 있는 근거를 제공합니다.

- **Technical Details**: XS-Video 데이터셋은 총 117,720 개의 비디오와 381,926 개의 샘플, 535 개의 주제를 포함하고 있으며, 비디오의 전파 영향력은 0에서 9까지의 등급으로 주어집니다. 논문은 또한 대규모 그래프 모델(NetGPT)을 제안하여, 서로 다른 형태의 그래프 구조 데이터를 처리하는 것과 동시에 대규모 언어 모델(LLM)의 추론 능력을 결합합니다. NetGPT는 이 새로운 세 단계 훈련 메커니즘을 기반으로 하여, 단기 비디오 전파 그래프를 이해하고 분석함으로써 전파 영향력을 예측할 수 있습니다.

- **Performance Highlights**: 실험 결과, NetGPT는 XS-Video 데이터셋에서 기존의 최첨단 방법들(GNNs, LLMs 및 멀티모달 LLMs)에 비해 월등한 성능을 보여주었습니다. 기존 방법들이 비디오 전파 분석에서 비효율적이라는 점을 보여주며, SPIR 작업에 대한 보다 정확한 접근이 필요하다는 점을 강조합니다. 이 모델은 특히 복잡한 그래프 구조와 비디오의 이종 특성을 포착하여 전파 영향력 수준을 예측하는 데 큰 강점을 보이며, 다양한 응용 분야에서 유용할 것입니다.



### Every Painting Awakened: A Training-free Framework for Painting-to-Animation Generation (https://arxiv.org/abs/2503.23736)
Comments:
          The project is available at: this https URL

- **What's New**: 이번 연구는 실제 정적 그림을 생동감 있게 만드는 훈련이 필요 없는 프레임워크를 제안합니다. 이 프레임워크는 이미지에서 비디오(image-to-video) 합성(I2V) 기술을 통해 정적 작품을 애니메이션할 수 있도록 하고 있습니다. 특히, 기존 I2V 방법들이 정적 사진에 대한 효과적인 움직임 해석에 어려움을 겪고 있다는 점에서 새로운 기법이 필요하다는 것을 강조하고 있습니다.

- **Technical Details**: 제안된 방법에서는 두 가지 주요 혁신이 도입됩니다. 첫째, Dual-path score distillation을 통해 실제 및 합성 데이터를 모두 기반으로 한 움직임 사전(motion priors)을 증류하여 정적 세부 정보를 보존하고 동적인 특성을 학습합니다. 둘째, 하이브리드 잠재 융합(hybrid latent fusion) 기법을 사용하여 실제 그림과 합성된 이미지에서 추출한 하이브리드 특징을 잠재 공간(latent space)에서 원활하게 통합합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 텍스트 프롬프트와의 의미적 일치를 크게 향상시키는 동시에 원작의 독특한 특성과 무결성을 충실히 유지하는 것으로 나타났습니다. 모델 학습이나 추가 파라미터 테스트 없이도 이러한 동적 효과를 달성할 수 있어, 기존 I2V 방법과의 플러그 앤 플레이 통합이 가능하다는 점에서 실용적인 활용 가능성을 제공합니다.



### Investigation of intelligent barbell squat coaching system based on computer vision and machine learning (https://arxiv.org/abs/2503.23731)
- **What's New**: 이번 연구는 인공지능(AI)과 컴퓨터 비전 기술을 활용하여 바벨 스쿼트 훈련의 효율성을 높이는 시스템을 개발하였습니다. 본 시스템은 실시간으로 문제를 진단하고 피드백을 제공하는 기능을 갖추고 있어 혼자서도 제대로 훈련할 수 있도록 지원합니다. 또한, 재생 모드를 통해 사용자가 이전 스쿼트를 분석하고 코멘트를 확인할 수 있습니다.

- **Technical Details**: 총 77명의 참가자로부터 8,151개의 스쿼트를 수집하여 각각 좋은 스쿼트 및 여섯 가지 문제로 분류했습니다. 이후, 세 가지 머신러닝 아키텍처를 통해 진단 모델을 훈련하였고, SHAP 방법을 적용하여 문제 예측의 정확성을 향상시켰습니다. 이 시스템은 스쿼트를 진단하는데 0.5초 이하의 시간을 소요합니다.

- **Performance Highlights**: 여섯 가지 문제에 대한 F1 점수는 각각 86.86%, 69.01%, 77.42%, 90.74%, 95.83%, 100%에 도달하였습니다. 시스템을 사용하여 훈련한 참가자들은 기술적으로 상당한 개선을 보였으며, 이는 전문 웨이트 리프팅 코치에 의해 평가되었습니다. 이 연구는 실시간으로 사용자 친화적인 바벨 스쿼트 피드백 및 훈련 시스템을 구축하는 것을 목표로 한 종합적인 연구입니다.



### KOFFVQA: An Objectively Evaluated Free-form VQA Benchmark for Large Vision-Language Models in the Korean Languag (https://arxiv.org/abs/2503.23730)
Comments:
          Accepted to CVPRW 2025, Workshop on Benchmarking and Expanding AI Multimodal Approaches

- **What's New**: 최근 대규모 비전-언어 모델(Visual-Language Models, VLMs)의 발전으로 다양한 평가 기준이 등장했습니다. 하지만 기존 평가 방법들은 주어진 응답 중에서 모델이 선택하도록 요구하거나, 주판 모델(judge model)을 사용하여 주관적이라는 문제점이 있었습니다. 본 연구에서는 한국어를 위한 새로운 평가 기준을 제공하는 KOFFVQA 벤치마크를 제안합니다.

- **Technical Details**: KOFFVQA는 275개의 주어진 이미지와 질문 쌍을 포함하며, 10가지 VLM 성능 측면을 평가하는 grading criteria(채점 기준)를 제공합니다. 각 응답은 미리 정의된 채점 기준을 기반으로 LLM(대형 언어 모델)으로 채점되며, 이는 평가의 신뢰성을 높이는 데 기여합니다. 이를 통해 작은 오픈 소스 모델이라도 신뢰할 수 있는 평가를 할 수 있습니다.

- **Performance Highlights**: KOFFVQA 벤치마크를 활용하여 47개의 VLM 모델을 평가한 결과, 한국어 언어에서의 성능은 영어 벤치마크에서의 성능과는 상이한 패턴을 보였습니다. 우리의 접근 방식은 기존 방법과 비교하여 평가의 일관성을 크게 향상시켰고, 이는 장기적인 응답을 평가할 때 발생할 수 있는 주관적 문제들을 줄여줍니다. LLM을 평가지로 사용한 방법은 특히 더 신뢰할 수 있는 결과를 제공했습니다.



### Exploring Temporal Dynamics in Event-based Eye Tracker (https://arxiv.org/abs/2503.23725)
Comments:
          Accepted by CVPR 2025 Event-based Vision Workshop

- **What's New**: 본 연구에서는 TDTracker라는 효과적인 안구 추적 프레임워크를 제안한다. 이 시스템은 빠른 안구 움직임을 신중하게 모델링하여 높은 속도와 정밀도로 안구를 추적한다. TDTracker는 3D convolutional neural networks를 이용하여 implicit short-term temporal dynamics를 포착하고, Frequency-aware Module, GRU, Mamba로 구성된 cascaded 구조를 활용하여 explicit long-term temporal dynamics를 추출한다.

- **Technical Details**: TDTracker는 두 가지 주요 구성 요소인 implicit temporal dynamic (ITD)와 explicit temporal dynamic (ETD)로 구성되어 있다. ITD는 3D convolutional neural networks를 사용하여 단기적인 시간적 특징을 효과적으로 추출하고, ETD는 세 가지 고급 시간 모델로 구성된 cascade 구조를 통해 장기적인 시간적 특징을 명시적으로 추출한다. 이 방법은 복잡하고 지속적인 시간 의존성을 포착하여 동적 추적 작업에서 성능을 향상시킨다.

- **Performance Highlights**: TDTracker는 SEET 데이터셋에서 최신의 성능(state-of-the-art, SOTA)을 달성했으며, CVPR 2025 이벤트 기반 안구 추적 챌린지에서 3위를 차지했다. TDTracker는 이전의 SOTA인 EventMamba보다 더 적은 부동소수점 연산(FLOPs)으로 우수한 성능을 기록하였다.



### LATex: Leveraging Attribute-based Text Knowledge for Aerial-Ground Person Re-Identification (https://arxiv.org/abs/2503.23722)
- **What's New**: 이번 논문에서는 Aerial-Ground person Re-Identification (AG-ReID) 분야에서 새로운 프레임워크인 LATex를 제안합니다. LATex는 사람 속성에 기반한 텍스트 지식을 활용하는 프롬프트 튜닝(prompt-tuning) 전략을 채택하여 특징 분별력을 향상시킵니다. 연구에서는 특히 속성 정보가 AG-ReID 작업의 성능을 어떻게 향상시킬 수 있는지를 탐구하고 있습니다.

- **Technical Details**: LATex 프레임워크는 세 가지 주요 구성 요소로 이루어져 있습니다: Attribute-aware Image Encoder (AIE), Prompted Attribute Classifier Group (PACG), Coupled Prompt Template (CPT). AIE는 CLIP의 사전 학습된 지식을 AG-ReID에 적용하기 위해 조정된 이미지 인코더입니다. PACG는 사람 속성을 예측하고, CPT는 속성 정보를 구조화된 문장으로 변환하여 CLIP의 텍스트 인코더에 의해 처리됩니다.

- **Performance Highlights**: 세 가지 AG-ReID 벤치마크에서의 실험을 통해 LATex의 효과성과 효율성을 입증하였습니다. 본 프레임워크는 전통적인 방법보다 적은 학습 비용으로 더 뛰어난 분별력을 제공합니다. 논문에서는 AG-ReID 과제에서 시각적 변화를 극복하는 데 있어 속성 기반 접근 방식의 중요성을 강조하고 있습니다.



### Effective Cloud Removal for Remote Sensing Images by an Improved Mean-Reverting Denoising Model with Elucidated Design Spac (https://arxiv.org/abs/2503.23717)
Comments:
          29 pages, 12 figures

- **What's New**: 이 논문에서는 기존의 diffusion models (DMs)의 한계를 극복하기 위한 새로운 cloud removal (CR) 모델인 EMRDM을 제안합니다. EMRDM은 mean-reverting diffusion models (MRDMs)를 기반으로 하여, 흐림이 있는 이미지에서 클라우드가 없는 이미지를 직접 생성할 수 있는 접근 방식을 제공합니다. 이를 통해 cloudy 이미지의 내재된 정보를 활용하고, 이미지 생성 과정에서의 일관성을 개선합니다.

- **Technical Details**: EMRDM은 모듈화된 프레임워크를 통해 동적으로 업데이트 가능한 모듈 디자인을 제공합니다. 이 모델은 일반적인 forward 및 backward process를 통해 CR 성능을 최적화하며, 특히 denoiser, training 과정, sampling 프로세스를 개선합니다. 다중 시계열 CR 기능을 위해서는 연속 이미지를 동시에 denoise 하는 네트워크를 개발하여 다양한 길이의 이미지 시퀀스를 처리합니다.

- **Performance Highlights**: 여러 mono-temporal 및 multi-temporal 데이터셋에 대한 실험 결과, EMRDM은 기존 CR 모델보다 뛰어난 성능을 보여줍니다. 특히, 제안된 framework는 구조적 일관성을 강화하며, 다양한 노이즈 수준에 최적화된 denoising 효과를 제공합니다. 이러한 거대한 성능 향상은 EMRDM의 혁신적인 모듈 설계와 효과적인 샘플링 프로세스 덕분입니다.



### HOIGen-1M: A Large-scale Dataset for Human-Object Interaction Video Generation (https://arxiv.org/abs/2503.23715)
Comments:
          CVPR 2025

- **What's New**: 이번 연구에서는 HOI (Human-Object Interaction) 생성을 위한 최초의 대규모 데이터셋인 HOIGen-1M을 소개합니다. HOIGen-1M은 다양한 출처에서 수집된 100만 개 이상의 고품질 비디오로 구성되어 있으며, 정확한 캡션이 있는 데이터들이 포함되어 있습니다. 기존 T2V (Text-to-Video) 모델들이 HOI 비디오를 정확히 생성하지 못하는 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: 데이터셋의 고품질 확보를 위해, MLLM (Multimodal Large Language Models)을 활용한 효율적인 프레임워크를 설계하였고, 인간 주석자에 의한 추가적인 비디오 정제 과정도 거쳤습니다. 또한, Mixture-of-Multimodal-Experts (MoME) 전략을 기반으로 한 새로운 비디오 설명 방법을 통해 정확한 텍스트 캡션을 생성하고 있습니다. 이 데이터는 HOI 비디오 생성을 위한 자동 평가 메트릭스를 제공하여 성능 향상을 도모하고 있습니다.

- **Performance Highlights**: 연구 결과, 기존 T2V 모델들이 고품질 HOI 비디오를 생성하는 데 어려움을 겪고 있음을 확인했습니다. HOIGen-1M 데이터셋으로 세밀하게 조정한 T2V 모델의 성능이 상당히 향상되었으며, Cog-VideoX-5B라는 오픈소스 모델이 상업 소프트웨어인 Kling 1.5와 비슷한 성능을 달성하였습니다. 이러한 결과는 HOIGen-1M의 중요성을 입증하고 있습니다.



### ElimPCL: Eliminating Noise Accumulation with Progressive Curriculum Labeling for Source-Free Domain Adaptation (https://arxiv.org/abs/2503.23712)
Comments:
          ICME 2025 camera-ready

- **What's New**: 이번 연구는 소스 데이터 없이 목표 모델을 학습하는 Source-Free Domain Adaptation (SFDA) 방법론을 소개합니다. 기존 방법에서 발생하는 고노이즈 샘플로 인한 문제를 해결하기 위해, 신뢰할 수 있는 유사 레이블(pseudo-label)을 생성하고 이를 기반으로 훈련할 수 있는 Progressive Curriculum Labeling (ElimPCL) 방식을 제안합니다. 이 방법은 어려운 샘플에서의 노이즈를 줄이고, 훈련이 진행되는 동안 잘못된 레이블의 영향을 최소화하는 것을 목표로 합니다.

- **Technical Details**: SFDA의 주요 기술적 성과는 고노이즈 샘플의 발생을 줄이고, 이를 개선하기 위한 Dual MixUP 기법을 도입했다는 점입니다. ElimPCL은 프로토타입 일관성을 기반으로 한 커리큘럼을 설정해 신뢰할 수 있는 샘플로부터 학습을 진행하며, Dual MixUP은 Feature Space 내에서 상호작용을 통해 노이즈 전파를 줄입니다. 이러한 과정을 통해 SFDA에서 모델이 어렵고 노이즈가 많은 샘플을 보다 효과적으로 분리할 수 있도록 지원합니다.

- **Performance Highlights**: 실험 결과, ElimPCL은 기존 최첨단(SOTA) 방법과 비교하여 어려운 태스크에서 최대 3.4%의 성능 향상을 보여주었습니다. 특히 도메인 변화가 심한 과제에서 ElimPCL은 더욱 강력한 경쟁력을 가지며, 다양한 기준 데이터세트에서 그 효과가 입증되었습니다. 이 연구는 SFDA 분야에서의 진전을 이끌어나갈 수 있는 가능성을 제시합니다.



### Expanding-and-Shrinking Binary Neural Networks (https://arxiv.org/abs/2503.23709)
- **What's New**: 본 논문에서는 binary neural networks (BNNs)의 제한된 표현 능력을 해결하기 위한 새로운 접근 방식으로 expanding-and-shrinking operation을 제안합니다. 이 방법은 계산 복잡도의 최소 증가로 BNN의 출력 특징의 표현 능력을 향상시킵니다. ES-BNN으로 명명된 이 모델은 다양한 BNN 아키텍처에서 우수한 일반화 성능을 보이며, 이미지 분류, 객체 탐지 및 생성적 확산 모델 과제에서 탁월한 결과를 보입니다.

- **Technical Details**: 해당 연구에서는 가중치와 활성화를 이진화하면서 발생하는 정보 손실을 최소화하기 위한 다양한 기존 방법을 논의합니다. BNN의 경우, 각 층의 출력 특징의 수치는 이진화에 의해 제약을 받으며, 이는 제한된 표현 용량으로 이어져 정확도 저하를 초래합니다. 제안된 expanding-and-shrinking operation은 이러한 제한을 극복하고 각 이진층에서의 표현 용량을 강화합니다.

- **Performance Highlights**: ES-BNN은 여러 벤치마크에서 우수한 성능을 입증하여 기존의 다양한 이진화 알고리즘보다 향상된 정확도를 달성했습니다. 특히, CNN 및 Transformers와 같은 다양한 네트워크 아키텍처에 적용 가능하며, 모든 다운스트림 애플리케이션의 일반화 성능 또한 뛰어납니다. 실험 결과는 이 방법이 다양한 실험지표에서 두드러진 성능 향상을 보인다는 것을 보여줍니다.



### 3D Dental Model Segmentation with Geometrical Boundary Preserving (https://arxiv.org/abs/2503.23702)
Comments:
          The IEEE/CVF Conference on Computer Vision and Pattern Recognition 2025

- **What's New**: 이번 논문에서는 CrossTooth라는 새로운 경계 보존 분할(segmentation) 방법을 제안합니다. 이 방법은 3D mesh selective downsampling을 활용하여 치아와 잇몸의 경계에서 더 많은 꼭짓점을 유지하며, 멀티 뷰 렌더링(multi-view rendering)된 이미지에서 추출한 교차 모달 경계 특징(cross-modal boundary features)을 결합하여 기하학적 표현을 강화합니다. CrossTooth는 기존 방법들보다 더 높은 분할 정확도를 달성하였으며, 이는 공공 intraoral scan 데이터셋에서의 실험으로 입증되었습니다.

- **Technical Details**: CrossTooth는 기존의 QEM 방식 대신 선택적 다운샘플링(selective downsampling) 방법을 개발하여 경계 세부정보를 보다 잘 보존합니다. 이 방법은 인트라오랄 스캔(intraoral scan)에서 얻은 곡률(curvature) 정보를 포함하여 렌더링된 이미지에서 생성된 밀집(다량의) 특징을 활용합니다. 마지막 단계에서는 MLP를 통해 멀티 뷰 이미지로부터의 특징 표현과 점구름(point cloud)으로부터의 특징 표현을 결합하여 정보 손실을 보상하는 접근 방식을 채택합니다.

- **Performance Highlights**: CrossTooth는 실험 결과, 기존의 최첨단 방법들에 비해 현저한 성능 향상을 보여주었습니다. 특별히, CrossTooth는 치아와 잇몸의 경계를 보다 정확하게 식별할 수 있으며, 이는 후속 데이터 처리 작업에서 매우 중요합니다. 실험에서 CrossTooth는 높은 정확도의 분할을 실현하며, 여러 연구에서 제시된 다른 방법들과 비교하여 더 우수한 결과를 기록하였습니다.



### Detail-aware multi-view stereo network for depth estimation (https://arxiv.org/abs/2503.23684)
- **What's New**: 이번 연구에서는 깊이 추정(depth estimation) 분야에서 기존의 다중 뷰 스테레오 방법들이 물체 경계(object boundaries)와 세부 영역(detail regions)의 복구에서 부족한 성능을 보이는 문제를 해결하기 위해, 세부 인식 다중 뷰 스테레오 네트워크(DA-MVSNet)를 제안합니다. 이 네트워크는 코스-투-파인(coarse-to-fine) 프레임워크를 사용하여 깊이 추정의 품질을 향상시킵니다.

- **Technical Details**: DA-MVSNet에서는 코스 단계에서 숨겨진 기하학적 깊이 단서(geometric depth clues)를 활용하여 물체 표면(object surfaces) 간의 기하학적 구조 관계를 유지하고 이미지 특징(image features)의 표현력을 강화합니다. 또한, 이미지 합성 손실(image synthesis loss)을 사용하여 세부 영역의 기울기 흐름(gradient flow)을 제약하고 물체 경계와 질감이 풍부한 영역에 대한 감독을 강화합니다.

- **Performance Highlights**: DTU 및 Tanks & Temples 데이터셋을 통한 광범위한 실험 결과, 제안한 방법이 경쟁력 있는 성능을 달성함을 보여줍니다. 연구에서 제안한 적응형 깊이 간격 조정 전략(adaptive depth interval adjustment strategy)은 물체 재구성의 정확도를 더욱 향상시키는데 기여합니다.



### The Devil is in the Distributions: Explicit Modeling of Scene Content is Key in Zero-Shot Video Captioning (https://arxiv.org/abs/2503.23679)
Comments:
          13 pages

- **What's New**: 본 연구는 Zero-shot video captioning의 새로운 방향으로, 기존의 방법들이 특정 장면 요소에만 집중하는 경향을 극복하기 위해 진보된 다중-세부도 텍스트 프롬프트 방식을 제안합니다. 이 방법은 명사구, 장면 그래프, 전체 문장으로 구성된 세 가지 메모리 뱅크를 만들어 더 정확하고 완전한 캡션을 생성할 수 있게 합니다. 또한 특정 주제를 둘러싼 자연어의 분포를 모델링하는 카테고리 인식 검색 메커니즘을 도입하여 결국 5.7%, 16.2%, 3.4%의 CIDEr 향상을 달성했습니다.

- **Technical Details**: 이 방법론은 세 가지 주요 구성 요소로 이루어집니다: (1) 세 가지의 진보된 세부도로 메모리 뱅크를 구성하여 각기 다른 메모리 뱅크에서 관련 텍스트 단위를 검색하는 것을 포함합니다. (2) 훈련 과정에서는 텍스트 임베딩의 변조된 형태를 활용하여 메모리 뱅크에서 유용한 프롬프트를 검색합니다. (3) 추론 과정에서는 카테고리 인식 검색을 통해 비주얼적으로 관련성이 높은 다양한 프롬프트를 생성합니다. 이러한 과정들은 텍스트 디코더에 포괄적인 의미 단서를 제공합니다.

- **Performance Highlights**: MSR-VTT, MSVD, VATEX 벤치마크에서 우리의 방법은 각각 5.7%, 16.2%, 3.4%의 CIDEr 개선을 보이며 기존의 최선의 방법들에 비해 우수한 성능을 입증하였습니다. 이로써 우리의 접근 방식이 Zero-shot video captioning 영역에서 의미 있는 발전을 제공함을 확인했습니다. 이러한 성과는 단순한 명사구나 전체 문장에 의한 제한적 검색에서 벗어나 다수의 세부 정보를 제공할 수 있는 다양한 텍스트 단위를 활용하는 것이 가능함을 보여줍니다.



### Learning Bijective Surface Parameterization for Inferring Signed Distance Functions from Sparse Point Clouds with Grid Deformation (https://arxiv.org/abs/2503.23670)
Comments:
          Accepted by Conference on Computer Vision and Pattern Recognition (CVPR) 2025. Project page:this https URL

- **What's New**: 이 논문에서는 희소 점 구름(sparse point clouds)으로부터 signed distance functions (SDFs)를 추론하는 새로운 접근법을 제시합니다. 기존 방법들은 희소 점 구름에서 연속적인 기하학적 정보를 얻는 데 어려움을 겪었습니다. 이 연구는 새로운 동적 변형 네트워크(dynamic deformation network)를 통해 SDF를 end-to-end 방식으로 예측하는 방식을 도입합니다.

- **Technical Details**: 제안된 방법은 bijective surface parameterization (BSP)을 통해 희소 점으로부터 연속적인 전역 표면을 학습합니다. 이 과정에서, 3D 점을 구면 매니폴드(sphere manifold)로 변환하고, 더 많은 샘플 코드를 사용하여 조밀화된 패치를 생성합니다. 또한, grid deformation optimization (GDO)을 적용하여 조밀화된 점들에 대한 SDF를 보다 정밀하게 추정합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 합성 데이터와 실제 스캔 데이터 모두에서 현재 최첨단 방법들을 크게 초월하는 성능을 보여주었습니다. 연구진은 제안한 방법의 가능성을 통해 더욱 복잡한 기하학적 구조를 가진 점 구름에서도 우수한 성능을 발휘할 수 있음을 확인했습니다.



### Context-Independent OCR with Multimodal LLMs: Effects of Image Resolution and Visual Complexity (https://arxiv.org/abs/2503.23667)
- **What's New**: 이 연구에서는 다양한 시각 복잡성을 가진 단일 문자 이미지를 사용하여 문맥에 독립적인 OCR(Optical Character Recognition) 작업을 조사하였습니다. 다중 모달 대형 언어 모델(multimodal LLMs)은 OCR에 있어 기존의 OCR 전용 모델보다 우수한 성능을 보이지만, 낮은 해상도에서 성능이 크게 저하된다는 점이 발견되었습니다. 300 ppi에서 기존 OCR 방법과 유사한 성능을 나타내지만, 150 ppi 이하에서는 성능이 심각하게 감소하게 됩니다.

- **Technical Details**: 이 연구에서는 2,136개의 jōyō kanji 문자를 대상으로 OCR 정확도를 평가하기 위해 문자 이미지의 시각 복잡성을 정량적으로 측정합니다. 복잡성 지수는 프랙탈 차원(fractal dimension)과 샤논 엔트로피(Shannon entropy)를 사용하여 계산됩니다. 각 문자의 프랙탈 차원은 박스 카운팅(box-counting) 방법으로 측정되며, 샤논 엔트로피는 픽셀 값의 분포를 기반으로 계산됩니다. 총 400개의 이미지가 생성되어 다중 모달 LLM과 기존 OCR 모델 간의 비교 실험이 수행됩니다.

- **Performance Highlights**: 결과적으로 다중 모달 LLMs는 300 ppi의 해상도에서 기존 OCR 방법과 유사한 정확도를 보이지만, 150 ppi 이하에서는 성능이 크게 저하됩니다. 특히 GPT-4o 모델은 높은 오인식률을 기록하며, 모든 해상도에서 반복적으로 잘못 인식된 문자들이 확인되었습니다. 또한, 오인식의 빈도는 문자의 복잡성과 관련이 있다고 나타났으며, 프랙탈 차원 및 엔트로피가 증가할수록 오인식되는 문자 수가 증가하는 경향이 관찰되었습니다.



### LiM-Loc: Visual Localization with Dense and Accurate 3D Reference Maps Directly Corresponding 2D Keypoints to 3D LiDAR Point Clouds (https://arxiv.org/abs/2503.23664)
Comments:
          8 pages, 6 figures

- **What's New**: 본 논문은 LiDAR 포인트 클라우드와 이미지를 결합하여 밀도 높고 정확한 3D 참조 맵을 생성하는 LiM-Loc 방법을 제안합니다. 이 기술은 기존의 특징 매칭(feature matching)을 회피하며, 거의 모든 키포인트를 정확하게 3D로 재구성할 수 있도록 합니다. 또한, 넓은 영역의 LiDAR 포인트 클라우드를 이용해 카메라에 보이지 않는 점들을 제거하고 2D-3D 일치 오류를 줄이는 방법을 포함하고 있습니다.

- **Technical Details**: LiM-Loc 방법은 광역 LiDAR 포인트 클라우드를 사용하여 숨겨진 포인트 제거(HPR, Hidden Point Removal) 및 구형 압축(spherical compression)을 수행합니다. 포인트 클라우드는 2D LiDAR 가상 이미지에 투영되며, 카메라 화면 좌표에서 완벽하게 겹치는 이미지를 통해 직접 매칭됩니다. 이 방법은 특정 지역 기능(local features)에 의존하지 않고 다양한 최신 기능에 적용될 수 있습니다.

- **Performance Highlights**: 제안된 방법은 실내 및 실외 데이터셋을 활용하여 여러 최첨단(local features) 특징에 적용해 정확도 향상을 입증하였습니다. 밀도가 높고 정확한 3D 참조 맵이 카메라 포즈 추정을 개선함을 보여주며, 기존보다 더 나은 성능을 기록했습니다. 또한, LiDAR와 카메라의 정확한 보정을 통해 센서 융합(sensor fusion)이 용이해져 다양한 응용 사례에 적합합니다.



### DeepDubber-V1: Towards High Quality and Dialogue, Narration, Monologue Adaptive Movie Dubbing Via Multi-Modal Chain-of-Thoughts Reasoning Guidanc (https://arxiv.org/abs/2503.23660)
Comments:
          11 pages, 5 figures

- **What's New**: 이 논문은 영화 더빙 기술의 현재 한계를 극복하기 위해 다중 모달 대형 언어 모델(MLLM)을 제안합니다. 기존의 더빙 기술은 화자의 나이와 성별과 같은 세부 사항을 잘 이해하지 못했지만, 제안된 방법은 이러한 세부 사항을 명확하게 파악하여 더빙 스타일을 효과적으로 적용할 수 있습니다. 또한, CoT(Chain-of-Thought) 주석이 포함된 새로운 데이터셋을 개발하여 연구의 기초를 다졌습니다. 이를 통해 여러 데이터셋에서 최신 기술들에 비해 성능이 향상되었음을 확인했습니다.

- **Technical Details**: DeepDubber 모델은 비디오 클립과 자막을 입력으로 받아 이를 기반으로 완전한 더빙 음성을 생성하는 것을 목표로 합니다. 이 모델은 단계별 추론을 통해 장면 유형, 화자 성별, 나이, 감정과 같은 핵심 의미 특징을 추출합니다. 특히, 다중 모달 CoT 학습을 통해 모델의 추론 능력을 강화하고, 이를 위해 조건부 DiT(디퓨전 기반) 음성 생성기를 통합하여 더빙의 품질을 높였습니다.

- **Performance Highlights**: 성능 평가 결과, 제안된 DeepDubber 모델은 여러 데이터셋에서 사용할 때 특히 높은 품질의 더빙을 제공했습니다. 예를 들어, V2C Animation 데이터셋에서 SPK-SIM과 EMO-SIM 메트릭이 각각 82.48%에서 89.74%, 66.24%에서 78.88%로 증가하였고, Grid 데이터셋의 LSE-D와 MCD-SL 메트릭은 각각 14.79에서 14.63, 5.24에서 4.74로 감소하는 성과를 보였습니다. 이러한 성과들을 통해 DeepDubber의 뛰어난 성능을 입증했습니다.



### Introducing the Short-Time Fourier Kolmogorov Arnold Network: A Dynamic Graph CNN Approach for Tree Species Classification in 3D Point Clouds (https://arxiv.org/abs/2503.23647)
- **What's New**: 이번 논문에서는 Short-Time Fourier Transform (STFT)을 기반으로 한 새로운 Kolmogorov-Arnold network (KAN)인 STFT-KAN을 소개합니다. STFT-KAN은 기존의 KAN 아키텍처를 개선하여 효율적인 파라미터 제어와 더 나은 해석력을 제공합니다. 이 연구는 경량 DGCNN (liteDGCNN) 아키텍처를 통해 나무 종 분류 문제에서 STFT-KAN을 활용하고, 기존의 KAN 변형 모델보다 우수한 성능을 나타냅니다.

- **Technical Details**: STFT-KAN은 다양한 주파수 범위에서 가변적인 창 크기를 활용하여 비정상 주파수 특징을 포착하는 유연한 표현을 제공합니다. 이 구조는 기존의 선형 층을 대체하여 모델의 복잡성을 줄이고, 학습 가능한 계수를 적용하여 주파수 패턴을 추출합니다. STFT-KAN은 경량 설계의 차세대 DGCNN 아키텍처에서 MLP를 대체하는 형태로 구현되었습니다.

- **Performance Highlights**: 실험 결과 STFT-KAN 기반의 liteDGCNN은 유사한 MLP 기반 모델에 비해 50%의 파라미터 수를 줄이며 경쟁력 있는 정확도를 보여주었습니다. 또한 최첨단 3D 포인트 클라우드 학습 방법들과 비교했을 때도 PointMLP-lite에 가까운 성능을 유지하며, 파라미터 수를 89% 줄이는 성과를 달성했습니다. 결과적으로 STFT-KAN은 KAN 기반 liteDGCNN 버전들 중에서도 가장 높은 정확도와 효율성을 보였습니다.



### Language-Guided Trajectory Traversal in Disentangled Stable Diffusion Latent Space for Factorized Medical Image Generation (https://arxiv.org/abs/2503.23623)
Comments:
          10 pages

- **What's New**: 이 연구는 의학적 이미지 데이터셋에 대해 미세 조정된 비전-언어 기초 모델의 잠재적 요인 분리(disentanglement) 능력을 최초로 조사합니다. 텍스트-이미지 확산 모델을 통해 의학 이미지 생성 및 보간(interpolation)을 위한 새로운 방법론을 제안하며, 이를 통해 다양한 요인들을 조작할 수 있는 가능성을 열었습니다. 또한, Classifier Flip Rate along a Trajectory (CFRT)라는 새로운 메트릭을 도입하여 요인 분리의 유효성을 검증합니다.

- **Technical Details**: 본 연구의 방법론은 두 단계로 구성됩니다. 첫 번째 단계에서는 사전 훈련된 Stable Diffusion 모델을 관련 데이터셋(예: CheXpert, ISIC)에 대해 미세 조정하며, 두 번째 단계에서는 텍스트 임베딩에 조건화된 DDIM(Denoising Diffusion Implicit Model)을 사용하여 경로를 따라 이미지를 합성합니다. 이때 텍스트 임베딩을 통해 특정 요인에 대한 잠재 공간의 경로를 식별할 수 있습니다.

- **Performance Highlights**: 실험 결과, 텍스트에 의해 조정된 Stable Diffusion 모델이 이미지 생성 과정에서 환자의 해부학적 구조와 진단 특성을 자연스럽게 분리시킬 수 있음을 보여줍니다. 생성된 이미지가 특정 속성을 유지하는 경향이 있으며, 이 속성은 시작 지점에서의 거리와 비례하여 강조됩니다. 이러한 결과는 기계 학습 모델이 더 강력하고 해석 가능한 표현을 배우도록 돕는 데 중요한 기여를 할 것입니다.



### Leveraging Vision-Language Foundation Models to Reveal Hidden Image-Attribute Relationships in Medical Imaging (https://arxiv.org/abs/2503.23618)
- **What's New**: 본 연구에서는 Vision-language foundation models (VLMs)가 고해상도의 이미지를 생성하며, 알려지지 않은 데이터 속성을 식별하는 데 어떻게 기여할 수 있는지를 탐구했습니다. 특히, 흉부 X-선 데이터셋에서 우리의 방법을 평가하여 이들 모델이 Structural Causal Models (SCMs)과 비교하여 높은 품질의 이미지를 생성하는 것을 보여주었습니다. 고해상도의 사실적인 이미지를 생성할 수 있는 VLM의 잠재력을 입증함으로써, 이러한 모델이 메타데이터의 한계로 인해 숨겨진 데이터 관계를 드러낼 수 있음을 처음으로 보여주었습니다.

- **Technical Details**: 이 연구에서는 Structural Causal Models (SCMs)를 활용한 반사실 생성 방법론을 제시하며, 이는 기존 속성 간의 관계를 명시적으로 강제함으로써 허위 상관관계 문제를 극복합니다. 또한, Stable Diffusion 모델을 기반으로 하는 우리의 생성 프로세스에서는 각 이미지의 속성을 언어적으로 안내하여 조건화합니다. 우리는 LANCE 기법을 사용하여 이미지 생성 시 개체 식별을 유지하는 한계를 극복하고, 의학적 이미지 배포를 잘 반영하도록 모델을 세밀하게 조정했습니다.

- **Performance Highlights**: 실험 결과, 미세 조정된 VLM이 정밀한 이미지 편집이 가능하다는 것을 보여줍니다. 또한, 데이터셋 속성이 학습 과정에 미치는 영향을 분석하여, 기존 데이터 레이블에 명시적으로 포함되지 않은 속성들을 밝혀냅니다. VLM의 단점으로는 알려지지 않은 데이터 생성 과정에서 허위 상관관계에 의존하여 실제 데이터 생성 과정과의 편차가 발생할 수 있다는 점이 두드러졌습니다.



### Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries (https://arxiv.org/abs/2503.23606)
Comments:
          Accepted to CVPR 2025. Project page: this https URL

- **What's New**: 이번 연구에서는 photon-limited 이미지를 이용해 물체의 깊이를 정확하게 추정하는 새로운 방법, Blurry-Edges를 제안합니다. Blurry-Edges는 이미지 패치를 경계, 색상, 부드러움과 같은 저수준 정보를 저장하고 시각화하는 새로운 방식입니다. 이 접근법을 사용해 비선형적이고 노이즈가 포함된 이미지 쌍으로부터 깊이를 효과적으로 계산할 수 있습니다.

- **Technical Details**: 제안된 방법은 변형 렌즈가 장착된 카메라로 정적 장면을 다양하게 초점 맞춘 이미지를 캡쳐하여 사용합니다. Blurry-Edges는 각 패치의 경계 위치, 부드러움, 색상에 관한 최적의 매개변수를 예측하기 위해 딥 뉴럴 네트워크를 활용합니다. 이를 통해 기존 DfD 알고리즘보다 4배 높은 노이즈 수준에서도 깊이를 안정적으로 추정할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 자연 이미지를 포함한 다양한 데이터셋에서 기존 DfD 알고리즘보다 더 높은 깊이 추정 정확도를 제공합니다. Blurry-Edges는 깊이 예측 외에도 모든 부드러움의 경계 지도와 노이즈 없는 색상 지도를 동시에 생성할 수 있습니다. 전반적으로 제안된 연구는 photon-limited 환경에서도 우수한 강건성을 가진 깊이 추정 방법을 제시하고 있습니다.



### PhysPose: Refining 6D Object Poses with Physical Constraints (https://arxiv.org/abs/2503.23587)
Comments:
          Project page: this https URL

- **What's New**: 이번 연구에서는 물리적 일관성을 적용한 새로운 자세 추정 방법인 PhysPose를 소개합니다. 기존 방법론들이 지니고 있던 다양한 물리적 및 기하학적 제약을 무시하는 문제를 해결하고자 합니다. PhysPose는 포스트 프로세싱 최적화를 통해 비침투(non-penetration) 및 중력(gravitational) 제약을 적용하여 더욱 정교한 자세 추정을 가능하게 합니다.

- **Technical Details**: PhysPose는 장면의 기하학적 정보를 활용하여 각 객체의 6D 포즈를 추정하는 프레임워크입니다. 초기 포즈 추정 후 포스트 프로세싱 방법으로 물리적 일관성을 확보하고, 이는 한 장면에서 물체 간의 부딪힘이나 부유 현상이 없도록 합니다. 이 방법론은 로봇 비전 시스템에서 사용되는 다양한 기법들과 호환될 수 있도록 설계되었습니다.

- **Performance Highlights**: PhysPose는 YCB-비디오와 HOPE-비디오 데이터셋에서 기존의 최신 기술들보다 우수한 성능을 보였습니다. 특히 실제 로봇 피킹 앤 플레이스(pick-and-place) 작업에서의 성공률을 크게 향상시켜, 물리적 일관성을 유지하는 것이 실제 응용에서 얼마나 중요한지를 강조합니다.



### DiT4SR: Taming Diffusion Transformer for Real-World Image Super-Resolution (https://arxiv.org/abs/2503.23580)
- **What's New**: 최근 대규모 프리트레인된 확산 모델들이 리얼 월드 이미지 슈퍼 해상도(Real-ISR) 문제 해결에 점점 더 많이 사용되고 있습니다. 기본 아이디어는 기존 이미지 생성 기술인 UNet 기반 아키텍처 대신 확산 변환기(Diffusion Transformer)를 활용하는 것입니다. 이 논문에서는 DiT4SR이라는 새로운 모델을 제안하여 대규모 DiT 모델을 Real-ISR에 맞게 조정합니다.

- **Technical Details**: DiT4SR은 저해상도(LR) 이미지에서 추출된 임베딩을 직접 주입하는 대신, LR 임베딩을 DiT의 기본 어텐션 메커니즘에 통합합니다. 이러한 방식을 통해 LR 공간과 생성된 공간 간의 양방향 정보 흐름을 가능하게 합니다. 또한, LR 가이드는 크로스 스트림 합성곱(convolution) 레이어를 통해 생성된 잠재 공간에 주입되어 DiT의 세밀한 정보 캡처 능력을 보완합니다.

- **Performance Highlights**: 많은 실험을 통해 DiT4SR은 Real-ISR 작업에서 우수한 성능을 발휘하는 것으로 입증되었습니다. LR 정보와의 원활한 상호작용을 통해 생성된 이미지는 점진적으로 최신화된 세부정보를 반영합니다. 이 모델은 기존의 ControlNet 접근법 없이도 DiT의 장점을 최대한 활용하여 보다 정교한 생성 능력을 보여줍니다.



### Multiview Image-Based Localization (https://arxiv.org/abs/2503.23577)
- **What's New**: 이 논문은 이미지 검색(image retrieval, IR) 접근법과 잠재적 3D 재구성을 결합한 하이브리드 접근법을 제안합니다. 이러한 방식은 데이터베이스에 이미지 피처만 저장하며, 3D 씬 재구성을 유지하지 않으면서 3D 방법의 장점을 활용합니다. 새로운 제안은 쿼리 카메라의 중심 추정을 상대적인 변환 추정에만 의존하며, 상대 회전 추정을 제거함으로써 더 정확한 로컬라이제이션을 제공합니다.

- **Technical Details**: 제안된 방법에서는 K개의 앵커 이미지로부터 쿼리 이미지의 포즈를 계산합니다. 쿼리 카메라는 각 앵커 이미지와의 상대적 포즈를 기반으로 절대 포즈를 추정하는 대신, 멀티뷰 대응관계를 통해 optimal pose를 계산합니다. 이를 통해 중복된 단계 없이 직접적인 피처를 이용한 정확한 로컬라이제이션을 실현합니다.

- **Performance Highlights**: 이 접근법은 7-장면(7-Scenes) 및 캠브리지 랜드마크(Cambridge Landmarks) 데이터셋에서 기존의 최첨단 기술에 비해 향상된 성능을 보여줍니다. 4개의 7장면에서 더 나은 로컬라이제이션을 보여주었고, 캠브리지 데이터셋에서는 모든 방법에 대한 성능 개선이 있었습니다. 또한 HLOC+SG와 비교했을 때 타이머 및 메모리 요구 측면에서 각각 5% 및 10%의 개선이 있었습니다.



### DASH: Detection and Assessment of Systematic Hallucinations of VLMs (https://arxiv.org/abs/2503.23573)
- **What's New**: 이 논문에서는 오픈 월드(open-world) 환경에서 비전-언어 모델(vision-language models, VLMs)의 체계적 환각(object hallucinations)을 탐지하고 평가하기 위한 자동화된 파이프라인인 DASH(Detection and Assessment of Systematic Hallucinations)를 제안합니다. 기존의 벤치마크가 작고 레이블이 있는 데이터셋에 의존하여 제한된 결과를 제공한 반면, DASH는 자연 사진 매니폴드(natural image manifold)를 최적화하여 VLM을 혼란스럽게 하는 이미지를 생성하고 이를 통해 환각 물체들을 식별합니다.

- **Technical Details**: DASH는 두 가지 주요 구성 요소인 DASH-OPT와 DASH-LLM로 구성됩니다. DASH-OPT는 이미지 기반의 검색 방식을 통해 VLM이 존재하지 않는 물체를 환각하게 만드는 이미지를 생성하도록 최적화됩니다. 반면, DASH-LLM은 대규모 언어 모델(LLM)에서 생성된 쿼리를 바탕으로 기능하며, 이 두 가지 접근 방식을 통해 이미지와 텍스트 쿼리를 탐색하여 FP-hallucinations를 유발하는 실체 이미지를 찾습니다.

- **Performance Highlights**: DASH를 PaliGemma와 여러 LLaVA-NeXT 모델에 적용한 결과, 19,000개 이상의 클러스터와 950,000개 이상의 이미지를 발견했습니다. 찾아낸 환각 이미지는 다른 VLM으로 성공적으로 이전되며, DASH를 통해 생성된 특정 이미지로 PaliGemma를 미세 조정하면 객체 환각 문제를 완화할 수 있음을 보입니다. 또한, DASH-B라는 새로운 벤치마크를 제안하여 현재 VLMs의 평가를 보다 신뢰할 수 있게 할 수 있음을 보여주고 있습니다.



### Enhancing Creative Generation on Stable Diffusion-based Models (https://arxiv.org/abs/2503.23538)
Comments:
          CVPR 2025 accepted paper

- **What's New**: 이 논문에서는 C3 (Creative Concept Catalyst)라는 새로운 접근 방식을 소개합니다. C3는 Stable Diffusion 모델 기반에서 창의성을 향상시키기 위해 훈련이 필요하지 않은 방법입니다. 기존 모델들이 `creative`라는 프롬프트를 포함시킬 때 원하는 결과를 얻기 어려운 점을 개선하고자 하였습니다.

- **Technical Details**: C3는 노이즈 제거 과정에서 선택적으로 특징(feature)을 증폭(amplify)하여 더 창의적인 출력을 유도합니다. 창의성에 대한 두 가지 주된 측면에 따라 증폭 계수를 선택하는 실용적인 가이드라인도 제공합니다. 이 방법은 방대한 계산 비용 없이 확산(difussion) 모델에서 창의성을 증진시키는 첫 번째 연구입니다.

- **Performance Highlights**: 우리는 C3의 효과를 다양한 Stable Diffusion 기반 모델에서 입증하였습니다. 이를 통해 사용자는 더 높은 창의성을 가진 이미지를 생성할 수 있으며, 모델의 생산성을 메리트를 증가시킬 수 있습니다. C3의 도입으로 해당 모델들의 활용 가능성이 더욱 확장될 것으로 기대합니다.



### BiPVL-Seg: Bidirectional Progressive Vision-Language Fusion with Global-Local Alignment for Medical Image Segmentation (https://arxiv.org/abs/2503.23534)
- **What's New**: BiPVL-Seg는 의료 이미지 세분화에서 비전(vision)과 언어(language)의 융합 및 임베딩(embedding) 정렬을 통합하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 전체적인 전략과 훈련 혁신을 통해 두 구성 요소가 서로 강화되어 세분화 성능을 향상시킵니다. 또한, BiPVL-Seg는 비가역적 진행 융합(bidirectional progressive fusion) 아키텍처를 도입하여 비전 및 텍스트 인코더 간의 단계별 정보 교환을 촉진합니다.

- **Technical Details**: 기존의 VLM(Vision-Language Model) 구조들은 비전 인코더에서 독립적으로 처리를 하여 크로스 모달(Cross-modal) 정렬을 약하게 만들었습니다. BiPVL-Seg는 이러한 약점을 극복하기 위해 비가역적 진행 융합을 통해 인코더의 모든 단계에서 지속적인 정보 교환을 실현합니다. 또한, 글로벌-로컬 대비 정렬(global-local contrastive alignment) 방식을 도입하여 의료용 텍스트와 비전 임베딩 간의 의미 있는 정렬을 가능하게 합니다.

- **Performance Highlights**: 다양한 의료 이미지 벤치마크(CT 및 MR 모드 포함)에서 실시된 광범위한 실험을 통해 BiPVL-Seg의 성능이 최첨단 방법들보다 우수한 것으로 나타났습니다. 이는 복잡한 다중 클래스 세분화에서 특히 두드러지며, 체계화된 해부학적 구조 및 종양 이미징에 이르기까지 폭넓은 성능을 겸비했습니다. 소스 코드는 GitHub 저장소에서 확인할 수 있습니다.



### ViLAaD: Enhancing "Attracting and Dispersing'' Source-Free Domain Adaptation with Vision-and-Language Mod (https://arxiv.org/abs/2503.23529)
Comments:
          15 pages

- **What's New**: 본 논문에서는 비전-언어 모델( Vision-and-Language models, ViL)을 활용하여 기존의 소스 데이터 없이 도메인 적응(Source-Free Domain Adaptation, SFDA) 방법을 확장한 새로운 접근 방식을 제안합니다. 구체적으로, 저자들은 Attracting and Dispersing(AaD) 방법을 기반으로 하여 ViL 모델을 통합하는 ViL-enhanced AaD (ViLAaD)를 소개합니다. ViLAaD는 AaD 프레임워크의 간단함과 유연성을 유지하면서도 ViL 모델을 활용하여 더 나은 적응 성능을 보여줍니다.

- **Technical Details**: ViLAaD 접근법은 소스 모델과 ViL 모델이 목표 데이터에 대해 적절한 초기값을 제공할 때, 목표 모델의 예측이 가까운 샘플들의 ViL 예측과 더 잘 일치하도록 하는 것을 기본 원리로 합니다. 이 방법은 실험을 통해 다섯 개의 ViL 모델(CIP, ALBEF, BLIP 등)을 사용하여 검증되었으며, 결과는 ViLAaD가 AaD 및 ViL 모델의 제로샷 분류보다 우수한 성능을 발휘함을 나타냅니다. 또한 ViLAaD는 ViL 프로프트 튜닝과 함께 대체 최적화(framework)적인 접근법으로 통합될 수 있어 유연성과 확장성을 제공합니다.

- **Performance Highlights**: ViLAaD++는 네 가지 SFDA 벤치마크에서 검증을 거쳐 여러 SFDA 시나리오에서 최첨단 성능을 달성했습니다. 실험 결과에 따르면, ViLAaD++는 Closed-set SFDA, Partial-set SFDA, Open-set SFDA 등 다양한 경우에서 최고의 성능을 보였습니다. 이러한 결과는 ViL 모델을 활용한 새로운 SFDA 방법론의 효과를 입증하며, 향후 연구에서의 가능성을 보여줍니다.



### BoundMatch: Boundary detection applied to semi-supervised segmentation for urban-driving scenes (https://arxiv.org/abs/2503.23519)
Comments:
          15 pages, 7 figures

- **What's New**: 본 논문에서는 BoundMatch라는 새로운 반자동 의미 세분화 프레임워크를 제안합니다. 이 프레임워크는 세분화 마스크와 의미 경계를 모두 일관성 정규화(consistency regularization)하는 방식을 도입하여, 객체 경계의 정확한 delineation을 달성합니다. 연구팀은 이 모델이 여러 데이터 세트에서 경쟁력 있는 성능을 발휘하며 객체 경계 관련 평가 지표를 크게 향상시킨다고 주장합니다.

- **Technical Details**: BoundMatch의 핵심 메커니즘은 Boundary Consistency Regularized Multi-Task Learning (BCRM)으로, 세분화 마스크와 계층적 기능에서 추출된 의미 경계 맵에 대해 예측 일치를 강요합니다. 또한, Boundary-Semantic Fusion (BSF) 모듈과 Spatial Gradient Fusion (SGF) 모듈을 사용하여 경계 정보를 세분화 디코더에 주입하고 마스크 기울기로 경계 예측을 조정하여 경계 품질을 높입니다. 이 프레임워크는 SAMTH라는 강력한 기반 모델을 활용하여 안정성을 향상시킵니다.

- **Performance Highlights**: 다양한 데이터 세트(Cityscapes, BDD100K, SYNTHIA, ADE20K 및 Pascal VOC)에 대한 광범위한 실험을 통해 BoundMatch는 기존 최첨단 방법과 비교하여 경쟁력 있는 성능을 발휘하며 경계 관련 지표에서 유의미한 개선을 보여줍니다. 또한, 대규모 비지도 데이터 시나리오와 모바일 배치에 적합한 경량 구조에서도 효과적임을 입증했습니다. 본 연구는 경량 모델에서도 강력한 성능을 유지하면서도 실제 상황에서 적용 가능성을 높였습니다.



### ReferDINO-Plus: 2nd Solution for 4th PVUW MeViS Challenge at CVPR 2025 (https://arxiv.org/abs/2503.23509)
- **What's New**: 본 논문은 ReferDINO-Plus라는 새로운 접근 방식을 제안합니다. 이 방법은 RVOS(Referring Video Object Segmentation)에서 이전 작업의 한계를 극복하고자 합니다. ReferDINO-Plus는 SAM2와 ReferDINO의 장점을 통합하여 마스크 품질과 객체 일관성을 향상시킵니다.

- **Technical Details**: ReferDINO-Plus는 두 단계의 전략을 기반으로 하며, 첫 번째 단계에서는 ReferDINO를 사용하여 객체 풀이 및 시공간 밀집 추론을 수행합니다. 두 번째 단계에서는 SAM2를 적용하여 마스크를 정제하고 증강하며, 조건부 마스크 융합(Conditional Mask Fusion) 전략을 사용하여 최종 마스크를 생성합니다.

- **Performance Highlights**: ReferDINO-Plus는 MeViS 테스트 세트에서 60.43의 J&F 점수를 달성하며, CVPR 2025 MeViS PVUW 챌린지에서 2위에 올라서게 되었습니다. 이는 RVOS 작업을 위한 최신 기준을 제시하며, 실용화 가능성이 높은 성능을 보여줍니다.



### Re-Aligning Language to Visual Objects with an Agentic Workflow (https://arxiv.org/abs/2503.23508)
Comments:
          33 pages, 20 figures, 17 tables, ICLR 2025

- **What's New**: 이번 연구에서는 VLM(vision-language models)으로부터 발생하는 비정확한 언어 표현을 보완하기 위해 LLM(large language model)에 의해 제어되는 agentic workflow(대리인적 작업 흐름)인 Real-LOD를 제안합니다. 이 workflow는 이미지와 텍스트 프롬프트를 유동적으로 조정함으로써 시각적 객체와 언어 간의 재조정을 수행합니다. 최종적으로 0.18M 이미지로 구성된 데이터셋을 통해 기존 LOD 방법 대비 약 50%의 성능 향상을 보여줍니다.

- **Technical Details**: Real-LOD는 계획(plan), 도구 사용(tool use), 반성(reflection) 단계로 구성되며, 이미지와 초기 언어 표현을 입력으로부터 감지된 객체와 함께 처리합니다. 이 시스템은 현재 상태를 자동으로 고려하고, 계획된 행동에 따라 이미지와 텍스트 프롬프트를 조정하여 VLM에 재설명 요청을 전달합니다. 각 단계는 반복적으로 진행되어 언어 표현의 품질을 서서히 개선하며, LOD 모델 훈련을 위한 언어-객체 데이터 쌍을 정제합니다.

- **Performance Highlights**: 이번 연구는 Real-LOD를 통해 데이터 양과 품질을 동시에 개선하며, LOD 성능 향상에 기여하였습니다. 학습된 모델은 Swin-B 백본 구조를 채택하고 있으며, 표준 벤치마크에서 타 방법 대비 성능을 약 50% 향상시켰습니다. 이는 LOD 훈련에 있어 데이터의 양뿐만 아니라 품질의 중요성을 실증적으로 보여주고 있습니다.



### Federated Self-Supervised Learning for One-Shot Cross-Modal and Cross-Imaging Technique Segmentation (https://arxiv.org/abs/2503.23507)
- **What's New**: 이 논문은 분산된 연합 학습(Decentralized Federated Learning)에서의 자기 지도 학습(self-supervised learning) 기반의 일회성 세분화(one-shot segmentation) 작업을 처음으로 시도하는 연구입니다. 연구는 의료 영상 처리 분야의 데이터 부족 문제를 해결하기 위한 방법으로, 다양한 모델을 통해 여러 출처에서 데이터 표현을 학습할 수 있는 가능성을 탐색합니다. 또한 CoWPro라는 기존의 자기 지도 몇 샷(segmentation few-shot) 세분화 프레임워크를 연합 학습 시나리오에 맞게 조정하여 성능을 향상시킵니다.

- **Technical Details**: 이 연구에서는 프레임워크의 성능을 증대하기 위해 융합 다이스 손실(fused dice loss)을 도입하였습니다. 이를 통해 CoWPro의 기본 성능보다 더 나은 성능을 달성할 수 있었습니다. 연구는 서로 다른 모달리티(modality) 및 이미징 기법을 가진 클라이언트들로부터의 데이터를 구성하는 방식으로 세분화 문제를 더욱 어렵게 만듭니다. 또한, 제안된 프레임워크는 로컬 클라이언트 데이터셋의 전혀 보지 못한 부분에 대해 효과적으로 평가됩니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 CoWPro의 FedAvg 버전에 비해 동등한 또는 더 나은 성능을 보였습니다. 특히 자원을 절약할 수 있는 사전 훈련(all-in one pre-training) 방식 덕분에 다운스트림(task downstream)의 세부 조정이 필요 없어 컴퓨팅 자원의 효율성을 높였습니다. 이 논문에서 제안한 새로운 의료 이미징 데이터셋은 총 95명의 환자로부터 수집된 MRI 스캔을 포함하여 기존의 데이터셋보다 유용성을 강화하며 연구의 투명성을 높입니다.



### Boosting Omnidirectional Stereo Matching with a Pre-trained Depth Foundation Mod (https://arxiv.org/abs/2503.23502)
Comments:
          Project page: this https URL

- **What's New**: DFI-OmniStereo는 새로운 옴니디렉셔널 스테레오 매칭 방법으로, 대규모로 사전 학습된 단안 깊이 모델을 활용하여 상대 단안 깊이 추정을 수행합니다. 이 방법은 반복 최적화 기반의 스테레오 매칭 아키텍처에 통합되어 있으며, 데이터 효율성과 일반화 성능을 향상시키기 위해 전용의 두 단계 훈련 전략을 도입합니다. DFI-OmniStereo는 Helvipad 데이터셋에서 기존 방법보다 약 16% 낮은 disparity MAE를 기록하며 최첨단 성과를 달성했다고 발표합니다.

- **Technical Details**: DFI-OmniStereo는 옴니디렉셔널 스테레오 이미지 쌍을 처리하기 위해 설계된 엔드 투 엔드 모델입니다. 이 모델은 첫 번째 단계에서 스테레오 매칭 헤드가 새로운 특성 공간에 적응할 수 있도록 훈련되며, 두 번째 단계에서는 일반화 성능을 유지하면서 스케일 불변 손실(scale-invariant loss)을 이용해 백엔드의 디코더를 미세 조정합니다. 이러한 과정은 반복 최적화를 통해 진행되며, 다양한 환경에서의 깊이 추정 정확도를 높입니다.

- **Performance Highlights**: DFI-OmniStereo는 Helvipad 데이터셋에서의 테스트 결과, 기존의 옴니디렉셔널 스테레오 매칭 방법들에 비해 성과가 뛰어난 것으로 나타났습니다. 이 방법은 샘플 효율성을 높이고 다른 데이터셋에 대한 일반화 능력 또한 보유하고 있습니다. 이렇게 하여 DFI-OmniStereo는 다양한 조명 조건과 깊이 범위를 갖는 여러 환경에서 깊이 정확도를 개선하는 데 기여합니다.



### Embedding Shift Dissection on CLIP: Effects of Augmentations on VLM's Representation Learning (https://arxiv.org/abs/2503.23495)
Comments:
          accepted at MIV at CVPR 2025

- **What's New**: 이번 연구에서는 CLIP와 같은 Vision Language Models(VLM)가 특정한 이미지 증강(augmentation) 기법에 따라 어떻게 임베딩(embedding)이 변하는지를 분석하였습니다. 9가지 보편적인 증강 기술(예: noise, blur, color jitter 등)을 사용하여 나타나는 변화를 정량적 및 정성적으로 분석했습니다. 연구 결과, noise와 perspective transform이 임베딩 변화에 더 큰 영향을 미친다는 사실을 밝혀냈으며, 이는 기계적 해석에 있어서 중요한 기초자료가 됩니다.

- **Technical Details**: 이 논문에서는 Conceptual Captions 데이터셋에서 13,312개의 이미지를 선택해 9가지 증강 기법을 실시하였습니다. 각 증강 기법은 albumentations 라이브러리를 이용하여 구현하였고, 수치적 분석을 제공하기 위해 L2 거리, 코사인 유사성(cosine similarity) 등 여러 통계 지표를 활용했습니다. 이를 통해 CLIP의 세부 사항 보존(detail preservation) 및 유사성 평가(similarity score) 등을 심층적으로 분석했습니다.

- **Performance Highlights**: 연구 결과, noise 변환이 다른 증강 기법보다 CLIP의 주의(attention) 분포에 가장 큰 영향을 미쳤으며, 이는 다른 기술들보다 세부 정보를 잘 보존하지 못하는 것으로 나타났습니다. 특정 증강 기법이 CLIP의 임베딩 표현 방식에 미치는 영향이 확인되었으며, 이를 통해 향후 연구의 방향성을 제시하고 있습니다. 그러나 CLIP의 해석 가능성(mechanistic interpretability)을 분석하는 데 있어서는 여전히 많은 연구가 필요합니다.



### Efficient Dynamic Attention 3D Convolution for Hyperspectral Image Classification (https://arxiv.org/abs/2503.23472)
- **What's New**: 이 논문은 개선된 3D-DenseNet 모델을 기반으로 한 동적 주의(convolution) 설계를 제안하여, 과적합(overfitting)을 방지하고 깊이를 증가시키면서 발생하는 기울기 소실(gradient vanishing) 문제를 완화하는 방법을 다룹니다. 이 새로운 설계는 여러 개의 병렬 convolutional kernel을 사용하고, 각 kernel에 동적 주의 가중치를 할당하여 풍부한 공간-스펙트럼 정보 활용을 목표로 합니다.

- **Technical Details**: 제안된 다이나믹 어텐션 컨볼루션(DAC) 모듈은 하이퍼스펙트럴 이미지의 공간적 특성을 고려하여 키 공간 구조에 더 중점을 둔다고 합니다. 또한 스펙트럴 차원에서는 서로 다른 밴드의 동적 판별(dynamical discrimination)을 가능하게 하여 정보의 중복성과 계산 복잡성을 완화할 수 있습니다. 이렇게 함으로써 네트워크의 깊이와 너비를 증가시키지 않으면서도 모델의 표현 능력을 향상시킵니다.

- **Performance Highlights**: 제안된 방법은 IN, UP, KSC 데이터셋에서 주류 하이퍼스펙트럴 이미지 분류 방법들을 초과한 성능을 시연합니다. 이를 통해 추론 속도(inference speed)와 정확도(accuracy) 모두에서 우수한 결과를 도출하였으며, 기존 방법과 비교했을 때 효과적인 피처 추출을 이루어 내었습니다.



### Internal Organ Localization Using Depth Images (https://arxiv.org/abs/2503.23468)
Comments:
          Accepted for German Conference on Medical Image Computing 2025 (BVM 2025)

- **What's New**: 이번 연구는 MRI 작업 흐름에서 환자 위치 자동화를 위한 RGB-D 카메라 기반 시스템의 가능성을 탐구합니다. 기존의 방법과 달리, 우리의 접근 방식은 MRI 스캔 데이터셋을 활용하여 심층 학습 모델을 학습시키고 깊이 이미지만으로 내부 장기의 위치를 추정하는데 초점을 맞춥니다. 연구 결과, 이 방법이 환자 스캐닝 절차를 간소화하고 환자 경험을 개선할 가능성을 보여주었습니다.

- **Technical Details**: 연구에 사용된 데이터셋은 약 10,000개의 전체 신체 MRI 스캔을 포함하며, 이를 통해 본 연구의 모델은 다양한 신체 유형과 해부학적 변이를 고려할 수 있습니다. U-Net 기반 아키텍처를 채택하여 깊이 이미지를 입력으로 받고 내부 장기의 분할 마스크를 출력합니다. 모델 훈련은 다중 레이블 설정으로 진행되며, Dice 손실과 이진 교차 엔트로피 손실을 가중 조합하여 사용합니다.

- **Performance Highlights**: 모델의 성능은 수동으로 라벨링된 데이터와 자동 생성된 데이터 두 가지로 평가하였으며, 양쪽 데이터셋에서 긍정적인 결과를 보였습니다. 연구 결과에 따르면, 모델은 대부분의 구조물에 대해 95백분위수가 30mm 이하의 Detection Offset Error를 기록하며, 특히 소프트 조직 장기에 대한 형태 추정에서 뛰어난 성능을 나타내었습니다. 결론적으로, 깊이 정보가 내부 장기의 해부학적 구조를 이해하는 데 중요한 역할을 할 수 있음을 보여주었습니다.



### OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Mod (https://arxiv.org/abs/2503.23463)
- **What's New**: OpenDriveVLA는 3D 환경 인식, 자율 차량 상태, 운전자의 명령을 기반으로 안정적인 주행 동작을 생성하도록 설계된 비전-언어 행동(VLA) 모델입니다. 이 모델은 개방형 소스의 사전 훈련된 대형 비전-언어 모델(VLM)을 활용하여 주행 경로를 예측합니다. 특히 OpenDriveVLA는 비전-언어 정렬 과정을 통해 동적이고 복잡한 주행 환경에서의 추론 능력을 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: OpenDriveVLA는 3D와 2D 시각적 토큰을 통합된 의미 공간으로 맞추기 위해 계층적 비전-언어 정렬 모듈을 도입합니다. 이 모델은 에고 차량, 주변 에이전트 및 정적인 도로 요소 간의 동적 관계를 모델링하여 자율 주행의 궤도 계획을 보장합니다. 결과적으로 OpenDriveVLA는 현실 세계의 다양한 주행 조건에 대한 확실한 이해를 기반으로 안정적인 경로 생성을 가능하게 합니다.

- **Performance Highlights**: nuScenes 데이터셋에 대한 광범위한 실험 결과, OpenDriveVLA는 오픈 루프 경로 계획 및 주행 관련 질의 응답 작업에서 최신 기술의 성능 기준을 설정하였습니다. 모델은 높은 수준의 주행 명령을 따르고 복잡한 시나리오에서도 경로를 견고하게 생성하는 능력을 입증하여 차세대 자율 주행 기술의 잠재력을 보여줍니다. 이러한 성능은 OpenDriveVLA의 효과적인 주행 경로 생성 및 주행 장면 이해 능력을 강조합니다.



### TextCrafter: Accurately Rendering Multiple Texts in Complex Visual Scenes (https://arxiv.org/abs/2503.23461)
- **What's New**: 이 논문은 Complex Visual Text Generation (CVTG) 작업을 탐구하며, 이는 시각 이미지 내 여러 영역에 분산된 복잡한 텍스트 내용을 생성하는 데 중점을 둡니다. 기존의 이미지 생성 모델은 왜곡된 시각적 텍스트를 생성하거나 일부 텍스트를 누락하는 문제를 겪고 있습니다. 이러한 문제를 해결하기 위해 우리는 TextCrafter라는 새로운 다중 시각 텍스트 렌더링 방법을 제안합니다. TextCrafter는 텍스트와 그 시각적 매체 간의 강력한 정렬을 보장하면서 복잡한 시각적 텍스트를 여러 구성 요소로 분해하는 점진적 전략을 사용합니다.

- **Technical Details**: TextCrafter는 훈련이 필요 없는 프레임워크로, 복잡한 시각 장면에서 여러 텍스트를 렌더링하는 데 초점을 맞추고 있습니다. 이 프레임워크는 인스턴스 융합, 영역 절연 및 텍스트 포커스라는 세 가지 주요 단계를 포함합니다. 첫 번째 단계인 인스턴스 융합에서는 텍스트와 그 환경 간의 강한 일관성을 유지하며, 두 번째 단계인 영역 절연은 사전 훈련된 DiT 모델의 위치적 선행 조건을 활용하여 텍스트 인스턴스의 레이아웃 정보를 초기화합니다. 마지막 단계인 텍스트 포커스에서는 시각 텍스트의 주목도를 증대시키는 메커니즘을 도입하여 텍스트 렌더링의 정확성을 높입니다.

- **Performance Highlights**: 우리는 CVTG-2K라는 새로운 벤치마크 데이터셋을 구축하여, 다양한 시각 텍스트 프롬프트를 평가하기 위한 강력한 기준을 제공합니다. TextCrafter의 성능은 정량적 및 정성적 실험을 통해 입증되었으며, 기존의 최첨단 방법들보다 뛰어난 효과성과 강건성을 나타냅니다. 이 결과는 복잡한 시각 텍스트 생성 작업에서 TextCrafter가 기존 방법을 초월하는 성능을 발휘함을 보여줍니다.



### Reinforcement Learning-based Token Pruning in Vision Transformers: A Markov Game Approach (https://arxiv.org/abs/2503.23459)
Comments:
          Accepted by IEEE International Conference on Multimedia & Expo (ICME) 2025

- **What's New**: 새로운 연구에서는 Vision Transformers (ViTs)의 토큰 가지치기(Pruning) 정책을 데이터를 적응적으로 배우기 위해 Reinforcement Learning (RL)을 활용하였습니다. 기존의 수작업으로 정의된 정책들을 넘어, 이 연구는 여러 레이어에 걸쳐 순차적인 결정 문제로 포맷하여 토큰 가지치기를 최적화합니다. Multi-Agent Proximal Policy Optimization (MAPPO)를 채택하여 각 토큰에 대해 개별적인 결정을 내릴 수 있게 하여 효율성과 정확성을 효과적으로 균형 잡는 보상 함수를 개발했습니다.

- **Technical Details**: RL4EViT 방법론은 ViT 아키텍처에 MAPPO 기반 가지치기 레이어를 통합하여冀 Markov Game 설정을 통해 중복 토큰을 가지치기 합니다. 입력으로는 이미지 데이터셋과 ViT 구성 요소, RL 구성 요소를 받아들이며, 이들 간의 효율적인 상호작용을 통해 학습합니다. 알고리즘에서 forward 및 backward 패스가 따로 정의되며, 먼저 사전 학습된 ViT를 기반으로 RL 구성 요소를 최적화하고, 필요 시 ViT 구성 요소를 미세 조정합니다.

- **Performance Highlights**: 이 방법을 통해 ImageNet-1k 데이터셋에서 최대 44%의 추론 속도 향상과 함께 정확도가 단 0.4% 감소하는 미미한 손실을 기록했습니다. 이는 기존 방법들과 비교했을 때 최고의 효율성과 정확성의 Trade-off를 이끌어내며, RL을 적용하여 데이터 적응성이 높은 가지치기 정책을 획기적으로 제안했습니다. 최종적으로, 이 연구는 토큰 가지치기에서 RL의 첫 적용으로서 중요한 기여를 하였습니다.



### CADFormer: Fine-Grained Cross-modal Alignment and Decoding Transformer for Referring Remote Sensing Image Segmentation (https://arxiv.org/abs/2503.23456)
- **What's New**: 본 논문에서는 고해상도 원격 탐지 이미지에서의 특정 목표 객체 분할을 위한 새로운 접근 방식인 CADFormer를 제안합니다. 기존의 참조 원격 탐지 이미지 분할(Referring Remote Sensing Image Segmentation, RRSIS) 방법들은 언어 특징의 맥락 정보를 미흡하게 활용하여 결과가 불완전하거나 잘못된 마스크를 생성하는 문제가 있었습니다. CADFormer는 세분화된 교차 모달 정렬과 디코딩 프로세스를 통해 이러한 문제를 해결하고, 복잡한 언어 표현 및 장면을 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: CADFormer는 구체적으로 언어와 비주얼 간의 상호 의존성을 모델링하여 세분화된 교차 모달 정렬을 달성하는 의미 상호 안내 정렬 모듈(Semantic Mutual Guidance Alignment Module, SMGAM)을 포함하고 있습니다. 이 모듈은 비전에서 언어, 언어에서 비전으로의 정렬을 모두 수행하여 균형 잡힌 다중 시각 특성을 쉽게 생성합니다. 또한, 텍스트를 강화한 교차 모달 디코더(Textual-enhanced Cross-Modal Decoder, TCMD)를 도입하여 디코딩 과정에서 언어 특징을 활용하고, 이를 통해 예측된 세분화 마스크의 품질을 향상시킵니다.

- **Performance Highlights**: CADFormer의 성능은 RRSIS-HR 및 RRSIS-D 데이터셋에서 철저히 평가되었으며, 기존 RRSIS 방법의 대부분을 능가하는 결과를 보여주었습니다. 새로운 RRSIS-HR 데이터셋은 고해상도의 복잡한 장면을 포함하여, RRSIS 방법이 도전적인 작업을 수행하는 데 필요한 기반을 제공합니다. 실험 결과는 CADFormer가 세밀한 언어 표현을 포함한 고해상도 이미지를 효율적으로 처리할 수 있음을 확인시켜 주었습니다.



### Efficient Token Compression for Vision Transformer with Spatial Information Preserved (https://arxiv.org/abs/2503.23455)
Comments:
          accepted by IEEE Transactions on Multimedia

- **What's New**: 비전 트랜스포머(Vision Transformer, ViT)의 복잡성을 줄이고 컴퓨팅 자원의 효율성을 높이기 위해, 본 논문에서는 'Prune and Merge'라는 새로운 토큰 압축 방법을 제안합니다. 이 방법은 토큰을 전처리하고 병합하는 기능을 통합하여 모델의 레이어별 압축을 가능하게 하며, 훈련 가능한 매개변수를 통해 중요한 정보를 보존하면서 압축을 수행합니다. 특히, 여러 하드웨어 환경과의 호환성을 고려하여 설계되었습니다.

- **Technical Details**: 제안된 'Prune and Merge' 모듈은 학습 가능한 병합 행렬과 재구성 행렬을 도입하여 동시에 토큰의 병합과 가지치기를 수행합니다. 또한, 기울기 가중치 주의(attention) 점수 계산 메커니즘을 통해 훈련 중에 토큰의 중요도를 평가하여, 별도의 계산 없이 압축 효율성을 극대화합니다. 이렇게 분류된 토큰을 바탕으로 효율적인 토큰 병합 및 가지치기를 수행하는 방법론이 핵심입니다.

- **Performance Highlights**: 다양한 데이터셋에 대한 실험 결과, 제안된 방법은 기존의 최첨단 기법들과 비교하여 최소한의 정확도 감소로 상당한 속도 향상을 달성했습니다. 예를 들어, DeiT-Small 모델에서는 ImageNet-1k 데이터셋에서 1.64배의 속도 향상과 함께 0.2%의 정확도 감소를 기록했습니다. 이를 통해 통합된 방법론의 효율성과 효과성이 입증되었습니다.



### Semantic-Spatial Feature Fusion with Dynamic Graph Refinement for Remote Sensing Image Captioning (https://arxiv.org/abs/2503.23453)
- **What's New**: 이 논문에서는 원거리 감지 이미지 자막 생성의 새로운 접근 방식인 'semantic-spatial feature fusion with dynamic graph refinement (SFDR)' 방법을 제안합니다. 이 방법은 시맨틱-스페이셜 피처 융합(SSFF)과 다이나믹 그래프 피처 리파인먼트(DGFR) 모듈로 구성되어 있습니다. 기존 방법들과 달리, 이 연구는 텍스트 정보의 역할을 강조하며, 원거리 감지 이미지의 시각적 특징과 문맥에 가장 연관성 높은 객체의 위치를 정밀하게 파악하는 도전에 대응하고 있습니다.

- **Technical Details**: 이 논문에서는 사전 훈련된 CLIP 모델을 사용하여 시각적 및 텍스트 피처를 추출하고, 이를 통해 멀티모달(feature alignment) 피처 정렬을 수행하는 새로운 구조를 소개합니다. SSFF 모듈은 멀티-레벨 피처 표현 전략을 통해 풍부한 시맨틱 및 공간 정보를 통합하며, DGFR 모듈에서는 그래프 어텐션 네트워크(GAT)를 활용하여 객체 피처 간의 관계를 포착합니다. 다이나믹 가중치 메커니즘은 현재 장면과 가장 관련이 깊은 객체를 우선시하고 덜 중요한 요소를 억제하는 방식으로 작동합니다.

- **Performance Highlights**: 제안된 SFDR 방법은 세 가지 벤치마크 데이터 세트에서 실험을 수행하여 기존 방법들과 비교해 뛰어난 효과를 입증했습니다. 이 연구는 원거리 감지 데이터에서 의미가 풍부하고 사전 정보에 기반한 설명 생성을 가능하게 하며, 이러한 접근법이 다양한 도시 계획, 군사 분석 및 환경 모니터링 분야에 적용 가능함을 보여줍니다. 코드와 구현은 제공된 URL에서 확인할 수 있습니다.



### VideoGen-Eval: Agent-based System for Video Generation Evaluation (https://arxiv.org/abs/2503.23452)
Comments:
          project:this https URL

- **What's New**: 본 논문에서는 비디오 생성 평가의 한계를 해결하기 위해 VideoGen-Eval이라는 에이전트 기반 평가 시스템을 제안합니다. 이 시스템은 LLM(대형 언어 모델) 기반 콘텐츠 구조화, MLLM(다중 모달 대형 언어 모델) 기반 콘텐츠 판단, 및 시간 밀도가 높은 차원 평가를 위한 패치 도구를 통합하여 평가의 동적, 유연성 및 확장성을 실현합니다. 또한, 기존의 최첨단 모델을 평가하기 위한 비디오 생성 벤치마크를 소개하며, 700개의 구조화된 프롬프트와 12,000개 이상의 생성된 비디오로 구성되어 있습니다.

- **Technical Details**: 제안된 시스템은 미디어 생성의 다양한 차원을 평가하기 위해 3개의 주요 구성 요소로 이루어져 있습니다. 첫 번째는 LLM을 사용하여 콘텐츠를 구조화하고, 두 번째는 MLLM을 통해 생성된 콘텐츠의 정확성을 판단하는 것입니다. 마지막으로, 시간 밀도가 높은 차원을 평가하기 위한 여러 도구를 포함한 패치 도구들이 동적으로 사용되어 평가의 신뢰성을 증가시킵니다. 이를 통해 시스템은 입력 프롬프트에 따라 관련 평가 측면을 결정할 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, VideoGen-Eval은 기존의 정적인 평가 방법보다 인간의 선호도와 더 높은 일치를 보이며, 비디오 생성 평가의 신뢰성을 높이는 데 기여합니다. 이 시스템은 최신 8개 모델을 대상으로 한 평가에서도 성능을 발휘하였고, 다양한 모델의 생성 패턴을 관찰할 수 있는 풍부한 샘플을 제공합니다. 따라서, 비디오 생성의 복잡한 평가 요구에 더욱 적합한 대안으로 자리잡을 것으로 기대됩니다.



### Beyond Academic Benchmarks: Critical Analysis and Best Practices for Visual Industrial Anomaly Detection (https://arxiv.org/abs/2503.23451)
- **What's New**: 이 논문은 산업 현장에서의 비주얼 이상 탐지(Anomaly Detection, AD)의 중요성과 현재 연구의 한계를 드러내고 있습니다. 특히, 실제 생산 데이터를 사용한 새로운 벤치마크를 수립함으로써, 기존에 제안된 방법들이 실제 적용에서 어떻게 성능을 발휘하는지 비교할 수 있도록 하였습니다. 이를 통해 학계와 산업 사이의 간극을 메우기 위한 새로운 시각을 제공하고 있습니다.

- **Technical Details**: 제안된 접근법은 네 가지 실제 세계 데이터셋에서 실험을 수행하여 최근의 어려움을 평가하는 데 중점을 두고 있습니다. 이 논문은 학술적 이론이 산업적 요구 사항과 얼마나 괴리가 있는지를 분석하며, 기존의 메트릭들이 실제 상황에서의 비용을 반영하지 않음을 강조합니다. 많은 최근 연구 결과가 AUROC와 같은 이미지 수준 메트릭에 의존하고 있지만, 그들 사이의 상대적 중요성을 고려할 필요가 있습니다.

- **Performance Highlights**: 다양한 실험을 통해 연구진은 입력 크기 변화, 분포 변화에 대한 모델의 강건성, 노이즈 데이터에서의 성능 등을 평가하였습니다. 이 벤치마크는 보다 신뢰할 수 있는 비교를 위해 표준화된 평가 프로토콜을 구현했으며, 주요 산업의 요구를 충족시키기 위한 여러 가지 새로운 메트릭도 소개하였습니다. 궁극적으로, 이 논문은 학계에서 산업으로의 한계를 뛰어넘기 위한 실질적인 방법론을 제시하고 있습니다.



### AU-TTT: Vision Test-Time Training model for Facial Action Unit Detection (https://arxiv.org/abs/2503.23450)
- **What's New**: 이 논문은 Facial Action Units (AUs) 탐지를 위한 새로운 방법인 AU-TTT를 제안합니다. AU-TTT는 Test-Time Training (TTT) 방식을 활용하여 AU 탐지에서의 일반화 문제를 해결하도록 설계되었습니다. 또한 이 방법은 bidirectional TTT 블록과 AU 전용 Region of Interest (RoI) 스캐닝 메커니즘을 포함하여 성능을 향상시키고, 다양한 데이터셋에서 견고성을 높이는데 기여합니다.

- **Technical Details**: AU-TTT는 TTT 레이어를 사용하여 모델의 가중치를 인코딩하며, 이는 전체 입력 시퀀스의 정보를 활용하여 강화됩니다. TTT 레이어는 기존의 self-attention을 대체하여, 각 입력 토큰에 대한 숨겨진 상태를 주기적으로 업데이트합니다. 이 과정에서 self-supervised reconstruction을 통해 각 입력의 예측 임베딩과 실제 임베딩 간의 차이를 최소화하는 손실 함수를 사용하여 학습합니다.

- **Performance Highlights**: AU-TTT는 도메인 내 및 도메인 간 시나리오에서 경쟁력 있는 성능을 보여줍니다. 기존의 AU 탐지 방법들은 제한된 데이터셋에서의 성능 저하 문제를 겪었지만, AU-TTT는 이러한 문제를 효과적으로 해결하여 더 나은 일반화 능력을 발휘합니다. 실험 결과, AU-TTT는 복잡한 얼굴 특징을 잘 포착하며, 다양한 환경에서도 더욱 강력한 성능을 발휘합니다.



### CA^2ST: Cross-Attention in Audio, Space, and Time for Holistic Video Recognition (https://arxiv.org/abs/2503.23447)
Comments:
          27 pages including appendix, TPAMI under review

- **What's New**: 이번 논문에서 제안한 Cross-Attention in Audio, Space, and Time (CA^2ST)는 비디오 인식에 있어 혁신적인 접근 방식을 보여줍니다. 특히, 기존의 모델들이 공간적(spatial) 및 시간적(temporal) 이해에서 균형을 이루지 못하는 반면, CA^2ST는 새로운 두 흐름(two-stream) 아키텍처인 Cross-Attention in Space and Time (CAST)를 통해 이 문제를 해결합니다. 이 방법은 RGB 입력만을 사용하여 비디오를 분석합니다.

- **Technical Details**: CAST는 각 레이어에서 Bottleneck Cross-Attention (B-CA)을 활용하여 공간 전문가(spatial experts)와 시간 전문가(temporal experts) 간의 정보 교환을 가능하게 합니다. 이를 통해 시너지를 내면서 예측을 수행하는 데 초점을 맞추고 있습니다. 또한, CAST를 확장하여 시청각 및 비디오 이해를 위한 Cross-Attention in Visual and Audio (CAVA)를 통합하였습니다.

- **Performance Highlights**: CA^2ST는 다양한 데이터셋인 EPIC-KITCHENS-100, Something-Something-V2, Kinetics-400에서 일관되게 좋은 성능을 보여주었습니다. 특히, CAVA는 UCF-101, VGG-Sound, KineticsSound, EPIC-SOUNDS와 같은 오디오-비주얼(action recognition) 벤치마크에서도 뛰어난 성능을 발휘하며, B-CA 모듈 내에서 여러 전문가 간의 효과적인 정보 교환을 입증합니다.



### Improving underwater semantic segmentation with underwater image quality attention and muti-scale aggregation attention (https://arxiv.org/abs/2503.23422)
Comments:
          Accepted by Pattern Analysis and Applications

- **What's New**: 이번 연구에서는 저조도 수중 이미지에서의 의미 세분화(semantic segmentation) 문제를 해결하기 위해 새로운 Transformer 기반의 프레임워크인 UnderWater SegFormer (UWSegFormer)를 제안합니다. 또한 Underwater Image Quality Attention (UIQA) 모듈과 Multi-scale Aggregation Attention (MAA) 모듈을 개발하여 수중 이미지의 품질을 향상시키고 세분화 성능을 개선하고자 하였습니다. 이 방식은 수중 환경의 이미지 품질 저하 문제를 효과적으로 해결할 수 있는 차별화된 접근입니다.

- **Technical Details**: UWSegFormer는 UIQA 모듈을 통해 수중 이미지 특성 채널의 고품질 의미 정보를 강화하고, MAA 모듈을 이용하여 다양한 스케일의 의미 정보를 집계합니다. 이러한 모듈들은 이미지의 세부 사항을 더욱 효과적으로 캡처할 수 있도록 설계되었으며, Edge Learning Loss (ELL)를 도입해 모델의 세분화 정확도를 향상시키는 추가적인 경계를 강조합니다. 전체 구조는 Transformer 기반으로 구성되어 있으며, 고해상도와 저해상도 피처 맵을 생성하여 입력 이미지를 효과적으로 나타냅니다.

- **Performance Highlights**: SUIM 및 DUT 데이터셋에서 실험을 진행한 결과, 제안된 방법은 기존의 SOTA 방법들에 비해 세분화의 완전성, 경계의 선명도, 그리고 주관적인 지각 세부사항에서 우수함을 나타냈습니다. 특히, SUIM 데이터셋에서는 82.12의 최고 mIoU, DUT 데이터셋에서는 71.41의 mIoU를 기록하였습니다. 이는 제안한 방법이 수중 조건에서도 효과적으로 작동함을 실증적으로 보여줍니다.



### GMapLatent: Geometric Mapping in Latent Spac (https://arxiv.org/abs/2503.23407)
- **What's New**: 본 연구는 GMapLatent라는 새로운 생성 모델을 소개합니다. 이 모델은 기하학적 매핑에 기반하여 데이터를 변환하고, 서로 다른 도메인의 잠재 공간을 정밀하게 정렬하는 혁신적인 접근 방식을 제공합니다. 이를 통해 모드 붕괴(mode collapse) 및 혼합(mixture) 문제를 예방하여 생성 모델의 일반화 능력을 향상시킵니다.

- **Technical Details**: GMapLatent 모형은 잠재 공간을 정형 파라미터 도메인으로 변환하기 위해 평형 중심 변환(barycenter translation), 최적 수송(optimal transport), 제약 하모닉 매핑(constrained harmonic mapping) 기법을 사용합니다. 이후 전이된 잠재 공간 간의 기하학적 등록(geometric registration)을 수행하여 각 클러스터 간의 엄격한 일치를 보장합니다. 이 과정은 변환된 잠재 공간 간의 일대일(bijective) 매핑을 실현하며, 최종적으로 정밀한 생성을 달성합니다.

- **Performance Highlights**: GMapLatent 모델은 그레이스케일 및 컬러 이미지에 대해 실험을 통해 그 효율성 및 효과성을 입증하였습니다. 기존 모델들에 비해 우수한 성능을 보였으며, 특히 모드 붕괴와 혼합 문제를 해결하는 데 있어 강력한 장점을 가지고 있습니다. 이 연구는 교차 도메인 생성의 새로운 가능성을 열어주며, 딥러닝 분야에서의 적용 가능성을 보여 줍니다.



### Diffusion Meets Few-shot Class Incremental Learning (https://arxiv.org/abs/2503.23402)
Comments:
          pre-print

- **What's New**: 이 논문에서는 적은 샘플로 새로운 클래스를 순차적으로 학습하는 few-shot class-incremental learning (FSCIL) 문제를 해결하기 위하여 Diffusion-FSCIL이라는 새로운 프레임워크를 제안합니다. 이 방식은 텍스트-이미지 확산 모델인 Stable Diffusion을 고정된 백본(backbone)으로 활용하며, 특히 대규모 사전 학습을 통해 얻은 생성 능력을 활용합니다. 기존의 FCIL 방법들이 겪는 기능 불일치(feature misalignment)와 기억 상실(catastrophic forgetting) 문제를 완화하는 데 중점을 두고 있습니다.

- **Technical Details**: Diffusion-FSCIL 방법은 다중 보완적 확산 기능(difussion features)을 추출하여 생성 편향(generative biases)을 방지하는 데 있어 미세 지원을 제공합니다. 또한, 텍스트 인코더(text encoder)를 활용하여 표현의 질을 증가시킬 수 있습니다. 제안된 방법은 모델의 학습 과정에서 기능 추출을 동시에 수행하며, 이를 통해 효율성을 극대화합니다. 마지막으로, 생성된 함수의 평균 구조는 강력한 클래스 분리를 유지하는 데 필요한 새로운 클래스 프로토타입을 학습하는 전략을 도입합니다.

- **Performance Highlights**: Diffusion-FSCIL은 CUB-200, miniImageNet, CIFAR-100 데이터셋을 기반으로 광범위한 실험을 통해 최신의 방법들을 능가하며 과거 학습된 클래스의 성능을 유지하고 새로운 클래스에 효과적으로 적응함을 보여주었습니다. 이 방법은 약 6M의 훈련 가능한 매개변수를 사용하며 빠른 훈련 시간을 자랑하여 다른 최신 모델들과 비교해도 훌륭한 결과를 도출하였습니다. 전체적으로 이 방법은 계산 비용이 낮고 학습 시간이 빨라 효율적인 학습이 가능합니다.



### A Large Scale Analysis of Gender Biases in Text-to-Image Generative Models (https://arxiv.org/abs/2503.23398)
- **What's New**: 이번 연구는 텍스트-투-이미지(T2I) 모델에서 성별 편향을 분석한 대규모 연구로, 일상적인 상황을 중심으로 진행되었습니다. 기존 연구가 직업에 대한 편향을 다루었다면, 본 연구는 일상 활동, 객체 및 맥락에서의 성별 연관성으로 분석을 확장하였습니다. 이를 위해 3,217개의 성 중립적인 프롬프트(prompt)를 생성하고, 5개의 선도적인 T2I 모델에서 각각 200개의 이미지를 생성하여 분석에 활용했습니다.

- **Technical Details**: 이미지를 생성하기 위한 프롬프트 처리는 Yi-1.5-34B 모델을 사용하여 진행되었으며, 다양한 활동(contexts), 객체(objects) 및 직업(occupations)을 위한 프롬프트가 포함되었습니다. HDBSCAN 클러스터링 알고리즘을 사용하여 프롬프트 임베딩(prompt embeddings)을 클러스터링하고, 이러한 클러스터를 LLM을 이용해 요약하는 과정을 설명하였습니다. VQAScore를 활용하여 생성된 이미지가 해당 프롬프트와 얼마나 잘 일치하는지를 측정하였으며,모델의 성능 검증을 위해 여러 LLM 모델을 비교하였습니다.

- **Performance Highlights**: 연구 결과, T2I 모델들이 전통적인 성 역할을 강화하며, 가정 역할에서의 성별 고정관념을 반영하고 여성을 재정 분야 활동에서 과소대표하는 경향을 보였습니다. 2,293,295개의 이미지를 분석한 결과, 여성은 주로 돌봄 및 인간 중심의 시나리오에서, 남성은 기술적 또는 육체적 노동 시나리오에서 더 많이 묘사되었습니다. 성별 연관성을 분석한 결과, 프롬프트의 변화에 따라 성별 편향에 미치는 영향이 크지 않음을 확인하였습니다.



### COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation (https://arxiv.org/abs/2503.23388)
Comments:
          Accepted to CVPR 2025

- **What's New**: 최근 비전-언어 모델(VLMs)인 COSMIC(클릭 지향 의미 다중 공간 통합)는 새로운 도메인에 대한 테스트 시간 적응에 있어 중요한 도전을 해결하기 위한 프레임워크입니다. COSMIC은 다중 분류에서 우수한 적응성을 발휘하며, 세 가지 주요 혁신인 이중 의미 그래프(Dual Semantics Graph)와 클리크 지도 하이퍼 클래스(Clique Guided Hyper-class)에 기반합니다. 이를 통해 혼합된 세미틱 정보를 활용하여 예측의 강인성을 개선하고 있습니다.

- **Technical Details**: COSMIC은 다중 세미틱 캐싱(multi-granular, cross-modal semantic caching)과 그래프 기반 쿼리 메커니즘을 활용하여 모델의 적응성을 증대시킵니다. 이중 의미 그래프(DSG)는 텍스트 특징, 조밀한 CLIP 및 미세 조정된 DINOv2 특징을 통합하여 보강된 의미 공간을 생성합니다. 클리크 지도 하이퍼 클래스(CGH)는 구조화된 클래스 관계를 이용하여 예측 강인성을 높인다.

- **Performance Highlights**: COSMIC은 여러 벤치마크에서 놀라운 성능을 기록하며, 특히 out-of-distribution 태스크에서 15.81%의 향상을 보였습니다. 또한 클립 RN-50을 활용한 크로스 도메인 생성에서도 5.33%의 성능 개선을 이뤘고, 코드가 공개되어 누구나 사용할 수 있습니다. 이러한 결과는 COSMIC의 혁신적인 접근 방식이 효과적임을 잘 보여줍니다.



### Enhancing Human Motion Prediction via Multi-range Decoupling Decoding with Gating-adjusting Aggregation (https://arxiv.org/abs/2503.23381)
- **What's New**: 이번 논문은 사람의 동작 예측에서 발생하는 문제를 해결하기 위해 새로운 접근법인 multi-range decoupling decoding with gating-adjusting aggregation (MD2GA)를 제안합니다. 기존의 방법들은 과거 정보와 미래 순간 간의 상관관계의 차이를 간과하여 예측 성능의 한계를 초래했습니다. MD2GA는 이러한 강한 시간적 상관관계를 활용하여 동작 표현 학습을 개선하고, 더 정교한 장기 및 단기 예측을 가능하게 합니다. 이 혁신적인 방법은 다른 동작 예측 모델에도 쉽게 통합될 수 있습니다.

- **Technical Details**: 제안된 방법은 두 단계로 구성되어 있습니다. 첫 번째 단계에서는 multi-range decoupling decoding 기법을 사용하여 공통 피처를 다양한 미래 길이로 변환하며, 각 디코더는 동작 패턴에 대한 서로 다른 통찰을 제공합니다. 두 번째 단계에서는 gating-adjusting aggregation이 입력 동작 데이터에 따라 다양한 통찰을 동적으로 결합하여 최적의 예측을 가능하게 합니다. 이는 각각의 예측 수평을 고려한 조정된 출력을 통해 이루어집니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 방법이 기존의 동작 예측 방법들과 비교하여 타당성과 우수성을 입증했습니다. 구체적으로, 다양한 시나리오와 예측 수평에서 인간의 동작을 정확하게 예측할 수 있는 능력을 보여 주었습니다. 결과적으로 MD2GA는 전통적인 프레임워크의 한계를 극복하고 HMP의 중요한 발전을 이끌어낼 수 있는 가능성을 나타냅니다.



### KernelDNA: Dynamic Kernel Sharing via Decoupled Naive Adapters (https://arxiv.org/abs/2503.23379)
- **What's New**: 본 연구는 기존 동적 합성곱(dyamic convolution) 기법의 한계를 극복하기 위한 새로운 경량 합성곱 커널 모듈 KernelDNA를 제안합니다. KernelDNA는 입력 의존적인 동적 라우팅(dynamic routing)과 사전 훈련된 정적 모듈레이션(static modulation)을 결합하여 매개변수 효율성(parameter efficiency)과 하드웨어 친화적인 추론(inference)을 보장합니다. 기존 방법들은 매개변수 증가 문제를 겪었지만, 본 연구는 계층 간 가중치 공유(cross-layer weight sharing)를 통해 이러한 문제를 해결합니다.

- **Technical Details**: KernelDNA는 입력 데이터에 따라 동적으로 커널을 조정하는 동시에 사전 훈련된 커널을 재사용하는 메커니즘을 도입합니다. 이를 통해 기존의 정적 합성곱 구조를 유지하면서도 입력에 적응한 커널 조정으로 표현력을 향상시킵니다. 연구에서는 세 가지 향상된 주의 메커니즘인 채널 주의(Channel Attention), 필터 주의(Filter Attention), 공간 주의(Spatial Attention)를 통합하여 각 합성곱 계층이 고유한 특성을 유지하도록 하였습니다.

- **Performance Highlights**: 실험 결과, KernelDNA는 기존의 동적 합성곱 방법들보다 우수한 정확도를 보여주며, 원래의 추론 속도도 거의 유지했습니다. 예를 들어, ResNet18 모델에서 KernelDNA는 1.2-5배의 매개변수 감소와 높은 정확도(74.23%)를 달성했습니다. 경량 모델에서도 뛰어난 성능을 보여주어 다양한 아키텍처에서 매개변수 효율성, 하드웨어 호환성 및 적응 성능의 새로운 조화를 이뤘습니다.



### JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization (https://arxiv.org/abs/2503.23377)
Comments:
          Work in progress. Homepage: this https URL

- **What's New**: 이 논문에서는 JavisDiT라는 새로운 Joint Audio-Video Diffusion Transformer를 소개합니다. 이 모델은 개방형 사용자 프롬프트로부터 고품질의 오디오 및 비디오 콘텐츠를 동시에 생성할 수 있도록 설계되었습니다. 특히, 시각적 및 청각적 요소 간의 동기화를 보장하기 위해 Hierarchical Spatial-Temporal Synchronized Prior (HiST-Sypo) Estimator라는 세밀한 시공간 정렬 메커니즘을 도입했습니다.

- **Technical Details**: JavisDiT는 Diffusion Transformer (DiT) 아키텍처를 기반으로 하며, 오디오 및 비디오 채널이 AV-DiT 블록을 공유하여 고품질의 두 가지 모달리티를 생성합니다. 이 시스템은 Spatio-Temporal Self-Attention, Coarse-Grained Cross-Attention, Fine-Grained Spatio-Temporal Self-Attention Cross-Attention의 세 가지 구조적 블록을 설계하여 구현됩니다. 특히, HiST-Sypo Estimator는 입력 조건 프롬프트에서 글로벌 및 세밀한 시공간 정보를 추출하여 두 채널 간의 동기화를 강화합니다.

- **Performance Highlights**: JavisDiT는 10,140개의 고품질 텍스트 캡션이 있는 비디오로 구성된 새로운 벤치마크인 JavisBench를 통해 기기에서 성능을 입증했습니다. 실험 결과는 JavisDiT가 기존 방법들보다 훨씬 높은 품질의 생성 및 정밀한 동기화를 달성한다는 것을 보여주며, 복잡한 장면 비디오 처리에서 특히 뛰어난 성과를 냈습니다. 이러한 결과는 JAVG 태스크에 대한 새로운 기준을 설정하게 됩니다.



### Map Feature Perception Metric for Map Generation Quality Assessment and Loss Optimization (https://arxiv.org/abs/2503.23370)
- **What's New**: 이 연구는 맵 생성 과제에서 생성된 맵의 진정성을 평가하기 위한 새로운 메트릭인 Map Feature Perception Metric (MFP)을 제안합니다. 기존 방법들이 주로 픽셀 단위의 오류 메트릭을 사용한 반면, MFP는 깊은 특성(feature)을 추출하여 카토그래픽 구조적 무결성과 위상적 관계를 종합적으로 인코딩합니다. 실험 결과, MFP는 기존의 메트릭에 비해 카토그래픽의 의미적 특징을 평가하는 데 뛰어난 능력을 보여줍니다.

- **Technical Details**: MFP 메트릭은 Vision Transformer (ViT)를 기반으로 하여 설계되었습니다. 이 메트릭은 생성된 맵과 목표 맵 간의 전역적인 특성과 공간적 일관성을 평가하며, 렌더링 품질을 높이는데 중요한 역할을 합니다. 본 연구에서는 다양한 생성 모델 환경에서 MFP가 기존의 L1, L2, SSIM 메트릭보다 성능을 향상시킨다는 점을 실험적으로 입증했습니다.

- **Performance Highlights**: MFP를 손실 함수(loss function)로 사용하여 카토그래픽 생성 모델이 향상된 품질의 맵을 생성할 수 있음을 나타냈습니다. 실험에서는 기존의 메트릭에 비해 성능 향상이 2%에서 50%까지 이르는 것으로 확인되었습니다. 이 연구는 생성 모델 최적화 시 카토그래픽 글로벌 속성과 공간적 일관성을 명시적으로 고려하는 것이 생성되는 맵의 지리적 가능성을 크게 향상시킨다는 결론을 내립니다.



### Towards Physically Plausible Video Generation via VLM Planning (https://arxiv.org/abs/2503.23368)
Comments:
          18 pages, 11 figures

- **What's New**: 최근 비디오 확산 모델(Video Diffusion Models, VDMs)은 매우 사실적인 비디오를 생성하는데 유망하지만, 물리 법칙을 잘 이해하지 못해 물리적으로 그럴듯한 비디오를 생성하는 데 어려움을 겪고 있습니다. 본 논문에서는 두 단계로 나누어진 새로운 영상-비디오 생성 프레임워크를 제안하여 물리적 요소를 명시적으로 통합합니다. 첫 번째 단계에서는 비전을 기반으로 한 언어 모델(Vision Language Model, VLM)을 사용하여 물리적 동적을 예측하는 조잡한 동작 경로를 계획하고, 두 번째 단계에서는 이러한 경로를 기반으로 VDM을 통해 비디오를 생성합니다.

- **Technical Details**: 제안된 프레임워크는 VLM을 coarse-level motion planner로, VDM을 fine-level motion synthesizer로 활용하는 두 단계 구조를 가집니다. VLM은 체계적 사고(chain-of-thought)와 물리적 사고를 결합하여 물리적 동적을 대략적으로 따르는 동작 경로를 생성하도록 합니다. 생성된 동작 경로에 따라 VDM은 일반적인 물리 법칙에 부합하는 세밀한 동작을 생성합니다. 노이즈를 추가하는 방식으로 세부 동작을 생성하여, 고유한 속도, 가속도 등의 정보를 효과적으로 모델링할 수 있습니다.

- **Performance Highlights**: 제안된 프레임워크는 주요 물리 기반 비디오 벤치마크에서 우수한 성능을 보여주었으며, 물리적으로 그럴듯한 동작을 생성하는 데 성공적이었습니다. 기존 비디오 생성 방법들과 비교했을 때, 제안된 방법은 더욱 뛰어난 결과를 기록하며, 사용자 연구를 통해 일반화 가능성과 효과성을 입증하였습니다. 이러한 성과는 VLM과 VDM을 활용하여 물리적으로 그럴듯한 비디오 생성의 가능성을 한층 더 높였음을 보여줍니다.



### FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning (https://arxiv.org/abs/2503.23367)
Comments:
          Technical Report

- **What's New**: 본 논문에서는 Visual Autoregressive (VAR) 모델링의 새로운 접근법인 FastVAR를 제안합니다. 기존의 VAR 방식은 이미지 해상도가 증가할수록 계산 복잡도와 런타임 지연이 급격하게 증가하는 문제를 가지고 있었습니다. FastVAR는 이 문제를 해결하기 위해, 캐시된 토큰 프루닝(cached token pruning) 전략을 도입하여, 효율적으로 해상도를 조정할 수 있는 방법을 모색합니다. 이를 통해, 성능 저하 없이 2.7배의 속도 향상이 가능하다는 점이 주목할 만합니다.

- **Technical Details**: FastVAR는 대규모 단계에서 주요 토큰만 전달하고, 이전 단계에서 캐시된 토큰을 사용하여 삭제된 슬롯을 복원하는 접근 방식을 취합니다. 이를 위해, Pivotal Token Selection (PTS)이라는 메커니즘을 개발하여 각 토큰의 중요도를 평가하고, 고주파(high-frequency) 토큰을 선택합니다. 이 과정에서, 제거된 토큰은 Cached Token Restoration (CTR) 기법을 통해 복원되어 2D 이미지 구조를 유지합니다. 이 모든 과정은 비훈련(training-free) 방식으로 구현되어 다양한 VAR 기반 모델에 쉽게 통합됩니다.

- **Performance Highlights**: 실험 결과, FastVAR는 FlashAttention 기반의 VAR 모델을 사용하여 2.7배의 속도 향상과 1% 미만의 성능 손실로 높은 해상도의 이미지를 생성할 수 있음을 보여줍니다. 특히, FastVAR는 15GB 메모리로 단일 NVIDIA 3090 GPU에서 1.5초 안에 2K 이미지를 생성할 수 있습니다. 이러한 성능은 VAR 모델이 고해상도 생성에서 직면하던 문제를 해결하며, 다음 단계의 예측을 위한 새로운 가능성을 제시합니다.



### OnSiteVRU: A High-Resolution Trajectory Dataset for High-Density Vulnerable Road Users (https://arxiv.org/abs/2503.23365)
- **What's New**: 이 논문에서는 도시화가 가속화되면서 교통 수요가 증가함에 따라 취약한 도로 사용자(Vulnerable Road Users, VRUs)의 안전 문제를 해결하기 위해 OnSiteVRU 데이터세트를 개발했다. 이 데이터세트는 다양한 시나리오를 포함하여, 모터 차량, 전기 자전거, 인간이 구동하는 자전거의 궤적 데이터를 제공하며, 약 17,429개의 궤적을 0.04초의 정밀도로 수집하였다. 이러한 데이터는 공중 전망의 자연 주행 데이터와 탑재된 실시간 동적 감지 데이터를 통합하여 교통 신호, 장애물 및 실시간 지도와 같은 환경 정보를 포함하고 있다.

- **Technical Details**: OnSiteVRU 데이터세트는 복잡한 도시 교차로, 도로 구간 및 도시 마을을 포함한 다양한 환경에서 VRU 행동을 포착하기 위해 다수의 감지 기술 및 고정밀 라이다(LiDAR)와 컴퓨터 비전 알고리즘을 통합하여 신뢰할 수 있는 데이터를 구축하였다. 이 데이터는 VRU의 행동 패턴에 대한 비상 및 링크 환경에서의 응답 정확성을 평가할 수 있는 기초 자료로 활용될 수 있다. 데이터의 익명화 기술과 자동 주석 도구를 사용하여 데이터의 프라이버시를 보장하면서도 주석의 효율성과 정확성을 향상시켰다.

- **Performance Highlights**: VRU_Data는 기존 데이터세트에 비해 VRU 밀도와 장면 커버리지가 뛰어나 VRU 행동 특성을 훨씬 더 종합적으로 나타낸다. 이는 자율 주행 알고리즘의 최적화 및 트래픽 흐름 모델링과 궤적 예측에 중요한 지원을 제공한다. 공개적으로 제공되기 때문에 academia 및 산업계에서 VRU 행동 모델링 및 교통 안전 연구를 advancing하는 중요한 자원이 될 것으로 기대된다.



### VideoFusion: A Spatio-Temporal Collaborative Network for Mutli-modal Video Fusion and Restoration (https://arxiv.org/abs/2503.23359)
- **What's New**: 이번 논문에서는 실제 상황에 적합한 비디오 데이터를 활용하여 다중 센서 융합(multi-sensor fusion) 연구의 한계를 극복하고자 하는 접근법을 제시합니다. 연구자는 220개의 시간 동기화된 적외선-가시성 비디오 쌍으로 구성된 M3SVD 데이터셋을 구축하였으며, 이는 153,797 프레임으로 구성되어 있어 비디오 융합 분야의 데이터 격차를 해소합니다. 또한, VideoFusion이라는 다중 모달 비디오 융합 모델을 제안하여, 서로 다른 모달리티 간의 상호작용을 극대화합니다.

- **Technical Details**: VideoFusion 모델은 여러 단계의 모듈로 구성되어 있으며, 특히 1) 교차 모달 정보 상호작용을 위한 차별적 강화 모듈(differential reinforcement module)과 2) 동적으로 다중 모달 특성을 통합하기 위한 모달리티 안내 융합 전략(complete modality-guided fusion strategy)을 포함합니다. 3) 이 외에도 양방향 시간 주의 메커니즘(bi-temporal co-attention mechanism)을 통해 시간적 문맥을 효과적으로 집계하여 프레임 간의 특성 표현을 강화합니다. 이러한 접근법은 전체적으로 보다 일관된 비디오 생성을 목표로 하고 있습니다.

- **Performance Highlights**: 실험 결과, VideoFusion은 기존의 이미지 중심 융합 패러다임에 비해 우수한 성능을 보이며, 특히 순차적 시나리오에서 시간 비일관성과 간섭을 효과적으로 완화합니다. 다중 모달 입력(multi-modal inputs)에서 생성된 비디오의 일관성을 유지하는 데 있어 VideoFusion이 가져다주는 향상이 두드러집니다. 이러한 결과들은 비디오 융합 연구에 있어 새로운 표준을 마련할 가능성을 보여줍니다.



### ControlFusion: A Controllable Image Fusion Framework with Language-Vision Degradation Prompts (https://arxiv.org/abs/2503.23356)
- **What's New**: 본 논문에서는 실제 세계의 복합적인 손상을 처리하기 위해 언어-비전 프롬프트를 활용한 제어 가능한 이미지 융합 프레임워크인 ControlFusion을 제안합니다. 이 모델은 물리적 이미징 메커니즘을 통합하여 복합 손상을 효과적으로 시뮬레이션할 수 있으며, 사용자가 정의한 손상 유형과 심각도를 반영할 수 있습니다. 특히, 프롬프트 조정을 통해 다양한 손상 수준에 적절히 대응하고, 사용자가 지시하지 않더라도 스스로 손상을 인식하여 자동으로 최적화를 수행합니다.

- **Technical Details**: ControlFusion 모델은 물리 기반의 손상 이미징 모델을 개발하여 조명, 날씨 및 센서 관련 왜곡과 같은 실제 손상 과정을 정확하게 시뮬레이션합니다. 이를 통해 합성 데이터와 실제 이미지를 연결하는 중요한 역할을 수행합니다. 또한, 동적 피쳐 분배를 가능하게 하는 프롬프트 조정 모듈을 통해 사용자 요구에 적응할 수 있는 고품질 융합 결과를 생성합니다.

- **Performance Highlights**: ControlFusion은 기존 SOTA (State Of The Art) 융합 방법들보다 뛰어난 융합 품질과 손상 처리 능력을 보여줍니다. 특히, 실제 세계 및 복합 손상에 대한 대응에서 강력한 성능을 발휘하며, 다양한 손상 수준에서도 안정적인 결과를 도출합니다. 실험 결과를 통해 이 방법이 일반화 문제를 극복하면서 사용자 맞춤형 솔루션을 제공할 수 있음을 입증하였습니다.



### DSPFusion: Image Fusion via Degradation and Semantic Dual-Prior Guidanc (https://arxiv.org/abs/2503.23355)
- **What's New**: DSPFusion(Degradation and Semantic Prior Fusion)은 저화질 이미지를 보다 효과적으로 융합하기 위한 새로운 프레임워크로, 기존의 이미지 융합 방법들이 고화질 이미지에 맞춰져 있음을 강조합니다. 특히 이 접근법은 diffusion 모델을 활용하여 복잡한 저화질 이미지를 다루는 데 필요한 정보 회복 및 융합 과정을 혁신적으로 처리합니다. 이 모델은 과거의 문제점을 해결하며, 정보 보강과 융합을 동시에 수행할 수 있는 단일화된 체계를 제공합니다.

- **Technical Details**: DSPFusion은 저화질 이미지에서 개별적으로 모드 특화된 degradation prior를 추출하고, 전반적인 저화질 semantic prior를 효과적으로 포착합니다. 나아가, 각 모드에서 필수적인 semantic 정보를 복원하는 과정을 통해 계산 효율성과 경량화를 달성합니다. 이 과정은 degradation 종류를 동적으로 식별하고, diffusion 모델이 semantic prior를 정제하여 융합 품질을 개선하는데 기여합니다.

- **Performance Highlights**: DSPFusion은 다양한 저화질 문제를 효과적으로 완화하면서, 최소한의 계산 비용으로 보완적인 컨텍스트를 통합하는 능력을 입증하였습니다. 여러 실험을 통해, 이 방법은 기존의 융합 방식에 비해 20배 이상의 속도를 자랑하며, 군사 탐지, 보안 감시, 보조 운전 등 실제 응용 분야에서 활용 가능성을 높이고 있습니다.



### Object Isolated Attention for Consistent Story Visualization (https://arxiv.org/abs/2503.23353)
Comments:
          6 pages, 4 figures

- **What's New**: 이 논문에서는 이야기 시각화(Story Visualization)에서 등장인물의 일관성을 유지하면서 자연스럽고 맥락에 맞는 장면을 생성하기 위해 강화된 Transformer 모듈을 제안합니다. 새로운 접근 방식을 통해 기존 방법들이 겪는 여러 문제를 해결하며, 특히 훈련 없이도 새로운 캐릭터와 스토리를 생성할 수 있는 가능성을 제시합니다. 이를 위해 분리된 self attention과 cross attention 메커니즘을 활용하여 논리적인 장면 생성을 보장합니다.

- **Technical Details**: 제안된 방법은 isolated self attention과 isolated cross attention 메커니즘을 사용하여 각 캐릭터의 특성을 독립적으로 처리합니다. 이러한 방식은 주의 맵을 정제하여 불필요한 영역에 대한 주의를 줄이고 특정 캐릭터의 주요 특성에 집중할 수 있도록 합니다. 또한, 상호 주의 메커니즘을 통해 서로 다른 캐릭터 간의 피처 혼합을 방지하며, 일관성을 강화합니다.

- **Performance Highlights**: 정성적(qualitative) 및 정량적(quantitative) 평가에서 본 방법이 현재의 기술들을 초과하는 성능을 보이며, 테스트 결과에서 시각적으로 일관되고 응집력 있는 스토리 시각화를 달성하는 효율성을 보여줍니다. 기존의 방법들보다 개선된 결과를 제공하면서도 재조정 없이 연속적인 캐릭터와 스토리라인 생성을 가능하게 하는 등 실용성이 높은 결과를 나타냈습니다.



### From Panels to Prose: Generating Literary Narratives from Comics (https://arxiv.org/abs/2503.23344)
- **What's New**: 이 논문은 시각 장애인 독자를 위한 만화 접근성을 해결하기 위해 자동화된 시스템을 개발하는 새로운 방법을 제시합니다. Magiv3라는 통합 모델을 통해 만화의 패널, 캐릭터, 텍스트 등 다양한 기능적 작업을 수행하며, 3300개 이상의 일본 만화 패널에 대한 인간 주석이 포함된 캡션을 공개합니다. 이러한 혁신적인 접근은 만화의 내러티브를 텍스트로 변환하여 독자들에게 깊이 있는 경험을 제공합니다.

- **Technical Details**: 논문에서는 만화의 비주얼 스토리텔링을 이해하기 위한 복잡한 작업을 수행하기 위해 구조화된 다단계 파이프라인을 제안합니다. 첫 단계에서는 패널, 캐릭터, 텍스트, 스피치 버블 꼬리를 식별하고 연결하며, Optical Character Recognition (OCR)을 통해 텍스트를 추출합니다. 그 후, 각 패널은 독립적으로 캡션화되어 장면, 동작 및 감정 단서를 포함하고, 마지막으로 LLM을 사용하여 작성된 정보를 바탕으로 일관된 내러티브가 생성됩니다.

- **Performance Highlights**: 测试 결과, Magiv3와 대형 비전-언어 모델(VLM) 통합을 통해 비주얼 스토리텔링의 깊이와 명확성을 유지하며, 시각적으로 장애가 있는 청중도 만화를 통해 몰입감 있는 경험을 할 수 있게 됩니다. 최종적으로, 이 시스템은 만화를 통해 텍스트 기반의 풍부한 이야기 경험을 제공하며, 시각적 콘텐츠를 인식할 수 없는 독자들에게 새로운 접근 방식을 제시합니다.



### Enhancing 3D Gaussian Splatting Compression via Spatial Condition-based Prediction (https://arxiv.org/abs/2503.23337)
Comments:
          The paper has been accepted by ICME2025 in March,2025

- **What's New**: 이번 연구에서는 3D Gaussian Splatting (3DGS)의 압축 효율성을 향상시키기 위해 예측 기법을 도입했습니다. 기존의 3DGS는 대량의 저장공간을 필요로 하는 한계를 가지고 있었으며, 예측 모듈을 통해 비트율을 효과적으로 절감할 수 있게 되었습니다. 연구는 특히 공간 조건 기반 예측 모듈을 통해 장면 정보를 활용하여 세밀한 정보의 보상을 학습하는 전략을 제안합니다.

- **Technical Details**: 3DGS는 장면을 나타내기 위해 많은 수의 타원체를 사용하며, 여러 속성 (위치, 크기, 회전, 불투명도 및 색상)을 포함합니다. 본 연구에서는 Feature Prediction Network (FP-Net)를 통해 공간 조건 정보와 학습 가능한 잔여 정보를 결합하여 앵커 특성을 예측합니다. 이를 통해 원래의 앵커 특성 대신에 잔여 정보만 인코딩하여 비트율을 상당히 줄이는 결과를 얻습니다.

- **Performance Highlights**: 다양한 실험 결과, 제안한 압축 프레임워크는 3DGS 대비 105배 이상의 크기 축소를 보여주었으며, SOTA 압축 기술인 HAC에 비해 비트율을 24.42% 절감했습니다. 이 모든 과정에서도 높은 품질의 렌더링을 유지하며, 이는 압축 작업에서 상당한 이득으로 간주됩니다. 향후 코드도 공개될 예정입니다.



### TraceMark-LDM: Authenticatable Watermarking for Latent Diffusion Models via Binary-Guided Rearrangemen (https://arxiv.org/abs/2503.23332)
Comments:
          14 pages, 6 figures,

- **What's New**: 본 논문은 Latent Diffusion Model (LDM)을 위한 새로운 방안인 TraceMark-LDM을 제안합니다. 이 방법은 기존의 워터마킹 기술의 한계를 극복하고, 이미지 품질을 보호하면서도 생성된 이미지의 출처를 정확히 표시할 수 있게 설계되었습니다. 또한, 이 알고리즘은 Gaussian 분포에서 무작위 변수 샘플링을 기반으로 하여 워터마크 정보를 활용해 rearrangement를 수행합니다.

- **Technical Details**: TraceMark-LDM은 생성된 이미지의 품질 저하 없이 워터마크를 통합하는 비파괴적 워터마킹 방식을 제공합니다. 워터마크 정보는 Gaussian 분포에서 샘플링된 무작위 변수를 재배열하는 과정에 이용되며, LDM 인코더를 정밀 조정하여 다른 왜곡에 대한 강인성을 강화합니다. 이러한 방식은 기존의 워터마킹 기법에 비해 워터마크 추출 오류를 줄이는 데 효과적입니다.

- **Performance Highlights**: TraceMark-LDM의 실험 결과는 기존 최첨단 기법에 비해 이미지 품질과 출처 표시 정확도가 우수함을 보여줍니다. 또한, 이 방법은 다양한 일반적인 공격 방법에 대해 뛰어난 강인성을 발휘하여, 공격의 강도에 관계없이 지속적으로 SOTA 방법들을 초월하는 성능을 나타냅니다.



### HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation (https://arxiv.org/abs/2503.23331)
Comments:
          CVPR2025

- **What's New**: 본 논문에서는 2D에서 3D로의 인간 포즈 추정(HPE)에서 발생하는 가림 문제를 해결하기 위한 혁신적인 두 단계 생성 밀착 방법인 Hierarchical Pose AutoRegressive Transformer(HiPART)를 제안합니다. 기존 방법들은 데이터의 희소성 문제와 가림 상황의 복잡성을 간과했습니다. HiPART는 원래 희소 2D 포즈에서 계층적인 2D 조밀한 포즈를 생성하여 가림 상황에서의 강력한 복원력을 입증하였습니다.

- **Technical Details**: HiPART는 두 개의 주요 모듈로 구성되어 있습니다. 첫 번째는 Multi-Scale Skeletal Tokenization(MSST) 모듈로, 밀접하게 밀착된 2D 포즈를 계층적인 토큰으로 양자화합니다. 두 번째는 Hierarchical AutoRegressive Modeling(HiARM) 스킴을 통해 계층적인 2D 포즈 생성을 달성합니다. 이 방법은 비유클리드 구조에 적합한 새로운 희소-밀착 및 중심-주변 전략을 도입하여 효율성을 보여줍니다.

- **Performance Highlights**: HiPART는 단일 프레임 기반 3D HPE에서 최신 성과를 달성하였으며, Human3.6M, 3DPW 등 다양한 벤치마크에서 뛰어난 강건성을 보여줍니다. 복잡한 시계열 인코더를 사용하는 방법들과 비교하여 현저하게 감소된 복잡성으로 동등하거나 우수한 성능을 발휘합니다. 나아가 HiPART는 기존의 시계열 방법과 독립적으로 작동하며 성능 향상에 더욱 기여할 수 있습니다.



### EagleVision: Object-level Attribute Multimodal LLM for Remote Sensing (https://arxiv.org/abs/2503.23330)
Comments:
          Under Review

- **What's New**: EagleVision은 원거리 감지(remote sensing)에서 객체 탐지(object detection) 및 속성 이해(attribute comprehension)에 초점을 맞춘 새로운 다중 모달 대형 언어 모델(MLLM)입니다. 기존의 MLLM이 가진 한계를 극복하기 위해 Attribute Disentangle 모듈을 적용하여 시각 토큰을 분리하여 명확한 속성을 표현할 수 있습니다. 이 모델은 EVAttrs-95K라는 새로운 대규모 데이터셋을 통해 훈련되어 있으며, EVBench라는 평가 벤치마크를 도입했습니다. 연구 결과 EagleVision은 정밀 탐지 및 속성 이해 작업에서 최첨단 성능을 달성했습니다.

- **Technical Details**: EagleVision의 핵심은 객체 속성 이해를 위한 Attribute Disentangle 모듈입니다. 이 모듈은 고유한 속성을 분리하여 명확하게 포착할 수 있는 시각 토큰을 생성합니다. 또한, 원거리 감지 데이터를 이용한 훈련 과정에서 일관된 특징 표현을 유지하며, 다양한 객체 속성을 밝혀냅니다. 이와 함께, EVAttrs-95K 데이터셋은 95,100개 이상의 객체에 대한 상세 속성을 정의하는 혁신적인 주석 프로세스를 통해 구축되었습니다.

- **Performance Highlights**: EagleVision은 객체 탐지에서 mAP를 기존 모델 대비 11.2% 향상시켰으며, 세 가지 데이터셋에서 각각 2.7% 및 0.3%의 개선 효과를 나타냈습니다. 이 모델은 EVBench에서의 실험을 통해 속성 이해능력에서도 중대한 이점을 증명하며, 탐지와 이해 간의 상호 발전을 도모합니다. 이러한 결과는 EagleVision이 기존 방법보다 더욱 실용적이고 뚜렷한 성과를 달성했다는 것을 보여줍니다.



### SpINR: Neural Volumetric Reconstruction for FMCW Radars (https://arxiv.org/abs/2503.23313)
- **What's New**: 이 논문에서는 FMCW (Frequency-Modulated Continuous-Wave) 레이더 데이터를 이용한 복합체적 재구성을 위한 새로운 프레임워크인 SpINR을 소개합니다. 기존의 레이더 이미징 기법은 이상적인 신호 모델을 가정하고 조밀한 개구 샘플링을 요구하여 해상도와 일반화의 한계가 있었습니다. SpINR은 주파수 도메인에서 작동하는 완전 미분 가능한 포워드 모델을 결합하여 이러한 문제를 해결하고, 레이더 장면 기하학을 보다 효율적이고 정확하게 학습할 수 있도록 합니다.

- **Technical Details**: SpINR의 핵심 통찰은 FMCW 레이더 시스템에서 비트 주파수와 산란자 거리 간의 선형 관계를 활용하는 것입니다. 이 모델은 계산이 용이한 미분 가능 포워드 모델을 구현하여 긴 시퀀스 대신 스파스 주파수 빈에 대해서만 레이더 응답을 분석합니다. 이 접근법은 기존 시간 도메인 기법보다 더 효율적이며, 다수의 송수신 쌍과 자세를 통해 관찰된 주파수 도메인 신호와 일치하는 복합체적 기하학을 재구성할 수 있도록 지원합니다.

- **Performance Highlights**: SpINR은 고전적인 백프로젝션 기법 및 기존 학습 기반 접근 방식에 비해 현저하게 우수한 성능을 보이며, CCC (Structural Similarity Index), Chamfer Distance, PSNR, LPIPS, IoU 등 여러 척도를 통해 높은 해상도와 정확한 재구성을 보여줍니다. 이 연구는 FMCW 레이더 신호에서의 신경 체적 재구성의 첫 응용을 나타내며, 레이더 기반 이미징 및 인지 시스템의 미래 연구 방향에 기여할 수 있는 가능성을 제공합니다.



### MoCha: Towards Movie-Grade Talking Character Synthesis (https://arxiv.org/abs/2503.23307)
Comments:
this https URL

- **What's New**: 이 논문은 모션 리얼리즘을 중시하는 비디오 생성의 최신 발전이 캐릭터 중심의 스토리텔링을 간과하고 있다는 점을 지적하며, 음성과 텍스트로부터 직접적으로 대화하는 캐릭터 애니메이션을 생성하는 새로운 작업인 Talking Characters를 소개합니다. 이를 위해 MoCha라는 새로운 모델을 제안하며, 이는 단순한 talking head를 넘어 다수의 캐릭터의 전체 초상화를 생성하는 것을 목표로 합니다. MoCha는 음성과 비디오를 정밀하게 동기화하기 위한 speech-video 윈도우 어텐션 메커니즘과 함께 여러 캐릭터의 대화를 지원하는 구조화된 프롬프트 템플릿을 제공합니다.

- **Technical Details**: MoCha 모델은 텍스트와 음성을 기반으로 고품질의 캐릭터 생성을 실현하는 최초의 디퓨전 트랜스포머(Diffusion Transformer)로, 외부 조건 없이 최전방 훈련을 통해 동작 다양성과 일반화를 개선합니다. 또한, 음성과 비디오 입력을 정렬하는 새로운 어텐션 메커니즘을 도입하고, 음성 라벨 및 텍스트 라벨이 모두 포함된 공동 훈련 전략을 통해 훈련됩니다. 그 결과, 캐릭터의 표정, 행동 및 상호작용을 자연스럽게 제어할 수 있습니다.

- **Performance Highlights**: MoCha는 인간의 선호도 연구와 벤치마크 비교를 포함한 포괄적인 정성적 및 정량적 평가를 통해 AI 생성 영화 스토리텔링의 새로운 기준을 설정했습니다. MoCha는 현실감, 표현력, 제어 가능성 및 일반화 능력에서 뛰어난 성과를 보여주며, 이는 영화 제작, 애니메이션, 가상 비서 등 다양한 분야에 걸쳐 광범위한 응용이 가능합니다.



### Learning Predictive Visuomotor Coordination (https://arxiv.org/abs/2503.23300)
- **What's New**: 이 논문은 인간의 시각운동(coordination) 예측을 위한 새로운 방법론을 제시합니다. 특히, 머리 자세(head pose), 시선(gaze), 상체 움직임(upper-body motion)을 예측하는 과제를 도입하여 이러한 신호의 구조적 시간적 의존성을 학습하는 Visuomotor Coordination Representation(VCR)을 제안합니다. 이는 기존의 연구들이 단독 신호들에 초점을 맞춘 것과 달리 다중 모달 신호의 결합을 통해 정확한 예측을 가능하게 합니다.

- **Technical Details**: 이 연구에서는 ego-vision과 운동 데이터를 통합한 확산(diffusion) 기반 모션 모델링 프레임워크를 확장하여 시각운동 예측을 수행합니다. 이 과정에서, kinematic 시퀀스를 인코딩하고 미래의 움직임 경로를 생성하는 방식을 통해, 시각운동 동작의 시간적 일관성을 높이고 예측의 정확성을 증대시킵니다. 또한, EgoExo4D 데이터셋을 활용하여 3D의 시선, 머리 자세, 상체 움직임을 평가합니다.

- **Performance Highlights**: 모델은 1초 예측 기간 동안 시각운동 번역에 대해 평균 59mm, 머리 회전 오류에 대해서는 13도라는 우수한 성능을 보였습니다. 이러한 결과는 다양한 현실 세계의 활동에서 일반화된 능력을 보여주며, 또한 시각운동 조정의 이해에 있어 다중 모달 통합의 중요성을 강조합니다. 이 연구는 로봇 공학, VR/AR 및 인간-컴퓨터 상호작용 분야에서 인간의 움직임 예측 능력을 향상시키기 위한 기초로 활용될 수 있습니다.



### ReasonGrounder: LVLM-Guided Hierarchical Feature Splatting for Open-Vocabulary 3D Visual Grounding and Reasoning (https://arxiv.org/abs/2503.23297)
- **What's New**: 이 논문에서는 Open-vocabulary 3D visual grounding과 reasoning을 위한 새로운 프레임워크인 ReasonGrounder를 제안합니다. ReasonGrounder는 고차원 3D Gaussian 필드를 사용하여 객체를 시각적 장애물 속에서도 정확하게 위치시키는 능력을 제공합니다. 이는 전통적인 방법들이 3D 주석에 의존함으로써 보이는 한계를 극복하고, 다양한 세멘틱을 처리할 수 있도록 해 줍니다.

- **Technical Details**: ReasonGrounder는 LVLM(대형 비전-언어 모델)을 기반으로 하여, 물리적 스케일에 따른 적응형 그룹화를 가능하게 하는 3D Gaussian feature fields를 사용합니다. 이를 통해 객체가 부분적으로 보이거나 완전히 가려지는 경우에도 계층적 3D Gaussian 스플래팅을 통해 정확한 지역화를 수행할 수 있습니다. 또한, ReasonGrounder는 2D 세그멘테이션 마스크와 멀티뷰 CLIP 임베딩을 결합하여 Gaussian 그룹을 선택합니다.

- **Performance Highlights**: 실험 결과, ReasonGrounder는 현실 세계 시나리오에서 3D grounding의 정확성을 크게 향상시켰습니다. 새로운 ReasoningGD 데이터셋은 10,000개 이상의 복잡한 장면과 2,000,000개의 주석을 포함하고 있어, open-vocabulary 3D grounding 및 amodal perception의 평가 시 중요한 자원이 됩니다. 이러한 발전은 자율 로봇 및 증강 현실과 같은 다양한 적용 분야에 기여할 것입니다.



### Language Guided Concept Bottleneck Models for Interpretable Continual Learning (https://arxiv.org/abs/2503.23283)
Comments:
          CVPR 2025; Project Page: this https URL

- **What's New**: 이 논문에서는 언어 가이드 Concept Bottleneck Models (CBMs)를 활용하여 지속적 학습(Continual Learning, CL)에서의 해석 가능성과 지식 보존 능력을 동시에 향상시키는 혁신적인 프레임워크를 소개합니다. 기존 CL 방법들은 주로 성능 향상에 집중했으나, 학습 과정의 해석 가능성이 중요한 문제로 대두되고 있습니다. 우리는 CLIP 모델과의 의미론적 일치를 통해 인지 가능한 개념을 학습하는 방식을 제안하며, 이는 다양한 작업 간에 일반화 가능한 지식을 제공합니다.

- **Technical Details**: 이 연구에서는 CLIP 모델을 사전 훈련된 언어 모델과 통합하여 개념 신경망의 구조를 구성했습니다. 구체적으로, CLIP 텍스트 인코더를 사용하여 개념 임베딩을 생성하고, 이를 바탕으로 정보량이 많은 개념을 선택합니다. 그 후, 이러한 개념을 기반으로 Concept Bottleneck Layer (CBL)를 구성하여 해석 가능성을 증대시키고, 기존 지식을 효과적으로 유지하며 재훈련의 필요성을 줄입니다.

- **Performance Highlights**: 이 방법은 여러 벤치마크 데이터셋에서 최첨단 방법들을 초월하여 최고의 성능을 기록하며, ImageNet.subset 데이터셋에서는 평균 정확도가 최대 3.06% 향상되었습니다. 또한, 모델의 예측에 대한 개념 시각화를 제공하여 해석 가능한 지속적 학습의 이해를 더욱 발전시킵니다. 이러한 성과는 해석 가능성과 지식 유지의 균형을 강조하며, 실제 적용 가능성을 높이는 데 기여합니다.



### AnyCam: Learning to Recover Camera Poses and Intrinsics from Casual Videos (https://arxiv.org/abs/2503.23282)
Comments:
          CVPR 2025 - For more details and code, please check out our project page under this https URL

- **What's New**: AnyCam은 동적인 비디오 시퀀스로부터 카메라 자세(camera pose)와 내부 파라미터(intrinsics)를 직접 추정하는 빠른 트랜스포머(transformer) 모델입니다. 기존의 SfM(System from Motion) 방식이 동적 장면 처리에 한계를 보였던 반면, AnyCam은 레이블이 없는 다양한 비디오 데이터셋을 사용할 수 있는 강력한 일반화 능력을 자랑합니다. 이 모델은 불확실성 기반의 손실(loss) 공식을 사용하여 훈련을 지원하고, 테스트 시에는 경량의 경로 정제(trajectory refinement) 단계를 포함하여 장기적인 드리프트를 방지합니다.

- **Technical Details**: AnyCam은 입력으로 비디오 프레임의 시퀀스를 받아들여, 각각의 프레임으로부터 카메라의 자세와 내부 파라미터를 직접 예측합니다. 수치적 불확실성을 모델링한 훈련 방식은 일반 대중이 접근할 수 있는 비디오 데이터로 학습할 수 있도록 구축되었습니다. 또한, 테스트 단계에서 성능을 개선하기 위한 추가적인 정제 절차를 구현하여 결과의 정확성을 높이고, 드리프트 현상을 최소화합니다.

- **Performance Highlights**: AnyCam은 여러 동적 기준 데이터셋에서 검증된 결과로, 카메라 자세와 내부 파라미터에 대한 정확도를 qualitatively 및 quantitatively 모두 확보했습니다. 이 모델은 기존 동적 상황에서의 SfM 방법들보다 현저히 빠른 성능을 자랑하며, 높은 품질의 4D 포인트 클라우드를 생성할 수 있는 능력을 보여줍니다. 이로 인해 AnyCam은 기존 레이블 기반 방법들과 동등한 성능을 발휘하면서도, 강력한 전이 학습(transfer learning)을 가능하게 합니다.



### Improved Ear Verification with Vision Transformers and Overlapping Patches (https://arxiv.org/abs/2503.23275)
- **What's New**: 귀 인식은 성인 동안 상대적으로 안정된 외모로 인해 유망한 생체 인식 방식으로 떠오르고 있습니다. 이 연구에서는 Vision Transformers (ViTs) 모델의 다양한 설정을 귀 인식에 적용하였으며, 겹치는 패치를 선택하는 전략을 사용하여 실험하였습니다. 겹치는 패치의 중요성이 입증되었고, 이는 48개의 실험 중 44개에서 우수한 성능을 나타내었습니다.

- **Technical Details**: 연구에서는 ViT-Tiny (ViT-T), ViT-Small (ViT-S), ViT-Base (ViT-B), ViT-Large (ViT-L)의 설정을 사용하여 OPIB, AWE, WPUT, EarVN1.0 데이터셋에서 실험하였습니다. 겹치는 패치를 사용함으로써 귀의 복잡한 특징을포착할 수 있었고, 귀 인식의 모델 성능에서 ViT-T 모델이 다른 모델들보다 지속적으로 우수한 성과를 보였습니다. 특히 패치 크기와 보폭을 설정하는 방법이 인식 성능에 중요한 영향을 미쳤습니다.

- **Performance Highlights**: 연구 결과, EarVN1.0 데이터셋에서 겹치는 패치를 사용한 경우 성능이 최대 10% 증가하였으며, ViT-T 모델이 AWE, WPUT, EarVN1.0 데이터셋에서 가장 높은 성과를 기록했습니다. 최적의 성능은 패치 크기 28x28과 보폭 14픽셀의 설정에서 달성되었습니다. 이 연구는 겹치는 패치를 선택한 transformer 아키텍처가 귀 생체 인식 작업에 효과적이고 높은 성능을 발휘할 수 있음을 확인시켜 줍니다.



### OwlSight: A Robust Illumination Adaptation Framework for Dark Video Human Action Recognition (https://arxiv.org/abs/2503.23266)
- **What's New**: 이 논문은 저조도 환경에서의 인간 행동 인식을 위한 새로운 프레임워크인 OwlSight를 제안합니다. 이 프레임워크는 동작 분류와 상호작용하여 어두운 비디오에서의 인간 행동 인식을 정밀하게 수행하도록 설계되었습니다. OwlSight는 Time-Consistency Module (TCM)과 Luminance Adaptation Module (LAM)을 포함하여, 이를 통해 저조도에서의 성능을 향상시킵니다.

- **Technical Details**: OwlSight는 저조도 환경에서 데이터 처리능력을 극대화하기 위해 설계된 한 단계 학습 모델로, TCM이 왜곡된 시공간 특징을 캡처하고 LAM이 입력 조도 분포에 따라 동적으로 밝기를 조정합니다. Reflect Augmentation Module (RAM)을 통해 맥락 속에서 조명을 최대한 활용하며, 이는 두 개의 상호작용 경로를 통해 행동 인식을 향상시킵니다. 또한, Dark-101이라는 새로운 대규모 데이터셋을 구축하여, 101개의 행동 카테고리에 걸쳐 18,310개의 저조도 비디오를 포함하고 있습니다.

- **Performance Highlights**: OwlSight는 4개의 저조도 행동 인식 벤치마크에서 최첨단 성능을 기록하였으며, ARID1.5에서는 이전 최고의 방법보다 5.36% 향상되었습니다. 이는 기존의 데이터셋에 비해 크기와 다양성 면에서 뛰어난 여러 데이터를 섭렵할 수 있게 하여 모델의 훈련을 용이하게 만든 점에서 큰 의의가 있습니다. Dark-101 데이터셋은 최적의 행동 인식을 위한 기반으로서 중요한 역할을 할 것입니다.



### FIESTA: Fisher Information-based Efficient Selective Test-time Adaptation (https://arxiv.org/abs/2503.23257)
- **What's New**: 이 논문은 얼굴 표정 인식(facial expression recognition) 분야에서, 비구속적인 환경에서의 도메인 변화(domain shift) 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 기존의 테스트 시간 적응(test-time adaptation; TTA) 방식은 매개변수 업데이트의 수동 선택에 의존하는데, 이는 효율성을 저하시킬 수 있습니다. 본 연구에서는 Fisher 정보(Fisher information)를 기반으로 한 선택적 적응(selective adaptation) 프레임워크를 도입하여, 가장 중요한 매개변수만을 동적으로 업데이트합니다.

- **Technical Details**: 제안된 Fisher 기반 선택적 적응 기법은 비디오 기반 얼굴 표정 인식에 적합하게 설계되었습니다. 이 방법은 매개변수 중요도를 Fisher 점수(Fisher scores)로 정량화하고, 이를 통해 모델 성능에 중요한 가중치만을 선택적으로 업데이트합니다. 또한, 이 과정은 시간적 일관성(temporal consistency) 제약과 결합되어 모델의 적응 과정을 보다 효율적이고 효과적으로 만듭니다.

- **Performance Highlights**: AffWild2 벤치마크 데이터세트에 대한 실험 결과, 제안된 접근 방식이 기존 TTA 방법보다 7.7% 향상된 F1 점수를 기록하며, 22,000개의 매개변수만을 업데이트하는 것으로 확인되었습니다. 이는 기존의 방법들보다 20배 이상 적은 매개변수를 사용하는 결과입니다. 또한, 최소한의 데이터(1-3 프레임)로부터 매개변수의 중요도를 효과적으로 추정할 수 있어, 실제 애플리케이션에서 TTA를 더욱 실용적으로 만들어 줍니다.



### Context in object detection: a systematic literature review (https://arxiv.org/abs/2503.23249)
Comments:
          Artificial Intelligence Review Journal

- **What's New**: 이 연구는 컴퓨터 비전에서 물체 탐지(object detection)에 대한 맥락(context)의 중요성을 강조합니다. 맥락 정보는 이미지를 통한 시각 데이터 분석을 명확하게 하고, 다양한 접근 방법을 통해 물체 탐지의 정밀도와 효율성을 개선하는 데 기여합니다. 265개 이상의 논문을 포함하여 일반 물체 탐지, 비디오 물체 탐지, 소형 물체 탐지 등 여러 카테고리에서 다양한 맥락의 측면을 다룹니다.

- **Technical Details**: 연구는 맥락 정보가 물체 탐지에서 어떤 역할을 하는지를 여러 관점에서 조사하며, 최신 맥락 기반(object detection) 접근 방식을 리뷰하고 비교합니다. 이 논문은 또한 일반 물체 탐지, 비디오 물체 탐지, 그리고 소수의 샘플을 사용하는 few-shot 물체 탐지와 같은 다양한 기법들을 아우르고 있습니다. 이러한 접근 방식들은 물체 탐지에서의 효과적인 맥락 통합 방법에 중점을 두고 있습니다.

- **Performance Highlights**: 최신 맥락 기반 물체 탐지 방법의 발전을 포괄적으로 개괄하며, 연구자들에게 맥락 정보를 깊이 이해하는 데 필요한 가치 있는 기여를 제공합니다. 물체 탐지의 여러 카테고리에서의 연구 동향을 조명하여, 향후 연구가 필요한 갭(gap)을 식별하는 데에도 도움을 줍니다. 이러한 문헌 리뷰는 향후 객체 탐지 기술의 발전에 기여할 것으로 기대됩니다.



### Z-SASLM: Zero-Shot Style-Aligned SLI Blending Latent Manipulation (https://arxiv.org/abs/2503.23234)
Comments:
          Accepted to the CVPR 2025 Workshop AI for Creative Visual Content Generation Editing and Understanding

- **What's New**: Z-SASLM(Zero-Shot Style-Aligned SLI Blending Latent Manipulation)은 기존의 스타일 블렌딩 방법의 한계를 극복하기 위해 도입된 새로운 아키텍처입니다. 전통적인 선형 블렌딩 접근법 대신 SLI(구형 선형 보간)를 활용하여 가중 스타일 표현을 조합합니다. 이 방식은 레이턴트 공간의 비선형 기하학을 활용하여 이미지의 내재적 구조를 보존하며, 고품질 스타일 블렌딩을 가능하게 합니다.

- **Technical Details**: Z-SASLM은 세 가지 주요 모듈로 구성되어 있습니다: 참조 이미지 인코딩 및 블렌딩, 텍스트 인코딩, 그리고 StyleAligned 이미지 생성 과정입니다. 이 과정은 또한 다양한 입력을 통합할 수 있는 멀티모달 콘텐츠 융합 모듈로 확장됩니다. SLI는 고차원 표현에서 레이턴트 공간의 곡률을 보존하며, 스타일 표현의 가중 조합이 의미 있는 영역에 유지되도록 합니다.

- **Performance Highlights**: 실험 결과, Z-SASLM은 여러 스타일 간의 높은 정렬도를 달성하며, 멀티모달 콘텐츠 융합 환경에서도 효과적인 스타일 통합이 가능합니다. Weighted Multi-Style DINO VIT-B/8 메트릭을 통해 생성된 스타일 일관성을 정량적으로 평가할 수 있는 새로운 지표를 제안하며, 이로 인해 다양한 스타일의 조화와 정합성을 평가할 수 있는 방법론을 제공합니다.



### Synthetic Art Generation and DeepFake Detection A Study on Jamini Roy Inspired Datas (https://arxiv.org/abs/2503.23226)
Comments:
          13 pages, 7 figures, 6 tables

- **What's New**: 이번 연구는 생성 AI와 예술의 교차점에서 발생하는 도전과 기회를 탐구합니다. 특히, 인도의 화가 자미니 로이(Jamini Roy)의 독특한 스타일을 중심으로 한 확산 기반 생성 모델을 조사하고 있습니다. 이를 위해 Stable Diffusion 3를 조정하여 세밀한 이미지를 생성하고, 실제와 AI 생성 작품이 혼합된 새로운 데이터셋을 구축했습니다.

- **Technical Details**: 연구에서는 먼저 생성된 이미지와 진품 이미지 간의 미세한 차이를 발견하기 위해 푸리에 영역 평가(Fourier domain assessments) 및 자기상관 메트릭(autocorrelation metrics)과 같은 정성 및 정량적 방법을 사용했습니다. 특히 생성 기술의 발전에 따라 기존의 심층 AI 이미지 분석에서의 단점을 해결할 방안을 모색하고 있습니다. 이 모델들이 문화적 맥락에서 생성된 작품의 독특한 특성을 포착할 수 있는지에 대한 질문이 제기되고 있습니다.

- **Performance Highlights**: 연구의 최종 목표는 생성된 예술 작품의 신뢰성을 감별하는 데 있어 새로운 접근 방식을 제공하는 것입니다. 자미니 로이 스타일의 작품을 포함하는 이 새로운 데이터셋은 기존 기술의 한계를 극복할 수 있는 기회를 제시합니다. 저자들은 현재의 진품 탐지 기술이 고품질의 문화적으로 특정한 딥페이크를 식별하기에 어려움이 있다는 점을 강조하며, 이는 예술의 진위성을 보호하기 위한 중요한 연구 방향이 될 것입니다.



### Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection (https://arxiv.org/abs/2503.23220)
Comments:
          16 pages (8 main), 5 figures, accepted at CVPR 2025

- **What's New**: 이번 연구에서는 DINO Teacher라는 새로운 기술을 제안하여 도메인 적응 객체 감지(Domain Adaptive Object Detection, DAOD)에서 Mean Teacher 프레임워크의 한계를 극복하고자 하였습니다. DINO Teacher는 두 가지 주요 구성 요소로 이루어져 있으며, 먼저 대규모 DINOv2 백본을 동결하여 소스 데이터에서만 새로운 라벨러를 훈련시키는 방법을 사용하였습니다. 이를 통해 Mean Teacher보다 더 정확한 레이블을 생성하며, 이후 학생 모델의 소스 및 타겟 이미지 패치 특징을 DINO 인코더와 정렬하여 일반화된 DINO 표현에 가까워지도록 합니다.

- **Technical Details**: DINO Teacher는 학생 모델이 생성하는 레이블의 품질에 제약을 받지 않고, 대규모 비전 기초 모델(Vision Foundation Models, VFM)을 통해 라벨 세분화 과정을 독립적으로 수행합니다. 그래서 소스와 타겟 이미지에서의 패치 수준 출력을 대규모 VFM에 맞추어 정렬함으로써 학생 모델이 학습하는 특징들이 더 잘 일반화될 수 있도록 합니다. 이를 통해 DAOD 데이터셋에서 현존하는 최첨단 성능을 달성하였습니다.

- **Performance Highlights**: 이 연구에서는 BDD100k에서 7.6%의 성능 향상을, Foggy Cityscapes에서 2.3% 향상을 달성한 결과를 보고하였습니다. 또한, ACDC 테스트 분할에서도 새로운 결과를 제시하여 다양한 악천후 조건에 대한 성능 향상을 이루었습니다. 이러한 결과들은 DINO Teacher가 도메인 적응에서 더 나은 성능을 발휘할 수 있음을 증명합니다.



### Action Recognition in Real-World Ambient Assisted Living Environmen (https://arxiv.org/abs/2503.23214)
- **What's New**: 본 논문은 Robust and Efficient Temporal Convolution Network (RE-TCN)을 제안하며, 이는 Ambient Assisted Living (AAL) 기술에서의 행동 인식 문제를 해결하기 위해 다양한 기술적 요소를 활용합니다. 이 모델은 Adaptive Temporal Weighting (ATW), Depthwise Separable Convolutions (DSC), 데이터 증강 기법으로 구성되어 있으며, 이러한 구성 요소들은 노이즈와 가림에 대한 강인성, 모델의 정확성 및 효율성을 높이는 데 기여합니다.

- **Technical Details**: RE-TCN의 ATW는 행동 시퀀스 내에서 가장 중요한 프레임에 초점을 맞추도록 설계되었으며, 각 프레임에 동적으로 중요도를 할당합니다. DSC는 깊이와 점을 나누어 합성곱을 수행하여 입력된 스켈레톤 데이터를 처리하는 데 필요한 매개변수와 연산의 수를 대폭 줄입니다. 데이터 증강 기법은 실제 환경에서의 모델의 강인성을 향상시키기 위해 설계되었습니다.

- **Performance Highlights**: RE-TCN은 NTU RGB+D 60, Northwestern-UCLA, SHREC'17, DHG-14/28의 네 가지 벤치마크 데이터셋을 통해 기존 모델들보다 높은 정확도와 노이즈 및 가림에 대한 강인성을 보여주었습니다. 이 모델은 AAL 환경에서의 실시간 행동 인식 성능을 크게 개선하며, 컴퓨팅 효율성에서도 우수한 성과를 나타냅니다.



### Convolutional Neural Networks Can (Meta-)Learn the Same-Different Relation (https://arxiv.org/abs/2503.23212)
- **What's New**: 이 연구는 메타-러닝(meta-learning) 접근 방식을 사용하여 CNN(Convolutional Neural Network)이 '같음-다름(same-different)' 관계를 효과적으로 학습할 수 있는지를 조사합니다. 기존의 학습 방식으로는 CNN이 이러한 관계를 일반화하는 데 실패하는 반면, 메타-러닝 기술이 적용된 경우에는 성공적인 결과를 보였습니다. 이로 인해 CNN의 학습 능력에 대한 새로운 시각을 제시하고 있습니다.

- **Technical Details**: 연구에서는 MAML(Model-Agnostic Meta-Learning) 알고리즘을 사용하여 다양한 관련 작업에 대한 최적의 초기 가중치(initial weights)를 찾습니다. 이 기법은 다양한 작업 간의 공통 구조를 캡처하여 각각의 네트워크가 작업을 더 쉽게 학습할 수 있도록 돕습니다. 특히, 이러한 메타-러닝 접근 방식이 CNN의 일반화 능력을 어떻게 향상시키는지를 평가하며, 다양한 심층 CNN 아키텍처를 적용하여 '같음-다름' 과제를 수행했습니다.

- **Performance Highlights**: 결과적으로, 메타-러닝을 적용한 CNN 모델들은 새로운 자극에 대해 '같음-다름' 관계를 더 잘 일반화하는 경향을 보였으며, 기존의 CNN 모델들에 비해 성능이 크게 향상되었습니다. 이러한 발견은 CNN이 인간과 유사한 시각적 추론을 수행할 수 있는 잠재력을 지니고 있음을 시사합니다. 이 연구는 CNN의 메타-러닝을 통해 더욱 복잡한 관계를 이해하는 길을 여는 데 기여할 것으로 기대됩니다.



### A GAN-Enhanced Deep Learning Framework for Rooftop Detection from Historical Aerial Imagery (https://arxiv.org/abs/2503.23200)
- **What's New**: 이 연구는 역사적 공중 사진에서 지붕 감지를 위한 최신 접근 방식을 제안합니다. 두 단계의 이미지 개선 파이프라인을 구축하여 Generative Adversarial Networks (GANs)를 사용한 색상화(image colorization)와 Real-ESRGAN을 통한 초해상도(super-resolution) 기술을 통합하였습니다. 그 결과, YOLOv11n 모델이 평균 정밀도(Mean Average Precision, mAP) 85%를 초과하는 성과를 보여줍니다. 이는 원래의 흑백 이미지에 비해 약 40% 향상된 수치입니다.

- **Technical Details**: 연구에서는 첫 번째 단계로 색상화 과정을 통해 역사적 흑백 이미지를 RGB 색상 데이터로 변환합니다. 이 과정은 DeOldify라는 GAN 기반의 아키텍처를 채택하여 진행되며, 객체 감지 모델의 기본 사전 훈련에 적합한 이미지를 생성합니다. 이어지는 단계에서는 Real-ESRGAN을 통해 초해상도 기술로 이미지 품질을 향상시키고, 지붕과 같은 도시 물체의 선명성을 높입니다.

- **Performance Highlights**: 실험 결과, 본 연구법을 통해 역사적 공중 이미지를 보다 정확하게 분석할 수 있게 되었고, 특히 Charleston, South Carolina에서의 사례를 통해 그 효과성을 입증했습니다. 기존의 지붕 감지 방법에 비해 이중 개선 프로세스가 더 나은 성능을 발휘했으며, 최근 생성된 자료와 함께 학습된 모델을 통해 감지 정확도가 눈에 띄게 향상되었습니다.



### Real-time Video Prediction With Fast Video Interpolation Model and Prediction Training (https://arxiv.org/abs/2503.23185)
Comments:
          ICIP 2024

- **What's New**: 이 논문은 IFRVP(Intermediate Feature Refinement Video Prediction)라는 실시간 비디오 예측 방법을 제안하며, 이는 제로 레이턴시(Zero Latency) 상호작용을 가능하게 합니다. 기존 비디오 예측 모델들은 계산 비용이 많이 들고 실시간 응용 프로그램에서 비현실적이었으나, IFRVP는 이러한 문제를 효율적으로 해결합니다. 특히, 본 논문은 여러 가지 각각의 훈련 방법을 제안하여 비디오 프레임 예측의 정확도와 계산 속도 간의 최고의 균형을 달성하는 것을 목적으로 합니다.

- **Technical Details**: IFRVP는 간단한 합성곱(convolution) 기반 프레임 보간 네트워크를 기반으로 하며 세 가지 모델 훈련 방법을 제안합니다. 제안된 방법들은 각각 재발(prediction), 임의(arbitrary), 독립적(prediction)인 예측 방식을 적용해, 시간 경과에 따라 프레임을 처리하는 데 있어 높은 정확도를 보여줍니다. 또한, ELAN 기반의 잔차 블록(residual block)을 도입하여 예측 속도와 정확성을 향상시켰습니다.

- **Performance Highlights**: 제안하는 IFRVP 모델은 이전의 비디오 예측 방법들과 비교할 때 20% 더 빠른 예측 속도를 제공하며, MS-SSIM(multi-scale structural similarity index) 성능 또한 DMVFN보다 개선되었습니다. 이를 통해 실제 응용에서 제로 레이턴시 상호작용을 실현할 수 있는 가능성을 보여주고 있습니다. 결과적으로, 이 연구는 미래의 비디오 통신 및 원거리 상호작용의 품질을 획기적으로 향상시킬 수 있는 기반이 될 것입니다.



### Enhancing Weakly Supervised Video Grounding via Diverse Inference Strategies for Boundary and Prediction Selection (https://arxiv.org/abs/2503.23181)
- **What's New**: 이번 논문에서 제안하는 약한 감독 비디오 기초(Weakly Supervised Video Grounding, WSVG) 방법은 기존의 Gaussian 기반 제안 방안의 한계를 해결하기 위해 새로운 경계 예측(boundary prediction) 및 최상위 예측(selection) 전략을 도입했습니다. 특히, 경계 예측에서는 다양한 Gaussian 분포를 활용하여 최적의 경계를 더 정확하게 캡처할 수 있도록 합니다. 또한, 최상위 예측 선택 과정에서는 제안의 질을 고려하면서 가장 의미 있는 예측을 선별하는 방법을 제시합니다.

- **Technical Details**: 제안된 방법에서는 Gaussian mixture proposal을 기반으로 주어진 쿼리와 관련된 시간 경계(start time, end time)를 예측합니다. 이를 위해 N개의 Gaussian mixture proposal을 생성하며 각 제안은 Gaussian 마스크와 그에 따른 주의 가중치를 포함합니다. 두 가지 추론 전략을 통해 경계 예측 및 최상위 예측 선택을 수행하며, 실험적으로 총 다섯 가지의 경계 예측 전략과 네 가지의 최상위 예측 선택 전략을 비교하여 최적의 조합을 찾습니다.

- **Performance Highlights**: ActivityNet Captions 및 Charades-STA 데이터셋을 통한 광범위한 실험에서 제안한 추론 전략이 성능을 개선하는 것을 입증했습니다. 특히, 기존의 훈련 없이도 성능 향상을 이룰 수 있었으며, 약한 감독 방식에서 중요한 의미 있는 순간을 효과적으로 포착하는 데 기여했습니다.



### Intelligent Bear Prevention System Based on Computer Vision: An Approach to Reduce Human-Bear Conflicts in the Tibetan Plateau Area, China (https://arxiv.org/abs/2503.23178)
- **What's New**: 이번 연구에서는 컴퓨터 비전(Computer Vision)과 사물인터넷(Internet of Things, IoT) 기술을 결합한 새로운 전략을 제안합니다. 이 전략은 티베트 고원에서 인간과 곰(Ursus arctos pruinosus) 간의 갈등을 줄이기 위한 것으로, K210 개발 보드를 사용하여 효율적이고 낮은 전력 소모로 실시간 곰 인식을 가능하게 합니다. 실험 결과, 평균 정밀도(mean Average Precision, mAP) 91.4%를 달성하며 뛰어난 성능을 입증했습니다.

- **Technical Details**: 연구 방법은 두 단계로 나뉘어 이루어졌습니다. 첫 번째 단계에서는 YOLOv5 프레임워크를 활용하여 곰 감지 모델을 훈련하였고, 두 번째 단계에서는 하드웨어에 이 모델을 임베딩하여 감지 및 억제 시스템을 설계했습니다. K210 보드에 최적화된 MaixHub 플랫폼에서 데이터를 수집하고, YOLOv5 모델을 미세 조정하여 낮은 전력 하드웨어에서의 곰 인식을 최적화했습니다.

- **Performance Highlights**: 시스템은 두 가지 주요 구성 요소로 이루어져 있으며, IoT 기반 센서와 곰 억제 메커니즘을 포함합니다. 이를 통해 희망적인 조치로써, 고립된 지역에서도 지속적으로 운영할 수 있으며, 생태계 보존 및 인간 안전 강화에 기여할 수 있습니다. 이 연구는 곰과의 갈등을 줄이는 지속 가능한 솔루션을 제공하여, 중국의 유수 지역에서도 개선된 인간 안전과 곰 보호를 달성할 수 있도록 합니다.



### NeuralGS: Bridging Neural Fields and 3D Gaussian Splatting for Compact 3D Representations (https://arxiv.org/abs/2503.23162)
Comments:
          Project page: this https URL

- **What's New**: 본 논문은 새로운 압축 방법인 NeuralGS를 소개합니다. NeuralGS는 기존의 복잡한 voxel 구조나 양자화 전략 없이도 원래의 3D Gaussian Splatting (3DGS)을 간결하게 압축할 수 있는 효율적인 방법을 제공합니다. 특히, NeuralGS는 Multi-Layer Perceptron (MLP) 신경망을 사용하여 3D 가우시안의 속성을 인코딩합니다.

- **Technical Details**: NeuralGS는 중요도 점수에 따라 클러스터링 전략을 채택하고 각 클러스터에 대해 작은 MLP를 적합시킵니다. 각 클러스터는 60,000번의 반복을 통해 속성을 적합시키며, 5개 층으로 구성된 MLP를 사용하고 Tanh 활성화 함수를 적용합니다. 학습률은 Adam 옵티마이저를 통해 조정됩니다.

- **Performance Highlights**: NeuralGS는 모델 크기를 평균 45배 줄였음에도 불구하고 시각적 품질에 손상을 주지 않으며, 기존의 Scaffold-GS 기반 압축 방법과 유사한 압축 성능을 발휘합니다. 실험 결과, NeuralGS는 3DGS 기반 방법들 중 최상의 결과를 보여주며, 커뮤니티에 대한 효과적인 압축 방향을 제시합니다.



### When 'YES' Meets 'BUT': Can Large Models Comprehend Contradictory Humor Through Comparative Reasoning? (https://arxiv.org/abs/2503.23137)
- **What's New**: 이 논문에서는 복잡하고 모순적인 내러티브를 통한 유머 이해의 어려움을 다루기 위해 다국어 및 다문화적 맥락에서 발췌한 1,262개의 만화 이미지로 구성된 새로운 벤치마크인 YesBut(V2)를 도입합니다. 이 데이터셋은 내러티브 이해의 다양한 측면을 포괄하는 종합적인 주석이 포함되어 있으며, 이를 통해 대조적인 요소들 사이의 비교적 추론을 평가합니다. 본 연구는 VLMs(vision-language models)가 유머를 포함하는 시각적 대비 구조를 해석하는 데 있어 갖는 한계를 드러내며, 향후 AI의 인지적 역할을 발전시키기 위한 기초를 제공합니다.

- **Technical Details**: 논문에서 제안한 YesBut 벤치마크에는 만화 내에서 표현된 시각적 및 텍스트 구성 요소에 대한 표면적 이해에서부터 깊이 있는 내러티브 추론에 이르기까지 네 가지 보완적 작업이 포함됩니다. 각 작업은 VLM이 만화의 복잡한 내러티브를 이해하는 성과를 평가하기 위해 설계되었습니다. VLMs 는 일반적으로 시각적 인식, 주요 요소 식별, 비교 분석, 환각 등의 분야에서 인간보다 크게 뒤떨어지는 성능을 보였습니다.

- **Performance Highlights**: 실험 결과, VLM들은 내러티브 유머를 이해하는 데 있어서 상당한 한계를 보였으며, 특히 비주얼 인식 및 주요 요소 분석에서 일반적인 오류가 발생했습니다. 모델의 성능 향상을 위한 텍스트 기반 훈련 전략 및 사회적 지식 강화 방법이 제안되었고, 이를 통해 향후 연구 방향이 명확해졌습니다. YesBut의 확장 이후 성공적인 평가를 위한 새로운 벤치마크 설정은 VLM의 심층적 세멘틱 추론 발전을 이끄는 데 중요한 기초를 형성합니다.



### LSNet: See Large, Focus Sma (https://arxiv.org/abs/2503.23135)
Comments:
          CVPR 2025 Camera-ready Version

- **What's New**: 본 논문은 경량 비전 네트워크 설계의 새 패러다임을 제시하며, 인간의 효율적인 시각 시스템에서 영감을 받아 'See Large, Focus Small' 전략을 도입합니다. 여기서, LS(Large-Small) convolution을 소개하여 큰 커널을 이용한 정보를 포착하고, 작은 커널을 통한 정교한 특성 집합을 가능하게 합니다. 이러한 방식을 통해 성능과 효율성을 동시에 추구하는 새로운 경량 모델 LSNet이 개발되었습니다.

- **Technical Details**: LS convolution은 큰-작은 컨볼루션을 조합하여, 넓은 시각적 정보를 수집하고 정밀한 특징 집합을 수행합니다. LS block을 구축하여 경량 모델 LSNet을 형성하며, 이를 통해 기존의 경량 네트워크들과 비교했을 때 뛰어난 성능과 효율성을 입증하였습니다. LSNet은 다양한 비전 작업에서 사용자의 요구를 충족시킬 수 있도록 설계되었습니다.

- **Performance Highlights**: 전문가들의 연구 결과, LSNet은 기존 경량 네트워크에 비해 각종 비전 작업에서 월등한 성능을 보여주었습니다. 다양한 실험을 통해 LSNet은 경쟁 모델들에 비해 높은 효율성과 성능을 달성하며, 경량 및 효율적인 모델 설계 분야에 새로운 기준이 될 것으로 기대됩니다. LSNet의 코드 및 모델은 공개되어 연구자들이 쉽게 활용할 수 있습니다.



### RefChartQA: Grounding Visual Answer on Chart Images through Instruction Tuning (https://arxiv.org/abs/2503.23131)
Comments:
          All models and code will be publicly available at this https URL

- **What's New**: 최근 Vision Language Models (VLMs)의 발전은 문서 시각화에서의 시각적 기반(visual grounding)에 중점을 두고 있다. 특히, 차트 이미지를 통한 인간-컴퓨터 상호작용을 개선하기 위한 RefChartQA라는 새로운 벤치마크를 소개한다. 이 벤치마크는 차트 질문 응답(Chart Question Answering, ChartQA)과 VAG(Visual Answer Grounding)를 통합하여 모델이 차트 이미지 내 여러 요소를 참조할 수 있도록 한다.

- **Technical Details**: RefChartQA는 다양한 차트 형식에서 신뢰할 수 있는 텍스트와 그래픽 요소를 추출하고, 숫자 및 텍스트 데이터에 대한 해석 가능한 추론을 제공하기 위해 세 가지 주요 기여를 한다. 첫째, 실제 차트 데이터 집합을 제공하고, 둘째, 멀티모달 대규모 언어 모델을 적응시키는 instruction-tuning 전략을 제안하며, 셋째, 구체적이고 의미 있는 해답을 제공함으로써 해석 가능성을 크게 향상시킨다. 이러한 구체적 접근 방식은 복잡한 숫자 쿼리에서의 잔상(hallucination)을 줄이는 데 도움을 준다.

- **Performance Highlights**: 실험 결과, 시각적 기반을 통한 공간 인지(spatial awareness)가 통합될 경우 응답의 정확도가 15% 이상 향상됨을 보여준다. 이를 통해 모델의 신뢰성을 개선하고 복잡한 쿼리에 대한 처리 능력을 높이는 데 기여한다. 추가적으로, TinyChart에서의 아키텍처 개선과 같은 주요 요소들이 텍스트-공간 정렬(text-spatial alignment)에도 영향을 미친다는 점을 강조하였다.



### Can DeepSeek-V3 Reason Like a Surgeon? An Empirical Evaluation for Vision-Language Understanding in Robotic-Assisted Surgery (https://arxiv.org/abs/2503.23130)
Comments:
          Technical Report

- **What's New**: DeepSeek-V3는 최근에 등장한 대규모 언어 모델(LLM)로, 일반적인 장면 이해, 질문 응답(QA), 텍스트 생성 작업에서 탁월한 성능을 보여줍니다. 이 연구에서는 DeepSeek-V3의 로봇 수술 시나리오에서의 대화 능력을 조사하며, 단일 구문 QA, 시각 QA 및 상세 설명과 같은 작업에 중점을 두고 있습니다. 평가 결과, DeepSeek-V3는 특정 프롬프트를 제공할 때 수술 도구 및 조직 인식 작업에서 좋은 성능을 보이나, 공간 위치 분석에서는 제한된 능력을 보입니다.

- **Technical Details**: DeepSeek-V3는 로봇 보조 수술(RAS) 환경에서의 복잡한 시각 환경 이해와 맥락 인식 도움을 중요하게 여기는 인공지능 모델입니다. 실험은 단일 구문 QA, 시각 QA, 그리고 상세 설명의 세 가지 패러다임으로 나뉘어 진행되며, EndoVis18과 CholecT50 데이터셋을 사용하여 성능을 평가합니다. 각 패러다임은 수술에 관련된 비주얼 질문 응답(VQA) 모델의 성능을 체계적으로 분석하기 위해 설계되었습니다.

- **Performance Highlights**: DeepSeek-V3는 EndoVis18 데이터셋에서 전체적으로 더 나은 성과를 보였으며, 단일 구문 QA에서는 모델이 질문 이해와 정보 처리에서 우수한 성과를 나타냈습니다. 그러나 수술 도구의 동작 및 공간 위치 분석에서는 여전히 한계를 보여주었으며, 깊이 있는 설명이 필요한 작업에서 낮은 성능을 보였습니다. CholecT50 데이터셋에서도 비슷한 성능을 발휘했지만, 수술 조직 분석은 두 모델 모두에게 가장 큰 도전 과제가 되고 있습니다.



### Evaluating Compositional Scene Understanding in Multimodal Generative Models (https://arxiv.org/abs/2503.23125)
- **What's New**: 이 연구는 최신 텍스트-이미지 모델인 DALL-E 3와 멀티모달 비전-언어 모델들이 복합적인 시각 장면을 이해하고 생성하는 능력을 평가합니다. 이전 세대의 모델들에 비해 현 모델들이 관계(relational) 작업을 더 잘 수행하는 경향이 있음을 보여주지만, 인간 참가자들의 성과에는 미치지 못하는 결과도 관찰되었습니다.

- **Technical Details**: 연구에서는 첫 번째 섹션에서 DALL-E 3의 공간 관계를 기반으로 한 이미지 생성 능력을 평가하고, 두 번째 섹션에서는 GPT-4와 같은 멀티모달 비전-언어 모델들의 관계 패턴 추론 능력을 분석합니다. 특히 복합적인 관계를 포함하는 다양한 프롬프트로 이 모델들을 테스트하여 그 성능을 확인했습니다.

- **Performance Highlights**: DALL-E 3는 이전 모델에 비해 관계 프롬프트에서 개선된 성능을 나타냈지만, 비일상적인 시나리오나 복잡한 관계가 포함된 프롬프트에 대해서는 성능이 하락했습니다. evaluated 모델들은 많은 객체들 (>5개)과 관련된 문제에서 인간 참가자들과 비교해 성과가 낮았으며, 이러한 결과는 현재 모델들이 시각 장면의 구성적 이해(compositional understanding)에서 더 많은 발전이 필요함을 시사합니다.



### Efficient Explicit Joint-level Interaction Modeling with Mamba for Text-guided HOI Generation (https://arxiv.org/abs/2503.23121)
Comments:
          Accepted to ICME 2025

- **What's New**: 본 논문에서는 텍스트 기반의 인간-객체 상호작용(Human-Object Interaction, HOI) 생성을 위한 새로운 접근 방식을 제안합니다. 제안된 Efficient Explicit Joint-level Interaction Model (EJIM)은 계산 효율성 있게 명시적인 관절 수준 상호작용 모델링을 수행합니다. 기존 방법들이 전체 인간 몸체를 단일 토큰으로 처리함으로써 세밀한 관절 수준 상호작용을 포착하는 데 한계를 보였던 반면, EJIM은 효율적인 이중 분기 구조로 이러한 문제를 해결하고 있습니다.

- **Technical Details**: EJIM은 Dual-branch HOI Mamba(DHM)와 Dual-branch Condition Injector(DCI)를 도입합니다. DHM은 인간 관절 관계를 더 잘 캡처하기 위해 척추 지향 공간 스캔 기법을 사용합니다. DCI는 텍스트의 의미와 객체 기하학을 인간과 객체의 동작에 통합하여 관절 수준에서 효율적인 상호작용 모델링을 가능하게 합니다. 또한 동적 상호작용 블록(Dynamic Interaction Block, DIB)과 점진적 마스킹 메커니즘을 통해 더욱 정확한 모델링을 보장합니다.

- **Performance Highlights**: EJIM은 BEHAVE 및 OMOMO 데이터셋에서 기존 방법들보다 훨씬 나은 모션 및 상호작용 생성 품질을 보이며, 추론 시간의 5%만으로 이를 달성합니다. 다양한 양적 및 질적 평가를 통해 EJIM의 효과성과 효율성을 입증하였고, 이를 통해 기존 모델들을 대폭 초월하는 성과를 보여줍니다. 또한, 논문에서는 모델 이해를 돕기 위한 다양한 절제 연구와 시각화를 제공합니다.



### Uncertainty-Instructed Structure Injection for Generalizable HD Map Construction (https://arxiv.org/abs/2503.23109)
Comments:
          17 pages, 10 figures

- **What's New**: 최근 자율주행 차량의 안전을 보장하기 위해 신뢰할 수 있는 고해상도(HD) 지도 구축의 중요성이 강조되고 있습니다. UIGenMap은 통계적 분포에서 불확실성을 재샘플링하고 훈련 데이터에 대한 과도한 의존도를 줄여주기 위한 구조 삽입 접근법을 제안합니다. 특히, 새로운 관점 검출(branch)을 도입하여 구조적 정보를 명확히 하고, 모델의 일반화능력을 향상시키는 방안을 모색합니다.

- **Technical Details**: UIGenMap은 불확실성을 인식하는 디코더가 동적으로 장면별 확률 분포를 샘플링하도록 설계되었습니다. 이 모델은 UI2DPrompt라는 기술을 통해 학습 가능한 PV 프롬프트를 구성하고 이를 혼합 주입(hybrid injection) 기법으로 지도 디코더에 통합합니다. 이 방식은 BEV 공간에 신뢰할 수 있는 구조적 표현을 주입하여 복잡한 환경에서도 강력한 퍼셉션 능력을 발휘할 수 있도록 합니다.

- **Performance Highlights**: UIGenMap은 nuScenes 데이터셋에서 지리적으로 분리된 데이터 파티션을 사용하여 평가한 결과, 특히 도시 기반 분할에서 +5.7 mAP의 개선을 보이며 최첨단 성능을 달성했습니다. 이로 인해 기존 방법에 비해 더 뛰어난 적응성과 일반화 능력을 입증하였습니다. 또한, 경량화된 질의 증류(Mimic Query Distillation) 모듈을 통해 실시간 추론이 가능하도록 구현되었습니다.



### A large-scale image-text dataset benchmark for farmland segmentation (https://arxiv.org/abs/2503.23106)
- **What's New**: 이번 논문에서는 전통적인 딥러닝 패러다임의 한계를 극복하기 위해 언어 기반의 학습 패러다임을 제안합니다. 특히 농지의 시공간적(spatiotemporal) 특징을 명확하게 표현할 수 있는 언어의 역할을 강조하고 있습니다. 이를 통해 농지의 동적 시간적 진화와 공간 이질성을 효과적으로 모델링할 수 있습니다. 새로운 FarmSeg-VL 데이터셋을 창출함으로써 이 분야의 연구에 필요한 기본적인 벤치마크 데이터셋을 제공합니다.

- **Technical Details**: FarmSeg-VL은 농지에 대한 언어 기반의 설명을 포함한 최초의 상세 이미지-텍스트 데이터셋으로, 농지의 시공간적(spatiotemporal) 특징을 지원합니다. 또한, 이 논문에서는 각 이미지에 대한 설명(caption)을 정확하게 부여하는 반자동(annotation) 방법을 개발하여 데이터의 질과 의미적 풍부성을 보장합니다. 데이터셋은 8개의 전형적인 농업 지역을 포함하여 공간 차원에서의 다양성을 갖추고 있습니다.

- **Performance Highlights**: FarmSeg-VL로 훈련된 VLMs(비전-언어 모델)와 라벨에만 의존하는 딥러닝 모델의 성능 분석을 제시하였습니다. 이를 통해 FarmSeg-VL이 농지 분할을 위한 표준 벤치마크로서의 잠재력을 시연하는 결과를 보여줍니다. 또한 농지의 고유한 속성, 단계적(pheno-logical) 특성, 공간 분포 등의 풍부한 시공간적 특성을 포괄합니다.



### Open-Vocabulary Semantic Segmentation with Uncertainty Alignment for Robotic Scene Understanding in Indoor Building Environments (https://arxiv.org/abs/2503.23105)
Comments:
          32 pages, 7 figures

- **What's New**: 이 논문에서는 장애인을 위한 자율 보조 로봇의 필요성이 증가하고 있음을 강조하고 있습니다. 이를 위한 새로운 접근 방식으로, 기존의 폐쇄 어휘 시스템의 한계를 극복하고, 개방형 어휘( open-vocabulary ) 장면 의미 분할 및 탐지 파이프라인을 제안합니다. 이 방법은 Vision Language Models (VLMs)와 Large Language Models (LLMs)을 활용하여 보다 효과적인 내비게이션을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 'Segment Detect Select' 프레임워크를 따릅니다. 지역을 분할하고, 그 지역을 탐지하며, 사용자 특정 요구에 맞춰 선택하는 과정을 통해 보다 자연스러운 내비게이션을 실현합니다. 이 과정에서 공간을 구분하고 의미를 인식하는 기능이 필수적이며, 이는 복잡한 환경에서 로봇의 가능성을 향상시킵니다.

- **Performance Highlights**: 기존 접근 방식의 제한을 벗어나, 새로운 시스템은 격리된 환경에서의 성공률을 높이는데 기여할 것으로 기대됩니다. 불확실한 장면 인식 문제를 해결하고, 모호한 환경에서도 정확한 인식을 가능케 합니다. 이는 장애인을 위한 보조 로봇의 자동차 기동성을 개선하고, 자율성을 높이는 데 중요한 역할을 할 것입니다.



### FRAME: Floor-aligned Representation for Avatar Motion from Egocentric Video (https://arxiv.org/abs/2503.23094)
Comments:
          Accepted at CVPR 2025

- **What's New**: 본 연구에서는 VR 및 AR 애플리케이션을 위한 헤드 마운트 바디-facing 스테레오 카메라를 활용한 자아 중심의 (Egocentric) 모션 캡쳐 기술의 한계를 극복하고자 한다. 특히, 무거운 가림 현상(occlusions)과 제한된 주석(real-world) 데이터 문제를 해결하기 위해 경량화된 VR 기반 데이터 수집 설정을 도입하였다. 이를 통해 지금까지의 ego-facing 카메라를 위한 가장 방대한 실제 데이터셋을 수집했다.

- **Technical Details**: 제안된 FRAME 아키텍처는 장치 자세(device pose)와 카메라 피드를 결합하여 최첨단의 바디 포즈 예측을 가능하게 한다. 각 데이터 출처의 특성이 상이하여 통합이 쉽지 않았으나, 기하학적으로 (geometrically) 타당한 다중 모달(multimodal) 통합 방식을 통해 이를 해결하였다. 이 프로세스는 현대 하드웨어에서 300 FPS(frame per second)로 실행될 수 있다.

- **Performance Highlights**: 제안된 접근 방식은 모델의 일반화(generalization) 능력을 향상시키기 위한 새로운 훈련 전략을 채택하였다. 기하학적 특성을 활용하여 과거 연구의 일반적인 아티팩트(artifacts)에서 자유로운 고품질 모션 캡쳐를 생성할 수 있었다. 질적 및 양적 평가뿐만 아니라 광범위한 비교를 통해 본 방법의 효과성을 입증하였다.



### Efficient Adaptation For Remote Sensing Visual Grounding (https://arxiv.org/abs/2503.23083)
- **What's New**: 이번 연구에서는 Parameter Efficient Fine Tuning (PEFT) 기법을 적용하여 원격 탐사(remote sensing, RS) 작업에 적합하도록 Grounding DINO와 OFA 모델을 조정했습니다. 연구 결과, LoRA 기법을 통해 DIOR-RSVG 데이터세트에서 최상위 성능을 달성했으며, Adapter와 BitFit 기술을 비교한 결과 Adapter가 고성능을 보였습니다. 이 연구는 PEFT 기법의 가능성을 강조하며, 전체 모델 학습의 비용 대비 효율적인 대안을 제시합니다.

- **Technical Details**: Parameter Efficient Fine Tuning (PEFT)은 모델의 최소한의 파라미터 집합만을 조정하여 계산 효율성을 제공합니다. Adapters는 경량 모듈로, 사전 훈련된 모델의 레이어 사이에 삽입되어 특정 작업 학습을 위한 추가 파라미터를 도입합니다. LoRA는 모델의 가중치 행렬에 저랭크 업데이트를 적용하여 학습 가능한 파라미터 수를 줄이고, BitFit은 모델 레이어의 편향 항만을 미세 조정하는 방법입니다.

- **Performance Highlights**: 실험 결과, Grounding DINO는 SOTA VG 모델로서, 텍스트 프롬프트와 이미지 내 특정 영역을 연결하는데 있어 뛰어난 성능을 발휘했습니다. Multi-scale deformable attention이 다양한 공간 해상도와 이미지 특징의 통합을 용이하게 하여 정확한 객체 탐지와 구문 로컬라이제이션을 가능하게 했습니다. 또한 모델의 계산 효율을 높이기 위해 고정 파라미터 비율을 측정하여 PEFT 기술을 통한 비용 절감 효과를 평가했습니다.



### InkFM: A Foundational Model for Full-Page Online Handwritten Note Understanding (https://arxiv.org/abs/2503.23081)
- **What's New**: 이 논문에서는 손으로 쓴 디지털 노트의 내용을 정확하게 해석하고 이해하는 방법을 개발하기 위한 새로운 모델인 InkFM을 소개합니다. InkFM은 28종의 스크립트에서 텍스트를 인식하고, 수학적 표현을 인식하며, 페이지를 텍스트와 그림 같은 개별 요소로 구분할 수 있는 고유한 기능을 제공합니다. 특정 데이터셋에 대해 소스 모델을 미세 조정(fine-tuning)하여 페이지 세분화와 텍스트 인식의 품질을 더욱 향상시킬 수 있음을 입증했습니다.

- **Technical Details**: InkFM은 다채로운 혼합 작업에 대해 훈련되어 있으며, 세 가지 핵심 작업인 세분화(segmentation), 분류(classification), 인식(recognition)을 통합하여 하나의 강력한 모델로 발전시킵니다. 세분화 작업에서는 단어와 그림을 어떤 객체에 할당하고 각 객체를 분류하는 기술이 포함됩니다. 또한, 이는 손으로 쓴 텍스트를 문자 시퀀스로 변환하는 인식 작업을 통해, 다양한 글쓰기 스타일에 적응할 수 있는 능력을 보여줍니다.

- **Performance Highlights**: InkFM은 다양한 스크립트에서의 텍스트 인식 및 스케치 분류에서 뛰어난 성능을 보여주며, 특히 잉크 기반 스타일 작업에서 SoTA(state-of-the-art) 품질을 달성했습니다. 예를 들어, IAM 데이터셋에서 영어 손글씨 텍스트 라인 세분화에서 경쟁력 있는 결과를 보였고, QuickDraw 데이터셋에서는 최신 정확도를 기록했습니다. 이 모델은 고유한 성능으로 시각적 요소를 탐지하고, 노트의 개별 요소를 효과적으로 구분할 수 있습니다.



### VGRP-Bench: Visual Grid Reasoning Puzzle Benchmark for Large Vision-Language Models (https://arxiv.org/abs/2503.23064)
Comments:
          8 pages

- **What's New**: 대형 비전-언어 모델(LVLMs)이 시각적 퍼즐을 해결하는 데 어려움을 겪고 있다는 점에 주목하여, VGRP-Bench라는 새로운 벤치마크를 소개했다. 이 벤치마크는 20개의 다양한 퍼즐로 구성되어 있으며, 고유의 난이도와 규칙을 가진 퍼즐을 통해 LVLM의 구조적 추론 능력을 평가한다. 이번 연구는 LVLM의 퍼즐 해결 성능 향상을 위한 체계적인 전략과 실험을 제공하여, 실제 문제 해결 능력을 분석하는 데 기여하고자 한다.

- **Technical Details**: 본 논문에서는 LVLM의 퍼즐 해결 과정을 평가하기 위해 시각적 그리드 추론 퍼즐 벤치마크인 VGRP-Bench를 제안하고 있다. 퍼즐은 난이도에 따라 기본, 중간, 고급으로 나뉘며, LVLM의 인지 능력, 규칙 준수, 그리고 퍼즐 해결 능력에 대한 세부적인 실험을 수행한다. 특히, 솔루션 감독 미세 조정(S-SFT) 및 추론 미세 조정(R-SFT)이라는 두 가지 후속 훈련 방식이 LVLM의 퍼즐 해결 능력 개선에 효과적임을 보여준다.

- **Performance Highlights**: 실험 결과, 최신 LVLM 모델들도 기본적인 4×4 스도쿠 퍼즐을 일관되게 해결하지 못하는 것으로 나타났다. 각 LVLM의 퍼즐 해결 성능에 영향을 미치는 다양한 요소들, 예를 들어 단서의 수, 그리드 크기, 그리고 규칙의 복잡성을 분석하였다. 두 가지 후속 훈련 전략을 도입한 결과, 퍼즐 해결 성능이 향상되었으나, 단지 훈련된 퍼즐에 대한 일반화에는 한계가 있음을 확인하였다.



### Shape and Texture Recognition in Large Vision-Language Models (https://arxiv.org/abs/2503.23062)
- **What's New**: LAS&T 데이터셋(대형 형상 및 질감 데이터셋)의 출시로 기존의 형상 및 질감 인식 연구에 혁신을 가져왔습니다. 이 데이터셋은 실제 이미지에서 자동으로 추출된 다양한 형상과 질감을 포함하는 방대한 컬렉션입니다. 이 연구는 선도적인 대형 비전-언어 모델(LVLMs)이 2D 및 3D 장면에서 형상, 질감 및 자재를 이해하는 능력을 평가하고 있습니다.

- **Technical Details**: 형상 인식 실험에서는 모델이 방향, 질감, 색상, 환경이 다른 동일한 형상을 맞추는 능력을 검사했습니다. LVLMs는 고수준의 의미적 피처에 주로 의존하며, 명확한 클래스 연관성이 없는 추상 형상에서의 인식에 어려움을 겪고 있습니다. 질감과 자재 인식 평가에서는 다양한 객체와 환경에서 동일 질감과 자재를 식별하는 능력을 측정했습니다.

- **Performance Highlights**: 결과적으로 LVLM의 형상 식별 능력은 여전히 인간 성능에 비해 상당히 낮은 것으로 나타났습니다. 특이하게도, LVLMs는 3D 장면에서 자재 인식에서 인간 수준의 성능에 근접하지만, 간단한 2D 질식 인식에서 인간보다 크게 뒤떨어졌습니다. LAS&T 데이터셋과 벤치마크는 형상 및 질감 평가를 위한 가장 크고 다양한 자원으로, 무료로 제공됩니다.



### CityGS-X: A Scalable Architecture for Efficient and Geometrically Accurate Large-Scale Scene Reconstruction (https://arxiv.org/abs/2503.23044)
Comments:
          Project page: this https URL

- **What's New**: CityGS-X는 기존의 복잡한 병합 및 분할 과정을 버리고 새로운 배치 수준의 다중 작업 렌더링 과정을 도입한 확장 가능한 아키텍처입니다. 이 혁신적인 설계는 동적 수준(Level-of-Detail) voxel 할당을 통해 여러 GPU에서 효율적인 렌더링을 가능하게 하여 성능과 확장성을 크게 향상시킵니다. 4개의 4090 GPU만으로도 5,000장 이상의 이미지를 5시간 만에 훈련 및 렌더링 할 수 있어 기존 방법들에 비해 월등한 능력을 보여줍니다.

- **Technical Details**: CityGS-X는 Parallelized Hybrid Hierarchical 3D Representation (PH^2-3D)를 기반으로 하여 다중 GPU의 병렬 훈련을 가능하게 하는 아키텍처입니다. 이 아키텍처는 각 GPU에서의 voxel-wise 병렬 처리를 통해 자동 로드 밸런싱을 이루며, 공유 Gaussian 디코더를 통해 분산 Gaussian으로 디코딩을 사용합니다. 또한, 배치 렌더링 기법을 통해 RGB, 깊이 및 노멀 같은 다중 작업을 효율적으로 수행하며, 배치 수준의 일관된 기하학적 제약을 통합하여 품질과 기하학적 정확도를 향상시킵니다.

- **Performance Highlights**: CityGS-X는 대규모 장면 데이터세트에 대한 실험에서 뛰어난 렌더링 품질과 빠른 훈련 속도를 입증하며, 다양한 RTX 4090 GPU 구성에서 효과적으로 작동할 수 있음을 보여줍니다. Out-Of-Memory(OOM) 문제 없이 상대적으로 저사양의 GPU에서도 원활하게 실행되며, 기존의 대규모 장면 재구성 방법들에 비해 우수한 성능을 보입니다. 이 연구는 향후 3D 장면 재구성 기술의 발전에 있어 중요한 토대를 마련합니다.



### STSA: Spatial-Temporal Semantic Alignment for Visual Dubbing (https://arxiv.org/abs/2503.23039)
Comments:
          Accepted by ICME 2025

- **What's New**: 이 논문에서는 Spatial-Temporal Semantic Alignment (STSA)라는 새로운 방법을 제안합니다. 이 방법은 공간(domain)과 시간(domain)에서의 의미적 특징을 정렬하여 동적 얼굴의 합성 안정성을 향상시키는 데 초점을 맞추고 있습니다. 이 논문은 소리 기반의 시각적 더빙이 현재의 기술로서의 한계를 극복하도록 도와줄 수 있는 가능성을 보여주고 있습니다.

- **Technical Details**: STSA는 이중 경로 정렬 메커니즘을 도입하여 공간 및 시간 도메인에서 여러 크기의 특징을 정렬합니다. Consistent Information Learning (CIL) 모듈을 통해 다중 스케일에서의 상호 정보를 최대화하여, 잘못 정렬된 정보를 교정할 수 있습니다. 또한, 확률적 히트맵을 사용하여 의미적 불확실성을 허용하는 방향성을 제공하여, 합성된 얼굴의 움직임이 부드럽게 유지되도록 합니다.

- **Performance Highlights**: 실험 결과, STSA 방법은 이미지 품질과 합성 안정성 면에서 우수성을 입증하였습니다. 전처리된 가중치와 추론 코드는 제공되며, 이로 인해 연구자들이 쉽게 접근하고 사용할 수 있을 것입니다. STSA의 도입을 통해 동작이 자연스럽고 현실적인 시각적 더빙이 이루어질 수 있음을 보여주고 있습니다.



### FreeInv: Free Lunch for Improving DDIM Inversion (https://arxiv.org/abs/2503.23035)
- **What's New**: 이번 연구에서는 FreeInv라는 새로운 방법을 제안하여 DDIM (Denoising Diffusion Implicit Models) 역전 과정의 경로 편차 문제를 효과적으로 해결하였습니다. 기존의 방법들은 학습 또는 복잡한 보상 전략을 사용하여 이 문제를 완화하려 했지만 시간과 계산 비용이 많이 들었습니다. FreeInv는 통계적 관점에서 여러 경로의 앙상블을 통해 경로 불일치를 줄이는 방식을 채택하였습니다.

- **Technical Details**: FreeInv는 잠재 표현(latent representation)을 무작위로 변환한 후 역전과 재구성 과정에서 동일한 변환을 유지합니다. 이 과정을 통해 예측된 노이즈를 평균내어 경로 불일치를 줄이는 방식으로 작동합니다. 이 방법은 기존의 이미지 편집 프레임워크에 쉽게 통합될 수 있으며, 특히 비디오 역전에 있어 더 높은 정확성과 효율성을 제공합니다.

- **Performance Highlights**: 포괄적인 양적 및 질적 평가는 PIE 벤치마크와 DAVIS 데이터 세트에서 FreeInv가 전통적인 DDIM 역전 방식보다 월등히 뛰어난 성능을 보여주었음을 확인하였습니다. FreeInv는 기존의 최첨단 역전 방법들과 비교했을 때 경쟁력을 갖추고 있으며, 뛰어난 계산 효율성을 자랑합니다. 이 연구는 이미지 및 비디오 편집에서 높은 재구성 충실도와 편집 결과 향상을 이끌어낼 수 있는 가능성을 보여줍니다.



### Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning (https://arxiv.org/abs/2503.23030)
Comments:
          Accepted by CVPR2025

- **What's New**: 이번 논문에서는 Generalized Zero-Shot Learning (GZSL)을 위한 새로운 시각적 및 의미적 프롬프트 협업 네트워크(VSPCN)를 제안하고 있습니다. 기존의 방법들과는 달리, 저자들은 시각적 프롬프트와 의미적 프롬프트를 동시에 학습하여 효과적으로 비주얼 피쳐를 적응시키는 방법을 채택하였습니다. 이를 통해 더욱 분별력 있는 시맨틱 관련 비주얼 피쳐를 얻을 수 있습니다.

- **Technical Details**: 제안된 방법에서는 약한 프롬프트 융합 메커니즘과 강한 프롬프트 융합 메커니즘을 설계하여 네트워크의 얕은 층과 깊은 층에서 정보를 적절히 융합합니다. VSPCN은 시각적 정보와 의미적 정보를 통합하여 더 나은 시맨틱 정렬을 달성합니다. 이러한 프롬프트 협업을 통해, 각종 GZSL 데이터셋에서 탁월한 성능을 보여줍니다.

- **Performance Highlights**: 실험 결과, 제안된 VSPCN은 기존의 가장 최신 방법들과 비교하여 세 가지 GZSL 벤치마크 데이터셋에서 최고의 성능을 기록하였습니다. 이는 해당 방법이 기존의 방법들보다 더 효과적인 시맨틱 관련 비주얼 피쳐 학습에 기여하고 있음을 시사합니다. 이러한 성과는 즉각적으로 GZSL 분야에서의 연구와 응용 가능성을 보여줍니다.



### Empowering Large Language Models with 3D Situation Awareness (https://arxiv.org/abs/2503.23024)
Comments:
          Accepted by CVPR 2025

- **What's New**: 이번 연구에서는 기존의 LLM(대형 언어 모델)을 3D 장면 이해에 적용하는 데 있어 중요하게 간과된 관찰자의 시점(egocentric perspective)을 고려하는 새로운 접근 방식인 View2Cap을 제안합니다. 이를 통해 자동으로 상황 인식 데이터를 생성하고, Vision-Language Models (VLMs)을 활용하여 고품질의 캡션과 질문-답변 쌍을 생성합니다. 또한, 상황 지침 모듈(Situation Grounding module)을 도입하여 LLMs가 3D 장면 내에서 상황 설명을 명확히 구분할 수 있도록 합니다.

- **Technical Details**: View2Cap는 RGB-D 비디오에서 수집된 3D 스캔으로부터 카메라 궤적을 활용하여 데이터 세트를 생성합니다. 이는 점군(point cloud)과 텍스트 설명 간의 연관성을 제공하여 동적인 3D 환경 내에서 상황 맥락을 포착할 수 있도록 합니다. 상황 지침 모듈은 기존의 3D LLM 아키텍처에 통합되어 모델이 각 객체의 상대적 위치와 방향을 예측할 수 있도록 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 우리의 방법이 기존 LLM의 3D 상황 인식을 크게 향상시키고 다양한 작업에서 성능을 향상시키는 것을 입증하였습니다. View2Cap 데이터세트를 사용하여 수작업 주석이나 3D 레이블 없이도 효과적으로 데이터 생성 비용을 줄이고, 풍부한 문맥 정보를 다룰 수 있도록 지원합니다.



### MeshCraft: Exploring Efficient and Controllable Mesh Generation with Flow-based DiTs (https://arxiv.org/abs/2503.23022)
- **What's New**: 본 논문에서는 MeshCraft라는 새로운 메쉬 생성 프레임워크를 도입하여 연속적 공간 확산(continuous spatial diffusion)을 활용하여 고효율적이고 제어 가능한 3D 메쉬 생성을 가능하게 합니다. 기존의 메소드에 비해 생성 속도를 크게 향상시켜 800 페이스 메쉬를 단 3.2초 만에 생성할 수 있으며, 이는 35배 더 빠른 속도입니다. 이렇게 개선된 기술은 아티스트들이 메쉬 생성에 소요되는 많은 수작업에서 벗어날 수 있도록 돕습니다.

- **Technical Details**: MeshCraft는 두 가지 핵심 구성요소로 구성됩니다: 1) 원시 메쉬를 연속적인 페이스 레벨 토큰으로 인코딩하고 다시 원래 메쉬로 디코딩하는 transformer 기반 VAE(Variational Auto-Encoder), 2) 페이스 수에 따라 조건화된 흐름 기반 확산 변환기(flow-based diffusion transformer)입니다. 이 과정을 통해 전체 메쉬 토폴로지를 동시에 생성할 수 있어, 메쉬의 품질과 생성 속도를 크게 개선할 수 있습니다. 또한, 네이티브 메쉬 생성을 구현하는 고유한 접근 방식을 채택하여, 저차원 연속 잠재 공간에서 메쉬를 모델링합니다.

- **Performance Highlights**: MeshCraft는 ShapeNet 데이터셋에서 최신 기술들을 초월하여 질적 및 양적 평가에서 우수한 성능을 입증하였으며, Objaverse 데이터셋에서도 보다 나은 성능을 보여줍니다. 또한, 생성되는 메쉬의 페이스 수에 대한 제어 기능이 있어 사용자 친화적인 조작이 가능합니다. 이 시스템은 생성 과정에서 큰 유연성과 효율성을 제공하여, 다양한 3D 객체 생성을 위한 실질적인 응용 가능성을 보여줍니다.



### The impact of tissue detection on diagnostic artificial intelligence algorithms in digital pathology (https://arxiv.org/abs/2503.23021)
Comments:
          25 pages, 2 tables, 3 figures, 1 supplementary figure

- **What's New**: 본 연구에서는 디지털 병리학(application of digital pathology)에서 조직 검출(tissue detection)의 중요성을 강조하며, 조직 검출 알고리즘의 세부사항이 잘 보고되지 않음을 언급합니다. Poor segmentation algorithm은 다운스트림 성능(downstream performance)에 병목 현상을 일으킬 수 있으며, 진단적으로 중요한 부분이 분석에서 제외될 경우 환자 안전을 위협할 수 있습니다. 따라서 AI 기반의 조직 검출 방식과 고전적인(thresholding) 방식을 비교하여, 성능 차이를 조사하는 것을 목표로 합니다.

- **Technical Details**: 연구진은 두 가지 조직 검출 알고리즘(thresholding and UNet++)을 사용하여 전립선 암의 Gleason grading을 위한 AI 모델을 훈련하였습니다. 총 33,823개의 전체 슬라이드 이미지(Whole Slide Images, WSIs)가 5개의 디지털 병리학 스캐너에서 스캔되어 AI 모델의 훈련에 사용되었습니다. 이후, 다운스트림 Gleason grading 알고리즘은 13개의 임상 사이트에서 스캔된 70,524개의 WSIs를 활용하여 훈련 및 테스트되었습니다.

- **Performance Highlights**: 전통적인(thresholding-based) 조직 검출에서 AI 기반 조직 검출로 전환함에 따라 전체적으로 발견되지 않은 조직 샘플 수가 116개(0.43%)에서 22개(0.08%)로 감소했습니다. 두 알고리즘 모두에서 조직 검출이 가능했던 슬라이드에서는 Gleason grading의 전반적인 성능 차이는 발견되지 않았습니다. 그러나 악성 슬라이드의 3.5%에서 AI 등급의 임상적으로 중요한 변동이 관찰되어, 진단 AI의 최적 성능을 위해서는 강력한 조직 검출이 중요함을 강조합니다.



### Multi-label classification for multi-temporal, multi-spatial coral reef condition monitoring using vision foundation model with adapter learning (https://arxiv.org/abs/2503.23012)
- **What's New**: 이번 연구에서는 DINOv2 비전 파운데이션 모델과 Low-Rank Adaptation (LoRA) 미세 조정 방법을 통합하여 산호초 생태계의 조건을 다중 레이블로 분류하는 새로운 접근법을 소개합니다. 이 방법은 태국 코 타오에서 실시된 잠수 조사로 수집된 다중 시계열 필드 이미지를 활용하였으며, 모든 이미지는 시민 과학 기반 보존 프로그램에서 사용되는 보편적인 기준에 따라 레이블이 붙여졌습니다.

- **Technical Details**: DINOv2-LoRA 모델은 기존의 전통적인 모델에 비해 64.77%의 일치율을 달성하여 60.34%의 정확도를 보인 최고의 전통 모델보다 우수한 성능을 보여주었습니다. 또한 LoRA를 통합함으로써 훈련 가능한 매개변수를 1,100M에서 5.91M로 줄였으며, 다양한 시계열 및 공간적 설정에서의 전이 학습 실험을 통해 DINOv2-LoRA의 뛰어난 일반화 가능성을 확인했습니다.

- **Performance Highlights**: 이 연구는 필드에서 수집된 다중 시계열 이미지를 기반으로 산호초 조건을 효과적으로 분류할 수 있는 방법을 제안하며, 이는 산호초 생태계의 모니터링, 보존 및 관리 도구로 활용될 수 있습니다. 이번 접근법은 다중 시계열 및 다중 공간 설정에서의 효율적인 파운데이션 모델 적응을 탐구한 최초의 연구로, 산호초 상태 분류의 발전에 이바지할 것입니다.



### On Geometrical Properties of Text Token Embeddings for Strong Semantic Binding in Text-to-Image Generation (https://arxiv.org/abs/2503.23011)
- **What's New**: 이번 논문에서는 Text-to-Image (T2I) 모델의 복잡한 장면에서 발생하는 텍스트-이미지 불일치 문제를 해결하기 위해 선택한 방법, 즉 	extbf{TeeMo} 프레임워크를 소개합니다. TeeMo는 텍스트 임베딩을 활용하여 강력한 의미적 결합(semantic binding)을 가능하게 하며, 기존 방법들보다 더 나은 성능을 보여줍니다. 이 방법은 미세 조정 없이도 다양한 데이터셋에서 높은 성능을 발휘하는 특징을 가지고 있습니다.

- **Technical Details**: TeeMo는 Causality-Aware Projection-Out (CAPO)와 Adaptive Token Mixing (ATM)으로 구성되어 있습니다. CAPO는 상호 텍스트 토큰 간의 CA 맵을 구분하는 데 도움을 주며, ATM은 손실 함수를 통해 서로 다른 Noun Phrase (NP) 간의 분리를 강화하면서도 내부 NP 간 결속성을 유지합니다. 연구를 통해 텍스트 토큰 임베딩의 기하학적 특성, 특히 각도 거리와 노름이 CA 맵의 차별화에 중요한 역할을 한다는 것을 경험적 및 이론적으로 분석하였습니다.

- **Performance Highlights**: TeeMo는 다양한 기준과 데이터셋에서 기존 방법보다 항상 우수한 성능을 나타냈습니다. 실험 결과는 TeeMo의 능력이 다양한 복잡한 장면에서 더 나은 텍스트-이미지 정렬을 가능하게 함을 보여줍니다. 이 프레임워크는 특히 여러 객체와 속성이 포함된 장면에서 그 효과를 극대화하는 것으로 확인되었습니다.



### FreeSplat++: Generalizable 3D Gaussian Splatting for Efficient Indoor Scene Reconstruction (https://arxiv.org/abs/2503.22986)
- **What's New**: 최근 3D Gaussian Splatting(3DGS)에 효율적인 feed-forward 방식을 통합하는 연구가 활발히 진행되고 있습니다. 그러나 기존 방법의 대부분은 소규모 지역의 희소 뷰 재구성에 초점을 맞추고 있어, 품질이나 효율성 측면에서 적절한 전체 장면 재구성을 제공하지 못합니다. 본 논문에서는 FreeSplat++를 제안하며, 이는 대규모 실내 전체 장면 재구성을 위한 대신 가능한 접근법으로, 재구성 속도를 크게 가속화하고 기하학적 정확성을 향상시킬 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: FreeSplat++는 전체 장면 재구성을 지원하기 위해 저비용 Cross-View Aggregation 프레임워크를 처음으로 제안하여 긴 입력 시퀀스를 효율적으로 처리할 수 있게 합니다. 또한, 다수의 뷰에서 겹치는 3D Gaussian 프리미티브를 점진적으로 집계하는 설계된 픽셀 단위의 triplet fusion 방법을 도입하여 중복성을 적절히 줄입니다. 마지막으로, 전체 장면 재구성에 필수적인 명확한 깊이 융합 접근 방식을 통해 floaters(부유물)를 효과적으로 줄이는 가중치 제거 전략을 제안합니다.

- **Performance Highlights**: 본 논문은 FreeSplat++가 기존의 일반화된 3DGS 방법을 현저하게 초월함을 입증하는 광범위한 실험을 실시하였습니다. 특히, 전체 장면 재구성 측면에서 우리의 방법은 유의미한 정확도 향상과 훈련 시간 단축을 보여 주었습니다. 깊이 정규화된 장면별 미세 조정을 통해 얻은 결과는 재구성 정확도와 효율성에서 상당한 개선을 이루어냈습니다.



### Optimal Transport-Guided Source-Free Adaptation for Face Anti-Spoofing (https://arxiv.org/abs/2503.22984)
Comments:
          15 pages, 7 figures

- **What's New**: 이 논문에서는 고객이 소량의 얼굴 데이터 샘플만을 사용하여 얼굴 안티 스푸핑 모델을 테스트 시간에 쉽게 조정할 수 있는 새로운 방법론을 소개합니다. 기존의 모델에서는 광범위한 훈련 데이터셋이 필요했지만, 본 연구는 이 문제를 해결하기 위해 소스 훈련 데이터나 모델 파라미터를 공유할 필요 없이 적은 양의 데이터로도 사용자 특화 조정을 가능하게 합니다.

- **Technical Details**: 우리는 기본 모델을 프로토타입 기반으로 설계하고, 최적 수송(Optimal Transport) 방식의 어댑터를 개발하여 경량 훈련 또는 훈련 없는 방식으로 조정이 가능하도록 하였습니다. 훈련 무료 접근 방식에서는 소스 프로토타입의 특성을 타겟 도메인으로 매핑하기 위해 최적 수송 변환을 적용하여, 모델이 타겟 도메인의 구조와 특징을 잡아낼 수 있도록 합니다.

- **Performance Highlights**: 저희 방법은 크로스 도메인 및 크로스 공격 설정에서 기존의 최신 방법들과 비교했을 때 평균적으로 HTER에서 19.17%, AUC에서 8.56%의 상대적인 개선을 달성했습니다. 이를 통해 우리의 접근 방식이 실제 데이터의 부족함을 고려할 때 우수한 성능을 발휘함을 입증하였습니다.



### indiSplit: Bringing Severity Cognizance to Image Decomposition in Fluorescence Microscopy (https://arxiv.org/abs/2503.22983)
- **What's New**: 본 연구는 기존의 이미지 분해 방법의 한계를 극복하기 위해 새로운 방법인 indiSplit을 제안합니다. 기존 방법들은 고정된 강도 비율의 입력 이미지를 기준으로 훈련되었으나, 이는 플루오레센스 현미경의 다양한 상대적 강도를 반영하지 못했습니다. indiSplit은 서로 다른 혼합 비율을 인식할 수 있도록 설계되어, 불확실한 혼합 비율을 효과적으로 처리할 수 있습니다.

- **Technical Details**: indiSplit은 이미지 복원을 위한 인기 있는 반복(iterative) 방법인 InDI에 기반하여, 주어진 입력 이미지의 열화 수준(혼합 비대칭성)을 예측하는 회귀 네트워크를 도입합니다. 또한, 혼합 비율에 따른 열화 인식을 통해 분해 가능한 제어 모듈을 포함하여 다양한 혼합 비율에서도 안정적인 추론을 가능하게 합니다. 이러한 모듈은 노멀라이제이션 요구사항을 해결하며, 플루오레센스 현미경의 도메인 지식을 활용하여 혼합 비율 추정 정확성을 향상시킵니다.

- **Performance Highlights**: indiSplit은 플루오레센스 현미경에서 필요한 두 가지 중요한 작업, 즉 이미지 분리(image splitting)와 bleedthrough 제거를 동시에 처리할 수 있는 성능을 보입니다. 본 연구는 5개의 공개 데이터셋에서 indiSplit의 적용 가능성을 실증적으로 증명했으며, 그 결과는 기존의 방법들에 비해 개선된 성능을 보여주었습니다. 모든 소스는 허용된 라이센스 하에 공개될 예정입니다.



### From Flatland to Space: Teaching Vision-Language Models to Perceive and Reason in 3D (https://arxiv.org/abs/2503.22976)
Comments:
          Project page: this https URL

- **What's New**: 최근 LVLM(대형 비전-언어 모델)에서의 발전이 시각-언어 이해를 크게 향상시키고 있지만, 공간 인식(spatial perception)에는 여전히 한계가 있습니다. 본 연구는 3D 표현을 통합하는 기존 접근 방식과는 달리, 공간적으로 관련된 이미지 데이터를 활용하여 VLM의 잠재력을 끌어내고자 합니다. 이 과정에서 3D 실제(scenes data with 3D ground-truth)를 기반으로 한 새로운 2D 공간 데이터 생성 및 주석_pipeline을 도입했습니다.

- **Technical Details**: 우리는 SPAR-7M이라는 대규모 데이터셋을 구축하였으며, 이 데이터셋은 기본적인 인지 작업부터 복잡한 추론 작업까지 다양한 공간 작업을 포함합니다. 또한, SPAR-Bench라는 새로운 벤치 마크를 설계하여 기존의 공간 능력을 평가하는 방법에 비해 보다 포괄적인 평가를 제공합니다. 이 벤치 마크는 단일 뷰(single-view) 및 다중 뷰(multi-view) 입력을 모두 지원합니다.

- **Performance Highlights**: SPAR-7M과 대규모 2D 데이터셋을 통해 훈련된 모델은 여러 2D 공간 기준에서 최첨단 성능을 거두었습니다. 또한, 3D 작업에 특화된 데이터셋에서의 추가 미세 조정이 가능하여 경쟁력 있는 결과를 얻을 수 있음을 보여주었습니다. 이러한 결과는 제안한 데이터셋이 공간 추론 향상에 효과적임을 강조합니다.



### Pallet Detection And Localisation From Synthetic Data (https://arxiv.org/abs/2503.22965)
Comments:
          10 pages, 9 images, 4 tables, submitted and accepted to ACRA 2024 (this https URL)

- **What's New**: 이 논문은 합성 데이터(synthetic data)와 기하학적 특징을 활용하여 팔레트 탐지(pallet detection) 및 위치 추정(localisation) 시스템의 효율성을 높이는 새로운 접근 방식을 제안하고 있습니다. 기존의 데이터 주석(annotation) 과정에서 소요되는 수고를 줄이기 위해 Unity 기반의 도메인 랜덤화(domain randomisation) 엔진을 구현했습니다. 그 결과, 실제 데이터셋에서 싱글 팔레트에 대해 0.995 mAP50의 성능과 4.2cm 미만의 평균 위치 정확도를 달성했습니다.

- **Technical Details**: 이 연구는 물류, 제조 및 창고 산업의 운영 최적화를 위한 팔레트 탐지 및 위치 추정 시스템을 개발하는 데 중점을 두었습니다. 자동화 시스템의 정확한 팔레트 감지를 위해, 저렴한 패시브 센서(passive sensor)인 RGB 카메라를 사용합니다. 이미지 획득(image acquisition), 코너 감지(corner detection) 및 포즈 추정(pose estimation) 과정을 포함하는 파이프라인 방식(pipelined approach)을 통해 더욱 효율적인 시스템을 구현하고자 합니다.

- **Performance Highlights**: 이 시스템은 팔레트를 5미터 범위 내에서 4.2cm 이하의 평균 위치 정확도와 8.2°의 평균 회전 정확도를 가지고 있습니다. 연구 결과, 팔레트의 3D 위치를 정확히 파악할 수 있으며, 패시브 RGB 카메라를 통해 비용 효율적인 솔루션을 제공합니다. 이는 바쁜 창고 환경에서의 센서 간섭 문제를 해결하고, 복잡한 수동 데이터 주석 과정을 필요로 하지 않는 장점이 있습니다.



### SuperEIO: Self-Supervised Event Feature Learning for Event Inertial Odometry (https://arxiv.org/abs/2503.22963)
- **What's New**: 이번 논문에서는 SuperEIO라는 새로운 프레임워크를 제안하여 이벤트 카메라의 이벤트 특징 탐지 및 매칭을 위한 학습 기반 접근법을 채택하였습니다. SuperEIO는 IMU(관성 측정 장치) 측정을 활용하여 더욱 견고한 이벤트-관성 오도메트리(event-inertial odometry)를 실현합니다. 이전의 전통적인 방법 대신 딥 러닝 네트워크를 사용하여 이벤트 특징 탐지의 정확성과 효율성을 높였습니다.

- **Technical Details**: SuperEIO는 CNN(합성곱 신경망)을 사용하여 이벤트 스트림에서 특징을 탐지하며, GNN(그래프 신경망)을 통해 루프 클로저(loop closure) 감지를 수행합니다. TensorRT를 통해 딥 네트워크의 추론 속도를 가속화하여 저전력 환경에서도 실시간으로 작동할 수 있도록 최적화되었습니다. 이러한 접근법은 실세계 시나리오에서 강력한 일반화 능력을 보여줍니다.

- **Performance Highlights**: 우리는 SuperEIO의 성능을 여러 공공 데이터셋에서 평가하여 공격적인 조건과 HDR(고다이내믹레인지) 장면에서도 탁월한 정확성과 견고성을 입증하였습니다. 본 연구 결과는 딥 러닝 기반 이벤트 오도메트리에 대한 폭넓은 연구를 촉진하기 위해 오픈소스로 제공됩니다. 제안된 시스템은 자원이 제한된 플랫폼에서도 실시간으로 동작 가능한 장점을 가지고 있습니다.



### OmniMMI: A Comprehensive Multi-modal Interaction Benchmark in Streaming Video Contexts (https://arxiv.org/abs/2503.22952)
Comments:
          To appear at CVPR 2025

- **What's New**: 이 연구에서는 스트리밍 비디오 맥락에서 OmniLLM의 상호작용 능력을 평가하기 위한 OmniMMI라는 포괄적인 다중 모드 상호작용 벤치마크를 소개합니다. OmniMMI는 1,121개의 비디오와 2,290개의 질문을 포함하고 있으며, 기존 비디오 벤치마크에서 간과된 스트리밍 비디오 이해와 능동적 추론이라는 두 가지 주요 문제에 초점을 맞추고 있습니다. 추가로, M4라는 새로운 프레임워크를 제안하여 효율적인 추론이 가능한 스트리밍 모델 구축을 목표로 하고 있습니다.

- **Technical Details**: OmniMMI는 주어진 비디오의 현재 및 과거 상태를 점진적으로 이해해야 하는 스트리밍 동적 상태 인식과 사용자의 의도 및 동적 맥락을 고려하여 응답을 능동적으로 생성해야 하는 능동적 추론과 턴 테이킹을 포함한 두 가지 주요 장애물에 대한 평가를 목표로 합니다. 이를 통해 다양한 비디오 대화형 모델을 평가하며, 특히 다중 턴 상호작용 및 스트리밍 비디오의 경우 많은 모델들이 한계에 직면하고 있음을 보여줍니다. 또한, M4 프레임워크는 기존 비디오 모델과의 정합성 없이도 능동적 추론을 가능하게 합니다.

- **Performance Highlights**: OmniMMI를 사용한 평가 결과, 대다수의 모델들이 스트리밍 비디오 관련 다중 턴 작업에서 어려움을 겪고 있으며, 이는 단일 추론 단계를 넘어가는 데 한계를 보이는 것으로 나타났습니다. 오디오 및 비디오 입력을 모두 처리하는 모델이 단순 비디오 입력 모델보다 우수하지 않음을 보여주며, 이는 모달리티 정렬의 부족을 암시합니다. 모델의 크기를 늘리는 것이 성능 향상으로 이어지지 않으며, 입력 길이를 최적화하고 메모리 효율성을 조화롭게 유지하는 것이 중요하다는 점이 강조됩니다.



### Enhancing Learnable Descriptive Convolutional Vision Transformer for Face Anti-Spoofing (https://arxiv.org/abs/2503.22936)
- **What's New**: 이 논문에서는 최신의 얼굴 안티 스푸핑(FAS) 기술인 LDCformer에 대해 소개하며, 이 기술을 훈련하기 위한 세 가지 혁신적인 전략을 제안합니다. 첫 번째 전략은 dual-attention supervision으로, 지역적인 live/spoof 주의(attentions)를 안내하여 미세한 살아 있는 특성(liveness features)을 배우도록 돕습니다. 두 번째 전략인 self-challenging supervision은 도전적인 훈련 데이터를 생성하여 특성의 판별력을 향상시키도록 설계되었습니다.

- **Technical Details**: 제안된 세 가지 훈련 전략은 다음과 같습니다. Dual-attention supervision은 두 개의 주의 추정기를 함께 훈련시켜 LDCformer가 지역적인 주의에 집중하도록 독려합니다. Self-challenging supervision은 살아 있는 이미지와 스푸핑 이미지를 혼합하여 부분 스푸핑 공격을 감지하는 능력을 강화합니다. 마지막으로, transitional triplet mining 전략은 교차 도메인 문제를 해결하며, 살아 있는 특성과 스푸핑 특성 사이의 전이 관계를 유지하면서 도메인 일반화 능력을 확장합니다.

- **Performance Highlights**: 광범위한 실험을 통해, 제안된 세 가지 훈련 전략의 공동 감독 하에 LDCformer가 이전의 방법들을 능가하는 성능을 보여주었습니다. 이 연구는 FAS 기준 데이터 세트에서 state-of-the-art 성능을 달성하며, 각 전략의 효과를 확인하는 절단 연구와 실험 비교를 통해 그 유용성을 입증했습니다.



### Bi-Level Multi-View fuzzy Clustering with Exponential Distanc (https://arxiv.org/abs/2503.22932)
- **What's New**: 이 연구에서는 멀티 뷰 환경에서 퍼지 c-평균(FCM) 클러스터링을 확장하는 방법을 제안합니다. 첫 번째로, 우리는 지열 커널 계수(heat-kernel coefficients, H-KC)와 가중치 인자를 고려한 중앙 집중식 지수 멀티 뷰 FCM(E-MVFCM)을 소개합니다. 두 번째로, E-MVFCM과는 달리 자동으로 특징 및 가중치 인자를 동시에 계산하는 지수 이중 멀티 뷰 FCM 클러스터링(EB-MVFCM)을 제안합니다.

- **Technical Details**: H-KC는 양자장 이론(quantum field theory, QFT)의 중요한 도구이며, 이는 군집화 과정에서 열 커널의 생성을 단순화할 수 있도록 도와줍니다. E-MVFCM과 EB-MVFCM 모두 H-KC의 명시적 형태를 제시하여 복잡한 데이터의 클러스터를 인식하는 데 도움을 줍니다. 이 연구에서 제안된 알고리즘의 도구 및 기능은 공개된 URL을 통해 이용 가능할 것입니다.

- **Performance Highlights**: 본 논문에서는 E-MVFCM과 EB-MVFCM의 객관적 기능과 최적화 방법을 자세히 설명하고, 이들 알고리즘이 다양한 멀티 뷰 환경에서 클러스터 품질 개선에 기여할 수 있음을 보여줍니다. 이러한 새로운 알고리즘들은 불확실성 감소 및 더 나은 패턴 인식을 위한 강력한 방법을 제공합니다.



### Unsupervised Feature Disentanglement and Augmentation Network for One-class Face Anti-spoofing (https://arxiv.org/abs/2503.22929)
- **What's New**: 본 연구에서는 기존의 one-class FAS 방법의 한계를 극복하고, 더욱 일반화 가능한 Face Anti-Spoofing 기법을 제안합니다. 제안된 방법인 Unsupervised Feature Disentanglement and Augmentation Network (UFDANet)는 얼굴 이미지를 분리된 특징을 통해 증대시키며 전반적인 성능을 개선합니다. UFDANet은 liveness와 domain 특징을 성공적으로 분리하여 효과적인 특징 학습을 촉진합니다.

- **Technical Details**: UFDANet의 주요 기술적 기법은 unsupervised feature disentangling 방법입니다. 이를 통해 liveness 특징과 domain 특징을 효과적으로 분리하며, out-of-distribution liveness feature augmentation 스킴을 통합하여 새로운 liveness 특징을 생성합니다. 이 방식은 서로 다른 데이터 도메인에서 발생할 수 있는 문제를 해결하여 잠재적 특징의 강건성을 높입니다.

- **Performance Highlights**: 포괄적인 실험 결과에 따르면, UFDANet은 기존의 one-class FAS 방법들을 초월하며, 최신 two-class FAS 기법들과 비교하여 유사한 성능을 나타냅니다. 제안된 방법은 unseen 공격 탐지에 있어서 높은 일반화 능력을 보이며, 실제 환경에서의 유효성을 입증합니다. 따라서, UFDANet은 facial identity authentication의 보안을 강화하는 데 중요한 기여를 할 수 있습니다.



### DIFFER: Disentangling Identity Features via Semantic Cues for Clothes-Changing Person Re-ID (https://arxiv.org/abs/2503.22912)
Comments:
          Accepted in CVPR 2025

- **What's New**: 이번 연구에서는 DIFFER(Disentangle Identity Features From Entangled Representations)라는 새로운 적대적 학습 방법을 제안합니다. 이 방법은 텍스트 설명을 활용하여 신원 관련 특징을 분리하는 데 초점을 맞추고 있습니다. 기존의 방법들이 몸체 형태를 모델링하는 데 집중하는 경우가 많았으나, 본 방법은 성별, 나이 및 스타일과 같은 중요한 생체 정보도 함께 고려합니다. 이로 인해 더 다채롭고 정확한 개인 인식을 가능하게 합니다.

- **Technical Details**: DIFFER는 NBDetach라는 메커니즘을 소개하여 특징을 분리하는데, 이는 텍스트 설명의 분리 가능한 성질을 활용하여 이루어집니다. 이 메커니즘은 특징 공간을 구분된 하위 공간으로 나누고, 기울기 반전 층을 통해 신원 관련 특징을 비생체 특징과 효과적으로 분리합니다. 이 방법은 훈련 중에만 설명이 필요하고, 추론 단계에서는 추가적인 외부 모달리티에 의존하지 않아 유연성을 높입니다.

- **Performance Highlights**: DIFFER는 LTCC, PRCC, CelebreID-Light 및 CCVID의 네 가지 벤치마크 데이터셋에서 평가되었으며, 모든 벤치마크에서 최첨단 성능을 보여줍니다. 특히, LTCC에서는 기본 방법보다 3.6%, PRCC에서는 3.4%, CelebReID-Light에서는 2.5%, CCVID에서는 1%의 top-1 정확도가 향상되었습니다. 이는 기존의 여러 방법들과 비교할 때 더욱 우수한 성능을 나타냅니다.



### Enhancing DeepLabV3+ to Fuse Aerial and Satellite Images for Semantic Segmentation (https://arxiv.org/abs/2503.22909)
- **What's New**: 이번 논문에서는 기존의 DeepLabV3+ 아키텍처를 개선하여 저해상도 위성 데이터와 고해상도 항공 이미지를 결합하는 새로운 접근법을 제안하고 있습니다. 특히, 새로운 전이 합성곱 레이어 블록을 도입하여 두 번째 입력을 업샘플링하고 고수준의 특징들과 융합하는 방식을 개발했습니다. 이를 통해 다양한 정보 소스의 유용성을 극대화하며, 세분화 작업에서의 정확성과 성능을 향상시키고자 합니다.

- **Technical Details**: DeepLabV3+ 아키텍처를 기반으로 한 이 연구에서는 블라인 업샘플링을 가중치 기반 업샘플링 모듈로 대체하여 디코더 단계에서의 세분화 맵 재구성을 개선하였습니다. 다양한 업샘플링 방법에 대한 실험을 수행해, 위성 데이터 주입 과정에서 더 나은 성능을 발휘할 수 있는 방안을 찾아냈습니다. 이러한 과정은 저해상도 위성 이미지와 고해상도 항공 이미지 간의 효과적인 융합을 가능하게 합니다.

- **Performance Highlights**: 논문에서는 두 데이터 소스의 융합을 통해 평균 교차 비율(Mean Intersection over Union, mIoU)을 84.91%로 달성했으며, 데이터 증강 없이도 우수한 성과를 보였습니다. 이는 고해상도 항공 이미지의 세부사항과 위성 이미지의 다채로운 정보가 결합됐을 때의 성능을 잘 보여줍니다. 또한, 기존의 DeepLabV3+ 아키텍처의 성능을 크게 개선하는 결과를 도출했습니다.



### SocialGen: Modeling Multi-Human Social Interaction with Language Models (https://arxiv.org/abs/2503.22906)
- **What's New**: 이 논문에서는 SocialGen이라는 통합된 모션-언어 모델을 소개하여 다양한 수의 개인 간의 상호작용 행동을 모델링할 수 있는 첫 번째 모델을 제안합니다. 기존의 이인(두 사람) 상호작용에 국한된 방법과 달리, 우리는 모션을 토큰화하고 언어 공간에 정렬하는 혁신적인 사회적 모션 표현을 도입했습니다. 이러한 정렬은 모델이 풍부한 사전 학습된 언어 지식을 활용할 수 있게 해주어 인간의 사회적 행동을 더 잘 이해하고 추론할 수 있도록 합니다.

- **Technical Details**: SocialGen 프레임워크는 다양한 그룹 크기에 대한 상호작용 모션을 모델링하며 모션과 언어 공간을 정렬하는 두 가지 주요 구성 요소로 구성되어 있습니다. 우리는 XH3D라는 새로운 모션 표현법을 제안하며, 이는 복잡한 다인(多人) 상호작용을 효과적으로 인코딩하고 디코딩하기 위해 설계되었습니다. 추가로, SocialX라는 데이터세트를 개발하여 텍스트 주석이 포함된 다인 상호작용 사례를 포괄적으로 수집하고, 이는 다양한 모션 관련 작업을 위한 기준을 설정합니다.

- **Performance Highlights**: 우리 방법론은 다인 상호작용 작업의 최초 종합 기준을 수립하며, 모션-언어 작업 전반에 걸쳐 최신 성능을 달성합니다. 구체적으로, 모델은 기존 방법들에 비해 더 부드럽고 자연스러운 인간 모션을 생성하는 데 성공했습니다. 다양한 모션 관련 작업을 지원하면서도 새로운 기준을 설정하여 다인 상호작용 모델링의 미래 연구에 강력한 기반을 제공할 것입니다.



### MedCL: Learning Consistent Anatomy Distribution for Scribble-supervised Medical Image Segmentation (https://arxiv.org/abs/2503.22890)
- **What's New**: 이번 연구는 'MedCL'이라는 새로운 스크리블 슈퍼바이즈 클러스터링 기반 프레임워크를 제안하여 의료 이미지를 위한 해부학적 분포를 학습하고자 합니다. 기존의 방법들이 충분한 스크리블 주석을 요구하거나 일반 장기의 분할에만 적용되는 것에 반해, MedCL은 불규칙한 병리학의 분할도 가능하게 합니다. 두 단계로 구성된 이 아키텍처는 기능 혼합 및 클러스터링을 통해 의료 이미지를 효과적으로 세분화할 수 있습니다.

- **Technical Details**: MedCL의 첫 번째 단계는 이미지 내 및 이미지 간의 기능 혼합을 통해 특징을 조화롭게 섞는 것이며, 이는 약한 감독을 바탕으로 이루어집니다. 두 번째 단계에서는 특징 클러스터링이 진행되며, 이때 해부학적 특성을 정규화하여 클러스터 간의 간결성과 판별 가능성을 보장합니다. 아울러, 본 연구는 SAM과 UNet 두 가지 백본 구조를 기반으로 MedCL을 구현하였습니다.

- **Performance Highlights**: MedCL은 소량의 스크리블 감독만을 활용해도 기존의 분할 방법들보다 현저히 우수한 성능을 입증하였습니다. 세 개의 공개 데이터셋 (MSCMRseg, BTCV, MyoPS)에서 평가 결과, MedCL은 정규 장기 및 복잡한 불규칙 병리의 분할에서 모두 향상된 결과를 보여주었습니다. 연구 코드 또한 제공되어, 추가적인 연구 및 활용이 가능하도록 하고 있습니다.



### AutoComPose: Automatic Generation of Pose Transition Descriptions for Composed Pose Retrieval Using Multimodal LLMs (https://arxiv.org/abs/2503.22884)
- **What's New**: 이번 연구에서는 AutoComPose라는 혁신적인 프레임워크를 도입하여, 인간 동작의 전환 설명을 자동으로 생성할 수 있는 최초의 방법을 제시합니다. 이전의 Composed Pose Retrieval (CPR)은 비싼 인간 주석이나 경험적 규칙 생성에 의존하고 있었는데, 이는 확장성 및 다양성을 제한해왔습니다. AutoComPose는 다중 모드 대형 언어 모델(Multimodal Large Language Models, MLLMs)을 활용하여 더욱 풍부하고 구조화된 동작 전환 설명을 생성합니다.

- **Technical Details**: AutoComPose는 두 단계의 접근 방식을 사용하여 세부적인 신체 부위 기반 동작 전환을 생성한 후, 이를 통합하고 바꿔서 완전한 설명으로 재구성합니다. 이 과정에서 미러링(mirroring) 및 스왑(swapping) 변형을 도입하여 다양성을 높이고 있습니다. 또한, 훈련 과정을 통해 순환 일관성(cyclic consistency) 제약을 포함시켜, 전환 설명의 논리적 일관성을 확보하고자 하였습니다.

- **Performance Highlights**: AutoComPose의 성능은 기존의 인간 주석 기반 방법이나 경험적 접근 방식에 비해 뛰어난 결과를 보였습니다. 제안된 두 개의 벤치마크(AIST-CPR 및 PoseFixCPR)에 대한 실험에서도 AutoComPose로 훈련된 모델이 높은 품질의 전환 설명을 유지하면서 주석 비용을 크게 절감할 수 있음을 증명하였습니다. 이 연구는 미래 CPR 연구를 위한 확장 가능한 기초를 설정합니다.



### Pairwise Matching of Intermediate Representations for Fine-grained Explainability (https://arxiv.org/abs/2503.22881)
- **What's New**: 이번 논문에서는 기존의 딥러닝 모델 설명 기법의 한계를 극복하기 위해 PAIR-X라는 새로운 설명 방법을 제안합니다. 이 방법은 모델의 중간 활성화(activation) 및 역전파된 중요도 점수를 활용하여 매우 세밀하고 국부적인 쌍별 비주얼 설명을 생성합니다. 동물 및 건물 재식별(re-ID)을 주요 사례 연구로 삼아, 35개의 공개 재식별 데이터셋에서 다양한 설명 기법들과 비교하여 정 qualitative한 개선 결과를 보여주었습니다.

- **Technical Details**: PAIR-X는 각각의 이미지 쌍에 대해 더 유용하고 해석 가능한 설명을 제공하기 위해 중간 모델 활성화와 역전파된 중요도 점수를 결합합니다. 저자들은 새로운 정량적 평가 메트릭을 제안하여 PAIR-X의 시각화가 올바른 이미지 매치에 대해 더욱 그럴듯하게 나타나는 것을 입증했습니다. 이러한 기술적 접근 방식은 이미지의 유사도 점수가 동일하더라도 올바른 매치와 잘못된 매치를 보다 쉽게 구별할 수 있도록 도와줍니다.

- **Performance Highlights**: 연구 결과, 동물 재식별 전문가들은 PAIR-X의 시각화가 기존 방법들보다 개선되었다고 만장일치로 동의하였으며, 그들의 작업에 즉시 적용 가능하다고 평가했습니다. PAIR-X는 특히 차별화된 주목을 받았으며, 이러한 해석 가능성의 향상은 전반적으로 모델의 신뢰성과 유용성을 높이는 데 기여할 것으로 기대됩니다.



### The Marine Debris Forward-Looking Sonar Datasets (https://arxiv.org/abs/2503.22880)
Comments:
          10 pages, 12 figures, Oceans Brest 2025 camera readyu

- **What's New**: 이번 연구에서는 해양 쓰레기를 감지하고 분류하기 위한 새로운 공공의 Forward-Looking Sonar (FLS) 데이터셋인 Marine Debris FLS를 제안합니다. 이 데이터셋은 수조(watertank), 회전대(turntable), 침수된 채석장(flooded quarry)의 세 가지 설정에서 수집된 다양한 이미지와 과제를 포함하고 있습니다. 이는 AI 시스템의 훈련을 위한 대규모 데이터셋의 부족 문제를 해결하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: 이 데이터셋은 해양 쓰레기 객체와 관련된 다양한 컴퓨터 비전 작업을 수행하기 위한 자료를 제공합니다. 객체 분류(object classification), 객체 탐지(object detection), 의미론적 분할(semantic segmentation), 패치 매칭(patch matching), 비지도 학습(unsupervised learning) 등의 작업을 포함하고 있으며, 각각의 설정과 과제는 세부적으로 설명되어 있습니다. 특히, ARIS Explorer 3000 센서를 사용하여 정밀한 데이터 수집이 이루어졌습니다.

- **Performance Highlights**: 초기 기준(result benchmark)을 통해 데이터셋의 유용성이 입증되었습니다. 연구진은 수집된 데이터를 바탕으로 다양한 실험을 진행하고 있으며, 각 시나리오에 따른 데이터의 변동성과 업데이트된 객체 클래스 목록은 모델의 정확성을 높이는 데 기여할 것입니다. 이 데이터셋은 바다에 버려진 인위적 쓰레기를 탐지하고 이해하는 데 중요한 기초 자료로 사용될 것입니다.



### SIGHT: Single-Image Conditioned Generation of Hand Trajectories for Hand-Object Interaction (https://arxiv.org/abs/2503.22869)
- **What's New**: 본 논문에서는 단일 이미지에서의 3D 손 궤적 생성이라는 새로운 작업을 소개합니다. 이 작업은 손-객체 상호작용 장면에 대한 이미지나 단독 객체 이미지에서 실제적이고 다양한 손 궤적을 생성합니다. SIGHT-Fusion 시스템을 통해 이러한 작업을 해결하고, 실제 물리 시뮬레이션에서 성공적인 작업 실행을 평가하기 위한 새로운 성과 측정 기준을 제안합니다.

- **Technical Details**: SIGHT-Fusion은 손-객체 상호작용 세부 정보를 추출하기 위한 정제된 파이프라인과 이를 처리하는 확산 기반( diffusion-based ) 조건부 모션 생성 모델로 구성됩니다. 본 연구는 비지도 학습(unsupervised) 방식으로 비디오 데이터와 손 궤적 주석과 함께 훈련되며, FPHAB와 HOI4D 데이터셋을 이용해 벤치마크를 구축합니다. 이를 통해 3D 손 궤적을 생성하는 과정에서 조건적 입력으로 추출된 특징을 활용합니다.

- **Performance Highlights**: SIGHT-Fusion은 기존 방법론과 비교해 자연스럽고 다양한 손 궤적을 생성하는 데 성공적이며, 미지의 객체에 대해서도 뛰어난 일반화 능력을 보여줍니다. 물리 시뮬레이션에서 생성된 궤적의 정확성을 확인하여, 만들어진 시퀀스의 진정성과 다운스트림 사용에의 적용 가능성을 입증했습니다. 실험 결과는 제안한 방법이 다양한 상황에서 효과적으로 사용될 수 있음을 시사합니다.



### Zero-shot Domain Generalization of Foundational Models for 3D Medical Image Segmentation: An Experimental Study (https://arxiv.org/abs/2503.22862)
- **What's New**: 이 연구는 의료 영상 분할에서 제로샷 도메인 일반화(zero-shot domain generalization)의 가능성을 탐구하며, 기존의 세분화(FM) 모델 6개와 12개의 공개 데이타셋을 활용하여 포괄적인 실험 연구를 수행하였습니다. 특히, 다양한 대규모 데이터로 훈련된 포언드 모델(Foundation Models, FMs)의 적용 가능성을 밝혀내고, 도메인 간 격차를 해소하는 스마트 프로프트 기법의 잠재력을 보여줍니다.

- **Technical Details**: 연구에서는 제로샷 일반화(zero-shot generalization) 가능성을 평가하기 위해 다양한 아나톰과 모달리티가 포함된 12개의 볼륨 세분화 데이터셋을 광범위하게 분석했습니다. 모델 카테고리는 특정 모달리티/일반적 지식, 사용자 프로프트 가능성에 따라 분류되었으며, 도메인 특정 FM과 비주얼 및 텍스트 프롬프트 FMs를 포함하였습니다.

- **Performance Highlights**: 실험 결과, 기존의 FMs는 도메인 훈련된 전문 모델에 비해 상당한 도메인 격차를 보였지만, 최근의 텍스트 프롬프트 모델들이 이 격차를 좁히는 경향을 보였습니다. 이 결과들은 의료 영상 컴퓨팅에서 도메인 일반화가 가능한 모델의 잠재적 방향성을 시사하며, 향후 연구를 위한 유망한 경로도 제시합니다.



### GmNet: Revisiting Gating Mechanisms From A Frequency View (https://arxiv.org/abs/2503.22841)
- **What's New**: 이 연구에서는 Gating Mechanism Network (GmNet)을 제안합니다. 이 모델은 다양한 주파수 성분의 정보를 효율적으로 활용하도록 설계되었습니다. GmNet은 기존 경량 모델에서 나타나는 저주파 편향을 최소화하여 이미지 분류 작업에서 뛰어난 성능을 보여줍니다. 이에 대한 이론적 분석과 실험을 통해 GmNet의 효과성을 입증하였습니다.

- **Technical Details**: 이 논문은 Gating Units이 신경망의 훈련 동역학에 미치는 영향을 주파수 관점에서 체계적으로 탐구합니다. Gated Linear Units (GLUs)는 정보 흐름을 적응적으로 조절하며, 활성화 함수와의 상호 작용을 통해 주파수 응답을 형성하는 방법을 분석합니다. 또한, 주파수 영역에서의 합성곱 정리를 바탕으로, 새로운 주파수 성분이 신경망의 학습 표현을 강화할 수 있음을 보여줍니다.

- **Performance Highlights**: GmNet은 EfficientFormer-L1 대비 Top-1 정확도에서 4.0% 향상된 성능을 보이며, A100 GPU에서 4배 빠른 지연 시간을 달성했습니다. 이 연구는 주파수 측면에서 Gated Linear Units의 훈련에 미치는 영향을 체계적으로 분석하여 새로운 효율적인 모델 설계를 위한 토대를 마련합니다. 이를 통해 신경망이 주파수 정보를 어떻게 더 효과적으로 활용할 수 있는지에 대한 통찰을 제공합니다.



### DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers (https://arxiv.org/abs/2503.22796)
- **What's New**: 이 논문에서는 Multimodal Diffusion Transformers (MMDiT)의 주의(attention) 메커니즘을 가속화하기 위한 새로운 압축 방법인 DiTFastAttnV2를 소개합니다. 이 방법은 이미징 품질을 유지하면서도 68%의 attention FLOPs 절감과 1.5배의 속도 향상을 달성합니다. 또한, 기존 접근 방법들과의 차별화를 위해 헤드 중심의 동적 주의 조절 및 캐싱 메커니즘을 도입합니다.

- **Technical Details**: DiTFastAttnV2는 MMDiT의 주의 패턴들을 정밀하게 분석하여, 서로 다른 주의 헤드의 비율과 행동을 기반으로 한 동적 캐싱 메커니즘을 형성합니다. 이 방법은 효율적인 융합 커널(Efficient Fused Kernel)을 통해 주의 메커니즘을 최적화하며, 모델 압축을 위한 검색 비용을 최소화합니다. 이를 통해 다차원 압축 계획 탐색을 현저히 단축시킵니다.

- **Performance Highlights**: 2K 이미지 생성 작업에서 DiTFastAttnV2는 주의 계산량을 68%까지 줄이면서도 생성 품질을 유지합니다. 또한, 전체적인 생성 속도를 1.5배 향상시켜 고해상도 이미지와 긴 비디오 생성에서도 효과적으로 활용할 수 있습니다. 이러한 성능 향상은 MMDiT의 복잡한 주의 패턴을 효율적으로 처리하는 데 기인합니다.



### Patronus: Bringing Transparency to Diffusion Models with Prototypes (https://arxiv.org/abs/2503.22782)
- **What's New**: 이 연구에서는 Denoising Diffusion Probabilistic Models (DDPMs) 기반의 해석 가능한 확산 모델인 Patronus를 소개합니다. Patronus는 ProtoPNet에서 영감을 받아 프로토타입 네트워크를 통합하여 생성 프로세스에 영향을 미치는 프로토타입을 추출하고 조절할 수 있습니다. 이 모델은 주석이나 텍스트 프롬프트 없이도 작동하며, 따라서 해석 가능성과 투명성을 증대시키는 새로운 경로를 제공합니다.

- **Technical Details**: Patronus의 설계는 이미지의 패치를 기반으로한 프로토타입 추출 및 표현 모듈을 포함하고 있습니다. 이 모듈은 입력 이미지를 패치 기반 특징 표현으로 변환하고, 각 프로토타입에 대한 유사성 점수를 계산하여 확산 과정의 조건으로 사용합니다. 또한, 프로토타입의 활성화 벡터를 통해 입력 이미지에 대한 의미 정보를 인코딩하는 방법을 제안합니다.

- **Performance Highlights**: Patronus는 세밀하게 의미 있는 시각적 특징을 잘 포착하며, 최신 SOTA (State-of-the-Art) 방법들과 비교해 경쟁력 있는 생성 품질을 달성합니다. 또한, 이 모델은 데이터셋 내에서 원치 않는 상관관계를 진단할 수 있는 잠재력을 보여주며, 생성 모델의 편향을 줄이고 공정성을 증진하는 데 유용한 도구로 활용될 수 있습니다.



### ORAL: Prompting Your Large-Scale LoRAs via Conditional Recurrent Diffusion (https://arxiv.org/abs/2503.24354)
- **What's New**: 본 논문에서는 기존의 방법들이 직면한 확장성과 조정 가능성의 한계를 해결하기 위해 새로운 조건부 반복 확산(conditional recurrent diffusion) 프레임워크인 ORAL을 소개합니다. ORAL은 고유한 조건화 메커니즘을 포함하여 모델 아키텍처와 작업 사양을 통합하여, 진화하는 기초 모델을 통해 효율적으로 전이 가능한 LoRA 파라미터를 생성할 수 있습니다. 이 접근법은 수십억 개의 파라미터를 가진 대형 언어 모델에서도 조정 가능성을 유지하면서 확장을 성공적으로 수행합니다.

- **Technical Details**: ORAL의 주요 기여는 LoRA 파라미터의 유연한 생성을 위한 새로운 조건화 메커니즘을 개발한 것입니다. 이 메커니즘은 모델 아키텍처 및 텍스트 기반 작업 사양을 입력으로 사용하여, 특정 다운스트림 작업에 맞춤화된 LoRA 파라미터를 생성할 수 있게 합니다. ORAL은 기존의 반복 확산 아키텍처를 기반으로 하여, 자원 집약적인 재교육 없이도 진화하는 기초 모델에 생성된 파라미터를 원활하게 전이할 수 있는 새로운 조건부 파라미터 생성 파이프라인을 제안합니다.

- **Performance Highlights**: 다양한 실험을 통해 ORAL은 7개의 언어 작업, 4개의 비전 작업, 3개의 다중 모달 작업을 수행하였으며, 5개의 사전 학습된 LLM을 사용하여 그 효율성을 입증했습니다. 연구 결과, ORAL은 7777억 개의 파라미터를 효과적으로 처리하면서도 전통적인 미세 조정 방법과 비슷하거나 우수한 성능을 보여줍니다. 이는 ORAL이 기존 방법과 비교할 때 scalability, controllability 및 portability를 모두 충족하는 새로운 기준을 정립하고 있음을 의미합니다.



### A Comparative Study of Scanpath Models in Graph-Based Visualization (https://arxiv.org/abs/2503.24160)
- **What's New**: 이 연구에서는 정보 시각화(InfoVis)의 효과를 극대화하기 위한 수단으로 시각적 주의 배분을 분석하는 새로운 접근 방식을 제시합니다. 우리는 40명의 참가자를 대상으로 한 Eye-tracking(ET) 실험을 수행하였으며, 초점은 다양한 질문 복잡도 하에서 그래프 분석에 있었습니다. 이 분석의 결과로서 생성 모델인 DeepGaze, UMSS, Gazeformer가 제시되고, 이들 모델의 정확성과 함께 질문 복잡도 및 노드 수의 영향을 탐구하게 됩니다.

- **Technical Details**: 정보 시각화 시스템은 대량의 데이터를 직관적으로 분석하기 위해 시각 요소를 활용합니다. 그러나 Eye-tracking 데이터 수집은 비용 및 프라이버시 문제로 인해 도전 과제가 됩니다. 본 연구에서는 인간의 주의 패턴을 예측하기 위해 컴퓨터 모델을 활용하여, 시각적 요소들이 정보 분석에 미치는 영향을 측정합니다. 또한, 기존의 시각적 주의력을 측정하는 방법의 한계를 검토하고 최첨단 생성 모델을 활용한 새로운 접근 방법을 제안합니다.

- **Performance Highlights**: 연구의 결과로, 시각적 주의 모델들이 사람의 시선 경로와 얼마나 유사한지를 평가하였으며, 특히 질문의 복잡도와 시각화의 복잡성이 이들 모델의 정확성에 미치는 영향을 확인했습니다. 이러한 접근 방식은 정보 시각화의 디자인을 향상시키고, 실제 시각적 변화에 대한 사용자 의식 수준을 평가할 수 있는 기회를 제공하며, 궁극적으로 InfoVis 시스템의 효율성을 높일 수 있는 기초 자료를 제공합니다.



### AI-Assisted Colonoscopy: Polyp Detection and Segmentation using Foundation Models (https://arxiv.org/abs/2503.24138)
Comments:
          This work has been submitted to the IEEE TMI for possible publication

- **What's New**: 본 논문에서는 대장 내시경에서의 폴립 탐지에 대한 새로운 접근 방식을 제시합니다. Deep Learning 모델을 통해 누락된 폴립의 80%를 탐지할 수 있음을 보여주고, foundation 모델들이 효과적인 솔루션으로 떠오르고 있습니다. 특히, zero-shot 및 few-shot 학습 능력을 통해 의료 영상에서의 데이터 부족 문제를 해결할 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: 이 연구에서는 polyp segmentation 문제를 instance segmentation 작업으로 정의하고, YOLOv8, YOLO-World, GroundingDINO 모델을 사용하여 폴립의 존재를 탐지하고, 이후 segmentation을 통해 형태적 특징을 정확하게 표현하고자 하였습니다. YOLO-World는 CLIP 기반의 텍스트 인코더를 통해 open-vocabulary 탐지 기능을 추가하여 훈련 데이터셋에는 존재하지 않는 타겟 카테고리도 인식할 수 있습니다. Grounding DINO는 텍스트 프롬프트를 사용하여 다양한 객체를 탐지하는 능력을 갖춘 확장 모델입니다.

- **Performance Highlights**: 연구 결과, foundation 모델들이 기존의 state-of-the-art 탐지 및 segmentation 모델보다 뛰어난 성능을 보였습니다. 다수의 모델이 zero-shot 평가에서 우수한 성능을 발휘했으며, 이는 특정 도메인에서의 모델 전문화가 성능에 중요한 영향을 미친다는 것을 보여줍니다. 따라서, 의료 적용에서 최적의 성능을 위해서는 도메인 특화 모델이 필요하며, 일반 모델은 효과적인 결과를 얻기 위해 fine-tuning이 요구됨을 알 수 있습니다.



### Learning 3D-Gaussian Simulators from RGB Videos (https://arxiv.org/abs/2503.24009)
- **What's New**: 이 논문에서는 3D 물리 시뮬레이터인 3DGSim을 소개하며, 이는 다중 뷰 RGB 비디오에서 물체의 동역학을 끝에서 끝까지 학습하는 혁신적인 접근을 제공합니다. 3DGSim은 이미지를 3D Gaussiann (Gaussian) 파티클 표현으로 인코딩하고, 이러한 동역학을 transformer를 통해 전파하며, 3D Gaussian splatting을 사용하여 프레임을 렌더링합니다. 이 방법은 물리적 특성을 포인트-와이즈(latent vectors) 잠재 벡터에 포함시키면서, 명시적인 연결 제약 조건을 두지 않고도 다양한 물리적 행동을 캡처할 수 있도록 합니다.

- **Technical Details**: 3DGSim은 RGB 비디오에서 입자 상호작용을 직접 학습하여 3D Gaussian 포인트 클라우드로 장면을 표현합니다. kNN 대신 시간적 포인트 클라우드 직렬화(temporal point cloud serialization)를 활용하여 모델의 확장성을 크게 향상시킵니다. 동역학 모델과 함께 역 물리 렌더링을 공동으로 훈련하며, 이는 모션 사전(prior)을 3D Gaussian 표현에 직접 포함시키고 렌더링 직전에만 이 잠재 벡터를 splats로 맵핑합니다.

- **Performance Highlights**: 3DGSim은 다양한 물리적 행동을 포착할 수 있으며, 이는 강체(rigid), 탄성(elastic), 천과 같은 상호작용을 포함합니다. 또한 현실적인 조명 효과를 생성하며, 보지 못했던 다중체 상호작용과 새로운 장면 수정에 대해 일반화되는 것을 가능하게 합니다. 이로 인해, 3DGSim은 로봇의 의사결정에서 신뢰성을 높이며, 물리적 정확성을 강화하는 데에 기여할 가능성이 큽니다.



### AMB-FHE: Adaptive Multi-biometric Fusion with Fully Homomorphic Encryption (https://arxiv.org/abs/2503.23949)
- **What's New**: 본 논문은 다중 생체 인식 모달리티를 결합하여 보안성과 사용자 편의성을 동시에 향상시키기 위한 간단하지만 유연한 접근 방식을 제시합니다. 제안된 방법은 다차원 생체 인증 프레임워크인 AMB-FHE(Adaptive Multi-Biometric Fusion with Fully Homomorphic Encryption)로, 여러 모달리티의 참조 템플릿을 동시 암호화하여 개인 정보 보호를 강화합니다. 이 시스템은 런타임 동안 보안 요구사항에 맞춰 조정이 가능하여 사용성을 개선합니다.

- **Technical Details**: AMB-FHE는 원활한 기능 평가를 위해 동형 암호화(homomorphic encryption) 기술을 사용하며, 이 시스템은 CASIA 홍채 데이터베이스와 MCYT 지문 데이터셋에서 BIMODAL 방식으로 특징을 추출하기 위한 심층 신경망(deep neural networks)을 활용합니다. 생체 인증 시스템에서 적합한 사용자 경험을 위해, 여러 모달리티로부터의 템플릿을 결합한 후, 이를 단일 암호문으로 저장하여 개인 정보 보호 및 효율적인 암호문 활용을 달성합니다. AMB-FHE는 결정 수준(decision-level) 융합을 적용하여 비교 결과를 이진(binary) 형태로 출력합니다.

- **Performance Highlights**: 본 연구에서 제안된 AMB-FHE는 다양한 모달리티의 생체 템플릿을 단일 암호문에서 저장하여, 암호화된 데이터를 효과적으로 활용할 수 있도록 합니다. 이 과정에서 제곱 암호문(squared ciphertext)을 생성하여 성능을 극대화함과 동시에, 사용자가 생체 데이터를 여러 번 제시할 필요를 줄임으로써 편리함을 더합니다. 결론적으로, AMB-FHE는 사용자 편의성을 개선하고 뛰어난 개인 정보 보호를 제공하는 유연한 생체 인증 체계를 제시합니다.



### An Explainable Neural Radiomic Sequence Model with Spatiotemporal Continuity for Quantifying 4DCT-based Pulmonary Ventilation (https://arxiv.org/abs/2503.23898)
Comments:
          43 pages, 13 figures

- **What's New**: 이번 연구에서는 폐암 환자의 폐 환기(ventilation) 평가를 위한 설명 가능한(Explainable) 신경 라디오믹(Neural Radiomic) 시퀀스 모델을 제안합니다. 기존의 핵의학 기법을 사용한 폐 환기 스캔이 시간이 많이 소요되고 비용이 비쌀 뿐만 아니라 방사선 노출을 수반하는 단점을 가지고 있습니다. 제안된 모델은 4차원 컴퓨터 단층촬영(4DCT)을 기반으로 하여, 환기 결함을 식별하는 것을 목표로 합니다.

- **Technical Details**: 연구는 VAMPIRE 데이터셋에서 45명의 폐암 환자를 분석하며, 각 환자에서 4DCT를 통해 폐 볼륨(segment)와 호흡 주기 동안의 복셀( voxel)별 라디오믹 피쳐(특징) 56차원을 추출하였습니다. 이를 통해 국소적인 강도(intensity)와 텍스처(texture) 다이나믹스를 포착하고, 시간에 따른 라디오믹 시퀀스를 형성하였습니다. 또한, 설명 가능한 장단기 기억(Long Short-Term Memory, LSTM) 네트워크를 개발하여 라디오믹 시퀀스를 기반으로 손상된 폐 영역을 식별합니다.

- **Performance Highlights**: 제안된 모델은 25개의 PET 사례에 대해 0.78(범위: 0.74-0.79), 20개의 SPECT 사례에 대해 0.78(범위: 0.74-0.82)의 평균 다이스 유사성 계수(Dice similarity coefficient)를 달성하며 강력한 성능을 보여주었습니다. 시간에 따른 중요성 맵(Temporal Saliency Maps)을 통해 환기 정량화에서 세 가지 주요 라디오믹 시퀀스를 설명할 수 있으며, 특히 폐의 호기(exhalation) 동안 손상이 있는 지역은 강도의 증가와 동질성(homogeneity)의 감소 경향을 보입니다.



### DiffScale: Continuous Downscaling and Bias Correction of Subseasonal Wind Speed Forecasts using Diffusion Models (https://arxiv.org/abs/2503.23893)
Comments:
          28 pages, 18 figures, preprint under review

- **What's New**: 본 연구는 S2S(서브시즌-투-시즌) 예측의 정확성을 향상시키기 위한 새로운 모델인 DiffScale을 제안합니다. 이 모델은 편향 없는 가이드를 사용하는 확산 모델로, 바람 속도 예측의 다운스케일링을 통해 지역 및 대규모 날씨 상황을 효과적으로 반영할 수 있도록 합니다. DiffScale은 여러 그리드 해상도 및 전진 시간에 걸쳐 모델 오류를 수정하고 조정할 수 있는 유연성을 제공합니다.

- **Technical Details**: DiffScale은 조건부 확률(conditional probabilities)을 샘플링의 가이드로 활용하는 확산 모델입니다. 이 모델은 S2S 예측의 밀도를 직접 추정하여, 자동 회귀(auto-regression)나 시퀀스 예측 없이도 효율적이고 유연한 예측을 가능하게 합니다. 연구에서는 ECMWF(유럽 중기 일기예보 센터)의 조도 해상도(S2S) 바람 속도 예측을 ERA5 재분석 데이터의 고해상도로 다운스케일링하는 synthetic experiments를 설계하였습니다.

- **Performance Highlights**: DiffScale은 최대 3주까지 기존의 기준 성능을 초과하는 예측 품질 향상을 이뤘습니다. 이 모델은 다양한 그리드 해상도에 일반화할 수 있으며, 모델 재훈련 없이도 새로운 스케일링 요소에 대응할 수 있는 다재다능한 도구입니다. 연구 결과는 에너지 부문에 중요한 사회경제적 이점을 제공할 수 있는 가능성을 보여줍니다.



### ZeroMimic: Distilling Robotic Manipulation Skills from Web Videos (https://arxiv.org/abs/2503.23877)
Comments:
          ICRA 2025. Project website: this https URL

- **What's New**: 이번 연구에서는 다소 수집하기 어려운 특정 형태의 데모에 의존하는 기존의 로봇 조작(Imitation Learning) 방식을 벗어나, ZeroMimic이라는 새로운 시스템을 개발했습니다. ZeroMimic은 사전 녹화된 인간 비디오 데이터에서 로봇 조작 스킬 정책(skil policies)을 추출하여 다양한 작업에서 즉시 사용할 수 있는 능력을 보여줍니다. 이 시스템은 기존의 비디오 이해 및 고급 그립 탐지기(grasp affordance detectors)의 발전을 활용하여 여러 조작 작업을 수행할 수 있습니다.

- **Technical Details**: ZeroMimic은 인간 동작 비디오를 사용하여 로봇의 조작 기술을 학습하며, 이를 위해 작업의 두 가지 주요 단계를 정의합니다: 물체를 적절히 집는 잡기 단계(grasping phase)와 안정적으로 유지하며 조작하는 후작업 단계(post-grasp phase)입니다. 본 시스템은 EpicKitchens 데이터셋에서 학습하며, 다양한 환경에서 사용할 수 있도록 여러 로봇 구현체에 맞춰 쉽게 교환할 수 있는 독립적인 정책을 생성합니다. 주요 구성 요소로는 3D 맵을 유지하기 위한 구조에서 운동 시스템이 포함되어 있습니다.

- **Performance Highlights**: ZeroMimic은 실제 환경에서 9가지 다양한 기술 평가에서 71.0%의 성공률을 기록하였으며, 시뮬레이션에서는 73.8%의 성공률을 보여주었습니다. 또한, 준비된 웹 비디오에서 보지 못한 새로운 객체들에도 일반화할 수 있는 능력을 가지고 있습니다. 이와 같은 성과는 다양한 일상 환경에서 로봇이 수행할 수 있는 작업의 범위를 확장하는 데 기여할 것입니다.



### Conformal uncertainty quantification to evaluate predictive fairness of foundation AI model for skin lesion classes across patient demographics (https://arxiv.org/abs/2503.23819)
- **What's New**: 이번 연구는 피부 병변 분류에서 비전 트랜스포머(ViT) 기반의 기초 모델을 사용하여 예측의 신뢰성과 공정성을 최적화하는 것입니다. 특히, 의사의 의사결정 프로세스를 이해하고 개선하기 위해 conformal analysis를 적용하여 예측 불확실성을 정량화합니다. 이 방법은 공정성 보장을 위해 인구 집단 차이를 고려한 적응형 F1 점수 기반 샘플링을 도입합니다.

- **Technical Details**: 연구에서 제안된 방법론은 피부 병변 분류와 불확실성 예측을 위한 최신 기초 모델과 conformal prediction 기반 불확실성 정량화를 결합하는 것입니다. 클래스 불균형 문제를 해결하기 위해 F1-점수 기반의 모델 무관 동적 샘플링 알고리즘을 적용하였으며, 이는 다양한 인구 집단 간의 예측 성능을 균형 있게 유지하게 합니다.

- **Performance Highlights**: 이 연구의 성과는 피부 암 진단에 있어 신뢰할 수 있는 예측 결과를 제공하는 것입니다. 각 환자에게 대해 불확실성 점수를 제공함으로써 모델의 공정성과 신뢰성을 높이는 것을 목표로 하였습니다. 이러한 접근법은 다양한 의료 데이터셋에서도 적용 가능하며, 의사결정의 투명성을 높일 수 있는 잠재력을 지니고 있습니다.



### Texture or Semantics? Vision-Language Models Get Lost in Font Recognition (https://arxiv.org/abs/2503.23768)
- **What's New**: 본 논문은 현대 비전-언어 모델(Visual-Language Models, VLMs)의 폰트 인식 능력을 평가하기 위해 Font Recognition Benchmark (FRB)을 제안합니다. 이 데이터셋은 15종의 일반적으로 사용되는 폰트로 구성되며, 쉬운 버전과 어려운 버전으로 나뉘어 폰트 인식의 어려움을 측정합니다. 현재 VLM들이 폰트 인식 작업에서 기대 이하의 성과를 보이고 있음을 밝혔습니다.

- **Technical Details**: Font Recognition Benchmark (FRB)는 두 가지 버전으로 나뉘며, 쉬운 버전은 다양한 폰트로 렌더링된 문장으로 구성되고, 어려운 버전은 폰트 이름을 잘못된 유형으로 표시하여 Stroop 효과를 도입합니다. 이 데이터셋은 Serif, Sans-Serif, Script & Decorative이라는 세 가지 주요 카테고리의 15개 폰트를 포함합니다. 이미지 생성에는 일관된 Python 스크립트를 사용하여 폰트 크기와 배경을 균일하게 유지합니다.

- **Performance Highlights**: 본 연구에서는 13개의 VLM 모델을 평가한 결과, 가장 높은 성과를 보인 모델조차 쉬운 버전에서 약 30%의 정확도만 달성했으며, 어려운 버전에서는 15%로 떨어졌습니다. Chain-of-Thought (CoT) 프롬프트는 성과 개선에 미미한 효과를 보였고, 적은 예시(Few-shot learning)조차도 VLM의 성능 향상에는 한계를 드러냈습니다. 이러한 결과는 VLM들이 폰트 인식에서 적절한 성능을 발휘하지 못하고 있음을 나타냅니다.



### StrokeFusion: Vector Sketch Generation via Joint Stroke-UDF Encoding and Latent Sequence Diffusion (https://arxiv.org/abs/2503.23752)
- **What's New**: 이번 연구에서는 StrokeFusion이라는 새로운 두 단계 프레임워크를 제안하여 벡터 형식으로 스케치를 생성하는 문제를 해결합니다. 이 프레임워크는 스케치를 정규화된 스트로크로 분해하고, Unsigned Distance Function (UDF) 맵을 통해 스트로크 시퀀스를 공동 인코딩하는 듀얼 모달 스케치 특징 학습 네트워크를 포함하고 있습니다. 이를 통해 스케치의 구조적 무결성과 의미적 특징을 보존하는 고충실도 스케치를 생성할 수 있습니다.

- **Technical Details**: StrokeFusion의 첫 번째 단계에서는 스트로크-유니폼 거리 함수 결합 인코딩을 도입하여 벡터 스트로크를 정규화된 원소로 분해합니다. 두 번째 단계에서는 스트로크 수준의 잠재 확산(diffusion) 모델을 통해 비순차적 방식으로 스트로크를 생성하며, 이는 고정된 생성 순서에 의존하지 않도록 합니다. 이러한 접근 방식은 스트로크의 위치, 스케일 및 궤적을 동시 조정하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 풍부한 실험을 통해 StrokeFusion은 최신 기술들을 초월하는 성능을 보여주며, 정량적 메트릭 및 질적 비교 모두에서 그 장점을 눈에 띄게 확인할 수 있었습니다. 실험 결과는 구조적 일관성을 유지하면서 스케치 생성을 가능하게 하는 효과성을 입증합니다. 추가적으로, 이 프레임워크는 고유의 스트로크 특징 벡터를 구성하여 비순차적이고 다양한 길이의 스트로크 시퀀스를 생성할 수 있도록 하여 기존 방법의 한계를 극복합니다.



### AdaMMS: Model Merging for Heterogeneous Multimodal Large Language Models with Unsupervised Coefficient Optimization (https://arxiv.org/abs/2503.23733)
Comments:
          CVPR 2025

- **What's New**: 최근 모델 병합(model merging) 기법이 다수의 대형 언어 모델(LLMs)을 결합하여 효과적으로 다양한 작업을 수행할 수 있음이 입증되었습니다. 그러나 기존의 모델 병합 방법들은 주로 동일한 아키텍처의 이종 모델에 초점을 맞추어 왔습니다. 이 논문에서는 이질적인 다중 모달 대형 언어 모델(MLLMs)에 특화된 새로운 모델 병합 방법인 AdaMMS(AdaMMS: Adaptive Model Merging Strategy)를 제안합니다.

- **Technical Details**: AdaMMS는 세 가지 단계인 매핑(mapping), 병합(merging), 검색(searching)으로 구성됩니다. 첫 번째 단계인 매핑에서는 서로 다른 아키텍처의 모델들 간의 매핑 함수를 설계해 병합 작업을 수행할 수 있도록 합니다. 두 번째 단계에서는 모델 가중치에 선형 보간(linear interpolation)을 적용하여 이질적인 MLLMs의 비대칭성을 조정하며, 세 번째 단계에서는 비지도 학습 방식으로 하이퍼파라미터를 선택하는 방법을 제안합니다.

- **Performance Highlights**: 많은 실험을 통해 AdaMMS가 다양한 시각-언어(vision-language) 벤치마크에서 이전의 모델 병합 방법보다 우수한 성능을 보임을 입증했습니다. 특히, Qwen 및 LLaMA 아키텍처를 기반으로 한 이질적인 MLLM 쌍에서 강력한 성능을 보여주며, 이전의 방법들과 비교해 적은 양의 데이터로도 고성능을 유지할 수 있는 안정성을 확인했습니다.



### Uni-Render: A Unified Accelerator for Real-Time Rendering Across Diverse Neural Renderers (https://arxiv.org/abs/2503.23644)
Comments:
          Accepted by HPCA'25

- **What's New**: 본 논문에서는 다양하고 일반적인 신경 렌더링 파이프라인을 지원하는 통합 신경 렌더링 가속기를 개발하였습니다. 이는 실시간(on-device) 렌더링을 가능하게 하여 여러 어플리케이션에서의 효율성과 호환성을 유지할 수 있도록 설계되었습니다. 특히, 이 가속기는 다양한 애플리케이션을 위해 고유한 렌더링 메트릭 요구사항에 맞춰 데이터 플로우를 동적으로 조정할 수 있는 리컨피규러블 하드웨어 아키텍처를 구현하고 있습니다.

- **Technical Details**: 기존의 신경 렌더링 파이프라인은 각각의 애플리케이션에 따라 다양한 게이지를 필요로 하며, 현재의 신경 렌더링 하드웨어는 이러한 다양한 요구를 만족하는 혁신적 솔루션이 부족합니다. 본 연구에서는 공통적으로 사용되는 마이크로 연산자를 기반으로 하여, 신경 렌더링 파이프라인의 요구에 맞게 데이터 플로우를 조절할 수 있는 초기 아키텍처를 제안합니다. 벤치마크 실험을 통해, 제안한 가속기는 최신 신경 렌더링 하드웨어보다 최대 119배의 속도 향상을 이루었습니다.

- **Performance Highlights**: Uni-Render라는 이름의 새로운 가속기는 다양한 렌더링 파이프라인에서 실시간(on-device) 신경 렌더링을 가능하게 하여, 차세대 신경 그래픽 애플리케이션의 발전을 이끌어갈 수 있는 잠재력을 지니고 있습니다. 논문에서 제시한 가속기는 5W의 전력 소비로 다양한 애플리케이션에 적합하며, 이는 실제 씬과 합성 씬에서 성능 및 효율성을 검증한 결과로 확인되었습니다. 이러한 성과는 30 FPS 이상의 속도로 다양한 신경 렌더링 작업을 처리할 수 있는 기회를 제공합니다.



### GenVP: Generating Visual Puzzles with Contrastive Hierarchical VAEs (https://arxiv.org/abs/2503.23598)
Comments:
          Accepted to ICLR 2025

- **What's New**: 이 논문에서는 Generative Visual Puzzles (GenVP)라는 새로운 프레임워크를 제안하여 Raven의 진행 매트릭스 (RPMs) 전체 생성 과정을 모델링합니다. 이 접근 방식은 기존의 RPM 문제 해결 방식과는 달리, 추상적인 규칙을 기반으로 완전한 새로운 퍼즐을 생성할 수 있는 능력을 갖추고 있습니다. GenVP는 기계가 기존의 퍼즐을 넘어서 일반화 할 수 있는 능력을 제공하여, AI의 창의성을 높이는 데 기여합니다.

- **Technical Details**: GenVP는 계층적 추론(hierarchical inference) 및 생성 파이프라인(generative pipeline)을 포함하는 그래픽 모델을 사용합니다. 이를 통해 Mixture of Experts (MoE) 기제를 도입하여 퍼즐 규칙 예측의 정확성을 개선하고, 불필요한 특성(노이즈)에 강건한 반응을 보입니다. 또한, novel contrastive learning scheme을 통해 교차 퍼즐 및 후보의 비교를 가능하게 하여 학습의 강건성을 높입니다.

- **Performance Highlights**: 다양한 실험에서 GenVP는 퍼즐 문제 해결 정확도뿐만 아니라 22가지 OOD(Out-Of-Distribution) 시나리오에서 SOTA 성능을 달성하였습니다. GenVP는 더욱 복잡한 해결 가능한 공간에서 효율적으로 일반화할 수 있으며, 기존의 방법들과 비교하여 문제 풀이 성능에서 확연히 우수한 결과를 보입니다. 조사 결과, GenVP는 추상 규칙을 효과적으로 이해하고 새로운 RPM을 생성하는 높은 수준의 능력을 보유하고 있습니다.



### Optimal Invariant Bases for Atomistic Machine Learning (https://arxiv.org/abs/2503.23515)
- **What's New**: 이번 연구는 원자 구조 분석을 위한 기계 학습 모델의 개선을 목표로 하며, 기존의 불완전한 descriptor를 제거하여 완전성을 만족하는 최소한의 descriptor 집합을 생성합니다. 이를 통해 연산 비용을 줄이고, 서로 다른 원자 환경을 효과적으로 구분할 수 있는 더 효율적인 모델을 제안합니다. 새롭게 개발된 message-passing network 구조는 최적의 Cartesian tensor invariants를 활용하여 각 뉴런에서 5-body 패턴을 인식할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: 연구에서는 패턴 인식 기술을 활용하여 descriptor의 기능적 의존성을 제거하고, 완전한 원자 환경을 설명하는 데 필요한 최소한의 함수 집합을 도출합니다. Atomistic Cluster Expansion (ACE)와 같은 기존의 설명 방식을 개선하고, 미완전한 신경망 구조를 증강하여 새로운 네트워크 아키텍처를 제안합니다. 이는 효율적인 descriptor 집합을 통해 높은 정확도를 유지하면서 낮은 계산 비용으로 실행될 수 있습니다.

- **Performance Highlights**: 제안된 모델은 최첨단 벤치마크에서 강력한 정확도를 보이며, 기능적으로 독립적인 subset을 사용하여 다양한 단일 원소 재료의 힘 예측에서 우수한 성능을 제공합니다. 이러한 결과는 비선형 ACE 형태에 특히 흥미로운 성격을 지니며, 완전한 descriptor 집합을 활용하는 것이 원자 환경 함수의 임의적 복제를 가능하게 함을 시사합니다.



### Visual Acuity Consistent Foveated Rendering towards Retinal Resolution (https://arxiv.org/abs/2503.23410)
- **What's New**: 본 논문에서는 시각적 정확성에 일관성을 부여하는 고유한 foveated rendering 기법인 Visual Acuity-Consistent Foveated Rendering (VaFR)을 제안합니다. 이전의 foveated rendering 방법들은 디스플레이 해상도의 증가에 따라 성능이 저하되는 문제를 겪었으나, 본 연구는 인간 시각 시스템(Human Visual System, HVS)의 인식을 기반으로 향상된 렌더링 성능을 달성합니다. 주목할 점은, 이 새로운 방법이 장치의 해상도에 독립적이며, 높은 해상도의 VR HMD에서 효과적으로 작동하도록 설계되었습니다.

- **Technical Details**: VaFR 접근법은 인간 시각 정확성 모델에서 유도된 새로운 log-polar mapping 함수를 사용하여 렌더링 효율성을 극대화합니다. 이 함수는 시각적 변별력이 자연스럽게 발휘될 수 있도록 시각 신호 범위를 수용하며, 아울러 일관된 렌더링 정보 출력을 보장합니다. 또한, VaFR은 log-polar tangent 축에서의 해상도를 인간 시각의 한계 내로 유지하며, 기존 방법들이 필요로 했던 장치 특정 매개변수 조정을 없앱니다.

- **Performance Highlights**: VaFR은 다양한 테스트 시나리오에서 기존 foveated rendering 방법들보다 우수한 지각적 시각 품질을 제공합니다. 특히, 3D 시나리오의 deferred rendering에서 6.5배에서 9.29배의 속도 향상을 이룩하였고, retinal 해상도에서 ray-casting을 통해서는 10.4배에서 16.4배의 성능 향상을 보였습니다. 또한, 8K 경로 추적을 통한 출력 프레임 속도가 부드럽게 유지되는 등의 결과를 달성하였습니다.



### Physically Ground Commonsense Knowledge for Articulated Object Manipulation with Analytic Concepts (https://arxiv.org/abs/2503.23348)
- **What's New**: 본 논문은 인간의 상식 지식을 활용하여 로봇의 물체 조작 기술을 발전시키기 위한 새로운 방법론을 제안합니다. 이 연구는 최근의 언어 모델들이 이해하는 의미 정보를 실제 물리 세계에 접목시키는 데 어려움이 있음을 지적하며, 이를 해결하기 위해 수학적 기호로 정의된 분석 개념을 도입했습니다. 이러한 접근 방식은 로봇이 물체의 구조와 기능을 이해할 수 있도록 돕고, 일반화된 조작 정책을 수립하는 데 기여합니다.

- **Technical Details**: 분석 개념은 세 가지 구성 요소로 이루어져 있습니다: 개념 정체성(concept identity), 분석 구조 지식(analytic structural knowledge), 그리고 분석 조작 지식(analytic manipulation knowledge)입니다. 각 개념은 고유한 기호를 가지며, 수학적 절차를 통해 물체의 공간 구조와 기능적 속성을 정량적으로 표현합니다. 이 방법론은 상호 작용이重要한 물리적 세부사항을 강조하는 데 중점을 두며, 이를 통해 로봇은 물체의 정확한 조작을 수행할 수 있게 됩니다.

- **Performance Highlights**: 상세한 시뮬레이션 및 실제 환경에서의 실험 결과, 본 접근법은 기존의 LLM 기반 방법론과 비교하여 기계 조작 성능에서 우수함을 입증했습니다. 특히, 제안된 절차적 방법이 로봇이 물체를 더 정확하고 해석 가능한 방식으로 조작할 수 있도록 돕는다는 점에서 큰 의의를 갖습니다. 이 연구는 다양한 유형의 물체 조작 작업에 성공적으로 적용될 수 있는 잠재력을 보여줍니다.



### Beyond Unimodal Boundaries: Generative Recommendation with Multimodal Semantics (https://arxiv.org/abs/2503.23333)
- **What's New**: 이번 논문에서는 Multimodal Generative Recommendation (MGR)이라는 새로운 접근 방식을 제안합니다. 기존의 Generative Recommendation (GR) 방법들이 주로 unimodal 데이터를 사용하였던 한계를 극복하여, 다양한 모달리티를 통합하는 방법론을 다룹니다. 저자들은 특히 모달리티 선택의 중요성과 그것이 GR 모델의 성능에 미치는 영향을 강조하고 있습니다.

- **Technical Details**: MGR-LF++라는 새로운 Late Fusion 프레임워크를 도입하여, 서로 다른 모달리티 정보를 효과적으로 관리하는 방법을 제안합니다. 이 프레임워크는 contrastive modality alignment 훈련 기법과 각 모달리티를 구분하는 특별한 토큰을 사용하여, 서로 다른 semantic IDs의 일치를 도모합니다. 이를 통해, 다양한 모달리티의 정보를 손실 없이 통합할 수 있는 방법을 모색합니다.

- **Performance Highlights**: MGR-LF++는 기존의 unimodal 접근 방법 대비 20% 이상의 성능 향상을 달성하였습니다. 저자들은 6개의 기준선 모델을 사용하여 3개 데이터셋에서 실험을 실시하였으며, 그 결과 다중 모달리티 정보를 활용하는 것이 Generative Recommendation의 효과를 크게 향상시킬 수 있음을 입증했습니다.



### LaViC: Adapting Large Vision-Language Models to Visually-Aware Conversational Recommendation (https://arxiv.org/abs/2503.23312)
- **What's New**: 이 연구에서는 시각적 정보를 통합한 새로운 대화형 추천 시스템인 LaViC (Large Vision-Language Conversational Recommendation Framework)를 제안합니다. LaViC는 대화 중 사용자의 요구를 파악하고 더 개인화된 제안을 제공하기 위해 이미지 표현을 효율적으로 처리하는 방법을 도입합니다. 이를 통해 패션과 인테리어와 같은 시각적 요구가 중요한 분야에서 효과적인 추천을 가능하게 합니다.

- **Technical Details**: LaViC의 기법은 두 단계로 구성됩니다. 첫 번째 단계인 'visual knowledge self-distillation'은 수백 개의 토큰으로 이루어진 제품 이미지를 소수의 시각적 토큰으로 압축하여 계산 비용을 줄입니다. 두 번째 단계인 'recommendation prompt tuning'에서는 대화 맥락과 증류된 시각적 토큰을 결합해 텍스트와 시각적 특징을 함께 포착하는 통합 메커니즘을 제공합니다.

- **Performance Highlights**: 광범위한 실험을 통해 LaViC는 텍스트 기반 대화형 추천 방법 및 오픈 소스 비전-언어 모델보다 뛰어난 성능을 보였습니다. LaViC는 또한 GPT-3.5-turbo와 같은 유명한 상용 모델에 비해 경쟁력 있는 정확도를 기록하며, 비주얼 데이터의 중요성을 강조하고 효과적인 비전-언어 통합의 가능성을 보여주었습니다. 이 연구는 새로운 'Reddit-Amazon' 데이터세트를 통해 진정한 사용자 쿼리를 기반으로 상호작용할 수 있는 기회를 제공합니다.



### SketchVideo: Sketch-based Video Generation and Editing (https://arxiv.org/abs/2503.23284)
Comments:
          CVPR 2025

- **What's New**: 이 논문에서는 텍스트 프롬프트나 이미지에 조건화된 비디오 생성 및 편집을 위한 새로운 스케치 기반 방법론을 제안합니다. 기존의 방법들은 장면 레이아웃이나 기하학적 세부 사항을 효과적으로 제어하는 데 한계가 있었으며, 본 연구는 이러한 문제를 해결하여 더욱 세밀한 로컬 수정과 동작 제어를 지원합니다. 새로운 스케치 조건 네트워크를 통해 DiT(Deep Image Translation) 기반 비디오 생성 아키텍처의 효율성을 크게 높이는 것을 목표로 하고 있습니다.

- **Technical Details**: 스케치 기반 인터랙션을 통해 사용자는 키프레임(해당 시간의 특정 프레임)에 스케치를 그려 원하는 컨텐츠 구조와 동작 정보를 효과적으로 전달할 수 있습니다. 이를 위해 저자는 새로운 inter-frame attention 메커니즘을 설계하여 모든 비디오 프레임 간의 관계를 분석하고, 스케치 입력에 기반하여 추가적인 비디오 삽입 모듈을 통해 원본 비디오의 공간적 특성과 동적 움직임을 일관성 있게 유지합니다. 이러한 설계로 인해 비디오 생성 과정에서 메모리 효율성을 극대화할 수 있습니다.

- **Performance Highlights**: 다양한 실험 결과, SketchVideo는 기존의 비디오 생성 및 편집 방식에 비해 뛰어난 성능을 보여줍니다. 특히, 스케치 기반 비디오 생성의 정밀한 제어와 로컬 수정의 유연성을 제공하여 창의적인 비디오 제작을 가능하게 하였습니다. 이러한 결과는 기존 기술들이 해결하지 못한 비디오 품질 저하 문제를 효과적으로 극복했음을 보여줍니다.



### A Lightweight Image Super-Resolution Transformer Trained on Low-Resolution Images Only (https://arxiv.org/abs/2503.23265)
- **What's New**: 이 논문은 LR(저해상도) 이미지만을 이용한 학습 방법을 채택하여 경량화된 비전 변환기 모델인 SwinIR을 활용한 최초의 연구입니다. 전통적인 SISR(Single-Image Super-Resolution) 모델들은 HR(고해상도) 이미지에 대한 대량의 훈련 데이터를 요구하지만, 본 연구는 LR만으로도 효과적으로 작업을 수행할 수 있음을 보여줍니다.

- **Technical Details**: 논문에서 제안한 MSTbic는 미세 이미지 초해상도에서 개발된 LR-only 다중 스케일 훈련 방법을 큰 규모의 실제 데이터로 적용한 것입니다. SwinIR 모델은 얕은 특징 추출 모듈과 깊은 특징 추출 파트, HR 이미지 재구성 모듈로 구성되어 있으며, 자기 주의(attention) 및 잔차 연결(residual connection)을 사용하여 기존의 CNN 구조와 비교해 뛰어난 성능을 나타냅니다.

- **Performance Highlights**: SwinIR 및 CNN 모델을 모두 비교한 결과, 제안된 방법이 기존의 CNN 기반 LR-only SISR 방법들에 비해 우수한 성능을 보였음이 입증되었습니다. 이는 Set5, Set14, BSD100, Urban100 및 Manga109와 같은 전통적인 SR 벤치마크 데이터셋에서 확인되었으며, 새로운 최첨단 성과를 달성한 것이 특징입니다.



### Geometry in Style: 3D Stylization via Surface Normal Deformation (https://arxiv.org/abs/2503.23241)
Comments:
          CVPR 2025. Our project page is at this https URL

- **What's New**: 이번 논문에서는 Geometry in Style이라는 새로운 메타드를 제안합니다. 기존의 기술들은 원형을 과도하게 제한하는 변형 방식이나 입력 형태를 크게 수정하는 방식으로 그 정체성을 손상시키는 경우가 있었습니다. 반면, 이 방법은 삼각형 메쉬의 변형을 각 정점의 주변에서 목표 법선 벡터로 표현하여 정체성을 유지하면서도 독창적인 스타일을 생성할 수 있습니다. 이것은 인간 조각가가 조각하는 과정과 유사한 방식으로 진행됩니다.

- **Technical Details**: Geometry in Style은 차별화 가능한 As-Rigid-As-Possible(dARAP) 레이어를 사용하여 각 정점의 법선을 목표 법선으로 적합하게 변형합니다. dARAP는 로컬 회전과 글로벌 단계를 통합하여 변형을 최적화하고, 이는 신경망의 레이어로 쉽게 통합될 수 있습니다. 이 접근 방식은 단 한번의 반복을 통해 고품질의 변형 결과를 제공하며, 비주얼 로스(visual loss)를 통해 텍스트 프롬프트와 조화롭게 변형을 유도합니다. 따라서 사용자는 다양한 형태에 쉽게 스타일을 적용할 수 있습니다.

- **Performance Highlights**: 제안된 방법은 이전의 변형 기술들보다 더 낮은 표면 왜곡(surface distortion)으로 목표 스타일을 실현할 수 있습니다. Geometry in Style 메소드는 사용자가 제공하는 텍스트 프롬프트를 통해 스타일을 지시하여 다양한 정체성을 유지하며, 스타일화 결과는 부분별로 조절할 수 있는 장점을 가지고 있습니다. 우리의 연구는 인풋 형태의 정체성을 충실히 따르는 고품질의 스타일화를 가능하게 하며, 간단하고 쉽게 구현할 수 있는 프레임워크로 소개됩니다.



### Aurelia: Test-time Reasoning Distillation in Audio-Visual LLMs (https://arxiv.org/abs/2503.23219)
- **What's New**: 이 논문에서는 오디오-비주얼 레이징(AV reasoning)의 복잡성을 해결하기 위해 새로운 프레임워크인 AURELIA를 소개합니다. AURELIA는 actor-critic (행동-비평자) 기반의 접근 방식을 활용하여, 테스트 시간에 AVLLMs(오디오-비주얼 대형 언어 모델)의 단계적 레이징 기능을 증진합니다. 논문은 또한 4500개의 오디오-비주얼 질문과 이에 대한 상세한 단계적 레이징을 제공하는 AVReasonBench 기준점을 제시합니다.

- **Technical Details**: AURELIA는 LLM(대형 언어 모델)의 추론 기능을 활용하여 다중 모드 오디오-비디오 이해를 위한 고품질 레이징 데이터를 생성하는 인터랙티브한 프레임워크입니다. 이 시스템은 오디오와 비주얼 신호의 상호 작용을 고려하여 각기 다른 레이징 작업을 수행하며, 기존의 AVLLMs가 가진 편향성을 줄이는 데 기여합니다. 또한, AVReasonBench는 다양한 레이징 능력을 평가하는 데 필요한 포괄적인 벤치마크를 제공합니다.

- **Performance Highlights**: AURELIA를 활용하여 AVLLMs의 성능이 최대 100% 상대적으로 개선되는 것을 확인하였습니다. AVReasonBench의 18개 기존 AVLLM에 대한 평가 결과, 이 모델들이 비디오와 오디오 데이터를 처리하는 데 현저한 한계를 보임을 보여주었습니다. 이러한 성능 증가는 오디오-비주얼 레이징의 중요성과 현실 세계에서의 응용 가능성을 강조합니다.



### OncoReg: Medical Image Registration for Oncological Challenges (https://arxiv.org/abs/2503.23179)
Comments:
          26 pages, 6 figures

- **What's New**: OncoReg Challenge는 현대 암 연구에서 의료 데이터의 활용을 극대화할 수 있는 두 단계의 프레임워크를 제공하여 연구자들이 이미지 등록(image registration) 방법을 개발하고 검증할 수 있도록 합니다. 첫 번째 단계에서는 공개 데이터셋을 사용하여 알고리즘을 설계 및 개선하고, 두 번째 단계에서는 안전한 병원 네트워크 내의 비공식 데이터셋에서 모델을 훈련시킵니다. 이 접근법은 환자의 프라이버시를 보호하면서도 더 일반화 가능한 AI 모델의 개발을 촉진합니다.

- **Technical Details**: OncoReg Challenge는 기존 Learn2Reg Challenge를 기반으로 하여 interventional cone-beam CT와 standard planning fan-beam CT 이미지를 결합하여 방사선 치료(radiotherapy)에서의 여러 도전에 대응합니다. 이미지 등록의 정확성은 암 치료에 있어 매우 중요하며, 치료의 동적 조정을 위한 정밀한 정렬이 필요합니다. 특히 저선량 개입형 cone-beam CT와 고해상도 진단용 fan-beam CT의 등록 과정은 방사선 치료의 효과성을 높이기 위해 필수적입니다.

- **Performance Highlights**: 이번 연구에서는 OncoReg Challenge의 참가자들이 제출한 알고리즘의 결과를 자세히 분석하였으며, feature extraction의 중요성을 강조했습니다. 새로운 방법은 그 다재다능성을 보여주었고, 기존 방법들도 여전히 유사한 성능을 유지했습니다. 딥러닝(deep learning)과 전통적인 방법이 모두 이미지 등록에서 중요한 역할을 하며, 특히 feature extraction의 조합이 가장 효과적임을 입증했습니다.



### Prediction of 30-day hospital readmission with clinical notes and EHR information (https://arxiv.org/abs/2503.23050)
- **What's New**: 이 논문은 임상 노트(clinical notes)와 전자 건강 기록(EHR)을 결합하여 30일 재입원을 예측하는 새로운 모델을 제안합니다. 병원 재입원률은 병원 치료의 질을 나타내는 지표로 간주되며, 이 모델은 의료 전문가가 환자의 건강 상황을 조기에 파악하고 필요한 치료를 제공할 수 있도록 지원할 수 있습니다. 이를 위해, 다양한 정보 유형에 대한 표현 방식을 탐구하며 그래프 신경망(GNN)을 사용하여 환자 데이터를 구조화하여 대규모 환자 코호트를 처리합니다.

- **Technical Details**: 이 연구에서 사용된 데이터셋은 MIMIC-IV 버전 2.2로, 2008년부터 2019년까지의 재원 환자 정보를 포함합니다. GNN 모델을 기반으로 하여, 각 입원을 그래프의 노드로 간주하고 유사한 특성을 가진 입원들을 연결합니다. 이는 환자 간의 복잡한 관계를 모델링할 수 있게 해주며, 임상 노트는 비구조적인 데이터로서 이 모형에서 중요한 역할을 합니다.

- **Performance Highlights**: 제안된 모델은 AUROC 0.72를 기록하고, 균형 정확도(balanced accuracy)는 66.7%로, 다양한 정보를 결합하는 것이 재입원 예측에 있어 중요함을 강조합니다. 전통적인 기계 학습(machin learning) 모델과 비교했을 때, 더 복잡한 관계를 캡처할 수 있는 GNN의 장점을 활용하여 높은 성능을 보여줍니다.



### MIL vs. Aggregation: Evaluating Patient-Level Survival Prediction Strategies Using Graph-Based Learning (https://arxiv.org/abs/2503.23042)
- **What's New**: 이번 연구에서는 종양 이질성(tumor heterogeneity)과 환자 내 변동성(intra-patient variability) 문제로 인해 암 환자의 예후 예측이 어렵다는 점을 다루고 있습니다. 이를 위해 우리는 Whole-Slide Images (WSIs)를 사용하여 생존 예측에서 가장 대표적인 슬라이드를 선택하는 방법을 모색했습니다. 마지막으로, 여러 WSI를 함께 고려하거나 자동으로 가장 중요한 슬라이드를 식별하는 다중 인스턴스 학습(multiple-instance learning, MIL) 방법을 통해 접근했습니다.

- **Technical Details**: 연구팀은 MMIST-ccRCC 데이터셋을 활용하여 명확 세포 신장 세포 암(ccRCC) 환자들을 대상으로 실험을 진행했습니다. 여러 가지 그래프 신경망(Graph Neural Networks) 아키텍처를 평가하여 다양한 생존 예측 전략을 비교하였습니다. WSI를 독립적인 샘플로 취급하는 기존 방법과, 여러 슬라이드의 예측을 집계하거나 가장 관련 있는 슬라이드를 자동으로 식별하는 전략이 포함되었습니다.

- **Performance Highlights**: 실험 결과, MIL 기반 선택 방법이 정확도를 향상시키는 것으로 나타났습니다. 이는 가장 대표적인 슬라이드를 선택하는 것이 생존 예측에 유리하다는 것을 강조합니다. 본 연구의 결과는 암 환자의 치료 결정 과정에서 보다 효과적인 데이터 활용 방안을 제시할 가능성을 보여줍니다.



### Towards Mobile Sensing with Event Cameras on High-mobility Resource-constrained Devices: A Survey (https://arxiv.org/abs/2503.22943)
Comments:
          32 pages, 9 figures

- **What's New**: 이 논문은 2014-2024년 동안의 논문을 조사하고, 이벤트 기반 모바일 센싱 시스템에 대한 포괄적인 개요를 제공합니다. 이벤트 기반 비전 기술의 장점을 강조하며, 고속으로 정확한 감지를 요구하는 개발되는 모바일 디바이스의 필요성에 맞춘 자료를 작성했습니다. 이 논문은 또한 모바일 센싱에서의 이벤트 카메라의 주요 응용프로그램과 관련된 도전 과제를 다루고 있으며, 미래 연구 방향을 제안합니다.

- **Technical Details**: 모바일 센싱의 기술적 세부사항으로는 이벤트 카메라가 제공하는 μs(마이크로초) 수준의 시간 해상도와 저지연(делay)이 있습니다. 모바일 디바이스는 환경 변화를 즉각적으로 보고할 수 있으며, 140 dB의 높은 동적 범위를 통해 다양한 조명 조건에서도 효과적으로 작동할 수 있습니다. 그러나 이벤트 기반 데이터 처리는 조명 변화에 민감하고, 고유한 의미 정보를 결여하여 feature extraction이 실패할 수 있으며, 데이터 양이 상당히 방대하여 모바일 디바이스의 계산 부담을 초과합니다.

- **Performance Highlights**: 이벤트 카메라는 저전력 소비(예: 0.5 W)가 가능하여 효율적으로 설계된 모바일 디바이스에 적합합니다. 이 기술은 고속 작업 및 효율성의 발전을 위한 모바일 디바이스에 힘을 실어주는 유망한 기술로 자리 잡고 있습니다. 모바일 센싱 작업의 다양한 목표에 맞도록 이벤트 기반 데이터 처리 알고리즘이 중요한 역할을 하며, 고속의 정확한 데이터 수집과 저지연 특성을 통해 모바일 디바이스의 성능을 향상시킬 수 있습니다.



### VizFlyt: Perception-centric Pedagogical Framework For Autonomous Aerial Robots (https://arxiv.org/abs/2503.22876)
Comments:
          Accepted at ICRA 2025. Projected Page: this https URL

- **What's New**: 이 논문에서는 자율 항공 로봇을 교육하기 위한 새로운 오픈소스 프레임워크인 	extit{VizFlyt}를 소개합니다. 	extit{VizFlyt}는 3D Gaussian Splatting을 사용하여 실시간으로 포토리얼리스틱한 시각 센서를 생성하며, 항공 로봇의 자율성 알고리즘 테스트를 안전하게 지원합니다. 이 프레임워크는 핸즈온 교육 과정에서 필요했던 하드웨어 의존성을 줄여주고, 다양한 환경에서의 테스트 가능성을 높입니다.

- **Technical Details**: 	extit{VizFlyt}는 Hardware-In-The-Loop (HITL) 접근 방식을 적용하여 로봇의 포즈(pose)를 기반으로 시각 정보를 실시간으로 합성합니다. 시각 정보는 3D Gaussian Splatting 기술을 통해 생성되며, RGBD 데이터 업데이트는 100Hz를 초과합니다. 이 시스템은 센서 퓨전 및 자율 주행 교육에 유용하도록 설계되어 있으며, 다양한 센서 및 데이터 수집 장치를 통합할 수 있도록 모듈화 되어 있습니다.

- **Performance Highlights**: 이 논문에서는 	extit{VizFlyt} 프레임워크를 사용하여 여러 실제 HITL 실험을 통해 항공 로봇 교육의 효과성을 입증하였습니다. 주요 작업으로는 Visual Odometry, 고속 장애물 회피, 알려진 및 미지의 간격을 통한 내비게이션 등을 포함합니다. 이를 통해 학생들이 자율성 개념을 다양한 수준에서 실습하며 익힐 수 있는 기회를 제공, 객체 인식 및 비행 제어에 필요한 기술을 개발할 수 있도록 하였습니다.



### Nonhuman Primate Brain Tissue Segmentation Using a Transfer Learning Approach (https://arxiv.org/abs/2503.22829)
- **What's New**: 본 연구에서는 비인간 영장류(Non-Human Primates, NHP)의 뇌 조직 분할을 향상시키기 위한 새로운 접근방식으로 STU-Net과 transfer learning을 결합한 방법을 제안합니다. 이는 인간의 뇌 MRI 데이터에서 전이된 지식을 활용하여 NHP 뇌 MRI의 분할 정확도를 높이는데 기여합니다. 주목할 만한 점은 기존의 한계를 극복하여 NHP 뇌 특유의 미세 해부학적 세부 사항을 효과적으로 포착할 수 있다는 점입니다.

- **Technical Details**: 비인간 영장류의 뇌 이미지는 주로 해부학적 차이와 해상도 제한으로 인해 분할이 어렵습니다. 연구에서는 STU-Net을 통해 이러한 도전을 해결하고, 특히 작은 피질 하 구조체인 피각(putamen)과 시상(thalamus) 등의 분할 성능을 향상시켰습니다. 최종적으로, DSC(다중 클래스의 Dice Similarity Coefficient)는 0.88 이상, IoU(Intersection over Union)는 0.8 이상, 그리고 HD95는 7 이하의 성능을 달성했습니다.

- **Performance Highlights**: 제안된 방법은 비인간 영장류의 뇌 조직 분할에서 새로운 기준을 제시합니다. 특히 작은 구조물의 분할에서 기존의 한계를 초월한 성과를 보였으며, 이는 진화 신경 과학 및 인류 건강에 관련된 신경 질환의 전임상 연구를 가속화할 잠재력을 가지고 있습니다. 나아가, 이 연구는 다중 클래스 뇌 조직 분할에 대한 강력한 방법론을 제시하여 향후 연구의 발전에 기여할 것입니다.



### Adaptive Integrated Layered Attention (AILA) (https://arxiv.org/abs/2503.22742)
- **What's New**: 이번 연구에서는 Adaptive Integrated Layered Attention (AILA)라는 신경망 아키텍처를 제안합니다. AILA는 다양한 네트워크 층 간의 적응형(feature reuse) 기능을 위해 밀집 스킵 연결(dense skip connections)과 여러 메커니즘을 융합하여 구성되어 있습니다. AILA는 가격 예측, 이미지 인식, 감정 분석의 세 가지 도전 과제를 평가받았으며, 기존의 강력한 딥러닝 모델과 유사한 성능을 보이면서도 훈련 및 추론 시간을 크게 단축시켰습니다.

- **Technical Details**: AILA는 두 가지 아키텍처, 즉 AILA-Architecture 1과 AILA-Architecture 2로 나뉘어 있습니다. AILA-Architecture 1은 층 간의 연결 메커니즘으로 간단한 선형 층(linear layers)을 사용하고, AILA-Architecture 2는_attention_ 메커니즘을 구현하여 이전 층의 출력을 선택적으로 강조합니다. 이러한 아키텍처는 각기 다른 태스크에 대해 개별적으로 훈련되며, 다양한 네트워크 깊이에서 관련 기능을 유연하게 재사용함으로써, 강력한 성능 향상을 이루어냅니다.

- **Performance Highlights**: AILA는 세 가지 기준 벤치마크에서 강력한 성능 지표를 달성했습니다. 가격 예측, CIFAR-10 데이터셋에 대한 이미지 인식, IMDB 영화 리뷰 데이터셋의 감정 분석에서 AILA-Architecture 1 및 2 모두 LSTM, Transformer, CNN과 같은 기존의 강력한 기준 모델과 경쟁하며 이를 초월하는 성과를 보여주었습니다. 결과적으로 AILA는 일반적인 고정 연결 방식이 아닌, 적응형 정보 흐름을 통해 복잡한 태스크에서 성능을 향상시키는 새로운 길을 열었습니다.



### Ancestral Mamba: Enhancing Selective Discriminant Space Model with Online Visual Prototype Learning for Efficient and Robust Discriminant Approach (https://arxiv.org/abs/2503.22729)
Comments:
          10 pages, 3 figures

- **What's New**: 이 논문에서는 Ancestral Mamba라는 새로운 접근 방식을 제안합니다. 이 방법은 온라인 프로토타입 학습을 선택적 차별 모델(Selective Discriminant Space Model)에 통합하여 효율적이고 강력한 온라인 지속 학습에 기여합니다. Ancestral Prototype Adaptation(APA)와 Mamba Feedback(MF)이라는 두 가지 핵심 요소가 포함되어 있어, 이전 지식을 보존하며 새로운 도전 과제에 적응하는 능력을 강화합니다.

- **Technical Details**: Ancestral Mamba는 비슷한 기술을 대체하기 위해 온라인 프로토타입 학습 기술을 적용합니다. APA 모듈은 클래스의 본질적인 특성을 포착하는 프로토타입을 지속적으로 학습하고 유지합니다. MF 메커니즘은 도전적인 패턴에 집중하여 의사결정 경계를 정제하는 시각적 피드백 루프 역할을 합니다.

- **Performance Highlights**: CIFAR-10 및 CIFAR-100 데이터셋에 대한 광범위한 실험을 통해 Ancestral Mamba가 기존의 최첨단 모델(State-Of-The-Art)과 비교하여 정확도 및 잊기 방지 측면에서 뛰어난 성능을 발휘함을 입증했습니다. 이 접근 방식은 진화하는 시각 패턴에 적응하는 능력을 보여주며, 효율성과 견고함을 함께 제공합니다.



### Dual Audio-Centric Modality Coupling for Talking Head Generation (https://arxiv.org/abs/2503.22728)
Comments:
          9 pages, 4 figures

- **What's New**: 이번 논문에서는 오디오 기반의 토킹 헤드 비디오 생성의 새로운 접근 방식인 Dual Audio-Centric Modality Coupling (DAMC) 프레임워크를 제안합니다. 이 프레임워크는 오디오 입력의 콘텐츠 및 동적 특징을 효과적으로 통합하여, 전통적인 방식에서 발생하는 입술 동기화(lip synchronization) 문제를 해결합니다. 두 가지 인코더 구조를 활용하여 콘텐츠 지각(Content-Aware)과 시각적 동기화(Dynamic-Sync)를 각각 구현하며, 이 과정을 통해 비디오 품질을 크게 향상시킵니다.

- **Technical Details**: DAMC 프레임워크는 Content-Aware Encoder와 Dynamic-Sync Encoder를 중심으로 구성되며, 각각은 오디오 입력에서 의미론적 콘텐츠와 동적 특징을 추출합니다. Cross-Synchronized Fusion Module (CSFM)을 통해 이 두 가지 특징을 융합하여, 내용 표현(content representation)과 입술 동기화(lip synchronization)를 강화합니다. 본 연구에서는 TTS(텍스트-음성 변환) 모듈의 통합을 통해 합성 음성에 대한 모델의 적응성을 높이는 방안도 설명합니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안하는 방법이 입술 동기화 정확도 및 이미지 품질과 같은 주요 메트릭에서 기존 최첨단 방법을 능가함을 입증하였습니다. 특히, 텍스트-음성 변환(TTS) 시스템의 합성 음성 데이터를 포함한 다양한 오디오 입력에 대해 강력한 일반화 성능을 보여줍니다. 이러한 결과는 고품질의 오디오 기반 토킹 헤드 생성에 유망한 해결책이 될 수 있음을 시사합니다.



### Hierarchical Adaptive Expert for Multimodal Sentiment Analysis (https://arxiv.org/abs/2503.22715)
Comments:
          11 pages, 3 figures

- **What's New**: 최근 멀티모달 감정 분석(multimodal sentiment analysis)은 다양한 통신 채널에서 인간의 감정을 이해하는 데 중요한 도구로 부각되고 있습니다. 기존 방법이 외형적으로는 많은 발전을 이루었으나, 각각의 모달리티에서 정보를 효과적으로 통합하거나 차별화하는 데 어려움을 겪고 있습니다. 이를 해결하기 위해 새로운 프레임워크인 HAEMSA(Hierarchical Adaptive Expert for Multimodal Sentiment Analysis)가 제안되었습니다.

- **Technical Details**: HAEMSA는 진화 최적화(evolutionary optimization)와 다중 작업 학습(multi-task learning) 기법을 결합하여 고유의 계층 구조를 형성합니다. 각 계층은 글로벌 및 로컬의 모달리티 표현을 캡처하고, 이를 통해 감정 분석의 정확성을 높입니다. 이 프레임워크는 불완전한 모달리티 조합 시나리오에서도 적응이 가능하며, 다양한 데이터로부터 효과적인 학습을 수행합니다.

- **Performance Highlights**: HAEMSA는 여러 벤치마크 데이터셋을 통해 우수한 성능을 입증했습니다. CMU-MOSEI에서는 7-class 정확도에서 2.6% 상승을 기록했으며, CMU-MOSI에서는 6.3% 향상되었습니다. 전반적으로 HAEMSA는 감정 인식(emotion recognition)에서도 최첨단 기술 대비 2.84% 개선된 weighted-F1 점수를 달성하여 복잡한 멀티모달 상호작용을 효과적으로 캡처하는 능력을 보여줍니다.



### TRIDIS: A Comprehensive Medieval and Early Modern Corpus for HTR and NER (https://arxiv.org/abs/2503.22714)
Comments:
          6 pages, 3 figures, 2 tables

- **What's New**: 이 논문에서는 TRIDIS(Tria Digita Scribunt)라는 오픈 소스의 중세 및 초기 현대 원고 코퍼스를 소개합니다. TRIDIS는 다양한 레거시 컬렉션을 통합하여 메타데이터 설명을 포함하고 있으며, 이전의 연구에서 일부 하위 집합이 활용되었지만, 이번에는 전체 코퍼스를 통합적으로 설명합니다. 새로운 전반적인 코퍼스 구성은 문학적 전통을 연구하는 데 필요한 언어적 측면을 고려하여 설계되었습니다.

- **Technical Details**: TRIDIS는 여러 개의 오픈 소스 하위 컬렉션을 결합하여 구성되며, 반영된 데이터는 공통된 스키마를 따른 일관된 구조로 포장됩니다. 또한, 연구에서는 고유한 Outlier-driven partition 전략을 제안하여 훈련 데이터와 테스트 데이터 간의 도메인 중첩 문제를 해결하고, 복잡한 레이아웃과 드문 어휘를 가진 예제를 테스트 세트로 정의하여 HTR 모델의 일반화 능력을 평가합니다. 이러한 접근 방식은 TrOCR 및 MiniCPM2.5와 같은 사전 훈련 모델을 사용하여 검증되었습니다.

- **Performance Highlights**: TRIDIS 코퍼스는 다체로운 문서 유형을 다루며, 복잡한 레이아웃과 큰 필기 변동성을 갖춘 자료에 중점을 둡니다. 초기 실험 결과는 outlier-driven 테스트 분할을 사용할 때 HTR 모델의 성능이 크게 저하된다는 것을 보여주며, 엄격한 평가 방법론의 중요성을 강조합니다. 이를 통해 HTR 훈련 및 평가에서 이전에 간과된 난제를 드러내고, 전통적인 HTR뿐만 아니라 후속 NLP 작업을 지원하는 리소스를 제공합니다.



### Chirp Localization via Fine-Tuned Transformer Model: A Proof-of-Concept Study (https://arxiv.org/abs/2503.22713)
Comments:
          19 pages, 8 figures

- **What's New**: 이번 연구는 electroencephalogram (EEG) 스펙트로그램에서 특이한 환율(Chirp-like patterns)을 감지하기 위한 자동화 도구를 개발하는 데 초점을 맞추고 있습니다. 기존의 연구에서는 이러한 패턴을 발견하는 방법이 부족했으며, 본 연구는 Vision Transformer (ViT) 모델의 세부 조정을 통해 이를 해결하고자 하였습니다. 이와 더불어 저차원 적응(Low-Rank Adaptation, LoRA)을 활용하여 모델의 적응 속도를 향상시켰습니다.

- **Technical Details**: 연구에서는 100,000개의 합성 스펙트로그램을 생성하여 환율 로컬리제이션(chirp localization)을 위한 최초의 대규모 벤치마크를 구축했습니다. 이 스펙트로그램들은 선형 및 지수적 주파수 스윕( frequency sweep)과 가우시안 노이즈, 스무딩 기술을 사용하여 신경 환율(neural chirps)을 모방합니다. ViT 모델은 회귀(regression)에 맞게 조정되었으며, MSE 손실 및 AdamW 최적화를 통해 훈련되었고, 학습률 스케줄러와 얼리 스탑을 적용하여 과적합을 방지하였습니다.

- **Performance Highlights**: 모델의 성능은 예측된 라벨과 실제 라벨 간의 Pearson 상관관계를 통해 평가되었습니다. 결과는 환율 시작 시간에 대해 0.9841의 강한 상관관계를 보이며, 추론 시간은 137초에서 140초로 안정적인 결과를 나타냈습니다. 이러한 접근 방식은 EEG의 시간-주파수 표현(TFR)에서 환율 분석을 위한 효율적인 도구를 제공하며, 방법론적 공백을 메울 수 있을 것으로 기대됩니다.



### From Eye to Mind: brain2text Decoding Reveals the Neural Mechanisms of Visual Semantic Processing (https://arxiv.org/abs/2503.22697)
Comments:
          27 pages, 7 figures

- **What's New**: 이 연구에서는 감각적 경험을 의미 있는 의미 표현으로 변환하는 신경 메커니즘을 이해하기 위해 새로운 접근 방식을 제안합니다. 전통적인 뇌 디코딩 방식과는 달리, fMRI 신호를 자연 이미지의 텍스트 설명으로 직접 디코딩합니다. 이 모델은 시각적 입력 없이 훈련되었으며, 복잡한 장면의 핵심 의미 내용을Capture하는 유의미한 캡션을 생성합니다.

- **Technical Details**: 새로운 심층 학습 모델은 최첨단 의미 디코딩 성능을 달성했으며, 더 높은 수준의 시각 영역인 MT+(Middle Temporal area), 배측 경로 시각 피질, 아래쪽 두정 피질이 의미 변환에서 중요한 역할을 한다는 것을 밝혔습니다. 이 연구는 카테고리별 디코딩을 통해 생명체와 움직임과 같은 의미적 차원의 미세한 신경 표현을 설명합니다. 이러한 텍스트 기반 디코딩 접근법은 뇌의 의미 인코딩에 대한 더 직접적이고 해석 가능한 윈도우를 제공합니다.

- **Performance Highlights**: 저희 연구는 복잡한 의미 처리의 신경 기초를 탐구하는 강력한 새로운 방법론을 제공합니다. 또한, 분산 의미 네트워크에 대한 이해를 정교화하고, 뇌 기반 언어 모델의 개발에 영감을 줄 수 있습니다. 이 성과는 기존의 시각 재구성이 아닌 보다 심층적인 의미 분석을 가능하게 하여 인지 신경 과학 분야에 기여할 것으로 기대됩니다.



### Skip-Vision: Efficient and Scalable Acceleration of Vision-Language Models via Adaptive Token Skipping (https://arxiv.org/abs/2503.21817)
- **What's New**: 이번 논문에서는 Multimodal Large Language Models (MLLMs)의 훈련 및 추론 비효율성을 해결하기 위해 Skip-Vision이라는 통합 프레임워크를 제안합니다. 기존의 토큰 압축 접근 방식에 두 가지 보완적인 가속 전략을 도입하여 훈련 시간을 최대 35%, 추론 FLOPs를 75%, 지연 시간을 45%까지 감소시킵니다. 연구 결과, Skip-Vision은 기존 방법에 비해 비슷하거나 우수한 성능을 유지하면서 높은 효율성을 제공합니다.

- **Technical Details**: Skip-Vision은 Feed-Forward Network (FFN) 계산에서 불필요한 시각 토큰을 건너뛰는 Skip-FFN 전략을 활용하여 시각 정보의 중복성을 줄입니다. 훈련 과정에서 FFN 레이어를 우회하여 효율성을 높이고, 추론 시 KV-cache에서 건너뛴 키-값 쌍을 제거하여 계산 속도를 향상합니다. 이는 토큰 병합 전략을 통해 더 간소화할 수 있습니다.

- **Performance Highlights**: 실험 표본에서 Skip-Vision은 훈련 시간을 35% 단축하고, 추론 시 FLOPs를 75% 절감했습니다. 또한, 지연 시간은 45% 감소하여 MLLMs의 효율성을 크게 높였습니다. 이 연구는 대규모 멀티모달 학습을 위한 실제적인 해결책을 제공하며 데이터 활용도를 높이고 모델의 확장성을 강화하는 데 기여합니다.



### DSU-Net:An Improved U-Net Model Based on DINOv2 and SAM2 with Multi-scale Cross-model Feature Enhancemen (https://arxiv.org/abs/2503.21187)
- **What's New**: 이번 논문에서는 Meta의 Segment Anything Model(SAM) 시리즈와 DINOv2와 같은 대규모 사전 훈련 모델의 한계를 극복하기 위한 새로운 접근법을 제안합니다. 특히, 특수 분야에서의 이미지 세그멘테이션 성능 향상을 위해 다중 스케일 기능 협업 프레임워크가 도입되었습니다. 새로운 모델은 많은 매개변수로 인한 훈련 비용과 특정 도메인 특성을 나타내는 능력이 부족한 문제를 해결하는 데 중점을 두고 있습니다.

- **Technical Details**: 제안된 프레임워크에서 DINOv2와 SAM2 간의 기능 협업 메커니즘이 구축되었습니다. 이는 자기 지도 학습(self-supervised learning) 모델이 추출한 고차원 의미 기능이 다중 스케일 기능 융합을 안내합니다. 또한 경량 어댑터 모듈(adapter module)과 크로스 모달, 크로스 레이어 기능 융합(unit)을 설계하여 기본 모델 매개변수를 고정한 상태에서 크로스 도메인 지식을 주입할 수 있습니다.

- **Performance Highlights**: 이 구조는 위장 대상 탐지(camouflage target detection) 및 현저한 객체 탐지(salient object detection)와 같은 다운스트림 작업에서 기존의 최신 방법들을 능가합니다. 또한 고비용의 훈련 과정 없이 효율적인 비주얼 이미지 세그멘테이션을 가능하게 하며, 다양한 다운스트림 작업 및 특수 분야에서의 상당한 응용 가치를 입증하고 있습니다.



New uploads on arXiv(cs.AI)

### RIG: Synergizing Reasoning and Imagination in End-to-End Generalist Policy (https://arxiv.org/abs/2503.24388)
- **What's New**: 이번 논문은 복잡한 오픈 월드 환경에서 작동하는 에이전트에 필요한 상상력(imagination)과 추론(reasoning)을 통합한 최초의 정책인 RIG를 소개합니다. 이전 연구에서는 이러한 능력이 분리된 모델로 구현되었지만, RIG는 데이터 파이프라인을 통해 두 가지 능력을 효과적으로 결합하여 학습의 효율성과 일반화 능력을 향상시킵니다. 또한, RIG는 추론과 미래 이미지를 생성하는 과정을 결합하여 환경의 동역학을 명확하게 모델링합니다.

- **Technical Details**: RIG는 오토회귀 Transformer를 통해 텍스트 추론, 저수준의 행동 제어 및 이미지 생성을 학습합니다. 초기 단계에서는 기존의 데이터에서 수집한 궤적(trajectory)을 바탕으로, 텍스트 추론이 포함된 궤적을 생성하여 RIG-basic을 훈련시키고, 이후에는 상상력을 적용하여 실패한 궤적을 수정하는 RIG-lookahead를 학습합니다. 이러한 접근 방식은 궤적의 예측된 이미지를 환경 상태로 활용하여 가상 궤적을 생성하고 이를 기반으로 추론하여 행동을 예측하는 구조를 제공합니다.

- **Performance Highlights**: RIG는 마인크래프트 환경에서 광범위한 실험을 통해 현재의 최첨단 성능을 크게 향상시켰습니다. 결과적으로 111시간의 비디오로 훈련함으로써 전작들에 비해 17배 더 높은 샘플 효율성을 보여주며, 다양한 환경 상호작용과 추론 중 미리보기 단계를 조정하여 견고성과 일반화 능력이 지속적으로 향상됨을 입증하였습니다.



### ACPBench Hard: Unrestrained Reasoning about Action, Change, and Planning (https://arxiv.org/abs/2503.24378)
Comments:
          Accepted to LM4Plan@AAAI 2025

- **What's New**: 이번 논문에서는 ACPBench 데이터셋의 발전된 버전인 ACPBench Hard를 소개합니다. ACPBench Hard는 개방형 질문으로 구성되어, 모델이 적합한 답을 생성해야 하며, 이는 기존의 선택형 질문보다 더 도전적인 특성을 지닙니다. 이러한 변화는 모델이 계획 수립 과정에서 필요한 동적 사고 능력을 평가할 수 있는 기회를 제공합니다.

- **Technical Details**: 기존 ACPBench 데이터셋은 행동과 변화에 대한 간단한 논리적 질문으로 구성되어 있지만, ACPBench Hard는 7개의 각기 다른 작업에 대해 생성적 질문을 제시합니다. 각 질문의 답변 정확성을 평가하기 위한 평가기(evaluator)도 도입되어, 모델의 성능을 보다 정교하게 측정할 수 있게 되었습니다. 이 작업들은 Planning Domain Definition Language (PDDL) 도메인을 기반으로 구현되어 있습니다.

- **Performance Highlights**: 실험 결과, 모든 테스트된 언어 모델은 65% 이하의 정확도를 보였으며, 특히 'atom reachability'와 같은 작업에서는 매우 낮은 정확성을 나타냈습니다. 가장 큰 모델조차도 계획을 신뢰성 있게 수행하기에는 부족한 성능을 보여주고 있어, 이러한 모델들이 여전히 발전할 여지가 큽니다. 전체 데이터셋의 성과를 종합하면, 현재의 언어 모델이 계획 수립 작업을 효과적으로 수행하기 위해서는 더 많은 연구와 발전이 필요됨을 알 수 있습니다.



### Contextual Preference Collaborative Measure Framework Based on Belief System (https://arxiv.org/abs/2503.24328)
Comments:
          in Chinese language

- **What's New**: 이 논문은 인간의 개입을 줄이기 위한 새로운 preference collaborative measure framework를 제안합니다. 이 프레임워크는 업데이트된 belief system을 바탕으로 하며, preference 측정의 정확도와 효율성을 향상시키는 데 기여합니다. 또한, 사용자의 공통된 선호를 발견하기 위한 방법론이 포함되어 있습니다.

- **Technical Details**: 논문에서는 규칙 간의 거리와 평균 내부 거리(average internal distance)를 정의하여 사용자 간의 공통된 선호(common preference)를 발견하는 방법을 제시합니다. PRA 알고리즘은 정보 손실을 최소화하면서 선호 규칙을 찾기 위한 기법으로, belief system에 따라 규칙의 확인과 선호 규칙의 분류를 위한 신뢰도(belief degree)와 편차(deviation degree)를 도입합니다. 이를 통해 최종적으로 Top-K 흥미로운 규칙을 필터링하는 시스템을 구현하였습니다.

- **Performance Highlights**: 제안된 IMCos 및 IMCov 알고리즘은 가중 코사인 유사도(weighted cosine similarity)와 상관 계수(correlation coefficients)를 사용하여 프레임워크의 정확도와 효율성을 검증합니다. 실험 결과 이 두 알고리즘은 기존 우수한 알고리즘들에 비해 대부분의 측면에서 뛰어난 성능을 발휘하였습니다. 이러한 결과는 이 새로운 프레임워크가 효과적임을 입증하는 중요한 사례로 작용합니다.



### PAARS: Persona Aligned Agentic Retail Shoppers (https://arxiv.org/abs/2503.24228)
- **What's New**: 이번 연구에서는 LLM(대형 언어 모델) 기반의 에이전트 프레임워크 PAARS를 제안하여 인공지능 쇼핑 에이전트의 행동을 인간 소비자와 일치시키고자 했습니다. 이 프레임워크는 과거 쇼핑 데이터를 기반으로 만들어진 페르소나(persona)를 이용하여 시뮬레이션한 쇼핑 세션을 생성합니다. 이 접근 방식은 개인의 행동하고 유사한 집단 레벨의 행동을 연구하여 사용자 행동에 대한 훨씬 더 신뢰할 수 있는 대안을 제공합니다.

- **Technical Details**: PAARS는 두 가지 주요 단계로 이루어진 페르소나 마이닝(persona mining) 방법론을 활용합니다. 첫 번째 단계에서는 고객의 쇼핑 이력을 기반으로 소비자 프로필을 생성하고, 두 번째 단계에서는 이 프로필을 통해 개인의 쇼핑 선호도를 추론합니다. 생성된 페르소나는 대형 언어 모델이 다양한 작업을 수행하는 데 필요한 필수 정보로 구성됩니다.

- **Performance Highlights**: 실험 결과, 페르소나를 사용한 쇼핑 에이전트는 기존 작업에 비해 높은 정렬 성능을 보였지만 여전히 인간 행동과의 격차가 존재함을 입증했습니다. 또한, PAARS의 초기 적용을 통한 자동화된 A/B 테스트의 가능성을 제시하며, 향후 에이전트 기반의 A/B 테스트 및 조사에서 유의미한 응용을 기대할 수 있습니다.



### All You Need is Sally-Anne: ToM in AI Strongly Supported After Surpassing Tests for 3-Year-Olds (https://arxiv.org/abs/2503.24215)
- **What's New**: 이 논문은 Theory of Mind (ToM) 개념을 AI 시스템에 적용하는 새로운 모델을 제시합니다. 특히, 기본적인 ToM 테스트를 초월하는 능력을 보여주며 AI에서 인간의 인지 방식과 유사한 결과를 얻었다고 주장합니다. 이러한 접근 방법은 Gradient Evaluation을 포함한 Recursive Inference를 통해 ToM을 획득할 수 있음을 보여줍니다.

- **Technical Details**: 제안된 모델은 Gradient-Based Inference (GBI)를 사용하여 믿음의 표현을 조정하며, Bayesian Inference를 통해 관찰된 데이터를 기반으로 믿음을 업데이트합니다. 또한, 모델은 Stochastic Gradient Descent (SGD)와 Adaptive Learning Rates를 통합하여 비정상적인 환경에서도 교육될 수 있는 능력을 강화합니다. 계층적 믿음을 표현하는 Recursive Inference 구조를 통해 더욱 정교한 추론이 가능해집니다.

- **Performance Highlights**: 실험 결과, 이 모델은 Sally-Anne 테스트와 Smarties 테스트에서 3세 아동의 평균 정확도와 유사한 성과를 달성했습니다. 두 가지 테스트 모두에서 모델은 falsel-belief를 올바르게 예측하며, 초기 인간 사회 인지 능력을 성공적으로 복제할 수 있음을 입증했습니다. 이러한 결과는 Gradient-Based Recursive Reasoning 방법이 ToM 개념을 AI 시스템에서 실현할 수 있는 가능성을 제시합니다.



### Agent-Based Simulations of Online Political Discussions: A Case Study on Elections in Germany (https://arxiv.org/abs/2503.24199)
Comments:
          15 pages, 3, ESWC, Workshop Paper

- **What's New**: 이번 연구에서는 소셜 미디어 플랫폼에서 사용자 참여에 영향을 미치는 요인인 역사적 맥락, 시간 제약 및 보상 기반 상호작용을 모형화한 에이전트 기반 시뮬레이션 접근 방식을 제안합니다. 특히, 정치적 담론에 관한 독일 트위터 데이터를 활용하여 AI 모델을 미세 조정하고 있습니다.

- **Technical Details**: 감성 분석(sentiment analysis), 아이러니 탐지(irony detection), 공격성 분류(offensiveness classification) 등을 위한 AI 모델을 생성하며, 이는 과거의 대화 기록과 동기, 자원 제약을 고려한 사용자 상호작용을 모델링합니다. 시뮬레이션은 기대 보상(expected rewards)에 따라 의사 결정을 하는 근시적 최적 반응(myopic best-response) 모델을 활용하여 에이전트의 행동을 규제합니다.

- **Performance Highlights**: 결과적으로 역사적 맥락이 AI가 생성한 반응에 미치는 영향을 강조하며, 다양한 제약 조건에서 사용자 참여가 어떻게 진화하는지를 보여줍니다. 이러한 시뮬레이션은 사용자 상호작용의 복잡성을 이해하는 데 도움을 줄 수 있습니다.



### Grounding Agent Reasoning in Image Schemas: A Neurosymbolic Approach to Embodied Cognition (https://arxiv.org/abs/2503.24110)
- **What's New**: 이번 연구에서는 embodiment cognition 이론과 agent 시스템 간의 간극을 메우기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 sensorimotor 경험의 반복적인 패턴인 이미지 스키마(image schemas)를 포괄하는 공식적 특성을 활용합니다. LLM을 맞춤화하여 자연어 설명을 이러한 패턴 기반의 공식 표현으로 변환함으로써, 기본 개념 구조에서 에이전트의 이해를 뒷받침하는 neurosymbolic 시스템을 생성할 수 있는 가능성을 열게 됩니다.

- **Technical Details**: 연구는 신경망(neural networks)과 상징적 언어(symbolic languages)의 통합된 접근 방식을 제공합니다. 이미지 스키마는 Mandler와 Cánovas의 연구에 따라 개념적 원소로 분해될 수 있으며, 이는 구체적 물리적 상황을 다루는 데 적합한 요구 사항들을 충족해야 합니다. 이 연구는 이미지 스키마를 표현하기 위한 formalism의 주요 속성들에 대해 논의하며, 이미지 스키마가 구조화하는 무수한 물리적 구성을 표현할 수 있는 기초를 다지고 있습니다.

- **Performance Highlights**: 제안된 접근법은 인간과 에이전트 간의 직관적이고 설명 가능한 상호작용을 가능하게 하면서 reasoning 및 자연어 이해의 향상을 보여줄 것으로 예상됩니다. 본 연구는 이미지 스키마의 개념을 다루는 기존의 작업과 비교하여, 개념 구조의 완전한 공식적 특성과 기존의 상징적 해결책을 사용하는 강점을 강조합니다. 결론적으로, 이러한 새로운 프레임워크는 현대 AI 시스템의 다음 단계로 나아가는 데 필요한 기초를 제공할 것으로 보입니다.



### Towards Scientific Intelligence: A Survey of LLM-based Scientific Agents (https://arxiv.org/abs/2503.24047)
Comments:
          34 pages, 10 figures

- **What's New**: 이번 연구는 최근 복잡해진 과학 연구를 지원하기 위해 대규모 언어 모델(LLM)에 기반한 과학적 에이전트의 필요성을 강조합니다. 이 에이전트는 가설 생성, 실험 설계, 데이터 분석 등을 자동화하며, 기존의 일반적인 LLM과 다르게 도메인-specific 지식과 고급 도구를 통합합니다. LLM 기반 과학 에이전트는 복잡한 데이터 유형을 처리할 수 있는 능력을 갖추어 과학 연구의 혁신을 촉진합니다. 아울러, 이 연구는 이러한 에이전트의 아키텍처, 설계 및 윤리적 고려 사항을 포괄적으로 검토합니다.

- **Technical Details**: LLM 기반 과학 에이전트의 아키텍처는 Planner, Memory, Tool Set의 세 가지 핵심 구성요소로 이루어집니다. Planner는 사용자가 제출한 과학적 작업을 세부 작업으로 나누고, Memory에서 필요한 지식을 검색하여 Tool Set을 통해 액션을 실행합니다. 이 과정은 반복적이며, Planner는 중간 결과를 검토하고 Memory를 업데이트하여 향후 결정을 개선하는 구조로 되어 있습니다. 이러한 설계는 과학적 방법론에 기초한 계층적 계획을 강화하여 신뢰성 있는 결과를 도출할 수 있도록 지원합니다.

- **Performance Highlights**: 과학적 에이전트는 일반적인 LLM 에이전트와 다르게 복잡한 과학 데이터를 처리하고, 정확한 실험적 제어를 가능하게 하며, 엄격한 검증 및 오류 검사를 수행합니다. 이들은 연구 속도를 높이고, 실험의 재현성을 보장하며, 새로운 가설 생성을 가능하게 합니다. 연구 결과, 이러한 에이전트는 다수의 과학 분야에서 복잡한 문제 해결을 지원하며, 효율적이고 윤리적인 과학적 발견을 위한 дорожная карта와 같은 역할을 수행할 것입니다.



### Pay More Attention to the Robustness of Prompt for Instruction Data Mining (https://arxiv.org/abs/2503.24028)
- **What's New**: 이 논문에서는 고품질 온라인 지침 데이터 마이닝을 위한 혁신적인 프레임워크를 제안하며, 프롬프트의 강건성(prompt robustness)이 데이터 마이닝 과정에 미치는 영향을 집중적으로 조사합니다. 특히, 적대적인 지침 데이터의 생성을 통해 LLMs의 적절한 응답 생성을 지원하는 방법이 소개됩니다. 이 연구는 Adversarial Instruction-Following Difficulty(AIFD) 메트릭을 도입하여 적대적인 지침 데이터가 응답 생성에 얼마나 기여하는지를 측정하는 방안을 제공합니다.

- **Technical Details**: 고품질 지침 데이터 마이닝을 위한 새로운 접근 방식으로, 사용자 프롬프트에 대해 문자 단위, 단어 단위 및 문장 단위의 공격을 실시하여 적대적인 지침 샘플을 생성합니다. 또한, AIFD 점수를 통해 각 지침 데이터의 품질을 평가하고, 이를 기반으로 금형 데이터(diamond data)를 선택합니다. Adversarial Instruction Output Embedding Consistency(AIOEC) 메트릭을 통해 적대적인 프롬프트와 사용자 입력 프롬프트 사이의 출력 임베딩 유사성을 측정하여 데이터 마이닝의 효율성을 높입니다.

- **Performance Highlights**: 두 개의 벤치마크 데이터셋을 기반으로 extensive 실험을 진행하여 제안된 방법들의 효과성을 입증하였습니다. 실험 결과는 제안한 두 가지 방법의 유효성을 강조하는 동시에, 프롬프트의 강건성을 고려하는 것이 실용적으로 매우 중요하다는 점을 강조합니다. 이 연구는 LLM을 위한 지침 데이터 품질 개선에 실질적인 기여를 하고 있습니다.



### AI2Agent: An End-to-End Framework for Deploying AI Projects as Autonomous Agents (https://arxiv.org/abs/2503.23948)
- **What's New**: AI2Agent는 AI 프로젝트 배포를 자동화하는 엔드 투 엔드 프레임워크입니다. 이는 복잡한 환경 설정 및 디버깅 문제를 해결하여 AI 기술의 통합을 촉진합니다. AI2Agent는 과거 사례에서 학습하여 동적으로 배포 전략을 조정하고, 사람의 개입을 최소화하여 더욱 효율적인 배포를 가능하게 합니다.

- **Technical Details**: AI2Agent는 세 가지 주요 구성 요소, 즉 가이드라인 기반 실행(Guideline-driven Execution), 자기 적응형 디버그(Self-adaptive Debug), 사례 및 솔루션 누적(Case & Solution Accumulation)으로 이루어져 있습니다. 가이드라인 기반 실행은 규칙을 따르도록 설계되어 일관성을 유지하며, 자기 적응형 디버그는 실시간 피드백을 바탕으로 실행 전략을 조정합니다. 이 프레임워크는 또한 지식 저장소(Knowledge Repository)를 활용하여 과거의 경험을 지속적으로 축적합니다.

- **Performance Highlights**: AI2Agent의 성능은 30개의 AI 배포 사례에서 평가되었으며, TTS, 텍스트-이미지 생성, 이미지 편집 등이 포함되었습니다. 실험 결과, AI2Agent는 배포 시간을 크게 단축하고 성공률을 높이며 오류를 줄이는 데 성공하였습니다. 이러한 결과는 AI2Agent가 보다 표준화되고 모듈화된 AI 배포를 가능하게 하는 잠재력을 보여줍니다.



### What the F*ck Is Artificial General Intelligence? (https://arxiv.org/abs/2503.23923)
Comments:
          Preprint; 10 pages;

- **What's New**: 이번 연구에서는 인공지능의 범위에서 인식되고 있는 일반 인공지능(AGI)의 정의에 대해 논의하며, AGI가 여전히 의미를 갖는지에 대한 질문을 제기합니다. 저자는 AGI를 '인간 과학자'로서의 인공지능으로 정의하고, 이를 통해 AGI가 특정 목표를 달성하는데 있어 어떤 기능을 해야 하는지를 설명합니다.

- **Technical Details**: AGI를 구현하기 위해서는 다양한 요소들이 필요하며, 특히 탐색(Searching)과 근사(Approximation)가 필수적이라고 강조합니다. AGI의 발전은 하드웨어의 발전에 의해 가능해졌으며, 이제는 샘플(sample) 및 에너지 효율성에서 병목 현상이 발생하고 있음을 알립니다. 저자는 AGI의 다양한 아키텍처로는 AlphaGo, AERA, NARS, Hyperon 등이 있음을 언급합니다.

- **Performance Highlights**: 저자는 AGI 개발에 있어 스케일 극대화(scale-maxing), 단순성 극대화(simp-maxing), 기능 제약 최적화(w-maxing) 등의 메타 접근 방식을 사용하는 것을 제안합니다. 결과적으로 AGI는 다양한 도구 및 메타 접근 방식의 융합으로 이루어질 것이라고 결론지으며, 특히 scale-maxed approximation이 AGI 연구에서 지배적인 요소가 될 것이라고 예측합니다.



### DebFlow: Automating Agent Creation via Agent Deba (https://arxiv.org/abs/2503.23781)
- **What's New**: DebFlow는 워크플로우 생성을 최적화하기 위해 협력적 논쟁 메커니즘을 활용하는 새로운 프레임워크입니다. 기존의 방법들과 달리, DebFlow는 반성(reflection) 학습을 통합하여 이전 경험을 바탕으로 성능을 향상시킵니다. 이 프레임워크는 여러 벤치마크 데이터셋에서 3% 평균 성능 향상을 기록하였으며, 자원 소비 또한 -37%로 줄였습니다. 특히, 논쟁(Debate) 요소가 성능에 중요한 역할을 한다는 점이 강조됩니다.

- **Technical Details**: DebFlow는 노드와 엣지로 연결된 LLM 호출 노드의 집합인 연산자 노드를 사용하여 검색 공간을 정의합니다. 이 연산자는 Ensemble, Review & Revise와 같은 일반적인 에이전트 작업을 나타내는 재사용 가능 조합입니다. DebFlow는 논쟁과 반성을 통해 LLM 에이전트를 최적화하고, 구조화된 다중 에이전트 논의를 통해 작업 명세와 과거 성능 로그를 분석하여 최적의 연산자 구성을 탐색합니다. 이를 통해 효율적인 워크플로우를 생성할 수 있습니다.

- **Performance Highlights**: DebFlow는 다양한 문제 영역에서 기존의 인간 설계 모델보다 높은 성능을 보였습니다. 두 개의 벤치마크 데이터셋에서 논쟁 요소를 제거했을 때 4%의 성능 하락이 관찰되었으며, 이는 반성 요소가 제거될 때의 2% 하락보다 더 큰 수치입니다. 이는 논쟁 메커니즘이 프레임워크 성능 향상에 중요한 역할을 함을 강력하게 입증합니다.



### MolGround: A Benchmark for Molecular Grounding (https://arxiv.org/abs/2503.23668)
- **What's New**: 현재 분자 이해(molecular understanding) 접근 방식은 주로 인간 인식의 기술적인 측면(descriptive aspect)에 초점을 맞추고 있으며, 특정 구조적 구성 요소(structural components)와 분자 개념을 연결하는 참조적 측면(referential aspect)은 여전히 탐색되지 않았습니다. 이를 해결하기 위해 본 연구는 모델의 참조 능력을 평가하기 위한 분자 기반 벤치마크(molecular grounding benchmark)를 제안합니다. 연구팀은 NLP, 화학정보학(cheminformatics), 분자 과학 분야의 기존 규범과 분자 기반 접근 방식을 일치시키고, 현재까지 가장 큰 분자 이해 벤치마크를 구축하여 79,000개의 QA 쌍을 포함하고 있습니다.

- **Technical Details**: 본 벤치마크는 분자 개념과 특정 구조적 구성 요소 간의 연관성을 평가하기 위해 설계되었습니다. 연구에서는 Chemical Name Entity Recognition (CNER), Name-Structure Mapping (NSM), Referential Substructure Localization (RSL) 등을 포함한 다양한 분자 기반 작업을 정의하였습니다. 이 시스템은 기존의 모델들, 특히 GPT-4o보다 성능이 우수하며, 전통적인 작업인 분자 캡셔닝(molecular captioning)과 ATC(classification) 정밀도를 향상시키기 위해 통합되었습니다.

- **Performance Highlights**: 본 연구는 NLP 기술이 과학을 위한 AI(AI for Science) 분야의 분자 이해를 향상시키는 데 중대한 역할을 할 수 있음을 강조합니다. 제안된 다중 에이전트 기반 프로토타입은 공개된 모델에 비해 뛰어난 성능을 보여주며, 기존의 분자 이해 작업의 정확도를 개선하는 데 기여하였습니다. 이러한 성과는 분자 이해의 새로운 패러다임을 제시하며, 향후 더 세분화된 모델 개발을 위한 기초를 제공합니다.



### GIScience in the Era of Artificial Intelligence: A Research Agenda Towards Autonomous GIS (https://arxiv.org/abs/2503.23633)
- **What's New**: 이 논문은 Generative AI와 대형 언어 모델(LLMs)의 발전을 통해 지리 정보 시스템(GIS)의 새로운 자율적인 운영 방식을 제안합니다. 자율 GIS(Autonomous GIS)의 개념을 발전시키고, 이를 통해 정보 생성 및 유통 과정에서 혁신을 도모하고자 합니다. 논문은 자율 GIS의 다섯 가지 목표 및 수준, 핵심 기능, 운영 스케일을 정의하는 프레임워크를 제시합니다.

- **Technical Details**: 자율 GIS는 LLM을 결정적인 핵심으로 활용하여 공간 분석을 수행하기 위한 지오프로세싱(geoprocessing) 워크플로우를 독립적으로 생성하고 실행할 수 있습니다. 이 논문에서는 자율 GIS가 기지 데이터 검색(geospatial data retrieval), 공간 분석(spatial analysis), 맵 제작(map making)을 수행할 수 있는 네 가지 프로프 오브 컨셉 GIS 에이전트를 통해 이를 입증합니다. 자율 GIS는 비즈니스 및 환경 문제를 해결하기 위해 독립적으로 사고하고 혁신할 수 있는 시스템으로 발전할 것이라고 진단합니다.

- **Performance Highlights**: 논문의 결론에서는 자율 GIS의 구현에서 발생할 수 있는 주요 도전 과제와 미래 연구 방향을 제시합니다. 여기에는 미세 조정(fine-tuning) 및 자체 성장(self-growing) 결정 핵심의 필요성, 자율 모델링(autonomous modeling), 그리고 자율 GIS의 윤리적 및 실용적 함의에 대한 고찰이 포함됩니다. 이를 통해 GIScience의 패러다임 변화의 기초를 구축하고, GIS가 글로벌 문제 해결에 기여할 수 있는 미래를 제시합니다.



### Intrinsically-Motivated Humans and Agents in Open-World Exploration (https://arxiv.org/abs/2503.23631)
- **What's New**: 이 연구는 성인, 아동, 인공지능(Artificial Intelligence) 에이전트를 복잡한 개방형 환경인 Crafter에서 비교하여 인간의 탐험에 관한 내재적 동기의 특성을 분석합니다. 특히 Entropy(엔트로피)와 Empowerment(권한 부여)의 목표가 인간 탐험 진전과 긍정적으로 상관관계가 있음을 발견했습니다. 또한, 초기 탐험에서는 상태 다양성이 더 많은 신호를 제공하고, 고급 탐험에서는 제어가 더 효과적인 목표일 수 있다는 점을 제시합니다. 최종적으로, 아동의 사적 언어 발화가 탐험에 긍정적인 영향을 미칠 수 있다는 초기 증거를 찾았습니다.

- **Technical Details**: 초기 분석을 통해 Crafter 환경에 대한 탐험 점수를 도입하고, 성인이 RL 에이전트보다 더 효율적으로 탐험하는 것을 발견했습니다. RL 에이전트는 내재적 보상만으로는 게임 구조에서 설계된 외재적 보상에 비해 성과가 떨어지는 경향이 있습니다. 연구에서는 정보 이론에서 일반적으로 사용되는 내재적 목표인 Entropy, Information Gain(정보 이득), Empowerment 간의 탐험 행동 최적화를 분석했습니다. Entropy와 Empowerment는 성인 탐험 점수와 강한 양의 상관관계를 보였으나, Information Gain과는 상관관계가 없었습니다.

- **Performance Highlights**: 연구 결과, 성인 및 아동의 탐험 행동에서 Entropy와 Empowerment의 목표가 탐험 성과와 일관되게 긍정적인 상관관계를 나타냈습니다. 성인은 모든 분야에서 아동보다 높은 점수를 기록했으며, 내재적 보상을 가진 에이전트는 외재적 보상을 기반으로 훈련된 에이전트에 비해 효율성이 낮았습니다. 탐험 점수를 측정하는 다양한 점수를 설계하여 성과를 분석했으며, 아동의 목표 기반 발화가 탐험 점수와의 강한 긍정적 상관관계를 나타냈습니다.



### Beyond Detection: Designing AI-Resilient Assessments with Automated Feedback Tool to Foster Critical Thinking (https://arxiv.org/abs/2503.23622)
- **What's New**: 본 연구는 ChatGPT와 같은 생성형 AI 도구가 학생의 비판적 사고(critical thinking) 및 창의성에 미치는 영향을 조사합니다. 기존 AI 감지 기술의 한계를 극복하기 위해, Bloom의 분류법(Bloom's Taxonomy)과 최신 자연어 처리 기법을 통합하여 AI의 응답 가능성을 예측하는 웹 기반 Python 도구를 제안합니다. 이 접근 방식은 고등 교육의 창의적 참여와 독창성을 유지하고자 합니다.

- **Technical Details**: 研究中，採用了混合方法（mixed-method approach）來評估生成型人工智慧在高等教育中的應用。該工具將Bloom的分類法與GPT-3.5 Turbo和BERT基於語義的相似性評估結合，用以分析任務的認知複雜性。它通過自動化認知分析來評估作業的AI解決能力，提供教育者設計反抗AI生成內容的作業的支持。

- **Performance Highlights**: 본 연구는 도구의 효과성을 강조하며, AI가 생성한 답변에 취약한 평가 과제를 자동으로 분석하여 교육자에게 유용한 인사이트를 제공합니다. 연구 결과, AI 응답 가능성을 최소화하기 위한 인지적 분석에 초점을 맞춘 과제 리디자인이 학문적 무결성을 향상시키는 데 기여하는 것으로 나타났습니다. 이 접근 방식은 AI의 긍정적 활용 측면을 전제로 합니다.



### An Organizationally-Oriented Approach to Enhancing Explainability and Control in Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2503.23615)
- **What's New**: 이 논문에서는 Multi-Agent Reinforcement Learning(MARL) 과정에 조직적 역할과 목표를 명시적으로 통합하는 새로운 프레임워크를 소개합니다. 이 프레임워크는 에이전트들이 조직적 제약을 충족하도록 안내하여 에이전트 행동의 설명 가능성과 제어를 향상시키고자 합니다. 일반적으로 MARL의 문헌은 개별 에이전트에 초점을 맞추고 있지만, 이 연구는 조직 수준에서의 협업 행동을 탐구하고 있습니다.

- **Technical Details**: MOISE+MARL 프레임워크는 Decentralized Partially Observable Markov Decision Process(Dec-POMDP) MARL 프레임워크와 조직적 모델인 ℳOISE^+를 통합합니다. 사용자는 역할이나 목표의 논리를 수동으로 정의할 수 있으며, 이로 인해 에이전트의 행동을 설명하는 기대 행동을 규정할 수 있습니다. 또한 Trajectory-based Evaluation in MOISE+MARL(TEMM) 방법을 통해 관측된 경로에서 암시적 역할과 목표를 일반화할 수 있습니다.

- **Performance Highlights**: 모든 환경에서 역할이 부여된 에이전트들은 TEMM의 정량적 측정에 따라 예상대로 행동하였습니다. TEMM에 의해 추론된 역할과 임무는 사전 정의된 사양과 밀접하게 일치하였으며, MOISE+MARL의 내부 일관성을 보여줍니다. 정책 기반 및 액터-크리틱 알고리즘은 에이전트가 안정적인 정책을 유지하도록 안내하는 데 특히 적합한 것으로 나타났으며, 이것은 TEMM의 안정적인 암시적 조직 생성을 가능하게 합니다.



### GenVP: Generating Visual Puzzles with Contrastive Hierarchical VAEs (https://arxiv.org/abs/2503.23598)
Comments:
          Accepted to ICLR 2025

- **What's New**: 이 논문에서는 Generative Visual Puzzles (GenVP)라는 새로운 프레임워크를 제안하여 Raven의 진행 매트릭스 (RPMs) 전체 생성 과정을 모델링합니다. 이 접근 방식은 기존의 RPM 문제 해결 방식과는 달리, 추상적인 규칙을 기반으로 완전한 새로운 퍼즐을 생성할 수 있는 능력을 갖추고 있습니다. GenVP는 기계가 기존의 퍼즐을 넘어서 일반화 할 수 있는 능력을 제공하여, AI의 창의성을 높이는 데 기여합니다.

- **Technical Details**: GenVP는 계층적 추론(hierarchical inference) 및 생성 파이프라인(generative pipeline)을 포함하는 그래픽 모델을 사용합니다. 이를 통해 Mixture of Experts (MoE) 기제를 도입하여 퍼즐 규칙 예측의 정확성을 개선하고, 불필요한 특성(노이즈)에 강건한 반응을 보입니다. 또한, novel contrastive learning scheme을 통해 교차 퍼즐 및 후보의 비교를 가능하게 하여 학습의 강건성을 높입니다.

- **Performance Highlights**: 다양한 실험에서 GenVP는 퍼즐 문제 해결 정확도뿐만 아니라 22가지 OOD(Out-Of-Distribution) 시나리오에서 SOTA 성능을 달성하였습니다. GenVP는 더욱 복잡한 해결 가능한 공간에서 효율적으로 일반화할 수 있으며, 기존의 방법들과 비교하여 문제 풀이 성능에서 확연히 우수한 결과를 보입니다. 조사 결과, GenVP는 추상 규칙을 효과적으로 이해하고 새로운 RPM을 생성하는 높은 수준의 능력을 보유하고 있습니다.



### Benchmarking Systematic Relational Reasoning with Large Language and Reasoning Models (https://arxiv.org/abs/2503.23487)
Comments:
          Submitted to ACL 2025

- **What's New**: 이 논문은 대규모 언어 모델(LLM)과 대규모 추론 모델(LRM)을 사용한 체계적 추론(systematic reasoning)의 중요성을 강조합니다. 모델의 성능은 종종 규칙적인 추론보다는 지름길에 의존하는 경향이 있으며, 이는 분포 외(out-of-distribution) 예제에서 성능 저하로 이어집니다. 저자들은 공간적 및 시간적 추론에 대한 문제를 통해 이러한 모델들이 어떻게 일반화하는지를 탐구하며, 체계적 일반화(Systematic Generalization, SG) 메트릭을 기반으로 LLM과 LRM의 추론 능력을 평가하는 것이 중요하다고 주장합니다.

- **Technical Details**: 논문에서는 공간 시간 추론(Spatial Temporal Reasoning, STaR) 벤치마크를 활용하여 모델의 성능을 분석합니다. STaR는 복합적 구조를 가지며, 이를 통해 전례 없는 문제 사례를 생성할 수 있어 데이터 세트 오염 문제를 피할 수 있습니다. 이러한 문제는 계산적으로 해결 가능하며, LRM이 접근할 수 있는 문제로 설계되었습니다.

- **Performance Highlights**: 많은 유명한 LLM과 LRM이 STaR에서 어려움을 겪지만, 무작위 기회보다 나은 성과를 보입니다. 모델 규모, 파인튜닝(fine-tuning) 및 체인 오브 띵크(CoT) 테스트 시간이 성능에 미치는 영향을 파악하며, 논문에서 다루는 문제의 복잡도와 모델의 일반화 능력 간의 관계를 평가합니다.



### A Systematic Decade Review of Trip Route Planning with Travel Time Estimation based on User Preferences and Behavior (https://arxiv.org/abs/2503.23486)
Comments:
          6 pages, 2 figures, 1 table

- **What's New**: 이 논문에서는 인공지능(AI)을 통한 적응형 여행 경로 계획(trip route planning) 및 여행 시간 추정(travel time estimation, TTE)의 발전을 체계적으로 탐색합니다. 도시 교통 시스템의 복잡성이 증가함에 따라 전통적인 내비게이션 방법은 동적 사용자 선호도, 실시간 교통 상황, 그리고 확장성 요구사항을 충족하는 데 어려움을 겪고 있습니다. 본 연구는 머신러닝(ML), 강화 학습(RL), 그래프 신경망(GNN) 등 기존 AI 기법과 메타 학습, 설명 가능한 AI(XAI), 생성 AI, 연합 학습(Federated Learning)과 같은 새로운 방법론을 탐구합니다.

- **Technical Details**: 현재의 내비게이션 시스템은 역사적 교통 데이터와 GPS 신호, 머신러닝 알고리즘을 기반으로 하여 최적의 경로를 제안하지만, 개인적인 선호도나 실시간 상황을 빠르게 반영하는 데 한계가 있습니다. 예를 들어, 요금 도로 회피, 풍경이 좋은 경로 선호, 환경 친화적인 옵션 선택 등 개별 사용자 선호를 반영하는 데 있어서 현재 시스템은 정적 사용자 데이터를 기반으로 하여 실시간 적응력이 부족합니다. 또한 윤리적 우려와 개인 정보 보호 문제는 향후 내비게이션 시스템의 신뢰성을 저해하는 중요한 요소로, 연합 학습과 같은 개인정보 보호 기술을 포함한 윤리적 AI 실천이 필요합니다.

- **Performance Highlights**: 많은 연구가 현재 내비게이션 시스템의 문제를 해결하기 위한 방안을 제시하고 있지만, 실시간 적응성과 개인화에 대한 해결책은 여전히 부족합니다. 특히, 사용자 행동에 따라 통합되고 지속적으로 학습하는 시스템 구현은 중요한 과제로 남아 있습니다. 또한, 동적 및 맥락 데이터의 실시간 통합이 부족한 것은 정확한 TTE 달성에서 큰 도전 과제입니다. 결론적으로, 미래 내비게이션 시스템은 보다 개인화되고 적응 가능하며 확장 가능한 솔루션을 제공하여 도시 이동성을 더욱 향상할 필요가 있습니다.



### Large Language Models Are Better Logical Fallacy Reasoners with Counterargument, Explanation, and Goal-Aware Prompt Formulation (https://arxiv.org/abs/2503.23363)
Comments:
          Accepted to NAACL 2025 Findings

- **What's New**: 본 연구에서는 논리적 오류 탐지를 위한 새로운 프롬프트(formulation) 기법을 제안하며, 이는 감독 학습(supervised) 및 비감독 학습(unsupervised) 환경에 모두 적용이 가능하다. 이 방법은 입력 텍스트에 암묵적 맥락 정보(implicit contextual information)를 통합하여 오류의 유효성을 평가하는 쿼리를 생성하고, 이를 기반으로 결과를 분류한다. 또한, 다섯 개의 데이터 세트를 사용한 평가 결과 시간적 모델들에 비해 현저한 성능 향상을 확인했다.

- **Technical Details**: 제안된 접근법은 네 개의 주요 단계로 구성되며, 첫 단계에서는 LLM을 이용해 맥락 개선을 통해 앵커 쿼리(context-informed queries)를 생성한다. 이후 생성된 쿼리를 통해 논리적 오류를 분류하며, 마지막 단계에서는 각 쿼리에 대해 신뢰도 기반으로 순위를 매긴다. 특이하게도, 본 연구에서는 각 입력 텍스트를 증강하기 위해 세 가지 유형의 암묵적 정보(반론(counterargument), 설명(explanation), 목표(goal))를 활용한다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 제로샷(zero-shot) 환경에서 Macro-F1 점수가 최대 0.60, 파인튜닝(fine-tuned) 모델에서는 최대 0.45 향상된 성능을 보였다. 따라서 본 접근법이 최첨단 모델들보다 월등한 결과를 드러내었으며, 이는 프롬프트 순위 매기기 방법의 효과적 활용에 기인한 것으로 분석되었다. 단계별로 수행된 심층 분석을 통해 제안 방법의 장점과 개선점을 추가적으로 검토하였다.



### A Survey of WebAgents: Towards Next-Generation AI Agents for Web Automation with Large Foundation Models (https://arxiv.org/abs/2503.23350)
- **What's New**: 최근 웹 기술의 발전으로 AI Agents가 등장하여 반복적이고 시간이 많이 소요되는 웹 작업을 자동으로 처리하는 데 큰 가능성을 보이고 있습니다. 특히, Large Foundation Models(LFMs)이 도입되어 인간과 유사한 언어 이해 및 추론 능력을 보여주며 웹 작업에서의 효율성을 극대화하고 있습니다. 이러한 AI Agents는 WebAgents로 명명되어 사람들의 일상 작업을 자동화하고 있으며, 이는 전반적인 생활의 질 향상에 기여할 수 있습니다.

- **Technical Details**: AI Agents는 크게 두 가지 범주로 나눌 수 있습니다: 강화 학습(Reinforcement Learning, RL) 기반 에이전트와 LFM 기반 에이전트입니다. RL 기반 에이전트는 환경과의 상호작용을 통해 최적의 정책을 학습하며, Q-Learning과 Policy Gradient 같은 알고리즘을 활용합니다. DNN(Deep Neural Networks)의 발전 덕분에 딥 강화 학습(Deep Reinforcement Learning, DRL) 알고리즘이 제안되어 복잡한 데이터에서 특징을 모델링하고 추출하는 데 성공을 거두었습니다.

- **Performance Highlights**: 현재 LFMs는 다양한 복잡한 작업을 수행하는 데 있어 놀라운 성과를 보여주고 있으며, 의료 및 금융 분야에서 실질적으로 활용되고 있습니다. 또한, 최근에 등장한 AutoGPT 프레임워크는 사용자 지시 없이도 복잡한 작업을 독립적으로 처리할 수 있는 기능으로 큰 주목을 받고 있습니다. 이러한 발전들은 사용자의 편의성을 크게 향상시키고, 일상적인 웹 작업을 자동으로 처리함으로써 생산성과 효율성을 극대화하는 방향으로 나아가고 있습니다.



### A Scalable Framework for Evaluating Health Language Models (https://arxiv.org/abs/2503.23339)
- **What's New**: 최근의 연구에 따르면, 대형 언어 모델(LLMs)은 개인화된 건강 정보를 제공할 때 유용한 응답을 생성하는 데 있어 잠재력을 가지고 있습니다. 이에 따라 건강 응용 프로그램에서 LLM의 채택이 증가하고 있으며, 정확성, 개인화 및 안전성을 포함한 여러 차원에서 응답 품질을 보장하기 위한 엄격하고 효율적인 평가 방법론이 필수적입니다.

- **Technical Details**: 이 연구에서는 Adaptive Precise Boolean rubrics라는 평가 프레임워크를 소개합니다. 이 방법은 모델 응답에서의 격차를 식별하기 위해 최소한의 목표 기반 질문 세트를 사용하여 개방형 질문의 인간 및 자동 평가를 간소화합니다. 이 접근법은 복잡한 평가 목표와 보다 구체적이고 정량화된 목표를 대조하는 최근의 연구를 기반으로 합니다.

- **Performance Highlights**: Adaptive Precise Boolean rubrics는 전문가와 비전문가 평가자 간의 높은 평가자 간 일치를 도출하며, 자동 평가에서도 전통적인 Likert 척도에 비해 더 높은 일치를 기록했습니다. 또한, 이 방법은 Likert 기반 방식의 평가 시간의 약 절반만을 요구해 효율성을 크게 향상시킵니다. 이는 건강 분야에서 LLM에 대한 더 광범위하고 비용 효율적인 평가를 가능하게 합니다.



### A Multi-Agent Framework with Automated Decision Rule Optimization for Cross-Domain Misinformation Detection (https://arxiv.org/abs/2503.23329)
- **What's New**: 이 논문은 다중 도메인에서의 허위 정보 탐지를 위한 새로운 다중 에이전트 프레임워크, MARO(MultiAgent Framework for cross-domain misinformation detection with Automated Decision Rule Optimization)를 제안합니다. 기존의 방법들이 특정 도메인에 편향되어 있는 결정 규칙을 수동적으로 디자인하는 데 그치는 반면, MARO는 다수의 전문가 에이전트를 활용하여 타겟 도메인 뉴스를 다각도로 분석하고, 질문 반사 메커니즘으로 분석 품질을 향상합니다. 이를 통해 다양한 도메인에서 효과적으로 사용 가능한 결정 규칙을 자동으로 최적화합니다.

- **Technical Details**: MARO는 두 가지 주요 모듈로 구성됩니다: 1) 다차원 분석 모듈(Multi-Dimensional Analysis Module)과 2) 결정 규칙 최적화 모듈(Decision Rule Optimization Module)입니다. 이 모듈은 각각 다양한 관점에서 뉴스를 분석하여 포괄적인 분석 보고서를 생성하고, 다양한 도메인에서의 검증 작업을 통해 결정 규칙을 반복적으로 최적화합니다. 이를 위해, LLM 기반의 에이전트를 사용하여 뉴스 항목에 대한 전반적인 분석을 수행합니다.

- **Performance Highlights**: 실험 결과, MARO는 여러 개의 LLM을 사용한 기존 최첨단 방법들보다 유의미한 향상을 보여줍니다. 특히, 다차원 분석 모듈과 결정 규칙 최적화 모듈 모두 MARO의 성능을 효과적으로 개선하는 것으로 나타났습니다. MARO는 다양한 도메인에서 허위 정보를 탐지하기 위한 강력한 도구로 자리 잡을 것으로 기대됩니다.



### Exploring Explainable Multi-player MCTS-minimax Hybrids in Board Game Using Process Mining (https://arxiv.org/abs/2503.23326)
Comments:
          36 pages, AAAI 2025 PRL

- **What's New**: 이 논문은 몬테카를로 트리 탐색(Monte-Carlo Tree Search, MCTS) 에이전트의 의사결정 및 행동을 설명하는 새로운 접근 방식을 다룹니다. MCTS의 한계점을 극복하기 위해, 저자들은 다중 플레이어 MCTS의 롤아웃 단계에 얕은 미니맥스(minimax) 검색을 통합하고 프로세스 마이닝(process mining) 기법을 사용합니다. 이를 통해 3대 3 체커(3v3 checkers) 게임에서 에이전트의 전략을 설명하는 방법을 제시합니다.

- **Technical Details**: MCTS는 많은 가능한 미래를 시뮬레이션하고 평가하여 복잡한 탐색 트리를 생성하는 샘플링 기반의 검색 알고리즘입니다. 반면, 미니맥스는 모든 가능한 이동과 결과를 철저히 탐색하는 결정론적 검색 알고리즘입니다. 이 논문에서는 MCTS와 미니맥스를 결합한 하이브리드를 통해 전략적 힘과 전술적 힘을 모두 활용하는 방법을 소개합니다.

- **Performance Highlights**: 이 연구는 프로세스 마이닝 기법을 적용하여 MCTS-미니맥스 하이브리드의 의사결정을 설명하는 데 초점을 맞추고 있습니다. 이 방법은 에이전트의 행동을 이해하는 데 있어 세 가지 기본 질문에 대한 설명을 제공합니다. 논문에서는 이 시스템이 3대 3 체커 게임에서 에이전트가 수행한 결정 과정을 더 투명하게 만들 것으로 기대하고 있습니다.



### AI Agents in Engineering Design: A Multi-Agent Framework for Aesthetic and Aerodynamic Car Design (https://arxiv.org/abs/2503.23315)
- **What's New**: 본 연구에서는 '디자인 에이전트(Design Agents)'라는 개념을 도입하여 자동차 설계 프로세스를 혁신적으로 변화시키고 있습니다. 이 접근법은 기존 엔지니어링 워크플로우에 AI 기반의 디자인 에이전트를 통합하여 효율성을 높이고 디자인 사이클을 단축하는 특징이 있습니다. 특히, autonomous computational systems을 활용하여 전통적인 수작업 과정을 자동화하여 시간을 단축시키는 것이 주요 내용입니다.

- **Technical Details**: 디자인 에이전트는 저희의 메소드에서 사용되는 estado de la técnica 시각-언어 모델(Vision-Language Models, VLMs)과 대형 언어 모델(Large Language Models, LLMs) 및 기하학적 딥러닝 기술을 활용합니다. 이러한 기술들은 초기 스케치부터 완전한 시뮬레이션에 이르기까지의 프로세스를 간소화하고, 예측 모델을 통해 공기역학적 평가를 신속하게 수행할 수 있게 해줍니다. Python API 및 AutoGen을 통해 모든 과정이 자동화되어 유연한 디자인 최적화를 가능하게 합니다.

- **Performance Highlights**: 제안된 다중 에이전트 프레임워크는 디자인 주기가 급격히 단축되며, 기존의 몇 주가 걸리던 과정을 몇 분으로 줄일 수 있는 능력을 보여줍니다. 디자인 에이전트들은 CAD 모델링, 3D 형상 생성을 통한 공기역학 평가, 그리고 지속적인 실시간 예측을 가능하게 하여 디자이너와 엔지니어 간의 협업을 활성화합니다. 연구 결과는 전통적인 자동차 디자인과 AI 기술의 융합을 통해 향후 다양한 엔지니어링 분야에서의 혁신 가능성을 강조합니다.



### SPIO: Ensemble and Selective Strategies via LLM-Based Multi-Agent Planning in Automated Data Scienc (https://arxiv.org/abs/2503.23314)
Comments:
          Under Review

- **What's New**: 이번 논문에서는 SPIO(Sequential Plan Integration and Optimization)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 LLM(대형 언어 모델)을 이용한 의사결정 방식을 통해 샘플 전략을 생성하고 최적화합니다. 기존의 단일 경로 워크플로우와 달리 SPIO는 다단계 처리 프로세스를 적용하여 데이터 전처리, 특성 엔지니어링, 모델링, 하이퍼파라미터 조정까지 아우릅니다. 이를 통해 다양한 전략을 탐색할 수 있는 기회를 제공합니다.

- **Technical Details**: SPIO 프레임워크는 네 가지 주요 모듈인 데이터 전처리(data preprocessing), 특성 엔지니어링(feature engineering), 모델 선택(model selection), 하이퍼파라미터 조정(hyperparameter tuning)으로 구성됩니다. 각 모듈에서는 독립적으로 후보 전략을 생성하는 전용 계획 에이전트(planning agents)가 존재합니다. SPIO는 또한 두 가지 변형인 SPIO-S(단일 최적 계획 선택)와 SPIO-E(상위 k개 계획을 조합)로 나뉘어 각각의 활용도를 극대화할 수 있습니다.

- **Performance Highlights**: Kaggle과 OpenML 데이터 세트를 대상으로 한 광범위한 실험에서 SPIO는 최신 방법론보다 우수한 성능을 나타냈습니다. SPIO의 적응형 다경로(reasoning) 접근 방식은 다양한 통찰력을 통합할 수 있어 고정된 단일 경로 워크플로우의 한계를 효과적으로 극복합니다. 이로 인해 SPIO는 예측 정확도를 지속적으로 향상시키고, 다양한 데이터 시나리오에 적응하며, 실행 신뢰성을 높이는 데 있어 탁월한 성과를 보이고 있습니다.



### LaViC: Adapting Large Vision-Language Models to Visually-Aware Conversational Recommendation (https://arxiv.org/abs/2503.23312)
- **What's New**: 이 연구에서는 시각적 정보를 통합한 새로운 대화형 추천 시스템인 LaViC (Large Vision-Language Conversational Recommendation Framework)를 제안합니다. LaViC는 대화 중 사용자의 요구를 파악하고 더 개인화된 제안을 제공하기 위해 이미지 표현을 효율적으로 처리하는 방법을 도입합니다. 이를 통해 패션과 인테리어와 같은 시각적 요구가 중요한 분야에서 효과적인 추천을 가능하게 합니다.

- **Technical Details**: LaViC의 기법은 두 단계로 구성됩니다. 첫 번째 단계인 'visual knowledge self-distillation'은 수백 개의 토큰으로 이루어진 제품 이미지를 소수의 시각적 토큰으로 압축하여 계산 비용을 줄입니다. 두 번째 단계인 'recommendation prompt tuning'에서는 대화 맥락과 증류된 시각적 토큰을 결합해 텍스트와 시각적 특징을 함께 포착하는 통합 메커니즘을 제공합니다.

- **Performance Highlights**: 광범위한 실험을 통해 LaViC는 텍스트 기반 대화형 추천 방법 및 오픈 소스 비전-언어 모델보다 뛰어난 성능을 보였습니다. LaViC는 또한 GPT-3.5-turbo와 같은 유명한 상용 모델에 비해 경쟁력 있는 정확도를 기록하며, 비주얼 데이터의 중요성을 강조하고 효과적인 비전-언어 통합의 가능성을 보여주었습니다. 이 연구는 새로운 'Reddit-Amazon' 데이터세트를 통해 진정한 사용자 쿼리를 기반으로 상호작용할 수 있는 기회를 제공합니다.



### GRASP: Municipal Budget AI Chatbots for Enhancing Civic Engagemen (https://arxiv.org/abs/2503.23299)
- **What's New**: 이번 연구 논문에서는 GRASP라는 맞춤형 AI 챗봇 프레임워크를 제안합니다. GRASP는 사용자 예산 질문에 대해 더 진실하고 기반이 확립된 답변을 제공하며, 기존의 정보 검색 시스템보다 더 높은 정확도를 자랑합니다. 이 시스템은 Retrieval-Augmented Generation (RAG) 프레임워크와 agentic workflow를 결합하여 공공 예산 정보 제공의 품질을 높입니다.

- **Technical Details**: GRASP는 'Generation with Retrieval'과 'Action System'을 통합하여 많은 양의 예산 문서에서 정보 검색을 수행합니다. 사용자가 질문할 때, RAG 프레임워크는 최신 공식 예산 문서에서 관련 정보를 검색하여 LLM에게 제공합니다. 또한 ReAct 에이전트 시스템을 통해 사용자의 쿼리를 더 작은 하위작업으로 분해하여 정확하고 직관적인 응답을 생성할 수 있도록 지원합니다.

- **Performance Highlights**: GRASP 챗봇은 지방 정부 예산 쿼리에 대해 78%의 정확도로 응답했는데, 이는 GPT-4o와 Gemini의 정확도인 60% 및 35%에 비해 월등히 높은 수치입니다. 이러한 성과는 GRASP가 일반 대중이 자신의 도시 예산에 대한 직관적이고 올바른 이해를 얻는 데 필요한 시간과 노력을 크게 줄여준다는 점에서 의미가 큽니다.



### Ethereum Price Prediction Employing Large Language Models for Short-term and Few-shot Forecasting (https://arxiv.org/abs/2503.23190)
- **What's New**: 본 연구는 이더리움 가격 예측에 대한 대규모 언어 모델(LLM)의 효과성을 분석하고, 기계 학습 분야에서 기존 모델들보다 우수한 성능을 보일 수 있다는 내용을 담고 있습니다. 특히, LLM을 사전 훈련된 모델에서 이더리움 가격 데이터의 고유한 특징에 적응시키는 새로운 접근법을 소개하고, 다양한 메트릭에서 최첨단 성과를 달성했음을 강조합니다. 이 연구는 예측 정확성을 더욱 개선하기 위한 감정 분석 통합의 가능성을 제시하며, LLM의 유용성을 보여줍니다.

- **Technical Details**: 연구에서 다룬 대규모 언어 모델(LLM)은 Llama-3, Llama-2 및 GPT-2와 같은 사전 훈련된 모델들을 포함합니다. 이 모델들은 특정 레이어를 동결하고 나머지를 미세 조정함으로써 이더리움 가격 예측을 위한 시간 시리즈 데이터에 최적화됩니다. 연구자들은 기계 학습(ML) 및 심층 학습(DL) 기술을 활용해 전통적인 접근 방식에 비해 더 효과적인 예측을 달성하였고, 색인 필요성과 감정 기반 접근 방식을 넘어서는 연구 방향을 제시했습니다.

- **Performance Highlights**: LLMs를 이용한 연구 결과, Mean Squared Error (MSE), Mean Absolute Error (MAE) 및 Root Mean Squared Error (RMSE)와 같은 다양한 지표에서 기존 최첨단 모델들을 초과하는 성과를 나타냈습니다. 이더리움 가격 예측이라는 특정 도메인에서 LLM들이 경쟁력을 가지며, 기계 학습 분산 금융(DeFi) 분야에서의 기여를 강조합니다. 이 연구는 금융 분석가, 투자자 및 거래자들이 효율적으로 가격 예측 도구를 개발하는 데 기여할 것입니다.



### AstroAgents: A Multi-Agent AI for Hypothesis Generation from Mass Spectrometry Data (https://arxiv.org/abs/2503.23170)
- **What's New**: 본 논문에서는 태양계 전역에서의 샘플 리턴 미션과 질량 분석기(mass spectrometry) 데이터의 증가로 인해 생명체 출현에 관한 가설을 생성하기 위한 새로운 방법이 필요하다는 점을 강조합니다. AstroAgents라는 대규모 언어 모델 기반의 다중 에이전트 AI 시스템을 소개하며, 이는 질량 분석기 데이터를 분석하고 기존의 우주 생물학(astrobiology) 문헌과 연결하여 신뢰할 수 있는 가설을 생성하는 데 도움을 줍니다.

- **Technical Details**: AstroAgents는 데이터 분석가(data analyst), 계획자(planner), 세 명의 도메인 과학자(domain scientists), 집계기(accumulator), 문헌 리뷰어(literature reviewer), 비평가(critic) 등 8명의 협력 에이전트로 구성되어 있습니다. 이 시스템은 질량 분석기 데이터를 사용자 제공 연구 논문과 함께 처리하고, 데이터 분석가는 데이터를 해석하며, 계획자는 특정 작업을 과학자 에이전트에게 할당합니다.

- **Performance Highlights**: AstroAgents의 성능 평가를 위해, 우주 생물학 전문가가 8개의 운석(meteorites) 및 10개의 토양 샘플에서 얻은 데이터로부터 생성된 100개 이상의 가설의 참신성 및 신뢰성을 평가하였습니다. 이 중 36%의 가설이 신뢰할 수 있다고 판단되었으며, 그 중 66%는 신규로 판별되었습니다.



### Agentic Large Language Models, a survey (https://arxiv.org/abs/2503.23037)
- **What's New**: 최근 에이전틱 LLMs(agentic large language models)의 발전은 연구자들 사이에서 큰 관심을 받고 있습니다. 이 논문에서는 이러한 LLM들이 (1) 추론(reasoning), (2) 행동(action), (3) 상호작용(interaction)하는 능력을 갖춘다고 정의하며, 이에 대한 문헌을 체계적으로 정리하고 있습니다. 에이전틱 LLMs는 의학, 물류, 금융 등 다양한 분야에 활용되고 있으며, 자가 반성(self-reflection) 및 역할놀이(role-playing)는 새로운 연구의 가능성을 열어줍니다.

- **Technical Details**: 에이전틱 LLMs는 자연어 처리(natural language processing), 도구 통합(tool integration), 강화 학습(reinforcement learning) 등의 다양한 기술적 발전에 의존하고 있습니다. 논문은 세 가지 범주로 문헌을 나누어 에이전트가 어떻게 더 지능적으로 행동하고 상호작용할 수 있도록 발전해왔는지를 설명합니다. 이러한 기술적 발전은 에이전트들이 환경과 상호작용함으로써 새로운 훈련 데이터를 생성하고, 더 나아가 기존의 LLM 교육 방식을 보완하는 데 기여합니다.

- **Performance Highlights**: 이 논문은 에이전틱 LLM의 성능을 높이기 위한 연구 의제를 제시하며, LLM들이 의사 결정 및 협업 문제 해결에 어떻게 기여할 수 있는지를 강조합니다. 또한, LLM들이 자가 훈련(self-training) 수행을 통해 더 많은 훈련 데이터를 생성할 수 있는 기회를 제공함으로써 언어 모델이 계속해서 학습할 수 있는 방법리를 제시합니다. 그러나 LLM들이 현실 세계에서 행동할 경우 발생할 수 있는 위험 요소에 대해서도 경고하고 있습니다.



### FindTheFlaws: Annotated Errors for Detecting Flawed Reasoning and Scalable Oversight Research (https://arxiv.org/abs/2503.22989)
Comments:
          43 pages, 3 figures. for associated repository, see this https URL

- **What's New**: 이번 논문에서는 AI 모델의 감독이 점점 더 어려워지는 문제를 해결하기 위한 새로운 데이터셋을 소개합니다. 이 데이터셋은 의료, 수학, 과학, 코딩 및 Lojban 언어를 포함한 다섯 가지 다양한 분야에서 활용됩니다. 저자들은 FindTheFlaws라는 이름의 데이터셋을 통해 전문 검증된 정확한 솔루션과 특정 오류를 강조한 잘못된 솔루션을 포함하고 있습니다.

- **Technical Details**: FindTheFlaws 데이터셋은 전문가 주석이 달린 긴 형식의 질문과 솔루션으로 구성됩니다. 연구에서는 debate, critique, prover-verifier games와 같은 다양한 AI 감독 방식의 확장성을 평가하며, 각 모델의 비판적 능력을 평가합니다. 모델들의 성능을 통해 특정 데이터셋에서 잘못된 성능을 보이는 모델이 보다 능력 있는 모델의 판단자로 사용될 수 있음을 제안합니다.

- **Performance Highlights**: 평가 결과, 일부 태스크/데이터셋 조합에서 전문 기준이 최고 모델의 성능을 초과하는 경우가 발견되었습니다. 이는 전문 지식 기반이 더욱 확장 가능한 감독 실험에 유리할 수 있음을 나타냅니다. 이 연구는 AI 감독의 미래 방향에 대한 중요한 통찰을 제공합니다.



### Identifying Multi-modal Knowledge Neurons in Pretrained Transformers via Two-stage Filtering (https://arxiv.org/abs/2503.22941)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 자연어 처리(NLP)와 컴퓨터 비전 분야에서 다중 모달 대형 언어 모델(MLLMs)의 발전으로 이어졌습니다. 이러한 모델들은 시각과 언어의 통합된 이해를 가능하게 하지만, 내부 처리의 불투명성과 허위 정보 생성을 포함한 도전과제를 안고 있습니다. 이에 따라, MLLMs에서 지식의 위치를 명확히 하는 방법에 대한 필요성이 대두되고 있습니다.

- **Technical Details**: 본 연구에서는 MiniGPT-4라는 Transformer 기반 MLLM을 활용해 특정 지식과 연관된 뉴런을 찾는 방법을 제안합니다. 지식 뉴런을 추출하기 위해 두 단계의 필터링을 수행하며, 첫 번째 단계는 inpainting을 활용한 활성화 차이 필터링이고, 두 번째 단계는 GradCAM을 이용한 기울기 기반 필터링입니다. 이러한 방법은 MS COCO 2017 데이터셋을 사용한 이미지 캡션 생성 작업에서 그 효과성을 입증하였습니다.

- **Performance Highlights**: 실험 결과, BLEU, ROUGE, BERTScore를 통한 정량적 평가 및 활성화 히트맵을 이용한 정성적 평가에서 제안한 방법이 기존 방법들보다 더 높은 정확도로 지식을 찾을 수 있음을 보여주었습니다. 본 연구는 MLLMs의 지식 시각화 및 설명 가능성에 기여하며, 향후 지식 편집 및 제어의 가능성을 열어줍니다.



### Factored Agents: Decoupling In-Context Learning and Memorization for Robust Tool Us (https://arxiv.org/abs/2503.22931)
- **What's New**: 이번 논문은 전통적인 단일 에이전트 시스템의 한계를 극복하기 위한 새로운 분리된 에이전트 아키텍처를 제안합니다. 이 접근 방식은 에이전트를 두 개의 전문 성분으로 분해합니다: 첫째, 동적으로 사용 가능한 정보를 사용자 프롬프트에서 활용하는 고수준 계획자이자 컨텍스트 학습자인 LLM; 둘째, 도구 형식과 출력을 기억하는 소형 언어 모델입니다. 이 구조는 전통적인 단일 에이전트 설계에서 발생하는 문제들을 해결하고 에이전트의规划 정확도와 오류 회복력을 향상시킵니다.

- **Technical Details**: 제안된 분리된 에이전트 아키텍처는 메모리와 컨텍스트 적응의 역할을 분리합니다. 더 큰 컨텍스트 학습자는 프롬프트에서 새로운 정보를 동적으로 통합하고 적절한 도구 사용 프롬프트를 계획합니다. 소형 언어 모델은 도구 API에 대한 장기 지식을 유지하고 검색하는 역할을 합니다. 이러한 구조는 전통적인 단일 에이전트 접근 방식의 단점을 완화하면서 핵심 이점은 유지합니다.

- **Performance Highlights**: 실험적 평가 결과, 제안된 아키텍처는 실제 작업에서 계획 정확도와 오류 회복력을 유의미하게 향상시키는 것으로 나타났습니다. 이 연구는 인-컨텍스트 학습과 정적 메모리 사이의 고유한 절충의 이점을 보여줍니다. 새로운 에이전트 아키텍처가 더욱 견고하고 적응 가능한 에이전틱 AI 시스템을 개발하는 유망한 경로를 제시한다고 결론짓습니다.



### LLM-based Agent Simulation for Maternal Health Interventions: Uncertainty Estimation and Decision-focused Evaluation (https://arxiv.org/abs/2503.22719)
- **What's New**: 이 연구는 제한된 데이터 환경에서 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 모성 건강 프로그램의 에이전트 기반 시뮬레이션을 수행합니다. 기존의 방법들이 많은 도메인 지식과 데이터 세트를 요구하는 대신, LLM은 폭넓은 세계 지식을 통해 더 적은 데이터로 복잡한 행동을 모델링할 수 있는 가능성을 보여줍니다. 특히, 자동 메시지와 라이브 대표자에 의한 건강 정보 전달에서 수혜자의 청취 행동을 예측하는 데 중점을 두었습니다.

- **Technical Details**: 본 연구에서는 LLM을 기반으로 한 시뮬레이션에서 에피스테믹 불확실성(epistemic uncertainty)을 이진 엔트로피(binary entropy)를 통해 추정하는 방법을 제안합니다. 여러 샘플을 사용하여 모델의 강건성을 향상시키고, 개별 모델보다 F1 점수와 모델 보정을 개선하는 앙상블(ensemble) 접근 방식을 적용했습니다. 이러한 방법은 데이터가 제한된 환경에서도 가능한 의사결정을 지원하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 보고된 결과는 개별 모델에 비해 향상된 성능을 보여주며, LLM의 예측이 데이터가 부족한 환경에서의 건강 개입 가능성과 시험 구현에 어떻게 기여할 수 있는지를 보여줍니다. 따라서 이 연구는 공공 건강, 재난 대응 등 다양한 분야에서 신속한 개입 평가가 필요한 상황에서도 적용할 수 있는 가능성을 제시합니다. 이와 관련된 코드와 프롬프트는 연구에서 사용된 링크에서 확인할 수 있습니다.



### CodeScientist: End-to-End Semi-Automated Scientific Discovery with Code-based Experimentation (https://arxiv.org/abs/2503.22708)
Comments:
          98 Pages (13 pages: main paper body; 85 pages: appendix)

- **What's New**: 이번 연구에서는 CodeScientist라는 새로운 자율 과학 발견(ASD) 시스템을 소개합니다. 이 시스템은 기존 코드베이스와 유사한 설계 공간을 탐색하는 한계를 극복하고, 아이디어 생성 및 실험 구성을 유전적 탐색(genetic search)의 형태로 재구성합니다. CodeScientist는 연구 논문과 코드 블록의 조합을 이용하여 자동화된 실험을 수행합니다.

- **Technical Details**: CodeScientist 시스템은 언어 모델을 부르는 것과 같은 도메인 내 일반 작업을 정의하는 코드 블록을 활용하여 연구 아이디어를 생성하는 자동화된 실험을 수백 건 수행합니다. 이 시스템은 기존 연구에서 수행한 평가 방식뿐만 아니라 외부 리뷰와 코드 리뷰, 복제 시도를 포함한 다각적 평가(multi-faceted evaluation)를 통해 발견된 결과들을 검증합니다.

- **Performance Highlights**: 이 시스템을 통해 19개의 발견이 이루어졌으며, 이 중 6개는 최소한의 신뢰성과 혁신성을 갖춘 것으로 평가되었습니다. 발견들은 새로운 작업, 에이전트, 메트릭 및 데이터를 포함하여, 기존의 벤치마크 최적화에서 더 넓은 발견으로의 질적 변화를 제안합니다.



### UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving (https://arxiv.org/abs/2503.24381)
Comments:
          14 pages; Dataset: this https URL Code: this https URL

- **What's New**: UniOcc는 점유 예측(occupancy forecasting) 및 현재 프레임 점유 예측을 위한 포괄적인 벤치마크입니다. 다양한 실제 데이터셋(nuScenes, Waymo)과 고충실도 시뮬레이터(CARLA, OpenCOOD)의 데이터를 통합하여 2D/3D 점유 레이블과 각 복셀(voxel) 흐름(flow) 주석을 제공합니다. 새로운 평가 지표를 통합하여 기존의 잘못된 중간 진리(ground-truth)에 의존하지 않고 점유 품질을 평가할 수 있는 robust한 방안을 제시합니다.

- **Technical Details**: UniOcc는 단일 데이터셋에 의존하던 기존 방법들의 제약을 극복하고 크로스 데이터셋 학습을 지원합니다. CARLA 시뮬레이션을 활용하여 다양한 훈련 데이터를 제공하고, 각 복셀에 대한 전향(forward) 및 역방향(reverse) 흐름 주석을 통해 동적 장면 단서를 포착할 수 있도록 합니다. 이는 협력적 주행(cooperative driving) 시나리오를 지원하는 최초의 데이터셋이기도 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 대규모 다양한 훈련 데이터와 명시적인 흐름 정보가 점유 예측 및 예측 성능을 유의미하게 향상시킨다는 것을 입증하였습니다. 우리는 UniOcc이 점유 중심 연구의 촉매제로 작용하여 자율 주행에서의 혁신을 촉진할 것이라고 기대합니다. 또한 기존 방법들이 크로스 도메인 일반화에 어려움을 겪고 있음을 보여주어 향후 연구의 방향성을 제시합니다.



### Any2Caption:Interpreting Any Condition to Caption for Controllable Video Generation (https://arxiv.org/abs/2503.24379)
Comments:
          Project Page: this https URL

- **What's New**: 본 연구는 사용자의 의도를 정확히 해석하는 데 발생하는 병목 현상을 해결하기 위해 Any2Caption이라는 새로운 프레임워크를 제시합니다. 이 프레임워크는 비디오 생성 시 다양한 조건을 주석으로 구조화하여 비디오 생성기에게 더 나은 지침을 제공합니다. 추가로 Any2CapIns라는 대규모 데이터셋을 구축하여 다양한 조건을 활용한 주석 튜닝을 가능하게 합니다.

- **Technical Details**: Any2Caption은 텍스트, 이미지, 비디오 및 특수한 신호(예: 지역, 모션, 카메라 포즈)를 포함한 다양한 입력을 해석하여 밀집하고 구조화된 캡션을 생성하는 MLLM 기반의 조건 인터프리터입니다. 이를 통해 비디오 생성 모델의 제어력과 비디오 품질이 향상됩니다. 본 연구는 337K의 사례와 407K의 조건으로 구성된 Any2CapIns 데이터셋을 통해 성능을 평가합니다.

- **Performance Highlights**: Any2Caption은 여러 SoTA 비디오 생성 모델과의 통합에서 높은 품질의 비디오 생성을 가능하게 하며, 특히 복합적인 조건을 처리하는 데 탁월한 성능을 보입니다. 실험 결과, Any2Caption을 통해 생성된 비디오는 보다 풍부하고 의미 있는 주석으로 향상되며, 제어 가능성과 비디오 품질에서 기존 모델들을 뛰어넘는 결과를 나타냅니다.



### Harnessing the Reasoning Economy: A Survey of Efficient Reasoning for Large Language Models (https://arxiv.org/abs/2503.24377)
Comments:
          In Progress; Paper list Repo: this https URL

- **What's New**: 최근 대형 언어 모델(Large Language Models, LLMs)에서의 발전은 복잡한 추론(task reasoning) 작업 수행 능력을 크게 향상시켰습니다. 이 연구에서는 신속한 직관적 사고(System 1)에서 느리지만 깊은 사고(System 2)로의 전환이 이루어졌으며, 이는 작업의 정확성을 높이지만 계산 비용이 증가하는 단점을 동반합니다. 따라서 성능과 계산 비용 사이의 균형을 맞추는 추론 경제(reasoning economy)의 개념이 중요하다는 점을 강조하고 있습니다.

- **Technical Details**: 이 연구에서는 LLMs의 훈련 후 및 테스트 시간 추론 단계에서의 추론 경제를 분석합니다. 첫째, LLMs에서의 추론 비효율성의 원인을 파악하고, 둘째 다양한 추론 패턴의 행동을 분석하며, 셋째 추론 경제를 달성하기 위한 잠재적 해결책을 모색합니다. 최종적으로는 효율적인 LLMs를 달성하기 위한 도전 과제와 해결책을 명확하게 제시합니다.

- **Performance Highlights**: LLMs는 Chain-of-Thought prompting의 도입으로 다양한 언어 이해 및 생성 작업에서 뛰어난 성능을 발휘하고 있습니다. 그러나 모든 작업이 깊은 사고를 필요로 하지 않기 때문에, 각 작업의 복잡도에 맞게 계산 노력을 조정할 필요가 있습니다. 효율적인 리소스 사용을 위해 필요한 추론 단계를 강조하며, 불필요한 중복을 줄이고 동적으로 계산 노력을 조정하는 방법을 제안합니다.



### Exploring the Effect of Reinforcement Learning on Video Understanding: Insights from SEED-Bench-R1 (https://arxiv.org/abs/2503.24376)
Comments:
          Technical Report (In Progress); Code released at: this https URL

- **What's New**: 본 논문은 멀티모달 대규모 언어 모델(MLLMs)의 비디오 이해를 평가하기 위한 새로운 벤치마크인 SEED-Bench-R1을 제안합니다. SEED-Bench-R1은 복잡한 일상적인 계획 작업을 여러 선택 질문 형태로 포함하여, 정교한 인식(perception)과 논리적 추론(logical reasoning)을 요구합니다. 또한, 이 벤치마크는 세 가지 수준의 일반화(generalization) 시나리오를 통해 MLLMs의 포스트 트레이닝(post-training) 방법을 체계적으로 평가합니다.

- **Technical Details**: SEED-Bench-R1은 현실적인 일상 활동을 기반으로 한 비디오를 사용하여, 모델이 목표를 이해하고 긴 시간 동안 시각적인 진행을 추적하며, 복잡한 환경 관찰을 인지하고, 세계 지식을 사용하여 다음 행동을 추론할 수 있도록 설계되었습니다. 이 벤치마크는 교육 데이터셋을 기반으로 하며, 명확하게 검증 가능한 정답을 제공하여 일반화 능력을 철저히 평가할 수 있는 구조로 되어 있습니다. Qwen2-VL-Instruct-7B를 사용하여 RL 및 감독된 파인튜닝(SFT) 방법을 비교하여 RL이 데이터 효율성과 성능 면에서 우수하다는 것을 보여주었습니다.

- **Performance Highlights**: 실험 결과, RL은 특히 OOD(Out-of-Distribution) 시나리오에서 SFT를 능가하며, 비디오 이해의 일반적인 벤치마크에서도 높은 성과를 보였습니다. RL은 시각적 인식을 향상시키고 COT(Chain of Thought) 토큰을 동적으로 쿼리하도록 모델을 교육하는 데 효과적이었습니다. 그러나 모델이 때때로 중요한 시각적 단서를 무시하는 등, 몇 가지 한계점도 드러났고, 이는 향후 연구와 개선 방향 설정에 중요한 요소가 될 것입니다.



### Effectively Controlling Reasoning Models through Thinking Intervention (https://arxiv.org/abs/2503.24370)
- **What's New**: 이번 논문에서는 Reasoning-enhanced 대규모 언어 모델(LLM)들이 최종 답변을 생성하기 전에 중간 사고 단계를 명확히 생성함으로써 복잡한 문제 해결에서 우수한 성능을 보인다는 점을 강조합니다. 저자들은 Thinking Intervention이라는 새로운 패러다임을 제안하여 모델의 내부 사고 과정을 명확하게 안내하고, 이로 인해 모델 행동을 조정할 수 있는 새로운 기회를 제공합니다.

- **Technical Details**: Thinking Intervention은 모델이 전통적인 프롬프트 엔지니어링을 넘어서서 사고 과정 중 특정 토큰 시퀀스를 삽입하거나 수정하여 더 세밀하게 제어할 수 있도록 합니다. 이 방식은 모델 훈련이 필요하지 않으며, 실제 환경에서 최소한의 엔지니어링 노력으로 배치할 수 있습니다. 또한 기존의 모델 제어 기법들과 호환되며, 올해 문맥 및 작업에 따라 적응적으로 사고 단계를 삽입하거나 수정할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: Thinking Intervention은 다양한 작업에서 성능을 크게 향상시킵니다. IFEval, SEP, XSTest 및 SORRY-Bench에서의 평가 결과, 이 접근법은 지침 따르기 작업에서 최대 6.7%의 정확도 향상과, 지침 계층 문제 해결에서 15.4% 개선, 그리고 안전 프롬프트에 대한 거부율을 40.0%까지 증가시켰습니다. 전반적으로, 저자들은 이 접근법이 LLM의 추론 프로세스에 대한 더 정밀하고 투명한 제어를 가능하게 한다고 주장합니다.



### Which LIME should I trust? Concepts, Challenges, and Solutions (https://arxiv.org/abs/2503.24365)
Comments:
          Accepted at the 3rd World Conference on eXplainable Artificial Intelligence (XAI 2025)

- **What's New**: 이 논문은 LIME(지역 해석 가능한 모델 비특정 설명)의 기초 개념과 알려진 한계를 종합적으로 탐구하고 정리한 최초의 조사입니다. LIME에 관한 다양한 적응 및 향상을 체계적으로 분류하고 주요 문제에 따른 분류법을 제공함으로써, 미래 연구의 방향을 제시하고 실무자들이 적합한 접근 방식을 식별하는 데 도움을 주고자 합니다. 특별히, 사용자 친화적인 웹사이트를 통해 LIME 관련 기술을 지속적으로 모니터링하고 정보를 업데이트합니다.

- **Technical Details**: LIME은 블랙박스 모델의 행동을 특정 인스턴스 주위에서 근사화하여 설명을 생성하는 모델 비특정 접근 방식입니다. 이 기술은 복잡한 모델에 대한 지역적 설명을 제공하지만, 안정성, 계산 비효율성, 특정 데이터 처리의 한계와 같은 여러 도전에 직면해 있습니다. 본 논문은 LIME 프레임워크 내부의 기술적 수정과 특정 문제를 해결하는 솔루션에 따라 분류된 새로운 분류 체계를 도입하여 LIME의 다양한 확장 및 변형을 분석합니다.

- **Performance Highlights**: LIME과 그 확장에 대한 체계적인 평가는 연구 관점에서 반드시 필요합니다. 연구자들은 각 모델의 특성과 실무 요구 사항에 따라 적절한 LIME 기술을 선택하는 데 어려움을 겪고 있습니다. 본 논문은 이러한 기술을 효율적으로 설명할 수 있도록 지원하며, 기존 방법의 장점과 한계를 조망하고 미래 연구를 위한 유망한 방향성을 제공합니다.



### Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation (https://arxiv.org/abs/2503.24361)
Comments:
          Project website: this https URL

- **What's New**: 이 논문은 시뮬레이션 데이터와 실제 데이터를 혼합하여 정책(policy)을 공동 학습(co-training)하는 새로운 방법론을 제안합니다. 최근의 연구에서 시뮬레이션 데이터를 사용한 정책 학습이 실제 데이터에서만 학습한 경우보다 성능이 크게 향상될 수 있다는 점이 부각되었습니다. 그러나, 실제로 이 방법이 어떻게 효과적인지에 대한 체계적인 이해는 부족한 상황입니다.

- **Technical Details**: 이 연구에서는 시뮬레이션 데이터와 실제 데이터를 효과적으로 혼합하는 방법을 제시하고, 로봇 팔(robot arm)과 휴머노이드(humanoid)의 다양한 작업(task)을 통해 이를 실증적으로 검증합니다. 시뮬레이션 데이터는 두 가지 주요 원천으로 나뉘며, 각각 작업 인지(task-aware) 시뮬레이션과 작업 비인식(task-agnostic) 시뮬레이션으로 구분됩니다. 이 연구는 시뮬레이션 데이터가 실제 환경에서 어떻게 개선될 수 있는지에 대한 통찰을 제공합니다.

- **Performance Highlights**: 실험 결과, 시뮬레이션 데이터를 활용한 공동 학습이 평균 38%의 성능 향상을 이루었음을 보여주었습니다. 연구는 시뮬레이션 데이터와 실제 데이터 간의 간섭이 클 경우에도 성능 개선이 가능하다는 것을 발견했습니다. 이러한 findings는 로봇 공학 실무자들에게 중요한 전략을 제공할 수 있습니다.



### SQuat: Subspace-orthogonal KV Cache Quantization (https://arxiv.org/abs/2503.24358)
- **What's New**: 이번 논문에서는 SQuat(Subspace-orthogonal KV cache quantization)이라는 새로운 접근법을 소개합니다. SQuat는 기존의 KV 캐시 양자화 방법과는 달리, 쿼리 텐서들로 구성된 서브스페이스를 활용하여 과거 토큰의 키 텐서를 양자화하는 과정에서 양자화 오류가 주의 메커니즘에 미치는 영향을 최소화합니다. 이 방법은 모델 재학습이나 추가적인 데이터 수집 없이도 적용될 수 있으며, 이론적 토대를 기반으로 개발되었습니다.

- **Technical Details**: SQuat은 주어진 사용자 프롬프트의 모든 토큰에서 쿼리 텐서를 통해 작업 관련 서브스페이스를 먼저 구성합니다. 그런 다음, 각 토큰의 키 텐서를 양자화하면서, 양자화된 키 텐서와 원래 키 텐서의 차이가 이 서브스페이스에 대해 직교하도록 유지합니다. 이를 통해 중요한 과업 정보에 대한 양자화 오류의 영향을 줄이고, 최적의 업데이트 규칙을 통한 효율적인 연산이 가능합니다.

- **Performance Highlights**: SQuat는 Llama-2-7B 모델을 기반으로 할 때, 피크 메모리 사용량을 2.17배에서 2.82배까지 줄일 수 있으며, 처리량은 2.45배에서 3.60배까지 향상됩니다. 또한, 이 방법은 기존의 다른 비튜닝(baseline) 방법들에 비해 더욱 우수한 성능을 발휘하며, 14개의 다양한 벤치마크 과제를 포함한 다양한 평가에서 그 효율성을 입증하였습니다.



### ORAL: Prompting Your Large-Scale LoRAs via Conditional Recurrent Diffusion (https://arxiv.org/abs/2503.24354)
- **What's New**: 본 논문에서는 기존의 방법들이 직면한 확장성과 조정 가능성의 한계를 해결하기 위해 새로운 조건부 반복 확산(conditional recurrent diffusion) 프레임워크인 ORAL을 소개합니다. ORAL은 고유한 조건화 메커니즘을 포함하여 모델 아키텍처와 작업 사양을 통합하여, 진화하는 기초 모델을 통해 효율적으로 전이 가능한 LoRA 파라미터를 생성할 수 있습니다. 이 접근법은 수십억 개의 파라미터를 가진 대형 언어 모델에서도 조정 가능성을 유지하면서 확장을 성공적으로 수행합니다.

- **Technical Details**: ORAL의 주요 기여는 LoRA 파라미터의 유연한 생성을 위한 새로운 조건화 메커니즘을 개발한 것입니다. 이 메커니즘은 모델 아키텍처 및 텍스트 기반 작업 사양을 입력으로 사용하여, 특정 다운스트림 작업에 맞춤화된 LoRA 파라미터를 생성할 수 있게 합니다. ORAL은 기존의 반복 확산 아키텍처를 기반으로 하여, 자원 집약적인 재교육 없이도 진화하는 기초 모델에 생성된 파라미터를 원활하게 전이할 수 있는 새로운 조건부 파라미터 생성 파이프라인을 제안합니다.

- **Performance Highlights**: 다양한 실험을 통해 ORAL은 7개의 언어 작업, 4개의 비전 작업, 3개의 다중 모달 작업을 수행하였으며, 5개의 사전 학습된 LLM을 사용하여 그 효율성을 입증했습니다. 연구 결과, ORAL은 7777억 개의 파라미터를 효과적으로 처리하면서도 전통적인 미세 조정 방법과 비슷하거나 우수한 성능을 보여줍니다. 이는 ORAL이 기존 방법과 비교할 때 scalability, controllability 및 portability를 모두 충족하는 새로운 기준을 정립하고 있음을 의미합니다.



### Pro-Routing: Proactive Routing of Autonomous Multi-Capacity Robots for Pickup-and-Delivery Tasks (https://arxiv.org/abs/2503.24325)
Comments:
          25 pages, 7 figures, and 1 table

- **What's New**: 이 논문은 다중 용량 자율 로봇(autonomous robots) 시스템에서 픽업 및 배송 요청을 효율적으로 처리하는 새로운 프로액티브 롤아웃 기반의 라우팅 프레임워크를 제안합니다. 기존 연구는 오프라인에서 문제를 해결하거나 실시간 요청을 처리 하였지만, 안정성(stability) 보장을 희생했습니다. 본 연구에서는 이 두 가지 접근 방식을 연결하여 알고리즘의 안정성을 유지하면서도 실시간 수요에 적응할 수 있는 방법을 제공합니다.

- **Technical Details**: 제안된 시스템은 고정된 최대 대기 시간(maximum wait times) 내에서 배치된 요청의 수에 대한 적절한 의사 결정을 지원합니다. 이 시스템은 1회 요청 롤아웃(one-request-at-a-time rollout) 알고리즘과 시간이 지남에 따라 수요에 적응할 수 있는 메커니즘을 포함합니다. 안정성을 보장하기 위해 효율적인 함수를 도입하고, 충분히 큰 로봇 함대를 구성하는 알고리즘을 제공합니다.

- **Performance Highlights**: 하버드 대학의 Evening Van Service의 실제 요청 데이터를 사례 연구로 사용하여 제안된 알고리즘의 효과를 검증했습니다. 실험 결과, 제안된 방법이 현재 배포된 알고리즘 대비 6% 더 많은 요청을 처리하면서도 평균 대기 시간을 33% 감소시킨 것으로 나타났습니다. 이로 인해 제안된 프레임워크의 실용성과 효율성이 입증되었습니다.



### BEATS: Bias Evaluation and Assessment Test Suite for Large Language Models (https://arxiv.org/abs/2503.24310)
Comments:
          32 pages, 33 figures, preprint version

- **What's New**: 이 연구에서는 BEATS라는 새로운 프레임워크를 소개하여 대규모 언어 모델(LLMs)의 편향(Bias), 윤리(Ethics), 공정성(Fairness), 사실성(Factuality)을 평가하는 방법을 제안합니다. 이 프레임워크를 바탕으로 29개의 다양한 지표를 통해 모델의 성능을 측정하는 편향 벤치마크를 개발했습니다. 이러한 지표들은 사회적 편견의 지속 가능성을 정량적으로 평가할 수 있는 가능성을 제공합니다.

- **Technical Details**: BEATS 프레임워크는 LLM의 편향, 윤리, 공정성 및 사실성을 평가하기 위한 체계적이며 확장 가능한 절차를 제공합니다. 이 프레임워크의 핵심은 다양한 사고와 윤리적 기준을 탐구하기 위해 설계된 질문 데이터셋입니다. 연구자는 이러한 질문을 통해 LLM의 응답을 분석하고 그 결과를 구조적 데이터베이스에 저장하여 벤치마크 평가를 수행합니다.

- **Performance Highlights**: 실험 결과, 업계 선두 모델의 37.65% 출력에서 어떤 형태의 편향이 발견되었습니다. BEATS 프레임워크와 벤치마크는 LLM 평가의 일관성과 반응성을 높여야 할 필요성이 있다는 것을 강조합니다. 이 연구는 다양한 AI 모델의 공정성과 윤리적 기준에 대한 인식을 높이고 지속 가능한 AI 모델 개발을 촉진하는 데 목표를 두고 있습니다.



### A Systematic Evaluation of LLM Strategies for Mental Health Text Analysis: Fine-tuning vs. Prompt Engineering vs. RAG (https://arxiv.org/abs/2503.24307)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 활용한 정신 건강 텍스트 분석을 위한 세 가지 접근 방식을 체계적으로 비교하였습니다: prompt engineering, retrieval augmented generation (RAG), 그리고 fine-tuning입니다. 이 연구는 LLaMA 3를 사용하여 감정 분류 및 정신 건강 상태 감지 작업을 두 개의 데이터셋에서 평가하였습니다. 연구 결과, fine-tuning이 감정 분류에서 91%, 정신 건강 조건 분류에서 80%의 정확도를 달성하였으며, prompt engineering과 RAG는 보다 유연한 배포가 가능하지만 보통의 성능(40-68% 정확도)을 보여주었습니다.

- **Technical Details**: 정신 건강 텍스트 분석을 위한 LLM의 세 가지 접근 방식은 fine-tuning, prompt engineering, RAG입니다. 특히, fine-tuning은 높은 정확도를 요구하지만 많은 컴퓨팅 리소스와 대규모 훈련셋을 필요로 합니다. 반면, prompt engineering과 RAG는 상대적으로 적은 자원으로 보다 유연한 배포가 가능하게 하며, 다양한 설정에서 효과적으로 구현할 수 있다는 장점이 있습니다.

- **Performance Highlights**: 이 연구는 정신 건강 분야에서 LLaMA 3 기반 모델의 효과를 입증하였으며, 감정 분류 및 정신 건강 상태 분류에서 매우 높은 정확도를 기록했습니다. 이러한 결과는 임상 환경에서 LLM 기반 솔루션의 구현에 있어 중요한 통찰력을 제공합니다. 향후 정신 건강 평가 도구의 개발에 중요한 의미를 가지며, 높은 정확도의 fine-tuning 외에도 prompt engineering과 RAG 접근 방식이 자원과 배포 유연성 면에서 유효한 대안이 된다는 점을 강조하고 있습니다.



### Evaluating machine learning models for predicting pesticides toxicity to honey bees (https://arxiv.org/abs/2503.24305)
- **What's New**: 이번 연구는 ApisTox라는 데이터세트를 중심으로, 벌 관련 독성 데이터를 수집 및 평가하였습니다. ApisTox는 꿀벌(Apis mellifera)에 대한 실험적으로 검증된 화학 독성 정보를 제공하는 가장 포괄적인 데이터세트로, 기존의 의료 데이터세트와는 다른 화학 공간을 표현하고 있습니다. 이 연구는 기계 학습(Machine Learning) 기술로 독성 예측의 한계를 조사하며, 현재 가장 발전된 알고리즘이 생물 의학 데이터에 대해서만 훈련되었음을 보여줍니다.

- **Technical Details**: 연구는 ApisTox 데이터세트를 1,035개의 화합물로 구성된 데이터셋으로 분석하였으며, 이는 ECOTOX, PPDB, BPDB 데이터베이스로부터 파생되었습니다. 데이터세트는 296개의 독성 화합물과 739개의 비독성 화합물로 구성되어 있으며, 특정 화학 구조에 대한 다양한 기계 학습 접근 방식을 통해 독성을 예측하고 있습니다. 이 데이터세트는 훈련-시험 분할을 마련하여 다른 알고리즘을 공정하게 비교할 수 있도록 설계되었으며, 이는 기존의 의료 화학 데이터세트와 차별화됩니다.

- **Performance Highlights**: ApisTox에 대한 기계 학습 알고리즘의 성능 저하가 나타나며, 현재의 최첨단 알고리즘이 농약 독성 및 환경 데이터를 일반화하는 데 한계가 있음을 보여줍니다. 연구는 다양한 기계 학습 접근 방식, 예를 들어 분자 지문(molecular fingerprints) 및 그래프 뉴럴 네트워크를 통해 독성 예측 모델의 잠재력을 평가하고, 이러한 방법이 환경적으로 중요한 곤충에 적합한 예측 결과를 도출하는 데 어려움을 겪고 있음을 강조합니다. 이러한 연구는 안전한 농약 개발 및 꿀벌과 같은 필수 꽃가루 매개체를 보호하는 데 중요한 기여를 할 것으로 기대됩니다.



### Shape Expressions with Inheritanc (https://arxiv.org/abs/2503.24299)
Comments:
          Accepted in Extended Semantic Web Conference, ESWC, 2025

- **What's New**: 이 논문은 Shape Expressions 언어(ShEx)의 상속 메커니즘을 공식적으로 도입합니다. 객체 지향 프로그래밍 언어의 상속에 영감을 받아 재사용성, 모듈성 및 더 유연한 데이터 모델링과 같은 장점을 제공합니다. 예제를 통해 상속 메커니즘의 주요 기능을 설명하고, ShEx 2.1의 의미론을 확장하여 구문 및 형식적 의미론을 제시합니다.

- **Technical Details**: ShEx는 RDF 데이터를 기술하고 검증하는 고급 언어로 2014년에 제안되었습니다. 이 논문에서 제안하는 상속 메커니즘은 자식 형태가 부모로부터 새로운 필수 속성을 확장하고, 부모의 속성에 추가 제약 조건을 부여할 수 있게 합니다. 여기서 다중 상속이 허용되며, 형태 표현은 추상적으로 정의되어 독립적으로 사용될 수 없도록 합니다.

- **Performance Highlights**: 상속 메커니즘은 형태 표현의 재사용과 조합을 용이하게 하여 데이터 모델링의 복잡성을 줄입니다. 또한 기존 ShEx 검증 알고리즘의 알고리즘 복잡성을 유지하면서 새로운 검증 알고리즘을 도출합니다. 이 연구는 IEEE에 의해 표준화 중인 ShEx 언어의 다음 버전에 제출될 예정입니다.



### Value of Information-based Deceptive Path Planning Under Adversarial Interventions (https://arxiv.org/abs/2503.24284)
Comments:
          10 pages, 4 figures

- **What's New**: 이번 논문에서는 적대적인 개입(adversarial interventions) 하에서 기만적 경로 계획(deceptive path planning, DPP) 문제를 해결하기 위한 새로운 MDP(Markov Decision Process) 기반 모델을 제안합니다. 기존의 연구들은 대부분 수동적 관찰자(passive observer)를 가정하고 있어 실질적인 문제 해결에 한계가 있었습니다. 저자들은 정보의 가치(value of information, VoI) 기반 목표를 통해 DPP 정책을 설계하는 새로운 방법론을 도입하고, 이를 통해 적대적 개입에 대응할 수 있는 기법을 개발했습니다.

- **Technical Details**: 저자들은 적대적 개입이 가능한 상황에서 DPP를 수행하기 위해 새로운 MDP 모델을 구축하며, 이를 위해 두 가지 새로운 VoI 기반 기만 측정 지표를 정의합니다. 이 모델은 관찰자의 정보 가치를 고려해 기만적인 정책의 영향을 평가할 수 있도록 설계되었습니다. 또한, 선형 계획(linear programming, LP) 이론을 활용하여 이러한 VoI DPP 문제를 효과적으로 해결할 수 있는 계산 방법을 도출하였습니다.

- **Performance Highlights**: 실험 결과, 제안한 솔루션 방법이 적대적인 환경에서 기만적인 경로를 달성하는 데 효과적임을 보여주었습니다. 기존 DPP 방법 및 보수적인 경로 계획(conservative path planning, CPP) 방법과 비교했을 때, 저자들의 VoI DPP 접근법이 더욱 우수한 성능을 발휘하는 것을 확인했습니다. 특히, 실제 관찰자 개입 상황에서 저비용 경로를 달성할 수 있는 능력이 강조되었습니다.



### AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World (https://arxiv.org/abs/2503.24278)
- **What's New**: 새로운 연구에서는 로봇 정책 평가의 자동화를 위해 AutoEval 시스템을 제안합니다. 이 시스템은 최소한의 인간 개입으로 일반화된 로봇 정책을 연중무휴로 평가할 수 있습니다. 사용자는 AutoEval 대기열에 평가 작업을 제출하여 자동으로 정책을 평가하고 성공률을 기록하는 등의 작업을 수행하도록 설계되었습니다.

- **Technical Details**: AutoEval은 대규모로 학습된 모델을 활용하여 자동화된 장면 리셋(scene reset) 및 성공 판별(success detection) 기능을 갖추고 있습니다. 이를 통해 기존의 수동 평가에서 요구되던 인간의 개입을 최소화하고, 평가 속도를 높일 수 있습니다. 연구팀은 BridgeData 로봇 환경에서 다양한 평가 장면을 제공하여 평가 효율성을 극대화하는 방안을 모색하였습니다.

- **Performance Highlights**: AutoEval은 24시간 동안 500회의 평가 에피소드를 처리할 수 있으며, 높은 신뢰도로 인간의 손으로 수행한 평가 결과와 잘 일치하는 평가 결과를 생성합니다. 이 시스템은 다양한 작업에 대한 신뢰성 있는 정책 성능 추정치를 제공하며, 특히 일반적인 시뮬레이션 방법으로는 평가하기 어려운 작업에서도 효과적으로 적용됩니다. 이를 통해 로봇 학습 연구의 민주화와 정책 비교의 공정성을 도모하고자 합니다.



### Evaluating and Designing Sparse Autoencoders by Approximating Quasi-Orthogonality (https://arxiv.org/abs/2503.24277)
- **What's New**: 이번 연구에서는 Sparse Autoencoders(SAEs)의 새로운 접근 방식을 제안합니다. 기존의 hyperparameter 조정 없이도 입력 임베딩과 피처 벡터 간의 실제 관계를 기반으로 한 새로운 이론적 근사치를 개발했습니다. 이를 통해 Approximate Feature Activation(AFA)와 함께 시각화 도구인 ZF Plot을 도입했습니다.

- **Technical Details**: SAEs는 dense embeddings을 해석 가능한 feature vectors의 선형 조합으로 분해할 수 있음이 입증되었습니다. 연구진은 *linear representation hypothesis (LRH)*와 *superposition hypothesis (SH)*의 정의에 근거하여 AFA의 개념을 도입하고, 연결된 새로운 평가 메트릭을 제안했습니다. top-AFA SAE 구조는 이러한 아이디어를 기반으로 하며 객관적인 이론적 근거를 제공합니다.

- **Performance Highlights**: 새로운 top-AFA SAE 구조는 최신 기술과 비교하여 유사한 reconstruction loss를 달성했습니다. 이 방식은 hyperparameter tuning을 요구하지 않으면서도, 기존의 top-k SAEs 이상으로 뛰어난 성능을 보입니다. 연구진의 접근법은 SAEs의 메커니즘 해석 가능성을 심화시킬 수 있는 실험적 연구의 새로운 방향을 열었습니다.



### Visual Acoustic Fields (https://arxiv.org/abs/2503.24270)
- **What's New**: 본 연구에서는 3D 공간 내에서 시각과 음향 신호의 교차 모델 관계를 링크하는 'Visual Acoustic Fields'라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 두 가지 주요 모듈, 즉 소리 생성(sound generation)과 소리 위치화(sound localization)를 포함합니다. 'Visual Acoustic Fields'는 3D Gaussian Splatting(3DGS)을 이용하여 시각 신호와 타격 소리를 연결하여, 현존하는 데이터셋과는 달리 3D 맥락에서 시각과 음향 신호를 연결합니다.

- **Technical Details**: 연구진은 다중 시점 이미지(multiview images)와 해당 타격 위치 및 연관된 소리로 이루어진 데이터셋을 수집하기 위한 파이프라인을 구현하였습니다. 구조-모션 추정(structure-from-motion) 기술을 사용하여 이미지의 카메라 포즈를 추정하고, 이 정보를 바탕으로 충격 소리와 그에 상응하는 시각 신호를 통합합니다. 'AudioCLIP' 기능을 사용하여 시각적 및 청각적 신호를 매칭하고, 타격점에서의 소리를 생성하기 위해 오디오 확산 모델을 활용합니다.

- **Performance Highlights**: 실험 결과, 수집된 시각-소리 쌍의 타격 위치를 3D 공간에서 정확하게 로컬라이징할 수 있음을 보여주었습니다. 예측된 타격 소리는 실제 타격 위치와 일치하며, 'Visual Acoustic Fields'를 통해 충격 지역이나 객체를 소리에 기반하여 정밀하게 검색할 수 있음을 증명하였습니다. 이 연구는 로봇 공학, 가상 현실 및 콘텐츠 생성 등 다양한 분야에서 응용될 가능성을 제시합니다.



### New Statistical Framework for Extreme Error Probability in High-Stakes Domains for Reliable Machine Learning (https://arxiv.org/abs/2503.24262)
- **What's New**: 이번 연구에서는 Extreme Value Theory (EVT)를 기반으로 한 새로운 통계적 프레임워크를 제안하여 기존의 평균 기반 검증 방법의 한계를 극복하고 최악의 예측 실패를 수치적으로 평가할 수 있는 방법을 제공합니다. 이 프레임워크는 Monte Carlo 교차 검증 기법과 통합되어 머신 러닝 모델의 최대 오류를 예측하는 데 사용할 수 있습니다. 이를 통해 극단적 오류를 정량화하는 새로운 접근 방식이 현실 세계의 데이터에 적용됨을 보여줍니다.

- **Technical Details**: Extreme Value Theory (EVT)는 극단적인 이벤트를 모델링하고 분석하기 위해 고안된 통계적 프레임워크입니다. EVT는 주로 데이터의 중심 경향을 설명하는 전통적인 통계 방법과 달리, 극단적 편차의 확률과 크기를 정량화하는 도구를 제공합니다. 이 연구에서는 GEV(Generalized Extreme Value) 및 GPD(Generalized Pareto Distribution)와 같은 분포를 통해 극단적 오류를 평가하기 위한 알고리즘을 제시합니다.

- **Performance Highlights**: 실험 결과, 제안된 EVT 기반 접근 방식은 회귀 작업에서 극단적 오류를 정량화하고 예측하는 데 효과적임이 입증되었습니다. 합성 데이터와 실제 데이터 세트를 사용하여 이 방법론의 유효성을 평가하였으며, EVT가 가지고 있는 이론적 기초와 실제 구현 방법에 대해 논의하였습니다. 이 연구는 모델 신뢰성을 평가하는 데 있어 EVT의 중요성을 강조하며, 불확실성 정량화가 핵심인 신기술의 안전한 배포를 보장합니다.



### Beyond a Single Mode: GAN Ensembles for Diverse Medical Data Generation (https://arxiv.org/abs/2503.24258)
- **What's New**: 이 논문은 Generative Adversarial Networks (GANs)의 집합을 사용하여 의료 이미징에서 인공 데이터 생성을 위한 새로운 방법을 제안합니다. GANs는 높은 품질 샘플 생성, 다양한 모드 커버리지를 제공하지만, 여전히 여러 문제를 겪고 있습니다. 이를 해결하기 위해, 저자들은 다중 목표 최적화 문제를 해결하여 최적의 GAN 앙상블을 선택하는 새로운 방법을 제시합니다.

- **Technical Details**: 이 연구는 22개의 다양한 GAN 아키텍처를 포함하고 있으며, 각 모델의 훈련 단계에서 샘플링하여 총 110개의 고유 구성을 만들었습니다. 생성된 의료 이미지의 품질을 높이기 위해, GAN 앙상블의 각 모델은 고유한 기여를 하여 중복성을 최소화합니다. 다수의 아키텍처와 훈련 방법을 통합하여 리얼 데이터의 복잡성을 잘 반영하는 강력한 앙상블을 구축하는 것을 목표로 합니다.

- **Performance Highlights**: 저자들은 제안한 GAN 앙상블 방법이 세 가지 의료 데이터 세트를 통해 수행된 광범위한 평가에서 우수한 결과를 보여줌을 입증했습니다. 이 연구는 기본 연구 및 임상 훈련, 알고리즘 검증에서의 활용성을 높이는데 중요한 기여를 합니다. 궁극적으로, 이 방식은 진단 모델링과 같은 다운스트림 작업의 효율성을 개선하는 데 기여할 것입니다.



### Spatio-temporal Prediction of Fine-Grained Origin-Destination Matrices with Applications in Ridesharing (https://arxiv.org/abs/2503.24237)
- **What's New**: 이 논문은 ridesharing 플랫폼에서의 네트워크 기반 여행자 요청의 정확한 시공간 예측에 대한 필요성을 강조합니다. 또한, 기존 연구에서 상대적으로 미혹된 지역 간 Origin-Destination (OD) 수요 예측 문제를 다룹니다. 새로운 예측 모델인 OD-CED를 소개하여 데이터 희소성(data sparsity) 문제를 완화하고, 의미론적(semantic) 및 지리적(geographic) 의존성을 포착하는 방법을 제시합니다.

- **Technical Details**: OD-CED 모델은 두 단계로 구성되어 있으며, 첫 번째는 N개의 미세 세포를 M개의 조잡한(super) 세포로 변환하여 계산 요구 사항을 줄이는 전처리(preprocess) 단계입니다. 두 번째는 여러 헤드(self-attention) 네트워크를 활용해 의미론적 및 지리적 의존성을 학습하는 학습(learning) 단계입니다. 이를 통해 OD 수요 예측 시 발생할 수 있는 데이터의 불균형과 희소성을 해결하고자 합니다.

- **Performance Highlights**: OD-CED 모델은 기존의 통계적 방법보다 최대 45%의 루트 평균 제곱 오차(root mean square error)를 감소시키고, 60%의 가중 평균 절대 백분율 오차(weighted mean absolute percentage error)를 개선했습니다. 이러한 성과는 OD 매트릭스의 희소성이 90%를 초과하는 경우에도 효과적으로 적용될 수 있습니다.



### What, How, Where, and How Well? A Survey on Test-Time Scaling in Large Language Models (https://arxiv.org/abs/2503.24235)
- **What's New**: 최근 데이터와 파라미터를 확장하는 것에 대한 관심이 줄어들면서, 테스트 시간 확장(test-time scaling, TTS)이라는 새로운 연구 주제가 부각되었습니다. TTS는 대규모 언어 모델(large language models, LLMs)의 문제 해결 능력을 더욱 향상시킬 수 있음을 보여줍니다. 특히, 수학 및 코딩과 같은 전문적 추론 과제뿐 아니라 열려 있는 질문 응답(open-ended Q&A) 과제 등 일반 과제에서도 큰 breakthroughs를 가능하게 합니다.

- **Technical Details**: 본 연구는 TTS 연구의 네 가지 핵심 차원인 무엇을 확장할 것인지, 어떻게 확장할 것인지, 어디서 확장할 것인지, 얼마나 잘 확장할 것인지에 따라 구조화된 통합적이고 다차원적인 프레임워크를 제안합니다. 이러한 분류법을 기반으로 우리는 방법론, 적용 시나리오, 평가 측면을 포함한 광범위한 검토를 수행합니다. 각 기법의 기능적 역할을 강조하는 체계화된 분해를 통해 TTS의 주요 개발 경과를 정리합니다.

- **Performance Highlights**: TTS의 분석을 통해 우리는 실용적인 배치를 위한 가이드를 제공하고 몇 가지 오픈 챌린지를 식별합니다. 여기에는 추가 확장, 기법의 기능적 본질 명확화, 다양한 작업에 대한 일반화 및 기타 속성에 대한 통찰력이 포함됩니다. 이러한 방향은 향후 TTS 연구에서 중요한 기여를 할 수 있는 가능성을 제시합니다.



### MB-ORES: A Multi-Branch Object Reasoner for Visual Grounding in Remote Sensing (https://arxiv.org/abs/2503.24219)
- **What's New**: 이 논문에서는 원격 감지 이미지에 대해 객체 검출(Object Detection, OD)과 시각적 기초(Visual Grounding, VG)를 통합하는 통합 프레임워크를 제안하고 있습니다. 전통적인 OD와 VG 작업을 위한 직관적인 사전 지식을 수립하기 위해, 언급 표현 데이터를 사용하여 오픈 세트 객체 감지기를 세밀 조정하고, 부분적으로 감독된 OD 작업으로 설정합니다. 이러한 구조를 통하여 모든 객체를 탐지하면서 특정 객체의 위치를 정확하게 찾는 것을 목표로 합니다.

- **Technical Details**: 제안된 모델은 객체 질의, 클래스 임베딩, 및 제안 위치로 구성된 그래프 표현을 사용하여 각 이미지를 구성합니다. 멀티-브랜치 네트워크는 공간적, 시각적, 범주적 특성을 통합하여 작업 인식 제안을 생성하며, 객체 추론 네트워크는 제안들 사이의 확률을 할당합니다. 이 과정은 마지막으로 언급된 객체를 로컬라이즈하기 위한 부드러운 선택 메커니즘으로 이어집니다.

- **Performance Highlights**: 이 방법은 OPT-RSVG 및 DIOR-RSVG 데이터 세트에서 뛰어난 성능을 입증하였으며, 최신 방법들에 비해 상당한 성능 개선을 보여 주었습니다. 전통적인 OD 기능을 유지하면서도 보다 다양한 시나리오에서 OD의 적용 가능성을 확대하였습니다. 또한, 이 논문의 코드는 연구 결과를 재현하고 실험할 수 있도록 제공될 예정입니다.



### DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting (https://arxiv.org/abs/2503.24210)
Comments:
          CVPR 2025. Project Page: this https URL

- **What's New**: 이번 논문에서는 DiET-GS라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 블러가 있는 다중 시점 이미지에서 선명한 3D 표현을 재구성할 수 있는 기술을 제공합니다. 여기서는 이 프레임워크를 통해 이벤트 스트림과 확산(회복) 모델을 결합하여 최적의 시각적 품질을 달성하는 방법을 설명합니다.

- **Technical Details**: DiET-GS는 두 단계의 훈련 전략을 사용하여 이벤트 스트림과 사전 훈련된 확산 모델의 지식을 효과적으로 결합합니다. 이를 통해 정밀한 색상 정보와 세부 정보를 재구성할 수 있는 새로운 수학적 프레임워크를 소개하며, Event Double Integral(EDI)을 사용하여 3DGAussian Splatting(3DGS)의 정교한 세부 정보를 회복합니다.

- **Performance Highlights**: 실험 결과, DiET-GS는 기존의 방법들과 비교하여 시각적 품질에서 크게 향상된 성능을 보여줍니다. 이는 합성 데이터와 실제 데이터 모두에서 검증되었으며, 최적의 색상과 세밀한 디테일을 복구하는 능력을 갖췄음을 입증합니다. 또한, 새로운 기술적 접근 방식이 기존의 이미지 복원 모델들보다 더 뛰어난 성능을 발휘합니다.



### Output Constraints as Attack Surface: Exploiting Structured Generation to Bypass LLM Safety Mechanisms (https://arxiv.org/abs/2503.24191)
Comments:
          15 pages, 13 figures, 4 tables Work In Progress

- **What's New**: 이번 연구는 Constrained Decoding Attack (CDA)라는 새로운 jailbreak 공격 방법을 제시하며, 기존 데이터-플레인(vulnerability) 취약점과는 다른 제어-플레인(control-plane) 공격 표면을 강조합니다. CDA는 구조화된 출력 제약을 악용하여 LLM의 안전 메커니즘을 우회하도록 설계되었습니다. 특히, 기존의 입력 프롬프트에 의존하는 공격 방식과 달리, CDA는 스키마 수준의 문법 규칙에 악의적 의도를 내포하여 공격을 수행합니다.

- **Technical Details**: 제어-플레인 취약점을 이용한 CDA는 LLM API의 구조화된 출력 기능을 통해 발생할 수 있는 새로운 보안 위협을 식별합니다. 연구자는 Chain Enum Attack을 통해, JSON Schema의 enum 기능을 남용하여 LLM의 안전 메커니즘을 우회할 수 있는 방법을 제시합니다. 이 공격 방식은 악의적인 질문의 의도를 숨기고 안전하지 않은 출력 내용을 생성함으로써 LLM의 보안성을 크게 위협합니다.

- **Performance Highlights**: Chain Enum Attack은 GPT-4 및 Gemini-2.0과 같은 다양한 산업 표준 LLM에서 5개의 안전 기준을 통한 공격 성공률 96.2%를 기록했습니다. 이는 기존 LLM 구조에서 간과된 취약점을 포함함으로써 LLM 안전성에 대한 재평가 및 패러다임 전환을 요구합니다. 연구자는 LLM 아키텍처의 보안 눈먼 지점을 강조하여, 현재의 데이터-플레인 위협에만 집중하는 안전 메커니즘이 필수 시스템에 대한 취약성을 남긴다고 주장합니다.



### Predicting Targeted Therapy Resistance in Non-Small Cell Lung Cancer Using Multimodal Machine Learning (https://arxiv.org/abs/2503.24165)
- **What's New**: 이 연구에서는 비소세포 폐암 환자들을 위한 osimertinib 저항성 예측을 위해 해석 가능한 다중 모달 머신러닝 모델을 개발하였습니다. 현재 osimertinib의 저항성을 정확하게 예측할 수 있는 표준 도구가 없다는 점을 해결하기 위해, 보험 의료기록을 비롯한 다양한 데이터를 통합하여 정밀한 폐암 관리와 치료 결정을 지원합니다.

- **Technical Details**: 모델은 히스토로지 이미지(histology images), 차세대 시퀀싱(next generation sequencing, NGS) 데이터, 인구 통계학적 정보 및 임상 기록을 포함한 여러 유형의 데이터를 통합합니다. 이 연구에서 개발된 다중 모달 모델은 다기관 데이터셋(multi-institutional dataset)에서 c-index 0.82를 기록하며, 이는 단일 모달 모델의 성능(c-index 0.75 및 0.77)보다 우수한 결과입니다.

- **Performance Highlights**: 연구 결과, 다중 모달 머신러닝 모델은 치료 저항성을 예측하는 데 있어 단일 모달 모델보다 더 뛰어난 성능을 보였습니다. 이는 여러 데이터 모달리티를 결합함으로써 환자 결과 예측에 있어 보다 정밀한 조치를 가능하게 한다는 점에서 중요합니다.



### Learning a Canonical Basis of Human Preferences from Binary Ratings (https://arxiv.org/abs/2503.24150)
Comments:
          25 pages, 11 figures

- **What's New**: 최근 생성 AI의 발전은 인간 피드백 강화 학습(reinforcement learning from human feedback, RLHF)과 같은 정렬 기법(alignment techniques) 덕분에 이루어졌습니다. 본 논문에서는 RLHF와 관련 기법이 어떻게 인간의 선호도를 이해하고 이러한 선호도를 바탕으로 생성모델을 조정하는지에 대한 내용을 다룹니다. 연구 결과, 약 5,000개의 선호도 중 21개의 핵심 선호 카테고리가 개인 간의 선호도 변화를 89% 이상 포착하고 있다는 것을 발견하였습니다.

- **Technical Details**: 이 연구에서 사용된 방법론은 기존의 이진 선택지 데이터셋을 기반으로 하여 인간의 암묵적인 선호 카테고리를 발견하는 데 초점을 맞췄습니다. 우리는 Chatbot Arena 데이터셋을 사용하고, 데이터셋에서 이진 선택의 이유를 도출하기 위해 GPT-4o를 활용하여 선호와 주제를 추출합니다. 이 결과, 4,469개의 고유한 선호와 3,012개의 고유한 주제가 도출되었으며, 이후 클러스터링(clustering)을 통해 21개의 선호와 주제로 최종 필터링하였습니다.

- **Performance Highlights**: 우리가 발견한 21개의 선호 카테고리는 89% 이상의 개인 간 선호도 변화를 설명할 수 있으며, 이는 인간의 심리학이나 얼굴 인식 연구에서의 핵심적인 발견과 유사합니다. 또한, 발견된 선호 집합은 모델 평가 및 훈련에 유용하게 활용되어, 특정 주제 또는 개인 사용자의 요구에 따라 더 나은 모델 정렬을 이끌어냅니다. 이 연구는 선호 데이터를 통해 생성 모델을 효과적으로 조정하는 방법을 제시하며, 관련 데이터와 코드는 GitHub에서 공개되었습니다.



### Resonance: Drawing from Memories to Imagine Positive Futures through AI-Augmented Journaling (https://arxiv.org/abs/2503.24145)
Comments:
          17 pages, 13 figures

- **What's New**: Resonance는 사용자의 과거 기억을 바탕으로 미래 활동에 대한 행동 지향적인 제안을 제공하는 AI 기반 저널링 도구입니다. 이 도구는 사용자가 새로운 기억을 기록할 때마다 제안을 제공하며, 사용자가 제안을 상상하도록 유도합니다. 랜덤화된 통제 연구를 통해 Resonance의 사용이 우울증을 줄이고 긍정적인 정서를 증진하는 데 효과적이라는 것을 발견했습니다.

- **Technical Details**: Resonance는 사용자의 저널 항목을 입력받아 대형 언어 모델(LLM)을 이용하여 개인화된 제안을 생성합니다. 해당 제안은 사용자의 긍정적인 기억을 참조하여 감정적인 반응을 유도합니다. 연구를 통해, 제안의 개인화 및 참신성이 사용자의 정서적 상태에 영향을 미친다는 것을 확인했습니다.

- **Performance Highlights**: 연구 결과, PHQ8 점수가 2주 후에 유의미하게 감소한 것으로 나타났습니다. 사용자가 제안을 실행할 가능성이 높을수록 일상적인 긍정적인 정서가 증가했습니다. 그러나 일부 사용자들은 프라이버시와 어려운 사건에 대한 상상의 어려움에 대한 우려를 표명했습니다.



### Graph Neural Network-Based Predictive Modeling for Robotic Plaster Printing (https://arxiv.org/abs/2503.24130)
- **What's New**: 이 논문은 로봇 팔을 사용한 파티클 기반 제작 공정에서 생성되는 표면을 예측하기 위해 그래프 신경망(Graph Neural Network, GNN) 모델링 접근 방식을 제안합니다. 이 접근 방식은 벽면에 시멘트 플라스터를 스프레이 방식으로 인쇄하는 과정과 관련이 있습니다. GNN 모델은 인코더-프로세서-디코더 아키텍처로 구성되어 있으며, 이를 통해 로봇 팔의 이동 경로, 속도 및 방향과 같은 특징을 활용해서 예측을 수행합니다.

- **Technical Details**: 제안된 GNN 모델은 실험 데이터를 사용하여 훈련되며, 베이지안 최적화 방법을 통해 하이퍼파라미터가 최적화됩니다. 이 모델의 주요 목표는 인쇄 프로세스의 시뮬레이터 역할을 하며, 최종적으로는 로봇 팔의 이동 경로를 생성하고 인쇄 매개변수를 최적화하여 자율 플라스터링 프로세스를 실현하는 것입니다. 또한, 이 모델은 예측 오류를 측정하여 기존의 벤치마크 모델과 비교할 때 성능이 크게 향상된 것을 보여줍니다.

- **Performance Highlights**: 제안된 모델은 예측 오류 측면에서 기존 벤치마크 모델과 비교할 때 상당한 개선을 나타냅니다. 특히, 다양한 시나리오에서 일반성을 보여주며 과거 데이터와 비교 시 예측 단계에서 향상된 오류 축척을 지닙니다. 이 모델은 하루에 최대 200 m²의 표면을 처리하며, 자재 사용량을 최대 20%까지 절감할 수 있는 효율적인 성능을 보여줍니다.



### PolypSegTrack: Unified Foundation Model for Colonoscopy Video Analysis (https://arxiv.org/abs/2503.24108)
- **What's New**: 이 논문에서는 colonoscopic 비디오에서 폴립(polyp)의 감지, 분할(segmentation), 분류(classification) 및 비지도 추적(unsupervised tracking)을 동시에 수행할 수 있는 새로운 기초 모델인 PolypSegTrack을 제안합니다. 기존의 방법들은 각각의 작업(Task)별로 특정한 미세 조정(fine-tuning)이 필요하거나, 추적 기능이 부족하거나, 도메인 특화된 사전 훈련(pre-training)에 의존하고 있었습니다. 제안된 방법은 새로운 조건부 마스크 손실(conditional mask loss)을 활용하여 데이터셋 사이의 유연한 훈련을 가능하게 하며, 이로 인해 특정 작업에 대한 미세 조정을 생략할 수 있게 됩니다.

- **Technical Details**: 제안한 PolypSegTrack 모델은 두 단계로 구성되어 있으며, 첫 번째 단계에서는 각 프레임의 바운딩 박스(bounding boxes), 세그멘테이션 마스크(segmentation masks), 클래스 확률(class probabilities)을 생성합니다. 두 번째 단계에서는 연속된 두 프레임 간의 객체를 매칭하여 추적을 수행합니다. 이 과정에서 비지도 및 비휴리스틱(Non-heuristic) 추적 방법이 사용되며, 객체 쿼리(object queries)를 통해 폴립 인스턴스를 신뢰성 있게 연결합니다. 모델은 자연 이미지에서 비지도(pre-trained on natural images)로 사전 훈련되어 도메인 특화된 데이터에 대한 의존도를 줄이고 있습니다.

- **Performance Highlights**: PolypSegTrack 모델은 ETIS, CVC-ColonDB, CVC-300, Kvasir-SEG 및 CVC-Clinic-DB 데이터셋으로 수행한 다양한 테스트에서 기존의 최첨단 방법들보다 월등한 성과를 거두었습니다. 모델은 검출(detection), 분할(segmentation), 분류(classification), 그리고 추적(tracking) 작업에서 모두 뛰어난 결과를 보여주었습니다. 이 모델은 폴립 진단의 속도와 정확성 및 일관성을 획기적으로 향상시킬 수 있는 잠재력을 가지고 있습니다.



### Artificial Conversations, Real Results: Fostering Language Detection with Synthetic Data (https://arxiv.org/abs/2503.24062)
- **What's New**: 이 연구는 고품질 훈련 데이터를 수집하는 대신 LLMs를 사용하여 생성된 합성 데이터의 가능성을 탐구합니다. 특히, 이탈리아어 직업 광고에서 포괄적 언어 감지를 위한 작업에 초점을 맞추어, 데이터 부족 문제를 해결하기 위해 합성 데이터 생성을 위한 파이프라인을 제안하고 효과적인 프롬프트 전략과 텍스트 길이, 특수 과제에서의 목표 위치 같은 요소가 모델 성능에 미치는 영향을 분석합니다.

- **Technical Details**: 연구 방법론은 LLM 기반 합성 데이터 생성을 위한 프레임워크를 수립하고, 이탈리아어 직업 광고에서 비포괄적 언어를 탐지하는 데 활용됩니다. 이 방식은 실제 및 생성된 데이터를 결합한 합성 데이터셋 생성, 다양한 프롬프트 기법의 적용, 합성 데이터에 대한 모델의 파인 튜닝을 포함합니다. 데이터셋은 70-30 비율로 훈련 및 평가용으로 분할되어 모델의 일반화 능력을 검증합니다.

- **Performance Highlights**: 실험 결과, 합성 데이터로 훈련된 세밀화 모델이 실제 및 합성 테스트 데이터 세트에서 다른 모델보다 일관되게 우수한 성능을 보였습니다. 이는 LLMs의 합성 데이터 활용이 비용 효율적이고 확장 가능한 솔루션임을 보여줍니다. 본 연구는 이러한 합성 데이터 사용의 실질적 임팩트와 한계를 논의하면서, 포괄적 언어 탐지 작업에 대한 새로운 패러다임을 제시합니다.



### Bayesian Predictive Coding (https://arxiv.org/abs/2503.24016)
- **What's New**: 본 논문에서는 예측 코딩(Predictive Coding, PC)의 확장을 제안하여 신경망 매개변수에 대한 사후 분포를 추정하는 방법인 Bayesian Predictive coding (BPC)을 소개합니다. 이 접근 방식은 PC의 지역성을 유지하면서도 Hebbian 가중치 업데이트를 위한 닫힌 형태의 업데이트를 제공합니다. BPC는 PC에 비해 전체 배치 환경에서 더 적은 에포크(epoch)로 수렴하며, 미니 배치(mini-batch) 환경에서도 경쟁력을 유지합니다. 이 방법은 또한 불확실성(uncertainty) 정량화에 있어서 기존 Bayesian 딥러닝 방법들과 유사한 성능을 제공합니다.

- **Technical Details**: BPC는 L개의 변수 계층을 가진 계층적 가우시안 생성 모델을 역전송하는 알고리즘입니다. 이 방법은 매개변수와 신경 활동을 결합하여 사용하며, 닫힌 형태의 업데이트 규칙을 통해 신경망의 매개변수에 대한 추정치를 제공합니다. 기존 PC에서는 최대 사후 추정치(MAXIMUM A POSTERIORI, MAP)와 최대 가능도 추정치(MAXIMUM LIKELIHOOD, ML)를 사용하지만, BPC에서는 근사 사후 분포를 나타내어 매개변수를 업데이트합니다. 이러한 방식은 각종 최적화 문제에서 기댓값-최대화(Expectation-Maximization, EM) 알고리즘을 사용하여 해결됩니다.

- **Performance Highlights**: BPC는 실험을 통해 PC와 전통적인 역전파(Backpropagation, BP) 알고리즘과 비교하여 유사한 성능을 나타내며, 특히 전체 배치 훈련에서 놀라울 정도로 적은 에포크로 수렴한다는 점이 주목할 만합니다. BPC의 학습된 사후 분포는 에피스템적 불확실성과 우연적 불확실성을 강력하게 정량화할 수 있으며, Bayesian 딥러닝에서 인기 있는 벤치마크와 비교해도 개선된 불확실성 정량화와 정확도를 제공합니다. 종합적으로 BPC는 불확실성을 인식하는 신경망 훈련을 위한 유효한 방법으로 제안됩니다.



### Learning 3D-Gaussian Simulators from RGB Videos (https://arxiv.org/abs/2503.24009)
- **What's New**: 이 논문에서는 3D 물리 시뮬레이터인 3DGSim을 소개하며, 이는 다중 뷰 RGB 비디오에서 물체의 동역학을 끝에서 끝까지 학습하는 혁신적인 접근을 제공합니다. 3DGSim은 이미지를 3D Gaussiann (Gaussian) 파티클 표현으로 인코딩하고, 이러한 동역학을 transformer를 통해 전파하며, 3D Gaussian splatting을 사용하여 프레임을 렌더링합니다. 이 방법은 물리적 특성을 포인트-와이즈(latent vectors) 잠재 벡터에 포함시키면서, 명시적인 연결 제약 조건을 두지 않고도 다양한 물리적 행동을 캡처할 수 있도록 합니다.

- **Technical Details**: 3DGSim은 RGB 비디오에서 입자 상호작용을 직접 학습하여 3D Gaussian 포인트 클라우드로 장면을 표현합니다. kNN 대신 시간적 포인트 클라우드 직렬화(temporal point cloud serialization)를 활용하여 모델의 확장성을 크게 향상시킵니다. 동역학 모델과 함께 역 물리 렌더링을 공동으로 훈련하며, 이는 모션 사전(prior)을 3D Gaussian 표현에 직접 포함시키고 렌더링 직전에만 이 잠재 벡터를 splats로 맵핑합니다.

- **Performance Highlights**: 3DGSim은 다양한 물리적 행동을 포착할 수 있으며, 이는 강체(rigid), 탄성(elastic), 천과 같은 상호작용을 포함합니다. 또한 현실적인 조명 효과를 생성하며, 보지 못했던 다중체 상호작용과 새로운 장면 수정에 대해 일반화되는 것을 가능하게 합니다. 이로 인해, 3DGSim은 로봇의 의사결정에서 신뢰성을 높이며, 물리적 정확성을 강화하는 데에 기여할 가능성이 큽니다.



### H2VU-Benchmark: A Comprehensive Benchmark for Hierarchical Holistic Video Understanding (https://arxiv.org/abs/2503.24008)
- **What's New**: 이번 논문에서는 비디오 이해 능력을 평가하기 위한 새로운 H2VU 벤치마크를 제안합니다. 현재의 벤치마크가 지닌 한계를 극복하고자 짧은 비디오와 1.5시간에 걸친 긴 레코딩을 포함한 다양한 비디오 길이를 평가할 수 있습니다. 그뿐만 아니라, 새로운 획기적인 평가 모듈인 'Counterfactual Reasoning'과 'Trajectory State Tracking'을 통해 단순한 지식 기반을 넘어서는 모델의 깊은 이해 능력을 검증합니다.

- **Technical Details**: H2VU-Benchmark는 비디오 이해 모델 평가를 위해 총 10,183개의 평가 작업을 포함하는 3단계 계층적 역량 분류 시스템을 개발했습니다. 일반 오프라인 비디오와 온라인 스트리밍 비디오라는 두 가지 주요 영역을 통해 평가를 진행하며, 각기 다른 유형의 인식 및 추론 작업을 포함합니다. 또한, 이를 통해 비디오 이해 모델의 동적 장면 이해 능력을 심도 있게 평가할 수 있습니다.

- **Performance Highlights**: H2VU의 실험 결과, 기존의 멀티모달 대형 언어 모델(MLLMs)이 새로운 평가 작업에서 상당한 개선 가능성을 지니고 있다는 것을 보여줍니다. 특히, 속임수를 활용한 이해(task)와 상태 궤적 추적(task)에서 모델의 성능 차이가 두드러졌습니다. 이 결과들은 현재 모델들이 비디오 콘텐츠 기반의 응답을 효과적으로 생성하기 위해서는 더 많은 개선이 필요하다는 것을 시사하며, 실제 세계의 비디오 이해 적용에서 여전히 도전 과제가 남아 있음을 강조합니다.



### CITRAS: Covariate-Informed Transformer for Time Series Forecasting (https://arxiv.org/abs/2503.24007)
- **What's New**: 이 논문에서는 CITRAS라는 새로운 Transformer 모델을 제안합니다. CITRAS는 시계열 예측을 위한 패치 기반 모델로, 과거와 미래를 아우르는 여러 목표 변수 및 공변량을 유연하게 활용합니다. 이 모델은 기존 Transformer의 자기 회귀(autoregressive) 특성을 유지하면서, KV Shift 및 Attention Score Smoothing이라는 두 가지 새로운 메커니즘을 도입합니다. 이를 통해 복잡한 공변량 간 의존성을 효과적으로 캡처하고 향상된 예측 정확도를 제공합니다.

- **Technical Details**: CITRAS는 공변량 정보를 담은 예측을 위해 Cross-시간 주의 모듈과 Cross-공변량 주의 모듈을 별도로 운영합니다. 주의 모듈 내에서, Attention Score Smoothing은 지역적으로 정확한 패치 간의 관계를 글로벌 공변량 수준으로 변환합니다. KV Shift 메커니즘은 미래의 알려진 공변량을 예측 과정에 통합하여, 공변량 간의 동시 의존성을 기반으로 예측을 수행합니다. 이러한 기술적 노하우는 기존 Transformer의 강력한 자기 회귀 성질을 완전히 보존합니다.

- **Performance Highlights**: CITRAS는 공변량 정보를 활용한 예측 및 다변량 예측(settings)에서 최첨단 성능을 달성했습니다. 실험 결과는 CITRAS가 복잡한 공변량 간 상관관계를 활용하여 예측 정확도를 향상시킬 수 있는 다재다능한 능력을 보여줍니다. 특히, 이 모델은 다양한 상황의 공변량을 효과적으로 활용하여 예측의 신뢰성을 높입니다. 결과적으로 CITRAS는 시계열 데이터의 다양한 변수 간의 복잡한 의존성을 포착하며, 미래의 알려진 공변량을 효과적으로 활용합니다.



### Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving (https://arxiv.org/abs/2503.24000)
Comments:
          21 pages, 18 figures, published to MLSys2025

- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 성능 향상을 위해 Key-Value 캐시(KV cache) 압축 기술을 다루고 있습니다. 최근 LLM에 대한 수요가 급증함에 따라 효율적인 추론 최적화 알고리즘이 필요해졌습니다. 연구자들은 KV 캐시가 LLM 서비스의 주요 성능 병목 현상임을 지적하고 있으며, 이를 해결하기 위한 방법론을 논의하고 있습니다.

- **Technical Details**: KV 캐시 압축에는 두 가지 주요 기술이 있습니다: 첫 번째는 양자화(quantization) 기반 방법으로, 이는 KV 캐시를 저정밀 표현으로 변환하여 GPU 메모리 사용량을 줄이는 기술입니다. 두 번째는 희소성(sparsity) 기반 방법으로, 중요하지 않은 KV 캐시 항목을 메모리에서 제거하거나 저속 메모리로 이동시키는 방식을 유도합니다. 이러한 방법들이 실질적으로 배포될 수 있는지는 여전히 불확실합니다.

- **Performance Highlights**: 논문은 KV 캐시 압축 알고리즘이 추론 성능을 향상시킬 수 있지만, 특정 배치 크기와 프롬프트 길이에서 성능 저하를 보일 수 있음을 보여줍니다. 또한, KV 캐시 압축으로 인해 생성되는 응답 길이가 길어질 수 있으며, 이는 최종 대기 시간을 증가시킬 수 있습니다. 연구팀은 이러한 문제를 해결하기 위한 여러 도구를 제공하여 KV 캐시 압축의 실제 배포를 용이하게 하려 합니다.



### DenseFormer: Learning Dense Depth Map from Sparse Depth and Image via Conditional Diffusion Mod (https://arxiv.org/abs/2503.23993)
- **What's New**: 본 논문에서는 자율주행에서 필수적인 깊이 완성(depth completion) 작업을 위한 새로운 방법인 DenseFormer를 제안합니다. DenseFormer는 전통적인 방식의 공간 전파 네트워크(spatial propagation network) 대신에 전이(diffusion) 모델을 통합하여 복잡한 깊이 맵을 생성합니다. 특히, 이 방법은 초기 랜덤 깊이 분포를 여러 번의 반복을 통해 점진적으로 개선하여 밀도 높은 깊이 맵을 생성하는 데 초점을 맞추고 있습니다.

- **Technical Details**: DenseFormer는 특징 추출 모듈(feature extraction module)과 깊이 개선 모듈(depth refinement module)을 포함하고 있습니다. 특징 추출 모듈은 다층 변형(attention) 구조를 활용하여 희소 깊이 맵(sparse depth maps)과 RGB 이미지에서 효과적으로 특징을 추출하고 통합합니다. 또한, 깊이 개선 모듈은 전이 과정을 통해 생성된 깊이 결과에 대해 다양한 범위에서 다단계 반복 개선(multi-step iterative refinement)을 적용하여 더욱 향상된 정확성을 제공합니다.

- **Performance Highlights**: KITTI 야외 장면 데이터셋에서 진행된 포괄적인 실험을 통해 DenseFormer가 기존의 클래식 깊이 완성 방법들보다 우수한 성능을 보임을 입증하였습니다. 이 연구 결과는 자율주행 기술에 있어 깊이 정보를 정확하게 생성하는 데 기여할 것으로 기대됩니다.



### Rubric Is All You Need: Enhancing LLM-based Code Evaluation With Question-Specific Rubrics (https://arxiv.org/abs/2503.23989)
Comments:
          Under Review

- **What's New**: 이번 논문에서는 LLM(대형 언어 모델)을 기반으로 한 코드 평가에 중점을 두고, 질문 특정 루브릭(question-specific rubric)을 활용하여 기존의 접근법에 비해 논리적 평가에서 더 나은 성능을 보인다고 주장합니다. 특히, 데이터 구조와 알고리즘, 객체 지향 프로그래밍 과목의 학생 제출물을 활용하여 새로운 데이터셋을 제안합니다. 또한, Leniency라는 새로운 평가 지표를 도입하여 전문가 평가에 비례한 평가 엄격성을 정량화하고 있습니다.

- **Technical Details**: 저자들은 코드 평가를 위해 세 가지 새로운 기법을 제안합니다: (1) Complete Rubric Evaluation (CRE)은 전체 루브릭을 기준으로 학생 제출물을 평가하여 개념적 이해에 중점을 둡니다. (2) Pointwise Rubric Evaluation (PRE)은 루브릭의 각 기준을 개별적으로 확인하여 세밀한 피드백을 제공합니다. (3) Ensembling Method Evaluation (EME)은 다수결 집계 방식을 통해 평가의 신뢰성을 높입니다.

- **Performance Highlights**: 연구 결과, 질문 특정 루브릭이 질문 무관 루브릭보다 성능이 우수하다는 것을 확인하였으며, 이는 코드의 정확성, 피드백의 관련성, 교육 목표와의 일치성을 개선했습니다. 이는 교육적 맥락에서 학생들에게 더 나은 피드백을 제공함으로써 프로그래밍 개념의 이해를 심화할 수 있는 가능성을 보여줍니다. 또한, 새로운 데이터셋과 기법은 향후 LLM 기반 평가 도구의 발전에도 기여할 것입니다.



### Deep Learning Model Deployment in Multiple Cloud Providers: an Exploratory Study Using Low Computing Power Environments (https://arxiv.org/abs/2503.23988)
Comments:
          15 pages, 7 figures

- **What's New**: 이 논문은 딥러닝(Deep Learning) 모델을 클라우드에서 배포하는 데 있어 비용 효율성과 실행 가능성을 평가하였다. 특히, GECToR 모델을 사용하여 AWS, Google Cloud, Azure 세 가지 주요 클라우드 플랫폼에서의 성능과 비용을 비교 분석하였다. 연구 결과 GPU 사용이 성능에서 우수하지만 평균 비용이 300% 더 높다는 결과를 도출하였다.

- **Technical Details**: 제안된 연구는 10개의 실험을 통해 7개의 실행 환경에서의 실시간 지연(latency), 하드웨어 사용 및 비용을 평가하였다. 이 연구는 CPU의 프로세서 캐시 크기가 비용 효율적 배포를 위한 중요한 요소라고 강조하며, 이를 통해 GPU 대비 50% 이상의 비용 절감이 가능함을 발견하였다. 또한, MLaaS(머신러닝 서비스) 플랫폼의 다양한 하드웨어 구성에서 딥러닝 모델을 실행하기 위한 최적의 옵션을 식별하였다.

- **Performance Highlights**: 클라우드 기반의 딥러닝 추론 솔루션은 자원 제약이 있는 사용자, 예를 들어 스타트업들에게 유리하다는 점을 밝히고 있다. 실험 결과, GEC 모델이 처리된 데이터의 정확도와 속도가 높지만, 높은 하드웨어 요구사항이 가격에 미치는 영향을 최소화하기 위해 최적의 하드웨어 구성을 선택할 필요가 있다. 이러한 연구는 특히 개발도상국의 기업들이 혁신을 위해 여전히 필요로 하는 기술적 장벽을 낮추는 데 기여할 것으로 기대된다.



### Deep Nets as Hamiltonians (https://arxiv.org/abs/2503.23982)
Comments:
          19+7 pages

- **What's New**: 이번 논문에서는 랜덤 초기화된 Multi-Layer Perceptron (MLP)을 Hamiltonian으로 간주하고, 이 Hamiltonian이 유도하는 에너지 풍경의 특성을 연구합니다. 특히 무한 너비의 한계에서 거의 전역 최솟값의 구조에 초점을 맞추며, Replica Trick을 사용하여 주어진 에너지에서의 엔트로피를 정확하게 계산합니다. 랜덤 MLP로부터 유도된 Gibbs 분포를 통해 입력 간의 겹침을 설명하는 saddle point 방정식도 도출합니다.

- **Technical Details**: 이 연구는 깊은 신경망의 이론적 분석을 위해 파라미터가 무작위로 초기화된 경우의 결과를 살펴보는 기존의 접근과는 반대로, 입력을 anneal하고 파라미터를 quenched 상태로 고려합니다. 또한, linear activation function을 포함한 다양한 비선형 활성화 함수에 대한 saddle point 방정식을 수치적으로 및 정확히 해결합니다. 논문은 MLP의 너비가 무한대에 가까울 때의 Gibbs 측정을 탐구합니다.

- **Performance Highlights**: MLP의 활성화 함수에 따라 매우 다양한 행동 양상을 발견하였으며, 예를 들어 비선형성 중 하나인 sin의 경우에는 전체 replica symmetry breaking을 보였습니다. 반면, shallow tanh 및 ReLU 네트워크나 깊은 형태의 MLP에서는 replica symmetry를 유지하는 결과가 나타났습니다. 이러한 결과는 모델의 구조와 성능 간의 관계를 보다 깊이 이해하는 데 기여할 것입니다.



### Noise-based reward-modulated learning (https://arxiv.org/abs/2503.23972)
- **What's New**: 최근 심화 학습(Reinforcement Learning, RL) 분야의 발전은 작업 성능에서 큰 개선을 가져왔습니다. 그러나, RL 환경에서 신경망을 훈련하는 것은 일반적으로 역전파(Backpropagation)와 결합되어 이루어지므로 자원 제약이 있는 환경이나 비미분 가능한 신경망에서는 적용이 제한됩니다. 본 논문에서는 지연 보상 시나리오에서의 한계를 해결하기 위해 새로운 노이즈 기반 학습 규칙을 도출하였으며, 이는 방향 미분 이론과 Hebbian 유사 업데이트를 결합하여 RL에서의 효율적이고 기울기 없는 학습을 가능하게 합니다.

- **Technical Details**: 제안된 학습 메커니즘은 스토캐스틱 노이즈 뉴런을 활용하여 기울기를 근사하는 구조입니다. 이 방법은 보상 예측 오류(Reward Prediction Error, RPE)를 최적화 목표로 삼고, 지연 보상이 있는 환경에서 과거 신경 상태와 미래 보상을 연결하는 적격성 추적(Eligibility Trace)을 통합하여 시간적 신뢰 할당을 용이하게 합니다. 노이즈 없는 전달이 불가능한 경우 여러 개의 노이즈 있는 전달을 평균하여 성능을 유지하는 방식으로 기획하였습니다.

- **Performance Highlights**: 실험 결과, 제안한 노이즈 기반 보상 조정 학습(Noise-based Reward-Modulated Learning, NRL) 방법이 RMHL보다 유의미하게 우수한 성능을 보였으며, 역전파 기반 기준선들과도 경쟁력을 유지하였습니다. 이는 저전력 및 실시간 애플리케이션을 위한 노이즈 기반 비유적 학습의 가능성을 강조합니다. 본 연구는 머신러닝, 신경형 컴퓨팅(Neuromorphic Computing), 신경과학(Neuroscience) 분야에서의 넓은 함축적 의미에 대해서도 논의하고 있습니다.



### AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inferenc (https://arxiv.org/abs/2503.23956)
- **What's New**: 이 논문은 최근의 대규모 시각 언어 모델(LVLMs)의 발전과 이들이 제기하는 계산적 문제를 해결하기 위한 새로운 방법인 AirCache를 제안합니다. AirCache는 KV(cache) 압축 기법으로, 시각 토큰 간의 중복성을 줄여 모델의 성능을 유지하면서도 추론 속도를 향상시키는 데 중점을 두고 있습니다. 특히, 중요도가 높은 시각 구성 요소를 선택하기 위한 엘리트 관찰 창(elite observation window)을 도입하여 다양한 이점을 제공합니다.

- **Technical Details**: AirCache의 주요 구성 요소는 시각 토큰의 중요도 점수화와 레이어별 KV cache 예산 할당입니다. 이 기법은 스스로 어텐션 점수를 활용하여 시각 토큰의 중요도를 평가하고, 이를 기반으로 압축 예산을 레이어별로 차별화하여 최적의 성능을 도출합니다. 논문에서는 이 기술을 통해 단순히 토큰 수를 줄이는 것이 아니라, 중요한 정보가 포함된 10%의 시각 KV cache만 유지하여도 모델 성능에 미미한 영향을 미친다고 보고합니다.

- **Performance Highlights**: 종합적인 실험 결과, AirCache는 여러 LVLM 및 벤치마크 데이터 세트에서 기존 방법들에 비해 현저히 개선된 성능을 보여줍니다. 이 방법을 통해 KV cache를 10%만 유지하면서도 디코딩 대기 시간을 29%에서 66%까지 줄일 수 있었으며, 캐시 유지 비율이 감소할수록 기존 방법들보다 성능적으로 우위에 있음을 입증했습니다.



### Green MLOps to Green GenOps: An Empirical Study of Energy Consumption in Discriminative and Generative AI Operations (https://arxiv.org/abs/2503.23934)
Comments:
          Published to MDPI Information - Artificial Intelligence Section

- **What's New**: 이 연구는 실제 MLOps 파이프라인에서 판별(Discriminative) 및 생성(Generative) AI 모델의 에너지 소비를 조사합니다. 연구 결과, 판별 모델의 경우 아키텍처와 하이퍼파라미터를 최적화함으로써 성능을 유지하면서 에너지를 줄일 수 있는 방법을 제시합니다. 생성 AI에서는 대규모 언어 모델(LLMs) 사이의 에너지 소비를 비교 분석하여 모델 크기와 요청 처리 용량 간의 균형이 중요함을 강조합니다.

- **Technical Details**: 연구는 소프트웨어 기반 전력 측정을 통해 다양한 설정, 모델 및 데이터셋에서의 에너지 소비 패턴을 분석합니다. 판별 AI의 경우 훈련 및 추론 동안 서로 다른 모델 아키텍처와 하이퍼파라미터가 에너지 소비에 미치는 영향을 평가하며, 생성 AI는 다양한 요청 및 토큰 사용을 통한 에너지 소비를 집중 분석합니다. 이러한 측정 방법은 다양한 환경에서 복제할 수 있도록 설계되었습니다.

- **Performance Highlights**: 결과적으로, 판별 모델의 경우 아키텍처와 하드웨어의 최적화가 에너지 소비를 크게 줄일 수 있음을 발견했습니다. 또한, 대규모 언어 모델의 에너지 효율성은 모델 크기, 추론 복잡성 및 요청 처리 용량 간의 균형에 따라 달라진다는 사실도 확인했습니다. 이러한 발견들은 ML 운영(MLOps)에서 지속 가능성을 높이는 데 필요한 실제 지침을 제시합니다.



### HumanAesExpert: Advancing a Multi-Modality Foundation Model for Human Image Aesthetic Assessmen (https://arxiv.org/abs/2503.23907)
- **What's New**: 이번 연구에서는 Human Image Aesthetic Assessment (HIAA)를 위한 혁신적인 구현 프레임워크를 제시하며, HIAA를 위해 처음으로 설계된 HumanBeauty 데이터셋을 도입했습니다. 이 데이터셋은 108,000개의 고퀄리티 인간 이미지를 포함하고 있으며 수작업으로 주석이 붙여져 있습니다. HIAA의 개선된 평가를 위해 Vision Language Model (VLM) 기반의 HumanAesExpert 모델을 제안하였습니다.

- **Technical Details**: HumanAesExpert는 12차원의 미적 기준을 활용하여 HIAA의 세부 측면을 평가하는 독창적인 Expert Head를 도입합니다. 이 모델은 Language Modeling (LM) Head와 Regression Head를 함께 사용하여 정확하고 세밀한 평가를 가능하게 합니다. 평가의 정밀도를 높이기 위해 세 가지 헤드의 점수를 집계하는 MetaVoter를 설계하였습니다.

- **Performance Highlights**: HumanAesExpert 모델은 HIAA 과제에서 기존 최첨단 모델들과 비교할 때 월등히 우수한 성능을 보여주었습니다. 연구진은 방대한 실험을 통해 HIAA 전반에 걸쳐 SOTA 성능을 달성함을 입증하였으며, 차세대 HIAA 커뮤니티의 발전을 위해 데이터셋, 모델 및 코드 공개를 계획하고 있습니다.



### Training-Free Text-Guided Image Editing with Visual Autoregressive Mod (https://arxiv.org/abs/2503.23897)
- **What's New**: 이번 논문에서는 VAR(Visual AutoRegressive) 기반의 새로운 텍스트 유도 이미지 편집 프레임워크를 제안하고 있습니다. 이 프레임워크는 명시적 역전(inversion) 과정 없이도 정밀하고 통제된 수정이 가능한 방법론을 제공합니다. 특히, 원본 이미지의 토큰 인덱스와 확률 분포를 저장하는 캐싱 메커니즘을 도입하여 텍스트 프롬프트와 이미지 간의 관계를 포착합니다.

- **Technical Details**: VAR 기반의 모델은 visual tokenizer와 transformer를 포함하여 이미지를 인코딩하고 합성하는 과정을 수행합니다. 이 과정에서 원본 이미지를 연속적(feature map)으로 인코딩한 후, 다중 척도의 이산적인 잔여 맵(residual maps)으로 양자화합니다. 또한, 적응형 미세 마스킹 전략을 통해 적절한 지역의 수정을 식별하여 원하지 않는 변경을 방지합니다.

- **Performance Highlights**: 제안된 AREdit 프레임워크는 훈련 없이 고충실도의 편집을 가능하게 하며, 기존의 확산(diffusion) 및 정정 흐름(rectified flow) 기반의 접근법과 비교하여 유사하거나 그 이상의 성능을 보여줍니다. 실제 테스트 결과는 1K 해상도의 이미지를 단 1.2초 만에 처리할 수 있는 속도를 자랑하며, 다양한 정량적 지표와 시각적 품질 면에서도 우수한 결과를 보였습니다.



### Better wit than wealth: Dynamic Parametric Retrieval Augmented Generation for Test-time Knowledge Enhancemen (https://arxiv.org/abs/2503.23895)
Comments:
          preprint

- **What's New**: 본 논문에서는 Dynamic Parametric RAG (DyPRAG)이라는 새로운 프레임워크를 제안하여 외부 문서를 효율적으로 파라미터로 변환함으로써 LLMs의 지식을 동적으로 향상시킵니다. DyPRAG는 파라미터 전환 모델을 활용하여 기존의 방법에 비해 훈련 및 저장 비용을 대폭 줄입니다. 또한, 이 프레임워크는 RAG 환각 문제를 완화하고, 모델의 지식 충돌을 해결하는 방식으로 실시간 성능 향상을 가능하게 합니다.

- **Technical Details**: DyPRAG는 최신 파라미터 변환 모델을 기반으로 하여 외부 문서를 LLM의 파라미터로 직접 변환하는 방식을 채택합니다. 이 방식은 테스트 시간에 필요한 지식 형성을 즉각적으로 수행하며, 이를 통해 전통적인 RAG 방식에 비해 더 낮은 추론 비용과 훈련 비용을 자랑합니다. 이는 효율적인 지식 주입을 통해 LLM의 성능을 극대화하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, DyPRAG는 다양한 데이터셋에서 기존 RAG 방법들을 능가하는 성능을 보였습니다. 특히 테스트 시간에 생성된 파라미터 지식과 맥락 지식을 결합함으로써 지식 융합에서 우수한 성과를 발휘하며, RAG 환각 문제를 효과적으로 완화하는 데 성공했습니다. DyPRAG-Combine 방법을 통해 높은 일반화 능력을 보여주며 실제 응용에서 신뢰할 수 있는 RAG 시스템 구축 가능성을 확인했습니다.



### DiffScale: Continuous Downscaling and Bias Correction of Subseasonal Wind Speed Forecasts using Diffusion Models (https://arxiv.org/abs/2503.23893)
Comments:
          28 pages, 18 figures, preprint under review

- **What's New**: 본 연구는 S2S(서브시즌-투-시즌) 예측의 정확성을 향상시키기 위한 새로운 모델인 DiffScale을 제안합니다. 이 모델은 편향 없는 가이드를 사용하는 확산 모델로, 바람 속도 예측의 다운스케일링을 통해 지역 및 대규모 날씨 상황을 효과적으로 반영할 수 있도록 합니다. DiffScale은 여러 그리드 해상도 및 전진 시간에 걸쳐 모델 오류를 수정하고 조정할 수 있는 유연성을 제공합니다.

- **Technical Details**: DiffScale은 조건부 확률(conditional probabilities)을 샘플링의 가이드로 활용하는 확산 모델입니다. 이 모델은 S2S 예측의 밀도를 직접 추정하여, 자동 회귀(auto-regression)나 시퀀스 예측 없이도 효율적이고 유연한 예측을 가능하게 합니다. 연구에서는 ECMWF(유럽 중기 일기예보 센터)의 조도 해상도(S2S) 바람 속도 예측을 ERA5 재분석 데이터의 고해상도로 다운스케일링하는 synthetic experiments를 설계하였습니다.

- **Performance Highlights**: DiffScale은 최대 3주까지 기존의 기준 성능을 초과하는 예측 품질 향상을 이뤘습니다. 이 모델은 다양한 그리드 해상도에 일반화할 수 있으며, 모델 재훈련 없이도 새로운 스케일링 요소에 대응할 수 있는 다재다능한 도구입니다. 연구 결과는 에너지 부문에 중요한 사회경제적 이점을 제공할 수 있는 가능성을 보여줍니다.



### MuseFace: Text-driven Face Editing via Diffusion-based Mask Generation Approach (https://arxiv.org/abs/2503.23888)
Comments:
          6 pages, 5 figures,IEEE International Conference on Multimedia & Expo 2025

- **What's New**: 이번 논문에서 제안된 MuseFace는 텍스트 프롬프트만을 이용하여 얼굴 편집을 가능하게 하는 텍스트 기반 얼굴 편집 프레임워크입니다. MuseFace는 텍스트에서 세분화된 마스크를 직접 생성하고 이를 기반으로 얼굴 이미지를 수정하는 두 가지 모델, 즉 Text-to-Mask diffusion 모델과 의미 인식(face editing) 모델을 통합하여 사용합니다. 이 프레임워크는 다양성(diversity)과 유연성(flexibility)뿐만 아니라 제어 가능성(controllability)을 동시에 충족하는 것을 목표로 하고 있습니다.

- **Technical Details**: MuseFace는 처음으로 텍스트 입력에 의해 다양한 위치-aware 세분화 마스크를 생성하는 방식을 도입합니다. 이를 통해 기존의 거칠고 제한적인 마스크 대신 세밀한 마스크를 생성함으로써 사용자가 편집 지점을 명확하게 제어할 수 있습니다. 이 시스템은 또한 두 가지 방식으로 활용될 수 있어, 사용자 제공 마스크와 함께 또는 마스크가 없는 모드에서 자율적으로 다양한 편집 제안을 생성할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: 광범위한 실험을 통해 MuseFace는 고충실도(high-fidelity) 성능을 발휘하며, 기존의 얼굴 편집 모델보다 월등한 성능을 입증하였습니다. 본 연구는 특화된 학습 데이터의 부족 문제를 해결하기 위해 점진적으로 진화하는 접근 방식을 제시하며, 기존의 방법과 비교했을 때 더욱 창의적인 생성 및 다양성을 확보할 수 있는 가능성을 보여줍니다.



### SchemaAgent: A Multi-Agents Framework for Generating Relational Database Schema (https://arxiv.org/abs/2503.23886)
Comments:
          19 pages, 16 figures

- **What's New**: 이 논문에서는 SchemaAgent라는 통합된 LLM(large language model) 기반의 다중 에이전트 프레임워크를 제안합니다. 이 프레임워크는 높은 품질의 데이터베이스 스키마를 자동으로 생성할 수 있는 첫 번째 시스템으로, 전문화된 역할을 에이전트에 할당하여 각자의 하위 작업을 효과적으로 협력하여 발전시킵니다. 최근 LLM의 성능을 활용하여 스키마 생성의 정확성을 높이고 오류를 줄이는 새로운 접근 방식을 제시합니다.

- **Technical Details**: SchemaAgent는 데이터베이스 스키마 생성을 위한 여섯 개의 역할로 구성됩니다. 이 시스템은 사용자 요구 분석, 개념적 데이터 모델 설계 및 논리적 데이터 모델 설계의 세 가지 하위 작업을 수행하며, 각 역할 간의 그룹 채팅(Mechanism)을 통해 오류를 신속하게 식별하고 수정할 수 있도록 합니다. 또한, 개념적 모델을 감독하고 검증하는 역할을 도입하여 오류 발생을 최소화하고, QA 엔지니어와 테스트 실행기로 구성된 추가 역할을 통해 스키마 품질을 평가합니다.

- **Performance Highlights**: RSchema라는 벤치마크를 통해 500개 이상의 요구사항과 스키마 쌍을 포함한 데이터셋을 성공적으로 구축하였습니다. 실험 결과, SchemaAgent는 기존의 체인 오브 생각(coT)이나 기본 프롬프트 방법론보다 뛰어난 성능을 보여줍니다. 이 프레임워크는 데이터베이스 스키마 생성 과정을 크게 개선하고, 오류를 효과적으로 줄이며 높은 품질의 출력을 보장합니다.



### GenSwarm: Scalable Multi-Robot Code-Policy Generation and Deployment via Language Models (https://arxiv.org/abs/2503.23875)
- **What's New**: 이번 연구는 GenSwarm라는 시스템을 소개합니다. GenSwarm는 대규모 언어 모델을 활용하여 사용자의 자연어 지시에 따라 다중 로봇 작업을 위한 제어 정책을 자동으로 생성하고 배포하는 종단간(end-to-end) 시스템입니다. 이 시스템은 제로샷 학습(zero-shot learning)을 통해 새로운 작업에 빠르게 적응할 수 있는 능력을 가지고 있습니다.

- **Technical Details**: GenSwarm는 하드웨어 및 소프트웨어 아키텍처가 확장 가능하여 시뮬레이션과 실제 다중 로봇 시스템 모두에서 효율적인 정책 배포를 지원합니다. 코드 정책의 화이트박스(white-box) 특성은 강한 재현성(reproducibility)과 해석 가능성(interpretability)을 보장합니다. 이 연구에서는 자연어로 된 간단한 지시사항을 기반으로 제어 정책을 생성하는 방법에 대한 새로운 접근 방식을 제안합니다.

- **Performance Highlights**: GenSwarm은 로봇 전문가뿐만 아니라 비전문가에게도 유용할 수 있는 실시간 명령에서 실행까지의 전체 기능을 현실화합니다. 연구진이 보고하는 바에 따르면, 이 시스템은 다양한 작업을 독립적으로 처리할 수 있는 능력을 보여주며, 다중 로봇 시스템의 제어 방안 개발에 큰 기여를 할 것으로 기대됩니다.



### Learned Image Compression and Restoration for Digital Pathology (https://arxiv.org/abs/2503.23862)
- **What's New**: 이 논문에서는 디지털 병리 이미지의 효율적 압축을 위한 새로운 딥러닝 기반 프레임워크인 CLERIC을 제안합니다. CLERIC은 전체 슬라이드 이미지(Whole Slide Images, WSIs)의 고해상도를 고려하여 설계되었으며, 병리학적 세부 사항을 유지하면서도 압축 효율성을 높이는 학습 가능한 리프팅 방식을 통합합니다. 이 프레임워크는 이미지 복원 과정에서 고품질의 조직 구조를 보장하기 위해 역 리프팅 변환을 적용합니다.

- **Technical Details**: CLERIC은 이미지 분석 단계에서 고주파 및 저주파 구성 요소로 분해하기 위해 리프팅 스킴 변환을 사용합니다. 이 과정에서 변형 잔여 블록(Deformable Residual Blocks, DRB)과 순환 잔여 블록(Recurrent Residual Blocks, R2B)을 포함한 병렬 인코더가 특징 추출 및 공간 적응성을 개선하여 복잡한 병리 이미지를 효과적으로 처리합니다. 이러한 기법은 이미지 세분화 및 초해상도 과제에서도 성공적으로 적용되었습니다.

- **Performance Highlights**: 실험 결과, CLERIC은 기존의 최첨단 학습 이미지 압축(Learned Image Compression, LIC) 모델에 비해 우수한 비율-왜곡(rate-distortion) 성능을 보여주었으며, 저장 요구량을 현저히 줄이면서도 진단 이미지의 품질을 유지함을 입증하였습니다. 또한, CLERIC은 다중 해상도 형식을 지원하여 표준 병리학적 시각화 소프트웨어와의 원활한 통합이 가능합니다.



### OrchMLLM: Orchestrate Multimodal Data with Batch Post-Balancing to Accelerate Multimodal Large Language Model Training (https://arxiv.org/abs/2503.23830)
- **What's New**: 최근 MLLMs(다중 모달 대형 언어 모델)의 주목받는 혁신으로 Modality Composition Incoherence라는 현상이 등장하였습니다. 이는 특정 모달리티의 비율이 서로 다른 예제 간에 극단적으로 변화함을 의미합니다. 이러한 문제는 미니 배치 불균형(mini-batch imbalances)을 해결하는 데 어려움을 주고 있으며, GPU 활용도를 낮추고 MLLM 훈련의 효율성과 확장성을 심각하게 저해합니다.

- **Technical Details**: 이 논문에서는 MLLM 훈련의 비효율성을 줄이기 위해 OrchMLLM이라는 포괄적인 프레임워크를 제안합니다. OrchMLLM은 Batch Post-Balancing Dispatcher 기법을 활용하여 순차적 데이터의 미니 배치 불균형을 효율적으로 제거합니다. 또한, MLLM Global Orchestrator를 통합하여 다중 모달리티 데이터를 조화롭게 처리하며 Modality Composition Incoherence 문제를 해결합니다.

- **Performance Highlights**: OrchMLLM을 2560개의 H100 GPU에서 84B MLLM을 학습하는 실험을 통해 평가한 결과, 41.6%의 모델 FLOPs Utilization(MFU)를 달성하면서 Megatron-LM보다 최대 3.1배 높은 처리량을 보였습니다. 이는 MLLM 훈련의 효율성과 확장성을显著 향상시키는 데 기여합니다.



### When Counterfactual Reasoning Fails: Chaos and Real-World Complexity (https://arxiv.org/abs/2503.23820)
- **What's New**: 이번 연구는 구조적 인과 모델(Structural Causal Models) 내에서 반사실적 추론(counterfactual reasoning)의 한계를 탐구한다. 특히, 실험적으로 반사실적 시퀀스 추정(counterfactual sequence estimation)을 조사하며, 예상치 못한 결과가 발생하는 경우를 강조한다. 연구진은 모델 불확실성(model uncertainty)이나 혼돈 동역학(chaotic dynamics)과 같은 현실적 가정을 바탕으로 예측과 실제 반사실적 경로 사이의 극심한 차이를 발견하였다.

- **Technical Details**: 반사실적 추론의 신뢰성을 평가하기 위해, 연구는 상태공간 모형(state-space model)을 활용하여 저차원 ODE(Ordinary Differential Equations) 기반의 진화를 포착하고, 구조적 인과 프레임워크를 통해 가상의 개입(hypothetical interventions)에 대한 추론을 진행했다. 연구진은 완벽한 지식이 없음에도 불구하고, 초기 미세한 변화(initial perturbations)나 파라미터의 오차가 상이한 경로를 초래할 수 있음을 보여주었다. 이로 인해 실제 상황에서 반사실적 예측의 신뢰성이 크게 저하된다는 사실이 강조되었다.

- **Performance Highlights**: 이 연구는 기존의 연구들과 달리, 매개변수 불확실성(parameter uncertainty)이 있는 혼돈 시스템에서 반사실적 추론을 수행하는 방법을 제시한다. 특히, 원본 시스템 관찰에서 반사실적 예측의 신뢰성을 바로 도출할 수 없음을 강조하며, 현장 적용을 위한 신뢰할 수 있는 반사실적 추론의 필요성을 역설한다. 이로써, 실제 시스템의 혼돈 또는 복잡한 동역학을 모델링할 때 발생하는 문제를 경고하며, 반사실적 추론의 적용에서 주의가 필요함을 일깨워준다.



### Conformal uncertainty quantification to evaluate predictive fairness of foundation AI model for skin lesion classes across patient demographics (https://arxiv.org/abs/2503.23819)
- **What's New**: 이번 연구는 피부 병변 분류에서 비전 트랜스포머(ViT) 기반의 기초 모델을 사용하여 예측의 신뢰성과 공정성을 최적화하는 것입니다. 특히, 의사의 의사결정 프로세스를 이해하고 개선하기 위해 conformal analysis를 적용하여 예측 불확실성을 정량화합니다. 이 방법은 공정성 보장을 위해 인구 집단 차이를 고려한 적응형 F1 점수 기반 샘플링을 도입합니다.

- **Technical Details**: 연구에서 제안된 방법론은 피부 병변 분류와 불확실성 예측을 위한 최신 기초 모델과 conformal prediction 기반 불확실성 정량화를 결합하는 것입니다. 클래스 불균형 문제를 해결하기 위해 F1-점수 기반의 모델 무관 동적 샘플링 알고리즘을 적용하였으며, 이는 다양한 인구 집단 간의 예측 성능을 균형 있게 유지하게 합니다.

- **Performance Highlights**: 이 연구의 성과는 피부 암 진단에 있어 신뢰할 수 있는 예측 결과를 제공하는 것입니다. 각 환자에게 대해 불확실성 점수를 제공함으로써 모델의 공정성과 신뢰성을 높이는 것을 목표로 하였습니다. 이러한 접근법은 다양한 의료 데이터셋에서도 적용 가능하며, 의사결정의 투명성을 높일 수 있는 잠재력을 지니고 있습니다.



### Thinking Longer, Not Larger: Enhancing Software Engineering Agents via Scaling Test-Time Compu (https://arxiv.org/abs/2503.23803)
- **What's New**: 이번 연구에서는 폐쇄형 소스 모델이나 자원 집약적인 모델에 기댄 기존 소프트웨어 엔지니어링 에이전트의 한계를 극복하기 위한 방법을 제안합니다. 개인이 배포할 수 있는 오픈 소스 LLM이 같은 수준의 코드 추론 성능을 얻을 수 있도록 하는 것이 이번 연구의 핵심입니다. 이를 위해 우리는 Test-Time Compute (TTC) 스케일링 프레임워크를 도입하여 더 큰 모델 대신에 향상된 추론 시간 계산을 활용하고, 내부 TTC와 외부 TTC라는 두 가지 상호 보완적인 전략을 개발합니다.

- **Technical Details**: 내부 TTC는 현실적인 멀티 단계 추론 과정과 일치하는 '개발 컨텍스트 기반 경로 합성' 방법을 도입하여 고품질의 소프트웨어 저장소로부터 데이터를 적재하고, 다양한 단계의 사고 과정을 포함하는 경로를 생성합니다. 반면, 외부 TTC는 보상 모델과 실행 검증을 통해 개발 결정의 중요한 순간에서 계산 자원을 전략적으로 배치하여 기존의 방법들에 대한 한계를 극복하는 새로운 접근 방식을 제공합니다. 이 방법은 각각의 단계에서 중간 출력을 평가하기 위한 전용 '프로세스 보상 모델'을 훈련시켜 솔루션 길이를 효과적으로 줄입니다.

- **Performance Highlights**: 우리의 연구 결과, 32B 모델이 SWE-bench Verified에서 46%의 문제 해결률을 달성하며 기존의 더 큰 모델들을 능가하는 성능을 보여주었습니다. 또한, 내부 TTC와 외부 TTC의 결합을 통해 모델의 성능을 더 향상시켰으며, 특히 어려운 문제에 대해 더 많은 토큰을 동적으로 할당핼 수 있는 능력이 향상된 것을 확인했습니다. 나아가, 우리는 연구 및 개발을 지원하기 위해 모든 훈련 데이터, 모델, 코드를 공개했습니다.



### Adaptive Layer-skipping in Pre-trained LLMs (https://arxiv.org/abs/2503.23798)
- **What's New**: 본 논문에서는 대형 언어 모델(LLMs)에서의 토큰 생성 시 계산 요구 사항이 어떻게 달라지는지를 탐구하며, FlexiDepth라는 새로운 방법을 제안합니다. FlexiDepth는 각 토큰에 대해 Transformer 레이어의 수를 동적으로 조절하여 계산 효율성을 높입니다. 기존의 방법들과 달리, FlexiDepth는 사전 훈련된 모델의 파라미터를 변경하지 않고 레이어 스킵을 구현할 수 있도록 설계되어 있습니다.

- **Technical Details**: FlexiDepth는 각 Transformer 레이어에 라우터(router)와 어댑터(adapter)를 추가하여 동적 레이어 스킵을 가능하게 합니다. 라우터는 입력된 상태를 바탕으로 레이어를 통과할지 스킵할지를 결정하며, 어댑터는 스킵된 상태의 표현을 조정하여 일관성을 유지합니다. 이 방식은 자동 회귀 생성(auto-regressive generation)과의 호환성을 보장하기 위해 모든 상태에 대해 KV 캐시를 계산합니다.

- **Performance Highlights**: 실험 결과에 따르면, FlexiDepth는 Llama-3-8B 모델에서 32개 레이어 중 8개를 스킵하면서도 100%의 성능을 유지하였고, 특히 연속 생성 작업에서 기존 레이어 스킵 방법들보다 뛰어난 성과를 보였습니다. FlexiDepth는 또한 다양한 토큰 유형 생성 시 계산 요구 사항이 어떻게 달라지는지를 보여주는 레이어 할당 패턴을 드러내며, 향후 연구를 위한 데이터셋을 공개했습니다.



### MGD-SAM2: Multi-view Guided Detail-enhanced Segment Anything Model 2 for High-Resolution Class-agnostic Segmentation (https://arxiv.org/abs/2503.23786)
- **What's New**: 이번 연구에서는 새로운 MGD-SAM2 모델을 제안합니다. MGD-SAM2는 고해상도 클래스 비의존 분할(HRCS) 작업을 위해 설계되었으며, Multi-view images와 SAM2의 일반적인 시각 정보의 상호작용을 결합하여 보다 정밀한 물체 분할을 달성합니다. 본 모델은 Multi-view Perception Adapter(MPAdapter), Multi-view Complementary Enhancement Module(MCEM), Hierarchical Multi-view Interaction Module(HMIM), Detail Refinement Module(DRM) 등 네 가지 새로운 모듈을 포함하고 있습니다.

- **Technical Details**: MGD-SAM2는 이미지의 전역 이미지와 지역 패치 간의 다중 뷰 기능 상호작용을 통해 세밀한 세분화를 달성합니다. MPAdapter는 SAM2의 인코더를 조정하여 HRCS 이미지에서 지역 세부 사항과 전역 의미를 강화합니다. MCEM와 HMIM은 다중 뷰 기능을 집계하여 지역 질감과 전역 문맥을 추가로 활용하며, DRM은 데이터 세트에서 세부 사항 손실을 보완하기 위해 서서히 복원된 고해상도 마스크 예측을 생성합니다.

- **Performance Highlights**: 실험 결과, MGD-SAM2는 여러 고해상도 및 일반 해상도 데이터 세트에서 뛰어난 성능을 나타냈습니다. 이전의 SOTA(State-of-the-art) 방법들을 초월하며 DIS5K, HRSOD, UHRSD, DAVIS-S와 같은 고해상도 데이터 세트와 DUTS, HKU-IS와 같은 일반 해상도 데이터 세트에서도 새로운 기록을 세웠습니다. 이는 MGD-SAM2의 효과성과 견고함을 입증하는 결과입니다.



### WinoWhat: A Parallel Corpus of Paraphrased WinoGrande Sentences with Common Sense Categorization (https://arxiv.org/abs/2503.23779)
- **What's New**: 이번 연구에서는 Winograd 스키마(Winograd Schema) 챌린지를 통해 대형 언어 모델(LLM)의 상식 추론(common sense reasoning) 능력을 평가하는 방법을 조명합니다. 새로운 데이터셋인 WinoWhat을 발표하여 WinoGrande의 검증 집합의 각 인스턴스를 패러프레이즈(paraphrase)했습니다. 이를 통해 LLM의 WinoGrande 성능이 패러프레이징에 강한지 여부를 테스트합니다.

- **Technical Details**: 연구는 Gemma 2, LlaMA 2, OPT 등의 오픈 소스 모델들을 WinoGrande에서 평가합니다. 또한, 데이터 누수(data leakage)에 대한 검증을 위해 LLM의 사전학습(pre-training) 데이터와 검증 집합 인스턴스의 매칭을 통해 두 개의 테스트 세트를 생성합니다. 모델 성능 평가를 위해 상식 지식 범주를 정의하여 각 범주별로 모델을 평가하며, 이는 상식 추론 과제에서 모델의 강약점을 이해하는 데 기여합니다.

- **Performance Highlights**: 모든 모델은 WinoWhat에서 상대적으로 낮은 성과를 보였으며, 이는 LLM의 WinoGrande에서의 추론 능력이 과대평가되었음을 암시합니다. 데이터 기억(memorization)이 모델 성능에 미치는 영향은 미미하다는 것을 확인하였으며, 기존의 WinoGrande 벤치마크 결과는 상식 획득을 나타내지 않음을 시사합니다. 이러한 결과는 상식 지식 평가를 위한 새로운 접근 방식을 요구합니다.



### WaveFormer: A 3D Transformer with Wavelet-Driven Feature Representation for Efficient Medical Image Segmentation (https://arxiv.org/abs/2503.23764)
- **What's New**: WaveFormer는 3D-transformer 아키텍처로, 의료 이미지 분석의 새로운 장을 열고 있습니다. 이 모델은 주파수 도메인 속성을 활용하여 효과적으로 맥락을 표현하고, 인간의 시각 인식 시스템의 상위 하향 메커니즘을 통해 설계되었습니다. WaveFormer는 디스크리트 웨이블릿 변환(Discrete Wavelet Transform, DWT)을 여러 스케일에서 활용하여 전역적 맥락과 세부 정보를 동시에 유지하는 혁신적인 접근을 제안합니다.

- **Technical Details**: WaveFormer는 두 가지 주요 설계 원칙을 바탕으로 구성됩니다: 효과적인 전역 맥락 모델링과 세부 정보 보존입니다. DWT를 사용해 저주파 서브밴드를 추출함으로써, WaveFormer는 자가참조(self-attention)를 더 компакт한 표현에서 수행하며 필수적인 맥락 정보를 보존합니다. 또한, IDWT(inverse DWT) 메커니즘을 통해 고해상도 세그멘테이션 마스크를 점진적으로 재구성하여 세밀한 구조를 캡처합니다.

- **Performance Highlights**: WaveFormer는 BraTS2023, FLARE2021, KiTS2023과 같은 주요 3D 의료 벤치마크에서 평가되었습니다. 실험 결과 WaveFormer는 기존 최첨단 방법과 비교하여 경쟁력 있는 정확도를 달성했으며, 모델 복잡성과 추론시간을 현저히 줄였습니다. 이러한 성능 덕분에 WaveFormer는 자원이 제한된 임상 환경에서도 효율적으로 배포할 수 있는 가능성을 보여주고 있습니다.



### LANID: LLM-assisted New Intent Discovery (https://arxiv.org/abs/2503.23740)
Comments:
          Published in LREC-COLING 2024

- **What's New**: 본 논문에서는 새로운 의도 인식(New Intent Discovery, NID)을 향상시키기 위해, 경량 NID 인코더를 대형 언어 모델(Large Language Models, LLM)의 세멘틱 리프레젠테이션을 통해 개선하는 LANID 프레임워크를 제안합니다. 기존 NID 방법의 한계를 극복하고, 효율적으로 작동하는 경량 모델링을 가능하게 합니다. 이 프레임워크는 LLM을 활용하여 유의미한 발화 쌍을 샘플링하고, 이를 기반으로 대비 학습(contrastive learning)을 적용하여 높은 성능을 보여줍니다.

- **Technical Details**: LANID는 K-최인접 알고리즘(K-nearest neighbors) 및 DBSCAN(Density-Based Spatial Clustering of Applications with Noise) 알고리즘을 사용하여 선택적 발화 쌍을 샘플링합니다. 이후 LLM에 이들 쌍 간의 관계를 질의하여 정보를 얻고, 이를 사용하여 대비 학습의 과제를 구성합니다. Small encoder를 대비 삼중 손실(contrastive triplet loss)로 훈련하여 효율성을 극대화하며, 데이터 샘플링과 교육 단계를 반복적으로 수행합니다.

- **Performance Highlights**: 실험 결과, LANID는 세 가지 NID 데이터셋에서 기존 강력한 기준선들을 초과하는 성과를 보여주었습니다. 언수퍼바이즈드(unsupervised) 및 세미-슈퍼바이즈드(semi-supervised) 설정 모두에서 뛰어난 효율성과 성능을 입증하였습니다. 최종적으로 경량화된 인코더도 기존과 같은 성능을 유지하며, 업계의 실질적인 응용에 적합한 솔루션을 제공합니다.



### Investigation of intelligent barbell squat coaching system based on computer vision and machine learning (https://arxiv.org/abs/2503.23731)
- **What's New**: 이번 연구는 인공지능(AI)과 컴퓨터 비전 기술을 활용하여 바벨 스쿼트 훈련의 효율성을 높이는 시스템을 개발하였습니다. 본 시스템은 실시간으로 문제를 진단하고 피드백을 제공하는 기능을 갖추고 있어 혼자서도 제대로 훈련할 수 있도록 지원합니다. 또한, 재생 모드를 통해 사용자가 이전 스쿼트를 분석하고 코멘트를 확인할 수 있습니다.

- **Technical Details**: 총 77명의 참가자로부터 8,151개의 스쿼트를 수집하여 각각 좋은 스쿼트 및 여섯 가지 문제로 분류했습니다. 이후, 세 가지 머신러닝 아키텍처를 통해 진단 모델을 훈련하였고, SHAP 방법을 적용하여 문제 예측의 정확성을 향상시켰습니다. 이 시스템은 스쿼트를 진단하는데 0.5초 이하의 시간을 소요합니다.

- **Performance Highlights**: 여섯 가지 문제에 대한 F1 점수는 각각 86.86%, 69.01%, 77.42%, 90.74%, 95.83%, 100%에 도달하였습니다. 시스템을 사용하여 훈련한 참가자들은 기술적으로 상당한 개선을 보였으며, 이는 전문 웨이트 리프팅 코치에 의해 평가되었습니다. 이 연구는 실시간으로 사용자 친화적인 바벨 스쿼트 피드백 및 훈련 시스템을 구축하는 것을 목표로 한 종합적인 연구입니다.



### KOFFVQA: An Objectively Evaluated Free-form VQA Benchmark for Large Vision-Language Models in the Korean Languag (https://arxiv.org/abs/2503.23730)
Comments:
          Accepted to CVPRW 2025, Workshop on Benchmarking and Expanding AI Multimodal Approaches

- **What's New**: 최근 대규모 비전-언어 모델(Visual-Language Models, VLMs)의 발전으로 다양한 평가 기준이 등장했습니다. 하지만 기존 평가 방법들은 주어진 응답 중에서 모델이 선택하도록 요구하거나, 주판 모델(judge model)을 사용하여 주관적이라는 문제점이 있었습니다. 본 연구에서는 한국어를 위한 새로운 평가 기준을 제공하는 KOFFVQA 벤치마크를 제안합니다.

- **Technical Details**: KOFFVQA는 275개의 주어진 이미지와 질문 쌍을 포함하며, 10가지 VLM 성능 측면을 평가하는 grading criteria(채점 기준)를 제공합니다. 각 응답은 미리 정의된 채점 기준을 기반으로 LLM(대형 언어 모델)으로 채점되며, 이는 평가의 신뢰성을 높이는 데 기여합니다. 이를 통해 작은 오픈 소스 모델이라도 신뢰할 수 있는 평가를 할 수 있습니다.

- **Performance Highlights**: KOFFVQA 벤치마크를 활용하여 47개의 VLM 모델을 평가한 결과, 한국어 언어에서의 성능은 영어 벤치마크에서의 성능과는 상이한 패턴을 보였습니다. 우리의 접근 방식은 기존 방법과 비교하여 평가의 일관성을 크게 향상시켰고, 이는 장기적인 응답을 평가할 때 발생할 수 있는 주관적 문제들을 줄여줍니다. LLM을 평가지로 사용한 방법은 특히 더 신뢰할 수 있는 결과를 제공했습니다.



### Unimodal-driven Distillation in Multimodal Emotion Recognition with Dynamic Fusion (https://arxiv.org/abs/2503.23721)
- **What's New**: 본 논문에서는 멀티모달 감정 인식(MERC) 시스템에서의 문제를 해결하기 위해 SUMMER라는 새로운 프레임워크를 제안합니다. SUMMER는 Sparse Dynamic Mixture of Experts(SDMoE), Hierarchical Cross-Modal Fusion(HCMF) 및 Interactive Knowledge Distillation(IKD) 같은 주요 구성 요소를 활용하여 이질적인 모드 통합을 지원합니다. 이를 통해 기존 모델들이 직면한 모달 이질성과 학습 지침 부족 문제를 해결하고 있습니다.

- **Technical Details**: SUMMER는 세 가지 모달(텍스트, 오디오, 비디오)을 처리하는 다중 모달 감정 인식을 위한 프레임워크입니다. SDMoE는 최신 동적 토큰 선택을 통해 중요 특징을 식별하고, HCMF는 이질적인 모드 간 관계를 포착하여 더 나은 전역 맥락 이해를 도와줍니다. IKD는 미리 훈련된 단일 모달 교사 모델을 통해 다중 모달 학생 모델을 지도하여 특성 분포 격차를 줄이고, 클래스 간 관계를 포착하는 소프트 레이블을 제공합니다.

- **Performance Highlights**: 이 연구의 실험 결과는 IEMOCAP 및 MELD 데이터셋에서 SUMMER가 최신 기법들을 능가하며, 특히 소수 감정 및 의미적으로 유사한 감정 인식에서 놀라운 성능을 보여줍니다. 이 혁신적인 접근 방식은 감정 인식을 위한 동적 토큰 선택과 이질적 모드의 융합을 통해 보다 정밀하고 강력한 결과를 도출하고 있습니다.



### GNN-Based Candidate Node Predictor for Influence Maximization in Temporal Graphs (https://arxiv.org/abs/2503.23713)
Comments:
          9 pages, 5 figures, Accepted in AAAI25 to AI4TS Workshop@AAAI 2025

- **What's New**: 이 논문은 소셜 미디어에서 영향을 미치는 노드를 효과적으로 식별하기 위한 새로운 학습 기반 접근 방식을 제안합니다. Graph Neural Networks (GNNs)과 Bidirectional Long Short-Term Memory (BiLSTM) 모델을 통합하여 동적인 네트워크에서 노드의 구조적 및 시간적 동향을 포착합니다. 이 혼합 프레임워크는 시드 세트 선정에 대한 후보 노드를 정확하게 예측할 수 있도록 합니다.

- **Technical Details**: 본 연구에서는 GNNs가 구조적 정보를 포착하고 BiLSTM이 시간적 의존성을 처리하며, 이 두 모델을 결합하여 동적인 네트워크의 변화에 따라 영향력을 극대화하는 시드 노드를 효과적으로 선정하는 방법을 제안합니다. 이 접근법은 그래프나 네트워크를 입력으로 받으며, 각 시간 스냅샷에 대해 후보 노드를 예측합니다. 이를 통해 90%의 예측 정확도를 달성하며, 복잡한 계산 요구 사항을 줄이는 데 기여합니다.

- **Performance Highlights**: 지속 가능한 네트워크 환경에서 이 방법은 광범위한 시뮬레이션을 통해 81%에서 98%의 정확도로 후보 노드 집합을 예측할 수 있음을 입증합니다. 이를 통해 네트워크 구성의 다양성에 대한 확장성을 보여주며, 효율적인 시드 계산을 통한 영향력 확산 최적화가 가능합니다. 본 방법은 바이럴 마케팅 및 사회적 네트워크 분석과 같은 분야에서 특히 효과적입니다.



### Towards Benchmarking and Assessing the Safety and Robustness of Autonomous Driving on Safety-critical Scenarios (https://arxiv.org/abs/2503.23708)
- **What's New**: 본 연구에서는 자율주행(Autonomous Driving) 시스템의 안전성과 견고성을 평가하기 위해 안전-중요 시나리오(safety-critical scenarios)의 정의를 체계적으로 제시합니다. 자연적 배포 변화(natural distribution shifts)와 적대적 공격(adversarial attack) 시나리오를 포함하는 정적 교통 시나리오뿐만 아니라 동적인 사고 시나리오를 평가합니다. 저자들은 SSAD라는 자율주행 안전 테스트 플랫폼을 개발하여 시스템의 인지 모듈 뿐만 아니라 시스템 수준에서도 포괄적인 평가를 진행합니다.

- **Technical Details**: SSAD 플랫폼은 다양한 안전-중요 시나리오를 분석하고 인공지능(AI) 구성요소 평가와 자율주행 시스템의 전체적인 평가를 통합합니다. 평가에는 인지 모듈의 성능 외에도 경로 완료율(route completion)과 충돌율(collision rate)과 같은 시스템 수준의 메트릭이 포함됩니다. 연구에서는 정적 및 동적 안전-중요 시나리오를 재정의하고, 자연적인 소음과 적대적 공격 방법을 고려하여 시스템의 신뢰성을 높입니다.

- **Performance Highlights**: SSAD는 자율주행 시스템의 안전성을 종합적으로 평가할 수 있도록 다각적인 접근 방식을 제공합니다. 다양한 안전-중요 시나리오를 구축하고, AI의 구성요소에 대한 평가와 함께 시스템의 전체 성능을 고려하여 실용적인 도구를 제공합니다. 결과적으로, SSAD는 자율주행 시스템의 안전성을 강화하고 미래의 상업적 배치를 위한 신뢰성을 높이는 데 기여할 것입니다.



### Remarks on the Polyak-Lojasiewicz inequality and the convergence of gradient systems (https://arxiv.org/abs/2503.23641)
- **What's New**: 이 연구는 Polyak-Lojasiewicz 불평등(PLI)의 일반화와 그것이 최적화 문제에서 경량 흐름(gradient flows)의 수렴 행동에 미치는 영향을 탐구합니다. 특히, 연속시간 선형 이차 조절기(CT-LQR) 정책 최적화 문제에 대한 연구가 이뤄지며, LQR에서의 PLI 강한 형태가 존재할 수 없음을 보여줍니다. 연구는 이론적인 분석에 대한 공감대 형성을 위해 기존 결과를 재조명하며, 다양한 불평등의 형태가 경량 흐름 해의 '프로파일'에 미치는 영향을 살펴봅니다.

- **Technical Details**: 이 논문은 PLI의 일반화가 최적화 문제에서의 수렴 속도에 어떻게 영향을 미치는지를 규명하는 데 집중합니다. 특히, 연속시간 LQR 정책 최적화 문제에서 gPLI의 만족 여부를 검토하며, 이 문제의 경우 gPLI가 항상 충족될 수 없음을 나타냅니다. 이론적 분석과 함께, 점근적인 수렴 속도가 다양한 형태의 PLI에 의해 어떻게 관련되는지를 탐구합니다.

- **Performance Highlights**: 연구 결과, weaker PLI 조건이 비용 함수의 극값 집합에 대한 글로벌 수렴과 최적성을 보장할 수 있음을 나타냅니다. 그러나, LQR 정책 최적화 문제는 이러한 조건들을 만족하지 않으며, 그로 인해 수렴 프로파일이 '서브-지수(sub-exponential)' 형태로 나타나는 경우가 많습니다. 마지막으로 L1 정규화를 포함하는 최적화 문제에의 확장 가능성이 논의됩니다.



### Finding Interest Needle in Popularity Haystack: Improving Retrieval by Modeling Item Exposur (https://arxiv.org/abs/2503.23630)
Comments:
          2 pages

- **What's New**: 본 논문에서는 추천 시스템의 지속적인 피드백 루프 속에서 발생하는 인기 편향(popularity bias) 문제를 해결하기 위한 새로운 접근법을 소개합니다. 기존 방법들과 달리, 이 연구는 실시간으로 노출 동역학(exposure dynamics)을 조절할 수 있는 점에서 차별성이 있습니다. 특히, 노출 확률(exposure probability)을 모델링하여 검색 단계에서의 순위를 조정하는 방법을 제안합니다.

- **Technical Details**: 우리가 제안하는 접근법은 노출 효과(exposure effects)와 참여 가능성(engagement likelihood)을 분리하여, 대규모 추천 플랫폼에서 공정성(fairness)과 참여도(engagement) 간의 조절 가능한 균형을 이룹니다. 실제 비디오 추천 시스템에서의 온라인 A/B 실험을 통한 검증을 실시하였습니다. 이 과정에서 노출 점수(retrieval scoring) 접근법이 추천 알고리즘의 성능을 어떻게 개선할 수 있는지를 보여주었습니다.

- **Performance Highlights**: 이 방법은 고유하게 검색된 항목(unique retrieval items)의 수를 25% 증가시키고, 과도한 인기 콘텐츠의 지배력을 40% 낮추는 결과를 얻었습니다. 또한, 사용자 참여 수준을 유지하면서 이러한 성과를 달성하여, 인기 편향을 완화하기 위한 확장 가능하고 배포 가능한 솔루션을 제시하게 되었습니다.



### Graph-Eq: Discovering Mathematical Equations using Graph Generative Models (https://arxiv.org/abs/2503.23617)
Comments:
          8 pages, 4 figures

- **What's New**: 본 논문에서는 데이터셋을 설명하는 의미 있고 정확하며 간결한 수학적 방정식을 발견하는 새로운 방법인 Graph-EQ를 제안합니다. 수학적 방정식을 그래픽 표현으로 나타내어 Graph Neural Networks (GNNs)를 활용, 이전에 보지 못한 새로운 방정식을 생성합니다. 기존의 유전자 프로그래밍 방법의 비효율성을 극복하기 위해 조건부 변분 오토인코더(Conditional Variational Autoencoder, CVAE)를 사용해 방정식 공간의 잠재 표현을 학습합니다.

- **Technical Details**: Graph-EQ는 대규모 방정식 집합에 대해 비지도 학습을 통해 방정식의 풍부한 잠재 표현을 학습합니다. 전통적인 방법과 달리, Graph-EQ는 방정식 공간을 직접 탐색하는 대신 베이esian 최적화(Bayesian optimization)를 통해 효율적으로 학습된 잠재 공간을 탐색합니다. 또한, 이 모델은 입력 방정식을 정확히 재구성하고, 학습된 잠재 표현이 새로운 유효 방정식으로 디코드될 수 있음을 보여줍니다.

- **Performance Highlights**: 20개의 알려진 기준 진실 방정식으로 구성된 데이터셋에서 Latent space 탐색을 수행하여 Graph-EQ가 대부분의 데이터셋에서 기준 진실 방정식을 성공적으로 발견함을 입증했습니다. 이 연구는 방정식 발견을 위한 그래프 생성 모델의 유용성을 뒷받침하며, Graph-EQ가 기존 방법에 비해 과적합(overfitting)의 위험을 줄이는 동시에 수학적 관계를 효과적으로 캡처할 수 있음을 보여줍니다.



### Interpretable Machine Learning in Physics: A Review (https://arxiv.org/abs/2503.23616)
- **What's New**: 본 논문은 머신러닝의 해석 가능성이 물리학에 응용되는 중요한 역할을 조명하고 있습니다. 머신러닝 알고리즘의 발전과 함께, 데이터에서 패턴을 학습하고 이를 기초로 새로운 과학적 발견을 촉진하는 과정에서 해석 가능성의 필요성이 강조됩니다. 특히, 인공지능(AI)이 인간 능력 너머의 발견을 가능하게 할 때, 이러한 모델의 개념과 예측을 이해할 수 있는 것이 필수적이라고 언급하고 있습니다.

- **Technical Details**: 해석 가능한 머신러닝(Interpretable Machine Learning)은 머신러닝 모델이 내리는 결정을 인간이 이해하고 설명할 수 있는 언어로 번역하는 과정입니다. 이 과정은 본질적으로 이해할 수 있는 알고리즘과 '블랙박스' 모델 간의 구분을 포함합니다. 간단한 모델은 더 쉽게 해석할 수 있지만, 복잡한 인공지능 알고리즘은 예측 결과를 도출하는 방법을 이해하는 데 어려움이 있으며, 이는 다양한 해석 도구의 개발을 요구합니다.

- **Performance Highlights**: 해석 가능성은 과학적 발견을 위한 핵심 요소로 작용하여, 연구자들이 머신러닝 모델의 예측을 이해하고 검증할 수 있게 합니다. 이를 통해 모델의 결정이 의미 있는 패턴에 기반하고 있음을 확인할 수 있으며, 이는 신뢰성과 재현성을 강화합니다. 또한, 해석 가능한 모델은 연구자들이 잘못된 예측의 원인을 분석하고 모델을 조정하여 개선하는 데 도움을 줍니다.



### Partial Transportability for Domain Generalization (https://arxiv.org/abs/2503.23605)
Comments:
this http URL

- **What's New**: 이 논문은 AI에서 예측의 성능 보증을 제공하는 데 있어 새로운 접근 방식을 제안합니다. 특히, 이전에는 신뢰할 수 있는 방법이 부족했던 새로운 데이터 분포에 대한 불확실성을 고려한 연구입니다. 저자들은 causal diagrams를 사용하여 데이터 생성 메커니즘에 대한 가정을 수립하고, 이에 따른 새로운 성과를 제시합니다.

- **Technical Details**: 논문에서는 partial identification과 transportability의 이론을 기반으로 하여, 목표 분포의 functional value를 경계 지을 수 있는 방법을 제안합니다. 이는 Neural Causal Models와 같은 기존의 매개변수화 구조를 채택하여 cross-population inference에 필요한 구조적 제약을 인코딩합니다. 또한 높은 표현력과 일관성을 보장하는 절차를 수행하고, 실제로 확장 가능한 추론을 위한 gradient-based optimization scheme을 제안합니다.

- **Performance Highlights**: 저자들은 제안된 방법론의 유용성을 실험을 통해 입증했습니다. 이러한 결과는 기존 예측기의 불확실성을 줄이고 새로운 도메인에서의 일반화 오류를 개선하는 데 중요한 기여를 할 것으로 기대됩니다. 이 연구는 AI의 실용적인 응용에 중요한 기틀을 마련하고 있습니다.



### DASH: Detection and Assessment of Systematic Hallucinations of VLMs (https://arxiv.org/abs/2503.23573)
- **What's New**: 이 논문에서는 오픈 월드(open-world) 환경에서 비전-언어 모델(vision-language models, VLMs)의 체계적 환각(object hallucinations)을 탐지하고 평가하기 위한 자동화된 파이프라인인 DASH(Detection and Assessment of Systematic Hallucinations)를 제안합니다. 기존의 벤치마크가 작고 레이블이 있는 데이터셋에 의존하여 제한된 결과를 제공한 반면, DASH는 자연 사진 매니폴드(natural image manifold)를 최적화하여 VLM을 혼란스럽게 하는 이미지를 생성하고 이를 통해 환각 물체들을 식별합니다.

- **Technical Details**: DASH는 두 가지 주요 구성 요소인 DASH-OPT와 DASH-LLM로 구성됩니다. DASH-OPT는 이미지 기반의 검색 방식을 통해 VLM이 존재하지 않는 물체를 환각하게 만드는 이미지를 생성하도록 최적화됩니다. 반면, DASH-LLM은 대규모 언어 모델(LLM)에서 생성된 쿼리를 바탕으로 기능하며, 이 두 가지 접근 방식을 통해 이미지와 텍스트 쿼리를 탐색하여 FP-hallucinations를 유발하는 실체 이미지를 찾습니다.

- **Performance Highlights**: DASH를 PaliGemma와 여러 LLaVA-NeXT 모델에 적용한 결과, 19,000개 이상의 클러스터와 950,000개 이상의 이미지를 발견했습니다. 찾아낸 환각 이미지는 다른 VLM으로 성공적으로 이전되며, DASH를 통해 생성된 특정 이미지로 PaliGemma를 미세 조정하면 객체 환각 문제를 완화할 수 있음을 보입니다. 또한, DASH-B라는 새로운 벤치마크를 제안하여 현재 VLMs의 평가를 보다 신뢰할 수 있게 할 수 있음을 보여주고 있습니다.



### Addressing Model Overcomplexity in Drug-Drug Interaction Prediction With Molecular Fingerprints (https://arxiv.org/abs/2503.23550)
Comments:
          Accepted to the GEM Workshop at ICLR 2025

- **What's New**: 최근의 딥러닝 모델들은 약물-약물 상호작용(drug-drug interactions, DDIs)을 정확히 예측하는 데 어려움을 겪고 있습니다. 본 연구에서는 분자 표현(molecular representations), 즉 Morgan fingerprints (MFPS), 그래프 기반 임베딩(graph-based embeddings), 그리고 MoLFormer의 transformer-derived embeddings를 사용하여 간단하면서도 효과적인 접근 방식을 탐구하였습니다. 이러한 방법은 DrugBank DDI 데이터셋의 평가에서 경쟁력 있는 성능을 보이며, 간단한 분자 표현이 충분하다는 것을 입증하였습니다.

- **Technical Details**: Morgan fingerprints는 분자 구조를 고정된 길이의 이진 벡터로 인코딩하여 특정 하위 구조의 존재를 나타냅니다. 또한, 그래프 신경망(graph convolutional networks, GCNs)은 분자를 그래프로 표현하여 원자와 화학 결합을 노드와 엣지로 정의합니다. 여성 음악의 작곡자인 MoLFormer는 SMILES 문자열을 입력으로 처리하여 화학적 문맥을 포착합니다. 이 모델은 DrugBank 데이터를 기반으로 DDI 분류 및 DDA 회귀 작업을 평가하는 모듈식 신경망으로 설계되었습니다.

- **Performance Highlights**: 평가 결과, MFPS는 Unseen DDI 분할에서 가장 높은 AUROC(99.4%) 및 AUPR(98.4%)를 기록하며, MoLFormer보다 높은 성능을 보였습니다. Unseen 1 Drug 분할에서는 MFPS와 사전 학습된 GCN 임베딩이 유사한 정확도를 보여주었으나, 이들은 더 일관된 예측을 제공하여 안정성 면에서 우수함을 나타냅니다. 전체적으로, MFPS와 GCN 임베딩은 정확성과 안정성 사이의 균형 잡힌 타협을 제공하며, DDI 예측에 있어 간단하고 해석 가능한 분자 표현의 통합 가치를 강화합니다.



### A Survey on Unlearnable Data (https://arxiv.org/abs/2503.23536)
Comments:
          31 pages, 3 figures

- **What's New**: 본 논문은 Unlearnable Data (ULD)라는 새로운 방어 기법을 제시합니다. ULD는 특정 데이터에서 유의미한 패턴을 학습하는 것을 방지하여 데이터 프라이버시와 보안을 보호합니다. 이 기술은 학습 데이터에 섭동(perturbation)을 추가하여 모델 성능을 저하시키고, 무단 모델이 유용한 표현을 추출하기 어렵게 만드는 것을 목표로 합니다. 기존의 연구들은 주로 적대적 공격(adversarial attacks)과 기계 비학습(machine unlearning)에 초점을 맞춰왔으나, 본 논문은 독립적인 ULD로의 포괄적인 리뷰를 제공합니다.

- **Technical Details**: 연구는 ULD의 생성 방법, 공개 벤치마크, 평가 지표, 이론적 기초 및 실제 응용 분야를 상세히 분석합니다. ULD는 훈련 데이터에 대한 섭동을 통해 모델이 유용한 표현을 학습하지 못하도록 수정하며, 이는 인식 가능성(perceptual quality)을 유지합니다. 또한, ULD는 기존의 머신 비학습과 적대적 공격과의 차별성을 명확히 하고, 각 접근방식의 강점과 한계를 비교합니다. 주요 기술적 도전 과제는 섭동의 인식 불가능성(imperceptibility)과 모델 성능 저하 사이의 균형을 맞추는 것입니다.

- **Performance Highlights**: 이 연구에서는 다양한 ULD 접근 방식의 효과성과 한계를 논의합니다. ULD 방식은 훈련된 모델이 일반화 성능이 저하되도록 설계되어 있으며, 데이터 프라이버시 보호 기술로서 유망한 잠재력을 갖추고 있습니다. 그러나 높은 계산 복잡성(computational complexity)과 모델 성능 저하 사이의 무역오프(trade-off)는 실용적인 적용에 있어 도전 과제가 되고 있습니다. 향후 연구 방향으로는 ULD의 효과성과 응용 가능성을 향상시킬 방안이 강조되고 있습니다.



### BiPVL-Seg: Bidirectional Progressive Vision-Language Fusion with Global-Local Alignment for Medical Image Segmentation (https://arxiv.org/abs/2503.23534)
- **What's New**: BiPVL-Seg는 의료 이미지 세분화에서 비전(vision)과 언어(language)의 융합 및 임베딩(embedding) 정렬을 통합하는 새로운 프레임워크를 제안합니다. 이 프레임워크는 전체적인 전략과 훈련 혁신을 통해 두 구성 요소가 서로 강화되어 세분화 성능을 향상시킵니다. 또한, BiPVL-Seg는 비가역적 진행 융합(bidirectional progressive fusion) 아키텍처를 도입하여 비전 및 텍스트 인코더 간의 단계별 정보 교환을 촉진합니다.

- **Technical Details**: 기존의 VLM(Vision-Language Model) 구조들은 비전 인코더에서 독립적으로 처리를 하여 크로스 모달(Cross-modal) 정렬을 약하게 만들었습니다. BiPVL-Seg는 이러한 약점을 극복하기 위해 비가역적 진행 융합을 통해 인코더의 모든 단계에서 지속적인 정보 교환을 실현합니다. 또한, 글로벌-로컬 대비 정렬(global-local contrastive alignment) 방식을 도입하여 의료용 텍스트와 비전 임베딩 간의 의미 있는 정렬을 가능하게 합니다.

- **Performance Highlights**: 다양한 의료 이미지 벤치마크(CT 및 MR 모드 포함)에서 실시된 광범위한 실험을 통해 BiPVL-Seg의 성능이 최첨단 방법들보다 우수한 것으로 나타났습니다. 이는 복잡한 다중 클래스 세분화에서 특히 두드러지며, 체계화된 해부학적 구조 및 종양 이미징에 이르기까지 폭넓은 성능을 겸비했습니다. 소스 코드는 GitHub 저장소에서 확인할 수 있습니다.



### If an LLM Were a Character, Would It Know Its Own Story? Evaluating Lifelong Learning in LLMs (https://arxiv.org/abs/2503.23514)
- **What's New**: 이번 논문에서는 큰 언어 모델(LLMs)이 다중 대화 상호작용 중에 일관된 성격을 보이는 현상을 관찰하고, 이를 평가하기 위해 LIFESTATE-BENCH라는 새로운 벤치마크를 도입합니다. LLMs의 상태 진화(상태 변화를 측정하는 지표)를 평가하며, 기존 벤치마크의 한계를 극복하고자 합니다. 특히 에피소드 기반 데이터셋을 통해 현재와 과거의 대화 맥락을 연결하는 방식을 탐구합니다.

- **Technical Details**: LIFESTATE-BENCH는 Hamlet 및 합성 스크립트와 같은 두 가지 에피소드 데이터셋으로 구성되어 있습니다. 이 데이터셋은 LLMs의 자기 인식, 기억 회상, 관계 추적을 평가하기 위한 사실 기반 질문 차원을 설계했습니다. 또한, 비모수적(non-parametric) 방법과 모수적(parametric) 방법을 통해 LLM의 장기 기억 능력을 측정하는 다양한 접근 방식을 탐구했습니다.

- **Performance Highlights**: 실험 결과, 비모수적 방법이 모수적 방법보다 상태 기반 학습에 더 효과적임을 보여주었습니다. 그러나 모든 모델은 대화가 진행됨에 따라 기억 상실(catasrophic forgetting) 문제에 직면하며, 지속적인 개선이 필요하다는 점이 분명해졌습니다. 본 논문은 LMs의 평생 학습 능력을 평가하고 향후 연구 방향을 제시하는 데 중요한 기여를 하고 있습니다.



### Buffer is All You Need: Defending Federated Learning against Backdoor Attacks under Non-iids via Buffering (https://arxiv.org/abs/2503.23511)
- **What's New**: 이번 논문에서 제안된 FLBuff는 Federated Learning(FL)에서 발생하는 backdoor 공격에 효과적으로 대응하기 위한 방안입니다. FL은 데이터 공유 없이도 클라이언트들이 공동으로 모델을 학습할 수 있게 해 주지만, 분산된 특성으로 인해 backdoor 공격에 취약합니다. 기존의 방어 기법들은 일반적으로 independent-and-identically-distributed(iid) 환경에서 설계되어 비정상적인(non-iid) 환경의 특성을 간과했습니다.

- **Technical Details**: FLBuff는 비정상적인(non-iid) 환경에서도 backdoor 공격에 효과적으로 대응하기 위해 설계된 방어 기법입니다. 이 시스템은 모델 업데이트를 benign과 malicious로 구분하는 데 도움을 주기 위해 representation 공간에서 omni-directional한 확장을 모형화합니다. 시스템 구조는 파라미터 서버(parameter server)와 n개의 원격 클라이언트로 구성되며, 각 클라이언트는 고유한 데이터 분포를 가집니다.

- **Performance Highlights**: FLBuff는 여러 데이터셋에서 통합적인 실험을 통해 State-of-the-art(SOTA) 방어 기법들보다 일관되게 우수한 성능을 보였습니다. 또한, FLBuff는 이전에 보지 못한 backdoor 공격에도 적용 가능하며 다양한 강력한 적응 공격에도 저항성을 보였습니다. 이러한 결과는 FLBuff의 실용성과 포괄성을 입증하였습니다.



### Boosting Omnidirectional Stereo Matching with a Pre-trained Depth Foundation Mod (https://arxiv.org/abs/2503.23502)
Comments:
          Project page: this https URL

- **What's New**: DFI-OmniStereo는 새로운 옴니디렉셔널 스테레오 매칭 방법으로, 대규모로 사전 학습된 단안 깊이 모델을 활용하여 상대 단안 깊이 추정을 수행합니다. 이 방법은 반복 최적화 기반의 스테레오 매칭 아키텍처에 통합되어 있으며, 데이터 효율성과 일반화 성능을 향상시키기 위해 전용의 두 단계 훈련 전략을 도입합니다. DFI-OmniStereo는 Helvipad 데이터셋에서 기존 방법보다 약 16% 낮은 disparity MAE를 기록하며 최첨단 성과를 달성했다고 발표합니다.

- **Technical Details**: DFI-OmniStereo는 옴니디렉셔널 스테레오 이미지 쌍을 처리하기 위해 설계된 엔드 투 엔드 모델입니다. 이 모델은 첫 번째 단계에서 스테레오 매칭 헤드가 새로운 특성 공간에 적응할 수 있도록 훈련되며, 두 번째 단계에서는 일반화 성능을 유지하면서 스케일 불변 손실(scale-invariant loss)을 이용해 백엔드의 디코더를 미세 조정합니다. 이러한 과정은 반복 최적화를 통해 진행되며, 다양한 환경에서의 깊이 추정 정확도를 높입니다.

- **Performance Highlights**: DFI-OmniStereo는 Helvipad 데이터셋에서의 테스트 결과, 기존의 옴니디렉셔널 스테레오 매칭 방법들에 비해 성과가 뛰어난 것으로 나타났습니다. 이 방법은 샘플 효율성을 높이고 다른 데이터셋에 대한 일반화 능력 또한 보유하고 있습니다. 이렇게 하여 DFI-OmniStereo는 다양한 조명 조건과 깊이 범위를 갖는 여러 환경에서 깊이 정확도를 개선하는 데 기여합니다.



### POINT$^{2}$: A Polymer Informatics Training and Testing Databas (https://arxiv.org/abs/2503.23491)
- **What's New**: 이 연구에서는 POINT²(POlymer INformatics Training and Testing)를 소개하며, 이는 고성능 폴리머 재료의 발견을 가속화하고 다양한 예측 요소를 통합하기 위한 포괄적인 기준 데이터베이스 및 프로토콜입니다. 기존의 라벨링 데이터셋과 약 100만 개의 가상 폴리머가 포함된 PI1M 데이터셋을 활용하여 ML 모델의 앙상블을 개발하고 있습니다. 이를 통해 폴리머의 속성 예측, 불확실성 평가, 모델 해석 가능성 및 합성 가능성까지 수행할 수 있는 접근 방식을 제시합니다.

- **Technical Details**: POINT²는 다양한 ML 모델(Quantile Random Forests, Multilayer Perceptrons, Graph Neural Networks 등)과 여러 폴리머 표현(Morgan, MACCS, RDKit, Topological, Atom Pair fingerprints 등)을 결합하여 다양한 폴리머 속성을 평가합니다. 이 연구는 예측 정확성 뿐만 아니라 불확실성 정량화(UQ), 모델 해석 가능성, 합성 가능성을 효과적으로 통합하였습니다. 각 모델은 예측 가능한 다양한 속성(예: 가스 투과성, 열 전도도, 밀도 등)에 대한 평가를 제공합니다.

- **Performance Highlights**: 이 연구는 폴리머 화학의 넓은 영역에서 투명성과 재현성을 보장하는 동시에, ML 모델의 신뢰성을 높이는 것을 목표로 합니다. POINT²의 데이터베이스는 폴리머 발견 및 최적화 과정에서 귀중한 리소스로 활용될 수 있으며, ML 기술의 진전을 통해 더욱 혁신적인 연구 촉진이 기대됩니다. 나아가, 불확실성을 모델 개발 및 의사결정 프레임워크에 통합하여 위험 인식 기반의 소재 디자인을 지지할 수 있습니다.



### Order Independence With Finetuning (https://arxiv.org/abs/2503.23483)
Comments:
          Published as a Bi-Align workshop paper at ICLR 2025

- **What's New**: 이 논문은 세트 기반 프롬프트(Set-Based Prompting, SBP)를 활용하여 대형 언어 모델(LLMs)의 순서 의존성을 줄이는 새로운 미세 조정 방법을 제안합니다. 기존의 SBP 방법은 순서를 변경해도 동일한 의미를 가지는 답안 후보에 대해 모델의 예측이 일관되도록 하는 데 초점을 맞추었습니다. 그러나 본 연구에서는 SBP를 훈련 과정에 통합함으로써 성능 저하를 방지하고 모델의 일반적인 언어 모델링 능력을 유지할 수 있음을 보였습니다.

- **Technical Details**: 논문에서는 SBP를 LLM의 훈련 과정에 통합하는 미세 조정 전략을 소개하며, 이를 통해 SBP 형식의 프롬프트가 모델의 학습된 매니폴드(training manifold)에 더 가까워지도록 합니다. 특히, 마진 기반 대조 손실(margin-based contrastive loss)을 사용하여 정답과 오답 간의 구분을 명확히 하는 방법을 채택하였습니다. 이를 통해 SBP 형식의 입력을 훈련할 때 발생하는 분포 이동을 효과적으로 해결하였습니다.

- **Performance Highlights**: 실험 결과, SBP로 미세 조정된 모델은 다중 선택 질문에서 순서에 독립적인 정답률을 크게 향상시키며, CSQA 및 ARC Challenge 데이터셋에 대한 일반화 성능도 개선되었습니다. 또한, 모델이 WikiText-103의 perplexity를 유지함으로써, 보다 넓은 언어 모델링 능력을 저하시키지 않으면서도 안정성을 확보하는 데 성공하였습니다.



### Handling Delay in Real-Time Reinforcement Learning (https://arxiv.org/abs/2503.23478)
Comments:
          Accepted at ICLR 2025. Code available at this https URL

- **What's New**: 본 연구는 실시간 강화 학습(real-time reinforcement learning, RL)에서 발생하는 지연 문제를 해결하기 위한 새로운 접근 방법을 제안합니다. 기존의 N계층 피드포워드 신경망은 레이어 지연(observational delay)로 인해 성능이 저하되는 문제를 나타냅니다. 저자들은 시간 스킵 커넥션(temporal skip connections)과 이력을 기반으로 한 관찰(history-augmented observations)을 결합하여 해결책을 제시하였습니다.

- **Technical Details**: 연구에서는 병렬 계산(parallel computation) 프레임워크를 활용하여, 각 레이어가 비동기적으로 작동하면서도 서로 다른 입력을 처리함으로써 네트워크의 처리를 가속화합니다. 그러나 이러한 병렬 처리 체계에서도 관찰 지연을 해결할 필요가 있으며, 이때 시간 스킵 커넥션을 통해 지연을 줄이는 방향으로 접근합니다. 또한, 다양한 아키텍처를 탐색하여 지연과 네트워크 표현력(expressivity) 간의 트레이드오프(trade-off)를 찾아냅니다.

- **Performance Highlights**: 실험 결과, 시간 스킵 커넥션과 이력 보강 관찰을 포함한 아키텍처가 다양한 RL 과제에서 우수한 성능을 발휘했음을 보여주었습니다. 또한, 병렬 뉴런 계산을 통해 표준 하드웨어에서 추론을 6~350% 가속화할 수 있음을 확인했습니다. 이를 바탕으로 실시간 설정에서 보다 효율적인 RL 에이전트를 위한 길을 열어주었습니다.



### Codehacks: A Dataset of Adversarial Tests for Competitive Programming Problems Obtained from Codeforces (https://arxiv.org/abs/2503.23466)
Comments:
          Accepted for publication at the 18th IEEE International Conference on Software Testing, Verification and Validation (ICST 2025)

- **What's New**: 이 논문에서는 Codeforces 플랫폼에서 자동으로 수집한 실패 유도 테스트 케이스를 기반으로 하는 새로운 데이터셋인 ‘Codehacks’를 소개합니다. 이 데이터셋은 5,578개의 프로그래밍 문제에 대해 288,617개의 해킹 사례를 포함하고 있으며, 각 문제는 자연어 설명과 함께 제공됩니다. Codehacks는 LLM(대형 언어 모델)을 통해 생성된 코드의 품질을 평가하는 데 중요한 자원으로 활용될 수 있습니다.

- **Technical Details**: Codehacks 데이터셋에는 프로그래밍 문제뿐만 아니라, 이러한 문제에 대한 2,196개의 제출 솔루션의 소스 코드도 포함되어 있습니다. 이러한 해킹 기술은 사용자가 제출한 솔루션의 취약성을 발견하기 위한 것으로, 수동으로 만들기에는 비용이 많이 드는 경계 사례 테스트를 제공해 줍니다. 논문에서는 추가 테스트가 필요하다고 강조하며, Codeforces의 온라인 판별 플랫폼이 유용한 자원이라는 점을 지적합니다.

- **Performance Highlights**: Codehacks는 LLM을 사용한 프로그램 합성 기술의 검증과 평가에서 중요한 기여를 할 것으로 기대됩니다. 과거에는 테스트 검증 시 존재하는 허위 부정 결과를 찾는 데 필요한 추가 테스트를 작동시키는 과정이 비용과 시간이 많이 소요되었습니다. Codeforces에서의 해킹 사례를 활용하여, 이러한 부정확한 결과를 줄일 수 있는 효과적인 방법을 제시합니다.



### Semantic-Preserving Transformations as Mutation Operators: A Study on Their Effectiveness in Defect Detection (https://arxiv.org/abs/2503.23448)
Comments:
          Accepted for publication in Mutation 2025 at the 18th IEEE International Conference on Software Testing, Verification and Validation (ICST 2025)

- **What's New**: 이번 연구에서는 결함 탐지(defect detection) 도구의 성능을 개선하기 위해 의미 보존 변환(semantic-preserving transformations)을 사용할 수 있는지 분석했습니다. 기존의 연구들은 의미적으로 동일한 코드에서 모델의 강건성을 향상시키기 위해 훈련 데이터를 강화하는 데 집중했지만, 이러한 코드가 실제 도구 성능 개선에 어떻게 사용될 수 있는지는 잘 알려져 있지 않았습니다. 이를 통해 우리는 LLMs(대형 언어 모델)와 도구의 조합이 기존에 알려진 방식과는 다른 새로운 접근을 제시할 수 있음을 발견했습니다.

- **Technical Details**: 연구진은 28개의 논문에서 94개의 의미 보존 변환을 수집하였으며, 이 중 39개의 변환을 실제로 구현하였습니다. 그러나 수작업 검토 결과 39개 중 23개가 코드 의미를 변경하는 것으로 나타났습니다. 최종적으로 16개의 변환을 사용하여 LLMs를 통해 결함 탐지 도구의 성능을 향상시킬 수 있는지를 실험하였습니다. 연구 과정에서 세 가지 앙상블 기법을 적용하여 성과를 평가하였습니다.

- **Performance Highlights**: 본 연구의 결과, 선택된 16개의 올바른 변환과 세 가지 앙상블 기법을 사용했음에도 결함 탐지 모델의 정확도를 향상시키지 못했습니다. 연구진은 의미 보존 변환을 재사용하는 것이 어렵고, 일부 변환이 의도치 않게 의미를 변경할 수 있음을 발견했습니다. 따라서 향후 연구에서는 이러한 구현의 어려움을 극복하기 위한 방안이 필요하다는 인사이트를 제공합니다.



### Speculative End-Turn Detector for Efficient Speech Chatbot Assistan (https://arxiv.org/abs/2503.23439)
Comments:
          Preprint

- **What's New**: 이번 논문에서는 대화 시스템의 중요한 문제인 end-turn detection (ETD) 문제를 해결하기 위해 ETD Dataset을 출시했습니다. 이 데이터셋은 텍스트 대화 데이터를 기반으로 생성된 합성 음성 데이터와 웹 소스에서 수집된 실제 음성 데이터로 구성되어 있습니다. 또한, 자원 제한 환경에서 실시간 ETD를 개선하기 위한 새로운 협업 추론 프레임워크인 SpeculativeETD를 제안합니다.

- **Technical Details**: SpeculativeETD는 경량의 GRU 기반 모델과 고성능 Wav2vec 기반 모델을 조합하여 효율성과 정확성을 균형있게 유지합니다. 경량 모델은 로컬 디바이스에서 빠르게 비말 단위를 탐지하고, 침묵이 감지되면 고성능 모델에 질의하여 효과적으로 턴의 종료 여부를 판단합니다. 이 접근 방식은 고성능 모델이 실시간으로 작동할 필요가 없으므로 필요한 계산량을 대폭 줄일 수 있습니다.

- **Performance Highlights**: 실험 결과, SpeculativeETD는 ETD 정확도를 크게 향상시키면서도 계산 요구량을 최소화하는 것으로 나타났습니다. 120,000개 이상의 샘플과 300시간 이상의 대화 데이터가 포함된 ETD 데이터셋은 모델 트레이닝 및 평가에 유용하게 활용될 것입니다. 데이터셋과 코드는 리뷰 후 공개될 예정입니다.



### What Makes an Evaluation Useful? Common Pitfalls and Best Practices (https://arxiv.org/abs/2503.23424)
- **What's New**: 최근 몇 년 사이 인공지능(AI)의 발전이 급격히 이루어짐에 따라 AI 커뮤니티에서는 잠재적인 안전 위험에 대한 우려가 커지고 있습니다. 본 논문에서는 AI 시스템의 안전한 사용과 개발을 위한 고품질 평가의 필요성을 강조하며, 이러한 평가를 위한 모범 사례를 제공하고 있습니다. 특히, 사이버 보안 사례를 통해 모델 평가의 모범 사례를 어떻게 정의하고 적용할 수 있는지를 설명합니다.

- **Technical Details**: AI 모델의 평가 설계는 위협 모델링(threat modeling)과 평가 설계를 잇는 초기 사고 과정 단계를 논의하는 것으로 시작됩니다. 또한 유용한 평가의 특성과 파라미터를 제시하고, 특정 평가 구축에서 전체적인 평가 스위트(suite) 구축으로 넘어갈 때 고려해야 할 사항들을 다룹니다. 이를 통해 AI 시스템의 안전 평가를 위한 체계적 접근 방식을 제안하고 있습니다.

- **Performance Highlights**: 이 연구의 주요 기여 중 하나는 결정 과정(decision making processes), 위협 모델링과 평가 설계(threat modeling and evaluation design) 간의 중요한 연결 고리를 수립한 점입니다. 모범 사례에 대한 명확한 원칙을 확립하고 이를 바탕으로 안전 평가를 위한 평가 스위트 구성을 위한 가이드를 제공합니다. 이 논문은 실험적 검증이 향후 중요한 작업이 될 것임을 지적하며, AI 기술의 보다 안전하고 책임 있는 발전에 기여할 것으로 기대됩니다.



### An Analysis of Decoding Methods for LLM-based Agents for Faithful Multi-Hop Question Answering (https://arxiv.org/abs/2503.23415)
- **What's New**: 본 연구에서는 대형 언어 모델(LLM)의 응답의 사실 충실도(faithfulness)를 향상시키기 위한 다채로운 디코딩 전략과 ReAct 프레임워크의 결합이 효과적임을 제안합니다. 특히, ReAct와 검증된 외부 지식을 활용하여 부정확성을 감소시키고, Multi-Hop Question Answering (QA) 과제에서 성능 향상을 이끌어낼 수 있음을 보여주었습니다. 따라서 본 연구는 LLM의 응답에 대한 맥락 적합성 평가에 기여하며, 훈련이 필요 없는 디코딩 방법을 사용하여 더욱 접근 가능하도록 하고 있습니다.

- **Technical Details**: ReAct 프레임워크는 순차적으로 의사결정 및 행동을 수행하는 구조로 이루어져 있으며, LLM이 특정 키워드를 포함한 문서의 첫 문장을 검색하는 도구를 사용하여 정보를 획득합니다. 본 연구에서는 Context-Aware Decoding (CAD), Decoding by Contrasting Layers (DoLa), Decoding by Contrasting Retrieval Heads (DeCoRe)와 같은 디코딩 방법을 분석합니다. 이러한 방법들은 LLM의 차별화된 계층이나 주의 헤드를 활용하여 결과의 사실성을 향상시키는 데 기여합니다.

- **Performance Highlights**: 연구 결과, ReAct와 DoLa를 결합했을 때 HotpotQA에서 F1 점수가 19.5에서 32.6으로 증가하였으며, 답변 인식률(Answer F1)에서 최대 13.1의 향상을 보였습니다. 다양한 데이터셋에서 일관된 성능 향상을 확인했으며, 지원 문서에서 정답이 포함된 비율(Answer Support Recall) 역시 명확하게 개선되었습니다. 이를 통해, 복합적인 질문 응답 과제에서 LLM의 효율성이 극대화될 수 있음을 확인했습니다.



### From Content Creation to Citation Inflation: A GenAI Case Study (https://arxiv.org/abs/2503.23414)
Comments:
          20 pages

- **What's New**: 최근 연구에 따르면, 많은 AI에 의해 생성된 의심스러운 학술 논문들이 Preprint 저장소에 올라가고 있는 것으로 나타났습니다. 이러한 논문들은 인용 조작을 목적으로 하며, 기술적인 깊이나 신뢰성이 부족합니다. 특히, 이들 논문은 반복적인 구조를 가지며, 의심스러운 저자들 사이에서 서로 인용되는 경향이 있습니다.

- **Technical Details**: 연구팀은 GenAI를 사용해 가짜 논문을 생성하고, 의심스러운 논문에 인용을 삽입하여 ResearchGate에 업로드했습니다. 이 실험은 이러한 조작이 얼마나 쉽게 이루어질 수 있는지를 평가하기 위한 것이었습니다. 발견된 논문들은 플랫폼의 검증을 우회하고, 공개적으로 접근 가능하며, 인용 지표를 부풀리는 데 기여할 수 있습니다.

- **Performance Highlights**: 조사 결과, AI 생성 논문은 기존 학술 지침을 우회하는 방식으로 인용 조작이 이루어질 수 있음을 보여주었습니다. 이러한 관행은 ResearchGate와 같은 플랫폼의 약점을 드러내며, 인용 기준이 고용 및 승진 결정에 쉽게 악용될 수 있다는 점이 우려됩니다. 최종적으로, 이 연구는 학술 무결성을 모니터링하기 위한 정책적 권고안을 제안하고 있습니다.



### GMapLatent: Geometric Mapping in Latent Spac (https://arxiv.org/abs/2503.23407)
- **What's New**: 본 연구는 GMapLatent라는 새로운 생성 모델을 소개합니다. 이 모델은 기하학적 매핑에 기반하여 데이터를 변환하고, 서로 다른 도메인의 잠재 공간을 정밀하게 정렬하는 혁신적인 접근 방식을 제공합니다. 이를 통해 모드 붕괴(mode collapse) 및 혼합(mixture) 문제를 예방하여 생성 모델의 일반화 능력을 향상시킵니다.

- **Technical Details**: GMapLatent 모형은 잠재 공간을 정형 파라미터 도메인으로 변환하기 위해 평형 중심 변환(barycenter translation), 최적 수송(optimal transport), 제약 하모닉 매핑(constrained harmonic mapping) 기법을 사용합니다. 이후 전이된 잠재 공간 간의 기하학적 등록(geometric registration)을 수행하여 각 클러스터 간의 엄격한 일치를 보장합니다. 이 과정은 변환된 잠재 공간 간의 일대일(bijective) 매핑을 실현하며, 최종적으로 정밀한 생성을 달성합니다.

- **Performance Highlights**: GMapLatent 모델은 그레이스케일 및 컬러 이미지에 대해 실험을 통해 그 효율성 및 효과성을 입증하였습니다. 기존 모델들에 비해 우수한 성능을 보였으며, 특히 모드 붕괴와 혼합 문제를 해결하는 데 있어 강력한 장점을 가지고 있습니다. 이 연구는 교차 도메인 생성의 새로운 가능성을 열어주며, 딥러닝 분야에서의 적용 가능성을 보여 줍니다.



### Diffusion Meets Few-shot Class Incremental Learning (https://arxiv.org/abs/2503.23402)
Comments:
          pre-print

- **What's New**: 이 논문에서는 적은 샘플로 새로운 클래스를 순차적으로 학습하는 few-shot class-incremental learning (FSCIL) 문제를 해결하기 위하여 Diffusion-FSCIL이라는 새로운 프레임워크를 제안합니다. 이 방식은 텍스트-이미지 확산 모델인 Stable Diffusion을 고정된 백본(backbone)으로 활용하며, 특히 대규모 사전 학습을 통해 얻은 생성 능력을 활용합니다. 기존의 FCIL 방법들이 겪는 기능 불일치(feature misalignment)와 기억 상실(catastrophic forgetting) 문제를 완화하는 데 중점을 두고 있습니다.

- **Technical Details**: Diffusion-FSCIL 방법은 다중 보완적 확산 기능(difussion features)을 추출하여 생성 편향(generative biases)을 방지하는 데 있어 미세 지원을 제공합니다. 또한, 텍스트 인코더(text encoder)를 활용하여 표현의 질을 증가시킬 수 있습니다. 제안된 방법은 모델의 학습 과정에서 기능 추출을 동시에 수행하며, 이를 통해 효율성을 극대화합니다. 마지막으로, 생성된 함수의 평균 구조는 강력한 클래스 분리를 유지하는 데 필요한 새로운 클래스 프로토타입을 학습하는 전략을 도입합니다.

- **Performance Highlights**: Diffusion-FSCIL은 CUB-200, miniImageNet, CIFAR-100 데이터셋을 기반으로 광범위한 실험을 통해 최신의 방법들을 능가하며 과거 학습된 클래스의 성능을 유지하고 새로운 클래스에 효과적으로 적응함을 보여주었습니다. 이 방법은 약 6M의 훈련 가능한 매개변수를 사용하며 빠른 훈련 시간을 자랑하여 다른 최신 모델들과 비교해도 훌륭한 결과를 도출하였습니다. 전체적으로 이 방법은 계산 비용이 낮고 학습 시간이 빨라 효율적인 학습이 가능합니다.



### Scaling Auditory Cognition via Test-Time Compute in Audio Language Models (https://arxiv.org/abs/2503.23395)
- **What's New**: 이 연구는 오디오 대형 언어 모델(Audio LLMs)의 청각 인지(Cognitive) 능력을 테스트하고 향상시키는 혁신적인 접근 방식을 제안합니다. 특히, 다양한 실세계 청각 환경에서의 성능을 평가하고, 오디오 LLM의 청각 처리 능력을 개선하기 위한 테스트 시간 컴퓨트(Test-Time Compute, TTC) 방법론을 소개합니다. 연구 결과에 따르면 오디오 LLMs는 더 어려운 청각 인지 작업에서 확실히 성능이 저하되며, 제안한 TTC 접근법이 이러한 장애물을 극복하는 데 중요한 역할을 합니다.

- **Technical Details**: 연구는 자가 수집된 데이터베이스를 사용하여 5가지 오디오 LLM을 평가했습니다. 각 모델은 기본 청각 인지 작업부터 더 복잡한 작업(예: 중첩되는 목소리 속에서 숫자를 기억하고 회상하는)까지 다양한 작업을 수행하였습니다. 다섯 가지 TTC 접근법이 도입되어 오디오 LLM의 청각 인지 능력을 향상시키는 데 기여했으며, 모델 구조와 작업의 복잡성에 따라 성능 향상이 다르게 나타났습니다.

- **Performance Highlights**: 연구 결과, 모든 모델이 인간의 인식보다 낮은 성능을 보였으나, GPT-4는 중첩된 음성을 인식하는 데 놀라운 성과를 나타냈습니다. 제안된 TTC 방법론은 9%에서 150%까지 다양한 성능 향상을 보여주었으며, 각 접근법의 최적 전략은 모델 구조와 작업의 복잡성에 크게 의존했습니다. 이러한 결과는 오디오 LLM이 실생활 적용(예: 보조 청취 장치, 음성 기반 AI 조수, 통신 기술)에서 더 적응력을 가진 모델로 발전할 수 있는 가능성을 열어줍니다.



### Spatiotemporal Learning of Brain Dynamics from fMRI Using Frequency-Specific Multi-Band Attention for Cognitive and Psychiatric Applications (https://arxiv.org/abs/2503.23394)
- **What's New**: 이번 연구에서는 Multi-Band Brain Net (MBBN)이라는 새로운 변환기 기반 프레임워크를 소개합니다. MBBN은 fMRI 데이터를 통해 뇌의 주파수 특이적 시공간 역학을 모델링하며, 이러한 모델은 scale-free (스케일 자유) 네트워크 원칙과 주파수 해상 멀티밴드 자기 주의(self-attention)를 결합하여 이전에는 탐지되지 않았던 정신적 상태와 관련된 네트워크 상호작용을 파악합니다.

- **Technical Details**: MBBN은 Bidirectional Encoder Representations from Transformers (BERT) 기반으로 설계되어 있으며, 뇌 시그널을 다양한 주파수 대역으로 나누어 기능적 및 위상적 특성을 캡처합니다. 이 모델은 전통적인 정적 상관 기반의 모델의 한계를 극복하고, 뇌 영역 간의 비선형 상호작용과 시간에 따라 변하는 관계를 역동적으로 모델링합니다.

- **Performance Highlights**: MBBN은 최신 방법들보다 최대 30.59% 높은 예측 정확도를 달성하였으며, ADHD(주의력 결핍 과잉 행동 장애) 및 ASD(자폐 스펙트럼 장애)와 같은 정신 건강 조건에서의 연결성 중단을 조명했습니다. 연구는 MBBN이 뇌 기능의 계층적 조직을 이해하는 데 중요한 통찰을 제공하며, 새로운 주파수 특이 바이오마커를 발견하는 데 기여할 수 있음을 보여줍니다.



### Pareto Continual Learning: Preference-Conditioned Learning and Adaption for Dynamic Stability-Plasticity Trade-off (https://arxiv.org/abs/2503.23390)
- **What's New**: 이 논문에서는 지속적 학습(Continual Learning, CL)에서 안정성(stability)과 유연성(plasticity)의 균형을 다루는 새로운 접근 방식을 제안합니다. 기존의 경험 재생(experience replay) 방법들이 고정된 균형을 목표로 했던 반면, 제안된 Pareto Continual Learning (ParetoCL)은 이를 다중 목표 최적화(multi-objective optimization) 문제로 재정의했습니다. 이 방식으로 다양한 무역 오프(balances) 내에서 동적으로 적응할 수 있는 모델을 제공합니다.

- **Technical Details**: ParetoCL은 선호 기반 모델을 도입하여 두 개의 목표인 안정성과 유연성 간의 무역 오프를 학습합니다. 모델은 안정성을 위한 재생 버퍼와 유연성을 위한 새로운 데이터에 대한 손실을 최소화하는 방식으로 훈련됩니다. 추론(inference) 단계에서, 모델은 각 샘플에 대한 최적의 무역 오프를 선택하여 더욱 자신감 있는 예측을 할 수 있게 됩니다.

- **Performance Highlights**: 다양한 데이터셋을 대상으로 한 실험에서 ParetoCL은 기존의 최첨단 지속적 학습 방법보다 뛰어난 성능을 보였습니다. ParetoCL을 통해 입력된 다양한 선호에 의해 안정성과 유연성 간의 무역 오프가 잘 분포된 Pareto 최적 솔루션 세트를 얻을 수 있습니다. 이러한 결과는 데이터 증대(data augmentation) 및 클래스 증대(class augmentation) 관점에서도 중요한 기여를 보여줍니다.



### COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation (https://arxiv.org/abs/2503.23388)
Comments:
          Accepted to CVPR 2025

- **What's New**: 최근 비전-언어 모델(VLMs)인 COSMIC(클릭 지향 의미 다중 공간 통합)는 새로운 도메인에 대한 테스트 시간 적응에 있어 중요한 도전을 해결하기 위한 프레임워크입니다. COSMIC은 다중 분류에서 우수한 적응성을 발휘하며, 세 가지 주요 혁신인 이중 의미 그래프(Dual Semantics Graph)와 클리크 지도 하이퍼 클래스(Clique Guided Hyper-class)에 기반합니다. 이를 통해 혼합된 세미틱 정보를 활용하여 예측의 강인성을 개선하고 있습니다.

- **Technical Details**: COSMIC은 다중 세미틱 캐싱(multi-granular, cross-modal semantic caching)과 그래프 기반 쿼리 메커니즘을 활용하여 모델의 적응성을 증대시킵니다. 이중 의미 그래프(DSG)는 텍스트 특징, 조밀한 CLIP 및 미세 조정된 DINOv2 특징을 통합하여 보강된 의미 공간을 생성합니다. 클리크 지도 하이퍼 클래스(CGH)는 구조화된 클래스 관계를 이용하여 예측 강인성을 높인다.

- **Performance Highlights**: COSMIC은 여러 벤치마크에서 놀라운 성능을 기록하며, 특히 out-of-distribution 태스크에서 15.81%의 향상을 보였습니다. 또한 클립 RN-50을 활용한 크로스 도메인 생성에서도 5.33%의 성능 개선을 이뤘고, 코드가 공개되어 누구나 사용할 수 있습니다. 이러한 결과는 COSMIC의 혁신적인 접근 방식이 효과적임을 잘 보여줍니다.



### KernelDNA: Dynamic Kernel Sharing via Decoupled Naive Adapters (https://arxiv.org/abs/2503.23379)
- **What's New**: 본 연구는 기존 동적 합성곱(dyamic convolution) 기법의 한계를 극복하기 위한 새로운 경량 합성곱 커널 모듈 KernelDNA를 제안합니다. KernelDNA는 입력 의존적인 동적 라우팅(dynamic routing)과 사전 훈련된 정적 모듈레이션(static modulation)을 결합하여 매개변수 효율성(parameter efficiency)과 하드웨어 친화적인 추론(inference)을 보장합니다. 기존 방법들은 매개변수 증가 문제를 겪었지만, 본 연구는 계층 간 가중치 공유(cross-layer weight sharing)를 통해 이러한 문제를 해결합니다.

- **Technical Details**: KernelDNA는 입력 데이터에 따라 동적으로 커널을 조정하는 동시에 사전 훈련된 커널을 재사용하는 메커니즘을 도입합니다. 이를 통해 기존의 정적 합성곱 구조를 유지하면서도 입력에 적응한 커널 조정으로 표현력을 향상시킵니다. 연구에서는 세 가지 향상된 주의 메커니즘인 채널 주의(Channel Attention), 필터 주의(Filter Attention), 공간 주의(Spatial Attention)를 통합하여 각 합성곱 계층이 고유한 특성을 유지하도록 하였습니다.

- **Performance Highlights**: 실험 결과, KernelDNA는 기존의 동적 합성곱 방법들보다 우수한 정확도를 보여주며, 원래의 추론 속도도 거의 유지했습니다. 예를 들어, ResNet18 모델에서 KernelDNA는 1.2-5배의 매개변수 감소와 높은 정확도(74.23%)를 달성했습니다. 경량 모델에서도 뛰어난 성능을 보여주어 다양한 아키텍처에서 매개변수 효율성, 하드웨어 호환성 및 적응 성능의 새로운 조화를 이뤘습니다.



### JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization (https://arxiv.org/abs/2503.23377)
Comments:
          Work in progress. Homepage: this https URL

- **What's New**: 이 논문에서는 JavisDiT라는 새로운 Joint Audio-Video Diffusion Transformer를 소개합니다. 이 모델은 개방형 사용자 프롬프트로부터 고품질의 오디오 및 비디오 콘텐츠를 동시에 생성할 수 있도록 설계되었습니다. 특히, 시각적 및 청각적 요소 간의 동기화를 보장하기 위해 Hierarchical Spatial-Temporal Synchronized Prior (HiST-Sypo) Estimator라는 세밀한 시공간 정렬 메커니즘을 도입했습니다.

- **Technical Details**: JavisDiT는 Diffusion Transformer (DiT) 아키텍처를 기반으로 하며, 오디오 및 비디오 채널이 AV-DiT 블록을 공유하여 고품질의 두 가지 모달리티를 생성합니다. 이 시스템은 Spatio-Temporal Self-Attention, Coarse-Grained Cross-Attention, Fine-Grained Spatio-Temporal Self-Attention Cross-Attention의 세 가지 구조적 블록을 설계하여 구현됩니다. 특히, HiST-Sypo Estimator는 입력 조건 프롬프트에서 글로벌 및 세밀한 시공간 정보를 추출하여 두 채널 간의 동기화를 강화합니다.

- **Performance Highlights**: JavisDiT는 10,140개의 고품질 텍스트 캡션이 있는 비디오로 구성된 새로운 벤치마크인 JavisBench를 통해 기기에서 성능을 입증했습니다. 실험 결과는 JavisDiT가 기존 방법들보다 훨씬 높은 품질의 생성 및 정밀한 동기화를 달성한다는 것을 보여주며, 복잡한 장면 비디오 처리에서 특히 뛰어난 성과를 냈습니다. 이러한 결과는 JAVG 태스크에 대한 새로운 기준을 설정하게 됩니다.



### FeRG-LLM : Feature Engineering by Reason Generation Large Language Models (https://arxiv.org/abs/2503.23371)
Comments:
          Accepted to NAACL 2025 Findings

- **What's New**: 이번 연구에서 제안하는 	extbf{FeRG-LLM}은 80억 개의 파라미터 규모로 자동으로 feature engineering을 수행할 수 있도록 설계된 대형 언어 모델입니다. 두 단계 대화(dialogue)를 통해 머신러닝 작업을 분석하고 새로운 feature를 발견하는 능력을 보유하고 있으며, Chain-of-Thought (CoT) 기능을 활용합니다. 이러한 자동화된 feature 생성 방법은 인적 자원의 부담을 줄이고 기업 환경에 적합한 솔루션을 제공합니다.

- **Technical Details**: FeRG-LLM은 두 단계의 대화 모델을 통해 Llama 3.1 8B 모델을 미세 조정하여 Direct Preference Optimization (DPO)를 통합함으로써 feature 생성을 위한 향상된 근거를 제공합니다. 이는 binary classification 작업을 포함한 여러 데이터셋에서 효과적으로 평가되었으며, 70B 모델과 동등하거나 더 나은 성능을 보였습니다. DPO는 피드백을 받아 모델의 성능을 개선할 수 있는 방법을 제공합니다.

- **Performance Highlights**: 실험 결과 FeRG-LLM은 자원 소모가 적고 추론 시간을 단축하면서도 대부분의 데이터셋에서 Llama 3.1 70B 모델과 유사하거나 더 나은 성능을 달성했습니다. 특히, 분류 작업에서 다른 연구보다 뛰어난 성능을 보였고 회귀 작업에서도 우수함을 입증했습니다. 리소스 제약이 있는 환경에서도 안정적인 feature 생성을 보장하여 머신러닝 성능을 높일 수 있습니다.



### Towards Physically Plausible Video Generation via VLM Planning (https://arxiv.org/abs/2503.23368)
Comments:
          18 pages, 11 figures

- **What's New**: 최근 비디오 확산 모델(Video Diffusion Models, VDMs)은 매우 사실적인 비디오를 생성하는데 유망하지만, 물리 법칙을 잘 이해하지 못해 물리적으로 그럴듯한 비디오를 생성하는 데 어려움을 겪고 있습니다. 본 논문에서는 두 단계로 나누어진 새로운 영상-비디오 생성 프레임워크를 제안하여 물리적 요소를 명시적으로 통합합니다. 첫 번째 단계에서는 비전을 기반으로 한 언어 모델(Vision Language Model, VLM)을 사용하여 물리적 동적을 예측하는 조잡한 동작 경로를 계획하고, 두 번째 단계에서는 이러한 경로를 기반으로 VDM을 통해 비디오를 생성합니다.

- **Technical Details**: 제안된 프레임워크는 VLM을 coarse-level motion planner로, VDM을 fine-level motion synthesizer로 활용하는 두 단계 구조를 가집니다. VLM은 체계적 사고(chain-of-thought)와 물리적 사고를 결합하여 물리적 동적을 대략적으로 따르는 동작 경로를 생성하도록 합니다. 생성된 동작 경로에 따라 VDM은 일반적인 물리 법칙에 부합하는 세밀한 동작을 생성합니다. 노이즈를 추가하는 방식으로 세부 동작을 생성하여, 고유한 속도, 가속도 등의 정보를 효과적으로 모델링할 수 있습니다.

- **Performance Highlights**: 제안된 프레임워크는 주요 물리 기반 비디오 벤치마크에서 우수한 성능을 보여주었으며, 물리적으로 그럴듯한 동작을 생성하는 데 성공적이었습니다. 기존 비디오 생성 방법들과 비교했을 때, 제안된 방법은 더욱 뛰어난 결과를 기록하며, 사용자 연구를 통해 일반화 가능성과 효과성을 입증하였습니다. 이러한 성과는 VLM과 VDM을 활용하여 물리적으로 그럴듯한 비디오 생성의 가능성을 한층 더 높였음을 보여줍니다.



### Mixture of Routers (https://arxiv.org/abs/2503.23362)
Comments:
          10 pages,4 figures

- **What's New**: 본 논문에서는 파라미터 효율적인 미세조정(PET-Fine Tuning, PEFT)과 Mixture-of-Experts (MoE)를 결합한 새로운 미세조정 방법인 Mixture of Routers (MoR)를 제안합니다. MoR은 여러 서브 라우터(sub-router)를 사용하여 전문가 모델(expert model)을 선택하고, 메인 라우터(main router)가 이들의 선택을 결정하여 모델의 성능을 개선합니다. MoR은 다양한 복잡성을 가질 수 있는 작업에 유연하게 적용 가능하며, 기존 MoE 모델의 라우터 레이어를 대체할 수 있는 플러그 앤 플레이 솔루션입니다.

- **Technical Details**: MoR은 여러 개의 서브 라우터(sub-router)를 통해 공동 선택을 수행하며, 각 서브 라우터의 기여로 최종 결정을 내리도록 설계되었습니다. 여기서 메인 라우터(main router)가 서브 라우터의 점수를 기반으로 상위 서브 라우터를 선택하여, 오류를 최소화할 수 있도록 합니다. 이러한 방식은 각 전문가 선택을 위한 점수를 조정하여, 최종 추론 단계에서 가장 우수한 전문가를 선택합니다.

- **Performance Highlights**: 여섯 가지 벤치마크에서 실험을 수행한 결과, MoR은 대부분의 작업에서 기존 모델에 비해 평균 1%의 성능 향상을 보였습니다. MoR은 다양한 NLP 및 상식 추론(Common Sense Reasoning, CR) 작업에서 우수한 성능을 나타내며, 최적의 경량화 미세조정 솔루션으로 자리매김할 수 있음을 보여줍니다. 또한, Consistent Routing Weighting (CRW)이라는 변형을 통해 전이 학습에서의 불안정성을 개선하여 모델의 안정성과 일반화 능력을 효과적으로 향상시켰습니다.



### Object Isolated Attention for Consistent Story Visualization (https://arxiv.org/abs/2503.23353)
Comments:
          6 pages, 4 figures

- **What's New**: 이 논문에서는 이야기 시각화(Story Visualization)에서 등장인물의 일관성을 유지하면서 자연스럽고 맥락에 맞는 장면을 생성하기 위해 강화된 Transformer 모듈을 제안합니다. 새로운 접근 방식을 통해 기존 방법들이 겪는 여러 문제를 해결하며, 특히 훈련 없이도 새로운 캐릭터와 스토리를 생성할 수 있는 가능성을 제시합니다. 이를 위해 분리된 self attention과 cross attention 메커니즘을 활용하여 논리적인 장면 생성을 보장합니다.

- **Technical Details**: 제안된 방법은 isolated self attention과 isolated cross attention 메커니즘을 사용하여 각 캐릭터의 특성을 독립적으로 처리합니다. 이러한 방식은 주의 맵을 정제하여 불필요한 영역에 대한 주의를 줄이고 특정 캐릭터의 주요 특성에 집중할 수 있도록 합니다. 또한, 상호 주의 메커니즘을 통해 서로 다른 캐릭터 간의 피처 혼합을 방지하며, 일관성을 강화합니다.

- **Performance Highlights**: 정성적(qualitative) 및 정량적(quantitative) 평가에서 본 방법이 현재의 기술들을 초과하는 성능을 보이며, 테스트 결과에서 시각적으로 일관되고 응집력 있는 스토리 시각화를 달성하는 효율성을 보여줍니다. 기존의 방법들보다 개선된 결과를 제공하면서도 재조정 없이 연속적인 캐릭터와 스토리라인 생성을 가능하게 하는 등 실용성이 높은 결과를 나타냈습니다.



### Beyond Unimodal Boundaries: Generative Recommendation with Multimodal Semantics (https://arxiv.org/abs/2503.23333)
- **What's New**: 이번 논문에서는 Multimodal Generative Recommendation (MGR)이라는 새로운 접근 방식을 제안합니다. 기존의 Generative Recommendation (GR) 방법들이 주로 unimodal 데이터를 사용하였던 한계를 극복하여, 다양한 모달리티를 통합하는 방법론을 다룹니다. 저자들은 특히 모달리티 선택의 중요성과 그것이 GR 모델의 성능에 미치는 영향을 강조하고 있습니다.

- **Technical Details**: MGR-LF++라는 새로운 Late Fusion 프레임워크를 도입하여, 서로 다른 모달리티 정보를 효과적으로 관리하는 방법을 제안합니다. 이 프레임워크는 contrastive modality alignment 훈련 기법과 각 모달리티를 구분하는 특별한 토큰을 사용하여, 서로 다른 semantic IDs의 일치를 도모합니다. 이를 통해, 다양한 모달리티의 정보를 손실 없이 통합할 수 있는 방법을 모색합니다.

- **Performance Highlights**: MGR-LF++는 기존의 unimodal 접근 방법 대비 20% 이상의 성능 향상을 달성하였습니다. 저자들은 6개의 기준선 모델을 사용하여 3개 데이터셋에서 실험을 실시하였으며, 그 결과 다중 모달리티 정보를 활용하는 것이 Generative Recommendation의 효과를 크게 향상시킬 수 있음을 입증했습니다.



### SalesRLAgent: A Reinforcement Learning Approach for Real-Time Sales Conversion Prediction and Optimization (https://arxiv.org/abs/2503.23303)
- **What's New**: 이번 논문에서는 SalesRLAgent라는 새로운 프레임워크를 제안합니다. 이 시스템은 전문화된 Reinforcement Learning (RL)을 활용하여 판매 대화 전반에 걸쳐 전환 확률(conversion probability)을 예측합니다. 기존의 LLM 기반 접근법과는 달리 SalesRLAgent는 전환 예측을 순차적 결정 문제로 처리하며, 이는 판매 전략에 대한 보다 실시간의 통찰력을 제공합니다.

- **Technical Details**: SalesRLAgent는 Azure OpenAI 임베딩(3072 차원)과 Meta-learning 기능을 결합하여 자신의 지식 한계를 이해합니다. 이 시스템은 판매 대화의 복잡한 역학을 반영하기 위해 GPT-4O를 이용해 생성된 합성 데이터(synthetic data)를 통해 훈련됩니다. 대화의 모든 턴을 추적하면서 전환 확률을 지속적으로 추정합니다.

- **Performance Highlights**: 실험 결과 SalesRLAgent는 96.7%의 전환 예측 정확도를 달성하여 LLM 전용 접근법보다 34.7% 더 우수한 성능을 보였습니다. 또한, 기존의 판매 플랫폼과 통합했을 때, 실시간 가이드를 사용하는 경우 43.2%의 전환율 증가를 나타내었습니다. 이는 SalesRLAgent가 판매 대화의 내용을 생성하는 데 그치지 않고, 전략적인 판매 인텔리전스를 제공하는 데 중점을 두고 있음을 보여줍니다.



### Two Heads Are Better than One: Model-Weight and Latent-Space Analysis for Federated Learning on Non-iid Data against Poisoning Attacks (https://arxiv.org/abs/2503.23288)
- **What's New**: 이 논문에서는 GeminiGuard라는 새로운 접근 방식을 통해 Federated Learning(FL)에서의 모델 오염 공격(Model Poisoning Attacks, MPA)에 대한 효과적인 방어책을 제안합니다. 기존 방어 기법들은 일반적으로 데이터가 iid(독립 동등 분포)일 것으로 가정하고 설계되었으나, 실제 FL 환경에서는 데이터가 비iid(비독립 비동등 분포)인 경우가 많습니다. GeminiGuard는 모델 가중치 분석(Model-weight analysis)과 잠재 공간 분석(Latent-space analysis)을 결합하여 다양한 비iid 시나리오에서 MPA를 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: GeminiGuard는 경량화(lightweight), 다목적(versatile), 비지도 학습(unsupervised)을 목표로 하고 있으며, 이를 통해 다양한 MPA와 비iid 환경에 대한 방어 성능을 향상시킵니다. 모델 업데이드를 필터링하는 초기 단계에서 모델 가중치 기반 분석을 수행하며, 이는 코사인 유사도(Cosine similarity)와 유클리드 거리(Euclidean distance)를 기반으로 합니다. 또한, 잠재 공간 분석 모듈은 여러 레이어(layer)에서 활성화의 평균 거리를 측정하여 수신된 모델 업데이트의 신뢰성을 평가합니다.

- **Performance Highlights**: 실험 결과, GeminiGuard는 다양한 비iid 시나리오에서 기존 방어 기법들에 비해 일관되게 더 높은 성능을 보였습니다. 예를 들어, CIFAR-10에서 IBA 공격(IBA attack) 하에 GeminiGuard는 0.18%의 공격 성공률(Attack Success Rate, ASR)을 기록했으며, 기존 방어 기법은 10%를 초과하는 공격 성공률을 보였습니다. 실험은 총 네 가지 비목표 MPA와 다섯 가지 백도어 공격(backdoor attacks)을 포함한 방어 성능을 평가하였으며, GeminiGuard는 SOTA(State-of-the-Art) 방어법들에 비해 뛰어난 성능을 보여주었습니다.



### Extracting Patient History from Clinical Text: A Comparative Study of Clinical Large Language Models (https://arxiv.org/abs/2503.23281)
- **What's New**: 이번 연구에서는 환자의 주요 불만(Chief Complaint, CC), 현재 병력(History of Present Illness, HPI), 과거 및 가족의 사회적 병력(Past, Family, and Social History, PFSH)과 관련된 의료 기록 엔터티(Medical History Entities, MHEs)를 추출하는 방식에 대해 발표하였습니다. 이는 비정형화된 임상 노트를 체계적인 전자 건강 기록(Electronic Health Records, EHRs)으로 변환하여 의료 제공의 연속성, 의료 코딩, 품질 지표 등의 downstream 작업을 효율화하는 데 기여합니다. 연구에서는 최신 임상 대형 언어 모델(Fine-tuned Clinical Large Language Models, cLLMs)을 활용하여 이러한 MHE를 인식하고, 노트 특성이 모델의 정확도에 미치는 영향을 분석하였습니다.

- **Technical Details**: 연구팀은 MTSamples 리포지토리에서 61개의 외래 환자 관련 임상 노트로부터 1,449개의 MHE를 주석 처리하고, 인지 모델을 인식하기 위해 7개의 최첨단 cLLMs를 파인튜닝(Fine-tuning) 하였습니다. 추가로, 문제, 테스트, 치료 및 기타 기본 의료 엔터티(Basic Medical Entities, BMEs)를 통합하여 모델 성능을 평가하였습니다. 실험은 zero-shot 설정에서 GPT-4o와 비교하여 이루어졌으며, 텍스트 특성이 모델의 정확도에 미치는 영향에 대한 오류 분석도 수행되었습니다.

- **Performance Highlights**: 연구 결과, cLLMs는 MHE 추출에 필요한 시간을 20% 이상 단축시킬 잠재력을 보여주었습니다. 그러나 다의성(polysomy)의 특성과 비의료 용어의 빈번한 사용으로 인해 MHE 탐지에서 여전히 어려움이 존재했습니다. 특히, GatorTron과 GatorTronS 두 가지 모델이 가장 높은 성능을 발휘했으며, 사전에 식별된 BME 정보 통합이 특정 엔터티에 대한 모델 성능 향상에 기여했습니다. 또한, 텍스트 길이, 엔터티 길이, 세분화와 같은 특성이 모델 성능에 미치는 영향에 대한 유의미한 결과가 도출되었습니다.



### Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions (https://arxiv.org/abs/2503.23278)
- **What's New**: MCP(모델 컨텍스트 프로토콜)는 AI 모델과 외부 도구 간의 상호작용을 표준화하여 데이터 사일로를 해소하고 다양한 시스템 간의 상호 운용성을 촉진하는 인터페이스입니다. 본 논문에서는 MCP의 핵심 구성 요소, 워크플로우, 서버 생애 주기의 세 가지 주요 단계인 생성, 운영 및 업데이트를 포괄적으로 다룹니다. 각 단계와 관련된 보안 및 개인정보 위험을 분석하고, 이를 완화하기 위한 전략을 제안합니다.

- **Technical Details**: MCP는 언어 서버 프로토콜(LSP)에 영감을 받아 만들어진 일반 목적의 프로토콜로, AI 애플리케이션이 외부 도구와 동적으로 통신할 수 있는 유연한 프레임워크를 제공합니다. 개발자들은 각 서비스에 대한 인터페이스를 수동으로 정의하고 인증을 관리해야 했지만, MCP는 이러한 복잡성을 줄이고 통합 절차를 표준화합니다. MCP는 특히 AI 에이전트가 작업 맥락에 따라 도구를 자율적으로 탐색하고 선택할 수 있도록 지원합니다.

- **Performance Highlights**: MCP는 출시 이후 빠르게 성장하며, GitHub, Slack, Blender와 같은 시스템에 모델 접근성을 제공하는 수천 개의 커뮤니티 기반 MCP 서버를 활용하는 생태계를 형성했습니다. MCP를 채택한 다양한 기업과 플랫폼의 사례는 그 가능성을 잘 보여주며, 현재도 많은 연구 및 개발의 기회가 남아 있습니다. MCP의 채택은 아직 초기 단계에 있지만, 보안, 도구 발견 및 원격 배포와 같은 문제에 대한 해결책이 마련될 경우, 더 큰 발전이 기대됩니다.



### Improved Ear Verification with Vision Transformers and Overlapping Patches (https://arxiv.org/abs/2503.23275)
- **What's New**: 귀 인식은 성인 동안 상대적으로 안정된 외모로 인해 유망한 생체 인식 방식으로 떠오르고 있습니다. 이 연구에서는 Vision Transformers (ViTs) 모델의 다양한 설정을 귀 인식에 적용하였으며, 겹치는 패치를 선택하는 전략을 사용하여 실험하였습니다. 겹치는 패치의 중요성이 입증되었고, 이는 48개의 실험 중 44개에서 우수한 성능을 나타내었습니다.

- **Technical Details**: 연구에서는 ViT-Tiny (ViT-T), ViT-Small (ViT-S), ViT-Base (ViT-B), ViT-Large (ViT-L)의 설정을 사용하여 OPIB, AWE, WPUT, EarVN1.0 데이터셋에서 실험하였습니다. 겹치는 패치를 사용함으로써 귀의 복잡한 특징을포착할 수 있었고, 귀 인식의 모델 성능에서 ViT-T 모델이 다른 모델들보다 지속적으로 우수한 성과를 보였습니다. 특히 패치 크기와 보폭을 설정하는 방법이 인식 성능에 중요한 영향을 미쳤습니다.

- **Performance Highlights**: 연구 결과, EarVN1.0 데이터셋에서 겹치는 패치를 사용한 경우 성능이 최대 10% 증가하였으며, ViT-T 모델이 AWE, WPUT, EarVN1.0 데이터셋에서 가장 높은 성과를 기록했습니다. 최적의 성능은 패치 크기 28x28과 보폭 14픽셀의 설정에서 달성되었습니다. 이 연구는 겹치는 패치를 선택한 transformer 아키텍처가 귀 생체 인식 작업에 효과적이고 높은 성능을 발휘할 수 있음을 확인시켜 줍니다.



### Learning Coordinated Bimanual Manipulation Policies using State Diffusion and Inverse Dynamics Models (https://arxiv.org/abs/2503.23271)
Comments:
          Project Page: this https URL. 12 pages, 12 figures, Accepted at ICRA 2025

- **What's New**: 새로운 논문에서는 로봇이 세탁과 같은 양손 작업을 수행할 때, 인간이 사용하는 예측적 조작 전략을 모방하여 두 손의 효과적인 협력을 구현하고 있다. 특히, 작업 관련 상태 전환을 분리하여 교차 모드(dynamic) 행동을 더 효과적으로 수행할 수 있게 하는 방법을 제안하였다. diffusion 모델을 사용하여 과거 데이터를 기반으로 미래 상태를 예측하고, 이를 통해 로봇의 움직임을 생성하는 새로운 모델링 기법이 소개되었다.

- **Technical Details**: 이 접근법은 두 가지 주요 모델로 구성된다: (1) 미래 상태를 예측하기 위한 diffusion 기반 상태 예측 모델, (2) 예측된 정보를 바탕으로 로봇 행동을 결정하는 역동력 모델이다. DDPM(Denoising Diffusion Probabilistic Model)을 변형하여 과거 시점의 노이즈가 있는 상태 데이터를 사용하여 현재 상태를 복원하는 방식을 채택하고, 반복적인 디노이징을 통해 최종 상태에 접근할 수 있도록 한다.

- **Performance Highlights**: 우리는 다양한 시뮬레이션 및 실제 환경에서의 조작 작업에서 이 프레임워크를 평가하였고, 기존의 최첨단(state-of-the-art) 방식보다 뛰어난 성능을 발휘함을 확인하였다. 특히 양손 조작 과제에서 우수한 성능을 보여주었으며, 다양한 목표 구성에서의 복잡한 작업 수행 능력이 탁월한 것으로 나타났다.



### Localized Graph-Based Neural Dynamics Models for Terrain Manipulation (https://arxiv.org/abs/2503.23270)
- **What's New**: 이번 논문에서는 로봇이 건설 현장과 외계 표면에서 잘 작업할 수 있도록 terrain dynamics modeling 및 manipulation을 위한 학습 기반 접근법을 소개합니다. Graph-based Neural Dynamics (GBND) 프레임워크를 활용하여, 다차원 terrain 변형을 효과적으로 예측하고 제어하는 방법이 제안되었습니다. 특히 이 방법은 로봇의 제어 입력과 현재 장면을 기반으로 한 작은 관심 영역(Region of Interest, RoI)을 동적으로 선택하여 그 안의 입자들만을 사용하여 예측 효율성을 크게 향상시킵니다.

- **Technical Details**: 이 연구의 핵심은 거대한 크기의 terrain을 더 작은 활성 서브그래프로 제한하여 그 안의 동역학을 예측하는 것입니다. 이 과정에서 그래픽 카드 메모리에 맞지 않는 수백만 개의 입자들로 구성된 대규모 terrain 그래프를 구축하였습니다. 또한, GBND와 RoI를 동시에 학습하는 방법을 통해 로봇-terrain 상호작용 중에 고정된 입자가 정적이라는 가정을 바탕으로 예측 속도를 현저히 향상시킴과 동시에 더 높은 예측 정확도를 달성하였습니다.

- **Performance Highlights**: 제안된 방법은 기존의 GBND 방법보다 수 배 빠르며, 다양한 재료로 구성된 terrain 조작 작업에 대한 실험을 통해 그 효율성과 효과성을 검증하였습니다. 이 연구결과는 건설 산업과 우주 탐사에서 로봇의 자율성과 효율성을 크게 향상시킬 것으로 기대됩니다. 또, 샘플링 복잡성을 줄이면서도 높은 예측 정확도를 유지하여 기존의 물리 기반 방법에서 직면하는 문제를 효과적으로 완화했습니다.



### FIESTA: Fisher Information-based Efficient Selective Test-time Adaptation (https://arxiv.org/abs/2503.23257)
- **What's New**: 이 논문은 얼굴 표정 인식(facial expression recognition) 분야에서, 비구속적인 환경에서의 도메인 변화(domain shift) 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 기존의 테스트 시간 적응(test-time adaptation; TTA) 방식은 매개변수 업데이트의 수동 선택에 의존하는데, 이는 효율성을 저하시킬 수 있습니다. 본 연구에서는 Fisher 정보(Fisher information)를 기반으로 한 선택적 적응(selective adaptation) 프레임워크를 도입하여, 가장 중요한 매개변수만을 동적으로 업데이트합니다.

- **Technical Details**: 제안된 Fisher 기반 선택적 적응 기법은 비디오 기반 얼굴 표정 인식에 적합하게 설계되었습니다. 이 방법은 매개변수 중요도를 Fisher 점수(Fisher scores)로 정량화하고, 이를 통해 모델 성능에 중요한 가중치만을 선택적으로 업데이트합니다. 또한, 이 과정은 시간적 일관성(temporal consistency) 제약과 결합되어 모델의 적응 과정을 보다 효율적이고 효과적으로 만듭니다.

- **Performance Highlights**: AffWild2 벤치마크 데이터세트에 대한 실험 결과, 제안된 접근 방식이 기존 TTA 방법보다 7.7% 향상된 F1 점수를 기록하며, 22,000개의 매개변수만을 업데이트하는 것으로 확인되었습니다. 이는 기존의 방법들보다 20배 이상 적은 매개변수를 사용하는 결과입니다. 또한, 최소한의 데이터(1-3 프레임)로부터 매개변수의 중요도를 효과적으로 추정할 수 있어, 실제 애플리케이션에서 TTA를 더욱 실용적으로 만들어 줍니다.



### Encrypted Prompt: Securing LLM Applications Against Unauthorized Actions (https://arxiv.org/abs/2503.23250)
- **What's New**: 이 논문은 Encrypted Prompt를 사용자 프롬프트에 추가하는 새로운 방법을 제안하여, 권한을 안전하게 검증하고 LLM의 행동이 지정된 권한 범위 내에서만 실행될 수 있도록 보장합니다. 이를 통해 기존의 보안 전략과는 달리, 권한이 부족한 경우 LLM의 작업이 실행되지 않도록 하여 보안을 크게 강화할 수 있습니다. 따라서, Prompt injection 공격에 대한 효과적인 방어 수단을 제공함으로써 LLM 통합 애플리케이션의 보안성을 높입니다.

- **Technical Details**: Encrypted Prompt는 세 가지 구성 요소로 이루어져 있습니다: 구분자(<D> 및 </D>), 현재 권한을 나타내는 Permission(<P>), 그리고 검증을 위한 공개키(<PK>)입니다. 이러한 구조를 통해 서버는 사용자 입력을 처리하기 전에 Encrypted Prompt에서 권한을 확인하고, LLM이 생성하는 각 행동이 허용된 범위 내에 있는지를 판단합니다. 만약 권한이 초과되면, 개발자가 정의한 대로 작업을 거부하거나 추가 확인을 요청할 수 있습니다.

- **Performance Highlights**: 이 새로운 접근법은 기존의 OS 수준 권한 관리 시스템에 비해 애플리케이션 레벨에서 보다 쉽게 구현할 수 있으며, 다양한 권한 규칙을 적용할 수 있는 유연성을 제공합니다. 이를 통해 권한을 동적으로 조정하고 사용자와의 상호작용에 따라 적절한 행동을 수행하게 됩니다. Encrypted Prompt는 추가적인 모델 학습 없이 빠르고 간편하게 다양한 플랫폼에 적용할 수 있어, 사용자에게 보다 안전한 시스템을 제공합니다.



### Simulation of Non-Ordinary Consciousness (https://arxiv.org/abs/2503.23245)
Comments:
          16 pages, 9 figures, 1 table

- **What's New**: 이 논문에서는 비정상적 의식 상태에 대한 상징적 구조를 탐구하는 AI 시스템인 Glyph를 소개합니다. Glyph는 전통적인 모델과는 다르게, 환각 유발 상태가 제공하는 독특한 상징적 경험을 다룹니다. 이를 통해, 심볼릭 인지(Symbolic cognition)를 모사하고 위상 변형을 통한 상징적 변환을 수행하도록 설계되었습니다.

- **Technical Details**: Glyph의 작동 방식은 세 가지 주요 기법으로 구성됩니다: Recursive Reentry (재귀적 재입회), Metaphoric Transformation (비유적 변환), Symbolic Destabilization (상징적 불안정화)입니다. 이 시스템은 고차원 언어 모델에서의 상징적 흐름을 수용하는 텐서적 수학 프레임워크를 통해 정의됩니다. 각 변환은 입력을 바탕으로 새로운 상징적 결과를 생성하도록 설정되어 있습니다.

- **Performance Highlights**: 실험 결과, Glyph는 다양한 상징적 프롬프트 카테고리에서 높은 엔트로피와 비유가 풍부한 언어를 생성하는 데 성공했습니다. 이러한 결과는 비정상적 인지 패턴의 출현을 입증하며, 언어를 통한 변형된 의식의 새로운 시뮬레이션 패러다임을 지지합니다. Glyph는 상징적 인지의 모델링, 비유 이론 탐구 및 재귀적으로 변화된 의미 공간에서의 지식 인코딩에 새로운 경로를 제공합니다.



### Evaluating how LLM annotations represent diverse views on contentious topics (https://arxiv.org/abs/2503.23243)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 데이터 라벨링에 활용하는 방법을 제시하고, 이 모델들이 기존 자연어 모델에 비해 어떻게 성능이 개선되었는지를 강조합니다. 많은 기존 문헌에서는 LLM이 정확도, 정밀도, 재현율 및 F1 점수와 같은 표준 지표에서 다른 모델보다 더 우수하다고 언급하고 있습니다. 그러나 LLM의 언어 모델에 내재된 편향 문제도 조명되고 있으며, 특히 논란이 많은 주제와 관련된 부분에서 이러한 편향이 문제가 될 수 있습니다.

- **Technical Details**: LLMs의 성능 평가를 위해 연구팀은 NLPositionality 데이터셋, POPQUORN의 공격성 및 공손성 데이터셋, 위키피디아 댓글 데이터셋을 사용하였습니다. 연구는 성별, 인종, 교육 수준과 같은 인구통계학적 그룹에 따른 LLM의 동의 여부를 분석합니다. 공정성을 보장하기 위해 세 가지 다른 프롬프트를 사용하여 각 데이터셋에 대해 레이블을 생성하였으며, LLM의 동의 수준은 개인 라벨러와의 일치도를 사용하여 측정했습니다.

- **Performance Highlights**: 결과적으로, LLM은 인구통계학적 기준에 따라 평가자와의 상당한 불일치를 보이지 않았습니다. 오히려 라벨링 작업의 난이도와 모델, 그리고 프롬프트 사용이 LLM 합의에 더 큰 영향을 미쳤습니다. 연구 결과, 특정 집단의 의견이 과소 대표되는 문제는 크지 않은 것으로 나타났으며, 이러한 동의 수준은 데이터셋에 따라 다르기 때문에 LLM의 편향 문제는 LLM 자체의 문제가 아닐 수 있음을 시사하고 있습니다.



### Beyond speculation: Measuring the growing presence of LLM-generated texts in multilingual disinformation (https://arxiv.org/abs/2503.23242)
- **What's New**: 본 연구는 대형 언어 모델(LLM)의 발전과 이로 인해 생성된 다국어 텍스트의 품질이 높아짐에 따라 발생하는 허위정보(disinformation) 사용 가능성에 대한 우려를 다룹니다. 연구는 LLM이 최근 허위정보 데이터셋에 존재한다는 최초의 실증적 증거를 제공하며, ChatGPT의 출시 이후 기계에 의해 생성된 콘텐츠의 증가를 문서화합니다.

- **Technical Details**: 연구는 LLM에 의해 생성된 콘텐츠의 비율, 다양한 언어와 플랫폼, 시간대에 따른 패턴을 분석합니다. 이러한 분석은 자연 생태계의 한계에도 불구하고 특정 '롱테일(longtail)' 컨텍스트가 간과된 위험에 직면해 있음을 지적합니다.

- **Performance Highlights**: 연구 결과를 통해, LLM이 생성하는 텍스트가 인간이 작성한 텍스트와 구별하기 어렵다는 것을 보여주며, 이는 허위정보의 확산에 중요한 영향을 미칠 수 있습니다. 또한, 기계 생성 콘텐츠의 증가가 특정 플랫폼과 언어에서 두드러지며, 연구자들에게 이와 관련된 주의를 촉구합니다.



### CCCI: Code Completion with Contextual Information for Complex Data Transfer Tasks Using Large Language Models (https://arxiv.org/abs/2503.23231)
Comments:
          The 29th International Conference on Evaluation and Assessment in Software Engineering

- **What's New**: 본 연구에서는 데이터 전송 작업을 위해 설계된 새로운 코드 완성 방법인 CCCI(Contextual Code Completion Initiative)를 소개합니다. CCCI는 대형 언어 모델(LLMs)에 데이터베이스 테이블 관계, 객체 모델 및 라이브러리 세부정보와 같은 맥락 정보를 통합하여 코드 완성의 정확성을 향상시킵니다. 이 접근 방식은 코드 완성의 효과를 높이기 위해 다양한 LLM을 결합하여 프로젝트 내에서 복잡한 종속성을 해결할 수 있습니다.

- **Technical Details**: CCCI 방법론은 289개의 Java 코드 조각을 분석하여 성능을 평가하였으며, 이는 819개의 운영 스크립트에서 추출되었습니다. 연구 결과, CCCI는 49.1%의 Build Pass 비율과 41.0%의 CodeBLEU 점수를 기록하여 복잡한 작업 완성을 어려워하는 최신 방법들과 견주어 유사한 성과를 보였습니다. 이를 통해 연구진은 CCCI가 기존 LLM의 성능을 142.6% 향상시킬 수 있음을 입증하였습니다.

- **Performance Highlights**: CCCI는 다양한 오픈 소스 및 폐쇄형 LLM에서 성능을 평가하였으며, GPT-4o 모델에 비해 현저한 개선을 보여주었습니다. 모델 성능 개선의 주요 요인은 프로젝트 내 맥락 정보에 대한 통합된 접근 방식으로, 이는 코드 완성의 실제 유용성을 증가시킬 것으로 기대됩니다. 전체적으로 CCCI는 데이터 전송 작업에 필요한 맥락적 정보의 중요성을 강조하며 개발자의 효율성을 높이는 데 기여할 것으로 보입니다.



### Synthetic Art Generation and DeepFake Detection A Study on Jamini Roy Inspired Datas (https://arxiv.org/abs/2503.23226)
Comments:
          13 pages, 7 figures, 6 tables

- **What's New**: 이번 연구는 생성 AI와 예술의 교차점에서 발생하는 도전과 기회를 탐구합니다. 특히, 인도의 화가 자미니 로이(Jamini Roy)의 독특한 스타일을 중심으로 한 확산 기반 생성 모델을 조사하고 있습니다. 이를 위해 Stable Diffusion 3를 조정하여 세밀한 이미지를 생성하고, 실제와 AI 생성 작품이 혼합된 새로운 데이터셋을 구축했습니다.

- **Technical Details**: 연구에서는 먼저 생성된 이미지와 진품 이미지 간의 미세한 차이를 발견하기 위해 푸리에 영역 평가(Fourier domain assessments) 및 자기상관 메트릭(autocorrelation metrics)과 같은 정성 및 정량적 방법을 사용했습니다. 특히 생성 기술의 발전에 따라 기존의 심층 AI 이미지 분석에서의 단점을 해결할 방안을 모색하고 있습니다. 이 모델들이 문화적 맥락에서 생성된 작품의 독특한 특성을 포착할 수 있는지에 대한 질문이 제기되고 있습니다.

- **Performance Highlights**: 연구의 최종 목표는 생성된 예술 작품의 신뢰성을 감별하는 데 있어 새로운 접근 방식을 제공하는 것입니다. 자미니 로이 스타일의 작품을 포함하는 이 새로운 데이터셋은 기존 기술의 한계를 극복할 수 있는 기회를 제시합니다. 저자들은 현재의 진품 탐지 기술이 고품질의 문화적으로 특정한 딥페이크를 식별하기에 어려움이 있다는 점을 강조하며, 이는 예술의 진위성을 보호하기 위한 중요한 연구 방향이 될 것입니다.



### Aurelia: Test-time Reasoning Distillation in Audio-Visual LLMs (https://arxiv.org/abs/2503.23219)
- **What's New**: 이 논문에서는 오디오-비주얼 레이징(AV reasoning)의 복잡성을 해결하기 위해 새로운 프레임워크인 AURELIA를 소개합니다. AURELIA는 actor-critic (행동-비평자) 기반의 접근 방식을 활용하여, 테스트 시간에 AVLLMs(오디오-비주얼 대형 언어 모델)의 단계적 레이징 기능을 증진합니다. 논문은 또한 4500개의 오디오-비주얼 질문과 이에 대한 상세한 단계적 레이징을 제공하는 AVReasonBench 기준점을 제시합니다.

- **Technical Details**: AURELIA는 LLM(대형 언어 모델)의 추론 기능을 활용하여 다중 모드 오디오-비디오 이해를 위한 고품질 레이징 데이터를 생성하는 인터랙티브한 프레임워크입니다. 이 시스템은 오디오와 비주얼 신호의 상호 작용을 고려하여 각기 다른 레이징 작업을 수행하며, 기존의 AVLLMs가 가진 편향성을 줄이는 데 기여합니다. 또한, AVReasonBench는 다양한 레이징 능력을 평가하는 데 필요한 포괄적인 벤치마크를 제공합니다.

- **Performance Highlights**: AURELIA를 활용하여 AVLLMs의 성능이 최대 100% 상대적으로 개선되는 것을 확인하였습니다. AVReasonBench의 18개 기존 AVLLM에 대한 평가 결과, 이 모델들이 비디오와 오디오 데이터를 처리하는 데 현저한 한계를 보임을 보여주었습니다. 이러한 성능 증가는 오디오-비주얼 레이징의 중요성과 현실 세계에서의 응용 가능성을 강조합니다.



### Action Recognition in Real-World Ambient Assisted Living Environmen (https://arxiv.org/abs/2503.23214)
- **What's New**: 본 논문은 Robust and Efficient Temporal Convolution Network (RE-TCN)을 제안하며, 이는 Ambient Assisted Living (AAL) 기술에서의 행동 인식 문제를 해결하기 위해 다양한 기술적 요소를 활용합니다. 이 모델은 Adaptive Temporal Weighting (ATW), Depthwise Separable Convolutions (DSC), 데이터 증강 기법으로 구성되어 있으며, 이러한 구성 요소들은 노이즈와 가림에 대한 강인성, 모델의 정확성 및 효율성을 높이는 데 기여합니다.

- **Technical Details**: RE-TCN의 ATW는 행동 시퀀스 내에서 가장 중요한 프레임에 초점을 맞추도록 설계되었으며, 각 프레임에 동적으로 중요도를 할당합니다. DSC는 깊이와 점을 나누어 합성곱을 수행하여 입력된 스켈레톤 데이터를 처리하는 데 필요한 매개변수와 연산의 수를 대폭 줄입니다. 데이터 증강 기법은 실제 환경에서의 모델의 강인성을 향상시키기 위해 설계되었습니다.

- **Performance Highlights**: RE-TCN은 NTU RGB+D 60, Northwestern-UCLA, SHREC'17, DHG-14/28의 네 가지 벤치마크 데이터셋을 통해 기존 모델들보다 높은 정확도와 노이즈 및 가림에 대한 강인성을 보여주었습니다. 이 모델은 AAL 환경에서의 실시간 행동 인식 성능을 크게 개선하며, 컴퓨팅 효율성에서도 우수한 성과를 나타냅니다.



### RECALL-MM: A Multimodal Dataset of Consumer Product Recalls for Risk Analysis using Computational Methods and Large Language Models (https://arxiv.org/abs/2503.23213)
- **What's New**: 이 연구에서는 미국 소비자 제품 안전 위원회(CPSC)의 리콜 데이터베이스를 기반으로 multimodal dataset인 RECALL-MM을 개발했습니다. 이 데이터셋은 과거 정보에 기반한 데이터 주도적 위험 평가를 지원하며, 생성적 방법(generative methods)을 통해 확장됩니다. 데이터셋의 패턴은 개선된 안전 조치를 통해 큰 영향을 미칠 수 있는 특정 영역을 강조합니다.

- **Technical Details**: 연구에서는 2000년부터 2024년까지의 6,874개의 리콜 데이터를 수집하였고, 이를 대형 언어 모델(LLM)로 확장하여 위험 평가를 위한 새로운 분류 및 시각적 설명을 추가했습니다. 데이터셋의 각 항목은 위험 분류, 제품 카테고리, 치료 유형 등의 주요 리콜 속성을 포함하고 있으며, GPT-4o를 활용해 내용을 구조화했습니다. 연구는 또한 LLM을 이용해 제품 이미지만으로도 잠재적인 위험을 예측하는 방법론을 소개합니다.

- **Performance Highlights**: 사례 연구를 통해 리콜 데이터의 유틸리티를 증명하고 제품 위험을 식별하는 데 어떻게 기여하는지를 보여줍니다. 첫 두 가지 사례 연구는 설계자들이 리콜된 제품 간의 패턴을 시각화하고 새로운 제품 아이디어를 전반적인 리콜 환경 속에 위치시킬 수 있음을 보여줍니다. 마지막 사례 연구에서는 LLM을 활용하여 제품 이미지에 기반한 위험 예측의 강점과 한계를 강조하며, 설계 과정 전반에 걸친 위험 인식의 중요성을 부각시킵니다.



### Enhancing Knowledge Graph Completion with Entity Neighborhood and Relation Contex (https://arxiv.org/abs/2503.23205)
- **What's New**: 이 논문에서는 지식 그래프 완성(KGC) 문제를 해결하기 위해 KGC-ERC라는 새로운 프레임워크를 제안합니다. KGC-ERC는 생성적 언어 모델에 엔티티 이웃(entity neighborhood)과 관계(context of relation) 정보를 통합하여 예측 성능을 향상시킵니다. 또한, 입력 토큰 제한 속에서 중요한 컨텍스트를 효율적으로 선택하기 위한 샘플링 전략을 소개하여 전체 환경 정보를 최적화합니다.

- **Technical Details**: KGC-ERC는 KGT5-context를 기반으로 하여 쿼리에 관계 컨텍스트를 추가하고 이를 통해 입력 시퀀스를 풍부하게 합니다. 이 프레임워크의 핵심 요소인 선택자(Selector) 모듈은 지식 그래프에서 유의미한 엔티티 및 관계 컨텍스트를 샘플링하여 필터링합니다. 이 과정에서 엔티티 이웃 샘플링과 관계 컨텍스트 샘플링을 사용하는데, 이를 통해 최대한 의미 있는 정보를 선택하여 생성적 언어 모델에 전달합니다.

- **Performance Highlights**: Wikidata5M, Wiki27K, FB15K-237-N 데이터셋에 대한 실험 결과, KGC-ERC는 기존의 최신 기법들에 비해 예측 성능과 확장성 모두에서 우수한 결과를 보여주었습니다. 이 모델은 MRR과 Hits@k와 같은 평가 지표에서 뛰어난 성과를 달성하였으며, 복잡한 관계를 효과적으로 처리할 수 있는 능력을 입증했습니다.



### The Challenge of Achieving Attributability in Multilingual Table-to-Text Generation with Question-Answer Blueprints (https://arxiv.org/abs/2503.23204)
- **What's New**: 이 논문에서는 저자들이 낮은 자원의 언어인 아프리카 언어로 구성된 TaTA 데이터셋에 대한 멀티링구얼(멀티언어) Table-to-Text 생성 작업에서 Question-Answer(QA) 청사진을 사용하여 결과의 신뢰성을 증대시키는 방법을 탐구하고 있습니다. 또한 이 작업은 첫 번째로 QA 청사진을 적용하여 신뢰성을 개선하기 위한 새로운 방법을 제안합니다. 저자들은 영어 예시에 대해서는 QA 청사진이 결과의 신뢰성을 높이는 데 효과적이라는 것을 발견했으나, 멀티링구얼 환경에서는 효과가 떨어진다고 보고했습니다.

- **Technical Details**: 저자들은 Seq2Seq 모델을 활용하여 입력된 표의 정보를 기반으로 유창하고 정확한 설명을 생성하는 작업을 수행했습니다. 이 논문에서는 QA 청사진이 포함된 TaTA 데이터셋에서 만큼 Seq2Seq 언어 모델을 미세조정(finetuning)했습니다. 그러나 멀티링구얼 환경에서는 영어에서 타겟 언어로 QA 청사진을 번역하는 과정에서 발생하는 부정확성으로 인해 제약이 있다고 합니다.

- **Performance Highlights**: QA 청사진을 사용한 결과, 모델이 영어 데이터에 대해 학습 및 평가되었을 때는 결과의 신뢰성이 향상된 것으로 나타났습니다. 그러나 멀티링구얼 환경에서는 기대하는 성과를 내지 못하였으며, 이는 영어에서 다른 언어로 번역되는 과정에서 발생하는 오류와 모델이 생성한 청사진을 제대로 활용하지 못하는 문제 때문이라고 분석되었습니다. 이 논문은 전반적인 성능 평가에 대한 깊이 있는 분석을 제공하여 향후 연구에 기초 자료를 제공하고 있습니다.



### Incorporating GNSS Information with LIDAR-Inertial Odometry for Accurate Land-Vehicle Localization (https://arxiv.org/abs/2503.23199)
- **What's New**: 이 논문은 고속에서의 로컬라이제이션(가용 위치 인식) 문제를 해결하기 위해 새로운 LIDAR 기반 로컬라이제이션 프레임워크를 제안합니다. 이 시스템은 다양한 센서의 정보를 포함한 3D 포인트 클라우드 맵을 이용하여 높은 정확도의 로컬라이제이션을 제공합니다. 또한, 오프라인 포인트 클라우드 맵을 활용하여 로컬라이제이션의 견고성을 향상시키고 수렴 속도를 높이는 혁신적인 등록 방법인 Dynamic-ICP를 제안합니다.

- **Technical Details**: 로컬라이제이션 프레임워크는 GNSS, IMU, LIDAR 기반의 오도메트리(odometry)를 통합하여 구현됩니다. 이 시스템은 사전 오프라인 3D 포인트 클라우드 맵과의 통합을 통해 온라인 피드백을 받아 최대한의 정확도를 유지합니다. Dynamic-ICP 알고리즘은 LIDAR 오도메트리에서 재로컬라이제이션 문제를 해결하고 로컬라이제이션 정확도를 향상시키기 위해 고안되었습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 다양한 데이터 세트의 맵에서 다른 로컬라이제이션 알고리즘보다 더 높은 정확도와 견고성을 보여주었습니다. 또한, GNSS와 IMU를 결합하여 전 세계 위치 인식의 정확성을 향상시키고, 느린 등록 알고리즘의 작동을 보완함으로써 전체 시스템의 안정성을 높였습니다. 복잡한 도시 환경에서도 안정적인 포즈 최적화 추정을 수행할 수 있는 것으로 나타났습니다.



### Large Language Models are Unreliable for Cyber Threat Intelligenc (https://arxiv.org/abs/2503.23175)
- **What's New**: 최근 연구들에서는 Large Language Models (LLMs)가 사이버 보안 분야에서 데이터 홍수를 제어하는 데 효과적으로 사용될 수 있다고 주장하고 있습니다. 이러한 가능성을 확인하기 위해 본 논문에서는 CTI(Cyber Threat Intelligence) 업무에 대한 평가 방법론을 제시하고, LLM의 일관성과 신뢰도를 정량화할 수 있는 방법을 소개합니다. 실험 결과, LLM들이 실제 보고서에서 충분한 성능을 보이지 않으며 일관성이 부족하고 과신하게 되는 경향이 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 350개의 위협 정보 보고서를 기반으로 LLM을 평가했습니다. 모델의 효과성, 일관성 및 신뢰도 보정을 검사하기 위해 새로운 평가 파이프라인을 설계하고 배포했습니다. 연구 결과, few-shot learning 및 fine-tuning이 CTI에 대한 성능을 개선하는 데 중요한 영향을 미치지 않았으며, LLM이 생성한 정보의 일관성도 의문을 제기합니다.

- **Performance Highlights**: LLMs의 출력은 반복적인 질문에서도 일관성이 부족하고, 같은 CTI 보고서에서 서로 다른 결과를 생성하는 경향이 있습니다. 이는 패치 관리와 같은 CTI에서 심각한 위험 요소로 작용할 수 있습니다. 연구 결과에 따르면, LLM의 신뢰도 보정이 낮아 정보 추출 및 생성에서 신뢰성을 보장할 수 없는 것으로 나타났습니다.



### Reasoning-SQL: Reinforcement Learning with SQL Tailored Partial Rewards for Reasoning-Enhanced Text-to-SQL (https://arxiv.org/abs/2503.23157)
- **What's New**: 이번 연구는 Text-to-SQL 작업을 위해 설계된 새로운 부분 보상 집합을 제안합니다. 이는 기존의 수동으로 구성된 경로에 의존하는 방법의 한계를 극복하고, 더 나은 추론 능력과 일반화 능력을 갖춘 모델 개발을 돕습니다. 연구팀은 Reward-driven self-exploration을 활용한 Reasoning-SQL 프레임워크를 구축하여, LLM의 내부 추론 과정을 자동으로 최적화합니다.

- **Technical Details**: 새로운 보상 집합은 Schema Linking, AI 피드백, N-gram 유사성 및 구문 체크를 포함하여 강화 학습(Reinforcement Learning, RL)에서 흔히 발생하는 보상 희소성 문제를 해결하도록 설계되었습니다. Group Relative Policy Optimization (GRPO)을 활용하여 이들 보상 신호를 효과적으로 통합하고, 다양한 후보 쿼리를 생성하여 최종 실행 정확성을 최적화합니다. 방식으로는 다수의 모델 크기를 실험하여 RL 전용 학습이 감독 세부 조정(Supervised Fine-Tuning, SFT)보다 더 높은 정확도를 달성하는 모습을 보여줍니다.

- **Performance Highlights**: 제안된 방법을 사용한 RL 훈련 모델이 BIRD 벤치마크에서 기존의 큰 상용 모델들보다 4% 및 3% 더 높은 성능을 제공합니다. 특히, 14B 파라미터를 가진 RL 훈련 모델은 보다 낮은 비용으로 더 나은 성과를 보여주어, 기존의 Text-to-SQL 시스템과 비교할 때 상대적으로 93% 더 저렴한 비용으로 72.78%의 실행 정확도를 기록했습니다. 이로 인해 기존 SFT 방식보다도 더 경쟁력 있는 성능을 발휘할 수 있음을 입증했습니다.



### Conversational Agents for Older Adults' Health: A Systematic Literature Review (https://arxiv.org/abs/2503.23153)
Comments:
          31 pages, 4 figures

- **What's New**: 이번 연구는 노인 건강을 위한 대화형 에이전트(Conversational Agents, CAs)에 대해 체계적인 리뷰를 제공하며, 72개의 논문을 분석하여 노인들이 경험하는 CAs의 특성과 기대를 조명합니다. 기존 연구들은 CAs의 역할이 챗봇과 음성 비서로 변화하고 있으며, 노인들이 CAs에 대한 낮은 수용성을 보이는 이유를 다양한 각도에서 살펴보았습니다. 특히, 노인들은 개인화, 자연어 대화의 필요성 및 시스템에 대한 통제력을 중시하며, 이러한 요구가 CAs 설계에 중요한 고려 사항임을 강조합니다.

- **Technical Details**: 이 연구는 노인을 위한 다양한 CAs의 특성과 사용 경험을 분석하며, 에이전트가 코치, 동반자, 건강 보조자 등 여러 역할을 수행할 수 있음을 보여줍니다. 연구는 인간-컴퓨터 상호작용(Human-Computer Interaction, HCI)의 관점에서 접근하여, CAs가 사용되는 다양한 헬스케어 시나리오와 기술의 진화에 따른 돌봄의 방향성을 탐구합니다. 노인들은 시스템 사용의 어려움, 개인 정보 보호 우려 등 여러 도전과제에 직면하고 있으며, 이러한 문제를 해결하기 위한 연구 방향도 제시합니다.

- **Performance Highlights**: 노인들은 CAs의 여러 기능과 자연어 대화를 기대하고 있지만, 낮은 수용성 및 다양한 우려로 인해 실제 사용이 제한되고 있습니다. 사용자들은 CAs가 여러 기능을 지원하고 개인화된 도움을 제공하기를 바라며, 이를 통한 건강 증진이 필요하다는 점을 강조했습니다. 이 연구는 노인을 위한 CAs 설계의 도전과제로 나이의 다양성, 독립성의 필요성, CAs의 효과 등을 논의하며 HCI 분야에 중요한 기여를 하고 있습니다.



### Agent-Based Modeling and Deep Neural Networks for Establishing Digital Twins of Secure Facilities under Sensing Restrictions (https://arxiv.org/abs/2503.23147)
Comments:
          This paper has been already published in the 2024 Interservice/Industry Training, Simulation, and Education Conference (I/ITSEC'24): this https URL The authors have obtained permission from I/ITSEC'24 organizers to release this paper on arXiv. Appropriate licensing is also applied

- **What's New**: 이 연구는 보안 핵시설에서 인간의 패턴 오브 라이프(POL)를 모니터링하기 위해 메타폴(MetaPOL)이라는 디지털 트윈 시스템을 도입하여, 비상사태 시나리오에 대한 응답과 정상 운영 간의 NPC(Non-Playable Character) 움직임의 차이를 분석했습니다. 실시간 시뮬레이션이 불가능한 환경에서 비정상적 상황을 예측하는 새로운 접근 방식을 제시합니다. 또한, 에이전트 기반 모델(ABM)을 기반으로 한 합성 움직임 궤적의 생성이 큰 역할을 했습니다.

- **Technical Details**: 해당 연구에서는 시설 직원의 POL에 대한 일화적 데이터를 활용하여 에이전트 기반 모델을 사용해 합성 움직임 궤적을 생성했습니다. 이러한 합성 궤적은 다음 위치와 머무는 시간 예측을 위한 딥 뉴럴 네트워크 서그레이트를 훈련시키는 데 사용되었습니다. 저자들은 다층 퍼셉트론(Multi-Layer Perceptron)과 혼합 밀도 네트워크(Mixture Density Network)를 이용해 합성 궤적을 예측하는 데 성공했습니다.

- **Performance Highlights**: 딥 뉴럴 네트워크에 의해 주도된 VR 환경 내 NPC의 움직임은 정상 운영 시나리오와 비상 응답 처리를 시뮬레이션할 때의 움직임과 유의미한 차이를 보였습니다. 연구 결과, 이러한 예측 시스템이 고Security 핵 시설에서의 안전성을 확보하는 데 중요한 역할을 할 수 있음을 입증하였습니다. 연구의 전반적인 결과는 보안 환경에서 시뮬레이션의 정확성을 크게 향상시킬 수 있는 잠재력을 보여줍니다.



### CodeARC: Benchmarking Reasoning Capabilities of LLM Agents for Inductive Program Synthesis (https://arxiv.org/abs/2503.23145)
- **What's New**: 새로운 평가 프레임워크인 CodeARC(코드 추상화 및 추론 챌린지)가 도입되었습니다. 이는 에이전트가 숨겨진 목표 함수를 질의하고 반복적으로 솔루션을 조정할 수 있는 상호작용 설정을 제공합니다. 기존의 정적 예제에 의존했던 평가 방식의 한계를 보완하며, 실제 상황을 반영할 수 있도록 설계되었습니다.

- **Technical Details**: CodeARC는 LLM 기반의 에이전트가 초기 입력-출력 예제 집합을 사용하여 작업을 시작하고, 새로운 입력으로 목표 함수를 질의하며, 미분 테스트 오라클을 통해 검증하고 디버깅할 수 있는 구조를 가집니다. 이 과정에서는 에이전트가 스스로 입력을 생성하고 피드백에 따라 솔루션을 수정해야 합니다.

- **Performance Highlights**: CodeARC를 사용한 실험 결과, 총 18개 모델 중 OpenAI의 o3-mini가 52.7%의 성공률로 가장 뛰어난 성과를 거두었습니다. 또한, LLaMA-3.1-8B-Instruct의 세부 조정(Fine-tuning)을 통해 최대 31%의 상대적 성능 향상을 달성했습니다.



### CrossMuSim: A Cross-Modal Framework for Music Similarity Retrieval with LLM-Powered Text Description Sourcing and Mining (https://arxiv.org/abs/2503.23128)
Comments:
          Accepted by ICME2025

- **What's New**: 이번 논문은 음악 유사성 검색(music similarity retrieval)을 위한 새로운 교차 모드 대조 학습 프레임워크(CrossModal contrastive learning framework)를 제안합니다. 이 프레임워크는 텍스트 기술의 개방적 특성을 활용하여 음악 유사성을 모델링하며, 전통적인 단일 모드(unimodal) 접근 방식의 한계를 극복합니다. 또한, LLM(대형 언어 모델)을 기반으로 하는 데이터 수집 방법을 도입하여 고품질 텍스트-음악 쌍 데이터의 부족 문제를 해결하고 있습니다.

- **Technical Details**: 제안된 시스템은 텍스트 인코더(text encoder)와 오디오 인코더(audio encoder)로 구성되어 있으며, 출력값을 동일한 차원으로 정렬하는 프로젝션 레이어(projection layer)를 포함합니다. 텍스트 인코더는 문장 임베딩을 생성하기 위해 최적화된 문장 변환기(sentence transformer) 모델을 사용하고, 오디오 인코더는 음악 자동 태깅(music auto-tagging) 작업을 위해 사전 학습된 소형 MTT(Music Tagging Transformer) 모델을 채택합니다. 정보 손실 최소화를 통한 통합 잠재 공간(unified latent space) 정렬 기법도 사용합니다.

- **Performance Highlights**: 이 프레임워크의 효과는 객관적 평가(objective metrics), 주관적 평가(subjective evaluations), 그리고 Huawei Music 스트리밍 플랫폼에서의 실험적 A/B 테스트를 통해 입증되었습니다. 실험 결과, 제안된 방법은 기존 벤치마크에 비해 현저한 성능 개선을 달성하였으며, 다양한 음악 추천 시스템에 적용 가능한 가능성을 보여줍니다.



### Evaluating Compositional Scene Understanding in Multimodal Generative Models (https://arxiv.org/abs/2503.23125)
- **What's New**: 이 연구는 최신 텍스트-이미지 모델인 DALL-E 3와 멀티모달 비전-언어 모델들이 복합적인 시각 장면을 이해하고 생성하는 능력을 평가합니다. 이전 세대의 모델들에 비해 현 모델들이 관계(relational) 작업을 더 잘 수행하는 경향이 있음을 보여주지만, 인간 참가자들의 성과에는 미치지 못하는 결과도 관찰되었습니다.

- **Technical Details**: 연구에서는 첫 번째 섹션에서 DALL-E 3의 공간 관계를 기반으로 한 이미지 생성 능력을 평가하고, 두 번째 섹션에서는 GPT-4와 같은 멀티모달 비전-언어 모델들의 관계 패턴 추론 능력을 분석합니다. 특히 복합적인 관계를 포함하는 다양한 프롬프트로 이 모델들을 테스트하여 그 성능을 확인했습니다.

- **Performance Highlights**: DALL-E 3는 이전 모델에 비해 관계 프롬프트에서 개선된 성능을 나타냈지만, 비일상적인 시나리오나 복잡한 관계가 포함된 프롬프트에 대해서는 성능이 하락했습니다. evaluated 모델들은 많은 객체들 (>5개)과 관련된 문제에서 인간 참가자들과 비교해 성과가 낮았으며, 이러한 결과는 현재 모델들이 시각 장면의 구성적 이해(compositional understanding)에서 더 많은 발전이 필요함을 시사합니다.



### How to safely discard features based on aggregate SHAP values (https://arxiv.org/abs/2503.23111)
- **What's New**: 이번 연구는 SHAP (SHapley Additive exPlanations) 값을 활용한 글로벌 기능 중요도 평가의 한계를 다룹니다. 기존 방법에서는 작은 SHAP 값이 해당 기능이 함수에 영향을 미치지 않는다고 판단했지만, 연구 결과 그 판단은 잘못될 수 있음을 발견했습니다. 우리는 SHAP 값을 데이터 지원 밖에서 평가할 때 발생하는 문제를 강조합니다.

- **Technical Details**: 연구에서는 기능 i에 대한 SHAP 값이 0이라고 하더라도, 함수가 Feature i에 의존하는 경우가 있음을 명확하게 보여줍니다. 이에 대한 해결책으로, 우리는 SHAP 값을 기초 분포의 주변 분포 곱으로 확장된 지원에서 집계하는 방법을 제안합니다. 이러한 수정으로 인해 작은 집계 SHAP 값이 해당 기능을 안전하게 제거할 수 있음을 증명합니다.

- **Performance Highlights**: 또한, KernelSHAP에 대한 결과도 확장하여, 확장된 분포에서 계산될 때 작은 집계 값이 기능 제거를 정당화함을 보였습니다. 이러한 결과는 KernelSHAP이 진정한 SHAP 값을 정확하게 근사하는지 여부와 관계없이 성립합니다. 우리의 연구는 SHAP 및 KernelSHAP 알고리즘에 대한 이론적 및 실용적 함의를 가지고 있습니다.



### Fast Training of Recurrent Neural Networks with Stationary State Feedbacks (https://arxiv.org/abs/2503.23104)
Comments:
          18 pages (including additional contents), 3 figures, 5 tables, code available at this https URL

- **What's New**: 이번 연구에서는 Recurrent Neural Networks (RNNs)의 효율성을 높이는 새로운 방법을 제안합니다. 기존의 Backpropagation Through Time (BPTT) 알고리즘 대신에 고정된 gradient feedback 메커니즘을 도입하여 정확한 gradient 전달의 효율적인 근사를 제공합니다. 이를 통해 훈련 오버헤드를 크게 줄이면서도 장기 의존성(capacity to capture long-term dependencies)을 유지할 수 있습니다.

- **Technical Details**: 본 논문에서는 상태-공간 모델(State-Space Model, SSM) 원리를 활용하여 구조화된 피드백 행렬을 정의함으로써 미래 시간 단계로부터 직접적으로 gradient를 전달합니다. 이 방식은 gradient backpropagation의 재귀적 계산을 우회하여 훈련 시간을 절약하는 동시에 RNN의 메모리 능력을 보존입니다. 제안된 알고리즘은 BPTT 대신 시간 역으로 SSM을 추론하여 장거리 의존성을 처리하는 능력을 활용합니다.

- **Performance Highlights**: 실험 결과, 언어 모델링 벤치마크에서 제안된 방법이 경쟁력 있는 perplexity 점수를 보여주었으며, 훈련 비용이 현저히 감소했습니다. 이 결과들은 SSM과 같은 피드백 방법을 설계함으로써 RNN의 효율적 장점을 활용할 수 있음을 시사합니다. 이러한 성과는 RNN이 실용적인 응용 분야에서 큰 잠재력을 지니고 있음을 나타냅니다.



### RL2Grid: Benchmarking Reinforcement Learning in Power Grid Operations (https://arxiv.org/abs/2503.23101)
- **What's New**: 이 논문에서는 전력망 운영의 혁신을 목표로 하는 RL2Grid라는 새로운 벤치마크를 소개합니다. RL2Grid는 전력 시스템 운영자들과 협력하여 개발되어, 전력망 제어의 발전을 가속화하고 RL(강화학습) 방법의 성숙도를 높이는 데 도움을 줍니다. 이 벤치마크는 전력망 작업에서의 임무, 상태 및 행동 공간, 보상 구조를 통합된 인터페이스로 표준화하여 RL 접근법의 평가 및 비교를 용이하게 합니다.

- **Technical Details**: RL2Grid는 전력망의 복잡한 동역학을 다루기 위해 RTE 프랑스에서 개발한 전력 시뮬레이션 프레임워크를 바탕으로 구축되었습니다. 이 벤치마크는 다양한 전력망 업무를 포함하고 있으며, 각 업무는 안전 제약 조건을 통합한 제약된 태스크 포멀라이제이션을 제공합니다(예: 전력 흐름, 발전기 제한 및 라인 제한). 또한, 에이전트가 그리드를 효과적으로 운영하도록 학습하는 과정을 분석하기 위해 RL2Grid의 설계 선택에 대한 포괄적인 분석이 수행됩니다.

- **Performance Highlights**: 우리는 RL2Grid 내의 전력망 제어 작업에 대해 여러 인기 있는 RL 알고리즘의 성능을 benchmark 하였습니다. 특히, 기본적인 그리드 작업(예: 선 재연결 및 대기 동작)을 기존 알고리즘의 교육 루프에 통합할 수 있는 휴리스틱 모듈이 구현되어, 모든 RL 알고리즘의 성능과 샘플 효율성이 크게 향상되었습니다. RL2Grid는 RL 방법을 실제 전력망 환경에서 성숙시키는 출발점으로 기능하며, 개방된 문제들과 연결된 여러 방향을 논의합니다.



### UNITYAI-GUARD: Pioneering Toxicity Detection Across Low-Resource Indian Languages (https://arxiv.org/abs/2503.23088)
- **What's New**: UnityAI-Guard는 저자원이 운영되는 인도 언어를 대상으로 한 이진 독성 분류 프레임워크로, 독성이 있는 콘텐츠를 식별하는 혁신적인 모델을 제공합니다. 기존의 시스템은 주로 자원이 풍부한 언어에 맞춰져 있었으나, UnityAI-Guard는 이 중요한 격차를 해소하기 위해 개발되었습니다. 이 프레임워크는 다양한 Brahmic/Indic 스크립트를 지원하며 연평균 F1 점수 84.23%를 달성했습니다.

- **Technical Details**: UnityAI-Guard는 888,000개의 훈련 인스턴스와 35,000개의 수동 검증 테스트 인스턴스를 활용하여 불균형 문제를 해결하고, 여러 언어에 대한 독성 분류 모델을 개발하였습니다. 자동 자막 및 API를 통해 다양한 기능을 제공하며, 전반적인 아키텍처는 효율성과 사용자 친화성을 고려하여 설계되었습니다. 모집단의 다양성과 정합성을 고려하여 두 명의 원어민이 샘플을 검토하여 데이터의 신뢰성을 높였습니다.

- **Performance Highlights**: UnityAI-Guard는 세 가지 크기의 모델을 사용하여 실험을 수행하였으며, 가장 큰 모델인 aya-expanse-8B가 87.21%의 정확도를 기록함으로써 독성 감지에서 더 나은 성능을 보여주었습니다. 실험 결과, 모델 크기가 커짐에 따라 F1 점수가 일관되게 향상되었으며, 특히 다양한 인도 스크립트를 처리하는 데 있어 큰 모델들이 뛰어난 능력을 보였습니다.



### The Reasoning-Memorization Interplay in Language Models Is Mediated by a Single Direction (https://arxiv.org/abs/2503.23084)
- **What's New**: 이 연구에서는 대형 언어 모델(LLMs)의 추론 및 기억 동학을 이해하기 위해 기계적 접근 방식을 제안합니다. 연구팀은 모델의 잔여 흐름(residual stream)에서 특정 선형 특성(Linear Features)을 찾아 이를 통해 모델의 추론과 기억 간의 균형을 조절할 수 있음을 보여주었습니다. 이러한 특성을 통해 LLMs의 추론 능력을 보다 효과적으로 활성화할 수 있는 방법이 제시되어, 더 견고하고 해석 가능한 생성 AI 시스템 개발의 기초가 될 것입니다.

- **Technical Details**: 연구자들은 LLM의 메모리 사용을 일반화능력(generalizability) 부족으로 정의하며, 이를 평가하기 위해 합성 추론 벤치마크를 설계했습니다. 조사 결과, 특정 선형 추론 특성(Linear Reasoning Features, LiReFs)이 모델의 활성화 공간 내에서 모델의 일반화 능력을 지배하며, 이를 통해 문제 해결을 위한 추론 능력을 향상시킬 수 있는 잠재력이 있음을 입증했습니다. 이 특성을 조작함으로써 모델이 강력한 일반화 기능을 발휘하도록 유도할 수 있습니다.

- **Performance Highlights**: 연구의 주요 결과는 네 가지 서로 다른 LLMs와 여섯 개 데이터셋에서 실험을 통해 확인되었습니다. LiReFs를 활용하면 모델의 추론 오류를 줄이고, 더 적절한 문제 해결 기능을 활성화하여 성능을 향상시킬 수 있습니다. 이 연구 결과는 LLMs의 추론 능력이 선형 특성에 의해 매개됨을 입증하며, 다양한 지식 영역과 언어에서의 모델 성능을 효과적으로 조절하는 새로운 기제를 제시합니다.



### Efficient Adaptation For Remote Sensing Visual Grounding (https://arxiv.org/abs/2503.23083)
- **What's New**: 이번 연구에서는 Parameter Efficient Fine Tuning (PEFT) 기법을 적용하여 원격 탐사(remote sensing, RS) 작업에 적합하도록 Grounding DINO와 OFA 모델을 조정했습니다. 연구 결과, LoRA 기법을 통해 DIOR-RSVG 데이터세트에서 최상위 성능을 달성했으며, Adapter와 BitFit 기술을 비교한 결과 Adapter가 고성능을 보였습니다. 이 연구는 PEFT 기법의 가능성을 강조하며, 전체 모델 학습의 비용 대비 효율적인 대안을 제시합니다.

- **Technical Details**: Parameter Efficient Fine Tuning (PEFT)은 모델의 최소한의 파라미터 집합만을 조정하여 계산 효율성을 제공합니다. Adapters는 경량 모듈로, 사전 훈련된 모델의 레이어 사이에 삽입되어 특정 작업 학습을 위한 추가 파라미터를 도입합니다. LoRA는 모델의 가중치 행렬에 저랭크 업데이트를 적용하여 학습 가능한 파라미터 수를 줄이고, BitFit은 모델 레이어의 편향 항만을 미세 조정하는 방법입니다.

- **Performance Highlights**: 실험 결과, Grounding DINO는 SOTA VG 모델로서, 텍스트 프롬프트와 이미지 내 특정 영역을 연결하는데 있어 뛰어난 성능을 발휘했습니다. Multi-scale deformable attention이 다양한 공간 해상도와 이미지 특징의 통합을 용이하게 하여 정확한 객체 탐지와 구문 로컬라이제이션을 가능하게 했습니다. 또한 모델의 계산 효율을 높이기 위해 고정 파라미터 비율을 측정하여 PEFT 기술을 통한 비용 절감 효과를 평가했습니다.



### InkFM: A Foundational Model for Full-Page Online Handwritten Note Understanding (https://arxiv.org/abs/2503.23081)
- **What's New**: 이 논문에서는 손으로 쓴 디지털 노트의 내용을 정확하게 해석하고 이해하는 방법을 개발하기 위한 새로운 모델인 InkFM을 소개합니다. InkFM은 28종의 스크립트에서 텍스트를 인식하고, 수학적 표현을 인식하며, 페이지를 텍스트와 그림 같은 개별 요소로 구분할 수 있는 고유한 기능을 제공합니다. 특정 데이터셋에 대해 소스 모델을 미세 조정(fine-tuning)하여 페이지 세분화와 텍스트 인식의 품질을 더욱 향상시킬 수 있음을 입증했습니다.

- **Technical Details**: InkFM은 다채로운 혼합 작업에 대해 훈련되어 있으며, 세 가지 핵심 작업인 세분화(segmentation), 분류(classification), 인식(recognition)을 통합하여 하나의 강력한 모델로 발전시킵니다. 세분화 작업에서는 단어와 그림을 어떤 객체에 할당하고 각 객체를 분류하는 기술이 포함됩니다. 또한, 이는 손으로 쓴 텍스트를 문자 시퀀스로 변환하는 인식 작업을 통해, 다양한 글쓰기 스타일에 적응할 수 있는 능력을 보여줍니다.

- **Performance Highlights**: InkFM은 다양한 스크립트에서의 텍스트 인식 및 스케치 분류에서 뛰어난 성능을 보여주며, 특히 잉크 기반 스타일 작업에서 SoTA(state-of-the-art) 품질을 달성했습니다. 예를 들어, IAM 데이터셋에서 영어 손글씨 텍스트 라인 세분화에서 경쟁력 있는 결과를 보였고, QuickDraw 데이터셋에서는 최신 정확도를 기록했습니다. 이 모델은 고유한 성능으로 시각적 요소를 탐지하고, 노트의 개별 요소를 효과적으로 구분할 수 있습니다.



### STSA: Spatial-Temporal Semantic Alignment for Visual Dubbing (https://arxiv.org/abs/2503.23039)
Comments:
          Accepted by ICME 2025

- **What's New**: 이 논문에서는 Spatial-Temporal Semantic Alignment (STSA)라는 새로운 방법을 제안합니다. 이 방법은 공간(domain)과 시간(domain)에서의 의미적 특징을 정렬하여 동적 얼굴의 합성 안정성을 향상시키는 데 초점을 맞추고 있습니다. 이 논문은 소리 기반의 시각적 더빙이 현재의 기술로서의 한계를 극복하도록 도와줄 수 있는 가능성을 보여주고 있습니다.

- **Technical Details**: STSA는 이중 경로 정렬 메커니즘을 도입하여 공간 및 시간 도메인에서 여러 크기의 특징을 정렬합니다. Consistent Information Learning (CIL) 모듈을 통해 다중 스케일에서의 상호 정보를 최대화하여, 잘못 정렬된 정보를 교정할 수 있습니다. 또한, 확률적 히트맵을 사용하여 의미적 불확실성을 허용하는 방향성을 제공하여, 합성된 얼굴의 움직임이 부드럽게 유지되도록 합니다.

- **Performance Highlights**: 실험 결과, STSA 방법은 이미지 품질과 합성 안정성 면에서 우수성을 입증하였습니다. 전처리된 가중치와 추론 코드는 제공되며, 이로 인해 연구자들이 쉽게 접근하고 사용할 수 있을 것입니다. STSA의 도입을 통해 동작이 자연스럽고 현실적인 시각적 더빙이 이루어질 수 있음을 보여주고 있습니다.



### Reproducibility Companion Paper: Making Users Indistinguishable: Attribute-wise Unlearning in Recommender Systems (https://arxiv.org/abs/2503.23032)
- **What's New**: 이번 논문에서는 이전 연구 "Making Users Indistinguishable: Attribute-wise Unlearning in Recommender Systems"에서 제시한 실험 결과를 reproducible하게 재현하는 방법을 설명합니다. 본 연구의 목표는 기존 방법의 유효성을 검증하고 다른 연구자들이 결과를 재현할 수 있도록 돕는 것입니다. 데이터셋, 소스코드 구조, 설정 파일, 실험 환경 및 재현된 실험 결과를 상세히 설명합니다.

- **Technical Details**: 본 연구는 추천 시스템에서 사용자의 민감한 속성을 보호하기 위해 Attribute Unlearning (AU) 기술을 적용합니다. 특히 Post-Training Attribute Unlearning (PoT-AU) 설정을 엄격하게 검토하며, 이를 위해 추천 성능과 unlearning 성능의 균형을 위해 두 가지 구성 요소로 이루어진 손실 함수 (loss function)를 설계합니다. U2U(User-to-User)와 D2D(Distribution-to-Distribution) 측정을 통해 실험에서 제안된 방법의 효과를 검증합니다.

- **Performance Highlights**: 세 가지 실제 데이터셋을 사용하여 실험을 수행하였으며, 추천 성능과 unlearning 성능, 효율성을 평가합니다. 각 추천 모델의 효과성을 NMF와 LightGCN을 통해 분석하며, 다양한 평가 지표(Accuracy, Precision, Recall, AUC)를 사용하여 제안된 방법의 성능을 기존 방법들과 비교합니다. 사용자 임베딩 분포의 변화를 분석하여 제안된 방법의 메커니즘을 이해하도록 돕습니다.



### Towards Understanding the Optimization Mechanisms in Deep Learning (https://arxiv.org/abs/2503.23016)
- **What's New**: 이 논문에서는 깊은 신경망(deep neural networks)의 감독 분류(supervised classification) 최적화 메커니즘을 확률 분포 추정(probability distribution estimation) 관점에서 탐구합니다. Fenchel-Young 손실(Fenchel-Young loss)을 사용하여 비볼록적(non-convex)인 적합 오류(fitting error)에도 불구하고 전역 최적(global optimal) 솔루션을 근사할 수 있음을 보여줍니다. 이 방법은 기울기 정규화(gradient norm)와 구조적 오류(structural error)를 동시에 최소화하는 것을 통해 가능해집니다.

- **Technical Details**: 논문은 모델의 매개변수(parameter)에 대한 기울기 독립성(gradient independence) 가정 하에, 구조적 오류가 모델 매개변수 수에 의해 제어된다는 것을 증명합니다. 즉, 매개변수 수가 많을수록 구조적 오류는 작아집니다. 이러한 결론은 과적합(over-parameterization) 및 무작위 초기화(random initialization)와 같은 기술에 대한 이론적 통찰(theoretical insights)을 제공합니다. Fenchel-Young 손실은 많은 손실 함수가 이 형태로 표현될 수 있다는 점에서 깊은 학습 분석의 복잡성을 줄이는 데 유용한 접근 방식으로 제시됩니다.

- **Performance Highlights**: 제안된 방법의 핵심 결론은 실험적 결과를 통해 검증되었습니다. 네트워크의 매개변수 수를 늘리고 매개변수 간의 독립성을 보장함으로써, 분류 문제는 특정 입출력(feature와 label) 조건부 확률 분포 추정과 동등하다는 것을 보여줍니다. 이러한 기법들을 통해 DNN의 훈련 메커니즘 및 비볼록 최적화(non-convex optimization)에서의 동작을 보다 잘 이해할 수 있게 됩니다.



### MSNGO: multi-species protein function annotation based on 3D protein structure and network propagation (https://arxiv.org/abs/2503.23014)
Comments:
          8 pages, 2 figures

- **What's New**: 최근 몇 년 동안, 단백질 기능 예측에서 AlphaFold2에 의해 예측된 고정밀 단백질 구조를 활용하여 예측 정확도가 크게 향상되었습니다. 특히, 단일 종의 단백질 기능 예측 방법은 큰 발전을 이루었지만, 다중 종의 단백질 기능 예측 방법은 여전히 PPI 네트워크 및 서열 기능을 기반으로 한 단계에 있습니다. 이 문제를 해결하기 위해, MSNGO 모델을 제안하며, 이는 구조적 특성과 네트워크 전파 방법을 통합합니다.

- **Technical Details**: MSNGO 모델은 그래프 표현 학습 기술을 활용하여 단백질 구조 접촉 맵에서 아미노산 표현을 추출하고, 그래프 합성 풀링 모듈을 사용하여 단백질 수준의 구조적 특성을 도출하는 구조적 모델을 학습합니다. ESM-2로부터 얻은 서열 특성을 포함시킨 후, 네트워크 전파 알고리즘을 적용하여 정보 집합 및 이질 네트워크 내에서 노드 표현을 업데이트합니다.

- **Performance Highlights**: MSNGO는 서열 기능 및 PPI 네트워크에 의존하는 이전의 다중 종 단백질 기능 예측 방법들보다 우수한 성능을 보였습니다. 이 모델을 통한 검증은 구조적 특성을 사용하는 것이 다중 종 단백질 기능 예측의 정확도를 현저하게 향상시킬 수 있음을 입증하였습니다.



### On Geometrical Properties of Text Token Embeddings for Strong Semantic Binding in Text-to-Image Generation (https://arxiv.org/abs/2503.23011)
- **What's New**: 이번 논문에서는 Text-to-Image (T2I) 모델의 복잡한 장면에서 발생하는 텍스트-이미지 불일치 문제를 해결하기 위해 선택한 방법, 즉 	extbf{TeeMo} 프레임워크를 소개합니다. TeeMo는 텍스트 임베딩을 활용하여 강력한 의미적 결합(semantic binding)을 가능하게 하며, 기존 방법들보다 더 나은 성능을 보여줍니다. 이 방법은 미세 조정 없이도 다양한 데이터셋에서 높은 성능을 발휘하는 특징을 가지고 있습니다.

- **Technical Details**: TeeMo는 Causality-Aware Projection-Out (CAPO)와 Adaptive Token Mixing (ATM)으로 구성되어 있습니다. CAPO는 상호 텍스트 토큰 간의 CA 맵을 구분하는 데 도움을 주며, ATM은 손실 함수를 통해 서로 다른 Noun Phrase (NP) 간의 분리를 강화하면서도 내부 NP 간 결속성을 유지합니다. 연구를 통해 텍스트 토큰 임베딩의 기하학적 특성, 특히 각도 거리와 노름이 CA 맵의 차별화에 중요한 역할을 한다는 것을 경험적 및 이론적으로 분석하였습니다.

- **Performance Highlights**: TeeMo는 다양한 기준과 데이터셋에서 기존 방법보다 항상 우수한 성능을 나타냈습니다. 실험 결과는 TeeMo의 능력이 다양한 복잡한 장면에서 더 나은 텍스트-이미지 정렬을 가능하게 함을 보여줍니다. 이 프레임워크는 특히 여러 객체와 속성이 포함된 장면에서 그 효과를 극대화하는 것으로 확인되었습니다.



### Learning Structure-enhanced Temporal Point Processes with Gromov-Wasserstein Regularization (https://arxiv.org/abs/2503.23002)
Comments:
          Accepted at the Web Conference workshop 2025

- **What's New**: 이 연구에서는 temporal point processes (TPPs)의 클러스터링 구조를 효과적으로 학습하는 새로운 정규화 방법인 Gromov-Wasserstein (GW) 정규화기를 적용합니다. 기존 TPP들은 일반적으로 이벤트 시퀀스의 내재된 클러스터링 구조를 간과하여 해석 가능성이 떨어집니다. 본 연구에서는 최대 우도 추정(maximum likelihood estimation)에 기반하여 클러스터링 구조를 시퀀스 수준의 임베딩에 부과합니다. 이 과정에서 비모수적 TPP 커널을 활용하여 유사성 행렬을 정규화합니다.

- **Technical Details**: 연구에서는 N개의 이벤트로 구성된 이벤트 시퀀스를 정의하고, 이를 기반으로 파라메트릭 TPP를 다변량 카운팅 프로세스로 표현합니다. TPP는 이벤트 시퀀스의 역학을 포착하기 위해 다변량 조건 강도 함수(multivariate conditional intensity function)를 사용하고, 기대되는 순간 발생률을 시간 t에서의 이벤트 이력에 따라 결정합니다. 이 연구에서는 고차원 클러스터링 정규화를 사용하여 유사성 행렬을 정규화하는 커널 행렬을 설계하고, 이를 통해 TPP의 학습을 보다 효과적으로 진행합니다.

- **Performance Highlights**: 실험 결과, 제안된 정규화기를 사용하여 학습된 TPP는 경쟁력 있는 예측 정확도를 달성하며, 클러스터링된 시퀀스 임베딩을 생성합니다. 이는 모델의 해석 가능성을 크게 향상시키면서도 예측 정확도를 유지하는데 기여합니다. 대규모 응용에서도 상대적으로 낮은 계산 복잡도를 유지하며, 모든 클러스터에 독립적인 복잡성으로 확장 가능한 해결책을 제공합니다. 이 방법은 파라메트릭 및 비파라메트릭 TPP 모델의 장점을 결합하여 효과적은 예측 및 클러스터링 성능을 보여줍니다.



### AuditVotes: A Framework Towards More Deployable Certified Robustness for Graph Neural Networks (https://arxiv.org/abs/2503.22998)
Comments:
          20 pages

- **What's New**: 이번 논문에서는 GNNs(그래프 신경망)의 새로운 프레임워크인 	extbf{AuditVotes}를 제안합니다. 이 프레임워크는 높은 클린 정확도(clean accuracy)와 인증된 강건성(certified robustness)을 동시에 달성하는 데 중점을 두고 있습니다. 기존의 방법들이 정확도와 강건성 사이의 중대한 핀치(pinch) 문제에 직면한 반면, AuditVotes는 데이터를 개선하고 예측 일관성을 보장하기 위한 두 가지 핵심 요소인 augmentation과 conditional smoothing을 통합합니다.

- **Technical Details**: 제안된 AuditVotes는 랜덤 스무딩(randomized smoothing) 기법을 사용하여 그래프의 데이터 품질을 향상시키고 예측의 일관성을 높입니다. augmentation은 사전 처리(pre-processing) 단계로 작용하여 랜덤 그래프에서 노이즈를 제거하고 데이터 품질을 크게 개선합니다. 그 후 conditional smoothing은 저품질 예측을 선택적으로 필터링하여 투표의 일관성을 높이는 후처리(post-processing) 단계 역할을 수행합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, AuditVotes는 클린 정확도, 인증된 정확도, 경험적 강건성을 크게 향상시키는 동시에 높은 계산 효율성을 유지합니다. 특히, Cora-ML 데이터 세트에서 공격자가 20개의 엣지를 임의로 추가할 경우, AuditVotes는 기존 랜덤 스무딩에 비해 클린 정확도를 437.1% 개선하고 인증된 정확도를 409.3% 향상시킵니다. 이는 실세계 응용 프로그램에서 인증된 강건한 GNNs의 배포를 위한 중요한 진전을 나타냅니다.



### DC-SGD: Differentially Private SGD with Dynamic Clipping through Gradient Norm Distribution Estimation (https://arxiv.org/abs/2503.22988)
Comments:
          Accepted at IEEE Transactions on Information Forensics & Security

- **What's New**: 새로운 접근 방식인 Dynamic Clipping DP-SGD (DC-SGD)를 제안했습니다. 이 프레임워크는 히스토그램을 활용하여 기울기 정규 분포를 추정하고, 클리핑 임계값 C를 동적으로 조정합니다. DC-SGD는 두 가지 새로운 메커니즘인 DC-SGD-P와 DC-SGD-E를 포함하여, 각각의 기울기 정규의 백분위수 및 기댓값 제곱 오차를 기반으로 클리핑 임계값을 조정합니다. 이러한 동적 조정은 하이퍼파라미터 튜닝의 부담을 크게 줄여줍니다.

- **Technical Details**: DC-SGD는 기울기 정규 분포 추정을 위해 차별적 개인 정보 보호(Differential Privacy) 방식의 히스토그램을 사용합니다. DC-SGD-P는 기울기 정규의 백분위수에 따라 클리핑 임계값을 조정하고, DC-SGD-E는 기댓값 제곱 오차를 최소화하여 클리핑 임계값을 최적화합니다. 이 과정을 통해 추가적인 하이퍼파라미터 조정 없이, 임계값 C를 동적으로 설정할 수 있습니다. Adam 옵티마이저와 통합하여 학습률 조정의 복잡성을 크게 줄이는 것이 특징입니다.

- **Performance Highlights**: 다양한 딥러닝 작업에 대한 실험에서 DC-SGD는 DP-SGD에 비해 하이퍼파라미터 튜닝 속도를 최대 9배 향상시켰습니다. CIFAR10 데이터세트에서는 같은 개인 정보 보호 예산 하에 DP-SGD보다 10.62%의 정확도 향상을 달성했습니다. 실험 결과, DC-SGD-P는 SVHN 데이터세트에서 2.13%의 개선을 보였으며, 전반적으로 DC-SGD 방식이 더 나은 모델 성능을 발휘함을 입증했습니다.



### PartialLoading: User Scheduling and Bandwidth Allocation for Parameter-sharing Edge Inferenc (https://arxiv.org/abs/2503.22982)
Comments:
          16 pages, 9 figures

- **What's New**: 이 논문에서는 다중 사용자 엣지 추론을 위한 매개변수 공유 AI 모델 로딩(PartialLoading) 프레임워크를 개발하여 높은 작업 처리량을 달성하는 방법을 제안합니다. 기존 연구에서 발견된 모델 로딩으로 인한 지연 문제를 해결하기 위해, 서로 다른 AI 모델 간에 공유할 수 있는 매개변수 블록을 활용하여 중복 로딩을 방지하는 접근 방식을 제시합니다. 이 프레임워크는 사용자의 요청을 자원 관리를 통해 효과적으로 스케줄링하고 배치함으로써 총 모델 로딩 시간을 상당히 단축하는 것을 목표로 합니다.

- **Technical Details**: 제안된 접근 방식은 사용자 스케줄링과 대역폭 할당의 두 하위 문제로 나누어져, 이를 순차적으로 해결함으로써 원래 문제를 풀 수 있음을 보여줍니다. 특히, NP-하드 문제로 제안된 사용자 스케줄링 문제는 동적 프로그래밍(Dynamic Programming) 기반의 알고리즘을 사용해 다룰 수 있으며, 일반적인 경우에는 그리디 휴리스틱(greedy heuristic) 방법을 통해 부분 최적해를 도출합니다. 이러한 전략은 사용자 요청을 최적으로 스케줄링하여 공유 매개변수를 재사용함으로써 모델 로딩 시간을 절약하는 것을 목표로 합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 프레임워크는 매개변수 공유를 활용하지 않는 사용자 스케줄링 전략에 비해 마감 기한 제약 하에서도 작업 처리량을 상당히 향상시킵니다. 특히, 최적 모델 로딩 순서를 제공하는 동적 프로그래밍 기반 알고리즘과 그리디 휴리스틱 알고리즘은 기존의 전략보다 현저한 성능 개선을 보였으며, 다양한 AI 모델을 호스팅하는 엣지 서버의 요구에 맞춘 효과적인 해결책을 제시합니다.



### XL-Instruct: Synthetic Data for Cross-Lingual Open-Ended Generation (https://arxiv.org/abs/2503.22973)
- **What's New**: XL-AlpacaEval은 대형 언어 모델(LLM)의 다국어 생성 성능을 평가하기 위한 새로운 벤치마크로 소개됩니다. 기존의 연구들에서 공통적으로 나타난 문제를 해결하기 위해 고품질의 합성 데이터를 생성하는 XL-Instruct 방법이 제안되었습니다. 이 방법을 활용한 세밀한 튜닝을 통해 모델의 성능이 크게 향상되었으며, 구체적으로 GPT-4o-Mini 대비 win rate가 7.4%에서 21.5%로 증가한 것으로 나타났습니다.

- **Technical Details**: 교차 언어 생성(cross-lingual generation)은 특정 언어로 된 질의를 이해하고 다른 언어로 응답을 생성하는 작업입니다. 연구자들은 기계 번역(MT)의 노이즈 문제와 정보 손실 문제를 지적하며, XL-Instruct라는 합성 데이터 생성 기술을 활용하여 고품질의 교차 언어 데이터를 대규모로 생성할 수 있음을 보여주었습니다. 이 방법이 적용된 모델은 영어나 다국어 생성 작업에서도 강력한 제로샷 전이(zero-shot transfer) 성능을 보였습니다.

- **Performance Highlights**: XL-Instruct의 활용으로 다양한 LLM의 교차 언어 성능이 일관되게 개선되었으며, 성능 평가에서 LLM의 'off-the-shelf' 성능이 낮음을 확인했습니다. 실험 결과, 성능 향상뿐만 아니라 다국어 후처리 단계에 XL-Instruct를 포함할 것을 강력히 추천합니다. 또한, XL-AlpacaEval 벤치마크와 XL-Instruct 데이터셋이 향후 교차 언어 LLM 연구에 기여할 것으로 기대됩니다.



### Enhancing Federated Learning Through Secure Cluster-Weighted Client Aggregation (https://arxiv.org/abs/2503.22971)
- **What's New**: 본 논문은 새로운 연합 학습(FL) 프레임워크인 ClusterGuardFL을 소개하고 있으며, 이는 클라이언트 업데이트에 동적으로 가중치를 할당하기 위해 비유사성 점수(dissimilarity scores), K-평균 클러스터링(k-means clustering) 및 재조정 신뢰 점수(reconciliation confidence scores)를 활용합니다. 이 프레임워크는 FL의 공정성과 보안을 강화하여 데이터 독성을 방지하는 목표를 가지고 있습니다. 실험 결과는 구조의 유효성을 입증하여 다양한 데이터 세트에서 모델 성능을 개선할 수 있음을 보여줍니다.

- **Technical Details**: ClusterGuardFL은 전세계 모델과 클라이언트 로컬 모델 간의 비유사성 점수를 이용하여 클러스터를 형성하고, 각 클러스터 내에서 개별 데이터 포인트에 대해 재조정 신뢰 점수를 계산합니다. 이러한 동적 가중치 할당 방식은 학습 과정의 집계 과정에서 적용되어 모델의 강건성과 개인 정보 보호를 강화합니다. FL의 비대칭적 특성과 클라이언트 자원에 따라 다양한 공정성을 제공할 수 있습니다.

- **Performance Highlights**: ClusterGuardFL은 다양한 실제 데이터 세트를 사용한 실험을 통해 그 효과를 입증하였고, 기존 FL 방법들에 비해 성능과 안정성을 향상시키는 결과를 도출하였습니다. 각 클라이언트의 기여를 공정하게 측정하여 전반적인 모델 성능을 개선함과 동시에, 데이터 독성 공격에 대한 강고함을 증가시켰습니다. 이를 통해 CLusterGuardFL은 FL 시스템의 공정성과 보안을 동시에 향상시키기 위한 효과적인 솔루션을 제공합니다.



### HRET: A Self-Evolving LLM Evaluation Toolkit for Korean (https://arxiv.org/abs/2503.22968)
- **What's New**: 최근 한국 대형 언어 모델(LLM)의 발전에 따라 여러 벤치마크와 평가 방법론이 생겨났지만, 표준화된 평가 프레임워크의 부재로 인해 불일치한 결과와 비교의 한계가 발생했습니다. 이를 해결하기 위해 우리는 한국 LLM에 특화된 오픈 소스인 HRET(하래 평가 도구킷)를 소개합니다. HRET는 로짓 기반 점수 측정, 정확 일치, 언어 불일치 패널화 및 LLM-as-a-Judge 평가를 포함한 다양한 평가 방법을 통합합니다.

- **Technical Details**: HRET는 모듈화된 레지스트리 기반 아키텍처를 갖추고 있으며 주요 벤치마크(HAE-RAE Bench, KMMLU, KUDGE, HRM8K)와 여러 인퍼런스 백엔드(vLLM, HuggingFace, OpenAI와 호환되는 엔드포인트)를 통합합니다. 자가 진화하는 파이프라인을 통해 HRET는 지속적인 발전을 지원하며, 한국어 NLP 연구에서 재현 가능하고 공정한 평가의 기초를 제공합니다. 이 도구킷은 직관적인 API와 간소화된 커맨드라인 인터페이스를 통해 연구자와 실무자 모두에게 접근성을 높여줍니다.

- **Performance Highlights**: HRET를 통해 한국 LLM의 평가에서 나타나는 재현성 및 비교 용이성 문제를 해결할 수 있습니다. 평가 방법의 표준화를 통해 HRET는 한국어 모델의 성능을 보다 신뢰성 있게 평가할 수 있는 기반을 제공합니다. 이 도구킷은 Apache License 2.0 하에 공개될 예정으로, 널리 사용되고 커뮤니티의 기여를 장려합니다.



### Student-Powered Digital Scholarship CoLab Project in the HKUST Library: Develop a Chinese Named-Entity Recognition (NER) Tool within One Semester from the Ground Up (https://arxiv.org/abs/2503.22967)
Comments:
          47 pages. Presented and submitted to DADH2024 conference (this https URL)

- **What's New**: 2024년 2월부터 HKUST 도서관은 AI 활용에 대한 교육의 범위를 확장하여, 학생들이 최신 기술을 활용할 수 있는 프로젝트인 '디지털 장학금 (Digital Scholarship, DS) CoLab'에 참여하도록 하고 있습니다. 이 프로그램은 뛰어난 인재를 양성하고, 학생들이 실용적 맥락에서 고급 기술을 활용할 수 있도록 지원하는 데 중점을 두고 있습니다.

- **Technical Details**: 이 논문에서는 2024 봄학기(2024년 2월부터 5월까지) 동안 학생 도우미로 참여한 Sherry Yip Sau Lai와 Berry Han Liuruo가 개발한 중국어 개체명 인식(Named-Entity Recognition, NER) 도구의 개발 과정을 다룹니다. 연구 및 계획 단계부터 실행에 이르기까지 한 학기 내에 완전한 제품을 개발하는 여정을 자세히 설명합니다.

- **Performance Highlights**: 이 프로젝트는 학생들이 중심이 되어 협력의 정신을 함양하며, 실용적 학습을 우선시하는 혁신적인 교육 모델의 힘과 잠재력을 보여줍니다. 도서관은 연간 1-2개의 프로젝트를 제공하여 학생들이 실무에서 고급 기술을 활용하고 도서관 운영 과제를 해결하는 데 기여하도록 지원합니다.



### Late Breaking Results: Breaking Symmetry- Unconventional Placement of Analog Circuits using Multi-Level Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2503.22958)
Comments:
          2 pages, 3 figures, Proceedings of the 62nd ACM/IEEE Design Automation Conference (DAC), 2025

- **What's New**: 이번 연구는 레이아웃 의존 효과(Layout-dependent effects, LDEs)가 아날로그 회로 성능에 미치는 영향을 강조합니다. 전통적으로 설계자들은 LDEs로 인한 변동을 완화하기 위해 대칭 배치를 사용해왔지만, 비선형적 특성으로 인해 기존 방법들이 효율적이지 못했습니다. 이 연구는 새로운 방식의 아날로그 레이아웃 디자인 공간을 탐색하기 위해 목표 지향적(multi-level), 다중 에이전트(multi-agent) 강화 학습(Q-learning) 프레임워크를 제안합니다.

- **Technical Details**: 제안하는 프레임워크는 아날로그 회로의 레이아웃 최적화를 위한 혁신적인 접근법을 제공합니다. 기존의 비기계적 방법인 simulated annealing과 비교하여, 제안된 방법은 보다 우수한 성능 변동 특성을 보여줍니다. 이 연구는 아날로그 레이아웃 자동화 분야에서 다중 에이전트 강화 학습(multi-agent RL)의 최초의 적용 사례로 주목받고 있습니다.

- **Performance Highlights**: 제안된 방법은 최신 레이아웃 기법들에 비해 성능 변동 측면에서 향상된 결과를 나타냅니다. 아날로그 회로의 효율성을 높이는 새로운 경로를 제시함으로써, 향후 연구 및 산업 응용에 긍정적인 영향을 미칠 것으로 기대됩니다. 이는 중요한 혁신으로써 아날로그 회로 설계의 새로운 가능성을 열어줍니다.



### Can LLMs Support Medical Knowledge Imputation? An Evaluation-Based Perspectiv (https://arxiv.org/abs/2503.22954)
Comments:
          10 pages, 3 figures, AMIA

- **What's New**: 이번 연구는 Medical Knowledge Graphs (KGs)의 불완전성을 해결하기 위해 Large Language Models (LLMs)를 활용하는 새로운 접근법을 제시합니다. 특히 LLM이 생성한 치료 매핑을 체계적으로 평가하여 신뢰성을 검증했습니다. 또한, LLM 사용 시 발생할 수 있는 위험 요소와 기존 임상 지침과의 불일치 문제에 대해 심층적으로 분석했습니다.

- **Technical Details**: 연구에서는 LLM이 결여된 치료 관계를 보완하기 위해 ICD, Mondo, ATC와 같은 의료 코딩 시스템을 활용했습니다. LLM은 임상 문헌과 약물 정보를 바탕으로 질병과 치료 간의 관계를 생성하고, 기존의 KGs와 비교할 수 있는 평가 프레임워크를 개발했습니다. 이 프레임워크는 LLM이 생성한 출력과 임상적으로 승인된 기준 간의 정합성을 분석하여 LLM의 강점과 한계를 파악합니다.

- **Performance Highlights**: 연구 결과 LLM은 일부 유용한 치료 제안을 생성할 수 있는 반면, 일관성 결여 및 오류 생성 가능성도 드러났습니다. 이러한 발견은 의료 분야에서 LLM의 통합이 신중해야 함을 강조하며, LLM의 안전한 활용을 위해 하이브리드 접근법의 필요성을 규명합니다. 전반적으로, 이 연구는 의료 KG의 개선을 위한 LLM 활용에 대한 경고를 제시하고, 투명한 검증의 중요성을 강조합니다.



### SUV: Scalable Large Language Model Copyright Compliance with Regularized Selective Unlearning (https://arxiv.org/abs/2503.22948)
- **What's New**: 이번 연구에서는 SUV (Selective Unlearning for Verbatim data)라는 새로운 선택적 학습 해제 프레임워크를 소개합니다. 이 프레임워크는 LLM이 저작권이 있는 콘텐츠를 기억하는 것을 방지하면서도 모델의 전체 유틸리티를 유지하도록 설계되었습니다. 저작권 침해 사례를 포착한 데이터셋을 구축하고, 직접 선호 최적화(Direct Preference Optimization)을 사용하여 기억된 콘텐츠를 대체하는 방안을 제시합니다.

- **Technical Details**: SUV 프레임워크는 전통적인 방법과 달리 슬라이딩 윈도우(sliding-window) 메커니즘을 사용하여 기억된 구간을 세분화하여 식별합니다. 또한, DPO를 통해 플라거리(표절)된 내용을 제거하고 무작위로 생성된 텍스트로 대체합니다. 이 과정에서 그라디언트 프로젝션(gradient projection)과 피셔 정보 정규화(Fisher information regularization)를 통합하여 모델의 성능 저하를 최소화합니다.

- **Performance Highlights**: 500개의 저명한 책으로 구성된 대규모 데이터셋을 사용하여 SUV의 성능을 검증하였으며, 저작권이 있는 콘텐츠의 기억을 크게 줄이면서도 관련 없는 작업에서의 성능에 미치는 영향은 미미하다는 사실을 입증했습니다. 우리의 접근 방식은 공개 기준에서도 우수한 성과를 보여주며, 기존의 방법들에 비해 유용성을 유지하면서 저작권 위험 완화에 효과적임을 강조합니다.



### DATAWEAVER: Authoring Data-Driven Narratives through the Integrated Composition of Visualization and Tex (https://arxiv.org/abs/2503.22946)
Comments:
          Accepted to EuroVis 2025. Published in Computer Graphics Forum. DOI: https://doi.org/10.1111/cgf.70098

- **What's New**: 본 논문에서는 데이터 중심 스토리텔링의 제작 과정을 개선하기 위해 DataWeaver라는 통합 저작 프레임워크를 제안합니다. 이 시스템은 시각화에서 텍스트로, 텍스트에서 시각화로의 작성을 모두 지원하며, 사용자는 데이터 사실에 기반한 내러티브를 생성할 수 있습니다. 데이터 시각화의 하이라이트를 통해 관련 내러티브 콘텐츠를 유도하는 'call-out' 상호작용을 포함하여, 데이터 기반 내러티브 작성을 보다 직관적이고 효율적으로 만들어 줍니다.

- **Technical Details**: DataWeaver는 데이터 시각화 및 내러티브 작성을 통합하는 독창적인 접근 방식을 채택하고 있습니다. 우선 사용자는 차트를 생성하고, 시각적 요소를 강조하며, 선택된 데이터 사실에 내러티브를 연결하여 스토리를 구현할 수 있습니다. 또한, 기존 내러티브에서 인사이트를 시각화하는 'text-to-vis' 작성을 지원하여, 저자가 데이터 탐색을 시작할 수 있는 적절한 추천을 제공합니다.

- **Performance Highlights**: 13명의 참여자를 대상으로 한 평가에서 DataWeaver의 유용성과 사용성을 입증하였습니다. 사용자 피드백을 통해 필터링 메커니즘, 차트 추천 및 사용자 맞춤형 옵션 개선 사항이 도출되었습니다. 이 연구는 데이터 중심 저작 경험을 효과적으로 지원하는 방향으로 시스템을 정교화할 수 있는 기회를 제공합니다.



### Adaptive Interactive Navigation of Quadruped Robots using Large Language Models (https://arxiv.org/abs/2503.22942)
Comments:
          10 pages, 9 figures

- **What's New**: 본 논문에서는 복잡한 환경에서 로봇 네비게이션을 위한 새로운 접근법인 Adaptive Interactive Navigation (AINav)을 제안합니다. 기존의 전통적인 네비게이션 방식이 충돌 없는 경로 생성에 초점을 맞추고 있어 재난 구역이나 혼잡한 창고와 같은 환경에서의 유용성이 제한됐던 점을 해결하고자 합니다. AINav는 환경과 능동적으로 상호작용하여 원래 도달할 수 없었던 목표에 대한 경로를 생성하는 방법론을 제공합니다.

- **Technical Details**: AINav는 대규모 언어 모델(LLM)을 기반으로 한 작업 계획, 동작 계획, 그리고 적응형 재계획의 세 가지 주요 구성 요소로 이루어져 있습니다. 이를 통해 로봇은 프리미티브 트리(primitive tree)를 만들어 작업을 분해하고, 강화 학습(RL)으로 훈련된 기술 라이브러리를 활용하여 복잡한 환경 내에서의 이동 및 상호작용을 원활히 수행할 수 있습니다. 또한 재계획 메커니즘은 새로운 관측 정보를 해석하여 실행 계획을 신속하게 조정할 수 있는 능력을 제공합니다.

- **Performance Highlights**: 종합적인 시뮬레이션 및 실제 실험을 통해 AINav의 효과성과 적응력이 다양한 시나리오에서 입증되었습니다. 이 방법론은 로봇이 불확실한 환경에서 능동적으로 반응하고 새로운 정보를 활용하여 작업을 완수할 수 있도록 지원합니다. 특히, 고난도 네비게이션 작업 처리와 새로운 관측에 대한 빠른 적응이 가능하다는 점에서 의미가 큽니다.



### FairSAM: Fair Classification on Corrupted Data Through Sharpness-Aware Minimization (https://arxiv.org/abs/2503.22934)
- **What's New**: 이 논문에서는 서로 다른 인구 집단 간의 성능 저하를 평가하기 위해 새로운 메트릭을 도입하고, Fairness-oriented한 전략을 통합한 FairSAM이라는 새로운 프레임워크를 제안합니다. 기존의 Sharpness-Aware Minimization(SAM) 방법이 전반적인 모델의 강건성을 높이는 데 기여했지만, 인구 집단 간의 성능 불균형 문제를 효과적으로 해결하지 못했다는 점을 강조합니다. FairSAM은 혼합된 데이터 환경에서 강건성과 공정성을 동시에 달성할 수 있도록 설계되었습니다.

- **Technical Details**: FairSAM 프레임워크는 인스턴스-재가중(SAM) 기법을 기반으로 하며, 다양한 노이즈 유형으로 오염된 데이터세트에 적합하게 조정됩니다. 이는 각 샘플에 손상이 가해지는 것을 근사하여 전체 배치에서 처리할 수 있도록 설계되었습니다. 새롭게 제안된 `Corrupted Degradation Disparity` 메트릭을 통해 정확도 저하를 다양한 인구 집단 사이에서 정량화할 수 있습니다.

- **Performance Highlights**: 여러 실제 데이터 세트에 대한 실험을 통해 FairSAM은 공정성과 강건성 모두에서 우수한 성과를 보여주었습니다. 특히, 제안한 공정성 메트릭인 Corrupted Degradation Disparity에서 개선된 점수를 기록하였고, 대부분의 경우 가장 낮은 최악의 그룹 정확도를 달성했습니다. FairSAM은 전통적인 공정성 방법들의 한계를 극복하고 각 인구 집단에 걸쳐 높은 정확도와 공정성을 동시에 유지하는 결과를 보여주었습니다.



### Predictive Traffic Rule Compliance using Reinforcement Learning (https://arxiv.org/abs/2503.22925)
Comments:
          12 pages, 7 figures. Preprint submitted to IEEE ITSC 2025

- **What's New**: 이번 논문에서는 자율주행차의 경로 계획에 안전성과 규제 준수를 통합하는 새로운 접근 방식을 제안합니다. 이 방법은 모션 플래너(motion planner)와 심층 강화 학습(deep reinforcement learning) 모델을 결합하여 잠재적인 교통 법규 위반을 예측합니다. 특히, 독일 도로 교통 규정의 주요 규칙들을 포함한 규칙 책(rule book)을 설정하여 복잡한 교통 정보를 처리하는 그래프 기반 상태 표현을 사용합니다.

- **Technical Details**: 이 연구는 전통적인 액터-크리틱(actor-critic) 구조에서 액터 네트워크를 모션 계획 모듈로 대체하여 예측 가능한 경로 생성을 보장하는 혁신점을 제시합니다. 모션 플래너는 상태 가치 함수(state-value function)의 예측 값을 직접적으로 활용하여 최적의 경로를 선택하며, 이는 장기적인 교통 법규 위반을 방지하는 데 기여합니다. 또한, 그래프 신경망(Graph Neural Network, GNN)을 사용하여 환경 상태를 효과적으로 표현하고, 다양한 도로 레이아웃과 주변 차량 개수를 포괄적으로 처리합니다.

- **Performance Highlights**: 모델은 공개된 독일 고속도로 데이터셋을 사용하여 실험을 수행했으며, 계획 지평선을 넘어서는 교통 법규 위반을 예측하고 방지할 수 있음을 보여 주었습니다. 특히, 복잡한 교통 조건에서 안전성을 크게 향상시키는 결과를 얻었습니다. 이러한 방식은 자율주행차의 의사결정 과정에서 강화 학습과 모션 계획 방법을 성공적으로 통합하여 향후 교통 법규 위반을 예방하는 새로운 경로를 열어줍니다.



### Enhancing DeepLabV3+ to Fuse Aerial and Satellite Images for Semantic Segmentation (https://arxiv.org/abs/2503.22909)
- **What's New**: 이번 논문에서는 기존의 DeepLabV3+ 아키텍처를 개선하여 저해상도 위성 데이터와 고해상도 항공 이미지를 결합하는 새로운 접근법을 제안하고 있습니다. 특히, 새로운 전이 합성곱 레이어 블록을 도입하여 두 번째 입력을 업샘플링하고 고수준의 특징들과 융합하는 방식을 개발했습니다. 이를 통해 다양한 정보 소스의 유용성을 극대화하며, 세분화 작업에서의 정확성과 성능을 향상시키고자 합니다.

- **Technical Details**: DeepLabV3+ 아키텍처를 기반으로 한 이 연구에서는 블라인 업샘플링을 가중치 기반 업샘플링 모듈로 대체하여 디코더 단계에서의 세분화 맵 재구성을 개선하였습니다. 다양한 업샘플링 방법에 대한 실험을 수행해, 위성 데이터 주입 과정에서 더 나은 성능을 발휘할 수 있는 방안을 찾아냈습니다. 이러한 과정은 저해상도 위성 이미지와 고해상도 항공 이미지 간의 효과적인 융합을 가능하게 합니다.

- **Performance Highlights**: 논문에서는 두 데이터 소스의 융합을 통해 평균 교차 비율(Mean Intersection over Union, mIoU)을 84.91%로 달성했으며, 데이터 증강 없이도 우수한 성과를 보였습니다. 이는 고해상도 항공 이미지의 세부사항과 위성 이미지의 다채로운 정보가 결합됐을 때의 성능을 잘 보여줍니다. 또한, 기존의 DeepLabV3+ 아키텍처의 성능을 크게 개선하는 결과를 도출했습니다.



### Pairwise Matching of Intermediate Representations for Fine-grained Explainability (https://arxiv.org/abs/2503.22881)
- **What's New**: 이번 논문에서는 기존의 딥러닝 모델 설명 기법의 한계를 극복하기 위해 PAIR-X라는 새로운 설명 방법을 제안합니다. 이 방법은 모델의 중간 활성화(activation) 및 역전파된 중요도 점수를 활용하여 매우 세밀하고 국부적인 쌍별 비주얼 설명을 생성합니다. 동물 및 건물 재식별(re-ID)을 주요 사례 연구로 삼아, 35개의 공개 재식별 데이터셋에서 다양한 설명 기법들과 비교하여 정 qualitative한 개선 결과를 보여주었습니다.

- **Technical Details**: PAIR-X는 각각의 이미지 쌍에 대해 더 유용하고 해석 가능한 설명을 제공하기 위해 중간 모델 활성화와 역전파된 중요도 점수를 결합합니다. 저자들은 새로운 정량적 평가 메트릭을 제안하여 PAIR-X의 시각화가 올바른 이미지 매치에 대해 더욱 그럴듯하게 나타나는 것을 입증했습니다. 이러한 기술적 접근 방식은 이미지의 유사도 점수가 동일하더라도 올바른 매치와 잘못된 매치를 보다 쉽게 구별할 수 있도록 도와줍니다.

- **Performance Highlights**: 연구 결과, 동물 재식별 전문가들은 PAIR-X의 시각화가 기존 방법들보다 개선되었다고 만장일치로 동의하였으며, 그들의 작업에 즉시 적용 가능하다고 평가했습니다. PAIR-X는 특히 차별화된 주목을 받았으며, 이러한 해석 가능성의 향상은 전반적으로 모델의 신뢰성과 유용성을 높이는 데 기여할 것으로 기대됩니다.



### Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models (https://arxiv.org/abs/2503.22879)
- **What's New**: 최근 State Space Models (SSMs)은 메모리 사용의 일관성과 높은 성능 덕분에 Transformers에 대한 매력적인 대안으로 떠오르고 있습니다. 그러나 클라우드 서비스나 자원 제한 장치에서 SSM을 확장하는 데 필요한 저장 요구량과 컴퓨팅 파워가 도전 과제가 되고 있습니다. 이를 해결하기 위해, Quamba2는 다양한 상황에 대한 효율성을 고려하여 W8A8, W4A8 및 W4A16 비트폭을 지원하는 포스트 트레이닝 양자화(PTQ) 프레임워크를 제공합니다.

- **Technical Details**: Quamba2는 SSM의 채널 순서 보존과 활성화 지속성을 기반으로 한 오프라인 양자화 방식을 제안합니다. 입력 데이터의 정렬 및 클러스터링을 통해 8비트 양자화를 처리하며, 상태 그룹별로 양자화를 적용하여 입력 의존 매개변수를 정밀하게 최적화합니다. 이러한 방식은 SSM의 속성을 활용하여 양자화 정확도를 높이고, 메모리 요구 사항을 줄이며, 성능 저하를 최소화합니다.

- **Performance Highlights**: Quamba2-8B는 여러 최신 SSM 양자화 방법들을 초월하여, 예비 채우기 및 생성 단계에서 각각 1.3배 및 3배 빠른 속도를 제공하며, 4배의 메모리 감소도 달성합니다. 평균 1.6%의 정확도 손실로 6개의 제로샷 작업에서 성능을 유지하는 동시에, MMLU 데이터셋에서의 평가를 통해 모델의 일반화 및 내구성을 입증하였습니다.



### Understanding Inequality of LLM Fact-Checking over Geographic Regions with Agent and Retrieval models (https://arxiv.org/abs/2503.22877)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 사실 확인(fact-checking)에서의 성능을 다양한 지역과 시나리오를 통해 평가하였습니다. 특히, LLM의 성능이 지역에 따라 다르게 나타나는 점을 강조하여, 이를 통해 오정보의 확산을 방지하는 방법을 모색하였습니다.

- **Technical Details**: 연구는 600개의 사실 확인이 이루어진 진술(statement)을 포함한 데이터셋을 사용하여 세 가지 실험 세팅을 평가하였습니다: (1) 진술만 있을 때, (2) 위키피디아 접근이 가능한 LLM 기반 에이전트를 사용할 때, (3) 공식 사실 확인이 제공되는 Retrieval-Augmented Generation (RAG) 시스템을 사용할 때입니다. 이러한 설정을 통해 각 모델의 성능 차이를 분석하였습니다.

- **Performance Highlights**: 연구 결과, GPT-4, Claude Sonnet 및 LLaMA를 포함한 어떤 LLM을 사용하더라도, 선진국(Global North) 출처의 진술이 개발도상국(Global South) 출처의 진술보다 상당히 더 높은 성능을 나타냈습니다. 이러한 차이는 위키피디아 에이전트 기반 시스템을 사용한 경우에 더욱 확대되었습니다. 이는 지역적 특성을 반영할 수 있는 자료의 균형 잡기와 강력한 검색 전략의 필요성을 강조하고 있습니다.



### Teaching LLMs Music Theory with In-Context Learning and Chain-of-Thought Prompting: Pedagogical Strategies for Machines (https://arxiv.org/abs/2503.22853)
Comments:
          11 pages, 4 figures, 3 tables. Published in Volume 1 of the Proceedings of the 17th International Conference on Computer Supported Music Education (CSME 2025). Presented on 3 April 2025 in Porto, Portugal

- **What's New**: 본 연구에서는 ChatGPT, Claude, Gemini와 같은 대형 언어 모델(LLMs)이 음악 이론(Theory of Music)을 배우는 데 있어 기본적인 능력을 평가합니다. 구체적으로, in-context learning과 chain-of-thought prompting을 사용하여 LLMs에 복잡한 개념을 어떻게 교육할 수 있는지를 탐구합니다. 이를 통해 사람 학습자에게 사용되는 교육 전략이 기계 교육에 어떻게 적용되는지를 분석합니다.

- **Technical Details**: 이 연구에서는 캐나다 왕립 음악원(RCM)의 6급 시험 문제를 사용하여 LLMs의 성능을 평가합니다. 주제에는 간격 및 화음 인식, 조 감지, 종지 분류, 그리고 리듬 분석이 포함되어 있습니다. 또한 ABC, Humdrum, MEI, MusicXML과 같은 다양한 음악 인코딩 형식이 이러한 작업에 적합한지 평가하며, 실험은 맥락이 없을 때와 있을 때 모두 진행됩니다.

- **Performance Highlights**: 결과에 따르면 맥락 없이 실행할 경우 MEI 형식의 ChatGPT가 52%로 가장 높은 성능을 보였습니다. 그러나 맥락이 있을 경우 MEI 형식의 Claude가 75%로 가장 높은 성능을 기록하였습니다. 향후 연구에서는 프롬프트를 더욱 정교하게 다듬고, 보다 고급 음악 이론 개념으로 범위를 확대할 계획입니다.



### RobuNFR: Evaluating the Robustness of Large Language Models on Non-Functional Requirements Aware Code Generation (https://arxiv.org/abs/2503.22851)
- **What's New**: 이번 논문에서는 Non-Functional Requirements (NFRs)을 고려한 코드 생성을 위한 LLMs의 강 robustness를 평가하는 RobuNFR 프레임워크를 제안합니다. RobuNFR은 설계, 가독성, 신뢰성, 성능의 네 가지 NFR 차원을 평가하며, 다양한 프롬프트 변형, 회귀 테스트(regression testing), 그리고 여러 작업 흐름(workflows)을 사용하여 LLM이 생성한 코드 품질의 변화를 분석합니다.

- **Technical Details**: RobuNFR은 코드 품질을 평가하기 위해 코드 설계, 신뢰성, 가독성 및 성능의 네 가지 차원을 포함하여, 각 차원과 관련된 메트릭을 사용합니다. 주로 세 가지 방법론—프롬프트 변화, 회귀 테스트, NFR 인지 코드 생성 작업 흐름—을 통해 LLM의 NFR 인지 코드 생성 능력을 평가합니다. 이를 통해 LLM의 강 robustness 문제가 어떻게 발생하는지를 파악할 수 있습니다.

- **Performance Highlights**: 실험 결과, NFR를 고려하지 않은 코드 생성은 안정적인 Pass@1 점수를 유지하는 반면, NFR이 포함된 프롬프트를 사용할 경우 최대 39%의 Pass@1 점수 감소와 함께 표준 편차가 증가함을 보여주었습니다. 이 외에도, NFR 인식 코드 생성은 기본적인 코드 정확성과 NFR 관련 메트릭의 저하를 나타내며, 다양한 프롬프트 간의 민감성을 강조합니다. 이러한 발견은 개발 과정 중 LLM의 NFR 인지 코드 생성을 개선하기 위한체계적인 방법의 필요성을 강조합니다.



### Nonhuman Primate Brain Tissue Segmentation Using a Transfer Learning Approach (https://arxiv.org/abs/2503.22829)
- **What's New**: 본 연구에서는 비인간 영장류(Non-Human Primates, NHP)의 뇌 조직 분할을 향상시키기 위한 새로운 접근방식으로 STU-Net과 transfer learning을 결합한 방법을 제안합니다. 이는 인간의 뇌 MRI 데이터에서 전이된 지식을 활용하여 NHP 뇌 MRI의 분할 정확도를 높이는데 기여합니다. 주목할 만한 점은 기존의 한계를 극복하여 NHP 뇌 특유의 미세 해부학적 세부 사항을 효과적으로 포착할 수 있다는 점입니다.

- **Technical Details**: 비인간 영장류의 뇌 이미지는 주로 해부학적 차이와 해상도 제한으로 인해 분할이 어렵습니다. 연구에서는 STU-Net을 통해 이러한 도전을 해결하고, 특히 작은 피질 하 구조체인 피각(putamen)과 시상(thalamus) 등의 분할 성능을 향상시켰습니다. 최종적으로, DSC(다중 클래스의 Dice Similarity Coefficient)는 0.88 이상, IoU(Intersection over Union)는 0.8 이상, 그리고 HD95는 7 이하의 성능을 달성했습니다.

- **Performance Highlights**: 제안된 방법은 비인간 영장류의 뇌 조직 분할에서 새로운 기준을 제시합니다. 특히 작은 구조물의 분할에서 기존의 한계를 초월한 성과를 보였으며, 이는 진화 신경 과학 및 인류 건강에 관련된 신경 질환의 전임상 연구를 가속화할 잠재력을 가지고 있습니다. 나아가, 이 연구는 다중 클래스 뇌 조직 분할에 대한 강력한 방법론을 제시하여 향후 연구의 발전에 기여할 것입니다.



### Data-driven worker activity recognition and picking efficiency estimation in manual strawberry harvesting (https://arxiv.org/abs/2503.22809)
- **What's New**: 이 연구에서는 상업적인 딸기 수확에서 수확자 효율성을 계산하기 위한 실제 시스템이 개발되었습니다. 수확한 과일의 무게, 지리적 위치, 카트 이동을 실시간으로 기록하기 위해 인스트루멘티드 피킹 카트가 사용되었습니다. CNN-LSTM 기반의 심층 신경망을 통해 수확자의 활동을 'Picking'과 'NoPicking'으로 분류하여 수확자 효율성을 추정할 수 있었습니다.

- **Technical Details**: 이 연구에서 사용된 인스트루멘티드 피킹 카트(iCarritos)는 두 개의 로드셀과 GNSS 수신기, 관성 측정 장치(IMU)를 갖추고 있어 수확된 과일의 무게, 지리적 위치 및 카트의 이동 데이터를 기록합니다. CNN-LSTM 기반의 알고리즘이 개발되어 수확 데이터를 분석하고, 수확자 효율성을 평가하기 위해 활동 인식이 이루어졌습니다. 이러한 시스템은 실제 수확 기간 동안 발생하는 다양한 비생산적인 작업들을 감지하고 영향을 줄 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, CNN-LSTM 모델은 평균 F1 점수 0.974의 정확도로 활동 인식을 수행했습니다. 수확 데이터 분석을 통해 수확자는 평균 73.56%의 시간을 적극적으로 딸기를 수확하며, 트레이를 채우는 데 평균 6.22분의 시간이 소요되었습니다. 제안된 기술은 상업 규모에서 자동화된 작업 모니터링과 수확 최적화에 기여할 수 있습니다.



### DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers (https://arxiv.org/abs/2503.22796)
- **What's New**: 이 논문에서는 Multimodal Diffusion Transformers (MMDiT)의 주의(attention) 메커니즘을 가속화하기 위한 새로운 압축 방법인 DiTFastAttnV2를 소개합니다. 이 방법은 이미징 품질을 유지하면서도 68%의 attention FLOPs 절감과 1.5배의 속도 향상을 달성합니다. 또한, 기존 접근 방법들과의 차별화를 위해 헤드 중심의 동적 주의 조절 및 캐싱 메커니즘을 도입합니다.

- **Technical Details**: DiTFastAttnV2는 MMDiT의 주의 패턴들을 정밀하게 분석하여, 서로 다른 주의 헤드의 비율과 행동을 기반으로 한 동적 캐싱 메커니즘을 형성합니다. 이 방법은 효율적인 융합 커널(Efficient Fused Kernel)을 통해 주의 메커니즘을 최적화하며, 모델 압축을 위한 검색 비용을 최소화합니다. 이를 통해 다차원 압축 계획 탐색을 현저히 단축시킵니다.

- **Performance Highlights**: 2K 이미지 생성 작업에서 DiTFastAttnV2는 주의 계산량을 68%까지 줄이면서도 생성 품질을 유지합니다. 또한, 전체적인 생성 속도를 1.5배 향상시켜 고해상도 이미지와 긴 비디오 생성에서도 효과적으로 활용할 수 있습니다. 이러한 성능 향상은 MMDiT의 복잡한 주의 패턴을 효율적으로 처리하는 데 기인합니다.



### Patronus: Bringing Transparency to Diffusion Models with Prototypes (https://arxiv.org/abs/2503.22782)
- **What's New**: 이 연구에서는 Denoising Diffusion Probabilistic Models (DDPMs) 기반의 해석 가능한 확산 모델인 Patronus를 소개합니다. Patronus는 ProtoPNet에서 영감을 받아 프로토타입 네트워크를 통합하여 생성 프로세스에 영향을 미치는 프로토타입을 추출하고 조절할 수 있습니다. 이 모델은 주석이나 텍스트 프롬프트 없이도 작동하며, 따라서 해석 가능성과 투명성을 증대시키는 새로운 경로를 제공합니다.

- **Technical Details**: Patronus의 설계는 이미지의 패치를 기반으로한 프로토타입 추출 및 표현 모듈을 포함하고 있습니다. 이 모듈은 입력 이미지를 패치 기반 특징 표현으로 변환하고, 각 프로토타입에 대한 유사성 점수를 계산하여 확산 과정의 조건으로 사용합니다. 또한, 프로토타입의 활성화 벡터를 통해 입력 이미지에 대한 의미 정보를 인코딩하는 방법을 제안합니다.

- **Performance Highlights**: Patronus는 세밀하게 의미 있는 시각적 특징을 잘 포착하며, 최신 SOTA (State-of-the-Art) 방법들과 비교해 경쟁력 있는 생성 품질을 달성합니다. 또한, 이 모델은 데이터셋 내에서 원치 않는 상관관계를 진단할 수 있는 잠재력을 보여주며, 생성 모델의 편향을 줄이고 공정성을 증진하는 데 유용한 도구로 활용될 수 있습니다.



### Post-Incorporating Code Structural Knowledge into LLMs via In-Context Learning for Code Translation (https://arxiv.org/abs/2503.22776)
- **What's New**: 이 논문에서는 코드 번역을 위한 접근 방식으로, 사전 훈련된 큰 언어 모델(LLM)에 코드 구조적 지식을 효과적으로 통합하는 방법을 제안합니다. 기존의 전통적인 방법들이 복잡한 아키텍처와 손실 함수에 의존하는 반면, 이 방법은 샘플 선택 전략을 정보 이론적인 관점에서 다시 탐구합니다. 이를 통해 정보 커버리지를 기반으로 한 더 정밀하고 일반적인 목표를 설정하며, 이를 추상 구문 트리(AST)를 통해 정량화할 수 있는 방법인 Coverage of AST (CAST)를 도입합니다.

- **Technical Details**: CAST는 주어진 테스트 소스 코드와 샘플 집합에 대해 AST를 추출하여 코드 트리의 서브 트리를 커버하는 방식으로 작동합니다. 이 방법은 NP-난해한 CAST 최대화를 위해 특별한 지문(fingerprint)을 사용하여, 트리에 효율적인 서브 트리 커버링과 샘플 선택 전략을 제시합니다. 구체적으로, 공존 행렬을 구성하여 서브 트리를 선택하고, 그 과정에서 표준 부분 모듈화 최적화 문제라는 것을 증명함으로써 이론적인 정당성을 제공합니다.

- **Performance Highlights**: 실험 결과, CAST 방법은 전통적인 유사도 및 다양성 기반의 샘플 선택 전략에 비해 LLM의 성능을 현저히 향상시키는 것을 보여주었습니다. 이번 연구를 통해 코드 구조적 지식은 모델 훈련 중 간과되는 경향이 있지만, 테스트 시에 효과적으로 통합될 수 있다는 중요한 통찰을 제시합니다. 단순히 모델 크기나 훈련 데이터를 늘리는 것만으로는 코드의 구문 지식이 충분히 확보되지 않음을 강조하고 있습니다.



### GroundHog: Revolutionizing GLDAS Groundwater Storage Downscaling for Enhanced Recharge Estimation in Bangladesh (https://arxiv.org/abs/2503.22771)
- **What's New**: 이번 연구는 GLDAS의 저해상도 데이터(25 km)를 활용하여 고해상도(2 km) 지하수 수위(GWL)를 예측하는 GroundHog 모델을 개발하였습니다. 이 모델은 다년간의 데이터를 기반으로 최대 및 최소 GWL을 예측할 수 있도록 설계되어, 효과적인 수자원 관리와 정책 결정에 기여합니다. 또한, ML 모델을 활용하여 데이터의 공백을 줄이고, 새로운 'Pseudo-Ground Truth' 데이터셋을 생성하여 다양한 지점의 GWL을 예측할 수 있게 되었습니다.

- **Technical Details**: GroundHog 모델은 다수의 수리-지질학적 요소(Hydro-geological Factors, HGFs)를 고려하여 GWL 예측에 필요한 입력 데이터를 수집합니다. 최대 및 최소 GWL 예측을 위한 모델은 Random Forest Regressor를 사용하며, 2001년부터 2022년까지의 실제 수치 데이터를 ground truth로 사용합니다. 이 연구에서는 Normalized Difference Water Index (NDWI)와 Normalized Difference Vegetation Index (NDVI) 등의 다양한 HGFs가 활용되며, 이들 요소는 수위 예측의 정확도를 높이는 데 기여합니다.

- **Performance Highlights**: 제안된 모델은 최대 GWL과 최소 GWL의 예측에서 각각 R^2 점수 0.855 및 0.963을 달성하였습니다. 고해상도 GWL을 생성하는 Upsampling 모델은 GLDAS의 입력을 기반으로 R^2 점수 0.96이라는 뛰어난 성과를 거두었습니다. 연구 결과는 2003-2024년 기간 동안의 GLDAS 데이터를 고해상도로 보강하여 지하수 재충전 추정을 가능케 하고, 자원 관리를 위한 중요한 트렌드를 제시합니다.



### MediTools -- Medical Education Powered by LLMs (https://arxiv.org/abs/2503.22769)
Comments:
          19 pages, 17 figures, 2 tables. Code available at this https URL

- **What's New**: AI (인공지능)의 발전은 급속도로 진행되고 있으며, 2022년 말부터 등장한 대형 언어 모델(LLMs)은 의학을 포함한 다양한 분야에서 이 기술을 채택할 기회를 창출하고 있습니다. 이 연구 프로젝트는 MediTools라는 AI 의료 교육 도구를 개발하여 의료 교육을 향상시키고 작업 흐름 문제를 해결하는 데 중점을 두고 있습니다. 이 프로토타입 애플리케이션은 실제 임상 시나리오를 시뮬레이션하는 대화형 도구 개발에 초점을 맞추고 있습니다.

- **Technical Details**: MediTools는 실생활의 다양한 피부질환을 나타내는 실제 환자 이미지를 사용하여 LLMs와 상호작용하는 피부과 사례 시뮬레이션 도구를 포함합니다. 이 플랫폼은 사용자가 진단 능력을 연습하고 임상 의사 결정 능력을 향상시키는 데 도움을 줍니다. 또한, 연구 논문에 대한 깊이 있는 통찰을 제공하는 AI 향상 PubMed 도구와 다양한 의학 전문 분야에 대한 기사 요약을 제공하는 Google News 도구가 포함되어 있습니다.

- **Performance Highlights**: 의료 전문가와 학생들 대상으로 실시된 포괄적인 설문조사를 통해 MediTools의 효과성과 사용자 만족도에 대한 초기 피드백을 수집하였습니다. 이러한 피드백은 애플리케이션의 추가 개발 및 개선에 대한 통찰력을 제공합니다. 이 연구는 의료 교육을 혁신하기 위한 AI 기반 도구의 잠재력을 보여주며, 지속적인 학습과 기술 개발을 위한 확장 가능하고 상호작용적인 플랫폼을 제공합니다.



### Boosting Large Language Models with Mask Fine-Tuning (https://arxiv.org/abs/2503.22764)
- **What's New**: 이 논문에서는 Mask Fine-Tuning (MFT)이라는 새로운 LLM 파인튜닝 패러다임을 소개합니다. MFT는 모델의 무결성을 의도적으로 파괴함으로써 놀라운 성능 향상을 이끌 수 있음을 보여줍니다. 이 연구는 LLM 파인튜닝의 기존 프로토콜을 통해 구조적 무결성이 반드시 필요하지 않음을 주장합니다.

- **Technical Details**: MFT는 사전 훈련된 LLM에 대해 작동되며, 이 LLM은 전체 파인튜닝(Fine-Tuning)을 통해 훈련됩니다. 여기서 MFT는 이 훈련된 LLM에 이진 마스크를 추가하여 특정 파라미터를 선택적으로 마스킹합니다. 최적화는 스트레이트-스루 그래디언트 추정기(gradient estimator)를 활용하여 이루어지며, 이는 배포형 학습(supervised learning)으로 안내됩니다.

- **Performance Highlights**: MFT는 다양한 도메인과 백본을 통해 일관된 성능 향상을 보였습니다. 예를 들어, LLaMA2-7B와 LLaMA3.1-8B 모델을 사용하여 각각 1.95% 및 1.88%의 평균 성능 향상을 기록했습니다. 이 연구는 기존의 파인튜닝 방식과 비교하여 성능을 크게 향상시킬 수 있는 새로운 관점을 제공합니다.



### The Cost of Local and Global Fairness in Federated Learning (https://arxiv.org/abs/2503.22762)
- **What's New**: 본 논문은 다중 클래스 연합 학습(Federated Learning; FL) 설정에서 글로벌(global) 및 로컬(local) 공정성을 고려하여 최소한의 정확도가 얼마나 손실되는지를 조사하는 프레임워크를 제안합니다. 기존의 연구들은 일반적으로 글로벌 공정성(예: 배제된 편향) 또는 로컬 공정성(예: 각 클라이언트 내에서의 형평성) 중 하나만 다루어 왔습니다. 새로운 프레임워크는 베이지안 최적 점수 함수에서 공정한 결과 예측기를 도출하는 간단한 후처리(post-processing) 알고리즘을 제공합니다.

- **Technical Details**: 우리의 제안된 프레임워크는 문제를 볼록 프로그램(convex program)으로 서술하며, 효율적인 해결을 위해 수신자 동작 특성(Receiver Operating Characteristic; ROC) 표면을 사용하여 근사합니다. 이 벨록 프로그램을 해결하는 데 복잡성이 높기 때문에, 단순형(simplex)을 사용하여 문제를 선형 프로그램(linear program; LP)으로 재구성합니다. 이 과정에서 우리의 알고리즘은 일반적으로 사용되는 공정성 지표(예: Statistical Parity (SP), Equal Opportunity (EOp), Equalized Odds (EO))를 지원합니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 현재 상용 여성 예측 분야(State of the Art; SOTA) 대비 정확도-공정성(tradeoff)에서 우수한 성능을 발휘하며, 통신 비용을 20% 절감하고 계산 비용을 33% 줄였습니다. 이로써 다중 클래스 환경에서도 효율적으로 공정성을 유지할 수 있는 가능성을 보여줍니다.



### Data Poisoning in Deep Learning: A Survey (https://arxiv.org/abs/2503.22759)
- **What's New**: 이 논문에서는 딥러닝(deep learning)에서의 데이터 오염(data poisoning) 공격을 체계적으로 조사하는 전문적인 서베이를 제공합니다. 기존 연구들은 데이터 오염에 대한 일반적인 관점을 제공했지만, 본 서베이는 딥러닝 모델에 특화된 심층 분석을 통해 문제를 다룹니다. 특히 대형 언어 모델(LLMs)에서의 데이터 오염과 그로 인한 보안 위험을 조명하는 것이 특징입니다.

- **Technical Details**: 데이터 오염 공격은 크게 모델 공격(model attacks), 회피 공격(evasion attacks), 데이터 오염 공격(data poisoning attacks)으로 분류됩니다. 데이터 오염 공격은 훈련 데이터에 악성 샘플을 주입하여 훈련 중에 모델의 신뢰성을 저하시킵니다. 본 서베이는 이러한 공격 방법론을 체계적으로 분류하고 분석하는 틀을 제공합니다.

- **Performance Highlights**: 이 연구에서 발표한 내용은 데이터 오염 공격의 개념적 및 기술적 성격을 포착하는 체계적인 분류를 포함하며, 이는 연구자들에게 이 분야를 탐구하는 데 유용한 지침이 됩니다. 또한, LLMs의 고유한 취약성을 강조하며, 이 분야의 향후 연구 방향을 제시함으로써 데이터 오염 연구의 발전에 기여할 것으로 기대됩니다.



### Towards an intelligent assessment system for evaluating the development of algorithmic thinking skills: An exploratory study in Swiss compulsory schools (https://arxiv.org/abs/2503.22756)
- **What's New**: 이 연구에서는 스위스 교육 시스템에 CT(Computational Thinking) 기술을 통합하기 위한 포괄적인 프레임워크를 개발하고 있습니다. 특히 알고리즘 설계 능력(Algorithmic Thinking, AT)에 초점을 맞추어 대규모 평가를 위한 활동을 구성했습니다. 이러한 접근법은 CT 개발에 영향을 미치는 활동의 특성과 이들 기술을 평가하는 방법을 명확히 설명합니다.

- **Technical Details**: 연구진은 CT의 실제적이고 발전적인 특성을 포착하는 역량 모델을 개발하였습니다. 이는 인지 능력, 나이 및 맥락에 맞춰 설계된 활동을 안내하는 역할을 합니다. 또한, AT 기술을 평가하기 위한 두 가지 변형의 활동을 개발하였으며, 하나는 비디지털 아티팩트(Non-digital artefacts)를 기반으로 하고 다른 하나는 디지털 아티팩트(Digital artefacts)를 기반으로 하는 자동 평가 시스템입니다.

- **Performance Highlights**: 제안된 도구는 스위스 내 다양한 연령대 및 교육적 맥락에서 AT 역량을 측정할 수 있음을 입증하였습니다. 평가 시스템은 실시간 확률적 평가(Real-time probabilistic assessment)를 제공하며, 전체 점수가 아닌 각 기술별 평가를 수행합니다. 또한, AT 역량은 점진적으로 발전하며, 전반적인 성별 차이는 없으나 학교 수준에서는 아티팩트 기반 환경의 맥락에 따라 차이가 나타납니다.



### Reasoning Under Threat: Symbolic and Neural Techniques for Cybersecurity Verification (https://arxiv.org/abs/2503.22755)
- **What's New**: 이 논문은 사이버 보안에서 자동화된 추론의 역할에 대한 포괄적인 개요를 제공합니다. 특히 접근 제어, 프로토콜 설계 및 취약성 탐지와 같은 다양한 도메인에서 보안 속성을 검증하기 위한 논리 체계의 사용을 분석합니다. SOTA 도구와 AI의 통합을 탐색하며, 확장성, 구성 가능성 및 다층 보안 모델링과 같은 중요한 연구 갭을 강조합니다. 앞으로의 연구 방향으로서는 형식적이고 자동화된, 설명 가능한 추론 기술의 발전을 촉진하는 것을 목표로 제시합니다.

- **Technical Details**: 형식 논리(formal logic) 및 자동화된 추론(automated reasoning)은 사이버 보안 분석의 중추 역할을 합니다. 다양한 논리 체계는 보안 특성(예: 기밀성, 인증)과 시스템 동작을 모델링하고 검증하기 위한 도구로 활용됩니다. 예를 들어,일차 논리(First-Order Logic)와 같은 전통적인 논리 시스템은 보안 속성의 공리적 모델링에 사용되며, 시간 논리(Temporal Logic)는 프로토콜의 시간 의존적 속성을 지정하는 데 필수적입니다.

- **Performance Highlights**: 형식적 방법(formal methods)이 적용된 사례로는 Coq, Lean, K 프레임워크와 같은 플랫폼에서의 보안 프로토콜 검증과 자동화된 취약성 탐지가 있습니다. 이러한 도구는 공리 증명과 기호적 자동화를 통해 프로토콜의 핵심 속성을 검증할 수 있는 기능을 제공합니다. 최근 하이브리드 신경-기호 접근법(neural-symbolic approaches)의 부상은 사이버 보안 분야에서 새로운 가능성을 제시하고 있습니다.



### Model Lake: a New Alternative for Machine Learning Models Management and Governanc (https://arxiv.org/abs/2503.22754)
- **What's New**: 이 논문에서는 머신러닝(ML) 모델 관리를 위한 중앙 집중식 프레임워크인 '모델 레이크(Model Lake)' 개념이 제시되었습니다. 기존의 모델 관리 방식은 여러 저장 시스템을 이용하며 비효율적이고 표준화된 방법론의 부족으로 어려움을 겪어왔습니다. 이러한 배경에서, 데이터 레이크의 아이디어를 차용하여, 조직 내 데이터셋, 코드 및 모델을 통합하여 관리할 수 있는 새로운 접근 방식이 필요해졌습니다.

- **Technical Details**: 모델 레이크는 ML 모델의 전체 수명주기를 관리하는 통합 플랫폼을 제공합니다. 모델 레이크는 'Github'와 유사하게 ML 모델을 카탈로그화하고 주석을 달아 조직 내 이해관계자들 간에 쉽게 접근할 수 있도록 만들어집니다. 또한, 데이터 엔지니어와 데이터 과학자들이 실시간으로 협업하고 실험할 수 있는 동적 작업 공간을 제공합니다. 이를 통해 모델 재사용성이 향상되고 모델 성능 모니터링이 용이해집니다.

- **Performance Highlights**: 모델 레이크 접근 방식을 채택함으로써 조직들은 모델 생애 주기 관리, 모델 탐색, 감사 및 재사용성을 높일 수 있습니다. 실제 사례를 통해 데이터, 코드 및 모델 관리의 변혁적 영향을 입증하고, 이로 인해 운영 효율성이 증대하며 거버넌스가 향상되는 것을 보여줍니다. 조직 내 데이터 분석 파이프라인의 모든 요소를 효과적으로 통합함으로써, AI 솔루션의 배포 속도가 빨라질 것으로 기대됩니다.



### From Individual to Group: Developing a Context-Aware Multi-Criteria Group Recommender System (https://arxiv.org/abs/2503.22752)
Comments:
          The 16th International Conference on Management of Digital EcoSystems, Nov 2024, Naples, Italy

- **What's New**: 이 연구에서는 다양한 개인의 선호를 고려해야 하는 그룹 의사결정 상황에서의 문제를 해결하기 위해 Context-Aware Multi-Criteria Group Recommender System (CA-MCGRS)을 개발하였습니다. 기존의 추천 시스템은 개별화에 효과적이지만 그룹 의사결정에서는 갈등하는 선호와 다양한 평가 기준을 다루는 데 한계가 있습니다. CA-MCGRS는 이러한 맥락적 요소와 다중 기준을 통합하여 추천의 정확도를 높이는 데 초점을 맞추었습니다.

- **Technical Details**: CA-MCGRS는 Multi-Head Attention 메커니즘을 활용하여 서로 다른 특징의 중요성을 동적으로 조정합니다. 이 모델은 교육 데이터셋에서 다양한 평가와 맥락 변수를 고려하여 실험을 진행하였습니다. 이러한 접근 방식은 추천 정확도를 향상시키는데 기여하는 것으로 나타났습니다.

- **Performance Highlights**: 실험 결과, CA-MCGRS는 네 가지 시나리오 모두에서 다른 접근 방식들보다 지속적으로 우수한 성과를 보였습니다. 따라서 본 연구의 발견은 그룹 추천 시스템의 개발에 있어 맥락과 다중 기준 평가를 포함하는 것이 얼마나 중요한지를 강조합니다.



### Advancing Spatiotemporal Prediction using Artificial Intelligence: Extending the Framework of Geographically and Temporally Weighted Neural Network (GTWNN) for Differing Geographical and Temporal Contexts (https://arxiv.org/abs/2503.22751)
- **What's New**: 이번 논문은 인공지능 신경망(Artificial Neural Networks, ANNs)의 수학적 프레임워크를 개선하여 예측 범죄 모델을 향상시키는 것을 목표로 합니다. 특히, 지리적이고 시간적인 가중치를 고려한 회귀(Geographically and Temporally Weighted Regression, GTWR) 문제의 해결을 위한 새로운 반-해석적 접근 방식을 제안하고 런던 범죄 데이터에 적용하여 높은 정확도를 입증합니다. 이 논문은 GTWNN(Geographically and Temporally Weighted Neural Network) 프레임워크에 대한 수학적 발전을 소개하며, 범죄 예측 문제에 신경망의 적절한 적용을 위한 다양한 수학적 확장을 제안합니다.

- **Technical Details**: GTWNN 모델은 Fotheringham et al. (2015)에 의해 소개된 GTWR의 확장으로, 외부 요인의 정보를 비선형적으로 결합할 수 있는 장점을 제공합니다. 그러나 본 연구는 GTWNN 모델에서 외부 요인에 연결된 계수 함수의 연속성을 결여하고 역사적 맥락 정보를 활용하지 못하는 한계를 지적합니다. 이를 극복하기 위해, 세 가지 수학적 확장을 통해 범죄 데이터에 대한 적합성을 높이고자 하였습니다.

- **Performance Highlights**: 논문에서는 런던 및 디트로이트 데이터 세트에 대해 다섯 가지의 새로운 ANN 모델을 적용하여 평가하였습니다. 각 모델의 성능을 비교한 결과, 역사 의존 모듈(history-dependent module)이라는 특정 확장모듈이 다른 모듈들을 일반적으로 초과하는 것으로 나타났습니다. 따라서, 제안된 방법들은 더욱 맥락지향적이고 정확한 ANN 접근 방식의 기초를 제공하여, 범죄 예측 모델링의 적합성을 향상시키는 데 기여할 것으로 보입니다.



### Adaptive Clipping for Privacy-Preserving Few-Shot Learning: Enhancing Generalization with Limited Data (https://arxiv.org/abs/2503.22749)
- **What's New**: 본 논문에서는 데이터 Privacy(프라이버시)와 모델 성능 간의 기본적인 트레이드오프를 해결하기 위해 Meta-Clip이라는 새로운 알고리즘을 제안합니다. 이는 Differentially Private (DP) 메타-러닝 알고리즘의 성능을 최대화하며 프라이버시 보호를 동시에 달성하는 것을 목표로 하며, Mini-Batch training(미니배치 학습) 중에 클리핑 조정 방법을 동적으로 적용하여 성능 향상을 꾀합니다. 또한, 기존의 프라이버시 기술들과 비교하여 우수한 프라이버시-유틸리티 균형을 보여깁니다.

- **Technical Details**: Meta-Clip은 DP 모델에 대한 다양한 메타-러닝 알고리즘, 즉 DP-Reptile과 DP-MetaSGD에 통합되어 있으며, 민감한 정보의 노출을 조절할 수 있도록 Adaptive Clipping(어댑티브 클리핑) 방법을 사용합니다. 훈련 과정 중 클리핑 임계치를 조정함으로써 Finely-tuned control(세부 조정)을 가능하게 하여 overfitting(과적합)을 줄이고, 메타-러닝 모델의 일반화 성능을 크게 향상시킵니다. 저자들은 실험을 통해 방법론이 처리 가능하다는 것을 입증하였습니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 수행된 실험을 통해, 메타-러닝 알고리즘의 성능이 향상되었으며, 특히 몇 개의 라벨 데이터만으로도 효과적인 학습이 가능함을 입증했습니다. 제안한 방법론은 기존의 프라이버시 보장 기법들보다 뛰어난 개인정보 보호와 유틸리티 보장을 자랑하며, 실제 어플리케이션에서 데이터가 제한된 환경에서도 적용 가능함을 보여줍니다. Adaptive Clipping 방법은 few-shot learning(소수 샷 학습) 문제에서 안전하고 정확한 모델 개발에 큰 기여를 할 것으로 기대됩니다.



### Ignite Forecasting with SPARK: An Efficient Generative Framework for Refining LLMs in Temporal Knowledge Graph Forecasting (https://arxiv.org/abs/2503.22748)
Comments:
          To be published in the 30th International Conference on Database Systems for Advanced Applications (DASFAA 2025)

- **What's New**: 이 논문에서는 Temporal Knowledge Graph (TKG) 예측에 대한 새로운 접근 방식으로 SPARK를 소개합니다. SPARK는 Seqeunce-level Proxy-Adapting 프레임워크로, LLMs를 활용한 TKG 예측의 문제를 해결합니다. 기존 LLM 기반 방법의 제한사항인 입력 길이, 출력 생성의 비효율성 및 리소스 집약적 보완 문제를 해결하려고 합니다.

- **Technical Details**: SPARK는 두 가지 주요 혁신을 통해 TKG 예측을 향상시킵니다. 첫째, Beam Sequence-Level Generation(BSL) 접근을 통해 TKG 예측을 top-K 시퀀스 생성 작업으로 전환하여, 단일 전파에서 효율적으로 다음 엔티티 분포를 생성합니다. 둘째, TKG Adapter를 통해 전통적인 TKG 모델을 훈련 가능한 프록시 어댑터로 활용하며, 이를 통해 LLM 출력의 보완을 수행하고 입력 길이 문제와 비용 및 시간이 많이 드는 파인튜닝을 극복합니다.

- **Performance Highlights**: 다양한 데이터셋을 활용한 실험을 통해 SPARK의 예측 성능과 일반화 능력 및 높은 효율성을 검증하였습니다. 실험 결과 SPARK는 기존 LLM 성능을 일관성 있게 향상시키며, IT 튜닝 모델과 비교해 경쟁력 있는 성능을 달성한 것으로 나타났습니다.



### LeForecast: Enterprise Hybrid Forecast by Time Series Intelligenc (https://arxiv.org/abs/2503.22747)
- **What's New**: 이 논문에서는 멀티디스플리너리 예측(multidisciplinary forecasting) 분야에서의 수요 증가에 대응하기 위해 leforecast{}라는 기업 인텔리전스 플랫폼을 소개합니다. 이 플랫폼은 시간 시계열 예측(time series forecasting) 작업에 최적화 되어 있으며, 역사적 데이터를 기반으로 인사이트를 도출하고 미래를 예측하는데 초점을 맞추고 있습니다. 또한, 대형 기초 모델(large foundation model)의 활용을 통해 복잡한 비즈니스 맥락을 해석하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: LeForecast는 시간 시계열 데이터와 다원 소스 정보(multi-source information)의 고급 해석을 결합한 세 가지 기둥 모델링 엔진(three-pillar modelling engine)을 통합하고 있습니다. 이 엔진은 대형 기초 모델(Le-TSFM), 다중 모달(multimodal) 모델 및 하이브리드 모델(hybrid model)로 구성되어 있어 시나리오 기반 예측을 가능하게 합니다. 또한, 모델 풀(model pool), 모델 프로파일링 모듈 및 원래 모델 아키텍처에 대한 두 가지 다른 융합 접근 방식을 포함하여, 모델 효율성을 극대화합니다.

- **Performance Highlights**: 실험 결과는 router-based fusion network와 대형 및 소형 모델의 조정을 통한 효율성을 입증하며, 이는 모델 개발 및 유지 관리의 중복 비용을 절감하는 데 유용합니다. 세 가지 산업 사용 사례에서 LeForecast의 배치를 리뷰한 결과, 이 플랫폼은 효율적이고 경쟁력 있는 성능을 제공하는 것으로 나타났습니다. 이 연구는 시간 시계열 기술의 연구와 기업 가속화에 기여할 수 있을 것으로 기대됩니다.



### Susceptibility of Large Language Models to User-Driven Factors in Medical Queries (https://arxiv.org/abs/2503.22746)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)이 의료 분야에서 어떻게 사용되고 있는지 살펴보았으며, 사용자 질문의 phrasing(문구)와 임상 정보의 완전성이 진단 정확도에 미치는 영향을 조사했습니다. 특히, 잘못된 정보의 framing(프레이밍), 출처의 권위, 모델의 persona(페르소나), 그리고 주요 임상 세부 정보의 생략이 LLM의 출력 신뢰성에 미치는 영향을 분석했습니다.

- **Technical Details**: 연구는 perturbation test와 ablation test라는 두 가지 실험을 통해 진행되었습니다. perturbation test에서는 다양한 주장 강도를 가진 잘못된 외부 의견을 도입하였고, ablation test에서는 특정 환자 정보 범주를 제거하며 모델의 성능을 평가했습니다. 이를 위해 MedQA와 Medbullets와 같은 공개 데이터셋을 사용해 GPT-4o, Claude 3.5 Sonnet 등과 같은 여러 상용 및 오픈 소스 모델을 비교했습니다.

- **Performance Highlights**: 모든 모델은 사용자 주도의 잘못된 정보에 취약했으며, 특히 상용 모델은 확정적이고 권위 있는 언어의 영향을 가장 많이 받았습니다. assertive tone(주장적인 어조)은 정확도에 가장 큰 부정적인 영향을 미쳤으며, ablation test에서는 신체 검사 결과와 실험실 결과의 생략이 성능 하락의 가장 큰 원인이 되었습니다. 결국, 연구 결과는 잘 구조화된 프롬프트와 완전한 임상 맥락의 필요성을 강조하며, 사용자들이 복잡한 사례에서 권위 있는 잘못된 정보 프레이밍을 피하고 전체 임상 정보를 제공해야 함을 시사합니다.



### Adaptive Integrated Layered Attention (AILA) (https://arxiv.org/abs/2503.22742)
- **What's New**: 이번 연구에서는 Adaptive Integrated Layered Attention (AILA)라는 신경망 아키텍처를 제안합니다. AILA는 다양한 네트워크 층 간의 적응형(feature reuse) 기능을 위해 밀집 스킵 연결(dense skip connections)과 여러 메커니즘을 융합하여 구성되어 있습니다. AILA는 가격 예측, 이미지 인식, 감정 분석의 세 가지 도전 과제를 평가받았으며, 기존의 강력한 딥러닝 모델과 유사한 성능을 보이면서도 훈련 및 추론 시간을 크게 단축시켰습니다.

- **Technical Details**: AILA는 두 가지 아키텍처, 즉 AILA-Architecture 1과 AILA-Architecture 2로 나뉘어 있습니다. AILA-Architecture 1은 층 간의 연결 메커니즘으로 간단한 선형 층(linear layers)을 사용하고, AILA-Architecture 2는_attention_ 메커니즘을 구현하여 이전 층의 출력을 선택적으로 강조합니다. 이러한 아키텍처는 각기 다른 태스크에 대해 개별적으로 훈련되며, 다양한 네트워크 깊이에서 관련 기능을 유연하게 재사용함으로써, 강력한 성능 향상을 이루어냅니다.

- **Performance Highlights**: AILA는 세 가지 기준 벤치마크에서 강력한 성능 지표를 달성했습니다. 가격 예측, CIFAR-10 데이터셋에 대한 이미지 인식, IMDB 영화 리뷰 데이터셋의 감정 분석에서 AILA-Architecture 1 및 2 모두 LSTM, Transformer, CNN과 같은 기존의 강력한 기준 모델과 경쟁하며 이를 초월하는 성과를 보여주었습니다. 결과적으로 AILA는 일반적인 고정 연결 방식이 아닌, 적응형 정보 흐름을 통해 복잡한 태스크에서 성능을 향상시키는 새로운 길을 열었습니다.



### CSPO: Cross-Market Synergistic Stock Price Movement Forecasting with Pseudo-volatility Optimization (https://arxiv.org/abs/2503.22740)
- **What's New**: 이번 연구에서는 Cross-market Synergy with Pseudo-volatility Optimization (CSPO) 프레임워크를 소개합니다. CSPO는 외부 선물 시장의 지식을 활용한 깊은 신경망 아키텍처를 구현하여, 주식 가격 예측 능력을 향상시킵니다. 기존의 예측 모델에 비해 더 정교한 기능을 제공하며, 모델의 최적화 과정에서 주식별 변동성을 고려합니다.

- **Technical Details**: CSPO는 Bi-level Dense Pricing Transformer (BDP-Former)라는 변환 기반 딥 뉴럴 아키텍처를 활용하여 선물 시장 정보를 통합합니다. 이를 통해 주식의 복잡한 상호작용을 포착하고, 예측 과정에서의 가격 안정성을 정량화하기 위해 의사 변동성(pseudo-volatility) 개념을 도입합니다. 다양한 주식의 변동성을 반영함으로써, CSPO는 보다 정확하고 견고한 예측 결과를 제공합니다.

- **Performance Highlights**: 실제 주식 시장 데이터셋을 사용한 광범위한 실험 결과, CSPO는 기존 모델에 비해 우수한 성능을 보였습니다. 특히, CSI300 지수를 포함한 다양한 평가 지표에서 더 나은 수익 곡선을 달성하였습니다. 이러한 결과는 CSPO 프레임워크와 그 구성 요소들이 효과적임을 입증합니다.



### Cyborg Data: Merging Human with AI Generated Training Data (https://arxiv.org/abs/2503.22736)
- **What's New**: 이 논문은 대규모 평가에서 사용되는 자동 채점 시스템(Automated Scoring, AS)의 새로운 접근 방식을 제안합니다. 기존 AS 시스템은 방대한 수의 수작업 점수가 필요했으며, 이는 시간과 비용이 많이 소모됩니다. 하지만 저자들은 큰 Generative Language Model이 소량의 데이터로도 새로운 작업에 일반화하는 능력을 갖춘 점을 바탕으로, ‘Teacher’ 모델이 ‘Student’ 모델을 가르치는 모델 증류 파이프라인을 제안합니다.

- **Technical Details**: 이 연구는 ‘Cyborg Data’라 불리는 데이터셋을 생성하는 방법을 설명합니다. Teacher 모델은 소규모 수작업 점수 데이터로 학습하고, 이를 기반으로 추가 점수를 생성하여 Student 모델을 훈련시킵니다. 이 과정은 수작업 데이터를 단 10%만 사용하더라도 Student 모델이 전체 데이터셋에서 교육된 모델과 유사한 성능을 보일 수 있음을 보여줍니다.

- **Performance Highlights**: 연구 결과, Student 모델이 Cyborg Data에서 훈련될 경우, 예전 수작업 점수로 전부 훈련된 모델과 유사한 성능을 나타내는 것으로 나타났습니다. 이는 제안된 방법이 AS 개발 비용을 크게 줄일 수 있는 가능성을 제시하며, 향후 연구 방향에 대한 논의도 포함되어 있습니다.



### Ancestral Mamba: Enhancing Selective Discriminant Space Model with Online Visual Prototype Learning for Efficient and Robust Discriminant Approach (https://arxiv.org/abs/2503.22729)
Comments:
          10 pages, 3 figures

- **What's New**: 이 논문에서는 Ancestral Mamba라는 새로운 접근 방식을 제안합니다. 이 방법은 온라인 프로토타입 학습을 선택적 차별 모델(Selective Discriminant Space Model)에 통합하여 효율적이고 강력한 온라인 지속 학습에 기여합니다. Ancestral Prototype Adaptation(APA)와 Mamba Feedback(MF)이라는 두 가지 핵심 요소가 포함되어 있어, 이전 지식을 보존하며 새로운 도전 과제에 적응하는 능력을 강화합니다.

- **Technical Details**: Ancestral Mamba는 비슷한 기술을 대체하기 위해 온라인 프로토타입 학습 기술을 적용합니다. APA 모듈은 클래스의 본질적인 특성을 포착하는 프로토타입을 지속적으로 학습하고 유지합니다. MF 메커니즘은 도전적인 패턴에 집중하여 의사결정 경계를 정제하는 시각적 피드백 루프 역할을 합니다.

- **Performance Highlights**: CIFAR-10 및 CIFAR-100 데이터셋에 대한 광범위한 실험을 통해 Ancestral Mamba가 기존의 최첨단 모델(State-Of-The-Art)과 비교하여 정확도 및 잊기 방지 측면에서 뛰어난 성능을 발휘함을 입증했습니다. 이 접근 방식은 진화하는 시각 패턴에 적응하는 능력을 보여주며, 효율성과 견고함을 함께 제공합니다.



### Zero-Shot LLMs in Human-in-the-Loop RL: Replacing Human Feedback for Reward Shaping (https://arxiv.org/abs/2503.22723)
Comments:
          20 pages, 2 figures, 5 Tables

- **What's New**: 본 논문에서는 강화학습에서 에이전트가 원치 않는 행동을 하게 되는 보상 불일치 문제를 해결하기 위한 두 가지 주요 기여를 제안합니다. 첫째, 자연어 처리(NLP)에 국한되지 않고, 제로샷(zero-shot) 오프더셸프(Off-the-shelf) 대규모 언어 모델(LLMs)을 사용하여 지속적인 제어작업에서 보상을 형성하는 방법을 확장했습니다. 둘째, LLM이 사람의 피드백에서 편향을 식별하고 수정할 수 있는 하이브리드 프레임워크(LLM-HFBF)를 도입하여, 편향이 반영된 보상 형성 과정을 개선했습니다.

- **Technical Details**: 이 연구에서는 MuJoCo 환경 내의 지속적인 제어 작업에서 보상 형성을 효과적으로 수행하기 위해 제로샷(zero-shot) LLM을 활용하였습니다. LLM-HFBF 프레임워크를 통해, LLM이 인간 피드백에서 발생할 수 있는 잠재적인 편향을 식별하고 이를 수정할 수 있도록 하여, 보상 형성 과정에 반영합니다. 이러한 접근 방식은 LLM의 제한과 인간 감독에서 발생하는 편향을 모두 다루어 더 균형 잡히고 신뢰할 수 있는 시스템을 만듭니다.

- **Performance Highlights**: 실험 결과에 따르면, 편향된 인간 피드백은 에이전트의 학습 성능을 크게 저하시킵니다. 평균 에피소딕 보상(AER)은 비편향 접근에서는 28.472인 반면, 보수적 편향이 있는 경우 7.039로 감소하였습니다. 반면, LLM 기반 접근 방식은 맞춤형 에지 케이스에서도 비편향 피드백과 일치하는 AER를 유지하며 성능을 향상시킵니다.



### Why Representation Engineering Works: A Theoretical and Empirical Study in Vision-Language Models (https://arxiv.org/abs/2503.22720)
- **What's New**: 이 논문에서는 Representation Engineering (RepE)을 Vision-Language Models (VLMs)로 확장하여, 다중모달 표현이 어떻게 보존되고 변환되는지를 분석합니다. 특히, 비주얼 입력이 언어 지식을 무시하고 잘못된 응답을 생성하는 문제를 해결하려고 합니다. 이 접근법을 통해 AI 시스템의 투명성, 공정성 및 안정성을 향상시킬 수 있는 새로운 이론적 프레임워크를 개발하였습니다.

- **Technical Details**: RepE는 고수준 개념인 정직성, 권한, 사실성 등을 이해하고 제어할 수 있는 구조적 프레임워크를 제공합니다. 새로운 연구에서는 (1) 주된 고유값(principal eigenvalue)이 신경 활동의 안정성을 보장하는 방향으로 작용하며, (2) 스펙트럼 갭(spectral gap)이 줄어들면서 여러 개념 간의 미세한 차이를 포착하는 서브도미넌트(eigenvector) 고유벡터를 허용한다는 두 가지 주요 현상을 설명합니다.

- **Performance Highlights**: 이 연구는 VLM에서의 RepE의 성공적인 적용을 보여주며, 신경 활동이 계층 간에 어떻게 전파되고 안정성을 유지하는지를 시각화하여 검증합니다. 이러한 접근 방식은 다양한 고수준 개념에 대한 분석 및 제어를 위한 안정적인 기반을 마련하여 AI 시스템의 복잡한 현상을 보다 이해하고 해석 가능하게 합니다.



### TRIDIS: A Comprehensive Medieval and Early Modern Corpus for HTR and NER (https://arxiv.org/abs/2503.22714)
Comments:
          6 pages, 3 figures, 2 tables

- **What's New**: 이 논문에서는 TRIDIS(Tria Digita Scribunt)라는 오픈 소스의 중세 및 초기 현대 원고 코퍼스를 소개합니다. TRIDIS는 다양한 레거시 컬렉션을 통합하여 메타데이터 설명을 포함하고 있으며, 이전의 연구에서 일부 하위 집합이 활용되었지만, 이번에는 전체 코퍼스를 통합적으로 설명합니다. 새로운 전반적인 코퍼스 구성은 문학적 전통을 연구하는 데 필요한 언어적 측면을 고려하여 설계되었습니다.

- **Technical Details**: TRIDIS는 여러 개의 오픈 소스 하위 컬렉션을 결합하여 구성되며, 반영된 데이터는 공통된 스키마를 따른 일관된 구조로 포장됩니다. 또한, 연구에서는 고유한 Outlier-driven partition 전략을 제안하여 훈련 데이터와 테스트 데이터 간의 도메인 중첩 문제를 해결하고, 복잡한 레이아웃과 드문 어휘를 가진 예제를 테스트 세트로 정의하여 HTR 모델의 일반화 능력을 평가합니다. 이러한 접근 방식은 TrOCR 및 MiniCPM2.5와 같은 사전 훈련 모델을 사용하여 검증되었습니다.

- **Performance Highlights**: TRIDIS 코퍼스는 다체로운 문서 유형을 다루며, 복잡한 레이아웃과 큰 필기 변동성을 갖춘 자료에 중점을 둡니다. 초기 실험 결과는 outlier-driven 테스트 분할을 사용할 때 HTR 모델의 성능이 크게 저하된다는 것을 보여주며, 엄격한 평가 방법론의 중요성을 강조합니다. 이를 통해 HTR 훈련 및 평가에서 이전에 간과된 난제를 드러내고, 전통적인 HTR뿐만 아니라 후속 NLP 작업을 지원하는 리소스를 제공합니다.



### Chirp Localization via Fine-Tuned Transformer Model: A Proof-of-Concept Study (https://arxiv.org/abs/2503.22713)
Comments:
          19 pages, 8 figures

- **What's New**: 이번 연구는 electroencephalogram (EEG) 스펙트로그램에서 특이한 환율(Chirp-like patterns)을 감지하기 위한 자동화 도구를 개발하는 데 초점을 맞추고 있습니다. 기존의 연구에서는 이러한 패턴을 발견하는 방법이 부족했으며, 본 연구는 Vision Transformer (ViT) 모델의 세부 조정을 통해 이를 해결하고자 하였습니다. 이와 더불어 저차원 적응(Low-Rank Adaptation, LoRA)을 활용하여 모델의 적응 속도를 향상시켰습니다.

- **Technical Details**: 연구에서는 100,000개의 합성 스펙트로그램을 생성하여 환율 로컬리제이션(chirp localization)을 위한 최초의 대규모 벤치마크를 구축했습니다. 이 스펙트로그램들은 선형 및 지수적 주파수 스윕( frequency sweep)과 가우시안 노이즈, 스무딩 기술을 사용하여 신경 환율(neural chirps)을 모방합니다. ViT 모델은 회귀(regression)에 맞게 조정되었으며, MSE 손실 및 AdamW 최적화를 통해 훈련되었고, 학습률 스케줄러와 얼리 스탑을 적용하여 과적합을 방지하였습니다.

- **Performance Highlights**: 모델의 성능은 예측된 라벨과 실제 라벨 간의 Pearson 상관관계를 통해 평가되었습니다. 결과는 환율 시작 시간에 대해 0.9841의 강한 상관관계를 보이며, 추론 시간은 137초에서 140초로 안정적인 결과를 나타냈습니다. 이러한 접근 방식은 EEG의 시간-주파수 표현(TFR)에서 환율 분석을 위한 효율적인 도구를 제공하며, 방법론적 공백을 메울 수 있을 것으로 기대됩니다.



### Modeling speech emotion with label variance and analyzing performance across speakers and unseen acoustic conditions (https://arxiv.org/abs/2503.22711)
Comments:
          11 pages, 5 figures

- **What's New**: 이번 연구는 스피치 감정 인식 모델에서 레이블의 불확실성을 다루기 위해 그레이더의 결정을 확률 밀도 함수(probability density function)로 활용하는 접근법을 제안합니다. 일반적으로 사용되는 합의 등급(consensus grades) 대신, 감정 점수의 확률 밀도 함수를 목표로 하여 이를 통해 모델의 성능을 향상시킬 수 있음을 보여주고 있습니다. 더불어, 다양한 음성 감정 모델의 성능을 평가하기 위해 다수의 테스트 세트와 성별, 화자에 따른 성능 분석도 중요함을 강조합니다.

- **Technical Details**: 연구팀은 MSP-Podcast 데이터셋을 활용하여 약 238시간의 영어 스피치 데이터를 분석했으며, 각 샘플의 감정 범주는 수동으로 할당된 발란스, 활성화 및 지배 점수를 통해 정의됩니다. 연구에서는 사전 훈련된 모델(pre-trained model)에서 감정 인식에 최적의 표현을 얻기 위해, 주요 FM(layer)에서의 계층 중요도를 분석하는 방법을 탐구하였습니다. 특히 TC-GRU 모델을 사용하여 HuBERT, WavLM, Whisper 등의 다양한 모델로부터 스피치 임베딩을 생성하고 이를 기반으로 감정 인식을 수행합니다.

- **Performance Highlights**: 모델의 성능 분석 결과, 여러 테스트 세트를 통해 비교할 때, 전통적인 1최적 가설(1-best hypothesis)이 훈련 데이터의 불균형에 의해 편향될 수 있음을 발견했습니다. 이로 인해 2-또는 3-최적 가설(2- or 3-best hypotheses)을 사용하는 것이 더 현실적인 음성 샘플의 포함 감정을 고려하는 데 유용하다는 결과가 도출되었습니다. 연구는 다양한 평가 지표와 데이터 세트를 통한 성능 평가가 모델 선택에 있어 중요함을 분명히 하고 있습니다.



### Validating Emergency Department Admission Predictions Based on Local Data Through MIMIC-IV (https://arxiv.org/abs/2503.22706)
Comments:
          36 pages, 3 figures, 6 tables

- **What's New**: 이번 연구는 그리스의 병원에서 개발된 소규모 데이터셋을 기반으로 한 입원 예측 모델을 MIMIC-IV 데이터셋을 활용하여 유효성을 검증하였습니다. 이 연구는 긴급 서비스 관리에서 환자 결과를 향상시키기 위한 효과적인 접근법을 제시합니다. MIMIC-IV 데이터는 데이터셋의 포괄성이 높아 모델 검증에 있어 중요한 기준이 됩니다.

- **Technical Details**: 연구에서는 데이터 전처리(Preprocessing) 이후, Linear Discriminant Analysis (LDA), K-Nearest Neighbors (KNN), Random Forest (RF), Recursive Partitioning and Regression Trees (RPART), Support Vector Machines (SVM Radial) 등 총 5개의 알고리즘을 평가하였습니다. 이 중 Random Forest (RF)가 0.9999의 AUC-ROC와 0.9997의 민감도(Sensitivity), 0.9999의 특이도(Specificity)를 기록하며 최고의 성능을 보였습니다.

- **Performance Highlights**: RF 알고리즘은 복잡한 데이터셋을 다루는 데 있어 강력한 성능을 발휘하였으며, 이를 통해 ED(응급의료) 관리 전략을 개선할 수 있는 실질적인 통찰을 제공합니다. 또한, MIMIC-IV는 소규모 로컬 데이터셋을 기반으로 한 모델 검증에 있어 귀중한 기준점으로 자리잡고 있습니다.



### Enhancing nonnative speech perception and production through an AI-powered application (https://arxiv.org/abs/2503.22705)
- **What's New**: 이 연구는 인공지능(AI)을 활용한 모바일 애플리케이션이 외국어 발음에 미치는 영향을 조사합니다. 기존 연구들이 이해 가능성(comprehensibility)과 명료성(intelligibility)에 초점을 맞춘 반면, 개인 발음 소리 개선은 소홀히 했음을 지적합니다. 이렇듯 개인 발음 소리에 대한 연구의 공백을 메우고자 AI 기반 훈련이 비원어민의 발음 지각과 생산에 미치는 효과를 다룹니다.

- **Technical Details**: 참여자들은 'heed-hid' 대비를 구별할 수 있는 능력을 평가하기 위해 사전 테스트(pretest)를 완료했습니다. 이후 Speakometer 모바일 애플리케이션으로 훈련을 진행하였으며, 이 앱은 영어 모음 녹음 작업과 발음 피드백(pronunciation feedback) 및 연습을 포함하고 있습니다. 사후 테스트(posttest)는 사전 테스트와 유사하게 수행되어 성과의 변화를 측정했습니다.

- **Performance Highlights**: 연구 결과는 훈련 후 비원어민들이 발음을 구별하는 정확도와 목표 대비의 생산에서 유의미한 개선을 보였음을 보여줍니다. 그러나 참여자들은 원어민 수준의 발음 능력에는 도달하지 못했습니다. 이 발견은 AI 기반 애플리케이션이 발음 습득을 촉진하는 데 효과적이며, 교실을 넘어 개인화된 대화형 발음 훈련에 활용될 가능성을 지원합니다.



### From Eye to Mind: brain2text Decoding Reveals the Neural Mechanisms of Visual Semantic Processing (https://arxiv.org/abs/2503.22697)
Comments:
          27 pages, 7 figures

- **What's New**: 이 연구에서는 감각적 경험을 의미 있는 의미 표현으로 변환하는 신경 메커니즘을 이해하기 위해 새로운 접근 방식을 제안합니다. 전통적인 뇌 디코딩 방식과는 달리, fMRI 신호를 자연 이미지의 텍스트 설명으로 직접 디코딩합니다. 이 모델은 시각적 입력 없이 훈련되었으며, 복잡한 장면의 핵심 의미 내용을Capture하는 유의미한 캡션을 생성합니다.

- **Technical Details**: 새로운 심층 학습 모델은 최첨단 의미 디코딩 성능을 달성했으며, 더 높은 수준의 시각 영역인 MT+(Middle Temporal area), 배측 경로 시각 피질, 아래쪽 두정 피질이 의미 변환에서 중요한 역할을 한다는 것을 밝혔습니다. 이 연구는 카테고리별 디코딩을 통해 생명체와 움직임과 같은 의미적 차원의 미세한 신경 표현을 설명합니다. 이러한 텍스트 기반 디코딩 접근법은 뇌의 의미 인코딩에 대한 더 직접적이고 해석 가능한 윈도우를 제공합니다.

- **Performance Highlights**: 저희 연구는 복잡한 의미 처리의 신경 기초를 탐구하는 강력한 새로운 방법론을 제공합니다. 또한, 분산 의미 네트워크에 대한 이해를 정교화하고, 뇌 기반 언어 모델의 개발에 영감을 줄 수 있습니다. 이 성과는 기존의 시각 재구성이 아닌 보다 심층적인 의미 분석을 가능하게 하여 인지 신경 과학 분야에 기여할 것으로 기대됩니다.



### Bridging Language Models and Financial Analysis (https://arxiv.org/abs/2503.22693)
Comments:
          28 pages

- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 자연어 처리에서 혁신적인 가능성을 열었으며, 특히 금융 부문에서 큰 변화를 이끌고 있습니다. 전통적인 방법이 다루기 어려운 복잡한 텍스트, 수치표, 시각적 차트를 포함한 금융 데이터를 효과적으로 처리하고 분석하는 새로운 경로를 제공하고 있습니다. 그러나 LLM의 신기술이 금융 산업에서 실질적으로 사용되기까지는 여전히 큰 갭이 존재하며, 이에 대한 종합적인 검토가 필요합니다.

- **Technical Details**: LLM은 대규모 텍스트 데이터를 처리하고, 긴 문맥에서도 미세한 맥락을 이해하며, 복잡한 추론 작업을 수행하는 데 뛰어난 능력을 보이고 있습니다. 특히 금융 분야에서 LLM의 활용은 요구되는 전문 지식과 데이터 분석의 복잡성 덕분에 더욱 주목받고 있습니다. 이 설문 조사는 데이터 세트와 모델 두 가지 측면에 중점을 두어 LLM의 현재 활용 현황을 검토하고, 향후 금융 분야에서의 적용 가능성을 탐구합니다.

- **Performance Highlights**: LLM을 활용한 금융 데이터의 분석과 처리는 주요 작업으로는 텍스트 분류, 정보 추출, 텍스트 요약 및 질문 응답을 포함합니다. 이러한 작업들은 각각 금융 텍스트를 분류하고, 구조화된 정보를 추출하며, 긴 문서를 압축하고, 복잡한 질문에 응답하는 능력을 강화해 주기 때문에 중요한 의미를 갖습니다. 이 논문은 LLM이 금융 분야에서 갖춘 잠재력을 강조하며, 향후 연구 방향과 LLM의 혁신적인 응용 가능성을 제시합니다.



### Enhancing Aviation Communication Transcription: Fine-Tuning Distil-Whisper with LoRA (https://arxiv.org/abs/2503.22692)
Comments:
          14 pages, 4 Figures, 4 Tables, Under review by Journal of Aerospace Information Systems

- **What's New**: 본 논문은 항공 통신의 문서화(Transcription)에 있어 최신 인공지능 기술을 적용하여 정확성을 향상시키기 위한 연구입니다. 특히 OpenAI의 Whisper 모델을 항공 통신에 맞게 미세 조정(Fine-tuning)하는 방법을 다룹니다. 이를 통해 효율적으로 Whisper의 한 버전인 distil-Whisper를 미세 조정하는 Parameter-Efficient Fine-tuning 방법인 Low-Rank Adaptation을 활용했습니다.

- **Technical Details**: 이 연구에서는 약 70시간 분량의 항공 교통 통제 데이터세트(Air Traffic Control Corpus)를 사용하여 실험을 진행했습니다. 또한 LoRA(Low-Rank Adaptation)의 하이퍼파라미터를 설정하기 위해 그리드 서치(Grid Search) 및 5-겹 교차 검증(5-fold Cross-validation)을 적용했습니다. 이 과정에서 Alpha = 64 및 Rank = 32를 초기 하이퍼파라미터로 설정하고 최적의 조합을 찾아냈습니다.

- **Performance Highlights**: 미세 조정 과정 후, 모델의 평균 단어 오류율(Word Error Rate)은 3.86%로 측정되어 매우 우수한 성능을 보였습니다. 이 결과는 항공기의 조종실(Cockpit)에서의 적용 가능성을 입증해 주며, 논문이 제시하는 방법론이 향후 항공 통신의 효율성을 높일 수 있음을 시사합니다.



### Qieemo: Speech Is All You Need in the Emotion Recognition in Conversations (https://arxiv.org/abs/2503.22687)
- **What's New**: 이번 연구에서 제안된 Qieemo 프레임워크는 pretrained automatic speech recognition (ASR) 모델을 효과적으로 활용하여 오디오 모달리티만으로도 정밀한 감정 분류를 실현합니다. 또한, 멀티모달 퓨전(Multimodal Fusion, MMF) 모듈과 교차 모달 어텐션(Cross-Modal Attention, CMA) 모듈을 설계하여 ASR 인코더로부터 추출된 음성 포스터리어그램(Phonetic Posteriorgram, PPG)과 감정 특징을 융합합니다. 실험 결과, Qieemo는 기존의 unimodal 및 multimodal 모델보다 뛰어난 성능을 보였으며, 감정 인식에서의 최신 기술보다 높은 정확도를 달성했습니다.

- **Technical Details**: Qieemo 프레임워크는 ASR 백본을 기반으로 한 MMF 및 CMA 모듈을 사용하여 입력 오디오만으로도 정확한 감정 레이블을 얻을 수 있도록 설계되었습니다. ASR 구조는 end-to-end attention 기반의 인코더-디코더(AED) 프레임워크를 활용하여 스펙트로그램에서 다차원 음성 특징을 추출합니다. 효율적인 컨포머 모델을 사용하여 PPG 특징을 추출하고, 여러 컨포머 블록으로 구성되어 있으며 이들은 멀티헤드 셀프-어텐션 및 컨볼루션 모듈 등으로 구성되어 있습니다.

- **Performance Highlights**: 실험에서는 IEMOCAP 데이터셋을 활용하여 Qieemo의 성능을 검증하였으며, unimodal 감정 인식 접근 방식이 multimodal 및 self-supervised 모델이 달성한 최신 정확도를 초과함을 입증하였습니다. 다양한 pretrained ASR 백본에 대한 실험을 통해 Qieemo 프레임워크의 보편성이 증명되었으며, 이를 통해 기존 ASR 시스템에 감정 인식 방식을 통합할 수 있는 가능성을 확인했습니다.



### Binary and Multi-Class Intrusion Detection in IoT Using Standalone and Hybrid Machine and Deep Learning Models (https://arxiv.org/abs/2503.22684)
Comments:
          Master's thesis, 80 pages, 18 figures, 4 tables

- **What's New**: 이번 연구는 IoT 시스템의 사이버 공격에 대한 민감성을 고려하여 침입 탐지의 중요성을 강조하고 있습니다. IoT23 데이터 세트를 기반으로 여러 머신 러닝(Machine Learning, ML) 및 딥 러닝(Deep Learning, DL) 기법과 하이브리드 모델을 사용하여 이중 및 다중 분류 침입 탐지를 탐구했습니다.

- **Technical Details**: 연구에서는 Random Forest (RF), Extreme Gradient Boosting (XGBoost), 인공신경망(Artificial Neural Network, ANN), K-최근접 이웃(K-Nearest Neighbors, KNN), 서포트 벡터 머신(Support Vector Machine, SVM) 및 컨볼루션 신경망(Convolutional Neural Network, CNN)과 같은 단일 모델을 사용했습니다. 하이브리드 모델은 RF, XGBoost, AdaBoost, KNN, SVM을 결합하여 생성되었으며, 이 두 개의 하이브리드 모델은 이중 및 다중 분류를 위한 투표 기반 하이브리드 분류기를 구성합니다.

- **Performance Highlights**: 모델의 성능은 정밀도(precision), 재현율(recall), 정확도(accuracy) 및 F1 점수(F1-score) 기준으로 평가되었으며, 각 모델의 성능을 비교했습니다. 이 연구는 하이브리드, 단독 ML 및 DL 기법이 IoT에서 침입 탐지 시스템(IDS)의 정확성과 확장성을 개선할 수 있는 방법을 상세히 설명하고 있습니다.



### SPDZCoder: Combining Expert Knowledge with LLMs for Generating Privacy-Computing Cod (https://arxiv.org/abs/2501.00363)
- **What's New**: 이 논문은 최근 증가하는 개인정보 보호 컴퓨팅에 대한 관심과, 이에 필요한 코드 작성의 어려움을 다루고 있습니다. 다양한 프로그래밍 언어에 의존하는 개발자들에게는 MPC(다자간 계산)와 함께하는 데이터 불가시성 요구사항이 도전 과제가 되고 있습니다. SPDCoder라는 새로운 프레임워크를 제안하여, 이 기술 문제를 해결하고자 하며, 추가적인 학습 데이터 없이 LLM(대형 언어 모델)을 활용할 수 있습니다.

- **Technical Details**: SPDZCoder는 전문가 지식을 수집하여 Python과 MP-SPDZ 간의 의미 표현 차이를 이해하고, 이를 바탕으로 변환 규칙을 만들어냅니다. 이 프레임워크는 세 가지 단계의 파이프라인으로 구성되어 있으며, 각각 리팩토링, 생성, 수리 단계로 나눠져 진행됩니다. 특히 리팩토링 단계에서 데이터 불가시성을 준수하는 형태로 코드를 변환하는 방법을 제시합니다.

- **Performance Highlights**: SPDZCoder는 313개의 예제와 함께하는 벤치마크 데이터셋 SPDZEval을 만들어 성능을 평가하였습니다. 실험 결과, SPDZCoder는 pass@1에서 85.94%, pass@2에서 92.01%의 정확성을 기록하며 이전 모델들을 크게 초월하는 성과를 보였습니다. 기존의 최선 성능 기준 모델과 비교해 20% 이상의 성능 격차를 달성함으로써 새로운 기술적 성과로 자리매김합니다.



### ActionStudio: A Lightweight Framework for Data and Training of Large Action Models (https://arxiv.org/abs/2503.22673)
Comments:
          15 pages; large action models; xLAM

- **What's New**: 이번 논문에서는 Action models의 중요성을 강조하며, 이를 통한 자율 에이전트의 복잡한 작업 수행 가능성을 설명합니다. ActionStudio라는 새로운 데이터 및 훈련 프레임워크를 소개하였으며, 이는 대규모 Action models을 위한 경량화되고 확장 가능한 솔루션입니다. 기존 인프라의 한계를 극복하고 다양한 훈련 패러다임을 지원하는 기능을 갖추고 있습니다.

- **Technical Details**: ActionStudio는 다양한 에이전트의 궤적을 표준화된 포맷으로 통합하고, LoRA, 전체 파인튜닝(full fine-tuning), 분산(distributed) 환경을 포함한 다양한 훈련 패러다임을 지원합니다. 또한, 강력한 전처리(preprocessing) 및 검증(verification) 도구를 통합하여 데이터 관리의 효율성을 높입니다. 이를 통해 에이전트 특정(fine-tuning) 훈련을 보다 쉽게 수행할 수 있습니다.

- **Performance Highlights**: 우리는 공용 및 실제 산업 벤치마크에서 ActionStudio의 효과를 검증하였으며, 강력한 성능과 실용적인 확장성(practical scalability)을 입증하였습니다. 연구 커뮤니티를 지원하기 위해 코드 및 데이터는 오픈 소스로 제공하고 있습니다.



### The Mathematical Relationship Between Layer Normalization and Dynamic Activation Functions (https://arxiv.org/abs/2503.21708)
Comments:
          New title, renamed DyISRU, added missing parentheses in proof of theorem 3, minor language corrections

- **What's New**: 최근 연구에서 제안된 Dynamic Tanh (DyT)는 layer normalization (LN)을 대체할 수 있는 방법입니다. 이 접근법은 실제적으로 유용하지만, 이론적인 근거가 부족하였습니다. 본 논문에서는 LN과 동적 활성화 함수 간의 수학적 관계를 규명하고, DyT를 LN에서 유도하는 방법을 제시하고 있어 이론적 이해를 심화하고 있습니다.

- **Technical Details**: DyT 함수는 LN의 특정 수학적 유도를 통해 개발되며, 이 과정에는 미분 방정식을 해결하는 단계가 포함됩니다. 연구자들은 LN의 입력에 대한 미분을 계산하고, 이를 단순화하여 DyT 함수를 도출하였습니다. 이러한 과정에서 정밀한 근사가 필요하다는 것을 발견하였으며, 이를 제거함으로써 Dynamic Inverse Square Root Unit (DyISRU)라는 대체 기능을 제안했습니다.

- **Performance Highlights**: DyISRU는 layer normalization의 정확한 대응 개념으로, 수치적으로 DyT보다 LN에 더 정확히 유사하다는 것을 증명했습니다. 이 연구는 변동성(variance) 가정에서 벗어나 새로운 요소별 변환을 제공함으로써 layer normalization을 대체할 가능성을 제시하고 있습니다. DyT는 사전 조정이 필요할 수 있지만, DyISRU는 훨씬 더 안정적인 성능을 기대할 수 있는 장점이 있습니다.



New uploads on arXiv(cs.LG)

### Effectively Controlling Reasoning Models through Thinking Intervention (https://arxiv.org/abs/2503.24370)
- **What's New**: 이번 논문에서는 Reasoning-enhanced 대규모 언어 모델(LLM)들이 최종 답변을 생성하기 전에 중간 사고 단계를 명확히 생성함으로써 복잡한 문제 해결에서 우수한 성능을 보인다는 점을 강조합니다. 저자들은 Thinking Intervention이라는 새로운 패러다임을 제안하여 모델의 내부 사고 과정을 명확하게 안내하고, 이로 인해 모델 행동을 조정할 수 있는 새로운 기회를 제공합니다.

- **Technical Details**: Thinking Intervention은 모델이 전통적인 프롬프트 엔지니어링을 넘어서서 사고 과정 중 특정 토큰 시퀀스를 삽입하거나 수정하여 더 세밀하게 제어할 수 있도록 합니다. 이 방식은 모델 훈련이 필요하지 않으며, 실제 환경에서 최소한의 엔지니어링 노력으로 배치할 수 있습니다. 또한 기존의 모델 제어 기법들과 호환되며, 올해 문맥 및 작업에 따라 적응적으로 사고 단계를 삽입하거나 수정할 수 있는 유연성을 제공합니다.

- **Performance Highlights**: Thinking Intervention은 다양한 작업에서 성능을 크게 향상시킵니다. IFEval, SEP, XSTest 및 SORRY-Bench에서의 평가 결과, 이 접근법은 지침 따르기 작업에서 최대 6.7%의 정확도 향상과, 지침 계층 문제 해결에서 15.4% 개선, 그리고 안전 프롬프트에 대한 거부율을 40.0%까지 증가시켰습니다. 전반적으로, 저자들은 이 접근법이 LLM의 추론 프로세스에 대한 더 정밀하고 투명한 제어를 가능하게 한다고 주장합니다.



### Which LIME should I trust? Concepts, Challenges, and Solutions (https://arxiv.org/abs/2503.24365)
Comments:
          Accepted at the 3rd World Conference on eXplainable Artificial Intelligence (XAI 2025)

- **What's New**: 이 논문은 LIME(지역 해석 가능한 모델 비특정 설명)의 기초 개념과 알려진 한계를 종합적으로 탐구하고 정리한 최초의 조사입니다. LIME에 관한 다양한 적응 및 향상을 체계적으로 분류하고 주요 문제에 따른 분류법을 제공함으로써, 미래 연구의 방향을 제시하고 실무자들이 적합한 접근 방식을 식별하는 데 도움을 주고자 합니다. 특별히, 사용자 친화적인 웹사이트를 통해 LIME 관련 기술을 지속적으로 모니터링하고 정보를 업데이트합니다.

- **Technical Details**: LIME은 블랙박스 모델의 행동을 특정 인스턴스 주위에서 근사화하여 설명을 생성하는 모델 비특정 접근 방식입니다. 이 기술은 복잡한 모델에 대한 지역적 설명을 제공하지만, 안정성, 계산 비효율성, 특정 데이터 처리의 한계와 같은 여러 도전에 직면해 있습니다. 본 논문은 LIME 프레임워크 내부의 기술적 수정과 특정 문제를 해결하는 솔루션에 따라 분류된 새로운 분류 체계를 도입하여 LIME의 다양한 확장 및 변형을 분석합니다.

- **Performance Highlights**: LIME과 그 확장에 대한 체계적인 평가는 연구 관점에서 반드시 필요합니다. 연구자들은 각 모델의 특성과 실무 요구 사항에 따라 적절한 LIME 기술을 선택하는 데 어려움을 겪고 있습니다. 본 논문은 이러한 기술을 효율적으로 설명할 수 있도록 지원하며, 기존 방법의 장점과 한계를 조망하고 미래 연구를 위한 유망한 방향성을 제공합니다.



### SQuat: Subspace-orthogonal KV Cache Quantization (https://arxiv.org/abs/2503.24358)
- **What's New**: 이번 논문에서는 SQuat(Subspace-orthogonal KV cache quantization)이라는 새로운 접근법을 소개합니다. SQuat는 기존의 KV 캐시 양자화 방법과는 달리, 쿼리 텐서들로 구성된 서브스페이스를 활용하여 과거 토큰의 키 텐서를 양자화하는 과정에서 양자화 오류가 주의 메커니즘에 미치는 영향을 최소화합니다. 이 방법은 모델 재학습이나 추가적인 데이터 수집 없이도 적용될 수 있으며, 이론적 토대를 기반으로 개발되었습니다.

- **Technical Details**: SQuat은 주어진 사용자 프롬프트의 모든 토큰에서 쿼리 텐서를 통해 작업 관련 서브스페이스를 먼저 구성합니다. 그런 다음, 각 토큰의 키 텐서를 양자화하면서, 양자화된 키 텐서와 원래 키 텐서의 차이가 이 서브스페이스에 대해 직교하도록 유지합니다. 이를 통해 중요한 과업 정보에 대한 양자화 오류의 영향을 줄이고, 최적의 업데이트 규칙을 통한 효율적인 연산이 가능합니다.

- **Performance Highlights**: SQuat는 Llama-2-7B 모델을 기반으로 할 때, 피크 메모리 사용량을 2.17배에서 2.82배까지 줄일 수 있으며, 처리량은 2.45배에서 3.60배까지 향상됩니다. 또한, 이 방법은 기존의 다른 비튜닝(baseline) 방법들에 비해 더욱 우수한 성능을 발휘하며, 14개의 다양한 벤치마크 과제를 포함한 다양한 평가에서 그 효율성을 입증하였습니다.



### ORAL: Prompting Your Large-Scale LoRAs via Conditional Recurrent Diffusion (https://arxiv.org/abs/2503.24354)
- **What's New**: 본 논문에서는 기존의 방법들이 직면한 확장성과 조정 가능성의 한계를 해결하기 위해 새로운 조건부 반복 확산(conditional recurrent diffusion) 프레임워크인 ORAL을 소개합니다. ORAL은 고유한 조건화 메커니즘을 포함하여 모델 아키텍처와 작업 사양을 통합하여, 진화하는 기초 모델을 통해 효율적으로 전이 가능한 LoRA 파라미터를 생성할 수 있습니다. 이 접근법은 수십억 개의 파라미터를 가진 대형 언어 모델에서도 조정 가능성을 유지하면서 확장을 성공적으로 수행합니다.

- **Technical Details**: ORAL의 주요 기여는 LoRA 파라미터의 유연한 생성을 위한 새로운 조건화 메커니즘을 개발한 것입니다. 이 메커니즘은 모델 아키텍처 및 텍스트 기반 작업 사양을 입력으로 사용하여, 특정 다운스트림 작업에 맞춤화된 LoRA 파라미터를 생성할 수 있게 합니다. ORAL은 기존의 반복 확산 아키텍처를 기반으로 하여, 자원 집약적인 재교육 없이도 진화하는 기초 모델에 생성된 파라미터를 원활하게 전이할 수 있는 새로운 조건부 파라미터 생성 파이프라인을 제안합니다.

- **Performance Highlights**: 다양한 실험을 통해 ORAL은 7개의 언어 작업, 4개의 비전 작업, 3개의 다중 모달 작업을 수행하였으며, 5개의 사전 학습된 LLM을 사용하여 그 효율성을 입증했습니다. 연구 결과, ORAL은 7777억 개의 파라미터를 효과적으로 처리하면서도 전통적인 미세 조정 방법과 비슷하거나 우수한 성능을 보여줍니다. 이는 ORAL이 기존 방법과 비교할 때 scalability, controllability 및 portability를 모두 충족하는 새로운 기준을 정립하고 있음을 의미합니다.



### NoProp: Training Neural Networks without Back-propagation or Forward-propagation (https://arxiv.org/abs/2503.24322)
- **What's New**: 본 논문에서는 전통적인 back-propagation 방식에 의존하지 않는 새로운 학습 방법인 NoProp을 소개합니다. 이전의 접근 방식과 달리, NoProp은 각 레이어가 독립적으로 노이즈 타겟을 디노이즈하는 방식으로 학습하도록 설계되었습니다. 이는 전파 과정이 필요하지 않으며, diffusion 및 flow matching 방법에서 영감을 얻어 개발되었습니다. 이렇게하여, NoProp은 계층적 표현을 학습하지 않으면서도 효과적인 학습 방법이 될 수 있음을 보여줍니다.

- **Technical Details**: NoProp은 variational diffusion models의 아이디어를 기반으로 하여, 각 레이어가 독립적으로 목표 레이블을 예측하도록 훈련시킵니다. 훈련 시에는 각 레이어가 노이즈가 포함된 레이블을 받아들이고, 자신의 예측 방향으로 한 걸음 나아가는 방식으로 작동합니다. 이 과정에서 forward pass가 필요하지 않아 NoProp이라는 이름이 붙여졌습니다. 논문에서는 MNIST, CIFAR-10, CIFAR-100 데이터셋을 활용한 실험 결과가 제시됩니다.

- **Performance Highlights**: NoProp은 기존의 back-propagation-free 방법들에 비해 더 높은 정확도를 달성하며, 복잡성이 낮고 계산 효율성이 뛰어난 것으로 보입니다. 이에 따라, NoProp은 효과적인 분산 학습을 가능하게 하며, 전통적인 기울기 기반 학습 패러다임에서 벗어나 새로운 경로를 제시합니다. 본 연구는 NoProp이 더 넓은 범위의 문제에 적용될 수 있는 가능성을 보여주는 중요한 단계가 될 것으로 예상됩니다.



### Evaluating machine learning models for predicting pesticides toxicity to honey bees (https://arxiv.org/abs/2503.24305)
- **What's New**: 이번 연구는 ApisTox라는 데이터세트를 중심으로, 벌 관련 독성 데이터를 수집 및 평가하였습니다. ApisTox는 꿀벌(Apis mellifera)에 대한 실험적으로 검증된 화학 독성 정보를 제공하는 가장 포괄적인 데이터세트로, 기존의 의료 데이터세트와는 다른 화학 공간을 표현하고 있습니다. 이 연구는 기계 학습(Machine Learning) 기술로 독성 예측의 한계를 조사하며, 현재 가장 발전된 알고리즘이 생물 의학 데이터에 대해서만 훈련되었음을 보여줍니다.

- **Technical Details**: 연구는 ApisTox 데이터세트를 1,035개의 화합물로 구성된 데이터셋으로 분석하였으며, 이는 ECOTOX, PPDB, BPDB 데이터베이스로부터 파생되었습니다. 데이터세트는 296개의 독성 화합물과 739개의 비독성 화합물로 구성되어 있으며, 특정 화학 구조에 대한 다양한 기계 학습 접근 방식을 통해 독성을 예측하고 있습니다. 이 데이터세트는 훈련-시험 분할을 마련하여 다른 알고리즘을 공정하게 비교할 수 있도록 설계되었으며, 이는 기존의 의료 화학 데이터세트와 차별화됩니다.

- **Performance Highlights**: ApisTox에 대한 기계 학습 알고리즘의 성능 저하가 나타나며, 현재의 최첨단 알고리즘이 농약 독성 및 환경 데이터를 일반화하는 데 한계가 있음을 보여줍니다. 연구는 다양한 기계 학습 접근 방식, 예를 들어 분자 지문(molecular fingerprints) 및 그래프 뉴럴 네트워크를 통해 독성 예측 모델의 잠재력을 평가하고, 이러한 방법이 환경적으로 중요한 곤충에 적합한 예측 결과를 도출하는 데 어려움을 겪고 있음을 강조합니다. 이러한 연구는 안전한 농약 개발 및 꿀벌과 같은 필수 꽃가루 매개체를 보호하는 데 중요한 기여를 할 것으로 기대됩니다.



### Open-Reasoner-Zero: An Open Source Approach to Scaling Up Reinforcement Learning on the Base Mod (https://arxiv.org/abs/2503.24290)
- **What's New**: Open-Reasoner-Zero(ORZ)는 대규모 추론 지향 강화 학습(RL) 훈련의 첫 번째 오픈 소스 구현체입니다. 이 프로젝트는 단순성과 확장성에 중점을 두고 있으며, 기존의 DeepSeek-R1-Zero를 넘어서는 성능을 목표로 하고 있습니다. 연구자들이 접근할 수 있는 다양한 훈련 자원과 데이터셋을 공유하며, 오픈 소스 커뮤니티의 민주화에 중점을 두고 있습니다.

- **Technical Details**: ORZ는 기본 모델(Qwen-32B)에서 대규모 RL 훈련을 직접 수행하는 전략을 적용합니다. Proximal Policy Optimization(PPO) 알고리즘을 활용하며, GAE(Generalized Advantage Estimation)를 사용하여 간단한 규칙 기반 보상 함수와 결합하여 사용합니다. 훈련 데이터는 수천 개의 수학 및 추론 문제로 구성되어 있으며, 모델이 복잡한 문제를 해결할 수 있도록 설계되었습니다.

- **Performance Highlights**: ORZ는 AIME2024, MATH500, GPQA Diamond 벤치마크에서 기존 모델보다 월등한 성능을 보이며, DeepSeek-R1-Zero보다 훈련 단계 수가 10분의 1에 불과합니다. 모델 성능은 훈련 데이터의 양이 증가할수록 지속적으로 개선되며 포화 상태에 도달하지 않음을 보여줍니다. 우리의 접근 방식은 단순한 RL 알고리즘 설계를 통해 효율적인 훈련 과정을 스케일업하는 것에 중점을 두고 있습니다.



### Value of Information-based Deceptive Path Planning Under Adversarial Interventions (https://arxiv.org/abs/2503.24284)
Comments:
          10 pages, 4 figures

- **What's New**: 이번 논문에서는 적대적인 개입(adversarial interventions) 하에서 기만적 경로 계획(deceptive path planning, DPP) 문제를 해결하기 위한 새로운 MDP(Markov Decision Process) 기반 모델을 제안합니다. 기존의 연구들은 대부분 수동적 관찰자(passive observer)를 가정하고 있어 실질적인 문제 해결에 한계가 있었습니다. 저자들은 정보의 가치(value of information, VoI) 기반 목표를 통해 DPP 정책을 설계하는 새로운 방법론을 도입하고, 이를 통해 적대적 개입에 대응할 수 있는 기법을 개발했습니다.

- **Technical Details**: 저자들은 적대적 개입이 가능한 상황에서 DPP를 수행하기 위해 새로운 MDP 모델을 구축하며, 이를 위해 두 가지 새로운 VoI 기반 기만 측정 지표를 정의합니다. 이 모델은 관찰자의 정보 가치를 고려해 기만적인 정책의 영향을 평가할 수 있도록 설계되었습니다. 또한, 선형 계획(linear programming, LP) 이론을 활용하여 이러한 VoI DPP 문제를 효과적으로 해결할 수 있는 계산 방법을 도출하였습니다.

- **Performance Highlights**: 실험 결과, 제안한 솔루션 방법이 적대적인 환경에서 기만적인 경로를 달성하는 데 효과적임을 보여주었습니다. 기존 DPP 방법 및 보수적인 경로 계획(conservative path planning, CPP) 방법과 비교했을 때, 저자들의 VoI DPP 접근법이 더욱 우수한 성능을 발휘하는 것을 확인했습니다. 특히, 실제 관찰자 개입 상황에서 저비용 경로를 달성할 수 있는 능력이 강조되었습니다.



### Evaluating and Designing Sparse Autoencoders by Approximating Quasi-Orthogonality (https://arxiv.org/abs/2503.24277)
- **What's New**: 이번 연구에서는 Sparse Autoencoders(SAEs)의 새로운 접근 방식을 제안합니다. 기존의 hyperparameter 조정 없이도 입력 임베딩과 피처 벡터 간의 실제 관계를 기반으로 한 새로운 이론적 근사치를 개발했습니다. 이를 통해 Approximate Feature Activation(AFA)와 함께 시각화 도구인 ZF Plot을 도입했습니다.

- **Technical Details**: SAEs는 dense embeddings을 해석 가능한 feature vectors의 선형 조합으로 분해할 수 있음이 입증되었습니다. 연구진은 *linear representation hypothesis (LRH)*와 *superposition hypothesis (SH)*의 정의에 근거하여 AFA의 개념을 도입하고, 연결된 새로운 평가 메트릭을 제안했습니다. top-AFA SAE 구조는 이러한 아이디어를 기반으로 하며 객관적인 이론적 근거를 제공합니다.

- **Performance Highlights**: 새로운 top-AFA SAE 구조는 최신 기술과 비교하여 유사한 reconstruction loss를 달성했습니다. 이 방식은 hyperparameter tuning을 요구하지 않으면서도, 기존의 top-k SAEs 이상으로 뛰어난 성능을 보입니다. 연구진의 접근법은 SAEs의 메커니즘 해석 가능성을 심화시킬 수 있는 실험적 연구의 새로운 방향을 열었습니다.



### New Statistical Framework for Extreme Error Probability in High-Stakes Domains for Reliable Machine Learning (https://arxiv.org/abs/2503.24262)
- **What's New**: 이번 연구에서는 Extreme Value Theory (EVT)를 기반으로 한 새로운 통계적 프레임워크를 제안하여 기존의 평균 기반 검증 방법의 한계를 극복하고 최악의 예측 실패를 수치적으로 평가할 수 있는 방법을 제공합니다. 이 프레임워크는 Monte Carlo 교차 검증 기법과 통합되어 머신 러닝 모델의 최대 오류를 예측하는 데 사용할 수 있습니다. 이를 통해 극단적 오류를 정량화하는 새로운 접근 방식이 현실 세계의 데이터에 적용됨을 보여줍니다.

- **Technical Details**: Extreme Value Theory (EVT)는 극단적인 이벤트를 모델링하고 분석하기 위해 고안된 통계적 프레임워크입니다. EVT는 주로 데이터의 중심 경향을 설명하는 전통적인 통계 방법과 달리, 극단적 편차의 확률과 크기를 정량화하는 도구를 제공합니다. 이 연구에서는 GEV(Generalized Extreme Value) 및 GPD(Generalized Pareto Distribution)와 같은 분포를 통해 극단적 오류를 평가하기 위한 알고리즘을 제시합니다.

- **Performance Highlights**: 실험 결과, 제안된 EVT 기반 접근 방식은 회귀 작업에서 극단적 오류를 정량화하고 예측하는 데 효과적임이 입증되었습니다. 합성 데이터와 실제 데이터 세트를 사용하여 이 방법론의 유효성을 평가하였으며, EVT가 가지고 있는 이론적 기초와 실제 구현 방법에 대해 논의하였습니다. 이 연구는 모델 신뢰성을 평가하는 데 있어 EVT의 중요성을 강조하며, 불확실성 정량화가 핵심인 신기술의 안전한 배포를 보장합니다.



### Advances in Continual Graph Learning for Anti-Money Laundering Systems: A Comprehensive Review (https://arxiv.org/abs/2503.24259)
- **What's New**: 최근 논문은 자금세탁 방지를 위한 지속적 그래프 학습(continual graph learning)의 필요성을 강조하고 있습니다. 자금세탁 방지(AML) 프로세스에서 기존 기계 학습 모델은 새로운 데이터를 학습하면서 이전 정보를 잃게 되는 문제가 발생하며, 이로 인해 효과성이 저하됩니다. 지속적 학습 방식은 이러한 문제를 해결하고, 모델이 새로운 정보를 통합하면서도 이전 지식을 유지할 수 있도록 도와줍니다.

- **Technical Details**: 이 논문에서는 그래프 신경망(graph neural networks, GNNs) 프레임워크를 사용하여 지속적 그래프 학습 방법들을 범주화하고, 반복 기반(replay-based), 정규화 기반(regularization-based), 구조 기반(architecture-based) 전략으로 나누어 평가합니다. 실험은 합성 데이터와 실제 AML 데이터 세트에서 수행되었으며, 다양한 하이퍼파라미터의 영향을 분석했습니다. 이러한 방법들은 기계 학습의 비효율성을 개선하고, 다양한 종류의 거래를 감지하기 위한 지능형 프레임워크를 제시합니다.

- **Performance Highlights**: 지속적 학습은 극단적인 클래스 불균형 및 진화하는 사기 패턴에 직면했을 때 모델의 적응성과 강인성을 향상시킴을 나타냈습니다. 실험 결과, 연구팀은 GNN과 지속적 학습 방법의 하이퍼파라미터가 성능과 망각에 미치는 영향을 명확히 분석해야 할 필요성을 강조했습니다. 최종적으로 이 연구는 AML을 위한 지속적 그래프 학습의 현재 상태를 심층적으로 논의하고, 향후 연구 방향을 제시하였습니다.



### Spatio-temporal Prediction of Fine-Grained Origin-Destination Matrices with Applications in Ridesharing (https://arxiv.org/abs/2503.24237)
- **What's New**: 이 논문은 ridesharing 플랫폼에서의 네트워크 기반 여행자 요청의 정확한 시공간 예측에 대한 필요성을 강조합니다. 또한, 기존 연구에서 상대적으로 미혹된 지역 간 Origin-Destination (OD) 수요 예측 문제를 다룹니다. 새로운 예측 모델인 OD-CED를 소개하여 데이터 희소성(data sparsity) 문제를 완화하고, 의미론적(semantic) 및 지리적(geographic) 의존성을 포착하는 방법을 제시합니다.

- **Technical Details**: OD-CED 모델은 두 단계로 구성되어 있으며, 첫 번째는 N개의 미세 세포를 M개의 조잡한(super) 세포로 변환하여 계산 요구 사항을 줄이는 전처리(preprocess) 단계입니다. 두 번째는 여러 헤드(self-attention) 네트워크를 활용해 의미론적 및 지리적 의존성을 학습하는 학습(learning) 단계입니다. 이를 통해 OD 수요 예측 시 발생할 수 있는 데이터의 불균형과 희소성을 해결하고자 합니다.

- **Performance Highlights**: OD-CED 모델은 기존의 통계적 방법보다 최대 45%의 루트 평균 제곱 오차(root mean square error)를 감소시키고, 60%의 가중 평균 절대 백분율 오차(weighted mean absolute percentage error)를 개선했습니다. 이러한 성과는 OD 매트릭스의 희소성이 90%를 초과하는 경우에도 효과적으로 적용될 수 있습니다.



### Many-to-Many Matching via Sparsity Controlled Optimal Transpor (https://arxiv.org/abs/2503.24204)
- **What's New**: 이 논문은 여러 점들의 집합을 서로 매칭하는 새로운 방법을 제안합니다. 기존 Optimal Transport (OT) 방법들은 many-to-many (다대다) 매칭을 수행하지 못하거나, 만족스러운 결과를 위해 정규화 매개변수를 세심하게 조정해야 했습니다. 제안하는 방법은 매칭 예산 제약을 명시적으로 인코딩하고, 일대일 매칭으로의 악화를 방지하는 것을 목표로 합니다.

- **Technical Details**: 논문에서는 매칭 예산 제약을 설정하여 운송 계획의 각 행과 열에 대한 매칭 제한을 부여합니다. 또한, 변형된 $q$-entropy 정규화를 통해 각 점이 매칭 예산을 최대한 충족하도록 유도합니다. 이 방법은 페널티 알고리즘을 활용하여 최적화되고, 이론적으로 수렴이 보장됩니다.

- **Performance Highlights**: 다양한 실험을 통해 제안된 SCOTM 방법이 기존의 최신 방법들보다 우수한 성능을 보임을 알아냈습니다. 특히, SCOTM은 기존 OT 변형들보다 운송 계획의 희소성을 더 유연하게 조절할 수 있는 것을 증명합니다. 실험 결과는 학생 과정 할당, 단백질 상호작용 네트워크 매칭 및 물체 인식 등 여러 실제 응용 분야에서도 좋은 성능을 보였습니다.



### NeuRaLaTeX: A machine learning library written in pure LaTeX (https://arxiv.org/abs/2503.24187)
- **What's New**: 이번 논문에서는 NeuRaLaTeX를 소개합니다. NeuRaLaTeX는 완전히 LaTeX로 작성된 첫 번째 딥러닝 라이브러리로, 네트워크 아키텍처, 손실 함수, 트레이닝 데이터 로딩 및 생성 방식, 하이퍼파라미터를 문서 내에서 명시할 수 있습니다. 문서가 컴파일되면 LaTeX 컴파일러가 모든 트레이닝 작업을 자동으로 수행합니다.

- **Technical Details**: NeuRaLaTeX는 100 포인트의 랜덤 스파이럴 데이터셋을 생성하여 두 층의 MLP(다층 퍼셉트론)를 학습합니다. 이 논문은 또한 라벨이 부여된 다른 랜덤 스파이럴 데이터셋에서 평가하며, 결과에 대한 그래프와 표를 생성합니다. 논문 컴파일에는 48 시간이 소요되며, NeuRaLaTeX의 전체 소스 코드는 논문의 소스 코드에 포함되어 있습니다.

- **Performance Highlights**: 우리는 새로운 두 가지 메트릭을 제안합니다: WIL(LaTeX로 작성된 비율) 메트릭과 SCOMISCOP(논문 소스 내 구현 비율) 메트릭입니다. 이 두 메트릭에서 우리가 최첨단 성능을 달성하며, ResNet 및 Transformer 논문, PyTorch, TensorFlow 라이브러리보다 성능이 우수합니다. NeuRaLaTeX의 상용화에 대한 투자 초대 및 자세한 자료는 제공된 URL에서 확인할 수 있습니다.



### Ride-Sourcing Vehicle Rebalancing with Service Accessibility Guarantees via Constrained Mean-Field Reinforcement Learning (https://arxiv.org/abs/2503.24183)
Comments:
          30 pages, 12 figures

- **What's New**: 이번 논문은 Uber, Lyft, Didi Chuxing과 같은 라이드 소싱 서비스의 급속한 확산이 도시 운송을 어떻게 변화시켰는지를 다룹니다. 저자들은 차량의 재배치(vehicle rebalancing) 문제를 해결하기 위해 스케일이 가능한 연속 상태 평균 필드 제어(continuous-state mean-field control, MFC) 및 강화 학습(reinforcement learning, MFRL) 모델을 도입합니다.

- **Technical Details**: 제안된 모델은 각 차량의 정확한 위치를 명시적으로 표현하며, 다른 차량의 분포에 의해 안내되는 연속적인 재배치 행동을 적용합니다. 최적 제어 수식 안에 형평성(accessibility constraint) 제약을 통합하여, 운영 효율성과 지역별 서비스 접근의 형평성을 조화롭게 결합하였습니다. 또한 이 모델은 차량-라이더 매칭, 재배치, 순항(cruising) 과정을 동시에 고려하여 현실적인 조건을 반영합니다.

- **Performance Highlights**: 심층적인 실제 데이터 기반 시뮬레이션을 사용하여 심천(Shenzhen)에서 광범위한 평가를 수행한 결과, 제안된 접근 방식은 수만 대의 차량 규모에서도 실시간 효율성과 견고성을 보여주었습니다. 이러한 성과는 라이드 소싱 서비스 운영의 복잡한 문제를 해결하는 데 기여할 것으로 기대됩니다.



### Predicting Targeted Therapy Resistance in Non-Small Cell Lung Cancer Using Multimodal Machine Learning (https://arxiv.org/abs/2503.24165)
- **What's New**: 이 연구에서는 비소세포 폐암 환자들을 위한 osimertinib 저항성 예측을 위해 해석 가능한 다중 모달 머신러닝 모델을 개발하였습니다. 현재 osimertinib의 저항성을 정확하게 예측할 수 있는 표준 도구가 없다는 점을 해결하기 위해, 보험 의료기록을 비롯한 다양한 데이터를 통합하여 정밀한 폐암 관리와 치료 결정을 지원합니다.

- **Technical Details**: 모델은 히스토로지 이미지(histology images), 차세대 시퀀싱(next generation sequencing, NGS) 데이터, 인구 통계학적 정보 및 임상 기록을 포함한 여러 유형의 데이터를 통합합니다. 이 연구에서 개발된 다중 모달 모델은 다기관 데이터셋(multi-institutional dataset)에서 c-index 0.82를 기록하며, 이는 단일 모달 모델의 성능(c-index 0.75 및 0.77)보다 우수한 결과입니다.

- **Performance Highlights**: 연구 결과, 다중 모달 머신러닝 모델은 치료 저항성을 예측하는 데 있어 단일 모달 모델보다 더 뛰어난 성능을 보였습니다. 이는 여러 데이터 모달리티를 결합함으로써 환자 결과 예측에 있어 보다 정밀한 조치를 가능하게 한다는 점에서 중요합니다.



### LLM4FS: Leveraging Large Language Models for Feature Selection and How to Improve I (https://arxiv.org/abs/2503.24157)
- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)을 기반으로 한 새로운 하이브리드 기능 선택 전략인 LLM4FS를 제안합니다. LLM4FS는 LLM의 의미적 추론 능력과 전통적인 데이터 기반 방법의 신뢰성을 결합하여 기능 선택 성능을 크게 향상시킵니다. 실험 결과는 LLM4FS가 기존의 LLM 및 전통적 데이터 기반 방법보다 뛰어난 성능을 발휘한다는 것을 보여줍니다.

- **Technical Details**: 기능 선택은 최적화 및 인공지능에서 중요한 단계로, 유용한 기능을 선택하여 성능과 계산 효율성을 높이는 것을 목표로 합니다. 기존의 전통적인 방법들은 데이터 필터링, 래퍼, 내장 세 가지로 나눌 수 있으며, 각각의 방법은 기능을 평가하고 선택하는 데 다른 접근 방식을 사용합니다. 본 논문에서는 LLM4FS가 각각의 약점과 강점을 보완하며 새로운 가능성을 제시한다고 설명합니다.

- **Performance Highlights**: LLM4FS를 사용한 비교 실험 결과, 제공된 네 가지 데이터셋(Bank, Credit-G, Pima Indians Diabetes, Give Me Some Credit)에서 성능 향상이 관찰되었습니다. 특히, LLM 기반 방법들이 데이터가 적을 때 전통적인 방법들에 비해 낮은 성능을 보였으나, LLM4FS는 상호작용을 통해 신뢰성을 높이는 데 성공했습니다. 이러한 성과는 향후 다양한 분야에서 기능 선택을 위한 LLM 활용을 촉진할 것으로 기대됩니다.



### Learning a Canonical Basis of Human Preferences from Binary Ratings (https://arxiv.org/abs/2503.24150)
Comments:
          25 pages, 11 figures

- **What's New**: 최근 생성 AI의 발전은 인간 피드백 강화 학습(reinforcement learning from human feedback, RLHF)과 같은 정렬 기법(alignment techniques) 덕분에 이루어졌습니다. 본 논문에서는 RLHF와 관련 기법이 어떻게 인간의 선호도를 이해하고 이러한 선호도를 바탕으로 생성모델을 조정하는지에 대한 내용을 다룹니다. 연구 결과, 약 5,000개의 선호도 중 21개의 핵심 선호 카테고리가 개인 간의 선호도 변화를 89% 이상 포착하고 있다는 것을 발견하였습니다.

- **Technical Details**: 이 연구에서 사용된 방법론은 기존의 이진 선택지 데이터셋을 기반으로 하여 인간의 암묵적인 선호 카테고리를 발견하는 데 초점을 맞췄습니다. 우리는 Chatbot Arena 데이터셋을 사용하고, 데이터셋에서 이진 선택의 이유를 도출하기 위해 GPT-4o를 활용하여 선호와 주제를 추출합니다. 이 결과, 4,469개의 고유한 선호와 3,012개의 고유한 주제가 도출되었으며, 이후 클러스터링(clustering)을 통해 21개의 선호와 주제로 최종 필터링하였습니다.

- **Performance Highlights**: 우리가 발견한 21개의 선호 카테고리는 89% 이상의 개인 간 선호도 변화를 설명할 수 있으며, 이는 인간의 심리학이나 얼굴 인식 연구에서의 핵심적인 발견과 유사합니다. 또한, 발견된 선호 집합은 모델 평가 및 훈련에 유용하게 활용되어, 특정 주제 또는 개인 사용자의 요구에 따라 더 나은 모델 정렬을 이끌어냅니다. 이 연구는 선호 데이터를 통해 생성 모델을 효과적으로 조정하는 방법을 제시하며, 관련 데이터와 코드는 GitHub에서 공개되었습니다.



### Reinforcement Learning for Safe Autonomous Two Device Navigation of Cerebral Vessels in Mechanical Thrombectomy (https://arxiv.org/abs/2503.24140)
- **What's New**: 본 연구는 뇌혈관에서 두 개의 장치를 사용하여 자율적으로 미세 카테터(micro-catheter)와 미세 가이드와이어(micro-guidewire)를 탐색하는 안전한 강화 학습(reinforcement learning, RL) 알고리즘을 제안합니다. 이를 통해 경동맥(carotid arteries)을 넘어 뇌혈관에 접근할 수 있도록 하여 환자 안전을 고려합니다. 연구진은 12개 환자 맞춤 혈관 케이스의 데이터를 사용하여 최초로 이를 실험하여 성공률 96%, 7.0초의 시술 시간, 0.24N의 평균 힘을 달성하였습니다.

- **Technical Details**: 연구는 Simulation Open Framework Architecture(SOFA)를 사용하여 뇌혈관의 복잡성을 시뮬레이션합니다. 제안된 알고리즘은 Soft Actor-Critic RL 알고리즘을 수정하여 사용할 수 있으며, 탐색 중에 가이드와이어 끝의 힘을 보상 함수에 통합하여 환자 안전 지표를 측정합니다. 수집된 데이터는 12개의 환자 특수 혈관 구조를 반영하여 RL 모델의 학습에 활용되었습니다.

- **Performance Highlights**: 우리의 시뮬레이션 결과, 제안된 자율 시스템은 미지의 뇌혈관 내에서 성공적으로 탐색을 수행하며, 96%의 성공률과 7.0초의 시술 시간을 기록했습니다. 평균 힘은 0.24N으로 제안된 혈관 파열 임계치인 1.5N을 크게 하회했습니다. 이 연구는 MT에 대한 자율 탐색이 가능하다는 점에서 중요한 이정표가 될 것입니다.



### CTSketch: Compositional Tensor Sketching for Scalable Neurosymbolic Learning (https://arxiv.org/abs/2503.24123)
Comments:
          15 pages, 6 figures

- **What's New**: 최근 연구에서는 신경망(Neural Networks)과 기호(Symbolic) 프로그램의 조합으로 공식화된 많은 계산 작업이 이점을 보임을 보여주고 있습니다. 본 논문에서는 CTSketch라는 새로운 신경상징적(neurosymbolic) 학습 알고리즘을 소개합니다. 이 알고리즘은 신경망을 훈련하는 데 있어 엔드-투-엔드 입력-출력 라벨만을 사용하는 것을 목표로 합니다.

- **Technical Details**: CTSketch는 신경상징적 추론의 확장성을 개선하기 위해 두 가지 기술을 활용합니다. 첫째, 기호 프로그램을 하위 프로그램으로 분해하고, 둘째, 각 하위 프로그램을 스케치된 텐서(Sketched Tensor)로 요약합니다. 이러한 전략을 통해 입력 분포와 요약에 대해 간단한 텐서 연산을 통해 프로그램의 출력 분포를 근사할 수 있습니다.

- **Performance Highlights**: 논문에서는 CTSketch를 신경상징적 문헌에서 여러 벤치마크를 통해 평가하였으며, 특히 확장성을 평가하기 위해 설계된 벤치마크에서 뛰어난 성능을 보였습니다. CTSketch는 이제까지 도달할 수 없었던 새로운 스케일의 학습을 가능하게 하여, 천 개 이상의 입력을 포함하는 작업에서 높은 정확성을 달성했습니다.



### Level the Level: Balancing Game Levels for Asymmetric Player Archetypes With Reinforcement Learning (https://arxiv.org/abs/2503.24099)
Comments:
          Accepted at the ACM International Conference on the Foundations of Digital Games (FDG) 2025

- **What's New**: 이번 연구는 비대칭 멀티플레이어 게임의 밸런스를 조정하기 위한 새로운 접근법을 제시합니다. 이전의 연구들이 주로 수동 테스트에 의존했던 반면, 이 연구는 레벨 디자인을 통해 플레이어의 능력 차이를 조정하려 합니다. Procedural Content Generation (PCG)과 Reinforcement Learning (RL)을 결합하여 다양한 플레이어 아키타입 간의 공정한 경쟁을 생성하는 방법을 탐구합니다.

- **Technical Details**: 이 연구에서는 PCG를 활용한 레벨 디자인을 사용하여 게임의 밸런스를 자동으로 조정합니다. 연구진은 레벨이 균형을 이루기 위해 필요한 훈련 단계 수가 플레이어 아키타입 간의 불균형 정도에 따라 증가하고, 밸런스를 이루는 데 필요한 정확도는 감소한다는 점을 발견했습니다. 또한, 액션 스페이스의 크기를 절반으로 줄여 훈련 속도를 높였습니다.

- **Performance Highlights**: 실험 결과, 네 가지 플레이어 아키타입을 기반으로 기존의 두 가지 방법보다 더 많은 비율의 레벨에서 균형을 이루는 가능성을 입증했습니다. 랜덤 서치 및 힐클라이밍 기법과 비교 평가한 결과, 제안한 방법이 더욱 효율적임을 나타냈습니다. 이러한 성과는 레벨 디자인을 통한 비대칭 밸런스 조정의 가능성을 보여줍니다.



### TransMamba: Flexibly Switching between Transformer and Mamba (https://arxiv.org/abs/2503.24067)
Comments:
          Preprint. Under review

- **What's New**: 최근 연구에서 Mamba라는 상태 공간 모델(State Space Model, SSM)이 소개되었으며, 이는 긴 시퀀스를 처리하는 데 있어 효율성을 보여주지만, 안정적인 맥락 학습과 멀티태스크 일반화에서 한계를 보이고 있습니다. 본 논문은 Transformer와 Mamba의 장점을 결합한 새로운 프레임워크인 TransMamba를 제안합니다. TransMamba는 공유 매개변수 행렬을 통해 서로 다른 모델 구조 간의 동적 전환을 가능하게 하여 교육 효율과 성능을 극대화합니다.

- **Technical Details**: TransMamba는 Transformer와 Mamba의 기계적 일관성을 활용하여, 긴 시퀀스에서 효율성을 극대화하고, 짧은 맥락에서는 Transformer 메커니즘을 적용하는 형태로 설계되었습니다. 이를 위해 Memory Converter를 통해 attention의 출력 결과를 SSM 호환 상태로 변환하며, TransPoint에서 이러한 변환이 원활하게 이루어지도록 보장합니다. 추가적으로, 다양한 레이어와 시퀀스 길이에 대한 최적의 TransPoint 일정을 탐구했습니다.

- **Performance Highlights**: 광범위한 실험 결과, TransMamba는 기존의 기준 모델들보다 뛰어난 훈련 효율성과 성능을 달성했습니다. 이 연구는 Transformer와 Mamba 간의 깊은 일관성을 검증하며, 다음 세대 시퀀스 모델링을 위한 확장 가능한 솔루션을 제공합니다. TransMamba는 효과성과 효율성을 모두 고려하여 hybrid 모델의 새로운 가능성을 열어줄 것으로 기대됩니다.



### Accelerated Airfoil Design Using Neural Network Approaches (https://arxiv.org/abs/2503.24052)
- **What's New**: 이 논문에서는 타겟 압력 분포(수축면 및 압력면)로부터의 에어포일(airfoil) 형태 예측과 그 반대를 수행하는 방법을 Convolutional Neural Networks (CNNs) 및 Deep Neural Networks (DNNs) 기술을 사용하여 소개합니다. 1600개의 에어포일 형태에 대한 데이터셋이 생성되었으며, Reynolds numbers (Re)이 10,000에서 90,000,000 사이에서, 공격각(Angle of Attack, AoA)이 0도에서 15도까지 변화하는 시뮬레이션이 수행되었습니다. 다양한 공기역학적 조건을 포착할 수 있도록 데이터셋을 엄격히 설계했습니다.

- **Technical Details**: 다섯 가지 다른 CNN 및 DNN 모델이 입력/출력 매개변수에 따라 개발되었습니다. 결과는 정제된 모델들이 효율성을 개선했음을 보여주며, DNN 모델은 복잡한 데이터셋에서 CNN 모델에 비해 훈련 시간을 다중 폴드로 줄이는 성과를 보였습니다. 예측된 에어포일 형태 및 압력 분포는 타겟 값과 밀접하게 일치하여 딥러닝 프레임워크의 효과성을 검증했습니다. 그러나 CNN 모델의 성능은 DNN 모델보다 뛰어난 것으로 나타났습니다.

- **Performance Highlights**: 최종적으로 10m 이상의 날개를 가진 비행기 모델이 코드 방향을 따라 압력 분포 예측을 위해 고려되었습니다. 제안된 CNN 및 DNN 모델은 유망한 결과를 보여주며, 이 연구는 딥러닝 모델이 공기역학 최적화를 가속화하고 고성능 에어포일 디자인을 발전시키는 잠재력을 강조합니다.



### Frequency-Aware Attention-LSTM for PM$_{2.5}$ Time Series Forecasting (https://arxiv.org/abs/2503.24043)
- **What's New**: 이 논문에서는 PM$_{2.5}$ 농도 예측의 정확성과 강인성을 향상시키기 위해 FALNet(주파수 인식 LSTM 네트워크)을 소개합니다. FALNet은 주파수 영역 분해(frequency-domain decomposition), 시간 모델링(temporal modeling), 및 주의 기반 정제(attention-based refinement)를 통합하여 구성됩니다. 이 모델은 STL 및 FFT를 적용하여 데이터의 경향(trend), 계절(seasonal), 그리고 노이즈가 제거된 잔차(residual) 구성 요소를 효과적으로 추출합니다.

- **Technical Details**: FALNet은 질소 농도 예측을 위해 주파수 분해와 강화된 주의 메커니즘을 통합한 하이브리드 모델입니다. 모델은 먼저 원시 오염물 데이터에서 구조적 특징을 추출하기 위해 시간-주파수 분해(time-frequency decomposition)를 적용합니다. 이후 이러한 특징을 LSTM 네트워크로 처리하고, 다중 주의 메커니즘(multi-head attention)으로 중요한 시점에 동적으로 집중합니다.

- **Performance Highlights**: FALNet은 실제 도시 대기질 데이터셋에서 실시된 실험을 통해 기존의 모델들에 비해 MAE, RMSE, 및 R²와 같은 표준 지표에서 일관되게 우수한 성능을 보였습니다. 이 모델은 오염 피크 및 비정상적인 조건에서의 급격한 변동을 포착하는 데 강력한 적응성을 보여줍니다. 이러한 결과는 FALNet이 실시간 공기 오염 예측, 환경 위험 평가, 및 의사 결정 지원을 위한 효과적이고 일반화 가능한 모델임을 검증합니다.



### Bayesian Predictive Coding (https://arxiv.org/abs/2503.24016)
- **What's New**: 본 논문에서는 예측 코딩(Predictive Coding, PC)의 확장을 제안하여 신경망 매개변수에 대한 사후 분포를 추정하는 방법인 Bayesian Predictive coding (BPC)을 소개합니다. 이 접근 방식은 PC의 지역성을 유지하면서도 Hebbian 가중치 업데이트를 위한 닫힌 형태의 업데이트를 제공합니다. BPC는 PC에 비해 전체 배치 환경에서 더 적은 에포크(epoch)로 수렴하며, 미니 배치(mini-batch) 환경에서도 경쟁력을 유지합니다. 이 방법은 또한 불확실성(uncertainty) 정량화에 있어서 기존 Bayesian 딥러닝 방법들과 유사한 성능을 제공합니다.

- **Technical Details**: BPC는 L개의 변수 계층을 가진 계층적 가우시안 생성 모델을 역전송하는 알고리즘입니다. 이 방법은 매개변수와 신경 활동을 결합하여 사용하며, 닫힌 형태의 업데이트 규칙을 통해 신경망의 매개변수에 대한 추정치를 제공합니다. 기존 PC에서는 최대 사후 추정치(MAXIMUM A POSTERIORI, MAP)와 최대 가능도 추정치(MAXIMUM LIKELIHOOD, ML)를 사용하지만, BPC에서는 근사 사후 분포를 나타내어 매개변수를 업데이트합니다. 이러한 방식은 각종 최적화 문제에서 기댓값-최대화(Expectation-Maximization, EM) 알고리즘을 사용하여 해결됩니다.

- **Performance Highlights**: BPC는 실험을 통해 PC와 전통적인 역전파(Backpropagation, BP) 알고리즘과 비교하여 유사한 성능을 나타내며, 특히 전체 배치 훈련에서 놀라울 정도로 적은 에포크로 수렴한다는 점이 주목할 만합니다. BPC의 학습된 사후 분포는 에피스템적 불확실성과 우연적 불확실성을 강력하게 정량화할 수 있으며, Bayesian 딥러닝에서 인기 있는 벤치마크와 비교해도 개선된 불확실성 정량화와 정확도를 제공합니다. 종합적으로 BPC는 불확실성을 인식하는 신경망 훈련을 위한 유효한 방법으로 제안됩니다.



### Tree-Guided $L_1$-Convex Clustering (https://arxiv.org/abs/2503.24012)
- **What's New**: 본 논문에서는 Tree-Guided $L_1$-Convex Clustering (TGCC)이라는 새로운 볼록 클러스터링 알고리즘을 제안합니다. 이 알고리즘은 나무 구조(weights of the tree structure)를 활용하여 최적화를 가속화하고 일반적인 볼록 클러스터링에서 빈번하게 발생하는 클러스터 분할(Cluster Splits) 문제를 해결합니다. TGCC는 동적 프로그래밍(dynamic programming) 접근 방식을 통해 손실 함수(loss function)를 효과적으로 최적화할 수 있습니다.

- **Technical Details**: TGCC 알고리즘의 중심에는 $L_1$-볼록 클러스터링이 있으며, 이는 클러스터 간의 결합을 최적화하여 연쇄적인 클러스터 경로(clusterpath)를 생성합니다. 다양한 최적화 방법을 적용하여 복잡한 대규모 데이터셋에서도 뛰어난 성능을 발휘할 수 있습니다. 특히, 이 알고리즘은 1,000,000개의 포인트로 이루어진 $	ext{R}^2$ 공간에서도 15초 이내에 클러스터 경로를 구축할 수 있습니다.

- **Performance Highlights**: TGCC는 전통적인 클러스터링 방식들에 비해 높은 계산 효율성을 보여줍니다. 또한, 병렬 또는 분산 컴퓨팅 없이도 강력한 성능을 발휘하며, 클러스터링의 성능을 저하시키지 않으면서도 계산 효율성을 극대화합니다. 이 연구는 TGCC 이외에도 이중 클러스터링(biclustering) 및 희소 볼록 클러스터링 알고리즘 역시 발전시킵니다.



### CITRAS: Covariate-Informed Transformer for Time Series Forecasting (https://arxiv.org/abs/2503.24007)
- **What's New**: 이 논문에서는 CITRAS라는 새로운 Transformer 모델을 제안합니다. CITRAS는 시계열 예측을 위한 패치 기반 모델로, 과거와 미래를 아우르는 여러 목표 변수 및 공변량을 유연하게 활용합니다. 이 모델은 기존 Transformer의 자기 회귀(autoregressive) 특성을 유지하면서, KV Shift 및 Attention Score Smoothing이라는 두 가지 새로운 메커니즘을 도입합니다. 이를 통해 복잡한 공변량 간 의존성을 효과적으로 캡처하고 향상된 예측 정확도를 제공합니다.

- **Technical Details**: CITRAS는 공변량 정보를 담은 예측을 위해 Cross-시간 주의 모듈과 Cross-공변량 주의 모듈을 별도로 운영합니다. 주의 모듈 내에서, Attention Score Smoothing은 지역적으로 정확한 패치 간의 관계를 글로벌 공변량 수준으로 변환합니다. KV Shift 메커니즘은 미래의 알려진 공변량을 예측 과정에 통합하여, 공변량 간의 동시 의존성을 기반으로 예측을 수행합니다. 이러한 기술적 노하우는 기존 Transformer의 강력한 자기 회귀 성질을 완전히 보존합니다.

- **Performance Highlights**: CITRAS는 공변량 정보를 활용한 예측 및 다변량 예측(settings)에서 최첨단 성능을 달성했습니다. 실험 결과는 CITRAS가 복잡한 공변량 간 상관관계를 활용하여 예측 정확도를 향상시킬 수 있는 다재다능한 능력을 보여줍니다. 특히, 이 모델은 다양한 상황의 공변량을 효과적으로 활용하여 예측의 신뢰성을 높입니다. 결과적으로 CITRAS는 시계열 데이터의 다양한 변수 간의 복잡한 의존성을 포착하며, 미래의 알려진 공변량을 효과적으로 활용합니다.



### Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving (https://arxiv.org/abs/2503.24000)
Comments:
          21 pages, 18 figures, published to MLSys2025

- **What's New**: 이 논문은 대규모 언어 모델(LLM)의 성능 향상을 위해 Key-Value 캐시(KV cache) 압축 기술을 다루고 있습니다. 최근 LLM에 대한 수요가 급증함에 따라 효율적인 추론 최적화 알고리즘이 필요해졌습니다. 연구자들은 KV 캐시가 LLM 서비스의 주요 성능 병목 현상임을 지적하고 있으며, 이를 해결하기 위한 방법론을 논의하고 있습니다.

- **Technical Details**: KV 캐시 압축에는 두 가지 주요 기술이 있습니다: 첫 번째는 양자화(quantization) 기반 방법으로, 이는 KV 캐시를 저정밀 표현으로 변환하여 GPU 메모리 사용량을 줄이는 기술입니다. 두 번째는 희소성(sparsity) 기반 방법으로, 중요하지 않은 KV 캐시 항목을 메모리에서 제거하거나 저속 메모리로 이동시키는 방식을 유도합니다. 이러한 방법들이 실질적으로 배포될 수 있는지는 여전히 불확실합니다.

- **Performance Highlights**: 논문은 KV 캐시 압축 알고리즘이 추론 성능을 향상시킬 수 있지만, 특정 배치 크기와 프롬프트 길이에서 성능 저하를 보일 수 있음을 보여줍니다. 또한, KV 캐시 압축으로 인해 생성되는 응답 길이가 길어질 수 있으며, 이는 최종 대기 시간을 증가시킬 수 있습니다. 연구팀은 이러한 문제를 해결하기 위한 여러 도구를 제공하여 KV 캐시 압축의 실제 배포를 용이하게 하려 합니다.



### Federated Structured Sparse PCA for Anomaly Detection in IoT Networks (https://arxiv.org/abs/2503.23981)
- **What's New**: 최근 IoT 환경에서 개인 정보 보호를 위해 페더레이티드 학습(federated learning)이 주목받고 있습니다. 그러나 기존의 페더레이티드 PCA(principal component analysis) 방법은 강력한 이상 탐지를 위한 중요한 기능인 희소성(sparsity)을 통합하지 못한 문제가 있습니다. 이에 본 연구에서는 IoT 네트워크를 위한 새로운 페더레이티드 구조 희소 PCA(FedSSP) 접근 방식을 제안하며, 이는 이중 희소성 정규화(double sparsity regularization)를 통합합니다.

- **Technical Details**: 제안된 모델은 각각의 로컬 게이트웨이에서 수집된 데이터를 고려하여 $	ext{ℓ}_{2,p}$-norm과 $	ext{ℓ}_{q}$-norm을 이용하여 행(row) 및 원소(element) 희소성을 규명합니다. 데이터의 재구성 오류를 최소화하는 최적화 문제를 효율적으로 해결하기 위해, 우리는 PAM(proximal alternating minimization) 알고리즘을 개발하였으며, 이는 이론적 수렴 보장을 제공하는 엄밀한 증명을 가지고 있습니다. 이러한 접근은 로컬 게이트웨이의 개별성을 고려하여 이상 탐지를 보다 효과적으로 수행합니다.

- **Performance Highlights**: 실제 데이터 세트에 대한 실험 결과, 구조적 희소성을 추가함으로써 모델의 해석 가능성과 탐지 정확도가 향상됨을 입증했습니다. 기존의 페더레이티드 PCA 모델보다 더 나은 성능을 보여주며, 데이터의 희소성이 이상 탐지에 필수적임을 확인하였습니다. 따라서 이 연구는 IoT 보안 향상을 위한 새로운 방향성을 제시합니다.



### Noise-based reward-modulated learning (https://arxiv.org/abs/2503.23972)
- **What's New**: 최근 심화 학습(Reinforcement Learning, RL) 분야의 발전은 작업 성능에서 큰 개선을 가져왔습니다. 그러나, RL 환경에서 신경망을 훈련하는 것은 일반적으로 역전파(Backpropagation)와 결합되어 이루어지므로 자원 제약이 있는 환경이나 비미분 가능한 신경망에서는 적용이 제한됩니다. 본 논문에서는 지연 보상 시나리오에서의 한계를 해결하기 위해 새로운 노이즈 기반 학습 규칙을 도출하였으며, 이는 방향 미분 이론과 Hebbian 유사 업데이트를 결합하여 RL에서의 효율적이고 기울기 없는 학습을 가능하게 합니다.

- **Technical Details**: 제안된 학습 메커니즘은 스토캐스틱 노이즈 뉴런을 활용하여 기울기를 근사하는 구조입니다. 이 방법은 보상 예측 오류(Reward Prediction Error, RPE)를 최적화 목표로 삼고, 지연 보상이 있는 환경에서 과거 신경 상태와 미래 보상을 연결하는 적격성 추적(Eligibility Trace)을 통합하여 시간적 신뢰 할당을 용이하게 합니다. 노이즈 없는 전달이 불가능한 경우 여러 개의 노이즈 있는 전달을 평균하여 성능을 유지하는 방식으로 기획하였습니다.

- **Performance Highlights**: 실험 결과, 제안한 노이즈 기반 보상 조정 학습(Noise-based Reward-Modulated Learning, NRL) 방법이 RMHL보다 유의미하게 우수한 성능을 보였으며, 역전파 기반 기준선들과도 경쟁력을 유지하였습니다. 이는 저전력 및 실시간 애플리케이션을 위한 노이즈 기반 비유적 학습의 가능성을 강조합니다. 본 연구는 머신러닝, 신경형 컴퓨팅(Neuromorphic Computing), 신경과학(Neuroscience) 분야에서의 넓은 함축적 의미에 대해서도 논의하고 있습니다.



### Green MLOps to Green GenOps: An Empirical Study of Energy Consumption in Discriminative and Generative AI Operations (https://arxiv.org/abs/2503.23934)
Comments:
          Published to MDPI Information - Artificial Intelligence Section

- **What's New**: 이 연구는 실제 MLOps 파이프라인에서 판별(Discriminative) 및 생성(Generative) AI 모델의 에너지 소비를 조사합니다. 연구 결과, 판별 모델의 경우 아키텍처와 하이퍼파라미터를 최적화함으로써 성능을 유지하면서 에너지를 줄일 수 있는 방법을 제시합니다. 생성 AI에서는 대규모 언어 모델(LLMs) 사이의 에너지 소비를 비교 분석하여 모델 크기와 요청 처리 용량 간의 균형이 중요함을 강조합니다.

- **Technical Details**: 연구는 소프트웨어 기반 전력 측정을 통해 다양한 설정, 모델 및 데이터셋에서의 에너지 소비 패턴을 분석합니다. 판별 AI의 경우 훈련 및 추론 동안 서로 다른 모델 아키텍처와 하이퍼파라미터가 에너지 소비에 미치는 영향을 평가하며, 생성 AI는 다양한 요청 및 토큰 사용을 통한 에너지 소비를 집중 분석합니다. 이러한 측정 방법은 다양한 환경에서 복제할 수 있도록 설계되었습니다.

- **Performance Highlights**: 결과적으로, 판별 모델의 경우 아키텍처와 하드웨어의 최적화가 에너지 소비를 크게 줄일 수 있음을 발견했습니다. 또한, 대규모 언어 모델의 에너지 효율성은 모델 크기, 추론 복잡성 및 요청 처리 용량 간의 균형에 따라 달라진다는 사실도 확인했습니다. 이러한 발견들은 ML 운영(MLOps)에서 지속 가능성을 높이는 데 필요한 실제 지침을 제시합니다.



### DiffScale: Continuous Downscaling and Bias Correction of Subseasonal Wind Speed Forecasts using Diffusion Models (https://arxiv.org/abs/2503.23893)
Comments:
          28 pages, 18 figures, preprint under review

- **What's New**: 본 연구는 S2S(서브시즌-투-시즌) 예측의 정확성을 향상시키기 위한 새로운 모델인 DiffScale을 제안합니다. 이 모델은 편향 없는 가이드를 사용하는 확산 모델로, 바람 속도 예측의 다운스케일링을 통해 지역 및 대규모 날씨 상황을 효과적으로 반영할 수 있도록 합니다. DiffScale은 여러 그리드 해상도 및 전진 시간에 걸쳐 모델 오류를 수정하고 조정할 수 있는 유연성을 제공합니다.

- **Technical Details**: DiffScale은 조건부 확률(conditional probabilities)을 샘플링의 가이드로 활용하는 확산 모델입니다. 이 모델은 S2S 예측의 밀도를 직접 추정하여, 자동 회귀(auto-regression)나 시퀀스 예측 없이도 효율적이고 유연한 예측을 가능하게 합니다. 연구에서는 ECMWF(유럽 중기 일기예보 센터)의 조도 해상도(S2S) 바람 속도 예측을 ERA5 재분석 데이터의 고해상도로 다운스케일링하는 synthetic experiments를 설계하였습니다.

- **Performance Highlights**: DiffScale은 최대 3주까지 기존의 기준 성능을 초과하는 예측 품질 향상을 이뤘습니다. 이 모델은 다양한 그리드 해상도에 일반화할 수 있으며, 모델 재훈련 없이도 새로운 스케일링 요소에 대응할 수 있는 다재다능한 도구입니다. 연구 결과는 에너지 부문에 중요한 사회경제적 이점을 제공할 수 있는 가능성을 보여줍니다.



### An End-to-End Comprehensive Gear Fault Diagnosis Method Based on Multi-Scale Feature-Level Fusion Strategy (https://arxiv.org/abs/2503.23887)
- **What's New**: 이 논문에서는 기어의 종단 간 결함 진단(end-to-end fault diagnosis) 요구 사항을 충족하기 위해 가속도 신호(acceleration signals)를 활용한 통합 지능형 결함 진단 방법이 제안되었습니다. Gabor 기반 적응 짧은 시간 푸리에 변환(Gabor-ASTFT)과 이중 나무 복소수 웨이브렛 변환(Dual-Tree Complex Wavelet Transform, DTCWT) 알고리즘을 기반으로 한 새로운 접근 방식입니다.

- **Technical Details**: 제안된 방법은 진동 센서를 사용하여 기어박스(base)에서 수집한 원시 1차원 가속도 신호에 대해 사전 세분화(pre-segmentation) 처리 단계를 포함합니다. Gabor-ASTFT와 DTCWT를 통해 원래의 신호를 2차원 시간-주파수(time-frequency) 표현으로 변환하여 결함(feature) 특성을 초기 추출하고 약한 특성을 확보하는 과정이 포함되어 있습니다. 또한, 특성 맵에 대해 업샘플링(upsampling) 및 다운샘플링(downsampling)을 수행하는 이중 채널 구조가 설정됩니다.

- **Performance Highlights**: 제안된 방법은 다양한 데이터 세트에서 비교 실험을 통해 기어의 결함 분류(fault classification) 요구 사항을 효과적으로 충족하는 것으로 입증되었습니다. 통합된 특성을 통해 다중 스케일 분석이 가능해지며, 잔여 구조(residual structure)를 포함한 합성곱 신경망(CNN) 모델이 깊은 특성 추출을 수행하는 데 사용되어 높은 성능을 보였습니다.



### Communication-Efficient and Personalized Federated Foundation Model Fine-Tuning via Tri-Matrix Adaptation (https://arxiv.org/abs/2503.23869)
- **What's New**: 이 논문은 효율적인 통신이 가능한 연합 학습 기반 LoRA 적응법인 CE-LoRA를 제안합니다. CE-LoRA는 클라이언트 간 데이터 이질성(data heterogeneity) 문제를 해결하고 통신 비용(communication cost)을 줄이며 모델 성능을 향상시키는 방향으로 설계되었습니다. 이 방법은 개인화된 파라미터 집합을 통해 각각의 클라이언트에 최적화된 모델 조정을 가능하게 합니다.

- **Technical Details**: CE-LoRA는 새로운 LoRA 파라미터 분해 방식으로, 추가적인 완전 순위(full-rank) 매트릭스를 도입하여 통신 비용과 매트릭스 크기를 줄입니다. 이 방법은 클라이언트의 데이터 유사성(client similarity)을 고려하여 연합 모델을 보다 효율적으로 집계할 수 있는 개인화된 방법을 제시합니다. 이러한 기법들은 통신의 비효율성과 데이터 분포의 비동일성(non-IID) 문제를 해결하는 데 중점을 둡니다.

- **Performance Highlights**: CE-LoRA는 다양한 대형 언어 모델(LLM) 및 비전-언어 모델(VLM) 과제에서 실험하여 통신 비용을 대폭 줄이며 성능 개선을 입증했습니다. 기존의 연합 학습 방법들에 비해 CE-LoRA는 클라이언트 별 모델 예측 정확도를 크게 향상시켰고, 데이터 재구성 공격에 대해서도 효과적으로 저항할 수 있음을 보여주었습니다.



### An extrapolated and provably convergent algorithm for nonlinear matrix decomposition with the ReLU function (https://arxiv.org/abs/2503.23832)
Comments:
          27 pages. Codes and data available from this https URL

- **What's New**: 이번 연구에서는 ReLU 함수에 기반한 비선형 행렬 분해(Nonlinear Matrix Decomposition, NMD) 모델을 제안합니다. 이 모델은 3B-ReLU-NMD라는 새로운 대안 모델을 통해 Latent-ReLU-NMD보다 더 나은 성능을 제공하며, rank 제한을 제거합니다. 또한, 최적화 과정에서 블록 좌표 하강법(Block Coordinate Descent, BCD)을 적용하였으며, 이를 통해 더 빠른 수렴속도를 달성했습니다.

- **Technical Details**: 제안된 3B-ReLU-NMD 모델은 파라미터 Θ를 Θ=WH로 설정하여 rank 제한을 극복합니다. 본 연구에서는 BCD라는 최적화 방법을 사용하여 3B-ReLU-NMD의 수렴성을 증명하였으며, 새로운 변형인 eBCD-NMD를 통해 수렴성을 개선했습니다. eBCD-NMD는 ReLU-NMD에 비해 매우 낮은 오류를 달성하는 경향을 보입니다.

- **Performance Highlights**: eBCD-NMD는 BCD-NMD보다 상당한 속도 향상을 이루었으며, 합성 데이터 및 실제 데이터셋에서 최신 기술 대비 좋은 성능을 보였습니다. 이러한 성능은 데이터 압축과 행렬 보완 문제에서의 응용 가능성을 제시합니다. 이 연구 결과는 NMD의 이론적 기초와 실제 구현에 기여할 것으로 기대됩니다.



### Node Embeddings via Neighbor Embeddings (https://arxiv.org/abs/2503.23822)
- **What's New**: 이 논문에서는 비모수적 그래프 표현 학습을 위한 두 가지 패러다임인 그래프 레이아웃과 노드 임베딩을 소개합니다. 기존의 최첨단 알고리즘인 force-directed layouts와 random-walk 기반 대조 학습 방법이 별개로 여겨져 오던 것을, 하나의 일관된 프레임워크로 접근할 수 있음을 보여줍니다. 특히, graph t-SNE와 graph CNE라는 두 가지 새로운 방법을 제안합니다.

- **Technical Details**: graph t-SNE는 2차원 그래프 레이아웃을 위한 이웃 임베딩 방법으로, 노드를 시각화하기 위해 활용됩니다. 반면에 graph CNE는 InfoNCE 목표를 최적화하여 고차원 노드 표현을 생성하는 대조 이웃 임베딩 방법입니다. 이러한 방법들은 기존의 알고리즘과는 달리 성격이 단순하며, 지역 구조 보존(local structure preservation) 측면에서 우수한 성능을 보입니다.

- **Performance Highlights**: 논문에서 제안된 graph t-SNE와 graph CNE는 기존의 최첨단 알고리즘들과 비교하여 지역 구조 보존에서 강력하게 우수한 성능을 나타냈습니다. 이는 두 가지 방법이 복잡한 알고리즘 없이도 뛰어난 결과를 보여줄 수 있음을 의미합니다. 또한, 이러한 연구는 그래프 표현 학습의 패러다임을 통합하고, 그래픽 시각화 및 고차원 표현 생성에 기여할 수 있습니다.



### When Counterfactual Reasoning Fails: Chaos and Real-World Complexity (https://arxiv.org/abs/2503.23820)
- **What's New**: 이번 연구는 구조적 인과 모델(Structural Causal Models) 내에서 반사실적 추론(counterfactual reasoning)의 한계를 탐구한다. 특히, 실험적으로 반사실적 시퀀스 추정(counterfactual sequence estimation)을 조사하며, 예상치 못한 결과가 발생하는 경우를 강조한다. 연구진은 모델 불확실성(model uncertainty)이나 혼돈 동역학(chaotic dynamics)과 같은 현실적 가정을 바탕으로 예측과 실제 반사실적 경로 사이의 극심한 차이를 발견하였다.

- **Technical Details**: 반사실적 추론의 신뢰성을 평가하기 위해, 연구는 상태공간 모형(state-space model)을 활용하여 저차원 ODE(Ordinary Differential Equations) 기반의 진화를 포착하고, 구조적 인과 프레임워크를 통해 가상의 개입(hypothetical interventions)에 대한 추론을 진행했다. 연구진은 완벽한 지식이 없음에도 불구하고, 초기 미세한 변화(initial perturbations)나 파라미터의 오차가 상이한 경로를 초래할 수 있음을 보여주었다. 이로 인해 실제 상황에서 반사실적 예측의 신뢰성이 크게 저하된다는 사실이 강조되었다.

- **Performance Highlights**: 이 연구는 기존의 연구들과 달리, 매개변수 불확실성(parameter uncertainty)이 있는 혼돈 시스템에서 반사실적 추론을 수행하는 방법을 제시한다. 특히, 원본 시스템 관찰에서 반사실적 예측의 신뢰성을 바로 도출할 수 없음을 강조하며, 현장 적용을 위한 신뢰할 수 있는 반사실적 추론의 필요성을 역설한다. 이로써, 실제 시스템의 혼돈 또는 복잡한 동역학을 모델링할 때 발생하는 문제를 경고하며, 반사실적 추론의 적용에서 주의가 필요함을 일깨워준다.



### Conformal uncertainty quantification to evaluate predictive fairness of foundation AI model for skin lesion classes across patient demographics (https://arxiv.org/abs/2503.23819)
- **What's New**: 이번 연구는 피부 병변 분류에서 비전 트랜스포머(ViT) 기반의 기초 모델을 사용하여 예측의 신뢰성과 공정성을 최적화하는 것입니다. 특히, 의사의 의사결정 프로세스를 이해하고 개선하기 위해 conformal analysis를 적용하여 예측 불확실성을 정량화합니다. 이 방법은 공정성 보장을 위해 인구 집단 차이를 고려한 적응형 F1 점수 기반 샘플링을 도입합니다.

- **Technical Details**: 연구에서 제안된 방법론은 피부 병변 분류와 불확실성 예측을 위한 최신 기초 모델과 conformal prediction 기반 불확실성 정량화를 결합하는 것입니다. 클래스 불균형 문제를 해결하기 위해 F1-점수 기반의 모델 무관 동적 샘플링 알고리즘을 적용하였으며, 이는 다양한 인구 집단 간의 예측 성능을 균형 있게 유지하게 합니다.

- **Performance Highlights**: 이 연구의 성과는 피부 암 진단에 있어 신뢰할 수 있는 예측 결과를 제공하는 것입니다. 각 환자에게 대해 불확실성 점수를 제공함으로써 모델의 공정성과 신뢰성을 높이는 것을 목표로 하였습니다. 이러한 접근법은 다양한 의료 데이터셋에서도 적용 가능하며, 의사결정의 투명성을 높일 수 있는 잠재력을 지니고 있습니다.



### An extension of linear self-attention for in-context learning (https://arxiv.org/abs/2503.23814)
- **What's New**: 이번 논문에서는 선형 자기 주의 (linear self-attention)를 확장하여 새로운 바이어스 매트릭스(bias matrix)를 도입했습니다. 이 확장은 단순함에도 불구하고, 확장된 선형 자기 주의 (extended linear self-attention)는 입력 매트릭스(input matrix) 및 두 개 이상의 매트릭스의 곱을 출력할 수 있는 기능을 갖추게 합니다. 이러한 유연한 매트릭스 조작 덕분에, 논문에서는 배치형 경량 회귀의 경량 경량 알고리즘(batch-type gradient descent algorithm for ridge regression) 구현을 통해 이 기술의 효용성을 보여줍니다.

- **Technical Details**: 확장된 선형 자기 주의(ELSA)는 기본적으로 입력 매트릭스에 바이어스 매트릭스를 추가하여, 기존의 선형 자기 주의보다 유연한 매트릭스 조작을 가능하게 합니다. 이 논문에서는 m x n 매트릭스의 정의와 더불어, identity matrix로서의 \( \bf I_m \)와 zero matrix로서의 \( \bf O_{m,n} \) 등을 사용하여 이론적 배경을 제공합니다. ELSA는 기계 학습에서 경량 회귀 및 최소 제곱 문제와 같은 다양한 문제를 해결할 수 있도록 설계되었습니다.

- **Performance Highlights**: 확장된 선형 자기 주의는 실제로 경량 회귀 문제에 배치형 경량 하강 알고리즘(batch-type gradient descent algorithm)으로 적용할 수 있는 가능성을 보여줍니다. 실험 결과, ELSA는 적절한 입력 형태를 통해 좋은 성능을 발휘하며, 즉각적인 반응성과 예측 정확도를 달성하게 됩니다. 이 논문은 ELSA의 활용 가능성과 함께 변화하는 매트릭스 형식 및 처리 방식이 기계 학습에서 또 다른 발전을 가져올 수 있음을 시사합니다.



### Accelerating High-Efficiency Organic Photovoltaic Discovery via Pretrained Graph Neural Networks and Generative Reinforcement Learning (https://arxiv.org/abs/2503.23766)
Comments:
          AI for Accelerated Materials Design - ICLR 2025

- **What's New**: 본 연구에서는 유기 태양전지(Organic Photovoltaics, OPV) 개발에 있어 그래프 신경망(Graph Neural Networks, GNN)과 GPT-2(Generative Pretrained Transformer 2)를 기반으로 한 강화 학습(Reinforcement Learning, RL) 전략을 결합하여 고효율의 OPV 분자를 디자인하는 새로운 프레임워크를 제안합니다. 이를 통해 예측된 효율이 21%에 가까운 후보 물질을 생성했으며, 대규모 OPV 데이터셋을 구축하여 연구 커뮤니티에 기여할 예정입니다. 이러한 접근법은 높은 성능의 모델을 구축하는 데 중요한 기초를 제공합니다.

- **Technical Details**: 본 연구의 개념적 기초는 대규모 GNN 사전 훈련과 RL의 통합으로, 이를 통해 51,000개의 유기 소분자를 사용하여 효과적인 분자 임베딩을 구축합니다. 사전 훈련 과정에서는 분자 마스킹 및 재구성과 HOMO/LUMO 에너지 예측 두 가지 주요 작업을 수행하여 모델이 OPV 성능에 중요한 전자적 특성을 포착하도록 도와줍니다. 또한, 강화 학습 루프에 GPT-2 기반 생성기를 도입하여, 생성된 분자의 구조가 높은 PCE 값을 가지도록 유도하며, 다단계 최적화를 통해 성능을 향상시킵니다.

- **Performance Highlights**: 실험적으로 수집된 약 2,500개의 기증자-수용자(D-A) 쌍의 데이터 세트를 통해 모델의 정확성을 평가했습니다. 사전 훈련된 모델은 비 사전 훈련 모델과 기존의 분자 지문(molecular fingerprint) 및 랜덤 포레스트(random forest) 모델과 비교했을 때, PCE 예측의 평균 제곱 오차(Mean Squared Error, MSE)를 뚜렷하게 낮추는 결과를 보였습니다. 모델의 최적화 과정에서 생성된 분자들이 확인된 바와 같이, 실험적으로 검증된 D-A 구조에 대한 디자인 가능성을 보여 주며, OPV 연구에 있어 혁신적인 진전을 이룰 수 있음을 확인했습니다.



### Time-Series Forecasting via Topological Information Supervised Framework with Efficient Topological Feature Learning (https://arxiv.org/abs/2503.23757)
- **What's New**: 이번 연구에서는 Topological Data Analysis (TDA)를 시간 시계열 예측에 통합하기 위한 몇 가지 주요 문제를 분석하고 해결하기 위한 새로운 프레임워크인 Topological Information Supervised (TIS) Prediction을 제안합니다. TDA는 본래 복잡한 데이터 구조에서 의미 있는 특징을 추출하는 데 뛰어난 도구로 알려져 있습니다. 특히, TIS 모델은 Conditional Generative Adversarial Networks (CGANs)와 신경망을 활용하여 기존의 TDA 접근 방식에서 발생하는 computational bottleneck 문제를 해결하고, 멀티 레벨 temporal dependencies를 효과적으로 포착합니다.

- **Technical Details**: TIS Prediction 프레임워크는 데이터에 내재된 temporal dependencies를 사용하여 다양한 딥러닝 모델들의 예측 정확도를 향상시킵니다. 이 연구에서는 short-term 및 long-term 예측에 최적화된 두 가지 모델, TIS-BiGRU와 TIS-Informer를 제안하고, 각각 bidirectional gated recurrent unit과 attention 메커니즘을 통해 시간을 효과적으로 관리합니다. 또한, topological consistency loss를 통합한 새로운 훈련 전략을 통해 모델의 예측 능력을 향상시키고 있습니다.

- **Performance Highlights**: 연구 결과, TIS 모델은 기존의 예측 모델들에 비해 상당히 우수한 성능을 보임을 입증하였습니다. 실험적으로 다양한 metric을 통해 TIS 모델이 topological information을 통합함으로써 개선된 예측 정확도를 달성하는 것을 보여주었습니다. 특히, 다양한 simplicial complexes가 모델 성능에 미치는 영향을 분석하면서, GAN 프레임워크가 topological features의 분포를 잘 포착하는데 중요한 역할을 한다는 사실도 확인하였습니다.



### PDSL: Privacy-Preserved Decentralized Stochastic Learning with Heterogeneous Data Distribution (https://arxiv.org/abs/2503.23726)
- **What's New**: 이번 연구에서는 PDSL(Privacy-Preserved Decentralized Stochastic Learning) 알고리즘을 제안하여 이질적인 데이터 배포와 개인 정보 유출 문제를 해결하고자 합니다. 본 알고리즘은 Shapley 값을 활용하여 각 에이전트가 이질적인 이웃의 기여도를 정확히 측정할 수 있게 해줍니다. 또한, differential privacy 메커니즘을 통해 에이전트가 그래디언트 정보를 제공할 때 개인 정보 유출을 방지합니다.

- **Technical Details**: PDSL 알고리즘의 각 라운드는 에이전트가 자신의 로컬 그래디언트와 크로스 그래디언트를 계산하는 것부터 시작됩니다. 이후, differential privacy 메커니즘에 따라 노이즈를 추가하여 Perturbed Gradient를 생성합니다. 마지막으로, 각 에이전트는 이 Perturbed Cross-gradient와 자신의 Perturbed Local Gradient를 기반으로 모델 업데이트를 합니다. 이때 Shapley 값을 통해 Perturbed Gradient의 기여도를 공정하게 평가하여 집계합니다.

- **Performance Highlights**: PDSL 알고리즘은 개인 정보 보호와 수렴(convergence) 측면에서 효과성을 검증하기 위한 엄밀한 이론적 분석과 다양한 실제 데이터셋을 통한 실험을 수행하였습니다. 실험 결과는 PDSL의 실용성 및 효율성을 입증하며, 기존 중앙 집중형 알고리즘보다 이질적인 데이터 환경에서 우수한 성능을 발휘함을 보여줍니다.



### Unimodal-driven Distillation in Multimodal Emotion Recognition with Dynamic Fusion (https://arxiv.org/abs/2503.23721)
- **What's New**: 본 논문에서는 멀티모달 감정 인식(MERC) 시스템에서의 문제를 해결하기 위해 SUMMER라는 새로운 프레임워크를 제안합니다. SUMMER는 Sparse Dynamic Mixture of Experts(SDMoE), Hierarchical Cross-Modal Fusion(HCMF) 및 Interactive Knowledge Distillation(IKD) 같은 주요 구성 요소를 활용하여 이질적인 모드 통합을 지원합니다. 이를 통해 기존 모델들이 직면한 모달 이질성과 학습 지침 부족 문제를 해결하고 있습니다.

- **Technical Details**: SUMMER는 세 가지 모달(텍스트, 오디오, 비디오)을 처리하는 다중 모달 감정 인식을 위한 프레임워크입니다. SDMoE는 최신 동적 토큰 선택을 통해 중요 특징을 식별하고, HCMF는 이질적인 모드 간 관계를 포착하여 더 나은 전역 맥락 이해를 도와줍니다. IKD는 미리 훈련된 단일 모달 교사 모델을 통해 다중 모달 학생 모델을 지도하여 특성 분포 격차를 줄이고, 클래스 간 관계를 포착하는 소프트 레이블을 제공합니다.

- **Performance Highlights**: 이 연구의 실험 결과는 IEMOCAP 및 MELD 데이터셋에서 SUMMER가 최신 기법들을 능가하며, 특히 소수 감정 및 의미적으로 유사한 감정 인식에서 놀라운 성능을 보여줍니다. 이 혁신적인 접근 방식은 감정 인식을 위한 동적 토큰 선택과 이질적 모드의 융합을 통해 보다 정밀하고 강력한 결과를 도출하고 있습니다.



### Steering Large Agent Populations using Mean-Field Schrodinger Bridges with Gaussian Mixture Models (https://arxiv.org/abs/2503.23705)
- **What's New**: 이 논문에서는 Mean-Field Schrödinger Bridge (MFSB) 문제를 해결하기 위해 새로운 최적 제어 정책을 제안합니다. 기존의 방법들은 공간 이산화나 신경망을 활용해 최적 솔루션을 근사하였으나, 저자는 학습 과정 없이도 클로즈드 형태의 솔루션을 근사할 수 있는 효율적인 파라미터화를 제안합니다. 이를 통해 유한한 수의 동일한 상호작용 에이전트의 행동을 제어할 수 있는 방법론을 제시합니다.

- **Technical Details**: 제안된 접근법은 초기 혼합물의 성분에서 최종 혼합물의 성분으로의 Gaussian-to-Gaussian Covariance Steering 문제를 해결하는 기본 정책의 혼합으로 구성됩니다. 여기에 반응적 제어를 가능하게 하는 반정의 성질을 활용하여 상태에 대한 확률적 하드 제약조건을 관리할 수 있으며, 수치적으로 다룰 수 있는 실행 가능한 해를 제공합니다. 이 방식은 다양한 수치 예제에 적용되어 그 유용성을 입증하였습니다.

- **Performance Highlights**: 제안된 방법은 Linear Time-Varying dynamics를 따르는 에이전트와 Gaussian Mixture Model 경계 분포에서 성능을 극대화합니다. 이를 통해 MFSB 문제를 보다 효율적으로 해결하는 동시에, 에이전트의 집합적 행동을 제어하는 데 있어 기존의 방법들에 비해 개선된 결과를 보여줍니다. 특히, 실용적인 로봇 및 다중 에이전트 경로 계획 응용 분야에서의 이용 가능성이 높습니다.



### A Low-complexity Structured Neural Network to Realize States of Dynamical Systems (https://arxiv.org/abs/2503.23697)
Comments:
          20 pages, 6 figures

- **What's New**: 본 논문은 비선형 적분 방정식(ordinary differential equations, ODEs)에서 파생된 동적 시스템의 학습을 위해 구조적 신경망(structured neural network, StNN)을 활용하는 새로운 접근 방식을 제안합니다. StNN은 데이터 기반 기법입니다. 여기서는 'Hankel operator'를 이용해 동적 시스템을 해결하는 방법을 모색하고 있으며, 이는 기존 방법들에 비해 효율성과 복잡성을 줄이는 장점이 있습니다.

- **Technical Details**: Hankel operator는 시간 지연(time-delay) 측정을 기반으로 하여 설계됩니다. 이 열 구조는 선형 방정식 시스템을 해결하는데 필요한 계산을 저복잡도의 알고리즘으로 전환하는 데 유용합니다. 논문에서는 StNN을 통해 동적 시스템의 상태를 학습하고 예측하는 방법을 제시하며, 네트워크 아키텍처 및 기존 신경망보다 우수한 점을 강조합니다.

- **Performance Highlights**: 수치 시뮬레이션을 통해 StNN이 기존 신경망 및 동적 시스템 분석 기법(SINDy, HAVOK)보다 낮은 복잡도로 동적 시스템의 상태를 보다 정확하게 예측할 수 있음을 보여줍니다. 연구 결과는 Lorenz 시스템과 같은 비선형 및 혼돈 시스템에 대한 장기적인 예측을 포함하여 StNN이 기존 대안에 비해 우수한 성능을 발휘함을 입증합니다.



### Data-Driven Forecasting of High-Dimensional Transient and Stationary Processes via Space-Time Projection (https://arxiv.org/abs/2503.23686)
- **What's New**: 본 논문에서는 Space-Time Projection (STP)이라는 데이터 기반 예측 접근법이 소개됩니다. STP는 고차원 시간 해상도 데이터를 다루며, 예측 기간을 포함한 교육 데이터에서 확장된 공간-시간 적절 정규 모드를 계산합니다. 이 방법은 예측 정확도를 향상시키며, 긴급 및 예측 구간 모두에서 활용됩니다.

- **Technical Details**: STP는 Proper Orthogonal Decomposition (POD) 이론에 기반을 두며, 차원 축소(dimensionality reduction)와 시간 지연 임베딩(time-delay embedding)을 내재적으로 포함합니다. 주어진 앙상블(ensemble)과 고정된 예측 기간에 대해 조정 가능한 유일한 매개변수는 잘라내기 순위(truncation rank)이며, 추가적인 하이퍼파라미터는 필요하지 않습니다. 힌드캐스트(hindcast) 정확도는 단기 예측 정확도의 신뢰할 수 있는 지표 역할을 합니다.

- **Performance Highlights**: STP의 효율성은 두 가지 데이터셋을 사용하여 입증됩니다: turbulent interstellar medium에서의 초신성 폭발에 대한 고유 이방성(transient, highly anisotropic) 시뮬레이션과 turbulent high-subsonic engineering flow의 실험적 속도장(velocity fields)입니다. 표준 Long Short-Term Memory (LSTM) 신경망과의 비교 연구 결과, STP는 항상 더욱 정확한 예측을 제공하며, 그 단순성과 강력한 성능 덕분에 고차원 일시적 및 혼란 프로세스 예측을 위한 해석 가능한 경쟁 기준을 제시합니다.



### Dynamic Operating System Scheduling Using Double DQN: A Reinforcement Learning Approach to Task Optimization (https://arxiv.org/abs/2503.23659)
- **What's New**: 이번 연구에서는 Double DQN (Double Deep Q Network) 기반의 새로운 운영 체제 스케줄링 알고리즘이 제안되었습니다. 이 알고리즘은 서로 다른 작업 유형과 시스템 부하에서의 성능을 실험을 통해 검증하였습니다. 기존의 전통적인 스케줄링 알고리즘과 비교했을 때, Double DQN 기반 알고리즘은 작업 우선 순위와 자원 할당 전략을 동적으로 조정할 수 있습니다.

- **Technical Details**: Double DQN 알고리즘은 작업 완료 효율성, 시스템 처리량, 응답 속도를 향상시키기 위해 설계되었습니다. 실험 결과는 가벼운 부하, 중간 부하, 심한 부하 상황 모두에서 높은 스케줄링 성능을 보여주며, 특히 I/O 집약적인 작업을 처리할 때 효과적입니다. 이 알고리즘은 시스템 상태에 따라 자원 할당을 지능적으로 조정하여 자원 낭비와 과도한 부하를 피할 수 있는 높은 최적화 능력을 나타냅니다.

- **Performance Highlights**: Double DQN 알고리즘은 작업 완료 시간과 시스템 응답 시간을 효과적으로 줄이는 데 성공했습니다. 향후 연구에서는 이 알고리즘을 더 복잡한 시스템, 특히 클라우드 컴퓨팅 및 대규모 분산 환경에서의 스케줄링 최적화에 적용하는 방안을 탐구할 예정입니다. 이 과정에서 네트워크 지연과 에너지 효율성과 같은 요소를 결합하여 알고리즘의 전반적인 성능과 적응력을 향상시키는 것을 목표로 하고 있습니다.



### A Survey of Reinforcement Learning-Based Motion Planning for Autonomous Driving: Lessons Learned from a Driving Task Perspectiv (https://arxiv.org/abs/2503.23650)
Comments:
          21 pages, 5 figures

- **What's New**: 이번 논문에서는 자율 주행(AD)에서의 모션 계획(MoP) 문제를 다루기 위해 발전된 강화 학습(RL) 기술의 적용을 종합적으로 검토하고 있습니다. RL의 설계 과정에 대한 체계적 설명이 부족한 상황에서, 이 설문은 다양한 주행 임무에 맞춘 RL 접근법의 필요성을 강조합니다. 특히, 이 논문은 각각의 주행 과제에 대한 경험과 통찰을 바탕으로 미래의 구현 방향을 제시합니다.

- **Technical Details**: 강화 학습(RL)은 환경과의 상호작용을 통해 정책을 자체적으로 생성하고, 장기적인 보상을 최대화하는 방식으로 학습합니다. MDP(마르코프 결정 과정)와 POMDP(부분적 관찰 MDP)라는 모델이 사용되며, 각 상태와 액션이 정의되고, 보상 함수를 통해 에이전트의 성능이 평가됩니다. 이 논문은 RL 방법론의 기초와 이를 AD의 모션 계획에 적용하는 방법에 대해 설명하고 있습니다.

- **Performance Highlights**: 최근 연구에서 RL 기반의 모션 계획 기술이 다양한 주행 작업에 효과적으로 적용되고 있다는 사실을 강조하고 있습니다. RL 기술이 자율 주행 분야에서 세계적인 챔피언을 초월할 정도로 발전했으며, 많은 연구가 RL의 보편적인 설계 패러다임과 이를 주행 작업에 맞추기 위한 커스터마이즈 전략을 제안하고 있습니다. 또한, 현재의 도전 과제를 논의하고 이를 해결하기 위한 탐색적 노력이 어떻게 진행되고 있는지도 다루고 있습니다.



### Simple Feedfoward Neural Networks are Almost All You Need for Time Series Forecasting (https://arxiv.org/abs/2503.23621)
- **What's New**: 이 연구에서는 간단한 피드포워드 신경망(SFNNs)이 Transformer나 GNNs와 같은 복잡한 모델보다 더 간단하고 빠르며, 때론 성능이 우수하다는 사실을 보여줍니다. SFNN은 특히 단일 변수(univariate) 시간 시계열 분석에서 뛰어난 성능을 나타내며, 다변량(multivariate) 모델도 강한 상관관계가 있는 데이터셋에서 경쟁력을 유지합니다. 이러한 결과는 SFNN이 복잡한 설계 선택을 피하면서도 성능을 높일 수 있음을 시사합니다.

- **Technical Details**: 제안된 SFNN 아키텍처는 단순한 구조를 가지고 있으며, 모델이 모든 시계열 데이터에서 사용할 수 있도록 설계되었습니다. SFNN은 서로 강한 상관관계를 가진 시리즈들에 대해 간단한 시리즈 간 매핑(series-wise mapping)을 추가하여 성능을 획기적으로 개선할 수 있습니다. 본 연구에서는 수학적으로 입력 매트릭스와 예측 목표의 형태를 정의하고, 평균 제곱 오차(MSE)를 최소화하는 모델 파라미터를 찾는 과정을 설명합니다.

- **Performance Highlights**: 실험 결과, SFNN은 기존의 최첨단 모델과 비슷하거나 더 나은 성능을 보입니다. 또한, 복잡한 모듈이나 아키텍처를 사용하지 않고도 24시간 예측 horizon뿐만 아니라 720시간 예측까지 성능을 유지할 수 있습니다. 이 연구는 SFNN이 차세대 시간 시계열 예측 방법의 비교 기준으로 활용될 수 있는 강력한 베이스라인이 됨을 강조합니다.



### Graph-Eq: Discovering Mathematical Equations using Graph Generative Models (https://arxiv.org/abs/2503.23617)
Comments:
          8 pages, 4 figures

- **What's New**: 본 논문에서는 데이터셋을 설명하는 의미 있고 정확하며 간결한 수학적 방정식을 발견하는 새로운 방법인 Graph-EQ를 제안합니다. 수학적 방정식을 그래픽 표현으로 나타내어 Graph Neural Networks (GNNs)를 활용, 이전에 보지 못한 새로운 방정식을 생성합니다. 기존의 유전자 프로그래밍 방법의 비효율성을 극복하기 위해 조건부 변분 오토인코더(Conditional Variational Autoencoder, CVAE)를 사용해 방정식 공간의 잠재 표현을 학습합니다.

- **Technical Details**: Graph-EQ는 대규모 방정식 집합에 대해 비지도 학습을 통해 방정식의 풍부한 잠재 표현을 학습합니다. 전통적인 방법과 달리, Graph-EQ는 방정식 공간을 직접 탐색하는 대신 베이esian 최적화(Bayesian optimization)를 통해 효율적으로 학습된 잠재 공간을 탐색합니다. 또한, 이 모델은 입력 방정식을 정확히 재구성하고, 학습된 잠재 표현이 새로운 유효 방정식으로 디코드될 수 있음을 보여줍니다.

- **Performance Highlights**: 20개의 알려진 기준 진실 방정식으로 구성된 데이터셋에서 Latent space 탐색을 수행하여 Graph-EQ가 대부분의 데이터셋에서 기준 진실 방정식을 성공적으로 발견함을 입증했습니다. 이 연구는 방정식 발견을 위한 그래프 생성 모델의 유용성을 뒷받침하며, Graph-EQ가 기존 방법에 비해 과적합(overfitting)의 위험을 줄이는 동시에 수학적 관계를 효과적으로 캡처할 수 있음을 보여줍니다.



### Make Autoregressive Great Again: Diffusion-Free Graph Generation with Next-Scale Prediction (https://arxiv.org/abs/2503.23612)
Comments:
          Draft #1

- **What's New**: 최근 이미지 생성 분야에서의 발전에 영감을 받아, 이 논문에서는 MAG라는 새로운 생성 프레임워크를 제안합니다. MAG는 전통적인 autoregressive 방법의 한계를 극복하며, 순서 없이 생성할 수 있는 이점이 있습니다. 모델이 고유한 노드 순서를 명시할 필요 없이 그래프의 모든 스케일을 점진적으로 생성합니다.

- **Technical Details**: MAG는 잠재 표현의 계층 구조를 활용하여 전체 그래프의 스케일을 생성합니다. 다음 스케일 예측(next-scale prediction) 접근 방식을 통해, 모델은 단일 토큰 맵에서 시작하여 점진적으로 해상도를 확장하는 과정을 학습합니다. 각 단계에서, transformer는 이전 스케일과 클래스 레이블을 기반으로 다음 스케일의 토큰 맵을 예측합니다.

- **Performance Highlights**: MAG는 일반 및 분자 그래프 데이터셋에서 기존의 최첨단 방법에게 경쟁력을 보여주었습니다. 실험 결과, 훈련 시간, 추론 속도, 데이터 효율성 면에서 우수한 성능을 보이며, 전통적인 autoregressive 방법들에 비해 훨씬 향상된 효율성을 입증했습니다.



### Autonomous Learning with High-Dimensional Computing Architecture Similar to von Neumann's (https://arxiv.org/abs/2503.23608)
Comments:
          20 pages including references, all contained in a single .tex file

- **What's New**: 이 논문에서는 고차원 벡터(예: H = 10,000)를 활용하여 인간과 동물의 학습을 모델링합니다. 이는 고전적인 (폰 노이만) 컴퓨팅 아키텍처와 유사하지만, 벡터를 사용하여 실시간으로 처리합니다. 이와 같은 모델은 데이터에서 학습하는 깊은 학습 방식과 유사하나, 생물학에 더 가까운 구조를 가지고 있습니다.

- **Technical Details**: 이 모델은 단기 작업 기억과 장기 데이터 저장소를 포함하는 인간 기억 및 학습의 심리학적 개념에 부합합니다. 벡터에 대한 연산은 확률적으로 발생하는 고차원 처리 장치(HPU)에서 수행되며, 이 과정은 조합성(슈퍼포지션)으로 이루어집니다. 논문에서는 인간과 동물의 인지를 모델링하기 위해 고차원 벡터를 이용한 아키텍처를 제안합니다.

- **Performance Highlights**: 로봇 학습의 응용을 통해 효과를 기대할 수 있으며, 언어 처리 등의 분야로도 확장될 것으로 보입니다. 향후에는 두뇌와 같은 물질 및 에너지를 소모하지 않고 계산을 수행하는 것에 초점을 두고 있습니다. 이와 같은 이론은 심리학과 생물학, 그리고 나노기술에 적합해야 하며, 대규모 실험을 통해 입증되어야 합니다.



### Partial Transportability for Domain Generalization (https://arxiv.org/abs/2503.23605)
Comments:
this http URL

- **What's New**: 이 논문은 AI에서 예측의 성능 보증을 제공하는 데 있어 새로운 접근 방식을 제안합니다. 특히, 이전에는 신뢰할 수 있는 방법이 부족했던 새로운 데이터 분포에 대한 불확실성을 고려한 연구입니다. 저자들은 causal diagrams를 사용하여 데이터 생성 메커니즘에 대한 가정을 수립하고, 이에 따른 새로운 성과를 제시합니다.

- **Technical Details**: 논문에서는 partial identification과 transportability의 이론을 기반으로 하여, 목표 분포의 functional value를 경계 지을 수 있는 방법을 제안합니다. 이는 Neural Causal Models와 같은 기존의 매개변수화 구조를 채택하여 cross-population inference에 필요한 구조적 제약을 인코딩합니다. 또한 높은 표현력과 일관성을 보장하는 절차를 수행하고, 실제로 확장 가능한 추론을 위한 gradient-based optimization scheme을 제안합니다.

- **Performance Highlights**: 저자들은 제안된 방법론의 유용성을 실험을 통해 입증했습니다. 이러한 결과는 기존 예측기의 불확실성을 줄이고 새로운 도메인에서의 일반화 오류를 개선하는 데 중요한 기여를 할 것으로 기대됩니다. 이 연구는 AI의 실용적인 응용에 중요한 기틀을 마련하고 있습니다.



### Bridging conformal prediction and scenario optimization (https://arxiv.org/abs/2503.23561)
- **What's New**: 이 논문은 conformal prediction과 scenario optimization 이 두 통계적 학습 프레임워크 간의 관계를 명확히 연결하는 새로운 접근을 제시합니다. 연구자들은 빌딩 블록인 vanilla conformal prediction을 통해 적절한 score function과 predictor map을 선택하는 방법을 rigorously 보여주어, 시나리오 프로그램과 관련된 제약 위반 확률을 회복할 수 있음을 입증하였습니다. 또한 비순응도 점수(nonconformity scores)의 순위를 일차원 시나리오 프로그램으로 처리하는 방법도 제안하여, 이와 같은 연결이 set predictor의 유효성에 대한 vanilla conformal prediction 보장을 회복하는 데 어떻게 기여할 수 있는지 설명합니다.

- **Technical Details**: conformal prediction은 주어진 상태에서의 예측을 정량적으로 평가하는 방법론을 제공합니다. 이 접근법은 비순응도 점수를 사용하여 결정의 품질을 평가하며, 이를 통해 생성된 예측 세트는 분포에 구애받지 않는 유효성 보장을 제공합니다. 반면 scenario approach는 최적화 문제의 해결에 있어 위반 인증서를 제공하여 널리 사용되는 기법으로, 이러한 두 가지 접근 방식을 결합하여 서로의 이점을 활용하는 방법이 제안됩니다.

- **Performance Highlights**: 연구 결과는 conformal prediction과 scenario optimization 간의 이론적 다리를 세우는 데 기여하며, 이는 나중에 두 분야에서 획득한 결과를 서로 전이 가능하게 할 것입니다. 이 논문에서는 문헌에서 파생된 기존 성과들에 대한 새로운 통찰을 제공하며, 모든 응용 프로그램에서 보편적으로 활용될 수 있는 매력적인 프레임워크를 제안합니다. 이러한 이론적 기초의 발전은 향후 연구 방향에 대한 실질적인 기초를 형성하는 데 중요한 역할을 할 것입니다.



### Redundant feature screening method for human activity recognition based on attention purification mechanism (https://arxiv.org/abs/2503.23537)
Comments:
          12 pages,7 figures

- **What's New**: 본 논문은 센서 기반 인간 활동 인식을 위한 MSAP라는 범용 어텐션 기능 정제 메커니즘을 제안합니다. 이를 통해 다중 스케일 네트워크에서 발생하는 피쳐 중복 문제를 해결하고, 최소한의 자원 소비로 네트워크 성능을 유지할 수 있도록 합니다. 또한, 네트워크 모듈 간 통합이 원활한 네트워크 수정 모듈을 설계하여 기존 심층 네트워크의 문제를 완화하려는 노력을 기울였습니다.

- **Technical Details**: 제안된 방법은 인터스케일 어텐션 스크리닝과 연결 방식을 통해 다중 스케일 특징에서 발생하는 피쳐 중복을 해결하는 데 중점을 두고 있습니다. 기본적으로 뚜렷한 특징 추출을 위해 1차원 합성곱을 사용하며, MSAP 구조를 통해 효과적인 다중 스케일 특징을 캡쳐할 수 있습니다. 또한, 레이어 간 노이즈 저감 네트워크 아키텍처를 통해 더욱 향상된 성능을 추구합니다.

- **Performance Highlights**: 제안된 방법은 4개의 공개 데이터세트를 기반으로 광범위한 실험을 통해 필터링된 데이터의 중복 특징을 효과적으로 감소시키고, 자원 소모가 적으면서도 뛰어난 성능을 보여주었습니다. 이 연구 결과는 착용 가능한 기술 수준에 맞춘 임베디드 배포 시스템을 구축해 실제 HAR 모델의 실용 가능성을 검증하였습니다. 이는 센서 기반 HAR 작업에서 기존 방법보다 실질적인 개선을 제시합니다.



### A Survey on Unlearnable Data (https://arxiv.org/abs/2503.23536)
Comments:
          31 pages, 3 figures

- **What's New**: 본 논문은 Unlearnable Data (ULD)라는 새로운 방어 기법을 제시합니다. ULD는 특정 데이터에서 유의미한 패턴을 학습하는 것을 방지하여 데이터 프라이버시와 보안을 보호합니다. 이 기술은 학습 데이터에 섭동(perturbation)을 추가하여 모델 성능을 저하시키고, 무단 모델이 유용한 표현을 추출하기 어렵게 만드는 것을 목표로 합니다. 기존의 연구들은 주로 적대적 공격(adversarial attacks)과 기계 비학습(machine unlearning)에 초점을 맞춰왔으나, 본 논문은 독립적인 ULD로의 포괄적인 리뷰를 제공합니다.

- **Technical Details**: 연구는 ULD의 생성 방법, 공개 벤치마크, 평가 지표, 이론적 기초 및 실제 응용 분야를 상세히 분석합니다. ULD는 훈련 데이터에 대한 섭동을 통해 모델이 유용한 표현을 학습하지 못하도록 수정하며, 이는 인식 가능성(perceptual quality)을 유지합니다. 또한, ULD는 기존의 머신 비학습과 적대적 공격과의 차별성을 명확히 하고, 각 접근방식의 강점과 한계를 비교합니다. 주요 기술적 도전 과제는 섭동의 인식 불가능성(imperceptibility)과 모델 성능 저하 사이의 균형을 맞추는 것입니다.

- **Performance Highlights**: 이 연구에서는 다양한 ULD 접근 방식의 효과성과 한계를 논의합니다. ULD 방식은 훈련된 모델이 일반화 성능이 저하되도록 설계되어 있으며, 데이터 프라이버시 보호 기술로서 유망한 잠재력을 갖추고 있습니다. 그러나 높은 계산 복잡성(computational complexity)과 모델 성능 저하 사이의 무역오프(trade-off)는 실용적인 적용에 있어 도전 과제가 되고 있습니다. 향후 연구 방향으로는 ULD의 효과성과 응용 가능성을 향상시킬 방안이 강조되고 있습니다.



### In-silico biological discovery with large perturbation models (https://arxiv.org/abs/2503.23535)
- **What's New**: 이번 논문에서는 복잡한 생물학적 시스템에서 발생하는 다양한 perturbation(experimental perturbation) 데이터를 통합하는 새로운 Deep Learning 모델인 Large Perturbation Model (LPM)을 소개합니다. LPM은 perturbation, readout 및 context를 별도의 차원으로 해석함으로써 여러 서로 다른 실험 데이터를 통합할 수 있습니다. 이 모델은 기존 방법보다 우수한 성능을 보이며, unseen experiments의 post-perturbation transcriptomes 예측, 화학적 및 유전적 perturbation 간의 분자 메커니즘을 식별하는 등의 다양한 생물학적 발견 작업에 적용할 수 있습니다.

- **Technical Details**: LPM은 perturbation(P), readout(R), context(C) 차원을 분리하여 각 차원을 조건부 변수로 나타내는 PRC-disentangled architecture를 채택합니다. 이 모델은 encoder-free 방식으로 관측값이나 공변량을 명시적으로 인코딩하지 않고, 다양한 실험적 맥락에서 얻어진 perturbation-response 법칙을 학습합니다. 이러한 접근은 실험 데이터의 대규모 통합을 용이하게 하며, 다양한 perturbation과 readout의 조합에서 발생하는 규칙을 이해하는 데 중요합니다.

- **Performance Highlights**: LPM은 실험 데이터를 통해 학습하여 post-perturbation 결과 예측에서 state-of-the-art 성능을 보입니다. 이 모델은 화학 및 유전적 perturbation에 대한 약물-표적 상호작용을 연구할 수 있는 능력을 갖추고 있으며, 유전자 간의 인과적 상호작용 네트워크를 추론하는 데도 활용됩니다. अंत में, 더 많은 데이터로 훈련할수록 LPM의 성능이 향상되어, 의약품 발견을 위한 적용 가능성을 더욱 높이고 있습니다.



### Handling Delay in Real-Time Reinforcement Learning (https://arxiv.org/abs/2503.23478)
Comments:
          Accepted at ICLR 2025. Code available at this https URL

- **What's New**: 본 연구는 실시간 강화 학습(real-time reinforcement learning, RL)에서 발생하는 지연 문제를 해결하기 위한 새로운 접근 방법을 제안합니다. 기존의 N계층 피드포워드 신경망은 레이어 지연(observational delay)로 인해 성능이 저하되는 문제를 나타냅니다. 저자들은 시간 스킵 커넥션(temporal skip connections)과 이력을 기반으로 한 관찰(history-augmented observations)을 결합하여 해결책을 제시하였습니다.

- **Technical Details**: 연구에서는 병렬 계산(parallel computation) 프레임워크를 활용하여, 각 레이어가 비동기적으로 작동하면서도 서로 다른 입력을 처리함으로써 네트워크의 처리를 가속화합니다. 그러나 이러한 병렬 처리 체계에서도 관찰 지연을 해결할 필요가 있으며, 이때 시간 스킵 커넥션을 통해 지연을 줄이는 방향으로 접근합니다. 또한, 다양한 아키텍처를 탐색하여 지연과 네트워크 표현력(expressivity) 간의 트레이드오프(trade-off)를 찾아냅니다.

- **Performance Highlights**: 실험 결과, 시간 스킵 커넥션과 이력 보강 관찰을 포함한 아키텍처가 다양한 RL 과제에서 우수한 성능을 발휘했음을 보여주었습니다. 또한, 병렬 뉴런 계산을 통해 표준 하드웨어에서 추론을 6~350% 가속화할 수 있음을 확인했습니다. 이를 바탕으로 실시간 설정에서 보다 효율적인 RL 에이전트를 위한 길을 열어주었습니다.



### Towards Trustworthy GUI Agents: A Survey (https://arxiv.org/abs/2503.23434)
Comments:
          10 pages, work in process

- **What's New**: 최근 GUI 에이전트(GUI agents)는 대규모 기초 모델(large foundation models)을 기반으로 하여 디지털 인터페이스와 상호작용할 수 있는 능력을 보유하고 있습니다. 이는 웹 자동화(web automation), 모바일 내비게이션(mobile navigation), 소프트웨어 테스트(software testing) 등 다양한 응용 프로그램 개발에 기여하고 있습니다. 그러나 이러한 에이전트의 자율성이 증가함에 따라 보안(security), 개인 정보 보호(privacy), 안전성(safety)와 관련된 주요 우려 사항이 제기되고 있습니다.

- **Technical Details**: 이 서베이는 GUI 에이전트의 신뢰성(trustworthiness)을 다섯 가지 중요한 차원에서 검토합니다: 보안 취약점(security vulnerabilities), 동적 환경에서의 신뢰성(reliability in dynamic environments), 투명성(Transparency) 및 설명 가능성(explainability), 윤리적 고려사항(ethical considerations), 평가 방법론(evaluation methodologies). 또한, 적대적 공격(adversarial attacks)에 대한 취약성, 순차적 의사 결정에서의 연쇄적 실패 모드(cascading failure modes), 현실적인 평가 기준(benchmarks)의 부족 등 주요 과제를 식별하였습니다.

- **Performance Highlights**: 이러한 문제들은 실제 환경에서의 배포(deployment)를 저해할 뿐만 아니라, 단순한 작업 성공(task success) 이상의 포괄적인 완화(strategy) 전략을 필요로 합니다. GUI 에이전트의 확산이 이루어짐에 따라, 강력한 안전 기준(safety standards)과 책임 있는 개발 관행(responsible development practices)의 수립이 필수적입니다. 이 서베이는 신뢰할 수 있는 GUI 에이전트를 발전시키기 위한 체계적인 이해(systematic understanding)와 향후 연구의 기초를 제공합니다.



### What Makes an Evaluation Useful? Common Pitfalls and Best Practices (https://arxiv.org/abs/2503.23424)
- **What's New**: 최근 몇 년 사이 인공지능(AI)의 발전이 급격히 이루어짐에 따라 AI 커뮤니티에서는 잠재적인 안전 위험에 대한 우려가 커지고 있습니다. 본 논문에서는 AI 시스템의 안전한 사용과 개발을 위한 고품질 평가의 필요성을 강조하며, 이러한 평가를 위한 모범 사례를 제공하고 있습니다. 특히, 사이버 보안 사례를 통해 모델 평가의 모범 사례를 어떻게 정의하고 적용할 수 있는지를 설명합니다.

- **Technical Details**: AI 모델의 평가 설계는 위협 모델링(threat modeling)과 평가 설계를 잇는 초기 사고 과정 단계를 논의하는 것으로 시작됩니다. 또한 유용한 평가의 특성과 파라미터를 제시하고, 특정 평가 구축에서 전체적인 평가 스위트(suite) 구축으로 넘어갈 때 고려해야 할 사항들을 다룹니다. 이를 통해 AI 시스템의 안전 평가를 위한 체계적 접근 방식을 제안하고 있습니다.

- **Performance Highlights**: 이 연구의 주요 기여 중 하나는 결정 과정(decision making processes), 위협 모델링과 평가 설계(threat modeling and evaluation design) 간의 중요한 연결 고리를 수립한 점입니다. 모범 사례에 대한 명확한 원칙을 확립하고 이를 바탕으로 안전 평가를 위한 평가 스위트 구성을 위한 가이드를 제공합니다. 이 논문은 실험적 검증이 향후 중요한 작업이 될 것임을 지적하며, AI 기술의 보다 안전하고 책임 있는 발전에 기여할 것으로 기대됩니다.



### Pareto Continual Learning: Preference-Conditioned Learning and Adaption for Dynamic Stability-Plasticity Trade-off (https://arxiv.org/abs/2503.23390)
- **What's New**: 이 논문에서는 지속적 학습(Continual Learning, CL)에서 안정성(stability)과 유연성(plasticity)의 균형을 다루는 새로운 접근 방식을 제안합니다. 기존의 경험 재생(experience replay) 방법들이 고정된 균형을 목표로 했던 반면, 제안된 Pareto Continual Learning (ParetoCL)은 이를 다중 목표 최적화(multi-objective optimization) 문제로 재정의했습니다. 이 방식으로 다양한 무역 오프(balances) 내에서 동적으로 적응할 수 있는 모델을 제공합니다.

- **Technical Details**: ParetoCL은 선호 기반 모델을 도입하여 두 개의 목표인 안정성과 유연성 간의 무역 오프를 학습합니다. 모델은 안정성을 위한 재생 버퍼와 유연성을 위한 새로운 데이터에 대한 손실을 최소화하는 방식으로 훈련됩니다. 추론(inference) 단계에서, 모델은 각 샘플에 대한 최적의 무역 오프를 선택하여 더욱 자신감 있는 예측을 할 수 있게 됩니다.

- **Performance Highlights**: 다양한 데이터셋을 대상으로 한 실험에서 ParetoCL은 기존의 최첨단 지속적 학습 방법보다 뛰어난 성능을 보였습니다. ParetoCL을 통해 입력된 다양한 선호에 의해 안정성과 유연성 간의 무역 오프가 잘 분포된 Pareto 최적 솔루션 세트를 얻을 수 있습니다. 이러한 결과는 데이터 증대(data augmentation) 및 클래스 증대(class augmentation) 관점에서도 중요한 기여를 보여줍니다.



### Solve sparse PCA problem by employing Hamiltonian system and leapfrog method (https://arxiv.org/abs/2503.23335)
Comments:
          2 tables

- **What's New**: 이 논문에서는 기존 Principal Component Analysis (PCA)의 해석 가능성 부족 문제를 해결하기 위한 새로운 sparse PCA 알고리즘을 제안합니다. 이 알고리즘은 부드러운 L1 패널티(smooth L1 penalty)를 적용하여 희소성을 부여하고, 기하학적 통합 기법을 사용한 해밀토니안(Hamiltonian) 형식으로 해결됩니다. 이를 통해 고차원 데이터를 저차원 특성 공간으로 변환하는 성능을 향상시키고자 합니다.

- **Technical Details**: 제안된 방법은 두 가지 수치적 방법을 구현하여 에너지 함수의 최소화를 달성합니다. 첫 번째는 Proximal Gradient (ISTA) 접근 방식이고, 두 번째는 leapfrog (4차 룽게-쿠타) 방법을 이용하는 방식입니다. 이러한 방법들을 통해 원본 데이터에서 희소 주성분(sparse principal components)을 추출하며, 디플레이션(deflation) 기법을 추가하여 성능을 더욱 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 sparse PCA 방법이 k-nearest neighbor 및 커널 능선 회귀(kernel ridge regression) 분류기를 사용할 때 전통적인 PCA에 비해 높은 분류 정확도를 달성하였음을 보여줍니다. 이는 얼굴 인식 데이터셋을 기반으로 하며, 제안된 방법의 우수성을 입증합니다. 향후 연구는 sparse PCA를 현대의 딥러닝 아키텍처와 통합하여 다중 모달 인식(multimodal recognition) 작업으로 확장할 계획입니다.



### SalesRLAgent: A Reinforcement Learning Approach for Real-Time Sales Conversion Prediction and Optimization (https://arxiv.org/abs/2503.23303)
- **What's New**: 이번 논문에서는 SalesRLAgent라는 새로운 프레임워크를 제안합니다. 이 시스템은 전문화된 Reinforcement Learning (RL)을 활용하여 판매 대화 전반에 걸쳐 전환 확률(conversion probability)을 예측합니다. 기존의 LLM 기반 접근법과는 달리 SalesRLAgent는 전환 예측을 순차적 결정 문제로 처리하며, 이는 판매 전략에 대한 보다 실시간의 통찰력을 제공합니다.

- **Technical Details**: SalesRLAgent는 Azure OpenAI 임베딩(3072 차원)과 Meta-learning 기능을 결합하여 자신의 지식 한계를 이해합니다. 이 시스템은 판매 대화의 복잡한 역학을 반영하기 위해 GPT-4O를 이용해 생성된 합성 데이터(synthetic data)를 통해 훈련됩니다. 대화의 모든 턴을 추적하면서 전환 확률을 지속적으로 추정합니다.

- **Performance Highlights**: 실험 결과 SalesRLAgent는 96.7%의 전환 예측 정확도를 달성하여 LLM 전용 접근법보다 34.7% 더 우수한 성능을 보였습니다. 또한, 기존의 판매 플랫폼과 통합했을 때, 실시간 가이드를 사용하는 경우 43.2%의 전환율 증가를 나타내었습니다. 이는 SalesRLAgent가 판매 대화의 내용을 생성하는 데 그치지 않고, 전략적인 판매 인텔리전스를 제공하는 데 중점을 두고 있음을 보여줍니다.



### Enhancing Physics-Informed Neural Networks with a Hybrid Parallel Kolmogorov-Arnold and MLP Architectur (https://arxiv.org/abs/2503.23289)
- **What's New**: 본 논문에서는 Hybrid Parallel Kolmogorov-Arnold Network (KAN)와 Multi-Layer Perceptron (MLP)을 결합한 새로운 아키텍처인 HPKM-PINN을 제안한다. 이 모델은 KAN의 해석 가능성과 MLP의 비선형 특성 학습을 통합하여 출력 결과를 가중 평균함으로써 예측 성능을 향상시킨다. 또한, 스케일링 팩터 {}ξ{}를 도입하여 둘 간의 균형을 최적화하는 방식으로 설계되었다.

- **Technical Details**: HPKM-PINN의 접근 방식은 고전적인 PINN과는 다르게, KAN과 MLP의 병렬 구조를 통해 서로 보완적인 강점을 이용한다. KAN은 비선형 함수 근사에 뛰어나지만 저주파 구성 요소를 추출하는 데 어려움을 겪는 반면, MLP는 폭넓은 주파수 대역에서 학습이 가능하나 매개변수 중복 문제에 직면해 있다. HPKM-PINN은 이런 두 기술의 장점을 결합하여 복잡한 PDE 문제를 효율적으로 해결할 수 있는 기반을 마련하였다.

- **Performance Highlights**: HPKM-PINN은 Poisson 및 Advection 방정식과 같은 기준 PDE 문제에서 기존 KAN 또는 MLP 모델에 비해 손실 값(상대 오차를 두 자릿수 감소)에서 유의미한 감소를 보였다. 또한, 이 프레임워크는 다양한 물리 시스템에 적용할 때 숫자적 안정성과 강인성을 나타내며, 고주파 세부 정보와 부드러운 저주파 특성을 동시에 포착하는 능력을 강조하고 있다. 이 결과는 HPKM-PINN이 효율적이고 다목적 해결책으로 자리 잡을 가능성을 보여준다.



### Two Heads Are Better than One: Model-Weight and Latent-Space Analysis for Federated Learning on Non-iid Data against Poisoning Attacks (https://arxiv.org/abs/2503.23288)
- **What's New**: 이 논문에서는 GeminiGuard라는 새로운 접근 방식을 통해 Federated Learning(FL)에서의 모델 오염 공격(Model Poisoning Attacks, MPA)에 대한 효과적인 방어책을 제안합니다. 기존 방어 기법들은 일반적으로 데이터가 iid(독립 동등 분포)일 것으로 가정하고 설계되었으나, 실제 FL 환경에서는 데이터가 비iid(비독립 비동등 분포)인 경우가 많습니다. GeminiGuard는 모델 가중치 분석(Model-weight analysis)과 잠재 공간 분석(Latent-space analysis)을 결합하여 다양한 비iid 시나리오에서 MPA를 효과적으로 처리할 수 있도록 설계되었습니다.

- **Technical Details**: GeminiGuard는 경량화(lightweight), 다목적(versatile), 비지도 학습(unsupervised)을 목표로 하고 있으며, 이를 통해 다양한 MPA와 비iid 환경에 대한 방어 성능을 향상시킵니다. 모델 업데이드를 필터링하는 초기 단계에서 모델 가중치 기반 분석을 수행하며, 이는 코사인 유사도(Cosine similarity)와 유클리드 거리(Euclidean distance)를 기반으로 합니다. 또한, 잠재 공간 분석 모듈은 여러 레이어(layer)에서 활성화의 평균 거리를 측정하여 수신된 모델 업데이트의 신뢰성을 평가합니다.

- **Performance Highlights**: 실험 결과, GeminiGuard는 다양한 비iid 시나리오에서 기존 방어 기법들에 비해 일관되게 더 높은 성능을 보였습니다. 예를 들어, CIFAR-10에서 IBA 공격(IBA attack) 하에 GeminiGuard는 0.18%의 공격 성공률(Attack Success Rate, ASR)을 기록했으며, 기존 방어 기법은 10%를 초과하는 공격 성공률을 보였습니다. 실험은 총 네 가지 비목표 MPA와 다섯 가지 백도어 공격(backdoor attacks)을 포함한 방어 성능을 평가하였으며, GeminiGuard는 SOTA(State-of-the-Art) 방어법들에 비해 뛰어난 성능을 보여주었습니다.



### UP-ROM : Uncertainty-Aware and Parametrised dynamic Reduced-Order Model, application to unsteady flows (https://arxiv.org/abs/2503.23236)
- **What's New**: 본 연구에서는 비선형 감소 기법을 통해 일시적 흐름을 처리하기 위한 새로운 과정을 제안합니다. 이 방법은 변동성 추정(variational inference)을 통한 신뢰도 측정이 가능한 변동 오토인코더(Variational Auto-Encoder, VAE)를 포함하고 있습니다. 잠재 공간 변환(latent space transformer)을 사용함으로써, 외부요소의 영향을 효과적으로 수렴하여 더 높은 일반화 능력을 발휘합니다. 이러한 접근법은 모델의 예측을 신뢰하고 의사 결정을 더 정확하게 수행할 수 있게 지원합니다.

- **Technical Details**: 연구에서 제안하는 모델인 UP-ROM은 비선형 동역학을 정량화하기 위한 다차원 축소 기법을 사용합니다. 이를 위해 VAE를 활용하여 고차원 데이터를 낮은 차원 공간으로 압축하며, 이 과정에서 시간 의존성도 고려합니다. 모델은 다양한 외부 자극 요인에 적응할 수 있는 교차 주의 메커니즘을 통해 작동합니다. 이 모델은 물리적 공간과 매개변수 공간 모두에서 불확실성 측정을 포함할 수 있도록 설계되었습니다.

- **Performance Highlights**: 본 연구의 결과, UP-ROM은 전통적인 모델들과 비교하여 계산 효율성이 크게 향상되었습니다. 데이터 수집 없이 전체 매개변수 공간을 샘플링할 수 있으며, 이는 특히 복잡한 동역학 시스템을 다룰 때 유리합니다. 모델의 적응적인 성격 덕분에 다양한 자극에 따라 체계적인 예측이 가능하며, 의료용, 공학적 응용에서의 활용 가능성을 제시합니다. 이는 실시간 시뮬레이션 및 제어 시스템에 큰 도움이 될 것으로 기대됩니다.



### Citegeist: Automated Generation of Related Work Analysis on the arXiv Corpus (https://arxiv.org/abs/2503.23229)
- **What's New**: 이 논문은 대형 언어 모델(Large Language Models)의 연구 커뮤니티에서의 활용을 개선하기 위해 Citegeist라는 응용 프로그램 파이프라인을 소개합니다. 이는 arXiv(Corpus)의 데이터를 기반으로 동적 Retrieval Augmented Generation (RAG)을 사용하여 관련된 연구 결과 섹션과 인용 기반 출력을 생성합니다. 이러한 접근법은 잘못된 소스를 생성하는 경향을 완화하고 과학 논문에 대한 직접적인 접근성을 제공합니다.

- **Technical Details**: Citegeist는 임베딩 기반 유사도 매칭(embedding-based similarity matching), 요약(summarization), 다단계 필터링(multi-stage filtering)을 혼합하여 사용합니다. 또한, 문서 데이터베이스의 지속적인 성장에 적응하는 최적화된 방법을 소개하여 새로운 논문이나 수정된 논문을 쉽게 통합할 수 있도록 합니다. 이를 통해 연구자들이 보다 효과적으로 기존 자료에 접근할 수 있는 방법을 제시합니다.

- **Performance Highlights**: 이 연구는 Citegeist의 기능을 공개 웹사이트와 다양한 LLM(대형 언어 모델) 구현과 호환되는 구현 하네스를 통해 제공합니다. 이러한 도구들은 과학 커뮤니티에서의 활용을 용이하게 하여 연구자들이 보다 나은 인용 기반 작업을 생성할 수 있도록 지원합니다.



### Unsupervised Learning: Comparative Analysis of Clustering Techniques on High-Dimensional Data (https://arxiv.org/abs/2503.23215)
- **What's New**: 이 논문은 K-means, DBSCAN, Spectral Clustering과 같은 주요 클러스터링 알고리즘을 비교 분석한 결과를 담고 있습니다. 특히, PCA, t-SNE, UMAP 등 다양한 차원 축소 기법을 활용하여 클러스터링 성능을 평가하는 새로운 프레임워크를 제시합니다. 연구 결과는 UMAP을 통한 전처리가 모든 알고리즘에서 클러스터링 품질을 향상시키며, Spectral Clustering이 복잡한 다변량 구조에서 뛰어난 성능을 보여준다는 점을 강조합니다.

- **Technical Details**: 본 연구는 MNIST, Fashion-MNIST, UCI HAR 데이터셋에 대한 K-means, DBSCAN, Spectral Clustering의 성능을 체계적으로 비교했습니다. 각 데이터셋은 서로 다른 클러스터링 난이도를 가지고 있으며, PCA, t-SNE, UMAP 세 가지 차원 축소 기법이 적용되었습니다. 모델 평가는 Silhouette Coefficient, Davies-Bouldin Index와 같은 내부 지표와 Adjusted Rand Index, Normalized Mutual Information과 같은 외부 지표를 통해 수행되었습니다.

- **Performance Highlights**: 실험 결과, 모든 알고리즘은 차원 축소의 혜택을 받았으며 특히 UMAP은 모든 알고리즘과 데이터셋에서 최고의 성능을 보여주었습니다. DBSCAN은 UMAP 전처리 전과 비교해 ARI에서 2-3배 향상된 성능을 나타냈습니다. 연구의 결과는 데이터 특성과 성능 요구 사항에 따라 적절한 알고리즘 선택을 위한 유용한 가이드를 제공하는 데 기여합니다.



### A QUBO Framework for Team Formation (https://arxiv.org/abs/2503.23209)
- **What's New**: 이번 연구에서는 팀 구성 문제(Team Formation Problem)에 대한 통합된 접근 방식인 TeamFormation을 제안합니다. 이 문제는 전문가 집합과 일정 작업이 주어질 때, 필요한 기술의 범위를 최대화하고 동시에 전문가 비용을 최소화하는 것을 목표로 합니다. 우리는 세 가지 비용 함수의 변형을 제시하고, 이를 통해 QUBO(Quadratic Unconstrained Binary Optimization) 문제로 표현 가능함을 보여줍니다.

- **Technical Details**: 논문에서 제안하는 TeamFormation 문제는 전문가와 작업, 기술 간의 관계를 정의합니다. 각 전문가는 특정 기술을 보유하며, 작업은 특정 기술의 조합을 필요로 합니다. 우리는 전문가 선정 문제를 QUBO 문제로 재구성하여, 에너지 최소화 문제로 접근하고, 두 가지 여러 솔루션 방법을 탐구합니다: QUBO 솔버와 그래프 신경망(GNN) 방식입니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 기존의 특정 변형을 위한 조합적 기준선을 종종 초월하는 고품질 솔루션을 지속적으로 찾습니다. 또한, GNN을 통해 학습한 노드 임베딩은 관련 문제 인스턴스를 효율적으로 해결하기 위한 전이 학습(Transfer Learning) 가능성을 보여줍니다. 이를 통해 팀 형성의 다양한 변형 간에 일관된 알고리즘적 접근을 가능하게 합니다.



### TRA: Better Length Generalisation with Threshold Relative Attention (https://arxiv.org/abs/2503.23174)
- **What's New**: 이번 연구에서는 트랜스포머 모델의 길이 일반화(length generalisation) 문제를 분석합니다. 이 문제는 주로 self-attention 메커니즘의 두 가지 주요 결함에 기인한다고 주장합니다. 첫째, 관련 없는 정보를 완전히 제거하지 못하는 것이고, 둘째, 위치와 관련된 학습된 편향이 관련 없는 키를 불필요하게 강조할 수 있다는 점입니다. 이를 해결하기 위한 새로운 기법으로 임계값 상대 주의(Threshold Relative Attention, TRA)를 제안하여 성능을 향상시키려 합니다.

- **Technical Details**: TRA 메커니즘은 주의(weighting) 메커니즘을 개선하여 일반적인 솔루션을 학습할 수 있도록 합니다. 이 방법은 우선 원시 주의 가중치에 기반하여 관련 없는 키를 마스킹(masking)하고, 그 다음에 남은 키 사이의 상대 거리를 계산합니다. 이 과정을 통해, TRA는 대규모 트랜스포머의 길이 일반화 성능을 개선하고, 고장률(perplexity)을 낮출 수 있음을 보여줍니다.

- **Performance Highlights**: TRA를 적용한 결과, 합성 벤치마크에서 길이 일반화가 크게 개선되었습니다. 또한, 배포 밖의 시퀀스 길이에서도 언어 모델링의 고장률이 우수하였으며, 최대 32배까지 성능이 향상되었습니다. 이러한 결과는 더욱 강력한 주의 메커니즘 개발에 기여할 것으로 기대됩니다.



### Graph ODEs and Beyond: A Comprehensive Survey on Integrating Differential Equations with Graph Neural Networks (https://arxiv.org/abs/2503.23167)
- **What's New**: 이 논문은 그래프 신경망(Graph Neural Networks, GNNs)과 미분 방정식(Differential Equations, DEs) 간의 융합에 대한 최신 연구 개요를 제공합니다. 특히 GNNs와 DEs의 교차점을 활용하여 복잡한 동적 시스템을 모델링하는 접근 방식을 강조하고 있으며, 이를 통해 물리 기반 학습, 시공간 모델링 및 과학 컴퓨팅과 같은 다양한 응용 프로그램에 적용할 수 있습니다. 또한, 기존 연구에서 다루지 않았던 새로운 방법론과 응용 사례를 포함하여 구조화된 분류법을 제시합니다.

- **Technical Details**: 그래프(G)라는 용어는 노드 집합(𝒱)과 연결 정보를 담고 있는 엣지 집합(ℰ)으로 정의되며, 여기서 노드는 시스템의 상태를 나타내고 엣지는 노드 간의 관계를 설명합니다. GNNs는 이러한 그래프 데이터로부터 학습하는 유연한 구조를 제공하며, 메시지 전달(message passing) 기법을 사용하여 각 노드는 주변 노드로부터 받은 정보를 집계하여 자신의 표현을 업데이트합니다. 이 연구에서는 여러 종류의 미분 방정식, 즉 보통 미분 방정식(ODEs), 편미분 방정식(PDEs) 및 확률적 미분 방정식(SDEs)과의 통합에 관한 다양한 방법론을 다룹니다.

- **Performance Highlights**: GNNs와 미분 방정식의 결합은 복잡한 시스템의 동적 행동을 모델링하는 데 있어 혁신적인 접근 방식을 제공하며, 그 효과는 분자 모델링, 교통 예측 및 전염병 확산과 같은 실제 시나리오에서 관찰됩니다. 특히 그래프 기반 신경 미분 방정식(Graph Neural Differential Equations)들은 시간적 동적 변화를 캡처하고, 그래프 구조에 내재된 공간적 관계를 활용하여 시스템의 복잡성을 이해하는 데 기여합니다. 연구자들은 이 분야의 향후 연구 방향을 제시하고 몇 가지 열린 문제를 논의하여, GNNs와 DEs의 융합이 과학 및 산업 분야에서 더욱 발전할 수 있는 가능성을 염두에 두고 진행합니다.



### Reasoning-SQL: Reinforcement Learning with SQL Tailored Partial Rewards for Reasoning-Enhanced Text-to-SQL (https://arxiv.org/abs/2503.23157)
- **What's New**: 이번 연구는 Text-to-SQL 작업을 위해 설계된 새로운 부분 보상 집합을 제안합니다. 이는 기존의 수동으로 구성된 경로에 의존하는 방법의 한계를 극복하고, 더 나은 추론 능력과 일반화 능력을 갖춘 모델 개발을 돕습니다. 연구팀은 Reward-driven self-exploration을 활용한 Reasoning-SQL 프레임워크를 구축하여, LLM의 내부 추론 과정을 자동으로 최적화합니다.

- **Technical Details**: 새로운 보상 집합은 Schema Linking, AI 피드백, N-gram 유사성 및 구문 체크를 포함하여 강화 학습(Reinforcement Learning, RL)에서 흔히 발생하는 보상 희소성 문제를 해결하도록 설계되었습니다. Group Relative Policy Optimization (GRPO)을 활용하여 이들 보상 신호를 효과적으로 통합하고, 다양한 후보 쿼리를 생성하여 최종 실행 정확성을 최적화합니다. 방식으로는 다수의 모델 크기를 실험하여 RL 전용 학습이 감독 세부 조정(Supervised Fine-Tuning, SFT)보다 더 높은 정확도를 달성하는 모습을 보여줍니다.

- **Performance Highlights**: 제안된 방법을 사용한 RL 훈련 모델이 BIRD 벤치마크에서 기존의 큰 상용 모델들보다 4% 및 3% 더 높은 성능을 제공합니다. 특히, 14B 파라미터를 가진 RL 훈련 모델은 보다 낮은 비용으로 더 나은 성과를 보여주어, 기존의 Text-to-SQL 시스템과 비교할 때 상대적으로 93% 더 저렴한 비용으로 72.78%의 실행 정확도를 기록했습니다. 이로 인해 기존 SFT 방식보다도 더 경쟁력 있는 성능을 발휘할 수 있음을 입증했습니다.



### Agent-Based Modeling and Deep Neural Networks for Establishing Digital Twins of Secure Facilities under Sensing Restrictions (https://arxiv.org/abs/2503.23147)
Comments:
          This paper has been already published in the 2024 Interservice/Industry Training, Simulation, and Education Conference (I/ITSEC'24): this https URL The authors have obtained permission from I/ITSEC'24 organizers to release this paper on arXiv. Appropriate licensing is also applied

- **What's New**: 이 연구는 보안 핵시설에서 인간의 패턴 오브 라이프(POL)를 모니터링하기 위해 메타폴(MetaPOL)이라는 디지털 트윈 시스템을 도입하여, 비상사태 시나리오에 대한 응답과 정상 운영 간의 NPC(Non-Playable Character) 움직임의 차이를 분석했습니다. 실시간 시뮬레이션이 불가능한 환경에서 비정상적 상황을 예측하는 새로운 접근 방식을 제시합니다. 또한, 에이전트 기반 모델(ABM)을 기반으로 한 합성 움직임 궤적의 생성이 큰 역할을 했습니다.

- **Technical Details**: 해당 연구에서는 시설 직원의 POL에 대한 일화적 데이터를 활용하여 에이전트 기반 모델을 사용해 합성 움직임 궤적을 생성했습니다. 이러한 합성 궤적은 다음 위치와 머무는 시간 예측을 위한 딥 뉴럴 네트워크 서그레이트를 훈련시키는 데 사용되었습니다. 저자들은 다층 퍼셉트론(Multi-Layer Perceptron)과 혼합 밀도 네트워크(Mixture Density Network)를 이용해 합성 궤적을 예측하는 데 성공했습니다.

- **Performance Highlights**: 딥 뉴럴 네트워크에 의해 주도된 VR 환경 내 NPC의 움직임은 정상 운영 시나리오와 비상 응답 처리를 시뮬레이션할 때의 움직임과 유의미한 차이를 보였습니다. 연구 결과, 이러한 예측 시스템이 고Security 핵 시설에서의 안전성을 확보하는 데 중요한 역할을 할 수 있음을 입증하였습니다. 연구의 전반적인 결과는 보안 환경에서 시뮬레이션의 정확성을 크게 향상시킬 수 있는 잠재력을 보여줍니다.



### How to safely discard features based on aggregate SHAP values (https://arxiv.org/abs/2503.23111)
- **What's New**: 이번 연구는 SHAP (SHapley Additive exPlanations) 값을 활용한 글로벌 기능 중요도 평가의 한계를 다룹니다. 기존 방법에서는 작은 SHAP 값이 해당 기능이 함수에 영향을 미치지 않는다고 판단했지만, 연구 결과 그 판단은 잘못될 수 있음을 발견했습니다. 우리는 SHAP 값을 데이터 지원 밖에서 평가할 때 발생하는 문제를 강조합니다.

- **Technical Details**: 연구에서는 기능 i에 대한 SHAP 값이 0이라고 하더라도, 함수가 Feature i에 의존하는 경우가 있음을 명확하게 보여줍니다. 이에 대한 해결책으로, 우리는 SHAP 값을 기초 분포의 주변 분포 곱으로 확장된 지원에서 집계하는 방법을 제안합니다. 이러한 수정으로 인해 작은 집계 SHAP 값이 해당 기능을 안전하게 제거할 수 있음을 증명합니다.

- **Performance Highlights**: 또한, KernelSHAP에 대한 결과도 확장하여, 확장된 분포에서 계산될 때 작은 집계 값이 기능 제거를 정당화함을 보였습니다. 이러한 결과는 KernelSHAP이 진정한 SHAP 값을 정확하게 근사하는지 여부와 관계없이 성립합니다. 우리의 연구는 SHAP 및 KernelSHAP 알고리즘에 대한 이론적 및 실용적 함의를 가지고 있습니다.



### Fast Training of Recurrent Neural Networks with Stationary State Feedbacks (https://arxiv.org/abs/2503.23104)
Comments:
          18 pages (including additional contents), 3 figures, 5 tables, code available at this https URL

- **What's New**: 이번 연구에서는 Recurrent Neural Networks (RNNs)의 효율성을 높이는 새로운 방법을 제안합니다. 기존의 Backpropagation Through Time (BPTT) 알고리즘 대신에 고정된 gradient feedback 메커니즘을 도입하여 정확한 gradient 전달의 효율적인 근사를 제공합니다. 이를 통해 훈련 오버헤드를 크게 줄이면서도 장기 의존성(capacity to capture long-term dependencies)을 유지할 수 있습니다.

- **Technical Details**: 본 논문에서는 상태-공간 모델(State-Space Model, SSM) 원리를 활용하여 구조화된 피드백 행렬을 정의함으로써 미래 시간 단계로부터 직접적으로 gradient를 전달합니다. 이 방식은 gradient backpropagation의 재귀적 계산을 우회하여 훈련 시간을 절약하는 동시에 RNN의 메모리 능력을 보존입니다. 제안된 알고리즘은 BPTT 대신 시간 역으로 SSM을 추론하여 장거리 의존성을 처리하는 능력을 활용합니다.

- **Performance Highlights**: 실험 결과, 언어 모델링 벤치마크에서 제안된 방법이 경쟁력 있는 perplexity 점수를 보여주었으며, 훈련 비용이 현저히 감소했습니다. 이 결과들은 SSM과 같은 피드백 방법을 설계함으로써 RNN의 효율적 장점을 활용할 수 있음을 시사합니다. 이러한 성과는 RNN이 실용적인 응용 분야에서 큰 잠재력을 지니고 있음을 나타냅니다.



### The geomagnetic storm and Kp prediction using Wasserstein transformer (https://arxiv.org/abs/2503.23102)
- **What's New**: 본 연구에서는 위성 측정, 태양 이미지 및 Kp 시계열 데이터를 포함한 이질적 데이터 소스를 통합한 새로운 멀티모달 Transformer 기반 프레임워크를 제안합니다. 이 프레임워크는 3일 및 5일 치 Kp 지수를 예측하는 데 중점을 두고 있으며, Wasserstein 거리(wasserstein distance)를 Transformer 및 손실 함수에 통합하여 모달리티 간의 확률 분포를 정렬합니다. NOAA 모델과의 비교 실험을 통해 조용한 지자기 활동과 폭풍 단계 모두를 정확히 포착하며 성능을 입증하였습니다.

- **Technical Details**: 지자기 폭풍은 태양풍 조건에 의해 발생하며, 이는 코로나 질량 방출(coronal mass ejections) 및 고속 태양풍 스트림(source)과 같은 원인으로 인해 발생합니다. Kp 지수는 이러한 지자기 교란의 세기를 정량화하는 기본 매개변수로, 연구에서는 이 과정과 Kp 지수에 영향을 미치는 물리적 및 수학적 메커니즘을 소개합니다. 연구는 주로 신경망(neural network)을 사용하여 3일 Kp 지수를 예측하고 NASA 데이터를 비교합니다.

- **Performance Highlights**: 이 연구는 고급 기계 학습(machine learning) 기술을 전통적인 모델과 통합하여 실시간 예측의 잠재력을 강조합니다. Kp 지수 예측의 정확성은 특히 전력망, 통신 네트워크 및 위성 작업과 같은 인프라에 대한 지자기 폭풍의 영향을 줄이는 데 필수적입니다. CMETNet이라는 앙상블 학습 프레임워크가 CME의 도착 시간을 예측하는 데 사용되며, 이로써 한국 기상청의 Kp 지수 예측 체계에 신뢰성을 더할 수 있습니다.



### RL2Grid: Benchmarking Reinforcement Learning in Power Grid Operations (https://arxiv.org/abs/2503.23101)
- **What's New**: 이 논문에서는 전력망 운영의 혁신을 목표로 하는 RL2Grid라는 새로운 벤치마크를 소개합니다. RL2Grid는 전력 시스템 운영자들과 협력하여 개발되어, 전력망 제어의 발전을 가속화하고 RL(강화학습) 방법의 성숙도를 높이는 데 도움을 줍니다. 이 벤치마크는 전력망 작업에서의 임무, 상태 및 행동 공간, 보상 구조를 통합된 인터페이스로 표준화하여 RL 접근법의 평가 및 비교를 용이하게 합니다.

- **Technical Details**: RL2Grid는 전력망의 복잡한 동역학을 다루기 위해 RTE 프랑스에서 개발한 전력 시뮬레이션 프레임워크를 바탕으로 구축되었습니다. 이 벤치마크는 다양한 전력망 업무를 포함하고 있으며, 각 업무는 안전 제약 조건을 통합한 제약된 태스크 포멀라이제이션을 제공합니다(예: 전력 흐름, 발전기 제한 및 라인 제한). 또한, 에이전트가 그리드를 효과적으로 운영하도록 학습하는 과정을 분석하기 위해 RL2Grid의 설계 선택에 대한 포괄적인 분석이 수행됩니다.

- **Performance Highlights**: 우리는 RL2Grid 내의 전력망 제어 작업에 대해 여러 인기 있는 RL 알고리즘의 성능을 benchmark 하였습니다. 특히, 기본적인 그리드 작업(예: 선 재연결 및 대기 동작)을 기존 알고리즘의 교육 루프에 통합할 수 있는 휴리스틱 모듈이 구현되어, 모든 RL 알고리즘의 성능과 샘플 효율성이 크게 향상되었습니다. RL2Grid는 RL 방법을 실제 전력망 환경에서 성숙시키는 출발점으로 기능하며, 개방된 문제들과 연결된 여러 방향을 논의합니다.



### Beyond Standard MoE: Mixture of Latent Experts for Resource-Efficient Language Models (https://arxiv.org/abs/2503.23100)
- **What's New**: 본 논문은 Mixture of Experts (MoE) 아키텍처의 한계를 극복하기 위해 Mixture of Latent Experts (MoLE)라는 새로운 매개변수화 방법론을 제안합니다. MoLE는 전문가 모듈을 공유 잠재 공간으로 매핑함으로써 모델의 파라미터 수와 계산 요구사항을 현저히 줄입니다. 이를 통해 기존 MoE 아키텍처보다 더 효율적으로 대규모 언어 모델(LLMs)을 확장할 수 있는 방법을 제시합니다.

- **Technical Details**: MoLE 접근법은 각 전문가 작업을 두 가지 주요 구성 요소로 체계적으로 분해합니다: 먼저 압축된 잠재 공간으로의 공유 투영을 수행하고, 그 다음 전문가에 특화된 변환을 적용합니다. 이 과정에서 MoLE는 각 전문가의 가중치 행렬을 인수 분해하여 매개변수 수를 크게 줄이고 계산 복잡성을 감소시킵니다. 알고리즘 측면에서, 최적의 인수 분해 조건을 수학적으로 정의하고, 효율적인 두 단계 변환 알고리즘을 개발합니다.

- **Performance Highlights**: 광범위한 실험 평가를 통해 MoLE는 기존 MoE 아키텍처와 비교했을 때 경쟁력 있는 성능을 유지하면서도 자원 요구사항을 획기적으로 줄임을 입증하였습니다. 이러한 점은 특히 자원이 제한된 환경에서도 MoLE 모델을 실용적으로 적용할 수 있는 가능성을 보여줍니다. 또한, MoLE는 다양한 언어 처리 작업에서 모델 능력을 보존하거나 향상시키는 것으로 나타났습니다.



### TRACE: Intra-visit Clinical Event Nowcasting via Effective Patient Trajectory Encoding (https://arxiv.org/abs/2503.23072)
Comments:
          Accepted by WWW'25 short paper track

- **What's New**: 이 논문은 전자 건강 기록(EHR)을 기반으로 한 임상의료 이벤트의 인차 예측(intra-visit nowcasting)에 관한 새로운 연구를 소개합니다. 이는 병원 방문 중 실시간으로 환자의 상태를 예측하는 데 중점을 두고 있으며, 기존의 방법들이 주로 방문 간(predictions between visits) 이벤트에 초점을 맞춘 것과 차별화됩니다. 특히 전통적으로 간과되었던 실험실 측정 예측을 다루기 위해 TRACE라는 새로운 Transformer 기반 모델을 제안합니다.

- **Technical Details**: TRACE는 환자의 진료 경로를 인코딩하기 위해 맞춤형 타임스탬프 임베딩(timestamp embedding)을 통합합니다. 이는 의학적 사건의 감소(decay) 성질과 주기적 패턴을 포착하여 모델이 과거 사건의 영향을 효과적으로 고려하고, 실험실 검사 결과의 시간 주기적 영향을 통합할 수 있도록 합니다. 또한, 주의(attention) 계산 중 덜 영향력 있는 정보를 필터링하여 모델의 강인성을 높이는 스무딩 마스크(Smooth Mask) 모듈을 도입했습니다.

- **Performance Highlights**: 실험 결과, TRACE는 MIMIC-III 및 MIMIC-IV 데이터셋에서 이전 방법들보다 우수한 성능을 보였으며, PR-AUC와 Precision@k 지표에서 가장 높은 점수를 기록했습니다. 이러한 결과는 TRACE가 의료 데이터의 복잡한 시간적 종속성을 효과적으로 모델링하고, 정확한 실험실 측정 예측을 제공할 수 있는 잠재력을 강조합니다. 최종 코드 또한 제공되어 향후 연구와 실제 적용 가능성을 지원합니다.



### Unsupervised Anomaly Detection in Multivariate Time Series across Heterogeneous Domains (https://arxiv.org/abs/2503.23060)
- **What's New**: 이번 논문은 인공지능(AI) 기반 IT 운영(AIOps)에서 멀티변량 시계열 데이터의 이상 탐지(Anomaly Detection, AD)를 위한 새로운 접근인 Domain-Invariant VAE for Anomaly Detection(DIVAD)을 소개합니다. 이는 도메인 일반화(domain generalization) 프레임워크를 사용하여, 서로 다른 도메인에서 나타나는 정상 행동의 변화를 효과적으로 처리하고자 하였습니다. 이 연구는 기존의 방법들이 종종 새로운 도메인에서 신뢰할 수 없는 성능을 보여주는 문제를 해결하고자 합니다.

- **Technical Details**: 이 논문은 멀티변량 시계열 데이터에서 이상 탐지를 수행하기 위한 새로운 이론적 형태를 제안합니다. Domain-Invariant VAE(DIVAD)는 정상적인 행동의 변화를 감지하기 위해 도메인 불변(domain-invariant) 표현을 학습하는 변형들을 포함합니다. 이 방법은 기존의 이상 탐지 기법들이 갖는 제한을 극복하기 위해 개발되었으며, 다양한 도메인에 대한 일반화 능력을 강화합니다.

- **Performance Highlights**: Exathlon 벤치마크를 이용한 평가 결과, DIVAD의 두 가지 주요 변형이 기존의 최선의 비지도 AD 방법보다 각각 20% 및 15% 개선된 최대 F1-점수를 기록하며 뛰어난 성능을 보여주었습니다. 추가적으로 Application Server Dataset에서도 DIVAD가 더 넓은 활용 가능성을 발휘함을 입증했습니다. 이 연구 결과는 AIOps 시나리오에서의 효율적인 해결책이 될 것으로 기대됩니다.



### Prediction of 30-day hospital readmission with clinical notes and EHR information (https://arxiv.org/abs/2503.23050)
- **What's New**: 이 논문은 임상 노트(clinical notes)와 전자 건강 기록(EHR)을 결합하여 30일 재입원을 예측하는 새로운 모델을 제안합니다. 병원 재입원률은 병원 치료의 질을 나타내는 지표로 간주되며, 이 모델은 의료 전문가가 환자의 건강 상황을 조기에 파악하고 필요한 치료를 제공할 수 있도록 지원할 수 있습니다. 이를 위해, 다양한 정보 유형에 대한 표현 방식을 탐구하며 그래프 신경망(GNN)을 사용하여 환자 데이터를 구조화하여 대규모 환자 코호트를 처리합니다.

- **Technical Details**: 이 연구에서 사용된 데이터셋은 MIMIC-IV 버전 2.2로, 2008년부터 2019년까지의 재원 환자 정보를 포함합니다. GNN 모델을 기반으로 하여, 각 입원을 그래프의 노드로 간주하고 유사한 특성을 가진 입원들을 연결합니다. 이는 환자 간의 복잡한 관계를 모델링할 수 있게 해주며, 임상 노트는 비구조적인 데이터로서 이 모형에서 중요한 역할을 합니다.

- **Performance Highlights**: 제안된 모델은 AUROC 0.72를 기록하고, 균형 정확도(balanced accuracy)는 66.7%로, 다양한 정보를 결합하는 것이 재입원 예측에 있어 중요함을 강조합니다. 전통적인 기계 학습(machin learning) 모델과 비교했을 때, 더 복잡한 관계를 캡처할 수 있는 GNN의 장점을 활용하여 높은 성능을 보여줍니다.



### Function Fitting Based on Kolmogorov-Arnold Theorem and Kernel Functions (https://arxiv.org/abs/2503.23038)
Comments:
          19 pages, 12 figures

- **What's New**: 본 논문은 Kolmogorov-Arnold representation theorem과 kernel 방법을 기반으로 한 통합 이론적 틀을 제안합니다. KAN(Kolmogorov-Arnold Networks)과 self-attention 메커니즘 간의 수학적 관계를 분석하여, 두 모델을 kernel 함수의 선형 결합으로 통합하는 kernel 기반 피처 피팅 프레임워크를 확립하였습니다. 또한, 기존 MHSA(Multi-Head Self-Attention)의 파라미터 수를 거의 50% 줄인 저랭크 Pseudo-MHSA 모듈을 제안하였습니다.

- **Technical Details**: 이 연구에서는 Kolmogorov-Arnold representation theorem을 통해 신경망의 표현력을 설명하고, self-attention 메커니즘과의 통합 방법을 설명합니다. KAN은 B-spline 기저 함수를 사용하여 비선형 패턴을 모델링하며, self-attention은 고차원 공간에서 유사성을 계산하는 kernel 방법과 일치합니다. 저랭크 근사 방법을 활용하여 Pseudo-MHSA를 설계하였으며, 이는 파라미터 수를 줄이면서도 계산 효율성을 개선합니다.

- **Performance Highlights**: CIFAR-10 데이터셋에서의 실험 결과, Pseudo-MHSA 모델은 동일 차원의 ViT(Vision Transformer) 모델과 유사한 성능을 발휘하였으며, MAE(Masked Autoencoder) 프레임워크 하에서도 유효성을 입증했습니다. 시각화 분석 결과, 다중 헤드 분포 패턴의 유사성을 보여주었습니다. 전체적으로 최적화 모델은 파라미터 효율성과 성능 면에서 상당한 장점을 나타냈습니다.



### Towards Understanding the Optimization Mechanisms in Deep Learning (https://arxiv.org/abs/2503.23016)
- **What's New**: 이 논문에서는 깊은 신경망(deep neural networks)의 감독 분류(supervised classification) 최적화 메커니즘을 확률 분포 추정(probability distribution estimation) 관점에서 탐구합니다. Fenchel-Young 손실(Fenchel-Young loss)을 사용하여 비볼록적(non-convex)인 적합 오류(fitting error)에도 불구하고 전역 최적(global optimal) 솔루션을 근사할 수 있음을 보여줍니다. 이 방법은 기울기 정규화(gradient norm)와 구조적 오류(structural error)를 동시에 최소화하는 것을 통해 가능해집니다.

- **Technical Details**: 논문은 모델의 매개변수(parameter)에 대한 기울기 독립성(gradient independence) 가정 하에, 구조적 오류가 모델 매개변수 수에 의해 제어된다는 것을 증명합니다. 즉, 매개변수 수가 많을수록 구조적 오류는 작아집니다. 이러한 결론은 과적합(over-parameterization) 및 무작위 초기화(random initialization)와 같은 기술에 대한 이론적 통찰(theoretical insights)을 제공합니다. Fenchel-Young 손실은 많은 손실 함수가 이 형태로 표현될 수 있다는 점에서 깊은 학습 분석의 복잡성을 줄이는 데 유용한 접근 방식으로 제시됩니다.

- **Performance Highlights**: 제안된 방법의 핵심 결론은 실험적 결과를 통해 검증되었습니다. 네트워크의 매개변수 수를 늘리고 매개변수 간의 독립성을 보장함으로써, 분류 문제는 특정 입출력(feature와 label) 조건부 확률 분포 추정과 동등하다는 것을 보여줍니다. 이러한 기법들을 통해 DNN의 훈련 메커니즘 및 비볼록 최적화(non-convex optimization)에서의 동작을 보다 잘 이해할 수 있게 됩니다.



### MSNGO: multi-species protein function annotation based on 3D protein structure and network propagation (https://arxiv.org/abs/2503.23014)
Comments:
          8 pages, 2 figures

- **What's New**: 최근 몇 년 동안, 단백질 기능 예측에서 AlphaFold2에 의해 예측된 고정밀 단백질 구조를 활용하여 예측 정확도가 크게 향상되었습니다. 특히, 단일 종의 단백질 기능 예측 방법은 큰 발전을 이루었지만, 다중 종의 단백질 기능 예측 방법은 여전히 PPI 네트워크 및 서열 기능을 기반으로 한 단계에 있습니다. 이 문제를 해결하기 위해, MSNGO 모델을 제안하며, 이는 구조적 특성과 네트워크 전파 방법을 통합합니다.

- **Technical Details**: MSNGO 모델은 그래프 표현 학습 기술을 활용하여 단백질 구조 접촉 맵에서 아미노산 표현을 추출하고, 그래프 합성 풀링 모듈을 사용하여 단백질 수준의 구조적 특성을 도출하는 구조적 모델을 학습합니다. ESM-2로부터 얻은 서열 특성을 포함시킨 후, 네트워크 전파 알고리즘을 적용하여 정보 집합 및 이질 네트워크 내에서 노드 표현을 업데이트합니다.

- **Performance Highlights**: MSNGO는 서열 기능 및 PPI 네트워크에 의존하는 이전의 다중 종 단백질 기능 예측 방법들보다 우수한 성능을 보였습니다. 이 모델을 통한 검증은 구조적 특성을 사용하는 것이 다중 종 단백질 기능 예측의 정확도를 현저하게 향상시킬 수 있음을 입증하였습니다.



### Learning Structure-enhanced Temporal Point Processes with Gromov-Wasserstein Regularization (https://arxiv.org/abs/2503.23002)
Comments:
          Accepted at the Web Conference workshop 2025

- **What's New**: 이 연구에서는 temporal point processes (TPPs)의 클러스터링 구조를 효과적으로 학습하는 새로운 정규화 방법인 Gromov-Wasserstein (GW) 정규화기를 적용합니다. 기존 TPP들은 일반적으로 이벤트 시퀀스의 내재된 클러스터링 구조를 간과하여 해석 가능성이 떨어집니다. 본 연구에서는 최대 우도 추정(maximum likelihood estimation)에 기반하여 클러스터링 구조를 시퀀스 수준의 임베딩에 부과합니다. 이 과정에서 비모수적 TPP 커널을 활용하여 유사성 행렬을 정규화합니다.

- **Technical Details**: 연구에서는 N개의 이벤트로 구성된 이벤트 시퀀스를 정의하고, 이를 기반으로 파라메트릭 TPP를 다변량 카운팅 프로세스로 표현합니다. TPP는 이벤트 시퀀스의 역학을 포착하기 위해 다변량 조건 강도 함수(multivariate conditional intensity function)를 사용하고, 기대되는 순간 발생률을 시간 t에서의 이벤트 이력에 따라 결정합니다. 이 연구에서는 고차원 클러스터링 정규화를 사용하여 유사성 행렬을 정규화하는 커널 행렬을 설계하고, 이를 통해 TPP의 학습을 보다 효과적으로 진행합니다.

- **Performance Highlights**: 실험 결과, 제안된 정규화기를 사용하여 학습된 TPP는 경쟁력 있는 예측 정확도를 달성하며, 클러스터링된 시퀀스 임베딩을 생성합니다. 이는 모델의 해석 가능성을 크게 향상시키면서도 예측 정확도를 유지하는데 기여합니다. 대규모 응용에서도 상대적으로 낮은 계산 복잡도를 유지하며, 모든 클러스터에 독립적인 복잡성으로 확장 가능한 해결책을 제공합니다. 이 방법은 파라메트릭 및 비파라메트릭 TPP 모델의 장점을 결합하여 효과적은 예측 및 클러스터링 성능을 보여줍니다.



### Buyer-Initiated Auction Mechanism for Data Redemption in Machine Unlearning (https://arxiv.org/abs/2503.23001)
Comments:
          Submitted to IEEE GLOBECOM 2025

- **What's New**: 최근 인공지능(AI)의 급속한 발전은 사용자 데이터에 대한 프라이버시 문제를 불러일으켰습니다. 이로 인해 GDPR(General Data Protection Regulation) 및 CCPA(California Consumer Privacy Act)와 같은 규제가 제정되었습니다. 기계 비학습(machine unlearning) 기술을 통해 AI 서비스 제공자는 사용자 데이터를 모델에서 제거하고, 이를 통해 규제 준수가 가능해졌습니다. 하지만 데이터의 과도한 삭제는 비용이 많이 들고 모델의 정확도를 저하시킬 수 있습니다.

- **Technical Details**: 이 논문에서는 데이터 환급을 위한 구매자 주도 경매 메커니즘을 제안합니다. 이 시스템은 사용자가 자신의 데이터 판매에 대해 적절한 보상을 받도록 하여 AI 서비스 제공자의 비용 부담을 완화하고 모델의 정확도를 높이는 데 기여합니다. 기존 연구에서 사용된 데이터 가격 책정 모델은 데이터 거래를 위한 시나리오에 중점을 두고 있으나, 데이터 환급 및 비학습 시나리오에서는 적용하기 어렵습니다. 본 연구는 서버가 사용자의 개인정보 분포에 대한 사전 지식이 필요 없는 유연한 가격을 제시할 수 있도록 돕습니다.

- **Performance Highlights**: 세부적인 시뮬레이션 평가 결과, 제안된 구매자 주도 경매 메커니즘은 사용자 개인 정보 및 유틸리티를 보다 잘 반영하여 모델 성능 향상에 기여합니다. 기존의 경직된 비학습 정책보다 유연하게 사용자와 서버의 이익을 조율할 수 있는 잠재력을 보여 줍니다. 또한, 이 새로운 프레임워크는 사회적 후생을 극대화하는 방향으로 나아가며 개인정보 보호를 강화하는 동시에 AI 서비스 제공자에게 이익이 되는 경로를 제시합니다.



### AuditVotes: A Framework Towards More Deployable Certified Robustness for Graph Neural Networks (https://arxiv.org/abs/2503.22998)
Comments:
          20 pages

- **What's New**: 이번 논문에서는 GNNs(그래프 신경망)의 새로운 프레임워크인 	extbf{AuditVotes}를 제안합니다. 이 프레임워크는 높은 클린 정확도(clean accuracy)와 인증된 강건성(certified robustness)을 동시에 달성하는 데 중점을 두고 있습니다. 기존의 방법들이 정확도와 강건성 사이의 중대한 핀치(pinch) 문제에 직면한 반면, AuditVotes는 데이터를 개선하고 예측 일관성을 보장하기 위한 두 가지 핵심 요소인 augmentation과 conditional smoothing을 통합합니다.

- **Technical Details**: 제안된 AuditVotes는 랜덤 스무딩(randomized smoothing) 기법을 사용하여 그래프의 데이터 품질을 향상시키고 예측의 일관성을 높입니다. augmentation은 사전 처리(pre-processing) 단계로 작용하여 랜덤 그래프에서 노이즈를 제거하고 데이터 품질을 크게 개선합니다. 그 후 conditional smoothing은 저품질 예측을 선택적으로 필터링하여 투표의 일관성을 높이는 후처리(post-processing) 단계 역할을 수행합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, AuditVotes는 클린 정확도, 인증된 정확도, 경험적 강건성을 크게 향상시키는 동시에 높은 계산 효율성을 유지합니다. 특히, Cora-ML 데이터 세트에서 공격자가 20개의 엣지를 임의로 추가할 경우, AuditVotes는 기존 랜덤 스무딩에 비해 클린 정확도를 437.1% 개선하고 인증된 정확도를 409.3% 향상시킵니다. 이는 실세계 응용 프로그램에서 인증된 강건한 GNNs의 배포를 위한 중요한 진전을 나타냅니다.



### DC-SGD: Differentially Private SGD with Dynamic Clipping through Gradient Norm Distribution Estimation (https://arxiv.org/abs/2503.22988)
Comments:
          Accepted at IEEE Transactions on Information Forensics & Security

- **What's New**: 새로운 접근 방식인 Dynamic Clipping DP-SGD (DC-SGD)를 제안했습니다. 이 프레임워크는 히스토그램을 활용하여 기울기 정규 분포를 추정하고, 클리핑 임계값 C를 동적으로 조정합니다. DC-SGD는 두 가지 새로운 메커니즘인 DC-SGD-P와 DC-SGD-E를 포함하여, 각각의 기울기 정규의 백분위수 및 기댓값 제곱 오차를 기반으로 클리핑 임계값을 조정합니다. 이러한 동적 조정은 하이퍼파라미터 튜닝의 부담을 크게 줄여줍니다.

- **Technical Details**: DC-SGD는 기울기 정규 분포 추정을 위해 차별적 개인 정보 보호(Differential Privacy) 방식의 히스토그램을 사용합니다. DC-SGD-P는 기울기 정규의 백분위수에 따라 클리핑 임계값을 조정하고, DC-SGD-E는 기댓값 제곱 오차를 최소화하여 클리핑 임계값을 최적화합니다. 이 과정을 통해 추가적인 하이퍼파라미터 조정 없이, 임계값 C를 동적으로 설정할 수 있습니다. Adam 옵티마이저와 통합하여 학습률 조정의 복잡성을 크게 줄이는 것이 특징입니다.

- **Performance Highlights**: 다양한 딥러닝 작업에 대한 실험에서 DC-SGD는 DP-SGD에 비해 하이퍼파라미터 튜닝 속도를 최대 9배 향상시켰습니다. CIFAR10 데이터세트에서는 같은 개인 정보 보호 예산 하에 DP-SGD보다 10.62%의 정확도 향상을 달성했습니다. 실험 결과, DC-SGD-P는 SVHN 데이터세트에서 2.13%의 개선을 보였으며, 전반적으로 DC-SGD 방식이 더 나은 모델 성능을 발휘함을 입증했습니다.



### Enhancing Federated Learning Through Secure Cluster-Weighted Client Aggregation (https://arxiv.org/abs/2503.22971)
- **What's New**: 본 논문은 새로운 연합 학습(FL) 프레임워크인 ClusterGuardFL을 소개하고 있으며, 이는 클라이언트 업데이트에 동적으로 가중치를 할당하기 위해 비유사성 점수(dissimilarity scores), K-평균 클러스터링(k-means clustering) 및 재조정 신뢰 점수(reconciliation confidence scores)를 활용합니다. 이 프레임워크는 FL의 공정성과 보안을 강화하여 데이터 독성을 방지하는 목표를 가지고 있습니다. 실험 결과는 구조의 유효성을 입증하여 다양한 데이터 세트에서 모델 성능을 개선할 수 있음을 보여줍니다.

- **Technical Details**: ClusterGuardFL은 전세계 모델과 클라이언트 로컬 모델 간의 비유사성 점수를 이용하여 클러스터를 형성하고, 각 클러스터 내에서 개별 데이터 포인트에 대해 재조정 신뢰 점수를 계산합니다. 이러한 동적 가중치 할당 방식은 학습 과정의 집계 과정에서 적용되어 모델의 강건성과 개인 정보 보호를 강화합니다. FL의 비대칭적 특성과 클라이언트 자원에 따라 다양한 공정성을 제공할 수 있습니다.

- **Performance Highlights**: ClusterGuardFL은 다양한 실제 데이터 세트를 사용한 실험을 통해 그 효과를 입증하였고, 기존 FL 방법들에 비해 성능과 안정성을 향상시키는 결과를 도출하였습니다. 각 클라이언트의 기여를 공정하게 측정하여 전반적인 모델 성능을 개선함과 동시에, 데이터 독성 공격에 대한 강고함을 증가시켰습니다. 이를 통해 CLusterGuardFL은 FL 시스템의 공정성과 보안을 동시에 향상시키기 위한 효과적인 솔루션을 제공합니다.



### Multimodal machine learning with large language embedding model for polymer property prediction (https://arxiv.org/abs/2503.22962)
- **What's New**: 이번 연구에서는 PolyLLMem이라는 다중 모달 아키텍처를 제안하여 Llama 3로 생성된 텍스트 임베딩과 Uni-Mol에서 유도된 분자 구조 임베딩을 통합하여 고분자 특성 예측 작업을 수행합니다. 이 아키텍처는 제한된 고분자 데이터셋에 기반하여 Low-rank adaptation (LoRA) 레이어를 활용해 임베딩을 정제함으로써 고분자 SMILES 표현의 화학적 관련성을 높였습니다. 이러한 통합된 접근법을 통해 PolyLLMem은 데이터를 효과적으로 활용하면서도 다양한 고분자 특성을 정확하게 예측할 수 있습니다.

- **Technical Details**: PolyLLMem의 주요 기술적 요소로는 Llama 3로부터 생성된 텍스트 임베딩과 Uni-Mol로부터 도출한 분자 구조 임베딩을 결합한 점이 특징입니다. LoRA 레이어를 적용하여 예측 작업에서 고분자 데이터셋을 반영한 임베딩을 미세 조정함으로써, 데이터의 부족함을 극복하고 전반적인 예측 성능을 개선할 수 있었습니다. 이 모델은 고분자 PSMILES에 인코딩된 화학 정보를 효과적으로 캡처하여, 고유의 화학 정보 기반 예측을 가능하게 합니다.

- **Performance Highlights**: PolyLLMem의 성능은 그래프 기반 모델 및 사전 훈련된 변환기 모델과 비교할 때 유사하거나 경우에 따라 더 나은 결과를 나타냈습니다. 22개의 고분자 특성을 예측하는 작업에서 PolyLLMem은 제한된 데이터셋으로도 우수한 성능을 발휘하며, 기존의 데이터 세트 수집이나 추가적인 사전 훈련 없이도 빠르고 간단하게 구현할 수 있습니다. 이러한 이점들은 고분자 자재의 발견을 가속화하는 데 기여할 수 있습니다.



### MNT-TNN: Spatiotemporal Traffic Data Imputation via Compact Multimode Nonlinear Transform-based Tensor Nuclear Norm (https://arxiv.org/abs/2503.22955)
- **What's New**: 이 논문에서는随机缺失값(ramdom missing values)의 문제를 다루기 위해 새로운 spatiotemporal traffic imputation 방법인 Multimode Nonlinear Transformed Tensor Nuclear Norm (MNT-TNN)을 제안했습니다. 기존의 Tensor Nuclear Norm (TTNN) 최적화 프레임워크를 기반으로 하여, 비선형 활성화와 다중 모드를 활용하여 교통 텐서의 내재된 spatiotemporal 상관관계 및 저랭크성(low-rankness)을 효과적으로 포착합니다. 또한, 매우 높은 결측률에서도 imputation 성능을 개선하기 위한 Augmented Transform-based Tensor Nuclear Norm Families (ATTNNs) 프레임워크를 도입했습니다.

- **Technical Details**: MNT-TNN 방법론은 교통 데이터의 spatiotemporal 상관관계를 모델링하기 위해 Multi-mode Transform을 사용하며, 비선형 활성화가 결합되어 다변량 시간 연속 시계열 데이터를 처리합니다. 또한, Proximal Alternating Minimization (PAM) 알고리즘을 설계하여 비볼록(nonconvex) 최적화 문제를 해결하며, 이론적 수렴 보장을 제공합니다. ATTNNs 프레임워크는 TTNN 기술을 보강하여 다양한 결측률에서 imputation 결과를 개선하도록 설계되었습니다.

- **Performance Highlights**: 실제 데이터셋에서 진행된 광범위한 실험 결과, MNT-TNN 및 ATTNNs는 기존의 최신 imputation 기법보다 우수한 성능을 보여주었습니다. 특히, 모든 결측률 범위에 걸쳐 spatiotemporal traffic imputation에서 뛰어난 성과를 달성했습니다. 논문에서는 또한 Deep Learning(DL) 기반의 imputation 방법의 한계에 대한 검토와 비랜덤 결측값(non-random missing value) imputation에 대한 초기 조사도 제공하고 있습니다.



### Graph Kolmogorov-Arnold Networks for Multi-Cancer Classification and Biomarker Identification, An Interpretable Multi-Omics Approach (https://arxiv.org/abs/2503.22939)
- **What's New**: 이번 연구에서는 다중 오믹스 데이터의 통합이 정밀 의학에서 큰 도전 과제가 된다는 점을 강조하고, 이를 위한 새로운 딥러닝 모델인 MOGKAN(Multi-Omics Graph Kolmogorov-Arnold Network)을 소개합니다. MOGKAN은 메신저 RNA, 마이크로 RNA 시퀀스, DNA 메틸화 데이터 및 단백질-단백질 상호작용(PPI) 네트워크를 통합하여 31종 암의 정확한 분류를 가능하게 합니다. 이 모델은 복잡한 다중 오믹스 데이터를 의미 있는 생물학적 특징을 유지하면서 차원 축소하는 하이브리드 접근 방식을 사용합니다.

- **Technical Details**: 모델 아키텍처는 Kolmogorov-Arnold 정리에 기반을 두고 있으며, 해석 가능성을 높이고 특징 분석을 향상시키기 위해 훈련 가능한 단변량 함수(univariate functions)를 활용합니다. 데이터 처리에는 DESeq2, LIMMA, LASSO 회귀 등을 통해 차원 축소와 함께 의미 있는 생물학적 신호를 보존하는 방식을 채택하였습니다. MOGKAN은 96.28%의 분류 정확도를 달성하고, 실험적 변동성을 낮추어 CNNs(Convolutional Neural Networks) 및 GNNs(Graph Neural Networks)와 비교하여 표준 편차를 1.58%에서 7.30%까지 줄였습니다.

- **Performance Highlights**: MOGKAN이 확인한 바이오마커는 Gene Ontology(GO) 및 Kyoto Encyclopedia of Genes and Genomes(KEGG) 풍부도 분석을 통해 암 관련 마커로 검증되었습니다. 이 모델은 분자 발암 메커니즘을 밝히는 능력을 가지고 있으며, 인산이노시톨 결합 물질 및 스핑고리피드 세포 과정을 조절하는 데 기여합니다. 통합된 다중 오믹스 데이터와 그래프 기반 딥러닝 접근법을 통해 MOGKAN은 예측 성능과 해석 가능성을 크게 개선하여 복잡한 다중 오믹스 데이터를 임상적으로 유용한 암 진단으로 전환할 수 있는 잠재력을 보여줍니다.



### FairSAM: Fair Classification on Corrupted Data Through Sharpness-Aware Minimization (https://arxiv.org/abs/2503.22934)
- **What's New**: 이 논문에서는 서로 다른 인구 집단 간의 성능 저하를 평가하기 위해 새로운 메트릭을 도입하고, Fairness-oriented한 전략을 통합한 FairSAM이라는 새로운 프레임워크를 제안합니다. 기존의 Sharpness-Aware Minimization(SAM) 방법이 전반적인 모델의 강건성을 높이는 데 기여했지만, 인구 집단 간의 성능 불균형 문제를 효과적으로 해결하지 못했다는 점을 강조합니다. FairSAM은 혼합된 데이터 환경에서 강건성과 공정성을 동시에 달성할 수 있도록 설계되었습니다.

- **Technical Details**: FairSAM 프레임워크는 인스턴스-재가중(SAM) 기법을 기반으로 하며, 다양한 노이즈 유형으로 오염된 데이터세트에 적합하게 조정됩니다. 이는 각 샘플에 손상이 가해지는 것을 근사하여 전체 배치에서 처리할 수 있도록 설계되었습니다. 새롭게 제안된 `Corrupted Degradation Disparity` 메트릭을 통해 정확도 저하를 다양한 인구 집단 사이에서 정량화할 수 있습니다.

- **Performance Highlights**: 여러 실제 데이터 세트에 대한 실험을 통해 FairSAM은 공정성과 강건성 모두에서 우수한 성과를 보여주었습니다. 특히, 제안한 공정성 메트릭인 Corrupted Degradation Disparity에서 개선된 점수를 기록하였고, 대부분의 경우 가장 낮은 최악의 그룹 정확도를 달성했습니다. FairSAM은 전통적인 공정성 방법들의 한계를 극복하고 각 인구 집단에 걸쳐 높은 정확도와 공정성을 동시에 유지하는 결과를 보여주었습니다.



### Learning Library Cell Representations in Vector Spac (https://arxiv.org/abs/2503.22900)
- **What's New**: Lib2Vec는 라이브러리 셀의 의미 있는 벡터 표현을 효율적으로 학습할 수 있도록 설계된 새로운 자체 지도 학습 프레임워크입니다. 이 프레임워크는 정규성 시험 자동 생성, Liberty 파일에서의 훈련 데이터 추출, 다양한 핀 수에 적응할 수 있는 주의 기반 모델 아키텍처를 포함합니다. 실험 결과는 Lib2Vec이 기능적 및 전기적 유사성을 효과적으로 캡처하고, 제한된 레이블 데이터 환경에서도 회로 학습 애플리케이션을 개선함을 보여줍니다.

- **Technical Details**: Lib2Vec은 라이브러리 셀의 기능적, 전기적 속성을 모델링하는 데 중점을 둡니다. VLSI 설계에서 셀은 기능적(기능 동작), 전기적(타이밍, 전력 등), 물리적(레이아웃 및 지오메트리) 속성으로 분류됩니다. 또한 Lib2Vec은 입력 조건에 대한 셀의 반응을 기반으로 의미론적 관계를 포착하고, 여러 ML 아키텍처에 호환되는 벡터 공간에서 라이브러리 셀의 표현을 학습하는 최초의 체계적 탐색을 제공합니다.

- **Performance Highlights**: Lib2Vec은 벡터 연산을 통해 'vector(BUF) - vector(INV) + vector(NAND) ~ vector(AND)'와 같은 의미 있는 관계를 보여줍니다. 이 프레임워크는 다양한 핀 수에 적응할 수 있는 유연한 아키텍처를 제공하며, 특정 속성 및 아크에 대한 임베딩을 생성하는 것을 가능하게 합니다. 실험 결과는 Lib2Vec이 기능적 및 전기적 속성을 효과적으로 캡처하고, 다양한 회로 학습 작업에서 활용될 수 있음을 나타냅니다.



### Task Tokens: A Flexible Approach to Adapting Behavior Foundation Models (https://arxiv.org/abs/2503.22886)
- **What's New**: 최신 연구에서는 imitation learning의 발전을 통해 transformer 기반의 behavior foundation models (BFMs)이 등장하여 인체 모사 에이전트에 대한 멀티모달, 인간 유사 제어가 가능해졌습니다. 그러나 특정 작업에 대한 결과를 최적화하려면 상세한 prompt engineering이 필요하여 suboptimal 결과를 초래할 위험이 있습니다. 이에 'Task Tokens'이라는 새로운 방법을 도입하여 BFM을 특정 작업에 효과적으로 맞춤화하면서도 유연성을 유지하는 방안을 제시합니다.

- **Technical Details**: 우리의 접근 방식은 reinforcement learning을 통해 새로운 작업 특화 인코더를 학습하고 원래의 BFM을 고정 상태로 유지하는데, 이를 통해 사용자 정의 priors를 도입할 수 있습니다. 이를 통해 reward design과 prompt engineering의 균형을 맞추어 성능 개선을 이루는 동시에 모델의 다양한 제어 특성을 유지합니다. Task Tokens는 관측치를 토큰에 매핑하는 작업 인코더를 학습하여 BFM의 추가 입력으로 활용합니다.

- **Performance Highlights**: Task Tokens의 효과를 다양한 작업에서 입증하였으며, out-of-distribution 시나리오에서도 그 성능을 보여주었습니다. 이 접근 방식은 BFM의 특정 제어 작업 적응을 가능하게 하면서도 일반화 능력을 유지하는 기대되는 방법으로, 다른 프롬프트 모달리티와도 호환성을 나타냅니다.



### Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models (https://arxiv.org/abs/2503.22879)
- **What's New**: 최근 State Space Models (SSMs)은 메모리 사용의 일관성과 높은 성능 덕분에 Transformers에 대한 매력적인 대안으로 떠오르고 있습니다. 그러나 클라우드 서비스나 자원 제한 장치에서 SSM을 확장하는 데 필요한 저장 요구량과 컴퓨팅 파워가 도전 과제가 되고 있습니다. 이를 해결하기 위해, Quamba2는 다양한 상황에 대한 효율성을 고려하여 W8A8, W4A8 및 W4A16 비트폭을 지원하는 포스트 트레이닝 양자화(PTQ) 프레임워크를 제공합니다.

- **Technical Details**: Quamba2는 SSM의 채널 순서 보존과 활성화 지속성을 기반으로 한 오프라인 양자화 방식을 제안합니다. 입력 데이터의 정렬 및 클러스터링을 통해 8비트 양자화를 처리하며, 상태 그룹별로 양자화를 적용하여 입력 의존 매개변수를 정밀하게 최적화합니다. 이러한 방식은 SSM의 속성을 활용하여 양자화 정확도를 높이고, 메모리 요구 사항을 줄이며, 성능 저하를 최소화합니다.

- **Performance Highlights**: Quamba2-8B는 여러 최신 SSM 양자화 방법들을 초월하여, 예비 채우기 및 생성 단계에서 각각 1.3배 및 3배 빠른 속도를 제공하며, 4배의 메모리 감소도 달성합니다. 평균 1.6%의 정확도 손실로 6개의 제로샷 작업에서 성능을 유지하는 동시에, MMLU 데이터셋에서의 평가를 통해 모델의 일반화 및 내구성을 입증하였습니다.



### Harnessing uncertainty when learning through Equilibrium Propagation in neural networks (https://arxiv.org/abs/2503.22810)
Comments:
          8 pages, 5 figures

- **What's New**: 본 연구에서는 Equilibrium Propagation (EP) 알고리즘을 사용하여 물리적 불확실성이 있는 하드웨어에서 학습할 수 있는 능력을 평가합니다. EP는 데이터 이동을 최소화하여 에너지 효율을 높이는 특징이 있어, 특히 신경형 시스템에서의 학습 프레임워크로 유망합니다. 연구 결과는 깊은 다층 신경망 구조가 유한한 불확실성 하에서도 성공적으로 학습할 수 있음을 보여줍니다.

- **Technical Details**: EP는 에너지 기반 모델 (Energy-Based Models, EBMs)에 대한 감독 학습을 수행하는 프레임워크로, 생성한 노드 값과 매개변수 조정의 최적화를 통해 학습합니다. 본 연구에서는 비선형 저항 네트워크 구조 내에서 EP의 확률적 프레임워크를 적용하여 매개변수 업데이트에 대한 측정 불확실성을 근사합니다. 포스트-활성화 노드에서의 노이즈를 추가하여 불확실성을 모의하고, EP가 조정한 후의 활동 차이를 비교하는 과정을 설명합니다.

- **Performance Highlights**: 테스트 결과, MNIST, KMNIST, FashionMNIST 데이터셋에서 유한한 수준의 불확실성을 가진 모델이 개선된 수렴성과 성능을 보여줍니다. 최적의 성능은 비판적 한계에 가까운 불확실성으로 훈련된 네트워크에서 나타났습니다. 본 연구는 EP를 이용한 자가 학습 하드웨어 구축에 대한 향후 연구의 기초 자료를 제공합니다.



### Data-driven worker activity recognition and picking efficiency estimation in manual strawberry harvesting (https://arxiv.org/abs/2503.22809)
- **What's New**: 이 연구에서는 상업적인 딸기 수확에서 수확자 효율성을 계산하기 위한 실제 시스템이 개발되었습니다. 수확한 과일의 무게, 지리적 위치, 카트 이동을 실시간으로 기록하기 위해 인스트루멘티드 피킹 카트가 사용되었습니다. CNN-LSTM 기반의 심층 신경망을 통해 수확자의 활동을 'Picking'과 'NoPicking'으로 분류하여 수확자 효율성을 추정할 수 있었습니다.

- **Technical Details**: 이 연구에서 사용된 인스트루멘티드 피킹 카트(iCarritos)는 두 개의 로드셀과 GNSS 수신기, 관성 측정 장치(IMU)를 갖추고 있어 수확된 과일의 무게, 지리적 위치 및 카트의 이동 데이터를 기록합니다. CNN-LSTM 기반의 알고리즘이 개발되어 수확 데이터를 분석하고, 수확자 효율성을 평가하기 위해 활동 인식이 이루어졌습니다. 이러한 시스템은 실제 수확 기간 동안 발생하는 다양한 비생산적인 작업들을 감지하고 영향을 줄 수 있도록 설계되었습니다.

- **Performance Highlights**: 실험 결과, CNN-LSTM 모델은 평균 F1 점수 0.974의 정확도로 활동 인식을 수행했습니다. 수확 데이터 분석을 통해 수확자는 평균 73.56%의 시간을 적극적으로 딸기를 수확하며, 트레이를 채우는 데 평균 6.22분의 시간이 소요되었습니다. 제안된 기술은 상업 규모에서 자동화된 작업 모니터링과 수확 최적화에 기여할 수 있습니다.



### Invariant Control Strategies for Active Flow Control using Graph Neural Networks (https://arxiv.org/abs/2503.22775)
- **What's New**: 최근 강화 학습(Reinforcement Learning, RL)이 능동적 유동 제어(active flow control, AFC) 작업에서 주목받고 있습니다. 초기 연구들은 이차원 실린더 주위의 유동 장(field)을 확장하여 항력을 완화하는 방법을 탐구했습니다. 연구자들은 이제 더 복잡한 난류 유동을 제어하는 데 RL을 적용하고 있으며, 이러한 방법의 효용성에도 불구하고 계산 비용이 크고 일반화 능력이 부족한 문제가 제기되고 있습니다.

- **Technical Details**: 연구팀은 이러한 문제를 해결하기 위해 그래프 신경망(Graph Neural Networks, GNNs)을 도입했습니다. GNN은 비구조적인 3차원 유동 데이터를 자연스럽게 처리하며, 카르테시안 그리드의 제약 없이 공간적 관계를 유지합니다. 또한, GNN은 회전, 반사 및 치환 불변성을 네트워크 아키텍처에 통합하여 제어 정책의 일반화를 향상시키고 있습니다.

- **Performance Highlights**: 실험은 Relexi라는 고성능 RL 프레임워크를 사용하여 이차원 실린더 벤치마크 문제를 다시 분석했습니다. GNN 기반의 제어 정책이 기존 방법들과 유사한 성능을 보여주고, 개선된 일반화 속성을 갖는 것을 입증했습니다. 본 연구에서는 GNN이 RL 기반 유동 제어에 적합한 아키텍처로 자리잡을 수 있음을 보여주었습니다.



### GroundHog: Revolutionizing GLDAS Groundwater Storage Downscaling for Enhanced Recharge Estimation in Bangladesh (https://arxiv.org/abs/2503.22771)
- **What's New**: 이번 연구는 GLDAS의 저해상도 데이터(25 km)를 활용하여 고해상도(2 km) 지하수 수위(GWL)를 예측하는 GroundHog 모델을 개발하였습니다. 이 모델은 다년간의 데이터를 기반으로 최대 및 최소 GWL을 예측할 수 있도록 설계되어, 효과적인 수자원 관리와 정책 결정에 기여합니다. 또한, ML 모델을 활용하여 데이터의 공백을 줄이고, 새로운 'Pseudo-Ground Truth' 데이터셋을 생성하여 다양한 지점의 GWL을 예측할 수 있게 되었습니다.

- **Technical Details**: GroundHog 모델은 다수의 수리-지질학적 요소(Hydro-geological Factors, HGFs)를 고려하여 GWL 예측에 필요한 입력 데이터를 수집합니다. 최대 및 최소 GWL 예측을 위한 모델은 Random Forest Regressor를 사용하며, 2001년부터 2022년까지의 실제 수치 데이터를 ground truth로 사용합니다. 이 연구에서는 Normalized Difference Water Index (NDWI)와 Normalized Difference Vegetation Index (NDVI) 등의 다양한 HGFs가 활용되며, 이들 요소는 수위 예측의 정확도를 높이는 데 기여합니다.

- **Performance Highlights**: 제안된 모델은 최대 GWL과 최소 GWL의 예측에서 각각 R^2 점수 0.855 및 0.963을 달성하였습니다. 고해상도 GWL을 생성하는 Upsampling 모델은 GLDAS의 입력을 기반으로 R^2 점수 0.96이라는 뛰어난 성과를 거두었습니다. 연구 결과는 2003-2024년 기간 동안의 GLDAS 데이터를 고해상도로 보강하여 지하수 재충전 추정을 가능케 하고, 자원 관리를 위한 중요한 트렌드를 제시합니다.



### The Cost of Local and Global Fairness in Federated Learning (https://arxiv.org/abs/2503.22762)
- **What's New**: 본 논문은 다중 클래스 연합 학습(Federated Learning; FL) 설정에서 글로벌(global) 및 로컬(local) 공정성을 고려하여 최소한의 정확도가 얼마나 손실되는지를 조사하는 프레임워크를 제안합니다. 기존의 연구들은 일반적으로 글로벌 공정성(예: 배제된 편향) 또는 로컬 공정성(예: 각 클라이언트 내에서의 형평성) 중 하나만 다루어 왔습니다. 새로운 프레임워크는 베이지안 최적 점수 함수에서 공정한 결과 예측기를 도출하는 간단한 후처리(post-processing) 알고리즘을 제공합니다.

- **Technical Details**: 우리의 제안된 프레임워크는 문제를 볼록 프로그램(convex program)으로 서술하며, 효율적인 해결을 위해 수신자 동작 특성(Receiver Operating Characteristic; ROC) 표면을 사용하여 근사합니다. 이 벨록 프로그램을 해결하는 데 복잡성이 높기 때문에, 단순형(simplex)을 사용하여 문제를 선형 프로그램(linear program; LP)으로 재구성합니다. 이 과정에서 우리의 알고리즘은 일반적으로 사용되는 공정성 지표(예: Statistical Parity (SP), Equal Opportunity (EOp), Equalized Odds (EO))를 지원합니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 현재 상용 여성 예측 분야(State of the Art; SOTA) 대비 정확도-공정성(tradeoff)에서 우수한 성능을 발휘하며, 통신 비용을 20% 절감하고 계산 비용을 33% 줄였습니다. 이로써 다중 클래스 환경에서도 효율적으로 공정성을 유지할 수 있는 가능성을 보여줍니다.



### Model Lake: a New Alternative for Machine Learning Models Management and Governanc (https://arxiv.org/abs/2503.22754)
- **What's New**: 이 논문에서는 머신러닝(ML) 모델 관리를 위한 중앙 집중식 프레임워크인 '모델 레이크(Model Lake)' 개념이 제시되었습니다. 기존의 모델 관리 방식은 여러 저장 시스템을 이용하며 비효율적이고 표준화된 방법론의 부족으로 어려움을 겪어왔습니다. 이러한 배경에서, 데이터 레이크의 아이디어를 차용하여, 조직 내 데이터셋, 코드 및 모델을 통합하여 관리할 수 있는 새로운 접근 방식이 필요해졌습니다.

- **Technical Details**: 모델 레이크는 ML 모델의 전체 수명주기를 관리하는 통합 플랫폼을 제공합니다. 모델 레이크는 'Github'와 유사하게 ML 모델을 카탈로그화하고 주석을 달아 조직 내 이해관계자들 간에 쉽게 접근할 수 있도록 만들어집니다. 또한, 데이터 엔지니어와 데이터 과학자들이 실시간으로 협업하고 실험할 수 있는 동적 작업 공간을 제공합니다. 이를 통해 모델 재사용성이 향상되고 모델 성능 모니터링이 용이해집니다.

- **Performance Highlights**: 모델 레이크 접근 방식을 채택함으로써 조직들은 모델 생애 주기 관리, 모델 탐색, 감사 및 재사용성을 높일 수 있습니다. 실제 사례를 통해 데이터, 코드 및 모델 관리의 변혁적 영향을 입증하고, 이로 인해 운영 효율성이 증대하며 거버넌스가 향상되는 것을 보여줍니다. 조직 내 데이터 분석 파이프라인의 모든 요소를 효과적으로 통합함으로써, AI 솔루션의 배포 속도가 빨라질 것으로 기대됩니다.



### Combating the Bullwhip Effect in Rival Online Food Delivery Platforms Using Deep Learning (https://arxiv.org/abs/2503.22753)
- **What's New**: 최근 온라인 음식 배달 서비스의 발전으로 인해 부패하기 쉬운 식품의 낭비가 심각한 문제로 대두되고 있습니다. 이 논문은 레스토랑, 온라인 음식 앱 및 고객 간의 제3자 물류 모델을 활용하고 있으며, 두 개의 단계로 구성된 LSTM(Long Short-Term Memory) 네트워크를 기반으로 한 수요 예측 모델을 제안합니다. 이 모델은 인트라데이와 일일 수요 예측을 통해 효율적인 공급망 관리를 목표로 합니다.

- **Technical Details**: 이 연구에서는 2023년 1월부터 2025년 1월까지 Swiggy와 Zomato의 데이터를 활용하여 LSTM 모델을 최적화합니다. 인트라데이 예측에서는 단기 변동을 포착하고, 일일 예측에서는 전체 수요를 예측합니다. Grid Search 기법을 통해 LSTM의 최적 하이퍼파라미터를 탐색하며, RMSE, MAE, R-squared와 같은 지표를 사용해 모델 성능을 평가합니다.

- **Performance Highlights**: 첫 번째 단계에서 Zomato는 0.69, Swiggy는 0.71의 R-squared 점수를 기록했습니다. 두 번째 단계에서는 Zomato 0.88, Swiggy 0.90으로 향상되었으며, 공급망의 불안정성이 각각 2.61에서 0.96로, 2.19에서 0.80으로 감소했습니다. 이는 제안된 모델이 식품 낭비를 줄이고 레스토랑 재고 수준을 최적화하는 데 효과적임을 보여줍니다.



### From Individual to Group: Developing a Context-Aware Multi-Criteria Group Recommender System (https://arxiv.org/abs/2503.22752)
Comments:
          The 16th International Conference on Management of Digital EcoSystems, Nov 2024, Naples, Italy

- **What's New**: 이 연구에서는 다양한 개인의 선호를 고려해야 하는 그룹 의사결정 상황에서의 문제를 해결하기 위해 Context-Aware Multi-Criteria Group Recommender System (CA-MCGRS)을 개발하였습니다. 기존의 추천 시스템은 개별화에 효과적이지만 그룹 의사결정에서는 갈등하는 선호와 다양한 평가 기준을 다루는 데 한계가 있습니다. CA-MCGRS는 이러한 맥락적 요소와 다중 기준을 통합하여 추천의 정확도를 높이는 데 초점을 맞추었습니다.

- **Technical Details**: CA-MCGRS는 Multi-Head Attention 메커니즘을 활용하여 서로 다른 특징의 중요성을 동적으로 조정합니다. 이 모델은 교육 데이터셋에서 다양한 평가와 맥락 변수를 고려하여 실험을 진행하였습니다. 이러한 접근 방식은 추천 정확도를 향상시키는데 기여하는 것으로 나타났습니다.

- **Performance Highlights**: 실험 결과, CA-MCGRS는 네 가지 시나리오 모두에서 다른 접근 방식들보다 지속적으로 우수한 성과를 보였습니다. 따라서 본 연구의 발견은 그룹 추천 시스템의 개발에 있어 맥락과 다중 기준 평가를 포함하는 것이 얼마나 중요한지를 강조합니다.



### Advancing Spatiotemporal Prediction using Artificial Intelligence: Extending the Framework of Geographically and Temporally Weighted Neural Network (GTWNN) for Differing Geographical and Temporal Contexts (https://arxiv.org/abs/2503.22751)
- **What's New**: 이번 논문은 인공지능 신경망(Artificial Neural Networks, ANNs)의 수학적 프레임워크를 개선하여 예측 범죄 모델을 향상시키는 것을 목표로 합니다. 특히, 지리적이고 시간적인 가중치를 고려한 회귀(Geographically and Temporally Weighted Regression, GTWR) 문제의 해결을 위한 새로운 반-해석적 접근 방식을 제안하고 런던 범죄 데이터에 적용하여 높은 정확도를 입증합니다. 이 논문은 GTWNN(Geographically and Temporally Weighted Neural Network) 프레임워크에 대한 수학적 발전을 소개하며, 범죄 예측 문제에 신경망의 적절한 적용을 위한 다양한 수학적 확장을 제안합니다.

- **Technical Details**: GTWNN 모델은 Fotheringham et al. (2015)에 의해 소개된 GTWR의 확장으로, 외부 요인의 정보를 비선형적으로 결합할 수 있는 장점을 제공합니다. 그러나 본 연구는 GTWNN 모델에서 외부 요인에 연결된 계수 함수의 연속성을 결여하고 역사적 맥락 정보를 활용하지 못하는 한계를 지적합니다. 이를 극복하기 위해, 세 가지 수학적 확장을 통해 범죄 데이터에 대한 적합성을 높이고자 하였습니다.

- **Performance Highlights**: 논문에서는 런던 및 디트로이트 데이터 세트에 대해 다섯 가지의 새로운 ANN 모델을 적용하여 평가하였습니다. 각 모델의 성능을 비교한 결과, 역사 의존 모듈(history-dependent module)이라는 특정 확장모듈이 다른 모듈들을 일반적으로 초과하는 것으로 나타났습니다. 따라서, 제안된 방법들은 더욱 맥락지향적이고 정확한 ANN 접근 방식의 기초를 제공하여, 범죄 예측 모델링의 적합성을 향상시키는 데 기여할 것으로 보입니다.



### Adaptive Clipping for Privacy-Preserving Few-Shot Learning: Enhancing Generalization with Limited Data (https://arxiv.org/abs/2503.22749)
- **What's New**: 본 논문에서는 데이터 Privacy(프라이버시)와 모델 성능 간의 기본적인 트레이드오프를 해결하기 위해 Meta-Clip이라는 새로운 알고리즘을 제안합니다. 이는 Differentially Private (DP) 메타-러닝 알고리즘의 성능을 최대화하며 프라이버시 보호를 동시에 달성하는 것을 목표로 하며, Mini-Batch training(미니배치 학습) 중에 클리핑 조정 방법을 동적으로 적용하여 성능 향상을 꾀합니다. 또한, 기존의 프라이버시 기술들과 비교하여 우수한 프라이버시-유틸리티 균형을 보여깁니다.

- **Technical Details**: Meta-Clip은 DP 모델에 대한 다양한 메타-러닝 알고리즘, 즉 DP-Reptile과 DP-MetaSGD에 통합되어 있으며, 민감한 정보의 노출을 조절할 수 있도록 Adaptive Clipping(어댑티브 클리핑) 방법을 사용합니다. 훈련 과정 중 클리핑 임계치를 조정함으로써 Finely-tuned control(세부 조정)을 가능하게 하여 overfitting(과적합)을 줄이고, 메타-러닝 모델의 일반화 성능을 크게 향상시킵니다. 저자들은 실험을 통해 방법론이 처리 가능하다는 것을 입증하였습니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋에서 수행된 실험을 통해, 메타-러닝 알고리즘의 성능이 향상되었으며, 특히 몇 개의 라벨 데이터만으로도 효과적인 학습이 가능함을 입증했습니다. 제안한 방법론은 기존의 프라이버시 보장 기법들보다 뛰어난 개인정보 보호와 유틸리티 보장을 자랑하며, 실제 어플리케이션에서 데이터가 제한된 환경에서도 적용 가능함을 보여줍니다. Adaptive Clipping 방법은 few-shot learning(소수 샷 학습) 문제에서 안전하고 정확한 모델 개발에 큰 기여를 할 것으로 기대됩니다.



### Ignite Forecasting with SPARK: An Efficient Generative Framework for Refining LLMs in Temporal Knowledge Graph Forecasting (https://arxiv.org/abs/2503.22748)
Comments:
          To be published in the 30th International Conference on Database Systems for Advanced Applications (DASFAA 2025)

- **What's New**: 이 논문에서는 Temporal Knowledge Graph (TKG) 예측에 대한 새로운 접근 방식으로 SPARK를 소개합니다. SPARK는 Seqeunce-level Proxy-Adapting 프레임워크로, LLMs를 활용한 TKG 예측의 문제를 해결합니다. 기존 LLM 기반 방법의 제한사항인 입력 길이, 출력 생성의 비효율성 및 리소스 집약적 보완 문제를 해결하려고 합니다.

- **Technical Details**: SPARK는 두 가지 주요 혁신을 통해 TKG 예측을 향상시킵니다. 첫째, Beam Sequence-Level Generation(BSL) 접근을 통해 TKG 예측을 top-K 시퀀스 생성 작업으로 전환하여, 단일 전파에서 효율적으로 다음 엔티티 분포를 생성합니다. 둘째, TKG Adapter를 통해 전통적인 TKG 모델을 훈련 가능한 프록시 어댑터로 활용하며, 이를 통해 LLM 출력의 보완을 수행하고 입력 길이 문제와 비용 및 시간이 많이 드는 파인튜닝을 극복합니다.

- **Performance Highlights**: 다양한 데이터셋을 활용한 실험을 통해 SPARK의 예측 성능과 일반화 능력 및 높은 효율성을 검증하였습니다. 실험 결과 SPARK는 기존 LLM 성능을 일관성 있게 향상시키며, IT 튜닝 모델과 비교해 경쟁력 있는 성능을 달성한 것으로 나타났습니다.



### LeForecast: Enterprise Hybrid Forecast by Time Series Intelligenc (https://arxiv.org/abs/2503.22747)
- **What's New**: 이 논문에서는 멀티디스플리너리 예측(multidisciplinary forecasting) 분야에서의 수요 증가에 대응하기 위해 leforecast{}라는 기업 인텔리전스 플랫폼을 소개합니다. 이 플랫폼은 시간 시계열 예측(time series forecasting) 작업에 최적화 되어 있으며, 역사적 데이터를 기반으로 인사이트를 도출하고 미래를 예측하는데 초점을 맞추고 있습니다. 또한, 대형 기초 모델(large foundation model)의 활용을 통해 복잡한 비즈니스 맥락을 해석하는 새로운 접근 방식을 제시합니다.

- **Technical Details**: LeForecast는 시간 시계열 데이터와 다원 소스 정보(multi-source information)의 고급 해석을 결합한 세 가지 기둥 모델링 엔진(three-pillar modelling engine)을 통합하고 있습니다. 이 엔진은 대형 기초 모델(Le-TSFM), 다중 모달(multimodal) 모델 및 하이브리드 모델(hybrid model)로 구성되어 있어 시나리오 기반 예측을 가능하게 합니다. 또한, 모델 풀(model pool), 모델 프로파일링 모듈 및 원래 모델 아키텍처에 대한 두 가지 다른 융합 접근 방식을 포함하여, 모델 효율성을 극대화합니다.

- **Performance Highlights**: 실험 결과는 router-based fusion network와 대형 및 소형 모델의 조정을 통한 효율성을 입증하며, 이는 모델 개발 및 유지 관리의 중복 비용을 절감하는 데 유용합니다. 세 가지 산업 사용 사례에서 LeForecast의 배치를 리뷰한 결과, 이 플랫폼은 효율적이고 경쟁력 있는 성능을 제공하는 것으로 나타났습니다. 이 연구는 시간 시계열 기술의 연구와 기업 가속화에 기여할 수 있을 것으로 기대됩니다.



### Graph-Based Uncertainty-Aware Self-Training with Stochastic Node Labeling (https://arxiv.org/abs/2503.22745)
- **What's New**: 새로운 연구에서는 그래프 구조를 활용한 불확실성 인식 자기 학습 프레임워크인 GUST를 제안합니다. 이 방법은 노드 분류 문제에서 높은 확신(over-confidence)을 해결해주는 데 집중하고 있습니다. 특히, GUST는 노드 레이블의 불확실성을 모델링하기 위해 베이지안(Bayesian) 영감을 얻은 모듈을 활용하며, EM(Expectation-Maximization) 절차를 통해 개선된 대표 레이블을 생성합니다.

- **Technical Details**: GUST 프레임워크의 핵심 요소는 베이지안 불확실성 추정, EM을 통한 확률적 노드 레이블링 및 반복적인 학습 과정입니다. 각 노드는 잠재적 임베딩 분포를 통해 노드 레벨의 불확실성을 캡처하며, EM 단계는 이러한 불확실성을 활용하여 레이블을 정제합니다. 이는 전통적인 자기 학습 전략과 차별화된 접근법으로, 그래프 기반 구조를 채택하여 노드의 불확실성을 효과적으로 관리합니다.

- **Performance Highlights**: 여러 벤치마크 그래프 데이터셋에 대한 실험 결과, GUST는 특히 레이블 데이터가 극히 드문 상황에서 최첨단 성능을 보였습니다. 기존의 연구들보다 비약적인 성능 향상을 이루어 냈으며, 노이즈가 존재하는 상황에서도 안정적인 레이블 생성을 보여주고 있습니다. 이를 통해 GUST는 저 레이블 환경에서도 신뢰할 수 있는 결과를 제공하는데 성공했습니다.



### Uncertainty-Aware Graph Self-Training with Expectation-Maximization Regularization (https://arxiv.org/abs/2503.22744)
- **What's New**: 이번 연구에서는 세미 슈퍼바이즈드 노드 분류를 위한 새로운 uncertainty-aware graph self-training 접근 방식을 제안합니다. 이 방법은 pseudo-label 생성을 위해 불확실성 메커니즘을 도입하며, Expectation-Maximization (EM) 정규화 스킴을 활용합니다. 기존의 고정된 pseudo-label에 의존하는 그래프 자가 훈련 방식과는 달리, 본 연구에서는 EM-inspired 불확실성 측정을 통해 레이블 신뢰도를 반복적으로 정제합니다.

- **Technical Details**: 우리의 Uncertainty-Aware Graph Self-Training (UGST) 방법은 두 가지 주요 구성 요소로 구성됩니다: (1) EM 기반 불확실성 모델링과 (2) 적응형 pseudo-label 정제입니다. EM 절차를 통해 불확실한 노드의 본질적인 불확실성을 추정하고, 이를 통해 훈련 과정을 안내하는 불확실한 소프트 할당으로 노드 표현을 변환합니다. 그런 다음, 불확실성 추정을 사용하여 pseudo-label을 생성하며, 이는 불확실한 노드를 보수적으로 처리하도록 조정됩니다.

- **Performance Highlights**: 여러 기준 그래프 데이터셋에 대한 광범위한 실험을 통해 우리의 방법이 강력한 기준선과 비교하여 최대 2.5%의 정확도 상승을 보여주었습니다. 또한, 다양한 반복에서 성능의 변동성이 낮아져 모델의 안정성이 크게 향상되었습니다. 이러한 높은 정확도와 낮은 분산을 통해 본 연구의 접근 방식은 향후 그래프 기반 학습 연구에 기여할 것으로 기대됩니다.



### Adaptive State-Space Mamba for Real-Time Sensor Data Anomaly Detection (https://arxiv.org/abs/2503.22743)
- **What's New**: 본 연구에서는 실시간 센서 데이터의 이상 탐지를 위한 새로운 프레임워크인 Adaptive State-Space Mamba (ASSM)을 제안합니다. 기존의 상태 공간 모델을 이미지 처리에 활용한 사례를 기반으로, ASSM은 스트리밍 센서 데이터에서 이상을 탐지하는 데 초점을 맞추고 있습니다. 특히, 상황에 따라 상태 업데이트를 조정하는 적응형 게이팅 메커니즘을 도입하여 모델의 효율성과 스케일러블함을 확보했습니다.

- **Technical Details**: ASSM 모델은 고속 스트리밍 센서 데이터에서의 이상 탐지를 위해 설계되었습니다. 이 모델은 은닉 상태 전이의 개념을 시계열 도메인으로 확장하고, 문맥 신호에 기초하여 강인한 이상 탐지를 위한 적응형 게이팅 메커니즘을 도입합니다. gated 변수가 계량하여 이전 상태와 현재 입력 간의 정보 흐름을 조절함으로써, 이상 징후를 더욱 효과적으로 탐지할 수 있습니다.

- **Performance Highlights**: ASSM의 성능은 실제 및 합성 센서 데이터셋에 대한 철저한 실험을 통해 입증되었습니다. 기존의 기준들과 비교했을 때, ASSM은 뛰어난 탐지 성능과 계산 효율성을 달성했습니다. 이 방법은 빠르고 신뢰할 수 있는 탐지가 요구되는 다른 시계열 작업으로 쉽게 확장될 수 있는 장점을 가집니다.



### Adaptive Integrated Layered Attention (AILA) (https://arxiv.org/abs/2503.22742)
- **What's New**: 이번 연구에서는 Adaptive Integrated Layered Attention (AILA)라는 신경망 아키텍처를 제안합니다. AILA는 다양한 네트워크 층 간의 적응형(feature reuse) 기능을 위해 밀집 스킵 연결(dense skip connections)과 여러 메커니즘을 융합하여 구성되어 있습니다. AILA는 가격 예측, 이미지 인식, 감정 분석의 세 가지 도전 과제를 평가받았으며, 기존의 강력한 딥러닝 모델과 유사한 성능을 보이면서도 훈련 및 추론 시간을 크게 단축시켰습니다.

- **Technical Details**: AILA는 두 가지 아키텍처, 즉 AILA-Architecture 1과 AILA-Architecture 2로 나뉘어 있습니다. AILA-Architecture 1은 층 간의 연결 메커니즘으로 간단한 선형 층(linear layers)을 사용하고, AILA-Architecture 2는_attention_ 메커니즘을 구현하여 이전 층의 출력을 선택적으로 강조합니다. 이러한 아키텍처는 각기 다른 태스크에 대해 개별적으로 훈련되며, 다양한 네트워크 깊이에서 관련 기능을 유연하게 재사용함으로써, 강력한 성능 향상을 이루어냅니다.

- **Performance Highlights**: AILA는 세 가지 기준 벤치마크에서 강력한 성능 지표를 달성했습니다. 가격 예측, CIFAR-10 데이터셋에 대한 이미지 인식, IMDB 영화 리뷰 데이터셋의 감정 분석에서 AILA-Architecture 1 및 2 모두 LSTM, Transformer, CNN과 같은 기존의 강력한 기준 모델과 경쟁하며 이를 초월하는 성과를 보여주었습니다. 결과적으로 AILA는 일반적인 고정 연결 방식이 아닌, 적응형 정보 흐름을 통해 복잡한 태스크에서 성능을 향상시키는 새로운 길을 열었습니다.



### CSPO: Cross-Market Synergistic Stock Price Movement Forecasting with Pseudo-volatility Optimization (https://arxiv.org/abs/2503.22740)
- **What's New**: 이번 연구에서는 Cross-market Synergy with Pseudo-volatility Optimization (CSPO) 프레임워크를 소개합니다. CSPO는 외부 선물 시장의 지식을 활용한 깊은 신경망 아키텍처를 구현하여, 주식 가격 예측 능력을 향상시킵니다. 기존의 예측 모델에 비해 더 정교한 기능을 제공하며, 모델의 최적화 과정에서 주식별 변동성을 고려합니다.

- **Technical Details**: CSPO는 Bi-level Dense Pricing Transformer (BDP-Former)라는 변환 기반 딥 뉴럴 아키텍처를 활용하여 선물 시장 정보를 통합합니다. 이를 통해 주식의 복잡한 상호작용을 포착하고, 예측 과정에서의 가격 안정성을 정량화하기 위해 의사 변동성(pseudo-volatility) 개념을 도입합니다. 다양한 주식의 변동성을 반영함으로써, CSPO는 보다 정확하고 견고한 예측 결과를 제공합니다.

- **Performance Highlights**: 실제 주식 시장 데이터셋을 사용한 광범위한 실험 결과, CSPO는 기존 모델에 비해 우수한 성능을 보였습니다. 특히, CSI300 지수를 포함한 다양한 평가 지표에서 더 나은 수익 곡선을 달성하였습니다. 이러한 결과는 CSPO 프레임워크와 그 구성 요소들이 효과적임을 입증합니다.



### ShieldAgent: Shielding Agents via Verifiable Safety Policy Reasoning (https://arxiv.org/abs/2503.22738)
- **What's New**: 이 논문에서는 ShieldAgent라는 최초의 LLM 기반의 가드레일 에이전트를 소개합니다. ShieldAgent는 다른 LLM 기반 자율 에이전트의 행동 궤적을 보호하고, 명확한 안전 정책 준수를 보장하기 위해 고안되었습니다. 기존의 기술이 단순한 텍스트 기반 필터링에 의존하는 반면, ShieldAgent는 에이전트의 행동의 고유성을 고려하여 관련 정책과의 검증을 효율적으로 수행합니다.

- **Technical Details**: ShieldAgent는 정책 문서에서 검증 가능한 규칙을 추출하여 이를 행동 기반 확률적 규칙 회로로 구조화하는 안전 정책 모델을 자동으로 구축합니다. 이 에이전트는 동적 환경과의 순차적 상호작용을 통해 변화하는 불안전한 행동을 포착하기 어렵게 만드는 두 가지 주요 과제를 해결하는 데 중점을 둡니다. 또한, ShieldAgent-Bench라는 3천 개의 안전 관련 에이전트 명령과 행동 궤적 쌍으로 구성된 데이터셋을 소개하여, 다양한 웹 환경에서 수집한 데이터를 기반으로 합니다.

- **Performance Highlights**: 실험 결과, ShieldAgent는 ShieldAgent-Bench 및 기존의 세 가지 벤치마크에서 SOTA 성능을 기록하며 평균적으로 이전 방법보다 11.3% 향상된 결과를 보여줍니다. 또한, API 쿼리를 64.7% 줄이고 추론 시간을 58.2% 단축하여 에이전트의 안전을 효과적으로 보장하면서도 높은 정밀도와 효율성을 입증합니다.



### Cyborg Data: Merging Human with AI Generated Training Data (https://arxiv.org/abs/2503.22736)
- **What's New**: 이 논문은 대규모 평가에서 사용되는 자동 채점 시스템(Automated Scoring, AS)의 새로운 접근 방식을 제안합니다. 기존 AS 시스템은 방대한 수의 수작업 점수가 필요했으며, 이는 시간과 비용이 많이 소모됩니다. 하지만 저자들은 큰 Generative Language Model이 소량의 데이터로도 새로운 작업에 일반화하는 능력을 갖춘 점을 바탕으로, ‘Teacher’ 모델이 ‘Student’ 모델을 가르치는 모델 증류 파이프라인을 제안합니다.

- **Technical Details**: 이 연구는 ‘Cyborg Data’라 불리는 데이터셋을 생성하는 방법을 설명합니다. Teacher 모델은 소규모 수작업 점수 데이터로 학습하고, 이를 기반으로 추가 점수를 생성하여 Student 모델을 훈련시킵니다. 이 과정은 수작업 데이터를 단 10%만 사용하더라도 Student 모델이 전체 데이터셋에서 교육된 모델과 유사한 성능을 보일 수 있음을 보여줍니다.

- **Performance Highlights**: 연구 결과, Student 모델이 Cyborg Data에서 훈련될 경우, 예전 수작업 점수로 전부 훈련된 모델과 유사한 성능을 나타내는 것으로 나타났습니다. 이는 제안된 방법이 AS 개발 비용을 크게 줄일 수 있는 가능성을 제시하며, 향후 연구 방향에 대한 논의도 포함되어 있습니다.



### A Methodology to extract Geo-Referenced Standard Routes from AIS Data (https://arxiv.org/abs/2503.22734)
- **What's New**: 본 연구는 AIS(Automatic Identification Systems) 데이터를 활용하여 해양 관심 지점 간의 표준 경로를 추출하는 방법론을 제안합니다. 이 방법론은 선박 행동의 일관성에 기반하여, 기후, 환경 또는 경제적 요인에 따른 패턴을 분석합니다. 특히, AIS 데이터 분할을 위해 유한상태기계(Finite State Machine, FSM)를 사용하고, 비지도 학습(Unsupervised Learning) 방식을 적용하여 표준 경로를 정의하는 새로운 개념을 제공합니다.

- **Technical Details**: 제안된 방법론은 해양 관심지점(POIs) 간의 경로를 분리하기 위해 AIS 데이터를 세분화하는 자동화된 단계별 접근 방식을 사용합니다. 이 과정에서는 출발 및 도착 POIs를 기준으로 세그먼트를 그룹화한 후, 각 그룹에 대해 반복 밀도 기반 군집화(Iterative Density-Based Clustering)를 적용하여 경로를 추출합니다. 이 방법은 기존 문헌에서 제안된 방법과 달리 사전 지식 없이도 작동하며, 역사적인 데이터 분석을 가능하게 합니다.

- **Performance Highlights**: 이 연구의 방법론은 Arctic 및 유럽, 중동, 북아프리카를 포함하는 1.15TB의 6년간 AIS 데이터에서 성공적으로 검증되었으며, 표준 경로를 추출하면서 5% 미만의 이상치를 나타냈습니다. 이러한 결과는 해양 관제 당국에게 효과적으로 네비게이션 행동을 분석하고 선박 경로 패턴을 분류하는 데 기여할 수 있습니다. 이 연구는 또한 선박 위치 예측을 개선하고, 구조 작업의 효율성을 높이는 데도 도움을 줄 수 있는 잠재력을 가지고 있습니다.



### RBFleX-NAS: Training-Free Neural Architecture Search Using Radial Basis Function Kernel and Hyperparameter Detection (https://arxiv.org/abs/2503.22733)
Comments:
          15 pages, 17 figures, IEEE Transactions on Neural Networks and Learning Systems

- **What's New**: 이번 논문은 RBFleX-NAS라는 새로운 훈련 없는(Training-Free) Neural Architecture Search (NAS) 프레임워크를 제안하며, 이는 활성화 함수와 마지막 레이어의 입력 피쳐(Feature) 지도도 고려하여 네트워크의 효율성을 평가합니다. 기존의 상태-of-the-art 훈련 없는 NAS 알고리즘들은 성능 예측의 정확도가 떨어지는 문제를 가지고 있었으나, RBFleX-NAS는 이러한 문제를 해결합니다. 또한, 새로운 활성화 설계 공간인 NAFBee도 제안하여 다양한 활성화 유형을 포함합니다.

- **Technical Details**: RBFleX-NAS는 Radial Basis Function (RBF) 커널을 활용하여 다양한 입력 이미지 간의 활성화 출력과 마지막 레이어의 입력 피쳐 경량 측정을 통해 네트워크 성능을 평가합니다. 하이퍼파라미터 탐지 알고리즘(Detection Algorithm)을 통해 RBF 커널을 지원하는 최적의 하이퍼파라미터를 탐색하며, 이를 통해 성능 향상이 도모됩니다. 이런 기술적 접근은 다양한 NAS 벤치마크에서 RBFleX-NAS의 효율성을 실험적으로 검증하는 데 중점을 두었습니다.

- **Performance Highlights**: RBFleX-NAS는 NAS-Bench-201 및 NAS-Bench-SSS와 같은 여러 NAS 벤치마크에서 탁월한 성능을 보이며, 기존 훈련 없는 NAS 방법들보다 더 높은 top-1 정확도를 달성하고, 빠른 검색 시간을 자랑합니다. 또한, 이전의 NAS 알고리즘들과 비교해 Kendall 상관관계가 높아 보다 정확한 예측이 가능합니다. 활성화 함수 탐색에 있어서도 RBFleX-NAS는 다른 NAS 알고리즘들보다 더 뛰어난 성능을 보입니다.



### Reasoning Beyond Limits: Advances and Open Problems for LLMs (https://arxiv.org/abs/2503.22732)
Comments:
          41 pages

- **What's New**: 본 논문은 최근의 생성적 추론(Generative Reasoning) 혁신이 대형 언어 모델(LLMs)의 복잡한 문제 해결 방식을 변화시킨 내용을 다룹니다. 예를 들어, DeepSeek-R1, OpenAI의 o1 & o3, GPT-4o 모델과 같은 것을 포함하여, 2023-2025년 사이에 발표된 상위 27개의 LLM 모델을 종합적으로 분석하였습니다. 논문에서는 일반 훈련 접근법, 믹스처 오브 엑스퍼트(Mixture-of-Experts, MoE), 정보 검색 증강 생성(Retrieval-Augmented Generation, RAG) 등을 포함한 다양한 훈련 방법론을 소개합니다.

- **Technical Details**: 모델을 정제하고 성능을 향상시키기 위한 방법으로는 인퍼런스 타임 스케일링(Inference-time scaling), 강화 학습(Reinforcement Learning), 수퍼바이즈드 파인튜닝(Supervised Fine-tuning) 및 증류(Distillation) 등이 포함됩니다. 이 논문은 LLM의 훈련 방법론을 카테고리별로 나누고, 골수 디자인 혁신 및 테스트 타임 컴퓨팅 스케일링(Test-time Compute Scaling)과 같은 요소도 고려하고 있습니다. 이러한 방법론을 통해 복잡한 과제에서의 투명한  다단계 추론을 생성할 수 있는 방향을 모색합니다.

- **Performance Highlights**: 우리가 분석한 선정된 LLM 모델들은 고급 수학 및 코딩 문제들과 같은 복잡한 작업에서 성능이 향상되었습니다. 특히, OmegaPRM 기법을 사용하여 150만 개 이상의 고품질 과정 주석을 자동으로 수집하는 알고리즘이 주목받고 있습니다. 이러한 접근을 통해 MATH500 및 GSM8K와 같은 벤치마크에서 성능을 크게 향상시킬 수 있음을 보여줍니다.



### MoRE-LLM: Mixture of Rule Experts Guided by a Large Language Mod (https://arxiv.org/abs/2503.22731)
Comments:
          2024 IEEE International Conference on Data Mining (ICDM)

- **What's New**: 이 논문에서는 인공지능 시스템의 신뢰성과 해석 가능성을 확보하기 위해 Large Language Model (LLM)을 활용하여 사람의 도메인 지식에 기초한 예측을 지원하는 새로운 접근법인 Mixture of Rule Experts Guided by a Large Language Model (MoRE-LLM)을 제안합니다. 이 방식은 데이터 기반 블랙박스 모델과 LLM에서 추출된 지식을 결합하여 도메인 지식에 맞춘 투명한 예측을 생성합니다. MoRE는 훈련 중 로컬 규칙 기반 서브모델을 발견하고 활용하게 함으로써 보다 신뢰성 있는 결과를 제공합니다.

- **Technical Details**: 제안된 MoRE-LLM 프레임워크는 특정 작업을 위해 소규모 모델을 가이드하는 LLM을 활용하여, 데이터 주도적 게이팅 모델이 특정 인스턴스에 어떤 규칙을 사용할지 결정합니다. 이를 통해 LLM이 현실 세계 훈련 데이터와 모순되는 허위 정보를 제거하고, 도메인 지식에 맞춘 규칙을 수정하여 해석력을 향상시킵니다. 모델 배포 후 LLM에 대한 접근이 필요하지 않도록 훈련 중 생성된 컨텍스트를 바탕으로 예측을 제공합니다.

- **Performance Highlights**: 수많은 표 형태 데이터 세트에서 MoRE-LLM을 평가하였으며, 해석 가능한 기준선 모델과 비해 성능을 비교하였습니다. 정량적 평가는 물론, LLM이 모델의 추론 과정에 대한 추가적인 맥락을 제공하여 이해 가능성과 신뢰성을 높이는 방식도 점검하였습니다. 결과적으로, MoRE-LLM은 기존 방법들과 비교해 높은 예측력을 유지하면서도 해석 가능성을 극대화했습니다.



### Harnessing Mixed Features for Imbalance Data Oversampling: Application to Bank Customers Scoring (https://arxiv.org/abs/2503.22730)
- **What's New**: 이 연구는 이항 분류(binary classification)에서 테이블 데이터(tabular data)에 대한 희귀 사건 탐지(rare event detection)를 조사합니다. 기존의 클래스 불균형(class imbalance)을 처리하는 기술인 SMOTE는 주로 연속형(input variables) 변수에 맞춰 설계되어 있습니다. 그러나, 실제로 많은 분류 작업은 연속형 및 범주형(mixed features) 변수를 포함하여가 예측 성능에 중요한 영향을 미칩니다.

- **Technical Details**: 이 연구에서는 혼합 특징(mixed features)을 위한 오버샘플링(overampling) 전략인 MGS-GRF를 도입합니다. 이 방법은 커널 밀도 추정기(kernel density estimator)와 지역적으로 추정된 전체 순위 공분산(locally estimated full-rank covariances)을 사용하여 연속형 변수를 생성하고, 범주형 변수는 일반화된 랜덤 포레스트(generalized random forest)에서 원본 샘플을 통해 추출합니다. MGS-GRF는 원본 데이터셋에 이미 존재하는 범주형 특성 조합만을 생성하고, 연속형과 범주형 특징 간의 의존성을 보존하는 두 가지 중요한 특성을 가진다고 평가됩니다.

- **Performance Highlights**: MGS-GRF는 LightGBM 분류기(classifiers)로 훈련된 데이터 세트와 함께 사용할 때, 다양한 합성 샘플 생성 전략에 비해 우수한 예측 성능을 보였습니다. 또한, 혼합 특성을 고려한 합성 공정은 PR 및 ROC AUC와 같은 다양한 예측 메트릭(predicitive metrics)에서 더 나은 성과를 나타내며, MGS-GRF가 가장 뛰어난 성능을 기록했습니다. 마지막으로, 이 방법은 금융 기관의 파이프라인에서 규제 요건을 준수하며 유망한 결과를 보여주고 있습니다.



### Uncertainty Weighted Gradients for Model Calibration (https://arxiv.org/abs/2503.22725)
- **What's New**: 본 논문은 딥 뉴럴 네트워크(DNN)의 모델 보정(model calibration) 문제를 분석하고, 기존의 Focal Loss(FL)와 그 변형의 효과를 통합한 새로운 손실 함수인 BSCE-GRA를 제안합니다. DNN은 분류 작업에서 과신(over-confidence) 또는 부족신(under-confidence)으로 인해 잘못 보정되는 경향이 있으며, 이로 인해 모형의 신뢰성이 저하됩니다. 저자는 손실 가중치 요인이 샘플 불확실성을 효과적으로 추정하여 보정 성능을 향상시킬 수 있다고 주장합니다.

- **Technical Details**: 제안된 BSCE-GRA 손실 함수는 샘플의 불확실성을 반영하여 그래디언트(gradient)를 스케일링하는 방식으로 최적화를 수행합니다. 기존의 손실 함수는 최적화 과정에서 불일치(misalignment) 문제와 불확실성 추정의 정밀도 부족으로 인해 최적의 보정 성능을 달성하지 못했다고 분석합니다. 이를 해결하기 위해, Brier Score를 손실 가중치 요인으로 사용하여 모든 로짓(logit)을 통해 더 정확한 불확실성 추정을 제공합니다.

- **Performance Highlights**: 다양한 모델 및 데이터셋에 대한 광범위한 실험을 통해 제안된 방법이 최신 기술(SOTA)에 해당하는 성능을 달성함을 입증하였습니다. BSCE-GRA는 불확실성 측정 지표와 일관된 효과성을 보이며, 밀접한 관련이 있는 Focal Loss 및 Dual Focal Loss와 비교하여 보다 효과적인 보정 능력을 보여 줍니다. 연구자들은 제안된 방법이 실제 애플리케이션에서 DNN의 신뢰성을 높일 수 있다고 강조합니다.



### A Spatial-temporal Deep Probabilistic Diffusion Model for Reliable Hail Nowcasting with Radar Echo Extrapolation (https://arxiv.org/abs/2503.22724)
- **What's New**: 최근 연구에서, SteamCast 모델이 레이더 에코를 활용한 우박(nowcasting) 예측을 위한 새로운 접근방식으로 소개되었습니다. 이 모델은 역사적인 기상 데이터에 기반하여 30분 동안의 예측을 6분 간격으로 제공하며, 약 1km x 1km 해상도를 지원합니다. SteamCast는 공간-시간 특성을 효과적으로 융합하여 기존의 딥러닝 모델보다 경쟁력 있는 결과를 보여줍니다. 이 모델은 레이더 데이터를 통한 우박 예측의 정확성을 크게 향상시킬 수 있는 가능성을 지니고 있습니다.

- **Technical Details**: SteamCast는 공간-시간 표현을 기반으로 한 심층 확률적 확산 모델로, 레이더 반사강도 데이터를 입력으로 사용합니다. 이 모델은 스테이블 디퓨전 아키텍처를 채택하여, 우박 예측에 필요한 효과적이고 유연한 공간-시간 레이더 에코를 표현합니다. 또한, 경량 조건 인코더를 사용하여 이웃 지역의 특성을 추출하고, U-Net 계층을 활용하여 30분 예측 결과를 생성합니다. 이 과정에서 크로스 어텐션을 통해 레이더 에코를 효과적으로 캡처하고, 여러 공간 및 시간 스케일에서 일관된 예측을 제공합니다.

- **Performance Highlights**: SteamCast는 Yan'an 시의 레이더 데이터를 기반으로 하여 30분 간격으로 우박 예측을 수행합니다. 레이더 반사강도 변수를 활용하여 9가지 수직 각도에서 예측을 진행하며, 기존의 PredRNN 및 VMRNN 모델과 비교해 우수한 성능을 기록하였습니다. 이 모델은 특히 작은 스케일의 레이더 에코 패턴을 정확하게 예측하는 데 강점을 보이고 있습니다. 또한, 계산 자원이 제한된 환경에서도 유용하게 활용될 수 있는 효율성을 자랑합니다.



### Zero-Shot LLMs in Human-in-the-Loop RL: Replacing Human Feedback for Reward Shaping (https://arxiv.org/abs/2503.22723)
Comments:
          20 pages, 2 figures, 5 Tables

- **What's New**: 본 논문에서는 강화학습에서 에이전트가 원치 않는 행동을 하게 되는 보상 불일치 문제를 해결하기 위한 두 가지 주요 기여를 제안합니다. 첫째, 자연어 처리(NLP)에 국한되지 않고, 제로샷(zero-shot) 오프더셸프(Off-the-shelf) 대규모 언어 모델(LLMs)을 사용하여 지속적인 제어작업에서 보상을 형성하는 방법을 확장했습니다. 둘째, LLM이 사람의 피드백에서 편향을 식별하고 수정할 수 있는 하이브리드 프레임워크(LLM-HFBF)를 도입하여, 편향이 반영된 보상 형성 과정을 개선했습니다.

- **Technical Details**: 이 연구에서는 MuJoCo 환경 내의 지속적인 제어 작업에서 보상 형성을 효과적으로 수행하기 위해 제로샷(zero-shot) LLM을 활용하였습니다. LLM-HFBF 프레임워크를 통해, LLM이 인간 피드백에서 발생할 수 있는 잠재적인 편향을 식별하고 이를 수정할 수 있도록 하여, 보상 형성 과정에 반영합니다. 이러한 접근 방식은 LLM의 제한과 인간 감독에서 발생하는 편향을 모두 다루어 더 균형 잡히고 신뢰할 수 있는 시스템을 만듭니다.

- **Performance Highlights**: 실험 결과에 따르면, 편향된 인간 피드백은 에이전트의 학습 성능을 크게 저하시킵니다. 평균 에피소딕 보상(AER)은 비편향 접근에서는 28.472인 반면, 보수적 편향이 있는 경우 7.039로 감소하였습니다. 반면, LLM 기반 접근 방식은 맞춤형 에지 케이스에서도 비편향 피드백과 일치하는 AER를 유지하며 성능을 향상시킵니다.



### PlatMetaX: An Integrated MATLAB platform for Meta-Black-Box Optimization (https://arxiv.org/abs/2503.22722)
- **What's New**: 최근의 동향에 따르면 최적화 문제는 점점 더 복잡해지고 있으며, 이에 따라 고급 최적화 기술이 필요하게 되었습니다. 메타 블랙 박스 최적화(MetaBBO)인 웨그는 메타 학습을 통해 최적화 알고리즘을 개선하는 데 유망한 접근 방식으로 떠올랐습니다. 이 논문에서는 강화 학습을 통합한 새로운 MATLAB 플랫폼인 PlatMetaX를 소개하며, 다양한 최적화 문제를 처리할 수 있는 포괄적인 프레임워크를 제공합니다.

- **Technical Details**: PlatMetaX는 단일 목표와 다중 목표 최적화 문제 모두를 지원하도록 설계되어 있으며, 기본 최적화 알고리즘과 평가 지표의 풍부한 세트를 갖추고 있습니다. 이 플랫폼은 MATLAB을 기반으로 하여 과학적 컴퓨팅의 강력한 기능을 제공합니다. 또한, 사용자 친화성을 강조하여, 사용자들이 메타 최적화를 쉽게 설정하고 실험할 수 있도록 돕습니다.

- **Performance Highlights**: PlatMetaX의 특징 중 하나는 기본 최적화 도구를 그대로 사용할 수 있는 기본 메타 최적화 도구 모음을 제공한다는 것입니다. 사용자는 매개변수를 조정할 필요 없이 기본 최적화 도구를 선택하고 배포할 수 있어, 메타 최적화 도구 개발에 집중하지 않아도 됩니다. 이 플랫폼은 메타 최적화 도구의 성능을 평가하기 위한 두 가지 새로운 지표를 도입하여 성능 평가를 보다 세밀하게 진행할 수 있도록 합니다.



### PowerGNN: A Topology-Aware Graph Neural Network for Electricity Grids (https://arxiv.org/abs/2503.22721)
- **What's New**: 이 논문은 재생 가능 에너지원의 통합이 증가하는 전력 시스템에서 전력 상태 예측의 중요성을 강조합니다. 기존 예측 방법의 한계를 극복하기 위해 전력 네트워크의 토폴로지를 고려한 Graph Neural Network (GNN) 프레임워크를 제안합니다. 이 프레임워크는 GraphSAGE 합성과 Gated Recurrent Units (GRUs)를 통합하여 시스템 역학에서 공간적 및 시간적 상관관계를 모델링하도록 설계되었습니다.

- **Technical Details**: 제안한 GNN은 버스와 송전선을 노드와 엣지로 표현하여 전력 네트워크를 그래프 기반으로 구성합니다. 이 모델은 NREL 118 시험 시스템에서 현실적인 재생 가능 발전 프로파일을 이용해 학습되고 평가됩니다. 물리적 구조와 시간적 동역학을 모두 통합하여 예측 정확도를 높이기 위한 특수 설계된 GNN 아키텍처를 사용합니다.

- **Performance Highlights**: 제안된 GNN은 기존의 완전 연결 신경망, 선형 회귀 및 이동 평균 모델을 포함한 기준 접근 방식보다 예측 정확도에서 뛰어난 성능을 발휘하였습니다. 평균 RMSE는 모든 예측 변수에서 0.13에서 0.17 사이에 달하며, 공간적 위치와 운영 조건에 걸쳐 일관된 성능을 보여줍니다. 이 결과들은 향후 재생 가능 에너지원이 높은 전력 시스템에서 확장 가능하고 견고한 예측을 가능하게 하는 토폴로지 인식을 강조합니다.



### Why Representation Engineering Works: A Theoretical and Empirical Study in Vision-Language Models (https://arxiv.org/abs/2503.22720)
- **What's New**: 이 논문에서는 Representation Engineering (RepE)을 Vision-Language Models (VLMs)로 확장하여, 다중모달 표현이 어떻게 보존되고 변환되는지를 분석합니다. 특히, 비주얼 입력이 언어 지식을 무시하고 잘못된 응답을 생성하는 문제를 해결하려고 합니다. 이 접근법을 통해 AI 시스템의 투명성, 공정성 및 안정성을 향상시킬 수 있는 새로운 이론적 프레임워크를 개발하였습니다.

- **Technical Details**: RepE는 고수준 개념인 정직성, 권한, 사실성 등을 이해하고 제어할 수 있는 구조적 프레임워크를 제공합니다. 새로운 연구에서는 (1) 주된 고유값(principal eigenvalue)이 신경 활동의 안정성을 보장하는 방향으로 작용하며, (2) 스펙트럼 갭(spectral gap)이 줄어들면서 여러 개념 간의 미세한 차이를 포착하는 서브도미넌트(eigenvector) 고유벡터를 허용한다는 두 가지 주요 현상을 설명합니다.

- **Performance Highlights**: 이 연구는 VLM에서의 RepE의 성공적인 적용을 보여주며, 신경 활동이 계층 간에 어떻게 전파되고 안정성을 유지하는지를 시각화하여 검증합니다. 이러한 접근 방식은 다양한 고수준 개념에 대한 분석 및 제어를 위한 안정적인 기반을 마련하여 AI 시스템의 복잡한 현상을 보다 이해하고 해석 가능하게 합니다.



### Hierarchical Adaptive Expert for Multimodal Sentiment Analysis (https://arxiv.org/abs/2503.22715)
Comments:
          11 pages, 3 figures

- **What's New**: 최근 멀티모달 감정 분석(multimodal sentiment analysis)은 다양한 통신 채널에서 인간의 감정을 이해하는 데 중요한 도구로 부각되고 있습니다. 기존 방법이 외형적으로는 많은 발전을 이루었으나, 각각의 모달리티에서 정보를 효과적으로 통합하거나 차별화하는 데 어려움을 겪고 있습니다. 이를 해결하기 위해 새로운 프레임워크인 HAEMSA(Hierarchical Adaptive Expert for Multimodal Sentiment Analysis)가 제안되었습니다.

- **Technical Details**: HAEMSA는 진화 최적화(evolutionary optimization)와 다중 작업 학습(multi-task learning) 기법을 결합하여 고유의 계층 구조를 형성합니다. 각 계층은 글로벌 및 로컬의 모달리티 표현을 캡처하고, 이를 통해 감정 분석의 정확성을 높입니다. 이 프레임워크는 불완전한 모달리티 조합 시나리오에서도 적응이 가능하며, 다양한 데이터로부터 효과적인 학습을 수행합니다.

- **Performance Highlights**: HAEMSA는 여러 벤치마크 데이터셋을 통해 우수한 성능을 입증했습니다. CMU-MOSEI에서는 7-class 정확도에서 2.6% 상승을 기록했으며, CMU-MOSI에서는 6.3% 향상되었습니다. 전반적으로 HAEMSA는 감정 인식(emotion recognition)에서도 최첨단 기술 대비 2.84% 개선된 weighted-F1 점수를 달성하여 복잡한 멀티모달 상호작용을 효과적으로 캡처하는 능력을 보여줍니다.



### Validating Emergency Department Admission Predictions Based on Local Data Through MIMIC-IV (https://arxiv.org/abs/2503.22706)
Comments:
          36 pages, 3 figures, 6 tables

- **What's New**: 이번 연구는 그리스의 병원에서 개발된 소규모 데이터셋을 기반으로 한 입원 예측 모델을 MIMIC-IV 데이터셋을 활용하여 유효성을 검증하였습니다. 이 연구는 긴급 서비스 관리에서 환자 결과를 향상시키기 위한 효과적인 접근법을 제시합니다. MIMIC-IV 데이터는 데이터셋의 포괄성이 높아 모델 검증에 있어 중요한 기준이 됩니다.

- **Technical Details**: 연구에서는 데이터 전처리(Preprocessing) 이후, Linear Discriminant Analysis (LDA), K-Nearest Neighbors (KNN), Random Forest (RF), Recursive Partitioning and Regression Trees (RPART), Support Vector Machines (SVM Radial) 등 총 5개의 알고리즘을 평가하였습니다. 이 중 Random Forest (RF)가 0.9999의 AUC-ROC와 0.9997의 민감도(Sensitivity), 0.9999의 특이도(Specificity)를 기록하며 최고의 성능을 보였습니다.

- **Performance Highlights**: RF 알고리즘은 복잡한 데이터셋을 다루는 데 있어 강력한 성능을 발휘하였으며, 이를 통해 ED(응급의료) 관리 전략을 개선할 수 있는 실질적인 통찰을 제공합니다. 또한, MIMIC-IV는 소규모 로컬 데이터셋을 기반으로 한 모델 검증에 있어 귀중한 기준점으로 자리잡고 있습니다.



### From Occurrence to Consequence: A Comprehensive Data-driven Analysis of Building Fire Risk (https://arxiv.org/abs/2503.22689)
- **What's New**: 이 연구는 미국 내 건물 화재 위험을 분석하기 위해 백만 건 이상의 화재 사건 보고서와 다양한 데이터 세트를 통합한 데이터 기반 프레임워크를 제시합니다. 이 프레임워크는 사회적 결정 요인, 건물 목록, 기상 조건 및 사건 특정 요인을 포함하여, 화재 발생과 결과에 영향을 미치는 주요 위험 요소를 식별합니다. 또한, 취약한 지역 사회가 높은 화재 위험에 직면하고 있음을 보여 주었으며, 이는 오래된 건물이나 비어 있는 건물의 존재와 관련이 깊습니다.

- **Technical Details**: 연구는 미국 내 화재 사건의 공간적 및 시간적 분포를 조사하고, 화재 발생 및 결과에 영향을 미치는 주요 요인을 확인하기 위해 고유한 데이터 세트를 활용합니다. 이 데이터 세트는 2012년부터 2022년까지 미국의 90% 이상의 카운티를 포함하여 100,000 건당 화재 사건 비율을 평가하였습니다. 연구에서 사용된 일반화 가법 모델(Generalized Additive Models, GAMs)은 사회적 결정 요인, 사업 비율, 건물 목록 및 기상 조건을 포함한 여러 요인과의 관계를 정량화하는 데 이용되었습니다.

- **Performance Highlights**: 연구 결과, 사회적 결정 요인과 건물 목록에서의 취약성이 높은 지역 사회가 화재 사건률이 높다는 강한 상관관계를 보였습니다. 또한, 기온이 낮고 건조한 기상 조건이 화재 위험을 증가시키는 것으로 나타났습니다. 이 연구는 자동 소화 시스템(AES)과 같은 특정 화재 예방 전략을 고안하는 데 도움을 줄 것으로 기대되며, 취약한 지역 사회를 위한 안전한 투자 조치를 뒷받침할 수 있습니다.



### RIG: Synergizing Reasoning and Imagination in End-to-End Generalist Policy (https://arxiv.org/abs/2503.24388)
- **What's New**: 이번 논문은 복잡한 오픈 월드 환경에서 작동하는 에이전트에 필요한 상상력(imagination)과 추론(reasoning)을 통합한 최초의 정책인 RIG를 소개합니다. 이전 연구에서는 이러한 능력이 분리된 모델로 구현되었지만, RIG는 데이터 파이프라인을 통해 두 가지 능력을 효과적으로 결합하여 학습의 효율성과 일반화 능력을 향상시킵니다. 또한, RIG는 추론과 미래 이미지를 생성하는 과정을 결합하여 환경의 동역학을 명확하게 모델링합니다.

- **Technical Details**: RIG는 오토회귀 Transformer를 통해 텍스트 추론, 저수준의 행동 제어 및 이미지 생성을 학습합니다. 초기 단계에서는 기존의 데이터에서 수집한 궤적(trajectory)을 바탕으로, 텍스트 추론이 포함된 궤적을 생성하여 RIG-basic을 훈련시키고, 이후에는 상상력을 적용하여 실패한 궤적을 수정하는 RIG-lookahead를 학습합니다. 이러한 접근 방식은 궤적의 예측된 이미지를 환경 상태로 활용하여 가상 궤적을 생성하고 이를 기반으로 추론하여 행동을 예측하는 구조를 제공합니다.

- **Performance Highlights**: RIG는 마인크래프트 환경에서 광범위한 실험을 통해 현재의 최첨단 성능을 크게 향상시켰습니다. 결과적으로 111시간의 비디오로 훈련함으로써 전작들에 비해 17배 더 높은 샘플 효율성을 보여주며, 다양한 환경 상호작용과 추론 중 미리보기 단계를 조정하여 견고성과 일반화 능력이 지속적으로 향상됨을 입증하였습니다.



### UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving (https://arxiv.org/abs/2503.24381)
Comments:
          14 pages; Dataset: this https URL Code: this https URL

- **What's New**: UniOcc는 점유 예측(occupancy forecasting) 및 현재 프레임 점유 예측을 위한 포괄적인 벤치마크입니다. 다양한 실제 데이터셋(nuScenes, Waymo)과 고충실도 시뮬레이터(CARLA, OpenCOOD)의 데이터를 통합하여 2D/3D 점유 레이블과 각 복셀(voxel) 흐름(flow) 주석을 제공합니다. 새로운 평가 지표를 통합하여 기존의 잘못된 중간 진리(ground-truth)에 의존하지 않고 점유 품질을 평가할 수 있는 robust한 방안을 제시합니다.

- **Technical Details**: UniOcc는 단일 데이터셋에 의존하던 기존 방법들의 제약을 극복하고 크로스 데이터셋 학습을 지원합니다. CARLA 시뮬레이션을 활용하여 다양한 훈련 데이터를 제공하고, 각 복셀에 대한 전향(forward) 및 역방향(reverse) 흐름 주석을 통해 동적 장면 단서를 포착할 수 있도록 합니다. 이는 협력적 주행(cooperative driving) 시나리오를 지원하는 최초의 데이터셋이기도 합니다.

- **Performance Highlights**: 광범위한 실험을 통해 대규모 다양한 훈련 데이터와 명시적인 흐름 정보가 점유 예측 및 예측 성능을 유의미하게 향상시킨다는 것을 입증하였습니다. 우리는 UniOcc이 점유 중심 연구의 촉매제로 작용하여 자율 주행에서의 혁신을 촉진할 것이라고 기대합니다. 또한 기존 방법들이 크로스 도메인 일반화에 어려움을 겪고 있음을 보여주어 향후 연구의 방향성을 제시합니다.



### Exploring the Effect of Reinforcement Learning on Video Understanding: Insights from SEED-Bench-R1 (https://arxiv.org/abs/2503.24376)
Comments:
          Technical Report (In Progress); Code released at: this https URL

- **What's New**: 본 논문은 멀티모달 대규모 언어 모델(MLLMs)의 비디오 이해를 평가하기 위한 새로운 벤치마크인 SEED-Bench-R1을 제안합니다. SEED-Bench-R1은 복잡한 일상적인 계획 작업을 여러 선택 질문 형태로 포함하여, 정교한 인식(perception)과 논리적 추론(logical reasoning)을 요구합니다. 또한, 이 벤치마크는 세 가지 수준의 일반화(generalization) 시나리오를 통해 MLLMs의 포스트 트레이닝(post-training) 방법을 체계적으로 평가합니다.

- **Technical Details**: SEED-Bench-R1은 현실적인 일상 활동을 기반으로 한 비디오를 사용하여, 모델이 목표를 이해하고 긴 시간 동안 시각적인 진행을 추적하며, 복잡한 환경 관찰을 인지하고, 세계 지식을 사용하여 다음 행동을 추론할 수 있도록 설계되었습니다. 이 벤치마크는 교육 데이터셋을 기반으로 하며, 명확하게 검증 가능한 정답을 제공하여 일반화 능력을 철저히 평가할 수 있는 구조로 되어 있습니다. Qwen2-VL-Instruct-7B를 사용하여 RL 및 감독된 파인튜닝(SFT) 방법을 비교하여 RL이 데이터 효율성과 성능 면에서 우수하다는 것을 보여주었습니다.

- **Performance Highlights**: 실험 결과, RL은 특히 OOD(Out-of-Distribution) 시나리오에서 SFT를 능가하며, 비디오 이해의 일반적인 벤치마크에서도 높은 성과를 보였습니다. RL은 시각적 인식을 향상시키고 COT(Chain of Thought) 토큰을 동적으로 쿼리하도록 모델을 교육하는 데 효과적이었습니다. 그러나 모델이 때때로 중요한 시각적 단서를 무시하는 등, 몇 가지 한계점도 드러났고, 이는 향후 연구와 개선 방향 설정에 중요한 요소가 될 것입니다.



### Policy Gradient for LQR with Domain Randomization (https://arxiv.org/abs/2503.24371)
- **What's New**: 본 논문은 Domain Randomization (DR)의 정책 경량화 기법에 대한 첫 번째 수렴 분석을 제공합니다. 이를 통해 DR 최적화 문제에서 정책 경량화 방법들이 어떻게 수렴하는지를 이론적으로 제시하며, 특히 균질성이 제한된 샘플 시스템에서의 전역적 수렴을 보장합니다. 또한, 초기 안정화 컨트롤러의 필요성을 없애는 할인 요인 감소 알고리즘을 제안하고 있습니다.

- **Technical Details**: 이 연구는 Linear Quadratic Regulation (LQR) 문제를 다루며, DR 목표의 샘플 평균 근사치를 최적화하는 방법론을 소개합니다. 정책 경량화 방법이 샘플 평균 근사 매개변수에 대해 전역 최적 솔루션으로 수렴함을 증명하며, 이는 이전 연구에서 나타난 이질성 편향을 제거합니다. 제안된 알고리즘은 여러 다이나믹 시스템을 안정화하는 첫 번째 순서 경량화 방법을 기반으로 하여, 기존 연구보다 확장된 형태로 제시됩니다.

- **Performance Highlights**: 실험 결과는 이론적 발견을 뒷받침하며, 향후 연구 방향으로 리스크 민감한 DR 구성 및 확률적 정책 경량화 알고리즘이 포함됩니다. 본 연구의 결과는 시뮬레이션에서 실제로의 전이에서 신뢰성 있는 강화 학습 알고리즘 설계 및 분석의 중요한 발전을 보여줍니다. 연구자들은 DR 목표를 다른 위험 메트릭으로 일반화하고, 확률적 정책 경량화 알고리즘의 수렴성을 탐구하는 데도 한계를 인식하고 있습니다.



### Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation (https://arxiv.org/abs/2503.24361)
Comments:
          Project website: this https URL

- **What's New**: 이 논문은 시뮬레이션 데이터와 실제 데이터를 혼합하여 정책(policy)을 공동 학습(co-training)하는 새로운 방법론을 제안합니다. 최근의 연구에서 시뮬레이션 데이터를 사용한 정책 학습이 실제 데이터에서만 학습한 경우보다 성능이 크게 향상될 수 있다는 점이 부각되었습니다. 그러나, 실제로 이 방법이 어떻게 효과적인지에 대한 체계적인 이해는 부족한 상황입니다.

- **Technical Details**: 이 연구에서는 시뮬레이션 데이터와 실제 데이터를 효과적으로 혼합하는 방법을 제시하고, 로봇 팔(robot arm)과 휴머노이드(humanoid)의 다양한 작업(task)을 통해 이를 실증적으로 검증합니다. 시뮬레이션 데이터는 두 가지 주요 원천으로 나뉘며, 각각 작업 인지(task-aware) 시뮬레이션과 작업 비인식(task-agnostic) 시뮬레이션으로 구분됩니다. 이 연구는 시뮬레이션 데이터가 실제 환경에서 어떻게 개선될 수 있는지에 대한 통찰을 제공합니다.

- **Performance Highlights**: 실험 결과, 시뮬레이션 데이터를 활용한 공동 학습이 평균 38%의 성능 향상을 이루었음을 보여주었습니다. 연구는 시뮬레이션 데이터와 실제 데이터 간의 간섭이 클 경우에도 성능 개선이 가능하다는 것을 발견했습니다. 이러한 findings는 로봇 공학 실무자들에게 중요한 전략을 제공할 수 있습니다.



### Faster Rates for No-Regret Learning in General Games via Cautious Optimism (https://arxiv.org/abs/2503.24340)
Comments:
          Appeared at STOC 2025

- **What's New**: 본 연구는 다수의 플레이어가 참여하는 일반-합 게임에서 $O(n \\log^{2} d \\log T)$의 플레이어별 레그렛(regret)을 달성하는 첫 번째 uncoupled learning 알고리즘을 제안합니다. 이는 기존의 Log-Regularized Lifted Optimistic FTRL 알고리즘이 달성한 $O(n \\cdot d \\log T)$ 레그렛보다 지수적으로 개선된 결과입니다. 또한, 알고리즘은 Optimistic Hedge와 비교하여 반복 횟수 $T$에 대한 의존도를 $\\log^{4} T$에서 $\\log T$로 줄였습니다.

- **Technical Details**: 우리의 알고리즘은 Classic Optimistic Multiplicative Weights Update (OMWU)과 적응형 비단조 학습률를 결합하여 플레이어의 레그렛이 너무 부정적일 때 더 신중하게 학습할 수 있도록 합니다. 이 알고리즘을 통해 등장하는 새로운 방법론은 Dynamic Learning Rate Control OMWU (DLRC-OMWU)로 명명됩니다. 각 반복에서 플레이어의 현재 레그렛 벡터에 기반하여 학습률을 조정하는 최적화 문제를 해결함으로써 플레이어의 학습 속도를 적절하게 조절하는 것이 핵심입니다.

- **Performance Highlights**: DLRC-OMWU 알고리즘은 다수 플레이어가 있는 일반-합 게임에서 레그렛의 새로운 이론적 경계를 제시하며, 기존의 Log-Regularized Lifted Optimistic FTRL에 비해 지수적으로 d에 대한 의존성을 낮추는 성과를 보였습니다. 본 연구는 레그렛 최소화에 대한 기존 알고리즘을 크게 초월하는 결과를 제공하며, 이러한 발견은 게임이론 뿐만 아니라 다양한 응용 프로그램에 널리 활용될 수 있습니다.



### Contextual Preference Collaborative Measure Framework Based on Belief System (https://arxiv.org/abs/2503.24328)
Comments:
          in Chinese language

- **What's New**: 이 논문은 인간의 개입을 줄이기 위한 새로운 preference collaborative measure framework를 제안합니다. 이 프레임워크는 업데이트된 belief system을 바탕으로 하며, preference 측정의 정확도와 효율성을 향상시키는 데 기여합니다. 또한, 사용자의 공통된 선호를 발견하기 위한 방법론이 포함되어 있습니다.

- **Technical Details**: 논문에서는 규칙 간의 거리와 평균 내부 거리(average internal distance)를 정의하여 사용자 간의 공통된 선호(common preference)를 발견하는 방법을 제시합니다. PRA 알고리즘은 정보 손실을 최소화하면서 선호 규칙을 찾기 위한 기법으로, belief system에 따라 규칙의 확인과 선호 규칙의 분류를 위한 신뢰도(belief degree)와 편차(deviation degree)를 도입합니다. 이를 통해 최종적으로 Top-K 흥미로운 규칙을 필터링하는 시스템을 구현하였습니다.

- **Performance Highlights**: 제안된 IMCos 및 IMCov 알고리즘은 가중 코사인 유사도(weighted cosine similarity)와 상관 계수(correlation coefficients)를 사용하여 프레임워크의 정확도와 효율성을 검증합니다. 실험 결과 이 두 알고리즘은 기존 우수한 알고리즘들에 비해 대부분의 측면에서 뛰어난 성능을 발휘하였습니다. 이러한 결과는 이 새로운 프레임워크가 효과적임을 입증하는 중요한 사례로 작용합니다.



### Self-Supervised Pretraining for Aerial Road Extraction (https://arxiv.org/abs/2503.24326)
- **What's New**: 이 논문의 주요 혁신은 라벨이 없는 데이터를 이용하여 항공 이미지 세분화 성능을 향상시키는 자기 감독형(pretraining) 방법을 제안하는 것입니다. 이를 위해 모델은 항공 이미지의 누락된 영역을 복원하는 inpainting 기반 전 훈련 기법을 사용하여, 도로 추출을 위한 미세 조정 단계에 들어갑니다. 이 방법은 라벨된 데이터에 대한 의존도를 줄이고, 일반화(generalization)를 개선하며 도메인 변화(domain shift)에 강인성을 높입니다.

- **Technical Details**: 제안된 방법은 세 가지 단계로 구성되어 있으며, 라벨이 없는 항공 이미지로 시작하여 자기 감독형 inpainting 단계를 통해 정보 구조를 학습합니다. 그런 다음, 도메인 갭을 해소하는 데 초점을 맞춘 두 번째 훈련 단계를 통해 도로 세분화(task)와 연계된 특성의 전이 가능성을 향상시키고자 합니다. 이러한 방식은 CNN 기반 모델 아키텍처와 데이터 세트에 관계없이 적용 가능하며, 최소한의 수정으로도 성능 향상을 꾀할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 전 훈련 방법이 세분화 정확도를 크게 향상시킴을 보여주었습니다. 특히 데이터가 적은 상황에서도 성능 상승을 이루어내며, 다양한 아키텍처에서 일관된 성능 개선을 보였습니다. 이를 통해 제안된 방법은 항공 이미지 분석의 확장 가능한 솔루션으로 자리매김할 수 있음을 나타냅니다.



### Sample-Optimal Private Regression in Polynomial Tim (https://arxiv.org/abs/2503.24321)
- **What's New**: 이 논문에서는 Gaussian 공변량을 가진 일반적인 최소 제곱 회귀 문제에서 개인적으로 예측 오류 보장을 얻는 작업을 다룹니다. 이 작업을 위한 첫 번째 샘플 최적의 다항 시간 알고리즘을 제공하며, 이는 순수 및 근사적 차별 프라이버시 하에서도 적용됩니다. 이전의 모든 효율적인 알고리즘은 샘플 복잡도가 비최적 차원 의존성을 가지고 있었거나, 프라이버시 매개변수에 대해 다항적으로 더 나쁜 의존성을 보였습니다.

- **Technical Details**: 이 알고리즘은 임의의 소수의 이상치에 강인하며, 이상치의 비율에 따른 최적 오류율을 달성합니다. 기술적인 기여는 두 가지로 나뉘어 있습니다: 첫째, Gaussian의 회복성 보장을 활용하여 sum-of-squares 프레임워크 내의 회귀를 위한 효율적인 sum-of-squares 알고리즘을 얻습니다. 둘째, 최근의 프라이버시 강인성 프레임워크를 일반화하여 입력 샘플의 공분산으로 유도된 기하학을 고려합니다.

- **Performance Highlights**: 이 알고리즘은 기존의 다른 알고리즘보다 우수한 성능을 보여 주며, 특히 공분산을 고려한 평균 추정에 있어 효율적인 알고리즘을 제공합니다. 우리는 프라이버시 매개변수에 대한 최적 의존성을 보여 주었으며, 이로 인해 보다 나은 데이터 보호와 예측 정확도를 동시에 달성합니다.



### A Systematic Evaluation of LLM Strategies for Mental Health Text Analysis: Fine-tuning vs. Prompt Engineering vs. RAG (https://arxiv.org/abs/2503.24307)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)을 활용한 정신 건강 텍스트 분석을 위한 세 가지 접근 방식을 체계적으로 비교하였습니다: prompt engineering, retrieval augmented generation (RAG), 그리고 fine-tuning입니다. 이 연구는 LLaMA 3를 사용하여 감정 분류 및 정신 건강 상태 감지 작업을 두 개의 데이터셋에서 평가하였습니다. 연구 결과, fine-tuning이 감정 분류에서 91%, 정신 건강 조건 분류에서 80%의 정확도를 달성하였으며, prompt engineering과 RAG는 보다 유연한 배포가 가능하지만 보통의 성능(40-68% 정확도)을 보여주었습니다.

- **Technical Details**: 정신 건강 텍스트 분석을 위한 LLM의 세 가지 접근 방식은 fine-tuning, prompt engineering, RAG입니다. 특히, fine-tuning은 높은 정확도를 요구하지만 많은 컴퓨팅 리소스와 대규모 훈련셋을 필요로 합니다. 반면, prompt engineering과 RAG는 상대적으로 적은 자원으로 보다 유연한 배포가 가능하게 하며, 다양한 설정에서 효과적으로 구현할 수 있다는 장점이 있습니다.

- **Performance Highlights**: 이 연구는 정신 건강 분야에서 LLaMA 3 기반 모델의 효과를 입증하였으며, 감정 분류 및 정신 건강 상태 분류에서 매우 높은 정확도를 기록했습니다. 이러한 결과는 임상 환경에서 LLM 기반 솔루션의 구현에 있어 중요한 통찰력을 제공합니다. 향후 정신 건강 평가 도구의 개발에 중요한 의미를 가지며, 높은 정확도의 fine-tuning 외에도 prompt engineering과 RAG 접근 방식이 자원과 배포 유연성 면에서 유효한 대안이 된다는 점을 강조하고 있습니다.



### Solving the Best Subset Selection Problem via Suboptimal Algorithms (https://arxiv.org/abs/2503.24300)
- **What's New**: 이 논문에서는 선형 회귀에서의 최적 부분 집합 선택(Best Subset Selection, BSS) 문제를 해결하기 위한 새로운 절차를 소개하며, 이 절차를 기존의 다른 하위 최적 알고리즘과 비교합니다. BSS 문제는 차원이 증가할수록 조합 가능한 부분 집합의 수가 급격히 증가하여 전역 최적 해(global optimal solution)를 찾는 것이 매우 어려워지므로, 계산 비용을 절감할 수 있는 효과적인 방법의 필요성이 강조됩니다. 따라서 하위 최적 절차의 중요성이 부각되며, 새로운 절차가 고차원 데이터의 부분 집합 선택 문제를 해결하는 데 경쟁력을 가진 알고리즘임을 시사합니다.

- **Technical Details**: 선형 회귀 모델은 y=Xβ+ε의 형태로 구성되며, 여기서 y는 응답 벡터, X는 설계 행렬, β는 계수 벡터, ε는 잡음 벡터입니다. BSS 문제는 NP-hard 문제로 광범위한 차원의 데이터에서 완벽한 최적화를 귀찮고, pseudo-norm으로 정의된 카드널리티 제약(constraint)을 가지고 있습니다. 저자들은 높은 차원의 데이터에서 BSS 문제를 해결하기 위해 네 가지 기존 알고리즘과 비교하여 새로운 하위 최적 알고리즘을 제안하고, 각 알고리즘의 성능을 분석합니다.

- **Performance Highlights**: 광범위한 계산 실험이 합성 데이터(synthetic data)와 실제 데이터를 사용하여 수행되어, 다양한 데이터 환경에서 알고리즘의 성능이 평가되었습니다. 결과적으로, 새로운 절차는 하위 최적 알고리즘으로서 긍정적인 성과를 거두었으며, 차원 수가 많을 때도 효과적인 근사 솔루션을 제공할 수 있음을 알게 되었습니다. 이러한 연구는 BSS 문제를 해결하기 위한 더 나은 방법론을 제시하고, 실제 데이터에 대한 적용 가능성을 높이는 데 기여할 것입니다.



### Fair Dynamic Spectrum Access via Fully Decentralized Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2503.24296)
Comments:
          To appear in WiOpt 2025

- **What's New**: 본 논문은 정보 공유 없이 여러 출처-목적지 쌍이 제약된 주파수 대역을 공유하는 분산형 무선 네트워크를 제안합니다. 각 출처는 자신의 전송 결과(성공 또는 충돌)를 통해 전송 전략을 적응하도록 학습하며, 네트워크 크기에 대한 사전 지식이 없습니다. 제안된 Fair Share RL (FSRL) 솔루션은 분산형 방식으로 공정성을 달성하는 새로운 강화 학습(RL) 기반 방법을 사용합니다.

- **Technical Details**: FSRL 에이전트는 (i) 반적응형(sem-adaptive) 타임 참조를 이용한 상태 증가(state augmentation), (ii) 위험 제어(risk control)와 시간 차 가능성(time difference likelihood)을 활용하는 아키텍처, (iii) 협력 없이 공정성을 목표로 하는 보상 구조를 통합하여 구성됩니다. 이 에이전트는 시간 슬롯(specific time slot) 간의 전송 결과를 분석하고 이에 따라 최적의 전송 전략을 결정합니다.

- **Performance Highlights**: 시뮬레이션 결과에 따르면, FSRL은 여러 출처와 단일 주파수 대역이 있는 환경에서 89.0% 더 공정하며, 평균적으로 48.1% 더 공정한 결과를 보였습니다. Jain의 공정성 지수(Jain's fairness index)를 사용하여 측정된 이 결과는 FSRL이 기존 강화 학습 알고리즘보다 뛰어난 성능을 나타냄을 의미합니다.



### Learning Velocity and Acceleration: Self-Supervised Motion Consistency for Pedestrian Trajectory Prediction (https://arxiv.org/abs/2503.24272)
- **What's New**: 본 논문은 보행자 궤적 예측에서 기존의 감독 학습(supervised learning) 대신 자가 감독 학습(self-supervised learning) 프레임워크를 제안합니다. 이 방법은 위치(position), 속도(velocity), 가속도(acceleration)를 명시적으로 모델링하여 궤적 예측의 정확도를 높입니다. 특히, 모션 일관성 평가(mechanical consistency evaluation) 전략을 도입하여 예측된 동작 경향과 역사적 동작을 비교하고, 이를 바탕으로 궤적 생성을 가이드합니다.

- **Technical Details**: 제안된 프레임워크는 세 개의 스트림 네트워크(three-stream network)를 기반으로 하며, 각 스트림은 Historical 데이터에서 위치, 속도, 가속도 정보를 처리합니다. 이 네트워크는 속도와 가속도 정보를 통해 위치 예측을 보강하며, 속도 스트림에는 가속도 기능이 추가되어 함께 최적화됩니다. 또한, 사회적 디코더(social decoders)를 활용하여 보행자 상호작용을 분석하고, 결과적으로 예측된 위치와 물리적 일관성을 유지합니다.

- **Performance Highlights**: ETH-UCY 및 Stanford Drone 데이터셋을 사용한 실험 결과, 제안된 방법은 기존의 방법들과 비교하여 최첨단 성능(state-of-the-art performance)을 달성했습니다. 특히, 장기 분포(long-tail distribution)로 인한 모델의 과적합 문제를 완화하고, 비정상적인 동작을 보다 정확하게 포착할 수 있는 능력을 보여주었습니다.



### Enhancing Image Resolution of Solar Magnetograms: A Latent Diffusion Model Approach (https://arxiv.org/abs/2503.24271)
Comments:
          Accepted for publication on A&A

- **What's New**: 최근 태양 자기장의 공간적 특성을 이해하는 것이 태양 내부 물리적 프로세스를 해독하고 지구에 미치는 영향을 파악하는 데 매우 중요합니다. 본 연구에서는 Michelson Doppler Imager (MDI) 데이터셋을 헬리오시즘 및 자기 이미저 (HMI) 기술로 보완하기 위해 새로운 diffusion model 기반의 Super-Resolution 기법을 제안합니다. 이로 인해 기존 MDI 이미지의 해상도를 2"/픽셀에서 0.5"/픽셀로 향상시킬 수 있었습니다.

- **Technical Details**: 연구진은 Latent Diffusion Model (LDM)을 사용하여 HMI 데이터에 대한 잔여물을 학습시킨 후, MDI와 HMI의 쌍 데이터로 세밀하게 조정했습니다. 기존 결정론적 모델들과 DDPM(Denoising Diffusion Probabilistic Models)을 비교하여, 제안된 LDM이 더 낮은 해상도의 이미지를 다루는 데 뛰어난 성능을 보임을 확인했습니다. Fourier 도메인에서의 분석을 통해 LDM이 2" 이하의 세부 항목을 해상할 수 있다는 점을 입증했습니다.

- **Performance Highlights**: 복원된 이미지의 품질은 PSNR, SSIM, FID, LPIPS 등의 고전적 지표를 이용해 평가되었습니다. LDM은 예측의 신뢰성을 평가하는 추가 기법을 개발하여 과거의 태양 이벤트의 동적 특성을 더 잘 이해할 수 있는 가능성을 열었습니다. 최종적으로 MDI 데이터를 Super-Resolve하여 태양 주기 23과 관련된 폭발 사건들을 더욱 세밀하게 연구할 수 있을 것입니다.



### GPU-centric Communication Schemes for HPC and ML Applications (https://arxiv.org/abs/2503.24230)
Comments:
          A surveyor on Communication Schemes for Distributed HPC and ML Applications. Article in consideration for journal publication

- **What's New**: 이 논문은 현대 이기종 슈퍼컴퓨팅 시스템에서의 프로세스 간 통신을 효율적으로 지원하기 위한 새로운 GPU 중심의 통신 방식들을 조사합니다. 기존의 GPU 인식 통신 모델은 CPU 스레드가 데이터를 조작해야 했으나, 새로운 방식은 GPU가 통신 작업을 직접 관리할 수 있도록 개선되었습니다. 이를 통해 계산과 통신의 중첩을 효율적으로 수행하고, 통신 지연을 최소화하며, GPU의 자율성을 높일 수 있습니다.

- **Technical Details**: 이 연구에서는 GPU와 네트워크 인터페이스 카드(NIC)의 기능을 활용하여 CPU의 개입 없이 GPU가 직접 데이터를 이동할 수 있는 새로운 통신 방식을 제안합니다. 제안된 세 가지 통신 방식은 스트림 트리거 통신(stream triggered communication), 커널 트리거 통신(kernel triggered communication), 그리고 커널 주도 통신(kernel initiated communication)으로 구분됩니다. 이들 각 방식은 GPU에 부착된 메모리 버퍼를 활용하여 효율적인 데이터 전송을 가능하게 합니다.

- **Performance Highlights**: 제안된 GPU 중심 통신 방식은 기존 통신 모델보다 훨씬 나은 성과를 보여줍니다. 연구 결과에 따르면, 이 새로운 방식들은 계산 및 통신 중첩을 강화하고, 통신 지연을 크게 줄이며, GPU의 자율성을 높이는 데 기여합니다. 이 논문은 GPU 중심 통신 방식의 구현에 필요한 하드웨어 및 소프트웨어 기능과 관련된 잠재적인 도전 과제들에 대해서도 논의합니다.



### MB-ORES: A Multi-Branch Object Reasoner for Visual Grounding in Remote Sensing (https://arxiv.org/abs/2503.24219)
- **What's New**: 이 논문에서는 원격 감지 이미지에 대해 객체 검출(Object Detection, OD)과 시각적 기초(Visual Grounding, VG)를 통합하는 통합 프레임워크를 제안하고 있습니다. 전통적인 OD와 VG 작업을 위한 직관적인 사전 지식을 수립하기 위해, 언급 표현 데이터를 사용하여 오픈 세트 객체 감지기를 세밀 조정하고, 부분적으로 감독된 OD 작업으로 설정합니다. 이러한 구조를 통하여 모든 객체를 탐지하면서 특정 객체의 위치를 정확하게 찾는 것을 목표로 합니다.

- **Technical Details**: 제안된 모델은 객체 질의, 클래스 임베딩, 및 제안 위치로 구성된 그래프 표현을 사용하여 각 이미지를 구성합니다. 멀티-브랜치 네트워크는 공간적, 시각적, 범주적 특성을 통합하여 작업 인식 제안을 생성하며, 객체 추론 네트워크는 제안들 사이의 확률을 할당합니다. 이 과정은 마지막으로 언급된 객체를 로컬라이즈하기 위한 부드러운 선택 메커니즘으로 이어집니다.

- **Performance Highlights**: 이 방법은 OPT-RSVG 및 DIOR-RSVG 데이터 세트에서 뛰어난 성능을 입증하였으며, 최신 방법들에 비해 상당한 성능 개선을 보여 주었습니다. 전통적인 OD 기능을 유지하면서도 보다 다양한 시나리오에서 OD의 적용 가능성을 확대하였습니다. 또한, 이 논문의 코드는 연구 결과를 재현하고 실험할 수 있도록 제공될 예정입니다.



### Data-driven construction of a generalized kinetic collision operator from molecular dynamics (https://arxiv.org/abs/2503.24208)
- **What's New**: 이번 연구는 분자 동역학(molecular dynamics)을 통해 일반화된 동역학적 충돌 연산자를 직접 학습하는 데이터 중심(data-driven) 접근 방식을 소개합니다. 기존의 Landau 충돌 모델과는 달리, 제안된 연산자는 집합적인 상호작용을 고려하여 비등방성(anisotropic) 형태를 취합니다. 이 연구는 충돌 에너지 전달에서 비등방성의 영향을 보존하는 것이 플라즈마 동역학 예측에 필수적이라는 점을 강조합니다.

- **Technical Details**: 이 연구에서는 공간적으로 동질적인 플라스마 시스템의 동역학을 고려하며, 총 운동량이 0이라는 가정을 두고, 입자의 속도에 대한 확률 밀도 함수(PDF)를 설정합니다. 충돌 연산자는 bilinear form을 따라 정의되며, 특정 커널(kernel)을 통해 입자 간 상호작용을 모델링합니다. 제안된 충돌 연산자는 기존의 Landau 모델보다 계산 복잡성이 낮으며, 3차원(3D)에서 동작합니다.

- **Performance Highlights**: 제안된 연산자는 단일 성분 플라즈마(one-component plasma) 시뮬레이션을 통해 평가되었으며, 넓은 Coulomb 결합 영역에서 동역학적 과정을 정확하게 모델링할 수 있음을 보여주었습니다. 이는 충돌 상호작용의 이질적인 특징을 반영한 결과로, 기존 Landau 형태에서의 한계를 극복함을 보여줍니다. 이 연구는 입자 간의 비등방적 상호작용을 포착하여 예측 정확성을 높일 수 있음을 강조합니다.



### A Comparison of Parametric Dynamic Mode Decomposition Algorithms for Thermal-Hydraulics Applications (https://arxiv.org/abs/2503.24205)
- **What's New**: 최근 몇 년간 인공지능(Artificial Intelligence) 기술의 발전과 대량 데이터의 가용성 덕분에 데이터로부터 모델을 학습하는 알고리즘이 인기를 끌고 있습니다. 본 논문은 이러한 학습 기법 중 Reduced Order Modelling 프레임워크의 다이나믹 모드 분해(Dynamic Mode Decomposition) 기술의 다양한 버전을 비교하고 이들의 장단점을 평가하는 데 기여합니다. 또한, 기존의 방법론이 다루지 못하는 파라메트릭(time series) 시간 시계열 문제에 대한 연구도 지속되고 있습니다.

- **Technical Details**: 다이나믹 모드 분해(Dynamic Mode Decomposition) 알고리즘은 주어진 시간 시계열 데이터셋을 통해 물리적 현상을 나타내는 최적의 선형 모델을 학습하는 것을 목표로 합니다. 이 기술은 원래 데이터셋의 기간을 넘어 시간을 진전시키는 데 사용할 수 있는 상태 연산자(state operator)를 생성합니다. 그러나 표준 형식의 다이나믹 모드 분해는 파라메트릭 시간 시계열을 처리할 수 없으며, 각각의 파라미터 실현에 대해 별도의 선형 모델을 도출해야 하는 문제가 있습니다.

- **Performance Highlights**: 본 연구에서는 두 가지 벤치마크 '실린더 위 흐름(flow over cylinder)' 테스트 케이스와 Politecnico di Milano의 DYNASTY 실험 시설에서 수집한 데이터셋을 포함한 세 가지 열유체(thermal-hydraulics) 문제를 고려하고 있습니다. 데이터셋은 FEniCS 유한 요소 솔버와 CFDbench 데이터셋에서 가져온 결과입니다. 이 연구는 다이나믹 모드 분해의 다양한 알고리즘을 비교 분석하여 각 기법의 장단점을 평가하고 있습니다.



### Traffic Engineering in Large-scale Networks with Generalizable Graph Neural Networks (https://arxiv.org/abs/2503.24203)
- **What's New**: 본 논문은 TELGEN이라는 새로운 트래픽 엔지니어링(TE) 알고리즘을 제안합니다. TELGEN은 다양한 네트워크 조건에서 TE 문제를 효율적으로 해결하도록 학습하며, 기존 학습 기반 방법들이 가진 한계를 극복합니다. 본 알고리즘은 최적 TE 솔루션을 예측하는 대신 최적 TE 알고리즘을 예측하는 방식으로 문제를 변환하여, 다양한 네트워크 및 트래픽 패턴에 대해 일반화된 성능을 발휘합니다.

- **Technical Details**: TELGEN은 그래프 신경망(GNN) 아키텍처를 기반으로 하며, 기존의 TE 알고리즘을 모방하여 수퍼바이즈드 러닝(supervised learning)으로 학습합니다. 이 과정에서 TELGEN은 작은 규모의 TE 문제를 LP(Linear Programming)로 모델링하고, 고전의 TE 알고리즘이 최적성을 달성하는 자세한 단계를 학습합니다. 이렇게 학습한 알고리즘은 네트워크의 토폴로지나 트래픽 패턴에 구애받지 않고 나중에 더 큰 문제에도 잘 일반화될 수 있습니다.

- **Performance Highlights**: TELGEN은 5000개 노드와 106개의 링크를 가진 무작위 및 실제 네트워크에서 평가되었습니다. 이 알고리즘은 최적성 간극이 3% 미만이면서 모든 경우에 대해 실행 가능성을 보장하고, 고전적인 최적 해결책보다 최대 84%의 문제 해결 시간을 절약했습니다. TELGEN은 또한 최신 학습 알고리즘보다 학습 시간과 문제 해결 시간을 각각 2-4배 줄여주는 성능을 보여주었습니다.



### Graph Neural Network-Based Predictive Modeling for Robotic Plaster Printing (https://arxiv.org/abs/2503.24130)
- **What's New**: 이 논문은 로봇 팔을 사용한 파티클 기반 제작 공정에서 생성되는 표면을 예측하기 위해 그래프 신경망(Graph Neural Network, GNN) 모델링 접근 방식을 제안합니다. 이 접근 방식은 벽면에 시멘트 플라스터를 스프레이 방식으로 인쇄하는 과정과 관련이 있습니다. GNN 모델은 인코더-프로세서-디코더 아키텍처로 구성되어 있으며, 이를 통해 로봇 팔의 이동 경로, 속도 및 방향과 같은 특징을 활용해서 예측을 수행합니다.

- **Technical Details**: 제안된 GNN 모델은 실험 데이터를 사용하여 훈련되며, 베이지안 최적화 방법을 통해 하이퍼파라미터가 최적화됩니다. 이 모델의 주요 목표는 인쇄 프로세스의 시뮬레이터 역할을 하며, 최종적으로는 로봇 팔의 이동 경로를 생성하고 인쇄 매개변수를 최적화하여 자율 플라스터링 프로세스를 실현하는 것입니다. 또한, 이 모델은 예측 오류를 측정하여 기존의 벤치마크 모델과 비교할 때 성능이 크게 향상된 것을 보여줍니다.

- **Performance Highlights**: 제안된 모델은 예측 오류 측면에서 기존 벤치마크 모델과 비교할 때 상당한 개선을 나타냅니다. 특히, 다양한 시나리오에서 일반성을 보여주며 과거 데이터와 비교 시 예측 단계에서 향상된 오류 축척을 지닙니다. 이 모델은 하루에 최대 200 m²의 표면을 처리하며, 자재 사용량을 최대 20%까지 절감할 수 있는 효율적인 성능을 보여줍니다.



### It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data (https://arxiv.org/abs/2503.24129)
Comments:
          Accepted to CVPR 2025, Project page: this https URL

- **What's New**: 이번 논문에서는 비전과 언어의 기본 모델들이 발전함에 따라 표현의 동질화(homogeneity)가 증가한다는 이론을 제시합니다. 특히, 서로 다른 모달리티(modality) 간의 거리(pairwise distance)가 더욱 유사해진다는 점에 주목하고, 이를 통해 전통적인 데이터 쌍이 없이도 '블라인드' 방식으로 비전과 언어 표현을 매칭하는 가능성을 검토합니다. 기존 연구에 대한 비판적 시각을 살펴보며, 우리가 제시하는 방법이 비전-언어 정렬의 새로운 가능성을 열어줄 것임을 강조하였습니다.

- **Technical Details**: 이 연구는 쌍관계 문제를 quadratic assignment problem (QAP)의 형태로 수학적으로 공식화합니다. 제안된 새로운 heuristic 기법은 기존의 알고리즘보다 더 효율적이며, 최적 매칭 문제에 대한 해결책을 제시합니다. 비전-언어 표현 간의 매칭을 평가하기 위해 33개의 비전 및 27개의 언어 모델을 사용해 대규모 연구를 수행하였으며, 이 과정에서 pairwise distances를 활용하여 유의미한 결과를 도출하였습니다.

- **Performance Highlights**: 연구 결과, 많은 비전-언어 과제에서 무감독 상태에서도 비전과 언어 표현을 유의미하게 매칭할 수 있음을 보여주었습니다. 이는 주석 없는 상태에서도 이미지의 의미를 분류할 수 있는 가능성을 여는 혁신적인 성과입니다. 제시된 무감독 분류기는 이미지-텍스트 대응이 전혀 없이도 분류 정확도를 달성할 수 있는 효능을 입증하였습니다.



### IMPACT: A Generic Semantic Loss for Multimodal Medical Image Registration (https://arxiv.org/abs/2503.24121)
Comments:
          Submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). This is a preprint version and has not been peer-reviewed

- **What's New**: 이번 연구에서는 의료 영상에서의 정합(Registration)을 위한 새로운 유사도 측정 기법인 IMPACT(Image Metric with Pretrained model-Agnostic Comparison for Transmodality registration)를 소개합니다. IMPACT는 다양한 이미지 등록 프레임워크(예: Elastix, Voxelmorph)에 통합될 수 있도록 설계된 일반적인 의미적 유사도 메트릭입니다. 이 메트릭은 특정 작업에 대한 훈련 없이 의료 영상에서 추출된 딥러닝 기반 피쳐를 비교함으로써 여러 가지 모달리티에서 폭넓게 적용 가능하게 합니다.

- **Technical Details**: IMPACT는 대규모 사전 훈련된 TotalSegmentator 모델의 피쳐와 Segment Anything Model(SAM) 및 기타 대규모 세분화 네트워크를 통합하여 이점이 있습니다. 이 방법은 강건하고 확장 가능하며 효율적인 멀티모달 이미지 등록 솔루션을 제공합니다. 연구팀은 IMPACT 손실을 흉부 CT/CBCT 및 골반 MR/CT 데이터셋을 포함한 다섯 개의 도전적인 등록 작업에 대해 평가했습니다.

- **Performance Highlights**: 수치 메트릭(예: Target Registration Error, Dice Similarity Coefficient)은 기존 방법 대비 해부학적 정렬에서 유의미한 개선을 보였습니다. 질적 분석에서도 노이즈, 아티팩트 및 모달리티 변동에 강한 Robustness를 확인했습니다. IMPACT는 임상 및 연구 응용에서 등록 성능을 향상시키는 데 기여할 수 있는 유용한 도구로, 멀티모달 의료 영상의 주요 도전 과제를 해결하고 있습니다.



### Inductive Graph Representation Learning with Quantum Graph Neural Networks (https://arxiv.org/abs/2503.24111)
Comments:
          18 pages, 6 figures

- **What's New**: 본 논문은 고전적 Graph Neural Networks (GNNs)의 한계를 극복하기 위해 Quantum Graph Neural Networks (QGNNs)라는 새로운 프레임워크를 제안합니다. 기존의 QGNN들이 특정 그래프 설계에 따라 제한적이었던 반면, 본 연구에서는 GraphSAGE 접근법에 영감을 받아 유연한 설계를 통해 더 많은 그래프 문제에 적용 가능하도록 했습니다. 또한, 파라메트리즈된 양자 합성곱 및 풀링 레이어를 통합하여 고전적 방법과 양자 접근 방식을 효과적으로 연결하였습니다.

- **Technical Details**: 제안하는 QGNN 프레임워크는 인덕티브 노드 임베딩을 위해 네이버후드 노드를 샘플링하고 이를 합쳐서 특징 정보를 조합하는 어그리게이터 기능을 활용합니다. 이 프레임워크는 QM9 데이터셋을 사용하여 노드 회귀 작업을 실험하여 강력한 성능 메트릭스를 기록했습니다. 연구에서는 회로가 파라메트리즈된 양자 합성곱 레이어로 구성되며, 여기에 QCNN 아키텍처의 통찰력을 반영하여 설계되었습니다. 이전 QGNN 아키텍쳐에서 발생했던 barren plateau 문제를 해결하여 더욱 복잡한 그래프 기반 문제를 효과적으로 처리할 수 있는 능력을 보였습니다.

- **Performance Highlights**: 제안된 QGNN은 QM9 데이터셋을 통해 비상당한 분자 데이터를 모델링하는 데 성공하였으며, 고전 GNN과 비슷한 성능을 달성했습니다. 특히, 원자 수가 다양한 분자에서도 강한 일반화 능력을 보이며, 고전 GNN보다 약간 더 나은 성능을 보였습니다. 또한, QGNN 프레임워크는 쿼빗 수가 증가함에 따라 성능 저하가 없음을 수치적으로 확인했으며, 이는 스케일링을 통한 복잡한 데이터 처리 가능성을 제시합니다.



### New universal operator approximation theorem for encoder-decoder architectures (Preprint) (https://arxiv.org/abs/2503.24092)
Comments:
          34 pages

- **What's New**: 본 논문에서는 신경망을 이용한 연산자 근사(operator approximation)의 최신 이론을 제시합니다. 특히, 고전적인 encoder-decoder 아키텍처에 적용할 수 있는 보편적인 연산자 근사 정리를 새롭게 제안합니다. 기존의 연구들과 달리, 근사 연산자 시퀀스가 컴팩트 집합으로부터 독립적으로 선택될 수 있는 경우를 고찰합니다.

- **Technical Details**: 연산자 G: D⊆𝒳→𝒴를 정의하며, 여기서 𝒳와 𝒴는 무한 차원 노름 공간입니다. 저자들은 컴팩트 집합에서의 균일 수렴(uniform convergence)을 고려하고, 이를 통해 encoder-decoder 아키텍처에 특화된 새로운 근사 속성을 도입합니다. 이 연구는 DeepONets 및 그와 유사한 여러 아키텍처에 대해 기존의 연산자 근사 정리를 통합하고 확장합니다.

- **Performance Highlights**: 저자들은 제안된 정리를 통해 연구 결과가 다양한 연산자 학습 프레임워크에서 강한 수렴 성질을 보인다는 것을 증명합니다. 기존의 연산자 근사법과 비교하여 이 방법은 독립적으로 설정된 컴팩트 집합에서도 신뢰할 수 있는 성능을 보여줍니다. 이로 인해 신경망 기반의 연산자 근사 기술이 발전될 가능성이 커지고 있습니다.



### Controlled Latent Diffusion Models for 3D Porous Media Reconstruction (https://arxiv.org/abs/2503.24083)
Comments:
          58 pages

- **What's New**: 이번 연구는 지질학적인 3D 디지털 구조 재구성을 위한 혁신적인 계산 프레임워크를 소개합니다. Latent Diffusion 모델을 EDM 프레임워크 내에서 적용하여, 희소한 푸레 오르막 구조를 정확하게 구분할 수 있는 동시에 대표적인 부피를 포착합니다. 이를 통해 이전에 가능했던 것보다 더 큰 볼륨을 생성할 수 있으며, porosity와 같은 쉽게 계산할 수 있는 통계에 기반해 다수의 복잡한 특성들을 일관성 있게 표현할 수 있습니다.

- **Technical Details**: 본 연구는 3D 이진 지질 볼륨 생성을 위한 Variational Autoencoder (VAE)를 최적화하여, 정보를 최소한으로 손실하면서 효율적인 압축을 가능하게 합니다. 또한, Transformer 조건 레이어를 통해 두 점 상관 함수와 같은 복합 통계적 입력을 처리할 수 있게 하였습니다. 우리의 새로운 샘플링 방법론인 Controlled Unconditional Sampling은 목표 분포의 커버리지를 향상시킬 수 있는 효과적 접근법입니다.

- **Performance Highlights**: 이 프레임워크는 픽셀 공간에서의 확산 모델들보다 더 나은 생성 품질을 달성하며, (256^3)보다 크게 부피 재구성이 가능합니다. 즉, 기존의 방법들과 비교했을 때 수치 연산 요구가 현저히 줄어들어 고해상도의 물리적으로 현실적인 샘플을 생성할 수 있게 되었습니다. 이러한 개선은 지질학적 특성을 정밀하게 표현할 수 있는 기회를 제공합니다.



### Riemannian Multiplicative Update for Sparse Simplex constraint using oblique rotation manifold (https://arxiv.org/abs/2503.24075)
Comments:
          8 pages, 1 figure

- **What's New**: 본 논문에서는 sparse simplex constraints(희소 단순 조건)가 포함된 low-rank(저차원) 문제를 해결하기 위한 새로운 manifold optimization(다양체 최적화) 방법을 제안합니다. 이 방법은 oblique rotation manifolds(경사지 회전 다양체)를 활용하여 문제를 재구성하고 새로운 Riemannian optimization(리만 최적화) 기법을 도입합니다. 실험 결과, 제안된 방법이 기존의 Euclidean(유클리드) 방법보다 효과적임을 보여줍니다.

- **Technical Details**: 논문에서 다루는 핵심 요소는 nonnegativity(비부정성), sparsity(희소성), 그리고 sum-to-1 normalization(합이 1이 되도록 정규화)입니다. 우리는 ℓ1/2-quasi-norm(ℓ1/2 준 노름)을 이용하여 희소성을 강조하는 최적화 문제를 설정하고, Riemannian Multiplicative Update(RMU) 방법을 통해 이를 해결합니다. RMU는 Riemannian gradient descent(리만 기울기 하강) 방식의 수렴 특성을 유지하면서도 편리하게 simplex constraint(단순 조건)를 다룰 수 있는 장점을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 RMU 방법이 기존의 Euclidean 방법과 비교했을 때 계산 효율성에서 우수한 성능을 보였습니다. 특히, 이 방법은 제약 조건을 직접 최소화 과정에 통합하여 처리하는 점이 눈에 띕니다. 이는 기존 연구들에서 사용된 추가적인 투영(projection) 또는 이중 접근 방식과는 다른 접근법입니다.



### Physics-informed neural networks for hidden boundary detection and flow field reconstruction (https://arxiv.org/abs/2503.24074)
Comments:
          21 pages, 17 figures

- **What's New**: 이 연구는 유체 역학에서 희소한 관측치로부터 숨겨진 고체 경계를 동시에 감지하고 유동장을 재구성하는 새로운 과제를 다루고 있습니다. 제안된 방법은 physics-informed neural network (PINN) 프레임워크를 통해 정적 또는 이동하는 고체 경계의 존재, 형태 및 동작을 추론합니다. 이를 통해 고체 지역의 노슬립(no-slip)/노펜테이션(no-penetration) 경계 조건을 유지하면서 유체 역학의 보존 법칙을 보장합니다.

- **Technical Details**: 모델은 governing equations에 body fraction 파라미터를 통합하여 작동합니다. 이 방법은 부분적인 유동장 데이터만을 사용하여 알려지지 않은 유동장을 동시에 재구성하고 body fraction 분포를 추론함으로써 고체 경계를 드러냅니다. 연구는 다양한 시나리오에서 프레임워크의 유효성을 검사하였고, 여기에는 비압축 Navier-Stokes 및 압축 Euler 흐름이 포함됩니다.

- **Performance Highlights**: 결과는 숨겨진 경계를 정확히 감지하고, 누락된 유동 데이터를 재구성하며, 이동하는 객체의 경로 및 속도를 추정하는 능력을 보여줍니다. 데이터 희소성, 속도만 측정된 경우 및 잡음의 영향을 분석하여 추론 정확도의 변화를 확인하였고, 제안된 방법이 제한된 실험 또는 수치 데이터를 사용할 때 강력성과 다용성을 가지는 잠재력을 강조합니다.



### From Colors to Classes: Emergence of Concepts in Vision Transformers (https://arxiv.org/abs/2503.24071)
Comments:
          Preprint. Accepted at The 3rd World Conference on eXplainable Artificial Intelligence

- **What's New**: 이번 연구에서는 Vision Transformers (ViTs)의 레이어별 정보 처리 과정을 분석하여, 각 레이어에서 인코딩되는 개념들의 복잡도를 조사합니다. 기존의 연구는 주로 Convolutional Neural Networks (CNNs)에 중점을 두었으나, ViTs에 대한 레이어-wise 분석은 부족했던 점을 보완합니다. 연구 결과, ViTs가 초기 레이어에서 기본적인 특징을 인코딩하고 후반 레이어에서 더 복잡한 개념을 점차적으로 학습한다는 사실을 확인했습니다.

- **Technical Details**: 본 연구에서는 CLIP-dissect 방법을 사용하여 ViTs의 레이어별 학습 프로세스를 분석합니다. 이 방법은 네트워크의 각 뉴런에 관련된 개념을 식별하기 위한 신뢰성 있는 neuron labeling 기법을 제공합니다. 초기 레이어는 주로 색상과 질감과 같은 기본 특징을 인코딩하며, 후반 레이어에서는 물체와 자연 요소와 같은 더 전문화된 개념을 인코딩합니다.

- **Performance Highlights**: 연구 결과, ViTs는 초기 레이어에서 더 보편적인 개념을, 후반 레이어에서는 더욱 다채롭고 전문화된 개념을 인코딩하여 레이어별 특징 추출의 계층 구조를 드러냅니다. 또한, 특정 하향 작업에 대한 미세 조정(finetuning)은 인코딩된 개념의 수를 줄이고, 보다 관련성 있는 개념으로 이동하게 하는 경향이 있습니다.



### HACTS: a Human-As-Copilot Teleoperation System for Robot Learning (https://arxiv.org/abs/2503.24070)
- **What's New**: HACTS (Human-As-Copilot Teleoperation System)는 로봇 팔과 원격 조작 하드웨어 간의 양방향 실시간 동기화를 가능하게 하는 새로운 시스템입니다. 이 시스템은 자율 차량의 조향 휠처럼 작동하며, 사용자가 개입하면서도 향후 학습을 위한 행동 수정 데이터를 수집할 수 있도록 도와줍니다. HACTS는 전적으로 저렴한 3D 프린팅 부품과 재래식 모터를 사용하여 구현되어 접근성이 높고 확장 가능하다는 장점이 있습니다.

- **Technical Details**: HACTS는 로봇이 자율적으로 수행하는 작업에서 필요한 인간 개입을 실시간으로 조정할 수 있게 해주는 양방향 제어 메커니즘을 포함하고 있습니다. 이 시스템은 간단하지만 효과적인 하드웨어 설정으로, 사용자가 로봇과 지속적으로 피드백을 주고받을 수 있도록 합니다. 또한, HACTS는 로봇 학습에서 중요한 행동 수정 데이터를 수집하는 데 기여하여 더 나은 성능을 획득할 수 있도록 합니다.

- **Performance Highlights**: HACTS는 모방 학습(IL)과 강화 학습(RL) 작업에서 성능을 대폭 향상시키며, IL 복구 능력과 데이터 효율성을 높이는 데 중점을 둡니다. 특히, HACTS는 정적 및 동적 시나리오 모두에서 일반화 능력을 향상시키며, 복잡한 HITL RL 설정을 지원해 로봇이 인간 개입과 자율 행동에서 적응하고 학습할 수 있게 합니다. 이러한 발전은 인간-로봇 협력 및 데이터 수집의 새로운 가능성을 열어줍니다.



### Artificial Conversations, Real Results: Fostering Language Detection with Synthetic Data (https://arxiv.org/abs/2503.24062)
- **What's New**: 이 연구는 고품질 훈련 데이터를 수집하는 대신 LLMs를 사용하여 생성된 합성 데이터의 가능성을 탐구합니다. 특히, 이탈리아어 직업 광고에서 포괄적 언어 감지를 위한 작업에 초점을 맞추어, 데이터 부족 문제를 해결하기 위해 합성 데이터 생성을 위한 파이프라인을 제안하고 효과적인 프롬프트 전략과 텍스트 길이, 특수 과제에서의 목표 위치 같은 요소가 모델 성능에 미치는 영향을 분석합니다.

- **Technical Details**: 연구 방법론은 LLM 기반 합성 데이터 생성을 위한 프레임워크를 수립하고, 이탈리아어 직업 광고에서 비포괄적 언어를 탐지하는 데 활용됩니다. 이 방식은 실제 및 생성된 데이터를 결합한 합성 데이터셋 생성, 다양한 프롬프트 기법의 적용, 합성 데이터에 대한 모델의 파인 튜닝을 포함합니다. 데이터셋은 70-30 비율로 훈련 및 평가용으로 분할되어 모델의 일반화 능력을 검증합니다.

- **Performance Highlights**: 실험 결과, 합성 데이터로 훈련된 세밀화 모델이 실제 및 합성 테스트 데이터 세트에서 다른 모델보다 일관되게 우수한 성능을 보였습니다. 이는 LLMs의 합성 데이터 활용이 비용 효율적이고 확장 가능한 솔루션임을 보여줍니다. 본 연구는 이러한 합성 데이터 사용의 실질적 임팩트와 한계를 논의하면서, 포괄적 언어 탐지 작업에 대한 새로운 패러다임을 제시합니다.



### AutoML Algorithms for Online Generalized Additive Model Selection: Application to Electricity Demand Forecasting (https://arxiv.org/abs/2503.24019)
Comments:
          13 pages, 1 figure

- **What's New**: 이번 연구에서는 전력 수요 예측을 위한 모델로 일반화 가법 모델(Generalized Additive Models, GAM)과 상태 공간 모델(State-Space model)을 결합하여 적응형 모델을 효율적으로 최적화하는 방법을 제안합니다. 이 연구의 핵심은 Keisler의 DRAGON 패키지를 사용하여 GAM 수식(formula)과 적응형 하이퍼파라미터(hyperparameter)를 자동으로 선택할 수 있는 구조를 정의하는 것입니다. 이를 통해 프랑스의 전력 수요 예측에서 제안된 방법의 유효성을 입증하게 됩니다.

- **Technical Details**: GAM은 비선형 공변량 효과를 합산하여 랜덤 타겟 변수(Y)를 모델링하며, 예측 정확도는 이러한 합의 항 선택에 따라 크게 달라집니다. 이 연구에서는 P-IRLS(패널화된 반복 가중치 최소제곱) 방법을 사용하여 모델을 추정하며, GAM 수식 및 적응형 하이퍼파라미터의 선택이 중요한 역할을 합니다. AutoML 알고리즘을 활용해 이 모델의 선택과 하이퍼파라미터 최적화를 자동화함으로써, 후보 모델 풀(Ω)에서 최고의 적응형 모델(f⋆, Q⋆)을 탐색하는 방법을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 관찰된 프랑스의 전력 수요 데이터를 잘 예측하는 것을 보여주었습니다. AutoML 접근 방식은 전력 수요 예측의 효율성을 개선하고, 모델 선택의 자율성을 증가시켜 예측 정확도를 높였음을 관찰할 수 있었습니다. 이는 특히 전력 시스템의 안정성 유지와 효율적 생산 조절에 기여할 것으로 기대됩니다.



### Crossmodal Knowledge Distillation with WordNet-Relaxed Text Embeddings for Robust Image Classification (https://arxiv.org/abs/2503.24017)
- **What's New**: 본 논문에서는 unimodal 학생 모델의 성능을 향상시키기 위한 multi-teacher crossmodal knowledge distillation (KD) 프레임워크를 제안합니다. 이 방법론은 CLIP 이미지 임베딩과 학습 가능한 WordNet-relaxed 텍스트 임베딩을 계층적 손실(hierarchical loss) 구조 하에 통합하여 라벨 유출(label leakage)을 완화하고, 보다 다양한 텍스트 신호를 제공합니다. 실험 결과, 이러한 전략이 기존 방식보다 학생 모델의 성능을 크게 향상시키는 데 기여함을 보여줍니다.

- **Technical Details**: 제안하는 방법에서는 WordNet이라는 의미론적 관계를 가진 단어 데이터베이스를 활용하여 CLIP 텍스트 임베딩의 의미적 풍부성을 높입니다. 학생 모델은 단순히 이미지에만 의존하는 경우 여전히 교사의 텍스트와 이미지의 상호작용을 통해 학습을 강화하는 데 필요한 다양한 신호를 수신할 수 있습니다. 계층적 손실과 코사인 정규화(cosine regularization)를 도입하여 정확한 클래스 이름 사용을 피하고, 일반적인 시각적 모달리티 특징을 강조함으로써 보다 강건한 학습이 가능합니다.

- **Performance Highlights**: 우리의 방법은 여섯 개의 공공 데이터셋에서 state-of-the-art(SOTA) 또는 최소한 두 번째로 우수한 성능을 기록하며, 이는 crossmodal KD 분야에서의 유의미한 발전을 보여줍니다. 또한, WordNet 기반의 정규화가 강력한 시각적 특징에 대한 의존도를 높이고, 텍스트 암기를 줄이며, 새로운 텍스트 신호를 효과적으로 활용함을 입증하였습니다. 이러한 결과들은 제안하는 접근 방식이 실질적인 성능 향상에 기여함을 나타냅니다.



### Learning 3D-Gaussian Simulators from RGB Videos (https://arxiv.org/abs/2503.24009)
- **What's New**: 이 논문에서는 3D 물리 시뮬레이터인 3DGSim을 소개하며, 이는 다중 뷰 RGB 비디오에서 물체의 동역학을 끝에서 끝까지 학습하는 혁신적인 접근을 제공합니다. 3DGSim은 이미지를 3D Gaussiann (Gaussian) 파티클 표현으로 인코딩하고, 이러한 동역학을 transformer를 통해 전파하며, 3D Gaussian splatting을 사용하여 프레임을 렌더링합니다. 이 방법은 물리적 특성을 포인트-와이즈(latent vectors) 잠재 벡터에 포함시키면서, 명시적인 연결 제약 조건을 두지 않고도 다양한 물리적 행동을 캡처할 수 있도록 합니다.

- **Technical Details**: 3DGSim은 RGB 비디오에서 입자 상호작용을 직접 학습하여 3D Gaussian 포인트 클라우드로 장면을 표현합니다. kNN 대신 시간적 포인트 클라우드 직렬화(temporal point cloud serialization)를 활용하여 모델의 확장성을 크게 향상시킵니다. 동역학 모델과 함께 역 물리 렌더링을 공동으로 훈련하며, 이는 모션 사전(prior)을 3D Gaussian 표현에 직접 포함시키고 렌더링 직전에만 이 잠재 벡터를 splats로 맵핑합니다.

- **Performance Highlights**: 3DGSim은 다양한 물리적 행동을 포착할 수 있으며, 이는 강체(rigid), 탄성(elastic), 천과 같은 상호작용을 포함합니다. 또한 현실적인 조명 효과를 생성하며, 보지 못했던 다중체 상호작용과 새로운 장면 수정에 대해 일반화되는 것을 가능하게 합니다. 이로 인해, 3DGSim은 로봇의 의사결정에서 신뢰성을 높이며, 물리적 정확성을 강화하는 데에 기여할 가능성이 큽니다.



### Deep Nets as Hamiltonians (https://arxiv.org/abs/2503.23982)
Comments:
          19+7 pages

- **What's New**: 이번 논문에서는 랜덤 초기화된 Multi-Layer Perceptron (MLP)을 Hamiltonian으로 간주하고, 이 Hamiltonian이 유도하는 에너지 풍경의 특성을 연구합니다. 특히 무한 너비의 한계에서 거의 전역 최솟값의 구조에 초점을 맞추며, Replica Trick을 사용하여 주어진 에너지에서의 엔트로피를 정확하게 계산합니다. 랜덤 MLP로부터 유도된 Gibbs 분포를 통해 입력 간의 겹침을 설명하는 saddle point 방정식도 도출합니다.

- **Technical Details**: 이 연구는 깊은 신경망의 이론적 분석을 위해 파라미터가 무작위로 초기화된 경우의 결과를 살펴보는 기존의 접근과는 반대로, 입력을 anneal하고 파라미터를 quenched 상태로 고려합니다. 또한, linear activation function을 포함한 다양한 비선형 활성화 함수에 대한 saddle point 방정식을 수치적으로 및 정확히 해결합니다. 논문은 MLP의 너비가 무한대에 가까울 때의 Gibbs 측정을 탐구합니다.

- **Performance Highlights**: MLP의 활성화 함수에 따라 매우 다양한 행동 양상을 발견하였으며, 예를 들어 비선형성 중 하나인 sin의 경우에는 전체 replica symmetry breaking을 보였습니다. 반면, shallow tanh 및 ReLU 네트워크나 깊은 형태의 MLP에서는 replica symmetry를 유지하는 결과가 나타났습니다. 이러한 결과는 모델의 구조와 성능 간의 관계를 보다 깊이 이해하는 데 기여할 것입니다.



### The more the merrier: logical and multistage processors in credit scoring (https://arxiv.org/abs/2503.23979)
Comments:
          34 pages, 14 figures

- **What's New**: 이 논문은 공정한 머신러닝(fair ML) 기법을 금융 분야, 특히 신용 평가(credit scoring)에 적용하는 데 초점을 맞추고 있습니다. 저자들은 새로운 기술인 logical processors(LP)를 통해 여러 민감 변수를 처리하는 데 있어 기존 문헌의 방법론을 활용하려고 합니다. 또한, 다단계 프로세서(multistage processors, MP)를 통해 다양한 공정성 기법의 조합이 시너지 효과를 창출할 수 있는지를 탐구합니다.

- **Technical Details**: 이 연구는 두 가지 빈틈을 좁히기 위해 설계되었습니다. 첫 번째로, 여러 민감 변수를 동시에 처리할 수 있는 방법의 필요성이 제기되고 있으며, 두 번째로는 단일 단계에서 작동하는 기존의 공정성 프로세서가 여러 단계에서 조합된 하이브리드 방법의 필요성을 강조하고 있습니다. 저자들은 비트 연산을 활용한 logical processors를 제안하여 다변량 문제를 단일 변량 문제로 축소함으로써 설계의 유연성을 증대시키고자 합니다.

- **Performance Highlights**: 실험 결과, logical processors가 여러 민감 변수를 효과적으로 처리하는 데 적합한 방법임을 보여줍니다. 또한, multistage processors는 기존 방법들의 성능을 개선할 수 있는 가능성을 시사합니다. 이 연구는 공정성과 정확성 사이의 균형을 이해하는 데 중요한 기여를 하며, 향후 연구 방향에 대한 통찰도 제공합니다.



### Machine Learning-assisted High-speed Combinatorial Optimization with Ising Machines for Dynamically Changing Problems (https://arxiv.org/abs/2503.23966)
- **What's New**: 본 논문에서는 임베디드 Ising 머신을 사용한 조합 최적화 방법을 제안하여 실행 시간 동안 파라미터 조정 없이 다양한 문제를 고속으로 해결할 수 있도록 한다. 이 접근법은 빠른 데이터 처리와 최소한의 시스템 대기 시간을 요구하는 실시간응용 프로그램에서 특히 유용하다. 특히, TDMA 스케줄링 문제의 예시를 통해 이 시스템이 기존 방법에 비해 속도상의 이점을 갖고 있음을 입증하였다.

- **Technical Details**: 제안된 기술은 두 가지 주요 아이디어로 구성된다: 낮은 대기 시간을 위한 방법과 확장성 및 다양성 향상을 위한 방법이다. 시뮬레이티드 비퍼케이션(simulated bifurcation)이라는 양자 영감을 받은 알고리즘을 FPGA에 구현하고, J 행렬의 무손실 압축 인코딩 방식을 제안하여 CPU와 Ising 머신 간의 데이터 전송 시간을 단축하였다. 이로써 가장 적은 수의 시간 진화 단계로 높은 품질의 해를 얻을 수 있도록 개선하였다.

- **Performance Highlights**: TDMA 스케줄링을 위한 시스템 데모에서 제안된 시스템은 문제의 변화에 적응할 수 있는 능력을 보여주었다. 저Latency 최적화 방법은 Ising 머신의 처리 속도를 크게 향상시키고, 네트워크에서 필요한 모든 매개 변수를 자동으로 추정할 수 있도록 설계되었다. 결과적으로, 기존 방법에 비해 이 시스템은 더 빠르고 효과적인 문제 해결을 가능하게 하였다.



### Detecting Localized Density Anomalies in Multivariate Data via Coin-Flip Statistics (https://arxiv.org/abs/2503.23927)
- **What's New**: 이 논문에서는 두 개의 다변량 데이터 세트를 비교하여 지역 밀도 이상치를 탐지하기 위한 방법인 EagleEye를 소개합니다. 이 방법은 동전 던지기와 유사한 과정으로 설명될 수 있으며, 이웃 데이터 포인트의 밀접한 관계를 분석합니다. 확산적인 데이터에서 지역적인 차이를 감지하는 능력이 뛰어난 특징이 있습니다.

- **Technical Details**: EagleEye는 세 가지 단계로 구성되어 있으며, 첫 번째 단계에서 이상치 점수를 기반으로 의심스러운 점을 플래그합니다. 두 번째 단계에서는 가장 높은 이상치 점수를 가진 점들을 제거하며, 마지막 단계에서 중요한 밀도 이상치가 존재하는 특징 공간의 점들을 파악합니다. 이 방법은 두 개의 데이터 세트 간의 지역적인 밀도 차이를 탐지하는 데 중점을 두고 있습니다.

- **Performance Highlights**: EagleEye는 합성 및 실제 데이터 세트에 대한 실험을 통해 그 효과를 입증하였으며, 허용 오차 범위 내에서 극소수의 이상치도 정확히 식별할 수 있었습니다. 예를 들어, LHC 데이터에서 불과 0.3%의 데이터 내의 입자 붕괴 사건을 성공적으로 확인하였습니다. 이 방법은 다른 이상치 탐지 기법들과는 달리 복잡한 초기 분석이나 훈련 과정을 필요로 하지 않아 효율적입니다.



### Model Hemorrhage and the Robustness Limits of Large Language Models (https://arxiv.org/abs/2503.23924)
Comments:
          33 pages, 18 figures

- **What's New**: 이번 연구는 대규모 언어 모델(LLMs)의 성능 저하 현상을 '모델 출혈(model hemorrhage)'로 정의하며, 이는 매개변수 조정 및 아키텍처 변화에 의한 성능 감소를 포함합니다. 특히 다양한 LLM 프레임워크를 분석하여 주의(attention) 메커니즘의 방해, 압축 기법에서의 정보 손실, 디코딩 조정 시 예측 편차 확대와 같은 주요 취약성 패턴을 식별했습니다. 이러한 통찰력을 바탕으로 성능 저하를 방지하기 위한 세 가지 완화 전략을 제안하며, 이는 모델 안정성을 평가하기 위한 기초적인 메트릭을 구축합니다.

- **Technical Details**: 모델 출혈의 주요 원인은 네트워크의 아키텍처 및 파라미터 수정으로 나타납니다. 본 연구에서는 그래디언트 인식 가지치기(gradient-aware pruning), 동적 양자화 스케일링(dynamic quantization scaling), 디코딩 보정(decoding calibration) 등의 방법을 통해 성능 유지 방안을 제시합니다. 이 과정에서 Transformer 아키텍처가 수정 방법에 따라 성능 저하의 강도를 결정짓는 내재적 강건성 임계값을 가진다는 점도 밝혀졌습니다.

- **Performance Highlights**: LLM의 크기가 증가하면서, 이전의 단일 모델 아키텍처는 성능 및 효율성 면에서 이중 도전에 직면합니다. 최근에 출시된 MoE 모델 DeepSeek-R1은 학습 및 추론 과정에서 병렬화와 비용 최적화의 새 경로를 제시하며, 이는 다양한 작업 간 효율적인 전환을 가능하게 합니다. 이러한 발전은 대규모 언어 모델의 실용적 활용에 기여하며, 목표 지향적인 미래 연구를 위한 새로운 아이디어를 제공합니다.



### Certified Approximate Reachability (CARe): Formal Error Bounds on Deep Learning of Reachable Sets (https://arxiv.org/abs/2503.23912)
- **What's New**: 최근 딥러닝을 활용하여 연속 시간 동적 시스템의 도달 집합(reachable set)을 계산하는 접근법이 기존의 레벨 세트(level-set) 방법보다 주목받고 있다. 이러한 접근법은 차원의 저주(curse of dimensionality)를 극복하는 장점이 있다. 그러나 훈련 과정에서 학습된 도달 집합의 정확성을 보장하지 않는 한계가 있다. 이를 해결하기 위해, 본 연구에서는 도달 집합의 정확성과 훈련 손실(training loss) 간의 관계를 설정하는 epsilon-근사 해밀턴-자코비 부분 미분 방정식(epsilon-approximate Hamilton-Jacobi PDE)을 도입한다.

- **Technical Details**: 연구에서는 동적 시스템의 역 도달 집합(backward reachable set, BRS)과 역 도달 튜브(backward reachable tube, BRT)의 개념을 사용하여 안전한 상태를 식별하는 방법을 제시한다. BRS는 주어진 타겟 집합에 도달할 수 있는 모든 상태의 집합이며, BRT는 특정 시간 내에 타겟 집합에 도달할 수 있는 상태의 집합이다. 이러한 상태는 안전하지 않은 조건으로 이어질 수 있으므로 회피해야 한다. 본 논문은 이론적인 테두리 내에서 HJ 방정식의 유일한 해를 제공하고 불확실성을 처리할 수 있는 프레임워크를 확보한다.

- **Performance Highlights**: 이 연구의 주요 기여는 훈련 손실과 도달 집합의 정확도 간의 관계를 수립하고, HJ 기반의 손실 함수가 관심 영역 전역에서 정해진 임계값 내에 남도록 하는 정형화된 경계를 제공한다. 또한, ε-제약을 받는 신경망 기반의 가치 함수는 실제 가치 함수의 인증된 근사치를 제공하여 도달 집합의 과대 및 과소 근사를 가능하게 한다. 이렇게 함으로써, 본 연구는 연속 동적 시스템의 학습된 도달 집합에 대한 안전성 보장을 처음으로 제시한다.



### Feature learning from non-Gaussian inputs: the case of Independent Component Analysis in high dimensions (https://arxiv.org/abs/2503.23896)
- **What's New**: 이 논문에서는 독립 성분 분석(Independent Component Analysis, ICA)과 심층 신경망의 필터가 유사함을 강조하며, FastICA 알고리즘과 확률적 경량 하강(Stochastic Gradient Descent, SGD) 방식의 표본 복잡성을 비교했습니다. 이를 통해 ICA의 단순한 단계와 함께 효과적으로 특징 학습을 이해할 수 있는 원리를 제시하고 있습니다. 특히 FastICA의 표본 복잡성이 이제까지 알려진 것보다 더 낮다는 것을 보여줍니다.

- **Technical Details**: 이 연구는 기초적인 비가우시안(Non-Gaussian) 초입 데이터에서의 특징 학습 이론을 정량화하여, FastICA와 SGD의 샘플 복잡성(sample complexity)을 rigorously하게 분석합니다. FastICA는 입력 차원이 d일 경우, 비가우시안 방향을 재구성하기 위해 최소한 n ≳ d^4개의 샘플이 필요하고, 온라인 SGD는 d^2개의 샘플로 그 최적의 샘플 복잡성을 도달할 수 있음을 보여줍니다. 이러한 발견은 데이터 의존 방식으로 손실을 부드럽게 하는 방식에서 찾을 수 있습니다.

- **Performance Highlights**: 결과적으로, FastICA가 이미지넷 데이터셋(Imagenet)에서 검색 단계를 거치는 동안 비가우시안 데이터의 강한 구조가 성능 저하를 보완한다는 사실을 시연했습니다. 연구에서 제시된 바와 같이, FastICA는 특정 조건과 환경에서 traditional한 PCA보다 더 나은 필터를 학습하는 데 있어서 중요한 역할을 합니다. 또한, 논문의 실험 결과, vanilla 온라인 SGD는 FastICA보다 더 나은 성능을 보여주며 이론적 기반을 확립했습니다.



### ZeroMimic: Distilling Robotic Manipulation Skills from Web Videos (https://arxiv.org/abs/2503.23877)
Comments:
          ICRA 2025. Project website: this https URL

- **What's New**: 이번 연구에서는 다소 수집하기 어려운 특정 형태의 데모에 의존하는 기존의 로봇 조작(Imitation Learning) 방식을 벗어나, ZeroMimic이라는 새로운 시스템을 개발했습니다. ZeroMimic은 사전 녹화된 인간 비디오 데이터에서 로봇 조작 스킬 정책(skil policies)을 추출하여 다양한 작업에서 즉시 사용할 수 있는 능력을 보여줍니다. 이 시스템은 기존의 비디오 이해 및 고급 그립 탐지기(grasp affordance detectors)의 발전을 활용하여 여러 조작 작업을 수행할 수 있습니다.

- **Technical Details**: ZeroMimic은 인간 동작 비디오를 사용하여 로봇의 조작 기술을 학습하며, 이를 위해 작업의 두 가지 주요 단계를 정의합니다: 물체를 적절히 집는 잡기 단계(grasping phase)와 안정적으로 유지하며 조작하는 후작업 단계(post-grasp phase)입니다. 본 시스템은 EpicKitchens 데이터셋에서 학습하며, 다양한 환경에서 사용할 수 있도록 여러 로봇 구현체에 맞춰 쉽게 교환할 수 있는 독립적인 정책을 생성합니다. 주요 구성 요소로는 3D 맵을 유지하기 위한 구조에서 운동 시스템이 포함되어 있습니다.

- **Performance Highlights**: ZeroMimic은 실제 환경에서 9가지 다양한 기술 평가에서 71.0%의 성공률을 기록하였으며, 시뮬레이션에서는 73.8%의 성공률을 보여주었습니다. 또한, 준비된 웹 비디오에서 보지 못한 새로운 객체들에도 일반화할 수 있는 능력을 가지고 있습니다. 이와 같은 성과는 다양한 일상 환경에서 로봇이 수행할 수 있는 작업의 범위를 확장하는 데 기여할 것입니다.



### A Channel-Triggered Backdoor Attack on Wireless Semantic Image Reconstruction (https://arxiv.org/abs/2503.23866)
- **What's New**: 이 논문에서는 기존의 입력 수준 공격 대신, 채널을 트리거로 사용하는 새로운 백도어 공격 패러다임인 채널 트리거 백도어 공격(Channel-Triggered Backdoor Attack, CT-BA)을 제안합니다. 이 공격은 특정 무선 채널의 기본 물리적 특성을 활용하여, 기존의 입력 기반 공격보다 더 비밀스럽고 위협적입니다. 연구자들은 다양한 페이딩 분포의 채널 이득 및 서로 다른 전력 스펙트럼 밀도를 가진 채널 잡음을 트리거로 활용하는 방식을 제안하고 있습니다.

- **Technical Details**: CT-BA는 전송된 기호가 특정 백도어 채널을 통과할 때 활성화되는 방식으로 작동합니다. 이 접근법은 E2E 세멘틱 통신 시스템의 세 가지 주요 관찰상의 기초를 두고 있습니다: (1) 훈련 과정에서 채널 전송 함수는 미분 가능해야 하며, 채널 이득 및 가우시안 잡음이 모델 파라미터 업데이트에 영향을 미칩니다. (2) 채널 이득의 분포 특성과 잡음의 파워 스펙트럼 밀도가 다릅니다. (3) 실제 무선 통신에서는 채널 조건이 동적으로 변화하며 심각한 저하가 발생할 수 있습니다. 이러한 특성을 활용해 CT-BA는 실험적인 유연성과 실용성을 크게 향상시킵니다.

- **Performance Highlights**: CT-BA는 세 가지 데이터셋(MNIST, CIFAR-10, ImageNet)에서 ViT 기반의 통합 소스-채널 코딩(Joint Source-Channel Coding, JSCC) 모델에 대해 강력한 공격 성공률을 기록하였습니다. 이 공격은 고도의 은닉성을 유지하면서도 효과적인 성능을 발휘합니다. 논문에서는 이 공격에 대한 간단하면서도 효과적인 방어 메커니즘에 대해 논의하고 있습니다.



### Free Parametrization of L2-bounded State Space Models (https://arxiv.org/abs/2503.23818)
Comments:
          8 pages

- **What's New**: 새로운 연구에서, Structured State-space Models (SSMs)의 파라미터화 방법이 제안되었으며, 이 방법은 L2RU라는 명칭으로 불립니다. L2RU는 안정성과 강건성을 보장하는 새로운 아키텍처로, 모든 파라미터 값에 대해 안정성을 유지하는 L-bound를 강제하는 방식으로 설계되었습니다. 기존의 복잡한 제약 조건 없이 최적화를 가능하게 하여, 시스템 식별(system identification) 및 최적 제어(optimal control)의 응용에서 효용성을 높입니다.

- **Technical Details**: L2RU 아키텍처는 선형 시불변(discrete-time LTI) 시스템 오프닝과 비선형 함수를 중첩하는 구조로 되어 있으며, 시스템 이론과 볼록 최적화(convex optimization) 도구를 활용하여 파라미터화합니다. 이 구조는 모든 파라미터 값에서 L2-bound를 보장합니다. 또한, 장기 입력 시퀀스에 최적화된 초기화 전략을 통해 성능을 향상시키고 있습니다.

- **Performance Highlights**: L2RU는 시스템 식별 과제를 통해 그 우수성을 입증하였으며, 기존의 다른 모델과 비교했을 때 더 나은 성능을 보여주고 있습니다. 특히 시스템 안정성과 강건성에 대한 요구가 있는 응용 분야에서 필요로 하는 파라미터화를 제공함으로써 L2RU의 활용 가능성을 넓혔습니다. 본 연구 결과는 적대적 공격(adversarial attacks)에 대한 모델의 탄력성을 향상시키는 데 기여합니다.



### Adaptive Attention-Based Model for 5G Radio-based Outdoor Localization (https://arxiv.org/abs/2503.23810)
Comments:
          6 pages, 6 figures

- **What's New**: 이 논문은 동적 환경에서의 라디오 기반 위치 인식(Localization)을 위한 적응형 프레임워크를 제안합니다. 이는 단일 레이어 퍼셉트론(Single-Layer Perceptron, SLP)을 기반으로 한 라우터/스위칭 메커니즘과 얕은 주의(attention) 모델을 결합하여 다양한 조건에 최적화된 특수화된 모델 간의 원활한 전환이 가능합니다. 이렇게 함으로써 정확성, 계산 효율성 및 환경 변화에 대한 견고함을 달성할 수 있습니다.

- **Technical Details**: 제안된 모델은 낮은 복잡도의 세 가지 위치 인식 모델을 설계하고, 이 모델들은 특정 상황에 맞게 최적화되었습니다. 라우터는 실시간 입력 특성에 기반하여 가장 적합한 모델을 동적으로 선택합니다. 이 모델은 대규모 MIMO 기지국에서 수집한 실제 차량 위치 데이터로 검증되었으며, 다양한 배치 조건에 원활하게 적응할 수 있는 능력을 보여주었습니다.

- **Performance Highlights**: 이 논문에서 제안하는 적응형 모델은 전통적인 모델보다 더 작은 크기와 낮은 계산 복잡도를 통해 더 높은 정확성을 달성합니다. 전반적으로, 이 모델은 자율주행, 스마트 교통 관리 등 다양한 응용 프로그램을 위한 신뢰성 있는 위치 정보를 제공하는 데 기여합니다. 이는 도시 환경에서의 라디오 기반 위치 인식의 도전 과제를 해결하는 데 중요한 발전을 이룬 것입니다.



### Force-Free Molecular Dynamics Through Autoregressive Equivariant Networks (https://arxiv.org/abs/2503.23794)
Comments:
          25 pages total (19 manuscript, 6 SI). 5 figures in manuscript, 3 figures and 2 tables in SI

- **What's New**: 본 논문에서는 TrajCast라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 autoregressive equivariant message passing networks를 기반으로 하여 원자 위치 및 속도의 직접 업데이트를 통해 전통적인 수치적 통합의 제약을 극복합니다. 이를 통해 TrajCast는 기존의 MD 시뮬레이션과 비교하여 긴 시간 간격에서도 정확한 예측이 가능합니다.

- **Technical Details**: TrajCast는 원자 간 상호작용을 예측하는 머신러닝 모델인 MLIPs와 달리 원자 시스템의 다음 상태를 직접 출력합니다. Δt라 불리는 시간 간격은 기존 MD 시뮬레이션의 시간 간격보다 최소 10배 이상 크며, 이를 통해 TrajCast는 전체 상태 궤적을 생성하여 시간에 따라 변화하는 속성과 동적 속성을 계산할 수 있도록 합니다. 또한, 화학 결합에 대한 가정이 없어 반응 시스템을 모델링할 수 있는 잠재력을 지니고 있습니다.

- **Performance Highlights**: TrajCast는 4,000개 이상의 원자를 포함한 고체에서 하루에 15 ns 이상의 궤적 데이터를 생성할 수 있습니다. 다양한 시스템에서 기존 MD 시뮬레이션과의 뛰어난 일치를 보여주며, 구조적, 동적, 에너지 특성에서 좋은 결과를 나타냅니다. 이 프레임워크는 대규모 시뮬레이션을 효율적으로 수행할 수 있게 하여 물질 발견을 가속화하고 전통적인 시뮬레이션 및 실험으로는 탐구할 수 없는 물리현상을 탐색할 수 있도록 합니다.



### Evaluation of (Un-)Supervised Machine Learning Methods for GNSS Interference Classification with Real-World Data Discrepancies (https://arxiv.org/abs/2503.23775)
Comments:
          34 pages, 25 figures

- **What's New**: 이 논문은 자율주행차, 통행료 시스템 및 디지털 타코그래프와 같은 응용 프로그램에서 차량 위치 확인(vhicle localization)의 중요성을 강조합니다. 글로벌 내비게이션 위성 시스템(GNSS) 수신기를 사용하여 정확한 위치 결정을 시도하지만, 간섭 신호(interference signals)로 인해 이 과정이 방해받을 수 있음을 다룹니다. 최근의 머신러닝(ML) 기반 접근 방식이 이러한 간섭 모니터링에서 뛰어난 성능을 보였지만, 실제 환경에서의 적용 가능성은 아직 평가되지 않았습니다.

- **Technical Details**: 해당 연구에서는 독일의 두 고속도로 위치와 오스트리아의 지탈 알프스에서 수행된 대규모 측정 캠페인을 설명합니다. ML 기술의 효과적인 구현을 위해서는 현실적인 간섭 신호(noise)와 관련된 훈련 데이터셋이 필요하며, 이 데이터셋은 법적 제한으로 인해 생성하기 어려워하는 문제를 다룹니다. 또, 최신의 감독형 ML 방법을 평가하고 비감독 학습(unsupervised learning)을 위한 의사 레이블링(pseudo-labeling)의 적용 가능성을 제시합니다.

- **Performance Highlights**: 데이터 불일치로 인해 데이터셋 결합의 어려움이 크며, 이상 탐지(outlier detection), 도메인 적응(domain adaptation), 데이터 증강(data augmentation) 기법 등을 평가하여 모델의 적응 능력을 보여줍니다. 이 연구는 ML 기반 방법들이 실제 응용에서 어떻게 성능을 발휘하는지를 진단하고, 데이터 간 변화를 수용하는 모델링의 가능성을 제시합니다.



### THEMIS: Towards Practical Intellectual Property Protection for Post-Deployment On-Device Deep Learning Models (https://arxiv.org/abs/2503.23748)
Comments:
          To Appear in the 34th USENIX Security Symposium, August 13-15, 2025

- **What's New**: 본 논문에서는 THEMIS라는 자동화 도구를 제안하여, 포스트 배포된 on-device 딥러닝 모델에 워터마크를 삽입하는 가능성을 보여줍니다. 이 도구는 모델의 읽기 전용 및 추론 전용 특성을 고려하여 네 가지 단계를 통해 워터마크를 추가할 수 있게 해줍니다. 기존의 워터마킹 기술이 다양한 제한 때문에 적용하기 어려운 상황에서 THEMIS는 이를 해결하기 위한 혁신적인 접근 방식을 제시합니다.

- **Technical Details**: THEMIS는 현업의 딥러닝 모바일 앱에서의 모델 추출 및 워터마크 삽입과 관련된 세 가지 주요 도전 과제를 다룹니다. 첫 번째는 암호화된 형태의 모델에서 직접 추출이 어려운 점으로, 실행 추적 메커니즘을 통해 이를 해결합니다. 두 번째는 모델의 변경 가능성을 높이기 위해 Model Rooting 기법을 통해 writable 모델을 재구성하며, 세 번째는 데이터 부족에 대응하기 위해 FFKEW라는 훈련이 필요 없는 독창적인 워터마킹 알고리즘을 도입합니다.

- **Performance Highlights**: THEMIS는 MobileNetV2, InceptionV3 및 EfficientNetV2 모델에 대해 80% 이상의 워터마크 성공률을 달성하며, 다양한 평가에서 모든 기준을 초과하는 성능을 보입니다. 또한, Google Play의 403개 실세계 DL 앱을 대상으로 한 조사에서는 81.14%의 성공률로 워터마크를 수용할 수 있었음을 보여, THEMIS의 실용성과 효율성을 입증합니다.



### Short-video Propagation Influence Rating: A New Real-world Dataset and A New Large Graph Mod (https://arxiv.org/abs/2503.23746)
- **What's New**: 이 논문은 Short-video Propagation Influence Rating (SPIR) 작업을 제안하며, 단기적인 인기 예측을 넘어서서 긴 시간에 걸친 비디오의 전파 영향을 추정하려는 새로운 접근 방식을 소개합니다. 이는 사용자의 다양한 상호작용 정보를 고려하여 단일 지표에 의존하지 않고, 실질적인 전파의 영향을 평가하는 것을 목표로 합니다. 또한 최초의 크로스 플랫폼(short-video propagation) 데이터셋인 XS-Video를 도입하여, 5개 주요 플랫폼에서 수집한 비디오 데이터로 구성된 신뢰할 수 있는 근거를 제공합니다.

- **Technical Details**: XS-Video 데이터셋은 총 117,720 개의 비디오와 381,926 개의 샘플, 535 개의 주제를 포함하고 있으며, 비디오의 전파 영향력은 0에서 9까지의 등급으로 주어집니다. 논문은 또한 대규모 그래프 모델(NetGPT)을 제안하여, 서로 다른 형태의 그래프 구조 데이터를 처리하는 것과 동시에 대규모 언어 모델(LLM)의 추론 능력을 결합합니다. NetGPT는 이 새로운 세 단계 훈련 메커니즘을 기반으로 하여, 단기 비디오 전파 그래프를 이해하고 분석함으로써 전파 영향력을 예측할 수 있습니다.

- **Performance Highlights**: 실험 결과, NetGPT는 XS-Video 데이터셋에서 기존의 최첨단 방법들(GNNs, LLMs 및 멀티모달 LLMs)에 비해 월등한 성능을 보여주었습니다. 기존 방법들이 비디오 전파 분석에서 비효율적이라는 점을 보여주며, SPIR 작업에 대한 보다 정확한 접근이 필요하다는 점을 강조합니다. 이 모델은 특히 복잡한 그래프 구조와 비디오의 이종 특성을 포착하여 전파 영향력 수준을 예측하는 데 큰 강점을 보이며, 다양한 응용 분야에서 유용할 것입니다.



### Integral regularization PINNs for evolution equations (https://arxiv.org/abs/2503.23729)
- **What's New**: 최근 발표된 논문에서는 진화 방정식(evolution equations)의 장기 통합(long-time integration)에서의 정확도를 높이기 위해 새로운 방법인 적분 정규화 PINNs(IR-PINNs)를 제안합니다. 이 방법은 손실 함수에 적분 기반의 잔차 항(residual term)을 포함시켜, 시간 간격을 작은 하위 간격으로 나누고 특정 시간 구간에 제한을 두어 temporal dynamics의 해상도와 상관성을 향상시킵니다.

- **Technical Details**: IR-PINNs는 기존의 physics-informed neural networks(PINNs)를 기반으로 하여, 이론적으로 시간에 대한 의존성을 보다 잘 반영할 수 있게끔 합니다. 이 접근법은 적응형 샘플링(adaptive sampling)을 활용하여 특정 지역에서의 collocation points의 분포를 동적으로 재조정하여 기울기가 급격히 변화하는 지역에서도 높은 정확성을 보장합니다. 따라서, IR-PINNs는 강력한 computational efficiency를 바탕으로 진화 방정식을 효과적으로 해결합니다.

- **Performance Highlights**: 수치 실험을 통해 IR-PINNs는 기존의 PINNs 및 다른 최신 방법들과 비교하여 장기적인 동작(long-time behaviors)을 우수하게 캡처함을 보여주었습니다. 이 연구는 IR-PINNs가 진화 방정식에서 보다 견고하고 정확한 해결책을 제공한다는 것을 검증하며, 이는 물리적, 생물학적 및 공학적 현상을 모델링하는 데 필요한 정확한 솔루션에 기여할 것으로 기대됩니다.



### MKA: Leveraging Cross-Lingual Consensus for Model Abstention (https://arxiv.org/abs/2503.23687)
Comments:
          To appear in Building Trust Workshop at ICLR 2025

- **What's New**: 본 연구에서는 LLMs(Large Language Models)의 신뢰성을 보장하기 위한 새로운 접근 방식을 제안합니다. 다국어(multilingual) 지식을 활용하여 모델의 응답에 대한 신뢰도를 조정하고, 불확실한 경우에는 응답을 삼가하도록 하는 파이프라인을 개발했습니다. 이 방법은 다양한 언어 모델에 적용되며, LLMs의 객관성을 높이기 위한 중요한 발판이 될 것으로 기대됩니다.

- **Technical Details**: MKA(Multilingual Knowledge Abstention) 파이프라인은 모델의 질문에 대한 응답을 여러 언어로 번역하여, 각각의 언어에서 생성된 응답을 비교하고 그 신뢰도를 조정하는 과정으로 이루어집니다. 심지어 저자원 언어(low-resource language)에서도 LLM의 성능을 향상시킬 수 있는 방법을 제시합니다. 이러한 방법은 신뢰성을 보장하기 위해 예를 들어 cosine similarity와 같은 기법을 활용하여 응답 간의 유사성을 측정합니다.

- **Performance Highlights**: 다국어 파이프라인을 통해 벵골어(Bengali)에서는 71.2%의 정확도 향상을 기록했으며, 영어에서도 15.5% 향상이 있음을 발견했습니다. 이는 특정 언어에서의 LLM 신뢰도 개선이 가능하다는 점을 시사합니다. 이러한 결과는 앞으로 더 많은 언어 및 모델에 대해 적용될 수 있는 잠재력을 보여줍니다.



### Scalable Geometric Learning with Correlation-Based Functional Brain Networks (https://arxiv.org/abs/2503.23653)
- **What's New**: 본 논문은 기능적 뇌 네트워크의 상관 행렬을 정밀하게 분석하는 새로운 기하학적 프레임워크를 제안합니다. 기존의 방법들이 겪었던 계산 비효율성과 수치적 불안정성을 극복하기 위해, 상관 행렬을 유클리드 공간에 매핑하는 미분 동형변환(diffeomorphic transformations)을 사용하여 실질적인 분석을 가능하게 합니다. 이 접근 방식은 기능적 연결 분석을 위한 기존의 머신러닝 알고리즘과 통합되어, 대규모 네트워크 연구에 적합한 성능을 제공합니다.

- **Technical Details**: 상관 행렬은 대칭 양의 정부호(symmetric positive-definite, SPD) 행렬 클래스에 속하지만, 일반적인 통계 분석 방법은 벡터 공간 구조에 기반하고 있어 적용이 어렵습니다. 본 논문에서는 상관 행렬 공간을 획득하는 새로운 기하학적 구조를 도입하고, 이를 통해 각각의 알고리즘을 기존 학습 패러다임에 직접 적용할 수 있도록 합니다. 이러한 방식은 비용이 많이 드는 수치적 작업을 단 한 번의 변환에 한정시킬 수 있어 계산 효율성을 극대화합니다.

- **Performance Highlights**: 시뮬레이션 연구 결과, 제안된 방법이 기존의 다중 다양체 기반 접근 방식에 비해 계산 속도를 현저히 향상시키면서 정확도 또한 높다는 것이 입증되었습니다. 실제 신경영상 데이터를 사용한 실험에선 행동 점수 예측, 안정 상태 fMRI에서의 주체의 지문 인식, 전기 생리학적 데이터의 가설 검증 등에서 이 프레임워크의 유용성이 입증되었습니다. 따라서 제안된 MATLAB 툴박스를 통해 기능적 뇌 네트워크 연구에 있어 상관 기하학의 적용을 촉진할 수 있습니다.



### Learning a Single Index Model from Anisotropic Data with vanilla Stochastic Gradient Descen (https://arxiv.org/abs/2503.23642)
- **What's New**: 본 연구에서는 비등방성(anisotropic) 가우시안 입력 데이터를 통해 Single Index Model (SIM)을 학습하는 방법을 조사합니다. Vanilla Stochastic Gradient Descent (SGD)를 사용하는 과정에서 데이터의 공분산 구조에 적응하는 SGD의 능력을 입증하며, 기존의 구형 SGD와 비교하여 더 일반적인 공분산 구조에서도 학습할 수 있음을 보여줍니다. 이 연구는 간단한 알고리즘인 vanilla SGD가 비등방성 공분산 구조를 학습할 수 있음을 직접적으로 보여주는 첫 번째 연구로, 이는 더 복잡한 입력 데이터 분석으로의 확장을 위한 초석이 될 것입니다.

- **Technical Details**: 연구는 SIM의 학습 동역학을 분석하는 데 중점을 두며, vanilla SGD가 데이터를 통한 공분산 구조에 자동으로 적응한다는 사실을 강조합니다. 우리는 샘플 복잡성(sample complexity)을 측정하는 새로운 상하계(upper and lower bounds)를 도출하며, 이는 공분산 구조의 특성에 의해 결정됩니다. 이 연구는 공분산 행렬(Q)의 구조적 요소와 단일 지수(w*) 간의 정렬, 링크 함수(f)의 정보 지수의 함수로 T(반복 횟수)를 특성화합니다.

- **Performance Highlights**: 본 연구의 주요 성과 중 하나는 SGD가 T 반복 후에도 단일 지수(w*)와의 상관관계(correlation)를 유지한다는 것입니다. 우리는 또한 Correlated Statistical Query (CSQ) 하한을 수립하여 함수로서의 공분산 구조(Q)의 효과적인 측정이 평균적으로 올바르다는 것을 제안합니다. 이러한 이론적 결과는 숫자 시뮬레이션의 도움으로 뒷받침되었으며, 공분산 구조에 따라 SGD의 학습 동역학을 이해하는 데 있어 중요한 통찰력을 제공합니다.



### A Constrained Multi-Agent Reinforcement Learning Approach to Autonomous Traffic Signal Contro (https://arxiv.org/abs/2503.23626)
Comments:
          Submitted to ACM Journal for Autonomous Transportation Systems

- **What's New**: 본 논문에서는 Adaptive Traffic Signal Control (ATSC) 문제를 제약 조건이 있는 Multi-Agent Reinforcement Learning (MARL) 문제로 설정하고, 이를 위한 새로운 알고리즘인 Multi-Agent Proximal Policy Optimization with Lagrange Cost Estimator (MAPPO-LCE)를 제안합니다. MAPPO-LCE는 Lagrange multiplier 방법을 통합하여 보상과 제약 간의 균형을 맞추며, 비용 예측기를 사용하여 안정적인 조정을 제공합니다.

- **Technical Details**: 제안된 알고리즘은 GreenTime, GreenSkip, PhaseSkip의 세 가지 제약을 도입하여 실제 시나리오에 부합하지 않는 교통 정책을 처벌합니다. MAPPO-LCE는 기본적인 MARL 알고리즘 세 가지와 비교하여 모든 환경과 교통 제약 조건에서 우수한 성과를 보여주며, 이는 대규모 교통 신호 제어를 위한 효과적인 접근 방식을 제시합니다.

- **Performance Highlights**: 세 가지 실세계 데이터셋에서 MAPPO-LCE는 기본 MARL 알고리즘인 MAPPO, IPPO, QTRAN을 각각 12.60%, 10.29%, 13.10% 개선하여 성능을 입증하였습니다. 이러한 결과는 제약이 있는 MARL이 실제 교통 네트워크에서 ATSC 방법을 배포하는 데 있어 소중한 도구가 될 수 있음을 시사합니다.



### Interpretable Machine Learning in Physics: A Review (https://arxiv.org/abs/2503.23616)
- **What's New**: 본 논문은 머신러닝의 해석 가능성이 물리학에 응용되는 중요한 역할을 조명하고 있습니다. 머신러닝 알고리즘의 발전과 함께, 데이터에서 패턴을 학습하고 이를 기초로 새로운 과학적 발견을 촉진하는 과정에서 해석 가능성의 필요성이 강조됩니다. 특히, 인공지능(AI)이 인간 능력 너머의 발견을 가능하게 할 때, 이러한 모델의 개념과 예측을 이해할 수 있는 것이 필수적이라고 언급하고 있습니다.

- **Technical Details**: 해석 가능한 머신러닝(Interpretable Machine Learning)은 머신러닝 모델이 내리는 결정을 인간이 이해하고 설명할 수 있는 언어로 번역하는 과정입니다. 이 과정은 본질적으로 이해할 수 있는 알고리즘과 '블랙박스' 모델 간의 구분을 포함합니다. 간단한 모델은 더 쉽게 해석할 수 있지만, 복잡한 인공지능 알고리즘은 예측 결과를 도출하는 방법을 이해하는 데 어려움이 있으며, 이는 다양한 해석 도구의 개발을 요구합니다.

- **Performance Highlights**: 해석 가능성은 과학적 발견을 위한 핵심 요소로 작용하여, 연구자들이 머신러닝 모델의 예측을 이해하고 검증할 수 있게 합니다. 이를 통해 모델의 결정이 의미 있는 패턴에 기반하고 있음을 확인할 수 있으며, 이는 신뢰성과 재현성을 강화합니다. 또한, 해석 가능한 모델은 연구자들이 잘못된 예측의 원인을 분석하고 모델을 조정하여 개선하는 데 도움을 줍니다.



### An Organizationally-Oriented Approach to Enhancing Explainability and Control in Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2503.23615)
- **What's New**: 이 논문에서는 Multi-Agent Reinforcement Learning(MARL) 과정에 조직적 역할과 목표를 명시적으로 통합하는 새로운 프레임워크를 소개합니다. 이 프레임워크는 에이전트들이 조직적 제약을 충족하도록 안내하여 에이전트 행동의 설명 가능성과 제어를 향상시키고자 합니다. 일반적으로 MARL의 문헌은 개별 에이전트에 초점을 맞추고 있지만, 이 연구는 조직 수준에서의 협업 행동을 탐구하고 있습니다.

- **Technical Details**: MOISE+MARL 프레임워크는 Decentralized Partially Observable Markov Decision Process(Dec-POMDP) MARL 프레임워크와 조직적 모델인 ℳOISE^+를 통합합니다. 사용자는 역할이나 목표의 논리를 수동으로 정의할 수 있으며, 이로 인해 에이전트의 행동을 설명하는 기대 행동을 규정할 수 있습니다. 또한 Trajectory-based Evaluation in MOISE+MARL(TEMM) 방법을 통해 관측된 경로에서 암시적 역할과 목표를 일반화할 수 있습니다.

- **Performance Highlights**: 모든 환경에서 역할이 부여된 에이전트들은 TEMM의 정량적 측정에 따라 예상대로 행동하였습니다. TEMM에 의해 추론된 역할과 임무는 사전 정의된 사양과 밀접하게 일치하였으며, MOISE+MARL의 내부 일관성을 보여줍니다. 정책 기반 및 액터-크리틱 알고리즘은 에이전트가 안정적인 정책을 유지하도록 안내하는 데 특히 적합한 것으로 나타났으며, 이것은 TEMM의 안정적인 암시적 조직 생성을 가능하게 합니다.



### Space of Data through the Lens of Multilevel Graph (https://arxiv.org/abs/2503.23602)
Comments:
          18 pages, 11 figures, ITADATA 2024 conference

- **What's New**: 이 연구는 데이터스페이스(dataspaces)의 복잡성을 해결하기 위해 새로운 데이터 구조를 도입합니다. 제안된 다층 그래프(multilevel graph)는 지역(local)에서 글로벌(global) 수준까지 여러 수준의 추상을 나타낼 수 있도록 설계되었습니다. 이 구조는 데이터 분석을 위한 강력한 프레임워크를 제공하며, 비구조화된 데이터(unstructured data)뿐만 아니라 구조화된 데이터(structured data)에도 적용이 가능합니다.

- **Technical Details**: 다층 그래프는 서로 관련된 그래프의 계층(hierarchy)으로 구성되며, 각 수준은 동일한 데이터의 다른 추상을 나타냅니다. 이 구조는 그래프 이론에서 알려진 축소(contraction) 개념을 반복적으로 적용하여 슈퍼노드(supernodes)로 축소하여 만들어집니다. 또한, 슈퍼노드를 그래프로 다시 확장(decontraction)하는 기능은 추적 가능성을 제공합니다.

- **Performance Highlights**: 예비 결과는 꿈의 보고서(dream reports)라는 실제 시나리오를 통해 제안된 접근 방식의 강점과 약점을 강조합니다. 제안된 축소 작업은 정보를 더 추상적이고 간결한 형태로 캡슐화하여 전역 정보를 제거하려는 시도로, 원래의 세부 정보를 추적할 수 있는 능력을 유지합니다. 그러나 그래프 조작에 관련된 계산 복잡성은 여전히 중요한 도전 과제로 남아있습니다.



### Exploring GPT-4 for Robotic Agent Strategy with Real-Time State Feedback and a Reactive Behaviour Framework (https://arxiv.org/abs/2503.23601)
- **What's New**: 이 연구는 GPT-4를 사용하여 휴머노이드 로봇이 시뮬레이션 및 실제 환경에서 작동하도록 하는 새로운 행동 방법을 제안합니다. 우리는 LLM이 제시한 목표에 기반하여 작업을 수행하도록 로봇을 조작할 수 있는 방법을 개발했습니다. 이를 통해 로봇이 매끄러운 전환과 안전성을 바탕으로 사용자 요청을 처리할 수 있게 하였습니다. 이 연구는 반복적인 사용자 피드백을 통해 LLM의 작업 요청을 업데이트하는 방식으로 진행됩니다.

- **Technical Details**: 우리는 LLM을 실제 로봇 시나리오에 결합하여 혁신적인 트리 기반 행동 시스템을 활용합니다. 로봇은 장기 목표를 설정하고, 실시간 메시지 전송 시스템으로부터의 고주파 상태 피드백을 고려하여 행동을 수행합니다. 또한, 이 방법론은 변화하는 환경에서 직접적인 피드백을 반영할 수 있는 반응적 작업 레이어를 동적으로 구성하는 데 중점을 두고 있습니다. 이는 일반적인 기계 학습 기법과는 달리, 다양한 시나리오를 처리할 수 있는 로봇의 가능성을 확장합니다.

- **Performance Highlights**: 우리의 실험은 이 접근 방식이 모든 요청을 원활하게 처리할 수 있는 작업 결과를 도출한다는 것을 보여주었습니다. 대부분의 사용자 요청이 다양한 목표 시간의 범위 내에서 성공적으로 달성되었습니다. 이 연구는 LLM의 출력 유효성을 해결하고, 고주파 요청에 대한 시스템이 효과적으로 작동하도록 개선된 성과를 나타냅니다. 또한, 휴머노이드 로봇의 다양한 작업에서 LLM 행동의 성능을 실험하였습니다.



### Online Convex Optimization and Integral Quadratic Constraints: A new approach to regret analysis (https://arxiv.org/abs/2503.23600)
- **What's New**: 이 논문에서는 강하게 볼록하고 Lipschitz 매끄러운 목적함수에 대한 1차 제약 온라인 볼록 최적화 알고리즘의 동적 응징(dynamic regret) 분석을 위한 새로운 접근법을 제안합니다. 특히, 피드백이 있는 선형 동적 시스템과 1차 오라클의 상호 연결성으로 표현될 수 있는 다양한 첫 번째 알고리즘에 적용 가능한 일반적인 분석을 제공합니다. 변동 적분 제약(variational Integral Quadratic Constraints, vIQCs)의 개념을 도입하여 시간 가변 단조 연산자에 대한 IQCs의 일반화를 수행합니다.

- **Technical Details**: 온라인 볼록 최적화(Online Convex Optimization, OCO) 알고리즘을 시간 가변 1차 오라클과 연결된 선형 동적 시스템으로 모델링합니다. 일반적으로 사용되는 첫 번째 알고리즘의 동적 응징을 제한하는 경계는 시간 가변 최적 해결책의 경로 길이 및 목적 함수의 변동을 포함하여 잡히고, 이를 통해 일반적인 분석 방법과 차별화됩니다. 변동 적분 제약(vIQCs)을 사용하여 시간 가변 오라클의 특성을 효과적으로 처리합니다.

- **Performance Highlights**: 본 논문은 특정의 경계 관리가 기존 OCO 결과와는 달리 기울기 유한성 또는 제한된 가능 집합의 가정을 요구하지 않는 점에서 우수합니다. 수치 분석은 응징이 함수 클래스 조건 수의 의존성을 포착하는 능력을 보여줍니다. 이 연구는 사전의 가정이 약할수록 분석의 일반성과 다양한 알고리즘 비교를 통해 얻어지는 통찰을 강조합니다.



### Multi-Objective Optimization and Hyperparameter Tuning With Desirability Functions (https://arxiv.org/abs/2503.23595)
- **What's New**: 이 논문은 다목적 최적화(multi-objective optimization)와 다목적 하이퍼파라미터 조정(multi-objective hyperparameter tuning)을 위한 바람직성 함수(desirability function) 접근법에 대한 소개를 제공합니다. 주요 내용은 Kuhn(2016)의 연구에 기반하여, `Python` 언어로 구현된 Kuhn의 `R` 패키지 `desirability`에 대해 설명하고 있습니다.

- **Technical Details**: 바람직성 함수는 다양한 목표를 동시에 고려해야 하는 최적화 문제에 적용되며, 이 논문에서는 직접 최적화(direct optimization)와 대체 모델 기반 최적화(surrogate model-based optimization)를 포함한 사례를 설명합니다. 또한, `Python` 패키지인 `spotdesirability`는 후속 매개변수 최적화(sequential parameter optimization) 프레임워크의 일환으로 사용 가능합니다.

- **Performance Highlights**: 세 가지 예제를 통해 바람직성 함수가 고전적 최적화(classical optimization), 대체 모델 기반 최적화(surrogate-model based optimization), 하이퍼파라미터 조정(hyperparameter tuning)에서 어떻게 활용되는지를 시연합니다. 이 예제들은 실제 문제에의 적용 가능성을 보여줍니다.



### DASH: Detection and Assessment of Systematic Hallucinations of VLMs (https://arxiv.org/abs/2503.23573)
- **What's New**: 이 논문에서는 오픈 월드(open-world) 환경에서 비전-언어 모델(vision-language models, VLMs)의 체계적 환각(object hallucinations)을 탐지하고 평가하기 위한 자동화된 파이프라인인 DASH(Detection and Assessment of Systematic Hallucinations)를 제안합니다. 기존의 벤치마크가 작고 레이블이 있는 데이터셋에 의존하여 제한된 결과를 제공한 반면, DASH는 자연 사진 매니폴드(natural image manifold)를 최적화하여 VLM을 혼란스럽게 하는 이미지를 생성하고 이를 통해 환각 물체들을 식별합니다.

- **Technical Details**: DASH는 두 가지 주요 구성 요소인 DASH-OPT와 DASH-LLM로 구성됩니다. DASH-OPT는 이미지 기반의 검색 방식을 통해 VLM이 존재하지 않는 물체를 환각하게 만드는 이미지를 생성하도록 최적화됩니다. 반면, DASH-LLM은 대규모 언어 모델(LLM)에서 생성된 쿼리를 바탕으로 기능하며, 이 두 가지 접근 방식을 통해 이미지와 텍스트 쿼리를 탐색하여 FP-hallucinations를 유발하는 실체 이미지를 찾습니다.

- **Performance Highlights**: DASH를 PaliGemma와 여러 LLaVA-NeXT 모델에 적용한 결과, 19,000개 이상의 클러스터와 950,000개 이상의 이미지를 발견했습니다. 찾아낸 환각 이미지는 다른 VLM으로 성공적으로 이전되며, DASH를 통해 생성된 특정 이미지로 PaliGemma를 미세 조정하면 객체 환각 문제를 완화할 수 있음을 보입니다. 또한, DASH-B라는 새로운 벤치마크를 제안하여 현재 VLMs의 평가를 보다 신뢰할 수 있게 할 수 있음을 보여주고 있습니다.



### Addressing Model Overcomplexity in Drug-Drug Interaction Prediction With Molecular Fingerprints (https://arxiv.org/abs/2503.23550)
Comments:
          Accepted to the GEM Workshop at ICLR 2025

- **What's New**: 최근의 딥러닝 모델들은 약물-약물 상호작용(drug-drug interactions, DDIs)을 정확히 예측하는 데 어려움을 겪고 있습니다. 본 연구에서는 분자 표현(molecular representations), 즉 Morgan fingerprints (MFPS), 그래프 기반 임베딩(graph-based embeddings), 그리고 MoLFormer의 transformer-derived embeddings를 사용하여 간단하면서도 효과적인 접근 방식을 탐구하였습니다. 이러한 방법은 DrugBank DDI 데이터셋의 평가에서 경쟁력 있는 성능을 보이며, 간단한 분자 표현이 충분하다는 것을 입증하였습니다.

- **Technical Details**: Morgan fingerprints는 분자 구조를 고정된 길이의 이진 벡터로 인코딩하여 특정 하위 구조의 존재를 나타냅니다. 또한, 그래프 신경망(graph convolutional networks, GCNs)은 분자를 그래프로 표현하여 원자와 화학 결합을 노드와 엣지로 정의합니다. 여성 음악의 작곡자인 MoLFormer는 SMILES 문자열을 입력으로 처리하여 화학적 문맥을 포착합니다. 이 모델은 DrugBank 데이터를 기반으로 DDI 분류 및 DDA 회귀 작업을 평가하는 모듈식 신경망으로 설계되었습니다.

- **Performance Highlights**: 평가 결과, MFPS는 Unseen DDI 분할에서 가장 높은 AUROC(99.4%) 및 AUPR(98.4%)를 기록하며, MoLFormer보다 높은 성능을 보였습니다. Unseen 1 Drug 분할에서는 MFPS와 사전 학습된 GCN 임베딩이 유사한 정확도를 보여주었으나, 이들은 더 일관된 예측을 제공하여 안정성 면에서 우수함을 나타냅니다. 전체적으로, MFPS와 GCN 임베딩은 정확성과 안정성 사이의 균형 잡힌 타협을 제공하며, DDI 예측에 있어 간단하고 해석 가능한 분자 표현의 통합 가치를 강화합니다.



### Question-Aware Knowledge Graph Prompting for Enhancing Large Language Models (https://arxiv.org/abs/2503.23523)
- **What's New**: 본 연구에서는 질문-인지 지식 그래프 프롬프팅(Question-Aware Knowledge Graph Prompting, QAP)이라는 새로운 접근 방식을 제안합니다. QAP는 질문 임베딩을 GNN 집계 과정에 통합하여 KG(coherent Knowledge Graph) 관련성을 동적으로 평가함으로써, 이론적 한계를 극복하려고 합니다. 또한, QAP는 글로벌 어텐션을 사용하여 답변 옵션 간의 관계를 포착하여 소프트 프롬프트의 지식을 강화합니다.

- **Technical Details**: QAP는 세 가지 주요 단계로 구성됩니다: (i) 서브그래프 검색(Subgraph Retrieval), (ii) 질문-인지 이웃 집계(Question-Aware Neighborhood Aggregation, QNA), (iii) 글로벌 어텐션 기반 프롬프팅(Global Attention-Derived Prompting, GTP). QNA는 질문에 따라 KG 정보를 강조하고 적절한 출력을 생성하도록 설계되었습니다. GTP는 모든 옵션 간의 관계를 포착하여 글로벌 정보를 포함하는 소프트 프롬프트 토큰 임베딩을 생성합니다.

- **Performance Highlights**: 실험 결과, QAP는 다양한 데이터셋에서 기존의 최첨단 방법들을 초월하는 성능을 보였습니다. 이는 QAP가 도메인 특화 추론 작업에서 효과적이고 우수한 결과를 낼 수 있음을 확인시켜 줍니다. QAP의 도입으로 LLM의 추론 능력이 크게 향상될 것으로 기대됩니다.



### Federated Self-Supervised Learning for One-Shot Cross-Modal and Cross-Imaging Technique Segmentation (https://arxiv.org/abs/2503.23507)
- **What's New**: 이 논문은 분산된 연합 학습(Decentralized Federated Learning)에서의 자기 지도 학습(self-supervised learning) 기반의 일회성 세분화(one-shot segmentation) 작업을 처음으로 시도하는 연구입니다. 연구는 의료 영상 처리 분야의 데이터 부족 문제를 해결하기 위한 방법으로, 다양한 모델을 통해 여러 출처에서 데이터 표현을 학습할 수 있는 가능성을 탐색합니다. 또한 CoWPro라는 기존의 자기 지도 몇 샷(segmentation few-shot) 세분화 프레임워크를 연합 학습 시나리오에 맞게 조정하여 성능을 향상시킵니다.

- **Technical Details**: 이 연구에서는 프레임워크의 성능을 증대하기 위해 융합 다이스 손실(fused dice loss)을 도입하였습니다. 이를 통해 CoWPro의 기본 성능보다 더 나은 성능을 달성할 수 있었습니다. 연구는 서로 다른 모달리티(modality) 및 이미징 기법을 가진 클라이언트들로부터의 데이터를 구성하는 방식으로 세분화 문제를 더욱 어렵게 만듭니다. 또한, 제안된 프레임워크는 로컬 클라이언트 데이터셋의 전혀 보지 못한 부분에 대해 효과적으로 평가됩니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 CoWPro의 FedAvg 버전에 비해 동등한 또는 더 나은 성능을 보였습니다. 특히 자원을 절약할 수 있는 사전 훈련(all-in one pre-training) 방식 덕분에 다운스트림(task downstream)의 세부 조정이 필요 없어 컴퓨팅 자원의 효율성을 높였습니다. 이 논문에서 제안한 새로운 의료 이미징 데이터셋은 총 95명의 환자로부터 수집된 MRI 스캔을 포함하여 기존의 데이터셋보다 유용성을 강화하며 연구의 투명성을 높입니다.



### Boosting Omnidirectional Stereo Matching with a Pre-trained Depth Foundation Mod (https://arxiv.org/abs/2503.23502)
Comments:
          Project page: this https URL

- **What's New**: DFI-OmniStereo는 새로운 옴니디렉셔널 스테레오 매칭 방법으로, 대규모로 사전 학습된 단안 깊이 모델을 활용하여 상대 단안 깊이 추정을 수행합니다. 이 방법은 반복 최적화 기반의 스테레오 매칭 아키텍처에 통합되어 있으며, 데이터 효율성과 일반화 성능을 향상시키기 위해 전용의 두 단계 훈련 전략을 도입합니다. DFI-OmniStereo는 Helvipad 데이터셋에서 기존 방법보다 약 16% 낮은 disparity MAE를 기록하며 최첨단 성과를 달성했다고 발표합니다.

- **Technical Details**: DFI-OmniStereo는 옴니디렉셔널 스테레오 이미지 쌍을 처리하기 위해 설계된 엔드 투 엔드 모델입니다. 이 모델은 첫 번째 단계에서 스테레오 매칭 헤드가 새로운 특성 공간에 적응할 수 있도록 훈련되며, 두 번째 단계에서는 일반화 성능을 유지하면서 스케일 불변 손실(scale-invariant loss)을 이용해 백엔드의 디코더를 미세 조정합니다. 이러한 과정은 반복 최적화를 통해 진행되며, 다양한 환경에서의 깊이 추정 정확도를 높입니다.

- **Performance Highlights**: DFI-OmniStereo는 Helvipad 데이터셋에서의 테스트 결과, 기존의 옴니디렉셔널 스테레오 매칭 방법들에 비해 성과가 뛰어난 것으로 나타났습니다. 이 방법은 샘플 효율성을 높이고 다른 데이터셋에 대한 일반화 능력 또한 보유하고 있습니다. 이렇게 하여 DFI-OmniStereo는 다양한 조명 조건과 깊이 범위를 갖는 여러 환경에서 깊이 정확도를 개선하는 데 기여합니다.



### POINT$^{2}$: A Polymer Informatics Training and Testing Databas (https://arxiv.org/abs/2503.23491)
- **What's New**: 이 연구에서는 POINT²(POlymer INformatics Training and Testing)를 소개하며, 이는 고성능 폴리머 재료의 발견을 가속화하고 다양한 예측 요소를 통합하기 위한 포괄적인 기준 데이터베이스 및 프로토콜입니다. 기존의 라벨링 데이터셋과 약 100만 개의 가상 폴리머가 포함된 PI1M 데이터셋을 활용하여 ML 모델의 앙상블을 개발하고 있습니다. 이를 통해 폴리머의 속성 예측, 불확실성 평가, 모델 해석 가능성 및 합성 가능성까지 수행할 수 있는 접근 방식을 제시합니다.

- **Technical Details**: POINT²는 다양한 ML 모델(Quantile Random Forests, Multilayer Perceptrons, Graph Neural Networks 등)과 여러 폴리머 표현(Morgan, MACCS, RDKit, Topological, Atom Pair fingerprints 등)을 결합하여 다양한 폴리머 속성을 평가합니다. 이 연구는 예측 정확성 뿐만 아니라 불확실성 정량화(UQ), 모델 해석 가능성, 합성 가능성을 효과적으로 통합하였습니다. 각 모델은 예측 가능한 다양한 속성(예: 가스 투과성, 열 전도도, 밀도 등)에 대한 평가를 제공합니다.

- **Performance Highlights**: 이 연구는 폴리머 화학의 넓은 영역에서 투명성과 재현성을 보장하는 동시에, ML 모델의 신뢰성을 높이는 것을 목표로 합니다. POINT²의 데이터베이스는 폴리머 발견 및 최적화 과정에서 귀중한 리소스로 활용될 수 있으며, ML 기술의 진전을 통해 더욱 혁신적인 연구 촉진이 기대됩니다. 나아가, 불확실성을 모델 개발 및 의사결정 프레임워크에 통합하여 위험 인식 기반의 소재 디자인을 지지할 수 있습니다.



### $p$-Adic Polynomial Regression as Alternative to Neural Network for Approximating $p$-Adic Functions of Many Variables (https://arxiv.org/abs/2503.23488)
Comments:
          10 pages

- **What's New**: 이 논문에서는 연속 함수 $	ext{Z}_p^n ightarrow 	ext{Z}_p$를 연속 함수의 선형 중첩으로 근사하는 새로운 방법을 제시합니다. 이러한 방식으로 고안된 다항 회귀 모델은 어떤 정확도로도 함수를 근사할 수 있습니다. 또한, 제안된 모델의 물리적 해석과 훈련 방법에 대해서도 논의됩니다. 이는 신경망 구조를 기반으로 한 가능한 $p$-adic 모델에 대한 간단한 대안으로 간주될 수 있습니다.

- **Technical Details**: 신경망은 다차원 함수의 근사를 위한 다중 매개변수 시스템으로, 각 그래프의 정점은 여러 개의 입력 엣지를 가지고 하나의 출력 엣지를 가지며, 이는 입력 변수 집합을 출력 변수로 매핑합니다. 이 연구는 $	ext{f}_{nn}(	ext{x},	ext{w})$ 형태의 모델 함수를 정의하며, 이 모델이 주어진 샘플 데이터에 대해 요구되는 정확도로 타겟 함수 $	ext{f}(	ext{x})$를 근사하는 매개변수 $	ext{w}$의 값을 찾는 것을 목표로 합니다. 또한 로스 함수의 최소화 문제를 통해 이 근사를 찾기 위한 방법을 제시합니다.

- **Performance Highlights**: 제안된 모델은 특히 $p$-adic 값의 매개변수 사용 측면에서 기존의 연속 함수 근사 방법에 비해 새로운 가능성을 보여줍니다. 논문에서는 선형 중첩 방식이 왜 효과적인지, 그리고 다항 회귀 모델이 주어진 정확도로 함수를 어떻게 효율적으로 근사할 수 있는지를 다룹니다. 데이터가 주어질 때, 이 모델의 사용은 기존 신경망보다 더 단순할 수 있으며, 기계 학습에서 새로운 접근 방식을 제안합니다.



### Benchmarking Systematic Relational Reasoning with Large Language and Reasoning Models (https://arxiv.org/abs/2503.23487)
Comments:
          Submitted to ACL 2025

- **What's New**: 이 논문은 대규모 언어 모델(LLM)과 대규모 추론 모델(LRM)을 사용한 체계적 추론(systematic reasoning)의 중요성을 강조합니다. 모델의 성능은 종종 규칙적인 추론보다는 지름길에 의존하는 경향이 있으며, 이는 분포 외(out-of-distribution) 예제에서 성능 저하로 이어집니다. 저자들은 공간적 및 시간적 추론에 대한 문제를 통해 이러한 모델들이 어떻게 일반화하는지를 탐구하며, 체계적 일반화(Systematic Generalization, SG) 메트릭을 기반으로 LLM과 LRM의 추론 능력을 평가하는 것이 중요하다고 주장합니다.

- **Technical Details**: 논문에서는 공간 시간 추론(Spatial Temporal Reasoning, STaR) 벤치마크를 활용하여 모델의 성능을 분석합니다. STaR는 복합적 구조를 가지며, 이를 통해 전례 없는 문제 사례를 생성할 수 있어 데이터 세트 오염 문제를 피할 수 있습니다. 이러한 문제는 계산적으로 해결 가능하며, LRM이 접근할 수 있는 문제로 설계되었습니다.

- **Performance Highlights**: 많은 유명한 LLM과 LRM이 STaR에서 어려움을 겪지만, 무작위 기회보다 나은 성과를 보입니다. 모델 규모, 파인튜닝(fine-tuning) 및 체인 오브 띵크(CoT) 테스트 시간이 성능에 미치는 영향을 파악하며, 논문에서 다루는 문제의 복잡도와 모델의 일반화 능력 간의 관계를 평가합니다.



### Order Independence With Finetuning (https://arxiv.org/abs/2503.23483)
Comments:
          Published as a Bi-Align workshop paper at ICLR 2025

- **What's New**: 이 논문은 세트 기반 프롬프트(Set-Based Prompting, SBP)를 활용하여 대형 언어 모델(LLMs)의 순서 의존성을 줄이는 새로운 미세 조정 방법을 제안합니다. 기존의 SBP 방법은 순서를 변경해도 동일한 의미를 가지는 답안 후보에 대해 모델의 예측이 일관되도록 하는 데 초점을 맞추었습니다. 그러나 본 연구에서는 SBP를 훈련 과정에 통합함으로써 성능 저하를 방지하고 모델의 일반적인 언어 모델링 능력을 유지할 수 있음을 보였습니다.

- **Technical Details**: 논문에서는 SBP를 LLM의 훈련 과정에 통합하는 미세 조정 전략을 소개하며, 이를 통해 SBP 형식의 프롬프트가 모델의 학습된 매니폴드(training manifold)에 더 가까워지도록 합니다. 특히, 마진 기반 대조 손실(margin-based contrastive loss)을 사용하여 정답과 오답 간의 구분을 명확히 하는 방법을 채택하였습니다. 이를 통해 SBP 형식의 입력을 훈련할 때 발생하는 분포 이동을 효과적으로 해결하였습니다.

- **Performance Highlights**: 실험 결과, SBP로 미세 조정된 모델은 다중 선택 질문에서 순서에 독립적인 정답률을 크게 향상시키며, CSQA 및 ARC Challenge 데이터셋에 대한 일반화 성능도 개선되었습니다. 또한, 모델이 WikiText-103의 perplexity를 유지함으로써, 보다 넓은 언어 모델링 능력을 저하시키지 않으면서도 안정성을 확보하는 데 성공하였습니다.



### Codehacks: A Dataset of Adversarial Tests for Competitive Programming Problems Obtained from Codeforces (https://arxiv.org/abs/2503.23466)
Comments:
          Accepted for publication at the 18th IEEE International Conference on Software Testing, Verification and Validation (ICST 2025)

- **What's New**: 이 논문에서는 Codeforces 플랫폼에서 자동으로 수집한 실패 유도 테스트 케이스를 기반으로 하는 새로운 데이터셋인 ‘Codehacks’를 소개합니다. 이 데이터셋은 5,578개의 프로그래밍 문제에 대해 288,617개의 해킹 사례를 포함하고 있으며, 각 문제는 자연어 설명과 함께 제공됩니다. Codehacks는 LLM(대형 언어 모델)을 통해 생성된 코드의 품질을 평가하는 데 중요한 자원으로 활용될 수 있습니다.

- **Technical Details**: Codehacks 데이터셋에는 프로그래밍 문제뿐만 아니라, 이러한 문제에 대한 2,196개의 제출 솔루션의 소스 코드도 포함되어 있습니다. 이러한 해킹 기술은 사용자가 제출한 솔루션의 취약성을 발견하기 위한 것으로, 수동으로 만들기에는 비용이 많이 드는 경계 사례 테스트를 제공해 줍니다. 논문에서는 추가 테스트가 필요하다고 강조하며, Codeforces의 온라인 판별 플랫폼이 유용한 자원이라는 점을 지적합니다.

- **Performance Highlights**: Codehacks는 LLM을 사용한 프로그램 합성 기술의 검증과 평가에서 중요한 기여를 할 것으로 기대됩니다. 과거에는 테스트 검증 시 존재하는 허위 부정 결과를 찾는 데 필요한 추가 테스트를 작동시키는 과정이 비용과 시간이 많이 소요되었습니다. Codeforces에서의 해킹 사례를 활용하여, 이러한 부정확한 결과를 줄일 수 있는 효과적인 방법을 제시합니다.



### Accelerated Stein Variational Gradient Flow (https://arxiv.org/abs/2503.23462)
Comments:
          Submitted to GSI'25, 9 pages, 2 figures, comments welcome

- **What's New**: 이번 논문에서는 Stein Variational Gradient Descent (SVGD) 방법의 속도를 향상시키기 위해 ASVGD라는 새로운 알고리즘을 제안합니다. ASVGD는 네스터로프(Nesterov) 방법에 기반한 가속화된 그래디언트 흐름을 통해 고차원 샘플링 문제를 효율적으로 해결하는 데 초점을 맞춥니다. 이 방법은 정확한 로그밀도 기울기 추정(score estimation) 없이도 입자들이 결정론적으로 진화하도록 합니다.

- **Technical Details**: ASVGD는 스텐(Stein) 메트릭으로 정규화된 확률 밀도의 메트릭 공간에서의 가속화된 그래디언트 흐름을 이용합니다. 또한, 입자들의 모멘텀 업데이트를 안정화하기 위해 Wasserstein 메트릭 정규화 기법을 연구합니다. 이 알고리즘은 상호작용하는 입자들의 위치와 모멘텀을 사용하여 AIG(Acelerated Information Gradient) 흐름을 근사합니다.

- **Performance Highlights**: 다양한 목표 분포에 대해 일반화된 이명세(Bilinear) 커널과 가우시안 커널을 사용한 toy numerical 예제들이 ASVGD의 효과성을 SVGD 및 다른 인기 샘플링 방법과 비교하여 보여줍니다. 실험 결과 ASVGD는 높은 차원의 문제에서 빠르고 효율적인 샘플링 성능을 나타내어 기존의 방법보다 우수한 성능을 발휘합니다.



### Semantic-Preserving Transformations as Mutation Operators: A Study on Their Effectiveness in Defect Detection (https://arxiv.org/abs/2503.23448)
Comments:
          Accepted for publication in Mutation 2025 at the 18th IEEE International Conference on Software Testing, Verification and Validation (ICST 2025)

- **What's New**: 이번 연구에서는 결함 탐지(defect detection) 도구의 성능을 개선하기 위해 의미 보존 변환(semantic-preserving transformations)을 사용할 수 있는지 분석했습니다. 기존의 연구들은 의미적으로 동일한 코드에서 모델의 강건성을 향상시키기 위해 훈련 데이터를 강화하는 데 집중했지만, 이러한 코드가 실제 도구 성능 개선에 어떻게 사용될 수 있는지는 잘 알려져 있지 않았습니다. 이를 통해 우리는 LLMs(대형 언어 모델)와 도구의 조합이 기존에 알려진 방식과는 다른 새로운 접근을 제시할 수 있음을 발견했습니다.

- **Technical Details**: 연구진은 28개의 논문에서 94개의 의미 보존 변환을 수집하였으며, 이 중 39개의 변환을 실제로 구현하였습니다. 그러나 수작업 검토 결과 39개 중 23개가 코드 의미를 변경하는 것으로 나타났습니다. 최종적으로 16개의 변환을 사용하여 LLMs를 통해 결함 탐지 도구의 성능을 향상시킬 수 있는지를 실험하였습니다. 연구 과정에서 세 가지 앙상블 기법을 적용하여 성과를 평가하였습니다.

- **Performance Highlights**: 본 연구의 결과, 선택된 16개의 올바른 변환과 세 가지 앙상블 기법을 사용했음에도 결함 탐지 모델의 정확도를 향상시키지 못했습니다. 연구진은 의미 보존 변환을 재사용하는 것이 어렵고, 일부 변환이 의도치 않게 의미를 변경할 수 있음을 발견했습니다. 따라서 향후 연구에서는 이러한 구현의 어려움을 극복하기 위한 방안이 필요하다는 인사이트를 제공합니다.



### Speculative End-Turn Detector for Efficient Speech Chatbot Assistan (https://arxiv.org/abs/2503.23439)
Comments:
          Preprint

- **What's New**: 이번 논문에서는 대화 시스템의 중요한 문제인 end-turn detection (ETD) 문제를 해결하기 위해 ETD Dataset을 출시했습니다. 이 데이터셋은 텍스트 대화 데이터를 기반으로 생성된 합성 음성 데이터와 웹 소스에서 수집된 실제 음성 데이터로 구성되어 있습니다. 또한, 자원 제한 환경에서 실시간 ETD를 개선하기 위한 새로운 협업 추론 프레임워크인 SpeculativeETD를 제안합니다.

- **Technical Details**: SpeculativeETD는 경량의 GRU 기반 모델과 고성능 Wav2vec 기반 모델을 조합하여 효율성과 정확성을 균형있게 유지합니다. 경량 모델은 로컬 디바이스에서 빠르게 비말 단위를 탐지하고, 침묵이 감지되면 고성능 모델에 질의하여 효과적으로 턴의 종료 여부를 판단합니다. 이 접근 방식은 고성능 모델이 실시간으로 작동할 필요가 없으므로 필요한 계산량을 대폭 줄일 수 있습니다.

- **Performance Highlights**: 실험 결과, SpeculativeETD는 ETD 정확도를 크게 향상시키면서도 계산 요구량을 최소화하는 것으로 나타났습니다. 120,000개 이상의 샘플과 300시간 이상의 대화 데이터가 포함된 ETD 데이터셋은 모델 트레이닝 및 평가에 유용하게 활용될 것입니다. 데이터셋과 코드는 리뷰 후 공개될 예정입니다.



### DGSAM: Domain Generalization via Individual Sharpness-Aware Minimization (https://arxiv.org/abs/2503.23430)
- **What's New**: 본 논문은 도메인 일반화(Domain Generalization, DG)를 위한 새로운 알고리즘인 DGSAM(Decreased-overhead Gradual Sharpness-Aware Minimization)을 제안합니다. DGSAM은 각 도메인에서의 결과를 고려하여 점진적으로 파라미터를 섭동(perturb) 시키며, 이는 기존의 Sharpness-Aware Minimization(SAM)이 가진 한계를 극복합니다. 특히, DGSAM은 계산 효율성을 유지하면서도 개별 도메인의 샤프니스(sharpness)를 일관되게 저하시킵니다.

- **Technical Details**: DGSAM은 샤프니스가 낮은 평평한 미니마(flat minima)를 찾기 위해 각 도메인의 손실 기울기를 사용하여 점진적으로 파라미터를 섭동합니다. 이는 기존의 SAM이 전체 손실의 샤프니스만 최소화하는 데 그쳤던 것과는 다릅니다. DGSAM은 또한 자원 소모를 줄이기 위해 기존 기울기를 재사용하여 계산 효율성을 크게 개선합니다.

- **Performance Highlights**: 실험 결과 DGSAM은 DomainBed 프로토콜에서 기존의 DG 알고리즘보다 우수한 성능을 보였습니다. 다양한 데이터셋에서 높은 평균 정확도와 낮은 표준 편차를 기록하며 도메인 변화에 대한 견고성을 나타냈습니다. 특히, DGSAM은 기존의 SAM 및 SAGM 방법들과 비교했을 때 개별 도메인에서의 샤프니스를 현저히 줄이는 성과를 보였습니다.



### Quantum-Assisted Machine Learning Models for Enhanced Weather Prediction (https://arxiv.org/abs/2503.23408)
- **What's New**: 본 연구는 양자 컴퓨팅(Quantum Computing)을 활용하여 기상 예측을 개선하는 양자 기계 학습(Quantum Machine Learning, QML)의 혁신적인 접근 방식을 제시합니다. 양자 게이트 순환 유닛(Quantum Gated Recurrent Units, QGRUs), 양자 신경망(Quantum Neural Networks, QNNs), 변분 양자 회로(Variational Quantum Circuits, VQCs) 등의 QML 모델을 사용하여 ERA5 데이터세트의 기상 시계열 데이터를 분석하였습니다. 연구 결과 QML 모델들이 이진 분류에서 특히 예측 및 분류 작업에서 합리적인 정확도를 달성할 수 있음을 보여줍니다.

- **Technical Details**: 본 연구에서는 ERA5 재분석 데이터셋을 사용하여 기상 예측을 위한 양자 강화 모델을 구현합니다. 이 데이터셋은 1940년부터 현재까지의 시간당 데이터를 포함하며, 31km의 그리드 해상도와 137개의 수직 수준으로 전 세계 기후를 포괄합니다. 특정 기상 변수(예: 2미터 온도)의 영향을 미치는 가장 관련성이 높은 특징을 식별하기 위해 피어슨 상관 계수를 계산하여 특징 선택을 수행하였습니다.

- **Performance Highlights**: QML 모델들이 기존의 전통적인 모델들을 초월하여 더 나은 예측 정확도를 제공합니다. 연구에서는 기온의 이진 분류 및 다중 클래스 분류를 위한 예측 성능을 평가하였으며, QML 모델들이 특히 기온 예측에서 뛰어난 성과를 나타냈습니다. 그러나 양자 하드웨어의 한계 및 잡음으로 인해 확장성과 일반화 문제도 존재하는 것을 확인했습니다.



### COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation (https://arxiv.org/abs/2503.23388)
Comments:
          Accepted to CVPR 2025

- **What's New**: 최근 비전-언어 모델(VLMs)인 COSMIC(클릭 지향 의미 다중 공간 통합)는 새로운 도메인에 대한 테스트 시간 적응에 있어 중요한 도전을 해결하기 위한 프레임워크입니다. COSMIC은 다중 분류에서 우수한 적응성을 발휘하며, 세 가지 주요 혁신인 이중 의미 그래프(Dual Semantics Graph)와 클리크 지도 하이퍼 클래스(Clique Guided Hyper-class)에 기반합니다. 이를 통해 혼합된 세미틱 정보를 활용하여 예측의 강인성을 개선하고 있습니다.

- **Technical Details**: COSMIC은 다중 세미틱 캐싱(multi-granular, cross-modal semantic caching)과 그래프 기반 쿼리 메커니즘을 활용하여 모델의 적응성을 증대시킵니다. 이중 의미 그래프(DSG)는 텍스트 특징, 조밀한 CLIP 및 미세 조정된 DINOv2 특징을 통합하여 보강된 의미 공간을 생성합니다. 클리크 지도 하이퍼 클래스(CGH)는 구조화된 클래스 관계를 이용하여 예측 강인성을 높인다.

- **Performance Highlights**: COSMIC은 여러 벤치마크에서 놀라운 성능을 기록하며, 특히 out-of-distribution 태스크에서 15.81%의 향상을 보였습니다. 또한 클립 RN-50을 활용한 크로스 도메인 생성에서도 5.33%의 성능 개선을 이뤘고, 코드가 공개되어 누구나 사용할 수 있습니다. 이러한 결과는 COSMIC의 혁신적인 접근 방식이 효과적임을 잘 보여줍니다.



### KernelDNA: Dynamic Kernel Sharing via Decoupled Naive Adapters (https://arxiv.org/abs/2503.23379)
- **What's New**: 본 연구는 기존 동적 합성곱(dyamic convolution) 기법의 한계를 극복하기 위한 새로운 경량 합성곱 커널 모듈 KernelDNA를 제안합니다. KernelDNA는 입력 의존적인 동적 라우팅(dynamic routing)과 사전 훈련된 정적 모듈레이션(static modulation)을 결합하여 매개변수 효율성(parameter efficiency)과 하드웨어 친화적인 추론(inference)을 보장합니다. 기존 방법들은 매개변수 증가 문제를 겪었지만, 본 연구는 계층 간 가중치 공유(cross-layer weight sharing)를 통해 이러한 문제를 해결합니다.

- **Technical Details**: KernelDNA는 입력 데이터에 따라 동적으로 커널을 조정하는 동시에 사전 훈련된 커널을 재사용하는 메커니즘을 도입합니다. 이를 통해 기존의 정적 합성곱 구조를 유지하면서도 입력에 적응한 커널 조정으로 표현력을 향상시킵니다. 연구에서는 세 가지 향상된 주의 메커니즘인 채널 주의(Channel Attention), 필터 주의(Filter Attention), 공간 주의(Spatial Attention)를 통합하여 각 합성곱 계층이 고유한 특성을 유지하도록 하였습니다.

- **Performance Highlights**: 실험 결과, KernelDNA는 기존의 동적 합성곱 방법들보다 우수한 정확도를 보여주며, 원래의 추론 속도도 거의 유지했습니다. 예를 들어, ResNet18 모델에서 KernelDNA는 1.2-5배의 매개변수 감소와 높은 정확도(74.23%)를 달성했습니다. 경량 모델에서도 뛰어난 성능을 보여주어 다양한 아키텍처에서 매개변수 효율성, 하드웨어 호환성 및 적응 성능의 새로운 조화를 이뤘습니다.



### Large Language Models Are Better Logical Fallacy Reasoners with Counterargument, Explanation, and Goal-Aware Prompt Formulation (https://arxiv.org/abs/2503.23363)
Comments:
          Accepted to NAACL 2025 Findings

- **What's New**: 본 연구에서는 논리적 오류 탐지를 위한 새로운 프롬프트(formulation) 기법을 제안하며, 이는 감독 학습(supervised) 및 비감독 학습(unsupervised) 환경에 모두 적용이 가능하다. 이 방법은 입력 텍스트에 암묵적 맥락 정보(implicit contextual information)를 통합하여 오류의 유효성을 평가하는 쿼리를 생성하고, 이를 기반으로 결과를 분류한다. 또한, 다섯 개의 데이터 세트를 사용한 평가 결과 시간적 모델들에 비해 현저한 성능 향상을 확인했다.

- **Technical Details**: 제안된 접근법은 네 개의 주요 단계로 구성되며, 첫 단계에서는 LLM을 이용해 맥락 개선을 통해 앵커 쿼리(context-informed queries)를 생성한다. 이후 생성된 쿼리를 통해 논리적 오류를 분류하며, 마지막 단계에서는 각 쿼리에 대해 신뢰도 기반으로 순위를 매긴다. 특이하게도, 본 연구에서는 각 입력 텍스트를 증강하기 위해 세 가지 유형의 암묵적 정보(반론(counterargument), 설명(explanation), 목표(goal))를 활용한다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 제로샷(zero-shot) 환경에서 Macro-F1 점수가 최대 0.60, 파인튜닝(fine-tuned) 모델에서는 최대 0.45 향상된 성능을 보였다. 따라서 본 접근법이 최첨단 모델들보다 월등한 결과를 드러내었으며, 이는 프롬프트 순위 매기기 방법의 효과적 활용에 기인한 것으로 분석되었다. 단계별로 수행된 심층 분석을 통해 제안 방법의 장점과 개선점을 추가적으로 검토하였다.



### HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation (https://arxiv.org/abs/2503.23331)
Comments:
          CVPR2025

- **What's New**: 본 논문에서는 2D에서 3D로의 인간 포즈 추정(HPE)에서 발생하는 가림 문제를 해결하기 위한 혁신적인 두 단계 생성 밀착 방법인 Hierarchical Pose AutoRegressive Transformer(HiPART)를 제안합니다. 기존 방법들은 데이터의 희소성 문제와 가림 상황의 복잡성을 간과했습니다. HiPART는 원래 희소 2D 포즈에서 계층적인 2D 조밀한 포즈를 생성하여 가림 상황에서의 강력한 복원력을 입증하였습니다.

- **Technical Details**: HiPART는 두 개의 주요 모듈로 구성되어 있습니다. 첫 번째는 Multi-Scale Skeletal Tokenization(MSST) 모듈로, 밀접하게 밀착된 2D 포즈를 계층적인 토큰으로 양자화합니다. 두 번째는 Hierarchical AutoRegressive Modeling(HiARM) 스킴을 통해 계층적인 2D 포즈 생성을 달성합니다. 이 방법은 비유클리드 구조에 적합한 새로운 희소-밀착 및 중심-주변 전략을 도입하여 효율성을 보여줍니다.

- **Performance Highlights**: HiPART는 단일 프레임 기반 3D HPE에서 최신 성과를 달성하였으며, Human3.6M, 3DPW 등 다양한 벤치마크에서 뛰어난 강건성을 보여줍니다. 복잡한 시계열 인코더를 사용하는 방법들과 비교하여 현저하게 감소된 복잡성으로 동등하거나 우수한 성능을 발휘합니다. 나아가 HiPART는 기존의 시계열 방법과 독립적으로 작동하며 성능 향상에 더욱 기여할 수 있습니다.



### AI Agents in Engineering Design: A Multi-Agent Framework for Aesthetic and Aerodynamic Car Design (https://arxiv.org/abs/2503.23315)
- **What's New**: 본 연구에서는 '디자인 에이전트(Design Agents)'라는 개념을 도입하여 자동차 설계 프로세스를 혁신적으로 변화시키고 있습니다. 이 접근법은 기존 엔지니어링 워크플로우에 AI 기반의 디자인 에이전트를 통합하여 효율성을 높이고 디자인 사이클을 단축하는 특징이 있습니다. 특히, autonomous computational systems을 활용하여 전통적인 수작업 과정을 자동화하여 시간을 단축시키는 것이 주요 내용입니다.

- **Technical Details**: 디자인 에이전트는 저희의 메소드에서 사용되는 estado de la técnica 시각-언어 모델(Vision-Language Models, VLMs)과 대형 언어 모델(Large Language Models, LLMs) 및 기하학적 딥러닝 기술을 활용합니다. 이러한 기술들은 초기 스케치부터 완전한 시뮬레이션에 이르기까지의 프로세스를 간소화하고, 예측 모델을 통해 공기역학적 평가를 신속하게 수행할 수 있게 해줍니다. Python API 및 AutoGen을 통해 모든 과정이 자동화되어 유연한 디자인 최적화를 가능하게 합니다.

- **Performance Highlights**: 제안된 다중 에이전트 프레임워크는 디자인 주기가 급격히 단축되며, 기존의 몇 주가 걸리던 과정을 몇 분으로 줄일 수 있는 능력을 보여줍니다. 디자인 에이전트들은 CAD 모델링, 3D 형상 생성을 통한 공기역학 평가, 그리고 지속적인 실시간 예측을 가능하게 하여 디자이너와 엔지니어 간의 협업을 활성화합니다. 연구 결과는 전통적인 자동차 디자인과 AI 기술의 융합을 통해 향후 다양한 엔지니어링 분야에서의 혁신 가능성을 강조합니다.



### SPIO: Ensemble and Selective Strategies via LLM-Based Multi-Agent Planning in Automated Data Scienc (https://arxiv.org/abs/2503.23314)
Comments:
          Under Review

- **What's New**: 이번 논문에서는 SPIO(Sequential Plan Integration and Optimization)라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 LLM(대형 언어 모델)을 이용한 의사결정 방식을 통해 샘플 전략을 생성하고 최적화합니다. 기존의 단일 경로 워크플로우와 달리 SPIO는 다단계 처리 프로세스를 적용하여 데이터 전처리, 특성 엔지니어링, 모델링, 하이퍼파라미터 조정까지 아우릅니다. 이를 통해 다양한 전략을 탐색할 수 있는 기회를 제공합니다.

- **Technical Details**: SPIO 프레임워크는 네 가지 주요 모듈인 데이터 전처리(data preprocessing), 특성 엔지니어링(feature engineering), 모델 선택(model selection), 하이퍼파라미터 조정(hyperparameter tuning)으로 구성됩니다. 각 모듈에서는 독립적으로 후보 전략을 생성하는 전용 계획 에이전트(planning agents)가 존재합니다. SPIO는 또한 두 가지 변형인 SPIO-S(단일 최적 계획 선택)와 SPIO-E(상위 k개 계획을 조합)로 나뉘어 각각의 활용도를 극대화할 수 있습니다.

- **Performance Highlights**: Kaggle과 OpenML 데이터 세트를 대상으로 한 광범위한 실험에서 SPIO는 최신 방법론보다 우수한 성능을 나타냈습니다. SPIO의 적응형 다경로(reasoning) 접근 방식은 다양한 통찰력을 통합할 수 있어 고정된 단일 경로 워크플로우의 한계를 효과적으로 극복합니다. 이로 인해 SPIO는 예측 정확도를 지속적으로 향상시키고, 다양한 데이터 시나리오에 적응하며, 실행 신뢰성을 높이는 데 있어 탁월한 성과를 보이고 있습니다.



### Reinforcement Learning for Active Matter (https://arxiv.org/abs/2503.23308)
Comments:
          16 pages, 8 figures

- **What's New**: 이 논문은 능동 시스템(active matter)에서 강화 학습(reinforcement learning, RL)을 활용하여 개별 입자 및 집단 동태를 제어하는 새로운 방법론을 제시합니다. 특히, 각 능동 입자의 최적 운동 전략과 능동 집단의 집합적 동작 조절에 중점을 두고 있습니다. ML과 RL의 발전으로 인해 능동 물질의 복잡성을 해결하기 위한 효율적인 접근법이 마련되었습니다.

- **Technical Details**: 능동 물질은 본질적인 추진 메커니즘을 가진 요소들로 구성되며, 에너지를 소비하여 운동을 수행합니다. 주요 모델로는 active Brownian particle (ABP) 모델과 연속체 이론이 있으며, 지원 모델은 비대칭성을 반영하여 비평형 행동을 설명합니다. RL은 정책을 학습하는 데 있어 실시간 적응 및 최적화를 가능하게 하여, 동적 환경에서 능동 물질의 행동을 효과적으로 조절합니다.

- **Performance Highlights**: 본 논문은 RL을 통해 개별 능동 입자의 내비게이션과 자원 탐색 전략을 최적화하는 방법을 제시하며, 집단 동작의 자가조직화와 목표 지향적 제어를 강조합니다. 이는 마이크로 로보틱스 및 바이오 의학 분야에서 새로운 혁신을 이끌 가능성이 있으며, 복잡한 시스템의 조작 및 최적화에 실질적인 응용을 촉진할 것입니다.



### Using Source-Side Confidence Estimation for Reliable Translation into Unfamiliar Languages (https://arxiv.org/abs/2503.23305)
Comments:
          7 pages, 5 figures, 1 table. Submitted to ACL 2025 System Demonstrations

- **What's New**: 이번 연구에서는 상호작용형 기계 번역(Interactive Machine Translation, MT) 시스템을 소개합니다. 이 시스템은 사용자들이 목표 언어에 능숙하지 않을 때도 신뢰성과 설명력을 향상시키기 위해 설계되었습니다. 특히, 번역 오류를 수정할 수 있는 사용자 개입을 허용하기 위해, 불확실성이 높은 단어를 강조하고 이에 대한 수정 제안을 제공합니다.

- **Technical Details**: 이 시스템은 소스 측 신뢰도 추정을 통해 불확실성이 높은 단어를 강조합니다. 구체적으로, 소스 단어의 임베딩에 대한 출력 시퀀스의 민감도를 측정하여 불확실성 점수를 계산합니다. 이렇게 얻은 점수는 단어의 온도를 평가하며, 밀접한 단어의 변화를 감지함으로써 보다 정확한 번역을 가능하게 합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 기존의 정렬 기반 방법보다 번역 오류 감지에서 우수한 성능을 보였습니다. 이 시스템은 특히 소스 언어에 능숙한 사용자들이 보다 신뢰할 수 있는 번역 결과를 받아볼 수 있게끔 하는 데 기여할 것입니다. 향후, 이러한 불확실성 점수는 사전 정의 검색과 같은 다른 애플리케이션에도 통합될 수 있습니다.



### Extracting Patient History from Clinical Text: A Comparative Study of Clinical Large Language Models (https://arxiv.org/abs/2503.23281)
- **What's New**: 이번 연구에서는 환자의 주요 불만(Chief Complaint, CC), 현재 병력(History of Present Illness, HPI), 과거 및 가족의 사회적 병력(Past, Family, and Social History, PFSH)과 관련된 의료 기록 엔터티(Medical History Entities, MHEs)를 추출하는 방식에 대해 발표하였습니다. 이는 비정형화된 임상 노트를 체계적인 전자 건강 기록(Electronic Health Records, EHRs)으로 변환하여 의료 제공의 연속성, 의료 코딩, 품질 지표 등의 downstream 작업을 효율화하는 데 기여합니다. 연구에서는 최신 임상 대형 언어 모델(Fine-tuned Clinical Large Language Models, cLLMs)을 활용하여 이러한 MHE를 인식하고, 노트 특성이 모델의 정확도에 미치는 영향을 분석하였습니다.

- **Technical Details**: 연구팀은 MTSamples 리포지토리에서 61개의 외래 환자 관련 임상 노트로부터 1,449개의 MHE를 주석 처리하고, 인지 모델을 인식하기 위해 7개의 최첨단 cLLMs를 파인튜닝(Fine-tuning) 하였습니다. 추가로, 문제, 테스트, 치료 및 기타 기본 의료 엔터티(Basic Medical Entities, BMEs)를 통합하여 모델 성능을 평가하였습니다. 실험은 zero-shot 설정에서 GPT-4o와 비교하여 이루어졌으며, 텍스트 특성이 모델의 정확도에 미치는 영향에 대한 오류 분석도 수행되었습니다.

- **Performance Highlights**: 연구 결과, cLLMs는 MHE 추출에 필요한 시간을 20% 이상 단축시킬 잠재력을 보여주었습니다. 그러나 다의성(polysomy)의 특성과 비의료 용어의 빈번한 사용으로 인해 MHE 탐지에서 여전히 어려움이 존재했습니다. 특히, GatorTron과 GatorTronS 두 가지 모델이 가장 높은 성능을 발휘했으며, 사전에 식별된 BME 정보 통합이 특정 엔터티에 대한 모델 성능 향상에 기여했습니다. 또한, 텍스트 길이, 엔터티 길이, 세분화와 같은 특성이 모델 성능에 미치는 영향에 대한 유의미한 결과가 도출되었습니다.



### Localized Graph-Based Neural Dynamics Models for Terrain Manipulation (https://arxiv.org/abs/2503.23270)
- **What's New**: 이번 논문에서는 로봇이 건설 현장과 외계 표면에서 잘 작업할 수 있도록 terrain dynamics modeling 및 manipulation을 위한 학습 기반 접근법을 소개합니다. Graph-based Neural Dynamics (GBND) 프레임워크를 활용하여, 다차원 terrain 변형을 효과적으로 예측하고 제어하는 방법이 제안되었습니다. 특히 이 방법은 로봇의 제어 입력과 현재 장면을 기반으로 한 작은 관심 영역(Region of Interest, RoI)을 동적으로 선택하여 그 안의 입자들만을 사용하여 예측 효율성을 크게 향상시킵니다.

- **Technical Details**: 이 연구의 핵심은 거대한 크기의 terrain을 더 작은 활성 서브그래프로 제한하여 그 안의 동역학을 예측하는 것입니다. 이 과정에서 그래픽 카드 메모리에 맞지 않는 수백만 개의 입자들로 구성된 대규모 terrain 그래프를 구축하였습니다. 또한, GBND와 RoI를 동시에 학습하는 방법을 통해 로봇-terrain 상호작용 중에 고정된 입자가 정적이라는 가정을 바탕으로 예측 속도를 현저히 향상시킴과 동시에 더 높은 예측 정확도를 달성하였습니다.

- **Performance Highlights**: 제안된 방법은 기존의 GBND 방법보다 수 배 빠르며, 다양한 재료로 구성된 terrain 조작 작업에 대한 실험을 통해 그 효율성과 효과성을 검증하였습니다. 이 연구결과는 건설 산업과 우주 탐사에서 로봇의 자율성과 효율성을 크게 향상시킬 것으로 기대됩니다. 또, 샘플링 복잡성을 줄이면서도 높은 예측 정확도를 유지하여 기존의 물리 기반 방법에서 직면하는 문제를 효과적으로 완화했습니다.



### A Lightweight Image Super-Resolution Transformer Trained on Low-Resolution Images Only (https://arxiv.org/abs/2503.23265)
- **What's New**: 이 논문은 LR(저해상도) 이미지만을 이용한 학습 방법을 채택하여 경량화된 비전 변환기 모델인 SwinIR을 활용한 최초의 연구입니다. 전통적인 SISR(Single-Image Super-Resolution) 모델들은 HR(고해상도) 이미지에 대한 대량의 훈련 데이터를 요구하지만, 본 연구는 LR만으로도 효과적으로 작업을 수행할 수 있음을 보여줍니다.

- **Technical Details**: 논문에서 제안한 MSTbic는 미세 이미지 초해상도에서 개발된 LR-only 다중 스케일 훈련 방법을 큰 규모의 실제 데이터로 적용한 것입니다. SwinIR 모델은 얕은 특징 추출 모듈과 깊은 특징 추출 파트, HR 이미지 재구성 모듈로 구성되어 있으며, 자기 주의(attention) 및 잔차 연결(residual connection)을 사용하여 기존의 CNN 구조와 비교해 뛰어난 성능을 나타냅니다.

- **Performance Highlights**: SwinIR 및 CNN 모델을 모두 비교한 결과, 제안된 방법이 기존의 CNN 기반 LR-only SISR 방법들에 비해 우수한 성능을 보였음이 입증되었습니다. 이는 Set5, Set14, BSD100, Urban100 및 Manga109와 같은 전통적인 SR 벤치마크 데이터셋에서 확인되었으며, 새로운 최첨단 성과를 달성한 것이 특징입니다.



### Joint Source-Environment Adaptation for Deep Learning-Based Underwater Acoustic Source Ranging (https://arxiv.org/abs/2503.23262)
- **What's New**: 이 연구에서는 사전 훈련된 딥러닝 기반 모델을 새로운 환경에 맞게 조정하는 방법을 제안합니다. 특히, 레이블 없이 비지도 학습을 통한 도메인 적응(unsupervised domain adaptation) 기법을 이용해 모델의 일반화 성능을 향상시키고, 신호 에너지를 기반으로 한 독립적인 추정 방법을 결합하여 예측 성능을 개선합니다.

- **Technical Details**: 이 방법은 데이터 전송 비용, 보안 및 개인 정보 문제를 고려하여 소스가 없는 도메인 적응(source-free domain adaptation)을 통해 수행됩니다. 제안된 방식은 수신 신호의 세기를 기반으로 하는 다른 추정 프로세스와 결합하여 모델의 불확실성을 줄입니다. 논문에서 소개된 샘플 공분산 행렬(sample covariance matrices)은 UWA(localization)에서 신호와 노이즈가 공동 가우시안 상황에서 충분한 통계량으로 사용됩니다.

- **Performance Highlights**: Bellhop로 생성한 데이터와 SWellEx-96 실험의 혼합 환경에서 제안된 방법의 우수성을 입증합니다. 실험 결과, 사전 훈련된 모델과 결합한 새로운 추정 프로세스가 더 강력한 성능을 발휘함을 보여줍니다. 특히, 분류 모델을 활용하여 예측의 불확실성을 정량화하고, 라벨 간의 거리 유지를 통한 모델의 정밀도를 높였습니다.



### Mismatch-Robust Underwater Acoustic Localization Using A Differentiable Modular Forward Mod (https://arxiv.org/abs/2503.23260)
- **What's New**: 이번 논문에서는 환경 불일치에 있는 수중 음향 위치 측정을 연구합니다. 특히, 우리는 사전 훈련된 신경망을 사용하여 음향 파동 전파를 최적화 프레임워크로 구현하고 있습니다. 이를 통해 테스트 환경에서의 데이터 부족 문제를 완화하고, 라벨이 없는 데이터 접근만으로도 효과적으로 적응할 수 있는 조건을 제공합니다.

- **Technical Details**: 우리의 연구는 모델링에서 물리 기반 모듈성을 도입하여, 루트 경로의 길이를 학습할 수 있도록 합니다. 방법론에 있어, 전방 모델에서의 입력을 최적화함으로써 발전된 형태의 UWA 위치 측정을 가능하게 합니다. 또한, 기울기 기반 최적화를 통해 네트워크 매개변수뿐만 아니라 입력에 대해 동시에 최적화하여 성능 저하를 최소화하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 우리는 경로의 정보 없이도 다중 경로 구조를 학습할 수 있는 모듈형 네트워크를 제안합니다. 이는 격리된 작업 단위가 결합되어 전체 모델의 성능을 향상시키는 데 기여합니다. 마지막으로, 본 연구의 네트워크 구조는 특정 환경에 맞게 재교육할 수 있는 잠재력을 가지고 있습니다.



### Joint Source-Environment Adaptation of Data-Driven Underwater Acoustic Source Ranging Based on Model Uncertainty (https://arxiv.org/abs/2503.23258)
- **What's New**: 본 논문은 수중 음향 위치 추적(UWA localization)을 위한 사전 훈련된 딥러닝 모델의 테스트 시 환경 적응(test-time adaptation) 방안을 제시합니다. 특히, 환경 불일치가 클수록 모델이 나타내는 '내재된 불확실성(implied uncertainty)'을 활용하여 테스트 샘플을 더 확실한 집합과 덜 확실한 집합으로 나누고 이를 통해 불확실한 샘플에 대한 레이블링을 개선합니다. 이 방법은 목표 환경의 레이블 데이터 없이도 테스트 단계에서 모델을 적응시키는 새로운 접근 방식을 제공합니다.

- **Technical Details**: 제안된 방법은 수신 신호 에너지를 기반으로 한 독립적인 추정치를 통합하여 적응성을 강화합니다. 또한, 모델 예측 불확실성을 정량화하는 효율적인 방법을 사용하여 수중 환경에 대한 적응을 수행하며, 이 과정에서 확률적 차원 감소 기법인 peakwise uncertainty(PU) 방법을 적용합니다. 실험 결과, 제안된 방법은 환경 간 불일치가 클 때 모델의 예측 정확성을 향상시킴을 보여줍니다.

- **Performance Highlights**: 제안한 적응 메커니즘은 실제 실험 데이터를 비롯하여 실제 해양 소음이 포함된 합성 데이터에서 광범위한 유효성을 입증하였습니다. 성능 결과는 다양한 조건과 환경에서 모델이 보다 정확한 수중 음향 위치 추적을 가능하게 함을 증명하였습니다. 또한, 이 연구는 수중 음향 localization의 정확도를 획기적으로 향상시킬 가능성을 강조합니다.



### FIESTA: Fisher Information-based Efficient Selective Test-time Adaptation (https://arxiv.org/abs/2503.23257)
- **What's New**: 이 논문은 얼굴 표정 인식(facial expression recognition) 분야에서, 비구속적인 환경에서의 도메인 변화(domain shift) 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 기존의 테스트 시간 적응(test-time adaptation; TTA) 방식은 매개변수 업데이트의 수동 선택에 의존하는데, 이는 효율성을 저하시킬 수 있습니다. 본 연구에서는 Fisher 정보(Fisher information)를 기반으로 한 선택적 적응(selective adaptation) 프레임워크를 도입하여, 가장 중요한 매개변수만을 동적으로 업데이트합니다.

- **Technical Details**: 제안된 Fisher 기반 선택적 적응 기법은 비디오 기반 얼굴 표정 인식에 적합하게 설계되었습니다. 이 방법은 매개변수 중요도를 Fisher 점수(Fisher scores)로 정량화하고, 이를 통해 모델 성능에 중요한 가중치만을 선택적으로 업데이트합니다. 또한, 이 과정은 시간적 일관성(temporal consistency) 제약과 결합되어 모델의 적응 과정을 보다 효율적이고 효과적으로 만듭니다.

- **Performance Highlights**: AffWild2 벤치마크 데이터세트에 대한 실험 결과, 제안된 접근 방식이 기존 TTA 방법보다 7.7% 향상된 F1 점수를 기록하며, 22,000개의 매개변수만을 업데이트하는 것으로 확인되었습니다. 이는 기존의 방법들보다 20배 이상 적은 매개변수를 사용하는 결과입니다. 또한, 최소한의 데이터(1-3 프레임)로부터 매개변수의 중요도를 효과적으로 추정할 수 있어, 실제 애플리케이션에서 TTA를 더욱 실용적으로 만들어 줍니다.



### Beyond Contrastive Learning: Synthetic Data Enables List-wise Training with Multiple Levels of Relevanc (https://arxiv.org/abs/2503.23239)
Comments:
          Code: this https URL

- **What's New**: 이 논문에서는 정보 검색(Information Retrieval, IR)에서 큰 언어 모델(LLM)을 활용하여 기존의 훈련 방식을 개선하는 SyCL (Synthetic ranking Context for List-wise training) 방법을 제안합니다. SyCL은 실제 문서를 사용하지 않고도 여러 수준의 관련성을 가진 합성 문서를 생성하여 IR의 효율성을 극대화합니다. 이를 통해 기존의 이진 라벨로 한정된 훈련 방식을 뛰어넘고, 더 복잡한 문서 순위 매기기를 가능하게 합니다.

- **Technical Details**: 제안된 SyCL 방법은 오픈 소스 LLM을 활용하여 MS MARCO 데이터셋에 대한 질의에 따라 네 가지 다른 관련성 수준을 가진 전방위 대량 합성 문서를 생성합니다. 이 문서들은 Wasserstein Distance를 손실 함수로 사용하여 훈련 중 상대적 라벨 불일치를 반영하여 모델의 점수 선택을 다르게 패널티합니다. SyCL은 대규모 IR 데이터셋(~2M 샘플)을 생성하며, 이로 인해 복잡한 훈련 파이프라인을 피하고 데이터 품질 문제를 완화합니다.

- **Performance Highlights**: SyCL을 사용한 실험 결과, 제안된 방법은 InfoNCE 기반의 전통적인 훈련 방식에 비해 성능이 현저히 향상됨을 보여줍니다. BEIR 데이터셋의 제로샷 평가에서 SyCL은 36.8에서 43.2로 평균 nDCG@10 점수를 개선하여, 실제 라벨이 있는 문서로 훈련된 모델과 유사한 성능을 달성하였습니다. 이 결과는 실제 문서 없이도 강력한 순위 매기기 성능을 구현할 수 있음을 잘 보여줍니다.



### Aurelia: Test-time Reasoning Distillation in Audio-Visual LLMs (https://arxiv.org/abs/2503.23219)
- **What's New**: 이 논문에서는 오디오-비주얼 레이징(AV reasoning)의 복잡성을 해결하기 위해 새로운 프레임워크인 AURELIA를 소개합니다. AURELIA는 actor-critic (행동-비평자) 기반의 접근 방식을 활용하여, 테스트 시간에 AVLLMs(오디오-비주얼 대형 언어 모델)의 단계적 레이징 기능을 증진합니다. 논문은 또한 4500개의 오디오-비주얼 질문과 이에 대한 상세한 단계적 레이징을 제공하는 AVReasonBench 기준점을 제시합니다.

- **Technical Details**: AURELIA는 LLM(대형 언어 모델)의 추론 기능을 활용하여 다중 모드 오디오-비디오 이해를 위한 고품질 레이징 데이터를 생성하는 인터랙티브한 프레임워크입니다. 이 시스템은 오디오와 비주얼 신호의 상호 작용을 고려하여 각기 다른 레이징 작업을 수행하며, 기존의 AVLLMs가 가진 편향성을 줄이는 데 기여합니다. 또한, AVReasonBench는 다양한 레이징 능력을 평가하는 데 필요한 포괄적인 벤치마크를 제공합니다.

- **Performance Highlights**: AURELIA를 활용하여 AVLLMs의 성능이 최대 100% 상대적으로 개선되는 것을 확인하였습니다. AVReasonBench의 18개 기존 AVLLM에 대한 평가 결과, 이 모델들이 비디오와 오디오 데이터를 처리하는 데 현저한 한계를 보임을 보여주었습니다. 이러한 성능 증가는 오디오-비주얼 레이징의 중요성과 현실 세계에서의 응용 가능성을 강조합니다.



### RECALL-MM: A Multimodal Dataset of Consumer Product Recalls for Risk Analysis using Computational Methods and Large Language Models (https://arxiv.org/abs/2503.23213)
- **What's New**: 이 연구에서는 미국 소비자 제품 안전 위원회(CPSC)의 리콜 데이터베이스를 기반으로 multimodal dataset인 RECALL-MM을 개발했습니다. 이 데이터셋은 과거 정보에 기반한 데이터 주도적 위험 평가를 지원하며, 생성적 방법(generative methods)을 통해 확장됩니다. 데이터셋의 패턴은 개선된 안전 조치를 통해 큰 영향을 미칠 수 있는 특정 영역을 강조합니다.

- **Technical Details**: 연구에서는 2000년부터 2024년까지의 6,874개의 리콜 데이터를 수집하였고, 이를 대형 언어 모델(LLM)로 확장하여 위험 평가를 위한 새로운 분류 및 시각적 설명을 추가했습니다. 데이터셋의 각 항목은 위험 분류, 제품 카테고리, 치료 유형 등의 주요 리콜 속성을 포함하고 있으며, GPT-4o를 활용해 내용을 구조화했습니다. 연구는 또한 LLM을 이용해 제품 이미지만으로도 잠재적인 위험을 예측하는 방법론을 소개합니다.

- **Performance Highlights**: 사례 연구를 통해 리콜 데이터의 유틸리티를 증명하고 제품 위험을 식별하는 데 어떻게 기여하는지를 보여줍니다. 첫 두 가지 사례 연구는 설계자들이 리콜된 제품 간의 패턴을 시각화하고 새로운 제품 아이디어를 전반적인 리콜 환경 속에 위치시킬 수 있음을 보여줍니다. 마지막 사례 연구에서는 LLM을 활용하여 제품 이미지에 기반한 위험 예측의 강점과 한계를 강조하며, 설계 과정 전반에 걸친 위험 인식의 중요성을 부각시킵니다.



### Convolutional Neural Networks Can (Meta-)Learn the Same-Different Relation (https://arxiv.org/abs/2503.23212)
- **What's New**: 이 연구는 메타-러닝(meta-learning) 접근 방식을 사용하여 CNN(Convolutional Neural Network)이 '같음-다름(same-different)' 관계를 효과적으로 학습할 수 있는지를 조사합니다. 기존의 학습 방식으로는 CNN이 이러한 관계를 일반화하는 데 실패하는 반면, 메타-러닝 기술이 적용된 경우에는 성공적인 결과를 보였습니다. 이로 인해 CNN의 학습 능력에 대한 새로운 시각을 제시하고 있습니다.

- **Technical Details**: 연구에서는 MAML(Model-Agnostic Meta-Learning) 알고리즘을 사용하여 다양한 관련 작업에 대한 최적의 초기 가중치(initial weights)를 찾습니다. 이 기법은 다양한 작업 간의 공통 구조를 캡처하여 각각의 네트워크가 작업을 더 쉽게 학습할 수 있도록 돕습니다. 특히, 이러한 메타-러닝 접근 방식이 CNN의 일반화 능력을 어떻게 향상시키는지를 평가하며, 다양한 심층 CNN 아키텍처를 적용하여 '같음-다름' 과제를 수행했습니다.

- **Performance Highlights**: 결과적으로, 메타-러닝을 적용한 CNN 모델들은 새로운 자극에 대해 '같음-다름' 관계를 더 잘 일반화하는 경향을 보였으며, 기존의 CNN 모델들에 비해 성능이 크게 향상되었습니다. 이러한 발견은 CNN이 인간과 유사한 시각적 추론을 수행할 수 있는 잠재력을 지니고 있음을 시사합니다. 이 연구는 CNN의 메타-러닝을 통해 더욱 복잡한 관계를 이해하는 길을 여는 데 기여할 것으로 기대됩니다.



### The Challenge of Achieving Attributability in Multilingual Table-to-Text Generation with Question-Answer Blueprints (https://arxiv.org/abs/2503.23204)
- **What's New**: 이 논문에서는 저자들이 낮은 자원의 언어인 아프리카 언어로 구성된 TaTA 데이터셋에 대한 멀티링구얼(멀티언어) Table-to-Text 생성 작업에서 Question-Answer(QA) 청사진을 사용하여 결과의 신뢰성을 증대시키는 방법을 탐구하고 있습니다. 또한 이 작업은 첫 번째로 QA 청사진을 적용하여 신뢰성을 개선하기 위한 새로운 방법을 제안합니다. 저자들은 영어 예시에 대해서는 QA 청사진이 결과의 신뢰성을 높이는 데 효과적이라는 것을 발견했으나, 멀티링구얼 환경에서는 효과가 떨어진다고 보고했습니다.

- **Technical Details**: 저자들은 Seq2Seq 모델을 활용하여 입력된 표의 정보를 기반으로 유창하고 정확한 설명을 생성하는 작업을 수행했습니다. 이 논문에서는 QA 청사진이 포함된 TaTA 데이터셋에서 만큼 Seq2Seq 언어 모델을 미세조정(finetuning)했습니다. 그러나 멀티링구얼 환경에서는 영어에서 타겟 언어로 QA 청사진을 번역하는 과정에서 발생하는 부정확성으로 인해 제약이 있다고 합니다.

- **Performance Highlights**: QA 청사진을 사용한 결과, 모델이 영어 데이터에 대해 학습 및 평가되었을 때는 결과의 신뢰성이 향상된 것으로 나타났습니다. 그러나 멀티링구얼 환경에서는 기대하는 성과를 내지 못하였으며, 이는 영어에서 다른 언어로 번역되는 과정에서 발생하는 오류와 모델이 생성한 청사진을 제대로 활용하지 못하는 문제 때문이라고 분석되었습니다. 이 논문은 전반적인 성능 평가에 대한 깊이 있는 분석을 제공하여 향후 연구에 기초 자료를 제공하고 있습니다.



### Large Language Models are Unreliable for Cyber Threat Intelligenc (https://arxiv.org/abs/2503.23175)
- **What's New**: 최근 연구들에서는 Large Language Models (LLMs)가 사이버 보안 분야에서 데이터 홍수를 제어하는 데 효과적으로 사용될 수 있다고 주장하고 있습니다. 이러한 가능성을 확인하기 위해 본 논문에서는 CTI(Cyber Threat Intelligence) 업무에 대한 평가 방법론을 제시하고, LLM의 일관성과 신뢰도를 정량화할 수 있는 방법을 소개합니다. 실험 결과, LLM들이 실제 보고서에서 충분한 성능을 보이지 않으며 일관성이 부족하고 과신하게 되는 경향이 있음을 보여줍니다.

- **Technical Details**: 이 연구에서는 350개의 위협 정보 보고서를 기반으로 LLM을 평가했습니다. 모델의 효과성, 일관성 및 신뢰도 보정을 검사하기 위해 새로운 평가 파이프라인을 설계하고 배포했습니다. 연구 결과, few-shot learning 및 fine-tuning이 CTI에 대한 성능을 개선하는 데 중요한 영향을 미치지 않았으며, LLM이 생성한 정보의 일관성도 의문을 제기합니다.

- **Performance Highlights**: LLMs의 출력은 반복적인 질문에서도 일관성이 부족하고, 같은 CTI 보고서에서 서로 다른 결과를 생성하는 경향이 있습니다. 이는 패치 관리와 같은 CTI에서 심각한 위험 요소로 작용할 수 있습니다. 연구 결과에 따르면, LLM의 신뢰도 보정이 낮아 정보 추출 및 생성에서 신뢰성을 보장할 수 없는 것으로 나타났습니다.



### CodeARC: Benchmarking Reasoning Capabilities of LLM Agents for Inductive Program Synthesis (https://arxiv.org/abs/2503.23145)
- **What's New**: 새로운 평가 프레임워크인 CodeARC(코드 추상화 및 추론 챌린지)가 도입되었습니다. 이는 에이전트가 숨겨진 목표 함수를 질의하고 반복적으로 솔루션을 조정할 수 있는 상호작용 설정을 제공합니다. 기존의 정적 예제에 의존했던 평가 방식의 한계를 보완하며, 실제 상황을 반영할 수 있도록 설계되었습니다.

- **Technical Details**: CodeARC는 LLM 기반의 에이전트가 초기 입력-출력 예제 집합을 사용하여 작업을 시작하고, 새로운 입력으로 목표 함수를 질의하며, 미분 테스트 오라클을 통해 검증하고 디버깅할 수 있는 구조를 가집니다. 이 과정에서는 에이전트가 스스로 입력을 생성하고 피드백에 따라 솔루션을 수정해야 합니다.

- **Performance Highlights**: CodeARC를 사용한 실험 결과, 총 18개 모델 중 OpenAI의 o3-mini가 52.7%의 성공률로 가장 뛰어난 성과를 거두었습니다. 또한, LLaMA-3.1-8B-Instruct의 세부 조정(Fine-tuning)을 통해 최대 31%의 상대적 성능 향상을 달성했습니다.



### SupertonicTTS: Towards Highly Scalable and Efficient Text-to-Speech System (https://arxiv.org/abs/2503.23108)
Comments:
          19 pages, preprint

- **What's New**: SupertonicTTS는 텍스트-투-스피치(TTS) 시스템으로, 음성 합성을 위한 향상된 확장성과 효율성을 제공합니다. 이 시스템은 음성을 연속적인 잠재 표현으로 인코딩하는 음성 자동 인코더, 흐름 매칭(flow-matching) 기법을 사용하는 텍스트-투-잠재(text-to-latent) 모듈, 발화 수준의 재생시간(duration) 예측기로 구성되어 있습니다. 이로 인해 기존의 복잡한 TTS 파이프라인이 단순화되어 G2P 모듈이나 외부 정렬기가 필요 없어졌습니다.

- **Technical Details**: SupertonicTTS는 낮은 차원의 잠재 공간을 설계하고, 잠재 표현을 시간 축을 따라 압축하여 효율성을 높였습니다. ConvNeXt 블록을 모든 모듈에서 널리 사용하여 경량의 효율적인 아키텍처를 구현했습니다. 또한, 교차 주의(cross-attention) 매커니즘을 적용하여 텍스트-스피치 정렬을 효과적으로 처리함으로써, 더 많은 데이터 도메인이나 언어로 확장할 때의 병목 현상을 줄였습니다.

- **Performance Highlights**: SupertonicTTS는 4400만 개의 매개변수로 경쟁력 있는 퍼포먼스를 보여주며, 매우 빠른 생성 속도를 자랑합니다. 실험을 통해 고충실도 음성을 빠르게 재구성할 수 있음을 입증하였고, 맥락 공유 배치 확장 방식(context-sharing batch expansion)을 통해 메모리와 계산 효율성을 개선했습니다. 오디오 샘플이 포함된 링크는 https://supertonictts.github.io/  입니다.



### InkFM: A Foundational Model for Full-Page Online Handwritten Note Understanding (https://arxiv.org/abs/2503.23081)
- **What's New**: 이 논문에서는 손으로 쓴 디지털 노트의 내용을 정확하게 해석하고 이해하는 방법을 개발하기 위한 새로운 모델인 InkFM을 소개합니다. InkFM은 28종의 스크립트에서 텍스트를 인식하고, 수학적 표현을 인식하며, 페이지를 텍스트와 그림 같은 개별 요소로 구분할 수 있는 고유한 기능을 제공합니다. 특정 데이터셋에 대해 소스 모델을 미세 조정(fine-tuning)하여 페이지 세분화와 텍스트 인식의 품질을 더욱 향상시킬 수 있음을 입증했습니다.

- **Technical Details**: InkFM은 다채로운 혼합 작업에 대해 훈련되어 있으며, 세 가지 핵심 작업인 세분화(segmentation), 분류(classification), 인식(recognition)을 통합하여 하나의 강력한 모델로 발전시킵니다. 세분화 작업에서는 단어와 그림을 어떤 객체에 할당하고 각 객체를 분류하는 기술이 포함됩니다. 또한, 이는 손으로 쓴 텍스트를 문자 시퀀스로 변환하는 인식 작업을 통해, 다양한 글쓰기 스타일에 적응할 수 있는 능력을 보여줍니다.

- **Performance Highlights**: InkFM은 다양한 스크립트에서의 텍스트 인식 및 스케치 분류에서 뛰어난 성능을 보여주며, 특히 잉크 기반 스타일 작업에서 SoTA(state-of-the-art) 품질을 달성했습니다. 예를 들어, IAM 데이터셋에서 영어 손글씨 텍스트 라인 세분화에서 경쟁력 있는 결과를 보였고, QuickDraw 데이터셋에서는 최신 정확도를 기록했습니다. 이 모델은 고유한 성능으로 시각적 요소를 탐지하고, 노트의 개별 요소를 효과적으로 구분할 수 있습니다.



### Concorde: Fast and Accurate CPU Performance Modeling with Compositional Analytical-ML Fusion (https://arxiv.org/abs/2503.23076)
Comments:
          15 pages, 17 figures, To be published in ISCA 2025

- **What's New**: Concorde는 마이크로아키텍처 성능 모델링을 위한 새로운 방법론으로, 성능 추정을 간소화하여 대규모 설계를 신속하게 탐색할 수 있게 합니다. 기존의 사이클 단위 시뮬레이터와는 달리, Concorde는 프로그램의 성능 특징을 압축된 성능 분포를 통해 예측하며, 이를 위해 단순한 분석 모델을 활용합니다. 이 방법은 약 2%의 평균 CPI 예측 오차로 사이클 단위 시뮬레이터보다 5배 이상의 빠른 속도를 자랑합니다.

- **Technical Details**: 기존의 사이클 단위 시뮬레이션 방법의 한계를 인식하고 머신러닝 기법을 활용한 성능 모델링을 제안합니다. Concorde는 성능 예측을 위해 단순한 분석 모델과 경량화된 다층 퍼셉트론(MLP) 모델을 결합하여 작업을 분리합니다. 이 방법은 성능 분포를 바탕으로 반복적인 데이터 학습 없이도 높은 정확도를 유지하며, 성능 예측은 1초 이내에 완료됩니다.

- **Performance Highlights**: Concorde의 성능은 다양한 프로그램과 마이크로아키텍처에 대해 높은 정확성을 유지하며, 단일 신경망 평가로 성능 예측을 신속히 수행합니다. 이 시스템은 150 million CPI 평가를 필요로 하는 미세한 성능 기여도 분석을 포함하여 약 한 시간 내에 가능함을 보여줍니다. 전체 매개변수 공간 내에서 거의 즉각적인 성능 예측이 가능하게 하여, 엔지니어의 마이크로아키텍처 설계 탐색을 더욱 효율적으로 만듭니다.



### VLM-C4L: Continual Core Dataset Learning with Corner Case Optimization via Vision-Language Models for Autonomous Driving (https://arxiv.org/abs/2503.23046)
- **What's New**: 본 연구에서는 VLM-C4L이라는 지속적 학습(Continual Learning) 프레임워크를 제안합니다. 이 프레임워크는 비전-언어 모델(Vision-Language Models, VLMs)을 활용하여 코너 케이스 데이터를 동적으로 최적화하고 향상시키며, 코너 케이스에서 학습할 수 있는 방안을 제공합니다. VLM-C4L은 고품질 데이터 추출 및 데이터 재생 전략을 결합하여 모델이 다양한 코너 케이스에서 점진적으로 학습하면서도 기존의 일상적인 시나리오에서의 성능을 유지할 수 있도록 합니다.

- **Technical Details**: VLM-C4L은 VLMs를 이용하여 효과적으로 코너 케이스 데이터를 추출하고 분배하며 강화하는 방식으로 작동합니다. 이 프레임워크는 기존 모델의 능력을 유지하면서 코어 데이터로 학습을 조정하여 코너 케이스 작업 처리를 개선합니다. 또한, 불확실성을 기반으로 하는 코어 데이터 추출 기법을 통해 이전에 학습한 코너 케이스 시나리오를 업데이트하여 새로운 케이스에 적응할 때 지식을 유지할 수 있도록 합니다.

- **Performance Highlights**: 광범위한 실제 자율 주행 데이터 세트, 특히 Waymo 및 CODA에서 VLM-C4L의 성능을 평가한 결과, 다양한 코너 케이스 시나리오에서 일관되게 성능이 향상됨을 보여주었습니다. 실험 결과는 VLM-C4L이 안전성이 Critical한 자율 주행 시스템을 위한 효과적이고 일반화 가능한 접근법임을 증명합니다. 이 연구는 자율 주행 모델이 제한된 데이터 환경에서도 특정 코너 케이스 시나리오를 효과적으로 처리하고 다양한 케이스를 적응할 수 있는 가능성을 열어줍니다.



### Agentic Large Language Models, a survey (https://arxiv.org/abs/2503.23037)
- **What's New**: 최근 에이전틱 LLMs(agentic large language models)의 발전은 연구자들 사이에서 큰 관심을 받고 있습니다. 이 논문에서는 이러한 LLM들이 (1) 추론(reasoning), (2) 행동(action), (3) 상호작용(interaction)하는 능력을 갖춘다고 정의하며, 이에 대한 문헌을 체계적으로 정리하고 있습니다. 에이전틱 LLMs는 의학, 물류, 금융 등 다양한 분야에 활용되고 있으며, 자가 반성(self-reflection) 및 역할놀이(role-playing)는 새로운 연구의 가능성을 열어줍니다.

- **Technical Details**: 에이전틱 LLMs는 자연어 처리(natural language processing), 도구 통합(tool integration), 강화 학습(reinforcement learning) 등의 다양한 기술적 발전에 의존하고 있습니다. 논문은 세 가지 범주로 문헌을 나누어 에이전트가 어떻게 더 지능적으로 행동하고 상호작용할 수 있도록 발전해왔는지를 설명합니다. 이러한 기술적 발전은 에이전트들이 환경과 상호작용함으로써 새로운 훈련 데이터를 생성하고, 더 나아가 기존의 LLM 교육 방식을 보완하는 데 기여합니다.

- **Performance Highlights**: 이 논문은 에이전틱 LLM의 성능을 높이기 위한 연구 의제를 제시하며, LLM들이 의사 결정 및 협업 문제 해결에 어떻게 기여할 수 있는지를 강조합니다. 또한, LLM들이 자가 훈련(self-training) 수행을 통해 더 많은 훈련 데이터를 생성할 수 있는 기회를 제공함으로써 언어 모델이 계속해서 학습할 수 있는 방법리를 제시합니다. 그러나 LLM들이 현실 세계에서 행동할 경우 발생할 수 있는 위험 요소에 대해서도 경고하고 있습니다.



### Engineering Microbial Symbiosis for Mars Habitability (https://arxiv.org/abs/2503.23015)
Comments:
          25 pages, 1 figure

- **What's New**: 이번 연구에서는 화성 식민지 건설에 대한 새로운 접근 방식을 제시합니다. 합성 생물학(synthetic biology)과 유전 공학(genetic engineering)의 최근 발전을 활용하여, 지구의 극한 환경에서 생존하는 미생물과 화성의 가상의 생명체 간의 공생 관계를 구축할 수 있는 가능성을 탐구하고 있습니다.

- **Technical Details**: 연구의 핵심은 화성 환경에서 생존할 수 있는 생명체를 설계하는 방법입니다. 실험 디자인(experimental designs), 실험실 시뮬레이션(laboratory simulations), 생물공학(bioengineering) 접근 방식 등 다양한 기술적 요소가 통합되어 있습니다. 이 과정에서 생명체의 내구성(durability) 및 적응력(adaptability)이 중요한 요소로 작용합니다.

- **Performance Highlights**: 엔도심비오시스(endosymbiosis)의 자연적 사례를 바탕으로, 화성에서 지속 가능한 인간 거주를 위한 새로운 생명체 설계를 제안합니다. 연구 결과는 국제적 협력과 강력한 행성 보호 정책의 중요성을 강조하며, 화성에서 생명이 번창할 수 있는 잠재력을 보여줍니다. 궁극적으로, 이 연구는 인류의 행성 간 거주 및 탐사에 대한 비전을 진전시키는 데 기여하고 있습니다.



### XL-Instruct: Synthetic Data for Cross-Lingual Open-Ended Generation (https://arxiv.org/abs/2503.22973)
- **What's New**: XL-AlpacaEval은 대형 언어 모델(LLM)의 다국어 생성 성능을 평가하기 위한 새로운 벤치마크로 소개됩니다. 기존의 연구들에서 공통적으로 나타난 문제를 해결하기 위해 고품질의 합성 데이터를 생성하는 XL-Instruct 방법이 제안되었습니다. 이 방법을 활용한 세밀한 튜닝을 통해 모델의 성능이 크게 향상되었으며, 구체적으로 GPT-4o-Mini 대비 win rate가 7.4%에서 21.5%로 증가한 것으로 나타났습니다.

- **Technical Details**: 교차 언어 생성(cross-lingual generation)은 특정 언어로 된 질의를 이해하고 다른 언어로 응답을 생성하는 작업입니다. 연구자들은 기계 번역(MT)의 노이즈 문제와 정보 손실 문제를 지적하며, XL-Instruct라는 합성 데이터 생성 기술을 활용하여 고품질의 교차 언어 데이터를 대규모로 생성할 수 있음을 보여주었습니다. 이 방법이 적용된 모델은 영어나 다국어 생성 작업에서도 강력한 제로샷 전이(zero-shot transfer) 성능을 보였습니다.

- **Performance Highlights**: XL-Instruct의 활용으로 다양한 LLM의 교차 언어 성능이 일관되게 개선되었으며, 성능 평가에서 LLM의 'off-the-shelf' 성능이 낮음을 확인했습니다. 실험 결과, 성능 향상뿐만 아니라 다국어 후처리 단계에 XL-Instruct를 포함할 것을 강력히 추천합니다. 또한, XL-AlpacaEval 벤치마크와 XL-Instruct 데이터셋이 향후 교차 언어 LLM 연구에 기여할 것으로 기대됩니다.



### SUV: Scalable Large Language Model Copyright Compliance with Regularized Selective Unlearning (https://arxiv.org/abs/2503.22948)
- **What's New**: 이번 연구에서는 SUV (Selective Unlearning for Verbatim data)라는 새로운 선택적 학습 해제 프레임워크를 소개합니다. 이 프레임워크는 LLM이 저작권이 있는 콘텐츠를 기억하는 것을 방지하면서도 모델의 전체 유틸리티를 유지하도록 설계되었습니다. 저작권 침해 사례를 포착한 데이터셋을 구축하고, 직접 선호 최적화(Direct Preference Optimization)을 사용하여 기억된 콘텐츠를 대체하는 방안을 제시합니다.

- **Technical Details**: SUV 프레임워크는 전통적인 방법과 달리 슬라이딩 윈도우(sliding-window) 메커니즘을 사용하여 기억된 구간을 세분화하여 식별합니다. 또한, DPO를 통해 플라거리(표절)된 내용을 제거하고 무작위로 생성된 텍스트로 대체합니다. 이 과정에서 그라디언트 프로젝션(gradient projection)과 피셔 정보 정규화(Fisher information regularization)를 통합하여 모델의 성능 저하를 최소화합니다.

- **Performance Highlights**: 500개의 저명한 책으로 구성된 대규모 데이터셋을 사용하여 SUV의 성능을 검증하였으며, 저작권이 있는 콘텐츠의 기억을 크게 줄이면서도 관련 없는 작업에서의 성능에 미치는 영향은 미미하다는 사실을 입증했습니다. 우리의 접근 방식은 공개 기준에서도 우수한 성과를 보여주며, 기존의 방법들에 비해 유용성을 유지하면서 저작권 위험 완화에 효과적임을 강조합니다.



### Identifying Multi-modal Knowledge Neurons in Pretrained Transformers via Two-stage Filtering (https://arxiv.org/abs/2503.22941)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 자연어 처리(NLP)와 컴퓨터 비전 분야에서 다중 모달 대형 언어 모델(MLLMs)의 발전으로 이어졌습니다. 이러한 모델들은 시각과 언어의 통합된 이해를 가능하게 하지만, 내부 처리의 불투명성과 허위 정보 생성을 포함한 도전과제를 안고 있습니다. 이에 따라, MLLMs에서 지식의 위치를 명확히 하는 방법에 대한 필요성이 대두되고 있습니다.

- **Technical Details**: 본 연구에서는 MiniGPT-4라는 Transformer 기반 MLLM을 활용해 특정 지식과 연관된 뉴런을 찾는 방법을 제안합니다. 지식 뉴런을 추출하기 위해 두 단계의 필터링을 수행하며, 첫 번째 단계는 inpainting을 활용한 활성화 차이 필터링이고, 두 번째 단계는 GradCAM을 이용한 기울기 기반 필터링입니다. 이러한 방법은 MS COCO 2017 데이터셋을 사용한 이미지 캡션 생성 작업에서 그 효과성을 입증하였습니다.

- **Performance Highlights**: 실험 결과, BLEU, ROUGE, BERTScore를 통한 정량적 평가 및 활성화 히트맵을 이용한 정성적 평가에서 제안한 방법이 기존 방법들보다 더 높은 정확도로 지식을 찾을 수 있음을 보여주었습니다. 본 연구는 MLLMs의 지식 시각화 및 설명 가능성에 기여하며, 향후 지식 편집 및 제어의 가능성을 열어줍니다.



### Bi-Level Multi-View fuzzy Clustering with Exponential Distanc (https://arxiv.org/abs/2503.22932)
- **What's New**: 이 연구에서는 멀티 뷰 환경에서 퍼지 c-평균(FCM) 클러스터링을 확장하는 방법을 제안합니다. 첫 번째로, 우리는 지열 커널 계수(heat-kernel coefficients, H-KC)와 가중치 인자를 고려한 중앙 집중식 지수 멀티 뷰 FCM(E-MVFCM)을 소개합니다. 두 번째로, E-MVFCM과는 달리 자동으로 특징 및 가중치 인자를 동시에 계산하는 지수 이중 멀티 뷰 FCM 클러스터링(EB-MVFCM)을 제안합니다.

- **Technical Details**: H-KC는 양자장 이론(quantum field theory, QFT)의 중요한 도구이며, 이는 군집화 과정에서 열 커널의 생성을 단순화할 수 있도록 도와줍니다. E-MVFCM과 EB-MVFCM 모두 H-KC의 명시적 형태를 제시하여 복잡한 데이터의 클러스터를 인식하는 데 도움을 줍니다. 이 연구에서 제안된 알고리즘의 도구 및 기능은 공개된 URL을 통해 이용 가능할 것입니다.

- **Performance Highlights**: 본 논문에서는 E-MVFCM과 EB-MVFCM의 객관적 기능과 최적화 방법을 자세히 설명하고, 이들 알고리즘이 다양한 멀티 뷰 환경에서 클러스터 품질 개선에 기여할 수 있음을 보여줍니다. 이러한 새로운 알고리즘들은 불확실성 감소 및 더 나은 패턴 인식을 위한 강력한 방법을 제공합니다.



### Nested Stochastic Gradient Descent for (Generalized) Sinkhorn Distance-Regularized Distributionally Robust Optimization (https://arxiv.org/abs/2503.22923)
Comments:
          30 pages, 20 figures, 1 table

- **What's New**: 이번 연구는 비볼록(Nonconvex) 배급 강건 최적화(Distributionally Robust Optimization, DRO) 문제를 다루며, 이것을 일반화된 sinkhorn 거리(Generalized Sinkhorn Distance)로 정의된 불확실성 집합을 사용해 해결하고자 합니다. 이러한 접근은 서로 다른 확률 지지 및 발산 함수(Divergence Function)를 가진 분포의 불확실성을 효과적으로 모델링할 수 있게 해줍니다.

- **Technical Details**: 본 연구에서는 데이터 샘플에 의존하는 듀얼 변수(Dual Variable)를 포함한 중첩(이중) 확률 프로그래밍(Nested Stochastic Programming) 형태의 새로운 듀얼 구성을 도출하였습니다. 이 구성은 중첩 확률 경량하강법(Nested Stochastic Gradient Descent, SGD) 알고리즘을 이용하여 해결되는 동역학을 가지고 있으며, 이는 스토캐스틱 근사(Stochastic Approximation)를 통해 중첩 확률 기울기를 추정합니다.

- **Performance Highlights**: 대규모 데이터셋을 대상으로 한 모델 훈련 실험을 통해, 제안된 sinkhorn DRO 형식과 중첩 SGD 알고리즘이 데이터 분포 변화에 대한 머신러닝 모델의 강건성을 성공적으로 향상시킨 것을 증명했습니다. 특이한 점은, 처음에는 비볼록 및 비계속 손실 함수에 대한 해결법을 개발함으로써, 현업에서의 실용성을 고려한 것입니다.



### The Marine Debris Forward-Looking Sonar Datasets (https://arxiv.org/abs/2503.22880)
Comments:
          10 pages, 12 figures, Oceans Brest 2025 camera readyu

- **What's New**: 이번 연구에서는 해양 쓰레기를 감지하고 분류하기 위한 새로운 공공의 Forward-Looking Sonar (FLS) 데이터셋인 Marine Debris FLS를 제안합니다. 이 데이터셋은 수조(watertank), 회전대(turntable), 침수된 채석장(flooded quarry)의 세 가지 설정에서 수집된 다양한 이미지와 과제를 포함하고 있습니다. 이는 AI 시스템의 훈련을 위한 대규모 데이터셋의 부족 문제를 해결하는 데 기여할 것으로 기대됩니다.

- **Technical Details**: 이 데이터셋은 해양 쓰레기 객체와 관련된 다양한 컴퓨터 비전 작업을 수행하기 위한 자료를 제공합니다. 객체 분류(object classification), 객체 탐지(object detection), 의미론적 분할(semantic segmentation), 패치 매칭(patch matching), 비지도 학습(unsupervised learning) 등의 작업을 포함하고 있으며, 각각의 설정과 과제는 세부적으로 설명되어 있습니다. 특히, ARIS Explorer 3000 센서를 사용하여 정밀한 데이터 수집이 이루어졌습니다.

- **Performance Highlights**: 초기 기준(result benchmark)을 통해 데이터셋의 유용성이 입증되었습니다. 연구진은 수집된 데이터를 바탕으로 다양한 실험을 진행하고 있으며, 각 시나리오에 따른 데이터의 변동성과 업데이트된 객체 클래스 목록은 모델의 정확성을 높이는 데 기여할 것입니다. 이 데이터셋은 바다에 버려진 인위적 쓰레기를 탐지하고 이해하는 데 중요한 기초 자료로 사용될 것입니다.



### Nonhuman Primate Brain Tissue Segmentation Using a Transfer Learning Approach (https://arxiv.org/abs/2503.22829)
- **What's New**: 본 연구에서는 비인간 영장류(Non-Human Primates, NHP)의 뇌 조직 분할을 향상시키기 위한 새로운 접근방식으로 STU-Net과 transfer learning을 결합한 방법을 제안합니다. 이는 인간의 뇌 MRI 데이터에서 전이된 지식을 활용하여 NHP 뇌 MRI의 분할 정확도를 높이는데 기여합니다. 주목할 만한 점은 기존의 한계를 극복하여 NHP 뇌 특유의 미세 해부학적 세부 사항을 효과적으로 포착할 수 있다는 점입니다.

- **Technical Details**: 비인간 영장류의 뇌 이미지는 주로 해부학적 차이와 해상도 제한으로 인해 분할이 어렵습니다. 연구에서는 STU-Net을 통해 이러한 도전을 해결하고, 특히 작은 피질 하 구조체인 피각(putamen)과 시상(thalamus) 등의 분할 성능을 향상시켰습니다. 최종적으로, DSC(다중 클래스의 Dice Similarity Coefficient)는 0.88 이상, IoU(Intersection over Union)는 0.8 이상, 그리고 HD95는 7 이하의 성능을 달성했습니다.

- **Performance Highlights**: 제안된 방법은 비인간 영장류의 뇌 조직 분할에서 새로운 기준을 제시합니다. 특히 작은 구조물의 분할에서 기존의 한계를 초월한 성과를 보였으며, 이는 진화 신경 과학 및 인류 건강에 관련된 신경 질환의 전임상 연구를 가속화할 잠재력을 가지고 있습니다. 나아가, 이 연구는 다중 클래스 뇌 조직 분할에 대한 강력한 방법론을 제시하여 향후 연구의 발전에 기여할 것입니다.



### Quantum Doeblin Coefficients: Interpretations and Applications (https://arxiv.org/abs/2503.22823)
Comments:
          88 pages, 2 figures

- **What's New**: 본 논문에서는 고전 정보 이론의 Doeblin 계수를 양자 채널(quantum channel)에 일반화한 양자 Doeblin 계수를 조사합니다. 새로운 양자 Doeblin 계수를 정의하며, 이 중 하나는 효율적으로 계산 가능하고, 연결성과 곱셈성을 포함한 여러 바람직한 특성을 가지고 있습니다.

- **Technical Details**: 양자 Doeblin 계수는 최소 싱글렛 분수(minimal singlet fractions), 배제 값(exclusion values), 역 최대 상호 정보(reverse max-mutual information) 등 다양한 해석으로 발전됩니다. 이러한 해석은 채널을 활용하여 상태 배제 작업(state-exclusion tasks)에서 달성할 수 있는 최상의 오류 확률(error probabilities)과 비례함을 나타냅니다.

- **Performance Highlights**: 양자 Doeblin 계수는 양자 기계 학습 알고리즘, 오류 완화 프로토콜, 노이즈가 있는 양자 가설 테스트(sample complexity), 노이즈 모델의 공정성 및 시간 변화 채널의 혼합 시간(mixing times)에 이르기까지 다양한 응용 분야에서 활용됩니다. 또한, 도블린 계수를 사용하는 분석은 다양한 경계 제약을 개선해 주며, 이전 연구 대비 일반성과 효율성 측면에서도 우수한 결과를 제공합니다.



### Patronus: Bringing Transparency to Diffusion Models with Prototypes (https://arxiv.org/abs/2503.22782)
- **What's New**: 이 연구에서는 Denoising Diffusion Probabilistic Models (DDPMs) 기반의 해석 가능한 확산 모델인 Patronus를 소개합니다. Patronus는 ProtoPNet에서 영감을 받아 프로토타입 네트워크를 통합하여 생성 프로세스에 영향을 미치는 프로토타입을 추출하고 조절할 수 있습니다. 이 모델은 주석이나 텍스트 프롬프트 없이도 작동하며, 따라서 해석 가능성과 투명성을 증대시키는 새로운 경로를 제공합니다.

- **Technical Details**: Patronus의 설계는 이미지의 패치를 기반으로한 프로토타입 추출 및 표현 모듈을 포함하고 있습니다. 이 모듈은 입력 이미지를 패치 기반 특징 표현으로 변환하고, 각 프로토타입에 대한 유사성 점수를 계산하여 확산 과정의 조건으로 사용합니다. 또한, 프로토타입의 활성화 벡터를 통해 입력 이미지에 대한 의미 정보를 인코딩하는 방법을 제안합니다.

- **Performance Highlights**: Patronus는 세밀하게 의미 있는 시각적 특징을 잘 포착하며, 최신 SOTA (State-of-the-Art) 방법들과 비교해 경쟁력 있는 생성 품질을 달성합니다. 또한, 이 모델은 데이터셋 내에서 원치 않는 상관관계를 진단할 수 있는 잠재력을 보여주며, 생성 모델의 편향을 줄이고 공정성을 증진하는 데 유용한 도구로 활용될 수 있습니다.



### Policy Optimization and Multi-agent Reinforcement Learning for Mean-variance Team Stochastic Games (https://arxiv.org/abs/2503.22779)
- **What's New**: 이번 논문에서는 평균-분산 팀 확률 게임 (Mean-Variance Team Stochastic Game, MV-TSG)을 연구하였습니다. 각 에이전트는 공동의 평균-분산 목표를 최대화하기 위해 독립적으로 행동하지만, 두 가지 주요 도전 과제가 존재합니다. 첫째, 분산 메트릭이 동적 설정에서 가산적이지도 않고 마르코프적이지도 않습니다. 둘째, 모든 에이전트의 동시에 정책 업데이트가 각 에이전트에게 비정상적인 환경을 야기합니다.

- **Technical Details**: 이 논문에서는 감도 기반 최적화의 관점에서 MV-TSG를 분석합니다. 성능 차이 및 성능 미분 공식이 유도되어 MV-TSG에 대한 최적화 정보를 제공합니다. 이를 통해 결정론적 내시 균형 정책의 존재를 증명하였으며, 이후 순차 업데이트 방식을 사용하는 평균-분산 다중 에이전트 정책 반복(MV-MAPI) 알고리즘을 제안하였습니다. MV-MAPI 알고리즘은 목적 함수의 일차 정상점으로 수렴함을 증명했습니다.

- **Performance Highlights**: 제안된 알고리즘은 에너지 관리 문제를 위한 다중 마이크로그리드 시스템(Microgrid Systems) 최적화에 성공적으로 적용되었습니다. 실험 결과는 알고리즘의 효과성을 확인시켜 주었으며, 이 연구는 평균-분산 메트릭을 최적화하기 위한 정책 반복 기반 알고리즘과 다중 에이전트 강화 학습 알고리즘을 개발한 최초의 연구입니다. 따라서, MV-TSG의 해결책을 찾는 데 있어 중요한 기여를 합니다.



### Congenital Heart Disease Classification Using Phonocardiograms: A Scalable Screening Tool for Diverse Environments (https://arxiv.org/abs/2503.22773)
Comments:
          12 pages, 6 figures

- **What's New**: 이 연구는 선천성 심장 질환(Congenital Heart Disease, CHD)을 탐지하기 위한 심층 학습 모델을 제안합니다. 이 모델은 주로 방글라데시의 쉬슈 병원에서 데이터를 수집하고, 음향 심전도(Phonocardiogram, PCG) 신호를 기반으로 작동합니다. 선진국에서는 CHD 사망률이 감소하고 있지만, 저소득 및 중소득 국가에서는 여전히 높은 사망률을 보이고 있습니다.

- **Technical Details**: 이 연구에서는 Eko DUO ECG + 디지털 청진기를 사용하여 방글라데시에서 PCG 신호를 수집하였습니다. 환자들은 비CHD와 CHD로 진단된 건강한 개인으로, 각 환자는 네 가지 심장판막(삼첨판, 폐동맥판, 대동맥판, 승모판)에서 연속적으로 PCG 신호를 15초 동안 수집받았습니다. 총 3,435개의 PCG 신호가 수집되었으며, 그 중 456명은 CHD로 진단받았습니다.

- **Performance Highlights**: 모델은 방글라데시의 주 데이터셋에서 94.1%의 정확도, 92.7%의 민감도 및 96.3%의 특이도를 달성하였습니다. 또한 PhysioNet 도전과제 2022 및 2016에서도 강한 성능을 보이며, 다양한 인구집단과 데이터 소스에 대한 일반화를 검증하였습니다. 한 위치에서의 PCG 신호 수집 시에도 85% 이상의 정확도를 유지하며, 저품질 녹음에서도 80%의 정확도를 기록했습니다.



### Boosting Large Language Models with Mask Fine-Tuning (https://arxiv.org/abs/2503.22764)
- **What's New**: 이 논문에서는 Mask Fine-Tuning (MFT)이라는 새로운 LLM 파인튜닝 패러다임을 소개합니다. MFT는 모델의 무결성을 의도적으로 파괴함으로써 놀라운 성능 향상을 이끌 수 있음을 보여줍니다. 이 연구는 LLM 파인튜닝의 기존 프로토콜을 통해 구조적 무결성이 반드시 필요하지 않음을 주장합니다.

- **Technical Details**: MFT는 사전 훈련된 LLM에 대해 작동되며, 이 LLM은 전체 파인튜닝(Fine-Tuning)을 통해 훈련됩니다. 여기서 MFT는 이 훈련된 LLM에 이진 마스크를 추가하여 특정 파라미터를 선택적으로 마스킹합니다. 최적화는 스트레이트-스루 그래디언트 추정기(gradient estimator)를 활용하여 이루어지며, 이는 배포형 학습(supervised learning)으로 안내됩니다.

- **Performance Highlights**: MFT는 다양한 도메인과 백본을 통해 일관된 성능 향상을 보였습니다. 예를 들어, LLaMA2-7B와 LLaMA3.1-8B 모델을 사용하여 각각 1.95% 및 1.88%의 평균 성능 향상을 기록했습니다. 이 연구는 기존의 파인튜닝 방식과 비교하여 성능을 크게 향상시킬 수 있는 새로운 관점을 제공합니다.



### Malicious and Unintentional Disclosure Risks in Large Language Models for Code Generation (https://arxiv.org/abs/2503.22760)
Comments:
          The 3rd International Workshop on Mining Software Repositories Applications for Privacy and Security (MSR4P&S), co-located with SANER 2025

- **What's New**: 이 논문은 코드 생성을 위해 훈련된 대형 언어 모델(LLM)이 훈련 데이터에 포함된 민감한 정보를 누설할 위험을 탐구합니다. 이러한 위험은 ' 의도하지 않은 암기(unintended memorization)'라는 개념으로, 우발적 공개와 악의적 공개로 나누어집니다. 본 연구는 LLM이 사용자에게 민감한 정보를 누설할 가능성을 평가하여 이 위험을 측정하는 새로운 방법론을 제공합니다.

- **Technical Details**: 의도하지 않은 암기에는 LLM이 훈련 데이터에서 특정 정보를 추출하여 사용자 프롬프트에 대한 응답으로 세부 정보를 공개하는 경향이 포함됩니다. 이러한 현상은 API 키와 같은 기밀 정보가 포함된 코드 조각을 누설할 수 있음으로써 보안 및 법적 우려를 초래할 수 있습니다. 본 연구는 OLMo 모델과 Dolma 훈련 데이터셋을 평가하여 데이터를 구성하고 처리하는 방식에 따른 위험 증감의 연관성을 보여줍니다.

- **Performance Highlights**: 우리의 평가 결과, 훈련 데이터 출처 및 처리 변화가 의도하지 않은 암기 위험에 상당한 영향을 미치며, 동일한 운영 변경이 한 위험을 증가시키고 다른 위험을 완화할 수 있음을 보였습니다. 또한 모델이 처리하는 정보의 유형, 사용자 상호작용 패턴, 배포 컨텍스트에 따라 위험이 달라진다는 점을 발견했습니다. 이러한 기여는 LLM 훈련 데이터 공급망에서 필요한 개인 정보 보호 및 보안 테스트를 위한 데이터 마이닝(data mining)을 활용하고 있습니다.



### Multiple Embeddings for Quantum Machine Learning (https://arxiv.org/abs/2503.22758)
- **What's New**: 이번 연구는 현재의 양자 기계 학습(quantum machine learning) 방법들이 데이터 임베딩(data embedding) 전략에 과도하게 의존하게 되어 발생하는 한계점에 초점을 맞추고 있습니다. 제안된 새로운 프레임워크는 여러 데이터 임베딩 전략을 통합하여 다양한 데이터 세트를 처리할 때 양자 컴퓨팅의 다양성을 최대한 활용할 수 있도록 합니다. 실험 결과, 이 프레임워크는 기존의 최신 방법들보다 현저한 성능 향상을 보여줍니다.

- **Technical Details**: 양자 기계 학습 모델의 일반화 능력(generalization capability) 부족 문제를 다루며, 단일 데이터 인코딩에 의존하는 기존 모델의 제한점을 분석합니다. 제안된 새로운 네트워크 프레임워크는 다양한 데이터 임베딩을 통합하여 성능을 유지하면서도 특정 데이터 세트에서 최대 20% 성능 향상을 이룹니다. 이 연구는 데이터 리업로딩(classifier)와 같은 기존의 접근 방식과 대조되는 새로운 방법론을 제공합니다.

- **Performance Highlights**: 제안된 모델은 벤치마크(preliminary benchmarks)에서 평가되었으며, 기존의 최첨단 양자 기계 학습 모델들보다 유의미하게 우수한 성능을 기록했습니다. 이 연구로 인해 양자 기계 학습 분야에서의 일반화 문제가 해결될 가능성이 열리며, 향후 실용적인 응용 프로그램에서 뛰어난 성과를 거둘 것으로 기대됩니다.



### Concept Map Assessment Through Structure Classification (https://arxiv.org/abs/2503.22741)
- **What's New**: 이번 연구는 317개의 개념 맵(Concept Map, CM) 구조를 조사하여 그 구조를 스포크(spoke), 체인(chain), 네트워크(network) 세 가지 유형으로 분류하고, 이를 통해 다중 클래스 분류 모델을 훈련 시켰습니다. 연구의 결과, 분류 정확도는 86%에 달했으며, 이는 개념 맵 평가 시스템에서 학생들에게 실시간 피드백을 제공하는 데 활용될 수 있는 가능성을 가지고 있습니다. 이러한 자동화된 평가가 교육 현장에서 어떻게 활용될 수 있는지를 보여주며, 실질적인 교육적 기여를 할 수 있는 방안을 제시합니다.

- **Technical Details**: 연구에서는 다양한 기계 학습 모델을 사용하여 개념 맵 구조를 분류하는 데 필요한 특징(feature)을 추출하는 과정이 포함되었습니다. 데이터셋은 균형을 맞추기 위해 무작위로 100개의 샘플로 나뉘었고, 20%는 검증 데이터로 사용되었으며, 나머지는 모델 훈련에 활용되었습니다. 특히, 결정 트리 모델(Decision Tree)이 다른 모델들보다 우수한 성능을 보였고, 가장 중요한 특징들로는 노드 당 평균 엣지 수와 엣지의 표준 편차 등이 포함되었습니다.

- **Performance Highlights**: 결정 트리 모델을 통해 86%의 높은 정확도로 개념 맵을 분류할 수 있었습니다. 이 성과는 교육자들이 개념 맵을 효과적으로 평가하고 학생들에게 실시간 피드백을 제공할 수 있는 자동화된 시스템을 구현하는데 기여할 수 있는 가능성을 보여줍니다. 향후 이 연구는 반복적인 수작업 분류의 필요성을 줄여주고, 학생의 학습 방향을 조정하는 데 중요한 도구가 될 것이라 기대됩니다.



### Symmetry-Informed Graph Neural Networks for Carbon Dioxide Isotherm and Adsorption Prediction in Aluminum-Substituted Zeolites (https://arxiv.org/abs/2503.22737)
- **What's New**: 이 논문에서는 SymGNN이라는 그래프 신경망 아키텍처를 도입하였습니다. 이 모델은 재료의 대칭성을 활용하여 흡착 특성 예측을 개선하는 데 중점을 두고 있습니다. 대칭 연산을 메시지 전달 메커니즘에 통합함으로써, 모델은 서로 다른 제올라이트 구조에 대한 매개변수 공유를 향상시켜 일반화 능력을 개선합니다.

- **Technical Details**: SymGNN은 대칭 정보를 포함하여 메시지 전달 방식의 개선을 목표로 한 설계로, 구조적 불확실성을 줄여줍니다. 실험적으로 CO2 흡착의 주요 경향을 포착하며, Si/Al 비율의 영향을 효과적으로 모델링합니다. 이 모델은 광고흡착성 ㄴ의 특성 분석에 사용되며, 유전자 알고리즘(genetic algorithm)을 통해 가능한 알루미늄 분포를 유추하는 응용 사례를 나타냅니다.

- **Performance Highlights**: 실험 및 일반화 작업에서 SymGNN은 CO2 흡착의 주요 경향을 성공적으로 나타내며, 재료 효율성을 평가할 수 있게 합니다. 기계학습 모델이 실제 재료 연구에 효과적임을 보여주고, 실험 데이터 및 생성적 접근 방식(generative approaches)을 활용한 정밀 조정 가능성을 제시합니다.



### A Large-Scale Vision-Language Dataset Derived from Open Scientific Literature to Advance Biomedical Generalist AI (https://arxiv.org/abs/2503.22727)
- **What's New**: 이번 논문에서는 Biomedica라는 오픈 소스 데이터셋을 소개합니다. Biomedica는 600만 개의 과학 기사와 2400만 개의 이미지-텍스트 쌍을 포함하고 있으며, 이를 통해 생물의학 인공지능(AI) 시스템의 성능을 향상시키는 것이 목표입니다. 웹 서버를 통해 제공되는 확장 가능한 스트리밍 및 검색 API는 AI 시스템과의 통합을 용이하게 합니다.

- **Technical Details**: Biomedica 데이터셋은 다양한 분야의 생물의학 연구 문헌을 포함하여, 다양한 카테고리의 이미지를 수집합니다. 각 인스턴스는 기사 수치와 이미지 수준의 메타데이터를 포함하고 있으며, 세밀한 주석이 제공됩니다. 이 데이터셋은 대규모로 저장 및 관리하기 어려울 수 있으나, Hugging Face에서 호스팅되어 필요할 때마다 스트리밍할 수 있습니다.

- **Performance Highlights**: Biomedica 데이터셋을 활용하여 구축된 다양한 AI 모델은 이전 시스템을 초과하는 성능을 보여주었습니다. 특히, BMC-CLIP과 BMC-SmolVLM 모델은 각각의 작업에서 이전의 모델들과 비교하여 성능이 크게 향상되었습니다. BIOMEDICA Index는 AI 에이전트 시스템이 의료 지침 기반 질문에 답변할 수 있도록 하여, 실질적인 임상적 응용에 기여할 수 있는 가능성을 보여줍니다.



### Chirp Localization via Fine-Tuned Transformer Model: A Proof-of-Concept Study (https://arxiv.org/abs/2503.22713)
Comments:
          19 pages, 8 figures

- **What's New**: 이번 연구는 electroencephalogram (EEG) 스펙트로그램에서 특이한 환율(Chirp-like patterns)을 감지하기 위한 자동화 도구를 개발하는 데 초점을 맞추고 있습니다. 기존의 연구에서는 이러한 패턴을 발견하는 방법이 부족했으며, 본 연구는 Vision Transformer (ViT) 모델의 세부 조정을 통해 이를 해결하고자 하였습니다. 이와 더불어 저차원 적응(Low-Rank Adaptation, LoRA)을 활용하여 모델의 적응 속도를 향상시켰습니다.

- **Technical Details**: 연구에서는 100,000개의 합성 스펙트로그램을 생성하여 환율 로컬리제이션(chirp localization)을 위한 최초의 대규모 벤치마크를 구축했습니다. 이 스펙트로그램들은 선형 및 지수적 주파수 스윕( frequency sweep)과 가우시안 노이즈, 스무딩 기술을 사용하여 신경 환율(neural chirps)을 모방합니다. ViT 모델은 회귀(regression)에 맞게 조정되었으며, MSE 손실 및 AdamW 최적화를 통해 훈련되었고, 학습률 스케줄러와 얼리 스탑을 적용하여 과적합을 방지하였습니다.

- **Performance Highlights**: 모델의 성능은 예측된 라벨과 실제 라벨 간의 Pearson 상관관계를 통해 평가되었습니다. 결과는 환율 시작 시간에 대해 0.9841의 강한 상관관계를 보이며, 추론 시간은 137초에서 140초로 안정적인 결과를 나타냈습니다. 이러한 접근 방식은 EEG의 시간-주파수 표현(TFR)에서 환율 분석을 위한 효율적인 도구를 제공하며, 방법론적 공백을 메울 수 있을 것으로 기대됩니다.



### Risk-Calibrated Affective Speech Recognition via Conformal Coverage Guarantees: A Stochastic Calibrative Framework for Emergent Uncertainty Quantification (https://arxiv.org/abs/2503.22712)
- **What's New**: 이 연구는 극단적인 운전자의 감정으로 인한 교통 안전 문제에 대응하기 위해 신뢰성 있는 감정 인식 시스템의 필요성을 강조합니다. 기존의 딥러닝 접근법에서는 과적합(overfitting) 및 신뢰도 추정의 부정확성이 문제로 나타났습니다. 본 논문에서는 Conformal Prediction (CP)과 Risk Control를 통합한 새로운 프레임워크를 제안하여, Mel-spectrogram 특성을 이용한 감정 인식을 보다 정확하게 구현합니다.

- **Technical Details**: 본 연구의 핵심 혁신은 분류기의 예측이 주어진 입력과 얼마나 잘 일치하는지를 측정하는 비순응 점수(nonconformity score)를 개발한 것입니다. 이를 통해 사용자 정의 위험 수준에 따른 통계적으로 엄격한 임계값을 도출하고, 예측 세트를 구축합니다. Risk Control 프레임워크를 통해 특정 작업에 맞게 손실 함수를 조정하며, 예측 세트의 크기를 동적으로 조절할 수 있습니다.

- **Performance Highlights**: IEMOCAP 및 TESS 데이터세트를 기반으로 한 교차 데이터 실험에서는 엄격한 커버리지(coverage) 보장과 함께 평균 예측 세트 크기(APSS)와 위험 수준 간의 강한 부정적인 상관관계를 보여주었습니다. 이 연구는 APSS를 분류 불확실성을 평가하기 위한 새로운 척도로 제안하며, 고급 감정 인식 시스템에서의 신뢰성을 크게 향상시키는 데 기여합니다.



### Modeling speech emotion with label variance and analyzing performance across speakers and unseen acoustic conditions (https://arxiv.org/abs/2503.22711)
Comments:
          11 pages, 5 figures

- **What's New**: 이번 연구는 스피치 감정 인식 모델에서 레이블의 불확실성을 다루기 위해 그레이더의 결정을 확률 밀도 함수(probability density function)로 활용하는 접근법을 제안합니다. 일반적으로 사용되는 합의 등급(consensus grades) 대신, 감정 점수의 확률 밀도 함수를 목표로 하여 이를 통해 모델의 성능을 향상시킬 수 있음을 보여주고 있습니다. 더불어, 다양한 음성 감정 모델의 성능을 평가하기 위해 다수의 테스트 세트와 성별, 화자에 따른 성능 분석도 중요함을 강조합니다.

- **Technical Details**: 연구팀은 MSP-Podcast 데이터셋을 활용하여 약 238시간의 영어 스피치 데이터를 분석했으며, 각 샘플의 감정 범주는 수동으로 할당된 발란스, 활성화 및 지배 점수를 통해 정의됩니다. 연구에서는 사전 훈련된 모델(pre-trained model)에서 감정 인식에 최적의 표현을 얻기 위해, 주요 FM(layer)에서의 계층 중요도를 분석하는 방법을 탐구하였습니다. 특히 TC-GRU 모델을 사용하여 HuBERT, WavLM, Whisper 등의 다양한 모델로부터 스피치 임베딩을 생성하고 이를 기반으로 감정 인식을 수행합니다.

- **Performance Highlights**: 모델의 성능 분석 결과, 여러 테스트 세트를 통해 비교할 때, 전통적인 1최적 가설(1-best hypothesis)이 훈련 데이터의 불균형에 의해 편향될 수 있음을 발견했습니다. 이로 인해 2-또는 3-최적 가설(2- or 3-best hypotheses)을 사용하는 것이 더 현실적인 음성 샘플의 포함 감정을 고려하는 데 유용하다는 결과가 도출되었습니다. 연구는 다양한 평가 지표와 데이터 세트를 통한 성능 평가가 모델 선택에 있어 중요함을 분명히 하고 있습니다.



### Enhancing Aviation Communication Transcription: Fine-Tuning Distil-Whisper with LoRA (https://arxiv.org/abs/2503.22692)
Comments:
          14 pages, 4 Figures, 4 Tables, Under review by Journal of Aerospace Information Systems

- **What's New**: 본 논문은 항공 통신의 문서화(Transcription)에 있어 최신 인공지능 기술을 적용하여 정확성을 향상시키기 위한 연구입니다. 특히 OpenAI의 Whisper 모델을 항공 통신에 맞게 미세 조정(Fine-tuning)하는 방법을 다룹니다. 이를 통해 효율적으로 Whisper의 한 버전인 distil-Whisper를 미세 조정하는 Parameter-Efficient Fine-tuning 방법인 Low-Rank Adaptation을 활용했습니다.

- **Technical Details**: 이 연구에서는 약 70시간 분량의 항공 교통 통제 데이터세트(Air Traffic Control Corpus)를 사용하여 실험을 진행했습니다. 또한 LoRA(Low-Rank Adaptation)의 하이퍼파라미터를 설정하기 위해 그리드 서치(Grid Search) 및 5-겹 교차 검증(5-fold Cross-validation)을 적용했습니다. 이 과정에서 Alpha = 64 및 Rank = 32를 초기 하이퍼파라미터로 설정하고 최적의 조합을 찾아냈습니다.

- **Performance Highlights**: 미세 조정 과정 후, 모델의 평균 단어 오류율(Word Error Rate)은 3.86%로 측정되어 매우 우수한 성능을 보였습니다. 이 결과는 항공기의 조종실(Cockpit)에서의 적용 가능성을 입증해 주며, 논문이 제시하는 방법론이 향후 항공 통신의 효율성을 높일 수 있음을 시사합니다.



### Truth in Text: A Meta-Analysis of ML-Based Cyber Information Influence Detection Approaches (https://arxiv.org/abs/2503.22686)
Comments:
          15 pages, 2 figures, 5 tables, 2 appendices

- **What's New**: 사이버 정보의 영향력, 또는 일반적으로 잘못된 정보는 사회의 발전과 정부의 안정성에 대한 가장 큰 위협 중 하나로 여겨진다. 특히 디지털 플랫폼의 전략적 사용을 통해 정보 왜곡이 일어나는 현상은 사이버 보안의 심각한 위협으로 간주된다. 이 연구는 머신 러닝(ML) 기술을 활용하여 온라인에서의 잘못된 정보 탐지의 효과를 메타 분석을 통해 정량적으로 평가하고, 다양한 ML 모델 유형의 성과를 조사하였다.

- **Technical Details**: 본 연구에서는 총 81개의 ML 탐지 기법을 샘플링하여 평균 79.18%의 정확도로 잘못된 정보를 탐지할 수 있음을 발견하였다. ML 모델은 전통적인 알고리즘(Support Vector Machines, Random Forest)과 딥 러닝 모델(Convolutional Neural Networks, Long Short-Term Memory networks, Transformer)이 포함되며, 이들 모델 간의 성능 차이는 통계적으로 유의미하지 않았다. 그러나 그룹 간 변동성이 높다는 점이 발견되어, 이는 다각적인 연구가 필요함을 나타낸다.

- **Performance Highlights**: 연구 결과에 따르면, ML 기반의 정보 탐지 기법들은 전반적으로 80% 이상의 정확도를 보였다. 이는 향후 잘못된 정보 탐지 방법의 복제 및 개발을 위한 기초 자료로 활용될 수 있으며, 특히 ML 모델 수준에서의 확장을 추천한다. 최종적으로, 연구는 데이터 세트 불일치, 평가 방법론 및 피처 엔지니어링 기술 등이 탐지 성과의 일반화된 전략에 대한 합의 부족에 기여하고 있음을 강조하였다.



### Pharmolix-FM: All-Atom Foundation Models for Molecular Modeling and Generation (https://arxiv.org/abs/2503.21788)
- **What's New**: 이번 연구에서는 PharMolixFM이라는 통합 프레임워크를 제안하여 구조 생물학에서의 분자 모델링 및 생성의 새로운 방향을 제시합니다. 이 프레임워크는 다중 생성 기술을 기반으로 하여 모든 원자(all-atom) 기반 모델을 구축하는 데 중점을 두고 있습니다. 또한, 이 연구는 작동 속도와 예측 정확성을 동시에 향상시키기 위해 혁신적인 방법론을 적용합니다.

- **Technical Details**: PharMolixFM은 다중 모드(multi-modal) 생성 모델을 통합하여 화합물의 원자 유형 및 좌표를 함께 캡처합니다. 이 프레임워크는 기능별 사전 정보를 통해 분자 작업을 일반화된 노이즈 제거 과정으로 구성하여, 다중 훈련 및 샘플링 전략의 영향을 체계적으로 연구할 수 있도록 합니다. 특히, 각각의 모델은 다중 생성 과정에서 다양한 우선 순위를 적용하여 통합된 접근 방식을 제공합니다.

- **Performance Highlights**: 실험 결과, PharMolixFM-Diff가 단백질-소분자 도킹 작업에서 AlphaFold3와 비교하여 83.4%의 예측 정확성을 나타내며, 훨씬 빠른 추론 속도를 자랑합니다 (약 4.6초 소요). 또한, 구조 기반 약물 설계에서는 PharMolixFM 모델이 생성하는 분자의 약물 가능성(druggability)이 일관되게 개선되는 결과를 보여주었습니다.



### The Mathematical Relationship Between Layer Normalization and Dynamic Activation Functions (https://arxiv.org/abs/2503.21708)
Comments:
          New title, renamed DyISRU, added missing parentheses in proof of theorem 3, minor language corrections

- **What's New**: 최근 연구에서 제안된 Dynamic Tanh (DyT)는 layer normalization (LN)을 대체할 수 있는 방법입니다. 이 접근법은 실제적으로 유용하지만, 이론적인 근거가 부족하였습니다. 본 논문에서는 LN과 동적 활성화 함수 간의 수학적 관계를 규명하고, DyT를 LN에서 유도하는 방법을 제시하고 있어 이론적 이해를 심화하고 있습니다.

- **Technical Details**: DyT 함수는 LN의 특정 수학적 유도를 통해 개발되며, 이 과정에는 미분 방정식을 해결하는 단계가 포함됩니다. 연구자들은 LN의 입력에 대한 미분을 계산하고, 이를 단순화하여 DyT 함수를 도출하였습니다. 이러한 과정에서 정밀한 근사가 필요하다는 것을 발견하였으며, 이를 제거함으로써 Dynamic Inverse Square Root Unit (DyISRU)라는 대체 기능을 제안했습니다.

- **Performance Highlights**: DyISRU는 layer normalization의 정확한 대응 개념으로, 수치적으로 DyT보다 LN에 더 정확히 유사하다는 것을 증명했습니다. 이 연구는 변동성(variance) 가정에서 벗어나 새로운 요소별 변환을 제공함으로써 layer normalization을 대체할 가능성을 제시하고 있습니다. DyT는 사전 조정이 필요할 수 있지만, DyISRU는 훨씬 더 안정적인 성능을 기대할 수 있는 장점이 있습니다.



### Efficient Learning for Entropy-Regularized Markov Decision Processes via Multilevel Monte Carlo (https://arxiv.org/abs/2503.21224)
Comments:
          46 pages, 6 figures; fixed formatting of definitions and titles

- **What's New**: 이번 연구에서는 큰 상태 및 행동 공간을 가진 상태 공간의 효율적인 학습 알고리즘 설계를 다루고 있습니다. 특히 엔트로피 정규화된 Markov 결정 과정(MDP)에 초점을 맞추고 있으며, 이론적 성능을 보장하는 새로운 다층 몬테 카를로(MLMC) 알고리즘을 제안하고 있습니다. 이 알고리즘은 Bellman 연산자의 근사화와 관련하여 더 나은 샘플 복잡도를 달성할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 본 연구는 Polish 상태 및 행동 공간을 갖는 MDP를 대상으로 하며, 오라클(oracle)이라고 불리는 생성 모델을 통해 상태-행동 쌍이 입력될 때 즉각적인 비용과 다음 상태를 반환하는 구조를 강조합니다. 제안된 MLMC 알고리즘은 Bellman 연산자를 근사하여, 편향된 몬테 카를로 추정이 제공하는 quasi-polynomial 샘플 복잡도를 가져오는 것과는 대조적으로, 비편향 랜덤화된 다층 근사가 polynomial 샘플 복잡도를 달성할 수 있음을 보여줍니다.

- **Performance Highlights**: 이론적 성능 보장은 수치 실험을 통해 검증되었으며, 특히 제안된 접근법은 상태 및 행동 공간의 크기와 무관하게 성능을 발휘할 수 있음을 강조합니다. 기존 알고리즘들이 일반적으로 상태 및 행동 공간의 크기에 따라 복잡도가 증가하는 것과는 달리, 본 연구의 알고리즘은 이러한 문제를 해결하는 데 중점을 두고 있습니다. 따라서 실제 환경에서 효율적으로 사용할 수 있는 가능성을 갖 춥니다.




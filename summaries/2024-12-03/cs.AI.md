New uploads on arXiv(cs.CL)

### Critical Tokens Matter: Token-Level Contrastive Estimation Enhence LLM's Reasoning Capability (https://arxiv.org/abs/2411.19943)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 대형 언어 모델(LLMs)의 추론 작업에서 중요한 역할을 하는 "critical tokens"를 식별하는 방법을 제안합니다. 특히, 비정상적인 추론 경로에서 오류를 유발하는 토큰들이 어떻게 잘못된 결과를 초래하는지를 분석합니다. 이를 통해 cDPO라는 새로운 접근법을 개발하여, 이러한 critical tokens에 대한 보상을 자동으로 제공하는 방법을 모색합니다. 이는 기존 DPO 알고리즘을 확장하여 구현되었습니다.

- **Technical Details**: cDPO는 기존의 DPO 알고리즘에서의 한계를 극복하기 위해, 올바른 경로와 잘못된 경로에서 각각 훈련된 모델을 사용하여 critical tokens를 효과적으로 식별합니다. 'Contrastive estimation' 접근법을 통해 두 모델의 생성 가능성을 비교함으로써, 비정상적인 경로에서의 critical tokens를 자동으로 식별할 수 있습니다. 이런 방식으로 수집된 정보는 최종 추론 결과를 향상시키기 위한 보상 신호로 활용됩니다.

- **Performance Highlights**: 실험 결과는 cDPO가 여러 기초 전략보다 우수한 성능을 나타낸다는 것을 보여주었습니다. 특히, 다양한 벤치마크에서 p<0.005의 유의미한 결과를 기록하며, 기존의 예시 중심 및 단계 중심의 방법들보다 더 나은 효과를 발휘했습니다. 이는 추론 작업에서 critical tokens의 중요성을 강조하며, 적용 가능한 다양한 상황에서 cDPO의 유용성을 입증합니다.



### On Domain-Specific Post-Training for Multimodal Large Language Models (https://arxiv.org/abs/2411.19930)
- **What's New**: 최근 다수의 다양한 분야에 적응할 수 있는 다중 모달 대형 언어 모델(MLLM)의 발전이 두드러진 성과를 내고 있습니다. 하지만 특정 과학 분야나 산업 응용 분야에 대한 모델의 적응은 상대적으로 덜 탐구되었습니다. 본 논문은 포스트 트레이닝(post-training)을 통한 MLLM의 도메인 적응(domain adaptation)을 체계적으로 연구하며, 데이터 생성(data synthesis), 트레이닝 파이프라인(training pipeline), 작업 평가(task evaluation)에 초점을 맞춥니다.

- **Technical Details**: 논문에서는 도메인 특화 이미지-자막 쌍을 사용하여 다양한 비주얼 지침 작업을 생성하는 비주얼 지침 합성기를 개발하였습니다. 생성된 합성 작업은 수동 규칙이나 GPT-4 및 GPT-4V에서 생성된 작업보다 MLLM의 도메인 특화 성능 향상에 효과적입니다. 또한, 단일 단계 교육 파이프라인을 적용하여 각 트레이닝 예제에서 합성 작업과 이미지-자막 쌍을 결합하여 작업 다양성을 증대시킵니다.

- **Performance Highlights**: 두 개의 도메인인 생물 의학(biomedicine)과 식품(food)에서 다양한 MLLM의 성능을 평가하였으며, 결과적으로 우리의 모델인 AdaMLLM은 여러 도메인 특화 작업에서 일반 MLLM보다 지속적으로 우수한 성능을 보였습니다. 이러한 결과는 도메인 특화 포스트 트레이닝이 MLLM의 성능 개선에 효과적임을 입증합니다.



### AIDetx: a compression-based method for identification of machine-learning generated tex (https://arxiv.org/abs/2411.19869)
- **What's New**: 이 논문에서는 데이터 압축 기술을 활용하여 기계 생성 텍스트를 탐지하는 새로운 방법인 AIDetx를 소개합니다. 기존의 딥 러닝 분류기와 달리 AIDetx는 높은 계산 비용과 해석 가능성의 제한을 극복하기 위해 압축 기반의 분류 프레임워크를 제안합니다. 이 방법은 인간이 작성한 텍스트와 AI가 생성한 텍스트에 대해 서로 다른 압축 모델을 구축하여 입력 데이터를 분류합니다.

- **Technical Details**: AIDetx는 유한 컨텍스트 모델(Finite-Context Models, FCM)을 활용하여 압축을 통한 텍스트 분류를 구현합니다. 각 클래스에 대해 기준 텍스트를 압축하여 모델을 구축하고, 새로운 문서가 도입되면 가장 높은 압축 비율을 제공하는 클래스에 할당됩니다. 이 과정에서 문서와 교육 텍스트 간의 교차 엔트로피를 최소화하는 클래스에 문서가 할당되며, FCM의 매개변수를 최적화하여 성능을 향상시킵니다.

- **Performance Highlights**: AIDetx는 두 개의 벤치마크 데이터셋에서 F1 점수가 각각 97%와 99%를 초과하는 성과를 달성하였습니다. 기존의 대규모 언어 모델(LLMs)과 비교했을 때, AIDetx는 해석 가능성과 계산 효율성을 제공하며, 훈련 시간과 하드웨어 요구 사항을 대폭 줄입니다. 전체 구현은 공개적으로 제공되어 있으며, 사용자는 명령줄 인터페이스로 모델 훈련 및 텍스트 분류 작업을 수행할 수 있습니다.



### Reverse Thinking Makes LLMs Stronger Reasoners (https://arxiv.org/abs/2411.19865)
Comments:
          20 pages

- **What's New**: 이 논문에서는 Reverse-Enhanced Thinking(RevThink)이라는 새로운 프레임워크를 소개합니다. RevThink는 데이터 증강(data augmentation)과 학습 목표(learning objectives)를 통해 대형 언어 모델(LLMs)에서 역방향 사고(reverse thinking)를 수행할 수 있도록 합니다. 이 방법은 원래 질문과 답변 외에도 앞에서의 추론(forward reasoning)과 뒤에서의 질문(backward question), 그리고 뒤에서의 추론(backward reasoning)을 포함하여 데이터를 증강합니다.  

또한 RevThink는 세 가지 주요 목표를 설정해 학생 모델(student model)을 훈련시키며, 이를 통해 모델이 질문으로부터 올바른 종합적 추론을 생성할 수 있도록 돕습니다.

- **Technical Details**: RevThink는 Chain-of-Thought와 같이 매우 구조화된 방식으로 추론을 수행하는 대형 모델을 활용합니다. 데이터 증강을 통해 생성된 두 가지 방향의 사고를 통해 학생 모델이 동시에 앞뒤로 사고할 수 있도록 돕습니다. 목표 설정은 (1) 질문에서 올바른 전진 추론을 생성, (2) 원래 질문으로부터 역 질문을 생성, (3) 역 질문으로부터 역 추론을 생성하는 것입니다. 이러한 방법론은 학생 모델이 테스트 시에도 효율적인 계산을 유지할 수 있도록 합니다.

- **Performance Highlights**: RevThink는 12개의 다양한 데이터셋에서 실험을 수행했으며, 평균적으로 학생 모델의 제로샷 성능(zero-shot performance) 대비 13.53% 향상된 성능을 보였습니다. 또한, 기존의 심볼릭 지식 증류(Symbolic Knowledge Distillation, SKD) 방법보다 6.84% 더 나은 성과를 기록했습니다. 데이터셋의 양이 적은 경우에도 학습한 정확도가 높아, 10%의 훈련 샘플만으로도 전체 데이터셋을 활용한 SKD의 성능을 초과할 수 있습니다. RevThink는 다양한 신규 데이터셋에서 또한 일반화(generalization)에 강점을 보입니다.



### What fifty-one years of Linguistics and Artificial Intelligence research tell us about their correlation: A scientometric review (https://arxiv.org/abs/2411.19858)
Comments:
          26 pages, 15 figures

- **What's New**: 이번 연구는 1974년부터 2024년까지의 51년간의 언어학(linguistics)과 인공지능(AI) 간의 강한 상관관계를 다룬 과학계량학(scientometric) 분석을 제공합니다. 이 연구는 5750개의 웹 오브 사이언스(Web of Science)에 색인된 기사와 2124개의 저널을 포함하며, 20835명의 저자가 794개 국가의 13773개 연구 센터에 소속되어 있음을 나타냅니다.

- **Technical Details**: 연구에 사용된 두 가지 강력한 소프트웨어인 CiteSpace와 VOSviewer는 지적 지형(intellectual landscape), 트렌드 이슈(trending issues), 그리고 재부각된 핫스팟(hotspots)의 시각적 맵을 생성하는 데 활용되었습니다. 1980년대와 1990년대에는 언어학과 AI 연구가 불안정하게 진행되었으며, 시간이 지남에 따라 publication이 불규칙했습니다.

- **Performance Highlights**: 하지만 2023년 이후 이 분야는 급격한 Publication 증가를 목격했으며, 2023년에는 1478개의 논문이, 2024년 1-3월 동안에는 546개의 논문이 발표되었습니다. 이 과정에서 새로운 문제와 핫스팟이 등장하고, 새로운 주제와 응용 프로그램이 개발되어 ChatGPT와 같은 강력한 딥 러닝 언어 모델을 포함하고 있습니다.



### Artificial intelligence contribution to translation industry: looking back and forward (https://arxiv.org/abs/2411.19855)
Comments:
          20 pages, 4 figures

- **What's New**: 이번 연구는 1980년부터 2024년까지 인공지능(AI)이 번역 산업에 기여한 연구를 종합적으로 분석하였습니다. 총 13220개의 논문을 WoS, Scopus, Lens의 세 가지 소스에서 수집하였으며, 과학계량(scientometric) 및 주제 분석(thematic analysis)을 통해 주요 클러스터, 주제 카테고리, 키워드 및 연구 센터를 조사하였습니다. 이 연구는 번역 산업에서 AI의 기여가 초기에는 미미했음을 보여주며, 기계 번역 기술 발전의 필요성을 강조합니다.

- **Technical Details**: 이 연구에서는 13220개의 논문을 기반으로 과학계량 분석을 실시하고, 관심 있는 18개의 논문을 선정하여 주제별로 검토하였습니다. 주요 분석 지표로는 burstness(버스트니스), centrality(중심성) 등이 사용되었으며, 각 논문들의 목적, 접근 방식, 발견 및 기여도를 정리하였습니다. 특히 초기의 규칙 기반 기계 번역과 통계적 기계 번역의 한계점에 대해 언급하면서도, 현재는 Neural Networking Algorithms 및 Deep Language Learning Models의 적용을 통해 번역의 질이 크게 향상되었음을 설명합니다.

- **Performance Highlights**: AI 기술의 발전에 따라 기계 번역의 성능이 향상되어 ChatGPT와 같은 최신 모델이 개발되었습니다. 그러나 여전히 저자원 언어, 다중 방언 및 자유 어순 언어와 같은 특정 문제를 해결하기 위한 철저한 연구가 필요하다는 것을 강조합니다. 마지막으로, 문화적·종교적 맥락에서 번역의 질을 높이기 위한 연구의 중요성을 언급하고 있습니다.



### Sensitive Content Classification in Social Media: A Holistic Resource and Evaluation (https://arxiv.org/abs/2411.19832)
- **What's New**: 이번 연구는 소셜 미디어 콘텐츠의 민감한 내용을 효과적으로 감지하기 위한 통합 데이터셋인 X-Sensitive를 소개합니다. 이 데이터셋은 갈등 언어, 저속한 표현, 성적인 내용, 약물 관련 콘텐츠, 자해, 스팸 등 6개의 민감한 카테고리를 다룹니다. 연구진은 기존의 데이터셋들이 다루지 못했던 민감한 내용을 인식하는 데 필요한 다양한 카테고리를 포괄한 데이터셋을 구축하여, 그 유용성을 입증하고 있습니다.

- **Technical Details**: X-Sensitive는 일관성 있는 검색 전략과 가이드라인을 바탕으로 수집되고 주석이 달린 데이터로 구성되어 있습니다. 본 연구에서는 대규모 언어 모델인 LLMs가 이 데이터셋를 통해 fine-tuning 되면, 기존 모델들보다 10-15%의 성능 향상을 가져오는 것을 확인했습니다. Research에 의한 데이터의 세부 카테고리 간 상호작용 및 다양한 주석자 특성의 변화를 연구하는 방식으로, 이 데이터셋은 다양한 마모드까지도 효과적으로 다룹니다.

- **Performance Highlights**: 성능 분석 결과, 80억 개의 파라미터를 가진 LLM이 X-Sensitive에서 fine-tuning 될 경우 최상의 결과를 나타냈습니다. 기존의 공개된 LLM들보다 10-15% 높은 성과를 보였으며, 특히 저희가 개발한 모델이 의미 있는 진전을 나타냈음을 강조하고 있습니다. 이 데이터셋과 성능이 뛰어난 모델들은 오픈 소스로 제공되어 접근 가능함을 알리고 있습니다.



### SDR-GNN: Spectral Domain Reconstruction Graph Neural Network for Incomplete Multimodal Learning in Conversational Emotion Recognition (https://arxiv.org/abs/2411.19822)
Comments:
          17 pages, 8 figures

- **What's New**: 본 논문은 대화 감정 인식에서 불완전한 다중 모드 학습을 위한 Spectral Domain Reconstruction Graph Neural Network (SDR-GNN)를 제안합니다. 기존의 그래프 신경망(GNN)은 노드 간의 이진 관계에만 초점을 맞추므로, 복잡한 고차원 정보를 포착하는 데 한계가 있었습니다. SDR-GNN은 연사와 문맥 간의 관계에 기반하여 발화 간의 감정적 의존성을 모델링할 수 있도록 감정 상호작용 그래프를 구성합니다.

- **Technical Details**: SDR-GNN은 발화의 의미 상호작용을 슬라이딩 윈도우를 사용하여 모델링하며, 각 발화의 일관된 의미 특징을 추출하기 위해 가중 관계 집합을 활용합니다. 또한, 스펙트럼 도메인에서 다중 주파수 집합을 수행하여 고주파 및 저주파 정보를 동시에 활용하여 불완전한 모드를 효율적으로 복구합니다. 마지막으로, 다중 헤드 어텐션을 적용하여 감정 인식을 위한 특징들을 융합하고 최적화합니다.

- **Performance Highlights**: 다양한 실제 데이터셋에 대한 광범위한 실험을 통해 SDR-GNN이 기존의 최첨단 방법들보다 우수한 성능을 나타냄을 입증하였습니다. SDR-GNN은 부족한 모드 데이터에서 더욱 효과적으로 감정 인식을 수행할 수 있는 특징을 제공합니다. 이러한 실험 결과는 다중 모드 학습의 새로운 가능성을 제시하며, 현재 MERC 분야의 연구 발전에 기여할 것입니다.



### INCLUDE: Evaluating Multilingual Language Understanding with Regional Knowledg (https://arxiv.org/abs/2411.19799)
- **What's New**: 이 논문은 다국어 대규모 언어 모델(LLM)의 성능 차이를 줄이기 위해 INCLUDE라는 새로운 평가 리소스를 개발했습니다. 총 197,243개의 QA 쌍을 포함하며, 44개 언어의 지역적 맥락을 고려한 평가가 가능하도록 설계되었습니다. 또한, 이 연구는 다국어 LLM의 공정하고 효과적인 배포를 위한 새로운 접근 방식을 제안합니다.

- **Technical Details**: INCLUDE 데이터셋은 52개 국가에서 수집된 다양한 시험 자료를 기반으로 구성됩니다. 평가를 위해, 학문적 시험, 전문 인증 및 지역 면허 시험 등 세 가지 유형의 시험을 포함합니다. 이 과정에서 질문은 원본 언어와 스크립트로 제공되며, 지역 및 문화적 지식을 평가하기 위해 다양한 질문 유형이 포함됩니다.

- **Performance Highlights**: 실험 결과, 현재의 모델들은 INCLUDE 평가에서 서로 다른 언어 간 성능 차이가 크고, 지역 지식이 필요한 질문에서 특히 낮은 점수를 기록했습니다. 또한, 모델은 지역 언어에 대해 훈련되지 않은 경우 성능이 제한적임을 보여주었으며, 글로벌 관점을 도입하면 성능 향상에 기여할 수 있음을 입증하였습니다.



### A Deep Learning Approach to Language-independent Gender Prediction on Twitter (https://arxiv.org/abs/2411.19733)
- **What's New**: 이번 연구는 Twitter 사용자들의 성별을 예측하기 위한 실험을 소개합니다. 언어에 구애받지 않는 언어적 특징을 기반으로 하여, 6개의 서로 다른 언어로 작성된 트윗들을 포함하는 TwiSty 데이터셋에서 수행되었습니다. 인공지능 모델로는 로지스틱 회귀(Logistic Regression, LR)와 피드포워드 신경망(Feed-Forward Neural Network, FFNN)을 활용하였습니다.

- **Technical Details**: 실험은 두 가지 설정에서 진행되었습니다: 상호 언어적(Inter-Lingual, IL) 설정과 교차 언어적(Cross-Lingual, CL) 설정입니다. IL 설정에서는 동일한 언어에 대해 훈련 및 테스트가 이루어진 반면, CL 설정에서는 이탈리아어와 독일어 데이터셋이 테스트 셋으로 남겨두고 나머지를 훈련 및 개발 세트로 합쳤습니다. IL 설정에서 LR이 가장 높은 정확도를 기록했으며, CL 설정에서는 3개의 숨겨진 층을 가진 FFNN이 최고 성과를 보였습니다.

- **Performance Highlights**: 훈련 세트의 크기가 작을 경우, 신경망 기반 모델이 전통적인 모델에 비해 성능이 낮았으나, 충분한 양의 데이터가 주어졌을 때 전통적인 모델을 큰 차이로 능가하는 결과를 보였습니다. 또한, 피처 분석을 통해 성별에 따라 남성과 여성이 사용하는 글쓰기 스타일이 다름을 확인하였습니다.



### Towards Santali Linguistic Inclusion: Building the First Santali-to-English Translation Model using mT5 Transformer and Data Augmentation (https://arxiv.org/abs/2411.19726)
- **What's New**: 이 논문은 인도, 방글라데시, 부탄, 네팔에서 약 700만 명의 사용자가 사용하는 산탈리(Santali) 언어에 대한 기계 번역 모델을 개발하고자 하는 최초의 시도를 다룹니다. 산탈리는 거의 세 번째로 많이 사용되는 오스트로아시아틱(Austroasiatic) 언어로, 저자들은 기존의 산탈리 언어 번역 모델이 없다는 점에 주목했습니다. 이 연구의 목적은 산탈리를 자연어 처리(NLP) 스펙트럼에 포함시키는 것입니다.

- **Technical Details**: 이 논문은 사용 가능한 산탈리 코퍼스를 기반으로 산탈리 번역 모델을 구축할 가능성을 조사했습니다. 결과적으로, 저자들은 전이 학습(transfer learning)이 산탈리 언어에 효과적으로 작용할 수 있음을 증명했습니다. 연구에서는 mt5와 같은 트랜스포머(transformer) 모델이 훈련되지 않은 트랜스포머보다 산탈리-영어 평행 코퍼스에서 더 좋은 성과를 보이며, 또한 mT5는 방글라(Bangla)보다 훨씬 더 많은 영어 데이터로 훈련된 결과 산탈리-영어의 성능이 더 우수하다는 점을 밝혔습니다.

- **Performance Highlights**: 데이터 증강(data augmentation)을 통해 저자들은 산탈리 번역 모델의 성능이 향상되었다고 보고했습니다. 연구 결과는 산탈리 언어의 기계 번역이 가능하다는 것을 나타내며, 향후 이 언어의 저자원(low-resource) 문제를 해결하는 데 기여할 수 있을 것으로 기대됩니다. 이 연구는 산탈리 언어가 더욱 글로벌하게 인식될 수 있는 기초를 마련한 것으로 평가됩니다.



### TakeLab Retriever: AI-Driven Search Engine for Articles from Croatian News Outlets (https://arxiv.org/abs/2411.19718)
- **What's New**: TakeLab Retriever는 크로아티아 뉴스 매체의 기사를 검색하고 수집하며 의미 분석을 수행하는 AI 기반 검색 엔진입니다. 이는 연구자들이 일반 검색 엔진으로는 제공받기 어려운 트렌드와 패턴을 발견할 수 있는 중요한 도구가 됩니다. 이 보고서에서는 TakeLab Retriever의 사용법과 설계 과정, 그리고 소프트웨어 엔지니어링의 도전 과제를 자세히 설명합니다. 최첨단 자연어 처리(NLP) 기술을 통해 사용자는 웹 애플리케이션을 통해 주제별 기사를 신속하게 검색할 수 있습니다.

- **Technical Details**: TakeLab Retriever는 한국어와 같은 저자원 언어의 뉴스 콘텐츠 분석을 위해 설계된 특별한 의미 검색 엔진입니다. 이 엔진은 품사(POS) 태깅, 토큰화, 의존 구문 분석, 개체명 인식(NER), 다중 레이블 주제 모델 등의 NLP 모델을 통합하여 기사를 분석합니다. 이를 통해 사용자는 기사의 비편향 데이터에 접근하며, 통계 및 시각화를 통해 온라인 크로아티아 뉴스의 역사와 현황을 깊이 있게 탐구할 수 있습니다. 또한, 이 엔진은 연구 질문에 따라 정보를 정확하고 편향 없이 검색할 수 있게 해줍니다.

- **Performance Highlights**: TakeLab Retriever는 2022년 11월부터 공공에 제공되었으며 현재(2024년 11월 기준) 33개 크로아티아 뉴스 매체의 1천만 개 이상의 기사를 분석하고 있습니다. 사용자는 복잡한 쿼리를 통해 기사를 손쉽게 검색하고, 다양한 기준으로 필터링함으로써 질 높은 분석을 수행할 수 있습니다. 웹 애플리케이션은 사용자 친화적인 인터페이스를 제공하며, 기사의 메타데이터와 통계 정보를 그래프 형태로 시각화하여 연구자들이 트렌드를 탐색하고 분석하는 데 도움을 줍니다.



### MIMDE: Exploring the Use of Synthetic vs Human Data for Evaluating Multi-Insight Multi-Document Extraction Tasks (https://arxiv.org/abs/2411.19689)
- **What's New**: 이 논문에서는 Multi-Insight Multi-Document Extraction (MIMDE)이라는 새로운 작업을 정의하고, 이를 통해 문서 집합에서 최적의 통찰을 추출하는 방법을 탐구합니다. MIMDE는 특히 설문 응답 분석, 의료 기록 처리 등 실제 응용 프로그램에서 많은 도움을 줄 수 있습니다. 연구진은 이를 위한 평가 프레임워크를 개발하고, 인간과 합성 데이터셋을 도입하여 LLM의 평가 가능성을 분석합니다.

- **Technical Details**: MIMDE 작업은 문서 집합에서 특정 정보(통찰)를 추출하고, 이를 원래 문서에 다시 매핑하는 두 단계로 구성됩니다. 이 과정에서 통찰은 인간 분석가에게 유용한 정보이자 여러 문서에서 공유 및 활용할 수 있는 정보입니다. 연구진은 20개의 최신 LLM 모델을 대상으로 하여 두 가지 데이터셋의 성능을 비교하고, 합성 데이터의 잠재적 사용 사례를 평가합니다.

- **Performance Highlights**: 분석 결과, 두 데이터셋에서 LLM이 통찰을 추출하는 능력 간에 0.71의 강한 상관관계가 발견되었습니다. 그러나 합성 데이터는 문서 수준 분석의 복잡성을 포착하지 못하는 한계를 보였습니다. 이러한 결과는 합성 데이터의 장단점을 조명하며 텍스트 분석 시스템 평가에 대한 중요한 지침을 제공합니다.



### ChineseWebText 2.0: Large-Scale High-quality Chinese Web Text with Multi-dimensional and fine-grained information (https://arxiv.org/abs/2411.19668)
Comments:
          ChineseWebTex2.0 dataset is available at this https URL

- **What's New**: 이번 논문에서는 MDFG-tool이라는 새로운 툴 체인을 제안하여, 다차원적이고 세분화된 정보를 가진 고품질 중국어 데이터를 대규모로 구축하는 방법을 설명합니다. 이 툴 체인은 기본적으로 두 단계로 나뉘며, 첫 번째 단계에서는 수작업으로 적용된 규칙을 사용하여 원자료에서 노이즈를 제거합니다. 두 번째 단계에서는 품질 평가 모델, 도메인 분류기, 독성 평가 모델의 하위 모듈을 포함하여 깨끗한 데이터의 품질과 안전성을 평가합니다.

- **Technical Details**: MDFG-tool은 특히 BERT 기반 품질 평가 모델, FastText 도메인 분류기, 독성 평가 모델을 결합하여 각 텍스트에 대한 품질 점수와 도메인 및 독성 레이블을 부여합니다. 이러한 세 가지 유형의 세분화된 정보를 통합하여 연구자들이 특정 요구 사항에 맞는 데이터를 쉽게 선택할 수 있도록 돕습니다. 최종적으로 중국어 데이터셋인 ChineseWebText2.0을 출시하며, 데이터셋의 규모는 3.8TB로, 각 텍스트는 품질 점수와 함께 다양한 레이블과 점수를 가지고 있습니다.

- **Performance Highlights**: ChineseWebText2.0은 현재 공개된 데이터셋 중에서 가장 큰 독성 레이블이 있는 텍스트 데이터셋의 일부를 포함하고 있어, 중국어 LLM의 독성 평가 및 안전성을 크게 향상시킬 수 있습니다. 연구자들은 이 데이터셋을 활용하여 다양한 시나리오에 맞춘 고품질의 도메인 특정 LLM 개발에 기여할 수 있습니다. 이 툴 체인의 최종 목표는 LLM의 발전을 위한 필수적이고 다차원적인 데이터베이스를 제공하는 것입니다.



### Truth or Mirage? Towards End-to-End Factuality Evaluation with LLM-OASIS (https://arxiv.org/abs/2411.19655)
Comments:
          15 pages. To be submitted to CL journal

- **What's New**: 이 논문에서는 LLM-Oasis라는 새로운 리소스를 소개합니다. LLM-Oasis는 사실 검증(factuality evaluation)용 end-to-end 평가자를 교육하기 위해 설계된 가장 큰 데이터셋으로, Wikipedia에서 주장을 추출하고 이를 바탕으로 사실적인 텍스트와 비사실적인 텍스트 쌍을 생성하는 방식으로 구성됩니다. 이 연구는 LLM의 환각 문제를 해결하기 위한 새로운 접근법을 제시합니다.

- **Technical Details**: LLM-Oasis는 두 가지 주요 단계를 통해 생성됩니다: 첫 번째로, Wikipedia에서 주장을 추출하고 일부 주장을 허위로 만들며, 그 결과 사실적인 텍스트와 비사실적인 텍스트 쌍을 만들어냅니다. 두 번째로, 인간 평가자가 데이터셋의 품질을 검증하고 사실성 평가 시스템의 벤치마킹을 위해 금 표준 테스트 세트를 생성합니다. 이러한 방법론은 LLM의 성능을 평가하는 데 중요한 기초 자료를 제공합니다.

- **Performance Highlights**: LLM-Oasis를 사용한 실험 결과, 최신 LLM인 GPT-4o가 60%의 정확도로 end-to-end 사실성 평가 작업을 수행하는 데 도달했습니다. 이는 LLM-Oasis가 현재의 기술 수준에서 상당한 도전 과제가 되고 있음을 보여줍니다. 이 연구는 사실성 평가 분야의 미래 연구를 이끌어 갈 잠재력을 강조합니다.



### LLM Teacher-Student Framework for Text Classification With No Manually Annotated Data: A Case Study in IPTC News Topic Classification (https://arxiv.org/abs/2411.19638)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이번 연구는 다국어 뉴스 분류 모델을 위한 새로운 teacher-student 프레임워크를 제안합니다. 이 프레임워크는 Generative Pretrained Transformer (GPT)를 교사 모델로 사용하여 슬로베니아어, 크로아티아어, 그리스어 및 카탈로니아어 뉴스 기사를 자동으로 주석 처리하여 IPTC Media Topic 훈련 데이터셋을 개발합니다. 이를 통해 수동 데이터 주석 작업 없이도 비교적 작은 모델 크기로 다국어 분류기를 구축할 수 있게 됩니다.

- **Technical Details**: 연구진은 LLM(large language models)을 활용하여 뉴스 기사의 주제를 자동으로 주석 처리하는 방법론을 제시합니다. GPT 모델을 교사 모델로 활용하여 자동으로 생성된 데이터셋은 BERT와 유사한 세부적인 학생 모델을 훈련하는 데 사용됩니다. 이 모델들은 적은 수의 훈련 인스턴스로도 높은 성능을 달성할 수 있으며, 제로샷(Zero-shot) 크로스링구얼(즉, 언어 간 전이 능력) 성능을 보임으로써 다국어 처리의 유용성을 강조합니다.

- **Performance Highlights**: 학생 모델은 교사 모델과 유사한 성능을 달성하며, 4개 언어에서 높은 제로샷 성능을 보여줍니다. 특히, 제로샷 크로스링구얼 능력이 뛰어나며, IPTC Media Topic 스키마의 최상위 카테고리를 이용한 다국어 분류 작업에 적합하도록 설계되었습니다. 이번 연구의 결론은 이 모델이 XLM-RoBERTa 모델이 지원하는 100개 언어에 적용 가능하다는 점이며, Hugging Face 리포지토리에 공개된 최고 성능 분류기로 실질적인 활용이 가능합니다.



### Can Large Language Models Reason about the Region Connection Calculus? (https://arxiv.org/abs/2411.19589)
Comments:
          13 pages. arXiv admin note: text overlap with arXiv:2309.15577

- **What's New**: 이번 연구는 Qualitative Spatial Reasoning(QSR) 분야에서 대형 언어 모델(LLMs)이 classical qualitative spatial reasoning 작업을 수행할 수 있는지를 조사한다. 특히, RCC-8(mereotopological Region Connection Calculus)에 대해 LLM의 성능을 평가하기 위해 여러 실험 쌍을 수행하였다. 이 연구는 LLM이 기존의 symbolic spatial reasoner를 대체할 수 있는지 알아보는 데에도 중점을 두고 있다.

- **Technical Details**: 연구에서는 RCC-8에 대한 세 가지 실험 쌍을 수행하여 composition table의 복원, 인간의 composition 선호도에 대한 정렬, 개념적 이웃(conceptual neighbourhood) 복원을 평가하였다. 각 실험은 LLM이 relation names에 대한 지식을 얼마나 의존하는지를 테스트하기 위해 익명 관계와 이름이 있는 관계를 사용하여 진행되었다. 각 인스턴스는 랜덤성을 측정하기 위해 30번 반복되었다.

- **Performance Highlights**: 대형 언어 모델의 성능이 기존의 QSR에 적용 가능한지에 대한 질문은 중요하며, 이를 통해 LLM이 스스로 reasoning 메커니즘을 구축할 수 있는 가능성을 탐색하고 있다. 연구 결과, LLM이 QSR의 일부 구성 작업에서 예상보다 뛰어난 성능을 보였음을 확인하였다. 그러나 이전 연구에서 지적된 바와 같이 LLM의 reasoning 능력에는 한계가 있을 수 있으며 이는 향후 연구에 중요한 고려 요소가 될 것이다.



### In-Context Learning with Noisy Labels (https://arxiv.org/abs/2411.19581)
- **What's New**: 본 논문에서는 'noisy labels'와 함께하는 in-context learning이라는 새로운 과제를 제안합니다. 이는 실제 세계의 데이터셋에서 발생하는 라벨의 왜곡을 다루기 위한 것입니다. 기존 연구들은 유용한 예제를 선정하는 것에 주안점을 두었으나, 노이즈가 포함된 라벨로 인해 성능 저하를 겪고 있다는 점을 간과했습니다.

- **Technical Details**: 새로운 문제 제시는 LLM이 입력 쿼리와 관련된 데모를 기반으로 작업을 수행하는 과정에서 노이즈가 포함된 라벨에 대한 해결책을 제공하는 것입니다. 본 연구는 노이즈가 섞인 라벨이 포함된 데이터셋을 사용하여 라벨 예측 과정에서의 문제를 다루고 있으며, 이를 위해 네 가지 기준 방법(correction, weighting, reordering, selection)을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안하는 방법은 noisy labels로 인한 성능 저하를 방지하는 데 효과적임을 보여줍니다. 이는 in-context learning 과정에서 안정성을 높이며, LLM이 작업을 더 잘 학습할 수 있도록 돕습니다. 이러한 연구는 향후 LLM의 실제 응용에 중요한 기여를 할 것으로 기대됩니다.



### ICPR 2024 Competition on Multilingual Claim-Span Identification (https://arxiv.org/abs/2411.19579)
Comments:
          To appear at ICPR 2024

- **What's New**: 이번 논문은 소셜 미디어 게시물에서 잘못된 정보(misinformation)와 가짜 뉴스(fake news)를 포함할 수 있는 주장(claim)을 자동으로 식별하는 방법에 대해 논의합니다. 이 연구는 'Claim Span Identification'(CSI)이라는 과제를 통해, 주어진 텍스트에서 주장과 관련된 특정 부분을 식별하는 도전적인 작업을 수행했습니다. 이 과제는 단순히 주장을 식별하는 이진 분류(binary classification) 문제를 넘어서, 언어의 의미(semantics)를 깊게 이해해야 하는 복잡한 작업입니다.

- **Technical Details**: 제안된 CSI 과제는 HECSI라는 새로운 데이터셋에서 수행되었으며, 이 데이터셋은 영어와 힌디어에서 약 16,000개의 게시물로 구성되어 있습니다. 각 게시물에는 인간 주석자(hand annotators)가 표시한 주장 구간(claim spans)이 포함되어 있으며, 두 가지 언어의 다양한 도메인에서의 주장을 포함하여 모델의 일반화(generalizability) 능력을 향상시키고자 했습니다. 주석자들은 트위터(twitter)에서 자주 접하는 다양한 주장 유형에 대한 예시를 학습한 후, 각 게시물에서 주장과 같은 문장이 있는지 판단하고 이를 최소 범위로 나타내도록 지시받았습니다.

- **Performance Highlights**: 총 9개 팀이 경쟁에 참여했으며, 대부분은 인도에 위치해 있지만 미국과 방글라데시에서의 참여도 있었습니다. 대회 주최측은 BERT 기반의 단순한 베이스라인 모델을 제공했으며, 참가 팀들은 다양한 방법으로 이 모델을 개선하기 위해 두 가지 방법을 제출할 수 있었습니다. 평가 방법으로는 Macro-F1 점수와 Jaccard 점수를 사용하여, 참가 팀들의 성과를 정량적으로 측정하였습니다.



### KV Shifting Attention Enhances Language Modeling (https://arxiv.org/abs/2411.19574)
Comments:
          22 pages

- **What's New**: 본 논문에서는 최근 대규모 언어 모델에서의 induction heads 메커니즘을 재조명하고, 이를 개선하기 위한 KV shifting attention이라는 새로운 접근 방식을 제안합니다. 기존의 induction heads는 두 개 이상의 레이어를 요구하지만, KV shifting attention은 단일 레이어에서도 효과적으로 induction heads 기능을 구현할 수 있도록 돕습니다. 이 방법의 이점은 모델의 깊이와 폭에 대한 요구사항을 줄이며 더 효율적인 학습을 가능하게 한다는 것입니다.

- **Technical Details**: KV shifting attention은 attention 메커니즘에서 키(key)와 값(value)을 분리해 구조적 요구를 없애는 혁신적인 방법입니다. 이를 통해 induction heads의 복잡성을 줄이고, 단일 레이어 변환기(transformer)로도 효과적인 induction 작업을 수행할 수 있습니다. 이 이론적 분석을 통해 KV shifting attention이 기존의 다층 변환기와 비교하여도 동등하거나 더 나은 성능을 발휘할 수 있음을 입증합니다.

- **Performance Highlights**: 실험 결과, KV shifting attention은 induction heads 학습 및 언어 모델링에 긍정적인 영향을 미치며, 10억 개 이상의 파라미터를 가진 프리트레인(pre-trained) 모델에서도 빠른 수렴과 향상된 성능을 보여줍니다. 이 방법은 다양한 규모의 모델에 걸쳐 보다 효율적이고 효과적인 언어 모델링을 가능하게 하며, 연구자는 이를 바탕으로 언어 모델의 성능을 극대화할 수 있음을 확인했습니다.



### Ensemble Watermarks for Large Language Models (https://arxiv.org/abs/2411.19563)
Comments:
          9 pages in the main body. Code is available at this http URL. arXiv admin note: substantial text overlap with arXiv:2405.08400

- **What's New**: 본 논문에서는 다중 특징 방법(multi-feature method)을 통해 LLM (Large Language Models)의 출력에 대한 고유 워터마크를 생성하는 방안을 제안합니다. 기존의 워터마크는 유연성이 부족하고 의역(paraphrasing) 공격에 취약한데, 이를 해결하기 위해 다양한 스타일 측정기술과 적합한 레드-그린(red-green) 워터마크 기능을 통합합니다. 이를 통해 98%의 높은 탐지율(detection rate)을 달성하며, 의역 공격 후에도 95%의 탐지율을 유지합니다.

- **Technical Details**: 제안된 앙상블 워터마크(ensemble watermark) 접근 방식은 LLM의 토큰과 문장 수준의 로짓(logits)을 변경하여 새로운 스타일 측정기술을 포함합니다. 특히, 이 방법은 다양한 기능을 조합할 수 있어 여러 요구사항 요구를 충족시키는 유연성을 제공합니다. 또한 모든 앙상블 구성에 대해 동일한 탐지 함수(detection function)를 사용 가능하여 유연성을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 앙상블 워터마크 접근 방식은 세 가지 LLM과 세 가지 다른 매개변수 설정(parameter settings)에서 의역 공격에 대한 저항력이 기존 방식보다 우수함을 보여줍니다. 또한, 단일 특징의 레드-그린 워터마크와 비교했을 때 49%의 탐지율에 그치던 것이 앙상블 구성으로 전체적으로 최고의 탐지율을 확보하는 것으로 나타났습니다. 이를 통해 LLM 데이터 생성에서의 책임성과 사회적 피해를 방지할 수 있는 가능성을 제시합니다.



### Initialization using Update Approximation is a Silver Bullet for Extremely Efficient Low-Rank Fine-Tuning (https://arxiv.org/abs/2411.19557)
Comments:
          Kaustubh Ponkshe and Raghav Singhal contributed equally to this work

- **What's New**: LoRA Silver Bullet(LoRA-SB) 메서드는 대형 언어 모델(LLM)의 파라미터 효율적인 미세 조정을 위한 새로운 접근 방식을 제안합니다. 고유한 초기화 전략을 사용하여 낮은 차원 부분 공간에서의 전체 미세 조정을 근사하는 방법입니다. 이 연구는 LoRA-XS의 아키텍처가 최적의 성능을 달성하기 위한 결정적인 조건을 제공한다고 이론적으로 입증했습니다.

- **Technical Details**: LoRA-SB는 고정된 매트릭스들 사이에 학습 가능한 (r × r) 매트릭스를 삽입하여 필요한 매개변수 수를 27~90배 줄이면서도 성능 향상을 이룹니다. 이 초기화 전략은 파라미터가 낮은 초기 그라디언트를 최적의 방식으로 근사하도록 설계되었습니다. 고차원 그라디언트 업데이트에 최적화된 스케일링을 가능하게 하여 하이퍼파라미터 조정의 필요성을 제거합니다.

- **Performance Highlights**: 수학적 추론, 일반 상식 추론, 언어 이해 과제를 포함한 4444개의 모델과 16161616개의 데이터셋을 통해 LoRA-SB는 기존 LoRA와 비교하여 성능이 뛰어난 결과를 보였습니다. 특히, 표준 LoRA보다 훨씬 적은 매개변수를 사용하면서도 개선된 성과를 나타내며, 이는 낮은 차원 부분 공간에서 전체 미세 조정을 시뮬레이션할 수 있다는 점을 보여줍니다.



### Training Agents with Weakly Supervised Feedback from Large Language Models (https://arxiv.org/abs/2411.19547)
- **What's New**: 대규모 언어 모델(LLM)을 기반으로 한 새로운 훈련 방법이 소개되었습니다. 이 방법은 전문가 경로(expert trajectories)나 명확한 피드백 없이도 비평가 LLM(critic LLM)으로부터 약한 감독 신호(weakly supervised signals)를 이용하여 에이전트를 훈련합니다. 이를 통해 에이전트는 이후 반복 과정을 통해 향상된 경로를 생성할 수 있습니다. 이 연구는 대규모 언어 모델의 범위와 유연성을 확장하는데 기여하고자 합니다.

- **Technical Details**: 에이전트는 초기 환경 교수법을 통해 경로를 생성하며, 이후 비평가 LLM이 이러한 경로를 평가하여 우수한 경로의 하위 집합을 선정합니다. 에이전트는 K개의 경로를 생성하고 이들 각각에 대해 점수를 부여받습니다. 구성된 경로는 다음 라운드의 에이전트 성능 향상을 위한 교육 데이터로 사용됩니다.

- **Performance Highlights**: API-bank 데이터셋을 통해 실시된 광범위한 테스트에서 에이전트의 성능이 지속적으로 개선되었으며, 공개 벤치마크 데이터셋에서 GPT-4에 필적하는 성능을 보여주었습니다. 이러한 결과는 적은 수의 매개변수를 가진 오픈 소스 모델을 사용했음에도 불구하고 달성되었습니다.



### COLD: Causal reasOning in cLosed Daily activities (https://arxiv.org/abs/2411.19500)
Comments:
          Paper accepted at NeurIPS 2024; Total 37 Pages

- **What's New**: 이번 논문에서는 COLD(Causal reasOning in cLosed Daily activities) 프레임워크를 제안하여 대규모 언어 모델(LLMs)의 인과적 추론 능력을 평가하고자 합니다. 기존의 두 가지 접근법(실제 세계의 관계를 학습하는 방법과 상징적 표현을 사용하는 방법) 간의 격차를 메우고, 일상 활동을 기반으로 한 인과 관계를 추론하는 데 중점을 둡니다. COLD 프레임워크는 인간의 일상적인 이해를 반영하여 인과적 질의를 생성할 수 있는 기회를 제공합니다.

- **Technical Details**: COLD 프레임워크는 인간의 일상 활동에 대한 이해를 기반으로 실생활에서의 인과관계를 탐구합니다. 이 프레임워크는 총 9백만 개의 인과 질의를 생성할 수 있으며, 이는 미니 튜링 테스트에 근접하여 일상적인 과제를 평가하는 데 사용됩니다. 인과적 추론 능력을 평가하기 위해 다양한 LLM을 사용하였으며, 인과 추론이 인간에게는 매우 단순한 작업임에도 불구하고 LLMs에게는 도전적임을 발견했습니다.

- **Performance Highlights**: 실험 결과, LLMs는 트리비얼한 상황에서도 인과적 추론을 수행하는 데 어려움을 겪었습니다. LLMs의 성능은 인과 이론을 잘 이해하지 못하는 '인과적 앵무새(Causal Parrots)'로 묘사되었는데, 이는 LLM들이 주어진 패턴을 단순히 암기하여 과제를 수행한다는 의미입니다. 본 연구는 인과적 추론의 강도를 확인하기 위해 백도어 기준(backdoor criterion)을 활용하여 특정 사건 간의 인과적 강도를 평가했습니다.



### A Simple and Provable Scaling Law for the Test-Time Compute of Large Language Models (https://arxiv.org/abs/2411.19477)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 테스트 시간 계산에 대한 확증된 스케일링 법칙을 갖는 일반적인 두 단계 알고리즘을 제안합니다. 이 알고리즘은 입력 문제를 받아들이고 우선 여러 후보 솔루션을 생성한 다음, 다수결 방식의 여러 라운드에서 최고의 솔루션을 선택합니다. 즉, 이러한 방식으로 LLM의 효율성을 극대화할 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: 제안된 두 단계 알고리즘은 생성 단계와 녹아웃(tournament) 단계로 구성됩니다. 첫 번째 단계에서 N개의 후보 솔루션을 생성하고, 두 번째 단계에서는 각 후보를 K회 비교하여 이긴 후보를 선택하는 방법입니다. 이 과정은 LLM 하나로만 수행할 수 있으며, 외부 검증기나 보상 모델이 필요하지 않습니다.

- **Performance Highlights**: 실험적으로 MMLU-Pro 벤치마크와의 결과를 통해 제안된 알고리즘의 효과성과 이론적 가정이 검증되었습니다. 알고리즘의 실패 확률은 N과 K의 증가에 따라 지수적으로 감소하며, 이는 LLM 호출 수의 증가에 따라 성공 확률이 1로 증가할 수 있음을 보여줍니다.



### Beyond Surface Structure: A Causal Assessment of LLMs' Comprehension Ability (https://arxiv.org/abs/2411.19456)
Comments:
          28 pages, 14 figures, 10 tables

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 표면 구조(surface structure)에 의존하는 것뿐만 아니라 심층 구조(deep structure)도 이해할 수 있는 능력을 평가하는 데 중점을 둡니다. 특히, causal mediation analysis를 통한 새로운 평가 방법을 제안하며, LLM의 성능을 깊이 있는 이해와 표면 인식 모두에서 분석함으로써 더 정확한 평가 기준을 마련하고자 합니다.

- **Technical Details**: 모델의 깊이 있는 이해를 확인하기 위해 저자들은 direct causal effect (DCE)와 indirect causal effect (ICE)를 정의하였고, 이 두 가지를 추정하는 것이 중요하다고 강조합니다. 두 개념의 상호 영향을 측정하는 것이 불가능하기 때문에, approximated DCE (ADCE)와 approximated ICE (AICE)와 같은 대체 지표를 개발하였습니다. ADCE는 LLM의 깊은 구조 이해 능력을 정량화하는 데 사용되며, 다양한 작업에 걸쳐 LLM의 성능을 평가할 수 있습니다.

- **Performance Highlights**: 연구 결과에 따르면, 대부분의 LLM은 정확도(prediction accuracy)가 증가함에 따라 심층 구조 이해 능력이 향상됨을 나타냅니다. 특히, 폐쇄형(closed-source) LLM은 깊은 구조에 더 의존하는 반면, 오픈 소스(open-source) LLM은 표면 구조에 더 민감하여 모델 규모가 커질수록 깊은 구조 이해로 전환되는 경향을 보입니다. 이러한 결과는 ADCE가 단순한 정확도 기준보다 더 포괄적인 LLM 성능 평가 기준이 됨을 입증합니다.



### Auto-RAG: Autonomous Retrieval-Augmented Generation for Large Language Models (https://arxiv.org/abs/2411.19443)
Comments:
          Code is available at this https URL

- **What's New**: 이번 논문에서는 Retrieval-Augmented Generation (RAG) 모델 향상을 위해 Auto-RAG라는 자율적인 반복 검색 모델을 소개합니다. 기존의 반복 검색 작업에서는 수동으로 규칙을 만들거나 few-shot prompting을 사용하여 대화를 진행했으나, Auto-RAG는 LLM의 의사결정을 중심으로 진행되어 이러한 수고를 덜어줍니다. 이 모델은 여러 번의 대화를 통해 리트리버(retriever)와 상호작용하며 유용한 정보를 지속적으로 검색해 나가는 방식을 채택합니다.

- **Technical Details**: Auto-RAG는 LLM이 복잡한 질문에 대한 답변을 위해 필요한 정보를 파악하고, 쿼리를 재구성하며, 리트리버와 지속적으로 새로운 정보를 쿼리하는 방식을 통해 동작합니다. 이 모델은 LLM의 사고 능력을 활용하여 필요한 경우에만 정보를 검색할 수 있는 기능을 가지고 있으며, 새로운 정보를 충분히 수집할 때까지 이 과정을 반복합니다. 최종적으로, 사용자가 만족할 수 있는 정보를 제공하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과는 Auto-RAG가 명확한 훈련 데이터가 없는 경우에도 뛰어난 성능을 발휘함을 보여줍니다. 또한 Auto-RAG는 질문의 복잡성과 검색된 정보의 유용성에 따라 반복 횟수를 자율적으로 조정할 수 있으며, 이러한 과정은 자연어로 표현되어 사용자에게 보다 직관적인 경험을 제공합니다. 이는 Auto-RAG가 기존의 RAG 시스템보다 더 높은 성능과 효율성을 제공함을 의미합니다.



### DENIAHL: In-Context Features Influence LLM Needle-In-A-Haystack Abilities (https://arxiv.org/abs/2411.19360)
- **What's New**: 이 연구는 Needle-in-a-haystack (NIAH) 테스트의 성능을 더욱 정밀하게 평가하기 위해 DENIAHL이라는 새로운 벤치마크를 개발하였습니다. DENIAHL은 데이터 크기, 패턴, 데이터 유형 같은 다양한 특성을 조절하여 LLM의 회상 능력에 미치는 영향을 분석합니다. 연구진은 LLaMA-2 7B와 GPT-3.5 모델을 활용하여 NIAH 성능의 차이를 관찰하였으며, 이를 통해 기존의 NIAH 연구를 확장하고 있습니다.

- **Technical Details**: DENIAHL은 데이터의 크기, 패턴 및 유형 등 3가지 카테고리로 나눈 과제를 포함합니다. 이 벤치마크는 키-값 형식의 데이터에서 길이를 조절하거나 패턴을 변경하여 모델의 기억 능력을 평가하였으며, 특히 데이터의 특성이 모델 성능에 미치는 영향을 분석했습니다. GPT-3.5와 LLaMA-2 7B를 비교하였고, 패턴이 깨지거나 혼합된 데이터 타입이 성능에 미치는 영향을 관찰했습니다.

- **Performance Highlights**: 연구 결과 GPT-3.5 모델은 다양한 NIAH 테스트에서 비교적 높은 성능을 보였으나, 혼합된 데이터의 경우에는 성능이 저하되는 경향을 보였습니다. LLaMA-2 7B는 대부분의 NIAH 테스트에서 보다 낮은 성능을 나타냈고, 숫자 데이터에서는 더 나은 회상 능력을 보여주었습니다. 특히, LLaMA-2 7B는 일관된 패턴보다 깨진 패턴에서 더 나은 성능을 보이며, NIAH 성능이 데이터 의존적임을 강조하고 있습니다.



### Extracting Information in a Low-resource Setting: Case Study on Bioinformatics Workflows (https://arxiv.org/abs/2411.19295)
- **What's New**: 자동화된 데이터 분석 파이프라인 구축을 위한 생물정보학(workflow) 워크플로우의 정보 추출 방법을 제안했습니다. 이 방법은 BioToFlow라는 새로운 주석 달린 코퍼스를 기반으로 하며, 16개의 개체(entity)를 포함해 총 52개의 문서에서 정보 추출을 수행합니다. 이 연구는 특정 개체에 대한 성능을 개선하고, 생물정보학 워크플로우에 관한 고성능 정보 추출이 가능함을 보여줍니다.

- **Technical Details**: 연구에서는 4가지 전략을 통해 워크플로우 정보 추출 문제를 해결하고자 했습니다: 1) 맞춤형 주석 코퍼스 생성, 2) 자가 회귀 언어 모델을 활용한 few-shot named-entity recognition (NER), 3) 기존 및 새로운 코퍼스를 이용한 masked 언어 모델 기반 NER, 4) NER 모델에 워크플로우 지식 통합. BioToFlow를 통해 SciBERT 기반의 NER 모델은 70.4의 F-measure를 기록하며, 이는 주석자 간 동의 채점과 비슷한 결과입니다.

- **Performance Highlights**: BioToFlow 코퍼스를 활용한 실험에서, 자가 회귀 언어 모델인 Llama-3-8B-Instruct를 사용하여 few-shot NER을 수행했습니다. 초반 결과는 등급 모드에서 40% 미만, 엄격 모드에서 31.2%의 성과를 보였습니다. 하지만 Encoder 기반 접근법을 통해 더 나은 정보 추출 성능을 탐색할 수 있는 가능성이 발견되었습니다.



### Consolidating and Developing Benchmarking Datasets for the Nepali Natural Language Understanding Tasks (https://arxiv.org/abs/2411.19244)
- **What's New**: 이번 연구에서는 네팔어의 복잡한 언어적 특성과 여러 방언을 고려하여, 자연어 처리(NLP) 모델 평가를 위한 새로운 벤치마크인 네팔어 언어 이해 평가(Nepali Language Understanding Evaluation, NLUE)를 소개합니다. 기존의 Nep-gLUE 벤치마크는 네 가지 태스크만을 다루었으나, NLUE는 총 12개의 다양한 태스크를 포함하여 더 포괄적인 평가를 가능하게 하였습니다. 새로운 데이터셋은 단일 문장 분류, 유사도 및 패러프레이즈 태스크, 자연어 추론(NLI) 태스크 등을 포함합니다.

- **Technical Details**: NLUE 벤치마크는 고품질의 태스크 별 데이터셋을 위한 자동 및 수동 프로세스의 조합을 통해 만들었습니다. 이를 통해 NLP 모델의 성능을 평가할 수 있도록, 감정 분석(Sentiment Analysis), 언어적 수용 가능성(Corpus of Linguistic Acceptability, CoLA), 패러프레이즈 탐지 등의 태스크 데이터를 추가했습니다. 또한, monolingual 모델과 multilingual 모델 모두에 대해 실험을 수행하여, 다양한 NLU 능력에 대한 종합적인 이해를 제공합니다.

- **Performance Highlights**: 모델 평가 결과, 기존 모델들이 복잡한 NLU 태스크를 효과적으로 처리하는 데 부족하다는 것을 관찰했습니다. NLUE의 추가 태스크는 네팔어 모델에 대한 좀 더 정확한 평가를 가능하게 하여, NLP 연구가 저자원 언어를 위한 발전에 크게 기여할 것입니다. 이 연구는 언어 이해 모델을 비교 및 발전시키기 위한 새로운 기준을 설정했습니다.



### How far can bias go? -- Tracing bias from pretraining data to alignmen (https://arxiv.org/abs/2411.19240)
- **What's New**: 이 연구는 주로 LLMs의 편향(bias)의 기원에 집중하여, 사전 학습(pre-training) 데이터의 성별-직업 편향(gender-occupation bias)과 그 결과 모델 출력(output) 사이의 상관관계를 조사합니다. 특히 Dolma 데이터셋과 OLMo 모델을 사용해, 훈련 데이터의 편향이 어떻게 모델 출력에서 강화되는지를 분석했습니다. 이는 LLM 개발 파이프라인 전반에 걸친 편향 추적에 중점을 두며, 사전 학습 단계에서의 편향 완화의 중요성을 강조하고 있습니다.

- **Technical Details**: 이 연구에서는 zero-shot prompting 및 token co-occurrence 분석(token co-occurrence analyses)을 사용하여 신경망 모델의 훈련 데이터를 세밀하게 분석했습니다. 또한, instruction-tuning을 통해 어떻게 편향 표현(bias expression)을 완화할 수 있는지를 타당화하며, 하이퍼파라미터(hyperparameters)와 프롬프트(prompt)의 변화가 편향 표현에 미치는 영향을 비교했습니다. 연구 결과, 모델 출력에서 훈련 데이터에 존재하는 성별 편향이 증가하는 경향이 나타났습니다.

- **Performance Highlights**: 연구 결과는 다음과 같았습니다: (1) Dolma 훈련 데이터에서 여성의 수가 실제 통계에 비해 부족하다는 점이 확인되었습니다. (2) OLMo 7B 기본 모델의 출력에서 성별 편향이 존재하며, 일부는 증폭되었습니다. (3) instruction-tuning 기법은 표현 편향을 완화시켰지만, 여전히 전반적인 고정관념적인 성별 연관성을 유지하고 있습니다. (4) 하이퍼파라미터와 프롬프트의 변화는 생성된 텍스트의 성별 비율에 미치는 영향이 미미했습니다.



### An Extensive Evaluation of Factual Consistency in Large Language Models for Data-to-Text Generation (https://arxiv.org/abs/2411.19203)
Comments:
          15 pages

- **What's New**: 이 논문은 대형 언어 모델(LLMs)의 사실 일관성을 평가하는 광범위한 연구를 수행함으로써 DTG(데이터-텍스트 생성) 분야에서의 연구 공백을 해결합니다. 다섯 개의 널리 사용되는 DTG 데이터셋과 다섯 가지 주요 LLM 패밀리를 다루며, 사실 일관성을 평가하기 위해 네 가지 최첨단 자동 지표와 필수적인 인적 평가를 사용합니다. 이 연구는 사실 일관성을 개선하기 위한 모델 크기의 영향을 제시하여, LLM 활용의 새로운 방향을 제시합니다.

- **Technical Details**: DTG는 구조화된 데이터(예: 표)나 반구조화된 데이터(예: 그래프)로부터 텍스트를 생성하는 작업을 의미합니다. 이 논문에서는 E2E, ViGGo, WikiTableText, DART, WebNLG의 다섯 가지 데이터셋을 사용하여 LLM의 사실 일관성을 평가합니다. 평가에는 SummaC-Conv, NEOverlap, AlignScore, QAFactEval의 네 가지 자동 지표를 사용하고, LLM의 평균 변화율(AROC)을 계산하여 모델 크기에 따른 사실 일관성의 개선 정도를 분석합니다.

- **Performance Highlights**: Llama 2 패밀리가 사실 일관성 텍스트 생성에서 가장 우수한 성능을 보이는 것으로 나타났습니다. T5 및 BART 같은 소형 모델도 크고 다양성이 적은 데이터셋에서는 강력한 사실 일관성을 달성할 수 있습니다. 모델 크기가 증가함에 따라 사실 일관성이 일반적으로 개선되는 경향을 보였으며, 소스-참조 간의 다이버전스(차이)는 LLM의 DTG 작업에서 사실 일관성을 저해하는 요소로 확인되었습니다.



### Beyond Logit Lens: Contextual Embeddings for Robust Hallucination Detection & Grounding in VLMs (https://arxiv.org/abs/2411.19187)
- **What's New**: 본 연구에서는 LMMs(대규모 다중 모달 모델)의 신뢰성 문제를 해결하기 위해, 기존 기법인 logit lens의 한계를 비판적으로 평가합니다. 우리는 LMM의 중간 레이어에서의 컨텍스트 기본 토큰 임베딩을 활용하여 환각(hallucinations) 감지 및 지향을 향상시키는 새로운 방법론을 제시합니다. 이 접근법은 제로 샷 객체 분할(Zero-Shot Object Segmentation)에서 지리적 시각적 질문 응답(Grounded Visual Question Answering)으로의 전환을 용이하게 합니다.

- **Technical Details**: LMM은 대규모 언어 모델(LLM)의 언어 능력을 활용하고 모달리티에 특화된 인코더를 통합하여 다중 모달 이해(multi-modal understanding)를 진전시킵니다. 본 논문에서는 logit lens 기법을 덜 인지적이고 오히려 간단하게 발생한 환각을 처리하는 데 극복할 수 없는 한계를 지적하고, 중간 레이어의 컨텍스트 기반 토큰 임베딩이 이러한 문제를 해결하는 데 효과적임을 증명합니다.

- **Performance Highlights**: 우리의 새로운 지향 기술은 매우 정밀한 바운딩 박스(bounding boxes)를 생성하여, LMM의 다중 모달 모델의 신뢰성 및 해석 가능성을 향상시키는 가능성을 제시합니다. 즉, 이전에 근소한 성능이었던 카테고리에서도 높은 감지 정확도를 보여주며, 환각 감지 능력을 크게 개선했습니다. 이러한 방식으로 사용자 신뢰를 구축하고 이러한 기술의 널리 도입을 유도할 수 있습니다.



### Integration of Contextual Descriptors in Ontology Alignment for Enrichment of Semantic Correspondenc (https://arxiv.org/abs/2411.19113)
Comments:
          Ontology alignment, contextual descriptors, semantic matching, knowledge representation, essential descriptors, ontology integration, hierarchical structure, semantic heterogeneity, ethical AI

- **What's New**: 이 논문은 컨텍스트 기반의 설명자(contextual descriptors)를 이용한 새로운 의미적 온톨로지 정렬 방식(semantic ontology alignment)을 제안합니다. 필수 및 컨텍스추얼 설명자를 통합하여 포괄적인 지식 모델(knowledge model)을 구축할 수 있도록 하는 형식을 개발하였습니다. 이 과정에서 '투명성(Transparency)'과 '프라이버시(Privacy)'의 관계를 예시로 삼아 개념 간의 갈등(conflict)을 분석하기 위한 수학적 장치(mathematical apparatus)를 소개합니다.

- **Technical Details**: 제안된 방법은 계층적 구조(hierarchical structure)를 통해 의미적 접근 방식을 구현하며, 인공지능(AI) 맥락에서의 개념 분석에 중점을 두고 있습니다. 실험 연구는 컨텍스추얼 설명자를 적용한 후 온톨로지 정렬 지표(ontology alignment metrics)가 유의미하게 개선되었음을 확인했습니다. 특히 프라이버시, 책임(responsibility), 자유(freedom) 및 자율성(autonomy) 분야에서 평균적으로 약 4.36%의 전반적인 개선 효과를 나타냈습니다.

- **Performance Highlights**: 결과는 제안된 접근 방식이 지식의 복잡도(complexity)와 그에 대한 맥락적 의존성(contextual dependence)을 보다 정확하게 반영할 수 있음을 보여줍니다. 이는 인공지능의 다양한 윤리적 측면을 고려할 때 중요한 의미를 지니며, 향후 연구 및 적용 가능성에 대한 기대를 높이고 있습니다.



### Pralekha: An Indic Document Alignment Evaluation Benchmark (https://arxiv.org/abs/2411.19096)
Comments:
          Work in Progress

- **What's New**: 이번 연구에서는 문서 수준에서의 정렬 평가를 위한 대규모 벤치마크 데이터셋인 Pralekha를 소개합니다. Pralekha는 11개 인디크 언어와 영어를 포함하여 2백만 개 이상의 문서로 구성되어 있으며, 정렬된 쌍과 정렬되지 않은 쌍의 비율이 1:2입니다. 이 데이터셋은 문서 수준의 병렬 데이터 마이닝 접근 방식의 효과성을 평가하는 데 도움을 줄 것입니다.

- **Technical Details**: Pralekha는 문서 정렬을 평가하기 위해 임베딩 모델, 입력 표현의 세분성(granularity) 수준, 정렬 알고리즘 등 세 가지 주요 차원에서 다양한 문서 수준 마이닝 접근 방식을 평가합니다. 또한, 문서 정렬을 위한 새로운 점수 제도인 문서 정렬 계수(Document Alignment Coefficient, DAC)를 제안하여, 특히 잡음이 많은 상황에서 베이스라인 풀링 접근 방식보다 20-30%의 향상된 정밀도와 15-20%의 F1 점수를 기록하였습니다. 이 연구는 인디크 언어에 대한 뛰어난 정렬 성능을 보여줍니다.

- **Performance Highlights**: Pralekha 기반의 실험에서는 문서 정렬 전략이 다양한 임베딩 모델, 입력 세분성, 정렬 알고리즘에 따라 평가되었습니다. DAC 점수 제도를 통해 상당한 성과를 달성하여 데이터 정렬에서 신뢰할 수 있는 성능 개선을 입증했습니다. 이 결과는 향후 다국어 기계 번역 모델의 품질과 응집력 향상에 기여할 수 있는 기초를 제공합니다.



### Way to Specialist: Closing Loop Between Specialized LLM and Evolving Domain Knowledge Graph (https://arxiv.org/abs/2411.19064)
Comments:
          Accepted by KDD 2025

- **What's New**: 이 논문에서는 기존의 일반화된 대형 언어 모델(LLMs)의 한계를 극복하기 위해 Way-to-Specialist (WTS) 프레임워크를 제안합니다. WTS는 검색 보강 생성(retrieval-augmented generation)과 지식 그래프(knowledge graphs)를 결합하여 전문화된 지식 없이도 LLM의 전문 능력을 향상시킵니다. 이 프레임워크는 특정 도메인 지식 그래프(Domain Knowledge Graph, DKG)와의 양방향 상호작용을 통해 LLM의 추론 능력을 향상시키는 새로운 접근 방식을 제공합니다.

- **Technical Details**: WTS 프레임워크는 두 가지 밀접하게 연결된 구성 요소로 이루어져 있습니다: DKG-Augmented LLM과 LLM-Assisted DKG Evolution입니다. 첫 번째 구성 요소는 DKG에서 질문 관련 도메인 지식을 검색하고 이를 통해 LLM의 추론 능력을 향상시킵니다. 두 번째 구성 요소는 처리된 작업에서 새로운 도메인 지식을 생성하고 이를 DKG를 발전시키는 데 활용합니다.

- **Performance Highlights**: WTS의 성능은 5개 도메인에 걸쳐 6개의 데이터 세트에서 검증되었습니다. 실험 결과, WTS는 4개의 전문 도메인에서 기존의 최고 성능(SOTA)을 초과했으며, 최대 성능 향상은 11.3%에 달합니다.



### DIESEL -- Dynamic Inference-Guidance via Evasion of Semantic Embeddings in LLMs (https://arxiv.org/abs/2411.19038)
- **What's New**: 최근 대화형 대규모 언어 모델(LLMs)은 개인 맞춤형 대화 및 질문 답변과 같은 과업에서 눈에 띄는 성공을 거두었습니다. 그러나 이러한 모델은 인간의 가치에 부합하지 않는 응답을 생성할 때가 많아 부적절하거나 위험한 결과를 초래할 수 있습니다. 본 논문에서는 이러한 문제를 해결하기 위해 DIESEL이라는 경량화된 추론 안내 기법을 제안하며, 이는 기존의 autoregressive 모델과 통합되어 안전성이 향상된 결과를 제공합니다.

- **Technical Details**: DIESEL은 기존 LLM이 제안한 토큰을 기준으로 미리 정의된 부정적 개념과의 유사성을 바탕으로 재순위를 매김으로써, 바람직하지 않은 결과를 피하도록 안내합니다. 이 과정은 크게 세 가지 단계, 즉 후보 선택, 의미적 잠재 공간 유사성 분석, 토큰 재순위 매기기로 나누어집니다. DIESEL은 간단한 텍스트 설명을 활용하여 유해한 개념을 효과적으로 필터링할 수 있으며, 이를 통해 추가적인 훈련 없이도 활용 가능하다는 장점을 가지고 있습니다.

- **Performance Highlights**: DIESEL은 Llama 3와 같은 최첨단 대화 모델에서 그 효율성을 입증했으며, 특히 jailbreak 공격과 같은 위험한 상황에서도 안전성을 유지하는 능력을 보여주었습니다. 우리의 평가에서는 DIESEL이 기존의 안전성 기법들을 능가하며 실행 시간 또한 크게 개선된 것으로 나타났습니다. 또한, DIESEL은 다양한 사용 사례에 일반화할 수 있는 능력을 지니고 있어, 단순히 안전 관련 필터링을 넘어선 응용 가능성을 포함하고 있습니다.



### A Survey on Automatic Online Hate Speech Detection in Low-Resource Languages (https://arxiv.org/abs/2411.19017)
Comments:
          34 pages, 12 figures

- **What's New**: 소셜 미디어 플랫폼의 영향력이 커지면서 혐오 발언(hate speech)이 전 세계적으로 증가하고 있습니다. 다양한 언어로 이루어진 소통 속에서 저자원(low-resource) 언어의 혐오 발언 탐지가 필요해 졌지만, 관련 데이터셋의 부족으로 인해 연구가 제한적입니다. 본 논문은 저자원 언어에서의 혐오 발언 탐지에 대한 구체적인 조사 결과를 제공하며, 사용되는 데이터셋과 기법을 상세히 설명합니다.

- **Technical Details**: 본 논문은 저자원 언어에서의 온라인 혐오 발언 감지를 위한 다양한 기법들을 분석합니다. 특히, 자연어 처리(NLP), 기계학습(ML) 및 딥러닝(DL) 기술을 활용한 최신 기법들이 소개되며, 혐오 발언의 다양한 카테고리와 관련된 배경 정보도 상세히 설명합니다. 또한, 이 연구는 혐오 발언 감지에서 발생하는 다양한 연구 과제와 기회에 대해서도 논의합니다.

- **Performance Highlights**: 혐오 발언 감지에 있어 여러 언어의 데이터셋 분석과 자동화 방법들이 강조됩니다. 특히 저자원 언어의 경우, 데이터셋의 접근성이 낮아 연구가 미비하지만, 본 조사에서는 이러한 언어들에 대한 탐색을 통해 연구자들에게 유용한 통찰을 제공합니다. 각 대륙별, 언어별로 구조화된 접근을 통해 저자원 언어에 대한 혐오 발언 탐지와 관련된 통찰도 제공합니다.



### Talking to oneself in CMC: a study of self replies in Wikipedia talk pages (https://arxiv.org/abs/2411.19007)
- **What's New**: 이 연구는 Wikipedia 토론 페이지에서 자기 답글에 대한 질적 분석을 제안합니다. 이 특이한 패턴은 두 개 이상의 메시지가 있는 스레드의 10% 이상에서 발생하며, 몇 가지 이유로 설명될 수 있습니다.

- **Technical Details**: 연구진은 두 번째 메시지의 어휘적 특성을 최초로 검토한 후, 일곱 가지 분류(typology) 카테고리를 제안하고, 각각 100개의 스레드를 영어와 프랑스어로 주석(annotation) 처리하는 데 사용했습니다. 이 연구에서 사용된 LLM(대형 언어 모델)은 특정 카테고리에서 중요한 어려움을 겪습니다.

- **Performance Highlights**: 인간 주석자들은 합리적인 전반적인 효율성에 도달했으나, 지시 조정된 LLM의 성능은 여러 카테고리에서 중요한 문제를 보였습니다. 이 연구는 인간과 인공지능 간의 차이를 비교하고 규명하는 데 초점을 맞추고 있습니다.



### USTCCTSU at SemEval-2024 Task 1: Reducing Anisotropy for Cross-lingual Semantic Textual Relatedness Task (https://arxiv.org/abs/2411.18990)
Comments:
          8 pages, 3 figures

- **What's New**: 이 논문은 다국어 간 의미 텍스트 관계성 과업(Cross-lingual semantic textual relatedness task)을 다루며, 사전 훈련된 모델인 XLM-R-base를 기반으로 하고 있습니다. 데이터 필터링 방법을 통해 다국어 처리의 문제를 경감시키고, 표준 정규성(whitening) 기술을 활용해 문장의 의미 유사성을 향상시켰습니다. 이 방법을 통해 스페인어에서 2위, 인도네시아어에서 3위를 기록하는 성과를 올렸습니다.

- **Technical Details**: 의미 텍스트 관계성(STR)은 두 문장 간의 다양한 공통점을 고려하는 개념으로, 주제 공유, 관점 표현 등 여러 요소를 포함합니다. 세미밸(SemEval) 2024의 트랙 C에서는 특정 목표 언어의 레이블 데이터 없이 다른 언어의 레이블 데이터만을 사용하여 시스템을 개발해야 합니다. 이를 해결하기 위해, 본 논문은 사전 훈련된 언어 모델을 기반으로 한 식별 방법과 함께, 문장 벡터로의 변환 과정에서 비동적 속성(anisotropic property)을 해소하기 위한 정규화 기법을 도입했습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 스페인어에서 2위, 인도네시아어에서 3위를 기록하며 여러 결과물들이 상위 10위 안에 진입하였습니다. 또한, 이 연구는 다국어 간 의미 관계성 과업에 대한 미래 조사 및 연구를 촉발할 수 있는 분석을 포함합니다. 이러한 성과는 다국어 모델이 직면한 문제를 해결하기 위한 기초로 활용될 수 있습니다.



### Zero-shot Slot Filling in the Age of LLMs for Dialogue Systems (https://arxiv.org/abs/2411.18980)
Comments:
          To appear in Proceedings of COLING 2025

- **What's New**: 이번 논문은 제로샷(Zero-shot) 슬롯 필링을 향상시키기 위한 새로운 접근법을 제안하고 있습니다. 주로 단일 턴의 텍스트 데이터에 초점을 맞춘 기존 방법들과 달리, 대화형(dialogue) 데이터의 복잡성을 다루기 위한 전략을 개발했습니다. 제안된 방법은 대화 데이터에서 슬롯 유도(slot induction) 및 블랙박스 지식 증류(knowledge distillation)를 통해 모델의 성능을 크게 향상시키며, 기존 모델보다 26% 높은 F1 점수를 기록합니다.

- **Technical Details**: 우리의 접근법은 대형 LLM(대형 언어 모델)으로부터 상대적으로 작은 모델로 지식을 전이하는 블랙박스 KD(지식 증류)에 기반한 데이터 주석 전략을 사용합니다. 슬롯 유도는 사전 정의된 슬롯 이름 대신 모든 가능한 슬롯 레이블-값 쌍을 예측하도록 모델을 지침하는 방식을 사용합니다. 이렇게하여 모델은 다양한 도메인에서 잘 일반화할 수 있는 능력을 향상시키며, 입력 텍스트와 문맥을 함께 사용하여 문맥 인지(Context-awareness)를 높입니다.

- **Performance Highlights**: 제안된 모델은 기존의 추출 기반(extractive) 모델을 34% 상대적으로 초과하는 F1 점수를 기록하며, 고객 센터와 같은 제품 설정에서 효율적으로 작동합니다. 모델은 실시간 대화 스트림에서 고도의 정확성을 유지하면서 낮은 지연(latency)을 유지합니다. 이러한 성능 향상은 기본적으로 데이터 수집 단계에서부터 시작되며, 주어진 다각적인 데이터 세트를 활용하여 모델이 다양한 도메인에서도 견고하게 일반화할 수 있도록 합니다.



### Rephrasing Electronic Health Records for Pretraining Clinical Language Models (https://arxiv.org/abs/2411.18940)
- **What's New**: 본 연구에서는 기존의 임상 노트를 LLM(대형 언어 모델)을 활용하여 재구성함으로써 합성 훈련 말뭉치를 생성하는 새로운 접근법을 제안합니다. 기존의 합성 방식과는 달리, 실제 임상 정보를 포함한 텍스트에 의존하지 않으므로, 환자 개인 정보 보호 문제를 더욱 효과적으로 해결할 수 있습니다. 이를 통해 훈련 데이터의 결핍 문제를 완화하고, 다양한 LLM을 이용하여 임상 언어 모델의 성능을 개선할 수 있는 잠재력을 보여주고 있습니다.

- **Technical Details**: 연구에서는 4개 종류의 소형 LLM(<10B)을 이용하여 임상 텍스트를 생성하고, 이 데이터를 기반으로 디코더 및 인코더 기반의 언어 모델을 훈련합니다. 또한, 세 가지 유형의 프롬프트를 사용하여 LLM의 성격과 특성에 맞게 의료 문서를 재구성하는 과정을 실험합니다. 주요 사용 데이터는 MIMIC-III EHR에서 추출한 퇴원 요약이며, 이 데이터의 성질에 따라 구체적인 프롬프트를 적용하여 20M 토큰의 합성 훈련 말뭉치를 생성합니다.

- **Performance Highlights**: 실험 결과, 재구성 방식이 이전의 합성 접근 방식보다 언어 모델링에서 더 낮은 혼란도(perplexity) 값을 기록하는 등 뛰어난 성능을 보였습니다. 이 방식을 통해 생성된 합성 임상 노트와 실제 임상 노트를 결합함으로써 언어 모델링 성능이 향상되었습니다. 최종적으로, 개발된 모델이 기존의 ClinicalBERT를 초과하는 성능을 입증하며, 재구성 접근 방식의 유효성을 강조하고 있습니다.



### ScratchEval: Are GPT-4o Smarter than My Child? Evaluating Large Multimodal Models with Visual Programming Challenges (https://arxiv.org/abs/2411.18932)
- **What's New**: 이 연구에서는 ScratchEval이라는 새로운 벤치마크를 제안하여 대규모 다중모달 모델(large multimodal models, LMMs)의 시각적 프로그래밍 추론 능력을 평가하고자 합니다. 기존의 평가 방법들은 특정 시나리오에 국한되어 있었지만, ScratchEval은 Scratch라는 블록 기반 시각적 프로그래밍 언어를 기반으로 하여 모델이 시각 정보를 처리할 수 있도록 설계되었습니다. 이를 통해 LMMs의 명령 이해 능력을 포괄적으로 평가할 수 있는 기회를 제공합니다.

- **Technical Details**: ScratchEval은 Scratch 스크립트가 포함된 305개의 다지선다 질문으로 구성되어 있으며, 각 질문은 문제 설명과 선택지를 포함합니다. 이 벤치마크는 일관된 논리적 사고 및 문제 해결 능력에 초점을 맞추며, 모델이 이미지, 그래픽 프로그래밍 언어 및 내재된 논리를 함께 이해해야 합니다. 질문은 수학, 논리적 사고, 그래픽 인식 및 공간 인식으로 분류되며, 다양한 인지 영역에서 모델의 능력을 평가합니다.

- **Performance Highlights**: 총 10개의 LMM을 평가한 결과, Gemini-1.5-Pro가 모든 카테고리에서 가장 높은 점수를 기록했지만, 대부분의 모델은 50% 정확도를 넘기기 어려웠습니다. 이는 LMM들이 비주얼 코드 추론 능력에서 한계를 지니고 있음을 시사합니다. 일반적으로, 수학 및 논리적 추론 작업에서 모델의 성능이 낮았고, 반면 그래픽 및 공간 인식 작업에선 상대적으로 더 나은 성과를 보였습니다. 이 연구는 적절한 프롬프트 기법이 LMM의 성능 향상에 기여할 수 있음을 보여주지만, 다중모달 LLM에 대한 연구는 더 필요합니다.



### The Impact of Example Selection in Few-Shot Prompting on Automated Essay Scoring Using GPT Models (https://arxiv.org/abs/2411.18924)
Comments:
          Accepted in AIED2024. This preprint has not undergone any post-submission improvements or corrections. The Version of Record of this contribution is published in Communications in Com-puter and Information Science, vol 2150, and is available online at this https URL

- **What's New**: 이번 연구는 GPT 모델을 활용한 자동 에세이 채점(Automated Essay Scoring, AES)에서 적은 수의 예시로 유도(prompting)할 때의 예시 선택이 성능에 미치는 영향을 조사하였습니다. 다양한 GPT-3.5 및 GPT-4 모델을 사용하여 예시의 선택 및 순서가 성능에 미치는 영향을 평가했습니다.

- **Technical Details**: 연구에서는 119개의 다양한 prompt를 사용하여 GPT 모델과 인간 평가자 점수 간의 합의 정도를 측정하기 위해 Quadratic Weighted Kappa (QWK)를 계산했습니다. 또한, 회귀 분석을 통해 예시 선택에 의해 도입된 편향을 정량적으로 평가하였습니다. 주요하게 발견된 것은 GPT-3.5가 GPT-4보다 예시에 더 민감하다는 것과 함께, 다수 라벨 편향(majority label bias) 및 최신 예시 편향(recency bias)이 존재한다는 점입니다.

- **Performance Highlights**: 연구 결과, careful example selection이 GPT-3.5 모델의 성능을 향상시켜 일부 GPT-4 모델보다 더 나은 결과를 보여주는 경향이 있음을 확인했습니다. 또한, 2023년 6월 버전의 GPT-4 모델이 너무 최신 버전은 아니지만, 가장 높은 안정성과 성능을 보였습니다. 이러한 결과는 AES를 위한 few-shot prompting에서 예시 선택의 중요성을 강조하며, 각 모델의 개별 성능 평가 필요성을 언급합니다.



### EzSQL: An SQL intermediate representation for improving SQL-to-text Generation (https://arxiv.org/abs/2411.18923)
Comments:
          Under Review at Expert System With Applications Journal

- **What's New**: 이번 연구에서는 EzSQL이라는 SQL 중간 표현(intermediate representation)을 제안하여 SQL과 자연어(natural language) 텍스트 시퀀스를 정렬하려고 합니다. EzSQL은 SQL 쿼리를 간소화하고 자연어와 더 가까운 형태로 만들어 주며, 이는 SQL-to-text 생성 모델에 입력으로 사용됩니다. 또한 Sql-to-text 생성모델을 사용하여 Text-to-SQL 파서의 성능을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: EzSQL은 SQL 쿼리의 구조를 단순화하여 자연어에 가깝도록 만드는 것을 목표로 합니다. EzSQL은 UNION과 INTERSECT와 같은 집합 연산(set operator)을 제거하고, JOIN과 중첩 서브쿼리(nested subquery) 또한 배제합니다. 이 과정에서 작업의 표현이 간단해져, pretrained language model(BART)의 입력으로 적합한 형태가 되며, 이는 자연어 텍스트 생성을 지원합니다.

- **Performance Highlights**: EzSQL을 사용하는 SQL-to-text 생성 모델은 WikiSQL과 Spider 데이터세트에서 최신 기술 수준(state-of-the-art) 성능을 달성하였습니다. 이 모델을 사용하여 생성된 데이터는 Text-to-SQL 파서의 성능을 향상시키는 데 기여할 수 있는 잠재력을 보여줍니다. 연구 결과는 EzSQL을 통한 접근 방식이 SQL-to-text 생성에서 효과적임을 입증하고 있습니다.



### Devising a Set of Compact and Explainable Spoken Language Feature for Screening Alzheimer's Diseas (https://arxiv.org/abs/2411.18922)
Comments:
          Published at ISCSLP 2024

- **What's New**: 이 연구는 알츠하이머병(Alzheimer's Disease, AD)의 조기 탐지를 위한 말을 기반으로 한 새롭고 효과적인 기능 집합을 제안합니다. 특히 Cookie Theft 그림 설명 과제를 활용하여, 대형 언어 모델(LLM)의 시각적 처리 능력을 결합한 기존의 언어적 특징을 초월한 설명 가능한 특징을 도출했습니다. 제안된 특징은 자동 AD 스크리닝의 해석성을 증대시켜 주목받고 있습니다.

- **Technical Details**: 이 연구에서 사용된 데이터 세트는 ADReSS Challenge 2020에서 파생된 Pitt Corpus의 일부로, 156개의 음성 샘플과 그에 해당하는 전사가 포함되어 있습니다. 참여자는 알츠하이머병이 없는 그룹과 있는 그룹으로 나뉘며, 새로운 11개의 기능을 제안하여 Cookie Theft 그림에 대한 설명을 분석했습니다. 이 기능들은 TF-IDF 및 LLM의 시각적 분석 능력을 활용하여 세부 사항을 정량화합니다.

- **Performance Highlights**: 실험 결과, 제안된 새 기능 세트는 기존의 40개의 언어적 기능을 초월하여 안정적인 성능을 보였습니다. 15개 차원만으로 ADReSS 테스트 세트에서 85.4%의 경쟁력 있는 정확도를 달성하였으며, 이는 기존 기능 집합에 비해 차원 효율성을 강조합니다. 따라서 이 연구는 AD 탐지에서 효과적이고 설명 가능한 접근 방식을 제시합니다.



### Sneaking Syntax into Transformer Language Models with Tree Regularization (https://arxiv.org/abs/2411.18885)
Comments:
          17 pages, 16 figures, 8 tables

- **What's New**: 이 논문은 Transformer 언어 모델에 syntactic inductive biases를 부드럽게 주입하는 새로운 방법인 TREEREG를 제안합니다. TREEREG는 silver parses에서 파생된 bracketing 결정들을 활용하여 벡터 은닉 상태에 대한 차별 가능한 직교성 제약을 설정하는 보조 손실 함수입니다. 이 방법은 모델의 아키텍처를 변경할 필요 없이 표준 LM 목표와 통합될 수 있습니다.

- **Technical Details**: TREEREG는 입력 문장의 각 구성 요소에 대한 은닉 상태 표현이 주변 문맥의 표현과 최대한 직교하게 유지되도록 하는 부드러운 구조적 제약을 제공합니다. 이 손실 함수는 훈련 동안 LM 손실에 정규화기로 추가됩니다. 또한 TREEREG는 사전 훈련된 LMs가 더 낮은 perplexity를 얻고, 적은 훈련 데이터로도 높은 syntactic generalization 성능을 달성하는 데 기여합니다.

- **Performance Highlights**: TREEREG를 통해 훈련된 LMs는 MultiNLI와 같은 대회적 NLI 벤치마크에서 성능 감소를 41.2점 완화시키며, out-of-distribution 데이터에서도 최대 9.2%의 perplexity 감소를 기록합니다. 또한, syntactic generalization 관련 테스트에서 최대 9.5점 개선된 성과를 보여 줍니다.



### Measuring Risk of Bias in Biomedical Reports: The RoBBR Benchmark (https://arxiv.org/abs/2411.18831)
- **What's New**: 이 논문은 생물의학(Biomedical) 연구의 방법론적 강도를 평가하기 위한 RoBBR 벤치마크를 제시합니다. 이 벤치마크는 체계적 검토(Systematic Review)에서 사용되는 위험-편향(risk-of-bias) 프레임워크를 기반으로 하여 연구의 질을 보다 신뢰성 있게 평가할 수 있도록 설계되었습니다. 이를 통해 연구 수집의 신뢰성과 권장 사항의 질을 향상시키는 데 기여하고자 합니다.

- **Technical Details**: RoBBR 벤치마크는 500개 이상의 연구에서 파생된 네 가지 주요 작업(Task)을 포함하고 있습니다. 각 작업은 연구 방법론 분석 및 위험-편향 평가는 물론, 총 2000개의 전문가 생성 편향 주석(bias annotations)을 담고 있습니다. 이러한 작업은 연구 논문(contents)과 세밀하게 정렬된 인간 검증 파이프라인을 포함하고 있어, 생물의학 논문의 편향 판단을 검증할 수 있는 도구로서 활용됩니다.

- **Performance Highlights**: 대규모 언어 모델(Large Language Models)을 RoBBR 벤치마크로 평가한 결과, 이들 모델이 전문가 수준의 성능에 상당히 미치지 못함을 발견했습니다. 따라서 RoBBR 벤치마크는 과학 데이터를 대규모로 집계하는 시스템에 있어 연구 품질을 측정하는 표준화된 도구로 활용될 수 있을 것입니다. 이 데이터셋은 연구자들이 접근할 수 있도록 제공됩니다.



### NewsEdits 2.0: Learning the Intentions Behind Updating News (https://arxiv.org/abs/2411.18811)
Comments:
          9 pages main body, 11 pages appendix

- **What's New**: 이 연구에서는 뉴스 기사의 업데이트 내용을 예측하기 위해 언어적 특징을 활용할 수 있다는 가설을 세웠습니다. 이를 위해 NewsEdits 2.0이라는 수정 의도 분류체계를 도입하였고, 9,200개의 문장 쌍을 주석 처리하여 고점수 앙상블 모델을 학습시켰습니다. 이 과정을 통해 기존 기사의 사실이 언제 업데이트될지를 예측할 수 있는 알고리즘을 개발했습니다.

- **Technical Details**: 본 연구는 다양한 이유로 작성된 뉴스 기사를 통해 사실의 업데이트 패턴을 분석하였습니다. NewsEdits 2.0 분류체계를 통해, 언어적 단서(linguistic cues)와 미래 시제 동사(future-tense verbs), 통계(statistics), 자주 업데이트되는 사건들(common updating events) 등의 언어적 패턴을 발견하였습니다. 모델 성능은 테스트 세트에서 Macro-F1 점수 0.58을 기록했으며, 특히 업데이트 가능성이 높은 문장에 대해서는 정확도가 0.74에 달했습니다.

- **Performance Highlights**: 마지막으로, LLM이 구식 데이터에서 문서를 검색할 경우 발생할 수 있는 오류를 줄이기 위한 사례 연구를 진행했습니다. 우리의 예측을 활용한 모델은 정확도가 거의 오라클 수준에 도달하는 성과를 보였습니다. 이 연구 결과는 LLM의 질문 응답 시스템에서 구식 정보를 다루는 데 있어 중요한 기여를 할 것으로 기대됩니다.



### On the Effectiveness of Incremental Training of Large Language Models (https://arxiv.org/abs/2411.18700)
- **What's New**: 이 논문은 대규모 언어 모델(LLM) 훈련에 있어 점진적 레이어 훈련(incremental layer-wise training) 방법의 효율성을 검토합니다. 기존의 훈련 방식이 아닌 점진적인 방법으로 레이어를 추가함으로써 훈련 과정을 최적화할 수 있을 것이라는 기대가 있었습니다. 그러나 연구 결과, 점진적 방법은 결국 더 큰 계산 비용을 요구하며 전통적인 훈련 방식과의 성능 차이를 메우기 위해 더 오랜 훈련이 필요하다는 것을 보여줍니다.

- **Technical Details**: 점진적 레이어 훈련은 네트워크의 하위 레이어가 먼저 훈련되어 안정화된 후, 상위 레이어를 훈련하는 방식입니다. 이러한 접근법은 높은 수준의 레이어가 낮은 수준의 레이어에서 학습된 표현을 의존하므로, 기초적인 언어적 특징을 잘 학습할 수 있도록 도와줍니다. 하지만, 이 방법의 효용은 이전 연구에서 경험적으로 검증되지 않았고, 이 연구는 이를 해결하기 위한 임상 실험을 진행했습니다.

- **Performance Highlights**: 결과적으로, 점진적 레이어 훈련은 초기의 계산 효율성을 보였지만, 전통적인 훈련과 비교했을 때 성능에서 유의미한 이점을 나타내지 않았습니다. 특히, 높은 수준의 레이어가 학습할 기초적인 특징이 안정화되기까지 시간이 더 소요되며, 이는 전체적인 계산 비용을 증가시킵니다. 따라서, 이 연구는 대규모 언어 모델 훈련에 있어 점진적 레이어 훈련의 유효성이 제한적임을 강조합니다.



### Semantic, Orthographic, and Morphological Biases in Humans' Wordle Gameplay (https://arxiv.org/abs/2411.18634)
- **What's New**: 이 논문에서는 Wordle 게임에서 인간 플레이어의 추측이 이전 추측의 의미론(semantics), 철자(orthography), 형태학(morphology)에 의해 영향을 받는다는 점을 보여줍니다. 실제 플레이어의 추측과 근접 최적(near-optimal) 추측을 비교함으로써, 인간 플레이어의 추측이 이전 추측과 의미적, 철자적, 형태적으로 유사한 경향이 있음을 입증합니다.

- **Technical Details**: Wordle은 플레이어가 숨겨진 다섯 글자 단어를 6회의 시도 내에 찾아야 하는 단어 추측 게임입니다. 이 연구에서는 최대 엔트로피 휴리스틱(maximum-entropy heuristic)을 사용하여 근접 최적 플레이를 추정하고, 인간의 플레이에서 인지적 부하를 줄이기 위해 이전 추측과 덜 차별화된 추측을 할 것이라는 가설을 제시합니다. 또한, 심리학의 프라이밍(priming) 개념을 통해 단어의 연관성이 중요한 상황에서 이전 정보가 영향을 미친다는 점을 탐구합니다.

- **Performance Highlights**: 연구 결과, Doddle이라는 오픈소스 Wordle 솔버는 평균 3.482회의 추측으로 게임을 완료할 수 있으며, 이는 최적의 평균 추측 수치인 3.421에 근접합니다. Doddle은 미니맥스(minimax)와 엔트로피(Shannon entropy) 기반의 두 가지 휴리스틱 방식을 통해 각 단계에서 불확실성을 줄이며 최적의 해결책을 찾습니다. 최적의 시작 추측은 'SALET'이며, 이는 평균적으로 가장 적은 추측으로 게임을 완료할 수 있도록 설계되었습니다.



### T2Vid: Translating Long Text into Multi-Image is the Catalyst for Video-LLMs (https://arxiv.org/abs/2411.19951)
Comments:
          13 pages, 9 figures, 5 tables. Project page: this https URL

- **What's New**: 이 논문은 멀티모달 대용량 언어 모델(MLLMs)의 이미지 처리 성공을 바탕으로 비디오 이해로의 확장을 연구합니다. 특히, 제로샷 추론(zero-shot inference)과 추가 파인튜닝(fine-tuning) 접근 방식을 조사하여, 효과적인 데이터 증강(data augmentation) 방법을 제안합니다. 또한, 비디오 데이터 샘플을 활용한 학습 효율성을 높이기 위해 T2Vid라는 새로운 방법론을 개발했습니다.

- **Technical Details**: 연구에서는 비디오 이해를 위해 이미지-LLMs를 활용하는 두 가지 접근 방식을 심층적으로 분석합니다. 특히 제로샷 추론의 한계로 일반화 제한과 시간적 이해 부족을 지적하며, 이러한 문제를 해결하기 위해 파인튜닝 접근 방식을 조사하였습니다. 그 결과, 훈련 데이터의 다양성 부족이 낮은 학습 효율성을 초래함을 발견하고, 이를 해결하기 위한 비디오 유사 샘플 합성 기법을 제안합니다.

- **Performance Highlights**: 제안된 T2Vid 방법은 고작 15%의 데이터 샘플로도 완전한 비디오 데이터셋을 사용하는 것과 비슷하거나 더 나은 성능을 달성하는 것으로 나타났습니다. 이 방법은 긴 비디오 이해 및 비디오 샘플 훈련 없이도 성능을 향상시킬 수 있습니다. 연구 결과는 MLLMs를 비디오 이해에 활용하는 데 대한 새로운 가능성을 제시하고, 고품질 데이터의 수집에 대한 논의에 불을 지필 것으로 기대하고 있습니다.



### Perception Test 2024: Challenge Summary and a Novel Hour-Long VideoQA Benchmark (https://arxiv.org/abs/2411.19941)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2312.13090

- **What's New**: 2024년 IEEE/CVF 유럽 컴퓨터 비전 컨퍼런스에서 두 번째 Perception Test 챌린지가 개최되었습니다. 이번에는 7개의 트랙이 새롭게 추가되어, 비디오, 오디오 및 텍스트 모달리티에 걸쳐 다양한 다중 모달 작업을 다루었습니다. 특히, 올해는 1시간 길이의 비디오 이해를 위한 새로운 비디오 질문 응답 벤치마크인 1h-walk VQA가 도입되었습니다.

- **Technical Details**: Perception Test 벤치마크는 기억 및 직관 물리학 이해 능력 등을 측정하기 위해 특별히 설계된 11,600개의 실세계 비디오로 구성되어 있습니다. 이 벤치마크는 모델이 긴 시간 맥락을 이해하고 추론하는 능력을 평가하기 위해 Walking Tours 데이터셋을 활용하여 1h-walk VQA를 생성했습니다. 각 트랙은 검증 및 테스트 단계로 나뉘며, 참여자들은 평가 모드를 포괄적으로 제출해야 했습니다.

- **Performance Highlights**: 이번 챌린지에서는 123개 팀으로부터 680건의 제출물이 접수되었습니다. 참여 팀은 올해 도입된 총 상금 20,000 유로를 놓고 경쟁하였으며, 모든 트랙에서 상위 모델들의 성능이 지난해보다 개선되었습니다. 자세한 결과는 워크숍 웹사이트에서 확인할 수 있습니다.



### VLSBench: Unveiling Visual Leakage in Multimodal Safety (https://arxiv.org/abs/2411.19939)
- **What's New**: 이번 연구에서는 Multimodal large language models (MLLMs)의 안전성과 관련하여 새로운 문제를 제기하고 있습니다. 특히, 기존의 데이터셋에서 시각적 정보 누출(Visual Safety Information Leakage, VSIL) 문제가 존재한다는 점을 발견했습니다. 이로 인해 MLLMs가 이미지와 텍스트 쿼리를 기반으로 민감한 내용을 쉽게 감지할 수 있으나, 실제 상황에서는 VSIL이 없는 이미지-텍스트 쌍도 많다는 것을 지적합니다. 이 연구는 이러한 문제를 해결하기 위한 새로운 벤치마크인 VLSBench를 구축했습니다.

- **Technical Details**: VLSBench는 2.4k 개의 이미지-텍스트 쌍을 포함한 멀티모달 안전 기준으로, 시각적 안전 정보의 누출을 방지합니다. 연구자들은 LLM을 활용하여 해로운 텍스트 쿼리와 이미지를 생성하고, 이후 이를 정제하여 안전한 텍스트 쿼리와 질 높은 이미지를 조합했습니다. 실험 결과, VLSBench는 기존의 MLLM 뿐만 아니라 공개 및 폐쇄 소스 모델에게도 상당한 도전 과제가 됩니다. 특히, 기존 안전 벤치마크가 VSIL 문제에 취약하다는 것을 경험적으로 확인했습니다.

- **Performance Highlights**: VLSBench에서의 실험 결과에 따르면, 단순한 텍스트 SFT 방법이 MLLM의 안전성을 유지하면서도 95% 이상의 성능을 달성할 수 있음을 보여줍니다. 반면, 멀티모달 정렬 방법은 VSIL이 없는 경우에 더 유망한 해결책이 될 수 있습니다. 이러한 발견들을 기반으로, MLLMs의 안전성 접근 방식에서 텍스트 및 멀티모달 정렬 방법의 효용성을 비교했습니다. 결과적으로, 시각적 안전 정보 누출이 텍스트 방식의 우수한 성능의 원인임을 입증했습니다.



### SIMS: Simulating Human-Scene Interactions with Real World Script Planning (https://arxiv.org/abs/2411.19921)
- **What's New**: 이 논문은 물리적으로 그럴듯한 장기 인간-장면 상호작용을 계획하고 제어하기 위한 새로운 프레임워크를 제시합니다. 특히, 인터넷에 존재하는 비디오 데이터를 활용하여 LLM 기반의 스크립트 추출 및 생성 과정을 통합합니다. 이를 통해 인과적이고 복잡한 시간 연속적인 인간 행동을 모사하며, 다양한 장면에 대한 이해를 통해 캐릭터의 동작을 유도할 수 있습니다.

- **Technical Details**: 프레임워크 SIMS(SIMultating huMan Scene interactions)는 고수준 계획에 LLM을, 저수준 제어에는 물리적 정책을 사용합니다. LLM을 통한 상호작용과 감정 변화를 실시간 비디오에서 추출하여 스크립트 데이터베이스를 구성하고, 이를 통해 키프레임을 생성합니다. 또한, CLIP 모델을 통해 장면 기하학과 텍스트 임베딩을 인식하여 고품질 동작 생성을 위한 듀얼 인식 정책을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 다양한 작업 수행에 있어 뛰어난 성능을 보이며 기존 방법들에 비해 일반화 능력이 향상되었습니다. 또한, 지능형 캐릭터들이 다양한 3D 환경에서 요구되는 다양한 동작을 실행하면서도 정교한 스타일을 갖출 수 있도록 합니다. 이 프레임워크의 코드는 곧 공개될 예정이며, 실제 응용 가능성이 높습니다.



### Classical and Quantum Algorithms for the Deterministic L-system Inductive Inference Problem (https://arxiv.org/abs/2411.19906)
Comments:
          16 pages, 1 figure

- **What's New**: 이 논문은 주어진 데이터, 특히 문자열 시퀀스에서 결정론적 문맥 자유 L-시스템(D0L-system)을 자동으로 추론하는 문제에 초점을 맞추고 있습니다. 이를 위해 우리는 문자열 시퀀스의 특성 그래프(characteristic graph)를 도입하고, 이 문제를 다루기 위한 고전적 정확한 알고리즘과 근사 양자 알고리즘을 제안합니다. 이러한 방법론은 L-시스템을 생성하는 과정을 자동화할 수 있는 가능성을 열어줍니다.

- **Technical Details**: 논문에서는 D0L 시스템을 최대 독립 집합(maximum independent set, MIS) 문제와 연결하는 정리를 제안합니다. 이를 통해 D0LII를 그래프 문제로 변환하고, 입력 데이터를 다루는 폴리노미얼 타임(Polynomial Time) 인코딩을 제안합니다. 연구자들은 양자 근사 최적화 알고리즘(Quantum Approximate Optimization Algorithm, QAOA)을 활용하여 이 문제를 해결하는 양자 알고리즘과 정확한 해법을 갖는 고전적 알고리즘을 구축하고 있습니다.

- **Performance Highlights**: 이 논문은 L-시스템 추론 문제에 대한 정량적 접근을 통해 기존 알고리즘보다 빠르고 효율적인 해결 방안을 제공합니다. 양자 컴퓨팅의 활용은 비선형 문제를 해결하는데 있어 새로운 가능성을 제공하며, 이는 머신러닝과 NP-완전 문제 접근에 유망한 결과를 보여줍니다. 논문에서 제안된 알고리즘은 L-시스템을 통해 생물학적 프로세스를 모델링하는 데 기여할 것으로 기대됩니다.



### A Cross-Corpus Speech Emotion Recognition Method Based on Supervised Contrastive Learning (https://arxiv.org/abs/2411.19803)
- **What's New**: 본 연구는 음성 감정 인식(Speech Emotion Recognition, SER)에서 대규모 공개 데이터셋 부족과 다양한 분포의 데이터 처리 시의 일반화 능력 한계를 극복하기 위한 방법을 제안합니다. 특히, 감독 대조 학습(supervised contrast learning)을 기반으로 한 새로운 크로스-코퍼스(cross-corpus) 음성 감정 인식 방법을 개발하였습니다.

- **Technical Details**: 제안된 방법은 두 단계의 세밀 조정(fine-tuning) 프로세스를 포함합니다. 첫 번째 단계에서는 자가 감독 음성 표현 모델이 여러 음성 감정 데이터셋을 사용하여 감독 대조 학습으로 세밀 조정되고, 두 번째 단계에서는 분류기(classifier)가 목표 데이터셋에 맞춰 세밀 조정됩니다.

- **Performance Highlights**: 실험 결과, WavLM 기반 모델이 IEMOCAP 데이터셋에서 77.41%의 비가중 정확도(unweighted accuracy, UA)를 기록하였고, CASIA 데이터셋에서는 96.49%에 도달하여 두 데이터셋에서 기존의 최첨단 결과를 초월하는 성능을 보였습니다.



### Voice Communication Analysis in Esports (https://arxiv.org/abs/2411.19793)
Comments:
          17 pages, 11 figures. Independent research

- **What's New**: 본 논문은 팀 기반 e스포츠에서 음성 통신의 효과성을 개선하기 위한 연구를 다루고 있으며, 특히 LoL(League of Legends) 게임을 중심으로 진행되었습니다. 이에 따라 LLM(Large Language Models)와 NLP(Natural Language Processing) 기술을 활용하여 효과적인 음성 커뮤니케이션의 지표를 개발하고 있습니다. 또한, 커뮤니케이션의 중복성과 비약한 소통 문제를 찾아 해결책을 제안하고자 합니다.

- **Technical Details**: 연구에서는 음성 파일을 분석하기 위해 Bain et al.의 방법을 바탕으로 Whisper로 음성을 텍스트로 전사하고, PixIT 모델을 통해 화자 단별 분석(speaker diarization)을 진행합니다. 이 과정에서 음성 간의 단어 정렬(forced-text alignment)을 수행하여 각 플레이어의 발화와 타임스탬프를 일치시킵니다. 현재 음성 통신에 관한 공개 데이터셋이 부족하여 few-shot 기법을 적용할 필요성이 강조되고 있습니다.

- **Performance Highlights**: 논문에서는 중복 통신의 탐지를 위해 문장 유사도(semantic similarity)를 활용하는 접근법을 제안합니다. TF-IDF와 같은 전통적인 방식 대신, 신경망 기반의 방법을 사용하여 더 나은 성능을 발휘합니다. 기법의 최종 목표는, e스포츠 팀의 커뮤니케이션의 질이 경기 성능에 미치는 양상과 상관관계를 분석하는 것입니다.



### MoTe: Learning Motion-Text Diffusion Model for Multiple Generation Tasks (https://arxiv.org/abs/2411.19786)
Comments:
          Five figures, six tables

- **What's New**: 최근 대화형 AI 모델과 Denoising Diffusion Model의 발전에 힘입어, 인간의 동작 분석이 크게 향상되었습니다. 본 논문에서는 텍스트와 모션의 주변, 조건 및 결합 분포를 학습하여 다양한 작업을 가능하게 하는 통합 다중 모드 모델인 MoTe를 소개합니다. MoTe는 텍스트-모션 생성, 모션 캡셔닝, 텍스트 주도 모션 생성을 지원하며, 간단한 입력 맥락 수정만으로 다양한 작업을 수행할 수 있습니다.

- **Technical Details**: MoTe는 Motion Encoder-Decoder (MED), Text Encoder-Decoder (TED), Motion-Text Diffusion Model (MTDM)이라는 세 가지 주요 구성 요소로 구성되어 있습니다. MED와 TED는 각각 모션 시퀀스와 텍스트 설명을 재구성하기 위해 잠재 임베딩(latent embeddings)을 추출하도록 훈련됩니다. MTDM은 다양한 작업을 처리하기 위해 입력 맥락에서 반복적인 denoising 과정을 수행합니다.

- **Performance Highlights**: 벤치마크 데이터셋에서 MoTe는 텍스트-모션 생성 작업에서 우수한 성능을 보여주었고, 모션 캡셔닝 작업에서도 경쟁력 있는 결과를 나타냈습니다. MoTe는 다양한 작업을 간단한 입력 맥락 쇄신으로 수행 가능하며, 실험 결과는 모션 임베딩이 텍스트-모션 및 모션-텍스트 작업 성능에 미치는 영향을 제시합니다.



### PerLA: Perceptive 3D Language Assistan (https://arxiv.org/abs/2411.19774)
- **What's New**: 이 논문에서는 3D 언어 도우미(3D Language Assistant, 3DLA)인 PerLA를 소개합니다. PerLA는 다양한 포인트 클라우드 영역에서 고해상도(local) 세부정보를 병렬로 캡처하고, 전체 포인트 클라우드로부터 얻은 하의 해상도(global) 맥락과 통합하여 LLM을 위한 보다 유용한 시각적 표현을 생성합니다. 이를 통해 기존 방법들보다 더 많은 세부정보를 보존하면서도 교육 안정성을 증진시키기 위한 새로운 손실 함수도 제안하였습니다.

- **Technical Details**: PerLA는 포인트 클라우드 처리를 위해 힐버트 곡선(Hilbert curve)을 기반으로 하는 알고리즘을 사용하여 고유한 정보를 효과적으로 집계합니다. 이 알고리즘은 지역(local) 및 글로벌(global) 정보를 결합하기 위해 교차 주의(cross-attention) 및 그래프 신경망(Graph Neural Network, GNN)을 활용하여 고차원적인 포인트 수준의 표현을 생성합니다. 이러한 접근법을 통해 PerLA는 3D QA 및 3D 조밀 캡셔닝의 벤치마크인 ScanQA와 ScanRefer, Nr3D에서 높은 성능을 보여줍니다.

- **Performance Highlights**: PerLA는 기존 3D 언어 도우미에 비해 눈에 띄는 성능 향상을 기록하였습니다. 특히 ScanQA에서 질문 응답 시 최대 +1.34 CiDEr의 이점을 보였고, ScanRefer와 Nr3D에서는 각각 +4.22 및 +3.88을 기록하여 조밀한 캡셔닝에서도 높은 효과를 나타냈습니다. 이러한 결과들은 PerLA의 효율적인 정보 집계와 세부 정보 보존 전략이 성공적이었음을 보여줍니다.



### LongVALE: Vision-Audio-Language-Event Benchmark Towards Time-Aware Omni-Modal Perception of Long Videos (https://arxiv.org/abs/2411.19772)
Comments:
          18 pages, 15 figures

- **What's New**: 이 연구에서는 LongVALE라는 혁신적인 비디오 이해 벤치마크를 제안합니다. 이 벤치마크는 105K개의 오미-모달(omni-modal) 이벤트와 정밀한 시간 경계 및 세밀한 관계 인식을 포함하는 캡션을 제공합니다. 기존 연구들이 비주얼 데이터에 국한되거나 품질이 낮은 데이터셋에 머물러 있는 반면, LongVALE는 8.4K개의 고품질 긴 비디오를 바탕으로 진행되었습니다.

- **Technical Details**: LongVALE는 세 가지 주요 구성 요소로 이루어진 자동화된 주석 파이프라인을 통해 생성됩니다. 이러한 구성 요소는 고품질 비디오 필터링, 세멘틱적으로 일관된 오미-모달 이벤트 경계 탐지, 그리고 크로스-모달 상관관계를 인식하는 이벤트 캡셔닝입니다. 이 과정을 통해 오미-모달 이벤트에서 세밀한 캡션을 생성하고, 동영상 내의 시각 및 청각 정보를 통합적으로 처리하는 방법을 제시합니다.

- **Performance Highlights**: LongVALE 기반 모델은 기존 비디오 대형 언어 모델(LLMs)보다 모든 오미-모달 작업에서 뛰어난 성능을 나타냈습니다. 또한 우리의 모델은 적은 데이터로도 제로샷(Zero-shot) 일반 오디오-비주얼 질문 답변(AVQA) 작업에서 우수한 결과를 기록하였습니다. 이러한 결과들은 LongVALE 데이터셋이 포괄적인 다중 모달 비디오 이해 발전에 기여할 수 있는 잠재력을 보여줍니다.



### Noro: A Noise-Robust One-shot Voice Conversion System with Hidden Speaker Representation Capabilities (https://arxiv.org/abs/2411.19770)
Comments:
          Submitted to IEEE OJSP

- **What's New**: 본 논문에서는 노이즈에 강한 원샷 음성 변환(One-shot Voice Conversion, VC) 시스템인 Noro를 소개합니다. Noro는 불안정한 인터넷 환경에서 수집된 소스 음성이 다양한 잡음으로 인해 효과가 감소하는 문제를 해결하기 위한 혁신적인 디자인 요소를 갖추고 있습니다. 특히 듀얼 브랜치 참조 인코딩 모듈과 잡음 무관 대조 스피커 손실(noise-agnostic contrastive speaker loss)을 활용하여 음성 변환의 정확성과 신뢰성을 향상시킵니다.

- **Technical Details**: Noro는 기본 VC 시스템 위에 구축되며, 소스 인코더와 참조 인코더를 통해 얻은 피치 및 의미적 표현을 기반으로 음성을 생성합니다. 새로운 듀얼 브랜치 참조 인코딩 모듈은 클린 참조 음성과 잡음이 포함된 참조 음성을 각각 인코딩하여 공유 가중치 구조로 작동합니다. 또한, noise-agnostic contrastive speaker loss를 통해 동일한 스피커의 음성을 최대한 비슷하게 표현하면서 다른 스피커의 음성과는 최소한의 유사성을 가지도록 학습합니다.

- **Performance Highlights**: 실험 결과, Noro는 클린 및 노이즈 환경 모두에서 기존의 기본 시스템보다 성능이 뛰어난 것을 확인했습니다. VC-SPK2VEC로 지칭되는 참조 인코더의 스피커 표현 능력을 평가한 결과, SUPERB 설정 하에서 여러 최첨단(self-supervised learning, SSL) 모델과 비교했을 때 경쟁력을 갖춘 5.32%의 균등 오류율(Equal Error Rate, EER)을 달성하였습니다. 이는 원샷 VC 작업을 통해 스피커 표현 학습의 가능성을 보여줍니다.



### CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation (https://arxiv.org/abs/2411.19650)
Comments:
          Project Webpage: this https URL

- **What's New**: 본 논문에서는 VLM에서 파생된 새로운 VLA 아키텍처인 CogACT를 소개합니다. 기존의 VLA 모델은 VLM을 단순한 방식으로 행동 예측에 재사용했으나, CogACT는 VLM의 출력을 조건으로 하는 특화된 행동 모듈을 제안합니다. 이 아키텍처는 행동 신호의 특성을 효과적으로 처리하며, 분산 기반 변환기(diffusion transformer)를 사용하여 성능을 크게 향상시킵니다.

- **Technical Details**: CogACT는 인지(cognition)와 행동(action) 능력을 분리하여 설계되었습니다. 이는 인간의 뇌처럼 각기 다른 뇌 영역이 시각, 언어 및 운동 기능을 처리하는 방식에 착안했습니다. 또한, 행동 모듈을 위한 다양한 백본 아키텍처를 체계적으로 연구했으며, 특히 Diffusion transformers를 사용한 순차적 모델링이 단일 단계 행동 예측보다 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 실험 결과, CogACT 모델은 기존 VLA 모델보다 뛰어난 작업 성능을 보였습니다. OpenVLA와 비교했을 때, 시뮬레이션 평가에서 35% 이상, 실제 로봇 실험에서 55% 이상의 성공률 향상을 기록했습니다. 또한, 큰 RT-2-X 모델보다도 18% 높은 절대 성공률을 실험에서 달성하며, 새로운 로봇과 작업에 대한 빠른 적응력을 보여주었습니다.



### Accelerating Multimodal Large Language Models via Dynamic Visual-Token Exit and the Empirical Findings (https://arxiv.org/abs/2411.19628)
- **What's New**: 이 연구에서는 기존의 Multimodal Large Language Models (MLLMs)에서 발생하는 시각적 토큰의 과도한 사용이 계산 비용을 증가시키고 불필요한 중복을 야기함을 지적하고 있습니다. 저자들은 MLLMs의 주의(attention) 행동을 조사하여 시각적 중복 문제를 해결하기 위한 간단하지만 효과적인 방법인 '동적 시각적 토큰 종료(Dynamic Visual-Token Exit, DyVTE)'를 제안하였습니다. DyVTE는 경량의 하이퍼 네트워크(hyper-networks)를 사용하여 텍스트 토큰의 상태를 인식하고 특정 계층 이후에 모든 시각적 토큰을 제거합니다.

- **Technical Details**: MLLM의 추론(inference) 과정은 세 가지 주요 단계로 나뉩니다: (i) 초기 융합(early fusion), (ii) 동내성 모델링(intra-modality modeling), (iii) 다중 모달 추론(multimodal reasoning)입니다. 연구 결과, 텍스트 토큰이 충분한 이미지 정보를 수신하면 시각적 토큰이 더 이상 추론에 기여하지 않음을 발견하였습니다. DyVTE는 이 발견을 바탕으로 효율성을 높이고 추론 속도를 증가시키는 방법으로 제안되었습니다.

- **Performance Highlights**: DyVTE는 다양한 벤치마크에서 LLaVA, VILA, Eagle, InternVL 과 같은 여러 MLLMs에 적용되어 실험 결과 효율성과 성능을 모두 향상시켰습니다. 예를 들어, LLaVA-1.5의 계산 비용을 최대 45.7% 감소시키면서도 성능의 현저한 하락 없이 경쟁력을 유지합니다. DyVTE는 Flash Attention과 FastV와 같은 기존의 시각적 토큰 가지치기(token pruning) 방법과의 호환성도 갖추고 있습니다.



### Knowledge Management for Automobile Failure Analysis Using Graph RAG (https://arxiv.org/abs/2411.19539)
Comments:
          7 pages, 6 figures, to be published in 2024 IEEE International Conference on Bid Data (BigData)

- **What's New**: 이 연구는 자동차 고장 분석을 위한 지식 관리 시스템을 제안합니다. 이 시스템은 대규모 언어 모델(LLM)과 지식 그래프(KG)를 결합한 Retrieval-Augmented Generation (RAG) 방식에 기반하고 있습니다. 특히 그래프 RAG는 젊은 엔지니어들이 기존의 KG에서 효과적으로 정보를 추출하고 이해할 수 있도록 최적화되었습니다. 이를 통해 고장 원인을 신속하게 식별할 수 있는 시스템으로의 발전을 목표로 합니다.

- **Technical Details**: 이 논문은 자동차 고장 분석에 있어 기존의 지식 그래프와 함께 사용할 수 있는 새로운 그래프 RAG 시스템을 제안합니다. 이 시스템은 LLMs의 출력 성능을 향상시키기 위한 ROUGE F1 점수를 기반으로 한 실험을 통해 그 효과성을 입증하였습니다. 특히, 그래프 RAG는 KG에서 유용한 서브 그래프를 추출하고, 젊은 엔지니어들이 쉽게 이해할 수 있도록 하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 제안된 방법에 의해 생성된 문장은 기존의 방법보다 평균 157.6% 향상된 ROUGE F1 점수를 기록했습니다. 이는 자동차 고장 분석을 위한 지식 관리에서 제안된 시스템의 효과성을 강조합니다. 이러한 성과는 고장 분석 전문 지식을 젊은 엔지니어에게 효과적으로 전달할 수 있는 가능성을 보여줍니다.



### TQA-Bench: Evaluating LLMs for Multi-Table Question Answering with Scalable Context and Symbolic Extension (https://arxiv.org/abs/2411.19504)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)이 복잡한 멀티-테이블 관계 데이터에서 질문 응답(QA) 작업을 효율적으로 수행할 수 있는 능력을 평가하기 위해 TQA-Bench라는 새로운 멀티-테이블 QA 벤치를 제안합니다. 기존 벤치마크는 단일 테이블 QA에 주로 초점을 맞추어 LLM의 복잡한 멀티-테이블 QA 능력을 제대로 평가하지 못하고 있습니다. TQA-Bench는 실제 공개 데이터셋에서 수집된 다양한 관계형 데이터베이스 인스턴스를 포함하고 있으며, 8K에서 64K 토큰까지의 다양한 멀티-테이블 컨텍스트 길이를 생성하는 유연한 샘플링 메커니즘을 도입합니다.

- **Technical Details**: TQA-Bench는 데이터 수집, 관계형 데이터 샘플링, 평가 작업 정의 및 기호 확장을 통한 질문 생성의 네 가지 주요 단계로 체계적으로 구성됩니다. 이를 통해 여러 데이터 소스(예: WorldBank, DataGov)에서 다수의 대규모 관계형 데이터베이스를 활용해 다양한 멀티-테이블 QA 작업을 생성합니다. 또한, 8K에서 64K 토큰까지의 다양한 컨텍스트 길이를 갖춘 평가 작업을 제안하여 실제 데이터를 처리하는 LLM의 성능을 정밀하게 평가합니다.

- **Performance Highlights**: 논문에서 실시한 광범위한 실험을 통해 다양한 LLM의 멀티-테이블 QA 작업에서의 성능을 평가하였고, 그 결과 복잡한 데이터 환경에서의 LLM의 도전 과제와 기회를 강조하고 있습니다. TQA-Bench는 LLM의 데이터 검색 및 패턴 매칭을 넘어서는 추론 능력을 평가할 수 있도록 기호 확장 기능을 통합하여 신뢰성을 높였습니다. 이러한 평가 결과는 복잡한 데이터 관리 작업에서 LLM 기반 애플리케이션의 설계 및 실행에 대한 귀중한 통찰력을 제공합니다.



### Actions and Objects Pathways for Domain Adaptation in Video Question Answering (https://arxiv.org/abs/2411.19434)
- **What's New**: 이번 논문에서는 Video QA(질문-답변)에서의 out-of-domain 일반화를 위한 새로운 방법론인 Actions and Objects Pathways (AOPath)를 제안합니다. AOPath는 대규모 pretrained 모델에서 추출한 기능을 활용하여 미세 조정 없이 일반화 능력을 향상시킵니다. 이는 인간의 뇌 구조에서 영감을 받아, pretrained 기능을 행동(feature)과 객체(feature)로 분리한 후 각기 다른 reasoning pathways를 통해 처리합니다.

- **Technical Details**: AOPath는 AOExtractor 모듈을 사용하여 도메인에 구애받지 않는 기능으로 변환하는 혁신적인 접근 방식을 채택하여 추가적인 학습 가능한 가중치를 도입하지 않습니다. 또한, 수행 과정에서 각 모달리티의 행동 특징과 객체 특징을 분리하여 추출하고, 이들 각각에 대해 별도의 reasoning pathway를 생성합니다. 제안된 방법론은 TVQA 데이터셋을 기준으로 평가되며, 이 데이터셋은 여러 장르로 구분되어 있습니다.

- **Performance Highlights**: AOPath는 기존의 분류기와 비교해 out-of-domain 데이터셋에서 5%, in-domain 데이터셋에서 4%의 성능 향상을 보여주었습니다. 또한 작은 버전인 AOPath_S는 훈련 가능한 가중치가 26,282개에 불과함에도 불구하고 이전의 방법들에 비해 3% 성능 향상을 이끌어 냈습니다. 이 접근법은 상대적으로 적은 매개변수로 높은 일반화 성능을 달성하였습니다.



### Libra: Leveraging Temporal Images for Biomedical Radiology Analysis (https://arxiv.org/abs/2411.19378)
- **What's New**: Libra는 의료 이미지를 이용한 방사선 보고서 생성을 위한 정확한 시간 인식을 위한 새로운 다중 모달 대규모 언어 모델(MLLM)입니다. 기존 모델들이 단일 이미지 분석에 주로 집중했음에도 불구하고, Libra는 과거 연구와 비교하여 현재 이미지를 평가하는 데 필요한 시간적 정보를 통합했습니다. 이를 통해 방사선 보고서 생성(RRG)에서 전문가급 성능을 달성하며, 임상 실습의 중대한 격차를 해결하고자 합니다.

- **Technical Details**: Libra는 RAD-DINO라는 사전 훈련된 이미지 인코더와 Meditron이라는 의료 대규모 언어 모델(LLM)을 통합하여 구성되었습니다. 이 모델은 Temporal Alignment Connector(TAC)를 사용하여 다양한 시간 지점에서의 이미지의 시간 정보를 캡처하고 통합합니다. TAC는 이미지의 고세분화 레이어별 특징을 추출하는 Layerwise Feature Extractor(LFE)와 과거의 연구로부터 얻어진 시간 참조를 통합하는 Temporal Fusion Module(TFM)로 구성됩니다.

- **Performance Highlights**: Libra는 MIMIC-CXR 데이터셋에서 방사선 보고서 생성(RRG) 작업의 최첨단 성능을 달성했습니다. 특히, RadCliQ 메트릭에서 12.9%의 개선이 이루어졌으며, 모든 어휘 메트릭에서도 이전 모델들보다 상당한 향상을 보였습니다. 이러한 성과는 Libra가 의료 이미지를 효과적으로 분석하고 해석하는 능력을 갖추고 있다는 것을 증명합니다.



### CLIP meets DINO for Tuning Zero-Shot Classifier using Unlabeled Image Collections (https://arxiv.org/abs/2411.19346)
- **What's New**: 이번 논문에서는 CLIP의 이미지 분류 성능을 개선하기 위해 DINO 기반의 라벨 없는 프롬프트 튜닝 기법인 NoLA (No Labels Attached)를 제안합니다. 이 방법은 자가 지도 학습(self-supervised learning) 모델의 시각적 특징과 대형 언어 모델(large language models)의 텍스트적 지식을 결합하여, 라벨이 없는 이미지를 사용해 분류 성능을 크게 향상시킵니다.

- **Technical Details**: NoLA는 세 가지 주요 단계로 구성됩니다: 첫째, LLM의 클래스 특성 설명을 활용해 강력한 텍스트 특징 임베딩을 생성합니다. 둘째, 이러한 텍스트 임베딩을 사용하여 DINO 모델의 시각적 특징과 통합된 Alignment 모듈을 훈련시키기 위한 의사 라벨(pseudo-labels)을 생성합니다. 마지막으로, 훈련된 Alignment 모듈을 이용해 CLIP의 비전 인코더를 DINO의 도움으로 프롬프트 튜닝합니다.

- **Performance Highlights**: 이 방법은 11개의 다양한 이미지 분류 데이터셋에서 평균 3.6%의 성능 향상을 보여주었으며, 기존의 최첨단 라벨 없는 분류 방법을 초월하는 효율적 접근방식을 제공합니다. NoLA는 특히 극소량의 라벨이 필요한 상황에서도 우수한 성능을 발휘하여, 미세 조정이 필요 없이 다양한 작업에서 뛰어난 유연성을 보여줍니다.



### Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation (https://arxiv.org/abs/2411.19331)
- **What's New**: 이 논문에서는 Open-Vocabulary Segmentation (OVS) 문제를 해결하기 위한 새로운 접근 방식인 Talk2DINO를 제안합니다. Talk2DINO는 DINOv2의 공간적 정확성과 CLIP의 언어 이해 능력을 결합하여 이미지와 텍스트 간의 상호작용을 향상시킵니다. 이 방법은 하부 네트워크의 미세 조정 없이 CLIP의 텍스트 임베딩을 DINOv2의 패치 수준 피쳐에 매핑하는 학습된 함수를 통해 이뤄집니다.

- **Technical Details**: Talk2DINO는 DINOv2의 자기 주의(attention) 맵을 활용하여 시각적 패치를 텍스트 임베딩과 선택적으로 정렬합니다. 학습 시, 이 메커니즘은 다양한 의미(region)의 시각적 특징을 강조하여 이미지의 세분화(segmentation) 품질을 높입니다. 데이터 수집 과정 없이 새로운 매핑 기능을 배우는 이 방식은 기존 CLIP 기반 모델의 한계를 극복하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, Talk2DINO는 다양한 비지도 OVS 벤치마크에서 최첨단 성능을 달성하였으며, 텍스트와 이미지 간의 의미적 상관성을 효과적으로 증대시킵니다. 이 방법은 배경 객체를 효과적으로 분리할 수 있도록 하여 더 자연스럽고 노이즈가 적은 분할 결과를 제공합니다. 논문은 Talk2DINO가 CLIP 유사 모델의 공간적 이해 한계를 해결할 수 있는 새로운 길을 열었다고 강조합니다.



### Examining Multimodal Gender and Content Bias in ChatGPT-4o (https://arxiv.org/abs/2411.19140)
Comments:
          17 pages, 4 figures, 3 tables. Conference: "14th International Conference on Artificial Intelligence, Soft Computing and Applications (AIAA 2024), London, 23-24 November 2024" It will be published in the proceedings "David C. Wyld et al. (Eds): IoTE, CNDC, DSA, AIAA, NLPTA, DPPR - 2024"

- **What's New**: 이번 연구에서는 ChatGPT-4o의 다중 모달(content generation) 콘텐츠 생성에 대해 조사하였습니다. 특히 성적 콘텐츠와 누드에 대한 처리 방식에서 폭력 및 약물 관련 주제와의 현저한 불균형이 나타났습니다. 이를 통해 ChatGPT-4o가 성적 내용과 누드를 지속적으로 검열하는 반면, 폭력과 약물 사용에 대해서는 더 관대한 태도를 보이는 것을 알 수 있었습니다.

- **Technical Details**: 자세한 분석 결과, 성별에 따른 차별Bias가 두드러지며, 여성 관련 콘텐츠는 남성 관련 콘텐츠보다 더 엄격한 규제를 받는 것으로 나타났습니다. 이러한 불균형은 과거 AI 논란에 대한 미디어의 감시와 공공의 반발이 영향을 미쳤을 것으로 추측됩니다. 따라서 기술 기업들은 자사의 평판을 보호하기 위해 민감한 이슈에 대한 엄격한 가이드라인을 설정하게 되었습니다.

- **Performance Highlights**: 이번 연구는 AI 주도 언어 및 다중 모달 모델에서의 편향Bias를 이해하는 데 기여하고, 보다 균형 잡히고 윤리적인 콘텐츠 조정Practices의 필요성을 강조합니다. AI 시스템이 단순한 정치적 올바름을 초월하여 진정으로 윤리적 기준과 책임을 들 upheld해야 한다고 주장합니다.



### VARCO-VISION: Expanding Frontiers in Korean Vision-Language Models (https://arxiv.org/abs/2411.19103)
Comments:
          24 pages, 15 figures, 4 tables. Model weights at this https URL. Benchmarks released at NCSOFT's HuggingFace repositories (K-MMBench, K-SEED, K-MMStar, K-DTCBench, K-LLaVA-W). VARCO-VISION is an open-source Korean-English VLM with OCR, grounding, and referring capabilities

- **What's New**: 이 논문에서는 VARCO-VISION이라는 오픈소스 한국어-영어 비전-언어 모델(vision-language model, VLM)을 소개합니다. 본 모델은 언어와 비주얼 정보를 단계적으로 학습할 수 있도록 설계되어, 기존의 모델 지식을 보존하면서도 뛰어난 성능을 보여줍니다. 전반적으로 VARCO-VISION은 이미지-텍스트 이해 및 생성 능력에서 다른 유사 모델에 비해 뛰어난 성능을 발휘하며, 실제 상황에서도 활용 가능한 능력을 갖추고 있습니다.

- **Technical Details**: VARCO-VISION-14B는 비전 인코더, 프로젝터 및 대형 언어 모델(Large Language Model, LLM)로 구성되어 있습니다. Qwen-2.5-14B-Instruct를 언어 기초 모델로 사용하고, SigLIP를 비전 인코더로 활용하여 LLaVA-OneVision의 아키텍처를 따릅니다. 모델은 4단계의 훈련 전략으로 시각적 및 언어적 능력을 점진적으로 습득하도록 설계되어 있으며, OCR, grounding 및 referring과 같은 특정 용도를 위한 특수 토큰도 추가됩니다.

- **Performance Highlights**: VARCO-VISION은 여러 평가 기준에서 동급 모델들과 비교하여 뛰어난 성능을 보여줍니다. 특히, K-LLaVA-W와 같은 한국어 생성 능력 및 텍스트 기반 벤치마크에서 우수한 점수를 기록하며, 전반적인 다중 모드 및 텍스트 전용 평가에서도 강력한 언어 능력을 발휘합니다. 최종 단계인 선호 최적화(preference optimization)를 통해 모델의 반응 가독성 및 유창성이 크게 향상되었습니다.



### MATATA: a weak-supervised MAthematical Tool-Assisted reasoning for Tabular Applications (https://arxiv.org/abs/2411.18915)
- **What's New**: MATATA는 테이블 데이터 문제 해결을 위한 새로운 비용 효과적인 방법을 제시합니다. 이 방법은 LLM 에이전트를 훈련하기 위해 Reasoning, Planning, Tool Use의 세 가지 과정을 포함하며, 특히 데이터 프라이버시가 중요한 민감한 비즈니스를 위한 로컬 호스팅에 적합합니다. MATATA는 3.8B/8B 규모의 Small Language Models (SLMs)를 통해 성능을 강화하며, 다양한 데이터 세트에서 유연하고 재사용 가능한 도구를 사용하여 효율적인 확장성을 실현합니다.

- **Technical Details**: MATATA는 Progressively Self-Improvement Paradigm과 Iterative Weak Supervision을 활용하여, 다양한 도구와 함께 다단계 Reasoning을 가능하게 합니다. 이 프레임워크는 문제가 복잡할 경우 이를 하위 작업(subtask)으로 나누고, 각 하위 작업을 SLM을 사용하여 해결합니다. 사용된 도구들은 데이터에 대한 민감도로 인해 폐쇄형 LLM 모델을 사용할 수 없는 비즈니스 문서의 상황에서도 매우 효과적입니다.

- **Performance Highlights**: 실험 결과 MATATA는 FinQA 및 TAT-QA에서 최신 성능을 달성하였으며, 오픈 소스 모델 기반의 Reasoning 프레임워크 중에서 우수한 결과를 보였습니다. 또한 MATATA 모델은 TabMWP에서 GPT-4 기반 프레임워크와 경쟁할 수 있는 성과를 보여 주목받고 있습니다. 이러한 결과는 MATATA의 효율성을 입증하며, 다양한 비즈니스 응용 프로그램에 적합한 솔루션으로 자리 잡았습니다.



### Evaluating Sparse Autoencoders on Targeted Concept Erasure Tasks (https://arxiv.org/abs/2411.18895)
- **What's New**: 본 연구에서는 Sparse Autoencoders (SAEs)의 성능 평가를 위한 새로운 메트릭을 소개합니다. 기존의 비지도 방식만으로 진행되던 SAE 평가의 한계를 극복하기 위해, SHIFT와 Targeted Probe Perturbation (TPP)이라는 새로운 방법을 제안하고 있습니다. 이러한 메트릭은 진화하는 SAE 기술의 품질을 보다 객관적으로 평가할 수 있게 해줍니다.

- **Technical Details**: SHIFT는 인간 주석자가 판단한 중요하지 않은 SAE 특징을 제거하여 분류기의 착오를 줄이는 방법입니다. 본 연구에서는 LLM을 활용하여 SHIFT를 자동화된 SAE 품질 평가 메트릭으로 변환하였습니다. TPP는 SAE가 유사 개념을 분리하는 능력을 수량화하여 SHIFT를 다양한 데이터셋으로 확장하는 기능을 가지고 있습니다.

- **Performance Highlights**: SHIFT와 TPP 메트릭은 여러 오픈 소스 모델에 적용되어 다양한 SAE 교육 하이퍼파라미터 및 아키텍처 간의 차이를 효과적으로 구분할 수 있음을 입증하였습니다. 이 연구의 결과들은 SAE의 품질 평가에 대한 새로운 기준을 제시하며, 보다 실질적인 모델 해석 가능성을 향상시킬 것으로 기대됩니다.



### ArEEG_Words: Dataset for Envisioned Speech Recognition using EEG for Arabic Words (https://arxiv.org/abs/2411.18888)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2402.15733

- **What's New**: 이 논문에서는 첫 번째 아랍어 EEG 데이터셋인 ArEEG_Words를 소개합니다. 이 데이터셋은 22명의 참가자로부터 수집된 EEG 신호로 구성되어 있으며, 이제까지 아랍어와 관련된 BCI 연구에서 사용 가능한 공개 데이터셋이 부족한 문제를 해결하고자 합니다.

- **Technical Details**: ArEEG_Words 데이터셋은 14채널 Emotiv Epoc X 장치를 사용하여 기록되었습니다. 참가자는 8시간 전부터 카페인, 알코올, 담배와 같은 신경계에 영향을 미치는 물질을 피하고, 평온한 방에서 16개의 아랍어 단어 중 하나를 마음속으로 상상하며 10초간 집중하도록 지시받았습니다.

- **Performance Highlights**: 총 352회의 EEG 기록이 수집되어 각 기록은 250ms 신호로 나뉘어 총 15,360개의 EEG 신호가 생성되었습니다. ArEEG_Words 데이터셋은 아랍어 EEG 연구 분야에서는 최초이며, 연구자들에게 공개되어 이 분야의 연구를 촉진할 것으로 기대됩니다.



### Reconstructing Animals and the Wild (https://arxiv.org/abs/2411.18807)
Comments:
          12 pages; project page: this https URL

- **What's New**: 이 논문에서는 자연 환경에서 동물 및 자연 경관을 재구성하는 새로운 방법을 제안합니다. 기존의 많은 연구가 동물의 2D 모델링에 집중된 반면, 이 연구는 환경과 동물을 동시에 고려하여 3D 재구성을 시도합니다. 이를 통해 연구자는 단일 이미지에서 복잡한 자연 장면을 재구성하는 새로운 접근 방식을 제시하고 있습니다.

- **Technical Details**: 제안된 방법은 Large Language Models (LLMs)의 최근 발전을 기반으로 하며, CLIP 임베딩을 구조화된 구성 장면 표현(compositional scene representation)으로 디코딩하는 자기 회귀 모델을 훈련시킵니다. 연구자는 또한 천 개의 자산과 함께 한 백만 개의 이미지로 구성된 합성 데이터셋을 생성하여 모델이 실제 이미지에서 동물과 환경을 재구성하는 데 일반화되도록 합니다.

- **Performance Highlights**: 연구 결과, 제안된 접근 방식은 실제 이미지에서 동물과 그 환경을 성공적으로 재구성할 수 있는 능력을 보여줍니다. 모델은 CLIP 임베딩을 활용하여 다양한 자산을 효과적으로 추정하며, 재구성된 장면은 해석 가능하고 편집 및 애니메이션 작업 또한 지원합니다. 공개 예정인 데이터셋과 코드는 향후 연구를 촉진하는 데 기여할 것입니다.



### UOE: Unlearning One Expert Is Enough For Mixture-of-experts LLMS (https://arxiv.org/abs/2411.18797)
- **What's New**: 이 연구는 Mixture-of-Experts (MoE)라는 대규모 언어 모델(LLM)의 새로운 특성과 함께 전통적인 학습 제거 방법의 한계를 파악하고, 이를 해결하기 위한 혁신적인 단일 전문가 제거 프레임워크인 UOE를 제안하고 있습니다. UOE는 특정 지식과 가장 관련이 깊은 전문가에 집중하여 제거 작업을 수행하며, 이 과정에서 모델의 유용성을 최대한 유지하도록 설계되었습니다. 전체적으로, 이 연구는 MoE LLM의 제거에 있어서 발생할 수 있는 독특한 문제를 해결하고자 하는 최초의 시도를 보여줍니다.

- **Technical Details**: UOE는 전문가 귀속(expert attribution) 기술을 사용하여 특정 포기의 요소와 가장 많이 연관된 전문가를 식별합니다. 이어서, 러우터(router)에 앵커 손실(anchor loss)을 적용하여 해당 전문가의 활성 상태를 안정화시킵니다. 이를 통해 전문가 선택의 빈번한 변화를 방지하고, 집중적이고 제어된 형태로 제거를 구현할 수 있습니다. UOE는 다양한 제거 알고리즘과 호환 가능하여 기존 방법과 통합하여 사용할 수 있습니다.

- **Performance Highlights**: 실험 결과, UOE 프레임워크는 다양한 MoE LLM 아키텍처에서 모델의 유용성을 평균 35% 향상시키고, 지식 제거의 질을 최대 5%까지 향상시켰습니다. 흥미롭게도, 모델 매개변수의 단 0.06%만을 제거하면서 이러한 성과를 달성하였습니다. 이는 MoE LLM에 대한 효율적인 제거 방식을 입증하며, 새로운 연구 방향을 제시합니다.



### Cyber-Attack Technique Classification Using Two-Stage Trained Large Language Models (https://arxiv.org/abs/2411.18755)
- **What's New**: 이번 연구에서는 사이버 공격 관련 데이터 부족 문제를 해결하기 위해 새로운 문장 분류 시스템을 제안합니다. 이 시스템은 사이버 위협 인텔리전스(CTI) 보고서에서 기술된 공격 기법을 자연어 문장으로 분류할 수 있는 능력을 갖추고 있습니다. 저자들은 데이터 보강과 두 단계의 훈련 방법론을 통해 낮은 리소스 환경에서도 효과적인 분류 성능을 달성할 수 있음을 보여주었습니다.

- **Technical Details**: 제안하는 모델은 MITRE ATT&CK 프레임워크의 공격 기술(TTPs)을 식별하기 위해 훈련된 대형 언어 모델(LLM)을 기반으로 합니다. 연구에서는 데이터의 불균형을 해결하기 위해 유사한 보조 데이터를 선택하여 첫 번째 단계에서 모델을 훈련하고, 그 후에는 주 데이터만을 이용해 모델을 더욱 세밀하게 조정하는 두 단계 훈련 방법을 사용합니다. 이를 통해 분류 성능을 향상시킬 수 있는 방법론이 제시됩니다.

- **Performance Highlights**: 연구 결과, 제안된 방법은 TRAM 데이터셋에서 Macro-F1 점수를 5에서 9% 포인트 향상시켰으며, Micro-F1 점수 또한 경쟁력 있는 수준을 유지했다는 것을 나타냅니다. 이러한 결과는 LLM을 활용한 접근이 사이버 공격 분류 작업에서 실질적인 효율성을 제공할 수 있음을 뒷받침합니다.



### Multi-Task Model Merging via Adaptive Weight Disentanglemen (https://arxiv.org/abs/2411.18729)
- **What's New**: 이번 연구에서는 모델 머징(model merging) 분야의 기존 접근 방식을 개선하기 위해 Task Arithmetic 속성을 재검토하고 Task Consistency 속성을 제안하였습니다. Task Consistency 속성은 병합된 모델의 성능을 향상시키면서 작업 간 간섭(interference)을 최소화하려는 목적을 가지고 있습니다. 실험적으로 이 속성을 충족하는 직교(task vectors) 작업 벡터를 찾는 과정이 성능 향상에 기여함을 보여주었습니다.

- **Technical Details**: Adaptive Weight Disentanglement(AWD)라는 새로운 방법론을 도입하여 전통적인 작업 벡터를 중복 벡터와 여러 개의 분리된 작업 벡터로 분해합니다. AWD의 주요 최적화 목표는 분리된 작업 벡터 간의 직교성을 달성하고자 하며, 이는 효율적인 성능 전이를 가능하게 합니다. 이 과정에서 코사인 유사도를 최소화하고 중복 벡터의 크기를 줄이는 두 가지 최적화 목표를 설정하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 AWD가 기존 모델 머징 접근 방식에 비해 일관되게 우수한 성능을 나타냄을 확인했습니다. 특히, ViT-B/32 모델에서 평균 정확도가 2.8%, ViT-L/14 모델에서 1.5% 향상되었습니다. 또한, 이 방법은 언어 모델에서도 효과적으로 일반화되며, 다양한 조건에서 향상된 강건성을 보였습니다.



### Evaluating Vision-Language Models as Evaluators in Path Planning (https://arxiv.org/abs/2411.18711)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 계획 능력에 대한 한계를 극복하기 위해, 비전 언어 모델(VLMs)을 계획 평가 도구로 사용할 수 있는지를 탐구합니다. 특히, 복잡한 경로 계획 시나리오에서 VLM의 성능을 평가하기 위한 새로운 벤치마크인 PathEval을 도입합니다.

- **Technical Details**: PathEval 벤치마크는 VLM이 시나리오 설명에서 최적 경로의 특성을 추상화하고, 각 경로에 대한 정밀한 저수준 인식을 나타내며, 이 정보를 통합하여 더 나은 경로를 결정하는 능력을 요구합니다. 실험 분석에 따르면, 최신 VLM 모델들은 벤치마크에서 중대한 도전에 직면하고 있으며, 주어진 시나리오를 정확하게 추상화하는 능력은 있지만, 저수준 세부사항을 인지하는 데 어려움을 겪고 있습니다.

- **Performance Highlights**: 실험 결과, VLM의 비전 구성 요소가 주요 bottleneck(병목 현상)으로 작용하며, 이는 단순한 end-to-end fine-tuning을 통해 해결될 수 없는 문제임을 보여줍니다. 따라서, 효과적인 경로 평가자가 되기 위해서는 이러한 비전 인코더의 작업별 차별적 적응(task-specific discriminative adaptation)이 필요합니다.



### An indicator for effectiveness of text-to-image guardrails utilizing the Single-Turn Crescendo Attack (STCA) (https://arxiv.org/abs/2411.18699)
- **What's New**: 이번 연구에서 소개된 Single-Turn Crescendo Attack (STCA)은 텍스트-투-텍스트 AI 모델의 윤리적 안전장치를 우회하여 해로운 콘텐츠를 생성하도록 유도하는 혁신적인 방법입니다. 본 기술은 단일 프롬프트 내에서 맥락을 전략적으로 소거하며, 이는 모델이 의도치 않은 출력을 생성하도록 속이는 데 사용됩니다. 또한, STCA를 텍스트-투-이미지 모델에 적용하여 DALL-E 3 모델의 안전장치를 침해하는 데 성공하였고, 검열이 없는 모델과의 비교를 통해 그 효과를 입증하였습니다.

- **Technical Details**: 본 연구는 Crescendo Attack을 소개하며, 이 공격은 텍스트-투-텍스트 모델의 안전 필터를 우회하기 위해 반복적인 맥락을 점진적으로 추가하는 방식으로 구성됩니다. 기존의 다중 회전(Multi-Turn) 접근법 외에도, STCA를 통해 단순한 한 번의 입력으로도 공격을 수행할 수 있게 되었으며, 이는 보다 효율적이고 자동화된 공격으로 이어집니다. 우리는 DALL-E 3를 대상으로 STCA의 효과를 평가하고, 어떤 정도로 그 모델이 '감옥에서 탈출(jailbroken)' 되었는지를 보여주기 위해 Flux.1 모델과의 비교를 통해 이를 정량화했습니다.

- **Performance Highlights**: DALL-E 3 모델에 대한 STCA 공격의 초기 실험 결과는 모델이 악의적인 요청에 대해 의도하지 않은 이미지를 생성하는 데 높은 취약성을 보임을 나타냅니다. 연구팀은 101개의 악의적인 시나리오를 생성하고, 이를 통해 각 시나리오가 발생한 결과를 평가했습니다. 이는 결국 텍스트-투-이미지 생성 모델의 안전 문제를 드러내며, AI 안전성 강화를 위한 연구 방향에 중요한 기초 자료를 제공합니다.



### Verbalized Representation Learning for Interpretable Few-Shot Generalization (https://arxiv.org/abs/2411.18651)
- **What's New**: 이번 연구에서는 Verbalized Representation Learning (VRL)이라는 새로운 방법론을 제안합니다. 이 방법은 적은 수의 데이터로 객체 인식을 위한 인간 해석 가능한 특징을 자동으로 추출합니다. VRL은 Vision-Language Model (VLM)을 활용하여 서로 다른 클래스 간의 차이와 같은 클래스 내 공통점을 자연어로 포착합니다.

- **Technical Details**: VRL은 몇 가지 이미지 샘플을 사용하여 자연어로 서술된 특징을 수치 벡터로 변환합니다. 이러한 수치적 특징은 분류기(Classifier) 훈련 및 추론에 사용될 수 있습니다. 연구에서는 이 방법이 iNaturalist 및 Kiki-Bouba 데이터셋에서 우수한 성능을 보여주었으며, 적은 데이터로도 24%의 성능 향상을 달성했습니다.

- **Performance Highlights**: VRL은 또한 인간이 주석한 특징들보다 20% 더 나은 성능을 보였습니다. 이로 인해 VRL은 세밀한 분류 작업에서도 높은 범용성을 보여줍니다. 결과적으로, 이 연구는 VRL이 모델의 해석 가능성을 높이며 낮은 자원 조건에서도 효과적으로 일반화할 수 있는 가능성을 제시합니다.



### Towards Advanced Speech Signal Processing: A Statistical Perspective on Convolution-Based Architectures and its Applications (https://arxiv.org/abs/2411.18636)
- **What's New**: 이 논문에서는 CNN(Convolutional Neural Networks), Conformers, ResNets, 및 CRNNs와 같은 컨볼루션 기반 모델을 조사하여 음성 신호 처리 분야에서의 응용을 설명합니다. 이 연구는 음성 인식, 화자 식별, 감정 인식, 음성 개선과 같은 다양한 적용 분야를 포함하고 있습니다. 저자는 각 모델의 장단점을 비교하고, 통계적 배경을 제공하며, 음성 기술 발전의 중요성을 강조합니다.

- **Technical Details**: 본 논문에서는 컨볼루션 프로세스가 어떻게 음성 신호의 수정에 적용되는지를 설명합니다. 컨볼루션은 두 함수의 곱을 합산하여 또 다른 결과를 생성하는 수학적 과정으로, LTI(Linear Time-Invariant) 시스템 분석에 효과적입니다. 음성 신호 처리에서는 음성 신호와 채널 임펄스 응답 간의 컨볼루션을 통해 다양한 간섭, 반향 및 감쇠의 영향을 모델링합니다.

- **Performance Highlights**: 딥러닝의 최근 발전은 CNN, Conformers, CRNN, ResNets 등의 깊고 컨볼루션 기반의 아키텍처를 가능하게 하여 음성 신호 처리의 정확도를 향상시킵니다. 이들 아키텍처는 노이즈와 변화가 심한 환경에서도 신뢰성 있는 성능을 발휘합니다. 특히 CNN은 속도와 특성 추출에서 이점을 가지며, CRNN은 시간적 동역학을 학습하여 성능을 강화합니다.



### MiniKV: Pushing the Limits of LLM Inference via 2-Bit Layer-Discriminative KV Cach (https://arxiv.org/abs/2411.18077)
- **What's New**: 이 논문에서는 LLM(대형 언어 모델)을 효과적으로 제공하기 위해 KV 캐시를 최적화하는 방법인 MiniKV를 소개합니다. MiniKV는 2비트 레이어 구별 KV 캐시를 통해 긴 컨텍스트 작업에서의 정확도를 유지하면서 KV 캐시 크기를 크게 줄이는데 초점을 맞추고 있습니다. 이 방법은 FlashAttention과 호환되는 특화된 CUDA 커널을 개발하여 구현됩니다. 실험 결과 MiniKV는 86%의 KV 캐시 압축 비율을 달성하면서도 98.5% 이상의 정확도를 회복합니다.

- **Technical Details**: MiniKV는 선택적 KV와 2비트 양자화를 결합하여 KV 캐시를 효과적으로 압축합니다. 이는 먼저 사전 채우기(pre-fill) 단계에서 지속적으로 중요한 토큰을 선택하고, 생성 단계에서 이들을 고정하여 양자화하는 방식을 사용합니다. 또한, 레이어 구별 정책을 통해 KV 캐시 선택 정책을 향상시키고, 이 과정에서 MiniKV의 메모리 소비를 줄이는 결과를 가져옵니다. 이를 통해 장기 맥락 작업에서의 효율성이 증가하며, 여러장치에서 활용 가능한 성능을 보여줍니다.

- **Performance Highlights**: MiniKV는 여러 긴 컨텍스트 작업에서 평가되었으며, 성능 실험 결과 특정 벤치마크에서 같은 종류의 시스템 대비 66.4% 높은 최대 처리량을 기록했습니다. 이 논문은 MiniKV가 44,000 토큰의 프롬프트 길이를 지원하며, KV 캐시의 최대 8배 감소를 실현할 수 있음을 보여줍니다. 이러한 결과들은 LLM의 효율성을 획기적으로 향상시킬 수 있는 방법을 제시합니다.



New uploads on arXiv(cs.IR)

### Cross-Domain Recommendation Meets Large Language Models (https://arxiv.org/abs/2411.19862)
Comments:
          12 pages

- **What's New**: 본 연구에서는 큰 언어 모델(LLMs)의 추론 능력을 활용하여 교차 도메인 추천(Cross-domain recommendation, CDR) 문제를 해결하는 새로운 접근 방식을 제안합니다. 기존의 CDR 모델이 복잡한 신경망 아키텍처에 의존하고 큰 데이터셋과 계산 자원을 필요로 하는 반면, LLMs는 상대적으로 단순하게 데이터가 부족한 상황에서도 효과적으로 작동할 수 있다는 점에서 새로운 가능성을 보여줍니다. 특히, 두 가지 새로운 프롬프트 디자인을 도입하여 LLMs가 기존의 CDR 기준선 모델을 넘어서는 성과를 낼 수 있음을 입증하였습니다.

- **Technical Details**: 이 연구에서는 두 가지 유형의 프롬프트 전략을 도입하여 CDR 도메인에서 사용자 상호작용 데이터를 효과적으로 활용하고, 훈련된 모델의 역할과 정보의 명확한 분리를 통해 LLM의 결과를 극대화할 수 있는 방안을 모색하였습니다. 프롬프트를 통해 LLM의 역할을 정의하고, 등록된 데이터와 업무 정의를 명확히 하여 LLM이 처리할 수 있는 정보를 체계적으로 설계했습니다. 이 과정에서 LLM의 추론 능력을 자유롭게 활용할 수 있도록 하여, 전통적인 CDR 방법의 한계를 극복하는 데 기여하고자 하였습니다.

- **Performance Highlights**: 실험 결과, LLM들은 랭킹 및 평점 예측 과제에서 기존의 최첨단 CDR 모델들과 비교하여 높은 성과를 보였습니다. 더 나아가, 다양한 도메인 조합에서의 성과를 평가함으로써, LLM이 단순히 데이터를 암기하는 것에 그치지 않고, 새로운 패턴을 발견하고 일반화된 추천을 생성하는 능력을 보여주었습니다. 이 연구의 결과는 LLM이 추천 시스템에서의 교차 도메인 추천 성능을 향상시킬 수 있는 유망한 방법임을 시사합니다.



### Know Your RAG: Dataset Taxonomy and Generation Strategies for Evaluating RAG Systems (https://arxiv.org/abs/2411.19710)
Comments:
          to be published in the 31st International Conference on Computational Linguistics (COLING 2025)

- **What's New**: 이번 논문에서는 Retrieval Augmented Generation (RAG) 시스템의 성능 측정을 위해 공용 Q&A 데이터셋을 사용하는 것이 비효율적일 수 있음을 보여줍니다. 또한, 기존 RAG 데이터셋 생성 툴들이 불균형한 데이터를 초래할 수 있음을 언급하며 이를 해결하기 위한 솔루션을 제시합니다. 우리는 데이터의 특성을 분석하고 이를 기반으로 한 라벨링 및 라벨 지향 데이터 생성을 통해 RAG 시스템의 성능 평가 방법을 개선할 수 있는 가능성을 탐구합니다.

- **Technical Details**: 논문은 RAG 시스템의 (context, query, answer) 삼중항의 라벨 분류법을 제안합니다. 제안된 라벨은 fact_single, summary, reasoning, unanswerable 등 다양한 질문 유형에 따라 구분됩니다. 이러한 분류는 RAG 시스템의 성능 평가 시 사용자의 요구에 맞는 정보를 제공할 수 있도록 돕습니다. 또한, 소형 LLM을 튜닝하여 Q&A 데이터셋을 효율적으로 생성하는 방법을 소개합니다.

- **Performance Highlights**: 논문에서 제안하는 솔루션은 RAG 시스템의 개발 과정에서 데이터 식별을 중요하게 생각하는 단계로 볼 수 있습니다. 공공 데이터셋에서의 라벨 분포를 조사하여, 데이터셋의 품질을 통해 RAG 시스템의 성능을 더 정확하게 평가할 수 있는 방법을 제시합니다. 또한, 기존 데이터 생성 방법들에 비해 저비용으로 보다 다양하고 정확한 Q&A 데이터셋을 생성할 수 있는 가능성을 보여줍니다.



### A Review of LLM-based Explanations in Recommender Systems (https://arxiv.org/abs/2411.19576)
- **What's New**: 이번 연구에서는 LLaMA와 ChatGPT와 같은 대형 언어 모델(LLMs)의 부상으로 추천 시스템의 설명 가능성이 크게 향상될 수 있는 기회를 탐색했습니다. 전체 문헌 리뷰를 통해 LLMs를 활용한 추천 설명 생성에 집중하는 연구를 실시했으며, 이는 투명성과 사용자 신뢰를 증진시키는 중요한 요소입니다.

- **Technical Details**: 우리는 ACM Computing Literature를 통해 ChatGPT 출시 이후의 논문들을 포괄적으로 조사하였고, 총 232개의 기사를 검토한 결과 오직 6차례의 논문만이 LLMs의 추천 설명 활용에 직접적으로 다루고 있음을 발견했습니다. 이를 통해 LLMs의 적용이 여전히 초기 단계임을 확인할 수 있었습니다.

- **Performance Highlights**: 선택된 연구들을 분석함으로써 현재의 방법론을 이해하고, 도전 과제를 식별하며, 미래 연구 방향을 제안했습니다. LLM이 추천 시스템의 설명을 개선할 수 있는 잠재성을 강조하며 더욱 투명하고 사용자 중심의 추천 설명 솔루션 개발을 촉구하고 있습니다.



### ContextGNN: Beyond Two-Tower Recommendation Systems (https://arxiv.org/abs/2411.19513)
Comments:
          14 pages, 1 figure, 5 tables

- **What's New**: 이 논문에서는 Context-based Graph Neural Networks (ContextGNNs)라는 새로운 딥러닝 아키텍처를 도입하여 추천 시스템의 link prediction 문제를 해결하고자 합니다. ContextGNN은 사용자의 지역 서브그래프 내의 익숙한 아이템에 대해 pair-wise 표현 방식을 사용하면서 탐색 아이템의 추천을 위해서는 두 타워 표현을 활용합니다. 이러한 통합 구조는 추천의 품질을 개선하고 다양한 데이터 특성에 적응할 수 있는 강력한 방법론이 됩니다.

- **Technical Details**: 추천 시스템은 사용자와 아이템의 쌍을 기반으로 예측을 수행하고, ContextGNN은 이를 위해 pair-wise와 두 타워 모델을 결합합니다. 사용자의 과거 상호작용 데이터를 통해 세밀한 패턴을 포착하며, 서로 연결되지 않은 아이템 쌍에 대해서는 두 타워 모델을Fallback으로 사용합니다. 최종 네트워크는 두 가지 추천 방법을 결합하여 사용자에게 맞춤화된 순위를 생성하는 데 초점을 맞추고 있습니다.

- **Performance Highlights**: ContextGNN은 기존의 전통적인 방법들과 GNN 기반 방법들보다 평균 20% 성능을 향상시켰습니다. 특히, Pair-wise 표현 기반의 최상위 결과와 비교했을 때 평균 344%의 성능 향상을 보였습니다. 실험 결과들은 ContextGNN이 현실 세계 데이터셋에서 다양성을 잘 처리할 수 있음을 보여줍니다.



### Zero-Indexing Internet Search Augmented Generation for Large Language Models (https://arxiv.org/abs/2411.19478)
- **What's New**: 이 논문에서는 Internet search augmented generation paradigm을 제안하여 기존의 retrieval augmented generation(RAG) 시스템을 발전시켰습니다. 이 새로운 방식은 Google이나 Bing과 같은 표준 검색 엔진 API를 활용하여 실시간으로 최신 정보를 통합함으로써 콘텐츠의 품질을 향상시킵니다. 고정된 전처리된 데이터셋을 사용하는 RAG 시스템의 한계를 해결하기 위해, 우리는 모든 세부 데이터를 동적으로 처리하는 접근 방식을 채택했습니다.

- **Technical Details**: 제안된 시스템은 (i) 단일 추론에서 인터넷 검색이 필요한지 판단하고 검색 키워드를 추출하는 parser-LLM; (ii) 검색 엔진 API에서 유입된 편향을 줄이기 위한 혼합 랭킹 전략; (iii) 각 HTML 파일에서 관련 정보를 정확하고 효율적으로 추출하는 extractor-LLM을 포함합니다. 이러한 요소들은 실시간으로 관련 정보를 제공하며, 고정된 코퍼스의 유지 관리 없이 사용될 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 인터넷 검색 증강 생성 방식은 기존 RAG 시스템과 비교하여 월등한 품질 향상을 보여주었으며, 비용 효율적인 처리도 가능함을 입증하였습니다. extractor-LLM은 다양한 벤치마크에서 관련 정보를 정확히 추출할 수 있으며, 부적절한 내용이 있을 경우 허위 사실을 생성하는 대신 이를 거부하는 능력도 뛰어납니다.



### Parallel and Mini-Batch Stable Matching for Large-Scale Reciprocal Recommender Systems (https://arxiv.org/abs/2411.19214)
- **What's New**: 이 연구는 안정적인 매칭 이론(stable matching theory)을 활용하여 상호 추천 시스템(reciprocal recommender systems, RRSs)의 컴퓨터 효율성을 개선하는 새로운 방법론을 제안합니다. 이전의 방법들은 10,000명을 초과하는 실제 데이터 처리에서 계산적인 한계에 부딪혔습니다. 하지만 본 연구는 병렬 처리(parallel processing)와 미니 배치(mini-batch) 접근 방식을 통해 최적화 효율성을 높였습니다. 실험을 통해 최대 백만 사용자 규모의 데이터를 처리할 수 있는 가능성을 입증했습니다.

- **Technical Details**: 제안된 방법은 사용자 수에 따라 제곱 증가하는 계산 복잡성을 해결하기 위해 병렬 및 미니 배치 계산을 사용합니다. 반복적 비례 조정 절차(iterative proportional fitting procedure, IPFP)를 개선하여 안정적인 매칭 상태에 도달할 수 있도록 하였으며, 행렬-벡터 곱셈을 기반으로 한 병렬 계산 방법을 소개합니다. 또한, 메모리 효율성을 높이기 위해 미니 배치 업데이트 방법을 제안하며, 이러한 접근 방식을 사용하여 메모리에 적합하지 않은 대량의 개체(combination)들을 처리할 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 컴퓨팅 속도를 크게 향상시켰으며, 최대 백만 샘플의 대용량 데이터 처리에 성공했습니다. 이러한 성과는 단일 그래픽 처리 장치(Graphic Processing Unit, GPU)에서 이루어졌으며, 추천 개수의 손실 없이 높은 성능을 유지하는 것이 확인되었습니다. 이 연구는 안정적인 매칭 이론에 기반한 RRS의 대규모 데이터 처리 가능성을 처음으로 입증한 사례로 평가받고 있습니다.



### Introducing Three New Benchmark Datasets for Hierarchical Text Classification (https://arxiv.org/abs/2411.19119)
Comments:
          16 pages, 11 figures

- **What's New**: 이번 논문에서는 연구 출판 분야에서 사용할 수 있는 세 가지 새로운 Hierarchical Text Classification (HTC) 벤치마크 데이터셋을 소개합니다. 이 데이터셋은 Web of Science 출판 데이터베이스에서 저널 제목과 초록을 포함하고 있습니다. 기존의 분류 스키마의 한계를 극복하기 위해 저널 기반 및 인용 기반 분류를 결합한 접근 방식을 제안합니다.

- **Technical Details**: 세 가지 데이터셋을 생성하기 위해, 기존의 두 가지 분류 스키마를 바탕으로 기초 데이터셋을 먼저 작성하였습니다. 이러한 스키마의 단점을 보완하기 위해, 두 가지 분류를 결합하여 데이터셋의 신뢰성과 견고성을 개선하려는 방안을 제안합니다. 또한, 클러스터링 기반 분석을 통해 각 데이터셋의 품질을 평가하였습니다.

- **Performance Highlights**: 제안된 데이터셋에서 동일한 클래스에 속하는 문서들이 다른 데이터셋에 비해 의미적으로 더 유사함을 보여주었습니다. 마지막으로, 이 세 가지 새로운 데이터셋에 대해 네 가지 최첨단 HTC 접근 방식의 분류 성능을 제공하여, 향후 연구를 위한 기초를 마련하였습니다.



### Headache to Overstock? Promoting Long-tail Items through Debiased Product Bundling (https://arxiv.org/abs/2411.19107)
- **What's New**: 본 논문에서는 장기적인 상품 번들링(product bundling) 시나리오에서 인기 편향(popularity bias)을 개선하기 위해 Distilled Modality-Oriented Knowledge Transfer(DiëT) 프레임워크를 제안합니다. 이는 사용자 피드백 특성이 주는 왜곡을 해결하고, 원래의 번들링 의도를 유지하는 데 초점을 맞추고 있습니다. 특히, 번들에서 아이템 간 상관관계를 포착하기 위해 Popularity-free Collaborative Distribution Modeling 모듈(PCD)을 도입하여 장기 상품들의 효과적인 프로모션을 가능하게 합니다.

- **Technical Details**: DieT 프레임워크는 인기 없는 특성을 최대한 활용하는 데 중점을 두며, 이를 통해 Pop-to-LT(인기 아이템에서 장기 아이템으로의 번들링) 시나리오에서 발생할 수 있는 성능 저하 문제를 해결합니다. PCD는 인트라-뷰(self-attention mechanism)를 사용하여 번들-아이템의 계층적 협력 분포를 모델링하며, UBT(Unbiased Bundle-aware Knowledge Transferring) 모듈은 번들-아이템 뷰에서 다중모달(multi-modal) 뷰로의 방향성 지식 전이를 수행하여 인기 편향을 완화합니다.

- **Performance Highlights**: 실제 데이터셋을 통해 진행된 광범위한 실험 결과, DieT는 여러 최신 기술(SOTA) 방법들에 비해 Pop-to-LT 번들링 성능을 유의미하게 향상시켰습니다. 이는 장기 아이템의 프로모션을 효과적으로 수행할 뿐만 아니라 다양한 설정에서도 높은 일반화 및 견고성을 유지함을 보여줍니다. 결국, 본 연구는 장기 아이템 프로모션을 위한 Pop-to-LT 번들링 시나리오를 최초로 구체화하여 시스템적 연구를 진행했으며, 강력한 성과를 입증하였습니다.



### Unifying Generative and Dense Retrieval for Sequential Recommendation (https://arxiv.org/abs/2411.18814)
- **What's New**: 본 논문은 generative retrieval 및 sequential dense retrieval 두 방법을 비교하고, LIGER라는 새로운 하이브리드 모델을 제안합니다. LIGER는 sequential dense retrieval의 강점을 generative retrieval에 통합하여 성능 격차를 줄이고, 특히 cold-start 아이템 추천에서 향상된 결과를 도출합니다. 이러한 연구는 추천 시스템에서 사용자의 요구에 맞는 성능 및 계산 효율성을 함께 고려하고 있습니다.

- **Technical Details**: 이 논문에서는 generative retrieval과 dense retrieval의 수식을 제시하며, TIGER와 같은 generative retrieval 방법이 두 단계의 훈련 과정을 따른다고 설명합니다. 첫 번째 단계에서 아이템의 특성을 바탕으로 텍스트 설명을 수집하고, 다음 단계에서는 이러한 설명을 사용하여 콘텐츠 모델(예: language encoder)을 통해 아이템의 텍스트 임베딩을 생성하는 방식입니다. 여기서 각 아이템은 semantic ID를 부여받으며, 이러한 ID는 추천 과정에서 사용됩니다.

- **Performance Highlights**: 실험 결과, dense retrieval 방법이 대부분의 데이터셋에서 generative retrieval보다 우수함을 보여주었습니다. 특히 cold-start 아이템 예측에서 generative retrieval 방법은 성능이 저조한 반면, dense retrieval은 강력한 성능을 발휘했습니다. LIGER 모델을 통해서는 generative retrieval의 성능이 개선되었으며, 아이템 추천의 효율성과 효과성을 높이는 데 기여하고 있습니다.



### Counterfactual Learning-Driven Representation Disentanglement for Search-Enhanced Recommendation (https://arxiv.org/abs/2411.18631)
- **What's New**: 이 논문에서는 검색 강화 추천(search-enhanced recommendation) 시스템에서의 사용자 클릭 행동의 도메인 차이를 해결할 새로운 접근 방식인 ClardRec를 제안합니다. ClardRec는 사용자 일반 관심과 검색 특정 의도를 구별하여 추천 성능을 향상시키기 위해 반사실적 학습(counterfactual learning) 기반의 표현 분리 프레임워크를 사용합니다. 이는 검색 도메인에서 원하지 않는 특성을 제거하고, 사용자 관심과 관련된 일반적 특성을 강조함으로써 이루어집니다.

- **Technical Details**: 제안된 ClardRec 프레임워크는 사용자의 클릭과 쿼리 간의 관계를 분석하고, 쿼리 관련 및 쿼리 비관련 특성을 분리하는 세 가지 카운터팩트 목적(triplet counterfactual objectives)을 설정합니다. 이를 통해 검색 도메인에서 쿼리 독립적인 아이템 표현(item representation)을 학습하고, 이를 추천 도메인에 지식 전이 및 데이터 증강(data augmentation)을 통해 더욱 유용하게 활용합니다. 이러한 방식으로 추천의 정확성을 높이는 데 중점을 두고 있습니다.

- **Performance Highlights**: ClardRec는 협업 필터링(collaborative filtering) 및 순차 추천(sequential recommendation) 시나리오에서 실제 데이터셋을 기반으로 한 포괄적인 실험을 통해 우수한 성능을 발휘함을 입증하였습니다. 다양한 백본(backbone) 모델에 적용하였을 때, 추천 품질이 크게 향상되었음을 확인할 수 있었습니다. 이 논문은 추천 시스템의 성능을 높이기 위한 혁신적인 방법론을 제시하며, 특히 사용자 일반적 관심을 반영한 추천의 필요성을 강조합니다.



### TakeLab Retriever: AI-Driven Search Engine for Articles from Croatian News Outlets (https://arxiv.org/abs/2411.19718)
- **What's New**: TakeLab Retriever는 크로아티아 뉴스 매체의 기사를 검색하고 수집하며 의미 분석을 수행하는 AI 기반 검색 엔진입니다. 이는 연구자들이 일반 검색 엔진으로는 제공받기 어려운 트렌드와 패턴을 발견할 수 있는 중요한 도구가 됩니다. 이 보고서에서는 TakeLab Retriever의 사용법과 설계 과정, 그리고 소프트웨어 엔지니어링의 도전 과제를 자세히 설명합니다. 최첨단 자연어 처리(NLP) 기술을 통해 사용자는 웹 애플리케이션을 통해 주제별 기사를 신속하게 검색할 수 있습니다.

- **Technical Details**: TakeLab Retriever는 한국어와 같은 저자원 언어의 뉴스 콘텐츠 분석을 위해 설계된 특별한 의미 검색 엔진입니다. 이 엔진은 품사(POS) 태깅, 토큰화, 의존 구문 분석, 개체명 인식(NER), 다중 레이블 주제 모델 등의 NLP 모델을 통합하여 기사를 분석합니다. 이를 통해 사용자는 기사의 비편향 데이터에 접근하며, 통계 및 시각화를 통해 온라인 크로아티아 뉴스의 역사와 현황을 깊이 있게 탐구할 수 있습니다. 또한, 이 엔진은 연구 질문에 따라 정보를 정확하고 편향 없이 검색할 수 있게 해줍니다.

- **Performance Highlights**: TakeLab Retriever는 2022년 11월부터 공공에 제공되었으며 현재(2024년 11월 기준) 33개 크로아티아 뉴스 매체의 1천만 개 이상의 기사를 분석하고 있습니다. 사용자는 복잡한 쿼리를 통해 기사를 손쉽게 검색하고, 다양한 기준으로 필터링함으로써 질 높은 분석을 수행할 수 있습니다. 웹 애플리케이션은 사용자 친화적인 인터페이스를 제공하며, 기사의 메타데이터와 통계 정보를 그래프 형태로 시각화하여 연구자들이 트렌드를 탐색하고 분석하는 데 도움을 줍니다.



### Knowledge Management for Automobile Failure Analysis Using Graph RAG (https://arxiv.org/abs/2411.19539)
Comments:
          7 pages, 6 figures, to be published in 2024 IEEE International Conference on Bid Data (BigData)

- **What's New**: 이 연구는 자동차 고장 분석을 위한 지식 관리 시스템을 제안합니다. 이 시스템은 대규모 언어 모델(LLM)과 지식 그래프(KG)를 결합한 Retrieval-Augmented Generation (RAG) 방식에 기반하고 있습니다. 특히 그래프 RAG는 젊은 엔지니어들이 기존의 KG에서 효과적으로 정보를 추출하고 이해할 수 있도록 최적화되었습니다. 이를 통해 고장 원인을 신속하게 식별할 수 있는 시스템으로의 발전을 목표로 합니다.

- **Technical Details**: 이 논문은 자동차 고장 분석에 있어 기존의 지식 그래프와 함께 사용할 수 있는 새로운 그래프 RAG 시스템을 제안합니다. 이 시스템은 LLMs의 출력 성능을 향상시키기 위한 ROUGE F1 점수를 기반으로 한 실험을 통해 그 효과성을 입증하였습니다. 특히, 그래프 RAG는 KG에서 유용한 서브 그래프를 추출하고, 젊은 엔지니어들이 쉽게 이해할 수 있도록 하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 제안된 방법에 의해 생성된 문장은 기존의 방법보다 평균 157.6% 향상된 ROUGE F1 점수를 기록했습니다. 이는 자동차 고장 분석을 위한 지식 관리에서 제안된 시스템의 효과성을 강조합니다. 이러한 성과는 고장 분석 전문 지식을 젊은 엔지니어에게 효과적으로 전달할 수 있는 가능성을 보여줍니다.



### TQA-Bench: Evaluating LLMs for Multi-Table Question Answering with Scalable Context and Symbolic Extension (https://arxiv.org/abs/2411.19504)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)이 복잡한 멀티-테이블 관계 데이터에서 질문 응답(QA) 작업을 효율적으로 수행할 수 있는 능력을 평가하기 위해 TQA-Bench라는 새로운 멀티-테이블 QA 벤치를 제안합니다. 기존 벤치마크는 단일 테이블 QA에 주로 초점을 맞추어 LLM의 복잡한 멀티-테이블 QA 능력을 제대로 평가하지 못하고 있습니다. TQA-Bench는 실제 공개 데이터셋에서 수집된 다양한 관계형 데이터베이스 인스턴스를 포함하고 있으며, 8K에서 64K 토큰까지의 다양한 멀티-테이블 컨텍스트 길이를 생성하는 유연한 샘플링 메커니즘을 도입합니다.

- **Technical Details**: TQA-Bench는 데이터 수집, 관계형 데이터 샘플링, 평가 작업 정의 및 기호 확장을 통한 질문 생성의 네 가지 주요 단계로 체계적으로 구성됩니다. 이를 통해 여러 데이터 소스(예: WorldBank, DataGov)에서 다수의 대규모 관계형 데이터베이스를 활용해 다양한 멀티-테이블 QA 작업을 생성합니다. 또한, 8K에서 64K 토큰까지의 다양한 컨텍스트 길이를 갖춘 평가 작업을 제안하여 실제 데이터를 처리하는 LLM의 성능을 정밀하게 평가합니다.

- **Performance Highlights**: 논문에서 실시한 광범위한 실험을 통해 다양한 LLM의 멀티-테이블 QA 작업에서의 성능을 평가하였고, 그 결과 복잡한 데이터 환경에서의 LLM의 도전 과제와 기회를 강조하고 있습니다. TQA-Bench는 LLM의 데이터 검색 및 패턴 매칭을 넘어서는 추론 능력을 평가할 수 있도록 기호 확장 기능을 통합하여 신뢰성을 높였습니다. 이러한 평가 결과는 복잡한 데이터 관리 작업에서 LLM 기반 애플리케이션의 설계 및 실행에 대한 귀중한 통찰력을 제공합니다.



### Integration of Contextual Descriptors in Ontology Alignment for Enrichment of Semantic Correspondenc (https://arxiv.org/abs/2411.19113)
Comments:
          Ontology alignment, contextual descriptors, semantic matching, knowledge representation, essential descriptors, ontology integration, hierarchical structure, semantic heterogeneity, ethical AI

- **What's New**: 이 논문은 컨텍스트 기반의 설명자(contextual descriptors)를 이용한 새로운 의미적 온톨로지 정렬 방식(semantic ontology alignment)을 제안합니다. 필수 및 컨텍스추얼 설명자를 통합하여 포괄적인 지식 모델(knowledge model)을 구축할 수 있도록 하는 형식을 개발하였습니다. 이 과정에서 '투명성(Transparency)'과 '프라이버시(Privacy)'의 관계를 예시로 삼아 개념 간의 갈등(conflict)을 분석하기 위한 수학적 장치(mathematical apparatus)를 소개합니다.

- **Technical Details**: 제안된 방법은 계층적 구조(hierarchical structure)를 통해 의미적 접근 방식을 구현하며, 인공지능(AI) 맥락에서의 개념 분석에 중점을 두고 있습니다. 실험 연구는 컨텍스추얼 설명자를 적용한 후 온톨로지 정렬 지표(ontology alignment metrics)가 유의미하게 개선되었음을 확인했습니다. 특히 프라이버시, 책임(responsibility), 자유(freedom) 및 자율성(autonomy) 분야에서 평균적으로 약 4.36%의 전반적인 개선 효과를 나타냈습니다.

- **Performance Highlights**: 결과는 제안된 접근 방식이 지식의 복잡도(complexity)와 그에 대한 맥락적 의존성(contextual dependence)을 보다 정확하게 반영할 수 있음을 보여줍니다. 이는 인공지능의 다양한 윤리적 측면을 고려할 때 중요한 의미를 지니며, 향후 연구 및 적용 가능성에 대한 기대를 높이고 있습니다.



### ICLERB: In-Context Learning Embedding and Reranker Benchmark (https://arxiv.org/abs/2411.18947)
- **What's New**: 본 연구에서는 기존 검색 문제로 간주되던 정보 검색을 추천 문제로 재구성하여, In-Context Learning (ICL)에서 효율적인 문서 선택을 위한 새로운 접근 방식을 제시합니다. 이 연구의 주요 기여 중 하나는 재조정된 평가 방식인 In-Context Learning Embedding and Reranker Benchmark (ICLERB)를 도입하여, ICL 성능 향상에 미치는 문서의 효용을 평가합니다. 또한, AI 피드백을 활용한 Reinforcement Learning-to-Rank from AI Feedback (RLRAIF) 알고리즘을 제안하며, 이는 적은 피드백으로 검색 모델의 성능을 개선하는 데 중점을 두고 있습니다.

- **Technical Details**: ICL는 LLM이 모델의 매개변수를 업데이트 하지 않고도 관련 문서나 시연을 포함한 프롬프트에 의해 새로운 작업을 수행할 수 있게 합니다. Retrieval-Augmented Generation (RAG) 방식은 쿼리 기반으로 정보를 동적으로 검색하여 LLM의 입력에 포함시켜 성능을 증대시킵니다. 기존의 검색 방법들은 주로 의미론적 관련성에 중점을 두었으며, 이 연구에서는 최대 유용성을 위한 문서 선택에 중점을 두고 새로운 평가 지표와 ICLERB를 개발했습니다.

- **Performance Highlights**: 실험 결과, ICLERB와 기존 벤치마크 간에 뚜렷한 성능 차이를 보였으며, RLRAIF 알고리즘으로 미세 조정된 소형 모델이 대형 모델보다 뛰어난 성과를 나타냈습니다. 이를 통해 현재 평가 방법의 한계를 강조하고 ICL에 최적화된 새로운 벤치마크 및 훈련 전략의 필요성을 분명히 했습니다. rlraif 기반의 모델은 ICL 작업에 대한 검색 성능을 크게 향상시켰으며, 이는 효율적이고 경제적인 접근법임을 보여줍니다.



New uploads on arXiv(cs.CV)

### T2Vid: Translating Long Text into Multi-Image is the Catalyst for Video-LLMs (https://arxiv.org/abs/2411.19951)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 멀티모달 대용량 언어 모델(MLLMs)의 이미지 처리 성공을 바탕으로 비디오 이해로의 확장을 연구합니다. 특히, 제로샷 추론(zero-shot inference)과 추가 파인튜닝(fine-tuning) 접근 방식을 조사하여, 효과적인 데이터 증강(data augmentation) 방법을 제안합니다. 또한, 비디오 데이터 샘플을 활용한 학습 효율성을 높이기 위해 T2Vid라는 새로운 방법론을 개발했습니다.

- **Technical Details**: 연구에서는 비디오 이해를 위해 이미지-LLMs를 활용하는 두 가지 접근 방식을 심층적으로 분석합니다. 특히 제로샷 추론의 한계로 일반화 제한과 시간적 이해 부족을 지적하며, 이러한 문제를 해결하기 위해 파인튜닝 접근 방식을 조사하였습니다. 그 결과, 훈련 데이터의 다양성 부족이 낮은 학습 효율성을 초래함을 발견하고, 이를 해결하기 위한 비디오 유사 샘플 합성 기법을 제안합니다.

- **Performance Highlights**: 제안된 T2Vid 방법은 고작 15%의 데이터 샘플로도 완전한 비디오 데이터셋을 사용하는 것과 비슷하거나 더 나은 성능을 달성하는 것으로 나타났습니다. 이 방법은 긴 비디오 이해 및 비디오 샘플 훈련 없이도 성능을 향상시킬 수 있습니다. 연구 결과는 MLLMs를 비디오 이해에 활용하는 데 대한 새로운 가능성을 제시하고, 고품질 데이터의 수집에 대한 논의에 불을 지필 것으로 기대하고 있습니다.



### AlphaTablets: A Generic Plane Representation for 3D Planar Reconstruction from Monocular Videos (https://arxiv.org/abs/2411.19950)
Comments:
          NeurIPS 2024

- **What's New**: AlphaTablets는 연속적인 3D 표면과 정밀한 경계 표시를 특징으로 하는 새로운 일반적인 3D 평면 표현 방식입니다. 이 방식은 3D 평면을 알파 채널을 가진 직사각형으로 표현하여 현재의 2D 및 3D 평면 표현의 장점을 결합했습니다. 이를 통해 정확하고 일관된 유연한 3D 평면 모델링이 가능해지며, 모노큘러 비디오에서 3D 평면 재구성을 위한 새로운 하향식 파이프라인을 제안합니다.

- **Technical Details**: AlphaTablets는 2D 슈퍼픽셀과 사전 훈련된 모델의 기하학적 단서를 활용하여 초기화됩니다. 최적화를 통해 geometry, texture, alpha channels를 조정하며, 이를 위해 효과적인 병합 기법이 도입됩니다. 최종적으로 iterative optimization과 merging을 통해 고체 표면 및 명확한 경계를 가진 3D 평면 구조를 재구성합니다.

- **Performance Highlights**: ScanNet 데이터셋에서의 광범위한 실험을 통해 3D 평면 재구성에서 최첨단 성능을 입증했습니다. AlphaTablets는 다양한 응용 프로그램에 대한 일반적인 3D 평면 표현으로서의 큰 잠재력을 강조합니다. 또한, 유연한 평면 기반 장면 편집 기능을 제공합니다.



### DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation (https://arxiv.org/abs/2411.19946)
- **What's New**: 이번 연구에서는 데이터셋 디스틸레이션(dataset distillation) 방법의 다각화된 접근 방식인 Diversity-driven EarlyLate Training (DELT) 방안을 제안합니다. 이 방법은 배치-to-글로벌 매칭(batch-to-global matching)에서 이미지를 다채롭게 만들어 훈련 효율성을 높일 수 있도록 설계되었습니다. DELT는 정의된 IPC 샘플을 작은 하위 작업으로 나누고 각 하위 집합의 최적화를 통해 독특한 분포로 변환하여 일관된 최적화 과정으로 생기는 균일성을 줄입니다.

- **Technical Details**: DELT는 각 카테고리의 이미지를 서로 다른 초기점에서 최적화하여 완성된 결과가 상당히 다양하게 나오도록 하며, 이러한 접근 방식은 기존 방법들과 비교할 때 상당히 단순합니다. 이 방식은 Synthetic 이미지 초기화를 위한 Teacher-ranked 실제 이미지 패치를 사용하여 최적화의 부족을 방지하고, 최종 생성된 이미지의 다양성을 높입니다. 전통적인 최적화 방법과 비교할 때, DELT는 ImageNet-1K에서 훈련 과정의 계산량을 39.3%까지 줄일 수 있음을 보여줍니다.

- **Performance Highlights**: DELT는 CIFAR, Tiny-ImageNet, ImageNet-1K와 같은 다양한 데이터셋에서 이전의 최첨단 방법들보다 평균 2~5% 더 우수한 성능을 보였습니다. 특히 ImageNet-1K에서 IPC 50에서 66.1%의 정확도를 달성했으며, 이는 기존 RDED보다 4.9% 향상된 결과입니다. 작은 데이터셋인 CIFAR-10에서도 RDED 및 SRe2L보다 각각 2.5%와 19.2% 개선된 성과를 달성했습니다.



### Free-form Generation Enhances Challenging Clothed Human Modeling (https://arxiv.org/abs/2411.19942)
Comments:
          23 pages, 25 figures

- **What's New**: 이번 논문은 의복의 비대칭 변형 문제를 해결하기 위한 새로운 하이브리드 프레임워크를 제안합니다. 기존의 Linear Blend Skinning (LBS) 방법에 의존하는 것과 달리, 의복의 위치에 따라 다른 처리 전략을 사용하여 미세한 기하학적 디테일을 캡처합니다. 특히, 느슨한 의복 영역을 모델링하기 위해 자유로운 형태의 생성기를 도입하였습니다. 이는 긴 드레스와 스커트와 같은 어려운 의복 상황에서의 현실성을 크게 개선합니다.

- **Technical Details**: 제안하는 하이브리드 프레임워크는 인체를 비의복, 변형, 생성의 세 가지 카테고리로 구분한 후, 각 영역에 대해 적합한 방법을 적용합니다. 빈 공간이 많은 의복 영역은 생성기가 모델링을 담당하고, 몸에 가까운 부분은 LBS 기법을 활용하여 변형을 처리합니다. 자유로운 형태의 생성기는 움직임에 덜 민감한 느슨한 의복의 역학을 모사하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 느슨한 의복을 포함하는 벤치마크 데이터셋에서 최첨단 성능을 달성했으며, 특히 시각적 충실도와 현실감에서 우수한 성과를 보였습니다. 다른 최신 방법들과 비교하였을 때, 우리의 접근법은 느슨한 의복의 세밀한 디테일을 성공적으로 캡처하여 파손된 아티팩트를 최소화했습니다. 이는 고전적인 2D 위치 맵이나 주제 특정 의복 템플릿을 학습할 필요 없이 가능해진 것입니다.



### Perception Test 2024: Challenge Summary and a Novel Hour-Long VideoQA Benchmark (https://arxiv.org/abs/2411.19941)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2312.13090

- **What's New**: 2024년 IEEE/CVF 유럽 컴퓨터 비전 컨퍼런스에서 두 번째 Perception Test 챌린지가 개최되었습니다. 이번에는 7개의 트랙이 새롭게 추가되어, 비디오, 오디오 및 텍스트 모달리티에 걸쳐 다양한 다중 모달 작업을 다루었습니다. 특히, 올해는 1시간 길이의 비디오 이해를 위한 새로운 비디오 질문 응답 벤치마크인 1h-walk VQA가 도입되었습니다.

- **Technical Details**: Perception Test 벤치마크는 기억 및 직관 물리학 이해 능력 등을 측정하기 위해 특별히 설계된 11,600개의 실세계 비디오로 구성되어 있습니다. 이 벤치마크는 모델이 긴 시간 맥락을 이해하고 추론하는 능력을 평가하기 위해 Walking Tours 데이터셋을 활용하여 1h-walk VQA를 생성했습니다. 각 트랙은 검증 및 테스트 단계로 나뉘며, 참여자들은 평가 모드를 포괄적으로 제출해야 했습니다.

- **Performance Highlights**: 이번 챌린지에서는 123개 팀으로부터 680건의 제출물이 접수되었습니다. 참여 팀은 올해 도입된 총 상금 20,000 유로를 놓고 경쟁하였으며, 모든 트랙에서 상위 모델들의 성능이 지난해보다 개선되었습니다. 자세한 결과는 워크숍 웹사이트에서 확인할 수 있습니다.



### SIMS: Simulating Human-Scene Interactions with Real World Script Planning (https://arxiv.org/abs/2411.19921)
- **What's New**: 이 논문은 물리적으로 그럴듯한 장기 인간-장면 상호작용을 계획하고 제어하기 위한 새로운 프레임워크를 제시합니다. 특히, 인터넷에 존재하는 비디오 데이터를 활용하여 LLM 기반의 스크립트 추출 및 생성 과정을 통합합니다. 이를 통해 인과적이고 복잡한 시간 연속적인 인간 행동을 모사하며, 다양한 장면에 대한 이해를 통해 캐릭터의 동작을 유도할 수 있습니다.

- **Technical Details**: 프레임워크 SIMS(SIMultating huMan Scene interactions)는 고수준 계획에 LLM을, 저수준 제어에는 물리적 정책을 사용합니다. LLM을 통한 상호작용과 감정 변화를 실시간 비디오에서 추출하여 스크립트 데이터베이스를 구성하고, 이를 통해 키프레임을 생성합니다. 또한, CLIP 모델을 통해 장면 기하학과 텍스트 임베딩을 인식하여 고품질 동작 생성을 위한 듀얼 인식 정책을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 다양한 작업 수행에 있어 뛰어난 성능을 보이며 기존 방법들에 비해 일반화 능력이 향상되었습니다. 또한, 지능형 캐릭터들이 다양한 3D 환경에서 요구되는 다양한 동작을 실행하면서도 정교한 스타일을 갖출 수 있도록 합니다. 이 프레임워크의 코드는 곧 공개될 예정이며, 실제 응용 가능성이 높습니다.



### Quantifying the synthetic and real domain gap in aerial scene understanding (https://arxiv.org/abs/2411.19913)
Comments:
          17 pages (including references), 5 figures, 2 tables. Accepted for publication in the "Scientific Bulletin", Series C, Electrical Engineering and Computer Science, ISSN 2286-3540

- **What's New**: 이 논문은 합성 이미지와 실제 이미지 간의 격차를 정량화하는 새로운 방법론을 제시합니다. Multi-Model Consensus Metric (MMCM)와 depth 기반 구조 메트릭을 활용하여 장면 복잡도를 평가합니다. 이는 특히 항공 장면 이해와 같은 탐색되지 않은 분야에서 큰 영향을 미칠 수 있습니다.

- **Technical Details**: 실험 분석에서는 실제 데이터셋(Dronescapes)과 합성 데이터셋(Skyscenes)을 사용하여 모델 성능을 평가했습니다. 연구 결과, 실제 장면이 최신 비전 변환기(state-of-the-art vision transformers) 간의 높은 일관성을 보이는 반면, 합성 장면은 변동성이 크고 모델 적응을 도전하게 합니다. 이러한 차이는 구조적 및 인지적 불일치를 강조합니다.

- **Performance Highlights**: 결과는 복잡성과 도메인 간 격차의 본질적인 문제를 강조하며, 향상된 시뮬레이션 충실도와 모델 일반화의 필요성을 시사합니다. 이 연구는 도메인 특성과 모델 성능 간의 상호작용에 대한 중요한 통찰을 제공하며, 항공 장면 이해에 대한 개선된 도메인 적응 전략 개발을 위한 경로를 제시합니다.



### $C^{3}$-NeRF: Modeling Multiple Scenes via Conditional-cum-Continual Neural Radiance Fields (https://arxiv.org/abs/2411.19903)
- **What's New**: 본 연구에서는 여러 3D 장면을 단일 Neural Radiance Field (NeRF)에 통합할 수 있는 새로운 조건적 및 지속적 프레임워크인 $C^{3}$-NeRF를 제안합니다. 기존 접근 방식이 장면 조절을 위해 고급 기능 추출기와 사전 학습된 정보를 활용하는 반면, 우리는 간단한 유사 장면 레이블을 사용하여 NeRF에서 여러 장면을 모델링합니다. 이 프레임워크는 이전에 학습한 장면에 대한 최소한의 망각을 유지하면서 여러 장면에 적응할 수 있는 능력도 내재하고 있습니다.

- **Technical Details**: $C^{3}$-NeRF는 Instant-NGP의 해싱 메커니즘에 장면 조절을 통합하여 여러 장면을 놀라운 렌더링 품질로 모델링합니다. 이를 통해 동일한 매개변수 집합으로 새로운 장면을 효과적으로 학습하면서 이전에 경험한 장면의 렌더링 충실도를 유지할 수 있습니다. 또한, 이 프레임워크는 장면 데이터를 별도로 저장할 필요 없이 생성적 재생(generative replay) 패러다임 하에서 작동하여 새로운 장면을 수용합니다.

- **Performance Highlights**: C3-NeRF는 다중 장면 렌더링 품질, 학습 효율성 및 새로운 장면에 대한 적응력에서 기존 방법들을 지속적으로 능가합니다. 특히, 기존 NeRF도 $C^{3}$ 조건부-지속적 패러다임 하에서 여러 장면을 모델링할 수 있지만, 훈련 시간이 많이 걸리고 성능이 C3-NeRF보다 낮습니다. 이 연구는 NeRF의 잠재력을 극대화하며, 실제 애플리케이션에 더 적합한 더 나은 확장성과 응용 가능성을 제공합니다.



### GuardSplat: Efficient and Robust Watermarking for 3D Gaussian Splatting (https://arxiv.org/abs/2411.19895)
Comments:
          Project page: this https URL and Code: this https URL

- **What's New**: 이번 논문에서는 3D Gaussian Splatting (3DGS) 자산의 저작권을 효과적으로 보호하기 위한 새로운 프레임워크인 GuardSplat을 제안합니다. GuardSplat은 CLIP-guided Message Decoupling Optimization 모듈을 활용하여 메시지 디코더를 훈련하고, Spherical-harmonic-aware Message Embedding 모듈을 통해 3D 구조를 유지하면서 물질에 메시지를 매끄럽게 삽입합니다. 이러한 접근 방식은 최소한의 변질을 유지하며 기존 방법보다 향상된 효율성을 제공합니다.

- **Technical Details**: GuardSplat은 두 가지 주요 모듈로 구성되어 있습니다. 첫 번째는 CLIP의 정렬 기능을 활용하여 메시지 디코더의 훈련을 극대화하는 방식입니다. 두 번째는 3D Gaussian의 SH 특성 내에 메시지를 삽입하기 위한 Spherical-harmonic-aware 모듈로, 이를 통해 높은 추출 정확성 및 낮은 최적화 비용을 달성합니다. 이 과정에서 악의적인 사용자로부터 메시지를 안전하게 보호할 수 있습니다.

- **Performance Highlights**: 실험 결과, GuardSplat은 최신 방법들보다 월등한 성능을 보였습니다. 3DGS 자산을 훈련하고 워터마킹하는 데 단 5분에서 20분이 소요되며, 이는 현업에서의 적용 가능성을 더욱 높여줍니다. GuardSplat은 뛰어난 일반화 능력과 빠른 최적화 속도를 통해 저작권 보호의 새로운 기준을 제시합니다.



### FlowCLAS: Enhancing Normalizing Flow Via Contrastive Learning For Anomaly Segmentation (https://arxiv.org/abs/2411.19888)
- **What's New**: FlowCLAS는 새로운 자기 지도(self-supervised) 기반 인프레임워크로, 기존 기법의 한계를 극복하기 위한 여러 혁신 요소를 포함하고 있습니다. 이 모델은 다채로운 라벨에 의존하지 않고 이상 감지 문제를 해결하며, 우선적으로 Frozen Vision Foundation 모델을 이용하여 특성을 추출합니다. 또한, Normalizing Flow 네트워크를 적용하여 라벨이 없는 데이터셋을 효과적으로 활용할 수 있도록 개선되었습니다. FlowCLAS는 ALLO 벤치마크에서 SOTA 성능을 기록하면서, 자율주행을 위한 도로 이상 감지에서도 경쟁력 있는 결과를 보여주고 있습니다.

- **Technical Details**: FlowCLAS는 Outlier Exposure와 Contrastive Learning을 잠재 공간(latent space)에 통합하여 모델의 차별화력을 향상시킵니다. 기존의 비지도(unsupervised) 방법들과 달리, 우리의 접근법은 이상 패턴을 포함하는 의사(outlier) 객체를 혼합하여 일반 객체와 비정상 객체를 구분합니다. 이 과정을 통해 FlowCLAS는 클래스를 명시적으로 요구하지 않고도 다양한 도메인에서 장면 수준의 이상 감지를 수행하는 능력을 보여줍니다. 이는 공간 로봇 공학과 자율주행 분야에서 각각의 도메인의 고유한 도전을 효과적으로 해결하는 혁신적인 방법입니다.

- **Performance Highlights**: FlowCLAS는 ALLO 이상 감지 벤치마크에서 기존의 모든 방법보다 확연하게 우수한 성능을 달성하였으며, 도로 이상 감지 벤치마크인 Fishyscapes Lost&Found와 Road Anomaly 데이터셋에서도 뛰어난 성과를 보였습니다. 또한 Road Anomaly에서 SOTA AUPRC(Area Under the Precision-Recall Curve) 성능을 기록하며, 라벨이 필요 없는 방법으로 여러 메트릭에서 최근의 감독(supervised) 접근법과 동등한 성과를 보여주고 있습니다. 이는 FlowCLAS가 다양한 도메인에서 효과적으로 적용될 수 있음을 입증합니다.



### SpaRC: Sparse Radar-Camera Fusion for 3D Object Detection (https://arxiv.org/abs/2411.19860)
Comments:
          18 pages, 11 figures

- **What's New**: 본 연구에서는 SpaRC라는 새로운 Sparse fusion transformer를 제안하여 3D 인식을 위해 Radar와 Camera의 포인트 특징을 통합합니다. 기존 방법들이 Dense Bird's Eye View (BEV) 기반 구조를 활용하여 깊이 추정을 하는 동안, SpaRC는 인코딩된 포인트 특징으로 직접 작동하여 효율성과 정확성을 극대화했습니다. 이 방법은 false positive 탐지 및 위치 정밀도에서 기존의 한계를 극복하여 새로운 차원의 성능을 보여줍니다.

- **Technical Details**: SpaRC는 세 가지 주요 기여를 통해 개발되었습니다: (1) cross-modal feature alignment를 위한 sparse frustum fusion (SFF), (2) 정확한 객체 위치 지정을 위한 range-adaptive radar aggregation (RAR), (3) 집중된 쿼리 집계를 위한 local self-attention (LSA)입니다. 이러한 설계를 통해 SpaRC는 불필요한 BEV 그리드 렌더링의 복잡성을 줄이고 감지 및 인식 효율성을 높입니다. 특히 nuScenes와 TruckScenes 벤치마크에서 성능을 평가하였으며, 기존 방법들을 초월하는 성과를 기록했습니다.

- **Performance Highlights**: 우리의 방식은 nuScenes 벤치마크에서 67.1 NDS와 63.1 AMOTA라는 최첨단 성능 지표를 달성하며, TruckScenes에서도 우수한 일반성을 보여주고 있습니다. 이는 LiDAR 기반의 기준과 견줄 수 있는 성능을 갖추고 있어 자율주행 분야에서의 응용 가능성을 더욱 높입니다. SpaRC는 복잡한 동적 환경에서 안전한 자율 주행을 위한 실시간 인식을 실현하는 데 기여할 것으로 기대됩니다.



### A Visual-inertial Localization Algorithm using Opportunistic Visual Beacons and Dead-Reckoning for GNSS-Denied Large-scale Applications (https://arxiv.org/abs/2411.19845)
- **What's New**: 스마트 시티의 발전으로 인해 대규모 도시 환경에서 지속적인 보행자 내비게이션의 수요가 급증하고 있습니다. 본 연구에서는 Augmented Reality (AR)와 Low-cost visual-inertial positioning 방법을 통해 GNSS가 제한된 지역에서도 신뢰할 수 있는 위치 추정을 가능하게 하는 저비용의 시각적 위치 인식 방법을 제안합니다. 이 방법은 MSGC 기반의 VPR 신경망과 PDR 알고리즘, Kalman filter를 활용하여 오류를 수정합니다.

- **Technical Details**: 제안된 방법은 MSGC-Net이라는 경량 시각 위치 인식 신경망을 사용하여 PDR 방법(MDR-PDR)을 보완하여 GNSS가 없는 대규모 환경에서도 저비용의 실시간 위치 추정을 가능하게 합니다. VPR 과정에서는 사용자에 의해 입력된 이미지가 미리 수집된 오프라인 이미지 비콘 데이터베이스와 매칭되어 위치를 추정합니다. 구조적으로 MSGC-Net은 다단계 MSGC 블록과 풀링 레이어로 구성되어 있으며, 다양한 스케일 정보를 추출하기 위해 dilation convolution을 사용합니다.

- **Performance Highlights**: 실험 결과에 따르면 제안한 VPR 방법은 두 개의 공개 데이터셋에서 MobileNetV3 기반 VPR 방법에 비해 Recall@1을 최소 3% 향상시키면서 매개변수 수를 63.37% 줄였습니다. VPR-PDR 알고리즘은 원래의 PDR 방법에 비해 현저히 40% 이상의 위치 정확성을 개선했습니다. 이러한 성과는 GNSS가 제한된 환경에서도 안정적인 위치 추정을 가능하게 하여 스마트 시티의 발전에 기여할 것으로 기대됩니다.



### Feedback-driven object detection and iterative model improvemen (https://arxiv.org/abs/2411.19835)
Comments:
          AI4EA24 preprint

- **What's New**: 이 논문에서는 효율적이고 고품질의 객체 탐지(annotation) 모델을 개선하기 위한 플랫폼을 개발하고 평가한 결과를 소개합니다. 사용자는 이미지를 업로드하고 주석을 추가하며, 이를 통해 객체 탐지 모델을 미세 조정할 수 있습니다. 이 플랫폼은 반자동 주석(semi-automatic annotation) 방식을 통해 최대 53%의 주석 시간 절약을 달성하였으며, 주석 품질을 유지하면서도 효율성을 크게 향상시키는 가능성을 보여줍니다.

- **Technical Details**: 플랫폼은 사용자가 주석 프로젝트를 생성하고 이미지 번들을 업로드한 후, 사전 훈련된 Single Shot Detector (SSD)를 사용하여 초기 객체 탐지를 수행할 수 있는 기능을 제공합니다. 사용자는 모델이 생성한 레이블을 수동으로 조정하여 모델의 성능을 점진적으로 개선하고, 이를 통해 향후 예측을 위한 스냅샷을 생성하는 주석 프로세스를 구현합니다. 플랫폼은 사용자 친화적인 인터페이스를 제공하여 대규모 이미지 데이터셋의 효율적인 주석을 지원합니다.

- **Performance Highlights**: 실험 결과, 반자동 주석 프로세스는 수동 레이블링에 비해 주석 시간을 최대 53.82% 줄이는 데 성공했습니다. 사용자와의 상호작용 노력 역시 현저히 감소하였으며, 반자동 주석의 F1 점수는 수동 주석의 품질과 일치하거나 이를 초과하는 결과를 보였습니다. 논문의 이 결과는 논의된 플랫폼이 고품질 객체 탐지 데이터셋을 생성할 수 있는 잠재력을 확인시켜 줍니다.



### SAT-HMR: Real-Time Multi-Person 3D Mesh Estimation via Scale-Adaptive Tokens (https://arxiv.org/abs/2411.19824)
Comments:
          16 pages, 12 figures

- **What's New**: 본 논문에서는 단일 RGB 이미지로부터 실시간 다인체 3D 인간 메쉬 추정을 위한 일단계 프레임워크를 제안합니다. 기존의 일단계 방법들은 고해상도 입력으로 최첨단(State-of-the-Art, SOTA) 성능을 달성하지만, 이미지의 작은 스케일에서 개인을 추정하는 데 주로 이점이 있으며, 이로 인해 계산 비용이 크게 증가합니다. 이에 따라 본 연구에서는 각 개인의 상대적 스케일에 따라 동적으로 조정되는 스케일 적응 토큰(scale-adaptive tokens)을 도입하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: 제안된 스케일 적응 토큰은 이미지 내에서 개별 개인의 스케일에 기반하여 처리 방식을 조정합니다. 작은 스케일의 개인은 높은 해상도로 처리되고, 큰 스케일의 개인은 낮은 해상도로 처리되며, 배경 영역은 추가적으로 증류되어 처리 비용을 줄입니다. 이러한 스케일 적응 토큰은 이미지 특징을 보다 효율적으로 인코딩하여 인간 메쉬의 회귀를 위한 후속 디코딩을 용이하게 합니다.

- **Performance Highlights**: 실험 결과, 본 방법은 높은 해상도 처리의 정확성 이점은 유지하면서도 계산 비용을 크게 줄여 실시간 추론이 가능함을 보여줍니다. 최종적으로, 본 방법은 24 FPS의 속도로 동작하며 다인체 3D 메쉬 추정에서 최고의 실시간 모델로서 SOTA 성능의 최대 5배 속도를 달성합니다.



### Gaussian multi-target filtering with target dynamics driven by a stochastic differential equation (https://arxiv.org/abs/2411.19814)
- **What's New**: 본 논문에서는 연속 시간에서의 타겟 동역학을 고려하는 다중 타겟 필터링 알고리즘을 제안합니다. Poisson point process (PPP)를 기반으로 한 타겟 출현 모델과 일반적인 선형 확률 미분 방정식에 따른 타겟 동작 모델 등이 포함되어 있습니다. 본 연구는 Kullback-Leibler divergence를 최소화하여 각 타겟의 평균 및 공분산을 계산하는 닫힌형 표현식을 도출합니다.

- **Technical Details**: 다중 타겟 필터링은 과거 측정된 시퀀스를 기반으로 현재 타겟 집합을 평가하는 것입니다. 논문은 연속 시간에서 다중 타겟 동적 모델이 어떻게 형성되는지 설명하고, 타겟의 출현 및 사라짐 모델이 포함된 연속-이산 다중 타겟 필터링 알고리즘을 제시합니다. 또한 비선형 확률 미분 방정식에 의한 타겟 동역학의 확장도 다룹니다.

- **Performance Highlights**: 본 연구 결과는 Gaussian continuous-discrete Poisson multi-Bernoulli mixture (PMBM) 필터의 성공적인 개발을 시연합니다. Kullback-Leibler divergence를 최소화하여 도출된 최적의 가우시안 근사치 및 이를 통해 얻어진 제안된 알고리즘은 다양한 응용 분야에서 뛰어난 성능을 보여줄 것으로 기대됩니다. 나아가, 비선형 SDE에 의해 구동되는 타겟 동역학에서의 필터 확장 기술도 제시됩니다.



### MoTe: Learning Motion-Text Diffusion Model for Multiple Generation Tasks (https://arxiv.org/abs/2411.19786)
Comments:
          Five figures, six tables

- **What's New**: 최근 대화형 AI 모델과 Denoising Diffusion Model의 발전에 힘입어, 인간의 동작 분석이 크게 향상되었습니다. 본 논문에서는 텍스트와 모션의 주변, 조건 및 결합 분포를 학습하여 다양한 작업을 가능하게 하는 통합 다중 모드 모델인 MoTe를 소개합니다. MoTe는 텍스트-모션 생성, 모션 캡셔닝, 텍스트 주도 모션 생성을 지원하며, 간단한 입력 맥락 수정만으로 다양한 작업을 수행할 수 있습니다.

- **Technical Details**: MoTe는 Motion Encoder-Decoder (MED), Text Encoder-Decoder (TED), Motion-Text Diffusion Model (MTDM)이라는 세 가지 주요 구성 요소로 구성되어 있습니다. MED와 TED는 각각 모션 시퀀스와 텍스트 설명을 재구성하기 위해 잠재 임베딩(latent embeddings)을 추출하도록 훈련됩니다. MTDM은 다양한 작업을 처리하기 위해 입력 맥락에서 반복적인 denoising 과정을 수행합니다.

- **Performance Highlights**: 벤치마크 데이터셋에서 MoTe는 텍스트-모션 생성 작업에서 우수한 성능을 보여주었고, 모션 캡셔닝 작업에서도 경쟁력 있는 결과를 나타냈습니다. MoTe는 다양한 작업을 간단한 입력 맥락 쇄신으로 수행 가능하며, 실험 결과는 모션 임베딩이 텍스트-모션 및 모션-텍스트 작업 성능에 미치는 영향을 제시합니다.



### PerLA: Perceptive 3D Language Assistan (https://arxiv.org/abs/2411.19774)
- **What's New**: 이 논문에서는 3D 언어 도우미(3D Language Assistant, 3DLA)인 PerLA를 소개합니다. PerLA는 다양한 포인트 클라우드 영역에서 고해상도(local) 세부정보를 병렬로 캡처하고, 전체 포인트 클라우드로부터 얻은 하의 해상도(global) 맥락과 통합하여 LLM을 위한 보다 유용한 시각적 표현을 생성합니다. 이를 통해 기존 방법들보다 더 많은 세부정보를 보존하면서도 교육 안정성을 증진시키기 위한 새로운 손실 함수도 제안하였습니다.

- **Technical Details**: PerLA는 포인트 클라우드 처리를 위해 힐버트 곡선(Hilbert curve)을 기반으로 하는 알고리즘을 사용하여 고유한 정보를 효과적으로 집계합니다. 이 알고리즘은 지역(local) 및 글로벌(global) 정보를 결합하기 위해 교차 주의(cross-attention) 및 그래프 신경망(Graph Neural Network, GNN)을 활용하여 고차원적인 포인트 수준의 표현을 생성합니다. 이러한 접근법을 통해 PerLA는 3D QA 및 3D 조밀 캡셔닝의 벤치마크인 ScanQA와 ScanRefer, Nr3D에서 높은 성능을 보여줍니다.

- **Performance Highlights**: PerLA는 기존 3D 언어 도우미에 비해 눈에 띄는 성능 향상을 기록하였습니다. 특히 ScanQA에서 질문 응답 시 최대 +1.34 CiDEr의 이점을 보였고, ScanRefer와 Nr3D에서는 각각 +4.22 및 +3.88을 기록하여 조밀한 캡셔닝에서도 높은 효과를 나타냈습니다. 이러한 결과들은 PerLA의 효율적인 정보 집계와 세부 정보 보존 전략이 성공적이었음을 보여줍니다.



### LongVALE: Vision-Audio-Language-Event Benchmark Towards Time-Aware Omni-Modal Perception of Long Videos (https://arxiv.org/abs/2411.19772)
Comments:
          18 pages, 15 figures

- **What's New**: 이 연구에서는 LongVALE라는 혁신적인 비디오 이해 벤치마크를 제안합니다. 이 벤치마크는 105K개의 오미-모달(omni-modal) 이벤트와 정밀한 시간 경계 및 세밀한 관계 인식을 포함하는 캡션을 제공합니다. 기존 연구들이 비주얼 데이터에 국한되거나 품질이 낮은 데이터셋에 머물러 있는 반면, LongVALE는 8.4K개의 고품질 긴 비디오를 바탕으로 진행되었습니다.

- **Technical Details**: LongVALE는 세 가지 주요 구성 요소로 이루어진 자동화된 주석 파이프라인을 통해 생성됩니다. 이러한 구성 요소는 고품질 비디오 필터링, 세멘틱적으로 일관된 오미-모달 이벤트 경계 탐지, 그리고 크로스-모달 상관관계를 인식하는 이벤트 캡셔닝입니다. 이 과정을 통해 오미-모달 이벤트에서 세밀한 캡션을 생성하고, 동영상 내의 시각 및 청각 정보를 통합적으로 처리하는 방법을 제시합니다.

- **Performance Highlights**: LongVALE 기반 모델은 기존 비디오 대형 언어 모델(LLMs)보다 모든 오미-모달 작업에서 뛰어난 성능을 나타냈습니다. 또한 우리의 모델은 적은 데이터로도 제로샷(Zero-shot) 일반 오디오-비주얼 질문 답변(AVQA) 작업에서 우수한 결과를 기록하였습니다. 이러한 결과들은 LongVALE 데이터셋이 포괄적인 다중 모달 비디오 이해 발전에 기여할 수 있는 잠재력을 보여줍니다.



### LaVIDE: A Language-Vision Discriminator for Detecting Changes in Satellite Image with Map References (https://arxiv.org/abs/2411.19758)
- **What's New**: 이 논문에서는 단일 이미지만으로 변화를 감지하는 기존의 한계를 극복하기 위해, 지도를 참조하여 위성 이미지의 변화를 감지하는 방법인 LaVIDE(Language-VIsion Discriminator)를 제안합니다. LaVIDE는 텍스트로 지도를 표현하여 고수준 카테고리 정보와 저수준 시각 정보를 비교합니다. 이 방식은 이미지와 지도의 정보 간의 격차를 줄이고, 보다 효과적인 변화 감지를 가능하게 합니다.

- **Technical Details**: LaVIDE는 두 개의 병렬 브랜치로 구성되어 있습니다. 지도 브랜치에서는 지도 정보를 텍스트로 변환하여 텍스트 임베딩을 추출하며, 객체 속성을 포함한 맥락 최적화 전략을 사용해 카테고리 정보를 강화합니다. 이미지 브랜치에서는 위상 모델의 피쳐 공간과 시맨틱 얼라인먼트를 보장하며, MoE(Mixture-of-Experts) 모듈을 통해 텍스트 임베딩과 비전 임베딩을 비교하여 변화 감지를 수행합니다.

- **Performance Highlights**: 제안된 방법은 네 개의 벤치마크 데이터셋에서 검증되었으며, 기존의 최첨단 변화 감지 알고리즘들을 초과하는 성능을 기록했습니다. 예를 들어, DynamicEarthNet 데이터셋에서 약 13.8%의 성능 향상을 보였으며, SECOND 데이터셋에서는 4.3%의 향상 효과를 얻었습니다. 이러한 결과는 LaVIDE가 위성 이미지와 지도 참조를 효과적으로 통합하여 변화 감지의 정확도를 높임을 보여줍니다.



### DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering (https://arxiv.org/abs/2411.19756)
- **What's New**: 새로운 방법인 DeSplat을 통해 Gaussian primitives의 볼륨 렌더링(volume rendering)을 기반으로 잡음 요소(distractors)와 정적(scene elements) 장면을 직접 분리할 수 있게 되었습니다. 기존 방법들이 사전 훈련된 모델에서 외부 의미 정보를 활용하는 데 반해, DeSplat은 이러한 추가적인 계산 부담을 줄이는 데 유리합니다.

- **Technical Details**: DeSplat은 각 카메라 뷰(camera view) 내에서 Gaussians를 초기화하여 뷰 특정 잡음 요소(distractors)를 재구성합니다. 이를 통해 정적 3D 장면과 잡음 요소를 알파 합성(alpha compositing) 단계에서 별도로 모델링할 수 있습니다. 이 과정은 멀티 뷰 일관성(multi-view consistency) 가정을 유지하면서 이루어집니다.

- **Performance Highlights**: DeSplat은 정적 요소와 잡음 요소를 명확하게 분리함으로써 렌더링 속도를 희생하지 않으면서도 이전의 잡음 없는 방법(distractor-free approaches)과 유사한 결과를 달성합니다. 세 가지 벤치마크 데이터 세트에서 DeSplat의 효과를 입증하였으며, 이는 새로운 뷰 합성에서의 유용성을 강조합니다.



### A Multi-Loss Strategy for Vehicle Trajectory Prediction: Combining Off-Road, Diversity, and Directional Consistency Losses (https://arxiv.org/abs/2411.19747)
Comments:
          Preprint, 7 pages, 4 figures and 2 tables

- **What's New**: 이번 연구에서는 자율주행 차량의 궤적 예측 문제의 주요 한계를 해결하기 위해 세 가지 새로운 손실 함수인 Offroad Loss, Direction Consistency Error, Diversity Loss를 도입했습니다. 이러한 손실 함수는 예측 경로를 주행 가능한 한계 내에 두고, 교통 방향에 일치하도록 하며, 다양한 주행 시나리오를 커버하도록 설계되었습니다. 기존의 'winner takes all' 학습 방식의 단점을 극복하여 모든 예측 모드에서 기능을 적용, 향상된 예측 성능을 도모합니다.

- **Technical Details**: 제안된 Offroad Loss는 궤적이 주행 가능한 경계 내에 위치하도록 보장하며, 경계를 넘는 경우 페널티를 점진적으로 증가시킵니다. Direction Consistency Error는 예측된 궤적을 예상되는 교통 흐름과 도로 레이아웃에 맞춰 정렬합니다. Diversity Loss는 다양한 예측 결과를 생성하도록 유도하며, 가능성 있는 모든 결과를 커버하는 예측을 장려합니다. 이러한 손실 함수는 모든 예측 모드에 걸쳐 중복 없이 적용되며, 전반적인 예측 품질을 향상시킵니다.

- **Performance Highlights**: nuScenes 및 Argoverse 2 데이터셋에서 선두 기준 모델들과 함께 진행된 광범위한 검증 결과, 제안한 손실 함수가 궤적 예측 모델의 안전성과 강건성을 크게 향상시키는 것으로 나타났습니다. 특히, 궤적 예측의 정확도를 유지하면서도 원본 데이터셋에서 오프로드 오류를 47% 줄였고, 공격 장면에서는 37%의 감소를 기록했습니다. 이러한 결과는 복잡한 환경에서의 자율주행 궤적 예측에 대한 새로운 벤치마크를 세우는 성과를 의미합니다.



### Real-Time Anomaly Detection in Video Streams (https://arxiv.org/abs/2411.19731)
- **What's New**: 이번 연구는 Othello와 LIASD 실험실 간의 CIFRE 협약의 일환으로 진행되며, 실시간 비디오 스트림에서 위험 요소를 탐지할 수 있는 인공지능 시스템 개발을 목표로 하고 있습니다. 이를 위해, 새로운 방법론인 시공간 분석(temporal and spatial analysis)을 결합한 접근법이 제안되었습니다. 객체 탐지(object detection), 인간 자세 탐지(human pose detection), 그리고 motion analysis를 통합하여 이상 탐지(anomaly detection)를 향상시키기 위한 여러 전략이 모색되었습니다.

- **Technical Details**: 제안된 아키텍처는 경고(alert) 또는 원인(cause)을 식별해야 하는지에 따라 이진 및 다중 클래스 분류(binary or multiclass classification)를 수행합니다. YOLO를 사용한 공간 분석(spatial analysis)과 VGG19 및 GRU로 구성된 Convolutional Recurrent Neural Network(CRNN)를 활용한 시간 분석(temporal analysis), 그리고 분류를 위한 다층 퍼셉트론(multi-layer perceptron) модели로 다양한 신경망 모델이 실험되었습니다. 각각의 모델은 데이터의 다양한 유형을 처리하며 병렬(parallel) 또는 직렬(serial) 방식으로 조합할 수 있습니다.

- **Performance Highlights**: 이 연구는 두 가지 독창적인 데이터 세트를 생성하여 감독 학습(supervised learning)을 통해 모델들을 훈련시켰습니다. 첫 번째 데이터 세트는 이상을 유발할 수 있는 객체들에 중점을 두고, 두 번째 데이터 세트는 비디오를 통한 이상 또는 비정상 상태를 포함하고 있습니다. 이러한 방법론은 비디오 스트림과 유한 비디오의 처리에서 더 큰 유연성과 정확성을 제공합니다.



### MonoPP: Metric-Scaled Self-Supervised Monocular Depth Estimation by Planar-Parallax Geometry in Automotive Applications (https://arxiv.org/abs/2411.19717)
Comments:
          Accepted at WACV 25, project page: this https URL

- **What's New**: 이번 연구에서는 자율주행차량에서 널리 사용되는 자가 지도화(셀프 수퍼바이즈드) 모노큘러 깊이 추정 기술을 개선하여, 단순히 하나의 비디오 데이터와 카메라 장착 위치만으로 메트릭 스케일 깊이 예측을 가능하게 하는 새로운 모델을 소개합니다. 본 연구는 플래너-패럴랙스 기하학(planar-parallax geometry)을 활용하여 장면 구조를 재구성하며, 이는 최신 자동차에서 쉽게 접근할 수 있는 정보입니다.

- **Technical Details**: 제안된 시스템은 다중 프레임 네트워크, 단일 프레임 네트워크 및 포즈 네트워크의 세 가지 주요 네트워크로 구성되어 있습니다. 이 중 다중 프레임 네트워크는 순차적인 프레임을 처리하며, 카메라 장착 위치와 플래너-패럴랙스 기하학을 활용하여 정적인 장면의 구조를 추정합니다. 이는 나중에 다른 네트워크에 지식을 전달하여, 스케일 정보와 동적 물체 마스크 등을 제공합니다.

- **Performance Highlights**: 이 모델은 드라이빙 벤치마크인 KITTI에서 최첨단 결과를 달성했으며, Cityscapes 데이터세트에서도 자가 지도화된 메트릭 스케일 깊이 예측을 시연한 첫 번째 방법 중 하나로 인정받고 있습니다. 이번 연구는 카메라의 장착 위치 정보만으로 신뢰할 수 있는 메트릭 깊이 결과를 생성할 수 있음을 입증하여, 자동차 인식 작업에서 높은 적용 가능성을 보여줍니다.



### Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection (https://arxiv.org/abs/2411.19715)
- **What's New**: 이 논문에서는 CLIP을 효과적이고 일반화된 얼굴 위조 탐지기로 변환하기 위해 설계된 Forensics Adapter라는 어댑터 네트워크를 소개합니다. 기존 방법들은 CLIP을 단순한 피쳐 추출기로 다루어, 작업 특정 적응이 부족하여 효율성이 제한됩니다. 이를 해결하기 위해 본 연구에서는 얼굴 위조 추적을 배우고, 이를 통해 CLIP의 시각 토큰을 강화하는 새로운 상호작용 전략을 제안합니다.

- **Technical Details**: Forensics Adapter는 특정 작업 목표에 의해 유도된 작업 특정 목표를 학습하여 가벼운 어댑터 네트워크를 CLIP과 함께 사용할 수 있도록 구성되었습니다. 이 어댑터는 얼굴 위조의 고유한 혼합 경계를 캡처하기 위한 설계로, 5.7M의 학습 가능한 파라미터만으로 다섯 개의 표준 데이터 세트에서 평균 약 7% 성능 향상을 기록했습니다. 어댑터와 CLIP 간의 상호작용 전략도 제안하여 두 컴포넌트 간의 지식 전달을 최적화했습니다.

- **Performance Highlights**: 실험 결과, 본 방법은 다섯 개의 공개 데이터 세트에서 기존 최첨단 방법들에 비해 상당한 성능 향상을 보였으며, 이는 새로운 연구의 강력한 기반이 될 것입니다. 얼굴 위조 탐지에서 CLIP을 위한 최적화된 과정을 통해, 실질적인 적용 가능성이 높아졌습니다. 본 연구는 얼굴 위조 탐지 분야에서 CLIP 기반 방법들의 기초로 자리잡을 수 있음을 보여줍니다.



### Explaining the Impact of Training on Vision Models via Activation Clustering (https://arxiv.org/abs/2411.19700)
- **What's New**: 이 논문은 Neuro-Activated Vision Explanations (NAVE)라는 새로운 방법을 제안합니다. NAVE는 시각 모델의 feature encoder에서 추출된 정보를 클러스터링하여 분석합니다. 이 방법은 모델의 예측을 설명하는 것이 아니라, 이미지의 어떤 부분이 유사하게 처리되는지를 보여줍니다.

- **Technical Details**: NAVE는 frozen network의 feature activations를 클러스터링하여 세그멘테이션 형태로 출력합니다. NAVE는 비지도 학습(unconditional learning) 방법으로써, 네트워크의 재조정 없이 입력 이미지를 분석합니다. 이 논문은 NAVE가 ViT(vision transformers)와 Clever Hans 효과에 의해 정보 포화 상태에 미치는 영향을 분석합니다.

- **Performance Highlights**: NAVE를 통해 교육 데이터셋과 감독 수준이 모델이 캡처하는 개념에 미치는 영향을 분석했습니다. 실험 결과, 최근의 self-supervised learning 기법이 supervised learning의 격차를 줄이는 데 효과적임을 확인했습니다. 또한, NAVE의 분석을 통해 augmentation과 Clever Hans 효과가 모델의 입력 해석 능력에 미치는 영향을 보여주었습니다.



### Gated-Attention Feature-Fusion Based Framework for Poverty Prediction (https://arxiv.org/abs/2411.19690)
Comments:
          The paper has accepted for publication at 5th International Conference on Data Engineering and Communication Technology (ICDECT)

- **What's New**: 이 연구는 개발 지역에서의 가난 수준을 정확하게 추정하는 도전 과제를 다룬다. 전통적인 조사 방법들은 비용이 많이 들고 오래되지 않아 실효성이 떨어지지만, 본 연구에서는 Gated-Attention Feature-Fusion Module (GAFM) 을 활용한 최첨단 Convolutional Neural Network (CNN) 아키텍처를 제안한다. 이 방법은 위성 이미지의 글로벌 및 로컬 특징을 결합하여 더 정확한 가난 추정치를 달성하는데 기여한다. 모델은 75%의 R² 점수를 기록하며, 기존의 기법들과 비교해 유의미한 성과를 보인다.

- **Technical Details**: 본 연구의 프레임워크는 CNN을 이용하여 낮 시간대의 위성 이미지에서 밤 시간대 조도의 예측을 학습하도록 설계되었다. GAFM 모듈은 글로벌 특징과 로컬 세부 정보를 효과적으로 융합하여 정보가 풍부한 표현을 확보하는데 중점을 두었다. 이러한 융합 프로세스는 밤 조도 예측의 정확도를 크게 향상시켰으며, 이는 기초 소득 예측 작업에 중요하다. 본 연구에서는 Google Earth에서 촬영한 고해상도 위성 이미지와 NASA Earth Observatory의 밤 조도 데이터를 활용한다.

- **Performance Highlights**: 본 연구에서는 제안된 GAFM 기반 프레임워크가 빈곤 추정 작업의 정확성을 좌우하는 핵심 요소임을 보여준다. GAFM 구조의 채택으로 모델은 관련 특징을 더욱 잘 학습하고 처리하게 되어, 밤 조도 예측의 정확도가 크게 향상되었다. 이로 인해 정부 및 자선 단체들이 효과적으로 자원을 관리할 수 있도록 지원하는 강력한 도구로 자리매김하게 될 것이다. 본 연구의 발전은 위성 이미지를 활용한 빈곤 추정 분야에 긍정적인 임팩트를 미칠 것으로 기대된다.



### SURE-VQA: Systematic Understanding of Robustness Evaluation in Medical VQA Tasks (https://arxiv.org/abs/2411.19688)
- **What's New**: 이번 논문은 Vision-Language Models (VLMs) 의 강인성(robustness)을 평가하는 새로운 프레임워크인 SURE-VQA를 소개합니다. SURE-VQA는 기존의 평가 방식이 놓치는 점들을 보완하여, 실제 데이터에서 발생하는 자연스러운 배포 변화(distribution shifts)에 대한 강인성을 분석하는 데 초점을 맞추고 있습니다. 이 프레임워크는 LLMs(large language models)를 활용하여 의미 분석을 강화하고, 의미 있는 기준(sanity baselines)을 정의하여 VLM의 성능을 보다 명확하게 측정할 수 있는 방안을 제공합니다.

- **Technical Details**: SURE-VQA는 세 가지 주요 요구사항을 중심으로 구성됩니다. 첫 번째로, 데이터의 자연스러운 변경(real world shifts)에 대한 강인성을 평가해야 하며, 두 번째로 전통적인 토큰 매칭(token matching) 지표를 넘어서 LLM을 활용해야 한다는 점입니다. 마지막으로, 의미 있는 기준을 포함함으로써 모델의 성능 이해를 돕고 있습니다. 이 프레임워크는 세 가지 의료 데이터셋을 활용하여 다양한 배포 변화에 대응하는 VLM의 강인성을 분석하는 연구에 적용되었습니다.

- **Performance Highlights**: 이 연구에서는 다양한 미세 조정(fine-tuning) 방법의 강인성을 비교하여 몇 가지 중요한 발견을 제시합니다. 특히, 데이터 없이도 잘 작동하는 기준이 있다는 것과 LoRA(method)가 가장 우수한 PEFT(parameter-efficient fine-tuning) 방법임을 확인했습니다. 그러나 모든 PEFT 방법이 배포 변화에 대한 강인성에서 다른 방법보다 일관되게 우수하지 않다는 결과도 나타났습니다. 연구 결과는 VLM 연구 커뮤니티에 중요한 통찰력을 제공하며, 코드도 공개되어 있으니 참고할 수 있습니다.



### TexGaussian: Generating High-quality PBR Material via Octree-based 3D Gaussian Splatting (https://arxiv.org/abs/2411.19654)
Comments:
          Technical Report

- **What's New**: TexGaussian은 최신 기술인 octant-aligned 3D Gaussian Splatting을 기반으로 하여 빠르고 고품질의 PBR(material) 생성을 가능하게 합니다. 이 방법은 입력 3D 메쉬의 최하위 노드에 각 3D Gaussian을 배치함으로써 albedo 맵뿐만 아니라 roughness 및 metallic에 대한 다중 뷰 이미지를 렌더링합니다. 기존의 2D diffusion 모델에 의존하지 않으며, 단일 피드 포워드 과정으로 PBR 재질을 생성할 수 있도록 회귀 방식으로 훈련됩니다.

- **Technical Details**: TexGaussian은 3D 공간에서 직접 PBR 재질을 생성하고 3D 점 구름의 비선형성으로 인한 블러 현상을 해결하는 3D Gaussian Splatting(3DGS)을 사용합니다. 입력 메쉬의 표면에서 밀집한 3D 점 구름을 샘플링하여 octree를 구축하고, 각 옥탄에 3D Gaussian을 배치합니다. 이를 통해 다중 뷰 이미지를 효율적으로 예측하며, 3D U-Net을 사용하여 매개변수를 예측합니다.

- **Performance Highlights**: 공개된 벤치마크에 대한 광범위한 실험 결과, TexGaussian은 기존 방식보다 시각적으로 더 우수한 PBR 재질을 생성하며, 조건부 및 비조건부 시나리오 모두에서 더 빠른 실행 속도를 보입니다. 또한 주어진 기하학과의 일관성을 개선하여 최종 렌더링 품질을 높입니다.



### Uniform Attention Maps: Boosting Image Fidelity in Reconstruction and Editing (https://arxiv.org/abs/2411.19652)
Comments:
          Accepted to WACV 2025

- **What's New**: 이번 연구에서는 기존의 tuning-free 방법들이 지니는 한계점을 분석하고, 이를 극복하기 위한 새로운 접근법을 제안합니다. 특히, cross-attention 메커니즘의 문제를 해결함으로써 이미지 재구성의 신뢰도를 크게 향상시키는 uniform attention maps을 도입했습니다. 이는 텍스트 조건 하에서도 이미지 편집의 일관성과 정확도를 증가시킵니다.

- **Technical Details**: 기존의 DDIM Inversion에서 발생하는 reconstruction error는 U-Net 내의 cross-attention 메커니즘에서 비롯되는 비정렬성과 관련이 있습니다. 이를 해결하기 위해 우리는 구조적 관점에서 재구성을 분석하고, 전통적인 cross-attention을 uniform attention maps으로 대체하는 새로운 방법을 제안합니다. 이 방법은 다양한 텍스트 조건에 의해 발생하는 왜곡을 최소화하며, 영상 편집 알고리즘과 통합된 adaptive mask-guided editing 기술을 추가하여 성능을 더욱 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 고신뢰도 이미지 재구성에서 뛰어난 성능을 나타내며 실재 이미지 조합 및 편집 시나리오에서도 강건한 결과를 보였습니다. 기존 방법 대비 높은 픽셀 정확도와 모든 유형의 텍스트 조건에서의 신뢰성을 입증하였습니다. 이 연구는 uniform attention maps이 diffusion 기반 이미지 처리 방법의 신뢰도와 다재다능성을 높이는 잠재력을 지니고 있음을 강조합니다.



### Accelerating Multimodal Large Language Models via Dynamic Visual-Token Exit and the Empirical Findings (https://arxiv.org/abs/2411.19628)
- **What's New**: 이 연구에서는 기존의 Multimodal Large Language Models (MLLMs)에서 발생하는 시각적 토큰의 과도한 사용이 계산 비용을 증가시키고 불필요한 중복을 야기함을 지적하고 있습니다. 저자들은 MLLMs의 주의(attention) 행동을 조사하여 시각적 중복 문제를 해결하기 위한 간단하지만 효과적인 방법인 '동적 시각적 토큰 종료(Dynamic Visual-Token Exit, DyVTE)'를 제안하였습니다. DyVTE는 경량의 하이퍼 네트워크(hyper-networks)를 사용하여 텍스트 토큰의 상태를 인식하고 특정 계층 이후에 모든 시각적 토큰을 제거합니다.

- **Technical Details**: MLLM의 추론(inference) 과정은 세 가지 주요 단계로 나뉩니다: (i) 초기 융합(early fusion), (ii) 동내성 모델링(intra-modality modeling), (iii) 다중 모달 추론(multimodal reasoning)입니다. 연구 결과, 텍스트 토큰이 충분한 이미지 정보를 수신하면 시각적 토큰이 더 이상 추론에 기여하지 않음을 발견하였습니다. DyVTE는 이 발견을 바탕으로 효율성을 높이고 추론 속도를 증가시키는 방법으로 제안되었습니다.

- **Performance Highlights**: DyVTE는 다양한 벤치마크에서 LLaVA, VILA, Eagle, InternVL 과 같은 여러 MLLMs에 적용되어 실험 결과 효율성과 성능을 모두 향상시켰습니다. 예를 들어, LLaVA-1.5의 계산 비용을 최대 45.7% 감소시키면서도 성능의 현저한 하락 없이 경쟁력을 유지합니다. DyVTE는 Flash Attention과 FastV와 같은 기존의 시각적 토큰 가지치기(token pruning) 방법과의 호환성도 갖추고 있습니다.



### GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding (https://arxiv.org/abs/2411.19626)
- **What's New**: 이번 논문은 Open-Vocabulary 3D 오브젝트 어포던스(affordance) 그라우딩을 위한 새로운 프레임워크인 GREAT(GeometRy-intEntion collAboraTive inference)를 제안합니다. 이 프레임워크는 객체의 불변 지오메트리 속성과 잠재적인 상호작용 의도를 탐지하여 3D 객체의 어포던스 지식을 형성합니다. 또한 점 이미지 어포던스 데이터셋 v2(PIADv2)를 소개하며, 이는 현재 가장 큰 3D 객체 어포던스 데이터셋입니다.

- **Technical Details**: GREAT는 다중 헤드 어포던스 체인의 사고(MHACoT) 추론 전략을 기반으로 하여 미세 조정된 MLLM(Multi-modal Large Language Models)을 사용하여 잠재적인 불변 기하학과 상호작용 의도를 추론합니다. 이후 크로스 모달 적응형 융합 모듈(CMAFM)을 이용해 점 구름 특징과 이미지 특징을 통합하여 3D 객체 어포던스를 정확히 그라우딩합니다. 이를 통해 시각적 컨텐츠와 기하학 정보를 효과적으로 결합합니다.

- **Performance Highlights**: GREAT는 24개의 일반적인 어포던스와 43개의 다양한 객체 카테고리, 15,000개 이상의 상호작용 이미지를 포함한 PIADv2 데이터셋을 통해 그 효과성을 입증합니다. 다수의 실험을 통해 GREAT의 우수성을 보여주며, 이는 이전 방법론들보다 더 넓은 어포던스 범위를 제대로 그라우딩할 수 있습니다. 이러한 연구는 로봇 조작, 장면 이해 및 행동 예측 등 여러 응용 분야에서의 가능성을 넓힙니다.



### FairDD: Fair Dataset Distillation via Synchronized Matching (https://arxiv.org/abs/2411.19623)
- **What's New**: 본 논문은 Fair Dataset Distillation (FDD) 프레임워크인 FairDD를 제안합니다. 이 프레임워크는 축소된 데이터셋에서 보호 특성(Protected Attributes, PA)에 대한 공정성을 확보할 수 있도록 설계되었습니다. 또한, FairDD는 기존 데이터 증류 기법들의 아키텍처 수정 없이 다양한 DD 접근 방식에 원활하게 적용될 수 있습니다. 기존의 데이터 축소 방법들이 주로 다수 집단에 집중되는 경향을 보인 반면, FairDD는 PA 그룹에 대한 동기화를 통해 균형 잡힌 생성이 가능합니다.

- **Technical Details**: FairDD는 첫째, 표현하는 집합을 PA에 따라 분할하고, 기존의 데이터 증류 방법들이 사용하는 단일 정렬 대상을 PA별 하위 대상으로 분해합니다. 둘째, 동기화된 매칭을 통해 생성된 샘플이 특정 PA 그룹의 크기에 영향을 받지 않고 각 PA 그룹에 부여됩니다. 이러한 방식은 기존의 DD를 공정한 데이터 증류 방식으로 변환하여 PA를 불균형적으로 생성하지 않도록 돕습니다.

- **Performance Highlights**: 공식적인 분석과 광범위한 실험 결과는 FairDD가 기존의 데이터 증류 방법들에 비해 공정성을 크게 개선하는데 기여함을 보여줍니다. FairDD는 Distribution Matching (DM) 및 Gradient Matching (GM)과 같은 다양한 데이터 증류 패러다임에서 우수성을 발휘합니다. 이를 통해 본 연구는 데이터 축소에서 공정성을 포함하려는 최초의 시도로, 실제 사용에 있어 데이터셋의 불공정성을 효과적으로 완화할 수 있음을 입증했습니다.



### Tortho-Gaussian: Splatting True Digital Orthophoto Maps (https://arxiv.org/abs/2411.19594)
Comments:
          This work has been submitted to the IEEE Transactions on Geoscience and Remote Sensing for possible publication

- **What's New**: 본 논문에서는 TDOM(진정 디지털 정사 사진) 생성을 위한 새로운 방법인 TOrtho-Gaussian을 소개합니다. 이 방법은 3D Gaussian Splatting(3DGS)에 영감을 받아, 최적화된 비등방성 가우시안 커널을 사용하여 TDOM을 생성하는 작업입니다. 특히, 본 논문에서는 전통적인 DSM(디지털 지표 모델)과 차단 감지의 필요성을 회피하여, 기하학적으로 우수한 솔루션을 제시합니다.

- **Technical Details**: TOrtho-Gaussian 방법에서는 가우시안 커널을 2D 이미지 평면에 정사각형 형태로 스플랫팅(평면 투영)하여 TDOM을 생성합니다. 이 방법은 대규모 지역의 TDOM 생성을 위해 분할 정복(divide-and-conquer) 전략을 채택하여 메모리 사용과 훈련 및 렌더링의 시간 효율성을 최적화합니다. 또한, 각기 다른 지역의 특성에 적응하는 완전 비등방성 가우시안 커널을 설계하여 반사 표면과 날씬한 구조의 렌더링 품질을 향상시킵니다.

- **Performance Highlights**: 실험 결과, TOrtho-Gaussian 방법은 기존 상용 소프트웨어보다 건물 경계의 정확성과 저질 텍스처 지역 및 건물 전면의 시각적 품질에서 뛰어난 성능을 보여줍니다. 이와 같은 결과는 본 연구가 대규모 도시 경관 재구성을 위한 TDOM 품질 및 확장성을 향상시키는 데 유망한 대안이 될 수 있음을 강조합니다.



### Gaussian Splashing: Direct Volumetric Rendering Underwater (https://arxiv.org/abs/2411.19588)
- **What's New**: 본 논문에서는 수중 이미지에서의 복잡한 과정을 혁신적으로 개선한 새로운 기법인 Gaussian Splashing을 제안합니다. 기존의 NeRFs가 수중 장면에서 느리게 작동하는 문제를 해결하면서, 그 성능을 획기적으로 향상시킵니다. 이 새로운 방법은 몇 분 안에 재구성을 완료하고, 140 FPS의 속도로 새로운 수중 장면을 렌더링할 수 있습니다. 이는 수중 환경에서 거리 장면 세부사항을 더욱 선명하게 드러냅니다.

- **Technical Details**: Gaussian Splashing은 3D Gaussian Splatting (3DGS)의 장점과 속도를 통합하여 산란(scaterring)을 효과적으로 캡처합니다. 이 방법은 새로운 이미지 형성 모델(image formation model)과 함께 렌더링(rendering) 및 깊이 추정(depth estimation) 절차를 혁신적으로 개선했습니다. 또한, 3DGS 손실 함수(loss function)에서도 혁신을 도입하여 수중 적응을 최적화합니다.

- **Performance Highlights**: 우리의 방법은 기존 데이터셋과 우리가 수집한 새로운 데이터셋에서 비교할 수 없는 속도와 세부사항으로 결과를 보여줍니다. 특히, 수중 환경에서의 기타 방법들과 비교할 때, 재구성한 이미지의 품질이 훨씬 우수하다는 점이 강조됩니다. 이 연구는 수중 이미징 분야에서의 재구성과 렌더링에 있어 새로운 가능성을 제시합니다.



### LDA-AQU: Adaptive Query-guided Upsampling via Local Deformable Attention (https://arxiv.org/abs/2411.19585)
Comments:
          Accepted by ACM MM2024

- **What's New**: 이번 논문에서는 전통적인 feature upsampling 방법들이 특정 feature 지침 부족 또는 고해상도 feature map 필요성으로 인해 성능 저하와 유연성 결여의 문제가 있다고 지적합니다. 저자들은 local self-attention의 특성이 feature upsampling 작업과 밀접하게 연관되어 있음을 발견하고, 이를 기반으로 새로운 upsampler인 LDA-AQU를 제안합니다. LDA-AQU는 query feature를 기반으로 인접 포인트의 위치와 집합 가중치를 적응적으로 조정하여 다양한 복잡한 상황에서도 효과적인 upsampling을 제공합니다.

- **Technical Details**: LDA-AQU는 feature upsampling 작업에서 local self-attention을 통합하여 adaptive upsampling을 구현합니다. 고정된 이웃 포인트의 사용으로 발생할 수 있는 비효율성을 극복하기 위해, LDA-AQU는 query point의 features와 맥락 정보에 기반하여 이웃 포인트의 위치를 동적으로 조정하는 deformation mechanism을 도입하였습니다. 이 방법은 단일 레이어에서 작동하며, 고해상도 feature maps가 필요하지 않으며, query-guided 기능을 가지고 있어 동적 upsampling kernels의 생성을 가능하게 합니다.

- **Performance Highlights**: LDA-AQU는 객체 탐지, 인스턴트 분할, 범위적 분할, 의미적 분할 등 네 가지 밀집 예측 작업에서 성능을 평가하였습니다. 각 작업에서 LDA-AQU는 이전의 state-of-the-art upsampler들을 지속적으로 초월하며, Faster R-CNN에서 +1.7 AP, Mask R-CNN에서 +1.5 AP, Panoptic FPN에서 +2.0 PQ, UperNet에서 +2.5 mIoU의 성능 향상을 달성하였습니다. 이 모든 결과는 LDA-AQU가 유사한 FLOPs와 파라미터를 유지하면서도 탁월한 성능을 가진다는 것을 입증합니다.



### Bootstraping Clustering of Gaussians for View-consistent 3D Scene Understanding (https://arxiv.org/abs/2411.19551)
- **What's New**: 이 논문은 3D Gaussian Splatting (3DGS)에 의미론적 요소를 주입하는 새로운 방법인 FreeGS를 제안합니다. FreeGS는 2D 레이블 없이도 적절한 뷰 일관성을 달성하여 3D 장면 이해를 개선할 수 있습니다. 핵심 아이디어는 IDentity-coupled Semantic Field (IDSF)를 도입하여 각 Gaussian에 대한 의미적 표현과 뷰 일관성 인덱스를 포착하는 것입니다.

- **Technical Details**: FreeGS는 IDSF를 최적화하는 데 있어 두 단계의 교대 전략을 채택합니다. 먼저, 3D 군집화 단계에서는 의미 정보가 Gaussian의 내재적 특성과 협력하여 비지도 방식으로 일관된 인스턴스를 추출합니다. 그 다음, 2D 증류 단계에서는 추출된 3D 인스턴스 인덱스가 다중 뷰 이미지의 상응 관계를 제한하여 뷰 일관성 있는 의미적 표현을 학습하도록 돕습니다.

- **Performance Highlights**: FreeGS는 기존의 최첨단 방법들과 견줄 만한 성능을 자랑하지만 복잡한 데이터 전처리 작업을 피할 수 있습니다. 다양한 데이터셋에 대한 실험을 통해, 자유로운 시각 세분화, 객체 선택, 및 3D 객체 탐지와 같은 작업에서의 효율성을 입증했습니다. 결과적으로 FreeGS는 고해상도의 실시간 렌더링을 허용하며, 3DGS의 디자인을 개선하여 보다 유연한 콘텐츠 조작을 지원합니다.



### ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration (https://arxiv.org/abs/2411.19548)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 ReconDreamer를 소개하여 자율주행 세계 모델로부터의 지식을 점진적으로 통합함으로써 드라이빙 장면 재구성을 개선하고 있습니다. ReconDreamer는 특히 대규모 기동(large maneuvers)의 효과적인 렌더링을 할 수 있는 최초의 방법으로, 최대 6미터까지의 차선 변경 등을 처리할 수 있습니다. DriveRestorer는 온라인 복구를 통해 유령 아티팩트를 완화하도록 설계되었으며, 보다 복잡한 기동을 위한 고품질 렌더링을 보장하는 점진적 데이터 업데이트 전략을 포함하고 있습니다.

- **Technical Details**: ReconDreamer는 온라인 복구 프로세스를 통해 학습 데이터를 확장하며, 각기 다른 훈련 단계에서의 렌더링 출력 샘플링을 사용하여 비디오 복원 데이터셋을 생성합니다. DriveRestorer는 훈련된 세계 모델에 의해 유령 아티팩트를 완화하도록 미세조정되며, 마스킹 전략을 통해 도전적인 구역의 복구를 강조합니다. 점진적 데이터 업데이트 전략을 통해 복잡한 기동에 대한 렌더링 품질을 유지하는 동시에 아티팩트를 단계적으로 복구합니다.

- **Performance Highlights**: 실험 결과, ReconDreamer는 NTA-IoU, NTL-IoU 및 FID에서 Street Gaussians보다 각각 24.87%, 6.72%, 29.97% 개선된 성능을 보였습니다. 또한, 사용자 연구에서 96.88%의 승률을 기록하며 DriveDreamer4D를 초월하는 성능을 입증했습니다. 이로 인해 ReconDreamer는 대규모 기동 렌더링에서 195.87%의 상대적 개선을 달성하였습니다.



### SkelMamba: A State Space Model for Efficient Skeleton Action Recognition of Neurological Disorders (https://arxiv.org/abs/2411.19544)
- **What's New**: 이 논문은 해부학적으로 안내되는 새로운 state-space model (SSM) 기반의 프레임워크를 소개하고, 이는 임상 진단 및 일반적인 행동 인식 작업에서의 최첨단 성능을 향상시킵니다. 본 접근 방식은 스켈레톤 모션 분석을 공간적, 시간적, 공간-시간적 스트림으로 분해하여 고유한 움직임 특성을 효율적으로 캡처합니다. 또한, 우리의 모델은 로컬 조인트 상호작용과 여러 해부학적 신체 부위 간의 글로벌 motion 패턴을 포착할 수 있도록 설계되었습니다.

- **Technical Details**: 이 모델은 각각의 해부학적 신체 부위를 고려하여 스켈레톤 모션 데이터를 구조적으로 분해합니다. 입력 시퀀스는 공간, 시간 및 공간-시간적 분석을 위한 전문 그룹으로 채널 표현이 분할됩니다. 이를 통해 우리의 구조화된 SSM 접근 방식은 복잡한 모션 모델링을 위한 multi-directional scanning 전략을 구현하여 사용자 정의된 방법으로 각 해부학적 그룹을 동시에 분석합니다.

- **Performance Highlights**: 우리의 모델은 NTU RGB+D 및 NW-UCLA와 같은 공공 행동 인식 기준에서 현재 최첨단 방법들을 능가하며, 최대 $3.2\%$의 정확도 향상과 함께 이전의 transformer 기반 모델보다 낮은 계산 복잡도로 성과를 냈습니다. 또한, 우리는 자동화된 질병 진단 가능성을 검증하기 위해 운동 기반 환자 신경 장애 분석을 위한 새로운 의료 데이터세트를 도입했습니다.



### Deepfake Media Generation and Detection in the Generative AI Era: A Survey and Outlook (https://arxiv.org/abs/2411.19537)
- **What's New**: 이 논문은 최근의 생성 모델링(Generative Modeling) 기술 발전에 따른 딥페이크(Deepfake) 콘텐츠의 현실감이 높아지고 있으며, 그로 인해 사람들이 조작된 미디어 콘텐츠를 탐지하는 데 실패하는 경우가 증가하고 있음을 강조합니다. 또한, 본 연구는 딥페이크 생성 및 탐지 기술에 대한 포괄적인 조사 결과 및 최신 기술들, 예를 들어 확산 모델(Diffusion Models)과 Neural Radiance Fields를 소개합니다.

- **Technical Details**: 딥페이크는 이미지, 비디오, 오디오 및 멀티모달(오디오-비주얼) 콘텐츠를 포함하며, 이를 기존 사용 절차에 따라 여러 유형으로 분류합니다. 연구자들은 딥페이크 생성을 위한 방법과 이를 탐지하기 위한 방법의 분류 체계를 구축하였으며, Generative Adversarial Networks (GANs)와 CNN(Convolutional Neural Networks) 기반의 탐지 방법들이 널리 사용되고 있음을 발견했습니다.

- **Performance Highlights**: 딥페이크 탐지를 위한 데이터세트를 수집하고, 현재 사용되는 최고의 탐지기들의 성능 순위를 제공하여 비교를 용이하게 했습니다. 새로운 멀티모달 벤치마크를 통해 딥페이크 탐지기의 일반화 능력을 평가한 결과, 현재의 최첨단 탐지기들은 새로운 및 더 강력한 생성 모델들이 생성한 딥페이크 콘텐츠에 대해 일반화 능력이 떨어짐을 보여주었습니다. 이 논문은 앞으로 보다 강력한 탐지기를 개발하기 위한 연구 방향도 제안하고 있습니다.



### QUOTA: Quantifying Objects with Text-to-Image Models for Any Domain (https://arxiv.org/abs/2411.19534)
Comments:
          12 pages, 6 figures

- **What's New**: 본 논문에서는 text-to-image 모델을 이용한 객체 수 측정 문제를 다루고 있습니다. QUOTA라는 프레임워크를 제안하여, 이러한 모델을 재학습하지 않고도 다양한 새로운 도메인에서 효과적인 객체 수량화를 가능하게 합니다. 이 접근 방식은 메타 학습(meta-learning) 전략을 통해 도메인 불변 프롬프트를 최적화하는 것을 특징으로 합니다.

- **Technical Details**: QUOTA는 두 개의 루프를 갖는 메타 학습 전략을 사용하여 프롬프트 최적화를 수행합니다. 내부 루프에서는 메타-훈련 도메인에서 프롬프트 매개변수를 최적화하며, 외부 루프에서는 메타-테스트 도메인에서 이를 정제하여 새로운 도메인에 일반화된 프롬프트를 학습합니다. 또한, 학습 가능한 수량화 및 도메인 토큰을 통합하여, 다양한 스타일 변화를 포착하면서도 객체 수 측정의 정확성을 유지합니다.

- **Performance Highlights**: 결과적으로 QUOTA는 기존 모델들과 비교하여 객체 수 측정 정확도와 의미 일치성 면에서 우수한 성능을 보였습니다. 새로운 benchmark를 통해 다양한 도메인에서의 객체 수량화의 정확성과 적응력을 엄격히 평가할 수 있었으며, QUOTA는 모든 도메인에서 효율적이고 확장 가능한 text-to-image 생성의 기준을 새롭게 설정하였습니다.



### RAGDiffusion: Faithful Cloth Generation via External Knowledge Assimilation (https://arxiv.org/abs/2411.19528)
Comments:
          Project website: this https URL

- **What's New**: RAGDiffusion이라는 새로운 Retrieval-Augmented Generation (RAG) 프레임워크를 제안하여 표준 화상 의류 자산 생성을 위한 이미지의 구조적 정확성을 높이고 허위 구조를 완화합니다. 이 모델은 외부 지식과 데이터베이스에서 인지된 정보를 통합하여 생성 과정에서 구조적 모호성을 제거합니다. 또한, RAGDiffusion은 두 가지 핵심 프로세스인 구조 집합화 및 전 수준 신뢰성 의류 생성을 사용합니다.

- **Technical Details**: RAGDiffusion은 대조 학습(Contrastive Learning)과 Structure Locally Linear Embedding (SLLE)을 이용하여 전 세계적인 구조 및 공간 랜드마크를 도출합니다. 이 구조 집합화 기술은 구조적 불확실성을 최소화하며, 세 가지 수준의 정렬을 통해 의류의 구조, 패턴, 그리고 디코딩 요소에서 충실도를 보장합니다. VAE(변분 오토인코더)로부터 구조 복원 시 왜곡을 줄이는 Parameter Gradual Encoding Adaptation (PGEA) 기법도 적용됩니다.

- **Performance Highlights**: RAGDiffusion은 복잡한 현실 세계 데이터셋에서 구조적으로 정밀하고 세부사항까지 충실한 의류 자산을 생성하는 데 큰 성능 향상을 보여주었습니다. 여러 실험 결과를 통해 기존 모델 대비 우수한 성능을 입증하며, 구조적 허위 현상 문제를 해결하고 높은 사양의 생성을 위한 선도적 노력으로 자리매김했습니다. 최근의 훈련 없이도 유망한 결과를 달성한 점이 주목할 만합니다.



### DisCoRD: Discrete Tokens to Continuous Motion via Rectified Flow Decoding (https://arxiv.org/abs/2411.19527)
Comments:
          20 pages 18 figures

- **What's New**: 이번 연구에서는 DisCoRD(Discrete Tokens to Continuous Motion via Rectified Flow Decoding)라는 새로운 방법을 제안합니다. 이 방법은 이산적인 동작 토큰을 연속적인 동작으로 변환하여 부드러운 모션을 생성하는 데 중점을 두고 있습니다. DisCoRD는 iterative refinement 과정을 통해 정밀한 다이내믹스를 포착함으로써 더욱 자연스러운 모션을 제공합니다.

- **Technical Details**: DisCoRD는 score-based models를 활용하여 이산적 동작 표현을 연속적인 도메인에서 해독하는 방법입니다. 이 모델은 고속 동작을 포착할 수 있는 iterative하고 stochastic한 성격을 이용하여, 조각 조각된 디스크리트 토큰을 원시 모션 공간에서 해독하여 자연스러운 모션을 생성합니다. 이 과정에서 기존의 deterministic decoder를 대체하여 더욱 세부적인 동작을 복원할 수 있는 장점을 갖춥니다.

- **Performance Highlights**: 실험 결과, DisCoRD는 HumanML3D에서 FID가 0.032, KIT-ML에서 0.169을 달성하여 현재의 최첨단 성능을 입증했습니다. 또한, 우리의 방법은 다양한 동작 생성 프레임워크와 독립적으로 조화를 이루며, 텍스트-투-모션, 음악-투-댄스 등 다양한 태스크에서 우수한 자연스러움을 보여줍니다. 새로운 평가 메트릭인 symmetric Jerk Percentage Error (sJPE)를 도입하여 복원도와 프레임 간 노이즈를 동시에 평가할 수 있는 방안을 마련했습니다.



### LokiTalk: Learning Fine-Grained and Generalizable Correspondences to Enhance NeRF-based Talking Head Synthesis (https://arxiv.org/abs/2411.19525)
- **What's New**: 이번 논문에서는 전통적인 Neural Radiance Fields (NeRF) 기반의 talking head 생성 방식에서 발생하는 시각적 아티팩트(visual artifacts) 및 높은 훈련 비용(training costs) 문제를 해결하기 위한 새로운 프레임워크 LokiTalk를 제안합니다. LokiTalk는 Lip movements, eye blinking, head pose, torso movements와 같은 세부적인 움직임을 보다 정교하게 처리하여 생동감 있는 얼굴 동역학을 제공하며, 훈련 효율성을 크게 향상시킵니다. 또한, ID-Aware Knowledge Transfer 모듈을 도입하여 다수의 개체 비디오에서 동적 및 정적 특징을 일반화하여 개별 캐릭터의 표현을 개선합니다.

- **Technical Details**: LokiTalk는 Region-Specific Deformation Fields를 통해 다양한 구역의 움직임과 음성 신호(driving signals) 간의 세부적인 관계를 수립합니다. 이 시스템은 여러 구역을 계층적으로 모델링하여 얼굴과 입의 움직임, 눈 깜빡임, 머리 움직임, 상체 움직임 등을 분리하여 처리합니다. 또한, ID-Aware Knowledge Transfer를 통해 개체 간의 공통된 특징을 학습하며 개인화된 특징을 추출하여 모델의 학습을 촉진합니다.

- **Performance Highlights**: LokiTalk는 이전 방법들과 비교하여 더 높은 충실도(high-fidelity)와 훈련 효율성을 보여주는 실험 결과를 제공합니다. 특히, 다수의 개체를 대상으로 하는 비디오에서 개인적이면서 통일된 표현을 생성할 수 있으며, 높은 수준의 사실감을 유지합니다. 전체적으로 이 방법은 상업적 적용을 위한 비용 효율성을 개선하고, 화질을 높이는 혁신적인 결과를 도출합니다.



### Retrieval-guided Cross-view Image Synthesis (https://arxiv.org/abs/2411.19510)
- **What's New**: 본 논문에서는 Cross-view image synthesis 분야에서 기존 방법론이 가진 한계점을 해결하기 위해 새로운 방법을 제안합니다. 특히, semantic segmentation maps와 같은 추가 데이터 없이 view-specific semantics에 중점을 두어 이미지의 품질과 사실감을 개선하는 접근 방식을 채택했습니다. 또한, 초점이 맞춰진 새로운 데이터셋인 VIGOR-GEN을 도입하여 다양한 도시 환경에서의 이미지 쌍을 제공합니다.

- **Technical Details**: 제안된 방법은 retrieval-guided framework를 통해 view-invariant semantics를 인식하고, 이를 기반으로 generator가 이미지 생성을 수행하게 합니다. 이 과정에서 semantic segmentation maps를 사용하지 않고, view-specific semantics를 강화하기 위한 노이즈와 모듈화된 스타일을 결합하여 시각적 특징을 다양화합니다. 전체 프로세스는 end-to-end로 진행되며, 기존의 복잡한 전처리 과정을 생략할 수 있습니다.

- **Performance Highlights**: 많은 실험 결과는 본 방법이 SSIM 및 FID 평가에서 기존의 최첨단 방법들을 크게 능가했음을 보여주었습니다. 특히, CVUSA, CVACT, VIGOR-GEN 데이터셋에서의 성능 개선이 두드러져, 보다 사실적인 이미지를 생성하는 데 성공했습니다. 이를 통해 제안된 방법의 각 구성 요소의 효능이 뒷받침되었습니다.



### Ditto: Motion-Space Diffusion for Controllable Realtime Talking Head Synthesis (https://arxiv.org/abs/2411.19509)
- **What's New**: 이번 연구에서는 Ditto라는 새로운 diffusion 기반 프레임워크를 제안하여, 음성을 기반으로 하는 실시간 Talking Head 합성을 가능하게 합니다. Ditto는 기존의 Variational Auto-Encoders (VAE) 대신, 명확한 정체성 무관한 움직임 공간을 통해 다양한 얼굴 움직임을 생성하고, 사진 한 장으로 생생한 비디오를 만들어냅니다. 이 방법은 더 빠르고 세밀한 제어를 제공하여 실시간 인터랙션 애플리케이션에서 활용할 수 있습니다.

- **Technical Details**: Ditto 프레임워크는 음성, 감정 레이블, 눈 상태를 조건으로 하여, Diffusion Transformer (DiT)를 통해 정밀한 얼굴과 머리 움직임의 제어를 가능하게 합니다. 이 과정에서 전통적인 VAE 기반의 잠재 공간을 제거하고, 대신 얼굴 애니메이션에 적합한 저차원의 움직임 공간을 구성하여 복잡성을 줄입니다. 이를 통해 데이터의 흐름을 최적화하고, 비디오 합성 속도를 크게 향상시켰습니다.

- **Performance Highlights**: 광범위한 실험 결과를 통해 Ditto가 기존 방법보다 우수한 움직임 제어 및 실시간 성능을 제공한다는 것을 입증했습니다. 특히, 이 프레임워크는 AI 보조기기와 같은 인터랙티브 애플리케이션의 필수 조건인 낮은 첫 번째 프레임 지연과 실시간 처리를 가능케 합니다. Ditto는 진짜 같은 Talking Head 비디오 생성을 통해 새로운 대화 애플리케이션을 위한 기초를 마련하고 있습니다.



### Diorama: Unleashing Zero-shot Single-view 3D Scene Modeling (https://arxiv.org/abs/2411.19492)
- **What's New**: 본 논문은 Diorama라는 제로샷(Zero-Shot) 오픈 월드 시스템을 소개합니다. 이 시스템은 단일 관측치(single-view RGB observation)로부터 CAD 객체를 통해 3D 장면을 재구성하는 방식으로 작업하며, 엔드 투 엔드 훈련이나 인간의 주석이 필요 없습니다. 기존의 방법들이 비싸고 부정확한 실세계 주석 또는 단조로운 합성 데이터에 의존했던 것과는 대조적입니다.

- **Technical Details**: Diorama는 두 가지 주요 구성요소로 이루어져 있습니다. 첫 번째는 입력 이미지에 대한 전체 장면 이해를 위한 오픈 월드 인식(Open-World Perception)으로, 객체 인식, 위치 확인, 깊이 및 노멀 추정, 건축 재구성 및 장면 그래프 생성을 포함합니다. 두 번째는 CAD 모델 검색, 9자유도(9-DoF) 자세 추정 및 의미적으로 인식된 장면 레이아웃 최적화를 통한 CAD 기반 장면 모델링입니다.

- **Performance Highlights**: 시스템의 성능은 합성 및 실제 데이터를 기준으로 평가됩니다. 실험 결과는 다른 방법들에 비해 현저하게 우수한 성능을 나타내며, 인터넷 이미지 및 텍스트에서 장면으로 환원하는 과정에서도 일반화된 성능을 보여줍니다. Diorama는 장면 생성의 유연성과 일반성을 강조하며 다양한 접근 방법을 통해 집합적인 장면을 재구성할 수 있는 강점을 가지고 있습니다.



### Interleaved-Modal Chain-of-Though (https://arxiv.org/abs/2411.19488)
- **What's New**: 이번 연구에서는 멀티모달 체인 오브 사고(Interleaved-modal Chain-of-Thought, ICoT)를 제안합니다. ICoT는 이미지와 텍스트를 결합하여 중간 추론 단계를 생성함으로써 최종 답변을 추론할 수 있도록 합니다. 이는 기존의 텍스트 전용 논리와는 달리, 이미지 정보를 포함하여 보다 정교한 연관성을 표현하는 데 초점을 맞추고 있습니다. ICoT는 VLM의 사고 과정을 인간의 사고 과정과 더욱 밀접하게 정렬시키는 혁신적인 접근법입니다.

- **Technical Details**: ICoT를 실현하는 데 있어 주목할 점은 Attention-driven Selection (ADS) 전략입니다. ADS는 VLM이 입력 이미지에서 패치를 선택하여 중간 생성 과정에 삽입함으로써 세밀한 시각적 논리를 생성하도록 합니다. 이 방법은 추가적인 매개변수를 요구하지 않으며, 클립 앤 플레이 방식으로 다양한 VLM에 쉽게 적용될 수 있습니다. ADS는 VLM의 주의 맵(attention map)을 활용하여 최적의 패치를 식별하고 선택하는 방식으로, 기존의 텍스트 전용 방법에 비해 무시할 수 있는 추론 지연(latency)을 제공합니다.

- **Performance Highlights**: 연구 결과, M3CoT, ScienceQA 및 LLaVA-W와 같은 벤치마크에서 ICoT를 통해 최대 14%의 성능 향상을 달성했습니다. 또한, 생성된 추론의 해석 가능성도 더욱 향상되었습니다. ICoT는 기존의 멀티모달 체인 오브 사고 방식과 비교할 때, 중첩된 모달 추론 프로세스의 혁신적인 기초를 제공하여 VLM의 추론 능력을 극대화하는 데에 기여합니다.



### V2SFlow: Video-to-Speech Generation with Speech Decomposition and Rectified Flow (https://arxiv.org/abs/2411.19486)
- **What's New**: 이번 논문에서는 V2SFlow라는 새로운 Video-to-Speech (V2S) 프레임워크를 소개합니다. 이 시스템은 무음 영상에서 직접 자연스럽고 이해 가능한 음성을 생성하는 데 중점을 둡니다. 기존 시스템들이 제한된 데이터셋에서 우수한 성능을 보였으나, 실제 환경에서는 성능이 저하되는 문제를 해결하고자 합니다.

- **Technical Details**: V2SFlow는 음성 신호를 내용(content), 음조(pitch), 화자 정보(speaker information)의 세 가지 하위 공간으로 분해합니다. 각 하위 공간은 독특한 음성 속성을 나타내고, 각 속성은 시각적 입력에서 직접 예측됩니다. 이를 통해, Rectified Flow Matching (RFM) 기반의 디코더를 활용하여 고해상도의 멜-스펙트로그램(mel-spectrogram)을 재구성합니다.

- **Performance Highlights**: 광범위한 실험 결과, V2SFlow는 최신 기술(state-of-the-art)과 비교하여 유의미한 성능 향상을 보여주며, 심지어 실제 발화(ground truth)보다 더 자연스러운 음성을 생성하는 성과를 거두었습니다. 이 기술은 다양한 실제 환경에서의 응용 가능성을 더욱 확대할 것으로 기대됩니다.



### Effective Fine-Tuning of Vision-Language Models for Accurate Galaxy Morphology Analysis (https://arxiv.org/abs/2411.19475)
- **What's New**: 이번 논문에서는 GalaxAlign이라는 새로운 방법론을 제안하여, 자연 이미지 데이터로 사전 학습된 파운데이션 모델을 우주 이미지 작업에 효과적으로 적응시켜 높은 정확도를 달성합니다. 이 방법은 세 가지 유형의 데이터를 정렬하는 대조 학습 아키텍처를 확장하여, 기하학적 기호, 텍스트 레이블, 우주 이미지를 통합합니다. 이를 통해 고비용의 사전 훈련 없이도 효과적인 파인 튜닝이 가능해집니다.

- **Technical Details**: GalaxAlign은 CLIP 아키텍처를 기반으로 하는 삼중 모달 학습 프레임워크로, 텍스트 설명, 우주 이미지, 기하학적 기호를 결합하여 우주 형태 분석에 적합합니다. 먼저 기호와 이미지를 함께 처리하는 이미지 인코더를 사용한 후, 두 번째 단계에서는 이미지 인코더가 우주 이미지만을 인코딩하도록 조정됩니다. 이 단계에서 세 가지 인코더가 각 모달리티에 특화되며, 우주 특징을 더욱 정교하게 표현할 수 있도록 합니다.

- **Performance Highlights**: GalaxAlign은 우주 분류와 유사성 검색에서의 효율성을 입증하였으며, 대규모의 우주 데이터 세트에 대한 의존성을 줄임으로써 고효율의 모델 학습을 가능하게 하였습니다. 실험 결과 이 방법론을 적용한 모델이 높은 정확도를 기록하며, 일반적으로 낮은 성능을 보였던 기존 접근 방식과 비교하여 더 나은 결과를 나타냄을 보여주었습니다.



### ForgerySleuth: Empowering Multimodal Large Language Models for Image Manipulation Detection (https://arxiv.org/abs/2411.19466)
- **What's New**: 이번 연구에서는 복합 멀티모달 대규모 언어 모델(M-LLM)을 활용하여 이미지 조작 탐지(image manipulation detection, IMD) 작업의 새로운 가능성을 모색하였습니다. 기존의 M-LLM이 이미지 조작 탐지에서 발생하는 문제점을 해결하기 위해 ForgerySleuth라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 텍스트 기반의 설명을 제공하고 조작된 영역을 표시하는 세분화(segmentation) 결과를 생성할 수 있습니다.

- **Technical Details**: ForgerySleuth는 M-LLM과 트레이스 인코더(trace encoder)를 결합하여 세계 지식을 활용하여 고-level(고수준) 이상을 탐지하고, 낮은 수준의 조작 흔적을 추적합니다. 또한, Chain-of-Clues 프롬프트를 사용하여 ForgeryAnalysis 데이터세트를 생성했으며, 딥러닝 모델의 세부 조정을 위한 감독 학습(supervised fine-tuning, SFT)이 가능하도록 하였습니다. 데이터 세트는 고급 언어 모델을 활용하여 전문적이고 세부적인 분석 지침을 포함하고 있습니다.

- **Performance Highlights**: 실험 결과, ForgerySleuth는 픽셀 수준의 조작 지역 로컬라이제이션(pixel-level localization) 작업에서 기존 최첨단 방법(State of the Art, SoTA)을 최대 24.7% 개선하였습니다. 또한, ForgeryAnalysis-Eval 평가에서 GPT-4o와 비교하여 35.8% 향상된 성능을 기록했습니다. 이러한 결과는 M-LLM을 통한 이미지 조작 탐지의 가능성을 보여주며, 데이터 분석 및 조작 감지 작업의 이해 가능성을 높였습니다.



### Robust Bayesian Scene Reconstruction by Leveraging Retrieval-Augmented Priors (https://arxiv.org/abs/2411.19461)
- **What's New**: 이 논문에서는 단일 RGBD 이미지를 기반으로 다중 객체 장면을 재구성하는 새로운 방법, BRRP(Bayesian Reconstruction with Retrieval-augmented Priors)를 제안합니다. BRRP는 기존의 메쉬 데이터 세트를 활용하여 정보가 풍부한 사전 분포를 구축함으로써 견고한 확률적 재구성을 수행할 수 있습니다. 이를 통해 BRRP는 노이즈가 있는 실제 객체 관찰의 문제를 극복하고, 불확실성을 측정하는 데도 유용한 분포를 생성합니다.

- **Technical Details**: BRRP는 첫 번째로 RGBD 이미지와 이에 상응하는 객체 분할 결과를 기반으로 재구성을 시작합니다. 이후 분류 결과에 따라 사전 분포에서 관련 요소를 검색하는 retrieval-augmented prior 개념을 도입합니다. Stein Variational Gradient Descent(SVGD)를 사용하여 객체 형상에 대한 후행 분포를 유추하며, 각 객체에 대한 Hilbert Maps로 정의됩니다.

- **Performance Highlights**: BRRP는 절차적으로 생성된 장면과 실제 환경 모두에서 정확한 재구성을 보여주며, 알려지지 않은 객체에 대해서도 견고한 성능을 나타냅니다. 실험을 통해 BRRP는 기존의 심층 학습 방법보다 더 강건하면서도 정보가 부족한 사전 방법보다 더 높은 정확도를 기록했습니다. 우리의 결과는 기계 조작과 관련된 응용 프로그램에서도 활용될 수 있는 신뢰할 수 있는 불확실성 측정을 제공합니다.



### Look Every Frame All at Once: Video-Ma$^2$mba for Efficient Long-form Video Understanding with Multi-Axis Gradient Checkpointing (https://arxiv.org/abs/2411.19460)
Comments:
          Project page: this https URL

- **What's New**: 비디오 데이터를 처리하기 위한 새로운 아키텍처인 Video-Ma2mba가 제안되었습니다. 이 아키텍처는 기존의 Transformer 기반 모델에서 attention 메커니즘을 State Space Models (SSMs)로 대체함으로써 시간과 메모리 요구사항을 선형으로 줄일 수 있도록 설계되었습니다. 또한 Multi-Axis Gradient Checkpointing (MA-GC) 방법을 도입하여 메모리 효율성을 높였습니다.

- **Technical Details**: Video-Ma2mba 아키텍처는 Mamba-2 구조의 변형으로, 비디오 시퀀스를 효율적으로 처리하기 위해 SSM을 사용합니다. 이러한 아키텍처 변경을 통해 attention 메커니즘의 복잡성을 선형으로 낮추었으며, MA-GC는 다축 방향으로 활성화를 저장함으로써 O(S)로 메모리 사용을 줄이는 데 기여합니다. 이는 모델이 1 FPS로 2시간 이상의 비디오 시퀀스를 처리할 수 있게 합니다.

- **Performance Highlights**: 실험 분석 결과, Video-Ma2mba는 1개의 GPU에서 수백만 개의 토큰 또는 2시간 이상의 긴 비디오 시퀀스를 효율적으로 처리할 수 있음을 입증하였습니다. 이 모델은 시계열 동역학을 상세하게 포착하여 비디오 이해 작업에서의 정확성과 응답의 관련성을 크게 향상시킵니다. 이러한 성능 개선은 기존 모델들에 비해 상당한 이점을 나타냅니다.



### Fleximo: Towards Flexible Text-to-Human Motion Video Generation (https://arxiv.org/abs/2411.19459)
- **What's New**: 이 논문은 인간의 움직임 비디오 생성을 텍스트와 기준 이미지만으로 가능하게 하는 새로운 작업을 제안합니다. 기존의 방법들은 영상과 포즈 비디오를 기반으로 했지만, 이로 인해 유연성과 정밀성이 떨어지는 문제가 있었습니다. 새롭게 제안된 Fleximo 프레임워크는 대규모 사전 훈련된 텍스트-3D 모델을 활용하여 이러한 문제를 해결하고자 합니다. 텍스트 및 참조 이미지만으로 고품질 비디오 생성을 가능하게 하므로 사용자에게 더 나은 컨트롤을 제공합니다.

- **Technical Details**: Fleximo는 입력된 텍스트를 LLM(대형 언어 모델)로 파싱하여 순차적인 모션 세그먼트로 나누는 방식으로 작동합니다. 그런 다음 T2M-GPT 모듈을 통해 각 모션 세그먼트에 대한 3D 메쉬를 생성하고, 이는 2D 공간에 투영되어 Skeleton 비디오 시퀀스로 바뀝니다. 이 과정은 앵커 포인트 기반 리스케일 방법과 스켈레톤 어댑터를 포함해 데이타의 일관성을 확보합니다. 이러한 기술적 접근은 최종적으로 영상 품질을 높이는 데 기여합니다.

- **Performance Highlights**: Fleximo 프레임워크는 MotionBench라는 새로운 벤치마크를 통해 평가되었고, 400개의 비디오에 걸쳐 다양한 정체성과 모션을 실험했습니다. MotionScore라는 새로운 메트릭을 도입하여 텍스트와 모션 간의 정렬 정확성을 평가하며, Fleximo는 기존의 텍스트 기반 영상 생성 방법보다 우수한 성능을 보였습니다. 질적 및 양적 지표 모두에서 Fleximo는 더 높은 품질의 출력 결과를 보여주며, 이 연구의 기여를 명확히 하고 있습니다.



### Multiview Equivariance Improves 3D Correspondence Understanding with Minimal Feature Finetuning (https://arxiv.org/abs/2411.19458)
- **What's New**: 이 논문에서는 Vision Transformer(ViTs) 기반의 모델들이 3D 공간 관계를 이해하는 능력을 평가하고 이를 개선하는 방안을 제시합니다. 기존에 2D 이미지를 주로 학습했던 이 모델들이 3D 구조를 인식하는 능력에 대한 깊이 있는 분석을 포함합니다. 눈에 띄는 점은 단일 객체에 대한 간단한 파인튜닝(finetuning)을 통해 3D 특성 이해도가 크게 향상되었다는 것입니다.

- **Technical Details**: 연구진은 3D 일관성을 평가하기 위해 다양한 시점에서의 다중 이미지에서 구현된 2D 특징들이 동일한 3D 포인트를 Represent하는 방식의 일관성을 바탕으로 3D 변환 불변성(view equivariance)을 조사했습니다. 결과적으로, DINOv2 모델이 기존의 다른 비전 모델에 비해 가장 우수한 성능을 보였습니다. 이를 통해 현행 비전 모델들이 3D 구조 이해에서 보여주었던 제약을 극복할 수 있는 단순하면서도 효과적인 방법을 제안합니다.

- **Performance Highlights**: DINOv2 모델은 3D 포즈 추정(pose estimation), 비디오 추적(video tracking), 그리고 의미론적 대응(semantic correspondence)을 포함한 여러 다운스트림 작업에서 성능 향상을 보여주었습니다. 특히, 단일 다중 뷰 쌍에 대한 한 번의 파인튜닝만으로도 DINOv2의 성능이 극적으로 개선된 것으로 나타났습니다. 연구 결과는 매우 유망하며, 3D 인식 모델의 발전을 위한 공개 코드를 통해 지속적인 연구가 이루어질 것으로 기대됩니다.



### GausSurf: Geometry-Guided 3D Gaussian Splatting for Surface Reconstruction (https://arxiv.org/abs/2411.19454)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 새로운 방법인 GausSurf를 소개합니다. 이는 3D Gaussian Splatting을 활용해 질 높은 표면 복원(surfaces reconstruction)을 달성하고, 텍스처가 풍부한 영역과 텍스처가 부족한 영역에서의 기하학적 가이드를 통합합니다. GausSurf는 기존의 방식들보다 더 나은 복원 품질을 제공합니다.

- **Technical Details**: GausSurf 방식은 두 가지 기본 영역인 텍스처가 풍부한 영역과 텍스처가 부족한 영역을 고려합니다. 텍스처가 풍부한 지역에서는 다중 관점 일관성을 통해 최적화를 가이드하며, 텍스처가 부족한 지역에서는 사전 학습된 normal estimation 모델을 사용합니다. 이 통합된 접근 방식은 Gaussians 최적화를 반복적으로 수행하여 최적화의 효율성과 정확성을 획기적으로 향상시킵니다.

- **Performance Highlights**: GausSurf 방식은 DTU 및 Tanks and Temples 데이터셋에서 고품질의 표면을 복원하며, 기존의 3D Gaussian 기반 방법들과 비교하여 재구성 품질과 시간 면에서 우수한 성능을 보입니다. 예를 들어, DTU 데이터셋에서 한 객체를 10분 이내에 고품질로 복원할 수 있습니다.



### Learning Visual Abstract Reasoning through Dual-Stream Networks (https://arxiv.org/abs/2411.19451)
Comments:
          10 pages, 6 figures

- **What's New**: 본 논문에서는 시각적 추론 작업에서의 한계를 극복하기 위해 듀얼 스트림 추론 네트워크(DRNet)를 제안합니다. 이 모델은 이미지를 효과적으로 분석하기 위해 두 개의 병렬 스트림을 사용하며, 여러 RPM 벤치마크에서 최상의 평균 성능을 달성합니다. 또한, DRNet는 다양한 배포 외 시나리오에서도 견고한 일반화 능력을 보여주었습니다.

- **Technical Details**: DRNet은 두 개의 고유한 스트림을 통해 이미지를 처리하며, CNN을 사용하여 국소(Local) 정보와 ViT를 통해 공간(Spatial) 정보에 주의합니다. 이 네트워크는 두 개의 스트림에서 추출된 고수준 특징을 결합한 뒤, 규칙 추출기를 통해 이미지 간의 관계를 유도하고 예측을 수행합니다. 이를 통해 DRNet은 비언어적 시각적 추론 문제에 효과적으로 적용될 수 있습니다.

- **Performance Highlights**: DRNet은 여러 데이터세트에서 다른 모델들보다 탁월한 성능을 보여주며, 객체 간의 규칙을 효과적으로 학습하여 시각적 추론 성능을 향상시킵니다. 시각화 결과는 학습된 규칙 표현이 규칙 카테고리에 따라 클러스터링될 수 있음을 보여주며, 이는 시각적 추상 과제를 용이하게 만듭니다. 이러한 성과는 두 개의 스트림 아키텍처가 시각적 추상 추론에서 중요한 역할을 할 수 있음을 시사합니다.



### Adaptive Interactive Segmentation for Multimodal Medical Imaging via Selection Engin (https://arxiv.org/abs/2411.19447)
- **What's New**: 본 논문은 의료 이미지 분석에서의 분할(segmentation) 문제를 해결하기 위해 새로운 전략 기반 상호작용 세분화 모델(SISeg)을 제안합니다. 이 모델은 다양한 의료 이미징 모달리티를 처리할 수 있도록 설계되었으며, 적응형 프레임 선택 엔진(AFSE)을 통합하여 효율성을 향상시킵니다. 이 접근법은 복잡한 의료 데이터를 효과적으로 처리하고 메모리 사용량을 줄이며, 분할 과정의 해석 가능성을 증가시킵니다.

- **Technical Details**: SISeg 모델은 Segment Anything Model 2(SAM2)를 기반으로 하며, 이미지 특성에 따라 적절한 프레임을 동적으로 선택하는 적응형 프레임 선택 엔진이 핵심입니다. 이 시스템은 의료 이미지에 대한 사전 지식 없이도 최적의 프롬프트 프레임을 선택하며, 영상 데이터의 시퀀스를 처리할 수 있는 능력을 갖추고 있습니다. 또한, 비지도 점수 매커니즘을 도입하여 다양한 의료 이미징 모달리티를 효과적으로 처리할 수 있습니다.

- **Performance Highlights**: SISeg 모델은 7개 대표 의료 이미징 모달리티를 포함한 10개의 데이터셋에서 진행된 광범위한 실험을 통해 그 강력한 적응성과 일반화 능력을 입증하였습니다. 이 모델은 질병 진단 및 치료 계획에서 효율성과 정확성을 향상시키며, 수작업 주석 부담을 줄이는 데 기여합니다. 모델의 성능은 특히 복잡한 의료 이미징 시나리오에서 더욱 두드러지는 것으로 나타났습니다.



### Actions and Objects Pathways for Domain Adaptation in Video Question Answering (https://arxiv.org/abs/2411.19434)
- **What's New**: 이번 논문에서는 Video QA(질문-답변)에서의 out-of-domain 일반화를 위한 새로운 방법론인 Actions and Objects Pathways (AOPath)를 제안합니다. AOPath는 대규모 pretrained 모델에서 추출한 기능을 활용하여 미세 조정 없이 일반화 능력을 향상시킵니다. 이는 인간의 뇌 구조에서 영감을 받아, pretrained 기능을 행동(feature)과 객체(feature)로 분리한 후 각기 다른 reasoning pathways를 통해 처리합니다.

- **Technical Details**: AOPath는 AOExtractor 모듈을 사용하여 도메인에 구애받지 않는 기능으로 변환하는 혁신적인 접근 방식을 채택하여 추가적인 학습 가능한 가중치를 도입하지 않습니다. 또한, 수행 과정에서 각 모달리티의 행동 특징과 객체 특징을 분리하여 추출하고, 이들 각각에 대해 별도의 reasoning pathway를 생성합니다. 제안된 방법론은 TVQA 데이터셋을 기준으로 평가되며, 이 데이터셋은 여러 장르로 구분되어 있습니다.

- **Performance Highlights**: AOPath는 기존의 분류기와 비교해 out-of-domain 데이터셋에서 5%, in-domain 데이터셋에서 4%의 성능 향상을 보여주었습니다. 또한 작은 버전인 AOPath_S는 훈련 가능한 가중치가 26,282개에 불과함에도 불구하고 이전의 방법들에 비해 3% 성능 향상을 이끌어 냈습니다. 이 접근법은 상대적으로 적은 매개변수로 높은 일반화 성능을 달성하였습니다.



### Any-Resolution AI-Generated Image Detection by Spectral Learning (https://arxiv.org/abs/2411.19417)
- **What's New**: 이 논문에서는 AI가 생성한 이미지의 탐지 분야에서 중요한 기여를 제공합니다. AI 모델이 생성한 이미지에 나타나는 스펙트럼 아티팩트를 포착하기 위해 스펙트럴 분포를 모델링하고 학습하는 새로운 방식을 배워 도입합니다. 이를 통해 생성된 이미지의 탐지 성능이 크게 향상되고, 다양한 생성 모델에서의 일반화 능력이 증가합니다.

- **Technical Details**: 저자들은 주파수 재구성을 프리텍스트 태스크(pretext task)로 활용하여, 실제 이미지의 스펙트럴 분포를 모델링합니다. 그들은 생성된 이미지가 이 모델의 아웃-오브-디스트리뷰션(out-of-distribution) 샘플로 간주될 수 있다는 점을 강조합니다. 또한, 스펙트럴 재구성 유사성(spectral reconstruction similarity) 개념을 도입하여 재구성된 주파수와 실제 이미지 간의 차이를 측정합니다.

- **Performance Highlights**: SPAI 접근법은 13개의 최근 생성 접근법에 대해 기존 최첨단 기술보다 5.5%의 절대적인 성능 향상을 달성했습니다. 이 모델은 또한 여러 온라인의 일반적인 교란에 대해 뛰어난 내성을 보이며, 스펙트럼 컨텍스트 어텐션(spectral context attention) 기법을 활용하여 고해상도 이미지에서도 세밀한 스펙트럼 불일치를 효과적으로 포착할 수 있습니다.



### AMO Sampler: Enhancing Text Rendering with Overshooting (https://arxiv.org/abs/2411.19415)
Comments:
          17 pages

- **What's New**: 논문에서는 텍스트-이미지 생성에서 텍스트 렌더링 품질을 향상시키기 위한 새로운 방법론을 소개합니다. 특히, 사전 학습된 rectified flow (RF) 모델을 위한 overshooting sampler를 도입하여, 텍스트의 정확한 묘사를 가능하게 합니다. 이 과정에서 Attention Modulated Overshooting sampler (AMO)를 활용해 각 이미지 패치의 텍스트 내용에 따른 주의 점수에 기초하여 overshooting 강도를 조절합니다.

- **Technical Details**: Overshooting sampler는 이론적으로 학습된 Ordinary Differential Equation (ODE)을 과시뮨하여 노이즈를 재도입하는 방식으로 작동합니다. Langevin dynamics 항을 추가하여 오류 누적을 보정함으로써 텍스트 렌더링 품질을 높입니다. AMO는 각 이미지 패치의 text content에 대한 attention score에 따라 overshooting 강도를 적응적으로 조절하여, 더 높은 정확도의 텍스트 생성을 가능하게 합니다.

- **Performance Highlights**: AMO는 최신 RF 기반 모델인 Stable Diffusion 3 (SD3)와 Flux에서 각각 32.3% 및 35.9%의 텍스트 렌더링 정확도 향상을 달성했습니다. 이러한 성능 개선은 이미지의 전체 품질이나 추론 비용을 증가시키지 않는 상태에서 이루어졌습니다. 이는 다양한 AI 애플리케이션에서 텍스트-이미지 생성 모델의 유용성을 크게 높일 것으로 기대됩니다.



### DreamBlend: Advancing Personalized Fine-tuning of Text-to-Image Diffusion Models (https://arxiv.org/abs/2411.19390)
Comments:
          Accepted to WACV 2025

- **What's New**: 이 논문에서는 DreamBlend라는 새로운 접근 방식을 제안합니다. DreamBlend는 초기 체크포인트에서의 prompt fidelity와 다양성을 후속 체크포인트에서의 subject fidelity와 결합하여 향상된 이미지를 생성합니다. 이를 통해 다양한 컨텍스트에서 특정 주제의 개인화된 이미지를 생성하는 데 있어 기존 방법보다 월등한 성능을 보입니다.

- **Technical Details**: 체크포인트의 이미지 생성 동안, 교차 주의(cross attention)를 사용하여 초기 체크포인트에서 생성된 이미지를 기반으로 후속 체크포인트의 이미지를 안내합니다. 이 방식은 다양한 prompt에 대해 subject fidelity, prompt fidelity 및 다양성을 개선합니다. 이를 통해 overfitting 문제를 최소화하고 고급 이미지를 생성할 수 있습니다.

- **Performance Highlights**: 실험 결과, DreamBlend는 기존의 최첨단 fine-tuning 방법들보다 높은 학습 효율성과 함께 더 나은 이미지를 생성하는 것이 입증되었습니다. 실제로 난이도가 높은 다양한 프롬프트에 대한 주제 충실도 및 다양성을 나타내며, 사용자들이 요구하는 개인화된 이미지 생성에 매우 효과적입니다.



### Enhancing Sketch Animation: Text-to-Video Diffusion Models with Temporal Consistency and Rigidity Constraints (https://arxiv.org/abs/2411.19381)
- **What's New**: 최근 연구에서는 설명적 텍스트 프롬프트에 기반한 입력 스케치 애니메이션 기법이 제안되었습니다. 이 접근 방식은 스케치의 스트로크에 대한 파라메트릭 표현을 활용하여 정밀한 애니메이션을 가능하게 합니다. 전통적인 방법보다 매끄럽고 정확한 모션을 추정할 수 있으며, 스케치의 형태를 유지합니다. 이 연구는 최첨단 기법을 초월하는 성능을 보여줍니다.

- **Technical Details**: 제안하는 방법은 각 스트로크를 비지어 곡선(Bézier curve)으로 표현하며, Length-Area (LA) 정규화 기법을 통해 시간적 일관성을 유지합니다. 또한, As-Rigid-As-Possible (ARAP) 손실을 적용하여 스케치 형태의 변형을 방지합니다. 이 방법은 이전 방법보다 더 나은 스케치-영상 일관성과 텍스트-영상 정렬을 달성합니다. SDS 손실을 이용해 애니메이션의 모션을 최적화합니다.

- **Performance Highlights**: 제안된 방법은 기존 연구들보다 더 높은 정량적 및 정성적 평가 성능을 기록하였습니다. 우리는 스케치 애니메이션에서 시간적 일관성과 로컬 강성을 유지하며, 스케치 형태를 보존하는 데 성공했습니다. 이 연구의 접근 방식은 직관적인 텍스트 입력만으로 고품질 애니메이션을 생성할 수 있다는 점에서 많은 혁신을 보여줍니다.



### Libra: Leveraging Temporal Images for Biomedical Radiology Analysis (https://arxiv.org/abs/2411.19378)
- **What's New**: Libra는 의료 이미지를 이용한 방사선 보고서 생성을 위한 정확한 시간 인식을 위한 새로운 다중 모달 대규모 언어 모델(MLLM)입니다. 기존 모델들이 단일 이미지 분석에 주로 집중했음에도 불구하고, Libra는 과거 연구와 비교하여 현재 이미지를 평가하는 데 필요한 시간적 정보를 통합했습니다. 이를 통해 방사선 보고서 생성(RRG)에서 전문가급 성능을 달성하며, 임상 실습의 중대한 격차를 해결하고자 합니다.

- **Technical Details**: Libra는 RAD-DINO라는 사전 훈련된 이미지 인코더와 Meditron이라는 의료 대규모 언어 모델(LLM)을 통합하여 구성되었습니다. 이 모델은 Temporal Alignment Connector(TAC)를 사용하여 다양한 시간 지점에서의 이미지의 시간 정보를 캡처하고 통합합니다. TAC는 이미지의 고세분화 레이어별 특징을 추출하는 Layerwise Feature Extractor(LFE)와 과거의 연구로부터 얻어진 시간 참조를 통합하는 Temporal Fusion Module(TFM)로 구성됩니다.

- **Performance Highlights**: Libra는 MIMIC-CXR 데이터셋에서 방사선 보고서 생성(RRG) 작업의 최첨단 성능을 달성했습니다. 특히, RadCliQ 메트릭에서 12.9%의 개선이 이루어졌으며, 모든 어휘 메트릭에서도 이전 모델들보다 상당한 향상을 보였습니다. 이러한 성과는 Libra가 의료 이미지를 효과적으로 분석하고 해석하는 능력을 갖추고 있다는 것을 증명합니다.



### CLIP meets DINO for Tuning Zero-Shot Classifier using Unlabeled Image Collections (https://arxiv.org/abs/2411.19346)
- **What's New**: 이번 논문에서는 CLIP의 이미지 분류 성능을 개선하기 위해 DINO 기반의 라벨 없는 프롬프트 튜닝 기법인 NoLA (No Labels Attached)를 제안합니다. 이 방법은 자가 지도 학습(self-supervised learning) 모델의 시각적 특징과 대형 언어 모델(large language models)의 텍스트적 지식을 결합하여, 라벨이 없는 이미지를 사용해 분류 성능을 크게 향상시킵니다.

- **Technical Details**: NoLA는 세 가지 주요 단계로 구성됩니다: 첫째, LLM의 클래스 특성 설명을 활용해 강력한 텍스트 특징 임베딩을 생성합니다. 둘째, 이러한 텍스트 임베딩을 사용하여 DINO 모델의 시각적 특징과 통합된 Alignment 모듈을 훈련시키기 위한 의사 라벨(pseudo-labels)을 생성합니다. 마지막으로, 훈련된 Alignment 모듈을 이용해 CLIP의 비전 인코더를 DINO의 도움으로 프롬프트 튜닝합니다.

- **Performance Highlights**: 이 방법은 11개의 다양한 이미지 분류 데이터셋에서 평균 3.6%의 성능 향상을 보여주었으며, 기존의 최첨단 라벨 없는 분류 방법을 초월하는 효율적 접근방식을 제공합니다. NoLA는 특히 극소량의 라벨이 필요한 상황에서도 우수한 성능을 발휘하여, 미세 조정이 필요 없이 다양한 작업에서 뛰어난 유연성을 보여줍니다.



### Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation (https://arxiv.org/abs/2411.19331)
- **What's New**: 이 논문에서는 Open-Vocabulary Segmentation (OVS) 문제를 해결하기 위한 새로운 접근 방식인 Talk2DINO를 제안합니다. Talk2DINO는 DINOv2의 공간적 정확성과 CLIP의 언어 이해 능력을 결합하여 이미지와 텍스트 간의 상호작용을 향상시킵니다. 이 방법은 하부 네트워크의 미세 조정 없이 CLIP의 텍스트 임베딩을 DINOv2의 패치 수준 피쳐에 매핑하는 학습된 함수를 통해 이뤄집니다.

- **Technical Details**: Talk2DINO는 DINOv2의 자기 주의(attention) 맵을 활용하여 시각적 패치를 텍스트 임베딩과 선택적으로 정렬합니다. 학습 시, 이 메커니즘은 다양한 의미(region)의 시각적 특징을 강조하여 이미지의 세분화(segmentation) 품질을 높입니다. 데이터 수집 과정 없이 새로운 매핑 기능을 배우는 이 방식은 기존 CLIP 기반 모델의 한계를 극복하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, Talk2DINO는 다양한 비지도 OVS 벤치마크에서 최첨단 성능을 달성하였으며, 텍스트와 이미지 간의 의미적 상관성을 효과적으로 증대시킵니다. 이 방법은 배경 객체를 효과적으로 분리할 수 있도록 하여 더 자연스럽고 노이즈가 적은 분할 결과를 제공합니다. 논문은 Talk2DINO가 CLIP 유사 모델의 공간적 이해 한계를 해결할 수 있는 새로운 길을 열었다고 강조합니다.



### GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks (https://arxiv.org/abs/2411.19325)
- **What's New**: 이 논문에서는 지리공간 애플리케이션에 특화된 VLM(비전-언어 모델) 평가를 위한 GEOBench-VLM이라는 종합 벤치마크를 소개합니다. 기존의 일반적인 VLM 벤치마크들은 지리공간 데이터의 복잡성을 다룰 수 없는 한계를 가지고 있습니다. GEOBench-VLM은 장면 이해, 객체 계수, 위치 지정, 세분화된 분류 및 시간 분석과 같은 지리공간 작업을 평가하는 데 초점을 맞추고 있습니다.

- **Technical Details**: GEOBench-VLM은 10,000개 이상의 수동 확인 지침을 포함하여 다양한 시각 조건과 객체 유형, 범위를 포괄하는 데이터셋을 특징으로 합니다. 벤치마크는 다양한 VLM의 정확성을 평가하기 위한 평가를 포함하고 있으며, 여러 최신 VLM의 성능을 비교하고 있습니다. 이 결과는 기존의 VLM이 지리공간 데이터 예시에 대한 도전에 직면하고 있음을 강조합니다.

- **Performance Highlights**: 최고 성능을 보인 GPT4o는 MCQ에서 40% 정확도로 무작위 추측 대비 두 배에 해당하는 성능을 보여주었으며, 이는 지리공간-specific 예제에서의 과제가 무엇인지 드러냅니다. 또한 LLaVA-OneVision은 객체 위치 지정에서 우수한 성능을 보였고, Qwen2-VL은 사건 탐지와 비광학 SAR 이미지를 해석하는 데 강점을 보였습니다. 이러한 비교 분석은 VLM을 지리공간 작업에서 분석할 필요성과 함께 개선할 영역을 식별하는 데 기여합니다.



### Trajectory Attention for Fine-grained Video Motion Contro (https://arxiv.org/abs/2411.19324)
Comments:
          Project Page: this http URL

- **What's New**: 이 논문에서는 비디오 생성에서 카메라 모션 제어의 도전 과제를 해결하기 위해 새로운 접근 방식인 trajectory attention을 도입합니다. 이 방법은 기존의 모션 모델이 가지는 높은 수준의 제약을 극복하고 더욱 정밀한 비디오 생성이 가능하게 합니다. 특히, 기존의 temporal attention을 보완하는 auxiliary branch로서 작용하여, 정밀한 모션 제어와 새로운 콘텐츠 생성 능력을 통합합니다.

- **Technical Details**: Trajectory attention은 비디오 생성 과정에서 사용할 수 있는 픽셀 경로를 따라 attention을 수행하여 카메라 모션 제어를 세밀하게 조정합니다. 이 방식은 기존의 temporal attention과 함께 작동하여, 각각의 attention 메커니즘이 가진 특징적 목표를 분리함으로써 더 나은 성능을 발휘합니다. 실험 결과는 카메라 모션 제어에서 정확성과 긴 범위의 일관성을 크게 향상시킴을 보여줍니다.

- **Performance Highlights**: 제안된 방법은 다양한 비디오 모션 제어 작업에 적용될 수 있으며, 특히 첫 프레임 가이드 비디오 편집에서 콘텐츠 일관성을 효과적으로 유지하는 성능을 보여줍니다. 실험을 통해 제한된 데이터와 계산 자원으로도 훈련이 가능하여 다양한 맥락과 프레임 범위에 일반화될 수 있음을 입증하였습니다.



### SAMa: Material-aware 3D Selection and Segmentation (https://arxiv.org/abs/2411.19322)
Comments:
          Project Page: this https URL

- **What's New**: 이 연구에서는 SAM2 비디오 선택 모델을 기반으로 하여 3D 자산의 소재 선택을 위한 Select Any Material(SAMa) 접근 방식을 소개합니다. 기존의 2D 이미지 도메인에서 이루어진 연구를 넘어, SAMa는 다양한 3D 표현에서 상호작용할 수 있는 효율적이고 정확한 소재 분할 및 선택 방법을 제공합니다. 이를 통해 사용자 클릭에 대한 연속적인 선택 마스크를 신속하게 재구성할 수 있으며, 이는 3D 작업의 편의성을 크게 향상시킵니다.

- **Technical Details**: SAMa는 희소한 뷰 세트로부터 포인트 클라우드 형태의 중간 소재 유사성 표현을 생성하고, 이 유사성 클라우드에서 최인접 이웃(nearest-neighbour) 검색을 통해 객체 표면 전체에 대한 정확한 선택 마스크를 생성합니다. 방법론은 멀티뷰 일관성을 고려하여 설계되었으며, 이를 통해 대조적 학습이나 특징 필드 전처리 없이도 몇 초 내에 선택 작업을 완수할 수 있습니다. 또한 모든 3D 표현에 적용 가능하며, 메쉬, 방사 필드 및 3D 가우시안에서 우수한 성능을 나타냅니다.

- **Performance Highlights**: SAMa는 선택 정확도 및 멀티뷰 일관성 측면에서 여러 강력한 기준 비교군을 초월하며, 빠른 선택 결과 시각화를 가능하게 합니다. 이 방법은 3D 형태의 물체에 대해 빠르고 효율적인 투영을 지원함으로써, 크로스 프레임 일관성 덕분에 3D 객체에 대한 선택을 2초 이내에 수행할 수 있도록 합니다. 연구는 물체 세분화, NeRF/가우시안 편집 및 다양한 응용 프로그램을 가능하게 하여, 실질적으로 물질적 질감의 조작 및 선택을 개선합니다.



### Enhancing Parameter-Efficient Fine-Tuning of Vision Transformers through Frequency-Based Adaptation (https://arxiv.org/abs/2411.19297)
Comments:
          24 pages

- **What's New**: 이 연구에서는 FreqFit이라는 새로운 Frequency Fine-tuning 모듈을 소개하여 비전 트랜스포머 모델의 적응성을 높이고자 합니다. 이 방법은 기존의 PEFT 메소드와 결합하여 모델의 성능을 크게 향상시킬 수 있습니다. FreqFit은 주파수 도메인에서 특징을 조작함으로써 보다 미세한 패턴을 효율적으로 캡처할 수 있는 장점을 가지고 있습니다.

- **Technical Details**: FreqFit은 ViT 블록 사이에 위치하여 Fast Fourier Transform (FFT)을 활용하여 공간 차원에서 특징을 주파수 도메인으로 변환합니다. 그런 다음 학습 가능한 필터를 사용하여 스펙트럼을 조절하고, 이를 역 FFT (iFFT)를 통해 다시 공간 도메인으로 전환하여 원래 입력에 잔여 연결로 더합니다. 이러한 과정은 기존의 PEFT 방법의 구조적 제약을 보완하는 실용적인 솔루션을 제공합니다.

- **Performance Highlights**: 연구결과, FreqFit은 24개의 다양한 데이터셋에서 기존의 PEFT 방법에 비해 1%에서 16%까지 성능 향상을 보여주었습니다. 예를 들어, FreqFit-LoRA는 CIFAR100에서 최첨단 기준 성능 대비 10% 이상의 개선을 달성하였으며, 이는 강한 정규화나 증강 없이 이루어졌습니다. 이러한 성과는 FreqFit이 모델 적응성에 크게 기여할 수 있음을 잘 보여줍니다.



### UrbanCAD: Towards Highly Controllable and Photorealistic 3D Vehicles for Urban Scene Simulation (https://arxiv.org/abs/2411.19292)
Comments:
          Project page: this https URL

- **What's New**: UrbanCAD는 자동으로 3D 차량 디지털 트윈을 생성하는 프레임워크를 소개하여 포토리얼리스틱(photorealistic)과 제어 가능성(controllability) 간의 균형을 유지합니다. 이 프레임워크는 단일 도시 이미지와 무료 3D CAD 모델, 손으로 제작된 재료의 집합으로부터 고도로 제어 가능한 3D 차량을 생성합니다. 생성된 디지털 트윈은 사실적인 360도 렌더링, 차량 삽입, 재료 전환, 재조명 및 구성 요소 조작을 지원하여 드문 시나리오 구축을 가능하게 합니다.

- **Technical Details**: UrbanCAD는 데이터 검색(retrieval)과 최적화(optimization)를 통해 작동하는 새로운 파이프라인을 제안합니다. 이 과정은 관찰 데이터를 기반으로 하며 유연한 제어성과 세밀한 세부 사항을 유지합니다. 이 방법은 다중 시점의 배경 관점을 통해 물리 기반 미분 가능 렌더링을 활용하고, 3D Gaussian Splatting을 통해 최적화된 CAD 모델을 사실적으로 렌더링합니다.

- **Performance Highlights**: 실험 결과 UrbanCAD는 포토리얼리스틱 렌더링에서 기존의 복원(based on reconstruction) 및 검색(based on retrieval) 방법들을 초월하는 성능을 보였습니다. 다양한 감지 모델이 UrbanCAD로 생성된 데이터에 대해 정확도를 유지하는 반면, 실제 데이터에 대해 적용 시 성능 저하를 나타내었습니다. 이는 UrbanCAD가 자율주행의 안전에 중요한 드문 시나리오 생성에서 중대한 발전을 이루었음을 시사합니다.



### SADG: Segment Any Dynamic Gaussian Without Object Trackers (https://arxiv.org/abs/2411.19290)
Comments:
          Project page this https URL

- **What's New**: SADG(Segment Any Dynamic Gaussian Without Object Trackers)는 동적 3D 장면의 세그멘테이션을 위한 새로운 접근법입니다. 이 방법은 객체 ID에 의존하지 않고 동적 Gaussian Splatting 표현과 의미 정보를 결합합니다. 또한, Segment Anything Model(SAM)로 생성된 마스크를 활용하여 데이터 기반의 세멘틱 요소를 진행합니다.

- **Technical Details**: SADG는 하드 피크셀 마이닝에 기반한 새로운 대조적 학습 목표를 정의합니다. 이를 통해 모델은 신속한 렌더링과 최소한의 후처리로 움직이는 객체의 세밀한 세그멘테이션을 수행할 수 있습니다. 이 방법은 32차원 compact 기능을 학습하며, 2D 세그멘테이션 모델의 기능에 의존하지 않도록 설계되었습니다.

- **Performance Highlights**: SADG는 제안된 벤치마크에서 탁월한 성능을 보여주며, 단일 뷰와 다중 뷰 장면에서 동적 세그멘테이션을 포함한 다양한 후속 작업을 효과적으로 처리합니다. 이 시스템은 객체 스타일 변환, 재색칠, 구성 및 제거와 같은 다양한 편집 작업에 사용될 수 있으며, 사용자 상호작용에 기반한 실시간 편집을 지원합니다.



### GMS-VINS:Multi-category Dynamic Objects Semantic Segmentation for Enhanced Visual-Inertial Odometry Using a Promptable Foundation Mod (https://arxiv.org/abs/2411.19289)
- **What's New**: GMS-VINS(Generalized Multi-Category Segmentation Visual-Inertial Navigation System)는 이동 물체가 많은 환경 속에서의 포즈 추정 정확도를 높이기 위해 향상된 SORT 알고리즘과 다중 카테고리 세분화 프레임워크를 통합한 혁신적인 비주얼 관성 항법 시스템입니다. 기존의 접근 방식은 동적인 물체를 잘 처리하지 못했으나, GMS-VINS는 다양한 동적 객체를 효과적으로 추적하고 분리하여 기존 방법의 한계를 극복합니다.

- **Technical Details**: 이 논문에서는 향상된 SORT(Simple Online and Real-Time Tracking) 알고리즘을 사용하여 복잡한 도시 환경에서 부분 장애물이 있을 때도 안정적으로 이동 물체를 추적할 수 있는 방법을 제시합니다. 또한, Promptable Foundation Model인 Segment Anything Model (SAM)을 사용하여 다양한 카테고리의 동적 객체에 대한 세분화 작업을 수행하였습니다. 이로 인해 특정 사례에 대한 재훈련 없이도 다양한 동적 환경에 대한 적응력을 향상시킬 수 있습니다.

- **Performance Highlights**: GMS-VINS는 다양한 공개 데이터 세트와 실제 시나리오에 대한 광범위한 실험을 통해 제안된 방법이 여러 시나리오에서 뛰어난 성능을 발휘하며 최신 방법들보다 더 나은 포즈 추정 정확도를 보여주는 것을 확인했습니다. 실험 결과는 GMS-VINS가 다양한 동적 객체를 효과적으로 처리할 수 있는 가능성을 보여주며, 실제 응용 프로그램에서의 활용 가능성을 강조합니다.



### OMNI-DC: Highly Robust Depth Completion with Multiresolution Depth Integration (https://arxiv.org/abs/2411.19278)
- **What's New**: 이번 논문에서는 Depth Completion (DC) 문제에 대한 새로운 모델 OMNI-DC를 제안합니다. OMNI-DC는 다양한 깊이 패턴에서도 뛰어난 일반화 성능을 보여주며, 기존의 DC 모델들이 갖는 한계를 극복하기 위해 다중 해상도 깊이 통합층과 확률 기반 손실 함수를 도입했습니다. 또한, 합성 데이터셋을 활용한 혼합 훈련 전략과 새로운 평가 프로토콜 Robust-DC를 통해 모델의 성능을 검증합니다.

- **Technical Details**: OMNI-DC는 고해상도 및 저해상도 데이터에 대해 명시적으로 깊이 관계를 모델링할 수 있는 다중 해상도 차별 가능한 깊이 통합층(Multi-res DDI)을 기반으로 합니다. 이는 깊이 통합 과정에서의 오류 축적을 방지하기 위해 설계되었습니다. 또한, 기존의 L1 손실 대신 확률 기반의 Laplacian 손실을 통합하여 모델이 깊이 불확실성을 더 잘 포착하도록 합니다. 모델은 5개의 대규모 합성 데이터셋에서 훈련되어 실제 벤치마크로의 일반화 능력을 보여줍니다.

- **Performance Highlights**: OMNI-DC는 Robust-DC 및 VOID 데이터셋에서 모든 벤치마크를 초과하는 성능을 보여줍니다. ETH3D의 야외 스플릿에서 MAE=0.312를 달성했으며, 이는 두 번째로 우수한 방법 G2-MD에 비해 59% 감소한 수치입니다. KITTI에서의 성능도 높아, 다른 KITTI 훈련 모델들보다 더 뛰어난 성과를 기록하였습니다. 이러한 결과는 OMNI-DC가 다양한 장면 및 깊이 패턴에서도 일관된 성능을 보여주는 DC 모델임을 입증합니다.



### On-chip Hyperspectral Image Segmentation with Fully Convolutional Networks for Scene Understanding in Autonomous Driving (https://arxiv.org/abs/2411.19274)
- **What's New**: 이 연구는 운전 씬에서의 다양한 물체의 near infrared (NIR) 스펙트럴 반사율을 활용하여 ADAS의 객체 세분화(segmentation) 개선을 목표로 하며, 실시간으로 처리할 수 있는 작은 크기의 스냅샷 하이퍼스펙트럴 카메라를 사용하는 가능성을 탐구합니다. 기존의 RGB 이미지는 메타머리즘(metamerism) 문제로 인한 객체 식별의 어려움이 있으나, 하이퍼스펙트럴 이미지는 더 많은 정보를 제공하여 이러한 문제를 해결할 수 있도록 합니다.

- **Technical Details**: 연구에서는 HSI-Drive 1.1 데이터셋을 기반으로 스펙트럴 분류 알고리즘에 대한 다양한 실험을 수행하며, 표준 tiny 완전 합성곱 신경망(FCN)이 하이퍼스펙트럴 이미지 분할 성능을 어떻게 개선할 수 있는지를 분석합니다. 본 연구에서 제시된 FCN 모델들은 메모리 사용과 처리 시간을 최적화하여 효율적인 세분화 성능을 보여줍니다.

- **Performance Highlights**: 제안된 HSI 세분화 프로토타입은 다양한 임베디드 컴퓨팅 플랫폼에서 메모리 풋프린트(memory footprint), 대기 시간(latency), 전력 및 에너지 소비 측면에서 성능이 평가되었습니다. 실험 결과, 경량 FCN 모델이 고전적인 스펙트럴 분류기보다 우수한 성능을 보여주며, 실시간 ADAS 구현에 적합하다는 것을 확인하였습니다.



### AGS-Mesh: Adaptive Gaussian Splatting and Meshing with Geometric Priors for Indoor Room Reconstruction Using Smartphones (https://arxiv.org/abs/2411.19271)
- **What's New**: 본 연구에서는 indoor scene의 3D 재구성을 위한 새로운 접근 방식을 제안합니다. 저해상도 depth 추정값과 상용 모노큘러 기하학 추정값을 적응형으로 결합하여 고충실도의 재구성을 돕는 정규화 전략을 개발했습니다. 이를 통해 이전 방법들에 비해 novel-view synthesis와 geometry extraction 모두에서 개선된 결과를 도출할 수 있었습니다. 또한, 전통적인 메쉬 재구성 기법들보다 더 세밀한 기하학적 형태 추출이 가능해졌습니다.

- **Technical Details**: 우리는 Low-resolution depth maps와 Pretrained monocular networks를 활용하여 새로운 정규화 전략인 Depth Normal Consistency(DNC)를 도입했습니다. DNC는 음영 추정치간의 일관성을 고려하여 depth map의 불일치를 필터링하는 기법입니다. 또한, Adaptive Normal Regularization(ANR)을 통해 노이즈가 많은 영역에서의 정규화를 완화하여 보다 정확한 정규화를 수행합니다. 이러한 기술들은 Gaussian Splatting 기반 방법의 3D와 2D 분석에서 성능 개선을 보여줍니다.

- **Performance Highlights**: 우리는 실험을 통해 제안한 방법이 indoor room reconstruction 에서 예측 정확도를 높였음을 입증했습니다. 3D Gaussian Splatting을 적용하여 메쉬 추출에서 더욱 섬세한 세부사항 복구를 가능하게 했습니다. Tsdf 및 IsoOctree meshing에 기반한 포스트프로세싱 기법을 통해 기존 방법들보다 뛰어난 결과를 보여 주었으며, photorealism을 높이고 촉각적 힘을 보강하는 데 기여했습니다.



### Improving Multi-Subject Consistency in Open-Domain Image Generation with Isolation and Reposition Attention (https://arxiv.org/abs/2411.19261)
- **What's New**: 이 논문은 여러 주제가 포함된 이미지 생성에서의 일관성을 높이기 위한 최신 방법인 IR-Diffusion을 소개합니다. 기존 방법들이 서로 간섭하는 문제와 위치 차이로 인한 불일치를 간과한 반면, IR-Diffusion은 Isolation Attention과 Reposition Attention을 통해 이러한 문제를 해결하고 있습니다. 특히, 각 주제가 서로 인용되지 않도록 하여 주제 중복(multi-subject fusion)을 방지합니다.

- **Technical Details**: IR-Diffusion에는 두 가지 주요 메커니즘이 포함되어 있습니다. 첫째, Isolation Attention은 각 주제가 독립적으로 작동하도록 보장하여 서로 간섭하는 것을 감소시킵니다. 둘째, Reposition Attention은 참조 이미지에서의 주제의 위치를 조정하여 대상 이미지와의 정렬을 최적화합니다. 이러한 조치들은 이미지 생성 과정에서 참조 정보를 보다 효과적으로 활용할 수 있게 합니다.

- **Performance Highlights**: IR-Diffusion은 open-domain 시나리오에서 다른 모든 기존 방법보다 월등한 성능을 발휘하며, 다수의 주제를 포함하는 일관된 이미지를 생성하는 데 성공했습니다. 광범위한 실험을 통해 제안된 방법들이 다중 주제 일관성(multi-subject consistency)을 높이고, 이미지 품질을 개선함을 입증하였습니다.



### Face2QR: A Unified Framework for Aesthetic, Face-Preserving, and Scannable QR Code Generation (https://arxiv.org/abs/2411.19246)
- **What's New**: 이번 논문에서는 Face2QR이라는 새로운 파이프라인을 제안하여, 개인화된 QR 코드를 생성하는 혁신적인 접근 방식을 소개합니다. 이 파이프라인은 얼굴 ID, 미적 품질 및 QR 코드의 스캔 가능성을 조화롭게 통합하여, 사용자 맞춤형 QR 코드를 생성합니다. 세 가지 주요 구성 요소인 ID-refined QR integration (IDQR), ID-aware QR ReShuffle (IDRS), 및 ID-preserved Scannability Enhancement (IDSE)를 통해 기존 방법들의 한계를 극복하고 있습니다.

- **Technical Details**: Face2QR 파이프라인은 세 단계를 거쳐 작동합니다. 첫 번째 단계에서는 IDQR 모듈을 통해 초기 QR 이미지를 생성하며, 이때 Stable Diffusion (SD) 기반의 모델과 InstantID 네트워크를 활용합니다. 두 번째 단계에서는 얼굴 ID와 QR 코드 패턴 간의 갈등을 해결하기 위해 IDRS를 사용하고, 마지막으로 IDSE를 통해 스캔 강도를 향상시키는 최적화 과정을 거칩니다.

- **Performance Highlights**: Face2QR은 기존 방법들에 비해 우수한 성능을 입증했습니다. 특히, 사용자 맞춤형 QR 코드 디자인 내에서 얼굴 인식 기능을 잘 보존하면서도 미적 품질을 유지하는 데 성공했습니다. 실험 결과, Face2QR은 ID-preserved aesthetic QR 생성에서 State-Of-The-Art (SOTA) 성능을 달성했습니다.



### InstanceGaussian: Appearance-Semantic Joint Gaussian Representation for 3D Instance-Level Perception (https://arxiv.org/abs/2411.19235)
Comments:
          technical report, 13 pages

- **What's New**: 이 논문에서는 3D 장면 이해를 위한 새로운 방법론인 InstanceGaussian을 제안합니다. 이 방법은 외관(appearance)과 의미(semantic) 특성을 동시에 학습하고 개체를 적응적으로 집합화하는 데 중점을 두고 있습니다. 특히, Semantic-Scaffold-GS 표현을 통해 외관과 의미 간의 균형을 이루고, 점진적인 공동 학습 전략을 통해 안정성과 세분화 정확도를 향상시킵니다. 

- **Technical Details**: InstanceGaussian은 세 개의 주요 기여를 통해 3D 장면 이해의 도전 과제를 해결합니다. 첫째, Semantic-Scaffold-GS 표현을 통해 다양한 표현 속성을 조정함으로써 객체의 기하학적 경계를 더 정확하게 학습할 수 있게 합니다. 둘째, 점진적인 외관-의미 공동 학습 전략을 도입하여 학습 과정에서 외관과 의미의 일관성을 유지합니다. 셋째, 최하향 방식의 카테고리 비독립적인 인스턴스 집합화 접근 방식을 통해 과다 또는 과소 세분화 문제를 해결합니다.

- **Performance Highlights**: 이 접근 방식은 카테고리 비독립적이며 개방된 어휘 3D 점 수준 세분화에서 최첨단 성능을 달성합니다. 제안된 표현 및 학습 전략의 효과성을 강조하며, 3D 장면 이해의 향상된 성능을 보여줍니다. 이러한 방법론은 자율 주행, 로봇 공학 및 증강 현실과 같은 다양한 응용 분야에 모두 유용한 기반을 제공합니다.



### Gaussians-to-Life: Text-Driven Animation of 3D Gaussian Splatting Scenes (https://arxiv.org/abs/2411.19233)
Comments:
          Project website: this https URL

- **What's New**: 이번 논문에서는 정적인 3D 씬에 생명을 불어넣기 위한 새로운 방법, Gaussians2Life를 제안합니다. 이 방법은 고품질 3D 씬의 일부를 애니메이션화하는 데 중점을 두고 있으며, 기존 비디오 확산 모델(video diffusion models)을 활용하여 생생한 애니메이션을 가능하게 합니다. 작업은 사용자가 정의한 텍스트 프롬프트와 대상을 포함한 바운딩 박스를 기반으로 3DGS 씬을 애니메이션화하는 것입니다.

- **Technical Details**: 제안된 방법은 정적인 장면에 대해 VDM(Video Diffusion Model)을 통한 다중 뷰 일관성이 있는 비디오 가이드를 생성합니다. 본 연구에서는 2D 비디오 애니메이션을 3D 모션으로 전환하는 기술을 사용하여 3DGS 프리미티브를 실제적인 3D 모션으로 변환합니다. 이를 통해 정적인 3DGS 씬을 다중 뷰에서 일관되게 애니메이션화할 수 있는 방법론을 제공합니다.

- **Performance Highlights**: 실험 평가를 통해 MipNeRF360 및 Instruct-NeRF2NeRF 데이터셋을 사용하여 제안된 모델의 효과성을 입증하였습니다. 기존의 단일 자산 애니메이션 방식과는 달리, 이 모델은 다양한 객체 클래스 애니메이션을 가능하게 하여 몰입감 있는 3D 경험을 창출합니다. 논문에서는 아키텍처 선택의 효과성을 나타내는 심층적인 삭제 실험(ablation study)도 제시하고 있습니다.



### Z-STAR+: A Zero-shot Style Transfer Method via Adjusting Style Distribution (https://arxiv.org/abs/2411.19231)
Comments:
          technical report

- **What's New**: 본 논문은 기존 스타일 전이 방법과는 다르게, 기본 diffusion 모델에서 지닌 자연스러운 스타일과 콘텐츠 분포를 활용하여 직접 스타일 정보를 추출하고 콘텐츠 이미지와 통합하는 새로운 접근 방식을 제안합니다. 이를 통해 재훈련 없이도 아티스틱한 스타일 전이를 가능하게 하며, 이는 기존의 기법들에서 보였던 스타일 표현의 제한을 극복하는 데 기여합니다. 또한, Cross-attention Reweighting 모듈을 도입하여 로컬 콘텐츠 특징을 이용하여 입력 패치에 가장 적합한 스타일 이미지 정보를 쿼리합니다.

- **Technical Details**: 본 연구는 dual denoising paths를 활용하여 콘텐츠 및 스타일 참고 이미지를 latent space에서 표현하고, 스타일 latent.codes가 콘텐츠 이미지의 denoising 프로세스를 안내하도록 설계되었습니다. 게다가, 스타일 이미지와 스타일화된 이미지 간의 색상 분포 불일치를 완화하기 위해 scaled adaptive instance normalization(SAIN) 기법을 적용합니다. 이 방법은 diffusion 모델의 잠재적인 스타일 및 콘텐츠 정보를 능동적으로 활용하여 전체적인 스타일 일치를 개선합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 원본 콘텐츠를 유지하면서도 스타일적인 요소를 풍부하게 통합하는 고품질 스타일 전이 결과를 생성하는 능력을 보여주었습니다. 또한, 이 방법은 비디오 스타일 전이 작업에도 유연하게 적용될 수 있으며, 여러 실험을 통해 스타일 분포와 콘텐츠 손실 간의 조화를 이루는 것으로 나타났습니다. 최종적으로, Z-STAR+는 기존 방식들과 비교하여 스타일 전이의 효과성과 우월성을 입증하였습니다.



### Automatic Prompt Generation and Grounding Object Detection for Zero-Shot Image Anomaly Detection (https://arxiv.org/abs/2411.19220)
Comments:
          Accepted to APSIPA ASC 2024

- **What's New**: 이 논문에서는 자동화된 산업 이미지 이상 탐지를 위한 새로운 제로샷(Zero-shot) 훈련 없는 접근법을 제안합니다. 이 방법은 세 가지 기초 모델로 구성된 멀티모달 기계 학습 파이프라인을 사용합니다. 특히, 대규모 언어 모델인 GPT-3를 활용하여 정상 및 비정상 제품의 예상 외형을 설명하는 텍스트 프롬프트를 생성하고, 이를 바탕으로 사물 탐지 모델인 Grounding DINO를 통해 이미지를 분석합니다.

- **Technical Details**: 제안된 방법은 언어 모델, 객체 탐지 및 제로샷 이미지-텍스트 매칭의 최근 발전을 활용하여 효율적이고 확장 가능한 품질 관리 시스템을 개발합니다. 텍스트 브랜치에서는 '정상' 및 '이상' 제품의 예상 외형을 설명하는 두 세트의 텍스트 프롬프트를 생성합니다. 이러한 프롬프트를 통해 추출된 제품 이미지를 생성된 프롬프트와 비교하여 이상 점수를 계산합니다.

- **Performance Highlights**: 실험 결과, MVTec-AD와 VisA라는 두 개의 산업 제품 이미지 데이터셋에서 높은 정확도로 다양한 결함 및 이상을 탐지하는 성과를 보였습니다. 본 방법은 모델 훈련 없이도 효율적이고 객관적인 품질 관리를 가능하게 하며, 기존의 최첨단 기술인 WinCLIP을 초월하는 성능을 보여주었습니다.



### Cross-Spectral Attention for Unsupervised RGB-IR Face Verification and Person Re-identification (https://arxiv.org/abs/2411.19215)
- **What's New**: 이번 연구에서는 RGB와 IR 스펙트럼 간의 생체인식 기법을 위한 새로운 비지도 크로스 스펙트럼 프레임워크를 제안합니다. 이 프레임워크는 (1) 크로스 스펙트럼 투표 기능을 통합한 새로운 pseudo triplet loss, (2) 여러 하위 공간을 활용한 크로스 스펙트럼 주의 네트워크, 그리고 (3) 구조적 희소성을 통해 더 효과적인 클러스터링을 수행하도록 설계되었습니다. 이 방법은 큰 스펙트럼 불일치를 해결하며, 기존의 감독 방식보다 더 나은 성능을 보여줍니다.

- **Technical Details**: 이 연구는 비지도 학습 환경에서 Re-ID와 Face Verification을 위한 크로스 스펙트럼 생체인식의 가능성을 탐구합니다. 이를 위해, 연구팀은 IR(적외선)과 RGB(가시광선) 이미지 간의 공통 정보를 효과적으로 추출하기 위한 신경망 아키텍처를 설계하였습니다. 또한 클러스터링과 투표 메커니즘을 결합하여 서로 다른 스펙트럼에서 관련된 샘플을 식별하고, 구조적 희소성을 통해 학습된 표현을 개선하는 접근 방식을 사용합니다.

- **Performance Highlights**: 연구 결과, 제안된 RGB-IR 생체인식 학습 프레임워크는 ARL-VTF와 RegDB 데이터셋에서 기존 최첨단 모델들과 비교했을 때 우수한 성능을 기록했습니다. 특히, 비지도 방법론에도 불구하고 특정 경우에는 전통적인 감독 학습 방법을 초과하는 성능을 보여주었습니다. 이는 다양한 운영 시나리오에 대한 감시 능력을 강화하는 데 기여할 것을 기대합니다.



### ANDHRA Bandersnatch: Training Neural Networks to Predict Parallel Realities (https://arxiv.org/abs/2411.19213)
Comments:
          New World!

- **What's New**: 이 연구는 Many-Worlds Interpretation (MWI)에서 영감을 받아 새로운 신경망 아키텍처를 소개합니다. 이 아키텍처는 각 레이어에서 동일한 입력 신호를 병렬 분기로 나누며, 이를 ANDHRA라는 Hyper Rectified Activation 을 사용합니다. 각 분기 레이어는 병합되지 않으며 별도의 네트워크 경로를 형성하여 여러 네트워크 헤드를 통한 출력 예측이 이루어집니다. ANDHRA 모듈을 사용하는 Bandersnatch 네트워크는 점진적으로 성장하며 훈련 중 추가 매개변수와 메모리를 요구합니다.

- **Technical Details**: ANDHRA 모듈은 주어진 입력 신호를 N개의 병렬 분기로 나누는 네트워크 모듈을 제안합니다. 신경망은 다양한 층에서 ANDHRA 모듈을 사용하여 네트워크 분기를 생성합니다. 각 레벨에서 분기 요소를 적용함으로써, 더 깊은 신경망 아키텍처에서도 그래디언트 전파 문제를 완화할 수 있습니다. 퍼포먼스 측정 및 비교는 CIFAR-10/100 데이터셋을 사용하여 자율적으로 행하고, 각 헤드의 손실 값을 통합하여 전체 네트워크를 공동 훈련합니다.

- **Performance Highlights**: CIFAR-10/100에 대한 실험 결과는 제안한 아키텍처가 기존의 기준 정확도를 유의미하게 초과했음을 보여줍니다. ANDHRA Bandersnatch 네트워크는 동일한 매개변수와 계산 비용으로도 성능 향상을 달성하였으며, 여러 헤드에서 발생하는 출력 예측은 통계적으로 유의미한 증대를 나타냅니다. 이러한 결과는 각 개별 헤드가 독립적으로 효과적인 디자인 결정을 통해 우수한 성능을 지향함을 뒷받침합니다.



### Track Anything Behind Everything: Zero-Shot Amodal Video Object Segmentation (https://arxiv.org/abs/2411.19210)
- **What's New**: TABE(Track Anything Behind Everything)라는 새로운 데이터셋, 파이프라인, 그리고 평가 프레임워크가 소개되었습니다. 기존의 방법들과 달리, TABE는 객체의 가시적 마스크를 바탕으로 첫 번째 프레임에서 단일 질의 마스크를 사용하여 유연하고 제로샷(zero-shot) 추론이 가능합니다. TABE-51 데이터셋은 인간의 추정이나 3D 재구성이 필요없이 매우 정확한 amodal segmentation 마스크를 제공합니다.

- **Technical Details**: 이 논문에서는 객체가 완전히 가려진 상황에서도 amodal completion을 처리할 수 있도록 설계된 TABE 파이프라인을 제안합니다. TABE는 전통적인 시각적 세분화 메트릭의 영향을 받지 않는 완료 성능을 평가하는 특별한 평가 프레임워크를 포함합니다. Amodal completion의 효율적인 처리를 위해, 다양한 기술적인 방법들이 논의되며, 특히 비디오 맥락에서의 중요성이 강조됩니다.

- **Performance Highlights**: TABE의 도입으로 복잡한 환경에서 객체 추적이 가능해지며, 이는 자율주행과 같은 중요한 작업에 큰 도움이 될 수 있습니다. 또한, amodal completion을 이용하여 객체 인식의 정확성을 높일 수 있다는 점도 강조되었습니다. 새로운 데이터셋과 평가 메트릭을 통해 모델의 amodal completion 성능을 분리하여 평가할 수 있는 방법이 마련되었습니다.



### Video Depth without Video Models (https://arxiv.org/abs/2411.19189)
- **What's New**: 본 논문은 RollingDepth라는 혁신적인 비디오 깊이 추정 모델을 제안합니다. 이 모델은 기존의 단일 이미지 라텐트 확산 모델을 사용하여 각 프레임에서 밀집 깊이를 추정하는 동시에 시간적인 일관성을 유지할 수 있도록 최적화된 등록 알고리즘을 활용합니다. 전통적인 비디오 깊이 추정 방식의 문제점을 해결하는 데 초점을 맞추고 있습니다.

- **Technical Details**: RollingDepth는 두 가지 주요 요소로 구성됩니다: (i) 단일 이미지 라텐트 확산 모델에서 파생된 다중 프레임 깊이 추정기로, 일반적으로 3개의 프레임으로 구성된 비디오 조각을 깊이 조각으로 매핑합니다. (ii) 다양한 프레임 속도에서 샘플링된 깊이 조각을 일관된 비디오로 조합하는 최적화 기반의 강력한 등록 알고리즘입니다. 이를 통해 긴 비디오에서도 일관된 깊이를 유지할 수 있습니다.

- **Performance Highlights**: RollingDepth는 수백 개의 프레임으로 구성된 긴 비디오에서도 효율적으로 처리할 수 있으며, 기존의 비디오 깊이 추정기와 고성능 단일 프레임 모델보다 더 정확한 깊이 비디오를 제공합니다. 최신 모델과 비교했을 때 제안된 방법은 시간적으로 일관된 깊이 추정의 정확성을 크게 향상시킵니다.



### SOWing Information: Cultivating Contextual Coherence with MLLMs in Image Generation (https://arxiv.org/abs/2411.19182)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 물리학에서의 확산 현상에서 영감을 받은 확산 생성 모델들이 이미지 합성을 위한 텍스트-비전-이미지 생성 작업(Media-Text to Image Generative modeling)에서의 정보 확산을 조정하는 방식을 제안합니다. Cyclic One-Way Diffusion (COW)와 Selective One-Way Diffusion (SOW)라는 두 가지 새로운 프레임워크를 도입해, 픽셀 수준의 조건 충실도를 달성하면서도 이미지 전반에 걸쳐 시각적 및 의미적 일관성을 유지합니다. 이러한 방법론들은 기존의 Diffusion 모델들이 가진 한계를 극복하고, 사용자 맞춤형 이미지 생성을 가능하게 합니다.

- **Technical Details**: COW 방법론은 효율적인 단방향 확산 프레임워크를 제공하여 정보 송수신의 정확성을 높이려 합니다. 여기서 MLLM(Multimodal Large Language Models)은 이미지 내 요소들의 관계를 분석하여 최적의 시각 조건을 결정하고, 동적인 주의 메커니즘(attention mechanism)을 통해 정보 확산의 방향과 강도를 조절합니다. 이를 통해 정보의 과도한 전파를 방지하고, 이미지 생성 과정에서 각 지역 간의 맥락적 관계를 유지할 수 있습니다.

- **Performance Highlights**: SOW는 600개의 512x512 이미지 그룹에 대한 1,200개의 응답을 포함한 실험에서 다른 방법들과 비교하여 조건의 일관성과 전반적인 충실도에서 지속적으로 더 높은 성과를 보였습니다. 이 방법은 5초 안에 이미지를 생성할 수 있으며, 기존의 DreamBooth와 같은 커스터마이징 방법들보다 현저히 빠른 속도를 제공합니다. 이러한 결과는 SOW가 더 유연하고 적응할 수 있는 생성 모델의 가능성을 열어준다는 것을 보여줍니다.



### HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos (https://arxiv.org/abs/2411.19167)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2406.09598

- **What's New**: 새롭게 소개된 HOT3D 데이터셋은 3D에서 개인 중심의 손과 객체 추적을 위한 공개 데이터셋입니다. 이 데이터셋은 19명의 주제가 33개의 다양한 단단한 객체와 상호작용하는 모습을 담은 833분 이상의 멀티 뷰 RGB/단색 이미지 스트림을 제공합니다. HOT3D는 손과 객체의 3D 포즈, 손과 객체의 3D 모델에 대한 상세한 주석을 포함하고 있으며, 주방, 사무실, 거실 등에서의 일반적인 행동을 반영한 시나리오로 구성되어 있습니다.

- **Technical Details**: HOT3D 데이터셋은 Meta의 Project Aria와 Quest 3라는 두 개의 최신 헤드 마운트 장치를 사용하여 기록되었습니다. 이 데이터셋은 3D 포즈 정보를 얻기 위해 소형 광학 마커를 부착한 손과 객체에 대해 전문적인 모션 캡처 시스템을 이용하였으며, 모든 스트림의 이미지가 동기화되어 멀티뷰 및 시간 정보를 활용한 방법 개발이 가능합니다. 손과 객체는 UmeTrack 및 MANO 형식으로 주석처리되어 있으며, PBR 재료를 사용한 3D 메시로 나타납니다.

- **Performance Highlights**: HOT3D의 실험 결과는 멀티뷰 방법이 단일 뷰 방법보다 명확하게 더 우수하는 성능을 보였습니다. 데이터셋을 통해 평가된 여러 작업에서, 3D 손 추적, 6DoF 객체 포즈 추정, 손에 있는 미지의 객체의 3D 리프팅 작업을 포함하여 강력한 기준선이 개발되었습니다. 이렇게 개발된 강력한 기준선은 증가하는 증강/혼합 현실 작업의 필요에 부합하며, 향후 연구에 많은 기여를 할 것으로 기대됩니다.



### Neural Shadow Ar (https://arxiv.org/abs/2411.19161)
Comments:
          10 pages

- **What's New**: 이번 연구에서는 Neural Shadow Art를 소개합니다. 이는 암시적 함수 표현(implicit function representation)을 활용하여 그림자 예술의 가능성을 확장하는 혁신적인 방법입니다. 이전의 접근 방식과는 달리, 우리의 방법은 조명 방향과 스크린 오리엔테이션을 최적화함으로써 입력 이진 이미지에 맞춰 프로젝션이 조정되도록 합니다.

- **Technical Details**: 우리는 3D 모델의 암시적 표현을 통해 조명 방향과 스크린 방향을 동시에 최적화하는 방법을 사용하였습니다. 이 과정에서는 입력 이미지의 강직 변형(rigid transformations)을 허용하며, 기하학적 매끄러움(geometric smoothness)과 볼륨 최적화(volume optimization)를 강화하였습니다. 이를 통해 적은 재료로 원하는 그림자 효과를 달성할 수 있습니다.

- **Performance Highlights**: 이 접근법은 복잡한 형태를 지닌 조각을 생성하는 데 우수한 성능을 보이며, 특히 현대 미술에서 예술적 효과를 극대화하는데 기여합니다. 우리는 또한 산업적 요구사항을 충족시키면서 재료 사용을 최소화하는 결과를 도출하였습니다. 최종적으로, 우리의 방법은 그림자 예술의 매력을 극대화하는데 중요한 기여를 하였습니다.



### LoRA of Change: Learning to Generate LoRA for the Editing Instruction from A Single Before-After Image Pair (https://arxiv.org/abs/2411.19156)
- **What's New**: 이 논문에서는 이미지 편집을 위한 시각적 지침을 활용한 LoRA of Change (LoC) 프레임워크를 제안합니다. 자연어의 모호성과 부족한 구체성을 극복하고, 이미지 쌍을 기반으로 사용자의 의도를 정확하게 반영할 수 있는 접근법을 제공합니다. 또한, LoRA Reverse 기법을 도입하여 쌍 데이터만으로 대규모 훈련이 가능하게 하였습니다.

- **Technical Details**: LoC 프레임워크는 '변화'를 인코딩하기 위해 특정 지침에 따라 동적으로 LoRA를 생성합니다. LoRA 생성은 하이퍼 네트워크 ℋ를 통해 이루어지며, 이를 통해 편집된 이미지 B′를 재구성할 수 있습니다. 기존 모델들에서 제한된 쿼드 데이터 문제를 해결하고 다양한 시각적 지침을 지원하기 위해 LoRA Reverse 최적화 기법이 포함되어 있습니다.

- **Performance Highlights**: 광범위한 정성적 및 정량적 실험을 통해 LoC 모델이 높은 품질의 이미지를 생성하며, 사용자의 의도에 부합하는 결과를 산출하는 것으로 나타났습니다. 또한 현업의 다양한 시각적 지침을 지원할 수 있는 능력이 입증되었으며, 이는 기존 방법들이 효율적으로 처리하지 못하던 부분입니다.



### Counting Stacked Objects from Multi-View Images (https://arxiv.org/abs/2411.19149)
Comments:
          13 pages

- **What's New**: 이번 연구에서는 기존의 3D 객체 집합 계산 방법의 한계를 극복하기 위해 새로운 3D 카운팅 방법을 제안했습니다. 이 방법은 멀티 뷰 이미지를 통해 객체 스택의 3D 기하학을 추정하고 점유 비율을 계산하는 두 가지 보완적인 하위 문제로 작업을 분해합니다. 이 접근 방식은 깊이 분석과 기하학적 재구성을 결합하여 비정형으로 쌓인 동일한 객체를 정확하게 카운트할 수 있도록 합니다.

- **Technical Details**: 제안된 3D 카운팅(3DC) 접근 방식은 개별 객체의 평균 부피와 전체 스택 부피를 활용하여 정확한 객체 수를 추정합니다. 객체는 기본적으로 균일하게 쌓여 있으며, 일부는 부분적으로 가시성이 있어 밀도 추정이 가능합니다. 이를 통해 모델은 제한된 시각적 단서만으로도 숨겨진 객체의 수를 추론할 수 있습니다.

- **Performance Highlights**: 3DC 방법은 다양한 실제 및 합성 데이터 세트를 통해 검증되었으며, 데이터 세트는 공적으로 제공될 예정입니다. 이 연구는 3D 카운팅 분야의 발전과 이를 바탕으로 한 엄격한 벤치마킹을 가능하게 합니다. 또한, 비정형 형태의 객체를 다루는 혁신적인 시도로서 산업 및 농업 분야에서의 필요를 크게 충족시킬 수 있습니다.



### Co-Learning: Towards Semi-Supervised Object Detection with Road-side Cameras (https://arxiv.org/abs/2411.19143)
Comments:
          Accepted at EAmSI24: Edge AI meets swarm intelligence

- **What's New**: 최근 딥러닝(deep learning)의 급속한 확장은 효율적인 반지도 학습(semi-supervised learning, SSL) 방법론의 발전에 큰 기여를 했습니다. 그러나 실제 환경에서 레이블 데이터의 수집은 비용이 많이 들고 노동 집약적이며, 때로는 부족한 경우도 발생합니다. 이러한 문제를 해결하기 위해, 연구진은 교사-학생 기반의 SSL 프레임워크인 Co-Learning을 개발하여 물체 탐지(object detection) 작업에서 효율성을 높이는 것을 목표로 하고 있습니다.

- **Technical Details**: Co-Learning은 10%의 레이블이 있는 데이터와 90%의 비레이블 데이터(unlabeled data)를 활용하여 반지도 학습을 수행합니다. 이 방법은 교사 네트워크(teacher network)와 학생 네트워크(student network) 간의 상호 학습(mutual learning) 및 레이블 정렬(annotation alignment) 전략을 통해 레이블 불일치(label inconsistency) 문제를 해결하고, 보다 일관된 의사 레이블(pseudo-labels)을 제공하여 전체 물체 탐지 성능을 향상시킵니다. 이러한 접근은 특히 엣지 디바이스(edge devices)에서 기계 학습 모델의 효율적인 훈련을 가능하게 합니다.

- **Performance Highlights**: 실험 결과 Co-Learning 프레임워크는 레이블 데이터의 양이 적음에도 불구하고 완전 지도 학습(full supervision) 솔루션과 유사한 성능을 달성했습니다. 이는 SSL의 장점을 통해 레이블링 비용을 절감하면서도 신뢰할 수 있는 물체 탐지 성능을 확보할 수 있음을 시사합니다. 또한, 이 연구는 자율 주행 모델의 도시 환경에서의 적용 가능성을 높이며, 기존 SSOD 방법의 한계를 극복하기 위한 중요한 발판이 될 것입니다.



### On Moving Object Segmentation from Monocular Video with Transformers (https://arxiv.org/abs/2411.19141)
Comments:
          WICCV2023

- **What's New**: 본 논문은 단일 이동 카메라에서의 움직이는 물체 탐지 및 분할을 위한 새로운 융합 아키텍처인 M3Former를 제안합니다. 이 아키텍처는 세분화(segmentation) 및 다중 모달 융합(multi-modal fusion)을 위한 강력한 성능을 발휘하는 Transformers를 활용합니다. 연구에서는 모노큘러 비디오에서의 움직임을 재구성하는 데 있어 2D 및 3D 움직임 표현의 중요성을 분석하고, 다양한 훈련 데이터의 필요성을 보여줍니다.

- **Technical Details**: M3Former는 Appearance와 Motion 피처를 결합한 두-stream 아키텍처를 사용하여, 서로 다른 모션 표현(Optical Flow, Scene Flow 등)을 효과적으로 분석합니다. Frozen expert models를 이용해 다양한 모션 정보를 계산하며, 간단한 데이터 증강 기법을 통해 모달 간 정렬을 개선합니다. 이렇게 만들어진 모델은 독립적으로 움직이는 객체를 탐지하고 분할하는 데 집중합니다.

- **Performance Highlights**: 실험 결과, 이 프레임워크는 Kitti와 Davis 데이터셋에서 최첨단(SotA) 성능 도달을 위한 다양한 데이터 세트의 필요성을 입증합니다. 특히, 다양한 훈련 데이터를 활용함으로써 실제 비디오에서 강력한 성능을 달성하는 것이 가능하다는 것을 보여줍니다. 전체적으로, 제안된 M3Former는 다중 모드 정보의 효율적인 융합을 통해 모션 세분화의 기준을 새롭게 설정합니다.



### MSG score: A Comprehensive Evaluation for Multi-Scene Video Generation (https://arxiv.org/abs/2411.19121)
- **What's New**: 이번 논문은 연속적인 시나리오에 기반한 다중 장면 비디오 생성을 위한 평가 지표를 다룹니다. 전통적인 짧은 비디오 생성과는 달리, 시나리오 기반 비디오는 캐릭터 일관성, 예술적 일관성, 심미적 품질 등을 고려해야 합니다. 저자들은 자동화된 점수 기반 평가 기준을 제안하여 더 객관적이고 효율적인 평가를 가능하게 합니다.

- **Technical Details**: MSG는 즉각적인 이웃 프레임에 대한 양방향 프레임 참조(Backward and Forward Frame Reference, BFFR)와 이전 장면의 주요 프레임을 참조하는 후향 장면 참조(Backward Scene Reference, BSR)로 구성됩니다. BFFR은 이전 및 이후 프레임을 고려하여 공간적 세부사항을 향상시키고 단기적인 시간 일관성을 유지합니다. BSR은 키프레임을 통해 이전 장면의 문맥을 유지하며 원활한 전환을 보장합니다.

- **Performance Highlights**: MSG는 Vid4와 REDS와 같은 벤치마크 데이터셋에서 평가되었습니다. 평가 지표로는 Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), 시간 일관성 측정이 포함됩니다. 그러나 실험 결과는 실패했으며, 현재 모델은 예상한 대로 작동하지 않는 상황입니다.



### Understanding and Improving Training-Free AI-Generated Image Detections with Vision Foundation Models (https://arxiv.org/abs/2411.19117)
- **What's New**: 최근 생성 모델의 발전에 따라 딥페이크와 같은 심각한 위험이 증가하고 있습니다. 전통적인 방법은 분류기를 훈련시키고 다양한 특징 추출 기법을 사용하였지만, 최근의 트레이닝 없는 검출 방법은 통계적 특성을 직접 활용하여 실제 이미지와 가짜 이미지를 구분합니다. RIGID라는 현재의 선도적인 접근법은 이미지 공간의 교란에 대한 DINOv2의 민감도를 활용하여 가짜 이미지를 탐지합니다.

- **Technical Details**: RIGID는 이미지 백본, 교란 유형 및 데이터 세트에 따른 탐지 성능 변화를 조사합니다. 실험을 통해 검출 성능이 모델의 강건성과 밀접한 관계가 있음을 밝혀내었으며, 자기 지도 학습(SSL) 모델은 보다 신뢰할 수 있는 표현을 제공합니다. 다양한 종합적 데이터 세트에서 Gaussian 노이즈와 블러링을 테스트한 결과, 얼굴 이미지에 대한 블러링이 더 효과적임을 발견했습니다.

- **Performance Highlights**: 새로운 방법으로 Contrastive Blur와 MINDER라는 심층 검출기가 도입되어 성능 향상에 기여했습니다. MINDER는 현재의 형태 기반 검출 프레임워크의 탐지 편향을 완화할 수 있는 최소 거리 선택을 통합합니다. 각종 데이터와 얼굴 이미지 전반에서 최고 성능을 달성하며, 이는 생성 및 검출 분야 모두에 귀중한 통찰력을 제공합니다.



### Timestep Embedding Tells: It's Time to Cache for Video Diffusion Mod (https://arxiv.org/abs/2411.19108)
Comments:
          Project: this https URL

- **What's New**: 이번 연구에서는 Timestep Embedding Aware Cache (TeaCache)라는 훈련 없는 캐싱 방법을 도입하였습니다. 기존의 모델 출력 재사용 방식은 타임스텝 간의 차이를 균일하게 처리했지만, TeaCache는 이러한 차이를 제대로 반영하여 더 효과적인 캐싱을 지원합니다. 이는 비디오 생성 속도를 빠르게 하면서도 시각적 품질을 유지하는 새로운 접근법을 제시합니다.

- **Technical Details**: TeaCache는 노이즈가 섞인 입력을 타임스텝 임베딩을 사용하여 조정함으로써 모델 출력과의 상관성을 높입니다. 이 과정에서 발생하는 차이를 더욱 정교하게 다듬기 위해 리스케일링 전략을 도입하고, 이를 기반으로 출력 캐싱을 위한 지표를 제공합니다. 이러한 접근법은 계산 비용을 최소화하면서도 성능을 극대화합니다.

- **Performance Highlights**: 실험 결과, TeaCache는 Open-Sora-Plan에 비해 최대 4.41배의 속도 향상을 달성함과 동시에 시각적 품질에서 0.07%의 미미한 저하만을 보였습니다. 이로 인해 빠른 속도를 유지하면서도 고품질의 비디오 생성을 가능하게 함을 입증하였습니다.



### Detailed Object Description with Controllable Dimensions (https://arxiv.org/abs/2411.19106)
Comments:
          9 pages, 5 figures

- **What's New**: 이번 논문은 시각 장애인을 위한 객체 설명의 중요성을 강조하며, 최근의 멀티모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)이 제공하는 잠재력을 다루고 있습니다. 특히, 사용자 의도에 맞춘 객체 설명을 생성하기 위한 새로운 기법인 'Dimension Tailor'를 제안하여, 객체 설명에서 사용자의 특정 요구 사항에 맞춘 세부 정보를 강조합니다. 이는 훈련 없이도 객체 설명의 질을 높이고, 특정 차원을 포함하거나 제외할 수 있는 유연성을 제공합니다.

- **Technical Details**: 제안된 'Dimension Tailor'는 세 가지 주요 단계인 차원 추출(dimension extracting), 차원 삭제(dimension erasing), 차원 보완(dimension supplementing)으로 구성됩니다. 초기 단계에서 긴 설명을 정의된 차원으로 변형하고, 불필요하거나 잘못된 차원을 제거하며, 사용자의 의도에 따라 누락된 차원을 추가하는 방식입니다. 이러한 구조는 MLLMs의 출력 텍스트가 사용자 의도에 맞춰지도록 도와줍니다.

- **Performance Highlights**: 실험 결과, Dimension Tailor는 최근 MLLMs의 성능을 지속적으로 개선하는 것으로 나타났습니다. 연구자들은 사용자가 지정한 차원의 관점에서 객체 설명의 제어 가능성을 평가하기 위한 세 가지 평가 지표를 설계했습니다. 이러한 접근법은 MLLMs가 객체를 설명할 때 특정 차원에 대한 선호도를 보이지만, 사용자 지침을 정확하게 따르지 못하는 경우도 있다는 사실을 보여줍니다.



### VARCO-VISION: Expanding Frontiers in Korean Vision-Language Models (https://arxiv.org/abs/2411.19103)
Comments:
          24 pages, 15 figures, 4 tables. Model weights at this https URL. Benchmarks released at NCSOFT's HuggingFace repositories (K-MMBench, K-SEED, K-MMStar, K-DTCBench, K-LLaVA-W). VARCO-VISION is an open-source Korean-English VLM with OCR, grounding, and referring capabilities

- **What's New**: 이 논문에서는 VARCO-VISION이라는 오픈소스 한국어-영어 비전-언어 모델(vision-language model, VLM)을 소개합니다. 본 모델은 언어와 비주얼 정보를 단계적으로 학습할 수 있도록 설계되어, 기존의 모델 지식을 보존하면서도 뛰어난 성능을 보여줍니다. 전반적으로 VARCO-VISION은 이미지-텍스트 이해 및 생성 능력에서 다른 유사 모델에 비해 뛰어난 성능을 발휘하며, 실제 상황에서도 활용 가능한 능력을 갖추고 있습니다.

- **Technical Details**: VARCO-VISION-14B는 비전 인코더, 프로젝터 및 대형 언어 모델(Large Language Model, LLM)로 구성되어 있습니다. Qwen-2.5-14B-Instruct를 언어 기초 모델로 사용하고, SigLIP를 비전 인코더로 활용하여 LLaVA-OneVision의 아키텍처를 따릅니다. 모델은 4단계의 훈련 전략으로 시각적 및 언어적 능력을 점진적으로 습득하도록 설계되어 있으며, OCR, grounding 및 referring과 같은 특정 용도를 위한 특수 토큰도 추가됩니다.

- **Performance Highlights**: VARCO-VISION은 여러 평가 기준에서 동급 모델들과 비교하여 뛰어난 성능을 보여줍니다. 특히, K-LLaVA-W와 같은 한국어 생성 능력 및 텍스트 기반 벤치마크에서 우수한 점수를 기록하며, 전반적인 다중 모드 및 텍스트 전용 평가에서도 강력한 언어 능력을 발휘합니다. 최종 단계인 선호 최적화(preference optimization)를 통해 모델의 반응 가독성 및 유창성이 크게 향상되었습니다.



### 360Recon: An Accurate Reconstruction Method Based on Depth Fusion from 360 Images (https://arxiv.org/abs/2411.19102)
- **What's New**: 본 논문에서는 360도 이미지의 잠재력을 활용한 새로운 MVS (Multi-View Stereo) 알고리즘인 360Recon을 제안합니다. 360도 이미지는 전통적인 pinhole 카메라에 비해 훨씬 더 넓은 시야각을 제공하여 VR (Virtual Reality) 및 AR (Augmented Reality)와 같은 응용 프로그램에서 중요한 역할을 합니다. 하지만 넓은 시야각으로 인해 발생하는 왜곡은 특성 추출(feature extraction) 및 매칭(matching) 과정에서 문제를 발생시킵니다.

- **Technical Details**: 360Recon은 구형(feature extraction) 모듈을 사용하여 왜곡 효과를 감소시킵니다. 또한 ERP (Equidistant Rectilinear Projection) 이미지에서 다중 스케일로 강화된 특성과 함께 구성한 3D 비용(volume) 체계를 결합하여 지역 기하학적 일관성을 높이면서도 정밀한 장면 재구성을 달성합니다. 이는 기존의 파노라마 재구성 데이터셋을 활용한 실험 결과에서도 입증되었습니다.

- **Performance Highlights**: 실험 결과, 360Recon은 깊이 추정(depth estimation) 및 3D 재구성에서 최첨단 성능을 보여줍니다. 높은 효율성을 유지하며, 기하학적 일관성을 갖춘 결과를 제공하여 관련 분야의 연구에 크게 기여할 것으로 기대됩니다.



### Tracking Progress Towards Sustainable Development Goal 6 Using Satellite Imagery (https://arxiv.org/abs/2411.19093)
- **What's New**: 본 연구는 아프리카 지역에서 수도 및 하수 시스템 접근을 평가하기 위한 새로운 모델링 프레임워크를 제시합니다. 이 프레임워크는 기존의 데이터 소스와 함께 최근의 비전통적 데이터 소스인 Afrobarometer 설문 조사 데이터와 위성 이미지(경로: Landsat 8 및 Sentinel-2), 그리고 딥러닝 기법(Meta의 DINO 모델)을 통합하여 개발되었습니다.

- **Technical Details**: 제안된 모델링 프레임워크는 위성 이미지를 통해 수도 접근 지역과 하수 시스템 접근 지역을 각각 96% 및 97%의 높은 정확도로 식별하는 능력을 보여주었습니다. 이 모델은 정책 입안자와 이해관계자가 수도와 하수 인프라 개선을 위한 우선적인 지역을 식별할 수 있는 도구로 활용될 수 있습니다. 또한, 인구 통계 데이터를 결합하여 국가 차원에서 수도 및 하수 시스템에 접근할 수 있는 인구 비율을 추정하고 추적할 수 있습니다.

- **Performance Highlights**: 모델은 다양한 아프리카 지역에서 수도와 하수 시스템의 접근성을 정확하게 평가하는 성과를 달성했습니다. 높은 정확성을 바탕으로, 정책 결정과 인프라 개선 노력을 위한 중요한 정보 제공이 가능하다는 점이 강조됩니다. 향후 이 접근법은 다른 지속 가능한 개발 목표(SDG) 평가에도 확장될 수 있는 잠재력을 가지고 있습니다.



### ObjectRelator: Enabling Cross-View Object Relation Understanding in Ego-Centric and Exo-Centric Videos (https://arxiv.org/abs/2411.19083)
- **What's New**: 이 논문에서는 Ego-Exo Object Correspondence 작업을 중심으로, Ego-centric(1인칭) 및 Exo-centric(3인칭) 영상 간 객체 매핑을 수행하는 새로운 방법인 ObjectRelator를 제안합니다. MCFuse와 XObjAlign이라는 두 가지 혁신적인 모듈이 포함되어 있으며, 이들을 통해 언어 및 시각 정보를 결합하고 서로 다른 관점 간 객체 표현의 일관성을 유지할 수 있습니다.

- **Technical Details**: 제안된 MCFuse 모듈은 시각-언어 모델에서 언어 정보를 활용하여 대상 객체의 위치 확인 정확도를 향상시킵니다. XObjAlign 모듈은 자기 지도 학습(self-supervised learning) 전략을 사용하여 쌍으로 된 ego와 exo 객체 마스크를 정렬하고, 이를 통해 서로 다른 관점에서 객체를 보다 잘 인식할 수 있도록 합니다.

- **Performance Highlights**: ObjectRelator는 Ego2Exo 및 Exo2Ego 작업에서 최첨단 성능을 달성하며, 기본 PSALM보다 10% 이상 향상된 결과를 달성했습니다. 연구 결과는 다중 모드 가이딩(multi-modal guidance)과 관점 간 정렬(cross-view alignment)의 잠재력을 강조하며, 향후 연구를 위한 기초를 제공합니다.



### Dynamic Attention and Bi-directional Fusion for Safety Helmet Wearing Detection (https://arxiv.org/abs/2411.19071)
- **What's New**: 이 논문은 건설 현장의 안전 헬멧 착용 감지를 위한 새로운 알고리즘을 제안합니다. 기존의 복잡한 환경에서 소형 및 겹치는 객체를 효과적으로 감지하기 위해 다이내믹 어텐션 메커니즘을 도입하여 Multi-scale perception을 강화했습니다. 특히, 작은 객체의 탐지를 개선하면서도 추가적인 계산 복잡도를 증가시키지 않는 방법을 제시합니다. Experimental results also show a noteworthy 1.7% improvement in mAP@[.5:.95].

- **Technical Details**: 제안하는 알고리즘에서는 Dynamic Attention Detection Head (DAHead)를 YOLOv8 모델에 통합하여 작은 타겟을 효과적으로 탐지할 수 있는 능력을 강화했습니다. 또한, Progressive Attention Feature Pyramid Network (PAFPN)를 Bi-directional Weighted Feature Pyramid Network (BWPPN)으로 대체하여 다양한 스케일의 입력 피처를 보다 포괄적으로 융합할 수 있도록 개선했습니다. CIoU loss 대신 Wise-IoU loss를 사용하여 모델의 수렴 속도와 효율성을 향상시켰습니다. 이로 인해 실시간 감시를 위한 경량화된 구현이 가능해졌습니다.

- **Performance Highlights**: 실험 결과, 제안된 DABFNet 모델은 기존 YOLOv8 모델에 비해 여러 데이터셋에서 성능이 크게 향상되었습니다. 특히, 복잡한 배경을 가진 작은 객체 탐지의 정확성이 증가하며, GFLOPs가 11.9% 감소하여 계산 효율성이 개선되었습니다. 이러한 성과는 건설 현장과 같은 실제 환경에서 안전 모니터링 시스템의 효율적인 운영에 기여할 수 있습니다.



### MaskRIS: Semantic Distortion-aware Data Augmentation for Referring Image Segmentation (https://arxiv.org/abs/2411.19067)
Comments:
          First two authors contributed equally

- **What's New**: 이번 연구에서는 Referring Image Segmentation (RIS) 분야에서 데이터 증강(data augmentation)이 기존 연구에서는 충분히 탐구되지 않았음을 강조합니다. 특히, 단순한 입력 마스킹(input masking) 방법을 사용하여 RIS 성능을 크게 향상시키는 "Masked Referring Image Segmentation (MaskRIS)"라는 새로운 훈련 프레임워크를 제안합니다. MaskRIS는 이미지와 텍스트의 마스킹을 통해 다양한 언어적 복잡성에 대한 모델의 강건성을 높일 수 있도록 설계되었습니다.

- **Technical Details**: MaskRIS 모델은 두 가지 주요 구성 요소로 이루어져 있습니다: (1) RIS에 대한 데이터 증강 기법으로서의 입력 마스킹과 (2) 입력 마스킹의 이점을 극대화하기 위한 Distortion-aware Contextual Learning (DCL)입니다. 이 과정에서 DCL은 원본 입력과 마스킹된 입력을 동시에 학습시켜 데이터의 다양성을 높이고 모델의 분석적 능력을 극대화합니다. 마스킹된 입력을 활용한 훈련은 모델이 특정 용어에 의존하는 것을 줄이고, 다양한 지시 표현을 이해하는 데 도움을 줍니다.

- **Performance Highlights**: MaskRIS는 기존의 RIS 방법들, 특히 약한 감독(weakly supervised) 설정에서도도 놀라운 성능 개선을 보여줍니다. RefCOCO, RefCOCO+, RefCOCOg 데이터셋에서 새로운 최첨단 성능을 달성하였으며, 다양한 RIS 모델에 쉽게 적용될 수 있습니다. 실험 결과는 MaskRIS가 기존 방법보다 월등한 성능을 발휘하고 있음을 보여줍니다.



### I Dream My Painting: Connecting MLLMs and Diffusion Models via Prompt Generation for Text-Guided Multi-Mask Inpainting (https://arxiv.org/abs/2411.19050)
Comments:
          Accepted at WACV 2025

- **What's New**: 이 논문은 다중 마스크 인페인팅(multi-mask inpainting)이라는 새로운 작업을 제안하며, 서로 다른 텍스트 프롬프트로 여러 영역을 동시에 채우는 방법을 다룹니다. 또한, 손상된 이미지를 입력으로 사용하여 다중 마스크 프롬프트를 자동 생성하는 것을 목표로 하는 fine-tuning 절차를 설계하였습니다. 이 접근법은 텍스트 기반 안내로 이미지 복원을 보다 창의적이고 세밀하게 진행할 수 있게 합니다.

- **Technical Details**: 제안된 시스템은 멀티모달 대형 언어 모델(MLLM)과 Stable Diffusion을 활용하여 다중 마스크 인페인팅 문제를 해결합니다. 모델은 QLoRA를 사용하여 미세 조정되어 있으며, rectified cross-attention 기법을 적용하여 생성된 프롬프트가 올바른 이미지 영역에 강제로 적용됩니다. 이 프로세스는 예술 작품과 같은 복잡한 이미지에서 뛰어난 결과를 도출하도록 설계되었습니다.

- **Performance Highlights**: 위키아트(WikiArt)와 Densely Captioned Images 데이터셋을 사용한 실험 결과, 제안된 파이프라인은 인상적이고 정확한 인페인팅 결과를 도출하였습니다. 모델이 생성하는 프롬프트는 다양한 사용자 수준에서 유용하며, 세밀한 다중 영역 인페인팅을 통해 컴퓨터 비전 작업을 위한 데이터 증대 도구로서도 활용될 수 있습니다.



### TAMT: Temporal-Aware Model Tuning for Cross-Domain Few-Shot Action Recognition (https://arxiv.org/abs/2411.19041)
- **What's New**: 이번 논문은 크로스 도메인 소수 샷 행동 인식 (CDFSAR) 문제를 해결하기 위한 새로운 접근 방식인 Temporal-Aware Model Tuning (TAMT)을 제안합니다. 특히, TAMT는 소스 데이터에서의 사전 훈련과 타겟 데이터에서의 미세 조정(fine-tuning)을 억제하여 모델 재훈련을 방지합니다. 이를 통해 계산 비용을 획기적으로 줄일 수 있습니다.

- **Technical Details**: TAMT는 Hierarchical Temporal Tuning Network (HTTN)을 도입하여, 지역적 시간 인식 어댑터(TAA)와 전역적 시간 인식 모멘트 튜닝(GTMT)을 사용하여 도메인 간 전이의 잠재력을 효과적으로 탐색합니다. TAA는 미리 학습된 모델의 중간 특징을 재조정할 수 있는 간단한 매개변수를 도입하여 타겟 데이터에 대한 효율적인 적응을 가능하게 합니다. GTMT는 스페이셜-템포럴 특징 분포를 기반으로 강력한 비디오 표현을 생성합니다.

- **Performance Highlights**: 다양한 비디오 벤치마크에서의 실험 결과는 TAMT가 최근 제안된 방법들보다 13%에서 31%까지 성능 개선을 이루었음을 보여줍니다. 이러한 성능 향상을 통해 TAMT는 최신 CDFSAR 결과에서 새로운 최첨단 성과를 기록하며, 보다 효율적인 훈련 비용을 보장합니다.



### 3D-WAG: Hierarchical Wavelet-Guided Autoregressive Generation for High-Fidelity 3D Shapes (https://arxiv.org/abs/2411.19037)
- **What's New**: 이 논문에서는 기존의 3D 생성 모델들의 한계를 극복하기 위한 3D-WAG(3D Hierarchical Wavelet-Guided Autoregressive Generation)라는 새로운 오토레그레시브 모델을 소개합니다. 기존의 방법들이 개별 voxel이나 포인트에서 '다음 토큰'을 예측하는 방식에서 벗어나, 3D 모양을 계층적이고 구조적으로 생성할 수 있도록 '다음 스케일' 예측으로 재정의합니다. 이를 통해 고해상도 3D 형상을 더 효율적이고 빠르게 생성할 수 있는 방식으로 발전했습니다.

- **Technical Details**: 3D-WAG는 형태를 다중 스케일의 웨이브렛 토큰 맵으로 인코딩하고, Transformer 기반의 아키텍처를 사용하여 오토레그레시브 방식으로 '다음 고해상도 토큰 맵'을 예측합니다. 이를 통해 형태 생성 과정에서 계층적인 접근 방식을 적용하여 계산 비용을 줄이고, 3D 형태의 기하학적 구조와 세부 사항을 보존합니다. 이 모델은 더욱 효율적으로 3D 형태를 생성할 수 있으며, 기존의 다른 생성 모델과 비교하여 주요 지표에서 우수한 성능을 보입니다.

- **Performance Highlights**: 3D-WAG는 Coverage, MMD, 1-NNA 등의 주요 지표에서 기존의 최신 3D 생성 모델들을 능가합니다. 모델을 다양한 조건(예: 클래스 및 텍스트)에 따라 생성할 수 있는 유연성을 갖추고 있어, 복잡한 작업을 잘 처리할 수 있습니다. 실험을 통해, 3D-WAG는 고충실도(high-fidelity)의 3D 모양을 생성하여 실제 데이터 분포와 밀접하게 일치하는 결과를 보여줍니다.



### PCDreamer: Point Cloud Completion Through Multi-view Diffusion Priors (https://arxiv.org/abs/2411.19036)
- **What's New**: 이번 연구에서는 포인트 클라우드 완성을 위한 새로운 방법인 PCDreamer를 제안합니다. 기존 방법들이 불완전한 포인트 클라우드에서 누락된 영역을 예측하는 데 주로 집중하는 반면, 이 방법은 다중 뷰 (multi-view) 확산 모델을 활용하여 보다 정확한 형상 생성을 가능하게 합니다. 이를 통해 이미지 세트가 전체 및 국소적인 형상 단서를 인코딩하여 완성도를 높이는 방식으로 발전하였습니다.

- **Technical Details**: PCDreamer는 고품질 포인트 클라우드를 완성하기 위해 세 가지 주요 모듈로 구성됩니다. 첫 번째 모듈은 단일 뷰(뷰)의 불완전한 포인트 클라우드를 입력받아 이를 기반으로 다중 뷰 이미지를 생성합니다. 두 번째 모듈에서는 이미지와 포인트 클라우드를 융합하여 초기 완전 형상을 생성하고, 마지막 모듈에서 확산 모델에서 도입된 불확실한 포인트를 제거하여 최종 완전 형상을 생성합니다.

- **Performance Highlights**: 연구 결과 PCDreamer가 미세한 세부사항을 회복하는 데 있어 탁월한 성능을 보여줍니다. 실험을 통해 PCDreamer는 기존의 포인트 클라우드 완성 방법들보다 훨씬 높은 완성도를 기록했으며, 특히 복잡한 형상의 세부정보 복원이 잘 이루어졌습니다. 이 연구는 3D 비전 및 관련 응용 분야에 중요한 기여를 할 것으로 예상됩니다.



### Locally-Focused Face Representation for Sketch-to-Image Generation Using Noise-Induced Refinemen (https://arxiv.org/abs/2411.19005)
Comments:
          Paper accepted for publication in 25th International Conference on Digital Image Computing: Techniques & Applications (DICTA) 2024

- **What's New**: 본 연구에서는 rudimentary face sketches를 고해상도 색 이미지로 변환하는 새로운 deep-learning 프레임워크를 제안합니다. Convolutional Block Attention-based Auto-encoder Network (CA2N)을 활용하여 얼굴 특징을 효과적으로 캡처하고 향상시키는 방법을 제시합니다. 이후, noise-induced conditional Generative Adversarial Network (cGAN) 과정을 통해 훈련되지 않은 도메인에서도 높은 성능을 유지할 수 있도록 합니다.

- **Technical Details**: 제안한 프레임워크는 두 단계의 학습 프로세스를 사용하여 이미지 정보 부족과 잡음을 처리합니다. 첫 번째 단계는 CA2N을 통해 특징 표현 학습을 수행하며, 이는 주요 얼굴 특징을 독립적으로 식별하고 처리합니다. 두 번째 단계에서는 noise-inducted adaptation learning 프로세스를 도입하여 cGAN을 통해 색상 이미지를 생성하고, 이미지 품질을 향상시키기 위한 다양한 손실 함수를 적용합니다.

- **Performance Highlights**: 실험 결과, CelebAMask-HQ, CUHK, CUFSF 데이터셋에서 각각 17, 23, 38의 FID 마진으로 기존 최고 방법을 능가하는 성능 지표를 달성했습니다. 본 모델은 sketch-to-image 생성에서 새로운 최첨단 기술을 설정하며, 범죄 식별과 같은 다양한 응용 분야에서 강력한 솔루션을 제공합니다.



### MVFormer: Diversifying Feature Normalization and Token Mixing for Efficient Vision Transformers (https://arxiv.org/abs/2411.18995)
- **What's New**: 이 논문에서는 비전 변환기(ViT)의 효율성을 향상시키기 위한 새로운 접근 방식을 소개합니다. 특히, 다양한 정규화를 통해 특징 학습을 다각화하려는 시도를 하고 있으며, 멀티-뷰 정규화(MVN)와 멀티-뷰 토큰 믹서(MVTM)라는 두 가지 새로운 구성 요소를 제안합니다. 이 방법들을 통합한 새로운 모델, 멀티-비전 변환기(MVFormer)가 기존의 CNN 기반 ViT보다 뛰어난 성능을 보여줍니다.

- **Technical Details**: MVN은 배치 정규화(BN), 레이어 정규화(LN), 인스턴스 정규화(IN)의 세 가지 다른 정규화 기능을 결합하여 특징 분포를 다양화합니다. MVTM은 다중 스케일의 깊이별 컨볼루션을 사용하는 토큰 믹서로, 각 단계별로 다르게 조정된 리셉티브 필드를 통해 다양한 시각적 패턴을 효과적으로 포착할 수 있도록 설계되었습니다. 이러한 설계는 전체적인 ViT 아키텍처의 유연성을 증대시킵니다.

- **Performance Highlights**: MVFormer는 이미지 분류, 객체 탐지, 인스턴스 및 의미론적 분할 작업에서 기존의 최신 CNN 기반 ViT보다 동일하거나 더 낮은 파라미터와 MACs로 성능을 발휘합니다. 특히, MVFormer-T, S 및 B 변형들은 ImageNet-1K 벤치마크에서 각각 83.4%, 84.3%, 84.6%의 최고 정확도를 달성했습니다. 이는 MVN과 MVTM의 조합을 통해 이루어진 성과로, 각 정규화의 독특한 특성이 성능에 의미있는 역할을 한다는 것을 입증합니다.



### SPAgent: Adaptive Task Decomposition and Model Selection for General Video Generation and Editing (https://arxiv.org/abs/2411.18983)
- **What's New**: 이 논문에서는 다양한 비디오 생성 및 편집 요구를 충족시키기 위한 새로운 시스템인 Semantic Planning Agent (SPAgent)를 제안합니다. 이 시스템은 최신 오픈 소스 모델들을 통합하고, 자동으로 최적의 모델을 선택하여 비디오 작업을 수행합니다. SPAgent는 사용자의 의도를 명확히 이해하고, 이에 따라 도구를 조정하는 혁신적인 프레임워크를 통해 비디오의 품질과 편집 가능성을 향상시킵니다.

- **Technical Details**: SPAgent는 세 가지 단계의 프로세스로 구성됩니다: (1) 분리된 의도 인식, (2) 원칙 기반 경로 계획, (3) 능력 기반 실행 모델 선택. 이 도구 라이브러리는 최신 오픈 소스 비디오 생성 및 편집 모델을 통합하여, 각 작업에 가장 적합한 모델을 자동으로 선택합니다. 또한, SPAgent는 비디오 품질 평가 능력을 강화하여, 새로운 모델을 자동으로 익히고 도구 라이브러리를 확장할 수 있습니다.

- **Performance Highlights**: 실험 결과, SPAgent는 다양한 비디오 작업에서 효과적으로 모델을 조정할 수 있는 능력을 보여주었으며, 그 결과는 매우 다양하고 적응력이 뛰어난 비디오 생성 및 편집을 가능하게 합니다. 이 시스템은 사용자의 복잡한 요구를 충족할 수 있도록 모델 간 협력을 가능하게 하여, 비디오 생성 및 편집의 전반적인 품질을 향상시킵니다.



### Det-SAM2:Technical Report on the Self-Prompting Segmentation Framework Based on Segment Anything Model 2 (https://arxiv.org/abs/2411.18977)
- **What's New**: Segment Anything Model 2 (SAM2)은 비디오 세분화 분야에서 뛰어난 성능을 보였으며, Det-SAM2라는 완전 자동화된 파이프라인 개발로 이어졌습니다. Det-SAM2는 YOLOv8 탐지 모델을 활용하여 객체 프롬프트를 자동 생성하고, 이를 통해 사람의 개입 없이 비디오 스트림에 대한 효율적인 분할과 수정이 가능합니다. 이 모델은 유한한 VRAM과 RAM 사용으로 무한한 길이의 비디오 처리도 가능하게 합니다.

- **Technical Details**: Det-SAM2는 SAM2의 강력한 객체 세분화 및 수정 기능을 유지하면서 초기 비디오 프레임에 대한 수동 프롬프트 입력이 필요 없도록 설계되었습니다. 주목할 점은 탐지 모델을 사용하여 프롬프트를 동적으로 생성하고, 이를 기반으로 SAM2가 실시간으로 세분화 예측을 수행할 수 있다는 것입니다. 또한, 메모리 뱅크 기능을 통해 이전 비디오에서 얻은 지식을 새로운 비디오에 적용할 수 있습니다.

- **Performance Highlights**: Det-SAM2의 성능은 SAM2와 비교해도 손색이 없으며, 기업 응용를 위해 세분화 결과의 후처리도 지원합니다. 이 시스템은 효과적인 수정 메커니즘을 통해 비디오 스트림에서 실시간으로 새로운 객체를 추가할 수 있고, 다양한 엔지니어링 최적화 기술을 통해 성능 오버헤드를 감소시켰습니다. 또한, 비디오 프레임을 축적하여 처리하는 방법을 통해 세분화 속도와 정확성을 모두 향상시킬 수 있었습니다.



### Perception of Visual Content: Differences Between Humans and Foundation Models (https://arxiv.org/abs/2411.18968)
- **What's New**: 이 연구는 인간과 기계가 생성한 주석(annotation)을 비교하여 각 주석이 이미지 해석에 미치는 영향을 파악하는 데 초점을 맞춘다. 특히 다양한 사회경제적 맥락을 나타내는 이미지에서 인간 주석과 ML 주석 간의 편향성을 분석한다. 연구 결과, 구조적으로는 낮은 유사성을 보이지만 이미지의 유사성이 어떻게 인식되는지에 대해서는 공통적인 패턴이 존재함을 보여준다.

- **Technical Details**: 연구는 손을 씻는 사람들의 이미지를 기반으로 하여 ML 모델의 예측 성능에 대한 인간과 기계 주석의 영향을 평가한다. 이 과정에서 주석은 벡터 표현으로 변환되어 ML 모델의 학습에 사용된다. 그 결과, 인간 주석의 경우 지역 분류(classification) 성능이 우수하고 균형 잡힌 반면, ML 주석은 소득 회귀(income regression)에서 가장 좋은 성과를 거두었다.

- **Performance Highlights**: 모델은 특정 사회경제적 배경의 사람들을 이미지로 통해 주석을 달면서, 편향성 문제를 해결하기 위한 새로운 방법론을 제안한다. 가시적인 성과로는 지역 분류에서 인간 주석이 더 나은 성과를 보여주며, 이는 주석의 품질이 ML 모델의 예측 정확도에 직접적인 영향을 미친다는 것을 보여준다. 또한 일반적인 편향성 문제를 해결하기 위한 실질적인 시사점을 제공합니다.



### SuperGaussians: Enhancing Gaussian Splatting Using Primitives with Spatially Varying Colors (https://arxiv.org/abs/2411.18966)
- **What's New**: SuperGaussians라는 새로운 방법을 제안하여 단일 Gaussian primitive에서 공간적으로 가변적인 색상과 불투명도를 활용하여 표현 능력을 개선합니다. 기존 방법의 제한사항을 극복하고 복잡한 장면의 텍스처와 기하학을 효과적으로 표현할 수 있도록 설계되어 있습니다. SuperGaussians는 세 개의 서로 다른 공간적으로 가변적인 함수 디자인을 통해 표현력을 증가시키며, 실험 결과 이들 모두가 베이스라인을 초과하는 성능을 보여줍니다.

- **Technical Details**: SuperGaussians는 기존 Gaussian 기법의 한계를 극복하기 위해 공간적으로 가변적인 색상과 투명도를 결합하여 각 Gaussian primitive의 표현 능력을 증가시킵니다. 세 가지 디자인 방법이 구현되었으며, 각각 쌍선형 보간(bilinear interpolation), 이동 가능한 커널(movable kernels), 작은 신경망(tiny neural networks)을 활용하여 다양한 시각적 표현을 가능하게 합니다. 이러한 접근 방식은 복잡한 장면에서도 보다 정밀한 텍스처 및 형태 표현을 가능하게 합니다.

- **Performance Highlights**: 실험은 Synthetic Blender, DTU, Mip-NeRF360 및 Tanks&Temples 데이터셋에서 수행되었으며, SuperGaussians는 모든 공간적으로 가변적인 함수가 2DGS 베이스라인을 초과하는 성능을 보였습니다. 특히, 이동 가능한 커널을 사용하는 SuperGaussians는 거의 모든 데이터셋에서 다른 Gaussian Splatting 기반 방법보다 우수한 노벨 뷰 합성(novel view synthesis) 성능을 기록했습니다. 또한, 제한된 수의 Gaussian primitives를 사용하여도 뛰어난 렌더링 품질을 달성할 수 있음을 입증했습니다.



### Random Sampling for Diffusion-based Adversarial Purification (https://arxiv.org/abs/2411.18956)
- **What's New**: 본 논문은 Denoising Diffusion Probabilistic Models (DDPMs)의 대안으로 랜덤 샘플링을 제안합니다. 기존 DDPM 샘플링은 안정적인 생성을 위한 것으로 설계되었으며, 이는 적대적 정화(adversarial purification)에는 최적의 솔루션이 아닐 수 있습니다. 랜덤 샘플링은 매 diffusion 과정에서 무작위 노이즈 공간에서 샘플링을 수행하여 더욱 강력한 적대적 공격에 대한 강인성을 제공합니다.

- **Technical Details**: 랜덤 샘플링 방식은 각 diffusion 단계에서 무작위 노이즈 공간에서 샘플을 추출하며, 이는 기존의 DDPM이나 DDIM 샘플링 방식이 인접한 노이즈 공간에서 샘플을 연속적으로 추출하는 방식과 대조됩니다. 논문에서 제안하는 mediator conditional guidance는 정화된 이미지와 깨끗한 이미지 입력 간의 예측 일관성을 보장합니다. 이를 통해 gradient-based guidance를 적용하여 샘플링 과정에서의 정확한 조건부 안내를 구현합니다.

- **Performance Highlights**: 랜덤 샘플링 방법은 다양한 샘플링 방법에 대한 평가에서 인상적인 개선을 보여줍니다. 연구진은 DiffAP라는 기준 방법을 설정하여 최첨단(SOTA) 접근 방식을 성능과 방어 안정성 모두에서 대폭 능가하였습니다. 특히, 강력한 공격 하에서도 DiffAP는 20% 이상의 강인성 향상을 10배의 샘플링 가속도로 달성했습니다.



### Waterfall Transformer for Multi-person Pose Estimation (https://arxiv.org/abs/2411.18944)
- **What's New**: 이번 논문에서는 다중 인물 자세 추정을 위한 새로운 'Waterfall Transformer' 아키텍처인 WTPose를 제안합니다. 이 구조는 변형된 Swin backbone과 transformer 기반의 waterfall 모듈로 구성되어 있어, 다양한 단계를 통해 다중 스케일(feature map)을 생성합니다. WTPose는 Cascade 아키텍처에서의 필터링을 통해 수용 범위를 확장하고 로컬 및 글로벌 컨텍스트를 포착하여 네트워크의 전체적인 표현 능력을 향상시킵니다.

- **Technical Details**: WTPose는 단일 패스(single-pass) 및 엔드 투 엔드(end-to-end) 학습 가능한 프레임워크로, 다중 인물 2D 자세 추정을 위한 다중 스케일 접근 방식을 탑재하고 있습니다. 이 구조는 dilated attention 메커니즘을 사용하여 더 넓은 수용 영역(Field-of-View, FOV)을 확보하고, 이미지로부터 로컬 및 글로벌 컨텍스트를 캡처합니다. 논문에서 제안하는 waterfall transformer 모듈(WTM)은 Swin transformer를 강화하여, 다양한 스케일의 특징을 모두 처리할 수 있도록 합니다.

- **Performance Highlights**: COCO 데이터셋에서 수행된 실험에서는 제안된 WTPose가 다른 transformer 기반 아키텍처보다 더 높은 성능을 보이는 것으로 나타났습니다. 특히, WTPose는 occluded joints를 포함한 모든 관절에 대해 더 정확한 예측을 가능하게 합니다. 이러한 성능 향상은 기존 모델들과 비교했을 때, 더 나은 다중 스케일 표현 및 더 나은 컨텍스트 정보 활용 덕분입니다.



### Revealing Key Details to See Differences: A Novel Prototypical Perspective for Skeleton-based Action Recognition (https://arxiv.org/abs/2411.18941)
- **What's New**: 이 논문에서는 ProtoGCN이라는 새로운 그래프 프로토타입 학습 방법을 소개합니다. ProtoGCN은 skeleton-based action recognition에서 유사한 동작을 구분하는 미세한 차이를 강조하여 정확한 분류를 가능하게 합니다. 특히, 프로토타입 재구성 네트워크를 통해 동작 프로토타입을 조합하여 독특한 표현을 생성합니다. 처음으로, 맨손 동작에서 얻은 특징을 메모리에 저장된 프로토타입으로 조합함으로써 인식 성능을 극대화합니다.

- **Technical Details**: ProtoGCN의 핵심은 프로토타입 재구성 네트워크로, 이는 액션 프로토타입의 연관적 조합으로 표현을 형성하도록 강제합니다. 이 프로토타입은 인체 관절 간의 다양한 관계 패턴을 나타내며, 입력 동작 특징의 연관적 응답을 통해 프로토타입을 샘플링하고 조립합니다. 또한, 동작 토폴로지 향상 모듈을 포함하여 더욱 풍부하고 표현력 있는 기능을 제공하고, 클래스별 대조 학습 전략을 통해 클래스 간 구별을 강조합니다.

- **Performance Highlights**: ProtoGCN은 NTU RGB+D, NTU RGB+D 120, Kinetics-Skeleton 및 FineGYM과 같은 여러 벤치마크 데이터셋에서 최첨단 성능을 달성했습니다. 이러한 결과는 제안된 방법의 효과성을 입증하며, 동작 특징을 효과적으로 포착하고 유사한 동작을 구별하는 데 큰 도움이 됩니다. Qualitative 및 quantitative 분석 모두 ProtoGCN이 우수한 성능을 제공한다는 것을 보여줍니다.



### Self-Cross Diffusion Guidance for Text-to-Image Synthesis of Similar Subjects (https://arxiv.org/abs/2411.18936)
- **What's New**: 이번 논문에서는 Self-Cross diffusion guidance를 제안하여 여러 유사한 피사체를 합성하는 과정에서 발생하는 subject mixing 문제를 해결하고자 합니다. 이 방법은 cross-attention maps와 self-attention maps 간의 중첩을 처벌하여 더욱 효과적으로 subject mixing을 제거합니다. 특히, 우리의 Self-Cross guidance는 가장 구분 가능한 패치 외에도 다른 패치에 대한 혼합 문제를 해결합니다. 우리는 또한 Similar Subject Dataset (SSD)을 새로운 벤치마크로 제공하여 평가의 신뢰성을 높입니다.

- **Technical Details**: Self-Cross guidance는 transformer 기반의 diffusion model에서 효과적이며, 자동으로 선택된 패치의 self-attention maps를 집계하여 피사체가 다른 피사체에 주의를 기울이지 않도록 합니다. 이 방법은 특별한 교육 과정 없이도 적용 가능하며, 기존의 단일 패치 기반 방법보다 성능이 우수합니다. Diffusion model은 latent space에서 작동하여 이미지 생성의 계산적 복잡성을 줄이며, cross-attention을 통해 다양한 프롬프트에 대해 조정 가능한 이미지 생성을 구현합니다.

- **Performance Highlights**: Self-Cross guidance의 도입으로, 우리가 실시한 정성적 및 정량적 분석 결과에서 subject mixing이 크게 줄었습니다. 또한, 새로운 SSD dataset과 VLM 기반의 메트릭을 통해 다양한 방법을 평가함으로써, 기존의 방법 보다도 더 높은 성능과 신뢰성을 입증하였습니다. 이 방법은 모델의 성능 향상뿐만 아니라, 기존의 subject neglect 문제도 완화하여 결과적으로 이미지 합성의 질을 높였습니다.



### Efficient Track Anything (https://arxiv.org/abs/2411.18933)
- **What's New**: 본 논문에서는 Segment Anything Model 2 (SAM 2)의 비효율성을 극복하기 위한 경량의 EfficientTAMs 모델을 제안합니다. EfficientTAMs는 기존 ViT(Vision Transformer) 기반의 이미지 인코더를 재사용하고, 메모리 모듈의 복잡성을 줄여 더 빠른 속도와 저전력의 비디오 객체 분할(video object segmentation)을 가능하게 합니다. 이 모델은 iPhone 15 Pro Max와 같은 모바일 장치에서도 원활하게 작동하며, 효율성을 강조합니다.

- **Technical Details**: EfficientTAMs는 일반적인, 비계층적(non-hierarchical) 이미지 인코더인 ViT-Tiny/-Small을 통해 SAM 2의 복잡성을 줄이고 유사한 성능을 유지합니다. 또한, 메모리 모듈의 효율적인 교차 주의(cross-attention) 방법을 소개하여 메모리 공간 토큰의 구조적 특성을 활용합니다. 이를 통해, 메모리 관련 계산의 복잡성을 줄이고 적절한 성능을 보여주고 있습니다.

- **Performance Highlights**: 실험 결과, EfficientTAMs는 다양한 비디오 분할 벤치마크에서 기존의 강력한 반 감독 학습(Semi-supervised) 비디오 객체 분할 방법들보다 더 우수한 성능을 보이며, 약 2배의 속도 증가 및 약 2.4배의 파라미터 감소를 기록하였습니다. 이미지 분할에서도 기존 SAM에 비해 약 20배의 속도 향상과 20배의 파라미터 감소를 보인다는 결과를 도출하였고, 이는 실질적인 모바일 비디오 객체 분할 애플리케이션에 적합한 성능으로 해석될 수 있습니다.



### VIPaint: Image Inpainting with Pre-Trained Diffusion Models via Variational Inferenc (https://arxiv.org/abs/2411.18929)
Comments:
          13 pages, 9 figures

- **What's New**: 이 논문은 VIPaint라는 새로운 방법론을 제안하여, 전이학습(pre-trained)된 diffusion model을 활용해 이미지 복원(inpainting) 문제를 해결하는 혁신적인 접근 방식을 보여줍니다. VIPaint는 고수준의 의미론(semantics)과 저수준의 세부정보를 동시에 추론하는 계층적 포스트리어(hierarchical posterior)를 정의함으로써 기존의 방법들을 뛰어넘는 성능을 보여줍니다. 기존의 알고리즘들이 대량의 훈련 세트와 파인튜닝(fine-tuning) 과정을 요구했던 것과 달리, 이 방법은 비염 번역(propagation) 없이도 빠르고 효율적으로 수행될 수 있습니다.

- **Technical Details**: VIPaint는 비선형 Markov 근사(non-Gaussian Markov approximation) 방식을 통해 진정한 (L)DM 포스트리어를 최적화합니다. 이 방법은 상태 공간(state space)에서의 노이즈 수준을 전략적으로 고려하는 계층적 포스트리어를 적용하며, 이를 통해 관측된 픽셀로부터 정보를 효과적으로 추론할 수 있습니다. 저자는 VIPaint의 효율적인 조건화 기법을 통해 기존의 이미지 복원 문제에서 발생하는 여러 주요 문제들, 특히 신뢰성 저하 문제를 해결하고 있음을 보여줍니다.

- **Performance Highlights**: VIPaint 방법론은 기존의 픽셀 기반 및 잠재적 잠수 모델(latent diffusion models)에서 이미지 복원, 블러 제거(deblurring), 초해상도(superresolution) 등 다양한 응용 분야에서 눈에 띄는 질적 및 양적 개선을 이뤄냈음을 강조합니다. 많은 실험을 통해 VIPaint는 데이터 다양성과 신뢰도(plausibility) 모두에서 기존 방법들 보다 우수한 성능을 발휘하는 것을 지속적으로 입증했습니다. 이러한 성능 향상은 이미지 생성 분야에서 VIPaint가 새로운 기준을 세울 수 있는 가능성을 강조합니다.



### Data Augmentation with Diffusion Models for Colon Polyp Localization on the Low Data Regime: How much real data is enough? (https://arxiv.org/abs/2411.18926)
- **What's New**: 의료 분야에서의 데이터 부족은 딥러닝 모델의 성능을 방해합니다. 이 논문에서는 생성 모델인 Denoising Diffusion을 사용하여 조인트 폴립 로컬라이제이션을 제공하는 콜로노스코피 이미지를 생성하는 여러 실험을 진행했습니다. 이 방법을 통해 기존의 제한된 데이터셋으로도 더 나은 성능을 달성할 수 있는 가능성을 보여주고 있습니다.

- **Technical Details**: 논문에서는 Denoising Diffusion Probabilistic Models (DDPM)을 기반으로 한 여러 생성 모델 훈련 방법을 소개합니다. 다양한 해상도의 데이터셋을 혼합하고, 업스케일링(autoencoding)을 활용하여 목표 해상도로 데이터 생성을 시도하였습니다. YOLO v9 기반의 모델을 사용하여 폴립 로컬라이제이션 작업에서 전이 학습을 실시했으며, 생성된 데이터의 품질 측정을 통해 성능을 검증하였습니다.

- **Performance Highlights**: 실험 결과, 생성된 데이터를 사전훈련(pre-training) 데이터로 사용함으로써 실제 데이터와의 통합에서 성능 향상이 확인되었습니다. 특히 데이터가 부족한 상황에서 생성된 데이터의 활용이 긍정적인 영향을 미치는 것으로 나타났습니다. 이러한 접근은 병원에서의 진단 및 치료에 도움을 줄 수 있는 효과적인 방법이 될 것입니다.



### Textured As-Is BIM via GIS-informed Point Cloud Segmentation (https://arxiv.org/abs/2411.18898)
Comments:
          Permission granted by all co-authors for the publication of the extended article to the conference paper "BIM Integration for Automated Identification of Relevant Geo-Context Information via Point Cloud Segmentation" (2023). URL: this https URL

- **What's New**: 이 논문은 3D 기하체 생성을 자동화하기 위한 새로운 방법론을 제시하고 있습니다. 특히, 감지된 데이터인 Point Cloud Data (PCD)와 지리정보 데이터인 GIS를 결합하여 보다 의미 있는 As-Is Building Information Models (BIM)을 생성할 수 있는 시스템을 설명합니다. 또한, 이 방법이 대규모 철도 프로젝트에 적용될 수 있음을 보여주며, 비용 절감의 잠재력을 강조합니다.

- **Technical Details**: 자동화를 위해 이 연구에서는 객체 인식(object recognition)과 의미론적 분할(semantic segmentation)을 위한 기계 학습(Machine Learning) 및 딥 러닝(Deep Learning) 모델을 활용합니다. 연구에서는 수정된 딥 신경망(DNN) 아키텍처인 2DPASS를 사용하여 빌딩, 식생, 수체, 도로, 철도 선로 등의 파라미터를 분할하고, 무료로 제공되는 LiDAR 스캔, 디지털 정사 사진 및 ATKIS 도형 파일과 같은 데이터를 기반으로 합니다.

- **Performance Highlights**: 연구 결과는 GIS 데이터와 결합한 As-Is BIM을 생성하는 프로세스에서 무료로 제공되는 GIS 데이터의 활용 가능성을 입증합니다. 초기 철도 프로젝트의 계획 단계에서 비용 절감의 잠재력은 특히 볼 만하며, 사용자들은 자동화된 Alignments 생성을 통해 복잡한 계획 작업을 단순화하고 효율적으로 진행할 수 있습니다. 제공된 분석과 사례 연구들은 이 방법론의 실용성을 뒷받침합니다.



### T2SG: Traffic Topology Scene Graph for Topology Reasoning in Autonomous Driving (https://arxiv.org/abs/2411.18894)
- **What's New**: 이 논문에서는 기존 HD(High Definition) 매핑 방법에서 간과되었던 교통 신호에 의해 제어되고 안내되는 차선과 그 관계를 명시적으로 모델링하는 새로운 Traffic Topology Scene Graph(T2SG)를 정의합니다. 새로운 그래프 기반 방법인 TopoFormer를 제안하여 T2SG 생성 작업을 수행하며, 이 방법은 Lane Aggregation Layer와 Counterfactual Intervention Layer를 포함하여 차선 간의 관계를 효과적으로 식별합니다. 이러한 혁신적인 접근 방식은 교통 장면에서의 topological 구조를 더욱 정확하고 설명 가능하게 만들어 주며, 자율 주행 시스템의 의사 결정 과정을 지원합니다.

- **Technical Details**: TopoFormer는 Lane Aggregation Layer(LAL)를 사용하여 차선의 기하학적 중심선 사이의 거리 정보를 기반으로 전역 정보를 집계합니다. 이와 함께 Counterfactual Intervention Layer(CIL)를 도입하여 교통 신호에 의해 형성되는 실제 도로 구조(예: 교차로, 직선 도로 등)를 모델링합니다. 이 과정에서 차선 간의 공간 정보와 도로 신호 간의 가이드 정보를 동시에 고려하여 관계를 추론하는 새로운 접근 방식을 적용합니다.

- **Performance Highlights**: TopoFormer는 OpenLane-V2 기반의 T2SG 생성 작업에서 기존의 모든 방법보다 우수한 성과를 보이며, 46.3 OLS 점수를 기록하여 최첨단 성능을 달성하였습니다. 이 결과는 T2SG와 TopoFormer의 효과를 입증하며, 이를 통해 교통 topology reasoning과 같은 후속 작업에서 실질적인 향상을 가져옵니다. 또한, 소스 코드를 공개할 계획이 있어 향후 연구에 긍정적인 영향을 미칠 것으로 기대됩니다.



### GTPC-SSCD: Gate-guided Two-level Perturbation Consistency-based Semi-Supervised Change Detection (https://arxiv.org/abs/2411.18880)
Comments:
          6 pages, 4 figures

- **What's New**: 본 연구에서는 기존의 일차원적인 perturbation을 넘어서, 이미지 레벨의 강-약 일관성과 특징 레벨의 일관성을 동시에 유지하는 새로운 방법인 Gate-guided Two-level Perturbation Consistency regularization 기반의 Semi-supervised change detection(SCDC) 방법인 GTPC-SSCD를 제안합니다. 이를 통해 주석이 없는 데이터를 적극적으로 활용할 수 있습니다. 또한, 샘플의 학습 복잡성을 평가할 수 있는 게이트 모듈을 설계하여 각 샘플에 대한 특징의 perturbation 필요성을 판단합니다.

- **Technical Details**: GTPC-SSCD 방법은 두 개의 데이터 집합으로 구성된 훈련 세트를 사용합니다. 주석이 있는 데이터 셋(𝒟l)과 주석이 없는 데이터 셋(𝒟u)으로 구성되며, 각각의 레이블과 대응하는 이미지 쌍이 포함됩니다. 이 방법은 특징의 perturbation과 이미지의 perturbation을 결합하여 성능을 개선하고, 네트워크의 학습 과정에서 더욱 효과적으로 주석이 없는 데이터의 잠재력을 탐색할 수 있도록 합니다.

- **Performance Highlights**: 다양한 실험을 통해 우리가 제안한 방법이 여섯 개의 공공 원거리 감지 데이터 세트에서 다른 최신 SSCD 방법 7개보다 우수한 성능을 보임을 입증했습니다. 특히 GTPC-SSCD 방법은 높은 정확도와 우수한 변화 감지 성능을 나타내어, 실질적인 응용 가능성을 보여주었습니다.



### Comprehensive Performance Evaluation of YOLOv11, YOLOv10, YOLOv9, YOLOv8 and YOLOv5 on Object Detection of Power Equipmen (https://arxiv.org/abs/2411.18871)
- **What's New**: 최근 전 세계 산업 생산의 급격한 발전과 함께 전력 장비의 신뢰성에 대한 수요가 지속적으로 증가하고 있습니다. 이 논문에서는 YOLOv5, YOLOv8, YOLOv9, YOLOv10과 최신 YOLOv11 방법의 전력 장비 객체 탐지 성능을 종합적으로 평가했습니다. 실험 결과, YOLOv11이 이전 모델들에 비해 mAP(mean Average Precision) 점수와 탐지 성능에서 최고치를 기록했습니다.

- **Technical Details**: YOLO(You Only Look Once) 알고리즘은 전력 장비 객체 탐지에 특화된 최신의 단일 단계(object detection) 탐지 알고리즘입니다. YOLOv11은 이전 YOLO 버전에서 개선된 효율성과 정확성을 바탕으로 채널 및 공간 정보의 통합을 위한 C2PSA 모듈을 도입했으며, 그로 인해 다양한 데이터 세트에 걸쳐 탐지 정확성을 향상시킬 수 있었습니다. YOLOv11의 손실 함수는 여러 구성 요소를 통합한 종합 손실 함수를 최소화하여 성능을 극대화합니다.

- **Performance Highlights**: YOLOv11은 정밀도와 속도의 균형을 이루어내며, 전력 장비 객체 탐지뿐만 아니라 인스턴스 분할(instance segmentation), 이미지 분류(image classification), 자세 추정(pose estimation) 등의 다양한 컴퓨터 비전 작업에도 확장 가능합니다. YOLOv11은 설계된 성능을 바탕으로, 실제 환경에서 직면하는 여러 도전과제를 해결할 수 있는 가능성을 가지고 있습니다.



### RIGI: Rectifying Image-to-3D Generation Inconsistency via Uncertainty-aware Learning (https://arxiv.org/abs/2411.18866)
Comments:
          Project Page: this https URL

- **What's New**: 이번 연구는 단일 이미지로부터 3D 객체를 생성하는 과정에서 발생하는 불확실성과 아티팩트를 해결하기 위해 3D Gaussian Splatting(3DGS) 기법을 활용합니다. 기존 방식들이 명확한 다중 뷰 이미지나 비디오를 이용해 3D 생성 과정에서의 일관성을 위해 노력했지만, 이들이 종종 생성하는 이미지 간의 불일치로 인해 문제를 초래했습니다. 본 연구에서는 Gaussian 모델 간의 확률적 데이터를 활용하여 불확실성 지도를 생성하고, 이를 기반으로 불확실성 인식 학습을 통해 최적화 과정을 개선했습니다.

- **Technical Details**: 본 연구의 방법론은 두 가지 주요 단계로 구성됩니다. 첫 번째 단계에서는 두 개의 Gaussian 모델을 동시에 최적화하며 이들 간의 차이를 통해 불확실성을 모델링합니다. 두 번째 단계에서는 불확실성 지도를 기반으로 픽셀 별 로스 가중치를 동적으로 조정하여 고불확실성 지역에서의 영향을 감소시킵니다. 이렇게 생성된 3D 자산은 최적화 과정에서 발생하는 텍스처 간 불일치를 완화할 수 있습니다.

- **Performance Highlights**: 연구 성과는 기존의 3D 생성 품질이 저하되는 문제를 해결함으로써 시각적으로 더 매끄러운 결과를 보여주고 있습니다. 실험 결과, 제안하는 방법이 3D 생성 품질을 정량적으로 및 정성적으로 향상시킨다는 것을 확인할 수 있었습니다. 이를 통해 이번 연구가 3D 자산 생성을 위한 차별화된 접근법을 제공하는 것임을 입증하고 있습니다.



### Improving Batch Normalization with TTA for Robust Object Detection in Self-Driving (https://arxiv.org/abs/2411.18860)
- **What's New**: 이 논문은 현재의 자율주행 인식 모델에서 모델 성능을 저하시킬 수 있는 테스트 데이터 및 훈련 데이터 간의 분포 변화 문제를 해결하기 위해 두 가지 강력한 방법을 제안합니다. 첫 번째로, Generalized-search Entropy Minimization (GSEM) 방법에 기반한 LearnableBN 레이어를 도입하여 Batch Normalization (BN) 통계를 동적으로 업데이트할 수 있습니다. 두 번째로, 의미적 일관성 기반의 이중 단계 적응 전략을 제안하여 모델의 적응 과정에서 불안정한 샘플을 제거합니다.

- **Technical Details**: 제안된 TTA 방법은 Batch Normalization의 통계를 예측하기 위해 보조 learnable 파라미터를 BN 레이어에 통합합니다. 이로 인해 기존의 EMA 방법의 한계를 극복하고, 내부 공변량 이동(internal covariate shift) 문제를 해결합니다. 또한, 적응 단계를 두 개의 단계로 나누고 학습 속도의 변화를 조정하여, 각 단계에서 샘플 예측의 의미적 일관성을 활용하여 불확실한 샘플을 거를 수 있습니다.

- **Performance Highlights**: NuScenes-C 데이터셋에 대한 종합적인 실험 결과, BEVFormer를 기본 모델로 사용할 경우 최대 약 8%의 성능 개선을 달성했습니다. 이러한 결과는 제안된 TTA 방법이 다양한 결함 유형과 심각도에서 효과적임을 보여줍니다. 향후, 이 연구의 소스코드는 공개될 예정입니다.



### COMPrompter: reconceptualized segment anything model with multiprompt network for camouflaged object detection (https://arxiv.org/abs/2411.18858)
Comments:
          SCIENCE CHINA Information Sciences 2024

- **What's New**: 이 연구에서는 camouflaged object detection (COD)을 위한 새로운 다중 프롬프트 네트워크인 COMPrompter를 제안합니다. 기존의 Segment Anything Model (SAM)을 확장하여 다중 프롬프트 전략을 통합함으로써 단일 프롬프트의 한계를 극복하고자 합니다. 새로운 엣지 그래디언트 추출 모듈인 EGEM을 통해 변별력을 높이면서도 기존의 SAM 구조에 변화를 주어 COD에 더 특화된 모델을 구축했습니다.

- **Technical Details**: COMPrompter는 경계 프롬프트와 박스 프롬프트의 상호 작용을 통해 정확한 객체 감지를 도모하는 두 가지 주요 모듈인 EGEM과 Box-Boundary Mutual Guidance Module (BBMG)을 포함합니다. EGEM은 이미지와 진실 라벨(ground truth)에 대한 dilation 및 canny 연산을 통해 경계 마스크를 생성하고, BBMG는 두 프롬프트 간의 연결성을 강화하여 세밀한 정보 추출을 촉진합니다. 또한, 이 네트워크는 Discrete Wavelet Transform (DWT)을 사용하여 고주파 신호를 추출하여 색인 기능을 보완합니다.

- **Performance Highlights**: 실험 결과, COMPrompter는 COD10K 벤치마크에서 기존의 최첨단 모델보다 평균 2.2% 더 높은 성능을 기록했습니다. 또한, 폴립 분할 적용에서 상위 모델들에 비해 우수한 성능을 보이며, COD 분야에서의 알림 메시지를 지속적으로 향상시키고 있습니다. 이러한 성과는 COMPrompter가 COD 지난 세대 기술을 초월했음을 보여줍니다.



### Improving Accuracy and Generalization for Efficient Visual Tracking (https://arxiv.org/abs/2411.18855)
Comments:
          WACV 2025

- **What's New**: 본 논문에서는 시각적 추적기의 OOD (out-of-distribution) 일반화 성능을 크게 향상시키는 SiamABC라는 새로운 시아미즈 트래커를 제안합니다. SiamABC는 새로운 아키텍처 설계를 통해 타겟의 동적 변화를 효과적으로 추적할 수 있으며, 이를 위해 빠른 역방향이 없는 동적 테스트 타임 적응(dynamic test-time adaptation) 방법을 포함하여 모델을 지속적으로 조정합니다.

- **Technical Details**: SiamABC는 정적 이미지 템플릿과 현재 시간의 검색 영역 이미지를 연결하는 시각-시간 다리(temporal bridge)를 개선하여 두 가지 주요 기여를 하고 있습니다. 또한, Fast Mixed Filtration라는 새로운 학습 가능한 레이어를 설계하여 이중 템플렛 및 이중 검색 지역(represents)에서 각 구성 요소의 관련성을 증진합니다. 학습 관점에서, 전이 관계 손실(transitive relation loss)을 도입하여 필터링된 표현 간의 유사성을 연결합니다.

- **Performance Highlights**: SiamABC는 다양한 벤치마크에서 SOTA(최첨단) 접근 방식들을 초월하며, 특히 AVisT 벤치마크에서는 47.2%의 AUC를 달성하였고, 이는 MixFormerV2-S보다 7.6% 향상된 성과입니다. 이 트래커는 CPU에서 100 FPS로 작동하여 빠른 속도를 유지하며, 효율성과 정확성을 동시에 만족시키는 뛰어난 성능을 보여줍니다.



### CrossTracker: Robust Multi-modal 3D Multi-Object Tracking via Cross Correction (https://arxiv.org/abs/2411.18850)
- **What's New**: CrossTracker는 카메라와 LiDAR 데이터의 단점을 서로 보완하기 위해 두 단계의 3D 다중 객체 추적 (MOT) 방법론을 제안합니다. 기존의 단일 단계기법에서는 카메라 데이터의 검증을 중시하였으나 CrossTracker는 LiDAR의 정보를 이용하여 카메라 검사를 개선하는 방법론을 포함하고 있습니다. 이를 통해 카메라와 LiDAR 간의 교차 보정이 이루어져, 보다 견고한 추적 결과를 도출할 수 있게 됩니다.

- **Technical Details**: CrossTracker는 세 가지 주요 모듈인 다중 모달 모델링 (M^3) 모듈, 초기 조잡한 궤적 생성을 위한 조잡한 궤적 생성 (C-TG) 모듈, 그리고 교차 보정을 수행하는 궤적 개선 (TR) 모듈로 구성되어 있습니다. M^3 모듈은 이미지, 포인트 클라우드 등 다양한 정보의 융합을 통해 궤적 생성을 위한 강력한 기준을 제공합니다. 이러한 구조는 기존의 단일 단계 아키텍처와는 달리 교차 보정 모듈을 활용하여 각 모달리티의 상호 보완적인 특성을 최대한 활용합니다.

- **Performance Highlights**: CrossTracker는 KITTI 추적 벤치마크에서 18개의 경쟁자를 능가하며 뛰어난 성능을 입증하였습니다. 특히, 조잡한 궤적 생성을 먼저 시행하고, 이를 보강하는 독립적인 개선 과정에서 다중 모달 데이터의 이점을 최대한 활용한 점이 강조됩니다. 이로 인해 CrossTracker는 보다 정확하고 안정적인 LiDAR 궤적 생성을 가능하게 하여 3D MOT의 내구성 및 신뢰성을 크게 향상시킵니다.



### FaithDiff: Unleashing Diffusion Priors for Faithful Image Super-resolution (https://arxiv.org/abs/2411.18824)
Comments:
          Project page: this https URL

- **What's New**: 이 논문은 FaithDiff라는 새로운 이미지를 고해상도로 복원하는 방법을 제안합니다. FaithDiff는 잠재 확산 모델(latent diffusion models, LDM)의 강력한 표현 능력을 활용하여 고품질과 신뢰성 있는 이미지 복원을 수행합니다. 이 방법은 기존의 확산 기반 방법과 달리 고품질 이미지로 사전 학습된 확산 모델의 정보를 활용하여 입력 이미지의 구조를 복원합니다.

- **Technical Details**: FaithDiff는 VAE(Variational Autoencoder)의 인코더를 사용하여 저해상도 이미지에서 잠재 공간(latent space)으로 매핑하고, 추출된 특징을 확산 과정에 맞춰 정렬하는 정렬 모듈을 개발합니다. 이 과정에서 이미지 설명으로부터 추출된 텍스트 임베딩을 보조 정보로 통합하여 구조적 정보를 추출하는 데 도움을 줍니다. 인코더와 확산 모델을 공동으로 미세 조정하여 더 효과적인 정보 전송이 이루어질 수 있도록 합니다.

- **Performance Highlights**: 실험 결과, FaithDiff는 기존의 최첨단 방법들과 비교할 때 뛰어난 성능을 발휘하고, 고해상도 복원 결과에서 높은 품질을 제공함을 입증했습니다. 이 방법은 가상의 데이터셋뿐만 아니라 실제 데이터셋에서도 그 효과성을 강조하며, 신뢰성 있는 구조 정보를 복원하는 데 있어서 중요한 역할을 하고 있음을 보여줍니다.



### Multi-Task Label Discovery via Hierarchical Task Tokens for Partially Annotated Dense Predictions (https://arxiv.org/abs/2411.18823)
- **What's New**: 최근 다중 밀집 예측 작업을 동시에 학습하는 연구에서 부분적으로 주석이 달린 데이터로부터 직접적인 픽셀 단위의 감독 신호를 얻는 것이 중요해졌습니다. 이전 연구들은 데이터의 다양한 해석을 고려하기 위해 교차 작업 일관성을 구축하거나 적대적 학습을 수행했습니다. 본 논문에서는 이러한 도전 과제를 해결하기 위해 글로벌 및 미세한 계층적 작업 토큰(HiTTs)을 최적화하여 픽셀 단위의 감독 신호를 발견하는 새로운 접근 방식을 제안합니다.

- **Technical Details**: HiTTs는 모델이 구획적 상관관계를 추론할 수 있도록 구체적으로 설계되며, 글로벌 작업 토큰은 서로 다른 작업의 피처 맵과 상호작용할 수 있도록 합니다. 또한, 각 작업별로 미세하게 조정된 작업 토큰을 학습하여 각 작업의 피처 맵과 밀접한 상호작용을 수행합니다. 이러한 학습된 글로벌 및 로컬 토큰은 서로 다른 수준에서의 작업별 밀집 레이블을 발견하는 데 사용됩니다.

- **Performance Highlights**: NYUD-v2, Cityscapes, PASCAL Context 데이터셋에서 본 논문에서 제안한 방법은 기존의 최첨단 방법들에 비해 의미 있는 성능 향상을 보여주었습니다. 이를 통해 밀집 예측 작업에서 직면하는 여러 도전 과제를 극복할 수 있으며, 특히 제한된 주석이 달린 데이터에 대해 우수한 성능을 발휘함을 입증했습니다.



### Enhancing Compositional Text-to-Image Generation with Reliable Random Seeds (https://arxiv.org/abs/2411.18810)
- **What's New**: 이번 연구는 텍스트-이미지 생성에서 초기 노이즈가 구성이미지의 정확성에 미치는 중요한 역할을 강조합니다. 연구자들은 특정 노이즈 패턴이 조합 프롬프트에 대해 더 신뢰할 수 있는 결과를 도출한다고 밝혔습니다. 새로운 접근 방식으로 신뢰할 수 있는 노이즈 패턴을 활용하여 생성된 이미지로 구성된 데이터셋을 자동으로 구축할 수 있음을 보여줍니다. 이를 통해 모델의 조합 능력을 크게 향상시킬 수 있습니다.

- **Technical Details**: 본 연구에서는 초기 시드(initial seed)가 조합 생성(compositional generation)에 미치는 영향을 심층적으로 분석합니다. 특히, 특정 시드가 모델이 객체를 배치하는 위치와 패턴을 안내하며, 이로 인해 정확한 객체 수 및 위치를 도출할 확률이 높아진다는 점을 설명합니다. 조합 프롬프트에 대해 더 나은 정확도를 제공하는 시드의 사용으로, 모델을 보다 효율적으로 구성하는 방법이 제시됩니다.

- **Performance Highlights**: 실험 결과, 안정적인 시드를 사용할 때 Stable Diffusion과 PixArt-α 모델의 숫자 조합 정확도가 각각 29.3% 및 19.5% 향상되며, 공간적 조합에서는 각각 60.7%와 21.1%의 개선을 보였습니다. 이러한 개선은 모델의 조합 생성 능력을 향상시키는 데 유의미한 영향을 미치며, 텍스트-이미지 모델의 성능을 보다 신뢰할 수 있는 방향으로 발전시킬 수 있는 가능성을 제시합니다.



### Lifting Motion to the 3D World via 2D Diffusion (https://arxiv.org/abs/2411.18808)
Comments:
          project page: this https URL

- **What's New**: 이 연구에서는 MVLift라는 새로운 방법론을 제안하여 2D 자세 시퀀스만을 사용하여 글로벌 3D 모션을 예측할 수 있음을 보여줍니다. 이 접근 방식은 3D 감독(3D supervision)이 필요 없고, 다양한 도메인에서 일반화될 수 있어 기존의 방법론이 가지고 있던 한계를 극복합니다. 본 연구는 복잡한 동작을 포함하여 인간, 동물 등 다양한 활동을 처리할 수 있는 잠재력을 가지고 있습니다.

- **Technical Details**: MVLift는 다단계 프레임워크(multi-stage framework)를 기반으로 하며, 2D 모션 확산 모델을 활용해 여러 시점에서 일관된 2D 자세 시퀀스를 순차적으로 생성합니다. 연구는 네 가지 단계로 구성되며, 첫 번째 단계에서 에피폴라 라인(epipolar lines)을 따르는 2D 자세 시퀀스를 예측하도록 모델을 학습합니다. 후속 단계에서는 multi-view 일관성을 보장하는 방법을 도입하여 3D 모션을 회복하는 데 필요한 기하학적 관계를 유지합니다.

- **Performance Highlights**: MVLift는 기존 방법론보다 우수한 성능을 보여주며, 다섯 개의 데이터셋에서 3D 감독이 아닌 상태에서도 기존 방법을 초월했습니다. 이 프레임워크는 다양한 데이터 세트에서 일관되게 강력한 결과를 도출하며, 특히 3D 감독이 없거나 최소한의 데이터가 필요한 경우에도 효과적입니다. 결과적으로 이 접근 방식은 2D 자세 시퀀스에서 글로벌 3D 모션을 추정하는 데 큰 기여를 할 것으로 기대됩니다.



### Reconstructing Animals and the Wild (https://arxiv.org/abs/2411.18807)
Comments:
          12 pages; project page: this https URL

- **What's New**: 이 논문에서는 자연 환경에서 동물 및 자연 경관을 재구성하는 새로운 방법을 제안합니다. 기존의 많은 연구가 동물의 2D 모델링에 집중된 반면, 이 연구는 환경과 동물을 동시에 고려하여 3D 재구성을 시도합니다. 이를 통해 연구자는 단일 이미지에서 복잡한 자연 장면을 재구성하는 새로운 접근 방식을 제시하고 있습니다.

- **Technical Details**: 제안된 방법은 Large Language Models (LLMs)의 최근 발전을 기반으로 하며, CLIP 임베딩을 구조화된 구성 장면 표현(compositional scene representation)으로 디코딩하는 자기 회귀 모델을 훈련시킵니다. 연구자는 또한 천 개의 자산과 함께 한 백만 개의 이미지로 구성된 합성 데이터셋을 생성하여 모델이 실제 이미지에서 동물과 환경을 재구성하는 데 일반화되도록 합니다.

- **Performance Highlights**: 연구 결과, 제안된 접근 방식은 실제 이미지에서 동물과 그 환경을 성공적으로 재구성할 수 있는 능력을 보여줍니다. 모델은 CLIP 임베딩을 활용하여 다양한 자산을 효과적으로 추정하며, 재구성된 장면은 해석 가능하고 편집 및 애니메이션 작업 또한 지원합니다. 공개 예정인 데이터셋과 코드는 향후 연구를 촉진하는 데 기여할 것입니다.



### GloFinder: AI-empowered QuPath Plugin for WSI-level Glomerular Detection, Visualization, and Curation (https://arxiv.org/abs/2411.18795)
- **What's New**: GloFinder는 QuPath 플러그인으로, 자동화된 사구체(glomeruli) 탐지를 도와줍니다. 기존의 오픈소스 도구들이 프로그래밍 교육을 요구하는 반면, GloFinder는 사용자가 클릭 몇 번으로 전체 슬라이드 이미지(WSI)에서 사구체를 탐지할 수 있도록 디자인되었습니다. 이 플러그인은 CircleNet 모델에 기반하여 약 16만 개의 수동 주석이 달린 사구체로 훈련됩니다.

- **Technical Details**: GloFinder는 CircleNet과 Weighted Circle Fusion(WCF)을 활용하여 효율적인 자동 탐지를 제공합니다. 사용자는 QuPath 환경에서 플러그인을 설치하고 단일 클릭으로 탐지 프로세스를 시작할 수 있습니다. 알고리즘은 WSI를 여러 오버랩 패치로 나누고, 각 패치에서 CircleNet 모델을 사용해 사구체를 탐지합니다. 최종 결과를 통합하기 위해 Non-Maximum Suppression(NMS) 알고리즘과 WCF가 적용됩니다.

- **Performance Highlights**: GloFinder는 평균 5% 이상의 탐지 성능 향상을 실현했으며, 사람-결합(annotation) 전략을 사용하는 경우 주석 시간을 68.59% 단축할 수 있습니다. 모델은 160,000개 이상의 주석된 데이터로 훈련되었으며, 다양한 데이터셋 간의 일반화를 통해 안정성과 정확성을 높였습니다. 이러한 성능은 신장병리학 연구 및 임상 실습에 유용한 도구로 자리 잡을 것으로 기대됩니다.



### MRI Breast tissue segmentation using nnU-Net for biomechanical modeling (https://arxiv.org/abs/2411.18784)
Comments:
          Deep Breath @ MICCAI 2024

- **What's New**: 이 논문은 2D 유방 촬영술(mammography)과 3D 자기 공명 영상(MRI) 통합을 개선하기 위한 연구를 제시합니다. nnU-Net 세분화 모델을 사용하여 조직을 효과적으로 식별하고, NiftySim과 FEBio라는 유한 요소 해석(Finite Element Analysis, FEA) 생체 역학 솔버의 성능을 비교 분석하였습니다. 이를 통해 유방 조직의 정확한 해석과 모델링에 대한 통찰을 제공합니다. 이 연구는 진단 정확도와 치료 계획 향상을 위한 중요한 기초 자료로 활용될 수 있습니다.

- **Technical Details**: 연구에 사용된 데이터셋은 166개의 T1 가중 비지방 포화 동적 조영 MRI 스캔으로 구성되어 있습니다. nnU-Net 아키텍처를 이용하여 다양한 유방 조직을 효과적으로 세분화했으며, 각 조직에 대한 Dice Coefficient 값을 0.94에서 0.83까지 기록했습니다. 세분화된 데이터는 3D 메쉬를 생성하고, NiftySim과 FEBio를 이용하여 유방 조직의 물리적 거동을 시뮬레이션하는 데 사용되었습니다. 이 과정은 세분화, 기하학 추출 및 메쉬 생성을 포함하여 전반적인 통합 프로세스의 중요한 단계를 형성합니다.

- **Performance Highlights**: nnU-Net 모델을 통해 유방 MRI 데이터의 6개 클래스 세분화를 수행하여 기존 방법보다 개선된 성과를 보여주었습니다. 실험 결과, NiftySim과 FEBio 간의 정확도와 신뢰성을 비교 분석함으로써, 유방 조직 반응에 대한 중요한 통찰을 제공하였으며, 이 결과들은 유방암 진단 및 치료 계획에 있어 임상적 활용 가능성을 높입니다. 이 연구의 결과는 유방 촬영과 MRI 데이터 통합을 통한 의학적 이미지 분석에서의 한 단계 진전을 의미합니다.



### Fall Leaf Adversarial Attack on Traffic Sign Classification (https://arxiv.org/abs/2411.18776)
- **What's New**: 이번 연구는 기존의 인간 제작 인공물에 의존하지 않고 자연에서 발생하는 물체, 특히 나뭇잎을 활용하여 새로운 유형의 적대적 공격(adversarial attack)을 제안합니다. 이는 피해자가 이러한 공격을 우연한 일로 보기 쉽게 만들어, 공격의 속성을 약화시키는 'plausible deniability'를 갖습니다. 해당 접근법은 교통 표지판이 잘못 분류되는 원인을 분석하여, 다양한 나뭇잎의 크기, 색상 및 위치 변화를 고려합니다.

- **Technical Details**: 본 연구는 LISA 데이터셋을 사용하여 나뭇잎이 어떻게 교통 표지판의 잘못 분류를 초래할 수 있는지를 분석합니다. 적대적 공격의 개념은 작은 이미지 입력 변경사항이 DNN의 예측을 크게 변형시킬 수 있다는 것을 강조하며, 본 연구에서는 이러한 공격의 성공 가능성과 엣지 탐지(edge detection)에 미치는 영향을 다룹니다. FGSM 및 PGD와 같은 기존 공격 방식과 달리, 나뭇잎을 이용한 공격 기술은 환경에서 자연스럽게 혼합되어 모델 오분류를 유도하게 됩니다.

- **Performance Highlights**: 연구의 요약적 결과는 나뭇잎이 교통 표지판에 부착될 경우 평균 97%의 높은 신뢰도로 잘못 분류된 이미지를 생성할 수 있음을 보여줍니다. 또한, 연구에서는 성공적인 공격과 비성공적인 공격 간의 엣지 탐지 관련 변수를 분석하고 논의함으로써, 보안 및 대응 방법에 대한 통찰을 제공합니다. 이러한 결과는 자율 주행 차량의 신뢰성과 안전성을 저해할 수 있는 중대한 위협으로 작용할 수 있습니다.



### CoVis: A Collaborative Framework for Fine-grained Graphic Visual Understanding (https://arxiv.org/abs/2411.18764)
- **What's New**: 본 논문은 협업 프레임워크인 CoVis를 제안합니다. 이 프레임워크는 이미지에서 최대한의 지식을 추출하고, 더 전체적인 관점에서 이미지를 이해하도록 돕기 위해 설계되었습니다. 또한, CoVis는 Fast Segment Anything Model(FastSAM)과 U-Net 모델을 기반으로 한 동적 조정을 통해 시각적 분석을 생성합니다.

- **Technical Details**: CoVis는 복합 단계별 이미지 분할과 언어 생성 방법을 적용하여 일반적인 시각 콘텐츠 이해를 목표로 합니다. 이 과정에서 FastSAM을 이용한 코스 그레인(segmentation) 모델과 U-Net 기반의 파인 그레인(segmentation) 모델을 활용하며, 생성된 텍스트의 품질 향상을 위해 Prompt Engineering 기법도 도입합니다. 이러한 프로세스는 다양한 시각적 요소를 평가하고 해석할 수 있는 텍스트를 생성합니다.

- **Performance Highlights**: 정량적 및 정성적 실험을 통해 CoVis는 기존의 방법보다 우수한 기능 추출 성능을 보였습니다. 특히, CoVis는 일반 목적으로 만들어진 대형 모델보다 더 포괄적이고 상세한 시각적 설명을 생성할 수 있습니다. 또한, 32명의 참여자를 대상으로 한 실험 결과, 이 프레임워크는 시각적 정보의 전달 효율성과 품질을 개선하는 데 효과적임을 입증하였습니다.



### DiffMVR: Diffusion-based Automated Multi-Guidance Video Restoration (https://arxiv.org/abs/2411.18745)
- **What's New**: 이 연구에서는 동적이고 실제 환경에서 가려진 영역을 복구하는 비디오 인페인팅(video inpainting)의 새로운 모델인 DiffMVR를 제안합니다. 특히 의료 환경에서 인체의 지속적인 동작 모니터링의 필요성에 의해 영감을 받았습니다. 이 모델은 적응형 레퍼런스 프레임을 활용하여 인페인팅 프로세스를 안내하는 동적 이중 가이딩 이미지 제도를 도입하여 비디오 프레임 간의 세밀하고 매끄러운 전환을 포착합니다.

- **Technical Details**: DiffMVR는 각 비디오 프레임에 대해 자동으로 두 개의 가이드 이미지를 생성합니다: 대칭 이미지와 과거의 비가려운 프레임입니다. 이 대칭 이미지는 프레임의 시각적 반쪽을 따라 생성되고, YOLOv8 모델을 통해 가장 최근에 완전히 보이는 객체를 찾는 과정에서 얻어진 과거 프레임은 각 프레임에 맞춰 처리됩니다. 이러한 데이터는 VAE를 통해 잠재 공간에 인코딩되며, 이 쿼리는 두 가이드 이미지의 키-값 쌍과 상호작용하여 이중 주의 점수를 생성하게 됩니다.

- **Performance Highlights**: DiffMVR는 향상된 모션 손실(motion loss) 항을 도입하여 연속 비디오 프레임 간의 시간적 일관성을 높였습니다. 이러한 접근법은 각 프레임을 개별적으로 처리하는 전통적인 방법과 달리, 연속적인 비디오 스트림을 유지하여 현실적인 장면의 동작과 미적 표현을 충실히 재구성합니다. 마지막으로, DiffMVR는 정량적 및 정성적 비교를 통해 최신 인페인팅 모델들보다 뛰어난 성능을 입증하였으며, AI 기반 비디오 인페인팅의 새로운 기준을 제시합니다.



### The Last Mile to Supervised Performance: Semi-Supervised Domain Adaptation for Semantic Segmentation (https://arxiv.org/abs/2411.18728)
Comments:
          28 pages, 6 figures

- **What's New**: 이 논문에서는 Semi-Supervised Domain Adaptation (SSDA) 설정을 통해 낮은 주석 비용으로도 감독 성능에 가까운 결과를 달성할 수 있는 방법을 제안합니다. 기존의 Unsupervised Domain Adaptation (UDA) 및 Semi-Supervised Learning (SSL) 방법들이 SSDA 설정에 적합하지 않음을 보여주며, SSDA 접근 방식의 잠재력을 강조합니다. 또한, 새로운 프레임워크는 일관성 정규화(consistency regularization)와 픽셀 대조 학습(pixel contrastive learning), 자기 학습(self-training) 기법을 통합하여 목표 도메인 라벨을 효과적으로 활용합니다.

- **Technical Details**: SSDA는 감독 데이터(source labeled data), 미라벨 데이터(target unlabeled data), 일부 목표 라벨을 함께 이용하여 감독 학습의 성능과 유사한 결과를 달성합니다. 이 연구에서는 효율성을 높이기 위해 클러스터가 조밀한 목표 표현을 형성하는 것을 목표로 하고, 도메인에 강인한 특징 추출기를 학습하여 소스 도메인 데이터를 활용합니다. 제안된 방법은 GTA→Cityscapes 벤치마크에서 최첨단 성능을 보여주며, 단 50개의 목표 라벨로도 감독 성능에 근접하는 결과를 달성합니다.

- **Performance Highlights**: 이 연구 결과는 단 50개의 표적 라벨을 통해 SSDA의 성능이 UDA보다 약 6.9 mIoU 증가함을 보여주며, SSDA 설정에서는 기존 UDA 방법들이 최적이 아니라고 판단합니다. 또한, 새로운 프레임워크는 다양한 벤치마크에서 추가 하이퍼파라미터 조정 없이도 효과성을 확인하고 있습니다. 이를 통해 SSDA는 실용적인 가치가 높고, 감독 학습과 유사한 성능을 달성할 수 있다는 점을 강조합니다.



### Generative Visual Communication in the Era of Vision-Language Models (https://arxiv.org/abs/2411.18727)
Comments:
          PhD Thesis

- **What's New**: 이번 연구에서는 최근 비전-언어 모델(vision-language models, VLMs)의 발전을 활용하여 효과적인 비주얼 커뮤니케이션 디자인을 자동으로 생성하는 방법을 탐구합니다. 기존 생성 모델들이 텍스트로부터 이미지를 생성하는 데는 큰 발전을 이루었지만, 복잡한 아이디어를 단순화된 추상 비주얼로 표현하는 데에는 한계를 보이고 있습니다. 이에 따라 모델의 작업 공간을 제한하고 특정 작업에 맞춘 정규화를 도입하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: 자유 손 스케치(free-hand sketching)는 아이디어와 개념을 표현하는 중요한 시각적 도구입니다. 본 논문에서는 CLIP을 사용하여 입력 이미지를 추상적인 스케치로 변환하는 프로세스를 최적화하는 방법을 제안합니다. 이 방법은 CLIP 인코더를 사용하여 사진의 기하학적 기반을 토대로 세밀하고 의미 있는 스케치를 생성합니다. 사전 훈련된 CLIP 모델의 최종 및 중간 활성화를 결합하여 기하학적 및 의미적 단순화를 달성합니다.

- **Performance Highlights**: 제안한 방법론은 다양한 수준의 추상화를 가능하게 하여 인간과 유사한 스타일의 스케치를 만들어낼 수 있습니다. 다수의 레벨의 추상화를 제공하며, 기존의 스케치 데이터 세트 또는 새로운 훈련 단계 없이 최종 출력을 조정할 수 있는 유연성을 가지고 있습니다. 스케치를 통해 입력 객체의 본질을 잘 포착하면서도 최적화된 성능을 발휘하는 결과를 보여 줍니다.



### Evaluating Vision-Language Models as Evaluators in Path Planning (https://arxiv.org/abs/2411.18711)
- **What's New**: 이번 연구에서는 대규모 언어 모델(LLMs)의 계획 능력에 대한 한계를 극복하기 위해, 비전 언어 모델(VLMs)을 계획 평가 도구로 사용할 수 있는지를 탐구합니다. 특히, 복잡한 경로 계획 시나리오에서 VLM의 성능을 평가하기 위한 새로운 벤치마크인 PathEval을 도입합니다.

- **Technical Details**: PathEval 벤치마크는 VLM이 시나리오 설명에서 최적 경로의 특성을 추상화하고, 각 경로에 대한 정밀한 저수준 인식을 나타내며, 이 정보를 통합하여 더 나은 경로를 결정하는 능력을 요구합니다. 실험 분석에 따르면, 최신 VLM 모델들은 벤치마크에서 중대한 도전에 직면하고 있으며, 주어진 시나리오를 정확하게 추상화하는 능력은 있지만, 저수준 세부사항을 인지하는 데 어려움을 겪고 있습니다.

- **Performance Highlights**: 실험 결과, VLM의 비전 구성 요소가 주요 bottleneck(병목 현상)으로 작용하며, 이는 단순한 end-to-end fine-tuning을 통해 해결될 수 없는 문제임을 보여줍니다. 따라서, 효과적인 경로 평가자가 되기 위해서는 이러한 비전 인코더의 작업별 차별적 적응(task-specific discriminative adaptation)이 필요합니다.



### Random Walks with Tweedie: A Unified Framework for Diffusion Models (https://arxiv.org/abs/2411.18702)
- **What's New**: 본 논문은 확산 모델(Generative Diffusion Model) 알고리즘을 설계하기 위한 단순한 템플릿을 제시합니다. 확산 샘플링이 일련의 랜덤 워크(random walks)로 해석되면서 새로운 이론적 틀을 제공합니다. 이러한 접근 방식은 기존의 Markov 체인 또는 역 확산 이론을 피하고, 랜덤 워크와 Tweedie의 공식을 중심으로 한 이론을 강조합니다. 주요 기여는 여러 알고리즘 간의 관계를 통일된 관점에서 이해할 수 있도록 돕는 것입니다.

- **Technical Details**: 딥 스코어 기반 확산 모델은 훈련 데이터로부터 확률 밀도의 로그의 기울기인 스코어 함수를 학습하는 생성 모델입니다. 이 스코어 함수는 확산의 수학적 이론을 기반으로 반복적인 알고리즘에서 새로운 샘플을 생성하는 데 사용됩니다. 본 프레임워크는 알고리즘적 선택을 명확히 분리하여 훈련 및 샘플링을 가능하게 하며, 샘플링 중 사용하는 노이즈 스케줄이 훈련 시 사용되는 것과 일치할 필요가 없도록 설계되었습니다. 이러한 유연성 덕분에 조건부 샘플링도 가능해졌습니다.

- **Performance Highlights**: 제안된 알고리즘 템플릿은 여러 기존 확산 모델들과의 일관성을 보여줍니다. 특히, 노이즈가 가득 찬 데이터에서의 훈련을 통해 고품질 이미지를 생성하는 다양한 방법론을 제시합니다. 기존의 방법들과 비교하여 더 간단하고 직관적인 알고리즘 구조를 갖추고 있으며, 샘플링 과정의 효율성을 높이는 데 기여하고 있습니다. 이 프로세스는 주어진 입력 이미지의 품질을 유지하면서도 효과적으로 역 문제를 해결할 수 있는 가능성을 열어줍니다.



### MatchDiffusion: Training-free Generation of Match-cuts (https://arxiv.org/abs/2411.18677)
Comments:
this https URL

- **What's New**: MatchDiffusion는 텍스트-비디오 확산 모델을 사용하여 매치 컷을 생성하는 최초의 훈련이 필요 없는 방법입니다. 이 방법은 초기 디노이징 단계에서 장면의 구조를 결정하고 후속 단계에서 세부 사항을 추가하는 확산 모델의 특성을 활용합니다. MatchDiffusion는 'Joint Diffusion'과 'Disjoint Diffusion'의 두 단계를 통해 매치 컷의 독창적이고 일관된 비디오를 생성합니다.

- **Technical Details**: MatchDiffusion에서 'Joint Diffusion' 단계는 두 개의 프롬프트를 단일 노이즈 샘플에서 초기화하여 서로 공유하는 구조를 형성합니다. 이후 'Disjoint Diffusion' 단계에서 각 비디오는 프롬프트에 따라 독립적으로 발전하여 의미가 다른 비디오 쌍을 생성합니다. 이 접근 방식은 매치 컷을生成하기 위한 구조적 일관성을 유지하면서도 고유한 콘텐츠를 제공합니다.

- **Performance Highlights**: 사용자 연구 및 척정을 통해 MatchDiffusion이 매치 컷 생성에서 높은 효과성과 가능성을 보여줍니다. 이 연구는 기존의 방법들과 비교하여 매치 컷 품질을 정량화할 수 있는 메트릭스를 제안하고 있습니다. 또한, 다양한 기술 수준의 창작자들이 매치 컷을 실험할 수 있도록 하여 이러한 강력한 도구의 민주화를 목표로 하고 있습니다.



### GaussianSpeech: Audio-Driven Gaussian Avatars (https://arxiv.org/abs/2411.18675)
Comments:
          Paper Video: this https URL Project Page: this https URL

- **What's New**: GaussianSpeech는 음성 오디오에서 사진처럼 생생한 개인화된 3D 인간 머리 아바타의 고품질 애니메이션 시퀀스를 합성하는 새로운 접근 방식을 소개합니다. 이 연구는 음성 신호와 3D Gaussian splatting을 결합하여 현실적이고 시간적으로 일관된 모션 시퀀스를 생성하는 방법을 제안합니다. 또한, 표현에 따라 색상 생성 및 주름 검출 기반 손실을 통해 얼굴의 세부 사항을 합성합니다.

- **Technical Details**: GaussianSpeech는 다양한 관점에서 포토리얼리스틱한 3D 아바타 애니메이션을 생성하기 위해 3D Gaussian 포인트를 명시적으로 사용하여 3D 애니메이션을 모델링하는 새로운 방법론을 제안합니다. 이 모델은 음성 입력에서 입술과 표정 특징을 추출하는 오디오 조건부 변환기(transformer) 모델을 활용하여 3D 애니메이션을 최적화합니다. 연구자들은 16대의 카메라로 구성된 멀티뷰(multi-view) 데이터셋을 수집하여 높은 해상도 비디오와 함께 고품질 오디오 데이터를 결합했습니다.

- **Performance Highlights**: GaussianSpeech는 실제 시간 렌더링 속도로 여러 가지 얼굴 표현 및 스타일을 포함하여 시각적으로 자연스러운 모션을 지속적으로 생성합니다. 특히, 이 모델은 기존 애니메이션 기법들보다 우수한 성능을 보이며, 다양한 표현에 따라 동적 주름과 같은 세부적인 디테일도 잘 포착합니다. 새로운 데이터셋은 6명의 원주율(native English) 스피커로 구성되어 있으며, 총 2500개의 시퀀스가 기록되었습니다.



### Active Data Curation Effectively Distills Large-Scale Multimodal Models (https://arxiv.org/abs/2411.18674)
- **What's New**: 본 연구는 지식 증류(Knowledge Distillation, KD) 기술을 사용하여 대규모 모델을 소규모 모델로 압축하는 기존 접근 방식 대신, 액티브 데이터 큐레이션(Active Data Curation)을 도입하여 대조적 멀티모달 프리트레이닝에 효과적인 증류 방법으로 제안합니다. ACID(Active Curation as Implicit Distillation)라는 온라인 배치 선택 방법이 다양한 모델과 데이터 설정에서 강력한 KD 기준선을 능가함을 보여줍니다. 또한, 데이터 큐레이션 전략이 표준 KD와 보완적으로 작용하여 높은 성능의 추론 효율적인 모델을 훈련할 수 있음을 입증합니다.

- **Technical Details**: 액티브 데이터 큐레이션은 현재 학습 중인 작은 모델(학생)과 고정된 큰 모델(참조) 간의 성능 격차를 줄이는 샘플을 자동으로 선택하여 지식을 전이하는 방식을 채택합니다. 연구팀은 ACID가 KD보다 효과적이고 훈련 계산 비용에 대한 스케일링에서 더 유리함을 입증했으며, 데이터 큐레이션을 기존 KD와 결합하여 성능을 더욱 향상시킬 수 있는 방법을 제시합니다. 이를 통해 ACED(ACID with Explicit Distillation)라는 간단하고 강력한 프리트레이닝 레시피를 개발하여 FLOP 효율적인 이미지-텍스트 대조 모델을 훈련합니다.

- **Performance Highlights**: ACED는 27개의 제로샷 분류 및 검색 작업에서 최신 기술(State-of-the-art) 결과를 달성하며, 추론 시 최대 11% 더 적은 FLOP을 요구합니다. 또한, ACED 모델은 이미지 캡션 및 비주얼 질문-응답 작업에서 더 큰 비전 인코더보다 뛰어난 성능을 보이며, 생성적인 멀티모달 모델 훈련을 위한 강력한 백본 역할을 수행합니다. 이러한 결과는 액티브 데이터 큐레이션과 KD의 결합이 높은 성능을 달성할 수 있음을 명확히 합니다.



### AC3D: Analyzing and Improving 3D Camera Control in Video Diffusion Transformers (https://arxiv.org/abs/2411.18673)
Comments:
          Project Page: this https URL

- **What's New**: 최근 3D 카메라 제어를 비디오 생성 모델에 통합하는 연구가 급증하고 있지만, 정밀도가 떨어지고 비디오 생성 품질이 저하되는 문제가 있었습니다. 본 연구에서는 카메라 움직임을 1차 원리의 관점에서 분석하여, 합성 품질을 손상시키지 않으면서도 정밀한 3D 카메라 제어를 가능하게 하는 인사이트를 발견했습니다. 새로운 AC3D(Camer Control) 아키텍처를 통해 비디오 생성 품질을 10% 향상시키고, 훈련 매개변수를 4배 감소시킬 수 있었습니다.

- **Technical Details**: 비디오 생성 모델에 대한 여러 연구를 통해 카메라의 저주파수 움직임 특성을 인지하며, 저주파수에 해당하는 노이즈 제거 과정에만 카메라-conditioning을 적용하는 방식으로 성능을 개선했습니다. 비디오 확산 변환 모델에서 카메라 포즈 추정을 내재적으로 수행하는 구성요소를 찾아내어, 모델의 30%만 카메라 제어에 사용함으로써 훈련 속도를 15% 높이고 시각 품질을 10% 개선했습니다. 또한, 카메라가 정적이지만 동적인 장면을 갖는 2만 개의 비디오 데이터셋을 보완하여 모델의 장면과 카메라 움직임을 더욱 명확히 구분하게 했습니다.

- **Performance Highlights**: AC3D 아키텍처는 기존 카메라 제어 모델인 MotionCtrl, CameraCtrl, VD3D 대비 비디오 충실도가 18% 향상되었고, 카메라 조정의 정확성이 25% 개선되었습니다. 이와 같은 성과를 통해, 생성된 비디오의 선호도 역시 90%에 달하는 비율로 타 모델에 비해 우위를 점하고 있습니다. 이는 AC3D가 정밀한 카메라 제어를 통해 고품질의 비디오 생성을 가능하게함을 보여줍니다.



### FactCheXcker: Mitigating Measurement Hallucinations in Chest X-ray Report Generation Models (https://arxiv.org/abs/2411.18672)
- **What's New**: 이 논문에서는 의료 비전-언어 모델이 방사선 보고서에서 정확한 정량적 측정을 생성하는 데 어려움을 겪는 문제를 해결하기 위해 FactCheXcker라는 모듈형 프레임워크를 소개합니다. FactCheXcker는 원본 보고서에 기반한 측정 쿼리를 생성하고, 대규모 언어 모델의 코드 생성 기능을 활용하여 잘못된 측정값을 교정합니다. 이 프레임워크는 훈련이나 모델 수정 없이도 모델 생성 보고서를 개선할 수 있도록 설계되었습니다.

- **Technical Details**: FactCheXcker는 세 가지 주요 구성요소로 이루어져 있습니다: Query Generator, Code Generator, Report Updater. Query Generator는 보고서에서 잠재적인 측정 불일치를 식별하고, Code Generator는 이미지에서 정확한 측정을 얻기 위해 특수화된 코드를 생성 및 실행하며, Report Updater는 이 검증된 측정치를 보고서에 통합합니다. 이 시스템은 측정 환각(measurement hallucination)과 객체 환각(object hallucination)의 두 가지 유형으로 발생하는 오류를 범주화합니다.

- **Performance Highlights**: FactCheXcker를 사용하여 11개 의료 보고서 생성 모델에서 ETT(기관내 튜브) 배치의 정확성을 평가한 결과, 이 시스템이 측정 환각을 크게 줄이고, ETT 탐지 및 배치 정확도를 향상시켰음을 보여주었습니다. 평균적으로 FactCheXcker는 측정 환각을 94% 감소시키는 성능 향상을 달성했습니다. 이로 인해 원래 보고서의 전반적인 품질도 유지되었습니다.



### TAPTRv3: Spatial and Temporal Context Foster Robust Tracking of Any Point in Long Video (https://arxiv.org/abs/2411.18671)
- **What's New**: TAPTRv3는 TAPTRv2의 Point Tracking(포인트 추적) 강건성을 장기간 비디오에 맞추어 개선한 새로운 모델입니다. 비디오의 길이가 증가함에 따라 포인트 추적의 변동성이 높아지는 문제를 해결하기 위해 공간적(spatial) 및 시간적(temporal) 맥락을 결합하여 효과적인 feature querying(특징 질의)을 통해 보다 견고한 추적을 가능하게 합니다. 이를 위해 Context-aware Cross-Attention(맥락 인식 교차 주의)와 Visibility-aware Long-Temporal Attention(가시성 인식 장기 시간 주의)와 같은 새로운 기법을 제안합니다.

- **Technical Details**: TAPTRv3는 TAPTRv2의 한계, 특히 긴 비디오에서의 낮은 feature querying 품질을 해결하기 위해 두 가지 주요 기술을 도입합니다. Context-aware Cross-Attention(CCA)은 주변 공간 맥락을 활용하여 이미지를 쿼리할 때 attention score의 품질을 향상시키고, Visibility-aware Long-Temporal Attention(VLTA)은 모든 과거 프레임에 대해 temporal attention을 수행하면서 가시성을 고려하여 feature drifting 문제를 해결합니다. 이러한 방법들은 장기간의 temporal context와 feature 변화를 보다 효과적으로 인식하는 데 기여합니다.

- **Performance Highlights**: TAPTRv3는 TAPTRv2에 비해 대다수의 도전적인 데이터셋에서 성능이 크게 향상되었으며, 최신 성능을 기록합니다. 또한, 대규모 데이터로 훈련된 방법들과 비교하더라도 경쟁력을 유지하여, 비디오 추적 분야에서 새로운 표준을 제시하는 모델로 자리를 잡았습니다. 이러한 성과는 TAPTRv3의 robust(강건한)한 구조와 향상된 특성 질의 메커니즘 덕분이라고 할 수 있습니다.



### SimCMF: A Simple Cross-modal Fine-tuning Strategy from Vision Foundation Models to Any Imaging Modality (https://arxiv.org/abs/2411.18669)
Comments:
          project page: this https URL. arXiv admin note: substantial text overlap with arXiv:2409.08083

- **What's New**: 이번 연구에서는 SimCMF라는 간단하고 효과적인 프레임워크를 제안하여 비전 기초 모델을 다양한 이미징 모달리티로 전이하는 문제를 다룹니다. 이는 자연 RGB 이미지로 훈련된 모델을 다른 물리적 특성을 지닌 이미징 모달리티로 전이할 수 있는 잠재력을 연구하고 있습니다. SimCMF는 교차 모달 정합 모듈과 프리트레인된 모델 백본으로 구성되어 있으며, 모달리티 비조화 문제를 해결합니다.

- **Technical Details**: SimCMF의 설계는 가장 기본적인 디자인에서 시작하여 다양한 기본 구성 요소를 통해 점진적으로 개선되었습니다. 이 과정에서 모달리티 정합을 위한 핵심 구성 요소를 식별하였습니다. 또한, 전체 미세 조정(Full Fine-tuning) 및 파라미터 효율적인 미세 조정(Parameter-efficient Fine-tuning) 전략을 포함한 포괄적인 실증 연구를 제공하여 교차 모달 미세 조정의 가능성을 확인하고 있습니다.

- **Performance Highlights**: SimCMF는 Segment Anything Model(SAM)에 적용되어 다양한 이미지 모달리티에서 우수한 성능을 보여주고 있습니다. 실험 결과, SimCMF는 평가된 모달리티에서 평균적으로 22.15%에서 53.88%로 분할 성능(mIoU)을 향상시킬 수 있었으며, 다른 기준선 모델들보다 consistently 더 나은 성능을 발휘했습니다. 이러한 결과는 SimCMF가 다양한 분야의 이미징 모달리티에 대한 비전 기초 모델 전이에 유용한 도구가 될 수 있음을 보여줍니다.



### Towards Chunk-Wise Generation for Long Videos (https://arxiv.org/abs/2411.18668)
- **What's New**: 이 논문에서는 오토 리그레시브(chunk-by-chunk) 방식으로 긴 비디오 생성을 위한 새로운 접근법을 제시하고 있습니다. 이 방법은 긴 비디오 생성 작업을 여러 개의 짧은 비디오 생성 하위 작업으로 나누어 메모리 요구 사항을 완화하고 각 하위 작업의 비용을 줄일 수 있습니다. 또한 이 논문은 초기 노이즈의 품질이 생성되는 각 청크의 비디오 품질에 미치는 영향을 분석하고 이를 해결하기 위한 효율적인 검색 방법을 제안합니다.

- **Technical Details**: 혁신적인 방법론인 오토 리그레시브(chunk-by-chunk) 비디오 생성을 통해 초기 노이즈 텐서의 품질을 평가하기 위한 빠른 평가 방법을 제안합니다. 이 방법은 k회만큼의 노이즈 제거 단계(denoising steps)를 거쳐 원래의 비디오를 기반으로 촬영된 서브 옵티멀 비디오를 생성할 수 있게 하며, 이를 통해 초기 노이즈가 유효한지 판단할 수 있습니다. 이는 메모리 요구사항을 감축시키고 작은 이미지-비디오(I2V) 모델에서 발생할 수 있는 오류 누적을 완화하는 데 중요한 역할을 합니다.

- **Performance Highlights**: 실험적으로, 오토 리그레시브(chunk-by-chunk) 비디오 생성이 긴 비디오 생성 작업에 적합한 방법이라는 것을 보여주었습니다. 특히, 초기 노이즈의 품질 평가 방법과 더불어 모델의 단순한 오토 리그레시브 접근법이 상당한 성능 향상을 가져올 수 있음을 확인하였습니다. 또한, OpenSoraPlan와 CogVideoX와 같은 큰 I2V 모델은 초기 노이즈에 대한 개입 없이도 충분한 품질을 유지할 수 있음을 입증했습니다.



### Point Cloud Unsupervised Pre-training via 3D Gaussian Splatting (https://arxiv.org/abs/2411.18667)
Comments:
          14 pages, 4 figures, 15 tables

- **What's New**: 본 논문은 GS$^3$라는 효율적인 프레임워크를 제안하며, 이는 3D Gaussian Splatting을 렌더링 기반 프레임워크에 통합하여 포인트 클라우드 표현을 학습합니다. 기존의 자가 지도 학습 기법들은 컴퓨터 자원과 메모리를 많이 소모했지만, GS$^3$는 이러한 문제를 해결하여 사전 훈련 속도를 약 9배 높이고 메모리 사용량을 0.25배 이하로 줄였습니다. 이 프레임워크는 포인트 클라우드 인코더를 RGB-D 이미지와 실제 이미지 간의 비교를 통해 학습합니다.

- **Technical Details**: GS$^3$ 프레임워크는 입력된 희소 RGB-D 이미지를 3D 공간으로 재투영하여 포인트 클라우드를 생성하고, 이 포인트 클라우드에서 포인트 단위 기능을 추출합니다. 학습된 포인트 클라우드 기능을 바탕으로 3D Gaussian 포인트를 예측하고, 타일 기반 래스터라이저를 사용하여 이미지를 렌더링합니다. 이러한 방식으로 포인트 클라우드 인코더의 사전 훈련이 이루어지고, 다양한 다운스트림 3D 작업으로 세밀하게 조정될 수 있습니다.

- **Performance Highlights**: GS$^3$ 프레임워크는 포인트 클라우드 인코더의 탁월한 전이 가능성을 보여주기 위해 3D 세그멘테이션, 탐지 및 재구성 작업에서 다양한 실험을 수행했습니다. 모든 다운스트림 작업에서 뛰어난 성능을 보였고, 이는 GS$^3$가 강력한 초기화 역할을 수행할 수 있음을 시사합니다. 따라서 이 연구는 렌더링 기반 자가 감독 학습 분야에서 새로운 접근법을 제시하며, 많은 실질적 응용 가능성을 열어줍니다.



### 3D Scene Graph Guided Vision-Language Pre-training (https://arxiv.org/abs/2411.18666)
Comments:
          14 pages, 8 figures, 7 tables

- **What's New**: 본 논문에서는 3D 시각-언어(Vision-Language, VL) 추론 분야에서 3D 장면 그래프와 자연어 간의 내재적 연결을 활용하여, 보다 간단하고 통합된 시나리오를 제안합니다. 이를 통해 특정한 작업에 의존하지 않고 다양한 VL 추론 작업에 적용 가능한 범용 모델을 구축하게 됩니다. 새로운 3D 장면 그래프 기반의 비전-언어 사전 훈련(VLP) 프레임워크를 도입하여, 3D 객체와 텍스트 간의 정렬을 둘러싼 컨트라스트 학습(Contrastive Learning) 전략을 개선하였습니다.

- **Technical Details**: 논문에서는 3D 장면 그래프를 기반으로 한 복수 수준의 정렬 메커니즘을 통해 3D 객체와 관련 텍스트를 연결하는 방법을 제안합니다. 이 과정에서 modality encoders, graph convolutional layers 및 cross-attention layers를 활용하여 보편적인 표현을 학습합니다. 특히, Masked Modality Learning을 통해 입력 모달리티의 마스킹된 부분을 재구성하며, 포인트 클라우드의 희소성과 불규칙성을 가정하여 위치 단서를 활용하여 객체의 의미 범주를 예측합니다.

- **Performance Highlights**: 본 연구에서 개발한 3D 장면 그래프 기반 비전-언어 사전 훈련 모델은 여러 세부적인 3D VL 작업에서 기존의 방법들과 비견할 만한 성능을 보여주었습니다. 특히 ScanRefer, Scan2Cap 및 ScanQA와 같은 데이터셋에서 사전 훈련 후 미세 조정을 통해기능이 강화되었습니다. 이로 인해 3D 시각 기초 작업, 3D 밀집 캡션 및 3D 질의 응답 작업에서 우수한 결과를 달성할 수 있었습니다.



### SpotLight: Shadow-Guided Object Relighting via Diffusion (https://arxiv.org/abs/2411.18665)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 기존의 확산 기반( diffusion-based) 신경 렌더러에 조명 제어를 추가하는 	extit{SpotLight}라는 방법을 제안합니다. 사용자에게 착한 제어를 가능하게 하기 위해, 객체의 원하는 그림자를 지정하는 방식으로 조명을 조절할 수 있습니다. 이는 복잡한 물체와 배경 간의 조화를 유지하면서 훈련 없이도 구현됩니다.

- **Technical Details**: SpotLight는 사전 훈련된 확산 기반 신경 렌더러 ℛ를 활용하여, 사용자 정의 그림자 마스크(masks)를 통한 조명 조절을 가능하게 합니다. 이 렌더러는 알베도(albedo), 음영(shading), 표면 법선(normals), 깊이(depth)와 같은 내재적(intrinsic) 맵을 입력받고, 이를 기반으로 RGB 이미지를 출력합니다. 그림자 영상을 안정화할 수 있는 기술적인 기여는 해당 그림자 기반 형상(masks)의 주입을 통해 이루어집니다.

- **Performance Highlights**: SpotLight는 기존의 개별화된 조명 모델들보다 우수한 결과를 도출하였으며, 정량적 및 정성적 평가에서 모두 뛰어난 성능을 입증했습니다. 사용자 연구를 통해 제안된 방법이 실제로 사용자에 의해 선호되는 결과를 만들어내는 것으로 확인되었습니다. 또한, 이를 위해 조명 제어를 위한 새로운 평가 데이터셋이 별도로 발표될 계획입니다.



### Spatiotemporal Skip Guidance for Enhanced Video Diffusion Sampling (https://arxiv.org/abs/2411.18664)
Comments:
          project page: this https URL

- **What's New**: 이번 논문에서는 Spatiotemporal Skip Guidance (STG)라는 새로운 샘플링 가이던스 기법을 소개합니다. STG는 트랜스포머 기반 비디오 확산 모델의 성능을 높이기 위한 간단한 훈련 없는 방법으로, 기존의 CFG(Circular Guidance Function) 방식의 단점을 극복합니다. 이 방법은 외부 모델이나 추가 훈련 없이 자가 변동(self-perturbation)을 통해 암묵적인 약한 모델을 이용합니다.

- **Technical Details**: STG는 시공간 레이어(spatiotemporal layers)를 선택적으로 건너뛰면서 원본 모델의 정렬된 저하 버전을 생성합니다. 이를 통해 샘플 품질(quality)을 향상시키면서도 다양성(diversity)과 동적인 정도(dynamics)를 보존합니다. 이 접근 방식은 추가적인 훈련 과정 없이도 원활하게 작동하며, 비디오 생성에서의 효율적이고 높은 성능의 가이던스를 보장합니다.

- **Performance Highlights**: STG는 기존의 CFG 방식의 문제점을 해결하고, 비디오 확산 모델에 대한 최소한의 변화로 품질을 향상시키는 데 성공했습니다. 이 기술은 다양한 샘플 생성에서도 높은 품질을 유지하는 장점을 가지고 있습니다. 중요한 점은 샘플의 다양성과 동적 요소를 훼손하지 않으면서도 더 나은 결과를 제공한다는 것입니다.



### HoliSDiP: Image Super-Resolution via Holistic Semantics and Diffusion Prior (https://arxiv.org/abs/2411.18662)
Comments:
          Project page: this https URL

- **What's New**: 본 논문에서는 HoliSDiP라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 텍스트-이미지 diffusion 모델을 활용하여 실제 이미지 슈퍼 해상도(Real-ISR)에 대한 향상된 가이드를 제공합니다. 기존 방법의 단점을 극복하기 위해 세분화(Semantic Segmentation)를 활용하여 보다 정확한 텍스트 및 공간 정보를 제공합니다.

- **Technical Details**: HoliSDiP는 세분화된 레이블을 간단한 텍스트 프롬프트로 사용하고, 세분화 마스크와 Segmentation-CLIP Map을 통해 밀집된 세분화 가이드를 제공합니다. 이러한 접근 방식은 노이즈가 많은 텍스트 프롬프트로 인한 부작용을 줄이고, 공간 정보를 효과적으로 통제할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, HoliSDiP는 다양한 Real-ISR 시나리오에서 이미지 품질이 크게 향상되는 것을 보여줍니다. 특히, 프롬프트 노이즈 감소와 공간 제어 향상을 통해 더욱 개선된 결과를 나타냅니다. 이는 이미지 슈퍼 해상도 분야에서 새로운 가능성을 제시합니다.



### OOD-HOI: Text-Driven 3D Whole-Body Human-Object Interactions Generation Beyond Training Domains (https://arxiv.org/abs/2411.18660)
- **What's New**: 이 논문에서는 텍스트 설명을 기반으로 한 전체 신체의 인간-물체 상호작용(Human-Object Interactions, HOIs)을 생성하는 새로운 방법론인 OOD-HOI를 제안합니다. 기존의 방법들이 신체의 일부에만 초점을 맞추었던 것과는 달리, OOD-HOI는 새로운 객체와 행동에 잘 일반화하여 사실적인 상호작용을 생성할 수 있는 가능성을 제공합니다.

- **Technical Details**: OOD-HOI는 이중 브랜치 상호적 확산 모델(dual-branch reciprocal diffusion model)을 통해 초기 상호작용 자세를 합성(flatten)하고, 예측된 접촉 영역을 기반으로 물리적 정확성을 향상시키는 접촉 지향 상호작용 정제기(contact-guided interaction refiner)를 포함합니다. 또한, 의미 조정(semantic adjustment)과 기하 변형(geometry deformation)을 통한 동적 적응 메커니즘(dynamic adaptation mechanism)을 적용하여 강건성을 향상시킵니다.

- **Performance Highlights**: 실험 결과, OOD-HOI는 기존 방법들에 비해 OOD 시나리오에서 더 현실적이고 물리적으로 타당한 3D 상호작용 자세를 생성할 수 있음을 보여주었습니다. 이러한 성과는 가상 현실, 증강 현실, 로봇 공학 및 애니메이션의 여러 응용 분야에서 활용될 수 있는 잠재력을 지니고 있습니다.



### DHCP: Detecting Hallucinations by Cross-modal Attention Pattern in Large Vision-Language Models (https://arxiv.org/abs/2411.18659)
Comments:
          18 pages, 5 figures

- **What's New**: 이 논문에서는 큰 시각-언어 모델(LVLM)의 환각(hallucination) 문제를 효과적으로 탐지하기 위한 새로운 접근 방식을 제안합니다. 제안된 방법인 DHCP(Detecting Hallucinations by Cross-modal Attention Patterns)는 LVLM의 추가 교육이나 추론 단계 없이도 환각을 탐지할 수 있는 경량 감지기를 개발합니다. 실험 결과는 DHCP가 환각 탐지에서 뛰어난 성능을 보임을 보여주며, LVLM의 신뢰성과 신뢰성을 높이는 데 기여합니다.

- **Technical Details**: LVLM은 일반적으로 CLIP 등을 사용하여 입력 이미지를 시각적 피쳐로 인코딩하는 시각 인코더, 시각-언어 모델, 시각-언어 정렬 모듈의 세 가지 구성 요소로 구성됩니다. 이 논문에서는 환각 상태에서 LVLM이 보이는 시각 토큰에 대한 교차 모드 주의(cross-modal attention) 패턴의 변화를 조사하여 이를 통해 환각을 탐지할 수 있는지 분석하였습니다. 이를 통해, 잘못된 객체를 인식하거나 대답이 불확실한 경우처럼 LVLM이 보이는 주의 패턴을 탐지하여 환각 상태를 파악합니다.

- **Performance Highlights**: DHCP는 심층적인 실험을 통해 여러 데이터 세트에서 시각-언어 모델의 환각 탐지에서 일관되게 높은 성능을 입증했습니다. 허위 경고가 발생한 경우가 있었고, 이들은 모델이 답변에 대한 불확실성을 보였던 경우에 주로 발생했습니다. 또한, 초기 실험 결과는 DHCP가 생성적 작업에서도 효과적으로 적용될 가능성을 시사합니다.



### HDI-Former: Hybrid Dynamic Interaction ANN-SNN Transformer for Object Detection Using Frames and Events (https://arxiv.org/abs/2411.18658)
Comments:
          17 pages, 11 figures

- **What's New**: 이 논문에서는 HDI-Former라는 하이브리드 동적 상호작용 ANN-SNN 트랜스포머를 제안합니다. 이는 프레임과 이벤트를 사용하여 고정밀도 및 에너지 효율적인 객체 탐지를 위해 직접 훈련된 최초의 하이브리드 ANN-SNN 아키텍처입니다. 기존의 방법들이 두 개의 독립적인 ANN 가지를 사용하는 데 비해, HDI-Former는 상호작용을 강조하여 성능을 크게 개선하고 있습니다.

- **Technical Details**: 기술적으로, HDI-Former는 새로운 의미 강화(self-attention) 메커니즘을 도입하여 ANN 트랜스포머 가지 내 이미지 인코딩 토큰 간의 상관관계를 강화합니다. 또한, 에너지 효율적인 Spiking Swin Transformer 가지를 설계하여 이벤트 스트림으로부터의 시간적 단서를 모델링합니다. 마지막으로, ANN과 SNN 서브 네트워크 간의 생체영감을 받은 동적 상호작용 메커니즘을 제안하여 교차 모달리티(interaction) 정보를 효과적으로 교환합니다.

- **Performance Highlights**: HDI-Former는 11개의 최신 기술(state-of-the-art) 방법 및 4개의 베이스라인을 크게 능가하는 성능을 보여줍니다. SNN 가지는 동일한 아키텍처의 ANN과 비교하여 10.57배 적은 에너지를 소비하면서 유사한 성능을 발휘합니다. 이러한 결과는 HDI-Former가 성능과 에너지 소비 간의 균형을 잘 이룬다는 것을 보여줍니다.



### AToM: Aligning Text-to-Motion Model at Event-Level with GPT-4Vision Reward (https://arxiv.org/abs/2411.18654)
- **What's New**: 최근 텍스트-모션 모델이 더 현실적인 인간 모션을 생성하는 새로운 가능성을 열었습니다. AToM이라는 프레임워크를 도입하여, GPT-4Vision의 피드백을 활용하여 생성된 모션과 텍스트 프롬프트 간의 정렬을 향상시켰습니다. 이 프레임워크는 모션 생성과 관련된 이벤트 수준 텍스트 설명 간의 복잡한 관계를 해결하기 위해 설계되었습니다.

- **Technical Details**: AToM은 세 가지 주요 단계로 구성됩니다. 첫째, 세 가지 유형의 이벤트 수준 텍스트 프롬프트와 생성된 모션의 쌍을 이루는 MotionPrefer 데이터셋을 구축합니다. 둘째, GPT-4Vision을 이용한 세부 모션 주석을 위한 패러다임을 설계하여 텍스트-모션 정렬 점수를 평가합니다. 마지막으로, 강화 학습을 통해 기존의 텍스트-모션 모델을 미세 조정합니다.

- **Performance Highlights**: 실험 결과, AToM은 텍스트-모션 생성의 이벤트 수준 정렬 품질을 크게 개선했습니다. MotionPrefer 데이터셋은 기존 데이터셋보다 더 큰 스케일과 품질을 자랑하며, 5.3K 텍스트 프롬프트와 80K 모션 선호 쌍으로 구성되어 있습니다. AToM의 효과적인 성능 개선은 다양한 모션 주입 방법, LoRA, 점수 필터링 및 강화 학습 전략의 효과를 탐구한 결과에서도 입증되었습니다.



### Surf-NeRF: Surface Regularised Neural Radiance Fields (https://arxiv.org/abs/2411.18652)
Comments:
          20 pages, 17 figures, 9 tables, project page can be found at this http URL

- **What's New**: 이번 논문에서는 Neural Radiance Fields(NeRF)의 기하학적 정확성을 높이기 위한 새로운 접근법을 제안합니다. 구체적으로, surface light field 모델의 커리큘럼 학습을 통해 NeRF가 더 정밀한 장면 표Representation에 수렴할 수 있도록 돕습니다. 추가적으로, 기하학적 부드러움과 노멀의 일관성, 그리고 램버트와 스페큘러 외관의 분리를 위한 네 가지 규제 항을 도입합니다. 이러한 방법으로 기존 NeRF 모델보다 14.4% 높은 노멀 정확도를 달성하였습니다.

- **Technical Details**: Surf-NeRF는 surface light field의 특성을 지역적으로 강화하기 위해 새로운 정규화 기법을 소개합니다. 이 기법을 활용하면 기하학적인 구조와 외관을 물리적으로 올바르게 분리하여, 더 정확한 radiance field를 생성할 수 있습니다. 이러한 접근법은 밀도와 외관의 연속적인 영역을 장려하여 shape-radiance ambiguity를 줄입니다. NeRF의 표면 위치를 가정하고, 이 지점에서 추가 샘플링을 수행하며 부드럽게 변하는 노멀을 생성하여 장면 기하를 보다 현실적으로 나타냅니다.

- **Performance Highlights**: 제안한 방법은 기존의 반사 기반 NeRF 변형에 비해 위치 부호화 NeRF에서 14.4% 및 그리드 기반 모델에서 9.2%의 성능 개선을 보여주었습니다. 또한, 기하학적 표Representation이 촬영된 장면과 일치하도록 조절된 NeRF를 통해 분리된 뷰 종속 외관을 구현하는 등의 일관성을 보장합니다. 이러한 성능 개선은 로봇 조작 및 내비게이션과 같은 복잡한 환경에서 기하학과 시각적 일관성이 중요한 응용 분야에서 중요한 의미를 가집니다.



### Verbalized Representation Learning for Interpretable Few-Shot Generalization (https://arxiv.org/abs/2411.18651)
- **What's New**: 이번 연구에서는 Verbalized Representation Learning (VRL)이라는 새로운 방법론을 제안합니다. 이 방법은 적은 수의 데이터로 객체 인식을 위한 인간 해석 가능한 특징을 자동으로 추출합니다. VRL은 Vision-Language Model (VLM)을 활용하여 서로 다른 클래스 간의 차이와 같은 클래스 내 공통점을 자연어로 포착합니다.

- **Technical Details**: VRL은 몇 가지 이미지 샘플을 사용하여 자연어로 서술된 특징을 수치 벡터로 변환합니다. 이러한 수치적 특징은 분류기(Classifier) 훈련 및 추론에 사용될 수 있습니다. 연구에서는 이 방법이 iNaturalist 및 Kiki-Bouba 데이터셋에서 우수한 성능을 보여주었으며, 적은 데이터로도 24%의 성능 향상을 달성했습니다.

- **Performance Highlights**: VRL은 또한 인간이 주석한 특징들보다 20% 더 나은 성능을 보였습니다. 이로 인해 VRL은 세밀한 분류 작업에서도 높은 범용성을 보여줍니다. 결과적으로, 이 연구는 VRL이 모델의 해석 가능성을 높이며 낮은 자원 조건에서도 효과적으로 일반화할 수 있는 가능성을 제시합니다.



### RoMo: Robust Motion Segmentation Improves Structure from Motion (https://arxiv.org/abs/2411.18650)
- **What's New**: 이 논문에서는 정적 세계 프레임에 대한 동적 요소를 식별하기 위한 새로운 비디오 기반 모션 세분화 접근법인 RoMo를 제안합니다. 이 방법은 광학 흐름(optical flow)과 에피폴라 구속조건(epipolar constraints)을 결합하고, 사전 학습된 비디오 세분화 모델을 활용하여 동적 장면의 세그먼트 마스크를 추론합니다. RoMo는 기존의 감독되지 않은 모션 세분화 방법과 합성 데이터로 훈련된 감독된 방법보다 월등한 성능을 보여줍니다.

- **Technical Details**: RoMo 접근법은 동적 객체로부터의 카메라 유도 모션을 분리하는 데 중점을 둡니다. 주된 단계는 인접한 프레임 간의 광학 흐름을 분석하여 가능성 높은 정적 픽셀을 식별하고, 이를 통해 얻은 잡음이 섞인 레이블을 사전 훈련된 세분화 모델의 특징과 결합하여 고품질의 세분화 마스크를 생성하는 것입니다. 이 절차는 반복적으로 이루어지며, 결과적으로 에피폴라 기하학의 추정치를 정제합니다.

- **Performance Highlights**: 실험 결과, RoMo는 동적 장면 벤치마크에서 카메라 포즈 추정 및 모션 세분화에서 최신 기법들보다 상당히 우수한 성능을 발휘합니다. 특히, MPI Sintel 데이터셋 및 DAVIS16, SegTrackv2와 같은 모션 세분화 벤치마크에 대해 뛰어난 성능을 보여줍니다. 새로운 데이터셋에서는 RoMo 기반의 SfM 파이프라인이 이전의 최신 기법들보다 훨씬 더 높은 성능을 발휘함을 입증하였습니다.



### Bi-ICE: An Inner Interpretable Framework for Image Classification via Bi-directional Interactions between Concept and Input Embeddings (https://arxiv.org/abs/2411.18645)
Comments:
          The first two authors equally contributed to this work, 27 pages, 19 figures, 9 tables

- **What's New**: 이 논문은 인공지능(AI) 시스템의 내부 메커니즘을 이해하기 위한 새로운 접근 방식인 inner interpretability를 소개합니다. 특히, 대규모 이미지 분류 과제를 위한 다단계 분석을 지원하는 개념적 프레임워크를 제시하고, Bi-directional Interaction between Concept and Input Embeddings (Bi-ICE) 모듈을 도입하여 해석 가능성을 높입니다. Bi-ICE 모듈은 입력값에 대한 예측을 인간이 이해할 수 있는 개념에 기반하여 수행하며, 개념의 기여도를 측정하고 이를 입력값 내에서 지역화합니다.

- **Technical Details**: 원천적으로 이미지 분류를 위한 개념 분해 접근 방식을 채택하여, 논문은 다단계 분석에서 발생하는 컴퓨터 및 알고리즘 수준을 정의합니다. 이 과정에서 기초적인 개념과 그 개념을 설명하는 벡터 표현을 설정합니다. Bi-ICE 모듈은 모델의 중심 허브 역할을 하여 이미지 임베딩을 수집하고, 이를 통해 여러 레벨에서 해석 가능성을 제공합니다.

- **Performance Highlights**: 논문은 이미지 분류 작업에서의 enhanced transparency를 통해 개념의 기여도와 위치를 측정하는 방법을 보여줍니다. Bi-ICE의 활용은 개념 학습과 수렴 과정의 알고리즘적 해석 가능성을 강조하며, 벡터화된 개념이 어떻게 모델의 결정에 기여하는지를 시각적으로 설명합니다. 본 연구는 이미지 기반 Explainable AI(XAI) 접근 방식에서 다단계 해석 가능성을 실질적으로 구현하는 데 기여할 것입니다.



### Scene Co-pilot: Procedural Text to Video Generation with Human in the Loop (https://arxiv.org/abs/2411.18644)
Comments:
          Videos are available at our project page: this https URL

- **What's New**: 이번 논문에서는 Scene Copilot이라는 프레임워크를 제안하여, 대형 언어 모델(LLMs)과 절차적 3D 장면 생성기를 결합했습니다. 이 프레임워크는 사용자 입력을 기반으로 3D 장면을 쉽게 생성할 수 있도록 돕는 기능을 포함하고 있습니다. 이를 통해, 사용자는 Blender의 사용자 인터페이스(UI)를 통해 직관적으로 3D 장면의 세부 사항을 조작하고 최종 출력 비디오를 제어할 수 있습니다.

- **Technical Details**: Scene Copilot은 Scene Codex, BlenderGPT와 함께 Human in the Loop를 포함하는 구조를 가지고 있습니다. Scene Codex는 텍스트 기반의 사용자 입력을 3D 장면 생성기가 이해할 수 있는 명령어로 변환합니다. BlenderGPT는 생성된 3D 장면을 보다 정밀하게 제어할 수 있는 직관적이고 직접적인 경로를 사용자에게 제공합니다. 또한, 코드 형식의 절차적 데이터셋을 통해 시스템의 기능을 더욱 향상시켰습니다.

- **Performance Highlights**: 다양한 실험을 통해 Scene Copilot 프레임워크가 3D 장면 및 비디오 생성의 사용자 정의 가능성을 보여주었습니다. 기존 LLMs를 활용하여 훈련 과정 없이도 시스템을 업데이트할 수 있는 점이 큰 장점으로 작용합니다. 이러한 시스템은 네이티브로 지원되는 멀티모달 기능을 통해 사용자 친화적인 경험을 제공하며, 최종적으로 사용자가 원하는 세부 사항을 간편하게 조정할 수 있도록 돕습니다.



### VLSBench: Unveiling Visual Leakage in Multimodal Safety (https://arxiv.org/abs/2411.19939)
- **What's New**: 이번 연구에서는 Multimodal large language models (MLLMs)의 안전성과 관련하여 새로운 문제를 제기하고 있습니다. 특히, 기존의 데이터셋에서 시각적 정보 누출(Visual Safety Information Leakage, VSIL) 문제가 존재한다는 점을 발견했습니다. 이로 인해 MLLMs가 이미지와 텍스트 쿼리를 기반으로 민감한 내용을 쉽게 감지할 수 있으나, 실제 상황에서는 VSIL이 없는 이미지-텍스트 쌍도 많다는 것을 지적합니다. 이 연구는 이러한 문제를 해결하기 위한 새로운 벤치마크인 VLSBench를 구축했습니다.

- **Technical Details**: VLSBench는 2.4k 개의 이미지-텍스트 쌍을 포함한 멀티모달 안전 기준으로, 시각적 안전 정보의 누출을 방지합니다. 연구자들은 LLM을 활용하여 해로운 텍스트 쿼리와 이미지를 생성하고, 이후 이를 정제하여 안전한 텍스트 쿼리와 질 높은 이미지를 조합했습니다. 실험 결과, VLSBench는 기존의 MLLM 뿐만 아니라 공개 및 폐쇄 소스 모델에게도 상당한 도전 과제가 됩니다. 특히, 기존 안전 벤치마크가 VSIL 문제에 취약하다는 것을 경험적으로 확인했습니다.

- **Performance Highlights**: VLSBench에서의 실험 결과에 따르면, 단순한 텍스트 SFT 방법이 MLLM의 안전성을 유지하면서도 95% 이상의 성능을 달성할 수 있음을 보여줍니다. 반면, 멀티모달 정렬 방법은 VSIL이 없는 경우에 더 유망한 해결책이 될 수 있습니다. 이러한 발견들을 기반으로, MLLMs의 안전성 접근 방식에서 텍스트 및 멀티모달 정렬 방법의 효용성을 비교했습니다. 결과적으로, 시각적 안전 정보 누출이 텍스트 방식의 우수한 성능의 원인임을 입증했습니다.



### On Domain-Specific Post-Training for Multimodal Large Language Models (https://arxiv.org/abs/2411.19930)
- **What's New**: 최근 다수의 다양한 분야에 적응할 수 있는 다중 모달 대형 언어 모델(MLLM)의 발전이 두드러진 성과를 내고 있습니다. 하지만 특정 과학 분야나 산업 응용 분야에 대한 모델의 적응은 상대적으로 덜 탐구되었습니다. 본 논문은 포스트 트레이닝(post-training)을 통한 MLLM의 도메인 적응(domain adaptation)을 체계적으로 연구하며, 데이터 생성(data synthesis), 트레이닝 파이프라인(training pipeline), 작업 평가(task evaluation)에 초점을 맞춥니다.

- **Technical Details**: 논문에서는 도메인 특화 이미지-자막 쌍을 사용하여 다양한 비주얼 지침 작업을 생성하는 비주얼 지침 합성기를 개발하였습니다. 생성된 합성 작업은 수동 규칙이나 GPT-4 및 GPT-4V에서 생성된 작업보다 MLLM의 도메인 특화 성능 향상에 효과적입니다. 또한, 단일 단계 교육 파이프라인을 적용하여 각 트레이닝 예제에서 합성 작업과 이미지-자막 쌍을 결합하여 작업 다양성을 증대시킵니다.

- **Performance Highlights**: 두 개의 도메인인 생물 의학(biomedicine)과 식품(food)에서 다양한 MLLM의 성능을 평가하였으며, 결과적으로 우리의 모델인 AdaMLLM은 여러 도메인 특화 작업에서 일반 MLLM보다 지속적으로 우수한 성능을 보였습니다. 이러한 결과는 도메인 특화 포스트 트레이닝이 MLLM의 성능 개선에 효과적임을 입증합니다.



### Towards Class-wise Robustness Analysis (https://arxiv.org/abs/2411.19853)
- **What's New**: 이 연구는 적대적 훈련(adversarial training)을 통해 강인하게 훈련된 분류 모델의 클래스 간 편향(class-to-class biases)을 조사하여 각 클래스의 강점과 약점을 분석하는 데 중점을 둡니다. 기존 연구들은 전반적인 모델 강인성 개선에만 집중했던 반면, 이 연구는 클래스별 취약성에 대한 중요성을 부각시킵니다. 특히, 클래스 간의 혼동이 어떻게 발생하는지를 이해하는 것이 중요하다고 강조합니다.

- **Technical Details**: 연구에서 제안된 Class False Positive Score (CFPS)는 각 클래스별 오분류 정도를 측정하는 혁신적인 방법으로, 특정 클래스에 대해 다른 클래스의 샘플이 얼마나 잘못 분류되는지를 평가합니다. 이 평가는 클래스의 취약성을 나타내며, CFPS가 높은 클래스는 공격자에 의해 쉽게 조작될 수 있습니다. 또한, CFPS는 기존의 클래스 정확도(class-wise accuracy)와 보완적인 관계를 가지며, 모델의 강인성과 취약성을 이해하는 데 중요한 통찰력을 제공합니다.

- **Performance Highlights**: CIFAR-10 데이터셋을 대상으로 한 실험 결과, 특정 클래스가 다른 클래스보다 낮은 정확도를 나타내는 경향이 발견되었습니다. 특히, 클래스 C3(새), C5(사슴), C6(개)는 약한 클래스로 분류되었으며, 이는 모든 강인 모델에서 일관되게 나타났습니다. 이러한 결과는 이미지 인식 모델의 신뢰성과 보안을 향상시키기 위해 취약한 클래스를 집중적으로 개선하는 것이 필요함을 시사합니다.



### Dual Risk Minimization: Towards Next-Level Robustness in Fine-tuning Zero-Shot Models (https://arxiv.org/abs/2411.19757)
Comments:
          NeurIPS 2024

- **What's New**: 이 논문에서는 Dual Risk Minimization (DRM)라는 새로운 접근 방식을 제안합니다. DRM은 전통적인 경험적 위험 최소화(ERM)와 최악의 경우 위험 최소화(WRM)를 결합하여 다운스트림 작업의 핵심 특징을 더 잘 보존합니다. 이 방식은 다양한 실제 벤치마크에서 새로운 최첨단 성능을 달성하는 데 중요한 역할을 합니다.

- **Technical Details**: DRM은 LLM을 통해 생성된 핵심 특징 설명을 활용하여 제로-샷(zero-shot) 예측을 유도합니다. 이렇게 생성된 예측은 최악의 경우 위험을 추정하는 대리 지표로 사용됩니다. 이 방법은 모델의 기대 성능과 최악의 경우 성능 사이의 균형을 맞추어 더욱 향상된 강건성을 제공합니다.

- **Performance Highlights**: DRM은 이미지넷(ImageNet)에서 CLIP ViT-L/14@336 모델의 OOD 성능을 75.9에서 77.1로 향상시키는 등 여러 데이터셋에서 성능을 크게 개선했습니다. WILDS-iWildCam과 WILDS-FMoW에서도 각각 47.1에서 51.8, 50.7에서 53.1로 성능이 향상되었습니다. 이 연구는 강건한 파인 튜닝의 새로운 가능성을 열어주며, 탁월한 한계를 설정했습니다.



### A Comprehensive Content Verification System for ensuring Digital Integrity in the Age of Deep Fakes (https://arxiv.org/abs/2411.19750)
- **What's New**: 이번 논문에서는 디지털 콘텐츠의 공유가 증가하는 시대에서, 콘텐츠 진위 확인의 필요성을 강조하고 있습니다. 개인 소셜 미디어 플랫폼을 넘어서는 이 시스템은 전통적인 블루 틱(verified profiles) 이상의 의미를 가지며, 사용자가 공유하는 이미지와 영상의 진위를 인증할 수 있도록 돕습니다. 특히, DALL-E나 Sora와 같은 쉽게 접근 가능한 AI 도구의 발전으로 인해 정보의 왜곡 위험이 커진 점을 지적합니다.

- **Technical Details**: 논문에서는 이미지를 게시물이나 스토리로 공유할 때 진위를 인증할 수 있는 콘텐츠 검증 시스템(Content Verification System)을 제안합니다. 이 시스템은 리셰어(resharing), 리포스팅(reposting) 등의 복잡한 네트워크를 통해 전파되는 콘텐츠의 신뢰성을 보장하는 데 필요한 장치를 제공합니다. 이런 방식으로 개인과 인플루언서들은 디지털 발자취의 신뢰성을 강화할 수 있습니다.

- **Performance Highlights**: 제안된 시스템은 블루 틱의 제한을 초월하며, 개인의 평판을 보호하는 데 중점을 둡니다. 이 시스템은 사용자에게 콘텐츠의 진정성을 검증할 수 있는 도구를 제공함으로써, 정보의 신뢰성을 높이고 소셜 미디어 플랫폼에서의 정보 혼란을 줄이는 데 기여할 것입니다.



### JetFormer: An Autoregressive Generative Model of Raw Images and Tex (https://arxiv.org/abs/2411.19722)
- **What's New**: 이번 연구에서는 JetFormer라는 새로운 autoregressive decoder-only transformer를 제안하여, 기존의 modality-specific components에 의존하지 않고 이미지를 직접 생성할 수 있게 되었습니다. 이 모델은 정상화 흐름(normalizing flow) 모델을 활용하여 이미지의 soft-token 표현을 얻고, 이를 autoregressive multimodal transformer와 함께 훈련시킵니다. 결과적으로 JetFormer는 고화질 이미지 생성 앱을 위한 강력한 가능성을 보여주며, 이전의 pretrained 이미지 autoencoder에 비해 경쟁력 있는 성능을 자랑합니다.

- **Technical Details**: JetFormer는 최신의 대화형 생성 모델을 지원하기 위해 설계되었습니다. 이 모델은 Gaussian mixture loss를 포함하여 훈련 중에 Gaussian noise의 변화를 조절함으로써 고차원 이미지를 생성하는 데 중점을 둡니다. 또한, JetFormer는 PCA 기법을 통해 자연 이미지의 중복성을 관리하는 두 가지 방법을 탐색하며, 이미지와 텍스트를 통합적으로 처리할 수 있는 구조적인 이점을 제공합니다.

- **Performance Highlights**: JetFormer는 텍스트-이미지 생성 및 비전-언어 이해 작업에서 이전의 모델들과 비교하여 뛰어난 성과를 보입니다. 특히, ImageNet의 클래스 조건 이미지 생성 및 웹 규모의 멀티모달 생성 실험에서 그 성능이 입증되었습니다. JetFormer는 통합된 아키텍처로 처음부터 끝까지 훈련할 수 있으며, 강력한 likelihood bounds를 생성할 수 있는 첫 번째 모델로 자리잡고 있습니다.



### The Streetscape Application Services Stack (SASS): Towards a Distributed Sensing Architecture for Urban Applications (https://arxiv.org/abs/2411.19714)
- **What's New**: SASS(Streetcape Application Services Stack)는 복잡한 도시 애플리케이션 개발을 지원하는 구조화된 프레임워크로, 모듈화 가능하고 분산된 어플리케이션 스택을 제공합니다. 이 시스템은 다양한 센서 유형으로부터의 데이터를 통합하고 실시간 처리의 복잡성을 줄이는 핵심 서비스들을 제공하여, 스마트 시티의 요구를 충족시키고 있습니다. 이를 통해 SASS는 기존 도시 시스템의 프로그래머블 및 이동성 문제를 해결하며, 다양한 인프라 환경에 적응할 수 있는 유연한 솔루션을 가능하게 합니다.

- **Technical Details**: SASS는 멀티모달 데이터 동기화, 시공간 데이터 융합(spatiotemporal data fusion), 분산 엣지 컴퓨팅(distributed edge computing)이라는 세 가지 기본 서비스를 제공합니다. 이러한 서비스는 이질적인 센서 유형에서의 데이터 스트림 동기화, 시공간적으로 분산된 데이터를 통합하고 경량 처리로 지연 시간을 감소시킵니다. SASS의 설계는 모듈화된 형식으로, 다양한 도시 센서 및 구성 요소를 관리하고 조합하기 쉽게 만들어 개발자들이 나름대로 응용 프로그램을 조직할 수 있도록 돕습니다.

- **Performance Highlights**: SASS는 두 개의 실제 테스트 환경에서 평가되었으며, 그 결과를 통해 멀티모달 데이터 동기화 서비스가 시간 불일치 오류를 88% 감소시킨 것으로 나타났습니다. 또한, 시공간 데이터 융합 서비스는 보행자와 차량의 탐지 정확도를 10% 이상 향상시켰으며, 분산 엣지 컴퓨팅 서비스는 시스템 처리량을 10배 이상 증가시켰습니다. 이러한 성과들은 SASS가 실시간이며 확장 가능한 도시 응용 프로그램을 지원하는 데 필요한 성능과 추상화를 제공한다는 것을 보여줍니다.



### Multimodal Whole Slide Foundation Model for Pathology (https://arxiv.org/abs/2411.19666)
Comments:
          The code is accessible at this https URL

- **What's New**: 최근 발전된 foundation model을 통해 컴퓨터 병리학(computational pathology)의 연구가 혁신적으로 변화하고 있습니다. 연구진은 335,645개의 whole slide image(WSI)를 사용하여 self-supervised learning(SSL)과 시각-언어 정렬(vision-language alignment)을 활용한 TITAN이라는 다중 모달 모델을 제안했습니다. TITAN은 희귀 질환 검색 및 암 예후 예측과 같은 자원이 제한된 임상 시나리오에 적합한 일반적인 슬라이드 표현(slide representations)과 병리 보고서 생성 기능을 가지고 있습니다.

- **Technical Details**: TITAN 모델은 시각 전용과 시각-언어 모델로 나뉘며, 베이스라인으로는 실험에서 선형 프로빙(linear probing), 몇 장의 샘플(few-shot) 및 제로샷(zero-shot) 분류 개념을 채택하였습니다. 기존의 병리학 리포트를 포함하여 423,122개의 합성 캡션(synthetic captions)을 생성하는 다중 모달 생성 AI 코파일럿(multimodal generative AI copilot)으로부터 학습되었습니다. 이 모델은 추가적인 미세 조정(fineting)이나 임상 레이블(clinical labels) 없이도 성능을 발휘합니다.

- **Performance Highlights**: TITAN은 다양한 임상 작업에서 평가되었으며, ROI 및 슬라이드 기반 foundation 모델보다 탁월한 성능을 보였습니다. 특히, 희귀 암 검색(rare cancer retrieval) 및 크로스 모달 검색(cross-modal retrieval) 분야에서도 뛰어난 결과를 기록하며, 병리학 보고서 생성(pathology report generation)에서도 효과적임을 입증하였습니다. 이 연구는 임상 데이터가 제한된 상황에서도 실질적인 병리학적 분석 및 보고에 기여할 수 있는 가능성을 보여줍니다.



### CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation (https://arxiv.org/abs/2411.19650)
Comments:
          Project Webpage: this https URL

- **What's New**: 본 논문에서는 VLM에서 파생된 새로운 VLA 아키텍처인 CogACT를 소개합니다. 기존의 VLA 모델은 VLM을 단순한 방식으로 행동 예측에 재사용했으나, CogACT는 VLM의 출력을 조건으로 하는 특화된 행동 모듈을 제안합니다. 이 아키텍처는 행동 신호의 특성을 효과적으로 처리하며, 분산 기반 변환기(diffusion transformer)를 사용하여 성능을 크게 향상시킵니다.

- **Technical Details**: CogACT는 인지(cognition)와 행동(action) 능력을 분리하여 설계되었습니다. 이는 인간의 뇌처럼 각기 다른 뇌 영역이 시각, 언어 및 운동 기능을 처리하는 방식에 착안했습니다. 또한, 행동 모듈을 위한 다양한 백본 아키텍처를 체계적으로 연구했으며, 특히 Diffusion transformers를 사용한 순차적 모델링이 단일 단계 행동 예측보다 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 실험 결과, CogACT 모델은 기존 VLA 모델보다 뛰어난 작업 성능을 보였습니다. OpenVLA와 비교했을 때, 시뮬레이션 평가에서 35% 이상, 실제 로봇 실험에서 55% 이상의 성공률 향상을 기록했습니다. 또한, 큰 RT-2-X 모델보다도 18% 높은 절대 성공률을 실험에서 달성하며, 새로운 로봇과 작업에 대한 빠른 적응력을 보여주었습니다.



### Self-Supervised Denoiser Framework (https://arxiv.org/abs/2411.19593)
- **What's New**: 본 논문에서는 산업 환경에서 X-ray CT를 활용한 이미지 복구의 도전 과제를 다루고 있습니다. 특히, 짧은 스캔 시간과 높은 처리량을 요구하는 비파괴 검사 시나리오에서의 복원 기술의 한계를 극복하기 위한 Self-supervised Denoiser Framework (SDF)를 소개합니다. SDF는 세밀하게 샘플링된 sinogram 데이터를 사전 훈련으로 활용하여, 샘플링이 부족한 sinogram 데이터로부터 복원된 이미지의 품질을 향상시키는 방법론입니다.

- **Technical Details**: SDF는 이미지 잡음을 제거하는 네트워크를 sinogram 공간에서 훈련시키는 방식을 채택합니다. 이는 별도의 ground-truth 이미지 데이터 없이, 하나의 sinogram 하위 집합으로부터 다른 sinogram 하위 집합을 예측하는 학습 작업으로 구성됩니다. 이러한 접근은 CT의 abundant data 모달리티인 sinogram을 활용하며, 적은 수의 측정 결과에서도 이미지를 크게 향상할 수 있는 효과를 보여줍니다.

- **Performance Highlights**: 실험 데이터셋을 기반으로 SDF는 다른 분석적 방법 및 자가 감독 방식을 초월한 이미지 품질을 생성함을 입증했습니다. 특히, 2D 팬빔 및 3D 콘빔 CT 설정 모두에서 신호 대 잡음 비율이 더 높은 성능을 나타냈습니다. 또한, 고품질 이미지 데이터가 부족한 상황에서 이미지 잡음 제거기를 미세 조정할 때도 SDF의 향상된 성능이 지속되는 것을 확인했습니다.



### A Comprehensive Framework for Automated Segmentation of Perivascular Spaces in Brain MRI with the nnU-N (https://arxiv.org/abs/2411.19564)
Comments:
          46 pages, 8 figures, 2 tables

- **What's New**: 이번 연구는 뇌의 혈관 주변 공간(Perivascular Spaces, PVS)의 자동 세분화를 위해 널리 사용되는 딥 러닝 모델, no-new-UNet (nnU-Net)을 최적화했습니다. PVS의 확대는 여러 신경퇴행성 질환과 관련이 있으며, 이를 정확하게 탐지할 수 있는 방법이 필요합니다. 연구팀은 건강한 참가자로부터 MRI 이미지를 수집하고, 다양한 이미지 처리 전략을 비교했습니다.

- **Technical Details**: 총 30명의 건강한 참가자에게서 T1 가중 MRI 이미지를 세 가지 다른 프로토콜로 얻었으며, 각 참가자에 대해 10개의 축 방향 슬라이스에서 PVS를 수동으로 세분화했습니다. 다양한 사전 처리 및 준지도 학습 방법을 사용하여 11개의 모델을 비교했으며, 성능 평가는 5회 교차 검증(5-fold cross validation, 5FCV)을 통해 이루어졌습니다. 주요 성능 지표는 Dice Similarity Coefficient (DSC)를 사용하였습니다.

- **Performance Highlights**: 모델의 성능은 반복적인 라벨 정리를 통해 급격히 향상되어 DSC는 85.7%에 도달했습니다. 또한 18개의 추가 데이터셋에서 얻은 가상 라벨을 이용한 준지도 학습으로 PVS 클러스터 수의 예측 정확도가 크게 개선되었습니다. 이번 연구는 뇌 MRI에서 PVS의 자동 정량화를 위한 강력하고 포괄적인 프레임워크를 제공하고 있습니다.



### Contextual Checkerboard Denoise -- A Novel Neural Network-Based Approach for Classification-Aware OCT Image Denoising (https://arxiv.org/abs/2411.19549)
Comments:
          Under review in Springer Journal of Medical Systems. Code available: this https URL

- **What's New**: 이 논문에서는 의료 이미지의 중요한 세부정보를 보존하면서 노이즈 이미지 집합에서만 학습할 수 있는 새로운 신경망 기반 방법인 'Contextual Checkerboard Denoising'을 도입합니다. 이를 통해 의료 이미지의 분류 및 분석에 필요한 해부학적 세부정보를 잃지 않으면서 이미지 품질을 개선합니다. 또한, 저자들은 진단 정확도를 향상시키면서 더 선명하고 세부적인 Optical Coherence Tomography (OCT) 이미지를 제공하는 기법을 실험적으로 입증하였습니다.

- **Technical Details**: 의료 이미지의 노이즈 제거는 원본 이미지를 추정하는 작업으로, 기존의 방법들은 노이즈를 제거하는 과정에서 중요한 세부정보를 손실하는 경향이 있습니다. 저자들은 새로운 손실 함수(custom loss function)를 제안하여 분류 손실(classification loss)과 노이즈 제거 손실(denoising loss)을 모두 고려하여 훈련합니다. 이에 따라, 이 방법은 단일 노이즈 이미지 데이터세트에서 학습할 수 있으며, 자가 감독(self-supervised) 방식으로 쉽게 구현할 수 있습니다.

- **Performance Highlights**: 제안된 'Contextual Checkerboard Denoising' 방법은 실제 OCT 이미지에서 실험을 수행한 결과, 이미지 품질을 현저히 개선하면서 중요한 세부 사항을 유지하는 데 성공했습니다. 이 방법은 기존의 접근법들에 비해 높은 진단 정확도를 보이며, 의료 이미지 분석에 필수적인 임상 세부정보를 보호하는 것이 중요한 특징입니다. 특히, 이 연구는 의료 이미지의 노이즈 문제를 해결하기 위한 심층 학습 기반 기법의 필요성을 강조하고 있습니다.



### Subjective and Objective Quality Assessment Methods of Stereoscopic Videos with Visibility Affecting Distortions (https://arxiv.org/abs/2411.19522)
Comments:
          13 pages

- **What's New**: 이번 연구에서는 두 가지 주요 기여를 소개합니다. 첫 번째로, 12개의 참조 비디오와 360개 왜곡된 비디오로 구성된 Full HD 해상도의 입체 영상 데이터셋(S3D dataset)을 생성했습니다. 이 테스트 자극은 원본 좌우 비디오 시퀀스에 안개와 연무 환경을 시뮬레이션하여 제작되었습니다.

- **Technical Details**: 두 번째로, 우리는 S3D 비디오를 위한 Opinion Unaware (OU) 및 Distortion Unaware (DU) 비디오 품질 평가 모델을 개발했습니다. 이 모델은 S3D 비디오의 개별 뷰로부터 Cyclopean 프레임을 생성하고, 비겹치는 블록으로 분할한 후, Natural Scene Statistics (NSS)를 분석하여 기반 모델을 설계했습니다.

- **Performance Highlights**: 제안된 알고리즘의 성능은 IRCCYN, LFOVIAS3DPh1, LFOVIAS3DPh2 등 여러 인기 있는 S3D 비디오 데이터셋에서 검증되었으며, 모든 데이터셋에서 일관된 성능을 보여 주었습니다. 또한, 2D 및 3D 이미지와 비디오 품질 평가 알고리즘과 비교하여 경쟁력 있는 성능을 나타내었습니다.



### Enhancing AI microscopy for foodborne bacterial classification via adversarial domain adaptation across optical and biological variability (https://arxiv.org/abs/2411.19514)
- **What's New**: 이 논문은 식품 안전과 품질을 보장하기 위한 식중독 균의 신속한 검출의 필요성을 강조합니다. 기존의 배양 기반 방법의 한계를 극복하기 위해, AI를 활용한 현미경 기술이 그 대안으로 제시됩니다. 특히, 적대적 도메인 적응(adversarial domain adaptation)과 이미지 증대(image augmentation)를 통해 박테리아 분류의 일반화 가능성을 높이고자 하는 새로운 접근법이 소개됩니다.

- **Technical Details**: 연구에서는 EfficientNetV2 아키텍처를 기반으로 하여 세균 분류 기술이 개발되었습니다. 도메인 적대적 신경망(DANN)과 다중 도메인 적대적 신경망(MDANN)을 활용하여, 적은 수의 라벨링된 샘플(1-5개)만으로도 변화하는 실험 조건에서도 잘 작동할 수 있도록 모델을 훈련했습니다. 다양한 조건에서 수집된 실험 데이터를 통해 모델의 일반화 성능이 평가되었으며, 실험 조건에 따라 테스트 데이터를 다르게 설정하여 효과를 검증하였습니다.

- **Performance Highlights**: DANN과 MDANN 모델은 목표 도메인 분류 정확도를 최대 54.45%, 43.44%, 31.67%까지 향상시켰고, 소스 도메인에서는 4.44% 이하의 최소 성능 저하를 기록했습니다. Grad-CAM 및 t-SNE 시각화를 통해 다양한 조건에서 도메인 불변 특징을 학습하는 모델의 능력도 확인되었습니다. 이러한 연구 결과는 자원이 제한된 환경에서도 신속하게 세균을 분류할 수 있는 확장 가능하고 적응 가능한 프레임워크를 제시합니다.



### An Approach Towards Learning K-means-friendly Deep Latent Representation (https://arxiv.org/abs/2411.19496)
- **What's New**: 본 논문은 클러스터링(Clustering) 문제에 대한 새로운 접근 방식을 제안합니다. 기존의 K-means와 같은 고전적인 알고리즘의 한계를 극복하기 위해, 클러스터링과 네트워크 파라미터를 연합하여 학습하는 방법을 소개하고 있습니다. 특히, 제안된 방법인 CenTering(CT) 손실 함수를 통해 클러스터링 친화적인 데이터 표현을 학습하는 동시에 K-means 기반 클러스터 중심을 찾아내는 과정이 포함됩니다.

- **Technical Details**: 기존의 K-means 알고리즘이 입력 공간을 고정시키는 반면, 본 연구에서는 데이터 배치마다 잠재 공간(latent space) 학습과 클러스터 중심 학습을 병행하는 대안적 방법을 제안합니다. 이 과정에서 제안된 CT 손실 함수는 잠재 공간을 클러스터링에 적합하도록 밀어주는 역할을 하며, 클러스터링 파라미터는 고전적인 K-means 목표를 최적화함으로써 학습됩니다. 이와 같은 방법을 통해, 네트워크 파라미터와 클러스터 파라미터는 번갈아 가며 업데이트됩니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 여러 벤치마크 데이터 세트에서 기존 방법들에 비해 향상된 Normalized Mutual Information(NMI) 및 정확도(ACC) 점수를 기록하는 것으로 나타났습니다. 이러한 결과는 제안된 방법이 클러스터링 성능을 효과적으로 개선할 수 있음을 보여줍니다. 이 연구는 클러스터링 문제 해결을 위한 새로운 방향을 제시하며, 향후 다양한 도메인에서의 적용 가능성을 드높이고 있습니다.



### FLARE: Towards Universal Dataset Purification against Backdoor Attacks (https://arxiv.org/abs/2411.19479)
Comments:
          13 pages

- **What's New**: 이번 논문에서는 Deep Neural Networks(DNNs)가 backdoor 공격에 취약하다는 사실을 강조하며, 데이터셋 정화(dataset purification)의 필요성을 제안합니다. 연구팀은 기존의 정화 방법들이 backdoor 트리거와 목표 레이블 사이의 연결이 benign한 특성보다 학습하기 더 간단하다는 암묵적인 가정에 의존하고 있다고 지적하였습니다. 그러나 이러한 가정은 모든 유형의 공격에서 항상 통용되지 않음을 보여줍니다, 특히 all-to-all(A2A) 및 untargeted(UT) 공격에서요.

- **Technical Details**: FLARE는 다양한 backdoor 공격에 대응하기 위한 보편적인 정화 방법으로, 모든 hidden layer에서 비정상적 활성화를 집계하여 클러스터링을 위한 표현을 구상합니다. FLARE는 첫 단계로 각 훈련 샘플의 포괄적인 잠재 표현을 구성하고, 두 번째 단계에서 클러스터 분석을 통해 오염된 샘플을 탐지합니다. 각 클러스터의 안정성을 평가하여 보다 안정적인 클러스터를 오염된 클러스터로 식별합니다.

- **Performance Highlights**: FLARE는 22개의 대표적인 backdoor 공격(A2O, A2A, UT 공격 등)에 대해 그 효과성을 검증하였으며, 잠재적인 적응형 공격에 대해서도 견고하게 대응할 수 있는 능력을 보였습니다. 광범위한 벤치마크 데이터셋 평가를 통해 FLARE의 우수한 성능이 입증되었습니다.



### Blurred LiDAR for Sharper 3D: Robust Handheld 3D Scanning with Diffuse LiDAR and RGB (https://arxiv.org/abs/2411.19474)
- **What's New**: 이 논문에서는 기존의 RGB 기반 3D 재구성이 저조도, 저질감 및 저반사율 환경에서 성능 저하에 직면하는 문제를 해결하기 위해, 새로운 방식인 '블러(LiDAR)'를 활용한 깊이 정보 캡쳐 방법을 제안합니다. 이 방법은 디퓨즈(분산) 플래시를 방출하여 더 넓은 장면 범위를 포괄하여 성능을 개선하며, RGB 정보와 결합하여 깊이 및 색상 추정을 더욱 정확하게 수행할 수 있도록 합니다.

- **Technical Details**: 제안된 방식은 Gaussian surfel 기반 렌더링 프레임워크와 장면 적응적 손실 함수를 활용하여 RGB와 디퓨즈 LiDAR 신호 간의 균형을 동적으로 조정합니다. 디퓨즈 LiDAR는 본래의 포인트 깊이를 측정하는 것과 달리, 공간적으로 흐릿한 깊이 정보를 제공하지만, 이를 통해 더 나은 장면 커버리지와 재구성을 가능하게 합니다. 또한, 이들은 낮은 신호 대 잡음비(SNR) 환경에서도 유용하게 동작합니다.

- **Performance Highlights**: 본 연구를 통해 디퓨즈 LiDAR가 전통적인 희소 LiDAR보다 성능이 우수하다는 것을 실험적으로 증명하였으며, 근본적으로 로봇 및 모바일 스캐닝에 있어 도전적인 환경에서도 정확한 3D 스캐닝을 가능하게 합니다. 특히, 실제 환경에서의 질적 실험과 정량적 평가를 통해 이 기술의 효과성을 입증하였습니다.



### MCUCoder: Adaptive Bitrate Learned Video Compression for IoT Devices (https://arxiv.org/abs/2411.19442)
- **What's New**: 이번 연구에서는 자원 제한이 있는 IoT 환경에 최적화된 오픈소스 비디오 압축 모델인 MCUCoder를 소개합니다. MCUCoder는 초경량 인코더를 사용하여 10.5K 매개변수와 약 350KB의 메모리 크기를 가지며, 이러한 제한된 하드웨어에서도 원활한 비디오 전송을 할 수 있도록 설계되었습니다. 커맨드 간단한 비트 스트림을 생성하고 네트워크 상황에 따라 비트 전송률을 동적으로 조정하는 기능을 제공합니다.

- **Technical Details**: MCUCoder는 비대칭(asymmetric) 압축 모델인 인코더-디코더 구조를 가지고 있으며, 정보의 중요도에 따라 채널을 정렬하여 데이터를 저장하는 확률적 드롭아웃(stochastic dropout) 훈련 방법을 사용합니다. 이 모델은 기존의 복잡한 인코딩 작업 없이 실제 인퍼런스 단계에서 정보를 수집할 수 있도록 설계되었습니다. 또한, MCUCoder는 INT8 quantization 기술을 적용하여 DSP 및 CMSIS-NN 가속기를 사용한 처리 성능을 추가적으로 향상시킵니다.

- **Performance Highlights**: MCUCoder는 기존의 M-JPEG와 같은 비디오 압축 형식에 비해 55.65%의 비트 전송률 절감을 이루어내며, 이를 통해 저전력 MCU에서도 높은 에너지 효율성을 달성합니다. 또한, MCUCoder는 실시간 애플리케이션에서 끊김 없는 비디오 재생을 보장하기 위해 적응형 비트 전송률 스트리밍 기능을 지원합니다. 이로 인해 IoT 디바이스에서의 비디오 전송 품질이 유지됩니다.



### 3D Wasserstein generative adversarial network with dense U-Net based discriminator for preclinical fMRI denoising (https://arxiv.org/abs/2411.19345)
- **What's New**: 이 논문에서는 3D Wasserstein GAN 구조를 기반으로 하는 새로운 fMRI 데이터 디노이징 알고리즘인 3D U-WGAN을 제안합니다. 이 방법은 심층 U-Net 기반 판별기를 도입하여 전임상 fMRI 데이터의 노이즈를 효과적으로 제거하는 데 중점을 둡니다. 기존의 방법에 비해 이미지 품질이 크게 향상되며, 신호 대 잡음 비율이 개선됩니다.

- **Technical Details**: 제안된 3D U-WGAN 모델은 생성자(generator)와 밀집형 U-Net 판별기(dense U-Net discriminator)를 포함하고 있습니다. 생성자는 노이즈가 포함된 fMRI 데이터를 입력 받아 실제 노이즈가 없는 fMRI 데이터와 유사한 추정치를 생성하는 역할을 합니다. 판별기는 두 이미지 셋을 비교하여 진짜와 추정치를 구분하는 작업을 수행합니다.

- **Performance Highlights**: 실험 결과, 3D U-WGAN은 기능적 및 작업 기반의 전임상 fMRI 데이터에서 이미지 품질을 크게 개선하고, 신호 대 잡음 비율을 증가시키며, 기존 알고리즘들보다 우수한 성능을 보여주었습니다. 이 방법은 전임상 rsfMRI 및 작업 fMRI 데이터에 효과적으로 적용되며, 다양한 노이즈 패턴을 처리할 수 있는 유연성을 가지고 있습니다.



### Towards a Mechanistic Explanation of Diffusion Model Generalization (https://arxiv.org/abs/2411.19339)
Comments:
          13 pages, 15 figures. Accepted to NeurIPS 2024 Workshop on Attributing Model Behavior at Scale

- **What's New**: 이번 연구에서는 로컬 노이징(denoising) 연산에 기반한 확산 모델(diffusion model)의 일반화 메커니즘을 제안합니다. 네트워크 및 경험적 노이저(denoiser)의 분석을 통해, 확산 모델에서 로컬 유도 바이어스(local inductive biases)를 식별하였으며, 이를 통해 최적의 노이저를 근사하는 방법을 보여줍니다. 또한 패치 기반의 경험적 노이저를 활용하여 확산 프로세스에서 일반화 동작을 재현할 수 있음을 입증합니다.

- **Technical Details**: 확산 모델은 데이터 분포에 가우시안 노이즈를 점진적으로 추가하는 확산 프로세스를 기반으로 하며, 이 과정에서 생성된 최적의 노이징 균형을 찾는 것이 핵심입니다. 본 연구는 네트워크 노이저의 근사 오류를 분석하여 실질적으로 유사한 오류를 보이는 여러 노이저의 특성을 발견했습니다. 주요 초점은 중간 확산 과정에서의 오류 분석을 통해 공통된 로컬 유도 바이어스를 찾아내는 것입니다.

- **Performance Highlights**: 패치 기반 노이저의 결과가 전체 노이저의 결과와 동등하게 나타나는 것을 보여주며, 네트워크 노이저가 일반화되는 설정에서도 패치 기반 노이저가 잘 근사할 수 있음을 입증합니다. 최종적으로, 패치 기반 노이저들을 조합함으로써 네트워크 노이저의 출력을 근사할 수 있다는 강력한 증거를 제공합니다. 이는 확산 모델의 일반화 메커니즘을 강력하게 뒷받침하는 결과로 해석될 수 있습니다.



### Generalized Gaussian Model for Learned Image Compression (https://arxiv.org/abs/2411.19320)
Comments:
          13 pages, 12 figures

- **What's New**: 이 논문에서는 기본 Gaussian 모델을 일반화한 일반화 Gaussian 모델(Generalized Gaussian Model, GGM)을 제안하여 잠재 변수의 분포를 보다 유연하게 모델링합니다. 특히, GGM은 Gaussian 모델에 비해 하나의 추가 매개변수인 shape parameter(beta)만을 도입하여 복잡성을 줄입니다. 개선된 훈련 방법과 zero-center quantization을 도입하여 GGM의 성능을 향상시키고, 다양한 학습된 이미지 압축 방법에서 우수한 결과를 보여줍니다.

- **Technical Details**: GGM은 Gaussian 모델의 확장으로, mean, scale, shape 매개변수를 포함하여 잠재 변수의 분포를 효과적으로 모델링합니다. 특히, GGM은 β(베타)에 따라 scale 매개변수의 하한을 설정할 수 있는 기법과 gradient rectification 방법을 통해 훈련과 테스트 간의 불일치를 완화합니다. 이로 인해 압축 성능을 개선할 수 있으며, Gaussian 혼합 모델과 비교하여 낮은 복잡도로 뛰어난 성능을 제공합니다.

- **Performance Highlights**: 실험 결과, 제안된 GGM은 다양한 학습된 이미지 압축 방법에서 기존 Gaussian 및 Gaussian 혼합 모델(GMM)을 초월하는 성능을 보입니다. GGM-e 방법은 GMM보다 우수한 압축 성능을 제공하며, GGM-m 및 GGM-c 방법은 Gaussian 모델과 비교하여 동일한 네트워크 계산 복잡성을 유지하면서 더 나은 압축 성능을 달성합니다.



### GRAPE: Generalizing Robot Policy via Preference Alignmen (https://arxiv.org/abs/2411.19309)
Comments:
          Website: this https URL

- **What's New**: 이 논문에서는 일반화된 로봇 정책(Generalized Robot Policy)을 통해 행동 클로닝에 의존하지 않고도 다양한 작업을 처리할 수 있는 GRAPE 모델을 소개합니다. GRAPE는 성공적인 trial과 실패한 trial 모두에서 보상을 모델링하는 방법을 사용하여 다양한 작업에 대한 일반화 능력을 향상시킵니다. 이를 통해 로봇이 복잡한 조작 작업을 독립적인 단계로 쪼개고, 커스터마이즈된 시공간 제약(spatiotemporal constraints)을 통해 선호 모델링을 자동으로 안내할 수 있습니다.

- **Technical Details**: GRAPE는 비전-언어-행동(VLA: vision-language-action) 모델의 경로 수준에서 정렬하여 작동합니다. 이 모델은 다양한 목표가 가능하도록 유연하게 커스터마이즈할 수 있는 제약을 제공하며, 안전(safety), 효율성(efficiency), 작업 성공(task success) 등을 포함한 다양한 기준에 맞춰 조정할 수 있습니다. 또한 GRAPE는 실제 및 시뮬레이션 환경에서 여러 작업을 평가하여 성능을 입증합니다.

- **Performance Highlights**: 실험 결과 GRAPE는 최신 VLA 모델의 성능을 크게 향상시켜, 기존의 작업(task)에 대한 성공률을 51.79% 증가시키고, 보지 못한 작업에 대해서는 60.36%의 성공률 증가를 보여줍니다. 추가로, GRAPE는 안전성과 효율성 등의 다양한 목표에 맞춰 조정 가능하며, 충돌율(collison rate)을 44.31% 감소시키고 rollout 단계 길이(step-length)를 11.15% 단축시켰습니다.



### Skeleton Detection Using Dual Radars with Integration of Dual-View CNN Models and mmPos (https://arxiv.org/abs/2411.19251)
Comments:
          This paper was presented at the 16th International Conference on Advanced Applied Informatics (IIAI AAI 2024)

- **What's New**: 본 연구는 mmWave 레이더를 활용한 포인트 클라우드(point cloud) 데이터의 통합을 통해 노인의 움직임을 효과적으로 추적하고 검출하는 새로운 접근 방식을 제안합니다. 기존의 이미지 처리 방법과 달리, 본 연구는 프라이버시 보호를 위해 비침습적(non-invasive)인 방법을 강조합니다. 특히, PointNet과 mmPose의 융합을 통해 노인의 낙상(fall) 사건을 실시간으로 탐지하는 데 기여하고자 합니다.

- **Technical Details**: 포인트 클라우드 데이터의 특성을 고려하여, 본 연구는 회전 불변성(rotation invariance), 변환 불변성(translation invariance), 그리고 지역성(locality)을 포함하는 세 가지 중요한 요소를 다룹니다. PointNet은 회전 및 변환 불변성을 관리하고, mmPose는 지역성 문제를 해결합니다. 또한, 두 개의 레이더에서 수집된 데이터를 통합하여 뼈대 검출의 정확성을 높이는 방식으로, 좌표(coordinates), 속도(velocity), 신호 대 잡음 비율(signal-to-noise ratio, SNR) 등 다양한 특징을 활용하여 학습 모델에 입력합니다.

- **Performance Highlights**: 제안된 Dual View CNN 모델은 PointNet과 mmPose의 결합으로 이루어져 있으며, 두 개의 mmWave 레이더를 사용하여 Mean Absolute Error (MAE) 측면에서 성능을 비교합니다. 무작위로 걷는 행위에 대해선 최적의 결과를 보이지 않지만, 팔을 흔드는 동작(harm swing case)에서는 뛰어난 성능을 나타냅니다. 이는 특정 움직임에 대한 높은 정확성을 강조하며, 추후 연구에 중요한 시사점을 제공합니다.



### Differentiable Voxel-based X-ray Rendering Improves Sparse-View 3D CBCT Reconstruction (https://arxiv.org/abs/2411.19224)
- **What's New**: DiffVox는 물리 기반의 미분 가능한 X-선 렌더링(physics-based differentiable X-ray rendering)을 사용하여 voxelgrid 표현을 직접 최적화함으로써 Cone-Beam Computed Tomography(CBCT) 재구성을 위한 자가 감독(self-supervised) 프레임워크입니다. 이 연구는 X-선 이미지 형성 모델의 다양한 구현이 3D 재구성 품질과 새로운 뷰 합성에 미치는 영향을 분석하였습니다. 특히, 정확한 Beer-Lambert 법칙의 구현을 통해 기존 CBCT 재구성 알고리즘보다 우수성을 보였습니다.

- **Technical Details**: 본 연구에서는 2D X-선 투영 이미지로부터 내부 3D 구조를 추정하기 위해, 자가 감독 복원 프레임워크인 DiffVox를 사용하여 voxelgrid 표현을 직접 최적화합니다. 일반적인 CBCT 재구성 기법들이 제한된 수의 X-선만으로 큰 성능 저하를 보일 때, DiffVox는 물리 기반 미분 가능한 X-선 렌더링을 통해 높은 품질의 재구성을 가능하게 합니다. Siddon의 방법과 삼선형 보간(trilinear interpolation) 두 가지 방법을 사용하여 Beer-Lambert 법칙을 근사합니다.

- **Performance Highlights**: DiffVox는 42명의 피험자로부터 150,000개 이상의 원시 X-선 이미지로 구축된 공개 데이터셋에서 평가되었으며, 이는 기존 CBCT 재구성 알고리즘보다 더욱 우수한 성능을 보여주었습니다. 특히 적은 수의 X-선으로도 고충실도의 3D CBCT 볼륨을 재구성할 수 있어, 방사선 노출을 줄이는 잠재력을 가지고 있습니다. 이러한 성과는 향후 진단 유용성 향상에도 기여할 것입니다.



### Lost & Found: Updating Dynamic 3D Scene Graphs from Egocentric Observations (https://arxiv.org/abs/2411.19162)
Comments:
          Webpage: this https URL

- **What's New**: 최근 연구에서 정적인 재구성을 통한 세분화가 성공적으로 이루어졌지만, 이것은 동적인 환경을 반영하지 못하는 한계를 지녔습니다. 'Lost & Found'라는 접근법은 이러한 한계를 해결하며, 카메라의 위치와 손의 위치 추정 정보를 기반으로 동적인 객체를 추적하는 방법을 제시합니다. 이 방법은 온라인으로 변환 가능한 장면 그래프를 사용하여 객체 간의 관계를 묘사함으로써, 더 나아가 로봇 응용에 활용될 수 있는 정보를 제공합니다.

- **Technical Details**: 우리는 사용자 상호작용을 추적하기 위해 관찰 장비로 Aria 안경을 사용합니다. 이 안경으로부터 얻은 데이터를 통해, 손의 위치와 함께 6DoF(6 Degrees of Freedom) 자세를 정확하게 추적합니다. 동적인 장면을 구성하는 객체와 가구의 상호작용을 시공간적으로 재구성하여 로봇이 접근할 수 있는 동적인 3D 장면 그래프를 생성합니다.

- **Performance Highlights**: 해당 방법은 세분화 오류에서 최첨단 객체 자세 추적기보다 34%에서 56% 향상된 성능을 보이며, 매끄러운 6DoF 객체 궤적을 생성합니다. 또한, 로봇이 인식한 상호작용 정보를 사용하여 드로어 안의 객체를 회수하는 등의 응용 사례를 통해, 동적 장면 그래프가 로봇 작업에 어떻게 활용될 수 있는지 보여줍니다.



### Bayesian Deconvolution of Astronomical Images with Diffusion Models: Quantifying Prior-Driven Features in Reconstructions (https://arxiv.org/abs/2411.19158)
Comments:
          5+5 pages, 16 figures, Machine Learning and the Physical Sciences Workshop, NeurIPS 2024

- **What's New**: 이번 연구에서는 천체 관측에서의 이미지 복원에 있어 diffusion models(DMs)과 diffusion posterior sampling(DPS) 알고리즘을 도입하고 있습니다. 이 방법은 고해상도 우주론적 시뮬레이션을 기반으로 한 학습을 통해 관측된 데이터를 바탕으로 후분포(posterior distribution)를 계산하는 데 중점을 두고 있습니다. 특히, 연구진은 Hyper Supreme Camera(HSC) 데이터를 테스트하여 Hubble Space Telescope(HST) 이미지와 유사한 해상도를 달성할 수 있었음을 보고하였습니다.

- **Technical Details**: 연구에서는 variance-preserving stochastic differential equation(VP-SDE) 접근법을 따릅니다. Denoising Diffusion Probabilistic Model(DDPM) 프레임워크를 통해 원본 데이터에서 노이즈 버전으로의 전이 과정과 이 역전이를 해결하는 과정으로 구성됩니다. 이때 Bayes’ rule을 사용하여 posterior term을 도입하고, score를 이용하여 후분포를 추정하는 방식입니다.

- **Performance Highlights**: 실험 결과, 연구진의 방법이 이미지 품질, 계산 효율성, 노이즈에 대한 강인성 등에서 전통적인 방법에 비해 개선된 성능을 보였음을 발견했습니다. 그러나 물리적 일관성을 유지하는 것과 실시간 처리 속도를 달성하는 데 몇 가지 어려움이 있었고, 이는 향후 연구에서 해결해야 할 과제로 남아 있습니다. 이 시스템의 구현은 GitHub에서 공개되어 있으며, 실험에 사용된 TNG100 시뮬레이션 데이터와 HSC 데이터도 포함되어 있습니다.



### Visual SLAMMOT Considering Multiple Motion Models (https://arxiv.org/abs/2411.19134)
- **What's New**: 본 논문에서는 SLAM(동시 위치 인식 및 지도 작성)과 MOT(다중 객체 추적)를 통합하여 실시간 데이터 처리의 필요성을 강조합니다. 전통적인 방식에서는 이 두 작업이 독립적으로 실행되어 서로의 성능을 제한하는 경우가 많았으나, 우리는 이를 결합한 새로운 접근 방식을 제안합니다. 이 논문에서는 LiDAR 기반 시스템에서 다중 운동 모델을 고려한 SLAMMOT의 가능성을 논의하며, 이를 비주얼 SLAMMOT로 확장하는 방법을 제안합니다.

- **Technical Details**: 비주얼 SLAMMOT는 센서의 데이터 수집, 전방 오도메트리, 후방 최적화 및 맵 작성 등을 포함한 여러 모듈로 구성됩니다. 논문에서는 기존의 SLAMMOT 방식을 기반으로 여러 운동 모델을 통합하여 SLAMMOT의 효율성을 개선하는 새로운 방법론을 소개합니다. 이 접근법은 LiDAR와 비전 기반 센서 간의 간극을 메우는 것을 목표로 하며, 복잡한 환경에서도 정확한 객체 추적 및 위치 추정을 가능하게 합니다.

- **Performance Highlights**: 실험 결과는 제안된 비주얼 SLAMMOT가 기존 SLAM 및 MOT 방법보다 더 나은 성능을 보여주며, 동적인 환경에서의 경쟁력을 입증했습니다. 또한, 여러 운동 모델을 고려함으로써 실제 주행 환경에서 객체 상태 추정의 정확도를 향상시킬 수 있음을 보여줍니다. 이 연구는 자율 주행 차량 시스템의 실시간 작동 능력을 크게 향상시킬 것으로 기대됩니다.



### Enhancing Neural Network Robustness Against Fault Injection Through Non-linear Weight Transformations (https://arxiv.org/abs/2411.19027)
Comments:
          5 pages, 6 figures

- **What's New**: 이번 연구는 DNN(Deep Neural Networks)에서 고장으로 인한 성능 저하 문제를 해결하기 위해 새로운 접근법을 제시합니다. 기존 연구에서는 활성화 함수의 범위를 제한하는 방법을 사용했다면, 본 연구는 DNN의 가중치(weight)를 제약하는 포화 활성화 함수(Saturated Activation Functions, SAF)를 사용하여 과도한 가중치 증가를 방지하고 있습니다. 이러한 SAF 적용 방법은 DNN의 내구성을 향상시킬 뿐만 아니라 성능도 소폭 개선하는 결과를 보여줍니다.

- **Technical Details**: 이 연구에서는 Tanh, Arctan 등 여러 SAF를 활용하여 DNN의 가중치를 조정합니다. 훈련 과정에서 SAF가 적용된 가중치로 네트워크를 학습하고, 실제 환경에서 고장이 발생할 수 있는 매체에 SAF가 적용되지 않은 가중치를 기록합니다. 추후 추론 시점에서, 고장을 포함한 가중치에 SAF를 적용하여 오류를 억제하는 방식입니다.

- **Performance Highlights**: CIFAR10, CIFAR100, ImageNet 2012 데이터셋에서 제안하는 방법을 검증하였으며, FP32 ResNet18 모델이 비트 오류율(bit-error rate, BER) 0.00001에서도 경미한 정확도 손실로 작동할 수 있음을 보여줍니다. SAF를 적용하지 않은 경우, 모델은 임의의 출력을 생성하는 반면, 본 방법을 사용하면 더욱 안정적인 결과를 도출합니다.



### Harden Deep Neural Networks Against Fault Injections Through Weight Scaling (https://arxiv.org/abs/2411.18993)
Comments:
          6 pages, 8 figures

- **What's New**: 최근의 연구에서 딥 뉴럴 네트워크(DNN)가 이미지 인식, 객체 감지 및 자연어 처리와 같은 다양한 스마트 응용 프로그램을 가능하게 하고 있습니다. 그러나 DNN이 하드웨어 장치에 배포되면 노화, 온도 변화 및 쓰기 오류와 같은 원치 않는 결함에 취약해집니다. 본 논문에서는 DNN 가중치를 상수로 곱한 다음 결함에 취약한 매체에 저장함으로써 무결성을 강화하는 간단하면서도 효과적인 방법을 제안합니다.

- **Technical Details**: 우리의 방법은 가중치를 저장하기 전에 원소별로 곱하고, 읽을 때는 원소별로 나누는 과정으로 구성되어 있습니다. 이러한 과정은 비트 플립으로 인해 발생하는 전체 절대 오류를 줄이는 데 효과적입니다. 우리는 32비트 부동소수점, 16비트 부동소수점 및 8비트 고정소수점의 세 가지 데이터 타입에 걸쳐 실험을 수행하였고, 이 방법이 8비트 고정소수점 ResNet50의 Top-1 정확도를 54.418만큼 개선함을 보였습니다.

- **Performance Highlights**: 제안된 방법은 이전의 오류 교정 코드 방법들과 대비하여 메모리 및 계산 오버헤드가 현저히 낮습니다. 우리의 방법은 단순한 원소별 곱셈 및 나눗셈만을 요구하며, 이는 DNN을 비트 플립으로부터 보호하는 데 중요한 기여를 합니다. 여러 모델과 데이터 타입에서의 성능 향상을 통해 이 방법의 일반화를 입증했습니다.



### FAN-Unet: Enhancing Unet with vision Fourier Analysis Block for Biomedical Image Segmentation (https://arxiv.org/abs/2411.18975)
Comments:
          arXiv admin note: text overlap with arXiv:2410.02523

- **What's New**: 본 논문에서는 FAN-UNet이라는 새로운 아키텍처를 제안합니다. 이 모델은 Fourier Analysis Network (FAN) 기반의 비전 백본과 U-Net 구조를 결합하여 의료 이미지 세분화에서의 장기 의존성과 주기성 모델링 문제를 효과적으로 해결합니다. Vision-FAN 레이어는 FAN 레이어와 자기 주의 메커니즘을 통합하여 장기 의존성과 주기적 관계를 잘 포착할 수 있도록 합니다.

- **Technical Details**: FAN-UNet의 핵심 모듈인 Vision-FAN Block은 FANLayer2D와 주의 메커니즘으로 구성되어 있습니다. 이 모듈은 2D 특성을 처리하며, 이미지 내 전역적 장기 의존성과 주기적 관계를 모델링합니다. position encoding 및 window-based self-attention을 통해 입력 특성 맵의 상대적인 관계를 이해하고 전역적 공간 의존성을 캡처합니다.

- **Performance Highlights**: 다양한 의료 이미징 데이터셋에 대한 실험을 통해 FAN-UNet은 모델 복잡성과 성능 간의 균형을 잘 유지하며, 높은 정확도(96.07%), mIoU(78.83%) 및 DSC(88.16%)를 기록했습니다. 이러한 성과는 모델의 효과성과 실용성을 뒷받침합니다.



### FiRe: Fixed-points of Restoration Priors for Solving Inverse Problems (https://arxiv.org/abs/2411.18970)
- **What's New**: 이 논문에서는 이미지 복원(composition restoration)에서 전통적인 denoising 모델을 넘어서서 새로운 priors인 Fixed-points of Restoration (FiRe) priors를 도입합니다. FiRe의 핵심 통찰력은 자연 이미지가 손상 연산자(degradation operator)와 복원 모델(restoration model)의 조합의 고정점(fixed point)으로 발생한다는 점입니다. 이러한 관점을 통해 다양한 복원 네트워크가 역문제를 해결하기 위한 priors로 효과적으로 활용될 수 있음을 보여줍니다.

- **Technical Details**: 이 논문에서 제안된 FiRe 프레임워크는 각 복원 모델이 훈련된 손상(degradation)과 결합하여 사용될 수 있는 가능성을 제시합니다. 구체적으로, FiRe는 일반 복원 모델과 그에 맞는 손상 단계만으로도 효과적인 추정치를 얻을 수 있는 방법으로 정의됩니다. 또한, 일반 복원 모델에 대한 정형화된 prior의 폐쇄형 표현을 유도하고, 이는 Plug-and-Play (PnP) 프레임워크와 이론적으로 연결됩니다.

- **Performance Highlights**: 실험 결과, FiRe 프레임워크는 다양한 역문제를 해결하는 과정에서 재구성 품질(resolution quality)을 획기적으로 개선함을 입증했습니다. 이전에 훈련된 복원 모델을 PnP 알고리즘으로 통합하는 새로운 패러다임을 제시하여, 다양한 상황에서 효과적인 성능을 보여주고 있습니다. 이러한 접근 방식은 복원 과정의 조건을 보다 정교하게 조정할 수 있도록 해 주어, 기존 방법들과의 비교에서 두드러진 개선을 나타냅니다.



### Deep Plug-and-Play HIO Approach for Phase Retrieva (https://arxiv.org/abs/2411.18967)
- **What's New**: 이번 연구에서는 비선형적이고 잘 정립되지 않은 특성을 가진 phase retrieval 문제를 해결하는 새로운 방법론을 제시합니다. 이러한 새로운 접근 방식은 학습 기반(Learning-based) 선행 지식을 활용하여 이미지 복원 성능을 크게 향상시킬 수 있습니다. 특히, plug-and-play 방식으로 하이브리드 입력-출력 방법(HIO)와의 통합을 통해 기존의 방법들과 비교하여 우수한 성능을 입증했습니다.

- **Technical Details**: 이 논문에서는 half-quadratic splitting을 기반으로 한 분석적인 업데이트 단계의 유도과정을 포함하는 이 방법의 수학적 발전을 소개합니다. 크게 두 가지 구성 요소인 학습 기반 선행 지식과 효율적인 업데이트 단계가 결합되어 문제를 해결하는 데 기여합니다. 이 방법은 extensive simulations를 통해 대규모 테스트 데이터셋에서 평가되었습니다.

- **Performance Highlights**: 결과적으로, 이 방법은 이미지 품질(image quality), 계산 효율성(computational efficiency), 초기화 및 잡음에 대한 강건성(robustness) 등에서 효과적임을 보여주었습니다. 특히, 기존의 솔루션보다 더 나은 성능을 제공하며 phase retrieval 분야에서의 가능성을 확장합니다.



### ScratchEval: Are GPT-4o Smarter than My Child? Evaluating Large Multimodal Models with Visual Programming Challenges (https://arxiv.org/abs/2411.18932)
- **What's New**: 이 연구에서는 ScratchEval이라는 새로운 벤치마크를 제안하여 대규모 다중모달 모델(large multimodal models, LMMs)의 시각적 프로그래밍 추론 능력을 평가하고자 합니다. 기존의 평가 방법들은 특정 시나리오에 국한되어 있었지만, ScratchEval은 Scratch라는 블록 기반 시각적 프로그래밍 언어를 기반으로 하여 모델이 시각 정보를 처리할 수 있도록 설계되었습니다. 이를 통해 LMMs의 명령 이해 능력을 포괄적으로 평가할 수 있는 기회를 제공합니다.

- **Technical Details**: ScratchEval은 Scratch 스크립트가 포함된 305개의 다지선다 질문으로 구성되어 있으며, 각 질문은 문제 설명과 선택지를 포함합니다. 이 벤치마크는 일관된 논리적 사고 및 문제 해결 능력에 초점을 맞추며, 모델이 이미지, 그래픽 프로그래밍 언어 및 내재된 논리를 함께 이해해야 합니다. 질문은 수학, 논리적 사고, 그래픽 인식 및 공간 인식으로 분류되며, 다양한 인지 영역에서 모델의 능력을 평가합니다.

- **Performance Highlights**: 총 10개의 LMM을 평가한 결과, Gemini-1.5-Pro가 모든 카테고리에서 가장 높은 점수를 기록했지만, 대부분의 모델은 50% 정확도를 넘기기 어려웠습니다. 이는 LMM들이 비주얼 코드 추론 능력에서 한계를 지니고 있음을 시사합니다. 일반적으로, 수학 및 논리적 추론 작업에서 모델의 성능이 낮았고, 반면 그래픽 및 공간 인식 작업에선 상대적으로 더 나은 성과를 보였습니다. 이 연구는 적절한 프롬프트 기법이 LMM의 성능 향상에 기여할 수 있음을 보여주지만, 다중모달 LLM에 대한 연구는 더 필요합니다.



### CovHuSeg: An Enhanced Approach for Kidney Pathology Segmentation (https://arxiv.org/abs/2411.18893)
Comments:
          Under review

- **What's New**: 이 논문에서는 CovHuSeg 알고리즘을 제안하여 신장 사구체(segmentation of kidney glomeruli)의 세분화 문제를 해결합니다. 기존의 모델들이 기하학적 특성, 즉 크기나 볼록도(convexity)를 완전히 포착하지 못하는 문제를 해결하기 위한 방법으로, CovHuSeg는 구형(anomalies) 이상에 적합하도록 설계된 후처리(post-processing) 방법입니다. 이 알고리즘은 세분화 마스크(mask)가 구멍이 없고 비정상적인 형태를 갖지 않도록 보장합니다.

- **Technical Details**: CovHuSeg 알고리즘은 볼록 껍질(convex hull)을 기준으로 세분화 과정을 개선하여 결과적으로 더욱 정확한 세분화 결과를 생성합니다. 이 방법은 특히 신장 병리 이미지에서 깊이 학습 모델(예: UNet, UNet++, UNet3+)을 사용하여 실험하였고, 한정된 데이터 상황에서도 뛰어난 성능을 입증했습니다. 볼록 껍질은 또한 물체의 경계를 정의하는데 필요한 정보로 활용되며, 이는 로봇 공학 및 영상 처리에서 중요한 역할을 합니다.

- **Performance Highlights**: 모든 실험에서 CovHuSeg 후처리를 적용한 모델들은 세분화 정밀도가 향상되는 결과를 보여 주었습니다. 특히 데이터가 제한적인 상황에서도 모든 메트릭에서 모델들의 성능이 증대되었습니다. 이러한 놀라운 결과는 CovHuSeg 알고리즘이 신장 사구체의 세분화 문제를 효과적으로 해결하고 있음을 강하게 시사합니다.



### ETSM: Automating Dissection Trajectory Suggestion and Confidence Map-Based Safety Margin Prediction for Robot-assisted Endoscopic Submucosal Dissection (https://arxiv.org/abs/2411.18884)
- **What's New**: 본 연구에서는 Robot-assisted Endoscopic Submucosal Dissection (ESD)의 수술 절차를 개선하기 위한 새로운 접근 방식을 제안합니다. 특히 ESD 절개 경로를 예측하는 데 있어 로봇 시스템을 활용하고, 이와 함께 Confidence Map을 기반으로 한 Safety Margin을 사용하는 프레임워크를 도입했습니다. 이는 수술 효율성과 정확성을 높이고, 의사 결정을 도와줄 수 있는 도구로서 기능합니다.

- **Technical Details**: 제안된 방법의 핵심은 ESD Trajectory and Confidence Map-based Safety Margin (ETSM) 데이터셋을 활용하여, Dissection Trajectory를 예측하는 것과 동시에 Dissection 영역에 대한 Confidence Map을 생성하는 것입니다. 이를 위해 Regression 기반 Confidence Map Prediction Network (RCMNet)를 개발하여, 다양한 Safety Margin 수준을 예측하며, Imitation Learning, LSTM 모델, Motion Indeterminacy Diffusion 등의 기존 기법과의 차별성을 두었습니다.

- **Performance Highlights**: 실험 결과, RCMNet은 Confidence Map 기반 Safety Margin 예측 작업에서 평균 절대 오차 (MAE) 3.18을 달성하며 우수한 성능을 보였습니다. 이는 기존 연구에서는 다루지 않았던 회귀 기반 접근 방식을 통해 이루어진 성과이며, 이로 인해 ESD 절차의 안전성을 더욱 높이고, 클리닉에서의 활용 가능성을 크게 향상시켰습니다.



### Multi-Task Learning for Integrated Automated Contouring and Voxel-Based Dose Prediction in Radiotherapy (https://arxiv.org/abs/2411.18767)
- **What's New**: 이번 연구는 멀티 태스크 러닝(multi-task learning, MTL) 접근 방식을 도입하여 자동 윤곽 추출(contouring) 및 용적 기반 선량 예측(voxel-based dose prediction) 작업을 통합했습니다. 기존의 자동화된 방사선 치료 계획 프로세스에서 이 두 작업은 별개로 수행되었으나, MTL을 활용하여 각각의 작업 간의 공통 정보를 최대한 활용함으로써 효율성을 대폭 향상시켰습니다. 연구에서 사용된 데이터셋은 내부 전립선암 데이터셋과 공개적으로 이용 가능한 두경부암 데이터셋(OpenKBP)으로 구성되었습니다.

- **Technical Details**: 두 가지 암 치료 부위에 대한 데이터셋을 사용하여 연구를 진행했습니다. 전립선암 데이터셋은 110명의 환자에게서 수집되었고, 두경부암 데이터셋은 OpenKBP 챌린지 데이터셋에서 수집한 328명의 환자로 구성되었습니다. 이 데이터는 CT 이미지, 방사선 치료 계획으로 생성된 볼륨 선량 분포 및 관련 영역(region of interest, ROI)의 윤곽 정보를 포함했습니다. 연구를 통해 전통적인 순차 모델과 비교하여 MTL의 효능을 검증했습니다.

- **Performance Highlights**: MTL을 기반으로 한 자동 윤곽 추출 및 선량 예측 모델은 순차적으로 처리된 모델들보다 향상된 성능을 나타냈습니다. 전립선 및 두경부 부위에서 평균 절대 차이(dose volume histogram metrics)가 각각 19.82% 및 16.33% 개선되었습니다. DICE score는 전립선에서 0.824, 두경부에서 0.716에 도달하여 기존의 자동 윤곽 추출 모델과 비교했을 때 유의미한 성과를 보여주었습니다. 이러한 개선은 자동화된 방사선 치료 계획의 정확성과 효율성을 높이는 데 기여할 것으로 기대됩니다.



### Multi-Task Model Merging via Adaptive Weight Disentanglemen (https://arxiv.org/abs/2411.18729)
- **What's New**: 이번 연구에서는 모델 머징(model merging) 분야의 기존 접근 방식을 개선하기 위해 Task Arithmetic 속성을 재검토하고 Task Consistency 속성을 제안하였습니다. Task Consistency 속성은 병합된 모델의 성능을 향상시키면서 작업 간 간섭(interference)을 최소화하려는 목적을 가지고 있습니다. 실험적으로 이 속성을 충족하는 직교(task vectors) 작업 벡터를 찾는 과정이 성능 향상에 기여함을 보여주었습니다.

- **Technical Details**: Adaptive Weight Disentanglement(AWD)라는 새로운 방법론을 도입하여 전통적인 작업 벡터를 중복 벡터와 여러 개의 분리된 작업 벡터로 분해합니다. AWD의 주요 최적화 목표는 분리된 작업 벡터 간의 직교성을 달성하고자 하며, 이는 효율적인 성능 전이를 가능하게 합니다. 이 과정에서 코사인 유사도를 최소화하고 중복 벡터의 크기를 줄이는 두 가지 최적화 목표를 설정하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 AWD가 기존 모델 머징 접근 방식에 비해 일관되게 우수한 성능을 나타냄을 확인했습니다. 특히, ViT-B/32 모델에서 평균 정확도가 2.8%, ViT-L/14 모델에서 1.5% 향상되었습니다. 또한, 이 방법은 언어 모델에서도 효과적으로 일반화되며, 다양한 조건에서 향상된 강건성을 보였습니다.



### Volume Rendering of Human Hand Anatomy (https://arxiv.org/abs/2411.18630)
Comments:
          10 pages

- **What's New**: 이 논문에서는 인체 손의 자기공명영상(MRI) 데이터셋을 위한 전송 함수(transfer function) 설계에 대해 연구합니다. 특히, 손의 근골격 기관을 중점적으로 다뤄 손의 주요 기능인 물체 조작을 지원하는 데 중요한 역할을 하는 조직들의 시각적 표현을 개선하는 방법을 제시합니다. 우리는 고품질의 볼륨 렌더링 이미지를 생성하며, 각 조직의 외관에 대한 세밀한 조작이 가능하다는 점을 강조합니다.

- **Technical Details**: 볼륨 렌더링 방법은 간접 볼륨 렌더링(IVR)과 직접 볼륨 렌더링(DVR)의 두 가지 범주로 나눌 수 있습니다. 본 연구에서는 볼륨 레이 캐스팅(volume ray casting) 기법을 사용하여 3D 스칼라 필드를 시각화하고, 각 3D 위치에서 빛의 상호작용을 설명하는 전송 함수의 사용을 강조합니다. 스칼라 값은 발광 색(emission color)과 불투명도(opacity)로 맵핑되어, 볼륨 데이터를 효율적으로 표현합니다.

- **Performance Highlights**: 실험 결과, 본 방법은 기존의 표면 및 볼륨 렌더링 기법과 비교할 때 손의 해부학적 시각화를 개선하는 데 성공했다고 보고됩니다. 또한, 손의 다양한 조직을 강조할 수 있는 두 가지 전송 함수를 제안하여, 시각적 맥락을 유지하면서도 해부학적 정보를 효과적으로 전달합니다. 본 연구는 손의 다양한 운동 시퀀스에서도 뛰어난 렌더링 품질을 보여주었습니다.



New uploads on arXiv(cs.AI)

### Handling irresolvable conflicts in the Semantic Web: an RDF-based conflict-tolerant version of the Deontic Traditional Schem (https://arxiv.org/abs/2411.19918)
- **What's New**: 이번 논문에서는 Deontic Traditional Scheme을 RDF와 SPARQL을 통해 구현하는 새로운 온톨로지를 제안합니다. 이 시스템은 서로 상충되는 의무나 금지, 허가를 명시하는 상황에서의 갈등을 처리하도록 설계되었습니다. 이는 일반적인 규범 이론 연구에서 중요한 진전을 나타내며, 기존의 접근 방식과는 다르게 다양한 갈등 유형을 모델링하고 이론적 대안을 제공합니다.

- **Technical Details**: 제안된 프레임워크는 RDF로 인코딩 되어 있으며, 이는 의미 웹(Semantic Web)의 기초가 되는 가장 널리 사용되는 지식 표현 언어입니다. 이 온톨로지는 주로 의무 개념에 국한되지 않고, 다양한 deontic modality의 상호작용을 모델링하고 논리적으로 추론하는 데 필요한 구조를 제공합니다. 데온틱 논리(deontic logic)는 규범적 추론을 모델링하기 위한 주요 도구로 사용되며, 논문의 주요 초점은 갈등의 정의 및 그 이론적 기초를 제시하는 것입니다.

- **Performance Highlights**: 이 연구는 법률 준수를 자동으로 검사하는 시스템의 개발을 통해 기존의 RDF 데이터 처리의 한계를 극복하는 것을 목표로 합니다. 법령 간의 갈등을 식별하고 업데이트 할 수 있는 LegalTech 애플리케이션 개발에 기여할 수 있는 가능성이 제시됩니다. 제안한 온톨로지는 고유한 구조를 가지고 있으며, 이를 통해 여러 갈등을 유연하게 다룰 수 있는 능력을 가지게 됩니다.



### PDDLFuse: A Tool for Generating Diverse Planning Domains (https://arxiv.org/abs/2411.19886)
Comments:
          218 Tables, 3 Figures, 4 Algorithms

- **What's New**: 이 논문에서는 자동 계획 시스템의 도메인 다양성을 증대시키기 위한 도구인 PDDLFuse를 제안합니다. 전통적인 방법과 달리, PDDLFuse는 기존 도메인을 단순히 번역하는 것이 아니라 새로운 도메인을 생성하여 Planning Domain Definition Language (PDDL)를 강화합니다. 도메인 랜덤화( Domain Randomization) 개념을 활용하여 새로운 계획 문제를 생성하고, 이를 통해 기존 계획 알고리즘의 범위를 확장할 수 있는 가능성을 보여줍니다.

- **Technical Details**: PDDLFuse는 도메인 생성기의 매개 변수를 조정하여 생성되는 도메인의 난이도를 조절할 수 있는 기능을 가지고 있습니다. 이는 기존 도메인 독립 계획자들이 복잡한 문제를 해결하는 데 어려움을 겪는 상황에서 매우 중요한 요소입니다. 논문에서는 또 다른 접근 방식을 도입하여 기존의 도메인을 융합하고, 더 다양하고 복잡한 계획 도메인을 효과적으로 생성함으로써 계획 연구의 발전에 기여하고자 합니다.

- **Performance Highlights**: 초기 테스트 결과, PDDLFuse는 전통적인 도메인 생성 방법에 비해 복잡하고 다양한 도메인을 효율적으로 생성할 수 있음을 나타냈습니다. 이는 새로운 계획자를 검증하고 기초 계획 모델을 테스트하는 데 도움을 주며, 탐색되지 않은 도메인에 대해서도 연구의 기회를 제공합니다. PDDLFuse의 능력은 기계 학습과 자동화된 계획 시스템의 강력한 진전을 의미하며, 계획 연구 분야에 중요한 기여를 하고 있습니다.



### Knowledge Management for Automobile Failure Analysis Using Graph RAG (https://arxiv.org/abs/2411.19539)
Comments:
          7 pages, 6 figures, to be published in 2024 IEEE International Conference on Bid Data (BigData)

- **What's New**: 이 연구는 자동차 고장 분석을 위한 지식 관리 시스템을 제안합니다. 이 시스템은 대규모 언어 모델(LLM)과 지식 그래프(KG)를 결합한 Retrieval-Augmented Generation (RAG) 방식에 기반하고 있습니다. 특히 그래프 RAG는 젊은 엔지니어들이 기존의 KG에서 효과적으로 정보를 추출하고 이해할 수 있도록 최적화되었습니다. 이를 통해 고장 원인을 신속하게 식별할 수 있는 시스템으로의 발전을 목표로 합니다.

- **Technical Details**: 이 논문은 자동차 고장 분석에 있어 기존의 지식 그래프와 함께 사용할 수 있는 새로운 그래프 RAG 시스템을 제안합니다. 이 시스템은 LLMs의 출력 성능을 향상시키기 위한 ROUGE F1 점수를 기반으로 한 실험을 통해 그 효과성을 입증하였습니다. 특히, 그래프 RAG는 KG에서 유용한 서브 그래프를 추출하고, 젊은 엔지니어들이 쉽게 이해할 수 있도록 하는 데 중점을 두고 있습니다.

- **Performance Highlights**: 제안된 방법에 의해 생성된 문장은 기존의 방법보다 평균 157.6% 향상된 ROUGE F1 점수를 기록했습니다. 이는 자동차 고장 분석을 위한 지식 관리에서 제안된 시스템의 효과성을 강조합니다. 이러한 성과는 고장 분석 전문 지식을 젊은 엔지니어에게 효과적으로 전달할 수 있는 가능성을 보여줍니다.



### A Local Information Aggregation based Multi-Agent Reinforcement Learning for Robot Swarm Dynamic Task Allocation (https://arxiv.org/abs/2411.19526)
- **What's New**: 이 논문에서는 동적 환경에서 로봇 스왐(wham)의 작업 할당(task allocation)을 최적화하는 방법을 탐구합니다. 로봇 협력을 위한 강력하고 유연하며 확장 가능한 전략을 수립하는 필요성을 강조하며, 분산 로봇 스왐 네트워크를 위해 설계된 새로운 프레임워크인 분산 부분 관찰 가능 마르코프 결정 과정(Dec_POMDP)을 도입합니다. 이 방법론의 핵심은 중앙 집중식 훈련과 분산 실행(CTDE)을 결합한 로컬 정보 집계 멀티 에이전트 결정적 정책 기울기(LIA_MADDPG) 알고리즘입니다.

- **Technical Details**: LIA_MADDPG 알고리즘에서 중앙 집중식 훈련 단계 동안, 로컬 정보 집계(LIA) 모듈이 이웃 로봇으로부터 중요한 데이터를 수집하여 의사 결정 효율성을 향상시킵니다. 분산 실행 단계에서는 환경 조건의 변화에 따라 작업 할당을 동적으로 조정하는 전략 개선 방법이 제안됩니다. 실험적 평가에서는 LIA 모듈이 CTDE 기반의 다양한 다중 에이전트 강화 학습(MARL) 방법에 통합 가능하고, 성능 향상에 크게 기여함을 보여줍니다.

- **Performance Highlights**: LIA_MADDPG는 여섯 가지 전통적인 강화 학습 알고리즘 및 휴리스틱 알고리즘과 비교하여 우수한 확장성(scalability), 환경 변화에의 빠른 적응성, 안정성과 수렴 속도를 효과적으로 유지하는 능력을 입증했습니다. 이러한 결과는 로봇 스왐에서 작업 할당을 개선하고 지역 협업 및 적응 전략 실행을 통한 LIA_MADDPG의 잠재력을 강조합니다.



### TQA-Bench: Evaluating LLMs for Multi-Table Question Answering with Scalable Context and Symbolic Extension (https://arxiv.org/abs/2411.19504)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)이 복잡한 멀티-테이블 관계 데이터에서 질문 응답(QA) 작업을 효율적으로 수행할 수 있는 능력을 평가하기 위해 TQA-Bench라는 새로운 멀티-테이블 QA 벤치를 제안합니다. 기존 벤치마크는 단일 테이블 QA에 주로 초점을 맞추어 LLM의 복잡한 멀티-테이블 QA 능력을 제대로 평가하지 못하고 있습니다. TQA-Bench는 실제 공개 데이터셋에서 수집된 다양한 관계형 데이터베이스 인스턴스를 포함하고 있으며, 8K에서 64K 토큰까지의 다양한 멀티-테이블 컨텍스트 길이를 생성하는 유연한 샘플링 메커니즘을 도입합니다.

- **Technical Details**: TQA-Bench는 데이터 수집, 관계형 데이터 샘플링, 평가 작업 정의 및 기호 확장을 통한 질문 생성의 네 가지 주요 단계로 체계적으로 구성됩니다. 이를 통해 여러 데이터 소스(예: WorldBank, DataGov)에서 다수의 대규모 관계형 데이터베이스를 활용해 다양한 멀티-테이블 QA 작업을 생성합니다. 또한, 8K에서 64K 토큰까지의 다양한 컨텍스트 길이를 갖춘 평가 작업을 제안하여 실제 데이터를 처리하는 LLM의 성능을 정밀하게 평가합니다.

- **Performance Highlights**: 논문에서 실시한 광범위한 실험을 통해 다양한 LLM의 멀티-테이블 QA 작업에서의 성능을 평가하였고, 그 결과 복잡한 데이터 환경에서의 LLM의 도전 과제와 기회를 강조하고 있습니다. TQA-Bench는 LLM의 데이터 검색 및 패턴 매칭을 넘어서는 추론 능력을 평가할 수 있도록 기호 확장 기능을 통합하여 신뢰성을 높였습니다. 이러한 평가 결과는 복잡한 데이터 관리 작업에서 LLM 기반 애플리케이션의 설계 및 실행에 대한 귀중한 통찰력을 제공합니다.



### Integrating Transit Signal Priority into Multi-Agent Reinforcement Learning based Traffic Signal Contro (https://arxiv.org/abs/2411.19359)
- **What's New**: 이번 연구는 다중 에이전트 강화 학습(MARL)을 기반으로 한 교통 신호 제어에 운송 신호 우선(TSP)을 통합합니다. 첫 번째 부분에서는 마이크로 시뮬레이션 환경에서 조정된 두 교차로에 대해 MARL 기반의 적응형 신호 제어를 개발했습니다. 중앙 집중식으로 훈련된 두 개의 에이전트는 전체 교차로 지연 시간 측면에서 조정된 작동 신호 제어와 비교하여 약간 더 나은 성능을 보여주었습니다.

- **Technical Details**: 이 연구는 VDN(Value Decomposition Network) 아키텍처를 사용하여 각 교차로에 대한 두 개의 에이전트를 중앙 집중식으로 훈련하는 방식을 채택했습니다. 두 번째 부분에서는 훈련된 신호 제어 에이전트를 배경 신호 제어기로 활용하고 이벤트 기반 TSP 에이전트를 개발했습니다. 독립적인 TSP 에이전트는 분산 훈련 및 분산 실행(DTDE) 프레임워크 아래에서 훈련되었고, 협력적 TSP 에이전트는 중앙 집중식 훈련 및 분산 실행(CTDE) 프레임워크에서 훈련되었습니다.

- **Performance Highlights**: 테스트 결과, 독립적인 TSP 에이전트는 TSP가 없는 경우에 비해 두 교차로에서 버스 지연 시간을 22% 줄였으며, 조정된 TSP 에이전트는 27% 지연 시간 감소를 달성했습니다. 두 경우 모두, 대부분의 측면 거리 움직임에 대해서는 지연 시간이 약간 증가했습니다. 독립 에이전트는 훈련 과정 내내 높은 변동성을 보여주었으나 최종적으로 두 에이전트는 동일한 버스 지연 값으로 수렴했습니다.



### OMuleT: Orchestrating Multiple Tools for Practicable Conversational Recommendation (https://arxiv.org/abs/2411.19352)
- **What's New**: 이 논문에서는 현실적인 대화형 추천 시스템(conversational recommender system, CRS)의 설계, 평가, 구현을 위한 체계적인 노력을 제시합니다. 사용자들이 자유형 텍스트를 입력하여 추천을 요청하고, 그에 따라 관련성 있고 다양한 항목을 받을 수 있도록 하는 것이 목적입니다. 기존 연구와 달리, 우리는 10개 이상의 도구를 갖춘 새로운 접근 방식을 제안하여 LLM(large language models)을 더 효과적으로 활용할 수 있습니다.

- **Technical Details**: 우리는 OMuleT(Orchestrating Multiple Tools)이라는 프레임워크를 제안하여 복잡한 요청을 처리하기 위해 다수의 도구를 LLM에 추가합니다. 이 방법은 사용자의 원시 발화를 정형화된 의도로 변환하고, 도구 실행 정책을 적용하여 추천 결과를 생성합니다. 이러한 접근은 시스템의 투명성과 제어성을 향상시키며, 성능에서도 기존 방법보다 우수한 결과를 보여줍니다.

- **Performance Highlights**: 모델은 두 개의 LLM(LLaMA-405B 및 GPT-4o)과 8가지 메트릭을 통해 광범위한 평가를 실시하였고, 그 결과 우리의 프레임워크를 사용하는 것이 기존 LLM보다 더 효과적이며 최상의 성능을 위해 다수의 도구가 필요함을 보여줍니다. 이러한 연구 결과는 실제 사용자 요청을 기반으로 한 CRS의 실용성을 강조하며, 내부 배포를 위한 설계와 배운 교훈을 공유합니다.



### Mars-PO: Multi-Agent Reasoning System Preference Optimization (https://arxiv.org/abs/2411.19039)
- **What's New**: 이번 논문에서는 Mars-PO라는 새로운 프레임워크를 제안하여 대형 언어 모델(LLMs)의 수학적 추론 능력을 향상시키고자 합니다. Mars-PO는 다중 에이전트 시스템(multi-agent system)을 통해 각 에이전트의 고품질 출력을 결합하고 하이브리드 양성 샘플 세트를 구성함으로써 안정적인 훈련에 필요한 선호 쌍을 생성합니다. 이 방법은 특정 에이전트의 약점을 극복하면서 공통된 양성 샘플에 상응하는 방식으로 에이전트를 정렬하여 성능 향상을 도모합니다.

- **Technical Details**: Mars-PO는 세 가지 주요 단계를 포함합니다: 응답 샘플 생성(Response Samples Generation), 선호 쌍 구성(Preference Pairs Construction), 및 하이브리드 선호 최적화(Hybrid Preference Optimization)입니다. 첫 단계에서는 여러 에이전트가 생성한 응답 샘플을 모아 다양하고 대표적인 출력을 확보합니다. 다음으로, 보상 모델을 사용하여 양성 샘플의 품질을 측정하고, 이어서 각 에이전트 특이적 부정 샘플과 결합하여 훈련 과정에 필요한 선호 쌍을 구성합니다.

- **Performance Highlights**: Mars-PO는 MATH 벤치마크에서 최신 지침 조정된 LLM인 Llama3.1-8B-Instruct의 정확도를 50.38%에서 57.82%로 향상시킵니다. 실험 결과, Mars-PO는 단일 에이전트 DPO 및 기타 고급 미세 조정 방법 등 기준 접근 방식들을 일관되게 초월하며, 이를 통해 캐패시티 향상과 정렬 최적화에서 새로운 기준을 설정합니다.



### Comprehensive Survey of Reinforcement Learning: From Algorithms to Practical Challenges (https://arxiv.org/abs/2411.18892)
Comments:
          79 pages

- **What's New**: 이 논문은 Reinforcement Learning (RL)의 최신 발전을 종합적으로 조사하며 다양한 알고리즘을 평가합니다. RL은 환경과의 상호작용을 통해 최적의 행동을 학습하는 데 중점을 두며, 심층 강화 학습(Deep Reinforcement Learning, DRL)과 같은 고급 기법으로 발전하고 있습니다. 논문의 목적은 RL 알고리즘의 선택과 구현에서 발생하는 문제점을 다루고 최적의 알고리즘 선택을 위한 실제적 통찰을 제공합니다.

- **Technical Details**: RL의 기본 개념은 에이전트가 환경과 상호작용하면서 목표를 달성하기 위해 행동을 선택하는 것입니다. 이 과정에서 에이전트는 상태를 인식하고, 행동을 통해 환경의 상태를 변화시킵니다. RL 시스템은 정책(policy), 보상 신호(reward signal), 가치 함수(value function), 그리고 선택적으로 환경 모델(model)로 구성됩니다. 두 가지 주요 방법론인 모델 프리(model-free)와 모델 기반(model-based)의 장단점을 논의하며, Markov Decision Process (MDP)와 탐험-착취 균형(exploration-exploitation dilemma)도 설명합니다.

- **Performance Highlights**: 이 논문은 RL 알고리즘의 강점과 약점을 다양한 환경에서 비교합니다. RL은 게임, 로봇 공학, 자율 시스템 및 스마트 교통 시스템과 같은 여러 분야에서 활용될 수 있습니다. 효과적인 RL의 활용은 복잡한 실제 문제 해결에 큰 기여를 할 것으로 기대됩니다. 이 연구는 RL의 잠재력을 극대화하려는 연구자 및 실무자에게 유용한 참고 자료로서 기능할 것입니다.



### The Performance of the LSTM-based Code Generated by Large Language Models (LLMs) in Forecasting Time Series Data (https://arxiv.org/abs/2411.18731)
- **What's New**: 이 논문은 ChatGPT, PaLM, LLaMa, 그리고 Falcon과 같은 대형 언어 모델(LLMs)이 시간 시계열 데이터 분석을 위한 심층 학습 모델 생성을 어떻게 수행하는지를 비교하고 분석합니다. 특히, 데이터 분석가가 복잡한 코드를 수동으로 작성하지 않고 LLM을 활용해 모델을 생성할 수 있는 가능성에 주목하고 있습니다. 이 연구는 LLM 사용의 정확성과 효율성을 강조하며, 특히 ChatGPT의 우수성을 조명합니다.

- **Technical Details**: 연구는 네 가지 주요 기준인 1) 명확성과 구체성, 2) 목적과 의도, 3) 맥락 정보, 4) 형식과 스타일에 따라 제어된 실험을 수행했습니다. 주어진 프롬프트에 대한 민감도 수준에 따라 LLM이 생성하는 심층 학습 모델의 성능을 분석하며, '온도(temperature)' 매개변수가 성능에 미치는 영향을 평가합니다. 실험은 단순한 프롬프트에서 복잡한 프롬프트에 이르기까지 다양한 설정을 포함합니다.

- **Performance Highlights**: 결과적으로, LLM은 수동으로 작성된 LSTM 모델과 유사한 성능을 내는 심층 학습 모델을 생성할 수 있는 것으로 확인되었습니다. 또한, ChatGPT는 다른 LLM보다 더 정확한 모델을 생성하는 경향을 보였으며, 성능은 온도 매개변수의 설정에 따라 달라지는 것으로 나타났습니다. 흥미롭게도, 복잡한 프롬프트가 항상 더 나은 결과를 가져오지 않는 경우도 관찰되었으며, 이는 모델 성능에 있어 프롬프트 설계의 중요성을 잘 보여줍니다.



### ScaleViz: Scaling Visualization Recommendation Models on Large Data (https://arxiv.org/abs/2411.18657)
Comments:
          Accepted at PAKDD 2024 (Oral)

- **What's New**: 이 논문에서는 기존의 데이터 세트의 통계를 분석하는 데 소모되는 시간을 대폭 줄일 수 있는 새로운 강화 학습 기반 프레임워크인 ScaleViz를 제안합니다. ScaleViz는 사용자가 제공하는 시간 예산 내에서 가장 효과적인 입력 통계 세트를 식별하여, 보다 빠르게 시각적 통찰력을 생성할 수 있도록 합니다. 이러한 기법은 특히 복잡하고 대규모의 실제 데이터 세트에서 효과적입니다.

- **Technical Details**: ScaleViz 프레임워크는 몇 가지 단계로 구성됩니다. 첫 번째로는 데이터 샘플의 다양성에 따라 각 통계의 계산 비용을 프로파일링합니다. 두 번째로는 회귀 모델을 사용하여 전체 데이터 세트의 크기에 맞춰 이 비용을 추정합니다. 세 번째 단계에서는 예산 인식형 강화 학습 기법을 통해 원래의 Vis-Rec 모델에서 가장 중요한 특징을 찾습니다. 마지막으로, 선택된 통계적 특성만을 전체 데이터 세트에서 계산하여 시각화 추천을 생성합니다.

- **Performance Highlights**: 논문에서는 ScaleViz가 두 가지 최신 ML 기반 Vis-Rec 모델과 네 개의 대규모 공개 데이터 세트를 사용하여 시각화 추천을 생성하는 데 있어 최대 10배의 속도 향상을 보여줍니다. 이는 결과적으로 시각화 추천의 정확도를 유지하면서 효율성을 크게 향상시키는 것을 가능하게 합니다. 이 접근법은 특히 대규모 데이터 작업에 필요한 계산 자원을 대폭 줄일 수 있는 잠재력을 지니고 있습니다.



### DELT: A Simple Diversity-driven EarlyLate Training for Dataset Distillation (https://arxiv.org/abs/2411.19946)
- **What's New**: 이번 연구에서는 데이터셋 디스틸레이션(dataset distillation) 방법의 다각화된 접근 방식인 Diversity-driven EarlyLate Training (DELT) 방안을 제안합니다. 이 방법은 배치-to-글로벌 매칭(batch-to-global matching)에서 이미지를 다채롭게 만들어 훈련 효율성을 높일 수 있도록 설계되었습니다. DELT는 정의된 IPC 샘플을 작은 하위 작업으로 나누고 각 하위 집합의 최적화를 통해 독특한 분포로 변환하여 일관된 최적화 과정으로 생기는 균일성을 줄입니다.

- **Technical Details**: DELT는 각 카테고리의 이미지를 서로 다른 초기점에서 최적화하여 완성된 결과가 상당히 다양하게 나오도록 하며, 이러한 접근 방식은 기존 방법들과 비교할 때 상당히 단순합니다. 이 방식은 Synthetic 이미지 초기화를 위한 Teacher-ranked 실제 이미지 패치를 사용하여 최적화의 부족을 방지하고, 최종 생성된 이미지의 다양성을 높입니다. 전통적인 최적화 방법과 비교할 때, DELT는 ImageNet-1K에서 훈련 과정의 계산량을 39.3%까지 줄일 수 있음을 보여줍니다.

- **Performance Highlights**: DELT는 CIFAR, Tiny-ImageNet, ImageNet-1K와 같은 다양한 데이터셋에서 이전의 최첨단 방법들보다 평균 2~5% 더 우수한 성능을 보였습니다. 특히 ImageNet-1K에서 IPC 50에서 66.1%의 정확도를 달성했으며, 이는 기존 RDED보다 4.9% 향상된 결과입니다. 작은 데이터셋인 CIFAR-10에서도 RDED 및 SRe2L보다 각각 2.5%와 19.2% 개선된 성과를 달성했습니다.



### Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM's Reasoning Capability (https://arxiv.org/abs/2411.19943)
Comments:
          Work in progress

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)에서의 비판적 토큰(critical tokens)의 존재를 밝히고, 이들 토큰이 잘못된 추론 경로에서의 결과에 미치는 영향을 분석한다. 새로운 접근 방식인 cDPO를 통해, 토큰 수준의 보상을 자동으로 인식하고 제공함으로써 정렬(alignment) 과정을 개선한다. 이 연구는 LLM의 추론 과제에서 비판적 토큰의 중요성을 강조하며, 이전보다 더 정교한 최적화 방법을 제안한다.

- **Technical Details**: cDPO는 대형 언어 모델에서의 비판적 토큰을 인식하기 위해 모델의 긍정적 및 부정적 경로의 생성 가능성을 비교하는 대조적 추정(contrastive estimation) 방법을 사용한다. 이러한 방식으로, 성공적인 결과를 생성하는 토큰과 그렇지 않은 토큰 간의 차이를 평가하여 비판적 토큰을 인식하고, 이 정보를 기반으로 토큰 레벨에서의 보상을 사용하여 모델을 재정렬한다. 연구에서는 Llama-3 및 deepseek-math 모델을 사용하여 비판적 토큰이 잘못된 결과에 미치는 영향을 실험적으로 검토하였다.

- **Performance Highlights**: cDPO는 여러 벤치마크에서 실행된 실험에서 기존의 예시 수준 및 단계 수준의 기준보다 우수한 성과를 보였다. 특히, GSM8K와 MATH500 벤치마크에서 조사된 결과는 제안된 방법이 토큰 수준의 보상을 포함하여 더 나은 정렬을 가능하게 한다는 것을 보여준다. 또한, cDPO는 잘못된 추론 경로에서 비판적 토큰을 효과적으로 식별함으로써, 결과의 정확성을 크게 향상시키는 것을 입증하였다.



### VLSBench: Unveiling Visual Leakage in Multimodal Safety (https://arxiv.org/abs/2411.19939)
- **What's New**: 이번 연구에서는 Multimodal large language models (MLLMs)의 안전성과 관련하여 새로운 문제를 제기하고 있습니다. 특히, 기존의 데이터셋에서 시각적 정보 누출(Visual Safety Information Leakage, VSIL) 문제가 존재한다는 점을 발견했습니다. 이로 인해 MLLMs가 이미지와 텍스트 쿼리를 기반으로 민감한 내용을 쉽게 감지할 수 있으나, 실제 상황에서는 VSIL이 없는 이미지-텍스트 쌍도 많다는 것을 지적합니다. 이 연구는 이러한 문제를 해결하기 위한 새로운 벤치마크인 VLSBench를 구축했습니다.

- **Technical Details**: VLSBench는 2.4k 개의 이미지-텍스트 쌍을 포함한 멀티모달 안전 기준으로, 시각적 안전 정보의 누출을 방지합니다. 연구자들은 LLM을 활용하여 해로운 텍스트 쿼리와 이미지를 생성하고, 이후 이를 정제하여 안전한 텍스트 쿼리와 질 높은 이미지를 조합했습니다. 실험 결과, VLSBench는 기존의 MLLM 뿐만 아니라 공개 및 폐쇄 소스 모델에게도 상당한 도전 과제가 됩니다. 특히, 기존 안전 벤치마크가 VSIL 문제에 취약하다는 것을 경험적으로 확인했습니다.

- **Performance Highlights**: VLSBench에서의 실험 결과에 따르면, 단순한 텍스트 SFT 방법이 MLLM의 안전성을 유지하면서도 95% 이상의 성능을 달성할 수 있음을 보여줍니다. 반면, 멀티모달 정렬 방법은 VSIL이 없는 경우에 더 유망한 해결책이 될 수 있습니다. 이러한 발견들을 기반으로, MLLMs의 안전성 접근 방식에서 텍스트 및 멀티모달 정렬 방법의 효용성을 비교했습니다. 결과적으로, 시각적 안전 정보 누출이 텍스트 방식의 우수한 성능의 원인임을 입증했습니다.



### Dynamic EEG-fMRI mapping: Revealing the relationship between brain connectivity and cognitive sta (https://arxiv.org/abs/2411.19922)
Comments:
          15 pages, Subjects: Machine Learning (cs.LG); Human-Computer Interaction (cs.HC); Signal Processing (eess.SP)

- **What's New**: 이번 연구는 EEG와 fMRI 모달리티 간의 동적 연결 패턴(dynamic connectivity patterns)을 조사하여 뇌 네트워크 상호작용에 대한 이해를 증진시키고자 했습니다. 이를 위해 EEG-fMRI 데이터의 정적(static) 및 동적(dynamic) 분석을 통합한 포괄적인 접근 방식을 사용했습니다. 연구 결과는 뇌의 본질적인 연결 네트워크(intrinsic connectivity networks) 내에서 모듈 조직(modular organization)을 밝혀내며 감각 시스템(sensory systems)과 기본 모드 네트워크(default mode network)의 중요한 역할을 강조했습니다.

- **Technical Details**: 연구에서는 슬라이딩 윈도우 기법(sliding window technique)을 사용하여 기능적 연결성이 시간이 지남에 따라 어떻게 변하는지를 평가했습니다. 이를 통해 뇌 연결의 일시적(transient) 특성을 더욱 분명히 밝혔습니다. 또한, 30-60초의 짧은 기간(data) 내에서 인지 상태(cognitive states)를 효과적으로 식별할 수 있다는 이전 연구와의 일치성을 보였습니다. 연결 강도(connectivity strength)와 인지 과정(cognitive processes) 간의 관계가 시각 상태에 따라 어떻게 달라지는지를 강조하여 향후 연구의 중요성을 나타내었습니다.

- **Performance Highlights**: 이 연구는 EEG와 fMRI 신호 간의 상호작용(interplay)을 이해하는 데 기여하며 인지 기능(cognitive functions) 및 임상적 문맥에 대한 신경 상관관계(neural correlates)를 추가로 탐구할 수 있는 기반을 마련했습니다. 앞으로의 연구는 이러한 방법론(methodologies)을 더욱 정제하고 다양한 인지 및 임상적 맥락에서의 응용 가능성을 탐구하는 데 집중해야 할 것입니다.



### SIMS: Simulating Human-Scene Interactions with Real World Script Planning (https://arxiv.org/abs/2411.19921)
- **What's New**: 이 논문은 물리적으로 그럴듯한 장기 인간-장면 상호작용을 계획하고 제어하기 위한 새로운 프레임워크를 제시합니다. 특히, 인터넷에 존재하는 비디오 데이터를 활용하여 LLM 기반의 스크립트 추출 및 생성 과정을 통합합니다. 이를 통해 인과적이고 복잡한 시간 연속적인 인간 행동을 모사하며, 다양한 장면에 대한 이해를 통해 캐릭터의 동작을 유도할 수 있습니다.

- **Technical Details**: 프레임워크 SIMS(SIMultating huMan Scene interactions)는 고수준 계획에 LLM을, 저수준 제어에는 물리적 정책을 사용합니다. LLM을 통한 상호작용과 감정 변화를 실시간 비디오에서 추출하여 스크립트 데이터베이스를 구성하고, 이를 통해 키프레임을 생성합니다. 또한, CLIP 모델을 통해 장면 기하학과 텍스트 임베딩을 인식하여 고품질 동작 생성을 위한 듀얼 인식 정책을 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크는 다양한 작업 수행에 있어 뛰어난 성능을 보이며 기존 방법들에 비해 일반화 능력이 향상되었습니다. 또한, 지능형 캐릭터들이 다양한 3D 환경에서 요구되는 다양한 동작을 실행하면서도 정교한 스타일을 갖출 수 있도록 합니다. 이 프레임워크의 코드는 곧 공개될 예정이며, 실제 응용 가능성이 높습니다.



### Quantifying the synthetic and real domain gap in aerial scene understanding (https://arxiv.org/abs/2411.19913)
Comments:
          17 pages (including references), 5 figures, 2 tables. Accepted for publication in the "Scientific Bulletin", Series C, Electrical Engineering and Computer Science, ISSN 2286-3540

- **What's New**: 이 논문은 합성 이미지와 실제 이미지 간의 격차를 정량화하는 새로운 방법론을 제시합니다. Multi-Model Consensus Metric (MMCM)와 depth 기반 구조 메트릭을 활용하여 장면 복잡도를 평가합니다. 이는 특히 항공 장면 이해와 같은 탐색되지 않은 분야에서 큰 영향을 미칠 수 있습니다.

- **Technical Details**: 실험 분석에서는 실제 데이터셋(Dronescapes)과 합성 데이터셋(Skyscenes)을 사용하여 모델 성능을 평가했습니다. 연구 결과, 실제 장면이 최신 비전 변환기(state-of-the-art vision transformers) 간의 높은 일관성을 보이는 반면, 합성 장면은 변동성이 크고 모델 적응을 도전하게 합니다. 이러한 차이는 구조적 및 인지적 불일치를 강조합니다.

- **Performance Highlights**: 결과는 복잡성과 도메인 간 격차의 본질적인 문제를 강조하며, 향상된 시뮬레이션 충실도와 모델 일반화의 필요성을 시사합니다. 이 연구는 도메인 특성과 모델 성능 간의 상호작용에 대한 중요한 통찰을 제공하며, 항공 장면 이해에 대한 개선된 도메인 적응 전략 개발을 위한 경로를 제시합니다.



### LUMIA: Linear probing for Unimodal and MultiModal Membership Inference Attacks leveraging internal LLM states (https://arxiv.org/abs/2411.19876)
- **What's New**: 이 논문에서는 Membership Inference Attacks (MIAs)에 대한 새로운 접근 방식인 LUMIA(Linear Probes for Membership Inference Attacks)를 제안합니다. 이전 연구들이 까만 박스(black-box) 모델에서 그레이 박스(grey-box) 모델로 초점을 옮겼으나, LUMIA는 언어 모델 내부의 활성화(activations)를 활용하여 MIAs를 탐지합니다. 이는 내부 정보를 활용함으로써 MIAs에 대한 검출 성능을 크게 향상시킬 수 있습니다.

- **Technical Details**: LUMIA는 각 레이어(layer)에 대해 Linear Probes (LPs)를 적용하여 모델의 내부 작업을 세밀하게 파악합니다. 해당 방법은 단일 모달(unimodal) 및 다중 모달(multimodal) 작업에서 다양한 모델 아키텍처(architecture), 크기(size), 데이터셋을 통해 시험되었습니다. 이를 통해 MIA를 탐지하는 데 있어 각 레이어에서의 감지 가능성이 변수화되었습니다.

- **Performance Highlights**: LUMIA는 단일 모달 MIA에서 이전 기술에 비해 평균 15.71%의 Area Under the Curve (AUC) 향상을 달성했습니다. 특히 AUC가 60%를 초과하는 경우는 65.33%로, 기존 기술 대비 46.80% 증가한 수치입니다. 다중 모달 모델에서도 시각적 입력이 MIAs 탐지에 중요한 기여를 하여, 85.90%의 실험에서 AUC가 60%를 초과했습니다.



### Enhanced anomaly detection in well log data through the application of ensemble GANs (https://arxiv.org/abs/2411.19875)
- **What's New**: 본 연구는 전통적인 방법인 Gaussian mixture models (GMMs) 대비 Ensemble GANs (EGANs) 프레임워크를 활용하여 석유 탐사에 필요한 well log 데이터의 분포를 모델링하고 이상치를 탐지하는 방법을 제안합니다. EGANs는 여러 GAN을 집합하여 복잡한 데이터 구조를 더욱 잘 파악하고, GMM보다 우수한 성능을 발휘하는 것으로 나타났습니다. 특히 GR, DT, NPHI 및 RHOB 데이터셋에서 이상 탐지의 정밀도와 F1 점수가 GMM을 초월하여 보다 신뢰할 수 있는 결과를 보여 줍니다.

- **Technical Details**: 연구에 사용된 데이터셋은 북해의 두 개 wells에서 수집된 상세한 well log 데이터로, GR, DT, NPHI, RHOB 항목에서 총 6553개의 데이터 포인트가 포함됩니다. K-Means clustering 기법을 통해 다차원 특성 간의 관계를 고려하여 데이터 클러스터링을 수행하고, 이후 Isolation Forest(IF) 알고리즘을 적용하여 선택된 클러스터에서 이상치를 추출합니다. 각 모델(GMM 및 EGAN)은 성능을 최적화하기 위해 맞춤형 하이퍼파라미터로 구성되어, 연구의 재현성을 용이하게 합니다.

- **Performance Highlights**: EGANs는 GR 데이터셋에서 0.62의 정밀도와 0.76의 F1 점수를 기록하여 GMM의 0.38 및 0.54를 초과했습니다. 유사하게, DT 데이터셋에서는 EGANs가 정밀도 0.70, F1 점수 0.79로 GMM의 수치보다 월등한 결과를 보였습니다. 연구 결과, EGANs의 도입은 전통적인 방법보다 이상 탐지 및 모델링에서 더 뛰어난 성능을 보여 주었다는 점이 강조됩니다.



### DeMo: Decoupled Momentum Optimization (https://arxiv.org/abs/2411.19870)
- **What's New**: 이 논문에서는 대규모 신경망 훈련 시, 가속기 간의 고속 인터커넥트(Interconnect) 없이도 효율적으로 훈련할 수 있는 새로운 방법을 제시합니다. 전체 최적화기 상태와 모델 파라미터를 동기화하는 것이 불필요하다는 것을 보여주며, 모멘텀 업데이트(Momentum Updates)를 분리해 가속기 간의 최적화기 상태를 조절된 방식으로 수렴시킬 수 있습니다. 이로 인해 기존의 최첨단 최적화기들에 비해 향상된 수렴 성능을 달성하는 방법인 디커플드 모멘텀(Decoupled Momentum, DeMo)을 소개합니다.

- **Technical Details**: DeMo는 데이터 병렬화(Data Parallel) 알고리즘과 융합된 최적화기(Fused Optimizer)로 설계되었습니다. 이 방법은 가속기 간 통신 요구 사항을 수십 배 줄여주며, 제한된 네트워크 대역폭(Bandwidth)과 이질적인 하드웨어에서도 대규모 신경망 훈련을 가능하게 합니다. 또한, 이 방법은 토폴로지(Topology)와 아키텍처(Architecture)에 구애받지 않으며, 컴퓨팅 및 메모리 오버헤드가 매우 적은 분산 훈련(distributed training)을 지원합니다.

- **Performance Highlights**: DeMo로 훈련된 모델은 AdamW로 훈련된 동등한 모델의 성능을 초과하거나 동등한 성능을 보여주며, 대규모 기초 모델(Foundation Models)의 사전 훈련 시 고속 인터커넥트의 필요성을 제거합니다. 이러한 결과는 DeMo의 효율성과 성능 개선 가능성을 강하게 뒷받침하고 있습니다. 또한, 구현된 코드는 GitHub에 오픈소스로 공개되어 있어, 연구자들이 쉽게 접근하여 사용할 수 있도록 하고 있습니다.



### Reverse Thinking Makes LLMs Stronger Reasoners (https://arxiv.org/abs/2411.19865)
Comments:
          20 pages

- **What's New**: 이 논문에서는 Reverse-Enhanced Thinking(RevThink)이라는 새로운 프레임워크를 소개합니다. RevThink는 데이터 증강(data augmentation)과 학습 목표(learning objectives)를 통해 대형 언어 모델(LLMs)에서 역방향 사고(reverse thinking)를 수행할 수 있도록 합니다. 이 방법은 원래 질문과 답변 외에도 앞에서의 추론(forward reasoning)과 뒤에서의 질문(backward question), 그리고 뒤에서의 추론(backward reasoning)을 포함하여 데이터를 증강합니다.  

또한 RevThink는 세 가지 주요 목표를 설정해 학생 모델(student model)을 훈련시키며, 이를 통해 모델이 질문으로부터 올바른 종합적 추론을 생성할 수 있도록 돕습니다.

- **Technical Details**: RevThink는 Chain-of-Thought와 같이 매우 구조화된 방식으로 추론을 수행하는 대형 모델을 활용합니다. 데이터 증강을 통해 생성된 두 가지 방향의 사고를 통해 학생 모델이 동시에 앞뒤로 사고할 수 있도록 돕습니다. 목표 설정은 (1) 질문에서 올바른 전진 추론을 생성, (2) 원래 질문으로부터 역 질문을 생성, (3) 역 질문으로부터 역 추론을 생성하는 것입니다. 이러한 방법론은 학생 모델이 테스트 시에도 효율적인 계산을 유지할 수 있도록 합니다.

- **Performance Highlights**: RevThink는 12개의 다양한 데이터셋에서 실험을 수행했으며, 평균적으로 학생 모델의 제로샷 성능(zero-shot performance) 대비 13.53% 향상된 성능을 보였습니다. 또한, 기존의 심볼릭 지식 증류(Symbolic Knowledge Distillation, SKD) 방법보다 6.84% 더 나은 성과를 기록했습니다. 데이터셋의 양이 적은 경우에도 학습한 정확도가 높아, 10%의 훈련 샘플만으로도 전체 데이터셋을 활용한 SKD의 성능을 초과할 수 있습니다. RevThink는 다양한 신규 데이터셋에서 또한 일반화(generalization)에 강점을 보입니다.



### Scaling Transformers for Low-Bitrate High-Quality Speech Coding (https://arxiv.org/abs/2411.19842)
- **What's New**: 이 논문은 현대 AI 파이프라인에서 음성의 생성 또는 이해를 위한 중요성에 대한 통찰을 제공합니다. 전통적인 음성 코덱 모델이 저자기(低自記) 수치로 제한되었다면, 본 연구에서는 파라미터 수를 대폭 증가시킨 트랜스포머 아키텍처를 활용하여 음성 품질을 향상시킬 수 있음을 보여줍니다. 특히 400 bps와 700 bps라는 극히 낮은 비트 전송률에서 최첨단 음성 품질 달성을 가능케 하는 방안을 제안하였습니다.

- **Technical Details**: 연구에서 제안된 모델은 주로 트랜스포머 기반으로 설계되어 1111B 파라미터 범위로 확장 가능성을 보여줍니다. 전통적인 Residual Vector Quantization (RVQ) 대신 Finite Scalar Quantization (FSQ) 방식의 양자화를 적용해 여러가지 훈련 및 추론의 복잡성을 해결하였습니다. 이를 통해 고품질 음성을 자연스럽게 재현하면서도 높은 압축 비율을 달성하는 바이트 중요도로서의 통신 효율성을 기대할 수 있습니다.

- **Performance Highlights**: 제안된 음성 코덱 모델은 객관적 및 주관적 테스트에서 기존 기준과 비교하여 현저한 성능 향상을 보여줍니다. 특히 400bps 및 700bps의 비트 전송률에서 높은 압축률을 유지하면서도 좋은 음질을 보장합니다. 이러한 결과는 최신 생성 아키텍처와 함께 음성을 이해하고 생성하기 위한 새로운 방향성을 제시합니다.



### Q-learning-based Model-free Safety Filter (https://arxiv.org/abs/2411.19809)
Comments:
          *Denotes equal contribution

- **What's New**: 이 논문은 복잡한 시스템에서 RO 시스템이 안전을 보장하는 데 있어 기존의 모델 기반 방법의 한계를 극복하기 위해 모델 프리(model-free) 방법을 기반으로 한 새로운 안전 필터 학습 프레임워크를 제안합니다. 특히, 제안된 방법은 기존의 안전 메커니즘보다 더 간단하고 효과적으로, 특정 작업 정책을 보호할 수 있는 Q-learning을 사용하여 안전을 보장합니다. 이 접근법은 다양한 RL 알고리즘과의 통합이 용이하여 실제 로봇 운용에 적합합니다.

- **Technical Details**: 제안하는 방법은 모델 프리(MD-free) 안전 필터로, 이론적으로 최적 조건에서 안전을 보장하며, 비최적 조건에서도 견고한 성능을 발휘합니다. 연구의 주요 기여로는 새로운 보상 구조를 도입하여 에피소드 동안 안전을 보장하고, 각 작업 정책과 안전 필터를 별도로 학습할 수 있는 구조를 제안합니다. 이 프레임워크는 큐-값 함수(Q-value function)를 학습하여 잠재적으로 위험한 행동을 필터링하여 안전성을 유지합니다.

- **Performance Highlights**: 제안된 접근 방식은 더블 인티그레이터(double integrator)와 더빈 차량(Dubin's car) 시스템을 포함한 시뮬레이션에서 검증되었으며, 소프트 로봇을 통한 실제 실험에서도 효과가 입증되었습니다. 이 연구는 훈련 부정확성에 강인하며, 복잡한 동적 시스템에서도 일반화됩니다. 새로운 안전 보장 메커니즘 덕분에 원활한 RL 통합과 훈련이 가능해집니다.



### Zero-shot Musical Stem Retrieval with Joint-Embedding Predictive Architectures (https://arxiv.org/abs/2411.19806)
Comments:
          Submitted to ICASSP 2025

- **What's New**: 이번 논문에서는 음악 믹스에서 적합한 스템(stem)을 검색하는 새로운 방법을 제안합니다. 우리는 Joint-Embedding Predictive Architectures(JEPA)에 기반한 방식을 도입하여, 인코더와 예측기가 공동으로 훈련됩니다. 이 모델은 다양한 악기를 조건으로 지정하여 제로샷(zero-shot) 스템 검색이 가능하다는 점에서 혁신적입니다. 또한, 대조 학습(contrastive learning)을 사용하여 인코더를 사전 훈련(pretraining)함으로써 성능을 크게 향상시킵니다.

- **Technical Details**: 우리 모델은 인코더 f_θ와 예측기 g_φ로 구성된 두 개의 훈련 가능한 네트워크로 이루어져 있습니다. 인코더는 Log-scaled Mel-Spectrograms(LMS)를 입력으로 받아 latent representation을 생성하고, 예측기는 이 embeddings를 활용하여 타겟 스템의 latent representation을 예측합니다. 이 과정은 두 단계로 나누어 진행되며, 첫번째 단계에서는 contrastive learning을 통해 인코더만 사전 훈련되고, 두번째 단계에서는 인코더와 예측기가 모두 훈련됩니다. 우리는 모델이 임의의 악기에 대한 조건을 지원하도록 설정하여, 이전의 방법들보다 더 풍부한 latent 공간을 학습할 수 있도록 합니다.

- **Performance Highlights**: 우리는 MUSDB18과 MoisesDB 데이터셋을 사용하여 모델의 검색 성능을 평가하였으며, 이전 기준 모델들보다 상당히 우수한 성능을 보였습니다. 다양한 조건의 레벨에서 모델의 디자인 선택들이 효과적인지 검증하였고, beat tracking 작업에서도 모델의 embedding이 시간 구조와 지역 정보를 보존한다는 것을 보여주었습니다. 이러한 결과는 이 모델이 스템 검색을 넘어 다른 작업에도 유용할 수 있음을 암시합니다.



### Advanced System Integration: Analyzing OpenAPI Chunking for Retrieval-Augmented Generation (https://arxiv.org/abs/2411.19804)
- **What's New**: 이 연구는 여러 (sub-) 시스템을 통합하여 고급 정보 시스템(Information Systems, ISs)을 만드는 중요성을 강조합니다. 이러한 통합 과정에서 동적 환경을 관리하는 데 어려움이 있으며, 기존 방법인 레지스트리를 통한 API 문서 제공이 효과적이나 제한이 있습니다. 특히, 대형 언어 모델(large Language Models, LLMs)이 API 문서를 기반으로 시스템 통합을 자동으로 생성할 수 있는 가능성을 보여주고 있습니다.

- **Technical Details**: 이 연구에서는 두 가지 주요 접근법을 제안합니다. 첫째, Retrieval Augmented Generation (RAG)을 활용하여 OpenAPIs의 청크화(chunking)을 통해 입력 토큰 길이를 줄이면서도 관련 정보를 보존합니다. 둘째, Discovery Agent를 통해 가장 관련성 높은 엔드포인트 요약을 수집하고 필요한 상세 정보를 요청합니다. 이를 통해 엔드포인트 검색 시 성능을 향상시키고 입력 토큰 수를 감소시킬 수 있게 됩니다.

- **Performance Highlights**: RestBench 벤치마크를 사용하여 RAG의 엔드포인트 검색 성능을 평가하였으며, 향상된 recall, precision, F1 스코어를 기록했습니다. Discovery Agent 또한 동일한 테스트 세트를 바탕으로 성능을 검증하였으며, 전반적으로 LLM 기반 및 포맷 특정 접근 방식이 단순 청크 방법보다 더 우수한 성과를 나타냈습니다. 연구 결과, 작업을 세분화하여 RAG 성능을 개선함으로써 토큰 수를 줄이고 정밀도를 높일 수 있음을 확인했습니다.



### Voice Communication Analysis in Esports (https://arxiv.org/abs/2411.19793)
Comments:
          17 pages, 11 figures. Independent research

- **What's New**: 본 논문은 팀 기반 e스포츠에서 음성 통신의 효과성을 개선하기 위한 연구를 다루고 있으며, 특히 LoL(League of Legends) 게임을 중심으로 진행되었습니다. 이에 따라 LLM(Large Language Models)와 NLP(Natural Language Processing) 기술을 활용하여 효과적인 음성 커뮤니케이션의 지표를 개발하고 있습니다. 또한, 커뮤니케이션의 중복성과 비약한 소통 문제를 찾아 해결책을 제안하고자 합니다.

- **Technical Details**: 연구에서는 음성 파일을 분석하기 위해 Bain et al.의 방법을 바탕으로 Whisper로 음성을 텍스트로 전사하고, PixIT 모델을 통해 화자 단별 분석(speaker diarization)을 진행합니다. 이 과정에서 음성 간의 단어 정렬(forced-text alignment)을 수행하여 각 플레이어의 발화와 타임스탬프를 일치시킵니다. 현재 음성 통신에 관한 공개 데이터셋이 부족하여 few-shot 기법을 적용할 필요성이 강조되고 있습니다.

- **Performance Highlights**: 논문에서는 중복 통신의 탐지를 위해 문장 유사도(semantic similarity)를 활용하는 접근법을 제안합니다. TF-IDF와 같은 전통적인 방식 대신, 신경망 기반의 방법을 사용하여 더 나은 성능을 발휘합니다. 기법의 최종 목표는, e스포츠 팀의 커뮤니케이션의 질이 경기 성능에 미치는 양상과 상관관계를 분석하는 것입니다.



### CAREL: Instruction-guided reinforcement learning with cross-modal auxiliary objectives (https://arxiv.org/abs/2411.19787)
- **What's New**: 이번 연구에서는 언어 지시문을 환경에 기반하여 해석하는 데 중요한 역할을 하는 CAREL(교차 모달 보조 강화 학습) 프레임워크를 제안합니다. 이 프레임워크는 비디오-텍스트 검색 분야에서 영감을 받아 보조 손실 함수와 새로운 지시 추적 방법을 사용하여 목표 달성 문제를 해결합니다. CAREL의 도입으로 언어 지시문과 환경 관찰 간의 정렬을 개선하고, 다중 모달 강화 학습 문제에서 뛰어난 샘플 효율성과 일반화 능력을 발휘할 수 있음을 보여주었습니다.

- **Technical Details**: CAREL는 X-CLIP 모델의 보조 손실을 도입하여 지시문과 관찰 간의 정렬을 향상시킵니다. 이 보조 손실은 기본 강화 학습 과제를 보완하는 역할을 하며, 의미 있는 정보를 추출하고 지시문과 효과적으로 정렬하는 학습 신호를 제공합니다. 새로운 방법인 지시 추적은 보조 점수를 사용하여 수행된 지시문 부분을 마스킹하고, 남은 작업에 집중하게 하여 샘플 효율성을 높이는 데 기여합니다.

- **Performance Highlights**: BabyAI 환경에서의 실험 결과, CAREL 프레임워크는 지시를 따르는 에이전트의 체계적인 일반화와 샘플 효율성을 크게 향상시켰습니다. 실험에서는 제안된 보조 손실 기능이 지시문과 환경 관찰 간의 교차 모달 정렬을 개선함으로써 전반적인 성능이 증대하는 경향을 보였습니다. CAREL은 최적화된 에이전트 성능을 실현하는 데 있어 обещательный 접근법(in promising approach)으로 자리잡을 가능성이 있습니다.



### Stock Price Prediction using Multi-Faceted Information based on Deep Recurrent Neural Networks (https://arxiv.org/abs/2411.19766)
- **What's New**: 본 연구는 주식 시장에서 주식 가격 예측을 위한 새로운 접근 방식을 제안합니다. 이는 Convolutional Neural Networks (CNN)과 Long Short-Term Memory (LSTM) 네트워크를 통합하고, 소셜 네트워크 데이터에 대한 감성 분석을 이용합니다. 이를 통해 가격에 관한 캔들스틱 데이터와 트위터로부터 얻은 통찰력을 결합하여 보다 정밀한 시장 경향 분석을 제공합니다.

- **Technical Details**: 제안된 방법론은 두 가지 주요 요소로 구성됩니다: 소셜 네트워크 및 캔들스틱 데이터에 대한 감성 분석입니다. 캔들스틱 데이터와 트위터 데이터를 결합하여 시장 트렌드와 패턴을 보다 상세하게 분석합니다. 더 나아가, Random Forest 알고리즘을 사용해 트윗을 긍정적 또는 부정적으로 분류하여 시장의 감성을 보다 세밀하게 평가합니다.

- **Performance Highlights**: 이 연구의 결과는 CNN이 단기 특징을 추출하고, LSTM이 장기 종속성을 모델링함으로써 시장 경향 분석을 보다 종합적으로 수행하는 데 기여함을 보여줍니다. 이러한 통합 접근 방식은 더 정확한 주식 가격 예측으로 이어지며, 투자 결정을 위한 보다 효과적인 방법을 제공합니다.



### Forecasting Foreign Exchange Market Prices Using Technical Indicators with Deep Learning and Attention Mechanism (https://arxiv.org/abs/2411.19763)
- **What's New**: 이 연구에서는 외환시장(foreign exchange market)에서 가격 행동을 정확하게 예측하기 위한 새로운 접근 방식을 제안합니다. 기존의 기술 지표(technical indicators)와 딥 뉴럴 네트워크(deep neural networks)를 활용하는 이 방법은, LSTM(Long Short-Term Memory)과 CNN(Convolutional Neural Network), 그리고 attention 메커니즘을 통합한 아키텍처로 구성됩니다. 이를 통해 가격 추세와 시장 변동성을 포착할 수 있습니다.

- **Technical Details**: 이 연구는 외환 통화 쌍 데이터에서 통계적 특성을 추출하기 위해 트렌드(trend)와 오실레이션(oscillation) 기술 지표를 사용합니다. LSTM과 CNN 네트워크는 병렬로 작동하여 미래 가격 변동을 예측하는데, LSTM 네트워크는 장기 의존성(long-term dependencies)을 캡처하고 CNN 네트워크는 지역 패턴(local patterns)을 추출합니다. 마지막으로 두 네트워크의 출력은 attention 메커니즘에 입력되어 각 특성과 시간적 의존성의 중요도를 학습합니다.

- **Performance Highlights**: 제안된 접근 방식은 여러 종류의 외환 통화 쌍에 대해 포괄적으로 검증되었으며, 가격 행동 예측에 대한 효과성을 보여줍니다. 이 모델은 기존의 벤치마크 모델(benchmark models)을 능가하는 성과를 나타내며, 특히 가장 관련성 높은 특성과 시간적 의존성에 집중할 수 있는 능력을 갖추고 있습니다.



### LaVIDE: A Language-Vision Discriminator for Detecting Changes in Satellite Image with Map References (https://arxiv.org/abs/2411.19758)
- **What's New**: 이 논문에서는 단일 이미지만으로 변화를 감지하는 기존의 한계를 극복하기 위해, 지도를 참조하여 위성 이미지의 변화를 감지하는 방법인 LaVIDE(Language-VIsion Discriminator)를 제안합니다. LaVIDE는 텍스트로 지도를 표현하여 고수준 카테고리 정보와 저수준 시각 정보를 비교합니다. 이 방식은 이미지와 지도의 정보 간의 격차를 줄이고, 보다 효과적인 변화 감지를 가능하게 합니다.

- **Technical Details**: LaVIDE는 두 개의 병렬 브랜치로 구성되어 있습니다. 지도 브랜치에서는 지도 정보를 텍스트로 변환하여 텍스트 임베딩을 추출하며, 객체 속성을 포함한 맥락 최적화 전략을 사용해 카테고리 정보를 강화합니다. 이미지 브랜치에서는 위상 모델의 피쳐 공간과 시맨틱 얼라인먼트를 보장하며, MoE(Mixture-of-Experts) 모듈을 통해 텍스트 임베딩과 비전 임베딩을 비교하여 변화 감지를 수행합니다.

- **Performance Highlights**: 제안된 방법은 네 개의 벤치마크 데이터셋에서 검증되었으며, 기존의 최첨단 변화 감지 알고리즘들을 초과하는 성능을 기록했습니다. 예를 들어, DynamicEarthNet 데이터셋에서 약 13.8%의 성능 향상을 보였으며, SECOND 데이터셋에서는 4.3%의 향상 효과를 얻었습니다. 이러한 결과는 LaVIDE가 위성 이미지와 지도 참조를 효과적으로 통합하여 변화 감지의 정확도를 높임을 보여줍니다.



### A Multi-Loss Strategy for Vehicle Trajectory Prediction: Combining Off-Road, Diversity, and Directional Consistency Losses (https://arxiv.org/abs/2411.19747)
Comments:
          Preprint, 7 pages, 4 figures and 2 tables

- **What's New**: 이번 연구에서는 자율주행 차량의 궤적 예측 문제의 주요 한계를 해결하기 위해 세 가지 새로운 손실 함수인 Offroad Loss, Direction Consistency Error, Diversity Loss를 도입했습니다. 이러한 손실 함수는 예측 경로를 주행 가능한 한계 내에 두고, 교통 방향에 일치하도록 하며, 다양한 주행 시나리오를 커버하도록 설계되었습니다. 기존의 'winner takes all' 학습 방식의 단점을 극복하여 모든 예측 모드에서 기능을 적용, 향상된 예측 성능을 도모합니다.

- **Technical Details**: 제안된 Offroad Loss는 궤적이 주행 가능한 경계 내에 위치하도록 보장하며, 경계를 넘는 경우 페널티를 점진적으로 증가시킵니다. Direction Consistency Error는 예측된 궤적을 예상되는 교통 흐름과 도로 레이아웃에 맞춰 정렬합니다. Diversity Loss는 다양한 예측 결과를 생성하도록 유도하며, 가능성 있는 모든 결과를 커버하는 예측을 장려합니다. 이러한 손실 함수는 모든 예측 모드에 걸쳐 중복 없이 적용되며, 전반적인 예측 품질을 향상시킵니다.

- **Performance Highlights**: nuScenes 및 Argoverse 2 데이터셋에서 선두 기준 모델들과 함께 진행된 광범위한 검증 결과, 제안한 손실 함수가 궤적 예측 모델의 안전성과 강건성을 크게 향상시키는 것으로 나타났습니다. 특히, 궤적 예측의 정확도를 유지하면서도 원본 데이터셋에서 오프로드 오류를 47% 줄였고, 공격 장면에서는 37%의 감소를 기록했습니다. 이러한 결과는 복잡한 환경에서의 자율주행 궤적 예측에 대한 새로운 벤치마크를 세우는 성과를 의미합니다.



### HVAC-DPT: A Decision Pretrained Transformer for HVAC Contro (https://arxiv.org/abs/2411.19746)
Comments:
          7 pages, 3 figures, 3 tables

- **What's New**: HVAC-DPT는 in-context Reinforcement Learning(RL)을 사용하는 Decision-Pretrained Transformer로, 여러 건물 구역에서 HVAC 시스템의 최적화를 가능하게 합니다. 이 방법은 기존의 많은 제어 전략들이 요구하는 방대한 훈련과 데이터 없이도 다양한 건물에서 즉시 적용될 수 있는 장점을 가집니다. HVAC-DPT는 HVAC 제어를 시퀀스 예측 문제로 설정하여, 불확실한 환경에서도 효율적으로 작동할 수 있도록 훈련되었습니다.

- **Technical Details**: HVAC-DPT는 다중 에이전트 강화 학습(multi-agent reinforcement learning, MARL) 문제로 HVAC 제어를 모델링하며, 각 에이전트는 서로 다른 구역을 관리합니다. 에이전트는 VAV 시스템의 최소 댐퍼 위치를 조정하며, 보상 함수는 에너지 소비의 부정적인 값을 기반으로 합니다. 실험을 통해 HVAC-DPT는 EnergyPlus를 사용해 HVAC 에너지 소비를 45% 줄이는 성과를 보여주었습니다.

- **Performance Highlights**: HVAC-DPT는 1년 간의 평가에서 기존의 기준 제어기와 비교하여 45%의 에너지 절감을 달성했습니다. 이 시스템은 다양한 건물 구조와 열역학적 동적 특성에 적응할 수 있는 데이터 효율적이고 확장 가능한 접근 방식을 제공합니다. HVAC-DPT의 적용은 건물 운영의 탄소 발자국을 줄이는 데 있어 혁신적인 잠재력을 보여줍니다.



### Amplifying human performance in combinatorial competitive programming (https://arxiv.org/abs/2411.19744)
Comments:
          Technical report. 18 pages, 8 figures

- **What's New**: 최근에는 경쟁 프로그래밍을 위한 복잡한 AI 시스템의 급격한 증가가 있었습니다. 이 연구에서는 Combinatorial Competitive Programming에 초점을 맞추어, 특히 비효율적 문제에 가능한 최적의 솔루션을 찾는 방법을 탐구합니다. 독창적인 AI와 인간의 협업을 통해, Hash Code와 같은 대회에서 긍정적인 결과를 얻었습니다.

- **Technical Details**: 이 방법론은 FunSearch라는 진화적 알고리즘을 활용하여, 인간 프로그래머가 설계한 휴리스틱 솔루션의 스코어링 함수(optimal scoring function)를 최적화하는 방식입니다. Hash Code와 같은 대회에서 인간이 설계한 솔루션의 성능을 크게 향상시켰습니다. 이를 통해 통합된 스코어링 함수는 경쟁 프로그래머들보다 높은 점수를 기록하게 되었습니다.

- **Performance Highlights**: 최종적으로, 이 방법은 Hash Code의 온라인 예선 라운드에서 상위 퍼센타일에 진입하는 성과를 보였습니다. 여러 대회에서 최상위 인간 팀들을 초월하는 성능을 보여온 결과, 최근의 AtCoder 대회에서도 좋은 성과를 기록했습니다. 이 연구는 AI와 인간 협업의 새로운 가능성을 제시하고 있습니다.



### Graph Neural Networks for Heart Failure Prediction on an EHR-Based Patient Similarity Graph (https://arxiv.org/abs/2411.19742)
- **What's New**: 이 논문은 심부전을 예측하기 위해 새로운 접근 방식으로 그래프 신경망(Graph Neural Networks, GNN)과 그래프 변환기(Graph Transformer, GT)를 활용한다. 환자 유사성 그래프를 통해 다음 병원 방문 시 심부전의 발생 가능성을 예측하는 방법을 제안한다. 이러한 방법론은 기존의 데이터 처리 방식과 차별화되어 의료 데이터의 복잡한 관계를 효과적으로 분석할 수 있는 가능성을 보여준다.

- **Technical Details**: 연구는 MIMIC-III 데이터셋을 기반으로 하여, K-최근접 알고리즘(K-Nearest Neighbors, KNN)을 통해 임상 진단, 절차 및 약물 데이터를 포함하는 환자 유사성 그래프를 구성하였다. GraphSAGE, Graph Attention Network (GAT), Graph Transformer (GT)의 세 가지 모델을 적용해 심부전 발생을 예측하였다. 모델 성능은 F1 점수, AUROC, AUPRC 등의 메트릭을 통해 평가되었으며, 그 결과는 기존 알고리즘과 비교되었다.

- **Performance Highlights**: GT 모델은 F1 점수 0.5361, AUROC 0.7925, AUPRC 0.5168을 기록하며 가장 우수한 성능을 보여주었다. Random Forest는 유사한 AUPRC 값을 달성했으나, GT 모델은 그래프 구조 내의 환자 관계를 활용하여 더 나은 해석 가능성을 제공하였다. 모델의 예측 패턴에 대한 분석을 통해 다양한 분류 그룹 간의 통찰을 제공, 의료 종사자들이 얻은 유사성을 활용할 수 있도록 돕는다.



### Improving generalization of robot locomotion policies via Sharpness-Aware Reinforcement Learning (https://arxiv.org/abs/2411.19732)
Comments:
          9 pages, 6 figures

- **What's New**: 본 논문에서는 로봇 공학의 샘플 효율성과 강건성 개선을 위한 새로운 접근법을 소개합니다. Sharpness-aware optimization 기법을 통합하여 gradient 기반 강화 학습 알고리즘의 성능을 향상시키는 방법을 제안합니다. 특히, contact-rich 환경에서 정책의 강건성을 크게 향상시키면서도 샘플 효율성을 유지하는 방법을 시연합니다. 이 연구는 시뮬레이션과 실제 세계 성능 간의 격차를 줄이는데 기여할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이 논문에서는 differentiable simulators를 사용하여 강화 학습에서의 정책 최적화를 지원합니다. 특히, sharpness-aware optimization 기법을 통해 losses의 평평한 최소값을 찾아 내는 과정을 도입했습니다. 이 접근법은 기존의 first-order 방법들이 접하는 일반화 문제를 해결하며, action noise에 대한 내성을 높여 robust한 정책 학습을 가능하게 합니다. 이는 gradient 기반 정보의 활용과 zero-order 방법 간의 trade-off를 효과적으로 극복합니다.

- **Performance Highlights**: SHAC-ASAM이라는 우리의 제안된 알고리즘은 Ant 및 Humanoid 환경에서 기존 SHAC보다 정책의 강건성을 크게 향상시켰습니다. 이 방법은 샘플 효율성과 일반화의 균형을 맞춰, 실제 응용에서 유용할 수 있는 강력한 정책을 개발할 수 있도록 합니다. 실험 결과는 SHAC-ASAM이 noise가 많은 혹은 out-of-distribution 환경에서도 탁월한 성능을 발휘함을 보여 주고 있습니다.



### JetFormer: An Autoregressive Generative Model of Raw Images and Tex (https://arxiv.org/abs/2411.19722)
- **What's New**: 이번 연구에서는 JetFormer라는 새로운 autoregressive decoder-only transformer를 제안하여, 기존의 modality-specific components에 의존하지 않고 이미지를 직접 생성할 수 있게 되었습니다. 이 모델은 정상화 흐름(normalizing flow) 모델을 활용하여 이미지의 soft-token 표현을 얻고, 이를 autoregressive multimodal transformer와 함께 훈련시킵니다. 결과적으로 JetFormer는 고화질 이미지 생성 앱을 위한 강력한 가능성을 보여주며, 이전의 pretrained 이미지 autoencoder에 비해 경쟁력 있는 성능을 자랑합니다.

- **Technical Details**: JetFormer는 최신의 대화형 생성 모델을 지원하기 위해 설계되었습니다. 이 모델은 Gaussian mixture loss를 포함하여 훈련 중에 Gaussian noise의 변화를 조절함으로써 고차원 이미지를 생성하는 데 중점을 둡니다. 또한, JetFormer는 PCA 기법을 통해 자연 이미지의 중복성을 관리하는 두 가지 방법을 탐색하며, 이미지와 텍스트를 통합적으로 처리할 수 있는 구조적인 이점을 제공합니다.

- **Performance Highlights**: JetFormer는 텍스트-이미지 생성 및 비전-언어 이해 작업에서 이전의 모델들과 비교하여 뛰어난 성과를 보입니다. 특히, ImageNet의 클래스 조건 이미지 생성 및 웹 규모의 멀티모달 생성 실험에서 그 성능이 입증되었습니다. JetFormer는 통합된 아키텍처로 처음부터 끝까지 훈련할 수 있으며, 강력한 likelihood bounds를 생성할 수 있는 첫 번째 모델로 자리잡고 있습니다.



### MonoPP: Metric-Scaled Self-Supervised Monocular Depth Estimation by Planar-Parallax Geometry in Automotive Applications (https://arxiv.org/abs/2411.19717)
Comments:
          Accepted at WACV 25, project page: this https URL

- **What's New**: 이번 연구에서는 자율주행차량에서 널리 사용되는 자가 지도화(셀프 수퍼바이즈드) 모노큘러 깊이 추정 기술을 개선하여, 단순히 하나의 비디오 데이터와 카메라 장착 위치만으로 메트릭 스케일 깊이 예측을 가능하게 하는 새로운 모델을 소개합니다. 본 연구는 플래너-패럴랙스 기하학(planar-parallax geometry)을 활용하여 장면 구조를 재구성하며, 이는 최신 자동차에서 쉽게 접근할 수 있는 정보입니다.

- **Technical Details**: 제안된 시스템은 다중 프레임 네트워크, 단일 프레임 네트워크 및 포즈 네트워크의 세 가지 주요 네트워크로 구성되어 있습니다. 이 중 다중 프레임 네트워크는 순차적인 프레임을 처리하며, 카메라 장착 위치와 플래너-패럴랙스 기하학을 활용하여 정적인 장면의 구조를 추정합니다. 이는 나중에 다른 네트워크에 지식을 전달하여, 스케일 정보와 동적 물체 마스크 등을 제공합니다.

- **Performance Highlights**: 이 모델은 드라이빙 벤치마크인 KITTI에서 최첨단 결과를 달성했으며, Cityscapes 데이터세트에서도 자가 지도화된 메트릭 스케일 깊이 예측을 시연한 첫 번째 방법 중 하나로 인정받고 있습니다. 이번 연구는 카메라의 장착 위치 정보만으로 신뢰할 수 있는 메트릭 깊이 결과를 생성할 수 있음을 입증하여, 자동차 인식 작업에서 높은 적용 가능성을 보여줍니다.



### CantorNet: A Sandbox for Testing Geometrical and Topological Complexity Measures (https://arxiv.org/abs/2411.19713)
Comments:
          Accepted at the NeurIPS Workshop on Symmetry and Geometry in Neural Representations, 2024

- **What's New**: 이번 논문은 CantorNet이라는 새로운 프레임워크를 제안하여 인공지능 신경망에서 자기 유사성(self-similarity) 현상을 연구합니다. CantorNet은 19세기 조르주 칸토르(Georg Cantor)의 칸토르 집합에서 영감을 받았으며, 이 집합은 자기 유사성을 가지는 무한한 점들의 집합입니다. 이 연구는 새로운 위상(topology)적 및 기하학적 복잡성(complexity) 측정을 통해 자기 유사성 현상을 탐구하는 샌드박스 역할을 합니다.

- **Technical Details**: CantorNet은 ReLU 신경망의 특징을 기반으로 하여, 다양한 Kolmogorov 복잡성을 포함하는 결정 경계(decision boundaries)를 형성합니다. 또한, CantorNet의 결정 경계는 아날리틱적으로 알려져 있으며 스펙트럼 전반에 걸쳐 임의의 형태로 불규칙할 수 있습니다. 이 논문은 복잡성 측정을 위한 테스트 환경을 제공할 뿐만 아니라, 기하학적으로 무시된 데이터 증강(data augmentation) 기법과 적대적 공격(adversarial attacks)이 발생할 수 있는 잠재적 문제를 보여줍니다.

- **Performance Highlights**: CantorNet은 복잡한 신경망 구조를 대칭(symmetry)을 통하여 설명하고, 다양한 기하학적 특징을 고려할 수 있는 가능성을 보여줍니다. 이 연구는 신경망의 일반화 능력을 향상시키기 위해 Kolmogorov 복잡성에 대한 이해를 심화시키는 데 주목하고 있습니다. 최종적으로, CantorNet은 다양한 복잡성 측정 기법을 시험할 수 있는 자연스러운 후보로 자리잡을 것입니다.



### ChineseWebText 2.0: Large-Scale High-quality Chinese Web Text with Multi-dimensional and fine-grained information (https://arxiv.org/abs/2411.19668)
Comments:
          ChineseWebTex2.0 dataset is available at this https URL

- **What's New**: 이번 논문에서는 MDFG-tool이라는 새로운 툴 체인을 제안하여, 다차원적이고 세분화된 정보를 가진 고품질 중국어 데이터를 대규모로 구축하는 방법을 설명합니다. 이 툴 체인은 기본적으로 두 단계로 나뉘며, 첫 번째 단계에서는 수작업으로 적용된 규칙을 사용하여 원자료에서 노이즈를 제거합니다. 두 번째 단계에서는 품질 평가 모델, 도메인 분류기, 독성 평가 모델의 하위 모듈을 포함하여 깨끗한 데이터의 품질과 안전성을 평가합니다.

- **Technical Details**: MDFG-tool은 특히 BERT 기반 품질 평가 모델, FastText 도메인 분류기, 독성 평가 모델을 결합하여 각 텍스트에 대한 품질 점수와 도메인 및 독성 레이블을 부여합니다. 이러한 세 가지 유형의 세분화된 정보를 통합하여 연구자들이 특정 요구 사항에 맞는 데이터를 쉽게 선택할 수 있도록 돕습니다. 최종적으로 중국어 데이터셋인 ChineseWebText2.0을 출시하며, 데이터셋의 규모는 3.8TB로, 각 텍스트는 품질 점수와 함께 다양한 레이블과 점수를 가지고 있습니다.

- **Performance Highlights**: ChineseWebText2.0은 현재 공개된 데이터셋 중에서 가장 큰 독성 레이블이 있는 텍스트 데이터셋의 일부를 포함하고 있어, 중국어 LLM의 독성 평가 및 안전성을 크게 향상시킬 수 있습니다. 연구자들은 이 데이터셋을 활용하여 다양한 시나리오에 맞춘 고품질의 도메인 특정 LLM 개발에 기여할 수 있습니다. 이 툴 체인의 최종 목표는 LLM의 발전을 위한 필수적이고 다차원적인 데이터베이스를 제공하는 것입니다.



### Multimodal Whole Slide Foundation Model for Pathology (https://arxiv.org/abs/2411.19666)
Comments:
          The code is accessible at this https URL

- **What's New**: 최근 발전된 foundation model을 통해 컴퓨터 병리학(computational pathology)의 연구가 혁신적으로 변화하고 있습니다. 연구진은 335,645개의 whole slide image(WSI)를 사용하여 self-supervised learning(SSL)과 시각-언어 정렬(vision-language alignment)을 활용한 TITAN이라는 다중 모달 모델을 제안했습니다. TITAN은 희귀 질환 검색 및 암 예후 예측과 같은 자원이 제한된 임상 시나리오에 적합한 일반적인 슬라이드 표현(slide representations)과 병리 보고서 생성 기능을 가지고 있습니다.

- **Technical Details**: TITAN 모델은 시각 전용과 시각-언어 모델로 나뉘며, 베이스라인으로는 실험에서 선형 프로빙(linear probing), 몇 장의 샘플(few-shot) 및 제로샷(zero-shot) 분류 개념을 채택하였습니다. 기존의 병리학 리포트를 포함하여 423,122개의 합성 캡션(synthetic captions)을 생성하는 다중 모달 생성 AI 코파일럿(multimodal generative AI copilot)으로부터 학습되었습니다. 이 모델은 추가적인 미세 조정(fineting)이나 임상 레이블(clinical labels) 없이도 성능을 발휘합니다.

- **Performance Highlights**: TITAN은 다양한 임상 작업에서 평가되었으며, ROI 및 슬라이드 기반 foundation 모델보다 탁월한 성능을 보였습니다. 특히, 희귀 암 검색(rare cancer retrieval) 및 크로스 모달 검색(cross-modal retrieval) 분야에서도 뛰어난 결과를 기록하며, 병리학 보고서 생성(pathology report generation)에서도 효과적임을 입증하였습니다. 이 연구는 임상 데이터가 제한된 상황에서도 실질적인 병리학적 분석 및 보고에 기여할 수 있는 가능성을 보여줍니다.



### Uniform Attention Maps: Boosting Image Fidelity in Reconstruction and Editing (https://arxiv.org/abs/2411.19652)
Comments:
          Accepted to WACV 2025

- **What's New**: 이번 연구에서는 기존의 tuning-free 방법들이 지니는 한계점을 분석하고, 이를 극복하기 위한 새로운 접근법을 제안합니다. 특히, cross-attention 메커니즘의 문제를 해결함으로써 이미지 재구성의 신뢰도를 크게 향상시키는 uniform attention maps을 도입했습니다. 이는 텍스트 조건 하에서도 이미지 편집의 일관성과 정확도를 증가시킵니다.

- **Technical Details**: 기존의 DDIM Inversion에서 발생하는 reconstruction error는 U-Net 내의 cross-attention 메커니즘에서 비롯되는 비정렬성과 관련이 있습니다. 이를 해결하기 위해 우리는 구조적 관점에서 재구성을 분석하고, 전통적인 cross-attention을 uniform attention maps으로 대체하는 새로운 방법을 제안합니다. 이 방법은 다양한 텍스트 조건에 의해 발생하는 왜곡을 최소화하며, 영상 편집 알고리즘과 통합된 adaptive mask-guided editing 기술을 추가하여 성능을 더욱 향상시킵니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 고신뢰도 이미지 재구성에서 뛰어난 성능을 나타내며 실재 이미지 조합 및 편집 시나리오에서도 강건한 결과를 보였습니다. 기존 방법 대비 높은 픽셀 정확도와 모든 유형의 텍스트 조건에서의 신뢰성을 입증하였습니다. 이 연구는 uniform attention maps이 diffusion 기반 이미지 처리 방법의 신뢰도와 다재다능성을 높이는 잠재력을 지니고 있음을 강조합니다.



### CogACT: A Foundational Vision-Language-Action Model for Synergizing Cognition and Action in Robotic Manipulation (https://arxiv.org/abs/2411.19650)
Comments:
          Project Webpage: this https URL

- **What's New**: 본 논문에서는 VLM에서 파생된 새로운 VLA 아키텍처인 CogACT를 소개합니다. 기존의 VLA 모델은 VLM을 단순한 방식으로 행동 예측에 재사용했으나, CogACT는 VLM의 출력을 조건으로 하는 특화된 행동 모듈을 제안합니다. 이 아키텍처는 행동 신호의 특성을 효과적으로 처리하며, 분산 기반 변환기(diffusion transformer)를 사용하여 성능을 크게 향상시킵니다.

- **Technical Details**: CogACT는 인지(cognition)와 행동(action) 능력을 분리하여 설계되었습니다. 이는 인간의 뇌처럼 각기 다른 뇌 영역이 시각, 언어 및 운동 기능을 처리하는 방식에 착안했습니다. 또한, 행동 모듈을 위한 다양한 백본 아키텍처를 체계적으로 연구했으며, 특히 Diffusion transformers를 사용한 순차적 모델링이 단일 단계 행동 예측보다 뛰어난 성능을 보였습니다.

- **Performance Highlights**: 실험 결과, CogACT 모델은 기존 VLA 모델보다 뛰어난 작업 성능을 보였습니다. OpenVLA와 비교했을 때, 시뮬레이션 평가에서 35% 이상, 실제 로봇 실험에서 55% 이상의 성공률 향상을 기록했습니다. 또한, 큰 RT-2-X 모델보다도 18% 높은 절대 성공률을 실험에서 달성하며, 새로운 로봇과 작업에 대한 빠른 적응력을 보여주었습니다.



### CAdam: Confidence-Based Optimization for Online Learning (https://arxiv.org/abs/2411.19647)
- **What's New**: 이번 논문에서는 CAdam이라는 새로운 최적화 전략을 제안합니다. CAdam은 기존의 Adam 최적화 방법의 한계를 극복하며, 파라미터 업데이트 시 모멘텀과 그래디언트의 일치성을 평가하여 더 효과적으로 대처합니다. 이 방법은 분산 변화 및 노이즈에 빠르게 적응할 수 있도록 설계되었습니다. 대규모 A/B 테스트를 통해 CAdam의 효과를 입증하였으며, 시스템의 총 상품 판매량(GMV)을 현저하게 증가시켰습니다.

- **Technical Details**: CAdam은 기존 Adam의 업데이트 규칙뿐만 아니라, 모멘텀과 그래디언트의 방향 일치를 고려하는 점에서 차별화됩니다. 모멘텀과 그래디언트가 같은 방향을 가리키면 업데이트를 진행하고, 반대 방향일 경우 업데이트를 일시 보류하여 데이터 분포의 변화를 추적합니다. 이러한 접근 방식은 새로운 데이터 분포에 더 빨리 적응할 수 있는 가능성을 제공합니다. CAdam의 목표는 노이즈와 의미 있는 분포 변화의 구분을 통해 Adam의 장점을 유지하는 것입니다.

- **Performance Highlights**: CAdam은 특히 노이즈가 있는 데이터 환경에서 빠른 적응력을 보이며, 기존의 다른 최적화 방법들보다 뛰어난 성능을 발휘했습니다. 연구 결과, CAdam은 합성 및 실제 데이터 세트에서 다른 잘 알려진 최적화 기법보다 효율성 및 노이즈 강인성에서 더 나은 결과를 보여줍니다. 특히 온라인 추천 시스템의 대규모 테스트에서, CAdam은 기존 Adam보다 모델 성능을 엄청나게 향상시켜 실제 상업적 성과를 크게 증가시켰습니다.



### GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding (https://arxiv.org/abs/2411.19626)
- **What's New**: 이번 논문은 Open-Vocabulary 3D 오브젝트 어포던스(affordance) 그라우딩을 위한 새로운 프레임워크인 GREAT(GeometRy-intEntion collAboraTive inference)를 제안합니다. 이 프레임워크는 객체의 불변 지오메트리 속성과 잠재적인 상호작용 의도를 탐지하여 3D 객체의 어포던스 지식을 형성합니다. 또한 점 이미지 어포던스 데이터셋 v2(PIADv2)를 소개하며, 이는 현재 가장 큰 3D 객체 어포던스 데이터셋입니다.

- **Technical Details**: GREAT는 다중 헤드 어포던스 체인의 사고(MHACoT) 추론 전략을 기반으로 하여 미세 조정된 MLLM(Multi-modal Large Language Models)을 사용하여 잠재적인 불변 기하학과 상호작용 의도를 추론합니다. 이후 크로스 모달 적응형 융합 모듈(CMAFM)을 이용해 점 구름 특징과 이미지 특징을 통합하여 3D 객체 어포던스를 정확히 그라우딩합니다. 이를 통해 시각적 컨텐츠와 기하학 정보를 효과적으로 결합합니다.

- **Performance Highlights**: GREAT는 24개의 일반적인 어포던스와 43개의 다양한 객체 카테고리, 15,000개 이상의 상호작용 이미지를 포함한 PIADv2 데이터셋을 통해 그 효과성을 입증합니다. 다수의 실험을 통해 GREAT의 우수성을 보여주며, 이는 이전 방법론들보다 더 넓은 어포던스 범위를 제대로 그라우딩할 수 있습니다. 이러한 연구는 로봇 조작, 장면 이해 및 행동 예측 등 여러 응용 분야에서의 가능성을 넓힙니다.



### FairDD: Fair Dataset Distillation via Synchronized Matching (https://arxiv.org/abs/2411.19623)
- **What's New**: 본 논문은 Fair Dataset Distillation (FDD) 프레임워크인 FairDD를 제안합니다. 이 프레임워크는 축소된 데이터셋에서 보호 특성(Protected Attributes, PA)에 대한 공정성을 확보할 수 있도록 설계되었습니다. 또한, FairDD는 기존 데이터 증류 기법들의 아키텍처 수정 없이 다양한 DD 접근 방식에 원활하게 적용될 수 있습니다. 기존의 데이터 축소 방법들이 주로 다수 집단에 집중되는 경향을 보인 반면, FairDD는 PA 그룹에 대한 동기화를 통해 균형 잡힌 생성이 가능합니다.

- **Technical Details**: FairDD는 첫째, 표현하는 집합을 PA에 따라 분할하고, 기존의 데이터 증류 방법들이 사용하는 단일 정렬 대상을 PA별 하위 대상으로 분해합니다. 둘째, 동기화된 매칭을 통해 생성된 샘플이 특정 PA 그룹의 크기에 영향을 받지 않고 각 PA 그룹에 부여됩니다. 이러한 방식은 기존의 DD를 공정한 데이터 증류 방식으로 변환하여 PA를 불균형적으로 생성하지 않도록 돕습니다.

- **Performance Highlights**: 공식적인 분석과 광범위한 실험 결과는 FairDD가 기존의 데이터 증류 방법들에 비해 공정성을 크게 개선하는데 기여함을 보여줍니다. FairDD는 Distribution Matching (DM) 및 Gradient Matching (GM)과 같은 다양한 데이터 증류 패러다임에서 우수성을 발휘합니다. 이를 통해 본 연구는 데이터 축소에서 공정성을 포함하려는 최초의 시도로, 실제 사용에 있어 데이터셋의 불공정성을 효과적으로 완화할 수 있음을 입증했습니다.



### Solving Rubik's Cube Without Tricky Sampling (https://arxiv.org/abs/2411.19583)
- **What's New**: 이 논문에서는 전통적인 샘플링 방식을 사용하지 않고, 정책 경량화 방법을 활용하여 루빅스 큐브를 해결하는 새로운 강화학습 알고리즘을 소개합니다. 기존 연구와 달리, 체크리스트 상태로부터 시작하는 방법을 채택하는 대신 완전히 섞인 큐브 상태에서 직접 학습을 진행합니다. 이 알고리즘은 상태 간의 비용 패턴을 예측하는 신경망을 사용하여 보다 효과적으로 큐브를 해결할 수 있는 방법을 제공합니다.

- **Technical Details**: 루빅스 큐브의 상태는 특정 규칙에 따라 스티커의 위치를 인코딩한 벡터로 정의됩니다. 행동(Action)은 이 상태 벡터에 대해 정해진 규칙에 따라 특정 스티커를 재배열하는 것입니다. 이 알고리즘은 루빅스 큐브의 상태 그래프 구조를 활용하여, 목표 상태로의 경로를 찾는 문제로 프레임화하여 접근합니다. 각 상태 간의 최소 비용은 필요한 재배열의 수로 정의되며, 이를 통해 큐브 해결에 필요한 행동 시퀀스를 찾아냅니다.

- **Performance Highlights**: 테스트 결과, 50,000회의 큐브가 섞인 상태에서 99.4%의 성공률로 문제를 해결했습니다. 이는 기존의 나무 탐색(Monte Carlo Tree Search) 방식에 의존하지 않고, 오직 정책 네트워크만을 이용하여 달성한 성과입니다. 이러한 결과는 희소 보상 문제에 대한 더 넓은 응용 가능성을 나타내며, 강화학습 분야에서 중요한 진전을 보여줍니다.



### Initialization using Update Approximation is a Silver Bullet for Extremely Efficient Low-Rank Fine-Tuning (https://arxiv.org/abs/2411.19557)
Comments:
          Kaustubh Ponkshe and Raghav Singhal contributed equally to this work

- **What's New**: LoRA Silver Bullet(LoRA-SB) 메서드는 대형 언어 모델(LLM)의 파라미터 효율적인 미세 조정을 위한 새로운 접근 방식을 제안합니다. 고유한 초기화 전략을 사용하여 낮은 차원 부분 공간에서의 전체 미세 조정을 근사하는 방법입니다. 이 연구는 LoRA-XS의 아키텍처가 최적의 성능을 달성하기 위한 결정적인 조건을 제공한다고 이론적으로 입증했습니다.

- **Technical Details**: LoRA-SB는 고정된 매트릭스들 사이에 학습 가능한 (r × r) 매트릭스를 삽입하여 필요한 매개변수 수를 27~90배 줄이면서도 성능 향상을 이룹니다. 이 초기화 전략은 파라미터가 낮은 초기 그라디언트를 최적의 방식으로 근사하도록 설계되었습니다. 고차원 그라디언트 업데이트에 최적화된 스케일링을 가능하게 하여 하이퍼파라미터 조정의 필요성을 제거합니다.

- **Performance Highlights**: 수학적 추론, 일반 상식 추론, 언어 이해 과제를 포함한 4444개의 모델과 16161616개의 데이터셋을 통해 LoRA-SB는 기존 LoRA와 비교하여 성능이 뛰어난 결과를 보였습니다. 특히, 표준 LoRA보다 훨씬 적은 매개변수를 사용하면서도 개선된 성과를 나타내며, 이는 낮은 차원 부분 공간에서 전체 미세 조정을 시뮬레이션할 수 있다는 점을 보여줍니다.



### Unimib Assistant: designing a student-friendly RAG-based chatbot for all their needs (https://arxiv.org/abs/2411.19554)
Comments:
          Accepted for Italian Workshop on Artificial Intelligence for Human Machine Interaction (AIxHMI 2024), November 26, 2024, Bolzano, Italy

- **What's New**: 본 연구는 OpenAI의 커스텀 GPT 기능을 활용한 Retrieval-Augmented Generation (RAG) 시스템을 통해 ChatGPT의 행동을 전문화하는 파일럿 연구입니다. 'Unimib Assistant'라는 챗봇은 밀라노-비코카 대학교(Unimib) 학생들의 특정 요구에 대한 정보와 솔루션을 제공하는 것을 목표로 합니다. 이 챗봇은 학생들의 질문-답변 접근 방식을 통해 사용자 맞춤형 지원을 제공하도록 설계되었습니다.

- **Technical Details**: 챗봇의 초기 커스터마이징 단계 후, 여섯 명의 학생을 대상으로 질적 사용성 테스트가 진행되었습니다. 이 과정에서 챗봇의 강점과 약점을 파악하여 향후 디자인 재구성을 위한 기초 자료를 마련했습니다. 사용자가 언급한 주요 한계 중 하나는 시스템이 항상 정확한 정보를 제공하지 않으며, 관련 정보를 간혹 누락하거나 클릭할 수 없는 링크를 생성한다는 점이었습니다.

- **Performance Highlights**: 사용자들은 챗봇의 사용자 친화적인 경험과 일반적인 신뢰성을 높이 평가하였으며, 응답의 구조 및 대화체 역시 긍정적이었습니다. 그러나, 사용자 만족도는 챗봇이 정확한 정보를 일관되게 제공하지 못하는 점에서 영향을 받았습니다. 향후 다양한 사용자의 피드백을 통해 Unimib Assistant를 개선하기 위한 심층 연구 및 반복 구현을 계획하고 있습니다.



### ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration (https://arxiv.org/abs/2411.19548)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 ReconDreamer를 소개하여 자율주행 세계 모델로부터의 지식을 점진적으로 통합함으로써 드라이빙 장면 재구성을 개선하고 있습니다. ReconDreamer는 특히 대규모 기동(large maneuvers)의 효과적인 렌더링을 할 수 있는 최초의 방법으로, 최대 6미터까지의 차선 변경 등을 처리할 수 있습니다. DriveRestorer는 온라인 복구를 통해 유령 아티팩트를 완화하도록 설계되었으며, 보다 복잡한 기동을 위한 고품질 렌더링을 보장하는 점진적 데이터 업데이트 전략을 포함하고 있습니다.

- **Technical Details**: ReconDreamer는 온라인 복구 프로세스를 통해 학습 데이터를 확장하며, 각기 다른 훈련 단계에서의 렌더링 출력 샘플링을 사용하여 비디오 복원 데이터셋을 생성합니다. DriveRestorer는 훈련된 세계 모델에 의해 유령 아티팩트를 완화하도록 미세조정되며, 마스킹 전략을 통해 도전적인 구역의 복구를 강조합니다. 점진적 데이터 업데이트 전략을 통해 복잡한 기동에 대한 렌더링 품질을 유지하는 동시에 아티팩트를 단계적으로 복구합니다.

- **Performance Highlights**: 실험 결과, ReconDreamer는 NTA-IoU, NTL-IoU 및 FID에서 Street Gaussians보다 각각 24.87%, 6.72%, 29.97% 개선된 성능을 보였습니다. 또한, 사용자 연구에서 96.88%의 승률을 기록하며 DriveDreamer4D를 초월하는 성능을 입증했습니다. 이로 인해 ReconDreamer는 대규모 기동 렌더링에서 195.87%의 상대적 개선을 달성하였습니다.



### Training Agents with Weakly Supervised Feedback from Large Language Models (https://arxiv.org/abs/2411.19547)
- **What's New**: 대규모 언어 모델(LLM)을 기반으로 한 새로운 훈련 방법이 소개되었습니다. 이 방법은 전문가 경로(expert trajectories)나 명확한 피드백 없이도 비평가 LLM(critic LLM)으로부터 약한 감독 신호(weakly supervised signals)를 이용하여 에이전트를 훈련합니다. 이를 통해 에이전트는 이후 반복 과정을 통해 향상된 경로를 생성할 수 있습니다. 이 연구는 대규모 언어 모델의 범위와 유연성을 확장하는데 기여하고자 합니다.

- **Technical Details**: 에이전트는 초기 환경 교수법을 통해 경로를 생성하며, 이후 비평가 LLM이 이러한 경로를 평가하여 우수한 경로의 하위 집합을 선정합니다. 에이전트는 K개의 경로를 생성하고 이들 각각에 대해 점수를 부여받습니다. 구성된 경로는 다음 라운드의 에이전트 성능 향상을 위한 교육 데이터로 사용됩니다.

- **Performance Highlights**: API-bank 데이터셋을 통해 실시된 광범위한 테스트에서 에이전트의 성능이 지속적으로 개선되었으며, 공개 벤치마크 데이터셋에서 GPT-4에 필적하는 성능을 보여주었습니다. 이러한 결과는 적은 수의 매개변수를 가진 오픈 소스 모델을 사용했음에도 불구하고 달성되었습니다.



### SkelMamba: A State Space Model for Efficient Skeleton Action Recognition of Neurological Disorders (https://arxiv.org/abs/2411.19544)
- **What's New**: 이 논문은 해부학적으로 안내되는 새로운 state-space model (SSM) 기반의 프레임워크를 소개하고, 이는 임상 진단 및 일반적인 행동 인식 작업에서의 최첨단 성능을 향상시킵니다. 본 접근 방식은 스켈레톤 모션 분석을 공간적, 시간적, 공간-시간적 스트림으로 분해하여 고유한 움직임 특성을 효율적으로 캡처합니다. 또한, 우리의 모델은 로컬 조인트 상호작용과 여러 해부학적 신체 부위 간의 글로벌 motion 패턴을 포착할 수 있도록 설계되었습니다.

- **Technical Details**: 이 모델은 각각의 해부학적 신체 부위를 고려하여 스켈레톤 모션 데이터를 구조적으로 분해합니다. 입력 시퀀스는 공간, 시간 및 공간-시간적 분석을 위한 전문 그룹으로 채널 표현이 분할됩니다. 이를 통해 우리의 구조화된 SSM 접근 방식은 복잡한 모션 모델링을 위한 multi-directional scanning 전략을 구현하여 사용자 정의된 방법으로 각 해부학적 그룹을 동시에 분석합니다.

- **Performance Highlights**: 우리의 모델은 NTU RGB+D 및 NW-UCLA와 같은 공공 행동 인식 기준에서 현재 최첨단 방법들을 능가하며, 최대 $3.2\%$의 정확도 향상과 함께 이전의 transformer 기반 모델보다 낮은 계산 복잡도로 성과를 냈습니다. 또한, 우리는 자동화된 질병 진단 가능성을 검증하기 위해 운동 기반 환자 신경 장애 분석을 위한 새로운 의료 데이터세트를 도입했습니다.



### Deepfake Media Generation and Detection in the Generative AI Era: A Survey and Outlook (https://arxiv.org/abs/2411.19537)
- **What's New**: 이 논문은 최근의 생성 모델링(Generative Modeling) 기술 발전에 따른 딥페이크(Deepfake) 콘텐츠의 현실감이 높아지고 있으며, 그로 인해 사람들이 조작된 미디어 콘텐츠를 탐지하는 데 실패하는 경우가 증가하고 있음을 강조합니다. 또한, 본 연구는 딥페이크 생성 및 탐지 기술에 대한 포괄적인 조사 결과 및 최신 기술들, 예를 들어 확산 모델(Diffusion Models)과 Neural Radiance Fields를 소개합니다.

- **Technical Details**: 딥페이크는 이미지, 비디오, 오디오 및 멀티모달(오디오-비주얼) 콘텐츠를 포함하며, 이를 기존 사용 절차에 따라 여러 유형으로 분류합니다. 연구자들은 딥페이크 생성을 위한 방법과 이를 탐지하기 위한 방법의 분류 체계를 구축하였으며, Generative Adversarial Networks (GANs)와 CNN(Convolutional Neural Networks) 기반의 탐지 방법들이 널리 사용되고 있음을 발견했습니다.

- **Performance Highlights**: 딥페이크 탐지를 위한 데이터세트를 수집하고, 현재 사용되는 최고의 탐지기들의 성능 순위를 제공하여 비교를 용이하게 했습니다. 새로운 멀티모달 벤치마크를 통해 딥페이크 탐지기의 일반화 능력을 평가한 결과, 현재의 최첨단 탐지기들은 새로운 및 더 강력한 생성 모델들이 생성한 딥페이크 콘텐츠에 대해 일반화 능력이 떨어짐을 보여주었습니다. 이 논문은 앞으로 보다 강력한 탐지기를 개발하기 위한 연구 방향도 제안하고 있습니다.



### Quantized Delta Weight Is Safety Keeper (https://arxiv.org/abs/2411.19530)
- **What's New**: 본 논문에서는 Delta Weight Quantization이라는 새로운 부분 압축 방법이 모델 보안과 효율성을 동시에 향상시킬 수 있는 가능성을 제시합니다. 최근의 연구들은 사용자 맞춤형 언어 모델의 보안 취약점, 특히 Fine-Tuning을 통한 공격에 대한 저항성을 평가하였습니다. BitDelta와 같은 새로운 압축 기술을 통해, 모델은 성능 손실을 최소화하면서 안전성을 크게 개선할 수 있음을 발견하였습니다. 이러한 연구는 향후 다중 테넌트 서비스에서 보다 안전하고 효율적인 선택이 가능하도록 합니다.

- **Technical Details**: 제안된 압축 방법은 BitDelta의 Sign Compression과 Parameter Healing의 두 가지 주요 구성 요소로 나눌 수 있습니다. Sign Compression은 Fine-Tuning 과정에서 발생하는 델타 가중치를 스칼라 값으로 줄여서 주요 정보를 하나의 비트로 압축하는 방법입니다. 이어지는 Parameter Healing은 압축된 델타 가중치를 원래 모델의 성능을 유지하면서 최적화하는 학습 가능한 매개변수를 도입합니다. 이러한 과정은 기존 모델과의 비교를 통해 성능 저하를 최소화하면서도 압축을 가능하게 합니다.

- **Performance Highlights**: Llama-2-7b-chat 모델을 사례로 하여 연구 결과를 분석한 바, Delta Weight Quantization을 적용할 경우, 위험 최소화 측면에서 전반적으로 66.17%의 안전성 향상과 90.53%의 유해한 백도어 공격 저항력 증가가 관찰되었습니다. 이 외에도, 감정 기복이 있는 데이터에 대해 최대 10%의 성능 손실로도 여러 공격에 대한 저항력을 높일 수 있음을 입증하였습니다. 이러한 성과는 인프라 비용을 줄이면서도 모델의 안전성을 동시에 향상시키는 효과적인 수단이 될 것으로 보입니다.



### RAGDiffusion: Faithful Cloth Generation via External Knowledge Assimilation (https://arxiv.org/abs/2411.19528)
Comments:
          Project website: this https URL

- **What's New**: RAGDiffusion이라는 새로운 Retrieval-Augmented Generation (RAG) 프레임워크를 제안하여 표준 화상 의류 자산 생성을 위한 이미지의 구조적 정확성을 높이고 허위 구조를 완화합니다. 이 모델은 외부 지식과 데이터베이스에서 인지된 정보를 통합하여 생성 과정에서 구조적 모호성을 제거합니다. 또한, RAGDiffusion은 두 가지 핵심 프로세스인 구조 집합화 및 전 수준 신뢰성 의류 생성을 사용합니다.

- **Technical Details**: RAGDiffusion은 대조 학습(Contrastive Learning)과 Structure Locally Linear Embedding (SLLE)을 이용하여 전 세계적인 구조 및 공간 랜드마크를 도출합니다. 이 구조 집합화 기술은 구조적 불확실성을 최소화하며, 세 가지 수준의 정렬을 통해 의류의 구조, 패턴, 그리고 디코딩 요소에서 충실도를 보장합니다. VAE(변분 오토인코더)로부터 구조 복원 시 왜곡을 줄이는 Parameter Gradual Encoding Adaptation (PGEA) 기법도 적용됩니다.

- **Performance Highlights**: RAGDiffusion은 복잡한 현실 세계 데이터셋에서 구조적으로 정밀하고 세부사항까지 충실한 의류 자산을 생성하는 데 큰 성능 향상을 보여주었습니다. 여러 실험 결과를 통해 기존 모델 대비 우수한 성능을 입증하며, 구조적 허위 현상 문제를 해결하고 높은 사양의 생성을 위한 선도적 노력으로 자리매김했습니다. 최근의 훈련 없이도 유망한 결과를 달성한 점이 주목할 만합니다.



### DisCoRD: Discrete Tokens to Continuous Motion via Rectified Flow Decoding (https://arxiv.org/abs/2411.19527)
Comments:
          20 pages 18 figures

- **What's New**: 이번 연구에서는 DisCoRD(Discrete Tokens to Continuous Motion via Rectified Flow Decoding)라는 새로운 방법을 제안합니다. 이 방법은 이산적인 동작 토큰을 연속적인 동작으로 변환하여 부드러운 모션을 생성하는 데 중점을 두고 있습니다. DisCoRD는 iterative refinement 과정을 통해 정밀한 다이내믹스를 포착함으로써 더욱 자연스러운 모션을 제공합니다.

- **Technical Details**: DisCoRD는 score-based models를 활용하여 이산적 동작 표현을 연속적인 도메인에서 해독하는 방법입니다. 이 모델은 고속 동작을 포착할 수 있는 iterative하고 stochastic한 성격을 이용하여, 조각 조각된 디스크리트 토큰을 원시 모션 공간에서 해독하여 자연스러운 모션을 생성합니다. 이 과정에서 기존의 deterministic decoder를 대체하여 더욱 세부적인 동작을 복원할 수 있는 장점을 갖춥니다.

- **Performance Highlights**: 실험 결과, DisCoRD는 HumanML3D에서 FID가 0.032, KIT-ML에서 0.169을 달성하여 현재의 최첨단 성능을 입증했습니다. 또한, 우리의 방법은 다양한 동작 생성 프레임워크와 독립적으로 조화를 이루며, 텍스트-투-모션, 음악-투-댄스 등 다양한 태스크에서 우수한 자연스러움을 보여줍니다. 새로운 평가 메트릭인 symmetric Jerk Percentage Error (sJPE)를 도입하여 복원도와 프레임 간 노이즈를 동시에 평가할 수 있는 방안을 마련했습니다.



### Density-Calibrated Conformal Quantile Regression (https://arxiv.org/abs/2411.19523)
- **What's New**: 이 논문은 Density-Calibrated Conformal Quantile Regression (CQR-d) 방법을 소개합니다. 이 새로운 접근 방식은 예측 간격을 구성할 때 특성 공간의 다양한 불확실성에 맞춰 조정됩니다. CQR-d는 로컬 데이터 밀도를 기반으로 한 로컬 및 글로벌 conformity 점수의 가중 조합을 포함하여, 예측 간격의 품질을 향상시킵니다.

- **Technical Details**: CQR-d 방법은 기존 conformal quantile regression에 대한 최근 연구를 기반으로 하며, 로컬 조정 요인을 도입하여 지역 데이터 구조에 유연하게 적응합니다. 이는 예측 간격을 더욱 조밀하게 만들면서도 원하는 커버리지 수준을 유지하게 합니다. 또한, Vovk와 동료들(2005)의 분포에 구애받지 않는 보장과 quantile regression의 로컬 적응성을 결합합니다.

- **Performance Highlights**: 시뮬레이션 연구와 R의 이분산 데이터셋 응용을 통해 CQR-d는 표준 CQR에 비해 평균 간격 폭을 8.6% 줄이면서도 원하는 커버리지를 효과적으로 유지하는 것을 보여줍니다. 특히, 명확한 로컬 불확실성 패턴이 있는 설정에서 CQR-d의 효과가 두드러지며, 이로 인해 복잡한 이질적 데이터 환경에서 예측 작업을 위한 유용한 도구가 됩니다.



### RL-MILP Solver: A Reinforcement Learning Approach for Solving Mixed-Integer Linear Programs with Graph Neural Networks (https://arxiv.org/abs/2411.19517)
- **What's New**: 이 논문은 Mixed-Integer Linear Programming (MILP) 문제를 해결하기 위한 강화 학습( Reinforcement Learning , RL) 기반의 새로운 접근 방식을 제안합니다. 기존의 ML 기반 방법들은 주로 부분 해를 생성하고 전통적인 최적화 수단에 의존하는 반면, 제안된 방법은 MILP 인스턴스와 직접 상호작용하여 완벽한 해를 찾는 과정입니다. 특히, RL 시스템을 통해 의사 결정 변수와 제약 조건 간의 관계를 학습할 수 있도록 설계하였습니다.

- **Technical Details**: MILP는 선형 목표 함수의 최소화 또는 최대화를 목표로 하며 특정 의사 결정 변수에 대해 정수성을 요구하는 문제입니다. 제안된 RL-MILP solver는 MILP의 특수한 경우인 모든 변수가 정수인 조건에서 작동하며, 문제의 두 단계를 나누어 훈련 보상 함수를 설계합니다. 이를 위해, Transformer 인코더 기반의 그래프 신경망( Graph Neural Network , GNN)을 활용하여 결정 변수 간의 복잡한 관계를 포착합니다.

- **Performance Highlights**: 실험 결과는 제안된 모델이 소규모 데이터셋에서 최적 해를 달성하고 대규모 데이터셋에서는 약 1%의 오차로 근사 최적 해를 찾을 수 있음을 보여줍니다. 이 방법은 전체 MILP 문제를 한 번에 해결할 수 있는 초기 연구로서 중요한 진전을 제공합니다. 또한, 이 연구는 MILP 문제 해결에 있어 머신러닝을 단독으로 이용한 최초의 방법으로 자리잡을 가능성이 큽니다.



### Knowledge-Data Fusion Based Source-Free Semi-Supervised Domain Adaptation for Seizure Subtype Classification (https://arxiv.org/abs/2411.19502)
- **What's New**: 이 논문은 전자뇌파(EEG) 기반 간질 발작 아형(classification)을 위한 새로운 접근법, KDF-MutualSHOT을 제안합니다. 이 방법은 사전 훈련된 모델을 새로운 데이터셋으로 이전하는 방식으로서, 데이터 보호를 고려합니다. KDF-MutualSHOT는 전문가 지식(feature extraction)과 원시 EEG 데이터(raw EEG data)를 통합한 최초의 알고리즘으로, 발작 아형 분류의 성능을 향상시키는 데 기여합니다.

- **Technical Details**: KDF-MutualSHOT은 지식-데이터 융합(Knowledge-Data Fusion)을 기반으로 하며, 소프트 결정 트리(Soft Decision Tree, SDT)와 데이터 기반 비전 트랜스포머(Vision Transformer, ViT) 간의 상호 학습을 통해 성능을 극대화합니다. 이 방법은 제젠-셔넌 발산(Jensen-Shannon Divergence)을 이용하여 두 모델이 서로 학습할 수 있도록 합니다. MutualSHOT는 새로운 타겟 데이터셋에 적합하도록 기존 SHOT 접근 방식을 개선한 것으로, 일관성 기반의 의사 라벨 선택(pseudo-label selection) 전략을 특징으로 합니다.

- **Performance Highlights**: KDF-MutualSHOT는 TUSZ 및 CHSZ 공개 데이터세트에서 다른 감독 학습 및 소스 프리 도메인 적응(source-free domain adaptation) 방법들보다 우수한 성능을 보였습니다. 이러한 성능 증가는 발작 아형 분류에서 정확도를 높이는 데 있어 혁신적인 접근법을 제공함을 나타냅니다. 연구 결과는 KDF-MutualSHOT가 전문가 지식과 원시 데이터의 결합이 중요한 역할을 한 것을 보여줍니다.



### COLD: Causal reasOning in cLosed Daily activities (https://arxiv.org/abs/2411.19500)
Comments:
          Paper accepted at NeurIPS 2024; Total 37 Pages

- **What's New**: 이번 논문에서는 COLD(Causal reasOning in cLosed Daily activities) 프레임워크를 제안하여 대규모 언어 모델(LLMs)의 인과적 추론 능력을 평가하고자 합니다. 기존의 두 가지 접근법(실제 세계의 관계를 학습하는 방법과 상징적 표현을 사용하는 방법) 간의 격차를 메우고, 일상 활동을 기반으로 한 인과 관계를 추론하는 데 중점을 둡니다. COLD 프레임워크는 인간의 일상적인 이해를 반영하여 인과적 질의를 생성할 수 있는 기회를 제공합니다.

- **Technical Details**: COLD 프레임워크는 인간의 일상 활동에 대한 이해를 기반으로 실생활에서의 인과관계를 탐구합니다. 이 프레임워크는 총 9백만 개의 인과 질의를 생성할 수 있으며, 이는 미니 튜링 테스트에 근접하여 일상적인 과제를 평가하는 데 사용됩니다. 인과적 추론 능력을 평가하기 위해 다양한 LLM을 사용하였으며, 인과 추론이 인간에게는 매우 단순한 작업임에도 불구하고 LLMs에게는 도전적임을 발견했습니다.

- **Performance Highlights**: 실험 결과, LLMs는 트리비얼한 상황에서도 인과적 추론을 수행하는 데 어려움을 겪었습니다. LLMs의 성능은 인과 이론을 잘 이해하지 못하는 '인과적 앵무새(Causal Parrots)'로 묘사되었는데, 이는 LLM들이 주어진 패턴을 단순히 암기하여 과제를 수행한다는 의미입니다. 본 연구는 인과적 추론의 강도를 확인하기 위해 백도어 기준(backdoor criterion)을 활용하여 특정 사건 간의 인과적 강도를 평가했습니다.



### Protecting Multiple Types of Privacy Simultaneously in EEG-based Brain-Computer Interfaces (https://arxiv.org/abs/2411.19498)
- **What's New**: 본 연구에서는 뇌-컴퓨터 인터페이스(BCI)에 의한 EEG(뇌파) 데이터로부터 다양한 개인 정보를 추론할 수 있다는 심각한 개인 정보 보호 문제를 강조하고 있습니다. 특히, 사용자 신원, 성별 및 BCI 경험 등 여러 유형의 개인 정보를 동시에 보호하는 방법을 제안합니다. 연구진은 개인 정보 보호를 위해 EEG 데이터를 변환할 수 있는 섭동(p perturbation) 기법을 설계하여 원본 EEG 데이터에서 개인 정보를 숨기면서도 BCI 작업 수행 능력을 유지할 수 있음을 입증했습니다.

- **Technical Details**: 연구에서는 EEG 데이터 집합을 수집하여 개인 정보가 포함된 사실을 분석합니다. 각 개인 정보 유형에 상관없이 관련성이 높은 미세 섭동을 생성한 다음, 서로 다른 개인 정보에 해당하는 섭동들을 중첩시켜 원본 EEG 데이터를 개인 정보 보호에 적합한 형태로 변환합니다. 이 과정에서 발생하는 개인 정보 레이블을 통해, 사용자 신원이나 성별 등을 추론할 수 있는 모델도 구축되고, 처리 알고리즘이 적용됩니다.

- **Performance Highlights**: 실험 결과, 제안된 개인 정보 보호 기법을 통해 개인 신원, 성별 및 BCI 경험의 분류 정확도가 유의미하게 감소했습니다. 반면, 주요 BCI 작업의 분류 정확도는 거의 영향을 받지 않아, BCI 시스템의 성능을 유지합니다. 이는 EEG 데이터를 사용하는 다양한 응용 프로그램에서 사용자 개인 정보 보호를 위해 효과적인 접근 방식을 제시하는 것입니다.



### Interleaved-Modal Chain-of-Though (https://arxiv.org/abs/2411.19488)
- **What's New**: 이번 연구에서는 멀티모달 체인 오브 사고(Interleaved-modal Chain-of-Thought, ICoT)를 제안합니다. ICoT는 이미지와 텍스트를 결합하여 중간 추론 단계를 생성함으로써 최종 답변을 추론할 수 있도록 합니다. 이는 기존의 텍스트 전용 논리와는 달리, 이미지 정보를 포함하여 보다 정교한 연관성을 표현하는 데 초점을 맞추고 있습니다. ICoT는 VLM의 사고 과정을 인간의 사고 과정과 더욱 밀접하게 정렬시키는 혁신적인 접근법입니다.

- **Technical Details**: ICoT를 실현하는 데 있어 주목할 점은 Attention-driven Selection (ADS) 전략입니다. ADS는 VLM이 입력 이미지에서 패치를 선택하여 중간 생성 과정에 삽입함으로써 세밀한 시각적 논리를 생성하도록 합니다. 이 방법은 추가적인 매개변수를 요구하지 않으며, 클립 앤 플레이 방식으로 다양한 VLM에 쉽게 적용될 수 있습니다. ADS는 VLM의 주의 맵(attention map)을 활용하여 최적의 패치를 식별하고 선택하는 방식으로, 기존의 텍스트 전용 방법에 비해 무시할 수 있는 추론 지연(latency)을 제공합니다.

- **Performance Highlights**: 연구 결과, M3CoT, ScienceQA 및 LLaVA-W와 같은 벤치마크에서 ICoT를 통해 최대 14%의 성능 향상을 달성했습니다. 또한, 생성된 추론의 해석 가능성도 더욱 향상되었습니다. ICoT는 기존의 멀티모달 체인 오브 사고 방식과 비교할 때, 중첩된 모달 추론 프로세스의 혁신적인 기초를 제공하여 VLM의 추론 능력을 극대화하는 데에 기여합니다.



### Action Engine: An LLM-based Framework for Automatic FaaS Workflow Generation (https://arxiv.org/abs/2411.19485)
Comments:
          Accepted at Utility Cloud Computing (UCC '24) conference

- **What's New**: 이 논문에서는 클라우드 네이티브 애플리케이션 개발자들이 겪고 있는 FaaS(Functions as a Service) 기반 애플리케이션 개발의 복잡성을 감소시키기 위해, Action Engine이라는 메커니즘을 제안합니다. 이 시스템은 Tool-Augmented Large Language Models (LLMs)을 활용하여 인간 언어 쿼리를 해석하고, 자동으로 FaaS 워크플로우를 생성함으로써 전문 지식과 수작업 디자인의 필요성을 줄입니다. Action Engine은 관련 함수들을 식별하고, 데이터 의존성을 관리하여 개발자가 제공한 쿼리에 대한 처리를 자동으로 수행합니다.

- **Technical Details**: Action Engine은 자동 FaaS 워크플로우 생성을 위한 종단 간 시스템입니다. 이 시스템은 플랫폼에 구애받지 않는 DAG(Directed Acyclic Graph) 워크플로우를 생성하며, 서브 태스크 간의 데이터 상호작용의 정확성을 보장하기 위해 데이터 의존성 모듈을 통합합니다. Action Engine은 생성된 워크플로우를 특정 FaaS 플랫폼의 형식으로 컴파일하고, 워크플로우 실행기에서 실행할 수 있도록 등록합니다. 사용자는 고수준 API 엔드포인트를 통해 워크플로우를 신속하게 실행할 수 있습니다.

- **Performance Highlights**: Action Engine의 평가 결과, 개발자의 개입 없이도 최대 20% 더 높은 정확도로 워크플로우를 생성할 수 있음을 보여줍니다. 이 시스템은 클라우드에 대한 지식이 부족한 개발자도 FaaS 워크플로우 생성을 가능하게 하고, 클라우드 네이티브 애플리케이션의 개발 주기를 신속하게 단축할 수 있습니다. 따라서 Action Engine은 클라우드 네이티브 애플리케이션 개발의 민주화를 촉진할 수 있는 잠재력을 지니고 있습니다.



### FLARE: Towards Universal Dataset Purification against Backdoor Attacks (https://arxiv.org/abs/2411.19479)
Comments:
          13 pages

- **What's New**: 이번 논문에서는 Deep Neural Networks(DNNs)가 backdoor 공격에 취약하다는 사실을 강조하며, 데이터셋 정화(dataset purification)의 필요성을 제안합니다. 연구팀은 기존의 정화 방법들이 backdoor 트리거와 목표 레이블 사이의 연결이 benign한 특성보다 학습하기 더 간단하다는 암묵적인 가정에 의존하고 있다고 지적하였습니다. 그러나 이러한 가정은 모든 유형의 공격에서 항상 통용되지 않음을 보여줍니다, 특히 all-to-all(A2A) 및 untargeted(UT) 공격에서요.

- **Technical Details**: FLARE는 다양한 backdoor 공격에 대응하기 위한 보편적인 정화 방법으로, 모든 hidden layer에서 비정상적 활성화를 집계하여 클러스터링을 위한 표현을 구상합니다. FLARE는 첫 단계로 각 훈련 샘플의 포괄적인 잠재 표현을 구성하고, 두 번째 단계에서 클러스터 분석을 통해 오염된 샘플을 탐지합니다. 각 클러스터의 안정성을 평가하여 보다 안정적인 클러스터를 오염된 클러스터로 식별합니다.

- **Performance Highlights**: FLARE는 22개의 대표적인 backdoor 공격(A2O, A2A, UT 공격 등)에 대해 그 효과성을 검증하였으며, 잠재적인 적응형 공격에 대해서도 견고하게 대응할 수 있는 능력을 보였습니다. 광범위한 벤치마크 데이터셋 평가를 통해 FLARE의 우수한 성능이 입증되었습니다.



### A Simple and Provable Scaling Law for the Test-Time Compute of Large Language Models (https://arxiv.org/abs/2411.19477)
Comments:
          Work in progress

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)의 테스트 시간 계산에 대한 확증된 스케일링 법칙을 갖는 일반적인 두 단계 알고리즘을 제안합니다. 이 알고리즘은 입력 문제를 받아들이고 우선 여러 후보 솔루션을 생성한 다음, 다수결 방식의 여러 라운드에서 최고의 솔루션을 선택합니다. 즉, 이러한 방식으로 LLM의 효율성을 극대화할 수 있는 가능성을 제시하고 있습니다.

- **Technical Details**: 제안된 두 단계 알고리즘은 생성 단계와 녹아웃(tournament) 단계로 구성됩니다. 첫 번째 단계에서 N개의 후보 솔루션을 생성하고, 두 번째 단계에서는 각 후보를 K회 비교하여 이긴 후보를 선택하는 방법입니다. 이 과정은 LLM 하나로만 수행할 수 있으며, 외부 검증기나 보상 모델이 필요하지 않습니다.

- **Performance Highlights**: 실험적으로 MMLU-Pro 벤치마크와의 결과를 통해 제안된 알고리즘의 효과성과 이론적 가정이 검증되었습니다. 알고리즘의 실패 확률은 N과 K의 증가에 따라 지수적으로 감소하며, 이는 LLM 호출 수의 증가에 따라 성공 확률이 1로 증가할 수 있음을 보여줍니다.



### Effective Fine-Tuning of Vision-Language Models for Accurate Galaxy Morphology Analysis (https://arxiv.org/abs/2411.19475)
- **What's New**: 이번 논문에서는 GalaxAlign이라는 새로운 방법론을 제안하여, 자연 이미지 데이터로 사전 학습된 파운데이션 모델을 우주 이미지 작업에 효과적으로 적응시켜 높은 정확도를 달성합니다. 이 방법은 세 가지 유형의 데이터를 정렬하는 대조 학습 아키텍처를 확장하여, 기하학적 기호, 텍스트 레이블, 우주 이미지를 통합합니다. 이를 통해 고비용의 사전 훈련 없이도 효과적인 파인 튜닝이 가능해집니다.

- **Technical Details**: GalaxAlign은 CLIP 아키텍처를 기반으로 하는 삼중 모달 학습 프레임워크로, 텍스트 설명, 우주 이미지, 기하학적 기호를 결합하여 우주 형태 분석에 적합합니다. 먼저 기호와 이미지를 함께 처리하는 이미지 인코더를 사용한 후, 두 번째 단계에서는 이미지 인코더가 우주 이미지만을 인코딩하도록 조정됩니다. 이 단계에서 세 가지 인코더가 각 모달리티에 특화되며, 우주 특징을 더욱 정교하게 표현할 수 있도록 합니다.

- **Performance Highlights**: GalaxAlign은 우주 분류와 유사성 검색에서의 효율성을 입증하였으며, 대규모의 우주 데이터 세트에 대한 의존성을 줄임으로써 고효율의 모델 학습을 가능하게 하였습니다. 실험 결과 이 방법론을 적용한 모델이 높은 정확도를 기록하며, 일반적으로 낮은 성능을 보였던 기존 접근 방식과 비교하여 더 나은 결과를 나타냄을 보여주었습니다.



### Towards Understanding Retrieval Accuracy and Prompt Quality in RAG Systems (https://arxiv.org/abs/2411.19463)
- **What's New**: 이 연구는 Retrieval-Augmented Generation (RAG) 시스템의 설계를 개선하기 위한 초기 탐색 연구를 수행했습니다. 연구팀은 RAG 시스템의 성능을 결정짓는 여러 디자인 요소와 그 영향력을 탐구하였습니다. 이를 통해 RAG 시스템의 신뢰성과 정확성 향상을 위한 9가지 실행 가능한 지침을 제시하였습니다.

- **Technical Details**: 연구에서는 데이터 검색(document retrieval) 단계와 생성(generation) 단계가 포함된 RAG 시스템의 일반적인 워크플로우를 분석하였습니다. 총 200,000회 이상의 API 호출을 포함하여, 세 가지 코드 관련 데이터셋과 세 가지 질문 응답 QA 데이터셋을 활용하였습니다. 주요 고려 요소로는 Retrieval Document Type, Retrieval Recall, Document Selection, 그리고 Prompt Techniques를 설정하였습니다.

- **Performance Highlights**: 연구 결과, 불필요한 문서들이 코드 생성 과제에서 오히려 긍정적인 영향을 미치는 등 흥미로운 결과가 나타났습니다. 특히, Retrieval Recall이 높더라도 RAG 시스템이 standalone LLMs보다 성능이 떨어지는 경우가 다수 있음을 확인했습니다. 이러한 분석을 통해, 다양한 상황에서 RAG 시스템의 성능을 향상시킬 수 있는 방법을 제시하여, 개발자들이 더 나은 설계를 할 수 있도록 도와주고자 합니다.



### Look Every Frame All at Once: Video-Ma$^2$mba for Efficient Long-form Video Understanding with Multi-Axis Gradient Checkpointing (https://arxiv.org/abs/2411.19460)
Comments:
          Project page: this https URL

- **What's New**: 비디오 데이터를 처리하기 위한 새로운 아키텍처인 Video-Ma2mba가 제안되었습니다. 이 아키텍처는 기존의 Transformer 기반 모델에서 attention 메커니즘을 State Space Models (SSMs)로 대체함으로써 시간과 메모리 요구사항을 선형으로 줄일 수 있도록 설계되었습니다. 또한 Multi-Axis Gradient Checkpointing (MA-GC) 방법을 도입하여 메모리 효율성을 높였습니다.

- **Technical Details**: Video-Ma2mba 아키텍처는 Mamba-2 구조의 변형으로, 비디오 시퀀스를 효율적으로 처리하기 위해 SSM을 사용합니다. 이러한 아키텍처 변경을 통해 attention 메커니즘의 복잡성을 선형으로 낮추었으며, MA-GC는 다축 방향으로 활성화를 저장함으로써 O(S)로 메모리 사용을 줄이는 데 기여합니다. 이는 모델이 1 FPS로 2시간 이상의 비디오 시퀀스를 처리할 수 있게 합니다.

- **Performance Highlights**: 실험 분석 결과, Video-Ma2mba는 1개의 GPU에서 수백만 개의 토큰 또는 2시간 이상의 긴 비디오 시퀀스를 효율적으로 처리할 수 있음을 입증하였습니다. 이 모델은 시계열 동역학을 상세하게 포착하여 비디오 이해 작업에서의 정확성과 응답의 관련성을 크게 향상시킵니다. 이러한 성능 개선은 기존 모델들에 비해 상당한 이점을 나타냅니다.



### Beyond Surface Structure: A Causal Assessment of LLMs' Comprehension Ability (https://arxiv.org/abs/2411.19456)
Comments:
          28 pages, 14 figures, 10 tables

- **What's New**: 이번 연구는 대형 언어 모델(LLMs)이 표면 구조(surface structure)에 의존하는 것뿐만 아니라 심층 구조(deep structure)도 이해할 수 있는 능력을 평가하는 데 중점을 둡니다. 특히, causal mediation analysis를 통한 새로운 평가 방법을 제안하며, LLM의 성능을 깊이 있는 이해와 표면 인식 모두에서 분석함으로써 더 정확한 평가 기준을 마련하고자 합니다.

- **Technical Details**: 모델의 깊이 있는 이해를 확인하기 위해 저자들은 direct causal effect (DCE)와 indirect causal effect (ICE)를 정의하였고, 이 두 가지를 추정하는 것이 중요하다고 강조합니다. 두 개념의 상호 영향을 측정하는 것이 불가능하기 때문에, approximated DCE (ADCE)와 approximated ICE (AICE)와 같은 대체 지표를 개발하였습니다. ADCE는 LLM의 깊은 구조 이해 능력을 정량화하는 데 사용되며, 다양한 작업에 걸쳐 LLM의 성능을 평가할 수 있습니다.

- **Performance Highlights**: 연구 결과에 따르면, 대부분의 LLM은 정확도(prediction accuracy)가 증가함에 따라 심층 구조 이해 능력이 향상됨을 나타냅니다. 특히, 폐쇄형(closed-source) LLM은 깊은 구조에 더 의존하는 반면, 오픈 소스(open-source) LLM은 표면 구조에 더 민감하여 모델 규모가 커질수록 깊은 구조 이해로 전환되는 경향을 보입니다. 이러한 결과는 ADCE가 단순한 정확도 기준보다 더 포괄적인 LLM 성능 평가 기준이 됨을 입증합니다.



### Learning Visual Abstract Reasoning through Dual-Stream Networks (https://arxiv.org/abs/2411.19451)
Comments:
          10 pages, 6 figures

- **What's New**: 본 논문에서는 시각적 추론 작업에서의 한계를 극복하기 위해 듀얼 스트림 추론 네트워크(DRNet)를 제안합니다. 이 모델은 이미지를 효과적으로 분석하기 위해 두 개의 병렬 스트림을 사용하며, 여러 RPM 벤치마크에서 최상의 평균 성능을 달성합니다. 또한, DRNet는 다양한 배포 외 시나리오에서도 견고한 일반화 능력을 보여주었습니다.

- **Technical Details**: DRNet은 두 개의 고유한 스트림을 통해 이미지를 처리하며, CNN을 사용하여 국소(Local) 정보와 ViT를 통해 공간(Spatial) 정보에 주의합니다. 이 네트워크는 두 개의 스트림에서 추출된 고수준 특징을 결합한 뒤, 규칙 추출기를 통해 이미지 간의 관계를 유도하고 예측을 수행합니다. 이를 통해 DRNet은 비언어적 시각적 추론 문제에 효과적으로 적용될 수 있습니다.

- **Performance Highlights**: DRNet은 여러 데이터세트에서 다른 모델들보다 탁월한 성능을 보여주며, 객체 간의 규칙을 효과적으로 학습하여 시각적 추론 성능을 향상시킵니다. 시각화 결과는 학습된 규칙 표현이 규칙 카테고리에 따라 클러스터링될 수 있음을 보여주며, 이는 시각적 추상 과제를 용이하게 만듭니다. 이러한 성과는 두 개의 스트림 아키텍처가 시각적 추상 추론에서 중요한 역할을 할 수 있음을 시사합니다.



### Adaptive Interactive Segmentation for Multimodal Medical Imaging via Selection Engin (https://arxiv.org/abs/2411.19447)
- **What's New**: 본 논문은 의료 이미지 분석에서의 분할(segmentation) 문제를 해결하기 위해 새로운 전략 기반 상호작용 세분화 모델(SISeg)을 제안합니다. 이 모델은 다양한 의료 이미징 모달리티를 처리할 수 있도록 설계되었으며, 적응형 프레임 선택 엔진(AFSE)을 통합하여 효율성을 향상시킵니다. 이 접근법은 복잡한 의료 데이터를 효과적으로 처리하고 메모리 사용량을 줄이며, 분할 과정의 해석 가능성을 증가시킵니다.

- **Technical Details**: SISeg 모델은 Segment Anything Model 2(SAM2)를 기반으로 하며, 이미지 특성에 따라 적절한 프레임을 동적으로 선택하는 적응형 프레임 선택 엔진이 핵심입니다. 이 시스템은 의료 이미지에 대한 사전 지식 없이도 최적의 프롬프트 프레임을 선택하며, 영상 데이터의 시퀀스를 처리할 수 있는 능력을 갖추고 있습니다. 또한, 비지도 점수 매커니즘을 도입하여 다양한 의료 이미징 모달리티를 효과적으로 처리할 수 있습니다.

- **Performance Highlights**: SISeg 모델은 7개 대표 의료 이미징 모달리티를 포함한 10개의 데이터셋에서 진행된 광범위한 실험을 통해 그 강력한 적응성과 일반화 능력을 입증하였습니다. 이 모델은 질병 진단 및 치료 계획에서 효율성과 정확성을 향상시키며, 수작업 주석 부담을 줄이는 데 기여합니다. 모델의 성능은 특히 복잡한 의료 이미징 시나리오에서 더욱 두드러지는 것으로 나타났습니다.



### Gradient Inversion Attack on Graph Neural Networks (https://arxiv.org/abs/2411.19440)
- **What's New**: 이번 연구는 Graph Federated Learning의 맥락에서 GNN(그래프 신경망)의 취약성을 탐구한 첫 번째 작업으로, 개인 데이터가 그래디언트를 통해 유출될 수 있는지를 조사합니다. 제안된 새로운 공격 기법인 GLG(Graph Leakage from Gradients)를 통해 그래디언트에서 그래프 데이터를 복구할 수 있는지에 대한 이론적 분석이 이루어졌습니다.

- **Technical Details**: 연구에서 사용된 그래프 𝒢는 노드 집합 𝒱와 엣지 집합 ℰ로 구성됩니다. GCN(Graph Convolutional Networks) 및 GraphSAGE와 같은 두 가지 일반적인 GNN 프레임워크에 대한 분석도 포함되어 있으며, 각 프레임워크가 그래디언트 기반 공격에 얼마나 취약한지를 평가합니다. 연구는 노드 분류 및 그래프 분류 작업 모두에 적용됩니다.

- **Performance Highlights**: 실험 결과, 그래디언트에서 일부 그래프 데이터가 유출되는 것을 확인하였으며, 다양한 설정에서의 취약성을 분석하였습니다. 이 연구는 GNN의 그래디언트 기반 공격에 대한 새로운 통찰을 제공하며, 개인 데이터 보호의 필요성을 강조합니다.



### Proto Successor Measure: Representing the Space of All Possible Solutions of Reinforcement Learning (https://arxiv.org/abs/2411.19418)
Comments:
          Under submission, 23 pages

- **What's New**: 이 논문에서는 \'Proto Successor Measure\'라는 새로운 개념을 소개하여 모든 가능한 강화 학습 솔루션의 기초 집합을 제시합니다. 이를 통해 에이전트는 환경 내에서 주어진 보상 함수에 대해 최적의 정책을 찾을 수 있으며, 추가적인 환경 상호작용 없이도 성공적인 성과를 낼 수 있습니다.

- **Technical Details**: Proto Successor Measure는 상태 전이 동역학에는 의존하지만 초기 상태 분포나 보상, 정책에 영향을 받지 않는 정책 독립적인 기초 함수 집합과 편향(bias)으로 구성됩니다. 각 정책에 대해 에이전트가 방문하는 상태와 행동 분포를 통해 발견할 수 있는 제약 조건을 이용하여, 최적의 정책을 한 번에 계산할 수 있는 효율적인 알고리즘을 개발하였습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 과제에서 빠르게 최적의 정책을 유도할 수 있는 능력을 보여주었습니다. 특히, 기존의 다중 작업 RL에 비해 일반화 능력이 뛰어나며, Zero-shot 학습을 활용한 전반적인 성능 개선을 입증하였습니다.



### AMO Sampler: Enhancing Text Rendering with Overshooting (https://arxiv.org/abs/2411.19415)
Comments:
          17 pages

- **What's New**: 논문에서는 텍스트-이미지 생성에서 텍스트 렌더링 품질을 향상시키기 위한 새로운 방법론을 소개합니다. 특히, 사전 학습된 rectified flow (RF) 모델을 위한 overshooting sampler를 도입하여, 텍스트의 정확한 묘사를 가능하게 합니다. 이 과정에서 Attention Modulated Overshooting sampler (AMO)를 활용해 각 이미지 패치의 텍스트 내용에 따른 주의 점수에 기초하여 overshooting 강도를 조절합니다.

- **Technical Details**: Overshooting sampler는 이론적으로 학습된 Ordinary Differential Equation (ODE)을 과시뮨하여 노이즈를 재도입하는 방식으로 작동합니다. Langevin dynamics 항을 추가하여 오류 누적을 보정함으로써 텍스트 렌더링 품질을 높입니다. AMO는 각 이미지 패치의 text content에 대한 attention score에 따라 overshooting 강도를 적응적으로 조절하여, 더 높은 정확도의 텍스트 생성을 가능하게 합니다.

- **Performance Highlights**: AMO는 최신 RF 기반 모델인 Stable Diffusion 3 (SD3)와 Flux에서 각각 32.3% 및 35.9%의 텍스트 렌더링 정확도 향상을 달성했습니다. 이러한 성능 개선은 이미지의 전체 품질이나 추론 비용을 증가시키지 않는 상태에서 이루어졌습니다. 이는 다양한 AI 애플리케이션에서 텍스트-이미지 생성 모델의 유용성을 크게 높일 것으로 기대됩니다.



### Concept-driven Off Policy Evaluation (https://arxiv.org/abs/2411.19395)
Comments:
          37 pages, 10 figures

- **What's New**: 이 논문에서는 Off-Policy Evaluation (OPE)의 변동성을 줄이기 위해 인간이 설명 가능한 개념을 활용한 새로운 접근 방식을 제안합니다. 기존의 OPE 방법이 가지고 있는 높은 분산 문제를 해결하기 위해, 개념 기반 OPE 추정기를 도입하며, 이는 알려진 개념과 정의된 개념 모두에서 편향되지 않고도 성능을 개선할 수 있음을 보여줍니다. 또한, 실세계 응용에서 많이 발생하는 사전 정의된 개념이 부족한 경우를 고려하여, 해석 가능하고 간결하며 다양한 매개변수화된 개념을 학습할 수 있는 종단간(end-to-end) 알고리즘을 개발하였습니다.

- **Technical Details**: 연구에서는 개념 병목 모델(Concept Bottleneck Models, CBMs)을 활용하여 OPE를 수행하는 방법을 보여줍니다. 이 접근 방식은 정책의 평가를 위해 역사적 데이터에서 학습된 해석 가능한 개념을 사용하여 불확실성을 줄이는 데 중점을 둡니다. 논문에서는 이러한 개념 기반 중요도 샘플링(Importance Sampling, IS) 추정기를 도입하여 기존 IS 추정기보다 낮은 분산을 보장하는 이론적 조건을 유도하고, 알려진 및 학습된 개념 모두에 대해 기존 방법들을 초월할 수 있음을 입증하고 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 개념 기반 OPE 추정기가 기존 방법들보다 뛰어난 성능을 보여주었으며, 평가 과정에서 변동성을 줄이는 데 효과적임을 입증했습니다. 특히, 개념 기반 추정기는 대상 개념에 대한 간섭이 가능하므로, 특정 개념에서의 변동성 문제를 해결하는 데 유리합니다. 이러한 해석 가능성과 유연성 덕분에, 제안된 방법은 복잡한 문제에서도 높은 수준의 성능을 유지할 수 있으며, 이는 OPE의 품질과 신뢰성을 크게 향상시킵니다.



### Global Tensor Motion Planning (https://arxiv.org/abs/2411.19393)
Comments:
          8 pages, 4 figures

- **What's New**: 이번 논문에서는 Global Tensor Motion Planning (GTMP)라는 샘플링 기반 모션 계획 알고리즘을 소개합니다. GTMP는 텐서 연산만으로 구성되어 있으며, 효율적인 벡터화된 샘플링과 충돌 검사, 검색을 가능하게 하는 무작위 다중 그래프 구조를 도입합니다. 이 알고리즘은 기존 샘플링 방법들보다 300배 빠른 배치 계획 성능을 보여주며, 현대 GPU/TPU와의 연동도 지원합니다.

- **Technical Details**: GTMP는 차원 d의 컴팩트한 구성 공간에서 경로 계획 문제를 다룹니다. 주요 기법으로 레이어식의 분산 구조를 이용하여, 여러 계획 사례에 대해 배치 충돌 확인과 배치 가치 반복(Value Iteration) 등을 가능하게 합니다. 또한, 스플라인과 관련된 구조를 도입하여 경량화된 요구사항으로 매끄러운 계획을 구현하며, JAX를 통해 효율적인 텐서 기반 처리가 가능해집니다.

- **Performance Highlights**: 실험 결과, GTMP는 PyBullet 충돌 검사와 비교할 때 배치 계획에서 300배 더 빠른 속도를 기록하며, 스플라인 분산 구조를 통해 경로의 매끄러움과 다양성도 향상되었습니다. 이러한 결과는 GTMP가 다양한 로봇 학습 작업과 대규모 응용 프로그램에 적합한 강력하고 확장 가능한 플래너로서의 가능성을 보여줍니다.



### Zero-Forget Preservation of Semantic Communication Alignment in Distributed AI Networks (https://arxiv.org/abs/2411.19385)
- **What's New**: 이 논문은 미래의 통신 네트워크가 대규모 분산 인공지능(AI)을 연결할 것이라는 전망에서 출발합니다. AI 쌍의 정렬된 사전 지식을 활용하여 고차원 데이터 전송을 고도로 압축된 의미 통신(semantic communications, SC)으로 변환할 수 있는 가능성을 제시합니다. 특히, 기존의 도메인 적응(domain adaptation, DA) 방법에 따른 SC 정렬의 손실을 방지하기 위한 '제로 포겟(Zero-Forget) DA(ZFDA)' 프레임워크를 제안합니다.

- **Technical Details**: ZFDA 프레임워크는 AI의 SC 정렬을 보존하면서도 도메인 적응을 가능하게 합니다. 이를 위해 희소 추가 수정(sparse additive modifications, SAM)을 설계하여 AI의 신경 매개변수를 크게 변경하지 않고도 조정할 수 있도록 합니다. SAM 최적화는 이산 마스크와 연속 변수를 분리하여 수행하며, 점수 기반 최적화를 통해 마스크를 처리합니다.

- **Performance Highlights**: 이미지 전송을 위한 SC 시스템에서의 실험 평가 결과, 제안된 프레임워크는 SC 정렬을 완벽하게 보존하며, DA 성능 저하 없이도 수행되거나 일부 경우에는 성능이 개선되었습니다. 추가 메모리 사용량은 1% 미만에 불과하여, 분산 AI 네트워크에서 매끄러운 SC를 가능하게 합니다.



### Marconi: Prefix Caching for the Era of Hybrid LLMs (https://arxiv.org/abs/2411.19379)
- **What's New**: 본 논문에서는 Hybrid LLMs를 위한 효율적인 prefix caching 시스템인 Marconi를 소개합니다. Marconi는 SSM(State Space Model) 상태와 KV 캐시를 동시에 관리하여 이전 시퀀스 상태를 효과적으로 활용할 수 있도록 합니다. 이 시스템은 기존 캐싱 논리를 개선하고, SSM의 제한점을 고려하여 새로운 입회 및 퇴출 정책을 도입합니다.

- **Technical Details**: Marconi는 SSM 후보의 재사용 가능성을 평가하는 데 있어 다양한 시나리오를 고려하는 입회 전략을 채택합니다. 이 시스템은 레디스 트리를 활용하여 시퀀스 중복성을 효과적으로 캡슐화합니다. 입회 정책과 함께, Marconi는 메모리와 컴퓨트 절약의 교환을 최적화하기 위한 FLOP-aware 퇴출 정책도 도입하여 효율적 캐시 관리를 수행합니다.

- **Performance Highlights**: Marconi는 다양한 워크로드와 Hybrid 모델 아키텍처에서 캐시 적중률을 평균 4.5~34.4배 향상시킵니다. 이로 인해 P95 TTFT 지연 시간은 36.1~71.1% 절감되는 효과를 나타냅니다. 마이크로 벤치마킹 결과는 Marconi가 길어진 컨텍스트 및 SSM 레이어 비율이 높은 시나리오에서 뛰어난 성능을 보여주는 경향을 확인하였습니다.



### Libra: Leveraging Temporal Images for Biomedical Radiology Analysis (https://arxiv.org/abs/2411.19378)
- **What's New**: Libra는 의료 이미지를 이용한 방사선 보고서 생성을 위한 정확한 시간 인식을 위한 새로운 다중 모달 대규모 언어 모델(MLLM)입니다. 기존 모델들이 단일 이미지 분석에 주로 집중했음에도 불구하고, Libra는 과거 연구와 비교하여 현재 이미지를 평가하는 데 필요한 시간적 정보를 통합했습니다. 이를 통해 방사선 보고서 생성(RRG)에서 전문가급 성능을 달성하며, 임상 실습의 중대한 격차를 해결하고자 합니다.

- **Technical Details**: Libra는 RAD-DINO라는 사전 훈련된 이미지 인코더와 Meditron이라는 의료 대규모 언어 모델(LLM)을 통합하여 구성되었습니다. 이 모델은 Temporal Alignment Connector(TAC)를 사용하여 다양한 시간 지점에서의 이미지의 시간 정보를 캡처하고 통합합니다. TAC는 이미지의 고세분화 레이어별 특징을 추출하는 Layerwise Feature Extractor(LFE)와 과거의 연구로부터 얻어진 시간 참조를 통합하는 Temporal Fusion Module(TFM)로 구성됩니다.

- **Performance Highlights**: Libra는 MIMIC-CXR 데이터셋에서 방사선 보고서 생성(RRG) 작업의 최첨단 성능을 달성했습니다. 특히, RadCliQ 메트릭에서 12.9%의 개선이 이루어졌으며, 모든 어휘 메트릭에서도 이전 모델들보다 상당한 향상을 보였습니다. 이러한 성과는 Libra가 의료 이미지를 효과적으로 분석하고 해석하는 능력을 갖추고 있다는 것을 증명합니다.



### DENIAHL: In-Context Features Influence LLM Needle-In-A-Haystack Abilities (https://arxiv.org/abs/2411.19360)
- **What's New**: 이 연구는 Needle-in-a-haystack (NIAH) 테스트의 성능을 더욱 정밀하게 평가하기 위해 DENIAHL이라는 새로운 벤치마크를 개발하였습니다. DENIAHL은 데이터 크기, 패턴, 데이터 유형 같은 다양한 특성을 조절하여 LLM의 회상 능력에 미치는 영향을 분석합니다. 연구진은 LLaMA-2 7B와 GPT-3.5 모델을 활용하여 NIAH 성능의 차이를 관찰하였으며, 이를 통해 기존의 NIAH 연구를 확장하고 있습니다.

- **Technical Details**: DENIAHL은 데이터의 크기, 패턴 및 유형 등 3가지 카테고리로 나눈 과제를 포함합니다. 이 벤치마크는 키-값 형식의 데이터에서 길이를 조절하거나 패턴을 변경하여 모델의 기억 능력을 평가하였으며, 특히 데이터의 특성이 모델 성능에 미치는 영향을 분석했습니다. GPT-3.5와 LLaMA-2 7B를 비교하였고, 패턴이 깨지거나 혼합된 데이터 타입이 성능에 미치는 영향을 관찰했습니다.

- **Performance Highlights**: 연구 결과 GPT-3.5 모델은 다양한 NIAH 테스트에서 비교적 높은 성능을 보였으나, 혼합된 데이터의 경우에는 성능이 저하되는 경향을 보였습니다. LLaMA-2 7B는 대부분의 NIAH 테스트에서 보다 낮은 성능을 나타냈고, 숫자 데이터에서는 더 나은 회상 능력을 보여주었습니다. 특히, LLaMA-2 7B는 일관된 패턴보다 깨진 패턴에서 더 나은 성능을 보이며, NIAH 성능이 데이터 의존적임을 강조하고 있습니다.



### Mapping Public Perception of Artificial Intelligence: Expectations, Risk-Benefit Tradeoffs, and Value As Determinants for Societal Acceptanc (https://arxiv.org/abs/2411.19356)
- **What's New**: 이 연구에서는 독일에서 1100명의 참여자를 대상으로 인공지능(AI)에 대한 대중의 인식을 분석하였습니다. 특히 AI의 미래 가능성에 대한 71개의 진술을 평가하고, 위험과 이익 간의 균형을 고려하였습니다. 결과적으로 AI에 대한 높아진 가치 평가에도 불구하고, 높은 위험과 제한된 이익이 연결되어 있다는 것을 발견했습니다.

- **Technical Details**: 이 연구에서 96.4%의 가치 평가 변동성은 인식된 위험과 이익에 의해 설명됩니다. 독일 참여자들은 각 시나리오에 대해 예상 확률에 관계없이 위험을 높게 인식했습니다. 이로 인해 AI에 대한 공공 정보와 기술 설계에서 인공지능 활용에 대한 이해도를 높일 필요성이 강조됩니다.

- **Performance Highlights**: 이 연구는 공공이 AI의 잠재적 영향과 미래 가능성을 어떻게 평가하는지를 심도 깊게 분석하여, 정책 결정과 혁신의 경로에 영향을 미치는 개인적 요인과 대중의 우려를 드러냅니다. 이 연구 결과는 AI 개발과정에서 대중의 가치와 우려를 조율하는 데 중요한 통찰을 제공합니다.



### An Adversarial Learning Approach to Irregular Time-Series Forecasting (https://arxiv.org/abs/2411.19341)
Comments:
          Accepted to AdvML-Frontiers Workshop @ NeurIPS 2024

- **What's New**: 이 논문은 비정상적 시계열 예측에서 발생하는 두 가지 주요 과제를 다루기 위해 적대적 학습 프레임워크를 제안합니다. 나쁜 모델이 인간의 직관과 일치하지 않는 예측을 할 수 있는 문제를 해결하기 위해, 전반적인 패턴과 국부적 변화의 균형을 맞춰야 한다는 점을 강조합니다. 또한 비정상적 시계열 데이터에 적대적 학습을 적용하기 위한 첫 번째 연구로 평가됩니다.

- **Technical Details**: 비정상적 시계열은 도착 간격과 수량에서 큰 변동을 특징으로 하며, 불규칙한 패턴을 예측하는 데 어려움을 겪게 됩니다. 주로 사용하는 평가 지표인 MAPE가 이러한 패턴의 독특성을 포착하지 못하고 비현실적인 예측을 처벌하지 않는 점 등의 문제를 지적합니다. 이 연구는 예측 모델에 있어 전통적인 접근 방식을 재조명하고, 고유한 특성에 맞는 적대적 구성 요소를 설계하여 성능을 개선하는 방법을 탐구합니다.

- **Performance Highlights**: 실제 데이터 세트에 대한 연구를 통해 제안된 프레임워크가 예측 정확성을 높이는 데 효과적임을 입증하였습니다. 기존의 예측 모델에 비해 인간의 직관과 더 잘 일치하는 결과를 도출해내었으며, 새로운 질적 분석 접근 방식을 통해 평가 지표를 개선하는 방안도 제시합니다. 논문의 구현 내용은 공개되어 있으며, 더 나은 모델링 및 평가 지표 개선에 대한 통찰을 제공합니다.



### Towards a Mechanistic Explanation of Diffusion Model Generalization (https://arxiv.org/abs/2411.19339)
Comments:
          13 pages, 15 figures. Accepted to NeurIPS 2024 Workshop on Attributing Model Behavior at Scale

- **What's New**: 이번 연구에서는 로컬 노이징(denoising) 연산에 기반한 확산 모델(diffusion model)의 일반화 메커니즘을 제안합니다. 네트워크 및 경험적 노이저(denoiser)의 분석을 통해, 확산 모델에서 로컬 유도 바이어스(local inductive biases)를 식별하였으며, 이를 통해 최적의 노이저를 근사하는 방법을 보여줍니다. 또한 패치 기반의 경험적 노이저를 활용하여 확산 프로세스에서 일반화 동작을 재현할 수 있음을 입증합니다.

- **Technical Details**: 확산 모델은 데이터 분포에 가우시안 노이즈를 점진적으로 추가하는 확산 프로세스를 기반으로 하며, 이 과정에서 생성된 최적의 노이징 균형을 찾는 것이 핵심입니다. 본 연구는 네트워크 노이저의 근사 오류를 분석하여 실질적으로 유사한 오류를 보이는 여러 노이저의 특성을 발견했습니다. 주요 초점은 중간 확산 과정에서의 오류 분석을 통해 공통된 로컬 유도 바이어스를 찾아내는 것입니다.

- **Performance Highlights**: 패치 기반 노이저의 결과가 전체 노이저의 결과와 동등하게 나타나는 것을 보여주며, 네트워크 노이저가 일반화되는 설정에서도 패치 기반 노이저가 잘 근사할 수 있음을 입증합니다. 최종적으로, 패치 기반 노이저들을 조합함으로써 네트워크 노이저의 출력을 근사할 수 있다는 강력한 증거를 제공합니다. 이는 확산 모델의 일반화 메커니즘을 강력하게 뒷받침하는 결과로 해석될 수 있습니다.



### PEFT-as-an-Attack! Jailbreaking Language Models during Federated Parameter-Efficient Fine-Tuning (https://arxiv.org/abs/2411.19335)
- **What's New**: 새로운 연구는 Federated Parameter-Efficient Fine-Tuning (FedPEFT)의 보안 문제에 대한 새로운 위협인 PEFT-as-an-Attack (PaaA)를 소개합니다. 이 연구는 PEFT 기술을 악용하여 Pre-trained Language Models (PLMs)의 안전 정렬을 우회하고, 악의적인 프롬프트에 대한 해로운 콘텐츠를 생성할 수 있음을 보여줍니다.

- **Technical Details**: FedPEFT는 데이터 프라이버시를 보장하고, 비효율적인 전체 매개변수 조정의 필요성을 줄이는 것을 목표로 합니다. 그러나 PaaA 같은 새로운 위협이 존재하며, 이는 1% 미만의 매개변수만 조정하여도 약 80%의 성공률을 기록할 수 있음을 보여주었습니다. 방어 전략으로 Robust Aggregation Schemes (RAS)와 Post-PEFT Safety Alignment (PPSA)도 검토되었으나, 데이터 분포의 비동질성에서 한계를 보였습니다.

- **Performance Highlights**: 실험 결과는 PaaA에 대한 기존 방어 전략들이 효과적이지 않음을 나타냅니다. RAS는 데이터 분포가 비동일할 때 효과를 발휘하지 못하며, PPSA는 공격 성공률을 10% 미만으로 낮출 수 있지만, 타겟 작업의 정확성을 심각하게 저하합니다. 이러한 발견은 FedPEFT 시스템에서 안전과 유용성 간의 균형을 맞춘 새로운 방어 전략의 필요성을 강조합니다.



### Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation (https://arxiv.org/abs/2411.19331)
- **What's New**: 이 논문에서는 Open-Vocabulary Segmentation (OVS) 문제를 해결하기 위한 새로운 접근 방식인 Talk2DINO를 제안합니다. Talk2DINO는 DINOv2의 공간적 정확성과 CLIP의 언어 이해 능력을 결합하여 이미지와 텍스트 간의 상호작용을 향상시킵니다. 이 방법은 하부 네트워크의 미세 조정 없이 CLIP의 텍스트 임베딩을 DINOv2의 패치 수준 피쳐에 매핑하는 학습된 함수를 통해 이뤄집니다.

- **Technical Details**: Talk2DINO는 DINOv2의 자기 주의(attention) 맵을 활용하여 시각적 패치를 텍스트 임베딩과 선택적으로 정렬합니다. 학습 시, 이 메커니즘은 다양한 의미(region)의 시각적 특징을 강조하여 이미지의 세분화(segmentation) 품질을 높입니다. 데이터 수집 과정 없이 새로운 매핑 기능을 배우는 이 방식은 기존 CLIP 기반 모델의 한계를 극복하는 데 기여합니다.

- **Performance Highlights**: 실험 결과, Talk2DINO는 다양한 비지도 OVS 벤치마크에서 최첨단 성능을 달성하였으며, 텍스트와 이미지 간의 의미적 상관성을 효과적으로 증대시킵니다. 이 방법은 배경 객체를 효과적으로 분리할 수 있도록 하여 더 자연스럽고 노이즈가 적은 분할 결과를 제공합니다. 논문은 Talk2DINO가 CLIP 유사 모델의 공간적 이해 한계를 해결할 수 있는 새로운 길을 열었다고 강조합니다.



### Structured Object Language Modeling (SoLM): Native Structured Objects Generation Conforming to Complex Schemas with Self-Supervised Denoising (https://arxiv.org/abs/2411.19301)
- **What's New**: 이 논문에서는 복잡한 스키마를 따르는 구조화된 객체(structured object) 생성 문제를 다루고 있습니다. 다양한 구성요소(facets) 간의 복잡한 종속성을 고려하여, LLM(Large Language Model)을 통해 객체의 자가 일관성을 보장하며 세계 지식에 기반하여 객체를 생성하는 새로운 방법론을 제안합니다. 자가 감독 학습(self-supervised learning) 기법을 이용해 모형 훈련을 진행하며, 각 객체의 복잡성을 자연스럽게 반영할 수 있는 구조를 가지고 있습니다.

- **Technical Details**: 제안된 모델은 MPT-7B라는 사전 훈련된 트랜스포머 아키텍처를 기반으로 하며, 기존 자료에서 노이즈 제거 방식으로 훈련됩니다. 이 접근 방식은 사람의 라벨링된 데이터를 필요로 하지 않고, 기존의 노이즈가 포함된 데이터를 사용하여 자가 감독 학습을 수행합니다. 데이터셋 구성과 노이즈 함수 적용을 통해, 긴 텍스트 처리 및 생성에서 GPU 메모리 사용량을 줄일 수 있는 효율적인 알고리즘을 구현합니다.

- **Performance Highlights**: 자체적으로 훈련된 모델은 기존의 최신 LLM들과 비교할 때 동등하거나 우수한 성능을 보여주면서도, 비용 효율성이 뛰어난 것으로 나타났습니다. 또한, 기본적으로 Prompt-engineering을 요구하지 않으므로, LLM의 호출 수를 줄이고 결과의 자가 일관성을 보장할 수 있습니다. 실험 결과는 제안된 구조화된 객체 언어 모델이 경제적인 방법으로도 높은 품질의 결과를 도출할 수 있음을 보여줍니다.



### BPQP: A Differentiable Convex Optimization Framework for Efficient End-to-End Learning (https://arxiv.org/abs/2411.19285)
Comments:
          NeurIPS 2024 Spotlight

- **What's New**: 최근 데이터 기반 의사결정 과정에서, BPQP라는 새로운 differentiable convex optimization 프레임워크가 소개되었습니다. 이 프레임워크는 경량화된 backward pass로 최적화 레이어의 학습 효율성을 극대화하는 방법을 제안합니다. BPQP는 KKT 매트릭스의 구조적 특성을 활용하여, 이론적으로나 실용적으로 여러 장점을 갖추고 있습니다. 특히, 복잡한 선형 시스템을 해결하지 않고도 빠르고 정확한 그래디언트를 계산하는 방법을 제공합니다.

- **Technical Details**: BPQP는 optimization layer의 backward pass 과정을 간소화하여, Quadratic Programming (QP) 문제로 변환하는 방법을 사용합니다. 이를 통해 forward와 backward pass의 분리를 이루어내며, 기존의 효율적인 solvers를 활용할 수 있게 합니다. 이 프레임워크는 Alternating Direction Method of Multipliers (ADMM) 알고리즘을 기본으로 사용하므로, differentiability가 필요 없는 상황에서도 유연하게 작동할 수 있습니다. 이러한 방식은 계산 비용을 크게 줄이는데 기여합니다.

- **Performance Highlights**: 실험 결과, BPQP는 100차원 Linear Programming, Quadratic Programming, Second Order Cone Programming 등의 경우에 기존의 differentiable 레이어들보다 각각 13.54배, 21.02배, 1.67배 빠른 성능을 자랑합니다. 이러한 효율성 향상은 BPQP의 실제 적용 가능성을 높이며, 예를 들어 포트폴리오 최적화에 있어 Sharpe ratio를 0.65에서 1.28로 증가시키는 데 기여했습니다. BPQP는 다양한 솔버와의 호환성을 가지며, 특정 문제 구조에 맞추어 최적의 성능을 발휘할 수 있습니다.



### On-chip Hyperspectral Image Segmentation with Fully Convolutional Networks for Scene Understanding in Autonomous Driving (https://arxiv.org/abs/2411.19274)
- **What's New**: 이 연구는 운전 씬에서의 다양한 물체의 near infrared (NIR) 스펙트럴 반사율을 활용하여 ADAS의 객체 세분화(segmentation) 개선을 목표로 하며, 실시간으로 처리할 수 있는 작은 크기의 스냅샷 하이퍼스펙트럴 카메라를 사용하는 가능성을 탐구합니다. 기존의 RGB 이미지는 메타머리즘(metamerism) 문제로 인한 객체 식별의 어려움이 있으나, 하이퍼스펙트럴 이미지는 더 많은 정보를 제공하여 이러한 문제를 해결할 수 있도록 합니다.

- **Technical Details**: 연구에서는 HSI-Drive 1.1 데이터셋을 기반으로 스펙트럴 분류 알고리즘에 대한 다양한 실험을 수행하며, 표준 tiny 완전 합성곱 신경망(FCN)이 하이퍼스펙트럴 이미지 분할 성능을 어떻게 개선할 수 있는지를 분석합니다. 본 연구에서 제시된 FCN 모델들은 메모리 사용과 처리 시간을 최적화하여 효율적인 세분화 성능을 보여줍니다.

- **Performance Highlights**: 제안된 HSI 세분화 프로토타입은 다양한 임베디드 컴퓨팅 플랫폼에서 메모리 풋프린트(memory footprint), 대기 시간(latency), 전력 및 에너지 소비 측면에서 성능이 평가되었습니다. 실험 결과, 경량 FCN 모델이 고전적인 스펙트럴 분류기보다 우수한 성능을 보여주며, 실시간 ADAS 구현에 적합하다는 것을 확인하였습니다.



### Contrastive representations of high-dimensional, structured treatments (https://arxiv.org/abs/2411.19245)
- **What's New**: 이 논문에서는 기존의 인과 효과 추정 방법이 고차원 구조화된 치료 개체를 다룰 때 발생하는 편향 문제를 다루고 있습니다. 저자들은 고차원 치료를 위한 새로운 대조적 방법을 제안하여 인과적인 잠재 변수를 식별하고 비인과적인 잠재 변수를 폐기하는 방법을 제시하고 있습니다. 이러한 접근법을 통해 인과 효과의 편향 없는 추정이 가능하다는 것을 보였습니다.

- **Technical Details**: 저자들은 구조적 인과 모델(Structural Causal Model, SCM) 프레임워크를 채택하여 복잡한 치료 개체를 설명하는 잠재 변수 집합을 정의하고 있습니다. 본 논문은 인과 효과를 추정하는 과정에서 노출된 모든 혼란 변수를 관찰하더라도, 치료 개체의 고차원 구조로 인해 발생하는 편향적인 결과를 다룹니다. 저자들은 대조 학습(contrastive learning)을 통해 인과적으로 관련된 잠재 변수를 효과적으로 학습하는 방법을 제안하고 있습니다.

- **Performance Highlights**: 논문은 합성 및 실제 데이터 세트를 사용하여 제안된 방법의 성능을 검증하였으며, 기존의 고차원 치료에 대한 인과 효과 추정 방법보다 우수한 성능을 보여주었습니다. 이를 통해 고차원 치료에 대한 인과 효과 추정의 정확도를 향상시켰으며, 이러한 인과적으로 관련된 표현은 약물 발견 및 제품 추천 시스템 개선에도 활용될 수 있습니다.



### SmartLLMSentry: A Comprehensive LLM Based Smart Contract Vulnerability Detection Framework (https://arxiv.org/abs/2411.19234)
- **What's New**: 이 논문은 블록체인 네트워크에서 디지털 자산 관리를 위한 스마트 계약의 보안 강화를 위해 SmartLLMSentry라는 새로운 프레임워크를 소개합니다. 기존의 규칙 기반 프레임워크는 새로운 탐지 규칙을 효율적으로 통합하는 데 한계가 있지만, SmartLLMSentry는 대형 언어 모델(LLMs)을 활용하여 이 프로세스를 간소화합니다.

- **Technical Details**: SmartLLMSentry는 ChatGPT와 같은 대형 언어 모델을 사용 하여 스마트 계약 취약성 탐지를 향상시킵니다. 모델 훈련 및 평가를 위해 랜덤하게 선택한 다섯 가지 취약성에 대한 특화된 데이터셋을 생성하였으며, 이는 LLM이 규칙을 통합하는 데 있어 중요한 역할을 합니다.

- **Performance Highlights**: 연구 결과, 충분한 데이터가 있을 경우 정확한 일치률이 91.1%에 달한 것으로 나타났습니다. 그러나 GPT-4는 규칙 생성에서 GPT-3에 비해 성능이 감소한 것으로 보입니다. SmartLLMSentry는 LLM 기반 규칙 통합을 통해 취약성 탐지의 속도와 정확성을 크게 향상시켜 블록체인 보안을 개선할 수 있는 새로운 접근 방식을 제공함을 보여줍니다.



### Pre-Training Graph Contrastive Masked Autoencoders are Strong Distillers for EEG (https://arxiv.org/abs/2411.19230)
Comments:
          24 pages

- **What's New**: 본 논문에서는 고밀도(HD) EEG 데이터와 저밀도(LD) EEG 데이터 간의 차이를 극복하기 위해 그래프 전이 학습(Graph Transfer Learning) 및 지식 증류(Knowledge Distillation) 문제로 접근합니다. 이를 통해 EEG-DisGCMAE라는 새로운 모델을 제안하며, 자가 감독(pre-training) 기법을 통해 레이블이 없는 EEG 데이터를 효과적으로 활용하고자 합니다. 이러한 접근 방식은 고급 EEG 진단에 필수적인 고밀도 데이터로부터 얻은 지식을 저밀도 데이터에 효과적으로 전이하는 데 중점을 두고 있습니다.

- **Technical Details**: EEG 데이터를 그래프로 구성하고 그래프 신경망(Graph Neural Networks, GNN)을 활용하여 복잡한 뇌 신호의 topological features를 추출합니다. 그래프 자가 감독(pre-training) 메커니즘을 도입하여 레이블이 적은 데이터에 대한 성능을 개선하고, 그래프 대응 지식 증류(Graph Knowledge Distillation) 문제를 다루기 위해 새로운 손실 함수(Graph Topology Distillation loss function)를 도입합니다. 이러한 통합된 접근 방식은 효과적인 모델을 구축하고 다양한 EEG 분석 작업에 일반화 가능성을 높입니다.

- **Performance Highlights**: 제안된 EEG-DisGCMAE 모델은 두 개의 임상 EEG 데이터셋에서 4개의 분류 작업에 대해 뛰어난 성능을 보여주었습니다. 실험 결과, 이 방식이 현대의 방법들에 비해 상당히 높은 효율성과 정확도를 발휘함을 확인했습니다. 특히 부족한 레이블 데이터에 대한 새로운 접근 방식으로, 확장성이 뛰어난 방안을 제공하여 실험적 기반의 진단 향상에 기여합니다.



### Habit Coach: Customising RAG-based chatbots to support behavior chang (https://arxiv.org/abs/2411.19229)
Comments:
          Accepted for Italian Workshop on Artificial Intelligence for Human Machine Interaction (AIxHMI 2024), November 26, 2024, Bolzano, Italy

- **What's New**: 이번 논문은 사용자 맞춤형 상호작용을 통해 습관 변화 지원을 목적으로 하는 GPT 기반 챗봇, Habit Coach의 반복적인 개발 과정을 다룹니다. 사용자 중심 디자인 접근법을 사용하여, Retrival-Augmented Generation (RAG) 시스템을 통해 챗봇을 개발했습니다. 이 시스템은 기본 언어 모델인 GPT-4를 재훈련하지 않고도 행동 개인화를 가능하게 합니다.

- **Technical Details**: Habit Coach 챗봇은 Cognitive Behavioral Therapy (CBT) 및 서사 치료 기법을 활용하여 상호작용을 맞춤화하는 전문적 프롬프트와 문서 검색 기능을 결합하여 개발되었습니다. 초기 단계에서 챗봇은 CBT에 대한 선언적 지식을 참고서적을 통해 제공받았지만, 정적 정보를 동적이고 상황에 적합한 상호작용으로 변환하는 데 어려움을 겪었습니다. 이를 통해 선언적 지식에 의존하는 것의 한계가 드러났으며, 점차 절차적 지식으로 전환하여 상호작용 전략을 정교화했습니다.

- **Performance Highlights**: 최종 평가에서 5명의 참가자가 5일 연속으로 챗봇과 상호작용하며 개인화된 CBT 개입을 경험했습니다. Self-Report Habit Index (SRHI)를 사용하여 개입 전후의 습관 강도를 측정한 결과, 개입 후 습관 강도의 감소가 나타났습니다. 이러한 결과는 RAG 기반 시스템에서 효과적인 개인화된 행동 변화 지원을 위한 절차적 지식의 중요성을 강조합니다.



### On the Unknowable Limits to Prediction (https://arxiv.org/abs/2411.19223)
- **What's New**: 이 논문은 예측 오류의 전통적인 이분법을 비판하고 차별 속도로 제거 가능한 오류 유형에 주목합니다. 새로운 분석 프레임워크를 제안하여 epistemic (인식적) 불확실성과 aleatoric (우연적) 불확실성을 명확히 구분했습니다. 또한, 예측 가능성은 정보 집합에 의존하며 조급한 예측 불가능성에 대한 주장을 경계해야 한다고 강조합니다.

- **Technical Details**: 예측 모델링에서 연구자들은 관심 있는 현상이나 개념을 정확하게 측정하고자 하며, 이는 ytrue (진짜 목표 변수)로 나타낼 수 있습니다. 그러나 측정 오류는 주관적인 평가, 회계 부정확성, 샘플링 편향 등 여러 요인에서 발생하며, 이는 yobserved (관찰된 목표 변수)로 개념화됩니다. 본 논문에서는 이러한 측정 오류가 예측 정확도에 어떻게 영향을 미치는지를 수학적으로 분석하며 bias-variance trade-off (편향-분산 트레이드오프)와 연결합니다.

- **Performance Highlights**: 제안된 프레임워크는 예측 정확도를 높이는 데 기여할 수 있는 통찰력을 제공합니다. 특히, 각 feature (특징)의 건전성과 측정 품질을 개선할 수 있는 방법을 제시하며, 이를 통해 모델의 신뢰성을 향상시킬 수 있습니다. 논문은 epistemic 오류를 줄이고 aleatoric 오류를 이해함으로써 예측 성능을 극대화할 수 있는 방법론을 논의합니다.



### On the Ethical Considerations of Generative Agents (https://arxiv.org/abs/2411.19211)
Comments:
          Accepted (poster) to Socially Responsible Language Modelling Research (SoLaR) Workshop at NeurIPS 2024

- **What's New**: 이번 논문에서는 Generative Agents 프레임워크가 인류 행동을 모사하는 데 있어 효과적이라는 점에서 생성 에이전트의 사용과 관련된 윤리적 과제에 대해 고찰합니다. 생성 에이전트의 적용과 관련하여 중요한 위험 요소들을 논의하고, 이와 관련된 추가 연구 방향을 제안하고자 합니다. 이 연구는 생성 에이전트와 같은 도구들이 가져올 수 있는 사회적 영향과 가능성을 강조하면서도, 그 발전이 수반하는 윤리적 우려를 살펴봅니다.

- **Technical Details**: Generative Agents는 생성 언어 모델(generative language model)을 사용하여 인간 행동을 독립적이고 창의적으로 모사할 수 있는 능력을 지니고 있습니다. 에이전트는 기억(memory)과 행동 파라미터(behavioral parameters)를 유지하며, 이를 통해 목표 달성을 위한 결정을 내립니다. 이러한 모델은 사람의 행동을 실질적으로 모사할 수 있는 잠재력을 가지고 있지만, 그 과정에서 발생할 수 있는 심각한 윤리적 과제가 존재합니다.

- **Performance Highlights**: 현재의 연구는 Generative Agents의 데이터를 잘못 해석할 위험을 경고하고 있으며, 이는 잘못된 인간 행동 모델링을 초래할 수 있습니다. 과도한 의인화(anthropomorphisation)는 사용자와 AI 간의 유해한 관계를 형성할 우려가 있으며, 이는 Generative Agents의 특성과 기능에 대한 오해를 초래할 수 있습니다. 이를 완화하기 위해서는 이러한 에이전트가 본질적으로 '지능적이지 않다'는 점을 명확히 하고, 그러한 왜곡 해석의 위험성을 보다 깊이 이해하기 위한 추가 연구가 필요합니다.



### Convex Regularization and Convergence of Policy Gradient Flows under Safety Constraints (https://arxiv.org/abs/2411.19193)
Comments:
          74 pages

- **What's New**: 이 논문에서는 안전 제약(safety constraints)이 있는 무한 수명 역동적 의사 결정 과정에서의 강화 학습(reinforcement learning, RL)을 연구합니다. 특히, 연속 상태-행동 공간에서 이러한 제약을 다루기 위해 보상 및 매개변수 정규화(regularization)를 결합한 이중 정규화된 RL 프레임워크를 제시합니다. 이러한 접근법은 최근의 평균 장(field theory) 및 Wasserstein gradient flows에 대한 발전을 활용하여 정책을 무한 차원 통계 매니폴드(statistical manifold)의 요소로 모델링합니다.

- **Technical Details**: 저자들은 안전 제약 문제의 해법 가능성을 확립하고, 기울기 흐름(gradient flow)을 촉진하기 위한 매끄럽고 유계(bounded) 근사를 정의하여 충분한 정규화에서 글로벌 솔루션으로의 지수적 수렴을 입증합니다. 이 연구는 표준 엔트로피 정규화(entropy regularization)와 같은 일반 정규화 함수에 대한 조건을 제공합니다. 또한 이론적 통찰력과 수렴 보장을 통해 복잡한 고차원 의사 결정 문제에서의 안전한 RL을 위한 강력한 프레임워크를 제안합니다.

- **Performance Highlights**: 논문에서 제안한 방법은 안전 제약이 있는 RL 문제에서의 안정성 향상과 수렴 속도를 개선하는 데 기여합니다. 여기에는 실용적인 RL 애플리케이션을 위한 입자 방법(particle method) 구현이 포함됩니다. 전반적으로 이 연구는 복잡한 환경에서 안전성을 보장할 수 있는 새로운 접근 방식을 제공하여 다양한 분야에 적용 가능성을 열고 있습니다.



### SOWing Information: Cultivating Contextual Coherence with MLLMs in Image Generation (https://arxiv.org/abs/2411.19182)
Comments:
          Project page: this https URL

- **What's New**: 이 논문에서는 물리학에서의 확산 현상에서 영감을 받은 확산 생성 모델들이 이미지 합성을 위한 텍스트-비전-이미지 생성 작업(Media-Text to Image Generative modeling)에서의 정보 확산을 조정하는 방식을 제안합니다. Cyclic One-Way Diffusion (COW)와 Selective One-Way Diffusion (SOW)라는 두 가지 새로운 프레임워크를 도입해, 픽셀 수준의 조건 충실도를 달성하면서도 이미지 전반에 걸쳐 시각적 및 의미적 일관성을 유지합니다. 이러한 방법론들은 기존의 Diffusion 모델들이 가진 한계를 극복하고, 사용자 맞춤형 이미지 생성을 가능하게 합니다.

- **Technical Details**: COW 방법론은 효율적인 단방향 확산 프레임워크를 제공하여 정보 송수신의 정확성을 높이려 합니다. 여기서 MLLM(Multimodal Large Language Models)은 이미지 내 요소들의 관계를 분석하여 최적의 시각 조건을 결정하고, 동적인 주의 메커니즘(attention mechanism)을 통해 정보 확산의 방향과 강도를 조절합니다. 이를 통해 정보의 과도한 전파를 방지하고, 이미지 생성 과정에서 각 지역 간의 맥락적 관계를 유지할 수 있습니다.

- **Performance Highlights**: SOW는 600개의 512x512 이미지 그룹에 대한 1,200개의 응답을 포함한 실험에서 다른 방법들과 비교하여 조건의 일관성과 전반적인 충실도에서 지속적으로 더 높은 성과를 보였습니다. 이 방법은 5초 안에 이미지를 생성할 수 있으며, 기존의 DreamBooth와 같은 커스터마이징 방법들보다 현저히 빠른 속도를 제공합니다. 이러한 결과는 SOW가 더 유연하고 적응할 수 있는 생성 모델의 가능성을 열어준다는 것을 보여줍니다.



### HOT3D: Hand and Object Tracking in 3D from Egocentric Multi-View Videos (https://arxiv.org/abs/2411.19167)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2406.09598

- **What's New**: 새롭게 소개된 HOT3D 데이터셋은 3D에서 개인 중심의 손과 객체 추적을 위한 공개 데이터셋입니다. 이 데이터셋은 19명의 주제가 33개의 다양한 단단한 객체와 상호작용하는 모습을 담은 833분 이상의 멀티 뷰 RGB/단색 이미지 스트림을 제공합니다. HOT3D는 손과 객체의 3D 포즈, 손과 객체의 3D 모델에 대한 상세한 주석을 포함하고 있으며, 주방, 사무실, 거실 등에서의 일반적인 행동을 반영한 시나리오로 구성되어 있습니다.

- **Technical Details**: HOT3D 데이터셋은 Meta의 Project Aria와 Quest 3라는 두 개의 최신 헤드 마운트 장치를 사용하여 기록되었습니다. 이 데이터셋은 3D 포즈 정보를 얻기 위해 소형 광학 마커를 부착한 손과 객체에 대해 전문적인 모션 캡처 시스템을 이용하였으며, 모든 스트림의 이미지가 동기화되어 멀티뷰 및 시간 정보를 활용한 방법 개발이 가능합니다. 손과 객체는 UmeTrack 및 MANO 형식으로 주석처리되어 있으며, PBR 재료를 사용한 3D 메시로 나타납니다.

- **Performance Highlights**: HOT3D의 실험 결과는 멀티뷰 방법이 단일 뷰 방법보다 명확하게 더 우수하는 성능을 보였습니다. 데이터셋을 통해 평가된 여러 작업에서, 3D 손 추적, 6DoF 객체 포즈 추정, 손에 있는 미지의 객체의 3D 리프팅 작업을 포함하여 강력한 기준선이 개발되었습니다. 이렇게 개발된 강력한 기준선은 증가하는 증강/혼합 현실 작업의 필요에 부합하며, 향후 연구에 많은 기여를 할 것으로 기대됩니다.



### DESIRE: Dynamic Knowledge Consolidation for Rehearsal-Free Continual Learning (https://arxiv.org/abs/2411.19154)
- **What's New**: 최근 AI의 지속적 학습(Continual Learning) 분야에서 경량화된 모듈을 도입하는 Parameter-Efficient Fine-Tuning(PEFT) 기법이 주목받고 있습니다. 본 논문에서는 기존의 정보 유출 문제를 해결하기 위해 LoRA 기반의 반복적 재학습 없이 수행되는 DESIRE 방법을 제안합니다. 이를 통해 새로운 클래스 학습 시 추가 제약 조건을 두지 않고 기존 지식을 통합하여 성능 저하를 최소화하는 접근방식을 취합니다.

- **Technical Details**: DESIRE 방법은 두 가지 주요 포스트 프로세싱 모듈을 통해 새로운 클래스와 기존 클래스 간의 지식 통합을 달성합니다. 첫 번째 모듈인 Dynamic Representation Consolidation은 두 세트의 LoRA 파라미터를 유지하고 동적으로 특징 표현을 통합하여 성능을 개선합니다. 두 번째 모듈인 Decision Boundary Refinement는 새로운 클래스 데이터 전용 훈련 시 발생하는 분류기 편향을 수정하여 더 일반화된 결정 경계를 학습하도록 돕습니다.

- **Performance Highlights**: 다양한 데이터셋에서의 실험 결과, DESIRE 방법은 기존의 반복적 학습 방법들과 비교하여 뛰어난 성능을 보였으며, 최신의 반복적 학습 기반 방법들과도 경쟁 가능한 성능을 갱신했습니다. 이 연구는 모델의 안정성(stability)과 유연성(plasticity) 간의 최적의 균형을 가지며, 작고 통계적 정보만으로도 성능을 크게 향상시킬 수 있음을 입증했습니다.



### On Moving Object Segmentation from Monocular Video with Transformers (https://arxiv.org/abs/2411.19141)
Comments:
          WICCV2023

- **What's New**: 본 논문은 단일 이동 카메라에서의 움직이는 물체 탐지 및 분할을 위한 새로운 융합 아키텍처인 M3Former를 제안합니다. 이 아키텍처는 세분화(segmentation) 및 다중 모달 융합(multi-modal fusion)을 위한 강력한 성능을 발휘하는 Transformers를 활용합니다. 연구에서는 모노큘러 비디오에서의 움직임을 재구성하는 데 있어 2D 및 3D 움직임 표현의 중요성을 분석하고, 다양한 훈련 데이터의 필요성을 보여줍니다.

- **Technical Details**: M3Former는 Appearance와 Motion 피처를 결합한 두-stream 아키텍처를 사용하여, 서로 다른 모션 표현(Optical Flow, Scene Flow 등)을 효과적으로 분석합니다. Frozen expert models를 이용해 다양한 모션 정보를 계산하며, 간단한 데이터 증강 기법을 통해 모달 간 정렬을 개선합니다. 이렇게 만들어진 모델은 독립적으로 움직이는 객체를 탐지하고 분할하는 데 집중합니다.

- **Performance Highlights**: 실험 결과, 이 프레임워크는 Kitti와 Davis 데이터셋에서 최첨단(SotA) 성능 도달을 위한 다양한 데이터 세트의 필요성을 입증합니다. 특히, 다양한 훈련 데이터를 활용함으로써 실제 비디오에서 강력한 성능을 달성하는 것이 가능하다는 것을 보여줍니다. 전체적으로, 제안된 M3Former는 다중 모드 정보의 효율적인 융합을 통해 모션 세분화의 기준을 새롭게 설정합니다.



### Examining Multimodal Gender and Content Bias in ChatGPT-4o (https://arxiv.org/abs/2411.19140)
Comments:
          17 pages, 4 figures, 3 tables. Conference: "14th International Conference on Artificial Intelligence, Soft Computing and Applications (AIAA 2024), London, 23-24 November 2024" It will be published in the proceedings "David C. Wyld et al. (Eds): IoTE, CNDC, DSA, AIAA, NLPTA, DPPR - 2024"

- **What's New**: 이번 연구에서는 ChatGPT-4o의 다중 모달(content generation) 콘텐츠 생성에 대해 조사하였습니다. 특히 성적 콘텐츠와 누드에 대한 처리 방식에서 폭력 및 약물 관련 주제와의 현저한 불균형이 나타났습니다. 이를 통해 ChatGPT-4o가 성적 내용과 누드를 지속적으로 검열하는 반면, 폭력과 약물 사용에 대해서는 더 관대한 태도를 보이는 것을 알 수 있었습니다.

- **Technical Details**: 자세한 분석 결과, 성별에 따른 차별Bias가 두드러지며, 여성 관련 콘텐츠는 남성 관련 콘텐츠보다 더 엄격한 규제를 받는 것으로 나타났습니다. 이러한 불균형은 과거 AI 논란에 대한 미디어의 감시와 공공의 반발이 영향을 미쳤을 것으로 추측됩니다. 따라서 기술 기업들은 자사의 평판을 보호하기 위해 민감한 이슈에 대한 엄격한 가이드라인을 설정하게 되었습니다.

- **Performance Highlights**: 이번 연구는 AI 주도 언어 및 다중 모달 모델에서의 편향Bias를 이해하는 데 기여하고, 보다 균형 잡히고 윤리적인 콘텐츠 조정Practices의 필요성을 강조합니다. AI 시스템이 단순한 정치적 올바름을 초월하여 진정으로 윤리적 기준과 책임을 들 upheld해야 한다고 주장합니다.



### Visual SLAMMOT Considering Multiple Motion Models (https://arxiv.org/abs/2411.19134)
- **What's New**: 본 논문에서는 SLAM(동시 위치 인식 및 지도 작성)과 MOT(다중 객체 추적)를 통합하여 실시간 데이터 처리의 필요성을 강조합니다. 전통적인 방식에서는 이 두 작업이 독립적으로 실행되어 서로의 성능을 제한하는 경우가 많았으나, 우리는 이를 결합한 새로운 접근 방식을 제안합니다. 이 논문에서는 LiDAR 기반 시스템에서 다중 운동 모델을 고려한 SLAMMOT의 가능성을 논의하며, 이를 비주얼 SLAMMOT로 확장하는 방법을 제안합니다.

- **Technical Details**: 비주얼 SLAMMOT는 센서의 데이터 수집, 전방 오도메트리, 후방 최적화 및 맵 작성 등을 포함한 여러 모듈로 구성됩니다. 논문에서는 기존의 SLAMMOT 방식을 기반으로 여러 운동 모델을 통합하여 SLAMMOT의 효율성을 개선하는 새로운 방법론을 소개합니다. 이 접근법은 LiDAR와 비전 기반 센서 간의 간극을 메우는 것을 목표로 하며, 복잡한 환경에서도 정확한 객체 추적 및 위치 추정을 가능하게 합니다.

- **Performance Highlights**: 실험 결과는 제안된 비주얼 SLAMMOT가 기존 SLAM 및 MOT 방법보다 더 나은 성능을 보여주며, 동적인 환경에서의 경쟁력을 입증했습니다. 또한, 여러 운동 모델을 고려함으로써 실제 주행 환경에서 객체 상태 추정의 정확도를 향상시킬 수 있음을 보여줍니다. 이 연구는 자율 주행 차량 시스템의 실시간 작동 능력을 크게 향상시킬 것으로 기대됩니다.



### MSG score: A Comprehensive Evaluation for Multi-Scene Video Generation (https://arxiv.org/abs/2411.19121)
- **What's New**: 이번 논문은 연속적인 시나리오에 기반한 다중 장면 비디오 생성을 위한 평가 지표를 다룹니다. 전통적인 짧은 비디오 생성과는 달리, 시나리오 기반 비디오는 캐릭터 일관성, 예술적 일관성, 심미적 품질 등을 고려해야 합니다. 저자들은 자동화된 점수 기반 평가 기준을 제안하여 더 객관적이고 효율적인 평가를 가능하게 합니다.

- **Technical Details**: MSG는 즉각적인 이웃 프레임에 대한 양방향 프레임 참조(Backward and Forward Frame Reference, BFFR)와 이전 장면의 주요 프레임을 참조하는 후향 장면 참조(Backward Scene Reference, BSR)로 구성됩니다. BFFR은 이전 및 이후 프레임을 고려하여 공간적 세부사항을 향상시키고 단기적인 시간 일관성을 유지합니다. BSR은 키프레임을 통해 이전 장면의 문맥을 유지하며 원활한 전환을 보장합니다.

- **Performance Highlights**: MSG는 Vid4와 REDS와 같은 벤치마크 데이터셋에서 평가되었습니다. 평가 지표로는 Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), 시간 일관성 측정이 포함됩니다. 그러나 실험 결과는 실패했으며, 현재 모델은 예상한 대로 작동하지 않는 상황입니다.



### PREBA: A Hardware/Software Co-Design for Multi-Instance GPU based AI Inference Servers (https://arxiv.org/abs/2411.19114)
- **What's New**: 본 연구에서는 NVIDIA의 Multi-Instance GPU (MIG) 기능을 활용하여 하나의 대형 GPU를 여러 개의 소형 GPU로 분할하는 방식에 대해 소개합니다. AI 추론을 위한 서버 설계에 있어 MIG의 효과를 평가하며, 데이터 전처리 단계에서 발생하는 성능 병목 현상을 해결하기 위한 PREBA라는 하드웨어/소프트웨어 공동 설계를 제안합니다. 내장된 FPGA 기반의 데이터 전처리 가속기를 사용하여 MIG의 잠재력을 극대화하며 동적 배칭 시스템을 통해 성능을 향상합니다.

- **Technical Details**: PREBA 시스템의 주요 구성 요소는 두 가지로, 하나는 MIG를 위한 데이터 처리 유닛(Data Processing Unit, DPU)으로, 반도체로 구성된 FPGA를 사용하여 중요한 전처리 작업을 오프로드합니다. 두 번째는 MIG의 특성을 고려하여 효율적인 배칭 알고리즘을 설계한 동적 배칭 시스템으로, 입력 데이터의 최대 배치 크기와 대기 시간 지연을 최적화합니다. 이러한 설정을 통해 AI 모델의 성능을 최대한 이용할 수 있습니다.

- **Performance Highlights**: PREBA는 실제 시스템에서 구현되어, 스루풋(throughput)은 3.7배 향상되었고, 지연(latency)은 3.4배 감소했습니다. 또한, 에너지 효율성은 3.5배 향상되었고 비용 효율성 또한 3.0배 개선되었습니다. 이러한 성과는 MIG 기반 인퍼런스 서버에서 데이터 전처리 병목 문제 및 배칭 시스템의 필요성을 처음으로 규명하고 탐색한 결과물입니다.



### Beautimeter: Harnessing GPT for Assessing Architectural and Urban Beauty based on the 15 Properties of Living Structur (https://arxiv.org/abs/2411.19094)
Comments:
          11 pages, 6 figure, and two tables

- **What's New**: Beautimeter는 GPT(Generative Pre-trained Transformer) 기술을 활용하여 건축 및 도시의 아름다움을 평가하는 새로운 도구입니다. 이 도구는 Christopher Alexander의 중심 이론을 바탕으로 하여 환경이 내재하는 생명 감각을 평가합니다. Beautimeter는 15가지 기본 속성을 분석의 기초로 삼아 환경의 아름다움을 정교하게 평가합니다.

- **Technical Details**: Beautimeter는 자연어 처리(NLP)에서 GPT의 고급 기능을 통합하여 구조들이 이러한 15가지 속성을 얼마나 잘 구현하고 있는지를 평가합니다. 이러한 속성에는 수준의 스케일(scale)과 두꺼운 경계(thick boundaries) 등이 포함됩니다. 사용자는 ChatGPT를 통해 공간의 인지된 아름다움과 일관성에 대한 통찰을 생성할 수 있습니다.

- **Performance Highlights**: 다양한 맥락에서 미적 품질을 분석하는 데 Beautimeter의 효과를 입증하는 일련의 사례 연구를 수행했습니다. 연구 결과는 Beautimeter가 건축가, 도시 계획가, 디자이너가 사람들과 깊이 공명하는 공간을 창출하는 데 유용한 도구임을 시사합니다. 본 논문은 이러한 기술이 건축 및 도시 디자인에 미치는 영향을 탐구하였으며, 설계 과정과 구축 환경 평가를 향상시킬 수 있는 잠재력을 강조합니다.



### ObjectRelator: Enabling Cross-View Object Relation Understanding in Ego-Centric and Exo-Centric Videos (https://arxiv.org/abs/2411.19083)
- **What's New**: 이 논문에서는 Ego-Exo Object Correspondence 작업을 중심으로, Ego-centric(1인칭) 및 Exo-centric(3인칭) 영상 간 객체 매핑을 수행하는 새로운 방법인 ObjectRelator를 제안합니다. MCFuse와 XObjAlign이라는 두 가지 혁신적인 모듈이 포함되어 있으며, 이들을 통해 언어 및 시각 정보를 결합하고 서로 다른 관점 간 객체 표현의 일관성을 유지할 수 있습니다.

- **Technical Details**: 제안된 MCFuse 모듈은 시각-언어 모델에서 언어 정보를 활용하여 대상 객체의 위치 확인 정확도를 향상시킵니다. XObjAlign 모듈은 자기 지도 학습(self-supervised learning) 전략을 사용하여 쌍으로 된 ego와 exo 객체 마스크를 정렬하고, 이를 통해 서로 다른 관점에서 객체를 보다 잘 인식할 수 있도록 합니다.

- **Performance Highlights**: ObjectRelator는 Ego2Exo 및 Exo2Ego 작업에서 최첨단 성능을 달성하며, 기본 PSALM보다 10% 이상 향상된 결과를 달성했습니다. 연구 결과는 다중 모드 가이딩(multi-modal guidance)과 관점 간 정렬(cross-view alignment)의 잠재력을 강조하며, 향후 연구를 위한 기초를 제공합니다.



### LADDER: Multi-objective Backdoor Attack via Evolutionary Algorithm (https://arxiv.org/abs/2411.19075)
- **What's New**: 이 논문은 진화 알고리즘을 이용한 다중 목표의 블랙박스 백도어 공격, 즉 LADDER를 제안합니다. LADDER는 피해자 모델에 대한 사전 지식 없이도 여러 공격 목표를 동시에 최적화할 수 있는 첫 번째 방법으로, 기존의 단일 목표 최적화 접근법과는 차별화됩니다. 이를 통해 새로운 이중 영역 관점에서 트리거의 은폐성을 향상시키고, 이미지 전처리 작업에 대한 강건성을 증가시킵니다.

- **Technical Details**: LADDER는 다중 목표 최적화 문제(MOP)로 구성되어 있으며 다중 목표 진화 알고리즘(MOEA)을 통해 해결됩니다. MOEA는 공격 목표 간의 트레이드오프를 유지하며 비지배 정렬(non-dominated sort) 기법을 사용해 최적의 솔루션으로 유도합니다. 본 연구는 저주파 영역에서 트리거를 최적화하고, 공격 효과성, 기능 보존, 이중 영역 은폐성 및 강건성을 동시에 유지할 수 있도록 합니다.

- **Performance Highlights**: 엄청난 실험 결과에 따르면, LADDER는 최소 99%의 공격 효과성을 달성하였고, 이미지 전처리 작업에 대한 90.23%의 공격 강건성을 보였습니다. 자연적인 은폐성 측면에서는 1.12배에서 196.74배 향상되었고, 스펙트럴 은폐성 또한 8.45배 개선되었습니다. 이러한 결과는 5개의 공개 데이터셋을 통해 측정되었으며, 현재의 은폐성 공격 대비 뛰어난 성능을 입증하였습니다.



### Way to Specialist: Closing Loop Between Specialized LLM and Evolving Domain Knowledge Graph (https://arxiv.org/abs/2411.19064)
Comments:
          Accepted by KDD 2025

- **What's New**: 이 논문에서는 기존의 일반화된 대형 언어 모델(LLMs)의 한계를 극복하기 위해 Way-to-Specialist (WTS) 프레임워크를 제안합니다. WTS는 검색 보강 생성(retrieval-augmented generation)과 지식 그래프(knowledge graphs)를 결합하여 전문화된 지식 없이도 LLM의 전문 능력을 향상시킵니다. 이 프레임워크는 특정 도메인 지식 그래프(Domain Knowledge Graph, DKG)와의 양방향 상호작용을 통해 LLM의 추론 능력을 향상시키는 새로운 접근 방식을 제공합니다.

- **Technical Details**: WTS 프레임워크는 두 가지 밀접하게 연결된 구성 요소로 이루어져 있습니다: DKG-Augmented LLM과 LLM-Assisted DKG Evolution입니다. 첫 번째 구성 요소는 DKG에서 질문 관련 도메인 지식을 검색하고 이를 통해 LLM의 추론 능력을 향상시킵니다. 두 번째 구성 요소는 처리된 작업에서 새로운 도메인 지식을 생성하고 이를 DKG를 발전시키는 데 활용합니다.

- **Performance Highlights**: WTS의 성능은 5개 도메인에 걸쳐 6개의 데이터 세트에서 검증되었습니다. 실험 결과, WTS는 4개의 전문 도메인에서 기존의 최고 성능(SOTA)을 초과했으며, 최대 성능 향상은 11.3%에 달합니다.



### Using a Feedback Loop for LLM-based Infrastructure as Code Generation (https://arxiv.org/abs/2411.19043)
Comments:
          4 pages, submitted to accepted by International Journal of Secondary Computing and Applications Research

- **What's New**: 이번 연구에서는 Large Language Models (LLMs)을 활용하여 Infrastructure as Code (IaC) 패러다임으로 인프라를 생성하는 능력에 대해 조사했습니다. 특히 cfn-lint라는 도구를 사용하여 생성된 CloudFormation 코드의 오류와 경고를 LLM이 피드백으로 수신하고 이를 수정하도록 하는 피드백 루프 시스템을 설계했습니다. 연구 결과, LLM의 피드백 루프 효과가 일정 시점 이후로는 급격히 감소하며 비효율적이게 됨을 발견했습니다.

- **Technical Details**: IaC는 클라우드 인프라 관리를 자동화하고 일관성을 보장하는 방식으로, LLM은 코드 생성에 도움을 줍니다. 하지만 IaC 코드의 정확한 작성을 어렵게 만드는 여러 암묵적인 규칙들이 존재합니다. 본 연구에서는 AWS CloudFormation 코드를 생성하기 위한 LLM의 능력을 분석하고, 생성된 코드를 cfn-lint를 통해 검증하는 과정을 설정했습니다.

- **Performance Highlights**: 실험 결과, LLM이 공략한 오류 및 경고의 총합이 다섯 번째 반복 이후로는 일정한 수치로 정체되는 경향을 보였습니다. 이는 LLM이 특정 오류를 정확히 수정하지 못해 이전 오류로 인해 발생한 여러 새로운 오류들이 쌓이는 효율이 저하된 결과로 나타났습니다. 이에 따라 LLM의 IaC 생성에서의 잠재적 한계가 드러났습니다.



### Enhancing Neural Network Robustness Against Fault Injection Through Non-linear Weight Transformations (https://arxiv.org/abs/2411.19027)
Comments:
          5 pages, 6 figures

- **What's New**: 이번 연구는 DNN(Deep Neural Networks)에서 고장으로 인한 성능 저하 문제를 해결하기 위해 새로운 접근법을 제시합니다. 기존 연구에서는 활성화 함수의 범위를 제한하는 방법을 사용했다면, 본 연구는 DNN의 가중치(weight)를 제약하는 포화 활성화 함수(Saturated Activation Functions, SAF)를 사용하여 과도한 가중치 증가를 방지하고 있습니다. 이러한 SAF 적용 방법은 DNN의 내구성을 향상시킬 뿐만 아니라 성능도 소폭 개선하는 결과를 보여줍니다.

- **Technical Details**: 이 연구에서는 Tanh, Arctan 등 여러 SAF를 활용하여 DNN의 가중치를 조정합니다. 훈련 과정에서 SAF가 적용된 가중치로 네트워크를 학습하고, 실제 환경에서 고장이 발생할 수 있는 매체에 SAF가 적용되지 않은 가중치를 기록합니다. 추후 추론 시점에서, 고장을 포함한 가중치에 SAF를 적용하여 오류를 억제하는 방식입니다.

- **Performance Highlights**: CIFAR10, CIFAR100, ImageNet 2012 데이터셋에서 제안하는 방법을 검증하였으며, FP32 ResNet18 모델이 비트 오류율(bit-error rate, BER) 0.00001에서도 경미한 정확도 손실로 작동할 수 있음을 보여줍니다. SAF를 적용하지 않은 경우, 모델은 임의의 출력을 생성하는 반면, 본 방법을 사용하면 더욱 안정적인 결과를 도출합니다.



### A Unified Platform for At-Home Post-Stroke Rehabilitation Enabled by Wearable Technologies and Artificial Intelligenc (https://arxiv.org/abs/2411.19000)
Comments:
          5 figures, 35 references

- **What's New**: 이번 논문에서는 뇌졸중 이후 환자를 위한 홈 재활 솔루션의 필요성을 제시합니다. 기존의 임상 환경을 넘어서는 지속적이고 개인화된 케어의 한계를 극복하기 위해, 웨어러블 센서와 주변 모니터링 기능을 통합한 스마트 홈 플랫폼을 소개합니다. 이 플랫폼은 다양한 재활 요구를 지원하며, 긴급한 개입을 제공하는 LLM(large language model) 기반의 지원 시스템을 포함하고 있습니다.

- **Technical Details**: 스마트 홈 플랫폼은 기계 학습(machine learning)을 활용한 plantar pressure arrays를 통해 운동 회복 평가를 수행하고, 이는 94%의 분류 정확도를 자랑합니다. 또한, 웨어러블 안구 추적 모듈과 주변 센서들을 이용하여 인지 평가 및 스마트 홈 제어를 수행하며, 이 센서의 작동 성공률은 100%, 지연 시간은 <1초입니다. LLM 기반의 에이전트인 Auto-Care는 건강 알림 및 환경 조절과 같은 실시간 개입을 제공하여 사용자에게 도움을 줍니다.

- **Performance Highlights**: 이 시스템은 사용자 만족도를 29% 향상시킵니다. 개인화된 장기 재활 관리를 위한 통합 플랫폼을 구축하여 만성 질환 관리 및 노인 인구 지원의 새로운 가능성을 열어줍니다. 지속적인 건강 모니터링을 통해 재활 과정에서의 효과성을 높이고, 사용자에게 맞춤형 솔루션을 제공합니다.



### GRU-PFG: Extract Inter-Stock Correlation from Stock Factors with Graph Neural Network (https://arxiv.org/abs/2411.18997)
Comments:
          17pages

- **What's New**: 이 논문에서는 기존 주식 예측 모델의 한계를 극복하기 위해 GRU-PFG(프로젝트 요인을 그래프에) 모델을 제안합니다. 이 모델은 오직 주식 요인만 입력으로 사용하고, 그래프 신경망을 통해 주식 간 상관관계를 추출하여 예측 성능을 향상합니다. GRU-PFG는 기존 주식 요인 모델보다 우수한 성능을 보이며, 더 복잡한 정보 기반 모델과도 유사한 예측 결과를 달성합니다.

- **Technical Details**: GRU-PFG 모델은 Gated Recurrent Unit(GRU)과 Graph Neural Networks(GNN)를 결합하여, 주식 요인을 그래프 네트워크에 프로젝션하여 주식 간의 관계를 학습합니다. 이 모델은 전통적인 MLP 및 LSTM 모델보다 고차원적인 요인 정보 추출을 가능하게 하며, 주식 관계를 파악할 수 있는 강력한 방법을 제공합니다. 특히, 정보의 불균형 문제를 해결하고 예측 성능을 향상시키기 위해 주식 요인만을 기반으로 합니다.

- **Performance Highlights**: 실험 결과 GRU-PFG 모델은 CSI300 데이터셋에서 0.134의 정보 수정을 기록하며, HIST의 0.131을 초과하고 GRU 및 Transformer 모델에 비해 월등한 성능을 보여줍니다. 이 모델은 과거 산업 관계에 의존하지 않으면서도 일반화 가능성이 높아, 반복적으로 변화하는 시장 환경에서 더욱 유리할 것으로 보입니다. 고유한 주식 요인만으로 입증된 이러한 성과는 다른 모델과의 차별성을 잘 나타냅니다.



### Harden Deep Neural Networks Against Fault Injections Through Weight Scaling (https://arxiv.org/abs/2411.18993)
Comments:
          6 pages, 8 figures

- **What's New**: 최근의 연구에서 딥 뉴럴 네트워크(DNN)가 이미지 인식, 객체 감지 및 자연어 처리와 같은 다양한 스마트 응용 프로그램을 가능하게 하고 있습니다. 그러나 DNN이 하드웨어 장치에 배포되면 노화, 온도 변화 및 쓰기 오류와 같은 원치 않는 결함에 취약해집니다. 본 논문에서는 DNN 가중치를 상수로 곱한 다음 결함에 취약한 매체에 저장함으로써 무결성을 강화하는 간단하면서도 효과적인 방법을 제안합니다.

- **Technical Details**: 우리의 방법은 가중치를 저장하기 전에 원소별로 곱하고, 읽을 때는 원소별로 나누는 과정으로 구성되어 있습니다. 이러한 과정은 비트 플립으로 인해 발생하는 전체 절대 오류를 줄이는 데 효과적입니다. 우리는 32비트 부동소수점, 16비트 부동소수점 및 8비트 고정소수점의 세 가지 데이터 타입에 걸쳐 실험을 수행하였고, 이 방법이 8비트 고정소수점 ResNet50의 Top-1 정확도를 54.418만큼 개선함을 보였습니다.

- **Performance Highlights**: 제안된 방법은 이전의 오류 교정 코드 방법들과 대비하여 메모리 및 계산 오버헤드가 현저히 낮습니다. 우리의 방법은 단순한 원소별 곱셈 및 나눗셈만을 요구하며, 이는 DNN을 비트 플립으로부터 보호하는 데 중요한 기여를 합니다. 여러 모델과 데이터 타입에서의 성능 향상을 통해 이 방법의 일반화를 입증했습니다.



### USTCCTSU at SemEval-2024 Task 1: Reducing Anisotropy for Cross-lingual Semantic Textual Relatedness Task (https://arxiv.org/abs/2411.18990)
Comments:
          8 pages, 3 figures

- **What's New**: 이 논문은 다국어 간 의미 텍스트 관계성 과업(Cross-lingual semantic textual relatedness task)을 다루며, 사전 훈련된 모델인 XLM-R-base를 기반으로 하고 있습니다. 데이터 필터링 방법을 통해 다국어 처리의 문제를 경감시키고, 표준 정규성(whitening) 기술을 활용해 문장의 의미 유사성을 향상시켰습니다. 이 방법을 통해 스페인어에서 2위, 인도네시아어에서 3위를 기록하는 성과를 올렸습니다.

- **Technical Details**: 의미 텍스트 관계성(STR)은 두 문장 간의 다양한 공통점을 고려하는 개념으로, 주제 공유, 관점 표현 등 여러 요소를 포함합니다. 세미밸(SemEval) 2024의 트랙 C에서는 특정 목표 언어의 레이블 데이터 없이 다른 언어의 레이블 데이터만을 사용하여 시스템을 개발해야 합니다. 이를 해결하기 위해, 본 논문은 사전 훈련된 언어 모델을 기반으로 한 식별 방법과 함께, 문장 벡터로의 변환 과정에서 비동적 속성(anisotropic property)을 해소하기 위한 정규화 기법을 도입했습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 스페인어에서 2위, 인도네시아어에서 3위를 기록하며 여러 결과물들이 상위 10위 안에 진입하였습니다. 또한, 이 연구는 다국어 간 의미 관계성 과업에 대한 미래 조사 및 연구를 촉발할 수 있는 분석을 포함합니다. 이러한 성과는 다국어 모델이 직면한 문제를 해결하기 위한 기초로 활용될 수 있습니다.



### Random Sampling for Diffusion-based Adversarial Purification (https://arxiv.org/abs/2411.18956)
- **What's New**: 본 논문은 Denoising Diffusion Probabilistic Models (DDPMs)의 대안으로 랜덤 샘플링을 제안합니다. 기존 DDPM 샘플링은 안정적인 생성을 위한 것으로 설계되었으며, 이는 적대적 정화(adversarial purification)에는 최적의 솔루션이 아닐 수 있습니다. 랜덤 샘플링은 매 diffusion 과정에서 무작위 노이즈 공간에서 샘플링을 수행하여 더욱 강력한 적대적 공격에 대한 강인성을 제공합니다.

- **Technical Details**: 랜덤 샘플링 방식은 각 diffusion 단계에서 무작위 노이즈 공간에서 샘플을 추출하며, 이는 기존의 DDPM이나 DDIM 샘플링 방식이 인접한 노이즈 공간에서 샘플을 연속적으로 추출하는 방식과 대조됩니다. 논문에서 제안하는 mediator conditional guidance는 정화된 이미지와 깨끗한 이미지 입력 간의 예측 일관성을 보장합니다. 이를 통해 gradient-based guidance를 적용하여 샘플링 과정에서의 정확한 조건부 안내를 구현합니다.

- **Performance Highlights**: 랜덤 샘플링 방법은 다양한 샘플링 방법에 대한 평가에서 인상적인 개선을 보여줍니다. 연구진은 DiffAP라는 기준 방법을 설정하여 최첨단(SOTA) 접근 방식을 성능과 방어 안정성 모두에서 대폭 능가하였습니다. 특히, 강력한 공격 하에서도 DiffAP는 20% 이상의 강인성 향상을 10배의 샘플링 가속도로 달성했습니다.



### NeuroLifting: Neural Inference on Markov Random Fields at Sca (https://arxiv.org/abs/2411.18954)
- **What's New**: 이번 논문에서는 대규모 Markov Random Fields (MRFs)에서의 추론을 위한 새로운 접근법인 'NeuroLifting'을 소개합니다. NeuroLifting은 Graph Neural Networks (GNNs)를 활용하여 결정 변수를 재매개변수화함으로써, 전통적인 최적화 기법인 gradient descent를 가능하게 합니다. 이 기술은 대규모 문제에 대한 효율성과 높은 해상도 품질을 동시에 추구하는 혁신적인 솔루션입니다.

- **Technical Details**: NeuroLifting의 핵심 기술은 MRF 내에서 결정 변수를 GNN으로 재매개변수화하여 표준 gradient descent 방법을 적용할 수 있게 하는 것입니다. 이는 전통적인 리프팅 기술을 비모수적 신경망 프레임워크로 확장함으로써 이루어지며, 신경망의 연속적이고 매끄러운 손실 경관을 활용하여 최적화 과정을 단순화합니다. 또한, 대규모 MRF 문제에 대해 선형 계산 복잡성 증가를 보여줍니다.

- **Performance Highlights**: 경험적 결과에 따르면, NeuroLifting은 중간 규모에서는 기존의 정확한 해법인 Toulbar2와 비슷한 수준의 해상도를 달성하며, 기존의 근사 추론 방법들을 현저히 초과하는 성능을 보입니다. 특히, 30,000개의 노드 이상의 대규모 MRF 문제에 대한 비약적인 해상도 품질을 제공하고, 계산 효율성 또한 크게 향상되었습니다.



### Knowledge Database or Poison Base? Detecting RAG Poisoning Attack through LLM Activations (https://arxiv.org/abs/2411.18948)
- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG)의 보안 문제를 다루고 있습니다. 연구자들은 RevPRAG라는 자동화된 탐지 파이프라인을 소개하여 RAG에서 발생할 수 있는 이물질 공격(poisoning attack)을 검출하고자 하였습니다. 기존의 방법론과 비교했을 때, 이 접근법은 모델의 성능에 영향을 미치지 않으면서도 높은 정확도를 자랑합니다.

- **Technical Details**: RevPRAG는 LLM의 활성화(activation)를 활용하여 RAG의 응답이 유효한 것인지, 또는 중독된 것인지 판별합니다. 연구자들은 다양한 LLM 아키텍처 및 데이터셋에서 98% 이상의 진정 양성 비율을 달성하였고, 활성화 패턴 분석을 통해 중독된 응답과 유효한 응답을 명확히 구분할 수 있음을 보여주었습니다.

- **Performance Highlights**: RevPRAG는 GPT2-XL, Llama2, Mistral 등의 여러 LLM 아키텍처에서 일관되게 높은 성능을 나타냈으며, 커스텀 검출 데이터셋에서 98% 이상의 정확도를 기록했습니다. 실험 결과, RAG의 잘못된 응답이 중독된 것인지 아니면 비사실적인 것인지 구별하는 능력이 뚜렷하게 나타났습니다.



### ScratchEval: Are GPT-4o Smarter than My Child? Evaluating Large Multimodal Models with Visual Programming Challenges (https://arxiv.org/abs/2411.18932)
- **What's New**: 이 연구에서는 ScratchEval이라는 새로운 벤치마크를 제안하여 대규모 다중모달 모델(large multimodal models, LMMs)의 시각적 프로그래밍 추론 능력을 평가하고자 합니다. 기존의 평가 방법들은 특정 시나리오에 국한되어 있었지만, ScratchEval은 Scratch라는 블록 기반 시각적 프로그래밍 언어를 기반으로 하여 모델이 시각 정보를 처리할 수 있도록 설계되었습니다. 이를 통해 LMMs의 명령 이해 능력을 포괄적으로 평가할 수 있는 기회를 제공합니다.

- **Technical Details**: ScratchEval은 Scratch 스크립트가 포함된 305개의 다지선다 질문으로 구성되어 있으며, 각 질문은 문제 설명과 선택지를 포함합니다. 이 벤치마크는 일관된 논리적 사고 및 문제 해결 능력에 초점을 맞추며, 모델이 이미지, 그래픽 프로그래밍 언어 및 내재된 논리를 함께 이해해야 합니다. 질문은 수학, 논리적 사고, 그래픽 인식 및 공간 인식으로 분류되며, 다양한 인지 영역에서 모델의 능력을 평가합니다.

- **Performance Highlights**: 총 10개의 LMM을 평가한 결과, Gemini-1.5-Pro가 모든 카테고리에서 가장 높은 점수를 기록했지만, 대부분의 모델은 50% 정확도를 넘기기 어려웠습니다. 이는 LMM들이 비주얼 코드 추론 능력에서 한계를 지니고 있음을 시사합니다. 일반적으로, 수학 및 논리적 추론 작업에서 모델의 성능이 낮았고, 반면 그래픽 및 공간 인식 작업에선 상대적으로 더 나은 성과를 보였습니다. 이 연구는 적절한 프롬프트 기법이 LMM의 성능 향상에 기여할 수 있음을 보여주지만, 다중모달 LLM에 대한 연구는 더 필요합니다.



### VIPaint: Image Inpainting with Pre-Trained Diffusion Models via Variational Inferenc (https://arxiv.org/abs/2411.18929)
Comments:
          13 pages, 9 figures

- **What's New**: 이 논문은 VIPaint라는 새로운 방법론을 제안하여, 전이학습(pre-trained)된 diffusion model을 활용해 이미지 복원(inpainting) 문제를 해결하는 혁신적인 접근 방식을 보여줍니다. VIPaint는 고수준의 의미론(semantics)과 저수준의 세부정보를 동시에 추론하는 계층적 포스트리어(hierarchical posterior)를 정의함으로써 기존의 방법들을 뛰어넘는 성능을 보여줍니다. 기존의 알고리즘들이 대량의 훈련 세트와 파인튜닝(fine-tuning) 과정을 요구했던 것과 달리, 이 방법은 비염 번역(propagation) 없이도 빠르고 효율적으로 수행될 수 있습니다.

- **Technical Details**: VIPaint는 비선형 Markov 근사(non-Gaussian Markov approximation) 방식을 통해 진정한 (L)DM 포스트리어를 최적화합니다. 이 방법은 상태 공간(state space)에서의 노이즈 수준을 전략적으로 고려하는 계층적 포스트리어를 적용하며, 이를 통해 관측된 픽셀로부터 정보를 효과적으로 추론할 수 있습니다. 저자는 VIPaint의 효율적인 조건화 기법을 통해 기존의 이미지 복원 문제에서 발생하는 여러 주요 문제들, 특히 신뢰성 저하 문제를 해결하고 있음을 보여줍니다.

- **Performance Highlights**: VIPaint 방법론은 기존의 픽셀 기반 및 잠재적 잠수 모델(latent diffusion models)에서 이미지 복원, 블러 제거(deblurring), 초해상도(superresolution) 등 다양한 응용 분야에서 눈에 띄는 질적 및 양적 개선을 이뤄냈음을 강조합니다. 많은 실험을 통해 VIPaint는 데이터 다양성과 신뢰도(plausibility) 모두에서 기존 방법들 보다 우수한 성능을 발휘하는 것을 지속적으로 입증했습니다. 이러한 성능 향상은 이미지 생성 분야에서 VIPaint가 새로운 기준을 세울 수 있는 가능성을 강조합니다.



### EzSQL: An SQL intermediate representation for improving SQL-to-text Generation (https://arxiv.org/abs/2411.18923)
Comments:
          Under Review at Expert System With Applications Journal

- **What's New**: 이번 연구에서는 EzSQL이라는 SQL 중간 표현(intermediate representation)을 제안하여 SQL과 자연어(natural language) 텍스트 시퀀스를 정렬하려고 합니다. EzSQL은 SQL 쿼리를 간소화하고 자연어와 더 가까운 형태로 만들어 주며, 이는 SQL-to-text 생성 모델에 입력으로 사용됩니다. 또한 Sql-to-text 생성모델을 사용하여 Text-to-SQL 파서의 성능을 향상시킬 수 있음을 보여줍니다.

- **Technical Details**: EzSQL은 SQL 쿼리의 구조를 단순화하여 자연어에 가깝도록 만드는 것을 목표로 합니다. EzSQL은 UNION과 INTERSECT와 같은 집합 연산(set operator)을 제거하고, JOIN과 중첩 서브쿼리(nested subquery) 또한 배제합니다. 이 과정에서 작업의 표현이 간단해져, pretrained language model(BART)의 입력으로 적합한 형태가 되며, 이는 자연어 텍스트 생성을 지원합니다.

- **Performance Highlights**: EzSQL을 사용하는 SQL-to-text 생성 모델은 WikiSQL과 Spider 데이터세트에서 최신 기술 수준(state-of-the-art) 성능을 달성하였습니다. 이 모델을 사용하여 생성된 데이터는 Text-to-SQL 파서의 성능을 향상시키는 데 기여할 수 있는 잠재력을 보여줍니다. 연구 결과는 EzSQL을 통한 접근 방식이 SQL-to-text 생성에서 효과적임을 입증하고 있습니다.



### Devising a Set of Compact and Explainable Spoken Language Feature for Screening Alzheimer's Diseas (https://arxiv.org/abs/2411.18922)
Comments:
          Published at ISCSLP 2024

- **What's New**: 이 연구는 알츠하이머병(Alzheimer's Disease, AD)의 조기 탐지를 위한 말을 기반으로 한 새롭고 효과적인 기능 집합을 제안합니다. 특히 Cookie Theft 그림 설명 과제를 활용하여, 대형 언어 모델(LLM)의 시각적 처리 능력을 결합한 기존의 언어적 특징을 초월한 설명 가능한 특징을 도출했습니다. 제안된 특징은 자동 AD 스크리닝의 해석성을 증대시켜 주목받고 있습니다.

- **Technical Details**: 이 연구에서 사용된 데이터 세트는 ADReSS Challenge 2020에서 파생된 Pitt Corpus의 일부로, 156개의 음성 샘플과 그에 해당하는 전사가 포함되어 있습니다. 참여자는 알츠하이머병이 없는 그룹과 있는 그룹으로 나뉘며, 새로운 11개의 기능을 제안하여 Cookie Theft 그림에 대한 설명을 분석했습니다. 이 기능들은 TF-IDF 및 LLM의 시각적 분석 능력을 활용하여 세부 사항을 정량화합니다.

- **Performance Highlights**: 실험 결과, 제안된 새 기능 세트는 기존의 40개의 언어적 기능을 초월하여 안정적인 성능을 보였습니다. 15개 차원만으로 ADReSS 테스트 세트에서 85.4%의 경쟁력 있는 정확도를 달성하였으며, 이는 기존 기능 집합에 비해 차원 효율성을 강조합니다. 따라서 이 연구는 AD 탐지에서 효과적이고 설명 가능한 접근 방식을 제시합니다.



### Federated Continual Graph Learning (https://arxiv.org/abs/2411.18919)
Comments:
          Under Review

- **What's New**: 이번 연구는 Federated Continual Graph Learning (FCGL)을 소개하며, 분산 형태의 진화하는 그래프에서 GNN을 조정하는 방법을 제시합니다. FCGL은 개인 데이터 저장 및 프라이버시 제약을 준수하면서 여러 클라이언트가 협력적으로 학습할 수 있도록 설계되었습니다. 이를 통해 기존의 중앙 집중형 CGL 방법의 한계를 극복하고, 분산된 그래프 데이터에서 집단 지성을 활용할 수 있는 새로운 가능성을 제시합니다.

- **Technical Details**: 연구에서는 Local Graph Forgetting (LGF)과 Global Expertise Conflict (GEC)이라는 두 가지 주요 도전 과제가 확인되었습니다. LGF는 새로운 작업에 적응할 때 로컬 GNN이 이전 지식을 잊게 되는 현상이고, GEC는 클라이언트의 전문성이 일관되지 않아 글로벌 GNN의 성능이 저하되는 문제를 의미합니다. 이를 해결하기 위해 POWER라는 프레임워크를 제안하며, 이는 경험 노드를 최대한 활용하여 LGF를 완화하고, GEC를 해결하기 위한 새로운 전략을 적용합니다.

- **Performance Highlights**: POWER 프레임워크는 8개의 다양한 그래프 데이터셋에서 평가를 진행하였으며, 결과적으로 기존 중앙 집중형 CGL 알고리즘의 연합 확장 및 비전 중심 연합 지속 학습 알고리즘보다 우수한 성능을 보였습니다. 이러한 성과는 FCGL이 분산 환경에서도 효과적으로 작동하며, 집단 지성을 활용하여 초기 작업의 성과를 잃지 않도록 도와줌을 보여줍니다. 연구 결과는 다양한 실제 응용 분야에서의 FCGL의 적용 가능성을 확장하는 데 기여할 것으로 기대됩니다.



### ArEEG_Words: Dataset for Envisioned Speech Recognition using EEG for Arabic Words (https://arxiv.org/abs/2411.18888)
Comments:
          arXiv admin note: substantial text overlap with arXiv:2402.15733

- **What's New**: 이 논문에서는 첫 번째 아랍어 EEG 데이터셋인 ArEEG_Words를 소개합니다. 이 데이터셋은 22명의 참가자로부터 수집된 EEG 신호로 구성되어 있으며, 이제까지 아랍어와 관련된 BCI 연구에서 사용 가능한 공개 데이터셋이 부족한 문제를 해결하고자 합니다.

- **Technical Details**: ArEEG_Words 데이터셋은 14채널 Emotiv Epoc X 장치를 사용하여 기록되었습니다. 참가자는 8시간 전부터 카페인, 알코올, 담배와 같은 신경계에 영향을 미치는 물질을 피하고, 평온한 방에서 16개의 아랍어 단어 중 하나를 마음속으로 상상하며 10초간 집중하도록 지시받았습니다.

- **Performance Highlights**: 총 352회의 EEG 기록이 수집되어 각 기록은 250ms 신호로 나뉘어 총 15,360개의 EEG 신호가 생성되었습니다. ArEEG_Words 데이터셋은 아랍어 EEG 연구 분야에서는 최초이며, 연구자들에게 공개되어 이 분야의 연구를 촉진할 것으로 기대됩니다.



### Redesigning the ensemble Kalman filter with a dedicated model of epistemic uncertainty (https://arxiv.org/abs/2411.18864)
- **What's New**: 이 논문은 에피스테믹 불확실성(epistemic uncertainty)을 고려한 가능론적 앙상블 칼만 필터(possibilistic ensemble Kalman filter)를 제안합니다. 확률론적 접근이 아닌 가능성 이론(possibility theory)을 통해 불확실성을 모델링하는 새로운 방법론을 제시하며, 기존의 앙상블 Kalman 필터와 비교하여 장점이 있음을 보여줍니다. 특히, 작은 샘플 크기에서도 우수한 성능을 보이며 진정한 유효성(aleatoric uncertainty)을 다룰 수 있다는 점에서 혁신적입니다.

- **Technical Details**: 논문은 가능론적 프레임워크를 바탕으로 에피스테믹 불확실성을 모델링하는 여러 불확실성 표현 방식의 한계를 논의합니다. 기존 앙상블 칼만 필터의 동작과 과정은 확률론적 방법에 기초하지만, 제안된 가능론적 방법은 최대화를 통한 확률 개념을 변경하는 방식을 사용합니다. 이 방법은 기존 방법들과의 대조에서 확률적 평균을 제공하는 Gaussian 맞춤법과 상응하는 새로운 개념을 소개합니다.

- **Performance Highlights**: 제안된 가능론적 앙상블 칼만 필터는 선형 및 비선형 동역학을 포함한 다양한 상황에서 성능을 평가받았습니다. 기존의 두 가지 버전의 앙상블 칼만 필터 및 비국소 칼만 필터(UKF)와 비교하여, 작은 샘플을 가지고도 더욱 향상된 예측 능력을 보여줍니다. 이를 통해 에피스테믹 불확실성을 효과적으로 처리할 수 있는 가능론적 접근의 유용성이 강조됩니다.



### An Integrated Artificial Intelligence Operating System for Advanced Low-Altitude Aviation Applications (https://arxiv.org/abs/2411.18845)
- **What's New**: 이 논문은 저고도 항공 응용 프로그램을 위해 설계된 포괄적인 인공지능 운영 체제를 소개합니다. 이 시스템은 고성능의 OrinFlight OS, 고급 이미지 분석을 지원하는 UnitedVision, 정밀한 환경 모델링을 제공하는 UnitedSense, 동적 경로 계획 및 내비게이션 시스템인 UnitedNavigator 등 여섯 개의 핵심 구성 요소를 통합합니다. 또한, UA DevKit 저코드 플랫폼은 사용자 친화적인 커스터마이징과 애플리케이션 개발을 가능하게 합니다.

- **Technical Details**: OrinFlight OS는 NVIDIA Orin 플랫폼과 UNIX 아키텍처를 기반으로 한 고성능 운영 체제이며, 저고도 항공 응용을 위한 독특한 도전 과제를 해결합니다. 이 시스템은 실시간 자원 관리 및 분산 데이터 처리를 통해 드론 운영의 신뢰성을 보장하며, 다양한 모듈 간의 효율적인 데이터 교환을 지원합니다. 이러한 기능들은 드론이 실시간 결정making을 위해 필요한 정보를 신속하고 정확하게 처리할 수 있도록 합니다.

- **Performance Highlights**: OrinFlight OS는 지연 없이 동적 환경 변화에 즉각적으로 반응할 수 있도록 최적화된 인터럽트 처리 메커니즘을 통합하여 드론의 저지연 제어를 가능하게 합니다. 이는 드론이 갑작스러운 장애물 회피나 급변하는 기상 변화에 신속하게 적응할 수 있도록 돕습니다. 또한, 본 시스템은 여러 드론 간의 효율적인 작업 배분과 실시간 통신을 통해 협동 작업을 향상시켜, 더 큰 응용 프로그램과 환경에서도 신뢰성 높은 운영이 가능하도록 합니다.



### RelCon: Relative Contrastive Learning for a Motion Foundation Model for Wearable Data (https://arxiv.org/abs/2411.18822)
- **What's New**: 이번 연구에서는 RelCon이라는 새로운 자기 지도 자기 대조 학습 방법을 소개합니다. 이 방법은 착용 가능한 센서에서 모션 파운데이션 모델을 훈련하기 위해 학습 가능한 거리 측정 방법과 부드러운 대조 손실을 결합합니다. RelCon은 87,376명의 참가자로부터 10억 개의 세그먼트를 사용하여 훈련되었으며, 다양한 다운스트림 작업에서 강력한 성능을 보여줍니다. 특별히 우리는 착용 가능한 기기의 모션 데이터에 대한 자기 지도 학습 모델의 일반화 가능성을 처음으로 입증합니다.

- **Technical Details**: RelCon은 특징적인 모티프와 도메인 특유의 의미 정보를 포착하는 학습 가능한 거리 측정 방법을 사용합니다. 이 접근법은 신호의 형태에 기반하여 의미 있는 쌍을 구성함으로써 더욱 유의미한 세그먼트를 찾아냅니다. Time-series 데이터에 적합한 새로운 대조 표현 학습 방법에 기반하여, 우리는 신호 위치나 방향에 관계없이 모티프를 비교하는 것을 통해 성능을 향상시킵니다. 기존의 대조 손실 방식과 달리 소프트 손실을 도입하여 의미 있는 기본 상태가 있는 군집을 형성할 수 있도록 합니다.

- **Performance Highlights**: RelCon으로 훈련된 모션 파운데이션 모델은 6개의 다양한 다운스트림 데이터셋에서 우수한 성능을 지속적으로 보여줍니다. 이 모델은 선행 연구들의 현재 최첨단 가속도계 자기 지도 접근 방식을 초월하며, 여러 평가 작업 전반에 걸쳐 일반화 가능성을 처음으로 입증합니다. 특히 활동 분류, 운동 분류 및 보행 분석와 같은 다양한 작업에서 뛰어난 성과를 보였습니다.



### Unifying Generative and Dense Retrieval for Sequential Recommendation (https://arxiv.org/abs/2411.18814)
- **What's New**: 본 논문은 generative retrieval 및 sequential dense retrieval 두 방법을 비교하고, LIGER라는 새로운 하이브리드 모델을 제안합니다. LIGER는 sequential dense retrieval의 강점을 generative retrieval에 통합하여 성능 격차를 줄이고, 특히 cold-start 아이템 추천에서 향상된 결과를 도출합니다. 이러한 연구는 추천 시스템에서 사용자의 요구에 맞는 성능 및 계산 효율성을 함께 고려하고 있습니다.

- **Technical Details**: 이 논문에서는 generative retrieval과 dense retrieval의 수식을 제시하며, TIGER와 같은 generative retrieval 방법이 두 단계의 훈련 과정을 따른다고 설명합니다. 첫 번째 단계에서 아이템의 특성을 바탕으로 텍스트 설명을 수집하고, 다음 단계에서는 이러한 설명을 사용하여 콘텐츠 모델(예: language encoder)을 통해 아이템의 텍스트 임베딩을 생성하는 방식입니다. 여기서 각 아이템은 semantic ID를 부여받으며, 이러한 ID는 추천 과정에서 사용됩니다.

- **Performance Highlights**: 실험 결과, dense retrieval 방법이 대부분의 데이터셋에서 generative retrieval보다 우수함을 보여주었습니다. 특히 cold-start 아이템 예측에서 generative retrieval 방법은 성능이 저조한 반면, dense retrieval은 강력한 성능을 발휘했습니다. LIGER 모델을 통해서는 generative retrieval의 성능이 개선되었으며, 아이템 추천의 효율성과 효과성을 높이는 데 기여하고 있습니다.



### NewsEdits 2.0: Learning the Intentions Behind Updating News (https://arxiv.org/abs/2411.18811)
Comments:
          9 pages main body, 11 pages appendix

- **What's New**: 이 연구에서는 뉴스 기사의 업데이트 내용을 예측하기 위해 언어적 특징을 활용할 수 있다는 가설을 세웠습니다. 이를 위해 NewsEdits 2.0이라는 수정 의도 분류체계를 도입하였고, 9,200개의 문장 쌍을 주석 처리하여 고점수 앙상블 모델을 학습시켰습니다. 이 과정을 통해 기존 기사의 사실이 언제 업데이트될지를 예측할 수 있는 알고리즘을 개발했습니다.

- **Technical Details**: 본 연구는 다양한 이유로 작성된 뉴스 기사를 통해 사실의 업데이트 패턴을 분석하였습니다. NewsEdits 2.0 분류체계를 통해, 언어적 단서(linguistic cues)와 미래 시제 동사(future-tense verbs), 통계(statistics), 자주 업데이트되는 사건들(common updating events) 등의 언어적 패턴을 발견하였습니다. 모델 성능은 테스트 세트에서 Macro-F1 점수 0.58을 기록했으며, 특히 업데이트 가능성이 높은 문장에 대해서는 정확도가 0.74에 달했습니다.

- **Performance Highlights**: 마지막으로, LLM이 구식 데이터에서 문서를 검색할 경우 발생할 수 있는 오류를 줄이기 위한 사례 연구를 진행했습니다. 우리의 예측을 활용한 모델은 정확도가 거의 오라클 수준에 도달하는 성과를 보였습니다. 이 연구 결과는 LLM의 질문 응답 시스템에서 구식 정보를 다루는 데 있어 중요한 기여를 할 것으로 기대됩니다.



### UOE: Unlearning One Expert Is Enough For Mixture-of-experts LLMS (https://arxiv.org/abs/2411.18797)
- **What's New**: 이 연구는 Mixture-of-Experts (MoE)라는 대규모 언어 모델(LLM)의 새로운 특성과 함께 전통적인 학습 제거 방법의 한계를 파악하고, 이를 해결하기 위한 혁신적인 단일 전문가 제거 프레임워크인 UOE를 제안하고 있습니다. UOE는 특정 지식과 가장 관련이 깊은 전문가에 집중하여 제거 작업을 수행하며, 이 과정에서 모델의 유용성을 최대한 유지하도록 설계되었습니다. 전체적으로, 이 연구는 MoE LLM의 제거에 있어서 발생할 수 있는 독특한 문제를 해결하고자 하는 최초의 시도를 보여줍니다.

- **Technical Details**: UOE는 전문가 귀속(expert attribution) 기술을 사용하여 특정 포기의 요소와 가장 많이 연관된 전문가를 식별합니다. 이어서, 러우터(router)에 앵커 손실(anchor loss)을 적용하여 해당 전문가의 활성 상태를 안정화시킵니다. 이를 통해 전문가 선택의 빈번한 변화를 방지하고, 집중적이고 제어된 형태로 제거를 구현할 수 있습니다. UOE는 다양한 제거 알고리즘과 호환 가능하여 기존 방법과 통합하여 사용할 수 있습니다.

- **Performance Highlights**: 실험 결과, UOE 프레임워크는 다양한 MoE LLM 아키텍처에서 모델의 유용성을 평균 35% 향상시키고, 지식 제거의 질을 최대 5%까지 향상시켰습니다. 흥미롭게도, 모델 매개변수의 단 0.06%만을 제거하면서 이러한 성과를 달성하였습니다. 이는 MoE LLM에 대한 효율적인 제거 방식을 입증하며, 새로운 연구 방향을 제시합니다.



### CoVis: A Collaborative Framework for Fine-grained Graphic Visual Understanding (https://arxiv.org/abs/2411.18764)
- **What's New**: 본 논문은 협업 프레임워크인 CoVis를 제안합니다. 이 프레임워크는 이미지에서 최대한의 지식을 추출하고, 더 전체적인 관점에서 이미지를 이해하도록 돕기 위해 설계되었습니다. 또한, CoVis는 Fast Segment Anything Model(FastSAM)과 U-Net 모델을 기반으로 한 동적 조정을 통해 시각적 분석을 생성합니다.

- **Technical Details**: CoVis는 복합 단계별 이미지 분할과 언어 생성 방법을 적용하여 일반적인 시각 콘텐츠 이해를 목표로 합니다. 이 과정에서 FastSAM을 이용한 코스 그레인(segmentation) 모델과 U-Net 기반의 파인 그레인(segmentation) 모델을 활용하며, 생성된 텍스트의 품질 향상을 위해 Prompt Engineering 기법도 도입합니다. 이러한 프로세스는 다양한 시각적 요소를 평가하고 해석할 수 있는 텍스트를 생성합니다.

- **Performance Highlights**: 정량적 및 정성적 실험을 통해 CoVis는 기존의 방법보다 우수한 기능 추출 성능을 보였습니다. 특히, CoVis는 일반 목적으로 만들어진 대형 모델보다 더 포괄적이고 상세한 시각적 설명을 생성할 수 있습니다. 또한, 32명의 참여자를 대상으로 한 실험 결과, 이 프레임워크는 시각적 정보의 전달 효율성과 품질을 개선하는 데 효과적임을 입증하였습니다.



### Generative Visual Communication in the Era of Vision-Language Models (https://arxiv.org/abs/2411.18727)
Comments:
          PhD Thesis

- **What's New**: 이번 연구에서는 최근 비전-언어 모델(vision-language models, VLMs)의 발전을 활용하여 효과적인 비주얼 커뮤니케이션 디자인을 자동으로 생성하는 방법을 탐구합니다. 기존 생성 모델들이 텍스트로부터 이미지를 생성하는 데는 큰 발전을 이루었지만, 복잡한 아이디어를 단순화된 추상 비주얼로 표현하는 데에는 한계를 보이고 있습니다. 이에 따라 모델의 작업 공간을 제한하고 특정 작업에 맞춘 정규화를 도입하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: 자유 손 스케치(free-hand sketching)는 아이디어와 개념을 표현하는 중요한 시각적 도구입니다. 본 논문에서는 CLIP을 사용하여 입력 이미지를 추상적인 스케치로 변환하는 프로세스를 최적화하는 방법을 제안합니다. 이 방법은 CLIP 인코더를 사용하여 사진의 기하학적 기반을 토대로 세밀하고 의미 있는 스케치를 생성합니다. 사전 훈련된 CLIP 모델의 최종 및 중간 활성화를 결합하여 기하학적 및 의미적 단순화를 달성합니다.

- **Performance Highlights**: 제안한 방법론은 다양한 수준의 추상화를 가능하게 하여 인간과 유사한 스타일의 스케치를 만들어낼 수 있습니다. 다수의 레벨의 추상화를 제공하며, 기존의 스케치 데이터 세트 또는 새로운 훈련 단계 없이 최종 출력을 조정할 수 있는 유연성을 가지고 있습니다. 스케치를 통해 입력 객체의 본질을 잘 포착하면서도 최적화된 성능을 발휘하는 결과를 보여 줍니다.



### Timing Matters: Enhancing User Experience through Temporal Prediction in Smart Homes (https://arxiv.org/abs/2411.18719)
Comments:
          7 pages + 1 reference, 5 figures, 5 tables

- **What's New**: 이번 논문은 IoT (Internet of Things) 장치가 우리의 행동 데이터에서 시간적 차원, 즉 행동의 순간을 예측하는 데 초점을 맞추었다. 기존 연구들은 주로 행동 예측에 집중했으나, 행동의 타이밍은 사용자 경험과 효율성에 미치는 영향이 크다. 우리는 11,000개의 행동 시퀀스와 이에 상응하는 날짜와 시간 스탬프를 포함하는 새로운 데이터셋을 제시했다. 이 데이터셋을 기반으로, 하루의 시간 간격에 대한 k-class 분류를 위한 모델을 제안하였다.

- **Technical Details**: 연구에서 제안한 모델은 사용자의 행동 시퀀스를 학습하여 행동의 실행 시점을 예측한다. 우리는 제안된 데이터셋에서 40%의 정확도로 96-class 분류를 달성했으며, 정확한 타임스탬프가 포함된 데이터셋에서는 80%의 정확도를 달성하였다. 이 모델은 기존의 Hidden Markov Models (HMMs)나 LSTM(Long Short-Term Memory)과 같은 접근 방식과 차별화된다. 우리의 모델은 시간 예측 문제에 도전하면서 인간 중심의 AI 개발에 기여할 예정이다.

- **Performance Highlights**: 제안된 모델은 스마트 홈 환경에서 사용자 행동의 동적 변화를 예측하는 데 효과적이다. 논문에서는 시퀀스 기반의 행동 예측 접근법의 한계를 넘어, 행동의 발생 시간 예측이 중요하다고 강조하였다. 정확한 시간 예측을 통해 스마트 시스템은 사용자에게 보다 능동적이고 개인화된 지원을 제공할 수 있게 된다. 이는 궁극적으로 스마트 홈의 사용자 경험을 혁신적으로 향상시킬 것이다.



### Explainable deep learning improves human mental models of self-driving cars (https://arxiv.org/abs/2411.18714)
Comments:
          * - equal contribution

- **What's New**: 이번 연구에서는 
자율주행차 (autonomous vehicle) 분야에서 블랙박스 모션 플래너의 행동을 설명하는 새로운 접근 방식인 
개념 포장 네트워크 (concept-wrapper network, CW-Net)를 소개합니다. 
CW-Net은 인간이 해석 가능한 개념에 기반하여 블랙박스 알고리즘의 의사결정을 명확히 하며, 주행 환경을 이해하는 데 있어 드라이버의 정신 모델을 개선하는 역할을 합니다.

- **Technical Details**: CW-Net은 사전 훈련된 딥 러닝 (deep learning) 네트워크의 최종 보상 층을 개념 분류기로 교체하고, 이후 특정 시나리오 타입과 주행 결정을 예측하도록 훈련됩니다. 
이 방법은 기존의 네트워크 구조에 영향을 주지 않으며 성능 저하 없이 설명의 정확성을 높이는 데 중점을 둡니다. 
실험 결과 CW-Net은 자율주행차에 배치되어 드라이버의 잘못된 정신 모델을 수정하는 데 성공했습니다.

- **Performance Highlights**: 본 연구는 자율주행차의 실제 환경에서 수행된 최초의 연구로, 
CW-Net이 개념 분류를 통해 설명력을 높인 점을 강조합니다. 
120명의 사용자에 대한 온라인 연구에서 통계적으로 유의미한 결과를 도출하며, 이로 인해 드라이버의 경계 의식이 향상되었습니다. 
이 연구는 자율주행차와 같은 고위험 시스템에서 인간과의 협력적 안전성을 개선할 수 있는 가능성을 제시합니다.



### Embracing AI in Education: Understanding the Surge in Large Language Model Use by Secondary Students (https://arxiv.org/abs/2411.18708)
Comments:
          6 main pages, 5 figures

- **What's New**: 이번 연구는 OpenAI의 ChatGPT와 같은 대형 언어 모델(LLMs)의 중고등학생 사용 실태를 조사했습니다. 300명 이상의 학생을 대상으로 한 설문 조사 결과, 무려 70%의 학생이 LLM을 사용한 경험이 있다고 응답했습니다. 이는 젊은 성인보다 높은 비율이며, 7학년부터 12학년까지 사용 비율이 일관되게 나타났습니다.

- **Technical Details**: 설문 조사는 미국 전역의 중고등학생을 대상으로 진행되었으며, 신뢰성 있는 응답을 확보하기 위해 간단한 질문지 형식으로 제작되었습니다. 응답자들은 교육의 다양성을 반영하며, 43개 주에서 피드백을 수집했습니다. 설문 결과, 학생들은 LLM을 언어 예술, 역사, 수학 과제 등 다양한 과목에서 활용하고 있으나, 정답률이나 역사적 맥락에서의 혼동으로 인한 우려가 표출되었습니다.

- **Performance Highlights**: LLM 사용자 중 84%가 ChatGPT를 이용한다고 응답했으며, 수학 과목에서도 28%의 학생이 LLM을 사용하고 있다고 밝혔습니다. 그러나 51%의 학생들은 현 모델에 대한 개선을 원하고 있으며, 더 coherent한 응답과 올바른 답변, 복잡한 질문에 대한 능력을 높이길 기대하고 있습니다. 최종적으로 이 연구는 LLM을 교육에 통합하기 위한 방향성을 제시하며, 다양한 학생들이 접근할 수 있는 방법을 모색하고 있습니다.



### Random Walks with Tweedie: A Unified Framework for Diffusion Models (https://arxiv.org/abs/2411.18702)
- **What's New**: 본 논문은 확산 모델(Generative Diffusion Model) 알고리즘을 설계하기 위한 단순한 템플릿을 제시합니다. 확산 샘플링이 일련의 랜덤 워크(random walks)로 해석되면서 새로운 이론적 틀을 제공합니다. 이러한 접근 방식은 기존의 Markov 체인 또는 역 확산 이론을 피하고, 랜덤 워크와 Tweedie의 공식을 중심으로 한 이론을 강조합니다. 주요 기여는 여러 알고리즘 간의 관계를 통일된 관점에서 이해할 수 있도록 돕는 것입니다.

- **Technical Details**: 딥 스코어 기반 확산 모델은 훈련 데이터로부터 확률 밀도의 로그의 기울기인 스코어 함수를 학습하는 생성 모델입니다. 이 스코어 함수는 확산의 수학적 이론을 기반으로 반복적인 알고리즘에서 새로운 샘플을 생성하는 데 사용됩니다. 본 프레임워크는 알고리즘적 선택을 명확히 분리하여 훈련 및 샘플링을 가능하게 하며, 샘플링 중 사용하는 노이즈 스케줄이 훈련 시 사용되는 것과 일치할 필요가 없도록 설계되었습니다. 이러한 유연성 덕분에 조건부 샘플링도 가능해졌습니다.

- **Performance Highlights**: 제안된 알고리즘 템플릿은 여러 기존 확산 모델들과의 일관성을 보여줍니다. 특히, 노이즈가 가득 찬 데이터에서의 훈련을 통해 고품질 이미지를 생성하는 다양한 방법론을 제시합니다. 기존의 방법들과 비교하여 더 간단하고 직관적인 알고리즘 구조를 갖추고 있으며, 샘플링 과정의 효율성을 높이는 데 기여하고 있습니다. 이 프로세스는 주어진 입력 이미지의 품질을 유지하면서도 효과적으로 역 문제를 해결할 수 있는 가능성을 열어줍니다.



### On the Effectiveness of Incremental Training of Large Language Models (https://arxiv.org/abs/2411.18700)
- **What's New**: 이 논문은 대규모 언어 모델(LLM) 훈련에 있어 점진적 레이어 훈련(incremental layer-wise training) 방법의 효율성을 검토합니다. 기존의 훈련 방식이 아닌 점진적인 방법으로 레이어를 추가함으로써 훈련 과정을 최적화할 수 있을 것이라는 기대가 있었습니다. 그러나 연구 결과, 점진적 방법은 결국 더 큰 계산 비용을 요구하며 전통적인 훈련 방식과의 성능 차이를 메우기 위해 더 오랜 훈련이 필요하다는 것을 보여줍니다.

- **Technical Details**: 점진적 레이어 훈련은 네트워크의 하위 레이어가 먼저 훈련되어 안정화된 후, 상위 레이어를 훈련하는 방식입니다. 이러한 접근법은 높은 수준의 레이어가 낮은 수준의 레이어에서 학습된 표현을 의존하므로, 기초적인 언어적 특징을 잘 학습할 수 있도록 도와줍니다. 하지만, 이 방법의 효용은 이전 연구에서 경험적으로 검증되지 않았고, 이 연구는 이를 해결하기 위한 임상 실험을 진행했습니다.

- **Performance Highlights**: 결과적으로, 점진적 레이어 훈련은 초기의 계산 효율성을 보였지만, 전통적인 훈련과 비교했을 때 성능에서 유의미한 이점을 나타내지 않았습니다. 특히, 높은 수준의 레이어가 학습할 기초적인 특징이 안정화되기까지 시간이 더 소요되며, 이는 전체적인 계산 비용을 증가시킵니다. 따라서, 이 연구는 대규모 언어 모델 훈련에 있어 점진적 레이어 훈련의 유효성이 제한적임을 강조합니다.



### Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignmen (https://arxiv.org/abs/2411.18688)
- **What's New**: 이 논문에서는 Multimodal Large Language Models (MLLMs)의 안전성을 향상시키기 위한 새로운 방안을 제시합니다. 기존의 RLHF(강화 학습 기반 인간 피드백)에 의한 안전 정렬이 해킹 공격(jailbreak attacks)에 취약함을 보이며, 이를 해결하기 위한 새로운 방어 프레임워크인 Immune을 소개합니다. Immune은 인퍼런스(inference) 단계에서 안전한 보상 모델을 활용하여 해킹 공격에 강력하게 대응합니다.

- **Technical Details**: Immune은 두 가지 주요 구성 요소로 이루어져 있습니다. 첫째, 인퍼런스 단계에서 MLLM을 정렬하는 방법으로 제어된 디코딩(controlled decoding)을 사용하며, 안전 인식 유틸리티 또는 보상 함수를 활용합니다. 둘째, 이 방어 전략을 KL-정규화 강화 학습 프레임워크 내에서 수학적으로 정형화하여, 적대적인 프롬프트 분포에 대한 강건성을 증명합니다.

- **Performance Highlights**: Immune은 다양한 jailbreak 벤치마크에서 실험을 통해 기존 최첨단 방어 전략들보다 뛰어난 성능을 보입니다. 예를 들어, LLaVA-1.6을 이용한 MM-SafetyBench에서 Immune은 공격 성공률을 기존 MLLM과 AdaShield에 비해 각각 57.82%와 16.78%로 감소시켰습니다. 이러한 결과는 Immune이 모델의 원래 기능을 유지하면서 안전성을 효과적으로 높인다는 것을 보여줍니다.



### MatchDiffusion: Training-free Generation of Match-cuts (https://arxiv.org/abs/2411.18677)
Comments:
this https URL

- **What's New**: MatchDiffusion는 텍스트-비디오 확산 모델을 사용하여 매치 컷을 생성하는 최초의 훈련이 필요 없는 방법입니다. 이 방법은 초기 디노이징 단계에서 장면의 구조를 결정하고 후속 단계에서 세부 사항을 추가하는 확산 모델의 특성을 활용합니다. MatchDiffusion는 'Joint Diffusion'과 'Disjoint Diffusion'의 두 단계를 통해 매치 컷의 독창적이고 일관된 비디오를 생성합니다.

- **Technical Details**: MatchDiffusion에서 'Joint Diffusion' 단계는 두 개의 프롬프트를 단일 노이즈 샘플에서 초기화하여 서로 공유하는 구조를 형성합니다. 이후 'Disjoint Diffusion' 단계에서 각 비디오는 프롬프트에 따라 독립적으로 발전하여 의미가 다른 비디오 쌍을 생성합니다. 이 접근 방식은 매치 컷을生成하기 위한 구조적 일관성을 유지하면서도 고유한 콘텐츠를 제공합니다.

- **Performance Highlights**: 사용자 연구 및 척정을 통해 MatchDiffusion이 매치 컷 생성에서 높은 효과성과 가능성을 보여줍니다. 이 연구는 기존의 방법들과 비교하여 매치 컷 품질을 정량화할 수 있는 메트릭스를 제안하고 있습니다. 또한, 다양한 기술 수준의 창작자들이 매치 컷을 실험할 수 있도록 하여 이러한 강력한 도구의 민주화를 목표로 하고 있습니다.



### Embodied Red Teaming for Auditing Robotic Foundation Models (https://arxiv.org/abs/2411.18676)
- **What's New**: 이번 논문에서는 복잡한 언어 변화를 모두 테스트하는 것의 어려움 때문에 로봇 모델의 안전성과 효과성을 평가하는 데 큰 도전이 있다는 점을 지적하고 있습니다. 기존의 벤치마크는 인간이 생성한 제한된 지침 세트에 의존하며, 안전성 평가를 간과함으로써 현실적인 성능을 과대평가할 수 있습니다. 이에 대해 본 연구는 Embodied Red Teaming(ERT)이라는 새로운 평가 방법을 제안하여 다양한 지침을 생성하고 로봇 모델의 역량을 시험하는 데 도움을 줍니다.

- **Technical Details**: 로봇 정책을 감사하는 과정에서 ERT는 로봇이 실패해야 하는 다양한 테스트 지침 {𝒄1,⋯,𝒄N}을 찾아내기 위한 최적화 문제로 정의됩니다. 이러한 지침들은 현재 환경에서 실행 가능한 작업을 나타내야 하며, 로봇이 실패할 수 있는 과제를 포함하여야 합니다. ERT는 비전 언어 모델(VLMs)을 활용하여 환경 컨텍스트를 통합한 현실적인 지침을 생성하는 자동화된 방법론을 제시합니다.

- **Performance Highlights**: 실험 결과에 따르면 최신 로봇 모델들이 ERT 테스트에서 자주 실패하거나 불안전한 행동을 보였으며, 이는 기존 벤치마크가 현실적인 성능과 안전성을 평가하는 데 한계가 있음을 강조합니다. ERT는 로봇 모델이 성공할 수 있는 지침에서의 성능 저하를 탐지할 수 있도록 도와주며, 이 논문은 언어 조건 로봇 모델의 테스트에 레드 팀밍을 적용한 최초의 연구로서 주목받고 있습니다.



### GaussianSpeech: Audio-Driven Gaussian Avatars (https://arxiv.org/abs/2411.18675)
Comments:
          Paper Video: this https URL Project Page: this https URL

- **What's New**: GaussianSpeech는 음성 오디오에서 사진처럼 생생한 개인화된 3D 인간 머리 아바타의 고품질 애니메이션 시퀀스를 합성하는 새로운 접근 방식을 소개합니다. 이 연구는 음성 신호와 3D Gaussian splatting을 결합하여 현실적이고 시간적으로 일관된 모션 시퀀스를 생성하는 방법을 제안합니다. 또한, 표현에 따라 색상 생성 및 주름 검출 기반 손실을 통해 얼굴의 세부 사항을 합성합니다.

- **Technical Details**: GaussianSpeech는 다양한 관점에서 포토리얼리스틱한 3D 아바타 애니메이션을 생성하기 위해 3D Gaussian 포인트를 명시적으로 사용하여 3D 애니메이션을 모델링하는 새로운 방법론을 제안합니다. 이 모델은 음성 입력에서 입술과 표정 특징을 추출하는 오디오 조건부 변환기(transformer) 모델을 활용하여 3D 애니메이션을 최적화합니다. 연구자들은 16대의 카메라로 구성된 멀티뷰(multi-view) 데이터셋을 수집하여 높은 해상도 비디오와 함께 고품질 오디오 데이터를 결합했습니다.

- **Performance Highlights**: GaussianSpeech는 실제 시간 렌더링 속도로 여러 가지 얼굴 표현 및 스타일을 포함하여 시각적으로 자연스러운 모션을 지속적으로 생성합니다. 특히, 이 모델은 기존 애니메이션 기법들보다 우수한 성능을 보이며, 다양한 표현에 따라 동적 주름과 같은 세부적인 디테일도 잘 포착합니다. 새로운 데이터셋은 6명의 원주율(native English) 스피커로 구성되어 있으며, 총 2500개의 시퀀스가 기록되었습니다.



### DHCP: Detecting Hallucinations by Cross-modal Attention Pattern in Large Vision-Language Models (https://arxiv.org/abs/2411.18659)
Comments:
          18 pages, 5 figures

- **What's New**: 이 논문에서는 큰 시각-언어 모델(LVLM)의 환각(hallucination) 문제를 효과적으로 탐지하기 위한 새로운 접근 방식을 제안합니다. 제안된 방법인 DHCP(Detecting Hallucinations by Cross-modal Attention Patterns)는 LVLM의 추가 교육이나 추론 단계 없이도 환각을 탐지할 수 있는 경량 감지기를 개발합니다. 실험 결과는 DHCP가 환각 탐지에서 뛰어난 성능을 보임을 보여주며, LVLM의 신뢰성과 신뢰성을 높이는 데 기여합니다.

- **Technical Details**: LVLM은 일반적으로 CLIP 등을 사용하여 입력 이미지를 시각적 피쳐로 인코딩하는 시각 인코더, 시각-언어 모델, 시각-언어 정렬 모듈의 세 가지 구성 요소로 구성됩니다. 이 논문에서는 환각 상태에서 LVLM이 보이는 시각 토큰에 대한 교차 모드 주의(cross-modal attention) 패턴의 변화를 조사하여 이를 통해 환각을 탐지할 수 있는지 분석하였습니다. 이를 통해, 잘못된 객체를 인식하거나 대답이 불확실한 경우처럼 LVLM이 보이는 주의 패턴을 탐지하여 환각 상태를 파악합니다.

- **Performance Highlights**: DHCP는 심층적인 실험을 통해 여러 데이터 세트에서 시각-언어 모델의 환각 탐지에서 일관되게 높은 성능을 입증했습니다. 허위 경고가 발생한 경우가 있었고, 이들은 모델이 답변에 대한 불확실성을 보였던 경우에 주로 발생했습니다. 또한, 초기 실험 결과는 DHCP가 생성적 작업에서도 효과적으로 적용될 가능성을 시사합니다.



### The Return of Pseudosciences in Artificial Intelligence: Have Machine Learning and Deep Learning Forgotten Lessons from Statistics and History? (https://arxiv.org/abs/2411.18656)
- **What's New**: 이 논문은 심층 학습(Deep Learning) 및 기계 학습(Machine Learning) 기술이 현대 사회에서 얼마나 광범위하게 적용되고 있는지를 강조합니다. 특히, 기계 학습 방법의 설계자와 사용자들이 상관관계와 인과관계의 차이를 간과하고 있다는 점을 지적합니다. 이러한 경향은 인공지능(AI) 시스템의 안전성과 윤리가 심각한 문제라는 사실을 잘 보여줍니다.

- **Technical Details**: 논문은 인과관계를 잘못 할당한 결과로 인생을 바꿀 수 있는 중대한 결정이나 개인의 안전과 프라이버시에 미치는 영향을 다룹니다. 예를 들어, 영국의 OASys(Offender Assessment System)와 같은 범죄자 평가 시스템은 범죄 행동 예측에 AI를 사용하고 있습니다. 이러한 기술이 인종, 성적 지향 등의 요인을 무시할 경우, 심각한 사회적 문제가 발생할 수 있습니다.

- **Performance Highlights**: 오늘날 AI 시스템이 의사결정 최적화 및 비용 감소를 도와주는 반면, 이러한 결정이 불공정하게 이루어질 위험이 존재합니다. AI가 판사나 경감의 역할을 대체할 경우, 그 결과는 명백하지 않고 종종 잘못된 방향으로 나아갈 수 있습니다. 이 논문은 AI의 윤리적 사용을 위해서는 인간의 판단을 유지해야 하며, 보다 공정한 메트릭스를 우선시해야 한다고 주장합니다.



### PRSI: Privacy-Preserving Recommendation Model Based on Vector Splitting and Interactive Protocols (https://arxiv.org/abs/2411.18653)
- **What's New**: 이 논문에서는 개인 정보 보호를 위해 설계된 새로운 추천 시스템인 PRSI(Privacy-preserving Recommendation System)를 제안합니다. 기존의 Federated Recommendation Systems(FedRec)의 보안 문제를 해결하기 위해, 데이터 수집과 추천 결과 전송을 위한 사전 처리 모듈을 포함한 두 가지 주요 단계를 도입했습니다. 이 시스템은 고객의 상호작용 정보를 보호하면서도 효율적으로 추천을 제공합니다.

- **Technical Details**: PRSI는 두 가지 주요 단계로 구성되어 있습니다: (1) 상호작용 정보 수집 및 (2) 추천 결과 전송입니다. 각 클라이언트는 사전 처리 모듈을 사용하고 난수 통신 방법을 통해 ID 정보와 IP 주소를 보호합니다. 추천 결과 전송 단계에서 중앙 서버는 triplets를 사용하여 보안 조건 하에 추천 결과를 각 클라이언트에 배포합니다.

- **Performance Highlights**: 제안된 방법론에 대한 여러 세트의 실험을 수행하여 보안성, 정확성 및 통신 비용을 검증하였습니다. 실험 결과, PRSI는 기존 시스템들에 비해 효과적으로 데이터를 보호하면서도 추천의 질을 유지하는 성능을 보였습니다.



### Dynamic Logistic Ensembles with Recursive Probability and Automatic Subset Splitting for Enhanced Binary Classification (https://arxiv.org/abs/2411.18649)
Comments:
          8 Pages, 2024 IEEE 15th Annual Ubiquitous Computing, Electronics \& Mobile Communication Conference (UEMCON)}. Published in the Proceedings of UEMCON 2024, \c{opyright}2024 IEEE

- **What's New**: 본 논문은 동적 로지스틱 앙상블 모델을 활용한 새로운 이진 분류 접근 방식을 제안합니다. 이 방법은 명시적인 특성 기반 분리가 부족한 데이터 세트에서 발생하는 문제를 해결하며, 전통적인 로지스틱 회귀를 확장하여 데이터 세트를 자동으로 여러 하위 집합으로 나눕니다. 이 과정에서 각 로지스틱 모델의 앙상블을 구축하여 분류 정확도를 향상시키는 것이 목표입니다.

- **Technical Details**: 제안된 모델은 재귀적 확률 계산을 핵심 혁신으로 포함하고 있으며, 대수적 조작과 수학적 귀납을 통해 도출된 방법입니다. 이러한 방식은 앙상블 깊이의 함수로서 최대 우도(maximum likelihood)와 비용 함수를 체계적으로 사용하여 재귀적 그래디언트를 분석적으로 도출하도록 지원합니다. 데이터의 내부 구조에 따라 자동으로 하위 집합을 분할하고, 이를 통해 해석 가능성을 유지하면서 복잡한 패턴을 포착하는 방법론을 소개합니다.

- **Performance Highlights**: 제안된 모델은 노이즈와 데이터를 변형하여 그룹 구조를 시뮬레이션한 맞춤형 데이터 세트에서 검증되었으며, 여러 레이어 구성에 따라 성능이 크게 향상되었습니다. 이 모델은 기존의 Bagging과 Boosting 같은 복잡한 앙상블 방법과 비교했을 때도 해석 가능성을 잃지 않으며, Python으로 구현되어 실제 머신러닝 응용 프로그램에 널리 적용될 수 있는 가능성을 보여줍니다.



### MADE: Graph Backdoor Defense with Masked Unlearning (https://arxiv.org/abs/2411.18648)
Comments:
          15 pages, 10 figures

- **What's New**: 이 논문에서는 Graph Neural Networks (GNNs)가 그래프 관련 작업에서 뛰어난 성능을 보이지만, 백도어 공격(backdoor attacks)에 취약하다는 점을 강조합니다. 연구자들은 훈련 데이터셋에 트리거 패턴을 주입하여 GNN의 예측 결과를 조작할 수 있음을 보여주었습니다. 이 연구는 GNN을 안전하게 지키기 위한 방법론을 제안하며, 특히 전통적인 이미지 방어 기법이 그래프 데이터에 잘 맞지 않는다는 점에 주목합니다.

- **Technical Details**: 저자들은 Graph Neural Networks의 고유한 구조적 특성과 이로 인해 발생하는 특정 문제를 분석합니다. MADE라는 새로운 방어 메커니즘을 제안하며, 이는 부정적인 엣지와 노드를 다루기 위해 독창적인 마스크 생성 메커니즘을 사용합니다. 이 접근 방식은 훈련 데이터셋에 들어간 불순물의 영향을 효과적으로 줄이고, 기존의 정화(cleaning) 데이터셋 사용을 피하면서도 성능을 유지합니다.

- **Performance Highlights**: MADE는 여러 그래프 분류 작업에서 공격 성공률(Attack Success Rate, ASR)을 현저히 줄이는 동시에 높은 분류 정확도를 유지한다는 실험 결과를 도출했습니다. 논문에서는 전통적인 방어 방법들과 비교하여 MADE의 우수성을 입증하였으며, GNN의 훈련 과정에서 그래프 구조의 유용성을 강조합니다.



### Towards Advanced Speech Signal Processing: A Statistical Perspective on Convolution-Based Architectures and its Applications (https://arxiv.org/abs/2411.18636)
- **What's New**: 이 논문에서는 CNN(Convolutional Neural Networks), Conformers, ResNets, 및 CRNNs와 같은 컨볼루션 기반 모델을 조사하여 음성 신호 처리 분야에서의 응용을 설명합니다. 이 연구는 음성 인식, 화자 식별, 감정 인식, 음성 개선과 같은 다양한 적용 분야를 포함하고 있습니다. 저자는 각 모델의 장단점을 비교하고, 통계적 배경을 제공하며, 음성 기술 발전의 중요성을 강조합니다.

- **Technical Details**: 본 논문에서는 컨볼루션 프로세스가 어떻게 음성 신호의 수정에 적용되는지를 설명합니다. 컨볼루션은 두 함수의 곱을 합산하여 또 다른 결과를 생성하는 수학적 과정으로, LTI(Linear Time-Invariant) 시스템 분석에 효과적입니다. 음성 신호 처리에서는 음성 신호와 채널 임펄스 응답 간의 컨볼루션을 통해 다양한 간섭, 반향 및 감쇠의 영향을 모델링합니다.

- **Performance Highlights**: 딥러닝의 최근 발전은 CNN, Conformers, CRNN, ResNets 등의 깊고 컨볼루션 기반의 아키텍처를 가능하게 하여 음성 신호 처리의 정확도를 향상시킵니다. 이들 아키텍처는 노이즈와 변화가 심한 환경에서도 신뢰성 있는 성능을 발휘합니다. 특히 CNN은 속도와 특성 추출에서 이점을 가지며, CRNN은 시간적 동역학을 학습하여 성능을 강화합니다.



### Enhancing Project Performance Forecasting using Machine Learning Techniques (https://arxiv.org/abs/2411.17914)
- **What's New**: 이 연구는 도시 도로 재건축 프로젝트의 성과 지표를 예측하기 위해 머신러닝 기반의 접근 방식을 제안합니다. 기존의 정적 기준 계획에 의존하는 방법들과 달리, 이 접근법은 프로젝트 진행의 동적인 특성과 외부 요인을 고려합니다. 특히, 각 작업 분류 구조(Work Breakdown Structure, WBS) 카테고리마다 비용 편차(cost variance)와 수익 가치(earned value)와 같은 성과 지표를 예측합니다.

- **Technical Details**: 제안된 모델은 자동 회귀 통합 이동 평균(Autoregressive Integrated Moving Average, ARIMA) 및 장기 단기 기억(Long Short-Term Memory, LSTM) 네트워크와 같은 시계열 예측 기법을 활용하여 과거 데이터와 프로젝트 진행에 기반한 미래 성과를 예측합니다. 또한, 모델은 날씨 패턴(weather patterns) 및 자원 가용성(resource availability)과 같은 외부 요인들을 특징(feature)으로 포함하여 예측 정확도를 향상시킵니다.

- **Performance Highlights**: 예측 모델은 머신러닝의 예측력을 활용하여 기준 계획(baseline plan)에서의 잠재적 편차를 사전에 식별할 수 있게 하여 프로젝트 관리자가 적시에 수정 조치를 취할 수 있도록 합니다. 연구는 도시 도로 재건축 프로젝트의 사례 연구를 통해 제안된 접근 방식의 효과성을 검증하려고 하며, 모델의 예측 결과를 실제 프로젝트 성과 데이터와 비교합니다. 이러한 연구 결과는 건설 산업의 프로젝트 관리 관행을 발전시키고, 데이터 기반의 프로젝트 성과 모니터링 및 제어 개선 솔루션을 제공합니다.



### Field Assessment of Force Torque Sensors for Planetary Rover Navigation (https://arxiv.org/abs/2411.04700)
- **What's New**: 본 논문은 행성 탐사 로버에 장착된 힘-토크 센서(force-torque sensors)의 성능과 활용 사례를 평가하며, 다양한 지형에서의 테스트 데이터를 기반으로 합니다. 힘-토크 센서는 로버의 탐사 성능과 주행 성능을 이해하는 데 직접적인 상호작용 힘을 측정할 수 있는 잠재력이 있습니다.

- **Technical Details**: 이 연구에서는 6륜 로버 \\acMaRTA의 필드 실험 데이터를 사용하여 힘-토크 센서의 장착 및 그로 인한 탐사 가능성을 조사합니다. 해당 센서는 로버의 바퀴 위에 장착되어 있으며, 이를 통해 지형 분류 및 실제 데이터에 기반한 끌어당기는 힘(drawbar pull) 추정에 대한 적합성을 평가합니다.

- **Performance Highlights**: 이 연구는 힘-토크 센서를 통해 로버가 비구조적 지형에서 탐사 작용을 더욱 개선할 수 있는 가능성에 대한 통찰력을 제공하며, 센서 통합 및 제어 알고리즘과 관련된 미래 연구의 방향성을 제안합니다. 이를 통해 로버의 내비게이션 능력을 향상시킬 수 있음을 강조합니다.



### Inference Scaling fLaws: The Limits of LLM Resampling with Imperfect Verifiers (https://arxiv.org/abs/2411.17501)
- **What's New**: 이 논문은 약한 언어 모델이 강한 모델과 동일한 정확도를 얻기 위해 반복 샘플링을 사용하여 문제를 해결할 수 있다는 희망을 제시합니다. 하지만 이 과정의 핵심 주장은, '검증기'(verifier)가 불완전할 경우 무한한 정확도 향상은 불가능하다는 것입니다. 특히, 검증기가 불완전할 경우 잘못된 솔루션이 실제로 통과할 확률이 존재하여 샘플링의 유효성에 제한이 있다는 점을 강조합니다.

- **Technical Details**: 이 연구는 언어 모델의 성능 향상에서 일반화 격차(generalization gap)에 대한 우려를 담고 있습니다. 특정 검증 기법인 유닛 테스트(unit tests)가 모달리티별로 정확성을 평가할 수 있지만, 약한 모델이 강한 모델의 성능을 맞추는데 제한적이라는 사실을 발견했습니다. 무한한 리소스가 있더라도, 약한 모델이 강한 모델의 성능을 한 번의 실행으로 따라잡는 경우는 드물다는 결론을 내렸습니다.

- **Performance Highlights**: 최적 샘플링 수는 10 미만으로, 잘못된 솔루션이 반환될 때의 부정적 유틸리티(utility)가 최적 샘플 수에 미치는 영향을 분석합니다. 과도한 업데이트는 성능 향상에 제한이 있으며, 잘못된 출력이 발생할 확률이 높은 경우, 약한 모델은 사실상 유용하지 않을 수 있습니다. 출력 품질에 대해서도 잦은 잘못된 답변이 코드의 전반적인 품질을 떨어뜨린다는 것을 발견하며, 이는 코딩 스타일 고수나 가독성 지표에서 명확하게 드러납니다.



### Neuro-Symbolic Evaluation of Text-to-Video Models using Formal Verification (https://arxiv.org/abs/2411.16718)
- **What's New**: 최근 텍스트-비디오 생성 모델(Sora, Gen-3, MovieGen, CogVideoX)의 발전이 합성 비디오 생성의 경계를 확장하고 있습니다. 이러한 모델은 로봇공학, 자율주행, 엔터테인먼트와 같은 다양한 분야에서 활용되고 있으며, 비디오 품질을 평가하기 위한 다양한 메트릭과 기준이 등장하고 있습니다. 그러나 기존 메트릭은 시각적 품질과 부드러움에 중점을 두고, 안전이 중요한 응용 분야에 필요한 시간적 일관성 및 텍스트-비디오 정렬을 간과하고 있습니다.

- **Technical Details**: NeuS-V는 텍스트-비디오 정렬을 엄격하게 평가할 수 있는 새로운 합성 비디오 평가 메트릭입니다. 우리의 접근 방식은 프롬프트를 정형화된 Temporal Logic (TL) 명세로 변환하고, 생성된 비디오를 오토마타 표현으로 번역하는 과정으로 진행됩니다. 그런 다음, 오토마타를 TL 명세와 비교하여 비디오의 텍스트-비디오 정렬을 형식적으로 검사합니다. 이러한 방식은 비디오 품질 평가에 있어 보다 엄격하고 해석 가능하게 만들어줍니다.

- **Performance Highlights**: NeuS-V는 기존 메트릭과 비교할 때 인간 평가와의 상관관계가 5배 이상 높은 것으로 나타났습니다. 해당 평가 결과는 현재의 비디오 생성 모델들이 시간적으로 복잡한 프롬프트에 대해 부족한 성능을 보이고 있음을 드러내며, 텍스트-비디오 생성 능력을 향상하기 위한 향후 연구의 필요성을 강조합니다. 이를 통해 우리는 합성 비디오 생성 모델의 잠재력 및 필요성을 더욱 분명히 확인할 수 있었습니다.




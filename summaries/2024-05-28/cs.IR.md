### DeeperImpact: Optimizing Sparse Learned Index Structures (https://arxiv.org/abs/2405.17093)
- **What's New**: 이번 논문에서는 기존의 DeepImpact 모델을 최적화하여 DeeperImpact 모델을 소개합니다. 이 모델은 더 뛰어난 문서 확장 방법과 훈련 전략을 사용하며, SPLADE 모델과의 성능 격차를 줄이는 데 중점을 두고 있습니다.

- **Technical Details**: DocT5Query를 사용하는 기존의 문서 확장 방식은 종종 비관련된 내용의 확장으로 인해 한계가 있었습니다. 이를 해결하기 위해 Llama 2 모델을 사용하여 문서 확장을 최적화하였습니다. Llama 2는 Meta에서 개발한 최신 대형 언어 모델로, 70억 개의 파라미터를 가지고 있으며, LoRA를 활용한 미세 조정과 8비트 양자화를 통해 메모리와 계산 효율을 극대화했습니다. 또한, CoCondenser 모델의 초기화, hard negatives 활용, 그리고 distillation 기법을 적용하여 훈련 과정을 최적화했습니다.

- **Performance Highlights**: DeeperImpact 모델은 최신 버전의 SPLADE 모델과 비교했을 때 상당한 성능 개선을 보여줍니다. 특히, 문서 확장 과정에서 Llama 2 모델을 사용함으로써, DocT5Query 대비 더 높은 품질의 결과를 도출할 수 있었습니다. 이 과정에서 Doc2Query 필터링 메커니즘을 통해 DocT5Query의 한계를 일부 극복할 수 있음을 확인했습니다.



### Robust kernel-free quadratic surface twin support vector machine with capped $L_1$-norm distance metric (https://arxiv.org/abs/2405.16982)
- **What's New**: 본 논문에서는 전통적인 Twin Support Vector Machine (TSVM)의 두 가지 주요 한계를 해결하기 위해, 새로운 강화된 Caped L_1-Norm을 적용한 커널-프리 (Kernel-Free) 이차 표면 TSVM (CL_1QTSVM)을 제안했습니다. 첫째, 이 모델은 L_2-Norm 대신 Caped L_1-Norm 거리 지표를 사용하여 이상치 (Outliers)에 대한 민감성을 줄입니다. 둘째, 커널 함수와 그 파라미터를 선택할 필요가 없어 시간 소모적인 과정을 회피할 수 있습니다.

- **Technical Details**: 제안된 CL_1QTSVM 모델의 주요 개선 사항은 다음과 같습니다. 1) Caped L_1-Norm 거리 지표를 활용하여 모델의 강건성을 향상시켰습니다. 2) 커널-프리 방법을 채택하여 적절한 커널 함수와 파라미터를 선택하는 시간 소모적 과정을 피했습니다. 3) 모델의 일반화 능력을 향상시키기 위해 L_2-Norm 정규화 항을 도입했습니다. 4) 제안된 모델을 효율적으로 해결하기 위해 반복 알고리즘을 개발했습니다. 5) 제안된 알고리즘의 수렴성, 시간 복잡성 및 국소 최적 해의 존재성을 이론적으로 분석했습니다.

- **Performance Highlights**: 다양한 데이터셋에서 수행된 수치 실험 결과는 제안된 CL_1QTSVM 모델이 기존 최첨단 방법들과 비교하여 강건성 측면에서 약간 더 나은 성과를 보임을 확인하였습니다.



### Multi-Behavior Generative Recommendation (https://arxiv.org/abs/2405.16871)
- **What's New**: 이번 논문에서는 다중 행동 순차 추천(MBSR)을 위한 새로운 모델인 MBGen을 제안합니다. MBGen은 아이템 시퀀스와 행동 타입을 활용하여 사용자의 다음 행동 의도를 예측하고, 이를 기반으로 다음 아이템을 추천합니다. 이를 통해 MBSR 문제를 새로운 생성적 추천 패러다임으로 재정의합니다.

- **Technical Details**: MBGen은 두 단계를 거쳐 추천 작업을 수행합니다. 첫 번째 단계에서는 주어진 아이템 시퀀스를 기반으로 다음 행동 타입을 예측하고, 두 번째 단계에서는 예측된 행동 타입과 아이템 시퀀스를 활용하여 다음 아이템을 예측합니다. 이 모델은 행동과 아이템을 토큰화하고, 이들을 교차로 배열하여 하나의 시퀀스를 구성합니다. 또한, 위치 라우팅을 통한 희소 아키텍처를 도입하여 모델의 확장성을 향상시켰습니다.

- **Performance Highlights**: 공개 데이터셋을 이용한 광범위한 실험을 통해 MBGen이 기존 MBSR 모델에 비해 31%에서 70%까지 성능이 향상됨을 확인했습니다. MBGen은 행동 타입과 아이템 간의 종속성을 효과적으로 모델링하고, 자동회귀적 학습을 통해 강력한 다중 작업 능력을 갖추었습니다.



### NoteLLM-2: Multimodal Large Representation Models for Recommendation (https://arxiv.org/abs/2405.16789)
Comments:
          19 pages, 5 figures

- **What's New**: 이번 연구에서는 대형 언어 모델(LLMs)을 활용하여 I2I 추천 시스템 내에서 멀티모달 표현을 향상시키는 방법을 탐구합니다. 기존의 멀티모달 대형 언어 모델(MLLMs)은 높은 품질의 대규모 멀티모달 데이터를 수집해야 하며, 이는 복잡하고 비용이 많이 드는 과정을 수반합니다. 이러한 문제를 해결하고자, 우리는 어떤 기존 LLMs과 비전 인코더를 통합하여 효율적인 멀티모달 표현 모델을 구축하는 엔드 투 엔드(end-to-end) 학습 방법을 제안합니다.

- **Technical Details**: 노트LLM-2(NoteLLM-2)라는 새로운 학습 프레임워크를 제안하며, 이는 멀티모달 표현을 위해 두 가지 방법을 사용합니다. 첫 번째는 프롬프트 관점을 바탕으로 멀티모달 In-Context Learning(mICL)방법을 적용하여 시각적 및 텍스트적 정보를 분리하고, 각각의 정보에 대해 일괄 대조 학습을 진행합니다. 두 번째 방법은 후반 융합 메커니즘(late fusion mechanism)을 활용하여 시각적 정보를 텍스트 정보에 직접 융합하는 것입니다.

- **Performance Highlights**: 우리의 방법론인 NoteLLM-2의 효과를 입증하기 위해 광범위한 실험을 수행했습니다. 그 결과, 멀티모달 정보가 적절히 융합되어 성능이 크게 향상됨을 확인할 수 있었습니다. 특히, 제안한 mICL 및 후반 융합 메커니즘이 시각적 정보의 손실을 최소화하고, 더 효과적인 멀티모달 표현을 생성하는데 중요한 역할을 했습니다.



### ReCODE: Modeling Repeat Consumption with Neural ODE (https://arxiv.org/abs/2405.16550)
Comments:
          Accepted by SIGIR 2024 (Short Paper)

- **What's New**: 최근 발표된 ReCODE 모델은 반복 소비(repeat consumption)를 효과적으로 모델링하기 위해 제안된 새로운 접근법입니다. ReCODE는 뉴럴 보통 미분방정식(ODE, Ordinary Differential Equations)을 활용하여 복잡한 소비 패턴을 캡처하기 때문에 기존의 사전 정의된 분포에 의존하지 않습니다.

- **Technical Details**: ReCODE 프레임워크는 두 가지 주요 모듈로 구성되어 있습니다. 첫째, 사용자의 기본 선호도를 캡처하는 정적 추천 모듈은 다양한 아이템에 대한 사용자의 기본 선호도를 예측합니다. 둘째, 각 사용자-아이템 쌍에 대해 반복 소비의 시간 패턴을 모델링하는 동적 반복 인식 모듈이 포함되어 있습니다. 이 접근 방식은 사용자의 정적 선호도와 동적 반복 의도를 모두 고려하여 포괄적인 사용자 선호도 모델링을 제공합니다. ReCODE는 또한 협업 필터링 기반 모델 및 순차 기반 모델 등 다양한 기존 추천 모델에 손쉽게 통합될 수 있도록 설계되었습니다.

- **Performance Highlights**: 두 개의 실제 데이터셋을 사용한 실험 결과, ReCODE는 기존의 기본 모델 성능을 긴 시간 동안 지속적으로 향상시켰으며, 반복 소비를 처리하는 기존의 타 모델들보다 우수한 성과를 보였습니다.



### LLMs for User Interest Exploration: A Hybrid Approach (https://arxiv.org/abs/2405.16363)
- **What's New**: 전통적인 추천 시스템의 강력한 피드백 루프 문제를 해결하기 위해 대규모 언어 모델(LLMs)과 고전적인 추천 모델을 결합한 하이브리드 계층적 프레임워크를 소개합니다. 이 프레임워크는 '관심 클러스터(interest clusters)'를 사용해 LLMs와 고전적인 추천 모델 간의 인터페이싱을 제어하며, 알고리즘 설계자가 클러스터의 세분성을 명확하게 결정할 수 있습니다. 관심 클러스터를 언어로 표현하고, 세밀하게 조정된 LLM을 사용해 이러한 클러스터 내에서 엄격하게 새 관심사(interest)를 생성합니다. 핵심은 사용자의 새 관심사를 탐색하는 것입니다.

- **Technical Details**: 상위 레벨에서, LLM은 시스템의 방대한 항목 수를 고려해 직접 다음 항목을 예측하는 대신, 새로운 관심사를 추론합니다. 낮은 레벨에서는 고전적인 추천 모델의 강력한 개인화 기능을 활용하여 생성된 관심사를 특정 '클러스터' 내의 항목 추천으로 연결합니다. 이 과정은 LLM의 추론과 일반화 능력을 활용하여, 사용자의 새로운 관심사를 효과적으로 탐색하는 동시에 도메인별 모델의 세부 지식으로 실제 항목 추천을 수행합니다. 또한 실제 사용자와의 상호작용 데이터를 사용해 LLM을 감독 및 미세 조정(supervised fine-tuning)하여 LLM이 사전에 정의된 '클러스터'와 일치하는 새로운 관심사를 생성하도록 합니다.

- **Performance Highlights**: 산업 규모의 상업적 플랫폼에서 수십 억명의 사용자에 대한 라이브 실험 결과, 사용자의 새로운 관심사를 탐색하고 플랫폼에 대한 사용자의 전반적인 만족도를 크게 향상시킨다는 것을 보여줬습니다. 사용자의 활동성이 증가하고 체류 시간이 길어진 것으로 나타났습니다.



### Finetuning Large Language Model for Personalized Ranking (https://arxiv.org/abs/2405.16127)
- **What's New**: 이번 연구에서는 큰 언어 모델(LLMs)의 추천 시스템 적용에 대한 새로운 프레임워크, Direct Multi-Preference Optimization (DMPO)를 소개합니다. DMPO는 여러 부정적인 샘플의 확률을 최소화하면서 긍정적인 샘플의 확률을 극대화함으로써 LLM 기반 추천 시스템의 성능을 향상시킵니다. 이를 통해 LLMs가 다양한 도메인에 걸쳐 추천 능력을 개선하는 것을 목표로 합니다.

- **Technical Details**: DMPO는 Direct Preference Optimization(DPO)의 한계를 극복하기 위해 개발되었습니다. DPO는 한 사용자의 상호작용 목록에서 한 개의 긍정적 샘플과 한 개의 부정적 샘플만을 고려하였으나, DMPO는 여러 부정적 샘플을 도입하여 긍정적 샘플과 부정적 샘플 간의 더 포괄적이고 균형 있는 관계를 설정합니다. 이는 대조 학습(contrastive learning) 방법에서 부정적 샘플링 전략을 차용하여 강화되었습니다.

- **Performance Highlights**: 세 개의 실제 데이터셋(‘Movielens-1M’, ‘Amazon Movies and TV’, ‘Amazon Video Games’)에서 DMPO를 통해 AUC 성능이 크게 향상되었습니다. 비교 실험에서 DMPO는 전통적인 추천 시스템과 기존의 LLM 기반 방법들 모두를 능가함을 입증했습니다. 또한, DMPO는 다양한 도메인 간의 추천 성능에서도 뛰어난 일반화 능력을 보였습니다.



### BankFair: Balancing Accuracy and Fairness under Varying User Traffic in Recommender System (https://arxiv.org/abs/2405.16120)
- **What's New**: 최신 연구에서 추천 시스템이 사용자와 제공자 양측의 요구를 모두 만족시키려는 'BankFair'라는 새로운 재랭킹(re-ranking) 접근법을 제안했습니다. 이는 경제학의 파산 문제를 응용한 것으로, 사용자 트래픽이 변동하는 상황에서도 일관된 사용자 서비스와 장기적인 제공자 공정성을 유지할 수 있도록 구성되었습니다.

- **Technical Details**: BankFair는 탈무드 규칙(Talmud rule)을 사용해 사용자 트래픽이 높은 시기에 공정성을 더 부여하고, 트래픽이 낮은 시기에 이를 보완하여 공정성을 유지합니다. 두 가지 주요 모듈로 구성되어 있으며, 첫째 모듈은 탈무드 규칙을 통해 각 기간별 공정성의 필요 수준을 결정하고, 둘째 모듈은 이 공정성 수준을 기반으로 온라인 재랭킹 알고리즘을 구현하여 즉각적인 사용자 정확성과 장기적인 제공자 공정성을 모두 보장합니다.

- **Performance Highlights**: 공개된 데이터셋과 실제 산업 데이터셋에 대한 실험 결과, BankFair는 정확성과 제공자 공정성 측면에서 모두 기존의 다른 방법들보다 뛰어남을 보였습니다. 이는 변동하는 사용자 트래픽에도 불구하고 높은 수준의 사용자 경험을 유지하며 장기적인 제공자 노출을 보장할 수 있는 잠재력을 증명합니다.



### IQLS: Framework for leveraging Metadata to enable Large Language Model based queries to complex, versatile Data (https://arxiv.org/abs/2405.15792)
- **What's New**: IQLS(Intelligent Query and Learning System)은 복잡한 데이터를 자연어로 쉽게 조회할 수 있도록 해주는 시스템입니다. 특히 물류 산업에서 데이터 수집 기술이 발전함에 따라 실시간으로 엄청난 양의 상호 연결된 데이터를 다루는 데 있어 이 시스템이 유용합니다.

- **Technical Details**: IQLS는 구조화된 데이터를 메타데이터와 데이터 모델을 기반으로 프레임워크로 매핑합니다. 이 프레임워크는 Large Language Model(언어 모델)로 구동되는 에이전트 환경을 만들어줍니다. 에이전트는 데이터를 한 번에 요청하는 대신, 여러 작은 컨텍스트 인지 결정을 통해 반복적으로 데이터를 필터링합니다. 필터링 후, IQLS는 사용자 쿼리에 따라 멀티모달 운송 정보 조회, 여러 제약 조건 하에서의 경로 계획 등 다양한 인터페이스를 통해 에이전트가 작업을 수행할 수 있게 합니다.

- **Performance Highlights**: IQLS는 캐나다 물류 분야의 사례 연구에서 지리 정보, 시각적 정보, 표형 데이터, 텍스트 데이터를 자연어로 쉽게 쿼리할 수 있도록 시연되었습니다. 사용자는 전 과정에서 시스템과 상호작용하며 이를 이끌어갈 수 있습니다.



### Learning-based models for building user profiles for personalized information access (https://arxiv.org/abs/2405.15791)
Comments:
          21 pages, 18 figures

- **What's New**: 이 연구는 문서 내용과 정보 요구를 표현하는 데 사용되는 어휘의 차이를 고려함으로써 문헌에 기여합니다. 사용자는 모든 연구 단계에 통합되어 그들의 맥락과 선호에 맞춰 적합한 정보를 제공받습니다.

- **Technical Details**: 이 단계에서 문서 내용과 정보를 더 잘 표현하기 위해 딥 러닝(deep learning) 모델이 사용됩니다. 이러한 모델은 문서와 쿼리의 복잡한 표현을 학습할 수 있으며, 텍스트 데이터에서 계층적(hierarchical), 순차적(sequential), 또는 주의기반(attention-based) 패턴을 포착할 수 있습니다.

- **Performance Highlights**: 이 접근법은 사용자의 맥락과 우선순위에 적합한 정보 제공에 있어 더 정교한 분석과 결과를 도출할 수 있습니다.



### Towards Fairness in Provably Communication-Efficient Federated Recommender Systems (https://arxiv.org/abs/2405.15788)
- **What's New**: 이번 연구에서는 다중 클라이언트의 병렬 학습으로 인한 통신 오버헤드를 줄이기 위해 랜덤 클라이언트 샘플링을 사용한 다양한 연합 학습(Federated Learning, FL) 기법을 탐구했습니다. 특히, 연합 추천 시스템(FRS)에서의 샘플링 효율성과 최적의 클라이언트 수 결정을 위해 샘플 복잡성 경계를 확립하여 통신 효율성을 향상시키면서도 정확성을 유지하는 방법을 제안했습니다. 또한 클라이언트 간의 클래스 불균형 문제를 해결하기 위해 RS-FairFRS를 도입하였으며, 이는 보호 속성을 공개하지 않고 공정성을 달성할 수 있는 두 단계의 듀얼-페어 업데이트 기법을 특징으로 합니다.

- **Technical Details**: 본 연구에서는 FedRec 모델을 사용하여 로컬 클라이언트에서 매트릭스 팩토라이제이션(MF)으로 학습하고, 사용자 개인 데이터의 교환 없이 서버에 업데이트를 전송하는 방식을 제안했습니다. 이 모델에서는 클라이언트가 서버와 아이템 그래디언트만 통신합니다. 연구의 핵심 기여로는 클라이언트 샘플링의 최적 비율을 이론적으로 규명하고, 두 단계의 듀얼-페어 업데이트 기술을 통해 공정성을 달성하는 RS-FairFRS 모델을 도입한 점이 있습니다.

- **Performance Highlights**: 실제 데이터셋(ML1M, ML100K)을 사용한 실험 결과, RS-FairFRS는 통신 비용을 약 47% 감소시키면서도 정확도를 유지함을 보였습니다. 또한, 다양한 인구 통계적 특징(연령, 성별)에 대해 약 40%의 인구 통계적 편향 감소를 통해 신뢰성 있는 공정성을 달성하였습니다.



### Enhancement of Subjective Content Descriptions by using Human Feedback (https://arxiv.org/abs/2405.15786)
- **What's New**: 새로운 정보 검색 시스템 ReFrESH가 소개되었습니다. ReFrESH는 주관적 콘텐츠 설명(SCDs)을 업데이트하여 사용자의 피드백을 반영하고, 문서의 내용을 보다 정확하게 검색할 수 있도록 돕는 기술입니다. 기존 SCDs는 사용자의 주관적 시각을 반영할 수 있어, 사용자가 제공한 피드백을 통해 이를 개선하는 기능을 갖추고 있습니다.

- **Technical Details**: ReFrESH는 사용자가 제공한 피드백을 기반으로 SCD를 점차적으로 개선합니다. 이 시스템은 관계 유지(Relation-preserving) 알고리즘을 사용하여 SCD와 문서 간의 관계를 수정 및 보완합니다. 또한, 명확하지 않은 피드백조차도 축적하여 SCD를 업그레이드하는 데 이용할 수 있는 기술입니다. 이를 통해 SCD의 관계를 보존하면서도 불량한 관계를 교정하고, 새로운 피드백을 효과적으로 받아들일 수 있습니다.

- **Performance Highlights**: ReFrESH는 불명확한 피드백을 포함한 모든 종류의 사용자 피드백을 효율적으로 반영합니다. 이를 통해 더 정확하고 사용자 맞춤형 정보를 제공하여 정보 검색 효율성을 높입니다. 또한, 기존 모델을 부분적으로 업데이트함으로써 전체 모델을 재훈련하는 비용을 절감할 수 있습니다.



### Multimodality Invariant Learning for Multimedia-Based New Item Recommendation (https://arxiv.org/abs/2405.15783)
- **What's New**: 새로운 논문에서는 새로운 아이템을 빠르게 추천하고, 현실 세계에서는 다양한 정도로 모달리티가 누락되는 문제를 해결하기 위해 'MILK (Multimodality Invariant Learning reCommendation)' 프레임워크를 제안합니다. 이 프레임워크는 사용자의 고유한 콘텐츠 선호도를 서로 다른 모달리티 누락 환경에서도 일관되게 유지하는 것을 목표로 하고 있습니다.

- **Technical Details**: MILK는 두 가지 모듈로 구성됩니다. 첫 번째는 '크로스-모달리티 정렬 모듈(cross-modality alignment module)'로, 사전 학습된 멀티미디어 항목 기능에서 의미론적 일관성을 유지합니다. 두 번째 모듈은 '크로스-환경 불변성 모듈(cross-environment invariance module)'로, 사이클 혼합(cyclic mixup)을 통해 트레이닝 데이터를 확장하여 어떤 모달리티 누락 환경에서도 사용자의 선호도를 학습할 수 있도록 합니다.

- **Performance Highlights**: 세 개의 실제 데이터셋에서의 광범위한 실험을 통해 MILK 프레임워크의 우수성을 검증했습니다. 특히, 새로운 아이템에 대한 추천 성능과 다양한 모달리티 누락 상황에서의 정확한 선호도 예측 능력이 뛰어난 것으로 나타났습니다.



### Cypher4BIM: Releasing the Power of Graph for Building Knowledge Discovery (https://arxiv.org/abs/2405.16345)
- **What's New**: 새로운 IFC(Industry Foundation Classes) 데이터의 그래픽 형태인 IFC-Graph가 개발되었습니다. 이번 연구에서는 그래픽 빌딩 정보 쿼리와 IFC-Graph를 위한 맞춤형 그래프 쿼리 언어를 개발했습니다. 이를 Cypher4BIM이라 명명했습니다.

- **Technical Details**: 연구에서는 IFC 데이터의 구조와 주요 정보 유형을 조사하고, 그래프 쿼리 언어인 Cypher를 분석한 후, 맞춤형 기능 쿼리 패턴을 개발했습니다. 이 과정에서 다섯 개의 IFC 모델을 사용하여 유효성을 검증했습니다.

- **Performance Highlights**: Cypher4BIM은 개별 인스턴스와 공간 구조, 공간 경계 및 공간 접근성 같은 IFC에서 복잡한 관계를 효과적으로 쿼리할 수 있음을 보여주었습니다. 이는 디지털 트윈과 같은 효과적인 빌딩 정보 쿼리를 필요로 하는 응용 프로그램에 기여할 수 있을 것입니다.




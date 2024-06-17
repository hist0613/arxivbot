### Harm Mitigation in Recommender Systems under User Preference Dynamics (https://arxiv.org/abs/2406.09882)
Comments:
          Recommender Systems; Harm Mitigation; Amplification; User Preference Modeling

- **What's New**: 이 논문에서는 사용자 관심의 진화와 해로운 콘텐츠와의 상호작용을 고려한 추천 시스템을 제안합니다. 추천이 사용자 행동에 미치는 영향을 모델링하고, 클릭률(CTR)을 최대화하면서 해로움을 줄일 수 있는 추천 정책을 찾고자 합니다. 특히, 단순한 접근 방식으로는 해로움이 완화되지 않을 수 있으며, 사용자의 선호도 동태를 무시하면 최적의 추천 정책을 도출하기 어려울 수 있음을 증명합니다.

- **Technical Details**: 이 연구는 사용자의 선호도 동태를 고려하는 모델을 바탕으로 합니다. 특히, 사용자의 선택 확률에 비례하는 점수를 이용해 항목을 순위 매기고, 상위 k개의 항목을 추천하는 top-k 추천 알고리즘을 사용합니다. 하지만, 이러한 접근법은 콘텐츠 다양성 감소, 편향성 증대, 잘못된 정보 확산 등의 부정적인 효과를 유발할 수 있습니다. 이를 해결하기 위해 사용자의 선호도 동태를 명시적으로 모델링하고 최적의 추천을 찾기 위한 알고리즘을 제안합니다. 본 연구는 다항-로짓(MNL) 모델을 기반으로 하며, 고정 점을 특성화하고 암묵 함수 정리를 사용해 기울기를 계산하는 알고리즘을 제안합니다.

- **Performance Highlights**: 반정합 영화 추천 설정에서 실험을 통해 제안된 정책이 기존의 기준선들에 비해 CTR을 최대화하면서 동시에 해로움을 줄이는 데 훨씬 뛰어난 성능을 보였습니다. 실험 결과, 최고 77%의 성능 향상을 달성했습니다.



### ClimRetrieve: A Benchmarking Dataset for Information Retrieval from Corporate Climate Disclosures (https://arxiv.org/abs/2406.09818)
- **What's New**: 기업의 기후 변화를 다루는 커뮤니케이션에서 생성된 방대한 양의 정성적 데이터를 처리하기 위해, 이해관계자들은 점점 더 Retrieval Augmented Generation (RAG) 시스템에 의존하고 있습니다. 그러나 도메인 특화 정보 검색을 평가하는 데 있어 중요한 격차가 여전히 존재합니다. 이 논문은 16개의 기후 관련 질문을 통해 30개의 지속 가능성 보고서를 분석하여 도메인 특화 정보 검색의 평가 방법을 제안합니다. 그 결과, 8.5K 이상의 질문-소스-답변 데이터셋을 구성하여 다양한 수준의 관련성을 표시하였습니다. 또한, 전문가 지식을 정보 검색에 통합하는 방법을 조사하기 위해 데이터셋을 이용한 사용 사례를 개발했습니다.

- **Technical Details**: 이 연구는 지속 가능성 분석가의 일반적인 작업을 시뮬레이션하여 데이터를 생성합니다. 질문과 해당하는 출처를 바탕으로 답변을 구성하여 데이터셋을 만듭니다. 조사 과정에서 SOTA 임베딩 모델이 도메인 전문성을 효과적으로 반영하지 못함을 확인하였습니다. 이를 통해 정보 검색에 새로운 접근법이 필요함을 강조합니다.

- **Performance Highlights**: 이 연구는 RAG 시스템의 도메인 특화 검색 성능을 평가하는 데 중요한 기여를 합니다. 전문가 지식을 정보 검색 과정에 통합하는 것은 비직관적이며, RAG 시스템의 실제 성능을 개선하는 데 주목할만한 영향을 미칩니다. 이 데이터셋은 새로운 접근방식을 평가하는 기준이 될 수 있습니다. 전문가 주석 데이터를 바탕으로 한 실험 결과, 임베딩 전략이 도메인의 특성을 완전히 반영하지 못함을 보여줍니다. 이는 기후변화와 같은 지식 집약적 도메인의 검색 성능을 향상시키기 위한 새로운 연구 필요성을 시사합니다.



### Soil nitrogen forecasting from environmental variables provided by multisensor remote sensing images (https://arxiv.org/abs/2406.09812)
- **What's New**: 이 연구는 멀티-센서 원격 감지 이미지와 고급 머신러닝 기법을 활용해 토양 질소 함량을 예측하는 프레임워크를 소개합니다. 특히 유럽 및 영국 지역을 커버하는 LUCAS 데이터베이스를 위성 센서를 통한 환경 변수와 결합해 혁신적인 특징의 데이터셋을 생성했습니다. 다양한 토지 피복 클래스에서의 테스트 결과, CatBoost 모델이 다른 방법들보다 정확도가 뛰어난 것으로 나타났습니다. 이 연구는 농업 관리 및 환경 모니터링 분야에서의 중요한 진전을 보여줍니다.

- **Technical Details**: 이 연구는 시작 전, 토양 질소 값의 왜곡된 분포를 해결하기 위해 정규화 과정을 거쳤습니다. 기계 학습 모델 훈련을 위해 LUCAS 데이터베이스와 구글 어스 엔진을 사용해 84개의 특징을 가진 21244개의 샘플을 포함한 다중 모드 데이터셋을 준비했습니다. CatBoost, LightGBM, XGBoost 등 다양한 트리 기반 모델들이 비교되었으며, 최고 성능 지표로 RMSE을 사용하였습니다. CatBoost 모델은 베이지안 최적화 및 5격 교차검증을 통해 하이퍼파라미터를 튜닝하였고, 최종 모델은 전체 데이터셋으로 훈련받았습니다.

- **Performance Highlights**: CatBoost 모델은 다른 모델들에 비해 예측 정확도가 높았으며, 다양한 특징 및 하이퍼파라미터 설정에서도 우수한 성능을 보였습니다. 'MAPE_total'과 'MAE_crop'에서 가장 낮은 오류율을 나타내었습니다. LightGBM은 하이퍼파라미터 최적화 후 MAPE_total에서 최상의 성능을 기록했습니다. XGBoost와 ExtraTrees 모델도 성능이 개선되었으나, CatBoost와 LightGBM의 성능에 미치지 못했습니다. R² 값은 모든 모델이 약 0.5 정도로 비슷했으며, CatBoost가 전체적으로 가장 낮은 MAPE를 보였습니다.



### IFA: Interaction Fidelity Attention for Entire Lifelong Behaviour Sequence Modeling (https://arxiv.org/abs/2406.09742)
Comments:
          7 pages, 2 figures

- **What's New**: 이 연구에서는 사용자 평생 행동 시퀀스(lifelong user behavior sequence)를 효율적으로 모델링하는 새로운 패러다임, Interaction Fidelity Attention(IFA)를 제안합니다. 기존의 추천 시스템은 대상 아이템과 유사한 짧은 하위 시퀀스를 샘플링하여 계산 복잡도를 줄였지만 정보 손실이 발생했습니다. IFA는 전체 후보 집합(candidate set)의 타겟 아이템을 모델에 한 번에 입력하고, 크로스 어텐션(cross attention)의 시간 복잡도를 줄여 상호작용 정보 손실 없이 성능을 유지합니다.

- **Technical Details**: IFA 모델에서는 모든 타겟 아이템을 한 번에 입력하고, 선형 트랜스포머(linear transformer)를 활용해 후보 집합과 시퀀스 간의 크로스 어텐션의 시간 복잡도를 감소시킵니다. 또한, 최적의 후보 집합 생성을 위해 모든 타겟 아이템 간의 관계를 추가적으로 모델링하고, 트레이닝과 추론의 일관성을 높이기 위해 손실 함수를 설계했습니다.

- **Performance Highlights**: 제안된 IFA 모델은 오프라인과 온라인 실험에서 Kuaishou 추천 시스템의 효과와 효율성을 입증했습니다. 기존의 샘플링 기반 접근 방식과 비교하여, 정보 손실을 줄이면서 계산 복잡도를 효과적으로 관리하는 데 성공했습니다.



### Enhancing Knowledge Retrieval with In-Context Learning and Semantic Search through Generative AI (https://arxiv.org/abs/2406.09621)
- **What's New**: 이번 연구에서는 Domain-Specific(도메인 특화) 질문에 대한 정확한 응답을 제공하는 더 나은 지식 검색 시스템을 제안하였습니다. Generative Text Retrieval(GTR) 모델을 통해 LLMs(대형 언어 모델)와 벡터 데이터베이스를 통합하여 효율적이고 정확한 데이터 검색을 실현했습니다. 이를 통해 탭 테이블 데이터와 비구조화된 데이터 모두를 처리할 수 있습니다.

- **Technical Details**: GTR 모델은 대형 언어 모델의 생성 능력과 벡터 데이터베이스의 빠르고 정확한 검색 능력을 결합하여 구축되었습니다. 모델은 탭 테이블 데이터와 비구조화된 데이터를 포함한 대규모 데이터베이스를 자연어 사용자 쿼리를 통해 이해하고 필요한 정보를 검색합니다. Fine-tuning(파인 튜닝)을 하지 않아도 모델이 고효율적으로 동작합니다.

- **Performance Highlights**: 수동 주석 및 공공 데이터 세트에서 GTR 모델은 90% 이상의 정확도를 달성하였으며, MS MARCO 데이터 세트에서는 Rouge-L F1 점수 0.98로 최고 성능을 입증하였습니다. 또한, Spider 데이터 셋에서 Execution Accuracy (EX) 0.82와 Exact-Set-Match(EM) 정확도 0.60을 달성하였습니다.



### Unraveling Anomalies in Time: Unsupervised Discovery and Isolation of Anomalous Behavior in Bio-regenerative Life Support System Telemetry (https://arxiv.org/abs/2406.09825)
Comments:
          12 pages, + Supplemental Materials, Accepted at ECML PKDD 2024 (European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases)

- **What's New**: 시스템 상태 모니터링에서 비정상적이거나 중요한 상태를 감지하는 것은 필수적입니다. 본 연구에서는 우주탐사용 바이오-재생 생명유지 시스템 (BLSS: Bio-Regenerative Life Support Systems) 내 비정상 상태를 분석합니다. EDEN ISS 프로젝트를 통해 남극의 우주 온실에서 얻은 텔레메트리 데이터를 사용해, 비정상 감지 결과를 시계열 클러스터링을 통해 분류하여 다양한 비정상 상태를 유형별로 분석합니다. MDI와 DAMP 알고리즘을 사용하여 단변량 및 다변량 환경에서 비정상 상태를 감지하고 체계적인 비정상 행동을 식별합니다.

- **Technical Details**: BLSS는 장기 우주 임무를 지원하기 위한 인공 생태계입니다. 연구진은 MDI와 DAMP 등 비감독 학습 (unsupervised learning) 방법을 사용하여 시스템 건강 상태를 모니터링합니다. 데이터는 EDEN ISS 프로젝트의 텔레메트리 데이터이며, MDI와 MERLIN 알고리즘을 사용해 시계열 데이터를 분류하고, K-Means 및 HAC (Hierachical Agglomerative Clustering) 클러스터링을 사용해 비슷한 비정상 행동을 군집화합니다. 실험적 검증을 통해 MDI와 DAMP의 상호보완적 성능을 평가하고, 다양한 형태의 비정상 상태를 구분하며, 반복적인 비정상 행동을 식별합니다.

- **Performance Highlights**: MDI와 DAMP 알고리즘은 상호 보완적인 결과를 제공합니다. 이러한 결합 방법은 다양한 비정상 상태를 효과적으로 감지하며, BLSS의 체계적인 비정상 행동을 분석하는 데 유용합니다. 특히, 실험을 통해 MDI와 DAMP가 EDEN ISS 프로젝트의 텔레메트리 데이터에서 식별한 비정상 상태의 재발을 식별할 수 있음을 확인했습니다.



### Enhancing Text Corpus Exploration with Post Hoc Explanations and Comparative Design (https://arxiv.org/abs/2406.09686)
Comments:
          The system is available at: this https URL. The user guide (including more examples) is at: this https URL

- **What's New**: 이 논문에서는 텍스트 코퍼스 탐색(TCE, Text Corpus Exploration)에 새롭게 도입된 방법을 소개합니다. 이 새로운 접근법은 사용자의 다양한 요구에 유연하게 대응하기 위해 후속 설명(post hoc explanations)과 다중 스케일(multiscale) 비교 디자인을 포함하고 있습니다. 이러한 개선된 도구들은 논문의 초록과 신문 아카이브 같은 코퍼스 탐색에 유용하게 활용될 수 있습니다.

- **Technical Details**: 새로운 접근법의 핵심은 'salience functions'라는 메커니즘을 통해 추천, 유사성 평가 및 공간 배치에 대한 사후 설명을 제공합니다. 이러한 설명은 코퍼스 전체에서 개별 문서에 이르기까지 다양한 스케일에서 작동합니다. 시각화 관점에서는 다중 스케일에서 비교를 지원하도록 설계된 여러 뷰가 포함되어 있습니다. 이 기법은 다양한 기저 알고리즘을 보완해 구체적이고 유연한 통합을 지원합니다.

- **Performance Highlights**: 프로토타입 시스템을 통해 다양한 사용자 시나리오에서 우리의 접근법이 어떻게 사용될 수 있는지를 설명합니다. 사용자 연구에서 연구자들이 다양한 작업을 성공적으로 수행할 수 있었음을 확인했습니다. 이 시스템은 특히 논문 초록과 신문 아카이브 같은 코퍼스에서 유용합니다. 또한, 비슷한 추천 시스템 및 공간 맵핑 도구보다 더 유연하게 작동할 수 있다는 점에서 유의미한 개선을 보여줍니다.




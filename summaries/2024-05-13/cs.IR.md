### ProCIS: A Benchmark for Proactive Retrieval in Conversations (https://arxiv.org/abs/2405.06460)
- **What's New**: 이 논문은 대화식 정보 검색의 새로운 영역인 '능동적(Proactive) 대화식 정보 검색 시스템'에 초점을 맞추고 있습니다. 기존의 시스템들이 사용자의 쿼리에 대해서만 반응하는 반면, 이번 연구에서는 다자간 대화를 모니터링하고 적절한 시기에 자동으로 관련 정보를 제공하는 시스템을 구축하는 것을 목표로 합니다.

- **Technical Details**: 연구팀은 280만 건의 대화 데이터셋(ProCIS)을 구축하고, 이를 통해 능동적 문서 검색에 필요한 데이터를 제공합니다. 또한, 깊이-k 풀링(depth-k pooling) 방법을 통해 고품질의 관련성 평가를 수행하였습니다. 대화의 어느 부분이 각 문서와 관련이 있는지에 대한 주석(annotation)도 수집하여, 능동적 검색 시스템을 평가할 수 있는 기반을 마련했습니다.

- **Performance Highlights**: 이 연구는 새로운 평가 지표인 '정규화 능동적 누적 이득(Normalized Proactive Discounted Cumulative Gain, npDCG)'을 도입하여 능동적 대화식 검색 시스템의 성능을 측정하였습니다. 다양한 모델을 벤치마킹하고, 특히 이 작업을 위해 개발된 새로운 모델에 대해 기준 결과를 제공합니다.



### Seasonality Patterns in 311-Reported Foodborne Illness Cases and Machine Learning-Identified Indications of Foodborne Illnesses from Yelp Reviews, New York City, 2022-2023 (https://arxiv.org/abs/2405.06138)
Comments:
          Paper counterpart to flash talk presented at 8th Annual Conference of the UConn Center for mHealth and Social Media, Advancing Public Health and Science with Artificial Intelligence

- **What's New**: 이 연구는 뉴욕시의 레스토랑 리뷰를 통해 식품 매개 질병의 발생을 파악하고자 했습니다. 특히 Yelp 리뷰와 메타데이터를 추출하여 Hierarchical Sigmoid Attention Network (HSAN)를 사용해 리뷰가 식품 매개 질병 사건과 관련이 있을 가능성을 평가하는 새로운 접근 방식을 탐구했습니다. 또한, 연구는 311시스템(식품 보건 관련 비상사태가 아닌 정보 제공 서비스)의 데이터와 비교하여 계절적 패턴을 분석하고자 했습니다.

- **Technical Details**: 연구팀은 Yelp Inc. 및 컬럼비아 대학교의 API를 통해 레스토랑 리뷰 및 비즈니스 주소를 추출했습니다. 이 리뷰들은 HSAN 분류기를 사용하여 평가되었으며, 이는 각 리뷰 세그먼트를 단어 임베딩 및 합성곱 신경망(CNNs)을 통해 인코딩하고, softmax 분류기를 사용하여 각 세그먼트를 분류한 다음, 세그먼트 예측을 종합하여 리뷰의 전체 예측을 산출합니다. HSAN은 0에서 1 사이의 점수를 생성하여 리뷰가 식품 매개 질병 사건을 설명할 가능성이 높을수록 높은 점수를 부여합니다.

- **Performance Highlights**: 2022-2023년 데이터 분석 결과, 평균 일일 온도와 평균 일일 HSAN 점수 사이에 유의미한 상관 관계가 없었습니다(ρ≈-0.05, p≈0.15). 또한, 계절별 뉴욕시의 평균 일일 HSAN 점수 분포나 평균 일일 온도 분포에도 유의한 차이가 발견되지 않았습니다. 이러한 결과는 계절과 식품 매개 질병 발생 사이에 명확한 연관성을 찾지 못했음을 시사합니다.



### Creating Geospatial Trajectories from Human Trafficking Text Corpora (https://arxiv.org/abs/2405.06130)
- **What's New**: 이 논문에서는 인신매매 경로를 자동으로 그리기 위해 'Narrative to Trajectory (N2T)' 정보 추출 시스템을 제안합니다. 이 시스템은 자연어 처리(Natural Language Processing, NLP) 기술을 사용하여 보고된 내러티브에서 관련 정보를 추출하고 지리공간 증강을 적용합니다.

- **Technical Details**: N2T 시스템은 NLP 기술을 활용하여 인신매매와 관련된 텍스트에서 지리적 위치를 인식하고, 데이터 전처리 및 데이터베이스 증강 기술을 통합하여 위치 탐지 방법을 향상시킵니다. 특히, 인신매매 텍스트 코퍼스에 대한 평가를 수행하여 시스템의 유효성을 검증합니다.



### Impedance vs. Power Side-channel Vulnerabilities: A Comparative Study (https://arxiv.org/abs/2405.06242)
- **What's New**: 이 연구는 임피던스 (impedance) 사이드 채널 분석이 어떻게 첨단 암호 기술의 보안 위협이 될 수 있는지를 보여주며, 전통적인 파워 (power) 사이드 채널과의 비교를 통해 그 효과와 가능성을 탐구합니다. 임피던스 사이드 채널은 칩 내부의 논리 상태에 따라 변화하는 임피던스의 차이를 이용하여 암호화된 키를 추출할 수 있는 새로운 방법을 제공합니다.

- **Technical Details**: 이 논문은 임피던스와 파워 사이드 채널의 성능을 비교 분석하여 Advanced Encryption Standard (AES) 암호화 키 추출을 실험적으로 평가합니다. 임피던스 사이드 채널은 칩의 내부 구조에서 발생하는 임피던스 변화를 분석하여, 소프트웨어 명령어 시 실행되는 구별 가능한 임피던스 프로파일을 통해 정보를 취득합니다. 사용된 장비로는 Vector Network Analyzer (VNA)가 있으며, 이는 장치의 input port reflection coefficient (S11)를 분석하여 장치의 임피던스를 계산합니다.

- **Performance Highlights**: 실험 결과 임피던스 사이드 채널 분석이 파워 사이드 채널 분석보다 높은 잠재력을 보여주며, 특히 전통적인 파워 사이드 채널이 만족스럽지 않은 결과를 보이는 시나리오에서 임피던스 사이드 채널이 더욱 효과적이고 강건함을 확인하였습니다. 이는 임피던스 사이드 채널이 암호학적 보안 강화에 중요한 역할을 할 수 있음을 시사합니다.




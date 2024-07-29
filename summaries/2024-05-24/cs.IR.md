### Top-Down Partitioning for Efficient List-Wise Ranking (https://arxiv.org/abs/2405.14589)
Comments:
          16 pages, 3 figures, 2 tables

- **What's New**: 이번 논문에서는 새로운 정렬(ranking) 알고리즘을 제안하고 있습니다. 기존의 sliding window 접근 방식의 문제점을 극복하기 위해 개발된 이 알고리즘은 리스트 상위 항목 우선 처리(top-down processing)를 채택하고 있습니다. 이는 문서 정렬(depth k)에 관한 새로운 접근법입니다.

- **Technical Details**: 기존의 sliding window 방법은 병렬처리가 불가능하고, 반복적인 재정렬과 하위 문서를 우선 처리하는 문제점이 있었습니다. 하지만 새로운 알고리즘은 pivot element를 사용하여 병렬 처리(parallelizable)를 가능하게 함으로써, 여러 문서를 동시에 처리할 수 있습니다. 이런 방식으로, 단일 모델 추론(inference) 호출의 복잡성을 줄였습니다.

- **Performance Highlights**: 이 새로운 알고리즘은 문서 정렬의 깊이가 100일 때 약 33%의 추론 호출 감소를 달성하면서, 이전 접근 방식들과 동등한 성능을 유지했습니다. 이는 여러 강력한 re-ranker들과의 비교에서 입증되었습니다.



### Look into the Future: Deep Contextualized Sequential Recommendation (https://arxiv.org/abs/2405.14359)
Comments:
          arXiv admin note: text overlap with arXiv:2404.18304 by other authors

- **What's New**: 이 논문에서는 기존 연속 추천 시스템보다 더 나은 성능을 보이는 새로운 프레임워크 'LIFT'를 제안했습니다. LIFT는 사용자의 과거와 미래 행동을 바탕으로 현재 프로필을 구성하여 더 효과적으로 사용자의 행동을 예측할 수 있는 방법을 제시합니다.

- **Technical Details**: LIFT 프레임워크는 실제 미래 정보를 사용하지 않고, 유사한 상호작용의 미래 정보를 'retrieval-based' 방법으로 가져와서 미래 컨텍스트로 사용합니다. 또한, 컨텍스트 자체에 내재된 정보를 활용하기 위해 행동 마스킹(behavior masking)을 포함한 혁신적인 사전 학습(pretraining) 방법론을 도입하여 컨텍스트 표현을 효과적으로 학습할 수 있도록 합니다.

- **Performance Highlights**: 실제 데이터셋을 대상으로 한 대규모 실험에서, LIFT는 연속 추천 시스템에서 클릭률(click-through rate) 예측 작업에 있어 강력한 기준 모델들을 뛰어넘는 성능 향상을 보여주었습니다.



### ASI++: Towards Distributionally Balanced End-to-End Generative Retrieva (https://arxiv.org/abs/2405.14280)
- **What's New**: ASI++는 정보 검색에서 새로운 패러다임인 생서적 검색(generative retrieval)을 위한 완전한 엔드 투 엔드(end-to-end) 방식의 접근법을 제안합니다. 이는 문서 ID와 검색 과제 간의 의미적 갭을 극복하고, ID 공간의 비효율적인 사용 문제를 해결하기 위해 고안되었습니다.

- **Technical Details**: ASI++는 저부에서 부터 ID 할당과 검색 성능 향상을 위한 몇 가지 중요한 혁신을 도입했습니다. 첫째, 분포 균형 기준(distributionally balanced criterion)을 통해 ID 할당의 불균형을 해결하고 ID 공간의 효율적인 사용을 촉진합니다. 둘째, 표현 병목 기준(representation bottleneck criterion)을 통해 밀집 표현(dense representations)을 향상시켜 ID 할당 학습의 병목 현상을 완화합니다. 셋째, 정보 일관성 기준(information consistency criterion)을 통해 이러한 과정을 정보 이론에 기반한 공동 최적화 프레임워크로 통합합니다.

- **Performance Highlights**: ASI++는 신경 양자화(neural quantization), 차별적 제품 양자화(differentiable product quantization), 잔여 양자화(residual quantization) 등 다양한 모듈 구조를 탐구하여 ID 할당 학습을 최적화했습니다. 공공 및 산업 데이터셋에 대한 광범위한 실험을 통해, ASI++는 검색 성능 향상과 ID 할당의 균형 성취에 효과적임을 입증했습니다.



### Identifying Breakdowns in Conversational Recommender Systems using User Simulation (https://arxiv.org/abs/2405.14249)
Comments:
          ACM Conversational User Interfaces 2024 (CUI '24), July 8--10, 2024, Luxembourg, Luxembourg

- **What's New**: 이번 연구에서는 대화 추천 시스템(conversational recommender systems)의 대화 고장(conversational breakdowns)을 체계적으로 검사하는 방법론을 제시합니다. 이 방법론은 시스템과 시뮬레이션 사용자(simulated users) 간에 생성된 대화를 사전에 정의된 고장 유형에 따라 분석하고, 책임 있는 대화 경로를 추출하며, 기본적인 대화 의도(dialogue intents) 측면에서 이를 특성화합니다.

- **Technical Details**: 사용자 시뮬레이션(user simulation)을 사용하여 대화 고장의 잠재적 발생 지점을 식별하는 데 필요한 대화를 간편하고 비용 효율적이며 시간 효율적으로 얻을 수 있습니다. 제안된 방법론은 대화 추천 시스템의 진단 도구이자 개발 도구로서 활용될 수 있습니다.

- **Performance Highlights**: 기존 대화 추천 시스템과 사용자 시뮬레이터를 사용한 사례 연구를 통해 몇 번의 반복만으로도 시스템이 대화 고장을 더 잘 견딜 수 있게 개선할 수 있음을 입증했습니다.



### Generative AI Search Engines as Arbiters of Public Knowledge: An Audit of Bias and Authority (https://arxiv.org/abs/2405.14034)
- **What's New**: 새로운 연구에서는 생성형 AI 시스템(ChatGPT, Bing Chat, Perplexity)의 응답 생성 방식과 공공 중요 주제에 대한 권위 확립 방식을 조사했습니다. 연구자들은 7일 동안 4개의 주제에 대해 48개의 실질적인 질문을 사용해 시스템 응답을 수집했습니다.

- **Technical Details**: 수집된 데이터는 감성 분석(sentiment analysis), 귀납적 코딩(inductive coding), 출처 분류(source classification) 등을 사용해 분석되었습니다. 이번 연구는 쿼리와 주제에 따른 감정 편향(sentiment bias) 및 출처의 상업적, 지리적 편향(commercial and geographic bias)에 대한 증거를 제공합니다.

- **Performance Highlights**: 분석 결과, 이러한 시스템들은 News and Media, Business, Digital Media 웹사이트에 크게 의존하면서, 근거가 불균형적인 소스를 사용하고 있는 것으로 나타났습니다. 이는 공공의 이익과 개인의 웰빙 관련 결정을 내릴 때 생성형 AI 시스템의 출력물을 비판적으로 검토해야 할 필요성을 강조합니다.



### Diffusion-Based Cloud-Edge-Device Collaborative Learning for Next POI Recommendations (https://arxiv.org/abs/2405.13811)
- **What's New**: 이 논문은 새로운 협력 학습 프레임워크, DCPR(Diffusion-Based Cloud-Edge-Device Collaborative Learning for Next POI Recommendations)을 소개합니다. 이는 위치 기반 소셜 네트워크(Location-Based Social Networks, LBSNs)에서의 다음 방문 장소(Point-of-Interest, POI) 추천을 위해 설계된 기술입니다. DCPR는 전통적인 중앙집중식 딥러닝 시스템이 가진 개인정보 문제와 최신성 제한 문제를 해결하고자 합니다.

- **Technical Details**: DCPR는 확산 모델(diffusion model)을 활용하여 클라우드-엣지-디바이스 아키텍처에서 동작합니다. 이 아키텍처는 지역별로 맞춤화된 POI 추천을 제공하며, 사용자 디바이스의 계산 부담을 줄입니다. DCPR는 글로벌 학습 프로세스와 로컬 학습 프로세스를 독특하게 결합하여 디바이스 계산 요구를 최소화합니다. 기존 중앙집중식 딥러닝보다 효율적이고 개인정보를 보호하는 장점이 있습니다.

- **Performance Highlights**: 두 개의 실제 데이터셋을 통해 평가한 결과, DCPR는 추천 정확도, 효율성, 새로운 사용자나 지역에 대한 적응성에서 뛰어난 성능을 보였습니다. 이는 기존 장치 기반 POI 추천 기술을 크게 발전시키는 중요한 단계로 평가됩니다.



### Using k-medoids for distributed approximate similarity search with arbitrary distances (https://arxiv.org/abs/2405.13795)
- **What's New**: 이 논문은 GMASK이라는 알고리즘을 소개합니다. 이는 분산 환경에서 임의의 거리 함수를 수용할 수 있는 유사성 검색(approximate similarity search)을 위한 일반적인 알고리즘입니다. 기존의 많은 유사성 검색 알고리즘들이 주로 유클리드 거리(Euclidean distance)를 사용하고, 이는 특정 문제에 적합하지 않다는 문제점이 있었습니다. 대신, GMASK는 k-중심(k-medoids)을 사용하여 더 다양한 거리 함수와 문제에 대응할 수 있게 설계되었습니다.

- **Technical Details**: GMASK는 데이터셋에서 보로노이 지역(Voronoi regions)을 형성하는 군집화 알고리즘(clustering algorithm)을 요구하며, 각 지역의 대표 요소를 반환합니다. 그런 다음 많은 차원과 희소성을 가진 대규모 데이터셋에 적합한 다층 인덱싱 구조(multilevel indexing structure)를 만듭니다. 이는 일반적으로 분산 시스템에 저장됩니다. GMASK는 어떠한 거리 함수도 호환 가능하도록 k-중심을 사용하여 구성되었습니다.

- **Performance Highlights**: 실험 결과는 GMASK가 실제 데이터셋에서도 적용 가능하며, 대안 알고리즘에 비해 유사성 검색 성능이 향상됨을 입증합니다. 또한, 높은 차원 데이터셋(high-dimensional datasets)에서 특정 Minkowski 거리 함수를 사용하는 장점에 대한 기존의 직관을 확인했습니다.



### Lusifer: LLM-based User SImulated Feedback Environment for online Recommender systems (https://arxiv.org/abs/2405.13362)
- **What's New**: Lusifer는 현재의 강화 학습 기반 추천 시스템 훈련의 큰 문제점인 동적이고 현실적인 사용자 상호작용의 부족을 해결합니다. 이는 대규모 언어 모델(LLMs)을 활용하여 시뮬레이션된 사용자 피드백을 생성함으로써 이루어집니다. 이 시스템은 사용자 프로필과 상호작용 기록을 합성하여 추천 항목에 대한 사용자의 반응과 행동을 시뮬레이션합니다.

- **Technical Details**: Lusifer의 작동 파이프라인은 프롬프트 생성(prompt generation)과 반복적인 사용자 프로필 업데이트(iterative user profile updates)를 포함합니다. 사용자 프로필은 각 평가 후 업데이트되어 진화하는 사용자 특성을 반영합니다. 이를 통해 현실적이고 동적인 피드백을 생성합니다.

- **Performance Highlights**: MovieLens100K 데이터셋을 사용한 개념 증명에서, Lusifer는 사용자 행동과 선호도를 정확하게 에뮬레이션하는 데 성공했습니다. 이로써 강화 학습 시스템을 훈련하는 데 이 환경을 사용할 수 있음을 검증합니다.



### Enhancing User Interest based on Stream Clustering and Memory Networks in Large-Scale Recommender Systems (https://arxiv.org/abs/2405.13238)
- **What's New**: 추천 시스템(Recommender Systems, RSs)은 사용자 관심 기반의 개인화된 추천 서비스를 제공하지만, 구매 행동이 부족하여 관심 분포가 희박한 사용자의 경우 추천 성능이 저조합니다. 이러한 문제를 해결하기 위해, 사용자 관심 향상(User Interest Enhancement, UIE)이라는 새로운 솔루션을 제안합니다. UIE는 스트림 클러스터링(stream clustering)과 메모리 네트워크(memory networks)을 기반으로 생성된 향상 벡터(enhancement vectors)와 개인화된 향상 벡터를 사용하여 사용자 프로필 및 사용자 역사 행동 시퀀스를 향상시킵니다.

- **Technical Details**: UIE는 사용자 관심 희소성 문제를 다루는 데 중점을 두며, 향상 벡터와 개인화된 향상 벡터를 통해 모델 성능을 크게 개선합니다. 이 솔루션은 랭킹 모델(ranking model)을 기반으로 한 종단 간(end-to-end) 접근방식으로 구현이 용이합니다. 또한, 유사한 방법을 롱테일 항목(long-tail items)에도 적용하여 탁월한 성능 향상을 달성했습니다.

- **Performance Highlights**: 대규모 산업용 추천 시스템에서 광범위한 오프라인 및 온라인 실험을 수행한 결과, UIE 모델은 특히 관심 분포가 희박한 사용자에 대해 다른 모델을 크게 능가하는 성능을 보여주었습니다. 현재 UIE는 여러 대규모 추천 시스템에 완전히 배포되어 주목할 만한 성능 향상을 이루었습니다.



### A Workbench for Autograding Retrieve/Generate Systems (https://arxiv.org/abs/2405.13177)
Comments:
          10 pages. To appear in the Resource & Reproducibility Track of SIGIR 2024

- **What's New**: 이 논문은 자동회귀적 대형 언어 모델(autoregressive Large Language Models, LLMs) 시대에서 정보 검색 시스템(Information Retrieval, IR)의 평가 문제를 다룹니다. 기존의 단락 수준 판단(passage-level judgments)에 의존하는 방법은 더 이상 효과적이지 않으며, 따라서 LLM을 통합한 여러 대안 평가 접근 방식을 탐색할 수 있는 작업대를 제공합니다.

- **Technical Details**: 이 논문에서 소개된 평가 접근 방식은 다음과 같습니다: 1. 응답의 관련성을 LLM에 물어보는 방식; 2. 응답에서 포함된 중요 사실들(nuggets) 세트를 LLM에 질문하는 방식; 3. 응답으로 시험 문제 세트를 푸는 방식. 연구자들이 관련 키 사실(nuggets)과 시험 문제 세트를 수작업으로 개선하고, 이러한 개선이 시스템 평가 및 리더보드 순위에 미치는 영향을 관찰할 수 있는 기능을 갖춘 작업대를 제공합니다.

- **Performance Highlights**: 이 작업대는 새롭고 재사용 가능한 테스트 컬렉션(test collections)의 개발을 촉진합니다. 다양한 평가 접근 방식을 통해 연구자들은 시스템 응답의 적합성을 효율적으로 판단할 수 있으며, 이는 정보 검색 시스템의 최종 성능을 높이는 데 기여할 수 있습니다.



### Push and Pull: A Framework for Measuring Attentional Agency (https://arxiv.org/abs/2405.14614)
- **What's New**: 이 논문은 디지털 플랫폼에서 사용자가 주의를 자신의 욕구, 목표, 의도에 따라 할당할 수 있는 '주의 에이전시(Attentional Agency)'를 측정하는 새로운 프레임워크를 제안합니다. 플랫폼은 사용자의 제한된 주의력을 확장하여 더 많은 정보를 제공하지만, 대개 사용자가 타인의 주의도 영향을 미칠 수 있게 합니다.

- **Technical Details**: 이 논문에서는 사용자가 정보를 자신의 주의 필드로 끌어들이는 능력과 타인의 주의 필드로 정보를 밀어 넣는 능력을 측정하는 포멀한 프레임워크를 소개합니다. 또한 정보 상품에서 경제적 가치를 캡처하기 위한 임베디드 광고(Embedded Advertising)와 같은 방법을 우회할 수 있는 생성적 기저 모델(Generative Foundation Models)의 의미를 다룹니다.

- **Performance Highlights**: 생성적 기저 모델을 사용하여 사용자가 임베디드 광고의 '주의 거래(Attentional Bargain)'를 우회하는 방법을 강조하고, 온라인에서 주의 에이전시의 분포를 이해하고 재구성하기 위한 정책 전략을 제시합니다.



### AutoLCZ: Towards Automatized Local Climate Zone Mapping from Rule-Based Remote Sensing (https://arxiv.org/abs/2405.13993)
Comments:
          accepted at 2024 IGARSS

- **What's New**: 이번 연구에서는 고해상도 원격 탐사(RS) 데이터를 활용하여 Local Climate Zones (LCZ) 분류 기능을 추출하는 새로운 LCZ 매핑 프레임워크인 AutoLCZ를 제안합니다. 이는 기존의 인간 상호작용에 의존한 지리 정보 시스템(GIS) 기반 방식과 달리, 대규모 지역에서 정확한 메타데이터를 획득하기 위한 자동화된 방법을 제공합니다.

- **Technical Details**: AutoLCZ는 LiDAR 데이터를 이용한 지형적 및 표면 피복 특성을 모델링하는 수치 규칙을 정의합니다. 이로써 RS 데이터를 사용하는 GIS 기반의 LCZ 분류가 가능해집니다. 특히 LiDAR 데이터에서 추출된 4가지 LCZ 특징을 통해 10가지 LCZ 유형을 구분하는 방식을 제안하였습니다.

- **Performance Highlights**: 뉴욕시(NYC)를 대상으로 한 개념 검증에서, 전송 LiDAR 조사를 활용하여 AutoLCZ는 10가지 LCZ 유형을 구분하는데 성공을 거두었습니다. 이를 통해 AutoLCZ가 대규모 RS 데이터에서 LCZ 매핑을 수행할 수 있는 가능성을 입증했습니다.



### From the evolution of public data ecosystems to the evolving horizons of the forward-looking intelligent public data ecosystem empowered by emerging technologies (https://arxiv.org/abs/2405.13606)
- **What's New**: 이번 연구는 공공 데이터 생태계(PDEs)의 발전 모델(Evolutionary Model of Public Data Ecosystems, EMPDE)을 실제 사례를 통해 검증하는 것을 목적을 두었습니다. 특히 'Intelligent Public Data Generation'이라는 6세대 PDE의 실현 가능성을 탐구했습니다. 이 연구는 라트비아, 세르비아, 체코, 스페인, 폴란드 등 5개의 유럽 국가 사례를 통해 이론적 모델을 검증했습니다.

- **Technical Details**: 6세대 PDE는 클라우드 컴퓨팅(Cloud Computing), 인공지능(AI), 자연어 처리 도구(Natural Language Processing tools), 생성형 인공지능(Generative AI), 대규모 언어 모델(Large Language Models, LLM) 등의 신기술에 의해 주도되는 패러다임 전환이 특징입니다. 이는 데이터 생태계 내의 비즈니스 프로세스의 자동화 및 증강에 기여할 수 있습니다. 이번 연구는 이러한 기술들이 단순히 구성 요소를 넘어 생태계의 행위자이자 이해관계자로서 혁신과 진보를 촉진할 수 있는 능력을 탐구했습니다.

- **Performance Highlights**: 연구를 통해 EMPDE 모델의 실무적 적용 가능성을 확인하였으며, 공공 데이터 생태계 관리 전략이 디지털 시대의 사회적, 규제적, 기술적 필수 요건에 맞게 조정될 수 있는 통찰을 제공했습니다. 특히 다양한 유럽 국가들에서의 구현 변이를 통해 각국의 문맥에 따른 PDE 역동성을 이해할 수 있었습니다.




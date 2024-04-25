### KGValidator: A Framework for Automatic Validation of Knowledge Graph  Construction (https://arxiv.org/abs/2404.15923)
Comments: Text2KG 2024, ESWC 2024

- **What's New**: 이 연구는 대규모 언어 모델(Large Language Models, LLMs)을 활용하여 지식 그래프(Knowledge Graph, KG) 완성 모델의 자동 평가에 대해 탐구합니다. KGValidator라는 새로운 프레임워크를 도입하여 지식 그래프의 일관성과 검증을 지원하며, LLM 자체의 내재된 지식, 사용자가 제공한 문헌 집합, 외부 지식 소스 등 다양한 정보원을 활용합니다.

- **Technical Details**: KGValidator는 Instructor, Pydantic 클래스, 함수 호출을 사용하여 LLM이 정확한 가이드라인을 따르도록 하며, 평가 지표를 계산하기 위한 올바른 데이터 구조를 출력하게 합니다. 이 프레임워크는 오픈 소스 라이브러리를 기반으로 하며, 어떠한 유형의 지식 그래프에도 쉽게 적용 및 확장이 가능합니다.

- **Performance Highlights**: 이 프레임워크는 널리 사용되는 KG 완성 벤치마크 데이터셋에 대한 효과적인 검증 수단으로 평가되었으며, 추가적인 맥락을 제공할 때 SoTA(State-of-the-Art) LLM의 평가 능력이 증진되는 효과를 확인하였습니다. 또한, 사용자 프롬프트와 입력을 함께 포함시키는 방식을 통해 제로 샷(zero-shot) 설정에서도 성공적인 성능을 보였습니다.



### Recursive Backwards Q-Learning in Deterministic Environments (https://arxiv.org/abs/2404.15822)
- **What's New**: 이 연구는 재귀적 역방향 Q-learning (Recursive backwards Q-learning, RBQL) 에이전트를 도입하여 강화 학습(reinforcement learning)의 새로운 접근법을 제시합니다. RBQL은 환경의 모델을 구축하고 최종 상태(terminal state)에 도달한 후 그 모델을 통해 값을 재귀적으로 역전파(recursively propagate)하여 더 빠른 학습이 가능하도록 설계되었습니다.

- **Technical Details**: RBQL 에이전트는 환경에 대한 모델을 활용하여 결정론적 문제(deterministic problems)에서 Q-learning의 성능을 개선합니다. 에이전트는 환경을 탐색하면서 상태(state)와 행동(action) 간의 매핑(mapping)을 구축하고, 최종 상태에 도달하면 학습한 정보를 바탕으로 가치(value)를 역전파하는 방식으로 최적의 해(optimal solutions)를 빠르게 추출할 수 있습니다.

- **Performance Highlights**: 미로(maze)를 통과하는 최단 경로를 찾는 예제에서 RBQL 에이전트는 일반 Q-learning 에이전트보다 월등한 성능을 보여주었습니다. 이는 RBQL이 각 상태를 최적의 가치로 평가할 수 있도록 만드는 효율적인 값 역전파 과정 덕분입니다.



### Multi-Agent Reinforcement Learning for Energy Networks: Computational  Challenges, Progress and Open Problems (https://arxiv.org/abs/2404.15583)
- **What's New**: 본 연구에서는 전기 그리드의 동적이고 발전하는 성격을 지원할 수 있는 새로운 기술로서 다중 에이전트 강화학습(Multi-agent Reinforcement Learning, MARL)을 도입하여 전력망의 분산화와 탈탄소화를 지원하고, 관련된 기술적 및 관리적 도전들을 완화하는 방법을 탐구합니다. 특히 전력망을 관리하는 주요 계산적 도전들을 지정하고, 그 해결을 위한 최근 연구 진행 상황을 검토하며, MARL을 사용하여 해결될 수 있는 개방형 도전들을 강조합니다.

- **Technical Details**: 이 논문은 MARL이 전력망 관리에서 중앙집중식 방법들을 대체할 수 있는 분산된 및 계층적 모델을 적용하는 방법에 중점을 둡니다. 여러 에이전트들이 환경과 서로 상호작용하면서 기대 유틸리티를 극대화하려는 목적으로 학습하는 시나리오에서, 이 연구는 특히 동작 최적화가 물리적 전기 네트워크에 의해 부과된 제약과 최적화 고려 사항을 고려해야 하는 도전을 제시합니다.

- **Performance Highlights**: MARL의 설계와 구현은 전력망의 탈중앙화와 탄소 배출 감소를 실현하면서, 실시간 네트워크 유지 및 전력 흐름 한계 관리와 같은 실제 전력 시스템 운영의 복잡성에 대처할 수 있는 능력을 개선했습니다. 특히, 분산 에너지 자원(DER: Distributed Energy Resources)의 통합을 위한 새로운 제어 전략, 발전 및 수요 예측의 고급 방법, 네트워크의 실시간 모니터링 및 제어를 위한 새로운 방법들에 대한 논의가 포함됩니다.



### GeoLLM-Engine: A Realistic Environment for Building Geospatial Copilots (https://arxiv.org/abs/2404.15500)
Comments: Earthvision 2024, CVPR Workshop

- **What's New**: GeoLLM-엔진은 지리적 원격 관측 플랫폼에서 현실적인 고차원 자연어 명령(Natural Language Commands)을 해석하고 작업 수행을 평가하기 위해 다양한 지리 공간 API 도구(Geospatial API Tools), 동적 지도/사용자 인터페이스(Dynamic Maps/UIs), 외부 다중모드 지식 기반(External Multimodal Knowledge Bases)을 통합한 환경을 제공합니다. 대규모 병렬 엔진을 사용하여 100개의 GPT-4-Turbo 노드를 거쳐 백만 개가 넘는 위성 이미지를 사용하여 50만 개 이상의 다양한 멀티-툴 작업(Multi-Tool Tasks)을 생성합니다.

- **Technical Details**: 이 연구에서는 복잡한 테스크를 수행하기 위한 '엔진(Engine)' 구축에 초점을 맞추며, 기존 벤치마크의 단순한 이미지-캡션 작업 접근법을 벗어나 독창적인 GeoLLM-엔진을 개발하였습니다. 특히, 자연어 처리(Natural Language Processing, NLP) 기반의 고정밀 작업 검증 기술을 도입하여 사람의 개입을 최소화하고, 벤치마크 생성과 검증 과정에서의 집중도와 정확성을 향상시켰습니다. 또한, GeoLLM-엔진은 Mapbox API나 Rasterio, GeoPandas 같은 오픈소스 라이브러리를 활용하여 효과적인 데이터 분석 및 작업 수행이 가능합니다.

- **Performance Highlights**: GeoLLM-엔진은 다양한 위성 이미지를 활용하여 고도로 복잡한 작업에서의 에이전트 성능을 평가합니다. 초기 실험 결과, GPT-4 Turbo 같은 최신 LLM을 사용하여 복잡성이 증가하는 다양한 작업에서 에이전트의 성능이 상당히 향상됨을 보여주었습니다. 새로운 벤치마크는 기존 보다 더 다양한 원격 탐지(Remote Sensing, RS) 응용 프로그램을 아우르며, 더욱 정교하고 실제와 가까운 지리공간 데이터 분석 거버넌스를 제공합니다.



### Multi-scale Intervention Planning based on Generative Design (https://arxiv.org/abs/2404.15492)
- **What's New**: 도시 환경에서 녹지 공간의 부족은 주민의 건강과 웰빙에 부정적인 영향을 미치는 중요한 문제로, 이 연구에서는 생성 AI(generative AI)를 활용하여 다양한 규모의 개입 계획에 초점을 맞춰 천연 기반 솔루션(Nature Based Solutions, NBS)을 적용하는 새로운 방법을 제안합니다. 연구는 이미지 간 변환(image-to-image) 및 이미지 인페인팅(image inpainting) 알고리즘을 이용하여 도시 지역의 녹지 공간 부족 문제를 해결하고, 테살로니키의 두 골목에서 이러한 접근 방식의 효과를 시연하였습니다.

- **Technical Details**: 생성 디자인(generative design)은 고급 알고리즘과 계산 기술을 특징으로 하며, 사전 정의된 매개변수와 제약을 기반으로 디자인 시나리오를 자동 생성하는 체계적인 접근 방식을 제공합니다. 연구팀은 간단한 그래피컬 유저 인터페이스(Graphical User Interface, GUI) 데스크톱 응용 프로그램을 개발하여 실제 시나리오에서 생성 디자인을 구현하였으며, HuggingFace 저장소의 모델을 기반으로 이미지 생성 애플리케이션을 개발하였습니다. 이를 이용한 실험에서는 테살로니키의 두 골목을 대상으로 60개의 이미지를 생성하여 다양한 개입 시나리오를 시각화하였습니다.

- **Performance Highlights**: 생성된 이미지들은 건축적 구성 및 실현 가능성면에서 평가되었으며, 이미지 간 변환 방법은 환경을 변경하나, 보여주는 결과물은 좋은 시각적 아이디어를 제공하였습니다. 한편, 이미지 인페인팅 방법은 개입 지역만을 수정하여 더 현실적이고 실현 가능한 결과물을 제공하였습니다. 양 방법 모두에서, 생성 이미지 하나를 생산하는데 필요한 시간은 각각 약 3분과 4분이 소요되었습니다. 설계 전문가들은 이러한 기술을 실제 설계 과정에서 시각적 아이디어를 제공하는 데 활용할 수 있으며, 여러 해결책을 신속하게 생성할 수 있는 강력한 이점을 가집니다.



### Neural Operators Learn the Local Physics of Magnetohydrodynamics (https://arxiv.org/abs/2404.16015)
Comments: 47 pages, 24 figures

- **What's New**: 이 연구에서는 이상적인 자기수력학(Magnetohydrodynamics, MHD) 문제에 적용된 플럭스 푸리에(Flux Fourier) 신경 연산자(neural operator) 모델의 변형에 대해 살펴봅니다. 이 모델은 기존의 수치 해석 방법을 대체할 수 있는 새로운 접근 방식을 제시하며, 샘플 분포 외부의 일반화(generalization), 연속 추론(continuous inference), 그리고 기존의 수치 방식에 비해 빠른 계산 속도를 가능하게 합니다.

- **Technical Details**: 이 연구는 Flux Fourier Neural Operator(FNO) 모델을 변형하여 각 물리 변수(밀도, 속도, 자기장, 에너지)에 대해 개별적으로 처리하도록 모델 아키텍처를 재설계하였습니다. 또한, Total Variation Diminishing(TVD) 속성을 갖도록 근사된 수치 플럭스에 손실 함수를 설계 및 적용하였습니다. 자기장 변수의 발산에 대한 손실을 적용하여 발산-자유 조건(divergence-free condition)을 시행하고, 아블레이션 연구(ablation study)를 통해 이 솔버에 적합한 유도 편향(inductive bias)을 검증했습니다.

- **Performance Highlights**: Flux FNO 모델은 연속 시간에 대한 추론, 분포 밖 샘플에 대한 추론에서 일반화 성능을 질적으로 평가하였고, 이상적인 MHD 문제의 대표적인 테스트 문제들을 해결하면서 전통적인 수치 방법론들과 비교되었습니다. 결과적으로, 이 모델은 고전적인 수치 스킴에 비해 빠른 계산 속도를 보이며 탁월한 성능을 보였습니다.



### Improving Dictionary Learning with Gated Sparse Autoencoders (https://arxiv.org/abs/2404.16014)
- **What's New**: 본 연구에서는 언어 모델의 활성화에서 해석 가능한 특징들을 비지도로 발견하는 효율적인 기술인 희소 오토인코더(Sparse Autoencoder, SAE)를 개선한 새로운 방식인 Gated Sparse Autoencoder (Gated SAE)를 도입합니다. 기존 SAE의 L1 패널티가 갖는 문제점을 해결하고, 스파스티(sparsity)와 재구성 정확도 간의 더 나은 트레이드오프를 제공합니다.

- **Technical Details**: Gated SAE는 기존의 SAE 구조를 개선하여 두 가지 주요 기능을 분리합니다: (a) 사용할 방향 결정과 (b) 해당 방향의 크기 추정. 이는 L1 패널티를 (a)에만 적용함으로써 부작용의 범위를 제한합니다. 또한, 두 변환(transformation) 간의 일부 가중치를 공유하여, Gated SAE가 기존 SAE와 동일한 너비에서 훨씬 많은 매개 변수 수와 연산 요구량을 증가시키지 않도록 합니다.

- **Performance Highlights**: 다양한 모델과 레이어에서 Gated SAE를 평가한 결과, 베이스라인 SAE에 비해 동일한 훈련 계산 조건에서 스파스티(sparsity)를 증가시키고 재구성 충실도(reconstruction fidelity)를 향상시킬 수 있는 파레토 개선(Pareto improvement)을 달성합니다. 또한, 특징의 해석 가능성(interpretablility) 면에서 베이스라인 SAE와 비교할 수 있는 결과를 보여줌으로써, Gated SAE가 기존 문제인 축소(shrinkage) 문제를 극복하는 것을 확인했습니다.




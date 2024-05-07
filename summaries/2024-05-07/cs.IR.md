### Adaptive Retrieval and Scalable Indexing for k-NN Search with  Cross-Encoders (https://arxiv.org/abs/2405.03651)
Comments: ICLR 2024

- **What's New**: 이 연구에서는 교차 인코더(Cross-encoder, CE) 모델과 벡터 임베딩 공간을 통해 CE 유사성을 근사하는 새로운 희소 행렬 분해 기반 방법을 제안합니다. 이 방법은 기존의 듀얼 인코더(Dual-encoder, DE) 또는 CUR 행렬 분해 기반 방식보다 효율적으로 항목 임베딩을 계산하고, CE 점수를 높은 정확도로 근사화하여 k-NN 검색을 수행합니다.

- **Technical Details**: 이 기법은 훈련 쿼리 세트에 대해 쿼리-항목 CE 점수를 포함하는 희소 행렬을 분해함으로써 오프라인에서 항목 임베딩을 계산합니다. 검색 시 항목 임베딩은 고정되며, a) 검색된 항목의 CE 점수를 근사화하는 오류를 최소화하는 테스트 쿼리 임베딩 추정, b) 업데이트된 테스트 쿼리 임베딩을 사용하여 더 많은 항목을 검색하는 두 단계를 번갈아 가며 수행합니다. 이 방법은 DE를 초기 임베딩 공간 구성에 활용하면서 DE의 집중적인 파인튜닝을 피할 수 있습니다.

- **Performance Highlights**: 이 k-NN 검색 방법은 DE 기반 접근 방식보다 최대 5% (k=1) 및 54% (k=100)까지 리콜을 개선합니다. 또한, 인덱싱 접근 방식은 CUR 기반 방식 대비 최대 100배, DE 증류 방법 대비 5배의 속도 향상을 달성하였으며, 기존 베이스라인을 유지하거나 개선한 k-NN 검색 리콜을 보였습니다.



### ID-centric Pre-training for Recommendation (https://arxiv.org/abs/2405.03562)
- **What's New**: 이 연구에서는 새로운 ID 중심(ID-centric) 추천 시스템 사전 학습 패러다임(IDP)을 제안합니다. 이 방식은 사전 학습 도메인에서 배운 정보가 풍부한 ID 임베딩을 새 도메인의 아이템 표현으로 직접 전달합니다. 이를 통해 전통적인 PLM(Pre-trained Language Model) 기반 추천 시스템의 한계를 극복하고자 합니다.

- **Technical Details**: 본 연구는 사전 학습 단계에서 ID 기반 순차적 모델과 함께 행동 및 모달리티(behavioral and modality) 정보를 모두 학습하는 크로스도메인 ID 매처(Cross-domain ID-matcher, CDIM)를 구축합니다. 튜닝 단계에서는 새 도메인 아이템의 모달리티 정보를 CDIM을 통해 크로스도메인 브릿지로 사용하고, 텍스트 정보를 사용하여 사전 학습 도메인에서 행동적이고 의미론적으로 유사한 아이템을 검색합니다. 검색된 사전 학습 ID 임베딩은 새 도메인 아이템의 임베딩을 생성하는 데 직접 사용됩니다.

- **Performance Highlights**: 실세계 데이터셋에서의 광범위한 실험을 통해, 본 모델은 추운 환경(cold setting)과 따뜻한 환경(warm setting) 모두에서 모든 기준 모델을 상당히 능가하는 성능을 보여주었습니다. 이러한 결과는 새 도메인에 대한 추천의 정확성과 효율성을 크게 향상시킬 수 있음을 시사합니다. 코드는 논문이 받아들여지면 공개될 예정입니다.



### Doing Personal LAPS: LLM-Augmented Dialogue Construction for  Personalized Multi-Session Conversational Search (https://arxiv.org/abs/2405.03480)
Comments: Accepted at SIGIR 2024 (Full Paper)

- **What's New**: 대화형 에이전트의 미래는 사용자에게 맞춤형 정보 응답을 제공할 것입니다. 그러나 모델을 개발하는 데 있어 가장 큰 도전 중 하나는 실제 사용자의 선호를 반영하고 여러 세션에 걸쳐 있는 대규모 대화 데이터셋의 부족입니다. 이전 방법들은 위저드-오브-오즈(Wizard-of-Oz) 설정에서 전문가에 의존했으며, 이는 특히 맞춤형 작업에 있어 확장하기 어려웠습니다. 우리의 방법론인 LAPS는 대규모 언어 모델(LLMs)을 활용하여 개인화된 대화를 생성하기 위해 단일 작업자를 안내하는 방식으로 이 문제를 해결하였습니다.

- **Technical Details**: LAPS는 대규모 언어 모델을 사용하여 개인화된 대화를 생성하는 데 중점을 둔다는 점에서 혁신적입니다. 이 기법은 인간 작업자가 멀티 세션 및 멀티 도메인 대화를 생성하도록 안내하여, 대화의 자연성과 다양성을 유지하면서 실제 사용자의 선호도를 추출하게 합니다. 또한, LAPS를 통해 생성된 데이터셋은 선호도 추출(Preference Extraction)과 맞춤형 응답 생성(Personalized Response Generation)을 훈련하기에 적합합니다.

- **Performance Highlights**: LAPS를 사용하여 생성된 데이터셋은 기존 데이터셋과 비교했을 때 전문가가 만든 것과 마찬가지로 자연스럽고 다양합니다. 이는 전적으로 합성 방식(Synthetic Methods)과 대비됩니다. 사용자의 실제 선호도에 더 잘 맞는 응답을 생성하기 위해 추출된 선호도를 사용하는 것이 단순한 대화 히스토리보다 가치가 있음을 보여줍니다. 전체적으로 LAPS는 LLM을 활용하여 보다 효율적이고 효과적으로 현실적인 맞춤형 대화 데이터를 생성하는 새로운 방법을 도입하였습니다.



### Improving (Re-)Usability of Musical Datasets: An Overview of the DOREMUS  Projec (https://arxiv.org/abs/2405.03382)
- **What's New**: DOREMUS는 세 개의 프랑스 기관의 데이터를 연결하고 탐색하기 위해 새로운 도구를 개발함으로써 음악에 대한 더 나은 설명을 제공합니다. 이 논문은 FRBRoo에 기반한 데이터 모델과 연결된 데이터 기술을 사용한 변환 및 연결 과정을 개괄적으로 설명하며, 웹 사용자의 요구에 맞춰 데이터를 사용할 수 있는 프로토타입을 제시합니다.

- **Technical Details**: DOREMUS 프로젝트는 FRBRoo(FRBR object-oriented) 데이터 모델을 활용하여 음악 관련 데이터의 구조화 및 표준화를 도모합니다. Linked Data 기술을 통해 데이터 변환 및 연결 작업을 수행하며, 이를 통해 다양한 소스의 정보가 통합됩니다.

- **Performance Highlights**: 이 연구는 웹 사용자들이 음악 데이터를 쉽게 탐색하고 활용할 수 있도록 하는 프로토타입을 성공적으로 구현했으며, 체계적인 데이터 모델과 효과적인 데이터 연결 과정을 통해 데이터 접근성과 활용도를 향상시켰습니다.



### Explainability for Transparent Conversational Information-Seeking (https://arxiv.org/abs/2405.03303)
Comments: This is the author's version of the work. The definitive version is published in: 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24), July 14-18, 2024, Washington, DC, USA

- **What's New**: 이 연구는 대화형 검색 시스템에서 정보의 투명성을 향상시키기 위한 새로운 접근법을 탐구합니다. 정보의 출처, 시스템의 신뢰도 및 제한사항에 대한 투명성이 사용자가 응답을 객관적으로 평가하는 능력을 향상시킬 수 있다는 가설을 설정하고 있습니다. 이를 위해 저자들은 사용자의 응답 유용성에 대한 설명의 질과 제시 방법이 어떤 영향을 미치는지에 대해 사용자 연구를 설계하였습니다.

- **Technical Details**: 이 연구는 다양한 설명 방식, 품질 및 프레젠테이션 모드(Explanation Type, Quality, and Presentation Mode)를 조사하여 대화형 정보 추구(Conversational Information Seeking; CIS) 시스템의 응답과 사용자가 검증할 수 있는 응답 사이의 간극을 줄이는 것을 목표로 합니다. 또한, 이 연구에서는 정보의 출처, 시스템의 신뢰도 및 응답의 잠재적 한계점(The Sources, System’s Confidence, and Potential Limitations of the Response)에 대해 투명하게 하는 것이 사용자의 정보 판단을 도울 수 있음을 주장합니다.

- **Performance Highlights**: 수집된 데이터 분석 결과, 설명에 잡음이 있는 경우 사용자 평가가 낮은 점수를 받았지만, 응답의 품질과 무관하게 점수가 민감하지 않았습니다. 설명의 제시 형식은 이 설정에서 중요한 요소가 아닐 수 있다는 결론도 얻어냈습니다. 전반적으로, 이 사용자 연구는 대화형 시스템 응답의 유용성과 사용자 평가에 미치는 잡음과 설명의 제시 모드의 영향에 대해 일반화 가능한 결과를 제공합니다.



### TF4CTR: Twin Focus Framework for CTR Prediction via Adaptive Sample  Differentiation (https://arxiv.org/abs/2405.03167)
- **What's New**: 새로운 CTR(click-through rate) 예측 프레임워크인 TF4CTR(Twin Focus Framework for CTR)가 제안되었습니다. 이 프레임워크는 기존의 병렬 구조 CTR 모델의 한계를 극복하고, 다양한 복잡성을 가진 샘플을 효과적으로 구분하여 처리할 수 있도록 설계되었습니다. 본 연구에서는 샘플 선택 임베딩 모듈(SSEM), 동적 융합 모듈(DFM), 그리고 트윈 포커스(Twin Focus) 손실 함수를 통해 모델의 예측 및 일반화 능력을 강화하는 것을 목표로 하고 있습니다.

- **Technical Details**: TF4CTR 프레임워크는 SSEM을 사용하여 입력 샘플에 가장 적합한 인코더를 적응적으로 선택하고, DFM은 다양한 인코더의 출력을 동적으로 융합하여 더 정확한 예측 결과를 도출합니다. 또한, 트윈 포커스 손실 함수는 복잡한 인코더에 대해 어려운 샘플을, 단순 인코더에 대해 쉬운 샘플을 학습하도록 특화된 훈련 신호를 제공하여 샘플 불균형의 영향을 줄이고 모델의 일반화 능력을 향상시킵니다.

- **Performance Highlights**: 실험 결과, TF4CTR 프레임워크는 5개의 실제 데이터셋에서 효과적이고 호환 가능함을 확인하였습니다. 이는 다양한 대표적 기준 모델들에 대해 모델-불특정적 방식으로 성능을 향상시킬 수 있음을 보여줍니다. 또한, 이 프레임워크는 기능 상호작용 정보(Feature Interaction Information)를 보다 효과적으로 포착하여 최종 예측 정확도를 높이는 데 기여합니다.



### Vector Quantization for Recommender Systems: A Review and Outlook (https://arxiv.org/abs/2405.03110)
- **What's New**: 이 논문은 추천 시스템(recommender systems)에서의 벡터 양자화(vector quantization, VQ) 방법에 대한 체계적인 탐구를 제공합니다. VQ는 데이터를 더 작은 수의 프로토타입 벡터로 압축하여 표현하는 기술입니다. 특히, 이 논문은 효율 및 품질 지향적 접근 방식에서 VQ의 사용을 분석하며, 대규모 언어 모델(large language models, LLMs)과의 통합을 통해 추천 품질을 향상시킬 수 있는 방안을 모색합니다.

- **Technical Details**: 이 연구에서는 다양한 VQ 기술, 표준 VQ, 병렬 VQ, 순차 VQ 및 차별화 가능한 VQ(differentiable VQ)를 포함하여 벡터 양자화의 고전적 및 현대적 접근법을 소개합니다. 또한, 이 논문은 추천 시스템에서 VQ의 적용을 위한 시스템적인 분류(taxonomies)를 제공하고, 훈련 단계, 적용 시나리오, VQ 기술 및 양자화 대상에 대해 상세히 논의합니다.

- **Performance Highlights**: 벡터 양자화는 유사성 검색(similarity search), 공간 압축(space compression), 모델 가속(model acceleration)과 같은 중요한 영역에서 성능을 크게 향상시키는 것으로 나타났습니다. 특히, 대규모 모델과 데이터 세트 처리에 있어서의 최적화에 효과적입니다. 또한, 다양한 모달리티 간의 상호 작용을 촉진하고, 특징을 매끄럽게 조정하여 생성 추천 프로세스를 개선하는데 기여합니다.



### Improve Temporal Awareness of LLMs for Sequential Recommendation (https://arxiv.org/abs/2405.02778)
Comments: 10 pages

- **Newsletter**: [{"What's New": "이 연구는 시간 정보 인식이 부족한 큰 언어 모델(Large Language Models, LLMs)의 한계를 극복하기 위해 설계된 새로운 프롬프팅 프레임워크 '템푸라(Tempura)'를 제안합니다. 이는 LLM을 이용한 순차적 추천(Sequential Recommendation, SRS) 과제의 성능을 향상시키기 위해 인간의 인지 과정에서 영감을 받은 세 가지 프롬프팅 전략을 사용합니다."}, {'Technical Details': '제안된 템푸라는 1) 문맥 내 학습(in-context learning) 모듈을 통해 역사적 상호 작용 시퀀스에서 순차 추천 작업을 학습합니다. 2) 시퀀스 내 클러스터 구조를 명시적으로 통합하여 모델의 이해도를 높이는 시간 구조 분석(temporal structure analysis) 모듈을 포함합니다. 3) 다양한 프롬프팅 전략에서 도출된 추천 결과를 집계하는 프롬프트 앙상블(prompt ensemble) 모듈로 구성됩니다.'}, {'Performance Highlights': 'MovieLens-1M 및 Amazon Review 데이터셋에서의 평가 결과, 템푸라는 LLMs의 순차적 추천 작업에 대한 제로샷(zero-shot) 능력을 상당히 향상시키는 것으로 나타났습니다. 이는 특히 시간 정보를 활용하는 데 있어서 기존의 방법보다 우수한 성능을 보여주며, LLM의 한계를 극복하는 데 중요한 진전을 나타냅니다.'}]



### Sign-Guided Bipartite Graph Hashing for Hamming Space Search (https://arxiv.org/abs/2405.02716)
- **What's New**: 이 연구에서는 가벼운 그래프 컨볼루션 해싱 모델인 LightGCH를 개발하고, 비파티트 그래프 해싱(Bipartite Graph Hashing, BGH)을 개선하기 위한 새로운 방법으로 sign-guided 프레임워크인 SGBGH를 제안합니다. 이는 Hamming 공간에서의 유사성을 향상시키기 위해 sign-guided 부정 샘플링(negative sampling) 및 sign-aware 대조 학습(contrastive learning)을 사용합니다.

- **Technical Details**: 기존의 BGCH 모델과 비교하여, LightGCH는 특집 산포(feature dispersion) 및 재구성 손실(reconstruction loss)과 같은 증강 방식을 제거함으로써 경량화를 달성했습니다. 이를 통해 모델과 노드 속성에 근거하여 성능을 분석하고, 이를 바탕으로 SGBGH 프레임워크를 개발하였습니다. SGBGH는 sign-guided 부정 샘플링을 통해 이웃 노드들 간의 Hamming 유사성을 끌어올리고, sign-aware 대조 학습을 통해 더 균일한 해시(embedding) 표현을 학습합니다.

- **Performance Highlights**: 실험 결과, SGBGH는 BGCH 및 LightGCH를 상당한 차이로 능가하는 해싱 성능과 훈련 효율성을 보여주었습니다. 특히, 실세계 데이터셋에서의 실증적 분석을 통해 SGBGH는 실제 인접 노드들 사이의 낮은 Hamming 유사성 문제와 전체 노드의 높은 Hamming 유사성 문제를 해결하는 것으로 나타났습니다.



### RLStop: A Reinforcement Learning Stopping Method for TAR (https://arxiv.org/abs/2405.02525)
Comments: Accepted at SIGIR 2024

- **What's New**: RLStop은 문서를 검토해야 하는 수를 최소화하며, 기존 탐색 기술 (Technology Assisted Review, TAR)에 보조를 제공하는 새로운 중단 규칙 (stopping rule)입니다. 이 규칙은 강화학습 (Reinforcement Learning, RL)에 기반하여 문서 순위에서 최적의 중단점을 식별하도록 훈련되었습니다.

- **Technical Details**: RLStop은 강화학습을 사용하여 순차적 결정을 내리는 에이전트 (agent)가 순위 목록의 문서를 검토할 때 수행됩니다. 이 방법은 연속적인 결정을 내리는 데 적합하며, 정책 (policy)에 의해 가이드됩니다. 각 상태는 배치 (batch)의 일부로 문서의 집합을 나타내며, 에이전트는 '중단 (STOP)'과 '계속 (CONTINUE)'의 두 가지 행동을 선택할 수 있습니다.

- **Performance Highlights**: 실험은 CLEF e-Health, TREC Total Recall, Reuters RCV1 등 여러 벤치마크 데이터 세트에서 수행되었고, RLStop은 목표 재현율 (target recall)을 충족하는 데 필요한 문서 검토 수를 현저히 줄이는 것으로 나타났습니다. 이는 다양한 대안적 접근법들을 능가하는 성능을 보여주었습니다.



### Axiomatic Causal Interventions for Reverse Engineering Relevance  Computation in Neural Retrieval Models (https://arxiv.org/abs/2405.02503)
Comments: 10 pages, 10 figures, accepted at SIGIR 2024 as perspective paper

- **What's New**: 이 연구는 신경 순위화 모델의 내부 결정 과정을 이해하기 위해 인과 개입을 사용하는 새로운 접근 방식을 제안합니다. 특히, 모델이 어떻게 관련성을 계산하는지 역설계하기 위한 메커니즘 해석 가능성(Mechanistic Interpretability) 방법을 사용하여 주의 집중 헤드(attention heads)가 문서의 중요도를 어떻게 평가하는지에 대한 새로운 통찰력을 제공합니다.

- **Technical Details**: 연구팀은 활성화 패치(activation patching) 설정을 통해 모델의 특정 부분에서 IR(Information Retrieval) 공리를 어느 정도 충족하는지를 평가합니다. 이 방법은 기존의 진단 데이터 세트의 한계를 극복하고, DistilBERT 기반 인코더 TAS-B를 사용하여 TFC1 공리에 따라 용어 빈도를 추적하는 메커니즘을 확인합니다.

- **Performance Highlights**: 이 연구는 TAS-B 모델이 TFC1 공리를 준수하며 용어 빈도를 파악하는 주의 집중 헤드 메커니즘을 발견함으로써 신경 순위화 모델의 해석 가능성을 향상시키는 데 기여합니다. 또한, 이러한 접근 방식은 모델의 안전한 배치를 보장하고, 순위화 능력과 안전성을 모두 향상시킬 수 있는 구성적 관련성 정의의 기초를 마련합니다.



### Characterizing the Dilemma of Performance and Index Size in  Billion-Scale Vector Search and Breaking It with Second-Tier Memory (https://arxiv.org/abs/2405.03267)
- **What's New**: 이 논문은 대규모 데이터셋에서 벡터 검색(Vector Search)을 진행할 때, 기존의 SSD 기반 그래프 및 클러스터 인덱스들이 성능 향상을 위해 크기를 늘리는 트레이드오프를 최초로 조사합니다. 이러한 문제를 해결하기 위해 저자들은 두 번째 계층 메모리(Second-tier Memory)라는 새로운 접근 방식을 제안하고 있습니다. 제안된 방식은 RDMA나 CXL 같은 빠른 연결기술을 사용해 연결된 휘발성 또는 비휘발성 메모리(NVM)를 활용합니다.

- **Technical Details**: 두 번째 계층 메모리는 접근의 세밀함(Fine-grained Access)이 특징이며, 이는 작은 랜덤 읽기(Random Read) 요구 사항과 잘 부합합니다. 연구팀은 이 메모리를 사용하여 그래프 및 클러스터 인덱스를 재설계하고, 이를 통해 인덱스 증폭(Index Amplification)을 상당히 줄이면서 최적의 성능을 도출할 수 있는 새로운 실행 엔진(Execution Engine) 및 인덱스 레이아웃(Index Layout)을 개발했습니다.

- **Performance Highlights**: 전통적인 SSD를 기반으로 한 인덱스 대비, 두 번째 계층 메모리를 활용한 그래프 인덱스는 최대 44% 줄어든 인덱스 저장 공간으로 최적의 성능을 달성할 수 있었고, 클러스터 인덱스는 인덱스 크기 증폭을 40%로 유지하면서 효과를 보였습니다. 또한, SSD에서의 일반적인 발견과는 달리, 두 번째 계층 메모리에서는 그래프 인덱스가 클러스터 인덱스보다 훨씬 더 나은 성능을 보여주었습니다.



### iSEARLE: Improving Textual Inversion for Zero-Shot Composed Image  Retrieva (https://arxiv.org/abs/2405.02951)
Comments: Extended version of the ICCV2023 paper arXiv:2303.15247

- **What's New**: 새로운 연구에서 제로-샷 구성된 이미지 검색(Zero-Shot Composed Image Retrieval, ZS-CIR)이 소개되었습니다. 이는 기존의 방대한 레이블이 지정된 데이터 세트에 대한 의존 없이 CIR을 수행할 수 있는 기능을 제공합니다. 이 연구에서는 CLIP 토큰 임베딩 공간에서 시각적 정보를 pseudo-word 토큰으로 매핑하고 상대 캡션과 결합하는 iSEARLE(improved zero-Shot composEd imAge Retrieval with textuaL invErsion) 접근 방식을 제안합니다. 또한, ZS-CIR 연구를 촉진하기 위해 CIRCO(Composed Image Retrieval on Common Objects in context) 벤치마킹 데이터 세트가 소개되었습니다.

- **Technical Details**: iSEARLE은 사전 훈련된 CLIP 비전-언어 모델을 활용하고, 최적화 기반의 Textual Inversion (OTI) 및 지식 증류를 포함하여 pseudo-word 토큰을 생성하고 사용합니다. 이 과정은 이미지-텍스트 매핑을 단순화하며, CIR 작업에서 높은 유연성과 정확성을 제공합니다. CIRCO 데이터 세트는 COCO 데이터 세트에서 파생되며, 각 쿼리별로 다수의 ground truth를 제공하여 보다 정확한 모델 평가를 가능하게 합니다.

- **Performance Highlights**: iSEARLE은 FashionIQ, CIRR, 그리고 새롭게 제안된 CIRCO를 포함하여 세 가지 다른 CIR 데이터 세트에서 최고의 성능을 달성하였습니다. 추가적으로 영역 변환(domain conversion)과 객체 구성(object composition) 설정에서도 우수한 일반화 능력을 확인할 수 있었습니다. 이 연구는 다양한 평가 설정에서의 향상된 성능을 제공하며, ZS-CIR 작업에 대한 새로운 방향을 제시하였습니다.



### MedPromptExtract (Medical Data Extraction Tool): Anonymization and  Hi-fidelity Automated data extraction using NLP and prompt engineering (https://arxiv.org/abs/2405.02664)
- **What's New**: MedPromptExtract는 비구조화된 의료 기록을 구조화된 데이터로 변환하여 추가 분석이 가능하게 하는 자동화 도구입니다. 이 도구는 반지도 학습(semi-supervised learning), 대규모 언어 모델(large language models), 자연어 처리(natural language processing), 프롬프트 엔지니어링(prompt engineering)을 결합하여 사용합니다. 특히, MedPromptExtract는 사용자의 로컬 하드웨어와 독립적으로 작동하여 서버에서 계산을 수행하므로, 실시간 데이터 처리가 가능합니다.

- **Technical Details**: MedPromptExtract는 EIGEN과 LayoutLM을 사용하여 의료 문서에서 데이터를 추출합니다. EIGEN은 광학 문자 인식(OCR)과 레이아웃 분석을 사용하고, LayoutLM은 문서 이미지에서 정보를 추출하기 위해 사전 훈련된 심층 신경망 모델입니다. 데이터 추출 과정에서는 실질적인 라벨 없이도 효과적인 학습이 가능하도록 하는 레이블링 함수(labeling functions)가 활용되며, 기계 학습을 통한 익명화 처리로 환자의 개인 정보를 보호합니다.

- **Performance Highlights**: MedPromptExtract는 914개의 퇴원 요약문 중 초기에 4개를 제외하고 852개를 처리하여 높은 처리 능력을 보여주었습니다. 사용된 지미니 API는 특정 매개변수를 설정하여 정확하고 결정적인 출력을 보장합니다. 인간 평가와의 비교에서 모델 응답은 7개의 12개 기능에 대해 AUC 0.9 이상을 달성하며, 양호한 데이터 추출 과정의 충실도를 나타냈습니다. 덕분에 데이터의 익명화 비용을 크게 줄이면서도 높은 품질의 데이터를 추출할 수 있습니다.



### Accelerating Medical Knowledge Discovery through Automated Knowledge  Graph Generation and Enrichmen (https://arxiv.org/abs/2405.02321)
Comments: 18 pages, 5 figures

- **What's New**: 의료 지식 그래프 자동화(Medical Knowledge Graph Automation, M-KGA)라는 새로운 접근 방식을 제안합니다. 이 방식은 사용자가 제공한 의료 개념을 BioPortal 온톨로지를 사용하여 의미론적으로 확장하고, 사전 훈련된 임베딩을 통합하여 지식 그래프의 완성도를 향상시킵니다.

- **Technical Details**: M-KGA는 두 가지 방법론을 도입합니다: 클러스터 기반 접근법(cluster-based approach)과 노드 기반 접근법(node-based approach). 이러한 방법들은 지식 그래프 내에서 숨겨진 연결을 발견하는 데 사용됩니다.

- **Performance Highlights**: 전자 건강 기록(Electronic Health Records, EHRs)에서 자주 발생하는 100개의 의료 개념을 사용한 엄격한 테스팅을 통해 M-KGA 프레임워크는 현재의 지식 그래프 자동화 기술의 한계를 해결할 수 있는 잠재력을 보여주었습니다.




### Turbo-CF: Matrix Decomposition-Free Graph Filtering for Fast  Recommendation (https://arxiv.org/abs/2404.14243)
Comments: 5 pages, 4 figures, 4 tables; 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2024) (to appear) (Please cite our conference version.)

- **What's New**: 이 연구에서는 교육(Training) 과정과 행렬 분해(Matrix Decomposition) 과정 없이도 빠르고 정확한 추천을 제공할 수 있는 새로운 협업 필터링(Collaborative Filtering, CF) 방식인 Turbo-CF를 소개합니다. Turbo-CF는 다항식 그래프 필터(Polynomial Graph Filter)를 사용하여 높은 계산 비용을 요구하는 기존의 방식들과는 달리, GPU 등의 현대 컴퓨터 하드웨어를 최대한 활용하여 빠른 성능을 보여줍니다. 또한, 네트워크 자체정렬(Network Alignment, NA) 방식에 대한 새로운 접근법인 Grad-Align+를 제시하는데, 이는 추가 정보 없이도 높은 정확도를 달성할 수 있도록 노드 속성을 확장하는 방법을 사용합니다.

- **Technical Details**: Turbo-CF는 품목 간 유사성 그래프(Item-Item Similarity Graph)를 구성하고, 저주파 신호만을 유지하는 다항식 저역통과 필터(Polynomial Low-Pass Filter, LPF)를 설계하여 사용합니다. 이 방식은 명시적인 행렬 분해를 요구하지 않으며, 실시간으로 추천을 제공할 수 있는 아키텍처를 갖추고 있습니다. 한편, Grad-Align+는 두 개의 그래프 신경망(Graph Neural Networks, GNNs)을 통해 노드 표현을 생성하고, 이를 기반으로 점진적으로 네트워크 간 노드 쌍을 찾아냄으로써 조정된 크로스네트워크 이웃 쌍(Aligned Cross-Network Neighbor-Pair, ACN)의 영향을 증대시키는 새로운 측정 방법을 사용합니다.

- **Performance Highlights**: Turbo-CF는 실제 벤치마크 데이터셋에서 1초 미만의 실행 시간을 달성하면서도 최고 수준의 추천 정확도를 보여주었습니다. 이는 기존의 그래프 필터링 기반 CF 방식과 비교했을 때 획기적인 개선을 의미합니다. Grad-Align+는 실제와 합성 데이터셋을 사용한 실험에서 효과적으로 품질을 향상시키고, 기존의 최첨단 NA 방법보다 월등한 성능을 나타내는 것을 입증했습니다.



### Collaborative Filtering Based on Diffusion Models: Unveiling the  Potential of High-Order Connectivity (https://arxiv.org/abs/2404.14240)
Comments: 10 pages, 6 figures, 4 tables; 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2024) (to appear) (Please cite our conference version.)

- **What's New**: 새로운 협업 필터링 방법인 CF-Diff를 도입했습니다. 이 방법은 사용자-아이템 상호작용의 생성 과정을 모방하는데 적합한 확산 모델을 기반으로 하여, 다중 홉(high-order connectivity) 이웃과 함께 협업 신호를 전체적으로 활용합니다.

- **Technical Details**: CF-Diff는 전진 확산 과정(forward-diffusion process)에서 임의의 노이즈를 사용자-아이템 상호작용에 추가하고, 역-제거 과정(reverse-denoising process)에서 우리의 학습 모델인 교차 주의-가이드 다중 홉 오토인코더(CAM-AE, Cross-Attention-Guided Multi-Hop Autoencoder)를 통해 원래의 상호작용을 점진적으로 복구합니다. CAM-AE는 주의-도움 오토인코더(attention-aided AE module)와 다중 홉 교차 주의 모듈(multi-hop cross-attention module)로 구성됩니다.

- **Performance Highlights**: CF-Diff는 벤치마크 추천 방법보다 우수한 성능을 보여주며, 최고의 경쟁자에 비해 최대 7.29%의 높은 성과를 달성함을 실험을 통해 입증하였습니다. 또한, 이 모델은 계산을 줄이면서도 원래의 교차 주의에서 생성된 임베딩과 밀접하게 일치하는 임베딩을 생성할 수 있으며, 사용자나 아이템의 수에 비례하여 계산 효율성이 선형적으로 확장될 수 있음을 보여줍니다.



### SPLATE: Sparse Late Interaction Retrieva (https://arxiv.org/abs/2404.13950)
Comments: To appear at SIGIR'24 (short paper track)

- **What's New**: 이 연구에서는 ColBERTv2 모델의 적응형인 SPLATE을 소개하고 있습니다. 이 모델은 'MLM adapter'를 사용하여 고정된 토큰 임베딩을 희소 어휘 공간으로 매핑하여, 전통적인 희소 검색 기법을 사용하여 후보 생성 단계를 수행할 수 있게 합니다. SPLATE는 CPU 환경에서 ColBERT를 실행하는데 특히 유리하며, PLAID ColBERTv2 엔진과 동등한 효과를 달성했습니다.

- **Technical Details**: SPLATE 모델은 ColBERTv2의 동결된 표현을 SParse LATE interaction, 즉 희소 늦은 상호작용 방식으로 적응시키는 방법을 제안합니다. SPLATE는 두 층 MLP를 학습하여 토큰의 변형된 표현을 어휘로 선형 매핑하고, 이를 통해 희소 SPLADE 벡터를 생성합니다. 이 모델은 기존의 MLP 모델을 사용하여 표현을 2배 줄인 후 원래 차원으로 다시 투영하는 방식을 사용합니다.

- **Performance Highlights**: SPLATE ColBERTv2 파이프라인은 문서 50개를 후보로 하여 10ms 이하에서 검색할 수 있으며, PLAID ColBERTv2 엔진과 동일한 효과를 나타냈습니다. 이는 희소 검색 기술을 사용하면서도 높은 효율성과 효과성을 동시에 달성한 결과입니다.



### Multi-Level Sequence Denoising with Cross-Signal Contrastive Learning  for Sequential Recommendation (https://arxiv.org/abs/2404.13878)
- **What's New**: 이 논문에서는 사용자 상호작용 시퀀스의 노이즈 항목을 효과적으로 처리하는 새로운 모델인 Multi-level Sequence Denoising with Cross-signal Contrastive Learning (MSDCCL)을 제안합니다. 이 모델은 'soft' 및 'hard' 탈노이즈(denoising) 전략을 동시에 활용하여 순차적 추천(Sequential Recommendation)에서 노이즈의 영향을 완화합니다. 또한, 타겟(Target)에 의식적인 사용자 관심 추출기(target-aware user interest extractor)를 통합하여 사용자의 장기 및 단기 관심사를 모델링합니다.

- **Technical Details**: MSDCCL은 소프트 레벨과 하드 레벨의 두 가지 탈노이즈 모듈을 포함합니다. 소프트 레벨 탈노이즈는 낮은 주목도 가중치를 할당하여 노이즈 항목을 완화시키고, 하드 레벨 탈노이즈는 Gumbel-Softmax 함수를 사용하여 관련 없는 항목을 제거합니다. 또한, 두 서브 모듈 간의 정보 교환을 안내하는 크로스 시그널 대조 학습(Cross-signal Contrastive Learning) 층이 도입됩니다.

- **Performance Highlights**: MSDCCL은 ML-100k, Beauty, Sports, Yelp 및 ML-1M 등 다섯 개의 공개 데이터셋에서 광범위한 실험을 통해 상태 최고의 기준 방법들을 상당히 뛰어넘는 성능을 보여주었습니다. 예를 들어, ML-100k 데이터셋에서 HR@5에서 HSD+BERT4Rec 및 AC-BERT4Rec과 비교하여 MSDCCL은 각각 100.00% 및 233.96%의 상대적 성능 향상을 보였습니다. 이러한 결과는 이 모델의 강력한 잠재력을 입증합니다.



### General Item Representation Learning for Cold-start Content  Recommendations (https://arxiv.org/abs/2404.13808)
Comments: 14 pages

- **What's New**: 이 논문에서는 콜드-스타트 상황에서 사전 분류 레이블 없이 Transformer 기반 아키텍처를 사용하여 다양한 모달(모드)들을 자연스럽게 통합할 수 있는 새로운 도메인/데이터 무관 아이템 표현 학습 프레임워크를 제안합니다. 이 접근 방식은 풍부한 원시 콘텐츠 정보를 활용하여 사용자의 세밀한 취향을 더 잘 보존할 수 있는 추천 시스템을 가능하게 합니다.

- **Technical Details**: 제안된 모델은 아이템의 다양한 특성 간의 모달 정렬을 갖추고 있으며, Transformer 기반 구조를 채택하여 사용자 활동만으로 종단 간(end-to-end) 훈련이 가능합니다. 이는 새로운 아이템이나 사용자에게도 개인화된 추천을 생성할 수 있는 능력을 향상시킬 뿐만 아니라, 분류 레이블에 의존하지 않고 미세한 선호도 차이를 인식할 수 있도록 합니다.

- **Performance Highlights**: 실제 세계 영화 및 뉴스 추천 벤치마크에서 수행된 광범위한 실험을 통해, 제안된 프레임워크는 상태 기반 최신 기술보다 더 세밀한 사용자 취향을 보존하며, 다양한 도메인에서 대규모로 적용될 수 있는 것을 확인하였습니다.



### Beyond Collaborative Filtering: A Relook at Task Formulation in  Recommender Systems (https://arxiv.org/abs/2404.13375)
Comments: Working paper

- **What's New**: 추천 시스템(Recommender Systems, RecSys)은 다양한 애플리케이션에서 필수적이 되었으며 우리 일상 경험에 깊은 영향을 끼치고 있습니다. 그러나 실제 상황에서 연구 과제의 정의를 단순화하고 의사결정 결과보다 의사결정 과정을 모델링하는데 너무 많은 중점을 두는 경향이 있습니다. 이러한 접근은 연구의 통합된 이해의 부족을 초래할 수 있습니다. 실제 사용자의 의사결정 과정 중에 접근할 수 있는 정보와 모델이 접근할 수 있는 입력 사이의 불일치가 존재하지만 모델은 여전히 사용자의 결정을 예측해야 합니다.

- **Technical Details**: 이 논문에서는 기존의 정적인 사용자-아이템 상호작용 행렬에서 빠진 값을 예측하는 것이 아니라, 동적이고 변화하는 맥락 속에서 다음 사용자 상호작용을 예측하는 것으로 추천 시스템을 개념화합니다. 이는 협업 필터링(collaborative filtering)이 역사적 기록에서 일반적인 선호도를 학습하는 데 효과적이지만, 실제 설정에서 동적인 맥락 요인(contextual factors)을 고려하는 것이 중요함을 강조합니다.

- **Performance Highlights**: 연구 과제를 도메인 특화 데이터셋(domain-specific datasets)과 응용 시나리오(application scenarios)를 기반으로 정의하면 보다 통찰력 있는 결과를 도출할 수 있습니다. 이러한 접근은 다양한 응용 시나리오에 대해 실행 가능한 해결책과 효과적인 평가를 가능하게 합니다.



### Two-Step SPLADE: Simple, Efficient and Effective Approximation of SPLADE (https://arxiv.org/abs/2404.13357)
Comments: published in Findings at ECIR'24

- **What's New**: SPLADE (Sparse Lexical Deep)와 관련된 새로운 쿼리 처리 전략을 제안하고 있습니다. 이는 이전의 단일 단계 SPLADE 처리보다 반응 시간을 최대 30배 개선하며, 도메인 내 및 도메인 외 데이터셋에서 유의미한 결과를 보여주고 있습니다.

- **Technical Details**: 이 연구에서 저자들은 SPLADE 벡터를 이용한 첫 번째 단계에서 가지치기(pruning) 및 재가중화(reweighting) 된 버전을 사용하며, 두 번째 단계에서는 원본 SPLADE 벡터를 사용하여 초기 단계에서 검색된 상위 문서들을 재점수화(rescoring)합니다. 이 방식은 적응형 최근접 이웃 탐색(approximated nearest neighbor search)에 영감을 받은 것으로, 효율성과 효과성 사이의 균형을 잘 맞추고 있습니다.

- **Performance Highlights**: 이 새로운 전처리 방식은 도메인 내 데이터셋에서 평균 및 꼬리 반응 시간을 각각 최대 30배 및 40배 개선하고, 도메인 외 데이터셋에서는 12배에서 25배의 평균 응답 시간 개선을 보여줍니다. 또한 60%의 데이터셋에서 통계적으로 유의미한 차이 없이 효율성 및 효과성을 유지합니다.



### MARec: Metadata Alignment for cold-start Recommendation (https://arxiv.org/abs/2404.13298)
- **What's New**: 이 논문에서는 새로운 MARec(Metadata Alignment for cold-start Recommendation) 이라는 방법을 제안하며, MARec는 콜드 스타트(Cold-start) 추천에서 경쟁력 있는 성과를 달성할 뿐만 아니라, 웜 스타트(Warm-start) 설정에서도 상태 최고 기술(SOTA: State of the Art)과 경쟁합니다. 콜드 스타트 문제는 새로운 사용자나 아이템에 대해 적절한 추천을 제공하는 것이며, 이는 전통적인 협업 필터링이 이전 사용자 상호 작용에 크게 의존하기 때문에 어려움이 있습니다. MARec는 이를 해결하기 위해 아이템 메타데이터를 활용하여 새로운 아이템들이 기존의 유사성 공간에 통합될 수 있도록 정규화 항을 사용합니다.

- **Technical Details**: MARec는 기존의 메트릭스 인수분해(Matrix Factorization) 및 오토인코더(Autoencoder) 접근방식을 보완하고, 아이템 메타 데이터를 기반으로 아이템 간의 유사성을 추정하여 사용자-아이템 상호작용의 유사성과 맞추어 학습합니다. 이를 통해 기존의 SOTA 모델과 결합하여 콜드 스타트 아이템에 대해 효과적인 추천을 실행할 수 있습니다. 또한, 대규모 언어 모델(Large Language Models, LLM)의 임베딩을 활용하여 의미적 특성을 적용한 추가적인 성능 향상이 있음을 증명합니다.

- **Performance Highlights**: MARec는 네 가지 콜드 스타트 데이터셋에서 기존 SOTA 결과를 +8.4%에서 +53.8%까지 상회하는 성과를 보여줍니다. 또한, 의미적 특성을 활용할 때의 추가적인 이득은 +46.8%에서 +105.5% 사이인 것으로 나타났습니다. MARec는 학습 시간면에서도 기존의 최고 수행 기준 모델보다 훨씬 빠르며, 웜 스타트 설정에서는 평균적으로 SOTA 결과에 단 0.8%만 뒤쳐지는 성과를 보입니다.



### STaRK: Benchmarking LLM Retrieval on Textual and Relational Knowledge  Bases (https://arxiv.org/abs/2404.13207)
Comments: 25 pages, 7 figures

- **What's New**: 새로운 반구조적 검색 벤치마크인 STARK는 비정형(예: 제품 설명) 및 구조화된 정보(예: 제품 엔티티 관계)를 모두 통합한 텍스트 및 관계형 지식 베이스에서 정보를 검색하는 것을 목표로 합니다. 이전 연구는 주로 텍스트 검색 또는 관계형 검색을 별개의 주제로 다뤄 왔지만 STARK는 두 모달리티를 통합하여 사용자 질의의 복잡성을 해결하려고 시도합니다.

- **Technical Details**: STARK는 대규모 텍스트와 수백만 개의 엔티티 관계를 포함하는 세 가지 SKB(Semi-structured Knowledge Bases)를 기반으로 자연스럽고 현실적인 사용자 쿼리를 시뮬레이션하고 정확한 ground-truth (정답)를 구성하는 파이프라인을 제안합니다. 이를 통해 제품 추천, 학술 논문 검색, 정밀의학 문의 등 다양한 실제 응용 프로그램을 커버합니다.

- **Performance Highlights**: STARK 벤치마크는 검색 시스템, 특히 대규모 언어 모델(Large Language Models: LLM)을 기반으로 한 접근 방식의 성능을 평가하는 데 있어 중요한 도전 과제를 제시합니다. 이는 현재의 검색 및 LLM 시스템이 텍스트와 관계적 요구 사항을 모두 처리하는 데 어려움을 겪고 있음을 시사하며, 추후 이 분야에서의 연구와 개선이 필요함을 강조합니다.



### SHE-Net: Syntax-Hierarchy-Enhanced Text-Video Retrieva (https://arxiv.org/abs/2404.14066)
- **What's New**: 이 연구에서는 동영상과 텍스트 간의 모달리티 갭(모드 차이)을 해소하기 위해 새로운 접근 방식을 제시하고 있습니다. 제안된 SHE-Net(Syntax-Hierarchy-Enhanced Text-Video Retrieval Method)은 텍스트의 의미론적 및 구문적 계층 구조를 이용하여 동영상 콘텐츠의 통합과 텍스트-비디오 유사성 계산을 안내합니다.

- **Technical Details**: SHE-Net는 텍스트의 구문 계층을 사용하여 비디오 인코딩과 텍스트-비디오 정렬 모듈을 가이드하는 새로운 방식을 도입하며, 텍스트의 문법 구조를 명시적으로 활용하여 더 세밀한 시각적 표현을 유도합니다. 그리고 각각의 동사와 명사를 비디오에서 해당하는 프레임과 이미지 영역에 매핑하여, 정보가 풍부한 비디오 특징을 추출합니다.

- **Performance Highlights**: SHE-Net은 MSR-VTT, MSVD, DiDeMo, ActivityNet의 네 가지 공개 텍스트-비디오 검색 데이터셋에서 평가되었으며, 실험 결과와 손실(Ablation) 연구는 제안된 방법의 이점을 확인시켜 줍니다. 이는 기존의 텍스트-비디오 검색 방법들이 간과했던 구문 구조를 활용함으로써 기능을 향상시킨 결과로 볼 수 있습니다.



### Track Role Prediction of Single-Instrumental Sequences (https://arxiv.org/abs/2404.13286)
Comments: ISMIR LBD 2023

- **What's New**: 이 연구에서는 딥 러닝(deep learning) 모델을 활용하여 단일 악기 음악 시퀀스에서 트랙 역할(track-role)을 자동으로 예측하는 새로운 방법을 소개합니다. 이는 AI 음악 생성 및 분석 분야에서의 미래 응용에 큰 가능성을 제시합니다.

- **Technical Details**: 심볼릭 도메인(symbolic domain)과 오디오 도메인(audio domain)에서 MIDI와 오디오 데이터를 사용하여 트랙 역할을 분류하기 위해 두 도메인에 대해 세분화된 사전 훈련된 모델(pre-trained models)을 활용했습니다. 심볼릭 도메인에서는 MusicBERT 모델을, 오디오 도메인에서는 주목 기능 융합(attention feature fusion)이 포함된 PANNs 모델을 사용하여 미세 조정(fine-tuning)하였습니다.

- **Performance Highlights**: 심볼릭 도메인에서는 87%의 예측 정확도를, 오디오 도메인에서는 84%의 예측 정확도를 달성했습니다. 특히, 사전 훈련된 모델을 미세 조정하는 전략이 처음부터 모델을 훈련하는 것보다 일관되게 우수한 성능을 보였습니다.



### Development and Evaluation of Dental Image Exchange and Management  System: A User-Centered Perspectiv (https://arxiv.org/abs/2206.01966)
Comments: 3 figures, 5 tables

- **What's New**: 이 연구는 우르미야(Urmia) 시의 사설 치과 분야에서 이미지 교환 시스템을 개발하기 위해 사용자 중심 방법(user-centered methods)과 프로토타이핑(prototyping)을 사용하여 설계되었습니다. 이는 의료 환경에 특화된 시스템 설계를 강조하며, 사용자의 요구를 바탕으로 한 소프트웨어 개발을 진행하였습니다.

- **Technical Details**: 소프트웨어 개발 생명 주기(software development life cycle)의 각 단계를 기반으로 정보를 수집하였으며, 대상 사용자의 필요를 파악하기 위해 인터뷰와 관찰이 사용되었습니다. 객체 지향 프로그래밍(Object-oriented programming)을 사용하여 프로토타입이 개발되었고, 이 프로토타입은 포커스 그룹 세션에서 평가되어 사용자의 만족을 이끌어 내었습니다.

- **Performance Highlights**: 개발된 시스템은 사용의 용이성(ease of use), 보안(security), 모바일 애플리케이션(mobile apps)이 가장 중요한 요구사항으로 반영되었습니다. 또한, 사용자 인터페이스 디자인(user interface design)과 유용성(usefulness)이 주요 고려 요소로 지적되었으며, 환자의 이미지 클립을 잃어버리거나 놓치지 않기 때문에 환자에게 방사선 노출도 감소시키는 장점이 있습니다.




New uploads on arXiv(cs.CV)

### CodePlot-CoT: Mathematical Visual Reasoning by Thinking with Code-Driven Images (https://arxiv.org/abs/2510.11718)
- **What's New**: 이번 논문에서는 시각적 도움을 요구하는 수학 문제 해결을 위해 CodePlot-CoT라는 새로운 코딩 기반의 Chain-of-Thought(코드 기반 사고 방식)를 제안합니다. 이 접근법은 Vision Language Models(VLMs)를 활용해 텍스트 추론과 실행 가능한 플로팅 코드 생성을 결합하여 '시각적 사고(visual thought)'로 수학 문제를 해결합니다. 또한, 178K 샘플로 구성된 Math-VR이라는 대규모 이중 언어 데이터셋과 벤치마크를 새롭게 구축하여 이를 바탕으로 성능을 평가하고 있습니다.

- **Technical Details**: 이 연구의 핵심은 VLMs가 코드 생성을 통해 시각적 추론을 수행할 수 있도록 하는 것입니다. 직접적으로 이미지를 생성하는 대신, 모델이 실행 가능한 플로팅 코드를 출력하여 이를 이미지로 렌더링하고, 이렇게 생성된 이미지를 VLM의 추론 과정에 다시 입력하여 문제를 해결합니다. 이를 위해 MatplotCode라는 고품질 이미지-코드 변환기를 개발하였고, 이 변환기를 사용하여 CodePlot-CoT 모델을 훈련시켰습니다.

- **Performance Highlights**: 실험 결과, 제안된 모델은 기존 기본 모델에 비해 최대 21%의 성능 향상을 보였습니다. 이를 통해 코드 기반의 사고 방식이 효과적임을 입증하였으며, 다중 모달 수학 추론의 새로운 방향을 제시합니다. 연구팀은 데이터셋과 코드, 사전 훈련된 모델을 공개하여 향후 연구에 기여할 계획입니다.



### Ev4DGS: Novel-view Rendering of Non-Rigid Objects from Monocular Event Streams (https://arxiv.org/abs/2510.11717)
- **What's New**: 본 논문은 비강체 변형 물체의 새로운 뷰 렌더링을 위한 최초의 접근 방식인 Ev4DGS를 소개합니다. 기존의 접근 방식들은 추가적인 RGB 입력이 필요했으나, Ev4DGS는 단일 모노큘러 이벤트 스트림만을 사용하여 이를 가능하게 합니다. 이벤트 카메라를 활용하여 비강체 물체의 3D 형태를 효과적으로 추정할 수 있음을 보여줍니다.

- **Technical Details**: 강체와 비강체 물체의 3D 복원과 새로운 뷰 합성을 위한 기술적인 세부 사항을 제시하였습니다. 사건 기반의 두 단계 훈련 과정을 통해 객체의 모션을 추적하고 4D 가우시안 방식으로 객체의 외형을 표현합니다. 이 프레임워크는 이벤트만으로 훈련할 수 있는 손실 함수를 정의하여, RGB 입력이 필요하지 않다는 이점을 가지고 있습니다.

- **Performance Highlights**: Ev4DGS는 기존 경쟁 기법들과 비교했을 때 높은 정확도를 기록하며, 실험을 통해 그 성능을 입증하였습니다. 본 연구는 새로운 합성 및 실제 데이터 시퀀스를 제공하여 기존 방법들과의 정량적 및 정성적 평가가 가능하도록 하였습니다. 또한, RGB 이미지의 명시적 복원을 요구하지 않음으로써 실용적인 한계를 극복했습니다.



### Point Prompting: Counterfactual Tracking with Video Diffusion Models (https://arxiv.org/abs/2510.11715)
Comments:
          Project link: this https URL

- **What's New**: 본 논문에서는 비디오 생성 및 추적의 밀접한 관계를 활용하여 사전 학습된 비디오 diffusion 모델이 제로샷(zero-shot) 포인트 추적을 수행하는 방법을 제시합니다. 연구진은 특히 쿼리 포인트에 독특하게 색칠된 마커를 두고, 비디오의 중간 노이즈 수준에서 나머지 비디오를 재생성하여 마커가 프레임을 통해 전파되도록 합니다. 이를 통해 자연 비디오에서는 보기 힘든 마커가 생성되것을 여러 영상 모델에 대한 실험을 통해 성능이 향상됨을 입증하였습니다.

- **Technical Details**: 본 연구에서 제안하는 접근법은 거짓 현실 모델링(counterfactual modeling)을 기반으로 하며, 초기 비디오 프레임에서 쿼리 포인트를 시각적으로 표시합니다. 이어서 SDEdit를 사용하여 비디오를 재생성하고, 이미지 처리 기술을 통해 각 프레임에서 포인트의 위치를 추적합니다. 이 모델은 기존의 고급 이해 과제와는 달리 텍스트 프롬프트로 쉽게 유도할 수 없는 추적 기능을 자동적으로 발생시키는 가능성을 보여줍니다. 초기 프레임을 수정하지 않고 부정적 프롬프트로 사용하는 간단한 방법을 통해 마커를 유지하는데 성공하였습니다.

- **Performance Highlights**: 실험 결과, 사전 학습된 비디오 diffusion 모델이 직접적으로 시각적 추적기로 사용할 수 있으며, 심지어 가림 처리(occlusion)를 통한 추적도 가능함을 보여주었습니다. 새로운 diffusion 프롬프트 전략을 사용하여 포인트를 신뢰성 있게 전파할 수 있었고, 반복적인 정제를 통해 추적 성능이 크게 향상되었습니다. 우리는 이전 제로샷(zero-shot) 추적 방법을 능가하는 성능을 기록하였으며, 이는 대규모 사전 학습된 비디오 diffusion 모델의 가능성을 한층 더 이해하는 데 기여할 것입니다.



### DiT360: High-Fidelity Panoramic Image Generation via Hybrid Training (https://arxiv.org/abs/2510.11712)
Comments:
this https URL

- **What's New**: 본 논문에서는 DiT360이라는 DiT 기반 프레임워크를 제안하여, 통합 훈련(hybrid training) 과정을 이용해 원근(perspective) 이미지와 파노라마(panoramic) 데이터를 기반으로 파노라마 이미지를 생성합니다. 기존 모델 디자인에 초점을 맞춘 방법들과는 달리, 본 연구에서는 대규모 고품질 실제 파노라마 데이터의 부족 문제를 해결하고자 하였습니다. DiT360은 여러 주요 모듈을 통해 서로 다른 도메인 간 변환(inter-domain transformation) 및 내부 도메인 증강(intra-domain augmentation)을 수행합니다.

- **Technical Details**: DiT360은 VAE(Variational Autoencoder) 이미지 레벨(pre-VAE)과 토큰(level post-VAE) 레벨 모두에서 여러 모듈을 적용합니다. 이미지 수준에서는 원근 이미지 가이드를 통해 교차 도메인 지식을 통합하며, 파노라마 이미지를 보완하고 다양성 및 사실감을 정규화합니다. 토큰 수준에서는 회전 및 왜곡 인식을 강화하기 위해 circular padding, yaw loss 및 cube loss 등의 하이브리드 감독(hybrid supervision)을 적용하려고 합니다.

- **Performance Highlights**: DiT360은 텍스트-투-파노라마(text-to-panorama), 인페인팅(inpainting), 아웃페인팅(outpainting) 과제에서 기존 방법들과 비교할 때 경계 일관성(boundary consistency)과 이미지 충실도(image fidelity) 모두에서 더 나은 성능을 보였습니다. 특히, DiT360은 Matterport3D 검증 세트에서 최첨단 성능을 달성하여, FID, Inception Score, BRISQUE와 같은 아홉 가지 측정 기준에서 기존 방법들을 초월하였습니다. 이 방법은 고해상도 및 사실적인 파노라마 이미지를 생성할 수 있는 능력을 가지며, 추가적인 조정 없이 인페인팅 및 아웃페인팅 작업을 지원합니다.



### Bayesian Topological Convolutional Neural Nets (https://arxiv.org/abs/2510.11704)
- **What's New**: 이 논문에서는 비확정성 및 데이터 부족 문제를 해결하기 위해 Bayes 목표를 포함한 새로운 Bayesian topological CNN(BTCNN)을 제안합니다. 이 모델은 맨홀의 정보를 활용하여 훈련 속도를 높이고 네트워크 파라미터에 대한 사전 분포를 설정함으로써 예측의 정확성을 향상시킵니다. 특히, 일관성 조건을 포함하여 훈련하는 방식이 특징이며, 이는 새로운 네트워크 구조의 성능을 향상시키는 데 기여합니다.

- **Technical Details**: BTCNN 아키텍처는 Topological Data Analysis(TDA) 및 Bayesian 추론을 통합하여 이미지 분류 작업에서 높은 정확도를 달성하도록 설계되었습니다. 이 모델은 변동성(variability)을 포함하여 네트워크 파라미터의 사전 분포를 고려하며, 수렴된 사후 분포(posterior distribution)로 훈련 과정을 타이트하게 구성합니다. 특히, 훈련 데이터가 적거나 변질된 경우에도 정확한 예측을 생성하는 데 중요한 역할을 합니다.

- **Performance Highlights**: BTCNN은 여러 기준 이미지 분류 데이터셋에서 전통적인 CNN, Bayesian Neural Networks(BNNs), Topological CNNs보다 우수한 성능을 보입니다. 한정된 데이터 환경에서의 성능이 개선되었으며, 테스트 결과 모델이 과도한 확신을 내리지 않고 불확실성을 잘 정량화한다는 점을 입증했습니다. 이러한 성과는 더 효율적이고 강력한 이미지 분류를 위한 새로운 하이브리드 접근법의 가능성을 강조합니다.



### Diffusion Transformers with Representation Autoencoders (https://arxiv.org/abs/2510.11690)
Comments:
          Technical Report; Project Page: this https URL

- **What's New**: 이 논문에서는 기존의 VAE(Variational Autoencoder)를 대신하여 사전 훈련된 표현 인코더(예: DINO, SigLIP, MAE)를 활용한 새로운 표현 기반 자동 인코더(RAEs)를 제안합니다. 이 모델들은 소스에서 고품질의 재구성과 의미적으로 풍부한 잠재 공간을 제공하며, 확장 가능한 트랜스포머 아키텍처를 허용합니다. 기존의 접근 방식의 한계를 극복하고, 생성 성능을 향상시키기 위한 새로운 방법론을 제시했습니다.

- **Technical Details**: RAE는 VAE를 대체하는 방식으로, 사전 훈련된 표현 인코더와 훈련된 디코더를 결합하여 구성됩니다. 이러한 구조는 의미론적으로 풍부하고 구조적으로 일관된 잠재 공간을 생성하며, 기존의 고차원 잠재 공간에서도 확실한 조합을 가능하게 합니다. 또한, 고차원 표현이 오히려 장점이 될 수 있는 잠재 공간 내에서 안정적이고 효율적으로 디퓨전 모델 훈련을 수행할 수 있음을 시연했습니다.

- **Performance Highlights**: RAE 기반의 DiTDH는 ImageNet에서 256x256 해상도에서 FID(Frechet Inception Distance) 점수 1.51을 달성하며, AutoGuidance 없이도 훌륭한 생성 성능을 보여주었습니다. 이는 RAEs가 기존의 VAE에 대한 대안으로서 효과적이라는 것을 입증합니다. 전반적으로 이러한 결과는 자동 인코딩의 역할을 단순한 압축 방식에서 의미론적 표현의 기초로 재정립합니다.



### Beyond 'Templates': Category-Agnostic Object Pose, Size, and Shape Estimation from a Single View (https://arxiv.org/abs/2510.11687)
- **What's New**: 이번 논문에서는 기존 방법들의 한계를 넘어 6D pose, size, shape을 동시에 예측하는 카테고리 비의존적(unified, category-agnostic) 프레임워크를 제안합니다. 이 프레임워크는 CAD 모델이나 템플릿, 카테고리 레이블 없이 단일 RGB-D 이미지로부터 6D 정보를 추정할 수 있습니다. 또한, Mixture-of-Experts로 강화된 Transformer 인코더를 활용하여 2D 비전 특징과 3D 포인트 클라우드를 융합합니다.

- **Technical Details**: 제안된 모델은 2개의 경량 디코더를 사용하여 6D pose와 shape을 동시에 예측합니다. 이 아키텍처는 28 FPS의 실시간 추론을 지원하며, SOPE 데이터셋의 인공 데이터만으로 훈련되었습니다. 이 과정에서 카테고리 간 일관성을 유지하면서도 카테고리 비의존적 추론을 가능하게 합니다.

- **Performance Highlights**: 훈련된 모델은 300개 이상의 카테고리를 포함한 다양한 벤치마크에서 평가되어 최고 성능을 나타냈습니다. 특히, 이전의 카테고리 기반 방법들에 비해 새로운 실제 객체에 대해서도 강력한 제로샷(zero-shot) 일반화를 달성했습니다. 이는 로봇 공학과 임베디드 AI에 대한 오픈 세트(open-set) 6D 이해를 위한 새로운 기준을 세운 것입니다.



### FACE: Faithful Automatic Concept Extraction (https://arxiv.org/abs/2510.11675)
Comments:
          39th Conference on Neural Information Processing Systems (NeurIPS 2025)

- **What's New**: 위 논문에서는 FACE(Faithful Automatic Concept Extraction)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 Kullback-Leibler (KL) 발산 정규화 항을 Non-negative Matrix Factorization (NMF)에 추가하여 모델의 원래 예측과 개념 기반 예측 간의 정렬을 보장합니다. 기존의 자동 개념 발견 방법들은 모델의 실제 의사결정 과정과 추출된 개념 간의 정렬 실패로 인해 설명의 신뢰성이 떨어졌습니다.

- **Technical Details**: FACE는 모델의 예측을 감독하여 개념 학습 중 예측 일관성을 강제합니다. 이는 분류기(supervisor) 감시 아래 개념 기반 활성화와 예측 간의 정렬을 보장하며, 이를 통해 모델이 실제 예측에 사용하는 특징을 잘 포착할 수 있습니다. 이러한 과정을 통해, FACE는 보다 신뢰성 높은 경험적 설명을 도출합니다.

- **Performance Highlights**: ImageNet, COCO, CelebA 데이터셋에서의 체계적인 평가 결과, FACE는 기존 방법들보다 신뢰성(faithfulness)과 희소성(sparsity) 지표 모두에서 우수한 성능을 보였습니다. FACE의 이론적 보장과 실험적 성과는 개념 기반 설명 방법의 신뢰성 제고를 목표로 하고 있습니다.



### InfiniHuman: Infinite 3D Human Creation with Precise Contro (https://arxiv.org/abs/2510.11650)
Comments:
          Accepted to ACM SIGGRAPH Asia 2025. Project website: this https URL

- **What's New**: 이번 논문은 3D 인간 아바타 생성을 위한 새로운 프레임워크인 InfiniHuman을 소개합니다. InfiniHuman은 기존의 foundation models를 증류하여 비용 효율적이고 이론적으로 무한한 규모로 풍부하게 주석이 달린 인간 데이터를 생성합니다. 본 프레임워크를 통해 111K의 다양한 정체성을 가진 데이터셋인 InfiniHumanData가 생성되며, 이는 다중 관점 RGB 이미지와 상세한 의류 이미지를 포함합니다.

- **Technical Details**: InfiniHumanGen은 텍스트, 신체 형태 및 의류 자산을 기반으로 하는 확산 기반(generative) 모델입니다. 이 모델은 사용자가 제공하는 다양한 입력을 통해 신속하고 사실적인 아바타 생성을 가능하게 합니다. InfiniHumanData는 다중 모드 주석을 갖춘 사람의 대표 카탈로그 역할을 하며, 각 정체성은 세밀한 텍스트 설명과 함께 제공됩니다.

- **Performance Highlights**: InfiniHuman 접근 방식은 기존 기술에 비해 시각적 품질, 생성 속도 및 제어 가능성이 향상되었습니다. 특히, Gen-Schnell 및 Gen-HRes 두 가지 모델이 합쳐져, 빠르고 상호작용적인 3D 생성과 고해상도, 사실적인 텍스처를 갖춘 메시 생성이 가능해집니다. 이 연구는 비용 효율적인 솔루션을 통해 고품질 아바타 생성을 민주화하여 다양한 응용 분야에서의 활용을 가능하게 합니다.



### PhySIC: Physically Plausible 3D Human-Scene Interaction and Contact from a Single Imag (https://arxiv.org/abs/2510.11649)
Comments:
          Accepted to ACM SIGGraphAsia 2025. Project website: this https URL

- **What's New**: PhySIC은 단일 RGB 이미지로부터 물리적으로 그럴듯한 인간-장면 상호작용 및 접촉 재구성을 위한 새로운 프레임워크입니다. 이를 통해 수치적으로 일관된 SMPL-X 인간 메쉬, 밀집된 장면 표면, 그리고 정점 수준의 접촉 맵을 복구합니다. 기존 방법들이 depth ambiguity와 occlusion 문제로 어려움을 겪는 반면, PhySIC은 체계적인 접근 방식으로 이러한 문제를 해결합니다.

- **Technical Details**: PhySIC은 초기 파라메트릭 본체 추정치와 조잡한 단안 깊이로 시작하여, 시각적 깊이와 비크기가 설정된 기하학을 융합하여 강력한 메트릭 스캐폴드를 만듭니다. 이 과정에서 occlusion-aware 인페인팅을 수행하고, 발코니나 바닥과 같은 누락된 지원 표면을 합성합니다. 최종적으로 깊이 정합, 접촉 우선 사항, 중첩 방지, 2D 재투영 일관성을 함께 강제하는 신뢰도 가중치 최적화를 통해 신체 자세와 카메라 매개변수, 전역 스케일을 정제합니다.

- **Performance Highlights**: PhySIC은 단일 이미지 기반의 기존 방법보다 뛰어난 성능을 보여주며, 평균 정점 장면 오류를 641 mm에서 227 mm로 줄이고 PA-MPJPE를 절반 줄여 42 mm로 감소시킵니다. 또한, 접촉 F1 스코어를 0.09에서 0.51로 향상시킨 결과를 보였습니다. 다양한 인터넷 이미지를 사용한 실험 결과는 우리의 접근 방식이 다양한 상호작용과 장면 유형에 매우 적합하다는 것을 입증합니다.



### IVEBench: Modern Benchmark Suite for Instruction-Guided Video Editing Assessmen (https://arxiv.org/abs/2510.11647)
Comments:
          Equal contributions from first two authors. Project page: this https URL Code: this https URL Dataset: this https URL

- **What's New**: 이번 논문에서는 Instruction-guided Video Editing(IVE)을 평가하기 위한 새로운 벤치마크 IVEBench를 제안합니다. IVEBench는 600개의 고품질 소스 비디오 데이터베이스를 포함하고 있으며, 7개의 의미적 차원에서 다양한 편집 작업을 처리할 수 있는 능력을 제공합니다. 이 벤치마크는 비디오 품질, 지침 준수 및 비디오 충실도를 포함하는 3차원 평가 프로토콜을 확립하여 사용자 친화적인 비디오 편집 경험을 지원합니다.

- **Technical Details**: IVEBench는 자동 및 수동 필터링을 통해 고품질 비디오 샘플을 수집하고, 총 30개의 주제를 정의하여 비디오의 의미적 범위를 확장했습니다. 각 소스 비디오에 대해서는 Qwen2.5-VL-72B 모델을 사용하여 적절한 길이의 캡션을 생성하고, Doubao-1.5-pro를 통해 편집 카테고리와 편집 프롬프트를 자동으로 선택합니다. 이는 기존 벤치마크의 한계를 극복하는 것입니다.

- **Performance Highlights**: 실험 결과, IVEBench는 스테이트 오브 더 아트의 IVE 방법을 벤치마킹하는 데 효과적인 것으로 나타났으며, 평가 결과가 인간의 인식과 높은 일치를 보이는 것으로 확인되었습니다. 이 연구는 비디오 편집 분야에 대한 귀중한 통찰력을 제공하며, 코드와 데이터셋을 오픈 소스로 공개하여 최신 IVE 방법을 지속적으로 추적할 계획입니다.



### NV3D: Leveraging Spatial Shape Through Normal Vector-based 3D Object Detection (https://arxiv.org/abs/2510.11632)
- **What's New**: 본 논문에서는 NV3D라는 새로운 모델을 제안합니다. 이 모델은 K-nearest neighbors (KNN)와 주성분 분석 (PCA)을 활용하여 계산된 정규 벡터로부터 로컬 특성을 이용합니다. NV3D는 정상 벡터 밀도 기반 샘플링과 시각적 인지(FOV)-기반 샘플링의 두 가지 샘플링 방식을 제공, 기존 데이터의 55%까지 제거하면서도 성능을 유지합니다.

- **Technical Details**: NV3D는 격자 기반으로 인식된 포인트들을 분할하고, 이를 통해 정규 벡터와 관련된 특징을 추출합니다. 기존의 방법들과는 다르게, NV3D는 인접한 포인트들로부터 정규 벡터를 활용하여 표면 방향과 관련된 중요 정보를 수집합니다. 또한, 셀 간 소통을 통해 각측 정규 특성을 쿼리로 사용하고, 복수의 벡터 특성을 키와 값으로 사용하는 요소 기반의 주의 메커니즘을 도입합니다.

- **Performance Highlights**: NV3D 모델은 KITTI 데이터셋에서 훈련되었으며 자동차와 자전거 감지에서 뛰어난 성능을 기록했습니다. 샘플링 없이도 NV3D는 86.60%와 80.18%의 최다 평균 정밀도(mAP)를 달성, 기존 Voxel R-CNN 대비 2.61% 및 4.23% 높은 성과를 보였습니다. 두 가지 샘플링 기법으로, NV3D는 자동차 감지에서 85.54%의 mAP을 달성하여, 약 55%의 포인트들이 필터링되었음에도 불구하고 여전히 기준선을 초과하는 결과를 보였습니다.



### EvoCAD: Evolutionary CAD Code Generation with Vision Language Models (https://arxiv.org/abs/2510.11631)
Comments:
          Accepted to IEEE ICTAI 2025

- **What's New**: 이번 연구에서는 EvoCAD라는 새로운 방법을 제안하여 컴퓨터 지원 디자인(CAD) 객체를 생성합니다. 이 방법은 비전 언어 모델과 진화 최적화를 결합하여 CAD 코드의 기호적 표현을 사용합니다. EvoCAD는 여러 CAD 객체를 샘플링한 후, 진화적 접근을 통해 최적화합니다. 특히, 3D 객체 간의 의미적 유사성을 나타내는 새로운 메트릭도 소개되었습니다.

- **Technical Details**: EvoCAD는 LLM(대형 언어 모델)과 VLM(비전 언어 모델)을 결합하여 CAD 객체 생성을 위한 새로운 방식을 모색합니다. 초기 세트에서 CAD 코드를 생성한 후, 진화적 전략에 따라 평가, 교차, 변이를 통해 최적화 과정을 거칩니다. 이 연구에서 제안된 새로운 메트릭은 객체의 토폴로지 특성에 기반하여, 3D 객체의 비교에서 의미적 유사성을 측정하는 데 초점을 맞춥니다.

- **Performance Highlights**: EvoCAD는 CADPrompt 벤치마크에서 이전 방법들과 비교했을 때 여러 메트릭에서 뛰어난 성능을 보여주었습니다. 특히, 복잡한 구조를 가진 CAD 객체의 경우, 이 모델은 특히 토폴로지적으로 올바른 객체를 생성하는 데 효과적입니다. 이 성과는 기존의 공간 메트릭을 보완하는 새로운 메트릭 덕분에 가능했습니다.



### High-resolution Photo Enhancement in Real-time: A Laplacian Pyramid Network (https://arxiv.org/abs/2510.11613)
Comments:
          accepted by TPAMI 2025

- **What's New**: 이 논문에서는 LLF-LUT++라는 피라미드 네트워크를 소개하여 사진 개선(Photo Enhancement) 성능과 계산 효율성을 동시에 높입니다. 최신 기술을 활용해 고해상도 이미지의 빠른 처리를 지원하면서도 뛰어난 성능을 발휘하고, 특히 이미지 적응형 3D LUT를 통해 결과 품질을 개선합니다. 이러한 접근법은 시즌별 사용이 가능한 올바른 성능과 효율을 목표로 합니다.

- **Technical Details**: LLF-LUT++는 폐쇄형 라플라시안 피라미드 분해 및 재구성을 통해 지역적 및 글로벌 연산자를 통합합니다. 이 네트워크는 시간 복잡도를 줄이기 위해 공간 주파수 변환기 가중치 예측기를 사용하여 최적의 가중치를 추출하고, 고주파 성분에서 에지 디테일을 세밀하게 조정하기 위해 지역 라플라시안 필터를 적용합니다. 이 전체 과정은 4K 해상도의 사진을 단일 GPU에서 단 13ms에 처리할 수 있게 합니다.

- **Performance Highlights**: 실험 결과 LLF-LUT++는 HDR+ 데이터셋에서 PSNR이 2.64 dB 향상되고, 4K 해상도의 이미지를 13 ms라는 빠른 시간에 처리할 수 있어 기존 방법들과 비교했을 때 눈에 띄게 성능이 개선된 것으로 확인되었습니다. 제안된 방식은 두 개의 벤치마크 데이터셋에서의 광범위한 실험을 통해 그 효과성도 입증되었습니다. 이를 통해, 최신 기술과 기존의 지역적 및 글로벌 방식의 통합이 효과적임을 강조합니다.



### ExpVid: A Benchmark for Experiment Video Understanding & Reasoning (https://arxiv.org/abs/2510.11606)
Comments:
          Data & Code: this https URL

- **What's New**: 이번 논문에서는 실험 영상에 대한 MLLMs (Multimodal Large Language Models)의 성능을 체계적으로 평가하기 위한 새로운 벤치마크인 ExpVid를 소개합니다. 기존 벤치마크들이 과학 실험을 수행하는 데 필요한 섬세한 절차와 장기적인 작업을 간과해온 반면, ExpVid는 3단계 작업 계층 구조를 도입하여 실험 비디오를 평가합니다. 이는 과학적 발견을 가속화할 수 있는 큰 가능성을 지닌 AI 도구로 자리잡을 것입니다.

- **Technical Details**: ExpVid는 생물학, 화학 및 의학을 포함한 13개 분야의 웻랩 실험을 중점적으로 다룹니다. 이 벤치마크는 도구, 재료 및 작업에 대한 세부 인식(fine-grained perception), 단계별 이해(procedural understanding), 그리고 과학적 추론(scientific reasoning)을 요구하는 3단계로 구성됩니다. 특히, 학문적 정확성을 보장하기 위해 각 비디오는 동료 검토(peer-reviewed) 논문과 짝지어져 있으며, 비디오와 논문의 연관성을 평가하기 위한 인간 전문가의 검증이 포함됩니다.

- **Performance Highlights**: 19개의 MLLM 모델을 평가한 결과, 모델들은 물체 인식(coarse object recognition) 및 단기 추론(short-horizon reasoning)에서는 강점을 보였지만, 시각적으로 유사한 도구 및 재료를 구별하거나 프로세스에 따른 상태 변화를 추적하는 데에는 어려움을 겪었습니다. 이러한 결과는 신뢰할 수 있는 시각적 기초(visual grounding) 및 구조적 추론(structured reasoning)이 실제 실험실 환경에서 절실히 필요함을 강조합니다. ExpVid는 연구자들이 MLLMs를 더 신뢰할 수 있는 실험 파트너로 개발하기 위한 로드맵을 제공합니다.



### ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training (https://arxiv.org/abs/2510.11605)
Comments:
          ICCV 2025, Project page: this https URL

- **What's New**: 이 논문에서는 Scene Coordinate Regression (SCR) 기법의 한계를 극복하기 위해 새로운 방법인 ACE-G를 제안합니다. 전통적인 SCR 기술은 훈련된 장면에만 일반화하여 조명이나 시점이 다른 쿼리 이미지에 대해 저조한 성능을 보였습니다. ACE-G는 전통적인 SCR의 설계 단점을 극복하기 위해 좌표 회귀기(coordinate regressor)와 장면 특화 맵 코드(scene-specific map code)를 분리하는 방식을 채택합니다.

- **Technical Details**: ACE-G에서는 일반적인 트랜스포머(transformer) 구조를 사용하여 수십만 개의 장면에서 사전 훈련(pre-training)을 수행합니다. 이러한 사전 훈련 과정은 맵 이미지를 미지의 쿼리 이미지에 매핑하는 데 있어 좋은 일반화 능력을 갖출 수 있도록 만듭니다. 기존의 SCR 방식을 대체할 수 있는 이 방법은 훈련된 데이터를 과적합(overfit)하는 문제를 해결하는 데 도움을 줍니다.

- **Performance Highlights**: 여러 도전적인 재위치화(relocalization) 데이터셋에서 ACE-G는 성능을 현저히 향상시키는 것으로 나타났습니다. 이 방법은 높은 정확도를 유지하면서도 계산 자원(computational footprint)을 경제적으로 유지할 수 있음을 입증하였습니다. ACE-G는 다양한 조건의 쿼리 이미지에서의 강건성(robustness)을 크게 증가시키며, 학습 기반의 시각적 재위치화 분야에 기여할 것으로 기대됩니다.



### MS-Mix: Unveiling the Power of Mixup for Multimodal Sentiment Analysis (https://arxiv.org/abs/2510.11579)
Comments:
          Under Review

- **What's New**: 이번 연구에서는 Multimodal Sentiment Analysis (MSA)에서 데이터 부족 문제를 해결하기 위해 MS-Mix라는 새로운 데이터 증강 프레임워크를 제안하고 있습니다. 기존의 mixup 전략의 한계를 극복하는 이 방법은 감정 인식을 위한 세 가지 주요 혁신 요소를 통합하여 고품질 샘플 생성을 보장합니다. 이러한 개선으로 MS-Mix는 기존 방법들보다 일관성 있고 매력적인 감정 분석 결과를 생성합니다.

- **Technical Details**: MS-Mix는 대화형 샘플 선택(Sentiment-Aware Sample Selection, SASS) 전략을 통해 반대 감정을 가진 샘플들의 혼합을 방지하고, 감정 강도 지침(Sentiment Intensity Guided, SIG) 모듈을 사용하여 각 모달리티의 감정 강도에 따라 동적으로 혼합 비율을 결정합니다. 또한 감정 정렬 손실(Sentiment Alignment Loss, SAL)을 도입하여 예측된 감정 분포를 실제 라벨과 정렬함으로써 모델의 일관성을 높입니다. 이와 같은 구성요소들이 결합되어 감정 정보의 효율적 활용을 가능하게 합니다.

- **Performance Highlights**: 세 개의 벤치마크 데이터셋과 여섯 가지 최신 모델 아키텍처에 대한 광범위한 실험 결과, MS-Mix는 기존 방법들보다 일관되게 성능이 우수함을 입증했습니다. 연구 결과에 따르면, MS-Mix는 서로 다른 백본 네트워크 환경에서도 강건한 멀티모달 감정 증강을 제공합니다. 이는 MSA 분야에서 새로운 기준을 설정하는 것으로 의미 있습니다.



### Benchmarking foundation models for hyperspectral image classification: Application to cereal crop type mapping (https://arxiv.org/abs/2510.11576)
Comments:
          Being reviewed for WHISPERS conference ( Workshop on Hyperspectral Image and Signal Processing: Evolution in Remote Sensing )

- **What's New**: 이 연구는 hyperspectral crop mapping을 위한 foundation models의 잠재력을 탐구합니다. HyperSigma, DOFA, 그리고 SpectralEarth 데이터셋으로 사전 훈련된 Vision Transformers를 benchmarking하여 cereal crop mapping의 성능을 평가합니다. 각 모델은 수동으로 라벨링된 데이터를 기반으로 훈련되었고 독립된 테스트 지역에서 평가되었습니다.

- **Technical Details**: 연구는 모로코의 두 개 지역을 대상으로 하여 이루어졌습니다. Hyperspectral imagery는 EnMAP 및 PRISMA 위성의 데이터를 활용하여 수집하였고, NaN 값을 포함하는 밴드는 제거되었습니다. 방대한 데이터셋에 사전 훈련된 Foundation 모델은 pixel-level classification에서 뛰어난 성능을 보이며, HyperSigma, DOFA, SpectralEarth 세 가지 모델이 선택되었습니다.

- **Performance Highlights**: SpectralEarth 모델은 OA 93.5%로 가장 높은 성능을 나타내었으며, DOFA는 OA 62.6%, HyperSigma는 OA 34.5%로 뒤를 이었습니다. SpectralEarth는 정확하고 일관된 분류 성능을 보였고, DOFA는 평균적인 성능을 기록했습니다. HyperSigma는 전반적으로 낮은 성과를 보이며 리얼 월드 데이터에 대한 일반화에서 한계를 드러냈습니다.



### A Framework for Low-Effort Training Data Generation for Urban Semantic Segmentation (https://arxiv.org/abs/2510.11567)
- **What's New**: 이 논문은 도시 장면 인식을 위한 훈련 데이터 생성 방식을 혁신하는 새로운 프레임워크를 소개합니다. 복잡한 3D 모델링 없이 비표시(unlabelled) 이미지와 불완전한 유사 라벨(pseudo-labels)만으로 전이 학습을 가능하게 하여 우리가 사용할 수 있는 낮은 노력의 합성 데이터(source data)를 빠르게 생성합니다. 이 방법은 일반적으로 필요했던 노동 집약적인 합성 데이터 제작 과정을 간소화하여 효율성을 높입니다.

- **Technical Details**: 제안된 방법은 사전 훈련된 diffusion model을 사용하여 소스와 관련 없는 데이터 소스에서도 효과적인 이미지 생성을 가능하게 합니다. 이 프레임워크는 주 타겟 도메인은 물론, 비표시 데이터로부터 전환된 레이블 맵을 사용하여 높은 품질의 결과물을 제공합니다. 이러한 방식은 나쁜 생성물(suboptimal generations)을 필터링하고, 이미지와 레이블 간의 불일치를 수정하며, 데이터 집합 간의 의미론적 일관성을 표준화합니다.

- **Performance Highlights**: 결과적으로, 이 프레임워크는 기존의 최고 수준의 이미지-이미지 전환(image-to-image translation) 기법들보다 최대 8%pt mIoU의 성능 향상을 보여주었습니다. 특히, 간단한 합성 데이터의 변환에서 기존의 고비용 합성 데이터 관리 방식과 동일한 수준의 성능을 달성할 수 있음을 입증했습니다. 이 연구는 도시 장면 이해를 위한 훈련 데이터 생성 과정에서 새로운 가능성을 제시하고 있습니다.



### SNAP: Towards Segmenting Anything in Any Point Cloud (https://arxiv.org/abs/2510.11565)
Comments:
          Project Page, this https URL

- **What's New**: SNAP(Segment Anything in Any Point cloud)는 서로 다른 도메인에서 포인트 기반 및 텍스트 기반 프롬프트를 모두 지원하는 통합 모델입니다. 기존의 3D 포인트 클라우드 분할 방식은 주로 실내 또는 실외와 같은 특정 도메인에 제한되어 있었고, 공간 클릭이나 텍스트 프롬프트 중 하나의 사용자 상호작용 방식만을 지원했습니다. 본 모델은 7개의 데이터 세트를 통해 크로스 도메인 일반화를 달성하며, 부정적 전이를 방지하기 위해 도메인 적응 정규화(domain-adaptive normalization)를 적용합니다.

- **Technical Details**: SNAP는 4개의 주요 부분으로 구성되어 있습니다: 포인트 클라우드 인코딩, 공간 프롬프트 분할, 텍스트 프롬프트 분할, 그리고 훈련입니다. 포인트 클라우드는 XYZ 좌표로 표현되며, Point Transformer V3(PTv3)를 사용하여 포인트 기반 임베딩을 추출합니다. 도메인 일반화를 지원하기 위해 일반적인 배치 정규화를 도메인 정규화로 대체하고, 이는 서로 다른 통계적 특성을 가진 데이터 세트를 그룹화하여 수행됩니다.

- **Performance Highlights**: SNAP는 9개의 제로샷 벤치마크에서 8개에서 최첨단 성능을 달성했으며, 텍스트 프롬프트 분할에 대한 5개의 평가된 벤치마크에서 경쟁력 있는 결과를 보여줍니다. 이 모델은 다양한 실내, 실외 및 공중 포인트 클라우드에서 일관되게 높은 품질의 세그멘테이션 결과를 제공합니다. SNAP는 전문 도메인별 접근 방식을 초월하거나 이를 초과하는 성능을 발휘하여, 확장 가능한 3D 주석을 위한 실용적인 도구가 될 수 있음을 입증했습니다.



### How many samples to label for an application given a foundation model? Chest X-ray classification study (https://arxiv.org/abs/2510.11553)
Comments:
          8 pages, 5 figures

- **What's New**: 이 연구는 흉부 X-레이(classification) 분류에서 필요한 라벨링된 샘플 수를 효율적으로 예측하기 위한 시스템적인 접근 방안을 제시합니다. 기존의 방대한 주석이 달린 데이터에 대한 의존도를 줄이기 위해, power-law fitting을 사용하여 특정 ROC-AUC 기준에 도달하기 위한 훈련 데이터 크기를 예측하는 방식을 도입했습니다. 특히 XrayCLIP과 XraySigLIP 모델이 ResNet-50 대비 훨씬 적은 수의 라벨된 예제로 높은 성능을 기록하였다는 점도 주목할 만합니다.

- **Technical Details**: 연구에서는 MIMIC-CXR라는 대중적인 오픈 데이터셋을 사용하여 21개의 병리(class) 클래스를 분류하기 위한 데이터셋을 구성했습니다. 모델 훈련을 위해 RadDINO-Maira2, XrayCLIP, XraySigLIP 등의 다양한 chest X-ray foundation 모델을 활용했으며, 훈련 과정에서 dropout layer, linear projection 등으로 구성된 분류 헤드를 적용했습니다. 모델의 성능을 평가하기 위해 여러 curve-fitting 접근 방식을 고려했고, 세 파라미터 power-law를 선택하여 AUC 특정 영역을 모델링했습니다.

- **Performance Highlights**: XrayCLIP과 XraySigLIP 모델은 극히 적은 라벨링된 데이터로도 효과적인 성능을 보였으며, 50개의 라벨링된 사례만으로도 최종 성능의 plateau를 정확하게 예측할 수 있는 경향을 보여주었습니다. 이를 통해 실무자들은 목표 성능을 달성하기 위해 필수 샘플만 라벨링함으로써 주석 비용을 최소화할 수 있는 기회를 얻게 됩니다. 또한, 이 연구는 learning curve 예측의 새롭고 효율적인 방법을 제시하여 비슷한 분야의 연구자들에게도 유용할 것입니다.



### ODI-Bench: Can MLLMs Understand Immersive Omnidirectional Environments? (https://arxiv.org/abs/2510.11549)
- **What's New**: 본 논문은 360도 omnidirectional images (ODIs) 이해를 위한 새로운 벤치마크인 ODI-Bench를 소개합니다. ODI-Bench는 2,000개의 고품질 ODIs와 4,000개 이상의 수동으로 주석 처리된 질문-답변(QA) 쌍을 포함하여, 일반 수준과 공간 수준의 ODI 이해를 평가하기 위한 10개의 세부 작업을 제공합니다. 또한, 현재의 대규모 다중 모드 언어 모델(MLLM)들이 ODIs의 몰입 환경을 이해하는 데 어려움을 겪고 있음을 보여주는 실험적 결과를 제시합니다.

- **Technical Details**: ODI 는 기존의 2D 이미지와 달리 360도 전 방향의 시각 정보를 제공하며, ODI-Bench는 이러한 ODIs의 구조적 특성을 반영하여 MLLMs의 일반 및 공간 이해 능력을 포괄적으로 평가합니다. 이 벤치마크는 정밀하게 설계된 자동화된 파이프라인과 전문가의 수동 주석을 통해 생성된 질문-답변 쌍을 통해 높은 신뢰성을 갖춥니다. 또한, ODI-Bench는 close-ended와 open-ended QA 설정에서의 성능 비교를 위해 두 가지 평가 형식을 제공하여 MLLMs의 종합적인 평가를 가능하게 합니다.

- **Performance Highlights**: ODI-Bench의 실험 결과에 따르면, MLLMs는 ODIs가 제공하는 몰입적인 맥락을 이해하는 데 여전히 어려움을 겪고 있습니다. 이러한 문제를 해결하기 위해 Omni-CoT라는 트레이닝이 필요 없는 방법을 제안하여, 시각적 단서와 텍스트 정보 간의 체인 오브 띵킹(reasoning)을 통해 MLLMs의 이해 능력을 개선합니다. Omni-CoT의 도입으로 MLLMs는 일반 및 공간 작업에서 ODIs에 대한 이해 능력을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### Massive Activations are the Key to Local Detail Synthesis in Diffusion Transformers (https://arxiv.org/abs/2510.11538)
- **What's New**: 본 연구에서는 Diffusion Transformers (DiTs) 내에서 발생하는 Massive Activations (MAs)에 대해 체계적으로 조사하였습니다. 이 연구는 MAs가 지역 세부 사항 합성(local detail synthesis)에 중요한 역할을 하며, 전체적인 의미(content)에는 미미한 영향을 미친다는 것을 발견했습니다. 또한 새로운 self-guidance 전략인 Detail Guidance (DG)를 제안하여, 훈련 없이도 MAs를 기반으로 지역 세부 사항의 충실도를 높일 수 있음을 보여줍니다.

- **Technical Details**: Diffusion 모델은 가우시안 노이즈를 점진적으로 저감하면서 데이터를 생성하는 방법입니다. DiT 모델에서는 모든 이미지 토큰에서 고정된 차원에서 Massive Activations가 발생하며, 이러한 활성화는 입력의 timestep embedding에 의해 조절됩니다. DG는 MAs를 방해하여 생성된 'detail-deficient' 모델을 통해 세부 사항 생성을 향상시키는 방식으로 작동합니다.

- **Performance Highlights**: DG는 Classifier-Free Guidance (CFG)와 원활하게 통합되어 세부 사항 정제를 추가적으로 가능하게 합니다. 실험 결과, 다양한 사전 훈련된 DiT 모델에서 DG가 일관되게 고운 세부 사항 품질을 향상시켰음을 보여줍니다. 이러한 개선은 이미지 및 비디오 합성에서의 고도화를 의미하며, 새로운 세부 사항 관리 방식을 제시합니다.



### mmWalk: Towards Multi-modal Multi-view Walking Assistanc (https://arxiv.org/abs/2510.11520)
Comments:
          Accepted by NeurIPS 2025 Datasets and Benchmarks Track. Data and Code: this https URL

- **What's New**: 이번 연구에서는 시각 장애인과 저시력 사용자(BLV)를 위한 안전한 보행 보조를 위한 새로운 멀티뷰 데이터셋인 mmWalk를 개발했습니다. 데이터셋은 120개의 수동 조작 경로와 62k개의 동기화된 프레임으로 구성되어 있으며, RGB, 깊이, 의미론적 모드에서 559k개 이상의 파노라마 이미지를 포함하고 있습니다. mmWalkVQA는 9개 카테고리의 시각적 질문-답변 쌍으로 구성된 벤치마크입니다.

- **Technical Details**: mmWalk는 여러 센서의 멀티모달 정보가 통합되었으며, 보행자, 안내견, 드론 시점에서의 액션을 포함합니다. 77개의 시나리오 카테고리에서 120개의 경로가 수집되었고, 각 경로는 BLV 사용자를 위한 접근성을 높이기 위해 위험한 환경과 특정 랜드마크를 강조합니다. VQA 쌍은 GPT-4o를 사용하여 생성되었으며, 이는 다양한 난이도의 질문을 포함합니다.

- **Performance Highlights**: 최신 비전-언어 모델(VLMs)은 mmWalkVQA에서 위험 평가와 내비게이션 작업에서 상당한 제한을 보였습니다. 연구 결과, 상태가 좋은 모델조차도 BLV 사용자를 위한 안전 및 경로 인지 과제에서 어려움을 겪고 있다는 것을 보여줍니다. mmWalk 데이터를 기반으로 한 모델의 특화된 성능은 실세계에서도 유용성을 증명하였습니다.



### LikePhys: Evaluating Intuitive Physics Understanding in Video Diffusion Models via Likelihood Preferenc (https://arxiv.org/abs/2510.11512)
Comments:
          22 pages, 9 figures

- **What's New**: 이 논문은 비디오 확산 모델(Video Diffusion Models)에서 직관적 물리 이해를 평가하기 위해 'LikePhys'라는 새로운 방법을 제안합니다. 이는 훈련이 필요 없는 방식으로 물리적으로 타당한 비디오와 불가능한 비디오를 구별하여 물리 이해도를 측정합니다. 'Plausibility Preference Error (PPE)'라는 평가 메트릭을 통해, 이전의 방식들보다 인간의 선호와 더 잘 일치하는 결과를 보입니다.

- **Technical Details**: LikePhys는 비디오 쌍을 렌더링하여 물리적으로 현실적인 현상과 통제된 물리 위반을 비교하고, 두 비디오의 시각적 외관이 일치하도록 해 물리 원칙의 위반으로 인한 차이를 평가합니다. 논문에서는 Rigid Body Mechanics, Continuum Mechanics, Fluid Mechanics, Optical Effects 등 4개의 물리 영역을 포함한 12개 시나리오로 구성된 벤치마크를 통해 여러 모델의 성능을 비교합니다.

- **Performance Highlights**: 연구 결과, 현재의 비디오 확산 모델은 복잡한 동역학에서 어려움을 겪고 있지만, 모델의 용량과 추론 설정이 증가함에 따라 물리 이해도가 향상되는 경향이 있음을 보였습니다. 이 논문은 물리 이해도 평가를 위한 새로운 지표를 제공하며, 현재 모델의 한계와 발전 가능성에 대한 통찰을 제안합니다.



### Situat3DChange: Situated 3D Change Understanding Dataset for Multimodal Large Language Mod (https://arxiv.org/abs/2510.11509)
Comments:
          Accepted to NeurIPS 2025 Datasets and Benchmarks Track. Dataset and Code: this https URL

- **What's New**: Situat3DChange 데이터셋이 소개되었습니다. 이 데이터셋은 121K의 QA 쌍, 36K의 변화 설명 및 17K의 재배치 지침을 포함하여 인간-AI 협업을 지원하는 세 가지 변화 이해 작업을 다룹니다. 또한, 11K의 인간 관찰 데이터를 바탕으로 공유된 인식 모델을 구축하여 AI와 인간이 함께 동적 환경을 이해할 수 있도록 도움을 줍니다.

- **Technical Details**: Situat3DChange는 인지적 정렬(cognitive alignment)을 탐구하며, 11K의 변경 사항 주석을 모아놓았습니다. SCReasoner라는 새로운 MLLM 구조를 도입하여 점 구름(point cloud) 비교를 효율적으로 수행하고, 최소한의 파라미터 오버헤드로 잘 알고 있는 점 구름 간의 차이에 주목합니다. 이를 통해 AI 시스템이 인간 나름의 인식을 바탕으로 환경을 해석하고 조작할 수 있는 기초를 마련합니다.

- **Performance Highlights**: 다양한 실험을 통해 MLLM이 Situat3DChange 작업에서의 성과와 한계를 평가하였습니다. 데이터 스케일링 및 도메인 간 전이 실험을 진행하여 보편적인 효과를 입증하며, Situat3DChange 데이터셋은 MLLM 훈련에 매우 유용함을 나타냅니다. 이러한 연구는 지각적으로 정렬된 체화된 에이전트를 개발하는 데 중요한 기여를 하고 있습니다.



### Towards Fast and Scalable Normal Integration using Continuous Components (https://arxiv.org/abs/2510.11508)
Comments:
          Accepted by the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026, first round. 17 pages, 9 figures, 6 tables

- **What's New**: 본 논문은 surface normal integration(표면 법선 통합) 문제를 다루고 있으며, 기존의 반복적인 global optimization(전역 최적화) 접근법의 한계를 극복하기 위해 이를 연속적인 구성 요소의 상대적 규모 추정으로 재구성하였습니다. 이 방법은 같은 구성 요소에 속한 픽셀들이 함께 변하도록 제약하여 최적화 변수의 수를 대폭 줄이고, 최적화 과정의 효율성을 크게 향상시킵니다. 또한, 이 방법은 몇 초 만에 정상 통합 기준에서 최첨단 결과를 달성하였으며, 대규모 해상도에서 픽셀 수준 접근법에 비해 속도를 10배 향상시킵니다.

- **Technical Details**: 저자들은 우선적으로 surface normal map(표면 법선 맵)에서 연속 구성 요소를 식별하는 효과적인 heuristic(휴리스틱)을 제안합니다. 이후, 각 구성 요소의 깊이를 독립적으로 재구성하고, 모든 픽셀의 깊이를 재조정하기 위해 단일 scale parameter(스케일 매개변수)를 최적화합니다. 이 과정에서 기존 방법의 discontinuity model(불연속성 모델)을 재정립하고, 최적화 항목의 균형을 맞추는 outlier reweighting mechanism(이상치 재가중치 메커니즘)을 추가하여 최적화 과정의 수렴을 가속화합니다.

- **Performance Highlights**: 우리의 제안된 방법은 DiLiGenT benchmark(기준)에서 수 초 안에 정상 통합의 최첨단 재구성 정확도를 달성합니다. 중간에서 높은 해상도의 normal maps(정상 맵)에 대해서도 실행 시간을 10배 줄이는 효과를 보여, 실질적인 사용에 매우 적합합니다. 이러한 구성 요소 기반의 접근 방식은 기존의 픽셀 수준 방법과의 호환성도 유지하여 불연속성을 효과적으로 보존할 수 있도록 합니다.



### AndesVL Technical Report: An Efficient Mobile-side Multimodal Large Language Mod (https://arxiv.org/abs/2510.11496)
Comments:
          Tech report of OPPO AndesVL Team

- **What's New**: 최근 클라우드 기반의 MLLMs(다중 모달 대형 언어 모델)인 AndesVL이 소개되었습니다. 이 모델은 Qwen3의 LLM을 기반으로 하여 0.6B에서 4B까지의 파라미터 범위를 갖고 있으며, 모바일 디바이스에서의 실행을 염두에 두고 설계되었습니다. 이 연구는 다양한 벤치마크에서 최초의 성능을 달성하며, 특히 지식 습득, 수학적 추론, 다중 이미지 처리, 다국어 이해 등 일반적인 기능들을 중점적으로 다루고 있습니다.

- **Technical Details**: MLLM의 일반적인 훈련 패러다임은 사전 훈련된 LLM을 활용하여 비주얼 인코더와 정합을 이루고, 지속적인 사전 훈련 및 파인 튜닝을 통해 다중 모달 입력을 처리할 수 있는 모델을 개발하는 것입니다. AndesVL에서는 1+N LoRA 아키텍처를 설계하여 다양한 작업에 적응할 수 있도록 하였으며, 이로 인해 훈련 및 배포 최적화가 이루어집니다. 또한, QAT(Quantization-Aware Training) 및 모델 압축 기법을 통해 모바일 디바이스에서의 높은 실행 효율성을 확보하였습니다.

- **Performance Highlights**: AndesVL은 4B 파라미터 내에서 각종 벤치마크와 비공식 모바일 벤치마크에서 뛰어난 성능을 입증하였습니다. 이 모델은 6.7배의 피크 디코딩 속도 증가, 최대 30.9%의 메모리 감소와 1.8 bits-per-weight의 가중치 압축을 달성했습니다. 이러한 성과는 AndesVL이 모바일 디바이스에서 실용화될 가능성을 높이며, MLLMs의 발전에 중요한 기여를 하고 있음을 나타냅니다.



### VA-GS: Enhancing the Geometric Representation of Gaussian Splatting via View Alignmen (https://arxiv.org/abs/2510.11473)
Comments:
          Accepted by NeurIPS 2025

- **What's New**: 본 논문에서는 3D Gaussian Splatting을 기반으로 한 새로운 방법론을 제안하여, 고품질의 실시간 뷰 합성뿐만 아니라 보다 정확한 표면 복원을 가능하게 합니다. 기존 연구의 한계를 보완하기 위해 이미지 에지를 활용한 손실 함수와 다중 뷰 정렬을 통합하여 지오메트릭 표현을 개선했습니다. 이는 조명 변화에 따른 모호성을 줄이고, 여러 뷰 간의 일관성을 확보하는 데 기여합니다.

- **Technical Details**: 제안된 방법은 edge-aware image cues를 활용하여 표면 경계를 선명히 하고, visibility-aware photometric alignment loss를 도입하여 다중 뷰 간의 기하학적 일관성을 보장합니다. 이 외에도, 조명 변화로 인한 변별력을 줄이기 위해 노멀 기반 제약 조건을 추가하여 Gaussian의 공간 방향을 정제하고, 깊은 이미지 피처를 활용하여 뷰 일관성을 강화합니다.

- **Performance Highlights**: 엄격한 벤치마크 테스트를 통해 제안된 기법이 표면 복원과 새로운 뷰 합성에서 최첨단 성능을 달성했음을 입증했습니다. 기존 방법들과의 비교를 통해 조명에 따른 아티팩트와 경계 명확성 문제를 획기적으로 개선한 것을 보여주었습니다. 이로 인해 복잡한 장면에서도 정확한 표면 복원이 가능하게 되었습니다.



### Coupled Degradation Modeling and Fusion: A VLM-Guided Degradation-Coupled Network for Degradation-Aware Infrared and Visible Image Fusion (https://arxiv.org/abs/2510.11456)
- **What's New**: 본 연구에서는 고유한 Degradation-Coupled Fusion 네트워크(VGDCFusion)를 제안하여 이미지 융합 과정과 이상 징후 모델을 긴밀하게 결합합니다. 이 네트워크는 vision-language models(VLMs)를 통해 이미지 품질을 인식하고 저하된 이미지를 더욱 효과적으로 융합할 수 있도록 합니다. 두 개의 주요 모듈인 Specific-Prompt Degradation-Coupled Extractor(SPDCE)와 Joint-Prompt Degradation-Coupled Fusion(JPDCF)를 통해 성능을 향상시킵니다.

- **Technical Details**: 제안된 VGDCFusion 아키텍처는 VDMC모듈을 통해 modality-specific(모달리티 특정) 저하 인식을 하는 SPDCE와 잔여 저하 필터링 및 상호 모달리티 특성 융합을 수행하는 JPDCF를 포함합니다. SPDCE는 다양한 스케일에서의 기능 추출을 지원하여 이미지의 저하를 효과적으로 억제합니다. JPDCF는 서로 다른 모달리티 간의 저하 인식을 통합하고 남은 왜곡 필터링을 강조합니다.

- **Performance Highlights**: VGDCFusion은 다양한 저하 시나리오에서 기존의 state-of-the-art 융합 접근 방식보다 월등한 성능을 발휘합니다. 이를 통해, 시각적 왜곡이나 세부사항 손실 없이 고품질 이미지를 생성할 수 있습니다. 실험 결과, 저하된 이미지를 직접 처리할 수 있는 역량 향상과 효율성을 통해 응용 가능성을 크게 높였습니다.



### Enhancing Maritime Domain Awareness on Inland Waterways: A YOLO-Based Fusion of Satellite and AIS for Vessel Characterization (https://arxiv.org/abs/2510.11449)
- **What's New**: 이 논문은 내수(水路) 해양영역 인식(MDA)을 개선하기 위한 새로운 프레임워크를 제안합니다. 이 프레임워크는 고해상도 위성 이미지와 자동식별시스템(AIS)에서 얻은 선박 궤적 데이터를 융합하여 협력 시스템의 취약점을 극복하고자 합니다. 비협력 위성 이미지를 활용하고 AIS 데이터와의 융합 접근법을 구현하여 어두운 선박(dark vessels)을 식별하고 협력 트래픽을 검증하는 기능을 가지고 있습니다.

- **Technical Details**: 본 연구에서는 You Only Look Once (YOLO) v11 객체 탐지 모델을 사용하여 선박과 바지선을 선박 유형, 바지의 덮개 상태, 운영 상태, 바지 수, 이동 방향에 따라 감지하고 특성화합니다. 이미지는 4,550개의 인스턴스로 주석이 달린 데이터셋을 바탕으로 하며, 미시시피 강의 5,973 평방마일 지역에서 수집되었습니다. 평가 결과는 여러 클래스(예: 예인선, 크레인 바지선, 벌크 선박 등)에 대한 F1 점수가 95.8%에 달하는 등 높은 정확성을 보였습니다.

- **Performance Highlights**: 선박 수 추정의 평균 절대 오차(MAE)는 2.4척에 불과하며, 특정 조건에서의 정확도는 98%에 이릅니다. 이 연구의 결과는 비협력 위성 감지와 AIS 융합의 가능성을 강조하며, 실시간 해양 감시, 이상 탐지, 및 고품질 데이터 생성을 지원합니다. 향후 작업에서는 주석이 달린 데이터셋의 확장, 시간 추적 기능의 도입, 그리고 다중 모드 딥러닝(multi-modal deep learning) 탐구를 통해 운영의 확장성을 더욱 높일 계획입니다.



### Robust Ego-Exo Correspondence with Long-Term Memory (https://arxiv.org/abs/2510.11417)
Comments:
          Accepted by NeurIPS 2025

- **What's New**: 이번 연구에서는 egocentric(자기중심) 및 exocentric(타인중심) 뷰 간의 객체 수준의 대응을 확립하기 위해 새로운 EEC(ego-exo correspondence) 프레임워크를 제안합니다. 기존의 Segment Anything Model 2 (SAM 2)를 기반으로 한 이 방법은 장기 메모리(long-term memory)를 활용하여 객체 분할의 강력한 성능을 보입니다. 새로운 접근 방식으로 이중 메모리 아키텍처와 Mixture-of-Experts (MoE)에 영감을 받은 적응형 특징 라우팅 모듈을 도입하였습니다.

- **Technical Details**: 제안하는 LM-EEC 모델은 두 가지 주요 구성 요소를 포함합니다: (i) 메모리-뷰 MoE 모듈로, 채널 및 공간 차원에서 각 전문가(feature)의 기여 가중치를 동적으로 할당합니다. (ii) 이중 메모리 뱅크 시스템은 필수적인 장기 정보를 효율적으로 유지하는 압축 전략을 활용하며, 에고 메모리와 엑소 메모리로 나뉘어 있습니다. 이러한 혁신적인 구조는 객체 세분화 성능을 크게 향상시킵니다.

- **Performance Highlights**: EgoExo4D 벤치마크에 대한 광범위한 실험에서 LM-EEC는 기존 모델 및 SAM 2 기초 모델보다 탁월한 성능을 보이며, 새로운 최첨단 결과를 달성하였습니다. 이 모델은 다양한 시나리오에서 강력한 일반화 능력을 보여주었으며, 복잡한 환경에서도 높은 정확도를 유지합니다.



### DocReward: A Document Reward Model for Structuring and Stylizing (https://arxiv.org/abs/2510.11391)
- **What's New**: 최근 에이전트 워크플로우(agentic workflows)의 발전으로 전문 문서 생성 자동화가 가능해졌습니다. 그러나 기존 연구는 텍스트 품질에 초점을 맞추어 시각적 구조와 스타일을 간과하고 있습니다. 이를 해결하기 위해 DocReward라는 문서 보상 모델을 제안하며, 이는 문서의 구조 및 스타일 기반으로 평가합니다.

- **Technical Details**: DocReward 모델은 32개 도메인과 267개 문서 유형을 포함한 117K 쌍의 문서를 커버하는 다중 도메인 데이터셋 DocPair를 사용하여 훈련됩니다. 이 모델은 텍스트 품질에 무관하게 문서의 구조와 스타일을 평가할 수 있는 포괄성과 텍스트 품질 무관성(textual-quality-agnosticism)을 감지할 수 있도록 설계되었습니다. 또한, Bradley-Terry 손실을 사용하여 훈련되어 문서 순위의 일관성을 유지합니다.

- **Performance Highlights**: DocReward는 고등 교육을 받은 인간 평가자에 의해 평가된 테스트 데이터셋에서 GPT-4o 및 GPT-5를 각각 30.6, 19.4 포인트 초과하여 성능을 입증했습니다. 문서 생성의 외적 평가에서도 DocReward는 60.8%의 높은 승률을 기록, 이는 GPT-5의 37.7%와 비교하여 상대적으로 월등한 성능을 보여줍니다.



### MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inferenc (https://arxiv.org/abs/2510.11387)
Comments:
          Accepted by NeurIPS 2025. Project Page: this https URL

- **What's New**: 이번 연구에서는 Gaussian Splatting을 통해 반사 모델링에 있어서 다중 보기 일관성(material consistency)을 도입하는 새로운 접근 방식을 제안합니다. 이는 물리 기반 환경 모델링과 결합하여 보다 정확한 반사를 학습할 수 있게 합니다. 또한, 간접 조명을 처리하기 위한 새로운 환경 모델링 전략을 도입하여 물리 기반의 간접 조명 렌더링을 가능하게 합니다.

- **Technical Details**: 제안된 방법에서는 2D Gaussians가 다중 보기 일관적인 물질 맵을 생성하도록 하여 이들의 물리적 속성에 기반한 설계를 진행합니다. 또한, 카메라 경로에 따라 물체 표면의 photometric variations를 추적하여 반사 점수로 정량화합니다. 이를 통해 개별 보기 반사 점수를 통합하는 spatial reflection fusion 모듈을 적용하여 일관된 반사 강도 우선순위를 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 기존 최첨단 방법보다 조명 추정 및 지오메트리 복원에서 우수한 성능을 입증하며, 새로운 보기 합성에서 state-of-the-art의 렌더링 품질을 달성했습니다. 여러 가지 벤치마크에서 진행된 평가 역시 제안된 방법의 뛰어난 성능을 확인시켜줍니다.



### Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessmen (https://arxiv.org/abs/2510.11369)
- **What's New**: 이번 논문은 강화 학습(Reinforcement Learning, RL) 기반의 이미지 품질 평가(IQA) 모델들이 뛰어난 일반화 능력을 갖추고 있지만, 이러한 특성을 유발하는 메커니즘과 주요 요소들이 충분히 연구되지 않았음을 제기합니다. 연구진은 RL 훈련을 통해 MLLM들이 중복된 시각적 표현을 간결하고 교차 도메인에서 정렬된 텍스트 표현으로 변환하는 과정을 강조하며, 이 변환이 IQA 모델의 일반화의 근본 원천임을 확인했습니다. 또한, 이 논문은 Reasoning 과정 없이 직접 이미지를 일치시키는 새로운 알고리즘인 RALI를 제안합니다.

- **Technical Details**: RALI 알고리즘은 대조 학습(Contrastive Learning)을 활용하여 이미지와 RL에 의해 학습된 일반화 가능한 텍스트 표현을 직접적으로 정렬합니다. 이를 통해 복잡한 Reasoning 프로세스를 의존할 필요가 없으며, LLM을 로드할 필요조차 없게 됩니다. 연구에서는 RALI를 통해 품질 점수 매기기(task)에서 강화 학습 기반 모델과 유사한 일반화 성능을 달성할 수 있다고 보고합니다.

- **Performance Highlights**: RALI 프레임워크는 기존의 Reasoning 기반 모델들에 비해 모델 파라미터와 추론 시간(inference time)을 5% 미만으로 유지하면서도 유사한 수준의 일반화 성능을 제공합니다. 이는 다양한 애플리케이션에서의 배포를 더욱 용이하게 하여, 특정 시나리오에서의 활용 가능성을 높이는 데 기여할 수 있습니다.



### Uncertainty-Aware ControlNet: Bridging Domain Gaps with Synthetic Image Generation (https://arxiv.org/abs/2510.11346)
Comments:
          Accepted for presentation at ICCV Workshops 2025, "The 4th Workshop on What is Next in Multimodal Foundation Models?" (MMFM)

- **What's New**: 본 논문에서는 데이터 부족 문제를 해결하기 위해 UnIACorN이라는 불확실성 인지 제어망(Uncertainty-Aware ControlNet)을 소개합니다. 이 모델은 레이블이 없는 데이터 소스를 활용하여 주어진 세분화(Segmentation) 작업에 적합한 주석 데이터셋을 생성할 수 있습니다. 또한, 기존의 ControlNet 구조에 불확실성을 제어하는 메커니즘을 추가하여, 라벨이 없는 데이터로부터 새로운 이미지 분포를 생성 가능하게 합니다.

- **Technical Details**: UnIACorN은 두 가지 독립적인 조건화 전략을 결합하여 작동합니다. 첫 번째는 주어진 레이블 분포에 따라 학습된 세멘틱 조건화(Semantic Conditioning)이며, 두 번째는 레이블이 없는 이미지 도메인에서 훈련된 불확실성 조건화(Uncertainty Conditioning)입니다. 이를 통해 레이블이 없는 이미지가 불확실성의 매핑을 통해 유용하게 사용되고, 이 과정에서 다양한 레이블 데이터셋을 생성할 수 있습니다.

- **Performance Highlights**: 실험을 통해 UnIACorN은 레이블이 있는 Spectralis OCT와 레이블이 없는 HOME-OCT 간의 도메인 간 차이를 보완하여 세분화 성능을 크게 향상시켰습니다. 이러한 방법은 기존의 스타일 전송(style transfer) 기법에 비해 높은 불확실성을 가진 레이블 데이터 생성을 가능하게 하며, 다양한 도메인 전환을 지원합니다. 교차 도메인 테스트에서 UnIACorN의 효과는 특히 트래픽 장면 실험을 통해 입증되었습니다.



### MMAP: A Multi-Magnification and Prototype-Aware Architecture for Predicting Spatial Gene Expression (https://arxiv.org/abs/2510.11344)
Comments:
          Accepted for presentation at the 2025 Pacific Rim International Conference on Artificial Intelligence (PRICAI 2025)

- **What's New**: 이 논문에서는 새로운 프레임워크인 MMAP (Multi-MAgnification and Prototype-enhanced architecture)를 제안합니다. MMAP는 세포의 분자 정보와 공간 정보를 모두 보존할 수 있는 Spatial Transcriptomics (ST) 기술에서 기존 방법들이 겪고 있는 두 가지 주요 문제, 즉 국소적인 피처 추출의 불충분한 세밀도와 글로벌 공간 정보의 부족을 동시에 해결합니다. 이는 고해상도의 조직 이미지를 보다 정확하게 해석하고 유전자 발현을 예측할 수 있는 새로운 접근법을 제공합니다.

- **Technical Details**: MMAP는 다중 배율(multi-magnification) 패치 표현을 통해 세밀한 조직 구조를 캡처하며, 대표적인 프로토타입 임베딩(prototype embeddings)을 학습하여 슬라이드 수준의 정보를 압축된 형태로 표현합니다. 이러한 구조는 깊은 신경망 방식으로 작동하며, 첫 단계에서는 다중 배율 뷰를 활용해 스팟 수준의 특성을 생성하고, 두 번째 단계에서는 프로토타입 은행을 통해 슬라이드 전체의 정보를 요약합니다. MMAP는 크로스-어텐션 메커니즘을 사용하여 이러한 로컬 및 글로벌 정보를 효과적으로 결합합니다.

- **Performance Highlights**: 광범위한 실험 결과에 따르면, MMAP는 Mean Absolute Error (MAE), Mean Squared Error (MSE), Pearson Correlation Coefficient (PCC) 등의 여러 평가 지표에서 기존의 최첨단 방법들을 지속적으로 초월하는 성능을 보여주었습니다. 또한, 이 모델은 비용 효율성과 확장 가능성을 유지하면서 고해상도 이미지 분석에 적합하게 개발되었습니다. 따라서 MMAP는 유전자 발현 예측의 정확성을 크게 향상시키는 잠재력을 지니고 있습니다.



### InternSVG: Towards Unified SVG Tasks with Multimodal Large Language Models (https://arxiv.org/abs/2510.11341)
- **What's New**: 이 논문은 멀티모달 대형 언어 모델(MLLMs)의 강력한 전이(transfer) 및 일반화(generalization) 능력을 활용하여 SVG 이해, 편집, 생성에 대한 통합 모델링(unified modeling)을 달성하는 InternSVG 패밀리를 제안합니다. SAgoge 라는 대규모 데이터셋은 정적 그래픽(static graphics)과 동적 애니메이션(dynamic animations)을 포함한 SVG 작업에 대한 포괄적인 데이터 기준을 제공합니다. 또한 SArena라는 동반 벤치마크를 통해 명확한 작업 정의와 표준화된 평가를 제공합니다.

- **Technical Details**: 이 논문에서 제안된 InternSVG 모델은 SVG 특화 토큰(specific tokens), 서브워드 기반(embedding initialization) 임베딩을 사용하며, 짧은 정적 SVG에서 긴 시퀀스 일러스트레이션 및 복잡한 애니메이션으로 진행되는 2단계 훈련 전략을 적용합니다. SAgoge는 아이콘, 긴 시퀀스 일러스트레이션, 과학적 다이어그램, 동적 애니메이션 등 다양한 난이도의 작업을 지원하며, 기존 데이터셋에 비해 더 깊은 계층과 풍부한 속성을 제공합니다. 이러한 자원에 기반하여, SNS는 긍정적 전이를 유도하고 전체 성과를 향상시킵니다.

- **Performance Highlights**: InternSVG는 SArena 및 기존 벤치마크 실험을 통해 상당한 성과 개선을 확인했습니다. 이 모델은 기존의 오픈 소스 및 독점 시스템에 비해 consistently overperforms합니다. 이러한 성과는 효과적인 전이 및 일반화가 이루어졌음을 부각시키며, SVG 관련 다양한 작업에서 뛰어난 성능을 보여줍니다.



### REACT3D: Recovering Articulations for Interactive Physical 3D Scenes (https://arxiv.org/abs/2510.11340)
Comments:
          8 pages

- **What's New**: REACT3D는 정적 3D 장면을 상호작용 가능한 시뮬레이션 복제물로 변환하는 새로운 제로샷(free-shot) 프레임워크입니다. 본 논문은 정적 장면으로부터 이동 가능한 물체를 추출하고 관절 타입과 모션 파라미터를 추론하는 다양한 기술을 포함합니다. 이 시스템은 표준 3D 장면 포맷을 물리적으로 가능한 시뮬레이션 형식으로 전환하여 로봇 시스템이 상호작용하도록 지원합니다.

- **Technical Details**: REACT3D는 포인트 클라우드나 메시와 같은 정적 3D 장면 표현을 입력으로 받아들여 배경을 보존한 채 상호작용이 가능한 디지털 트윈을 구성합니다. 이 과정에서 이미지 기반 관절 추정기와 분할 기반 모델을 활용하여 개체의 관절 구조를 회복하고 시뮬레이션이 가능한 3D 장면을 생성합니다. 또한, 이 시스템은 장면 통합을 통해 ROS, Isaac Sim, PyBullet 등 다양한 플랫폼과 호환성을 제공합니다.

- **Performance Highlights**: REACT3D는 다양한 실내 장면에서 탐지 및 분할, 관절 메트릭 기준으로 최첨단 성능을 달성하여 프레임워크의 효과를 입증했습니다. 또한, 생성된 3D 자산은 다양한 렌더러 및 물리 시뮬레이터에서 유용하게 활용될 수 있습니다. 이 시스템은 대규모 상호작용 장면 생성의 장벽을 낮추고, 고품질의 물리적 3D 자산을 제공하는 효과적인 솔루션을 제시합니다.



### Evaluating the effects of preprocessing, method selection, and hyperparameter tuning on SAR-based flood mapping and water depth estimation (https://arxiv.org/abs/2510.11305)
- **What's New**: 이 연구는 Synthetic Aperture Radar (SAR) 이미지를 이용해 홍수 매핑(flood mapping)과 수심 추정(water depth estimation) 방법의 사전 처리(preprocessing), 적용 방법 및 하이퍼파라미터의 선택이 결과에 미치는 영향을 평가합니다. 연구에서는 2019년과 2021년에 프랑스 가롱 강(Garonne River)에서 발생한 두 개의 홍수 사건에 대해 수치 해석(hydrodynamic simulations) 및 현장 관측 데이터를 참조 데이터로 사용하였습니다. 결과에 따르면, 스펙클 필터 선택이 홍수 범위의 추정에 큰 영향을 미치며, 올바른 방법론 선택이 필수적임을 강조합니다.

- **Technical Details**: 연구에서는 SAR 이미지를 통해 홍수 매핑 및 수심 필드를 생성하기 위한 워크플로우를 제안하고, 이것의 각 처리 단계(사전 처리, 홍수 매핑, 수심 추정)의 영향력을 조사했습니다. SAR 이미지의 사전 처리를 위해 메디안 필터, 리 필터, 리 시그마 필터, 프로스트 필터 및 최신 딥러닝 기반의 SAR2SAR 방법을 포함한 다섯 가지 방법이 평가되었으며, 이러한 방법은 상이한 하이퍼파라미터 설정으로 실행되었습니다. 모든 단계에서 결과의 변동성을 분석하여 최적화를 위한 강력한 설계를 식별하는 것이 연구의 주요 목표입니다.

- **Performance Highlights**: 연구의 결과, 조정된 비지도 학습 방법(예: 지역 임계값 설정(local thresholding)이나 변화 감지(change detection))이 감독 학습 방법보다 성능 면에서 우수하였음을 보여주었습니다. 또한, 사전 처리 및 홍수 매핑 단계에서 발생하는 복합적인 불확실성(compounded uncertainty)은 수심 필드 추정의 높은 변동성을 초래합니다. 이러한 분석 결과들은 운영적 사용을 위한 다양한 조합 소스 중에서 가장 신뢰할 수 있는 구성을 식별하는 데 기여하였습니다.



### sketch2symm: Symmetry-aware sketch-to-shape generation via semantic bridging (https://arxiv.org/abs/2510.11303)
- **What's New**: 이 논문은 스케치 기반 3D 재구성을 위한 새로운 방법인 Sketch2Symm을 제안합니다. 이 방법은 두 단계로 구성되어 있으며, 첫 번째 단계에서는 스케치를 이미지로 변환하여 부족한 정보를 보강합니다. 두 번째 단계에서는 구조적 정규성을 향상시키기 위해 대칭 제약을 포함하여 3D 형태를 재구성합니다.

- **Technical Details**: Sketch2Symm의 첫 번째 단계는 VGG-19 모델을 사용하여 입력 스케치와 참조 이미지를 처리하고, 스케치의 기하학적 구조를 기반으로 소스 이미지를 변형하여 스케치에 맞는 이미지를 생성합니다. 두 번째 단계에서는 RGB2Point 방법을 채택하여 이미지에서 3D 점 구름을 생성하며, 이때 대칭 제약 조건을 추가하여 구조적 신뢰성을 높입니다.

- **Performance Highlights**: 공식 스케치 데이터셋에 대한 실험 결과, Chamfer Distance, Earth Mover's Distance, F-Score에서 기존 스케치 기반 재구성 방법보다 우수한 성능을 보였습니다. 이 결과는 제안한 세 가지 기법이 3D 재구성의 정확성과 일반화 능력에 긍정적인 영향을 준다는 것을 보여줍니다.



### When Does Supervised Training Pay Off? The Hidden Economics of Object Detection in the Era of Vision-Language Models (https://arxiv.org/abs/2510.11302)
Comments:
          23 pages, 4 figures, 4 tables

- **What's New**: 이번 논문에서는 전통적인 수퍼바이즈드(Object Detection) 기법인 YOLO와 제로샷(Zero-Shot) 비전을 제공하는 VLM인 Gemini Flash 2.5 간의 비용 효과성을 비교합니다. 1,000개의 COCO 이미지와 200개의 다양한 제품 이미지를 기반으로 한 체계적인 평가를 통해 두 방법론의 경제적 메트릭을 수치적으로 도출하고, 선택의 기준이 되는 적정 성능을 정의합니다. 이러한 비용 효과 분석은 구체적인 아키텍처 선택을 위한 정량적 기준을 제시합니다.

- **Technical Details**: YOLO는 수퍼바이즈드 학습에 의존하여 고전적인 감지 정확도를 제공하지만, 이 과정에서 높은 주석 비용이 발생합니다. 반면, VLM은 자연어 쿼리를 통해 제로샷 감지를 수행하며, 수작업 주석 없이도 작동합니다. 본 논문은 기존 연구에서 다루어지지 않았던 기술 성능과 경제적 요소를 중심으로 비용 모델을 수립하고, 여러 조건에서 두 접근 방식의 효용을 비교하는 체계적인 방법론을 제시합니다.

- **Performance Highlights**: 수퍼바이즈드 YOLO는 표준 카테고리에서 91.2%의 정확도를 기록했고, Gemini는 68.5%의 성능을 보였습니다. 그러나 Gemini는 제품 카테고리에서 52.3%의 정확도를 발휘하여 일부 희귀 클래스에 대한 탐지도 가능하게 합니다. 효율성 측면에서, Gemini는 100,000회 추론에서 감지당 비용이 $0.00050로 YOLO의 $0.143에 비해 상당히 낮습니다.



### $Δ\mathrm{Energy}$: Optimizing Energy Change During Vision-Language Alignment Improves both OOD Detection and OOD Generalization (https://arxiv.org/abs/2510.11296)
Comments:
          Accepted by NeruIPS2025

- **What's New**: 이 논문은 비전-언어 모델(vision-language models, VLMs)의 OOD(out-of-distribution) 데이터에 대한 일반화 능력을 개선하는 방법을 제시합니다. 새로운 OOD 점수인 ΔEnergy를 도입하여 기존 에너지 기반 OOD 점수보다 우수한 성능을 보여줍니다. 또한, ΔEnergy는 변동(covariate) 변화 아래 OOD 일반화도 동시 개선할 수 있으며, 통합 파인튜닝(framework) 프레임워크를 통해 VLM의 강건성을 향상시킬 수 있음을 강조합니다.

- **Technical Details**: ΔEnergy는 비전-언어 정렬을 수정할 때 발생하는 에너지 변화를 정량화하여 닫힌 세트(closed-set) 클래스와 열린 세트(open-set) OOD 클래스를 구별하는 접근법입니다. 이 방법은 몇 가지 파라미터를 사용하는 EBM(energy-based method)와 결합하여 정보를 최대한 활용합니다. 이를 통해, EBM 방법이 OOD 탐지와 일반화를 모두 개선하는 것이 이론적으로 입증됩니다.

- **Performance Highlights**: 광범위한 OOD 탐지 및 일반화 벤치마크에서 이루어진 실험 결과, ΔEnergy는 최근 기법보다 10%에서 25%까지 AUROC(Area Under Receiver Operating Characteristic)에서 뛰어난 성능을 보였습니다. 따라서 본 논문에서 제안한 방법은 다양한 배포(distribution) 유형에서 강력한 OOD 탐지 성능을 발휘하며, 실용적인 응용에서 활용할 수 있을 것으로 기대됩니다.



### Human Uncertainty-Aware Data Selection and Automatic Labeling in Visual Question Answering (https://arxiv.org/abs/2510.11295)
- **What's New**: 본 연구에서는 Visual Question Answering (VQA)에서 Large Vision-Language Models (VLMs)의 성능을 개선하기 위해 human uncertainty (HU)를 모델링하는 새로운 방법론을 제시합니다. 저자들은 HaDola라는 새로운 프레임워크를 도입하여, 한정된 HU 레이블을 사용해 모델의 정확도와 신뢰성을 높이는 방법을 제안하고 있습니다. 이 접근 방식은 기존의 방법들이 간과했던 HU의 중요성을 강조하며, 데이터 선택 및 자동 레이블링 과정을 통합하고 있습니다.

- **Technical Details**: HaDola는 네 가지 단계로 구성된 파이프라인을 가지고 있습니다: 해별(discriminate), 자기 주석(self-annotate), 오류 트리거(error trigger), 훈련(training)입니다. 이 프레임워크는 작은 HU 주석 시드 세트에서 출발하여, 반복적으로 데이터에 대한 감독을 확장하고 유해한 샘플을 식별하는 과정을 거칩니다. 또한, VQA-Accuracy와 같은 기존 평가 지표는 HU를 반영하지 못한다는 문제를 지적하며, HU에 기반한 새로운 평가 지표인 HU-acc를 제안합니다.

- **Performance Highlights**: HaDola는 VQAv2와 VizWiz 데이터셋에서 광범위한 실험을 통해 기존의 state-of-the-art (SOTA) 모델들을 능가하는 성능을 보였습니다. 특히, 적은 양의 HU 주석 데이터로도 높은 정확도와 신뢰할 수 있는 모델을 달성하였으며, 모델의 조정(calibration) 또한 우수한 결과를 나타냈습니다. 이는 HU를 효과적으로 활용하여 학습 전략을 개선할 수 있음을 보여줍니다.



### EEMS: Edge-Prompt Enhanced Medical Image Segmentation Based on Learnable Gating Mechanism (https://arxiv.org/abs/2510.11287)
Comments:
          Accepted by BIBM 2025

- **What's New**: 본 논문에서는 EEMS라는 새로운 의학 이미지 세분화(model segmentation) 모델을 소개합니다. 이 모델은 Edge-Aware Enhancement Unit (EAEU)와 Multi-scale Prompt Generation Unit (MSPGU)를 결합하여 성능을 크게 향상시킵니다. EAEU는 다중 주파수(feature extraction) 특성을 활용하여 경계 인식을 개선하고, MSPGU는 프롬프트 기반 접근법을 통해 정확한 타겟 위치를 보장합니다. 이를 통해 EEMS는 복잡한 배경에서도 향상된 세분화 정확성을 얻게 됩니다.

- **Technical Details**: EEMS는 고전적인 인코더-디코더 구조를 채택하여 의학 이미지의 다중 스케일 및 다중 주파수 정보를 효과적으로 캡처합니다. EAEU 모듈은 경계 정보를 강화하고, MSPGU는 각 단계에서 프롬프트 정보를 생성합니다. 핵심 구성 요소인 Dual-Source Adaptive Gated Fusion Unit (DAGFU)는 EAEU와 MSPGU의 출력을 효율적으로 융합하여 세분화 마스크를 생성합니다. DAGFU는 가변 게이팅 메커니즘(learnable gating mechanism)을 사용하여 특징 간의 적절한 정보 조절을 가능하게 합니다.

- **Performance Highlights**: ISIC2018과 같은 데이터셋에 대한 테스트 결과, EEMS는 이전 모델들보다 뛰어난 성능과 신뢰성을 보여줍니다. 이 모델은 특히 복잡한 이미지 및 노이즈에 강한 Robustness를 자랑하며, 임상 도구로서의 가능성을 높이고 있습니다. EEMS는 정밀한 경계 세분화와 다양한 병변 형태에 대한 처리 능력을 갖추고 있어 의학적 진단 및 치료 계획 병행에 유용한 도구로 자리잡을 것으로 기대됩니다.



### Exploring and Leveraging Class Vectors for Classifier Editing (https://arxiv.org/abs/2510.11268)
Comments:
          Accepted in NeurIPS 2025

- **What's New**: 이 논문에서는 Class Vectors라는 새로운 개념을 소개하여, 이미지 분류기(image classifiers)의 편집을 보다 효율적으로 할 수 있는 방법을 제안합니다. 기존 접근법은 오류 수정이나 재교육에 많은 비용을 소모했으나, Class Vectors는 이러한 문제를 해결하며 클래스 특유의 표현(adaptation)을 캡처합니다. 이러한 Class Vectors는 분류기의 의사결정 경계를 업데이트하거나 잠재 공간(latent space)에서 피쳐를 조절하여 안전하고 유연한 편집을 가능하게 합니다.

- **Technical Details**: Class Vectors는 각 클래스의 표현을 잠재 공간에서 분리하여 파악합니다. 이 접근법은 클래스 간의 semantic shift를 효과적으로 포착하며, 간단한 클래스 산술(class arithmetic)을 통해 고급 개념 편집을 지원합니다. 또한, 제안된 방법은 저변에서 선형성과 직교성을 활용하여 다양한 작업에서 편집의 효과성을 높입니다. 하지만, 클래스 표현이 잠재 공간에서 잘 구조화되어 있다고 가정하는 몇 가지 제한 사항이 존재합니다.

- **Performance Highlights**: 제안된 Class Vectors의 성능은 여러 응용 분야에서 입증되었으며, unlearning, 환경 적응(environmental adaptation), 적대적 방어(adversarial defense) 및 적대적 트리거 최적화(adversarial trigger optimization)와 같은 활용 가능성을 보여주고 있습니다. 실험 결과는 이런 접근 방식이 기존 방법보다 더 유연하고 효율적이라는 것을 나타냅니다. 그러나, 클래스 경계가 모호한 다중 레이블 분류(multi-label classification) 작업으로의 확장은 여전히 도전 과제가 되고 있습니다.



### A Large-Language-Model Assisted Automated Scale Bar Detection and Extraction Framework for Scanning Electron Microscopic Images (https://arxiv.org/abs/2510.11260)
Comments:
          14 pages, 6 figures

- **What's New**: 본 연구에서는 다중 모드(multi-modal) 자동 스케일 바(scale bar) 감지 및 추출 프레임워크를 제안합니다. 이 프레임워크는 객체 감지(object detection), 텍스트 감지(text detection), 텍스트 인식(text recognition)을 동시에 수행하며, 대규모 언어 모델(LLM) 에이전트를 활용하여 정확한 분석을 제공합니다. 기존의 수작업 방식에 비해 처리 효율성과 정확성을 크게 향상시켜 MICroscopy(SEM) 이미지에서 스케일 바 자동 추출을 가능하게 합니다.

- **Technical Details**: 제안된 프레임워크는 자동 데이터 세트 생성(Auto-DG), 스케일 바 객체 감지, 하이브리드 광학 문자 인식(hybrid OCR) 시스템을 통한 정보 추출, LLM 에이전트에 의한 결과 분석 및 검증의 네 가지 단계로 구성됩니다. 특히, Auto-DG 모듈은 다양한 SEM 이미지로 구성된 데이터 세트를 생성하여 모델의 강력한 훈련과 일반화 효능을 보장합니다. 하이브리드 OCR 시스템은 DenseNet 및 Convolutional Recurrent Neural Network(CRNN) 알고리즘을 사용하여 설계되었습니다.

- **Performance Highlights**: 제안된 모델은 100%의 정밀도와 95.8%의 재현율을 기록하며, 평균 정밀도(mean Average Precision, mAP) 99.2%를 달성하여 뛰어난 성능을 입증했습니다. 하이브리드 OCR 시스템은 89%의 정밀도, 65%의 재현율, 75%의 F1 점수를 기록하여 기존의 단독 엔진들보다 우수한 성능을 보였습니다. 이와 같은 높은 성과는 과학적 이미지 분석을 위한 신뢰성 있는 도구로 자리매김하도록 합니다.



### DTEA: Dynamic Topology Weaving and Instability-Driven Entropic Attenuation for Medical Image Segmentation (https://arxiv.org/abs/2510.11259)
Comments:
          Accepted by BIBM 2025

- **What's New**: 본 연구는 의료 영상 분할에서 기존 방법들이 가지는 구조적 표현의 한계와 불충분한 맥락 모델링 문제를 해결하기 위해 DTEA 모델을 제안합니다. 이 모델은 Semantic Topology Reconfiguration (STR) 및 Entropic Perturbation Gating (EPG) 모듈이 통합된 새로운 스킵 연결 구조를 특징으로 합니다. STR은 다중 스케일 시맨틱 피처를 동적 하이퍼 그래프 형태로 재구성하여 해부학적 의존성을 모형화하고, EPG는 채널의 불안정성을 평가하여 임상적으로 중요한 영역을 강조합니다.

- **Technical Details**: DTEA 모델은 변환기(Transformer) 아키텍처를 사용하여 인코더와 디코더 모두에서 장기적이며 지역적인 시맨틱 피처를 추출합니다. 모델의 구조는 U자 형태로, 네 개의 단계인 특징 전처리, Semantic Topology Reconfiguration (STR), Entropic Perturbation Gating (EPG), 특징 후처리를 포함합니다. STR은 고차의 해부학적 의존성을 명시적으로 모델링하고, EPG는 비선형 혼돈 맵과 엔트로피 기반 채널 선택을 통합하여 정보를 최적화합니다.

- **Performance Highlights**: 본 연구의 모듈은 세 가지 벤치마크 데이터셋에서 기존 방법들보다 우수한 분할 정확도를 달성했으며, 다양한 임상 설정에서도 더 나은 일반화 능력을 보여주었습니다. DTEA는 CNN과 변환기를 포함한 여러 백본 아키텍처와 호환되며, 병변 영역의 시각적 분리를 크게 향상시켜 줍니다. 광범위한 실험 결과는 제안된 모듈들이 의료 영상 분할 작업에서 강력한 강건성, 효율성 및 일반화 능력을 보임을 입증합니다.



### Nepali Sign Language Characters Recognition: Dataset Development and Deep Learning Approaches (https://arxiv.org/abs/2510.11243)
Comments:
          6 pages, 9 figures

- **What's New**: 이번 연구는 네팔 수화(Nepali Sign Language, NSL)를 위한 첫 번째 벤치마크 데이터셋을 소개합니다. 이 데이터셋은 구조적 및 시각적 특성을 포착하기 위해 설계된 36개의 제스처 클래스와 클래스당 1,500개의 샘플로 구성되어 있습니다. 이를 통해 저자들은 저명하지 않은 수화에 대한 연구를 촉진하고자 합니다.

- **Technical Details**: 모바일넷 V2(MobileNetV2)와 레즈넷50(ResNet50) 아키텍처를 사용하여 NSL 데이터셋에서 인식 성능을 평가했습니다. 결과적으로 모바일넷 V2는 90.45%, 레즈넷50은 88.78%의 분류 정확도를 달성했습니다. 이 연구는 합성곱 신경망(convolutional neural networks)이 자원이 부족한 환경에서도 수화 인식 작업에 효과적임을 보여줍니다.

- **Performance Highlights**: 연구 결과는 특별히 저자원 환경에서 수화 인식 작업에 대한 심층 학습(deep learning) 접근 방식을 체계적으로 평가한 첫 번째 시도로, 이전 연구에서 다루어지지 않았던 수화 연구에 대한 가능성을 알리고 있습니다. 특히 전이 학습(transfer learning)과 미세 조정(fine-tuning)의 잠재력이 강조되고 있습니다.



### LightPneumoNet: Lightweight Pneumonia Classifier (https://arxiv.org/abs/2510.11232)
Comments:
          13 pages (including references), 5 figures

- **What's New**: 본 연구에서는 LightPneumoNet이라는 경량의 합성곱 신경망(CNN) 모델을 제안합니다. 이 모델은 흉부 X-레이에서 폐렴을 정확하게 진단할 수 있도록 설계되었으며, 5,856개의 공개 데이터셋을 기반으로 학습되었습니다. 기존의 복잡한 아키텍처와 달리, 이 모델은 388,082개의 학습 가능한 매개변수로 구성되어 있어 메모리 사용량이 1.48MB로 최소화되었습니다.

- **Technical Details**: LightPneumoNet은 224x224로 이미지 크기를 조정하고, 그레이스케일 변환 및 픽셀 정규화를 포함한 전처리 과정을 거쳤습니다. 데이터 증강(rotation, zoom, shear) 기법을 적용하여 과적합(overfitting)을 방지하였습니다. 이 모델은 4개의 합성곱 층 블록으로 이루어져 있으며, 복잡한 전이 학습(Transfer Learning) 전략에 의존하지 않고 효율적인 구조를 갖추고 있습니다.

- **Performance Highlights**: 모델은 독립적인 테스트 세트에서 94.2%의 전체 정확도, 92%의 정밀도 및 96%의 F1 스코어를 달성하여 뛰어난 성능을 보였습니다. 특히, 민감도(Recall)는 99%로 폐렴 사례를 효과적으로 식별하며 임상적으로 знач여지는 잘못된 음성율을 최소화하였습니다. 이 모델은 저비용 하드웨어에서도 배치할 수 있어 의료 격차가 있는 지역에서도 접근 가능한 진단 도구로써 병원에 기여할 수 있을 것으로 기대됩니다.



### Investigating Identity Signals in Conversational Facial Dynamics via Disentangled Expression Features (https://arxiv.org/abs/2510.11223)
- **What's New**: 이 연구는 개인의 얼굴 표정의 동적 요소만으로 신원을 식별할 수 있는지를 조사합니다. FLAME 3D morphable model을 활용하여 얼굴 형태와 표정 동역학을 명확히 분리하고, 대화 비디오에서 프레임별 매개변수를 추출합니다. CANDOR 데이터셋을 통해 얼굴 동적 요소가 강력한 신원 신호를 포함한다는 것을 입증하며, 동적 식별의 안정성을 저해하는 불안정한 형태 추정의 영향을 분석합니다.

- **Technical Details**: 연구는 CANDOR 대화 코퍼스를 기반으로 하여, 1,429명의 화자로부터 수집한 데이터를 사용했습니다. FLAME 모델을 통해 얼굴 매개변수를 3D로 분리하고, VGGHeads를 통해 이를 프레임 단위로 회귀시킵니다. Conformer 모델을 이용해 동적 데이터를 처리하며, 통계적 패턴 분석과 강화 학습 방법을 통해 분류 성능을 개선합니다.

- **Performance Highlights**: 이 방식은 1,429-way 분류에서 정확도 61.14%를 기록하여 우연의 결과보다 458배 높은 성과를 나타냅니다. 연구진은 드리프트-잡음비(DNR)를 도입하여 형태와 표정의 분리 신뢰성을 수치적으로 평가하며, DNR이 인식 성능과 강한 부정 상관관계를 가진다고 보고합니다. 이를 통해 대화에서의 개인 특유의 정체성을 드러내는 신호를 확인할 수 있습니다.



### Class Prototypes based Contrastive Learning for Classifying Multi-Label and Fine-Grained Educational Videos (https://arxiv.org/abs/2510.11204)
Comments:
          Published at CVPR 2023

- **What's New**: 최근 아동의 온라인 미디어 소비 증가에 따라 어린 학습자를 위한 교육 콘텐츠 필터링 도구가 필요해졌습니다. 이 논문에서는 온라인 비디오에서 교육 콘텐츠를 감지하는 새로운 접근 방식을 제안합니다. 주요 초점은 문해력(literacy)과 수학(math)이라는 두 가지 교육 콘텐츠 클래스에 있으며, 각 클래스에 대해 Common Core Standards를 기반으로 한 주요 코드를 선택하였습니다.

- **Technical Details**: 논문에서는 비디오에 여러 종류의 교육 콘텐츠가 포함될 수 있으므로, 이를 다중 레이블(multi-label) 정밀 분류(fine-grained classification) 문제로 정의합니다. 클래스 프로토타입(class prototypes) 기반의 지도 대조 학습(supervised contrastive learning) 접근 방식을 제안하여 각 콘텐츠 형식에 대한 프로토타입을 학습하고, 멀티모달(transformer network) 네트워크를 활용하여 비디오 내에서 시각적(visual) 및 청각적(audio) 신호 간의 상호작용을 캡처합니다.

- **Performance Highlights**: 제안된 방법은 교육 연구자들에 의해 세밀하게 레이블이 부여된 193시간 분량의 비디오로 구성된 APPROVE 데이터셋에서 강력한 기준선(baselines)을 초과 성능을 보였습니다. 또한, Youtube-8M 및 COIN과 같은 다른 벤치마크에서도 우수한 결과를 달성하였습니다. 이번 연구는 교육 비디오의 정밀한 분류를 위한 데이터 기반 접근 방식을 발전시키기 위한 기초를 제공합니다.



### FlexAC: Towards Flexible Control of Associative Reasoning in Multimodal Large Language Models (https://arxiv.org/abs/2510.11190)
Comments:
          19 pages, 11 figures. Accepted by the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)

- **What's New**: 이번 논문에서는 멀티모달 대형 언어 모델(MLLMs)의 연결적 사고(associative reasoning) 조절력을 향상시키기 위한 새로운 접근법인 Flexible Association Control (FlexAC)을 제안합니다. 기존 방법들은 사실 기반 및 창의적 시나리오에서의 적응력을 저해하는 연결적 사고 조절의 유연성이 부족했습니다. FlexAC는 연결적 행동을 조절하기 위한 경량화된 비학습 프레임워크로, 환각(hallucination) 정보를 활용하여 연결적 방향을 설정하고, 이를 포괄적으로 제어할 수 있는 방법을 소개합니다.

- **Technical Details**: FlexAC는 두 가지 주요 단계로 구성됩니다: 먼저 Offline Control Vector Construction에서 환각된 응답을 통해 연결적 벡터를 추출하고, 이렇게 생성된 벡터는Inference-Time Control에서 모델의 행동을 유도하는 데 사용됩니다. 이 과정에서, 중간 레이어의 표현이 연결적 행동을 암호화하고 있으며, 이에 대한 수동적 및 능동적 조절이 가능함을 보여주었습니다. 누적된 연결적 벡터와 태스크 특화 연결적 벡터를 결합함으로써 다양한 연상 방향에 따른 응답을 생성할 수 있습니다.

- **Performance Highlights**: FlexAC는 특히 Creation-MMBench에서 최대 5.8배의 창의성을 향상시키고, CHAIR에서 환각 비율을 29% 감소시키는 성과를 거두었습니다. 이러한 결과는 정신적 연상 작용의 유연한 조절이 가능함을 보여주며, 기존의 기준을 초월하는 성능 개선을 입증하였습니다. 다양한 실험을 통해 FlexAC의 연결적 행동 제어가 효과적임을 확인하였습니다.



### Saudi Sign Language Translation Using T5 (https://arxiv.org/abs/2510.11183)
Comments:
          11 pages, supplementary, SPECOM 2025

- **What's New**: 이 논문은 새로운 데이터셋을 활용하여 사우디 수화(Saudi Sign Language, SSL) 번역을 위한 T5 모델의 적용 가능성을 탐구합니다. SSL 데이터셋은 다양한 시나리오에서 종합적인 평가를 가능하게 하는 세 가지 도전적인 시험 프로토콜을 포함하고 있습니다. 또한, 얼굴 가리기와 같은 SSL의 독특한 특성이 수화 인식 및 번역에 도전 요인으로 작용합니다.

- **Technical Details**: 우리의 실험에서는 YouTubeASL 데이터셋에서 사전 훈련된 T5 모델을 SSL 데이터셋에서 직접 훈련한 모델과 비교하여 미국 수화(American Sign Language, ASL) 데이터의 사전 훈련이 모델 성능에 미치는 영향을 조사하였습니다. 실험 결과, YouTubeASL에서 사전 훈련이 모델 성능을 약 $3	imes$ 개선함을 보여주며, 이는 수화 모델의 언어 간 전이 가능성을 나타냅니다. 특히, 우리는 포즈 기반 접근 방식을 사용하여 수화의 외관을 생략하면서 교차 언어 사전 훈련이 성능을 크게 향상시킬 수 있음을 보여주었습니다.

- **Performance Highlights**: 본 연구의 결과는 대규모 ASL 데이터를 활용하여 SSL 번역을 향상시키는 이점을 강조하며, 보다 효과적인 수화 번역 시스템 개발에 대한 통찰력을 제공합니다. 특히 수화 번역(SLT)에서 T5 모델의 효과성을 입증하고, 사우디 수화와 같은 저자원 언어에 대해 여러 모델을 비교함으로써 자원 부족 문제를 해결하려고 합니다. 결과적으로, 다양한 채널의 데이터 수집과 사전 훈련이 SSL의 번역 성능 향상에 중요한 역할을 하는 것을 확인했습니다.



### BLEnD-Vis: Benchmarking Multimodal Cultural Understanding in Vision Language Models (https://arxiv.org/abs/2510.11178)
Comments:
          Code and Dataset to be released

- **What's New**: 본 논문에서는 BLEnD-Vis라는 새로운 멀티모달, 다문화 벤치마크를 소개하여 비전-언어 모델(VLM)에서 문화적 지식의 강건성을 평가합니다. 이 벤치마크는 16개 지역을 포함한 313개의 문화적으로 구체화된 질문 템플릿을 구성하고, 세 가지 형식의 다중 선택 질문을 생성합니다. BLEnD-Vis는 기존 평가 방법의 한계를 넘어서, 문화적 이해의 강건성을 세밀하게 분석할 수 있는 기회를 제공합니다.

- **Technical Details**: BLEnD-Vis는 BLEnD 데이터셋을 기반으로 하여 원본 다중 선택 질문(MCQ), 재구성된 MCQ, VQA 스타일의 MCQ의 세 가지 평가 형식을 개발합니다. 이 평가 시스템은 모델 성능에 대한 언어적 재구성 및 모드 변환의 영향을 통제된 방식으로 비교할 수 있게 합니다. 4,916개의 이미지와 21,782개의 MCQ 인스턴스를 포함하는 BLEnD-Vis 벤치마크는 인간의 주석을 통해 검증된 데이터로 구성됩니다.

- **Performance Highlights**: BLEnD-Vis에서 13개의 VLM을 평가한 결과, 모델의 성능은 매개변수 수와 강한 상관관계가 없으며, 언어적 재구성이 성능을 저하시킬 수 있음을 확인했습니다. 또한, 시각적 단서가 VQA 성능을 개선하는 데 기여하지만, 동일한 사실에 대한 텍스트와 이미지 간의 강건한 일관성은 여전히 도전 과제가 됩니다. BLEnD-Vis는 문화적 지식의 민감함과 모델 성능 간의 간극을 드러내, 향후 다문화적으로 더 적합한 VLM의 개발에 기여할 수 있는 비판적인 시험 공간을 제공합니다.



### G2L:From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Knowledge Distillation (https://arxiv.org/abs/2510.11176)
- **What's New**: 이번 연구에서는 기존의 giga-scale 모델이 가지는 높은 연산 비용 문제를 해결하기 위한 새로운 접근법인 G2L 프레임워크를 제안합니다. 이 프레임워크는 giga-scale 모델의 15%에 불과한 파라미터만으로도 유사한 성능을 내는 large-scale 모델을 개발할 수 있도록 도와줍니다. 이를 통해 특정 암 타입에 특화된 모델을 훈련시키는 데 필요한 데이터 양을 대폭 줄일 수 있습니다.

- **Technical Details**: G2L 프레임워크는 지식 증류(knowledge distillation) 기법을 활용합니다. 이는 1K의 병리 슬라이드를 사용하여 특정 암에 대한 large-scale 모델을 최적화하는 방식입니다. 실험에는 H-optimus-0이라는 giga-scale 모델을 선생(train teacher)으로, Hibou-L이라는 large-scale 모델을 학생(student)으로 설정하여 서로의 특징을 효과적으로 전이합니다.

- **Performance Highlights**: G2L 디스틸 모델은 동일한 크기의 기존 large-scale 모델들을 초월하여 성능을 개선하였고, 일부 벤치마크에서는 giga-scale 모델을 초과하는 결과를 보여주었습니다. 특히, 이미지 변동성에 대한 저항성을 나타내는 더 높은 강건성 지수를 기록함으로써, 실제 환경에서도 신뢰성을 확보할 수 있음을 입증하였습니다.



### Reliable Cross-modal Alignment via Prototype Iterative Construction (https://arxiv.org/abs/2510.11175)
- **What's New**: 본 논문에서는 스타일 정보를 억제하면서 의미 정보의 정렬을 최적화하기 위한 새로운 방법론인 PICO를 제안합니다. 기존의 방법들이 정보 편향이나 손실을 초래하는 스타일 정보의 영향력을 무시하는 반면, PICO는 의미 정보를 정렬하는 데 집중합니다. 가장 중요한 것은, PICO가 각 피처 열의 의미 정보의 확률을 정량화하고 이를 임베딩 상호작용 중 가중치로 활용하여 스타일 정보의 간섭을 효과적으로 억제한다는 점입니다.

- **Technical Details**: PICO 방법론은 프로토타입 반복 구조를 통해 스타일 프로토타입을 구축하며, 각 피처 열이 제공하는 의미 정보의 확률을 계산합니다. 이는 주요 작업인 퍼포먼스 피드백 기반 가중치 함수에 의해 수행됩니다. 통계 분석을 통해 상호작용 결과의 신호 배분을 분석하여 각 피처 열에 대한 유사 의미 확률을 도출합니다.

- **Performance Highlights**: PICO는 다양한 기준 모델과 벤치마크에서 실험을 수행하며, 기존의 최신 기술들보다 5.2%에서 14.1% 더 우수한 성능을 기록했습니다. 이는 PICO의 새로운 접근 방식이 기존의 품질 저하 문제를 해결하며 더 나은 성능을 낼 수 있음을 시사합니다.



### CoPRS: Learning Positional Prior from Chain-of-Thought for Reasoning Segmentation (https://arxiv.org/abs/2510.11173)
Comments:
          18 pages, 6 figures, 6 tables

- **What's New**: 본 논문에서는 CoPRS라는 새로운 다중 모달 체인-오브-생각(Multi-modal Chain-of-Thought, MCoT) 기반의 포지셔널 인식 모델을 소개하고 있습니다. 이는 언어 추론과 세분화(segmentation)를 연결하는 해석 가능하고 미분 가능한 포지셔널 프라이어를 제시하여 기존의 한계를 극복하고자 합니다. CoPRS는 이미지-명령어 입력에 대해 reasoning 과정을 수행한 다음, 타겟 지역을 집중하는 감지 heatmap을 생성하여 세분화 마스크 디코딩을 향상시킵니다.

- **Technical Details**: CoPRS는 이미지와 텍스트 입력을 사용하여 구조화된 정책 모델을 통해 reasoning과 시각적 인식을 통합합니다. 학습 가능한 집중 토큰(concentration token)이 이미지와 명령어 맥락(context)을 집계하여 집중 쿼리를 생성하고, 이 쿼리는 타겟을 집중시키기 위한 heatmap으로 변환되어 세분화 마스크 예측에 활용됩니다. 또한, Group Relative Policy Optimization (GRPO) 전략을 도입하여 훈련을 통합하고, 시각적 비전 경로와 언어 경로를 결합하여 효과적으로 세분화할 수 있도록 합니다.

- **Performance Highlights**: CoPRS는 RefCOCO 시리즈와 ReasonSeg 데이터셋에서 최첨단 성능을 기록하고 있으며, 각 분할에서 최고의 cIoU/gIoU를 달성하고 있습니다. 연구 결과에 따르면 heatmap의 품질이 최종 마스크의 정확도에 강력한 영향을 미쳐 reasoning 출력과 세분화 성능 간의 일관된 연관성을 지지합니다. 전반적으로, CoPRS는 reasoning을 통한 집중과 정교한 마스크 생성을 촉진하는 데 강력한 성과를 보이고 있습니다.



### Multiview Manifold Evidential Fusion for PolSAR Image Classification (https://arxiv.org/abs/2510.11171)
Comments:
          The paper has 14 pages and 7 figures

- **What's New**: 본 논문에서는 다중 뷰 기하학적 구조를 활용하여 데이터를 더 신뢰성 있게 통합하기 위한 새로운 프레임워크인 다중 뷰 다면 증거 융합 네트워크(MMEFnet)를 제안합니다. 이는 PolSAR 데이터에서 Covariance Matrix와 다양한 특성들을 효과적으로 융합하여 예측 정확도를 높이는 동시에 불확실성을 고려합니다. 또한, Dempster-Shafer 이론을 활용한 융합 전략을 통해 예측 결과의 해석성과 신뢰성을 개선합니다.

- **Technical Details**: MMEFnet는 Covariance Matrix와 multi-features를 서로 다른 다면 구조로 모델링하며, 각각 Hermitian Positive Definite(HPD) manifold와 Grassmann manifold에서 처리됩니다. 두 개의 고유한 kernel metric learning 네트워크를 통해 각 뷰의 다면 표현을 학습하고, DS 결합 규칙을 사용해 이러한 뷰로부터 나오는 신뢰도가 반영된 증거를 융합합니다. 이 과정에서 학습된 특징들은 Dirichlet 분포로 변환되어 신뢰도와 불확실성을 수치적으로 평가합니다.

- **Performance Highlights**: 제안된 MMEFnet 방법은 세 가지 실제 PolSAR 데이터셋에 대한 광범위한 실험을 통해 기존 방법들보다 지속적으로 더 우수한 정확도, 강인성, 해석가능성을 입증하였습니다. 특히 복잡한 다중 차원 데이터에서 각 클래스에 대한 지원 정도를 나타내는 증거 벡터로서 출력을 해석하여, 더욱 신뢰할 수 있는 의사결정을 가능하게 합니다. 이러한 결과는 heterogeneous한 PolSAR 장면에서 분류 성능을 향상시키는 동시에 불확실성을 효과적으로 관리할 수 있음을 보여줍니다.



### Validation of an Artificial Intelligence Tool for the Detection of Sperm DNA Fragmentation Using the TUNEL In Situ Hybridization Assay (https://arxiv.org/abs/2510.11142)
- **What's New**: 이 연구는 인공 지능(AI) 도구를 이용해 정자 DNA 파편화(SDF)를 효과적으로 감지하는 방법을 제시합니다. 기존의 정액 분석법과는 달리, 이 새로운 방법은 화학적 평가 없이 단계 대조 현미경(phase contrast microscopy) 이미지만을 기반으로 SDF를 예측할 수 있습니다. TUNEL(Terminal deoxynucleotidyl transferase dUTP nick end labeling) 검사를 금표 기준으로 삼아 비파괴적으로 정자를 분류할 수 있는 방법론이 소개되었습니다.

- **Technical Details**: 제안된 모형은 이미지 처리 기술과 최근의 transformer 기반 기계 학습 모델인 GC-ViT를 결합하여 정자의 DNA 파편화를 예측합니다. 설계된 앙상블 모델은 순수 transformer 비전 모델 및 형태학적 요소만 사용하는 모델과 비교 평가되었습니다. 이러한 접근은 정자의 DNA 무결성을 평가하는 새로운 방법을 제공하며, 이는 생식 의학의 진전을 의미합니다.

- **Performance Highlights**: 제안된 프레임워크는 60%의 민감도(sensitivity)와 75%의 특이도(spec specificity)를 달성할 수 있는 가능성을 보여주었습니다. 이는 임상 진단과 치료 적용에 있어 DNA 무결성에 기반한 실시간 정자 선택이 가능하다는 점에서 의의가 큽니다. 관찰된 성능은 정자의 생존 가능성을 유지하면서도 고해상도 이미지를 통한 분석이 가능함을 시사합니다.



### video-SALMONN S: Streaming Audio-Visual LLMs Beyond Length Limits via Memory (https://arxiv.org/abs/2510.11129)
- **What's New**: 비디오-SALMONN S는 3시간 분량의 비디오를 1 FPS(Frames Per Second) 및 360p 해상도로 처리할 수 있는 최초의 스트리밍 오디오-비주얼 LLM(large language model)입니다. 이 모델은 고정된 메모리 예산 내에서 작동하며, 정보 손실을 최소화하기 위해 새로운 TTT(테스트 타임 트레이닝) 메모리 모듈과 프롬프트 의존 메모리 리더를 도입합니다. 이를 통해 여러 시간 분량의 비디오를 효율적으로 이해할 수 있습니다.

- **Technical Details**: 모델은 (i) TTT 메모리 모듈을 통해 잔여 시퀀스 정보를 보존하며, (ii) 프롬프트 의존 메모리 독서를 통해 제약된 메모리에서 중요한 맥락 관련 콘텐츠를 선택적으로 검색합니다. TTT는 Hessian-free conjugate-gradient 방식으로 최적화되어, 메모리 효율성을 대폭 향상시킵니다. 이 모델은 분류 성능을 높이기 위해 조회 손실을 최소화하는 데 집중합니다.

- **Performance Highlights**: video-SALMONN S는 Video-MME, LVBench, VideoEvalPro와 같은 장기 비디오 벤치마크에서 높은 품질의 이해도를 유지하며, 전체 정확도 74.2% 및 Video-MME의 장기 파트에서 67.8%의 성능을 기록합니다. 8B 매개변수를 가진 이 모델은 오프라인 및 스트리밍 기준선보다 뛰어난 성능을 발휘하며, 최신 모델 중에서도 가장 우수한 성능을 자랑합니다.



### Demystifying Numerosity in Diffusion Models -- Limitations and Remedies (https://arxiv.org/abs/2510.11117)
- **What's New**: 이 논문에서는 최신 모델들이 숫자 세기(numerosity) 지침을 제대로 따르지 못하는 문제를 다룬다. 특히, diffusion models가 데이터셋을 확장하고 모델 크기를 늘리는 것만으로는 기대하는 성능 향상을 이룰 수 없다는 점을 보여준다. 저자들은 GrayCount250 및 NaturalCount6이라는 두 개의 합성 데이터셋을 만들어 성능을 평가했으며, 노이즈 초기화가 숫자 세기에 영향을 미친다는 것을 발견하였다.

- **Technical Details**: 연구진은 diffusion models의 효과적인 성능 향상을 위해 count-aware layout 정보 주입을 제안하였다. GrayCount250 데이터셋에서의 정확도를 20.0%에서 85.3%로, NaturalCount6에서는 74.8%에서 86.3%로 개선할 수 있었다. 본 연구를 통해 노이즈 프라이어(noise prior)가 객체 수에 대한 빈도를 제한하는 경향이 있음을 알 수 있었다.

- **Performance Highlights**: Diffusion models의 성능에 대한 정확도 검증 결과, GrayCount250과 NaturalCount6 모두에서 유의미한 개선을 이룩하였다. 2B에서 12B로의 모델 확장에도 불구하고 성능 저하가 관찰되었으며, 노이즈 초기화의 역할이 중요함을 보여주었다. 이러한 결과는 공간적 인식 지침을 따르는 데 있어 diffusion models의 근본적인 한계를 재조명할 수 있는 기회를 제공한다.



### Connecting Giants: Synergistic Knowledge Transfer of Large Multimodal Models for Few-Shot Learning (https://arxiv.org/abs/2510.11115)
Comments:
          Accepted by IJCAI 2025

- **What's New**: 이 논문에서는 Synergistic Knowledge Transfer (SynTrans)라는 새로운 프레임워크를 제안하여 대규모 멀티모달 모델로부터 다양한 지식을 효과적으로 전이하여 기존의 Few-Shot Learning (FSL) 시스템을 강화하는 방법을 탐구합니다. SynTrans는 CLIP 모델을 강력한 스승으로 사용하고, 몇 개의 샘플로도 분류작업을 수행할 수 있는 비전 인코더를 학생으로 활용하여 비지도 프록시 작업을 통해 시맨틱 정합 시각 지식을 증류합니다. 이는 기존의 부정확한 방법론의 한계를 줄이고, 고품질의 시맨틱 지식을 추출하여 FSL 성능을 향상시키는 데 기여합니다.

- **Technical Details**: SynTrans는 세 단계를 통해 작동하며, 첫째로 CLIP 모델을 사용하여 비지도 방식으로 시각 지식을 추출합니다. 다음으로, Synergistic Knowledge Mining(SynMine) 모듈이 대규모 언어 모델을 활용해 심도 있는 텍스트 설명을 생성하며, 이는 비주얼-언어 모델에 의해 정제되어 클래스 특성에 대한 풍부한 의미 이해를 도출합니다. 마지막으로, 비주얼-시맨틱 브리징(Visual-Semantic Bridging) 모듈은 비주얼과 시맨틱 공간 간의 양방향 지식 전이를 촉진하여 클래스별 분류자 가중치를 생성합니다.

- **Performance Highlights**: 실험 결과, SynTrans는 단순한 몇 개의 샘플 기반 비전 인코더와 결합하여도 네 개의 FSL 데이터 세트에서 기존 최첨단 방법들을 뛰어넘는 성능을 보여줍니다. 이는 잘 정의된 시맨틱 지식을 사용하여 인간처럼 직관적으로 새로운 클래스를 분류하는 능력을 기계 학습에 통합하고, FSL 분야의 새로운 방향을 제시합니다. SynTrans는 대규모 멀티모달 모델에서 지식을 체계적으로 통합하는 최초의 프레임워크로, FSL의 성능 향상을 위한 기초를 마련합니다.



### Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignmen (https://arxiv.org/abs/2510.11112)
Comments:
          NeurIPS 2025 Spotlight

- **What's New**: 이 논문에서는 Longitudinal multimodal data를 효과적으로 모델링하기 위해 $	exttt{DiPro}$라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 크게 Static과 Dynamic feature를 분리하고, EHR 데이터와의 다중 시간 척도 정렬을 통해 두 가지 주요 문제를 해결합니다. 즉, 중복된 CXR 시퀀스와 EHR 데이터 간의 시간 불일치를 극복하는 방식으로, 분석의 질을 강화합니다.

- **Technical Details**: $	exttt{DiPro}$는 세 가지 모듈로 구성됩니다: 첫째, Spatiotemporal Disentanglement 모듈은 연속 CXR에서 정적 해부학적 정보와 동적 병리학적 변화를 분리합니다. 둘째, Progression-Aware Enhancement 모듈은 CXR 쌍의 순서를 반대로 하여 각 유형의 feature의 일관성을 유지하면서 동적 feature의 학습을 보장합니다. 마지막으로, Multimodal Fusion via Multiscale Alignment 모듈은 지역 CXR 쌍과 EHR 데이터의 동기화를 통해 전체적인 데이터의 일관성을 유지합니다.

- **Performance Highlights**: MIMIC 데이터셋에 대한 광범위한 실험 결과, $	exttt{DiPro}$는 질병 진행 예측과 일반 ICU 예측 작업에서 최신 기술인 SOTA(state-of-the-art) 성능을 달성했습니다. 정량적 평가에서도 이 모델이 기존의 임상 지식과 잘 정렬된다는 것을 보여줍니다. 이러한 결과는 $	exttt{DiPro}$의 효과적인 시간적 임상 역학 추출 능력을 입증합니다.



### MoMaps: Semantics-Aware Scene Motion Generation with Motion Maps (https://arxiv.org/abs/2510.11107)
Comments:
          Accepted at ICCV 2025, project page: this https URL

- **What's New**: 이 논문은 실제 비디오에서 의미 있고 기능적으로 중요한 3D 모션 프라이어(motion priors)를 배우는 과제를 다루고 있습니다. 우리는 MoMap(모션 맵)이라는 새로운 픽셀 정렬 표현을 제안하여 3D 장면 모션을 예측할 수 있도록 합니다. 이 방법은 기존 생성 이미지 모델에서 생성된 MoMap를 사용하여 향후 3D 장면 모션을 효과적으로 예측할 수 있게 합니다.

- **Technical Details**: 우리는 50,000개 이상의 실제 비디오에서 MoMap 데이터베이스를 생성하고, 이 표현들에 대해 확산 모델(diffusion model)을 훈련시켰습니다. MoMap는 정지된 카메라에서 고정된 시간 간격 동안의 장면을 캡처한 모션 스냅샷입니다. MoMap를 통해 카메라 모션과 객체 모션을 분리하고, 이미지 생성 모델과 결합하여 모션 예측을 수행할 수 있습니다.

- **Performance Highlights**: 실험 결과, 우리의 접근 방법이 그럴싸하고 의미적으로 일관된 3D 장면 모션을 생성하는 데 효과적임을 보여줍니다. 또한, MoMap를 통해 새로운 2D 비디오 프레임 합성 파이프라인을 제시하여 3D 모션의 일관성을 유지하는 비디오 생성의 새로운 방향을 제시합니다. 이러한 방법은 다양한 작업에 유용하게 활용될 수 있으며, 3D 모션 생성의 발전을 위한 향후 가능성을 열어줍니다.



### Compositional Zero-Shot Learning: A Survey (https://arxiv.org/abs/2510.11106)
Comments:
          Survey paper with 36 pages, 8 plots and 4 figures

- **What's New**: 이 논문은 Compositional Zero-Shot Learning (CZSL)에 대한 첫 번째 종합 조사 결과를 제시하고, 최신 CZSL 방법들을 체계적으로 검토합니다. 이 연구에서는 데이터 불균형 문제를 다루기 위해 다양한 방법론을 도입하고, 이를 명시적인 비대칭성(disentanglement) 기반으로 분류합니다. 또한, 각 접근 방식의 장단점을 비교 분석하며, 미래 연구 방향을 제시합니다. 이 조사 연구는 CZSL 분야의 기초 자료로서 향후 발전에 기여할 것으로 기대됩니다.

- **Technical Details**: CZSL은 'seen' 클래스에서 'unseen' 클래스로의 효율적인 지식 전이를 기반으로 하며, 각 클래스는 객체와 관련 속성의 조합으로 정의됩니다. 모델은 사전에 정해진 조합(set of compositions)을 기반으로 예측을 수행하며, 맥락에 따라 객체 및 속성을 구분하고 재조합할 수 있어야 합니다. 연구진은 이 분야의 다양한 방법론을 비대칭성을 기준으로 네 가지 가족으로 분류하고, 이를 통해 각 방법들이 직면한 도전 미션과 성능적 요구사항을 정리하였습니다.

- **Performance Highlights**: 폐쇄형(closed-world) 및 개방형(open-world) CZSL 설정에서의 주요 성능 경향을 식별하였습니다. 특히, 크로스 모달(cross-modal) 접근법의 성공적인 발전은 CZSL 방법론의 새로운 패러다임을 제공합니다. 학술 대회 및 저널에서의 발표 추세 분석을 통해 CZSL 관련 연구의 중요성과 효과를 강조하며, 현존하는 모델들이 직면한 도전과제를 정리하고 유망한 연구 방향을 제안합니다.



### CoDefend: Cross-Modal Collaborative Defense via Diffusion Purification and Prompt Optimization (https://arxiv.org/abs/2510.11096)
- **What's New**: 이번 연구에서는 Multimodal Large Language Models (MLLMs)의 시각적 모달리티에 대한 방어 메커니즘을 제안합니다. 특히, 쌍을 이룬 adversarial... clean 이미지 데이터셋을 활용하여, 지도학습(supervised diffusion)을 기반으로 한 노이즈 제거 프레임워크를 구축했습니다. 이 접근 방식은 기존 적용된 정제 방법들과는 달리 더 높은 품질의 재구성을 달성하면서도 MLLM을 더욱 견고하게 만듭니다.

- **Technical Details**: 제안된 방법은 고급 지도학습 기반의 노이즈 제거기술을 사용합니다. 기존의 비지도 학습(unsupervised) 방법들과는 다르게, 이 방법은 안정적인 확산 모델을 미세 조정(fine-tuning)하여 특정 작업에 따른 데이터를 처리합니다. 이렇게 함으로써 다중 모달(multi-modal) 작업에서 방어 성능을 극대화하며, 특히 이미지 캡셔닝과 시각적 질문 응답에 대해 관련성과 일반성을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 이미지 캡셔닝 및 시각적 질문 응답에서 견고성을 효과적으로 개선하며, 다양한 보이지 않는 공격 전략에 대해서도 강력한 저항력을 보입니다. 또한, 상대적으로 낮은 컴퓨팅 비용으로도 높은 방어 성능을 발휘하여 실제 다양한 조건에서의 활용 가능성을 보여줍니다.



### Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution (https://arxiv.org/abs/2510.11092)
Comments:
          NeurIPS 2025

- **What's New**: SeerDrive라는 새로운 엔드투엔드 자율주행 프레임워크를 소개합니다. 이 프레임워크는 환경의 동적 변화와 차량의 미래 행동 간의 쌍방향 상호작용을 모델링합니다. 이를 통해 기존의 일회성 계획 접근 방식을 극복하고, 상황 인식 기반의 더 정교한 경로 계획을 가능하게 합니다.

- **Technical Details**: SeerDrive는 두 가지 주요 구성 요소로 이루어져 있습니다. 첫 번째는 미래 인식(planning) 기능으로, 미래의 bird's-eye view (BEV) 특징을 경로 계획에 주입합니다. 두 번째는 반복적 장면 모델링(scene modeling)과 차량 계획(vehicle planning)으로, 예측된 장면과 계획된 경로를 상호 피드백하는 방식으로 개선합니다.

- **Performance Highlights**: 많은 실험 결과, SeerDrive는 NAVSIM과 nuScenes 벤치마크에서 기존의 최신 기술을 크게 초월하는 성능을 보여주었습니다. 이는 제안한 설계의 효과성을 입증하며, 더 복잡한 동적 환경에서 상황 인식 기반의 의사결정( decision-making)을 가능하게 했습니다.



### Text-Enhanced Panoptic Symbol Spotting in CAD Drawings (https://arxiv.org/abs/2510.11091)
Comments:
          7 pages, 3figures. This version is the original submitted manuscript of the paper accepted by The 12th International Conference on Behavioural and Social Computing

- **What's New**: 이번 연구에서는 CAD(Computer-Aided Design) 도면에서 텍스트 주석을 통합한 새로운 패노프틱(symbol spotting) 기법을 제안합니다. 기존의 기법들이 기하학적 원시(primitives) 요소에만 초점을 맞춘 반면, 본 논문은 전반적인 도면 이해를 위해 텍스트와 기하학적 원시를 함께 모델링하는 방법론을 발전시켰습니다. 이러한 접근방식은 CAD 도면의 복합적인 참조와 의미를 보다 효과적으로 포착하는 데 기여합니다.

- **Technical Details**: 제안된 방법은 CAD 도면을 스캐닝하여 다양한 원시 요소로 분해하고, 각 원시는 노드로 표현되는 그래프 구조로 구성됩니다. 이 구조에서 텍스트 원시 요소는 별도의 타입으로 포함되며, CNN(Convolutional Neural Network)을 통해 초기 특징을 추출합니다. 이후 Transformer 기반의 아키텍처가 업데이트를 통해 노드 특징을 갱신하고, 타입-인식 주의(attention) 메커니즘이 공존하는 원시 간의 관계를 모델링하여 성능을 높입니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 기법들보다 뛰어난 성능을 보이며, 복잡한 CAD 도면에서 기하학적 원시와 텍스트 주석을 모두 활용하여 주목할 만한 결과를 얻었습니다. 특히, 다양한 CAD 시나리오에서의 모델의 실용성과 안정성을 검증하며 최첨단 성능을 달성하는 데 성공하였습니다.



### Source-Free Object Detection with Detection Transformer (https://arxiv.org/abs/2510.11090)
Comments:
          IEEE Transactions on Image Processing

- **What's New**: 본 논문에서는 Source-Free Object Detection (SFOD)에 대한 새로운 접근법인 Feature Reweighting ANd Contrastive Learning NetworK (FRANCK)을 제안합니다. FRANCK는 기존의 Faster R-CNN 모델에 국한되지 않고, Detection Transformer (DETR) 아키텍처에 맞춰 설계된 SFOD 프레임워크로, query-centric feature enhancement를 최적화합니다. 이 방법론은 step-wise로 objectness score, contrastive learning, feature distillation을 통해 DETR의 성능을 극대화하는 데 중점을 둡니다.

- **Technical Details**: FRANCK는 네 개의 주요 구성 요소를 포함합니다: (1) Objectness Score-based Sample Reweighting (OSSR) 모듈은 멀티 스케일 인코더 feature map에서 주의 기반의 objectness 점수를 계산하며, 덜 인식된 지역에 대한 검출 손실의 가중치를 재조정합니다. (2) Contrastive Learning with Matching-based Memory Bank (CMMB) 모듈은 클래스 간 대조 학습을 통해 메모리 뱅크에 멀티 레벨 feature를 통합합니다. (3) Uncertainty-weighted Query-fused Feature Distillation (UQFD) 모듈은 예측 품질에 따라 가중치를 재조정하며, query feature fusion을 통해 feature distillation을 개선합니다. (4) Dynamic Teacher Updating Interval (DTUI)로 개선된 셀프 트레이닝 파이프라인은 pseudo-label 품질을 최적화합니다.

- **Performance Highlights**: 우리는 여러 널리 사용되는 벤치마크에서 FRANCK의 성능을 평가하였으며, DETR 기반 SFOD에서 최첨단 성능을 달성하는 결과를 보여주었습니다. FRANCK는 기존 방법보다 더 강력한 발전을 이루었음을 확인하였고, DETR 아키텍처에 대한 적합성이 뛰어나는 것을 입증하였습니다. 이 연구는 DETR 기반 모델을 활용한 SFOD에서 도메인 적응과 일반화 능력을 향상시키기 위한 중요한 기반을 마련했습니다.



### ROFI: A Deep Learning-Based Ophthalmic Sign-Preserving and Reversible Patient Face Anonymizer (https://arxiv.org/abs/2510.11073)
Comments:
          Accepted to Nature NPJ Digital Medicine

- **What's New**: 이 논문에서는 안과 분야의 개인 정보 보호를 위한 딥러닝 기반 프레임워크인 ROFI를 소개합니다. ROFI는 약한 감독 학습(Weakly Supervised Learning)과 신경 신원 변환(Neural Identity Translation)을 활용하여 얼굴 특징을 익명화하면서 질병의 특징은 유지할 수 있도록 설계되었습니다. 이 방법은 98% 이상의 정확도로 11개의 안과 질환에 대해 100%의 진단 민감도와 높은 일치를 달성합니다.

- **Technical Details**: ROFI는 두 가지 주요 설계를 통해 실행됩니다. 첫째는 데이터 기반 안과 징후 탐지(Data-Driven Ophthalmic Sign Detection)로, 대규모의 환자 얼굴 이미지에서 질병 관련 징후를 자율적으로 분리하는 신경망을 사용합니다. 둘째는 가역적인 신경 신원 변환(Reversible Neural Identity Translation)으로, 이는 Transformer의 유연한 특징 변환 기능을 활용하여 얼굴 신원의 가역적인 변환을 가능하게 합니다.

- **Performance Highlights**: ROFI는 세 개의 임상 센터에서 17,181명의 환자를 대상으로 실시된 종합 연구에 기반을 두고 있습니다. 연구 결과, ROFI는 의료 AI 모델과 잘 통합되며, 약 95% 이상의 이미지에서 안면 익명화를 성공적으로 수행했습니다. 또한, 원본 이미지의 유사성을 98% 이상으로 재구성할 수 있어 개인의 병기록을 효율적으로 검색할 수 있도록 지원합니다.



### LSVOS 2025 Challenge Report: Recent Advances in Complex Video Object Segmentation (https://arxiv.org/abs/2510.11063)
Comments:
          16 pages, 9 figures

- **What's New**: 이번 LSVOS 2025 챌린지는 클래식 비디오 객체 분할(VOS), 참조 비디오 객체 분할(RVOS) 외에 복잡한 환경에서의 비디오 객체 분할을 위한 새로운 트랙인 복잡 VOS(MOSEv2)를 도입합니다. MOSEv2는 더 많은 및 더 복잡한 목업과 관련된 도전 과제를 포함하여 실세계 환경에서의 비디오 객체 분할의 난이도를 크게 늘리며, 이전 진전을 바탕으로 더욱 현실적인 시나리오를 제공합니다.

- **Technical Details**: MOSEv2 데이터셋은 5,024개의 비디오와 701,976개의 고품질 마스크, 200개 카테고리에 걸쳐 10,074개 객체를 포함하여 기하급수적으로 증가된 데이터로 구성됩니다. 이 데이터셋은 공간 해상도를 저하시킬 수 있는 작은 및 밀집된 타겟, 빈번한 소실 및 재출현 사건 등과 같은 표준 벤치마크에서 잘 잡히지 않는 실패 모드를 강조합니다. 평가 설정은 모델이 다양한 감독 및 상호작용 규칙에서 어떻게 작동하는지에 대한 폭넓은 관점을 제공합니다.

- **Performance Highlights**: 이번 MOSEv2 트랙에서 가장 우수한 성적을 거둔 팀은 테스트 세트에서 39.89%의 𝒥&ℱ˙ 07준위를 달성했습니다. 이 해결책은 고급 알고리즘을 활용하여 다양한 난이도를 극복하고, 복잡한 환경에서도 안정적인 객체 분할 성능을 발휘할 수 있는 방법론을 제안합니다. MOSEv2 챌린지는 실세계 상황에서의 강건한 일반화를 위한 새로운 기준으로 자리매김하며 이러한 성질이 점차 중요해지고 있음을 보여줍니다.



### Zero-shot Face Editing via ID-Attribute Decoupled Inversion (https://arxiv.org/abs/2510.11050)
Comments:
          Accepted by ICME2025

- **What's New**: 본 논문에서는 ID-속성 분리 후처리(ID-Attribute Decoupled Inversion) 기반의 제로샷(face editing) 방법을 제안합니다. 기존의 텍스트 안내 확산 모델은 얼굴 편집 시 ID와 구조적 일관성을 유지하는 데 어려움을 겪고 있습니다. 우리 방법은 얼굴 표현을 ID와 속성 특징으로 분해하여, 독립적으로 제어할 수 있는 구조를 형성합니다.

- **Technical Details**: 우리 방법의 핵심은 얼굴 이미지 임베딩을 ID(feature)로, 텍스트 임베딩을 속성(attribute)으로 사용하여 동시에 가이드를 제공하는 것입니다. 이 과정을 통해 초깃값(latent code)을 생성하고, 역 확산(reverse diffusion) 프로세스를 통해 수정된 이미지를 생성합니다. 또한, 훈련을 위해 69,900개의 얼굴 속성 설명으로 구성된 데이터셋을 구축하여 모델을 fine-tune 합니다.

- **Performance Highlights**: 실험 결과, 우리 방법은 ID 보존, 구조적 일관성, 편집 품질 면에서 기존의 최신 방법들을 초월했습니다. 특히, 텍스트 프롬프트만을 사용하여 다양한 속성 편집 작업을 처리하며, 지역 특정 입력이 필요하지 않아 높은 속도로 편집이 가능합니다. 이는 일반 이미지 편집에서의 시간 소모를 줄여 줍니다.



### Benchmarking Deep Learning Models for Laryngeal Cancer Staging Using the LaryngealCT Datas (https://arxiv.org/abs/2510.11047)
- **What's New**: 본 연구에서는 표준화된 데이터셋이 부족한 후두암 이미지 연구를 위해 LaryngealCT라는 기준을 제시합니다. 이 기준은 The Cancer Imaging Archive (TCIA)로부터 집계된 1,029개의 컴퓨터 단층촬영(CT) 스캔으로 구성되어 있습니다. 후두를 포함한 1mm 등방성(isotropic) 볼륨의 이익이 약하게 감독된 파라미터 검색 프레임워크를 통해 추출되었습니다.

- **Technical Details**: 3D 딥러닝 아키텍처(3D CNN, ResNet18,50,101, DenseNet121)가 초기(Tis, T1, T2)와 고급(T3, T4) 단계의 분류 및 T4와 비-T4의 분류 작업에서 성과를 평가했습니다. 3D CNN은 AUC 0.881과 F1-macro 0.821을 기록하였고, ResNet18은 AUC 0.892와 F1-macro 0.646로, 두 작업에서 다른 모델들보다 우수한 성능을 보였습니다.

- **Performance Highlights**: 모델 설명 가능성을 평가하기 위해 3D GradCAM을 사용하여 갑상선 연골 오버레이를 활용했습니다. 이 결과, 비-T4 사례에서 더 높은 peri-cartilage 주의가 나타났으며, T4 예측에서는 집중적인 활성화가 관찰되었습니다. LaryngealCT는 오픈소스 데이터, 사전 훈련된 모델, 통합된 설명 가능성 도구를 제공하여 후두 종양학에서 임상 결정을 지원하는 AI 연구를 위한 재현 가능한 기초를 제공합니다.



### Enhancing Zero-Shot Anomaly Detection: CLIP-SAM Collaboration with Cascaded Prompts (https://arxiv.org/abs/2510.11028)
Comments:
          Accepted by PRCV

- **What's New**: 이 논문은 산업 이상 탐지에서 제로 샷 이상 세분화(zero-shot anomaly segmentation) 작업을 위한 새로운 2단계 프레임워크를 제안합니다. 이 프레임워크는 CLIP과 SAM의 강력한 이상 지역화(anomaly localization) 및 경계 인식(boundary perception) 능력을 효과적으로 활용하여 이상 지역을 정확하게 세분화합니다. Co-Feature Point Prompt Generation (PPG) 모듈과 Cascaded Prompts for SAM (CPS) 모듈을 통해 SAM의 성능을 최적화하고, 이상 지역에 보다 집중할 수 있도록 유도합니다.

- **Technical Details**: PPG 모듈은 CLIP과 SAM의 협력적인 이용을 통해 이상 지역의 극단적인 이상 값과 주변 지역의 유사성을 고려하여 긍정 및 부정 포인트 프롬프트(point prompts)를 생성합니다. 이를 통해 SAM이 긍정적인 포인트 기능을 강조하고 부정적인 기능은 무시하게 합니다. CPS 모듈은 혼합된 포맷을 사용하여 SAM의 경계 인식 능력을 강화하고, 불완전한 세분화 및 노이즈 문제를 해결합니다.

- **Performance Highlights**: 여러 데이터셋에서 일관된 실험 검증을 통해 이 방법이 제로 샷 이상 세분화 작업에서 최신 기술 수준의 결과를 달성함을 보여줍니다. 특히 Visa 데이터셋에서 F1-max 및 AP 메트릭 지표에서 각각 10.3% 및 7.7% 더 나은 성능을 기록하며 최신 방법들을 초월합니다.



### Vlaser: Vision-Language-Action Model with Synergistic Embodied Reasoning (https://arxiv.org/abs/2510.11027)
- **What's New**: 해당 논문은 Vision-Language Models (VLMs)와 Vision-Language-Action (VLA) 모델 간의 간극을 극복하기 위해 Vlaser라는 새로운 모델을 소개합니다. Vlaser는 고급 추론과 저급 제어를 통합하여 로봇 제어를 위한 강력한 기반 모델로 설계되었습니다. 기존의 모델보다 다양한 경험적 기준에서 최고 성능을 달성하며, 이를 통해 로봇의 효율적인 상호작용을 지원합니다.

- **Technical Details**: Vlaser 구조는 일반적인 VLM 백본과 저급 제어를 위한 행동 전문가(Action Expert)로 나뉩니다. 이 모델은 InternVL3을 기반으로 하여, 2B 및 8B 크기의 최적화된 모델을 사용합니다. 각 모델은 비전 인코더로 InternViT를 이용하고, Qwen2.5 LLMs와 결합하여 임베디드 추론과 로봇 제어 기능에 중점을 두고 있습니다.

- **Performance Highlights**: Vlaser는 Vlaser-6M 데이터 세트를 기반으로 다양한 임베디드 추론 기준에서 최고의 성능을 기록하고 있습니다. 이러한 성능은 일반화 능력을 입증하며, 단일 모형 내에서 개방형 및 폐쇄형 제어 시나리오에 모두 적용 가능합니다. 특히, 로봇 조작에 관한 이론적 통찰력을 제공하며, 향후 임베디드 비전-언어 모델 개발에 필수적인 데이터 스트림은 무엇인지 체계적으로 분석하였습니다.



### GIR-Bench: Versatile Benchmark for Generating Images with Reasoning (https://arxiv.org/abs/2510.11026)
- **What's New**: 이번 논문에서는 통합된 다중 모드 모델을 평가하기 위한 새로운 벤치마크인 GIR-Bench를 소개합니다. GIR-Bench는 이해와 생성 간의 일관성을 평가하고, 논리적 제약을 적용한 텍스트-이미지 생성 기능 및 다단계 추론을 통한 편집 능력을 체계적으로 평가할 수 있는 길을 제공합니다. 기존의 여러 벤치마크는 이러한 복잡한 비주얼 작업에 대한 평가를 충분히 수행하지 못했으며, GIR-Bench는 더 정교하고 명확한 평가 기준을 제시합니다.

- **Technical Details**: GIR-Bench는 세 가지 하위 범주로 구성되어 있습니다: 이해-생성 일관성(GIR-Bench-UGC), 논리 중심의 텍스트-이미지 생성(GIR-Bench-T2I), 그리고 편집에 대한 다단계 추론(GIR-Bench-Edit)입니다. 각 하위 범주마다 고유한 작업별 평가 파이프라인을 설계하여 모델의 성능을 세밀하게 평가할 수 있도록 하였습니다. 이러한 접근 방식은 기존 MLLM-as-a-Judge 패러다임의 편향을 완화하는 데 기여합니다.

- **Performance Highlights**: 종합 모델과 생성 전용 시스템 간의 광범위한 실험 결과, 통합 모델은 이유 중심의 비주얼 작업에서 더 나은 성능을 보였지만, 여전히 이해와 생성을 연결하는 데는 지속적인 격차를 보였습니다. GIR-Bench의 설계를 통해 이러한 격차를 더 잘 이해하고 개선하기 위한 구체적인 기준을 마련함으로써, 연구자들이 다중 모드 모델의 진화를 더 효율적으로 추진할 수 있도록 돕습니다.



### GeoVLMath: Enhancing Geometry Reasoning in Vision-Language Models via Cross-Modal Reward for Auxiliary Line Creation (https://arxiv.org/abs/2510.11020)
Comments:
          22 pages

- **What's New**: 이 연구에서는 보조선(auxiliary lines)을 활용하여 입체 기하학 문제를 해결하는 새로운 접근법을 제시합니다. 기존의 이미지 편집 모델들이 기하학적 정밀성을 갖춘 보조선을 그리는데 어려움을 겪는 반면, 본 연구는 텍스트 설명을 통해 보조선 구성을 생성하여 LVLM과의 일치를 높입니다. 또한, 강화학습(Reinforcement Learning) 프레임워크를 통해 도표와 텍스트 간의 정렬을 향상시키는 방법을 제안합니다.

- **Technical Details**: 제안된 GeoVLMath 모델은 3B 및 7B 규모로, 장기적으로 비지도학습을 기반으로 한 GRPO(Group Relative Policy Optimization) 방식을 통해 보조선이 포함된 도형 텍스트 정렬의 정확성을 극대화합니다. 이때, 입력 도형과 생성된 보조선 설명 간의 일치성을 평가하는 크로스 모달 보상 모델을 중심으로 구성됩니다. 이 과정에서 실제 시험 문제에서 추출한 AuxSolidMath 데이터셋에서 3,018개의 입체 기하학 문제를 사용하여 훈련합니다.

- **Performance Highlights**: GeoVLMath는 작은 규모에서도 기존 오픈 소스 및 폐쇄 소스 LVLM 모델에 비해 경쟁력 있는 성능을 보여줍니다. 특히, Qwen2.5-VL-32B-Instruct 및 GPT-4o와 같은 더 큰 모델보다 개선된 성능을 나타내며, 보조선 구성을 활용한 지오메트릭 추론에서 높은 정확성을 기록했습니다. 이는 매개변수의 단순한 확장보다 보조선 구성을 기반으로 한 감독이 기하학적 추론 향상에 더 효과적임을 입증합니다.



### High-Resolution Spatiotemporal Modeling with Global-Local State Space Models for Video-Based Human Pose Estimation (https://arxiv.org/abs/2510.11017)
Comments:
          This paper is accepted to ICCV 2025

- **What's New**: 본 논문은 비디오 기반 인간 자세 추정(VHPE)을 위한 새로운 모델인 GLSMamba를 제안합니다. GLSMamba는 글로벌 스페이셜 템포럴 맘바(Global Spatiotemporal Mamba)와 로컬 리파인먼트 맘바(Local Refinement Mamba)로 구성되어, 고해상도 시퀀스에서의 글로벌 및 로컬 동작을 개별적으로 학습합니다. 이 방식은 기존의 모델들이 겪던 글로벌 및 로컬 동적 모델링 간의 균형 문제를 해결할 수 있는 가능성을 제공합니다.

- **Technical Details**: GLSMamba는 6D 선택적 시공간 스캔(6D Selective Space-Time Scan) 기법을 통해 고해상도 시퀀스의 글로벌 표현을 효율적으로 추출합니다. 또한, 창형 시공간 스캔(Windowed Space-Time Scan)을 기반으로 하는 로컬 정제 맘바(Local Refinement Mamba)는 지역적인 픽셀 간의 세부 사항을 강화하여 고주파 변화를 명확히 포착합니다. 이러한 구성은 기존 모델의 제약을 극복하고 더 나은 계산적으로 효율적인 성능을 제공합니다.

- **Performance Highlights**: 광범위한 벤치마크 데이터셋(PoseTrack2017, PoseTrack2018, PoseTrack21, Sub-JHMDB)에서 GLSMamba는 최신 VHPE 접근법을 능가하는 성능을 입증했습니다. 또한, 제안된 각 구성 요소의 효과를 분석하는 실험을 통해 모델의 적절성을 확인하였습니다. 최적의 계산적 거래를 유지하면서도 탁월한 성능을 달성함으로써, GLSMamba는 비디오 기반 인간 자세 추정 분야에서 중요한 기여를 하고 있습니다.



### COCO-Tree: Compositional Hierarchical Concept Trees for Enhanced Reasoning in Vision Language Models (https://arxiv.org/abs/2510.11012)
Comments:
          EMNLP 2025 (main)

- **What's New**: 이 논문에서는 'COCO-Tree'라는 새로운 접근 방식을 제안하여, 이미지 내 여러 객체와 그 관계를 이해하는 데 어려움을 겪고 있는 현대 비전 언어 모델(VLM)의 조합 추론(compositional reasoning) 성능을 개선하는 방법을 소개합니다. COCO-Tree는 큰 언어 모델(LLM)에서 학습된 신경-기호 개념 트리를 통해 VLM의 언어적 추론을 보강하며, 이 과정에서 비트 검색(beam search) 방식의 추론 프로세스를 채택하여 조합 성능을 향상시킵니다.

- **Technical Details**: COCO-Tree는 텍스트 입력을 구조적으로 비슷하지만 의미적으로 다른 형태소(entity)로 재귀적으로 분해하여, 이에 따라 LLM 추론기와 함께 신경-기호(neurosymbolic) 개념 트리를 학습합니다. 이 방법은 VLM 출력에 신경-기호 학습 경로 개념을 추가하여 조합 성능을 개선하고 그 예측에 대한 이론적 근거를 제공합니다. 논문은 COCO-Tree가 여러 오픈 소스 VLM에서 어떻게 평가되었는지를 다룹니다.

- **Performance Highlights**: COCO-Tree는 Winoground, EqBench, ColorSwap, SugarCrepe의 네 가지 벤치마크 데이터셋에서 7개의 서로 다른 오픈 소스 VLM에 대해 5-10%의 조합 능력 향상을 보여줍니다. 이를 통해 COCO-Tree는 VLM의 해석 가능성을 높여주고, 각 구성 요소의 효과를 검증하기 위한 다양한 절단(ablation) 연구를 수행하였습니다. 기존 방법들과 비교할 때, COCO-Tree의 접근 방식은 보다 자원 집약적이지 않으면서도 해석 가능성을 함께 제공하는 점에서 두드러집니다.



### Frequency Domain Unlocks New Perspectives for Abdominal Medical Image Segmentation (https://arxiv.org/abs/2510.11005)
- **What's New**: 본 논문에서는 저대비( low-contrast) 의료 이미징 상황에서 정확한 종양 및 정상 조직 분할을 위한 새로운 FASS (Foreground-Aware Spectrum Segmentation) 프레임워크를 제안합니다. FASS는 배경과 목표 영역을 더욱 효과적으로 구분하여 세부 구조를 인식할 수 있도록 설계된 여러 모듈로 구성되어 있습니다. 이 연구는 특히 복잡한 조건에서도 모델의 성능을 개선하며, 저대비 상황에서도 높은 성능을 보이는 Segmentation 기술의 필요성을 강조합니다.

- **Technical Details**: FASS 프레임워크는 세 가지 주요 모듈로 구성되어 있습니다: Foreground-Aware (FA) 모듈, Feature-Level Frequency Enhancement (FLFE) 모듈, 그리고 Edge Constraint (EC) 모듈입니다. FA 모듈은 배경과 입력 이미지 특징 간의 대조를 극대화하여 foreground feature 추출 능력을 강화합니다. FLFE 모듈은 wavelet transform 을 기반으로 고주파(high-frequency) 특성을 강조하여 경계 인식을 증진시키고, EC 모듈은 세분화 경계에서 기하학적 연속성을 보장합니다.

- **Performance Highlights**: 다양한 의료 데이터셋에서 광범위한 실험을 통해 FASS 프레임워크의 높은 성능을 입증했습니다. 각 모듈의 독립적인 기여도가 각각의 성능 향상에 크게 기여했으며, FASS 모델은 현존하는 최첨단(segmentation) 방법들보다 뛰어난 결과를 보여주었습니다. 본 연구의 결과는 저대비 이미지에서의 Segmentation 성능을 현저히 향상시키며, 복잡한 의료 이미징 상황에 대한 적응력을 개선합니다.



### ContextGen: Contextual Layout Anchoring for Identity-Consistent Multi-Instance Generation (https://arxiv.org/abs/2510.11000)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 다중 객체 생성(Multi-instance image generation)에서의 주요 한계를 극복하기 위해 ContextGen이라는 새로운 Diffusion Transformer 프레임워크를 소개합니다. 특히, Contextual Layout Anchoring (CLA) 및 Identity Consistency Attention (ICA)라는 두 가지 혁신적인 메커니즘을 통해 객체 배치와 정체성 유지를 동시에 달성합니다.

- **Technical Details**: ContextGen은 사용자 제공 또는 자동 생성된 복합 레이아웃 이미지를 사용하여 정확한 공간 제어를 가능하게 하고, 참조 이미지를 통합하여 인스턴스 정보 손실 문제를 해결합니다. 이 연구는 IMIG-100K라는 대규모 계층 구조 데이터셋을 생성하여 현재 데이터 부족 문제를 해결하고, 각 인스턴스의 정체성을 보존하기 위해 세밀한 정보를 전달하는 ICA 메커니즘을 도입합니다.

- **Performance Highlights**: ContextGen은 COCO-MIG, LayoutSAM-Eval, LAMICBench++의 세 가지 벤치마크에서 최첨단 성능을 달성하며, 기존의 방법들보다 인스턴스 수준의 성공률 +3.4%, 공간 정확도 +5.9%를 개선하여 정체성 보존에서 상업 시스템을 능가하는 결과를 보였습니다. 이러한 성과는 CLA와 ICA의 효율성을 입증합니다.



### Perspective-aware 3D Gaussian Inpainting with Multi-view Consistency (https://arxiv.org/abs/2510.10993)
- **What's New**: 이 논문에서는 다중 시점 일관성(multi-view consistency)을 보장하기 위한 새로운 접근 방식인 PAInpainter를 소개합니다. PAInpainter는 관점 인식 콘텐츠 전파(perspective-aware content propagation)와 일관성 검증(consistency verification)을 통해 3D Gaussian 이채우기를 진행합니다. 이 방법은 다양한 관점에서 적응적으로 샘플링한 이미지를 사용하여 전반적인 일관성과 텍스처 충실도를 크게 향상시킵니다.

- **Technical Details**: PAInpainter는 관점 그래프 샘플링(perspective graph sampling), 콘텐츠 전파(content propagation), 그리고 일관성 검증(consistency verification)이라는 세 가지 주요 구성 요소로 이루어져 있습니다. 이를 통해 3D Gaussian 장면에서 잃어버린 영역을 여러 시점에서 반복적으로 보완하고, 그 과정에서 인페인팅된 이미지를 사용하여 3D Gaussian 표현을 최적화합니다. 또한, 이중 특징 검증 메커니즘이 텍스처와 기하학적 일관성을 평가하여 신뢰할 수 있는 결과를 추출합니다.

- **Performance Highlights**: PAInpainter는 다양한 시나리오에서 높은 품질의 3D Gaussian 이채우기를 달성하였으며, SPIn-NeRF 및 NeRFiller 데이터셋에서 각각 26.03 dB 및 29.51 dB의 PSNR 점수를 기록하며 기존 방법들과 비교하여 우수한 성능을 입증했습니다. 대규모 실험을 통해 PAInpainter가 다양한 3D 장면에서 뛰어난 일반화 능력을 보이며 시각적 충실도 및 일관성 측면에서도 탁월한 결과를 도출함을 강조합니다.



### A Survey on Agentic Multimodal Large Language Models (https://arxiv.org/abs/2510.10991)
- **What's New**: 최근 자율 에이전트 시스템의 출현으로, 전통적이고 정적이며 도메인 특화된 AI 에이전트에서 보다 동적이고 적극적이며 일반화 가능한 에이전틱 AI로의 변화가 일어나고 있습니다. 본 논문은 에이전틱 다중 모달 대형 언어 모델(Agentic MLLMs)에 대한 포괄적인 조사 연구를 소개하며, 기존 MLLM 기반 에이전트와의 개념적 기초 및 차별성을 명확히 합니다. 에이전틱 MLLMs의 구조는 내부 지능 기능, 외부 도구 호출, 환경 상호작용의 세 가지 기본 차원에 따라 정리되었습니다.

- **Technical Details**: 본 논문에서는 에이전틱 MLLMs의 주요 특징으로, 시스템의 지휘자로 작용하는 내부 지능 기능, 다양한 외부 도구를 활용하여 문제 해결 능력을 확장하는 외부 도구 호출, 동적 현실 시나리오에서 목표 지향적 행동을 유지하는 환경 상호작용을 설명합니다. 에이전틱 MLLMs는 기존의 정적이고 고정된 워크플로우에서 벗어나, 상황에 맞춰 전략 및 작업 흐름을 조정할 수 있는 자율적 의사 결정자로서의 기능을 강조합니다. 이 새로운 패러다임은 기존 MLLM에서의 단순한 질의-응답 접근법이 아닌, 도메인 간 일반화 및 적응적인 문제 해결 과정을 가능하게 합니다.

- **Performance Highlights**: 에이전틱 MLLMs는 그들이 가진 본질적 특성 덕분에 전략과 워크플로우를 동적으로 조정하며, 작업 수행에 있어서의 능동성을 보여줍니다. 이들은 다양한 작업과 환경에서 작동할 수 있는 가능성을 가지고 있으며, 따라서 넓은 범위의 활용 사례에 적합합니다. 이 연구는 에이전틱 MLLMs의 발전 경과를 체계적으로 정리하고, 주요 도전 과제와 향후 연구 방향을 제시하여 이 빠르게 진화하는 분야에서 최신 동향을 추적할 수 있는 기초 자료를 제공합니다.



### Mixup Helps Understanding Multimodal Video Better (https://arxiv.org/abs/2510.10986)
- **What's New**: 본 연구에서는 다중 모달 비디오 이해(multi-modal video understanding) 분야에서의 오버피팅(overfitting) 문제를 해결하기 위한 새로운 방법론, Multimodal Mixup (MM)와 Balanced Multimodal Mixup (B-MM)을 제안합니다. MM은 멀티모달 특성(feature) 집합에서의 Mixup 전략을 적용하여 오버피팅을 감소시키며, B-MM은 각 모달의 학습 기여도에 따라 혼합 비율을 동적으로 조정하여 모달 간 균형 잡힌 표현 학습을 촉진합니다.

- **Technical Details**: 연구는 두 가지 입력 모달인 오디오(mam_{a})와 비디오(mvm_{v})를 고려하고, 멀티모달 모델은 각 모달에 대한 단일 모달 인코더를 사용하여 특징을 추출합니다. 이 인코더의 출력은 앙상블 되며, 공통 학습 목표에 따라 서로 다른 모달의 혼합 비율이 조정됩니다. 이 과정은 모달 간의 기여도를 기반으로 동적으로 진행되며, 모델이 특정 강한 모달에 의존하지 않도록 합니다.

- **Performance Highlights**: 다양한 벤치마크 데이터셋(CREMAD, Kinetic-Sounds, UCF-101)에서의 실험 결과, MM 및 B-MM 방식이 기존의 전통적인 융합(fusion) 기법과 최신 균형 잡힌 멀티모달 학습 방법에 비해 우수한 성능을 보인다는 것을 확인하였습니다. 이러한 결과는 제안된 방법들이 오버피팅 방지 및 멀티모달 협업(multimodal cooperation) 향상에 효과적임을 입증합니다.



### Chart-RVR: Reinforcement Learning with Verifiable Rewards for Explainable Chart Reasoning (https://arxiv.org/abs/2510.10973)
Comments:
          23 pages

- **What's New**: 이번 논문에서는 Large Vision-Language Models (LVLMs)의 한계를 해결하기 위해 새로운 프레임워크인 Chart-RVR를 제안합니다. 이 프레임워크는 Group Relative Policy Optimization (GRPO)와 자동 검증 가능한 보상을 결합하여 차트 추론에서의 견고성과 설명 가능성을 증대시킵니다. 특히, Chart-RVR은 차트 유형 분류, 표 구조 재구성, 절차 일치율을 최적화하는 세 가지 보상을 포함하고 있습니다.

- **Technical Details**: Chart-RVR는 3억 개의 매개변수를 가진 LVLM에 적용되어 표준 감독 미세 조정(supervised fine-tuning, SFT) 방식보다 우수한 성능을 보입니다. 새로운 방법론은 Reinforcement Fine-Tuning (RFT) 기법을 활용하여 모델의 예측 정확도를 높이고, 지속가능한 훈련을 가능하게 하며, 두 가지 목표(tasks)인 차트 유형 예측과 표 재구성을 위한 검증 가능한 보상을 사용합니다. 마지막으로, GRPO와 검증 가능한 보상의 결합은 모델의 훈련 시 안정성을 증가시킵니다.

- **Performance Highlights**: Chart-RVR 모델은 6개의 차트 추론 벤치마크에서 최첨단 성능을 달성하여 기존의 유사한 크기의 모델들을 초월합니다. 이 연구는 또한 모델이 더 해석 가능한 CoT (chain-of-thought) 논리를 생성함으로써 신뢰성과 신뢰성을 강화하는 방법을 보여줍니다. 종합적으로, Chart-RVR은 차트 추론의 정확도와 해석 가능성 모두에서 개선을 나타내었습니다.



### IUT-Plug: A Plug-in tool for Interleaved Image-Text Generation (https://arxiv.org/abs/2510.10969)
- **What's New**: 이 논문에서는 IUT-Plug라는 모듈을 제안하여 기존의 Vision Language Models (VLMs)의 다중 모달 이미지-텍스트 생성에서 발생하는 세 가지 주요 문제를 해결합니다. 이 모듈은 Image Understanding Trees (IUT)라는 계층적 구조를 활용하여, 논리, 개체 정체성 및 스타일 유지에 도움을 줍니다. IUT-Plug는 기존 모델과 통합될 수 있도록 설계되어 있으며, 비용이 많이 드는 전체 재훈련이 필요하지 않습니다.

- **Technical Details**: IUT-Plug는 두 가지 단계로 구성된 프레임워크에서 작동합니다. 첫 번째 단계에서는 동적인 IUT-Plug 추출 모듈이 시각 장면을 계층적 기호 구조로 파싱합니다. 두 번째 단계에서는 내러티브 흐름(narrative flow)과 이미지 합성 메커니즘이 장치 간 일관성을 보장합니다. 이러한 접근법은 인식의 강도를 신경망에서 논리적 정밀도로 분리하여 시스템의 일관성을 유지하면서 현대 생성 모델의 유연성을 확보합니다.

- **Performance Highlights**: IUT-Plug를 통해 기존 벤치마크에서 정확도가 향상되고, 다양한 다중 모달 질문 응답(QA) 시나리오에서 발생하는 세 가지 주요 형태의 컨텍스트 드리프트(context drift)가 효과적으로 완화되었다는 실험 결과를 제시합니다. 또한, 3,000개의 인간 생성 질문-답변 쌍에 기반한 새로운 벤치마크를 구성하여 다이나믹 평가 프로토콜을 도입하였습니다. 이로 인해 IUT-Plug는 다중 모달 생성에서의 구성 일관성(compositional consistency)을 측정하고 달성하는 새로운 기준을 세웠습니다.



### Towards Distribution-Shift Uncertainty Estimation for Inverse Problems with Generative Priors (https://arxiv.org/abs/2510.10947)
Comments:
          Code is available at this https URL

- **What's New**: 이 논문에서는 의료 이미지를 재구성하는 문제에 대한 새로운 접근 방식인 인스턴스 수준의 캘리브레이션 프리 불확실성 지표(instance-level calibration-free uncertainty indicator)를 제안합니다. 이 지표는 훈련 분포에 대한 사전 지식 없이도 배포할 수 있으며, 재훈련 비용이 발생하지 않습니다. 이미지를 재구성할 때, 내부 캘리브레이션 세트를 활용하여 이미지의 일관성을 평가할 수 있으며 이를 통해 분포 이동 메커니즘을 효과적으로 감지할 수 있습니다.

- **Technical Details**: 제안된 불확실성 지표는 재구성 작업에서 측정 값의 무작위 하위 샘플을 사용하여 서로 다른 측정 집합으로부터의 재구성이 일관된지 평가하는 방식으로 작동합니다. 재구성이 생성적 사전(generative prior)으로 비 훈련 데이터(out-of-distribution)인지 여부를 가늠하는 지표로써 활용됩니다. 이 지표는 MNIST 숫자의 토모그래픽 재구성(tomographic reconstruction)에 적용되었으며, 훈련 데이터로는 숫자 '0'만 사용하고 테스트 데이터로는 모든 숫자를 포함한 실험에서 검증되었습니다.

- **Performance Highlights**: 실험 결과, 훈련 배포 내(in-distribution) 데이터로부터의 재구성은 서로 간의 변동성이 적은 반면, 훈련 배포 외(out-of-distribution) 데이터에서의 재구성은 높은 변동성과 더 큰 재구성 오류를 보였습니다. 이러한 결과는 제안된 불확실성 지표가 유용함을 입증하며, 의료 진단 환경에서 신뢰성 있는 이미지 재구성을 위해 추가적인 측정이 필요하게 되는 상황을 자동 경고하는 메커니즘으로 활용될 수 있습니다.



### DKPMV: Dense Keypoints Fusion from Multi-View RGB Frames for 6D Pose Estimation of Textureless Objects (https://arxiv.org/abs/2510.10933)
Comments:
          12 pages, 9 figures, submitted to ICRA 2026

- **What's New**: 이 논문에서는 RGB 이미지만을 사용하여 다중 뷰의 keypoint-level fusion을 수행하는 DKPMV라는 새로운 파이프라인을 제안합니다. 기존 방법들은 depth data에 의존하거나 다중 뷰 기하학적 단서를 충분히 활용하지 못하는 점에서 제한적이었습니다. 따라서 본 연구는 keypoint network를 개선하고, 대칭 인식을 위한 훈련 방식을 적용하여 6D 포즈 추정의 정확성을 향상시킵니다. ROBI 데이터셋을 통한 광범위한 실험 결과, DKPMV는 최신 RGB 및 RGB-D 기반 접근 방식을 초월하는 성능을 보였습니다.

- **Technical Details**: DKPMV 파이프라인은 조밀한 keypoint 레벨의 융합을 달성하기 위해, 심층 네트워크를 통해 다중 뷰 RGB 이미지를 입력으로 사용합니다. 이 프로세스에서는 먼저 YOLOv1을 통해 객체의 2D 바운딩 박스를 감지하고, 이를 기반으로 KeypointNet-SAT 네트워크에서 조밀한 keypoint 예측을 생성합니다. 이후 다중 뷰 매칭 모듈을 사용하여 각 객체 인스턴스에 대한 일관된 keypoint 대응 관계를 확립합니다. 이러한 접근 방식은 개체의 기하학적 특징을 통해 포즈 추정의 정확성을 높이는 데 기여합니다.

- **Performance Highlights**: DKPMV는 ROBI 데이터셋을 바탕으로 대규모 실험을 진행한 결과, 최신 RGB 및 RGB-D 기반 방법에 비해 뛰어난 성능을 발휘했습니다. 특히, 기존의 다중 뷰 접근 방식이 겪는 깊이 손실 및 비율 모호성 같은 문제를 효과적으로 해결하였습니다. 대칭 물체에 대한 예측 모호성을 해소하기 위해 추가된 대칭 인식을 통한 훈련 전략(SAT)은 예측 정확도를 더욱 개선하였습니다. 이 연구는 실시간 로봇 인식에 대한 적용 가능성을 확대하는 데 기여할 것입니다.



### FG-CLIP 2: A Bilingual Fine-grained Vision-Language Alignment Mod (https://arxiv.org/abs/2510.10921)
- **What's New**: FG-CLIP 2는 영어와 중국어를 위한 새로운 이중 언어( bilingual) 비전-언어( vision-language) 모델로, 미세한 정렬(fine-grained alignment)을 개선하기 위해 개발되었습니다. 기존의 모델들이 언어 표현, 시각적 콘텐츠, 및 객체 속성의 세밀한 정렬에서 어려움을 겪던 문제를 해결하기 위해, FG-CLIP 2는 지역-텍스트 매칭(region-text matching)과 긴 캡션 모델링(long-caption modeling) 기법을 활용하고 있습니다. 또한, Textual Intra-modal Contrastive (TIC) 손실을 도입하여 의미적으로 유사한 캡션을 더 잘 구분할 수 있도록 설계되었습니다.

- **Technical Details**: FG-CLIP 2는 두 단계 학습 프레임워크를 따릅니다. 첫 번째 단계에서는 이미지-텍스트 쌍을 활용하여 전역 정렬(global alignment)을 수행하고, 두 번째 단계에서는 지역 정렬(region-level alignment) 및 미세 대비 신호(fine-grained contrastive signals)를 포함하여 모델의 성능을 재조정합니다. 텍스트 인코더는 최대 입력 길이를 64에서 196 토큰으로 확장하여 긴 설명을 수용할 수 있으며, 비전 측면에서는 데이터 적응형 해상도(data-adaptive resolution) 전략을 채택하여 일관된 훈련과 추론을 가능하게 합니다.

- **Performance Highlights**: FG-CLIP 2는 29개의 데이터셋에서 진행된 광범위한 실험을 통해 기존 모델들을 능가하며, 영어와 중국어 모두에서 최첨단 결과를 달성했습니다. 또한, 중국 멀티모달 이해를 위한 새로운 벤치마크(benchmark)를 제시하였으며, 이를 통해 긴 캡션 검색 및 바운딩 박스 분류와 같은 과제를 더 철저히 평가할 수 있습니다. FG-CLIP 2는 다국어 환경에서의 비전-언어 조정 능력을 강화하여, 향후 연구와 실제 적용에 기여할 모델로 자리 잡을 것으로 기대됩니다.



### DreamMakeup: Face Makeup Customization using Latent Diffusion Models (https://arxiv.org/abs/2510.10918)
- **What's New**: 이번 논문에서는 GANs의 기술적 한계를 극복하기 위해 DreamMakeup이라는 새로운 훈련 없는 확산 기반 메이크업 커스터마이징 방법을 소개합니다. DreamMakeup은 얼굴 구조와 정체성을 보존하는 동시에 다양한 조건 입력(예: 참조 이미지, 특정 RGB 색상, 텍스트 설명)을 통해 메이크업을 광범위하게 개인화할 수 있는 장점을 제공합니다. 이는 기존 GAN과 최근의 확산 기반 프레임워크보다 커스터마이징 및 색상 일치 기능에서 현저히 향상된 성능을 보여줍니다.

- **Technical Details**: DreamMakeup은 초기 중단 DDIM 인버전을 활용하여 주어진 얼굴 이미지의 구조를 보존하면서 메이크업 스타일을 목표로 하는 픽셀 공간에서의 변형을 가능하게 합니다. 이 과정은 하모니와 일관성을 위해 고급 크로스 어텐션 제어와 보간 가이드 샘플링을 포함하여, 사용자가 원하는 메이크업 스타일을 충족하도록 조정됩니다. 이를 통해 사용자가 다양한 조건을 사용하여 메이크업 프로세스를 유도할 수 있습니다.

- **Performance Highlights**: DreamMakeup은 실세계 글로벌 AI 메이크업 서비스와 색상 메이크업 작업에서 경쟁 우위를 보이며, 메이크업 전송 작업에서도 최신 확산 및 GAN 기반 프레임워크를 초과하는 성과를 보였습니다. 이 모델은 Large Language Models (LLMs)와의 통합이 용이하여, 다양한 사용자 요구에 대한 응답성을 극대화할 수 있습니다. 또한, 컴퓨팅 비용이 저렴하여, 고성능 그래픽 카드로도 4초 미만의 지연 시간으로 색상 전환을 수행할 수 있습니다.



### SceneTextStylizer: A Training-Free Scene Text Style Transfer Framework with Diffusion Mod (https://arxiv.org/abs/2510.10910)
- **What's New**: 이 논문에서는 SceneTextStylizer라는 새로운 방법을 소개하여, 장면 이미지 내의 텍스트에 대해 유연하고 높은 충실도의 스타일 전이를 가능하게 합니다. 기존의 방법들이 전체 이미지에 대한 스타일 전이와 텍스트 내용 수정에만 집중해왔던 것과 달리, 이 방법은 특정 텍스트 영역에 대한 프롬프트 기반의 스타일 전이를 제공하며, 텍스트 가독성과 스타일 일관성을 유지하는 데 중점을 두고 있습니다. 이 알고리즘은 훈련이 필요 없는 확산 기반의 프레임워크로, 경량한 Feature Injection 모듈과 지역 제어 메커니즘을 활용하여 텍스트의 스타일을 효과적으로 전이합니다.

- **Technical Details**: SceneTextStylizer는 텍스트 영역 스타일 전이를 위한 training-free 확산 모델을 기반으로 하여, DDIM inversion과 self-attention 기능을 통해 내용과 스타일의 생성을 분리합니다. 각 디노이징 단계에서 거리 기반 마스크를 적용하여 텍스트 지역 내에서 지역적으로 최적화하고, 푸리에 변환을 이용한 스타일 향상 모듈을 통해 시각적 품질을 높입니다. 이 방법은 사전 훈련된 모델 내에서 플러그 앤 플레이 방식으로 작동하여, 추가적인 훈련이나 미세 조정이 필요 없습니다.

- **Performance Highlights**: 다양한 실험 결과에서 SceneTextStylizer는 기존의 최첨단 방법들과 비교하여 시각적 충실도와 텍스트 보존 모두에서 우수한 성능을 발휘함을 보여주었습니다. 이 방법은 텍스트 스타일 변형의 자유로운 형태를 효과적으로 해결하며, 텍스트 특정 스타일 변환을 가능하게 합니다. 논문에서 소개된 방식은 텍스트 지역에서의 지역 조사 및 자연스러운 블렌딩을 보장하여, 기존 방법들의 한계를 극복하고 유연하고 다양한 스타일링을 제공합니다.



### Topological Alignment of Shared Vision-Language Embedding Spac (https://arxiv.org/abs/2510.10889)
Comments:
          24 pages, 5 figures, 19 tables

- **What's New**: 본 논문에서는 멀티링구얼 Contrastive Vision-Language Model (VLM)에서의 구조적 비일치를 해결하기 위해 ToMCLIP(Topological Alignment for Multilingual CLIP)라는 새로운 프레임워크를 제안합니다. 기존 모델들은 주로 개별 인스턴스 수준의 정렬(instance-level alignment)만을 고려했지만, ToMCLIP은 토포로지 분석을 통해 전반적인 구조적 정렬을 이룰 수 있도록 합니다. 이를 통해 영어와 다른 언어 간의 성능 격차를 줄이려는 노력이 포함되어 있습니다.

- **Technical Details**: ToMCLIP은 Persistent Homology를 활용하여 토포로지 정렬 손실(topological alignment loss)을 정의하고, 이를 통해 언어 간 공통 임베딩 공간에서 구조적 정렬을 강제합니다. 본 연구에서는 그래프 희소화(graph sparsification) 전략을 통해 필수 다이어그램(persistence diagram)을 근사하는 방법도 개발하였습니다. 이를 통해 MCLIP이 기존의 점 대 점 정렬(point-wise alignment)에서 벗어나 보다 전반적인 구조적 일관성을 유지할 수 있도록 하고 있습니다.

- **Performance Highlights**: 제안된 ToMCLIP 방식은 CIFAR-100 데이터셋에서 제로샷(zero-shot) 정확도를 개선하고, xFlickr&CO에서의 멀티링구얼 검색 성능을 더욱 향상시키는 결과를 보였습니다. 실험 결과는 다양한 시나리오에서 멀티링구얼 표현의 구조적 일관성이 개선되었음을 입증하였습니다. 또한, 이 방법은 VLM뿐만 아니라 일반적으로 표현 학습에 토포로지 정렬을 도입할 수 있는 가능성을 제시합니다.



### Where on Earth? A Vision-Language Benchmark for Probing Model Geolocation Skills Across Scales (https://arxiv.org/abs/2510.10880)
- **What's New**: 본 논문에서는 EarthWhere라는 VLM 이미지 지리적 위치 측정을 위한 종합 벤치마크를 소개합니다. 이 benchmark는 VLM이 시각 인식, 단계별 추론(step-by-step reasoning) 및 증거 사용(evidence use)을 평가하는 데 초점을 맞추고 있습니다. EarthWhere는 두 가지 보완적인 지리적 위치 스케일에서 총 810개의 전 세계 이미지로 구성됩니다. 이러한 체계적인 평가 방식을 통해 모델의 성능을 객관적으로 분석할 수 있는 새로운 기회를 제공합니다.

- **Technical Details**: EarthWhere는 WhereCountry와 WhereStreet라는 두 가지 태스크로 구성되어 있습니다. WhereCountry는 국가 수준의 지리적 위치 측정 태스크로 500개의 영상이 포함되어 있으며, WhereStreet는 다단계 추론이 필요한 거리 수준의 이미지로 310개가 수록되어 있습니다. 평가 지표로는 k km 내 위치 정확도(Acc@k)와 텍스트 기반의 계층적 경로 점수를 사용하며, 모델의 추론 정확성을 보다 상세히 분석하기 위해 인간 검증된 시각 단서를 사용합니다.

- **Performance Highlights**: 평가 결과, Gemini-2.5-Pro 모델이 56.32%의 평균 정확도로 가장 우수한 성과를 보였고, 오픈 소스 모델인 GLM-4.5V는 34.71%로 뒤쳐졌습니다. 웹 검색 및 추론이 시각힌트가 제한된 상황에서는 성능 향상을 보장하지 않으며, 지역 편향이 발견되었습니다. 이러한 발견은 모델들이 편향을 완화하고 견고한 세분화된 위치 측정을 달성하는 것이 여전히 도전 과제임을 시사합니다.



### rareboost3d: a synthetic lidar dataset with enhanced rare classes (https://arxiv.org/abs/2510.10876)
- **What's New**: 본 논문에서는 기존의 실세계 데이터셋에서 희귀 클래스(rare class)의 부족 문제를 해결하기 위해 RareBoost3D라는 새로운 합성(synthetic) 포인트 클라우드 데이터셋을 소개합니다. 이 데이터셋은 LiDAR 기반의 인식 기술을 위해 더 많은 희귀 클래스 인스턴스를 제공합니다. 또한, CSC 손실(CSC loss)이라는 새로운 방법을 제안하여 합성 데이터와 실세계 데이터 간의 세미틱 추상화(semantic alignment)를 효과적으로 수행합니다.

- **Technical Details**: RareBoost3D 데이터셋은 CARLA 시뮬레이터를 사용하여 생성되었으며, 다양한 도시와 농촌 장면에서 포인트 클라우드 시퀀스를 수집합니다. 총 29개의 세미틱 레이블로 구성되어 있으며, 각 시퀀스는 약 60,000개의 스캔을 포함하고 있습니다. CSC 손실은 대조 학습(contrastive learning)을 기반으로 하여 합성 데이터와 실세계 데이터의 특징을 정렬하는 기법입니다.

- **Performance Highlights**: 실험 결과, CSC 손실을 통해 세미틱 분할(segmentation) 모델의 성능이 약 2%에서 3% 향상되었습니다. 이를 통해 합성 데이터가 실세계에서의 일반화 성능을 높이는 데 크게 기여함을 확인했습니다. RareBoost3D는 희귀 클래스의 대표 학습을 강화해주는 효과적인 자원으로 작용합니다.



### FastHMR: Accelerating Human Mesh Recovery via Token and Layer Merging with Diffusion Decoding (https://arxiv.org/abs/2510.10868)
Comments:
          Project page: this https URL

- **What's New**: 최근의 3D 인간 메시 회복(3D Human Mesh Recovery, HMR) 모델들은 강력한 성능을 보였지만, 깊은 transformer 아키텍처와 중복 토큰으로 인해 높은 계산 비용과 복잡성을 경험했습니다. 이 논문에서는 Mean Per Joint Position Error (MPJPE)에 미치는 영향을 최소화하여 계층을 선택적으로 병합하는 Error-Constrained Layer Merging (ECLM)과 최종 예측에 기여하지 않는 배경 토큰을 병합하는 Mask-guided Token Merging (Mask-ToMe) 두 가지 HMR 전용 병합 전략을 도입합니다. 성능 손실을 보완하기 위해 통계적 맥락을 포함하고 대규모 모션 캡처 데이터셋에서 학습한 포즈 프라이어를 활용하는 확산 기반 디코더를 제안합니다.

- **Technical Details**: 3D HMR은 단안 카메라로 촬영된 이미지나 비디오에서 인간의 포즈와 형태를 추정하는 과정을 포함합니다. 연구자들은 대칭 레이어와 공간적 중복을 통해 계산 효율성을 높이고, 두 가지 병합 전략 ECLM과 Mask-ToMe를 제안하였습니다. ECLM은 출력 차이가 오류 기준 아래에 있는 레이어를 병합하여 모델 깊이를 줄이고 정확성을 유지하며, Mask-ToMe는 불필요한 배경 토큰을 병합하여 토큰 수를 줄입니다. 이를 통해 연산 부하를 크게 줄이면서도 정확도 저하를 최소화하는 전략을 실시합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 최대 2.3배의 속도를 증가시키면서 기초 모델보다 약간 향상된 성능을 보여줍니다. 이 연구는 더 낮은 계산 비용으로 실시간 또는 자원 제한 환경에서도 적용 가능한 효율적인 HMR 시스템을 보장합니다. FastHMR 프레임워크는 대규모 포즈 데이터에서 학습한 프라이어를 사용하여 더 정확하고 부드러운 메시를 생성하는 데 기여하며, 기존 transformer 기반의 HMR 파이프라인보다 향상된 결과를 제공합니다.



### From Detection to Mitigation: Addressing Bias in Deep Learning Models for Chest X-Ray Diagnosis (https://arxiv.org/abs/2510.10822)
Comments:
          Preprint of an article published in Pacific Symposium on Biocomputing \c{opyright} 2026 World Scientific Publishing Co., Singapore, this http URL

- **What's New**: 본 연구는 남성과 여성, 연령, 인종 기반의 불평등을 목표로 한 포괄적인 편향 감지 및 완화 프레임워크를 제시합니다. 우리는 최신 CNN-XGBoost 파이프라인을 다중 레이블 분류를 지원하도록 확장하고, 흉부 X선(CXR) 이미지를 이용한 진단 작업의 성능을 평가했습니다. XGBoost 분류기를 활용해 CNN의 마지막 레이어를 교체함으로써, 소그룹 간의 공정성을 향상시키면서도 예측 성능을 유지 또는 개선하는 결과를 보였습니다.

- **Technical Details**: 연구에서는 CheXpert와 MIMIC라는 두 개의 대규모 공개 CXR 데이터셋을 사용합니다. 각 데이터셋은 성별, 연령 및 인종에 대한 인구 통계 정보를 포함하고 있으며, 우리는 이를 통해 모델 성능을 분석하고자 합니다. 최종 레이어에서 CNN을 제거한 후, 임베딩을 동결하고 XGBoost 분류기로 재훈련함으로써 편향을 완화하는 경량화된 전략을 제안합니다.

- **Performance Highlights**: 우리의 방법은 다양한 CNN 기반 아키텍처에 효과적으로 통합될 수 있으며, 성과와 편향 감소에서 유사한 향상을 보여주었습니다. XGBoost는 성과, 공정성, 계산 비용 간의 최적의 트레이드오프를 제공하는 것으로 나타났고, 활성 학습과 결합할 경우 모든 인구 통계 소그룹에서 편향을 가장 크게 줄일 수 있음을 입증하였습니다.



### MSCloudCAM: Cross-Attention with Multi-Scale Context for Multispectral Cloud Segmentation (https://arxiv.org/abs/2510.10802)
Comments:
          7 pages, 2 Figures

- **What's New**: 본 논문에서는 환경 모니터링과 기후 연구를 위한 신뢰할 수 있는 분석을 방해하는 클라우드(구름) 문제를 해결하기 위해 개발된 MSCloudCAM 모델을 소개합니다. 이 모델은 멀티스펙트럴 및 멀티센서 클라우드 세분화를 위해 설계되었으며, 네트워크는 Swin Transformer와 ASPP, PSP 모듈을 통합하여 다중 스케일 컨텍스트 집합을 구현합니다. MSCloudCAM은 구체적으로 맑은 하늘, 얇은 구름, 두꺼운 구름, 구름 그림자 네 가지 범주로 분류하는 성능을 발휘합니다.

- **Technical Details**: MSCloudCAM은 Swin Transformer 인코더와 다중 문맥 모듈, 크로스 어텐션 융합 메커니즘을 통합하여 장거리 스펙트럼-공간 의존성을 포착하고 세분화된 로컬 구조를 유지합니다. 입력으로는 Sentinel-2와 Landsat-8의 멀티스펙트럴 이미지를 사용하며, 각 이미지는 13 또는 11개의 스펙트럴 채널로 구성됩니다. 이 모델의 디코더는 보조 감독 기능을 통해 최종 밀집 예측을 생성합니다.

- **Performance Highlights**: CloudSEN12 및 L8Biome 데이터셋에 대한 포괄적 실험 결과 MSCloudCAM은 최첨단 세분화 정확도를 달성하며, 주요 기준 아키텍처를 초과하는 성과를 보였습니다. 또한 이 모델은 경쟁력 있는 파라미터 효율성과 FLOPs를 유지하여, 대규모 지구 관측 작업 및 실제 응용 프로그램에 적합함을 입증하였습니다. 이러한 결과는 MSCloudCAM의 효과성과 실용성을 강조하며, 다양한 센서와 스펙트럼 도메인에서의 정확한 픽셀 단위 분류를 가능하게 합니다.



### Full segmentation annotations of 3D time-lapse microscopy images of MDA231 cells (https://arxiv.org/abs/2510.10797)
Comments:
          6 pages, 2 figures, 4 tables

- **What's New**: 이번 논문에서는 이미징(duplicated 처리된) 데이터셋을 위한 고품질의 공개 세분화 주석(annotation)을 소개합니다. 특히, 동적인 형태의 세포를 포함한 생체 이미지 분석에서 처음으로 완전한 3D 시간간격 세분화 주석이 공개됩니다. 이 데이터셋은 시간에 따른 세포 이동을 기록한 두 개의 시퀀스를 포함하고 있으며, 세 명의 인간 주석자에 의해 주석이 달렸습니다.

- **Technical Details**: Fluo-C3DL-MDA231 데이터셋은 생체 조직의 세포 구조를 포괄적으로 3D로 표현하며, 기존의 CTC 주석들과 일관성을 유지합니다. 각 시퀀스의 주석은 다수결 투표를 통해 통합하였고, 이전의 주석과 비교하여 품질 및 커버리지 면에서 우수한 성능을 입증하였습니다. 특히, 이 데이터셋은 고도로 동적인 세포의 세분화 및 분석에 적합합니다.

- **Performance Highlights**: 제안된 3D 주석은 CTC에서 제공하는 자동 생성된 은색 진실을 포함하여 세포 이미지의 복잡성을 더 잘 표현합니다. 이 데이터셋은 세포 세분화 알고리즘의 개발 및 테스트에 활용될 수 있으며, 커다란 데이터세트를 제공함으로써 세포 분석의 발전을 도울 것으로 기대됩니다. 다양한 방식에서 세포 추적 및 세분화를 향상시킬 수 있는 기초 자료로 사용될 수 있습니다.



### ImHead: A Large-scale Implicit Morphable Model for Localized Head Modeling (https://arxiv.org/abs/2510.10793)
Comments:
          ICCV 2025

- **What's New**: 최근 몇 년간 3D 변형 모델(3DMMs)은 표현력 있는 3D 아바타 모델링 및 생성에 있어 최첨단 방법론으로 자리 잡았습니다. 그러나 기존 모델은 엄격한 토폴로지에 기반하여 복잡한 전체 머리 형태를 표현하는 데 어려움을 겪고 있습니다. 이에 따라, 우리는 imHead라는 새로운 방법을 제안하여, 더 나은 표현력과 얼굴 특징의 지역 편집을 가능하게 하고자 합니다.

- **Technical Details**: imHead는 단일 압축된 정체성(latent identity) 공간을 유지하면서 지역 특성에 대한 중간 표현을 도입하여, 지역 편집을 용이하게 합니다. 또한, 우리는 4K의 독특한 정체성을 가진 대규모 데이터세트를 구성하여 imHead의 훈련을 지원하며, 이는 3DMM의 모델링 능력을 크게 향상시킵니다. 과거 데이터세트의 크기를 10배 증가시킴으로써 다양한 정체성과 표현을 포착할 수 있도록 하였습니다.

- **Performance Highlights**: imHead는 다양한 정체성과 표현을 표현할 수 있는 능력이 뛰어나며, 이전의 접근 방식보다 우수한 성능을 보입니다. 특히, 사용자가 지역적으로 편집할 수 있는 기능을 제공하여 3D 얼굴 조작을 보다 직관적으로 만들어 줍니다. 진행된 실험에서 제안한 모델은 실제적이고 다양한 머리 형태를 표현하는 데 성공을 거두었습니다.



### DISC-GAN: Disentangling Style and Content for Cluster-Specific Synthetic Underwater Image Generation (https://arxiv.org/abs/2510.10782)
- **What's New**: 이 논문에서는 Disentangled Style-Content GAN (DISC-GAN)이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 스타일-콘텐츠 분리를 통해 수중 이미지 합성을 위한 클러스터 특정 훈련 전략을 통합하여 포토리얼리스틱한 수중 이미지를 생성하는 것을 목표로 합니다. 기존의 생성 모델이 다양하고 비균일한 수중 환경의 조건을 모델링하는 데 한계를 가지고 있음을 언급하며, 이를 해결하기 위해 K-means 클러스터링을 활용하여 데이터셋을 스타일 특정 도메인으로 분할하고 각 클러스터에서 독립적으로 모델을 훈련합니다.

- **Technical Details**: DISC-GAN은 각각의 스타일 클러스터에서 독립적으로 훈련되어 도메인 특유의 특성을 보존하도록 설계되었습니다. 이 프레임워크는 Adaptive Instance Normalization (AdaIN)을 사용하여 스타일과 콘텐츠의 잠재 표현을 통합하고 최종 합성 이미지를 생성합니다. 연구에서는 RSUIGM 데이터셋을 활용하여 다양한 Jerlov 수역 유형의 물리학적으로 정확한 스타일 변화를 확보하고, K-means 클러스터링을 통해 스타일을 기반으로 한 도메인으로 데이터를 분할하여 훈련합니다.

- **Performance Highlights**: 제안된 모델의 성능은 구조적 유사도 지수(SSIM) 0.9012, 평균 신호 대 잡음 비율(PSNR) 32.5118 dB, Frechet Inception Distance (FID) 13.3728로 최첨단 결과를 기록하였습니다. 이러한 성능 지표들은 본 모델이 높은 품질의 수중 이미지를 생성할 수 있음을 증명합니다. 결과적으로 DISC-GAN은 비균일한 수중 환경에서의 합성 작업에서 뛰어난 성능을 보입니다.



### Structured Spectral Graph Learning for Multi-label Abnormality Classification in 3D Chest CT Scans (https://arxiv.org/abs/2510.10779)
Comments:
          22 pages, 14 figures

- **What's New**: 이 논문에서는 3D Chest CT 스캔의 다중 라벨 분류 문제를 해결하기 위해 새로운 그래프 기반 프레임워크인 CT-SSG(Structural Spectral Graph for Computed Tomography)를 제안합니다. 이 방법은 3D CT 볼륨을 구조화된 그래프로 표현하여 슬라이스 간의 의존성을 효과적으로 처리합니다. 이 모델은 임상 배치에 적합하도록 복잡성을 유지하면서 강력한 성능을 보이는 것이 특징입니다.

- **Technical Details**: CT-SSG는 3D CT 볼륨을 세 개의 축 슬라이스 트리플렛으로 구성된 노드로 모델링하며, 이들 노드는 스펙트럴 그래프 합성을 통해 처리됩니다. 또한, 공간 인식을 명시적으로 포함하기 위해 축 방향 위치 임베딩과 z축을 따라 슬라이스 간 간격을 반영하는 엣지 가중치 전략을 도입했습니다. 이러한 접근은 CT 스캔의 전체적인 볼륨 구조와 세밀한 슬라이스 수준 정보를 통합하는 것입니다.

- **Performance Highlights**: CT-SSG는 독립적인 세 가지 데이터 세트에서 강력한 교차 데이터 세트 일반화를 달성했으며, 최신 비주얼 인코더들과 비교했을 때 경쟁력 있는 성능을 보입니다. 또한, 자동화된 방사선 보고서 생성 및 복부 CT 데이터로의 전이 실험을 통해 접근법의 광범위한 적용 가능성을 입증했습니다. 다양한 집합 전략, 엣지 가중치 체계 및 그래프 연결 패턴의 영향을 평가하기 위한 포괄적인 아블레이션 연구도 완료했습니다.



### EGD-YOLO: A Lightweight Multimodal Framework for Robust Drone-Bird Discrimination via Ghost-Enhanced YOLOv8n and EMA Attention under Adverse Condition (https://arxiv.org/abs/2510.10765)
- **What's New**: 이 논문은 드론과 새를 정확하게 식별하는 것이 공중 안전과 보안 시스템 개선에 매우 중요하다는 점을 강조합니다. VIP CUP 2025 데이터셋을 사용하여 EGD-YOLOv8n이라는 가볍지만 강력한 객체 탐지 모델을 제안합니다. 이 모델은 이미지 특성을 더욱 정교하게 캡처하고 이해함으로써 탐지의 정확성과 효율성을 높입니다.

- **Technical Details**: EGD-YOLOv8n 모델은 RGB 이미지, IR 이미지 및 두 가지를 결합한 세 가지 버전으로 훈련되었습니다. 이 모델은 중요 흥미 세부 사항에 집중하고 계산량을 줄이기 위해 똑똑한 디자인 변경과 attention layers를 사용하며, 다양한 형태와 크기의 객체에 적응할 수 있는 특별한 탐지 헤드를 채용했습니다. 모델 성능은 mAP와 F1-score를 기반으로 평가되었으며, 촬영 환경의 왜곡에 대한 견고성을 갖추고 있습니다.

- **Performance Highlights**: 제안된 모델은 결합된 이미지를 사용하여 최고의 정확성과 신뢰도를 달성하였고, 일반 GPU에서 실시간으로 사용할 수 있을 만큼 빠르게 작동하였습니다. 최종적으로, 45,000개의 훈련 이미지와 6,500개의 검증 이미지에 대한 평가에서 뛰어난 mAP와 추론 성능을 보이며 기존 기준을 초월한 성과를 거두었습니다. EGD-YOLOv8n은 안전하고 드론 밀집 지역에 적합한 멀티모달 UAV 감시의 발전을 이끌 것입니다.



### Restricted Receptive Fields for Face Verification (https://arxiv.org/abs/2510.10753)
- **What's New**: 이 논문에서는 심층 신경망의 결정을 이해하는 것이 중요하다는 점을 강조하며, 얼굴 인식을 위한 새로운 유사성 메트릭을 제안합니다. 기존의 포스트 호크(post-hoc) 방법들의 비판을 바탕으로, 이 연구는 모델의 결정 과정이 본래적으로 해석 가능하도록 설계하는 접근 방식을 채택합니다. 제안된 방법은 두 개의 얼굴 이미지 간의 유사성을 패치 수준의 점수의 합으로 정의하여 보다 직관적인 설명을 제공합니다.

- **Technical Details**: 제안된 접근 방식은 28x28 및 56x56 크기의 패치를 사용하여 112x112 크기의 얼굴 이미지에서 유사성을 측정하는 새로운 거리 메트릭을 활용합니다. 이 방법은 조금 수정된 ResNet 아키텍처를 사용하여 패치 수준의 유사성을 통해 면접 목적에 맞게 결정 과정을 명확하게 합니다. 따라서 포스트 호크 분석에 의존하지 않고, 보다 인간이 이해하기 쉬운 방식으로 유사성을 평가할 수 있게 됩니다.

- **Performance Highlights**: 논문에서는 56x56 패치를 사용할 때, 기존의 최첨단 방법보다 더 높은 검증 성능을 달성하는 것을 보여줍니다. 특히, 28x28 패치 또한 경쟁력 있는 성능을 발휘하며, 제한된 수용영역 내에서 검증 정확도를 향상시키는 놀라운 결과를 나타냅니다. 이로써 본 연구는 얼굴 인식에서의 해석 가능성을 높이는 새로운 가능성을 제시하고 있습니다.



### Uncovering Anomalous Events for Marine Environmental Monitoring via Visual Anomaly Detection (https://arxiv.org/abs/2510.10750)
- **What's New**: 이 연구에서는 해양 생물 다양성을 평가하기 위한 자동 비디오 모니터링을 위한 비주얼 이상 탐지(Visual Anomaly Detection, VAD)의 첫 번째 다중 주석자 벤치마크 데이터세트 AURA를 소개합니다. AURA 데이터세트를 통해 우리는 여러 VAD 모델을 평가하고, 흥미롭거나 이상한 사건을 자동으로 식별할 수 있는 가능성을 탐구합니다. 모델의 성능은 훈련 데이터의 양과 '정상' 장면을 정의하는 시각적 콘텐츠의 변동성에 크게 영향을 받음을 보여줍니다.

- **Technical Details**: 논문에서 제안하는 VAD는 깊이 신경망(Deep Neural Networks)을 기반으로 하며, 정상 데이터만을 이용하여 훈련됩니다. 연구팀은 두 개의 수중 장면에서 다양한 VAD 접근 방식을 평가할 뿐만 아니라, 다중 주석자 시스템을 통해 이벤트 경계의 주관성을 해결하고자 합니다. 이를 통해 수중 환경에서의 시각적 이상 탐지의 적용 가능성을 넓히고, 주관적인 요소를 감소시키는 방법을 모색합니다.

- **Performance Highlights**: 연구 결과는 AURA 데이터세트가 수중 이벤트 탐지에 대한 벤치마크로서 유용하다는 것을 나타냅니다. 또한, 모델 성능이 주석자 간의 차이에 따라 크게 달라진다는 점에서 다중 주석자의 필요성을 강조하고 있습니다. 이 연구는 해양 생물 다양성 모니터링의 과학적 탐사를 지원할 뿐만 아니라, 스케일러블한 접근 방식을 제공하는 가능성도 시사합니다.



### Seeing My Future: Predicting Situated Interaction Behavior in Virtual Reality (https://arxiv.org/abs/2510.10742)
Comments:
          Project Page: this https URL

- **What's New**: 이 연구에서는 사용자의 상호작용을 예측하여 반응하는 능동적 AR/VR 시스템을 만드는 데 기여하는 새로운 계층적 프레임워크를 제안합니다. 이 프레임워크는 인간의 의도를 이해하고, 세부적인 행동을 예측하는 데 초점이 맞춰져 있습니다. 특히, Dynamic Graph Convolutional Network (GCN)를 활용하여 인간과 환경 간의 상호작용을 보다 효과적으로 캡처합니다.

- **Technical Details**: 제안된 방법론은 역사적 관측값을 기반으로 사용자가 어디로 이동할지(trajectory), 어디를 바라볼지(gaze), 어떤 객체와 상호작용할지를 예측합니다. 이 구조는 사용자의 시선, 머리 자세, 손/신체 궤적과 같은 최신 AR/VR 장치에서 수집된 데이터를 활용합니다. 또한, 환경의 객체 상호작용은 시뮬레이션에서 직접 로그를 작성하거나 깊이 센서 및 IoT 기술을 통해 추론될 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크가 실제 VR 환경에서 사용자의 시선 패턴과 미래 행동을 성공적으로 예측할 수 있음을 보여주었습니다. 이 방법은 모든 메트릭에서 뛰어난 성과를 달성하며, 사용자의 행동을 예측하고 가상의 환경을 적절히 조정할 수 있는 실용적인 응용 프로그램 개발에 기여할 가능성을 제시합니다.



### WorldMirror: Universal 3D World Reconstruction with Any-Prior Prompting (https://arxiv.org/abs/2510.10726)
Comments:
          Project page, code, and models will be publicly available soon

- **What's New**: WorldMirror는 다목적 3D 기하학 예측 작업을 위해 설계된 통합 모델입니다. 기존의 이미지 기반 접근법의 한계를 넘어, 카메라 포즈, 내부 매개변수, 깊이 맵 등을 융합하여 여러 3D 표현을 효율적으로 생성합니다. 이를 통해 구조적 모호성을 해결하고, 사용자에게 일관된 3D 출력을 제공합니다.

- **Technical Details**: WorldMirror는 Multi-Modal Prior Prompting 메커니즘을 통해 다양한 기하학적 우선 정보를 통합합니다. 카메라 포즈와 내부 매개변수는 단일 토큰으로 인코딩되고, 깊이 맵은 공간 정보를 포함한 밀집 토큰으로 변환됩니다. 또한 훈련 중 동적 우선 주입 스킴을 통해 모형이 사용 가능한 다양한 우선 조합에 적응하도록 합니다.

- **Performance Highlights**: WorldMirror는 다양한 벤치마크에서 최첨단 성능을 보여주며, 기존의 3D 재구성 방법들을 초능가합니다. 카메라 및 깊이 예측에서 VGGT와 π3π를 초과하며, Surface Normal 예측에서 StableNormal을 초과하고, Novel View Synthesis 성능에서도 AnySplat을 능가합니다.



### Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos (https://arxiv.org/abs/2510.10691)
- **What's New**: 이 논문은 초점 흐림(defocus blur) 및 모션 흐림(motion blur) 모노큘러 비디오로부터 높은 품질의 동적 가우시안 스플래팅(dynamic Gaussian Splatting)을 가능하게 하는 통합 프레임워크를 제안합니다. 기존 방법들은 각각의 흐림 유형을 따로 처리하기 때문에 두 가지를 동시에 다루는 능력이 부족했습니다. 저자들은 흐림 예측 네트워크(blur prediction network)를 사용하여 픽셀 단위의 신뢰할 수 있는 흐림 커널을 추정함으로써 이 문제를 해결하고자 합니다.

- **Technical Details**: 이 연구에서는 흐림 커널 기반의 합성을 통해 두 가지 흐림 유형을 공동으로 모델링합니다. 흐림 예측 네트워크(BP-Net)는 장면 및 카메라 정보와 함께 흐림 인지 희소성 제약조건을 활용하여 신뢰할 수 있는 흐림 커널과 픽셀 단위의 강도를 예측합니다. 또한, 불완전한 영역에 대한 가우시안 부족 문제를 해결하기 위해 동적 가우시안 밀집화(dynamic Gaussian densification) 전략을 도입하였습니다.

- **Performance Highlights**: 저자들은 이 방법이 초점 흐림 및 모션 흐림이 있는 모노큘러 비디오로부터 포토리얼리스틱한 새로운 뷰 합성을 생성하는 데 기존 방법들보다 더 우수한 성능을 보인다고 주장합니다. 다양한 실험을 통해 그들의 방법이 상태-of-the-art 기법들보다 확연한 성능 개선을 이루었음을 입증하였습니다. 논문에서 제안한 코드와 훈련된 모델은 공개될 예정입니다.



### Action-Dynamics Modeling and Cross-Temporal Interaction for Online Action Understanding (https://arxiv.org/abs/2510.10682)
Comments:
          10 pages, 9 figures

- **What's New**: 본 논문에서는 액션 이해(action understanding) 문제를 해결하기 위한 새로운 프레임워크인 State-Specific Model (SSM)을 제안합니다. 이 모델은 액션 검출(action detection)과 액션 예측(action anticipation) 작업을 통합하고 향상시킬 수 있도록 설계되었습니다. 특히, 이 모델은 비디오 내의 중복 정보와 잡음을 감소시키고, 에이전트의 의도가 액션에 미치는 영향을 모델링합니다.

- **Technical Details**: SSM의 핵심 구성 요소로는 Critical State-Based Memory Compression (CSMC) 모듈이 있는데, 이는 원본 프레임 시퀀스를 중요한 상태로 압축하여 정보의 중복성을 최소화합니다. 또한, Action Pattern Learning (APL) 모듈은 다차원 관계를 활용하여 상태 전이 그래프(State-Transition Graph)를 구성하고, 이를 통해 액션 동역학(action dynamics)을 모델링합니다. Cross-Temporal Interaction (CTI) 모듈은 과거 및 현재 정보와 의도의 상호작용을 모델링하여 향후 액션 표현을 개선합니다.

- **Performance Highlights**: 포괄적인 실험을 통해, 여러 기준 데이터셋인 EPIC-Kitchens-100, THUMOS'14, TVSeries 및 Parkinson's Disease Mouse Behaviour (PDMB) 데이터셋에서 SSM의 우수한 성능이 입증되었습니다. 이 결과는 액션 동역학 학습과 시간 간 상호작용의 중요성을 강조하며, 향후 액션 이해 연구를 위한 기초를 마련합니다. SSM은 다양한 데이터셋에서 강력하고 일반화된 효과를 보여 다른 최신 기술들보다 더 나은 성능을 발휘합니다.



### MSM-Seg: A Modality-and-Slice Memory Framework with Category-Agnostic Prompting for Multi-Modal Brain Tumor Segmentation (https://arxiv.org/abs/2510.10679)
Comments:
          Under Review

- **What's New**: 본 연구에서는 다중 모드 뇌종양 분할을 위한 MSM-Seg 프레임워크를 제안합니다. 이 프레임워크는 다중 모드 및 절단 간 정보를 통합하여 범주에 구애받지 않는 효율적인 프로프트를 사용합니다. 이를 통해, 기존의 데이터에서 발생하는 문제들을 해결하고, 임상에서의 적용성을 확대할 수 있습니다.

- **Technical Details**: MSM-Seg는 모달리티 및 슬라이스 메모리 어텐션 (MSMA) 기술을 이용하여 서로 다른 MRI 스캔 간의 크로스 모달 및 절단 관계를 탐색합니다. 또한, 다중 스케일 카테고리 비구분 프롬프트 인코더(MCP-Encoder)와 모달리티 적응형 융합 디코더(MF-Decoder)를 설계하여 분할의 일관성을 높이고 예측 오류를 줄이는 데 기여합니다. 이 접근법은 다양한 MRI 데이터셋에서 향상된 분할 정확성을 보여줍니다.

- **Performance Highlights**: MSM-Seg는 다양한 MRI 데이터셋에서 대조군들과 비교했을 때 뛰어난 성능을 발휘하였습니다. 특히, 이 프레임워크는 다중 모드 전이 및 교차 모드 관계를 잘 포착하여 뇌종양의 하위 영역을 위한 종합적인 이해를 제공하는 데 중점을 둡니다. 우리는 이 방법이 최신 기술들보다도 뛰어난 성과를 이루었다는 것을 실험을 통해 입증하였습니다.



### Image-to-Video Transfer Learning based on Image-Language Foundation Models: A Comprehensive Survey (https://arxiv.org/abs/2510.10671)
Comments:
          Draft version, work in progress

- **What's New**: 이 연구는 이미지-언어 기초 모델(ILFM)을 비디오 도메인으로 확장하는 최근 동향을 종합적으로 다룹니다. 특히, 관찰된 이미지-비디오 전이 학습(image-to-video transfer learning)의 두 가지 주요 전략인 동결된 특징(frozen features)과 수정된 특징(modified features)을 분석하여, 이들이 비디오-텍스트 학습(video-text learning)에 미치는 영향을 고찰합니다. 이러한 새로운 구도는 비디오 이해 과제를 해결하는 데 기여할 수 있는 혁신적인 접근 방식을 제공합니다.

- **Technical Details**: 이 연구에서는 현대의 ILFM을 기반으로 하여 비디오 도메인에서의 효과적인 정보 처리 방법을 모색합니다. 이미지에서 비디오로의 전이 학습에서는 정적 공간 정보(spatial information) 처리에서 복잡한 시공간 관계(spatial-temporal relationships) 모델링으로의 전환이 필요합니다. 또한, 파라미터 효율적인 미세 조정(parameter-efficient fine-tuning) 기술과 같은 경량 모듈을 기존 구조에 추가하여, 동적 정보를 효과적으로 캡처하는 방안을 제시합니다.

- **Performance Highlights**: 연구의 실험적 분석 결과, 기존의 이미지-언어 기초 모델에서 비디오 도메인으로의 전이 학습이 가능하다는 것을 입증했습니다. 예를 들어 UniFormerV2 모델은 이미지-텍스트 모델에서 시작하여 우수한 성능을 발휘했으며, 이는 고유한 비디오 상황에 맞춘 사용자 지정 전이 전략(task-specific transfer techniques)의 유용성을 강조합니다. 이 연구는 또한 비디오 이해 과제에서의 다양한 모델 조합과 이들이 겪고 있는 도전 과제를 제시하며, 향후 연구 방향을 제안합니다.



### AdaViewPlanner: Adapting Video Diffusion Models for Viewpoint Planning in 4D Scenes (https://arxiv.org/abs/2510.10670)
- **What's New**: 최근 Text-to-Video (T2V) 모델이 현실 세계의 기하학과 물리 법칙을 시각적으로 시뮬레이션할 수 있는 강력한 능력을 보여주었습니다. 이에 착안하여, 우리는 주어진 4D 장면에서 비디오 생성 우선 순위를 활용하여 시점 계획(viewpoint planning)의 실현 가능성을 탐구하였습니다. 이를 위해 T2V 모델을 시점 예측을 위한 두 단계 패러다임으로 조정하는 방법을 제안합니다.

- **Technical Details**: 제안하는 방법은 먼저 4D 장면 표현을 pre-trained T2V 모델에 주입하여, 자동 학습 브랜치를 통해 시점 불가지론(viewpoint-agnostic)적인 4D 장면을 형성합니다. 이후, 생성된 비디오와 4D 장면을 입력으로 사용하여 카메라 외적 잡음 제거(camera extrinsic denoising) 과정을 통해 시점 추출을 공식화합니다. 이 두 단계 통합 방식을 통해, 입력된 4D 장면의 좌표계에 정렬된 카메라 포즈 시퀀스와 예측된 카메라 시점에서 4D 장면을 시각화하는 비디오를 얻을 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안한 방법이 기존 경쟁자들보다 큰 차이로 우수함을 보여주었습니다. 또한, 여러 ablation 연구를 통해 우리의 주요 기술 설계의 효과성을 검증하였습니다. 이 연구는 비디오 생성 모델이 4D 상호작용을 위한 가능성을 가지고 있다는 것을 입증합니다.



### Scalable Face Security Vision Foundation Model for Deepfake, Diffusion, and Spoofing Detection (https://arxiv.org/abs/2510.10663)
Comments:
          18 pages, 9 figures, project page: this https URL

- **What's New**: 이 논문은 풍부하고 레이블이 없는 실제 얼굴 데이터를 활용하여 얼굴 보안 작업에서의 일반화를 개선하기 위한 첫 번째 시도로, FS-VFM이라는 확장 가능한 self-supervised pre-training 프레임워크를 제안합니다. 우리는 3C라는 세 가지 학습 목표를 도입하여 masked image modeling (MIM)과 instance discrimination (ID)을 결합하였습니다. 이를 통해 FS-VFM은 실제 얼굴의 지역 패턴과 글로벌 의미를 인코딩할 수 있도록 합니다.

- **Technical Details**: FS-VFM은 다양한 얼굴 마스킹 전략을 통해 MIM을 위한 CRFR-P 마스킹을 구현합니다. 이 방법은 모델이 의미 있는 intra-region Consistency와 복잡한 inter-region Coherency를 추구하도록 명시적으로 유도합니다. 또한, 자가 증류(self-distillation) 메커니즘을 통해 MIM과 ID를 결합하여 지역-글로벌 Correspondence를 확립합니다.

- **Performance Highlights**: FS-VFM은 downstream Face Security 작업에 대해 vanilla vision transformers (ViTs)를 활용하여 cross-dataset deepfake 탐지, cross-domain 얼굴 위조 방지 및 보지 못한 확산 얼굴 법의학에서 우수한 성능을 보여줍니다. 또한, FS-Adapter를 통해 경량화된 plug-and-play 구조를 적용하여 효율적인 성능을 제공합니다. 11개의 공공 벤치마크에서 다양한 VFMs보다 일관되게 더 나은 일반화를 보였으며, 효율성과 성능 간의 우수한 무역을 성취했습니다.



### Stability Under Scrutiny: Benchmarking Representation Paradigms for Online HD Mapping (https://arxiv.org/abs/2510.10660)
- **What's New**: 이 논문은 자율주행의 기본 모듈 중 하나인 온라인 고해상도(HD) 맵의 시간적 안정성을 체계적으로 연구한 첫 번째 포괄적인 벤치마크를 제시하고 있습니다. 기존 온라인 맵 구축 모델들이 각 프레임의 맵핑 정확도 향상에만 초점을 두고 있는 반면, 맵핑 안정성에 대한 연구는 부족했습니다. 이를 보완하기 위해 Presence, Localization, Shape Stability를 포함하는 다차원 안정성 평가 프레임워크를 제안하고, 이를 단일 지표인 평균 안정성(mAS) 점수로 통합했습니다.

- **Technical Details**: 논문에서는 42개의 모델과 변형에 대한 광범위한 실험을 통해 정확도(mean Average Precision, mAP)와 안정성(mean Average Stability, mAS) 간의 독립적인 성능 차원을 발견했습니다. 또한 핵심 모델 설계 선택이 정확성과 안정성 모두에 미치는 영향을 분석하여, 높은 정확도와 높은 안정성의 기여 요인을 확인했습니다. 이러한 연구 결과는 자율주행 시스템의 신뢰성 향상에 중요한 기초자료가 됩니다.

- **Performance Highlights**: 이 연구의 가장 두드러진 점은 정확도와 안정성을 모두 고려해야 한다는 점을 강조하고, 이를 위해 공개 벤치마크를 출시할 계획임을 알린 것입니다. 새로운 평가 메트릭스를 통해 시간적 안정성을 핵심 평가 기준으로 삼아 자율주행의 발전을 촉진할 것으로 기대합니다. 벤치마크 도구, 코드 및 모델은 제공된 URL에서 공개될 예정입니다.



### A Machine Learning Perspective on Automated Driving Corner Cases (https://arxiv.org/abs/2510.10653)
- **What's New**: 본 논문에서는 자율주행의 필수 안전 운영을 위해, 데이터 배포(distribution)를 고려한 새로운 방법론을 제안합니다. 기존의 예제 기반 코너 케이스(corner case) 정의의 한계를 넘어서, 머신러닝 모델 훈련 데이터에 대한 일반화를 목표로 합니다. 새로운 시각을 바탕으로, 개별 샘플의 인식에 효과적인 코너 케이스 인식을 위한 프레임워크를 제공합니다.

- **Technical Details**: 본 연구는 코너 케이스를 두 가지 유형, 즉 의미적 코너 케이스(semantic CCs)와 공분산 코너 케이스(co-variate CCs)로 정의합니다. 이러한 코너 케이스는 각각 샘플의 특정 지역에 영향을 미치는 새로운 객체 인스턴스나 기후 변화와 같은 요소를 포함합니다. 또한, 데이터 분포를 반영한 인식을 위한 프레임워크를 제시하며, 새로운 패러다임을 통해 인식 성능을 향상시킵니다.

- **Performance Highlights**: 이 연구에서 제안하는 접근법은 여러 표준 벤치마크에서 강력한 성능을 보이며, 기존의 코너 케이스 분류 체계를 통합합니다. 이를 통해 새로운 'Foggy Lost and Found' 데이터셋에 대한 평가도 수행하며, 코너 케이스 인식을 위한 원칙적 기반을 마련합니다. 이 결과는 코너 케이스 인식에 있어 수동 정의가 필요하지 않음을 강조합니다.



### DEMO: Disentangled Motion Latent Flow Matching for Fine-Grained Controllable Talking Portrait Synthesis (https://arxiv.org/abs/2510.10650)
Comments:
          5 pages

- **What's New**: DEMO는 오디오 기반의 화법 비디오 합성을 위해 새로운 flow-matching 생성 프레임워크를 제안합니다. 이 프레임워크는 입술 움직임, 머리 자세 및 눈의 시선을 독립적으로 조작할 수 있도록 분리된 고품질 제어를 제공합니다. 특히, 움직임 자동 인코더를 기반으로 한 이 구조화된 잠재 공간에서 정확하고 논리적인 움직임 경로 생성을 가능케 합니다.

- **Technical Details**: DEMO의 주요 구성 요소는 이중 단계로 진행됩니다. 첫 번째 단계에서는 전처리된 움직임 자동 인코더를 통해 세밀하게 제어 가능한 얼굴 움직임 표현을 위한 잠재 공간을 구축하고, 두 번째 단계에서는 변환기 기반의 예측기를 사용하여 오디오 입력을 움직임 잠재와 매핑하는 최적 수송 흐름 정합을 적용합니다. 이를 통해 오디오에 조건화된 고충실도 비디오 프레임을 생성할 수 있습니다.

- **Performance Highlights**: DEMO는 비디오의 현실감, 입술과 오디오의 동기화, 움직임의 신뢰성에서 기존의 방법들을 크게 초월하는 성능을 보여주었습니다. 다양한 벤치마크에 대한 광범위한 실험을 통해 시각적 사실성과 움직임 정확성에서 최첨단 성과를 달성했으며, 정밀한 분리된 움직임 조작과 플로우 기반 생성 모델링의 결합이 새로운 컨트롤 가능한 화법 비디오 합성의 패러다임을 제공함을 입증했습니다.



### GraphTARIF: Linear Graph Transformer with Augmented Rank and Improved Focus (https://arxiv.org/abs/2510.10631)
- **What's New**: 이 논문에서는 기존의 선형 Attention 메커니즘이 가지는 표현력 저하 문제를 다루며, 이를 개선하기 위한 하이브리드 프레임워크인 GraphTARIF를 제안합니다. GraphTARIF는 값 행렬에 게이트가 있는 로컬 그래프 네트워크(branch)를 결합하여 Attention 맵의 랭크를 증가시키고, 학습 가능한 로그-파워 함수로 Attention 점수를 조절하여 엔트로피를 줄입니다. 이로 인해 모델의 분류 능력이 향상되어, 그래프 기반의 다양한 웹 관련 데이터셋에서 경쟁력을 가지게 됩니다.

- **Technical Details**: GraphTARIF 모델은 선형 그래프 변환기(linear Graph Transformer)로, 로컬 강화 모듈(local enhancement module)과 학습 가능한 로그-파워 함수를 통합하여 Attention 랭크를 높이고 엔트로피를 낮추는 방향으로 설계되었습니다. 이 방법은 특히 노드 레벨 작업에서의 표현 가능성을 개선하면서도 선형 Attention의 확장성(scalability)을 유지합니다. 논문은 이론적 분석을 통해 이러한 개선이 노드 간의 구별을 어떻게 향상시키는지를 명시합니다.

- **Performance Highlights**: GraphTARIF는 동질적(homophilic) 및 이종적(heterophilic) 그래프 학습 작업에서 우수한 성능을 보여주었습니다. 다양한 웹 관련 데이터셋에 대한 실험을 통해 기존 기준선과 비교하여 일관된 성능 향상을 입증했습니다. 특히, 선형 Self-Attention의 한계를 극복하면서도 효율성과 정확성을 동시에 달성하는 데 초점을 맞추었습니다.



### OmniQuality-R: Advancing Reward Models Through All-Encompassing Quality Assessmen (https://arxiv.org/abs/2510.10609)
- **What's New**: 본 논문에서는 다중 작업 품질 추론을 위한 연속적이고 해석 가능한 보상 신호로 변환하는 OmniQuality-R이라는 통합 보상 모델링 프레임워크를 제안합니다. 기존의 시각적 평가 방법들이 단일 작업에 제한되어 있었던 것을 극복하며, 다차원적 추론을 통해 보다 전반적인 품질 평가를 가능하게 합니다. 이 프레임워크는 다양한 품질 관련 작업을 지원하여 더 강력한 일반화를 도와줍니다.

- **Technical Details**: OmniQuality-R 프레임워크는 Implicit Question Analysis와 Explicit Reasoning Structures를 학습하기 위한 두 단계의 훈련 과정을 포함합니다. 첫 단계인 Cold-start Rejective Sampling Fine-Tuning 단계에서 informative plan-reason trajectories를 샘플링하여 체계적인 데이터셋을 구축합니다. 두 번째 단계에서는 Group Relative Policy Optimization (GRPO)을 사용해 변별력 있는 정책 최적화를 진행하며 Gaussian 기반 보상 함수를 도입해 연속적 점수 예측을 지원합니다.

- **Performance Highlights**: OmniQuality-R은 미적 품질 평가, 기술적 품질 평가 및 텍스트-이미지 정렬과 같은 세 가지 주요 IQA 작업에서 평가되었습니다. 향상된 일반화 성능을 보여주며 다양한 도메인에서 최적의 성능을 발휘합니다. 이 외에도 OmniQuality-R은 텍스트-이미지 생성 모델을 안내하고 개선하는 데 사용될 수 있는 테스트 시간 추론 모듈로 활용될 수 있습니다.



### ViSurf: Visual Supervised-and-Reinforcement Fine-Tuning for Large Vision-and-Language Models (https://arxiv.org/abs/2510.10606)
- **What's New**: 이번 논문에서는 Large Vision-and-Language Models (LVLMs)을 위한 새로운 포스트 트레이닝 패러다임인 ViSurf를 제안합니다. ViSurf는 Supervised Fine-Tuning (SFT)과 Reinforcement Learning with Verifiable Rewards (RLVR)의 장점을 통합하여 단일 단계에서 이를 수행하는 방식입니다. 기존 방법들의 한계를 분석하고, 각각의 오브젝티브를 통합하여 ViSurf 오브젝티브를 확립했습니다.

- **Technical Details**: ViSurf의 핵심은 RLVR 롤아웃에 실제 레이블을 주입함으로써 외부 감독과 내부 강화를 동시에 제공하는 것입니다. 이는 SFT와 RLVR의 오브젝티브와 기울기 분석을 통해 이루어졌으며, 이론적으로 두 가지 패러다임의 유사성을 보여줍니다. 또한, ViSurf의 안정적이고 최적화된 훈련 과정을 위해 세 가지 새로운 리워드 제어 전략이 도입되었습니다.

- **Performance Highlights**: ViSurf는 다양한 벤치마크에서 SFT와 RLVR, SFT → RLVR 방법보다 더 우수한 성능을 발휘했습니다. 또한, ViSurf로 훈련된 모델은 기존 연구에서 확립된 추론 능력을 나타내며, VQA 작업에서의 안정적인 성능을 통해 치명적인 망각 현상을 효과적으로 완화하였습니다. 이와 함께, 제안된 리워드 제어 메커니즘의 기여가 중요한 것으로 확인되었습니다.



### A Simple and Better Baseline for Visual Grounding (https://arxiv.org/abs/2510.10587)
Comments:
          ICME2025

- **What's New**: 이 논문에서는 'FSVG'라는 시각적 기초(feature selection-based) 모델을 제안하여 시각적 기초 과제에서의 언어적 의미의 전파를 개선합니다. 이 모델은 복잡한 반복 과정을 피하고, 언어와 시각적 모달리티를 동시에 사용하여 효율적으로 시각적 특징을 추출할 수 있도록 합니다. 특히, 언어와 관련된 시각적 특징만을 선택하여 계산 비용을 줄이는 새로운 메커니즘을 포함하고 있습니다.

- **Technical Details**: FSVG는 입력으로 RGB 이미지와 텍스트 설명을 결합하여 언어적 관련성을 가진 시각적 특징을 추출합니다. 이 모델은 기존의 직렬 구조가 아닌, 병렬 구조를 통해 언어적 특징이 전체 시각적으로 추출하는 과정에서 지속적으로 전파되도록 설계되어 있습니다. 또한, 고급 언어적 특징을 바탕으로 시각적 특징을 선택하여 효율성을 높이는 방식을 채택하고 있습니다.

- **Performance Highlights**: 실험 결과에 따르면, FSVG는 정확도와 효율성 사이의 더 나은 균형을 이룸으로써 기존 최첨단 방법들보다 뛰어난 성능을 보여주고 있습니다. 네 가지 주요 데이터셋을 기반으로 FSVG는 기존 방법들과 비교해 동등한 모델 파라미터를 유지하면서도 더 나은 결과를 이루어냈습니다. 본 연구의 결과는 컴퓨터 비전과 자연어 처리를 통합하는 발전에 기여할 것으로 기대됩니다.



### Equipping Vision Foundation Model with Mixture of Experts for Out-of-Distribution Detection (https://arxiv.org/abs/2510.10584)
- **What's New**: 이 연구에서는 사전 훈련된 비전 기초 모델들이 OOD 감지에 미치는 영향을 체계적으로 조사하였습니다. 특히 DINOv2 모델이 in-domain (ID) 데이터에 대한 파인튜닝 없이도 효과적인 OOD 감지를 위한 강력한 특징 공간(feature space)을 제공한다는 사실을 발견했습니다. 또한 Mixture of Feature Experts (MoFE) 모듈과 Dynamic-$\beta$ Mixup 전략을 도입하여 특징 학습을 향상시키고 복잡한 결정 경계(decision boundaries)를 정제하는 방법을 제안했습니다.

- **Technical Details**: 연구에서는 다양한 비전 기초 모델의 특징 공간을 분석하고, DINOv2가 다른 기초 모델보다 OOD 감지에 적합하다는 것을 입증합니다. MoFE 모듈은 특징을 하위 공간(subspace)으로 나누어 전문가 모델을 통해 각각 최적화하도록 설계되었습니다. 동적 베타 분포를 활용한 Dynamic-$\beta$ Mixup 기법은 카테고리의 학습 난이도에 따라 가중치를 조정하여, OOD 감지 성능을 보다 향상시킵니다.

- **Performance Highlights**: 광범위한 실험 결과, 제안한 모형이 여러 기존 방법들보다 현저히 우수한 성능을 보였음을 확인했습니다. 특히 DINOv2 모델은 단순한 KNN 메트릭만으로도 복잡한 방법들과 유사한 성능을 달성했습니다. OOD 검증 베이스라인에서 DINOv2와 MoFE의 조합이 기여한 점이 두드러지며, 연구는 OOD 감지 과제에서의 기초 모델 활용 가능성을 확장하는 방향으로 나아가고 있습니다.



### Injecting Frame-Event Complementary Fusion into Diffusion for Optical Flow in Challenging Scenes (https://arxiv.org/abs/2510.10577)
- **What's New**: 이번 논문에서는 고속 및 저조도 장면에서의 광학 흐름(optical flow) 추정의 문제를 해결하기 위해 새로운 프레임워크인 Diff-ABFlow를 제안합니다. 이 프레임워크는 프레임 카메라와 이벤트 카메라의 조합을 통해 형상(boundary) 정보와 시각적 외관(appearance) 정보를 결합하여 최적의 광학 흐름 추정을 수행합니다. 기존 방법들이 제약을 많이 받았던 시각적 특징의 저하를 극복하기 위해 확산 모델(diffusion model)을 활용해 노이즈가 있는 흐름(noisy flow)에서 깨끗한 흐름(clear flow)으로 매핑하는 방식을 도입했습니다.

- **Technical Details**: Diff-ABFlow 프레임워크는 Attention 기반 외관-경계 융합 모듈(Attention-ABF)과 다조건 반복 노이즈 감소 디코더(MC-IDD)를 포함하여 구성됩니다. Attention-ABF는 프레임 카메라와 이벤트 카메라의 외관-경계 상호 보완성을 활용하여 고품질의 융합 피쳐를 생성합니다. MC-IDD는 DDIM 구조를 기반으로 하여 시각적 특징, 운동 특징, 시간 임베딩을 결합하여 노이즈 감소 과정을 안내하는 혁신적인 광학 흐름 백본을 형성합니다.

- **Performance Highlights**: 광학 흐름 추정의 성능을 평가하기 위해 합성 및 실제 데이터셋에서 광범위한 실험을 수행하였으며, Diff-ABFlow가 고속 및 저조도 조건에서 모범적인 성능을 달성한다는 점을 입증했습니다. 기존 기법들보다 탁월한 성능을 보여주며, 강화된 일반화 능력과 강력한 내구성을 가지고 있습니다. 이는 고속 및 저조도 장면의 광학 흐름 추정 분야에 중요한 기여를 합니다.



### UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation (https://arxiv.org/abs/2510.10575)
- **What's New**: 이번 논문에서는 시각적 이해와 생성에 필요한 통일된 토크나이저, UniFlow를 제안합니다. 기존 토크나이저들의 한계를 극복하기 위해, UniFlow는 고수준의 의미 추상화와 저수준의 픽셀 복원 간의 갈등을 해결하며, 다양한 비주얼 인코더와 호환되는 구조를 갖추고 있습니다. 실험 결과, UniFlow는 뛰어난 이해 능력과 함께 고해상도 픽셀 복원을 가능하게 함을 보여줍니다.

- **Technical Details**: UniFlow는 레이어별 적응형 자기 증류(layer-wise adaptive self-distillation) 방법을 활용하여 사전 학습된 비전 기초 모델을 인코더로 사용합니다. 또한, 패치(patch) 기반의 픽셀 흐름(patch-wise pixel flow) 디코더를 도입하여, 조건부 흐름을 모델링함으로써 효과적인 픽셀 복원을 달성합니다. 이 과정에서, 고수준의 의미적 특징을 사용하여 디코더의 훈련 갈등을 완화하고 효율성을 높입니다.

- **Performance Highlights**: UniFlow는 13개의 도전 과제를 포함한 7개의 주요 작업에 대한 실험을 통해 강력한 성능을 입증하였습니다. 7B UniFlow-XL 모델은 14B TokenFlow-XL 모델을 7.75% 더 뛰어난 성과로 초과 달성하였으며, 픽셀 생성 및 복원 성능에서도 새로운 최첨단 결과를 달성하였습니다. 이러한 성과는 UniFlow가 시각적 이해와 생성을 모두 아우를 수 있는 다재다능한 모델임을 보여줍니다.



### Deep semi-supervised approach based on consistency regularization and similarity learning for weeds classification (https://arxiv.org/abs/2510.10573)
Comments:
          Submitted to EURASIP Journal on Image and Video Processing

- **What's New**: 이 논문에서는 잡초 종류 분류를 위한 새로운 딥 세미-슈퍼바이즈드 방법을 제안합니다. 이 방법은 consistency regularization과 similarity learning을 결합하여, 레이블링된 데이터가 부족한 상황에서도 효과적이고 강력한 잡초 인식을 가능케 합니다. 또한, ConvNeXt 인코더를 기반으로 한 오토 인코더 아키텍처를 통해 레이블이 없는 데이터의 유용성을 극대화합니다.

- **Technical Details**: 제안된 방법은 DeepWeeds 데이터 세트에서 실험을 통해 검증되었습니다. 이는 자동화된 시스템의 자율적인 결과를 도출하는 데 중요한 이점을 제공합니다. 고전적인 머신 러닝 기술과 달리, 딥 러닝 모델은 데이터에서 자동으로 중요한 특징을 추출하는 강력한 능력을 가지고 있어, 데이터 라벨링의 시간 소모적인 과정에서 해방될 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 최신의 슈퍼바이즈드 딥 러닝 모델들과 비교했을 때 우수한 성능을 보여주었습니다. 특히, 잡초 식별 문제에서 제안된 방법이 효과적임을 입증하였으며, 레이블 데이터가 제한된 조건에서도 높은 분류 정확성을 달성했습니다. 이 연구는 농업 분야에서 신뢰성 있는 잡초 관리 시스템 구축에 기여할 것으로 기대됩니다.



### MRS-YOLO Railroad Transmission Line Foreign Object Detection Based on Improved YOLO11 and Channel Pruning (https://arxiv.org/abs/2510.10553)
- **What's New**: 본 연구에서는 철도 환경에서 전송선 외부 물체 감지의 문제점을 해결하기 위해 YOLO11을 기반으로 한 개선된 알고리즘 MRS-YOLO를 제안했습니다. 특히, 다양한 크기와 형태의 외부 물체에 대한 기능 추출 능력을 강화하기 위한 다중 스케일 적응형 커널 깊이 특징 융합(MAKDF) 모듈을 도입했습니다. 이를 통해 감지 효율성과 정확도를 향상시킬 수 있는 기반을 마련하였습니다.

- **Technical Details**: MRS-YOLO 알고리즘은 C3k2_MAKDF와 같은 새로운 모듈을 포함하여 특징을 보다 효과적으로 통합하고 활용할 수 있도록 돕는 Re-calibration Feature Fusion Pyramid Network (RCFPN)를 설계하였습니다. 또한, 공간 및 채널 재구성 감지 헤드(SC_Detect)를 도입하여 모델의 전반적인 감지 성능을 향상시켰습니다. 마지막으로, 채널 가지치기(channel pruning) 기술을 활용하여 모델의 중복성을 줄이고, 전체 파라미터 수와 GFLOPs를 감소시켰습니다.

- **Performance Highlights**: 실험 결과, MRS-YOLO 알고리즘의 mAP50은 94.8%, mAP50:95는 86.4%로 개선되어 기준선 대비 각각 0.7 및 2.3 퍼센트 포인트 상승했습니다. 또한, 전체 파라미터와 GFLOPs는 각각 44.2% 및 17.5% 감소하여 감지 효율성이 크게 향상되었습니다. 이러한 결과는 개선된 알고리즘이 철도 전송선의 외부 물체 감지 작업에 더욱 잘 적용될 수 있음을 증명합니다.



### GLOFNet -- A Multimodal Dataset for GLOF Monitoring and Prediction (https://arxiv.org/abs/2510.10546)
- **What's New**: 본 논문은 Glacial Lake Outburst Floods (GLOF) 예측을 위한 새로운 다중모드 데이터셋인 GLOFNet을 소개합니다. GLOFNet은 카라코람의 Shisper Glacier를 중심으로 구성된 데이터셋으로, 시각적 지표와 물리적 전조를 통합하여 예측 연구에 필요한 조화로운 데이터를 제공합니다. 기존 연구가 사건 후 매핑에 초점을 맞춘 것과 달리, 이 데이터셋은 예측에 최적화된 정보를 제공합니다.

- **Technical Details**: GLOFNet은 Sentinel-2 멀티스펙트럴 이미지, NASA ITS_LIVE 속도 제품, MODIS 지표면 온도 기록 등 세 가지 보완적인 자료원을 통합하여 구성됩니다. 데이터 전처리에는 구름 마스킹(cloud masking), 품질 필터링(quality filtering), 정규화(normalization), 시간 보간(temporal interpolation), 증강(augmentation), 주기적 인코딩(cyclical encoding) 등이 포함되어 있습니다. 이러한 과정은 다중 모드 데이터셋의 조화를 이루기 위한 것입니다.

- **Performance Highlights**: 탐색적 분석 결과, 계절 별 빙하 속도 주기(seasonal glacier velocity cycles), 약 0.8 K의 장기 온난화(long-term warming), 그리고 빙권 조건의 공간적 이질성(spatial heterogeneity)이 발견되었습니다. GLOFNet은 클라우드 오염(cloud contamination), 클래스 불균형(class imbalance), 해상도(coarse resolution) 등의 문제를 해결하여 희귀 재해 예측을 위한 다중모드 딥러닝(deep learning) 접근 방식을 위한 견고한 기반을 제공합니다.



### MCE: Towards a General Framework for Handling Missing Modalities under Imbalanced Missing Rates (https://arxiv.org/abs/2510.10534)
Comments:
          This is the accepted version of an article that has been published in \textbf{Pattern Recognition}. The final published version will be available soon

- **What's New**: 이번 논문에서는 다중 모달 학습(multi-modal learning)의 새로운 접근 방식인 Modality Capability Enhancement (MCE)를 제안합니다. MCE는 학습 능력 향상(Learning Capability Enhancement, LCE)과 표현 능력 향상(Representation Capability Enhancement, RCE)이라는 두 가지 상호 보완적인 요소를 포함하여 비대칭적인 결측 모달리티 문제를 해결하고자 합니다. 기존의 연구는 데이터셋 수준에서의 균형에 집중했던 반면, MCE는 샘플 수준의 변동성을 고려하여 모달리티의 유틸리티를 최적화합니다.

- **Technical Details**: MCE 프레임워크는 두 가지 주요 구성 요소로 구성됩니다. 첫째, LCE는 모달리티의 학습 진전을 동적으로 균형 있게 조정하며, 글로벌 모달리티 가용성과 현재 모달리티의 학습 상태를 평가하여 차별화된 인센티브를 부여합니다. 둘째, RCE는 서브셋 예측(subset prediction)과 교차 모달 완성(cross-modal completion) 작업을 통해 특성 품질을 향상시킵니다.

- **Performance Highlights**: 다양한 멀티 모달 벤치마크에서 실시된 포괄적인 평가 결과, MCE는 다양한 결측 구성 환경에서 최첨단 기술(state-of-the-art)보다 일관되게 성능이 뛰어난 것으로 나타났습니다. 본 연구는 다중 모달 학습이 직면한 도전과제를 직시하고, 이를 해결하기 위한 원리적이고 실용적인 솔루션을 제시하였습니다.



### Layout-Independent License Plate Recognition via Integrated Vision and Language Models (https://arxiv.org/abs/2510.10533)
- **What's New**: 이번 연구는 다양한 번호판 레이아웃과 까다로운 실제 조건에서 안정적으로 작동할 수 있는 패턴 인식 프레임워크를 제시합니다. 제안된 시스템은 현대적이고 고정밀의 detection network와 transformer 기반의 vision model을 통합한 인식 단계를 포함하고 있습니다. 이 접근 방식은 명시적인 휴리스틱 수정이나 수동 레이아웃 분류에 의존하지 않고 번호판에 특화된 구조적 패턴과 포맷 규칙을 학습합니다.

- **Technical Details**: 시스템의 인식 단계는 문자 식별(character identification)과 post-OCR refinement를 통합하여 원활한 프로세스를 수행합니다. 이를 통해 시각적 및 언어적 단서를 공동으로 최적화하며, 소음(noise), 왜곡(distortion), 비정상적인 글꼴(unconventional fonts)에도 불구하고 OCR 정확도를 향상시키는 반복적(refinement) 개선이 가능합니다. 실험은 다양한 국제 데이터셋(IR-LPR, UFPR-ALPR, AOLP)에서 레이아웃 독립적인 인식을 달성함을 보여줍니다.

- **Performance Highlights**: 실험 결과는 최근의 segmentation-free 접근 방식에 비해 뛰어난 정확도와 견고성을 보여주며, 인식 단계 내에서 패턴 분석을 포함시켜 컴퓨터 비전(computer vision)과 언어 모델링(language modelling) 간의 간극을 메울 수 있음을 강조합니다. 이 연구는 지능형 교통과 감시 애플리케이션에서의 적응성을 높이는 데 기여할 것으로 기대됩니다.



### Unified Open-World Segmentation with Multi-Modal Prompts (https://arxiv.org/abs/2510.10524)
Comments:
          Accepted to ICCV2025

- **What's New**: COSINE은 통합된 오픈 월드 세그멘테이션 모델로, 오픈 어휘(open-vocabulary) 세그멘테이션과 컨텍스트(context) 세그멘테이션을 멀티 모달 프롬프트(multi-modal prompts)로 통합합니다. 이 모델은 입력 이미지와 멀티 모달 프롬프트의 표현을 추출하고 정렬하여 다양한 세분화(mask)를 생성합니다. COSINE은 이전의 다양한 아키텍처적 차이와 학습 목표의 불일치를 극복하여 오픈 월드 세그멘테이션의 가능성을 극대화합니다.

- **Technical Details**: COSINE은 여러 파운데이션 모델(foundation models)로 구성된 모델 풀(Model Pool)을 활용하여 이미지와 프롬프트의 표현을 추출합니다. 이 표준화된 입력 형식은 구조적 통합을 촉진하며, SegDecoder라는 단일 디코더 전용 세그멘테이션 모델이 이를 공동으로 처리할 수 있게 합니다. SegDecoder는 이미지와 여러 프롬프트를 정렬하는 Image-Prompt Aligner 모듈과 객체 쿼리 간의 상호작용을 모델링하는 Multi-Modality Decoder를 포함합니다.

- **Performance Highlights**: COSINE은 오픈 어휘 및 컨텍스트 세그멘테이션 작업에서 모두 뛰어난 성능 향상을 보여주었습니다. 실험 결과는 разные modality branches 간의 시너지 협력이 단일 모달 접근 방식에 비해 일반화 성능을 크게 향상시킨다는 것을 강조합니다. 이 연구는 연구 커뮤니티에 중요한 통찰을 제공하며, 오픈 월드 세그멘테이션 분야의 발전을 이끌 것으로 기대됩니다.



### Receptive Field Expanded Look-Up Tables for Vision Inference: Advancing from Low-level to High-level Tasks (https://arxiv.org/abs/2510.10522)
- **What's New**: 이 연구에서는 기존의 LUT(look-up table) 방법의 한계를 극복하고 CNN(Convolutional Neural Network)의 수용 영역(receptive field)을 확장하는 새로운 방법을 제시합니다. 기존 LUT는 수용 영역 크기에 따라 테이블 크기가 기하급수적으로 증가하는 문제를 가지고 있었으나, 본문에서는 고정된 테이블 크기를 유지하면서 성능을 향상시킬 수 있는 다양한 기술을 제안합니다. 핵심 기여는 데이터의 중요도에 따라 양자화 해상도를 적응적으로 할당하는 최적 격자 벡터 양자화(lattice vector quantization) 학습 방법입니다.

- **Technical Details**: 제안된 방법은 입력 데이터 벡터를 최적의 VQ(codewords)로 근사하는 것을 목표로 하고 있으며, conventional free-form VQ 방식의 단점을 피하기 위해 LVQ를 활용합니다. LVQ는 격자 포인트가 보다 효율적으로 벡터 공간을 커버할 수 있도록 하며, 정규성을 통해 테이블 조회 작업을 보다 간단하고 빠르게 수행할 수 있습니다. 또한, 불규칙 확장 합성곱(irregular dilated convolutions)과 U자 형태로 연결된 LUT 구조를 통해 수용 영역을 더욱 확장할 수 있는 기술도 함께 제시됩니다.

- **Performance Highlights**: 실험을 통해 제안된 RFE-LUT 방법이 고급 비전 문제인 이미지 분할 작업에서도 우수한 성능을 발휘하며, 저급 이미지 초해상도 작업에서 기존 LUT 방법에 비해 현저한 성능 개선을 보여줍니다. 또한, 메모리와 계산 자원이 제한된 상황에서도 속도와 정확도, 메모리 효율성을 효과적으로 균형 잡는 것을 입증하였습니다. 총체적으로 LVQ의 최적 설계가 LUT 기반 추론을 저급 비전에서 고급 비전 작업으로 확장하는 데 중요한 역할을 한다는 것을 보여주고 있습니다.



### VR-Thinker: Boosting Video Reward Models through Thinking-with-Image Reasoning (https://arxiv.org/abs/2510.10518)
- **What's New**: VR-Thinker는 새로운 thinking-with-image 프레임워크를 사용하여 비디오 선호를 평가하는 멀티모달 보상 모델(RM)을 제시합니다. 이 모델은 시각적 추론 작업을 통합하여 복잡한 데이터를 효율적으로 처리할 수 있도록 설계되었습니다. 특히, VR-Thinker는 사용자 정의 가능한 시각적 메모리 윈도우와 프레임 선택 기능을 통해 더욱 정확하고 신뢰할 수 있는 추론을 가능하게 합니다.

- **Technical Details**: VR-Thinker는 세 단계의 훈련 파이프라인으로 구성됩니다. 첫 번째 단계에서는 Cold Start를 통해 기본적인 텍스트 추론 기술을 학습합니다. 두 번째 단계인 Rejection Sampling Fine-Tuning은 고품질 추론을 강화하기 위해 올바른 판단을 가진 샘플만을 선택하여 모델을 훈련합니다. 마지막으로, Group Relative Policy Optimization(GRPO)을 적용하여 시각적 세부 정보 탐색과 고품질 추론을 학습하게 합니다.

- **Performance Highlights**: VR-Thinker는 열린 소스 모델 중에서 비디오 선호 벤치마크에서 최첨단 정확도를 달성했습니다. 특히, 7B 모델은 VideoGen Reward에서 80.5%, GenAI-Bench에서 82.3%, MJ-Bench-Video에서 75.6%의 성능을 보였습니다. 이러한 결과는 멀티모달 보상 모델에서 시각적 추론의 중요성을 입증하며, 더욱 향상된 사용성과 신뢰성을 보여줍니다.



### Jigsaw3D: Disentangled 3D Style Transfer via Patch Shuffling and Masking (https://arxiv.org/abs/2510.10497)
Comments:
          23 pages, 16 figures and 1 table

- **What's New**: Jigsaw3D는 3D 스타일 전송의 새로운 접근법으로, 스타일과 내용을 분리하여 빠르고 뷰 일관성이 있는 스타일화를 가능하게 합니다. 참조 패치 (reference patches)의 공간 재배치 및 무작위 마스킹을 활용하여 객체의 의미를 억제하고 스타일 통계만을 분리하는 독창적인 방법을 소개합니다. 이 방식은 Jigsaw3D가 기존 방법들보다 월등한 성능을 나타낼 수 있게 합니다.

- **Technical Details**: Jigsaw3D는 다중 뷰 확산(diffusion) 기반 파이프라인으로, ‘jigsaw transform’이라는 기법을 사용하여 스타일 정보를 보존하면서 전반적인 구조를 파괴합니다. 이 시스템은 조건부 스타일 레퍼런스와 기하학(geometry) 정보를 통합하여 스타일 변화가 가능한 특징 재조합을 수행합니다. 주목해야 할 점은 스타일 정보는 'reference-attention' 모듈에 의해 동적으로 주입되며, 이는 고유의 스타일 통계를 유지할 수 있게 합니다.

- **Performance Highlights**: Jigsaw3D는 3D 스타일 벤치마크에서 뛰어난 스타일 충실도와 다중 뷰 일관성을 달성하였으며, 자산별 최적화 없이도 작동합니다. 이 방법은 부분 스타일화(partial stylization), 다중 객체 장면 스타일링(multi-object scene styling), 그리고 타일 가능한 텍스처 생성(tiled texture generation)에도 일반화할 수 있는 능력을 입증했습니다.



### Head-wise Adaptive Rotary Positional Encoding for Fine-Grained Image Generation (https://arxiv.org/abs/2510.10489)
- **What's New**: 이 논문은 ROTARY POSITION EMBEDDING (RoPE)의 한계를 지적하고, 새로운 HARoPE 방법론을 제안합니다. HARoPE는 특화된 헤드별 적응형 회전 위치 인코딩 메커니즘으로, 상대위치 특성을 유지하면서도 기존 RoPE의 주요 한계를 해결합니다. 특히, 이 연구는 공간 데이터를 모델링하는 데 있어 필요한 세부적인 구조적 편향을 캡처하는 데 도움이 됩니다.

- **Technical Details**: HARoPE는 SINGULAR VALUE DECOMPOSITION (SVD)을 통해 매핑 전 학습 가능한 선형 변환을 삽입하여 기능합니다. 이를 통해 각 주의 헤드에 대해 독립적인 수신 필드를 부여하고, 이를 통해 다차원 데이터의 동적 주파수 재분배 및 의미적 정렬을 구현할 수 있습니다. 이 방법은 빈틈없이 다차원 회전과 상호작용을 촉진하며, 주의가 상대적 차이에만 의존하도록 합니다.

- **Performance Highlights**: HARoPE는 ImageNet 데이터셋과 텍스트-이미지 생성 모델인 Flux 및 MMDiT에서 기존의 RoPE 및 여러 확장 방식보다 성능이 뛰어난 결과를 나타냅니다. 이는 HARoPE가 이미지 생성 모델에서 포지셔널 인식(Pozitional Awareness)을 향상시키는 효과적인 솔루션임을 보여줍니다. 이 연구는 HARoPE가 대규모 텍스트-이미지 생성 아키텍처와 잘 보완될 수 있음을 입증하였습니다.



### Towards Self-Refinement of Vision-Language Models with Triangular Consistency (https://arxiv.org/abs/2510.10487)
- **What's New**: 이 논문은 외부 감독 없이도 비전-언어 모델(VLM)의 자가 개선(self-refinement) 능력을 검증합니다. 제안된 프레임워크는 Triangular Consistency 원리를 바탕으로 하여, VLM이 스스로 고품질의 지도 데이터를 생성할 수 있도록 돕습니다. 특히, 이미지-질문-답변 트리플릿을 생성하고 이를 통해 지속적으로 모델의 성능을 개선하는 방법을 보여줍니다.

- **Technical Details**: 프레임워크는 세 단계로 구성되며, 첫 번째 단계에서는 VLM의 지시 사항 생성 능력을 향상시키기 위해 멀티태스크 지시 조정(multi-task instruction tuning)을 추가합니다. 두 번째 단계에서는 비표시 데이터에서 생성된 이미지-질문-답변 트리플릿을 Triangular Consistency 원리를 적용하여 필터링합니다. 마지막 단계에서는 필터링된 합성 데이터를 이용해 모델을 추가로 업데이트합니다.

- **Performance Highlights**: 실험 결과, 자가 생성된 지시 사항들이 VLM을 개선할 수 있음을 다양한 벤치마크를 통해 입증하였습니다. LLaVA-1.5 모델을 기준으로 하여, 외부 감독 없이도 일관되게 성능 개선을 이루어냈습니다. 이 연구는 VLM의 학습 메커니즘에 대한 통찰력을 제공하고 향후 연구를 위한 바탕을 마련할 것으로 기대합니다.



### MSF-Mamba: Motion-aware State Fusion Mamba for Efficient Micro-Gesture Recognition (https://arxiv.org/abs/2510.10478)
- **What's New**: 이번 논문에서는 마이크로 제스처 인식(MGR)을 위한 새로운 모델인 Motion-aware State Fusion Mamba (MSF-Mamba)를 제안합니다. MSF-Mamba는 기존의 Mamba 모델에 로컬 시공간 의존성을 통합하여 더욱 정교하게 미세한 손동작을 인식할 수 있도록 설계되었습니다. 이 모델은 고유한 모션 인식 기능을 포함해 정보의 선형적 수집 방식을 유지하면서도 로컬 및 글로벌 시공간 모델링을 지원하는 장점이 있습니다.

- **Technical Details**: MSF-Mamba는 모션 인지 상태 융합(module)을 도입하여 중앙 프레임 차이(Central Frame Difference, CFD)를 기반으로 하여 로컬 시공간 정보를 활용합니다. 논문의 또 다른 변종인 MSF-Mamba+는 다중 스케일 모션 인지 융합을 지원하며, 동적인 스케일 가중치 모듈(adaptive scale weighting module)을 통해 통합된 상태를 가중치에 따라 조정할 수 있는 기능을 제공합니다. 이로 인해 두 모델 모두 MGR에서 중요한 미세한 동작 신호를 효과적으로 캡처할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과, 두 공개 MGR 데이터셋을 기반으로 한 MSF-Mamba는 최신 기술(State-of-the-Art, SoTA)을 달성하였으며, 기존 CNN, Transformer 및 SSM 모델들과 비교할 때 높은 효율성을 유지하며 성능에서 월등한 결과를 보였습니다. 예를 들어, MSF-Mamba+는 SMG 및 iMiGUE 데이터셋에서 baseline 모델인 VideoMamba에 비해 각각 2.9%와 3.0%의 정확도 향상을 이뤄냈습니다.



### DAGLFNet:Deep Attention-Guided Global-Local Feature Fusion for Pseudo-Image Point Cloud Segmentation (https://arxiv.org/abs/2510.10471)
- **What's New**: 이 논문에서는 LiDAR 기반 포인트 클라우드(3D point cloud)의 의미 분할(semantic segmentation)을 위한 새로운 프레임워크인 DAGLFNet을 제안합니다. 이 구조는 포인트 클라우드의 무질서한 데이터를 정형화된 2D 이미지 표현으로 변환하면서도 중요한 구별 가능한 특징을 포착할 수 있도록 설계되었습니다. 또한, 이 프레임워크는 글로벌-로컬 특징 융합, 다중 분기 특징 추출, 깊이 특징 유도 주의 메커니즘을 통합하여 정확성과 계산 효율성을 모두 강화합니다.

- **Technical Details**: DAGLFNet 프레임워크는 이미지 표현으로의 매핑 과정에서 겪는 경계 특징의 흐릿함 문제를 해결하기 위해, GL-FFE(글로벌-로컬 특징 융합 인코딩) 모듈과 MB-FE(다중 분기 특징 추출) 네트워크를 활용합니다. GL-FFE 모듈은 전역 및 지역의 기하학적 관계를 캡처하고, MB-FE 네트워크는 경계 특징의 표현력을 강화하여 수용장(receptive field)을 확장하는 데 중점을 둡니다. 또한, FFDFA(특징 융합을 통한 깊이 특징 유도 주의) 전략을 도입하여 특징 통합 단계에서 거리 정보를 반영하여 정확한 교차 채널(feature) 융합을 개선합니다.

- **Performance Highlights**: DAGLFNet은 SemanticKITTI 및 nuScenes의 검증 세트에서 각각 69.83% 및 78.65%의 평균 교차 합집합(mIoU) 점수를 달성하며 뛰어난 성능을 입증했습니다. 이 프레임워크는 임베디드 플랫폼에서도 성공적으로 배포될 수 있어 실시간 의미 분할(real-time semantic segmentation)이 가능하다는 잠재력을 보여줍니다. 이에 따라 실시간 LiDAR 기반 응용 프로그램에서의 활용 가능성이 크게 확대되었습니다.



### When Images Speak Louder: Mitigating Language Bias-induced Hallucinations in VLMs through Cross-Modal Guidanc (https://arxiv.org/abs/2510.10466)
- **What's New**: 이번 논문에서는 Vision-Language Models(VLMs)의 환각(hallucination) 문제를 해결하기 위해 Cross-Modal Guidance(CMG)라는 새로운 훈련 없는 추론(inference) 알고리즘을 제안합니다. CMG는 특정 transformer 기반 디코더 레이어에서 주의(attention) 가중치를 랜덤으로 마스킹하여 시각-언어 인식을 손상시키는 방법을 사용합니다. 이를 통해 언어 편향을 줄이고 시각적 맥락에 대한 인식을 강조하며, VLM들의 성능을 높일 수 있습니다.

- **Technical Details**: CMG는 기존의 VLM에서 발생하는 언어 편향 문제를 해결하기 위해, 마스킹한 주의 가중치를 활용하여 출력 로짓(logit) 값을 보정하는 방식으로 작동합니다. 이 과정에서 생성된 출력의 확률 분포를 조정하고, 모델이 시각 정보를 좀 더 잘 반영하도록 유도합니다. CMG는 VCD와 ConVis와 같은 기존 방법들과 달리 추가적인 훈련 없이도 효과적으로 VLM의 환각 문제를 해결할 수 있는 방법을 제공합니다.

- **Performance Highlights**: 실험 결과, CMG는 다양한 환각 전용 벤치마크에서 성능을 향상시켰으며, POPE와 HallusionBench 벤치마크에서 VCD 및 ConVis에 비해 뛰어난 결과를 보였습니다. MME 벤치마크에서 CMG는 VCD보다 13.54%의 성능 향상을 보여주었고, POPE 벤치마크에서 LLaVA-v1.5-7B 모델은 85.48%의 전반적인 정확도를 기록했습니다. CMG는 훈련 없는 추론 접근 방식 중에서도 두드러진 성과를 나타내며, VLM의 환각 문제 해결에 기여하는 것으로 평가됩니다.



### Post-TIPS Prediction via Multimodal Interaction: A Multi-Center Dataset and Framework for Survival, Complication, and Portal Pressure Assessmen (https://arxiv.org/abs/2510.10464)
Comments:
          81 pages, 13 figures

- **What's New**: Transjugular intrahepatic portosystemic shunt (TIPS)는 portal hypertension (PH)에 대한 효과적인 치료 방법이지만, 생존 예측에서 변동성이 크고 간성 혼수(Overt Hepatic Encephalopathy, OHE)가 빈번하게 발생합니다. 이를 해결하기 위해 제안된 MultiTIPS는 처음으로 공개된 다중 센터 데이터 세트로, TIPS의 예후를 예측하기 위한 새로운 다중 모드(prognostic framework)를 기반으로 합니다. 이 모델은 세 가지 핵심 모듈로 구성되어 있습니다.

- **Technical Details**: 첫 번째 모듈인 dual-option segmentation은 반지도 학습(semi-supervised learning) 및 기초 모델(foundation model) 기반 파이프라인을 통합하여 제한된 주석(annotation)으로 견고한 ROI(segment) 분할을 달성합니다. 두 번째 모듈인 multimodal interaction은 다중 요소의 상호작용을 통해 예측 모델의 정확성과 강인성을 향상시키는 다수의 기술(Multi-grained Radiomics Attention, Progressive Orthogonal Disentanglement, Clinically Guided Prognostic Enhancement)을 포함합니다. 세 번째 모듈인 multi-task prediction은 생존 예측, portal pressure gradient (PPG) 예측, OHE 예측을 동시에 수행하여 포괄적인 예후 평가를 가능하게 합니다.

- **Performance Highlights**: 실험 결과, MultiTIPS는 최신 기술들보다 우수한 성능을 보이며, 강력한 도메인 간 일반화(cross-domain generalization) 및 해석 가능성을 입증했습니다. 이 모델은 TIPS 예후 예측에서 다중 작업 처리(multi-task handling) 및 전 범위 예측을 지원하며, 기존 단일 임무 모델보다 월등한 결과를 도출합니다. 데이터 세트와 코드가 공개되었으므로, 향후 임상 적용에 대한 기대를 높이고 있습니다.



### Learning from Disagreement: A Group Decision Simulation Framework for Robust Medical Image Segmentation (https://arxiv.org/abs/2510.10462)
- **What's New**: 이 논문에서는 의료 이미징 세분화에서의 평가자 간 변동성(inter-rater variability, IRV)을 다루기 위해 새로운 방법론인 그룹 결정 시뮬레이션 프레임워크(Group Decision Simulation framework)를 소개합니다. 이 방법은 임상 패널의 협업적 의사결정 과정을 모방하여 전문가의 불일치를 귀중한 신호로 활용합니다. 특히, 전문가의 스타일을 개별 latent space에서 표현할 수 있는 Expert Signature Generator (ESG)를 활용하여 AI 시스템이 더 견고하고 신뢰할 수 있게 만드는 방법론을 개발하였습니다.

- **Technical Details**: 제안된 GDS 프레임워크는 Pyramid Vision Transformer (PVT)와 변이 오토인코더(Variational autoencoder)를 기반으로 합니다. 전문가 서명 생성기(ESG)는 주어진 이미지에 대한 주석 스타일을 모델링하기 위한 latent space를 구축하며, 시뮬레이션된 상담 모듈(SCM)은 이 latent space에서 다양한 전문가 서명을 샘플링하여 최종 세분화를 생성합니다. 이 과정에서 주석 집합의 통계적 분포를 고려하여 각 전문가의 개별 스타일을 뚜렷하게 식별하는 구조로 설계되었습니다.

- **Performance Highlights**: 본 방법은 CBCT와 MRI 데이터세트에서 각각 92.11%와 90.72%의 Dice 점수를 기록하며 최신 기술을 초월하는 성능을 입증하였습니다. 기계 학습 모델은 고차원에서의 전문가 간 합의뿐만 아니라 애매한 영역에서도 뛰어난 성능을 발휘하여 임상 분야에서의 적용 가능성을 높였습니다. 이러한 접근은 전문가의 불일치를 단순한 노이즈가 아니라 유용한 신호로 인식함으로써 진단에 있어 더 나은 신뢰성을 제공합니다.



### On the Problem of Consistent Anomalies in Zero-Shot Industrial Anomaly Detection (https://arxiv.org/abs/2510.10456)
Comments:
          Published in TMLR (10/2025)

- **What's New**: 이 논문은 일관성 있는 이상(anomaly)을 탐지하기 위해 새로운 알고리즘인 Consistent-Anomaly Detection Graph (CoDeGraph)를 소개합니다. 이는 기존의 zero-shot 이미지 이상 분류 및 세분화 방법의 한계를 극복하며, 특히 여러 이미지에서 반복적으로 나타나는 유사한 결함을 효과적으로 분리합니다. CoDeGraph는 이미지 수준의 그래프를 구축하여 커뮤니티 탐지를 통해 일관성 있는 이상을 필터링합니다.

- **Technical Details**: CoDeGraph의 핵심 통찰은 정상 패치(normal patch)는 여러 테스트 이미지 간의 유사성을 지속적으로 유지하는 반면, 일관성 있는 이상 패치는 서로 유사한 매치가 소진된 후 갑작스러운 유사성 스파이크를 보인다는 것입니다. 이를 통해, 연구자들은 'neighbor-burnout'이라는 개념을 도입하여, 일관성 있는 이상을 탐지하고 구분하기 위한 엔더런스 비율(endurance ratio) 메트릭을 개발했습니다. CoDeGraph는 이 과정을 통해 이미지 간의 연결성을 탐지하며, 이를 통해 강력한 성능을 발휘합니다.

- **Performance Highlights**: 실험 결과, CoDeGraph는 MVTec AD와 Visa 데이터셋에서 최고 성능을 기록하였으며, 특히 일관성 있는 이상 데이터셋에서는 F1-score에서 최대 14.9% 개선을 보였습니다. 또한, DINOv2 백본을 사용함으로써 세분화 성능이 더욱 향상되어 F1-score 69.1%, AP 71.9%에 도달했습니다. 이는 다양한 아키텍처에서의 견고함을 보여주며, CoDeGraph는 차세대 zero-shot 방법으로서의 가능성을 입증하고 있습니다.



### MonoSE(3)-Diffusion: A Monocular SE(3) Diffusion Framework for Robust Camera-to-Robot Pose Estimation (https://arxiv.org/abs/2510.10434)
- **What's New**: MonoSE(3)-Diffusion은 모노큘러 이미지로부터의 로봇 포즈 추정을 위한 새로운 확산 모델 기반 프레임워크입니다. 이 프레임워크는 평균 제거(denoising) 확산 프로세스를 조건으로 하여 카메라와 로봇 모델을 기반으로 포즈를 추정합니다. 이 접근 방식은 기존의 고정 스케일 변형 방법이 가지는 한계를 극복하고, 다양한 포즈를 생성해 네트워크의 일반화 능력을 향상시키는 데 목표를 두고 있습니다.

- **Technical Details**: MonoSE(3)-Diffusion 프레임워크는 두 가지 주요 프로세스로 구성됩니다: 가시성 제약 확산 프로세스(Visibility-constrained diffusion process, VisDiff)와 시간 단계 인식 역방향 프로세스(Timestep-aware reverse process, RevDiff). 훈련 단계에서는 ground-truth 포즈를 점진적으로 노이즈가 포함된 변형으로 샘플링하여 다양한 포즈 샘플을 생성합니다. 또한, 카메라의 시야 제약을 통합하여 포즈 샘플이 적절한 분포와 다양성을 유지하도록 합니다.

- **Performance Highlights**:  본 접근 방식은 DREAM 및 RoboKeyGen이라는 두 가지 벤치마크에서 성능 향상을 보이며, 특히 가장 어려운 데이터셋에서 66.75의 AUC를 달성하여 기존 방법에 비해 32.3%의 성능 향상을 기록했습니다. 이는 시간 단계 인식 세분화와 포즈 추정의 안정성을 강화하는데 기여합니다. 결과적으로 MonoSE(3)-Diffusion은 카메라-로봇 포즈 추정의 강력한 성능을 입증하고 있습니다.



### Taming a Retrieval Framework to Read Images in Humanlike Manner for Augmenting Generation of MLLMs (https://arxiv.org/abs/2510.10426)
Comments:
          12 pages, 5 figures

- **What's New**: 본 논문은 전통적인 Retrieval-Augmented Generation(RAG) 접근법의 한계를 극복하기 위해 Human-Like Retrieval-Augmented Generation(HuLiRAG) 프레임워크를 소개합니다. HuLiRAG는 'what-where-reweight' 단계를 통해 멀티모달 질문 답변을 더욱 신뢰성 있게 처리할 수 있게 합니다. 이 방법은 특히 정밀한 비주얼 정보가 필요한 상황에서 강화된 성능을 보여줍니다.

- **Technical Details**: HuLiRAG는 queries를 open-vocabulary detection(무작위 단어 탐지)으로 후보 지표에 고정하고, GroundingDINO와 SAM을 활용해 지역적 세부 정보를 예측하여 공간적으로 해석하는 모듈로 구성됩니다. 이 시스템은 또한 지역과 글로벌 정보를 균형 잡기 위해 learnable positive-negative optimization 방법을 통해 재조정됩니다. 이를 통해 절차적 사고의 명확한 흐름을 유지하며, 효과적인 질의를 지원합니다.

- **Performance Highlights**: 실험 결과 WebQA와 MultimodalQA 데이터셋에서 HuLiRAG가 지역 인식 재정렬과 공간 감독을 통해 사실 일관성을 향상시키고, 지시적 구속을 강화하여 비주얼 질문 답변 성능이 개선됨을 보였습니다. 특히, 이 연구는 언어 모델이 이미지의 세부 사항을 '읽는' 능력을 인간처럼 향상시켰음을 보여줍니다.



### Towards Cybersickness Severity Classification from VR Gameplay Videos Using Transfer Learning and Temporal Modeling (https://arxiv.org/abs/2510.10422)
- **What's New**: 본 연구는 VR(가상 현실) 게임 영상에서 고급 시각적 특징을 추출하여 사이버 멀미(cybersickness)의 심각도를 예측하는 새로운 방법론을 제시합니다. 최근 더 많은 연구가 멀티모달 딥 러닝 접근 방식을 통해 VR 센서 데이터를 활용하고 있으나, 비디오 기반 특성을 통한 예측은 제한적으로 다뤄졌습니다. 이를 해결하기 위해 InceptionV3 모델을 사용해 비디오에서 시각적 특징을 전이 학습(transfer learning)하여 LSTM(Long Short-Term Memory) 네트워크로 전달해 사이버 멀미의 변화를 시간에 따라 추적하고 예측합니다.

- **Technical Details**: 본 연구에서는 VRGameplay 비디오의 각 프레임에서 2048 차원의 특징 벡터를 추출하고, 이를 1D max pooling을 통해 downsampling하여 LSTM 네트워크의 입력으로 사용합니다. InceptionV3 모델을 통해 추출된 고급 시각적 특징을 LSTM에 입력하여 시간적 의존성을 캡처하며, 5-겹 stratified cross-validation을 통해 모델의 일반화 능력을 평가했습니다. 이러한 과정에서 Adam optimizer와 categorical cross-entropy 손실 함수를 사용하여 모델을 훈련하고, 조기 중단(early stopping) 기법을 적용했습니다.

- **Performance Highlights**: 연구 결과, 제안된 접근 방식은 사이버 멀미 심각도 분류에서 68.4%의 정확도를 달성했으며, 이는 기존의 비디오 데이터로만 훈련된 모델보다 높은 성능을 나타냅니다. 이 연구는 VR 개발자들이 VR 환경에서 사이버 멀미를 평가하고 완화하기 위한 실용적인 도구를 제공하며, 향후 비디오 기반 시간적 모델링에 대한 연구를 위한 기초를 마련합니다.



### Combo-Gait: Unified Transformer Framework for Multi-Modal Gait Recognition and Attribute Analysis (https://arxiv.org/abs/2510.10417)
- **What's New**: 본 연구는 Combo-Gait라는 다중 모달 및 다중 작업 프레임워크를 제안하여 2D 실루엣과 3D SMPL 피쳐를 결합하여 강력한 보행 인식을 수행합니다. 기존의 보행 인식 방법들이 주로 2D 또는 3D 단일 모달에 의존했던 것과 달리, 본 연구는 이들 모달을 결합하여 더 복잡한 인간의 보행 패턴을 포착할 수 있게 합니다. 또한, 복합적인 인간 속성 추정(예: 나이, 성별, BMI)도 동시에 수행하는 다중 작업 학습 전략을 도입합니다.

- **Technical Details**: 본 연구에서 제안하는 Combo-Gait는 비디오에서 2D 실루엣과 3D SMPL 피쳐를 추출하여 이를 융합하는 네트워크 구조를 가지고 있습니다. 이 아키텍처는 CNN 인코더와 다층 퍼셉트론(MLP)을 사용하여 보행 특성과 인간 속성을 통합합니다. 또한, 전체 프로세스의 효율성을 높이기 위해 범용 트랜스포머 아키텍처를 활용하여 다양한 작업에서의 학습을 지원합니다.

- **Performance Highlights**: BRIAR 데이터세트에서의 실험 결과, Combo-Gait는 기존 최첨단 방법들보다 우수한 보행 인식 성능을 보였으며 인간 속성 추정에서도 높은 정확성을 기록했습니다. 이러한 결과는 복합적인 모달 및 다중 작업 학습이 실제 환경에서의 보행 기반 인간 이해를 증진시키는 가능성을 보여줍니다. 또한, 다양한 거리와 피치 각도 조건에서도 효과적인 성능을 발휘함을 입증하였습니다.



### Guided Image Feature Matching using Feature Spatial Order (https://arxiv.org/abs/2510.10414)
- **What's New**: 본 논문에서는 이미지 피처 매칭(image feature matching)의 효율성을 개선하기 위해 특징 공간 순서(feature spatial order) 개념을 진보적 매칭 프레임워크(progressive matching framework)에 통합하였습니다. 이는 에피폴라 기하학(epipolar geometry)과 결합되어 매칭의 정확성과 효율성을 한층 높이는 방법을 제안합니다. 특히, 이미지 회전의 영향을 제거하기 위한 적합한 정렬 방법도 제시하였습니다.

- **Technical Details**: 기존의 피처 매칭 기법들은 특징 설명자(feature descriptor)를 사용하여 이미지 내의 피처를 비교하거나 구성하는 데 초점을 두었습니다. 하지만 많은 수의 피처 포인트가 감지되는 경우 매칭에 상당한 시간이 소요됩니다. 본 연구에서는 특징 공간 순서를 활용하여 잠재적인 매칭 범위를 계산하고 불필요한 매칭을 필터링하여 효율성을 높이었습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 전통적인 방법보다 통계적으로 더 효율적이고 더 정확한 피처 매칭을 제공함을 보여주었습니다. 다양한 벤치마크 데이터셋과 실제 이미지를 사용한 실험을 통해 이 기술의 활용 가능성을 입증하였습니다. 이는 많은 응용 분야에서 컴퓨터 비전의 발전에 기여할 것으로 예상됩니다.



### Mesh-Gait: A Unified Framework for Gait Recognition Through Multi-Modal Representation Learning from 2D Silhouettes (https://arxiv.org/abs/2510.10406)
- **What's New**: 이번 연구에서는 Mesh-Gait라는 새로운 다중 모달 복식 걷기 인식 프레임워크를 소개합니다. 이 프레임워크는 2D 실루엣에서 직접 3D 표현을 재구성하여 두 가지 모달리티의 장점을 통합합니다. 기존 방법에 비해 Mesh-Gait는 3D 조인트나 메쉬에서 직접 3D 기능을 학습하기 어려운 점을 해결하고, 중간 표현으로 3D 히트맵을 사용하여 계산 효율성을 유지합니다. 이 방법은 실시간 적용에 적합한 해결책을 제공합니다.

- **Technical Details**: Mesh-Gait의 아키텍처는 이중 분기 구조로 구성되며, 2D와 3D 기능 분기로 나뉩니다. 2D 기능 분기는 2D 실루엣으로부터 2D 걷기 기능을 추출하는 데 사용되고, 3D 기능 분기는 2D 실루엣에서 3D 메쉬를 재구성하고, 해당 모델로부터 3D 걷기 기능을 추출합니다. 훈련 과정에서는 트리플 손실, 교차 엔트로피 손실, L1 손실 및 L2 손실의 조합을 사용하여 모델을 최적화합니다.

- **Performance Highlights**: Mesh-Gait는 여러 벤치마크 데이터셋에서 평가되어 탁월한 성과를 보였습니다. 특히, 전통적인 2D 방법이 어려움을 겪는 변동 시점, 부분 차단 및 환경 소음이 있는 어려운 상황에서도 높은 인식 정확도와 강력한 견고성을 보여주었습니다. 이 접근 방식은 계산 효율성을 개선하여 제한된 자원에서도 실시간 걷기 인식이 가능하게 합니다.



### AVoCaDO: An Audiovisual Video Captioner Driven by Temporal Orchestration (https://arxiv.org/abs/2510.10395)
Comments:
          Project webpage: this https URL

- **What's New**: 이번 논문은 AVoCaDO라는 강력한 시청각 비디오 캡셔너를 제안합니다. AVoCaDO는 시각 및 청각 이벤트 간의 시간 동기화에 중점을 두고 있으며, 이 모델은 영상 이해 및 생성에 기여할 수 있도록 실질적인 의미를 담은 설명을 생성합니다. 두 단계의 후속 교육 프로세스를 통해, AVoCaDO는 고품질의 시청각 캡션 데이터셋을 활용하여, 시간적인 일관성과 정확도를 향상시킵니다.

- **Technical Details**: AVoCaDO의 후속 교육 파이프라인은 두 가지 단계로 구성됩니다: 첫 번째는 AVoCaDO SFT로, 107K의 시청각 캡션 데이터셋을 사용해 모델을 미세 조정합니다. 두 번째는 AVoCaDO GRPO로, 여기서는 주요 이벤트 정렬에 기반한 보상 함수를 도입하여 시각 및 청각 정보의 시간적 일관성을 최적화합니다. 이러한 보상 최적화는 대화를 더욱 정확하게 하고 반복을 줄이는 데 기여합니다.

- **Performance Highlights**: 실험 결과, AVoCaDO는 여러 시청각 캡셔닝 벤치마크에서 기존의 오픈 소스 모델들을 능가하는 성능을 보였습니다. 특히, UGC-VideoCap 데이터셋에서는 상업적 모델인 Gemini-2.5-Pro를 초과하는 성과를 달성했습니다. VDC 및 DREAM-1K 벤치마크에서도 경쟁력 있는 성능을 보이며, 비주얼 전용 설정에서도 우수한 결과를 냈습니다.



### Identifying bias in CNN image classification using image scrambling and transforms (https://arxiv.org/abs/2510.10383)
Comments:
          62 pages, Master's thesis

- **What's New**: 이 연구에서는 CNN(Convolutional Neural Networks) 모델의 검증된 성능에도 불구하고, 데이터셋의 편향(hidden bias)과 배경 소음(background noise)이 분류 과정에 미치는 영향을 분석합니다. 특히, 블랭크 배경을 포함하지 않은 이미지에서 CNN이 어떤 영향을 받는지 알아보고, 배경 정보를 사용하지 않고도 배경 소음을 식별할 수 있는 새로운 접근 방식을 제안합니다. 이 방법은 다양한 데이터셋에서 효과적으로 테스트되었으며, 해당 연구의 결과를 통해 CNN의 의사결정 과정에서의 편향 문제를 해결할 수 있는 방안을 모색합니다.

- **Technical Details**: 연구에서는 자연(natural), 비자연(non-natural), 혼합(mixed) 데이터셋의 6가지 서로 다른 데이터셋을 사용하여 CNN의 학습 과정에서 발생할 수 있는 편향을 연구합니다. 자연 데이터셋은 현실 세계에서 수집된 이미지로 조명 조건, 각도 및 배경이 다양하여 모델의 견고성을 높입니다. 비자연 데이터셋은 알고리즘을 통해 생성된 데이터로 상대적으로 변동성이 적으며, 특정 작업을 해결하기 위한 경향이 있습니다. 혼합 데이터셋은 실제 이미지와 합성 이미지를 결합하여 모델의 일반화 성능을 향상시키기 위해 설계되었습니다.

- **Performance Highlights**: 수행한 실험 결과, 제안된 두 가지 방법(이미지 타일 나누기 및 다양한 이미지 변환 적용)이 여러 데이터셋에서 배경 소음과 맥락 정보를 효과적으로 구별할 수 있음을 보여주었습니다. 특히, VGG16 네트워크를 사용하여 자연과 비자연 데이터셋 모두에서 높은 정확성을 달성했으며, CNN이 특정 태스크에서 불필요한 특성을 학습하는 경향이 있음을 발견했습니다. 이 연구는 CNN의 학습 방식에 대한 깊은 통찰을 제공하며, 이미지 분류 문제에서의 편향을 해결하기 위한 새로운 방법론을 제시합니다.



### Self-Supervised Multi-Scale Transformer with Attention-Guided Fusion for Efficient Crack Detection (https://arxiv.org/abs/2510.10378)
Comments:
          The paper has been published at Automation in Construction journal. The paper has 53 pages and 11 figures

- **What's New**: 이 논문은 수작업 주석 없이도 효과적인 픽셀 수준의 균열 분할을 달성할 수 있는 가능성을 탐구합니다. 이를 위해, Crack-Segmenter라는 완전 자가 지도 학습 프레임워크를 개발하였으며, 이는 Scale-Adaptive Embedder, Directional Attention Transformer, Attention-Guided Fusion의 세 가지 모듈로 구성되어 있습니다. 이 방식은 효율적인 크랙 감지를 가능하게 하여 교통 기관들이 보다 비용 효율적으로 인프라 모니터링을 수행할 수 있도록 합니다.

- **Technical Details**: Crack-Segmenter는 수동 주석 없이도 균열을 정확히 분할할 수 있도록 설계된 자가 지도 방식의 프레임워크입니다. 이 방법은 다중 해상도의 균열 특징을 포착하는 Scale-Adaptive Embedder, 균열의 연속성을 유지하는 Directional Attention Transformer, 다중 스케일 표현을 적응적으로 결합하는 Attention-Guided Fusion 모듈을 포함합니다. 이러한 모듈은 새로운 일관성 손실(consistency loss) 기법과 결합되어 효과적인 자가 지도 학습을 지원합니다.

- **Performance Highlights**: 본 연구는 10개의 공개 데이터 세트에서 13개의 최첨단 감독 모델에 대해 평가를 수행하였으며, 모든 주요 성능 지표에서 일관되게 우수한 성능을 나타냈습니다. 평균 Intersection over Union(mIoU), Dice score, XOR, Hausdorff Distance(HD) 등을 포함한 주요 메트릭에서 통계적으로 유의미한 개선을 보여주었습니다. 이 결과는 주석 없는 균열 검출이 가능할 뿐만 아니라, 효율적이라는 점을 입증합니다.



### Vision4PPG: Emergent PPG Analysis Capability of Vision Foundation Models for Vital Signs like Blood Pressur (https://arxiv.org/abs/2510.10366)
Comments:
          BHI abstract extended

- **What's New**: PPG(Photoplethysmography) 센서가 최근 웨어러블 및 임상 기기에서 널리 사용되며, 비침습적이고 실시간으로 생리학적 통찰을 제공합니다. 본 연구에서는 Vision Foundation Models(VFM)을 활용하여 PPG 작업에서 SOTA(state-of-the-art) 성능을 달성할 수 있음을 보여주며, 특히 혈압 추정에서 우수한 결과를 나타냅니다. 기존 시간 시리즈 모델을 사용한 연구와 비교할 때, VFMs는 STFT(Short-Time Fourier Transform)와 같은 2D 변환을 통해 PPG 신호를 처리하는 새로운 접근 방식을 제공합니다.

- **Technical Details**: 본 연구에서는 PPG 처리에 적합한 여러 모델을 비교하였습니다. MOMENT라는 일반 시간 시리즈 모델과 PPG에 전문화된 PPG-GPT 모델을 선택하여 VFM과 비교 분석을 실시하였습니다. DINOv3와 SIGLIP-2라는 최신 VFM을 활용하여 PPG 데이터를 변환하고, Parameter-Efficient Fine-Tuning(PEFT) 기법을 적용하여 모델의 성능을 최적화했습니다. 또한, 총 7개 데이터셋을 사용하여 비침습적 혈압 추정을 포함한 다양한 생리적 작업을 수행했습니다.

- **Performance Highlights**: Vision4PPG라는 제안된 접근법은 다양한 비침습적 생리 신호 작업에서 우수한 결과를 보였습니다. 혈압 추정 외에도 심박수, 호흡수, SPO2, 혈중 생화학 물질 농도 추정에서 SOTA 성능을 달성했습니다. 이는 VFMs가 PPG 신호 분석에 있어 보편적으로 적용 가능한 가능성을 보여주며, 임상 연구자들에게 효율적이고 강력한 도구를 제공합니다. 본 연구는 다양한 데이터셋에서 우수한 성능을 입증하여, 생리학적 신호 처리 분야에서 VFMs의 잠재력을 확인했습니다.



### PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion (https://arxiv.org/abs/2510.10365)
Comments:
          NeurIPS 2025

- **What's New**: 이번 논문에서는 PointMAC이라는 포인트 클라우드 완성(point cloud completion)을 위한 테스트 시간 적응(test-time adaptation) 프레임워크를 제안합니다. 이 방법은 추가적인 감독(supervision) 없이 각 샘플에 대해 세부 조정을 가능하게 하며, 기존의 정적 추론(static inference)을 넘어선 능동적인 적응을 보여줍니다. PointMAC는 메타 보조 학습(meta-auxiliary learning)에 기반하여, 여러 구조적 불완전성과 센서에 의한 왜곡을 모방하는 보조 목표로 모델을 최적화합니다.

- **Technical Details**: 이 방법은 Model-Agnostic Meta-Learning (MAML) 프레임워크를 활용하여 보조 목표(auxiliary objectives)와 주요 완성 작업(primary completion task) 간의 일관성을 유지합니다. 인퍼런스(inference) 시, 고정된 디코더를 유지하며 공유된 인코더를 실시간으로 조정하여 각 샘플에 맞는 정밀한 완성을 수행합니다. 또 다른 메커니즘인 Adaptive λ-Calibration은 주된 목표와 보조 목표 간의 경량화를 처리하여 모델의 적응성을 안정화합니다.

- **Performance Highlights**: PointMAC은 합성 데이터(synthetic data), 시뮬레이션된 데이터(simulated data) 및 실제 데이터(real-world datasets)에서 최첨단 결과(state-of-the-art results)를 달성했습니다. 이 방법은 특히 포인트 클라우드의 다양한 도메인에서 강력한 일반화 및 적응 능력을 보여주며, 샘플별 완성도(sample-specific completions)의 질을 향상시켰습니다. 기존 기술의 한계를 극복하며, 포인트 클라우드 완성 문제에 메타 보조 학습을 최초로 적용한 사례로 주목받고 있습니다.



### Ortho-Fuse: Orthomosaic Generation for Sparse High-Resolution Crop Health Maps Through Intermediate Optical Flow Estimation (https://arxiv.org/abs/2510.10360)
Comments:
          6 Figures, 9 pages

- **What's New**: 이 논문에서는 Ortho-Fuse라는 새로운 프레임워크를 소개합니다. 이 시스템은 전통적인 방법에 비해 orthomosaic (정사 영상) 생성을 위한 요구 overlap (중첩) 비율을 줄이면서도 신뢰할 수 있는 결과를 제공합니다. 이를 통해 농업 모니터링 기술의 채택 장벽을 낮추고, 정밀 농업 (precision agriculture) 시스템의 경제적 가능성을 높일 수 있는 경로를 제시합니다.

- **Technical Details**: Ortho-Fuse는 중간 optical flow (광학 흐름) 추정을 통해 연속적인 항공 이미지 간의 전환 이미지를 합성하며, 이를 통해 특징 대응을 인위적으로 증가시킵니다. 이 방법은 전통적인 photogrammetric (사진측량) 방법이 필요로 하는 70-80%의 overlap 비율을 약 20% 줄이는 데 성공했습니다. 이러한 기술적 기법은 인공지능 (AI) 기반 시스템이 데이터 수집 요건을 충족시키면서도 고품질의 복합 지도를 생성할 수 있도록 도와줍니다.

- **Performance Highlights**: 경험적 검증 결과, Ortho-Fuse는 최소 overlap 요구 사항을 20% 감소시키며, 농작물 건강 분석에 대한 정확도를 유지하는 데 기여했습니다. 또한, 이 시스템은 효율성을 극대화하여 산업 표준 소프트웨어와 비교해 성능 저하를 줄이는 데 유리함을 입증했습니다. 이러한 결과는 농업 분야에서 AI 기반 모니터링 시스템의 채택을 촉진하는 중요한 요소로 작용할 것입니다.



### Ordinal Scale Traffic Congestion Classification with Multi-Modal Vision-Language and Motion Analysis (https://arxiv.org/abs/2510.10342)
Comments:
          7 pages, 4 figures. Preprint submitted to arXiv in October 2025

- **What's New**: 이번 논문은 지능형 교통 시스템 및 실시간 도시 교통 관리를 위한 정확한 교통 혼잡 분류의 중요성을 강조합니다. 새로운 멀티모달 프레임워크는 개방 어휘 비주얼-언어 추론(open-vocabulary visual-language reasoning)인 CLIP, 객체 검출(object detection)인 YOLO-World 및 MOG2 기반 배경 차감(motion analysis via MOG2-based background subtraction)을 결합하여 혼잡 수준을 예측합니다.

- **Technical Details**: 제안된 시스템은 1(자유 흐름)에서 5(심각한 혼잡)까지의 서열(scale)로 혼잡 수준을 분류합니다. 또한, 움직임 기반의 신뢰도 가중치를 통합하고 주석 처리된 시각적 출력을 생성하여 해석 가능성을 높였습니다. 실험 결과, 모델은 76.7%의 정확도, 0.752의 F1 스코어, 0.684의 Quadratic Weighted Kappa(QWK)를 달성했으며, 단일 모달 기법(unimodal baselines)보다 유의미하게 우수한 성과를 보였습니다.

- **Performance Highlights**: 이 연구의 결과는 프레임워크가 서열 구조를 유지하고 비주얼-언어 및 움직임 모달리티를 활용하는 데 효과적임을 증명합니다. 향후 개선 사항으로는 차량 크기 측정(vehicle sizing)과 정제된 밀도 메트릭(refined density metrics)의 통합이 예정되어 있습니다.



### From Programs to Poses: Factored Real-World Scene Generation via Learned Program Libraries (https://arxiv.org/abs/2510.10292)
Comments:
          NeurIPS 2025

- **What's New**: FactoredScenes는 제한된 데이터로부터 현실적인 3D 장면을 합성하는 새로운 프레임워크를 제안합니다. 이 시스템은 방의 기본 구조를 활용하여 사물의 다양한 자세를 학습하는 방식으로 현실적인 장면 생성을 목표로 합니다. 기존의 데이터를 기반으로 장면을 생성하기 위해, FactoredScenes는 계층적으로 구성된 장면 개념을 도입하여 장면 생성을 보다 효율적으로 수행합니다.

- **Technical Details**: FactoredScenes는 장면을 프로그램 구조의 계층적 개념으로 분해합니다. 이 과정은 공간을 구성하는 데 필요한 5단계로 나뉘며, 첫 번째로는 방 구조를 포착하는 프로그램 라이브러리를 학습합니다. 그런 다음, 대형 언어 모델(large language models)을 사용하여 장면 프로그램을 생성하고, 이 프로그램을 실행하여 장면의 배치를 찾아낸 후, 마지막으로 객체의 자세를 예측하여 3D 객체를 배치합니다.

- **Performance Highlights**: FactoredScenes는 이전 연구들보다 현실적인 장면 레이아웃 생성을 개선하는 데 성공했습니다. FID와 KID 메트릭스를 기준으로 평가한 결과, 기능 사용에서 644.1%의 상대적 향상을 기록했습니다. 또한, 인간 연구를 통해 FactoredScenes가 생성한 장면이 실제 ScanNet 장면과 구별하기 어려운 수준으로 평가되었습니다.



### SAM2LoRA: Composite Loss-Guided, Parameter-Efficient Finetuning of SAM2 for Retinal Fundus Segmentation (https://arxiv.org/abs/2510.10288)
Comments:
          Accepted for publication at the 2025 International Conference on Machine Learning and Applications (ICMLA)

- **What's New**: SAM2LoRA는 fundus 이미지 분할을 위한 새로운 파라미터 효율적인 미세 조정 전략으로, Segment Anything Model 2 (SAM2)에 저차원 어댑터(low-rank adapter)를 통합하여 구현됩니다. 이 전략은 원래 훈련 가능한 매개변수의 5% 이하만 필요로 하여, 기존의 방법보다 훨씬 빠른 추론 속도를 자랑합니다. 또한 복합 손실 함수(composite loss function)를 도입하여 다양한 데이터셋에서 최적의 네트워크 조정을 가능하게 합니다.

- **Technical Details**: SAM2는 계층적 비전 트랜스포머(Hierarchical Vision Transformer) 아키텍처에 기반하여 다중 규모 기능 디코딩(multi-scale feature decoding)을 지원합니다. LoRA는 미세 조정 과정에서 가중치의 변화를 낮은 순위로 가정하여 효율적인 업데이트를 가능하게 하며, 이미지를 인코딩하는 모듈과 마스크 디코딩하는 모듈 모두에 통합됩니다. 또한, 세 가지 손실 요소인 segmentationBCE, SoftDice, FocalTversky를 결합한 손실 함수를 사용하여 복잡한 분할 작업을 최적화합니다.

- **Performance Highlights**: SAM2LoRA는 11가지의 도전적인 fundus 분할 데이터셋에서 검사된 결과, 혈관(segmentation of blood vessels) 분할에서 0.86, 시신경 원반(optic disc) 분할에서 0.93의 다이스 점수를 달성하며, AUC 값도 각각 최대 0.98과 0.99에 도달하여 최신 기술의 성능을 입증하였습니다. 이 연구는 적은 훈련 비용에도 불구하고 뛰어난 성능을 유지하며, 기존 방법들에 비해 훈련 부담을 크게 줄입니다.



### Bridging Perspectives: Foundation Model Guided BEV Maps for 3D Object Detection and Tracking (https://arxiv.org/abs/2510.10287)
- **What's New**: 이 논문에서는 DualViewDistill라는 새로운 하이브리드 탐지 및 추적 프레임워크를 제안합니다. 이 프레임워크는 시점(view) 기반의 특징과 조감도(bird's-eye view, BEV) 기반의 특징을 결합하여 3D 물체 탐지 및 추적의 성능을 개선합니다. BEV 맵은 DINOv2라는 비전 모델을 통해 가이드되어, 객체 및 공간 표현의 이점을 함께 활용할 수 있습니다.

- **Technical Details**: DualViewDistill는 DINOv2의 기술을 이용하여 다양한 카메라 이미지에서 추출한 특징을 LiDAR 포인트 클라우드에 투영합니다. 이로 인해, BEV 공간으로 평균 풀링을 통해 생성된 BEV 의사 레이블을 통해 탐지 및 추적에서 더 풍부하고 표현력이 뛰어난 장면 맵을 형성합니다. 이를 통해 기존의 지도 기반 탐지기의 제한을 극복하고 하이브리드 표현을 통해 성능을 향상시키는 것이 가능합니다.

- **Performance Highlights**: nuScenes 및 Argoverse 2 벤치마크에서 확인된 바와 같이, DualViewDistill는 최첨단 성능을 기록하며 기존의 카메라 기반 탐지 및 추적 방식보다 개선된 결과를 보여주었습니다. 특히, AMOTA 지표에서 유의미한 향상을 나타내었으며, ID 스위치 문제를 줄였습니다. 이 연구는 자율 주행을 위한 신뢰할 수 있는 인식을 위한 한 걸음을 내딛는 데 기여하고 있습니다.



### VividAnimator: An End-to-End Audio and Pose-driven Half-Body Human Animation Framework (https://arxiv.org/abs/2510.10269)
Comments:
          Comments: 10 pages, 6 figures

- **What's New**: 본 논문에서는 음성과 희박한 손 자세 조건에 의해 구동되는 고품질 반신(half-body) 인간 애니메이션을 생성하기 위한 새로운 프레임워크인 VividAnimator를 제안합니다. 기존의 방법들은 뻣뻣한 머리 움직임과 흐릿한 손 이미지 문제로 고생하는데, VividAnimator는 이러한 문제를 해결할 수 있는 세 가지 주요 혁신 요소를 포함하고 있습니다. 이 프레임워크는 수치적으로 우수한 손 세부 사항과 자연스러운 제스처 전환을 가능하게 하는 방법을 제공합니다.

- **Technical Details**: VividAnimator의 핵심 구성 요소 중 하나는 손의 세부 묘사 향상을 위한 Hand Clarity Codebook (HCC)입니다. 이 코드북은 고품질 손 이미지를 통해 사전 훈련된 VAE 모델을 활용하여 손 텍스처의 품질을 높입니다. 두 번째 요소인 Dual-Stream Audio-Aware Module (DSAA)은 오디오 신호를 처리하여 입술 동기화와 자연스러운 머리 자세를 분리하여 모델의 표현력을 높입니다. 마지막으로 Pose Calibration Trick (PCT)을 통해 자세 정렬을 개선하며 매끄러운 제스처 전환을 도모합니다.

- **Performance Highlights**: 실험 결과 Vivid Animator는 손 디테일, 제스처의 사실성, 정체성 일관성 면에서 최신 기술을 초월하는 성능을 보여줍니다. 정량적 지표와 정성적 평가 모두에서 이러한 성과를 입증하며, Vivid Animator는 고품질 반신 인간 애니메이션 생성을 위한 새로운 기준을 세우게 됩니다. 이 연구는 엔터테인먼트, 영화 제작 및 가상 캐릭터 창작과 같은 다양한 분야에서의 응용 가능성을 제시합니다.



### Opacity-Gradient Driven Density Control for Compact and Efficient Few-Shot 3D Gaussian Splatting (https://arxiv.org/abs/2510.10257)
- **What's New**: 이 논문에서는 3D Gaussian Splatting (3DGS)의 최적화를 재조정하여 효율성을 우선시하는 새로운 프레임워크를 제시합니다. 기존의 Adaptive Density Control (ADC) 방식을 개선하여, 일정한 알고리즘을 통해 렌더링 오류의 경량 프록시로서 opacity gradient를 사용하는 Error-Driven Densification 기법을 도입했습니다. 이 새로운 접근법은 기존의 과도한 밀도 조정 및 pruning 문제를 해결하여, 더 효율적이고 고품질의 3D 재구성을 가능하게 합니다.

- **Technical Details**: 제안된 프레임워크는 기본적인 3DGS 최적화 알고리즘을 수정하여, 밀도를 조정하는 과정에서 opacity gradient를 기반으로 한 새로운 오류 주도 기법을 도입합니다. 이 기법은 이전의 밀도 조정 방식이 가진 비효율성을 해결하기 위해 더 보수적인 pruning 일정을 필요한 곳에 배치합니다. 이것은 깊이 상관 손실과 결합되어, 최신 기법들과 비교하여 효율성에서 극적인 개선을 이룰 수 있는 방법론을 제공합니다.

- **Performance Highlights**: 테스트 결과, LLFF 데이터셋에서 제안된 모델은 FSGS보다 40% 이상 더 컴팩트한 성능을 보여주었으며(32k vs. 57k primitives), Mip-NeRF 360 데이터셋에서는 약 70%의 크기 감소를 기록했습니다. 이러한 놀라운 성능 개선은 reconstructed metrics에서의 적절한 trade-off를 통해 이루어졌으며, few-shot view synthesis의 품질과 효율성 간의 새로운 상태를 확립했습니다.



### Are Video Models Emerging as Zero-Shot Learners and Reasoners in Medical Imaging? (https://arxiv.org/abs/2510.10254)
- **What's New**: 최근 대규모 생성 모델의 발전은 간단한 자가 회귀(autoregressive) 모델이 적절히 확장될 때 강력한 제로샷 제너럴리제이션(zero-shot generalization) 성능을 보여줄 수 있음을 입증했습니다. 본 논문에서는 이러한 자가 회귀 비디오 모델링 원리를 의학 이미징 작업에 적용할 수 있는지를 조사하며, 의학 데이터로 훈련되지 않은 대규모 비전 모델(Large Vision Model, LVM)을 네 가지 대표적 작업인 장기 분할(organ segmentation), 잡음 제거(denoising), 초해상도(super-resolution), 그리고 동작 예측(motion prediction)에서 평가합니다. LVM은 CT 스캔에서 해부학적 구조를 세밀하게 구분할 수 있으며, 제로샷 설정에서도 높은 성능을 달성했습니다.

- **Technical Details**: LVM은 대규모 이미지 및 비디오 데이터에 대해 훈련된 시각 자가 회귀(transformer) 모델로, 4D CT 데이터에서 122명의 환자로부터 1,820개 이상의 3D CT 이미지를 평가했습니다. 이 모델은 의료 데이터에 대한 사전 조사가 없었음에도 불구하고 모든 작업에서 우수한 성능을 보였으며, 특히 방사선 치료에서 3D CT 비자화 예측에서 높은 공간 정확도를 달성했습니다. 이는 LVM이 환자 특유의 호흡 동역학을 캡처할 수 있도록 하는 인지적 추론 능력을 발휘했음을 나타냅니다.

- **Performance Highlights**: 우리는 단일 대규모 비디오 모델이 의학 이미징 작업(장기 분할, 탐지, 잡음 제거, 초해상도, 동작 예측)을 다룰 수 있음을 입증했으며, 작업별 재훈련 없이 이러한 다양한 작업에 적용될 수 있음을 보여줍니다. LVM은 제로샷 설정에서도 DVF(Deformation Vector Field) 기반 및 생성 모델보다 동작 예측에서 우수한 성능을 발휘하며, 이는 자가 회귀 비디오 모델이 시간적 의존성과 동적 패턴을 캡처하는 데 뛰어난 성능을 발휘한다는 점을 시사합니다. 이러한 결과는 통합된 의료 이미징 프레임워크를 개발하는 가능성을 강조하며, 세분화된 아키텍처 및 수작업으로 만든 작업 설계에 대한 의존성을 줄일 수 있음을 시사합니다.



### MRI Brain Tumor Detection with Computer Vision (https://arxiv.org/abs/2510.10250)
Comments:
          12 pages, 8 figures, final project report for CS4100 (Machine Learning), Northeastern University, April 2024

- **What's New**: 이 연구는 MRI 스캔에서 뇌종양을 자동으로 탐지하고 분할하는 데 딥 러닝 기술의 적용을 탐구합니다. 기초 로지스틱 회귀, 합성곱 신경망(CNN), 잔여 네트워크(ResNet) 등을 사용하여 뇌종양을 효과적으로 분류하고, U-Net 및 EfficientDet을 통해 종양의 로컬라이제이션과 식별을 개선합니다. 결과적으로, 딥 러닝이 의료 이미징에서 진단의 정확성과 효율성을 높이는 데 기여할 수 있음을 보여줍니다.

- **Technical Details**: 본 연구에서는 7,023개의 MRI 이미지를 포함하는 Brain Tumor MRI Dataset과 110명의 환자에서 수집된 LGG Segmentation Dataset을 활용합니다. 이 데이터셋은 glioma, meningioma, no tumor, pituitary의 네 가지 클래스로 세분화되며, 이는 종양의 형태와 유전적 아형, 환자 결과 간의 연구를 지원합니다. 또한, 소프트웨어는 PyTorch를 기반으로 구현되어 있으며, CNN과 ResNet 모델의 다양한 아키텍처를 최적화하여 뇌종양 탐지 및 분할의 도전 과제를 해결합니다.

- **Performance Highlights**: 로지스틱 회귀 모델은 낮은 정확도를 보였으나, CNN 모델은 이와 비교해 99% 이상의 높은 정확도를 기록했습니다. ResNet은 복잡한 이미지를 처리하는 데 효과적이며, 깊은 네트워크 구조 덕분에 AUC 성능이 개선되었습니다. 깊이 있는 네트워크 특성이 뇌 MRI 이미지에서 중요한 세부사항을 학습하여, 뇌종양의 정확한 분류를 가능하게 합니다.



### Semantic Visual Anomaly Detection and Reasoning in AI-Generated Images (https://arxiv.org/abs/2510.10231)
Comments:
          27 pages, 7 figures

- **What's New**: AI 생성 콘텐츠(AIGC)의 급속한 발전으로 사실감 있는 이미지 합성이 가능해졌지만, 이러한 이미지들은 종종 의미적 이상(semaic anomalies)을 포함하고 있습니다. 이 연구는 AIGC 이미지의 의미적 이상 탐지 및 추론을 공식화하고, 이러한 이상의 구조화된 주석을 갖춘 대규모 벤치마크인 AnomReason을 도입합니다. AnomReason은 이러한 주석을 생성하기 위한 모듈러 다중 에이전트 파이프라인인 AnomAgent를 사용하여 품질을 유지하면서도 대규모 데이터 작업을 가능하게 합니다.

- **Technical Details**: AnomAgent는 AI 생성 이미지에서 의미적 이상을 탐지하기 위해 엔티티 파싱(entity parsing), 이상 감지(anomaly mining), 구조화된 출력 생성을 포함하는 세 가지 단계로 구성됩니다. 각 단계는 해석 가능하고 높은 정확도를 가진 의미적 이상 주석을 생성하기 위한 전문화된 에이전트들이 협력합니다. 이상 탐지의 결과로는 이름, 현상, 추론 및 심각도 점수가 포함된 구조화된 이상 목록이 생성됩니다.

- **Performance Highlights**: AnomReason을 기반으로 미세 조정된 모델은 강력한 시각-언어 기준을 초과하는 일관된 성과를 보였습니다. 제안된 새로운 지표인 SemAP와 SemF1은 의미적 이상 탐지 성능 평가를 용이하게 하여, 이미지 생성 모델의 의미적 일관성을 향상시키는 데 기여합니다. 이 연구는 AI 생성 이미지의 의미적 진위를 평가하고, 설명 가능한 딥페이크 탐지 및 생성 모델의 논리적 일관성을 검토하는 데 중요한 적용 가능성을 보여줍니다.



### A Style-Based Metric for Quantifying the Synthetic-to-Real Gap in Autonomous Driving Image Datasets (https://arxiv.org/abs/2510.10203)
Comments:
          7 pages, 4 figures

- **What's New**: 이번 논문에서는 자율주행 시스템에서 합성 데이터셋의 신뢰성을 확보하기 위해 'synthetic-to-real gap'을 정량화하는 체계적인 프레임워크를 제시합니다. 새로운 평가 지표인 Style Embedding Distribution Discrepancy (SEDD)를 도입하여 합성 데이터와 실제 데이터 간의 스타일 차이를 평가합니다. 이 연구는 실제 데이터셋과 합성 데이터셋을 기반으로 한 벤치마크를 수립하고, 다양한 데이터셋과 시뮬레이션 방법에서 실험을 수행하여 효과성을 입증합니다.

- **Technical Details**: 제안된 모델은 실세계 이미지와 다양한 날씨 조건의 합성 이미지를 입력으로 받아들입니다. 훈련 단계에서 특징 추출기와 스타일 추출기를 통해 특징 임베딩을 생성하고, 이를 바탕으로 Metric learning을 통해 중간 손실(Center Loss)과 NTXent Loss를 포함한 손실 함수를 최적화합니다. 평가 단계에서는 학습된 특징 임베딩을 후처리하여 합성-실제 간의 차이를 정량화하는 신뢰성 메트릭을 계산합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 합성 데이터와 실제 데이터 간의 갭을 정량화할 수 있는 능력을 보여줍니다. 연구 결과는 데이터 기반 자율주행 시스템의 발전을 지원하는 표준화된 품질 관리 도구를 제공하며, 향후 효과적인 훈련 파이프라인 디자인을 위한 기초자료로 활용될 수 있습니다.



### From Generic to Specialized: A Subspecialty Diagnostic System Powered by Self-Supervised Learning for Cervical Histopathology (https://arxiv.org/abs/2510.10196)
Comments:
          32 pages, 6 figures

- **What's New**: 본 논문은 자궁경부암 진단을 위한 Cervical Subspecialty Pathology (CerS-Path) 시스템을 소개합니다. 이 시스템은 약 1억 9천만 개의 조직 패치와 140,000장의 슬라이드에서 진행된 자기 지도 학습을 통해 자궁경부 특화된 특징 추출기를 구축하였으며, 250만 개의 이미지-텍스트 쌍과의 다중 모달 교육을 통해 기능을 향상시켰습니다. CerS-Path는 드문 암 분류 및 다중 모달 Q&A 등 8개의 진단 기능을 지원하며, 이전 모델들보다 범위와 임상 적용 가능성에서 우수함을 보여줍니다.

- **Technical Details**: CerS-Path는 두 가지 상호작용하는 전처리 단계로 개발되었습니다. 첫 번째 단계는 자기 지도 학습(self-supervised learning)으로, 이는 자궁경부 특화 특징 추출기를 만드는 데 사용되었습니다. 두 번째 단계는 다중 모달 강화(multimodal enhancement)로, 이미지와 텍스트 쌍을 통합하여 여러 하위 진단 기능과 결합하였습니다.

- **Performance Highlights**: CerS-Path는 5개 센터에서 3,173건에 대한 사전 테스트를 통해 99.38%의 선별 감도를 유지하며 우수한 일반화 성능을 보였습니다. 이는 자궁경부암의 진단 번역 및 선별에 있어 상당한 발전을 나타내며, 다양한 진단 기능을 실현하는 데 기여합니다.



### B2N3D: Progressive Learning from Binary to N-ary Relationships for 3D Object Grounding (https://arxiv.org/abs/2510.10194)
- **What's New**: 본 연구에서는 3D 객체의 위치를 자연어로 Localizing하는 과정을 개선하기 위한 새로운 progressive relational learning framework을 제안합니다. 기존의 방법이 이진 관계 모델링에 한정되어 있을 때, 우리는 n-ary 관계로 확장하여 다중 모달 관계 이해의 글로벌 인식을 가능하게 합니다. 이를 통해 더욱 복잡한 장면에서도 객체를 효과적으로 식별할 수 있는 방법을 제공합니다.

- **Technical Details**: 본 연구는 주어진 텍스트 설명을 기반으로 객체 관계를 예측하는 binary-to-n-ary progressive relational learning module (B2N-PRL)을 포함합니다. 우리는 훈련 데이터에서 특정 객체에 대한 주석이 부족한 상황을 해결하기 위해 그룹화된 감독 손실 함수를 설계했습니다. 또한, 주어진 n-ary 관계로 이루어진 scene graph를 구축하고 이를 통합하기 위해 혼합 주의 기법을 사용하는 다중 모달 네트워크를 활용합니다.

- **Performance Highlights**: ReferIt3D와 ScanRefer 벤치마크에서의 실험 결과, 본 연구에서 제안한 방법이 최신 기술들에 비해 우수한 성능을 보인 것을 확인했습니다. 이 연구는 3D 객체의 로컬라이제이션 작업에서 글로벌 관계 인식을 강조하며, n-ary 관계의 인식에서 높은 정확도를 나타냅니다. 전체적으로 제안된 방법은 3D 환경에서의 효율적인 정보 처리를 가능하게 합니다.



### Fairness Without Labels: Pseudo-Balancing for Bias Mitigation in Face Gender Classification (https://arxiv.org/abs/2510.10191)
Comments:
          8 pages. Accepted for publication in the ICCV 2025 Workshop Proceedings (2nd FAILED Workshop). Also available on HAL (hal-05210445v1)

- **What's New**: 이 연구는 자동화된 성별 분류 모델이 훈련 데이터의 인구 통계적 편향을 반영하고 강화하는 경향이 있음을 강조하며, 이러한 편향을 완화하기 위한 새로운 방법인 pseudo-balancing을 제안합니다. 이 방법은 주로 레이블이 없는 이미지에서 성별 균형을 유지하는 데 초점을 맞추며, 기존의 지리적 분포에 다음uidas 연결된 진짜 레이블 없이 데이터의 균형을 맞추는 기술적 접근을 제공합니다. 전체적으로 이 연구는 인구 통계적으로 균형 잡힌 데이터셋을 활용하여 모델의 공정성과 정확성을 개선하는 방법을 모색합니다.

- **Technical Details**: 연구에서는 FairFace 데이터세트를 사용하여 레이블이 없는 이미지에서 pseudo-balancing을 수행합니다. 이 방법은 (1) 표본 선택 단계에서만 모델을 수정하고, (2) 신뢰 기반 샘플 선택을 통해 모델의 정확성을 유지하며, (3) 기존의 적대적 학습이나 분포 가정 없이 자기 훈련을 수행합니다. 두 가지 시나리오를 통해 이 방법의 효능과 한계를 평가하며, 특히 성별 균형 유지에 중점을 두고 있습니다.

- **Performance Highlights**: pseudo-balancing 방법을 적용한 결과, 모델은 79.81%의 정확도를 기록하며, 이는 기존 기준선보다 6.53% 향상된 수치입니다. 성별 정확성의 격차는 44.17% 감소되었으며, 동아시아 하위 그룹에서도 49%를 초과하는 기존 격차가 5.01%로 줄어들었습니다. 이러한 결과는 레이블 감독 없이도 인구 통계적으로 균형 잡힌 데이터셋을 활용하면 기존 비 대칭 컴퓨터 비전 모델의 편향을 줄이는 강력한 자원이 될 수 있음을 시사합니다.



### TCMA: Text-Conditioned Multi-granularity Alignment for Drone Cross-Modal Text-Video Retrieva (https://arxiv.org/abs/2510.10180)
- **What's New**: 이 논문은 Drone Video-Text Match Dataset (DVTMD)을 구축하여 무인 항공기(UAV) 비디오와 텍스트 간의 상관관계를 강화하는 데 기여합니다. DVTMD는 2,864개의 비디오와 14,320개의 세분화된 캡션을 포함하고 있으며, 각 비디오는 다각적인 정보(예: 인간 행동, 사물, 환경 등)를 캡처하여 텍스트-비디오 간의 매칭 품질을 높입니다. 텍스트 조건부 다중 세분화 정렬 프레임워크(TCMA)도 함께 제안되어, 비디오와 텍스트 간의 정밀한 정렬을 가능하게 합니다.

- **Technical Details**: DVTMD는 기존의 일반적인 모션 지향 레이블 대신 비디오의 여러 보완적 요소에 대해 세분화된 주석을 제공합니다. 이 프레임워크는 전역 비디오-문장 정렬, 문장 안내 프레임 집합, 단어 안내 패치 정렬을 통합하여 비디오에서 특정 세부 사항을 포착합니다. 추가적으로, 워드 및 패치 선택 모듈과 텍스트 적응형 동적 온도 메커니즘을 도입하여 각각의 텍스트 유형에 맞춰 주의 집중도를 조정합니다.

- **Performance Highlights**: DVTMD와 CapERA를 기반으로 한 실험에서 제안된 TCMA는 텍스트-비디오 및 비디오-텍스트 검색에서 각각 45.5% R@1 및 42.8% R@1을 기록하며 최첨단 성과를 달성했습니다. 이는 해당 데이터셋과 방법론의 효과성을 입증하는 결과입니다. 본 연구는 드론 비디오-텍스트 검색 분야에 대한 최초의 포괄적인 벤치마크를 설정합니다.



### HccePose(BF): Predicting Front \& Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation (https://arxiv.org/abs/2510.10177)
Comments:
          International Conference on Computer Vision, ICCV 2025 (Highlight) this https URL

- **What's New**: 이 연구는 객체의 앞면과 뒷면의 3D 좌표를 동시에 예측하는 신경망 기반의 새로운 접근 방식을 제안합니다. 또한, 이 방법은 두 표면 사이의 3D 좌표를 조밀하게 샘플링하여 초조밀 2D-3D 대응을 생성함으로써 Pose Estimation (자세 추정)의 정확도를 향상시킵니다. 연구에서는 Hierarchical Continuous Coordinate Encoding (HCCE)을 활용하여 더 정확하고 효율적인 좌표 표현을 제공합니다.

- **Technical Details**: HCCE 방법론은 각 표면 좌표의 xx, yy, zz 구성 요소를 개별적으로 인코딩하며, 이러한 구성 요소를 다단계 연속 코드로 변환하기 위해 미러링 연산을 수행합니다. 또한, 네트워크 훈련 중 각 구성 요소에 대해 별도의 히스토그램을 계산하여 학습 난이도를 반영하고, 이를 기반으로 손실 함수에서 계층 코드의 가중치를 조정합니다. 이를 통해 안정적인 훈련과 더 나은 예측 정확도를 달성합니다.

- **Performance Highlights**: 실험 결과는 제안한 방법이 기존의 최첨단(RGB 기반) 방법보다 BOP 점수에서 2.4% 우수하며, RGB에서 RGB-D 데이터로 전환 시 4.7% 향상을 달성했다는 것을 보여줍니다. 또한 2D 분할 작업에서도 기존의 최상의 접근 방식을 3.7% 초과하여 그 효율성을 추가로 입증하였습니다.



### ViConEx-Med: Visual Concept Explainability via Multi-Concept Token Transformer for Medical Image Analysis (https://arxiv.org/abs/2510.10174)
Comments:
          This work has been submitted to the IEEE for possible publication

- **What's New**: 이번 연구에서는 ViConEx-Med라는 새로운 트랜스포머 기반 프레임워크를 제안하여 시각적 개념 설명 가능성을 향상시키고자 하였습니다. 이 모델은 여러 개념을 대표하는 학습 가능한 토큰을 도입하여 시각적 개념을 예측하고 정위치화하는 기능을 갖추고 있습니다. 또한, 기존의 개념 기반 모델들과 비교했을 때 실제 의료 데이터셋에서도 높은 성능을 보여줍니다.

- **Technical Details**: ViConEx-Med는 다중 개념 토큰을 활용하는 비전 트랜스포머 기반 아키텍처를 사용합니다. 특수한 주의(attention) 레이어를 통해 시각적 및 텍스트 기반 개념 토큰을 처리하며, 개념 수준의 정위치화 지도(concept-level localization maps)를 생성합니다. 이 모델은 이미지에서 예측된 개념의 시각적 정위치화를 가능하게 하며, 오직 개념 수준의 레이블만 사용하여 훈련됩니다.

- **Performance Highlights**: 실험 결과, ViConEx-Med는 여섯 개의 의료 이미지 데이터셋에서 최고의 정확도를 기록하며, 기존 개념 기반 모델과 블랙박스 모델과 경쟁할 수 있는 성능을 입증했습니다. 이러한 성능은 특히 고위험 의료 상황에서 신뢰성을 높이고, 임상의의 의사결정을 지원하는 데 유용할 것으로 기대됩니다. 추가적으로, SynSkin이라는 합성 데이터셋을 도입하여 기존 데이터셋의 성능을 향상하는 데 기여하였습니다.



### SparseUWSeg: Active Sparse Point-Label Augmentation for Underwater Semantic Segmentation (https://arxiv.org/abs/2510.10163)
- **What's New**: SparseUWSeg는 수중 이미지 분석을 위한 혁신적인 프레임워크로, 전문가의 주석 작업을 최대한 활용하기 위해 능동 샘플링 전략을 사용합니다. 이 프레임워크는 희소한 포인트 레이블을 효과적으로 조합하고 전파하여, 수중 생태 모니터링에서의지를 높이는 새로운 접근법을 제시합니다. SparseUWSeg의 주요 기여는 간단하면서도 효과적인 인터랙티브 주석 도구를 설계하고 배포하여, 연구자들이 자신의 데이터에 고품질 세분화 마스크를 생성할 수 있도록 돕는 것입니다.

- **Technical Details**: SparseUWSeg는 두 가지 주요 구성 요소로 이루어져 있습니다: 슈퍼픽셀(segmentation) 및 SAM2 기반의 하이브리드 희소 레이블 증강 전략과 전문가 주석을 더 잘 안내하기 위한 능동 포인트 샘플링 전략입니다. 기존의 D+NN과 초수적 방법들에서 각자 장점을 효율적으로 활용하여 성능을 극대화하고, 10점 이상의 포인트 집합에서 +3% 이상의 mIoU 개선을 달성합니다. 이 프레임워크는 또한 수중 데이터의 밀집 픽셀 레이블 생성을 지원하는 단순한 인터랙티브 주석 도구를 출시합니다.

- **Performance Highlights**: SparseUWSeg는 두 가지 다양한 수중 데이터셋에서 실험을 통해 기존의 최신 기법들보다 성능이 뛰어난 것을 입증하였으며, D+NN 대비 최대 +5%의 mIoU 향상을 기록합니다. 이 결과는 수중 생태 모니터링에 대한 최신 심층 학습 세분화 기법의 적용 가능성을 향상시키고 있습니다. 이를 통해 신청자는 최소한의 주석 작업으로도 안정적이고 고품질의 세분화 결과를 획득할 수 있게 됩니다.



### SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation (https://arxiv.org/abs/2510.10160)
Comments:
          NeurIPS 2025

- **What's New**: 이 논문에서는 Referring Image Segmentation (RIS)의 새로운 프레임워크인 SaFiRe를 제안합니다. 이 프레임워크는 인간의 인지 과정을 모방하여 전반적인 이해를 먼저 형성한 후, 세부 사항을 검토하는 두 단계의 방식을 따릅니다. 또한, aRefCOCO라는 새로운 벤치마크를 도입하여 모호한 표현에서 RIS 모델을 평가합니다.

- **Technical Details**: SaFiRe는 Mamba의 scan-then-update 속성을 활용하여 효율적인 다중 사이클 정제를 가능하게 합니다. 이 구조는 선형 복잡성을 유지하여 사용자에게 편리함을 제공합니다. 우리는 또한 두 가지 도전적인 실제 시나리오, 즉 여러 개체가 포함된 객체가 산만하게 하는 표현과 객체 클래스가 명시되지 않은 범주 암시 표현을 다루고 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 SaFiRe는 기존의 최첨단 모델보다 뛰어난 성능을 보여주었습니다. 제안된 데이터 셋과 표준 데이터 셋 모두에서 SaFiRe의 효율성이 입증되었습니다. 이러한 성과는 RIS 문제의 처리 능력을 크게 향상시킬 것으로 기대됩니다.



### ReMix: Towards a Unified View of Consistent Character Generation and Editing (https://arxiv.org/abs/2510.10156)
- **What's New**: 이 논문에서는 ReMix라는 통합 프레임워크를 제안하여 캐릭터 일관성을 갖춘 이미지 생성 및 편집을 수행합니다. 이 프레임워크는 ReMix 모듈과 IP-ControlNet 두 가지 핵심 컴포넌트로 구성되어 있으며, 멀티모달 추론 능력을 활용하여 입력 이미지의 시맨틱 기능을 편집하고, 이를 통해 flexible한 작업을 지원합니다. 또한, 기존 방법들이 가지는 한계를 극복하는 새로운 접근 방식을 제시하고 있습니다.

- **Technical Details**: ReMix 모듈은 사전 훈련된 MLLM(Multimodal Language Model)을 이용해 텍스트 지시사항과 참조 이미지를 모두 처리합니다. IP-ControlNet은 ControlNet 아키텍처를 확장하여 시맨틱 정보와 레이아웃 단서를 분리하고, 공유된 노이즈 공간 내에서 참조 이미지와 타겟 이미지의 노이즈를 동시에 제거하는 ε-equivariant latent space를 도입합니다. 이를 통해 픽셀 수준의 일관성을 보장하면서도 높은 품질의 이미지를 생성합니다.

- **Performance Highlights**: ReMix는 개인화된 생성, 이미지 편집, 스타일 전송, 다조건 합성 등 다양한 작업을 수행할 수 있는 능력을 갖추고 있습니다. Extensive experiment를 통해 효율성과 효과성을 입증하였으며, 캐릭터 일관성을 갖춘 이미지 생성 및 편집을 위한 통합 프레임워크로서의 가능성을 보여줍니다. 이 접근 방식은 DiT(backbone) 재훈련 없이도 이미지 생성을 가능하게 하여, 훈련 비용을 크게 줄이는 동시에 원본 생성을 유지합니다.



### Stroke Locus Net: Occluded Vessel Localization from MRI Modalities (https://arxiv.org/abs/2510.10155)
Comments:
          This version of the paper was accepted in the ADMA 2025 conference in Kyoto, Japan

- **What's New**: 이번 연구에서는 MRI 스캔만을 사용하여 뇌졸중 진단을 위한 새로운 딥 러닝 파이프라인인 Stroke Locus Net을 도입합니다. 이 시스템은 병변(segmentation) 검출과 혈관(localization) 위치 파악을 결합하여, 뇌졸중으로 손상된 혈관을 정확하게 식별하는 데 초점을 맞추고 있습니다. 기존의 방법은 대부분 병변 검출에 치중되어 있었지만, 이 연구는 혈관의 개별적인 위치를 효과적으로 확인하는 방법을 제시합니다.

- **Technical Details**: Stroke Locus Net은 nnUNet을 기반으로 한 병변 검출 분기와 혈관 매핑을 위한 동맥 아틀라스를 결합한 구조입니다. 또한, pGAN을 사용하여 MRI에서 MRA 이미지를 합성하는 생성 분기를 포함하고 있습니다. 이 프로세스는 MRI 스캔에서 병변을 세분화하고, 그에 따른 혈관의 위치를 판별하여 정밀한 뇌졸중 진단을 가능하게 합니다.

- **Performance Highlights**: 우리의 방법은 T1 MRI 스캔 상에서 차단된 혈관을 성공적으로 로컬라이징하는 데 유망한 결과를 보여주었습니다. 이 시스템은 뇌졸중 진단의 속도를 높이고, 더 나은 의사결정을 지원할 잠재력이 있습니다. Stroke Locus Net의 목표는 혈관의 빠르고 정확한 위치 확인을 통해 뇌졸중 치료의 효율성을 향상시키는 것입니다.



### Color3D: Controllable and Consistent 3D Colorization with Personalized Colorizer (https://arxiv.org/abs/2510.10152)
Comments:
          Project Page this https URL

- **What's New**: 이번 연구에서는 Color3D라는 새로운 프레임워크를 소개합니다. 이 프레임워크는 단색 입력으로부터 정적 및 동적 3D 장면을 색칠하는 데 매우 적응성이 뛰어나며, 사용자 안내에 따라 시각적으로 다양한 복원 결과를 제공합니다. 기존 방법들이 정적 시나리오에만 초점을 맞추고 여러 뷰 간의 색상 일관성을 유지하기 위해 평균화를 사용함으로써 색상 다양성을 희생하는 반면, Color3D는 색상 다양성과 조정 가능성을 유지하면서도 크로스 뷰 및 시간 일관성을 보장합니다.

- **Technical Details**: Color3D의 핵심 통찰력은 하나의 주요 뷰를 색칠한 후, 이를 기반으로 개인화된 색상화기를 미세 조정하여 새로운 뷰와 시간 단계로 색상을 전파하는 것입니다. 이를 통해 각 장면에 대해 개인화된 색상화기를 최적화하고, 색상 정보 전파 작업을 단순화하여 일관된 결과를 도출합니다. Color3D는 Lab 색상 공간의 Gaussian splatting 표현을 사용하여 색상 재구성을 직접적으로 수행하며, 이를 통해 사용자 의사에 맞춘 역동적인 색칠이 가능해집니다.

- **Performance Highlights**: 다양한 정적 및 동적 3D 색칠 벤치마크에서 Color3D의 성능을 검증한 결과, 사용자 의도에 맞는 일관되고 색상 풍부한 렌더링을 제공함을 확인했습니다. Color3D는 실세계 장면에서도 시각적으로 만족스러운 결과를 달성하여 기존의 방법들과 비교했을 때 정량적 및 정성적으로 우수한 성능을 보였습니다. 이러한 결과는 Color3D가 복잡한 3D 색칠 문제를 보다 다루기 쉬운 단일 이미지 패러다임으로 재구성할 수 있음을 보여줍니다.



### YOLOv11-Litchi: Efficient Litchi Fruit Detection based on UAV-Captured Agricultural Imagery in Complex Orchard Environments (https://arxiv.org/abs/2510.10141)
- **What's New**: 이번 논문은 UAV(무인 항공기) 기반의 리치 검출을 위해 특별히 설계된 YOLOv11-Litchi라는 경량화되고 강력한 탐지 모델을 소개합니다. 이 모델은 작은 대상 크기, 대규모 매개변수로 인한 배포의 어려움, 빈번한 대상 occlusion(가림 현상)과 같은 주요 문제를 해결하는 데 중점을 두고 있습니다. 이를 위해 다중 스케일 잔여 모듈(multi-scale residual module), 경량화된 피처 융합 방법, 그리고 occlusion 탐지 헤드가 도입되었습니다.

- **Technical Details**: YOLOv11-Litchi는 YOLOv11 프레임워크를 기반으로 하며, 모델 매개변수 크기는 6.35MB로 YOLOv11의 기준보다 32.5% 더 작습니다. 이 모델은 개선된 mAP(mean Average Precision) 90.1%와 F1-Score 85.5%를 달성하여 높은 정확도를 유지하면서도 경량화를 실현했습니다. 또한, 초당 57.2 프레임(FPS)의 속도로 실시간 탐지 요구 사항을 충족합니다.

- **Performance Highlights**: 실험 결과, YOLOv11-Litchi는 복잡한 과수원 환경에서 UAV 기반의 리치 탐지에 적합하다는 것을 입증했습니다. 이 모델은 농업의 정밀 관리를 위한 광범위한 응용 가능성을 보여주며, 전통적인 수동 방법의 제한을 초월하여 보다 빠르고 정확한 솔루션을 제공합니다. 더 나아가 이 연구는 UAV와 심층 학습의 통합이 농업 생산성을 얼마나 향상시킬 수 있는지를 시사합니다.



### DeepFusionNet: Autoencoder-Based Low-Light Image Enhancement and Super-Resolution (https://arxiv.org/abs/2510.10122)
Comments:
          12 pages, 11 figures

- **What's New**: 본 논문에서는 DeepFusionNet 아키텍처를 통해 저조도(low-light) 이미지의 처리 성능을 향상시키는 방법을 제안하고 있습니다. 기존의 autoencoder 기반 방법들은 낮은 SSIM(Structural Similarity Index)과 PSNR(Peak Signal-to-Noise Ratio) 점수를 보였으나, DeepFusionNet은 이러한 문제를 해결하고 있습니다. 이 네트워크는 약 250만 개의 파라미터로 구성되어 비교적 낮은 계산 능력으로도 효과적인 성능을 발휘합니다.

- **Technical Details**: DeepFusionNet은 저조도 이미지와 흐릿한 이미지의 초해상도(super-resolution)를 처리하기 위한 새로운 접근 방식을 제공합니다. 본 연구에서는 LOL-v1 데이터셋을 사용하여 이 아키텍처의 성능을 평가하였고, PSNR 점수 26.30과 SSIM 92.8%를 달성하였습니다. 또한, DeepFusionNet을 기반으로 한 초해상도 모델은 약 10만 개의 파라미터로 설계되어 있습니다.

- **Performance Highlights**: DeepFusionNet을 통한 초해상도 방법은 검증 세트(validation set)에서 PSNR 25.30과 SSIM 80.7%를 기록하며, 기존 GAN 기반 기법에 비해 효율적인 성능을 보여줍니다. 이러한 결과는 처리 속도와 품질을 동시에 고려할 수 있는 가능성을 열어줍니다.



### Multi Class Parkinsons Disease Detection Based on Finger Tapping Using Attention-Enhanced CNN BiLSTM (https://arxiv.org/abs/2510.10121)
- **What's New**: 이번 연구에서는 손가락 두드리기를 기반으로 한 다중 클래스 파킨슨병(Parkinson's Disease, PD) 탐지 시스템을 제안하였습니다. 제안된 모델은 attention을 강화한 CNN-BiLSTM 아키텍처를 사용하여 PD의 심각도를 분류합니다. 심박수, 주파수 및 진폭 기반의 특성을 기반으로 비디오 데이터를 처리하여, 이는 기존의 방법들보다 높은 정확도를 기대할 수 있습니다.

- **Technical Details**: 제안된 모델은 57가지 특성을 활용하여 PD를 5개의 범주로 분류합니다. Conv1D MaxPooling 블록을 통해 입력 시퀀스를 재구성하고 지역적 공간 의존성을 포착합니다. BiLSTM 레이어를 통해 시간적 동학을 모델링하고, attention 메커니즘을 추가하여 가장 정보가 유의미한 시간적 특성에 집중하게 합니다.

- **Performance Highlights**: 모델은 5개의 중증도 클래스를 구분하는 데 강력한 성능을 보여주어 PD의 자동화된 심각도 탐지에 있어 주목할 만한 가능성을 제시합니다. 공간적 및 시간적 표현을 결합한 attention 메커니즘을 통해 정확성을 향상시킬 수 있으며, 이는 클리닉에서 PD 모니터링 및 경과 추적을 지원하는 비침습적 도구로서 기대됩니다.



### ImmerIris: A Large-Scale Dataset and Benchmark for Immersive Iris Recognition in Open Scenes (https://arxiv.org/abs/2510.10113)
- **What's New**: 본 논문은 ImmerIris라는 대규모 아이리스 데이터셋을 소개하며, 이는 VR 헤드셋을 통해 수집된 499,791개의 안구 이미지를 포함하고 있습니다. ImmerIris는 오프축(off-axis) 이미지를 수집한 최초의 대규모 공개 데이터셋 중 하나로, 아이리스 인식 연구에 중대한 기여를 할 것으로 기대됩니다.

- **Technical Details**: 전통적인 아이리스 인식 시스템은 정면에서 이미지를 수집하는 반면, Immersive setup에서는 기울어진 헤드셋 카메라를 통해 off-axis 이미지를 포착합니다. 이로 인해에 발생하는 세 가지 주요 도전 과제는: 1) 관점 왜곡(perspective distortion), 2) 품질 저하(quality degradation), 3) 클래스 내 변동성(intra-class variation)입니다. 논문에서는 이러한 도전 과제를 극복하기 위해 정규화(normailzation) 없이도 안구 이미지로부터 직접 학습하는 간단한 접근 방식을 제안합니다.

- **Performance Highlights**: 제안된 기법은 기존의 SOTA(State-Of-The-Art) 방법들보다 더 나은 성능을 보이며, 정규화 기반 접근 방식의 의존성을 감소시키는 것을 목표로 합니다. 결과적으로, 새로운 프레임워크는 복잡한 요인들이 존재하는 immserive 환경에서 더욱 우수한 성과를 보여주었고, 이는 향후 아이리스 인식을 위한 유망한 발전 방향을 제시합니다.



### Training-Free In-Context Forensic Chain for Image Manipulation Detection and Localization (https://arxiv.org/abs/2510.10111)
- **What's New**: 이번 논문에서는 In-Context Forensic Chain (ICFC)이라는 새로운 프레임워크를 제안합니다. ICFC는 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)을 활용하여 훈련이 필요 없는 이미지 조작 로컬라이제이션(Image Manipulation Localization, IML)을 가능하게 합니다. 이 접근 방식은 물체화된 규칙 구성(Objectified Rule Construction)과 적응형 필터링(Adaptive Filtering)을 통합하며, 전문가의 법의학적 워크플로우를 반영하는 다단계 점진적 추론(Multi-step Progressive Reasoning) 파이프라인을 사용합니다.

- **Technical Details**: ICFC는 불확실한 법의학적 신호를 해석 가능한 규칙으로 변환하고 관련 증거를 적응적으로 선택하기 위해 규칙 분해 및 필터링(Rule Decomposition and Filtering, RDF) 기법을 적용합니다. 이 시스템은 MLLM과 전문가의 피드백을 결합하여 17개의 조작 범주를 포함하는 68개의 규칙으로 구성된 객체화된 규칙 집합(Objectified Rule Set, ORS)을 생성합니다. 최종적으로 ICFC는 이미지 수준에서의 분류, 픽셀 수준의 로컬라이제이션 및 텍스트 수준의 해석 가능성을 위한 정교한 분석을 제공합니다.

- **Performance Highlights**: ICFC는 여러 벤치마크에서 최첨단 훈련 프리가 적용된 방법들을 능가하고 약한 감독 및 완전 감독 접근 방식에 비해 경쟁력 있는 성능을 달성하였습니다. 또한, 이 모델은 복잡한 IML 작업을 다루기 위해 명확한 추론 경로로 MLLM이 수행할 수 있도록 해줍니다. 결국, 이 연구는 사진 조작 탐지와 법의학적 분석을 위한 유용한 도구로 자리잡을 것으로 예상됩니다.



### Uncertainty-Aware Post-Detection Framework for Enhanced Fire and Smoke Detection in Compact Deep Learning Models (https://arxiv.org/abs/2510.10108)
Comments:
          Accepted and to be presented at the International Conference on Smart Multimedia (ICSM 2025) - this https URL

- **What's New**: 이 논문에서는 화재 및 연기 감지의 정확성을 높이기 위해 불확실성을 고려한 후처리 프레임워크를 제안합니다. 기존의 비전 기반 감지 방법이 효율성과 신뢰성 간의 균형을 맞추는 데 어려움을 겪고 있다는 점을 강조하고 있으며, YOLOv5n과 YOLOv8n과 같은 경량 딥러닝 모델이 사용되고 있다는 것을 알리고 있습니다. 본 프레임워크는 감지 점수를 통계적 불확실성과 도메인 관련 시각적 단서를 통해 조정하여, 감지 후 신뢰도를 보강합니다.

- **Technical Details**: 제안된 포스트-디텍션 프레임워크는 YOLO와 같은 기본 객체 감지 모델에서 감지 결과를 정제하기 위해 불확실성 추정 및 감지 영역 특성 분석을 통합합니다. 기존의 후처리 방법들과 달리 이 프레임워크는 휴리스틱 기반의 신뢰도 조정을 대신하여 학습된 Confidence Refinement Network (CRN)를 도입하여 더 적응적이고 견고한 감지 파이프라인을 돕습니다. 각 감지된 경계 상자는 색상, 가장자리 및 질감 특성에 따라 평가되며, 이 과정을 통해 화재 및 연기 지역과의 일관성을 확보합니다.

- **Performance Highlights**: D-Fire 데이터셋에 대한 실험 결과, 제안된 후처리 방법이 기존의 기준에 비해 정밀도, 재현율 및 평균 정밀도(Mean Average Precision)가 향상되었음을 보여줍니다. 이 연구는 경량 딥러닝 모델이 실제 화재 및 연기 감지에 더욱 견고하게 작용할 수 있도록 지원하는 후처리 신뢰도 보강의 효과를 강조합니다. 또한 기존 기술들과 비교할 때, 이 방법은 컴퓨터 자원의 소모가 적으면서도 유의미한 성능 향상을 제시합니다.



### Answer-Consistent Chain-of-thought Reinforcement Learning For Multi-modal Large Langauge Models (https://arxiv.org/abs/2510.10104)
- **What's New**: 이 논문은 강화 학습 기반의 모델에서 Reasoning-Answer Mismatch 문제를 해결하기 위해 'Answer-Consistent Reinforcement Learning (ACRE)'을 제안합니다. 이전의 모델들이 답변의 정확성만을 최적화하는 데 집중함에 따라, Reasoning과 최종 답변 간의 불일치가 발생하는 경향이 있었습니다. ACRE는 이 문제를 해결하기 위해 GRPO를 수정하고 추가적인 일관성 검사 과정으로 구성된 새로운 보상 메커니즘을 도입합니다.

- **Technical Details**: ACRE는 다단계 Reasoning 과정과 최종 답변을 동시에 생성하고, 이후 선택지들을 섞은 후에 동일한 Reasoning 트레이스를 활용하여 새로운 답변을 예측하도록 합니다. 두 과정에서의 답변이 일관된 경우에만 최대 보상을 부여하고, 그렇지 않은 경우에는 낮은 보상을 할당하여 Reasoning의 신뢰성을 증진합니다. 이 방식을 통해 모델이 오류가 발생하기 쉬운 경로를 따라가는 것을 방지하고, 좀 더 신뢰할 수 있는 Reasoning 능력을 개발하도록 유도합니다.

- **Performance Highlights**: ACRE는 Video Reasoning 및 다중 모달 수학 Reasoning 벤치마크에서 2.2% 및 1.5%의 향상을 이루어냈습니다. 이러한 향상은 ACRE가 데이터셋에 대해 불필요한 편향을 줄이는데 기여함을 보여줍니다. 또한, ACRE는 기존 GRPO 기반 모델보다 더 신뢰할 수 있는 결과를 제공하여 실제 응용에서의 유용성을 높입니다.



### Cooperative Pseudo Labeling for Unsupervised Federated Classification (https://arxiv.org/abs/2510.10100)
Comments:
          Accepted by ICCV 2025

- **What's New**: 본 논문은 Unsupervised Federated Learning (UFL)을 CLIP을 활용하여 분류 문제에 처음으로 확장하였으며, 이를 위해 새로운 방법론인 FedCoPL(Federated Cooperative Pseudo Labeling)을 제안합니다. 기존의 UFL 연구들은 주로 표현 학습 및 군집 과제에 집중하였으나, 이번 연구에서는 분류 문제를 탐구하는 새로운 기회를 제시합니다. FedCoPL을 통해 클라이언트는 추정한 의사 레이블 분포를 서버에 업로드하고, 서버는 이를 조정하여 클래스 간의 글로벌 불균형을 방지합니다.

- **Technical Details**: FedCoPL은 협력적 의사 레이블링(cooperative pseudo labeling)과 부분 프롬프트 집계(partial prompt aggregation)라는 두 가지 핵심 컴포넌트를 포함합니다. 클라이언트는 신뢰도 기반 및 엔트로피 기반 필터링 방법을 사용하여 의사 레이블 분포를 추정하고, 서버는 각 클라이언트의 분포를 조정하여 클래스 불균형 문제를 해결합니다. 더불어, 시각적 프롬프트는 서버에서 집계되고, 개인화된 지식을 인코딩하는 텍스트 프롬프트는 로컬에 유지되도록 하는 부분 프롬프트 집계 프로토콜을 도입하였습니다.

- **Performance Highlights**: 전통적인 연합 프롬프트 학습 벤치마크를 사용한 광범위한 실험 결과, 제안된 FedCoPL이 기존 방법들에 비해 우수한 성능을 보임을 확인하였습니다. 본 연구는 CLIP의 제로샷(zero-shot) 분류 능력을 활용하여 더 복잡한 과제를 처리할 수 있는 새로운 가능성을 보여줍니다. 또한, 데이터의 불균형이 존재하는 경우에도 효과적으로 클라이언트 간의 협업을 촉진하고 개인화를 유지할 수 있는 방법론을 제시하였습니다.



### Gesplat: Robust Pose-Free 3D Reconstruction via Geometry-Guided Gaussian Splatting (https://arxiv.org/abs/2510.10097)
- **What's New**: 이번 논문에서는 Gesplat이라는 3D Gaussian Splatting(3DGS) 기반의 새로운 프레임워크를 소개합니다. 이 프레임워크는 비태그된 희소 이미지로부터 견고한 새로운 시점 합성을 가능하게 하고 기하학적으로 일관된 재구성을 제공합니다. 기존의 COLMAP을 기반으로 한 희소 점 구름 초기화 접근 방식 대신, VGGT라는 기반 모델을 활용하여 보다 신뢰할 수 있는 초기 포즈와 밀집 점 구름을 얻습니다.

- **Technical Details**: Gesplat은 몇 가지 주요 혁신 사항을 통합하며, 첫째로 하이브리드 Gaussian 표현이 있습니다. 이는 상호 시점 일관성을 향상시키기 위한 위치-형상 이중 최적화를 포함합니다. 둘째로, 그래프 기반 속성 정제 모듈을 설계하여 장면 세부정보를 향상시키고, 셋째로, 깊이 추정을 개선하기 위한 흐름 기반 깊이 정규화를 적용하여 훈련 중 렌더링 품질을 향상시킵니다. 이러한 기술들이 통합되어 높은 품질의 3D 재구성과 새로운 시점 합성을 제공합니다.

- **Performance Highlights**: 실험 결과, Gesplat은 LLFF 및 Tanks and Temples 데이터 세트에서 희소 보기 비전 방식의 이미지로부터 장면 재구성과 새로운 시점 합성에서 기존 비포즈 방법들에 비해 월등한 성능을 보였습니다. 이는 효과적인 기하학적 우선값을 도입하여 장면 구조를 제약하고, 세부 회복을 위한 최적화 및 정규화 기술이 결합된 결과입니다. 대규모 복잡한 데이터 세트에서의 실험을 통해, Gesplat의 강력한 성능이 입증되었습니다.



### Tracking the Spatiotemporal Evolution of Landslide Scars Using a Vision Foundation Model: A Novel and Universal Framework (https://arxiv.org/abs/2510.10084)
- **What's New**: 이 연구는 대규모 산사태 상처의 시공간적(evolution of spatiotemporal) 변화를 추적하는 새로운 프레임워크를 제안합니다. 기존 연구들은 주로 단일 단계 또는 전후 단계에서의 산사태 식별에 초점을 맞추었으나, 이 연구는 영상 기반 모델(vision foundation model)을 사용하여 보다 효과적으로 변화를 추적할 수 있도록 합니다. 이 프레임워크는 이산(discrete) 광원거리 원격 센서 이미지를 연속적인 비디오 시퀀스로 재구성하는 혁신적인 접근 방식을 제안합니다.

- **Technical Details**: 제안된 프레임워크는 지식 유도(knowledge-guided), 자동 전파(auto-propagation), 상호 정제(interactive refinement)의 패러다임 내에서 작동하며, 이를 통해 산사태 상처의 지속적이고 정확한 식별을 보장합니다. 비디오 세분화(video segmentation) 위해 개발된 비전 기반 모델을 활용하여, 산사태 상처의 변화를 효과적으로 추적할 수 있습니다. 이 연구는 또한 두 개의 대표 사례인 포스트 실패(Po-stfailure) Baige 산사태와 활성(active) Sela 산사태(2017-2025)의 적용을 통해 검증되었습니다.

- **Performance Highlights**: 실험 결과에 따르면, 제안된 프레임워크는 산사태 상처의 지속적인 추적을 가능하게 하여, 조기 경고를 위한 실패 전 조짐(failure precursors) 및 이차 위험과 장기 안정성 평가에 필수적인 실패 이후(post-failure) 진화를 포착하는 데 효과적입니다. 이는 대규모 산사태 모니터링 및 관리를 크게 향상시킬 수 있는 가능성을 보여줍니다. 연구 결과는 조기 경고 시스템 및 자연재해 관리에서 중요한 통찰을 제공합니다.



### Probabilistic Hyper-Graphs using Multiple Randomly Masked Autoencoders for Semi-supervised Multi-modal Multi-task Learning (https://arxiv.org/abs/2510.10068)
- **What's New**: 이 논문에서는 Masked Autoencoders(MAE)를 기반으로 한 확률적 하이퍼 그래프(Probabilistic Hyper-Graphs using Masked Autoencoders, PHG-MAE)라는 혁신적인 모델을 제시합니다. 기존의 신경망 그래프와 현대의 MAE 방식을 통합하여 데이터 기반으로 모달리티 간의 상호 의존성을 학습할 수 있도록 설계되었습니다. 이 모델은 고유한 뉴럴 네트워크로 설정된 그래프 구조의 제약을 극복하여, 다양한 모달리티를 보다 유연하게 처리할 수 있게 합니다.

- **Technical Details**: PHG-MAE는 기존 MAE 알고리즘의 확장을 통해 전처리(pre-training)와 세부 조정(fine-tuning)을 단일 학습 루프에서 수행할 수 있습니다. 전체 모달리티를 마스킹하는 방식으로, 단순한 패치 레벨 마스킹이 아닌 더 풍부한 표현력을 제공합니다. 이 모델은 다양한 입력 및 출력 모달리티를 처리하며, 특히 세분화 작업에 중점을 두고 있습니다.

- **Performance Highlights**: 모델은 Dronescapes 멀티 모달 멀티 태스크 벤치마크와 이전에 보지 못한 비디오에서 경쟁력 있는 성능을 보여줍니다. 작은 CNN 네트워크(매개변수 수 150k~4.4M)에서도 우수한 비디오 일관성을 달성할 수 있으며, 고품질의 예측을 위한 데이터 파이프라인을 오픈 소스로 제공하여 연구 환경에서의 활용을 보다 용이하게 합니다.



### Collaborative Learning of Semantic-Aware Feature Learning and Label Recovery for Multi-Label Image Recognition with Incomplete Labels (https://arxiv.org/abs/2510.10055)
- **What's New**: 이번 논문에서는 불완전한 레이블을 가진 다중 레이블 이미지 인식을 위한 새로운 방법인 CLSL(CoLLaborative Learning of Semantic-aware feature learning and Label recovery)을 제안합니다. 이 방법은 의미적 특징 학습과 레이블 복구 문제를 통합된 학습 프레임워크로 통합하여 해결합니다. 논문에서 개발한 방법은 시맨틱 연관 특징 학습 모듈과 시맨틱 유도 특징 향상 모듈로 구성되어 있습니다.

- **Technical Details**: 제안된 CLSL 방법은 먼저 글로벌 시각적 특징과 레이블 임베딩을 융합하여 의미적 특징을 생성하는 의미 관련 특징 학습 모듈을 설계합니다. 이후, 시각적 및 의미적 특징 공간 간의 정렬을 개선하기 위해 저계수 쌍선형 모델을 활용하여 고품질 시맨틱 인식 특징을 생성합니다. 마지막으로, 시맨틱 인식 특징 학습과 레이블 복구를 통합하는 협업 학습 프레임워크를 도입하여 동적으로 레이블을 복구하고 이를 반복적으로 최적화합니다.

- **Performance Highlights**: 세 가지 데이터셋(MS-COCO, VOC2007, NUS-WIDE)에 대한 실험 결과, 제안된 CLSL 프레임워크는 불완전한 레이블을 가진 다중 레이블 이미지 인식에서 최첨단 성능을 달성하였습니다. 기존 방법들과 비교하여 새로운 방식이 대폭 향상된 성능을 보였으며, 시맨틱 인식 특징의 구별 능력을 동적으로 향상시키고 무시된 레이블을 능동적으로 복구하는 능력을 보여주었습니다.



### DREAM: A Benchmark Study for Deepfake REalism AssessMen (https://arxiv.org/abs/2510.10053)
- **What's New**: 이 논문에서는 Deepfake의 시각적 사실성 평가에 대한 새로운 작업인 DREAM(Deepfake REalism AssessMent)을 제시합니다. DREAM은 딥페이크 비디오의 사실성을 자동으로 평가하여 인간의 지각에 근접하는 방법으로, 정보 신뢰성에 대한 심각한 위협이 되는 딥페이크 기술의 평가 및 개선에 기여할 가능성이 있습니다. 이 연구는 다양한 품질의 딥페이크 비디오 데이터셋과 3,500명의 인간 평가자가 수집한 140,000개의 사실성 점수와 텍스트 설명으로 구성된 포괄적인 벤치마크를 생성하여 향후 연구의 기초를 마련합니다.

- **Technical Details**: DREAM은 각각의 동영상에 대해 인간 평가자에 의해 주어진 주관적인 점수를 기반으로 하는 새로운 머신 러닝 모델을 사용합니다. 기존의 딥페이크 탐지 모델과는 달리, DREAM 모델은 '진짜' 또는 '가짜'와 같은 객관적인 레이블 대신 '매우 높은 사실성', '평균적인 사실성' 등의 주관적인 레이블을 사용합니다. 논문은 또한, CLIP 모델을 현실성 평가에 적응시킨 새로운 방법인 DA-CLIP을 제안하며, 이는 기존 방법보다 우수한 성능을 보입니다.

- **Performance Highlights**: DREAM은 대규모 크라우드소싱으로 수집된 고품질 주석을 기반으로 하여 각 비디오당 평균 92개의 평가 점수를 기록했습니다. 이는 기존 연구와 비교하여 현실성 평가에 있어 신뢰성을 크게 향상시킵니다. 또한, 텍스트 설명을 기반으로 한 다중 모달 정보도 수집되어, DREAM 벤치마크의 유용성을 더합니다. 본 논문의 제안된 방법은 모든 기존 방법을 초월하며, 텍스트 기반의 설명 능력이 뛰어난 것으로 입증되었습니다.



### Think Twice to See More: Iterative Visual Reasoning in Medical VLMs (https://arxiv.org/abs/2510.10052)
Comments:
          25 pages, 21 figures

- **What's New**: 의료 비전-언어 모델(VLMs)은 주로 단일 통과의 추론에 의존하여 지역화된 시각적 단서를 간과하는 경향이 있습니다. 이를 보완하기 위해, 이 논문에서는 인간 전문가의 반복적인 사고 과정을 모방하는 새로운 VLM 프레임워크인 ViTAR를 소개합니다. ViTAR는 'think-act-rethink-answer'라는 인지 사슬을 통해 의료 이미지를 상호작용 가능한 객체로 취급하며, 다단계 시각적 추론을 가능하게 합니다.

- **Technical Details**: 우리는 1K의 고품질 지침 데이터셋을 만들어 전문가 수준의 진단 행동을 인코딩하였습니다. 또한, 16K의 비주얼 질문 응답(VQA) 훈련 데이터를 수집하여 세밀한 시각 진단을 지원합니다. 이 연구는 주관적 세밀 조정을 포함한 2단계 훈련 전략을 도입하여 인지 경로를 안내하고, 의사결정을 최적화하기 위해 강화 학습을 활용합니다.

- **Performance Highlights**: ViTAR는 여러 VQA 벤치마크에서 우수한 성능을 보여줍니다. 시각적 주의 분석 결과, 'think'에서 'rethink' 단계로 넘어가면서 ViTAR는 임상적으로 중요한 지역에 시각적 기초를 더 강하게 고정하게 됩니다. 이러한 언급은 ViTAR의 성능 향상에 대한 메커니즘 통찰을 제공합니다.



### Complementary and Contrastive Learning for Audio-Visual Segmentation (https://arxiv.org/abs/2510.10051)
Comments:
          Accepted to IEEE Transactions on Multimedia

- **What's New**: 이 논문에서는 오디오-비주얼 분할(Audio-Visual Segmentation, AVS)을 위한 새로운 프레임워크인 Complementary and Contrastive Transformer (CCFormer)를 제안하였습니다. CCFormer는 오디오 신호와 비주얼 데이터를 통합하여 더 정교한 픽셀 단위의 분할 맵을 생성하는 것을 목표로 합니다. 기존의 CNN 및 Transformer 기반 방법들이 가진 한계를 극복하기 위해 설계된 CCFormer는 시공간적 맥락을 효과적으로 포착할 수 있습니다.

- **Technical Details**: CCFormer는 먼저 Early Integration Module (EIM)을 통해 멀티 스케일 비주얼 특성과 오디오 데이터를 병렬로 통합하여 교차 모달 보완성을 증가시킵니다. 또한, Multi-query Transformer Module (MTM)은 오디오 쿼리와 비주얼 특성 간의 상호작용을 통해 프레임 간 객체 인식 능력을 향상시킵니다. 마지막으로 Bi-modal Contrastive Learning (BCL)을 통해 통합된 특성 공간에서 두 모달리티 간의 정렬을 촉진합니다.

- **Performance Highlights**: 실험 결과, CCFormer는 AVSBench-object 및 AVSBench-semantic 데이터셋에서 기존 방법보다 상당히 높은 성능을 기록하여 최신 기술 수준의 벤치마크를 수립하였습니다. 이 성과는 CCFormer가 제안한 모듈들의 효과적인 조합을 통해 이루어졌습니다. 이러한 발전은 다양한 응용 분야에서 오디오 및 비주얼 정보를 더 잘 이해하고 처리할 수 있는 가능성을 보여줍니다.



### P-4DGS: Predictive 4D Gaussian Splatting with 90$\times$ Compression (https://arxiv.org/abs/2510.10030)
- **What's New**: 이번 논문에서는 P-4DGS라는 새로운 동적 3D Gaussian 표현 방식을 제안합니다. 기존의 동적 장면들의 시간적 및 공간적 중복성을 간과한 알고리즘들의 한계를 극복하기 위해, 이 방법은 영상 압축 기법에서 영감을 받아 설계된 공간-시간 예측 모듈을 포함하고 있습니다. 이를 통해 3D 앵커 포인트를 기반으로 하여 동적 3D Gaussian의 메모리 사용량을 감소시키는 것을 목표로 합니다.

- **Technical Details**: P-4DGS는 공간 예측(stochastic prediction)과 시간 예측(temporal prediction) 구조를 결합하여 동적 3D Gaussian 간의 상관관계를 최대한 활용합니다. 3D 앵커 포인트를 사용하여 근처의 Gaussian을 하나의 앵커 포인트로 예측함으로써 프리미티브 수를 줄이고, MLP(다층 퍼셉트론)를 사용해 각 3D Gaussian의 변형 벡터를 예측합니다. 또한, 적응형 양자화(adaptive quantization) 및 문맥 기반 엔트로피 코딩(contextual entropy coding)을 통해 3D 앵커 포인트의 크기를 추가로 줄여, 압축 효율성을 높입니다.

- **Performance Highlights**: P-4DGS는 기존 동적 3DGS 표현 방식과 비교하여 탁월한 복원 품질과 빠른 렌더링 속도를 보여주었습니다. 실험 결과, 평균 1MB 크기로 저장되며, 합성 및 실제 장면에 대해 각각 최대 40배 및 90배의 압축률을 달성했습니다. 이러한 높은 압축 효율성을 바탕으로, 메모리 사용량을 줄이면서도 향상된 렌더링 품질을 제공합니다.



### Q-Adapter: Visual Query Adapter for Extracting Textually-related Features in Video Captioning (https://arxiv.org/abs/2510.10022)
Comments:
          ACM Multimedia Asia 2025

- **What's New**: 이번 논문은 비디오 자막 생성 분야에서 Parameter-Efficient Fine-Tuning (PEFT) 접근 방식을 사용하는 Q-Adapter라는 새로운 방법을 제안합니다. Q-Adapter는 작은 시각적 어댑터 모듈을 통해 MLLMs(Multimodal Large Language Models)의 효율적인 미세 조정을 가능하게 하며, 기존의 방법들과 달리 외부 텍스트 감독 없이 캡션 생성에 더 관련된 시각적 특징을 효과적으로 추출할 수 있습니다. Q-Adapter는 학습 가능한 쿼리 토큰과 게이팅 레이어를 Vision Encoder에 통합하여 모델의 성능을 극대화합니다.

- **Technical Details**: Q-Adapter는 학습 가능한 쿼리 토큰과 게이팅 메커니즘을 통해 Vision Encoder에 삽입되는 경량의 어댑터 구조를 제안합니다. 이 방법은 모델이 주어진 비디오에서 어떤 시각적 특징이 캡션 생성을 위해 더 중요한지를 동적으로 학습할 수 있도록 하며, 명시적인 주석이나 외부 자료 없이도 작업 관련 시각적 특징을 효과적으로 추출할 수 있습니다. 본 연구에서는 Q-Adapter의 다양한 하이퍼 파라미터와 설계 선택들이 미세 조정 효율성에 미치는 영향을 분석합니다.

- **Performance Highlights**: Q-Adapter는 MSR-VTT와 MSVD와 같은 잘 알려진 비디오 자막 데이터셋에서 PEFT 접근 방식을 취하는 방법들 가운데 최고 성능을 기록했습니다. Q-Adapter는 전체 미세 조정 접근 방식에 비해 단 1.4%의 파라미터만을 요구하며, 캡션 품질과 파라미터 효율성을 균형 있게 유지할 수 있습니다. 이러한 결과는 Q-Adapter가 비디오-언어 모델링에 있어 확장 가능성을 보여주며, 텍스트와 비디오 간의 효과적인 정렬을 위한 강력한 잠재력을 시사합니다.



### MIMO: A medical vision language model with visual referring multimodal input and pixel grounding multimodal outpu (https://arxiv.org/abs/2510.10011)
Comments:
          CVPR 2025

- **What's New**: 본 논문에서는 MIMO라는 통합 의료 비전 언어 모델을 제안합니다. MIMO는 시각적 참조(visual referring) 멀티모달 입력과 픽셀 기초(pixeld grounding) 멀티모달 출력을 결합하여 복잡한 의료 이미지를 이해할 수 있도록 설계되었습니다. 또한, 의료 용어를 텍스트 출력의 이미지 내에서 기초화할 수 있는 기능을 제공합니다.

- **Technical Details**: MIMO는 visual prompts를 임베딩으로 모델링하여 이미징 특징과 동일한 공간에서 조정합니다. 이 과정에서 다중 모달 입력 정렬(Multi-modal Input Aligner)을 통해 서로 다른 모달리티 간의 간극을 연결합니다. MIMO는 대량 언어 모델(LLM)을 기반으로 하여 세분화 토큰을 디코드하여 기초화 마스크를 얻게 하고, 이러한 마스크는 LLM의 자연어 출력과 연관된 의미 패턴과 연결됩니다.

- **Performance Highlights**: MIMO는 895k 샘플로 구성된 종합적인 의료 멀티모달 데이터세트인 MIMOSeg로 학습되었습니다. 여러 하위 의료 멀티모달 작업에 대한 실험 결과를 통해 MIMO가 기존 모델에서 제공하지 않았던 시각적 참조와 픽셀 기초의 독특한 결합 능력을 입증했습니다. 이 연구는 의료 비전 언어 모델에서 시각적 참조와 픽셀 기초를 동시에 통합한 첫 번째 작업으로, 의료 이미지 이해와 텍스트 추론의 독특한 도전을 해결하는 데 기여합니다.



### BurstDeflicker: A Benchmark Dataset for Flicker Removal in Dynamic Scenes (https://arxiv.org/abs/2510.09996)
Comments:
          Accepted by NeurIPS 2025

- **What's New**: 이 논문에서는 짧은 노출로 촬영된 이미지에서 발생하는 플리커 아티팩트(flicker artifacts)를 해결하기 위한 BurstDeflicker라는 스케일 가능(burstable)한 벤치마크를 제안합니다. 플리커 아티팩트는 롤링 셔터 카메라(rolling shutter cameras)와 AC 전원 조명의 시간적 변동 간의 상호작용으로 발생합니다. BurstDeflicker는 세 가지 상호 보완적인 데이터 수집 전략을 통해 구성되었으며, 이는 연구자들이 플리커 제거에 대한 연구를 발전시킬 기반을 제공합니다.

- **Technical Details**: 버스트 디플리커(BurstDeflicker)에서 제안하는 주요 접근 방식 중 첫 번째는 Retinex 기반 합성 파이프라인입니다. 이 방법은 플리커 제거의 목표를 재정의하고 주요 플리커 관련 속성(예: 강도(intensity), 면적(area), 주파수(frequency))의 조절을 통해 다양한 플리커 패턴을 생성할 수 있도록 돕습니다. 두 번째로, 여러 장면에서 캡처한 4,000개의 실제 플리커 이미지를 통해 실제 플리커 아티팩트의 공간적, 시간적 특성을 이해하는 데 도움을 줍니다. 마지막으로, 움직임을 포함한 이미지 쌍을 확보하기 위해 그린 스크린(green-screen) 방법을 제안합니다.

- **Performance Highlights**: BurstDeflicker 데이터셋은 4,000개의 실제 플리커 이미지 쌍과 함께 다양한 장면에서 촬영된 3,690개 동적 이미지 쌍으로 구성되어 있습니다. 이 데이터셋은 플리커 제거 연구의 기초가 되며, 다양한 실험 결과는 제안된 데이터셋이 플리커 제거에 매우 효과적임을 보여줍니다. 우리 연구는 향후 플리커 제거 분야의 연구를 크게 발전시킬 것으로 기대됩니다.



### FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering (https://arxiv.org/abs/2510.09995)
Comments:
          Accepted by NeurIPS 2025

- **What's New**: 본 논문에서는 독창적인 플레어 데이터 생성 방법을 제안하여, 이는 3단계로 구성된다: 파라미터화된 템플릿 생성, 조명 법칙을 반영한 2D 합성, 물리 엔진 기반 3D 렌더링이다. 이를 통해 FlareX라는 새로운 혼합 플레어 데이터셋을 생성하며, 2D 및 3D 관점을 모두 포함한다. 이 데이터셋은 95개의 플레어 패턴에서 유도된 9,500개의 2D 템플릿과 60개의 3D 장면에서 렌더링된 3,000개의 플레어 이미지 쌍을 제공한다.

- **Technical Details**: 제안된 데이터 생성 프레임워크는 Blender라는 3D 물리 엔진을 기반으로 하며, 이는 실제 렌즈 플레어를 더 잘 시뮬레이트하기 위해 중요한 요소들을 파라미터화한다. 조명 법칙을 통해 플레어의 세기와 빛의 위치 간의 관계를 수립하고, 추정된 깊이 맵을 통해 합성 이미지 내의 플레어의 세기를 더 현실감 있게 나타낸다. 마지막으로, 플레어를 적절한 위치에 배치하는 특정 3D 장면을 렌더링하며, 실제 물리 법칙을 따르는 이미지 쌍을 생성한다.

- **Performance Highlights**: 제안된 방법과 데이터셋은 다양한 실험을 통해 그 효과가 입증되었다. 플레어 제거 성능 평가를 위해, 플레어가 포함된 이미지로부터 실제 플레어 없는 이미지를 얻는 마스킹 방법을 통해 모델 성능을 측정할 수 있는 가능성을 제시하였다. 이 연구는 플레어 제거 분야의 향후 연구에 기여할 것으로 예상된다.



### Scaling Traffic Insights with AI and Language Model-Powered Camera Systems for Data-Driven Transportation Decision Making (https://arxiv.org/abs/2510.09981)
- **What's New**: 이 연구는 기존의 교통 카메라 인프라를 활용하여 고해상도 및 장기적인 교통 모니터링을 가능하게 하는 AI 기반 프레임워크를 제안합니다. YOLOv11 모델을 정교하게 조정하여 실시간으로 교통 밀도와 분류 지표를 추출하며, 비정상적인 PTZ(여닫이 카메라) 카메라로 인한 일관성 없는 데이터를 해결하기 위해 새로운 그래프 기반 시점 보정 방법을 도입했습니다. 아울러, 도메인 특화 대형 언어 모델이 통합되어 연속적인 교통 패턴을 자동으로 요약하는 기능을 제공합니다.

- **Technical Details**: 이 연구는 2025년 뉴욕시의 혼잡 요금제 시행 초기 시기에 약 1,000 대의 교통 카메라에서 수집한 9백만 장 이상의 이미지를 사용하여 시스템을 검증했습니다. 그래프 기반 시점 보정 기법은 PTZ 카메라의 경첩 각도를 추정하고, 주도적 시점 그룹을 클러스터링하여 비정상적인 카메라 피드를 관리합니다. 고유한 대형 언어 모델(LLM) 모듈은 방대한 비디오 데이터를 주기적으로 해석 가능한 요약으로 변환하여 실시간 교통 데이터를 효과적으로 처리합니다.

- **Performance Highlights**: 시스템의 적용 결과로는, 혼잡 완화 지역 내 차량 밀도가 9% 감소하고 트럭 물동량이 초기 감소를 보이며 보복 현상이 나타났습니다. 보행자 및 자전거 이용자의 활동은 지속적으로 증가했습니다. 예제 기반 프롬프트를 사용한 실험은 LLM의 수치 정확도를 향상시키고 허위 진술을 줄이는 결과를 보여 주었습니다. 이 프레임워크는 대규모 정책 관련 교통 모니터링에서 인프라 호환 솔루션으로서의 잠재력을 잘 보여줍니다.



### J-RAS: Enhancing Medical Image Segmentation via Retrieval-Augmented Joint Training (https://arxiv.org/abs/2510.09953)
- **What's New**: 이번 논문은 Joint Retrieval Augmented Segmentation (J-RAS)이라는 새로운 방법론을 제안합니다. J-RAS는 이미지 분할(segmentation) 모델과 검색(retrieval) 모델을 통합하여 훈련시키는 joint-training 방식으로, 효율적인 이미지 분할을 위한 가이드 역할을 합니다. 이 접근법은 검색 모델이 특정한 분할 작업에 유익한 이미지-마스크 쌍을 학습하도록 하여, 분할 모델이 보다 풍부한 해부학적 이해(anatomical understanding)를 갖도록 합니다.

- **Technical Details**: J-RAS는 두 개의 독립적인 훈련 단계와 함께 결합된 훈련을 포함합니다. 이 방법은 검색된 유사 사례를 통해 제공된 맥락적 안내(contextual guidance)를 활용하여, 분할 모델이 예측을 개선할 수 있도록 하며, 동시에 검색 모델도 분할 작업에 따라 업데이트됩니다. 이를 통해 검색 모델은 원시 유사성(raw similarity)에 의존하지 않고 분할 관련 기능을 학습하게 됩니다.

- **Performance Highlights**: 논문에서는 J-RAS가 다양한 분할 백본(U-Net, TransUNet, SegFormer, SAM)의 성능을 향상시키는 것을 다루고 있으며, ACDC 및 M&Ms와 같은 두 개의 벤치마크 데이터셋에서 검증되었습니다. 예를 들어, ACDC 데이터셋에서 J-RAS를 적용하지 않은 SegFormer의 평균 Dice 점수는 0.8708이었으나, J-RAS를 적용한 후에는 0.9115로 크게 개선되었습니다. 이러한 결과는 J-RAS의 효과성과 아키텍처 및 데이터셋 전반에 걸친 일반화 가능성을 강조합니다.



### A Multi-Strategy Framework for Enhancing Shatian Pomelo Detection in Real-World Orchards (https://arxiv.org/abs/2510.09948)
- **What's New**: 본 논문은 상업적 요구에 부합하는 정확한 양호 감지를 위한 자동화 기술의 필요성을 강조하며, 특히 샤티안 포멜로(Shatian pomelo)의 검출에 중점을 둡니다. 기존 연구들은 특정 이론 또는 데이터셋 시나리오에 맞춘 맞춤형 네트워크를 사용했으나, 실제 환경에서는 성능 저하를 겪고 있습니다. 이 연구는 이러한 문제를 해결하기 위해 네 가지 주요 도전 과제를 식별합니다: 이미징 장치(imaging device), 조명 조건(lighting conditions), 객체 크기 변화(object scale variation), 그리고 가림(occlusion).

- **Technical Details**: 이 논문에서 제시한 다중 전략 프레임워크는 다양한 이미징 장치와 복잡한 과수원 환경에서 발생하는 색조 변화(tone variation)를 해결하기 위해 다중 시나리오 데이터셋인 STP-AgriData를 활용합니다. 또한 불규칙한 조명 조건을 시뮬레이션하기 위해 대비 조정과 밝기 변경과 같은 데이터 증강(data augmentation) 기술을 적용합니다. 마지막으로, 객체 크기 변화와 가림 문제를 해결하기 위한 REAS-Det 네트워크를 설계하였습니다. 이 네트워크에서는 RFAConv와 C3RFEM 모듈을 사용하여 수용 영역(receptive field)을 확대하고, MultiSEAM 구조와 soft-NMS를 통해 가림 문제를 해결합니다.

- **Performance Highlights**: 실험 결과, 제안된 네트워크는 87.6%의 정밀도(precision, P)와 74.9%의 재현율(recall, R), 그리고 각각 82.8%의 mAP@.50과 53.3%의 mAP@.50:.95를 기록하였습니다. 이러한 성능은 최신 검출 방법(state-of-the-art detection methods)과 비교했을 때 우수함을 보여주고 있습니다. 따라서 본 연구가 제안한 접근법은 샤티안 포멜로의 정확한 자동 검출을 위한 효과적인 솔루션으로 평가될 수 있습니다.



### Explainable Human-in-the-Loop Segmentation via Critic Feedback Signals (https://arxiv.org/abs/2510.09945)
Comments:
          Submitted to a computer vision conference (under review)

- **What's New**: 이번 연구에서는 인간 피드백을 단순한 추가 레이블로 보지 않고, 모델의 오류를 교정하기 위한 개입 신호(interventional signals)로 활용하는 새로운 접근 방식을 제안합니다. 이를 통해 단순히 더 많은 데이터를 제공하는 것이 아니라, 모델이 왜 잘못 예측했는지를 식별하고 체계적으로 수정하는 방법론을 개발하였습니다. 이러한 인간-기계 협력 방식은 예상치 못한 오류를 발견했을 때 매우 효과적입니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 메커니즘을 통합합니다: Critic Interface, Counterfactual Data Generation, Feedback Propagation입니다. Critic Interface는 사용자가 세분화 오류를 수정하고 피드백을 제공할 수 있는 시각적 편집 도구를 제공하며, Counterfactual Data Generation은 사용자 수정에 따른 대조 쌍을 생성합니다. Feedback Propagation은 수정 사항을 시각적으로 유사한 이미지에 확장하여 데이터셋 전반에 걸쳐 교정 작업을 효율적으로 진행할 수 있게 합니다.

- **Performance Highlights**: 본 연구에서는 제안한 프레임워크가 베이스라인 재교육 방법에 비해 주석 작업을 3-4배 줄이는 동시에, 도전적인 cubemap 데이터에서 최대 9 mIoU 포인트(상대적 12-15% 향상) 증가한 세분화 정확도를 얻었다고 보고하였습니다. 또한, Cityscapes, ADE20K와 같은 벤치마크 데이터셋에서도 경쟁력 있는 성능을 유지하며, 새로운 도메인에 대한 일반화 능력도 향상되었습니다.



### Semi-disentangled spatiotemporal implicit neural representations of longitudinal neuroimaging data for trajectory classification (https://arxiv.org/abs/2510.09936)
Comments:
          Accepted at the MICCAI 2025 Learning with Longitudinal Medical Images and Data Workshop

- **What's New**: 본 논문은 뇌 노화 경로를 모델링하기 위해 새로운 데이터 기반 방법을 제안합니다. 각 개인의 T1-weighted MRI 데이터를 연속 함수로 표현하기 위해 Implicit Neural Representations (INRs)을 사용하여 뇌 전체의 노화 경로를 모델링합니다. 이러한 방식은 기존의 전통적인 모델의 한계를 극복하고, 보다 정교한 분석을 가능하게 합니다.

- **Technical Details**: 이 연구는 뇌 노화를 따르는 지속적인, 개인 별 경로를 가정하여, 여러 개의 T1-weighted MRI 스캔을 사용하여 각 시간 지점에서의 공간적 변화를 모델링합니다. 각 주제에 대해 설정된 고정된 INR 아키텍처를 사용하여 경로 분류기를 개발하고, 이 분류기는 주제 별 매개변수를 입력으로 사용하여 경로를 분류합니다. 이는 네트워크 전반에 걸쳐 잔여 연결을 포함한 다층 퍼셉트론(MLP)에 의해 구현됩니다.

- **Performance Highlights**: IRN 기반 방법은 비정상적으로 샘플링된 데이터 세트에서 81.3%의 정확도를 달성하여 뇌 노화 경로를 분류하는 작업에서 표준 딥 러닝 모델을 초과합니다. 이는 기존의 이미지 기반 분석 방법보다 더 뛰어난 성능을 보여줍니다. 새로운 접근 방식은 뇌 노화에 대한 데이터 기반 이해의 발전에 기여할 것으로 기대됩니다.



### Denoising Diffusion as a New Framework for Underwater Images (https://arxiv.org/abs/2510.09934)
- **What's New**: 본 논문에서는 해양 에코시스템 연구에 필요한 고품질 수중 이미지를 확보하기 위한 새로운 접근 방식을 제안합니다. 수중 이미지의 저하된 품질을 개선하기 위해 Denoising Diffusion 모델을 활용하여 다양한 이미지 유형을 포함하는 데이터셋을 확장하며 Controlnet을 통해 이미지 품질을 향상시킵니다. 이는 해양 생물에 대한 연구와 모니터링을 보다 효과적으로 지원할 수 있게 합니다.

- **Technical Details**: 제안된 방법론은 Stable Diffusion v2.0을 기반으로 하는 다각적인 Denoising Diffusion 파이프라인으로 구성됩니다. Controlnet을 사용하여 이미지 전처리 과정을 조절하고, 사용자 프롬프트를 기반으로 특정 영역의 목표 편집을 가능하게 합니다. 이러한 접근 방식을 통해 수중 이미지의 아티팩트 제거, 색 대비 조정 및 세부 사항 향상이 이루어집니다.

- **Performance Highlights**: 제안한 방법은 수중 환경에서 발생하는 특정 문제를 해결하며, Deep Learning 모델의 학습을 위한 중요한 고품질 데이터셋 생성을 촉진합니다. 이 방법론은 연구자들이 해양 자원 조사 및 보호를 위해 고도의 해양 공학 발전에 기여할 수 있는 견고한 모델 개발을 가능하게 합니다. 최종적으로 이 연구는 기후 변화로 인한 해양 생물의 이동 패턴 이해와 관련된 실질적인 탐사 작업에 기여할 것입니다.



### HeadsUp! High-Fidelity Portrait Image Super-Resolution (https://arxiv.org/abs/2510.09924)
- **What's New**: 이번 연구에서는 포트레이트 이미지 수퍼 해상도 문제인 PortraitISR을 다루고 HeadsUp이라는 새로운 단일 단계 확산 모델을 제안합니다. 이 모델은 얼굴 지역에 중점을 두며, 고해상도 포트레이트 이미지를 복원하고 확대하는 동시에 혼합 기반 방법의 경계 아티팩트를 최소화합니다. HeadsUp은 고유한 얼굴 인식 관리 메커니즘을 통해 얼굴의 모호성을 줄이고, PortraitSR-4K라는 고품질 4K 포트레이트 ISR 데이터 세트를 구축하여 모델 학습을 지원합니다.

- **Technical Details**: HeadsUp 모델은 단일 단계 확산 모델을 기반으로 하며, 얼굴 지역에 대한 감독 메커니즘을 결합해 얼굴 부분에 초점을 맞춥니다. 복원 과정에서 혼합된 일반 이미지와 전용 얼굴 ISR 모델을 사용할 필요 없이, 자율적으로 포트레이트의 고품질 배경과 얼굴 세부 정보를 복원합니다. 모델 학습을 위한 PortraitSR-4K 데이터 세트는 고해상도 포트레이트 사진 30,000장을 포함하고 있습니다.

- **Performance Highlights**: 다양한 실험을 통해 HeadsUp이 포트레이트 ISR에서 최첨단 성능을 달성했음을 입증하였습니다. 또한, 일반 이미지 및 정렬된 얼굴 데이터셋에서도 경쟁력을 유지하며, 기존의 방법들보다 인식 품질과 충실도가 높습니다. 이 연구는 포트레이트 이미지 수퍼 해상도 분야에서 중요한 기초 자료로 활용될 것입니다.



### SpectralCA: Bi-Directional Cross-Attention for Next-Generation UAV Hyperspectral Vision (https://arxiv.org/abs/2510.09912)
Comments:
          The work consists of three chapters, includes 12 figures, 4 tables, 31 references, and 1 appendix. A version of this work has been accepted for presentation at the 2025 IEEE 8th International Conference on Methods and Systems of Navigation and Motion Control

- **What's New**: 이 연구는 UAV(무인 항공기)의 필요성이 증가하는 환경에서 신뢰성 있게 작동할 수 있는 새로운 기술을 제시합니다. 특히, 이 연구는 HSI(하이퍼스펙트럴 이미징)를 UAV 인지 시스템에 통합하기 위한 딥러닝 아키텍처의 개발을 목표로 하고 있습니다. 이는 탐색, 물체 감지 및 지형 분류와 같은 중요한 작업을 지원합니다.

- **Technical Details**: 연구 방법론은 SpectralCA 블록을 도입하여 Mobile 3D Vision Transformer(MDvT)를 수정하는 것에 기반하고 있습니다. 이 블록은 분광적(spectral) 및 공간적(spatial) 기능을 융합하기 위해 양방향 크로스 어텐션(bi-directional cross-attention)을 활용하여 정확성을 향상시키고, 매개변수와 추론 시간(inference time)을 줄입니다. 또한, 하이브리드 2D/3D 컨볼루션 아키텍처를 설계하고 기존 HSI 방법을 리뷰하는 과정도 포함됩니다.

- **Performance Highlights**: 실험 평가는 WHU-Hi-HongHu 데이터셋에서 진행되었으며, Overall Accuracy, Average Accuracy 및 Kappa 계수를 사용하여 결과를 평가했습니다. 연구 결과, 제안된 아키텍처가 UAV 인식 효율성을 향상시키고, 탐색 및 환경 모니터링 작업에서 실시간(real-time) 운영을 가능하게 함을 확인했습니다.



### An uncertainty-aware framework for data-efficient multi-view animal pose estimation (https://arxiv.org/abs/2510.09903)
- **What's New**: 이번 연구에서는 동물 행동을 정량적으로 분석하기 위한 다중 뷰 포즈 추정에서의 정확도를 높이는 새로운 프레임워크를 제시합니다. 기존의 제한된 라벨 데이터와 불확실성 추정의 문제를 해결하기 위해, 사전 학습된 트랜스포머 구조를 결합한 모델 증류(multi-view transformer, MVT) 방식으로 모든 뷰의 정보를 동시에 처리할 수 있도록 설계되었습니다.

- **Technical Details**: 본 연구의 기술적 핵심은 여러 기술의 조합을 통해 이루어지는 초기 크로스뷰 정보 융합입니다. 이를 위해 우리는 픽셀 패치를 랜덤으로 마스킹하여 강인한 크로스뷰 대응을 학습하는 패치 마스킹 방식과, 카메라 보정이 가능한 설정에서는 기하학적 일관성을 유지하는 3D 증강 기법을 적용했습니다. 이와 함께 Ensemble Kalman Smoother(EKS) 알고리즘을 비선형 버전으로 개선하여 불확실성 추정을 향상시켰습니다.

- **Performance Highlights**: 연구 결과, 제안한 방법은 세 가지 다양한 동물 종(파리, 쥐, 초리)에서 기존 방법을 능가하는 성능을 보였습니다. 특히 트랜스포머 구조와 패치 마스킹 기법을 통해 초기 융합으로 인해 단일 뷰보다 월등한 성과를 달성했습니다. 이 연구는 동물 행동 분석에서 신뢰할 수 있는 포즈 추정을 위한 실용적이고 불확실성 인식(system) 시스템의 기초를 마련하였습니다.



### LTGS: Long-Term Gaussian Scene Chronology From Sparse View Updates (https://arxiv.org/abs/2510.09881)
- **What's New**: 이 연구에서는 LTGS(Long-Term Gaussian Scene chronology)라는 새로운 장기적인 장면 표현 방법을 제안하여, 강한 구조적 프라이어를 활용하여 기존의 희소한 캡처에서 환경의 변화를 효율적으로 감지하고 업데이트할 수 있습니다. 이 방법은 일상의 동적 변화가 잦은 환경을 다루며, 이전 정보의 손실 없이 처리할 수 있는 장점이 있습니다.

- **Technical Details**: LTGS 프레임워크는 처음에 주어진 파라미터가 없는 Gaussian splatting 표현을 바탕으로 하여 장면의 장기적인 연대기를 확립합니다. 이는 객체를 템플릿 Gaussian으로 구성하여, 개별 객체에 대한 추적과 재위치 파악을 통해 장면을 업데이트하고 재구축합니다.

- **Performance Highlights**: 실제 환경에서 여러 객체의 동적 변화를 다룬 데이터셋을 수집하여 LTGS의 유용성을 평가했습니다. 실험 결과, LTGS는 다른 기존 방법들에 비해 월등히 우수한 재구성 품질을 보이며, 신속하고 경량한 업데이트를 가능하게 했습니다.



### Geometry-Aware Scene Configurations for Novel View Synthesis (https://arxiv.org/abs/2510.09880)
- **What's New**: 이번 논문에서는 불완전한 관측값으로부터 몰입형 실내 환경을 생성하기 위해 장면 적응형(scene-adaptive) 전략을 제안합니다. 실내 장면은 복잡한 레이아웃, 혼잡, 가림 및 평면 벽과 같은 다양한 요소를 포함하고 있어 자원 관리가 중요합니다. 저자들은 기하학적 사전 정보(geometric priors)를 활용하여 최적의 기초 배치를 유도하여 기존 방식보다 효율적인 장면 표현을 제공합니다.

- **Technical Details**: 연구에서는 기하학적 스카폴드(geometric scaffold)를 바탕으로 입력 관측의 통계 정보를 수집하고 기초(base)를 최적 배치합니다. 또한, 장면 기하학과 측정 통계에 기반하여 가상의 뷰포인트(virtual viewpoints)를 생성하여 불완전한 기하학적 배치를 보완합니다. 이 모든 과정에서 NeRF(Neural Radiance Fields)와 같은 신경망 기반 방법을 활용하며, 실제 실내 장면을 대상으로 성능을 검증합니다.

- **Performance Highlights**: 제안한 장면 적응형 전략은 기존 NeRF 표현 방식에 비해 상황에 맞는 기초 배치를 통해 결과 품질을 현저히 향상시킵니다. 실험 결과, ScanNet++ 및 Zip-NeRF 데이터셋을 사용하여 실제 환경에서 개선된 성능을 입증하였고, 이는 장면 기하학에 따라 효율적으로 최적화된 다양한 표현 모델에 적용 가능합니다.



### CHUG: Crowdsourced User-Generated HDR Video Quality Datas (https://arxiv.org/abs/2510.09879)
- **What's New**: 이번 연구는 UGC(사용자 생성 콘텐츠)의 HDR(High Dynamic Range) 비디오 품질 평가(VQA)를 위한 최초의 대규모 설문조사인 CHUG를 소개합니다. 기존 HDR-VQA 데이터셋은 주로 전문적으로 생성된 콘텐츠(PGC)에 초점을 맞추어 UGC의 품질 평가를 위한 중요한 이해 부족을 남겨두었습니다. CHUG 데이터셋은 856개의 UGC-HDR 소스 비디오로 구성되어 있으며, 총 5,992개의 비디오가 생성되어 다양한 해상도와 비트레이트를 통해 실제 시나리오를 시뮬레이션합니다.

- **Technical Details**: CHUG는 대규모 UGC-HDR 비디오 품질 데이터셋을 구축하기 위해 실제 HDR 비디오를 사용자로부터 수집했습니다. 이 데이터셋은 다양한 콘텐츠와 왜곡을 보장하기 위해 큐레이션되었으며, 비트레이트 사다리(비트레이트 레더) 인코딩을 적용하여 실제 스트리밍 조건을 시뮬레이션했습니다. 또한, 각 비디오는 10초로 최대 절단되며, 1080p 이상의 해상도 변경은 없었습니다.

- **Performance Highlights**: 211,848개의 인식 평가가 수행된 대규모 연구를 통해 CHUG는 UGC-HDR 비디오의 품질에 대한 신뢰성 있는 주관적 평가를 제공합니다. 이를 통해 노레퍼런스(NR) HDR-VQA 모델 개발을 지원할 수 있으며, CHUG은 UGC-HDR 왜곡을 분석하기 위한 벤치마크 역할을 합니다. 수집된 평가에서 모든 비디오에 대해 평균 35개의 평가가 이루어졌으며, 데이터의 신뢰성을 보장하기 위해 엄격한 필터링 기준이 적용되었습니다.



### Fast Self-Supervised depth and mask aware Association for Multi-Object Tracking (https://arxiv.org/abs/2510.09878)
- **What's New**: 이 논문은 다중 객체 추적(MOT)에서 IoU(Intersection-over-Union)을 사용하던 기존 방식을 대체할 새로운 접근 방식을 제안합니다. 특히, 유사한 물체나 가려진 객체의 경우 IoU가 불안정할 수 있으며, Segmentation mask의 IoU 계산이 비용이 많이 들기 때문에 깊이(depth)와 mask 특성을 융합하여 대체합니다. 이 방법은 셀프 슈퍼바이즈드(self-supervised)로 훈련된 압축된 인코더를 통해 안정적인 객체 표현을 생성하며, 이는 추가적인 유사성 신호로 작용합니다.

- **Technical Details**: 이 새로운 방법은 특정 2D 신호(예: bounding box IoU) 외에 깊이 정보와 Segmentation mask 정보를 결합하여 더 정밀한 공간적 특성을 추출합니다. 종합적으로, 이 접근은 경량화된 셀프 슈퍼바이즈드 인코더를 사용하여, 시간적으로 안정적이고 노이즈가 적은 특징을 생성합니다. 또한, Segmentation mask IoU를 계산하지 않고도 Segmentation mask를 개선할 수 있는 MOT 접근법을 최초로 제안합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법이 SportsMOT 및 DanceTrack과 같은 어려운 벤치마크에서 검증되며 기존의 방법들에 비해 향상된 성능을 보였습니다. 매우 혼잡하고 가려진 객체를 포함하는 복잡한 비선형 움직임에서도 우수한 성능을 보여주었으며, MOT17과 같은 상대적으로 간단한 벤치마크에서도 경쟁력 있는 성능을 달성하였습니다.



### Cluster-Aware Prompt Ensemble Learning for Few-Shot Vision-Language Model Adaptation (https://arxiv.org/abs/2510.09867)
Comments:
          Accepted to the journal Pattern Recognition in 2025

- **What's New**: 본 논문에서는 Vision-Language Models (VLMs)에 새로운 접근법인 Cluster-Aware Prompt Ensemble Learning (CAPEL)을 제안합니다. CAPEL은 기존의 prompt ensembling 방법들이 텍스트 특징을 평균화할 때 발생하는 문제를 해결합니다. 제안된 프레임워크는 이미지의 클래스 클러스터에 따라 각각의 분류기를 할당하며, 로짓(logits) 공간에서의 앙상블을 수행하여 시각적 특징 분포와 더 잘 맞춰집니다.

- **Technical Details**: CAPEL은 여러 클래스 클러스터를 보존하도록 설계되었으며, 각 클러스터는 고유한 프롬프트로 표현됩니다. 또한, 클러스터별 정규화 항을 도입하여 서로 다른 클러스터에 대해 프롬프트가 독립적이고 전문화된 상태를 유지할 수 있도록 합니다. 이 방법은 잘못된 프롬프트의 주의 가중치를 동적으로 조절하는 적응형 가중치 기법을 통합하여 다양한 데이터셋과 작업에서의 성능을 보장합니다.

- **Performance Highlights**: CAPEL은 11개의 데이터셋을 통한 포괄적인 실험을 통해 성능의 우수성을 입증했습니다. 특히, 몇 초 학습(few-shot learning) 및 도메인 일반화 설정에서 뚜렷한 개선을 보여주었습니다. 기존의 상태 높은 방법들과 비교했을 때, CAPEL의 접근법이 더 뛰어난 일반화 및 정확성을 보임을 확인할 수 있었습니다.



### Cell Instance Segmentation: The Devil Is in the Boundaries (https://arxiv.org/abs/2510.09848)
Comments:
          Accepted at IEEE Transactions On Medical Imaging (TMI)

- **What's New**: 이 논문에서는 Ceb(Cell boundaries)라는 새로운 픽셀 클러스터링 방법을 제안합니다. 이 방법은 세포 경계의 특징과 레이블을 활용하여 전경 픽셀을 세포 인스턴스로 나누어줍니다. 기존의 방법들이 픽셀 단위 목표를 기반으로 할 때 나타나는 기하학적 속성 손실 문제를 해결하고자 합니다.

- **Technical Details**: Ceb는 수정된 워터셰드 알고리즘을 사용하여 확률 맵에서 잠재적인 전경-전경 경계를 추출합니다. 각 경계 후보에 대해 경계 특징 표현(경계 서명)을 구성하고, 이를 통해 경계 분류기를 사용하여 이진 경계 레이블을 예측합니다. 최종적으로, 예측된 경계 레이블에 따라 이웃 분야를 나누거나 병합하여 세포 인스턴스를 생성합니다.

- **Performance Highlights**: Ceb는 여섯 가지 데이터셋에서 기존 픽셀 클러스터링 방법들보다 우수한 성과를 나타냈습니다. 또한, Ceb는 SOTA 세포 인스턴스 분할 방법들과 비교했을 때 경쟁력 있는 성능을 보였습니다. 비디오 설정에서 시간 일관성을 포함하여 성능을 더욱 향상시키는 방법을 제안합니다.



### Exploration of Incremental Synthetic Non-Morphed Images for Single Morphing Attack Detection (https://arxiv.org/abs/2510.09836)
Comments:
          Workshop paper accepted NeurIPS 2025

- **What's New**: 본 논문은 합성 얼굴 데이터(synthetic face data)를 활용하여 단일 변형 공격 탐지(S-MAD) 기술을 향상시키는 방법을 연구하였습니다. 또한 프라이버시 문제로 인해 실제 이미지를 대규모로 확보하기 어려운 한계를 극복하는 방법을 제시합니다. 다양한 변형 도구와 교차 데이터셋 평가 체계를 사용하여 연구가 진행되었습니다.

- **Technical Details**: 연구의 핵심은 '비변형(non-morphed)' 이미지라는 합성 이미지를 기존의 데이터셋에 통합하여 학습 과정에서 정교하게 조절된 양의 합성 이미지를 추가하는 것입니다. 이를 통해 S-MAD의 성과를 향상시키고자 하였습니다. 실험 결과, 합성 데이터의 무분별한 사용은 최적의 성능을 보장하지 않음을 강조합니다.

- **Performance Highlights**: 단지 합성 데이터만 활용한 경우에 가장 높은 동등 오류율(Equal Error Rate, EER)을 달성했으며, 이는 S-MAD의 운영 시나리오에서 합성 데이터에만 의존하는 것이 최선의 선택이 아니라는 것을 의미합니다. 정교하게 구성된 합성 이미지와 실제 이미지를 점진적으로 통합할 경우 일반화 성능이 개선되는 것을 보여주었습니다.



### Post Processing of image segmentation using Conditional Random Fields (https://arxiv.org/abs/2510.09833)
- **What's New**: 이번 연구에서는 위성 이미지의 분할(process)에서 발생하는 저해상도(quality) 특징 때문에 출력 이미지가 불명확함을 다룸으로써, 더 나은 선명도를 얻기 위한 조건부 무작위 장(field, CRF)의 적합성을 찾고자 하였습니다.

- **Technical Details**: 여러 종류의 CRF를 실험하고, 각 CRF가 우리의 목적에 적합한지를 검토하였습니다. 연구는 저품질 위성 이미지와 고품질 항공 사진이라는 두 가지 데이터셋에서 진행되었으며, 다양한 CRF 모델을 비교하여 최적의 결과를 도출하는 데 집중했습니다.

- **Performance Highlights**: 결과적으로, 각기 다른 접근 방식의 장단점을 비교하여 어떤 CRF가 이미지에서 최상의 성능을 발휘하는지를 보여주었습니다. 이 연구는 이미지 분할에서 CRF를 활용하는 데 있어 중요한 통찰력을 제공합니다.



### Task-Aware Resolution Optimization for Visual Large Language Models (https://arxiv.org/abs/2510.09822)
Comments:
          Accepted as a main conference paper at EMNLP 2025. 9 pages (main content), 7 figures

- **What's New**: 본 논문은 다양한 비전-언어 작업에 대한 해상도 선호도를 종합적으로 조사하여 비전-대형 언어 모델(VLLMs)의 성능 개선을 위한 새로운 접근 방식을 제안합니다. 기존의 VLLMs는 고정 해상도를 가정하여 작업 수행 시 성능 저하를 초래했는데, 이를 해결하기 위해 최적의 해상도를 결정하는 경험적 공식을 개발했습니다. 또한 기존의 VLLM을 효율적으로 재조정할 수 있는 파라미터 효율적인 미세 조정 기술을 제안합니다.

- **Technical Details**: 제안된 방법은 두 가지 주요 요소에 기반합니다: 이미지 복잡성과 모델 불확실성 변동. 첫 번째는 주어진 이미지의 내재적 복잡성을 측정하며, 두 번째는 다른 해상도에서의 모델 예측 불확실성의 변동성을 측정합니다. 이러한 요소를 통합한 경험적 공식을 통해 각 비전-언어 작업에 필요한 최적 해상도를 정량적으로 결정합니다. 또한, 경험적 연구를 통해 이 공식의 유효성을 검증했습니다.

- **Performance Highlights**: 본 방법은 기존 VLLM 체크포인트를 기반으로 후 훈련(post-training) 전략을 통해 해상도를 확장합니다. 실험 결과, 제안된 파라미터 효율적인 미세 조정 방식이 성능 저하 없이 효율적인 해상도 조정을 가능하게 하며, 다양한 비전-언어 작업에서 그 효과를 입증했습니다. 최적 해상도를 식별한 후, 신뢰할 수 있는 성능 향상을 위해 몇 가지 파라미터만을 업데이트하여 뛰어난 효율성-성능 균형을 달성했습니다.



### Towards Understanding Ambiguity Resolution in Multimodal Inference of Meaning (https://arxiv.org/abs/2510.09815)
Comments:
          Accepted to International Conference on Development and Learning (ICDL) 2025

- **What's New**: 이 논문은 다중모드( мультимодальный) 맥락에서 외국어를 학습하는 새로운 환경을 조사합니다. 학습자는 이미지와 결합된 문장에서 생소한 단어의 의미를 유추해야 합니다. 연구를 통해 특정 데이터에서 성공적인 외국어 학습에 필요한 최적의 모호성(ambiguity) 수준을 이해하는 것이 목표입니다.

- **Technical Details**: 연구는 인간 참가자를 대상으로 이미지-텍스트 쌍을 사용하여 진행되었습니다. 참가자들이 마스크 처리된 단어의 의미를 유추하는 데 영향을 미치는 데이터(이미지 및 텍스트)의 특성과 참가자의 언어 배경과의 상관성을 분석합니다. 또한 AI 시스템이 참가자 성과를 추론하는 능력을 평가하고, 이를 통해 향후 방향성을 제시합니다.

- **Performance Highlights**: 모호성에 대한 내성(tolerance)은 외국어 학습 성공과 밀접한 관련이 있습니다. 이 연구는 AI 시스템이 예시의 난이도를 예측하고 적절히 조정하는 모델을 구현할 가능성을 탐색하며, 이를 통해 학습자의 반응에 기반한 적시에 맞춤형 지원을 제공할 수 있을 것으로 기대합니다.



### Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping (https://arxiv.org/abs/2510.09741)
- **What's New**: 최근 연구에서는 멀티모달 대형 언어 모델(MLLMs)이 복잡한 장면에서 작은 세부사항이나 공간적 관계를 놓치는 문제를 해결하기 위한 방법을 제안합니다. 새로운 방법인 AttWarp는 쿼리와 관련된 콘텐츠에 더 많은 해상도를 할당하고, 덜 중요한 영역을 압축하여 글로벌 컨텍스트를 유지하는 경량화된 방식을 도입했습니다. 이 방법은 모델의 가중치나 아키텍처를 변경하지 않고도 비주얼 정보를 비균일하게 재분배하여 원본 이미지의 모든 정보는 유지하면서도 세부 사항을 더욱 쉽게 인식할 수 있도록 도와줍니다.

- **Technical Details**: AttWarp는 MLLM의 교차 모달 어텐션(cross-modal attention)을 활용하여 입력 이미지를 직선적으로 왜곡(rectilinear warping) 처리하는 방법입니다. 입력 이미지와 쿼리를 기준으로 교차 모달 어텐션 맵을 추출한 후, 이를 통해 주의 점수 매트릭스(Attention Score Matrix)를 생성하고 이를 기반으로 높이와 너비에 따라 각각의 중요도를 수치화합니다. 이 프로파일들을 바탕으로 상대적으로 더 높은 중요도를 가진 영역을 확장하고 낮은 중요도를 가진 영역은 압축하는 왜곡 과정을 통해 시각 정보를 처리합니다.

- **Performance Highlights**: AttWarp는 TextVQA, GQA, DocVQA, POPE, MMMU를 포함한 다섯 가지 벤치마크에서 반복적으로 정확도를 향상시키며, 조합적 추론(compositional reasoning)을 강화하고 환각(hallucination)을 줄이는 데 성공했습니다. 네 가지 경쟁적인 기준선(baselines)보다 뛰어난 성능을 보여주었으며, 다양한 MLLM 백본(backbone)과 어텐션 소스에서의 일반화 가능성도 확인되었습니다. 또한, AttWarp-Chain과 AttWarp-Distill 등의 방법론을 통해 성능을 더욱 향상시킬 수 있는 가능성을 제시하였습니다.



### Multi Camera Connected Vision System with Multi View Analytics: A Comprehensive Survey (https://arxiv.org/abs/2510.09731)
- **What's New**: 이번 논문은 Connected Vision Systems (CVS)에 대한 최초의 포괄적이고 통합된 리뷰를 제공하며, multi-view multi-camera (MVMC) 추적, re-identification (Re-ID), 행동 이해 (AU)를 단일 프레임워크로 통합합니다. 기존 연구들은 개별 작업에 집중해 왔으나, 이 논문은 그러한 작업들의 연계성을 강조하면서 CVS의 주요 구성 요소를 네 가지로 나누어서 이해를 돕고 있습니다. 또한, 최신 데이터셋, 방법론, 결과 및 평가 지표를 체계적으로 정리하여 해당 분야의 발전을 명확히 보여줍니다.

- **Technical Details**: CVS는 여러 시각 센서가 동기화되어 시각 데이터를 캡처, 분석 및 공유하는 분산 프레임워크를 의미합니다. MVMC 추적은 싱글 카메라 시스템을 넘어서서 중첩된 시점을 통해 대규모 카메라 네트워크에서 동작하며, 다양한 환경 변수 속에서도 관측 대상을 지속적으로 추적합니다. 논문에서는 MVMC의 작업들이 어떻게 상호 작용하며 강력한 분석 파이프라인을 형성하는지를 설명하고, 분석을 위한 구조적 세분화를 제공합니다.

- **Performance Highlights**: 이번 리뷰는 CVS가 다양한 실제 응용에서 중요한 역할을 수행하도록 만드는 능력을 강조합니다. 연구 결과, CV 시스템에서 MVMC 기술은 행동 분석, 이상 탐지 및 예측 모델링을 포함한 높은 수준의 의미론적 이해로 발전해 왔습니다. 최신 기술인 lifelong learning, 개인정보 보호 및 federated learning과 같은 새로운 기술이 다루어지며, 이는 CVS의 미래 발전을 위한 필수 요소로 지목되고 있습니다.



### Adaptive Fusion Network with Temporal-Ranked and Motion-Intensity Dynamic Images for Micro-expression Recognition (https://arxiv.org/abs/2510.09730)
- **What's New**: 이번 논문에서는 미세 표정 인식(MER)을 위한 새로운 방법론이 제안되었습니다. 첫째로, 시간적 진행을 강조하는 Temporal-ranked dynamic image와 미세한 움직임을 강조하는 Motion-intensity dynamic image라는 두 가지 보완적인 표현을 제안합니다. 둘째로, 이러한 두 가지 표현을 최적화하여 통합하는 Adaptive fusion network(AFN)를 도입하여, 미세 표정의 특성을 강조하고 노이즈를 억제합니다.

- **Technical Details**: 제안된 Temporal-ranked dynamic image는 ME가 포함된 프레임을 정렬하여, ME의 미세하고 일시적인 움직임을 모델링하는 데 도움을 줍니다. 반면 Motion-intensity dynamic image는 각각의 프레임과 apex 프레임 간의 Optical Flow를 계산하여 움직임의 강도를 가중치로 부여합니다. AFN은 이러한 두 이미지 표현의 상호작용을 촉진하는 적응형 융합 메커니즘을 사용하여 정보를 통합합니다.

- **Performance Highlights**: 세 가지 벤치마크 데이터셋(CASME-II, SAMM, MMEW)에서 실험한 결과, 제안된 방법은 뛰어난 성능을 보였습니다. 특히, CASME-II에서 AFN은 93.95%의 정확도와 0.897의 UF1 점수를 기록하며 새로운 최고 기록을 세웠습니다. 또한 SAMM에서는 82.47%의 정확도와 0.665의 UF1 점수로 클래스 간 균형 잡힌 인식을 보여주었으며, MMEW에서는 76.00%의 정확도를 달성하여 일반화 능력을 확인했습니다.



### NNDM: NN_UNet Diffusion Model for Brain Tumor Segmentation (https://arxiv.org/abs/2510.09681)
- **What's New**: NNDM (NN_UNet Diffusion Model)은 MRI에서 뇌 종양을 정확하게 탐지하고 세분화하기 위해 설계된 새로운 하이브리드 프레임워크입니다. 이 시스템은 NN-UNet의 강력한 특징 추출과 확산 확률 모델의 생성 능력을 결합하여 종양 경계 명확성을 개선합니다. NNDM은 입력 세그멘테이션 마스크를 점진적으로 정제하여 구조적 불일치를 수정하며, 이를 통해 기존의 U-Net 및 변환기 베이스라인보다 더 나은 성능을 보여줍니다.

- **Technical Details**: NNDM은 NN-UNet 아키텍처와 조건부 확산 모델을 결합하여 초기 세그멘테이션 마스크를 생성합니다. 이 과정에서 모델은 예측 값과 실제 값 간의 잔차 오류 분포를 학습하며, 굵은 경계 분할을 위한 향상을 이룹니다. 전체 목표는 세그멘테이션 정확도와 확산 일관성을 결합한 최종 목표를 최적화하여 높은 정밀도의 뇌 종양 세분화를 이끌어내는 것입니다.

- **Performance Highlights**: BraTS 2021 데이터셋에서 NNDM은 Dice 계수 및 Hausdorff 거리 메트릭에서 기존 U-Net 및 GAN 기반 모델보다 뛰어난 성능을 기록했습니다. 이 결과는 NNDM이 다양한 MRI 모달리티와 종양 하위 영역에서 강력한 경향성을 갖추고 있음을 입증합니다. 이러한 실험 결과는 뇌 종양 분석의 자동화에서 상태를 한 단계 발전시킬 가능성을 보여줍니다.



### Knowledge-Aware Mamba for Joint Change Detection and Classification from MODIS Times Series (https://arxiv.org/abs/2510.09679)
- **What's New**: 이 논문은 MODIS(Moderate Resolution Imaging Spectroradiometer) 시계열을 활용한 변화 탐지를 위한 새로운 접근 방식인 KAMamba(Knowledge-aware Mamba)를 제안합니다. 주요 기여는 클래스 전환에 대한 지식을 활용하기 위한 지식 기반 전환 손실(KAT-loss) 도입입니다. 또한, 다중 작업 학습 접근 방식을 통해 모델의 제약 조건을 개선하고, MODIS 시계열 정보에서 정보 간섭을 분리하기 위해 새로운 SSTMamba 모듈을 설계했습니다.

- **Technical Details**: KAMamba는 변화 탐지에서 두 가지 주요 목표를 가지고 있습니다: 클래스 조건부 전환 지식을 인코딩하여 전환 바이어스를 줄이고, 공간, 시간 및 스펙트럼 요인을 분리하여 진정한 변화를 계절성과 구분하는 것입니다. 이를 위해 KAT-loss와 세 가지 손실(PreC-loss, PostC-loss, Chg-loss)을 결합하여 클래스 구별성을 향상시키고, MODIS 신호의 동적 토큰 가지치기 및 변형 상태 전환을 통해 계산 비용을 줄이는 SDMamba 백본을 사용합니다.

- **Performance Highlights**: 실험 결과로 Saskatchewan, Canada의 MODIS 시계열 데이터셋에서 제안한 방법이 기존 기준선보다 변화 탐지에 대해 평균 1.5-6%의 F1 점수 개선을, LULC(Land Use Land Cover) 분류에서 약 2%의 OA(Overall Accuracy), AA(Average Accuracy), Kappa 점수 개선을 보였습니다. 이는 KAMamba의 효과성을 입증하며 향후 환경 모니터링 및 자원 관리에 기여할 것으로 기대됩니다.



### OmniSAT: Compact Action Token, Faster Auto Regression (https://arxiv.org/abs/2510.09667)
- **What's New**: 본 논문에서 제안하는 Omni Swift Action Tokenizer(OmniSAT)는 기존의 Auto-regressive 모델들의 효율성을 증진하기 위해 개발되었습니다. OmniSAT는 고품질의 압축을 통해 긴 시퀀스를 효과적으로 단축시키고, B-Spline 인코딩을 통해 일관된 비주얼-언어-액션 매핑을 용이하게 합니다. OmniSAT는 또한 인간의 행동 패턴을 로봇 시연에 혼합함으로써 크로스-엠바디먼트 학습 전략을 도입하여 일반화 가능성을 높였습니다.

- **Technical Details**: OmniSAT는 연속적인 행동을 압축하여 고정 길이의 토큰 리스트로 인코딩하며, 위치, 회전 및 그리퍼의 세부 사항을 각각 다루는 다중 단계 잔여 양자화를 적용합니다. 이 과정은 코드북 인덱스를 생성하여, 주어진 프로그램에 대해 통일된 액션 패턴을 참조할 수 있게 합니다. OmniSAT는 대규모 데이터 세트 Droid에서 사전 훈련되어, 37.68배의 압축을 달성하면서 밀리미터 수준의 재구성 정밀도를 유지합니다.

- **Performance Highlights**: OmniSAT는 실제 로봇 및 다양한 시뮬레이션 실험에서 기존 방법들에 비해 높은 압축 비율과 낮은 재구성 오류를 달성했습니다. Auto-regressive 훈련에 적용되었을 때, 단축된 시퀀스 길이는 더 빠른 수렴과 성능 향상으로 이어졌습니다. 이 연구에서의 중요한 기여는 OmniSAT를 통한 효율적이고 효과적인 AR 훈련을 가능하게 하여, 다양한 실험 환경에서 지속적인 성과 향상을 입증한 것입니다.



### TreeNet: Layered Decision Ensembles (https://arxiv.org/abs/2510.09654)
- **What's New**: 본 논문에서는 TreeNet이라는 새로운 계층적 결정 앙상블 학습 방법론을 소개하며, 이는 의료 영상 분석을 위해 설계되었습니다. TreeNet은 신경망, 앙상블 학습, 트리 기반 결정 모델의 중요한 기능을 통합하여 구축되었습니다. 이 모델은 복잡한 머신 러닝 작업에서 우수한 성능을 제공할 수 있는 강력하고 적응 가능한 솔루션으로 자리잡고 있습니다.

- **Technical Details**: TreeNet은 각 레이어마다 결정 트리 앙상블을 사용하는 층별 처리 방식을 채택하고 있으며, 이 방식은 잔여 신경망의 순 전파 과정을 모방합니다. 이 연구에서는 정확도(Accuracy), 정밀도(Precision), 재현율(Recall) 및 훈련과 평가 시간을 기준으로 한 주요 메트릭을 사용하여 평가가 이루어졌습니다. 훈련 데이터의 50%를 사용할 경우 F1 점수는 0.77로 나타났으며, 이는 데이터 절감에 따른 성능 저하를 보여줍니다.

- **Performance Highlights**: 제안된 방법론은 Kvasir V1, Kvasir V2, Hyper Kvasir의 벤치마크 데이터셋에서 효과적으로 적용되어, 각각 0.75, 0.78, 0.82의 F1 점수를 달성했습니다. 특히, Kvasir V1에서는 훈련 시간이 40분 이내로 단축되었으며, CPU 전용 기기에서도 작동이 가능합니다. TreeNet은 실시간 응용 프로그램에서도 30 프레임/초의 속도를 보여줍니다.



### Ultralytics YOLO Evolution: An Overview of YOLO26, YOLO11, YOLOv8 and YOLOv5 Object Detectors for Computer Vision and Pattern Recognition (https://arxiv.org/abs/2510.09653)
Comments:
          16 pages, 5 Tables, 5 Figures

- **What's New**: YOLO26 (2025)의 발표는 Ultralytics YOLO 모델의 최신 발전을 나타냅니다. YOLO26은 Distribution Focal Loss (DFL) 제거와 Non-Maximum Suppression (NMS) 없는 추론을 기본으로 설계되어, 여러 작업을 동시에 처리할 수 있는 다기능 모델로 진화했습니다. 이 혁신은 저전력 및 임베디드 장치에서의 배포 가능성을 크게 향상시켰습니다.

- **Technical Details**: YOLO26에서의 주요 기술적 변화는 두 가지로 요약됩니다. 첫째, DFL 기반의 분포 회귀를 경량의 파라미터화로 대체하여 모델의 복잡성을 줄였습니다. 둘째, NMS 없는 엔드 투 엔드 인퍼런스 경로로 수정하여 전통적인 대기 시간 병목 현상을 제거하고, 시나리오 특화 조정이 필요 없는 예측 생성을 가능하게 했습니다.

- **Performance Highlights**: 객체 탐지, 인스턴스 분할, 분류, 자세 검출 및 방향성 바운딩 박스 탐지를 하나의 통합된 모델로 실현합니다. ProgLoss와 STAL 기술의 도입은 훈련의 안정성을 높이고, 소형 객체에 대한 인식을 개선하는 데 기여합니다. 이 모델은 MS COCO 데이터셋에 대한 벤치마크 결과에서 높은 정확성과 효율성을 입증했습니다.



### TinyViT-Batten: Few-Shot Vision Transformer with Explainable Attention for Early Batten-Disease Detection on Pediatric MRI (https://arxiv.org/abs/2510.09649)
Comments:
          8 pages, 3 figures, 1 table. Submitted to International Conference on Computational Intelligence and Sustainable Engineering Solutions (CISES)

- **What's New**: 이번 논문에서는 Batten 질병(신경 세포 회백질 침착증)을 조기에 진단하기 위한 새로운 AI 모델인 TinyViT-Batten을 소개합니다. 이 모델은 소량의 학습 데이터로도 몇 가지 사례를 통해 훈련될 수 있도록 설계되었습니다. 특히, 획기적으로 간과되기 쉬운 MRI 신호를 감지하는 데 초점을 맞추고 있으며, 실질적으로 신속하고 정확한 진단이 가능합니다.

- **Technical Details**: TinyViT-Batten은 몇 가지 샷(few-shot) Vision Transformer (ViT) 프레임워크를 사용하여 만듭니다. 이 모델은 대형 Teacher ViT에서 5M 파라미터로 효율적으로 압축되어 훈련됩니다. Prototypical loss를 활용한 metric 기반의 few-shot learning으로 미세 조정이 이루어져, 다양한 데이터셋에서 배우지 못한 사례를 효과적으로 인식할 수 있습니다.

- **Performance Highlights**: 이 모델은 79개의 유전적으로 확인된 Batten 질병 MRI에서 약 91%의 높은 정확도와 0.95 이상인 ROC 아래 면적을 달성하였습니다. 또한, Grad-CAM을 통합하여 질병과 관련된 뇌 영역을 강조 표시함으로써 해석 가능한 예측을 가능하게 합니다. 이 작은 크기로도 뛰어난 성능을 보여주며, Batten 질병 조기 탐지를 위한 실용적인 AI 솔루션으로 자리잡고 있습니다.



### Adversarial Attacks Leverage Interference Between Features in Superposition (https://arxiv.org/abs/2510.11709)
- **What's New**: 이 논문에서는 적대적 예제(adversarial examples)가 신경망에서 언제, 왜 발생하는지에 대한 기존의 두 가지 관점을 극복하려는 시도를 합니다. 저자들은 적대적 취약성이 신경망의 효율적인 정보 인코딩에서 기인할 수 있다는 새로운 주장을 제시하고, 중첩(superposition)이라는 개념을 통해 적대적 공격의 패턴이 어떻게 예측 가능한지를 설명합니다. 특히, 중첩은 신경망이 더 많은 특성을 표현할 수 있게 하여 공격자가 이를 이용할 수 있는 구조를 만들어낸다고 주장합니다.

- **Technical Details**: 이 논문은 선형 표현 가설(linear representation hypothesis, LRH)과 중첩(superposition) 이론을 기반으로 신경망이 입력의 의미적 특성을 선형 방향으로 나타낸다고 설명합니다. 이를 통해 특정한 데이터 속성이 어떻게 중첩 지오메트리(specific superposition geometries)를 유도하고, 이들이 적대적 변동(adversarial perturbations)을 실현하는 데 직접적으로 작용하는지를 보여줍니다. 또한, 적대적 공격의 전이성(transferability) 및 클래스별 취약성(class-specific vulnerability) 패턴에 대한 통찰을 제공합니다.

- **Performance Highlights**: 이 연구는 CIFAR-10 데이터셋을 기반으로 훈련된 비전 트랜스포머(ViT) 모델에서 발견된 패턴도 일관되게 지속된다는 것을 보여줍니다. 저자들은 중첩이 적대적 취약성을 유도하는 데 충분하지만 필수적이지는 않다고 결론짓고, 이를 통해 알고리즘의 취약성(vulnerability mechanism)을 새롭게 정의합니다. 이러한 이해는 사전적으로 정보를 기반으로 한 안전한 방어 기법의 개발에 기여할 것입니다.



### QeRL: Beyond Efficiency -- Quantization-enhanced Reinforcement Learning for LLMs (https://arxiv.org/abs/2510.11696)
Comments:
          Code is available at this https URL

- **What's New**: QeRL은 대규모 언어 모델(LLMs)을 위한 양자화 기반 강화 학습(Quantization-enhanced Reinforcement Learning) 프레임워크를 제안합니다. 이 프레임워크는 NVFP4 양자화와 저차원 적응(LoRA)을 결합하여 강화 학습의 롤아웃 단계를 가속화하고 메모리 사용량을 줄입니다. 실험 결과, QeRL은 롤아웃 단계에서 1.5배 이상의 속도 향상을 달성했으며, 단일 H100 80GB GPU에서 32B LLM의 훈련을 가능하게 한 첫 번째 프레임워크로 자리 잡았습니다.

- **Technical Details**: QeRL은 NVFP4 양자화를 기반으로 하고, 롤아웃 및 사전 채우기(pre-filling) 단계에서 Marlin 기반 접근 방식을 통합합니다. 양자화 소음(quantization noise)이 정책 엔트로피(policy entropy)를 향상시켜 탐색을 지원하도록 동적으로 조정되는 적응형 양자화 소음(AQN) 메커니즘을 도입합니다. 이 프레임워크는 채널별 무작위 소음을 주입하고, 매트릭스 곱셈(multiplication) 전에 정밀한 제어를 통해 효율적인 활용이 가능합니다.

- **Performance Highlights**: QeRL은 vanilla LoRA보다 훈련 속도와 보상 성능 모두에서 우수성을 입증하였습니다. 예를 들어, Qwen2.5-7B-Instruct 모델에서 GSM8K 점수 90.8을 기록하며 기존의 16비트 LoRA 및 QLoRA를 초과하는 성능을 제공했습니다. 끝까지(end-to-end) 훈련에서 QLoRA 대비 약 1.8배의 속도 향상을 보였으며, MATH 500에서 풀 파라미터 미세 조정(full fine-tuning)과 동등한 정확도를 기록했습니다.



### Scaling Language-Centric Omnimodal Representation Learning (https://arxiv.org/abs/2510.11693)
Comments:
          NeurIPS 2025

- **What's New**: 최근의 다중 모달 임베딩 접근 방식은 대조 학습(Contrastive Learning, CL)으로 미세 조정된 다중 모달 대형 언어 모델(Multimodal Large Language Models, MLLMs)을 활용하여 유망한 결과를 보여주고 있습니다. 그러나 이러한 접근 방식의 우수성의 기저에 있는 이유는 아직 충분히 탐구되지 않았습니다. 이 연구는 생성적 사전 학습(Generative Pretraining) 중에 암묵적인 크로스 모달 정렬(Cross-modal Alignment)을 달성하는 것이 MLLM 기반 접근 방식의 핵심 이점이라고 주장합니다.

- **Technical Details**: 우리의 연구는 MLLM의 임베딩 공간 패턴을 조사하고 미세 조정 전후의 비대칭성(Anisotropy)과 커널 유사성(Kernel Similarity) 구조를 분석합니다. 우리는 LCO-Emb라는 언어 중심의 범모달 임베딩 프레임워크를 제안합니다. 이 프레임워크는 언어 중심의 데이터 쌍을 활용해 CL을 경량 정제 단계로 작용하도록 하여, 적은 계산 자원으로 MLLM의 기존 생성 능력을 최대화합니다.

- **Performance Highlights**: LCO-Emb는 다양한 백본 모델과 벤치마크 실험을 통해서 언어 중심의 데이터 세트만으로 훈련된 기존의 모달 임베딩 모델을 초월하는 성능을 달성하고 있습니다. 더욱이, 대조적 세분화(Contrastive Refinement)를 통해 얻게 되는 다중 모달 표현 능력은 MLLM의 생성 능력과 긍정적인 상관관계를 가진다는 새로운 세대-표현 확장 법칙(Generation-Representation Scaling Law, GRSL)을 발견했습니다. 이러한 발견은 다중 모달 표현의 향상을 위한 생성 능력 강화의 중요성을 알립니다.



### SCOOP'D: Learning Mixed-Liquid-Solid Scooping via Sim2Real Generative Policy (https://arxiv.org/abs/2510.11566)
Comments:
          Project page is at this https URL

- **What's New**: 본 논문에서는 복잡한 도구-객체 상호 작용을 고려한 로봇 스쿠핑(scooping) 기술을 제안합니다. SCOOP'D라는 새로운 방법을 통해, 다양한 시나리오에서 실제 아이템을 스쿠핑하는 데 있어 뛰어난 성능을 보입니다. 이 방법은 OmniGibson 시뮬레이터를 활용하여 6,480개의 데모를 수집하고, 사전 학습된 상태 정보를 통해 스쿠핑 능력을 학습합니다.

- **Technical Details**: SCOOP'D는 두 개의 Diffusion Policy 모델로 구성되어 있습니다. 하나는 스쿠핑 전에 적절한 ladle 포즈를 얻기 위한 초기 모델이며, 다른 하나는 세밀한 스쿠핑 동작을 학습합니다. 이 방법은 고급 시각 기반 모델(SAM2)을 활용하여, 입체적이면서도 복잡한 동작을 신속하게 학습할 수 있도록 합니다.

- **Performance Highlights**: 이 연구는 465회의 실제 테스트에서 80% 이상의 성공률을 기록하며, 'Level 1' 난이도의 물체들을 대상으로 성과를 내고 있습니다. 또한, 결과는 기존 방법들과 비교하여 월등히 우수하며, 다양한 상황에서 일반화된 성능을 보이고 있습니다.



### Evaluating Reasoning Faithfulness in Medical Vision-Language Models using Multimodal Perturbations (https://arxiv.org/abs/2510.11196)
- **What's New**: 이 논문은 시각-언어 모델(VLMs)의 의료 분야 적용과 관련하여 CoT(Chain-of-Thought) 설명의 신뢰성을 평가하는 새로운 프레임워크를 제안합니다. 이 연구는 흉부 X선 영상 질의 응답(VQA)에서 설명의 신뢰성을 검사하는 임상적으로 근거가 있는 접근 방식을 통해 설명의 신뢰관계가 어떻게 변하는지를 분석합니다. 이를 통해 의료 분야에서의 VLM의 신뢰도를 분석할 수 있는 중요한 기준을 제공합니다.

- **Technical Details**: 연구에서는 세 가지 축인 임상 신뢰도(clinical fidelity), 인과 귀속(causal attribution), 신뢰도 보정(confidence calibration)을 통해 VLM의 설명을 평가합니다. 제시된 방법론은 정밀한 입력의 교란을 사용하여 모델이 결과 flips에 적절한 이유를 제시하는지 여부를 테스트합니다. 전체 연구 과정에서 6개의 VLM을 벤치마킹하여, 설명 품질이 최종 답변 정확성과 분리되어 있음을 밝힙니다.

- **Performance Highlights**: 연구 결과, 임상 데이터와의 일관성이 낮거나 인과 귀속 점수가 낮은 모델들이 발견되었습니다. 다양한 입력 수정에 대한 테스트에서 개방형 소스 모델이 최종 답변의 정확도에서는 비슷하지만, 인과 귀속 측면에서는 상용 모델이 더 높은 점수를 기록합니다. 이러한 결과는 의료 분야에서 VLM을 배치할 때의 위험과 더불어 최종 답변 정확도 외의 더 폭넓은 평가 기준의 필요성을 강조합니다.



### Generalisation of automatic tumour segmentation in histopathological whole-slide images across multiple cancer types (https://arxiv.org/abs/2510.11182)
- **What's New**: 본 연구는 병리학적 이미지의 일반적인 종양 분할 모델을 개발하여 다양한 암 유형에서의 성능을 검토하는 것을 목표로 하였습니다. 종양 세분화(tumour segmentation)를 자동화함으로써 병리학자에게 도움을 줄 수 있는 가능성을 보여줍니다. 기존 모델들이 주로 단일 암 유형에 국한되었던 반면, 본 모델은 4,000명 이상의 환자에서 수집한 20,000개 이상의 전체 슬라이드 이미지(whole-slide images)를 통해 훈련되었습니다.

- **Technical Details**: 모델은 20,270개의 WSIs를 사용하여 개발되었고, 여러 암 유형에 대한 세분화 성능을 평가하기 위해 다양한 스캐너에서 획득한 이미지가 포함되었습니다. Dice 유사도 계수(Dice similarity coefficient, DSC)를 통해 자동 세분화 성능을 평가하였으며, 평균 DSC는 82%에서 94% 사이로 확인되었습니다. 특히 자궁 내막암(endometrial carcinoma)에 대해 90% 이상의 높은 성능을 기록하였습니다.

- **Performance Highlights**: 모델의 세분화 성능은 수작업 세분화와 비교하였고, 전체적으로 높은 민감도(sensitivity)와 특이도(specificity) 결과를 보였습니다. 예외로 방광암(bladder cancer)에서 조기 단계의 종양 변종 시료가 약간 낮은 성능을 보였으나, 대부분의 다른 암 유형에서는 만족스러운 결과를 얻었습니다. 이 연구는 다양한 환자 집단과 슬라이드 스캐너에서의 보편적인 종양 세분화가 가능하다는 것을 입증하였습니다.



### Lightweight Facial Landmark Detection in Thermal Images via Multi-Level Cross-Modal Knowledge Transfer (https://arxiv.org/abs/2510.11128)
- **What's New**: 이 논문에서 제안하는 Multi-Level Cross-Modal Knowledge Distillation (MLCM-KD) 프레임워크는 RGB에서 열화상으로의 지식 이전을 효과적으로 분리하여 열상 얼굴 랜드마크 탐지 모델을 보다 정확하고 효율적으로 만듭니다. 특히, Dual-Injected Knowledge Distillation (DIKD) 방법론을 통해 RGB와 열화상 간의 심층적인 의미적 일치를 이끌어냅니다. 이 접근법은 단순한 일방향 지식 주입의 한계를 극복하고, 양방향 메커니즘으로 교수-학생 모델 간의 관계를 개선합니다.

- **Technical Details**: MLCM-KD는 두 개의 계층 수준으로 구성되어 있습니다. 첫 번째인 Knowledge Transfer Level (KTL)에서는 RGB 교사 네트워크로부터 열학생 네트워크로 Landmark-specific 지식을 전달하며, 두 번째인 Model Compression Level (MCL)에서는 열화상 입력을 이용한 모델 압축을 수행합니다. DIKD 메커니즘은 각 계층에서 구조적 특성과 세부적 특징을 묘사하면서, 학생 네트워크가 RGB 교사의 구조적 특징을 모방하도록 강제합니다.

- **Performance Highlights**: 실험 결과, 이 방법은 여러 공공 데이터셋에서 새로운 최첨단 성능을 달성하였으며, 기존의 전통적인 일방향 지식 증류 및 이미지 변환 방법들에 비해 정확성과 효율성에서 모두 크게 개선되었습니다. MLCM-KD를 통해 생성된 경량 모델은 극도의 저조도 환경 및 자원 제약이 있는 시나리오에서도 높은 성능을 유지하고 있어, 실제 응용에서 강력한 가치를 보여줍니다.



### The Easy Path to Robustness: Coreset Selection using Sample Hardness (https://arxiv.org/abs/2510.11018)
- **What's New**: 이 논문에서는 데이터 중심(data-centric) 관점에서 적대적(Adversarial) 강인성을 높이는 방법으로 EasyCore라는 새로운 코어셋(selection) 선택 알고리즘을 제안합니다. EasyCore는 낮은 평균 입력 기울기 노름(Average Input Gradient Norm, AIGN)을 가진 샘플만을 선택하여 학습에 사용하며, 여기서 AIGN은 샘플의 적대적 취약성을 추정하는 지표로 활용됩니다. 실험 결과 EasyCore로 선택한 데이터를 기반으로 학습된 모델이 상대적으로 높은 적대적 정확도(adversarial accuracy)를 달성함을 보였습니다.

- **Technical Details**: EasyCore는 훈련 과정에서 주어진 샘플의 AIGN을 기반으로 내구성(resilience)이 높은 샘플을 선택하는 방법론입니다. AIGN은 샘플의 기울기(norm)의 평균을 구하여 샘플의 학습 용이성을 정량화하며, 이는 적대적 공격에 대한 취약성과 밀접한 연관이 있습니다. 또한, EasyCore는 기존의 코어셋 방법들과 비교했을 때 효율적이고 모델에 구애받지 않으며, 특정 데이터셋에 대해 한번만 AIGN을 계산하면 되기 때문에 계산 비용을 낮추는 장점이 있습니다.

- **Performance Highlights**: EasyCore를 통해 표준 훈련과 TRADES 적대적 훈련 각각에서 최대 7% 및 5%의 적대적 정확도 향상을 이룰 수 있음을 보였습니다. 기존의 코어셋 선택 방법에 비해 더 높은 성능으로, 특히 불확실한 샘플을 선택하는 기존 방법들의 한계를 극복했습니다. 이 연구는 데이터 중심의 접근 방식이 어떻게 모델의 적대적 강인성을 개선할 수 있는지를 실증적으로 보여줍니다.



### Into the Unknown: Towards using Generative Models for Sampling Priors of Environment Uncertainty for Planning in Configuration Spaces (https://arxiv.org/abs/2510.11014)
Comments:
          Under Review

- **What's New**: 이번 연구에서는 환경 불확실성을 캡처할 수 있는 확률적 우선순위(prior)를 제공하는 샘플링 기반 파이프라인을 소개합니다. 대규모 사전 훈련된 생성 모델을 활용하여 제로샷(zero-shot) 방식으로 공간적-의미적 관계를 포착합니다. 이 접근법은 부분 관찰에 기초하여 RGB-D 포인트 클라우드(point cloud)를 복원하여 로봇의 구성 공간 계획에 직접 사용할 수 있도록 만들어졌습니다.

- **Technical Details**: 로봇은 부분적으로만 보이는 환경에서 작업하기 때문에, 생성 모델을 사용하여 샘플링하는 것이 필요합니다. 이 연구에서는 VLM(visual-language model) 기반의 이미지 아웃페인팅(outpainting) 모델을 활용하여 부분 관찰에서 RGB 이미지를 확대 생성한 후, 단안 깊이 추정기(monocular depth estimator)를 통해 RGB-D 포인트 클라우드를 생성합니다. 이 포인트 클라우드는 모션 계획과 객체 감지에 사용될 수 있으며, 다양한 환경 불확실성을 표현하는 데 효과적입니다.

- **Performance Highlights**: 실험 결과, 제안된 파이프라인이 실제 환경에서의 목표 객체 탐색 문제를 해결하는 데 유용하다는 것을 보여줍니다. Matterport3D 데이터셋을 기반으로 총 1010개의 장면을 설정하고, 이 데이터셋에서 유도된 진실 값(ground-truth)을 평가한 결과 성능이 우수한 것으로 나타났습니다. 미래 연구 방향으로는 생성 모델이 플래닝 최적화를 어떻게 지원할 수 있는지를 더욱 탐구해야 하며, 실제 세계에서의 성능을 측정하는 방법도 개발해야 합니다.



### On the Optimal Representation Efficiency of Barlow Twins: An Information-Geometric Interpretation (https://arxiv.org/abs/2510.10980)
Comments:
          7 pages

- **What's New**: 본 논문에서는 Self-supervised learning (SSL)의 효율성을 정량화하기 위한 새로운 정보 기하학적 프레임워크를 제안합니다. 기존의 SSL 알고리즘은 레이블이 없는 데이터에서 의미 있는 표현을 학습하는 데 성공적이었으나, 서로 다른 SSL 패러다임의 효율성을 비교하고 이해하기 위한 통합된 이론적 틀이 부족했습니다. 이 연구는 Fisher Information Matrix (FIM)의 스펙트럼 성질을 기반으로 효과적인 내재 차원을 정의하고, 새로운 관점에서 SSL을 분석합니다.

- **Technical Details**: 우리는 표현 공간의 통계 매니폴드와 평균 Fisher Information Matrix (FIM)의 스펙트럼에 기반하여 표현 효율성을 정의합니다. Barlow Twins 방법에 대한 이론적 분석을 통해, 특정 가정 하에 Barlow Twins가 최적의 표현 효율성(η=1)을 달성함을 증명합니다. 이는 cross-correlation matrix를 단위 행렬에 가깝게 만드는 목표가 어떻게 효과적인 표현을 유도하는지를 설명합니다.

- **Performance Highlights**: Barlow Twins 방법은 두 왜곡된 데이터 뷰 간의 cross-correlation matrix를 최대한 단위 행렬에 가깝게 유지함으로써 좋은 표현을 학습합니다. 우리는 이 방법이 평균 Fisher Information Matrix의 스펙트럼 성질에 기반하여 효율성을 극대화한다는 것을 이론적으로 입증하였습니다. 본 연구는 Barlow Twins의 효과성을 이해하는 엄격한 이론적 기초를 제공하고 SSL 알고리즘을 분석하기 위한 새로운 기하학적 관점을 제시합니다.



### Comparative Evaluation of Neural Network Architectures for Generalizable Human Spatial Preference Prediction in Unseen Built Environments (https://arxiv.org/abs/2510.10954)
Comments:
          The 15th International Workshop on Structural Health Monitoring (IWSHM)

- **What's New**: 이 논문은 Cyber-Physical-Social Infrastructure Systems (CPSIS) 내에서 인간의 공간 선호도를 예측하는 능력에 대해 다룹니다. 특히, 훈련 중에 경험하지 못한 환경 구성에서의 선호도를 예측하는 모델의 일반화 가능성을 조사합니다. 다양한 신경망 구조의 효과를 비교하여, 어떤 모델이 새로운 레이아웃을 가장 잘 일반화하는지 알아봅니다.

- **Technical Details**: 우리는 Graph Neural Networks, Convolutional Neural Networks, 그리고 표준 feedforward Neural Networks의 비교 연구를 수행했습니다. 공원 환경을 간소화하여 생성된 합성 데이터를 사용하여 각 모델의 성능을 평가하였습니다. 각 모델은 이질적인 물리적, 환경적, 사회적 특징으로 영향을 받는 선호도를 예측하는 능력에 따라 평가됩니다.

- **Performance Highlights**: 다양한 신경망 아키텍처의 일반화 점수는 precision-recall curve의 아래 면적을 기반으로 계산됩니다. 이러한 접근 방법은 불균형 데이터에 적합하며, 보지 않은 새로운 환경에서의 인간 행동 모델링에 있어 각 신경망 구조의 타당성을 평가하는 데 도움을 줍니다.



### Optimally Deep Networks -- Adapting Model Depth to Datasets for Superior Efficiency (https://arxiv.org/abs/2510.10764)
Comments:
          6 pages, 3 figures, 1 table

- **What's New**: 새로운 논문에서는 Optimally Deep Networks (ODNs)를 소개합니다. 이 네트워크는 모델의 깊이를 주어진 작업의 복잡도에 맞추어 조절하여, 과도한 신경망 훈련을 피하고 효율성을 높이는 방법을 제안합니다. 특히, "progressive depth expansion"이라는 훈련 전략을 통해 더浅 깊이에서 시작하여 성능에 도달할 때까지 점진적으로 깊이를 조정합니다.

- **Technical Details**: ODNs는 특정 데이터셋에 최적화된 깊이만 사용하여 불필요한 레이어를 제거합니다. 이 전략은 메모리 사용량을 줄이고, 계산 효율성을 향상시키며, 자원이 제한된 장치에서의 배포를 용이하게 합니다. 또한, ODN은 기존의 대규모 네트워크 아키텍처에 쉽게 적용 가능하며, ResNet, MobileNet 등 여러 모델에 대해 적용할 수 있습니다.

- **Performance Highlights**: 실험 결과, ResNet-18과 ResNet-34 모델이 각각 MNIST와 SVHN 데이터셋에 대해 메모리 사용량을 각각 98.64%와 96.44% 줄이면서도 경쟁력 있는 정확도를 유지했습니다. 이로 인해 ODNs는 자원 소모를 최소화하면서도 높은 정확도를 성취할 수 있습니다. 이는 깊이 탐색 전략을 통해 모델의 효율성을 크게 향상시킨 결과입니다.



### VLM-Guided Adaptive Negative Prompting for Creative Generation (https://arxiv.org/abs/2510.10715)
Comments:
          Project page at: this https URL

- **What's New**: 이번 논문에서는 VLM-Guided Adaptive Negative-Prompting이라는 혁신적인 방법을 제안하여 기존의 이미지 생성 모델들에서의 창의력 문제를 해결합니다. 이 방법은 사전 훈련된 모델의 가중치를 수정하거나 특별히 선별된 데이터셋이 필요하지 않으며, 기존 확산(diffusion) 파이프라인에 원활하게 통합 가능합니다. 특히, 우리의 접근법은 단순히 새로운 시각적 개념을 생성하는 것에 그치지 않고, 복잡한 구성 문맥에서도 창의력을 유지하는 데 초점을 맞추고 있습니다.

- **Technical Details**: 우리는 비전-언어 모델(VLM)을 활용하여 생성 과정의 중간 출력을 분석하고, 기존의 시각적 개념에서 벗어나도록 동적으로 유도하는 방식을 사용합니다. 생성 과정에서 누적된 부정적인 프롬프트는 다음 디노이징 단계에 통합되어 성과를 향상시킵니다. 이 방법은 너무 많은 계산 자원을 소모하지 않으면서도 exploratory creativity의 일관된 향상을 보여주며, 다양한 VLM 모델 및 확산 파이프라인과 함께 실험을 수행했습니다.

- **Performance Highlights**: 우리는 창의성의 측면에서 기존의 방법들보다 더 나은 결과를 나타냈습니다. 확장된 실험을 통해 우리의 접근법이 단일 객체 생성에 국한되지 않고, 창의적인 객체의 일관된 집합을 생성할 수 있는 능력을 지니고 있음을 입증했습니다. 이로써 우리는 복잡한 구성 문맥에서도 혁신적인 출력을 생성하며, 기존의 텍스트 설명의 제약을 넘어서는 실용적인 경로를 제공합니다.



### JND-Guided Light-Weight Neural Pre-Filter for Perceptual Image Coding (https://arxiv.org/abs/2510.10648)
Comments:
          5 pages, 4 figures. Submitted to the IEEE International Symposium on Circuits and Systems (ISCAS) 2026

- **What's New**: 이번 논문은 Just Noticeable Distortion (JND) 가이드를 사용한 이미지 코딩의 인지 압축 효율성을 향상시키기 위한 새로운 기법을 제안합니다. FJNDF-Pytorch라는 통합 벤치마크를 개발하였으며, 여기에 기반한 경량화된 CNN 프레임워크를 구축하여 최첨단 압축 효율성을 달성하였습니다. 실험 결과, 이 방법은 기존의 기술을 초월하며 여러 데이터셋에서 뛰어난 성능을 발휘함을 보여줍니다.

- **Technical Details**: FJNDF-Pytorch 프레임워크는 JND 모델링 및 주입 단계를 중심으로 설계되었으며, JND 모델을 통해 인체 시각 시스템(HVS)의 지각 한계 이내의 최대 왜곡을 정량화합니다. 이 플랫폼은 여러 표준 공개 인코더와 데이터셋, 그리고 재현 가능한 연구를 위한 객관적인 품질 메트릭을 통합하여 JND 사전 필터의 개발 및 검증을 수월하게 합니다. JND 주입은 DCT 계수의 크기를 줄여 지각적 중복성을 제거하는 신호 수정을 수행합니다.

- **Performance Highlights**: 제안된 경량 네트워크는 1080p 이미지를 처리하는 데 단 7.15 GFLOPs의 계산비용만을 요구하며, 이는 최근 경량 네트워크의 14.1%에 불과합니다. 실험 결과, 기존의 주목할 만한 방법들과 비교하여 일관되게 더 뛰어난 압축 효율성을 보이는 것으로 나타났습니다. 이러한 성과는 효율성과 성능에서 모두 뛰어난 솔루션임을 입증하며, 연구자는 이 오픈 소스 구현을 통해 재현 가능한 연구를 지원합니다.



### ImpMIA: Leveraging Implicit Bias for Membership Inference Attack under Realistic Scenarios (https://arxiv.org/abs/2510.10625)
- **What's New**: 이 논문에서는 Membership Inference Attack (MIA)에서의 최신 기법인 ImpMIA를 소개합니다. ImpMIA는 기존의 참조 모델에 의존하지 않고, 신경망의 암묵적 편향(implicit bias)을 이용하여 훈련 샘플을 식별합니다. 기존의 가정들을 제거함으로써, 보다 현실적인 환경에서의 공격 성능이 크게 향상되었습니다.

- **Technical Details**: ImpMIA는 Karush-Kuhn-Tucker (KKT) 최적 조건을 통해 훈련 샘플을 식별합니다. 이 방법은 모델의 가중치와 훈련 데이터의 서브셋만을 가지고 작동하며, 훈련 절차나 데이터 분포에 대한 추가 정보 없이도 효과적으로 작동합니다. 이 공격은 훈련 데이터의 경량화된 그라디언트 계산을 통해 이루어집니다.

- **Performance Highlights**: ImpMIA는 세 가지 벤치마크 데이터셋에서 평가되었으며, 기존의 블랙박스 및 화이트박스 공격 방법들보다 우수한 성능을 보였습니다. 우리의 연구 결과에 따르면, 많은 참조 모델 기반 방법들이 의존하는 가정들이 제거될 때 성능이 크게 감소하지만, ImpMIA는 이러한 변화에 영향받지 않는다는 점이 강조되었습니다.



### UltraScatter: Ray-Based Simulation of Ultrasound Scattering (https://arxiv.org/abs/2510.10612)
Comments:
          Accepted at IEEE IUS 2025

- **What's New**: 이 논문에서는 UltraScatter라는 새로운 확률적 레이 트레이싱(Probabilistic Ray Tracing) 프레임워크를 소개하여 초음파 산란을 효율적이고 현실적으로 모델링합니다. 기존의 정통 초음파 시뮬레이션 방법들은 높은 정확도를 자랑하나 많은 계산 비용이 소요되는 반면, UltraScatter는 이 과정을 단 몇 초 만에 B-모드 이미지를 생성할 수 있는 대안으로 제시됩니다. 이를 위해, 조직은 산란 확률과 산란 진폭의 체적 필드로 표현되며, 레이의 상호작용은 자유 비행 델타 트래킹(Free-flight Delta Tracking)으로 시뮬레이션됩니다.

- **Technical Details**: UltraScatter는 Monte-Carlo 방식의 레이 트레이싱 기법을 활용하여 초음파 트랜스듀서에서 발생한 압력 파동을 모델링합니다. 압력 신호는 각 매체에서 전파되면서 발생하는 산란 이벤트를 기록하며, 이는 트랜스듀서에 수신된 음향파로 변환됩니다. 이 방식은 여러 매체의 감쇠, 흡수 및 다중 산란을 모델링 하는 모듈식 구조를 통해 구현됩니다.

- **Performance Highlights**: 검증 데이터와의 비교를 통해 UltraScatter는 현실적인 스펙클(speckle) 및 포함 패턴을 생성하는 것으로 나타났습니다. 이는 wave-based 방법들에 비해 확장성이 뛰어나며, 짧은 시간 내에 고품질의 B-모드 이미지를 생성할 수 있음을 보여줍니다. UltraScatter는 기존의 방법들보다 이미지 품질을 향상시키며, 더욱 정밀한 구조적 경계를 제공함을 입증하였습니다.



### SpikeGrasp: A Benchmark for 6-DoF Grasp Pose Detection from Stereo Spike Streams (https://arxiv.org/abs/2510.10602)
- **What's New**: 이번 논문에서는 생물학적 지능의 원리를 참고하여 센서 데이터를 직접적인 3D 포인트 클라우드로 변환하는 전통적인 방법 대신, 신경망 방식의 SpikeGrasp를 소개합니다. SpikeGrasp는 스파이크 카메라의 비동기 이벤트를 처리하여 그립 자세를 추론하며 점진적으로 가설을 개선합니다. 이러한 접근 방식은 고전적인 포인트 클라우드 기반 시스템보다 극적으로 데이터 효율성을 증가시키며, 특히 복잡하고 텍스처가 없는 장면에서 뛰어난 성능을 보입니다.

- **Technical Details**: SpikeGrasp는 생물학적 바이오 인스파이어드 아키텍처를 기반으로 하며, 비동기 스파이크 데이터를 반복적으로 정교화하는 메커니즘을 갖추고 있습니다. 삼각 스파이크 카메라로부터 비동기 이벤트 입력을 처리하여 시각적 경로, 통합 피질, motor 시스템 등을 통해 최종 6-DoF grasp pose를 결정하는 과정을 포함합니다. 이 시스템은 스파이크를 행동으로 매핑하는 end-to-end 학습 방식을 사용하며, 중간 기하학적 표현을 우회할 수 있습니다.

- **Performance Highlights**: SpikeGrasp는 대규모 합성 벤치마크 데이터셋을 활용하여 실험하였으며, 기존의 전통적인 포인트 클라우드 기반 방법보다 높은 성능을 발휘했습니다. 이 연구는 변형된 환경에서 물체를 안정적으로 조작하는 것을 가능하게 하여 향후 자연에서 관찰되는 유연하고 효율적인 조작 시스템으로 나아가는 기반을 마련합니다. 실험 결과, 스파이크 데이터의 효율성을 강조하며 제한된 훈련 샘플에도 불구하고 강력한 일반화 성능을 입증하였습니다.



### BitMar: Low-Bit Multimodal Fusion with Episodic Memory for Edge Devices (https://arxiv.org/abs/2510.10560)
Comments:
          6 pages, BabyLM Workshop, EMNLP 2025

- **What's New**: BitMar는 자원 제한이 있는 하드웨어에서 효과적인 이미지-텍스트 생성을 위해 외부 인간과 유사한 에피소딕 메모리를 제안하는 양자화된 다중 모달 변환기 모델입니다. 이 모델은 1.58비트 인코더를 사용하여 텍스트와 비전을 위한 경량화된 양자화된 임베딩을 생성합니다. BitMar는 데이터의 맥락 일관성을 높이기 위해 각 변환기 레이어에 대해 메모리를 조건부로 적용하는 인코더를 특징으로 합니다.

- **Technical Details**: BitMar의 네 가지 단계 파이프라인은 다음과 같습니다. 1) 1.58비트 텍스트 및 비전 인코더가 경량 양자화 임베딩을 생성합니다. 2) 교차 모달 융합 모듈이 공유 잠재 공간 내에서 모달리티를 정렬합니다. 3) 512개의 키-값 슬롯을 가진 에피소딕 메모리가 관련된 다중 모달 문맥을 검색합니다. 4) BitNet 기반 디코더가 검색된 메모리에 따라 각 변환기 레이어를 조건부로 제공합니다.

- **Performance Highlights**: BitMar는 극단적인 압축 상태에서도 competitive performance를 달성하며, 이미지 캡셔닝과 다중 모달 이해에서 낮은 대기 시간과 최소한의 메모리 사용으로 우수한 결과를 보여줍니다. 이 아키텍처는 통합된 768 차원의 공간 내에서 모든 모듈을 유지하여 통합을 간소화하고 프로젝션 오버헤드를 최소화합니다. 이러한 특성 덕분에 BitMar는 엣지 배포에 적합합니다.



### SuperEx: Enhancing Indoor Mapping and Exploration using Non-Line-of-Sight Perception (https://arxiv.org/abs/2510.10506)
Comments:
          8 pages, 9 Figures , Project webpage: this https URL

- **What's New**: 이번 연구는  로봇 탐사에 비선형 시야(nonline-of-sight, NLOS) 감지를 적용하여 실내 환경 탐사 및 매핑의 정확도를 높이고자 합니다. 기존 시스템은 로봇의 인식이 시야에 국한되어 있어, 탐사가 비효율적이고 시간에 민감한 상황에서 문제를 발생시킵니다. 이에 우리는 단일 광자 LiDAR를 사용해 숨겨진 물체를 감지할 수 있는 기술을 통합하여, 복잡한 실내 환경에서 좋은 탐사 효율성을 달성했습니다.

- **Technical Details**: 우리는 SuperEx라는 새로운 프레임워크를 소개하며, NLOS 감지를 매핑 탐사 루프에 직접 통합하여 기능을 강화합니다. 이 프레임워크는 (1) 타이밍 히스토그램을 통해 비어 있는 NLOS 영역을 수정하고, (2) 물리 기반의 재구성과 데이터 기반 접근 방식을 결합하여 점유된 구조를 재구성하는 두 가지 주요 기능을 갖추고 있습니다. 우리가 실험에 사용한 KTH Floorplan 데이터셋과 복잡한 시뮬레이션 맵에서, 30% 미만의 커버리지 시 mapping accuracy가 12% 향상되었습니다.

- **Performance Highlights**: NLOS 감지를 포함한 우리의 접근 방식은 기존의 시야 중심(narrow view) 방식에 비해 탐사 효율성을 개선했습니다. 전체적인 매핑 정확도와 탐사 성능이 향상된 결과는 복잡한 실내 환경에서 신뢰할 수 있는 매핑을 가능하게 합니다. 이 연구는 또한 일반 로봇에서 사용할 수 있는 실용적인 기술을 제시하여, 시간에 민감한 다양한 환경의 탐사 및 구조 활동에 기여할 것으로 기대됩니다.



### Towards Efficient 3D Gaussian Human Avatar Compression: A Prior-Guided Framework (https://arxiv.org/abs/2510.10492)
Comments:
          10 pages, 4 figures

- **What's New**: 이 논문에서 제안하는 새로운 3D 아바타 코딩 프레임워크는 휴먼 프라이어(human prior)와 기준-타겟 변환(canonical-to-target transformation)을 활용하여 고품질의 3D 아바타 비디오 압축을 초저 비트 전송률에서 가능하게 합니다. 특히, 네트워크가 필요 없는 방식으로 훈련된 기준 가우시안 아바타는 아바타의 외형 모델링을 위한 기초로서 작용하며, 이를 통해 데이터의 중복을 최소화할 수 있습니다. 또한, 이 프레임워크는 단 한 번의 압축만으로 모든 프레임에 대한 기준 아바타를 공유할 수 있게 합니다.

- **Technical Details**: 제안된 효율적인 3D 아바타 코딩 프레임워크는 차량에 대한 세부 구조를 설정하였고, 여러 뷰 비디오를 통해 기준 3D 가우시안 아바타를 훈련합니다. 아바타의 캐노니컬 표현은 3D 가우시안 속성으로 정의되며, 인물의 움직임을 흐름 파라미터로 모델링하여 효율적으로 전송할 수 있도록 도와줍니다. 각 프레임에서는 리니어 블렌드 스키닝(Linear Blend Skinning) 알고리즘을 통해 기준 아바타를 변형시키는 방식으로 타겟 아바타를 생성합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 기존의 2D 및 3D 코덱은 물론 학습 기반의 동적 3D 가우시안 스플래팅 압축 방법보다 월등한 비율-왜곡 성능을 보였습니다. ZJU-MoCap 및 MonoCap 데이터셋을 통해 일반화된 성능을 입증하였으며, 이는 메타버스 적용에서 매끄럽고 몰입감 있는 멀티미디어 경험을 지원하는 방향으로 나아가게 합니다.



### ArtPerception: ASCII Art-based Jailbreak on LLMs with Recognition Pre-tes (https://arxiv.org/abs/2510.10281)
Comments:
          30 pages, 22 figures. This preprint has been accepted for publication in Elsevier JOURNAL OF NETWORK AND COMPUTER APPLICATIONS (JNCA)

- **What's New**: 이 논문은 ArtPerception이라는 새로운 블랙박스 탈옥 프레임워크를 소개합니다. 이 시스템은 ASCII 아트를 활용하여 최첨단 대형 언어 모델(LLMs)의 보안 조치를 우회하는 방식으로 설계되었습니다. ArtPerception은 비효율적인 반복 공격 방법 대신, 체계적인 두 단계의 방법론을 도입하여 효과성을 극대화합니다.

- **Technical Details**: ArtPerception의 첫 번째 단계는 특정 모델에 대한 최적의 ASCII 아트 인식 파라미터를 결정하기 위한 1회 사전 테스트(pre-test)를 수행합니다. 두 번째 단계에서는 이 정보를 활용하여 매우 효율적인 원샷 공격을 실행합니다. 또한, 연구에서는 LLM의 인식 능력을 평가하기 위한 수정된 레벤슈타인 거리(Modified Levenshtein Distance, MLD) 메트릭을 제안했습니다.

- **Performance Highlights**: 실험을 통해 네 가지 최첨단 오픈소스 LLM에 대한 뛰어난 탈옥 성능을 입증했습니다. ArtPerception은 실제 상업 모델(GPT-4o, Claude Sonnet 3.7 및 DeepSeek-V3)에서도 성공적으로 적용될 수 있음을 보여주었으며, 일반적인 방어 기법(LLaMA Guard 및 Azure의 콘텐츠 필터)에 대한 강인성을 입증했습니다.



### X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Mod (https://arxiv.org/abs/2510.10274)
Comments:
          preprint, technical report, 33 pages

- **What's New**: 이번 논문은 Vision-Language-Action (VLA) 모델의 훈련을 위한 새로운 Soft Prompt 방식을 제안합니다. 이 방식은 다양한 로봇 플랫폼에서 수집된 이질적 데이터셋을 효과적으로 활용하여, 각기 다른 데이터 소스에 대한 학습 가능한 임베딩을 도입합니다. 이를 통해 VLA 모델은 다양한 하드웨어 구성과 데이터 타입을 다루는 능력을 향상시킵니다.

- **Technical Details**: 제안된 X-VLA는 소프트 프롬프트를 이용하여 Transformer 엔코더를 기반으로 하는 새로운 VLA 아키텍처입니다. X-VLA는 하드웨어 구성 설정을 잘 반영한 모형을 제공하며, 여러 시스템 구조에 적응할 수 있도록 설계되었습니다. 또한, 다양한 환경과 작업에 따른 통합된 특징 추출을 가능하게 하여, VLA 학습의 범위와 유연성을 극대화합니다.

- **Performance Highlights**: X-VLA-0.9B는 6개의 시뮬레이션 및 3개의 실제 로봇 환경에서 평가되었으며, 여러 벤치마크에서 SOTA 성능을 달성하였습니다. 특히, 1,200번의 데모를 통해 실제에서 섬세한 옷 접기 작업을 수행하며 2분 내에 평균 1개의 옷을 접을 수 있는 속도를 기록했습니다. 또한, 파라미터 조정량이 1%에 불과함에도 높은 성공률을 자랑하는 등, 효율적인 학습과 강력한 적응 성능을 보여주었습니다.



### INR-Bench: A Unified Benchmark for Implicit Neural Representations in Multi-Domain Regression and Reconstruction (https://arxiv.org/abs/2510.10188)
- **What's New**: 본 논문에서는 Implicit Neural Representations (INRs)의 효과성 및 한계에 대한 깊은 통찰을 제공하고, Neural Tangent Kernel (NTK) 이론을 활용하여 다양한 모델 아키텍처, 포지셔널 인코딩 및 비선형 프리미티브가 신호의 주파수 응답에 미치는 영향을 분석합니다. 연구의 일환으로 다중 모달 INR 작업을 위한 최초의 포괄적인 벤치마크인 INR-Bench를 소개하였으며, 이는 56개의 좌표 MLP 모델과 22개의 좌표 KAN 모델로 구성되어 있습니다. 이 벤치마크를 통해 다양한 신경 모델의 강점과 한계를 강조하고, 향후 연구를 위한 강력한 기반을 마련하였습니다.

- **Technical Details**: 이 연구는 KAN과 MLP 모델 아키텍처의 NTK 스펙트럼 분포에 미치는 영향을 분석하였습니다. KAN은 낮은 주파수 학습에서 MLP보다 작은 스펙트럼 편향을 보이나, 높은 계산 복잡성 문제를 겪습니다. FKAN은 KAN을 위한 풀 수학적으로 학습 가능한 포지셔널 인코딩으로, 새로운 주파수 구성 요소를 신호에서 학습할 수 있도록 돕습니다. 또한, ReLU, Sine, Gaussian 등의 비선형 프리미티브가 NTK 스펙트럼 분포에 미치는 영향을 분석하여, 각 활성화 함수의 장단점을 도출하였습니다.

- **Performance Highlights**: INR-Bench는 56개의 좌표 MLP와 22개의 KAN 모델을 포함하여 9개의 다중 모달 작업에서 성능을 평가합니다. 이 데이터셋은 다양한 주파수 학습, 역 추론 및 일반화 능력을 효과적으로 평가할 수 있는 구조를 갖추고 있습니다. 특히, FKAN을 포함한 포지셔널 인코딩과 여러 활성화 함수가 신경망의 표현 능력에 미치는 영향을 심층적으로 탐구함으로써, 신경 모델의 향후 발전을 위한 방향성을 제시하고 있습니다.



### Dejavu: Post-Deployment Learning for Embodied Agents via Experience Feedback (https://arxiv.org/abs/2510.10181)
- **What's New**: 이번 연구에서는 Embodied agent가 고정된 데이터 세트 분포에 의존하여 학습하는 한계를 극복하기 위해 "Dejavu"라는 새로운 post-deployment learning framework를 제안합니다. 이를 통해 Experience Feedback Network (EFN)을 사용하여 Vision-Language-Action (VLA) 정책을 새로운 실행 메모리로 보강할 수 있도록 합니다. EFN은 과거 성공적인 행동 경험을 자동으로 식별하고 이를 기반으로 행동 예측을 조정하는 방식을 채택합니다.

- **Technical Details**: 연구에서 정의한 경험은 비전, 언어, 행동이 동기화된 궤적으로 구성되며, 이러한 궤적은 VLA 인터페이스에 맞춰 experience bank에 저장됩니다. EFN은 현재 관찰값과 이전의 경험을 결합하여 추가적인 행동을 예측하는 네트워크로 훈련되며, 이 과정에서 비슷한 관찰 결과를 복원하기 위한 강화 학습(reinforcement learning) 신호를 사용합니다. EFN은 soft actor-critic 알고리즘을 활용하여 최적화되며, 이는 탐색의 효율성을 높이고 안정적인 가치 학습을 가능하게 합니다.

- **Performance Highlights**: 다양한 Embodied task에서 실시한 실험 결과에 따르면, EFN은 고정된 baseline보다 적응력, 강건성 및 성공율을 유의미하게 향상시키는 것으로 나타났습니다. 연구는 LIBERO와 AgiBot G1 로봇을 사용하여 처치와 현장 실험을 수행했고, 모든 환경에서 EFN이 VLA 정책의 배치 성능을 개선함을 확인했습니다. 이는 EFN이 포스트 배치 후 지속적으로 행동을 개선하는 데 기여할 가능성을 보여줍니다.



### Enabling High-Quality In-the-Wild Imaging from Severely Aberrated Metalens Bursts (https://arxiv.org/abs/2510.10083)
- **What's New**: 본 연구에서는 초박형 나노포토닉 메탈렌스 카메라를 사용하여 실시간 이미징의 도전을 해결하고 있습니다. 메탈렌스는 기존의 굴절 광학에 비해 크기와 무게의 극적인 감소를 약속하지만, 심각한 색수차(chromatic aberration)와 낮은 광 효율 등으로 인해 실제 채택이 제한되고 있습니다. 우리의 방법은 메탈렌스 카메라용으로 맞춤 설계된 멀티이미지 복원 프레임워크와 결합되어, 노이즈 및 왜곡을 능동적으로 교정하며 성능을 향상시킵니다.

- **Technical Details**: 이 작업은 가벼운 합성곱 신경망(convolutional network)과 메모리 효율적인 버스트 융합(burst fusion) 알고리즘을 사용하여 급격히 저하된 메탈렌스 이미지 시퀀스에서 노이즈와 왜곡을 교정합니다. 또한, 이 방법은 메탈렌스 카메라의 명확한 초점을 유지하는 데 최적화되어 있으며, 복원 복잡도를 최소화하면서도 고품질 이미지를 재구성합니다. 더불어, 메탈렌스와 복합광학 간의 짝을 이룬 학습 데이터가 필요하지 않습니다.

- **Performance Highlights**: 다양한 실제 환경에서의 실험 결과, 본 연구의 접근법은 기존의 버스트 모드 및 단일 이미지 복원 방식보다 일관되게 우수한 성능을 발휘했습니다. 우리의 프로토타입은 초박형 메탈렌스를 중심으로 한 카메라로, 다양한 실세계 조건에서 고품질 이미징을 입증했습니다. 이러한 연구 결과는 메탈렌스 기반 카메라의 실용적인 도입이 가능하다는 것을 보여줍니다.



### SecureWebArena: A Holistic Security Evaluation Benchmark for LVLM-based Web Agents (https://arxiv.org/abs/2510.10073)
- **What's New**: 본 논문에서는 LVLM 기반 웹 에이전트의 보안을 평가하기 위한 최초의 포괄적 벤치마크인 SecureWebArena를 소개합니다. 기존의 벤치마크들이 제한적인 범위의 해석으로 제한되었던 것과 달리, 본 벤치마크는 E-commerce 플랫폼, 커뮤니티 포럼 등 6개의 시뮬레이트된 웹 환경을 포함하여 다양한 작업 및 공격 설정을 아우르는 2,970개의 고품질 궤적을 제공합니다. 이로 인해 다양한 사용자 수준 및 환경 수준의 취약성을 포괄적으로 분석할 수 있게 됩니다.

- **Technical Details**: SecureWebArena는 6개의 공격 벡터를 정의하여 사용자 조작 및 환경 위협을 모두 포괄하는 체계적인 분류를 제공합니다. 각 에이전트의 실패를 세 가지 주요 차원, 즉 내부 추론, 행동 궤적, 작업 결과를 통해 분석하는 다층 평가 프로토콜을 도입하여 보다 정밀한 리스크 분석이 가능합니다. 이러한 방식으로 에이전트의 보안 취약성을 심도 있게 분석할 수 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 9개의 주요 LVLM이 일관되게 미세한 적대적 조작에 취약하다는 것을 발견하였고, 모델의 전문화와 보안 사이의 중요한 trade-offs를 드러냈습니다. 모든 에이전트가 다양한 공격 벡터에 대해 저항력을 가지지 못함을 보여주는 결과는 헌신적이고 신뢰할 수 있는 웹 에이전트 배포를 위해 SecureWebArena가 필수적인 진단 도구가 될 것임을 암시합니다.



### Translution: Unifying Self-attention and Convolution for Adaptive and Relative Modeling (https://arxiv.org/abs/2510.10060)
Comments:
          technical report

- **What's New**: 이 논문에서는 Translution이라는 새로운 연산을 소개합니다. 이 연산은 self-attention의 적응적 식별 기능과 convolution의 상대적 인코딩 장점을 결합합니다. 그러나 이러한 통합은 파라미터 수의 상당한 증가를 초래하여 현재 사용 가능한 대부분의 계산 자원을 초과합니다. 그래서 우리는 파라미터 수를 줄이기 위해 α-Translution이라는 경량 변형을 제안합니다.

- **Technical Details**: Translution은 쿼리(queries), 키(keys), 값(values) 계산 시 각 거리와 방향에 대해 별도의 매개변수(매트릭스)를 할당하는 convolution 스타일 접근 방식을 사용합니다. 이는 Translution이 상대적 구조를 효과적으로 인코딩할 수 있도록 합니다. 이 논문에서는 컴퓨터 비전과 자연어 처리 작업에서의 실험을 통해 Translution과 α-Translution이 self-attention보다 더 나은 정확도를 달성함을 보입니다. 두 아키텍처는 Vision Transformer(ViT)와 Generative Pre-trained Transformer(GPT)입니다.

- **Performance Highlights**: 실험 결과, Translution과 α-Translution은 self-attention을 넘는 정확도를 보여줍니다. α-Translution은 '이상적'인 원본 Translution보다는 낮은 정확도를 보이지만, self-attention보다 높은 결과를 기록했습니다. 이 연구는 기존의 convolutional neural networks와 self-attention의 한계를 극복하고 새로운 신경망 설계를 위한 길을 여는 데 기여하고 있습니다.



### CLoD-GS: Continuous Level-of-Detail via 3D Gaussian Splatting (https://arxiv.org/abs/2510.09997)
- **What's New**: 이 논문은 3D Gaussian Splatting (3DGS) 기술을 기반으로 한 새로운 Continuous Level of Detail (CLoD) 접근 방식을 소개합니다. 이는 기존의 Discrete Level of Detail (DLoD) 방식의 단점을 해결하고, 단일 모델을 통해 부드럽고 연속적인 품질 스케일링을 가능하게 합니다. 새로운 CLoD-GS 프레임워크를 통해 각 Gaussian primitive에 대해 거리 의존적인 감소 매개변수를 도입하여 시각적 품질을 효과적으로 개선합니다.

- **Technical Details**: CLoD-GS는 각 Gaussian primitive의 불투명도를 동적으로 조절하는 학습 가능한 매개변수를 추가하여, 모델이 원거리에서도 강력하게 작동하도록 학습합니다. 또한, 가상 거리 스케일링 메커니즘과 점 수 정규화 손실을 도입하여 학습 과정에서 모델이 더욱 컴팩트한 표현을 학습하도록 유도합니다. 이 접근 방식은 기존의 방법에서 오는 저장 오버헤드와 시각적 아티팩트를 제거하는 효과를 가져옵니다.

- **Performance Highlights**: CLoD-GS는 하나의 컴팩트 모델을 통해 부드러운 품질 스케일 가능한 렌더링을 제공하며, 다양한 성능 목표에서 높은 충실도를 달성합니다. 실험 결과, CLoD-GS는 복잡한 장면에서도 효과적으로 작동할 수 있는 가능성을 제시합니다. 이러한 결과는 실시간 신경 렌더링 애플리케이션이 더 확장 가능하고 시각적으로 일관된 방식으로 발전하는 데 기여할 수 있습니다.



### Generative Latent Video Compression (https://arxiv.org/abs/2510.09987)
Comments:
          Preprint. Supplementary material in Openreview

- **What's New**: 이 논문은 비디오 압축에 대한 새로운 접근법인 Generative Latent Video Compression (GLVC)을 소개합니다. GLVC는 기존의 신경 비디오 코덱에서 나타나는 품질 변동 문제를 해결하기 위해 고안되었습니다. 이 방식은 비디오 프레임을 감각적으로 정렬된 잠재 공간(perceptually aligned latent space)으로 변환하고, 이는 비율-왜곡 최적화(rate-distortion optimization)와 감각적 제약을 분리합니다.

- **Technical Details**: GLVC는 사전 훈련된 연속 토크나이저(pretrained continuous tokenizer)를 사용하여 비디오 프레임을 잠재 공간으로 매핑합니다. 새로운 코덱 아키텍처는 잠재 도메인(latent domain)에서의 최적화를 위해 재설계되었으며, 통합된 인트라/인터 코딩(unified intra/inter coding) 및 순환 메모리(recurrent memory) 메커니즘을 도입하여 성능을 향상시킵니다. 이러한 설계는 이전 신경 비디오 코덱의 인사이트를 활용하여 이루어졌습니다.

- **Performance Highlights**: GLVC는 여러 벤치마크를 통해 DISTS 및 LPIPS 메트릭에서 최첨단 성능을 달성했습니다. 사용자 연구 결과에 따르면 GLVC는 최신 신경 비디오 코덱과 대등한 품질을 유지하면서도 약 절반의 비율로 압축할 수 있음을 확인했습니다. 이는 실용적인 감각적 비디오 압축(perceptual video compression)을 위한 중요한 진전을 나타냅니다.



### MTMD: A Multi-Task Multi-Domain Framework for Unified Ad Lightweight Ranking at Pinteres (https://arxiv.org/abs/2510.09857)
Comments:
          AdKDD 2025

- **What's New**: 이 논문에서는 가벼운 광고 추천 시스템의 중요한 요소인 경량 광고 순위 매기기(layer)에서 멀티 태스크 멀티 도메인(MTMD) 아키텍처를 제안합니다. 기존의 모델들이 여러 광고 도메인과 광고 제품을 통합하여 처리하기 어려운 문제를 해결하기 위해, 도메인 전문가를 활용한 혼합 전문가 아키텍처를 도입하였습니다. 이 접근법은 광고 클릭률(CTR) 및 전환률(CVR)을 포함하는 여러 최적화 목표를 동시에 다룰 수 있는 장점을 제공합니다.

- **Technical Details**: MTMD는 쿼리 타워와 아이템 타워로 구성된 두 개의 타워 아키텍처를 바탕으로 설계되었습니다. 각 타워는 다양한 광고 서피스나 제품 유형에 맞춘 도메인 전문가들로 구성되며, 이들을 통해 입력 데이터와 출력 작업을 통합적으로 관리할 수 있습니다. 또한, 도메인 간 지식 전이를 촉진하기 위해 도메인 적응 모듈(domain adaptation module)을 도입하여 전문가 간의 상호작용을 강조합니다.

- **Performance Highlights**: 이 MTMD 아키텍처는 오프라인 손실 값(off-line loss value)을 12%에서 36%까지 개선시켰고, 이는 클릭당 비용(cost per click)을 2% 줄이는데 기여하였습니다. 실제로, 이 단일 모델은 Pinterest 광고 추천 시스템에 배포되었으며, 이전의 9개 모델을 대체하여 광고 클릭률(CTR), 좋은 클릭 이벤트(GCTR), 클릭 비용(CPC) 등의 온라인 메트릭에서도 상당한 성과를 보였습니다.



### Text Prompt Injection of Vision Language Models (https://arxiv.org/abs/2510.09849)
- **What's New**: 이 논문에서는 대규모 비전 언어 모델(VLMs)에 대한 텍스트 프롬프트 주입(text prompt injection) 공격 방법을 연구했습니다. 텍스트 프롬프트를 이미지에 주입하여 VLM이 원본 이미지 내용과 일치하지 않는 반응을 생성하도록 유도하는 방식입니다. 이 접근법은 다른 공격 방법들에 비해 주로 대규모 모델에 효과적이며, 계산 자원 소모가 적다는 점이 특징입니다.

- **Technical Details**: 이 연구에서는 랜덤한 이미지 내에 텍스트 프롬프트를 삽입하여 VLM의 인식 과정을 방해하는 알고리즘을 설계했습니다. 공격의 효율성을 높이기 위해, 이미지의 색상 일관성이 높은 영역을 식별하고 그곳의 픽셀을 변형하여 텍스트의 윤곽을 생성하는 방식을 사용했습니다. 특히 l∞(l-infinity) 제약 조건을 설정하여 원본 이미지의 외관이 심하게 변경되지 않도록 했습니다.

- **Performance Highlights**: 실험 결과, 고급 VLM 모델(예: Llava-Next-72B, Qwen-VL-Max 및 GPT-4)은 텍스트 프롬프트 주입 공격에 대해 높은 성공률을 나타냈습니다. 특히, Llava-Next-72B 모델은 가장 많은 기준 사례에서 성공적이었고, 이 모델의 파라미터 수와 명령 수용 능력 간의 긍정적인 상관관계를 발견했습니다. 이러한 결과는 VLM 역시 적절한 보호 장치가 없을 경우 공격에 취약하다는 점을 강조하고 있습니다.



### Harnessing Self-Supervised Deep Learning and Geostationary Remote Sensing for Advancing Wildfire and Associated Air Quality Monitoring: Improved Smoke and Fire Front Masking using GOES and TEMPO Radiance Data (https://arxiv.org/abs/2510.09845)
Comments:
this https URL

- **What's New**: 이 연구는 NASA의 TEMPO 위성 미션에서 얻은 전례 없는 시간별 데이터와 자기 지도(Self-Supervised) 심층 학습의 발전을 활용하여 미국 서부의 산불 및 공기 질 관리 향상을 제시합니다. 이 연구는 산불 전선과 연기 기둥의 거의 실시간 시간별 확산을 매핑하는 데 있어 심층 학습의 효율성을 보여줍니다.

- **Technical Details**: 이 연구에서는 GOES-18과 TEMPO 데이터를 사용하여 연기 기둥을 구름과 성공적으로 구별하는 혁신적인 자기 지도 심층 학습 시스템을 사용합니다. 서로 다른 감지 모달리티(modalities)에서 생성된 연기 및 화재 마스크 간에 강력한 일치를 보여줍니다.

- **Performance Highlights**: 또한, 동일한 사례에 대한 운영 제품보다 상당한 개선이 이루어진 것을 강조합니다. 이를 통해 산불과 대기질 관리의 새로운 가능성이 열렸습니다.



### Decomposer Networks: Deep Component Analysis and Synthesis (https://arxiv.org/abs/2510.09825)
Comments:
          13 Pages, 4 figures

- **What's New**: 본 논문에서는 Decomposer Networks (DecompNet)을 제안한다. 이는 입력을 해석 가능한 여러 구성 요소로 분해하는 의미론적 오토인코더(semantic autoencoder)이다. 기존의 오토인코더가 단일 잠재 표현(latent representation)으로 입력을 압축하는 반면, DecompNet은 N개의 병렬 브랜치를 유지하여 각 브랜치에 다른 브랜치들의 재구성을 차감한 잔여 입력(residual input)을 할당한다.

- **Technical Details**: DecompNet은 가우스-자이델 스타일의 블록 좌표 경량화를 비분화(differentiable) 네트워크로 풀어내어 구성 요소 간의 명시적인 경쟁을 촉진하여 비율적으로 의미 있는 표현을 생성한다. 이 모델은 PCA(주성분 분석), NMF(비음수 행렬 분해) 및 객체 중심 모델(MONet, IODINE, Slot Attention)과의 관계를 설정하고, 첫 번째 의미론적 오토인코더로서 모든 브랜치에서 하나를 제외한 업데이트 규칙을 구현한다. 각 브랜치는 다른 브랜치가 모델링할 수 없는 요소를 모델링하도록 강제함으로써 설계적으로 의미론적 해리(semantic disentanglement)를 생성한다.

- **Performance Highlights**: DecompNet을 통해 생성된 구성 요소들은 나름대로 독립적이며, 이는 깊은 비분화(Unrolled) 방법들보다도 훨씬 더 파라미터를 적게 사용할 수 있게 한다. 제안된 모델은 입력을 선형이 아닌 의미론적 구성 요소로 분해할 수 있는 가능성을 열어주며, 각 브랜치가 재구성된 입력 데이터와 어떻게 관계가 있는지를 명확히 보여주고 있다. 결국, 이 구조는 데이터 해석 및 시그널 프로세싱(sl signal processing) 분야에서 더 나은 성능을 발휘하는 길을 제시한다.



### Cross-Sensor Touch Generation (https://arxiv.org/abs/2510.09817)
Comments:
          CoRL 2025

- **What's New**: 이번 논문은 비주얼 촉각 센서의 다양성으로 인해 발생하는 일반적 촉각 표현 개발의 어려움을 해결하기 위한 두 가지 새로운 접근 방식을 제안합니다. 첫 번째 방법은 쌍(pair) 데이터를 활용한 Touch2Touch 방식이며, 두 번째는 중간 깊이 표현(depth representation)을 사용하는 Touch-to-Depth-to-Touch(T2D2) 방식입니다. 두 방법은 다양한 센서들을 통해 촉각 신호를 생성하고, 데이터의 가용성과 응용 필요에 따라 유연한 솔루션을 제공합니다.

- **Technical Details**: 입력으로 사용되는 센서에 따라 촉각 신호를 변환하는 크로스 센서 변환 문제를 다루며, 조건부 확산 모델을 활용하여 서로 다른 센서에서 수집된 쌍 데이터를 기반으로 촉각 신호를 생성합니다. 깊이 맵(depth map)을 중간 표현으로 사용하여 새로운 센서의 통합을 최소한의 데이터 수집으로 가능하게 합니다. 실험은 로봇의 손 안 물체 포즈 추정(in-hand object pose estimation) 및 행동 복제(behavior cloning)에서 두 가지 조작 작업으로 평가되었습니다.

- **Performance Highlights**: 제안된 접근 방식을 통해 서로 다른 센서에 대해 정확한 촉각 신호 생성을 가능하게 하여 로봇이 다른 센서용으로 설계된 알고리즘을 성공적으로 사용할 수 있음을 입증합니다. 두 가지 작업에서, 모델은 소스(sensor) 센서의 측정값을 기반으로 타겟(target) 센서에 맞는 촉각 이미지를 생성해야 하며, 이러한 방식을 통해 로봇 조작 시나리오에서의 실용성을 보여줍니다.



### Causality $\neq$ Decodability, and Vice Versa: Lessons from Interpreting Counting ViTs (https://arxiv.org/abs/2510.09794)
- **What's New**: 이번 연구는 비전 트랜스포머(vision transformers, ViTs)에서 정보의 디코드 가능성(decodability)과 인과성(causality) 간의 관계를 탐구합니다. 특히 물체 세기를 위해 파인튜닝된 ViTs를 대상으로 하여, 각 레이어에서 토큰의 역할을 확인합니다. 연구 결과, 중간 레이어의 객체 토큰은 강한 인과적 영향을 미치지만 약한 디코드 가능성을 보이는 특이한 패턴이 발견되었습니다.

- **Technical Details**: 연구진은 activation patching을 사용해 깨끗한 이미지와 손상된 이미지 쌍의 숨겨진 활성화를 이식하여 예측에 미치는 영향을 실험합니다. 이로 인해 중간 레이어에서 객체 토큰의 활성화가 예측에 큰 영향을 미치는 방식이 확인되었습니다. 반면, 최종 레이어의 객체 토큰은 높은 디코드 가능성을 보이나 실질적인 예측 영향력이 없음을 보여주었습니다.

- **Performance Highlights**: 연구 결과는 디코드 가능성과 인과성이 상호 교환 가능한 개념이 아님을 강조합니다. 디코드 가능성과 인과성 사이의 불일치는 ViTs의 숨겨진 계산 회로를 드러내는 중요한 정보로 작용할 수 있습니다. 이러한 관점은 기계적 해석 가능성(mechanistic interpretability)을 더 깊이 이해하는 데 필요하다고 주장합니다.



### Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry (https://arxiv.org/abs/2510.09740)
Comments:
          Accepted to NeurIPS 2025 Workshop on Reliable ML from Unreliable Data

- **What's New**: 이번 논문에서는 Neural Collapse Geometry(NCAL-R) 기반의 신뢰할 수 있는 능동 학습(Active Learning) 방법론을 제안합니다. 이 방법은 불확실한 주석(annotations) 하에서도 효과적으로 샘플을 선택할 수 있도록 지원합니다. NCAL-R은 구조적 안정성과 왜곡을 평가하는 Class-Mean Alignment Perturbation 점수와, 훈련 checkpoints 간의 표현의 변동성을 포착하는 Feature Fluctuation 점수를 도입하여, 이러한 두 신호를 결합하여 샘플을 선택합니다.

- **Technical Details**: 연구에서는 pool 기반 능동 학습 설정을 고려하고, 수량화된 Class-Mean Alignment Perturbation(CMAP)와 Feature Fluctuation(FF) 점수를 사용하여 샘플의 구조적 영향을 분석합니다. NCAL-R은 높은 CMAP와 FF 점수를 가진 샘플들을 우선적으로 선택하여 일반화 오류를 감소시키고, 샘플 선택에 있어 불확실성을 줄입니다. 이 방법은 별도의 보조 네트워크나 특정 과제 조정 없이도 일반적인 백본(backbone) 및 모달리티에 적용 가능하다는 장점이 있습니다.

- **Performance Highlights**: NCAL-R은 ImageNet-100 및 CIFAR-100 데이터셋에서 이전의 능동 학습 기준선에 비해 일관되게 더 높은 정확도를 기록하며, 레이블이 부정확한 상황에서도 개선된 견고성을 보여주었습니다. 또한, NCAL-R은 배포된 실제 라벨링 파이프라인에서 신뢰할 수 있는 운영을 위한 중요한 진전을 나타냅니다. 실험 결과, 새로운 분포(out-of-distribution) 데이터에 대한 일반화 능력 또한 향상되었음을 보여주었습니다.



### VisRAG 2.0: Evidence-Guided Multi-Image Reasoning in Visual Retrieval-Augmented Generation (https://arxiv.org/abs/2510.09733)
- **What's New**: 본 논문에서는 Vision-Language 모델(즉 VLM)을 위해 외부 시각적 지식을 활용한 Visual retrieval-augmented generation (VRAG) 방식의 한계를 극복하는 새로운 프레임워크인 EVisRAG를 제안합니다. 이를 통해 여러 이미지 간의 시각적 증거를 보다 효과적으로 통합하고, 잘못된 결론 도출을 줄일 수 있습니다. EVisRAG는 퍼지한 질문에 대해 여러 이미지를 통해 수집한 증거를 바탕으로 최종 답안을 도출해내는 방식으로 작동합니다.

- **Technical Details**: EVisRAG는 우선적으로 검색된 이미지를 관찰하고, 각 이미지에 대한 증거를 기록한 이후, 집계된 증거를 바탕으로 최종 답변을 도출하는 구조를 가지고 있습니다. 이를 효과적으로 훈련하기 위해, Reward-Scoped Group Relative Policy Optimization (RS-GRPO) 알고리즘을 도입하여, 세분화된 보상을 스코프 특정 토큰에 바인딩하여 시각적 인지 및 추리 능력을 최적화합니다.

- **Performance Highlights**: 여러 시각적 질문 응답 기준에서의 실험 결과에 따르면, EVisRAG는 기존의 VRAG보다 평균 27% 향상된 성과를 보여줍니다. 이는 EVisRAG가 질문 관련 증거를 정확하게 탐지하고 시각적으로 유추하여 정확한 답변을 도출하는 능력 덕분입니다. 따라서, EVisRAG는 향상된 시각적 인식을 통해 질문 이해도 및 응답 품질을 높이는데 기여하고 있습니다.



### Layout-Aware Parsing Meets Efficient LLMs: A Unified, Scalable Framework for Resume Information Extraction and Evaluation (https://arxiv.org/abs/2510.09722)
- **What's New**: 이번 연구에서는 이력서 정보 추출을 자동화하기 위한 레이아웃 인식(layout-aware) 및 효율성을 최적화한 프레임워크를 제안합니다. 이 시스템은 다양한 문서 형식을 정규화하기 위한 미세 조정된(layout parser) 레이아웃 파서를 사용하며, 빠른 추론을 위해 병렬 프롬프트(parallel prompting)를 기반으로 한 효율적인 LLM 추출기를 통합합니다. 또한 새로운 벤치마크 데이터셋을 지원하는 강력한 2단계 자동 평가 프레임워크로 특징 지어집니다.

- **Technical Details**: 시스템은 다양한 이력서 레이아웃을 처리하기 위해 PDF 메타데이터와 OCR 내용을 융합하는 통합 레이아웃 파싱 모델을 소개합니다. 또한 효율적인 LLM 추출을 위해 작업 분해(task decomposition) 전략을 채택하여 토큰 사용량(token usage)과 응답 시간을 줄이도록 설계되었습니다. 마지막으로, 헝가리안 알고리즘(Hungarian algorithm)을 사용하여 엔티티 정렬(entity alignment)과 다중 전략(field matching)을 적용한 2단계 평가 프레임워크를 개발했습니다.

- **Performance Highlights**: 실험 결과, 저희 시스템은 정확성과 효율성 모두에서 최신 기술을 뛰어넘는 성능을 보여주었습니다. 특히, 미세 조정된 Qwen3-0.6B-SFT 모델은 Claude-4와 같은 최고 모델의 정확도를 초과하면서도 3-4배 빠른 추론 속도를 기록했습니다. 전체 시스템은 Alibaba의 스마트 인사 플랫폼에 완전히 배포되어 있으며, 실시간 이력서 파싱을 지원합니다.



### Deep Neural Networks Inspired by Differential Equations (https://arxiv.org/abs/2510.09685)
Comments:
          35 Pages, 3 figures

- **What's New**: 이번 논문에서는 심층 신경망(deep neural networks) 아키텍처와 확률적 동적 모델링 방법을 미분 방정식(differential equations)의 관점에서 종합적으로 리뷰합니다. 특히, 일반 미분 방정식(ODEs)와 확률적 미분 방정식(SDEs)을 기반으로 한 모델을 분석하고 이들의 특성과 성능을 비교합니다. 이러한 접근 방식은 해석 가능성(interpretability)과 일반화(generalization) 성능 향상에 기여할 새로운 연구 방향을 제안합니다.

- **Technical Details**: 본 논문에서는 ODE와 SDE 기반의 신경망 디자인 방법론을 체계적으로 정리하며, ODE를 통해 신경망 아키텍처를 이해하고 설계하는 데 필요한 이론적 틀을 제시합니다. ODE 기반의 모델은 예를 들어 ResNet과 같은 연속적인 진화(evolution)를 통해 물리적 직관(intuition)과 해석 가능성을 개선합니다. SDE는 확률적 시스템을 도입함으로써 경로의 변동성을 추가하여 다양한 분포적 변화를 포착하고, 데이터 생성 분야에서도 중요한 역할을 수행합니다.

- **Performance Highlights**: 연구에서 제안하는 모델들은 여러 실험 작업에서 DE 기반 모델들의 성능을 비교하면서 우수한 해석 가능성과 일반화 능력을 보여줍니다. ODE와 SDE를 통한 규제 기법(stochastic regularization methods)은 네트워크 구조와 상호작용하여 성능을 더욱 향상시킵니다. 저자들은 앞으로 신경망에서의 무작위성(randomness)의 역할을 더 깊이 탐구해야 한다고 강조하며, 이러한 접근이 심층 학습의 미래 발전에 중요한 기여를 할 것이라고 전망합니다.



### Semantic-Cohesive Knowledge Distillation for Deep Cross-modal Hashing (https://arxiv.org/abs/2510.09664)
- **What's New**: 최근의 연구에서는 SODA라는 새로운 심Semantic cohesive knowledge distillation 스킴이 제안되었습니다. 이 방법은 다중 레이블 정보를 새로운 텍스트 모달리티로 도입하며, 이미지와 레이블 간의 상호작용을 최적화하여 해시 코드 학습을 개선합니다. SODA는 이미지와 텍스트 모달리티 간의 의미적 유사성을 효과적으로 보존하는 해밍 스페이스를 학습하는 데 기여합니다.

- **Technical Details**: SODA에서는 두 단계의 크로스 모달 Teacher-Student 네트워크를 설계하여 다양한 모달리티 간의 의미적 특성을 포착합니다. 여기서 다중 레이블 정보를 ground-truth 레이블 프롬프트로 재구성하여, 이미지에 표현된 의미적 요소를 더 직관적으로 묘사합니다. 이미지 해밍 스페이스는 레이블 모달리티와 긴밀히 연결되어, 서로 간의 의미적 관련성을 극대화합니다.

- **Performance Highlights**: SODA는 두 개의 벤치마크 데이터셋에서 최신 기법들보다 우수한 성능을 보이고 있음을 실험적으로 입증하였습니다. 이 모델은 기존의 방법들이 간과했던 다중 레이블의 풍부한 의미적 정보를 활용하여 우수한 검색 성능을 달성합니다. 특히 SODA는 이미지과 텍스트 모달리티 간의 해시 코드 유사성을 효과적으로 학습하여 성능 향상에 기여합니다.



### Learning What Matters: Steering Diffusion via Spectrally Anisotropic Forward Nois (https://arxiv.org/abs/2510.09660)
- **What's New**: 이번 연구에서는 Diffusion Probabilistic Models (DPMs)에 대한 유도 편향(inductive bias)을 명시적으로 설계하여 데이터의 목표 분포에 더 잘 맞도록 훈련 및 샘플링 과정을 개선하고자 합니다. 특히, 등방(isotropic) forward covariance를 대체할 구조화된 비등방(anisotropic) 잡음 연산자를 도입하여 주파수 대각선 주파수(frequency-diagonal) 코바리언스를 적용했습니다. 이러한 접근을 통해 우리는 지정된 주파수 대역을 강조하거나 억제하면서도 forward 과정은 Gaussian 형태를 유지할 수 있습니다.

- **Technical Details**: 연구에서는 Spectrally Anisotropic Gaussian Diffusion (SAGD)이라는 새로운 비등방 가우시안 조작자를 제안하였습니다. 이 조작자는 Fourier 기반으로 구조화된 공분산을 포함하고 있어, 잡음 첨가 과정에서 특정 주파수 대역의 정보를 조정할 수 있도록 합니다. 결과적으로, 이 방법은 모델이 특정 데이터 분포의 특징을 더 효과적으로 학습하도록 돕습니다. DPM의 기존 구조에 몇 줄의 코드로 통합이 가능하여 전체 파이프라인을 손상시키지 않는 장점이 있습니다.

- **Performance Highlights**: SAGD를 활용한 모델은 여러 자연 이미지 데이터셋에서 전통적인 확산 모델보다 더 나은 성능을 보여줍니다. 이 방법은 주파수 대역을 선택적으로 무시하면서도 학습할 수 있다는 점에서 획기적입니다. 실험 결과, SAGD를 통해 학습된 모델은 알려진 오염을 무시하고 깨끗한 분포를 회복하는 능력까지 발휘하고 있습니다. 최종적으로 이러한 결과는 DPM의 유도 편향을 정교하게 설계하는 것이 성능 향상에 크게 기여할 수 있음을 보여줍니다.



### Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models (https://arxiv.org/abs/2510.09658)
- **What's New**: 이 연구는 새로운 파라미터 전이 방법인 GradFix를 소개합니다. GradFix는 특정 작업에 대한 모델의 적응을 포착하는 파라미터 변화(태스크 벡터)를 활용하여, 이전에 교육된 모델에서 최신 모델로의 지식 전이를 가능하게 합니다. 특히, 기존의 파인튜닝을 반복할 필요 없이 라벨된 샘플 몇 개만으로도 지식을 효과적으로 이전할 수 있습니다.

- **Technical Details**: GradFix는 목표 모델의 경량 손실 구조와 기울기 신호를 이용하여 태스크 벡터를 마스킹하는 방법입니다. 이를 통해, 목표 모델의 지역 손실 구역에 맞게 태스크 벡터를 재기반화하여 업데이트를 생성합니다. 이 과정은 목표 모델의 손실을 감소시킨다는 이론적 보증을 제공하며, 그리드 정보를 기반으로 효과적인 전이를 달성합니다.

- **Performance Highlights**: 실험 결과, GradFix는 시각 및 자연어 처리 벤치마크에서 기존의 간단한 태스크 벡터 추가 및 소수 샘플 파인튜닝과 비교할 때, 효과적인 성능 향상을 보여주었습니다. 낮은 데이터 상황에서도 작업 지식의 효과적인 이동을 가능하게 하여, 새로운 프리트레인 모델에서의 변화를 최소화할 수 있습니다.




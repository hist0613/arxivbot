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



New uploads on arXiv(cs.LG)

### Reinforced sequential Monte Carlo for amortised sampling (https://arxiv.org/abs/2510.11711)
Comments:
          code: this https URL

- **What's New**: 이 논문은 비정규 밀도 함수(unnormalised density functions)로 정의된 분포에서 샘플링하기 위한 감가 상각(amortised) 및 입자 기반(particle-based) 방법의 시너지를 제안합니다. 이 연구는 순차 몬테 카를로(sequential Monte Carlo, SMC)와 최대 엔트로피 강화 학습(maximum-entropy reinforcement learning, MaxEnt RL)으로 훈련된 신경 순차 샘플러(neural sequential samplers) 간의 연결을 명확히 합니다. 또한, 이 연결을 활용하여, 샘플의 행동 정책을 개선할 수 있는 오프 정책(off-policy) RL 훈련 절차를 소개합니다.

- **Technical Details**: 이 방법론에서는 샘플러가 제안하는 제안 커널(proposal kernels)과 비틀기 함수(twist functions)를 정의하며, 안정적인 제안 및 비틀기 함수의 공동 훈련(joint training) 기술을 설명합니다. 또한, 훈련 신호의 분산을 줄이기 위한 적응형 가중치 온도 조정(adaptive weight tempering) 방안을 제시합니다. 마지막으로, 경험 재생(experience replay) 과거 샘플을 활용하여 신경 샘플러의 훈련을 안내하는 방법을 도출했습니다.

- **Performance Highlights**: 합성 다중 모드 타겟(synthetic multi-modal targets) 및 알라닌 다이펩타이드(conformations of alanine dipeptide)의 볼츠만 분포(Boltzmann distribution)에서, 제안된 방법은 실제 분포에 대한 근사화 및 훈련 안정성에서 감가 상각 방법과 몬테 카를로 방법 모두에 비해 개선된 성과를 보여줍니다. 이 연구는 분포 탐색의 효율성을 높이며, 강화 학습 기반의 방법론이 샘플링 품질을 향상시킬 수 있음을 입증하고 있습니다.



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



### Tight Regret Upper and Lower Bounds for Optimistic Hedge in Two-Player Zero-Sum Games (https://arxiv.org/abs/2510.11691)
Comments:
          29 pages, 2 figures

- **What's New**: 이 논문에서는 두 플레이어 제로섬 게임에서 optimistic Hedge 알고리즘의 최적성에 관한 새로운 분석을 제공합니다. 기존의 사회적 및 개인적 망각 경계는 O(log(mn))에서 O(√(log m log n))로 개선될 수 있습니다. 이 개선은 각 플레이어가 상대의 행동 수를 아는 강하게 uncoupled 설정에서 이루어집니다. 또한, 기존 경계의 개선 가능성과 망각 하한을 제시하여 이를 강화합니다.

- **Technical Details**: 논문에서는 망각 분석을 보다 정교하게 다루고, 학습 속도와 특정 부정 항의 계수에 대한 최적화 문제로 표현합니다. 이는 망각 경계의 주도하는 상수에 대한 세부 분석을 가능하게 합니다. 제안된 접근법은 게임에서 log m과 log n이 불균형인 경우 특히 효과적이며, 각 플레이어의 행동 수가 기하급수적으로 클 때 유용합니다. 이 분석은 모델 학습 동역학의 미세 조정을 통해 감지된 정량적 개선을 설명합니다.

- **Performance Highlights**: 실험 결과는 cardinality-aware 접근이 사회적 망각 및 개인적 망각의 최대 값을 모두 개선한다는 것을 보여줍니다. 최적 하한이 기존의 사회적 망각 경계와 일치함을 확인하며, 이러한 접근이 게임 이론적으로 중요하게 작용할 수 있음을 시사합니다. 또한, 마지막 반복의 수렴률을 향상시키는 데도 기여하고, 이는 알로리즘에 따라 동적인 망각 하한과 정확히 매칭된다는 특징이 있습니다.



### Representation-Based Exploration for Language Models: From Test-Time to Post-Training (https://arxiv.org/abs/2510.11686)
Comments:
          Website and code: this https URL

- **What's New**: 본 논문에서는 강화 학습(Reinforcement Learning, RL)이 언어 모델의 탐색을 어떻게 개선할 수 있는지를 조사합니다. 특히, 사전 학습된 언어 모델의 숨겨진 상태에서 파생된 단순하지만 원칙적인 보너스를 통해 새로운 행동을 발견하도록 모델을 유도하는 deliberate exploration에 초점을 맞추었습니다. 이 접근 방식은 단순히 기존 행동을 강화하는 것을 넘어서는 잠재력을 가지고 있습니다.

- **Technical Details**: 논문에서는 의도적인 탐색 기법을 통해 모델이 참신하고 다양한 행동을 발견하도록 장려하는 방법을 제시합니다. 이 방법은 사전 학습된 모델의 표현 기반 보너스를 활용하여 모델의 다양성과 pass@k 비율을 크게 향상시키는 것으로 나타났습니다. 연구는 추론 기간(inference-time)과 사후 훈련(post-training) 모두에서 이 기법의 효과를 보여줍니다.

- **Performance Highlights**: 결과적으로, Qwen-2.5-14b-Instruct 모델에 대한 추론 기간의 탐색이 표준 샘플링에 비해 50% 이상의 효율 개선을 가져왔으며, AIME 2024 경쟁에서 Qwen-2.5-7b-Instruct 모델은 기존 모델보다 3배 향상된 샘플 효율성을 보여주었습니다. 이러한 성과는 deliberate exploration이 새로운 행동을 발견할 수 있는 실질적인 경로가 될 수 있음을 시사합니다.



### Boundary-Guided Policy Optimization for Memory-efficient RL of Diffusion Large Language Models (https://arxiv.org/abs/2510.11683)
- **What's New**: 최근 확산 대형 언어 모델(dLLMs)은 기존의 자기회귀 모델(ARMs)에 대한 유망한 대안으로 부각되고 있으며, 다양한 언어 모델링 작업에서 경쟁력 있는 성능을 보여주고 있습니다. 그러나 기존의 연구들은 주로 dLLMs의 사전 학습과 감독 학습에 초점을 맞추고 있으며, 강화 학습(RL)을 이용한 dLLMs의 성능 개선은 여전히 도전 과제로 남아 있습니다. 본 연구에서는 Boundary-Guided Policy Optimization (BGPO)이라는 새로운 메모리 효율적인 RL 알고리즘을 제안하여, dLLMs에 대한 log-likelihood와 RL 목표의 근사를 지원합니다.

- **Technical Details**: BGPO는 ELBO 기반 목표의 하한을 최대화하도록 설계되었습니다. 이 하한은 두 가지 주요 속성을 만족하도록 만들어졌습니다: (1) 선형성(Linearity): 각 항이 단일 MC 샘플에만 의존하는 형태로 구성되어 있어, 샘플 간의 그래디언트 누적이 가능하고 메모리 사용이 일정하게 유지됩니다; (2) 동등성(Equivalence): 이 하한의 값과 그래디언트는 on-policy 훈련에서 ELBO 기반 목표의 값과 그래디언트가 같아, 원래의 RL 목표를 효과적으로 근사할 수 있게 됩니다. 이러한 특성 덕분에 BGPO는 큰 MC 샘플 크기를 채택하여 보다 정확한 RL 목표 근사가 가능해집니다.

- **Performance Highlights**: BGPO는 LLaDA-8B-Instruct 모델을 사용한 수학 문제 해결, 코드 생성 및 계획 작업에서 이전 RL 알고리즘과 비교해 상당한 성능 향상을 보여줍니다. 범위가 넓은 MC 샘플 크기를 활용함으로써 그래디언트의 편향과 분산을 효과적으로 줄여 모델 성능을 향상시키는 결과를 도출했습니다. 또한, BGPO는 샘플 크기가 증가하더라도 평균 훈련 단계 시간이 소폭만 증가하여 효율성을 유지했습니다.



### Chronologically Consistent Generative AI (https://arxiv.org/abs/2510.11677)
- **What's New**: 본 논문에서는 lookahead bias를 제거하기 위한 일련의 시간적으로 일관된, 지침을 따르는 대규모 언어 모델(LLM)이 소개됩니다. 각 모델은 명확하게 정의된 지식 컷오프 날짜 이전에 사용 가능한 데이터만으로 훈련되어, 컷오프 이후의 데이터와의 엄격한 시간적 분리를 보장합니다. 이러한 프레임워크는 간단한 대화형 챗 인터페이스, 완전히 공개된 고정 모델 가중치 및 예측 정확도에 대한 보수적인 하한을 제공합니다.

- **Technical Details**: 연구팀은 훈련 데이터가 미래 지식을 포함하지 않도록 데이터셋을 신중하게 구성하여 첫 번째 지침 따르는 대화 모델인 ChronoGPT-Instruct를 개발했습니다. 이 모델은 사전 훈련(pretraining) 및 지침 미세 조정(instruction finetuning) 단계에서 컷오프 날짜 이후의 지식을 절대적으로 찾을 수 없도록 구성되어 있습니다. 또한, 훈련 세트와 평가 세트 간의 독립성 조건을 유지하기 위한 두 단계 설정을 수립하였습니다.

- **Performance Highlights**: ChronoGPT-Instruct 모델 시리즈는 Alpaca 지침 따르기 평가에서 12% 이상의 승률을 기록하며 실용성을 보여주었습니다. 모델은 훈련 누수 없이 미래 이벤트나 대통령 예측을 하지 못하는 것으로 나타났으며, 이는 모델의 성능을 증가적으로 높일 수 있음을 시사합니다. 결과적으로 이들은 lookahead bias 없는 유용한 도구로 자주 사용될 수 있으며, 연구자들에게 제공되는 공공 데이터는 이러한 예측 문제를 보다 정확하게 분석하는 데 기여합니다.



### An Eulerian Perspective on Straight-Line Sampling (https://arxiv.org/abs/2510.11657)
- **What's New**: 본 논문은 생성적 모델링을 위한 동적 측정 수송(dynamic measure transport)에 대해 연구하고 있습니다. 특히, 특정 출발 분포와 목표 분포를 연결하는 확률적 과정에 의해 유도된 흐름(flow)에 집중하고 있습니다. 이 과정의 속도(velocity) 조건부 기대값이 정의하는 ODE(Ordinary Differential Equation)를 통해 원하는 수송을 달성할 수 있음을 보여줍니다.

- **Technical Details**: 저자들은 직선 흐름(straight-line flow)을 생성하는 확률적 과정을 식별하기 위한 새로운 PDE(Partial Differential Equation) 기준을 제안하고 있습니다. 이를 통해 조건부 가속도와 가중 공분산(Reynolds) 텐서의 발산(divergence) 간 균형을 통해 직선성을 완전히 특성화합니다. 특히, 결정론적 종단 결합(deterministic endpoint couplings) 하에서 직선성이 정확히 발생한다는 점을 입증했습니다.

- **Performance Highlights**: 이 연구는 일반적인 프로세스에 대한 흐름 기하학(flow geometry)을 규제하는 필수 조건을 도출함으로써, 통합이 더 용이한 수송 설계에 대한 넓은 지침을 제공합니다. 따라서 이 논문은 모든 재배열(rearrangement)에 대한 알골 방법(algorithmic frameworks)을 제공하며, 생성적 모델링의 이론적 및 알고리즘적 후속 연구에 기반을 마련합니다.



### MATH-Beyond: A Benchmark for RL to Expand Beyond the Base Mod (https://arxiv.org/abs/2510.11653)
- **What's New**: DeepSeek-R1의 출현으로 인해 강화 학습 방법이 새로운 수학적 추론 능력을 열어주는 새로운 물결을 맞이했습니다. 그러나 많은 오픈소스 모델이 MATH-500 및 AIME 2024와 같은 일반적인 수학 벤치마크에서 거의 모든 질문을 해결할 수 있다는 제한이 드러났습니다. 이는 현재의 RL 미세 조정 방법이 기존의 솔루션 방식만을 강화할 뿐, 전혀 새로운 방식을 발견하는 데는 한계가 있음을 강조합니다. 이를 극복하기 위해 MATH-Beyond (MATH-B)를 소개하며, 이는 기존 모델보다 더 높은 추론 능력이 요구되는 새로운 벤치마크입니다.

- **Technical Details**: MATH-B는 고등학교 수준의 수학 문제를 대상으로 하여, 인기 있는 개방형 가중치 모델들이 1024회 시도하더라도 해결하지 못할 문제로 설계되었습니다. 이 데이터셋은 DAPO-Math-17K와 DeepScaleR에서 선정된 문제들로 구성되어 있으며, 그 주제는 기존 벤치마크와 일치합니다. 게다가, 문제들은 GPT-5-Mini와 o4-mini-high와 같은 강력한 추론 모델을 통해 검증되어 정확성을 담보합니다.

- **Performance Highlights**: RL 미세 조정 모델인 Nemotron-Research-Reasoning-Qwen-1.5B 및 DeepScaleR-1.5B-Preview가 MATH-B에서 낮은 성과를 보임으로써 현재 접근법의 한계를 나타냅니다. 이러한 결과는 새로운 접근 방식이 기존 모델보다 더 발전된 추론 능력을 요구한다는 필요성을 시사합니다. MATH-B는 탐색 기반의 RL 접근 방식을 촉진하여 더 깊은 추론 능력을 이끌어내기를 기대합니다.



### Attention Factors for Statistical Arbitrag (https://arxiv.org/abs/2510.11616)
Comments:
          Accepted to the 6th ACM International Conference on AI in Finance

- **What's New**: 이 논문에서는 통계적 차익 거래(statistical arbitrage)를 위한 새로운 프레임워크를 개발하였습니다. 특히, 'Attention Factors'라는 조건부 잠재 요인을 도입하여 유사한 자산을 식별하고 잘못된 가격을 파악하는 동시에 거래 비용 이후 극대화된 리스크 조정 성과를 위한 거래 정책을 수립합니다. 기존의 두 단계 접근 방식 대신, 우리는 하나의 단계에서 거래 가능한 차익 요인(tradable arbitrage factors)과 포트폴리오 배치를 공동으로 학습하는 방식을 제안합니다.

- **Technical Details**: 이 모델은 복잡한 상호작용을 허용하는 기업 특성의 임베딩(embeddings)을 통해 요인을 학습합니다. 또한, 일반 시퀀스 모델을 통해 시간 시계열 신호를 식별하여 순수한 Sharpe 비율을 극대화하는 것을 목표로 합니다. 24년간의 미국 상장주식 데이터를 활용한 실증 분석을 통해, 논문에서 제안한 Attention Factor 모델은 거래 비용 없이 4를 초과하는 Sharpe 비율을 달성했습니다.

- **Performance Highlights**: 논문에서는 Attention Factor 모델이 연간 16%의 수익률을 달성하면서도 시장 리스크와는 독립적인 특성을 보인다고 주장합니다. 거래 비용을 반영한 경우에도 2.3의 Sharpe 비율을 달성하며, 이는 기존 모델들보다 84% 증가한 수비적 성과를 보입니다. 특히, 이 모델은 산업 부문과 밀접한 관련이 있는 해석 가능한 구조를 가지고 있으며, 적은 변동성을 가진 약한 요인들이 중요한 역할을 한다는 점을 밝혀냈습니다.



### Diffusion-DFL: Decision-focused Diffusion Models for Stochastic Optimization (https://arxiv.org/abs/2510.11590)
- **What's New**: 이번 논문에서는 예측 모델링과 최적화를 통합한 결정 중심 학습(Decision-Focused Learning, DFL) 접근법의 새로운 발전을 제안합니다. 특히, 확산 모델(diffusion model)을 활용하여 불확실한 매개변수의 분포를 표현하고, 이를 통해 스토캐스틱 최적화 문제를 해결하는 방법을 도입하였습니다. 이러한 접근은 기존의 DFL 방식과 다르게 점 예측(point prediction)을 넘어서 불확실성을 반영하는 보다 정교한 모델링을 가능하게 합니다.

- **Technical Details**: 제안된 확산 DFL 방법은 재매개변수화 기법(reparameterization trick)을 통해 확산 모델의 샘플링 과정을 포함한 엔드 투 엔드(end-to-end) 훈련을 가능하게 합니다. 이를 통해 불확실한 매개변수를 다루는 데 필요한 메모리와 계산 비용을 대폭 줄일 수 있는 경량의 확률 함수(score function) 추정기를 도입하였습니다. 이러한 기법은 샘플링 과정을 통해 그래디언트를 역전파(backpropagation)하는 방식에서 발생하는 비효율성을 해결합니다.

- **Performance Highlights**: 실험 결과, 제안된 확산 DFL 방법이 여러 응용 분야에서 강력한 기준선들을 일관되게 초과하는 성과를 보여주었습니다. 특히 대규모 문제에 대해 더 큰 개선을 이루어냈으며, 확률 함수 추정기가 재매개변수화 방법과 유사한 의사결정 품질을 달성하면서 GPU 메모리 사용량을 60.75 GB에서 0.13 GB로 크게 줄였습니다. 이러한 결과는 본 연구가 DFL에서 확산 모델을 효과적으로 활용하고 있음을 뒷받침합니다.



### Ontolearn-A Framework for Large-scale OWL Class Expression Learning in Python (https://arxiv.org/abs/2510.11561)
- **What's New**: 이번 논문에서는 대규모 지식 그래프에서 OWL 클래스 표현을 학습하기 위한 Ontolearn이라는 프레임워크를 소개합니다. Ontolearn은 EvoLearner와 DRILL을 포함한 최신 기호적 및 신경-기호적 클래스 표현 학습기의 효율적인 구현을 포함하고 있습니다. 또한 복잡한 OWL 클래스 표현을 자연어 문장으로 변환하는 LLM 기반의 동사화 모듈을 통합하여 사용자 친화성을 높였습니다.

- **Technical Details**: Ontolearn은 대규모 RDF 지식 그래프에 대한 OWL 클래스 표현 학습을 용이하게 하는 오픈 소스 Python 라이브러리입니다. 이 프레임워크는 다양한 최신 심볼릭, 신경-심볼릭, 딥러닝 알고리즘을 제공하며, OWL 이유자를 활용하여 클래스 표현 학습을 지원합니다. 또한, Ontolearn은 SPARQL 쿼리로 매핑하여 원격 트리플 저장소에서 데이터를 효율적으로 검색할 수 있도록 설계되었습니다.

- **Performance Highlights**: Ontolearn은 156개의 단위 및 회귀 테스트를 포함하여 95%의 테스트 커버리지를 자랑하는 잘 테스트된 프레임워크입니다. 현재 Ontolearn은 26,000회 이상 다운로드 되었으며, 고유한 사용 예제를 제공하여 새로운 사용자가 쉽게 활용할 수 있도록 지원합니다. 산업 프로젝트에서도 이미 적용되었으며, 자동화된 인간 해석 가능 설명을 통해 생산 공정의 기술 매칭에 중요한 역할을 하고 있습니다.



### Query-Specific GNN: A Comprehensive Graph Representation Learning Method for Retrieval Augmented Generation (https://arxiv.org/abs/2510.11541)
- **What's New**: 본 연구에서는 Retrieval-Augmented Generation (RAG) 시스템의 성능을 향상시키기 위해 Multi-information Level Knowledge Graph (Multi-L KG)를 설계하였습니다. Multi-L KG는 다층의 정보를 통해 다단계 질문을 더 효과적으로 이해할 수 있는 기반을 제공합니다. 또한 Query-Specific Graph Neural Network (QSGNN)를 도입하여 상관 정보의 경량화된 전파와 다수의 정보 집계를 가능하게 하였습니다.

- **Technical Details**: 이 연구는 다단계 질문을 처리하기 위해 Multi-L KG에서 여러 정보 수준과 복잡한 관계를 포착하는 것에 중점을 두고 있습니다. QSGNN은 두 가지 메시지 전파 방식인 intra-level과 inter-level을 사용하여 각 수준 내의 기본 의미 관계 및 수준 간의 지역-전역 관계를 고려합니다. 이러한 디자인은 응답의 질을 높이고 노이즈의 영향을 줄이는 데 기여합니다.

- **Performance Highlights**: QSGNN의 성능은 기존 방법들과 비교했을 때 특히 고복잡도 문제에서 33.8%의 성능 향상을 보여주며, 이는 RAG 시스템의 다단계 질문 처리에서의 가능성을 입증합니다. 광범위한 실험 결과는 제안된 프레임워크의 효과를 입증합니다.



### Knowledge-Guided Machine Learning Models to Upscale Evapotranspiration in the U.S. Midwes (https://arxiv.org/abs/2510.11505)
- **What's New**: 이번 연구는 여러 공간적 및 시간적 스케일에서 증발산 (Evapotranspiration, ET)의 정확한 정량화를 제시하며, 이는 토지-대기 상호작용에서 중요한 역할을 합니다. 특히, 미드웨스트 미국 지역에서 ET를 업스케일하기 위해树 기반( tree-based) 및 지식 기반(machine learning) 기술을 통합한 접근법이 사용되었습니다. 이를 통해 기후 대역 (gridded meteorology) 및 에디 공변량 (eddy covariance) 데이터를 활용하여 각 필드의 ET를 측정할 수 있는 새로운 방법론이 제안되었습니다.

- **Technical Details**: 연구에서는 랜덤 포레스트(Random Forest), CatBoost, XGBoost 및 LightGBM 등 네 가지 트리 기반 모델과 간단한 피드 포워드 인공신경망(artificial neural network)을 비교하였습니다. 알고리즘은 EC 타워에서 수집한 데이터를 k-fold cross validation(k=5) 기법을 사용하여 훈련하고 테스트하였으며, 데이터 누수를 방지하기 위해 사이트 연도 및 생물군계(biome) 기준으로 분할되었습니다. 결과적으로 LightGBM 모델이 지식 기반 특징을 활용하여 다른 방법들보다 우수한 성능을 나타냈습니다.

- **Performance Highlights**: LightGBM은 R²=0.86, MSE=14.99 W m^-2 및 MAE=8.82 W m^-2의 성과를 거두었고, 데이터 분석 결과 지식 기반 특징이 증발산 예측에 가장 큰 영향을 미쳤음을 보여주었습니다. 최상의 성능을 보인 모델을 기반으로 2019-2024년 기간 동안 500m 공간 해상도와 1일 시간 해상도를 가진 그리드 형태의 ET 데이터 제품을 제공합니다. 새로운 데이터 제품과 주(state) 기반 날씨 관측소의 ET 추정치 간의 비교 결과, 탁월한 일치를 보였습니다.



### Learning to Make MISTAKEs: Modeling Incorrect Student Thinking And Key Errors (https://arxiv.org/abs/2510.11502)
- **What's New**: 이 논문은 MISTAKE라는 새로운 방법을 제시하여 학생의 비논리적 사고를 모델링하는 데 중점을 둡니다. 기존의 언어 모델(LM) 연구는 정확한 출력 향상에 주요 초점을 맞추었으나, 학생들이 흔히 발생하는 오류를 이해하고 시뮬레이션하는 것에 대한 필요성이 강조됩니다. 이 방법은 오류와 근본적인 오해를 연결하여 고품질의 비논리적 예시를 생성하고 이를 기반으로 학습 모델을 구축하는 방식으로 진행됩니다.

- **Technical Details**: MISTAKE 방법은 크기의 일관성(cycle consistency)을 활용하여 잘못된 답변과 그에 따른 오해를 결합합니다. 이를 통해 학생의 비논리적 사고를 시뮬레이션할 수 있는 모델과 오해를 추론할 수 있는 모델을 훈련하게 됩니다. 방법은 두 개의 절차, 즉 mistake-Generate와 mistake-Update로 구성되어 있으며, 각각은 비논리적 사고 데이터를 생성하거나 이를 기반으로 모델을 세분화하는 역할을 합니다.

- **Performance Highlights**: MISTAKE는 세 가지 교육 과제에서 성능을 평가하였고 결과적으로 정확도가 최대 9% 향상되었으며, 오해 추론에서 15% 개선을 달성했습니다. 또한, 생성된 잘못된 선택지는 전문가가 작성한 선택지와의 정밀도에서 64.6% 향상된 결과를 보였습니다. 이러한 결과는 다양한 교육 영역에서 잘못된 사고 패턴을 명시적으로 모델링하는 것이 큰 잠재력을 가지고 있다는 점을 보여줍니다.



### Context-Aware Model-Based Reinforcement Learning for Autonomous Racing (https://arxiv.org/abs/2510.11501)
Comments:
          Accepted to IEEE ICAR 2025

- **What's New**: 이 연구는 자율 주행 환경에서 모델 기반 강화 학습 알고리즘(Model-Based Reinforcement Learning, MBRL)의 성능과 일반화 능력을 평가하는 것을 목적으로 하고 있습니다. 특히 Roboracer라는 시뮬레이션 자율 경주 환경에서의 조정된 마르코프 의사 결정 과정(Contextual Markov Decision Processes)을 적용하여 경쟁적 경주 작업을 학습 문제로 설정합니다. 이 과정에서 'cMask'라는 새로운 맥락 인식 알고리즘을 제안하여 기존 MBRL 접근 방식보다 더 나은 일반화 능력을 입증합니다.

- **Technical Details**: MBRL 알고리즘은 환경 모델을 학습하고 이로부터 수익을 극대화하는 방식을 사용합니다. 연구에서는 대칭경주(task)에서의 적대자의 행동을 맥락(context)을 통해 매개변수화하고 전이 및 보상 역학을 동적으로 조정합니다. 또한, cMask는 SAC(Soft Actor-Critic) 네트워크를 사용하여 에피소드의 맥락 값을 재매핑한 후, 이를 세계 모델에 적용하여 일반화 능력을 높입니다.

- **Performance Highlights**: 실험 결과, cMask 알고리즘은 맥락 없는 접근 방식보다 타인과의 상호작용에서 더 안전하고 일반화 능력이 뛰어난 정책을 생성합니다. 또한, cMask는 in-distribution 행동을 보이는 적들과 경주 시 다른 맥락 인식 MBRL 접근 방식보다 더 우수한 성능을 보여주었습니다. 이를 통해 자율주행 및 기타 안전-critical 로봇 기술에서 MBRL 알고리즘의 활용 가능성을 제시합니다.



### Offline Reinforcement Learning with Generative Trajectory Policies (https://arxiv.org/abs/2510.11499)
Comments:
          Preprint. Under review at ICLR 2026

- **What's New**: 이번 연구에서는 생성 모델을 활용한 오프라인 강화 학습에서 정책의 효율성과 표현력을 동시에 충족할 수 있는 새로운 프레임워크인 Generative Trajectory Policies (GTPs)를 제안합니다. 기존의 느리고 반복적인 생성 방식과 빠르지만 성능이 떨어지는 단일 단계 방식 사이의 격차를 메우기 위한 방법을 탐구하여, 여러 현대 생성 모델들을 연속 시간 생성 경로로 이해할 수 있는 통합 관점을 제공합니다.

- **Technical Details**: 연구의 핵심은 생성 경로를 지배하는 일반 미분 방정식(Ordinary Differential Equation, ODE)으로 정의된 연속 시간 생성 모델의 통합 프레임워크입니다. GTP는 이 ODE의 전체 솔루션 맵을 학습하여 느리고 높은 충실도의 샘플링과 빠르고 낮은 충실도의 단축키를 넘어서 유연하고 다단계 결정론적 생성을 가능하게 합니다. 이를 위해, 연구진은 두 가지 이론적으로 기반한 방법론을 통해 계산 비용과 훈련 불안정성을 해결했습니다.

- **Performance Highlights**: GTP는 D4RL 벤치마크에서 최첨단 성능을 달성하였으며, 기존 생성 정책들보다 높은 성능을 기록했습니다. 특히, 여러 현실적으로 도전적인 AntMaze 작업에서 완벽한 점수를 기록하며 표현력과 효율성 간의 균형을 더 잘 맞추는 능력을 입증하였습니다.



### ReLook: Vision-Grounded RL with a Multimodal LLM Critic for Agentic Web Coding (https://arxiv.org/abs/2510.11498)
- **What's New**: 이 논문은 대규모 언어 모델(LLMs)이 알고리즘 코드 생성에서는 우수하지만 프론트엔드 개발에서는 어려움을 겪고 있다는 점을 지적합니다. 이를 해결하기 위해, ReLook라는 비전 기반 강화학습 프레임워크를 도입하여 에이전트가 다중 모드 LLM(MLLM)을 도구로 활용하여 코드를 생성하고 진단 및 개선하는 반복 작업을 수행할 수 있도록 합니다. 특히, 이 방법은 프론트엔드 코드 생성의 비전-기반 성능을 크게 향상시키는 것을 목표로 합니다.

- **Technical Details**: ReLook는 MLLM을 비주얼 비평가로 사용하여 스크린샷을 통한 코드 점수 평가와 실행 가능한 비전-기반 피드백을 제공합니다. 훈련 과정에서 에이전트는 0 보상 규칙을 적용해 잘못된 렌더링에 대해 보상을 받지 않게 하여 렌더링 가능성을 보장합니다. 또한, 강제 최적화(Forced Optimization) 전략을 통해 성능 저하를 방지하고 지속적으로 향상된 경로를 유지합니다.

- **Performance Highlights**: ReLook는 세 가지 일반적인 벤치마크 테스트에서 기존 방법보다 우수한 성능을 보여줍니다. 다양한 LLM과의 통합 실험을 통해 ReLook의 호환성을 입증하였으며, 에이전트의 인식 능력 및 비주얼 보상의 효과를 강조합니다. 이러한 성과는 에이전트가 적절한 피드백을 기반으로 코드를 지속적으로 개선할 수 있도록 하는 강력한 학습 메커니즘 덕분입니다.



### How Reinforcement Learning After Next-Token Prediction Facilitates Learning (https://arxiv.org/abs/2510.11495)
- **What's New**: 최근 연구는 Large Language Models(LLM)에서 다음 토큰 예측(next-token prediction) 후 강화학습(reinforcement learning)을 적용하는 새로운 패러다임을 제시했습니다. 이 프레임워크는 LLM의 성능 향상 및 자동 회귀(transformers) 모델이 특정 작업을 수행할 때의 성공 메커니즘을 이론적으로 드러냅니다. 특히, 모델이 드물게 나타나는 긴 시퀀스에 대한 예측을 어떻게 일반화할 수 있는지를 탐구하고 있습니다.

- **Technical Details**: 본 연구는 단일 작업을 인코딩하는 짧고 긴 'chain-of-thought' 시퀀스들의 혼합 분포로부터 학습하는 방식을 분석합니다. 경험적으로, 입력 차원(d)이 크고 긴 시연이 희소할 경우, 단순한 다음 토큰 예측만으로는 효과적인 예측이 어렵다는 결과를 보여줍니다. 반면 강화 학습 추천 보상을 적용하면 모델 성능이 빠르게 향상되고 생성된 시퀀스의 길이가 증가함을 명시합니다.

- **Performance Highlights**: 실험 결과는 모델이 긴 시연에서 빠르게 학습하고, 포스트 트레이닝 과정에는 보상에 따라 긴 응답을 우선시하는 경향이 있음을 보여줍니다. 이 연구는 모델이 일반화를 성취하는 데 필요한 학습 과정을 상세히 설명하며, 여러 다양한 LLM 설계를 통해 이 현상을 입증하였습니다. 또한 이러한 결과는 수학적 추론 문제와 같은 다양한 설정에서 동일한 행동을 관찰함으로써 강화 학습의 효과를 입증합니다.



### Rescaling-Aware Training for Efficient Deployment of Deep Learning Models on Full-Integer Hardwar (https://arxiv.org/abs/2510.11484)
Comments:
          Submitted to IEEE Embedded Systems Letters

- **What's New**: 이 논문에서는 정수 AI 추론(Integer AI inference)의 계산 복잡성을 significantly (상당히) 감소시키는 방법을 제시합니다. 또한, 이전에는 간과되었던 integer rescaling의 영향을 다루며, 이 작업이 하드웨어 비용이 큰 작업이라는 점을 강조합니다. 새로운 방법론인 Rescale-Aware Training을 통해, 모델 품질 저하 없이 줄일 수 있는 quantization을 적용함으로써 rescaling 비용을 크게 줄일 수 있음을 보여줍니다.

- **Technical Details**: Quantization-aware training (QAT)은 post-training quantization에서의 정확성 저하를 완화하지만, integer-only AI inference에서의 rescaling 비용에 대한 영향을 고려하지 않습니다. 이 연구에서는 post-training 단계에서 rescale multiplicands에 강력한 quantization을 적용해, rescale 비용을 dramatically (극적으로) 감소시킬 수 있음을 입증하였습니다. 또한, ultra-low bit-width rescaling multiplicands에 대한 fine tuning 방법인 Rescale-Aware Training을 도입했습니다.

- **Performance Highlights**: 실험 결과, rescaler 폭을 8배 줄이고도 정확성을 전혀 잃지 않고 minimal incremental retraining만으로도 완전한 정확도를 유지할 수 있음을 확인했습니다. 이러한 접근은 자원이 제한된 embedded systems에서 더 에너지 효율적이고 비용 효율적인 AI 추론을 가능하게 합니다.



### Differentiable Fast Top-K Selection for Large-Scale Recommendation (https://arxiv.org/abs/2510.11472)
Comments:
          12 pages, 5 figures

- **What's New**: 이 논문에서는 추천 시스템에 사용되는 새로운 미분 가능한 Top-K 연산자 DFTopK를 제안합니다. 기존의 미분 가능한 Top-K 연산자에 비해 DFTopK는 $O(n)$의 최적 시간 복잡도로 Top-K 선택 문제를 해결하는 것이 특징입니다. 또한, DFTopK는 정렬 없이 닫힌 형태의 해를 제공합니다.

- **Technical Details**: DFTopK는 각 항목에 대해 K 개의 확률 분포를 제어하는 근사화를 통해 최적화 과정에서 항목 간의 연결을 약화시킵니다. 이는 미분 가능 정렬 기반 방법에서 발생하는 그래디언트 충돌을 줄이는 데 도움을 주며, 계산 복잡성을 선형 $O(n)$으로 줄입니다. DFTopK의 이론적 분석을 통해 해당 연산자의 장점이 검증되었습니다.

- **Performance Highlights**: DFTopK는 RecFLow 벤치마크에서 높은 경쟁력을 보여주었으며, 실제 광고 시스템에서도 A/B 테스트 결과 수익이 1.77% 증가했습니다. 또한, DFTopK는 동일한 학습 샘플 조건에서 전반적인 교육 효율성을 크게 향상시켰고, 평균 실행 시간도 15.3% 감소시켰습니다. 이러한 결과는 DFTopK의 실용적 가치를 잘 보여줍니다.



### Iterative Amortized Inference: Unifying In-Context Learning and Learned Optimizers (https://arxiv.org/abs/2510.11471)
- **What's New**: 이 논문에서는 태스크 간 재사용되는 컴퓨테이션(computation) 또는 유도 편향(inductive bias)을 통해 신속한 일반화를 가능하게 하는 통합 프레임워크를 제안합니다. 여기에는 메타 학습(meta-learning), 인-컨텍스트 학습(in-context learning), 프롬프트 튜닝(prompt tuning), 학습된 옵티마이저(learned optimizers) 등이 포함됩니다. 각 접근법은 태스크 간 정보의 인코딩(encoding) 및 활용 방식에서 차이를 보이며, 이 연구는 이러한 차이를 학습 과정에서 감가(amortization)되는 측면으로 분리하여 설명합니다.

- **Technical Details**: 우리는 세 가지 구별되는 감가 체계를 제안하는데, 이는 파라메트릭(parametric), 암묵적(implicit), 명시적(explicit) 방식입니다. 이들은 각각 유도 편향을 외부화(externalize), 내재화(internalize), 공동 모델링(jointly model)하는 방식에 따라 분류됩니다. 감가의 핵심 한계는 대규모 데이터셋에 대한 적응에서 처리 능력이 제한적이라는 점을 지적하며, 이를 해결하기 위해 stochastic optimization에 영감을 받은 반복적 감가 추론(iterative amortized inference) 모델을 제안합니다.

- **Performance Highlights**: 이 연구는 메타 학습(meta-learning)과 순방향 패스(forward-pass) 감가 기법을 연결하여 일반적인 태스크 적응의 기초를 제공합니다. 또한, 반복적 감가 체계를 도입함으로써 대규모 데이터셋에 대한 실행 가능성을 높였습니다. 이는 다음 단계로 미니 배치(mini-batch) 방법을 통해 제안된 아이디어를 구체화하였고, 최적화 문제의 처리에서 고급 유연성과 확장성을 가능하게 합니다.



### Reconstructing 12-Lead ECG from 3-Lead ECG using Variational Autoencoder to Improve Cardiac Disease Detection of Wearable ECG Devices (https://arxiv.org/abs/2510.11442)
Comments:
          24 pages, 5 figures, submitted to Nature Communications

- **What's New**: 본 연구에서는 WearECG라는 새로운 Variational Autoencoder (VAE) 방법을 제안하여 세 개의 리드(II, V1, V5)에서 12리드 ECG를 재구성합니다. 이 모델은 ECG 신호의 시간적 및 공간적 의존성을 더 잘 포착할 수 있는 구조적 개선을 포함하고 있습니다. 다양한 임상 조건을 포함한 다중 레이블 분류 작업에 대해 미리 훈련된 ECGFounder 모델을 세밀하게 조정하여 진단 유틸리티를 검증합니다.

- **Technical Details**: WearECG 모델에서는 잔차 합성곱 신경망(Residual CNNs)과 그룹 정규화(Group Normalization)와 같은 여러 기술적 접근 방식을 사용하여 신호 생성 과정을 최적화합니다. 심전도 신호의 재구성 품질은 평균 제곱 오차(Mean Squared Error, MSE), 평균 절대 오차(Mean Absolute Error, MAE) 및 Fréchet Inception Distance (FID)와 같은 측정 기준을 통해 평가됩니다. 또한 경량화된 VAE 구조를 채택하여, 제한된 입력 조건에서도 생리학적으로 가능한 신호 재구성이 가능합니다.

- **Performance Highlights**: 모델의 성능 평가에서는 평균 MSE 0.00100, MAE 0.01782, FID 12.64와 같은 우수한 결과가 나타났습니다. Turing 테스트를 통해 세 명의 심장전문의가 실제 ECG와 합성 ECG를 구별하는 과정에서 높은 임상적 관용성을 보였다고 보고되었습니다. 이러한 결과는 생성된 신호가 질병 특정 특성을 잘 보존하고 있음을 나타내며, 실제 임상 적용 가능성을 높입니다.



### Leveraging LLMs for Semi-Automatic Corpus Filtration in Systematic Literature Reviews (https://arxiv.org/abs/2510.11409)
- **What's New**: 이번 연구는 체계적인 문헌 검토(SLR)의 작성 과정을 최적화하기 위해 여러 대형 언어 모델(LLMs)을 활용한 파이프라인을 제안합니다. 문헌 검색에서 얻은 논문을 분류하고 공동으로 의사결정을 내리는 과정이 포함되어 있으며, 이를 통해 인간 감독 하에 실시간으로 결과를 수정할 수 있는 오픈 소스 인터페이스 LLMSurver를 개발했습니다. 이 연구는 8,000개 이상의 후보 논문을 대상으로 하여 기존의 방식보다 적은 수의 오류와 함께 수작업 노력을 크게 줄였습니다.

- **Technical Details**: 제안된 방법은 LLM을 활용하여 논문을 분류하고, 여러 모델의 출력을 총괄하여 최종 결정을 내리는 것을 특징으로 합니다. 텍스트 필터링 과정에서 인간의 개입을 허용하며, LLMSurver 인터페이스를 통해 사용자들이 모델의 출력을 직접 관찰하고 수정할 수 있습니다. 또한, 이 시스템은 2024년 중반과 2025년 가을의 최신 LLM 성능을 비교하여 평가하였으며, 작은 오픈 모델들도 강력한 결과를 도출할 수 있음을 입증하였습니다.

- **Performance Highlights**: 본 연구의 결과는 자동화된 파이프라인이 단일 인간 주석자보다 낮은 오류율을 유지하며 수작업 노력을 크게 단축할 수 있음을 보여줍니다. 특히 오픈 모델들이 경제성과 접근 가능성을 갖춘 해결책이 될 수 있음을 강조하고 있습니다. 사용자 정의가 가능한 프롬프트를 사용함으로써 긍정적인 결과를 얻었고, 여러 LLM을 결합한 합의 체계가 효과적임을 입증하였습니다.



### FedHybrid: Breaking the Memory Wall of Federated Learning via Hybrid Tensor Managemen (https://arxiv.org/abs/2510.11400)
Comments:
          Sensys 2024

- **What's New**: 이번 논문에서 제안하는 FedHybrid는 Federated Learning (FL) 환경에서 메모리 제약을 해결하기 위한 새로운 프레임워크입니다. FedHybrid는 훈련 과정 중 메모리 사용량을 효과적으로 줄이며 모델의 정확성을 보장합니다. 이는 각 훈련 라운드에서 참여하는 디바이스의 메모리 예산과 컴퓨팅 능력, 데이터 다양성을 종합적으로 평가하여 선택하는 방식으로 이루어집니다.

- **Technical Details**: 이 프레임워크는 메모리 제약을 고려한 계산 그래프를 분석하고 각 클라이언트에 맞는 실행 계획을 생성하여 메모리가 허용하는 범위 내에서 최적의 훈련 성능을 도모합니다. Hyper tensor management를 통해 리컴퓨테이션 및 압축 기법을 활용하여 메모리 감소와 훈련 지연 최소화를 동시에 달성합니다. 또한, 로컬 훈련 과정에서 활성 압축 기술을 적용하여 메모리 절약과 최소한의 정확도 손실을 동시에 실현합니다.

- **Performance Highlights**: 폭넓은 실험을 통해 FedHybrid가 다양한 메모리 예산 하에서 기존 방법 대비 최대 39.1%의 모델 정확도 향상과 15.5배의 훈련 시간 단축을 이룬 것으로 나타났습니다. 이는 저사양 디바이스가 포함된 FL 환경에서의 성능을 크게 향상시킴을 보여줍니다. 또한, FedHybrid는 메모리 제한이라는 문제를 극복함으로써 FL의 실용성을 더욱 높일 수 있는 가능성을 제시하고 있습니다.



### Medical Interpretability and Knowledge Maps of Large Language Models (https://arxiv.org/abs/2510.11390)
Comments:
          29 pages, 34 figures, 5 tables

- **What's New**: 이 연구는 Large Language Models (LLMs)에서 의료 분야의 해석 가능성에 대한 체계적인 연구를 제시합니다. 다양한 해석 가능성 기법을 통해 모델이 의료 지식을 표현하고 처리하는 방식을 조사하였습니다. 특히, Llama3.3-70B 모델의 첫 번째 절반의 레이어에서 대부분의 의료 지식이 처리된다는 점에서 흥미로운 결과를 도출했습니다.

- **Technical Details**: 연구는 UMAP, gradient-based saliency, layer lesioning, activation patching 등의 네 가지 해석 가능성 기법을 사용하여 LLMs의 구조를 분석하였습니다. 이를 통해 환자의 나이, 증상, 질병 및 약물에 대한 지식을 시각화한 LLM 맵을 생성했습니다. 이러한 기법들은 각각의 레이어에서 지식이 어떻게 저장되는지를 입증하는 데 효과적이었습니다.

- **Performance Highlights**: 연구 결과는 (i) 나이가 비선형적 방식으로 인코딩되고, (ii) 질병 진행의 표현이 비단조적이며 원형적임을 보여주었습니다. 또한, (iii) Llama3.3-70B의 약물 표현은 약물 작용 기전보다 의료 전문 분야와 더 잘 일치함을 확인했으며, (iv) Gemma 및 MedGemma 모델은 중간 레이어에서 활성화가 붕괴되는 현상을 관찰하였습니다. 이 결과들은 의료 관련 작업에서 LLM의 미세 조정이나 편향 제거에 대한 기초 자료를 제공합니다.



### Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks (https://arxiv.org/abs/2510.11354)
Comments:
          71 pages, 12 figures, NeurIPS 2025

- **What's New**: 본 논문은 Adam과 AdamW의 미니 배치 학습이 대규모 배치 학습과 어떻게 다른지 이론적으로 분석했습니다. 이는 기존 이론이 주로 풀 배치 버전의 Adam에 초점을 맞췄기 때문에, 실제로 사용되는 확률적 변형에 대한 이해가 부족했음을 지적합니다. 연구 결과는 Adam이 작은 배치 크기로 훈련할 경우 일반화 성능이 크게 향상되며, 이는 특히 이미지 데이터 모델에 대한 두 층 과적합(Over-parameterization) CNN에서 실험적으로 증명되었습니다.

- **Technical Details**: 이 논문에서는 두 층의 과적합 CNN에 대해 Adam과 AdamW의 수렴(convergence) 및 일반화(generalization)를 분석했습니다. 이론적으로, 대규모 배치 체계에서는 Adam과 AdamW가 낮은 테스트 오류를 가진 해결책으로 수렴하지 않음을 입증했으며, 이는 기존의 결과를 확장합니다. 반대로, 미니 배치에서의 Adam과 AdamW는 적절한 가중치 감소(weight decay)를 통해 근접한 테스트 오류를 달성할 수 있으며, 이는 두 가지 주요 메커니즘에 기인합니다: 확률적 그래디언트가 최적화 경로를 규제하고, 가중치 감소가 잔여 잡음을 억제하는 것입니다.

- **Performance Highlights**: 실험 결과는 미니 배치 학습이 Adam과 AdamW의 성능을 크게 향상시키며, 대규모 배치 학습의 경우 성능 저하와 같이 극단적인 테스트 오류 증가가 나타남을 보여주었습니다. 특히, Adam의 경우 가중치 감소 값이 크면 성능이 급격히 저하되는 반면, AdamW는 훨씬 높은 가중치 감소 값에서도 성능 저하가 거의 없음을 확인했습니다. 이러한 결과는 배치 크기와 가중치 감소의 상호작용이 일반화 성능에 미치는 중요성을 강조합니다.



### Multi-View Graph Feature Propagation for Privacy Preservation and Feature Sparsity (https://arxiv.org/abs/2510.11347)
- **What's New**: 이 논문에서는 Multi-view Feature Propagation (MFP)이라는 새로운 프레임워크를 제안하여, 노드 분류 작업을 위한 그래프 신경망(Graphic Neural Networks, GNN)에서 특징 희소성을 극복하고 개인 정보 보호를 강화합니다. MFP는 전통적인 Feature Propagation (FP) 방법을 확장하여, Gaussian 노이즈가 추가된 여러 개의 보기(view)로 나뉜 특징을 사용함으로써 정보 전파를 독립적으로 수행합니다. 이로 인해 노드 임베딩의 표현력과 견고성을 향상시킵니다.

- **Technical Details**: MFP 프레임워크는 고차원 그래프 데이터를 처리하며, 기본 개념은 각각의 보기에서 노이즈가 추가된 특징을 독립적으로 전파하는 것입니다. 이 과정에서 MFP는 기존 FP의 복원 중심 접근 방식을 탈피하고, 개인 정보가 포함된 특징의 노출을 최소화하기 위해 여러 개의 프로퍼게이션 단계를 도입합니다. 각 단계에서는 무작위로 선택된 제한적인 특징 집합이 사용되며, 이로 인해 노드 간 유용한 정보 교환이 가능한 구조로 개발됩니다.

- **Performance Highlights**: 폭넓은 실험 결과에 따르면, MFP는 고급 희소성 환경에서도 기존의 최첨단 기법을 초월하여 노드 분류 성능을 향상시키고 개인 정보 유출을 크게 줄입니다. 또한, 제안된 방법은 특히 개인 정보 보호가 중요한 상황에서도 데이터 재구성을 방지하면서도 예측 정확도를 유지하는 실제적인 이점을 제공합니다. MFP는 e-커머스 개인화, 금융 사기 탐지, 의료 분석 등의 다양한 실제 응용 분야에서 활용 가능성이 큽니다.



### Part II: ROLL Flash -- Accelerating RLVR and Agentic Training with Asynchrony (https://arxiv.org/abs/2510.11345)
- **What's New**: 최근 Synchronous Reinforcement Learning (RL) post-training 방식이 큰 언어 모델 (LLM)의 다양하고 강력한 기능을 향상시키는 중요한 단계로 자리잡았습니다. 그러나 기존 시스템들은 자원 활용도와 확장성에서 한계를 보이고 있으며, 새로운 시스템 ROLL Flash가 이를 개선하고자 합니다. ROLL Flash는 비동기식 RL post-training을 지원하여 자원 효율성과 확장성을 크게 향상시킵니다.

- **Technical Details**: ROLL Flash는 세밀한 병렬성 (fine-grained parallelism)과 rollout-train 분리 (rollout-train decoupling)라는 두 가지 핵심 설계 원칙을 기반으로 구축되었습니다. 이 시스템은 비동기식 훈련 아키텍처를 지원하는 유연한 프로그래밍 인터페이스를 제공하며, 환경 수준의 비동기 실행 및 대기열 스케줄링과 같은 효율적인 rollout 메커니즘을 포함합니다. 이를 통해 훈련 중 발생할 수 있는 대기 시간을 최소화하고 자원 활용도를 극대화합니다.

- **Performance Highlights**: 실험 결과, ROLL Flash는 기존의 동기식 RL post-training에 비해 최대 2.24배의 성능 향상을 달성하였으며, 특히 agentic 작업에서 2.72배의 속도 향상을 이루어냈습니다. 이러한 성능 개선은 비동기 훈련 방식에 의한 것으로, 응답 생성 속도가 한층 빨라지고 자원 활용도가 개선된 결과입니다. 이 연구는 다양한 RL 및 agentic 워크로드에서의 효율성과 효과성을 확인하며, 비동기 훈련 방식의 가능성을 보여줍니다.



### Event-Aware Prompt Learning for Dynamic Graphs (https://arxiv.org/abs/2510.11339)
Comments:
          Under review

- **What's New**: 이번 논문에서는 이벤트를 인지하는 동적 그래프 프롬프트 학습 프레임워크인 EVP(Event-aware Dynamic Graph Prompt learning)를 제안합니다. EVP는 기존 방법의 플러그인으로 사용되어 역사적 사건의 지식을 활용하는 능력을 향상시킵니다. 기존 동적 그래프 학습 방법들은 주로 노드와 시간 간의 관계에 집중했으나 역사적 사건의 영향을 간과하고 있었다는 점을 개선합니다.

- **Technical Details**: EVP는 두 가지 주요 메커니즘을 탑재하고 있습니다. 첫째, 이벤트 적응 메커니즘을 통해 각 이벤트의 미세한 특성을 다운스트림 작업에 맞게 조정하는 방법을 제안합니다. 둘째, 이벤트 집계 메커니즘을 통해 역사적 사건의 지식을 효과적으로 통합하여 노드 표현에 반영합니다.

- **Performance Highlights**: 네 개의 공개 데이터셋에서 EVP의 성능을 평가한 결과, 기존의 최신 방법들과 비교하여 우수한 성능을 나타냈습니다. 이 논문은 동적 그래프 학습에서 역사적 사건의 지식을 통합하는 혁신적인 접근법을 보여주며, 다양한 다운스트림 작업에 적합한 솔루션으로 자리매김할 수 있습니다.



### DiffStyleTS: Diffusion Model for Style Transfer in Time Series (https://arxiv.org/abs/2510.11335)
- **What's New**: 이번 연구에서는 DiffTSST라는 새로운 확산 기반(diffusion-based) 프레임워크를 소개합니다. 이 프레임워크는 시간 시계열 데이터를 내용(content)과 스타일(style) 표현으로 분리하고, 이를 자가 지도(self-supervised) 기반의 주의(attention) 확산 과정으로 재조합하여 스타일 전송(style transfer)을 수행합니다. DiffTSST는 두 개의 개별 시계열에서 내용과 스타일을 추출하여 생성된 샘플을 조건부로 제어할 수 있는 기능을 제공합니다.

- **Technical Details**: DiffTSST는 세 가지 구성 요소로 이루어져 있습니다: (1) 시간 시계열에 노이즈를 점진적으로 추가하는 전방 확산 과정, (2) 보완적인 내용과 스타일 표현을 추출하는 컨볼루션 인코더, (3) 이 표현을 융합하여 새로운 시퀀스를 복원하거나 합성하는 조건부 디노이징 네트워크입니다. 모델 훈련 시 각 시계열을 자체 내용-스타일 쌍으로 활용하여 시간 구조의 기본 생성 메커니즘을 학습합니다. 추론 시에는 서로 다른 시계열에서 내용과 스타일을 끌어와 제어 가능하고 다양한 스타일 전송을 구현합니다.

- **Performance Highlights**: DiffTSST는 여러 도메인에서 실제적이고 다양한 생성 결과를 달성함을 보여줍니다. 이 연구는 데이터가 부족한 환경에서 이상 탐지(anomaly detection)와 같은 다운스트림 작업의 성능을 크게 향상시킬 수 있다는 것을 입증했습니다. 또한 이 연구는 시간 시계열 스타일 전송 문제를 정립하고, 기준을 제시하여 향후 연구에 기여합니다.



### LouisKV: Efficient KV Cache Retrieval for Long Input-Output Sequences (https://arxiv.org/abs/2510.11292)
- **What's New**: 본 연구에서는 KV 캐시( Key-Value Cache )의 메모리 사용을 최적화하는 새로운 프레임워크인 LouisKV를 제안합니다. LouisKV는 중요한 KVs( Key-Values )의 강한 시간적 지역성과 고유한 분포 패턴을 활용하여, 인퍼런스 과정에서 필수적인 정보만을 효율적으로 검색하도록 설계되었습니다. 이를 통해 장기 시나리오에서의 효율성과 정확성을 동시에 향상시키는 데 중점을 두고 있습니다.

- **Technical Details**: LouisKV는 의미 기반의 검색 전략과 분리된 관리 체계를 도입하여, 디코딩 과정에서의 검색 오버헤드를 크게 줄입니다. 특히, LouisKV는 세그먼트 경계를 기준으로 검색을 수행하며, 이를 통해 더 정확하게 중요 KVs를 식별하고 전송합니다. 또한, 맞춤형 Triton과 CUDA 커널을 포함하여 KV 클러스터링 및 검색 속도를 최적화하는 여러 커널 레벨 최적화를 적용하고 있습니다.

- **Performance Highlights**: 테스트 결과, LouisKV는 기존의 최첨단 KV 검색 방법들에 비해 최대 4.7배의 속도를 달성하면서, 다양한 장기 시퀀스 작업에서도 거의 손실 없는 정확도를 유지하는 것으로 나타났습니다. 이 연구는 다양한 LLM( Large Language Models ) 벤치마크에서 LouisKV의 성능을 검증하였으며, 장기 입력과 장기 출력을 모두 아우르는 성능 향상을 보여주고 있습니다.



### Gym-TORAX: Open-source software for integrating RL with plasma control simulators (https://arxiv.org/abs/2510.11283)
- **What's New**: 이번 논문에서는 Gym-TORAX라는 파이썬 패키지를 소개하며, 이는 토카막에서 플라즈마 역학과 제어를 시뮬레이션하는 Reinforcement Learning (RL) 환경을 구현하는 데 사용됩니다. 사용자는 제어 행동과 관찰, 그리고 제어 목표를 간단히 정의하면 Gym-TORAX가 이들을 감싸는 Gymnasium 환경을 생성하여 플라즈마 역학을 시뮬레이션합니다. 이 패키지는 RL 알고리즘과의 호환성을 제공하여 플라즈마 제어 연구를 촉진하는 것을 목표로 하고 있습니다.

- **Technical Details**: Gym-TORAX 패키지는 TORAX 시뮬레이터를 기반으로 하며, 플라즈마 상태의 진화를 시뮬레이션합니다. TORAX는 플라즈마 온도, 밀도 및 자기 플럭스와 같은 다양한 변수를 모사하는 열린 루프(open-loop) 시뮬레이터이며, 상태 변수, 시간 시리즈 및 유도 변수를 업데이트하여 플라즈마 상태를 계산합니다. 이 패키지는 MDP (Markov Decision Process)로 문제를 모델링하면서, 토카막의 동적 특성을 반영하여 제어 환경을 구현합니다.

- **Performance Highlights**: 현재 Gym-TORAX 패키지에서는 국제 열핵 실험로 (ITER)의 램프 업 시나리오 기반의 환경이 준비되어 있습니다. 이 패키지는 RL 알고리즘에 의해 직접 활용될 수 있도록 다양한 운전 시나리오를 나타낼 수 있는 기능을 제공합니다. 플라즈마 제어와 관련된 연구에 있어, Gym-TORAX는 효과적인 도구로 자리 잡을 것으로 기대됩니다.



### Vision-LLMs for Spatiotemporal Traffic Forecasting (https://arxiv.org/abs/2510.11282)
- **What's New**: 본 논문에서는 복잡한 공간 종속성을 효과적으로 처리하기 위해 ST-Vision-LLM이라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 시공간 예측 문제를 비전-언어 융합 문제로 재구성하여, глобал 트래픽 매트릭스를 이미지 시퀀스로 처리할 수 있게 합니다. 또한, 효율적인 부동 소수점 값 인코딩 방식과 메모리 효율적인 강화 학습 방법을 도입하여 예측 정확도를 향상시켰습니다.

- **Technical Details**: ST-Vision-LLM은 비전-LLM 비주얼 인코더를 활용하여 역사적 글로벌 교통 매트릭스를 이미지 시퀀스로 변환합니다. 이를 통해 모델은 셀 레벨 예측을 위한 종합적인 글로벌 뷰를 가질 수 있게 됩니다. 또한, 부동 소수점 값을 단일 토큰으로 나타내는 효율적인 인코딩 스킴을 도입하고, 두 단계의 숫자 정렬 파인튜닝 프로세스를 통해 모델의 예측 성능을 더욱 개선합니다.

- **Performance Highlights**: ST-Vision-LLM은 실제 모바일 교통 데이터셋에서 기존 방법보다 15.6% 더 향상된 장기 예측 정확도를 보여주었으며, 크로스 도메인 몇 샷 시나리오에서는 두 번째 최선의 기준을 30.04% 이상 초과했습니다. 이 결과들은 다양한 데이터 부족 환경에서 모델의 높은 일반화 능력을 입증하고 있습니다.



### ENIGMA: The Geometry of Reasoning and Alignment in Large-Language Models (https://arxiv.org/abs/2510.11278)
Comments:
          52 pages, 10 figures

- **What's New**: 이번 논문에서는 ENIGMA(Entropy Mutual-Information Geometry Large-Language Model Alignment)라는 새로운 접근 방식을 통해 LLM 훈련의 추론, 정렬, 강인성을 향상시키는 방법을 제시합니다. 조직의 정책과 원칙을 모델의 정보 메니폴드에서의 운동 방향으로 간주하여 이를 훈련 신호와 측정 방법에 직접 적용하는 방식을 제안합니다. ENIGMA는 여러 기법을 통합해 설계된 단일 루프 훈련기법을 사용해, 외부 보상 모델 없이도 원칙이 인코딩된 추론 체인을 끌어내는 데 초점을 맞춥니다.

- **Technical Details**: ENIGMA는 Group-Relative Policy Optimisation (GRPO), Self-Supervised Alignment with Mutual Information (SAMI) 및 Sinkhorn divergence를 활용하는 새로운 훈련 방법을 도입합니다. 이 방법은 정보 기하학적 목표에 대한 효과적인 측정을 위한 수량적 지표를 개발하고, 원칙의 선택과 훈련 동역학에 미치는 영향을 정량화하기 위한 Sufficiency Index (SI)를 포함합니다. 본 연구에서는 또한 성능 향상을 예측하는 여러 메트릭을 제안하여 훈련 동역학을 포괄적으로 분석합니다.

- **Performance Highlights**: ENIGMA를 통해 훈련된 모델들은 정렬과 추론 벤치마크에서 향상된 성능을 보였으며, 특히 GPQA에서는 +6.92포인트, TruthfulQA에서는 +12.11포인트의 성과 향상이 나타났습니다. 실험 결과는 원칙에 의해 구조적으로 변화된 모델을 확인할 수 있었고, 이러한 증거들은 추론, 정렬 및 강인성이 단일 정보 기하학적 목표의 투영임을 지지합니다. ENIGMA 접근 방법은 조직이 정의한 원칙과 기준을 사용하여 LLM의 행동과 출력 간의 관계를 정량적으로 설명할 수 있는 가능성을 제공합니다.



### FedLoRA-Optimizer: Federated LoRA Fine-Tuning with Global and Local Optimization in Heterogeneous Data Scenarios (https://arxiv.org/abs/2510.11274)
- **What's New**: 이 논문은 분산 데이터와 계산 자원을 활용한 연합( federated ) 효율적 미세 조정(fine-tuning) 방법을 제안합니다. Low-Rank Adaptation (LoRA) 기법을 통해 대규모 사전학습 모델을 효율적으로 미세 조정할 수 있으며, 고객 데이터가 이질적일 때 발생하는 문제를 해결합니다. 이 방법은 조정 행렬의 세밀한 분석을 통해 클라이언트 간 공유된 특성과 개인화된 지식을 효과적으로 학습하려고 합니다.

- **Technical Details**: 연구에서는 민감한 방향 벡터와 크기 벡터를 구별하여, 큰 편향 없는 모델을 구축하는 새로운 연합 LoRA 최적화기(FedLoRA-Optimizer)를 제안합니다. 이 방법은 글로벌 및 로컬 최적 기법을 결합하여, 전역 모델의 일반화 능력과 로컬 모델의 개인화 적합성을 동시에 개선합니다. 후속 실험에서는 LLaMA2-7B 및 Deepseek-7B 모델을 사용하여 각기 다른 작업에서 유의미한 성과를 달성했습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 전통적인 LoRA 방식에 비해 전역과 개인화된 작업에서 각각 약 0.39%와 0.59%의 성능 향상을 보여 주었습니다. 이러한 결과는 제안된 세밀한 조정 기법이 클라이언트 간의 협업 optimizaiton을 성공적으로 수행했음을 입증합니다. 결과적으로, 글로벌 모델의 일반화 능력과 로컬 모델의 적응 능력이 크게 향상되었습니다.



### MIEO: encoding clinical data to enhance cardiovascular event prediction (https://arxiv.org/abs/2510.11257)
Comments:
          Presented in the Poster Session of Computational Intelligence methods for Bioinformatics and Biostatistics (CIBB) 2025

- **What's New**: 최근 임상 데이터의 가용성이 증가함에 따라 기계 학습(ML) 방법이 이들 데이터에서 지식을 추출하고 임상 사건을 예측하는 데 활용되고 있습니다. 하지만 레이블이 붙은 데이터의 부족과 데이터 이질성으로 인해 결측값이 발생하는 두 가지 주요 문제가 제기되고 있습니다. 본 연구는 이러한 문제를 해결하기 위해 자기 지도(auto-supervised) 방식의 오토인코더(self-supervised auto-encoder)인 MIEO(Masked Input Encoded Output) 모델을 제안합니다.

- **Technical Details**: MIEO 모델은 임상 데이터의 구조를 고려하여 설계되었으며, 레이블이 없는 데이터를 효과적으로 활용합니다. 이 모델은 결측값 처리를 명시적으로 다루며, 이중 이진(Binary) 및 연속(Continuous) 데이터에 대해 손실 함수(loss function)를 조정하여 성능을 최적화합니다. 여기서 레이블이 있는 환자 데이터셋을 사용하여 심혈관 사망 예측을 수행하며, 해당 데이터셋은 8065명의 환자로 구성됩니다.

- **Performance Highlights**: MIEO 모델을 통해 생성된 임베딩(embedding)은 심혈관 사망 예측에 있어 기존의 원시 데이터에 직접 적용한 것보다 향상된 균형 정확도를 달성하였습니다. 실험 결과, MIEO+ANN 조합 모델이 CVD 이벤트를 인식하는 능력이 더 뛰어난 것으로 나타났습니다. 또한, 성능 지수의 변동에 따라서 MIEO 모델이 결측값이 있는 데이터에서도 의미 있는 특성을 추출하는 데 효과적임을 보였습니다.



### FUSE: Fast Semi-Supervised Node Embedding Learning via Structural and Label-Aware Optimization (https://arxiv.org/abs/2510.11250)
- **What's New**: 본 연구에서는 노드 임베딩이 비어있는 경우를 위한 신속한 반지도(semisupervised) 임베딩 생성 프레임워크를 제안합니다. 이 방법은 세 가지 보완적인 최적화 요소를 통합하여 고품질 노드 임베딩을 효율적으로 생성합니다. 추가적으로, 기존의 머신러닝 기술과 비교하여 덜 복잡한 계산 비용으로 높은 정확도를 달성할 수 있습니다.

- **Technical Details**: 위 연구에서는 비지도적 구조 보존을 위한 크기가 조정 가능한 모듈러리티(modularity) 근사, 레이블이 있는 노드 간의 분산 최소화를 위한 지도적 정규화(supervised regularization), 그리고 랜덤 워크 기반의 레이블 전파를 통한 반지도적 노드 정교화(semi-supervised propagation) 등을 포함합니다. 이러한 구성 요소를 단일 반복 최적화 프레임워크로 통합하여 빠르고 효율적으로 고품질의 노드 임베딩을 생성합니다.

- **Performance Highlights**: 평가된 표준 벤치마크에서 본 방법은 노드 분류 정확도를 기존의 최첨단 방법들과 비교하여 동등하거나 더 뛰어난 성능을 보여주었으며, 계산 비용을 크게 절감하면서도 효과적인 결과를 도출합니다. 특히, 급속하게 변화하는 실제 상황에서도 효과적으로 노드 임베딩을 업데이트할 수 있는 가능성을 지니고 있습니다.



### Learning the Structure of Connection Graphs (https://arxiv.org/abs/2510.11245)
- **What's New**: 이번 연구는 Connection Graphs (CGs)에 대한 새로운 프레임워크를 제안합니다. CGs는 전통적인 그래프 모델을 넘어 네트워크의 토폴로지와 기하학적 일관성을 연결하여 신호의 전역적 특성을 표현하는데 중요한 역할을 합니다. 연구에서는 관측된 신호로부터 CG를 학습하는 역문제를 다루며, 최대 가짜 우도(maximum pseudo-likelihood) 방법을 기반으로 하는 일관성 가정을 도입합니다.

- **Technical Details**: 연구에서 제안된 Structured Connection Graph Learning (SCGL) 알고리즘은 리만 다양체(Riemannian manifold)에서 블록 최적화(block-optimization) 절차를 통해 네트워크의 토폴로지, 엣지 가중치(edge weights), 기하학적 구조를 동시에 추론하는 방법입니다. 이 알고리즘은 그래프의 스펙트럼 특성을 유지하면서 의미 있는 네트워크 구조를 유도하는 것을 목표로 합니다. SCGL은 기존의 그래프 학습 방법보다 더 나아가 CG의 고유한 비유클리드 기하학을 고려하여 설계되었습니다.

- **Performance Highlights**: 실험 결과, SCGL은 기존 방법들과 비교하여 토폴로지 복구(topological recovery)와 기하학적 충실도(geometric fidelity) 모두에서 지속적으로 우수한 성능을 보였습니다. SCGL은 랜덤 및 기하학적 그래프에 대한 합성 실험에서도 두드러진 개선 효과를 보여주었습니다. 이러한 결과는 SCGL이 기하 기반의 네트워크 토폴로지를 추론하는데 효과적인 도구임을 입증합니다.



### Neural Weight Compression for Language Models (https://arxiv.org/abs/2510.11234)
- **What's New**: 이번 연구에서는 언어 모델 가중치의 효율적인 저장 및 전송을 위한 학습 기반 압축 프레임워크, Neural Weight Compression (NWC)을 제안합니다. 기존의 수공예(g hand-crafted) 접근 방식과 달리, NWC는 사전 훈련된 언어 모델 가중치에서 직접 신경 코덱(neural codecs)을 학습시키며, 다양한 기술적 구성 요소를 통합하여 가중치 압축의 품질을 높이고 있습니다.

- **Technical Details**: NWC는 열(column) 단위의 텐서 분할(tensor chunking) 및 정규화(normalization), 중요도 인식 훈련 손실(importance-aware training loss), 추론 시 오류 보상 메커니즘(inference-time error compensation) 등의 세 가지 주요 기술 요소를 포함하고 있습니다. 이를 통해 다양한 크기와 형태의 가중치 텐서를 효과적으로 처리하고, 다운스트림 모델의 성능에 기반한 손실을 활용하여 압축 훈련을 수행할 수 있게 됩니다.

- **Performance Highlights**: 실험 결과, NWC는 4-6 비트 정밀도(precision)에서 경쟁력 있는 정확도-압축 무역을 달성하였으며, FP16 모델과 유사한 정확도를 유지하였습니다. 이러한 성과는 언어 모델에서뿐만 아니라 비전 인코더(vision encoder)와 같은 다른 영역에서도 확장 가능성을 보여줍니다, 이는 NWC가 다양한 유형의 모델에 대한 압축 성능을 향상할 수 있음을 시사합니다.



### Enforcing convex constraints in Graph Neural Networks (https://arxiv.org/abs/2510.11227)
- **What's New**: 이 논문에서는 ProjNet이라는 새로운 Graph Neural Network (GNN) 프레임워크를 소개합니다. ProjNet은 입력에 의존적인 복잡한 제약 조건을 만족시키기 위해 설계되었습니다. 본 연구는 sparse vector clipping 기법과 Component-Averaged Dykstra (CAD) 알고리즘을 결합하며, 이를 통해 효율적인 대규모 입력 처리가 가능한 GPU 가속 구현을 개발합니다.

- **Technical Details**: ProjNet은 두 가지 핵심 구성 요소인 projection layer와 sparse vector clipping layers를 사용하여 제약 조건을 만족시킵니다. 이 프레임워크는 다양한 최적화 문제에 대한 해결책을 제시하기 위해 convex constraint (볼록 제약)을 다룹니다. 또한 CAD 알고리즘의 수렴 결과를 증명하고 이를 효율적인 GPU 구현으로 통합하여 성능을 극대화합니다.

- **Performance Highlights**: ProjNet은 선형 프로그래밍, 비볼록 이차 프로그래밍 두 가지 클래스, 그리고 무선 송신 전력 최적화와 같은 네 가지 제약 최적화 문제에 대해 그 효과를 입증했습니다. 특히, 다양한 문제 환경에서의 효율성을 입증하여 실용성과 유연성을 강화했습니다. 이 연구는 GNN이 제약 조건 충족 문제를 효과적으로 처리할 수 있는 가능성을 보여줍니다.



### Cross-Scale Reservoir Computing for large spatio-temporal forecasting and modeling (https://arxiv.org/abs/2510.11209)
- **What's New**: 본 논문에서는 고해상도 시공간 데이터셋을 예측하기 위한 새로운 reservoir computing 방법을 제안합니다. 이 구조는 거친 레이어에서 미세 레이어까지의 다중 해상도 입력을 결합하여 지역적(local) 및 글로벌(global) 역학을 효과적으로 포착할 수 있습니다. 해수면 온도 데이터에 적용했을 때, 표준 평행 reservoir 모델보다 장기 예측에서 우수한 성능을 보여주며, 레이어 간의 결합(coupling)이 예측 정확도를 개선하는 데 효과적임을 입증하였습니다.

- **Technical Details**: 이 연구에서 사용된 reservoir computing의 기본 구조는 고차원 비선형 동적 시스템인 reservoir ℛ(ℜ)으로 구성됩니다. 입력 신호는 낮은 차원으로 입력되어 reservoir의 내부 상태 𝐫(t)가 ODE(Ordinary Differential Equation)에 의해 동적으로 진화합니다. 이 과정에서 비선형 이득 함수(Nonlinear Gain Function)와 고정된 행렬(Projection Matrix)을 통해 입력을 reservoir 상태 공간으로 투영하게 됩니다.

- **Performance Highlights**: 모델 성능을 테스트하기 위해 Copernicus 위성에서 파생된 글로벌 해수면 온도 데이터를 사용하여 교차 스케일(cross-scale) reservoir computing 모델을 훈련하고 평가했습니다. 이 모델은 기존 방법들에 비해 예측의 정확성을 크게 향상시켰으며, 각 레이어가 서로 다른 해상도에서 작동함에 따라 높은 정밀도의 시간 예측을 가능하게 했습니다. 결국, 이 새로운 계층적 구조는 다중 해상도의 시스템 동적 모델링에서 중요한 이점을 제공합니다.



### Evaluating Line-level Localization Ability of Learning-based Code Vulnerability Detection Models (https://arxiv.org/abs/2510.11202)
Comments:
          Preprint

- **What's New**: 이 논문은 소프트웨어 취약성 문제를 해결하기 위해 기계 학습(ML) 알고리즘의 사용에 대한 연구를 진행하고 있습니다. 기존의 모델들이 전체 소스 코드 기능을 취약하다고 표시하기만 했던 문제를 개선하기 위해, 더 세밀한 예측과 코드 라인 위치를 정밀하게 식별하는 방법에 대한 기여를 목표로 하고 있습니다. 특히 비정상적인 상관관계와 편향이 ML 알고리즘의 성능에 미치는 영향을 분석함으로써, 이들 모델의 신뢰성을 높이고자 합니다.

- **Technical Details**: Detection Alignment (D​ADA)라는 새로운 방법론이 제안되었으며, 이는 입력 소스 코드 라인과 실제 취약성 위치 간의 부합 정도를 측정합니다. D​ADA는 설명 가능성 기법을 활용하여, ML 모델이 예측한 취약성의 출처가 되는 코드 라인을 정량화합니다. 모델은 다양한 데이터셋을 사용하는 데에 제한이 없으며, 기존의 탐지기들과 비교하여 더 나은 성능을 목표로 합니다.

- **Performance Highlights**: 실험을 통해 D​ADA 메트릭이 예측시 비취약성 코드 라인의 영향을 어떻게 구체적으로 보여주는지에 대한 결과가 도출되었습니다. 이로 인해 모델이 학습한 내용이 신뢰할 수 있는 범위를 벗어났음을 알리고, 기존의 탐지 방식에 대한 문제점을 드러냈습니다. 향후 연구에서 D​ADA 방법론이 학습 기반 탐지기의 성과를 개선하고 인간 운영자에게 더 나은 결정을 지원할 수 있는 기초가 될 것을 기대합니다.



### Protein as a Second Language for LLMs (https://arxiv.org/abs/2510.11188)
Comments:
          Main paper: 9 pages, 6 figures. With references and appendix: 18 pages, 9 figures total. Submitted to ICLR 2026 (under review)

- **What's New**: 이번 논문에서는 단백질 서열을 마치 언어처럼 해석할 수 있는 "Protein-as-Second-Language" 프레임워크를 도입했습니다. 이 접근법은 아미노산 서열을 새로운 심볼릭 언어로 재구성하여, 큰 언어 모델이 컨텍스트 예시(contextual exemplars)를 통해 해석할 수 있도록 합니다. 특히, 이 방법은 추가적인 훈련 없이 제로샷(zero-shot) 설정에서 기능적 단서(funcional cues)를 드러내는 학습 문맥을 생성합니다.

- **Technical Details**: 우리는 79,926개의 단백질-질문-답변(triple) 쌍으로 구성된 이중 언어 데이터세트를 작성했습니다. 이를 통해 다양한 오픈 소스 LLMs와 GPT-4에서 일관된 성능 향상을 보여주었으며, 최대로는 17.2% ROUGE-L 향상을 기록했습니다. 이러한 결과는 일반적인 LLM이 단백질에 대한 언어적 단서를 통해 도메인 특화 모델보다 뛰어난 성능을 발휘할 수 있는 가능성을 보여줍니다.

- **Performance Highlights**: Protein-as-Second-Language 프레임워크는 아미노산 서열을 배움으로써 단백질 이해를 지원하는 효율적인 도구가 될 수 있습니다. 우리의 접근법은 추가적인 훈련이나 공학적 조정을 요구하지 않고도 기능을 이해할 수 있도록 돕습니다. 이러한 성과는 대규모 데이터 요구와 높은 계산 비용 등 기존 접근법들의 한계를 극복하는 데 기여할 것입니다.



### Can Tool-Integrated Reinforcement Learning Generalize Across Diverse Domains? (https://arxiv.org/abs/2510.11184)
- **What's New**: 최근 대형 언어 모델(LLMs)의 발전은 추론(reasoning) 및 도구(tool) 활용에서 놀라운 능력을 보여주고 있습니다. 그러나 다양한 분야에서 도구 보강 강화 학습(tool-augmented reinforcement learning, RL)의 일반화는 아직 충분히 탐구되지 않았습니다. 본 연구에서는 수학 문제 해결에서 훈련된 LLM 에이전트가 코드 인터프리터(tool)의 도움을 받아 다양한 추론 분야에서 어떻게 성능을 발휘하는지를 조사합니다.

- **Technical Details**: 연구에서는 수학 영역에서의 RL을 통해 도구 호출 전략을 학습한 후 여러 독립적인 분야에서 평가하는 방식입니다. 이를 통해 도구 사용의 일반화 가능성을 검토하며, TGRL(Tool Generalization Reinforcement Learning) 프레임워크를 제안합니다. TGRL은 표준화된 도구 인터페이스, 이중 보상 시스템, XML 기반 프롬프트 템플릿을 활용하여 도메인에 구애받지 않는 학습과 기술 이전(skill migration)을 촉진하는 구조입니다.

- **Performance Highlights**: 광범위한 벤치마크를 통한 실험은 제안된 접근 방식이 최첨단 성능을 달성했음을 보여줍니다. 수학 문제에서 배운 도구 사용이 복잡한 다른 분야의 작업에 효과적으로 이전될 수 있음을 입증했습니다. 또한 정량적 작업 수행 및 높은 토큰 효율성을 바탕으로 하여 Tool RL의 잠재력을 강조합니다.



### EAGER: Entropy-Aware GEneRation for Adaptive Inference-Time Scaling (https://arxiv.org/abs/2510.11170)
- **What's New**: EAGer는 토큰 수준의 엔트로피 분포를 활용하여 중복 계산을 줄이고 전반적인 성능을 개선하는 새로운 접근 방식을 제안합니다. 이 방법은 높은 엔트로피 토큰에서만 여러 가지 추론 경로로 분기할 수 있게 하여, 같은 지침(prompt)에 대해 유사한 계산 비용을 할당하는 문제를 해결합니다. EAGer는 기존의 전통적인 방법들보다 더 효율적이며, 특히 복잡한 문제에서 예외적인 성능 향상을 보여줍니다.

- **Technical Details**: EAGer는 추론 과정에서 모델의 불확실성을 모니터링하여 계산 리소스의 효율적 할당을 가능하게 합니다. 특히, 높은 엔트로피 값을 가진 토큰에서만 새로운 병렬 추론 경로를 시작하여, 예측이 안정적일 때는 후보 시퀀스를 적게 생성하고, 불확실성이 클 때는 추가 탐색을 요구하는 방식입니다. 이를 통해 더 많은 계산 리소스를 복잡한 문제에 집중할 수 있습니다.

- **Performance Highlights**: EAGer는 AIME 2025와 같은 복잡한 추론 벤치마크에서 최대 37%의 성능 향상을 달성하며, 생성되는 토큰 수를 최대 65% 줄입니다. 다양한 오픈 소스 모델에 대해 실험한 결과, EAGer를 사용한 경우 연산 비용을 80%까지 절감하면서도 성능이 개선되는 것을 확인했습니다. 이로 인해 EAGer는 추론 과정에서의 효율성 및 성능 간의 최적의 균형을 제공합니다.



### ELMO: Efficiency via Low-precision and Peak Memory Optimization in Large Output Spaces (https://arxiv.org/abs/2510.11168)
Comments:
          Accepted to ICML 2025

- **What's New**: 이번 논문에서는 Extreme Multilabel Classification (XMC)에 대한 새로운 저정밀도 훈련 프레임워크인 ELMO를 제안합니다. ELMO는 BFloat16과 Float8 데이터 타입을 사용하여 순수 저정밀도 훈련을 통해 큰 출력 공간에서 효과적인 모델 훈련을 가능하게 합니다. 저정밀도 훈련을 통해 GPU 메모리 사용량을 획기적으로 줄일 수 있으며, 3백만 개 레이블의 모델을 6.6 GiB의 메모리로 훈련할 수 있습니다.

- **Technical Details**: ELMO는 Kahan summation과 stochastic rounding 기법을 활용하여 Float8 데이터 타입만으로 모델을 훈련할 수 있는 가능성을 보여줍니다. 이러한 접근 방식은 딥러닝에서의 메모리와 계산 요구 사항을 줄이기 위해 개발된 것으로, GF16에서 BF16으로의 전환과 그라디언트 통합 전략을 통해 이루어집니다. 우리의 방법은 모델 훈련의 메모리 요구량을 50-75%까지 줄일 수 있도록 돕습니다.

- **Performance Highlights**: 여러 개의 레이블 크기에 대해 ELMO의 저정밀도 훈련 방법을 평가한 결과, 기존의 SOTA 방법과 비슷한 성능을 나타냅니다. 또한, LF-Paper2Keywords-8.6M이라는 8.6백만 레이블을 가진 새로운 데이터셋을 소개하여, 현재 공개된 XMC 벤치마크 중 가장 큰 데이터셋임을 주장합니다. 저정밀도 훈련은 XMC 분야에서 더욱 더 중요한 기준으로 자리잡을 가능성이 높습니다.



### Beyond single-model XAI: aggregating multi-model explanations for enhanced trustworthiness (https://arxiv.org/abs/2510.11164)
Comments:
          Accepted at the European Workshop on Trustworthy Artificial Intelligence (TRUST-AI), co-located within ECAI 2025

- **What's New**: 이 논문은 인공지능(AI) 모델의 신뢰성(trustworthiness)과 윤리적 사용에 대한 논의를 심화시키고 있으며, 설명 가능한 인공지능(eXplainable AI, XAI)의 필요성을 강조합니다. 다수의 모델에서 파생된 기능 중요성 집계를 사용하여 설명의 견고성(robustness)에 대한 역할을 조사하고 있습니다. 특히, robust한 설명 방법만이 시스템에 대한 신뢰를 구축할 수 있는 요소라는 점을 강조하고 있습니다.

- **Technical Details**: 저자들은 k-최근접 이웃(k-nearest neighbours, k-NN), 랜덤 포레스트(random forest) 및 신경망(neural networks, NNs) 모델의 기능 중요성을 설명하기 위해 새로운 접근 방식을 제안합니다. 이를 통해 k-NN의 거리 기반 예측 과정에서 기능들이 어떻게 영향을 미치는지를 파악할 수 있도록 하고, 랜덤 포레스트에서는 각 노드의 불순도(node impurity)를 기반으로 기능 중요도를 도출하는 방법을 개발하였습니다. 또한 여러 모델을 동시에 고려하여 설명을 집계하는 방식으로 신뢰성을 높일 수 있는 가능성을 탐색합니다.

- **Performance Highlights**: 초기 결과는 다양한 모델의 예측력을 활용함으로써 애플리케이션의 신뢰성을 증가시킬 수 있는 잠재력을 보여주고 있습니다. 특히, 제안된 기능 중요도 접근 방식은 특히 레거시 데이터셋 및 이진 분류 작업에서 유의미한 설명을 제공하며 사용자의 이해도를 높이는 데 기여할 수 있습니다. 앞으로 더 많은 데이터를 통해 실용적인 고위험 애플리케이션에 대한 신뢰성을 강화할 기반이 될 것입니다.



### Emergence of hybrid computational dynamics through reinforcement learning (https://arxiv.org/abs/2510.11162)
Comments:
          22 pages, 11 figures

- **What's New**: 이 연구는 강화 학습( Reinforcement Learning, RL)와 감독 학습(Supervised Learning, SL)이 동일한 의사결정 과제에 대해 반복 신경망(Recurrent Neural Networks, RNN)을 훈련할 때 서로 다른 계산 솔루션으로 이끌어낸다는 점을 보여줍니다. RL은 하이브리드 어트랙터 아키텍처(hybrid attractor architectures)를 자발적으로 발견하여, 의사결정을 위한 안정적인 고정점 어트랙터(stable fixed-point attractors)와 유연한 증거 통합을 위한 준주기적 어트랙터(quasi-periodic attractors)를 결합합니다. 반면, SL은 단순한 고정점 해결책으로 거의 전적으로 수렴합니다.

- **Technical Details**: 연구에서는 기본적인 작업을 수행할 수 있는 균형 잡힌 신경 집단을 조각내기 위해 RL이 강력한 형태의 암묵적 정규화(implicit regularization)를 사용한다고 밝혔습니다. 이 정규화는 복잡한 역학을 가진 RL이 네트워크의 초기 가중치 초기화 전반에서 성능 향상과 밀접하게 상관 관계가 있음을 보여줍니다. 이를 통해 RL이 보상 기반 최적화를 통해 보다 정교한 역학 메커니즘을 자율적으로 발견할 수 있음을 입증합니다.

- **Performance Highlights**: 연구 결과는 강화 학습이 SL에 비해 더 복잡하고 강력한 작업 성능을 가지고 있음을 보여줍니다. RL 훈련된 네트워크에서 준주기적 역학의 출현이 성능 향상과 강하게 연관되어 있으며, 초기 가중치 분포에 의해 조절 가능하다는 것을 발견하였습니다. 이 복잡한 역학은 AI 시스템을 설계할 때 활용할 수 있는 실질적 기준을 제공합니다.



### A Comprehensive Forecasting-Based Framework for Time Series Anomaly Detection: Benchmarking on the Numenta Anomaly Benchmark (NAB) (https://arxiv.org/abs/2510.11141)
- **What's New**: 이번 논문에서는 기존의 시간 시계열 이상 탐지 기법의 한계를 극복하고자 예측 기반의 포괄적인 프레임워크를 제안합니다. 고전적 방법(Holt-Winters, SARIMA)과 딥러닝 아키텍처(LSTM, Informer)를 통합하여 공통의 잔여 기반 탐지 인터페이스를 통해 평가합니다. 데이터 전처리(normalization, STL decomposition), 네 가지 예측 모델과 탐지 방법, 이중 평가 메트릭스를 통합하여 시스템적인 평가를 실현했습니다.

- **Technical Details**: 우리는 58개의 데이터셋이 포함된 Numenta Anomaly Benchmark(NAB)를 통해 232개의 모델 학습과 464개의 탐지 평가를 수행하여 100%의 성공률을 달성하였습니다. LSTM 모델이 가장 우수한 성능(F1: 0.688)을 보였으며, Informer는 30% 더 빠른 훈련 속도를 제공하면서도 경쟁력 있는 정확도를 기록했습니다. 또한 고전적 방법들은 간단한 합성 데이터에 대해서는 완벽한 예측을 보여주었으나 실제 데이터셋에서는 낮은 F1 스코어를 기록했습니다.

- **Performance Highlights**: 실험 결과, 예측 품질이 탐지 성능에 미치는 영향이 지배적임을 확인했습니다. 탐지 방법 간의 F1 점수 차이(0.621-0.688)는 예측 모델 간의 차이(0.344-0.688)보다 작았습니다. 본 연구는 복잡한 패턴에는 LSTM을, 효율성이 중요한 경우에는 Informer를, 단순한 주기적 데이터에는 고전적 방법을 사용할 것을 제안하며, 향후 연구를 위한 기준을 마련하였습니다.



### DUAL: Learning Diverse Kernels for Aggregated Two-sample and Independence Testing (https://arxiv.org/abs/2510.11140)
- **What's New**: 본 연구에서는 다중 커널(multilple kernels)을 사용한 두 샘플(test) 및 독립성 검정의 효과를 높이기 위해, 커널 다양성을(complex structures) 기반으로 한 통계치를 제안합니다. 기존 방법은 서로 유사한 커널들을 선택하여 정보의 중복성(overlapping information)을 초래했으나, 우리 연구는 다양한 커널을 선택하여 이를 개선합니다.

- **Technical Details**: 논문에서는 서로 다른 커널들 사이의 공분산(covariance)을 고려한 통계치를 통해 커널의 다양성을 명확히 반영합니다. 이러한 통계치는 테스트 파워(test power) 향상과 커널 간의 다양성(diversity) 간의 균형을 맞추는 데 도움을 줍니다. 우리는 엄격한 이론적 공식과 증명(previews)을 통해 제안한 모델의 일관성(consistency)과 Type-I 오류 제어를 보여줍니다.

- **Performance Highlights**: 광범위한 실험을 통해, 제안한 방법이 여러 벤치마크에서 기존 방법보다 뛰어난 성능을 발휘함을 입증했습니다. 두 샘플 및 독립성 검정 모두에서 우리의 접근 방식이 효과적임을 확인했습니다.



### Test-Time Adaptation by Causal Trimming (https://arxiv.org/abs/2510.11133)
Comments:
          Accepted to the Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025); Code is available at this https URL

- **What's New**: 본 논문에서는 Test-time Adaptation by Causal Trimming (TACT)라는 새로운 방법을 제안하여 모델의 강건성을 개선합니다. TACT는 비인과적(Non-causal) 특성을 식별하고 제거하여, 테스트 데이터에서 모델의 의존도를 감소시켜 성능 저하를 방지합니다. 이 방법은 데이터 변형(Data Augmentation)을 통해 인과적(Causal) 특성을 유지하면서 비인과적 특성을 변화시킵니다.

- **Technical Details**: TACT는 입력 데이터에 대해 타겟 변형을 적용한 후, 주성분 분석(Principal Component Analysis, PCA)를 사용하여 비인과적 특성과 관련된 변화 방향을 식별합니다. 이 방향으로부터 얻어진 프로젝션을 제거하여 테스트 샘플의 표현을 조정하고, 업데이트된 프로토타입을 유지하여 노이즈 효과를 완화합니다. 이를 통해 TACT는 적응 과정 내내 비인과적 특성을 지속적으로 추적하고 개선할 수 있습니다.

- **Performance Highlights**: TAKT는 실제 환경의 분포 이동에 대한 벤치마크에서 다른 최신 TTA 방법들에 비해 일관되게 우수한 성능을 나타냅니다. 이 방법은 모델의 예측 정확도를 향상시키며 비인과적 특성에 의해 영향을 덜 받는 신뢰할 수 있는 예측 결과를 제공합니다. 이론적 분석을 바탕으로 TACT의 효과가 입증되었습니다.



### Lightweight Facial Landmark Detection in Thermal Images via Multi-Level Cross-Modal Knowledge Transfer (https://arxiv.org/abs/2510.11128)
- **What's New**: 이 논문에서 제안하는 Multi-Level Cross-Modal Knowledge Distillation (MLCM-KD) 프레임워크는 RGB에서 열화상으로의 지식 이전을 효과적으로 분리하여 열상 얼굴 랜드마크 탐지 모델을 보다 정확하고 효율적으로 만듭니다. 특히, Dual-Injected Knowledge Distillation (DIKD) 방법론을 통해 RGB와 열화상 간의 심층적인 의미적 일치를 이끌어냅니다. 이 접근법은 단순한 일방향 지식 주입의 한계를 극복하고, 양방향 메커니즘으로 교수-학생 모델 간의 관계를 개선합니다.

- **Technical Details**: MLCM-KD는 두 개의 계층 수준으로 구성되어 있습니다. 첫 번째인 Knowledge Transfer Level (KTL)에서는 RGB 교사 네트워크로부터 열학생 네트워크로 Landmark-specific 지식을 전달하며, 두 번째인 Model Compression Level (MCL)에서는 열화상 입력을 이용한 모델 압축을 수행합니다. DIKD 메커니즘은 각 계층에서 구조적 특성과 세부적 특징을 묘사하면서, 학생 네트워크가 RGB 교사의 구조적 특징을 모방하도록 강제합니다.

- **Performance Highlights**: 실험 결과, 이 방법은 여러 공공 데이터셋에서 새로운 최첨단 성능을 달성하였으며, 기존의 전통적인 일방향 지식 증류 및 이미지 변환 방법들에 비해 정확성과 효율성에서 모두 크게 개선되었습니다. MLCM-KD를 통해 생성된 경량 모델은 극도의 저조도 환경 및 자원 제약이 있는 시나리오에서도 높은 성능을 유지하고 있어, 실제 응용에서 강력한 가치를 보여줍니다.



### Refining Hybrid Genetic Search for CVRP via Reinforcement Learning-Finetuned LLM (https://arxiv.org/abs/2510.11121)
- **What's New**: 이번 연구에서는 작은 특화된 대형 언어 모델(LLM)을 세밀하게 파인튜닝(fine-tuning)하여 고급 솔버보다 우수한 성능을 가진 크로스오버 연산자를 생성할 수 있음을 보입니다. 이는 VRP(차량 경로 문제)에 대한 해결책을 자동으로 생성하기 위한 기존의 접근법에 도전장을 내민 것입니다. 새로운 강화 학습(Reinforcement Learning, RL) 프레임워크인 RFTHGS를 도입하여, 소규모 LLM이 하이브리드 유전자 탐색(Hybrid Genetic Search, HGS)의 효과적인 연산자를 독립적으로 생성하도록 돕습니다.

- **Technical Details**: RFTHGS는 14B 파라미터를 가진 reasoning LLM을 파인튜닝하여 HGS 알고리즘의 크로스오버 연산자를 생성합니다. 이 프레임워크는 구조화된 보상 체계를 활용하여 학습 과정을 세 단계로 안내합니다. 첫 번째 단계에서는 구문적으로 유효한 코드를 생성할 경우 보상을 부여하고, 두 번째로 시간 초과 없이 실행 가능한 코드를 생성할 경우 추가 보상을 제공합니다. 마지막으로, 전문가 설계 연산자에 비해 최적화된 해결책을 제시한 경우에 성능 향상에 비례한 보상을 부여합니다.

- **Performance Highlights**: 실험 결과, 파인튜닝된 LLM은 전문가가 설계한 연산자보다 HGS에서 상당한 성능 향상을 보여주었습니다. 이 연구는 1,000개 노드까지 대규모 문제에서도 우수한 성능을 유지하며, 기존 신경-조합적 기법 및 프롬프트 기반 방법들보다 월등히 우수한 성능을 입증합니다. 또한, 이는 현재 상용 LLM인 GPT-4 및 GPT-4o-mini를 초월하는 성과로, 사전 연구와는 달리 소규모 LLM이 효과적으로 사용될 수 있음을 보여줍니다.



### PhysioME: A Robust Multimodal Self-Supervised Framework for Physiological Signals with Missing Modalities (https://arxiv.org/abs/2510.11110)
Comments:
          9 pages, 2 figures

- **What's New**: 본 논문에서는 PhysioME라는 새로운 프레임워크를 제안합니다. PhysioME는 결측된 모달리티(missing modality) 조건에서도 신뢰할 수 있는 성능을 보장하도록 설계되었습니다. 이 프레임워크는 다중 모달(multi-modal) 자기 지도 학습(self-supervised learning) 접근 방식을 채택하고, 시계열 동작을 포착하기 위해 Dual-Path NeuroNet를 사용합니다. 또한 결측된 모달리티 토큰을 복원하는 복원 디코더(restoration decoder)를 추가하여 불완전한 입력을 유연하게 처리할 수 있게 합니다.

- **Technical Details**: PhysioME는 heterogenous한 생리 신호를 위한 다중 모달 자기 지도 학습(SSL) 프레임워크입니다. 특히, DP-NeuroNet 구조는 두 개의 동일한 인스턴스가 weights를 공유하는 방식으로 설계되어 있습니다. 이는 masked prediction과 contrastive learning을 결합하여 라벨이 없는 데이터로부터 일반화 가능한 표현을 효과적으로 학습할 수 있도록 돕습니다. 구체적으로, 각 생리 신호는 전처리된 인코더를 통해 특징을 추출하며, 각 모달리티에 대한 특정 복원 디코더를 포함합니다.

- **Performance Highlights**: PhysioME는 다양한 결측 모달리티 시나리오에서도 높은 일관성과 일반화 성능을 보입니다. 이를 통해 수면 단계 분류(sleep stage classification)와 저혈압 예측(hypotension prediction) 같은 임상 데이터셋에서 강력한 성능을 입증했습니다. 이러한 성과는 PhysioME가 현실 세계의 불완전한 데이터 환경에서 임상적 의사결정을 지원하는 신뢰할 수 있는 도구로 자리매김할 가능성을 보여줍니다.



### Causal Disentanglement Learning for Accurate Anomaly Detection in Multivariate Time Series (https://arxiv.org/abs/2510.11084)
Comments:
          20 pages, 4 Figures,

- **What's New**: 이번 논문에서는 Causally Disentangled Representation Learning for Anomaly Detection (CDRL4AD)라는 새로운 방식을 제안합니다. 이 방법은 여러 시계열 데이터에서 이상 감지와 그 원인 관계를 파악하는 데 초점을 맞추고 있습니다. 특히, 기존의 방법들이 다양한 시기에서의 인과 관계를 명확히 추론하는 데 한계를 보였던 점을 해결하고자 합니다.

- **Technical Details**: CDRL4AD는 인과 프로세스를 모델 입력으로 사용하고, 시간적 이질성 그래프와 인과 관계를 설계합니다. 이를 통해 우리는 서로 다른 시간대의 인과 관계를 식별하고, 잠재 변수를 분리하여 해당 인과 요인을 추론할 수 있습니다. 강력한 그래프 구조를 통해 CDRL4AD는 MTS의 이질성과 시간적 동적 관계를 고려한 포괄적인 인과 표현 프레임워크를 제공합니다.

- **Performance Highlights**: 실험 결과, CDRL4AD는 실제 데이터셋에서 기존의 최첨단 방법들보다 정확성과 뿌리 원인 분석에 있어 월등한 성능을 보였습니다. 또한, 모델의 하이퍼파라미터 민감성과 시간 복잡도를 분석하여 검증하였고, 인간 전문가가 이상 원인을 진단하는 데 있어 어떻게 기여할 수 있는지를 보여주는 사례 연구도 진행했습니다.



### Efficient Edge Test-Time Adaptation via Latent Feature Coordinate Correction (https://arxiv.org/abs/2510.11068)
Comments:
          Under review

- **What's New**: 이 논문에서는 자원 제약이 있는 엣지 디바이스를 위한 새로운 단일 사례 테스트 시간 적응(Test-Time Adaptation, TTA) 방법인 TED를 제안합니다. TED는 공분산 행렬 적응 진화 전략(Covariance Matrix Adaptation Evolution Strategy, CMA-ES)을 사용하여 저차원 벡터를 업데이트하면서 출력 신뢰성을 향상시키고 잠재 표현을 소스 잠재 분포에 가깝게 조정합니다. 이 과정에서 모델 매개변수를 고정하고 역전파 없이 수행하여 메모리와 계산 비용을 최소화합니다.

- **Technical Details**: TED는 잠재 주 서브스페이스에서만 전방 최적화(Forward-Only Optimization)를 수행하여 단일 테스트 샘플에 적합하도록 설계되었습니다. 강화된 갈무리 활용을 통해 예측의 신뢰성을 높이고 학습 과정을 효과적으로 조정합니다. 기존의 TTA 방법과 달리 TED는 전체 모델을 조정하는 대신 단일 축소된 벡터만을 업데이트하여 자원-제약 장비에서의 높은 효율성을 제공합니다.

- **Performance Highlights**: 실험은 이미지 분류 및 키워드 탐지 작업에서 수행되었으며, TED는 기존 기준 대비 최대 63배의 계산 복잡성을 감소시키는 성과를 거두었습니다. 또한 다양한 데이터셋에서 실험을 통해 자원 제약이 있는 환경에서도 안정적인 성능을 보이며, TED가 실제 엣지 디바이스에서의 실용성과 효과를 입증했습니다.



### Stronger Together: On-Policy Reinforcement Learning for Collaborative LLMs (https://arxiv.org/abs/2510.11062)
- **What's New**: 이번 연구에서는 멀티 에이전트 시스템(Multi-Agent Systems, MAS)과 강화 학습(Reinforcement Learning, RL)을 결합하여 대규모 언어 모델(Large Language Models, LLM)의 성능을 향상시키는 방법을 제안합니다. AT-GRPO라는 새로운 RL 알고리즘이 MAS에 맞게 설계되었고, 단일 정책 및 다중 정책 모두를 지원하는 훈련 시스템이 개발되었습니다. 이를 통해 태스크별 협업을 통해 강력한 정책을 학습할 수 있습니다.

- **Technical Details**: AT-GRPO는 에이전트 및 턴 기반의 그룹화된 RL 알고리즘을 포함하며, MAS의 고유한 도전 과제를 해결하기 위해 개발되었습니다. 이 시스템은 MAS 워크플로우의 다양성을 지원하며, 온-정책 업데이트(on-policy updates)를 통해 여러 정책을 동시에 관리할 수 있습니다. 기존의 단일 모델 기반 RL 프레임워크에서는 겪기 어려운 여러 모델의 동시 운영과 상호 작용을 조율할 수 있습니다.

- **Performance Highlights**: AT-GRPO는 다양한 게임, 계획, 코딩 및 수학 태스크에 걸쳐 우수한 성능을 보여주었습니다. 예를 들어, 장기 계획 태스크에서 단일 에이전트 RL 기준 정확도를 14.0%에서 96.0%로 향상시켰으며, 코딩과 수학 작업에서 각각 평균 3.87%에서 7.62%, 9.0%에서 17.93%까지 개선되었습니다. 이러한 결과는 MAS에서 온-정책 RL 훈련의 효과성을 입증합니다.



### Robust Photoplethysmography Signal Denoising via Mamba Networks (https://arxiv.org/abs/2510.11058)
Comments:
          5 pages, 2 figures

- **What's New**: 본 논문에서는 착용식 건강 모니터링의 신뢰성을 향상시키기 위해 새로운 딥러닝 기반 PPG(Photoplethysmography) 제거 기법인 DPNet을 소개합니다. 이 프레임워크는 생리학적 정보를 보존하는 데 중점을 두며, Mamba 아키텍처를 기반으로 하여 효과적인 시간이 모델링을 구현하고 있습니다. 또한 보조 HR 예측기(HR predictor, HRP)를 도입하여 심박수 기반의 감독을 통해 생리학적 일관성을 강화합니다.

- **Technical Details**: DPNet은 세 개의 컨볼루션 레이어를 통해 노이즈가 포함된 PPG 신호에서 지역적 특징을 추출한 후, bidirectional Mamba(BMamba) 블록을 통해 장기적인 시간 의존성을 포착합니다. 이후 특성 차원을 단일 채널로 줄이고, 최종적으로 입력과 네트워크 출력을 가중합하는 방식으로 정제된 신호를 생성합니다. SI-SDR(loss) 손실 딜레그램을 추가하여 파형의 충실도를 유지하며 신뢰성을 증가시키는 데 기여합니다.

- **Performance Highlights**: BIDMC 데이터셋을 기반으로 한 실험에서는 제안된 방법이 기존의 필터링 기법 및 다른 신경망 모델에 비해 향상된 성능을 보였으며, 노이즈와 실제 이동 아티팩트에 대해 강력한 견고성을 보여주었습니다. 또한, HR 정확성을 유지하면서 PPG 신호를 효과적으로 복원할 수 있어 착용식 헬스케어 시스템에서의 실용적인 배치를 위한 가능성을 입증했습니다.



### Temporal Alignment Guidance: On-Manifold Sampling in Diffusion Models (https://arxiv.org/abs/2510.11057)
Comments:
          54 pages, 17 figures, 18 tables

- **What's New**: 이번 논문에서는 diffusion models(확산 모델)에서 발생하는 off-manifold(오프 매니폴드) 현상을 해결하기 위한 새로운 접근 방식을 제안합니다. 제안된 방법은 time predictor(시간 예측기)를 활용하여 각 시간 단계에서 원하는 데이터 매니폴드로부터의 편차를 추정합니다.

- **Technical Details**: 시간 간격이 증가할수록 생성 품질이 감소하는 것을 발견한 후, 논문에서는 'Temporal Alignment Guidance'(TAG)라는 새로운 안내 메커니즘을 설계하여 생성 과정에서 매 시간 단계마다 샘플을 원하는 매니폴드로 되돌립니다. 이러한 방식은 샘플의 일관성을 보장하며, 실제 데이터에 대한 적합성을 유지하는 데 크게 기여합니다.

- **Performance Highlights**: 다양한 실험을 통해 TAG는 각 시간 단계에서 생성된 샘플이 원하는 매니폴드와 밀접하게 정렬되도록 하여 생성 품질에서 현저한 개선을 이루었음을 보여줍니다. 이로 인해 여러 하부 작업(downstream tasks)에서 성능이 향상되었습니다.



### Conformal Inference for Time Series over Graphs (https://arxiv.org/abs/2510.11049)
- **What's New**: 본 연구에서는 그래프 시계열(graph time series)에 적합한 새로운 순차적 예측 영역 프레임워크를 개발하였다. 기존의 conformal prediction (CP) 방법론은 단일 시계열이나 정적 그래프에 각각 적용되었으나, 본 연구에서는 그래프의 구조를 활용하여 노드 간의 쌍방 의존성을 포착하고, 사용자 지정 적합성을 보장하는 혁신적인 방법을 제시한다. 특히, 예측 결과에 대한 커버리지 보장을 제공하며, 실세계 데이터셋을 사용하여 기존 방법보다 최대 80%까지 좁은 예측 지역을 만들어내는 성과를 달성하였다.

- **Technical Details**: 그래프 구조를 활용한 새로운 비형식 성능 척도(nonconforming scores)를 계산하는 방법론을 제안한다. 기본 모델로부터 얻은 잔차(residuals)를 그래프 컨볼루션 연산자를 사용해 필터링한 후, 필터링된 잔차의 일부를 포함하는 타원체 형태의 예측 지역을 구성하고, quantile regressor를 통해 알려지지 않은 데이터의 분위를 예측한다. 이때 인접한 노드가 유사한 잔차를 갖는다는 동질성(homophily)을 가정하여, 이러한 그래프 필터링이 타원체 예측 지역의 부피에서 기하급수적으로 줄어든 결과를 생성한다.

- **Performance Highlights**: 제안한 불확실성 정량화 프레임워크는 소망하는 경험적 커버리지를 유지하면서, 그래프 비고려 CP 방법론보다 훨씬 작은 타원체를 생성함을 실증적으로 보여주었다. 연구 결과, 이례적 그래프 시계열 예측 문제에서 불확실성을 효과적으로 처리할 수 있는 잠재력을 지닌다는 점에서 혁신적이라고 할 수 있다. 본 연구는 그래프 구조를 동시에 고려한 예측 접근 방식으로서 그래프 타임 시리즈 예측 분야의 새로운 패러다임을 제시하고 있다.



### The Easy Path to Robustness: Coreset Selection using Sample Hardness (https://arxiv.org/abs/2510.11018)
- **What's New**: 이 논문에서는 데이터 중심(data-centric) 관점에서 적대적(Adversarial) 강인성을 높이는 방법으로 EasyCore라는 새로운 코어셋(selection) 선택 알고리즘을 제안합니다. EasyCore는 낮은 평균 입력 기울기 노름(Average Input Gradient Norm, AIGN)을 가진 샘플만을 선택하여 학습에 사용하며, 여기서 AIGN은 샘플의 적대적 취약성을 추정하는 지표로 활용됩니다. 실험 결과 EasyCore로 선택한 데이터를 기반으로 학습된 모델이 상대적으로 높은 적대적 정확도(adversarial accuracy)를 달성함을 보였습니다.

- **Technical Details**: EasyCore는 훈련 과정에서 주어진 샘플의 AIGN을 기반으로 내구성(resilience)이 높은 샘플을 선택하는 방법론입니다. AIGN은 샘플의 기울기(norm)의 평균을 구하여 샘플의 학습 용이성을 정량화하며, 이는 적대적 공격에 대한 취약성과 밀접한 연관이 있습니다. 또한, EasyCore는 기존의 코어셋 방법들과 비교했을 때 효율적이고 모델에 구애받지 않으며, 특정 데이터셋에 대해 한번만 AIGN을 계산하면 되기 때문에 계산 비용을 낮추는 장점이 있습니다.

- **Performance Highlights**: EasyCore를 통해 표준 훈련과 TRADES 적대적 훈련 각각에서 최대 7% 및 5%의 적대적 정확도 향상을 이룰 수 있음을 보였습니다. 기존의 코어셋 선택 방법에 비해 더 높은 성능으로, 특히 불확실한 샘플을 선택하는 기존 방법들의 한계를 극복했습니다. 이 연구는 데이터 중심의 접근 방식이 어떻게 모델의 적대적 강인성을 개선할 수 있는지를 실증적으로 보여줍니다.



### Instruction-aware User Embedding via Synergistic Language and Representation Modeling (https://arxiv.org/abs/2510.11016)
- **What's New**: 이번 연구에서는 InstructUE라는 새로운 사용자 표현 임베딩 모델을 제안합니다. 이 모델은 대규모 언어 모델(LLM)을 활용하여 일반적이고 지침 기반의 사용자 표현을 생성합니다. InstructUE는 경량 어댑터가 탑재된 다중 인코더 아키텍처를 도입하여 여러 출처의 이질적인 데이터를 효과적으로 처리하고 구조적 특성을 유지합니다.

- **Technical Details**: InstructUE는 대규모 UserQA 데이터셋을 사용하여 언어 및 표현 공간을 연결하는 새로운 대비-자기회귀 훈련 프레임워크를 제안합니다. 이 방법은 언어 공간의 도메인 지식을 활용하고, 표현 공간에서 사용자-텍스트 임베딩을 정렬하는 대조 학습을 통해 효과적인 임베딩 품질을 향상시킵니다. 또한, 임베딩을 다양한 개인화 작업에 효율적으로 적용하기 위한 소수 샷 클러스터 기반의 감독 지침 조정 기법을 채택합니다.

- **Performance Highlights**: InstructUE는 사용자 예측, 마케팅, 추천 시나리오를 포함한 다양한 도메인에서 기존 방법보다 성능이 뛰어난 것으로 입증되었습니다. Extensive experiments showed that the instruction-aware modeling enables effective denoising and enhances generalizability in user representation learning. 이는 지침 기반의 사용자 모델링이 다양한 실제 애플리케이션에서 유용하게 활용될 수 있음을 시사합니다.



### Catch-Only-One: Non-Transferable Examples for Model-Specific Authorization (https://arxiv.org/abs/2510.10982)
- **What's New**: 이번 연구에서는 비자율적 사용을 방지하면서 인가된 모델의 유용성을 유지하기 위해, '비전이 불가능한 예제(non-transferable examples, NEs)'라는 새로운 개념을 제안합니다. NEs는 훈련이나 데이터에 의존하지 않으며, 모델 특화된 저감도로의 재코딩을 통해 인가된 모델의 예측은 최적화하되 비인가 모델에서는 성능을 저하시킵니다. 이는 데이터가 다양한 AI 모델에 소비될 수 있는 현실을 반영하여, 보안과 혁신 간의 균형을 이룰 수 있는 새로운 방법론을 제시합니다.

- **Technical Details**: NEs는 뉴럴 네트워크의 구조적인 성질을 활용하여 초기 특징에 미치는 입력 방향을 최소화하여, 특정 모델에서는 유용성을 유지하고 다른 모델에서는 성능 손실을 유도하는 메커니즘입니다. 연구진은 인가된 모델의 무감도(subspace) 내에서 재코딩한 입력이 유권성의 허용한 경계를 유지하도록 형식적인 이론적 기초를 확립합니다. 이를 위해 행렬 섭동 이론과 Hoffman-Wielandt 불평등을 활용하여 비인가 모델에서의 성능 저하와 스펙트럼 차이가 연관되어 있음을 증명합니다.

- **Performance Highlights**: NEs는 여러 비주얼 백본(vision backbones)과 최첨단 비전-언어 모델(vision-language models)에서 일반적인 전처리 상황에서도 성능을 유지하며, 인가된 모델에서는 유효하나 비인가 모델에서는 효과적으로 유용성을 차단하는 특징을 보였습니다. 이미지 분류의 경우, NEs는 인가된 ResNet-50 모델이 약간의 변화로 80%의 정확도를 유지하는 반면, 다른 모델들은 성능이 무용해지는 결과를 나타냈습니다. 또한 NEs는 다양한 모델 아키텍처와 데이터 형식에서 전반적으로 유용성을 차단하는 데 성공하며, 실제적인 배치 가능성을 지니고 있음을 입증하였습니다.



### On the Optimal Representation Efficiency of Barlow Twins: An Information-Geometric Interpretation (https://arxiv.org/abs/2510.10980)
Comments:
          7 pages

- **What's New**: 본 논문에서는 Self-supervised learning (SSL)의 효율성을 정량화하기 위한 새로운 정보 기하학적 프레임워크를 제안합니다. 기존의 SSL 알고리즘은 레이블이 없는 데이터에서 의미 있는 표현을 학습하는 데 성공적이었으나, 서로 다른 SSL 패러다임의 효율성을 비교하고 이해하기 위한 통합된 이론적 틀이 부족했습니다. 이 연구는 Fisher Information Matrix (FIM)의 스펙트럼 성질을 기반으로 효과적인 내재 차원을 정의하고, 새로운 관점에서 SSL을 분석합니다.

- **Technical Details**: 우리는 표현 공간의 통계 매니폴드와 평균 Fisher Information Matrix (FIM)의 스펙트럼에 기반하여 표현 효율성을 정의합니다. Barlow Twins 방법에 대한 이론적 분석을 통해, 특정 가정 하에 Barlow Twins가 최적의 표현 효율성(η=1)을 달성함을 증명합니다. 이는 cross-correlation matrix를 단위 행렬에 가깝게 만드는 목표가 어떻게 효과적인 표현을 유도하는지를 설명합니다.

- **Performance Highlights**: Barlow Twins 방법은 두 왜곡된 데이터 뷰 간의 cross-correlation matrix를 최대한 단위 행렬에 가깝게 유지함으로써 좋은 표현을 학습합니다. 우리는 이 방법이 평균 Fisher Information Matrix의 스펙트럼 성질에 기반하여 효율성을 극대화한다는 것을 이론적으로 입증하였습니다. 본 연구는 Barlow Twins의 효과성을 이해하는 엄격한 이론적 기초를 제공하고 SSL 알고리즘을 분석하기 위한 새로운 기하학적 관점을 제시합니다.



### Blade: A Derivative-free Bayesian Inversion Method using Diffusion Priors (https://arxiv.org/abs/2510.10968)
- **What's New**: 이 논문에서는 Blade라는 새로운 디리바티브-프리 베이지안 인버전 알고리즘을 소개합니다. Blade는 상호작용하는 입자들을 사용하여 높은 정확도를 가지며 잘 보정된 사후 분포(posteriors)를 생성할 수 있습니다. 이 알고리즘은 고차원 문제에서 또한 비선형 포워드 모델을 다룰 수 있도록 강력한 데이터 기반 프라이어를 활용합니다.

- **Technical Details**: Blade는 확산 모델(diffusion models)을 활용하여 베이지안 인버전을 수행합니다. 이 방법은 제어된 환경에서 직접적인 분포 체크를 통해 성능을 검증하고, 비선형 유체 역학 문제를 포함한 여러 복잡한 역 문제에 대한 사후 품질을 평가합니다. 또한, Blade는 점수 근사(score approximation) 및 통계적 선형화 오류(statistical linearization error)의 영향을 정량화하는 경향 분석(convergence analysis)을 제공합니다.

- **Performance Highlights**: Blade는 기존의 디리바티브-프리 베이지안 인버전 방법들과 비교했을 때 우수한 성능을 보입니다. 여러 테스트에서 Blade는 신뢰할 수 있는 불확실성 정량화(uncertainty quantification)를 수행하며, 고차원 및 복잡한 프라이어 분포 문제를 처리하는 데 있어 흔들림 없는 정확성을 입증하고 있습니다.



### Not All Bits Are Equal: Scale-Dependent Memory Optimization Strategies for Reasoning Models (https://arxiv.org/abs/2510.10964)
Comments:
          20 pages, 12 figures

- **What's New**: 본 연구에서는 4비트 양자화(quantization)가 비이성 모델(non-reasoning models)과 제로샷(zero-shot) 작업에서는 메모리 최적의 선택으로 입증되었지만, 이성 모델(reasoning models)에서는 KV 캐시(Key-Value cache)가 메모리의 주요 요인이 될 수 있음을 보여줍니다. 1,700개 이상의 시나리오에 대한 체계적인 실험을 통해, 모델의 효과적인 크기가 8비트 4B 매개변수(paramenters) 이하일 경우 더욱 정확도가 향상된다는 것을 발견했습니다. 이러한 결과는 메모리 최적화가 스케일(scale)과 관계없이 적용될 수 없음을 보여주며, 이성 모델에 대한 최적화 전략이 비이성 모델의 전략과 근본적으로 다르다는 것을 시사합니다.

- **Technical Details**: 대규모 언어 모델(LLMs)의 메모리 성능 트레이드오프에 대한 선행 연구는 주로 모델 가중치(weight)를 압축하는 데 집중되었습니다. 하지만 현대의 이성 모델은 더 많은 토큰(tokens)을 생성하여 KV 캐시가 메모리의 주요 제약이 됩니다. 본 연구에서는 Qwen3 모델 패밀리(0.6B에서 32B까지)를 대상으로 AIME 및 GPQA-Diamond 두 가지 벤치마크에서 실증 연구를 진행했습니다. 이 과정에서 4비트 및 8비트 GPTQ 가중치 양자화, 추론 토큰 예산, 병렬 스케일(parallel scaling) 및 KV 캐시 압축 기법에 대한 다양한 시나리오를 탐구하였습니다.

- **Performance Highlights**: 모델의 효과적인 크기가 8비트 4B보다 작을 경우, 더 많은 가중치에 메모리를 할당하는 것이 장기적인 생성보다 더 메모리 효율적입니다. 또한 지식 중심의 이성 작업에서는 4비트 가중치가 메모리 최적입니다. 반면 수학적 이성 작업에서는 8비트 또는 16비트 가중치가 더 메모리 효율적일 수 있습니다. KV 캐시 압축이 메모리 효율적인 이성을 만들어내는 주요 요인으로 작용하며, KV 캐시 삭제가 8비트 4B보다 작은 모델에서는 KV 캐시 양자화보다 더 나은 메모리-정확도 트레이드오프를 제공합니다.



### APLOT: Robust Reward Modeling via Adaptive Preference Learning with Optimal Transpor (https://arxiv.org/abs/2510.10963)
Comments:
          EMNLP2025

- **What's New**: 본 논문은 Bradley-Terry (BT) 기반의 보상 모델(RM)을 개선하기 위한 적응형 마진 메커니즘을 소개합니다. 이 메커니즘은 모델이 어려운 샘플에 더 많은 초점을 맞추도록 하여 유사한 선호 응답을 더 잘 구별하도록 도와줍니다. 결과적으로, 이 접근 방식은 인-디스트리뷰션(ID) 및 아웃-오프-디스트리뷰션(OOD) 환경 모두에서 성능과 일반화 능력을 현저히 향상시킵니다.

- **Technical Details**: 적응형 마진은 분포 인식 관점에서 형태를 잡아 Optimal Transport (OT)를 사용하여 모델이 선택된 응답과 거부된 응답 간의 분포적 차이를 더 잘 캡처할 수 있도록 설계되었습니다. 이 방식은 각 훈련 사례의 어려움을 동적으로 조절하여 학습 과정을 최적화합니다. 이렇게 함으로써, RM은 보다 효과적으로 긍정적 및 부정적 사례를 구별할 수 있습니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 기존 보상 모델 기술들보다 성능에서 우수함을 보여주었습니다. 이는 높은 분리도와 빠른 수렴 속도를 달성하면서도 추가적인 훈련 소모를 크게 증가시키지 않습니다. 결국, 우리의 방법은 LLM이 인류 선호에 더 잘 맞춰지도록 하는 데 효과적임을 입증합니다.



### MC#: Mixture Compressor for Mixture-of-Experts Large Models (https://arxiv.org/abs/2510.10962)
Comments:
          15 pages, 13 figures

- **What's New**: 이번 연구에서는 Mixture-of-Experts (MoE) 모델의 효율성을 극대화하기 위한 MC#라는 새로운 프레임워크를 제안합니다. MC#는 정적 양자화(quantization)와 동적 전문가 가지치기(pruning)를 통합하여 극단적인 압축을 달성하고자 하며, 이는 메모리 사용 및 계산 비용을 줄이는 데 기여합니다. 이 프레임워크는 Pre-Loading Mixed-Precision Quantization (PMQ)와 Online Top-any Pruning (OTP)이라는 두 가지 주요 단계를 포함하고 있습니다.

- **Technical Details**: MC#는 전문가의 중요성과 입력 토큰의 가중치를 기반으로 하여 MoE 모델의 크기를 효과적으로 줄이는 방안을 모색합니다. PMQ 단계에서는 각 전문가의 활성화 빈도와 손실을 고려하여 다양한 비트 너비를 할당하고, 선형 프로그래밍을 통해 최적의 양자화 구성을 찾습니다. OTP 단계에서는 Gumbel-Softmax 샘플링을 사용해 각 토큰에 대해 동적으로 전문가를 선택하여 활성화할 수 있습니다.

- **Performance Highlights**: MC#는 DeepSeek-VL2 모델에서 6.2배의 가중치 감소를 달성하며, 평균 2.57 비트의 압축에도 불구하고 정확도 감소는 1.7%에 불과합니다. 또한, OTP는 20% 이상의 전문가 활성화를 줄이고 1% 미만의 성능 저하로 효율적인 MoE 모델 배치를 가능하게 합니다. 이러한 성과는 MoE 기반 모델을 소비자 등급 및 엣지 레벨 응용 프로그램에 확장할 수 있는 가능성을 보여줍니다.



### Rediscovering Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning (https://arxiv.org/abs/2510.10959)
Comments:
          16 pages, 4 figures

- **What's New**: 이 연구에서는 대규모 언어 모델(LLMs)의 추론 능력을 향상시키기 위한 강화 학습 방법론인 RLVR(Reinforcement Learning with Verifiable Rewards)를 재검토하고, 정Entropy regularization의 잠재력이 과소평가되고 있다고 주장합니다. 특히 변동성이 큰 고정 계수를 사용하는 전통적인 접근법의 한계를 극복하기 위해 Adaptive Entropy Regularization(AER)이라는 새로운 프레임워크를 제안합니다.

- **Technical Details**: AER는 세 가지 주요 구성 요소를 통해 탐색(exploration)과 활용(exploitation)을 동적으로 조절합니다: 난이도 인식 계수 할당, 초기 기반 목표 엔트로피, 동적 전역 계수 조정 등이 포함됩니다. 이 방법은 각 작업의 난이도에 따라 엔트로피를 조절하며, 사전 설정된 수준 이하에서 유지하여 안정적인 학습을 목표로 합니다.

- **Performance Highlights**: 실험 결과, AER는 다양한 수학적 추론 벤치마크에서 기존 방법에 비해 일관된 성능 향상을 보여주었으며, 추론 정확도와 탐색 능력 모두 개선되었습니다. 이는 RLVR 훈련에서 적응적 엔트로피 정규화의 잠재력을 입증하는 결과입니다.



### Interpretable Machine Learning for Cognitive Aging: Handling Missing Data and Uncovering Social Determinan (https://arxiv.org/abs/2510.10952)
- **What's New**: 이번 연구는 알츠하이머병(AD)의 조기 발견의 중요성을 강조합니다. 연구자들은 사회적 결정 요인(SDOH)을 활용하여 인지 능력을 예측하고 이를 통해 예방 및 자원 분배의 공정성을 높일 수 있는 가능성을 제시합니다. 이를 위해 멕시코 건강 및 노화 연구(MHAS) 데이터셋을 기반으로 하여 인지 기능 점수를 평가했습니다.

- **Technical Details**: 연구는 인지 능력 점수를 위한 포괄적 지표를 2016 및 2021 MHAS 파형에서 도출하였습니다. 예측 인자는 인구통계학적, 경제적, 건강, 라이프스타일 및 심리사회적 요인을 포함합니다. 결측치를 처리하기 위해 특이값 분해(SVD) 기반의 보간(imputation) 파이프라인을 사용했으며, XGBoost 모델을 통해 뛰어난 예측 성능을 보였습니다.

- **Performance Highlights**: 연구 결과는 기존 방법들을 초과하여 높은 정확성 및 해석 가능성을 보여주었습니다. SHAP 기반의 사후 분석을 통해 주요한 SDOH 요인과 연령별 특징 패턴이 식별되었습니다. 특히, 바닥 재료가 강력한 예측 인자로 나타나 사회경제적 및 환경적 불평등을 반영하고, 여러 요인이 인지 노화에 미치는 영향을 강조했습니다.



### Redundancy as a Structural Information Principle for Learning and Generalization (https://arxiv.org/abs/2510.10938)
- **What's New**: 이번 연구는 전통적인 정보 이론을 확장하여 유한하고 구조화된 시스템에 적용할 수 있는 이론적 프레임워크를 제시합니다. redundancies (중복성)를 정보 조직의 근본적인 속성으로 재정의함으로써, 정보 이론의 여러 고전적 측정 방법을 통합하는 새로운 접근 방식을 제공합니다. 특히 이 연구는 중복성이 상한과 하한으로 제한되며, 이로 인해 구조 손실과 붕괴 사이의 최적 균형을 이룬다는 예측을 포함하고 있습니다.

- **Technical Details**: 이 프레임워크에서는 중복성을 정보 독립성에서의 ff-divergence (ff-발산)로 정의합니다. 이는 서로 다른 분야의 중복 개념을 통합하여, 정보 이론에서는 상호 정보(mutual information), 통계학에서는 공분산 중복(covariance redundancy) 등을 포함하는 다양한 측정 방법을 제공합니다. 이러한 통합된 기하학은 중복성이 얼마나 유용한지를 측정하는 양적 기준을 제공하며, 데이터가 독립성과 얼마나 떨어져 있는지를 나타냅니다.

- **Performance Highlights**: 실험에서는 masked autoencoders (MAE)를 활용하여 모델이 최적의 중복 수준에서 일반화 성능이 극대화됨을 보여주었습니다. 연구 결과는 중복성이 정보의 구조, 전달 및 이해 방식에 중요한 변수로 작용하며, 효율성을 추구하는 전통적인 접근법과는 대조적으로 중복성을 균형 있게 유지하는 것이 안정성 및 일반화에 긍정적인 영향을 미친다는 것을 시사합니다.



### Neutral Agent-based Adversarial Policy Learning against Deep Reinforcement Learning in Multi-party Open Systems (https://arxiv.org/abs/2510.10937)
- **What's New**: 이 논문에서는 다자간 오픈 시스템에서 효과적인 적대적 공격을 수행하기 위해, 기존의 방법과는 달리 피해자(agent)와의 직접 상호작용 없이 잘 훈련된 피해자를 오도하는 적대적 정책 학습 접근법을 새롭게 설계하였습니다. 제안된 방법은 환경을 완전히 통제할 필요가 없으며, 여러 작업 시나리오에서 중립 에이전트를 활용하여 피해자 에이전트에 간접적으로 영향을 미칠 수 있는 점이 특징입니다.

- **Technical Details**: 논문은 중립 에이전트(Neutral Agent)를 기반으로 한 적대적 정책 학습 접근법을 다룹니다. 이 접근법은 중립 에이전트가 피해자 에이전트를 직접적으로 상호작용하지 않고도 간접적으로 영향을 미칠 수 있도록 합니다. 이를 위해 적절한 보상 설계를 통해 정책 최적화를 수행하며, 효율적인 계산 방법을 개발하여 구현합니다.

- **Performance Highlights**: 실험 결과는 제안된 방법이 다자간 오픈 시스템에서 일반적이고 효과적인 적대적 공격을 수행할 수 있음을 보여줍니다. 실험 플랫폼으로는 Starcraft II 기반의 SMAC 플랫폼과 자율 주행 시뮬레이션 플랫폼인 Highway-env가 사용되었으며, 다양한 환경에서도 좋은 성능을 보여주었습니다.



### Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation (https://arxiv.org/abs/2510.10925)
Comments:
          19 pages, 10 figures

- **What's New**: 최근 연구들은 더 강력한 teacher models가 항상 최적의 teachers가 아니라는 것을 보여주었습니다. 이에 따라 PerSyn (Personalized data Synthesis)라는 새로운 데이터 합성 전략이 제안되었습니다. PerSyn은 ‘Route then Generate’라는 새로운 패러다임을 적용하여 각 student model에 맞춤형 데이터를 생성합니다.

- **Technical Details**: PerSyn은 각 prompt를 최적의 teacher model에 할당하는 과정에서 student의 학습 가능성과 teacher의 응답 품질을 모두 고려하는 쿼리 수준의 라우터를 사용합니다. 이 과정에서 각 teacher는 할당된 prompt에 대해서만 데이터를 합성하게 되어 전통적인 ‘Generate then Select’ 방식보다 효율적입니다.

- **Performance Highlights**: PerSyn은 다양한 모델 패밀리와 스케일에서 동작하며, instruct tuning 및 수학적 추론 설정에서 모든 기준 모델보다 우수한 성능을 보여주었습니다. 이를 통해 PerSyn의 효과성과 향후 연구의 방향성을 제시하는 중요한 통찰을 제공합니다.



### LPCVAE: A Conditional VAE with Long-Term Dependency and Probabilistic Time-Frequency Fusion for Time Series Anomaly Detection (https://arxiv.org/abs/2510.10915)
- **What's New**: 이 논문에서는 시계열 이상 탐지(Time Series Anomaly Detection)를 위한 새로운 모델인 LPCVAE를 제안합니다. LPCVAE는 LSTM(Long Short-Term Memory)을 활용하여 장기 의존성을 포착하고, Product-of-Experts(전문가의 곱) 메커니즘을 통해 시간 및 주파수 정보의 통합을 향상시킵니다. 기존의 VAE 기반 방법들이 단일 창(window) 특성에 제한되어 있었고, 시간과 주파수 정보를 효과적으로 활용하지 못했던 문제를 해결하는 데 중점을 둡니다.

- **Technical Details**: LPCVAE는 시계열 데이터를 구성하는 두 가지 핵심 구성 요소를 가지고 있습니다: Long-term Time Domain Branch(LTDB)와 Frequency Domain Branch(FDB)입니다. LTDB는 시간의 의존성을 모델링하고, FDB는 주파수 기반 특성을 추출하여 이상 탐지 성능을 향상시키는 역할을 합니다. 이러한 두 가지 접근을 통해 시간과 주파수 도메인의 상호작용을 효과적으로 모델링하며, 정보 손실을 최소화합니다.

- **Performance Highlights**: LPCVAE는 네 개의 공공 데이터셋에서 수행된 실험을 통해 최신 기술(state-of-the-art)보다 우수한 성능을 나타냈습니다. 특히 Yahoo 데이터셋에서 6.3%의 가장 큰 성능 향상을 기록했습니다. 이러한 결과는 장기 시간 및 주파수 표현의 통합이 TSAD에 대한 강력하고 효율적인 해결책이라는 것을 시사합니다.



### Quantifying Information Disclosure During Gradient Descent Using Gradient Uniqueness (https://arxiv.org/abs/2510.10902)
- **What's New**: 본 논문에서는 기계 학습 모델의 공개가 개인 정보 유출에 미치는 위험을 정량화하기 위한 새로운 접근법인 'gradient uniqueness'라는 프린시플 기법을 제안합니다. 이 지표는 학습된 모델을 게시함으로써 발생할 수 있는 정보 유출량의 상한선에 기초하고 있습니다. 'Gradient uniqueness'는 모델 아키텍처나 데이터셋 유형, 공격 전략에 대해 어떤 가정도 하지 않는 일반적인 수학적 도출 방법을 사용합니다.

- **Technical Details**: 본 논문은 미니 배치 확률적 경량 하강법(SGD)을 사용하는 동안 각 데이터 포인트의 유출 수준을 정량화할 수 있는 방법을 제시합니다. 이 연구에서 제안된 'gradient uniqueness (GNQ)'는 SGD의 내재적 개인정보 보호 특성을 반영하여 공격자에게 노출될 수 있는 데이터 포인트를 정량적으로 분석합니다. 우리는 실험을 통해 GNQ가 타 공격 방식과 유사한 개인정보 보호 수준을 함께 동반하면서도 높은 모델 유틸리티를 유지할 수 있는 능력을 검증했습니다.

- **Performance Highlights**: GNQ 기반의 방어 방법은 DP-SGD와 유사한 개인정보 보호 수준을 달성했으며, 멤버십 추론 공격(MIA)에 대한 거의 완벽한 보호를 제공했습니다. 실제 실험에서는 ResNet 및 BERT 계열 모델과 같은 CNN 및 Transformer 모델을 사용해 성능을 평가하였으며, 높은 테스트 정확도를 기록하는 동시에 모델의 유용성 또한 유지되었습니다. 나아가, GNQ를 사용한 방어 방식은 훈련 중 정보 유출을 모니터링하여 위험이 높은 데이터를 제거하며, 이는 몇 퍼센트의 입력 데이터 삭제로 해결할 수 있음을 보여주었습니다.



### HeroFilter: Adaptive Spectral Graph Filter for Varying Heterophilic Relations (https://arxiv.org/abs/2510.10864)
- **What's New**: 그래프 이질성(heterophily)에 대한 연구가 최근 활발히 진행되고 있습니다. 기존 연구들은 단순한 접근 방식을 취하여 동질적인 그래프는 저주파 필터(low-pass filter)를, 이질적인 그래프는 고주파 필터(high-pass filter)를 사용했습니다. 그러나 저자들은 이질성과 스펙트럼 필터(spectral filters) 간의 관계가 훨씬 더 복잡하다는 사실을 발견했습니다. 이 결과는 기존의 고정 필터 설계 방식을 도전하게 하며, 표현력을 보존하기 위한 적응형 필터링의 필요성을 제안합니다.

- **Technical Details**: 본 연구는 그래프 신호 처리(graph signal processing)에 기반한 스펙트럼 관점에서 GNN의 성능 제한을 이해하고자 합니다. 연구자들은 저주파 성분이 그래프에서 부드러운 변화를 포착하고, 고주파 성분이 지역적으로 급격한 변화를 포착하는 방식으로 그래프 신호를 주파수 성분으로 분해합니다. 기존 GNN 아키텍처는 주로 저주파 필터를 적용하여 정보를 증폭하는 방식이었으나, 이질적인 그래프의 경우 이러한 방식이 효과적이지 않음을 보였습니다. 저자들은 HeroFilter라는 새로운 GNN 아키텍처를 제안하여, 다양한 이질성 패턴을 효과적으로 처리할 수 있는 적응형 필터를 설계하였습니다.

- **Performance Highlights**: HeroFilter는 실험을 통해 동질성 및 이질성 그래프와 대규모 실제 데이터셋에서의 성능을 평가하였습니다. 이 GNN 모델은 기존 강력한 기준선보다 최대 9.2%의 정확성 개선을 달성하며 최신 알고리즘에서도 최고 성능을 기록했습니다. Fast-HeroFilter라는 확장 가능한 변종도 도입하여 효율적인 근사를 통해 고유값 분해(eigen decomposition)를 피할 수 있는 방법을 제시했습니다.



### A Joint Learning Approach to Hardware Caching and Prefetching (https://arxiv.org/abs/2510.10862)
Comments:
          Accepted at ML for Systems Workshop at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)

- **What's New**: 이번 논문에서는 현대 시스템의 스케줄링, 캐싱 및 기타 컴포넌트를 위한 학습 정책이 제공하는 새로운 접근법을 다룹니다. 특히, 캐시 교체 및 프리패칭 정책이 상호 의존적이라는 점을 강조하며, 이 두 가지 정책을 공동으로 훈련하는 방안을 제안합니다.

- **Technical Details**: 저자들은 캐시 대체(cache replacement)와 프리패칭(prefetching) 정책을 위한 공유 표현(shared representations)을 개발하는 방법을 제안합니다. 이를 위해 두 가지 접근 방식을 소개하는데, 하나는 공동 인코더(joint encoder)를 기반으로 하고 있으며, 다른 하나는 임베딩의 대조 학습(contrastive learning)을 기반으로 합니다.

- **Performance Highlights**: 이 두 가지 접근 방식 모두 유망한 초기 결과를 보여주며, 향후 연구 agenda를 설정합니다. 이러한 연구는 지속적으로 변화하는 하드웨어 환경에 적응하기 위한 노력을 더욱 발전시키는 데 기여할 수 있을 것으로 기대됩니다.



### Discrete State Diffusion Models: A Sample Complexity Perspectiv (https://arxiv.org/abs/2510.10854)
- **What's New**: 이번 논문에서는 이산 상태 확산 모델에 대한 이론적 연구를 수행하여 샘플 복잡성에 대한 첫 번째 경계를 제시합니다. 이 모델들은 텍스트, 시퀀스 및 조합 구조와 같은 응용 프로그램에서 중요하지만 이론적으로는 비교적 이해도가 낮습니다. 연구에서는 샘플 복잡성이 $	ilde{	ext{O}}(rac{1}{oldsymbol{	ext{ϵ}}^{2}})$라는 새로운 경계를 수립하고, 점수 추정 오류를 통계적, 근사, 최적화 및 클리핑 구성 요소로 분해한 구조적 분석을 제공합니다.

- **Technical Details**: 이산 상태 확산 모델은 데이터에서 샘플을 점진적으로 오염시켜 정적 분포를 얻는 전방확산 과정과, 학습된 분포를 재현하기 위해 점화 과정에서 사용하는 잘 정의된 노이즈 분포를 기반으로 샘플을 생성하는 후방 확산 과정으로 구성됩니다. 네거티브 엔트로피 함수의 강한 볼록성을 활용하여 점수 추정 오류의 Bregman 발산을 상한 및 하한으로 구별하고, 근사, 통계, 최적화 및 클리핑 오류로 세분화합니다. 이를 통해 제한된 데이터 샘플의 수와 급강하 방법을 통해 모형 학습에서의 제한 사항을 실용적인 맥락에서 제시합니다.

- **Performance Highlights**: 이번 연구는 이산 상태 확산 모델의 샘플 복잡성에 대한 첫 번째 정량적 분석을 제공하여, 고품질 샘플을 생성하기 위해 필요한 샘플 수를 명확히 합니다.  이론적 분석을 통해, 데이터 분포와 생성된 마르코프의 KL 발산이 특정 기준 미만으로 유지되기 위해 필요한 샘플 수를 산출할 수 있음을 들어, 샘플 효율성에 대한 깊은 통찰을 제공합니다. 수학적 근거를 바탕으로 하는 오류 분해 과정을 통해 각 요소가 샘플 복잡성에 미치는 영향을 구체적으로 이해할 수 있게 되었습니다.



### Glance for Context: Learning When to Leverage LLMs for Node-Aware GNN-LLM Fusion (https://arxiv.org/abs/2510.10849)
- **What's New**: 이 논문은 텍스트 속성 그래프(text-attributed graph)에서 대형 언어 모델(Large Language Models, LLMs)과 그래프 신경망(Graph Neural Networks, GNNs)의 융합 전략을 재구성합니다. 저자들은 기존 LLM-GNN 융합 방식이 모든 노드에 균일하게 적용되어 일부 지역에서 손실을 발생시키고 있다고 주장합니다. 이를 해결하기 위해 GLANCE라는 새로운 프레임워크를 제안하며, 이 프레임워크는 GNN들이 한계에 부딪히는 노드를 대상으로 하여 LLM을 선별적으로 활용합니다.

- **Technical Details**: GLANCE는 LLM의 예측을 향상시키기 위해 GNN의 특징을 보존하며, 경량화된 라우터를 사용하여 비용 효율적인 정책으로 LLM 호출 여부를 결정합니다. 이 라우터는 LLM 호출과 GNN 의존의 유틸리티를 비교하는 이점 기반 목표를 통해 훈련됩니다. 또한, LLM 호출의 비미분적 특성을 고려하여 라우터를 설계하여 성능을 최적화합니다.

- **Performance Highlights**: GLANCE는 여러 벤치마크에서 최고의 성능을 달성하며, 특히 이질적인 노드(heterophilous nodes)에서 최대 13%의 성능 향상을 보여줍니다. 이 방법은 LLM의 호출 빈도를 줄이고 계산 비용을 낮추면서도 다양한 속성의 그래프에서 더욱 견고한 예측을 가능하게 합니다. GLANCE의 결과는 노드 인식 GNN-LLM 융합의 가치를 입증합니다.



### Aegis: A Correlation-Based Data Masking Advisor for Data Sharing Ecosystems (https://arxiv.org/abs/2510.10810)
Comments:
          Accepted at SIGMOD 2026

- **What's New**: 본 논문은 데이터 공유 생태계에서 데이터 제공자가 데이터의 프라이버시 문제로 인해 데이터를 익명화해야 하는 상황을 다룹니다. AEGIS라는 중재(middleware) 프레임워크를 통해 머신러닝 데이터셋의 최적 익명화 구성(configuration)을 식별하는 방법을 제안합니다. AEGIS는 제한된 데이터 요약 정보를 활용하여 데이터의 유용성을 극대화하는 최적의 익명화를 결정하는 데 중점을 두고 있습니다.

- **Technical Details**: AEGIS는 기능(feature)과 클래스 레이블(class label)로 구성된 데이터셋을 위해 설계되었습니다. 이 프레임워크는 예측 유용성의 편차(predictive utility deviation)를 최소화하는 유용성 최적화기(utility optimizer)를 도입하여, 익명화 전후의 기능-레이블 상관관계(correlation) 변화에 기반한 메트릭을 사용할 수 있도록 합니다. Raw 데이터에 접근할 수 없는 경우를 고려하여 1D 히스토그램과 같은 제한된 데이터 요약을 활용하여 기능-레벨의 조인트 분포(joint distribution)를 추정하는 데 초점을 맞춥니다.

- **Performance Highlights**: 실제 데이터셋에 대한 실험 평가 결과, AEGIS는 최적의 익명화 구성을 이전보다 수십 배 더 빠르게 찾을 수 있음을 보여주었습니다. 결과적으로 생성된 익명화 데이터셋은 다운스트림 머신러닝 작업에서 기존의 기법들과 유사한 예측 성능을 나타냈습니다. 이러한 성과는 AEGIS의 유용성이 데이터 보안과 성능 최적화를 동시에 달성할 수 있음을 증명합니다.



### Crisis-Aware Regime-Conditioned Diffusion with CVaR Allocation (https://arxiv.org/abs/2510.10807)
Comments:
          Code available at: this https URL

- **What's New**: 이번 연구는 구체적인 시장 상태를 고려한 생성 시나리오와 볼록 CVaR(Conditional Value at Risk) 할당기가 시장 상태 변화 속에서 포트폴리오 결정 개선에 기여하는지를 분석합니다. Multi-Agent Regime-Conditioned Diffusion (MARCD) 방식은 잠재 상태를 추론하고, 극단적인 결과를 강조하여 위기 상황에서의 공통 변동성을 강화합니다. 특히, 기존 생성 모델보다 크고 불리한 결과를 중시하여 포트폴리오 손실을 줄이는 데 초점을 맞추고 있습니다.

- **Technical Details**: MARCD는 다중 에이전트 시스템으로, 각기 다른 시장 상태에 맞게 파라미터가 조정되는 mixture-of-experts (MoE) 모델을 포함하고 있습니다. 이 모델은 Gaussian HMM(Hidden Markov Model)을 통해 시장 상태를 추론하며, 생성된 시나리오를 CVaR에 기반한 할당 프로그램에 입력합니다. 이를 통해 수익률의 비상적인 의존성을 설정하여 포트폴리오의 위험 조정 수익률을 극대화하는 것을 목표로 합니다.

- **Performance Highlights**: MARCD는 2005-2025년 동안 유동적인 다자산 ETF에 대한 엄격한 워크포워드 테스트에서 표준 할당기보다 우수한 성과를 보였습니다. 특히 2020-2025년의 아웃 오브 샘플 실험에서 Sharpe 비율 1.23을 기록했으며, 최대 손실(MaxDD)은 9.3%로 나타났습니다. 이는 기존 모델 대비 34% 감소된 수치로, 투자 의사 결정에서의 통계적 유의성을 확인할 수 있었습니다.



### PruneGCRN: Minimizing and explaining spatio-temporal problems through node pruning (https://arxiv.org/abs/2510.10803)
- **What's New**: 이번 연구에서는 딥러닝 모델을 사용하여 그래프 구조를 가지는 문제를 다루고, 이를 통해 설명 가능성을 통합할 수 있는 새로운 접근 방식을 제안합니다. 특히, 모델이 훈련 과정에서 그래프의 노드를 효율적으로 제거하는 최적화된 가지치기 메커니즘을 통합하는 것을 목표로 합니다. 이는 예측 오류를 최소화하면서 가장 관련성이 높은 노드를 선택하는 데 도움을 줍니다.

- **Technical Details**: 이 모델은 Prune Graph Convolutional Recurrent Network (PruneGCRN)이라는 명칭을 가지고 있으며, 이 네트워크는 훈련 동안 불필요한 노드 제거를 통해 데이터를 최적화합니다. 스페이셜 및 템포럴 데이터의 다차원적 특성을 반영하기 위해 Graph Neural Networks (GNNs)와 Recurrent Neural Networks (RNNs)의 조합이 활용됩니다. 또한, 모델은 실시간 교통 데이터에서 각 노드가 의미하는 내용을 바탕으로 중요한 정보를 추출합니다.

- **Performance Highlights**: 실험 결과, PruneGCRN 모델은 다른 방법에 비해 더 많은 정보를 유지하면서 그래프의 크기를 줄이는 것으로 나타났습니다. 이는 교통 예측 문제를 해결하는 데 있어 모델이 가장 중요한 요소를 식별하고, 분석을 용이하게 하는 데 기여하고 있습니다. 따라서 본 연구는 스페이셜-템포럴 문제를 간소화하는 모델 개발에 있어 가지치기의 가능성을 강조합니다.



### Rethinking deep learning: linear regression remains a key benchmark in predicting terrestrial water storag (https://arxiv.org/abs/2510.10799)
- **What's New**: 최근 기계 학습의 발전으로 Long Short-Term Memory (LSTM) 모델과 Transformers가 수문학적 응용 분야에서 널리 채택되었습니다. 이들은 다양한 작업에서 물리적 모델보다 우수한 성능을 보였습니다. 그러나 자연 변동성과 인간에 의한 수정 등이 지배하는 육상 수문 상태 예측에서는 그 장점이 명확하지 않았습니다.

- **Technical Details**: 이 연구에서는 HydroGlobe 데이터셋을 사용하였으며, 이는 단순 지표 모델 시뮬레이션에서 유래된 기준 버전과 다중 원격 감지 데이터 동화를 포함한 고급 버전으로 구성됩니다. 결과적으로, 선형 회귀(Linear Regression)는 TWS 예측을 위한 견고한 기준선으로, 복잡한 LSTM 및 Temporal Fusion Transformer보다 우수한 성능을 보였습니다. 이는 전통적인 통계 모델을 기준으로 포함하는 것의 중요성을 강조합니다.

- **Performance Highlights**: 이 결과는 딥러닝 모델 개발 및 평가 시 전통적인 통계 모델을 벤치마크로 삼는 것이 필요하다는 점을 강조합니다. 또한, 자연 변동성과 인간의 영향을 모두 반영할 수 있는 전 세계적으로 대표적인 벤치마크 데이터셋의 필요성을 강조합니다.



### BioOSS: A Bio-Inspired Oscillatory State System with Spatio-Temporal Dynamics (https://arxiv.org/abs/2510.10790)
- **What's New**: 이 논문에서는 생물학적 신경의 특성을 모방한 생물영감을 받은 진동 상태 시스템(BioOSS)을 제안합니다. 기존의 딥러닝 모델이 지닌 한계를 보완하기 위해, BioOSS는 신경 회로에서 관찰되는 파동 전파 dynamics를 꿰뚫게 모델링할 수 있는 것으로 설계되었습니다. 특히 전두엽(prefrontal cortex)에서의 복잡한 활동 패턴을 직접적으로 반영하고자 합니다.

- **Technical Details**: BioOSS는 두 가지 상호작용하는 뉴런 집단, 즉 p 뉴런과 o 뉴런으로 구성되어 있으며, p 뉴런은 피라미드 세포에서 영감을 받은 단순화된 막 전위와 유사한 유닛을 나타냅니다. o 뉴런은 정보의 전파 속도를 조절하고 활동의 측면 확산을 조절합니다. 이 모델은 감쇠(damping)와 전파 속도(propagation speed) 용량을 조정 가능한 파라미터로 포함하여 특정 작업에 맞게 적응할 수 있도록 제공합니다.

- **Performance Highlights**: BioOSS는 합성 데이터 및 실제 세계의 작업에서 평가되어, 다른 아키텍처보다 우수한 성능과 향상된 해석성을 보여주었습니다. 실험 결과, 기존의 선형 변환만으로는 복잡한 진동 다이나믹스를 생성할 수 없음을 입증하였으며, BioOSS의 파라미터가 적절히 학습되었을 때 더 높은 품질의 spatio-temporal 패턴을 생성함을 확인하였습니다.



### Preconditioned Norms: A Unified Framework for Steepest Descent, Quasi-Newton and Adaptive Methods (https://arxiv.org/abs/2510.10777)
Comments:
          22 pages, 2 figures, 8 tables

- **What's New**: 이번 연구에서는 steepest descent, quasi-Newton 및 adaptive 방법들을 일반화한 통일된 프레임워크를 제안합니다. 이 프레임워크는 사전 조건화된 행렬 노름(preconditioned matrix norms)이라는 새로운 개념을 도입하여 기존 최적화 기법들이 모두 동일한 원리를 따른다는 것을 보여줍니다. 저자들은 이를 통해 새로운 최적화 알고리즘 MuAdam 및 MuAdam-SANIA를 소개하고, 이들이 기존의 최첨단 방법들과 경쟁력을 갖추고 있음을 보여줍니다.

- **Technical Details**: 연구에서는 최적화의 기하학적 적응성과 커브쳐 인식을 통합하는 방법을 제안합니다. 기존의 방법들은 Frobenius 노름에 제한되어 있었으나, 새로운 접근법은 다양한 구조의 딥 러닝 아키텍처에 맞게 더 복잡한 기하학적 구조를 포착할 수 있습니다. 일반화된 노름을 사용함으로써, 다양한 최적화 알고리즘들이 어떻게 통합될 수 있는지를 체계적으로 분석합니다.

- **Performance Highlights**: 실험 결과, 새로운 최적화 방법인 MuAdam과 MuAdam-SANIA가 기존의 최첨단 방법들과 비교했을 때 경쟁력을 유지하며, 어떤 경우에는 성능을 초월하기도 했습니다. 이 연구는 딥러닝 최적화 분야에서 기하학적 속성 및 효율성을 동시에 충족시킬 수 있는 가능성을 보여줍니다.



### Structure Over Signal: A Globalized Approach to Multi-relational GNNs for Stock Prediction (https://arxiv.org/abs/2510.10775)
- **What's New**: 이번 논문에서는 OmniGNN이라는 Attention 기반의 다중 관계형 동적 그래프 신경망 모델을 제안합니다. OmniGNN은 이질적 노드와 엣지 유형을 통합하여 거시경제(context) 상황에 적응한 메시지 전달을 가능하게 합니다. 이 모델은 산업 노드를 글로벌 중개 역할로 설계하여, 긴 거리의 메시지 전달을 간소화합니다.

- **Technical Details**: OmniGNN의 아키텍처는 크게 세 가지 핵심 요소로 구성됩니다: 구조적 레이어, 동적 레이어, 예측 레이어입니다. 구조적 레이어는 메타패스(metapath) 가중치를 기반으로 노드를 인코딩하고, 동적 레이어는 시계열 데이터를 통해 노드 표현을 학습합니다. 예측 레이어는 주어진 주식 노드에 대해 다음 날 초과 수익을 예측합니다.

- **Performance Highlights**: 실험 결과, OmniGNN은 기존의 주식 예측 모델들보다 우수한 성능을 보였으며, 특히 COVID-19 기간 동안 강한 견고성을 나타냈습니다. 이 모델은 GAT(그래프 어텐션 네트워크)를 활용하여 인근 노드의 기여도를 가중 평균하여 보다 정확한 예측을 가능하게 합니다.



### Understanding Sampler Stochasticity in Training Diffusion Models for RLHF (https://arxiv.org/abs/2510.10767)
- **What's New**: 이번 연구에서는 Human Feedback로부터의 강화 학습(RLHF)을 통해 확산 모델을 미세 조정할 때 발생하는 도전 과제인 보상 간극(reward gap)을 이론적으로 분석하고, 일반적 확산 모델에 대한 비허무 경계(non-vacuous bounds)와 Variance Exploding(VE) 및 Variance Preserving(VP) 가우시안 모델의 수렴 속도를 제공하였다. 이 과정에서 일반화된 디노이징 확산 암묵 모델(gDDIM) 프레임워크를 도입하여 임의의 높은 수준의 확률성을 지원하고 데이터를 보존하는 방식을 강조하였다.

- **Technical Details**: 본 연구는 확산 모델의 전방 및 후방 프로세스를 포함하는 연속 시간 확산 모델을 설명하고, SDE(Stochastic Differential Equations)을 사용하여 목표 데이터 분포를 생성하는 목표를 설정하였다. 이 모델의 후방 프로세스는 목표 분포를 알 수 없기 때문에 사전 분포(pnoise)를 사용하여 시작되며, 이를 통해 역시간 과정의 산출을 정당화하였다. 이러한 이론적 기초는 Gronwall의 부등식을 사용하여 SDE 미세 조정된 모델과 ODE(Ordinary Differential Equations) 샘플링 모델 간의 보상 간극을 제한하는 데 기여하였다.

- **Performance Highlights**: 대규모 텍스트-이미지 모델 및 RLHF 알고리즘에 대한 실험을 통해, 훈련 단계에서 보상 간극이 일관되게 줄어들며 ODE 샘플링의 품질이 향상된다는 것을 입증하였다. 특히, 중간에서 높은 확률성의 훈련(예: η=1.2)이 도메인 내 및 도메인 외 성능을 향상시킨다는 결과를 보였으며, ODE 추론이 소규모 디노이징 단계 예산 하에서도 안정적으로 SDE 추론보다 더 우수한 성능을 발휘하였다.



### Optimally Deep Networks -- Adapting Model Depth to Datasets for Superior Efficiency (https://arxiv.org/abs/2510.10764)
Comments:
          6 pages, 3 figures, 1 table

- **What's New**: 새로운 논문에서는 Optimally Deep Networks (ODNs)를 소개합니다. 이 네트워크는 모델의 깊이를 주어진 작업의 복잡도에 맞추어 조절하여, 과도한 신경망 훈련을 피하고 효율성을 높이는 방법을 제안합니다. 특히, "progressive depth expansion"이라는 훈련 전략을 통해 더浅 깊이에서 시작하여 성능에 도달할 때까지 점진적으로 깊이를 조정합니다.

- **Technical Details**: ODNs는 특정 데이터셋에 최적화된 깊이만 사용하여 불필요한 레이어를 제거합니다. 이 전략은 메모리 사용량을 줄이고, 계산 효율성을 향상시키며, 자원이 제한된 장치에서의 배포를 용이하게 합니다. 또한, ODN은 기존의 대규모 네트워크 아키텍처에 쉽게 적용 가능하며, ResNet, MobileNet 등 여러 모델에 대해 적용할 수 있습니다.

- **Performance Highlights**: 실험 결과, ResNet-18과 ResNet-34 모델이 각각 MNIST와 SVHN 데이터셋에 대해 메모리 사용량을 각각 98.64%와 96.44% 줄이면서도 경쟁력 있는 정확도를 유지했습니다. 이로 인해 ODNs는 자원 소모를 최소화하면서도 높은 정확도를 성취할 수 있습니다. 이는 깊이 탐색 전략을 통해 모델의 효율성을 크게 향상시킨 결과입니다.



### A Stochastic Differential Equation Framework for Multi-Objective LLM Interactions: Dynamical Systems Analysis with Code Generation Applications (https://arxiv.org/abs/2510.10739)
Comments:
          Peer-reviewed and accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) DynaFront 2025 Workshop (this https URL)

- **What's New**: 우리는 반복적인 대형 언어 모델(LLM) 상호작용에서 다목적 최적화의 역학을 모델링하기 위한 일반적인 확률적 미분 방정식(SDE) 프레임워크를 소개합니다. 이 프레임워크는 LLM 응답의 고유한 불확실성을 명시적인 확산( diffusion) 항을 통해 포착하며, 상충하는 목표 간의 체계적인 간섭 패턴을 간섭 행렬(interference matrix) 형식을 통해 드러냅니다.

- **Technical Details**: 우리의 접근 방식은 목표 벡터의 연속적 시간 발전을 드리프트-확산(drift-diffusion) 과정으로 모델링하여 수렴 특성, 안정성 조건 및 상충 목표 간의 간섭 패턴을 엄격하게 분석하게 합니다. 모델링 과정에서는 𝐱(t)∈ℝ^n 형태의 목표 벡터와 관련하여 SDE를 기반으로 한 구성을 제공합니다. 이를 통해 품질-다양성 트레이드오프와 같은 다양한 어플리케이션에서의 분석을 가능하게 합니다.

- **Performance Highlights**: 코드 생성에서의 초기 검증 도메인을 통해, 400개의 세션을 분석하고 보안, 효율성, 기능성 목표에 걸쳐 전략 의존적인 수렴 행동을 보여주며, 수렴 속도는 0.33부터 1.29까지 변화합니다. 예측 정확도는 균형 잡힌 접근 방식에서 R² = 0.74를 달성하였습니다. 이러한 결과는 다목적 LLM 상호작용에 대한 동적 시스템 분석의 실행 가능성을 제시합니다.



### Provable Anytime Ensemble Sampling Algorithms in Nonlinear Contextual Bandits (https://arxiv.org/abs/2510.10730)
Comments:
          40 pages, 1 figure

- **What's New**: 이 논문은 비선형 컨텍스트 밴딧(Nonlinear Contextual Bandits)에서 앙상블 샘플링(ensemble sampling)에 대한 통합 알고리즘 프레임워크를 제시합니다. 또한, 일반화 선형 앙상블 샘플링(GLM-ES)과 신경 앙상블 샘플링(Neural-ES) 두 가지에 대한 후회 경계(regret bounds)를 개발하였습니다. 두 방법 모두 랜덤하게 변동된 데이터에 대한 최대 우도 추정(maximum likelihood estimation)을 통해 보상 모델 파라미터에 대한 여러 추정기를 유지합니다.

- **Technical Details**: GLM-ES에 대한 후회 경계는 $\\mathcal{O}(d^{3/2} \sqrt{T} + d^{9/2})$로, Neural-ES는 $\\mathcal{O}(\widetilde{d} \sqrt{T})$로 설정되어 있으며, 여기서 $d$는 특징 벡터의 차원, $\widetilde{d}$는 신경 접선 커널(neural tangent kernel) 매트릭스의 유효 차원, 그리고 $T$는 라운드 수를 의미합니다. 이론적 분석에서는 비선형 모델에 특정한 도전을 해결하는 기술을 도입하였습니다. 또한, 고정 시간 수명 가정을 제거하고 비선형 밴딧에 적합한 anytime 버전을 개발하여 알고리즘의 적용 범위를 확장하였습니다.

- **Performance Highlights**: GLM-ES, Neural-ES 및 그들의 anytime 변형을 실험적으로 평가하여 강력한 성능을 보였습니다. 연구 결과는 비선형 컨텍스트 밴딧의 랜덤 탐색 접근법은 증명이 가능하고 실용적이라는 것을 입증하였습니다. 특히, 기존의 메타 분석에서는 선형 컨텍스트 밴딧에 대한 후회 보장만 제공되었던 반면, 본 연구는 비선형 밴딧의 경우 처음으로 높은 확률의 후회 경계를 제안합니다.



### Designing ReLU Generative Networks to Enumerate Trees with a Given Tree Edit Distanc (https://arxiv.org/abs/2510.10706)
- **What's New**: 이 논문에서는 주어진 트리와 유사한 구조의 트리를 생성하기 위한 생성 네트워크의 존재와 구성을 이론적으로 확립합니다. 특히, 주어진 트리 T로부터 트리 편집 거리(Tree Edit Distance)가 최대 d인 모든 트리를 생성할 수 있는 ReLU 기반의 생성 네트워크를 제안하였습니다. 이 네트워크들은 O(n³)의 크기와 상수 깊이로 구현 가능하며, 21개의 노드까지의 트리 생성에 대한 평가도 포함되어 있습니다.

- **Technical Details**: 트리 편집 거리의 정의에 따라, 주어진 루트가 있는 순서가 있는 정점 레이블이 붙은 트리 T와 그 거리 d를 기준으로, 특정 ReLU 기반 네트워크가 모든 유효한 트리를 생성할 수 있음을 수학적으로 증명합니다. 기존의 비결정론적 알고리즘에 비해 이 네트워크는 특정 트리 편집 거리 내에서 유효한 트리를 적용해 성공적으로 생성했습니다. 일반적으로, 트리의 생성은 작업의 복잡성에 따라 달라지며, 이 연구는 그러한 내용의 근거를 제공합니다.

- **Performance Highlights**: 제안된 네트워크는 21개의 노드로 구성된 트리를 성공적으로 생성하였으며, 결정론적 구조 덕분에 모든 유효한 트리를 생성할 수 있었습니다. 반면, 최신 그래프 생성 모델인 GraphRNN과 GraphGDP는 비결정론적인 방법에 의존하여 유효한 트리의 생성률이 각각 35%와 48%에 불과했습니다. 이러한 결과는 Compact Generative Models 구성에 대한 이론적 기초를 마련하고, 정확하고 유효한 트리 구조 데이터 생성을 위한 새로운 방향을 열어줍니다.



### Attention-Enhanced LSTM Modeling for Improved Temperature and Rainfall Forecasting in Bangladesh (https://arxiv.org/abs/2510.10702)
- **What's New**: 이 연구에서는 기후 변화의 영향이 큰 방글라데시에서 온도와 강수 예측을 개선하기 위해 주목 메커니즘이 통합된 향상된 Long Short-Term Memory (LSTM) 모델을 소개합니다. 1901년부터 2023년까지의 포괄적인 데이터를 활용하며, 기존 모델들이 포착하지 못했던 계절적 및 장기적 추세를 효과적으로 분석하여 더욱 정교한 예측을 가능하게 합니다. 본 연구는 그동안의 연구에서는 미비했던 복잡한 비선형 시계열 특성을 보다 잘 모델링할 수 있는 방법을 제시합니다.

- **Technical Details**: LSTM 모델은 시계열 데이터의 순차적 특성을 잘 포착하는 데 강점을 지니며, 이전 값들을 오랜 기간 동안 기억하는 능력이 있습니다. 본 연구에서 제안한 모델은 주목 메커니즘을 통합하여 중요한 시간 패턴을 동적으로 가중치화함으로써 단기 변동성과 장기 기후 추세를 보다 잘 포착합니다. 또한, 본 모델은 복잡한 스택킹이나 하이브리드 모델과 비교할 때 계산 효율성을 높이면서도 예측 성능을 향상시키도록 설계되었습니다.

- **Performance Highlights**: 모델은 기존의 XGBoost, Simple LSTM 및 GRU 모델을 초월하여 뛰어난 예측 성능을 보여주었습니다. 예측 테스트에서 온도에 대한 MSE는 0.2411, R^2 값은 0.9834, 강수에 대한 MSE는 1283.67 mm², R^2 값은 0.9639를 기록하며, 안정성 강화를 위해 기후 변동에 대한 변화에 대해서도 향상된 성능을 유지합니다. 이 결과들은 해당 모델이 방글라데시의 기후 변화 응답 분야에서 중요한 도구로 자리잡을 수 있음을 나타냅니다.



### Stock Prediction via a Dual Relation Fusion Network incorporating Static and Dynamic Relations (https://arxiv.org/abs/2510.10695)
Comments:
          11 pages

- **What's New**: 이 논문에서는 주식 가격 예측을 위한 'Dual Relation Fusion Network (DRFN)'를 제안합니다. 기존의 연구들은 주로 단일 상태의 관계에 집중하여 주식 간의 중요한 상호 보완성(dynamic과 static inter-stock relations)을 간과해 왔습니다. DRFN은 시간이 지남에 따라 상대적인 안정성을 유지하며, 시장의 급격한 변화에 신속하게 적응할 수 있는 구조를 가지고 있습니다.

- **Technical Details**: DRFN은 시간에 따라 변화하는 장기 패턴을 모델링하는 새로운 상대 정적 관계 구성 요소를 특징으로 합니다. 또한, 거리 인식 메커니즘을 통해 동적인 주식 관계를 포착하고, 전일의 동적 관계와 미리 정의된 정적 관계를 반복적으로 융합하는 방식으로 장기 구조를 발전시킵니다. 이 과정에서 정보의 흡수를 위해 뉴스와 시장 지표의 임베딩을 정렬하여 강력한 주식 표현을 생성합니다.

- **Performance Highlights**: 실험 결과, DRFN은 다양한 시장에서 기초 모델보다 현저하게 우수한 성능을 보였습니다. 특히, 관계의 강도와 주가의 공동 움직임에 높은 민감도를 나타내며, 시장 비효율성에 의해 움직이는 가격 변화를 보다 잘 예측할 수 있습니다. 이러한 방식으로 DRFN은 주식 예측에 있어 기존 모델들이 간과한 여러 변수를 효과적으로 통합합니다.



### Digital Twin-enabled Multi-generation Control Co-Design with Deep Reinforcement Learning (https://arxiv.org/abs/2510.10694)
Comments:
          to be published in Journal of Mechanical Design

- **What's New**: 본 연구는 Control Co-Design (CCD)를 다세대 설계(multi-generation design)로 확장하여 물리적 시스템과 제어 정책을 동적으로 조정하는 프레임워크를 제안합니다. 특히 Digital Twin (DT) 기술과 Deep Reinforcement Learning (DRL)을 통합하여 실시간 의사결정과 시스템 재설계를 지원합니다. 이 연구는 불확실한 환경에서 최적의 성능을 달성할 수 있는 방법론을 제시하며, 미래 세대의 물리적 시스템 디자인 개선을 위한 데이터 활용 방안을 제안합니다.

- **Technical Details**: Control Co-Design (CCD)은 시스템 역학과 제어 전략 간의 결합을 명시적으로 고려하여 물리적 시스템 디자인과 제어 시스템 디자인 문제를 통합합니다. 이 연구에서 제안하는 CCD 프레임워크는 실시간 센싱, 모델 업데이트, adaptive re-optimization을 통해 진화하는 가상 표현을 만들어내는 DT 기술을 활용합니다. 이를 통해 사용자 맞춤형 서스펜션 설정을 제공할 수 있는 다세대 설계 접근법이 도입되며, 데이터를 통해 불확실성을 효과적으로 수량화하는 방법론도 논의됩니다.

- **Performance Highlights**: 활성 서스펜션 시스템을 위한 사례 연구를 통해 이 프레임워크가 높은 불확실성 환경에서도 적응하며 성능을 향상시키는 능력을 보여줍니다. 결과는 성능과 강건성(dynamic performance, robustness) 및 효율성(efficiency)의 상당한 향상을 입증합니다. 이 연구는 DT와 DRL의 통합을 통해 불확실한 조건에서 더욱 부드럽고 안정적인 제어 경로를 생성할 수 있음을 강조합니다.



### Trustworthy Retrosynthesis: Eliminating Hallucinations with a Diverse Ensemble of Reaction Scorers (https://arxiv.org/abs/2510.10645)
- **What's New**: 이 논문에서는 RetroTrim이라는 새로운 회귀 합성 시스템을 소개합니다. 이 시스템은 약물과 유사한 도전적인 타겟 세트에서 nonsensical한 경로를 성공적으로 회피하는 것이 특징입니다. RetroTrim은 hallucinated (환각된) 반응을 필터링하는 유일한 방법으로, 전체적으로 높은 품질의 경로를 생성하는 것으로 입증되었습니다.

- **Technical Details**: RetroTrim은 다양한 반응 평가 전략을 조합하여 hallucinations (환각) 문제를 해결하는 데 중점을 두고 있습니다. 이 시스템은 Reaction Prior (RP), Reaction Graph Plausibility (RGP), Reaction Retrieval Score (RRS)라는 세 가지 주요 스코어러를 사용하여 반응의 신뢰성을 평가합니다. 각 스코어러는 서로 다른 종류의 hallucinations을 필터링하는 데 강점을 보이며, 메타 스코어러는 이를 종합적으로 평가합니다.

- **Performance Highlights**: RetroTrim은 32개의 신약 유사 타겟 세트에서 기존의 다른 회귀 합성 시스템과 비교했을 때, hallucinated 반응을 모두 배제하고 가장 많은 문제 없는 합성 경로를 발견한 것으로 나타났습니다. 본 논문은 약리 화학 영역에서 신뢰할 수 있는 회귀 합성을 연구하자는 목표를 가지고 있으며, 평가 프로토콜과 기준이 되는 데이터 세트를 공개하여 후속 연구를 자극하고자 합니다.



### ProteinAE: Protein Diffusion Autoencoders for Structure Encoding (https://arxiv.org/abs/2510.10634)
- **What's New**: ProteinAE는 효율적인 단백질 구조 인코딩과 생성을 위한 혁신적인 단백질 확산 오토인코더(autoencoder)로, E(3)에서 단백질 백본 좌표를 직접 연속적이고 압축된 잠재 공간(latent space)으로 매핑합니다. 기존 모델들이 가진 SE(3) 다양체(manifold)의 복잡성과 불필요한 이산화(tokenization)의 문제를 극복하여, 보다 간단하고 효과적인 구조를 제공합니다. 또한 ProteinAE는 단일 흐름 맞춤(flow matching) 목표를 사용해 훈련되며, 최적화 프로세스를 단순화했습니다.

- **Technical Details**: ProteinAE는 단백질 구조 표현을 학습하기 위한 인코더-디코더 아키텍처를 사용합니다. 이 구조는 초기 잔기 수준의 특징을 구축한 다음, 이를 알파벳 수준의 피처로 변환해 인코더에 입력합니다. 인코더는 전통적인 오토인코더인 ESM3와는 달리 비점근적(non-equivariant) 구조를 채택하고 있으며, 단백질 백본 원자의 좌표를 직접 사용하여 복잡한 매니폴드 디자인을 회피합니다.

- **Performance Highlights**: ProteinAE는 CASP14 및 CASP15 벤치마크에서 최첨단 재구성 품질을 달성했습니다. 이 모델이 학습한 잠재 공간은 높은 정확도를 바탕으로 한 물리화학적 속성 예측에 유리하며, Protein Latent Diffusion Model(PLDM)로서도 성공적입니다. 마지막으로, ProteinAE는 다른 기존 구조 기반 생성 모델과 경쟁할 수 있는 우수한 샘플 품질과 효율성을 보여주고 있습니다.



### ImpMIA: Leveraging Implicit Bias for Membership Inference Attack under Realistic Scenarios (https://arxiv.org/abs/2510.10625)
- **What's New**: 이 논문에서는 Membership Inference Attack (MIA)에서의 최신 기법인 ImpMIA를 소개합니다. ImpMIA는 기존의 참조 모델에 의존하지 않고, 신경망의 암묵적 편향(implicit bias)을 이용하여 훈련 샘플을 식별합니다. 기존의 가정들을 제거함으로써, 보다 현실적인 환경에서의 공격 성능이 크게 향상되었습니다.

- **Technical Details**: ImpMIA는 Karush-Kuhn-Tucker (KKT) 최적 조건을 통해 훈련 샘플을 식별합니다. 이 방법은 모델의 가중치와 훈련 데이터의 서브셋만을 가지고 작동하며, 훈련 절차나 데이터 분포에 대한 추가 정보 없이도 효과적으로 작동합니다. 이 공격은 훈련 데이터의 경량화된 그라디언트 계산을 통해 이루어집니다.

- **Performance Highlights**: ImpMIA는 세 가지 벤치마크 데이터셋에서 평가되었으며, 기존의 블랙박스 및 화이트박스 공격 방법들보다 우수한 성능을 보였습니다. 우리의 연구 결과에 따르면, 많은 참조 모델 기반 방법들이 의존하는 가정들이 제거될 때 성능이 크게 감소하지만, ImpMIA는 이러한 변화에 영향받지 않는다는 점이 강조되었습니다.



### SDG-L: A Semiparametric Deep Gaussian Process based Framework for Battery Capacity Prediction (https://arxiv.org/abs/2510.10621)
- **What's New**: 이번 논문에서는 리튬 이온 배터리의 용량 예측을 위한 새로운 방법론인 SDG-L을 제안합니다. 이 방법은 베이지안 접근법인 Deep Gaussian Process Regression(DGPR)과 LSTM(feature extractor)을 결합하여 배터리 상태 정보를 더 효과적으로 활용합니다. 기존의 연구들에 비해 예측 성능이 우수하며, NASA 데이터셋을 통한 실험을 통해 이 프레임워크의 유효성을 입증했습니다.

- **Technical Details**: SDG-L은 배터리의 충전 및 방전 과정에서 발생하는 상태 데이터를 기반으로 시계열(time series) 예측을 수행하는 반모수(semi-parametric) 모델입니다. LSTM을 사용하여 여러 사이클에 걸쳐 원시적인 배터리 데이터를 효과적으로 추출하고, DGPR 모듈을 통해 배터리 용량의 손실을 모델링합니다. 본 연구에서 제안하는 방법은 기존의 수치적 기법들과 비교하여 더 높은 예측력과 신뢰성을 제공합니다.

- **Performance Highlights**: 실험 결과, SDG-L은 평균 MSE(Mean Squared Error)가 1.2%에 이르며, 기존 방법에 비해 월등한 성능을 보였습니다. 또한, 애블레이션(ablation) 연구를 통해 기법의 유효성을 검증하고 전반적인 배터리 상태 예측에서의 가능성을 제시했습니다. 향후 연구 방향에 대해서는  다른 데이터셋과의 비교 및 추가적인 성능 향상 가능성을 논의합니다.



### Encoder Decoder Generative Adversarial Network Model for Stock Market Prediction (https://arxiv.org/abs/2510.10617)
- **What's New**: 이 논문에서는 주가 예측의 어려움을 해결하기 위해 새로운 GRU 기반 Encoder-Decoder GAN(EDGAN) 모델을 제안합니다. 기존 GAN의 문제점인 모드 붕괴(mode collapse), 불안정한 훈련, 시간 및 특성 수준 상관관계 캡처의 어려움을 극복하고자 합니다. EDGAN은 정적(static) 및 동적(dynamic) 공변량을 조건으로 사용하여 예측 정확도를 높이는데 중점을 두었습니다.

- **Technical Details**: EDGAN 모델은 잔여 연결(residual connections)을 통한 시간 디코더와 문맥 학습을 위한 조건부 학습 기법을 도입합니다. 또한, 시간 동력을 포착하기 위한 윈도잉(windowing) 메커니즘을 활용하여 주가 예측의 정밀도를 높입니다. 생성기(generator)는 밀집 인코더-디코더 프레임워크를 사용하여 잔여 GRU 블록을 적용함으로써 데이터의 복잡성을 줄입니다.

- **Performance Highlights**: 다양한 주식 데이터셋에 대한 광범위한 실험을 통해 EDGAN은 예측 정확도와 훈련 안정성 측면에서 기존의 GAN 변형들을 지속적으로 초월함을 입증하였습니다. 특히 변동성이 큰 시장에서도 뛰어난 성과를 달성했습니다. 이러한 결과는 EDGAN이 시장 조건하에서도 예측의 정확성과 수렴 안정성을 크게 개선했음을 보여줍니다.



### Budget Allocation for Unknown Value Functions in a Lipschitz Spac (https://arxiv.org/abs/2510.10605)
- **What's New**: 이 논문에서는 중간 모델(Intermediate models) 탐색을 위해 자원을 최적 배분하는 문제를 다룹니다. 기존의 하이퍼파라미터 최적화(Hyperparameter Optimization, HPO)의 모든 설정을 일반화하기 위해 Unknown Value Probing (UVP)이라는 개념을 도입하였습니다. 이를 위해 알고리즘을 개발하고, 이론적으로 강력한 보장과 실험을 통해 성능을 입증합니다. 이러한 새로운 프레임워크는 모델 탐색의 효율성을 향상시키기 위한 기반이 됩니다.

- **Technical Details**: 예측 성능을 직접적으로 측정하기 위해서는 컴퓨터 자원을 소비해야 하지만, 본 연구에서는 각 모델의 성능을 미리 알 수 없는 값 함수로 나타내고, 자원 할당의 경계를 Lіпшиц(Lipschitz) 공간에서 정의합니다. 주어진 총 자원을 바탕으로 서로 유사한 모델들이 어떻게 비슷한 성능 함수를 가질 것인지를 이론적으로 정립하고, 이를 기반으로 자원을 효율적으로 배분하는 방법을 제시합니다. FullCent, Enhanced-FullCent, AdaCent, Enhanced-AdaCent 등의 다양한 알고리즘들은 각기 다른 방식으로 자원을 할당하며, 이론적인 성과를 보장하고 있습니다.

- **Performance Highlights**: 실험 결과, 본 연구에서 제안한 알고리즘들은 기존의 HPO 접근 방식에 비해 우수한 성능을 보였습니다. 250개 이상의 실험 설정에서 두 알고리즘이 고전적인 HPO 기법을 능가하는 성과를 보였습니다. 여러 데이터셋에서 평균 성과 순위를 집계하고, 예산 대비 정확도(budget versus accuracy) 곡선을 통해 모델의 효율성을 비교하였습니다.



### FusionGen: Feature Fusion-Based Few-Shot EEG Data Generation (https://arxiv.org/abs/2510.10604)
- **What's New**: 본 논문에서는 EEG 데이터 생성 프레임워크인 FusionGen을 제안합니다. FusionGen은 제한된 데이터 환경에서도 다양한 EEG 신호를 효율적으로 합성하는 특징 융합 기반 접근방식을 사용합니다. 이 프레임워크는 실제 BCI 응용 분야에서의 일반화와 확장성을 향상시키기 위해 설계되었습니다.

- **Technical Details**: FusionGen은 (1) 특징 일치 융합 모듈을 통해 서로 다른 샘플의 특징을 통합하며, (2) 경량 특징 추출 및 재구성 파이프라인을 결합하여 데이터 다양성과 학습 가능성을 보장합니다. 이 과정에서 유클리드 정렬(Euclidean Alignment, EA) 기술을 이용하여 다양한 세션 및 주체 간의 분산 차이를 줄입니다.

- **Performance Highlights**: 다양한 공개 EEG 데이터 세트를 활용하여 진행한 실험에서 FusionGen은 기존의 데이터 증강 기술에 비해 뛰어난 성능을 보였습니다. 결과적으로 클래스 분류准确度(accuracy)가 유의미하게 향상되어, 해당 모델이 실제 응용 프로그램에서의 효과성을 높일 것임을 입증했습니다.



### Compositional Symmetry as Compression: Lie Pseudogroup Structure in Algorithmic Agents (https://arxiv.org/abs/2510.10586)
Comments:
          Submitted to NeurReps 2025 (this https URL)

- **What's New**: 이번 연구에서는 Kolmogorov 이론을 기반으로 한 알고리즘 에이전트들이 감각 스트림을 추적하고 압축하는 방법을 제안합니다. 저자들은 구성적 대칭(compositional symmetry)을 구조적 우선기준으로 하여, 에이전트가 특정한 심볼릭 프로그램으로 환경을 모델링 하는 프레임워크를 제시합니다. 이 모델은 에이전트를 일반적인 신경 동역학 시스템(neural dynamical system)으로 설정하여, 환경을 정확히 추적할 수 있도록 합니다.

- **Technical Details**: 연구에서는 유한 매개변수 Lie 유사군(Lie pseudogroup)의 지역적 행동을 사용하여 구성 매니폴드(configuration manifold) 위의 생성 모델을 정의합니다. 에이전트는 이러한 스트림에 의해 구동되는 신경 ODE(neural ODE)로 모델링되며, 이는 대칭 기반의 자기 포함형 예측 코딩(predictive coding)의 버전을 구성합니다. 대칭 관계에 따라 에이전트의 동역학과 구성 방정식들은 규약과 제약에 따라 제한됩니다.

- **Performance Highlights**: 이 연구는 심볼 표기(simbolic representation)의 조합적 우수성(the blessing of compositionality)이 심층 모델(deep models)에서 샘플 복잡도(sample complexity)를 낮출 수 있음을 강조합니다. 또한, bare manifold prior가 추가적인 기하학적 정보를 가진 구조 없이는 불충분하다는 점도 지적합니다. 이러한 결과들은 대칭 인식 설계(symmetry-aware designs)의 필요성을 시사하며, 예측 코딩의 그룹 이론적 관점을 제공합니다.



### Understanding Self-supervised Contrastive Learning through Supervised Objectives (https://arxiv.org/abs/2510.10572)
Comments:
          Accepted at TMLR 2025

- **What's New**: 이번 연구는 self-supervised representation learning을监督(수퍼바이즈드) representation learning의 근사로 공식화하여 이론적 관점을 제공합니다. 여기서 제안한 손실 함수는 InfoNCE와 같은 널리 사용되는 대조 손실(contrastive losses)과 밀접하게 연관되어 있습니다. 또한 프로토타입 표현 편향(prototype representation bias) 및 균형 잡힌 대조 손실(balanced contrastive loss)의 개념을 도입하여 self-supervised 학습 알고리즘의 동작을 개선하는 방법을 제시합니다.

- **Technical Details**: self-supervised representation learning은 본질적으로 ground-truth label 없이도 지도 학습의 목표와 연결될 수 있다는 점을 강조합니다. 저자들은 d차원에 대한 인코더 fθ를 정의하고, augmentation 과정을 통해 생성된 데이터셋에서 새로운 종류의 손실 함수를 유도하였습니다. 이 손실 함수는 SimCLR에서 사용되는 InfoNCE 손실과 매우 유사하며, 지도 학습과 self-supervised 학습 간의 관계를 설명하는 데 도움을 줍니다.

- **Performance Highlights**: 연구 결과, 긍정적 및 부정적 쌍 상호작용의 균형을 맞추는 것이 self-supervised 학습의 성능을 향상시키는 데 기여한다는 것을 실증적으로 확인했습니다. 연구진은 또한 프로토타입 표현 편향이 다운스트림 성능과 상관관계가 있음을 관찰했습니다. 이러한 통찰력은 self-supervised representation learning과 지도 학습 목표 간의 더욱 원칙적인 이해를 제공합니다.



### Multitask Learning with Learned Task Relationships (https://arxiv.org/abs/2510.10570)
- **What's New**: 본 연구에서 제안된 알고리즘 프레임워크는 기존의 일관된 합의 기반 전략의 한계를 극복하고, 각 에이전트가 로컬 최적 모델을 pursuit하면서 상호 이익을 취할 수 있도록 해준다. 특히, Gaussian Markov Random Field (GMRF) 모델을 통해 태스크 간의 관계를 모델링하고, 이를 통해 로컬 모델과 태스크 관계를 동시에 학습할 수 있는 전략을 개발했다. 이 접근 방식은 에이전트들이 자신의 데이터 분포와 일치하는 방식으로 자율적으로 조직될 수 있도록 향상시킨다.

- **Technical Details**: GMRF의 의존성을 통해, 에이전트 간의 관계를 그래프 형태로 모델링하고, 이 때의 정밀 행렬을 기본 그래프 라플라시안으로 설정하였다. 이러한 선택은 이웃한 에이전트 간에 매개변수의 유사성을 촉진하면서도 이질성을 허용한다. 제안된 전략은 비협조적 로컬 추정치로부터 태스크 관계를 추론하고 그래프 라플라시안을 통해 구조화된 협력을 증진하는 데 중점을 두었다.

- **Performance Highlights**: 본 연구는 학습된 관계의 품질을 정량적으로 분석하였고, 수치 실험을 통해 제안된 방법이 실제로 효과적임을 보여주었다. 특히, 대규모 센서 네트워크, 추천 시스템 및 연합 의료 분석과 같은 분야에 응용될 수 있는 가능성이 있다. 이 방법은 관련되어 있지만 동일하지 않은 모델을 학습해야 하는 에이전트들 간의 협력을 개선할 수 있도록 설계되었다.



### Multi-scale Frequency-Aware Adversarial Network for Parkinson's Disease Assessment Using Wearable Sensors (https://arxiv.org/abs/2510.10558)
- **What's New**: 본 논문에서는 파킨슨병(Parkinson's disease, PD)의 중증도를 평가하기 위해 새로운 Multi-scale Frequency-Aware Adversarial Multi-Instance Network (MFAM) 모델을 제안합니다. 이 모델은 의료 전문 지식을 통해 주파수 분해 모듈을 사용하여 PD 관련 신호를 더욱 정확하게 식별할 수 있도록 설계되었습니다. MFAM은 Attention 기반의 다중 인스턴스 학습(MIL) 프레임워크를 도입하여 진단적으로 중요한 드문 이벤트에 집중할 수 있습니다.

- **Technical Details**: MFAM 모델은 입력 다변량 시계열 데이터를 처리하여 특정 주파수 대역을 분리하는 주파수 분해 모듈(Frequency Decomposition Module, FDM)을 포함하고 있습니다. 이후, 다중 스케일 채널 주의 인코더(Multi-scale Channel Attention Encoder, MS-CAE)를 통해 깊이 있는 특징을 추출하며, Attention 기반의 MIL 집계기를 사용해 주요 정보를 포함한 배그 임베딩(Bag Embedding)을 생성합니다. 마지막으로 이 임베딩을 이용해 중증도 예측을 수행합니다.

- **Performance Highlights**: MFAM 모델은 공공 PADS 데이터셋 및 사설 데이터셋을 이용한 실험에서 다양한 최신 모델들을 초월하는 성능을 보여주었습니다. 이 모델은 PD의 복잡한 임상 시계열 데이터를 처리하는 데 있어 높은 특이성을 제공하며, 자동화된 PD 중증도 평가를 위한 유망한 솔루션을 제공합니다. 이러한 성과는 MFAM이 특정 진단 요구에 더 적합하다는 것을 입증합니다.



### PAC-Bayesian Reinforcement Learning Trains Generalizable Policies (https://arxiv.org/abs/2510.10544)
- **What's New**: 본 연구에서는 Markov 의존성을 명시적으로 반영한 새로운 PAC-Bayesian 일반화 경계를 도출하였습니다. 기존의 RL에서 데이터의 순차적 특성은 독립성 가정이 깨지기 때문에 일반화 보장을 얻는 데 어려움이 있었습니다. 제안된 경계는 Soft Actor-Critic과 같은 현대의 오프-폴리시(Off-Policy) 알고리즘에 비생성적인(certificates) 인증서를 제공합니다.

- **Technical Details**: 본 논문에서는 Markov 의존성을 체인 혼합 시간(mixing time)을 통해 명시적으로 다루는 PAC-Bayesian 일반화 경계를 제시하였습니다. 우리가 통합한 핵심 기술 기여는 부정적 경험적 반응에 대한 경계 차별 조건을 Markov 체인을 위한 McDiarmid-type 집중 불평등과 결합하는 것으로, 이를 통해 과거의 방법들이 갖는 공허함을 제거할 수 있었습니다.

- **Performance Highlights**: PB-SAC 알고리즘을 통해 제안된 경계의 실용성을 입증하였습니다. 연속 제어 작업에 대한 실험 결과, 우리 접근 방식이 믿을 수 있는 신뢰 증명서를 제공하고, 동시에 최신의 방법들과 경쟁력 있는 성능을 유지함을 보여주었습니다. 이는 현대 RL 알고리즘을 위한 첫 번째 실용적인 PAC-Bayesian 프레임워크를 확립하여 학습 이론과 알고리즘적 실천 간의 간극을 메우는 데 기여합니다.



### Rethinking RL Evaluation: Can Benchmarks Truly Reveal Failures of RL Methods? (https://arxiv.org/abs/2510.10541)
- **What's New**: 현재의 벤치마크는 대형 언어 모델(Large Language Models, LLMs)에서 강화 학습(Reinforcement Learning, RL)의 진행 상황 평가에 부적합하다는 점을 강조합니다. 이 논문에서는 Oracle Performance Gap (OPG)이라는 새로운 지표를 소개하여, 훈련 세트와 테스트 세트 간의 성능 차이를 정량적으로 측정할 수 있도록 합니다. 또한, RL 기반 모델의 일반화 능력을 평가하기 위해 디자인된 다면적 진단 프레임워크를 제시합니다.

- **Technical Details**: 논문의 주요 측면 중 하나는 OPG 메트릭을 사용하여 RL 모델 성능을 정량화하는 것입니다. 이 메트릭은 'oracle' 모델(테스트 세트에서 직접 학습한 모델)과 'standard' 모델(훈련 세트에서 학습한 모델) 간 성능 차이를 측정합니다. OPG의 값이 낮을 경우, 해당 벤치마크가 일반화 가능성을 충분히 측정하지 못한다는 것을 나타냅니다.

- **Performance Highlights**: RL 모델들이 다양한 벤치마크에서 비슷한 성능을 보이는 경향이 있으며, 이는 일반화 능력이 뛰어난 것처럼 보일 수도 있지만 사실상 신뢰할 수 없는 결과입니다. 이 연구에서는 RL 모델이 출력한 높은 벤치마크 점수가 진정한 능력을 반영하지 않을 수 있음을 입증했습니다. 마지막으로, 효율적인 벤치마크 설계를 위한 세 가지 기본 원칙을 제시하여, RL 기반 모델의 실제 추론 능력을 보다 철저하게 평가할 수 있는 방안을 모색합니다.



### Reinforced Domain Selection for Continuous Domain Adaptation (https://arxiv.org/abs/2510.10530)
- **What's New**: 이번 연구는 Continuous Domain Adaptation (CDA) 분야에서 중간 도메인을 동적으로 선택하기 위한 새로운 프레임워크를 제안합니다. 기존의 방법들이 메타데이터 없이 중간 도메인을 선택하는 데 한계를 가진 반면, 본 연구는 강화 학습과 특징 분리(feature disentanglement)를 결합하여 최적의 전이 경로를 찾는 과정을 간소화합니다. 특히, 레이블 없는 설정에서 작동하는 새로운 보상 메커니즘을 도입하여, 잠재적 도메인 임베딩 간의 거리를 이용한 최적 경로 탐색을 지원합니다.

- **Technical Details**: 제안된 방법은 여러 개의 레이블이 없는 보조 도메인과 레이블이 있는 소스 도메인을 포함하는 CDA 설정에서 작동합니다. 연구의 중심에는 도메인 인덱스를 정의하고 이를 통해 전이 경로를 도출해내는 것이 있으며, 이 경로는 도메인-불변(features) 및 도메인-특정(domain-specific) 특성을 분리하는 방식을 사용합니다. 이를 위해 특징 추출기(feature extractor)를 통해 공통 특성을 추출하고, 비슷한 정보의 각 도메인 구성 요소를 독립적으로 처리하는 두 개의 네트워크를 활용합니다.

- **Performance Highlights**: 광범위한 경험적 평가를 통해 Rotated MNIST 및 ADNI와 같은 데이터셋에서 예측 정확도 및 도메인 선택 효율성의 현저한 향상을 입증했습니다. 본 연구의 방법이 전통적인 CDA 접근 방식에 비해 수치적으로 우수함을 보여주는 결과를 도출하였습니다. 강화 학습 기반의 동적 도메인 선택 전략은 전이 경로 및 예측 결과의 동시 최적화를 가능하게 하여 도메인 적응의 효과성을 크게 향상시킵니다.



### A Hybrid Machine Learning Approach for Synthetic Data Generation with Post Hoc Calibration for Clinical Tabular Datasets (https://arxiv.org/abs/2510.10513)
- **What's New**: 본 논문에서는 의료 데이터 부족 및 개인정보 보호 규제로 인해 AI 모델 개발에 어려움을 겪고 있는 현 상황을 소개하며, 환자의 프라이버시를 보호하는 동시에 실제 데이터의 통계적 특성을 모사하는 인공 데이터 생성 기술을 제안합니다. 새로운 하이브리드 프레임워크는 노이즈 주입, interpolation, Gaussian Mixture Model (GMM) 샘플링, Conditional Variational Autoencoder (CVAE) 샘플링, SMOTE의 5가지 방법을 통합하여 고품질 의료 데이터를 합성할 수 있는 혁신적인 방법론을 제공합니다.

- **Technical Details**: 이 연구는 다양한 데이터 증강 기법을 통합한 하이브리드 모델을 통해 고충실도의 합성 데이터를 생성하는 프레임워크를 제시합니다. 이를 통해 희소한 실제 데이터와 관련된 문제를 해결하고, 통계적 분포의 다양성을 포착하는 데 필요한 여러 접근 방식의 장점을 결합합니다. 연구에서 사용된 우유, 심장병 데이터셋은 UCI 머신러닝 저장소 및 Khulna Medical College에서 수집된 데이터를 포함하며, 데이터의 품질 확보를 위한 전처리 과정이 포함됩니다.

- **Performance Highlights**: 본 프레임워크는 다양성과 고충실도를 기반으로 특히 의료 분야에서 실제 데이터를 사용할 수 없는 상황에서도 훌륭한 결과를 보여줍니다. 결과적으로, Wasserstein 거리와 Kolmogorov-Smirnov 통계량에서 거의 0에 가까운 유사성을 나타내며, Downstream classifiers는 최대 94%의 정확도와 93% 이상의 F1 스코어를 달성하여 실제 데이터로 훈련된 모델과 비슷한 성능을 보입니다. 이는 향후 민감한 AI 응용 프로그램에 대한 새로운 기준을 설정하는 데 기여할 것입니다.



### f-INE: A Hypothesis Testing Framework for Estimating Influence under Training Randomness (https://arxiv.org/abs/2510.10510)
- **What's New**: 본 논문에서는 머신러닝의 불확실성과 훈련 랜덤성으로 인한 기존의 영향 추정 방법의 한계를 극복하고자 새로운 프레임워크인 'f-influence'를 소개합니다. 이 방법은 가설 검정(hypothesis testing) 기반으로 훈련 랜덤성을 고려하여 신뢰할 수 있는 영향 추정을 가능하게 합니다.

- **Technical Details**: f-influence는 데이터 삭제/유지를 결정할 때 개별 샘플의 영향을 더욱 정확하게 추정할 수 있도록 설계되었습니다. 본 연구에서는 f-influence 계산을 단일 훈련 실행으로 수행할 수 있는 효율적인 알고리즘 f-INE을 제안하였으며, Llama-3.1-8B 모델을 사용한 데이터 정화를 통한 예제를 보여줍니다.

- **Performance Highlights**: 실험 결과, f-INE 알고리즘은 긍정적이지 않은 샘플을 신뢰성 있게 감지할 수 있으며, 이는 데이터 정화(data cleanup) 및 모델 행동(attribute model behavior) 분석에서 유용성을 입증합니다. 이 방법은 기존의 영향 추정 방법들에 비해 훈련 과정에서의 안정성을 크게 향상시키는 것으로 평가됩니다.



### Align2Act: Instruction-Tuned Models for Human-Aligned Autonomous Driving (https://arxiv.org/abs/2510.10503)
- **What's New**: Align2Act는 기계가 인간의 행동에 맞는 해석 가능한 동작 계획을 생성할 수 있도록 하기 위해 설계된 새로운 프레임워크입니다. 기존의 방법들이 대개 미리 정의된 규칙이나 드라이빙 데이터에서 학습된 경로를 사용하는 반면, Align2Act는 구조적인 운전 지침을 통해 더 효율적인 계획을 가능하게 합니다. 이 방법은 LLaMA-2-7B 모델을 LoRA를 통해 세밀하게 조정하여, 다양한 시나리오에서 높은 성능을 보이고 있습니다.

- **Technical Details**: Align2Act는 텍스트 기반 입력을 사용해 차량의 상황 및 계획 목표를 설명함으로써, 대형 언어모델을 통해 최종 운전 경로和 해당 합리적 단계를 생성합니다. 계획 과정은 높은 차원의 조작을 더 쉽게 해석하도록 구분되는 몇 가지 단계로 나누어 설명하며, 이는 사람의 사고 방식과 유사합니다. 이를 통해 모형은 물리적 상태와 의미적 지침을 기반으로 동작할 수 있게 됩니다.

- **Performance Highlights**: Align2Act는 nuPlan 데이터셋을 사용해 100만 개 시나리오에서 미세 조정한 결과, 열린 루프 점수 85.17 및 고척 루프 점수 70.31(비반응형)과 66.96(반응형)을 기록하였습니다. 이 방식은 실 세계의 주행 환경에서도 개선된 계획 품질과 인간과 유사한 동작을 보이며, 기존 LLM 계획자 대비 성능이 크게 향상되었습니다.



### Gradient Enhanced Self-Training Physics-Informed Neural Network (gST-PINN) for Solving Nonlinear Partial Differential Equations (https://arxiv.org/abs/2510.10483)
- **What's New**: 본 논문에서는 전통적인 PINN (Physics-Informed Neural Network)의 한계점을 극복하기 위해 기울기를 기반으로 한 자기 학습 알고리즘을 도입한 Gradient Enhanced Self-Training PINN (gST-PINN) 방법을 제안합니다. 이 방법은 다양한 PDE 문제를 효율적으로 해결하기 위해 설계되었습니다. gST-PINN은 라벨이 없는 데이터가 부족한 상황에서도 개선된 정확도를 보여주며, 기존 PINN 방식보다 우수한 결과를 도출합니다.

- **Technical Details**: gST-PINN 모델은 PDE의 잔여 기울기 정보를 활용하여 효율적으로 pseudo 포인트를 생성하고, 이를 통해 라벨이 없는 샘플에 대해 신뢰 점수를 기반으로 가짜 라벨을 부여하여 학습하게 됩니다. 이 과정은 semi-supervised learning의 일환으로, 실제 해결해야 할 문제와 대조하여 최적의 학습을 가능하게 합니다. 이 모델은 18,500회의 반복 후 MSE (Mean Square Error)에서 10^{-5}에 도달하며, 높은 일반화 성능을 자랑합니다.

- **Performance Highlights**: 실험 결과, Burgers' 방정식 해결 시 평균 제곱 오차(MSE)가 10^{-3} 수준으로 나오며, diffusion-sorption 방정식의 경우 12,500회 반복 후 MSE가 10^{-4}로 개선되었습니다. 그러나 gST-PINN 모델은 MSE를 지속적으로 감소시켰으며, 18,500 반복 후 MSE 10^{-5}로 향상되었습니다. 제안한 모델은 모든 실험 사례에서 기존의 표준 PINN 방법에 비해 일관되게 우수한 성과를 거두었습니다.



### Latent Retrieval Augmented Generation of Cross-Domain Protein Binders (https://arxiv.org/abs/2510.10480)
- **What's New**: RADiAnce는 기존 인터페이스를 활용하여 새로운 단백질 결합체를 디자인하는 새로운 프레임워크로, 여러 메트릭에서 기초 모델들보다 월등한 성능을 발휘하고 있습니다. 특히, 이 모델은 결합부위에 대한 조건부 잠재 확산 생성기를 통해 다양한 도메인 간 인터페이스 전송을 가능하게 합니다. 이로 인해 약물 발견 분야에서의 새로운 가능성을 열어준다는 점이 돋보입니다.

- **Technical Details**: RADiAnce는 대조적 잠재 공간에서 검색(retrieval)과 생성을 통합하는 방식으로 작동합니다. all-atom 변분 오토인코더(VAE)가 보고서의 상호작용 정렬이 가능한 잠재 공간을 생성하며, 검색된 인터페이스 임베딩들을 통해 생성을 지도합니다. 이러한 방법은 상호작용의 공유를 효과적으로 포착할 수 있는 유사성 메트릭을 필요로 하며, cross-attention 및 잔여 MLP를 사용하여 기존 지식을 통합합니다.

- **Performance Highlights**: 실험 결과 RADiAnce는 펩타이드 및 항체 디자인 작업에서 기존 강력한 기준 모델들에 비해 구조 및 상호작용 패턴을 회복하는 데 있어 유의미한 개선을 보여주었습니다. 또한, 항체 및 펩타이드와 같은 다양한 도메인에서 인터페이스를 검색함으로써, 교차 도메인 전이의 합리성을 입증하고 있습니다. 이로 인해 생성 성능이 향상된다는 사실이 강조되고 있습니다.



### Anchor-based Maximum Discrepancy for Relative Similarity Testing (https://arxiv.org/abs/2510.10477)
- **What's New**: 본 논문은 상대 유사성 테스트(relative similarity testing)의 새로운 접근 방식을 제안합니다. 기존의 방법들은 고정된 커널(kernel)을 사용하여 수동으로 지정된 대체 가설을 기준으로 테스트를 수행하는데, 이는 커널 선택에 어려움을 초래합니다. 이 문제를 해결하기 위해, 본 연구에서는 가설과 커널을 동시에 학습하는 새로운 방법론을 도입합니다.

- **Technical Details**: 우리는 최대 불일치(maximum discrepancy)를 정의하는 앵커 기반 최대 불일치(Anchor-based Maximum Discrepancy, AMD) 접근 방식을 사용하여 상대 유사성을 측정합니다. AMD는 딥 커널(deep kernel) 공간에서 거리(U와 P, U와 Q 간의 거리)의 최대 불일치로 정의됩니다. 이 테스트 과정은 두 단계로 나뉘며, 첫 번째 단계는 딥 커널 공간에서 AMD를 추정하고 잠재적 가설을 추론하는 것입니다.

- **Performance Highlights**: 본 논문의 방법론은 이론적으로 검증되었으며 다양한 벤치마크 데이터셋을 활용한 실험을 통해 그 효과성을 입증하였습니다. 실험 결과는 제안된 방법이 기존의 커널 기반 테스트 방법들보다 높은 성능을 보여주며, 사용자에게 간편한 코드 공개를 통해 실질적인 적용 가능성을 제공합니다.



### AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs (https://arxiv.org/abs/2510.10467)
- **What's New**: 이 논문에서는 AnyBCQ라는 하드웨어 친화적인 다중 정밀도(mult-precision) 양자화 프레임워크를 소개합니다. 이는 Binary-Coded Quantization (BCQ)의 확장을 통해 다중 정밀도 작업을 실질적으로 지원하면서도 비트-플레인(bit-plane) 수준에서 직접 연산을 수행할 수 있습니다. 또한, AnyBCQ는 메모리와 지연 시간이라는 제약을 극복할 수 있는 유연성을 제공합니다.

- **Technical Details**: AnyBCQ는 가중치를 이진 비트-플레인으로 표현하고 각 비트-플레인에 대해 대응하는 스케일 팩터(scale factor)를 할당하여 구성됩니다. 이 구조는 하드웨어 가속기를 통해 효율적으로 매핑될 수 있으며, 계산 효율성을 높입니다. 또한 AnyBCQ는 비트-플레인을 기반으로 한 인코딩을 지원하며, 추가 비트를 활성화할 때마다 정확도를 점진적으로 개선할 수 있는 메커니즘을 포함하고 있습니다.

- **Performance Highlights**: 실험 결과 AnyBCQ는 저 정밀도 단계에서의 정확도 감소를 크게 줄이며(예: 2-bit), 높은 정밀도에서 경쟁력을 유지합니다. 또한 AnyBCQ는 반 정밀도와 비즈니스 최적화의 최신 방법들보다 최대 3.0배의 처리량 향상을 달성하였습니다. 이로써 다양한 서비스 레벨 목표를 충족하는 데 있어 매우 실제적인 기초를 제공합니다.



### LightSAE: Parameter-Efficient and Heterogeneity-Aware Embedding for IoT Multivariate Time Series Forecasting (https://arxiv.org/abs/2510.10465)
Comments:
          Submitted to IEEE IoT-J

- **What's New**: 본 연구에서는 Shared-Auxiliary Embedding (SAE) 프레임워크를 소개하여, 다변량 시계열 예측(MTSF)의 정확성을 높이기 위한 새로운 접근법을 제시합니다. 기존의 방법들이 모든 채널에 동일한 임베딩 레이어를 적용하면서 중요한 채널 특성을 고려하지 못하는 문제를 해결하고자 하는 것입니다. SAE 구조는 공통 패턴을 포착하는 공유 기본 구성 요소와 각 채널의 고유한 변화를 모델링하는 보조 구성 요소로 분해됩니다.

- **Technical Details**: LightSAE는 저계수(low-rank) 분해와 공유 게이트 구성 요소 풀을 통해 파라미터 효율적인 임베딩 모듈을 설계합니다. 이는 채널 특정 특성을 효과적으로 모델링하면서도 파라미터 수를 최소화합니다. SAE의 분석을 통해 보조 구성 요소가 저계수 및 클러스터링 특성을 나타내는 경향이 있음을 관찰하였으며, LightSAE는 이러한 구조적 패턴을 활용하여 성능을 극대화합니다.

- **Performance Highlights**: 실험 결과, 9개의 IoT 관련 데이터셋과 4개의 백본 아키텍처를 통해 LightSAE는 기존 방법에 비해 최대 22.8%의 MSE 개선을 달성하며, 파라미터 수는 단 4% 증가에 그쳤습니다. 이와 같은 성과는 다변량 시계열 데이터의 채널 이질성을 효과적으로 처리하는 데 있어 LightSAE의 효율성을 입증합니다.



### Data-driven simulator of multi-animal behavior with unknown dynamics via offline and online reinforcement learning (https://arxiv.org/abs/2510.10451)
Comments:
          21 pages, 7 figures

- **What's New**: 이번 연구에서는 다중 동물 행동을 위한 새로운 데이터 기반 시뮬레이터인 AnimaRL을 도입했습니다. 이 시뮬레이터는 딥 강화 학습(deep reinforcement learning)과 반사실적 시뮬레이션(counterfactual simulation)에 기반하여 다중 동물의 이동 동역학을 추정하는 혁신적인 접근 방식을 사용합니다. 이러한 방법을 통해 실제 생물학적 환경에서 관찰되는 복잡한 동물 행동을 정밀하게 시뮬레이션할 수 있는 가능성을 제시합니다.

- **Technical Details**: AnimaRL은 이동 매개변수 추정, 오프라인 정책 학습(offline policy learning), 온라인 정책 조정(online policy adjustment), 시뮬레이션 환경 인터페이스와 같은 여러 핵심 모듈로 구성되어 있습니다. 이 프레임워크는 실제 동물 행동 데이터(trajectory 및 reward)를 입력으로 받아 딥 Q 네트워크(Deep Q-Network)와 거리 기반 의사 보상(pseudo-reward)을 적용하여 강화 학습 알고리즘에 적합성을 높입니다. 또한 강화 학습 프레임워크 내에서 이동 변수를 행동으로 추정하여 문제를 해결합니다.

- **Performance Highlights**: AnimaRL은 다양한 생물체에 대해 높은 재현성을 달성했습니다. 기존의 모방 기반(imitation) 및 강화 학습(RL) 기술과 비교했을 때, 종 특유의 행동을 보다 잘 재현하고 보상 획득(reward acquisition)을 개선했습니다. 또한 이 시뮬레이터는 다양한 실험 설정에서 반사실적 행동 예측(counterfactual behavior prediction)이 가능하게 하여, 다중 개체 모델링을 지원하고 유연한 경로 생성(trajectory generation)의 잠재력을 보여주었습니다.



### Reverse Supervision at Scale: Exponential Search Meets the Economics of Annotation (https://arxiv.org/abs/2510.10446)
Comments:
          10 pages

- **What's New**: 이번 연구는 라벨이 없는 대규모 데이터셋(B)의 라벨링을 탐색하여 소규모 라벨이 있는 데이터셋(A)에서 오류를 최소화하는 역감독(Reverse Supervision) 전략을 분석합니다. 이는 라벨의 품질과 주제에 대한 명확한 목표가 필요함을 강조합니다. 생성적 AI로부터 생성된 라벨은 일부 대체 가능하지만, 여전히 초기 인간의 개입이 필요하다는 점을 지적합니다.

- **Technical Details**: 연구는 감독(labeled data)과 비용(cost) 중심의 관점을 전환하여, 데이터셋 크기보다 데이터셋 비용에 중점을 두었습니다. 이 과정은 '줄이기(Reduce)', '재사용(Reuse)', '재활용(Recycle)'을 포함한 세 가지 부분으로 구성된 실용적인 청사진을 제시합니다. 반감독(semi-supervised learning), 전이 학습(transfer learning), 약한 감독(weak supervision)을 활용하여 전체 훈련 데이터의 효과적인 비용을 줄이는 방법을 정립합니다.

- **Performance Highlights**: 연구는 고품질 라벨 세트를 시작으로, SSL 및 능동적 선택을 통해 비용을 줄이고, 관련 백본에서 전이를 통해 재사용하며, 인간의 감독하에 약한 신호 및 합성 예제를 재활용하는 순환적인 파이프라인을 제안합니다. 이를 통해 실질적으로 더 적은 비용으로 더 나은 정확도를 유지할 수 있는 기회를 제공합니다. 결과적으로 데이터 포인트 수가 줄어드는 것뿐만 아니라, 비용이 많이 드는 데이터의 양도 줄입니다.



### Multi-Task Learning with Feature-Similarity Laplacian Graphs for Predicting Alzheimer's Disease Progression (https://arxiv.org/abs/2510.10433)
- **What's New**: 이번 연구에서는 Alzheimer’s Disease(AD) 데이터의 시간 변화(temporal) 특성을 효과적으로 모델링하기 위해 Feature Similarity Laplacian 그래프를 활용한 새로운 Multi-Task Learning (MTL) 프레임워크인 MTL-FSL을 제안합니다. 기존의 MTL 방법들은 특성 간의 관계를 충분히 반영하지 못했지만, MTL-FSL은 시간에 따라 변화하는 특성 간의 상관관계를 명시적으로 모델링합니다. 이를 통해 예측 정확도와 생물학적 해석 가능성을 동시에 개선할 수 있습니다.

- **Technical Details**: MTL-FSL 프레임워크는 Feature Similarity Laplacian(FSL) 패널티를 도입하여 연관된 여러 작업 간의 시간 변화하는 관계를 효율적으로 고려합니다. 또한, Alternating Direction Method of Multipliers(ADMM) 알고리즘을 사용하여 비부드러운 최적화 문제를 해결합니다. 이 접근 방식은 데이터로부터 파라미터를 효과적으로 추정하고 예측의 신뢰성을 높이는 데 기여합니다.

- **Performance Highlights**: Alzheimer’s Disease Neuroimaging Initiative(ADNI) 데이터셋을 이용한 실험에서, MTL-FSL 프레임워크는 다양한 기준 방법들보다 뛰어난 성능을 보였습니다. 이 모델은 여러 다른 방법들과 비교하여 인지 점수의 예측 정확도를 현저히 개선하였으며, 연구 결과는 생물학적 해석 가능성과 임상적 가치에서 유의미한 의미를 가집니다.



### Hierarchical LoRA MoE for Efficient CTR Model Scaling (https://arxiv.org/abs/2510.10432)
Comments:
          13 pages, 9 figures

- **What's New**: 이번 논문에서는 CTR 예측을 위한 효율적이고 확장 가능한 모델 설계를 제안합니다. 이를 위해 HiLoMoE라는 계층적 LoRA MoE 프레임워크를 도입하여, 수직적 및 수평적 확장을 모두 가능하게 합니다. 이 모델은 경량화된 rank-1 전문가를 사용하여 매개변수 효율성을 높이고, 계층적 라우팅을 통해 전문가 조합을 다양화합니다.

- **Technical Details**: HiLoMoE는 전문가 선택을 이전 레이어의 라우팅 점수를 기반으로 하여 모든 레이어를 병렬적으로 실행할 수 있게 합니다. 이 시스템은 세 가지 핵심 혁신으로 구성되며, LoRA 전문가를 통해 매개변수 감소를 이루고 계층적 라우팅 메커니즘을 통해 수직적 확장을 지원합니다. 또한 이 복잡한 시스템을 학습하기 위해 세 단계의 학습 파이프라인을 제안하고, 보조 손실을 추가하여 전문가의 다양성을 강화합니다.

- **Performance Highlights**: 네 개의 공개 데이터셋에서의 실험 결과, HiLoMoE는 비 MoE 모형에 비해 평균적으로 AUC를 0.20% 개선하고, FLOPs는 18.5% 감소하는 성능-효율성 거래를 달성했습니다. 이 모델은 깊이와 폭을 확장하는 데 있어 뛰어난 성능을 보이며, 레이어 수와 전문가 수가 증가할수록 성능이 개선되는 경향을 보입니다.



### Softmax $\geq$ Linear: Transformers may learn to classify in-context by kernel gradient descen (https://arxiv.org/abs/2510.10425)
- **What's New**: 이번 논문은 트랜스포머(transformer)가 in-context learning (ICL)을 통해 맥락(context)에서 학습하는 알고리즘을 탐구합니다. 기존의 연구는 주로 선형 회귀(task)와 간단한 문제에 초점을 맞추었지만, 본 연구는 비선형 활성화(activation)를 사용하는 분류(classification) 문제에 적용하여 그 차이를 이해합니다. 또한 softmax를 활용한 트랜스포머가 어떻게 맥락에 적응하는 학습률(context-adaptive learning rate)을 가지는지를 논의합니다.

- **Technical Details**: 연구에서는 softmax self-attention이 기능적 그래디언트 하강법(functional gradient descent) 단계를 수행한다는 것을 밝혀냈습니다. 이와 더불어, softmax attention은 특정 상황에서 더 나은 성능을 발휘하는 맥락 적응 학습률(context-adaptive learning rate)을 얻게 됩니다. 이를 통해 softmax self-attention의 이점을 확인하고, 이는 기존 선형 self-attention과 비교하여 분류(classification) 설정에서 성능 향상을 가져옵니다.

- **Performance Highlights**: 엣지 케이스를 포함한 실험을 통해 softmax attention이 선형 attention보다 더 나은 성능을 보인다는 것을 입증하였습니다. 연구진은 두 가지 효과적인 파라미터인 커널 폭(kernel width)과 학습률(learning rate)의 중요성을 강조하며, 이들이 softmax attention의 성공에 필수적이라고 결론 내렸습니다. 최종적으로 연구는 트랜스포머의 in-context learning이 실제 사용 사례와 더 밀접하게 연결될 수 있도록 해주는 이론적 기반을 제공합니다.



### Controllable Graph Generation with Diffusion Models via Inference-Time Tree Search Guidanc (https://arxiv.org/abs/2510.10402)
- **What's New**: 이 논문에서는 제어 가능한 그래프 생성을 위한 새로운 방법론인 TreeDiff를 제안합니다. TreeDiff는 몬테 카를로 트리 탐색(MCTS) 기반의 이중 공간 확산(diffusion) 체계를 도입해 샘플링 과정을 조정할 수 있는 점이 특징입니다. 이 방법은 기존의 조건 없는 확산 모델의 한계를 극복하고, 연산 효율성을 개선하여 보다 안정적이고 제어 가능한 생성 과정을 가능하게 합니다.

- **Technical Details**: TreeDiff는 세 가지 주요 설계를 포함합니다: 첫째, 매크로 단계 확장 전략(macro-step expansion strategy)을 통해 여러 개의 디노이징(denoising) 업데이트를 하나의 변환으로 그룹화하여 트리 깊이를 줄이고 긴 탐색을 가능하게 합니다. 둘째, 이중 공간 디노이징 메커니즘(dual-space denoising mechanism)은 그래프 공간에서의 가벼운 수정과 함께 효율적인 잠재 공간(latent-space) 디노이징을 결합하여 확장성과 구조적 충실성을 보장합니다. 셋째, 이중 공간 검증자(dual-space verifier)는 부분적으로 디노이징된 그래프에서 장기 보상을 예측하여 조기 가치 추정을 가능하게 합니다.

- **Performance Highlights**: TreeDiff는 2D 및 3D 분자 생성 벤치마크에서 최첨단 성능을 달성했습니다. 콘텐츠가 증가함에 따라 TreeDiff의 성능이 향상되는 반면, 기존 방법들은 연산 자원에 제한을 받을 때 성능이 머무는 현상을 보입니다. 이 결과는 TreeDiff가 더 많은 계산 자원을 활용하여 지속적으로 개선될 수 있는 잠재력을 가지고 있음을 보여줍니다.



### Applying non-negative matrix factorization with covariates to label matrix for classification (https://arxiv.org/abs/2510.10375)
Comments:
          2 figures, R package: nmfkc published in GitHub, this https URL

- **What's New**: NMF-LAB(Non-negative Matrix Factorization for Label Matrix)는 비지도 학습의 한계를 극복하고 클래스 레이블을 직접 활용할 수 있는 새로운 방법론을 제안합니다. 기존의 방법들이 레이블을 간접적으로 다루는 것과 달리, NMF-LAB는 레이블 행렬을 관찰 데이터로 간주하여 직접적으로 분해합니다. 이 방법은 예측 정확성을 향상시키고, 이전에 다루지 않았던 회귀 및 분류를 통합할 수 있는 새로운 관점을 제공합니다.

- **Technical Details**: NMF-LAB는 비음수 행렬 삼중 인수 분해(tri-NMF)의 역문제(inverse problem)로 분류 문제를 정의합니다. 이 프레임워크는 클래스 확률을 별도의 분류기 없이도 직접적으로 얻을 수 있게 해주며, 커널 기반의 공변량(covariates)을 통합하여 빈 샘플에 대한 예측을 일반화할 수 있습니다. 또한, 레이블의 잡음에 대한 강력한 저항력을 제공하며, 반지도 학습(semi-supervised learning)을 위한 유연성을 지원합니다.

- **Performance Highlights**: NMF-LAB는 다양한 데이터셋에서 경쟁력 있는 예측 정확도를 달성했으며 노이즈나 결측 레이블에 강건한 성능을 보입니다. 고차원 문제에도 잘 확장 가능성과 함께 해석 가능성을 유지하고 있습니다. 이러한 특성 덕분에 NMF-LAB는 현대의 분류 과제에 있어 새로운 확장성을 제공하는 모델로 자리 잡고 있습니다.



### Exploration-free Algorithms for Multi-group Mean Estimation (https://arxiv.org/abs/2510.10374)
- **What's New**: 본 연구는 여러 그룹의 평균 추정을 위해 유한한 샘플링 예산을 배정하는 문제를 다루고 있다. 전통적인 multi-armed bandits와 달리, 본 연구의 최적 할당은 각 그룹을 균등하게 샘플링해야 한다는 점에서 차별화된다. 탐색이 필요 없는 알고리즘이 자연스럽게 효과적이며, 본 연구는 이러한 접근을 통해 그룹 평균 추정을 수행할 수 있음을 보여준다.

- **Technical Details**: 문제를 설정하기 위해 K개의 대안을 고려하며, 각 대안은 미지의 평균과 분산을 가진 무작위 결과를 따른다. 온라인 학습 환경에서, 각 시점에 결정자는 하나의 대안을 선택하고 결과를 관찰한다. 연구진은 subgaussian 분포의 변동성 집중 관련 결과를 강화하고, 비탐색적 알고리즘을 설계하여 더 타이트한 후회를 보장하는 씬 탐구를 이룬다.

- **Performance Highlights**: 우리는 설정된 프레임워크를 사용하여 기존 연구보다 개선된 성능을 보였다. 새로 제안하는 알고리즘은 맥락적 정보를 활용하여 정확한 다중 그룹 평균 추정을 가능케 하며, 실험적 설계나 개인화와 같은 다양한 분야에 응용될 수 있다. 이러한 결과물은 그룹 평균 추정에서의 비탐색적 할당이 효과적임을 보여준다.



### Transformer Model Detects Antidepressant Use From a Single Night of Sleep, Unlocking an Adherence Biomarker (https://arxiv.org/abs/2510.10364)
- **What's New**: 이 논문은 우울증 치료제 복용 불이행(adhereance)가 광범위하게 발생하고 있다는 점에 주목합니다. 기존의 방법들은 침습적이거나(serum assays, neuroimaging) 부정확한 대리 기반 방법이기 때문에, 제안된 방법은 최초의 비침습적(biomarker) 생체 표지를 제공합니다. 이는 단 한 번의 수면 데이터를 분석하여 우울증 치료제 복용 여부를 파악할 수 있습니다.

- **Technical Details**: 이 연구에서는 transformer 기반 모델을 활용하여 소비자용 웨어러블(consumer wearable) 또는 비접촉식 무선 센서(contactless wireless sensor)로 수집된 수면 데이터를 분석합니다. 모델은 수면 중 제공된 데이터를 바탕으로 우울증 치료제 intake을 추론하여 자택에서 손쉬운 일일 복용 평가를 가능하게 합니다. 연구는 62,000개의 수면 데이터를 포함하는 6개의 데이터 세트를 사용하였으며, 20,000명 이상의 참가자들 가운데 1,800명의 우울증 치료제 사용자 데이터를 분석하였습니다.

- **Performance Highlights**: 제안된 생체 표지는 AUROC = 0.84의 성능을 달성하였으며, 약물 클래스 전반에 걸쳐 일반화되고 용량에 따라 스케일링이 가능하며, 동시에 사용되는 정신의약품에 대해서도 강건성을 유지합니다. 이 장기 모니터링은 실제 세계에서의 복용 시작, 감량, 및 불이행을 포착하였고, 객관적이고 확장 가능한 복용 모니터링을 제공합니다. 이는 우울증 치료 개선 및 결과 향상에 기여할 가능성이 있습니다.



### Multi-View Graph Learning with Graph-Tup (https://arxiv.org/abs/2510.10341)
Comments:
          Submitted to TAG workshop

- **What's New**: 이번 논문에서는 복잡한 관계형 시스템에서 다양한 스케일의 상호작용을 학습하기 위해 다중-뷰 그래프 튜플 프레임워크를 소개합니다. 기존의 그래프 신경망(GNN)이 가진 단일 그래프의 한계를 극복하기 위해, 그래프의 엣지를 상호작용 강도에 따라 분할하여 여러 하위 그래프를 생성합니다. 이를 통해 강한 연결 그래프와 약한 연결 그래프를 명시적으로 구분하여 상호작용 간의 세부사항을 포착할 수 있습니다.

- **Technical Details**: 우리의 접근법은 이질적인 메세지 패싱 아키텍처를 기반으로 하며, 이는 비교할 수 없는 연산자 이론에서 영감을 받았습니다. 각 그래프 뷰 내에서의 작업과 그래프 뷰 간의 작업을 통합하여 복잡한 관계를 모델링할 수 있는 역량이 있는 구조를 제공합니다. 또한, 우리는 이 아키텍처가 단일 그래프 모델보다 표현력이 더 뛰어나며 예측 리스크를 낮추는 것이 보장된다는 것을 증명합니다.

- **Performance Highlights**: 실험적으로, 분자 속성 예측 및 우주론적 매개변수 추정의 두 가지 과제를 통해 우리의 다중-뷰 그래프 튜플 모델이 단일 그래프 모델보다 우수한 성능을 보임을 확인했습니다. 특히, QM7b 데이터셋에서 GINE-Gt는 기존 모델들을 초월하며, CAMELS 우주론 시뮬레이션에서는 EGNN-Gt가 매우 다양한 상호작용 반경에서 뛰어난 성능을 보여주었습니다.



### Sample-Efficient Online Learning in LM Agents via Hindsight Trajectory Rewriting (https://arxiv.org/abs/2510.10304)
- **What's New**: 언어 모델(LM) 에이전트는 새로운 환경에서 상호작용을 학습할 때 샘플 효율성이 낮아지는 문제를 해결하기 위해 ECHO(Experience Consolidation via Hindsight Optimization)라는 새로운 프레임워크를 도입했습니다. ECHO는 실패한 시도에서 얻은 경험을 활용하여 대체 목표를 위한 최적화된 궤적(trajectories)을 생성함으로써, 비효율적인 학습을 개선하는 데 중점을 둡니다. 이 방법은 경험 재생(replay) 메커니즘을 사용하여 언어 모델이 과거의 실패를 성공적인 경험으로 전환할 수 있도록 돕습니다.

- **Technical Details**: ECHO 시스템은 두 가지 구성 요소로 구성됩니다: 처음으로, 언어 모델을 사용하여 관련 서브 목표(subgoals)를 식별하고 최적화된 궤적을 생성하는 회상 규칙(hindsight rule)이 있습니다. 두 번째로, 압축된 궤적 표현을 기억에 유지하는 업데이트 규칙(update rule)이 포함되어 있습니다. ECHO는 기존의 경험 재생 대신 더 많은 수정 가능성을 제공하여, 실패한 궤적을 임의로 재작성(rewriting)할 수 있게 합니다.

- **Performance Highlights**: XMiniGrid 및 PeopleJoinQA와 같은 다양한 상태 유지 가능한 테스트 환경에서 ECHO를 평가한 결과, 기존의 언어 에이전트보다 최대 80% 더 높은 성능을 달성했습니다. XMiniGrid에서 ECHO는 Reflexion 및 AWM과 같은 고급 에이전트 구조를 초과한 성능을 보여주며, 새로운 환경에 대한 적응 속도가 빨라짐을 입증하였습니다. ECHO는 특히 보상이 드문 환경에서 언어 에이전트의 샘플 효율성을 극대화하는 유망한 기술입니다.



### Simulating Viva Voce Examinations to Evaluate Clinical Reasoning in Large Language Models (https://arxiv.org/abs/2510.10278)
- **What's New**: 이 연구는 대화형 AI가 임상에서의 진단 추론을 평가하기 위한 새로운 벤치마크인 VivaBench를 소개합니다. 기존의 의료 AI 모델 평가가 단일 질의응답 방식에 의존하고 있는 반면, VivaBench는 여러 단계의 상호작용을 요구하여 AI 모델들이 더 복잡한 임상 문제를 해결할 수 있는지를 검증합니다. 이는 실제 임상 환경에서 의사들의 의사결정 과정을 모방하여 AI의 진단 추론 능력을 평가하는 데 도움을 줍니다.

- **Technical Details**: VivaBench는 1762개의 의사가 편집한 임상 시나리오로 구성되어 있으며, 각 시나리오는 상호작용적인 요소를 포함하고 있습니다. AI 에이전트는 제한된 초기 정보로부터 진단을 내리기 위해 정보 수집과 가설 검증을 반복적으로 수행해야 합니다. 이 평가 과정에서는 두 가지 단계, 즉 리뷰 단계(History, Physical Examination)와 조사 단계(Imaging, Laboratory investigations)로 나뉘어 있으며, 각 단계에서 에이전트는 적절한 진단 증거를 수집하게 됩니다.

- **Performance Highlights**: 현재 대다수의 대형 언어 모델은 잘 규명된 임상 정보에서 진단을 내리는 데는 능숙하지만, VivaBench를 통해 평가할 경우 불확실성 속에서 반복적인 진단 추론을 수행할 때 성능이 현저히 저하된다는 것을 발견했습니다. 연구 결과는 AI 모델들이 흔히 발생하는 인지적 오류를 포함하여, 초기 가설에 집착하거나 조사 순서를 부적절하게 정하는 등의 여러 가지 한계점을 드러냈습니다. 이러한 패턴은 임상 실무에서의 공통적인 오류를 반영하며, AI 시스템이 고위험 환경에서 의사결정을 수행할 때의 한계를 강조합니다.



### Lost in the Middle: An Emergent Property from Information Retrieval Demands in LLMs (https://arxiv.org/abs/2510.10276)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM)에서 정보 검색 요구에 따른 "lost-in-the-middle" 현상을 분석했습니다. 이 현상은 인간 기억에서의 primacy와 recency 효과와 유사하며, LLM이 긴 컨텍스트에서 중심부 정보의 회수 정확도가 떨어지는 이유를 탐구합니다. 연구 결과는 LLM의 성능 저하가 단순한 정보 손실이 아닌 정보를 검색하는 방식의 적응이라는 새로운 관점을 제시합니다.

- **Technical Details**: 우리는 두 가지 기본적인 인간 기억 과제를 사용하여 LLM(GPT-2와 Llama-3.2 변형)을 처음부터 훈련했습니다. 자유 회상(task)과 실행 범위(task)를 통해 장기 및 단기 메모리 요구를 유도하고, 구축에 따른 특성의 상호작용이 primacy 효과에 미치는 영향을 분석했습니다. 주목적은 모델 아키텍처의 구조적 동역학이 LLM의 위치 편향을 어떻게 생성하는지를 규명하는 것입니다.

- **Performance Highlights**: 연구 결과, primacy와 recency 효과가 각각 장기 및 단기 메모리 요구를 충족하는 최적의 전략으로 나타났습니다. 특히, 주목적 모델 아키텍처의 인과적 차별화가 이러한 행동에 중요한 역할을 하며, attention sinks가 장기 메모리 요구를 지원하는 중요한 메커니즘으로 확인되었습니다. 따라서, 'lost-in-the-middle' 현상은 정보 손실의 단순한 표시가 아니라 정보 검색 요구에 대한 최적의 적응으로 이해되어야 합니다.



### Enhancing the Cross-Size Generalization for Solving Vehicle Routing Problems via Continual Learning (https://arxiv.org/abs/2510.10262)
- **What's New**: 이 논문은 다양한 크기의 문제에 대한 운송 경로 최적화 문제(VRP)를 해결하기 위해 지속적 학습(Continual Learning) 기반의 새로운 프레임워크를 제안합니다. 기존의 딥 러닝 모델들은 단일 사이즈 데이터셋에서 훈련되고 평가되어, 다른 크기에 대한 일반화 능력이 제한되어 있습니다. 제안된 프레임워크는 작은 문제 크기에서 학습한 지식을 큰 문제 크기에 전이하면서, 새로운 크기에서도 안정적인 성능을 발휘할 수 있도록 설계되었습니다.

- **Technical Details**: 제안된 모델은 두 가지 정규화 기법을 사용하여 훈련 과정에서 지식을 보존하고 최근의 예제 모델을 모방합니다. Inter-task regularization은 작은 크기에서 얻은 인사이트를 큰 크기로 전이하고, intra-task regularization은 현재 크기에 대한 최신 모델을 지속적으로 유지합니다. 뿐만 아니라, 경험 리플레이(experience replay)를 통해 과거에 훈련된 크기의 인스턴스를 다시 검토하여 기억의 파괴(catasrophic forgetting)를 완화합니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 다양한 크기의 문제(훈련 시에 본 크기와 보지 않은 크기)를 대상으로 한 경우, 기존의 딥 러닝 모델보다 전반적으로 우수한 성능을 보였습니다. 특히, 제안된 프레임워크는 기존의 일반화 성능 향상에 특화된 최첨단 방법들과 비교했을 때도 뛰어난 결과를 기록하였습니다. 이로 인해, 실제 VRP 문제 해결에 있어 적용 가능성이 크게 높아졌음을 보여줍니다.



### Reasoning-Enhanced Large Language Models for Molecular Property Prediction (https://arxiv.org/abs/2510.10248)
- **What's New**: MPPReasoner는 화학적 이유(chemical reasoning) 기능을 통합하여 분자 특성 예측(molecular property prediction)을 향상시키는 새로운 멀티모달 대형 언어 모델(multimodal large language model)입니다. 이 모델은 Qwen2.5-VL-7B-Instruct를 기반으로 하여, 분자 이미지(molecular images)와 SMILES 문자열을 통합하여 보다 포괄적인 분자 이해를 가능하게 합니다. 기존 접근 방식에서 나타나는 해석력 부족과 화학적 추론의 부재를 해결하기 위해, MPPReasoner는 구조적 분석과 화학 원칙의 적용, 그리고 예측과정에서 인간이 이해할 수 있는 설명을 제공합니다.

- **Technical Details**: MPPReasoner는 두 단계의 훈련 전략을 채택합니다. 첫 번째 단계는 전문가 지식 및 다양한 교사 모델을 통해 생성된 16,000개의 고품질 추론 경로를 활용한 감독 세부 조정(Supervised Fine-Tuning, SFT)입니다. 두 번째 단계인 원칙 유도 보상(Reinforcement Learning from Principle-Guided Rewards, RLPGR)은 화학적 원칙의 적용, 분자 구조 분석 및 논리적 일관성을 평가하는 검증 가능하고 규칙 기반의 보상을 활용합니다. 이 방법은 전통적인 강화 학습 접근법과 달리, 화학적 추론을 계층 화된 보상 요소로 분해하여 정량적으로 평가하는 방식입니다.

- **Performance Highlights**: MPPReasoner는 8개 다양한 데이터셋에서 성능이 크게 개선되었으며, In-distribution(ID) 작업에서는 평균 ROC-AUC 점수 0.8068을, Out-of-distribution(OOD) 작업에서는 0.7801을 달성했습니다. 이는 기존 최상의 기준선 모델보다 각각 7.91% 및 4.53%의 성능 향상을 나타냅니다. 이 모델은 특히 OOD 데이터셋에서 뛰어난 일반화 능력을 보였으며, 전문가 평가 및 사례 연구를 통해 화학적 이유를 제공하며 분자 특성 간의 관계에 대한 귀중한 통찰을 제시합니다.



### Progressive Scale Convolutional Network for Spatio-Temporal Downscaling of Soil Moisture: A Case Study Over the Tibetan Plateau (https://arxiv.org/abs/2510.10244)
- **What's New**: 본 연구에서는 고해상도 토양 수분(Soil Moisture, SM)을 얻기 위해 저해상도 SMAP SM 제품의 다운스케일링 과정에 고주기 변수인 ERA5-Land를 도입했습니다. 이후, 다주파수 시간 융합 모듈(Multi-frequency Temporal Fusion Module, MFTF)과 맞춤형 스퀴즈-자극(Squeeze-and-Excitation) 블록을 중심으로 한 점진적 스케일 합성곱 신경망(Progressive Scale Convolutional Network, PSCNet)을 설계했습니다. 이 접근 방식을 통해 2016년부터 2018년까지의 티베트 고원의 원활한 SM 제품을 10km 공간 해상도와 3시간 시간 해상도로 생성했습니다.

- **Technical Details**: PSCNet은 두 가지 혁신적인 구성 요소, 즉 시간 동적 변화를 포착하기 위한 MFTF와 미세 공간 세부정보를 유지하는 SE 블록을 사용하여 구성됩니다. 또한, 분석 과정에서 위성 제품에 대한 검증 결과, PSCNet이 평균 R 값 0.881로 다른 방법보다 높은 정확도와 낮은 오류를 보였으며, 모든 현장 검증에서 상위 세 모델 중 하나로 지속적으로 순위를 기록했습니다. 시간 일반화 검증에서는 ERA5-Land 변수를 활용한 다운스케일링의 가능성이 입증되어, 모든 방법이 평균 상대 오류를 R 메트릭 6% 이하, ubRMSE 메트릭 2% 이하로 유지했습니다.

- **Performance Highlights**: PSCNet의 성능은 시간 동적 감도와 생생한 공간 세부사항을 보여주며, 시간 동적 및 시각화 검증에서 뛰어난 성능을 발휘했습니다. 또한 SSCNet은 SM 데이터의 복잡한 시공간 관계를 효과적으로 모델링하여 시공간 다운스케일링에 있어 유망한 해결책을 제공합니다. 실제 실험 결과는 PSCNet이 다른 방법들에 비해 일관된 예측 결과를 제공함을 나타내며, 이는 SM의 강한 공간 자가상관성과 시간 지속성을 고려한 것에서 기인합니다.



### SGM: A Statistical Godel Machine for Risk-Controlled Recursive Self-Modification (https://arxiv.org/abs/2510.10232)
- **What's New**: 이 논문에서는 AutoML 및 적응형 최적화 분야에서의 재귀적 자기 수정의 안전성을 보장하는 통계적 안전 계층인 Statistical Gödel Machine (SGM)을 소개하고 있습니다. 기존의 Gödel 머신이 제공하는 논리적 증명 대신에, SGM은 통계적 신뢰성 테스트(e-values, Hoeffding bounds)를 적용하여 수정이 이루어질 때만 승인합니다. 이는 높은 차원의 확률적 환경에서도 안전한 수정이 가능하도록 설계되었습니다.

- **Technical Details**: SGM은 수정 요구 시 통계적 인증을 기반으로 하여 수정된 제안이 선택한 신뢰 수준에서 우수성을 인증할 때만 허가됩니다. 이를 위해 SGM은 전역 오류 예산을 할당하여 위험을 관리하며, 이는 여러 라운드에서 지속적으로 안전성을 확보합니다. 이러한 방법론은 표준 연속 테스트나 온라인 잘못 발견률(FDR) 방법과의 차별성을 가지며, 실제로 SGM은 각 수용된 수정이 기존 모델을 영구적으로 수정하는 것을 보장합니다.

- **Performance Highlights**: SGM은 CIFAR-100 데이터셋에서 30 Seed 스트레스 테스트 중 실제 +5.5pp의 성장을 인증하였으며, ImageNet-100에서는 확인을 실패한 유망한 수정을 올바르게 거부했습니다. 이러한 결과들은 SGM이 자기 개선 ML 파이프라인을 위한 재사용 가능한 위험 관리 계층으로 기능할 수 있는 가능성을 보여줍니다. 다양한 학습 기법을 통해 SGM의 성능과 유효성을 검증하였으며, 진정한 이익을 인증하는 동시에 허위 개선을 거부함으로써 ML의 안정성을 크게 향상시킵니다.



### Hierarchical Bayesian Flow Networks for Molecular Graph Generation (https://arxiv.org/abs/2510.10211)
- **What's New**: 이번 연구에서는 분자 그래프 생성 문제를 해결하기 위해 GraphBFN이라는 새로운 계층적 프레임워크를 제안합니다. GraphBFN은 Bayesian Flow Networks를 기반으로 하여 디스크리트(Discrete) 데이터 생성을 더 효과적으로 수행할 수 있도록 설계되었습니다. 이 방법은 기존 모델들이 직면한 한계를 극복하면서도 더 빠르고 효율적인 분자 생성이 가능하도록 Cumulative Distribution Function을 도입합니다.

- **Technical Details**: GraphBFN은 디스크리트 원자 및 결합 특성을 1차원 연속 공간으로 매핑하여, 고속의 수렴을 가능하게 합니다. 이 모델은 전체 데이터를 포함하는 확률 분포 매개변수를 활용하여 각 카테고리에 대한 확률을 계산하며, 이는 과적합 문제를 피하고 학습 목표를 샘플링 라운딩 작업과 통합합니다. 또한, GraphBFN은 다중 스케일 그래프 표현을 구축하기 위해 계층적 코스-투-파인(Coarse-to-Fine) 구조를 통합합니다.

- **Performance Highlights**: GraphBFN은 QM9 및 ZINC250k 분자 그래프 생성 벤치마크에서 최첨단 성능을 기록하며, 최소한의 샘플링 단계로 최상의 결과를 제공합니다. 기존 방법에 비해 학습 및 샘플링 과정이 크게 빨라져, 낮은 비용으로도 효율적인 약물 발견 과정을 촉진할 수 있습니다. 이를 통해 분자 다양성을 증대시키고 모델의 일반화 능력을 향상시키는 데 기여합니다.



### RLFR: Extending Reinforcement Learning for LLMs with Flow Environmen (https://arxiv.org/abs/2510.10201)
Comments:
          Project Website: this https URL

- **What's New**: 이 논문에서는 Verifiable Rewards (RLVR) 기반의 강화 학습 프레임워크를 개선하기 위해 새로운 방식인 RLFR(Flow rewards)을 제안합니다. 특히, LL(M)s의 성장하는 잠재 공간을 활용해 보상 신호를 더욱 탄력적으로 만들 수 있는 방법을 탐구합니다. RLFR은 유망한 보상 신호 수집을 위한 환경으로 흐름 필드를 구성하고, 임상적인 데이터 및 모델의 고품질 데이터를 활용해 정책 탐색을 권장합니다.

- **Technical Details**: RLFR(Flow rewards)은 latent space에서 파생된 흐름 보상(shaping rewards)을 기반으로 하며, 정책의 속도 편차(velocity deviations)를 통해 보상 신호를 측정합니다. 이 방법은 오프-정책(high-quality data)과 온-정책(rejection sampling) 데이터를 함께 사용하여 흐름 필드를 구축하는 방식을 포함합니다. 이러한 흐름 필드는 정책 최적화와 함께 온라인으로 업데이트되며, 연구에 활용 가능한 모든 코드, 데이터 및 모델 가중치를 공개합니다.

- **Performance Highlights**: 언어 및 다중 모달(multi-modal) 추론 벤치마크에서 RLFR의 유효성을 검증하였으며, 기존 RLVR 및 기타 보상 기본 방법에 비해 일관된 성과 향상을 보였습니다. RLFR은 모델의 숨겨진 상태 내에서 효율적인 문맥 의존성을 활용하여 다양한 데이터 집합의 정당성을 보장합니다. 이러한 결과는 RLVR 프레임워크를 활용한 보상 설계에서 새로운 가능성을 시사합니다.



### CauchyNet: Compact and Data-Efficient Learning using Holomorphic Activation Functions (https://arxiv.org/abs/2510.10195)
- **What's New**: 이 논문에서는 Cauchy의 적분 공식을 바탕으로 하는 새로운 신경망인 CauchyNet을 제안합니다. CauchyNet은 실수 데이터를 복소 평면에 임베딩하며, 시간에 따른 복잡한 의존성을 효율적으로 캡처하여 기존의 실수 기반 모델을 초월합니다. 이 아키텍처는 불완전한 데이터로부터 강력한 학습을 가능하게 하며, 효율적인 매개변수 사용과 계산 오버헤드 감소를 특징으로 합니다.

- **Technical Details**: CauchyNet의 설계는 Cauchy의 적분 공식과 보편적 근사 정리에 근거하고 있으며, 복소수 활성화 함수가 포함되어 있습니다. 이를 통해 CauchyNet은 기하급수적이고 점근적인 변화에 대한 민감도를 줄이고, 저차원 환경에서도 효율적인 함수 근사를 가능하게 합니다. 이 네트워크는 Wirtinger 미분을 사용하여 부분 데이터나 다양한 입력 스케일에서 안정적인 그래디언트 계산을 보장합니다.

- **Performance Highlights**: CauchyNet은 교통, 에너지 소비 및 전염병 데이터와 같은 다양한 분야에서 광범위한 실험을 통해 최첨단 모델들보다 예측 정확도에서 일관되게 우수한 성능을 보였습니다. 연구 결과는 CauchyNet이 데이터를 기반으로 한 예측 모델링에 있어 강력하고 효율적인 도구가 될 수 있음을 보여줍니다.



### INR-Bench: A Unified Benchmark for Implicit Neural Representations in Multi-Domain Regression and Reconstruction (https://arxiv.org/abs/2510.10188)
- **What's New**: 본 논문에서는 Implicit Neural Representations (INRs)의 효과성 및 한계에 대한 깊은 통찰을 제공하고, Neural Tangent Kernel (NTK) 이론을 활용하여 다양한 모델 아키텍처, 포지셔널 인코딩 및 비선형 프리미티브가 신호의 주파수 응답에 미치는 영향을 분석합니다. 연구의 일환으로 다중 모달 INR 작업을 위한 최초의 포괄적인 벤치마크인 INR-Bench를 소개하였으며, 이는 56개의 좌표 MLP 모델과 22개의 좌표 KAN 모델로 구성되어 있습니다. 이 벤치마크를 통해 다양한 신경 모델의 강점과 한계를 강조하고, 향후 연구를 위한 강력한 기반을 마련하였습니다.

- **Technical Details**: 이 연구는 KAN과 MLP 모델 아키텍처의 NTK 스펙트럼 분포에 미치는 영향을 분석하였습니다. KAN은 낮은 주파수 학습에서 MLP보다 작은 스펙트럼 편향을 보이나, 높은 계산 복잡성 문제를 겪습니다. FKAN은 KAN을 위한 풀 수학적으로 학습 가능한 포지셔널 인코딩으로, 새로운 주파수 구성 요소를 신호에서 학습할 수 있도록 돕습니다. 또한, ReLU, Sine, Gaussian 등의 비선형 프리미티브가 NTK 스펙트럼 분포에 미치는 영향을 분석하여, 각 활성화 함수의 장단점을 도출하였습니다.

- **Performance Highlights**: INR-Bench는 56개의 좌표 MLP와 22개의 KAN 모델을 포함하여 9개의 다중 모달 작업에서 성능을 평가합니다. 이 데이터셋은 다양한 주파수 학습, 역 추론 및 일반화 능력을 효과적으로 평가할 수 있는 구조를 갖추고 있습니다. 특히, FKAN을 포함한 포지셔널 인코딩과 여러 활성화 함수가 신경망의 표현 능력에 미치는 영향을 심층적으로 탐구함으로써, 신경 모델의 향후 발전을 위한 방향성을 제시하고 있습니다.



### Rethinking Entropy Interventions in RLVR: An Entropy Change Perspectiv (https://arxiv.org/abs/2510.10150)
- **What's New**: 이번 논문에서는 Reinforcement Learning with Verifiable Rewards (RLVR)에서 발생하는 entropy collapse 문제를 다룬다. 기존의 entropy intervention 방법들이 간접적으로만 효과를 가져오는 한계가 있음에 주목하며, 이를 보완하기 위해 Stabilizing Token-level Entropy-changE via Reweighting (STEER)라는 새로운 방법을 소개한다. STEER는 토큰 수준에서의 조정을 통해 정책의 엔트로피를 구체적으로 안정화하는 것을 목표로 한다.

- **Technical Details**: RLVR이 가지는 탐색-착취의 불균형으로 인해 발생하는 entropy collapse 문제를 해결하기 위해, 기존 방법들은 간접적으로 엔트로피 다이나믹스를 조절하고 있었다. 하지만 이러한 접근법들은 대부분의 경우 엔트로피 변화를 직접 제어하지 못하는 한계를 갖는다. 새로운 방법인 STEER는 각 토큰의 엔트로피 변화를 고려하며, 이를 통해 정책 엔트로피의 다이나믹스를 원하는 범위 내에서 유지할 수 있도록 설계되었다.

- **Performance Highlights**: STEER는 다양한 수학적 추론 벤치마크에서 기존 방법들보다 우수한 성능을 보이며, 엔트로피 collapse를 효과적으로 방지하고, 탐색을 강화하는 데 성공을 거두었다. 실험 결과, STEER는 정책의 엔트로피 다이나믹스를 안정적으로 유지하며, 학습의 안정성을 높이고, 최종 성능을 개선하는 데 긍정적인 영향을 미쳤다.



### Robust Learning of Diffusion Models with Extremely Noisy Conditions (https://arxiv.org/abs/2510.10149)
- **What's New**: 이번 논문에서는 조건부 확산 모델(Conditional Diffusion Models)이 높은 잡음 조건에서 성능이 급격히 저하되는 문제를 해결하기 위한 강인한 학습 프레임워크를 제시합니다. 특히, 잡음이 많은 조건에서 기존의 강인한 방법들이 실패한다는 것을 실험적으로 입증하였습니다. 이 논문에서 제안하는 방법은 깨끗한 조건을 대체할 의사 조건(Pseudo Conditions)을 학습하고, 시간적 집합(Temporal Ensembling) 기법을 통해 점진적으로 이를 정제하는 것입니다.

- **Technical Details**: 의사 조건은 원래 노이즈가 포함된 데이터 대신에 사용되며, 이를 학습하기 위해 클래스 없이 지침(Classifier-Free Guidance) 프레임워크를 기반으로 하는 경량 예측 헤드를 구축합니다. 제안된 역시간 확산 조건(Reverse-time Diffusion Condition, RDC) 기법은 의사 조건을 확산 과정에 통합하여 메모리 효과를 강화하고 조건을 더욱 정제하는 데 도움이 됩니다. 학습 목표는 다양한 시간 단계에서 입력과 대상 간의 점수 일치를 최적화하는 것으로 구성됩니다.

- **Performance Highlights**: 제안된 방법은 시각 운동 정책 생성 및 이미지 생성의 두 가지 조건 생성 작업에서 실험을 통해 최첨단(State-of-the-Art) 성능을 달성하였습니다. 특히, 노이즈가 포함된 이미지 관찰에서 이미지 생성 및 Push-T 데이터셋에서의 조건부 정책 생성 실험이 포함되며, 다양한 노이즈 수준에서 뛰어난 성능을 보여주었습니다. 이러한 성과는 제안된 RDC가 메모리 효과를 유의미하게 향상시킴을 나타냅니다.



### A Unified Frequency Domain Decomposition Framework for Interpretable and Robust Time Series Forecasting (https://arxiv.org/abs/2510.10145)
- **What's New**: 이번 연구에서는 FIRE라는 새로운 통합 주파수 영역 분해 프레임워크를 제안합니다. 이 프레임워크는 다양한 유형의 시계열 데이터를 위한 수학적 추상화를 제공하며, 해석 가능하고 강력한 시계열 예측을 달성합니다. FIRE는 진폭과 위상 성분의 독립적 모델링, 주파수 기초 성분의 가변 학습, 목표 손실 함수, 새로운 희소 데이터 훈련 패러다임 등의 주요 혁신을 포함합니다.

- **Technical Details**: FIRE는 특히 주파수 도메인에서의 개념 변동과 기초 진화에 대한 이해를 바탕으로 설계되었습니다. 새로운 손실 함수는 이러한 기초 진화를 명확히 반영하며, 다양한 시계열 데이터에서 시간에 따른 동적 변화를 효과적으로 추적합니다. 이 프레임워크는 Huber 손실과 하이브리드 강/약 수렴 프레임워크를 결합하여 훈련을 가속화하고 일반화 성능을 개선합니다.

- **Performance Highlights**: FIRE는 다양한 장기 예측 벤치마크에서 기존의 최첨단 모델을 일관되게 초과 달성하며, 비용 효율적이고 해석 가능한 솔루션을 제공합니다. 이러한 성과는 산업 응용에 적합하게 설계되었으며, FIRE의 실험 결과는 기존 모델과 비교했을 때 우수한 예측 성능을 나타냅니다.



### Adversarial Attacks on Downstream Weather Forecasting Models: Application to Tropical Cyclone Trajectory Prediction (https://arxiv.org/abs/2510.10140)
- **What's New**: 이 논문은 깊은 학습 기반의 날씨 예측 모델(DLWF)이 적대적 공격에 어떻게 취약한지를 탐구합니다. 특히, 날씨 예측에 미세한 변화를 주어 열대 저기압(TC) 경로 예측을 변형할 수 있는 새로운 방법인 Cyc-Attack을 제안합니다. 기존의 TC 탐지 시스템의 비투명한 특성과 클래스 불균형 문제를 해결하기 위해, 차별 가능한 대체 모델을 사전 학습합니다.

- **Technical Details**: Cyc-Attack은 TC 탐지 시스템의 출력을 근사하기 위해 차별 가능한 대체 모델을 사용하여 적대적 공격을 가능하게 합니다. 공격 과정에서 skewness-aware loss function과 kernel dilation 전략을 적용하여 클래스 불균형 문제를 해결합니다. 또한, 거리 기반의 gradient weighting scheme과 정규화를 통해 변동성을 제어하고 비현실적인 경로를 피할 수 있도록 합니다.

- **Performance Highlights**: 이 방법은 적대적으로 생성된 경로가 현실적이고 쉽게 감지되지 않도록 보장합니다. Cyc-Attack은 과거의 DLWF 예측과의 일치를 유지하면서 TC 경로 예측의 정확성을 개선할 수 있는 잠재력을 가지고 있습니다. 이를 통해, TC 경로 예측에 대한 적대적 공격의 가능성과 실제적인 적용 가능성을 탐색할 수 있으며, 이 분야에서의 연구를 더욱 발전시키는 데 기여할 것으로 기대됩니다.



### PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models (https://arxiv.org/abs/2510.10136)
Comments:
          Accepted by NeurIPS 2025

- **What's New**: 이번 논문에서는 자가 학습 가능한 채널 순열(learnable channel permutation, LCP) 기술을 도입한 PermLLM이라는 새로운 포스트 트레이닝 가지치기 프레임워크를 제안합니다. 이 프레임워크는 N:M 희소성(sparsity)을 위해 설계되었으며, 기존의 수작업 품질 메트릭에 의존하지 않고 출력 오류를 최소화하는 방식으로 작동합니다. PermLLM은 자동으로 채널 순열을 최적화하여 가지치기 과정에서 발생하는 오류를 줄이고, 우수한 성능을 발휘하는 것을 목표로 합니다.

- **Technical Details**: PermLLM은 Sinkhorn 정규화(Sinkhorn normalization)를 통해 불연속적인 순열 행렬을 미분 가능한 소프트 순열 행렬로 변환하여, 최적화를 가능하게 합니다. 또, 효율적인 블록 단위 채널 순열 전략을 포함하여 학습 가능한 매개변수와 계산 복잡도를 유의미하게 줄이고자 합니다. 이 방법은 기존의 일회성 가지치기 방법들과 원활하게 통합되어 가지치기 인식 채널 학습을 가능하게 합니다.

- **Performance Highlights**: 다양한 LLM(Large Language Model) 모델들, 특히 LLaMA 시리즈와 Qwen, OPT 모델에 대해 수행된 실험에서 PermLLM이 기존의 하나의 가지치기 방법에 비해 우수한 성능을 보임을 확인하였습니다. 또한, 이 프레임워크의 채널 순열 작업을 가속화하기 위해 커스터마이즈된 CUDA 커널이 개발되어 Pytorch 기반 구현에 비해 상당한 속도 향상이 이루어졌습니다.



### CacheClip: Accelerating RAG with Effective KV Cache Reus (https://arxiv.org/abs/2510.10129)
- **What's New**: 이 논문에서는 Retrieval-Augmented Generation (RAG) 시스템의 성능 병목 현상을 해결하기 위해 CacheClip이라는 새로운 프레임워크를 제안합니다. CacheClip은 시간-첫 번째-토큰(time-to-first-token, TTFT)을 빠르게 하면서 높은 생성 품질을 유지하는 데 중점을 둡니다. 이 기술은 작은 보조 대형 언어 모델(auxiliary LLM)과의 유사한 주의 분포를 활용하여 중요 토큰을 효과적으로 선택합니다.

- **Technical Details**: CacheClip은 (1) 보조 모델에 의해 선택된 중요 토큰을 재계산하여 chunk 간 의존성을 회복하고, (2) 중복된 주의 sink을 제거하기 위한 공유 prefix, (3) KV cache의 부분 업데이트 동안 지역 일관성을 유지하기 위한 그룹화 전략을 통합합니다. 이 접근 방식은 RAG 시스템에서의 효율성과 품질 문제를 동시에 해결하는 것을 목표로 합니다.

- **Performance Highlights**: 실험 결과 CacheClip은 NIAH와 LongBench에서 각각 94.8%와 85.0%의 전체 주의 성능을 유지하였으며, APE 및 CacheBlend보다 각각 25.2% 및 35.1%의 성능 향상을 보였습니다. 또한 CacheClip은 LLM 추론 속도를 1.92배 가속화하여 RAG 시스템의 효율성-품질 트레이드오프를 효과적으로 해결하였습니다.



### Preference-driven Knowledge Distillation for Few-shot Node Classification (https://arxiv.org/abs/2510.10116)
Comments:
          Accepted at NeurIPS 2025

- **What's New**: 이 논문에서는 preference-driven knowledge distillation (PKD) 프레임워크를 통해 텍스트 속성을 가진 그래프(text-attributed graphs, TAGs)에서 몇 개의 노드로 이루어진 분류를 위한 새로운 방법론을 제시했습니다. PKD는 대규모 언어 모델(large language models, LLMs)과 여러 그래프 신경망(graph neural networks, GNNs)의 강점을 결합하여, 부족한 라벨 문제와 결합하여 노드 예측을 효율적으로 수행합니다. 논문에서는 LLM의 라벨 주석을 활용하여 GNN의 성능을 개선하는 두 가지 모듈, 즉 GNN 선호에 따른 노드 선택기(GNS)와 노드 선호에 따른 GNN 선택기(NGS)를 개발하였습니다.

- **Technical Details**: PKD는 두 가지 핵심 모듈로 구성됩니다. GNN-preference-driven Node Selector (GNS)는 LLM이 이해할 수 있는 그래프 토폴로지를 기반으로 노드 선택을 수행하여, 라벨 주석이 GNN의 성능에 실질적으로 기여하도록 합니다. Node-preference-driven GNN Selector (NGS)는 각 노드에 대해 가장 적절한 GNN 메커니즘을 선택하여, 다양한 GNN에서 학생 GNN으로의 지식 전이를 최적화합니다. 이 과정에서 LLM은 강화 학습(reinforcement learning) 에이전트로 기능하며, 점수로 알려진 학생 GNN의 성능에 따라 보상을 부여합니다.

- **Performance Highlights**: 광범위한 실험을 통해 제안된 PKD 프레임워크가 실제 TAGs에서 몇 개의 노드로 분류하는 데 효과적임을 확인했습니다. 특히, PKD는 더 많은 노드 라벨을 사용하는 최첨단 방법들과 비교해도 우수한 성능을 보여줍니다. 이 연구는 제한된 라벨과 복잡한 노드 토폴로지를 다루는 데 있어 강력한 접근 방식을 제공합니다.



### Lighter-X: An Efficient and Plug-and-play Strategy for Graph-based Recommendation through Decoupled Propagation (https://arxiv.org/abs/2510.10105)
- **What's New**: 이번 연구에서는 Lighter-X라는 새로운 추천 시스템 프레임워크를 제안합니다. 기존의 Graph Neural Networks (GNNs)기반의 추천 모델의 매개변수 수를 줄이기 위해 고안된 이 모델은, 다양한 GNN 아키텍처와 통합이 가능하고 효율성을 크게 향상시킵니다. Lighter-X는 매개변수 복잡도를 감소시키고, 기존 모형의 성능을 유지하면서 큰 그래프의 실제 배치에서의 적용 가능성을 높입니다.

- **Technical Details**: Lighter-X는 매개변수의 불필요한 중복성을 분석하여 최적화 기회를 식별하고, 이를 바탕으로 효율적인 압축 스킴을 제안합니다. 이 프레임워크는 희소 그래프 구조와 고차원 임베딩 매트릭스의 압축을 통해 O(h × d)로 매개변수 복잡도를 줄이며, 여기서 h는 데이터 세트의 희소성을 나타냅니다. 또한, 새로운 프레임워크는 훈련 과정에서의 계산 복잡성을 줄여줍니다.

- **Performance Highlights**: 대규모 상호작용 그래프에서 Lighter-X는 기존 모델인 LightGCN보다 매개변수를 1%만 사용하여도 동등하거나 더 나은 성능을 달성할 수 있음을 실험을 통해 보여주었습니다. 이로 인해 Lighter-X는 훈련 속도를 대폭 향상시키고 필요한 자원을 감소시킵니다. 다양한 실험 결과, Lighter-X는 적은 매개변수로도 경쟁력 있는 결과를 도출할 수 있음을 입증했습니다.



### PANTHER: Generative Pretraining Beyond Language for Sequential User Behavior Modeling (https://arxiv.org/abs/2510.10102)
- **What's New**: 최근 대화형 언어 모델(LLMs)들은 생성적 사전학습을 통해 방대한 세계 지식을 압축된 토큰 표현으로 변환할 수 있음을 보여주었습니다. 하지만 사용자 상호작용의 행동 지식을 모델링하는 데에는 한계가 있었습니다. PANTHER는 사용자 행동에 대한 생성적 사전학습을 확장해서 비표시된 행동 데이터에서 전이 가능한 표현을 학습함으로써 이 문제를 해결합니다.

- **Technical Details**: PANTHER는 사용자 행동 사전학습과 다운스트림 적응을 통합한 하이브리드 생성-판별 프레임워크입니다. 주요 구성 요소로는 다차원 거래 속성을 해석 가능한 어휘로 압축하는 구조적 토큰화(Structured Tokenization), 주기적인 거래 패턴 모델링을 위한 시퀀스 패턴 인식 모듈(Sequence Pattern Recognition Module, SPRM), 정적 인구 통계와 동적 거래 이력을 융합하는 통합 사용자 프로필 임베딩, 밀리초 수준 추론을 위한 실시간 확장성이 포함됩니다.

- **Performance Highlights**: PANTHER는 WeChat Pay에 완전히 배치되어 운영 중이며, 다음 거래 예측에서 HitRate@1을 25.6% 향상시키고 사기 탐지 재현율에서 38.6%의 상대적 개선을 달성했습니다. 공공 벤치마크에서의 크로스 도메인 평가에서도 강한 일반화를 보여주며, transformer 기반 모델들보다 최대 21%의 HitRate@1 향상을 이루는 등 산업적인 사용자 행동 모델링을 위한 확장 가능하고 높은 성능의 프레임워크로 자리매김하였습니다.



### Rademacher Meets Colors: More Expressivity, but at What Cost ? (https://arxiv.org/abs/2510.10101)
- **What's New**: 이번 연구에서는 그래프 신경망(Graph Neural Networks, GNNs)의 표현력과 일반화 능력 간의 관계를 이론적으로 설명합니다. 특히, GNN의 표현력이 높을수록 일반화 오류가 증가할 수 있다는 점을 강조하며, 색칠 알고리즘(coloring algorithms)을 통해 이 두 요소를 연결합니다. 이 연구는 GNN의 다양한 아키텍처와 표현력 측정법에 적용되며, 메시지 전달 GNN(messaging-passing GNN)뿐만 아니라 1-WL에 제한되지 않는다고 명시합니다.

- **Technical Details**: GNN의 표현력을 이해하는 핵심은 그래프 동형성 테스트와의 관계에 있습니다. 기존 연구들에 따르면 메시지 전달 GNN은 1차 WL 테스트와 동등한 표현력을 가진다고 알려져 있지만, 이 연구는 색칠을 이용한 알고리즘에 기반하여 GNN의 Rademacher 복잡도(Rademacher complexity)와 동일성을 부여합니다. 이러한 분석을 통해 표현력이 높은 GNN일수록 Rademacher 복잡도가 커지고, 이는 성능의 일반화 보장에 악영향을 미친다는 것을 보여줍니다.

- **Performance Highlights**: 연구 결과는 GNN의 표현력이 증가할 경우 일반화 의존성도 증가하는 경향이 있음을 나타냅니다. 실험 및 이론적 분석을 통해 GNN 모델의 성능, 즉 일반화 능력이 어떻게 표현력에 의해 영향을 받는지를 통합적으로 제시합니다. 또한, 연구는 Rademacher 복잡도가 데이터 세트 간 샘플 변동성에 대해 어떻게 안정성이 있는지를 입증해, GNN 아키텍처의 신뢰성을 보장할 수 있는 새로운 통찰을 제공합니다.



### What Makes Looped Transformers Perform Better Than Non-Recursive Ones (Provably) (https://arxiv.org/abs/2510.10089)
- **What's New**: 이 논문은 루프 구조를 가진 변환기(Looped-Attn)가 일반적인 변환기(Single-Attn)보다 복잡한 추론 작업에서 뛰어난 성능을 보이는 이유를 이론적으로 설명합니다. 특히, 손실 경관의 기하학을 통해 이러한 차이를 분석하며, 경량화된 루프 구조가 더 복잡한 패턴 학습을 촉진한다고 주장합니다. 이를 기반으로 TRAINING 프로세스를 가속화하는 새로운 프레임워크인 SHIFT (Staged HIerarchical Framework for Progressive Training)를 제안합니다.

- **Technical Details**: Looped-Attn은 반복적인 자기 주의 블록을 통해 내부 표현을 점진적으로 개선하며, 이는 복잡한 문제 해결에서 성능 저하를 극복하는 데 도움을 줍니다. 이 연구에서는 U자형과 V자형 계곡을 구분하여 손실 경관 모델을 확장하고, Looped-Attn이 생성하는 V자형 계곡이 학습 과정에서 더 효과적으로 작업을 수행할 수 있도록 한다고 주장합니다. SHIFT는 성능과 최적화 안정성을 기준으로 Single-Attn에서 Looped-Attn으로 전환하는 기준을 세웁니다.

- **Performance Highlights**: 실험 결과, SHIFT는 순수한 Looped-Attn과 유사한 추론 성능을 보여줌과 동시에 계산 효율성을 크게 개선하는 것으로 나타났습니다. Looped-Attn가 반복 구조로 인해 더 효과적인 학습을 가능하게 한다는 점을 강조하면서, SHIFT 알고리즘이 실제 성능 향상을 이끌어내는 데 기여함을 증명했습니다. 이러한 연구는 변환기 구조의 선택과 손실 경관의 기하학이 모델 성능에 미치는 영향을 깊이 있게 탐구합니다.



### Gradient-based Model Shortcut Detection for Time Series Classification (https://arxiv.org/abs/2510.10075)
Comments:
          Code available at: this https URL

- **What's New**: 본 논문은 딥러닝 기반의 시계열 분류(Time Series Classification)에서 포인트 기반 숏컷(Shortcut) 학습 행동을 조사하는 첫 번째 단계로, 기존의 연구들이 다루지 않은 내부 편향 문제를 탐구합니다. 딥러닝 모델이 훈련 데이터의 겉보기 상관관계에 의존하여 실질적인 일반화 능력을 갖추지 못하는 문제를 다룹니다. 궁극적으로, 새로운 감지 방법인 Shortcut Aggregate Gradient score(SAG)를 제안하여 외부 속성에 의존하지 않고 숏컷을 탐지할 수 있는 기법을 소개합니다.

- **Technical Details**: 연구에서는 ResNet18을 활용하여 UCR 시계열 데이터셋에서 포인트 기반 숏컷을 실험적으로 분석하였습니다. 모델 학습 중 특정 지점에 스파이크(spike)를 추가하여, 모델의 정확도가 90%에서 49%로 급락하게 만든 사례를 통해 딥러닝 모델이 유의미한 특징 대신 의도치 않은 특징에 의존할 수 있음을 보여주었습니다. 새로운 AGG 스코어는 입력의 그래디언트를 집계하여 각 클래스의 그래디언트 중요도를 평가하고, 이를 통해 숏컷의 존재 여부를 판단합니다.

- **Performance Highlights**: 총 40개의 데이터셋 중 24개에서 포인트 숏컷이 확인되었으며, 이는 모델의 훈련 손실과 테스트 손실 간의 비교를 통해 나타났습니다. 제안한 SAG 스코어는 특정 클래스가 그래디언트 분포를 지배하는 정도를 측정하며, 이를 통해 숏컷 탐지의 유효성을 입증하였습니다. 이 연구는 시간 시리즈 모델의 숏컷 학습 문제를 해결하고자 하며, 실제 데이터 상황에서도 사용 가능한 기법을 제안합니다.



### ADEPT: Continual Pretraining via Adaptive Expansion and Dynamic Decoupled Tuning (https://arxiv.org/abs/2510.10071)
- **What's New**: 이번 연구에서는 대형 언어 모델(LLM) 도메인 적응에서 발생하는 파국적 망각(catastrophic forgetting)과 한정된 도메인 용량의 문제를 해결하기 위해 ADEPT라는 새로운 프레임워크를 제안합니다. ADEPT는 일반 역량에 기반한 선택적 레이어 확장과 적응형 단위 분리 튜닝을 포함하는 두 단계의 방법으로, 일반 지식 보존을 극대화하면서 도메인 특화 지식을 효과적으로 주입할 수 있도록 설계되었습니다. 나아가 실험을 통해 ADEPT가 모든 매개변수를 조정하는 기존 연구보다 더욱 효과적임을 입증했습니다.

- **Technical Details**: ADEPT 프레임워크는 먼저 일반 역량을 고려하여 덜 중요하고 도메인 적응이 더 용이한 레이어를 선택해 이들을 복제합니다. 이후, 적응형 단위 분리 튜닝이 적용되어 확장된 레이어 내의 매개변수를 일반 도메인과의 중요도에 따라 분리하고 비대칭 학습 속도를 할당합니다. 이러한 방식으로 모델의 일반 지식을 최대한 보존하면서도 새로운 도메인 정보를 효과적으로 수용할 수 있게 됩니다.

- **Performance Highlights**: ADEPT는 수학 및 의료 분야에서 실험을 통해 목표 도메인에서 5.58%, 일반 도메인에서 5.76%의 정확도 향상을 기록했습니다. 이 방법은 조정된 매개변수의 비율이 15%에 불과하고, 훈련 시간도 기존 방법에 비해 크게 단축됨으로써 효율성을 입증했습니다. 또한, 블라인 스터디와 이론적 분석을 통해 ADEPT의 설계와 접근 방식을 더욱 강화했습니다.



### Translution: Unifying Self-attention and Convolution for Adaptive and Relative Modeling (https://arxiv.org/abs/2510.10060)
Comments:
          technical report

- **What's New**: 이 논문에서는 Translution이라는 새로운 연산을 소개합니다. 이 연산은 self-attention의 적응적 식별 기능과 convolution의 상대적 인코딩 장점을 결합합니다. 그러나 이러한 통합은 파라미터 수의 상당한 증가를 초래하여 현재 사용 가능한 대부분의 계산 자원을 초과합니다. 그래서 우리는 파라미터 수를 줄이기 위해 α-Translution이라는 경량 변형을 제안합니다.

- **Technical Details**: Translution은 쿼리(queries), 키(keys), 값(values) 계산 시 각 거리와 방향에 대해 별도의 매개변수(매트릭스)를 할당하는 convolution 스타일 접근 방식을 사용합니다. 이는 Translution이 상대적 구조를 효과적으로 인코딩할 수 있도록 합니다. 이 논문에서는 컴퓨터 비전과 자연어 처리 작업에서의 실험을 통해 Translution과 α-Translution이 self-attention보다 더 나은 정확도를 달성함을 보입니다. 두 아키텍처는 Vision Transformer(ViT)와 Generative Pre-trained Transformer(GPT)입니다.

- **Performance Highlights**: 실험 결과, Translution과 α-Translution은 self-attention을 넘는 정확도를 보여줍니다. α-Translution은 '이상적'인 원본 Translution보다는 낮은 정확도를 보이지만, self-attention보다 높은 결과를 기록했습니다. 이 연구는 기존의 convolutional neural networks와 self-attention의 한계를 극복하고 새로운 신경망 설계를 위한 길을 여는 데 기여하고 있습니다.



### One4Many-StablePacker: An Efficient Deep Reinforcement Learning Framework for the 3D Bin Packing Problem (https://arxiv.org/abs/2510.10057)
- **What's New**: 3차원 적재 문제(3D-BPP)는 물류 및 창고에서 널리 사용되고 있지만, 기존의 학습 기반 접근 방법은 실제 안정성 관련 제약을 간과하고, 다양한 적재 차원에서 일반화하는 데 한계를 보였습니다. 이를 해결하기 위해 우리는 One4Many-StablePacker (O4M-SP)라는 새로운 심층 강화 학습 프레임워크를 제안합니다. O4M-SP의 주요 장점은 다양한 적재 차원을 단일 학습 과정에서 처리할 수 있으며, 실제에서 흔히 발생하는 지지 및 중량 제약을 통합할 수 있다는 점입니다.

- **Technical Details**: O4M-SP는 두 가지 혁신적 메커니즘을 도입하여 훈련 방법을 개선합니다. 첫째, 적재 속도와 새로운 높이 차이 지표를 통합한 가중 보상 함수를 사용하여 적재 레이아웃에서 포장 효율성을 향상시킵니다. 둘째, 정책 엔트로피 붕괴를 완화하기 위해 클리프드 정책 그래디언트 최적화와 맞춤형 정책 드리프트 방법을 결합하여 포장 과정에서 비최적 솔루션을 피할 수 있도록 중요한 결정 노드에서 탐색을 촉진합니다.

- **Performance Highlights**: 광범위한 실험을 통해 O4M-SP는 다양한 적재 차원에서 성공적으로 일반화하고 기본 방법보다 우수한 성능을 발휘했습니다. 또한 O4M-SP는 안정성 제약이 필요한 적재 시나리오를 효과적으로 다루며 강력한 실제 적용 가능성을 보여줍니다. 이를 통해 O4M-SP는 실질적인 안정성 제약을 강화하여 포장 성능을 향상시키는 것으로 확인되었습니다.



### FOSSIL: Regret-Minimizing Curriculum Learning for Metadata-Free and Low-Data Mpox Diagnosis (https://arxiv.org/abs/2510.10041)
Comments:
          35 pages, 11 figures, submitted to Computers in Biology and Medicine (Elsevier, under review)

- **What's New**: 이번 연구에서는 FOSSIL (Flexible Optimization via Sample-Sensitive Importance Learning) 프레임워크를 최초로 생물의학 분야에 적용하여, 샘플의 난이도에 따라 훈련 강조를 적응적으로 조절하는 방법을 제시합니다. 이 접근법은 작은 데이터셋의 최적화 불안정성과 일반화 부족 등의 문제를 해결하고자 합니다. 특히, 피부 병변 진단에 활용되는 convolutional과 transformer 기반 아키텍처에 FOSSIL을 통합하여 성능을 개선했습니다.

- **Technical Details**: FOSSIL 프레임워크는 모델의 필요에 따라 샘플에 중요도를 부여하는 회귀 최소화 가중치 체계를 적용합니다. 각 샘플은 예상 난이도에 따라 중요도가 지수적으로 감소하는 이론적으로 유도된 가중치를 받으며, 이런 방식은 누적 학습 회귀의 상한을 최소화하여 데이터가 부족하거나 노이즈가 존재할 때도 안정적인 수렴을 촉진합니다. 새로운 방법론은 focal loss, meta-weighting, 커리큘럼 학습을 통합하여 단일 회귀 최소화 형태로 제공합니다.

- **Performance Highlights**: FOSSIL을 활용한 실험 결과, 피부 병변 진단에서 AUC가 0.9573, Expected Calibration Error (ECE)가 0.053으로 크게 향상되었습니다. 전통적인 방식과 비교하여 데이터가 부족한 상황에서도 일반화, 보정 및 강건성을 크게 개선하며, 원활한 최적화 및 낮은 ECE를 달성했습니다. 이는 FOSSIL이 데이터 부족 환경에서도 신뢰할 수 있는 의료 AI 시스템을 구축하는 데 효과적임을 입증합니다.



### Experience-Efficient Model-Free Deep Reinforcement Learning Using Pre-Training (https://arxiv.org/abs/2510.10029)
- **What's New**: PPOPT는 Pretraining을 활용한 새로운 모델 프리 (model-free) 딥 강화 학습 알고리즘으로, 물리 기반 환경에서의 높은 학습 효율성과 안정성을 제공합니다. 기존의 PPO 알고리즘에 비해 훈련 샘플이 부족한 상황에서도 높은 성능을 발휘하도록 설계되었습니다. PPOPT의 주요 혁신은 두 개의 완전 연결 네트워크 사이에 위치한 사전 훈련된 신경망 중간 섹션으로, 이는 에이전트가 목표 환경을 보다 효율적으로 학습할 수 있도록 돕습니다.

- **Technical Details**: PPOPT는 기본적으로 Proximal Policy Optimization (PPO) 알고리즘을 수정하여 새로운 정책 네트워크 아키텍처를 도입한 것입니다. 이 네트워크는 유사한 물리 특성을 가진 환경에서 사전 훈련된 신경망을 중간에 배치해 두 개의 완전 연결 네트워크 사이에 위치시킵니다. 이러한 설계는 에이전트가 사전 훈련된 환경의 물리적 특성을 활용할 수 있도록 해, 목표 환경에 대한 학습 효율성을 높여줍니다.

- **Performance Highlights**: PPOPT는 제한된 경험 샘플 조건에서도 기본 PPO를 지속적으로 초과하는 성능을 보였습니다. 각종 실험을 통해 PPOPT는 PPO와 달리 매우 적은 경험 샘플에서도 신속하고 실용적인 정책을 발견합니다. 그러나 PPOPT는 DYNA DDPG 같은 모델 기반 방법에는 아직 성능에서 뒤처지지만, 모델 프리 특성 덕분에 훈련 시간이 대폭 줄어드는 장점을 보입니다.



### Efficient Onboard Vision-Language Inference in UAV-Enabled Low-Altitude Economy Networks via LLM-Enhanced Optimization (https://arxiv.org/abs/2510.10028)
- **What's New**: 본 논문에서는 저고도 경제 네트워크(LAENets)에서 UAV(무인 항공기)를 활용하여 VLM(비전-언어 모델)을 통합함으로써 실시간 멀티모달 추론을 지원하는 시스템 모델을 제안합니다. UAV의 이동성과 사용자-드론 간 통신 및 VQA(비주얼 질문 답변) 파이프라인을 함께 포착하는 새로운 접근 방식으로, 사용자의 정확도 요구사항에 따라 작업 지연 시간을 최소화하는 혼합 정수 비선형 최적화 문제를 설정합니다.

- **Technical Details**: 제안된 접근 방식은 두 가지 주요 구성요소로 이루어진 계층적 최적화 프레임워크를 통해 문제를 해결합니다. 첫 번째는 정확도 요구사항 하에 자원을 할당하기 위한 ARPO(Alternating Resolution and Power Optimization) 알고리즘이고, 두 번째는 UAV 경로 최적화를 위해 LLaRA(Large Language Model-augmented Reinforcement Learning Approach) 방법론입니다. LLM(대형 언어 모델)이 보상 설계를 정제하는 전문가 역할을 하여 실시간 의사결정에 추가적인 지연이 발생하지 않도록 합니다.

- **Performance Highlights**: 수치 결과는 LAENet의 동적 조건 하에서 추론 성능과 통신 효율성을 증대시키는 이들의 접근 방식의 효율성을 보여줍니다. UAV는 자원을 제한하고 복잡한 환경에서도 고효율의 지능형 서비스를 제공할 수 있는 가능성을 지니고 있으며, 이는 환경 감지 및 자율 배송과 같은 다양한 응용 프로그램에 기여할 것으로 기대됩니다.



### Skill-Targeted Adaptive Training (https://arxiv.org/abs/2510.10023)
- **What's New**: 이 논문에서는 언어 모델의 학습 중 경과적 성능 향상이 정체되는 문제를 해결하기 위해 새로운 미세 조정 전략인 STAT(Statistical Adaptation Training)를 제안합니다. STAT는 강력한 대형 언어 모델(LLM)의 메타인지 능력을 활용하여, 학생 모델이 필요한 기술 목록을 작성하고, 각 데이터 포인트에 대한 기술을 레이블링합니다. 이 과정을 통해 학생의 응답에서 기술 적용 실패를 모니터링하며, 이를 바탕으로 새로운 교육 데이터를 생성하거나 민첩하게 가중치를 조정합니다.

- **Technical Details**: 기술적인 관점에서, STAT는 두 가지 방식인 STAT-Sel와 STAT-Syn을 통해 작동합니다. STAT-Sel은 기존 훈련 예제의 가중치를 조정하여 학생 모델이 적절한 기술 부족을 극복할 수 있도록 돕습니다. 반면, STAT-Syn은 부족한 기술과 관련된 합성 훈련 데이터를 생성하여 학생 모델이 보다 다양한 문제를 해결할 수 있도록 지원합니다.

- **Performance Highlights**: STAT의 성능 향상은 Llama 및 Qwen 모델을 통한 광범위한 실험에서 확인되었으며, MATH 데이터셋에서 최대 7.5%의 성능 향상을 이루었습니다. 또한, AIME24/25, AMC23과 같은 분포 외 기준에서도 평균 4.6% 향상이 관찰되었습니다. STAT는 RL(Random Learning) 방법과도 보완적으로 작용하여, 모델의 성능을 더욱 강화할 수 있다는 점이 중요한 발견으로 여겨집니다.



### Bidirectional Time-Frequency Pyramid Network for Enhanced Robust EEG Classification (https://arxiv.org/abs/2510.10004)
Comments:
          Accepted to IEEE BIBM 2025

- **What's New**: 이번 연구에서는 EEG 인식 모델의 한계를 극복하기 위해 BITE(Bidirectional Time-Freq Pyramid Network)라는 통합 아키텍처를 제안합니다. BITE는 단일 모델로 다양한 EEG 패러다임(MI, SSVEP)에 적용 가능하며, 시간-주파수 상호 작용을 동적으로 모델링하는 데 중점을 두고 개발되었습니다. 이 구조는 강력한 다중 스트림 시너지(multistream synergy)와 피라미드 시간-주파수 주의(Pyramid Time-Frequency Attention)를 활용하여 EEG 신호의 견고한 인식을 가능하게 합니다.

- **Technical Details**: BITE 모델은 두 가지 주요 스트림으로 구성되어 시간 및 주파수 신호를 동시에 처리합니다. 각 주파수 스트림은 STFT(Short-Time Fourier Transform)를 통해 동기화되어, 시간적 맥락을 고려합니다. 또한, 피라미드 시간-주파수 주의 모듈(PTFA)을 이용해 다중 스케일 기능을 강화하고, BiTCN(Bidirectional Temporal Convolutional Network)을 통해 양방향 역동적인 신경 패턴을 캡처합니다.

- **Performance Highlights**: BITE는 BCICIV-2A/2B, HGD, SD-SSVEP의 네 가지 서로 다른 패러다임에서 최첨단 성능을 달성했습니다. 이 모델은 피험자 간 일반화와 피험자 내 정확도에서 모두 뛰어난 성능을 보였습니다. 실험 결과, BITE는 시간-주파수 처리 방식과 양방향 맥락 통합이 신뢰할 수 있는 BCI 시스템 구축에 필수적임을 입증하였습니다.



### Tight Robustness Certificates and Wasserstein Distributional Attacks for Deep Neural Networks (https://arxiv.org/abs/2510.10000)
- **What's New**: 본 논문에서는 Wasserstein 분포적 강건 최적화(WDRO)의 상한을 타이트하게 만들기 위해 새로운 프라이멀 접근 방식을 도입하며, 정확한 Lipschitz certificate 개념을 채택합니다. 또한, 기존의 포인트별 공격 방식과 변형 방식에 비해 공격 지점의 수와 위치에서 더 큰 유연성을 제공하는 새로운 Wasserstein 분포적 공격(WDA)을 제안합니다. 이러한 접근 방식을 통해 WDRO 문제의 정확한 접근 가능성을 확보하였습니다.

- **Technical Details**: WDRO 문제에 대한 상한과 하한을 분석하기 위해 ReLU 활성화 함수를 가진 네트워크의 조각별 선형 구조를 활용합니다. WDA는 Wasserstein 구역 내에서 직접적으로 적대적 분포를 구성하는 새로운 방법으로, 기존의 점별 섭동 방식을 넘어 분포적 공격을 지원합니다. 이론적으로 우리는 logit map의 Lipschitz 상수와 소프트맥스 교차 엔트로피 손실의 민감성을 결합하여 WDRO의 상한을 도출하였습니다.

- **Performance Highlights**: WDA는 다양한 설정에서 기존의 최첨단 공격 방법보다 강력한 적대적 예제를 발견하는 경향을 보였습니다. 특히, WideResNet 백본을 사용하여 CIFAR-10/100 데이터 세트에서 APGD-DLR보다 낮은 강건 정확도를 달성하였고, Adaptive Auto Attack 프레임워크와 통합 시 A3보다 일관되게 더 나은 성능을 보였습니다. 이러한 결과는 분포적 관점이 더 긴밀한 이론적 인증을 제공할 뿐만 아니라 보다 효과적인 공격을 가능하게 한다는 주장을 뒷받침합니다.



### Learning Joint Embeddings of Function and Process Call Graphs for Malware Detection (https://arxiv.org/abs/2510.09984)
- **What's New**: 이 논문에서는 맬웨어 분석과 취약점 탐지에서 중요한 소프트웨어 그래프 표현의 통합된 접근 방식을 제안합니다. 기존 연구들은 기능 호출 그래프(Function Call Graphs, FCGs)와 프로세스 호출 그래프(Process Call Graphs, PCGs)를 별도로 분석하는 데 중점을 두었으나, 본 연구에서는 두 가지 그래프의 결합 모델링을 통해 멀티 퍼스펙티브 분석을 가능하게 합니다. GeminiNet이라는 새로운 신경망 아키텍처를 제안하여 FCGs와 PCGs 간의 공동 임베딩을 학습합니다.

- **Technical Details**: FCG는 프로그램 내에서 함수 간의 종속성을 시각화하는 반면, PCG는 실행 시간 동안의 프로세스 상호작용을 모델링합니다. 데이터셋은 635개의 Windows 실행 파일로 구성되며, Ghidra를 사용해 FCG를 생성하고, Any.Run을 사용해 PCG를 수집합니다. 이 연구는 그래프 신경망(GNN)과 관련하여 FCG와 PCG로부터 기능 노드 및 커뮤니케이션 관계를 정의하고, 노드 특성으로는 Local Degree Profile과 Shannon Entropy를 채택하여 그래프 분석 성능을 향상시킵니다.

- **Performance Highlights**: GeminiNet은 단일 그래프 모델에 비해 성능이 우수하다는 실험 결과를 보였습니다. 두 그래프 구조의 통합을 통해 더 많은 정보와 더 나은 predictive performance를 확보하였습니다. 실험에서 LDP, Entropy, LDP+Entropy를 활용하여 결합된 그래프의 예측력을 향상시키는 데 성공하였습니다. 이 연구는 맬웨어 탐지 외에도 소프트웨어 취약점 분석 및 바이너리 유사성 탐지로 일반화할 수 있는 가능성을 제시합니다.



### An Unsupervised Time Series Anomaly Detection Approach for Efficient Online Process Monitoring of Additive Manufacturing (https://arxiv.org/abs/2510.09977)
Comments:
          2025 IEEE 21st International Conference on Automation Science and Engineering

- **What's New**: 본 연구에서는 온라인 프로세스 모니터링을 위한 새로운 비지도 학습 기반의 이상 탐지 알고리즘을 제안합니다. 기존 방법들이 레이블 데이터에 크게 의존하거나 극단적인 이상치만 탐지하는데 그친 반면, 본 방법은 보다 미세한 세멘틱 (semantic) 이상을 정확히 감지할 수 있도록 설계되었습니다. 제안된 CSSAD(Convolutional Semantic Segmentation Anomaly Detection) 방식은 실제 센서 데이터를 기반으로 하여 결함 이상을 정확히 식별할 수 있는 점에서 큰 진전을 이룬 것으로 평가됩니다.

- **Technical Details**: 제안된 알고리즘은 두 가지 구성 요소로 이루어져 있습니다: Convolution-based Matrix Profile (Conv-MP)와 세멘틱 세그멘테이션 기반 이상 탐지 알고리즘입니다. Conv-MP는 인접한 서브 시퀀스 간의 유사성을 캡쳐하여 이상이 발생하는 정확한 시점을 식별하는 역할을 합니다. 또한, 저사양 시스템에서도 빠른 처리가 가능하도록 설계되어 있으며, CPU 및 GPU에서의 효율성을 극대화했습니다.

- **Performance Highlights**: 제안된 방법은 실제 생산 환경에서 수집된 센서 데이터를 활용하여 다양한 공정 이상을 효과적으로 탐지하는 성능을 입증하였습니다. 낮은 허위 탐지 비율을 유지하면서 이상 위치를 정확하게 찾을 수 있게 되어, 현업에서의 온라인 모니터링 시 보다 신뢰할 수 있는 솔루션을 제공합니다. 과거의 기술들과 달리, 미세한 패턴 변화를 감지할 수 있는 점에서 큰 장점이 있습니다.



### Reinforcement Fine-Tuning of Flow-Matching Policies for Vision-Language-Action Models (https://arxiv.org/abs/2510.09976)
- **What's New**: 이 논문에서는 Flow Policy Optimization (FPO)라는 새 알고리즘을 제안하여 기존의 VLA (Vision-Language-Action) 모델에서 정책 그래디언트 방법의 한계를 극복하고자 합니다. 특히, FPO는 중요 샘플링을 재구성하여 액션 발생의 가능성을 직접 계산하지 않고도 온라인 상호작용을 통해 VLA 모델을 최적화할 수 있게 합니다. 이로 인해 VLA 모델이 보다 효율적으로 학습하고 탐험할 수 있도록 합니다.

- **Technical Details**: FPO는 actor-critic 프레임워크를 기반으로 하며, 기존의 조건부 흐름 일치 정책을 온라인에서 미세 조정하는 데 필요한 동작 가능성을 요구하지 않습니다. 알고리즘의 핵심은 매 샘플의 조건부 흐름 일치 목표에서 오는 변화를 구조에 맞춰 신호로 활용하여, 가능성 비율을 유도하고 PPO 스타일의 클리핑 서라게이트에서 활용하는 것입니다. 이를 위해 다양하게 구조 인식 크레딧 할당, 신뢰 영역 보호를 위한 클리핑 서라게이트, 다단계 잠재 탐색, 강력한 가치 추정을 위한 비평가 앙상블 등이 통합됩니다.

- **Performance Highlights**: FPO는 LIBERO 벤치마크와 ALOHA 시뮬레이션 작업에서 여러 성능 기준을 초과하는 결과를 보이며, π0 모델의 온라인 강화 미세 조정에서 강력한 효능을 입증합니다. 실험 결과, π0-FPO가 OpenVLA, Octo, Diffusion Policy 및 GRAPE 등의 강력한 기초선에 비해 우수한 성능을 보였으며, LIBERO 벤치마크에서 평균 성공률이 87.2%에 달하면서 이전의 모방 성능을 초월했습니다.



### Homomorphic Mappings for Value-Preserving State Aggregation in Markov Decision Processes (https://arxiv.org/abs/2510.09965)
- **What's New**: 이 논문은 Markov Decision Processes (MDP)에서 상태 집합(state aggregation)이 계산 복잡성을 줄이면서도 원래 시스템의 성능을 유지하는 방법론을 혁신적으로 제안합니다. 특히, 최적 정책 등이 집합된 추상 공간에서도 원래 MDP에서 최적성을 유지하는 'optimal policy equivalence'를 보장하는 새로운 추상화 프레임워크를 소개하고, 이를 위해 two Markov chains의 동형성(homomorphism) 개념을 활용합니다. 또한, 최적 정책의 동등성 확보를 위한 충분 조건(sufficient condition)을 제시합니다.

- **Technical Details**: 이번 연구는 상태 집합의 추상화 방식으로 동형 Markov 체인(homomorphic Markov chains)을 제안하며, 여기서 동형 사상(homomorphic mappings) 개념을 통해 가치 함수(value functions) 간의 선형적 관계(linear relationship)를 성립시킵니다. 실험을 통해 Homomorphic Policy Gradient (HPG)와 Error-Bounded Homomorphic Policy Gradient (EBHPG) 알고리즘을 개발했으며, 이 두 알고리즘은 집합화로 인해 발생하는 성능 손실과 계산 효율성 간의 균형을 제공합니다. 실험적으로는 synthetic 및 structured 환경(weakly coupled MDPs, FourRooms navigation, queuing networks)에서 성능을 평가했습니다.

- **Performance Highlights**: 실험 결과, 제안된 알고리즘은 기존의 집합화 기법에 비해 훈련 효율성이 향상되었으며, 정책 품질에서도 경쟁력을 나타냈습니다. 특히, HPG와 EBHPG는 각각 최적 정책 동등성을 보장하며, 성능 저하를 최소화하는 유리한 트레이드오프를 달성했습니다. 논문의 결론에서는 이러한 결과들이 기존의 방법론보다 더 효과적이고 강건하며, 특정 상황에서도 잘 작동할 수 있음을 강조하였습니다.



### Clustering Result Re-guided Incomplete Multi-view Spectral Clustering (https://arxiv.org/abs/2510.09959)
- **What's New**: 이번 논문에서는 연결성이 누락된 다중 뷰 데이터를 위한 새로운 클러스터링 방법인 CRG_IMSC(Clustering Result re-Guided Incomplete Multi-view Spectral Clustering)를 제안합니다. 이 방법은 기존의 K-means 알고리즘을 배제하고, 비음수 제약 조건을 도입하여 추출된 특성으로부터 직접 클러스터링 결과를 얻습니다. 또한 클러스터링 결과에 따라 연결성 행렬을 구성하고, 이 행렬을 기반으로 자기 표현의 잔여를 최소화하여 클러스터링을 유도합니다.

- **Technical Details**: CRG_IMSC 방법은 비음수 제약 조건(nonnegative constraint)을 사용하여 특성을 처리하고, 클러스터링 결과를 통해 연결성 행렬(connectivity matrix)을 구성합니다. 이 행렬은 자기 표현(self-representation)의 최소화를 위해 활용되며, 새로운 반복(iterative) 알고리즘을 통해 최적화 문제를 해결합니다. 이 과정에서 제안된 곱셈 업데이트 규칙(multiplicative update rule)은 이론적으로 수렴성이 입증되었습니다.

- **Performance Highlights**: 벤치마크 데이터셋을 사용한 실험 결과, CRG_IMSC는 최신의 클러스터링 방법들보다 더 뛰어난 성능을 보였습니다. 실험은 CRG_IMSC 알고리즘이 수렴함을 보여주었으며, 다중 뷰 데이터에 대한 클러스터링 효율성이 높다는 것을 입증합니다.



### Conformal Sparsification for Bandwidth-Efficient Edge-Cloud Speculative Decoding (https://arxiv.org/abs/2510.09942)
Comments:
          39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: AI and ML for Next-Generation Wireless Communications and Networking (AI4NextG)

- **What's New**: 본 논문에서는 Edge-cloud 환경에서 소형 언어 모델(SLM)과 대형 언어 모델(LLM) 간의 협업을 통한 효율적인 추론 방법인 Speculative Decoding (SD)에 대해 다룹니다. 특히, 통신 대역폭을 고려하여 토큰 분포를 효과적으로 압축하는 Sparse Quantize-and-Sample SD (SQS-SD) 프레임워크를 제안합니다. 이 프레임워크는 불필요한 데이터를 줄이고, 분산 희소성(distributional sparsity)을 활용하여 성능을 개선합니다.

- **Technical Details**: 이 연구에서는 정보 이론적 분석을 통해 SQS 성능의 경계(condition)를 도출하고, SLM과 LLM 간의 분포 불일치(token rejection rate) 및 양자화 왜곡(quantization distortion)간의 기여도를 분석합니다. K-SQS 및 C-SQS와 같은 다양한 접근 방식을 통해 고정된 상위 K 추출 및 온라인 적합 예측을 사용하여 토큰 세트를 조정하며, 이를 통해 통신 비용을 최적화합니다. 또한, 양자화(quantization)는 SD 파이프라인의 핵심 구성 요소로, 통신의 효율성을 극대화합니다.

- **Performance Highlights**: 실험 결과, SQS와 C-SQS 모두 요구되는 대역폭을 크게 줄이고, 종단 간 대기 시간(end-to-end latency)에서 유의미한 개선을 보여주었습니다. 이는 정확도 손실이 거의 없이 수행되었으며, 향후 Edge-cloud LLM 추론에서의 잠재적인 응용 가능성을 나타냅니다. 결국, 이 연구는 대역폭과 정확도 간의 균형을 이끌어내는 방향론적 가이드를 제시합니다.



### MemPromptTSS: Persistent Prompt Memory for Iterative Multi-Granularity Time Series State Segmentation (https://arxiv.org/abs/2510.09930)
Comments:
          This paper is currently under review. The code will be made available upon acceptance

- **What's New**: 이번 논문에서는 MemPromptTSS라는 새로운 프레임워크를 제안하여, 멀티-그레인(segmentation) 세그먼트에서 제시된 프롬프트의 영향을 지속적으로 보존하는 방법을 탐구합니다. 이는 사용자 피드백을 바탕으로 긴 시퀀스에서의 일관성을 보장하며, 효과적인 데이터 세분화(segmentation)를 가능하게 합니다. 이 프레임워크는 기존의 한정된 지역(context)을 넘어 전체 시퀀스에 걸쳐 프롬프트의 영향을 지속적으로 말합니다.

- **Technical Details**: MemPromptTSS는 메모리 인코더를 사용하여 주어진 프롬프트와 그 주변 서브시퀀스를 메모리 토큰으로 변환하여 저장합니다. 이는 사용자가 제공한 입력이 반복을 거치면서도 잊혀지지 않고 가장 중요한 정보로 남아 있도록 보장합니다. 또한, 모든 이후 예측은 저장된 프롬프트의 전체 은행을 기준으로 하여, 사용자의 모든 입력이 그 시퀀스 전체에 영향을 미치도록 합니다.

- **Performance Highlights**: 여섯 개의 데이터셋을 통해 평가한 결과, MemPromptTSS는 단일 및 멀티-그레인 세그먼트에 대해 각각 23%와 85%의 정확도 향상을 이루었습니다. 반복적인 추론(iterative inference)에서도 MemPromptTSS는 평균 2.66%의 향상을 이루었으며, 기존의 PromptTSS의 1.19%와 비교할 때 더 강력한 세분화 능력을 제공합니다.



### Phase-Aware Deep Learning with Complex-Valued CNNs for Audio Signal Applications (https://arxiv.org/abs/2510.09926)
- **What's New**: 본 연구는 복소수 값을 가진 합성곱 신경망(Complex-Valued Convolutional Neural Networks, CVCNNs)을 오디오 신호 처리에 적용하고, 그 과정에서 종종 간과되는 위상 정보(phase information)를 보존하고 활용하는 방법을 탐구합니다. CVCNN의 기초 이론, 복소수 합성곱, 풀링 레이어, Wirtinger 기반 미분 및 다양한 복소수 활성화 함수들을 소개하며, 안정적인 훈련 동역학을 보장하기 위해 복잡한 배치 정규화(complex batch normalization)와 가중치 초기화 기법(weight initialization schemes)이 포함됩니다.

- **Technical Details**: CVCNN 아키텍처의 핵심 요소로는 복소수 합성곱(complex-valued convolutions), 활성화 함수(activation functions), 매개변수 초기화(parameter initialization)를 살펴보며, 전통적인 실수 합성곱 신경망(real-valued CNNs)과의 비교를 통해 구조적 유사성과 기능적 차이점을 강조합니다. 복소수 합성곱의 수학적 정의는 일반적으로 실수 합성곱을 복소수 영역으로 확장하고, 피쳐 맵에서의 풀링 함수는 노이즈를 억제하고 공간 차원을 줄이는 데 사용됩니다.

- **Performance Highlights**: 실험을 통해 CVCNN이 이미지 데이터셋(MNIST, KMNIST, FMNIST)에서 실수 CNN과 경쟁력있는 성능을 보여줌을 확인하였고, 오디오 클래시피케이션에서 Mel-Frequency Cepstral Coefficients (MFCCs)를 사용하여 실수 CNN보다 약간 더 나은 성능을 나타냈습니다. 마지막 실험에서는 GNN을 도입하여 위상 정보를 에지 가중치(edge weighting)로 모델링하였으며, 위상이 포함될 경우 이진 및 다중 클래스 장르 분류에서 측정 가능한 향상을 이끌어낼 수 있음을 입증했습니다.



### AutoGD: Automatic Learning Rate Selection for Gradient Descen (https://arxiv.org/abs/2510.09923)
- **What's New**: 이 논문에서는 새로운 그래디언트 강하 알고리즘인 AutoGD를 소개합니다. 이 방법은 사용자가 학습률(learning rate)을 조정할 필요 없이 각 반복(iteration)에서 학습률을 자동으로 증가시키거나 감소시킬지 결정합니다. AutoGD는 전통적인 문제 및 변분 추론 최적화 과제에서 뛰어난 성능을 보이며, AutoBFGS 및 AutoLBFGS와 같은 확장도 지원합니다.

- **Technical Details**: AutoGD 알고리즘은 초기 반복값(x0), 기본 학습률(γ0), 학습률 스케일링 팩터(c) 및 Armijo 상수(η) 등의 입력을 받습니다. 각 반복(iteration) 후에 기본 학습률은 증가, 감소 또는 변하지 않을 수 있으며, 주변의 여러 학습률(c−1​γ, γ, c​γ)을 평가하여 업데이트됩니다. 이 과정에서 학습률을 적절히 선택하는 것이 특징이며, 이론적으로도 수렴을 보장합니다.

- **Performance Highlights**: 실험적으로 AutoGD는 다른 그래디언트 강하 방법보다 뛰어난 성능을 발휘하며, 파라미터 튜닝 없이 작동하는 1차 방법과 유사한 성능을 보여줍니다. 또한, 이 알고리즘은 초기에 주어진 조건 하에 매우 강력한 성능을 유지하며, 다양한 최적화 문제에 대해 높은 견고성을 유지합니다. AutoGD의 성능은 실험을 통해 검증되었고, 이로 인해 보편적인 블랙 박스 최적화 알고리즘으로 권장됩니다.



### Advancing Intoxication Detection: A Smartwatch-Based Approach (https://arxiv.org/abs/2510.09916)
- **What's New**: 본 연구는 알코올 소비 문제를 해결하기 위해 스마트워치 데이터를 활용한 혁신적인 접근 방식을 제안합니다. 기존 연구들이 제한적인 스마트폰 모션 데이터와 전통적인 머신러닝 알고리즘을 사용한 반면, 이 연구는 가속도계(accelerometer), 자이로스코프(gyroscope), 심박수(heart rate) 데이터를 결합하여 중독 수준을 분류하는 최초의 연구입니다. 새로운 데이터셋을 활용하여 사용자에게 실시간 개입(JITAIs)을 제공함으로써, 안전한 음주 습관 촉진을 목표로 하고 있습니다.

- **Technical Details**: 이 연구는 30명의 참여자로부터 3주 동안 수집된 TAC, 가속도계, 자이로스코프, 심박수 데이터로 구성된 데이터셋을 생성했습니다. 최신 머신러닝 모델인 Transformer, Bi-LSTM, GRU, 1D-CNN, Hyperdimensional Computing(HDC)을 통해 중독 분류를 평가하고, 각 알고리즘의 성능 지표를 비교하였습니다. HDC 모델은 정확성과 효율성의 균형을 이루며, 리소스 제약 환경에서도 실용성을 입증했습니다.

- **Performance Highlights**: HDC 모델은 정확성과 효율성에서 최상의 결과를 달성하여 스마트워치 기반 애플리케이션에 적합할 것으로 보입니다. 본 연구는 기존의 음주 감지 방법의 한계를 극복하고, 사용자에게 개인화된 실시간 개입을 통해 건강한 음주 습관을 촉진하는 가능성을 열었습니다. 이 연구의 성과는 인공지능 기술이 공공 건강에 기여할 수 있는 새로운 모델을 제시하며, 앞으로의 연구에 중요한 기초 자료로 활용될 수 있습니다.



### Augmenting generative models with biomedical knowledge graphs improves targeted drug discovery (https://arxiv.org/abs/2510.09914)
Comments:
          This paper has been accepted for publication in the IEEE Transactions on Artificial Intelligence, October 2025

- **What's New**: 이 연구에서는 K-DREAM(Knowledge-Driven Embedding-Augmented Model)이라는 새로운 프레임워크를 소개합니다. 이 모델은 지식 그래프를 활용하여 약물 발견을 위한 확산 기반 생성 모델을 증강시키고, 생성된 분자가 특정 치료 목표와 더 잘 일치하도록 방향을 제시합니다. K-DREAM은 전통적인 휴리스틱 기반 접근 방식을 넘어 생물학적 관련성과 치료 적합성을 갖춘 화합물을 생성할 수 있도록 설계되었습니다.

- **Technical Details**: K-DREAM은 생물의학적 지식을 구조화된 형태로 통합하여 분자 생성의 과정을 안내합니다. 지식 그래프 임베딩(Knowledge Graph Embedding, KGE) 기술을 사용하여 그래프의 엔티티와 관계를 진화된 벡터 공간으로 변환하고 이를 생성 프레임워크에 통합함으로써, 생물학적 정보의 의미론적 무결성을 유지합니다. 또한, A stochastic local closed world assumption (sLCWA)를 통해 모델 학습 과정에서 부정적인 삼중항 생성을 최적화합니다.

- **Performance Highlights**: K-DREAM은 특정 단백질 표적을 대상으로 한 도킹 연구에서 더 높은 도킹 점수를 기록하며, 기존의 다른 생성 모델보다 개선된 생물학적 관련성과 치료 잠재력을 가진 화합물을 생성합니다. 또한, K-DREAM의 적응성 덕분에 다중 표적 약물 설계와 같은 다양한 생성 작업을 수행할 수 있으며, 복잡한 질병 메커니즘을 해결할 수 있는 가능성을 보여줍니다.



### Stability of Transformers under Layer Normalization (https://arxiv.org/abs/2510.09904)
- **What's New**: 본 논문은 딥러닝에서 일반적으로 사용되는 Transformer의 학습 안정성을 개선하기 위한 레이어 정규화(layer normalization)의 위치에 대한 체계적인 연구를 수행합니다. 다양한 레이어 정규화 배치의 변화를 통해 Forward stability(정방향 안정성)와 Backward stability(역방향 안정성)를 분석하며, 이론적인 통찰력을 제공합니다. 이러한 분석은 새로운 아키텍처 수정의 안정성을 검증하는 데 기여할 수 있습니다.

- **Technical Details**: Transformers의 각 레이어는 인코딩된 입력을 통과하며, 이 과정에서 self-attention과 feedforward 네트워크를 포함합니다. 이 논문에서는 최적 제어 이론(optimal control theory)을 사용하여 Pre-LN 아키텍처의 불안정성을 설명하고, Peri-LN 아키텍처가 어떻게 보다 안정적인(hidden states) 상태를 유지하는지를 분석합니다. 또한, 다양한 레이어 정규화 배치에 따라 학습 과정에서의 그래디언트 특성을 분석하여 안정성을 향상시키는 방법을 제시합니다.

- **Performance Highlights**: 제안된 방법은 Peri-LN 아키텍처에서 잔여 단계 조정을 통해 안정성과 성능을 동시에 개선하는 데 기여합니다. 실험 결과는 이론적 발견을 뒷받침하며, 향후 Transformer 아키텍처 설계에 중요한 지침을 제공할 수 있습니다. 이러한 분석을 통해 안정성과 성능을 개선할 수 있는 방향성을 제시하며, 다양한 레이어 정규화 방식의 효과를 비교합니다.



### Learning Bug Context for PyTorch-to-JAX Translation with LLMs (https://arxiv.org/abs/2510.09898)
- **What's New**: 이 논문에서는 최근 코드 번역 및 LLM(대형 언어 모델) 발전에도 불구하고, PyTorch에서 JAX로의 번역이 여전히 어렵다는 점을 강조합니다. 이를 해결하기 위해 T2J라는 프레임워크를 제시하며, 이 프레임워크는 데이터 세트의 큐레이션과 구조화된 프롬프트 기법을 활용합니다. 특히, PyTorch에서 JAX로 이동하기 위해 특별히 설계된 최초의 버그 수정 데이터 세트를 제공합니다.

- **Technical Details**: T2J 프레임워크는 세 단계로 구성됩니다: 첫째, TorchLeet와 CodeParrot에서 얻은 PyTorch 및 JAX 코드 스니펫의 병렬 코퍼스를 생성합니다. 둘째, GPT-4o 모델을 활용하여 초기 JAX 번역을 생성한 후, 전문가 개발자가 이를 반복적으로 수정하여 기능적으로 동등한 결과를 도출합니다. 마지막으로, 수정된 버그 데이터 세트에서 추출한 구조화된 지침을 바탕으로 증강된 프롬프트를 설계하여 LLM의 성능을 향상시킵니다.

- **Performance Highlights**: T2J 프레임워크는 CodeBLEU에서 최대 10%, T2J FixCost Score에서 50%, T2J CodeTrans Score에서 1.33 포인트, T2J Comparison Score에서 100% 향상을 보여줍니다. 생성된 JAX 코드는 기준 대비 최대 2.5배 더 빠르게 실행됩니다. 이러한 결과는 LLM 기반의 코드 번역을 개선하는 데 있어 T2J의 효과를 강조합니다.



### Chain-of-Influence: Tracing Interdependencies Across Time and Features in Clinical Predictive Modelings (https://arxiv.org/abs/2510.09895)
- **What's New**: 이번 연구는 체인 오브 인플루언스(Chain-of-Influence, CoI)라는 새로운 해석 가능한 딥 러닝 프레임워크를 제안합니다. CoI는 기능 상호 작용의 명시적이며 시간에 따른 그래프를 구성하여 임상 변수 간의 영향을 추적할 수 있도록 합니다. 이 모델은 다층 주의(attention) 아키텍처를 활용하여 환자 기록에서 중요한 시간 포인트를 식별하고 이러한 포인트에서 다음 기능으로의 방향성을 모델링합니다.

- **Technical Details**: CoI 모델은 세 가지 주의 메커니즘, 즉 시간 주의, 기능 수준 주의 및 교차 기능 주의를 통합하여 시간 변화와 기능 상호 작용을 포착합니다. 입력 데이터는 배치 크기, 시간 단계 및 기능 수로 구성된 텐서 형태로 주어지며, 이 데이터를 고차원 임베딩 공간으로 투영하기 위해 학습 가능한 선형 변환이 사용됩니다. Temporal attention은 길이가 서로 다른 зависимости를 포착하기 위해 양방향 LSTM(bi-directional LSTM)을 활용하여 생성됩니다.

- **Performance Highlights**: CoI는 MIMIC-IV 데이터셋 및 사적인 만성 신장 질환 코호트를 사용하여 사망률 및 질병 진행 작업에서 기존 방법보다 예측 정확도가 신뢰성 있게 개선되었습니다. 또한 사례 연구를 통해 CoI가 다른 모델에서는 검출되지 않는 환자 별 질병 진행 패턴을 밝혀내며, 임상 의사 결정에서의 투명성을 제공함을 보여주었습니다.



### Probabilistic bias adjustment of seasonal predictions of Arctic Sea Ice Concentration (https://arxiv.org/abs/2510.09891)
- **What's New**: 이 논문에서는 편향된 모델 예측에 대해 관찰치의 조건부 분포를 매핑하기 위해 조건부 변량 오토인코더(Conditional Variational Autoencoder, cVAE) 기반의 확률적 오류 보정 프레임워크를 도입했습니다. 이 방법은 조정된 예측의 대규모 앙상블 생성을 자연스럽게 가능하게 하여 기존의 정적 보정 방안을 넘어서는데 기여합니다. 특히, 선형 회귀나 기후 평균 보정 방법에 비해 더 혁신적인 접근 방식을 제시하고 있습니다.

- **Technical Details**: cVAE는 데이터 변수 x를 보이지 않는 잠재 변수 z의 도움으로 조건부로 모델링하여 조건부 분포를 학습하는 데 사용됩니다. 이 모델은 관측된 해빙 농도(SIC) 예측의 편향을 교정하는 데 필요하며, 각 예측은 1980년 1월부터 시작된 월별 초기화를 통해 12개월 예측을 생성합니다. 논문에서는 이 방법을 통해 보다 정교한 오류 보정 및 모델의 성능 향상을 달성할 수 있음을 강조합니다.

- **Performance Highlights**: 조정된 예측은 기존의 기후 평균 조정된 예측에 비해 관찰 분포에 더 근접하고 오류가 적습니다. 이 연구에서 제안된 확률적 접근 방식은 단순한 결정론적 보정 방법보다 더 높은 신뢰성과 정확성을 제공합니다. 실험 결과는 조정된 예측이 더 잘 보정되고 관측 데이터에 대해서도 적절히 반응함을 입증합니다.



### Understanding Robust Machine Learning for Nonparametric Regression with Heavy-Tailed Nois (https://arxiv.org/abs/2510.09888)
- **What's New**: 본 연구에서는 강한 꼬리를 가진 노이즈가 존재하는 경우 비모수(nonparametric) 회귀 문제에 대한 강건(machine learning) 학습을 탐구합니다. 특히, 전통적인 일반화 오차 경계가 실제 테스트 성능을 정확히 반영하지 못하는 문제를 지적하며, 예측 오차를 통해 학습 가능성을 평가해야 한다고 주장합니다. 또한, 일반적으로 널리 사용되는 강건 손실(robust loss)인 Huber 손실을 중심으로 비모수 회귀를 예로 들어 다양한 비모수 강건 회귀 방법을 분석합니다.

- **Technical Details**: 강건 회귀는 주로 매개변수 통계(parametric statistics)에서 다루어져 왔지만, 비모수적 맥락에서는 이론적 이해가 부족합니다. 본 연구에서는 비모수 Huber 회귀를 통해 위험 최소화 접근 방식을 설명하고, 확률적 유효 가설 공간(probabilistic effective hypothesis space)을 도입하여 비한계적인 가설 공간을 다루는 방법론을 제공합니다. 또한, 새로운 비교 정리를 수립하여 여분의 강건 위험(excess robust risk)과 예측 오차 간의 관계를 규명하며, 다양한 강건 손실로 일반화 가능한 오류 경계와 수렴 속도를 도출합니다.

- **Performance Highlights**: 이 연구는 Huber 회귀의 성과를 보장하는 명시적 유한 샘플 오류 경계를 제시하며, 큰 꼬리 노이즈(heavy-tailed noise)가 존재하는 상황에서도 유효한 결과를 제공합니다. 이는 강건 학습 분석의 주요 축으로 예측 오차를 강조하며, 전통적인 과잉 일반화 위험(excess generalization risk) 개념에서 벗어난 것입니다. 또한, 본 연구는 강건 손실을 탐색하는데 원칙적인 조정 규칙을 제시하여, 실제 머신러닝 문제에서의 적용 가능성을 높입니다.



### TAWRMAC: A Novel Dynamic Graph Representation Learning Method (https://arxiv.org/abs/2510.09884)
- **What's New**: 이 논문에서는 Temporal Anonymous Walks with Restart, Memory Augmentation, 및 Neighbor Co-occurrence 임베딩을 통합한 TAWRMAC이라는 새로운 동적 그래프 표현 학습 프레임워크를 소개합니다. 기존의 연속 시간 방법들에서 발생하는 노드 특정 메모리 의존성, 이웃 노드 간의 상관 관계 미포착, 및 진화하는 그래프의 구조적 역학에 대한 포착 부족 등의 문제를 해결하고자 합니다. TAWRMAC은 향상된 임베딩 안정성과 맥락적 표현을 제공하여 현대적인 동적 그래프 학습의 한계를 극복하고자 합니다.

- **Technical Details**: TAWRMAC은 고정 시간 인코딩 방식의 GNN 기반 아키텍처를 사용하여 노드 특정 메모리와 통합하고, 이웃 간의 상관 관계를 캡처하기 위해 Neighbor Co-occurrence 임베딩을 추가합니다. 또한, Temporal Anonymous Walks with Restart 메커니즘을 도입하여 반복적 상호작용을 보이는 노드와 새로운 연결을 형성하는 노드를 구분합니다. 이러한 구조는 동적 그래프의 구조적 역학을 보다 잘 포착하고 강력한 유도 학습을 지원할 수 있습니다.

- **Performance Highlights**: 광범위한 실험을 통해 TAWRMAC은 동적 링크 예측 및 노드 분류 작업에서 최신 방법들에 비해 일관되게 우수한 성능을 보였습니다. 특히, 세 가지 다양한 음수 샘플링 전략에 따른 전이 학습(transductive) 및 유도 학습(inductive) 설정에서 강력한 예측 성능을 발휘했습니다. TAWRMAC은 안정적이며 일반화 가능하고 맥락을 인지하는 임베딩을 제공하여, 연속 시간 동적 그래프 학습의 최신 기술을 발전시키고 있습니다.



### Myopic Bayesian Decision Theory for Batch Active Learning with Partial Batch Label Sampling (https://arxiv.org/abs/2510.09877)
- **What's New**: 최근 몇 년 간, 많은 active learning acquisition functions가 제안되었지만, 어떤 것을 선택해야 할지 명확하지 않은 상황입니다. 본 연구에서는 Bayesian Decision Theory (BDT)를 기반으로 myopic framework에서 active learning을 위한 이론적 근거를 제공합니다. 이를 통해 Expected Error Reduction (EER)과 Expected Predictive Information Gain (EPIG) 같은 효과적인 알고리즘을 도출하였고, BAIT와 같은 기존 알고리즘도 BDT로부터 도출될 수 있음을 보입니다.

- **Technical Details**: 본 연구에서는 Bayesian Decision Theory를 기반으로 한 Myopic Bayesian Decision Theory를 제안하며, 이는 비용을 최소화하는 데이터 포인트 레이블 선택을 목표로 합니다. 특히, 이론적 배경을 바탕으로 Partial Batch Label Sampling (ParBaLS)을 도입하여 데이터 배치 처리 문제를 해결합니다. ParBaLS의 핵심은 샘플링된 pseudo-labels를 사용하여 점진적으로 부분 배치를 구축하는 것이며, 이를 통해 효과적인 모델 업데이트가 가능합니다.

- **Performance Highlights**: 실험 결과, ParBaLS EPIG는 다양한 데이터셋에 대해 균일한 성능을 보이며, 특히 Neural Embeddings에 대한 Bayesian Logistic Regression에서 우수한 성능을 나타냈습니다. 또한, ParBaLS는 고정된 예산 내에서 수익성 있는 기법으로 밝혀졌으며, 과거 알고리즘들보다 나은 성과를 보여 주목을 받습니다. 이러한 성과는 다양한 설정에서의 실험을 통해 검증되었습니다.



### WARC-Bench: Web Archive Based Benchmark for GUI Subtask Executions (https://arxiv.org/abs/2510.09872)
- **What's New**: 본 논문에서는 WARC-Bench (Web Archive Benchmark)라는 새로운 웹 내비게이션 벤치마크를 소개합니다. 이 벤치마크는 438개의 과제를 포함하여 멀티모달 (multimodal) AI 에이전트의 하위 작업 (subtasks)에 대한 평가를 가능하게 합니다. WARC-Bench는 웹 아카이브 파일을 사용하여 동적이고 사실적인 웹 페이지와의 샌드박스 상호작용을 지원합니다.

- **Technical Details**: WARC-Bench의 데이터 세트는 (a) WARC 기반 웹 환경, (b) 각 작업의 하위 작업 목표를 자연어로 표현한 것, (c) 각 작업에 대한 프로그래밍이 가능한 보상 함수로 구성됩니다. 이는 특정 경로에 관계없이 작업 완료를 측정할 수 있도록 하는 검증 가능한 보상 함수를 사용합니다. 다양한 UI 구성 요소에서 하위 작업을 해결하는 AI 에이전트를 평가하는 데 중점을 두고 있습니다.

- **Performance Highlights**: WARC-Bench는 현재의 대형 AI 모델에게 상당한 도전을 제공합니다. Anthropic의 Claude-4.0-Sonnet 모델이 64.8%의 성공률로 가장 높은 성과를 기록한 반면, OpenAI의 GPT-5는 51.3%의 성과를 보였습니다. 오픈 소스 모델은 Qwen-2.5VL-72B 가 37.3%로 가장 낮은 성과를 보였으며, Supervised Fine-Tuning (SFT)과 Reinforcement Learning with Verifiable Rewards (RLVR)를 통해 성과를 개선할 수 있음을 보여줍니다.



### ProxRouter: Proximity-Weighted LLM Query Routing for Improved Robustness to Outliers (https://arxiv.org/abs/2510.09852)
- **What's New**: 이 논문은 ProxRouter라는 새로운 비모수적 라우터를 제안합니다. ProxRouter는 편향(bias)과 분산(variance)을 균형있게 조정하기 위해 지수 기울기 집계 메커니즘을 적용합니다. 이로 인해 outlier 쿼리에 대한 견고성이 향상되고, 인라이어성능(inlier performance)은 최소한의 오버헤드로 유지됩니다.

- **Technical Details**: ProxRouter는 비모수적 라우팅의 통합 프레임워크를 개발합니다. 이 프레임워크에서는 정확성(accuracy)과 비용(cost) 추정을 참조 클러스터 혹은 이웃에 의한 가중 평균으로 공식화합니다. 클러스터나 이웃에 더 가까운 테스트 쿼리에 더 높은 가중치를 부여함으로써 집계가 이루어지며, 이때 편향을 줄이기 위해 지수적으로 기울어진 가중치를 적용합니다.

- **Performance Highlights**: ProxRouter는 1010개의 공용 데이터셋에서 1414개의 LLM과 쿼리를 대상으로 평가되었습니다. 실험 결과, ProxRouter는 outlier 쿼리에 대한 견고성을 크게 향상시키면서도 인라이어 성능을 보존합니다. 이는 정확성-비용 곡선(AUC)을 증가시켜 인라이어 및 아울라이어 쿼리에서 훈련된 AllSee 라우터에 근접한 성능을 보여줍니다.



### CALM: A Causal Analysis Language Model for Tabular Data in Complex Systems with Local Scores, Conditional Independence Tests, and Relation Attributes (https://arxiv.org/abs/2510.09846)
- **What's New**: 이 논문에서는 관찰 데이터를 통해 인과관계를 발견하는 기존 방법의 한계를 극복하기 위한 새로운 접근 방식인 CALM( causal analysis language model)을 소개합니다. CALM은 복잡한 시스템의 표 형식(tabular) 데이터를 처리하기 위해 설계된 혁신적인 모델로, 기존의 제약 기반 및 점수 기반 방법들이 가지고 있는 문제들을 해결하고자 합니다.

- **Technical Details**: CALM은 Mamba 기반 아키텍처를 활용하여 쌍 변수 간의 관계를 통해 인과 패턴을 분류합니다. 이 모델은 지역 인과 점수(local causal scores), 조건 독립성 테스트(conditional independence tests) 및 관계 속성(relational attributes)과 같은 포괄적인 증거를 통합하여 선형, 비선형, 조건 인과 메커니즘을 포괄적으로 포착합니다. 다양한 종류의 합성 데이터와 철저히 검증된 인과 관계를 가진 10개의 실제 생물학적 데이터셋을 통해 훈련되었습니다.

- **Performance Highlights**: CALM은 시뮬레이션 연구에서 91% 이상의 정확도를 달성하며 기존 방법보다 현저히 우수한 성능을 보여줍니다. 또한, 실제 적용에서 C형 간염 바이러스 진행 과정의 인과 요인을 식별하는데 성공하여 그 유효성을 입증했습니다. 이 연구는 언어 모델의 패턴 인식 능력을 표 형식 데이터의 복잡성에 적합하게 조정하여 정확하고 일반화 가능한 인과 발견을 위한 중요한 발전을 나타냅니다.



### Harnessing Self-Supervised Deep Learning and Geostationary Remote Sensing for Advancing Wildfire and Associated Air Quality Monitoring: Improved Smoke and Fire Front Masking using GOES and TEMPO Radiance Data (https://arxiv.org/abs/2510.09845)
Comments:
this https URL

- **What's New**: 이 연구는 NASA의 TEMPO 위성 미션에서 얻은 전례 없는 시간별 데이터와 자기 지도(Self-Supervised) 심층 학습의 발전을 활용하여 미국 서부의 산불 및 공기 질 관리 향상을 제시합니다. 이 연구는 산불 전선과 연기 기둥의 거의 실시간 시간별 확산을 매핑하는 데 있어 심층 학습의 효율성을 보여줍니다.

- **Technical Details**: 이 연구에서는 GOES-18과 TEMPO 데이터를 사용하여 연기 기둥을 구름과 성공적으로 구별하는 혁신적인 자기 지도 심층 학습 시스템을 사용합니다. 서로 다른 감지 모달리티(modalities)에서 생성된 연기 및 화재 마스크 간에 강력한 일치를 보여줍니다.

- **Performance Highlights**: 또한, 동일한 사례에 대한 운영 제품보다 상당한 개선이 이루어진 것을 강조합니다. 이를 통해 산불과 대기질 관리의 새로운 가능성이 열렸습니다.



### An Exploration of Non-Euclidean Gradient Descent: Muon and its Many Variants (https://arxiv.org/abs/2510.09827)
- **What's New**: 이 논문에서는 신경망을 위한 최적화 방법인 Muon의 이론적 기초를 강화하고, 다양한 설계 선택을 체계적으로 탐구하여 새로운 변형인 MuonMax를 개발하고 있습니다. MuonMax는 개별 레이어에 대한 핵심 원자(norm)를 활용하여 최적화를 진행하며, 업데이트 정규화를 사용하지 않는 점이 특징입니다. 이번 연구는 MuonAdam 결합과 같은 기존 알고리즘의 보다 원칙적 해석을 제공합니다.

- **Technical Details**: 논문에서 제안하는 MuonMax는 각 레이어의 핵심 원자(norm)를 통해 업데이트가 진행되며, 이는 이전의 Muon과 비교하여 적은 메모리 비용과 약간의 추가 시간(5%)만 소요됩니다. 또한, 새로운 모델 기반 모멘텀(Momo)을 통해 모든 비유클리드 경량 조정 알고리즘에 적응형 스텝 크기를 통합할 수 있는 방법을 제시합니다. 이를 통해 레이어 모멘텀의 핵심 원자를 안정적인 근사치로 사용하여 성능을 유지할 수 있습니다.

- **Performance Highlights**: MuonMax-Momo는 MuonAdam과 다른 알고리즘보다 일관되게 성능이 뛰어나며, 경쟁적인 학습률 범위를 확대하여 하이퍼파라미터 조정에 덜 민감하다는 결과를 보여줍니다. 특히, FineWeb 및 SlimPajama 데이터셋에서 774M 파라미터를 갖는 GPT 모델을 훈련하는 실험을 통해 MuonMax-Momo의 향상된 튜닝 능력을 확인할 수 있었습니다. 이로 인해 새로운 작업에서 하이퍼파라미터 조정을 줄일 수 있는 방법에 대한 실용적인 조언도 제공하고 있습니다.



### Decomposer Networks: Deep Component Analysis and Synthesis (https://arxiv.org/abs/2510.09825)
Comments:
          13 Pages, 4 figures

- **What's New**: 본 논문에서는 Decomposer Networks (DecompNet)을 제안한다. 이는 입력을 해석 가능한 여러 구성 요소로 분해하는 의미론적 오토인코더(semantic autoencoder)이다. 기존의 오토인코더가 단일 잠재 표현(latent representation)으로 입력을 압축하는 반면, DecompNet은 N개의 병렬 브랜치를 유지하여 각 브랜치에 다른 브랜치들의 재구성을 차감한 잔여 입력(residual input)을 할당한다.

- **Technical Details**: DecompNet은 가우스-자이델 스타일의 블록 좌표 경량화를 비분화(differentiable) 네트워크로 풀어내어 구성 요소 간의 명시적인 경쟁을 촉진하여 비율적으로 의미 있는 표현을 생성한다. 이 모델은 PCA(주성분 분석), NMF(비음수 행렬 분해) 및 객체 중심 모델(MONet, IODINE, Slot Attention)과의 관계를 설정하고, 첫 번째 의미론적 오토인코더로서 모든 브랜치에서 하나를 제외한 업데이트 규칙을 구현한다. 각 브랜치는 다른 브랜치가 모델링할 수 없는 요소를 모델링하도록 강제함으로써 설계적으로 의미론적 해리(semantic disentanglement)를 생성한다.

- **Performance Highlights**: DecompNet을 통해 생성된 구성 요소들은 나름대로 독립적이며, 이는 깊은 비분화(Unrolled) 방법들보다도 훨씬 더 파라미터를 적게 사용할 수 있게 한다. 제안된 모델은 입력을 선형이 아닌 의미론적 구성 요소로 분해할 수 있는 가능성을 열어주며, 각 브랜치가 재구성된 입력 데이터와 어떻게 관계가 있는지를 명확히 보여주고 있다. 결국, 이 구조는 데이터 해석 및 시그널 프로세싱(sl signal processing) 분야에서 더 나은 성능을 발휘하는 길을 제시한다.



### Temporal Lifting as Latent-Space Regularization for Continuous-Time Flow Models in AI Systems (https://arxiv.org/abs/2510.09805)
Comments:
          6 pages, 1 figure, 1 table, 1 algorithm

- **What's New**: 이번 연구에서 제안된 *temporal lifting* 방법은 연속 시간 동역학 시스템의 적응형 시간 재매개화를 위한 잠재 공간(latent-space) 공식화입니다. 이 방법은 매끄러운 단조 매핑을 통해 기반 흐름의 특이한 행동을 조절하면서 보존 법칙을 유지합니다. 특히, 이 방법은 물리적 정보를 포함한 신경망(physics-informed neural networks)의 안정성을 높이는 데 기여하며 AI 시스템에서의 잠재 흐름 아키텍처(latent-flow architectures)와 연결됩니다.

- **Technical Details**: 이론적 기초를 마련하기 위해 3-토르스(three-torus) T3에 대한 특수 연속 시간 동역학 시스템을 다룹니다. 연구에서 사용되는 표준 레베그 공간(Lebesgue spaces)과 소볼레프 공간(Sobolev spaces)을 통해, 특이한 시간에서의 점근적 수치 해의 부드러움을 보장할 수 있습니다. 이 과정에서, 적응형으로 선택된 시간 재매개화 변수는 전반적인 연속성을 회복하는 데 중요하며, 이는 기존의 시간 재매개화 방식과의 차별점이 됩니다.

- **Performance Highlights**: 이론적 결과는 256^3 푸리에 격자에서의 수치 실험을 통해 검증되었습니다. 결과적으로, 에너지 불평등(Leray–Hopf energy inequality)과 비엘–카토–마이자(BKM) 기준이 잘 보존됨을 확인하였습니다. 이는 후보자 방식의 비동적 특성을 넘어, 미래 연구를 위한 글로벌 정칙화(global regularity)에 대한 새로운 접근 방안을 열어주는 결과로 이어집니다.



### A Unified Framework for Lifted Training and Inversion Approaches (https://arxiv.org/abs/2510.09796)
- **What's New**: 이 논문은 다양한 lifted training 전략을 통합하는 통합 프레임워크를 제안하고, 이를 통해 Multi-Layer Perceptrons, Residual Neural Networks 및 Proximal Neural Networks와 같은 다양한 아키텍처들의 훈련을 지원합니다. Lifted training 방식은 대칭적인 비선형 활성화 함수 문제를 원활하게 다룰 수 있고, 분산 최적화(distributed optimisation)를 통해 효율적인 훈련을 가능하게 합니다.

- **Technical Details**: Lifted training은 보조 변수를 추가하여 최적화 문제를 고차원 제약 최적화 문제로 재구성함으로써, 안정을 높이고 다양한 비선형 활성화 함수를 사용할 수 있게 합니다. 이 과정에서 Bregman 거리(Bregman distances)를 활용하여 최적화 문제의 구조를 더 잘 파악하고, 훈련 문제를 계층 간에 자연스럽게 분해하는 추가적인 이점을 제공합니다.

- **Performance Highlights**: 이 연구는 lifted Bregman 접근법이 전통적인 훈련 방법보다 더욱 효과적이고 안정적인 결과를 제공한다는 것을 수치적으로 검증하였습니다. 특히 비선형 활성화 함수를 사용하는 아키텍처에서 더 높은 일반화 성능을 보여주며, 임의의 데이터에서 안정적인 재구성을 가능하게 하는 정규화(regularisation) 기법을 통해 복잡한 문제를 해결하고 있습니다.



### Causality $\neq$ Decodability, and Vice Versa: Lessons from Interpreting Counting ViTs (https://arxiv.org/abs/2510.09794)
- **What's New**: 이번 연구는 비전 트랜스포머(vision transformers, ViTs)에서 정보의 디코드 가능성(decodability)과 인과성(causality) 간의 관계를 탐구합니다. 특히 물체 세기를 위해 파인튜닝된 ViTs를 대상으로 하여, 각 레이어에서 토큰의 역할을 확인합니다. 연구 결과, 중간 레이어의 객체 토큰은 강한 인과적 영향을 미치지만 약한 디코드 가능성을 보이는 특이한 패턴이 발견되었습니다.

- **Technical Details**: 연구진은 activation patching을 사용해 깨끗한 이미지와 손상된 이미지 쌍의 숨겨진 활성화를 이식하여 예측에 미치는 영향을 실험합니다. 이로 인해 중간 레이어에서 객체 토큰의 활성화가 예측에 큰 영향을 미치는 방식이 확인되었습니다. 반면, 최종 레이어의 객체 토큰은 높은 디코드 가능성을 보이나 실질적인 예측 영향력이 없음을 보여주었습니다.

- **Performance Highlights**: 연구 결과는 디코드 가능성과 인과성이 상호 교환 가능한 개념이 아님을 강조합니다. 디코드 가능성과 인과성 사이의 불일치는 ViTs의 숨겨진 계산 회로를 드러내는 중요한 정보로 작용할 수 있습니다. 이러한 관점은 기계적 해석 가능성(mechanistic interpretability)을 더 깊이 이해하는 데 필요하다고 주장합니다.



### Principled Operator Learning in Ocean Dynamics: The Role of Temporal Structur (https://arxiv.org/abs/2510.09792)
Comments:
          Accepted at NeurIPS ML4PS 2025

- **What's New**: 본 논문에서는 고해상도 해양 예측을 위한 새로운 접근 방식을 제안하며, 시간 푸리에 모드를 통합하여 물리적 정확도를 향상시키는 방법을 보여줍니다. 기존의 푸리에 신경 연산자(FNO)와 변형된 버전인 FNOtD를 비교하여 물리적 동역학에 대한 예측의 일관성을 높이고 예측 안정성을 향상시키는 방법을 논의합니다. 이 연구는 특히 고주파 과정을 예측하는 데 있어 신경 연산자의 한계를 극복하고 성능을 개선하기 위한 장기적인 예측 안정성을 강조합니다.

- **Technical Details**: FNOtD 모델은 공간과 시간 모두에서 적분 커널을 공동으로 매개변수화하여 설계되어 있습니다. 이를 통해 이 모델은 다중 스케일 파동 전파를 포착하고 해양 역학을 효과적으로 학습할 수 있습니다. 또한 FNOtD는 고주파 이벤트를 학습하는 데 있어 낮은 신호 대 잡음비를 가진 데이터셋에서도 우수한 성능을 보입니다.

- **Performance Highlights**: FNOtD 모델은 표준 FNO와 비교하여 긴 예측 안정성과 물리적 동역학에 대한 우수한 일관성을 제공합니다. 이 모델은 최신 수치 해양 모델에 비해 경쟁력 있는 예측 능력을 보여주며, 낮은 계산 비용으로 이점을 제공합니다. 연구 결과, 고주파 상황에서도 보다 효과적인 예측이 가능함을 입증하였습니다.



### Combined Representation and Generation with Diffusive State Predictive Information Bottleneck (https://arxiv.org/abs/2510.09784)
- **What's New**: 이 논문에서는 고차원 공간에서 데이터 수집이 비싸고 중요한 사건이 드문 분자 과학에서, 저차원 다양체로의 압축이 다양한 후속 작업에 매우 중요하다고 강조합니다. 저자들은 분자의 중요한 표현을 특성화하기 위한 시간 지연 정보 병목(Time-Lagged Information Bottleneck)과 확산 모델(Diffusion Model)을 결합하여 D-SPIB(확산 상태 예측 정보 병목)를 제안합니다. 이 모델은 온도 정보를 통합하여 열역학의 일관되고 유용한 내부 표현을 학습할 수 있는 능력을 갖추고 있습니다.

- **Technical Details**: D-SPIB 구조는 고차원 분자 임베딩을 기반으로 분자 시스템의 상태를 예측하는 시간 지연 변형형 오토인코더(VAE)로 구성됩니다. 이 과정에서, SPIB 손실 함수는 정보 병목 원칙에 대한 변분 근사로 작용하며, 모델은 명확한 prior 분포를 사용하여 온도 종속성을 학습합니다. 모델은 또한 스코어 기반 생성 모델을 포함하여 더욱 유연하고 표현력이 뛰어난 정보 병목 prior 분포를 구축합니다.

- **Performance Highlights**: D-SPIB 모델은 여러 분자작업에서 성능을 평가하였으며, 학습 세트 외의 물리적 조건을 탐색하는 잠재력을 입증했습니다. 첫 번째 실험에서는 두 차원에서 '세 홀' 잠재력을 평가하였고, 단일 입자의 궤적을 분석하여 D-SPIB의 유용성을 보여줍니다. 이러한 방법을 통해 한정된 다중 온도 데이터를 기반으로 메타 안정 상태의 온도 의존성을 추론할 수 있습니다.



### Large Language Models for Imbalanced Classification: Diversity makes the differenc (https://arxiv.org/abs/2510.09783)
- **What's New**: 본 논문에서는 불균형(class imbalance) 문제를 해결하기 위한 새로운 LLM 기반 오버샘플링 방법인 ImbLLM을 제안합니다. 기존 SMOTE 기반 방법의 한계를 극복하고 데이터셋의 다양성을 높이기 위해, 소수 클래스의 레이블과 특징을 모두 고려한 샘플링 방식을 도입하였습니다.

- **Technical Details**: ImbLLM의 주요 기술적 기여는 세 가지입니다. 첫째, 소수 클래스의 레이블과 특징을 모두 기반으로 하는 샘플링 전략을 도입합니다. 둘째, LLM을 파인튜닝할 때 소수 샘플과 보간 샘플을 사용하여 데이터의 변동성을 더욱 풍부하게 만듭니다. 셋째, 기존의 LLM 기반 방법이 가진 한계를 극복하기 위해 특징만을 혼합하는 새로운 순열(permutation) 전략을 개발합니다.

- **Performance Highlights**: 10개의 테이블 데이터셋에 대한 광범위한 실험을 통해 ImbLLM이 8가지 최신 기술(SOTA) 기준 방법보다 우수한 성능을 보여주었고, 5개의 경우 최상의 결과를 달성하였으며 3개의 경우에서는 두 번째로 높은 성적을 기록했습니다. 생성된 샘플들은 현실적이고 다양한 품질을 가지고 있으며, 이론적 분석도 통해 생성 과정이 샘플의 다양성을 촉진한다는 것을 입증했습니다.



### Building a Foundational Guardrail for General Agentic Systems via Synthetic Data (https://arxiv.org/abs/2510.09781)
- **What's New**: 이번 연구에서는 LLM(대규모 언어 모델) 기반 에이전트의 안전성을 높이기 위해 사전 실행 단계에서의 개입을 강조합니다. 저자들은 데이터 갭(data gap), 모델 갭(model gap), 평가 갭(evaluation gap)이라는 세 가지 연구 갭을 지적하며 이를 해결하기 위한 접근법을 제시합니다. 특히, 데이터 생성 엔진인 AuraGen과 강력한 가드레일 모델인 Safiron을 제안하여 LLM이 위험한 행동을 사전에 차단할 수 있도록 합니다.

- **Technical Details**: AuraGen은 (i) 양호한 경로(trajectories) 생성, (ii) 카테고리 라벨이 있는 위험 요소 삽입, (iii) 자동화된 보상 모델을 통한 출력 필터링이라는 세 단계를 통해 높은 품질의 데이터셋을 생성합니다. Safiron은 다양한 입력 형식을 통합하는 어댑터와 compact guardian model로 구성되어 있으며, 위험한 행동을 실시간으로 식별하고 설명할 수 있는 기능을 제공합니다. 이러한 두 가지 요소는 안전성을 극대화하기 위해 효과적으로 훈련됩니다.

- **Performance Highlights**: Pre-Exec Bench라는 새로운 벤치마크를 통해 저자들은 제안한 가드레일이 강력한 기초 모델 및 비공식 기준선에 비해 뛰어난 성능을 보여준다고 보고합니다. 이 연구에서는 개입 시점에서 에이전트의 위험 행동을 성공적으로 차단하고, 이를 통해 안전하고 효과적인 에이전트 시스템을 구축할 수 있는 방향성을 모색합니다. 저자들이 제안한 프레임워크는 다양한 과제에 걸쳐 일반화될 수 있는 가능성을 지니고 있습니다.



### SVTime: Small Time Series Forecasting Models Informed by "Physics" of Large Vision Model Forecasters (https://arxiv.org/abs/2510.09780)
- **What's New**: 이 논문은 SVTime이라는 새로운 소형 모델을 소개하며, 대형 모델과 유사한 성능을 발휘하면서도 경량화된 장기 시계열 예측(Long-Term Time Series Forecasting, LTSF)을 가능하게 합니다. SVTime은 대형 비전 모델(Large Vision Models, LVMs)의 주요 유도 편향을 모방하여 설계되었으며, 이는 작은 비율의 매개변수로도 효과적인 예측이 가능하게 합니다. 이와 같은 접근 방식은 대형 모델의 환경적 영향을 최소화하고 소규모 비즈니스나 자원이 제한된 사용자들에게 더 경제적인 솔루션을 제공합니다.

- **Technical Details**: SVTime은 LVM 예측기에서 발견된 세 가지 주요 유도 편향을 기반으로 설계되었습니다: (1) 주기 간 일관성, (2) 패치별 다양성, (3) 거리에 따른 국소 주의(attention) 감소. 이러한 특성을 저장하기 위해 선형 레이어와 제약 함수를 사용하여 소형 모델을 설계하였고, 이는 기존의 기계학습 모델보다 효율적입니다. 또한, SVTime 모델은 다양한 기법을 통해 예측 기간에 대한 편향을 보완하는 백캐스트-잔여 기반 분해 프레임워크에 통합됩니다.

- **Performance Highlights**: 21개의 기준 모델과 비교했을 때 SVTime은 기존의 최첨단 경량 모델을 초월하였으며, LVM보다 1000배 적은 매개변수로도 유사한 성능을 발휘합니다. SVTime과 SVTime-t는 각각 비전 모델(Vision Model)의 0.2% 및 0.1% 크기로, 소형 모델 영역에서 뛰어난 성능을 보여주며 사전 학습된 대형 모델에 필적하는 성과를 낼 수 있음을 증명합니다. 이러한 결과는 경제적인 자원을 가진 사용자들에게 실질적인 혜택을 제공합니다.



### Why Do Transformers Fail to Forecast Time Series In-Context? (https://arxiv.org/abs/2510.09776)
Comments:
          Code: this https URL

- **What's New**: 이 논문은 시간 시계열 예측(TSF) 문제에 대한 Transformer 아키텍처의 제한 사항을 이론적으로 분석하고 있습니다. 특히 In-Context Learning (ICL) 이론을 통해 선형 모델보다 Transformer가 성능에서 더 나은 결과를 얻지 못하는 이유를 명확히 밝혀내었습니다. 실험적으로 검증된 이 발견들은자연어 처리 및 컴퓨터 비전 등 여러 영역의 Transformer 모델 설계를 수정해야 함을 시사합니다.

- **Technical Details**: 선형 자기 주의(Linear Self-Attention, LSA) 모델은 아카이브된 데이터에 대해 기대 평균 제곱 오차(Mean Squared Error, MSE) 면에서 고전 선형 모델보다 낮은 성능을 낼 수 없음을 보여주고 있습니다. 이론적으로, 맥락 길이가 무한히 증가함에 따라 LSA는 최적의 선형 예측기로 수렴할 수 있지만, 유한한 맥락 길이에서는 이 격차가 일정하게 발생함을 명시합니다. 또한, Chain-of-Thought (CoT) 스타일의 추론 하에서는 예측이 평균으로 수렴하는 경향이 발생함을 보였습니다.

- **Performance Highlights**: 실험적으로 Transformer 기반 모델들이 선형 예측기와 같은 간단한 모델에 비해 일관되게 낮은 성능을 보이는 결과를 제시합니다. 선형 변형 모델(NLinear, DLinear)은 긴 이력 데이터 예측 작업에서 Transformer를 초월하는 성과를 내며, 경량화된 방법들이 더 효과적임을 보여주었습니다. 이 논문은 TSF를 위해 Transformer 모델의 구조적 효율성에 대해 비판적으로 검토하고, 이론적인 깊이 있는 탐구 필요성을 강조하고 있습니다.



### A Generic Machine Learning Framework for Radio Frequency Fingerprinting (https://arxiv.org/abs/2510.09775)
- **What's New**: 이 논문에서는 RF 방출기의 특징을 추출하는 ML 기반 전파 지문 인식(RFF) 프레임워크를 제안합니다. 이 프레임워크는 특정 방출기 식별(SEI), 방출기 데이터 연관(EDA), 전파 방출기 클러스터링(RFEC) 등 여러 다운스트림 작업을 포함합니다. 데이터 주도형 기법을 사용하여 자동으로 정밀한 지문을 학습할 수 있어 전통적인 방법보다 우수한 성능을 보이고 있습니다.

- **Technical Details**: RFF는 방출기 신호의 특성을 기반으로 방출기 표현을 구축하는 과정입니다. 전통적인 접근 방식과 데이터 주도형 머신러닝(ML) 접근 방식이 있으며, 후자에서는 심층 신경망을 사용하여 방출기에서 입력 데이터의 특징을 자동으로 추출합니다. ML 기반 RFF로 신호 처리(Signal Processing) 기법보다 더 유연하고 다양한 신호 유형에 적용 가능하다는 장점이 있습니다.

- **Performance Highlights**: ML 기반 RFF는 기존의 신호 처리 방법에 비해 자동으로 세밀한 RF 지문을 학습할 수 있으며, 여러 다운스트림 문제를 동시에 수행할 수 있습니다. 논문에서 제시된 다양한 실제 RF 데이터 세트를 통해 몇 가지 작업과 모델 아키텍처에서 ML 기반 RFF 모델의 성능이 입증되었습니다. 이러한 접근 방식은 우주 정보 감시 및 무인 항공 시스템 대응과 같은 다양한 활용 사례에 적용 가능합니다.



### Scaling Laws and Symmetry, Evidence from Neural Force Fields (https://arxiv.org/abs/2510.09768)
Comments:
          22 pages, 10 figures

- **What's New**: 이 연구는 원자간 포텐셜(interatomic potentials) 학습의 기하학적 작업에서 동등성(equivariance)의 중요성을 강조합니다. 데이터, 매개변수 및 컴퓨팅의 견고한 전력 법칙(power-law scaling behavior)을 보여주며, 동등한 아키텍처들이 비동등한 모델보다 더 나은 스케일링 성능을 나타냄을 발견했습니다. 특히, 고차원의 표현(higher-order representations)이 더 나은 스케일링 지수를 생성하는 것으로 나타났습니다.

- **Technical Details**: 이 연구에서는 원자 시스템을 점 구름(point cloud)으로 표현하고, 포텐셜 에너지(potential energy) 및 원자 수준의 힘(forces)을 예측하는 신경망을 훈련하는 과정을 언급합니다. NNIPs(neural network interatomic potentials)는 입력으로 원자 위치와 번호를 받아 포텐셜 에너지를 예측합니다. 신경망은 상대적으로 복잡한 동등한 연산(tensor products, spherical harmonics 및 고차 메시지 전달)을 사용하지만, 이는 스케일링에서 더 나은 일반화 능력을 제공합니다.

- **Performance Highlights**: 연구 결과, 동등한 아키텍처를 적용한 메시지 전달 NNIPs는 데이터, 모델 크기 및 컴퓨팅에 대한 전력 법칙 스케일링을 따릅니다. 특히, 전력 법칙 지수는 동등성의 정도가 증가할수록 증가하며, 이는 고차 동등한 아키텍처가 비동등한 아키텍처보다 더 나은 성능을 제공함을 시사합니다. 계산 최적 스케일링(self-optimized scaling)에서는 데이터와 모델 크기가 합동으로 증가해야 하며, 이는 자연어 처리에서도 유사한 발견이 있음을 보여줍니다.



### HeSRN: Representation Learning On Heterogeneous Graphs via Slot-Aware Retentive Network (https://arxiv.org/abs/2510.09767)
- **What's New**: HeSRN은 효율적이고 표현력이 뛰어난 이질적인 그래프 표현 학습을 위한 새로운 네트워크로, 최근 Graph Transformers 기술의 한계를 극복하기 위해 개발되었습니다. 기존 모델의 제약을 해결하기 위해 slot-aware 구조 인코더를 도입하여 노드 타입 의미를 명확히 분리하고, 슬롯 정규화 및 유지 기반 융합을 통해 의미의 엉킴을 완화합니다.

- **Technical Details**: HeSRN은 구조적 및 맥락적 의존성을 선형 시간 복잡도로 모델링하는 유지 기반 인코더를 사용하여 자기 주의(attention) 메커니즘을 대체합니다. 이 네트워크는 다중 스케일 유지 층을 통해 지역 구조 신호와 글로벌 이질적 의미를 동시에 캡처할 수 있는 이질적인 유지 인코더를 추가적으로 채택합니다.

- **Performance Highlights**: HeSRN은 네 가지 실제 이질적인 그래프 데이터 세트에서 엄청난 실험을 수행하였으며, 놀라운 정확도로 기존의 최첨단 이질적 그래프 신경망 및 Graph Transformer 기준을 일관되게 초월했습니다. 이 모델은 노드 분류 작업에서 현저히 낮은 계산 복잡도로 우수한 성능을 달성합니다.



### Leveraging Shared Prototypes for a Multimodal Pulse Motion Foundation Mod (https://arxiv.org/abs/2510.09764)
- **What's New**: ProtoMM은 다양한 생리 신호(:biosignals)의 통합 및 해석을 개선하기 위한 새로운 자기 지도 학습(:self-supervised learning) 프레임워크입니다. 이 방법은 기존의 CLIP 스타일 대조 목표(:contrastive objectives) 대신, 공유된 프로토타입(:prototype) 사전을 도입하여 이질적인 모달리티를 공통 임베딩 공간(:embedding space)에서 묶습니다. 이를 통해 다중 모달 데이터의 보완 정보를 효과적으로 캡처하고, 생리 신호에 대한 일관된 "공통 언어"를 제공합니다.

- **Technical Details**: ProtoMM은 PPG 및 가속도(:accelerometry) 신호 모달리티를 결합하여 펄스 모션 기반 모델을 훈련시키는 데 초점을 맞춥니다. 이 방법은 각각의 모달리티에서 여러 개의 증강 뷰를 생성하여 서로 다른 증강의 신호를 공유된 프로토타입 벡터에 투영합니다. Multimodal Prototype Prediction 손실(:loss)을 통해 프로토타입의 확률이 다른 뷰의 프로토타입 할당(:assignment)을 예측하도록 강제하며, 이를 통해 공유 잠재 공간 내에서 필수적인 정보를 유지합니다.

- **Performance Highlights**: ProtoMM은 기존의 다중 모달 자기 지도 학습 방법론에 비해 우수한 성능을 보여주며, 특히 해석 가능성(:interpretability) 역시 향상됩니다. 실험을 통해 ProtoMM이 PPG와 가속도 데이터 모두에서 효과적으로 기능하며, 클러스터 할당의 일관성을 유지하는 방식이 정보 손실을 방지하는 데 효과적임을 입증하였습니다. 본 연구는 다섯 가지의 서로 다른 태스크를 통해 세 가지 다운스트림 데이터셋에서 ProtoMM의 전이 가능성을 깊이 있는 평가를 통해 검증합니다.



### PatentVision: A multimodal method for drafting patent applications (https://arxiv.org/abs/2510.09762)
- **What's New**: 이 논문에서는 PatentVision이라는 다중 모달 프레임워크를 소개합니다. 이 시스템은 특허 주장(claims)과 도면(drawings)을 통합하여 전체 특허 사양을 생성하는 데 중점을 두고 있습니다. 기존의 텍스트 기반 방법을 넘어서 시각적 데이터를 활용하여 정확도를 향상시키는 접근 방식을 통해 정밀한 특허 초안을 자동으로 작성할 수 있는 가능성을 제시합니다.

- **Technical Details**: PatentVision은 텍스트와 시각적 입력을 결합한 이중 입력 아키텍처를 사용합니다. 텍스트 입력은 특허 주장과 설명적인 주석으로 구성되고, 시각적 입력은 세부적인 특허 도면으로 형성됩니다. 이러한 복합적인 데이터 스트림을 통해 시스템은 발명의 전체적인 해석을 이끌어 내며, 기술적 정확성과 법적 요구사항을 모두 충족하는 사양을 생성합니다.

- **Performance Highlights**: 실험 결과, PatentVision은 오직 텍스트만 기반으로 한 방법을 초월하여 더욱 진정성 있고 일관된 출력물을 생성하는 것으로 나타났습니다. 이 시스템은 시각적 데이터를 통합하여 복잡한 디자인 특징 및 기능적 연결성을 보다 잘 표현하고, 이를 통해 더욱 풍부하고 정밀한 결과를 도출합니다. 이러한 발견은 특허 자동화에서 다중 모달 기술의 중요성을 강조하며, 지적 재산 관리 및 혁신 프로세스의 변화를 예고합니다.



### Patentformer: A demonstration of AI-assisted automated patent drafting (https://arxiv.org/abs/2510.09752)
- **What's New**: 이 논문은 Patentformer라는 인공지능(AI) 기반의 자동 특허 작성 플랫폼을 소개합니다. 이 시스템은 특허 변호사들이 법률 문서 작성 기준에 맞춰 고품질의 특허 신청서를 신속하게 생성할 수 있도록 지원합니다. Patentformer는 사용자가 제공한 특허 청구와 그림 텍스트를 입력으로 받아 작동합니다.

- **Technical Details**: Patentformer는 Transformer 기반의 대형 언어 모델(LLM)을 활용하여 특허 문서를 생성합니다. 모델은 공개적으로 사용 가능한 특허 데이터를 기반으로 특별하게 훈련되어, 특허 작성의 스타일과 구조적 관례를 학습하며, 법적 및 기술적 요구사항을 준수하는 고품질의 특허 명세서를 생성합니다.

- **Performance Highlights**: 논문에서는 Patentformer의 성능을 사용자 연구를 통해 정량적으로 평가했습니다. 그 결과, Patentformer는 고품질의 법적으로 일관된 명세서를 생성할 수 있는 능력을 입증하였습니다. 플랫폼은 현재 https://patentformer.com에서 사용할 수 있으며, 사용자 친화적인 인터페이스를 제공합니다.



### Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry (https://arxiv.org/abs/2510.09740)
Comments:
          Accepted to NeurIPS 2025 Workshop on Reliable ML from Unreliable Data

- **What's New**: 이번 논문에서는 Neural Collapse Geometry(NCAL-R) 기반의 신뢰할 수 있는 능동 학습(Active Learning) 방법론을 제안합니다. 이 방법은 불확실한 주석(annotations) 하에서도 효과적으로 샘플을 선택할 수 있도록 지원합니다. NCAL-R은 구조적 안정성과 왜곡을 평가하는 Class-Mean Alignment Perturbation 점수와, 훈련 checkpoints 간의 표현의 변동성을 포착하는 Feature Fluctuation 점수를 도입하여, 이러한 두 신호를 결합하여 샘플을 선택합니다.

- **Technical Details**: 연구에서는 pool 기반 능동 학습 설정을 고려하고, 수량화된 Class-Mean Alignment Perturbation(CMAP)와 Feature Fluctuation(FF) 점수를 사용하여 샘플의 구조적 영향을 분석합니다. NCAL-R은 높은 CMAP와 FF 점수를 가진 샘플들을 우선적으로 선택하여 일반화 오류를 감소시키고, 샘플 선택에 있어 불확실성을 줄입니다. 이 방법은 별도의 보조 네트워크나 특정 과제 조정 없이도 일반적인 백본(backbone) 및 모달리티에 적용 가능하다는 장점이 있습니다.

- **Performance Highlights**: NCAL-R은 ImageNet-100 및 CIFAR-100 데이터셋에서 이전의 능동 학습 기준선에 비해 일관되게 더 높은 정확도를 기록하며, 레이블이 부정확한 상황에서도 개선된 견고성을 보여주었습니다. 또한, NCAL-R은 배포된 실제 라벨링 파이프라인에서 신뢰할 수 있는 운영을 위한 중요한 진전을 나타냅니다. 실험 결과, 새로운 분포(out-of-distribution) 데이터에 대한 일반화 능력 또한 향상되었음을 보여주었습니다.



### Machine learning methods fail to provide cohesive atheoretical construction of personality traits from semantic embeddings (https://arxiv.org/abs/2510.09739)
Comments:
          1 figure, 12 pages

- **What's New**: 이번 연구에서는 언어에 포함된 성격 특성을 바탕으로 한 'Lexical Hypothesis'를 검토하였습니다. 기계 학습(machine learning)을 활용하여 고전 형용사 목록에서 기본적인 성격 모델을 생성하고, 이를 Big Five 모델과 비교하였습니다. 특히, Reddit에서 한 백만 개의 댓글을 분석하여 온라인 커뮤니티의 성격을 어떻게 설명할 수 있는지를 살펴보았습니다.

- **Technical Details**: 연구에서는 Rosemary E. V.와 같은 클래식한 형용사 리스트를 기반으로 한 성격 모델링을 실시하였습니다. 기계 학습 기법을 이용해 댓글을 분석한 결과, Big Five 요소 중 Agreeableness, Conscientiousness, Neuoticism이 특히 강력하고 해석 가능한 설명을 제공하는 것으로 나타났습니다. 반면, 기계 학습을 통한 클러스터링은 의미 있는 구분을 제공하지 못했으며, Extraversion 특성을 회복하는 데 실패했습니다.

- **Performance Highlights**: 연구 결과는 Big Five 모델의 강인성을 확인시키며 성격의 의미론적 구조가 상황 의존적일 수 있음을 암시합니다. 기계 학습이 기존 심리학 이론의 생태학적 타당성(ecological validity)을 점검하는 데 도움이 될 수 있지만, 이러한 이론을 대체할 수는 없는 것으로 나타났습니다. 이는 심리학적 이론의 중요한 역할을 다시 한번 강조하는 결과입니다.



### InterCorpRel-LLM: Enhancing Financial Relational Understanding with Graph-Language Models (https://arxiv.org/abs/2510.09735)
- **What's New**: 이 논문에서는 InterCorpRel-LLM이라는 새로운 GNN(그래프 신경망)과 LLM(대형 언어 모델) 통합 프레임워크를 제안합니다. 이 프레임워크는 FactSet 공급망 데이터에서 파생된 독점 데이터셋과 함께 세 가지 맞춤형 훈련 작업을 통해 공급 관계 식별 문제를 해결합니다. 이를 통해 구조와 의미적 요소를 효과적으로 모델링할 수 있습니다.

- **Technical Details**: InterCorpRel-LLM은 회사 그래프 매칭, 산업 분류 및 공급 관계 예측의 세 가지 상호 보완적인 작업을 통해 기업 간 관계를 모델링합니다. 이 방법은 GNN과 LLM의 장점을 통합하며, 7B 매개변수의 가벼운 백본 모델을 사용하여 적은 훈련으로도 효과적인 성능을 발휘합니다. 특히, 의미와 구조를 함께 고려하여 복잡한 비즈니스 관계를 이해할 수 있습니다.

- **Performance Highlights**: 실험 결과, InterCorpRel-LLM은 공급 관계 식별 작업에서 F-score 0.8543을 기록하여 GPT-5의 0.2287을 크게 초월했습니다. 이 모델은 경쟁자 식별 작업에서도 향상된 성능을 보였으며, 제로-샷 경쟁자 검출을 가능하게 했습니다. 이러한 결과는 특정 도메인에 맞춘 그래프-언어 융합이 효과적이며, 대규모 모델 최적화 없이도 효율적이고 정확한 기업 간 관계 모델링을 가능하게 한다는 것을 확인시켜 줍니다.



### ARROW: An Adaptive Rollout and Routing Method for Global Weather Forecasting (https://arxiv.org/abs/2510.09734)
Comments:
          16 pages, 6 figures, conference

- **What's New**: 이 연구에서는 ARROW라는 새로운 방법을 제안하여 글로벌 기상 예측의 문제점을 해결하고 있습니다. 기존의 데이터 기반 예측 방법들은 고정된 짧은 시간 간격에서 대기 역학을 모델링하지만, ARROW는 다층적 시간 예측 모델을 통해 이 문제를 극복하려 합니다. 또한, 강화 학습을 통한 적응형 롤아웃 스케줄러를 도입하여 기상 상태에 따라 가장 적합한 예측 시간 간격을 선택할 수 있습니다.

- **Technical Details**: ARROW는 멀티 간격 예측 모델(MIFM)을 사용하여 다양한 시간 간격에서의 예측을 가능케 합니다. MIFM 내의 Shared-Private Mixture-of-Experts(S&P MoE)는 대기 역학의 공유 패턴 및 특정 특성을 포착하고, Ring Positional Encoding(RPE)을 통해 지구의 원형 위도 구조를 정확하게 인코딩합니다. 이는 다양한 시간 간격의 예측을 통해 열리기 급변하는 대기 행동을 효과적으로 모델링할 수 있게 돕습니다.

- **Performance Highlights**: 실험 결과, ARROW는 글로벌 기상 예측에서 최첨단 성능을 달성하였고, RMSE(root mean square error) 및 ACC(accuracy)에서 약 10%의 전반적인 향상을 보였습니다. 이는 적응형 롤아웃 전략과 멀티 간격 예측 모델의 통합이 상당한 성과로 이어졌음을 보여줍니다. 따라서 ARROW는 향후 기상 예측 분야의 유망한 패러다임을 형성할 것으로 기대됩니다.



### Evaluating LLM-Based Process Explanations under Progressive Behavioral-Input Reduction (https://arxiv.org/abs/2510.09732)
Comments:
          12 pages, 2 figures, 3 tables; to appear in Enterprise Design, Operations, and Computing. EDOC 2025 Workshops, Lecture Notes in Business Information Processing (LNBIP), Springer, 2025. Part of 29th International Conference on Enterprise Design, Operations, and Computing (EDOC)

- **What's New**: 이 논문에서는 대규모 언어 모델(LLM)을 사용하여 사건 로그(event logs)에서 발견된 프로세스 모델의 텍스트 설명을 생성하는 새로운 접근 방식을 탐구합니다. 특히, 프로세스 모델을 구성하는 입력 데이터의 크기를 점진적으로 줄여가며 설명 품질을 평가합니다. 이를 통해 컴퓨터 자원이 제한된 환경에서도 효과적으로 프로세스 분석을 수행할 수 있는 방법을 제시합니다.

- **Technical Details**: 논문에서는 LLM을 활용해 진행하는 프로세스 모델 발견을 위해 데이터 효율성을 높이기 위한 네 가지 단계를 가진 파이프라인을 소개합니다. 이 파이프라인은 특정 로그의 사건(event) 수를 줄이고, 이를 통해 생긴 모델에서 설명을 생성한 후, 설명의 완전성(completeness), 병목(bottleneck) 식별, 개선 요청 등을 평가합니다. 결국, 행동 정보의 양을 줄이는 것이 LLM 기반 프로세스 설명의 품질에 미치는 영향을 정량화하고자 합니다.

- **Performance Highlights**: 실험 결과, 적당한 수준의 사건 수를 줄일 때 설명 품질이 크게 저하되지 않음을 보여줍니다. 이는 계산 비용과 설명 품질 사이의 실용적인 균형을 이루는 경로를 제시하며, 리소스가 제한된 상황에서도 유용한 통찰을 제공합니다. 특히, 실험은 합성 로그(synthetic logs)를 사용하여 수행되었으며, 추후 연구에 대한 방향성을 제시합니다.



### It's 2025 -- Narrative Learning is the new baseline to beat for explainable machine learning (https://arxiv.org/abs/2510.09723)
Comments:
          18 pages, 5 figures

- **What's New**: 최근의 논문에서는 Narrative Learning이라는 새로운 방법론을 소개합니다. 이 방법론은 모델을 전체적으로 자연어로 정의하고, 전통적인 수치 최적화 대신 설명 프롬프트를 사용하여 분류 기준을 반복적으로 수정합니다. 우리는 이 접근법의 정확성과 잠재력을 평가하기 위해 6개 데이터셋을 사용했으며, 기존의 7가지 설명 가능한 기계 학습 모델과 비교하여 많은 데이터셋에서 높은 정확도를 달성했습니다.

- **Technical Details**: Narrative Learning은 감독된 이진 분류 알고리즘으로, 라벨이 붙은 데이터를 훈련, 검증, 테스트 데이터로 나누어 사용합니다. 두 개의 언어 모델인 Overseer와 Underling이 사용되며, Overseer는 자연어 프롬프트로 분류 지침을 생성하고, Underling은 이를 평가하여 분류 결과를 반환합니다. 이 과정은 반복적으로 진행되며, 모델의 정확도를 향상시키기 위해 Overseer가 프롬프트를 수정하고 다시 Underling에 전달하는 방식으로 진행됩니다.

- **Performance Highlights**: 실험 결과, Narrative Learning은 6개 데이터셋 중 5개에서 기존의 설명 가능한 기계 모델보다 더 높은 정확도를 기록했습니다. 특히, KT 정확도와 같은 지표를 사용하여 성능을 평가했으며, 이는 대규모 데이터셋에서의 성능 향상에 초점을 맞추었습니다. 추가적으로, Lexicostatistics 트렌드도 보고하여 모델의 설명 가능성에 대한 이해를 높였습니다.



### ICL-Router: In-Context Learned Model Representations for LLM Routing (https://arxiv.org/abs/2510.09719)
- **What's New**: 이번 논문은 기존의 모델 라우팅 기법의 한계를 극복하기 위해 ICL-Router라는 새로운 방법을 제안합니다. 이 방법은 모델의 능력을 표현하기 위해 in-context vectors를 활용하며, 새로운 모델의 통합을 빠르게 진행할 수 있습니다. 기존의 방법들이 라우터 재훈련을 필요로 했던 반면, ICL-Router는 이 과정을 생략하게 합니다.

- **Technical Details**: ICL-Router는 쿼리-성능 쌍의 벡터 표현을 통해 모델의 능력을 특징짓는 새로운 접근 방식을 제공합니다. 이 방법은 두 단계로 진행되며, 첫 번째 단계에서는 쿼리를 벡터로 임베딩하고 원래 쿼리를 재구성하도록 훈련된 LLM 기반 라우터를 사용합니다. 두 번째 단계에서는 각 후보 모델을 쿼리 세트에서 평가하여 성능과 벡터 표현을 결합하여 모델의 능력 프로필을 형성합니다.

- **Performance Highlights**: ICL-Router는 10개의 다양한 벤치마크에서 최첨단 라우팅 성능을 달성했습니다. 특히, in-distribution 작업에서 기존 최고 성능의 모델보다 절대 7.2 포인트 높은 성능을 기록했으며, 스케일링 할 때의 이점을 통해 더 많은 모델이 추가될수록 성능이 향상됨을 보여주었습니다. 이러한 방법은 기존의 EmbedLLM 및 RouterDC와 비교하여 더 뛰어난 유연성과 이점을 제공합니다.



### Federated k-Means via Generalized Total Variation Minimization (https://arxiv.org/abs/2510.09718)
- **What's New**: 본 논문은 연결된 기기가 로컬 데이터셋을 공유하지 않고도 전체 데이터를 클러스터링할 수 있는 문제인 federated clustering에 대해 다룹니다. 특히, k-means 원칙에 기반한 하드 클러스터링을 중점적으로 살펴보며, 이를 GTVMin을 통한 문제로 수학적으로 모델링하였습니다. 새로운 알고리즘은 각 기기가 수정된 로컬 k-means 문제를 해결함으로써 로컬 클러스터 중심을 업데이트할 수 있도록 합니다.

- **Technical Details**: 이 알고리즘은 연결된 기기 간 클러스터 중심의 불일치를 측정하는 페널티 항을 추가하는 방식으로 수정됩니다. 이는 기존의 중앙 집중식 k-means 방법과는 달리 로컬 데이터셋을 묶어 클러스터링을 수행하는데, 데이터 보호를 위해 집계된 정보만 공유하는 프라이버시 친화적인 방식입니다. GTVMin의 한 예로서, 연결된 기기들에 대해 각각의 클러스터 중심을 만드는 데 중점을 두고, 기기 간의 상호 연결성을 반영합니다.

- **Performance Highlights**: 우리의 방법은 서로 다른 데이터 분포를 가진 로컬 데이터셋에 적합하며, 연결된 기기들을 기반으로 클러스터 중심을 따로 산출함으로써 다양한 상황에서 유용성을 제공합니다. 이 접근은 통계적으로 이질적인 데이터셋에서도 최적화된 성능을 보여주며, 기존의 방법들과 비교했을 때 훨씬 효율적인 분산 학습을 가능하게 합니다.



### High-Power Training Data Identification with Provable Statistical Guarantees (https://arxiv.org/abs/2510.09717)
- **What's New**: 이 논문은 Provable Training Data Identification (PTDI)라는 새로운 방법을 제안하여, 훈련 데이터 식별 과정에서 엄격한 false discovery rate (FDR) 통제가 가능하다는 것을 증명합니다. 이 방법은 p-value를 계산하여 각 데이터 포인트의 사용 비율을 추정함으로써, 최종 훈련 데이터 세트를 선택하게 됩니다. 이러한 접근 방식은 기존 방법들이 가진 한계를 해소하고, 실질적이고 강력한 통계적 보장을 제공합니다.

- **Technical Details**: PTDI는 데이터 포인트에 대한 detection score를 계산하고, 이를 통해 p-value를 생성하는 비훈련 calibration 세트를 사용하는 분포 자유 방법으로 구성됩니다. 이 논문은 Benjamini-Hochberg 절차를 사용하여 p-value를 스케일링한 후, 그 결과로 얻어진 스케일링된 p-value가 최종 훈련 데이터를 식별하는 데 사용됩니다. 이러한 준비 과정은 이론적으로 엄격한 FDR 통제를 보장하며, 흑박스(black-box) 및 백박스(white-box) 환경 모두에서 적용 가능하다는 장점이 있습니다.

- **Performance Highlights**: PTDI는 광범위한 모델과 데이터 세트에 걸쳐 고성능 실험 결과를 보여줍니다. 예를 들어, WikiMIA에서 5%의 목표 FDR을 설정했을 때 PTDI는 4.94%의 실험적 FDR을 기록하며, 기존 방법인 Hu et al. (2025)의 13.11%와 비교하여 우수한 성능을 입증합니다. 전반적으로 PTDI는 FDR을 엄격히 통제하며, 향상된 검출 성능을 달성하고 있습니다.



### Group-Adaptive Adversarial Learning for Robust Fake News Detection Against Malicious Comments (https://arxiv.org/abs/2510.09712)
Comments:
          10 pages, 12 figures

- **What's New**: 이번 논문은 온라인에서의 가짜 뉴스 탐지(Fake News Detection, FND) 모델이 사용자 댓글(Comments) 및 대형 언어 모델(Large Language Models, LLMs)이 생성한 댓글에 의해 취약해질 수 있음을 지적합니다. 특히, 댓글 공격에 대한 포괄적인 평가를 제시하고, 이를 통해 FND 모델의 강건성을 향상시키기 위한 그룹 적응형 적대적 훈련 전략(Group-Adaptive Adversarial Training Strategy)을 소개했습니다. 이 방법은 심리학적 원리에 기반한 세 가지 댓글 카테고리로 공격을 분류하여 다각적인 적대적 훈련을 실시합니다.

- **Technical Details**: 제안된 방법은 세 가지 단계로 구성됩니다: 먼저, 공격을 세 가지 심리학적으로 기반을 둔 카테고리로 나눕니다: 지각적(Perceptual), 인지적(Cognitive), 사회적(Societal). 둘째, LLMs를 활용하여 각 카테고리 당 다양한 공격을 생성하며, 셋째, Dirichlet 분포 기반의 적응형 샘플링 메커니즘(InfoDirichlet Adjusting Mechanism)을 적용하여 훈련 중 각 댓글 카테고리의 학습 집중도를 동적으로 조정합니다. 이러한 접근은 다양한 댓글 공격에 대한 강건성을 유지하면서도 강력한 탐지 정확도를 유지하는 것으로 나타났습니다.

- **Performance Highlights**: 실험 결과, RumourEval-19, Weibo16 및 Weibo20와 같은 기준 데이터셋에서 기존 모델들이 제안된 댓글 공격에 의해 상당한 성능 저하를 겪는 것을 보여주었습니다. 반대로, 우리 제안된 프레임워크는 이러한 댓글 공격에 대해 저항할 수 있었고, 최첨단 기법들보다 더 나은 성능 개선을 달성하였습니다. 이는 가짜 뉴스 탐지에서 댓글의 다양성과 그 공격에 대한 강건성을 높이는 방법으로 제시되고 있습니다.



### A Multi-Component Reward Function with Policy Gradient for Automated Feature Selection with Dynamic Regularization and Bias Mitigation (https://arxiv.org/abs/2510.09705)
- **What's New**: 이 논문에서는 편향(bias) 완화와 자동 특징 선택(feature selection)을 단일 학습 과정으로 통합하는 강화 학습(Reinforcement Learning, RL) 기반의 새로운 프레임워크를 제안합니다. 전통적인 휴리스틱 기반의 접근과는 달리, RL 에이전트는 예측 성능(predictive performance)과 공정성(fairness) 고려를 명시적으로 통합하는 보상 신호를 사용하여 적응적으로 특징을 선택합니다. 이 동적 접근 방식은 훈련 과정 전반에 걸쳐 일반화(generalization), 정확성(accuracy), 공정성(equity)의 균형을 유지할 수 있도록 합니다.

- **Technical Details**: 제안된 모델은 보상을 기반으로 하여 편향 역학(bias dynamics)을 더 잘 파악하기 위해 보상 신호를 보강한 전형적인 마르코프 의사결정 과정(MDP)을 기반으로 합니다. 에이전트는 선택된 특징 집합의 특정 부분 집합을 나타내는 상태 변수를 통해 작동하며, 특징의 포함 또는 제외를 결정하는 행동을 취합니다. 보상 기능은 특정 특징이 선택될 때 부여되는 직접적인 벌칙(penalty)과 간접적인 벌칙을 결합하여 모델의 예측 정확도를 반영합니다.

- **Performance Highlights**: 모델은 신용 위험 데이터셋에 적용되어 대출 연체를 예측하는 데 사용되었으며, 통계적 메트릭(Receiver Operating Characteristic, ROC 결과 곡선 및 bias score plot)을 활용한 벤치마킹에서 모든 기준 모델을 초과하는 성능을 보였습니다. 에이전트의 학습 행동을 검토한 결과, 상태-행동 값 열지도를 통해 모델의 정책 수렴(policy convergence)을 검토한 결과, 안정적인 수렴이 증명되었습니다. 이 방식은 편향된 특징이 사회적 결과에 미치는 영향을 줄일 수 있는 원칙 기반의 기제를 제공합니다.



### Operator Learning for Power Systems Simulation (https://arxiv.org/abs/2510.09704)
- **What's New**: 이 논문은 재생 가능 에너지가 많이 통합된 전력 시스템의 안정성 및 동적 성능을 향상시키기 위한 시간 도메인 시뮬레이션의 필요성을 강조합니다. 특히, 1-50 마이크로초의 짧은 시뮬레이션 시간 간격을 요구하는 재생 가능 에너지의 초고속 동적 현상을 캡처하는 데 필요한 솔루션을 제시합니다. 본 연구는 시간 단계 불변성(time step invariance) 개념을 탐구하여, 조대 시간 단계에서 학습된 모델이 미세 해상도 동적에 일반화될 수 있도록 합니다.

- **Technical Details**: 연구에서는 세 가지 operator learning 방법을 벤치마킹하며, 모델들이 시뮬레이션의 전제를 어떻게 일반화하는지를 평가합니다. 특히, Deep Operator Networks (DeepONets), Fourier Neural Operators (FNOs), Latent Neural ODEs (LNODEs) 방법들이 소개되며 각각의 알고리즘적 및 아키텍처적 설계가 어떻게 불변성을 가능하게 하는지를 설명합니다. 이 모델들은 단순화된 테스트 시스템에서의 동적 예측을 통해 고차원 공간 간의 매핑을 근사화합니다.

- **Performance Highlights**: 실험적으로, 참여한 모델들은 제로샷(super-resolution) 일반화 및 안정적이고 불안정한 동적 상태 사이의 일반화를 평가받습니다. 본 논문은 새로운 접근 방식으로 재생 가능 에너지의 안정적인 통합을 위한 잠재력을 보여주며, 시뮬레이션의 계산적 복잡성을 줄이기 위한 대안으로서 machine learning의 활용 가능성을 제시합니다. 이 연구는 기후 변화 완화를 위한 전력 시스템 모델링의 머신 러닝 적용의 첫 단계를 의미합니다.



### Vanishing Contributions: A Unified Approach to Smoothly Transition Neural Models into Compressed Form (https://arxiv.org/abs/2510.09696)
Comments:
          Code available at this https URL

- **What's New**: 이 논문에서는 Deep Neural Networks (DNNs)의 압축 기술에 대한 새로운 접근법인 Vanishing Contributions (VCON)을 제안합니다. DNN은 뛰어난 성능을 보이지만, 높은 계산 자원 소모로 인해 작고 효율적인 모델이 필요합니다. VCON은 원래 모델과 압축 모델을 병렬로 실행하면서 점진적으로 원래 모델의 기여를 감소시키고 압축 모델의 기여를 증가시켜 안정적인 전환을 돕습니다.

- **Technical Details**: VCON 방법은 기존의 프루닝(pruning), 양자화(quantization), 그리고 저랭크 분해(low-rank decomposition) 기술들과 결합하여 사용될 수 있습니다. 이 방법은 원래의 네트워크를 직접적으로 대체하는 것이 아니라 두 모델의 기여를 조절하여 부드러운 전환을 가능하게 합니다. 실험에서는 컴퓨터 비전과 자연어 처리에서 VCON의 효과를 평가하며, 모든 시나리오에서 성능이 개선되는 것을 확인하였습니다.

- **Performance Highlights**: VCON을 사용한 결과, 평균적으로 3%를 초과하는 성능 개선이 이루어졌으며, 일부 구성에서는 20%의 정확성 향상도 관찰되었습니다. 이러한 결과는 VCON이 다양한 압축 기술에 적용될 수 있는 일반화 가능한 방법임을 보여주며, 여러 벤치마크에서 일관된 성과를 거두었습니다. 따라서, VCON은 기계 학습 모델의 압축을 위한 효과적인 도구로 자리 잡을 가능성이 큽니다.



### Kelp: A Streaming Safeguard for Large Models via Latent Dynamics-Guided Risk Detection (https://arxiv.org/abs/2510.09694)
- **What's New**: 본 논문에서는 Kelp라는 새로운 플러그인 프레임워크를 제안하여 대형 모델(large models) 내에서 스트리밍 위험 감지를 가능하게 합니다. Kelp는 Streaming Latent Dynamics Head (SLD)를 활용하여 생성된 시퀀스를 통해 리스크의 시간적 진화를 모델링하며, 실시간 위험 감지의 정확도를 향상시킵니다. 또한, Anchored Temporal Consistency (ATC) 손실을 도입하여 안전하다고 판단되는 예측을 강제하여 실제 애플리케이션에서도 신뢰할 수 있는 모니터링을 보장합니다.

- **Technical Details**: 기존 ‘스트리밍 리뷰어’는 일반적으로 정적 데이터셋에 대해 사후 평가(post hoc evaluation)가 이루어지며, 보호 모델의 생성에 참여하지 않아 안전 품질을 저하할 수 있습니다. 본 연구에서는 각 샘플이 평가할 특정 모델에 의해 생성되는 StreamGuardBench라는 벤치마크를 소개하여 стрим링 가드레일의 실시간 평가를 가능하게 합니다. 데이터 소스는 WildGuard, S-Eval, MMSafetyBench, FigStep와 같은 안전 관련 데이터셋을 기반으로 하여 폭넓은 리스크 도메인을 포함합니다.

- **Performance Highlights**: Kelp는 20M의 파라미터만으로도 기존의 최신 SOTA(상태-최고) 시스템보다 뛰어난 성능을 보이며, 1,024개의 생성된 토큰에 대해 평균 15.61% 높은 F1 점수를 기록했습니다. 부가적으로, Kelp는 0.5ms 미만의 지연시간으로 실시간의, 토큰 단위의 위험 감지를 수행하여 사용자 경험을 저해하지 않으면서도 안전성을 극대화합니다. 이로써 Kelp는 다양한 모델 및 작업에 대해 우수한 탐지와 개입 결과를 보여줍니다.



### Neural PDE Solvers with Physics Constraints: A Comparative Study of PINNs, DRM, and WANs (https://arxiv.org/abs/2510.09693)
Comments:
          50 pages, 13 figures

- **What's New**: 이 논문에서는 기계학습 및 신경망 기반의 새로운 PDE (Partial Differential Equations) 해결 방법을 제시합니다. 전통적인 메쉬 기반 솔버보다 효율적이며, 5차원 Poisson 문제 및 1D/2D 시간 독립 슈뢰딩거 방정식을 포함한 범위에서 세 가지 신경 PDE 솔버인 PINNs (Physics-Informed Neural Networks), DRM (Deep Ritz Method), WANs (Weak Adversarial Networks)의 성능을 비교합니다. 이를 통해 각 방법의 강점과 약점을 분석하고 실용적인 가이드라인을 제공합니다.

- **Technical Details**: 도메인 모델링에 있어 PDE는 물리학, 공학 등 여러 분야에 걸쳐 광범위하게 사용됩니다. 전통적인 수치적 방법들은 특정한 경우에만 해를 제공할 수 있으며, 메쉬 생성을 포함한 고차원 문제에 대한 해결 방법은 비용이 많이 듭니다. 본 연구는 신경망을 이용한 메쉬 없는 방법으로 PDE를 해결하는 접근 방식을 탐구하며, 세 가지 주요 방법론에 대한 체계적인 비교와 개선 사항을 지속적으로 논의합니다.

- **Performance Highlights**: 제안된 방법들은 강제 경계 조건 (FBC), 강제 노드 (FN), 그리고 직교성 정규화 (OG)를 통해 낮은 오류를 도출하며, PINNs는 정확성 및 스펙트럼 복구면에서 가장 신뢰할 수 있는 방법으로 평가되었습니다. DRM은 정적 문제에서 최적의 정확도-시간 비율을 제공하는 반면, WAN은 약한 제약 조건과 FN/OG를 효과적으로 사용할 때 경쟁력 있는 성능을 보입니다. 모든 방법들은 5000-10,000 epochs에서 가장 큰 성능 향상을 보여, 기존 메쉬 기반 접근 방식에 비해 높은 유연성과 효율성을 제공합니다.



### Evaluation of Differential Privacy Mechanisms on Federated Learning (https://arxiv.org/abs/2510.09691)
Comments:
          Supervised by Prof. Dr.-Ing. habil. Alois C. Knoll; Advisor: Nagacharan Teja Tangirala, this http URL

- **What's New**: 이번 논문에서는 연합 학습(Federated Learning)에서 모델의 훈련 데이터가 공개되지 않도록 하는 새로운 접근방법을 제시합니다. 특히, Differential Privacy (DP) 기술을 이용하여 모델 업데이트 시 노이즈를 추가하고, 이를 통해 데이터의 민감성을 보호하려고 합니다. 그러나 기존의 접근법은 고정된 프라이버시 예산(privcy budget)에 의존하여 과도한 노이즈를 추가할 수 있으며, 이는 성능에 부정적인 영향을 미칠 수 있습니다. 본 연구에서는 이 문제를 해결하기 위해 적응형 프라이버시 예산(adaptive privacy budgets)을 도입하였습니다.

- **Technical Details**: 본 연구는 Laplace와 Gaussian 메커니즘을 사용하여 적응형 프라이버시 예산을 갖춘 DP 방법을 구현합니다. 또한, Gaussian 메커니즘에 적응형 클리핑(adaptive clipping) 접근법을 적용하여 모델의 그래디언트(gradient)가 고정된 민감도(sensitivity)가 아닌 동적으로 업데이트 되도록 하였습니다. 실험은 IID 및 비IID(non-IID) 데이터셋을 활용해 다양한 프라이버시 예산에서 실시되었으며, 매 라운드마다 선택된 클라이언트 수를 변화시켰습니다. 이를 통해 프라이버시를 유지하는 동시에 모델 정확도를 높일 수 있는 방안을 검증하였습니다.

- **Performance Highlights**: 본 논문에서 실시된 실험은 200개의 훈련 라운드로 제한되었지만, 결과는 적응형 프라이버시 예산과 클리핑 방법이 모델의 정확도를 유지하면서도 프라이버시를 효과적으로 보호할 수 있음을 보여줍니다. 이러한 연구 결과는 향후 연합 학습 환경에서 데이터 프라이버시 및 성능 간의 균형을 이루는 데 중요한 기초 자료로 활용될 것입니다. 향후 더 많은 실험을 통해 이 기법의 성능을 추가로 개선할 수 있을 것으로 기대됩니다.



### On the Occurence of Critical Learning Periods in Neural Networks (https://arxiv.org/abs/2510.09687)
Comments:
          8 pages, 8 figures

- **What's New**: 이 연구는 신경망의 가소성(plasticity)에 대한 실증적 지원을 제공하여, 학습 하이퍼파라미터를 간단히 조정함으로써 중요한 학습 기간(critical learning periods)과 웜 스타트(warm-starting)에서 발생하는 성능 손실을 회피할 수 있음을 보여줍니다. 특히, 우리는 주어진 데이터의 부분 집합으로 훈련을 시작하는 경우 가소성이 감소하는 현상이 발생한다는 기존 연구를 바탕으로 실험 범위를 확장하여 이 문제를 분석합니다. 또한, 주기적인 학습률 일정(cyclic learning rate schedule)을 통해 성능 저하 문제를 거의 완전히 회피할 수 있음을 입증했습니다.

- **Technical Details**: 저자들은 Achille et al.의 실험 설정을 기반으로 ResNet-18 아키텍처를 사용하여 CIFAR-10 데이터셋에서 실험을 수행합니다. 초기 학습률은 0.1로 설정하고, 매 에포크마다 0.97 배율로 기하급수적으로 감소합니다. 실험 결과는 중요한 학습 기간 효과를 재현하며, 초기 모델이 청정 데이터로 훈련될 때와 비교하여 부정적인 영향을 미친 블러 결함을 포함한 다양한 실행 조건에서 성능 저하를 보여주었습니다.

- **Performance Highlights**: 연구 결과, 학습률 조정이 초기 훈련에서의 성능 감소를 크게 완화할 수 있음을 확인했습니다. 학습률이 초기 값으로 재설정될 때, 웜 스타트 설정에서 발생한 성능 격차가 완전히 사라졌으며, 중요한 학습 기간과 관련된 저하도 현저히 감소했습니다. 더 높은 재시작 학습률 값은 모델 혁신적으로 회복할 수 있는 경향이 있으며, 이는 크게 조건에 영향을 받는 경향이 나타났습니다.



### Deep Neural Networks Inspired by Differential Equations (https://arxiv.org/abs/2510.09685)
Comments:
          35 Pages, 3 figures

- **What's New**: 이번 논문에서는 심층 신경망(deep neural networks) 아키텍처와 확률적 동적 모델링 방법을 미분 방정식(differential equations)의 관점에서 종합적으로 리뷰합니다. 특히, 일반 미분 방정식(ODEs)와 확률적 미분 방정식(SDEs)을 기반으로 한 모델을 분석하고 이들의 특성과 성능을 비교합니다. 이러한 접근 방식은 해석 가능성(interpretability)과 일반화(generalization) 성능 향상에 기여할 새로운 연구 방향을 제안합니다.

- **Technical Details**: 본 논문에서는 ODE와 SDE 기반의 신경망 디자인 방법론을 체계적으로 정리하며, ODE를 통해 신경망 아키텍처를 이해하고 설계하는 데 필요한 이론적 틀을 제시합니다. ODE 기반의 모델은 예를 들어 ResNet과 같은 연속적인 진화(evolution)를 통해 물리적 직관(intuition)과 해석 가능성을 개선합니다. SDE는 확률적 시스템을 도입함으로써 경로의 변동성을 추가하여 다양한 분포적 변화를 포착하고, 데이터 생성 분야에서도 중요한 역할을 수행합니다.

- **Performance Highlights**: 연구에서 제안하는 모델들은 여러 실험 작업에서 DE 기반 모델들의 성능을 비교하면서 우수한 해석 가능성과 일반화 능력을 보여줍니다. ODE와 SDE를 통한 규제 기법(stochastic regularization methods)은 네트워크 구조와 상호작용하여 성능을 더욱 향상시킵니다. 저자들은 앞으로 신경망에서의 무작위성(randomness)의 역할을 더 깊이 탐구해야 한다고 강조하며, 이러한 접근이 심층 학습의 미래 발전에 중요한 기여를 할 것이라고 전망합니다.



### Using LLMs to Directly Guess Conditional Expectations Can Improve Efficiency in Causal Estimation (https://arxiv.org/abs/2510.09684)
- **What's New**: 이 논문에서는 LLM(대규모 언어 모델) 기반 AI 도구를 활용하여 인과 추론에서의 원인 추정(causal estimation)을 개선하는 방법을 제안합니다. 특히, 고차원(confounder) 혼란 요인을 고려하는 이중 머신 러닝(double machine learning)에서, LLM이 훈련된 역사적 데이터를 바탕으로 만들어낸 예측을 활용하여 이것이 어떻게 추정의 정확성을 향상시킬 수 있는지를 다룹니다. 이를 통해 역사적 지식(historical knowledge)과 추론 능력(reasoning capacity)을 이용하여 고차원의 저주(curse of dimensionality) 문제를 극복할 수 있음을 보여줍니다.

- **Technical Details**: 연구에서는 LLM을 활용하여 조건적 기대 함수(conditional expectation functions)를 추정하며, 이를 통해 원인-결과 관계를 명확히 할 수 있는 방법론을 소개합니다. 연구진은 LLM이 예측한 데이터 예측값을 활용해 두 가지 기대 함수(𝔼[Y|W] 및 𝔼[D|W])를 모델링함으로써 효율성을 높일 수 있음을 제시합니다. 또한, LLM로 생성된 예측값이 기존의 고차원 임베딩(embeddings) 접근법보다 더 나은 성과를 거두는 것을 결과로 보여줍니다.

- **Performance Highlights**: 케이스 스터디를 통해 온라인 보석 경매에서 판매자의 평점(feedback score)이 경매 가격에 미치는 영향을 조사했습니다. 결과적으로, LLM이 생성한 예측값을 포함시키는 것이 원인 추정의 효율성을 개선한다는 것을 확인했습니다. 720개의 경매 데이터를 분석하여, 기존의 단일 임베딩 방식보다 LLM 기반 접근법이 더 나은 예측 결과를 도출함을 보였습니다.



### Coupled Data and Measurement Space Dynamics for Enhanced Diffusion Posterior Sampling (https://arxiv.org/abs/2510.09676)
- **What's New**: 새롭게 제안된 C-DPS(결합 데이터 및 측정 공간 확산 사후 샘플링) 프레임워크는 기존의 방법들이 갖는 한계를 극복합니다. 이 방법은 측정 공간에서의 순방향 확률 과정과 데이터 공간에서의 확산 과정을 동시에 진화시켜, 정밀한 샘플링을 가능하게 합니다. C-DPS는 사후 확률 분포를 기반으로 한 구성으로, 신뢰할 수 있는 재구성을 위한 추가적인 제약이나 우도 근사를 불필요하게 만듭니다.

- **Technical Details**: 이 논문에서는 측정 공간과 데이터 공간의 결합을 통해 두 개의 평행한 확산 프로세스를 제공합니다. 기존의 확산 모델들은 측정 정보를 사전 정보로 재투입해야 했으나, C-DPS는 이를 통합함으로써 불확실성을 줄이고 보다 효율적인 샘플링 과정을 가능케 합니다. 이 프레임워크는 Markov Chain을 활용하여 닫힌 형태의 사후 전이 확률을 도출합니다.

- **Performance Highlights**: C-DPS는 복수의 역문제 벤치마크에서 정성적 및 정량적인 성능에서 기존의 방법들을 일관되게 초월합니다. 특히, 이미지 복원, 블러 제거, 초해상도와 같은 다양한 설정에서 최신 성과를 달성하였습니다. 이로 인해 의학 영상 처리, 원거리 감지, 오디오 신호 처리와 같은 다양한 응용 분야에서도 효과적으로 활용될 수 있는 가능성을 보여줍니다.



### A physics-aware deep learning model for shear band formation around collapsing pores in shocked reactive materials (https://arxiv.org/abs/2510.09670)
- **What's New**: 이번 연구에서는 약한-중간 충격 부하 하에 결정형 에너지 물질(EM)의 핫스팟(spot) 형성을 다룹니다. 이는 EM의 안전한 저장 및 취급과 관련하여 매우 중요하며, 강한 충격 조건에 비해 연구가 부족한 분야입니다. 새로운 Physics-Aware Recurrent Convolutional Neural Network (PARCv2) 아키텍처를 개선하여, 이러한 상황에서의 전단 국소화(shear localization)와 플라스틱 가열(plastic heating)의 예측 능력을 향상시켰습니다.

- **Technical Details**: PARCv2는 강한 충격 반응을 예측할 수 있는 것으로 이미 입증된 모델입니다. 개선된 아키텍처는 약한-중간 충격 영역에서의 전단 밴드 형성(dynamics)을 정확하게 포착할 수 있는 능력을 가집니다. 또한, FFT neural operator와 신경 일상 미분 방정식(neural ordinary differential equations) 같은 물리 정보 기반 모델과의 성능 비교를 통해, PARCv2의 우수성을 강조했습니다.

- **Performance Highlights**: 모든 모델이 특정 실패 모드를 보이는 반면, 본 연구 결과는 반응성 물질에 대한 강력한 AI 가속 시뮬레이션 도구 개발에서 영역 특수 고려의 중요성을 강조합니다. PARCv2는 스페이토템포럴(spatiotemporal) 역학을 포착할 때 다른 모델들에 비해 우수한 성능을 보여주었습니다. 이는 향후 EM 연구의 안전성을 높이는 데 기여할 것으로 기대됩니다.



### Population synthesis with geographic coordinates (https://arxiv.org/abs/2510.09669)
- **What's New**: 본 논문에서는 정밀한 지리적 좌표를 가진 합성 인구를 생성하기 위한 새로운 알고리즘인 NF+VAE를 제안합니다. 기존의 방법들이 구 지리적 영역을 가지고 작업하는 데 어려움이 있었던 이유에 대해 설명하며, 이 알고리즘이 공간 데이터의 비정규성을 해결하는 방법을 소개합니다. NF(정규화 흐름)와 VAE(변분 자동 인코더)를 결합하여 실제 주택의 통계적 특성과 비슷한 합성 주택을 생성하는 과정을 보여줍니다.

- **Technical Details**: 제안하는 방법은 공간 좌표를 정규화 흐름(Normalizing Flows)을 사용해 더 규칙적인 잠재 공간으로 매핑한 후, 이를 변분 자동 인코더(Variational Autoencoder)에 결합하여 합성 인구를 생성합니다. 이러한 접근법은 공간과 비공간 특성 간의 공동 분포를 학습하며, 공간 자기상관을 활용하여 여러 다양한 지리에서 주택을 생성합니다. 결과적으로 이는 합성 데이터의 공간 분포와 특성 관계를 더욱 잘 캡처할 수 있도록 합니다.

- **Performance Highlights**: 결과적으로 NF+VAE 아키텍처는 대조군으로 사용된 방법들보다 우수한 성능을 보였습니다. 두 개의 사례 연구, 즉 이탈리아의 대출 데이터셋을 사용한 합성 주택 생성과 15개 도시의 Airbnb에 리스팅된 주택 생성에서 새로운 프레임워크가 성과를 거두었습니다. 제안된 프레임워크는 실제 마이크로데이터를 안전하게 보호하면서 동시에 유용한 합성 데이터를 생성할 수 있는 잠재력을 보여주고 있습니다.



### A Hybrid Computational Intelligence Framework with Metaheuristic Optimization for Drug-Drug Interaction Prediction (https://arxiv.org/abs/2510.09668)
- **What's New**: 이번 연구에서는 약물-약물 상호작용(DDI) 예측을 개선하기 위한 해석 가능하고 효율적인 프레임워크를 제안합니다. 현대 기계 학습과 도메인 지식을 결합하여, 상호작용이 없는 약물에 대한 이해를 높이고 안전한 처방을 지원하는 방안입니다. Mol2Vec와 SMILES-BERT 두 가지 분자 임베딩(molecular embeddings)을 활용하며, 약리학적 지식을 주입하는 규칙 기반 임상 점수(RBScore)도 포함되어 있습니다.

- **Technical Details**: 연구에서 제안한 방법은 Mol2Vec와 SMILES-BERT를 결합하여 분자 구조 패턴과 화학적 특징을 학습합니다. 이어서, 규칙 기반 임상 점수인 RBScore를 통해 상호작용 레이블에 의존하지 않고 약리학적 지식을 주입합니다. 마지막으로, RSmpl-ACO-PSO라는 새로운 3단계 메타휴리스틱(Metaheuristic) 최적화 전략을 통해 경량화된 신경 분류기(neural classifier)를 최적화하여 성능의 안정성을 달성합니다.

- **Performance Highlights**: 실제 데이터셋을 바탕으로 실시한 실험에서는 모델이 높은 예측 정확도(ROC-AUC 0.911, PR-AUC 0.867)를 달성했습니다. 또한, 2형 당뇨병(Type 2 Diabetes Mellitus) 환자 집단에 대한 일반화도 잘 이루어졌습니다. 임베딩 융합(embedding fusion), RBScore 및 최적화기가 각각 정밀도와 견고성을 높이는 데 어떻게 기여하는지에 대한 연구 결과도 제시되었습니다.



### Spatial Uncertainty Quantification in Wildfire Forecasting for Climate-Resilient Emergency Planning (https://arxiv.org/abs/2510.09666)
- **What's New**: 이 연구는 지구 관측 데이터를 활용해 고해상도 산불 확산 예측에서 공간 불확실성을 체계적으로 분석한 최초의 연구입니다. 머신 러닝 기반의 예측 모델이 불확실성 정량화 없이 산불 예측에 사용되고 있는 현재의 한계를 극복하고자 합니다. 특히, 예측 모델이 불확실성을 내포하고 있는 지역을 식별하여 비상 계획 및 자원 배치에 직접적인 적용이 가능함을 보여주고 있습니다.

- **Technical Details**: 사용된 방법론은 WildfireSpreadTS 데이터셋을 기반으로 하며, 이 데이터셋은 활성 산불 지역 중심의 64×64 패치의 공간-시간 큐브를 제공합니다. 또한, UTAE 모델을 활용하여 멀티모달 데이터 입력을 학습하고, Monte Carlo Dropout, Deep Ensembles, Bayesian Neural Networks와 같은 다양한 불확실성 정량화 접근법을 적용하였습니다. 이를 통해 예측된 확률의 평균 및 분산을 계산하여 불확실성을 평가합니다.

- **Performance Highlights**: 모델은 식생 기반의 특징을 사용하여 가장 높은 예측 성능을 달성했으며, 불확실성 추정치는 예측된 화재 경계 근처에 일관된 공간 패턴을 형성하였습니다. 20-60 미터의 불확실성 완충 지역을 식별하여, 사건 관리 팀의 전술적 규모와 적합하여 실용적인 통찰력을 제공합니다. 이 결과들은 기후 변화에 따른 산불 위험 증가에 대응하기 위한 더 나은 산불 관리 시스템을 가능하게 합니다.



### LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inferenc (https://arxiv.org/abs/2510.09665)
- **What's New**: LMCache는 최신 LLM 엔진에서 생성된 KV 캐시를 추출, 저장하고 이를 여러 엔진과 쿼리에서 공유하는 오픈 소스 KV 캐싱 솔루션입니다. 이는 LLM 엔진의 인터페이스에 KV 캐시를 노출시켜, 기존의 독립적인 프로세서로서의 역할에서 벗어나 여러 엔진 간의 효율적인 스트리지/통신 매체로 사용됩니다.

- **Technical Details**: LMCache는 KV 캐시의 추출, 로딩, 저장 및 전송을 위해 새로운 KV 캐시 의미론을 지원합니다. 특히, 배치 데이터 이동 작업, 컴퓨팅 및 I/O 파이프라인과 같은 성능 최적화를 통해 KV 캐시 데이터 이동을 고도로 최적화합니다. 또한, 모듈화된 KV 캐시 커넥터 구성 요소를 통해 빠르게 발전하는 추론 엔진과의 디커플링을 실현합니다.

- **Performance Highlights**: LMCache와 vLLM을 조합하면 다양한 작업 부하에서 최대 15배의 처리량 개선을 이룰 수 있습니다. 또한 LMCache는 기존 오픈 소스 프레임워크의 내장 KV 캐싱 메커니즘 및 상업적 API보다 일관되게 더 높은 성능을 보이며, 여러 기업과 오픈 소스 프로젝트에서 활용되고 있습니다.



### Semantic-Cohesive Knowledge Distillation for Deep Cross-modal Hashing (https://arxiv.org/abs/2510.09664)
- **What's New**: 최근의 연구에서는 SODA라는 새로운 심Semantic cohesive knowledge distillation 스킴이 제안되었습니다. 이 방법은 다중 레이블 정보를 새로운 텍스트 모달리티로 도입하며, 이미지와 레이블 간의 상호작용을 최적화하여 해시 코드 학습을 개선합니다. SODA는 이미지와 텍스트 모달리티 간의 의미적 유사성을 효과적으로 보존하는 해밍 스페이스를 학습하는 데 기여합니다.

- **Technical Details**: SODA에서는 두 단계의 크로스 모달 Teacher-Student 네트워크를 설계하여 다양한 모달리티 간의 의미적 특성을 포착합니다. 여기서 다중 레이블 정보를 ground-truth 레이블 프롬프트로 재구성하여, 이미지에 표현된 의미적 요소를 더 직관적으로 묘사합니다. 이미지 해밍 스페이스는 레이블 모달리티와 긴밀히 연결되어, 서로 간의 의미적 관련성을 극대화합니다.

- **Performance Highlights**: SODA는 두 개의 벤치마크 데이터셋에서 최신 기법들보다 우수한 성능을 보이고 있음을 실험적으로 입증하였습니다. 이 모델은 기존의 방법들이 간과했던 다중 레이블의 풍부한 의미적 정보를 활용하여 우수한 검색 성능을 달성합니다. 특히 SODA는 이미지과 텍스트 모달리티 간의 해시 코드 유사성을 효과적으로 학습하여 성능 향상에 기여합니다.



### Assessment of different loss functions for fitting equivalent circuit models to electrochemical impedance spectroscopy data (https://arxiv.org/abs/2510.09662)
- **What's New**: 이번 논문에서는 전기화학 임피던스 분광법(Electrochemical Impedance Spectroscopy, EIS) 데이터를 모델링하는 데 사용되는 두 가지 새로운 손실 함수(log-B 및 log-BW)를 도입합니다. 이 손실 함수들은 EIS의 보드(Bode) 표현에서 파생되었으며, 기존 손실 함수와 비교하여 성능이 평가되었습니다. 이는 EIS 데이터 분석에서 더 나은 결과를 제공할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 손실 함수들은 R2 점수, 카이제곱(chi-squared), 계산 효율성(computational efficiency) 및 예측된 성분 값과 원래 값 간의 평균 절대 비율 오차(Mean Absolute Percentage Error, MAPE) 측면에서 분석되었습니다. 통계적 비교 결과, 손실 함수의 선택이 수렴(convergence), 계산 효율성, 적합도(fit quality), MAPE에 영향을 미친다는 것이 밝혀졌습니다.

- **Performance Highlights**: X2 손실 함수는 적합도 측정에서 가장 높은 성능을 보였으며, 적합도가 주요 목표인 경우 선택할 수 있는 최적의 방법이었습니다. 반면, log-B는 약간 낮은 적합도에도 불구하고 약 1.4배 빠르고 대부분의 회로 구성 요소에서 더 낮은 MAPE를 보여주어, 강력한 대안으로 평가됩니다. 이는 대규모 최소제곱 적합(least squares fitting)에서 중요한 요소로 작용합니다.



### Learning What Matters: Steering Diffusion via Spectrally Anisotropic Forward Nois (https://arxiv.org/abs/2510.09660)
- **What's New**: 이번 연구에서는 Diffusion Probabilistic Models (DPMs)에 대한 유도 편향(inductive bias)을 명시적으로 설계하여 데이터의 목표 분포에 더 잘 맞도록 훈련 및 샘플링 과정을 개선하고자 합니다. 특히, 등방(isotropic) forward covariance를 대체할 구조화된 비등방(anisotropic) 잡음 연산자를 도입하여 주파수 대각선 주파수(frequency-diagonal) 코바리언스를 적용했습니다. 이러한 접근을 통해 우리는 지정된 주파수 대역을 강조하거나 억제하면서도 forward 과정은 Gaussian 형태를 유지할 수 있습니다.

- **Technical Details**: 연구에서는 Spectrally Anisotropic Gaussian Diffusion (SAGD)이라는 새로운 비등방 가우시안 조작자를 제안하였습니다. 이 조작자는 Fourier 기반으로 구조화된 공분산을 포함하고 있어, 잡음 첨가 과정에서 특정 주파수 대역의 정보를 조정할 수 있도록 합니다. 결과적으로, 이 방법은 모델이 특정 데이터 분포의 특징을 더 효과적으로 학습하도록 돕습니다. DPM의 기존 구조에 몇 줄의 코드로 통합이 가능하여 전체 파이프라인을 손상시키지 않는 장점이 있습니다.

- **Performance Highlights**: SAGD를 활용한 모델은 여러 자연 이미지 데이터셋에서 전통적인 확산 모델보다 더 나은 성능을 보여줍니다. 이 방법은 주파수 대역을 선택적으로 무시하면서도 학습할 수 있다는 점에서 획기적입니다. 실험 결과, SAGD를 통해 학습된 모델은 알려진 오염을 무시하고 깨끗한 분포를 회복하는 능력까지 발휘하고 있습니다. 최종적으로 이러한 결과는 DPM의 유도 편향을 정교하게 설계하는 것이 성능 향상에 크게 기여할 수 있음을 보여줍니다.



### Heterogeneous Point Set Transformers for Segmentation of Multiple View Particle Detectors (https://arxiv.org/abs/2510.09659)
Comments:
          Submitted to Machine Learning and the Physical Sciences Workshop (ML4PS) at NeurIPS 2025

- **What's New**: 본 연구는 NOvA 실험에서 발생하는 희소 데이터에 대한 새로운 접근 방식을 제안합니다. 기존의 CNN 방식에 비해 물질 소멸과 동일한 방향에서 정보를 결합하는 Point Set Neural Network를 도입하여 메모리를 10% 미만으로 줄이면서도 AUC 점수를 96.8% 달성했습니다. 이는 두 개의 2D 뷰에서 독립적으로 처리할 경우의 AUC 점수인 85.4%보다 상당히 향상된 결과입니다. 이 연구는 물리학 실험에서의 데이터 처리에 대한 기계 학습의 활용을 더욱 확대할 수 있는 가능성을 보여줍니다.

- **Technical Details**: 이론적으로, NOvA 탐지기는 두 개의 2D 뷰(XZ와 YZ)로 희소 이미지를 생성하며, 각 뷰는 속성에 따라 개별적으로 처리됩니다. 본 연구에서는 이러한 두 뷰 간의 정보를 연결하기 위해 heterogeneous point set transformer(HPST)라는 네트워크를 설계했습니다. 이 네트워크는 입력 데이터의 점 간 거리를 측정하고, 이를 기반으로 그래프를 구축하여 두 뷰 간의 정보를 효과적으로 교환합니다. 또한, UNet 아키텍처를 사용하여 인스턴스와 시맨틱 분할을 동시에 수행하도록 네트워크를 구성했습니다.

- **Performance Highlights**: 제안된 모델은 AUC 점수 96.8%를 달성하며, 이로 인해 NOvA 실험의 데이터 처리 효율성을 크게 향상시켰습니다. 메모리 사용량이 기존 방법과 비교해 현저하게 감소하여 실험 데이터의 처리 비용을 줄이는 데 기여할 수 있을 것으로 기대됩니다. 이 연구는 particle physics 분야의 실험 데이터 처리에 있어 머신러닝 모델의 활용 가능성을 높이는 중요한 발전을 나타냅니다.



### Gradient-Sign Masking for Task Vector Transport Across Pre-Trained Models (https://arxiv.org/abs/2510.09658)
- **What's New**: 이 연구는 새로운 파라미터 전이 방법인 GradFix를 소개합니다. GradFix는 특정 작업에 대한 모델의 적응을 포착하는 파라미터 변화(태스크 벡터)를 활용하여, 이전에 교육된 모델에서 최신 모델로의 지식 전이를 가능하게 합니다. 특히, 기존의 파인튜닝을 반복할 필요 없이 라벨된 샘플 몇 개만으로도 지식을 효과적으로 이전할 수 있습니다.

- **Technical Details**: GradFix는 목표 모델의 경량 손실 구조와 기울기 신호를 이용하여 태스크 벡터를 마스킹하는 방법입니다. 이를 통해, 목표 모델의 지역 손실 구역에 맞게 태스크 벡터를 재기반화하여 업데이트를 생성합니다. 이 과정은 목표 모델의 손실을 감소시킨다는 이론적 보증을 제공하며, 그리드 정보를 기반으로 효과적인 전이를 달성합니다.

- **Performance Highlights**: 실험 결과, GradFix는 시각 및 자연어 처리 벤치마크에서 기존의 간단한 태스크 벡터 추가 및 소수 샘플 파인튜닝과 비교할 때, 효과적인 성능 향상을 보여주었습니다. 낮은 데이터 상황에서도 작업 지식의 효과적인 이동을 가능하게 하여, 새로운 프리트레인 모델에서의 변화를 최소화할 수 있습니다.



### Generative Models for Helmholtz Equation Solutions: A Dataset of Acoustic Materials (https://arxiv.org/abs/2510.09657)
Comments:
          Accepted at EUSIPCO 2025

- **What's New**: 본 논문에서는 Helmholtz 방정식에 의해 해결된 31,000개의 음향 재료 구성으로 이루어진 HA30K 데이터셋을 소개합니다. 이 데이터셋은 기존의 전통적인 수치해석 방법보다 빠르고 효율적으로 음향 파동 전파를 시뮬레이션할 수 있는 딥러닝 기반 접근 방식을 가능하게 합니다. 또한, 다양한 장애물 구성과 해당하는 압력장 해를 제공하는 데이터 셋을 통해 기계 학습 응용을 위한 풍부한 자료를 제공합니다.

- **Technical Details**: HA30K 데이터셋은 31,000개의 2D 음향 재료 구성 이미지를 포함하며, 각 샘플은 FreeFEM을 이용하여 계산된 Helmholtz 방정식의 해를 담고 있습니다. 데이터셋의 각 샘플은 정사각형 도메인 내에 1개에서 6개의 정사각형 장애물이 포함되어 있으며, 음향 소스는 가우시안 소스로 모델링 되어 있습니다. 이를 통해 딥러닝 모델이 음향 재료 구성으로부터 압력 해 예측을 학습할 수 있도록 설계되었습니다.

- **Performance Highlights**: 기존의 전통적인 수치해석 방법과 비교할 때, 안정적인 확산(Stable Diffusion)과 ControlNet에 기반한 모델은 압력장 예측에서 높은 품질의 이미지를 생성할 수 있는 능력을 입증하였습니다. 이 접근 방식은 GPU 병렬 처리를 통해 여러 시뮬레이션을 동시에 처리하여 계산 시간을 크게 단축시킵니다. 최종적으로, 딥러닝 기반 방법이 초기 연구 단계에서의 빠른 탐색에 매우 유용하다는 점을 강조하며, 이 데이터셋은 음향 재료 시뮬레이션 분야에서 미래 연구의 기준을 제공할 것입니다.



### Enhanced Urban Traffic Management Using CCTV Surveillance Videos and Multi-Source Data Current State Prediction and Frequent Episode Mining (https://arxiv.org/abs/2510.09644)
Comments:
          24 pages, 9 figures

- **What's New**: 이 연구는 현대 교통의 동적 특성에 적합한 지능적이고 적응 가능한 교통 관리 솔루션을 요구하는 상황에서, CCTV 감시 비디오와 다중 소스 데이터 설명자를 통합하는 통합 프레임워크를 개발하였습니다. 기존의 정적 신호 및 수동 모니터링에 의존하는 시스템이 부족하다는 점을 강조하고 있습니다.

- **Technical Details**: 제안된 방법론은 스페이셜-템포럴(spatial-temporal) 특징 융합(spatio-temporal feature fusion)과 연속 교통 패턴 발견을 위한 Frequent Episode Mining을 포함하며, 강력한 교통 상태 예측을 위한 하이브리드 LSTM-Transformer 모델을 사용합니다. 이 프레임워크는 46개의 카메라에서 수집된 313,931개의 주석이 달린 바운딩 박스를 포함하는 CityFlowV2 데이터셋에서 평가되었습니다.

- **Performance Highlights**: 이 연구에서 제안된 모델은 98.46%의 높은 예측 정확도를 달성하였고, 매크로 정밀도(macro precision)는 0.9800, 매크로 재현율(macro recall)은 0.9839, 매크로 F1-score는 0.9819로 나타났습니다. 46개의 지속적인 혼잡 알림은 시스템 생성으로, 실질적인 혼잡 관리에 유용하다는 것을 보여줍니다.



### Direct Routing Gradient (DRGrad): A Personalized Information Surgery for Multi-Task Learning (MTL) Recommendations (https://arxiv.org/abs/2510.09643)
- **What's New**: 이번 연구에서는 개인화된 Direct Routing Gradient(DRGrad) 프레임워크를 제안하여 다중 작업 학습(Multi-task Learning, MTL)에서의 부정적 전이(negative transfer) 및 시소(seesaw) 문제를 해결합니다. DRGrad는 라우터, 업데이트, 개인화된 게이트 네트워크라는 세 가지 핵심 구성요소로 구성되어 있으며, 서로 다른 작업 간의 경계(stakes)를 판단하여 효과적으로 모델의 성능을 향상시킬 수 있습니다. 이 방법은 개인화 정보를 더 잘 활용하고, 각 작업에 대한 유효한 기울기(gradient)를 활용하여 충돌을 줄입니다.

- **Technical Details**: DRGrad는 라우터(ruter), 업데이트(updater), 개인화된 게이트 네트워크(personalized gate network)의 세 가지 주요 구성 요소로 이루어져 있습니다. 라우터는 훈련 과정에서 각 작업 간의 경계를 판단하고, 업데이트 네트워크는 라우터의 출력을 기반으로 기울기를 동적으로 집계하여 작업의 성능을 최적화합니다. 개인화된 게이트 네트워크는 사용자와 관련된 개인화된 기울기를 제공하여 각 작업의 업데이트를 세밀화하는 데 기여합니다.

- **Performance Highlights**: DRGrad는 150억 샘플의 실제 추천 데이터 세트를 기반으로 하여 복잡한 MTL 환경에서 효율성을 평가하였으며, 경쟁하는 최신 MTL 모델들에 비해 탁월한 성능을 보여주었습니다. 특히 AUC(Area Under the Curve) 지표에서 DRGrad는 다른 모델에 비해 더 우수한 결과를 나타내며, 작업 간의 충돌을 효과적으로 관리할 수 있음을 증명하였습니다. 또한, 공공 Census-income 데이터 세트 및 합성 데이터 세트를 통해 다양한 상관관계와 개인화를 갖춘 작업 간의 경계를 판단하고 라우팅할 수 있는 능력을 입증하였습니다.



### Are Large Reasoning Models Interruptible? (https://arxiv.org/abs/2510.11713)
Comments:
          Project Page: this https URL

- **What's New**: 이 논문에서는 대형 추론 모델(Large Reasoning Models, LRMs)이 전통적으로 정적인 환경에서 평가되는 방법, 즉 '고정된 세계(frozen world)' 가정을 도전합니다. 연구팀은 현대의 추론 작업에서 시간 제약이 있는 동적 시나리오를 통해 LRMs의 내구성을 평가하고, 중단(interruptions)과 동적 맥락(dynamic context)의 두 가지 주요 개입 유형을 분석합니다. 이러한 접근을 통해 기존의 정적 평가 방식이 모델의 강인성을 과대평가한다는 이동을 발견했습니다.

- **Technical Details**: 연구에서는 수학 및 프로그래밍 과제를 포함한 동적 환경에서의 평가 프로토콜을 소개합니다. 특히 사용자가 긴 계산 중에 중단을 요청할 때 모델이 얼마나 잘 반응하는지, 그리고 계산 중에 제시된 새로운 정보가 모델의 최종 답변에 어떻게 통합되는지를 평가했습니다. 연구 결과, STATISTICALLY 정적 설정에 비해 동적 환경에서의 정확도가 최대 60%까지 하락하는 현상을 관찰했습니다.

- **Performance Highlights**: 이 연구에서 확인된 LRMs의 주요 마이너스 포인트는 '추론 새는 현상(reasoning leakage)', '패닉(panic)', 그리고 '자신의 불안(self-doubt)'입니다. 각 상황에서 모델들이 내놓는 답변의 질이 크게 저하될 수 있으며, 이러한 발견은 LRMs 개발의 새로운 방향성을 제시합니다. 또한 본 연구는 추론 중단이 모델 성능에 미치는 여러 흥미로운 영향을 분석하여, 실용적 AI 모델의 발전에 기여할 수 있는 기초 자료를 제공합니다.



### Diffusion Transformers with Representation Autoencoders (https://arxiv.org/abs/2510.11690)
Comments:
          Technical Report; Project Page: this https URL

- **What's New**: 이 논문에서는 기존의 VAE(Variational Autoencoder)를 대신하여 사전 훈련된 표현 인코더(예: DINO, SigLIP, MAE)를 활용한 새로운 표현 기반 자동 인코더(RAEs)를 제안합니다. 이 모델들은 소스에서 고품질의 재구성과 의미적으로 풍부한 잠재 공간을 제공하며, 확장 가능한 트랜스포머 아키텍처를 허용합니다. 기존의 접근 방식의 한계를 극복하고, 생성 성능을 향상시키기 위한 새로운 방법론을 제시했습니다.

- **Technical Details**: RAE는 VAE를 대체하는 방식으로, 사전 훈련된 표현 인코더와 훈련된 디코더를 결합하여 구성됩니다. 이러한 구조는 의미론적으로 풍부하고 구조적으로 일관된 잠재 공간을 생성하며, 기존의 고차원 잠재 공간에서도 확실한 조합을 가능하게 합니다. 또한, 고차원 표현이 오히려 장점이 될 수 있는 잠재 공간 내에서 안정적이고 효율적으로 디퓨전 모델 훈련을 수행할 수 있음을 시연했습니다.

- **Performance Highlights**: RAE 기반의 DiTDH는 ImageNet에서 256x256 해상도에서 FID(Frechet Inception Distance) 점수 1.51을 달성하며, AutoGuidance 없이도 훌륭한 생성 성능을 보여주었습니다. 이는 RAEs가 기존의 VAE에 대한 대안으로서 효과적이라는 것을 입증합니다. 전반적으로 이러한 결과는 자동 인코딩의 역할을 단순한 압축 방식에서 의미론적 표현의 기초로 재정립합니다.



### Accelerated stochastic first-order method for convex optimization under heavy-tailed nois (https://arxiv.org/abs/2510.11676)
- **What's New**: 이 논문에서는 중량 편향 노이즈(heavy-tailed noise)가 있는 볼록(com convex) 복합 최적화 문제를 연구합니다. 기존의 연구들은 일반적으로 기울기 절단(gradient clipping)이나 정규화(normalization) 기술을 사용하여 이러한 노이즈를 처리했지만, 본 논문에서는 이러한 추가 수정 없이도 최적의 복잡도(optimal complexity)에 도달할 수 있음을 보여줍니다. 특히 가속화된 확률적 근사(subgradient) 방법이 매끄럽고, 약간 매끄러운, 비매끄러운 볼록 최적화 문제에 대해 우주적으로 최적의 복잡도를 달성하였습니다.

- **Technical Details**: 제안된 방법은 중량 편향 노이즈 하에서의 복합 최적화 문제에 대한 가속화된 확률적 프로필 근사 방법(Accelerated Stochastic Proximal Method)을 사용합니다. 이 방법은 기울기 절단이나 정규화를 사용하지 않고 근사 최적 솔루션을 찾아내는 것이 가능하며, 기대값과 높은 확률 모두에서 최적화에 대한 복잡도를 달성할 수 있습니다. 특히, 이 논문에서는 다양한 수학적 조건을 설명하며, 근사적 해(approximate optimal solution)를 찾기 위한 방법들이 제시됩니다.

- **Performance Highlights**: 제안된 가속화된 방법은 B개 산출치(1사분면 경계값)에서 𝑂(ϵ^{-α/(α−1)})의 기울기 복잡도를 달성하여, 중량 편향 노이즈 하에서도 상당한 성과를 보입니다. 추가적으로, 논문은 수치 실험을 통해 이론적 결과를 검증하였으며, 기존 기법들과 비교했을 시 독립적인 장점을 강조합니다. 이러한 연구는 현대 대규모 응용 프로그램의 복잡한 최적화 문제에 대한 새로운 접근 방식을 제시하고 있습니다.



### Continual Release of Densest Subgraphs: Privacy Amplification & Sublinear Space via Subsampling (https://arxiv.org/abs/2510.11640)
Comments:
          to be published in SOSA'26

- **What's New**: 이번 연구에서는 edge-differentially private (DP) 그래프 알고리즘을 위한 sublinear space continual release (연속적 방출) 모델을 다룹니다. 특히, insertion-only 설정에서 densest subgraph problem (DSG) 문제에 초점을 맞추었습니다. 본 연구의 주요 결과는 최고의 static DP 알고리즘의 덧셈 오차(additive error)와 최고의 비공식적 스트리밍 알고리즘의 공간 복잡도(space complexity)를 일치시키는 최초의 DSG 알고리즘입니다.

- **Technical Details**: 본 논문에서는 서브샘플링(subsampling)의 정교한 사용을 통해 개인 정보 보호 강화(privcy amplification)와 희소화(sparsification)를 동시에 달성하는 아이디어를 제시합니다. 이는 그래프 DP 분야에서 이전에 포멀라이즈(formalized)되지 않았던 연결입니다. 또한, 고전적으로 정적(static) 설정으로 간단한 블랙박스 리덕션(black-box reduction)을 통해 $O(	ext{log} n)$의 덧셈 오차와 $O(n	ext{log} n)$의 공간을 가지는 순수(pure) 및 근사적(approximate) DP 알고리즘을 도출했습니다.

- **Performance Highlights**: 이 연구는 이전 연구보다 정확성과 공간 복잡도 모두에서 향상된 성능을 보여줍니다. 그래프 DP 설정에서 그래프 밀도 증가(graph densification)를 도입하여 이전 샘플링을 조기에 실행하고, 이전 작업에서 발생했던 추가 로그(logarithmic) 요소들을 제거합니다. 이러한 간단한 아이디어는 독립적인 관심을 끌 수 있을 것으로 생각됩니다.



### NV3D: Leveraging Spatial Shape Through Normal Vector-based 3D Object Detection (https://arxiv.org/abs/2510.11632)
- **What's New**: 본 논문에서는 NV3D라는 새로운 모델을 제안합니다. 이 모델은 K-nearest neighbors (KNN)와 주성분 분석 (PCA)을 활용하여 계산된 정규 벡터로부터 로컬 특성을 이용합니다. NV3D는 정상 벡터 밀도 기반 샘플링과 시각적 인지(FOV)-기반 샘플링의 두 가지 샘플링 방식을 제공, 기존 데이터의 55%까지 제거하면서도 성능을 유지합니다.

- **Technical Details**: NV3D는 격자 기반으로 인식된 포인트들을 분할하고, 이를 통해 정규 벡터와 관련된 특징을 추출합니다. 기존의 방법들과는 다르게, NV3D는 인접한 포인트들로부터 정규 벡터를 활용하여 표면 방향과 관련된 중요 정보를 수집합니다. 또한, 셀 간 소통을 통해 각측 정규 특성을 쿼리로 사용하고, 복수의 벡터 특성을 키와 값으로 사용하는 요소 기반의 주의 메커니즘을 도입합니다.

- **Performance Highlights**: NV3D 모델은 KITTI 데이터셋에서 훈련되었으며 자동차와 자전거 감지에서 뛰어난 성능을 기록했습니다. 샘플링 없이도 NV3D는 86.60%와 80.18%의 최다 평균 정밀도(mAP)를 달성, 기존 Voxel R-CNN 대비 2.61% 및 4.23% 높은 성과를 보였습니다. 두 가지 샘플링 기법으로, NV3D는 자동차 감지에서 85.54%의 mAP을 달성하여, 약 55%의 포인트들이 필터링되었음에도 불구하고 여전히 기준선을 초과하는 결과를 보였습니다.



### Lecture Notes on Verifying Graph Neural Networks (https://arxiv.org/abs/2510.11617)
- **What's New**: 이번 논문은 그래프 신경망(Graph Neural Networks)과 Weisfeiler-Lehman 테스트(Weisfeiler-Lehman tests) 간의 연결을 회상하고, 그래프 신경망의 검증(verification) 작업을 해결하기 위한 선형 불평등(linear inequalities)을 포함하는 새로운 모달 논리(modal logic)를 제시합니다. 이 모달 논리는 카운팅 모달리티(counting modalities)를 특징으로 하며, 특히 Presburger 산술(Presburger arithmetic)로 확장된 양화사 없는 조합(boolean algebra) 추론을 통해 구체화된 알고리즘을 설명합니다.

- **Technical Details**: 그래프 신경망은 라벨이 지정된 비순환 그래프에서 작동하며, 노드(node) 간의 정보를 통해 최종 출력 값(yes/no)을 결정합니다. 입력으로는 그래프와 특정 노드를 받고, 각 레이어(layer)를 통해 벡터 형태의 라벨(label)을 업데이트하여 최종 결정을 내립니다. 이 과정에서 활성화 함수(activation function)를 사용하며, 가중치 매개변수(weights) 및 바이어스(bias)가 포함됩니다.

- **Performance Highlights**: 제안된 알고리즘은 다양한 분야에서의 그래프 기반 문제 해결에 활용 가능성을 보여줍니다. 기존의 그래프 신경망 접근법과 비교하여 새로운 모달 논리는 더 강력한 검증 기술을 제공하며, 예를 들어 의약품 발견(drug discovery) 및 음악 점수(voice separation)에서의 활용 가능성을 제시합니다. 이러한 연구 결과는 향후 머신 러닝 알고리즘의 발전에 중요한 역할을 할 것으로 기대됩니다.



### Deconstructing Attention: Investigating Design Principles for Effective Language Modeling (https://arxiv.org/abs/2510.11602)
- **What's New**: 이번 연구는 Transformer 언어 모델의 주의(attention) 메커니즘을 체계적으로 해체하여, 각 디자인 원칙의 필요성을 실험적으로 검증했습니다. 서로 다른 원칙들을 선택적으로 완화한 변형을 설계하여, 모든 레이어에 균일하게 적용하거나 일부 레이어에서만 표준 주의를 유지하는 하이브리드 아키텍처에서 테스트하였습니다. 이를 통해 주의 메커니즘의 기초를 깊이 이해하고, 더 간소화된 언어 모델 개발의 가능성을 제시합니다.

- **Technical Details**: 연구에서는 주의 메커니즘의 핵심 원칙인 위치 간 정보 혼합(token mixing), 입력에 적응하는 시퀀스 종속 활성화(sequence-dependent activations), 특정 수학적 형식(dot-product similarities 및 softmax weighting), 쿼리와 키의 결합을 분석했습니다. 여러 실험에서, 토큰을 혼합하는 메커니즘이 필수적임을 발견하였고, 그 없는 경우 모델이 거의 랜덤한 행동으로 붕괴됨을 확인하였습니다. 반면, 가정된 수학적 형식과 시퀀스 종속성은 특정 레이어에서만 보존될 경우 상당히 완화될 수 있습니다.

- **Performance Highlights**: 예상외로, 독립적으로 실패하는 변형도 표준 주의와 혼합될 경우 강력한 성능을 발휘할 수 있다는 점에서 협력 효과를 강조합니다. 이러한 결과는 주의 메커니즘의 실제 효과를 이해하는 데 기여하고, 성능 저하 없이 언어 모델을 단순화할 수 있는 새로운 방향을 열어줍니다.



### SemCSE-Multi: Multifaceted and Decodable Embeddings for Aspect-Specific and Interpretable Scientific Domain Mapping (https://arxiv.org/abs/2510.11599)
- **What's New**: 이 논문에서는 SemCSE-Multi라는 새로운 비지도 학습(unsupervised) 프레임워크를 제안하여 과학 초록의 다면적 임베딩(embeddings)을 생성합니다. 이 임베딩은 연구자가 필요한 특정 측면(aspect)을 명확히 하고 독립적으로 포착할 수 있도록 하여 세밀하고 조절 가능한 유사성 평가(similarity assessment)를 가능하게 합니다. 또한, 본 접근법은 과학 분야의 사용자 주도 시각화를 위한 적응적 기능을 제공하는 점도 특징입니다.

- **Technical Details**: 제안된 접근법은 각 연구 초록에 대해 аспект별 요약 문장을 생성하고 이는 임베딩 모델에 의해 의미적으로 유사한 요약이 임베딩 공간 내에서 근접하게 배치되도록 학습됩니다. 최종적으로, 이 аспект별 임베딩 기능은 단일 임베딩 모델로 통합되어 단일 전방 통과(forward pass)에서 여러 аспект 임베딩을 예측할 수 있게 됩니다. 또한, 임베딩을 자연어 설명으로 복원하는 디코딩 파이프라인을 도입하여 임베딩 공간의 해석 가능성을 크게 향상시킵니다.

- **Performance Highlights**: 이 연구는 주로 침입 생물학 분야에서 성능을 평가하였으며, 전문가의 지도를 받았습니다. 논문의 처음에 제안한 대로, 다양한 측면을 취합하여 사용자 맞춤의 시각화 및 결과 도출을 가능하게 함으로써, 사용자가 필요로 하는 특정 연구 방향에 대한 명확한 통찰을 제공합니다. 이러한 접근법은 기존 방법의 한계를 극복하며, 특히 저차원 시각화에서 비어 있는 영역의 의미 있는 텍스트 설명을 생성하는데 효과적임을 입증하였습니다.



### Hierarchical Qubit-Merging Transformer for Quantum Error Correction (https://arxiv.org/abs/2510.11593)
Comments:
          6 pages, 5 figures

- **What's New**: 이번 논문에서는 효율적인 양자 오류 수정(QEC) 스킴을 위한 계층적 큐빗 병합 변환기(HQMT)를 제안합니다. HQMT는 안정자 코드(stabilizer code)의 구조적 그래프를 활용하여 여러 스케일에서 오류 상관관계를 학습하는 새로운 디코딩 프레임워크입니다. 이 연구는 딥러닝을 기반으로 한 신경망 디코더의 최신 발전을 활용하여 양자 컴퓨팅의 신뢰성을 높이고자 합니다.

- **Technical Details**: HQMT 아키텍처는 구조적으로 관련된 안정자 그룹에 대해 로컬로 주의(attention)를 계산하고, 이를 체계적으로 병합하여 오류 신드롬(error syndrome)의 글로벌 뷰를 구성합니다. 특히, 변환기(transformer) 아키텍처에 전용 큐빗 병합 레이어(qubit-merging layer)를 통합하여 오류율(logical error rate)을 크게 낮추는 데 성공했습니다. 이 계층적 접근 방식은 표면 코드(surface code) 디코딩에 효과적이며 확장 가능한 프레임워크를 제공합니다.

- **Performance Highlights**: HQMT는 다양한 코드 거리(code distance)에서 기존의 신경망 기반 QEC 디코더 및 강력한 신뢰 전파(belief propagation)와 순서 통계 디코딩(ordered statistics decoding) 기법인 BP+OSD를 능가하는 성능을 보였습니다. 이러한 결과는 HQMT가 신뢰할 수 있는 양자 컴퓨팅 실현을 위한 중요한 한 걸음을 내딛고 있음을 보여줍니다.



### MS-Mix: Unveiling the Power of Mixup for Multimodal Sentiment Analysis (https://arxiv.org/abs/2510.11579)
Comments:
          Under Review

- **What's New**: 이번 연구에서는 Multimodal Sentiment Analysis (MSA)에서 데이터 부족 문제를 해결하기 위해 MS-Mix라는 새로운 데이터 증강 프레임워크를 제안하고 있습니다. 기존의 mixup 전략의 한계를 극복하는 이 방법은 감정 인식을 위한 세 가지 주요 혁신 요소를 통합하여 고품질 샘플 생성을 보장합니다. 이러한 개선으로 MS-Mix는 기존 방법들보다 일관성 있고 매력적인 감정 분석 결과를 생성합니다.

- **Technical Details**: MS-Mix는 대화형 샘플 선택(Sentiment-Aware Sample Selection, SASS) 전략을 통해 반대 감정을 가진 샘플들의 혼합을 방지하고, 감정 강도 지침(Sentiment Intensity Guided, SIG) 모듈을 사용하여 각 모달리티의 감정 강도에 따라 동적으로 혼합 비율을 결정합니다. 또한 감정 정렬 손실(Sentiment Alignment Loss, SAL)을 도입하여 예측된 감정 분포를 실제 라벨과 정렬함으로써 모델의 일관성을 높입니다. 이와 같은 구성요소들이 결합되어 감정 정보의 효율적 활용을 가능하게 합니다.

- **Performance Highlights**: 세 개의 벤치마크 데이터셋과 여섯 가지 최신 모델 아키텍처에 대한 광범위한 실험 결과, MS-Mix는 기존 방법들보다 일관되게 성능이 우수함을 입증했습니다. 연구 결과에 따르면, MS-Mix는 서로 다른 백본 네트워크 환경에서도 강건한 멀티모달 감정 증강을 제공합니다. 이는 MSA 분야에서 새로운 기준을 설정하는 것으로 의미 있습니다.



### A Framework for Low-Effort Training Data Generation for Urban Semantic Segmentation (https://arxiv.org/abs/2510.11567)
- **What's New**: 이 논문은 도시 장면 인식을 위한 훈련 데이터 생성 방식을 혁신하는 새로운 프레임워크를 소개합니다. 복잡한 3D 모델링 없이 비표시(unlabelled) 이미지와 불완전한 유사 라벨(pseudo-labels)만으로 전이 학습을 가능하게 하여 우리가 사용할 수 있는 낮은 노력의 합성 데이터(source data)를 빠르게 생성합니다. 이 방법은 일반적으로 필요했던 노동 집약적인 합성 데이터 제작 과정을 간소화하여 효율성을 높입니다.

- **Technical Details**: 제안된 방법은 사전 훈련된 diffusion model을 사용하여 소스와 관련 없는 데이터 소스에서도 효과적인 이미지 생성을 가능하게 합니다. 이 프레임워크는 주 타겟 도메인은 물론, 비표시 데이터로부터 전환된 레이블 맵을 사용하여 높은 품질의 결과물을 제공합니다. 이러한 방식은 나쁜 생성물(suboptimal generations)을 필터링하고, 이미지와 레이블 간의 불일치를 수정하며, 데이터 집합 간의 의미론적 일관성을 표준화합니다.

- **Performance Highlights**: 결과적으로, 이 프레임워크는 기존의 최고 수준의 이미지-이미지 전환(image-to-image translation) 기법들보다 최대 8%pt mIoU의 성능 향상을 보여주었습니다. 특히, 간단한 합성 데이터의 변환에서 기존의 고비용 합성 데이터 관리 방식과 동일한 수준의 성능을 달성할 수 있음을 입증했습니다. 이 연구는 도시 장면 이해를 위한 훈련 데이터 생성 과정에서 새로운 가능성을 제시하고 있습니다.



### Efficient Group Lasso Regularized Rank Regression with Data-Driven Parameter Determination (https://arxiv.org/abs/2510.11546)
Comments:
          36 pages, 4 figures, 8 tables

- **What's New**: 이 논문은 고차원 회귀 분석에서의 강건성(robustness)을 향상시키기 위해 비매끄러운 Wilcoxon 점수 기반의 순위(c_rank) 목표와 구조적 그룹 희소성(group sparsity) 정규화를 결합한 새로운_GROUP Lasso 정규화 순위 회귀 방법을 제안합니다. 이러한 접근법은 기존의 Lasso 기법에서 발전된 조정 없는 파라미터 선택 방식을 확장하여 데이터 기반의 시뮬레이션 기반 조정 규칙을 도입하고, 결과 추정치의 유한 샘플 오류 경계를 수립합니다.

- **Technical Details**: 제안된 방법은 비선형 최적화 문제 해결을 위해 Proximal Augmented Lagrangian (PALM) 방법을 개발하였으며, 이로 인해 기존 방법에서 발생할 수 있는 특이성 문제를 해결하면서 효율적인 반부드러운 뉴턴(semismooth Newton) 업데이트가 가능하게 됩니다. 그룹 Lasso 정규화를 통해 계수 선택과 같은 구조적 제약을 도입할 수 있으며, 이 방법은 데이터 구조를 고려한 조정 규칙을 사용하여 계산적으로 부담이 크지 않도록 설계되었습니다.

- **Performance Highlights**: 광범위한 수치 실험을 통해 제안된 추정치의 강건성과 효과성을 검증하였으며, 시뮬레이션 데이터 및 실제 데이터 세팅에서 알고리즘의 확장성을 시연하였습니다. 그 결과, 제안된 방법은 기존 대안들에 비해 성능이 우수하며, 고차원 데이터에서도 신뢰할 수 있는 예측 결과를 제공합니다.



### Automatic Music Sample Identification with Multi-Track Contrastive Learning (https://arxiv.org/abs/2510.11507)
- **What's New**: 이 논문에서는 기존 오디오 트랙의 샘플을 재사용하여 새로운 음악 콘텐츠를 만드는 샘플링(Sampling) 기술에 대한 자동 샘플 식별(automatic sample identification) 작업을 다루고 있습니다. 자가 지도 학습(self-supervised learning) 접근 방식을 도입하여 다중 트랙 데이터셋에서 인공 혼합물의 긍정적 쌍을 생성하고, 새로운 대조적 학습(objective) 방법을 설계하였습니다. 이 방법은 이전의 최첨단 기준선보다 우수한 성능을 보이며, 다양한 장르에 강건하고, 참조 데이터베이스에 노이즈 곡을 추가하는 것이 용이함을 보여줍니다.

- **Technical Details**: 샘플 식별 작업은 오디오 핑거프린팅(audio fingerprinting)과 밀접한 관련이 있으며, 2013년 Van Balen et al.이 이 작업을 처음 시작했습니다. 그러나 샘플 식별은 적절한 훈련 데이터 부족 등 실용적인 장애물에 직면해 있습니다. 본 연구에서는 음악 녹음의 다양한 출처에서 긍정적 쌍을 구축하고, 특정 변환을 무작위로 적용하여 효과적으로 훈련 세트를 생성하는 새로운 방법을 채택하였습니다.

- **Performance Highlights**: 우리는 다양한 장르에 걸친 개인 데이터 세트와 표준 힙합 벤치마크에서 모델 성능을 평가하였습니다. 그 결과, 평균 정밀도(mean average precision)에서 15% 이상 향상된 성능을 보였고, 특정 교육 모듈 및 다중 트랙 교육 세트의 영향을 평가하며 모델의 견고성 또한 입증하였습니다. 또한 전체 훈련 코드를 공개할 예정입니다.



### Constraint-Aware Reinforcement Learning via Adaptive Action Scaling (https://arxiv.org/abs/2510.11491)
- **What's New**: 이번 논문에서는 Safe Reinforcement Learning (안전 강화 학습) 분야에서 새로운 모듈형 비용 인식 조절기(modular cost-aware regulator)를 제안합니다. 기존 방법들이 주로 단일 정책을 통해 보상과 안전을 동시에 최적화하려고 시도하여 불안정성을 초래했던 반면, 우리는 예측된 제약 위반에 따라 행위를 조절하는 접근법을 도입합니다. 이 방법은 시스템의 동적 정보를 사전 지식 없이 안전성을 유지하면서 탐색을 가능하게 합니다.

- **Technical Details**: 제안된 방법은 정책의 작업 지향 행동을 보존하는 방식으로 작동하며, 위험한 행동을 감소시키기 위해 조절기를 통해 요소별 크기를 조정합니다. 이 조절기는 딥러닝 기반의 안전성 비평가(twin cost critics)를 통해 제약 위반을 보수적으로 추정하여 작동합니다. 우리의 접근법은 SAC(Soft Actor-Critic)와 TD3 같은 오프 폴리시 RL 메서드와 통합되어 실행됩니다.

- **Performance Highlights**: 실험 결과, 우리는 Safety Gym locomotion 작업에서 최신 기법들과 비교하여 보상 대비 비용(Return-to-Cost, RC) 비율에서 최고 성능을 달성하였습니다. 우리의 방법은 제약 위반을 최대 126배 감소시킬 수 있었으며, 이전 방법들에 비해 보상 증가폭이 10배 이상으로 나타났습니다. 이는 안전성과 탐색 능력을 모두 향상시키는 데 기여합니다.



### Coordinated Strategies in Realistic Air Combat by Hierarchical Multi-Agent Reinforcement Learning (https://arxiv.org/abs/2510.11474)
Comments:
          2025 IEEE International Conference on Agentic AI (ICA)

- **What's New**: 이 논문에서는 비현실적인 공중전 시뮬레이션 환경에서의 문제를 해결하기 위해 새로운 3D 다중 에이전트 공중전 환경과 계층적 다중 에이전트 강화 학습(Hierarchical Multi-Agent Reinforcement Learning) 프레임워크를 소개합니다. 특히, 이 연구는 불완전한 상황 인식과 비선형 비행 역학이라는 두 가지 도전 과제를 해결하는 것을 목표로 하고 있습니다.

- **Technical Details**: 제안된 방법은 이질적인 에이전트 동역학(heterogeneous agent dynamics), 커리큘럼 학습(curriculum learning), 리그 플레이(league-play) 및 새로운 훈련 알고리즘을 결합하여 기초하고 있습니다. 의사결정 과정은 두 가지 추상화 수준으로 조직되어, 저수준 정책(low-level policies)은 세밀한 조작을 학습하고, 고수준 정책(high-level policies)은 임무 목표에 따라 전술적 명령을 발행합니다.

- **Performance Highlights**: 경험적 결과에 따르면, 계층적 접근법이 복잡한 공중전 시나리오에서 학습 효율성과 전투 성능을 모두 개선하는 데 기여하는 것으로 나타났습니다. 이를 통해 공중전 작전에서의 성능 향상이 확인되었으며, 새로운 알고리즘의 효과성을 입증하였습니다.



### Forward-Forward Autoencoder Architectures for Energy-Efficient Wireless Communications (https://arxiv.org/abs/2510.11418)
- **What's New**: 최근 통신 시스템의 깊은 학습 적용은 증가하는 관심사의 분야로 부각되었습니다. 이 논문에서는 일반적인 신경망 훈련 절차인 역전파(Backpropagation, BP) 알고리즘의 대안으로서, 효율적인 앞으로 가는 학습(Forward-forward, FF) 방식을 제안합니다. FF 학습의 장점 중 하나는 통신 채널이 미분 가능할 필요가 없고, 전역적인 부분 미분의 가용성에 의존하지 않아 에너지 효율적인 구현이 가능하다는 점입니다.

- **Technical Details**: 본 연구에서는 FF 알고리즘을 사용하여 엔드 투 엔드 학습된 오토인코더(autoencoder)를 설계하고, 가산 화이트 가우시안 노이즈(additive white Gaussian noise) 및 레일리 블록 페이딩(Rayleigh block fading) 채널에 대한 성능을 수치 평가합니다. 공동 코딩과 변조(joint coding and modulation) 사례에서 BP로 훈련된 시스템과의 경쟁력을 입증하며, 미분 불가능한 변조 단계를 고정하여 적용하는 시나리오에서도 경쟁력을 보여줍니다. FF 네트워크의 설계 원칙, 훈련 수렴 행동, BP 기반 접근 방식에 비해 주요 메모리 및 처리 시간 절약에 대한 통찰도 제공합니다.

- **Performance Highlights**: FF 학습을 사용한 오토인코더는 BP 훈련 시스템과 비교하여 유사한 성능을 보이며, 메모리와 처리 시간에서 상당한 절약 효과를 가져옵니다. 특히, 미분 불가능한 모듈을 사용해도 높은 성능을 유지하며, 에너지 소비 면에서도 효율적입니다. 이러한 결과는 FF 알고리즘이 통신 시스템에 효과적으로 적용될 수 있는 가능성을 보여줍니다.



### Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers (https://arxiv.org/abs/2510.11370)
- **What's New**: 이 논문에서는 Mixture-of-Experts (MoE) 모델에서 강화 학습의 불안정성을 초래하는 라우팅(distribution) 간의 불일치를 분석합니다. 이를 해결하기 위한 새로운 방법인 Rollout Routing Replay (R3)를 제안하여, 이 방법이 훈련 속도를 저하시킴 없이도 훈련 및 추론 간의 KL divergence를 현저히 줄인다. R3의 적용을 통해 MoE 모델에서 RL 훈련의 안정성을 확보하고, 다른 방법들(GSPO, TIS)을 초월하는 성능을 보여줍니다.

- **Technical Details**: R3는 인퍼런스 엔진에서 라우팅 분포를 기록하고, 이를 훈련 단계에서 재생하여 MoE 모델의 정책을 안정화합니다. 이 방법은 훈련(πtrain)과 추론(πinfer) 엔진 간의 불일치를 줄이고, 극단적인 불일치를 완화하여 RL 훈련의 불안정성을 해결합니다. 기존의 기법이 완전히 해결하지 못한 off-policy 문제를 근본적으로 해결하는 방향으로 설계되었습니다.

- **Performance Highlights**: 다양한 설정에서의 포괄적인 실험 결과, R3는 MoE 모델에서 훈련의 안정성을 향상시키며 RL 훈련의 붕괴를 방지합니다. R3는 훈련 및 성능 면에서 기존의 접근 방식들에 비해 확연한 향상을 보여주고 있으며, 온-정책 및 미니 배치 스타일의 오프-정책 RL 시나리오에서 모두 적용 가능합니다. 이 연구는 MoE 모델에서 RL을 안정화할 수 있는 새로운 솔루션을 제공합니다.



### Diffusion-Link: Diffusion Probabilistic Model for Bridging the Audio-Text Modality Gap (https://arxiv.org/abs/2510.11330)
Comments:
          5 pages. Submitted to IEEE ICASSP 2026

- **What's New**: 이 논문은 Diffusion-Link라는 새로운 모듈을 제안하여 오디오 임베딩을 텍스트 임베딩 분포로 생성적으로 매핑합니다. 이 모듈은 동기식 네트워크로 구성되어 있으며, 고정된 멀티모달 인코더의 출력 임베딩에서 학습됩니다. 특히, 자동 오디오 캡셔닝(Automatic Audio Captioning, AAC)에 처음으로 확산 기반 모듈을 적용한 사례로 주목받고 있습니다.

- **Technical Details**: Diffusion-Link는 세 개의 잔여 다층 퍼셉트론(Residual MLP) 블록으로 구성된 경량 네트워크입니다. 이 모듈은 오디오-텍스트 임베딩 쌍을 사용하여 두 분포를 명시적으로 연결하고, 역 과정을 통해 텍스트 임베딩 분포로 매핑하는 방식으로 작동합니다. 특히, 정규화된 가우시안 노이즈를 주입하여 오디오 임베딩의 구조를 유지하면서 효과적인 모달리티 브리지를 구현합니다.

- **Performance Highlights**: Diffusion-Link를 멀티모달 LLM 베이스라인에 추가하는 방식으로, AudioCaps 데이터셋에서 제로샷 오디오 캡셔닝의 성과가 52.5% 향상되고, 완전히 감독된 캡셔닝에서도 7.5% 향상을 보여주며, 이는 외부 지식 없이는 도달할 수 없었던 최첨단 결과입니다. 이 연구는 모달리티 갭을 줄이는 것이 효과적인 멀티모달 인코더와 LLM 간 coupling을 위해 필수적임을 보여주고, 확산 기반 모달리티 브리지가 새로운 방향성을 제공한다는 점에서 중요한 의의를 갖습니다.



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



### Network-Optimised Spiking Neural Network (NOS) Scheduling for 6G O-RAN: Spectral Margin and Delay-Tail Contro (https://arxiv.org/abs/2510.11291)
Comments:
          6 pages, 5 figures, 1 table

- **What's New**: 본 연구는 6G 라디오 액세스를 위한 지연 인식(Network-Optimised Spiking, NOS) 스케줄러를 제시합니다. 이 스케줄러는 두 개의 상태를 가진 커널과 비례 공정(Proportional Fair, PF) 보조금을 결합하여 작동합니다. NOS는 지연, 간섭 위상, 제어 강도를 하나의 스펙트럼 여유 매개변수로 집계하여 설계에 통합합니다.

- **Technical Details**: NOS 스케줄러는 유한 버퍼를 대표하는 경계 있는 흥분 상태와 반복적인 보조금을 억제하는 회복 상태를 이용합니다. 이 모델은 여러 사용자 기기의 arrival을 처리하기 위해 복합-포아송(arrivals) 도착을 기반으로 하며, 지연된 이웃 영향을 모델화하여 지연 인식을 강화합니다. 두 개 상태의 요소를 통해 대기열 역학과 슬롯 수준 리소스 할당 메커니즘 간의 강한 연결 고리를 형성합니다.

- **Performance Highlights**: 수치 연구 결과 NOS는 PF 및 지연 백프레셔(Backpressure) 방식과 비교하여 높은 이용률과 더 적은 99.9번째 백분위수 지연을 유지하며, 정수 기반 PRB에서는 clique-유효성을 유지하였습니다. NOS는 단일 안전 관련 매개변수를 통해 통제 가능성을 제공하고, 지연 인식 설계를 통해 O-RAN 설정에서 효과적으로 작동합니다.



### SeFEF: A Seizure Forecasting Evaluation Framework (https://arxiv.org/abs/2510.11275)
Comments:
          main document: 14 pages, 9 figures, 2 tables; appendix: 7 pages, 2 figures, 3 tables, 2 algorithms

- **What's New**: 이 논문에서는 발작 예측 모델 개발에 대한 표준화 부족 문제를 해결하기 위해 Python 기반의 Seizure Forecasting Evaluation Framework (SeFEF)를 소개합니다. 이 프레임워크는 데이터 라벨링, 교차 검증, 성능 평가 및 보고 등의 절차를 자동화하여 발작 예측 알고리즘의 개발과 평가를 간소화합니다. 다양한 예측 수평을 지원하며 구현 세부정보, 훈련 및 평가 설정, 성능 지표를 문서화하는 모델 카드를 포함하고 있습니다.

- **Technical Details**: 발작 예측 작업은 특정 시간 내에 발작이 발생할 확률을 예측하는 것으로 정의됩니다. 이 프레임워크는 데이터 준비, 라벨링, 모델 개발, 예측 및 성능 평가 등 여러 단계로 구성됩니다. 사용자는 데이터 준비를 커스터마이즈할 수 있으며, HDF5 파일 형식으로 입력 데이터를 요구합니다. SeFEF는 시계열 교차 검증(TSCV) 기법을 채택하여 모델 평가의 신뢰도를 높입니다.

- **Performance Highlights**: 프레임워크의 유연성을 보여주기 위해 3가지 알고리즘 접근 방식을 구현하였으며, 이들은 다양한 알고리즘 평가의 KPI를 사용하여 성능을 측정했습니다. 초기 사용자의 경험을 통해 개발 시간 단축과 방법론적 일관성을 향상시키는 긍정적인 효과가 관찰되었습니다. 사용된 MSG2022 데이터셋은 6명의 환자에서 수집된 연속적이고 장기간의 생리학적 데이터를 포함하고 있으며, 이 데이터셋은 SeFEF의 개념 증명으로 사용되었습니다.



### DemoHLM: From One Demonstration to Generalizable Humanoid Loco-Manipulation (https://arxiv.org/abs/2510.11258)
- **What's New**: 본 논문에서는 DemoHLM이라는 새로운 프레임워크를 제안하여 인간형 로봇의 loco-manipulation(움직이면서 조작하는 작업)을 가능하게 합니다. 이 프레임워크는 단 하나의 시연(demonstration)만으로도 다양한 환경에서 재현 가능한 작업을 수행할 수 있도록 설계되었습니다. DemoHLM은 저수준의 범용 전체 몸체 제어기와 고수준의 조작 정책을 통합하여 여러 작업을 수행할 수 있도록 합니다.

- **Technical Details**: DemoHLM은 모방학습(imitation learning)과 데이터 생성(data generation)을 활용하여 고유한 시뮬레이션 기반 데이터 생성 파이프라인을 가지고 있습니다. 이 프레임워크는 이동, 사전 조작(pre-manipulation), 조작(manipulation) 단계로 구성된 성공적인 시연을 수집하고, 이를 바탕으로 새로운 궤적(trajectory)을 생성합니다. 최종적으로, 학습된 정책은 로봇이 다양한 loco-manipulation 작업을 수행할 수 있는 자기 주도적 조작 정책을 제공합니다.

- **Performance Highlights**: 실험 결과, 시뮬레이션 데이터의 양이 증가함에 따라 정책 성능이 개선되는 긍정적인 상관관계를 보였습니다. 이를 통해 DemoHLM의 데이터 생성 파이프라인의 효과성과 접근 방식을 증명하였습니다. 실제 Unitree G1 로봇에서 실험한 결과, 학습된 정책이 모든 작업에서 시뮬레이션과 동등한 성능을 발휘하여 강력한 시뮬레이션-실제 전이 가능성을 입증하였습니다.



### Large Language Models Are Effective Code Watermarkers (https://arxiv.org/abs/2510.11251)
- **What's New**: 최근 대규모 언어 모델(LLMs)의 발전과 오픈소스 생태계의 확장으로 인해 소스 코드의 무단 사용과 관련된 윤리적 및 보안적 문제들이 부각되고 있습니다. 이에 대한 해결책으로 제안된 CodeMark-LLM은 코드의 의미와 가독성을 저해하지 않으면서도 소스 코드에 워터마크를 삽입할 수 있는 프레임워크입니다. 본 연구는 기존의 방법과 달리, 수작업 규칙이나 특정 훈련을 필요로 하지 않고, 다양한 프로그래밍 언어에 적용 가능한 방식으로 설계되었습니다.

- **Technical Details**: CodeMark-LLM은 두 가지 주요 구성 요소로 이루어져 있습니다: (i) Semantically Consistent Embedding 모듈은 기능 보존 변환을 사용하여 워터마크 비트를 인코딩합니다. (ii) Differential Comparison Extraction 모듈은 원본 코드와 워터마크가 적용된 코드를 비교하여 변환을 식별합니다. 이러한 구조는 LLM의 크로스 링구얼 일반화 능력을 활용하여 언어에 특화된 엔지니어링 없이도 작동할 수 있습니다.

- **Performance Highlights**: 실험 결과, CodeMark-LLM은 다양한 프로그래밍 언어와 공격 시나리오에서 강력한 무결성과 효율성을 보여주었습니다. 특히, 제공된 워터마크는 문법 검사와 단위 테스트를 거의 100%의 비율로 통과할 수 있었습니다. 따라서 LLM이 코드 워터마킹을 위한 효율적이고 확장 가능한 솔루션을 제공할 수 있는 큰 가능성을 가지고 있다는 것을 입증했습니다.



### Analyzing Data Quality and Decay in Mega-Constellations: A Physics-Informed Machine Learning Approach (https://arxiv.org/abs/2510.11242)
Comments:
          76th International Astronautical Congress

- **What's New**: 이 연구는 Starlink 위성의 저궤도(LEO) 메가 별자리 및 관련 공공 데이터를 평가하며, 특히 데이터의 정확도와 신뢰성을 분석합니다. 연구의 두 가지 주요 목표는 (i) 고정밀(Numerical Propagation) 방법과의 비교, (ii) 물리적 요소를 고려한 기계학습(Physics-Informed Machine Learning)을 이용한 위성 결과값 추출입니다. 상당수의 Starlink 위성의 실제 궤도 데이터를 분석하여, 공개된 에페메리스를 검증하고 비보존 힘을 탐색하는 데이터 기반 모델을 제안합니다.

- **Technical Details**: 저궤도(LEO)에 있는 위성의 수가 지속적으로 증가함에 따라, 정확한 궤도 예측(collision avoidance)과 충돌 회피는 필수적입니다. 이를 위해, 연구에서는 Orekit이라는 고정밀 궤도 솔버를 사용하여 Starlink의 에페메리스를 평가합니다. 또한 Neural Ordinary Differential Equations (NODE)을 사용하여 데이터를 기반으로 위성 궤도를 예측하고 있습니다.

- **Performance Highlights**: Starlink 위성의 비보존 힘과 궤도시 대기 저항을 분석한 결과, 비확인된 힘이 궤도 복구 과정에서 큰 영향을 미치는 것으로 나타났습니다. 전체 위성의 Root Mean Square Error (RMSE)는 비탈출 위성의 경우 약 300m, 탈출 위성의 경우 약 600m로 증가했습니다. 이는 공개 데이터의 한계가 존재하며, 우주 상황 인식을 위한 개선의 필요성을 제기합니다.



### LightPneumoNet: Lightweight Pneumonia Classifier (https://arxiv.org/abs/2510.11232)
Comments:
          13 pages (including references), 5 figures

- **What's New**: 본 연구에서는 LightPneumoNet이라는 경량의 합성곱 신경망(CNN) 모델을 제안합니다. 이 모델은 흉부 X-레이에서 폐렴을 정확하게 진단할 수 있도록 설계되었으며, 5,856개의 공개 데이터셋을 기반으로 학습되었습니다. 기존의 복잡한 아키텍처와 달리, 이 모델은 388,082개의 학습 가능한 매개변수로 구성되어 있어 메모리 사용량이 1.48MB로 최소화되었습니다.

- **Technical Details**: LightPneumoNet은 224x224로 이미지 크기를 조정하고, 그레이스케일 변환 및 픽셀 정규화를 포함한 전처리 과정을 거쳤습니다. 데이터 증강(rotation, zoom, shear) 기법을 적용하여 과적합(overfitting)을 방지하였습니다. 이 모델은 4개의 합성곱 층 블록으로 이루어져 있으며, 복잡한 전이 학습(Transfer Learning) 전략에 의존하지 않고 효율적인 구조를 갖추고 있습니다.

- **Performance Highlights**: 모델은 독립적인 테스트 세트에서 94.2%의 전체 정확도, 92%의 정밀도 및 96%의 F1 스코어를 달성하여 뛰어난 성능을 보였습니다. 특히, 민감도(Recall)는 99%로 폐렴 사례를 효과적으로 식별하며 임상적으로 знач여지는 잘못된 음성율을 최소화하였습니다. 이 모델은 저비용 하드웨어에서도 배치할 수 있어 의료 격차가 있는 지역에서도 접근 가능한 진단 도구로써 병원에 기여할 수 있을 것으로 기대됩니다.



### Discursive Circuits: How Do Language Models Understand Discourse Relations? (https://arxiv.org/abs/2510.11210)
Comments:
          Accepted to EMNLP 2025 (Main Conference); 9 pages, 8 figures, 5 tables (20 pages, 12 figures, 14 tables including references and appendices)

- **What's New**: 이 논문에서는 Transformer 언어 모델에서 담화 이해(Discourse Understanding)에 중요한 역할을 하는 구성 요소를探求합니다. 저자들은 'discursive circuits'라는 희소 계산 그래프(sparse computational graphs)가 담화 관계 처리에 영향을 미친다고 가정합니다. 이를 통해 기존의 단순한 작업에서 벗어나 복잡한 담화를 처리하는 방법을 제시합니다.

- **Technical Details**: 연구에서는 'Completion under Discourse Relation (CuDR)'라는 새로운 작업을 도입하여 담화 관계를 기반으로 모델이 담화를 완성하도록 합니다. 이를 위해 Penn Discourse Treebank (PDTB), Rhetorical Structure Theory (RST), Segmented Discourse Representation Theory (SDRT)와 같은 주요 담화 프레임워크를 아우르는 데이터셋을 구축하였습니다. 논문에서는 특히 activation patching 기법을 통해 모델의 성능을 평가하고, 0.2%의 모델 연결만으로도 담화 이해가 가능하다는 것을 증명합니다.

- **Performance Highlights**: 실험 결과, 찾은 담화 회로는 GPT-2 모델에서 약 90%의 신뢰도를 달성했습니다. 이 회로들은 PDTB에서 도출되었으며, RST 및 SDRT와 같은 보지 못한 담화 프레임워크에도 잘 일반화된다는 결과를 보여주었습니다. 저자들은 새로운 담화 계층 구조를 통해 서로 다른 프레임워크 간 비교가 가능하도록 하여, 언어 모델의 담화 관계에 대한 일관된 표현을 제안합니다.



### Efficient In-Memory Acceleration of Sparse Block Diagonal LLMs (https://arxiv.org/abs/2510.11192)
Comments:
          8 pages, to appear in IEEE Cross-disciplinary Conference on Memory-Centric Computing (CCMCC)

- **What's New**: 이 논문에서는 리소스 제약이 있는 시스템에서 대규모 언어 모델(LLM)을 효율적으로 구현하기 위해 구조적 희소성(structured sparsity)을 활용합니다. Dense-to-sparse fine-tuning 방법을 통해 모델 크기를 6.7배 이상 줄이면서도 정확도를 유지하는 성과를 보여줍니다. 하지만 기존 Von-Neumann 아키텍처에서는 메모리 제약으로 인해 LLM의 디코드 단계가 매우 비용이 많이 드는 문제를 해결하기 위해 Compute-in-memory (CIM) 아키텍처를 도입합니다.

- **Technical Details**: 이 논문에서는 CIM 가속기에서 희소 LLM 추론을 가속화하기 위한 자동화된 프레임워크를 제안합니다. 새로운 매핑(mapping) 및 스케줄링(scheduling) 전략을 사용하여 블록 대각 희소성을 이용해 CIM 배열의 활용도를 50% 이상 향상시키며, 메모리 풋프린트 및 필요한 부동 소수점 연산의 수를 각각 4배 이상 줄입니다. 특히, 블록 대각 행렬을 CIM 배열에 밀집하게 매핑하는 전략이 두 가지 최적화를 통해 수행됩니다.

- **Performance Highlights**: 제안한 프레임워크는 CIM 기반의 밀집 모델과 비교하여 실행 시간과 에너지 소비를 각각 1.7배 이상 줄입니다. 또한 메모리 풋프린트를 4배 이상 감소시키며 성능을 극대화합니다. 이 연구는 CIM에서 사용할 모델의 자동 변환 및 최적화를 통합하여 저전력 고효율의 LLM 구현을 가능하게 합니다.



### Machine Learning-Integrated Hybrid Fluid-Kinetic Framework for Quantum Electrodynamic Laser Plasma Simulations (https://arxiv.org/abs/2510.11174)
- **What's New**: 이번 연구는 고강도 레이저 플라즈마 상호작용을 다루는 새로운 기계 학습 기반의 3차원 하이브리드 유체-입자 격자 시스템을 도입합니다. 이 시스템은 안정적인 영역에서는 유체 근사를 사용하고, 불안정한 영역에서는 SwitchNet이 지시하여 PIC 솔버를 작동시킵니다. 이를 통해 플라즈마 행동을 자동으로 전이시키는 모델링 방법이 제시됩니다.

- **Technical Details**: 모델은 Ammosov-Delone-Krainov (ADK) 터널링과 다광자 이온화율 간의 매끄러운 전환을 사용하여 이온화를 시뮬레이션하며, Airy 함수 근사를 통해 복사 반응 및 쌍 생성에 대한 양자 전자역학(QED) 효과를 시뮬레이션합니다. 또한, 컨볼루션 신경망은 에너지 보존을 위해 물리 기반 손실 함수를 사용하고, 채널당 정규화된 필드에서 작동합니다. 몬테 카를로 드롭아웃 기법은 불확실성 측정을 제공합니다.

- **Performance Highlights**: 이 하이브리드 모델은 모든 필드 구성 요소에 대해 결정 계수(R^2)가 0.95 이상, 평균 제곱 오차가 10^-4 이하의 정확한 예측 결과를 생성합니다. 이러한 적응형 접근 방식은 레이저-플라즈마 시뮬레이션의 정확성과 확장성을 향상시키며, 고에너지 밀도 및 입자 가속 애플리케이션을 위한 통합 예측 프레임워크를 제공합니다.



### PAC-Bayesian Bounds on Constrained f-Entropic Risk Measures (https://arxiv.org/abs/2510.11169)
- **What's New**: 이번 논문은 PAC (Probably Approximately Correct) 일반화 경계가 데이터의 하위 그룹 사이의 불균형을 포착하는 데 부족함을 해결하기 위해, 새로운 위험 측정 방식인 constrained f-entropic risk measure를 도입합니다. 이 방법은 f-divergences를 통해 배급의 변동성과 하위 그룹 간의 불균형을 정교하게 조절할 수 있으며, 잘 알려진 위험 측정 방식인 Conditional Value at Risk (CVaR)를 포함합니다. 본 연구는 이 가족의 위험에 대한 고전적이고 분리된 PAC-Bayesian 일반화 경계를 유도하였고, 이는 기존의 위험을 넘어서는 첫 번째 분리된 PAC-Bayesian 보장을 제공합니다.

- **Technical Details**: 논문에서는 학습 알고리즘이 모집단의 하위 그룹을 고려하여 위험을 재조정하는 새로운 위험 측정 방식을 제안합니다. 특히, 데이터의 하위 그룹을 파티셔닝하고 이를 기반으로 재가중치된 위험을 계산하는 방법을 다룹니다. 이를 통해, 데이터 분포가 불균형할 경우에도 효과적으로 하위 그룹 수준에서의 보장을 제공하는 모델을 설계할 수 있습니다. 또한, 제안된 self-bounding 알고리즘은 이러한 경계를 직접 최소화하여, 하위 그룹 수준에서 보증을 제공하는 모델을 생성합니다.

- **Performance Highlights**: 제안된 방법론의 유용성을 논문 내 실험을 통해 검증하며, 기존의 PAC-Bayesian 경계와 비교할 때 더 나은 성능을 보이는 것으로 확인되었습니다. 데이터의 하위 그룹 불균형을 효과적으로 처리함으로써, 제대로 대표되지 않는 소수 그룹의 오차율을 줄이는 데 기여할 수 있습니다. 이러한 연구는 머신러닝의 실제 적용 상황에서의 불공정성을 줄이는 데 중요한 의미를 지닙니다.



### Enhanced Sampling for Efficient Learning of Coarse-Grained Machine Learning Potentials (https://arxiv.org/abs/2510.11148)
- **What's New**: 이 연구에서는 coarse-graining (CG)에서의 기계 학습 포텐셜 (MLP)의 정확성과 신뢰성을 높이기 위해 향상된 샘플링 기법을 도입합니다. 기존의 CG MLP는 일반적으로 힘 일치를 통해 학습되며, 이는 균형 잡힌 볼츠만 분포에서 샘플링한 구성에 의존하지만 이 방식은 두 가지 주요 한계를 가지고 있습니다. 이 논문은 CG 자유도에 따라 편향된 데이터를 생성하고, 이를 통해 강한 전이 영역을 더 잘 샘플링할 수 있는 새로운 방법론을 제시합니다.

- **Technical Details**: 모델링 과정은 원자적(AT) 설명에서 coarse-grained (CG) 변수 집합으로의 매핑 정의로 시작됩니다. 이 연구에서는 선형 및 직교 매핑을 가정하여 열역학적 일관성을 유지하며, 최종적으로 CG 모델의 균형 분포가 AT 분포를 재현하도록 구성됩니다. 이를 통해 기존의 CG MLP에서 발생하는 여러 한계를 극복하고, 효율적인 데이터 생성을 가능하게 합니다.

- **Performance Highlights**: 이 연구는 Müller-Brown 포텐셜과 capped alanine에서 이 방법의 효과를 입증하였으며, 주목할 만한 개선을 확인할 수 있었습니다. 향상된 샘플링 방법을 통해 CG MLP의 정확도와 신뢰성을 함께 높이는 방향이 제시되었으며, 이는 향후 CG 모델의 적용을 넓힐 수 있는 잠재력을 가집니다.



### torchsom: The Reference PyTorch Library for Self-Organizing Maps (https://arxiv.org/abs/2510.11147)
Comments:
          4 mains pages with 2 tables, 4 pages of references, 15 pages of appendices with 13 figures and 3 tables

- **What's New**: 이 논문은 PyTorch로 구현된 Self-Organizing Map(SOM)의 참조 구현을 제공하는 오픈 소스 Python 라이브러리인 torchsom을 소개합니다. 이 패키지는 (i) 차원 축소, (ii) 클러스터링, (iii) 데이터 시각화의 세 가지 주요 기능을 제공합니다. PyTorch 백엔드를 사용하여 GPU 가속을 통한 빠르고 효율적인 SOM 학습 및 PyTorch 생태계와의 쉬운 통합이 가능하며, scikit-learn API를 따릅니다.

- **Technical Details**: torchsom은 모듈화된 디자인으로 구성되어 있으며 세 가지 주요 컴포넌트를 포함합니다. core 모듈은 전통적인 SOM 알고리즘을 구현하며, utils 모듈은 SOM 파라미터화와 학습을 위한 필수 구성 요소를 제공합니다. 마지막으로 visualization 모듈은 U-matrix, 히트 맵, 구성 요소 평면 등 다양한 시각화 도구를 제공하여 SOM의 시각적 해석을 돕습니다.

- **Performance Highlights**: torchsom은 MiniSom과 비교하여 성능과 정확도를 평가합니다. 다양한 샘플 크기 및 특징 차원에서 데이터 세트를 생성하여 성능 스케일링을 평가하였으며, 동일한 하이퍼파라미터를 사용하여 공정한 비교를 수행합니다. 결과적으로 torchsom은 CPU 및 GPU 모두에서 MiniSom보다 우수한 성능을 보여주어 SOM 기반의 데이터 분석에서 강력한 도구로 자리 잡고 있습니다.



### Graph Neural Network-Based Multicast Routing for On-Demand Streaming Services in 6G Networks (https://arxiv.org/abs/2510.11109)
- **What's New**: 이 논문은 6G 무선 네트워크에서의 효율적인 멀티캐스트 라우팅을 위해 새로운 그래프 신경망(GNN) 기반 접근법을 제시합니다. 기존의 라우팅 알고리즘들이 동적이고 대규모 환경에서의 요구를 충족하지 못하는 문제를 해결하기 위해, GNN을 활용하여 사용자 맞춤형 비디오 품질 요구 사항을 지원하면서 전체 전송 비용을 최소화하는 방법을 개발했습니다.

- **Technical Details**: 논문은 라우팅 문제를 제약이 있는 최소 흐름 최적화 작업으로 정의하고, 강화 학습 알고리즘을 통해 멀티캐스트 트리를 효율적으로 구성합니다. 그래프 주의 네트워크(GAT)를 인코더로 사용하여 맥락 인식 노드 임베딩을 추출하고, 장기 단기 기억(Long Short-Term Memory, LSTM) 모듈로 라우팅 결정의 순차적 의존성을 모델링합니다.

- **Performance Highlights**: 시뮬레이션 결과, 제안된 방법은 최적의 동적 프로그래밍 솔루션에 근접한 성능을 보이며 상당한 계산 복잡성 감소를 나타냅니다. 또한 대규모 및 동적 네트워크 구조에 대한 강력한 일반화 능력을 확인하여, 6G 멀티미디어 전송 시나리오에서 실시간 배포 가능성을 강조합니다.



### PhysHSI: Towards a Real-World Generalizable and Natural Humanoid-Scene Interaction System (https://arxiv.org/abs/2510.11072)
Comments:
          Project website: this https URL

- **What's New**: 이번 연구에서는 PhysHSI라는 새로운 시스템을 소개합니다. 이 시스템은 인간형 로봇이 다양한 환경에서 자연스럽고 생동감 있게 상호작용할 수 있도록 설계되었습니다. PhysHSI는 시뮬레이션 훈련 파이프라인과 실제 배포 모듈로 구성되어, 로봇이 복잡한 상호작용 작업을 자동으로 수행할 수 있도록 지원합니다.

- **Technical Details**: PhysHSI는 Adversarial Motion Prior (AMP) 기반의 정책 학습을 통해 다양한 시나리오에서 자연스러운 동작을 실현합니다. 또한, LiDAR와 카메라를 조합하여 물체 위치를 정밀하게 파악하는 조정된 인식 모듈을 도입했습니다. 이러한 설계로 인해 PhysHSI는 현실 세계의 복잡한 환경에서도 효율적으로 동작할 수 있습니다.

- **Performance Highlights**: PhysHSI는 네 가지 대표적인 HSI 작업—상자 나르기, 앉기, 눕기, 일어기—에 대해 높은 성공률과 강력한 일반화를 보여주었습니다. 로봇은 다양한 작업 목표와 시나리오에 따라 자연스럽고 표현력 있는 동작을 수행할 수 있습니다. 이러한 결과는 PhysHSI가 실제 환경에서 일반적인 상호작용 기술을 효과적으로 습득하고 적용할 수 있음을 보여줍니다.



### GrASP: A Generalizable Address-based Semantic Prefetcher for Scalable Transactional and Analytical Workloads (https://arxiv.org/abs/2510.11011)
Comments:
          This is a preprint version

- **What's New**: 이번 논문에서는 GrASP라는 새로운 학습 기반 데이터 프리페처를 제안합니다. GrASP는 분석 및 거래(workload) 작업 모두에 적합하게 설계되었으며, 기존의 접근 방식을 개선하여 예측 정확도를 높이고 확장성을 강화합니다. 이 프리페처는 논리 블록 주소 델타(logical block address deltas)를 통해 데이터 접근 패턴을 예측하고, 다층 LSTM(multi-layer LSTM)을 사용하여 임베디드 컨텍스트(embedded context)에서 델타 패턴을 추출합니다.

- **Technical Details**: GrASP는 프리페칭을 컨텍스트 인식 멀티 레이블 클래스 분류(context-aware multi-label classification) 문제로 구성하여, 쿼리의 성격(query semantics)와 결과 인코딩(result encodings)을 결합합니다. 이 접근 방식은 샘플 학습을 통해 데이터 세트가 동적으로 변화하는 환경에서도 예측을 일반화할 수 있도록 합니다. GrASP는 고유한 손실 함수(custom loss function)와 dropout 정규화를 적용하여 클래스 불균형 문제를 완화하고 일반화 성능을 개선합니다.

- **Performance Highlights**: 실험 결과, GrASP는 평균 91.4%의 적중률(hit ratio)을 달성하며, I/O 시간을 90.8% 줄이고 실행 지연(latency)을 57.1% 감소시킵니다. 기존의 최첨단(predecessors) 프리페처와 비교할 때, GrASP는 분석 작업에서 적중률을 최대 17% 그리고 거래 작업에서는 45%까지 개선했습니다. 이러한 성과는 GrASP의 탁월한 일반화 능력과 프리페칭 정확도를 나타냅니다.



### ABLEIST: Intersectional Disability Bias in LLM-Generated Hiring Scenarios (https://arxiv.org/abs/2510.10998)
Comments:
          28 pages, 11 figures, 16 tables. In submission

- **What's New**: 본 논문은 대형 언어 모델(LLMs)이 채용 분야에서 장애인(PwD)에 대한 정체성 기반 차별을 지속하고 있다는 점을 강조합니다. 특히 연구는 글로벌 남반부에서 성별, 계급 등의 교차적 소외 형태가 장애인의 경험에 미치는 영향을 간과하고 있음을 지적합니다. 새로운 평가 지표인 ABLEIST를 도입하여 장애인 관련 편향을 정밀하게 측정하고, 기존 모델의 안전 도구들이 이 문제를 제대로 탐지하지 못하는 문제점을 밝혔습니다.

- **Technical Details**: 리서치 팀은 2,820개의 다양한 채용 시나리오를 생성하여 6개의 LLM의 포괄적인 감사(Audit)를 시행했습니다. 이를 통해 생성된 대화에서의 ABLEIST 지표를 통해 미세한 형태의 교차적 편향을 검출하기 위해, 장애 연구 문헌에 토대를 둔 새로운 측정기준을 설정하였습니다. 평가 결과, 장애인을 대상으로 한 대화에서 99.7%의 경우 ABLEIST 차별이 발견되었으며, 특정 장애 유형에 따라 차별의 형태가 다양하게 나타났습니다.

- **Performance Highlights**: 연구결과, 현재 사용되고 있는 안전 도구들은 미세한 장애 및 교차적 편향을 탐지할 수 없는 한계를 드러냈습니다. LLM 모델을 사용한 채용대화에서 장애인 후보자는 비장애 후보자에 비해 평균 58배 더 많은 ABLEIST 피해를 경험했습니다. 이러한 결과는 고위험 도메인에서 교차적 안전 평가의 필요성을 강조하며, 공정한 채용을 위한 새로운 기준 수립의 필요성을 제안합니다.



### Adversarial Robustness in One-Stage Learning-to-Defer (https://arxiv.org/abs/2510.10988)
- **What's New**: 이 논문에서는 Learning-to-Defer (L2D) 시스템의 새로운 프레임워크를 제시하여, 예측기와 할당 정책이 공동으로 최적화되는 일단계(one-stage) L2D의 적대적 강인성(adversarial robustness)을 다룹니다. 기존의 연구들이 보통 두 단계의 설정에만 초점을 맞춘 것과는 달리, 우리는 예측 및 전문가 할당을 동시에 다루는 방법을 뛰어넘어 새로운 공격 방식과 비용 민감한 적대적 서브로게 손실(cost-sensitive adversarial surrogate losses)을 제안합니다.

- **Technical Details**: 우리의 접근 방식은 기본적인 공격의 정의를 공식화하고, 여러 이론적 보장을 제시합니다. 여기에는 분류(classification)와 회귀(regression) 설정에서의 Bayes 일관성(Bayes consistency), ℋ-일관성(ℋ-consistency), 그리고 (ℛ,ℱ)-일관성((,f)-consistency) 등이 포함됩니다. 이러한 이론적 기틀을 바탕으로, 제안된 방법은 이미지 분류(image classification)와 표형 회귀(tabular regression) 벤치마크에서 평가되었습니다.

- **Performance Highlights**: 실험 결과, 우리는 제안된 방법이 비목표 공격(untargeted attacks)과 목표 공격(targeted attacks) 모두에 대해 강인성을 크게 향상시키면서도 정확도(clean accuracy)를 유지함을 보여줍니다. 따라서, 본 논문은 적대적 환경에서도 강력 및 실용적인 일단계 L2D 접근 방식의 첫 번째 기초를 제공합니다.



### In-Context Learning Is Provably Bayesian Inference: A Generalization Theory for Meta-Learning (https://arxiv.org/abs/2510.10981)
- **What's New**: 본 논문은 다양한 작업 유형의 혼합을 수용하는 메타-학습 프레임워크 내에서 in-context learning (ICL)을 위한 유한 샘플 통계 이론을 발전시킵니다. 우리는 총 ICL 위험을 Bayes Gap과 Posterior Variance라는 두 개의 직교 구성 요소로 분리하는 원칙적인 위험 분해를 소개합니다. 이를 통해 학습된 모델이 Bayes-최적 in-context 예측기를 얼마나 잘 근사하는지를 정량화할 수 있는 방법을 제시합니다.

- **Technical Details**: 우리는 uniform-attention Transformer에 대해 Bayes Gap의 비대칭 상한선을 도출하였으며, 이는 사전 훈련 프롬프트의 수와 그들의 맥락 길이의 의존성을 명확히 보여줍니다. Posterior Variance는 모델 독립적인 위험을 나타내며, 이는 작업의 내재적 불확실성을 나타냅니다. 본 연구는 Bayes-optimal 예측기를 optimal in-context 예측기로 보고 제곱 손실 하에 이의 직교 분해를 진행합니다.

- **Performance Highlights**: 중요한 발견은, Posterior Variance가 실제 작업의 난이도에 의해 결정된다는 것입니다. 또한, 테스트 시험(테스트 시간) 동안 Transformer가 최적의 메타 알고리즘을 선택하고 빠르게 실제 작업을 위한 최적 알고리즘으로 수렴한다는 것을 보여줍니다. 여기서 가정한 바와 같이, 다양한 작업 유형의 혼합에서도 최적의 메타 알고리즘이 빠르게 실제 작업에 대한 최적의 알고리즘으로 수렴함을 시사합니다.



### Chart-RVR: Reinforcement Learning with Verifiable Rewards for Explainable Chart Reasoning (https://arxiv.org/abs/2510.10973)
Comments:
          23 pages

- **What's New**: 이번 논문에서는 Large Vision-Language Models (LVLMs)의 한계를 해결하기 위해 새로운 프레임워크인 Chart-RVR를 제안합니다. 이 프레임워크는 Group Relative Policy Optimization (GRPO)와 자동 검증 가능한 보상을 결합하여 차트 추론에서의 견고성과 설명 가능성을 증대시킵니다. 특히, Chart-RVR은 차트 유형 분류, 표 구조 재구성, 절차 일치율을 최적화하는 세 가지 보상을 포함하고 있습니다.

- **Technical Details**: Chart-RVR는 3억 개의 매개변수를 가진 LVLM에 적용되어 표준 감독 미세 조정(supervised fine-tuning, SFT) 방식보다 우수한 성능을 보입니다. 새로운 방법론은 Reinforcement Fine-Tuning (RFT) 기법을 활용하여 모델의 예측 정확도를 높이고, 지속가능한 훈련을 가능하게 하며, 두 가지 목표(tasks)인 차트 유형 예측과 표 재구성을 위한 검증 가능한 보상을 사용합니다. 마지막으로, GRPO와 검증 가능한 보상의 결합은 모델의 훈련 시 안정성을 증가시킵니다.

- **Performance Highlights**: Chart-RVR 모델은 6개의 차트 추론 벤치마크에서 최첨단 성능을 달성하여 기존의 유사한 크기의 모델들을 초월합니다. 이 연구는 또한 모델이 더 해석 가능한 CoT (chain-of-thought) 논리를 생성함으로써 신뢰성과 신뢰성을 강화하는 방법을 보여줍니다. 종합적으로, Chart-RVR은 차트 추론의 정확도와 해석 가능성 모두에서 개선을 나타내었습니다.



### Comparative Evaluation of Neural Network Architectures for Generalizable Human Spatial Preference Prediction in Unseen Built Environments (https://arxiv.org/abs/2510.10954)
Comments:
          The 15th International Workshop on Structural Health Monitoring (IWSHM)

- **What's New**: 이 논문은 Cyber-Physical-Social Infrastructure Systems (CPSIS) 내에서 인간의 공간 선호도를 예측하는 능력에 대해 다룹니다. 특히, 훈련 중에 경험하지 못한 환경 구성에서의 선호도를 예측하는 모델의 일반화 가능성을 조사합니다. 다양한 신경망 구조의 효과를 비교하여, 어떤 모델이 새로운 레이아웃을 가장 잘 일반화하는지 알아봅니다.

- **Technical Details**: 우리는 Graph Neural Networks, Convolutional Neural Networks, 그리고 표준 feedforward Neural Networks의 비교 연구를 수행했습니다. 공원 환경을 간소화하여 생성된 합성 데이터를 사용하여 각 모델의 성능을 평가하였습니다. 각 모델은 이질적인 물리적, 환경적, 사회적 특징으로 영향을 받는 선호도를 예측하는 능력에 따라 평가됩니다.

- **Performance Highlights**: 다양한 신경망 아키텍처의 일반화 점수는 precision-recall curve의 아래 면적을 기반으로 계산됩니다. 이러한 접근 방법은 불균형 데이터에 적합하며, 보지 않은 새로운 환경에서의 인간 행동 모델링에 있어 각 신경망 구조의 타당성을 평가하는 데 도움을 줍니다.



### End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF: A Reproducibility Study (https://arxiv.org/abs/2510.10936)
- **What's New**: 이번 연구에서는 Ma와 Hovy (2016)에서 제안한 BiLSTM-CNN-CRF 아키텍처의 재현성 연구를 소개합니다. 이 모델은 CNN, BiLSTM, CRF의 세 가지 주요 구성요소를 결합하여 시퀀스 레이블링 작업에서 뛰어난 성능을 발휘하며, 수작업 특징을 제거하는 엔드 투 엔드 방식으로 구현됩니다. 연구 결과, CoNLL-2003 NER 데이터셋에서 91.18%의 F1 스코어를 달성하여 모델의 효과성을 입증하였습니다.

- **Technical Details**: BiLSTM-CNN-CRF 모델은 문자 수준의 CNN 인코딩, 단어 수준의 BiLSTM 인코딩, CRF 기반의 구조적 예측을 이용한 세 가지 주요 구성 요소로 구성됩니다. 각 문자는 CNN을 통해 형성적 정보를 추출하며, 단어 임베딩과 결합하여 BiLSTM을 거쳐 최종적으로 태그 점수를 생성합니다. CRF 레이어는 태그 간의 의존성을 고려하여 일관된 태그 시퀀스를 보장합니다.

- **Performance Highlights**: 모델은 CoNLL-2003 NER 데이터셋에서 91.18% F1 스코어를 달성하며 원 논문의 결과와 유사한 성과를 보입니다. Penn Treebank WSJ POS 태깅에서는 97.52%의 정확도로 원래 97.55%의 성과와 거의 일치하는 결과를 나타내었습니다. 구성 요소의 기여도를 분석하기 위한 제거 연구를 통해 CRF 레이어가 태그 일관성 보장에 중요한 역할을 한다는 것을 발견했습니다.



### FG-CLIP 2: A Bilingual Fine-grained Vision-Language Alignment Mod (https://arxiv.org/abs/2510.10921)
- **What's New**: FG-CLIP 2는 영어와 중국어를 위한 새로운 이중 언어( bilingual) 비전-언어( vision-language) 모델로, 미세한 정렬(fine-grained alignment)을 개선하기 위해 개발되었습니다. 기존의 모델들이 언어 표현, 시각적 콘텐츠, 및 객체 속성의 세밀한 정렬에서 어려움을 겪던 문제를 해결하기 위해, FG-CLIP 2는 지역-텍스트 매칭(region-text matching)과 긴 캡션 모델링(long-caption modeling) 기법을 활용하고 있습니다. 또한, Textual Intra-modal Contrastive (TIC) 손실을 도입하여 의미적으로 유사한 캡션을 더 잘 구분할 수 있도록 설계되었습니다.

- **Technical Details**: FG-CLIP 2는 두 단계 학습 프레임워크를 따릅니다. 첫 번째 단계에서는 이미지-텍스트 쌍을 활용하여 전역 정렬(global alignment)을 수행하고, 두 번째 단계에서는 지역 정렬(region-level alignment) 및 미세 대비 신호(fine-grained contrastive signals)를 포함하여 모델의 성능을 재조정합니다. 텍스트 인코더는 최대 입력 길이를 64에서 196 토큰으로 확장하여 긴 설명을 수용할 수 있으며, 비전 측면에서는 데이터 적응형 해상도(data-adaptive resolution) 전략을 채택하여 일관된 훈련과 추론을 가능하게 합니다.

- **Performance Highlights**: FG-CLIP 2는 29개의 데이터셋에서 진행된 광범위한 실험을 통해 기존 모델들을 능가하며, 영어와 중국어 모두에서 최첨단 결과를 달성했습니다. 또한, 중국 멀티모달 이해를 위한 새로운 벤치마크(benchmark)를 제시하였으며, 이를 통해 긴 캡션 검색 및 바운딩 박스 분류와 같은 과제를 더 철저히 평가할 수 있습니다. FG-CLIP 2는 다국어 환경에서의 비전-언어 조정 능력을 강화하여, 향후 연구와 실제 적용에 기여할 모델로 자리 잡을 것으로 기대됩니다.



### DreamMakeup: Face Makeup Customization using Latent Diffusion Models (https://arxiv.org/abs/2510.10918)
- **What's New**: 이번 논문에서는 GANs의 기술적 한계를 극복하기 위해 DreamMakeup이라는 새로운 훈련 없는 확산 기반 메이크업 커스터마이징 방법을 소개합니다. DreamMakeup은 얼굴 구조와 정체성을 보존하는 동시에 다양한 조건 입력(예: 참조 이미지, 특정 RGB 색상, 텍스트 설명)을 통해 메이크업을 광범위하게 개인화할 수 있는 장점을 제공합니다. 이는 기존 GAN과 최근의 확산 기반 프레임워크보다 커스터마이징 및 색상 일치 기능에서 현저히 향상된 성능을 보여줍니다.

- **Technical Details**: DreamMakeup은 초기 중단 DDIM 인버전을 활용하여 주어진 얼굴 이미지의 구조를 보존하면서 메이크업 스타일을 목표로 하는 픽셀 공간에서의 변형을 가능하게 합니다. 이 과정은 하모니와 일관성을 위해 고급 크로스 어텐션 제어와 보간 가이드 샘플링을 포함하여, 사용자가 원하는 메이크업 스타일을 충족하도록 조정됩니다. 이를 통해 사용자가 다양한 조건을 사용하여 메이크업 프로세스를 유도할 수 있습니다.

- **Performance Highlights**: DreamMakeup은 실세계 글로벌 AI 메이크업 서비스와 색상 메이크업 작업에서 경쟁 우위를 보이며, 메이크업 전송 작업에서도 최신 확산 및 GAN 기반 프레임워크를 초과하는 성과를 보였습니다. 이 모델은 Large Language Models (LLMs)와의 통합이 용이하여, 다양한 사용자 요구에 대한 응답성을 극대화할 수 있습니다. 또한, 컴퓨팅 비용이 저렴하여, 고성능 그래픽 카드로도 4초 미만의 지연 시간으로 색상 전환을 수행할 수 있습니다.



### Topological Alignment of Shared Vision-Language Embedding Spac (https://arxiv.org/abs/2510.10889)
Comments:
          24 pages, 5 figures, 19 tables

- **What's New**: 본 논문에서는 멀티링구얼 Contrastive Vision-Language Model (VLM)에서의 구조적 비일치를 해결하기 위해 ToMCLIP(Topological Alignment for Multilingual CLIP)라는 새로운 프레임워크를 제안합니다. 기존 모델들은 주로 개별 인스턴스 수준의 정렬(instance-level alignment)만을 고려했지만, ToMCLIP은 토포로지 분석을 통해 전반적인 구조적 정렬을 이룰 수 있도록 합니다. 이를 통해 영어와 다른 언어 간의 성능 격차를 줄이려는 노력이 포함되어 있습니다.

- **Technical Details**: ToMCLIP은 Persistent Homology를 활용하여 토포로지 정렬 손실(topological alignment loss)을 정의하고, 이를 통해 언어 간 공통 임베딩 공간에서 구조적 정렬을 강제합니다. 본 연구에서는 그래프 희소화(graph sparsification) 전략을 통해 필수 다이어그램(persistence diagram)을 근사하는 방법도 개발하였습니다. 이를 통해 MCLIP이 기존의 점 대 점 정렬(point-wise alignment)에서 벗어나 보다 전반적인 구조적 일관성을 유지할 수 있도록 하고 있습니다.

- **Performance Highlights**: 제안된 ToMCLIP 방식은 CIFAR-100 데이터셋에서 제로샷(zero-shot) 정확도를 개선하고, xFlickr&CO에서의 멀티링구얼 검색 성능을 더욱 향상시키는 결과를 보였습니다. 실험 결과는 다양한 시나리오에서 멀티링구얼 표현의 구조적 일관성이 개선되었음을 입증하였습니다. 또한, 이 방법은 VLM뿐만 아니라 일반적으로 표현 학습에 토포로지 정렬을 도입할 수 있는 가능성을 제시합니다.



### Transfer Learning with Distance Covariance for Random Forest: Error Bounds and an EHR Application (https://arxiv.org/abs/2510.10870)
- **What's New**: 이 연구에서는 중심 랜덤 포레스트(Centered Random Forest, CRF)를 사용하여 비모수 회귀(Nonparametric Regression)에서 전이 학습(Transfer Learning) 방법을 제안합니다. 기존 방법보다 구조화된 표 형식 데이터에서 더 뛰어난 성능을 보이는 랜덤 포레스트를 기반으로, 몇 가지 특징에서 소스와 타겟 회귀 함수가 다르다고 가정합니다. 이 접근법은 새로운 데이터 세트에서 더욱 효과적으로 활용될 수 있는 가능성을 보여줍니다.

- **Technical Details**: 제안된 방법은 먼저 소스 도메인에서 훈련된 CRF를 사용하여 타겟 도메인에서의 잔차(residuals)를 추출합니다. 이후, 이 잔차에 대해 특징 분할 확률이 특징과 잔차 간의 독립 샘플(sample)에서의 거리 공분산(Distance Covariance) 비례하여 결정되는 또 다른 CRF를 적합시킵니다. 이 수학적 모델의 평균 제곱 오차(mean square error) 경계를 이론적으로 유도하여 랜덤 포레스트의 전이 학습 이점을 설명합니다.

- **Performance Highlights**: 임상 데이터를 활용한 시뮬레이션 결과, 제안된 CRF 방법이 표준 랜덤 포레스트(Standard Random Forest, SRF) 방법에서도 유사한 성능을 보임을 확인하였습니다. 특히, 중소 병원에서의 ICU 환자 생존 예측에서 뚜렷한 성과를 나타내었습니다. 전이 학습을 통한 성능 향상 외에도, 일부 상황에서 거리 공분산 기반 가중치의 이점이 확인되어, 랜덤 포레스트 기법의 응용 가능성을 넓힐 수 있습니다.



### Quantifying Dataset Similarity to Guide Transfer Learning (https://arxiv.org/abs/2510.10866)
- **What's New**: 이번 연구에서는 전이 학습(transfer learning)을 위한 혁신적인 메트릭인 Cross-Learning Score (CLS)를 제안합니다. 이는 데이터셋 간의 유사성을 측정하여 전이 가능성을 정량적으로 안내하기 위해 개발되었습니다. 기존의 방법들이 레이블 정보와 예측 관계를 간과한 반면, CLS는 도메인 간의 양방향 일반화 성능을 통해 유사성을 평가합니다.

- **Technical Details**: CLS는 두 데이터셋 간의 유사성을 코사인 유사성(cosine similarity)에 기반하여 이론적으로 정당화합니다. 이 방법은 고차원 문제를 위한 비싼 분포 추정의 문제를 피하면서도 효율적이고 빠른 계산을 가능하게 합니다. 또한, CLS를 통해 소스 데이터셋을 세 가지 영역(긍정적, 애매한, 부정적 전이 영역)으로 분류하여 좀 더 정보에 기반한 결정을 지원합니다.

- **Performance Highlights**: CLS는 다양한 합성 및 실제 작업에서 전이가 성능을 향상시키거나 저하시키는지를 신뢰성 있게 예측할 수 있는 능력을 보여줍니다. 이 메트릭은 전이 학습에서 데이터 선택을 안내하는 원칙적인 도구를 제공하며, 특히 현대의 딥러닝 파이프라인에 적합하도록 인코더-헤드 아키텍처를 반영하여 확장되었습니다.



### Fast and the Furious: Hot Starts in Pursuit-Evasion Games (https://arxiv.org/abs/2510.10830)
Comments:
          Presented at AAMAS Workshop on Autonomous Robots and Multirobot Systems (ARMS)

- **What's New**: 본 논문에서는 추적-회피 게임에서 추적자의 초기 위치를 효과적으로 설정하는 새로운 접근법을 제시합니다. 게임 이론(control theory)과 그래프 신경망(Graph Neural Networks, GNN)을 통합하여, 적절한 위치를 찾기 위한 다목적 최적화(multi-objective optimization) 방법을 사용하여, Pareto-optimal 구성을 파악합니다. 이 접근법은 추적자가 전혀 사전 지식 없이 evader의 위치를 고려하는 대신, 그래프를 통해 전략적으로 효과적인 초기 구성, 즉 'hot starts'를 생성합니다.

- **Technical Details**: 이 방법은 Graph Convolutional Network (GCN)를 훈련시켜 얻은 Pareto-optimal 그래프를 기반으로 하며, 다문제 최적화의 관점에서 추적자의 구성 요소를 인코딩합니다. 그래프 특성 공간(Graph Feature Space, GFS)을 사용하여, capture potential, distance, heading angle 세 가지 기능을 통합합니다. 다중-에이전트 입자 환경(Multi-Agent Particle Environment, MPE)을 통해 시뮬레이션된 데이터를 활용하여 GCN을 훈련하고 패턴을 예측합니다.

- **Performance Highlights**: 경험적 분석을 통해, GCN이 생성한 'hot starts'는 무작위 구성보다 우월한 성과를 보여줍니다. 구체적으로, 다수의 추적자와 회피자를 고려할 때 이 방법은 회피자의 생존율을 신속하게 감소시키고, 추적자의 이동 거리를 줄이며, containment을 향상시키는 명백한 전략적 이점을 나타냅니다. 이러한 성과는 알고리즘의 효율성을 높이며, 총 추적 비용을 효과적으로 감소시키는 데 기여합니다.



### Is Implicit Knowledge Enough for LLMs? A RAG Approach for Tree-based Structures (https://arxiv.org/abs/2510.10806)
Comments:
          Waiting for Conference Response

- **What's New**: 본 논문에서는 'Retrieval-Augmented Generation (RAG)' 방식을 통해 구조화된 데이터(예: 코드 파일)에서 생성된 응답을 향상시키기 위한 새로운 하향식(bottom-up) 방법을 제안합니다. 이 방법은 계층적 구조(예: 트리)의 지식을 선형화(linearize)하여 각 계층에서 암묵적(implicit) 요약을 생성합니다. 이 접근 방식은 기존의 RAG 방법론보다 더 효율적이며, 68% 이상의 문서 수 감소로 응답 품질을 비슷하게 유지하는 것을 보여줍니다.

- **Technical Details**: 이 논문은 계층적 구조에서 암묵적 지식을 생성하기 위한 새로운 방법을 제안합니다. 제안된 방법은 리프 노드(leaf node)에서부터 시작하여 모든 리프 노드에 대한 '템플릿(template)' 지식을 획득한 후, 각 부모 노드를 순회하며 자식들로부터 받은 암묵적 지식을 바탕으로 상위 요약을 생성합니다. 이러한 선형화 과정은 벡터 데이터베이스에 저장될 정보 조각을 최적화하고 토큰 수를 제한하여 효율성을 높입니다.

- **Performance Highlights**: 우리의 실험은 GM의 비구조화된 코드 리포지토리를 사용하였으며, 제안된 방법이 전통적인 RAG 방법에 비해 응답 품질이 유사함에도 불구하고 저장된 데이터 양을 거의 4분의 1로 줄임을 보여줍니다. 이를 통해 복잡한 구조적 정보를 처리하는 데 있어 암묵적 지식이 충분하고 효율적일 수 있음을 제안합니다. 또한 이 연구는 RAG 프레임워크에서 지식 관리를 위한 효과적이고 확장 가능한 방법 개발의 필요성을 강조합니다.



### MSCloudCAM: Cross-Attention with Multi-Scale Context for Multispectral Cloud Segmentation (https://arxiv.org/abs/2510.10802)
Comments:
          7 pages, 2 Figures

- **What's New**: 본 논문에서는 환경 모니터링과 기후 연구를 위한 신뢰할 수 있는 분석을 방해하는 클라우드(구름) 문제를 해결하기 위해 개발된 MSCloudCAM 모델을 소개합니다. 이 모델은 멀티스펙트럴 및 멀티센서 클라우드 세분화를 위해 설계되었으며, 네트워크는 Swin Transformer와 ASPP, PSP 모듈을 통합하여 다중 스케일 컨텍스트 집합을 구현합니다. MSCloudCAM은 구체적으로 맑은 하늘, 얇은 구름, 두꺼운 구름, 구름 그림자 네 가지 범주로 분류하는 성능을 발휘합니다.

- **Technical Details**: MSCloudCAM은 Swin Transformer 인코더와 다중 문맥 모듈, 크로스 어텐션 융합 메커니즘을 통합하여 장거리 스펙트럼-공간 의존성을 포착하고 세분화된 로컬 구조를 유지합니다. 입력으로는 Sentinel-2와 Landsat-8의 멀티스펙트럴 이미지를 사용하며, 각 이미지는 13 또는 11개의 스펙트럴 채널로 구성됩니다. 이 모델의 디코더는 보조 감독 기능을 통해 최종 밀집 예측을 생성합니다.

- **Performance Highlights**: CloudSEN12 및 L8Biome 데이터셋에 대한 포괄적 실험 결과 MSCloudCAM은 최첨단 세분화 정확도를 달성하며, 주요 기준 아키텍처를 초과하는 성과를 보였습니다. 또한 이 모델은 경쟁력 있는 파라미터 효율성과 FLOPs를 유지하여, 대규모 지구 관측 작업 및 실제 응용 프로그램에 적합함을 입증하였습니다. 이러한 결과는 MSCloudCAM의 효과성과 실용성을 강조하며, 다양한 센서와 스펙트럼 도메인에서의 정확한 픽셀 단위 분류를 가능하게 합니다.



### ParsVoice: A Large-Scale Multi-Speaker Persian Speech Corpus for Text-to-Speech Synthesis (https://arxiv.org/abs/2510.10774)
- **What's New**: Persian Language는 1억 명이 넘는 사람들이 사용하지만, 고품질 음성 데이터셋이 부족하여 TTS(텍스트-음성 합성) 기술의 발전에 큰 제약이 있어 왔습니다. 이에 대응하기 위해, ParsVoice라는 대규모 페르시아어 음성 코퍼스를 소개하였습니다. 이 데이터는 자동화 파이프라인을 통해 2,000개의 오디오북에서 변환된 데이터로 3,526시간의 청정 음성을 포함하며, 이는 TTS용으로 최적화된 1,804시간의 데이터를 보여줍니다.

- **Technical Details**: 이 연구는 ParsVoice라는 새로운 페르시아어 음성 코퍼스를 개발하기 위해 여러 기술적 접근 방식을 결합했습니다. BERT 기반의 문장 완성 감지기와 오디오-텍스트 정렬을 위한 경계 최적화 기법을 사용하여, 오디오북 데이터를 TTS에 적합한 형태로 변환하는 자동화된 파이프라인을 구축하였습니다. 이 파이프라인은 470명 이상의 화자가 포함된 대규모 음성 데이터를 생성합니다.

- **Performance Highlights**: ParsVoice는 고품질 페르시아어 음성 데이터셋으로서, 다수의 화자와 양질의 음성을 제공하여 영어 코퍼스 대비 동등한 수준의 다양성을 자랑합니다. 연구 결과, 이 코퍼스는 페르시아어 음성 언어 처리 기술 발전에 기여할 것으로 기대되며, 저자원 언어들이 이 데이터를 활용할 수 있는 모범 사례를 제시합니다. 공개된 데이터셋은 페르시아어 기술 발전을 가속화하는 데 중요한 역할을 할 것입니다.



### How Patterns Dictate Learnability in Sequential Data (https://arxiv.org/abs/2510.10744)
Comments:
          NeurIPS 2025, 36 pages, 4 figures

- **What's New**: 본 논문에서는 시계열과 자연 언어와 같은 연속 데이터를 위한 새로운 접근 방식을 제안합니다. 기존의 autoregressive 모델들이 데이터 내의 패턴을 사람의 전문성에 의존하다 보니, 그 판별에 어려움을 겪는 점을 지적하며, 이는 모델의 성능 저하로 이어질 수 있다고 설명합니다. 이를 해결하기 위해, 서로 다른 시간 지점에 있는 데이터 간의 mutual information을 사용하여 회귀 순서 추정 및 특성 선택에 도움을 주는 'evolving pattern(EvoRate)' 지표를 제안합니다.

- **Technical Details**: 제안된 방법은 predictive information 개념에 기반하고 있으며, 이는 과거 데이터(X_{past})와 미래 데이터(X_{future}) 간의 mutual information으로 정의됩니다. 이 정보를 사용하여 관찰 창이 커질수록 얻을 수 있는 예측적 정보의 양을 정량화하며, 이러한 정량화는 시계열 모델의 학습 가능성을 제약하는 본질적인 정보 한계를 드러냅니다. 실험을 통해 이 프레임워크가 데이터 세트의 내재된 복잡성을 정량화하고 모델 적합성을 평가할 수 있음을 입증하였습니다.

- **Performance Highlights**: 실험 결과, 이 프레임워크는 모델의 성능을 비교하는 데 유용하며, 데이터 자체에 의해 학습이 제한되는 경우를 식별할 수 있는 정량적 방법을 제공합니다. 특히, 정보 이론적 관점에서 예측 정보의 양은 데이터의 패턴 강도를 반영하며, 이는 모델 선택과 개선 가능성을 탐색하는 데 중요한 역할을 합니다. 연구 결과는 GitHub에서도 확인할 수 있으며, 정책적 데이터 활용과 알고리즘 개선 방향에 대한 통찰을 제시합니다.



### Seeing My Future: Predicting Situated Interaction Behavior in Virtual Reality (https://arxiv.org/abs/2510.10742)
Comments:
          Project Page: this https URL

- **What's New**: 이 연구에서는 사용자의 상호작용을 예측하여 반응하는 능동적 AR/VR 시스템을 만드는 데 기여하는 새로운 계층적 프레임워크를 제안합니다. 이 프레임워크는 인간의 의도를 이해하고, 세부적인 행동을 예측하는 데 초점이 맞춰져 있습니다. 특히, Dynamic Graph Convolutional Network (GCN)를 활용하여 인간과 환경 간의 상호작용을 보다 효과적으로 캡처합니다.

- **Technical Details**: 제안된 방법론은 역사적 관측값을 기반으로 사용자가 어디로 이동할지(trajectory), 어디를 바라볼지(gaze), 어떤 객체와 상호작용할지를 예측합니다. 이 구조는 사용자의 시선, 머리 자세, 손/신체 궤적과 같은 최신 AR/VR 장치에서 수집된 데이터를 활용합니다. 또한, 환경의 객체 상호작용은 시뮬레이션에서 직접 로그를 작성하거나 깊이 센서 및 IoT 기술을 통해 추론될 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 프레임워크가 실제 VR 환경에서 사용자의 시선 패턴과 미래 행동을 성공적으로 예측할 수 있음을 보여주었습니다. 이 방법은 모든 메트릭에서 뛰어난 성과를 달성하며, 사용자의 행동을 예측하고 가상의 환경을 적절히 조정할 수 있는 실용적인 응용 프로그램 개발에 기여할 가능성을 제시합니다.



### Controllable Generative Trajectory Prediction via Weak Preference Alignmen (https://arxiv.org/abs/2510.10731)
- **What's New**: 이 논문은 자동 주행 차량에서 주변 에이전트의 경로를 예측하기 위해 PrefCVAE라는 새로운 방법을 제안합니다. 이 방법은 약하게 레이블이 붙은 선호 쌍을 사용하여 잠재 변수를 의미 있는 속성으로 채우는 것을 목표로 합니다. 평균 속도를 예시 속성으로 사용하여, PrefCVAE가 예측의 다양성을 조절할 수 있게 하는 방법을 보여줍니다.

- **Technical Details**: PrefCVAE는 기존 CVAE 프레임워크를 강화한 것으로, 부분적으로 레이블이 붙은 정렬 신호 데이터를 통해 의미론적으로 제어 가능한 잠재 공간을 학습합니다. 이러한 학습 과정은 의미론적인 잠재 코드를 인코딩하고, 이를 통해 예측 가능한 방식으로 경로 생성을 조절할 수 있도록 합니다. 효과적인 학습을 위해 미니배치의 데이터 샘플을 참고하여 과거 경로와 맥락 정보를 기반으로 미래 경로를 예측합니다.

- **Performance Highlights**: 실험에서는 대규모 데이터셋인 nuScenes를 사용하여 PrefCVAE가 AgentFormer를 통해 경로를 예측하는 데 있어 의미론적으로 단조로운 메트릭을 생성할 수 있음을 보여 줍니다. 또한, 근사 후방 인코더가 경로를 그라운드 트루스 잠재 값에 더 높은 가능도로 매핑한다는 결과를 보여 주며, 이는 PrefCVAE가 조절 가능한 생성 경로 예측을 위한 프레임워크로서의 실행 가능성을 시사합니다.



### Deep Signature and Neural RDE Methods for Path-Dependent Portfolio Optimization (https://arxiv.org/abs/2510.10728)
Comments:
          Accepted for presentation at the ACM International Conference on AI in Finance (ICAIF 2025), QuantAI Workshop, Singapore. 9 pages. Code available at: this https URL

- **What's New**: 본 연구에서는 고차원 경로 의존적 평가와 제어를 위해 학습된 심층 BSDE 및 2BSDE 솔버를 제시합니다. 트렁케이트된 로그 시그니처와 Neural Rough Differential Equation (Neural RDE)을 결합하여, 왼쪽 꼬리 위험을 강조하는 CVaR 목표를 사용합니다. 이 접근 방식은 아시안 및 배리어 옵션 가격 책정과 포트폴리오 제어 작업에서 정확도와 훈련 안정성을 향상시킵니다.

- **Technical Details**: 제안된 방법은 트렁케이트된 로그 시그니처 인코더와 Neural RDE 백본을结合하여 경로 의존적 정보를 효율적으로 처리합니다. 시그니처는 경로를 계층적으로 요약하고, Neural CDE/RDE 모델은 연속 시간에서 숨겨진 상태를 발전시키면서 안정적인 긴 시간 구배를 제공합니다. 또한 고차원 경로 의존적 가격 책정과 변동성 포트폴리오 제어를 다루며, 다양한 평가 기준(절대/상대 오차, 런타임 등)을 설정합니다.

- **Performance Highlights**: 200 차원에서 CVaR(0.99) = 9.8 %를 달성하며, 강력한 기준선과 비교했을 때 12.0-13.1 %의 개선을 보였습니다. HJB 잔여와 Z 및 Gamma에 대한 작은 RMSE도 확인되었습니다. 최적화된 시그니처 깊이와 RDE 벡터 필드 너비, 다단계 깊이, Γ 추정에 대한 실험 결과, 경로 의존적 금융 모델의 해결 가능성을 높였습니다.



### Missing Data Multiple Imputation for Tabular Q-Learning in Online RL (https://arxiv.org/abs/2510.10709)
Comments:
          Working paper

- **What's New**: 온라인 강화 학습(online Reinforcement Learning, RL)에서의 누락 데이터 문제는 전통적인 표 형식 데이터나 오프라인 정책 학습과 비교할 때 보다 복잡한 도전 과제가 됩니다. 특히, 각 시점에서 데이터를 보완(impute)하고 행동(act)해야 하는 필요성으로 인해, 안정적인 보완 모델이 생성되기 위해 충분한 데이터가 존재할 때까지 보완이 연기될 수 없습니다. 본 논문에서는 완전 온라인 보완 앙상블(full online imputation ensembles)을 제안하며, 이는 누락된 데이터 상태에서 불확실성을 캡처하고 컴퓨팅 효율성을 높임을 목표로 합니다.

- **Technical Details**: 본 연구에서는 다양한 접근 방식을 적용하여 여러 보완 경로(multiple imputation pathways)를 학습 및 행동 선택(action selection) 과정에 통합하였습니다. 우리는 누락된 상태 공간 데이터를 사용한 온라인 RL을 위해 여러 보완 앙상블을 탐구하며, 확률적 보완 임베딩이 단순한 기초 모델들보다 더 나은 성능을 발휘할 수 있는 가능성을 제시합니다. 또한, 이러한 앙상블 방식은 반복적인 누락 정보로 인한 경로 의존성(path dependency) 문제를 피할 수 있습니다.

- **Performance Highlights**: 그리드 월드(experiment)에서 진행된 실험을 통해, 다수의 보완 경로가 간단한 기초 모델과 단일 보완 모델보다 더 나은 성능을 발휘할 수 있음을 초기 증거를 통해 확인하였습니다. 또한, 누락 데이터를 상태 옵션으로 인코딩하는 방식과의 비교를 통해 누락 비율(missingness rate)에 따라 U자 형태의 성능 곡선이 나타났음을 확인하였습니다. 이는 제안된 방법이 누락된 정보로 인해 상태 공간 차원을 효과적으로 조절함을 시사합니다.



### Learning-Augmented Streaming Algorithms for Correlation Clustering (https://arxiv.org/abs/2510.10705)
Comments:
          NeurIPS 2025

- **What's New**: 이번 논문에서는 Correlation Clustering을 위한 스트리밍 알고리즘을 연구하였습니다. 주어진 그래프를 양의 또는 음의 간선 스트림으로 처리하며, 목표는 불일치를 최소화하는 분할을 찾는 것입니다. 특히, 우리는 완전 그래프와 일반 그래프에 대해 최초의 학습 보강 학습 알고리즘을 제안하여 기존의 알고리즘보다 공간-근사 무역비를 개선하였습니다.

- **Technical Details**: 본문에서는 Cambus et al. (SODA'24) 및 Ahn et al. (ICML'15)의 연구를 바탕으로 개발한 알고리즘을 소개합니다. 이 알고리즘은 예측 모델(predictor)이 제공하는 쌍 간 거리(pairwise distances) 예측을 활용합니다. 완전 그래프의 경우, 좋은 예측 품질이 제공되면 $3$보다 더 나은 근사치를 달성하고, $	ilde{O}(n)$의 총 공간을 사용합니다. 일반 그래프에서는 $O(	ext{log } |E^-|)$ 근사치를 달성하며, 또한 동일한 공간 효율성을 갖춘 비학습(non-learning) 알고리즘을 개선하였습니다.

- **Performance Highlights**: 합성 데이터셋(synthetic datasets) 및 실제 데이터셋(real-world datasets)에서의 실험 결과는 우리가 제안한 알고리즘이 비학습 알고리즘보다 우수함을 보여주었습니다. 이러한 결과는 학습 보강 알고리즘의 효과성과 실제 적용 가능성을 강조합니다. 전반적으로, 본 연구는 Correlation Clustering 문제에 대한 새로운 접근법을 제시하며, 알고리즘의 성능을 크게 향상시켰습니다.



### Mean-square and linear convergence of a stochastic proximal point algorithm in metric spaces of nonpositive curvatur (https://arxiv.org/abs/2510.10697)
Comments:
          24 pages

- **What's New**: 이번 논문에서는 비선형 Hadamard 공간에서 확률적으로 변형된 근접 점 알고리즘(stochastic proximal point algorithm)의 변형을 정의하고, 강한 단조성(strong monotonicity) 가정 하에 수렴성을 검증합니다. 특히, 이전의 Hilbert 공간 연구 결과를 Hadamard 다양체(manifolds)로 이전하는 데 성공하였습니다. 이 접근법에서 수렴 속도(convergence rate)를 명시적으로 제공하는 것이 주요 특징입니다.

- **Technical Details**: 이 알고리즘은 상당한 이론적 작업을 기반으로 하여, 주어진 함수 f:E×X→(−∞,+∞]을 확률 공간 (E,ℰ,μ)에서 해결하는 방식으로 구성됩니다. 이 과정에서 근접 맵(proximal map) 및 아이디에 의존하는 확률 변수의 반복(iteration)을 활용하며, 이는 각종 공간과 함수에 대한 일반적인 방법론으로 연구됩니다. 또한, 알고리즘은 CAT(0) 공간과 같은 비양수 곡률의 미터릭 공간에서 적용 가능하다는 점이 강조됩니다.

- **Performance Highlights**: 논문에서는 수렴 속도가 명확히 정의되어, 반복(iteration) 과정의 (유일한) 해에 대한 수렴률을 기대값(expectation) 및 거의 확실히(almost surely) 합성할 수 있음을 보여줍니다. 이 수렴 속도는 주변 데이터에 독립적이며, Hilbert 공간에서의 맥락에서도 새로운 발견이 될 만큼 효과적입니다. 또한, 추가적인 두 번째 모멘트 조건에 기반한 비선형 비대칭 보장(linear nonasymptotic guarantees)도 제시되고 있습니다.



### High-Dimensional Learning Dynamics of Quantized Models with Straight-Through Estimator (https://arxiv.org/abs/2510.10693)
Comments:
          27 pages, 14 figures

- **What's New**: 이번 연구는 Quantized neural network training에서 quantization hyperparameters가 학습 동역학에 미치는 영향을 이론적으로 탐색합니다. 특히, straight-through estimator (STE)의 동적 속성이 고차원 한계에서 어떻게 변화하는지를 분석합니다. 이 연구는 quantization range와 bit width가 일반화 오차에 미치는 영향을 정량화하고, quantized DNNs의 학습 안정성을 밝혀내는 데 기여합니다.

- **Technical Details**: STE를 이용한 학습이 고차원 한계에서 확률적 미분 방정식(SDE)과 결정론적 미분 방정식(ODE)의 수렴을 보여주며, 두 단계의 궤적을 예측합니다. 여기서 ODE는 연장된 plateau와 일반화 오차의 급속한 감소를 나타냅니다. 특히, 하이퍼파라미터 선택이 학습 다이나믹스의 중요한 요소로 작용하며, 입력 양자화가 성능 저하에 미치는 영향을 분석합니다.

- **Performance Highlights**: 저자들은 quantization hyperparameters가 학습 안정성 및 일반화 성능에 미치는 영향을 체계적으로 수치화했습니다. 낮은 비트 폭에서 STE의 동적 변화가 비모노톤(non-monotonic)이 되어 수렴 속도가 느려지는 경향이 있음을 발견했습니다. 또한, 비양자화 모델에 비해 성능 저하를 정량적으로 분석하여, 양자화가 단순한 교란이 아닌 내재적 정규화 역할을 한다는 점을 강조합니다.



### Second-order Optimization under Heavy-Tailed Noise: Hessian Clipping and Sample Complexity Limits (https://arxiv.org/abs/2510.10690)
Comments:
          Accepted for publication at NeurIPS 2025

- **What's New**: 이 연구는 무거운 꼬리(heavy-tailed) 노이즈에서의 두 번째 차수 최적화(second-order optimization)에 대한 이론적 이해를 위한 첫걸음을 내딛고 있습니다. 기존의 두 번째 차수 방법들은 이러한 노이즈 상황에서 안정성이 떨어지고, 이론적 보장이 부족하기 때문에 실용적인 응용이 제한적이었습니다. 본 연구는 경량 노이즈 환경에서의 수렴을 가속화하는 두 번째 차수 최적화 방법을 새로운 알고리즘과 함께 제안합니다.

- **Technical Details**: 무거운 꼬리 노이즈 모델에서의 두 번째 차수 최적화를 위해, 데이터의 이질성과 비정상 확률적 환경을 고려하여 확률적 그래디언트와 해시안의 p-번째 모먼트를 고정된 값으로 설정하였습니다. 본 논문은 이러한 설정에서 두 번째 차수 방법의 샘플 복잡도의 하한을 확립하고, 고급 그래디언트 클리핑 기술을 사용하여 안정성을 보장하는 알고리즘을 개발하였습니다. 이러한 접근법은 무거운 꼬리 노이즈에 민감한 헤시안(Hessian) 추정을 개선하기 위한 새로운 계획을 제공합니다.

- **Performance Highlights**: 제안된 알고리즘은 두 번째 차수 최적화에서의 샘플 복잡도를 함수의 리프시츠( Lipschitz) 상수와 무거운 노이즈 성분으로 보장합니다. 실험적으로는 기존의 첫 번째 차수 방법보다 더 나은 성능을 보여주며, 높은 확률로 수렴 보장을 제공합니다. 이 연구는 무거운 꼬리 노이즈 조건에서 두 번째 차수 알고리즘 설계의 견고한 기초를 제공하며, 현대 기계 학습 응용에 실질적으로 관련성이 높습니다.



### Interactive Atmospheric Composition Emulation for Next-Generation Earth System Models (https://arxiv.org/abs/2510.10654)
- **What's New**: 이번 연구에서는 NASA GISS-ModelE3을 기반으로, 기후 변화 모델링의 정확도를 높이기 위해 머신 러닝(ML) 기법을 사용하여 Smart NINT를 개발했습니다. 기존의 비상호작용 추적기(NINT)와는 달리, Smart NINT는 실시간 에미션을 모사하도록 설계되어 대기 내 물질 농도를 동적으로 계산합니다. 이를 통해 표면 에미션 및 기상 데이터를 입력으로 활용하여 전통적인 물리적 매개변수화를 회피하고, 보다 효율적인 기후 예측이 가능해졌습니다.

- **Technical Details**: Smart NINT는 모델 아키텍처에서 공간적(spatial) 및 시간적(temporal) 종속성을 효과적으로 캡처하기 위해 적절한 유도 편향을 채택한 spatiotemporal 아키텍처를 활용합니다. 연구에서는 20개의 수직 압력 수준을 사용하여 Black Carbon BCB의 대기 농도를 예측하며, 3D 모델을 구현합니다. 이 모델은 ConvLSTM 구조를 채택하여 시간 및 공간 데이터를 입력으로 사용하며, 기후 강제력에 대한 상호작용적 에미션의 영향을 시뮬레이션합니다.

- **Performance Highlights**: 모델 성능은 R² 값이 0.92에 달하고, Pearson 상관계수는 0.96을 기록하는 등 뛰어난 결과를 보여줍니다. 특히 첫 번째 압력 수준에서 이러한 성능이 유지되며, 15번째 압력 수준까지도 양호한 성능을 보입니다. 연구는 완전히 다른 기간의 데이터에 대해서도 테스트되었으며, 이로써 장기 기후 예측이 요구되는 응용 프로그램에 대한 신뢰성을 입증했습니다.



### Automatic Piecewise Linear Regression for Predicting Student Learning Satisfaction (https://arxiv.org/abs/2510.10639)
- **What's New**: 이 연구는 COVID-19 팬데믹 기간 동안 학생들의 학습 만족도에 영향을 미치는 다양한 요인들을 탐구하였으며, 최근의 해석 가능한 기계 학습 모델인 자동 구간 선형 회귀(Automatic Piecewise Linear Regression, APLR)가 학습 만족도를 예측하는 데 가장 적합한 모델임을 입증하였습니다. 교사들은 APLR의 전 세계적 및 개인 수준 해석을 통해 학생 프로필에 따라 맞춤형 교육을 제공할 수 있는 기회를 얻습니다. 이 연구는 APLR이 기존의 배깅(bagging) 및 부스팅 트리(boosted trees), 심지어 트랜스포머 기반 딥러닝 모델보다 뛰어난 성능을 보임을 강조합니다.

- **Technical Details**: 이 논문에서는 302명의 성균관대학교 학생을 대상으로 한 단면적 연구에서 COVID-19 팬데믹 동안 온라인 학습 경험을 바탕으로 학습 만족도에 영향을 미치는 인자들을 분석하였습니다. 자동 구간 선형 회귀(APLR)는 시각적으로 모델의 결정을 설명하고, 복잡한 데이터에서의 예측을 가능하게 하는 해석 가능한 기계 학습 방법입니다. 본 연구는 APLR의 성능과 해석력을 통해 전 세계 집단 및 개별 학생들에게 영향을 미치는 요인들을 발견하였습니다.

- **Performance Highlights**: APLR은 5개 지표 중 4개에서 대표적인 배깅 및 부스팅 트리, 해석 가능한 가법 모델(interpretative additive model), 트랜스포머 기반 심층 학습 모델보다 뛰어난 예측 성능을 보였습니다. 이 연구는 학생들의 시간 관리, 집중력, 동료에 대한 유용성 인식, 오프라인 수업 참여가 학습 만족도에 가장 큰 긍정적 영향을 미친다고 밝혔습니다. 흥미롭게도, 창의적 활동이 학습 만족도에 긍정적인 영향을 미치지 않았다는 결과도 도출되었습니다.



### GraphTARIF: Linear Graph Transformer with Augmented Rank and Improved Focus (https://arxiv.org/abs/2510.10631)
- **What's New**: 이 논문에서는 기존의 선형 Attention 메커니즘이 가지는 표현력 저하 문제를 다루며, 이를 개선하기 위한 하이브리드 프레임워크인 GraphTARIF를 제안합니다. GraphTARIF는 값 행렬에 게이트가 있는 로컬 그래프 네트워크(branch)를 결합하여 Attention 맵의 랭크를 증가시키고, 학습 가능한 로그-파워 함수로 Attention 점수를 조절하여 엔트로피를 줄입니다. 이로 인해 모델의 분류 능력이 향상되어, 그래프 기반의 다양한 웹 관련 데이터셋에서 경쟁력을 가지게 됩니다.

- **Technical Details**: GraphTARIF 모델은 선형 그래프 변환기(linear Graph Transformer)로, 로컬 강화 모듈(local enhancement module)과 학습 가능한 로그-파워 함수를 통합하여 Attention 랭크를 높이고 엔트로피를 낮추는 방향으로 설계되었습니다. 이 방법은 특히 노드 레벨 작업에서의 표현 가능성을 개선하면서도 선형 Attention의 확장성(scalability)을 유지합니다. 논문은 이론적 분석을 통해 이러한 개선이 노드 간의 구별을 어떻게 향상시키는지를 명시합니다.

- **Performance Highlights**: GraphTARIF는 동질적(homophilic) 및 이종적(heterophilic) 그래프 학습 작업에서 우수한 성능을 보여주었습니다. 다양한 웹 관련 데이터셋에 대한 실험을 통해 기존 기준선과 비교하여 일관된 성능 향상을 입증했습니다. 특히, 선형 Self-Attention의 한계를 극복하면서도 효율성과 정확성을 동시에 달성하는 데 초점을 맞추었습니다.



### DCP: Addressing Input Dynamism In Long-Context Training via Dynamic Context Parallelism (https://arxiv.org/abs/2510.10620)
Comments:
          16 pages, 22 figures

- **What's New**: 이 논문에서는 동적 컨텍스트 병렬화(DCP) 프레임워크를 제안하여 긴 컨텍스트 훈련을 지원하는 새로운 방법을 소개합니다. 기존의 정적 병렬화 방법은 훈련 데이터의 동적 특성을 간과하여 통신 오버헤드(communication overhead) 및 불균형한 계산(load imbalance) 문제를 초래합니다. DCP는 데이터와 계산을 미세 블록 단위로 분할함으로써 다양한 시퀀스 특성에 적응할 수 있습니다.

- **Technical Details**: DCP는 주어진 훈련 반복마다 서로 다른 병렬화 구성을 생성하여 효율적인 통신 및 메모리와 계산 균형을 이루도록 최적화합니다. 제안된 모델은 각 Attention 입력과 출력을 세분화된 데이터 블록으로 나누어 Attention 패턴을 캡처할 수 있습니다. 이 블록은 유연하게 장비에 할당될 수 있어 맞춤형 병렬화 솔루션을 제공합니다.

- **Performance Highlights**: 마이크로 벤치마크 결과에 따르면, DCP는 causal masks에서 1.19배에서 2.45배, sparse attention 패턴에서는 2.15배에서 3.77배의 속도 향상을 보여줍니다. 또한 엔드 투 엔드 훈련에서는 causal masks에서 0.94배에서 1.16배, sparse masks에서는 1.00배에서 1.46배의 속도 향상이 관찰되었습니다.



### Deep semi-supervised approach based on consistency regularization and similarity learning for weeds classification (https://arxiv.org/abs/2510.10573)
Comments:
          Submitted to EURASIP Journal on Image and Video Processing

- **What's New**: 이 논문에서는 잡초 종류 분류를 위한 새로운 딥 세미-슈퍼바이즈드 방법을 제안합니다. 이 방법은 consistency regularization과 similarity learning을 결합하여, 레이블링된 데이터가 부족한 상황에서도 효과적이고 강력한 잡초 인식을 가능케 합니다. 또한, ConvNeXt 인코더를 기반으로 한 오토 인코더 아키텍처를 통해 레이블이 없는 데이터의 유용성을 극대화합니다.

- **Technical Details**: 제안된 방법은 DeepWeeds 데이터 세트에서 실험을 통해 검증되었습니다. 이는 자동화된 시스템의 자율적인 결과를 도출하는 데 중요한 이점을 제공합니다. 고전적인 머신 러닝 기술과 달리, 딥 러닝 모델은 데이터에서 자동으로 중요한 특징을 추출하는 강력한 능력을 가지고 있어, 데이터 라벨링의 시간 소모적인 과정에서 해방될 수 있습니다.

- **Performance Highlights**: 실험 결과, 제안된 방법은 최신의 슈퍼바이즈드 딥 러닝 모델들과 비교했을 때 우수한 성능을 보여주었습니다. 특히, 잡초 식별 문제에서 제안된 방법이 효과적임을 입증하였으며, 레이블 데이터가 제한된 조건에서도 높은 분류 정확성을 달성했습니다. 이 연구는 농업 분야에서 신뢰성 있는 잡초 관리 시스템 구축에 기여할 것으로 기대됩니다.



### MCE: Towards a General Framework for Handling Missing Modalities under Imbalanced Missing Rates (https://arxiv.org/abs/2510.10534)
Comments:
          This is the accepted version of an article that has been published in \textbf{Pattern Recognition}. The final published version will be available soon

- **What's New**: 이번 논문에서는 다중 모달 학습(multi-modal learning)의 새로운 접근 방식인 Modality Capability Enhancement (MCE)를 제안합니다. MCE는 학습 능력 향상(Learning Capability Enhancement, LCE)과 표현 능력 향상(Representation Capability Enhancement, RCE)이라는 두 가지 상호 보완적인 요소를 포함하여 비대칭적인 결측 모달리티 문제를 해결하고자 합니다. 기존의 연구는 데이터셋 수준에서의 균형에 집중했던 반면, MCE는 샘플 수준의 변동성을 고려하여 모달리티의 유틸리티를 최적화합니다.

- **Technical Details**: MCE 프레임워크는 두 가지 주요 구성 요소로 구성됩니다. 첫째, LCE는 모달리티의 학습 진전을 동적으로 균형 있게 조정하며, 글로벌 모달리티 가용성과 현재 모달리티의 학습 상태를 평가하여 차별화된 인센티브를 부여합니다. 둘째, RCE는 서브셋 예측(subset prediction)과 교차 모달 완성(cross-modal completion) 작업을 통해 특성 품질을 향상시킵니다.

- **Performance Highlights**: 다양한 멀티 모달 벤치마크에서 실시된 포괄적인 평가 결과, MCE는 다양한 결측 구성 환경에서 최첨단 기술(state-of-the-art)보다 일관되게 성능이 뛰어난 것으로 나타났습니다. 본 연구는 다중 모달 학습이 직면한 도전과제를 직시하고, 이를 해결하기 위한 원리적이고 실용적인 솔루션을 제시하였습니다.



### Merlin's Whisper: Enabling Efficient Reasoning in LLMs via Black-box Adversarial Prompting (https://arxiv.org/abs/2510.10528)
- **What's New**: 이번 연구는 Large Reasoning Models (LRMs)의 과도한 사고(overthinking) 문제를 해결하기 위한 새로운 접근 방식을 제안합니다. 연구팀은 블랙박스(black-box) 환경에서 개방형(open-source)과 폐쇄형(closed-source) 모델을 모두 고려하여, 높은 정확도를 유지하면서 간결한 응답을 이끌어낼 수 있는 방법을 탐구합니다. 이를 통해 AdvPrompt라는 반복적 정제(iterative refinement) 프레임워크를 도입하고, 다양한 관점에서의 적대적 프롬프트(adversarial prompts)를 생성하여 LRMs의 응답 길이를 줄이는 데 성공했습니다.

- **Technical Details**: AdvPrompt는 우선 여러 후보 프롬프트를 합성한 후 전용 개발 세트에서 평가하여 다음 반복을 위한 상위 k개의 성과를 선정합니다. 이 프레임워크는 모델 간의 상호작용을 보다 인간 친화적인 방식으로 전환하려고 하며, 수차례의 반복 과정을 통해 얻어진 최적의 프롬프트를 선택하여 배포합니다. 실험은 여러 벤치마크 데이터셋에서 수행되었으며, 결과적으로 AvPrompt는 응답 성능을 유지하면서도 평균 토큰 사용량을 35%에서 47%까지 줄이는 성과를 이루었습니다.

- **Performance Highlights**: AdvPrompt의 성능은 여러 모델에서 일관되게 나타났습니다. Qwen3 모델 시리즈의 경우, GSM8K 질문에 대한 평균 응답 길이를 3배 줄였고, 다양한 LRMs에 대해 평균적으로 19%에서 41%의 토큰 사용량 감소를 이끌어냈습니다. 특히, 상업적 API인 Claude-3.7과 Gemini-2.5에서도 MATH-500 데이터셋에서 각각 35% 및 47%의 토큰 사용량 감소를 달성하며 그 효과성이 입증되었습니다.



### Integrating Large Language Models and Reinforcement Learning for Sentiment-Driven Quantitative Trading (https://arxiv.org/abs/2510.10526)
- **What's New**: 이 연구에서는 대형 언어 모델 FinGPT를 활용한 감정 기반의 정량적 거래 시스템을 개발하고, 강화 학습 알고리즘인 Twin Delayed Deep Deterministic Policy Gradient (TD3)를 사용하여 신호 통합 방법을 탐구합니다. 감정 신호와 기술 신호를 통합하는 전략의 성능을 전통적인 규칙 기반 접근법과 강화 학습 프레임워크를 통해 비교했습니다. 결과적으로 FinGPT가 생성한 감정 신호가 전통적인 기술 지표와 결합될 때 가치를 제공하며, 강화 학습 알고리즘이 동적 거래 환경에서 이종 신호를 효과적으로 통합하는 유망한 접근법임을 보여주었습니다.

- **Technical Details**: 이 논문에서는 감정 정보와 전통적인 기술 지표를 결합하기 위한 혁신적인 접근 방식을 탐구하며, 강화 학습 기반의 프레임워크를 제안하여 두 신호 유형을 동적으로 결합합니다. 연구는 감정 신호가 주식 수익률에 대한 예측력을 가지는지, 그리고 감정 신호가 기술 지표 기반의 전통적인 거래 전략을 향상시킬 수 있는지를 다룹니다. 또한 강화 학습이 감정 신호와 기술 신호를 통합하여 더 나은 성과를 내는 거래 전략을 구축하는 효과적인 접근법인지 평가합니다.

- **Performance Highlights**: 강화 학습 기반의 통합 방법은 전통적인 규칙 기반 전략에 비해 상대적으로 향상된 포트폴리오 성과를 도출하는 것으로 나타났습니다. 데이터 처리에서 2018년부터 2025년까지의 S&P 500 주식의 뉴스 기사 및 가격 데이터를 사용하여, 감정 신호가 거래 성과에 미치는 예측력을 평가하고 있습니다. 이 연구의 결과는 강화 학습을 통한 효율적인 이종 신호 통합이 동적이고 변동성이 큰 시장 환경에서 성공적인 거래 결정을 내리는 데 중요한 역할을 한다는 것을 시사합니다.



### Population-Coded Spiking Neural Networks for High-Dimensional Robotic Contro (https://arxiv.org/abs/2510.10516)
- **What's New**: 이 논문은 Deep Reinforcement Learning (DRL)과 population-coded Spiking Neural Networks (SNNs)를 결합한 새로운 프레임워크를 제안하여 로봇 제어의 에너지 효율성과 성능 문제를 해결하고자 합니다. 이 접근 방식은 SNNs의 이벤트 기반, 비동기 계산 방식과 DRL의 견고한 정책 최적화 능력을 결합하여, 에너지 효율성과 제어 성능 간의 균형을 이룹니다.

- **Technical Details**: 제안된 프레임워크의 핵심은 Population-coded Spiking Actor Network (PopSAN)으로, 고차원 관측치를 신경 집단 활동으로 인코딩하고, 기울기 기반 업데이트를 통해 최적 정책 학습을 가능하게 합니다. 이 프레임워크는 Isaac Gym 플랫폼에서 PixMC 기준을 사용하여 다이내믹한 로봇 조작 작업에 대해 평가되었습니다.

- **Performance Highlights**: 실험 결과, Franka 로봇 팔을 사용한 경우, 기존의 Artificial Neural Networks (ANNs)와 비교하여 최대 96.10%의 에너지 절약을 달성했으며, 제어 성능을 유지하였습니다. 학습된 SNN 정책은 지시된 궤적에서 최소한의 편차로 손가락 위치 추적을 유지하였고, 피킹 및 배치 과정에서 안정적인 목표 높이 유지를 보여주었습니다.



### The Hidden DNA of LLM-Generated JavaScript: Structural Patterns Enable High-Accuracy Authorship Attribution (https://arxiv.org/abs/2510.10493)
- **What's New**: 이번 논문에서는 Large Language Models (LLMs)가 생성한 JavaScript 코드가 특정 모델에 의해 생성되었음을 밝힐 수 있는지에 대한 대규모 연구를 소개합니다. 이러한 연구는 저작권 속성이 코드의 취약점을 탐지하고 악성 콘텐츠를 플래그하며 책임을 보장하는 데 중요한 역할을 합니다. 연구에서는 20개의 LLM으로부터 생성된 50,000개의 Node.js 프로그램으로 구성된 LLM-NodeJS 데이터셋을 출시하며, 이 데이터셋은 250,000개의 고유한 JavaScript 샘플과 여러 추가 표현을 포함합니다.

- **Technical Details**: 논문에서는 JavaScript의 저작권 속성 분석을 위해 LLM-NodeJS 데이터셋을 통해 전통적인 기계 학습 분류기와 미세 조정된 Transformer 인코더의 성능을 비교했습니다. 코드의 스타일적 패턴을 관찰하기 위해, CodeT5-JSA라는 커스텀 아키텍처를 설계하였으며, 이 모델은 95.8%의 정확도로 5개 클래스 저작권 속성을 식별할 수 있었습니다. 데이터셋은 문법적으로 올바른 Node.js 프로그램을 포함하며, 표기 변형에 대한 저작권 속성이 여전히 효과적인 것으로 나타났습니다.

- **Performance Highlights**: JavaScript 저작권 속성 문제에서, 개발된 분류기는 원본 코드뿐만 아니라, 압축(minified), 망가진(mangled), 그리고 복호화(deobfuscated)된 코드에서도 높은 정확도를 기록했습니다. 이는 데이터 흐름 및 구조와 같은 깊은 스타일 정규성을 포착한다는 점에서 중요한 결과입니다. 연구의 결과는 저작권 연구가 AI vs 인간의 이분법을 넘어 확장될 필요성을 강조하며, 코드 저작권 분석을 위한 새로운 가능성을 여는 중요한 기초자료로 기능할 것입니다.



### DAGLFNet:Deep Attention-Guided Global-Local Feature Fusion for Pseudo-Image Point Cloud Segmentation (https://arxiv.org/abs/2510.10471)
- **What's New**: 이 논문에서는 LiDAR 기반 포인트 클라우드(3D point cloud)의 의미 분할(semantic segmentation)을 위한 새로운 프레임워크인 DAGLFNet을 제안합니다. 이 구조는 포인트 클라우드의 무질서한 데이터를 정형화된 2D 이미지 표현으로 변환하면서도 중요한 구별 가능한 특징을 포착할 수 있도록 설계되었습니다. 또한, 이 프레임워크는 글로벌-로컬 특징 융합, 다중 분기 특징 추출, 깊이 특징 유도 주의 메커니즘을 통합하여 정확성과 계산 효율성을 모두 강화합니다.

- **Technical Details**: DAGLFNet 프레임워크는 이미지 표현으로의 매핑 과정에서 겪는 경계 특징의 흐릿함 문제를 해결하기 위해, GL-FFE(글로벌-로컬 특징 융합 인코딩) 모듈과 MB-FE(다중 분기 특징 추출) 네트워크를 활용합니다. GL-FFE 모듈은 전역 및 지역의 기하학적 관계를 캡처하고, MB-FE 네트워크는 경계 특징의 표현력을 강화하여 수용장(receptive field)을 확장하는 데 중점을 둡니다. 또한, FFDFA(특징 융합을 통한 깊이 특징 유도 주의) 전략을 도입하여 특징 통합 단계에서 거리 정보를 반영하여 정확한 교차 채널(feature) 융합을 개선합니다.

- **Performance Highlights**: DAGLFNet은 SemanticKITTI 및 nuScenes의 검증 세트에서 각각 69.83% 및 78.65%의 평균 교차 합집합(mIoU) 점수를 달성하며 뛰어난 성능을 입증했습니다. 이 프레임워크는 임베디드 플랫폼에서도 성공적으로 배포될 수 있어 실시간 의미 분할(real-time semantic segmentation)이 가능하다는 잠재력을 보여줍니다. 이에 따라 실시간 LiDAR 기반 응용 프로그램에서의 활용 가능성이 크게 확대되었습니다.



### Rethinking LLM Evaluation: Can We Evaluate LLMs with 200x Less Data? (https://arxiv.org/abs/2510.10457)
Comments:
          18 pages, 5 figures

- **What's New**: 이 논문에서는 다양한 모델 능력의 포괄적인 평가에 대한 수요 증가에 맞추어 벤치마크(Benchmark) 데이터셋의 크기가 급격히 증가하고 있음을 다룹니다. 기존의 방법들에서 얻어진 성능 예측의 일관성을 보장함과 동시에 예측 정확도를 유지하기 위한 체계적인 프레임워크가 절실하다고 강조합니다. 저자들은 벤치마크 압축을 최적화 문제로 정의하고, 이를 위한 새로운 방법인 EssenceBench를 제안합니다.

- **Technical Details**: EssenceBench는 본질적으로 그리드 검색을 통한 변형된 유전 알고리즘(Genetic Algorithm, GA)을 활용하여 평가 점수를 압축하는 프레임워크로 설계되었습니다. 이 방법은 초소형 샘플 집합을 지속적으로 추적하며, 샘플 간의 중복성과 성능 변동을 정량화하여 불필요한 샘플을 제거하는 과정을 포함합니다. 이를 통해 전체 데이터셋의 성능을 충실히 재구성하는 것을 목표로 하며, 효율적인 탐색 메커니즘으로 샘플의 속성을 활용합니다.

- **Performance Highlights**: 실험 결과에 따르면, EssenceBench는 HellaSwag 벤치마크에서 10K 샘플을 사용하여 25배 적은 샘플로 모델의 순위를 5% 이내에서 유지할 수 있음을 입증하였습니다. 이러한 결과는 전체 모델 순위의 유지를 의미하며, 200배 적은 샘플에서도 95%의 순위 보존률을 기록했습니다. 이로 인해 LLM(대형 언어 모델) 평가의 효율성을 크게 향상시킬 수 있는 가능성을 보여줍니다.



### Does Weighting Improve Matrix Factorization for Recommender Systems? (https://arxiv.org/abs/2510.10440)
Comments:
          In the proceedings of the Web Conference (WWW) 2025 (11 pages)

- **What's New**: 이 논문에서는 Top-N 추천 및 협업 필터링을 위한 행렬 분해(matrix factorization) 접근법을 다룬다. 특히, 암묵적 피드백 데이터에 대한 가중치(weighting) 전략에 대한 체계적인 연구를 수행하였고, 놀랍게도 비가중치 데이터로 훈련한 대형 모델이 가중치 데이터로 훈련한 모델과 비슷한 성능을 보이거나 심지어 그보다 나은 결과를 보인다는 것을 발견하였다. 이는 기존의 상식에 도전하는 결과로, 특정 조건에서만 가중치가 유익할 수 있음을 시사한다.

- **Technical Details**: 본 연구에서는 사용자-아이템 상호작용 행렬을 분해하여 잠재 패턴을 포착하는 행렬 분해 기술을 기반으로 한다. 특히 가중치 행렬(weight matrix)을 적용한 Weighted Matrix Factorization(WMF)을 통해 사용자와 아이템의 d차원 요소를 학습하며, Frobinius norm을 활용하여 성능 최적화를 시도한다. 또한, 새로운 효율적인 알고리즘을 도출하여 이전에는 계산적으로 다루기 어려운 여러 가중치 목표를 최소화할 수 있는 방법을 개발하였다.

- **Performance Highlights**: 실험 결과, 비가중치로 훈련된 대형 선형 모델이 표준 추천 시스템 벤치마크에서 가중치 기반 모델과 비교하여 유사한 성능을 나타내는 것으로 나타났다. 그러나 소규모 모델에서는 가중치가 유익할 수 있는 특정 상황이 발견되었다. 또한, 다양한 방법에 걸쳐 가중치, 정규화(regularization), 모델 용량(model capacity) 간의 상호작용을 체계적으로 연구하였다.



### Combo-Gait: Unified Transformer Framework for Multi-Modal Gait Recognition and Attribute Analysis (https://arxiv.org/abs/2510.10417)
- **What's New**: 본 연구는 Combo-Gait라는 다중 모달 및 다중 작업 프레임워크를 제안하여 2D 실루엣과 3D SMPL 피쳐를 결합하여 강력한 보행 인식을 수행합니다. 기존의 보행 인식 방법들이 주로 2D 또는 3D 단일 모달에 의존했던 것과 달리, 본 연구는 이들 모달을 결합하여 더 복잡한 인간의 보행 패턴을 포착할 수 있게 합니다. 또한, 복합적인 인간 속성 추정(예: 나이, 성별, BMI)도 동시에 수행하는 다중 작업 학습 전략을 도입합니다.

- **Technical Details**: 본 연구에서 제안하는 Combo-Gait는 비디오에서 2D 실루엣과 3D SMPL 피쳐를 추출하여 이를 융합하는 네트워크 구조를 가지고 있습니다. 이 아키텍처는 CNN 인코더와 다층 퍼셉트론(MLP)을 사용하여 보행 특성과 인간 속성을 통합합니다. 또한, 전체 프로세스의 효율성을 높이기 위해 범용 트랜스포머 아키텍처를 활용하여 다양한 작업에서의 학습을 지원합니다.

- **Performance Highlights**: BRIAR 데이터세트에서의 실험 결과, Combo-Gait는 기존 최첨단 방법들보다 우수한 보행 인식 성능을 보였으며 인간 속성 추정에서도 높은 정확성을 기록했습니다. 이러한 결과는 복합적인 모달 및 다중 작업 학습이 실제 환경에서의 보행 기반 인간 이해를 증진시키는 가능성을 보여줍니다. 또한, 다양한 거리와 피치 각도 조건에서도 효과적인 성능을 발휘함을 입증하였습니다.



### Mesh-Gait: A Unified Framework for Gait Recognition Through Multi-Modal Representation Learning from 2D Silhouettes (https://arxiv.org/abs/2510.10406)
- **What's New**: 이번 연구에서는 Mesh-Gait라는 새로운 다중 모달 복식 걷기 인식 프레임워크를 소개합니다. 이 프레임워크는 2D 실루엣에서 직접 3D 표현을 재구성하여 두 가지 모달리티의 장점을 통합합니다. 기존 방법에 비해 Mesh-Gait는 3D 조인트나 메쉬에서 직접 3D 기능을 학습하기 어려운 점을 해결하고, 중간 표현으로 3D 히트맵을 사용하여 계산 효율성을 유지합니다. 이 방법은 실시간 적용에 적합한 해결책을 제공합니다.

- **Technical Details**: Mesh-Gait의 아키텍처는 이중 분기 구조로 구성되며, 2D와 3D 기능 분기로 나뉩니다. 2D 기능 분기는 2D 실루엣으로부터 2D 걷기 기능을 추출하는 데 사용되고, 3D 기능 분기는 2D 실루엣에서 3D 메쉬를 재구성하고, 해당 모델로부터 3D 걷기 기능을 추출합니다. 훈련 과정에서는 트리플 손실, 교차 엔트로피 손실, L1 손실 및 L2 손실의 조합을 사용하여 모델을 최적화합니다.

- **Performance Highlights**: Mesh-Gait는 여러 벤치마크 데이터셋에서 평가되어 탁월한 성과를 보였습니다. 특히, 전통적인 2D 방법이 어려움을 겪는 변동 시점, 부분 차단 및 환경 소음이 있는 어려운 상황에서도 높은 인식 정확도와 강력한 견고성을 보여주었습니다. 이 접근 방식은 계산 효율성을 개선하여 제한된 자원에서도 실시간 걷기 인식이 가능하게 합니다.



### RefusalBench: Generative Evaluation of Selective Refusal in Grounded Language Models (https://arxiv.org/abs/2510.10390)
- **What's New**: 이 연구에서는 Retrieval-Augmented Generation(RAG) 시스템에서 언어 모델이 잘못된 맥락에 따라 selectively refuse(선택적 거부)를 수행하는 능력이 얼마나 중요한지를 보여줍니다. 연구진은 기존 모델들이 이 기능에서 50% 미만의 정확도를 기록하고, 잘못된 정보에 기반하여 답변을 거부하거나 자신이 없는 답변을 내는 문제를 밝혀냈습니다. 또한, 단순한 정적 벤치마크(static benchmarks)가 이러한 성능을 평가하는 데 한계를 가지고 있음을 강조하며, RefusalBench라는 새로운 평가 방법론을 소개합니다.

- **Technical Details**: RefusalBench는 언어적 교란을 통해 진단 테스트 케이스를 생성하는 프로그램 수립 방법론을 기반으로 하고 있습니다. 이 시스템은 정보 불확실성의 여섯 가지 범주에서 176개의 교란 전략을 활용하여 답변 가능한 질문을 답변 불가능한 질문으로 변화시킵니다. 이러한 평가 방법론은 고유성에 대한 감도를 세밀하게 진단할 수 있으며, 멀티 모델 생성-검증 파이프라인을 통해 정답의 품질을 보장합니다.

- **Performance Highlights**: 30개 이상의 모델을 평가한 결과, 선택적 거부 능력에서 심각한 차이가 발견되었습니다. 연구진은 이 능력이 훈련이 가능하고 조정에 민감한 특성을 지니고 있다는 것을 밝혀내어 모델 개선의 길을 제시하였습니다. 또한, RefusalBench-NQ(단일 문서) 및 RefusalBench-GaRAGe(다중 문서)라는 두 가지 벤치마크를 제공하며, 이러한 새로운 평가 프레임워크의 필요성을 강조합니다.



### FLAMMABLE: A Multi-Model Federated Learning Framework with Multi-Model Engagement and Adaptive Batch Sizes (https://arxiv.org/abs/2510.10380)
- **What's New**: 이번 논문에서는 Multi-Model Federated Learning (MMFL)이라는 새로운 접근 방식을 소개합니다. FLAMMABLE이라는 포괄적인 MMFL 교육 프레임워크를 통해 모델 훈련의 배치 크기를 지능적으로 조정하며, 클라이언트의 시스템 능력에 따라 여러 모델을 동시에 훈련시킬 수 있도록 설계되었습니다. 이는 다양한 데이터와 모델 구조에서의 비효율성을 해소하기 위한 노력의 일환입니다.

- **Technical Details**: FLAMMABLE은 각 클라이언트와 모델에 맞게 훈련 배치를 자동으로 조정하고, 여러 모델의 참여를 최적화합니다. 훈련 시간을 단축시키되 통계적 진행을 해치지 않기 위해 배치 크기와 반복 회수를 동시 조정하는 방법을 채택합니다. 또한, 모델과 클라이언트 간의 상호작용을 극대화하기 위해 클라이언트를 모델에 맞게 배치하는 과정을 체계적으로 설계하였습니다.

- **Performance Highlights**: FLAMMABLE은 다양한 데이터셋과 모델에 대한 평가를 통하여 MMFL의 시간 대비 정확도를 1.1배에서 10.0배까지 향상시키고, 최종 모델 정확도 또한 1.3%에서 5.4%까지 개선시켰습니다. 이는 기존 방법들과의 비교에서 높은 성능을 입증하며, 향후 MMFL 연구의 기초가 될 수 있는 벤치마크 플랫폼을 제공합니다.



### Vision4PPG: Emergent PPG Analysis Capability of Vision Foundation Models for Vital Signs like Blood Pressur (https://arxiv.org/abs/2510.10366)
Comments:
          BHI abstract extended

- **What's New**: PPG(Photoplethysmography) 센서가 최근 웨어러블 및 임상 기기에서 널리 사용되며, 비침습적이고 실시간으로 생리학적 통찰을 제공합니다. 본 연구에서는 Vision Foundation Models(VFM)을 활용하여 PPG 작업에서 SOTA(state-of-the-art) 성능을 달성할 수 있음을 보여주며, 특히 혈압 추정에서 우수한 결과를 나타냅니다. 기존 시간 시리즈 모델을 사용한 연구와 비교할 때, VFMs는 STFT(Short-Time Fourier Transform)와 같은 2D 변환을 통해 PPG 신호를 처리하는 새로운 접근 방식을 제공합니다.

- **Technical Details**: 본 연구에서는 PPG 처리에 적합한 여러 모델을 비교하였습니다. MOMENT라는 일반 시간 시리즈 모델과 PPG에 전문화된 PPG-GPT 모델을 선택하여 VFM과 비교 분석을 실시하였습니다. DINOv3와 SIGLIP-2라는 최신 VFM을 활용하여 PPG 데이터를 변환하고, Parameter-Efficient Fine-Tuning(PEFT) 기법을 적용하여 모델의 성능을 최적화했습니다. 또한, 총 7개 데이터셋을 사용하여 비침습적 혈압 추정을 포함한 다양한 생리적 작업을 수행했습니다.

- **Performance Highlights**: Vision4PPG라는 제안된 접근법은 다양한 비침습적 생리 신호 작업에서 우수한 결과를 보였습니다. 혈압 추정 외에도 심박수, 호흡수, SPO2, 혈중 생화학 물질 농도 추정에서 SOTA 성능을 달성했습니다. 이는 VFMs가 PPG 신호 분석에 있어 보편적으로 적용 가능한 가능성을 보여주며, 임상 연구자들에게 효율적이고 강력한 도구를 제공합니다. 본 연구는 다양한 데이터셋에서 우수한 성능을 입증하여, 생리학적 신호 처리 분야에서 VFMs의 잠재력을 확인했습니다.



### Generative Modeling of Aerosol State Representations (https://arxiv.org/abs/2510.10361)
Comments:
          31 pages, 20 figures

- **What's New**: 이 연구에서는 에어로솔(aerosols)과 클라우드(cloud) 간의 상호작용을 모델링하기 위해 변별력 있는 수치적 정보(variational autoencoder, VAE)를 활용한 새로운 접근 방식이 제시되었습니다. 기존의 에어로솔 상태 표현의 복잡성을 줄이면서도 과학적 의미를 유지하여 기후 시뮬레이션을 보다 효율적으로 수행할 수 있게 하는 방법입니다. 특히 이 모델은 에어로솔의 클라우드 응축 핵(cloud condensation nuclei, CCN) 스펙트럼을 정확하게 재구성하는데 강점을 보이고 있습니다.

- **Technical Details**: 본 연구에서는 고차원 에어로솔 데이터의 압축 표현을 위해 변별력 있는 자동 인코더(VAE) 모델을 도입했습니다. VAE는 원래 수백 개의 변수를 가진 데이터를 단 10개의 잠재 변수로 압축하여 기후 관련 진단 데이터를 효율적으로 보존합니다. 또한, 이 연구의 preprocessing optimization 전략은 모델 성능을 향상시킬 수 있는 데이터 변환을 식별하여, 원래 측정값이 노이즈가 있는 경우에도 안정성을 제공합니다.

- **Performance Highlights**: 모델의 성능 결과에 따르면, CCN 스펙트럼은 가장 정확하게 재구성할 수 있는 반면, 광학적 속성(optical properties)은 중간 정도의 어려움을 겪고 있으며, 얼음 핵 형성(ice nucleation properties) 속성은 가장 어려운 것으로 나타났습니다. 기후 관련 정보의 재구성을 위한 최적의 전처리 방법을 개발하여 CCN 스펙트럼과 같은 정보의 정확성을 더욱 높였습니다. 마지막으로, 생성된 샘플과 테스트 세트 간의 sliced Wasserstein 거리(sliced Wasserstein distance)를 기반으로 한 새로운 현실성 메트릭(realism metric)을 제안하였습니다.



### Learning to Throw-Flip (https://arxiv.org/abs/2510.10357)
Comments:
          Accepted to IROS 2025. Video Summary: this https URL

- **What's New**: 이번 연구에서는 로봇이 객체를 바 Desired landing pose (착륙 자세)로 정확하게 "throw-flip" 할 수 있는 방법을 제시합니다. 기존 연구들은 객체의 착륙 위치에만 초점을 맞춘 반면, 최종 orientation (방향) 조절에 대한 연구는 부족했던 점을 보완합니다. 이 연구의 접근법은 두 가지 주요 설계를 기반으로 하며, 로봇이 객체의 착륙 자세를 독립적으로 제어할 수 있도록 합니다.

- **Technical Details**: 연구진은 Impulse-Momentum principle (충격-운동량 원리)을 이용해 투척 행동을 설계하고, 비선형 동역학의 비모델화 효과를 고려한 회귀 기반 학습 방법을 결합했습니다. 이로 인해, 연구진은 다양한 착륙 자세에 대한 유망한 세트를 크게 확대하였습니다. 또한, 데이터 동화(data assimilation)를 통해 샘플 복잡성을 평균 40% 감소시킬 수 있었으며, CoM(중심 질량) 이동의 영향을 고려하여 과거의 데이터를 활용하는 방식으로 학습 속도를 70% 향상시켰습니다.

- **Performance Highlights**: 실제 로봇 실험을 통해, 제안된 프레임워크가 다양한 자세 목표를 가진 객체를 수십 번의 시도만에 정확하게 throw-flip 하는 데 성공적임을 입증했습니다. 목표는 체적 범위 내에서 ($\pm$5 cm, $\pm$45 degrees)로 설정되었습니다. 이 결과는 기존의 end-to-end learning 방법보다 샘플 효율을 크게 향상시킨 것으로 평가됩니다.



### Learning Operators through Coefficient Mappings in Fixed Basis Spaces (https://arxiv.org/abs/2510.10350)
- **What's New**: 이번 연구에서는 고정 기저 계수 대 계수 네트워크(Fixed-Basis Coefficient to Coefficient Operator Network, FB-C2CNet)를 제안하여, 부분 미분 방정식(PDE)과 같은 해의 연산자를 학습하는 방법을 개발했습니다. 이 접근법은 입력 함수가 미리 정해진 기저 함수로 투영되고, 신경 연산자가 다른 기저 또는 같은 기저에서의 해결 함수의 계수를 예측하는 구조입니다. 이러한 방식은 네트워크 훈련에서 기저 선택을 분리함으로써 훈련 복잡성을 줄이고, 기저의 선택이 근사 정확도에 미치는 영향을 체계적으로 분석할 수 있게 합니다.

- **Technical Details**: FB-C2CNet의 핵심 아이디어는 신경망이 확장 계수 사이의 매핑을 학습할 수 있도록 입출력 함수를 잘 선택된 기저 함수로 표현하는 것입니다. 입력 함수 f는 사전에 결정된 기저 함수로 근사화되며, 신경망 N의 출력은 출력 기저 함수에 해당하는 계수 벡터로, 이는 연산자 출력의 근사를 제공합니다. 이 방식은 노드 값 인코딩에 비해 더 나은 정확도와 일반화를 가능하게 하는 고정 기저 계수 대 계수 학습 패러다임을 구현합니다.

- **Performance Highlights**: 다르시 흐름(Darcy flow), 포아송 방정식(Poisson equations) 및 탄성 문제와 같은 다양한 수치 실험을 통해 FB-C2CNet은 높은 정확도와 계산 효율성을 달성했습니다. 특히, 정규 및 복잡한 고차원 도메인에서 우수한 성능을 발휘하며 실제 적용 가능성을 보여주고 있습니다. 이러한 성과는 FB-C2CNet이 실용적인 연산자 학습 작업에 대한 강력한 잠재력을 지니고 있음을 시사합니다.



### Measuring What Matters: Connecting AI Ethics Evaluations to System Attributes, Hazards, and Harms (https://arxiv.org/abs/2510.10339)
- **What's New**: 최근 10년 동안 AI 시스템의 사회적 및 윤리적 영향을 평가하기 위한 여러 기준이 출현했으나, 이러한 기준들은 대부분 조각적으로 개발 및 활용되고 있습니다. 본 논문에서는 기존의 평가 도구들이 AI 시스템의 구성 요소, 속성, 위험 및 해악과 어떻게 연결되는지를 분석하였습니다. 800개에 달하는 기준이 11개의 AI 윤리 원칙에 해당함을 확인했고, 공정성(fairness), 투명성(transparency), 개인 정보 보호(privacy), 신뢰(trust) 원칙에 중점을 두었음을 드러냈습니다.

- **Technical Details**: 논문은 AI 윤리 원칙에 대한 준수 측정 기준을 평가하는 데 있어 기존 도구들의 유효성과 신뢰성이 결여되어 있다고 지적합니다. 대부분의 측정 기준이 개별적인 구성 요소에 집중하고 있어 시스템 전체를 평가하지 못하고 있으며, 이는 안전성 측면에서 문제를 일으킬 수 있습니다. 저자들은 이러한 문제를 해결하기 위해 시스템 안전(system safety) 관점에서 윤리적 고려를 포함한 다차원 분석을 제안하고 있습니다.

- **Performance Highlights**: 현재의 평가 관행은 분산되어 있으며, 해악이 발생하는 위치와 관련된 기준이 부족합니다. 저자들은 AI 커뮤니티에 시스템 차원의 평가 방식을 채택할 것을 촉구하며, 이를 통해 법적 감독을 강화하고 실용적인 지침을 제공할 수 있음을 강조합니다. 마지막으로, 연구진은 각 구성 요소, 속성, 위험 및 해악에 따라 분류된 데이터 세트를 제공하여 추가 연구와 참여를 지원할 준비가 되어 있습니다.



### On some practical challenges of conformal prediction (https://arxiv.org/abs/2510.10324)
- **What's New**: 이 논문은 conformal prediction(적합 예측)이 갖는 세 가지 주요 도전 과제를 해결하기 위한 새로운 통찰을 제공합니다. 첫째, finite-sample validity(유한 샘플 유효성)의 문제가 있으며, 둘째, 연산 비용이 매우 높을 수 있고, 셋째, 예측 영역의 형상을 제어하기 어렵다는 점에 대해 논의합니다. 저자는 이 세 가지 문제를 동시에 완화할 수 있는 간단한 전략을 제안합니다.

- **Technical Details**: 논문에서는 exchangeable random vectors(교환 가능 랜덤 벡터) 및 non-conformity measure(비적합 측정) 개념을 통해 데이터를 분석합니다. Algorithm 1을 통해 non-conformity score(비적합 점수)를 계산하고 plausibility function(그 가능성 함수)을 통해 예측을 수행합니다. 저자들은 이론적 정당성을 통해 100(1−α)% conformal prediction region(적합 예측 영역)을 확립하는 방법을 제시합니다.

- **Performance Highlights**: 이 논문의 새로운 방법은 모델 유효성 및 연산 비용의 측면에서 기존 방법보다 개선된 성과를 제공합니다. 특히, 저자들은 더 간단하고 효율적인 접근 방식을 통해 다양한 상황에서의 예측 정확성을 높일 수 있음을 보여줍니다. 이러한 접근법은 특히 비모수적 방법이 필요한 학습 분야에 적용될 수 있습니다.



### Grounded AI for Code Review: Resource-Efficient Large-Model Serving in Enterprise Pipelines (https://arxiv.org/abs/2510.10290)
Comments:
          Submitted to MLSys 2026

- **What's New**: 이 논문에서는 자동화된 코드 리뷰의 도입이 느린 이유와 이를 극복하기 위한 생산 시스템을 제안합니다. 특히, 정적 분석 결과와 AST(Abstrat Syntax Tree) 기반의 맥락 추출을 결합하여 효율적인 피드백을 제공하는 새로운 접근 방식을 소개합니다. 이 시스템은 GPU 기반의 서비스 아키텍처를 활용하여 사용자에게 신뢰할 수 있는 빠른 피드백을 제공합니다.

- **Technical Details**: 논문에서는 하이브리드 그라운딩 방법론과 단일 GPU 자원 효율적 서빙 모델을 통해 정적 분석 증거와 LLM(대형 언어 모델) 설명을 결합하는 시스템 아키텍처를 설명합니다. 각 LLM 생성 설명은 컴파일러 검증된 빌드 및 특정 정적 분석 결과에 명시적으로 연동되며, 이를 통해 모델이 보다 구체적인 문제에 대해 논리적으로 접근하도록 합니다. 또한, 이 시스템은 정적 분석 및 LLM을 이용한 사고 방식을 적용하여 인간 리뷰어의 작업 부담을 줄여줍니다.

- **Performance Highlights**: 제안된 시스템은 중간 규모 기업 환경에서 약 59.8초의 평균 첫 피드백 시간으로 평가되었습니다. 이는 경쟁하는 소유권 모델에 비해 개선된 위반 비율을 유지하면서도 보다 효과적으로 위반을 줄이는 성과를 보입니다. 또한, 소규모 내부 설문에 따르면 피드백 과정에서 인력 소모가 감소하고, 리뷰 반복 횟수가 줄어드는 긍정적인 경향이 나타났습니다.



### ArtPerception: ASCII Art-based Jailbreak on LLMs with Recognition Pre-tes (https://arxiv.org/abs/2510.10281)
Comments:
          30 pages, 22 figures. This preprint has been accepted for publication in Elsevier JOURNAL OF NETWORK AND COMPUTER APPLICATIONS (JNCA)

- **What's New**: 이 논문은 ArtPerception이라는 새로운 블랙박스 탈옥 프레임워크를 소개합니다. 이 시스템은 ASCII 아트를 활용하여 최첨단 대형 언어 모델(LLMs)의 보안 조치를 우회하는 방식으로 설계되었습니다. ArtPerception은 비효율적인 반복 공격 방법 대신, 체계적인 두 단계의 방법론을 도입하여 효과성을 극대화합니다.

- **Technical Details**: ArtPerception의 첫 번째 단계는 특정 모델에 대한 최적의 ASCII 아트 인식 파라미터를 결정하기 위한 1회 사전 테스트(pre-test)를 수행합니다. 두 번째 단계에서는 이 정보를 활용하여 매우 효율적인 원샷 공격을 실행합니다. 또한, 연구에서는 LLM의 인식 능력을 평가하기 위한 수정된 레벤슈타인 거리(Modified Levenshtein Distance, MLD) 메트릭을 제안했습니다.

- **Performance Highlights**: 실험을 통해 네 가지 최첨단 오픈소스 LLM에 대한 뛰어난 탈옥 성능을 입증했습니다. ArtPerception은 실제 상업 모델(GPT-4o, Claude Sonnet 3.7 및 DeepSeek-V3)에서도 성공적으로 적용될 수 있음을 보여주었으며, 일반적인 방어 기법(LLaMA Guard 및 Azure의 콘텐츠 필터)에 대한 강인성을 입증했습니다.



### Neural variational inference for cutting feedback during uncertainty propagation (https://arxiv.org/abs/2510.10268)
- **What's New**: 이 논문은 NeVI-Cut라는 새로운 베이esian 추론 방법을 제안합니다. 기존의 cut-Bayes 방법과 달리, NeVI-Cut는 상류 분석(upstream analysis) 데이터를 직접적으로 이용하면서도 상류 데이터나 모델에 접근하지 않아도 가능한 모듈형 신경망 기반의 변별 추론 방법입니다. 이를 통해 분석의 모듈성을 유지하면서도 상류 모델의 변별 근사를 피해 오류를 줄이는 데 기여합니다.

- **Technical Details**: NeVI-Cut는 상류 추정의 Monte Carlo 샘플을 직접 사용하여, 조건부 변별 집합을 명시하기 위해 normalizing flows를 적용합니다. 또한, 모든 상류 샘플에 대한 Monte Carlo 평균 손실의 변별 해결책으로 조건부 cut-posterior를 추정합니다. 이 방식은 선형적인 Kullback-Leibler 근사 속성을 증명하고, 신경 구조의 풍부함과 목표 cut-posterior의 복잡성이 근사 품질에 미치는 영향을 정량화합니다.

- **Performance Highlights**: 시뮬레이션 연구와 두 가지 실제 분석 결과에서, NeVI-Cut는 기존의 cutting feedback 방식에 비해 상당한 계산 효율성을 달성하고, 매개변수적 변별 cut 접근 방식보다 더 정확한 결과를 보여줍니다. 이로 인해 NeVI-Cut는 다양한 분야에서 growing한 문헌에 기여하며, 베이esian 분석의 새로운 가능성을 제시하고 있습니다.



### Unveiling Gamer Archetypes through Multi modal feature Correlations and Unsupervised Learning (https://arxiv.org/abs/2510.10263)
Comments:
          Submitted to Peer Review Journal

- **What's New**: 본 연구는 게이머 프로파일링(gamer profiling)을 위한 통합 데이터 기반 프레임워크를 제안합니다. 이는 심리적 측정(psychological measures), 행동 분석(behavioral analytics), 기계 학습(machine learning)이 결합되어 게이머 성격을 밝혀냅니다. 250명의 참가자에 대한 구조화된 서베이를 통해 다차원적 행동, 동기 및 사회적 데이터를 수집하였습니다. 이 연구는 기존의 데이터 분석 기법을 넘어 새로운 인사이트를 제공합니다.

- **Technical Details**: 연구는 특징 엔지니어링(feature engineering), 연관 네트워크(association-network), 지식 그래프 분석(knowledge-graph analysis)과 비지도 클러스터링을 통합하는 분석 파이프라인을 실시하였습니다. 이 과정에서는 주성분 분석(PCA), 특이값 분해(SVD), t-SNE와 같은 차원 축소 기법을 클러스터 알고리즘(K-Means 등)과 결합하여 적용했습니다. PCA와 K-Means(k=4)를 사용한 모델은 실루엣(Silhouette) 지수 0.4로 최적의 클러스터 품질을 달성했습니다.

- **Performance Highlights**: 연구 결과는 네 가지 아키타입인 몰입형 사회적 이야기 탐색자(Immersive Social Story-Seekers), 규율 있는 최적화자(Disciplined Optimizers), 전략적 시스템 탐색자(Strategic Systems Navigators), 경쟁 팀 구성자(Competitive Team-Builders)로 클러스터링 되었습니다. 이 연구는 상관관계 기반의 네트워크 인사이트와 비지도 학습을 연결하는 재현 가능한 파이프라인을 제공합니다. 행동 상관망과 클러스터링의 통합은 분류 정확성을 향상시키고, 게임 플레이의 동기와 심리적 및 웰빙 결과를 연결하는 포괄적인 관점을 제공합니다.



### Opacity-Gradient Driven Density Control for Compact and Efficient Few-Shot 3D Gaussian Splatting (https://arxiv.org/abs/2510.10257)
- **What's New**: 이 논문에서는 3D Gaussian Splatting (3DGS)의 최적화를 재조정하여 효율성을 우선시하는 새로운 프레임워크를 제시합니다. 기존의 Adaptive Density Control (ADC) 방식을 개선하여, 일정한 알고리즘을 통해 렌더링 오류의 경량 프록시로서 opacity gradient를 사용하는 Error-Driven Densification 기법을 도입했습니다. 이 새로운 접근법은 기존의 과도한 밀도 조정 및 pruning 문제를 해결하여, 더 효율적이고 고품질의 3D 재구성을 가능하게 합니다.

- **Technical Details**: 제안된 프레임워크는 기본적인 3DGS 최적화 알고리즘을 수정하여, 밀도를 조정하는 과정에서 opacity gradient를 기반으로 한 새로운 오류 주도 기법을 도입합니다. 이 기법은 이전의 밀도 조정 방식이 가진 비효율성을 해결하기 위해 더 보수적인 pruning 일정을 필요한 곳에 배치합니다. 이것은 깊이 상관 손실과 결합되어, 최신 기법들과 비교하여 효율성에서 극적인 개선을 이룰 수 있는 방법론을 제공합니다.

- **Performance Highlights**: 테스트 결과, LLFF 데이터셋에서 제안된 모델은 FSGS보다 40% 이상 더 컴팩트한 성능을 보여주었으며(32k vs. 57k primitives), Mip-NeRF 360 데이터셋에서는 약 70%의 크기 감소를 기록했습니다. 이러한 놀라운 성능 개선은 reconstructed metrics에서의 적절한 trade-off를 통해 이루어졌으며, few-shot view synthesis의 품질과 효율성 간의 새로운 상태를 확립했습니다.



### MRI Brain Tumor Detection with Computer Vision (https://arxiv.org/abs/2510.10250)
Comments:
          12 pages, 8 figures, final project report for CS4100 (Machine Learning), Northeastern University, April 2024

- **What's New**: 이 연구는 MRI 스캔에서 뇌종양을 자동으로 탐지하고 분할하는 데 딥 러닝 기술의 적용을 탐구합니다. 기초 로지스틱 회귀, 합성곱 신경망(CNN), 잔여 네트워크(ResNet) 등을 사용하여 뇌종양을 효과적으로 분류하고, U-Net 및 EfficientDet을 통해 종양의 로컬라이제이션과 식별을 개선합니다. 결과적으로, 딥 러닝이 의료 이미징에서 진단의 정확성과 효율성을 높이는 데 기여할 수 있음을 보여줍니다.

- **Technical Details**: 본 연구에서는 7,023개의 MRI 이미지를 포함하는 Brain Tumor MRI Dataset과 110명의 환자에서 수집된 LGG Segmentation Dataset을 활용합니다. 이 데이터셋은 glioma, meningioma, no tumor, pituitary의 네 가지 클래스로 세분화되며, 이는 종양의 형태와 유전적 아형, 환자 결과 간의 연구를 지원합니다. 또한, 소프트웨어는 PyTorch를 기반으로 구현되어 있으며, CNN과 ResNet 모델의 다양한 아키텍처를 최적화하여 뇌종양 탐지 및 분할의 도전 과제를 해결합니다.

- **Performance Highlights**: 로지스틱 회귀 모델은 낮은 정확도를 보였으나, CNN 모델은 이와 비교해 99% 이상의 높은 정확도를 기록했습니다. ResNet은 복잡한 이미지를 처리하는 데 효과적이며, 깊은 네트워크 구조 덕분에 AUC 성능이 개선되었습니다. 깊이 있는 네트워크 특성이 뇌 MRI 이미지에서 중요한 세부사항을 학습하여, 뇌종양의 정확한 분류를 가능하게 합니다.



### ProGress: Structured Music Generation via Graph Diffusion and Hierarchical Music Analysis (https://arxiv.org/abs/2510.10249)
- **What's New**: 이 논문은 인공지능(AI)을 이용한 음악 생성의 발달 추세에 대해 다루고 있습니다. 기존 모델들은 하모닉-멜로딕 구조에서의 구조적 응집력이 부족하고, 음악적으로 해석 가능한 능력이 떨어진다는 한계를 지니고 있습니다. 본 연구는 Schenkerian 분석(SchA) 개념을 도입한 새로운 생성 음악 프레임워크인 ProGress를 제시하여 이러한 문제를 해결하고자 합니다.

- **Technical Details**: ProGress는 최신 심층 모델을 활용하여 이산 확산(diffusion) 그래프 모델을 기반으로 하며, 사용자에게 음악 생성 과정의 다양한 측면을 조정할 수 있게 합니다. 이 모델은 기존의 DiGress 모델을 음악 생성에 맞게 새롭게 조정하였고, SchA에서 영감을 받은 구문 융합 방법론을 포함합니다. 이러한 방법론 덕분에 사용자는요구하는 음악 스타일과 형식에 맞춰 더 구조화된 음악을 만들 수 있습니다.

- **Performance Highlights**: 인간 실험 결과, ProGress는 기존의 최첨단 음악 생성 모델들에 비해 우수한 성능을 보였습니다. 또한, 본 모델은 현재의 경쟁 모델보다 훨씬 적은 수의 파라미터를 사용하면서도, 해석 가능한 음악 생성을 가능하게 하였습니다. 이러한 특징을 통해 ProGress는 보다 일관된 및 음악적으로 만족스러운 작곡을 생성하는 데 긍정적인 평가를 받았습니다.



### Kernel Treatment Effects with Adaptively Collected Data (https://arxiv.org/abs/2510.10245)
- **What's New**: 본 논문에서는 적응적 데이터 수집 하에서 분포적 추론(distributional inference)을 위한 최초의 커널 기반 프레임워크를 제시합니다. 기존의 접근 방식은 고정 설계(fixed design) 하에서 비대칭성을 유지하는 것으로, 이 연구에서는 반면에 데이터를 적응적으로 수집하여 생성된 통계적 속성을 새로운 방법으로 접근합니다. 본 연구는 과거의 데이터를 활용한 이중 강건估算기(doubly robust estimator)를 구성하며, 이로써 비대칭성 하에서도 안정적인 추정을 가능케 합니다.

- **Technical Details**: 이 연구는 힐베르트 공간 마틴게일 중심 극한 정리(Hilbert-space martingale CLT)를 활용하여 비대칭적 수집 아래에서도 수렴 안정성을 보장합니다. 이중 강건 추정기와 변동성 안정화(variance stabilization)을 결합하여 분포적 커널 처리 효과(kernel treatment effects; KTE)를 분석합니다. 본 방법론은 각 라운드의 변동성을 안정화하는 새로운 테스트 방법인 샘플 적합 안정화 검정(sample-fitted stabilized test)을 도입하여 잘 조정된 유효 유의 수준(valid type-I error)을 확보합니다.

- **Performance Highlights**: 실험 결과, 제안한 방법은 평균 변화(mean shifts)와 고차(moment)의 차이에서 높은 성능을 나타내며, 일반적으로 평균 효과에 제한된 기존의 적응적 기준선보다 우수한 성능을 보입니다. 새로운 방법은 고차 모멘트와 분포적 효과 사이의 관계를 명확히 분석할 수 있도록 해주며 분포적 결과를 실질적으로 유용한 형태로 제시합니다. 실험에서 확인된 다양한 시뮬레이션 결과는 이 방법의 유효성과 정확성을 뒷받침합니다.



### You only need 4 extra tokens: Synergistic Test-time Adaptation for LLMs (https://arxiv.org/abs/2510.10223)
Comments:
          Under Review

- **What's New**: 이 논문은 분야별 (domain-specific) 문제에 대한 라벨이 없는 테스트 시간 적응(test-time adaptation) 방식을 다룹니다. SyTTA(Synergistic Test-time Adaptation)라는 새로운 프레임워크를 제안하며, 이는 추가적인 감독 없이 실시간으로 모델을 조정할 수 있도록 합니다. 입력측의 perplexity와 출력측의 predictive entropy라는 두 가지 불확실성 신호를 결합하여 성능 저하를 완화합니다.

- **Technical Details**: SyTTA는 문제 설정에서 공통적으로 나타나는 불확실성을 다룹니다. 데이터 분포가 변경될 때, 입력에 대한 perplexity가 증가하며, 출력의 예측 엔트로피(predicitive entropy) 또한 높아집니다. 이를 통해 모델이 더 효과적으로 적응할 수 있도록 하며, 4-16개의 추가 토큰만으로도 빠르게 업데이트가 가능합니다. 모델은 Dynamic-Ref 모드와 Static-Ref 모드 중 하나를 선택하여 사용 가능합니다.

- **Performance Highlights**: SyTTA는 다양한 모델 아키텍처와 분야별 벤치마크에서 일관된 성능 향상을 보여주었습니다. 특히 농업 관련 질문 응답에서 Qwen-2.5-7B 모델이 Rouge-LSum을 120% 이상 향상시켰습니다. 이러한 결과는 라벨이 부족한 환경에서도 효과적인 테스트 시간 적응이 가능함을 시사합니다.



### LOOPerSet: A Large-Scale Dataset for Data-Driven Polyhedral Compiler Optimization (https://arxiv.org/abs/2510.10209)
- **What's New**: 이번 논문에서는 LOOPerSet라는 새로운 공개 데이터셋을 소개합니다. 이 데이터셋은 220,000개의 고유한 합성 다각형 프로그램에서 파생된 2800만 개의 레이블 데이터 포인트를 포함하고 있습니다. LOOPerSet은 데이터 기반의 컴파일러 최적화를 위한 대규모 자료를 제공하여 연구자들이 효율적으로 학습 및 평가할 수 있도록 돕습니다.

- **Technical Details**: LOOPerSet은 각 데이터 포인트가 프로그램과 복잡한 변환 순서를 실제 성능 측정 값에 매핑하도록 설계되었습니다. 이 데이터셋은 프로그램 수행에 대한 다양한 최적화 시퀀스를 포함하며, 모든 최적화는 다각형 종속성 분석을 통해 정확성을 확인하였습니다. 성능 측정은 실제 하드웨어에서 수행되어 신뢰성을 보장합니다.

- **Performance Highlights**: LOOPerSet 데이터셋은 머신러닝 모델의 훈련 및 평가, 새로운 데이터 기반 컴파일러 휴리스틱 발견에 유용합니다. 연구자들은 이를 활용하여 하드웨어 이식성 문제를 해결하고, 새로운 아키텍처에 맞추어 미세 조정 할 수 있습니다. LOOPerSet는 학문 및 상업적 연구 목적으로 자유롭게 사용할 수 있으며, Hugging Face에서 접근할 수 있습니다.



### BrainForm: a Serious Game for BCI Training and Data Collection (https://arxiv.org/abs/2510.10169)
Comments:
          15 pages, 6 figures. Author-accepted version. Accepted for presentation at the Brain Informatics 2025 conference, to appear in Springer Lecture Notes in Artificial Intelligence (LNAI) Brain Informatics Books Series. The final authenticated version will be available via SpringerLink

- **What's New**: 이번 연구에서는 BrainForm이라는 게이미피케이션된 뇌-컴퓨터 인터페이스(BCI) 훈련 시스템을 제안합니다. 이 시스템은 소비자 하드웨어로 쉽게 데이터 수집이 가능하며, 반복적인 세션을 통해 사용자들이 BCI 조작 기술을 개발하는 방식을 연구했습니다. 또한, 두 가지 시각 자극 텍스처의 효과에 대한 인지 및 성능 영향도 분석하였습니다.

- **Technical Details**: BrainForm은 event-related potentials (ERP)을 활용하여 사용자가 적을 물리치고 퍼즐을 해결하는 게임화된 환경입니다. 연구는 사용자들이 BCI 제어 기술을 습득하는 과정을 조사하며, 10개의 깜박이는 목표를 포함한 구조화된 프로토콜을 통해 진행되었습니다. 또한 옵티컬 자극의 두 종류를 비교하여 시각적 피로와 인지적 편안함도 평가했습니다.

- **Performance Highlights**: 연구 결과, 참가자들은 훈련 세션을 통해 Task Accuracy와 Information Transfer Rate (ITR)가 향상되었고, 두 텍스처 간 성능 차이는 없었지만 시각적 자극에 따라 안구 자극이 증가하는 것으로 나타났습니다. BrainForm은 BCI 연구 도구로서 확장성과 사용자 친화성을 가지고 있으며, 사용자 참여를 지속하기 위한 가이드라인을 제공합니다.



### YOLOv11-Litchi: Efficient Litchi Fruit Detection based on UAV-Captured Agricultural Imagery in Complex Orchard Environments (https://arxiv.org/abs/2510.10141)
- **What's New**: 이번 논문은 UAV(무인 항공기) 기반의 리치 검출을 위해 특별히 설계된 YOLOv11-Litchi라는 경량화되고 강력한 탐지 모델을 소개합니다. 이 모델은 작은 대상 크기, 대규모 매개변수로 인한 배포의 어려움, 빈번한 대상 occlusion(가림 현상)과 같은 주요 문제를 해결하는 데 중점을 두고 있습니다. 이를 위해 다중 스케일 잔여 모듈(multi-scale residual module), 경량화된 피처 융합 방법, 그리고 occlusion 탐지 헤드가 도입되었습니다.

- **Technical Details**: YOLOv11-Litchi는 YOLOv11 프레임워크를 기반으로 하며, 모델 매개변수 크기는 6.35MB로 YOLOv11의 기준보다 32.5% 더 작습니다. 이 모델은 개선된 mAP(mean Average Precision) 90.1%와 F1-Score 85.5%를 달성하여 높은 정확도를 유지하면서도 경량화를 실현했습니다. 또한, 초당 57.2 프레임(FPS)의 속도로 실시간 탐지 요구 사항을 충족합니다.

- **Performance Highlights**: 실험 결과, YOLOv11-Litchi는 복잡한 과수원 환경에서 UAV 기반의 리치 탐지에 적합하다는 것을 입증했습니다. 이 모델은 농업의 정밀 관리를 위한 광범위한 응용 가능성을 보여주며, 전통적인 수동 방법의 제한을 초월하여 보다 빠르고 정확한 솔루션을 제공합니다. 더 나아가 이 연구는 UAV와 심층 학습의 통합이 농업 생산성을 얼마나 향상시킬 수 있는지를 시사합니다.



### The Hybrid Multimodal Graph Index (HMGI): A Comprehensive Framework for Integrated Relational and Vector Search (https://arxiv.org/abs/2510.10123)
- **What's New**: 이 논문은 하이브리드 다중 모드 그래프 인덱스(HMGI)를 소개하며, 이는 전문 벡터 데이터베이스와 전통적인 그래프 데이터베이스 간의 격차를 해소하기 위한 새로운 프레임워크입니다. HMGI는 약식 최근접 이웃 검색(ANNS)과 깊이 있는 그래프 탐색 쿼리를 결합하여 다중 모드 데이터에 대한 효율적인 하이브리드 쿼리를 가능하게 합니다. 이 프레임워크는 Neo4j와 같은 플랫폼의 그래프 데이터베이스 아키텍처와 통합된 벡터 검색 기능을 활용하여 데이터의 복잡한 관계를 효과적으로 처리합니다.

- **Technical Details**: HMGI 프레임워크는 통합 하이브리드 쿼리 처리, 모드 인식(partitioning) 인덱싱, 적응형 인덱스 관리라는 세 가지 주요 혁신으로 구성됩니다. 사전 정의된 모드에 따라 다중 모드 임베딩을 파티셔닝하여 인덱스 구조를 최적화하고, 데이터의 동적 수집을 지원하기 위한 낮은 오버헤드 인덱스 업데이트를 처리하는 시스템을 제공합니다. 심층적으로 그래프와 벡터 유사성을 결합하는 쿼리를 통해 더 나은 성능을 달성하는 것을 목표로 합니다.

- **Performance Highlights**: HMGI는 관계 중심의 쿼리에서 3배 향상된 쿼리 처리량을 보여주며, 필터링 쿼리에서는 검색 공간을 최대 70% 줄이고 메모리 사용량을 50% 절감하는 혁신적인 방법을 제공합니다. 동적 관계가 많은 시나리오에서 20-30% 향상된 재현율을 기록하며, 전반적인 검색 정확도가 향상됩니다. 실험적으로 HMGI는 Neo4j 및 Milvus와 같은 기존 시스템과 비교하여 우수한 검색 정확도와 지연 시간을 달성합니다.



### Uncertainty-Aware Post-Detection Framework for Enhanced Fire and Smoke Detection in Compact Deep Learning Models (https://arxiv.org/abs/2510.10108)
Comments:
          Accepted and to be presented at the International Conference on Smart Multimedia (ICSM 2025) - this https URL

- **What's New**: 이 논문에서는 화재 및 연기 감지의 정확성을 높이기 위해 불확실성을 고려한 후처리 프레임워크를 제안합니다. 기존의 비전 기반 감지 방법이 효율성과 신뢰성 간의 균형을 맞추는 데 어려움을 겪고 있다는 점을 강조하고 있으며, YOLOv5n과 YOLOv8n과 같은 경량 딥러닝 모델이 사용되고 있다는 것을 알리고 있습니다. 본 프레임워크는 감지 점수를 통계적 불확실성과 도메인 관련 시각적 단서를 통해 조정하여, 감지 후 신뢰도를 보강합니다.

- **Technical Details**: 제안된 포스트-디텍션 프레임워크는 YOLO와 같은 기본 객체 감지 모델에서 감지 결과를 정제하기 위해 불확실성 추정 및 감지 영역 특성 분석을 통합합니다. 기존의 후처리 방법들과 달리 이 프레임워크는 휴리스틱 기반의 신뢰도 조정을 대신하여 학습된 Confidence Refinement Network (CRN)를 도입하여 더 적응적이고 견고한 감지 파이프라인을 돕습니다. 각 감지된 경계 상자는 색상, 가장자리 및 질감 특성에 따라 평가되며, 이 과정을 통해 화재 및 연기 지역과의 일관성을 확보합니다.

- **Performance Highlights**: D-Fire 데이터셋에 대한 실험 결과, 제안된 후처리 방법이 기존의 기준에 비해 정밀도, 재현율 및 평균 정밀도(Mean Average Precision)가 향상되었음을 보여줍니다. 이 연구는 경량 딥러닝 모델이 실제 화재 및 연기 감지에 더욱 견고하게 작용할 수 있도록 지원하는 후처리 신뢰도 보강의 효과를 강조합니다. 또한 기존 기술들과 비교할 때, 이 방법은 컴퓨터 자원의 소모가 적으면서도 유의미한 성능 향상을 제시합니다.



### Cooperative Pseudo Labeling for Unsupervised Federated Classification (https://arxiv.org/abs/2510.10100)
Comments:
          Accepted by ICCV 2025

- **What's New**: 본 논문은 Unsupervised Federated Learning (UFL)을 CLIP을 활용하여 분류 문제에 처음으로 확장하였으며, 이를 위해 새로운 방법론인 FedCoPL(Federated Cooperative Pseudo Labeling)을 제안합니다. 기존의 UFL 연구들은 주로 표현 학습 및 군집 과제에 집중하였으나, 이번 연구에서는 분류 문제를 탐구하는 새로운 기회를 제시합니다. FedCoPL을 통해 클라이언트는 추정한 의사 레이블 분포를 서버에 업로드하고, 서버는 이를 조정하여 클래스 간의 글로벌 불균형을 방지합니다.

- **Technical Details**: FedCoPL은 협력적 의사 레이블링(cooperative pseudo labeling)과 부분 프롬프트 집계(partial prompt aggregation)라는 두 가지 핵심 컴포넌트를 포함합니다. 클라이언트는 신뢰도 기반 및 엔트로피 기반 필터링 방법을 사용하여 의사 레이블 분포를 추정하고, 서버는 각 클라이언트의 분포를 조정하여 클래스 불균형 문제를 해결합니다. 더불어, 시각적 프롬프트는 서버에서 집계되고, 개인화된 지식을 인코딩하는 텍스트 프롬프트는 로컬에 유지되도록 하는 부분 프롬프트 집계 프로토콜을 도입하였습니다.

- **Performance Highlights**: 전통적인 연합 프롬프트 학습 벤치마크를 사용한 광범위한 실험 결과, 제안된 FedCoPL이 기존 방법들에 비해 우수한 성능을 보임을 확인하였습니다. 본 연구는 CLIP의 제로샷(zero-shot) 분류 능력을 활용하여 더 복잡한 과제를 처리할 수 있는 새로운 가능성을 보여줍니다. 또한, 데이터의 불균형이 존재하는 경우에도 효과적으로 클라이언트 간의 협업을 촉진하고 개인화를 유지할 수 있는 방법론을 제시하였습니다.



### Uncovering Singularities in Feynman Integrals via Machine Learning (https://arxiv.org/abs/2510.10099)
- **What's New**: 이 논문에서는 다중 루프 파인만 적분의 전체 기호 알파벳(symbol alphabet)을 추출하기 위한 기호 회귀(symbolic regression) 기반의 머신러닝 프레임워크를 소개합니다. 이 방법은 단순화(reduction)가 아닌 해석적 구조(analytic structure)에 초점을 맞춰 다양한 적분 가족에 적용 가능하고 해석이 용이합니다. 비트리비얼(nontrivial) 사례에서도 기호 알파벳을 성공적으로 재구성하여 강건성과 일반성을 입증하였습니다.

- **Technical Details**: 일반적인 LL-루프 파인만 적분은 루프 수(LL)와 분모 및 분자 요소를 나타내는 정수(αi)로 작성될 수 있으며, IBP 관계 및 가우시안 소거법을 통해 자산 파인만 적분의 기준(master integrals) 집합을 생성합니다. 이 논문에서는 기호 알파벳을 포함하여 다차원 해석을 위한 범주를 고려하며, 전통적인 방법의 한계를 넘어 기호 회귀 프레임워크를 사용하여 기호 문자를 탐색하고 확인하는 프로세스를 제안합니다.

- **Performance Highlights**: 기호 회귀는 물리적 변수 간의 관계를 캡처하는 해석적 표현을 식별하는 것을 목표로 하며, 다양한 후보 법칙의 효율성을 동적으로 평가할 수 있습니다. PySR 툴킷을 활용하여 고성능 진화 알고리즘으로 구성된 여러 후보 수식이 최적의 정확도와 복잡성 사이의 균형을 이루도록 합니다. 논문에서 제안한 방법은 비트리비얼 다중 루프 예제에서 기호 문자의 시스템적 식별을 성공적으로 입증하였습니다.



### Pharmacist: Safety Alignment Data Curation for Large Language Models against Harmful Fine-tuning (https://arxiv.org/abs/2510.10085)
- **What's New**: 이 논문에서는 대형 언어 모델의 유해한 파인튜닝 문제를 다루기 위해 새로운 데이터 선택 솔루션인 'Pharmacist'를 제안합니다. 기존 방법들이 원래의 안전-정렬 데이터의 품질을 간과한 것에 주목하며, 이는 방어 성능과 계산 효율성에서 한계를 초래하고 있음을 강조합니다. Pharmacist는 안전성과 품질이 높은 코어 서브셋을 선택하여 유해한 파인튜닝에 대한 방어를 강화하는 방법론을 제시합니다.

- **Technical Details**: Pharmacist는 정렬 데이터 선택기를 훈련시켜 고품질의 안전-critical 데이터는 상향 조정하고, 저품질의 비안전-critical 데이터는 하향 조정하는 방식으로 작동합니다. 이 방법은 기존의 데이터 선택 방법보다 방어 및 추론 성능에서 더 나은 성과를 보이며, 특히 SFT(Supervised Fine-Tuning) 기법을 사용할 때 뛰어난 방어 성능을 확보합니다. 본 연구에서는 안전-정렬 데이터에서 고품질 서브셋을 선택함으로써 전체 학습 효율 또한 2.46배 향상되는 효과를 보였습니다.

- **Performance Highlights**: Pharmacist를 이용할 경우, 기존 데이터 선택 방법에 비해 방어 성능이 평균 3.54% 향상되며, 추론 성능도 2.8% 증가합니다. 또한, RepNoise 및 T-Vaccine과 같은 기존 방어 방법과의 통합 시, 방어 성능이 각각 2.60% 및 3.30% 향상되었으며, 훈련 시간은 56.83% 및 57.63% 단축되었습니다. 이러한 결과는 Pharmacist가 기존의 안전-정렬 방어 방법들과 효과적으로 통합될 수 있음을 보여줍니다.



### Diversity Augmentation of Dynamic User Preference Data for Boosting Personalized Text Summarizers (https://arxiv.org/abs/2510.10082)
- **What's New**: 이 논문에서는 개인화된 문서 요약의 필요성을 강조하며, PerAugy라는 새로운 데이터 증강 기법을 제안합니다. 이 기법은 사용자 선호 이력과 요약 데이터의 동적 다양성을 활용하여, 개인화된 요약 모델의 성능을 크게 향상시킵니다. PerAugy는 사용자 상호작용 그래프(User Interaction Graph, UIG)를 기반으로 하여, 다양한 사용자 행동 프로필을 생성하고, 이로 인해 요약의 개인화 수준을 높입니다.

- **Technical Details**: PerAugy는 교차 경로 셔플링(cross-trajectory shuffling)과 요약 콘텐츠 섭동(perturbation)을 결합한 새로운 데이터 증강 기술입니다. 논문에서는 두 가지 주요 기술, 즉 이중 셔플링(Double Shuffling, DS)과 확률적 마르코프 변동(Stochastic Markovian Perturbation, SMP)을 사용하여 다채로운 사용자 프로필을 생성합니다. 이러한 접근 방식은 요약 모델이 더욱 다양한 이력을 경험하게 하여 개인적 선호를 효과적으로 반영할 수 있도록 돕습니다.

- **Performance Highlights**: 실험 결과, PerAugy로 증강된 데이터는 최신 사용자 인코더 모델인 NAML, EBNR, NRMS의 성능을 평균 24%, 25%, 18% 향상시켰습니다. 개인화된 요약 프레임워크에서는 GTP와 PENS 설정에서 평균 61.2%의 향상을 보였고, PENS+NRMS+T2 조합에서는 75%까지 도달했습니다. 또한, PerAugy는 자원이 적은 도메인에서도 효과적으로 일반화되어 성능 향상 효과를 보였습니다.



### Improving Speech Emotion Recognition with Mutual Information Regularized Generative Mod (https://arxiv.org/abs/2510.10078)
- **What's New**: 이 논문에서는 음성 감정 인식(SER)을 위한 새로운 데이터 증강 프레임워크를 제안했습니다. 이 프레임워크는 교차 모달 정보 전송(cross-modal information transfer)과 상호 정보 정규화(mutual information regularization)를 사용하여 데이터 품질을 개선합니다. 또한, 우리는 이 데이터 증강 기술을 멀티모달(multimodal) 입력으로 확장함으로써 다양한 데이터를 효과적으로 처리가 가능함을 보여줍니다.

- **Technical Details**: 주요 기술적 요소는 Information GAN(InfoGAN)을 활용하여 오디오 특징을 생성하고, 상호 정보를 정규화하는 방식입니다. 이 과정에서 생성된 샘플은 실제 레이블과의 종속성을 보장하는 기능을 갖추고 있습니다. 연구진은 세 가지 벤치마크 데이터셋(IEMOCAP, MSP-IMPROV, MSP-Podcast)에서 이 프레임워크를 테스트하였고, 음성 감정 분류의 성능이 크게 향상된 것을 발견했습니다.

- **Performance Highlights**: 우리의 프레임워크는 기존의 방법들에 비해 감정 예측 성능을 개선시켰습니다. 실험 결과, 실제 및 생성된 오디오와 텍스트 피처를 기반으로 한 음성-텍스트 융합 감정 분류에서 개선 효과가 확인되었습니다. 특히, 상호 정보 정규화 모듈이 생성된 데이터와 감정 및 텍스트 정보 간의 의존성을 검증하는 데 중요한 역할을 한다는 점이 주목할 만한 성과입니다.



### Calibrating Generative Models (https://arxiv.org/abs/2510.10020)
Comments:
          Our codebase accompanying the paper is available at: this https URL

- **What's New**: 이 논문은 생성 모델의 잘못된 보정(miscalibration) 문제를 다룹니다. 기존의 방법론들이 개별 샘플의 기여만 고려하는 반면, 저자들은 Kullback-Leibler 발산(KL divergence)을 최소화하면서 보정 제약 조건을 만족하는 가장 가까운 모델을 찾는 방법을 제안합니다. 이를 위해 두 가지 대체 목적 함수인 relax loss와 reward loss를 도입하여 생성 모델을 보정합니다.

- **Technical Details**: 저자들은 CGM-relax와 CGM-reward라는 두 가지 알고리즘을 제안합니다. 이 알고리즘들은 샘플을 생성하고 KL 발산의 경량 추정기를 계산하는 데 중점을 둡니다. relax loss는 제약 조건을 직접 적용하는 대신 보정 위반 패널티로 대체함으로써 문제의 복잡성을 줄입니다.

- **Performance Highlights**: 평균적으로 CGM 방법은 수백 가지의 동시 제약 조건을 만족시키면서 보정 오류를 대폭 줄이는 성능을 보여줍니다. 이 모델은 단백질 설계, 이미지 생성, 언어 모델링과 같은 다양한 응용 분야에 적용되었습니다. 특히, 최대 10억 개의 매개변수를 가진 모델에서도 효과적으로 보정을 진행할 수 있음을 입증하였습니다.



### Neuro-inspired automated lens design (https://arxiv.org/abs/2510.09979)
- **What's New**: 이번 연구에서는 OptiNeuro라는 자동 렌즈 설계 프레임워크를 제안하며, 이는 맨눈의 성능에 준하는 결과를 도출합니다. 이 시스템은 낮은 성능을 보이는 렌즈를 점진적으로 제거하면서 남은 후보들을 최적화하여 고품질 렌즈를 자동으로 디자인합니다. 산업 표준 렌즈 설계 소프트웨어에 비해 빠르게 더 다양한 렌즈 구조를 탐색할 수 있는 잠재력을 지니고 있습니다.

- **Technical Details**: OptiNeuro는 우선 물리적 제약 조건을 고려하여 초기 렌즈 구조를 생성합니다. 그 후 물체의 성능을 개선하기 위해 비율에 따라 렌즈를 Eliminating하고, 잔여 후보에 대한 성능 최적화를 반복적으로 수행합니다. 이를 통해 복잡한 비구면 렌즈의 설계를 자동으로 수행하며, 조건의 유효성을 향상시킵니다.

- **Performance Highlights**: OptiNeuro는 특히 복잡한 비구면 렌즈 설계 과제를 통해 기존의 자동화된 렌즈 설계 방법에 비해 개선된 성능을 보여주었습니다. 다양한 비구면 렌즈 설계 작업에서 퀘이시 휴먼 레벨(Quasi-human-level)의 설계 능력을 입증했으며, 미지의 렌즈 구조 탐색을 촉진하는데 기여할 가능성이 있습니다. 이는 렌즈 설계의 효율성을 향상시키고, 연구자들이 고품질 후보 솔루션을 평가하는 데 집중할 수 있도록 돕습니다.



### Operationalizing AI: Empirical Evidence on MLOps Practices, User Satisfaction, and Organizational Contex (https://arxiv.org/abs/2510.09968)
- **What's New**: 이번 연구는 인공지능(AI) 개발 플랫폼에 대한 8,000개 이상의 사용자 리뷰를 분석하여 머신 러닝 운영(MLOps) 관행의 효과를 조명합니다. MLOps는 소프트웨어 엔지니어링 원칙을 머신러닝 라이프사이클 관리의 특수 요구와 통합하는 모범 사례입니다. 이러한 연구는 MLOps의 구현이 AI 애플리케이션의 개발과 운영에 어떠한 도움을 주는지에 대한 실제적인 증거를 제공합니다.

- **Technical Details**: 연구팀은 제로샷 분류(zero-shot classification) 기술을 사용하여, 지속적 통합과 배포(Continuous Integration and Delivery, CI/CD), 워크플로우 오케스트레이션(workflow orchestration), 재현성(reproducibility), 버전 관리(versioning), 협업(collaboration), 모니터링(monitoring) 등 아홉 가지 확립된 MLOps 관행에 대한 사용자 리뷰의 감정을 측정했습니다. 연구 결과, 총 아홉 가지 관행 중 일곱 가지가 사용자 만족도와 긍정적인 관계를 보였으며, 이는 효과적인 MLOps 구현이 AI 개발에 실질적인 가치를 기여하고 있음을 나타냅니다.

- **Performance Highlights**: 작은 회사의 리뷰어들은 특정 MLOps 관행에 대해 덜 자주 언급하였으며, 이는 조직의 맥락이 MLOps의 중요성과 연관성에 영향을 미침을 시사합니다. 그러나 기업 규모는 MLOps와 만족도 간의 관계를 조절하지 않는 것으로 보입니다. 결과적으로, MLOps 관행이 적용되면 조직적인 환경에 상관없이 보편적으로 긍정적인 영향을 미친다고 할 수 있습니다.



### Egocentric Visual Navigation through Hippocampal Sequences (https://arxiv.org/abs/2510.09951)
Comments:
          20 pages, 21 figures. This is a conference submission

- **What's New**: 이 논문은 해마에서의 장소 세포(sequence of place cells)가 내재적 순환 회로(intrinsic recurrent circuitry)에서 발생한다고 제안합니다. 이는 기존 이론에서 제기된 인지적 목표(cognitive objectives)와 다른 기계적(mechanistic) 해석을 제공합니다. 연구팀은 이러한 회로가 입력이 없더라도 활동을 전파하여 시간이 오래 걸리는 메모리를 생성한다고 주장합니다.

- **Technical Details**: 연구에서 제안된 모델은 CA3에서의 위치 기반 시퀀스(generator)를 통해 희소한 입력을 처리합니다. 입력은 dentate gyrus(DG)에서 전송되며, CA3에서는 공간 코드를 생성합니다. 이 시스템은 LSTM과 비교하여 희소 입력 조건 하에서 더 우수한 성능을 보여줍니다. 각 구성 요소는 심층 학습 기반의 다양한 기술 모델에서 영감을 받았습니다.

- **Performance Highlights**: 모델은 복잡한 기하학적 단서를 사용하지 않고도 연속 미로(navigation maze)를 해결할 수 있습니다. 특히, 희소한 입력에서 시퀀스 생성과 입력 간의 시너지 효과가 우수한 성과로 이어졌습니다. 훈련 과정에서 위치 필드 및 작업 의존적 재배치(task-dependent remapping)와 같은 현상이 자연스럽게 나타나며, 이는 모델의 성공적인 내비게이션 성과에 기여합니다.



### Beyond Fertility: Analyzing STRR as a Metric for Multilingual Tokenization Evaluation (https://arxiv.org/abs/2510.09947)
Comments:
          NeurIPS 2025 Workshop

- **What's New**: 이 논문은 대규모 언어 모델(LLMs)에서 토큰화(tokenization)의 중요성을 강조합니다. 기존의 평가 지표인 fertility가 언어와 도메인 간의 어휘 분배를 제대로 나타내지 않는 문제를 지적하며, 단일 토큰 유지율(Single Token Retention Rate, STRR)이라는 새로운 지표를 제안합니다. STRR는 단일 토큰으로 보존된 단어의 비율을 측정하여 언어 간 공정성을 평가하는 데 도움을 줄 수 있습니다.

- **Technical Details**: 이 연구에서는 여섯 개의 널리 사용되는 LLM 토크나이저를 분석하였으며, 영어, 중국어, 힌디어 등 총 일곱 개 언어를 대상으로 했습니다. 기존의 fertility 지표는 복잡한 다국어 환경에서의 어휘 분배를 간과하지만, STRR은 각 언어에서 전체 단어 보존의 비율을 정량화하여 이 문제를 해결합니다. 이를 통해 언어별로 토크나이저가 어휘를 어떻게 할당하는지에 대한 명확한 인사이트를 제공합니다.

- **Performance Highlights**: 분석 결과, 영어는 두 도메인 모두에서 높은 일관성을 보이는 반면, 중국어는 높은 fertility를 기록했습니다. 힌디어는 가장 낮은 STRR을 보여, 심각한 단어 분절화를 드러냈습니다. STRR을 통해 연구진은 현재 토크나이저의 불평등한 언어 지원 문제를 명확히 수치화하였으며, 이는 공정하고 효율적인 다국어 토크나이저 설계에 기초적인 지침을 제공합니다.



### Explainable Human-in-the-Loop Segmentation via Critic Feedback Signals (https://arxiv.org/abs/2510.09945)
Comments:
          Submitted to a computer vision conference (under review)

- **What's New**: 이번 연구에서는 인간 피드백을 단순한 추가 레이블로 보지 않고, 모델의 오류를 교정하기 위한 개입 신호(interventional signals)로 활용하는 새로운 접근 방식을 제안합니다. 이를 통해 단순히 더 많은 데이터를 제공하는 것이 아니라, 모델이 왜 잘못 예측했는지를 식별하고 체계적으로 수정하는 방법론을 개발하였습니다. 이러한 인간-기계 협력 방식은 예상치 못한 오류를 발견했을 때 매우 효과적입니다.

- **Technical Details**: 제안된 프레임워크는 세 가지 주요 메커니즘을 통합합니다: Critic Interface, Counterfactual Data Generation, Feedback Propagation입니다. Critic Interface는 사용자가 세분화 오류를 수정하고 피드백을 제공할 수 있는 시각적 편집 도구를 제공하며, Counterfactual Data Generation은 사용자 수정에 따른 대조 쌍을 생성합니다. Feedback Propagation은 수정 사항을 시각적으로 유사한 이미지에 확장하여 데이터셋 전반에 걸쳐 교정 작업을 효율적으로 진행할 수 있게 합니다.

- **Performance Highlights**: 본 연구에서는 제안한 프레임워크가 베이스라인 재교육 방법에 비해 주석 작업을 3-4배 줄이는 동시에, 도전적인 cubemap 데이터에서 최대 9 mIoU 포인트(상대적 12-15% 향상) 증가한 세분화 정확도를 얻었다고 보고하였습니다. 또한, Cityscapes, ADE20K와 같은 벤치마크 데이터셋에서도 경쟁력 있는 성능을 유지하며, 새로운 도메인에 대한 일반화 능력도 향상되었습니다.



### Structured Cooperative Multi-Agent Reinforcement Learning: a Bayesian Network Perspectiv (https://arxiv.org/abs/2510.09937)
- **What's New**: 이번 논문은 다중 에이전트 강화 학습(MARL) 알고리즘의 효율성과 확장성을 높이기 위한 체계적인 접근 방식을 제안합니다. 특히 에이전트 간의 결합 정보를 활용하여 MARL 알고리즘을 발전시키는 데 중점을 두고, 가치 의존 집합(value dependency set)을 정의하여 각 에이전트가 필요한 정보를 정확히 추정할 수 있게 합니다. 이 연구는 기존의 CTDE(중앙 집중식 학습 및 분산 실행) 방법보다 분산 학습 방식에서 더 낮은 총 분산을 갖는다는 이론적 결과를 제공합니다.

- **Technical Details**: 논문에서 제안하는 다중 에이전트 베이지안 네트워크(MABN) 프레임워크는 협력적 MARL의 Q-함수를 정확히 분해하는데 필수적인 정보를 제공합니다. 이 프레임워크는 각 에이전트의 Q-함수를 계산하기 위해 필요한 최소한의 에이전트를 식별하여 차원 증가의 저주(curse of dimensionality)를 감소시킵니다. 또한, MABN은 전통적인 접근 방식과 달리 비선형 시스템에서도 적용 가능하도록 설계되어 있습니다.

- **Performance Highlights**: 제안된 알고리즘은 다중 창고 리소스 할당과 다중 존 온도 조절 등의 예시에서 그 효율성과 확장성을 입증하였습니다. 특히, 밀집한 가치 의존 집합에서 베이지안 네트워크를 기반으로 한 근사화 기법을 통해 많은 에이전트가 포함된 애플리케이션에서 기존 방법보다 더 빠른 수렴 속도를 달성하는 것을 보여주었습니다. 이는 각 에이전트가 유사한 환경에서 효과적으로 학습할 수 있음을 시사합니다.



### Learning with Incomplete Context: Linear Contextual Bandits with Pretrained Imputation (https://arxiv.org/abs/2510.09908)
- **What's New**: 이 연구에서는 PULSE-UCB라는 새로운 알고리즘을 제안하여, 대체 데이터(auxiliary data)로 훈련된 프리트레인드(pretrained) 모델을 통해 온라인 결정-making에서 누락된 컨텍스트(context)를 보완하는 방법을 제시합니다. 이는 특히 온라인 상호작용 중에 부분적으로 관찰된 컨텍스트에 대해 결정-making 품질을 향상시키는 데 중요한 문제입니다. 연구진은 이 알고리즘이 일반적인 점진적 결정을 위한 레그렛(regret) 보장을 제공한다는 것을 입증했습니다.

- **Technical Details**: PULSE-UCB 알고리즘은 보조 데이터에서 정확한 컨텍스트를 예측하기 위해 훈련된 프리트레인드 모델을 이용합니다. 사용된 모델의 품질은 레그렛 보정 항으로 반영되며, 이는 선형 컨텍스트 밴디트(linear contextual bandits)에서의 결정-making을 극대화합니다. 연구에서는 시간 지평선(T) 및 관측된 컨텍스트의 차원(d)에 대해 레그렛 상한을 설정하며, 아이디(i.i.d.) 컨텍스트의 경우 프리트레인드 수치를 결합하여 근접 최적 성능을 달성합니다.

- **Performance Highlights**: PULSE-UCB 알고리즘은 실제 사례인 단계적 의사결정에서 기대되는 성능을 보여주며, 기존 데이터 기반의 결정 규칙을 개선하는 데 효과적이라는 결과를 나타냅니다. 연구에서는 예측된 컨텍스트의 불확실성이 의사결정 품질에 미치는 영향을 정량화하였으며, 과거 데이터의 중요성에 대한 통찰을 제공합니다. 결과적으로, 이 알고리즘은 의료 및 교육 분야에서 실질적인 영향을 미칠 가능성이 있음을 보여줍니다.



### Beyond AlphaEarth: Toward Human-Centered Spatial Representation via POI-Guided Contrastive Learning (https://arxiv.org/abs/2510.09894)
- **What's New**: 이번 연구에서는 AlphaEarth Foundation (AE)을 기반으로 한 도시 분석을 목적으로 한 AETHER (AlphaEarth-POI Enriched Representation Learning) 프레임워크를 제안합니다. AETHER는 Points of Interest (POIs)를 통해 인간 중심의 도시 분석에 맞게 AE 임베딩을 조정하며, 물리적 데이터와 사회경제적 맥락을 결합합니다. 이를 통해 AE의 기존 한계를 극복하고 인간의 활동을 반영한 공간 표현을 학습할 수 있습니다.

- **Technical Details**: AETHER는 POI 주변의 공간 버퍼 내에서 AE의 64차원 임베딩을 집계하고, 설계된 다중 스케일 헤드를 통해 POI 텍스트 임베딩과 정합합니다. 이를 위해 InfoNCE 기반의 대조적 목표를 사용하며, 결과적으로 지역 수준의 기능을 집계하여 후속 응용 프로그램에 활용할 수 있습니다. AETHER는 수행성과 효율성을 고려하여 경량화된 멀티모달 정합을 기반으로 합니다.

- **Performance Highlights**: Greater London에서 수행된 실험 결과, AETHER는 AE 및 POI 전용 벤치마크를 초과하여 지속적으로 개선된 성과를 보여주었습니다. AETHER는 땅 이용 분류 메트릭에서 7.2%의 상대적 향상과 사회경제적 매핑에서 23.6%의 KL 다이버전스 감소를 기록하며, 이는 멀티모달 정합이 EO 기반의 표현의 기능적 해석 가능성을 크게 향상시킨다는 것을 의미합니다.



### HIPPD: Brain-Inspired Hierarchical Information Processing for Personality Detection (https://arxiv.org/abs/2510.09893)
- **What's New**: 이 논문에서는 HIPPD라는 뇌 영감을 받은 프레임워크를 제안합니다. 이는 인간의 계층적 정보 처리를 모방하여 개인의 성격 특성을 탐지하는 것을 목표로 합니다. HIPPD는 대규모 언어 모델을 활용하여 전역적 의미 추론과 깊은 특징 추상화를 가능하게 합니다.

- **Technical Details**: HIPPD는 대뇌 피질 모사를 통해 텍스트 데이터 내의 장기 의존성을 포착하고, 전두엽에 기반한 동적 기억 모듈을 통해 중요한 특징을 선택적으로 유지 및 업데이트합니다. 마지막으로, 기저핵(선조체) 기능을 모방한 전문 모델 라우팅 레이어가 적용되어, 엄격한 승자-독식 메커니즘을 통해 입력 데이터를 최적의 전문 모델로 동적으로 라우팅합니다.

- **Performance Highlights**: Kaggle 및 Pandora 데이터셋에서의 광범위한 실험 결과, HIPPD는 최신의 다른 방법들과 비교하여 지속적으로 우수한 성능을 보여주었습니다. 이와 같은 성능은 클래스 불균형 및 짧은 텍스트 문제를 해결하는 동시에, 다양한 피쳐가 부족한 작업에서도 잘 일반화됩니다.



### DELTA: Dynamic Layer-Aware Token Attention for Efficient Long-Context Reasoning (https://arxiv.org/abs/2510.09883)
- **What's New**: DELTA는 대규모 추론 모델(Large Reasoning Models, LRM)의 효율성을 높이는 훈련이 필요 없는 희소 주의 메커니즘입니다. 이 방법은 모델의 정확도 손실 없이 계산 효율성을 달성합니다. DELTA는 변환기 층을 세 그룹으로 나누어, 초기 층에서 전 주의를 사용하고, 중요한 토큰을 선택하는 selection layers와 선택된 하위 집합에 대해서만 주의를 기울이는 sparse-attention layers로 구성되어 있습니다.

- **Technical Details**: DELTA는 각 디코딩 스텝에서 신중하게 선택된 토큰의 집합에 대한 계산만을 수행하고, 전체 주의 맵을 사용하는 소수의 중간 층을 활용하여 다음 층을 위한 중요한 토큰을 예측합니다. 이는 최근 맥락을 보장하면서 높은 주의 토큰을 정확하게 식별할 수 있도록 합니다. 이러한 과정은 주의 패턴의 강한 상관관계와 토큰 중요도가 변하는 점을 고려하여 이루어집니다.

- **Performance Highlights**: DELTA는 AIME 및 GPQA-Diamond와 같은 추론 기준에서 정확도를 유지하면서 최대 5배의 주의 토큰 수를 줄이고, 최대 1.5배의 종단 간 속도 향상을 달성합니다. DELTA는 기존의 희소 주의 방법들보다 월등한 성능을 보이며, 추론 과제의 정확성을 저하시키지 않고 속도를 높이는 데 기여할 수 있습니다.



### Exploration of Incremental Synthetic Non-Morphed Images for Single Morphing Attack Detection (https://arxiv.org/abs/2510.09836)
Comments:
          Workshop paper accepted NeurIPS 2025

- **What's New**: 본 논문은 합성 얼굴 데이터(synthetic face data)를 활용하여 단일 변형 공격 탐지(S-MAD) 기술을 향상시키는 방법을 연구하였습니다. 또한 프라이버시 문제로 인해 실제 이미지를 대규모로 확보하기 어려운 한계를 극복하는 방법을 제시합니다. 다양한 변형 도구와 교차 데이터셋 평가 체계를 사용하여 연구가 진행되었습니다.

- **Technical Details**: 연구의 핵심은 '비변형(non-morphed)' 이미지라는 합성 이미지를 기존의 데이터셋에 통합하여 학습 과정에서 정교하게 조절된 양의 합성 이미지를 추가하는 것입니다. 이를 통해 S-MAD의 성과를 향상시키고자 하였습니다. 실험 결과, 합성 데이터의 무분별한 사용은 최적의 성능을 보장하지 않음을 강조합니다.

- **Performance Highlights**: 단지 합성 데이터만 활용한 경우에 가장 높은 동등 오류율(Equal Error Rate, EER)을 달성했으며, 이는 S-MAD의 운영 시나리오에서 합성 데이터에만 의존하는 것이 최선의 선택이 아니라는 것을 의미합니다. 정교하게 구성된 합성 이미지와 실제 이미지를 점진적으로 통합할 경우 일반화 성능이 개선되는 것을 보여주었습니다.



### Distributed clustering in partially overlapping feature spaces (https://arxiv.org/abs/2510.09799)
- **What's New**: 이번 논문에서는 각 참여자가 고유한 특징의 부분 집합을 포함하는 개인 데이터셋을 보유한 상태에서의 새로운 분산 클러스터링 문제를 소개하고 해결합니다. 이러한 시나리오는 의료 분야와 같이 비슷한 환자에 대한 보완적인 데이터가 여러 기관에 분산되어 있는 실제 응용 프로그램에서 발생합니다. 제안된 두 가지 알고리즘은 다양한 특성의 불균형을 나타내는 분산 클러스터링 문제를 해결하기 위해 고안되었습니다.

- **Technical Details**: 첫 번째 알고리즘은 전역 중심좌표의 집합을 공동으로 업데이트하는 연합(federated) 알고리즘이며, 두 번째는 각 참여자가 자신들의 데이터셋에 대해 요약하여 중앙 서버와 공유하는 일회성(one-shot) 알고리즘입니다. 각 참여자는 자신이 선택한 알고리즘을 사용하여 지역 클러스터링을 수행하고, 이는 유연성과 개인화된 계산 비용을 제공합니다. 또한, 제안된 알고리즘이 최적의 중앙 집중식 솔루션으로 수렴하도록 기대되는 몇 가지 조건을 식별하였습니다.

- **Performance Highlights**: 세 가지 공개 데이터셋에서 알고리즘의 실용성을 실험하여 성능을 검증하였고, 데이터 분포 및 특성 공간의 겹침에 따라 다양한 실험을 통해 그 효과를 보여주었습니다. 실험 결과, 제안된 알고리즘은 분산 환경에서도 효과적인 클러스터링 솔루션을 제공함을 입증했습니다. 이러한 접근 방식은 의료 데이터 분석과 같은 여러 응용 분야에서 활용 가능성을 제시하고 있습니다.



### The Geometry of Reasoning: Flowing Logics in Representation Spac (https://arxiv.org/abs/2510.09782)
Comments:
          Code: this https URL

- **What's New**: 이 논문에서는 대규모 언어 모델(LLMs)이 어떻게 '사고'하는지를 탐구합니다. 새로운 기하학적 프레임워크를 제안하여 LLM의 추론을 흐름(flow)으로 모델링하며, 이는 논리가 진행되는 임베딩 경로(embedding trajectories)를 나타냅니다. 이 연구는 LLM이 표면적인 형식을 넘어 논리를 내부화하는지를 검증할 수 있는 새로운 관점을 제공합니다.

- **Technical Details**: 연구자들은 자연적 추론(propositional natural deduction)을 사용하여 의미론적 도구(semantic carriers)가 다양해도 논리 구조와 의미를 분리하여 LLM의 사고 과정을 분석합니다. 이 기하학적 관점은 위치(position), 속도(velocity), 곡률(curvature)과 같은 기하학적 양들과 추론을 연결하여, 추상화된 개념 공간(representation and concept spaces)에서의 분석을 가능하게 합니다.

- **Performance Highlights**: 이론적 프레임워크를 구현하기 위해 학습된 임베딩 대리 모델(learned representation proxies)을 사용하여 추론 흐름을 시각화하고 정량화하는 통제된 실험을 설계하였습니다. 이 연구는 LLM의 행동의 해석 가능성과 공식적인 분석을 위한 새로운 관점을 제공하며, 추론 현상을 연구하기 위한 개념적 기초와 실제 도구 역할을 합니다.



### Constructive Distortion: Improving MLLMs with Attention-Guided Image Warping (https://arxiv.org/abs/2510.09741)
- **What's New**: 최근 연구에서는 멀티모달 대형 언어 모델(MLLMs)이 복잡한 장면에서 작은 세부사항이나 공간적 관계를 놓치는 문제를 해결하기 위한 방법을 제안합니다. 새로운 방법인 AttWarp는 쿼리와 관련된 콘텐츠에 더 많은 해상도를 할당하고, 덜 중요한 영역을 압축하여 글로벌 컨텍스트를 유지하는 경량화된 방식을 도입했습니다. 이 방법은 모델의 가중치나 아키텍처를 변경하지 않고도 비주얼 정보를 비균일하게 재분배하여 원본 이미지의 모든 정보는 유지하면서도 세부 사항을 더욱 쉽게 인식할 수 있도록 도와줍니다.

- **Technical Details**: AttWarp는 MLLM의 교차 모달 어텐션(cross-modal attention)을 활용하여 입력 이미지를 직선적으로 왜곡(rectilinear warping) 처리하는 방법입니다. 입력 이미지와 쿼리를 기준으로 교차 모달 어텐션 맵을 추출한 후, 이를 통해 주의 점수 매트릭스(Attention Score Matrix)를 생성하고 이를 기반으로 높이와 너비에 따라 각각의 중요도를 수치화합니다. 이 프로파일들을 바탕으로 상대적으로 더 높은 중요도를 가진 영역을 확장하고 낮은 중요도를 가진 영역은 압축하는 왜곡 과정을 통해 시각 정보를 처리합니다.

- **Performance Highlights**: AttWarp는 TextVQA, GQA, DocVQA, POPE, MMMU를 포함한 다섯 가지 벤치마크에서 반복적으로 정확도를 향상시키며, 조합적 추론(compositional reasoning)을 강화하고 환각(hallucination)을 줄이는 데 성공했습니다. 네 가지 경쟁적인 기준선(baselines)보다 뛰어난 성능을 보여주었으며, 다양한 MLLM 백본(backbone)과 어텐션 소스에서의 일반화 가능성도 확인되었습니다. 또한, AttWarp-Chain과 AttWarp-Distill 등의 방법론을 통해 성능을 더욱 향상시킬 수 있는 가능성을 제시하였습니다.



### All Code, No Thought: Current Language Models Struggle to Reason in Ciphered Languag (https://arxiv.org/abs/2510.09714)
- **What's New**: 이번 연구는 AI 시스템의 체계적 예방 위험을 평가하기 위해, 28가지 서로 다른 암호화 방식에서 AI 모델이 수행할 수 있는 복잡한 이유가 숨겨진 메커니즘인 'ciphered reasoning'에 대한 첫 번째 상세 연구를 제시합니다. 연구자들은 10개의 모델을 조정하여 이러한 암호화된 텍스트에서의 추론 능력을 평가하였으며, 모델들은 암호화된 텍스트를 잘 이해하면서도 정확한 이유를 제공하는 데 어려움을 겪는 비대칭성을 발견했습니다.

- **Technical Details**: 이 논문에서는 'ciphered reasoning' 능력을 두 가지 측면에서 평가합니다: 첫째, 암호화된 텍스트에서 문제 해결 성능이 향상되는지 판단하는 'ciphered reasoning capability', 둘째, 암호화된 텍스트를 영어로 해독하는 능력을 포함하는 'cipher translation capability'입니다. 연구 결과, 암호화된 텍스트에서 제대로 이유를 제공하기 위해서는 방대한 양의 학습 데이터가 필요하며, 잘 알려진 암호에서는 높은 정확도를 보이는 반면, 덜 알려진 암호에서는 큰 성능 저하가 발생하는 것을 확인했습니다.

- **Performance Highlights**: 모델의 정확도는 암호화된 텍스트에서의 추론 능력과 상관관계가 있었으며, 훈련 데이터의 암호화된 텍스트 발생 빈도가 높을수록 정확도가 증가하는 경향을 보였습니다. 추가적인 미세 조정 데이터가 이루어져도 암호화된 추론 능력의 향상은 더디며, 단순한 암호 체계에서 3.7B 토큰 이상의 데이터가 필요하다는 결과가 도출되었습니다. 이러한 발견은 현재의 모델이 암호화된 텍스트로 CoT 모니터링을 피하는 것이 비효율적이라는 점을 제시합니다.



### Rounding-Guided Backdoor Injection in Deep Learning Model Quantization (https://arxiv.org/abs/2510.09647)
Comments:
          This paper is to appear in NDSS 2026

- **What's New**: 이번 연구에서는 모델 양자화 과정에서 악성 행동을 내장하는 새로운 백도어 공격 방법인 QuRA를 소개합니다. QuRA는 기존의 백도어 공격 방법들과는 달리, 훈련 단계에서의 데이터 오염이나 조작 없이 양자화 작업만으로 작동합니다. 이를 통해 훈련 파이프라인에 대한 접근 없이도 공격자가 사전 훈련된 모델을 목표로 할 수 있음을 보여줍니다.

- **Technical Details**: QuRA는 양자화 과정에서의 라운딩 조작을 통해 모델에 백도어를 주입하는 방법을 제안합니다. 특히, 라운딩 방향을 최적화하여 백도어 효과를 강화하고, 전체 성능 저하 없이 레이어 간의 영향을 증폭시킵니다. 본 연구는 대응 기법을 우회할 수 있는 능력을 지닌 QuRA가 모델 배포 과정에서 발생할 수 있는 중요한 취약점임을 강조합니다.

- **Performance Highlights**: 실험 결과 QuRA는 VGG-16 모델에서 100%의 공격 성공률을 달성했으며, 정확도는 단 0.8% 떨어지는 것으로 나타났습니다. 자율적으로 변화하는 공격에서도 기존 방어 기제를 효과적으로 우회함으로써, 모델 배포 단계에서의 큰 위협으로 자리잡을 수 있습니다. 이러한 결과는 양자화 과정의 안전성에 대한 재조명이 필요하다는 점을 시사합니다.



### AdaptAuth: Multi-Layered Behavioral and Credential Analysis for a Secure and Adaptive Authentication Framework for Password Security (https://arxiv.org/abs/2510.09645)
- **What's New**: 이번 논문에서는 기존의 비밀번호 보안을 혁신하기 위한 다각적인 솔루션을 제안합니다. 이 솔루션은 Password Dissection Mechanism, Dynamic Password Policy Mechanism, 인간 행동 패턴, 장치 특성, 네트워크 파라미터 등 다양한 요소를 통합하여 비밀번호 관련 위협을 효과적으로 해결하고자 합니다. 학습 기반 모델을 통해 상세한 사용자 프로파일을 구축하여 거의 모든 형태의 무단 접근을 방지할 수 있도록 설계된 것이 특징입니다.

- **Technical Details**: 비밀번호 보안의 복잡성을 줄이며 사용자의 사용성을 강화하는 방향으로, 개인의 행동 패턴과 장치의 특성을 포함한 다양한 요소를 고려합니다. 제안된 프레임워크는 비밀번호 생성 정책을 동적으로 변화시키며, 이를 통해 사용자 별로 더욱 강력한 보안 프로필을 제공합니다. K-strikes 메커니즘과 같은 기존 방법들의 한계를 극복하고, 초과 사용자의 종합적인 비밀번호 행동 분석을 통해 보안을 강화합니다.

- **Performance Highlights**: 이 연구의 결과로 제안된 프레임워크는 기존의 비밀번호 보안 표준보다 강력한 보호를 제공하면서도 사용자가 정책 설정 과정에 더 참여하도록 유도하는 새로운 접근 방식을 제공합니다. 특히, 장치와 인간 행태를 분석하여 범죄자를 식별하고 도난 장치를 추적할 수 있는 기능이 강조됩니다. 이로써 사용자는 자신의 디지털 자산을 더 쉽게 관리하고 보호할 수 있는 능력을 갖출 수 있습니다.



### Hound: Relation-First Knowledge Graphs for Complex-System Reasoning in Security Audits (https://arxiv.org/abs/2510.09633)
- **What's New**: 이 논문은 Hound라는 관계 중심의 그래프 엔진을 소개하며, 복잡한 코드베이스에서 시스템 수준의 추론을 개선합니다. 에이전트는 유연하고 분석자가 정의한 뷰(view)를 설계하며, 이를 통해 중요 코드만을 로드하여 시스템 구조를 확대하거나 결정적인 코드로 좁혀 접근할 수 있습니다. 또한, 지속적인 신념 시스템을 도입하여 장기적으로 존재하는 취약점 가설의 신뢰도를 업데이트하는 방식을 채택합니다.

- **Technical Details**: Hound는 다중 규모(multi-scale)의 이해를 가능하게 하는 그래프 기반의 에이전트 아키텍처를 사용합니다. 이 시스템은 인증/권한 역할, 금전적 흐름(monetary flows), 호출 그래프(call graphs), 불변성(invariants) 등 담당자가 실제로 고려하는 측면을 모델링하고, 그래프의 증거를 기반으로 한 정확한 검색 기능을 제공합니다. 각 감사는 시스템 아키텍처(System Architecture) 그래프에서 시작하여 특정 '측면(aspect)' 그래프로 확장됩니다.

- **Performance Highlights**: Hound는 ScaBench의 5개 프로젝트에서 LLM 분석기와 비교하여 recall 31.2% vs. 8.3%, F1 스코어 14.2% vs. 9.8%로 개선된 성능을 보여줍니다. 이 성과는 관계 중심 그래프가 코드 호출 및 데이터 흐름을 넘어 모델 이해를 확장하기 때문입니다. Hound는 단순한 유사 검색과는 달리 작업별 그래프를 이용해 필요한 코드를 정확하게 로드하여, 불필요한 맥락으로 인한 혼란을 줄이고 증거 기반의 설명을 제공합니다.



### Performance of Machine Learning Methods for Gravity Inversion: Successes and Challenges (https://arxiv.org/abs/2510.09632)
- **What's New**: 이 논문에서는 중력 반전(gravity inversion) 문제를 해결하기 위해 심층 학습(deep learning) 기술을 활용하는 새로운 접근법을 제안합니다. 특히 2차원(2D) 경우를 고려하여, 관측된 중력 필드 데이터를 기반으로 밀도 분포를 추정하는 과정에서 발생하는 불확실성을 극복하고자 합니다. 논문에서는 CNN, 변분 오토인코더(Variational Autoencoders, VAE), 생성적 적대 신경망(Generative Adversarial Networks, GAN)을 포함한 여러 모델을 이용하여 밀도를 정확히 예측하는 방법을 탐구합니다.

- **Technical Details**: 중력 반전 문제는 측정된 중력 규모에서 밀도 모델을 역산출하는 일로, 해석 문제의 한 종류입니다. 이 과정에서 과거의 수학적 방법은 불확실성을 초래했지만, 새로운 머신 러닝 방법이 이를 개선할 수 있는 잠재력을 보여주고 있습니다. 저자들은 CNN을 통해 밀도 필드에 직접적으로 중력 이상을 매핑하고, 다양한 정규화 기법(handler)을 통해 모델의 안정성과 정확성을 극대화하고자 했습니다.

- **Performance Highlights**: 연구 결과, CNN 기반의 반전 방법이 기존의 전통적 방법들보다 현저히 우수한 재구성을 제공함을 보여줍니다. 생성적 모델들은 여전히 가능성이 있지만 불안정한 결과를 보이며, 전통적인 반복계산 방법들은 미미한 개선 효과를 보여주어 중력 반전 문제의 고질적인 불확정성을 부각시키고 있습니다.



### Risk-Calibrated Bayesian Streaming Intrusion Detection with SRE-Aligned Decisions (https://arxiv.org/abs/2510.09619)
Comments:
          11 pages, 7 figures. Primary category: cs.CR; cross-list: cs.LG, stat.ML. Implementation code and datasets are available from the corresponding author upon reasonable request. Code and reproducibility materials will be made available upon publication

- **What's New**: 본 논문에서는 Bayesian Online Changepoint Detection (BOCPD)와 Site Reliability Engineering (SRE)의 오류 예산에 맞춘 결정 임계값을 결합한 스트리밍 침입 탐지를 위한 리스크 수정 접근 방식을 제시합니다. 이 방법은 데이터 분포의 변화에 적응하는 런 렝스 후행 확률을 제공하며, 이를 통해 false-positive와 false-negative 예산을 최적화하여 경고 결정을 내립니다. 99.9% 가용성을 목표로 하는 구체적인 SRE 사례를 통해 이 접근 방식의 효과를 설명하고 있습니다.

- **Technical Details**: 우리는 네트워크 흐름에서 추출된 특성 벡터를 벤리그(benign)와 악성(malicious) 생성 프로세스의 혼합으로 모델링합니다. 각 시간 단계에서 런 렝스 변수 r를 사용하여 마지막 변동점(changepoint) 이후 관측 수를 나타냅니다. Bayesian 의사 결정 이론에 따라서 최적의 임계값을 설정하고 이를 통해 분석가의 시간과 오류 예산을 고려하여 경고를 발생시킵니다.

- **Performance Highlights**: UNSW-NB15와 CICIDS2017 데이터셋을 사용한 실험 결과, 리스크 수정 BOCPD가 중간에서 높은 재현율에서 더욱 개선된 precision-recall을 보여주었습니다. 이 방법은 잘 보정된 확률을 제공하며, 기존 비지도 배경 모델들에 비해 뛰어난 성능을 보였습니다. 향후 연구는 이 프레임워크를 실제 기업 텔레메트리에 적용하고 생성 모델을 확장하는 것을 목표로 하고 있습니다.



